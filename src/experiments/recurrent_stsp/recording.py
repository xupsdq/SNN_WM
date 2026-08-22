"""Sparse spike/STSP recording and explicit task evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import torch

from .config import TiddiaNetworkConfig
from .scheduler import SparseEventScheduler


PathLike = Union[str, os.PathLike]


def atomic_torch_save(payload: object, path: PathLike) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(destination.parent), delete=False, suffix=".tmp"
        ) as handle:
            temporary_name = handle.name
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def atomic_json_dump(payload: object, path: PathLike) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            delete=False,
            suffix=".tmp",
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


@dataclass(frozen=True)
class SpikeRecordingConfig:
    populations: Tuple[int, ...] = (0, 1, 2, 3, 4)
    fraction_per_population: float = 1.0
    start_ms: float = 100.0
    chunk_steps: int = 2_048

    def __post_init__(self) -> None:
        if not self.populations:
            raise ValueError("At least one spike population must be recorded.")
        if not 0.0 < self.fraction_per_population <= 1.0:
            raise ValueError("fraction_per_population must lie in (0, 1].")
        if self.start_ms < 0.0 or self.chunk_steps <= 0:
            raise ValueError("Spike recording start/chunk size is invalid.")


@dataclass(frozen=True)
class StspProbeRecordingConfig:
    populations: Tuple[int, ...] = (0, 1, 2, 3, 4)
    source_fraction_per_population: float = 0.10
    max_edges_per_population: int = 2_048
    start_ms: float = 100.0
    snapshot_interval_ms: float = 5.0

    def __post_init__(self) -> None:
        if not self.populations:
            raise ValueError("At least one STSP source population is required.")
        if not 0.0 < self.source_fraction_per_population <= 1.0:
            raise ValueError("source_fraction_per_population must lie in (0, 1].")
        if self.max_edges_per_population <= 0:
            raise ValueError("max_edges_per_population must be positive.")
        if self.start_ms < 0.0 or self.snapshot_interval_ms <= 0.0:
            raise ValueError("STSP recording timing is invalid.")


@dataclass(frozen=True)
class TaskEvaluationConfig:
    target_population: int = 0
    window_start_ms: float = 3_350.0
    window_stop_ms: float = 5_200.0
    minimum_target_rate_hz: float = 3.0
    minimum_margin_hz: float = 1.0

    def __post_init__(self) -> None:
        if self.target_population < 0:
            raise ValueError("target_population must be non-negative.")
        if not 0.0 <= self.window_start_ms < self.window_stop_ms:
            raise ValueError("Evaluation window must be non-empty and non-negative.")
        if self.minimum_target_rate_hz < 0.0 or self.minimum_margin_hz < 0.0:
            raise ValueError("Rate thresholds must be non-negative.")


class SparseSpikeRecorder:
    """Record selected neurons in device-side chunks, then store sparse events."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        network: TiddiaNetworkConfig,
        config: SpikeRecordingConfig,
        *,
        device: Union[str, torch.device],
    ) -> None:
        self.network = network
        self.config = config
        self.device = torch.device(device)
        selected: List[torch.Tensor] = []
        population_ids: List[torch.Tensor] = []
        population_size = network.selective_population_size
        count = max(1, int(population_size * config.fraction_per_population))
        for population in config.populations:
            if population < 0 or population >= network.n_memories:
                raise ValueError("Recorded population lies outside n_memories.")
            start = population * population_size
            selected.append(torch.arange(start, start + count, dtype=torch.int64))
            population_ids.append(
                torch.full((count,), population, dtype=torch.int16)
            )
        self.neuron_ids_cpu = torch.cat(selected)
        self.population_ids_cpu = torch.cat(population_ids)
        self.neuron_ids = self.neuron_ids_cpu.to(self.device)
        self._buffer = torch.empty(
            (config.chunk_steps, self.neuron_ids.numel()),
            dtype=torch.bool,
            device=self.device,
        )
        self._buffer_rows = 0
        self._buffer_start_step = 0
        self._time_chunks: List[torch.Tensor] = []
        self._sender_chunks: List[torch.Tensor] = []
        self._population_chunks: List[torch.Tensor] = []

    def record(self, spikes: torch.Tensor, event_step: int) -> None:
        time_ms = event_step * self.network.dt_ms
        if time_ms <= self.config.start_ms:
            return
        if spikes.shape != (self.network.n_neurons,):
            raise ValueError("spikes must have one element per neuron.")
        if self._buffer_rows == 0:
            self._buffer_start_step = event_step
        expected = self._buffer_start_step + self._buffer_rows
        if event_step != expected:
            raise ValueError("Spike recorder requires contiguous simulation steps.")
        self._buffer[self._buffer_rows].copy_(spikes[self.neuron_ids])
        self._buffer_rows += 1
        if self._buffer_rows == self.config.chunk_steps:
            self.flush()

    def flush(self) -> None:
        if self._buffer_rows == 0:
            return
        dense = self._buffer[: self._buffer_rows].to("cpu")
        positions = torch.nonzero(dense, as_tuple=False)
        if positions.numel() > 0:
            rows = positions[:, 0]
            columns = positions[:, 1]
            event_steps = rows + self._buffer_start_step
            self._time_chunks.append(
                event_steps.to(torch.float64).mul_(self.network.dt_ms)
            )
            self._sender_chunks.append(self.neuron_ids_cpu[columns].to(torch.int32))
            self._population_chunks.append(
                self.population_ids_cpu[columns].to(torch.int16)
            )
        self._buffer_rows = 0

    def payload(self) -> Dict[str, object]:
        self.flush()
        empty_time = torch.empty(0, dtype=torch.float64)
        empty_sender = torch.empty(0, dtype=torch.int32)
        empty_population = torch.empty(0, dtype=torch.int16)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "dt_ms": self.network.dt_ms,
            "recording_config": asdict(self.config),
            "recorded_neuron_ids": self.neuron_ids_cpu.to(torch.int32),
            "recorded_neuron_population_ids": self.population_ids_cpu,
            "times_ms": (
                torch.cat(self._time_chunks) if self._time_chunks else empty_time
            ),
            "sender_ids": (
                torch.cat(self._sender_chunks) if self._sender_chunks else empty_sender
            ),
            "population_ids": (
                torch.cat(self._population_chunks)
                if self._population_chunks
                else empty_population
            ),
        }

    def save(self, path: PathLike) -> Path:
        return atomic_torch_save(self.payload(), path)


def _sources_for_edge_ids(row_ptr: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
    return torch.searchsorted(row_ptr[1:], edge_ids, right=True)


class ContinuousStspProbeRecorder:
    """Snapshot analytically recovered continuous u/x trajectories.

    The scheduler stores u/x at the last presynaptic event, exactly as the
    event-driven synapse does.  At each snapshot this recorder analytically
    evolves both variables to the requested time without mutating simulation
    state.  Thus the artifact contains continuous STSP state, not stale cache
    values between spikes.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        scheduler: SparseEventScheduler,
        config: StspProbeRecordingConfig,
    ) -> None:
        self.scheduler = scheduler
        self.network = scheduler.config
        self.config = config
        interval_scaled = config.snapshot_interval_ms / self.network.dt_ms
        self.interval_steps = int(math.floor(interval_scaled + 0.5))
        if self.interval_steps < 1 or not math.isclose(
            interval_scaled, self.interval_steps, rel_tol=0.0, abs_tol=1e-7
        ):
            raise ValueError("STSP snapshot interval must lie on the time grid.")
        self.edge_ids, self.edge_population_ids = self._select_edges()
        edges = scheduler.connectivity.plastic
        self.sources = _sources_for_edge_ids(edges.row_ptr, self.edge_ids)
        self.targets = edges.targets[self.edge_ids].to(torch.int32)
        self.weights = edges.weights[self.edge_ids]
        self.tau_rec = edges.tau_rec_ms[self.edge_ids]
        self.tau_fac = edges.tau_fac_ms[self.edge_ids]
        self._times: List[float] = []
        self._u: List[torch.Tensor] = []
        self._x: List[torch.Tensor] = []
        self._next_release: List[torch.Tensor] = []

    def _select_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        row_ptr = self.scheduler.connectivity.plastic.row_ptr
        device = row_ptr.device
        population_size = self.network.selective_population_size
        all_edges: List[torch.Tensor] = []
        all_populations: List[torch.Tensor] = []
        source_count = max(
            1, int(population_size * self.config.source_fraction_per_population)
        )
        for population in self.config.populations:
            if population < 0 or population >= self.network.n_memories:
                raise ValueError("STSP population lies outside n_memories.")
            source_start = population * population_size
            first = int(row_ptr[source_start].item())
            last = int(row_ptr[source_start + source_count].item())
            available = last - first
            count = min(self.config.max_edges_per_population, available)
            if count == 0:
                continue
            if count == available:
                chosen = torch.arange(first, last, dtype=torch.int64, device=device)
            else:
                offsets = torch.div(
                    torch.arange(count, dtype=torch.int64, device=device) * available,
                    count,
                    rounding_mode="floor",
                )
                chosen = offsets + first
            all_edges.append(chosen)
            all_populations.append(
                torch.full((count,), population, dtype=torch.int16, device=device)
            )
        if not all_edges:
            raise ValueError("No plastic edges are available for STSP probes.")
        return torch.cat(all_edges), torch.cat(all_populations)

    @torch.no_grad()
    def record(self, event_step: int) -> None:
        time_ms = event_step * self.network.dt_ms
        if time_ms <= self.config.start_ms:
            return
        start_step = int(math.floor(self.config.start_ms / self.network.dt_ms + 0.5))
        if (event_step - start_step) % self.interval_steps != 0:
            return
        state = self.scheduler.plastic_state
        elapsed = time_ms - state.last_spike_time_ms[self.edge_ids]
        u = self.network.stsp_u + (
            state.u[self.edge_ids] - self.network.stsp_u
        ) * torch.exp(-elapsed / self.tau_fac)
        x = 1.0 + (state.x[self.edge_ids] - 1.0) * torch.exp(
            -elapsed / self.tau_rec
        )
        next_u = u + self.network.stsp_u * (1.0 - u)
        next_release = self.weights * next_u * x
        self._times.append(time_ms)
        self._u.append(u.to("cpu"))
        self._x.append(x.to("cpu"))
        self._next_release.append(next_release.to("cpu"))

    def payload(self) -> Dict[str, object]:
        n_edges = self.edge_ids.numel()
        shape = (0, n_edges)
        u = torch.stack(self._u) if self._u else torch.empty(shape, dtype=torch.float32)
        x = torch.stack(self._x) if self._x else torch.empty(shape, dtype=torch.float32)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "dt_ms": self.network.dt_ms,
            "recording_config": asdict(self.config),
            "times_ms": torch.as_tensor(self._times, dtype=torch.float64),
            "edge_ids": self.edge_ids.to("cpu", dtype=torch.int64),
            "source_ids": self.sources.to("cpu", dtype=torch.int32),
            "target_ids": self.targets.to("cpu", dtype=torch.int32),
            "source_population_ids": self.edge_population_ids.to(
                "cpu", dtype=torch.int16
            ),
            "weight_pa": self.weights.to("cpu"),
            "tau_rec_ms": self.tau_rec.to("cpu"),
            "tau_fac_ms": self.tau_fac.to("cpu"),
            "u": u,
            "x": x,
            "ux": u * x,
            "next_spike_release_pa": (
                torch.stack(self._next_release)
                if self._next_release
                else torch.empty(shape, dtype=torch.float32)
            ),
        }

    def save(self, path: PathLike) -> Path:
        return atomic_torch_save(self.payload(), path)


def evaluate_task(
    spike_payload: Dict[str, object],
    config: TaskEvaluationConfig,
) -> Dict[str, object]:
    """Decode selective-population identity from sparse spikes."""

    times = torch.as_tensor(spike_payload["times_ms"], dtype=torch.float64)
    populations = torch.as_tensor(spike_payload["population_ids"], dtype=torch.int64)
    recorded_populations = torch.as_tensor(
        spike_payload["recorded_neuron_population_ids"], dtype=torch.int64
    )
    unique_populations = sorted(int(item) for item in recorded_populations.unique())
    if config.target_population not in unique_populations:
        raise ValueError("The target population was not recorded.")
    duration_s = (config.window_stop_ms - config.window_start_ms) / 1_000.0
    in_window = (times > config.window_start_ms) & (times <= config.window_stop_ms)
    rates: Dict[str, float] = {}
    for population in unique_populations:
        neurons = int((recorded_populations == population).sum().item())
        events = int((in_window & (populations == population)).sum().item())
        rates[str(population)] = events / (neurons * duration_s)
    ordered = sorted(
        ((rate, int(population)) for population, rate in rates.items()),
        key=lambda item: (-item[0], item[1]),
    )
    winner_rate, winner_population = ordered[0]
    competing_rates = [
        rate for population, rate in rates.items()
        if int(population) != config.target_population
    ]
    best_competitor = max(competing_rates) if competing_rates else 0.0
    target_rate = rates[str(config.target_population)]
    margin = target_rate - best_competitor
    success = (
        winner_population == config.target_population
        and target_rate >= config.minimum_target_rate_hz
        and margin >= config.minimum_margin_hz
    )
    return {
        "evaluation_config": asdict(config),
        "population_rates_hz": rates,
        "winner_population": winner_population,
        "winner_rate_hz": winner_rate,
        "target_rate_hz": target_rate,
        "best_competitor_rate_hz": best_competitor,
        "target_margin_hz": margin,
        "success": bool(success),
        "criterion": (
            "target is the deterministic rate winner; target rate and target-minus-"
            "best-competitor margin meet the configured thresholds"
        ),
    }


__all__ = [
    "ContinuousStspProbeRecorder",
    "SparseSpikeRecorder",
    "SpikeRecordingConfig",
    "StspProbeRecordingConfig",
    "TaskEvaluationConfig",
    "atomic_json_dump",
    "atomic_torch_save",
    "evaluate_task",
]
