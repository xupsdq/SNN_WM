"""Matched-query causal experiment for recurrent STSP post-query endpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import platform
import time
from typing import Dict, Optional, Tuple, Union

import torch

from .checkpoint import (
    NetworkCheckpoint,
    capture_network_checkpoint,
    continuous_plastic_state,
    replace_plastic_state,
    restore_network_checkpoint,
)
from .connectivity import SparseRecurrentConnectivity
from .protocol import ExternalInputEngine, ItemLoadingSignal, WorkingMemoryProtocolConfig
from .recording import atomic_json_dump, atomic_torch_save
from .runner import SimulationRunConfig
from .scheduler import SparseRecurrentNetwork


PathLike = Union[str, os.PathLike]


@dataclass(frozen=True)
class MatchedQueryExperimentConfig:
    """Two different histories followed by one physically identical query."""

    history_populations: Tuple[int, int] = (0, 1)
    query_population: int = 2
    history_origin_ms: float = 50.0
    query_origin_ms: float = 250.0
    cue_duration_ms: float = 75.0
    total_time_ms: float = 400.0
    delay_readout_start_ms: float = 200.0
    response_window_start_ms: float = 250.0
    response_window_stop_ms: float = 400.0
    stsp_sample_edges: int = 4_096
    poisson_input: bool = True
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if len(set(self.history_populations)) != 2:
            raise ValueError("The two history populations must differ.")
        if self.query_population in self.history_populations:
            raise ValueError("The common query must differ from both histories.")
        if self.history_origin_ms < 0.0 or self.cue_duration_ms <= 0.0:
            raise ValueError("History timing is invalid.")
        if self.history_origin_ms + self.cue_duration_ms >= self.query_origin_ms:
            raise ValueError("History cue must end before the common query.")
        if not 0.0 <= self.delay_readout_start_ms < self.query_origin_ms:
            raise ValueError("Delay readout must end at the query boundary.")
        if not (
            self.query_origin_ms <= self.response_window_start_ms
            < self.response_window_stop_ms
            <= self.total_time_ms
        ):
            raise ValueError("Response window must lie after the query boundary.")
        if self.query_origin_ms + self.cue_duration_ms > self.total_time_ms:
            raise ValueError("Query cue must end within the experiment.")
        if self.stsp_sample_edges <= 0:
            raise ValueError("stsp_sample_edges must be positive.")


@dataclass
class _BoundaryState:
    history_population: int
    protocol: WorkingMemoryProtocolConfig
    network: NetworkCheckpoint
    external: Dict[str, object]
    pre_release: torch.Tensor
    delay_population_rates_hz: torch.Tensor


@dataclass
class _BranchResult:
    name: str
    recipient_history: int
    donor_history: int
    population_rates_hz: torch.Tensor
    population_spike_counts: torch.Tensor
    post_release: torch.Tensor
    total_spikes: int
    plastic_edge_events: int
    static_edge_events: int


def _grid_step(time_ms: float, dt_ms: float) -> int:
    scaled = time_ms / dt_ms
    rounded = int(math.floor(scaled + 0.5))
    if not math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("Experiment times must lie on the simulation grid.")
    return rounded


def _protocol_for_history(
    network_config,
    experiment: MatchedQueryExperimentConfig,
    history_population: int,
) -> WorkingMemoryProtocolConfig:
    resolved_seed = network_config.seed if experiment.seed is None else experiment.seed
    return WorkingMemoryProtocolConfig(
        total_time_ms=experiment.total_time_ms,
        background_stop_ms=experiment.total_time_ms,
        poisson_input=experiment.poisson_input,
        item_loading=(
            ItemLoadingSignal(history_population, experiment.history_origin_ms),
            ItemLoadingSignal(experiment.query_population, experiment.query_origin_ms),
        ),
        cue_duration_ms=experiment.cue_duration_ms,
        eta_end_origin_ms=experiment.total_time_ms + network_config.dt_ms,
        seed=resolved_seed,
    )


def _advance_one_step(
    network: SparseRecurrentNetwork, external: ExternalInputEngine, event_step: int
):
    external_ex, external_in, external_current = external.pop_current()
    result = network.step(
        external_spikes_ex_pa=external_ex,
        external_spikes_in_pa=external_in,
        external_current_0_pa=external_current,
    )
    external.emit(event_step)
    external.advance()
    return result


def _population_counts(spikes: torch.Tensor, network_config) -> torch.Tensor:
    selective = network_config.selective_population_size
    selected = spikes[: network_config.n_memories * selective]
    return selected.reshape(network_config.n_memories, selective).sum(dim=1)


def _capture_boundary(
    network: SparseRecurrentNetwork,
    connectivity: SparseRecurrentConnectivity,
    experiment: MatchedQueryExperimentConfig,
    history_population: int,
) -> _BoundaryState:
    network.reset()
    protocol = _protocol_for_history(connectivity.config, experiment, history_population)
    external = ExternalInputEngine(
        connectivity.config,
        protocol,
        device=network.device,
        dtype=network.dtype,
    )
    boundary_step = _grid_step(experiment.query_origin_ms, connectivity.config.dt_ms)
    delay_start_step = _grid_step(
        experiment.delay_readout_start_ms, connectivity.config.dt_ms
    )
    delay_counts = torch.zeros(
        connectivity.config.n_memories,
        dtype=torch.int64,
        device=network.device,
    )
    for event_step in range(1, boundary_step + 1):
        result = _advance_one_step(network, external, event_step)
        if event_step > delay_start_step:
            delay_counts += _population_counts(result.spikes, connectivity.config)
    delay_duration_s = (
        experiment.query_origin_ms - experiment.delay_readout_start_ms
    ) / 1_000.0
    delay_rates = delay_counts.to(torch.float64) / (
        connectivity.config.selective_population_size * delay_duration_s
    )
    _, _, pre_release = continuous_plastic_state(network)
    return _BoundaryState(
        history_population=history_population,
        protocol=protocol,
        network=capture_network_checkpoint(network),
        external=external.state_dict(),
        pre_release=pre_release.clone(),
        delay_population_rates_hz=delay_rates,
    )


def _run_branch(
    network: SparseRecurrentNetwork,
    connectivity: SparseRecurrentConnectivity,
    experiment: MatchedQueryExperimentConfig,
    recipient: _BoundaryState,
    donor: _BoundaryState,
) -> _BranchResult:
    restore_network_checkpoint(network, recipient.network)
    replace_plastic_state(network, donor.network.plastic)
    external = ExternalInputEngine(
        connectivity.config,
        recipient.protocol,
        device=network.device,
        dtype=network.dtype,
    )
    external.load_state_dict(recipient.external)
    total_step = _grid_step(experiment.total_time_ms, connectivity.config.dt_ms)
    response_start_step = _grid_step(
        experiment.response_window_start_ms, connectivity.config.dt_ms
    )
    response_stop_step = _grid_step(
        experiment.response_window_stop_ms, connectivity.config.dt_ms
    )
    counts = torch.zeros(
        connectivity.config.n_memories,
        dtype=torch.int64,
        device=network.device,
    )
    total_spikes = torch.zeros((), dtype=torch.int64, device=network.device)
    plastic_events = 0
    static_events = 0
    for event_step in range(recipient.network.step_index + 1, total_step + 1):
        result = _advance_one_step(network, external, event_step)
        total_spikes += result.spikes.sum()
        plastic_events += result.dispatch.plastic_events
        static_events += result.dispatch.static_events
        if response_start_step < event_step <= response_stop_step:
            counts += _population_counts(result.spikes, connectivity.config)
    response_duration_s = (
        experiment.response_window_stop_ms - experiment.response_window_start_ms
    ) / 1_000.0
    rates = counts.to(torch.float64) / (
        connectivity.config.selective_population_size * response_duration_s
    )
    _, _, post_release = continuous_plastic_state(network)
    return _BranchResult(
        name="history{}_from_stsp{}".format(
            recipient.history_population, donor.history_population
        ),
        recipient_history=recipient.history_population,
        donor_history=donor.history_population,
        population_rates_hz=rates.to("cpu"),
        population_spike_counts=counts.to("cpu"),
        post_release=post_release.clone(),
        total_spikes=int(total_spikes.item()),
        plastic_edge_events=plastic_events,
        static_edge_events=static_events,
    )


def _squared_norm(value: torch.Tensor) -> float:
    return float(torch.sum(value * value, dtype=torch.float64).item())


def _distance(left: torch.Tensor, right: torch.Tensor) -> float:
    return math.sqrt(_squared_norm(left - right))


def _donor_projection(
    candidate: torch.Tensor, recipient: torch.Tensor, donor: torch.Tensor
) -> Optional[float]:
    direction = donor - recipient
    denominator = _squared_norm(direction)
    if denominator == 0.0:
        return None
    numerator = float(
        torch.sum((candidate - recipient) * direction, dtype=torch.float64).item()
    )
    return numerator / denominator


def _branch_payload(branch: _BranchResult) -> Dict[str, object]:
    return {
        "name": branch.name,
        "recipient_history": branch.recipient_history,
        "donor_history": branch.donor_history,
        "population_rates_hz": branch.population_rates_hz.tolist(),
        "population_spike_counts": branch.population_spike_counts.tolist(),
        "total_spikes_after_boundary": branch.total_spikes,
        "plastic_edge_events_after_boundary": branch.plastic_edge_events,
        "static_edge_events_after_boundary": branch.static_edge_events,
    }


def _sample_edge_ids(num_edges: int, count: int, device: torch.device) -> torch.Tensor:
    resolved = min(num_edges, count)
    return torch.div(
        torch.arange(resolved, dtype=torch.int64, device=device) * num_edges,
        resolved,
        rounding_mode="floor",
    )


def run_matched_query_substitution(
    connectivity: SparseRecurrentConnectivity,
    output_directory: PathLike,
    *,
    experiment: Optional[MatchedQueryExperimentConfig] = None,
    runtime: Optional[SimulationRunConfig] = None,
    overwrite: bool = False,
    connectivity_path: Optional[PathLike] = None,
) -> Dict[str, object]:
    """Run intact and reciprocal STSP-substitution branches on one graph."""

    resolved_experiment = experiment or MatchedQueryExperimentConfig()
    resolved_runtime = runtime or SimulationRunConfig(progress_interval_steps=0)
    config = connectivity.config
    for population in (
        *resolved_experiment.history_populations,
        resolved_experiment.query_population,
    ):
        if not 0 <= population < config.n_memories:
            raise ValueError("Experiment population lies outside n_memories.")
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("Experiment output directory is non-empty.")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(resolved_runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    dtype = {"float32": torch.float32, "float64": torch.float64}[
        resolved_runtime.dtype
    ]
    network = SparseRecurrentNetwork(
        connectivity,
        device=device,
        dtype=dtype,
        source_chunk_size=resolved_runtime.source_chunk_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    boundaries: Dict[int, _BoundaryState] = {}
    for history in resolved_experiment.history_populations:
        boundaries[history] = _capture_boundary(
            network,
            connectivity,
            resolved_experiment,
            history,
        )
    history_a, history_b = resolved_experiment.history_populations
    branch_pairs = (
        (history_a, history_a),
        (history_b, history_b),
        (history_a, history_b),
        (history_b, history_a),
    )
    branches: Dict[str, _BranchResult] = {}
    for recipient_history, donor_history in branch_pairs:
        result = _run_branch(
            network,
            connectivity,
            resolved_experiment,
            boundaries[recipient_history],
            boundaries[donor_history],
        )
        branches[result.name] = result
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    intact_a = branches["history{}_from_stsp{}".format(history_a, history_a)]
    intact_b = branches["history{}_from_stsp{}".format(history_b, history_b)]
    swap_a_from_b = branches["history{}_from_stsp{}".format(history_a, history_b)]
    swap_b_from_a = branches["history{}_from_stsp{}".format(history_b, history_a)]
    pre_distance = _distance(
        boundaries[history_a].pre_release,
        boundaries[history_b].pre_release,
    )
    spike_projection_a = _donor_projection(
        swap_a_from_b.population_rates_hz,
        intact_a.population_rates_hz,
        intact_b.population_rates_hz,
    )
    spike_projection_b = _donor_projection(
        swap_b_from_a.population_rates_hz,
        intact_b.population_rates_hz,
        intact_a.population_rates_hz,
    )
    state_projection_a = _donor_projection(
        swap_a_from_b.post_release,
        intact_a.post_release,
        intact_b.post_release,
    )
    state_projection_b = _donor_projection(
        swap_b_from_a.post_release,
        intact_b.post_release,
        intact_a.post_release,
    )
    reciprocal_spike_projections = [
        value for value in (spike_projection_a, spike_projection_b) if value is not None
    ]
    reciprocal_state_projections = [
        value for value in (state_projection_a, state_projection_b) if value is not None
    ]
    metrics = {
        "pre_query_release_l2_distance": pre_distance,
        "intact_response_l2_distance_hz": _distance(
            intact_a.population_rates_hz, intact_b.population_rates_hz
        ),
        "intact_post_query_release_l2_distance": _distance(
            intact_a.post_release, intact_b.post_release
        ),
        "history{}_from_history{}_spike_donor_projection".format(
            history_a, history_b
        ): spike_projection_a,
        "history{}_from_history{}_spike_donor_projection".format(
            history_b, history_a
        ): spike_projection_b,
        "history{}_from_history{}_successor_donor_projection".format(
            history_a, history_b
        ): state_projection_a,
        "history{}_from_history{}_successor_donor_projection".format(
            history_b, history_a
        ): state_projection_b,
        "mean_reciprocal_spike_donor_projection": (
            None
            if not reciprocal_spike_projections
            else sum(reciprocal_spike_projections) / len(reciprocal_spike_projections)
        ),
        "mean_reciprocal_successor_donor_projection": (
            None
            if not reciprocal_state_projections
            else sum(reciprocal_state_projections) / len(reciprocal_state_projections)
        ),
        "delay_population_rates_hz": {
            str(history): boundaries[history].delay_population_rates_hz.to("cpu").tolist()
            for history in resolved_experiment.history_populations
        },
        "pilot_direction_supported": bool(
            reciprocal_spike_projections
            and reciprocal_state_projections
            and sum(reciprocal_spike_projections) > 0.0
            and sum(reciprocal_state_projections) > 0.0
        ),
        "interpretation_boundary": (
            "single-network descriptive post-query endpoint pilot; it does not "
            "test the Fig.4 sequential STSP-to-next-window-firing mechanism or "
            "provide confirmatory behavioral/network-level inference"
        ),
    }
    sample_ids = _sample_edge_ids(
        connectivity.plastic.num_edges,
        resolved_experiment.stsp_sample_edges,
        device,
    )
    state_samples = {
        "edge_ids": sample_ids.to("cpu"),
        "pre_query_release": {
            str(history): boundaries[history].pre_release[sample_ids].to("cpu")
            for history in resolved_experiment.history_populations
        },
        "post_query_release": {
            name: branch.post_release[sample_ids].to("cpu")
            for name, branch in branches.items()
        },
    }
    atomic_torch_save(state_samples, output / "state_samples.pt")
    atomic_json_dump(
        {
            "schema_version": 1,
            "network": asdict(config),
            "experiment": asdict(resolved_experiment),
            "runtime": asdict(resolved_runtime),
        },
        output / "experiment_config.json",
    )
    atomic_json_dump(
        {name: _branch_payload(branch) for name, branch in branches.items()},
        output / "branches.json",
    )
    atomic_json_dump(metrics, output / "metrics.json")
    run_info = {
        "status": "complete",
        "elapsed_s": elapsed,
        "device": str(device),
        "dtype": resolved_runtime.dtype,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "connectivity_path": (
            None if connectivity_path is None else str(Path(connectivity_path).resolve())
        ),
        "neurons": config.n_neurons,
        "connections": connectivity.num_edges,
        "plastic_connections": connectivity.plastic.num_edges,
        "branch_count": len(branches),
    }
    atomic_json_dump(run_info, output / "run_info.json")
    manifest = {
        "schema_version": 1,
        "tasks": {
            "matched_query_simulation": {
                "depends_on": [
                    "experiment_config.json",
                    (
                        "in_memory_connectivity"
                        if connectivity_path is None
                        else str(Path(connectivity_path).resolve())
                    ),
                ],
                "outputs": [
                    "branches.json",
                    "state_samples.pt",
                    "metrics.json",
                    "run_info.json",
                ],
            },
        },
    }
    atomic_json_dump(manifest, output / "artifact_manifest.json")

    # Drop full-size branch endpoints before returning a JSON-safe summary.
    return {
        "status": "complete",
        "output_directory": str(output.resolve()),
        "metrics": metrics,
        "run_info": run_info,
    }


__all__ = ["MatchedQueryExperimentConfig", "run_matched_query_substitution"]
