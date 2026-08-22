"""External-input protocols for the PyTorch port of the Tiddia WM model.

The public configuration mirrors the input builders in the upstream NEST
repository.  The runtime keeps those inputs outside the recurrent graph so a
single persisted 20M-edge connectivity artifact can be reused by many tasks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .config import TiddiaNetworkConfig
from .scheduler import SparseDelayBuffer


@dataclass(frozen=True)
class ItemLoadingSignal:
    """Load one item by stimulating one selective excitatory population."""

    population_id: int
    origin_ms: float


@dataclass(frozen=True)
class RandomNonspecificSignal:
    """Stimulate a deterministic random subset of excitatory neurons."""

    origin_ms: float
    fraction: float = 0.15


@dataclass(frozen=True)
class PeriodicReadoutInterval:
    """Half-open interval in which periodic nonspecific pulses are generated."""

    start_ms: float
    stop_ms: float


@dataclass(frozen=True)
class WorkingMemoryProtocolConfig:
    """Complete external-input and task timing configuration.

    ``upstream_run_protocol`` reproduces the active protocol in upstream
    ``run_model.py``: 3 s presimulation, one 350 ms cue to population zero,
    3 s post-cue simulation, Poisson background, and the late -1 mV offset.
    Optional readout/noise/periodic fields expose every upstream input builder.
    """

    total_time_ms: float = 6_000.0
    background_start_ms: float = 0.0
    background_stop_ms: float = 6_000.0
    poisson_input: bool = True
    eta_exc_mv: float = 23.7
    eta_inh_mv: float = 20.5
    sigma_exc_mv: float = 1.0
    sigma_inh_mv: float = 1.0
    external_change_interval_ms: float = 1.0

    item_loading: Tuple[ItemLoadingSignal, ...] = (ItemLoadingSignal(0, 3_000.0),)
    cue_duration_ms: float = 350.0
    cue_amplitude_factor: float = 1.15

    nonspecific_readout_origins_ms: Tuple[float, ...] = ()
    readout_duration_ms: float = 250.0
    readout_amplitude_factor: float = 1.05

    random_nonspecific: Tuple[RandomNonspecificSignal, ...] = ()

    periodic_intervals: Tuple[PeriodicReadoutInterval, ...] = ()
    periodic_period_ms: float = 300.0
    periodic_duration_ms: float = 100.0
    periodic_amplitude_factor: float = 1.075

    eta_end_origin_ms: float = 5_200.0
    eta_exc_end_delta_mv: float = -1.0
    seed: int = 143_202_461

    def __post_init__(self) -> None:
        positive = (
            "total_time_ms",
            "external_change_interval_ms",
            "cue_duration_ms",
            "readout_duration_ms",
            "periodic_period_ms",
            "periodic_duration_ms",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be finite and positive.".format(name))
        if not 0.0 <= self.background_start_ms < self.background_stop_ms:
            raise ValueError("The background interval must be non-empty and non-negative.")
        if self.background_stop_ms > self.total_time_ms:
            raise ValueError("Background input cannot stop after the simulation.")
        for signal in self.item_loading:
            if signal.population_id < 0 or signal.origin_ms < 0.0:
                raise ValueError("Item-loading population and origin must be non-negative.")
        for signal in self.random_nonspecific:
            if signal.origin_ms < 0.0 or not 0.0 <= signal.fraction <= 1.0:
                raise ValueError("Random signal origin/fraction is invalid.")
        for interval in self.periodic_intervals:
            if not 0.0 <= interval.start_ms < interval.stop_ms <= self.total_time_ms:
                raise ValueError("Periodic intervals must lie inside the simulation.")
        for origin in self.nonspecific_readout_origins_ms:
            if origin < 0.0:
                raise ValueError("Readout origins must be non-negative.")
        if self.eta_end_origin_ms < 0.0:
            raise ValueError("eta_end_origin_ms must be non-negative.")

    @classmethod
    def upstream_run_protocol(cls, **overrides) -> "WorkingMemoryProtocolConfig":
        return replace(cls(), **overrides)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExternalSignalDescription:
    """Serializable description of one compiled generator-to-target signal."""

    name: str
    kind: str
    target_count: int
    start_ms: float
    stop_ms: Optional[float]
    rate_hz: Optional[float]
    weight_pa: Optional[float]
    current_mean_pa: Optional[float]
    current_std_pa: Optional[float]


def noise_current_parameters(
    eta_mv: float,
    sigma_mv: float,
    tau_m_ms: float,
    *,
    change_interval_ms: float = 1.0,
    capacitance_pf: float = 250.0,
) -> Tuple[float, float]:
    """Port upstream ``noise_params`` without changing its units or formula."""

    mean_pa = capacitance_pf / tau_m_ms * eta_mv
    std_pa = (
        math.sqrt(2.0 / (tau_m_ms * change_interval_ms))
        * capacitance_pf
        * sigma_mv
    )
    return mean_pa, std_pa


def poisson_rate_and_weight(
    eta_mv: float,
    sigma_mv: float,
    tau_m_ms: float,
    *,
    tau_syn_ms: float = 2.0,
    capacitance_pf: float = 250.0,
) -> Tuple[float, float]:
    """Port upstream ``get_rate_and_weight_poisson``.

    ``sigma_mv`` is deliberately accepted but does not enter the upstream
    formula.  Keeping that signature makes the scientific correspondence
    explicit and prevents a silent reinterpretation of existing parameters.
    """

    del sigma_mv
    mean_pa = capacitance_pf / tau_m_ms * eta_mv
    rate_per_ms = (tau_m_ms * mean_pa / capacitance_pf) ** 2 / (
        2.0 * (tau_m_ms + tau_syn_ms)
    )
    if rate_per_ms <= 0.0:
        raise ValueError("Poisson-equivalent eta must be non-zero.")
    return rate_per_ms * 1_000.0, mean_pa / (rate_per_ms * tau_syn_ms)


def _grid_step(time_ms: float, dt_ms: float) -> int:
    scaled = time_ms / dt_ms
    rounded = int(math.floor(scaled + 0.5))
    if not math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("All protocol times must lie on the simulation grid.")
    return rounded


def _signal_seed(seed: int, signal_index: int, stream: int) -> int:
    state = np.random.SeedSequence([seed, signal_index, stream]).generate_state(
        1, dtype=np.uint64
    )[0]
    return int(state % np.uint64(2**63 - 1))


class _RuntimeSignal:
    def __init__(
        self,
        *,
        name: str,
        targets: torch.Tensor,
        delay_steps: torch.Tensor,
        start_step: int,
        stop_step: Optional[int],
        dt_ms: float,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> None:
        self.name = name
        self.targets = targets
        self.delay_steps = delay_steps
        self.start_step = start_step
        self.stop_step = stop_step
        self.dt_ms = dt_ms
        self.dtype = dtype
        self.generator = generator

    def active(self, event_step: int) -> bool:
        return event_step > self.start_step and (
            self.stop_step is None or event_step <= self.stop_step
        )


class _PoissonSignal(_RuntimeSignal):
    def __init__(self, *, rate_hz: float, weight_pa: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rate_hz = rate_hz
        self.weight_pa = weight_pa
        self._mean_counts = torch.full(
            (self.targets.numel(),),
            rate_hz * self.dt_ms / 1_000.0,
            dtype=self.dtype,
            device=self.targets.device,
        )

    def emit(self, event_step: int, buffer: SparseDelayBuffer) -> None:
        if not self.active(event_step):
            return
        counts = torch.poisson(self._mean_counts, generator=self.generator)
        active = counts > 0.0
        if not bool(active.any().item()):
            return
        buffer.schedule_edges(
            self.targets[active],
            counts[active] * self.weight_pa,
            self.delay_steps[active],
            validate_indices=False,
        )


class _CurrentSignal(_RuntimeSignal):
    def __init__(
        self,
        *,
        mean_pa: float,
        std_pa: float,
        change_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mean_pa = mean_pa
        self.std_pa = std_pa
        self.change_steps = change_steps
        self._held = torch.full(
            (self.targets.numel(),),
            mean_pa,
            dtype=self.dtype,
            device=self.targets.device,
        )

    def emit(self, event_step: int, buffer: SparseDelayBuffer) -> None:
        if not self.active(event_step):
            return
        relative = event_step - self.start_step - 1
        if relative % self.change_steps == 0:
            if self.std_pa == 0.0:
                self._held.fill_(self.mean_pa)
            else:
                self._held.normal_(
                    mean=self.mean_pa, std=self.std_pa, generator=self.generator
                )
        buffer.schedule_edges(
            self.targets,
            self._held,
            self.delay_steps,
            validate_indices=False,
        )


class ExternalInputEngine:
    """Compile and schedule every upstream external-input family.

    Call ``pop_current`` before the recurrent network step, then ``emit`` with
    the resulting one-based grid step, and finally ``advance``.  This matches
    the recurrent scheduler and NEST's delayed event ordering.
    """

    def __init__(
        self,
        network: TiddiaNetworkConfig,
        protocol: WorkingMemoryProtocolConfig,
        *,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.network = network
        self.protocol = protocol
        self.device = torch.device(device)
        self.dtype = dtype
        if protocol.external_change_interval_ms < network.dt_ms:
            raise ValueError("External change interval cannot be below network dt.")
        self.change_steps = _grid_step(
            protocol.external_change_interval_ms, network.dt_ms
        )
        self.spike_buffer = SparseDelayBuffer(
            network.n_neurons,
            network.max_delay_steps,
            device=self.device,
            dtype=dtype,
        )
        self.current_buffer = SparseDelayBuffer(
            network.n_neurons,
            network.max_delay_steps,
            device=self.device,
            dtype=dtype,
        )
        self._signals: List[_RuntimeSignal] = []
        self.descriptions: List[ExternalSignalDescription] = []
        self._compile()

    def _targets(self, start: int, stop: int) -> torch.Tensor:
        return torch.arange(start, stop, dtype=torch.int64, device=self.device)

    def _delays(self, targets: torch.Tensor, signal_index: int) -> torch.Tensor:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.protocol.seed, signal_index, 41])
        )
        raw = rng.uniform(
            self.network.delay_min_ms,
            self.network.delay_max_ms,
            size=targets.numel(),
        )
        steps = np.floor(raw / self.network.dt_ms + 0.5).astype(np.int16)
        steps = np.clip(
            steps, self.network.min_delay_steps, self.network.max_delay_steps
        )
        return torch.as_tensor(steps, dtype=torch.int16, device=self.device)

    def _generator(self, signal_index: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(_signal_seed(self.protocol.seed, signal_index, 73))
        return generator

    def _add_signal(
        self,
        *,
        name: str,
        targets: torch.Tensor,
        start_ms: float,
        stop_ms: Optional[float],
        eta_mv: float,
        sigma_mv: float,
        tau_m_ms: float,
        force_current: bool = False,
    ) -> None:
        signal_index = len(self._signals)
        common = dict(
            name=name,
            targets=targets,
            delay_steps=self._delays(targets, signal_index),
            start_step=_grid_step(start_ms, self.network.dt_ms),
            stop_step=(
                None if stop_ms is None else _grid_step(stop_ms, self.network.dt_ms)
            ),
            dt_ms=self.network.dt_ms,
            dtype=self.dtype,
            generator=self._generator(signal_index),
        )
        if self.protocol.poisson_input and not force_current:
            rate_hz, weight_pa = poisson_rate_and_weight(
                eta_mv,
                sigma_mv,
                tau_m_ms,
                tau_syn_ms=self.network.tau_syn_ms,
                capacitance_pf=self.network.capacitance_pf,
            )
            signal = _PoissonSignal(
                rate_hz=rate_hz, weight_pa=weight_pa, **common
            )
            description = ExternalSignalDescription(
                name, "poisson", targets.numel(), start_ms, stop_ms,
                rate_hz, weight_pa, None, None,
            )
        else:
            mean_pa, std_pa = noise_current_parameters(
                eta_mv,
                sigma_mv,
                tau_m_ms,
                change_interval_ms=self.protocol.external_change_interval_ms,
                capacitance_pf=self.network.capacitance_pf,
            )
            signal = _CurrentSignal(
                mean_pa=mean_pa,
                std_pa=std_pa,
                change_steps=self.change_steps,
                **common,
            )
            description = ExternalSignalDescription(
                name, "current", targets.numel(), start_ms, stop_ms,
                None, None, mean_pa, std_pa,
            )
        self._signals.append(signal)
        self.descriptions.append(description)

    def _compile(self) -> None:
        p = self.protocol
        n = self.network
        exc = self._targets(0, n.n_exc)
        inh = self._targets(n.n_exc, n.n_neurons)
        self._add_signal(
            name="background_exc",
            targets=exc,
            start_ms=p.background_start_ms,
            stop_ms=p.background_stop_ms,
            eta_mv=p.eta_exc_mv,
            sigma_mv=p.sigma_exc_mv,
            tau_m_ms=n.tau_m_exc_ms,
        )
        self._add_signal(
            name="background_inh",
            targets=inh,
            start_ms=p.background_start_ms,
            stop_ms=p.background_stop_ms,
            eta_mv=p.eta_inh_mv,
            sigma_mv=p.sigma_inh_mv,
            tau_m_ms=n.tau_m_inh_ms,
        )

        # Upstream uses tau_m_inh for this excitatory offset; retain it exactly.
        self._add_signal(
            name="late_exc_offset",
            targets=exc,
            start_ms=p.eta_end_origin_ms,
            stop_ms=None,
            eta_mv=p.eta_exc_end_delta_mv,
            sigma_mv=0.0,
            tau_m_ms=n.tau_m_inh_ms,
            force_current=True,
        )

        selective_size = n.selective_population_size
        for index, cue in enumerate(p.item_loading):
            if cue.population_id >= n.n_memories:
                raise ValueError("Item-loading population lies outside n_memories.")
            start = cue.population_id * selective_size
            self._add_signal(
                name="item_loading_{}".format(index),
                targets=self._targets(start, start + selective_size),
                start_ms=cue.origin_ms,
                stop_ms=cue.origin_ms + p.cue_duration_ms,
                eta_mv=p.eta_exc_mv * (p.cue_amplitude_factor - 1.0),
                sigma_mv=0.0,
                tau_m_ms=n.tau_m_exc_ms,
            )

        for index, origin in enumerate(p.nonspecific_readout_origins_ms):
            self._add_signal(
                name="nonspecific_readout_{}".format(index),
                targets=exc,
                start_ms=origin,
                stop_ms=origin + p.readout_duration_ms,
                eta_mv=p.eta_exc_mv * (p.readout_amplitude_factor - 1.0),
                sigma_mv=0.0,
                tau_m_ms=n.tau_m_exc_ms,
            )

        for index, signal in enumerate(p.random_nonspecific):
            count = int(signal.fraction * n.n_exc)
            rng = np.random.default_rng(
                np.random.SeedSequence([p.seed, index, 97])
            )
            selected = rng.choice(n.n_exc, size=count, replace=False).astype(np.int64)
            targets = torch.as_tensor(selected, dtype=torch.int64, device=self.device)
            self._add_signal(
                name="random_nonspecific_{}".format(index),
                targets=targets,
                start_ms=signal.origin_ms,
                stop_ms=signal.origin_ms + p.cue_duration_ms,
                eta_mv=p.eta_exc_mv * (p.cue_amplitude_factor - 1.0),
                sigma_mv=0.0,
                tau_m_ms=n.tau_m_exc_ms,
            )

        periodic_origins: List[float] = []
        for interval in p.periodic_intervals:
            value = interval.start_ms
            while value < interval.stop_ms - 1e-9:
                periodic_origins.append(value)
                value += p.periodic_period_ms
        for index, origin in enumerate(periodic_origins):
            self._add_signal(
                name="periodic_readout_{}".format(index),
                targets=exc,
                start_ms=origin,
                stop_ms=origin + p.periodic_duration_ms,
                eta_mv=p.eta_exc_mv * (p.periodic_amplitude_factor - 1.0),
                sigma_mv=0.0,
                tau_m_ms=n.tau_m_exc_ms,
            )

    def pop_current(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spike_ex, spike_in = self.spike_buffer.pop_current()
        current_positive, current_negative = self.current_buffer.pop_current()
        return spike_ex, spike_in, current_positive + current_negative

    @torch.no_grad()
    def emit(self, event_step: int) -> None:
        if event_step < 1:
            raise ValueError("event_step is one-based and must be positive.")
        for signal in self._signals:
            if isinstance(signal, _PoissonSignal):
                signal.emit(event_step, self.spike_buffer)
            else:
                signal.emit(event_step, self.current_buffer)

    def advance(self) -> None:
        self.spike_buffer.advance()
        self.current_buffer.advance()

    def reset(self) -> None:
        self.spike_buffer.reset()
        self.current_buffer.reset()
        # Recompile to restore RNG streams and held-current state exactly.
        self._signals.clear()
        self.descriptions.clear()
        self._compile()

    def description_dicts(self) -> List[Dict[str, object]]:
        return [asdict(item) for item in self.descriptions]


__all__ = [
    "ExternalInputEngine",
    "ExternalSignalDescription",
    "ItemLoadingSignal",
    "PeriodicReadoutInterval",
    "RandomNonspecificSignal",
    "WorkingMemoryProtocolConfig",
    "noise_current_parameters",
    "poisson_rate_and_weight",
]
