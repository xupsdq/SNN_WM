"""Sparse recurrent event scheduling and network stepping on PyTorch devices."""

from dataclasses import dataclass
import math
from typing import Iterator, Optional, Tuple, Union

import torch

from .connectivity import (
    PlasticCsrEdges,
    SparseRecurrentConnectivity,
    StaticCsrEdges,
)
from .nest_equivalent import (
    IafPscExpParameters,
    IafPscExpPropagators,
    IafPscExpState,
    iaf_psc_exp_step,
    make_iaf_psc_exp_state,
    tsodyks3_event_values,
)


TensorOrScalar = Union[torch.Tensor, float]


@dataclass(frozen=True)
class DispatchStats:
    """Number of sources and edges processed for one emitted spike set."""

    active_sources: int
    plastic_events: int
    static_events: int

    @property
    def total_events(self) -> int:
        return self.plastic_events + self.static_events


@dataclass
class PlasticRuntimeState:
    """Mutable per-edge state owned by one event scheduler."""

    u: torch.Tensor
    x: torch.Tensor
    last_spike_time_ms: torch.Tensor


class SparseDelayBuffer:
    """Aggregate edge currents by delay slot, sign channel, and target."""

    def __init__(
        self,
        n_neurons: int,
        max_delay_steps: int,
        *,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> None:
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive.")
        if max_delay_steps < 1:
            raise ValueError("max_delay_steps must be positive.")
        if not dtype.is_floating_point:
            raise TypeError("SparseDelayBuffer requires a floating-point dtype.")
        self.n_neurons = n_neurons
        self.max_delay_steps = max_delay_steps
        self.slot_count = max_delay_steps + 1
        self.excitatory = torch.zeros(
            (self.slot_count, n_neurons), dtype=dtype, device=device
        )
        self.inhibitory = torch.zeros_like(self.excitatory)
        self.cursor = 0

    def schedule_edges(
        self,
        targets: torch.Tensor,
        values: torch.Tensor,
        delay_steps: torch.Tensor,
        *,
        validate_indices: bool = True,
    ) -> None:
        if not (targets.numel() == values.numel() == delay_steps.numel()):
            raise ValueError("targets, values, and delay_steps must have equal length.")
        if targets.numel() == 0:
            return
        target_index = targets.to(dtype=torch.int64)
        delay_index = delay_steps.to(dtype=torch.int64)
        if validate_indices:
            if bool((delay_index < 1).any().item()) or bool(
                (delay_index > self.max_delay_steps).any().item()
            ):
                raise ValueError("An event delay lies outside the ring-buffer range.")
            if bool((target_index < 0).any().item()) or bool(
                (target_index >= self.n_neurons).any().item()
            ):
                raise ValueError("An event target lies outside the neuron range.")
        slots = (delay_index + self.cursor) % self.slot_count
        flat_index = slots * self.n_neurons + target_index
        self.excitatory.view(-1).index_add_(
            0, flat_index, torch.clamp_min(values, 0.0)
        )
        self.inhibitory.view(-1).index_add_(
            0, flat_index, torch.clamp_max(values, 0.0)
        )

    def pop_current(self) -> Tuple[torch.Tensor, torch.Tensor]:
        excitatory = self.excitatory[self.cursor].clone()
        inhibitory = self.inhibitory[self.cursor].clone()
        self.excitatory[self.cursor].zero_()
        self.inhibitory[self.cursor].zero_()
        return excitatory, inhibitory

    def advance(self) -> None:
        self.cursor = (self.cursor + 1) % self.slot_count

    def reset(self) -> None:
        self.excitatory.zero_()
        self.inhibitory.zero_()
        self.cursor = 0


def _expanded_edge_ids(row_ptr: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    if sources.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=row_ptr.device)
    starts = row_ptr[sources]
    counts = row_ptr[sources + 1] - starts
    nonempty = counts > 0
    starts = starts[nonempty]
    counts = counts[nonempty]
    if counts.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=row_ptr.device)
    total = int(counts.sum().item())
    group_offsets = torch.cumsum(counts, dim=0) - counts
    bases = starts - group_offsets
    return torch.repeat_interleave(bases, counts) + torch.arange(
        total, dtype=torch.int64, device=row_ptr.device
    )


class SparseEventScheduler:
    """Dispatch spikes over source-major CSR edges without dense matrices."""

    def __init__(
        self,
        connectivity: SparseRecurrentConnectivity,
        *,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        source_chunk_size: int = 256,
    ) -> None:
        if not dtype.is_floating_point:
            raise TypeError("SparseEventScheduler requires a floating-point dtype.")
        if source_chunk_size <= 0:
            raise ValueError("source_chunk_size must be positive.")
        requested_device = torch.device(device)
        self.dtype = dtype
        self.connectivity = connectivity.to(requested_device, float_dtype=dtype)
        self.device = self.connectivity.plastic.row_ptr.device
        self.config = self.connectivity.config
        self.source_chunk_size = source_chunk_size
        self.plastic_state = PlasticRuntimeState(
            u=self.connectivity.plastic.initial_u.clone(),
            x=self.connectivity.plastic.initial_x.clone(),
            last_spike_time_ms=torch.zeros_like(self.connectivity.plastic.initial_u),
        )
        self.delay_buffer = SparseDelayBuffer(
            self.config.n_neurons,
            self.config.max_delay_steps,
            device=self.device,
            dtype=dtype,
        )
        self._last_dispatch_time_ms = 0.0

    @property
    def storage_bytes(self) -> int:
        runtime = (
            self.plastic_state.u.numel() * self.plastic_state.u.element_size()
            + self.plastic_state.x.numel() * self.plastic_state.x.element_size()
            + self.plastic_state.last_spike_time_ms.numel()
            * self.plastic_state.last_spike_time_ms.element_size()
            + self.delay_buffer.excitatory.numel()
            * self.delay_buffer.excitatory.element_size()
            + self.delay_buffer.inhibitory.numel()
            * self.delay_buffer.inhibitory.element_size()
        )
        return self.connectivity.storage_bytes + runtime

    def reset(self) -> None:
        self.plastic_state.u.copy_(self.connectivity.plastic.initial_u)
        self.plastic_state.x.copy_(self.connectivity.plastic.initial_x)
        self.plastic_state.last_spike_time_ms.zero_()
        self.delay_buffer.reset()
        self._last_dispatch_time_ms = 0.0

    def _source_chunks(self, sources: torch.Tensor) -> Iterator[torch.Tensor]:
        yield from sources.split(self.source_chunk_size)

    def _dispatch_static(
        self,
        edges: StaticCsrEdges,
        sources: torch.Tensor,
    ) -> int:
        event_count = 0
        for source_chunk in self._source_chunks(sources):
            edge_ids = _expanded_edge_ids(edges.row_ptr, source_chunk)
            if edge_ids.numel() == 0:
                continue
            self.delay_buffer.schedule_edges(
                edges.targets[edge_ids],
                edges.weights[edge_ids],
                edges.delay_steps[edge_ids],
                validate_indices=False,
            )
            event_count += edge_ids.numel()
        return event_count

    def _dispatch_plastic(
        self,
        edges: PlasticCsrEdges,
        sources: torch.Tensor,
        time_ms: float,
    ) -> int:
        event_count = 0
        baseline_u = self.config.stsp_u
        for source_chunk in self._source_chunks(sources):
            edge_ids = _expanded_edge_ids(edges.row_ptr, source_chunk)
            if edge_ids.numel() == 0:
                continue
            elapsed = time_ms - self.plastic_state.last_spike_time_ms[edge_ids]
            updated_u, updated_x, released = tsodyks3_event_values(
                u=self.plastic_state.u[edge_ids],
                x=self.plastic_state.x[edge_ids],
                elapsed_ms=elapsed,
                baseline_u=baseline_u,
                tau_rec_ms=edges.tau_rec_ms[edge_ids],
                tau_fac_ms=edges.tau_fac_ms[edge_ids],
                weight=edges.weights[edge_ids],
            )
            self.plastic_state.u.index_copy_(0, edge_ids, updated_u)
            self.plastic_state.x.index_copy_(0, edge_ids, updated_x)
            self.plastic_state.last_spike_time_ms[edge_ids] = time_ms
            self.delay_buffer.schedule_edges(
                edges.targets[edge_ids],
                released,
                edges.delay_steps[edge_ids],
                validate_indices=False,
            )
            event_count += edge_ids.numel()
        return event_count

    @torch.no_grad()
    def dispatch_spikes(
        self,
        spikes: torch.Tensor,
        *,
        time_ms: float,
    ) -> DispatchStats:
        """Update active synapses and enqueue their delayed target currents."""

        if spikes.shape != (self.config.n_neurons,):
            raise ValueError("spikes must have one element per neuron.")
        if spikes.device != self.device:
            raise ValueError("spikes must already be on the scheduler device.")
        if not math.isfinite(time_ms) or time_ms < self._last_dispatch_time_ms:
            raise ValueError("time_ms must be finite and nondecreasing.")
        sources = torch.nonzero(spikes.to(dtype=torch.bool), as_tuple=False).flatten()
        plastic_events = self._dispatch_plastic(
            self.connectivity.plastic, sources, time_ms
        )
        static_events = self._dispatch_static(self.connectivity.static, sources)
        self._last_dispatch_time_ms = time_ms
        return DispatchStats(
            active_sources=sources.numel(),
            plastic_events=plastic_events,
            static_events=static_events,
        )

    def pop_current(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.delay_buffer.pop_current()

    def advance(self) -> None:
        self.delay_buffer.advance()


@dataclass(frozen=True)
class RecurrentStepResult:
    """Observable state produced by one recurrent network grid step."""

    time_ms: float
    spikes: torch.Tensor
    voltage_mv: torch.Tensor
    dispatch: DispatchStats


class SparseRecurrentNetwork:
    """10k-neuron-capable recurrent simulator built on the equivalent kernels."""

    def __init__(
        self,
        connectivity: SparseRecurrentConnectivity,
        *,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        source_chunk_size: int = 256,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = connectivity.config
        self.scheduler = SparseEventScheduler(
            connectivity,
            device=self.device,
            dtype=dtype,
            source_chunk_size=source_chunk_size,
        )
        common = {
            "c_m": self.config.capacitance_pf,
            "tau_syn_ex": self.config.tau_syn_ms,
            "tau_syn_in": self.config.tau_syn_ms,
            "dt": self.config.dt_ms,
        }
        self.exc_params = IafPscExpParameters(
            tau_m=self.config.tau_m_exc_ms,
            t_ref=self.config.refractory_exc_ms,
            e_l=self.config.resting_exc_mv,
            v_th=self.config.threshold_exc_mv,
            v_reset=self.config.reset_exc_mv,
            **common,
        )
        self.inh_params = IafPscExpParameters(
            tau_m=self.config.tau_m_inh_ms,
            t_ref=self.config.refractory_inh_ms,
            e_l=self.config.resting_inh_mv,
            v_th=self.config.threshold_inh_mv,
            v_reset=self.config.reset_inh_mv,
            **common,
        )
        self.exc_propagators = IafPscExpPropagators.from_parameters(self.exc_params)
        self.inh_propagators = IafPscExpPropagators.from_parameters(self.inh_params)
        self.exc_state = self._new_neuron_state(self.config.n_exc, self.exc_params)
        self.inh_state = self._new_neuron_state(self.config.n_inh, self.inh_params)
        self.step_index = 0

    def _new_neuron_state(
        self, n_neurons: int, params: IafPscExpParameters
    ) -> IafPscExpState:
        return make_iaf_psc_exp_state(
            (n_neurons,), params, device=self.device, dtype=self.dtype
        )

    def reset(self) -> None:
        self.scheduler.reset()
        self.exc_state = self._new_neuron_state(self.config.n_exc, self.exc_params)
        self.inh_state = self._new_neuron_state(self.config.n_inh, self.inh_params)
        self.step_index = 0

    def _vector(self, value: TensorOrScalar) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        try:
            return torch.broadcast_to(tensor, (self.config.n_neurons,))
        except RuntimeError as exc:
            raise ValueError(
                "External input must be scalar or have one value per neuron."
            ) from exc

    @torch.no_grad()
    def step(
        self,
        *,
        external_current_pa: TensorOrScalar = 0.0,
        external_current_0_pa: TensorOrScalar = 0.0,
        external_spikes_ex_pa: TensorOrScalar = 0.0,
        external_spikes_in_pa: TensorOrScalar = 0.0,
    ) -> RecurrentStepResult:
        recurrent_ex, recurrent_in = self.scheduler.pop_current()
        incoming_ex = recurrent_ex + self._vector(external_spikes_ex_pa)
        incoming_in = recurrent_in + self._vector(external_spikes_in_pa)
        current = self._vector(external_current_pa)
        current_0 = self._vector(external_current_0_pa)
        split = self.config.n_exc

        self.exc_state, exc_spikes = iaf_psc_exp_step(
            self.exc_state,
            self.exc_params,
            propagators=self.exc_propagators,
            incoming_spikes_ex=incoming_ex[:split],
            incoming_spikes_in=incoming_in[:split],
            incoming_current_0=current_0[:split],
            constant_current=current[:split],
        )
        self.inh_state, inh_spikes = iaf_psc_exp_step(
            self.inh_state,
            self.inh_params,
            propagators=self.inh_propagators,
            incoming_spikes_ex=incoming_ex[split:],
            incoming_spikes_in=incoming_in[split:],
            incoming_current_0=current_0[split:],
            constant_current=current[split:],
        )
        spikes = torch.cat((exc_spikes, inh_spikes))
        time_ms = (self.step_index + 1) * self.config.dt_ms
        dispatch = self.scheduler.dispatch_spikes(spikes, time_ms=time_ms)
        self.scheduler.advance()
        self.step_index += 1
        voltage = torch.cat(
            (
                self.exc_state.absolute_voltage(self.exc_params),
                self.inh_state.absolute_voltage(self.inh_params),
            )
        )
        return RecurrentStepResult(
            time_ms=time_ms,
            spikes=spikes,
            voltage_mv=voltage,
            dispatch=dispatch,
        )
