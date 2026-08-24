"""Sparse STSP boundary states and exact presynaptic-event replay."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

import torch

from .connectivity import PlasticCsrEdges
from .nest_equivalent import tsodyks3_event_values
from .scheduler import SparseRecurrentNetwork


DeviceLike = Union[str, torch.device]


@dataclass(frozen=True)
class IndexedPlasticState:
    """Continuous ``u/x`` for an indexed edge subset at one time boundary.

    Values are canonicalized at ``time_ms``.  Replacing a network state stamps
    that boundary as the new last-event time, which is dynamically equivalent
    to retaining the historical event timestamp together with recovered values.
    """

    time_ms: float
    edge_ids: torch.Tensor
    u: torch.Tensor
    x: torch.Tensor

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_ms) or self.time_ms < 0.0:
            raise ValueError("Indexed STSP time must be finite and non-negative.")
        if self.edge_ids.dtype != torch.int64:
            raise TypeError("Indexed STSP edge_ids must use int64.")
        if self.edge_ids.ndim != 1:
            raise ValueError("Indexed STSP edge_ids must be one-dimensional.")
        if self.u.shape != self.edge_ids.shape or self.x.shape != self.edge_ids.shape:
            raise ValueError("Indexed STSP values must match edge_ids.")
        if self.u.device != self.x.device or self.edge_ids.device != self.u.device:
            raise ValueError("Indexed STSP tensors must share one device.")
        if not self.u.dtype.is_floating_point or self.x.dtype != self.u.dtype:
            raise TypeError("Indexed STSP u/x must share a floating-point dtype.")

    @property
    def num_edges(self) -> int:
        return int(self.edge_ids.numel())

    def to(
        self,
        device: DeviceLike,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> "IndexedPlasticState":
        resolved_dtype = self.u.dtype if dtype is None else dtype
        return IndexedPlasticState(
            time_ms=self.time_ms,
            edge_ids=self.edge_ids.to(device=device),
            u=self.u.to(device=device, dtype=resolved_dtype),
            x=self.x.to(device=device, dtype=resolved_dtype),
        )


@dataclass(frozen=True)
class RawIndexedPlasticState:
    """Indexed scheduler state with its exact per-edge event timestamps.

    Unlike :class:`IndexedPlasticState`, this representation is not
    canonicalized to ``time_ms``.  It is used when a replay must reproduce the
    floating-point update path of a live scheduler rather than only the
    mathematically equivalent continuous boundary state.
    """

    time_ms: float
    edge_ids: torch.Tensor
    u: torch.Tensor
    x: torch.Tensor
    last_event_time_ms: torch.Tensor

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_ms) or self.time_ms < 0.0:
            raise ValueError("Raw indexed STSP time must be finite and non-negative.")
        if self.edge_ids.dtype != torch.int64 or self.edge_ids.ndim != 1:
            raise TypeError("Raw indexed STSP edge_ids must be one-dimensional int64.")
        tensors = (self.u, self.x, self.last_event_time_ms)
        if any(tensor.shape != self.edge_ids.shape for tensor in tensors):
            raise ValueError("Raw indexed STSP values must match edge_ids.")
        if any(tensor.device != self.edge_ids.device for tensor in tensors):
            raise ValueError("Raw indexed STSP tensors must share one device.")
        if not self.u.dtype.is_floating_point:
            raise TypeError("Raw indexed STSP values must be floating point.")
        if self.x.dtype != self.u.dtype or self.last_event_time_ms.dtype != self.u.dtype:
            raise TypeError("Raw indexed STSP values must share one dtype.")

    def to(
        self,
        device: DeviceLike,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> "RawIndexedPlasticState":
        resolved_dtype = self.u.dtype if dtype is None else dtype
        return RawIndexedPlasticState(
            time_ms=self.time_ms,
            edge_ids=self.edge_ids.to(device=device),
            u=self.u.to(device=device, dtype=resolved_dtype),
            x=self.x.to(device=device, dtype=resolved_dtype),
            last_event_time_ms=self.last_event_time_ms.to(
                device=device, dtype=resolved_dtype
            ),
        )


def _validated_edge_ids(
    edge_ids: torch.Tensor,
    *,
    num_edges: int,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.as_tensor(edge_ids, dtype=torch.int64, device=device).flatten()
    if indices.numel() == 0:
        return indices
    if bool((indices < 0).any().item()) or bool((indices >= num_edges).any().item()):
        raise ValueError("An indexed STSP edge lies outside the plastic graph.")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("Indexed STSP edge_ids must be unique.")
    return indices


@torch.no_grad()
def capture_indexed_plastic_state(
    network: SparseRecurrentNetwork,
    edge_ids: torch.Tensor,
    *,
    storage_device: DeviceLike = "cpu",
) -> IndexedPlasticState:
    """Capture continuous state for selected edges without copying all edges."""

    scheduler = network.scheduler
    edges = scheduler.connectivity.plastic
    indices = _validated_edge_ids(
        edge_ids, num_edges=edges.num_edges, device=network.device
    )
    time_ms = network.step_index * network.config.dt_ms
    state = scheduler.plastic_state
    elapsed = torch.clamp_min(time_ms - state.last_spike_time_ms[indices], 0.0)
    baseline_u = network.config.stsp_u
    u = baseline_u + (state.u[indices] - baseline_u) * torch.exp(
        -elapsed / edges.tau_fac_ms[indices]
    )
    x = 1.0 + (state.x[indices] - 1.0) * torch.exp(
        -elapsed / edges.tau_rec_ms[indices]
    )
    target = torch.device(storage_device)
    return IndexedPlasticState(
        time_ms=time_ms,
        edge_ids=indices.to(device=target),
        u=u.to(device=target),
        x=x.to(device=target),
    )


@torch.no_grad()
def capture_raw_indexed_plastic_state(
    network: SparseRecurrentNetwork,
    edge_ids: torch.Tensor,
    *,
    storage_device: DeviceLike = "cpu",
) -> RawIndexedPlasticState:
    """Capture exact indexed scheduler values and their last-event times."""

    scheduler = network.scheduler
    indices = _validated_edge_ids(
        edge_ids,
        num_edges=scheduler.connectivity.plastic.num_edges,
        device=network.device,
    )
    state = scheduler.plastic_state
    target = torch.device(storage_device)
    return RawIndexedPlasticState(
        time_ms=network.step_index * network.config.dt_ms,
        edge_ids=indices.to(device=target),
        u=state.u[indices].to(device=target),
        x=state.x[indices].to(device=target),
        last_event_time_ms=state.last_spike_time_ms[indices].to(device=target),
    )


@torch.no_grad()
def replace_indexed_plastic_state(
    network: SparseRecurrentNetwork,
    donor: IndexedPlasticState,
) -> None:
    """Replace selected continuous STSP while preserving all fast state."""

    current_time = network.step_index * network.config.dt_ms
    if not math.isclose(
        donor.time_ms,
        current_time,
        rel_tol=0.0,
        abs_tol=network.config.dt_ms * 1e-6,
    ):
        raise ValueError("Indexed donor state must come from the same time boundary.")
    state = network.scheduler.plastic_state
    indices = _validated_edge_ids(
        donor.edge_ids, num_edges=state.u.numel(), device=network.device
    )
    if indices.numel() == 0:
        return
    state.u.index_copy_(0, indices, donor.u.to(state.u))
    state.x.index_copy_(0, indices, donor.x.to(state.x))
    state.last_spike_time_ms.index_fill_(0, indices, current_time)


def edge_sources_for_ids(
    edges: PlasticCsrEdges,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    """Return the presynaptic neuron of each source-major CSR edge id."""

    indices = _validated_edge_ids(
        edge_ids,
        num_edges=edges.num_edges,
        device=edges.row_ptr.device,
    )
    if indices.numel() == 0:
        return indices
    return torch.searchsorted(edges.row_ptr[1:], indices, right=True)


def _indexed_parameters(
    edges: PlasticCsrEdges,
    state: IndexedPlasticState,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    graph_indices = state.edge_ids.to(device=edges.weights.device)
    target = state.u.device
    dtype = state.u.dtype
    return (
        edges.weights[graph_indices].to(device=target, dtype=dtype),
        edges.tau_rec_ms[graph_indices].to(device=target, dtype=dtype),
        edges.tau_fac_ms[graph_indices].to(device=target, dtype=dtype),
        edge_sources_for_ids(edges, graph_indices).to(device=target),
    )


def passive_evolve_indexed_state(
    edges: PlasticCsrEdges,
    state: IndexedPlasticState,
    *,
    end_time_ms: float,
    baseline_u: float,
) -> IndexedPlasticState:
    """Evolve selected state to a later boundary with no presynaptic events."""

    if not math.isfinite(end_time_ms) or end_time_ms < state.time_ms:
        raise ValueError("Passive STSP end time must not precede the start state.")
    _, tau_rec, tau_fac, _ = _indexed_parameters(edges, state)
    elapsed = end_time_ms - state.time_ms
    u = baseline_u + (state.u - baseline_u) * torch.exp(-elapsed / tau_fac)
    x = 1.0 + (state.x - 1.0) * torch.exp(-elapsed / tau_rec)
    return IndexedPlasticState(end_time_ms, state.edge_ids.clone(), u, x)


def indexed_next_release(
    edges: PlasticCsrEdges,
    state: IndexedPlasticState,
    *,
    baseline_u: float,
) -> torch.Tensor:
    """Return hypothetical next-spike released current for selected edges."""

    weights, _, _, _ = _indexed_parameters(edges, state)
    next_u = state.u + baseline_u * (1.0 - state.u)
    return weights * next_u * state.x


def replay_indexed_plastic_state(
    edges: PlasticCsrEdges,
    start: IndexedPlasticState,
    *,
    spike_times_ms: torch.Tensor,
    spike_sources: torch.Tensor,
    end_time_ms: float,
    baseline_u: float,
) -> IndexedPlasticState:
    """Exactly replay source spikes through ``tsodyks3`` on selected edges."""

    if not math.isfinite(end_time_ms) or end_time_ms < start.time_ms:
        raise ValueError("Replay end time must not precede the start state.")
    times = torch.as_tensor(
        spike_times_ms, dtype=start.u.dtype, device=start.u.device
    ).flatten()
    sources = torch.as_tensor(
        spike_sources, dtype=torch.int64, device=start.u.device
    ).flatten()
    if times.shape != sources.shape:
        raise ValueError("Replay spike times and sources must have equal shape.")
    if times.numel():
        if bool((times < start.time_ms).any().item()) or bool(
            (times > end_time_ms).any().item()
        ):
            raise ValueError("A replay spike lies outside the requested interval.")
        order = torch.argsort(times, stable=True)
        times = times[order]
        sources = sources[order]

    weights, tau_rec, tau_fac, edge_sources = _indexed_parameters(edges, start)
    source_order = torch.argsort(edge_sources, stable=True)
    sorted_sources = edge_sources[source_order]
    unique_sources, counts = torch.unique_consecutive(
        sorted_sources, return_counts=True
    )
    offsets = torch.cumsum(counts, dim=0) - counts
    positions = {
        int(source.item()): source_order[int(offset.item()) : int(offset.item() + count.item())]
        for source, offset, count in zip(unique_sources, offsets, counts)
    }

    u = start.u.clone()
    x = start.x.clone()
    last_event = torch.full_like(u, start.time_ms)
    for event_time, source in zip(times, sources):
        selected = positions.get(int(source.item()))
        if selected is None:
            continue
        elapsed = event_time - last_event[selected]
        updated_u, updated_x, _ = tsodyks3_event_values(
            u=u[selected],
            x=x[selected],
            elapsed_ms=elapsed,
            baseline_u=baseline_u,
            tau_rec_ms=tau_rec[selected],
            tau_fac_ms=tau_fac[selected],
            weight=weights[selected],
        )
        u.index_copy_(0, selected, updated_u)
        x.index_copy_(0, selected, updated_x)
        last_event.index_fill_(0, selected, float(event_time.item()))

    elapsed_to_end = end_time_ms - last_event
    u = baseline_u + (u - baseline_u) * torch.exp(-elapsed_to_end / tau_fac)
    x = 1.0 + (x - 1.0) * torch.exp(-elapsed_to_end / tau_rec)
    return IndexedPlasticState(end_time_ms, start.edge_ids.clone(), u, x)


@torch.no_grad()
def replay_raw_indexed_plastic_state(
    edges: PlasticCsrEdges,
    start: RawIndexedPlasticState,
    *,
    spike_times_ms: torch.Tensor,
    spike_sources: torch.Tensor,
    end_time_ms: float,
    baseline_u: float,
) -> IndexedPlasticState:
    """Replay events from exact scheduler values and last-event timestamps."""

    if not math.isfinite(end_time_ms) or end_time_ms < start.time_ms:
        raise ValueError("Replay end time must not precede the raw start state.")
    times = torch.as_tensor(
        spike_times_ms, dtype=start.u.dtype, device=start.u.device
    ).flatten()
    sources = torch.as_tensor(
        spike_sources, dtype=torch.int64, device=start.u.device
    ).flatten()
    if times.shape != sources.shape:
        raise ValueError("Replay spike times and sources must have equal shape.")
    if times.numel():
        if bool((times < start.time_ms).any().item()) or bool(
            (times > end_time_ms).any().item()
        ):
            raise ValueError("A replay spike lies outside the requested interval.")
        order = torch.argsort(times, stable=True)
        times = times[order]
        sources = sources[order]

    indexed = IndexedPlasticState(
        start.time_ms,
        start.edge_ids,
        start.u,
        start.x,
    )
    weights, tau_rec, tau_fac, edge_sources = _indexed_parameters(edges, indexed)
    source_order = torch.argsort(edge_sources, stable=True)
    sorted_sources = edge_sources[source_order]
    unique_sources, counts = torch.unique_consecutive(
        sorted_sources, return_counts=True
    )
    offsets = torch.cumsum(counts, dim=0) - counts
    positions = {
        int(source.item()): source_order[
            int(offset.item()) : int(offset.item() + count.item())
        ]
        for source, offset, count in zip(unique_sources, offsets, counts)
    }

    u = start.u.clone()
    x = start.x.clone()
    last_event = start.last_event_time_ms.clone()
    for event_time, source in zip(times, sources):
        selected = positions.get(int(source.item()))
        if selected is None:
            continue
        elapsed = event_time - last_event[selected]
        updated_u, updated_x, _ = tsodyks3_event_values(
            u=u[selected],
            x=x[selected],
            elapsed_ms=elapsed,
            baseline_u=baseline_u,
            tau_rec_ms=tau_rec[selected],
            tau_fac_ms=tau_fac[selected],
            weight=weights[selected],
        )
        u.index_copy_(0, selected, updated_u)
        x.index_copy_(0, selected, updated_x)
        last_event.index_fill_(0, selected, float(event_time.item()))

    elapsed_to_end = end_time_ms - last_event
    u = baseline_u + (u - baseline_u) * torch.exp(-elapsed_to_end / tau_fac)
    x = 1.0 + (x - 1.0) * torch.exp(-elapsed_to_end / tau_rec)
    return IndexedPlasticState(end_time_ms, start.edge_ids.clone(), u, x)


def event_induced_release_delta(
    edges: PlasticCsrEdges,
    start: IndexedPlasticState,
    actual_end: IndexedPlasticState,
    *,
    baseline_u: float,
) -> torch.Tensor:
    """Subtract the no-spike counterfactual from an actual release endpoint."""

    if not torch.equal(start.edge_ids, actual_end.edge_ids):
        raise ValueError("STSP delta endpoints must describe identical edges.")
    if actual_end.time_ms < start.time_ms:
        raise ValueError("STSP delta endpoint precedes its start state.")
    passive = passive_evolve_indexed_state(
        edges,
        start,
        end_time_ms=actual_end.time_ms,
        baseline_u=baseline_u,
    )
    return indexed_next_release(
        edges, actual_end, baseline_u=baseline_u
    ) - indexed_next_release(edges, passive, baseline_u=baseline_u)


__all__ = [
    "IndexedPlasticState",
    "RawIndexedPlasticState",
    "capture_indexed_plastic_state",
    "capture_raw_indexed_plastic_state",
    "edge_sources_for_ids",
    "event_induced_release_delta",
    "indexed_next_release",
    "passive_evolve_indexed_state",
    "replace_indexed_plastic_state",
    "replay_indexed_plastic_state",
    "replay_raw_indexed_plastic_state",
]
