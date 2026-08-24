"""Exact recurrent-network checkpoints and STSP-only interventions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple, Union

import torch

from .nest_equivalent import IafPscExpState
from .scheduler import SparseRecurrentNetwork


@dataclass(frozen=True)
class PlasticStateSnapshot:
    """Complete event-driven state of every plastic edge at one boundary."""

    time_ms: float
    u: torch.Tensor
    x: torch.Tensor
    last_spike_time_ms: torch.Tensor

    @property
    def num_edges(self) -> int:
        return int(self.u.numel())


@dataclass(frozen=True)
class NetworkCheckpoint:
    """All mutable recurrent state required for exact branch replay."""

    step_index: int
    exc_state: IafPscExpState
    inh_state: IafPscExpState
    plastic: PlasticStateSnapshot
    recurrent_excitatory_buffer: torch.Tensor
    recurrent_inhibitory_buffer: torch.Tensor
    recurrent_buffer_cursor: int
    last_dispatch_time_ms: float
    n_neurons: int
    n_plastic_edges: int
    dt_ms: float

    @property
    def time_ms(self) -> float:
        return self.step_index * self.dt_ms


def _storage_device(
    network: SparseRecurrentNetwork,
    storage_device: Optional[Union[str, torch.device]],
) -> torch.device:
    return network.device if storage_device is None else torch.device(storage_device)


def _clone_tensor(tensor: torch.Tensor, target: torch.device) -> torch.Tensor:
    return tensor.detach().to(target).clone()


def _clone_neuron_state(
    state: IafPscExpState, target: torch.device
) -> IafPscExpState:
    return IafPscExpState(
        v_m_relative=_clone_tensor(state.v_m_relative, target),
        i_syn_ex=_clone_tensor(state.i_syn_ex, target),
        i_syn_in=_clone_tensor(state.i_syn_in, target),
        i_0=_clone_tensor(state.i_0, target),
        i_1=_clone_tensor(state.i_1, target),
        refractory_count=_clone_tensor(state.refractory_count, target),
    )


def capture_plastic_state(
    network: SparseRecurrentNetwork,
    *,
    storage_device: Optional[Union[str, torch.device]] = None,
) -> PlasticStateSnapshot:
    target = _storage_device(network, storage_device)
    state = network.scheduler.plastic_state
    return PlasticStateSnapshot(
        time_ms=network.step_index * network.config.dt_ms,
        u=_clone_tensor(state.u, target),
        x=_clone_tensor(state.x, target),
        last_spike_time_ms=_clone_tensor(state.last_spike_time_ms, target),
    )


def capture_network_checkpoint(
    network: SparseRecurrentNetwork,
    *,
    storage_device: Optional[Union[str, torch.device]] = None,
) -> NetworkCheckpoint:
    """Capture a branchable state without mutating the running network."""

    target = _storage_device(network, storage_device)
    delay = network.scheduler.delay_buffer
    return NetworkCheckpoint(
        step_index=network.step_index,
        exc_state=_clone_neuron_state(network.exc_state, target),
        inh_state=_clone_neuron_state(network.inh_state, target),
        plastic=capture_plastic_state(network, storage_device=target),
        recurrent_excitatory_buffer=_clone_tensor(delay.excitatory, target),
        recurrent_inhibitory_buffer=_clone_tensor(delay.inhibitory, target),
        recurrent_buffer_cursor=int(delay.cursor),
        last_dispatch_time_ms=float(network.scheduler._last_dispatch_time_ms),
        n_neurons=network.config.n_neurons,
        n_plastic_edges=network.scheduler.connectivity.plastic.num_edges,
        dt_ms=network.config.dt_ms,
    )


def _restore_neuron_state(
    checkpoint: IafPscExpState, device: torch.device, dtype: torch.dtype
) -> IafPscExpState:
    def restored(tensor: torch.Tensor) -> torch.Tensor:
        target_dtype = dtype if tensor.dtype.is_floating_point else tensor.dtype
        return tensor.to(device=device, dtype=target_dtype).clone()

    return IafPscExpState(
        v_m_relative=restored(checkpoint.v_m_relative),
        i_syn_ex=restored(checkpoint.i_syn_ex),
        i_syn_in=restored(checkpoint.i_syn_in),
        i_0=restored(checkpoint.i_0),
        i_1=restored(checkpoint.i_1),
        refractory_count=restored(checkpoint.refractory_count),
    )


def _neuron_state_to_dict(
    state: IafPscExpState, target: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        "v_m_relative": _clone_tensor(state.v_m_relative, target),
        "i_syn_ex": _clone_tensor(state.i_syn_ex, target),
        "i_syn_in": _clone_tensor(state.i_syn_in, target),
        "i_0": _clone_tensor(state.i_0, target),
        "i_1": _clone_tensor(state.i_1, target),
        "refractory_count": _clone_tensor(state.refractory_count, target),
    }


def _neuron_state_from_dict(payload: Dict[str, torch.Tensor]) -> IafPscExpState:
    return IafPscExpState(
        v_m_relative=payload["v_m_relative"],
        i_syn_ex=payload["i_syn_ex"],
        i_syn_in=payload["i_syn_in"],
        i_0=payload["i_0"],
        i_1=payload["i_1"],
        refractory_count=payload["refractory_count"],
    )


def network_checkpoint_to_dict(
    checkpoint: NetworkCheckpoint,
    *,
    storage_device: Union[str, torch.device] = "cpu",
) -> Dict[str, object]:
    """Serialize a checkpoint with tensors only, for ``weights_only`` loads."""

    target = torch.device(storage_device)
    return {
        "schema_version": 1,
        "step_index": int(checkpoint.step_index),
        "exc_state": _neuron_state_to_dict(checkpoint.exc_state, target),
        "inh_state": _neuron_state_to_dict(checkpoint.inh_state, target),
        "plastic": {
            "time_ms": float(checkpoint.plastic.time_ms),
            "u": _clone_tensor(checkpoint.plastic.u, target),
            "x": _clone_tensor(checkpoint.plastic.x, target),
            "last_spike_time_ms": _clone_tensor(
                checkpoint.plastic.last_spike_time_ms, target
            ),
        },
        "recurrent_excitatory_buffer": _clone_tensor(
            checkpoint.recurrent_excitatory_buffer, target
        ),
        "recurrent_inhibitory_buffer": _clone_tensor(
            checkpoint.recurrent_inhibitory_buffer, target
        ),
        "recurrent_buffer_cursor": int(checkpoint.recurrent_buffer_cursor),
        "last_dispatch_time_ms": float(checkpoint.last_dispatch_time_ms),
        "n_neurons": int(checkpoint.n_neurons),
        "n_plastic_edges": int(checkpoint.n_plastic_edges),
        "dt_ms": float(checkpoint.dt_ms),
    }


def network_checkpoint_from_dict(payload: Dict[str, object]) -> NetworkCheckpoint:
    """Restore a checkpoint saved by ``network_checkpoint_to_dict``."""

    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported network checkpoint schema.")
    plastic = payload["plastic"]
    return NetworkCheckpoint(
        step_index=int(payload["step_index"]),
        exc_state=_neuron_state_from_dict(payload["exc_state"]),
        inh_state=_neuron_state_from_dict(payload["inh_state"]),
        plastic=PlasticStateSnapshot(
            time_ms=float(plastic["time_ms"]),
            u=plastic["u"],
            x=plastic["x"],
            last_spike_time_ms=plastic["last_spike_time_ms"],
        ),
        recurrent_excitatory_buffer=payload["recurrent_excitatory_buffer"],
        recurrent_inhibitory_buffer=payload["recurrent_inhibitory_buffer"],
        recurrent_buffer_cursor=int(payload["recurrent_buffer_cursor"]),
        last_dispatch_time_ms=float(payload["last_dispatch_time_ms"]),
        n_neurons=int(payload["n_neurons"]),
        n_plastic_edges=int(payload["n_plastic_edges"]),
        dt_ms=float(payload["dt_ms"]),
    )


def restore_network_checkpoint(
    network: SparseRecurrentNetwork, checkpoint: NetworkCheckpoint
) -> None:
    """Restore a checkpoint into an identical connectivity instance."""

    if checkpoint.n_neurons != network.config.n_neurons:
        raise ValueError("Checkpoint neuron count does not match the network.")
    if checkpoint.n_plastic_edges != network.scheduler.connectivity.plastic.num_edges:
        raise ValueError("Checkpoint plastic-edge count does not match the network.")
    if not math.isclose(checkpoint.dt_ms, network.config.dt_ms, rel_tol=0.0):
        raise ValueError("Checkpoint dt does not match the network.")
    network.exc_state = _restore_neuron_state(
        checkpoint.exc_state, network.device, network.dtype
    )
    network.inh_state = _restore_neuron_state(
        checkpoint.inh_state, network.device, network.dtype
    )
    network.step_index = checkpoint.step_index
    replace_plastic_state(network, checkpoint.plastic)
    delay = network.scheduler.delay_buffer
    if (
        checkpoint.recurrent_excitatory_buffer.shape != delay.excitatory.shape
        or checkpoint.recurrent_inhibitory_buffer.shape != delay.inhibitory.shape
    ):
        raise ValueError("Checkpoint recurrent delay-buffer shape mismatch.")
    delay.excitatory.copy_(checkpoint.recurrent_excitatory_buffer.to(delay.excitatory))
    delay.inhibitory.copy_(checkpoint.recurrent_inhibitory_buffer.to(delay.inhibitory))
    if not 0 <= checkpoint.recurrent_buffer_cursor < delay.slot_count:
        raise ValueError("Checkpoint recurrent delay-buffer cursor is invalid.")
    delay.cursor = checkpoint.recurrent_buffer_cursor
    network.scheduler._last_dispatch_time_ms = checkpoint.last_dispatch_time_ms
    network.scheduler.clear_transient_instrumentation()


@torch.no_grad()
def replace_plastic_state(
    network: SparseRecurrentNetwork,
    donor: PlasticStateSnapshot,
    *,
    edge_ids: Optional[torch.Tensor] = None,
) -> None:
    """Replace only STSP state while leaving neurons and queued events intact."""

    current_time = network.step_index * network.config.dt_ms
    if not math.isclose(
        donor.time_ms,
        current_time,
        rel_tol=0.0,
        abs_tol=network.config.dt_ms * 1e-6,
    ):
        raise ValueError("Donor STSP state must come from the same time boundary.")
    state = network.scheduler.plastic_state
    if donor.num_edges != state.u.numel():
        raise ValueError("Donor STSP state has the wrong number of edges.")
    if edge_ids is None:
        state.u.copy_(donor.u.to(state.u))
        state.x.copy_(donor.x.to(state.x))
        state.last_spike_time_ms.copy_(
            donor.last_spike_time_ms.to(state.last_spike_time_ms)
        )
        return
    indices = edge_ids.to(device=network.device, dtype=torch.int64)
    if indices.numel() == 0:
        return
    if bool((indices < 0).any().item()) or bool((indices >= state.u.numel()).any().item()):
        raise ValueError("An STSP intervention edge lies outside the plastic graph.")
    state.u.index_copy_(0, indices, donor.u.to(state.u)[indices])
    state.x.index_copy_(0, indices, donor.x.to(state.x)[indices])
    state.last_spike_time_ms.index_copy_(
        0, indices, donor.last_spike_time_ms.to(state.last_spike_time_ms)[indices]
    )


@torch.no_grad()
def reset_plastic_state_to_no_event_baseline(
    network: SparseRecurrentNetwork,
    *,
    edge_ids: Optional[torch.Tensor] = None,
) -> None:
    """Remove trial-evoked STSP while preserving its heterogeneous baseline.

    The reset value is the analytically evolved initial state at the current
    boundary under zero presynaptic events.  Stamping that continuous value at
    the current time prevents subsequent recovery from reintroducing the
    erased trial history.
    """

    time_ms = network.step_index * network.config.dt_ms
    state = network.scheduler.plastic_state
    edges = network.scheduler.connectivity.plastic
    initial_u = edges.initial_u
    initial_x = edges.initial_x
    baseline_u = network.config.stsp_u
    no_event_u = baseline_u + (initial_u - baseline_u) * torch.exp(
        torch.full_like(edges.tau_fac_ms, -time_ms) / edges.tau_fac_ms
    )
    no_event_x = 1.0 + (initial_x - 1.0) * torch.exp(
        torch.full_like(edges.tau_rec_ms, -time_ms) / edges.tau_rec_ms
    )
    if edge_ids is None:
        state.u.copy_(no_event_u)
        state.x.copy_(no_event_x)
        state.last_spike_time_ms.fill_(time_ms)
        return
    indices = edge_ids.to(device=network.device, dtype=torch.int64)
    if indices.numel() == 0:
        return
    if bool((indices < 0).any().item()) or bool((indices >= state.u.numel()).any().item()):
        raise ValueError("An STSP reset edge lies outside the plastic graph.")
    state.u.index_copy_(0, indices, no_event_u[indices])
    state.x.index_copy_(0, indices, no_event_x[indices])
    state.last_spike_time_ms.index_fill_(0, indices, time_ms)


@torch.no_grad()
def continuous_plastic_state(
    network: SparseRecurrentNetwork,
    *,
    time_ms: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return continuous u, x, and hypothetical next-spike release per edge."""

    resolved_time = (
        network.step_index * network.config.dt_ms if time_ms is None else float(time_ms)
    )
    state = network.scheduler.plastic_state
    edges = network.scheduler.connectivity.plastic
    elapsed = resolved_time - state.last_spike_time_ms
    if bool((elapsed < -network.config.dt_ms * 1e-6).any().item()):
        raise ValueError("Requested STSP state precedes an edge's last spike.")
    elapsed = torch.clamp_min(elapsed, 0.0)
    baseline_u = network.config.stsp_u
    u = baseline_u + (state.u - baseline_u) * torch.exp(-elapsed / edges.tau_fac_ms)
    x = 1.0 + (state.x - 1.0) * torch.exp(-elapsed / edges.tau_rec_ms)
    next_u = u + baseline_u * (1.0 - u)
    next_release = edges.weights * next_u * x
    return u, x, next_release


__all__ = [
    "NetworkCheckpoint",
    "PlasticStateSnapshot",
    "capture_network_checkpoint",
    "capture_plastic_state",
    "continuous_plastic_state",
    "network_checkpoint_from_dict",
    "network_checkpoint_to_dict",
    "replace_plastic_state",
    "reset_plastic_state_to_no_event_baseline",
    "restore_network_checkpoint",
]
