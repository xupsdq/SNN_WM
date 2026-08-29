"""Shared in-process Interface for successor boundary replay."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.experiments.common.monitored_dms import (
    build_layer_input_shapes,
    snapshot_boundary_state,
)
from src.experiments.common.ping_common import LAYER_KEYS, prepare_network_state
from src.experiments.common.runtime import seed_everything


FAST_STATE_KEYS = ("v_mem", "g_e", "res", "inh_trace")
STSP_STATE_KEYS = ("u", "x")
Boundary = Mapping[str, Mapping[str, Any]]


def restore_boundary_state(
    net: Any,
    boundary: Boundary,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
    *,
    mode: str,
    device: torch.device,
) -> None:
    """Reset the network, then restore either STSP-only or the full persisted boundary."""
    prepare_network_state(net, *[int(value) for value in layer_input_shapes["layer1"]])
    keys = FAST_STATE_KEYS + STSP_STATE_KEYS if mode == "full_boundary" else STSP_STATE_KEYS
    with torch.no_grad():
        for layer_name in LAYER_KEYS:
            layer = getattr(net, layer_name)
            for state_name in keys:
                if state_name not in boundary[layer_name]:
                    continue
                value = torch.as_tensor(boundary[layer_name][state_name], device=device)
                if state_name == "inh_trace":
                    target = layer.lateral_inh.inh_trace
                elif state_name == "u":
                    target = layer.u_pre
                elif state_name == "x":
                    target = layer.x_pre
                else:
                    target = getattr(layer, state_name)
                target.copy_(value.to(dtype=target.dtype))


def capture_successor_transition(
    ctx: Any,
    *,
    boundary: Boundary,
    input_seq: torch.Tensor,
    current_time: int,
    passive: bool,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Capture one successor while restoring STSP and resetting Layer-3 fast state."""
    return _capture_transition(
        ctx,
        boundary=boundary,
        input_seq=input_seq,
        current_time=current_time,
        passive=passive,
        random_seed=random_seed,
        restore_mode="stsp_only",
        reset_layer3_fast_state=True,
    )


def continue_successor_transition(
    ctx: Any,
    *,
    boundary: Boundary,
    input_seq: torch.Tensor,
    current_time: int,
    passive: bool,
    random_seed: int,
    restore_mode: str,
) -> dict[str, np.ndarray]:
    """Continue from a captured boundary without resetting its restored fast state."""
    return _capture_transition(
        ctx,
        boundary=boundary,
        input_seq=input_seq,
        current_time=current_time,
        passive=passive,
        random_seed=random_seed,
        restore_mode=restore_mode,
        reset_layer3_fast_state=False,
    )


def correct_passive_successor_effects(
    active: Mapping[str, np.ndarray],
    passive: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    active_l3 = np.asarray(active["layer3_ux_post"], dtype=np.float32) - np.asarray(
        active["layer3_ux_pre"], dtype=np.float32
    )
    passive_l3 = np.asarray(passive["layer3_ux_post"], dtype=np.float32) - np.asarray(
        passive["layer3_ux_pre"], dtype=np.float32
    )
    return {
        "early_layer2_event_map": np.asarray(active["early_layer2_event_map"], dtype=np.float32)
        - np.asarray(passive["early_layer2_event_map"], dtype=np.float32),
        "layer3_successor_ux": active_l3 - passive_l3,
    }


def audit_stsp_only_restore(
    ctx: Any,
    boundary: Boundary,
    *,
    input_shape: Sequence[int],
) -> dict[str, bool]:
    batch_size = int(_to_numpy(boundary["layer1"]["u"]).shape[0])
    shapes = build_layer_input_shapes(ctx.net, batch_size, *[int(value) for value in input_shape])
    restore_boundary_state(ctx.net, boundary, shapes, mode="stsp_only", device=ctx.device)
    restored = snapshot_boundary_numpy(ctx.net)
    all_stsp_exact = all(
        _bitwise_equal(boundary[layer][state], restored[layer][state])
        for layer in LAYER_KEYS
        for state in STSP_STATE_KEYS
        if state in boundary[layer]
    )
    fast_state_uniform = all(
        len(restored[layer][state]) <= 1
        or _bitwise_equal(
            restored[layer][state],
            np.repeat(restored[layer][state][:1], len(restored[layer][state]), axis=0),
        )
        for layer in LAYER_KEYS
        for state in FAST_STATE_KEYS
    )
    return {
        "all_stsp_exact": bool(all_stsp_exact),
        "fast_state_uniform": bool(fast_state_uniform),
    }


def snapshot_boundary_numpy(net: Any) -> dict[str, dict[str, np.ndarray]]:
    snapshot = snapshot_boundary_state(net)
    return {
        layer: {state: _to_numpy(value).copy() for state, value in states.items()}
        for layer, states in snapshot.items()
    }


def repeat_boundary(receiver: Boundary, repeats: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        layer: {
            state: np.concatenate([_to_numpy(value)] * int(repeats), axis=0)
            for state, value in states.items()
        }
        for layer, states in receiver.items()
    }


def advance_network_step(
    net: Any,
    input_spikes: torch.Tensor,
    *,
    current_time: int,
    layer2_replay_input: torch.Tensor | None = None,
    layer3_time: int | None = None,
    monitor_layer1: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    s1, monitor = net.layer1.forward_step(
        input_spikes,
        current_time,
        training=False,
        monitor=monitor_layer1,
        stsp_mode="dynamic",
    )
    s1_p = net.pool1(s1.float())
    l2_input = s1_p if layer2_replay_input is None else layer2_replay_input
    s2, _ = net.layer2.forward_step(
        l2_input,
        current_time,
        training=False,
        monitor=False,
        stsp_mode="dynamic",
    )
    s2_p = net.pool2(s2.float())
    s3, _ = net.layer3.forward_step(
        s2_p,
        current_time if layer3_time is None else int(layer3_time),
        training=False,
        monitor=False,
        stsp_mode="dynamic",
    )
    return s1, s1_p, s2, s2_p, s3, monitor


def layer_ux_checkpoint(layer: Any, batch_size: int) -> np.ndarray:
    u = layer.u_pre.detach().reshape(batch_size, -1).cpu().numpy()
    x = layer.x_pre.detach().reshape(batch_size, -1).cpu().numpy()
    return np.concatenate([u, x], axis=1).astype(np.float32, copy=False)


def _capture_transition(
    ctx: Any,
    *,
    boundary: Boundary,
    input_seq: torch.Tensor,
    current_time: int,
    passive: bool,
    random_seed: int,
    restore_mode: str,
    reset_layer3_fast_state: bool,
) -> dict[str, np.ndarray]:
    seed_everything(int(random_seed))
    batch_size, stimulus_steps, channels, height, width = input_seq.shape
    shapes = build_layer_input_shapes(ctx.net, batch_size, channels, height, width)
    restore_boundary_state(ctx.net, boundary, shapes, mode=restore_mode, device=ctx.device)
    ctx.net.layer3.reset_decision_state()
    if reset_layer3_fast_state:
        with torch.no_grad():
            ctx.net.layer3.v_mem.fill_(ctx.net.layer3.V_L)
            ctx.net.layer3.lateral_inh.reset_state(ctx.net.layer3.output_shape)
    l3_pre = layer_ux_checkpoint(ctx.net.layer3, batch_size)
    zero = torch.zeros((batch_size, channels, height, width), dtype=torch.bool, device=ctx.device)
    early_map: torch.Tensor | None = None
    total_steps = int(stimulus_steps + ctx.cfg.fixed_b_post_steps)
    early_cutoff = min(int(ctx.cfg.fixed_b_early_window_steps), int(stimulus_steps))
    with torch.no_grad():
        for local_step in range(total_steps):
            external = input_seq[:, local_step] if local_step < stimulus_steps and not passive else zero
            _, _, s2, _, _, _ = advance_network_step(
                ctx.net,
                external,
                current_time=int(current_time) + local_step,
                layer2_replay_input=None,
                layer3_time=local_step,
                monitor_layer1=False,
            )
            if local_step < early_cutoff:
                value = s2.detach().to(torch.float32)
                early_map = value.clone() if early_map is None else early_map + value
    if early_map is None:
        raise RuntimeError("Early Layer-2 capture window was empty")
    return {
        "early_layer2_event_map": early_map.cpu().numpy().astype(np.float32, copy=False),
        "layer3_ux_pre": l3_pre,
        "layer3_ux_post": layer_ux_checkpoint(ctx.net.layer3, batch_size),
    }


def prepare_layer2_stsp_transplant(
    receiver: Boundary,
    donor_indices: np.ndarray,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, slice],
    dict[str, bool],
]:
    """Return native, Layer-2 u/x swap and own-sham rows in that fixed order."""
    donor_indices = np.asarray(donor_indices, dtype=np.int64)
    row_count = len(next(iter(next(iter(receiver.values())).values())))
    slices = {
        "native": slice(0, row_count),
        "layer2_swap": slice(row_count, 2 * row_count),
        "own_sham": slice(2 * row_count, 3 * row_count),
    }
    combined: dict[str, dict[str, np.ndarray]] = {}
    for layer, states in receiver.items():
        combined[layer] = {}
        for state, value in states.items():
            array = _to_numpy(value)
            swap = array[donor_indices] if layer == "layer2" and state in {"u", "x"} else array
            combined[layer][state] = np.concatenate((array, swap, array), axis=0)

    audit = {
        "layer2_only_mix_exact": all(
            _bitwise_equal(
                combined[layer][state][slices["layer2_swap"]],
                _to_numpy(value)[donor_indices]
                if layer == "layer2" and state in STSP_STATE_KEYS
                else value,
            )
            for layer, states in receiver.items()
            for state, value in states.items()
        ),
        "own_sham_boundary_exact": all(
            _bitwise_equal(combined[layer][state][slices["own_sham"]], value)
            for layer, states in receiver.items()
            for state, value in states.items()
        ),
    }
    return combined, slices, audit


def _bitwise_equal(first: np.ndarray | torch.Tensor, second: np.ndarray | torch.Tensor) -> bool:
    left = np.ascontiguousarray(_to_numpy(first))
    right = np.ascontiguousarray(_to_numpy(second))
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


__all__ = [
    "FAST_STATE_KEYS",
    "STSP_STATE_KEYS",
    "advance_network_step",
    "audit_stsp_only_restore",
    "capture_successor_transition",
    "continue_successor_transition",
    "correct_passive_successor_effects",
    "layer_ux_checkpoint",
    "prepare_layer2_stsp_transplant",
    "repeat_boundary",
    "restore_boundary_state",
    "snapshot_boundary_numpy",
]
