from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass(frozen=True)
class Layer3ReplaySnapshot:
    v_mem: torch.Tensor
    g_e: torch.Tensor
    res: torch.Tensor
    inh_trace: torch.Tensor
    u_pre: torch.Tensor | None
    x_pre: torch.Tensor | None
    input_trace: torch.Tensor | None
    eligibility_trace: torch.Tensor | None
    firing_times: torch.Tensor | None
    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    readout_step: int


def _extract_grouped_voltage_vector(net, voltage_snapshot: torch.Tensor) -> np.ndarray:
    grouped = net.layer3.get_grouped_voltage(voltage_snapshot.to(torch.float32))
    return grouped.mean(dim=-1).detach().cpu().numpy().astype(np.float64, copy=False)


def _snapshot_layer3_for_replay(net, readout_step: int) -> Layer3ReplaySnapshot:
    layer = net.layer3
    return Layer3ReplaySnapshot(
        v_mem=layer.v_mem.detach().cpu().to(torch.float32).clone(),
        g_e=layer.g_e.detach().cpu().to(torch.float32).clone(),
        res=layer.res.detach().cpu().clone(),
        inh_trace=layer.lateral_inh.inh_trace.detach().cpu().to(torch.float32).clone(),
        u_pre=None if getattr(layer, "u_pre", None) is None else layer.u_pre.detach().cpu().to(torch.float32).clone(),
        x_pre=None if getattr(layer, "x_pre", None) is None else layer.x_pre.detach().cpu().to(torch.float32).clone(),
        input_trace=None if getattr(layer, "input_trace", None) is None else layer.input_trace.detach().cpu().to(torch.float32).clone(),
        eligibility_trace=None if getattr(layer, "eligibility_trace", None) is None else layer.eligibility_trace.detach().cpu().to(torch.float32).clone(),
        firing_times=None if getattr(layer, "firing_times", None) is None else layer.firing_times.detach().cpu().to(torch.float32).clone(),
        input_shape=tuple(int(v) for v in layer.input_trace.shape) if getattr(layer, "input_trace", None) is not None else tuple(int(v) for v in (1, layer.in_channels, 1, 1)),
        output_shape=tuple(int(v) for v in layer.output_shape),
        readout_step=int(readout_step),
    )


def _repeat_tensor_for_batch(tensor: torch.Tensor | None, batch_size: int, device: torch.device):
    if tensor is None:
        return None
    out = tensor.to(device=device)
    if out.shape[0] == int(batch_size):
        return out.clone()
    if out.shape[0] == 1:
        reps = [int(batch_size)] + [1] * (out.ndim - 1)
        return out.repeat(*reps).clone()
    raise ValueError(f"Cannot expand snapshot batch dimension from {out.shape[0]} to {batch_size}")


def _restore_layer3_probe_onset_snapshot(net, snapshot: Layer3ReplaySnapshot, batch_size: int, device: torch.device) -> None:
    layer = net.layer3
    expected_input_shape = (int(batch_size), int(snapshot.input_shape[1]), int(snapshot.input_shape[2]), int(snapshot.input_shape[3]))
    layer.reset_state(expected_input_shape)
    with torch.no_grad():
        layer.v_mem.copy_(_repeat_tensor_for_batch(snapshot.v_mem, batch_size, device=device))
        layer.g_e.copy_(_repeat_tensor_for_batch(snapshot.g_e, batch_size, device=device))
        layer.res.copy_(_repeat_tensor_for_batch(snapshot.res, batch_size, device=device))
        layer.lateral_inh.inh_trace.copy_(_repeat_tensor_for_batch(snapshot.inh_trace, batch_size, device=device))
        if snapshot.u_pre is not None and layer.u_pre is not None:
            layer.u_pre.copy_(_repeat_tensor_for_batch(snapshot.u_pre, batch_size, device=device))
        if snapshot.x_pre is not None and layer.x_pre is not None:
            layer.x_pre.copy_(_repeat_tensor_for_batch(snapshot.x_pre, batch_size, device=device))
        if snapshot.input_trace is not None and layer.input_trace is not None:
            layer.input_trace.copy_(_repeat_tensor_for_batch(snapshot.input_trace, batch_size, device=device))
        if snapshot.eligibility_trace is not None and layer.eligibility_trace is not None:
            layer.eligibility_trace.copy_(_repeat_tensor_for_batch(snapshot.eligibility_trace, batch_size, device=device))
        if snapshot.firing_times is not None and layer.firing_times is not None:
            layer.firing_times.copy_(_repeat_tensor_for_batch(snapshot.firing_times, batch_size, device=device))


def replay_layer3_probe_phase(
    net,
    probe_onset_snapshot: Layer3ReplaySnapshot,
    modified_probe_s2p_trace: torch.Tensor,
    *,
    stsp_mode: str,
) -> Dict[str, object]:
    trace = modified_probe_s2p_trace
    if trace.ndim == 4:
        trace = trace.unsqueeze(0)
    if trace.ndim != 5:
        raise ValueError("modified_probe_s2p_trace must have shape [T, C, H, W] or [B, T, C, H, W]")
    batch_size, probe_steps, channels, height, width = trace.shape
    trace = trace.to(device=next(net.parameters()).device, dtype=torch.float32)
    _restore_layer3_probe_onset_snapshot(net, probe_onset_snapshot, batch_size=batch_size, device=trace.device)
    readout_snapshot = None
    with torch.no_grad():
        for t_step in range(int(probe_steps)):
            _, monitor_data = net.layer3.forward_step(
                trace[:, t_step, ...],
                int(t_step),
                training=False,
                monitor=bool(int(t_step) == int(probe_onset_snapshot.readout_step)),
                stsp_mode=stsp_mode,
            )
            if int(t_step) == int(probe_onset_snapshot.readout_step):
                readout_snapshot = monitor_data["v_mem_snapshot"].detach().cpu().to(torch.float32)
    if readout_snapshot is None:
        raise RuntimeError("Replay readout snapshot was not captured")
    grouped_voltage = _extract_grouped_voltage_vector(net, readout_snapshot)
    flat_times = net.layer3.firing_times.detach().cpu()
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    prediction_probe = (min_indices // net.layer3.neurons_per_class).long()
    prediction_probe[~has_fired] = -1
    first_fire_t_probe = min_times.clone()
    first_fire_t_probe[~has_fired] = -1
    return {
        "grouped_voltage": grouped_voltage,
        "readout_snapshot": readout_snapshot,
        "prediction_probe": prediction_probe.numpy().astype(np.int64, copy=False),
        "first_fire_t_probe": first_fire_t_probe.numpy().astype(np.int64, copy=False),
        "trace_shape": (int(batch_size), int(probe_steps), int(channels), int(height), int(width)),
    }


__all__ = [
    "Layer3ReplaySnapshot",
    "_snapshot_layer3_for_replay",
    "replay_layer3_probe_phase",
]
