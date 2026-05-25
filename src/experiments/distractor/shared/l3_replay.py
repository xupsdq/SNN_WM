from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.experiments.common.ping_common import prepare_network_state


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


@dataclass(frozen=True)
class L3RegionSpec:
    region_id: int
    row_index: int
    col_index: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int


@dataclass(frozen=True)
class L3TraceCaptureResult:
    grouped_voltage: np.ndarray
    readout_snapshot: torch.Tensor
    probe_s2p_trace: torch.Tensor
    probe_onset_snapshot: Layer3ReplaySnapshot
    first_fire_t_probe: np.ndarray
    prediction_probe: np.ndarray
    readout_step: int


def center_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    return arr - np.mean(arr, axis=-1, keepdims=True)


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


def run_dms_with_l3_trace_capture(
    net,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    *,
    delay_steps: int,
    stsp_mode: str,
    readout_step: int,
    phase_reset: bool = True,
) -> L3TraceCaptureResult:
    if sample_spikes.ndim != 5 or probe_spikes.ndim != 5:
        raise ValueError("sample_spikes and probe_spikes must have shape [B, T, C, H, W]")
    batch_size, _, channels, height, width = sample_spikes.shape
    if probe_spikes.shape[0] != batch_size:
        raise ValueError("sample_spikes and probe_spikes must share batch size")
    device = sample_spikes.device
    zero_input = torch.zeros((batch_size, channels, height, width), dtype=sample_spikes.dtype, device=device)
    probe_s2p_chunks: list[torch.Tensor] = []
    readout_snapshot = None
    current_time = 0

    prepare_network_state(net, batch_size, channels, height, width)

    def step_full_network(input_t: torch.Tensor, *, capture_probe_trace: bool, l3_time: int) -> None:
        nonlocal current_time, readout_snapshot
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())
        if capture_probe_trace:
            probe_s2p_chunks.append(s2_p.detach().cpu().to(torch.float32))
        _, monitor_data = net.layer3.forward_step(
            s2_p,
            l3_time,
            training=False,
            monitor=bool(capture_probe_trace and int(l3_time) == int(readout_step)),
            stsp_mode=stsp_mode,
        )
        if capture_probe_trace and int(l3_time) == int(readout_step):
            readout_snapshot = monitor_data["v_mem_snapshot"].detach().cpu().to(torch.float32)
        current_time += 1

    with torch.no_grad():
        for t_step in range(int(sample_spikes.shape[1])):
            step_full_network(sample_spikes[:, t_step, ...], capture_probe_trace=False, l3_time=current_time)
        for _ in range(int(delay_steps)):
            step_full_network(zero_input, capture_probe_trace=False, l3_time=current_time)
        net.layer3.reset_decision_state()
        if phase_reset:
            net.layer3.v_mem.fill_(net.layer3.V_L)
            net.layer3.lateral_inh.reset_state(net.layer3.output_shape)
        probe_onset_snapshot = _snapshot_layer3_for_replay(net, readout_step=readout_step)
        for t_step in range(int(probe_spikes.shape[1])):
            probe_l3_time = int(t_step) if phase_reset else int(current_time)
            step_full_network(probe_spikes[:, t_step, ...], capture_probe_trace=True, l3_time=probe_l3_time)

    if readout_snapshot is None:
        raise RuntimeError("Probe readout snapshot was not captured")
    grouped_voltage = _extract_grouped_voltage_vector(net, readout_snapshot)
    flat_times = net.layer3.firing_times.detach().cpu()
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    prediction_probe = (min_indices // net.layer3.neurons_per_class).long()
    prediction_probe[~has_fired] = -1
    first_fire_t_probe = min_times.clone()
    first_fire_t_probe[~has_fired] = -1
    return L3TraceCaptureResult(
        grouped_voltage=grouped_voltage,
        readout_snapshot=readout_snapshot,
        probe_s2p_trace=torch.stack(probe_s2p_chunks, dim=1),
        probe_onset_snapshot=probe_onset_snapshot,
        first_fire_t_probe=first_fire_t_probe.numpy().astype(np.int64, copy=False),
        prediction_probe=prediction_probe.numpy().astype(np.int64, copy=False),
        readout_step=int(readout_step),
    )


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


def make_l3_region_masks(height: int, width: int, mask_mode: str = "1x1") -> list[L3RegionSpec]:
    mode = str(mask_mode).strip().lower()
    if mode not in {"1x1", "2x2"}:
        raise ValueError(f"Unsupported --l3-mask-mode: {mask_mode}")
    block = 1 if mode == "1x1" else 2
    row_positions = list(range(0, max(1, height - block + 1), block))
    col_positions = list(range(0, max(1, width - block + 1), block))
    if not row_positions or row_positions[-1] != max(0, height - block):
        row_positions.append(max(0, height - block))
    if not col_positions or col_positions[-1] != max(0, width - block):
        col_positions.append(max(0, width - block))
    row_positions = sorted(dict.fromkeys(int(v) for v in row_positions))
    col_positions = sorted(dict.fromkeys(int(v) for v in col_positions))
    regions: list[L3RegionSpec] = []
    region_id = 0
    for row_index, row_start in enumerate(row_positions):
        row_end = min(height, row_start + block)
        for col_index, col_start in enumerate(col_positions):
            col_end = min(width, col_start + block)
            regions.append(
                L3RegionSpec(
                    region_id=int(region_id),
                    row_index=int(row_index),
                    col_index=int(col_index),
                    row_start=int(row_start),
                    row_end=int(row_end),
                    col_start=int(col_start),
                    col_end=int(col_end),
                )
            )
            region_id += 1
    return regions


def _center_class_effects(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    return arr - arr.mean(axis=1, keepdims=True)


def _apply_zero_region(trace_batch: torch.Tensor, region: L3RegionSpec) -> None:
    trace_batch[:, :, :, region.row_start:region.row_end, region.col_start:region.col_end] = 0.0


def _apply_replacement_region(trace_batch: torch.Tensor, donor_trace: torch.Tensor, region: L3RegionSpec) -> None:
    trace_batch[:, :, :, region.row_start:region.row_end, region.col_start:region.col_end] = donor_trace[
        None, :, :, region.row_start:region.row_end, region.col_start:region.col_end
    ]


def _run_region_replays(
    net,
    base_trace: torch.Tensor,
    probe_onset_snapshot: Layer3ReplaySnapshot,
    regions: Sequence[L3RegionSpec],
    *,
    stsp_mode: str,
    batch_size: int,
    donor_trace: torch.Tensor | None = None,
) -> np.ndarray:
    if not regions:
        return np.zeros((0, net.layer3.num_classes), dtype=np.float64)
    base = base_trace.to(torch.float32)
    donor = None if donor_trace is None else donor_trace.to(torch.float32)
    outputs: list[np.ndarray] = []
    for start in range(0, len(regions), int(batch_size)):
        chunk = list(regions[start:start + int(batch_size)])
        modified = base.unsqueeze(0).repeat(len(chunk), 1, 1, 1, 1)
        for local_idx, region in enumerate(chunk):
            if donor is None:
                _apply_zero_region(modified[local_idx:local_idx + 1], region)
            else:
                _apply_replacement_region(modified[local_idx:local_idx + 1], donor, region)
        replay_out = replay_layer3_probe_phase(
            net=net,
            probe_onset_snapshot=probe_onset_snapshot,
            modified_probe_s2p_trace=modified,
            stsp_mode=stsp_mode,
        )
        outputs.append(np.asarray(replay_out["grouped_voltage"], dtype=np.float64))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, net.layer3.num_classes), dtype=np.float64)


def run_l3_deletion_analysis_for_pair(
    net,
    dynamic_capture: L3TraceCaptureResult,
    static_capture: L3TraceCaptureResult,
    regions: Sequence[L3RegionSpec],
    *,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    v_dyn = np.asarray(dynamic_capture.grouped_voltage[0], dtype=np.float64)
    v_sta = np.asarray(static_capture.grouped_voltage[0], dtype=np.float64)
    v_drop_dyn = _run_region_replays(
        net=net,
        base_trace=dynamic_capture.probe_s2p_trace[0],
        probe_onset_snapshot=dynamic_capture.probe_onset_snapshot,
        regions=regions,
        stsp_mode="dynamic",
        batch_size=batch_size,
        donor_trace=None,
    )
    v_drop_sta = _run_region_replays(
        net=net,
        base_trace=static_capture.probe_s2p_trace[0],
        probe_onset_snapshot=static_capture.probe_onset_snapshot,
        regions=regions,
        stsp_mode="static_frozen",
        batch_size=batch_size,
        donor_trace=None,
    )
    d_dyn = v_dyn[None, :] - v_drop_dyn
    d_sta = v_sta[None, :] - v_drop_sta
    return {
        "D_dyn": d_dyn,
        "D_sta": d_sta,
        "E_dyn": _center_class_effects(d_dyn),
        "E_sta": _center_class_effects(d_sta),
    }


def run_l3_replacement_analysis_for_pair(
    net,
    dynamic_capture: L3TraceCaptureResult,
    static_capture: L3TraceCaptureResult,
    regions: Sequence[L3RegionSpec],
    *,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    v_dyn = np.asarray(dynamic_capture.grouped_voltage[0], dtype=np.float64)
    v_sta = np.asarray(static_capture.grouped_voltage[0], dtype=np.float64)
    v_sta_to_dyn = _run_region_replays(
        net=net,
        base_trace=static_capture.probe_s2p_trace[0],
        probe_onset_snapshot=static_capture.probe_onset_snapshot,
        regions=regions,
        stsp_mode="static_frozen",
        batch_size=batch_size,
        donor_trace=dynamic_capture.probe_s2p_trace[0],
    )
    v_dyn_to_sta = _run_region_replays(
        net=net,
        base_trace=dynamic_capture.probe_s2p_trace[0],
        probe_onset_snapshot=dynamic_capture.probe_onset_snapshot,
        regions=regions,
        stsp_mode="dynamic",
        batch_size=batch_size,
        donor_trace=static_capture.probe_s2p_trace[0],
    )
    r_plus = v_sta_to_dyn - v_sta[None, :]
    r_minus = v_dyn[None, :] - v_dyn_to_sta
    return {
        "R_plus": r_plus,
        "R_minus": r_minus,
        "R_plus_tilde": _center_class_effects(r_plus),
        "R_minus_tilde": _center_class_effects(r_minus),
    }


def vector_similarity(pred_vec: np.ndarray, target_vec: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred_vec, dtype=np.float64).reshape(-1)
    target = np.asarray(target_vec, dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(pred) * np.linalg.norm(target)), 1e-12)
    cosine = float(np.dot(pred, target) / denom)
    pearson = float("nan")
    spearman = float("nan")
    if pred.size >= 2 and np.std(pred) > 1e-12 and np.std(target) > 1e-12:
        pearson = float(stats.pearsonr(pred, target).statistic)
        spearman = float(stats.spearmanr(pred, target).statistic)
    return {"cosine": cosine, "pearson": pearson, "spearman": spearman}


def nanargmax_with_default(values: np.ndarray, default: int = -1) -> int:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return int(default)
    return int(np.nanargmax(arr))


def summarize_l3_mechanism_results(df_results: pd.DataFrame) -> Dict[str, object]:
    mean_bias_magnitude = float(df_results["bias_magnitude"].mean()) if len(df_results) else 0.0
    mean_dynamic_vs_static_deletion_contrast = (
        float(df_results["deletion_dynamic_minus_static_kstar"].mean(skipna=True)) if len(df_results) else float("nan")
    )
    mean_static_to_dynamic_push = (
        float(df_results["replacement_push_kstar"].mean(skipna=True)) if len(df_results) else float("nan")
    )
    mean_dynamic_to_static_pullback = (
        float(df_results["replacement_pullback_kstar"].mean(skipna=True)) if len(df_results) else float("nan")
    )
    mean_reconstruction_cosine_plus = (
        float(df_results["reconstruction_cosine_plus"].mean(skipna=True)) if len(df_results) else float("nan")
    )
    mean_reconstruction_cosine_minus = (
        float(df_results["reconstruction_cosine_minus"].mean(skipna=True)) if len(df_results) else float("nan")
    )
    direction_match_rate_plus = float(df_results["direction_match_plus"].mean(skipna=True)) if len(df_results) else float("nan")
    direction_match_rate_minus = float(df_results["direction_match_minus"].mean(skipna=True)) if len(df_results) else float("nan")
    summary = {
        "mean_reconstruction_cosine_plus": mean_reconstruction_cosine_plus,
        "mean_reconstruction_cosine_minus": mean_reconstruction_cosine_minus,
        "direction_match_rate_plus": direction_match_rate_plus,
        "direction_match_rate_minus": direction_match_rate_minus,
        "overall": {
            "num_pairs": int(len(df_results)),
            "mean_bias_magnitude": mean_bias_magnitude,
            "mean_dynamic_vs_static_deletion_contrast": mean_dynamic_vs_static_deletion_contrast,
            "mean_static_to_dynamic_push": mean_static_to_dynamic_push,
            "mean_dynamic_to_static_pullback": mean_dynamic_to_static_pullback,
            "mean_reconstruction_cosine_plus": mean_reconstruction_cosine_plus,
            "mean_reconstruction_cosine_minus": mean_reconstruction_cosine_minus,
            "direction_match_rate_plus": direction_match_rate_plus,
            "direction_match_rate_minus": direction_match_rate_minus,
        },
        "by_bias_direction": [],
    }
    if len(df_results):
        grouped = (
            df_results.groupby("bias_direction", sort=True)
            .agg(
                count=("pair_id", "size"),
                mean_bias_magnitude=("bias_magnitude", "mean"),
                mean_push=("replacement_push_kstar", "mean"),
                mean_pullback=("replacement_pullback_kstar", "mean"),
                mean_deletion_contrast=("deletion_dynamic_minus_static_kstar", "mean"),
            )
            .reset_index()
        )
        summary["by_bias_direction"] = grouped.to_dict("records")
    return summary


__all__ = [
    "L3RegionSpec",
    "L3TraceCaptureResult",
    "Layer3ReplaySnapshot",
    "_snapshot_layer3_for_replay",
    "center_vector",
    "make_l3_region_masks",
    "nanargmax_with_default",
    "replay_layer3_probe_phase",
    "run_dms_with_l3_trace_capture",
    "run_l3_deletion_analysis_for_pair",
    "run_l3_replacement_analysis_for_pair",
    "summarize_l3_mechanism_results",
    "vector_similarity",
]
