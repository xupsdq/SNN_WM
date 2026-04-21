from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.distractor.shared.pair_sampling import (
    build_dataset_arrays,
    build_pair_specs,
)
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.voltage_readout import resolve_readout_step
from src.plotting.common.io import (
    COLOR_DYNAMIC,
    COLOR_STATIC,
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
)
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    ALPHA_SCATTER,
    CMAP_DIVERGING,
    CMAP_IMAGE_GRAY,
    COLOR_ACCENT_BLUE,
    COLOR_ACCENT_ORANGE_ALT,
    COLOR_ACCENT_RED,
    COLOR_ACCENT_TEAL,
    GRID_ALPHA,
    LINE_WIDTH_GUIDE,
    apply_standard_legend,
    case_grid_figsize,
)
from src.plotting.experiments.l3_accumulator_mechanism_experiment_plot_lib import (
    render_figures_from_results as render_retained_plot_only_figures,
    write_plot_bundle_manifest,
)

DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_OUTPUT_DIR = "results/l3_accumulator_mechanism_experiment"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_DELAY_MS = 500.0
DEFAULT_PROBE_MS = 100.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_PROBES = 20
DEFAULT_SAMPLES_PER_PROBE = 12
DEFAULT_MAX_PAIRS = 240
DEFAULT_NUM_SIM_BINS = 5
DEFAULT_L3_MASK_MODE = "1x1"
DEFAULT_TEMPORAL_POOL = "mean"
DEFAULT_SAVE_CASE_COUNT = 4


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


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
class L3TraceCaptureResult:
    grouped_voltage: np.ndarray
    readout_snapshot: torch.Tensor
    probe_s2p_trace: torch.Tensor
    probe_onset_snapshot: Layer3ReplaySnapshot
    first_fire_t_probe: np.ndarray
    prediction_probe: np.ndarray
    readout_step: int


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        return None
    return scalar


def _to_json_ready(value):
    if isinstance(value, dict):
        return {str(key): _to_json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _safe_float(value)
    return value


def _save_json(payload: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json_ready(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_dataset(dataset_root: str, split: str):
    train_loader, _, test_loader = build_mnist_skeleton_loader(
        root=dataset_root,
        batch_size=1,
        input_size=28,
        num_workers=0,
    )
    split_name = str(split).strip().lower()
    if split_name == "train":
        return train_loader.dataset
    if split_name == "test":
        return test_loader.dataset
    raise ValueError(f"Unsupported split: {split}")


def _center_vector(vector: np.ndarray) -> np.ndarray:
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
    probe_s2p_chunks: List[torch.Tensor] = []
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


def make_l3_region_masks(height: int, width: int, mask_mode: str = "1x1") -> List[L3RegionSpec]:
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
    regions: List[L3RegionSpec] = []
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


def _region_values_to_grid(regions: Sequence[L3RegionSpec], values: Sequence[float]) -> np.ndarray:
    if not regions:
        return np.zeros((0, 0), dtype=np.float64)
    n_rows = max(region.row_index for region in regions) + 1
    n_cols = max(region.col_index for region in regions) + 1
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    for region, value in zip(regions, values):
        grid[int(region.row_index), int(region.col_index)] = float(value)
    return grid


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
    outputs: List[np.ndarray] = []
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


def _vector_similarity(pred_vec: np.ndarray, target_vec: np.ndarray) -> Dict[str, float]:
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


def _nanargmax_with_default(values: np.ndarray, default: int = -1) -> int:
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


def _select_case_pairs(df_results: pd.DataFrame, save_case_count: int) -> pd.DataFrame:
    if df_results.empty:
        return df_results.copy()
    selected_pair_ids: List[int] = []
    used_probes: set[int] = set()
    ordered = df_results.sort_values(
        ["bias_magnitude", "reconstruction_cosine_plus", "replacement_push_kstar"],
        ascending=[False, False, False],
        kind="stable",
    )
    for row in ordered.itertuples(index=False):
        if len(selected_pair_ids) >= int(save_case_count):
            break
        probe_id = int(row.probe_id)
        pair_id = int(row.pair_id)
        if probe_id in used_probes:
            continue
        selected_pair_ids.append(pair_id)
        used_probes.add(probe_id)
    if len(selected_pair_ids) < int(save_case_count):
        for pair_id in ordered["pair_id"].astype(int).tolist():
            if pair_id not in selected_pair_ids:
                selected_pair_ids.append(pair_id)
            if len(selected_pair_ids) >= int(save_case_count):
                break
    return (
        df_results[df_results["pair_id"].isin(selected_pair_ids)]
        .copy()
        .sort_values(["bias_magnitude", "pair_id"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )


def _plot_heatmap(ax, grid: np.ndarray, *, title: str, cmap: str = CMAP_DIVERGING) -> None:
    finite = np.asarray(grid, dtype=np.float64)
    vmax = np.nanmax(np.abs(finite)) if np.isfinite(finite).any() else 1.0
    vmax = max(vmax, 1e-6)
    image = ax.imshow(grid, cmap=cmap, aspect="equal", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_vector_bar(ax, delta_v: np.ndarray, *, title: str, compare: np.ndarray | None = None, label_compare: str = "recon") -> None:
    classes = np.arange(len(delta_v), dtype=np.int64)
    ax.bar(classes - 0.2, delta_v, width=0.4, color=COLOR_ACCENT_BLUE, label="delta_V")
    if compare is not None:
        ax.bar(classes + 0.2, compare, width=0.4, color=COLOR_ACCENT_ORANGE_ALT, alpha=ALPHA_BAR, label=label_compare)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Centered voltage")
    ax.grid(alpha=GRID_ALPHA, axis="y")
    if compare is not None:
        apply_standard_legend(ax)


def plot_case_deletion_maps(
    df_cases: pd.DataFrame,
    images: torch.Tensor,
    *,
    pair_id_to_index: Mapping[int, int],
    delta_v: np.ndarray,
    e_sta: np.ndarray,
    e_dyn: np.ndarray,
    regions: Sequence[L3RegionSpec],
) -> plt.Figure:
    apply_publication_style()
    n_cases = max(1, len(df_cases))
    fig, axes = plt.subplots(n_cases, 5, figsize=case_grid_figsize(n_cases, width=17.0, row_height=3.4), squeeze=False)
    for row_index, row in enumerate(df_cases.itertuples(index=False)):
        pair_index = int(pair_id_to_index[int(row.pair_id)])
        k_star = int(row.bias_direction)
        probe_image = images[int(row.probe_id), 0].numpy().astype(np.float64, copy=False)
        axes[row_index, 0].imshow(probe_image, cmap=CMAP_IMAGE_GRAY)
        axes[row_index, 0].set_title(f"Probe {int(row.probe_label)}")
        axes[row_index, 0].axis("off")
        _plot_vector_bar(axes[row_index, 1], delta_v[pair_index], title=f"delta_V (pair {int(row.pair_id)})")
        _plot_heatmap(
            axes[row_index, 2],
            _region_values_to_grid(regions, e_sta[pair_index, :, k_star]),
            title=f"E_sta[:, {k_star}]",
        )
        _plot_heatmap(
            axes[row_index, 3],
            _region_values_to_grid(regions, e_dyn[pair_index, :, k_star]),
            title=f"E_dyn[:, {k_star}]",
        )
        _plot_heatmap(
            axes[row_index, 4],
            _region_values_to_grid(regions, e_dyn[pair_index, :, k_star] - e_sta[pair_index, :, k_star]),
            title=f"(E_dyn - E_sta)[:, {k_star}]",
        )
    fig.tight_layout()
    return fig


def plot_case_replacement_maps(
    df_cases: pd.DataFrame,
    *,
    pair_id_to_index: Mapping[int, int],
    delta_v: np.ndarray,
    delta_hat_plus: np.ndarray,
    delta_hat_minus: np.ndarray,
    r_plus_tilde: np.ndarray,
    r_minus_tilde: np.ndarray,
    regions: Sequence[L3RegionSpec],
) -> plt.Figure:
    apply_publication_style()
    n_cases = max(1, len(df_cases))
    fig, axes = plt.subplots(n_cases, 4, figsize=case_grid_figsize(n_cases, width=15.5, row_height=3.4), squeeze=False)
    for row_index, row in enumerate(df_cases.itertuples(index=False)):
        pair_index = int(pair_id_to_index[int(row.pair_id)])
        k_star = int(row.bias_direction)
        _plot_heatmap(
            axes[row_index, 0],
            _region_values_to_grid(regions, r_plus_tilde[pair_index, :, k_star]),
            title=f"R_plus_tilde[:, {k_star}]",
        )
        _plot_heatmap(
            axes[row_index, 1],
            _region_values_to_grid(regions, r_minus_tilde[pair_index, :, k_star]),
            title=f"R_minus_tilde[:, {k_star}]",
        )
        _plot_vector_bar(
            axes[row_index, 2],
            delta_v[pair_index],
            title="delta_V vs Delta_hat_plus",
            compare=delta_hat_plus[pair_index],
            label_compare="Delta_hat_plus",
        )
        _plot_vector_bar(
            axes[row_index, 3],
            delta_v[pair_index],
            title="delta_V vs Delta_hat_minus",
            compare=delta_hat_minus[pair_index],
            label_compare="Delta_hat_minus",
        )
    fig.tight_layout()
    return fig


def plot_summary_metrics(df_results: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    metric_names = [
        ("deletion_dynamic_minus_static_kstar", "Deletion\ncontrast"),
        ("replacement_push_kstar", "Push"),
        ("replacement_pullback_kstar", "Pullback"),
    ]
    means = [float(df_results[name].mean(skipna=True)) if len(df_results) else float("nan") for name, _ in metric_names]
    sems = []
    for name, _ in metric_names:
        values = df_results[name].to_numpy(dtype=np.float64, copy=False) if len(df_results) else np.asarray([], dtype=np.float64)
        finite = values[np.isfinite(values)]
        sem = float(np.std(finite, ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0
        sems.append(sem)
    axes[0].bar(np.arange(len(metric_names)), means, yerr=sems, color=[COLOR_ACCENT_BLUE, COLOR_ACCENT_ORANGE_ALT, COLOR_ACCENT_TEAL])
    axes[0].set_xticks(np.arange(len(metric_names)))
    axes[0].set_xticklabels([label for _, label in metric_names])
    axes[0].set_ylabel("Mean centered effect")
    axes[0].set_title("Deletion / replacement")
    axes[0].grid(alpha=GRID_ALPHA, axis="y")

    _plot_reconstruction_cosine_axis(axes[1], df_results)
    _plot_argmax_reconstruction_axis(axes[2], df_results)
    fig.tight_layout()
    return fig


def _plot_reconstruction_cosine_axis(ax, df_results: pd.DataFrame) -> None:
    box_data = [
        df_results["reconstruction_cosine_plus"].to_numpy(dtype=np.float64, copy=False),
        df_results["reconstruction_cosine_minus"].to_numpy(dtype=np.float64, copy=False),
    ]
    ax.boxplot(box_data, labels=["plus", "minus"], showmeans=True)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Reconstruction Cosine with Final Bias Vector")
    ax.grid(alpha=GRID_ALPHA, axis="y")


def _plot_argmax_reconstruction_axis(ax, df_results: pd.DataFrame) -> None:
    direction_rates = [
        float(df_results["direction_match_plus"].mean(skipna=True)) if len(df_results) else float("nan"),
        float(df_results["direction_match_minus"].mean(skipna=True)) if len(df_results) else float("nan"),
    ]
    ax.bar([0, 1], direction_rates, color=[COLOR_ACCENT_RED, COLOR_ACCENT_TEAL])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["plus", "minus"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Direction match rate")
    ax.set_title("Argmax Direction Match")
    ax.grid(alpha=GRID_ALPHA, axis="y")


def plot_reconstruction_cosine(df_results: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(PUBLICATION_TWO_COLUMN_FIGSIZE[0] * 0.55, PUBLICATION_TWO_COLUMN_FIGSIZE[1]))
    _plot_reconstruction_cosine_axis(ax, df_results)
    fig.tight_layout()
    return fig


def plot_argmax_reconstruction(df_results: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(PUBLICATION_TWO_COLUMN_FIGSIZE[0] * 0.55, PUBLICATION_TWO_COLUMN_FIGSIZE[1]))
    _plot_argmax_reconstruction_axis(ax, df_results)
    fig.tight_layout()
    return fig


def plot_pair_level_scatter(df_results: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    axes[0].scatter(
        df_results["top_push_value_kstar"].to_numpy(dtype=np.float64, copy=False),
        df_results["bias_magnitude"].to_numpy(dtype=np.float64, copy=False),
        alpha=ALPHA_SCATTER,
        color=COLOR_DYNAMIC,
    )
    axes[0].set_xlabel("Top replacement push")
    axes[0].set_ylabel("Bias magnitude")
    axes[0].set_title("Push vs bias magnitude")
    axes[0].grid(alpha=GRID_ALPHA)

    axes[1].scatter(
        df_results["reconstruction_cosine_plus"].to_numpy(dtype=np.float64, copy=False),
        df_results["bias_magnitude"].to_numpy(dtype=np.float64, copy=False),
        alpha=ALPHA_SCATTER,
        color=COLOR_STATIC,
    )
    axes[1].set_xlabel("Reconstruction cosine")
    axes[1].set_ylabel("Bias magnitude")
    axes[1].set_title("Reconstruction vs bias magnitude")
    axes[1].grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="L3 accumulator mechanism readout experiment.")
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--delay-ms", type=float, default=DEFAULT_DELAY_MS)
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--samples-per-probe", type=int, default=DEFAULT_SAMPLES_PER_PROBE)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--l3-mask-mode", type=str, default=DEFAULT_L3_MASK_MODE, choices=["1x1", "2x2"])
    parser.add_argument("--temporal-pool", type=str, default=DEFAULT_TEMPORAL_POOL, choices=["sum", "mean"])
    parser.add_argument("--save-case-count", type=int, default=DEFAULT_SAVE_CASE_COUNT)
    parser.add_argument("--skip-deletion", action="store_true")
    parser.add_argument("--skip-replacement", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.max_probes) <= 0:
        raise ValueError("--max-probes must be positive.")
    if int(args.samples_per_probe) <= 0:
        raise ValueError("--samples-per-probe must be positive.")
    if int(args.max_pairs) <= 0:
        raise ValueError("--max-pairs must be positive.")
    if int(args.save_case_count) <= 0:
        raise ValueError("--save-case-count must be positive.")

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=1.0 * ms, sample_ms=float(args.sample_ms), probe_ms=float(args.probe_ms))
    if spec.sample_steps <= 0 or spec.probe_steps <= 0:
        raise ValueError("sample/probe durations must resolve to positive steps.")
    delay_steps = int(round((float(args.delay_ms) * ms) / spec.dt))
    if delay_steps < 0:
        raise ValueError("--delay-ms must be non-negative.")

    layout = prepare_result_layout(args.output_dir)
    result_root = layout.root
    data_dir = layout.data_dir
    metrics_dir = layout.metrics_dir
    figures_dir = layout.figure_dir
    logs_dir = layout.log_dir
    meta_dir = layout.meta_dir

    dataset = _load_dataset(dataset_root=args.dataset_root, split=args.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    num_classes = len(set(int(label) for label in labels.tolist()))
    class_index = build_class_index(dataset, num_classes=num_classes)
    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(float(args.sample_ms), float(args.probe_ms)),
    )
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )
    df_pairs = build_pair_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=int(args.max_probes),
        samples_per_probe=int(args.samples_per_probe),
        num_bins=int(DEFAULT_NUM_SIM_BINS),
        max_pairs=int(args.max_pairs),
        seed=int(args.seed),
    )

    spike_cache: Dict[tuple[int, int], torch.Tensor] = {}

    def get_encoded_image_spikes(image_id: int, steps: int) -> torch.Tensor:
        key = (int(image_id), int(steps))
        cached = spike_cache.get(key)
        if cached is not None:
            return cached
        batch = images[int(image_id)].unsqueeze(0).to(device=device, dtype=torch.float32)
        encoded = encode_images(encoder, batch, int(steps))[0].detach().cpu().to(torch.float32).contiguous()
        spike_cache[key] = encoded
        return encoded

    pair_rows: List[Dict[str, object]] = []
    v_dyn_rows: List[np.ndarray] = []
    v_sta_rows: List[np.ndarray] = []
    delta_v_rows: List[np.ndarray] = []
    delta_hat_plus_rows: List[np.ndarray] = []
    delta_hat_minus_rows: List[np.ndarray] = []
    e_sta_rows: List[np.ndarray] = []
    e_dyn_rows: List[np.ndarray] = []
    d_sta_rows: List[np.ndarray] = []
    d_dyn_rows: List[np.ndarray] = []
    r_plus_rows: List[np.ndarray] = []
    r_minus_rows: List[np.ndarray] = []
    r_plus_tilde_rows: List[np.ndarray] = []
    r_minus_tilde_rows: List[np.ndarray] = []
    probe_trace_dyn_rows: List[np.ndarray] = []
    probe_trace_sta_rows: List[np.ndarray] = []
    snapshot_v_mem_dyn_rows: List[np.ndarray] = []
    snapshot_v_mem_sta_rows: List[np.ndarray] = []
    snapshot_g_e_dyn_rows: List[np.ndarray] = []
    snapshot_g_e_sta_rows: List[np.ndarray] = []
    snapshot_res_dyn_rows: List[np.ndarray] = []
    snapshot_res_sta_rows: List[np.ndarray] = []
    snapshot_inh_dyn_rows: List[np.ndarray] = []
    snapshot_inh_sta_rows: List[np.ndarray] = []
    snapshot_u_dyn_rows: List[np.ndarray] = []
    snapshot_u_sta_rows: List[np.ndarray] = []
    snapshot_x_dyn_rows: List[np.ndarray] = []
    snapshot_x_sta_rows: List[np.ndarray] = []
    snapshot_input_dyn_rows: List[np.ndarray] = []
    snapshot_input_sta_rows: List[np.ndarray] = []
    snapshot_elig_dyn_rows: List[np.ndarray] = []
    snapshot_elig_sta_rows: List[np.ndarray] = []
    snapshot_fire_dyn_rows: List[np.ndarray] = []
    snapshot_fire_sta_rows: List[np.ndarray] = []

    regions: List[L3RegionSpec] | None = None
    for row in tqdm(df_pairs.itertuples(index=False), total=len(df_pairs), desc="L3AccumulatorPairs"):
        sample_spikes = get_encoded_image_spikes(int(row.sample_id), spec.sample_steps).unsqueeze(0).to(device=device)
        probe_spikes = get_encoded_image_spikes(int(row.probe_id), spec.probe_steps).unsqueeze(0).to(device=device)
        dynamic_capture = run_dms_with_l3_trace_capture(
            net=net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=delay_steps,
            stsp_mode="dynamic",
            readout_step=readout_step,
            phase_reset=True,
        )
        static_capture = run_dms_with_l3_trace_capture(
            net=net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=delay_steps,
            stsp_mode="static_frozen",
            readout_step=readout_step,
            phase_reset=True,
        )
        if regions is None:
            trace_height = int(dynamic_capture.probe_s2p_trace.shape[-2])
            trace_width = int(dynamic_capture.probe_s2p_trace.shape[-1])
            regions = make_l3_region_masks(trace_height, trace_width, mask_mode=args.l3_mask_mode)

        v_dyn = np.asarray(dynamic_capture.grouped_voltage[0], dtype=np.float64)
        v_sta = np.asarray(static_capture.grouped_voltage[0], dtype=np.float64)
        delta_v = _center_vector(v_dyn) - _center_vector(v_sta)
        bias_magnitude = float(np.linalg.norm(delta_v, ord=2))
        bias_direction = int(np.argmax(delta_v))

        if args.skip_deletion:
            nan_matrix = np.full((len(regions), num_classes), np.nan, dtype=np.float64)
            deletion = {"D_dyn": nan_matrix.copy(), "D_sta": nan_matrix.copy(), "E_dyn": nan_matrix.copy(), "E_sta": nan_matrix.copy()}
        else:
            deletion = run_l3_deletion_analysis_for_pair(
                net=net,
                dynamic_capture=dynamic_capture,
                static_capture=static_capture,
                regions=regions,
                batch_size=int(args.batch_size),
            )

        if args.skip_replacement:
            nan_matrix = np.full((len(regions), num_classes), np.nan, dtype=np.float64)
            replacement = {
                "R_plus": nan_matrix.copy(),
                "R_minus": nan_matrix.copy(),
                "R_plus_tilde": nan_matrix.copy(),
                "R_minus_tilde": nan_matrix.copy(),
            }
        else:
            replacement = run_l3_replacement_analysis_for_pair(
                net=net,
                dynamic_capture=dynamic_capture,
                static_capture=static_capture,
                regions=regions,
                batch_size=int(args.batch_size),
            )

        delta_hat_plus = np.nansum(np.asarray(replacement["R_plus_tilde"], dtype=np.float64), axis=0)
        delta_hat_minus = np.nansum(np.asarray(replacement["R_minus_tilde"], dtype=np.float64), axis=0)
        sim_plus = _vector_similarity(delta_hat_plus, delta_v)
        sim_minus = _vector_similarity(delta_hat_minus, delta_v)

        e_sta_k = np.asarray(deletion["E_sta"][:, bias_direction], dtype=np.float64)
        e_dyn_k = np.asarray(deletion["E_dyn"][:, bias_direction], dtype=np.float64)
        r_plus_k = np.asarray(replacement["R_plus_tilde"][:, bias_direction], dtype=np.float64)
        r_minus_k = np.asarray(replacement["R_minus_tilde"][:, bias_direction], dtype=np.float64)

        pair_rows.append(
            {
                "pair_id": int(row.pair_id),
                "probe_id": int(row.probe_id),
                "sample_id": int(row.sample_id),
                "probe_label": int(row.probe_label),
                "sample_label": int(row.sample_label),
                "similarity_public_or_initial": float(row.similarity_public_or_initial),
                "similarity_bin": str(row.similarity_bin),
                "similarity_bin_index": int(row.similarity_bin_index),
                "bias_magnitude": bias_magnitude,
                "bias_direction": int(bias_direction),
                "pred_dynamic": int(dynamic_capture.prediction_probe[0]),
                "pred_static": int(static_capture.prediction_probe[0]),
                "first_fire_t_dynamic": int(dynamic_capture.first_fire_t_probe[0]),
                "first_fire_t_static": int(static_capture.first_fire_t_probe[0]),
                "top_dynamic_deletion_region_for_k_star": _nanargmax_with_default(e_dyn_k),
                "top_static_deletion_region_for_k_star": _nanargmax_with_default(e_sta_k),
                "deletion_dynamic_minus_static_kstar": float(np.nanmean(e_dyn_k - e_sta_k)),
                "top_static_to_dynamic_push_region_for_k_star": _nanargmax_with_default(r_plus_k),
                "top_dynamic_to_static_pullback_region_for_k_star": _nanargmax_with_default(r_minus_k),
                "top_push_value_kstar": _safe_float(np.nanmax(r_plus_k)) if np.isfinite(r_plus_k).any() else None,
                "top_pullback_value_kstar": _safe_float(np.nanmax(r_minus_k)) if np.isfinite(r_minus_k).any() else None,
                "replacement_push_kstar": float(np.nanmean(r_plus_k)),
                "replacement_pullback_kstar": float(np.nanmean(r_minus_k)),
                "reconstruction_cosine_plus": float(sim_plus["cosine"]),
                "reconstruction_pearson_plus": float(sim_plus["pearson"]),
                "reconstruction_spearman_plus": float(sim_plus["spearman"]),
                "reconstruction_cosine_minus": float(sim_minus["cosine"]),
                "reconstruction_pearson_minus": float(sim_minus["pearson"]),
                "reconstruction_spearman_minus": float(sim_minus["spearman"]),
                "direction_match_plus": int(np.argmax(delta_hat_plus) == bias_direction) if np.isfinite(delta_hat_plus).any() else 0,
                "direction_match_minus": int(np.argmax(delta_hat_minus) == bias_direction) if np.isfinite(delta_hat_minus).any() else 0,
                "readout_step": int(readout_step),
            }
        )

        v_dyn_rows.append(v_dyn)
        v_sta_rows.append(v_sta)
        delta_v_rows.append(delta_v)
        delta_hat_plus_rows.append(delta_hat_plus)
        delta_hat_minus_rows.append(delta_hat_minus)
        e_sta_rows.append(np.asarray(deletion["E_sta"], dtype=np.float64))
        e_dyn_rows.append(np.asarray(deletion["E_dyn"], dtype=np.float64))
        d_sta_rows.append(np.asarray(deletion["D_sta"], dtype=np.float64))
        d_dyn_rows.append(np.asarray(deletion["D_dyn"], dtype=np.float64))
        r_plus_rows.append(np.asarray(replacement["R_plus"], dtype=np.float64))
        r_minus_rows.append(np.asarray(replacement["R_minus"], dtype=np.float64))
        r_plus_tilde_rows.append(np.asarray(replacement["R_plus_tilde"], dtype=np.float64))
        r_minus_tilde_rows.append(np.asarray(replacement["R_minus_tilde"], dtype=np.float64))
        probe_trace_dyn_rows.append(dynamic_capture.probe_s2p_trace[0].numpy().astype(np.float32, copy=False))
        probe_trace_sta_rows.append(static_capture.probe_s2p_trace[0].numpy().astype(np.float32, copy=False))
        snapshot_v_mem_dyn_rows.append(dynamic_capture.probe_onset_snapshot.v_mem.numpy().astype(np.float32, copy=False))
        snapshot_v_mem_sta_rows.append(static_capture.probe_onset_snapshot.v_mem.numpy().astype(np.float32, copy=False))
        snapshot_g_e_dyn_rows.append(dynamic_capture.probe_onset_snapshot.g_e.numpy().astype(np.float32, copy=False))
        snapshot_g_e_sta_rows.append(static_capture.probe_onset_snapshot.g_e.numpy().astype(np.float32, copy=False))
        snapshot_res_dyn_rows.append(dynamic_capture.probe_onset_snapshot.res.numpy())
        snapshot_res_sta_rows.append(static_capture.probe_onset_snapshot.res.numpy())
        snapshot_inh_dyn_rows.append(dynamic_capture.probe_onset_snapshot.inh_trace.numpy().astype(np.float32, copy=False))
        snapshot_inh_sta_rows.append(static_capture.probe_onset_snapshot.inh_trace.numpy().astype(np.float32, copy=False))
        snapshot_u_dyn_rows.append(dynamic_capture.probe_onset_snapshot.u_pre.numpy().astype(np.float32, copy=False) if dynamic_capture.probe_onset_snapshot.u_pre is not None else np.zeros((1,), dtype=np.float32))
        snapshot_u_sta_rows.append(static_capture.probe_onset_snapshot.u_pre.numpy().astype(np.float32, copy=False) if static_capture.probe_onset_snapshot.u_pre is not None else np.zeros((1,), dtype=np.float32))
        snapshot_x_dyn_rows.append(dynamic_capture.probe_onset_snapshot.x_pre.numpy().astype(np.float32, copy=False) if dynamic_capture.probe_onset_snapshot.x_pre is not None else np.zeros((1,), dtype=np.float32))
        snapshot_x_sta_rows.append(static_capture.probe_onset_snapshot.x_pre.numpy().astype(np.float32, copy=False) if static_capture.probe_onset_snapshot.x_pre is not None else np.zeros((1,), dtype=np.float32))
        snapshot_input_dyn_rows.append(dynamic_capture.probe_onset_snapshot.input_trace.numpy().astype(np.float32, copy=False) if dynamic_capture.probe_onset_snapshot.input_trace is not None else np.zeros((1,), dtype=np.float32))
        snapshot_input_sta_rows.append(static_capture.probe_onset_snapshot.input_trace.numpy().astype(np.float32, copy=False) if static_capture.probe_onset_snapshot.input_trace is not None else np.zeros((1,), dtype=np.float32))
        snapshot_elig_dyn_rows.append(dynamic_capture.probe_onset_snapshot.eligibility_trace.numpy().astype(np.float32, copy=False) if dynamic_capture.probe_onset_snapshot.eligibility_trace is not None else np.zeros((1,), dtype=np.float32))
        snapshot_elig_sta_rows.append(static_capture.probe_onset_snapshot.eligibility_trace.numpy().astype(np.float32, copy=False) if static_capture.probe_onset_snapshot.eligibility_trace is not None else np.zeros((1,), dtype=np.float32))
        snapshot_fire_dyn_rows.append(dynamic_capture.probe_onset_snapshot.firing_times.numpy().astype(np.float32, copy=False) if dynamic_capture.probe_onset_snapshot.firing_times is not None else np.zeros((1,), dtype=np.float32))
        snapshot_fire_sta_rows.append(static_capture.probe_onset_snapshot.firing_times.numpy().astype(np.float32, copy=False) if static_capture.probe_onset_snapshot.firing_times is not None else np.zeros((1,), dtype=np.float32))

    df_results = pd.DataFrame(pair_rows).sort_values(["pair_id"], kind="stable").reset_index(drop=True)
    df_cases = _select_case_pairs(df_results, save_case_count=int(args.save_case_count))
    pair_id_to_index = {int(pair_id): idx for idx, pair_id in enumerate(df_results["pair_id"].astype(int).tolist())}

    v_dyn_arr = np.stack(v_dyn_rows, axis=0) if v_dyn_rows else np.zeros((0, num_classes), dtype=np.float64)
    v_sta_arr = np.stack(v_sta_rows, axis=0) if v_sta_rows else np.zeros((0, num_classes), dtype=np.float64)
    delta_v_arr = np.stack(delta_v_rows, axis=0) if delta_v_rows else np.zeros((0, num_classes), dtype=np.float64)
    delta_hat_plus_arr = np.stack(delta_hat_plus_rows, axis=0) if delta_hat_plus_rows else np.zeros((0, num_classes), dtype=np.float64)
    delta_hat_minus_arr = np.stack(delta_hat_minus_rows, axis=0) if delta_hat_minus_rows else np.zeros((0, num_classes), dtype=np.float64)
    e_sta_arr = np.stack(e_sta_rows, axis=0) if e_sta_rows else np.zeros((0, 0, num_classes), dtype=np.float64)
    e_dyn_arr = np.stack(e_dyn_rows, axis=0) if e_dyn_rows else np.zeros((0, 0, num_classes), dtype=np.float64)
    d_sta_arr = np.stack(d_sta_rows, axis=0) if d_sta_rows else np.zeros((0, 0, num_classes), dtype=np.float64)
    d_dyn_arr = np.stack(d_dyn_rows, axis=0) if d_dyn_rows else np.zeros((0, 0, num_classes), dtype=np.float64)
    r_plus_arr = np.stack(r_plus_rows, axis=0) if r_plus_rows else np.zeros((0, 0, num_classes), dtype=np.float64)
    r_minus_arr = np.stack(r_minus_rows, axis=0) if r_minus_rows else np.zeros((0, 0, num_classes), dtype=np.float64)
    r_plus_tilde_arr = np.stack(r_plus_tilde_rows, axis=0) if r_plus_tilde_rows else np.zeros((0, 0, num_classes), dtype=np.float64)
    r_minus_tilde_arr = np.stack(r_minus_tilde_rows, axis=0) if r_minus_tilde_rows else np.zeros((0, 0, num_classes), dtype=np.float64)
    summary_metrics = summarize_l3_mechanism_results(df_results)
    summary_metrics["case_pair_ids"] = df_cases["pair_id"].astype(int).tolist()
    summary_metrics["assumptions"] = {
        "repo_directory": "src/experiments",
        "readout_step_rule": "decision_offset_minus_one_with_clipping",
        "readout_step": int(readout_step),
        "l3_region_definition": str(args.l3_mask_mode),
        "temporal_pool_summary": str(args.temporal_pool),
        "full_pair_trace_and_snapshot_save": True,
    }

    pair_csv = save_tidy_csv(df_results, data_dir / "pair_results.csv", sort_by=["pair_id"])
    pair_vectors_npz = data_dir / "pair_vectors.npz"
    np.savez_compressed(
        pair_vectors_npz,
        pair_id=df_results["pair_id"].to_numpy(dtype=np.int64, copy=False),
        V_dyn=v_dyn_arr,
        V_sta=v_sta_arr,
        delta_V=delta_v_arr,
        Delta_hat_plus=delta_hat_plus_arr,
        Delta_hat_minus=delta_hat_minus_arr,
    )
    pair_deletion_npz = data_dir / "pair_l3_deletion_maps.npz"
    np.savez_compressed(
        pair_deletion_npz,
        pair_id=df_results["pair_id"].to_numpy(dtype=np.int64, copy=False),
        E_sta=e_sta_arr,
        E_dyn=e_dyn_arr,
        D_sta=d_sta_arr,
        D_dyn=d_dyn_arr,
    )
    pair_replacement_npz = data_dir / "pair_l3_replacement_maps.npz"
    np.savez_compressed(
        pair_replacement_npz,
        pair_id=df_results["pair_id"].to_numpy(dtype=np.int64, copy=False),
        R_plus=r_plus_arr,
        R_minus=r_minus_arr,
        R_plus_tilde=r_plus_tilde_arr,
        R_minus_tilde=r_minus_tilde_arr,
    )
    pair_trace_npz = data_dir / "pair_traces_or_snapshots.npz"
    np.savez_compressed(
        pair_trace_npz,
        pair_id=df_results["pair_id"].to_numpy(dtype=np.int64, copy=False),
        probe_s2p_trace_dyn=np.stack(probe_trace_dyn_rows, axis=0),
        probe_s2p_trace_sta=np.stack(probe_trace_sta_rows, axis=0),
        snapshot_v_mem_dyn=np.stack(snapshot_v_mem_dyn_rows, axis=0),
        snapshot_v_mem_sta=np.stack(snapshot_v_mem_sta_rows, axis=0),
        snapshot_g_e_dyn=np.stack(snapshot_g_e_dyn_rows, axis=0),
        snapshot_g_e_sta=np.stack(snapshot_g_e_sta_rows, axis=0),
        snapshot_res_dyn=np.stack(snapshot_res_dyn_rows, axis=0),
        snapshot_res_sta=np.stack(snapshot_res_sta_rows, axis=0),
        snapshot_inh_trace_dyn=np.stack(snapshot_inh_dyn_rows, axis=0),
        snapshot_inh_trace_sta=np.stack(snapshot_inh_sta_rows, axis=0),
        snapshot_u_dyn=np.stack(snapshot_u_dyn_rows, axis=0),
        snapshot_u_sta=np.stack(snapshot_u_sta_rows, axis=0),
        snapshot_x_dyn=np.stack(snapshot_x_dyn_rows, axis=0),
        snapshot_x_sta=np.stack(snapshot_x_sta_rows, axis=0),
        snapshot_input_trace_dyn=np.stack(snapshot_input_dyn_rows, axis=0),
        snapshot_input_trace_sta=np.stack(snapshot_input_sta_rows, axis=0),
        snapshot_eligibility_trace_dyn=np.stack(snapshot_elig_dyn_rows, axis=0),
        snapshot_eligibility_trace_sta=np.stack(snapshot_elig_sta_rows, axis=0),
        snapshot_firing_times_dyn=np.stack(snapshot_fire_dyn_rows, axis=0),
        snapshot_firing_times_sta=np.stack(snapshot_fire_sta_rows, axis=0),
        region_row_start=np.asarray([region.row_start for region in regions], dtype=np.int64),
        region_row_end=np.asarray([region.row_end for region in regions], dtype=np.int64),
        region_col_start=np.asarray([region.col_start for region in regions], dtype=np.int64),
        region_col_end=np.asarray([region.col_end for region in regions], dtype=np.int64),
        readout_step=np.asarray([int(readout_step)], dtype=np.int64),
    )
    summary_json = _save_json(summary_metrics, metrics_dir / "summary_metrics.json")
    write_plot_bundle_manifest(meta_dir)

    empty_paths = {"png": "", "pdf": "", "svg": ""}
    fig1_paths = empty_paths.copy()
    fig2_paths = empty_paths.copy()
    reconstruction_cosine_paths = empty_paths.copy()
    argmax_reconstruction_paths = empty_paths.copy()
    fig4_paths = empty_paths.copy()
    if not bool(args.skip_figures):
        fig1 = plot_case_deletion_maps(
            df_cases=df_cases,
            images=images,
            pair_id_to_index=pair_id_to_index,
            delta_v=delta_v_arr,
            e_sta=e_sta_arr,
            e_dyn=e_dyn_arr,
            regions=regions,
        )
        fig1_paths = save_figure_all_formats(fig1, figures_dir / "figure_1_case_deletion_maps")
        plt.close(fig1)

        fig2 = plot_case_replacement_maps(
            df_cases=df_cases,
            pair_id_to_index=pair_id_to_index,
            delta_v=delta_v_arr,
            delta_hat_plus=delta_hat_plus_arr,
            delta_hat_minus=delta_hat_minus_arr,
            r_plus_tilde=r_plus_tilde_arr,
            r_minus_tilde=r_minus_tilde_arr,
            regions=regions,
        )
        fig2_paths = save_figure_all_formats(fig2, figures_dir / "figure_2_case_replacement_maps")
        plt.close(fig2)

        retained_figure_paths = render_retained_plot_only_figures(df_results=df_results, figures_dir=figures_dir)
        reconstruction_cosine_paths = retained_figure_paths["reconstruction_cosine"]
        argmax_reconstruction_paths = retained_figure_paths["argmax_reconstruction"]
        fig4_paths = retained_figure_paths["figure_4_pair_level_scatter"]

    run_config_payload = {
            "model_path": str(Path(args.model_path).resolve()),
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "split": str(args.split),
            "output_dir": str(result_root.resolve()),
            "device": str(device),
            "seed": int(args.seed),
            "config_argument": args.config,
            "sample_ms": float(args.sample_ms),
            "delay_ms": float(args.delay_ms),
            "probe_ms": float(args.probe_ms),
            "batch_size": int(args.batch_size),
            "max_probes": int(args.max_probes),
            "samples_per_probe": int(args.samples_per_probe),
            "max_pairs": int(args.max_pairs),
            "l3_mask_mode": str(args.l3_mask_mode),
            "temporal_pool": str(args.temporal_pool),
            "save_case_count": int(args.save_case_count),
            "skip_deletion": bool(args.skip_deletion),
            "skip_replacement": bool(args.skip_replacement),
            "skip_figures": bool(args.skip_figures),
            "readout_step": int(readout_step),
            "assumptions": summary_metrics["assumptions"],
            "outputs": {
                "pair_results_csv": str(Path(pair_csv).resolve()),
                "pair_vectors_npz": str(pair_vectors_npz.resolve()),
                "pair_l3_deletion_maps_npz": str(pair_deletion_npz.resolve()),
                "pair_l3_replacement_maps_npz": str(pair_replacement_npz.resolve()),
                "pair_traces_or_snapshots_npz": str(pair_trace_npz.resolve()),
                "summary_metrics_json": str(summary_json.resolve()),
                "figure_1_png": fig1_paths["png"],
                "figure_2_png": fig2_paths["png"],
                "figure_4_png": fig4_paths["png"],
                "reconstruction_cosine_png": reconstruction_cosine_paths["png"],
                "reconstruction_cosine_pdf": reconstruction_cosine_paths["pdf"],
                "reconstruction_cosine_svg": reconstruction_cosine_paths["svg"],
                "argmax_reconstruction_png": argmax_reconstruction_paths["png"],
                "argmax_reconstruction_pdf": argmax_reconstruction_paths["pdf"],
                "argmax_reconstruction_svg": argmax_reconstruction_paths["svg"],
            },
    }
    run_config_path = save_run_config(run_config_payload, result_root)
    _save_json(run_config_payload, meta_dir / "run_config.snapshot.json")
    summary_payload = {
            "experiment": "l3_accumulator_mechanism_experiment",
            "pair_count": int(len(df_results)),
            "mean_bias_magnitude": float(df_results["bias_magnitude"].mean()) if len(df_results) else None,
            "artifact_summary_metrics_json": str(summary_json.resolve()),
            "run_config_json": str(Path(run_config_path).resolve()),
            "primary_figure_1": "reconstruction_cosine",
            "primary_figure_2": "argmax_reconstruction",
            "summary_text": "L3/s2p-based reconstruction recovers both the continuous bias-vector similarity and the dominant argmax direction of the final decision bias.",
    }
    summary_path = save_summary_json(summary_payload, result_root)
    save_summary_json(summary_payload, metrics_dir, filename="summary.json")
    _save_json(
        {
            "experiment": "l3_accumulator_mechanism_experiment",
            "pair_count": int(len(df_results)),
            "mean_bias_magnitude": float(df_results["bias_magnitude"].mean()) if len(df_results) else None,
            "summary_metrics_json": str(summary_json.resolve()),
        },
        metrics_dir / "main_metrics.json",
    )
    # TODO: Large pair trace/snapshot arrays remain under data/ because they are intermediate analysis payloads, not summary metrics.
    run_log_path = save_log_lines(
        [
            "experiment=l3_accumulator_mechanism_experiment",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"pairs={len(df_results)}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_path.resolve()}",
            f"reconstruction_cosine_png={reconstruction_cosine_paths['png']}",
            f"argmax_reconstruction_png={argmax_reconstruction_paths['png']}",
        ],
        logs_dir,
    )

    print("\n=== L3 Accumulator Mechanism Experiment Summary ===")
    print(f"Pairs analysed: {len(df_results)}")
    print(f"Mean bias magnitude: {float(df_results['bias_magnitude'].mean()):.4f}" if len(df_results) else "Mean bias magnitude: n/a")
    print(f"Mean push cosine: {float(df_results['reconstruction_cosine_plus'].mean(skipna=True)):.4f}" if len(df_results) else "Mean push cosine: n/a")
    print(f"Saved outputs under: {result_root.resolve()}")
    print(f"Saved: {pair_csv}")
    print(f"Saved: {pair_vectors_npz}")
    print(f"Saved: {pair_deletion_npz}")
    print(f"Saved: {pair_replacement_npz}")
    print(f"Saved: {pair_trace_npz}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {run_config_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {run_log_path}")


if __name__ == "__main__":
    main()
