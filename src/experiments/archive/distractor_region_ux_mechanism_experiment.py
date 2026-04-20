from __future__ import annotations

"""
Supplementary distractor-region ux support analysis.

This script now serves as a supporting analysis provider for Fig.5 rather than
the primary backbone. It preserves the triplet/region helpers, probe-region
bundles, and Layer1 regional support diagnostics, but its outputs should be
interpreted as supplementary evidence only.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.units import ms
from src.core.network import SDNN_Network
from src.experiments.common.dataset import build_dataset_arrays
from src.experiments.common.distractor_triplets import (
    build_triplet_specs,
    load_mnist_dataset,
    prepare_triplet_spike_batch,
)
from src.experiments.common.distractor_utils import (
    apply_input_mask_to_spike_batch,
    dilate_mask,
    foreground_mask,
)
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.voltage_readout import resolve_readout_step
from src.plotting.common.io import (
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
)
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    ALPHA_FILL,
    ALPHA_SCATTER_LIGHT,
    DISTRACTOR_REGION_CONDITION_COLORS,
    GRID_ALPHA,
    LINE_WIDTH_REFERENCE,
    apply_standard_legend,
)

DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_OUTPUT_DIR = "results/distractor_region_ux_mechanism_experiment"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_DELAY1_MS = 400.0
DEFAULT_DISTRACTOR_MS = 200.0
DEFAULT_DELAY2_MS = 400.0
DEFAULT_PROBE_MS = 60.0
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_PROBES = 100
DEFAULT_SAMPLES_PER_PROBE = 20
DEFAULT_MAX_TRIPLETS = 5000
DEFAULT_NUM_SIM_BINS = 5
DEFAULT_FOREGROUND_THRESHOLD = 0.0
DEFAULT_DILATION_RADIUS = 1
DEFAULT_WINNER_WINDOW_FRAC = 0.4
DEFAULT_TIE_THRESHOLD = 0.02
DEFAULT_REDISTRIBUTION_FRACTION = 0.35
EPS = 1e-12
SMOKE_NOTE = "smoke experiment should be run in torch_env"

LAYER_KEYS: tuple[str, ...] = ("layer1", "layer2", "layer3")
MAIN_REGION_ORDER: tuple[str, ...] = ("sample_only", "distractor_only", "shared")
BASELINE_CONDITION = "baseline_intact"
REFERENCE_SAMPLE_CONDITION = "sample_reference"
REFERENCE_DISTRACTOR_CONDITION = "distractor_reference"


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay1_ms: float
    distractor_ms: float
    delay2_ms: float
    probe_ms: float
    phase_reset: bool = True


@dataclass(frozen=True)
class ProbeRegionBundle:
    probe_region_masks: dict[str, np.ndarray]
    sample_phase_masks: dict[str, np.ndarray]
    distractor_phase_masks: dict[str, np.ndarray]
    layer_region_masks: dict[str, dict[str, np.ndarray]]
    metadata: dict[str, object]


@dataclass
class RolloutCapture:
    grouped_voltage: np.ndarray
    probe_grouped_voltage_trace: np.ndarray
    probe_l1_trace: torch.Tensor
    probe_l2_trace: torch.Tensor
    probe_l3_trace: torch.Tensor
    readout_step: int
    prediction_probe: np.ndarray
    first_fire_t_probe: np.ndarray
    preprobe_states: dict[str, dict[str, torch.Tensor]]
    intervention_record: dict[str, object]
    note: str = ""


LAYER1_PANEL_FILENAMES = {
    "panel_a": "supp_fig5_layer1_panel_a_triplet_region_definition",
    "panel_b": "supp_fig5_layer1_panel_b_observed_mean_ux_by_region",
    "panel_c": "supp_fig5_layer1_panel_c_observed_excess_and_mass_by_region",
    "panel_d": "supp_fig5_layer1_panel_d_predicted_vs_observed",
    "panel_e": "supp_fig5_layer1_panel_e_rank_probability",
    "panel_f": "supp_fig5_layer1_panel_f_layer1_support_vs_winner",
}
CONDITION_COLORS = {
    **dict(DISTRACTOR_REGION_CONDITION_COLORS),
    BASELINE_CONDITION: DISTRACTOR_REGION_CONDITION_COLORS.get("full_dynamic", "#4C78A8"),
    "sample_only": DISTRACTOR_REGION_CONDITION_COLORS.get("only_SP", "#E45756"),
    "distractor_only": DISTRACTOR_REGION_CONDITION_COLORS.get("only_DP", "#F58518"),
    "shared": DISTRACTOR_REGION_CONDITION_COLORS.get("only_SDP", "#2F4B7C"),
}


@property
def _sample_steps(self: ExperimentSpec) -> int:
    return int(round((self.sample_ms * ms) / self.dt))


@property
def _delay1_steps(self: ExperimentSpec) -> int:
    return int(round((self.delay1_ms * ms) / self.dt))


@property
def _distractor_steps(self: ExperimentSpec) -> int:
    return int(round((self.distractor_ms * ms) / self.dt))


@property
def _delay2_steps(self: ExperimentSpec) -> int:
    return int(round((self.delay2_ms * ms) / self.dt))


@property
def _probe_steps(self: ExperimentSpec) -> int:
    return int(round((self.probe_ms * ms) / self.dt))


ExperimentSpec.sample_steps = _sample_steps
ExperimentSpec.delay1_steps = _delay1_steps
ExperimentSpec.distractor_steps = _distractor_steps
ExperimentSpec.delay2_steps = _delay2_steps
ExperimentSpec.probe_steps = _probe_steps


def _sem(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _validate_positive(name: str, value: int | float, *, allow_zero: bool = False) -> None:
    scalar = float(value)
    if allow_zero:
        if scalar < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        return
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive.")


def center_vector(x: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr - float(arr.mean())


def safe_cosine(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor, eps: float = EPS) -> float:
    aa = center_vector(a)
    bb = center_vector(b)
    norm_a = float(np.linalg.norm(aa))
    norm_b = float(np.linalg.norm(bb))
    if norm_a <= float(eps) or norm_b <= float(eps):
        return float("nan")
    return float(np.dot(aa, bb) / (norm_a * norm_b))


def _safe_r2(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[finite]
    yy = yy[finite]
    if xx.size < 2 or float(np.std(xx)) <= EPS or float(np.std(yy)) <= EPS:
        return float("nan")
    corr = np.corrcoef(xx, yy)[0, 1]
    return float(corr * corr)


def _clean_numeric_pair(x: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(xx) & np.isfinite(yy)
    return xx[finite], yy[finite]


def _correlation_summary(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
) -> dict[str, float]:
    xx, yy = _clean_numeric_pair(x, y)
    if xx.size < 2 or float(np.std(xx)) <= EPS or float(np.std(yy)) <= EPS:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "r2": float("nan"),
            "n": int(xx.size),
        }
    pearson = stats.pearsonr(xx, yy)
    spearman = stats.spearmanr(xx, yy)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "r2": float(_safe_r2(xx, yy)),
        "n": int(xx.size),
    }


def _rank_pattern(values_by_region: Mapping[str, float]) -> str:
    values = {str(region): float(values_by_region[str(region)]) for region in MAIN_REGION_ORDER}
    if not all(np.isfinite(values[region]) for region in MAIN_REGION_ORDER):
        return "undefined"
    order_index = {region: idx for idx, region in enumerate(MAIN_REGION_ORDER)}
    ranked = sorted(MAIN_REGION_ORDER, key=lambda region: (-values[region], order_index[region]))
    return ">".join(ranked)


def _is_canonical_region_order(values_by_region: Mapping[str, float]) -> float:
    values = {str(region): float(values_by_region[str(region)]) for region in MAIN_REGION_ORDER}
    if not all(np.isfinite(values[region]) for region in MAIN_REGION_ORDER):
        return float("nan")
    return float(values["shared"] > values["distractor_only"] > values["sample_only"])


def _winner_label(value: float, tie_threshold: float) -> str:
    if not np.isfinite(float(value)):
        return "nan"
    if float(value) > float(tie_threshold):
        return "sample"
    if float(value) < -float(tie_threshold):
        return "distractor"
    return "tie"


def _pooled_mask(mask: np.ndarray, pool_layer: Any) -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(mask, dtype=np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(
        tensor,
        kernel_size=pool_layer.kernel_size,
        stride=pool_layer.stride,
        padding=pool_layer.padding,
        dilation=getattr(pool_layer, "dilation", 1),
        ceil_mode=getattr(pool_layer, "ceil_mode", False),
    )
    return np.asarray(pooled.squeeze(0).squeeze(0).numpy() > 0.0, dtype=bool)


def _conv_support_mask(mask: np.ndarray, conv_layer: Any) -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(mask, dtype=np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, int(conv_layer.kernel_size), int(conv_layer.kernel_size)), dtype=torch.float32, device=tensor.device)
    supported = F.conv2d(tensor, kernel, stride=int(conv_layer.stride), padding=int(conv_layer.padding))
    return np.asarray(supported.squeeze(0).squeeze(0).numpy() > 0.0, dtype=bool)


def _project_phase_mask(mask: np.ndarray, phase_foreground: np.ndarray, dilation_radius: int) -> np.ndarray:
    base = np.asarray(mask, dtype=bool)
    if int(dilation_radius) > 0 and base.any():
        return dilate_mask(base, int(dilation_radius)) & np.asarray(phase_foreground, dtype=bool)
    return base & np.asarray(phase_foreground, dtype=bool)


def build_probe_region_bundle(
    *,
    net: SDNN_Network,
    sample_image: torch.Tensor,
    distractor_image: torch.Tensor,
    probe_image: torch.Tensor,
    foreground_threshold: float,
    dilation_radius: int,
) -> ProbeRegionBundle:
    sample_fg = foreground_mask(sample_image, threshold=foreground_threshold)
    distractor_fg = foreground_mask(distractor_image, threshold=foreground_threshold)
    _ = probe_image

    sample_only = sample_fg & ~distractor_fg
    distractor_only = distractor_fg & ~sample_fg
    shared = sample_fg & distractor_fg
    region_union = sample_only | distractor_only | shared
    region_null = np.zeros_like(region_union, dtype=bool)
    sample_only_phase = _project_phase_mask(sample_only, sample_fg, dilation_radius)
    shared_sample_phase = _project_phase_mask(shared, sample_fg, dilation_radius)
    distractor_only_phase = _project_phase_mask(distractor_only, distractor_fg, dilation_radius)
    shared_distractor_phase = _project_phase_mask(shared, distractor_fg, dilation_radius)
    sample_union = sample_only_phase | shared_sample_phase
    distractor_union = distractor_only_phase | shared_distractor_phase
    layer1_masks = {
        "sample_only": sample_only,
        "distractor_only": distractor_only,
        "shared": shared,
        "null": region_null,
    }
    layer2_masks = {name: _pooled_mask(mask, net.pool1) for name, mask in layer1_masks.items()}
    layer2_output_masks = {name: _conv_support_mask(mask, net.layer2) for name, mask in layer2_masks.items()}
    layer3_masks = {name: _pooled_mask(mask, net.pool2) for name, mask in layer2_output_masks.items()}
    metadata: dict[str, object] = {
        "sample_foreground_area": int(sample_fg.sum()),
        "distractor_foreground_area": int(distractor_fg.sum()),
        "sample_only_area": int(sample_only.sum()),
        "distractor_only_area": int(distractor_only.sum()),
        "shared_area": int(shared.sum()),
    }
    for layer_name, layer_masks in (("layer1", layer1_masks), ("layer2", layer2_masks), ("layer3", layer3_masks)):
        for region_name, region_mask in layer_masks.items():
            metadata[f"{layer_name}_{region_name}_region_size"] = int(np.asarray(region_mask, dtype=bool).sum())
    return ProbeRegionBundle(
        probe_region_masks={
            "sample_only": sample_only,
            "distractor_only": distractor_only,
            "shared": shared,
            "null": region_null,
            "union": region_union,
            "foreground": region_union,
        },
        sample_phase_masks={"sample_only": sample_only_phase, "shared": shared_sample_phase, "union": sample_union, "foreground": sample_fg},
        distractor_phase_masks={"distractor_only": distractor_only_phase, "shared": shared_distractor_phase, "union": distractor_union, "foreground": distractor_fg},
        layer_region_masks={"layer1": layer1_masks, "layer2": layer2_masks, "layer3": layer3_masks},
        metadata=metadata,
    )


def _reset_decision_window(net: SDNN_Network, *, phase_reset: bool) -> None:
    net.layer3.reset_decision_state()
    if bool(phase_reset):
        with torch.no_grad():
            net.layer3.v_mem.fill_(net.layer3.V_L)
            net.layer3.lateral_inh.reset_state(net.layer3.output_shape)


def _capture_preprobe_states(net: SDNN_Network, *, stsp_mode: str) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        if str(stsp_mode) == "static_frozen":
            u_state = torch.full_like(layer.u_pre.detach(), float(layer.stsp_U))
            x_state = torch.ones_like(layer.x_pre.detach())
        else:
            u_state = layer.u_pre.detach().clone()
            x_state = layer.x_pre.detach().clone()
        out[layer_key] = {
            "u": u_state.detach().cpu().to(torch.float32),
            "x": x_state.detach().cpu().to(torch.float32),
            "ux": (u_state * x_state).detach().cpu().to(torch.float32),
        }
    return out


def _extract_grouped_voltage_vector(net: SDNN_Network, voltage_snapshot: torch.Tensor) -> np.ndarray:
    grouped = net.layer3.get_grouped_voltage(voltage_snapshot.to(torch.float32))
    return grouped.mean(dim=-1).detach().cpu().numpy().astype(np.float64, copy=False)


def run_distractor_rollout_capture(
    net: SDNN_Network,
    sample_spikes: torch.Tensor,
    distractor_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    *,
    delay1_steps: int,
    delay2_steps: int,
    stsp_mode: str,
    readout_step: int,
    sample_input_mask: torch.Tensor | np.ndarray | None = None,
    distractor_input_mask: torch.Tensor | np.ndarray | None = None,
    phase_reset: bool = True,
    before_probe_fn: Callable[[SDNN_Network], dict[str, object]] | None = None,
    note: str = "",
) -> RolloutCapture:
    if sample_spikes.ndim != 5 or distractor_spikes.ndim != 5 or probe_spikes.ndim != 5:
        raise ValueError("All spike tensors must have shape [B, T, C, H, W].")
    batch_size, _, channels, height, width = sample_spikes.shape
    masked_sample = apply_input_mask_to_spike_batch(sample_spikes, sample_input_mask, mode="remove")
    masked_distractor = apply_input_mask_to_spike_batch(distractor_spikes, distractor_input_mask, mode="remove")
    prepare_network_state(net, batch_size, channels, height, width)

    zero_input = torch.zeros((batch_size, channels, height, width), dtype=sample_spikes.dtype, device=sample_spikes.device)
    current_time = 0
    probe_grouped_frames: list[np.ndarray] = []
    probe_l1_frames: list[torch.Tensor] = []
    probe_l2_frames: list[torch.Tensor] = []
    probe_l3_frames: list[torch.Tensor] = []
    readout_snapshot: torch.Tensor | None = None

    def step_network(input_t: torch.Tensor, *, phase: str, phase_step: int, force_l3_time: int | None = None) -> None:
        nonlocal current_time, readout_snapshot
        monitor_probe = phase == "probe"
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())
        l3_time = current_time if force_l3_time is None else force_l3_time
        _, monitor_data = net.layer3.forward_step(
            s2_p,
            l3_time,
            training=False,
            monitor=monitor_probe or int(phase_step) == int(readout_step),
            stsp_mode=stsp_mode,
        )
        if phase == "probe":
            probe_l1_frames.append(s1_p.detach().cpu().to(torch.float32))
            probe_l2_frames.append(s2.detach().cpu().to(torch.float32))
            probe_l3_frames.append(s2_p.detach().cpu().to(torch.float32))
            v_snapshot = monitor_data.get("v_mem_snapshot", net.layer3.v_mem.detach())
            probe_grouped_frames.append(_extract_grouped_voltage_vector(net, v_snapshot))
        if phase == "probe" and int(phase_step) == int(readout_step):
            readout_snapshot = monitor_data.get("v_mem_snapshot", net.layer3.v_mem.detach()).detach().cpu().to(torch.float32)
        current_time += 1

    with torch.no_grad():
        for t_step in range(int(masked_sample.shape[1])):
            step_network(masked_sample[:, t_step, ...], phase="sample", phase_step=t_step)
        for _ in range(int(delay1_steps)):
            step_network(zero_input, phase="delay1", phase_step=0)
        _reset_decision_window(net, phase_reset=phase_reset)
        for t_step in range(int(masked_distractor.shape[1])):
            force_t = int(t_step) if bool(phase_reset) else None
            step_network(masked_distractor[:, t_step, ...], phase="distractor", phase_step=t_step, force_l3_time=force_t)
        for _ in range(int(delay2_steps)):
            step_network(zero_input, phase="delay2", phase_step=0)
        intervention_record = {} if before_probe_fn is None else dict(before_probe_fn(net))
        preprobe_states = _capture_preprobe_states(net, stsp_mode=stsp_mode)
        _reset_decision_window(net, phase_reset=phase_reset)
        for t_step in range(int(probe_spikes.shape[1])):
            force_t = int(t_step) if bool(phase_reset) else None
            step_network(probe_spikes[:, t_step, ...], phase="probe", phase_step=t_step, force_l3_time=force_t)

    if readout_snapshot is None:
        raise RuntimeError("Probe readout snapshot was not captured.")

    flat_times = net.layer3.firing_times
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    prediction_probe = (min_indices // net.layer3.neurons_per_class).long()
    prediction_probe[~has_fired] = -1
    first_fire_t_probe = min_times.clone()
    first_fire_t_probe[~has_fired] = -1
    first_fire_t_probe = first_fire_t_probe.to(torch.long)
    return RolloutCapture(
        grouped_voltage=_extract_grouped_voltage_vector(net, readout_snapshot),
        probe_grouped_voltage_trace=np.stack(probe_grouped_frames, axis=0),
        probe_l1_trace=torch.stack(probe_l1_frames, dim=0),
        probe_l2_trace=torch.stack(probe_l2_frames, dim=0),
        probe_l3_trace=torch.stack(probe_l3_frames, dim=0),
        readout_step=int(readout_step),
        prediction_probe=prediction_probe.detach().cpu().numpy().astype(np.int64, copy=False),
        first_fire_t_probe=first_fire_t_probe.detach().cpu().numpy().astype(np.int64, copy=False),
        preprobe_states=preprobe_states,
        intervention_record=intervention_record,
        note=str(note),
    )


def _resolve_winner_window(probe_steps: int, readout_step: int, window_frac: float) -> np.ndarray:
    window_size = max(3, int(math.ceil(float(window_frac) * float(probe_steps))))
    end = max(int(readout_step) + 1, int(probe_steps))
    start = max(0, end - window_size)
    return np.arange(start, int(probe_steps), dtype=np.int64)


def _build_winner_rows(
    *,
    batch_df: pd.DataFrame,
    condition_name: str,
    condition_capture: RolloutCapture,
    sample_reference_capture: RolloutCapture,
    distractor_reference_capture: RolloutCapture,
    winner_window_steps: np.ndarray,
    tie_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trial_rows: list[dict[str, object]] = []
    timecourse_rows: list[dict[str, object]] = []
    probe_steps = int(condition_capture.probe_grouped_voltage_trace.shape[0])
    winner_membership = np.zeros(probe_steps, dtype=np.int64)
    winner_membership[winner_window_steps] = 1
    for batch_idx, triplet_row in enumerate(batch_df.itertuples(index=False)):
        mix_trace = np.asarray(condition_capture.probe_grouped_voltage_trace[:, batch_idx, :], dtype=np.float64)
        sample_trace = np.asarray(sample_reference_capture.probe_grouped_voltage_trace[:, batch_idx, :], dtype=np.float64)
        distractor_trace = np.asarray(distractor_reference_capture.probe_grouped_voltage_trace[:, batch_idx, :], dtype=np.float64)
        c_sample = np.asarray([safe_cosine(mix_trace[t], sample_trace[t]) for t in range(probe_steps)], dtype=np.float64)
        c_distractor = np.asarray([safe_cosine(mix_trace[t], distractor_trace[t]) for t in range(probe_steps)], dtype=np.float64)
        winner_trace = c_sample - c_distractor
        winner_probe = float(np.nanmean(winner_trace[winner_window_steps]))
        winner_name = _winner_label(winner_probe, tie_threshold=tie_threshold)
        trial_rows.append(
            {
                "triplet_id": int(triplet_row.triplet_id),
                "condition": str(condition_name),
                "trial_id": f"{condition_name}_{int(triplet_row.triplet_id)}",
                "probe_id": int(triplet_row.probe_id),
                "sample_id": int(triplet_row.sample_id),
                "distractor_id": int(triplet_row.distractor_id),
                "sample_label": int(triplet_row.sample_label),
                "distractor_label": int(triplet_row.distractor_label),
                "probe_label": int(triplet_row.probe_label),
                "prediction_probe": int(condition_capture.prediction_probe[batch_idx]),
                "prediction_matches_sample": int(int(condition_capture.prediction_probe[batch_idx]) == int(triplet_row.sample_label)),
                "prediction_matches_distractor": int(int(condition_capture.prediction_probe[batch_idx]) == int(triplet_row.distractor_label)),
                "W_probe": float(winner_probe),
                "C_sample_probe": float(np.nanmean(c_sample[winner_window_steps])),
                "C_distractor_probe": float(np.nanmean(c_distractor[winner_window_steps])),
                "sample_win": int(winner_name == "sample"),
                "distractor_win": int(winner_name == "distractor"),
                "tie": int(winner_name == "tie"),
                "winner_label": str(winner_name),
                "first_fire_t_probe": int(condition_capture.first_fire_t_probe[batch_idx]),
            }
        )
        for t_step in range(probe_steps):
            timecourse_rows.append(
                {
                    "triplet_id": int(triplet_row.triplet_id),
                    "condition": str(condition_name),
                    "time_step": int(t_step),
                    "in_winner_window": int(winner_membership[t_step]),
                    "C_sample": float(c_sample[t_step]),
                    "C_distractor": float(c_distractor[t_step]),
                    "W_t": float(winner_trace[t_step]),
                }
            )
    return trial_rows, timecourse_rows


def _region_support_metrics(
    state_ux: torch.Tensor,
    spatial_mask: np.ndarray,
    *,
    baseline_ux: float,
) -> dict[str, float]:
    mask_bool = np.asarray(spatial_mask, dtype=bool)
    area = int(mask_bool.sum())
    if area <= 0:
        return {
            "region_size": 0,
            "support_mass": float("nan"),
            "mean_ux": float("nan"),
            "mean_positive_excess_ux": float("nan"),
        }
    arr = state_ux.detach().cpu().to(torch.float32).squeeze(0).numpy().astype(np.float64, copy=False)
    ux_map = arr.mean(axis=0)
    positive_excess_map = np.maximum(ux_map - float(baseline_ux), 0.0)
    support_mass_map = np.maximum(arr - float(baseline_ux), 0.0).mean(axis=0)
    mean_ux = float(np.sum(ux_map[mask_bool]) / float(area))
    mean_positive_excess_ux = float(np.sum(positive_excess_map[mask_bool]) / float(area))
    support_mass = float(np.sum(support_mass_map[mask_bool]) / float(area))
    return {
        "region_size": int(area),
        "support_mass": float(support_mass),
        "mean_ux": float(mean_ux),
        "mean_positive_excess_ux": float(mean_positive_excess_ux),
    }


def build_region_support_rows(
    *,
    batch_df: pd.DataFrame,
    bundles: Sequence[ProbeRegionBundle],
    capture: RolloutCapture,
    condition_name: str,
    layer_static_u: Mapping[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for batch_idx, triplet_row in enumerate(batch_df.itertuples(index=False)):
        bundle = bundles[batch_idx]
        for layer_key in LAYER_KEYS:
            ux_state = capture.preprobe_states[layer_key]["ux"][batch_idx : batch_idx + 1, ...]
            baseline_ux = float(layer_static_u[layer_key])
            for region_name in MAIN_REGION_ORDER:
                metrics = _region_support_metrics(ux_state, bundle.layer_region_masks[layer_key][region_name], baseline_ux=baseline_ux)
                rows.append(
                    {
                        "triplet_id": int(triplet_row.triplet_id),
                        "condition": str(condition_name),
                        "layer": str(layer_key),
                        "region": str(region_name),
                        "baseline_ux": float(baseline_ux),
                        "mean_ux": float(metrics["mean_ux"]),
                        "mean_positive_excess_ux": float(metrics["mean_positive_excess_ux"]),
                        "support_mass": float(metrics["support_mass"]),
                        "region_size": int(metrics["region_size"]),
                    }
                )
    return rows


def build_trial_region_support_summary(df_region_support: pd.DataFrame) -> pd.DataFrame:
    if df_region_support.empty:
        return pd.DataFrame(
            columns=[
                "triplet_id",
                "condition",
                "U_sample_only",
                "U_distractor_only",
                "U_shared",
                "U_total",
                "mean_ux_sample_only",
                "mean_ux_distractor_only",
                "mean_ux_shared",
            ]
        )
    grouped = (
        df_region_support.groupby(["triplet_id", "condition", "region"], sort=False)
        .agg(U_region=("support_mass", "mean"), mean_ux_region=("mean_ux", "mean"))
        .reset_index()
    )
    pivot_mass = grouped.pivot_table(index=["triplet_id", "condition"], columns="region", values="U_region")
    pivot_ux = grouped.pivot_table(index=["triplet_id", "condition"], columns="region", values="mean_ux_region")
    summary = pivot_mass.reset_index().rename(
        columns={"sample_only": "U_sample_only", "distractor_only": "U_distractor_only", "shared": "U_shared"}
    )
    summary["U_total"] = summary[["U_sample_only", "U_distractor_only", "U_shared"]].sum(axis=1, min_count=1)
    ux_frame = pivot_ux.reset_index().rename(
        columns={"sample_only": "mean_ux_sample_only", "distractor_only": "mean_ux_distractor_only", "shared": "mean_ux_shared"}
    )
    merged = summary.merge(ux_frame, on=["triplet_id", "condition"], how="left")
    return merged.sort_values(["triplet_id", "condition"], kind="stable").reset_index(drop=True)


def simulate_layer1_input_boundary_stsp(
    *,
    layer: Any,
    sample_spikes: torch.Tensor,
    distractor_spikes: torch.Tensor,
    delay1_steps: int,
    delay2_steps: int,
) -> dict[str, torch.Tensor]:
    if sample_spikes.ndim != 5 or distractor_spikes.ndim != 5:
        raise ValueError("Layer1 STSP simulation expects sample and distractor spikes with shape [B, T, C, H, W].")
    if tuple(sample_spikes.shape[0:1] + sample_spikes.shape[2:]) != tuple(distractor_spikes.shape[0:1] + distractor_spikes.shape[2:]):
        raise ValueError("Sample and distractor spike batches must agree on batch and spatial dimensions.")
    u = torch.full(
        (sample_spikes.shape[0], sample_spikes.shape[2], sample_spikes.shape[3], sample_spikes.shape[4]),
        float(layer.stsp_U),
        dtype=torch.float32,
        device=sample_spikes.device,
    )
    x = torch.ones_like(u)
    zero_input = torch.zeros_like(u)
    decay_x = float(layer.stsp_decay_x)
    decay_u = float(layer.stsp_decay_u)
    U = float(layer.stsp_U)

    def _advance(input_t: torch.Tensor) -> None:
        nonlocal u, x
        x = 1.0 + (x - 1.0) * decay_x
        u = U + (u - U) * decay_u
        gain = u * x
        mask = input_t > 0
        x = torch.where(mask, x - gain, x)
        u = torch.where(mask, u + U * (1.0 - u), u)
        x = torch.clamp(x, 0.0, 1.0)
        u = torch.clamp(u, 0.0, 1.0)

    with torch.no_grad():
        for t_step in range(int(sample_spikes.shape[1])):
            _advance(sample_spikes[:, t_step, ...].to(torch.float32))
        for _ in range(int(delay1_steps)):
            _advance(zero_input)
        for t_step in range(int(distractor_spikes.shape[1])):
            _advance(distractor_spikes[:, t_step, ...].to(torch.float32))
        for _ in range(int(delay2_steps)):
            _advance(zero_input)
    return {
        "u": u.detach().cpu().to(torch.float32),
        "x": x.detach().cpu().to(torch.float32),
        "ux": (u * x).detach().cpu().to(torch.float32),
    }


def build_layer1_trial_support_summary(
    df_region_support: pd.DataFrame,
    *,
    layer_key: str = "layer1",
    condition_name: str = BASELINE_CONDITION,
) -> pd.DataFrame:
    subset = df_region_support[
        (df_region_support["layer"] == str(layer_key))
        & (df_region_support["condition"] == str(condition_name))
    ].copy()
    if subset.empty:
        columns = [
            "triplet_id",
            "condition",
            f"{layer_key}_U_sample_only",
            f"{layer_key}_U_distractor_only",
            f"{layer_key}_U_shared",
            f"{layer_key}_mean_ux_sample_only",
            f"{layer_key}_mean_ux_distractor_only",
            f"{layer_key}_mean_ux_shared",
        ]
        return pd.DataFrame(columns=columns)
    pivot_mass = subset.pivot_table(index=["triplet_id", "condition"], columns="region", values="support_mass")
    pivot_ux = subset.pivot_table(index=["triplet_id", "condition"], columns="region", values="mean_ux")
    summary = pivot_mass.reset_index().rename(
        columns={region: f"{layer_key}_U_{region}" for region in MAIN_REGION_ORDER}
    )
    ux_frame = pivot_ux.reset_index().rename(
        columns={region: f"{layer_key}_mean_ux_{region}" for region in MAIN_REGION_ORDER}
    )
    return summary.merge(ux_frame, on=["triplet_id", "condition"], how="left").sort_values(
        ["triplet_id", "condition"], kind="stable"
    ).reset_index(drop=True)


def build_layer1_composition_trial_rows(
    *,
    batch_df: pd.DataFrame,
    bundles: Sequence[ProbeRegionBundle],
    observed_state: Mapping[str, torch.Tensor],
    predicted_state: Mapping[str, torch.Tensor],
    baseline_ux: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for batch_idx, triplet_row in enumerate(batch_df.itertuples(index=False)):
        bundle = bundles[batch_idx]
        row: dict[str, object] = {
            "triplet_id": int(triplet_row.triplet_id),
            "sample_id": int(triplet_row.sample_id),
            "distractor_id": int(triplet_row.distractor_id),
            "probe_id": int(triplet_row.probe_id),
            "sample_label": int(triplet_row.sample_label),
            "distractor_label": int(triplet_row.distractor_label),
            "probe_label": int(triplet_row.probe_label),
            "layer1_baseline_U": float(baseline_ux),
        }
        observed_mean: dict[str, float] = {}
        observed_excess: dict[str, float] = {}
        observed_mass: dict[str, float] = {}
        predicted_mean: dict[str, float] = {}
        predicted_excess: dict[str, float] = {}
        predicted_mass: dict[str, float] = {}
        for region_name in MAIN_REGION_ORDER:
            mask = bundle.layer_region_masks["layer1"][region_name]
            observed_metrics = _region_support_metrics(
                observed_state["ux"][batch_idx : batch_idx + 1, ...],
                mask,
                baseline_ux=baseline_ux,
            )
            predicted_metrics = _region_support_metrics(
                predicted_state["ux"][batch_idx : batch_idx + 1, ...],
                mask,
                baseline_ux=baseline_ux,
            )
            row[f"region_size_{region_name}"] = int(observed_metrics["region_size"])
            row[f"observed_mean_ux_{region_name}"] = float(observed_metrics["mean_ux"])
            row[f"observed_mean_excess_ux_{region_name}"] = float(observed_metrics["mean_positive_excess_ux"])
            row[f"observed_support_mass_{region_name}"] = float(observed_metrics["support_mass"])
            row[f"predicted_mean_ux_{region_name}"] = float(predicted_metrics["mean_ux"])
            row[f"predicted_mean_excess_ux_{region_name}"] = float(predicted_metrics["mean_positive_excess_ux"])
            row[f"predicted_support_mass_{region_name}"] = float(predicted_metrics["support_mass"])
            observed_mean[region_name] = float(observed_metrics["mean_ux"])
            observed_excess[region_name] = float(observed_metrics["mean_positive_excess_ux"])
            observed_mass[region_name] = float(observed_metrics["support_mass"])
            predicted_mean[region_name] = float(predicted_metrics["mean_ux"])
            predicted_excess[region_name] = float(predicted_metrics["mean_positive_excess_ux"])
            predicted_mass[region_name] = float(predicted_metrics["support_mass"])
        row["observed_rank_pattern"] = _rank_pattern(observed_mean)
        row["predicted_rank_pattern"] = _rank_pattern(predicted_mean)
        row["rank_match"] = int(row["observed_rank_pattern"] == row["predicted_rank_pattern"])
        row["observed_rank_pattern_mean_excess"] = _rank_pattern(observed_excess)
        row["predicted_rank_pattern_mean_excess"] = _rank_pattern(predicted_excess)
        row["rank_match_mean_excess"] = int(row["observed_rank_pattern_mean_excess"] == row["predicted_rank_pattern_mean_excess"])
        row["observed_rank_pattern_support_mass"] = _rank_pattern(observed_mass)
        row["predicted_rank_pattern_support_mass"] = _rank_pattern(predicted_mass)
        row["rank_match_support_mass"] = int(row["observed_rank_pattern_support_mass"] == row["predicted_rank_pattern_support_mass"])
        row["observed_canonical_order"] = float(_is_canonical_region_order(observed_mean))
        row["predicted_canonical_order"] = float(_is_canonical_region_order(predicted_mean))
        row["observed_canonical_order_mean_excess"] = float(_is_canonical_region_order(observed_excess))
        row["predicted_canonical_order_mean_excess"] = float(_is_canonical_region_order(predicted_excess))
        row["observed_canonical_order_support_mass"] = float(_is_canonical_region_order(observed_mass))
        row["predicted_canonical_order_support_mass"] = float(_is_canonical_region_order(predicted_mass))
        rows.append(row)
    return rows


def build_layer1_composition_summary(df_trial: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = (
        ("mean_ux", "mean_ux", "rank_match", "canonical_order"),
        ("mean_excess_ux", "mean_excess_ux", "rank_match_mean_excess", "canonical_order_mean_excess"),
        ("support_mass", "support_mass", "rank_match_support_mass", "canonical_order_support_mass"),
    )
    for state_type in ("observed", "predicted"):
        for metric_name, metric_suffix, _, canonical_suffix in metrics:
            for region_name in MAIN_REGION_ORDER:
                values = df_trial[f"{state_type}_{metric_suffix}_{region_name}"].to_numpy(dtype=np.float64, copy=False)
                finite = np.isfinite(values)
                rows.append(
                    {
                        "summary_type": "region_stat",
                        "state_type": str(state_type),
                        "metric_name": str(metric_name),
                        "region": str(region_name),
                        "event": "",
                        "mean": float(np.mean(values[finite])) if finite.any() else float("nan"),
                        "sem": float(_sem(values[finite])) if finite.any() else float("nan"),
                        "probability": float("nan"),
                        "n": int(finite.sum()),
                    }
                )
            comparisons = (
                ("shared>distractor_only", f"{state_type}_{metric_suffix}_shared", f"{state_type}_{metric_suffix}_distractor_only"),
                ("distractor_only>sample_only", f"{state_type}_{metric_suffix}_distractor_only", f"{state_type}_{metric_suffix}_sample_only"),
                ("shared>sample_only", f"{state_type}_{metric_suffix}_shared", f"{state_type}_{metric_suffix}_sample_only"),
            )
            for event_name, left_col, right_col in comparisons:
                left = df_trial[left_col].to_numpy(dtype=np.float64, copy=False)
                right = df_trial[right_col].to_numpy(dtype=np.float64, copy=False)
                finite = np.isfinite(left) & np.isfinite(right)
                rows.append(
                    {
                        "summary_type": "pairwise_win_rate",
                        "state_type": str(state_type),
                        "metric_name": str(metric_name),
                        "region": "",
                        "event": str(event_name),
                        "mean": float("nan"),
                        "sem": float("nan"),
                        "probability": float(np.mean(left[finite] > right[finite])) if finite.any() else float("nan"),
                        "n": int(finite.sum()),
                    }
                )
            canonical_col = f"{state_type}_{canonical_suffix}"
            canonical_values = df_trial[canonical_col].to_numpy(dtype=np.float64, copy=False)
            finite = np.isfinite(canonical_values)
            rows.append(
                {
                    "summary_type": "rank_probability",
                    "state_type": str(state_type),
                        "metric_name": str(metric_name),
                        "region": "",
                        "event": "shared>distractor_only>sample_only",
                        "mean": float("nan"),
                        "sem": float("nan"),
                        "probability": float(np.mean(canonical_values[finite])) if finite.any() else float("nan"),
                    "n": int(finite.sum()),
                }
            )
    for metric_name, _, match_col, _ in metrics:
        values = df_trial[match_col].to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(values)
        rows.append(
            {
                "summary_type": "rank_match",
                "state_type": "predicted_vs_observed",
                "metric_name": str(metric_name),
                "region": "",
                "event": "predicted_pattern_equals_observed_pattern",
                "mean": float("nan"),
                "sem": float("nan"),
                "probability": float(np.mean(values[finite])) if finite.any() else float("nan"),
                "n": int(finite.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["summary_type", "state_type", "metric_name", "region", "event"], kind="stable"
    ).reset_index(drop=True)


def build_layer1_formula_fit_summary(df_trial: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_name, metric_suffix in (("mean_ux", "mean_ux"), ("mean_excess_ux", "mean_excess_ux"), ("support_mass", "support_mass")):
        pooled_pred: list[np.ndarray] = []
        pooled_obs: list[np.ndarray] = []
        for region_name in MAIN_REGION_ORDER:
            observed = df_trial[f"observed_{metric_suffix}_{region_name}"].to_numpy(dtype=np.float64, copy=False)
            predicted = df_trial[f"predicted_{metric_suffix}_{region_name}"].to_numpy(dtype=np.float64, copy=False)
            rows.append(
                {
                    "metric_name": str(metric_name),
                    "region": str(region_name),
                    **_correlation_summary(predicted, observed),
                }
            )
            pred_clean, obs_clean = _clean_numeric_pair(predicted, observed)
            if pred_clean.size > 0:
                pooled_pred.append(pred_clean)
                pooled_obs.append(obs_clean)
        pooled_pred_arr = np.concatenate(pooled_pred) if pooled_pred else np.asarray([], dtype=np.float64)
        pooled_obs_arr = np.concatenate(pooled_obs) if pooled_obs else np.asarray([], dtype=np.float64)
        rows.append(
            {
                "metric_name": str(metric_name),
                "region": "all",
                **_correlation_summary(pooled_pred_arr, pooled_obs_arr),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric_name", "region"], kind="stable").reset_index(drop=True)


def build_layer1_to_winner_bridge_summary(df_trial: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    bridge_df = df_trial.copy()
    bridge_df["U_sample_only_layer1"] = bridge_df["observed_support_mass_sample_only"]
    bridge_df["U_distractor_only_layer1"] = bridge_df["observed_support_mass_distractor_only"]
    bridge_df["U_shared_layer1"] = bridge_df["observed_support_mass_shared"]
    predictors = ("U_sample_only_layer1", "U_distractor_only_layer1", "U_shared_layer1")
    regression_df, regression_meta = fit_standardized_regression(
        bridge_df,
        response_col="W_probe",
        predictor_cols=predictors,
    )
    corr_rows: list[dict[str, object]] = []
    for predictor in predictors:
        corr_rows.append(
            {
                "predictor": str(predictor),
                **_correlation_summary(
                    bridge_df[predictor].to_numpy(dtype=np.float64, copy=False),
                    bridge_df["W_probe"].to_numpy(dtype=np.float64, copy=False),
                ),
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    if regression_df.empty:
        merged = corr_df.copy()
        for column_name in ("beta", "se", "ci_low", "ci_high", "t_value", "p_value"):
            merged[column_name] = float("nan")
    else:
        merged = regression_df.merge(corr_df, on="predictor", how="outer")
    merged["model_r2"] = float(regression_meta["r2"])
    merged["n"] = int(regression_meta["n"])
    return merged.sort_values(["predictor"], kind="stable").reset_index(drop=True), regression_meta

def fit_standardized_regression(
    df: pd.DataFrame,
    *,
    response_col: str,
    predictor_cols: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    use_cols = [str(response_col), *[str(col) for col in predictor_cols]]
    subset = df[use_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(subset) < len(predictor_cols) + 2:
        return pd.DataFrame(columns=["predictor", "beta", "se", "ci_low", "ci_high", "t_value", "p_value"]), {"r2": float("nan"), "n": int(len(subset))}
    y_raw = subset[response_col].to_numpy(dtype=np.float64, copy=False)
    x_raw = subset[list(predictor_cols)].to_numpy(dtype=np.float64, copy=False)
    y_std = np.std(y_raw, ddof=1)
    x_std = np.std(x_raw, axis=0, ddof=1)
    valid = np.isfinite(y_raw)
    for idx in range(x_raw.shape[1]):
        valid &= np.isfinite(x_raw[:, idx]) & (float(x_std[idx]) > EPS)
    valid &= float(y_std) > EPS
    y_raw = y_raw[valid]
    x_raw = x_raw[valid, :]
    if len(y_raw) < len(predictor_cols) + 2:
        return pd.DataFrame(columns=["predictor", "beta", "se", "ci_low", "ci_high", "t_value", "p_value"]), {"r2": float("nan"), "n": int(len(y_raw))}
    y = (y_raw - float(np.mean(y_raw))) / float(np.std(y_raw, ddof=1))
    x = (x_raw - np.mean(x_raw, axis=0, keepdims=True)) / np.std(x_raw, axis=0, ddof=1, keepdims=True)
    design = np.column_stack([np.ones(len(y), dtype=np.float64), x])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    residual = y - fitted
    dof = max(1, len(y) - design.shape[1])
    sigma2 = float(np.dot(residual, residual) / float(dof))
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    t_values = beta / np.maximum(se, EPS)
    p_values = 2.0 * stats.t.sf(np.abs(t_values), df=dof)
    rows = []
    for idx, predictor in enumerate(predictor_cols, start=1):
        rows.append(
            {
                "predictor": str(predictor),
                "beta": float(beta[idx]),
                "se": float(se[idx]),
                "ci_low": float(beta[idx] - 1.96 * se[idx]),
                "ci_high": float(beta[idx] + 1.96 * se[idx]),
                "t_value": float(t_values[idx]),
                "p_value": float(p_values[idx]),
            }
        )
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    ss_res = float(np.sum((y - fitted) ** 2))
    return pd.DataFrame(rows), {"r2": float(1.0 - ss_res / max(ss_tot, EPS)), "n": int(len(y))}


def _layer1_region_color(region_name: str) -> str:
    return {
        "sample_only": CONDITION_COLORS["sample_only"],
        "distractor_only": CONDITION_COLORS["distractor_only"],
        "shared": CONDITION_COLORS["shared"],
    }.get(str(region_name), "#777777")


def _select_layer1_example_triplet(df_triplets_aug: pd.DataFrame) -> int:
    if df_triplets_aug.empty:
        raise ValueError("No triplets available for Layer1 panel A.")
    required_cols = ["layer1_sample_only_region_size", "layer1_distractor_only_region_size", "layer1_shared_region_size"]
    valid = df_triplets_aug.copy()
    for col in required_cols:
        valid = valid[valid[col].fillna(0) > 0]
    if valid.empty:
        valid = df_triplets_aug.copy()
    valid = valid.assign(
        layer1_region_total=valid[["layer1_sample_only_region_size", "layer1_distractor_only_region_size", "layer1_shared_region_size"]].fillna(0).sum(axis=1)
    )
    return int(valid.sort_values(["layer1_region_total", "triplet_id"], ascending=[False, True], kind="stable").iloc[0]["triplet_id"])


def plot_layer1_panel_a_triplet_region_definition(
    *,
    images: torch.Tensor,
    bundle: ProbeRegionBundle,
    triplet_row: Mapping[str, object],
) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    sample_img = images[int(triplet_row["sample_id"])].detach().cpu().squeeze().numpy()
    distractor_img = images[int(triplet_row["distractor_id"])].detach().cpu().squeeze().numpy()
    image_specs = (
        (axes[0], sample_img, f"Sample #{int(triplet_row['sample_id'])}"),
        (axes[1], distractor_img, f"Distractor #{int(triplet_row['distractor_id'])}"),
    )
    for ax, image_arr, title in image_specs:
        ax.imshow(image_arr, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
    overlay_ax = axes[2]
    overlay_ax.imshow(np.maximum(sample_img, distractor_img), cmap="gray", vmin=0.0, vmax=1.0)
    rgba = np.zeros((*sample_img.shape, 4), dtype=np.float32)
    for region_name in MAIN_REGION_ORDER:
        color = _layer1_region_color(region_name).lstrip("#")
        rgb = np.asarray([int(color[idx : idx + 2], 16) for idx in (0, 2, 4)], dtype=np.float32) / 255.0
        mask = np.asarray(bundle.layer_region_masks["layer1"][region_name], dtype=bool)
        rgba[mask, :3] = rgb
        rgba[mask, 3] = 0.72
    overlay_ax.imshow(rgba)
    overlay_ax.set_title("Sample/Distractor-defined regions")
    overlay_ax.axis("off")
    legend_lines = [
        f"{region_name}: |{region_name}|={int(np.asarray(bundle.layer_region_masks['layer1'][region_name], dtype=bool).sum())}"
        for region_name in MAIN_REGION_ORDER
    ]
    fig.suptitle(
        f"Triplet {int(triplet_row['triplet_id'])}: sample_only=S\\D, distractor_only=D\\S, shared=S&D\n" + "   ".join(legend_lines),
        fontsize=11.0,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


def _bar_region_metric(
    ax: Any,
    *,
    df_trial: pd.DataFrame,
    column_prefix: str,
    ylabel: str,
) -> None:
    x = np.arange(len(MAIN_REGION_ORDER), dtype=np.float64)
    means = []
    sems = []
    colors = []
    for region_name in MAIN_REGION_ORDER:
        values = df_trial[f"{column_prefix}_{region_name}"].to_numpy(dtype=np.float64, copy=False)
        finite = values[np.isfinite(values)]
        means.append(float(np.mean(finite)) if finite.size else float("nan"))
        sems.append(float(_sem(finite)) if finite.size else float("nan"))
        colors.append(_layer1_region_color(region_name))
    ax.bar(x, means, color=colors, alpha=ALPHA_BAR)
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="black", elinewidth=LINE_WIDTH_REFERENCE, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["sample-only", "distractor-only", "shared"])
    ax.set_ylabel(ylabel)
    ax.grid(alpha=GRID_ALPHA, axis="y")


def plot_layer1_panel_b_observed_mean_ux_by_region(df_trial: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    _bar_region_metric(ax, df_trial=df_trial, column_prefix="observed_mean_ux", ylabel="Observed mean ux")
    fig.tight_layout()
    return fig


def plot_layer1_panel_c_observed_excess_and_mass_by_region(df_trial: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
    _bar_region_metric(axes[0], df_trial=df_trial, column_prefix="observed_mean_excess_ux", ylabel="Observed mean positive excess ux")
    _bar_region_metric(axes[1], df_trial=df_trial, column_prefix="observed_support_mass", ylabel="Observed support mass per area")
    fig.tight_layout()
    return fig


def plot_layer1_panel_d_predicted_vs_observed(
    df_trial: pd.DataFrame,
    df_formula_fit: pd.DataFrame,
) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.2))
    pooled_pred: list[np.ndarray] = []
    pooled_obs: list[np.ndarray] = []
    for region_name in MAIN_REGION_ORDER:
        predicted = df_trial[f"predicted_mean_ux_{region_name}"].to_numpy(dtype=np.float64, copy=False)
        observed = df_trial[f"observed_mean_ux_{region_name}"].to_numpy(dtype=np.float64, copy=False)
        xx, yy = _clean_numeric_pair(predicted, observed)
        if xx.size <= 0:
            continue
        pooled_pred.append(xx)
        pooled_obs.append(yy)
        ax.scatter(xx, yy, color=_layer1_region_color(region_name), alpha=ALPHA_SCATTER_LIGHT, s=26, label=region_name)
    pooled_x = np.concatenate(pooled_pred) if pooled_pred else np.asarray([], dtype=np.float64)
    pooled_y = np.concatenate(pooled_obs) if pooled_obs else np.asarray([], dtype=np.float64)
    if pooled_x.size >= 2 and float(np.std(pooled_x)) > EPS:
        slope, intercept = np.polyfit(pooled_x, pooled_y, deg=1)
        x_line = np.linspace(float(pooled_x.min()), float(pooled_x.max()), num=128)
        ax.plot(x_line, slope * x_line + intercept, color="#111111", linewidth=1.4)
    pooled_row = df_formula_fit[
        (df_formula_fit["metric_name"] == "mean_ux") & (df_formula_fit["region"] == "all")
    ]
    if not pooled_row.empty:
        fit_row = pooled_row.iloc[0]
        ax.text(
            0.02,
            0.98,
            f"Pearson r={float(fit_row['pearson_r']):.2f}\nSpearman ρ={float(fit_row['spearman_rho']):.2f}\nR²={float(fit_row['r2']):.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
        )
    ax.set_xlabel("Predicted mean ux")
    ax.set_ylabel("Observed mean ux")
    ax.grid(alpha=GRID_ALPHA)
    apply_standard_legend(ax, compact=True)
    fig.tight_layout()
    return fig


def plot_layer1_panel_e_rank_probability(df_summary: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.8))
    event_order = ["shared>distractor_only", "distractor_only>sample_only", "shared>sample_only", "shared>distractor_only>sample_only"]
    x = np.arange(len(event_order), dtype=np.float64)
    width = 0.34

    def _extract_prob(state_type: str, event_name: str) -> float:
        subset = df_summary[
            (df_summary["state_type"] == str(state_type))
            & (df_summary["metric_name"] == "mean_ux")
            & (df_summary["event"] == str(event_name))
        ]
        if subset.empty:
            return float("nan")
        return float(subset.iloc[0]["probability"])

    observed = [_extract_prob("observed", event_name) for event_name in event_order]
    predicted = [_extract_prob("predicted", event_name) for event_name in event_order]
    ax.bar(x - width / 2.0, observed, width=width, color=CONDITION_COLORS[BASELINE_CONDITION], alpha=ALPHA_BAR, label="Observed")
    ax.bar(x + width / 2.0, predicted, width=width, color=CONDITION_COLORS["shared"], alpha=ALPHA_BAR, label="Predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(event_order)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Probability")
    ax.grid(alpha=GRID_ALPHA, axis="y")
    apply_standard_legend(ax, compact=True)
    fig.tight_layout()
    return fig


def plot_layer1_panel_f_layer1_support_vs_winner(df_bridge: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8))
    order = ["U_sample_only_layer1", "U_distractor_only_layer1", "U_shared_layer1"]
    subset = df_bridge.set_index("predictor").reindex(order)
    x = np.arange(len(order), dtype=np.float64)
    colors = [_layer1_region_color(region_name) for region_name in MAIN_REGION_ORDER]
    ax.bar(x, subset["beta"], color=colors, alpha=ALPHA_BAR)
    ax.errorbar(
        x,
        subset["beta"],
        yerr=[subset["beta"] - subset["ci_low"], subset["ci_high"] - subset["beta"]],
        fmt="none",
        ecolor="black",
        elinewidth=LINE_WIDTH_REFERENCE,
        capsize=3,
    )
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["sample-only", "distractor-only", "shared"])
    ax.set_ylabel("Standardized beta on W_probe")
    ax.grid(alpha=GRID_ALPHA, axis="y")
    if not subset.empty and np.isfinite(float(subset["model_r2"].iloc[0])):
        ax.text(
            0.02,
            0.98,
            f"Model R²={float(subset['model_r2'].iloc[0]):.2f}\nN={int(subset['n'].iloc[0])}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
        )
    fig.tight_layout()
    return fig


def save_layer1_composition_panels(
    *,
    layout,
    images: torch.Tensor,
    df_triplets_aug: pd.DataFrame,
    region_bundle_by_triplet: Mapping[int, ProbeRegionBundle],
    df_layer1_trial: pd.DataFrame,
    df_formula_fit: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_bridge: pd.DataFrame,
) -> dict[str, object]:
    figure_paths: dict[str, object] = {}
    example_triplet_id = _select_layer1_example_triplet(df_triplets_aug)
    example_row = df_triplets_aug[df_triplets_aug["triplet_id"] == int(example_triplet_id)].iloc[0].to_dict()
    fig_a = plot_layer1_panel_a_triplet_region_definition(
        images=images,
        bundle=region_bundle_by_triplet[int(example_triplet_id)],
        triplet_row=example_row,
    )
    figure_paths[LAYER1_PANEL_FILENAMES["panel_a"]] = save_figure_all_formats(
        fig_a,
        layout.figure_base(LAYER1_PANEL_FILENAMES["panel_a"]),
    )
    plt.close(fig_a)
    fig_b = plot_layer1_panel_b_observed_mean_ux_by_region(df_layer1_trial)
    figure_paths[LAYER1_PANEL_FILENAMES["panel_b"]] = save_figure_all_formats(
        fig_b,
        layout.figure_base(LAYER1_PANEL_FILENAMES["panel_b"]),
    )
    plt.close(fig_b)
    fig_c = plot_layer1_panel_c_observed_excess_and_mass_by_region(df_layer1_trial)
    figure_paths[LAYER1_PANEL_FILENAMES["panel_c"]] = save_figure_all_formats(
        fig_c,
        layout.figure_base(LAYER1_PANEL_FILENAMES["panel_c"]),
    )
    plt.close(fig_c)
    fig_d = plot_layer1_panel_d_predicted_vs_observed(df_layer1_trial, df_formula_fit)
    figure_paths[LAYER1_PANEL_FILENAMES["panel_d"]] = save_figure_all_formats(
        fig_d,
        layout.figure_base(LAYER1_PANEL_FILENAMES["panel_d"]),
    )
    plt.close(fig_d)
    fig_e = plot_layer1_panel_e_rank_probability(df_summary)
    figure_paths[LAYER1_PANEL_FILENAMES["panel_e"]] = save_figure_all_formats(
        fig_e,
        layout.figure_base(LAYER1_PANEL_FILENAMES["panel_e"]),
    )
    plt.close(fig_e)
    fig_f = plot_layer1_panel_f_layer1_support_vs_winner(df_bridge)
    figure_paths[LAYER1_PANEL_FILENAMES["panel_f"]] = save_figure_all_formats(
        fig_f,
        layout.figure_base(LAYER1_PANEL_FILENAMES["panel_f"]),
    )
    plt.close(fig_f)
    return figure_paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supplementary Fig.5 analysis: distractor-region ux support and Layer1 local composition diagnostics."
    )
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--delay1-ms", type=float, default=DEFAULT_DELAY1_MS)
    parser.add_argument("--distractor-ms", type=float, default=DEFAULT_DISTRACTOR_MS)
    parser.add_argument("--delay2-ms", type=float, default=DEFAULT_DELAY2_MS)
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--samples-per-probe", type=int, default=DEFAULT_SAMPLES_PER_PROBE)
    parser.add_argument("--max-triplets", type=int, default=DEFAULT_MAX_TRIPLETS)
    parser.add_argument("--num-sim-bins", type=int, default=DEFAULT_NUM_SIM_BINS)
    parser.add_argument("--foreground-threshold", type=float, default=DEFAULT_FOREGROUND_THRESHOLD)
    parser.add_argument("--dilation-radius", type=int, default=DEFAULT_DILATION_RADIUS)
    parser.add_argument("--winner-window-frac", type=float, default=DEFAULT_WINNER_WINDOW_FRAC)
    parser.add_argument("--tie-threshold", type=float, default=DEFAULT_TIE_THRESHOLD)
    parser.add_argument("--redistribution-fraction", type=float, default=DEFAULT_REDISTRIBUTION_FRACTION)
    parser.add_argument(
        "--analysis-mode",
        type=str,
        default="both",
        choices=["full_current", "layer1_composition", "both"],
    )
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    if bool(args.smoke):
        args.batch_size = min(int(args.batch_size), 8)
        args.max_probes = min(int(args.max_probes), 6)
        args.samples_per_probe = min(int(args.samples_per_probe), 2)
        args.max_triplets = min(int(args.max_triplets), 24)
    for name, value, allow_zero in (
        ("--batch-size", int(args.batch_size), False),
        ("--max-probes", int(args.max_probes), False),
        ("--samples-per-probe", int(args.samples_per_probe), False),
        ("--max-triplets", int(args.max_triplets), False),
        ("--num-sim-bins", int(args.num_sim_bins), False),
        ("--dilation-radius", int(args.dilation_radius), True),
    ):
        _validate_positive(name, value, allow_zero=allow_zero)
    for name, value in (
        ("--sample-ms", float(args.sample_ms)),
        ("--delay1-ms", float(args.delay1_ms)),
        ("--distractor-ms", float(args.distractor_ms)),
        ("--delay2-ms", float(args.delay2_ms)),
        ("--probe-ms", float(args.probe_ms)),
        ("--winner-window-frac", float(args.winner_window_frac)),
        ("--tie-threshold", float(args.tie_threshold)),
        ("--redistribution-fraction", float(args.redistribution_fraction)),
    ):
        _validate_positive(name, value, allow_zero=False)

    seed_everything(int(args.seed))
    if str(args.device).strip().lower() == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; requested --device cuda.")
        raise RuntimeError("CUDA requested but not available.")
    device = resolve_device(args.device)
    run_layer1_analysis = str(args.analysis_mode) in {"layer1_composition", "both"}
    spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=float(args.sample_ms),
        delay1_ms=float(args.delay1_ms),
        distractor_ms=float(args.distractor_ms),
        delay2_ms=float(args.delay2_ms),
        probe_ms=float(args.probe_ms),
        phase_reset=True,
    )
    for phase_name, steps in (
        ("sample", spec.sample_steps),
        ("delay1", spec.delay1_steps),
        ("distractor", spec.distractor_steps),
        ("delay2", spec.delay2_steps),
        ("probe", spec.probe_steps),
    ):
        if int(steps) <= 0:
            raise ValueError(f"{phase_name} steps must resolve to a positive integer.")

    layout = prepare_result_layout(args.output_dir)
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model path missing: {model_path.resolve()}")
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")
    dataset = load_mnist_dataset(dataset_root=args.dataset_root, split=args.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    net, encoder = load_model_and_encoder(
        model_path=model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(float(args.sample_ms), float(args.distractor_ms), float(args.probe_ms)),
    )
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )
    winner_window_steps = _resolve_winner_window(int(spec.probe_steps), int(readout_step), float(args.winner_window_frac))

    class_index = {int(cls): np.where(labels == int(cls))[0].tolist() for cls in np.unique(labels)}
    df_triplets = build_triplet_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=int(args.max_probes),
        samples_per_probe=int(args.samples_per_probe),
        num_bins=int(args.num_sim_bins),
        max_triplets=int(args.max_triplets),
        seed=int(args.seed),
    )
    region_bundle_by_triplet: dict[int, ProbeRegionBundle] = {}
    for triplet_row in df_triplets.itertuples(index=False):
        region_bundle_by_triplet[int(triplet_row.triplet_id)] = build_probe_region_bundle(
            net=net,
            sample_image=images[int(triplet_row.sample_id)],
            distractor_image=images[int(triplet_row.distractor_id)],
            probe_image=images[int(triplet_row.probe_id)],
            foreground_threshold=float(args.foreground_threshold),
            dilation_radius=int(args.dilation_radius),
        )
    triplet_rows: list[dict[str, object]] = []
    for triplet_row in df_triplets.itertuples(index=False):
        row = dict(triplet_row._asdict())
        row.update(region_bundle_by_triplet[int(triplet_row.triplet_id)].metadata)
        triplet_rows.append(row)
    df_triplets_aug = pd.DataFrame(triplet_rows).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)

    layer_static_u = {layer_key: float(getattr(net, layer_key).stsp_U) for layer_key in LAYER_KEYS}
    region_support_rows: list[dict[str, object]] = []
    trial_winner_rows: list[dict[str, object]] = []
    winner_timecourse_rows: list[dict[str, object]] = []
    layer1_composition_rows: list[dict[str, object]] = []
    batch_starts = range(0, len(df_triplets_aug), int(args.batch_size))
    for batch_start in tqdm(batch_starts, total=max(1, math.ceil(len(df_triplets_aug) / int(args.batch_size))), desc="DistractorWinnerUX"):
        batch_df = df_triplets_aug.iloc[batch_start : batch_start + int(args.batch_size)].copy().reset_index(drop=True)
        batch_triplet_ids = batch_df["triplet_id"].astype(int).tolist()
        batch_bundles = [region_bundle_by_triplet[triplet_id] for triplet_id in batch_triplet_ids]
        sample_spikes, distractor_spikes, probe_spikes = prepare_triplet_spike_batch(
            images=images,
            batch_df=batch_df,
            encoder=encoder,
            sample_steps=int(spec.sample_steps),
            distractor_steps=int(spec.distractor_steps),
            probe_steps=int(spec.probe_steps),
            device=device,
        )
        zero_distractor_spikes = torch.zeros_like(distractor_spikes)
        capture_baseline = run_distractor_rollout_capture(net=net, sample_spikes=sample_spikes, distractor_spikes=distractor_spikes, probe_spikes=probe_spikes, delay1_steps=int(spec.delay1_steps), delay2_steps=int(spec.delay2_steps), stsp_mode="dynamic", readout_step=int(readout_step), phase_reset=spec.phase_reset, note=BASELINE_CONDITION)
        capture_sample_ref = run_distractor_rollout_capture(net=net, sample_spikes=sample_spikes, distractor_spikes=zero_distractor_spikes, probe_spikes=probe_spikes, delay1_steps=int(spec.delay1_steps), delay2_steps=int(spec.delay2_steps), stsp_mode="dynamic", readout_step=int(readout_step), phase_reset=spec.phase_reset, note=REFERENCE_SAMPLE_CONDITION)
        capture_distractor_ref = run_distractor_rollout_capture(net=net, sample_spikes=distractor_spikes, distractor_spikes=zero_distractor_spikes, probe_spikes=probe_spikes, delay1_steps=int(spec.delay1_steps), delay2_steps=int(spec.delay2_steps), stsp_mode="dynamic", readout_step=int(readout_step), phase_reset=spec.phase_reset, note=REFERENCE_DISTRACTOR_CONDITION)
        captures: dict[str, RolloutCapture] = {BASELINE_CONDITION: capture_baseline}
        if run_layer1_analysis:
            predicted_layer1_preprobe = simulate_layer1_input_boundary_stsp(
                layer=net.layer1,
                sample_spikes=sample_spikes,
                distractor_spikes=distractor_spikes,
                delay1_steps=int(spec.delay1_steps),
                delay2_steps=int(spec.delay2_steps),
            )
            layer1_composition_rows.extend(
                build_layer1_composition_trial_rows(
                    batch_df=batch_df,
                    bundles=batch_bundles,
                    observed_state=capture_baseline.preprobe_states["layer1"],
                    predicted_state=predicted_layer1_preprobe,
                    baseline_ux=layer_static_u["layer1"],
                )
            )
        for condition_name, capture in captures.items():
            region_support_rows.extend(build_region_support_rows(batch_df=batch_df, bundles=batch_bundles, capture=capture, condition_name=condition_name, layer_static_u=layer_static_u))
            rows_trial, rows_time = _build_winner_rows(batch_df=batch_df, condition_name=condition_name, condition_capture=capture, sample_reference_capture=capture_sample_ref, distractor_reference_capture=capture_distractor_ref, winner_window_steps=winner_window_steps, tie_threshold=float(args.tie_threshold))
            trial_winner_rows.extend(rows_trial)
            winner_timecourse_rows.extend(rows_time)

    df_region_support = pd.DataFrame(region_support_rows).sort_values(["triplet_id", "condition", "layer", "region"], kind="stable").reset_index(drop=True)
    df_trial_support = build_trial_region_support_summary(df_region_support)
    df_trial_metrics = pd.DataFrame(trial_winner_rows).sort_values(["triplet_id", "condition"], kind="stable").reset_index(drop=True)
    df_timecourse = pd.DataFrame(winner_timecourse_rows).sort_values(["triplet_id", "condition", "time_step"], kind="stable").reset_index(drop=True)
    df_baseline = df_trial_metrics[df_trial_metrics["condition"] == BASELINE_CONDITION].copy().merge(df_trial_support[df_trial_support["condition"] == BASELINE_CONDITION].drop(columns=["condition"]), on="triplet_id", how="left", validate="one_to_one")
    df_layer1_support_baseline = build_layer1_trial_support_summary(df_region_support, layer_key="layer1", condition_name=BASELINE_CONDITION)

    layer1_trial_columns = [
        "triplet_id",
        "sample_id",
        "distractor_id",
        "probe_id",
        "sample_label",
        "distractor_label",
        "probe_label",
        "layer1_baseline_U",
        "region_size_sample_only",
        "region_size_distractor_only",
        "region_size_shared",
        "observed_mean_ux_sample_only",
        "observed_mean_ux_distractor_only",
        "observed_mean_ux_shared",
        "observed_mean_excess_ux_sample_only",
        "observed_mean_excess_ux_distractor_only",
        "observed_mean_excess_ux_shared",
        "observed_support_mass_sample_only",
        "observed_support_mass_distractor_only",
        "observed_support_mass_shared",
        "predicted_mean_ux_sample_only",
        "predicted_mean_ux_distractor_only",
        "predicted_mean_ux_shared",
        "predicted_mean_excess_ux_sample_only",
        "predicted_mean_excess_ux_distractor_only",
        "predicted_mean_excess_ux_shared",
        "predicted_support_mass_sample_only",
        "predicted_support_mass_distractor_only",
        "predicted_support_mass_shared",
        "observed_rank_pattern",
        "predicted_rank_pattern",
        "rank_match",
        "observed_rank_pattern_mean_excess",
        "predicted_rank_pattern_mean_excess",
        "rank_match_mean_excess",
        "observed_rank_pattern_support_mass",
        "predicted_rank_pattern_support_mass",
        "rank_match_support_mass",
        "observed_canonical_order",
        "predicted_canonical_order",
        "observed_canonical_order_mean_excess",
        "predicted_canonical_order_mean_excess",
        "observed_canonical_order_support_mass",
        "predicted_canonical_order_support_mass",
        "W_probe",
        "winner_label",
    ]
    if layer1_composition_rows:
        df_layer1_composition_trial = pd.DataFrame(layer1_composition_rows).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)
    else:
        df_layer1_composition_trial = pd.DataFrame(columns=layer1_trial_columns)
    if run_layer1_analysis and not df_layer1_composition_trial.empty:
        df_layer1_composition_trial = df_layer1_composition_trial.merge(
            df_baseline[["triplet_id", "W_probe", "winner_label"]],
            on="triplet_id",
            how="left",
            validate="one_to_one",
        )
        df_layer1_composition_summary = build_layer1_composition_summary(df_layer1_composition_trial)
        df_layer1_formula_fit = build_layer1_formula_fit_summary(df_layer1_composition_trial)
        df_layer1_bridge, layer1_bridge_meta = build_layer1_to_winner_bridge_summary(df_layer1_composition_trial)
    else:
        df_layer1_composition_summary = pd.DataFrame(
            columns=["summary_type", "state_type", "metric_name", "region", "event", "mean", "sem", "probability", "n"]
        )
        df_layer1_formula_fit = pd.DataFrame(
            columns=["metric_name", "region", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "r2", "n"]
        )
        df_layer1_bridge = pd.DataFrame(
            columns=["predictor", "beta", "se", "ci_low", "ci_high", "t_value", "p_value", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "model_r2", "n"]
        )
        layer1_bridge_meta = {"r2": float("nan"), "n": 0}

    triplet_export_columns = [
        "triplet_id",
        "sample_id",
        "distractor_id",
        "probe_id",
        "sample_label",
        "distractor_label",
        "probe_label",
        "sp_similarity",
        "dp_similarity",
        "sd_similarity",
        "sample_foreground_area",
        "distractor_foreground_area",
        "sample_only_area",
        "distractor_only_area",
        "shared_area",
        "layer1_sample_only_region_size",
        "layer1_distractor_only_region_size",
        "layer1_shared_region_size",
        "layer2_sample_only_region_size",
        "layer2_distractor_only_region_size",
        "layer2_shared_region_size",
        "layer3_sample_only_region_size",
        "layer3_distractor_only_region_size",
        "layer3_shared_region_size",
    ]
    triplet_specs_csv = save_tidy_csv(df_triplets_aug[triplet_export_columns], layout.data_file("supp_triplet_specs.csv"), sort_by=["triplet_id"])
    region_support_csv = save_tidy_csv(df_region_support, layout.data_file("supp_trial_region_support_by_condition.csv"), sort_by=["triplet_id", "condition", "layer", "region"])
    trial_support_csv = save_tidy_csv(df_trial_support, layout.data_file("supp_trial_region_support_summary.csv"), sort_by=["triplet_id", "condition"])
    trial_metrics_csv = save_tidy_csv(df_trial_metrics, layout.data_file("supp_trial_winner_metrics_by_condition.csv"), sort_by=["triplet_id", "condition"])
    baseline_metrics_csv = save_tidy_csv(df_baseline, layout.data_file("supp_trial_winner_metrics_baseline.csv"), sort_by=["triplet_id"])
    timecourse_csv = save_tidy_csv(df_timecourse, layout.data_file("supp_winner_timecourse_long.csv"), sort_by=["triplet_id", "condition", "time_step"])
    layer1_support_baseline_csv = save_tidy_csv(df_layer1_support_baseline, layout.data_file("supp_layer1_trial_support_summary.csv"), sort_by=["triplet_id", "condition"])
    layer1_trial_csv = save_tidy_csv(df_layer1_composition_trial, layout.data_file("supp_layer1_composition_trial_metrics.csv"), sort_by=["triplet_id"])
    layer1_summary_csv = save_tidy_csv(df_layer1_composition_summary, layout.data_file("supp_layer1_composition_summary.csv"), sort_by=["summary_type", "state_type", "metric_name", "region", "event"])
    layer1_formula_fit_csv = save_tidy_csv(df_layer1_formula_fit, layout.data_file("supp_layer1_formula_fit_summary.csv"), sort_by=["metric_name", "region"])
    layer1_bridge_csv = save_tidy_csv(df_layer1_bridge, layout.data_file("supp_layer1_to_winner_bridge_summary.csv"), sort_by=["predictor"])

    figure_paths: dict[str, object] = {}
    if not bool(args.skip_figures) and run_layer1_analysis and not df_layer1_composition_trial.empty:
        figure_paths.update(
            save_layer1_composition_panels(
                layout=layout,
                images=images,
                df_triplets_aug=df_triplets_aug,
                region_bundle_by_triplet=region_bundle_by_triplet,
                df_layer1_trial=df_layer1_composition_trial,
                df_formula_fit=df_layer1_formula_fit,
                df_summary=df_layer1_composition_summary,
                df_bridge=df_layer1_bridge,
            )
        )
    run_config_path = Path(save_run_config({"model_path": str(model_path), "dataset_root": str(args.dataset_root), "split": str(args.split), "device": str(device), "seed": int(args.seed), "output_dir": str(layout.root.resolve()), "sample_ms": float(args.sample_ms), "delay1_ms": float(args.delay1_ms), "distractor_ms": float(args.distractor_ms), "delay2_ms": float(args.delay2_ms), "probe_ms": float(args.probe_ms), "batch_size": int(args.batch_size), "max_probes": int(args.max_probes), "samples_per_probe": int(args.samples_per_probe), "max_triplets": int(args.max_triplets), "num_sim_bins": int(args.num_sim_bins), "foreground_threshold": float(args.foreground_threshold), "dilation_radius": int(args.dilation_radius), "winner_window_frac": float(args.winner_window_frac), "tie_threshold": float(args.tie_threshold), "redistribution_fraction": float(args.redistribution_fraction), "analysis_mode": str(args.analysis_mode), "smoke": int(bool(args.smoke)), "readout_step": int(readout_step), "winner_window_steps": winner_window_steps.tolist(), "smoke_note": SMOKE_NOTE}, layout.root))
    smoke_command = f"conda run -n torch_env python src/experiments/distractor_region_ux_mechanism_experiment.py --device cuda --smoke --analysis-mode {str(args.analysis_mode)} --output-dir {layout.root}"
    summary_payload = {
        "experiment": "distractor_region_ux_mechanism_experiment",
        "role": "supporting_analysis_only",
        "scientific_question": "This export is supplementary only: it characterizes region-level and Layer1 local support patterns without serving as the primary Fig.5 backbone.",
        "mechanistic_interpretation": "Use this script as supporting context for triplet regions and Layer1-local support diagnostics. The new Fig.5 main conclusion should instead be read from the fusion+formation backbone.",
        "winner_definition": {"sample_reference": "Clean DMS control (S -> P) with zero distractor input in the distractor slot.", "distractor_reference": "Control (D -> P) with the distractor treated as the preceding item and zero distractor input in the distractor slot.", "winner_timecourse": "W(t) = sim(v_mix(t), v_S(t)) - sim(v_mix(t), v_D(t))", "winner_probe": "W_probe is the mean of W(t) over the probe readout window.", "winner_window_steps": [int(v) for v in winner_window_steps.tolist()]},
        "regional_ux_definition": "Region metrics are computed as unit-area values on the sample-only, distractor-only, and shared supports. mean_ux, mean_excess_ux, and support_mass are all normalized by region_size.",
        "layer1_composition": {
            "enabled": bool(run_layer1_analysis),
            "role": "supplementary_only",
            "scientific_question": "Supplementary Layer1 analysis: whether pre-probe STSP shows a local chunk composition law and how strongly that local structure aligns with winner outcome.",
            "region_definition": "Layer1 regions reuse build_probe_region_bundle(): sample_only=S\\D, distractor_only=D\\S, shared=S&D. Probe does not participate in region construction.",
            "stsp_formula_used": {
                "recovery_update": "x = 1 + (x - 1) * decay_x; u = U + (u - U) * decay_u",
                "spike_update": "gain = u * x; x = x - gain; u = u + U * (1 - u)",
                "consistency": "The simulation matches src/core/network.py::stsp_dynamics_jit and is driven only by the true Layer1 input spikes across sample, delay1, distractor, and delay2."
            },
            "observed_region_order_summary": df_layer1_composition_summary[(df_layer1_composition_summary["state_type"] == "observed") & (df_layer1_composition_summary["metric_name"] == "mean_ux")].to_dict(orient="records"),
            "predicted_region_order_summary": df_layer1_composition_summary[(df_layer1_composition_summary["state_type"] == "predicted") & (df_layer1_composition_summary["metric_name"] == "mean_ux")].to_dict(orient="records"),
            "rank_match_summary": df_layer1_composition_summary[df_layer1_composition_summary["summary_type"] == "rank_match"].to_dict(orient="records"),
            "predicted_vs_observed_fit": df_layer1_formula_fit.to_dict(orient="records"),
            "layer1_to_winner_bridge": {"model_r2": float(layer1_bridge_meta["r2"]), "n_trials": int(layer1_bridge_meta["n"]), "rows": df_layer1_bridge.to_dict(orient="records")},
            "saved_artifacts": {
                "supp_layer1_trial_support_summary_csv": str(Path(layer1_support_baseline_csv).resolve()),
                "supp_layer1_composition_trial_metrics_csv": str(Path(layer1_trial_csv).resolve()),
                "supp_layer1_composition_summary_csv": str(Path(layer1_summary_csv).resolve()),
                "supp_layer1_formula_fit_summary_csv": str(Path(layer1_formula_fit_csv).resolve()),
                "supp_layer1_to_winner_bridge_summary_csv": str(Path(layer1_bridge_csv).resolve()),
                "figures": {key: value for key, value in figure_paths.items() if str(key).startswith("supp_fig5_layer1_")},
            },
        },
        "saved_artifacts": {"supp_triplet_specs_csv": str(Path(triplet_specs_csv).resolve()), "supp_trial_region_support_by_condition_csv": str(Path(region_support_csv).resolve()), "supp_trial_region_support_summary_csv": str(Path(trial_support_csv).resolve()), "supp_trial_winner_metrics_by_condition_csv": str(Path(trial_metrics_csv).resolve()), "supp_trial_winner_metrics_baseline_csv": str(Path(baseline_metrics_csv).resolve()), "supp_winner_timecourse_long_csv": str(Path(timecourse_csv).resolve()), "supp_layer1_trial_support_summary_csv": str(Path(layer1_support_baseline_csv).resolve()), "supp_layer1_composition_trial_metrics_csv": str(Path(layer1_trial_csv).resolve()), "supp_layer1_composition_summary_csv": str(Path(layer1_summary_csv).resolve()), "supp_layer1_formula_fit_summary_csv": str(Path(layer1_formula_fit_csv).resolve()), "supp_layer1_to_winner_bridge_summary_csv": str(Path(layer1_bridge_csv).resolve()), "run_config_json": str(Path(run_config_path).resolve()), "figures": figure_paths},
        "smoke": {"enabled": bool(args.smoke), "command": smoke_command, "note": SMOKE_NOTE},
    }
    summary_path = save_summary_json(summary_payload, layout.root)
    run_log_path = save_log_lines(["experiment=distractor_region_ux_mechanism_experiment", "role=supporting_analysis_only", f"model_path={args.model_path}", f"dataset_root={args.dataset_root}", f"seed={int(args.seed)}", f"device={device}", f"triplets={int(df_baseline['triplet_id'].nunique()) if len(df_baseline) else 0}", f"result_root={layout.root.resolve()}", f"smoke_note={SMOKE_NOTE}", f"summary_json={summary_path.resolve()}"], layout.log_dir)
    print(f"Triplets analysed: {int(df_baseline['triplet_id'].nunique()) if len(df_baseline) else 0}")
    print(f"Analysis mode: {args.analysis_mode}")
    print(f"Layer1 composition trials: {int(len(df_layer1_composition_trial))}")
    print(f"Winner window steps: {winner_window_steps.tolist()}")
    print(f"Smoke note: {SMOKE_NOTE}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Run log: {run_log_path.resolve()}")


if __name__ == "__main__":
    main()
