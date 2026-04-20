import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.experiments.ping_memory.shared.ping_api import (
    ExperimentSpec,
    LAYER_KEYS,
    NO_PING_LABEL,
    build_class_index,
    build_stratified_splits,
    compute_sample_and_noise_bias,
    decode_accuracy_with_splits,
    encode_images,
    format_ping_target_label,
    generate_balanced_trial_specs,
    load_model_and_encoder,
    override_tau_u_ms,
    parse_float_list,
    parse_seed_list,
    prepare_network_state,
    reset_l3_decision_window,
    seed_everything,
    select_monotonic_ping_indices,
    snapshot_ux_state,
    summarize_metrics,
    validate_trial_specs,
)
from src.experiments.ping_memory.shared.boundary_api import (
    DEFAULT_TAU_U_MS as SHARED_DEFAULT_TAU_U_MS,
    calibrate_ping_per_example as shared_calibrate_ping_per_example,
    compute_delta_summary as shared_compute_delta_summary,
    run_seed_experiment as shared_run_seed_experiment,
)
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.plotting.common.io import apply_publication_style, save_figure_all_formats
from src.plotting.common.theme_tokens import (
    ALPHA_BAR_SOFT,
    ALPHA_FILL,
    CMAP_DIVERGING,
    CMAP_OVERLAP,
    CMAP_SEQUENTIAL,
    CMAP_SEQUENTIAL_ALT,
    CMAP_SEQUENTIAL_CONTRAST,
    FIGSIZE_THREE_PANEL,
    FIGSIZE_THREE_PANEL_SUMMARY,
    FIGSIZE_TWO_BY_THREE,
    FIGSIZE_TWO_BY_TWO,
    FIGSIZE_TWO_PANEL,
    GRID_ALPHA,
    LINE_WIDTH_REFERENCE,
    LINE_WIDTH_SECONDARY,
    MARKER_CIRCLE,
    apply_standard_legend,
    horizontal_panel_figsize,
)
from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.experiments.ping_memory.shared.shuffle_ops import apply_trial_shuffle_ux_in_place, build_trial_shuffle_plan
from src.platform.legacy_adapters.units import ms


DEFAULT_TAU_U_MS = SHARED_DEFAULT_TAU_U_MS
DEFAULT_FIXED_SAMPLE_TO_PROBE_GAP_MS = 1000.0
DEFAULT_SHORT_PING_DURATION_MS_LIST = "5"
DEFAULT_LONG_PING_DURATION_MS_LIST = "30"
DEFAULT_SHORT_PING_TARGETS = (
    "0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,"
    "0.015,0.020,0.025,0.030,0.035"
)
DEFAULT_LONG_PING_TARGETS = "0.005,0.010,0.020,0.025,0.030,0.035,0.040,0.050,0.060,0.070"
DEFAULT_PING_DRIVE_CANDIDATES = "0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,4.0,5.0,6.0"
DEFAULT_S_SEL_TOPK_FRAC = 0.20
DEFAULT_SHORT_MAIN_DURATIONS = {5.0}
DEFAULT_LONG_MAIN_DURATIONS = {20.0, 30.0}
DEFAULT_DELAY1_MS_FALLBACK = 400.0
PHASE_ORDER = ["pre_ping", "post_ping_immediate", "pre_probe"]
PHASE_LATENT_LAYERS = ["layer2", "layer3"]

REGIME_SHORT = "short"
REGIME_LONG = "long"
REGIME_SHARED = "shared"

CONTROL_DYNAMIC_PING = "dynamic_ping"
CONTROL_DYNAMIC_NO_PING = "dynamic_no_ping"
CONTROL_STATIC_PING = "static_ping"
CONTROL_SHUFFLE_UX_PING = "shuffle_ux_ping"
CONTROL_PING_ONLY = "ping_only"

ROLE_MAIN = "main"
ROLE_SUPPLEMENTAL = "supplemental"
ROLE_CONTROL = "control"


@dataclass(frozen=True)
class ConditionDef:
    condition: str
    condition_label: str
    condition_family: str
    ping_target_label: str
    ping_target_frac: float
    ping_duration_ms: float
    delay2_ms: float
    regime: str
    control_mode: str
    analysis_role: str
    stsp_mode: str = "dynamic"
    zero_sample: bool = False
    shuffle_ux: bool = False


def _format_duration_tag(duration_ms: float) -> str:
    return f"{int(round(float(duration_ms))):02d}ms"


def _regime_from_duration(duration_ms: float) -> str:
    return REGIME_SHORT if float(duration_ms) <= 7.0 else REGIME_LONG


def _analysis_role_for_duration(duration_ms: float, regime: str) -> str:
    duration_ms = float(duration_ms)
    if regime == REGIME_SHORT:
        return ROLE_MAIN if any(np.isclose(duration_ms, v) for v in DEFAULT_SHORT_MAIN_DURATIONS) else ROLE_SUPPLEMENTAL
    if regime == REGIME_LONG:
        return ROLE_MAIN if any(np.isclose(duration_ms, v) for v in DEFAULT_LONG_MAIN_DURATIONS) else ROLE_SUPPLEMENTAL
    return ROLE_CONTROL


def make_condition_name(
    ping_target_label: str,
    ping_duration_ms: float,
    regime: str,
    control_mode: str,
) -> str:
    if control_mode == CONTROL_DYNAMIC_NO_PING:
        return "sample_ping__no_ping_shared"
    return f"{regime}__{control_mode}__{_format_duration_tag(ping_duration_ms)}__{ping_target_label}"


def make_condition_label(
    ping_target_label: str,
    ping_duration_ms: float,
    control_mode: str,
    regime: str,
) -> str:
    if control_mode == CONTROL_DYNAMIC_NO_PING:
        return "Dynamic STSP + no ping"
    control_text = {
        CONTROL_DYNAMIC_PING: "Dynamic ping",
        CONTROL_STATIC_PING: "Static + ping",
        CONTROL_SHUFFLE_UX_PING: "Shuffle u/x + ping",
        CONTROL_PING_ONLY: "Ping only (no sample)",
    }[control_mode]
    return f"{regime.title()} | {control_text} | {ping_target_label} | {ping_duration_ms:.0f} ms"


def derive_delay2_ms(
    delay1_ms: float,
    ping_duration_ms: float,
    fixed_sample_to_probe_gap_ms: float,
) -> float:
    delay2_ms = float(fixed_sample_to_probe_gap_ms - delay1_ms - ping_duration_ms)
    if delay2_ms <= 0.0:
        raise ValueError(
            "Derived delay2_ms must be positive: "
            f"fixed_sample_to_probe_gap_ms={fixed_sample_to_probe_gap_ms}, "
            f"delay1_ms={delay1_ms}, ping_duration_ms={ping_duration_ms}"
        )
    return delay2_ms


def build_baseline_condition(
    delay1_ms: float,
    fixed_sample_to_probe_gap_ms: float,
    ping_duration_ms: float = 0.0,
) -> ConditionDef:
    duration_ms = float(ping_duration_ms)
    delay2_ms = derive_delay2_ms(
        delay1_ms=delay1_ms,
        ping_duration_ms=duration_ms,
        fixed_sample_to_probe_gap_ms=fixed_sample_to_probe_gap_ms,
    )
    return ConditionDef(
        condition=make_condition_name(NO_PING_LABEL, duration_ms, REGIME_SHARED, CONTROL_DYNAMIC_NO_PING),
        condition_label=make_condition_label(NO_PING_LABEL, duration_ms, CONTROL_DYNAMIC_NO_PING, REGIME_SHARED),
        condition_family=CONTROL_DYNAMIC_NO_PING,
        ping_target_label=NO_PING_LABEL,
        ping_target_frac=0.0,
        ping_duration_ms=duration_ms,
        delay2_ms=delay2_ms,
        regime=REGIME_SHARED,
        control_mode=CONTROL_DYNAMIC_NO_PING,
        analysis_role=ROLE_CONTROL,
        stsp_mode="dynamic",
        zero_sample=False,
        shuffle_ux=False,
    )


def build_dynamic_condition_defs(
    ping_target_fracs: Sequence[float],
    ping_duration_ms: float,
    delay1_ms: float,
    fixed_sample_to_probe_gap_ms: float,
    regime: Optional[str] = None,
    analysis_role: Optional[str] = None,
    include_no_ping: bool = False,
) -> List[ConditionDef]:
    duration_ms = float(ping_duration_ms)
    regime_name = _regime_from_duration(duration_ms) if regime is None else str(regime)
    role_name = _analysis_role_for_duration(duration_ms, regime_name) if analysis_role is None else str(analysis_role)
    delay2_ms = derive_delay2_ms(
        delay1_ms=delay1_ms,
        ping_duration_ms=duration_ms,
        fixed_sample_to_probe_gap_ms=fixed_sample_to_probe_gap_ms,
    )
    out: List[ConditionDef] = []
    if include_no_ping:
        out.append(
            build_baseline_condition(
                delay1_ms=delay1_ms,
                fixed_sample_to_probe_gap_ms=fixed_sample_to_probe_gap_ms,
                ping_duration_ms=duration_ms,
            )
        )
    for target_frac in ping_target_fracs:
        target_label = format_ping_target_label(float(target_frac))
        out.append(
            ConditionDef(
                condition=make_condition_name(target_label, duration_ms, regime_name, CONTROL_DYNAMIC_PING),
                condition_label=make_condition_label(target_label, duration_ms, CONTROL_DYNAMIC_PING, regime_name),
                condition_family=CONTROL_DYNAMIC_PING,
                ping_target_label=target_label,
                ping_target_frac=float(target_frac),
                ping_duration_ms=duration_ms,
                delay2_ms=delay2_ms,
                regime=regime_name,
                control_mode=CONTROL_DYNAMIC_PING,
                analysis_role=role_name,
                stsp_mode="dynamic",
                zero_sample=False,
                shuffle_ux=False,
            )
        )
    return out


def build_long_control_condition_defs(base_conditions: Sequence[ConditionDef]) -> List[ConditionDef]:
    out: List[ConditionDef] = []
    for base in base_conditions:
        if base.control_mode != CONTROL_DYNAMIC_PING or base.regime != REGIME_LONG:
            continue
        out.extend(
            [
                ConditionDef(
                    condition=make_condition_name(base.ping_target_label, base.ping_duration_ms, base.regime, CONTROL_STATIC_PING),
                    condition_label=make_condition_label(base.ping_target_label, base.ping_duration_ms, CONTROL_STATIC_PING, base.regime),
                    condition_family=CONTROL_STATIC_PING,
                    ping_target_label=base.ping_target_label,
                    ping_target_frac=base.ping_target_frac,
                    ping_duration_ms=base.ping_duration_ms,
                    delay2_ms=base.delay2_ms,
                    regime=base.regime,
                    control_mode=CONTROL_STATIC_PING,
                    analysis_role=ROLE_CONTROL,
                    stsp_mode="static_frozen",
                ),
                ConditionDef(
                    condition=make_condition_name(base.ping_target_label, base.ping_duration_ms, base.regime, CONTROL_SHUFFLE_UX_PING),
                    condition_label=make_condition_label(base.ping_target_label, base.ping_duration_ms, CONTROL_SHUFFLE_UX_PING, base.regime),
                    condition_family=CONTROL_SHUFFLE_UX_PING,
                    ping_target_label=base.ping_target_label,
                    ping_target_frac=base.ping_target_frac,
                    ping_duration_ms=base.ping_duration_ms,
                    delay2_ms=base.delay2_ms,
                    regime=base.regime,
                    control_mode=CONTROL_SHUFFLE_UX_PING,
                    analysis_role=ROLE_CONTROL,
                    stsp_mode="dynamic",
                    shuffle_ux=True,
                ),
                ConditionDef(
                    condition=make_condition_name(base.ping_target_label, base.ping_duration_ms, base.regime, CONTROL_PING_ONLY),
                    condition_label=make_condition_label(base.ping_target_label, base.ping_duration_ms, CONTROL_PING_ONLY, base.regime),
                    condition_family=CONTROL_PING_ONLY,
                    ping_target_label=base.ping_target_label,
                    ping_target_frac=base.ping_target_frac,
                    ping_duration_ms=base.ping_duration_ms,
                    delay2_ms=base.delay2_ms,
                    regime=base.regime,
                    control_mode=CONTROL_PING_ONLY,
                    analysis_role=ROLE_CONTROL,
                    stsp_mode="dynamic",
                    zero_sample=True,
                ),
            ]
        )
    return out


def build_ping_drive(zero_input: torch.Tensor, ping_amp: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(zero_input) * ping_amp.view(-1, 1, 1, 1)


def snapshot_ux_component_layer_means(net, batch_size: int) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key, None)
        if layer is None or getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
            raise ValueError(f"{layer_key} is missing STSP state")
        u = layer.u_pre.detach().view(batch_size, -1).mean(dim=1).cpu().numpy().astype(np.float32, copy=False)
        x = layer.x_pre.detach().view(batch_size, -1).mean(dim=1).cpu().numpy().astype(np.float32, copy=False)
        gain = (layer.u_pre * layer.x_pre).detach().view(batch_size, -1).mean(dim=1).cpu().numpy().astype(np.float32, copy=False)
        out[layer_key] = {"u": u, "x": x, "ux": gain}
    return out


def flatten_trace_list(trace_list: List[torch.Tensor], batch_size: int, n_neurons: int) -> np.ndarray:
    if len(trace_list) == 0:
        return np.zeros((0, batch_size, n_neurons), dtype=np.float32)
    stacked = torch.stack(trace_list, dim=0).float()
    return stacked.view(stacked.shape[0], stacked.shape[1], -1).cpu().numpy().astype(np.float32, copy=False)


def compute_ping_window_predictions(layer3_trace_tbn: np.ndarray, neurons_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = int(layer3_trace_tbn.shape[1])
    pred = np.full(batch_size, -1, dtype=np.int64)
    first_fire_t = np.full(batch_size, -1, dtype=np.int64)
    if layer3_trace_tbn.shape[0] == 0:
        return pred, first_fire_t
    any_spike_tb = layer3_trace_tbn.any(axis=2)
    for i in range(batch_size):
        if not bool(any_spike_tb[:, i].any()):
            continue
        first_t = int(np.argmax(any_spike_tb[:, i]))
        first_neuron = int(np.argmax(layer3_trace_tbn[first_t, i]))
        pred[i] = int(first_neuron // neurons_per_class)
        first_fire_t[i] = first_t
    return pred, first_fire_t


def compute_ping_window_metrics(
    ping_traces: Dict[str, List[torch.Tensor]],
    dt: float,
    num_classes: int,
    neurons_per_class: int,
    batch_size: int,
    n_neurons_by_layer: Dict[str, int],
) -> Dict[str, object]:
    duration_steps = len(ping_traces["layer1"])
    duration_s = float(duration_steps * dt) if duration_steps > 0 else 1.0
    metrics: Dict[str, object] = {}
    layer3_trace_tbn: Optional[np.ndarray] = None

    for layer_key in LAYER_KEYS:
        flat_tbn = flatten_trace_list(ping_traces[layer_key], batch_size=batch_size, n_neurons=n_neurons_by_layer[layer_key])
        if duration_steps == 0:
            activation_fraction = np.zeros(batch_size, dtype=np.float32)
            spike_rate_hz = np.zeros(batch_size, dtype=np.float32)
            active_fraction = np.zeros(batch_size, dtype=np.float32)
            active_fraction_t = np.zeros((0, batch_size), dtype=np.float32)
            integrated_active_fraction_ms = np.zeros(batch_size, dtype=np.float32)
            spike_count_total = np.zeros(batch_size, dtype=np.float32)
            spike_count_vector = np.zeros((batch_size, n_neurons_by_layer[layer_key]), dtype=np.float32)
        else:
            activation_fraction = flat_tbn.mean(axis=(0, 2)).astype(np.float32, copy=False)
            spike_rate_hz = (activation_fraction / duration_s).astype(np.float32, copy=False)
            active_fraction = (flat_tbn > 0.0).any(axis=0).mean(axis=1).astype(np.float32, copy=False)
            active_fraction_t = (flat_tbn > 0.0).mean(axis=2).astype(np.float32, copy=False)
            integrated_active_fraction_ms = (active_fraction_t.sum(axis=0) * float(dt / ms)).astype(np.float32, copy=False)
            spike_count_total = flat_tbn.sum(axis=(0, 2)).astype(np.float32, copy=False)
            spike_count_vector = flat_tbn.sum(axis=0).astype(np.float32, copy=False)

        metrics[layer_key] = {
            "activation_fraction": activation_fraction,
            "spike_rate_hz": spike_rate_hz,
            "active_fraction": active_fraction,
            "active_fraction_t": active_fraction_t,
            "integrated_active_fraction_ms": integrated_active_fraction_ms,
            "spike_count_total": spike_count_total,
            "spike_count_vector": spike_count_vector,
        }
        if layer_key == "layer3":
            layer3_trace_tbn = flat_tbn

    assert layer3_trace_tbn is not None
    layer3_spike_count = np.asarray(metrics["layer3"]["spike_count_vector"], dtype=np.float32)
    layer3_class_counts = layer3_spike_count.reshape(layer3_spike_count.shape[0], num_classes, -1).sum(axis=2)
    pred_ping, first_fire_t_ping = compute_ping_window_predictions(layer3_trace_tbn, neurons_per_class=neurons_per_class)
    metrics["layer3_features"] = layer3_spike_count
    metrics["layer3_class_counts"] = layer3_class_counts.astype(np.float32, copy=False)
    metrics["prediction_ping"] = pred_ping
    metrics["first_fire_t_ping"] = first_fire_t_ping
    metrics["ping_duration_steps"] = duration_steps
    return metrics


def compute_sample_aligned_selectivity(layer3_class_counts: np.ndarray, sample_labels: np.ndarray) -> np.ndarray:
    out = np.zeros(len(sample_labels), dtype=np.float32)
    for i, sample_label in enumerate(sample_labels.tolist()):
        counts = layer3_class_counts[i]
        sample_count = float(counts[int(sample_label)])
        other_mask = np.ones(len(counts), dtype=bool)
        other_mask[int(sample_label)] = False
        other_mean = float(counts[other_mask].mean()) if other_mask.any() else 0.0
        out[i] = sample_count - other_mean
    return out


def compute_sample_selective_spike_fraction(
    sample_spike_count_vector: np.ndarray,
    ping_spike_count_vector: np.ndarray,
    topk_frac: float = DEFAULT_S_SEL_TOPK_FRAC,
) -> Dict[str, np.ndarray]:
    if sample_spike_count_vector.shape != ping_spike_count_vector.shape:
        raise ValueError(
            "sample_spike_count_vector and ping_spike_count_vector must share shape, "
            f"got {sample_spike_count_vector.shape} vs {ping_spike_count_vector.shape}"
        )
    if sample_spike_count_vector.ndim != 2:
        raise ValueError(f"Expected 2D spike count matrices, got ndim={sample_spike_count_vector.ndim}")
    n_trials, n_neurons = sample_spike_count_vector.shape
    if n_neurons <= 0:
        raise ValueError("Expected at least one L1 neuron")

    k = max(1, min(n_neurons, int(np.ceil(float(topk_frac) * float(n_neurons)))))
    topk_idx = np.argpartition(sample_spike_count_vector, kth=n_neurons - k, axis=1)[:, -k:]
    selected_ping_spikes = np.take_along_axis(ping_spike_count_vector, topk_idx, axis=1).sum(axis=1)
    total_ping_spikes = ping_spike_count_vector.sum(axis=1)
    s_sel = np.divide(
        selected_ping_spikes,
        total_ping_spikes,
        out=np.zeros_like(selected_ping_spikes, dtype=np.float32),
        where=total_ping_spikes > 0.0,
    )
    topk_relevance = np.take_along_axis(sample_spike_count_vector, topk_idx, axis=1)
    return {
        "S_L1": s_sel.astype(np.float32, copy=False),
        "topk_count": np.full(n_trials, k, dtype=np.int64),
        "ping_topk_spikes": selected_ping_spikes.astype(np.float32, copy=False),
        "ping_non_topk_spikes": (total_ping_spikes - selected_ping_spikes).astype(np.float32, copy=False),
        "topk_relevance_mean": topk_relevance.mean(axis=1).astype(np.float32, copy=False),
        "non_topk_relevance_mean": (
            (sample_spike_count_vector.sum(axis=1) - topk_relevance.sum(axis=1)) / float(max(1, n_neurons - k))
        ).astype(np.float32, copy=False),
    }


def compute_multiclass_fisher_stats(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape={features.shape}")
    if len(features) != len(labels):
        raise ValueError("features and labels length mismatch")
    if len(features) == 0:
        return {"between_class_var": float("nan"), "within_class_var": float("nan"), "fisher_ratio": float("nan")}

    feat = features.astype(np.float64, copy=False)
    y = labels.astype(np.int64, copy=False)
    global_mean = feat.mean(axis=0)
    between = 0.0
    within = 0.0
    dim = float(max(1, feat.shape[1]))
    for cls in np.unique(y):
        cls_feat = feat[y == cls]
        if len(cls_feat) == 0:
            continue
        cls_mean = cls_feat.mean(axis=0)
        between += float(len(cls_feat)) * float(np.sum((cls_mean - global_mean) ** 2) / dim)
        within += float(np.sum((cls_feat - cls_mean) ** 2) / dim)
    between /= float(max(1, len(feat)))
    within /= float(max(1, len(feat)))
    fisher = between / within if within > 0.0 else float("nan")
    return {
        "between_class_var": float(between),
        "within_class_var": float(within),
        "fisher_ratio": float(fisher),
    }


def choose_legacy_duration_ms(duration_ms_list: Sequence[float]) -> float:
    if len(duration_ms_list) == 0:
        raise ValueError("duration_ms_list is empty")
    for value in duration_ms_list:
        if np.isclose(float(value), 10.0):
            return float(value)
    return float(duration_ms_list[0])


def run_ping_calibration_session(
    net,
    sample_spikes: torch.Tensor,
    spec: ExperimentSpec,
    ping_amp: torch.Tensor,
    stsp_mode: str = "dynamic",
) -> Dict[str, object]:
    batch_size, t_sample, c, h, w = sample_spikes.shape
    if t_sample != spec.sample_steps:
        raise ValueError(f"Sample step mismatch: {t_sample} vs {spec.sample_steps}")

    prepare_network_state(net, batch_size, c, h, w)
    zero_input = torch.zeros((batch_size, c, h, w), device=sample_spikes.device)
    ping_drive = build_ping_drive(zero_input, ping_amp.to(sample_spikes.device))
    ping_traces: Dict[str, List[torch.Tensor]] = {layer_key: [] for layer_key in LAYER_KEYS}
    current_time = 0
    n_neurons_by_layer: Dict[str, int] = {}

    def step_network(input_t: torch.Tensor, ping_drive_t: Optional[torch.Tensor] = None, force_l3_time: Optional[int] = None, record_ping: bool = False) -> None:
        nonlocal current_time, n_neurons_by_layer
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, stsp_mode=stsp_mode, ping_drive=ping_drive_t)
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())
        t_for_l3 = current_time if force_l3_time is None else force_l3_time
        s3, _ = net.layer3.forward_step(s2_p, t_for_l3, training=False, monitor=False, stsp_mode=stsp_mode)
        if not n_neurons_by_layer:
            n_neurons_by_layer = {
                "layer1": int(s1.view(batch_size, -1).shape[1]),
                "layer2": int(s2.view(batch_size, -1).shape[1]),
                "layer3": int(s3.view(batch_size, -1).shape[1]),
            }
        if record_ping:
            ping_traces["layer1"].append(s1.detach().to(torch.bool).cpu())
            ping_traces["layer2"].append(s2.detach().to(torch.bool).cpu())
            ping_traces["layer3"].append(s3.detach().to(torch.bool).cpu())
        current_time += 1

    for t in range(spec.sample_steps):
        step_network(sample_spikes[:, t, ...])
    for _ in range(spec.delay_steps):
        step_network(zero_input)
    reset_l3_decision_window(net)
    for t in range(spec.ping_steps):
        step_network(zero_input, ping_drive_t=ping_drive, force_l3_time=t, record_ping=True)

    return compute_ping_window_metrics(
        ping_traces=ping_traces,
        dt=spec.dt,
        num_classes=net.layer3.num_classes,
        neurons_per_class=net.layer3.neurons_per_class,
        batch_size=batch_size,
        n_neurons_by_layer=n_neurons_by_layer,
    )


def run_ping_probe_session(
    net,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    spec: ExperimentSpec,
    ping_amp: torch.Tensor,
    *,
    stsp_mode: str = "dynamic",
    intervention_fn: Optional[Callable] = None,
    batch_meta: Optional[Dict[str, np.ndarray]] = None,
    zero_sample: bool = False,
) -> Dict[str, object]:
    batch_size, t_sample, c, h, w = sample_spikes.shape
    if t_sample != spec.sample_steps:
        raise ValueError(f"Sample step mismatch: {t_sample} vs {spec.sample_steps}")
    if probe_spikes.shape[1] != spec.probe_steps:
        raise ValueError(f"Probe step mismatch: {probe_spikes.shape[1]} vs {spec.probe_steps}")

    prepare_network_state(net, batch_size, c, h, w)
    zero_input = torch.zeros((batch_size, c, h, w), device=sample_spikes.device)
    ping_drive = build_ping_drive(zero_input, ping_amp.to(sample_spikes.device))
    ping_traces: Dict[str, List[torch.Tensor]] = {layer_key: [] for layer_key in LAYER_KEYS}
    ux_timecourse: List[Dict[str, object]] = []
    sample_l1_spike_count_vector: Optional[np.ndarray] = None
    n_neurons_by_layer: Dict[str, int] = {}
    current_time = 0
    intervention_record: Dict[str, int] = {"applied": 0}
    if batch_meta is None:
        batch_meta = {}

    def step_network(
        input_t: torch.Tensor,
        ping_drive_t: Optional[torch.Tensor] = None,
        force_l3_time: Optional[int] = None,
        record_ping: bool = False,
        record_sample_l1: bool = False,
    ) -> None:
        nonlocal current_time, sample_l1_spike_count_vector, n_neurons_by_layer
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, stsp_mode=stsp_mode, ping_drive=ping_drive_t)
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())
        t_for_l3 = current_time if force_l3_time is None else force_l3_time
        s3, _ = net.layer3.forward_step(s2_p, t_for_l3, training=False, monitor=False, stsp_mode=stsp_mode)
        if not n_neurons_by_layer:
            n_neurons_by_layer = {
                "layer1": int(s1.view(batch_size, -1).shape[1]),
                "layer2": int(s2.view(batch_size, -1).shape[1]),
                "layer3": int(s3.view(batch_size, -1).shape[1]),
            }
        if record_sample_l1:
            s1_flat = s1.detach().to(torch.bool).view(batch_size, -1).cpu().numpy().astype(np.float32, copy=False)
            if sample_l1_spike_count_vector is None:
                sample_l1_spike_count_vector = np.zeros_like(s1_flat, dtype=np.float32)
            sample_l1_spike_count_vector += s1_flat
        if record_ping:
            ping_traces["layer1"].append(s1.detach().to(torch.bool).cpu())
            ping_traces["layer2"].append(s2.detach().to(torch.bool).cpu())
            ping_traces["layer3"].append(s3.detach().to(torch.bool).cpu())
        current_time += 1

    def record_ux_timepoint(phase: str, phase_step: int) -> None:
        layer_means = snapshot_ux_component_layer_means(net, batch_size=batch_size)
        ux_timecourse.append(
            {
                "phase": phase,
                "phase_step": int(phase_step),
                "global_time_step": int(max(0, current_time - 1)),
                "time_ms": float(current_time * spec.dt / ms),
                "layer_means": layer_means,
            }
        )

    for t in range(spec.sample_steps):
        sample_input_t = zero_input if zero_sample else sample_spikes[:, t, ...]
        step_network(sample_input_t, record_sample_l1=True)
        record_ux_timepoint("sample", t)
    for t in range(spec.delay_steps):
        step_network(zero_input)
        record_ux_timepoint("delay1", t)

    pre_ping_ux = snapshot_ux_state(net, batch_size)
    if intervention_fn is not None:
        intervention_record = intervention_fn(net, batch_meta)

    reset_l3_decision_window(net)
    for t in range(spec.ping_steps):
        step_network(zero_input, ping_drive_t=ping_drive, force_l3_time=t, record_ping=True)
        record_ux_timepoint("ping", t)

    post_ping_immediate_ux = snapshot_ux_state(net, batch_size)
    if sample_l1_spike_count_vector is None:
        sample_l1_spike_count_vector = np.zeros((batch_size, n_neurons_by_layer["layer1"]), dtype=np.float32)
    ping_metrics = compute_ping_window_metrics(
        ping_traces=ping_traces,
        dt=spec.dt,
        num_classes=net.layer3.num_classes,
        neurons_per_class=net.layer3.neurons_per_class,
        batch_size=batch_size,
        n_neurons_by_layer=n_neurons_by_layer,
    )
    ping_metrics["sample_l1_spike_count_vector"] = sample_l1_spike_count_vector.astype(np.float32, copy=False)

    for t in range(spec.post_ping_steps):
        step_network(zero_input, force_l3_time=spec.ping_steps + t)
        record_ux_timepoint("delay2", t)

    pre_probe_ux = snapshot_ux_state(net, batch_size)
    reset_l3_decision_window(net)
    for t in range(spec.probe_steps):
        step_network(probe_spikes[:, t, ...], force_l3_time=t)
        record_ux_timepoint("probe", t)

    flat_times = net.layer3.firing_times
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    pred_probe = (min_indices // net.layer3.neurons_per_class).long()
    pred_probe[~has_fired] = -1
    fire_probe = min_times.clone()
    fire_probe[~has_fired] = -1

    return {
        "prediction_ping": torch.as_tensor(ping_metrics["prediction_ping"], dtype=torch.long),
        "first_fire_t_ping": torch.as_tensor(ping_metrics["first_fire_t_ping"], dtype=torch.long),
        "prediction_probe": pred_probe.detach().cpu().long(),
        "first_fire_t_probe": fire_probe.detach().cpu().long(),
        "pre_ping_ux": pre_ping_ux,
        "post_ping_immediate_ux": post_ping_immediate_ux,
        "pre_probe_ux": pre_probe_ux,
        "ping_metrics": ping_metrics,
        "ux_timecourse": ux_timecourse,
        "intervention_record": intervention_record,
    }


def build_unique_sample_table(df_specs: pd.DataFrame) -> pd.DataFrame:
    return (
        df_specs[["sample_index", "sample_label"]]
        .drop_duplicates()
        .sort_values(["sample_label", "sample_index"])
        .reset_index(drop=True)
    )


def select_ping_strengths_for_example(df_example: pd.DataFrame, targets: Sequence[float]) -> Tuple[List[Dict[str, object]], str]:
    sub = df_example.sort_values("ping_drive_amp").reset_index(drop=True)
    amps = sub["ping_drive_amp"].to_numpy(dtype=np.float64)
    activation = sub["l1_activation_fraction"].to_numpy(dtype=np.float64)
    chosen_idx = select_monotonic_ping_indices(amps, activation, targets)
    method = "fixed_activation_target"
    if chosen_idx is None:
        method = "fallback_monotone_nearest"
        chosen_idx = []
        start_idx = 0
        for target in targets:
            candidate_idx = np.arange(start_idx, len(amps), dtype=np.int64)
            if len(candidate_idx) == 0:
                candidate_idx = np.array([len(amps) - 1], dtype=np.int64)
            idx = int(candidate_idx[np.argmin(np.abs(activation[candidate_idx] - float(target)))])
            chosen_idx.append(idx)
            start_idx = idx
    rows: List[Dict[str, object]] = []
    for target, idx in zip(targets, chosen_idx):
        base_row = sub.iloc[int(idx)].to_dict()
        achieved = float(base_row["l1_activation_fraction"])
        rows.append(
            {
                **base_row,
                "ping_target_label": format_ping_target_label(float(target)),
                "ping_target_frac": float(target),
                "achieved_activation_frac": achieved,
                "activation_residual": achieved - float(target),
                "selection_method": method,
            }
        )
    return rows, method


def calibrate_ping_per_example(
    net,
    encoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    ping_amp_candidates: Sequence[float],
    ping_target_fracs: Sequence[float],
    batch_size: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, Dict[str, float]]]]:
    df_unique = build_unique_sample_table(df_specs)
    cached_batches: List[Tuple[pd.DataFrame, torch.Tensor]] = []
    for start in range(0, len(df_unique), batch_size):
        batch = df_unique.iloc[start : start + batch_size].copy()
        sample_imgs = torch.stack([dataset[int(i)][0] for i in batch["sample_index"].tolist()], dim=0).to(device)
        sample_spikes = encode_images(encoder, sample_imgs, spec.sample_steps)
        cached_batches.append((batch, sample_spikes))

    rows: List[Dict[str, object]] = []
    total_steps = len(ping_amp_candidates) * max(1, len(cached_batches))
    pbar = tqdm(total=total_steps, desc=f"Calibration {spec.ping_ms:.0f}ms", leave=True)
    try:
        for ping_amp in ping_amp_candidates:
            for batch, sample_spikes in cached_batches:
                amp_batch = torch.full((len(batch),), float(ping_amp), dtype=torch.float32, device=device)
                with torch.no_grad():
                    ping_metrics = run_ping_calibration_session(net, sample_spikes, spec, amp_batch, stsp_mode="dynamic")
                l1_spike_count_total = np.asarray(ping_metrics["layer1"]["spike_count_total"], dtype=np.float32)
                n_l1 = float(max(1, ping_metrics["layer1"]["spike_count_vector"].shape[1]))
                D_L1 = l1_spike_count_total / n_l1
                for i, row in enumerate(batch.itertuples(index=False)):
                    item = {
                        "sample_index": int(row.sample_index),
                        "sample_label": int(row.sample_label),
                        "ping_drive_amp": float(ping_amp),
                        "ping_duration_ms": float(spec.ping_ms),
                        "delay2_ms": float(spec.post_ping_ms),
                        "l1_activation_fraction": float(ping_metrics["layer1"]["activation_fraction"][i]),
                        "D_L1": float(D_L1[i]),
                        "l1_spike_count_total": float(l1_spike_count_total[i]),
                    }
                    for layer_key in LAYER_KEYS:
                        item[f"{layer_key}_activation_fraction"] = float(ping_metrics[layer_key]["activation_fraction"][i])
                        item[f"{layer_key}_spike_rate_hz"] = float(ping_metrics[layer_key]["spike_rate_hz"][i])
                        item[f"{layer_key}_active_fraction"] = float(ping_metrics[layer_key]["active_fraction"][i])
                        item[f"{layer_key}_spike_count_total"] = float(ping_metrics[layer_key]["spike_count_total"][i])
                    rows.append(item)
                pbar.update(1)
    finally:
        pbar.close()

    df_calibration = pd.DataFrame(rows).sort_values(["ping_duration_ms", "sample_label", "sample_index", "ping_drive_amp"]).reset_index(drop=True)

    selected_rows: List[Dict[str, object]] = []
    lookup: Dict[int, Dict[str, Dict[str, float]]] = {}
    for sample_index, sub in df_calibration.groupby("sample_index", sort=False):
        picked, method = select_ping_strengths_for_example(sub.copy(), ping_target_fracs)
        lookup[int(sample_index)] = {}
        for item in picked:
            target_label = str(item["ping_target_label"])
            lookup[int(sample_index)][target_label] = {
                "ping_drive_amp": float(item["ping_drive_amp"]),
                "achieved_activation_frac": float(item["achieved_activation_frac"]),
                "activation_residual": float(item["activation_residual"]),
                "D_L1": float(item.get("D_L1", 0.0)),
                "selection_method": method,
            }
            selected_rows.append(item)

    df_selected = pd.DataFrame(selected_rows).sort_values(["ping_duration_ms", "sample_label", "sample_index", "ping_target_frac"]).reset_index(drop=True)
    return df_calibration, df_selected, lookup


def build_error_destination_columns(df_trials: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    out = df_trials.copy()
    out["is_error_probe"] = (out["prediction_probe"] != out["probe_label"]).astype(np.int64)
    out["pred_is_original_sample"] = (out["prediction_probe"] == out["sample_label"]).astype(np.int64)
    out["pred_is_probe"] = (out["prediction_probe"] == out["probe_label"]).astype(np.int64)
    out["pred_is_silent"] = (out["prediction_probe"] == -1).astype(np.int64)
    valid_other = (
        (out["prediction_probe"] >= 0)
        & (out["prediction_probe"] < num_classes)
        & (out["prediction_probe"] != out["sample_label"])
        & (out["prediction_probe"] != out["probe_label"])
    )
    out["pred_is_other"] = valid_other.astype(np.int64)
    out["error_is_original_sample"] = ((out["is_error_probe"] == 1) & (out["pred_is_original_sample"] == 1)).astype(np.int64)
    out["error_is_silent"] = ((out["is_error_probe"] == 1) & (out["pred_is_silent"] == 1)).astype(np.int64)
    out["error_is_other"] = ((out["is_error_probe"] == 1) & (out["pred_is_other"] == 1)).astype(np.int64)
    out["error_is_probe"] = 0
    return out


def compute_error_destination_summary(df_subset: pd.DataFrame) -> Dict[str, float]:
    err = df_subset[df_subset["is_error_probe"] == 1]
    if len(err) == 0:
        return {
            "error_to_sample_fraction": 0.0,
            "error_to_probe_fraction": 0.0,
            "error_to_silent_fraction": 0.0,
            "error_to_other_fraction": 0.0,
            "n_error": 0,
        }
    return {
        "error_to_sample_fraction": float(err["error_is_original_sample"].mean()),
        "error_to_probe_fraction": float(err["error_is_probe"].mean()),
        "error_to_silent_fraction": float(err["error_is_silent"].mean()),
        "error_to_other_fraction": float(err["error_is_other"].mean()),
        "n_error": int(len(err)),
    }


def compute_ux_phase_metrics(
    df_ux_features: pd.DataFrame,
    condition_defs: Sequence[ConditionDef],
    decode_splits: int,
    seed: int,
    num_classes: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cond_idx, cond in enumerate(condition_defs):
        for layer_idx, layer_key in enumerate(PHASE_LATENT_LAYERS):
            for phase_idx, phase in enumerate(PHASE_ORDER):
                sub = (
                    df_ux_features[
                        (df_ux_features["condition"] == cond.condition)
                        & (df_ux_features["layer"] == layer_key)
                        & (df_ux_features["phase"] == phase)
                    ]
                    .sort_values("trial_id")
                    .reset_index(drop=True)
                )
                if len(sub) == 0:
                    continue
                features = np.stack(sub["feature"].tolist(), axis=0).astype(np.float32, copy=False)
                labels = sub["sample_label"].to_numpy(dtype=np.int64, copy=False)
                splits = build_stratified_splits(
                    labels=labels,
                    n_splits=decode_splits,
                    test_ratio=0.3,
                    seed=seed + 200 + cond_idx * 10 + phase_idx + layer_idx * 100,
                )
                decode_acc = decode_accuracy_with_splits(x=features, y=labels, splits=splits, num_classes=num_classes, device=device)
                fisher_stats = compute_multiclass_fisher_stats(features=features, labels=labels)
                rows.append(
                    {
                        "seed": int(seed),
                        "condition": cond.condition,
                        "condition_label": cond.condition_label,
                        "condition_family": cond.condition_family,
                        "ping_target_label": cond.ping_target_label,
                        "ping_target_frac": cond.ping_target_frac,
                        "ping_duration_ms": cond.ping_duration_ms,
                        "delay2_ms": cond.delay2_ms,
                        "regime": cond.regime,
                        "control_mode": cond.control_mode,
                        "analysis_role": cond.analysis_role,
                        "layer": layer_key,
                        "phase": phase,
                        "ux_decode_acc": float(decode_acc),
                        "between_class_var": float(fisher_stats["between_class_var"]),
                        "within_class_var": float(fisher_stats["within_class_var"]),
                        "fisher_ratio": float(fisher_stats["fisher_ratio"]),
                    }
                )
    return pd.DataFrame(rows)


def _make_shuffle_intervention(batch_meta: Dict[str, np.ndarray]) -> Callable:
    donor_idx = np.asarray(batch_meta["donor_batch_index"], dtype=np.int64)

    def _intervention(local_net, _meta: Dict[str, np.ndarray]) -> Dict[str, int]:
        apply_trial_shuffle_ux_in_place(local_net, donor_idx)
        return {"applied": 1, "n_self_swap": int(np.sum(donor_idx == np.arange(len(donor_idx), dtype=np.int64)))}

    return _intervention


def run_seed_experiment(
    net,
    encoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    ping_lookup: Dict[int, Dict[str, Dict[str, float]]],
    ping_target_fracs: Sequence[float],
    batch_size: int,
    decode_splits: int,
    seed: int,
    device: torch.device,
    include_no_ping_baseline: bool = True,
    condition_defs: Optional[Sequence[ConditionDef]] = None,
    s_sel_topk_frac: float = DEFAULT_S_SEL_TOPK_FRAC,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if condition_defs is None:
        condition_defs = build_dynamic_condition_defs(
            ping_target_fracs=ping_target_fracs,
            ping_duration_ms=float(spec.ping_ms),
            delay1_ms=float(spec.delay_ms),
            fixed_sample_to_probe_gap_ms=float(spec.delay_ms + spec.ping_ms + spec.post_ping_ms),
            include_no_ping=include_no_ping_baseline,
        )
    else:
        condition_defs = list(condition_defs)

    feature_buf: Dict[str, List[np.ndarray]] = {cond.condition: [] for cond in condition_defs}
    trial_rows: List[Dict[str, object]] = []
    regime_rows: List[Dict[str, object]] = []
    ux_feature_rows: List[Dict[str, object]] = []
    ux_timecourse_rows: List[Dict[str, object]] = []

    for start in tqdm(range(0, len(df_specs), batch_size), desc=f"Seed {seed} {spec.ping_ms:.0f}ms"):
        batch = df_specs.iloc[start : start + batch_size].copy()
        trial_ids = batch["trial_id"].to_numpy(dtype=np.int64)
        sample_indices = batch["sample_index"].to_numpy(dtype=np.int64)
        sample_labels = batch["sample_label"].to_numpy(dtype=np.int64)
        probe_labels = batch["probe_label"].to_numpy(dtype=np.int64)

        sample_imgs = torch.stack([dataset[int(i)][0] for i in sample_indices.tolist()], dim=0).to(device)
        probe_imgs = torch.stack([dataset[int(i)][0] for i in batch["probe_index"].tolist()], dim=0).to(device)
        sample_spikes = encode_images(encoder, sample_imgs, spec.sample_steps)
        probe_spikes = encode_images(encoder, probe_imgs, spec.probe_steps)

        donor_idx_b, plan_info = build_trial_shuffle_plan(sample_labels=sample_labels, probe_labels=probe_labels, rng=random.Random(seed + start + 991))
        batch_meta = {
            "trial_id": trial_ids,
            "sample_label": sample_labels,
            "probe_label": probe_labels,
            "donor_batch_index": donor_idx_b,
        }

        for cond in condition_defs:
            if not np.isclose(cond.ping_duration_ms, float(spec.ping_ms)):
                raise ValueError(f"Condition duration {cond.ping_duration_ms} ms does not match spec duration {spec.ping_ms} ms.")
            if cond.control_mode == CONTROL_DYNAMIC_NO_PING:
                ping_amp_np = np.zeros(len(batch), dtype=np.float32)
                selection_method = "no_ping"
                calibration_achieved = np.zeros(len(batch), dtype=np.float32)
                calibration_D_L1 = np.zeros(len(batch), dtype=np.float32)
                activation_residual = np.zeros(len(batch), dtype=np.float32)
            else:
                ping_amp_np = np.array(
                    [float(ping_lookup[int(sample_index)][cond.ping_target_label]["ping_drive_amp"]) for sample_index in sample_indices.tolist()],
                    dtype=np.float32,
                )
                selection_method = "per_example_fixed_activation_target"
                calibration_achieved = np.array(
                    [float(ping_lookup[int(sample_index)][cond.ping_target_label]["achieved_activation_frac"]) for sample_index in sample_indices.tolist()],
                    dtype=np.float32,
                )
                calibration_D_L1 = np.array(
                    [float(ping_lookup[int(sample_index)][cond.ping_target_label].get("D_L1", 0.0)) for sample_index in sample_indices.tolist()],
                    dtype=np.float32,
                )
                activation_residual = np.array(
                    [float(ping_lookup[int(sample_index)][cond.ping_target_label]["activation_residual"]) for sample_index in sample_indices.tolist()],
                    dtype=np.float32,
                )

            ping_amp_t = torch.as_tensor(ping_amp_np, dtype=torch.float32, device=device)
            intervention_fn = _make_shuffle_intervention(batch_meta) if cond.shuffle_ux else None
            with torch.no_grad():
                out = run_ping_probe_session(
                    net=net,
                    sample_spikes=sample_spikes,
                    probe_spikes=probe_spikes,
                    spec=spec,
                    ping_amp=ping_amp_t,
                    stsp_mode=cond.stsp_mode,
                    intervention_fn=intervention_fn,
                    batch_meta=batch_meta,
                    zero_sample=cond.zero_sample,
                )
            pred_ping = out["prediction_ping"].numpy().astype(np.int64, copy=False)
            fire_ping = out["first_fire_t_ping"].numpy().astype(np.int64, copy=False)
            pred_probe = out["prediction_probe"].numpy().astype(np.int64, copy=False)
            fire_probe = out["first_fire_t_probe"].numpy().astype(np.int64, copy=False)
            ping_metrics = out["ping_metrics"]
            feature_buf[cond.condition].append(np.asarray(ping_metrics["layer3_features"], dtype=np.float32))
            l1_ping_spike_count_vector = np.asarray(ping_metrics["layer1"]["spike_count_vector"], dtype=np.float32)
            sample_l1_spike_count_vector = np.asarray(ping_metrics["sample_l1_spike_count_vector"], dtype=np.float32)
            s_sel_metrics = compute_sample_selective_spike_fraction(sample_spike_count_vector=sample_l1_spike_count_vector, ping_spike_count_vector=l1_ping_spike_count_vector, topk_frac=s_sel_topk_frac)
            n_l1 = float(max(1, l1_ping_spike_count_vector.shape[1]))
            D_L1 = (np.asarray(ping_metrics["layer1"]["spike_count_total"], dtype=np.float32) / n_l1).astype(np.float32, copy=False)
            S_L1 = np.asarray(s_sel_metrics["S_L1"], dtype=np.float32)
            E_sel_dose = (D_L1 * S_L1).astype(np.float32, copy=False)
            selectivity = compute_sample_aligned_selectivity(np.asarray(ping_metrics["layer3_class_counts"], dtype=np.float32), sample_labels)

            phase_state_map = {
                "pre_ping": out["pre_ping_ux"],
                "post_ping_immediate": out["post_ping_immediate_ux"],
                "pre_probe": out["pre_probe_ux"],
            }
            for layer_key in PHASE_LATENT_LAYERS:
                for phase, state_dict in phase_state_map.items():
                    gain = np.asarray(state_dict[layer_key]["gain"], dtype=np.float32)
                    for i in range(len(batch)):
                        ux_feature_rows.append(
                            {
                                "seed": int(seed),
                                "condition": cond.condition,
                                "condition_label": cond.condition_label,
                                "condition_family": cond.condition_family,
                                "ping_target_label": cond.ping_target_label,
                                "ping_target_frac": cond.ping_target_frac,
                                "ping_duration_ms": cond.ping_duration_ms,
                                "delay2_ms": cond.delay2_ms,
                                "regime": cond.regime,
                                "control_mode": cond.control_mode,
                                "analysis_role": cond.analysis_role,
                                "layer": layer_key,
                                "phase": phase,
                                "trial_id": int(trial_ids[i]),
                                "sample_label": int(sample_labels[i]),
                                "feature": gain[i].astype(np.float32, copy=False),
                            }
                        )

            for timepoint in out["ux_timecourse"]:
                for layer_key in LAYER_KEYS:
                    for i in range(len(batch)):
                        ux_timecourse_rows.append(
                            {
                                "seed": int(seed),
                                "condition": cond.condition,
                                "condition_label": cond.condition_label,
                                "condition_family": cond.condition_family,
                                "ping_target_label": cond.ping_target_label,
                                "ping_target_frac": cond.ping_target_frac,
                                "ping_duration_ms": cond.ping_duration_ms,
                                "delay2_ms": cond.delay2_ms,
                                "regime": cond.regime,
                                "control_mode": cond.control_mode,
                                "analysis_role": cond.analysis_role,
                                "trial_id": int(trial_ids[i]),
                                "layer": layer_key,
                                "phase": str(timepoint["phase"]),
                                "phase_step": int(timepoint["phase_step"]),
                                "time_step": int(timepoint["global_time_step"]),
                                "time_ms": float(timepoint["time_ms"]),
                                "u_value": float(timepoint["layer_means"][layer_key]["u"][i]),
                                "x_value": float(timepoint["layer_means"][layer_key]["x"][i]),
                                "ux_value": float(timepoint["layer_means"][layer_key]["ux"][i]),
                            }
                        )

            for layer_key in LAYER_KEYS:
                active_fraction_t = np.asarray(ping_metrics[layer_key]["active_fraction_t"], dtype=np.float32)
                per_step_peak = np.zeros(len(batch), dtype=np.float32) if active_fraction_t.size == 0 else active_fraction_t.max(axis=0).astype(np.float32, copy=False)
                for i in range(len(batch)):
                    regime_rows.append(
                        {
                            "seed": int(seed),
                            "condition": cond.condition,
                            "condition_label": cond.condition_label,
                            "condition_family": cond.condition_family,
                            "ping_target_label": cond.ping_target_label,
                            "ping_target_frac": cond.ping_target_frac,
                            "ping_duration_ms": cond.ping_duration_ms,
                            "delay2_ms": cond.delay2_ms,
                            "regime": cond.regime,
                            "control_mode": cond.control_mode,
                            "analysis_role": cond.analysis_role,
                            "trial_id": int(trial_ids[i]),
                            "sample_index": int(sample_indices[i]),
                            "layer": layer_key,
                            "activation_fraction": float(ping_metrics[layer_key]["activation_fraction"][i]),
                            "spike_rate_hz": float(ping_metrics[layer_key]["spike_rate_hz"][i]),
                            "active_fraction": float(ping_metrics[layer_key]["active_fraction"][i]),
                            "spike_count": float(ping_metrics[layer_key]["spike_count_total"][i]),
                            "integrated_active_fraction_ms": float(ping_metrics[layer_key]["integrated_active_fraction_ms"][i]),
                            "per_step_active_peak": float(per_step_peak[i]),
                        }
                    )

            for i in range(len(batch)):
                achieved_l1_activation = float(ping_metrics["layer1"]["activation_fraction"][i])
                integrated_current = float(ping_amp_np[i] * cond.ping_duration_ms)
                intervention_record = out.get("intervention_record", {"applied": 0})
                trial_rows.append(
                    {
                        "seed": int(seed),
                        "trial_id": int(trial_ids[i]),
                        "sample_index": int(sample_indices[i]),
                        "sample_label": int(sample_labels[i]),
                        "probe_label": int(probe_labels[i]),
                        "condition": cond.condition,
                        "condition_label": cond.condition_label,
                        "condition_family": cond.condition_family,
                        "ping_target_label": cond.ping_target_label,
                        "ping_target_frac": cond.ping_target_frac,
                        "ping_duration_ms": cond.ping_duration_ms,
                        "delay2_ms": cond.delay2_ms,
                        "regime": cond.regime,
                        "control_mode": cond.control_mode,
                        "analysis_role": cond.analysis_role,
                        "stsp_mode": cond.stsp_mode,
                        "zero_sample": int(cond.zero_sample),
                        "shuffle_ux": int(cond.shuffle_ux),
                        "shuffle_ux_applied": int(intervention_record.get("applied", 0)),
                        "shuffle_n_self_swap": int(intervention_record.get("n_self_swap", plan_info.get("n_self_swap", 0))),
                        "ping_drive_amp": float(ping_amp_np[i]),
                        "selection_method": selection_method,
                        "selected_calibration_achieved_activation_frac": float(calibration_achieved[i]),
                        "selected_calibration_D_L1": float(calibration_D_L1[i]),
                        "selected_activation_residual": float(activation_residual[i]),
                        "prediction_ping": int(pred_ping[i]),
                        "first_fire_t_ping": int(fire_ping[i]),
                        "prediction_probe": int(pred_probe[i]),
                        "first_fire_t_probe": int(fire_probe[i]),
                        "is_correct_ping": int(pred_ping[i] == sample_labels[i]),
                        "is_silent_ping": int(pred_ping[i] == -1),
                        "is_correct_probe": int(pred_probe[i] == probe_labels[i]),
                        "is_silent_probe": int(pred_probe[i] == -1),
                        "sample_aligned_selectivity": float(selectivity[i]),
                        "D_L1": float(D_L1[i]),
                        "S_L1": float(S_L1[i]),
                        "E_sel_dose": float(E_sel_dose[i]),
                        "d_int": float(D_L1[i]),
                        "s_sel": float(S_L1[i]),
                        "s_sel_topk": int(s_sel_metrics["topk_count"][i]),
                        "ping_spikes_topk": float(s_sel_metrics["ping_topk_spikes"][i]),
                        "ping_spikes_non_topk": float(s_sel_metrics["ping_non_topk_spikes"][i]),
                        "sample_relevance_topk_mean": float(s_sel_metrics["topk_relevance_mean"][i]),
                        "sample_relevance_non_topk_mean": float(s_sel_metrics["non_topk_relevance_mean"][i]),
                        "l1_ping_activation_fraction": achieved_l1_activation,
                        "achieved_activation_frac": achieved_l1_activation,
                        "achieved_l1_activation_fraction": achieved_l1_activation,
                        "dose_l1_activation_ms": achieved_l1_activation * cond.ping_duration_ms,
                        "integrated_perturbation_current": integrated_current,
                        "l1_ping_spike_rate_hz": float(ping_metrics["layer1"]["spike_rate_hz"][i]),
                        "l2_ping_spike_rate_hz": float(ping_metrics["layer2"]["spike_rate_hz"][i]),
                        "l3_ping_spike_rate_hz": float(ping_metrics["layer3"]["spike_rate_hz"][i]),
                        "l1_ping_active_fraction": float(ping_metrics["layer1"]["active_fraction"][i]),
                        "l2_ping_active_fraction": float(ping_metrics["layer2"]["active_fraction"][i]),
                        "l3_ping_active_fraction": float(ping_metrics["layer3"]["active_fraction"][i]),
                        "l1_ping_spike_count": float(ping_metrics["layer1"]["spike_count_total"][i]),
                        "l2_ping_spike_count": float(ping_metrics["layer2"]["spike_count_total"][i]),
                        "l3_ping_spike_count": float(ping_metrics["layer3"]["spike_count_total"][i]),
                        "l2_integrated_active_fraction_ms": float(ping_metrics["layer2"]["integrated_active_fraction_ms"][i]),
                        "l3_integrated_active_fraction_ms": float(ping_metrics["layer3"]["integrated_active_fraction_ms"][i]),
                    }
                )

    df_trials = pd.DataFrame(trial_rows).sort_values(["seed", "trial_id", "condition"]).reset_index(drop=True)
    df_trials = build_error_destination_columns(df_trials, num_classes=net.layer3.num_classes)
    df_regime_trials = pd.DataFrame(regime_rows).sort_values(["seed", "trial_id", "condition", "layer"]).reset_index(drop=True)
    df_ux_features = pd.DataFrame(ux_feature_rows)
    df_ux_timecourse = pd.DataFrame(ux_timecourse_rows).sort_values(["seed", "condition", "trial_id", "layer", "time_step"]).reset_index(drop=True)

    metrics_rows: List[Dict[str, object]] = []
    for cond in condition_defs:
        features = np.concatenate(feature_buf[cond.condition], axis=0) if feature_buf[cond.condition] else np.zeros((0, 0), dtype=np.float32)
        sub = df_trials[df_trials["condition"] == cond.condition].copy()
        bias_sample, bias_noise = compute_sample_and_noise_bias(sub, num_classes=net.layer3.num_classes)
        err_stats = compute_error_destination_summary(sub)
        metrics_rows.append(
            {
                "seed": int(seed),
                "condition": cond.condition,
                "condition_label": cond.condition_label,
                "condition_family": cond.condition_family,
                "ping_target_label": cond.ping_target_label,
                "ping_target_frac": cond.ping_target_frac,
                "ping_duration_ms": cond.ping_duration_ms,
                "delay2_ms": cond.delay2_ms,
                "regime": cond.regime,
                "control_mode": cond.control_mode,
                "analysis_role": cond.analysis_role,
                "stsp_mode": cond.stsp_mode,
                "zero_sample": int(cond.zero_sample),
                "shuffle_ux": int(cond.shuffle_ux),
                "ping_drive_amp_mean": float(np.mean(sub["ping_drive_amp"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "decode_ping_acc": float(
                    decode_accuracy_with_splits(
                        x=features,
                        y=sub["sample_label"].to_numpy(dtype=np.int64),
                        splits=build_stratified_splits(
                            labels=sub["sample_label"].to_numpy(dtype=np.int64),
                            n_splits=decode_splits,
                            test_ratio=0.3,
                            seed=seed + 131,
                        ),
                        num_classes=net.layer3.num_classes,
                        device=device,
                    )
                ) if len(sub) else 0.0,
                "probe_accuracy": 100.0 * float(np.mean(sub["is_correct_probe"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "probe_silent_rate": float(np.mean(sub["is_silent_probe"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "sample_bias": float(bias_sample),
                "noise_bias": float(bias_noise),
                "sample_bias_minus_noise": float(bias_sample - bias_noise),
                "sample_aligned_selectivity": float(np.mean(sub["sample_aligned_selectivity"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "D_L1": float(np.mean(sub["D_L1"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "S_L1": float(np.mean(sub["S_L1"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "E_sel_dose": float(np.mean(sub["E_sel_dose"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "d_int": float(np.mean(sub["d_int"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "s_sel": float(np.mean(sub["s_sel"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "error_to_sample_fraction": float(err_stats["error_to_sample_fraction"]),
                "error_to_probe_fraction": float(err_stats["error_to_probe_fraction"]),
                "error_to_silent_fraction": float(err_stats["error_to_silent_fraction"]),
                "error_to_other_fraction": float(err_stats["error_to_other_fraction"]),
                "achieved_activation_frac": float(np.mean(sub["l1_ping_activation_fraction"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "achieved_l1_activation_fraction": float(np.mean(sub["achieved_l1_activation_fraction"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "dose_l1_activation_ms": float(np.mean(sub["dose_l1_activation_ms"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "integrated_perturbation_current": float(np.mean(sub["integrated_perturbation_current"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "ping_spikes_topk": float(np.mean(sub["ping_spikes_topk"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "ping_spikes_non_topk": float(np.mean(sub["ping_spikes_non_topk"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "sample_relevance_topk_mean": float(np.mean(sub["sample_relevance_topk_mean"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "sample_relevance_non_topk_mean": float(np.mean(sub["sample_relevance_non_topk_mean"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "l1_ping_spike_rate_hz": float(np.mean(sub["l1_ping_spike_rate_hz"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "l2_ping_spike_rate_hz": float(np.mean(sub["l2_ping_spike_rate_hz"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "l3_ping_spike_rate_hz": float(np.mean(sub["l3_ping_spike_rate_hz"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "l1_ping_active_fraction": float(np.mean(sub["l1_ping_active_fraction"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "l2_ping_active_fraction": float(np.mean(sub["l2_ping_active_fraction"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "l3_ping_active_fraction": float(np.mean(sub["l3_ping_active_fraction"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "l1_ping_spike_count": float(np.mean(sub["l1_ping_spike_count"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "l2_ping_spike_count": float(np.mean(sub["l2_ping_spike_count"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "l3_ping_spike_count": float(np.mean(sub["l3_ping_spike_count"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "l2_integrated_active_fraction_ms": float(np.mean(sub["l2_integrated_active_fraction_ms"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
                "l3_integrated_active_fraction_ms": float(np.mean(sub["l3_integrated_active_fraction_ms"].to_numpy(dtype=np.float64))) if len(sub) else 0.0,
            }
        )

    df_ux_phase = compute_ux_phase_metrics(df_ux_features=df_ux_features, condition_defs=condition_defs, decode_splits=decode_splits, seed=seed, num_classes=net.layer3.num_classes, device=device)
    return df_trials, pd.DataFrame(metrics_rows), df_regime_trials, df_ux_phase, df_ux_timecourse


def add_baseline_deltas(df_metrics_seed: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    base = (
        df_metrics_seed[df_metrics_seed["control_mode"] == CONTROL_DYNAMIC_NO_PING][["seed"] + list(metric_cols)]
        .rename(columns={col: f"{col}_baseline" for col in metric_cols})
        .drop_duplicates(subset=["seed"])
        .reset_index(drop=True)
    )
    merged = df_metrics_seed.merge(base, on="seed", how="left")
    for col in metric_cols:
        merged[f"delta_{col}"] = merged[col] - merged[f"{col}_baseline"]
    return merged


def add_phase_baseline_deltas(df_phase_seed: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    base = (
        df_phase_seed[df_phase_seed["control_mode"] == CONTROL_DYNAMIC_NO_PING][["seed", "layer", "phase"] + list(metric_cols)]
        .rename(columns={col: f"{col}_baseline" for col in metric_cols})
        .drop_duplicates(subset=["seed", "layer", "phase"])
        .reset_index(drop=True)
    )
    merged = df_phase_seed.merge(base, on=["seed", "layer", "phase"], how="left")
    for col in metric_cols:
        merged[f"delta_{col}"] = merged[col] - merged[f"{col}_baseline"]
    return merged


def compute_delta_summary(df_metrics_seed: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "decode_ping_acc",
        "sample_bias_minus_noise",
        "probe_accuracy",
        "error_to_sample_fraction",
        "D_L1",
        "S_L1",
        "E_sel_dose",
        "l2_ping_spike_count",
        "l2_ping_active_fraction",
        "l3_ping_spike_count",
        "l3_ping_active_fraction",
    ]
    delta_cols = [f"delta_{col}" for col in metric_cols]
    merged = df_metrics_seed.copy()
    if any(col not in merged.columns for col in delta_cols):
        merged = add_baseline_deltas(merged, metric_cols=metric_cols)
    group_cols = [
        "condition",
        "condition_label",
        "condition_family",
        "regime",
        "control_mode",
        "analysis_role",
        "ping_duration_ms",
        "delay2_ms",
        "ping_target_label",
        "ping_target_frac",
    ]
    return (
        merged.groupby(group_cols, as_index=False)
        .agg(
            achieved_activation_frac=("achieved_activation_frac", "mean"),
            achieved_activation_frac_sem=("achieved_activation_frac", "sem"),
            decode_ping_acc_mean=("decode_ping_acc", "mean"),
            decode_ping_acc_sem=("decode_ping_acc", "sem"),
            delta_decode_ping_acc_mean=("delta_decode_ping_acc", "mean"),
            delta_decode_ping_acc_sem=("delta_decode_ping_acc", "sem"),
            delta_sample_bias_minus_noise_mean=("delta_sample_bias_minus_noise", "mean"),
            delta_sample_bias_minus_noise_sem=("delta_sample_bias_minus_noise", "sem"),
            delta_probe_accuracy_mean=("delta_probe_accuracy", "mean"),
            delta_probe_accuracy_sem=("delta_probe_accuracy", "sem"),
            delta_error_to_sample_fraction_mean=("delta_error_to_sample_fraction", "mean"),
            delta_error_to_sample_fraction_sem=("delta_error_to_sample_fraction", "sem"),
            D_L1_mean=("D_L1", "mean"),
            D_L1_sem=("D_L1", "sem"),
            S_L1_mean=("S_L1", "mean"),
            S_L1_sem=("S_L1", "sem"),
            E_sel_dose_mean=("E_sel_dose", "mean"),
            E_sel_dose_sem=("E_sel_dose", "sem"),
            l2_ping_spike_count_mean=("l2_ping_spike_count", "mean"),
            l2_ping_spike_count_sem=("l2_ping_spike_count", "sem"),
            l2_ping_active_fraction_mean=("l2_ping_active_fraction", "mean"),
            l2_ping_active_fraction_sem=("l2_ping_active_fraction", "sem"),
            l3_ping_spike_count_mean=("l3_ping_spike_count", "mean"),
            l3_ping_spike_count_sem=("l3_ping_spike_count", "sem"),
            l3_ping_active_fraction_mean=("l3_ping_active_fraction", "mean"),
            l3_ping_active_fraction_sem=("l3_ping_active_fraction", "sem"),
            d_int_mean=("d_int", "mean"),
            d_int_sem=("d_int", "sem"),
            s_sel_mean=("s_sel", "mean"),
            s_sel_sem=("s_sel", "sem"),
        )
        .fillna(0.0)
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def build_main_grid_seed_summary(df_metrics_seed: pd.DataFrame, df_phase_seed: pd.DataFrame) -> pd.DataFrame:
    phase_value_cols = [
        "ux_decode_acc",
        "delta_ux_decode_acc",
        "between_class_var",
        "delta_between_class_var",
        "within_class_var",
        "delta_within_class_var",
        "fisher_ratio",
        "delta_fisher_ratio",
    ]
    id_cols = [
        "seed",
        "condition",
        "condition_label",
        "condition_family",
        "ping_target_label",
        "ping_target_frac",
        "ping_duration_ms",
        "delay2_ms",
        "regime",
        "control_mode",
        "analysis_role",
    ]
    pivot_parts: List[pd.DataFrame] = []
    for metric_col in phase_value_cols:
        sub = df_phase_seed[id_cols + ["layer", "phase", metric_col]].copy()
        sub["phase_metric"] = sub["layer"].astype(str) + "__" + sub["phase"].astype(str) + "__" + metric_col
        pivot = sub.pivot_table(index=id_cols, columns="phase_metric", values=metric_col, aggfunc="first").reset_index()
        pivot.columns.name = None
        pivot_parts.append(pivot)
    out = df_metrics_seed.copy()
    for pivot in pivot_parts:
        out = out.merge(pivot, on=id_cols, how="left")
    return out


def summarize_numeric_frame(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=list(group_cols))
    numeric_cols = [
        col
        for col in df.columns
        if col not in group_cols and pd.api.types.is_numeric_dtype(df[col]) and col not in {"seed", "trial_id", "sample_index"}
    ]
    return summarize_metrics(df, group_cols=group_cols, value_cols=numeric_cols)


def summarize_ux_timecourse(df_ux_timecourse: pd.DataFrame) -> pd.DataFrame:
    if len(df_ux_timecourse) == 0:
        return pd.DataFrame()
    grouped = (
        df_ux_timecourse.groupby(
            [
                "condition",
                "condition_label",
                "condition_family",
                "ping_target_label",
                "ping_target_frac",
                "ping_duration_ms",
                "delay2_ms",
                "regime",
                "control_mode",
                "analysis_role",
                "layer",
                "phase",
                "phase_step",
                "time_step",
                "time_ms",
            ],
            as_index=False,
        )
        .agg(
            u_mean=("u_value", "mean"),
            u_sem=("u_value", "sem"),
            x_mean=("x_value", "mean"),
            x_sem=("x_value", "sem"),
            ux_mean=("ux_value", "mean"),
            ux_sem=("ux_value", "sem"),
            n_trials=("trial_id", "nunique"),
        )
        .fillna({"u_sem": 0.0, "x_sem": 0.0, "ux_sem": 0.0})
        .sort_values(["condition", "layer", "time_step"])
        .reset_index(drop=True)
    )
    return grouped


def build_phase_component_summary(df_ux_timecourse_summary: pd.DataFrame) -> pd.DataFrame:
    if len(df_ux_timecourse_summary) == 0:
        return pd.DataFrame()
    snapshot_plan = [
        ("pre_ping", "delay1"),
        ("post_ping_immediate", "ping"),
        ("pre_probe", "delay2"),
    ]
    id_cols = [
        "condition",
        "condition_label",
        "condition_family",
        "ping_target_label",
        "ping_target_frac",
        "ping_duration_ms",
        "delay2_ms",
        "regime",
        "control_mode",
        "analysis_role",
        "layer",
    ]
    rows: List[pd.DataFrame] = []
    for phase_snapshot, source_phase in snapshot_plan:
        sub = df_ux_timecourse_summary[df_ux_timecourse_summary["phase"] == source_phase].copy()
        if len(sub) == 0:
            continue
        sub = (
            sub.sort_values(id_cols + ["phase_step", "time_step"])
            .groupby(id_cols, as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        sub["phase_snapshot"] = phase_snapshot
        sub["source_phase"] = source_phase
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["control_mode", "regime", "ping_duration_ms", "ping_target_frac", "layer", "phase_snapshot"]).reset_index(drop=True)


def _latent_preprobe_columns(df_grid_summary: pd.DataFrame) -> List[str]:
    cols = [
        "layer2__pre_probe__delta_ux_decode_acc_mean",
        "layer3__pre_probe__delta_ux_decode_acc_mean",
        "layer2__pre_probe__delta_fisher_ratio_mean",
        "layer3__pre_probe__delta_fisher_ratio_mean",
    ]
    return [col for col in cols if col in df_grid_summary.columns]


def _latent_immediate_columns(df_grid_summary: pd.DataFrame) -> List[str]:
    cols = [
        "layer2__post_ping_immediate__delta_ux_decode_acc_mean",
        "layer3__post_ping_immediate__delta_ux_decode_acc_mean",
        "layer2__post_ping_immediate__delta_fisher_ratio_mean",
        "layer3__post_ping_immediate__delta_fisher_ratio_mean",
    ]
    return [col for col in cols if col in df_grid_summary.columns]


def add_composite_summary_metrics(df_grid_summary: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    out = df_grid_summary.copy()
    chance = 1.0 / float(num_classes)
    latent_cols = _latent_preprobe_columns(out)
    immediate_cols = _latent_immediate_columns(out)
    out["preprobe_latent_gain_mean"] = out[latent_cols].max(axis=1) if latent_cols else 0.0
    out["immediate_latent_gain_mean"] = out[immediate_cols].max(axis=1) if immediate_cols else 0.0
    out["overt_threshold_flag"] = (
        (out["regime"] == REGIME_LONG)
        & (out["control_mode"] == CONTROL_DYNAMIC_PING)
        & (out["decode_ping_acc_mean"] > chance + 0.10)
    ).astype(np.int64)
    out["beneficial_short_flag"] = (
        (out["regime"] == REGIME_SHORT)
        & (out["control_mode"] == CONTROL_DYNAMIC_PING)
        & (out["delta_sample_bias_minus_noise_mean"] > 0.0)
        & (out["preprobe_latent_gain_mean"] > 0.0)
    ).astype(np.int64)
    out["low_burden_short_flag"] = (out["l2_ping_spike_count_mean"] <= out["l2_ping_spike_count_mean"].median()).astype(np.int64)
    out["reorganization_flag"] = (
        (out["regime"] == REGIME_LONG)
        & (out["decode_ping_acc_mean"] > chance + 0.10)
        & ((out["immediate_latent_gain_mean"].abs() > 1e-8) | (out["preprobe_latent_gain_mean"].abs() > 1e-8))
        & ((out["delta_sample_bias_minus_noise_mean"].abs() > 1e-8) | (out["delta_probe_accuracy_mean"].abs() > 1e-8))
    ).astype(np.int64)
    return out


def build_same_dose_pairs(df_grid_summary: pd.DataFrame, max_relative_gap: float = 0.10) -> pd.DataFrame:
    dynamic = df_grid_summary[df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING].copy()
    short = dynamic[dynamic["regime"] == REGIME_SHORT].copy()
    long = dynamic[dynamic["regime"] == REGIME_LONG].copy()
    if len(short) == 0 or len(long) == 0:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    short = short.sort_values(["D_L1_mean", "ping_duration_ms", "ping_target_frac"]).reset_index(drop=True)
    long = long.sort_values(["D_L1_mean", "ping_duration_ms", "ping_target_frac"]).reset_index(drop=True)
    long_dose = long["D_L1_mean"].to_numpy(dtype=np.float64)
    for _, s_row in short.iterrows():
        short_dose = float(s_row["D_L1_mean"])
        denom = np.maximum(np.maximum(np.abs(long_dose), abs(short_dose)), 1e-9)
        rel_gap = pd.Series(np.abs(long_dose - short_dose) / denom, index=long.index)
        idx = int(np.argmin(rel_gap.to_numpy(dtype=np.float64)))
        if float(rel_gap.iloc[idx]) > max_relative_gap:
            continue
        l_row = long.iloc[idx]
        rows.append(
            {
                "pair_kind": "same_dose",
                "duration_short_ms": float(s_row["ping_duration_ms"]),
                "strength_short": float(s_row["ping_target_frac"]),
                "D_L1_short": float(s_row["D_L1_mean"]),
                "duration_long_ms": float(l_row["ping_duration_ms"]),
                "strength_long": float(l_row["ping_target_frac"]),
                "D_L1_long": float(l_row["D_L1_mean"]),
                "relative_dose_gap": float(rel_gap.iloc[idx]),
                "delta_sample_bias_short": float(s_row["delta_sample_bias_minus_noise_mean"]),
                "delta_sample_bias_long": float(l_row["delta_sample_bias_minus_noise_mean"]),
                "decode_short": float(s_row["decode_ping_acc_mean"]),
                "decode_long": float(l_row["decode_ping_acc_mean"]),
                "preprobe_latent_short": float(s_row.get("preprobe_latent_gain_mean", 0.0)),
                "preprobe_latent_long": float(l_row.get("preprobe_latent_gain_mean", 0.0)),
                "S_L1_short": float(s_row["S_L1_mean"]),
                "S_L1_long": float(l_row["S_L1_mean"]),
                "l2_spike_count_short": float(s_row["l2_ping_spike_count_mean"]),
                "l2_spike_count_long": float(l_row["l2_ping_spike_count_mean"]),
                "l2_active_fraction_short": float(s_row["l2_ping_active_fraction_mean"]),
                "l2_active_fraction_long": float(l_row["l2_ping_active_fraction_mean"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["relative_dose_gap", "duration_short_ms", "strength_short", "duration_long_ms", "strength_long"]).reset_index(drop=True)
    out.insert(0, "pair_id", [f"same_dose_{i:02d}" for i in range(len(out))])
    return out


def build_same_burden_pairs(df_grid_summary: pd.DataFrame, max_relative_gap: float = 0.15) -> pd.DataFrame:
    dynamic = df_grid_summary[df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING].copy()
    short = dynamic[dynamic["regime"] == REGIME_SHORT].copy()
    long = dynamic[dynamic["regime"] == REGIME_LONG].copy()
    if len(short) == 0 or len(long) == 0:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    long_l2_spike = long["l2_ping_spike_count_mean"].to_numpy(dtype=np.float64)
    long_l2_frac = long["l2_ping_active_fraction_mean"].to_numpy(dtype=np.float64)
    for _, s_row in short.iterrows():
        short_l2_spike = float(s_row["l2_ping_spike_count_mean"])
        short_l2_frac = float(s_row["l2_ping_active_fraction_mean"])
        spike_denom = np.maximum(np.maximum(np.abs(long_l2_spike), abs(short_l2_spike)), 1e-9)
        frac_denom = np.maximum(np.maximum(np.abs(long_l2_frac), abs(short_l2_frac)), 1e-9)
        spike_gap = pd.Series(np.abs(long_l2_spike - short_l2_spike) / spike_denom, index=long.index)
        frac_gap = pd.Series(np.abs(long_l2_frac - short_l2_frac) / frac_denom, index=long.index)
        combined_gap = 0.5 * (spike_gap + frac_gap)
        idx = int(np.argmin(combined_gap.to_numpy(dtype=np.float64)))
        if float(combined_gap.iloc[idx]) > max_relative_gap:
            continue
        l_row = long.iloc[idx]
        rows.append(
            {
                "pair_kind": "same_burden",
                "duration_short_ms": float(s_row["ping_duration_ms"]),
                "strength_short": float(s_row["ping_target_frac"]),
                "duration_long_ms": float(l_row["ping_duration_ms"]),
                "strength_long": float(l_row["ping_target_frac"]),
                "combined_burden_gap": float(combined_gap.iloc[idx]),
                "l2_spike_count_short": float(s_row["l2_ping_spike_count_mean"]),
                "l2_spike_count_long": float(l_row["l2_ping_spike_count_mean"]),
                "l2_active_fraction_short": float(s_row["l2_ping_active_fraction_mean"]),
                "l2_active_fraction_long": float(l_row["l2_ping_active_fraction_mean"]),
                "delta_sample_bias_short": float(s_row["delta_sample_bias_minus_noise_mean"]),
                "delta_sample_bias_long": float(l_row["delta_sample_bias_minus_noise_mean"]),
                "decode_short": float(s_row["decode_ping_acc_mean"]),
                "decode_long": float(l_row["decode_ping_acc_mean"]),
                "D_L1_short": float(s_row["D_L1_mean"]),
                "D_L1_long": float(l_row["D_L1_mean"]),
                "S_L1_short": float(s_row["S_L1_mean"]),
                "S_L1_long": float(l_row["S_L1_mean"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["combined_burden_gap", "duration_short_ms", "strength_short", "duration_long_ms", "strength_long"]).reset_index(drop=True)
    out.insert(0, "pair_id", [f"same_burden_{i:02d}" for i in range(len(out))])
    return out


def select_focused_long_control_targets(df_grid_summary: pd.DataFrame, num_classes: int, delay1_ms: float, fixed_gap_ms: float) -> List[ConditionDef]:
    chance = 1.0 / float(num_classes)
    dynamic_long = df_grid_summary[(df_grid_summary["regime"] == REGIME_LONG) & (df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING)].copy()
    if len(dynamic_long) == 0:
        return []
    rows: List[pd.Series] = []
    for duration_ms, sub in dynamic_long.groupby("ping_duration_ms", sort=True):
        overt = sub[sub["decode_ping_acc_mean"] > chance + 0.10].sort_values(["D_L1_mean", "ping_target_frac"])
        if len(overt) > 0:
            rows.append(overt.iloc[0])
        destructive = sub.sort_values(["delta_sample_bias_minus_noise_mean", "delta_probe_accuracy_mean", "D_L1_mean"], ascending=[True, True, False])
        if len(destructive) > 0:
            rows.append(destructive.iloc[0])
    unique_keys = set()
    conds: List[ConditionDef] = []
    for row in rows:
        key = (float(row["ping_duration_ms"]), float(row["ping_target_frac"]))
        if key in unique_keys:
            continue
        unique_keys.add(key)
        conds.extend(
            build_dynamic_condition_defs(
                ping_target_fracs=[float(row["ping_target_frac"])],
                ping_duration_ms=float(row["ping_duration_ms"]),
                delay1_ms=delay1_ms,
                fixed_sample_to_probe_gap_ms=fixed_gap_ms,
                regime=REGIME_LONG,
                analysis_role=ROLE_CONTROL,
                include_no_ping=False,
            )
        )
    return conds


def select_representative_conditions(df_grid_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    baseline = df_grid_summary[df_grid_summary["control_mode"] == CONTROL_DYNAMIC_NO_PING]
    if len(baseline) > 0:
        rows.append({"role_name": "no_ping", **baseline.iloc[0].to_dict()})
    short_dynamic = df_grid_summary[(df_grid_summary["regime"] == REGIME_SHORT) & (df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING)].copy()
    long_dynamic = df_grid_summary[(df_grid_summary["regime"] == REGIME_LONG) & (df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING)].copy()
    if len(short_dynamic) > 0:
        beneficial = short_dynamic[(short_dynamic["delta_sample_bias_minus_noise_mean"] > 0.0) & (short_dynamic.get("preprobe_latent_gain_mean", 0.0) > 0.0)].copy()
        if len(beneficial) == 0:
            beneficial = short_dynamic.copy()
        beneficial = beneficial.sort_values(["delta_sample_bias_minus_noise_mean", "preprobe_latent_gain_mean", "l2_ping_spike_count_mean"], ascending=[False, False, True])
        rows.append({"role_name": "short_beneficial_candidate", **beneficial.iloc[0].to_dict()})
        rows.append({"role_name": "short_stronger_candidate", **short_dynamic.sort_values(["D_L1_mean", "ping_target_frac"], ascending=[False, False]).iloc[0].to_dict()})
    if len(long_dynamic) > 0:
        overt_flag_col = "overt_threshold_flag" if "overt_threshold_flag" in long_dynamic.columns else "overt_threshold_flag_mean"
        sort_cols = [col for col in [overt_flag_col, "decode_ping_acc_mean", "D_L1_mean"] if col in long_dynamic.columns]
        ascending = [False if col != "D_L1_mean" else True for col in sort_cols]
        rows.append({"role_name": "long_moderate_candidate", **long_dynamic.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()})
        rows.append({"role_name": "long_destructive_candidate", **long_dynamic.sort_values(["delta_sample_bias_minus_noise_mean", "delta_probe_accuracy_mean", "decode_ping_acc_mean"], ascending=[True, True, False]).iloc[0].to_dict()})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["role_name", "condition"]).reset_index(drop=True)


def build_selected_timecourse_summary(df_timecourse_summary: pd.DataFrame, df_representatives: pd.DataFrame) -> pd.DataFrame:
    if len(df_timecourse_summary) == 0 or len(df_representatives) == 0:
        return pd.DataFrame()
    conds = df_representatives[["condition", "role_name"]].drop_duplicates(subset=["condition", "role_name"])
    return df_timecourse_summary.merge(conds, on="condition", how="inner").sort_values(["role_name", "layer", "time_step"]).reset_index(drop=True)


def collect_sample_relevance_sanity(
    net,
    encoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    cond: ConditionDef,
    ping_lookup: Dict[int, Dict[str, Dict[str, float]]],
    batch_size: int,
    device: torch.device,
    topk_frac: float,
    max_trials: int = 24,
) -> pd.DataFrame:
    sub_specs = df_specs.iloc[: max(1, min(max_trials, len(df_specs)))].copy().reset_index(drop=True)
    sample_indices = sub_specs["sample_index"].to_numpy(dtype=np.int64)
    sample_labels = sub_specs["sample_label"].to_numpy(dtype=np.int64)
    probe_labels = sub_specs["probe_label"].to_numpy(dtype=np.int64)
    sample_imgs = torch.stack([dataset[int(i)][0] for i in sample_indices.tolist()], dim=0).to(device)
    probe_imgs = torch.stack([dataset[int(i)][0] for i in sub_specs["probe_index"].tolist()], dim=0).to(device)
    sample_spikes = encode_images(encoder, sample_imgs, spec.sample_steps)
    probe_spikes = encode_images(encoder, probe_imgs, spec.probe_steps)
    ping_amp_np = np.array([float(ping_lookup[int(sample_index)][cond.ping_target_label]["ping_drive_amp"]) for sample_index in sample_indices.tolist()], dtype=np.float32)
    ping_amp_t = torch.as_tensor(ping_amp_np, dtype=torch.float32, device=device)
    donor_idx_b, _ = build_trial_shuffle_plan(sample_labels=sample_labels, probe_labels=probe_labels, rng=random.Random(123))
    batch_meta = {"trial_id": sub_specs["trial_id"].to_numpy(dtype=np.int64), "sample_label": sample_labels, "probe_label": probe_labels, "donor_batch_index": donor_idx_b}
    intervention_fn = _make_shuffle_intervention(batch_meta) if cond.shuffle_ux else None
    with torch.no_grad():
        out = run_ping_probe_session(net=net, sample_spikes=sample_spikes, probe_spikes=probe_spikes, spec=spec, ping_amp=ping_amp_t, stsp_mode=cond.stsp_mode, intervention_fn=intervention_fn, batch_meta=batch_meta, zero_sample=cond.zero_sample)
    sample_vec = np.asarray(out["ping_metrics"]["sample_l1_spike_count_vector"], dtype=np.float32)
    ping_vec = np.asarray(out["ping_metrics"]["layer1"]["spike_count_vector"], dtype=np.float32)
    n_trials, n_neurons = sample_vec.shape
    k = max(1, min(n_neurons, int(np.ceil(float(topk_frac) * float(n_neurons)))))
    topk_idx = np.argpartition(sample_vec, kth=n_neurons - k, axis=1)[:, -k:]
    mask = np.zeros_like(sample_vec, dtype=np.int64)
    mask[np.arange(n_trials)[:, None], topk_idx] = 1
    relevance_flat = sample_vec.reshape(-1)
    mask_flat = mask.reshape(-1)
    ping_flat = ping_vec.reshape(-1)
    bins = np.linspace(float(relevance_flat.min(initial=0.0)), float(relevance_flat.max(initial=1.0)) + 1e-6, 21)
    hist_topk, edges = np.histogram(relevance_flat[mask_flat == 1], bins=bins)
    hist_non, _ = np.histogram(relevance_flat[mask_flat == 0], bins=bins)
    rows: List[Dict[str, object]] = []
    total_ping = float(max(1.0, ping_flat.sum()))
    for idx in range(len(edges) - 1):
        rows.append(
            {
                "condition": cond.condition,
                "condition_label": cond.condition_label,
                "regime": cond.regime,
                "ping_duration_ms": cond.ping_duration_ms,
                "ping_target_frac": cond.ping_target_frac,
                "bin_left": float(edges[idx]),
                "bin_right": float(edges[idx + 1]),
                "topk_count": int(hist_topk[idx]),
                "non_topk_count": int(hist_non[idx]),
                "topk_fraction": float(hist_topk[idx] / max(1, hist_topk.sum())),
                "non_topk_fraction": float(hist_non[idx] / max(1, hist_non.sum())),
                "ping_spike_share_topk": float(ping_flat[mask_flat == 1].sum() / total_ping),
                "ping_spike_share_non_topk": float(ping_flat[mask_flat == 0].sum() / total_ping),
                "topk_size": int(k),
                "n_neurons": int(n_neurons),
            }
        )
    return pd.DataFrame(rows)


def _phase_heatmap(df: pd.DataFrame, value_col: str, title: str, ax) -> None:
    if len(df) == 0 or value_col not in df.columns:
        ax.axis("off")
        return
    durations = sorted(df["ping_duration_ms"].dropna().unique().tolist())
    strengths = sorted(df["ping_target_frac"].dropna().unique().tolist())
    pivot = df.pivot_table(index="ping_target_frac", columns="ping_duration_ms", values=value_col, aggfunc="mean").reindex(index=strengths, columns=durations)
    im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", origin="lower", cmap=CMAP_DIVERGING)
    ax.set_xticks(np.arange(len(durations)))
    ax.set_xticklabels([f"{float(v):.0f}" for v in durations])
    ax.set_yticks(np.arange(len(strengths)))
    ax.set_yticklabels([f"{float(v):.3f}" for v in strengths])
    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Strength")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _save_runtime_figure(fig: plt.Figure, save_path: str) -> None:
    base_path = Path(save_path)
    save_figure_all_formats(fig, base_path.with_suffix(""))
    plt.close(fig)


def plot_short_beneficial_window(df_grid_summary: pd.DataFrame, save_path: str) -> None:
    sub = df_grid_summary[(df_grid_summary["regime"] == REGIME_SHORT) & (df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING)].copy()
    if len(sub) == 0:
        return
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL)
    for ax, color_col, title in [
        (axes[0], "S_L1_mean", "Short beneficial window colored by S_L1"),
        (axes[1], "l2_ping_spike_count_mean", "Short beneficial window colored by L2 spike count"),
    ]:
        sc = ax.scatter(sub["D_L1_mean"].to_numpy(dtype=np.float64), sub["delta_sample_bias_minus_noise_mean"].to_numpy(dtype=np.float64), c=sub[color_col].to_numpy(dtype=np.float64), cmap=CMAP_SEQUENTIAL, s=80, edgecolors="#222222", linewidths=0.4)
        ax.axhline(0.0, color="#999999", linewidth=LINE_WIDTH_REFERENCE, linestyle="--")
        ax.set_xlabel("D_L1 = total L1 spikes / N_L1")
        ax.set_ylabel("Delta sample bias - noise")
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def plot_short_latent_improvement(df_grid_summary: pd.DataFrame, save_path: str) -> None:
    sub = df_grid_summary[(df_grid_summary["regime"] == REGIME_SHORT) & (df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING)].copy()
    if len(sub) == 0:
        return
    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    panels = [
        ("layer2__pre_probe__delta_ux_decode_acc_mean", "Layer2 pre-probe u*x decode delta"),
        ("layer3__pre_probe__delta_ux_decode_acc_mean", "Layer3 pre-probe u*x decode delta"),
        ("layer2__pre_probe__delta_fisher_ratio_mean", "Layer2 pre-probe Fisher delta"),
        ("layer3__pre_probe__delta_fisher_ratio_mean", "Layer3 pre-probe Fisher delta"),
    ]
    for ax, (col, title) in zip(axes.flatten(), panels):
        if col not in sub.columns:
            ax.axis("off")
            continue
        ax.scatter(sub["D_L1_mean"].to_numpy(dtype=np.float64), sub[col].to_numpy(dtype=np.float64), c=sub["S_L1_mean"].to_numpy(dtype=np.float64), cmap=CMAP_SEQUENTIAL_ALT, s=80, edgecolors="#222222", linewidths=0.4)
        ax.axhline(0.0, color="#999999", linewidth=LINE_WIDTH_REFERENCE, linestyle="--")
        ax.set_xlabel("D_L1")
        ax.set_ylabel(title)
        ax.set_title(title)
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def plot_long_overt_reactivation(df_grid_summary: pd.DataFrame, save_path: str) -> None:
    sub = df_grid_summary[(df_grid_summary["regime"] == REGIME_LONG) & (df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING)].copy()
    if len(sub) == 0:
        return
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL)
    _phase_heatmap(sub, "decode_ping_acc_mean", "Long ping-window decode", axes[0])
    _phase_heatmap(sub, "delta_sample_bias_minus_noise_mean", "Long delta sample bias - noise", axes[1])
    latent_col = "layer3__pre_probe__delta_ux_decode_acc_mean" if "layer3__pre_probe__delta_ux_decode_acc_mean" in sub.columns else "preprobe_latent_gain_mean"
    _phase_heatmap(sub, latent_col, "Long pre-probe latent change", axes[2])
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def plot_behavior_latent_overt_dissociation(df_grid_summary: pd.DataFrame, save_path: str) -> None:
    sub = df_grid_summary[df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING].copy()
    if len(sub) == 0:
        return
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL)
    _phase_heatmap(sub, "delta_sample_bias_minus_noise_mean", "Behavior: delta sample bias - noise", axes[0])
    _phase_heatmap(sub, "delta_probe_accuracy_mean", "Behavior: delta probe accuracy", axes[1])
    _phase_heatmap(sub, "decode_ping_acc_mean", "Overt: ping-window decode", axes[2])
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def plot_dose_burden_comparison(df_grid_summary: pd.DataFrame, save_path: str) -> None:
    sub = df_grid_summary[df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING].copy()
    if len(sub) == 0:
        return
    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_TWO_BY_TWO)
    panels = [
        ("l2_ping_spike_count_mean", "delta_sample_bias_minus_noise_mean", "L2 spike count", "Delta sample bias - noise"),
        ("l2_ping_spike_count_mean", "decode_ping_acc_mean", "L2 spike count", "Ping-window decode"),
        ("S_L1_mean", "delta_sample_bias_minus_noise_mean", "S_L1", "Delta sample bias - noise"),
        ("D_L1_mean", "delta_sample_bias_minus_noise_mean", "D_L1", "Delta sample bias - noise"),
    ]
    for ax, (x_col, y_col, xlabel, ylabel) in zip(axes.flatten(), panels):
        sc = ax.scatter(sub[x_col].to_numpy(dtype=np.float64), sub[y_col].to_numpy(dtype=np.float64), c=sub["ping_duration_ms"].to_numpy(dtype=np.float64), cmap=CMAP_SEQUENTIAL_CONTRAST, s=80, edgecolors="#222222", linewidths=0.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def plot_component_timecourse_selected_conditions(df_timecourse_selected: pd.DataFrame, value_mean_col: str, value_sem_col: str, ylabel: str, title: str, save_path: str) -> None:
    if len(df_timecourse_selected) == 0:
        return
    layer_order = ["layer2", "layer3"]
    roles = df_timecourse_selected["role_name"].dropna().unique().tolist()
    apply_publication_style()
    fig, axes = plt.subplots(len(layer_order), max(1, len(roles)), figsize=(4.8 * max(1, len(roles)), 4.0 * len(layer_order)), squeeze=False)
    for row_idx, layer_key in enumerate(layer_order):
        for col_idx, role_name in enumerate(roles):
            ax = axes[row_idx][col_idx]
            sub = df_timecourse_selected[(df_timecourse_selected["layer"] == layer_key) & (df_timecourse_selected["role_name"] == role_name)].copy()
            if len(sub) == 0:
                ax.axis("off")
                continue
            ax.plot(sub["time_ms"].to_numpy(dtype=np.float64), sub[value_mean_col].to_numpy(dtype=np.float64), color="#1f77b4", linewidth=LINE_WIDTH_SECONDARY)
            if value_sem_col in sub.columns:
                lo = sub[value_mean_col].to_numpy(dtype=np.float64) - sub[value_sem_col].to_numpy(dtype=np.float64)
                hi = sub[value_mean_col].to_numpy(dtype=np.float64) + sub[value_sem_col].to_numpy(dtype=np.float64)
                ax.fill_between(sub["time_ms"].to_numpy(dtype=np.float64), lo, hi, color="#1f77b4", alpha=ALPHA_FILL)
            ax.set_title(f"{role_name} | {layer_key}")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def plot_ping_window_activity_monitoring(df_regime_summary: pd.DataFrame, save_path: str) -> None:
    sub = df_regime_summary[df_regime_summary["control_mode"] == CONTROL_DYNAMIC_PING].copy()
    if len(sub) == 0:
        return
    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_TWO_BY_TWO)
    for ax, (layer, value_col, title) in zip(
        axes.flatten(),
        [
            ("layer2", "active_fraction_mean", "Layer2 active fraction"),
            ("layer2", "spike_count_mean", "Layer2 spike count"),
            ("layer3", "active_fraction_mean", "Layer3 active fraction"),
            ("layer3", "spike_count_mean", "Layer3 spike count"),
        ],
    ):
        layer_sub = sub[sub["layer"] == layer]
        _phase_heatmap(layer_sub, value_col, title, ax)
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def plot_achieved_activation_sanity(df_calibration: pd.DataFrame, df_selected: pd.DataFrame, save_path: str) -> None:
    if len(df_calibration) == 0 or len(df_selected) == 0:
        return
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL_SUMMARY)
    axes[0].scatter(df_selected["ping_target_frac"].to_numpy(dtype=np.float64), df_selected["achieved_activation_frac"].to_numpy(dtype=np.float64), c=df_selected["ping_duration_ms"].to_numpy(dtype=np.float64), cmap=CMAP_SEQUENTIAL, s=50, edgecolors="#222222", linewidths=0.3)
    axes[0].set_xlabel("Target fraction")
    axes[0].set_ylabel("Achieved L1 activation fraction")
    axes[0].set_title("Target vs achieved activation")
    axes[1].scatter(df_selected["ping_target_frac"].to_numpy(dtype=np.float64), df_selected["D_L1"].to_numpy(dtype=np.float64), c=df_selected["ping_duration_ms"].to_numpy(dtype=np.float64), cmap=CMAP_SEQUENTIAL_ALT, s=50, edgecolors="#222222", linewidths=0.3)
    axes[1].set_xlabel("Target fraction")
    axes[1].set_ylabel("D_L1")
    axes[1].set_title("Target vs D_L1")
    mean_by_duration = df_selected.groupby("ping_duration_ms", as_index=False)["D_L1"].mean()
    axes[2].plot(mean_by_duration["ping_duration_ms"].to_numpy(dtype=np.float64), mean_by_duration["D_L1"].to_numpy(dtype=np.float64), marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_SECONDARY, color="#d62728")
    axes[2].set_xlabel("Duration (ms)")
    axes[2].set_ylabel("Mean D_L1")
    axes[2].set_title("Duration vs D_L1")
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def plot_latent_state_snapshots(df_phase_summary: pd.DataFrame, save_path: str) -> None:
    if len(df_phase_summary) == 0:
        return
    dynamic = df_phase_summary[df_phase_summary["control_mode"] == CONTROL_DYNAMIC_PING].copy()
    if len(dynamic) == 0:
        return
    apply_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_TWO_BY_THREE)
    panels = [
        ("layer2", "ux_decode_acc_mean", "Layer2 u*x decode"),
        ("layer3", "ux_decode_acc_mean", "Layer3 u*x decode"),
        ("layer2", "fisher_ratio_mean", "Layer2 Fisher"),
        ("layer3", "fisher_ratio_mean", "Layer3 Fisher"),
        ("layer2", "delta_fisher_ratio_mean", "Layer2 Fisher delta"),
        ("layer3", "delta_fisher_ratio_mean", "Layer3 Fisher delta"),
    ]
    for ax, (layer, value_col, title) in zip(axes.flatten(), panels):
        sub = dynamic[(dynamic["layer"] == layer) & (dynamic["phase"] == "pre_probe")].copy()
        _phase_heatmap(sub, value_col, title, ax)
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def plot_phase_component_snapshots(df_phase_component_summary: pd.DataFrame, save_path: str) -> None:
    if len(df_phase_component_summary) == 0:
        return
    dynamic = df_phase_component_summary[df_phase_component_summary["control_mode"] == CONTROL_DYNAMIC_PING].copy()
    if len(dynamic) == 0:
        return
    apply_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_TWO_BY_THREE)
    panels = [
        ("layer2", "u_mean", "Layer2 mean u"),
        ("layer2", "x_mean", "Layer2 mean x"),
        ("layer2", "ux_mean", "Layer2 mean u*x"),
        ("layer3", "u_mean", "Layer3 mean u"),
        ("layer3", "x_mean", "Layer3 mean x"),
        ("layer3", "ux_mean", "Layer3 mean u*x"),
    ]
    for ax, (layer, value_col, title) in zip(axes.flatten(), panels):
        sub = dynamic[(dynamic["layer"] == layer) & (dynamic["phase_snapshot"] == "pre_probe")].copy()
        _phase_heatmap(sub, value_col, title, ax)
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def plot_sample_relevance_sanity(df_relevance_sanity: pd.DataFrame, save_path: str) -> None:
    if len(df_relevance_sanity) == 0:
        return
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL_SUMMARY)
    centers = 0.5 * (df_relevance_sanity["bin_left"].to_numpy(dtype=np.float64) + df_relevance_sanity["bin_right"].to_numpy(dtype=np.float64))
    width = np.diff(df_relevance_sanity["bin_right"].to_numpy(dtype=np.float64), prepend=df_relevance_sanity["bin_left"].iloc[0])
    axes[0].bar(centers, df_relevance_sanity["topk_fraction"].to_numpy(dtype=np.float64), width=width, alpha=0.7, label="Top-k")
    axes[0].bar(centers, df_relevance_sanity["non_topk_fraction"].to_numpy(dtype=np.float64), width=width, alpha=0.5, label="Non top-k")
    axes[0].set_title("Relevance ranking histogram")
    axes[0].set_xlabel("Sample relevance score")
    axes[0].set_ylabel("Fraction")
    apply_standard_legend(axes[0])
    topk_size = int(df_relevance_sanity["topk_size"].iloc[0])
    n_neurons = int(df_relevance_sanity["n_neurons"].iloc[0])
    axes[1].bar(["Top-k", "Non top-k"], [topk_size, max(0, n_neurons - topk_size)], color=["#1f77b4", "#bbbbbb"])
    axes[1].set_title("Top-k neuron count")
    axes[1].set_ylabel("Neuron count")
    axes[2].bar(["Top-k", "Non top-k"], [float(df_relevance_sanity["ping_spike_share_topk"].iloc[0]), float(df_relevance_sanity["ping_spike_share_non_topk"].iloc[0])], color=["#d62728", "#999999"])
    axes[2].set_title("Ping spike share")
    axes[2].set_ylabel("Fraction of ping spikes")
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def plot_recruitment_heatmaps(df_regime_summary: pd.DataFrame, save_path: str) -> None:
    plot_ping_window_activity_monitoring(df_regime_summary, save_path)


def plot_fisher_heatmaps(df_phase_summary: pd.DataFrame, save_path: str) -> None:
    if len(df_phase_summary) == 0:
        return
    dynamic = df_phase_summary[df_phase_summary["control_mode"] == CONTROL_DYNAMIC_PING].copy()
    if len(dynamic) == 0:
        return
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL)
    for ax, layer in zip(axes, ["layer2", "layer3"]):
        sub = dynamic[(dynamic["layer"] == layer) & (dynamic["phase"] == "pre_probe")].copy()
        _phase_heatmap(sub, "fisher_ratio_mean", f"{layer} pre-probe Fisher", ax)
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def plot_pair_comparison(df_pairs: pd.DataFrame, save_path: str, title: str) -> None:
    if len(df_pairs) == 0:
        return
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL_SUMMARY)
    x = np.arange(len(df_pairs))
    axes[0].bar(x - 0.18, df_pairs["delta_sample_bias_short"].to_numpy(dtype=np.float64), width=0.36, label="Short")
    axes[0].bar(x + 0.18, df_pairs["delta_sample_bias_long"].to_numpy(dtype=np.float64), width=0.36, label="Long")
    axes[0].set_title("Delta sample bias - noise")
    apply_standard_legend(axes[0])
    axes[1].bar(x - 0.18, df_pairs["decode_short"].to_numpy(dtype=np.float64), width=0.36, label="Short")
    axes[1].bar(x + 0.18, df_pairs["decode_long"].to_numpy(dtype=np.float64), width=0.36, label="Long")
    axes[1].set_title("Ping-window decode")
    burden_key = "l2_spike_count_short" if "l2_spike_count_short" in df_pairs.columns else "D_L1_short"
    burden_long_key = "l2_spike_count_long" if "l2_spike_count_long" in df_pairs.columns else "D_L1_long"
    axes[2].bar(x - 0.18, df_pairs[burden_key].to_numpy(dtype=np.float64), width=0.36, label="Short")
    axes[2].bar(x + 0.18, df_pairs[burden_long_key].to_numpy(dtype=np.float64), width=0.36, label="Long")
    axes[2].set_title("Matched control variable")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df_pairs["pair_id"].tolist(), rotation=45, ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    _save_runtime_figure(fig, save_path)


def build_textual_summary(df_grid_summary: pd.DataFrame, df_same_dose: pd.DataFrame, df_same_burden: pd.DataFrame, num_classes: int) -> str:
    if len(df_grid_summary) == 0:
        return "No grid summary rows were generated."
    chance = 1.0 / float(num_classes)
    lines: List[str] = []
    short = df_grid_summary[(df_grid_summary["regime"] == REGIME_SHORT) & (df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING)].copy()
    long = df_grid_summary[(df_grid_summary["regime"] == REGIME_LONG) & (df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING)].copy()
    if len(short) > 0:
        beneficial = short[(short["delta_sample_bias_minus_noise_mean"] > 0.0) & (short.get("preprobe_latent_gain_mean", 0.0) > 0.0)].copy()
        if len(beneficial) > 0:
            best = beneficial.sort_values(["delta_sample_bias_minus_noise_mean", "preprobe_latent_gain_mean", "l2_ping_spike_count_mean"], ascending=[False, False, True]).iloc[0]
            lines.append(
                "Short ping: beneficial latent window detected at "
                f"{best['ping_duration_ms']:.0f} ms / {best['ping_target_frac']:.3f}; "
                f"D_L1={best['D_L1_mean']:.4f}, S_L1={best['S_L1_mean']:.4f}, "
                f"L2 burden={best['l2_ping_spike_count_mean']:.2f}, "
                f"delta_bias={best['delta_sample_bias_minus_noise_mean']:.4f}. "
                f"Overt decode needed: {'no' if best['decode_ping_acc_mean'] <= chance + 0.05 else 'unclear'}."
            )
        else:
            lines.append("Short ping: no condition satisfied the current beneficial latent-window rule.")
    else:
        lines.append("Short ping: no short dynamic conditions were run.")
    if len(long) > 0:
        overt = long[long["decode_ping_acc_mean"] > chance + 0.10].sort_values(["ping_duration_ms", "D_L1_mean"])
        if len(overt) > 0:
            thresh = overt.iloc[0]
            lines.append(
                "Long ping: overt reactivation threshold appears at "
                f"{thresh['ping_duration_ms']:.0f} ms / {thresh['ping_target_frac']:.3f}; "
                f"D_L1={thresh['D_L1_mean']:.4f}, decode={thresh['decode_ping_acc_mean']:.4f}."
            )
        else:
            lines.append("Long ping: no overt reactivation threshold crossed the current decode criterion.")
        reorg_flag_col = "reorganization_flag" if "reorganization_flag" in long.columns else "reorganization_flag_mean"
        reorg = long[long[reorg_flag_col] > 0.5].copy() if reorg_flag_col in long.columns else pd.DataFrame()
        if len(reorg) > 0:
            worst = reorg.sort_values(["delta_sample_bias_minus_noise_mean", "delta_probe_accuracy_mean", "l2_ping_spike_count_mean"], ascending=[True, True, False]).iloc[0]
            lines.append(
                "Long ping: overt reactivation co-occurs with latent-state reorganization at "
                f"{worst['ping_duration_ms']:.0f} ms / {worst['ping_target_frac']:.3f}; "
                f"immediate_latent={worst['immediate_latent_gain_mean']:.4f}, "
                f"preprobe_latent={worst['preprobe_latent_gain_mean']:.4f}, "
                f"delta_probe={worst['delta_probe_accuracy_mean']:.4f}."
            )
        else:
            lines.append("Long ping: reorganization evidence was not strong under the current automatic rule.")
    else:
        lines.append("Long ping: no long dynamic conditions were run.")
    lines.append(
        "Same-dose control: " + (
            f"{df_same_dose.iloc[0]['duration_short_ms']:.0f} vs {df_same_dose.iloc[0]['duration_long_ms']:.0f} ms at matched D_L1."
            if len(df_same_dose) > 0
            else "no short/long pair met the current D_L1 matching threshold."
        )
    )
    lines.append(
        "Same-burden control: " + (
            f"{df_same_burden.iloc[0]['duration_short_ms']:.0f} vs {df_same_burden.iloc[0]['duration_long_ms']:.0f} ms at matched L2 burden."
            if len(df_same_burden) > 0
            else "no short/long pair met the current burden matching threshold."
        )
    )
    if len(short) > 0 and len(long) > 0:
        lines.append("Overall: the sweep is more consistent with two regimes, selective beneficial latent modulation under short ping and broader overt-but-destructive reactivation under long ping.")
    return "\n".join(lines)


def validate_dual_regime_outputs(df_metrics_seed: pd.DataFrame, df_same_dose: pd.DataFrame, df_same_burden: pd.DataFrame) -> None:
    base_counts = df_metrics_seed[df_metrics_seed["control_mode"] == CONTROL_DYNAMIC_NO_PING].groupby("seed").size()
    if len(base_counts) > 0 and not (base_counts == 1).all():
        raise RuntimeError(f"Expected exactly one shared baseline row per seed, got {base_counts.to_dict()}")
    for col in ["D_L1", "S_L1", "E_sel_dose"]:
        if col in df_metrics_seed.columns:
            vals = df_metrics_seed[col].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(vals)):
                raise RuntimeError(f"Non-finite values detected in {col}")
    if "S_L1" in df_metrics_seed.columns:
        vals = df_metrics_seed["S_L1"].to_numpy(dtype=np.float64)
        if ((vals < -1e-9) | (vals > 1.0 + 1e-9)).any():
            raise RuntimeError("S_L1 must lie within [0, 1]")
    if len(df_same_dose) == 0:
        print("[Warn] same-dose matching produced no pairs.")
    if len(df_same_burden) == 0:
        print("[Warn] same-burden matching produced no pairs.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-regime ping boundary experiment with mechanistic short/long analyses.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--save-dir", type=str, default="results/ping_ux_gated_boundary")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed-list", type=str, default="42")
    parser.add_argument("--num-trials", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay1-ms", type=float, default=400.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--tau-u-ms", type=float, default=DEFAULT_TAU_U_MS)
    parser.add_argument("--fixed-sample-to-probe-gap-ms", type=float, default=DEFAULT_FIXED_SAMPLE_TO_PROBE_GAP_MS)
    parser.add_argument("--ping-drive-candidates", type=str, default=DEFAULT_PING_DRIVE_CANDIDATES)
    parser.add_argument("--decode-splits", type=int, default=5)
    parser.add_argument("--experiment-regime", type=str, default="all", choices=["short", "long", "all"])
    parser.add_argument("--short-ping-duration-ms-list", type=str, default=DEFAULT_SHORT_PING_DURATION_MS_LIST)
    parser.add_argument("--long-ping-duration-ms-list", type=str, default=DEFAULT_LONG_PING_DURATION_MS_LIST)
    parser.add_argument("--short-ping-target-fracs", type=str, default=DEFAULT_SHORT_PING_TARGETS)
    parser.add_argument("--long-ping-target-fracs", type=str, default=DEFAULT_LONG_PING_TARGETS)
    parser.add_argument("--s-sel-topk-frac", type=float, default=DEFAULT_S_SEL_TOPK_FRAC)
    parser.add_argument("--control-coverage", type=str, default="focused", choices=["focused", "full", "none"])
    parser.add_argument("--ping-target-fracs", type=str, default=None)
    parser.add_argument("--ping-duration-ms-list", type=str, default=None)
    parser.add_argument("--ping-ms", type=float, default=None)
    parser.add_argument("--delay2-ms", type=float, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    layout = prepare_result_layout(args.save_dir)
    result_root = layout.root
    data_dir = str(layout.data_dir)
    figure_dir = str(layout.figure_dir)
    log_dir = layout.log_dir

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    seeds = parse_seed_list(args.seed_list)
    ping_drive_candidates = parse_float_list(args.ping_drive_candidates)
    if args.ping_duration_ms_list is not None:
        shared_durations = sorted(set(parse_float_list(args.ping_duration_ms_list)))
        short_duration_ms_list = shared_durations
        long_duration_ms_list = shared_durations
    else:
        short_duration_ms_list = sorted(set(parse_float_list(args.short_ping_duration_ms_list)))
        long_duration_ms_list = sorted(set(parse_float_list(args.long_ping_duration_ms_list)))
    if args.ping_ms is not None:
        short_duration_ms_list = [float(args.ping_ms)]
        long_duration_ms_list = [float(args.ping_ms)]
    if args.ping_target_fracs is not None:
        shared_targets = sorted(set(parse_float_list(args.ping_target_fracs)))
        short_ping_target_fracs = shared_targets
        long_ping_target_fracs = shared_targets
    else:
        short_ping_target_fracs = sorted(set(parse_float_list(args.short_ping_target_fracs)))
        long_ping_target_fracs = sorted(set(parse_float_list(args.long_ping_target_fracs)))

    run_short = args.experiment_regime in {"short", "all"}
    run_long = args.experiment_regime in {"long", "all"}
    longest_ping = max(short_duration_ms_list + long_duration_ms_list + [0.0])
    if args.delay2_ms is not None:
        print("[Warn] --delay2-ms is ignored; delay2 is derived from --fixed-sample-to-probe-gap-ms.")
    if float(args.tau_u_ms) != DEFAULT_TAU_U_MS:
        print(f"[Warn] Requested tau_u_ms={args.tau_u_ms}; using fixed default {DEFAULT_TAU_U_MS} ms for this experiment.")

    load_spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=float(args.sample_ms),
        delay_ms=float(args.delay1_ms),
        ping_ms=longest_ping,
        post_ping_ms=derive_delay2_ms(delay1_ms=float(args.delay1_ms), ping_duration_ms=longest_ping, fixed_sample_to_probe_gap_ms=float(args.fixed_sample_to_probe_gap_ms)),
        probe_ms=float(args.probe_ms),
    )

    print(
        f"[Init] device={device} | seeds={seeds} | trials/seed={args.num_trials} | batch_size={args.batch_size}\n"
        f"[Init] experiment_regime={args.experiment_regime} | short_durations={short_duration_ms_list} | long_durations={long_duration_ms_list}\n"
        f"[Init] short_targets={short_ping_target_fracs}\n"
        f"[Init] long_targets={long_ping_target_fracs}\n"
        f"[Init] fixed_gap={args.fixed_sample_to_probe_gap_ms} | s_sel_topk_frac={args.s_sel_topk_frac} | control_coverage={args.control_coverage}"
    )

    net, encoder = load_model_and_encoder(args.model_path, device, load_spec)
    override_tau_u_ms(net, tau_u_ms=DEFAULT_TAU_U_MS, dt=load_spec.dt)

    _, _, test_loader = build_mnist_skeleton_loader(root=args.dataset_root, batch_size=64, input_size=28, num_workers=0)
    dataset = test_loader.dataset
    class_index = build_class_index(dataset, num_classes=args.num_classes)

    calibration_frames: List[pd.DataFrame] = []
    selected_frames: List[pd.DataFrame] = []
    trial_frames: List[pd.DataFrame] = []
    metrics_seed_frames: List[pd.DataFrame] = []
    regime_trial_frames: List[pd.DataFrame] = []
    ux_phase_seed_frames: List[pd.DataFrame] = []
    ux_timecourse_seed_frames: List[pd.DataFrame] = []
    seed_trial_specs: Dict[int, pd.DataFrame] = {}
    ping_lookup_by_seed: Dict[int, Dict[float, Dict[int, Dict[str, Dict[str, float]]]]] = {}

    baseline_spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=float(args.sample_ms),
        delay_ms=float(args.delay1_ms),
        ping_ms=0.0,
        post_ping_ms=derive_delay2_ms(delay1_ms=float(args.delay1_ms), ping_duration_ms=0.0, fixed_sample_to_probe_gap_ms=float(args.fixed_sample_to_probe_gap_ms)),
        probe_ms=float(args.probe_ms),
    )
    baseline_cond = build_baseline_condition(delay1_ms=float(args.delay1_ms), fixed_sample_to_probe_gap_ms=float(args.fixed_sample_to_probe_gap_ms))

    for seed in seeds:
        seed_everything(seed)
        rng = random.Random(seed)
        df_specs = generate_balanced_trial_specs(class_index=class_index, num_trials=args.num_trials, num_classes=args.num_classes, rng=rng)
        validate_trial_specs(df_specs, num_classes=args.num_classes)
        seed_trial_specs[int(seed)] = df_specs.copy()
        ping_lookup_by_seed[int(seed)] = {}

        print(f"[Run] seed={seed} | shared dynamic no-ping baseline")
        df_trials_base, df_metrics_base, df_regime_base, df_phase_base, df_time_base = shared_run_seed_experiment(
            net=net,
            encoder=encoder,
            dataset=dataset,
            df_specs=df_specs,
            spec=baseline_spec,
            ping_lookup={},
            ping_target_fracs=[],
            batch_size=args.batch_size,
            decode_splits=args.decode_splits,
            seed=seed,
            device=device,
            include_no_ping_baseline=False,
            condition_defs=[baseline_cond],
            s_sel_topk_frac=float(args.s_sel_topk_frac),
        )
        trial_frames.append(df_trials_base)
        metrics_seed_frames.append(df_metrics_base)
        regime_trial_frames.append(df_regime_base)
        ux_phase_seed_frames.append(df_phase_base)
        ux_timecourse_seed_frames.append(df_time_base)

        duration_target_plan: List[Tuple[str, float, List[float]]] = []
        if run_short:
            for duration_ms in short_duration_ms_list:
                duration_target_plan.append((REGIME_SHORT, float(duration_ms), short_ping_target_fracs))
        if run_long:
            for duration_ms in long_duration_ms_list:
                duration_target_plan.append((REGIME_LONG, float(duration_ms), long_ping_target_fracs))

        for regime_name, duration_ms, ping_target_fracs in duration_target_plan:
            delay2_ms = derive_delay2_ms(delay1_ms=float(args.delay1_ms), ping_duration_ms=float(duration_ms), fixed_sample_to_probe_gap_ms=float(args.fixed_sample_to_probe_gap_ms))
            spec = ExperimentSpec(dt=1.0 * ms, sample_ms=float(args.sample_ms), delay_ms=float(args.delay1_ms), ping_ms=float(duration_ms), post_ping_ms=float(delay2_ms), probe_ms=float(args.probe_ms))
            print(f"[Run] seed={seed} | calibration regime={regime_name} duration={duration_ms:.0f} ms | delay2={delay2_ms:.0f} ms")
            df_calibration_seed, df_selected_seed, ping_lookup = shared_calibrate_ping_per_example(net=net, encoder=encoder, dataset=dataset, df_specs=df_specs, spec=spec, ping_amp_candidates=ping_drive_candidates, ping_target_fracs=ping_target_fracs, batch_size=args.batch_size, device=device)
            df_calibration_seed["seed"] = int(seed)
            df_calibration_seed["regime"] = regime_name
            df_selected_seed["seed"] = int(seed)
            df_selected_seed["regime"] = regime_name
            calibration_frames.append(df_calibration_seed)
            selected_frames.append(df_selected_seed)
            ping_lookup_by_seed[int(seed)][float(duration_ms)] = ping_lookup

            condition_defs = build_dynamic_condition_defs(ping_target_fracs=ping_target_fracs, ping_duration_ms=float(duration_ms), delay1_ms=float(args.delay1_ms), fixed_sample_to_probe_gap_ms=float(args.fixed_sample_to_probe_gap_ms), regime=regime_name, analysis_role=_analysis_role_for_duration(float(duration_ms), regime_name), include_no_ping=False)
            print(f"[Run] seed={seed} | dynamic {regime_name} grid duration={duration_ms:.0f} ms")
            df_trials_seed, df_metrics_seed, df_regime_seed, df_phase_seed, df_time_seed = shared_run_seed_experiment(
                net=net,
                encoder=encoder,
                dataset=dataset,
                df_specs=df_specs,
                spec=spec,
                ping_lookup=ping_lookup,
                ping_target_fracs=ping_target_fracs,
                batch_size=args.batch_size,
                decode_splits=args.decode_splits,
                seed=seed,
                device=device,
                include_no_ping_baseline=False,
                condition_defs=condition_defs,
                s_sel_topk_frac=float(args.s_sel_topk_frac),
            )
            trial_frames.append(df_trials_seed)
            metrics_seed_frames.append(df_metrics_seed)
            regime_trial_frames.append(df_regime_seed)
            ux_phase_seed_frames.append(df_phase_seed)
            ux_timecourse_seed_frames.append(df_time_seed)

    df_calibration = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    df_selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    df_trials = pd.concat(trial_frames, ignore_index=True)
    df_metrics_seed = pd.concat(metrics_seed_frames, ignore_index=True)
    df_regime_trials = pd.concat(regime_trial_frames, ignore_index=True)
    df_ux_phase_seed = pd.concat(ux_phase_seed_frames, ignore_index=True)
    df_ux_timecourse_seed = pd.concat(ux_timecourse_seed_frames, ignore_index=True)

    metrics_for_delta = [
        "decode_ping_acc", "sample_bias_minus_noise", "probe_accuracy", "error_to_sample_fraction",
        "sample_aligned_selectivity", "D_L1", "S_L1", "E_sel_dose", "d_int", "s_sel",
        "achieved_l1_activation_fraction", "dose_l1_activation_ms", "integrated_perturbation_current",
        "l2_ping_spike_count", "l3_ping_spike_count", "l2_ping_active_fraction", "l3_ping_active_fraction",
        "l2_integrated_active_fraction_ms", "l3_integrated_active_fraction_ms",
    ]
    df_metrics_seed = add_baseline_deltas(df_metrics_seed, metric_cols=metrics_for_delta)
    df_ux_phase_seed = add_phase_baseline_deltas(df_ux_phase_seed, metric_cols=["ux_decode_acc", "between_class_var", "within_class_var", "fisher_ratio"])
    df_grid_seed_summary = build_main_grid_seed_summary(df_metrics_seed=df_metrics_seed, df_phase_seed=df_ux_phase_seed)

    grid_group_cols = ["condition", "condition_label", "condition_family", "ping_target_label", "ping_target_frac", "ping_duration_ms", "delay2_ms", "regime", "control_mode", "analysis_role"]
    df_grid_summary = summarize_numeric_frame(df_grid_seed_summary, group_cols=grid_group_cols).sort_values(["control_mode", "regime", "ping_duration_ms", "ping_target_frac"]).reset_index(drop=True)
    df_grid_summary = add_composite_summary_metrics(df_grid_summary, num_classes=int(args.num_classes))

    if run_long and args.control_coverage != "none":
        if args.control_coverage == "full":
            long_dynamic_rows = [ConditionDef(condition=str(row["condition"]), condition_label=str(row["condition_label"]), condition_family=str(row["condition_family"]), ping_target_label=str(row["ping_target_label"]), ping_target_frac=float(row["ping_target_frac"]), ping_duration_ms=float(row["ping_duration_ms"]), delay2_ms=float(row["delay2_ms"]), regime=str(row["regime"]), control_mode=str(row["control_mode"]), analysis_role=str(row["analysis_role"])) for _, row in df_grid_summary[(df_grid_summary["regime"] == REGIME_LONG) & (df_grid_summary["control_mode"] == CONTROL_DYNAMIC_PING)].iterrows()]
        else:
            long_dynamic_rows = select_focused_long_control_targets(df_grid_summary=df_grid_summary, num_classes=int(args.num_classes), delay1_ms=float(args.delay1_ms), fixed_gap_ms=float(args.fixed_sample_to_probe_gap_ms))
        control_condition_defs = build_long_control_condition_defs(long_dynamic_rows)
        controls_by_duration: Dict[float, List[ConditionDef]] = {}
        for cond in control_condition_defs:
            controls_by_duration.setdefault(float(cond.ping_duration_ms), []).append(cond)
        for seed in seeds:
            df_specs = seed_trial_specs[int(seed)]
            for duration_ms, conds in sorted(controls_by_duration.items()):
                spec = ExperimentSpec(dt=1.0 * ms, sample_ms=float(args.sample_ms), delay_ms=float(args.delay1_ms), ping_ms=float(duration_ms), post_ping_ms=derive_delay2_ms(delay1_ms=float(args.delay1_ms), ping_duration_ms=float(duration_ms), fixed_sample_to_probe_gap_ms=float(args.fixed_sample_to_probe_gap_ms)), probe_ms=float(args.probe_ms))
                print(f"[Run] seed={seed} | long control set duration={duration_ms:.0f} ms | n={len(conds)}")
                df_trials_ctrl, df_metrics_ctrl, df_regime_ctrl, df_phase_ctrl, df_time_ctrl = shared_run_seed_experiment(net=net, encoder=encoder, dataset=dataset, df_specs=df_specs, spec=spec, ping_lookup=ping_lookup_by_seed[int(seed)][float(duration_ms)], ping_target_fracs=[], batch_size=args.batch_size, decode_splits=args.decode_splits, seed=seed, device=device, include_no_ping_baseline=False, condition_defs=conds, s_sel_topk_frac=float(args.s_sel_topk_frac))
                trial_frames.append(df_trials_ctrl)
                metrics_seed_frames.append(df_metrics_ctrl)
                regime_trial_frames.append(df_regime_ctrl)
                ux_phase_seed_frames.append(df_phase_ctrl)
                ux_timecourse_seed_frames.append(df_time_ctrl)

        df_trials = pd.concat(trial_frames, ignore_index=True)
        df_metrics_seed = pd.concat(metrics_seed_frames, ignore_index=True)
        df_regime_trials = pd.concat(regime_trial_frames, ignore_index=True)
        df_ux_phase_seed = pd.concat(ux_phase_seed_frames, ignore_index=True)
        df_ux_timecourse_seed = pd.concat(ux_timecourse_seed_frames, ignore_index=True)
        df_metrics_seed = add_baseline_deltas(df_metrics_seed, metric_cols=metrics_for_delta)
        df_ux_phase_seed = add_phase_baseline_deltas(df_ux_phase_seed, metric_cols=["ux_decode_acc", "between_class_var", "within_class_var", "fisher_ratio"])
        df_grid_seed_summary = build_main_grid_seed_summary(df_metrics_seed=df_metrics_seed, df_phase_seed=df_ux_phase_seed)
        df_grid_summary = summarize_numeric_frame(df_grid_seed_summary, group_cols=grid_group_cols).sort_values(["control_mode", "regime", "ping_duration_ms", "ping_target_frac"]).reset_index(drop=True)
        df_grid_summary = add_composite_summary_metrics(df_grid_summary, num_classes=int(args.num_classes))

    df_phase_summary = summarize_numeric_frame(df_ux_phase_seed, group_cols=grid_group_cols + ["layer", "phase"]).sort_values(["control_mode", "regime", "ping_duration_ms", "ping_target_frac", "layer", "phase"]).reset_index(drop=True)
    df_regime_summary = summarize_numeric_frame(df_regime_trials, group_cols=grid_group_cols + ["layer"]).sort_values(["control_mode", "regime", "ping_duration_ms", "ping_target_frac", "layer"]).reset_index(drop=True)
    df_ux_timecourse_summary = summarize_ux_timecourse(df_ux_timecourse_seed)
    df_phase_component_summary = build_phase_component_summary(df_ux_timecourse_summary)
    df_same_dose = build_same_dose_pairs(df_grid_summary)
    df_same_burden = build_same_burden_pairs(df_grid_summary)
    df_representatives = select_representative_conditions(df_grid_summary)
    df_timecourse_selected = build_selected_timecourse_summary(df_ux_timecourse_summary, df_representatives)

    relevance_sanity_frames: List[pd.DataFrame] = []
    if len(df_representatives) > 0 and run_short:
        short_rep = df_representatives[df_representatives["role_name"] == "short_beneficial_candidate"]
        if len(short_rep) > 0:
            rep_row = short_rep.iloc[0]
            seed0 = int(seeds[0])
            duration_ms = float(rep_row["ping_duration_ms"])
            rep_cond = ConditionDef(condition=str(rep_row["condition"]), condition_label=str(rep_row["condition_label"]), condition_family=str(rep_row["condition_family"]), ping_target_label=str(rep_row["ping_target_label"]), ping_target_frac=float(rep_row["ping_target_frac"]), ping_duration_ms=duration_ms, delay2_ms=float(rep_row["delay2_ms"]), regime=str(rep_row["regime"]), control_mode=str(rep_row["control_mode"]), analysis_role=str(rep_row["analysis_role"]))
            sanity_spec = ExperimentSpec(dt=1.0 * ms, sample_ms=float(args.sample_ms), delay_ms=float(args.delay1_ms), ping_ms=duration_ms, post_ping_ms=float(rep_row["delay2_ms"]), probe_ms=float(args.probe_ms))
            relevance_sanity_frames.append(collect_sample_relevance_sanity(net=net, encoder=encoder, dataset=dataset, df_specs=seed_trial_specs[seed0], spec=sanity_spec, cond=rep_cond, ping_lookup=ping_lookup_by_seed[seed0][duration_ms], batch_size=args.batch_size, device=device, topk_frac=float(args.s_sel_topk_frac)))
    df_relevance_sanity = pd.concat(relevance_sanity_frames, ignore_index=True) if relevance_sanity_frames else pd.DataFrame()

    validate_dual_regime_outputs(df_metrics_seed=df_metrics_seed, df_same_dose=df_same_dose, df_same_burden=df_same_burden)

    calibration_csv = os.path.join(data_dir, "ping_calibration_per_example.csv")
    selected_csv = os.path.join(data_dir, "ping_calibration_selected_table.csv")
    trials_csv = os.path.join(data_dir, "trial_predictions.csv")
    summary_csv = os.path.join(data_dir, "metrics_condition_summary.csv")
    delta_csv = os.path.join(data_dir, "metrics_delta_summary.csv")
    regime_csv = os.path.join(data_dir, "metrics_ping_regime_by_layer.csv")
    ux_timecourse_csv = os.path.join(data_dir, "metrics_ux_timecourse_ping_by_condition.csv")
    grid_seed_csv = os.path.join(data_dir, "metrics_grid_seed_summary.csv")
    phase_summary_csv = os.path.join(data_dir, "metrics_phase_level_summary.csv")
    phase_component_csv = os.path.join(data_dir, "metrics_phase_component_summary.csv")
    grid_summary_csv = os.path.join(data_dir, "metrics_grid_summary.csv")
    ux_phase_csv = os.path.join(data_dir, "metrics_ux_phase_decode.csv")
    ux_sep_csv = os.path.join(data_dir, "metrics_ux_phase_separability.csv")
    regime_trials_csv = os.path.join(data_dir, "metrics_recruitment_trial_level.csv")
    selected_timecourse_csv = os.path.join(data_dir, "metrics_ux_timecourse_selected_conditions.csv")
    same_dose_csv = os.path.join(data_dir, "metrics_same_dose_pairs.csv")
    same_burden_csv = os.path.join(data_dir, "metrics_same_burden_pairs.csv")
    representative_csv = os.path.join(data_dir, "metrics_representative_conditions.csv")
    relevance_csv = os.path.join(data_dir, "metrics_sample_relevance_sanity.csv")
    summary_text_path = os.path.join(data_dir, "summary_text.txt")

    if len(df_calibration) > 0:
        df_calibration.to_csv(calibration_csv, index=False)
    if len(df_selected) > 0:
        df_selected.to_csv(selected_csv, index=False)
    df_trials.to_csv(trials_csv, index=False)
    df_grid_summary.to_csv(summary_csv, index=False)
    shared_compute_delta_summary(df_metrics_seed).to_csv(delta_csv, index=False)
    df_regime_summary.to_csv(regime_csv, index=False)
    df_ux_timecourse_summary.to_csv(ux_timecourse_csv, index=False)
    df_grid_seed_summary.to_csv(grid_seed_csv, index=False)
    df_phase_summary.to_csv(phase_summary_csv, index=False)
    df_phase_component_summary.to_csv(phase_component_csv, index=False)
    df_grid_summary.to_csv(grid_summary_csv, index=False)
    phase_decode_cols = [
        "condition", "condition_label", "condition_family", "ping_target_label", "ping_target_frac",
        "ping_duration_ms", "delay2_ms", "regime", "control_mode", "analysis_role", "layer", "phase",
        "ux_decode_acc_mean", "ux_decode_acc_sem", "delta_ux_decode_acc_mean", "delta_ux_decode_acc_sem",
    ]
    phase_sep_cols = [
        "condition", "condition_label", "condition_family", "ping_target_label", "ping_target_frac",
        "ping_duration_ms", "delay2_ms", "regime", "control_mode", "analysis_role", "layer", "phase",
        "between_class_var_mean", "between_class_var_sem", "delta_between_class_var_mean", "delta_between_class_var_sem",
        "within_class_var_mean", "within_class_var_sem", "delta_within_class_var_mean", "delta_within_class_var_sem",
        "fisher_ratio_mean", "fisher_ratio_sem", "delta_fisher_ratio_mean", "delta_fisher_ratio_sem",
    ]
    df_phase_summary[[col for col in phase_decode_cols if col in df_phase_summary.columns]].to_csv(ux_phase_csv, index=False)
    df_phase_summary[[col for col in phase_sep_cols if col in df_phase_summary.columns]].to_csv(ux_sep_csv, index=False)
    df_regime_trials.to_csv(regime_trials_csv, index=False)
    df_timecourse_selected.to_csv(selected_timecourse_csv, index=False)
    df_same_dose.to_csv(same_dose_csv, index=False)
    df_same_burden.to_csv(same_burden_csv, index=False)
    df_representatives.to_csv(representative_csv, index=False)
    df_relevance_sanity.to_csv(relevance_csv, index=False)

    plot_short_beneficial_window(df_grid_summary, os.path.join(figure_dir, "plot_short_beneficial_window.png"))
    plot_short_latent_improvement(df_grid_summary, os.path.join(figure_dir, "plot_short_latent_improvement.png"))
    plot_long_overt_reactivation(df_grid_summary, os.path.join(figure_dir, "plot_long_overt_reactivation.png"))
    plot_behavior_latent_overt_dissociation(df_grid_summary, os.path.join(figure_dir, "plot_behavior_latent_overt_dissociation.png"))
    plot_dose_burden_comparison(df_grid_summary, os.path.join(figure_dir, "plot_dose_burden_comparison.png"))
    plot_component_timecourse_selected_conditions(df_timecourse_selected, value_mean_col="u_mean", value_sem_col="u_sem", ylabel="Mean u", title="Selected-condition u timecourse", save_path=os.path.join(figure_dir, "plot_selected_u_timecourse.png"))
    plot_component_timecourse_selected_conditions(df_timecourse_selected, value_mean_col="x_mean", value_sem_col="x_sem", ylabel="Mean x", title="Selected-condition x timecourse", save_path=os.path.join(figure_dir, "plot_selected_x_timecourse.png"))
    plot_component_timecourse_selected_conditions(df_timecourse_selected, value_mean_col="ux_mean", value_sem_col="ux_sem", ylabel="Mean u*x", title="Selected-condition u*x timecourse", save_path=os.path.join(figure_dir, "plot_selected_ux_timecourse.png"))
    plot_ping_window_activity_monitoring(df_regime_summary, os.path.join(figure_dir, "plot_ping_window_activity_monitoring.png"))
    plot_achieved_activation_sanity(df_calibration, df_selected, os.path.join(figure_dir, "plot_achieved_activation_sanity.png"))
    plot_latent_state_snapshots(df_phase_summary, os.path.join(figure_dir, "plot_latent_state_snapshots.png"))
    plot_phase_component_snapshots(df_phase_component_summary, os.path.join(figure_dir, "plot_phase_component_snapshots.png"))
    plot_sample_relevance_sanity(df_relevance_sanity, os.path.join(figure_dir, "plot_sample_relevance_sanity.png"))
    plot_recruitment_heatmaps(df_regime_summary, os.path.join(figure_dir, "plot_recruitment_heatmaps.png"))
    plot_fisher_heatmaps(df_phase_summary, os.path.join(figure_dir, "plot_fisher_heatmaps.png"))
    plot_pair_comparison(df_same_dose, os.path.join(figure_dir, "plot_same_dose_comparison.png"), "Same-dose short vs long")
    plot_pair_comparison(df_same_burden, os.path.join(figure_dir, "plot_same_burden_comparison.png"), "Same-burden short vs long")

    summary_text = build_textual_summary(df_grid_summary=df_grid_summary, df_same_dose=df_same_dose, df_same_burden=df_same_burden, num_classes=int(args.num_classes))
    with open(summary_text_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    run_config_path = save_run_config(
        {
            "model_path": args.model_path,
            "dataset_root": args.dataset_root,
            "device": str(device),
            "seeds": seeds,
            "num_trials_per_seed": int(args.num_trials),
            "batch_size": int(args.batch_size),
            "num_classes": int(args.num_classes),
            "tau_u_ms": DEFAULT_TAU_U_MS,
            "experiment_regime": args.experiment_regime,
            "control_coverage": args.control_coverage,
            "fixed_sample_to_probe_gap_ms": float(args.fixed_sample_to_probe_gap_ms),
            "s_sel_topk_frac": float(args.s_sel_topk_frac),
            "outputs": {
                "condition_summary": summary_csv,
                "delta_summary": delta_csv,
                "phase_summary": phase_summary_csv,
                "phase_component_summary": phase_component_csv,
                "same_dose_pairs": same_dose_csv,
                "same_burden_pairs": same_burden_csv,
                "representatives": representative_csv,
                "summary_text": summary_text_path,
            },
        },
        result_root,
    )
    summary_path = save_summary_json(
        {
            "experiment": "ping_ux_gated_boundary_refactor",
            "seed_count": int(len(seeds)),
            "condition_rows": int(len(df_grid_summary)),
            "same_dose_pair_count": int(len(df_same_dose)),
            "same_burden_pair_count": int(len(df_same_burden)),
            "summary_text_path": summary_text_path,
            "run_config_json": str(run_config_path.resolve()),
        },
        result_root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=ping_ux_gated_boundary_refactor",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"device={device}",
            f"seeds={len(seeds)}",
            f"condition_rows={len(df_grid_summary)}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_path.resolve()}",
        ],
        log_dir,
    )

    print(f"[Done] trials      -> {trials_csv}")
    print(f"[Done] summary     -> {summary_csv}")
    print(f"[Done] delta       -> {delta_csv}")
    print(f"[Done] same dose   -> {same_dose_csv}")
    print(f"[Done] same burden -> {same_burden_csv}")
    print(f"[Done] summary txt -> {summary_text_path}")
    print(f"[Done] run config  -> {run_config_path}")
    print(f"[Done] summary     -> {summary_path}")
    print(f"[Done] run log     -> {run_log_path}")


if __name__ == "__main__":
    main()
