from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.experiments.ping_memory.shared.ping_api import (
    ExperimentSpec,
    LAYER_KEYS,
    build_stratified_splits,
    compute_sample_and_noise_bias,
    decode_accuracy_with_splits,
    format_ping_target_label,
    prepare_network_state,
    reset_l3_decision_window,
    snapshot_ux_state,
)
from src.experiments.ping_memory.shared.shuffle_ops import apply_trial_shuffle_ux_in_place, build_trial_shuffle_plan
from src.platform.legacy_adapters.units import ms

DEFAULT_TAU_U_MS = 2000.0
DEFAULT_S_SEL_TOPK_FRAC = 0.20
PHASE_ORDER = ["pre_ping", "post_ping_immediate", "pre_probe"]
PHASE_LATENT_LAYERS = ["layer2", "layer3"]
CONTROL_DYNAMIC_NO_PING = "dynamic_no_ping"
CONTROL_DYNAMIC_PING = "dynamic_ping"


@dataclass(frozen=True)
class BoundaryConditionDef:
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


def _default_regime_name(duration_ms: float) -> str:
    return "short" if float(duration_ms) <= 7.0 else "long"


def _build_default_condition_defs(
    ping_target_fracs: Sequence[float],
    *,
    ping_duration_ms: float,
    delay2_ms: float,
    include_no_ping: bool,
) -> List[BoundaryConditionDef]:
    regime = _default_regime_name(float(ping_duration_ms))
    duration_tag = f"{int(round(float(ping_duration_ms))):02d}ms"
    out: List[BoundaryConditionDef] = []
    if include_no_ping:
        out.append(
            BoundaryConditionDef(
                condition="sample_ping__no_ping_shared",
                condition_label="Dynamic STSP + no ping",
                condition_family=CONTROL_DYNAMIC_NO_PING,
                ping_target_label="no_ping",
                ping_target_frac=0.0,
                ping_duration_ms=float(ping_duration_ms),
                delay2_ms=float(delay2_ms),
                regime="shared",
                control_mode=CONTROL_DYNAMIC_NO_PING,
                analysis_role="control",
            )
        )
    for target_frac in ping_target_fracs:
        target_label = format_ping_target_label(float(target_frac))
        out.append(
            BoundaryConditionDef(
                condition=f"{regime}__{CONTROL_DYNAMIC_PING}__{duration_tag}__{target_label}",
                condition_label=f"{regime.title()} | Dynamic ping | {target_label} | {float(ping_duration_ms):.0f} ms",
                condition_family=CONTROL_DYNAMIC_PING,
                ping_target_label=target_label,
                ping_target_frac=float(target_frac),
                ping_duration_ms=float(ping_duration_ms),
                delay2_ms=float(delay2_ms),
                regime=regime,
                control_mode=CONTROL_DYNAMIC_PING,
                analysis_role="main",
            )
        )
    return out


def build_ping_drive(zero_input: torch.Tensor, ping_amp: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(zero_input) * ping_amp.view(-1, 1, 1, 1)


def snapshot_ux_component_layer_means(net, batch_size: int) -> Dict[str, Dict[str, np.ndarray]]:
    state = snapshot_ux_state(net, batch_size=batch_size)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for layer_key, layer_state in state.items():
        out[layer_key] = {
            "u": np.asarray(layer_state["u"], dtype=np.float32).mean(axis=1),
            "x": np.asarray(layer_state["x"], dtype=np.float32).mean(axis=1),
            "ux": np.asarray(layer_state["gain"], dtype=np.float32).mean(axis=1),
        }
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
    for batch_idx in range(batch_size):
        if not bool(any_spike_tb[:, batch_idx].any()):
            continue
        first_t = int(np.argmax(any_spike_tb[:, batch_idx]))
        first_neuron = int(np.argmax(layer3_trace_tbn[first_t, batch_idx]))
        pred[batch_idx] = int(first_neuron // neurons_per_class)
        first_fire_t[batch_idx] = first_t
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
    for idx, sample_label in enumerate(sample_labels.tolist()):
        counts = layer3_class_counts[idx]
        sample_count = float(counts[int(sample_label)])
        other_mask = np.ones(len(counts), dtype=bool)
        other_mask[int(sample_label)] = False
        other_mean = float(counts[other_mask].mean()) if other_mask.any() else 0.0
        out[idx] = sample_count - other_mean
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

    topk_count = max(1, min(n_neurons, int(np.ceil(float(topk_frac) * float(n_neurons)))))
    topk_idx = np.argpartition(sample_spike_count_vector, kth=n_neurons - topk_count, axis=1)[:, -topk_count:]
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
        "topk_count": np.full(n_trials, topk_count, dtype=np.int64),
        "ping_topk_spikes": selected_ping_spikes.astype(np.float32, copy=False),
        "ping_non_topk_spikes": (total_ping_spikes - selected_ping_spikes).astype(np.float32, copy=False),
        "topk_relevance_mean": topk_relevance.mean(axis=1).astype(np.float32, copy=False),
        "non_topk_relevance_mean": (
            (sample_spike_count_vector.sum(axis=1) - topk_relevance.sum(axis=1)) / float(max(1, n_neurons - topk_count))
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


def run_ping_calibration_session(
    net,
    sample_spikes: torch.Tensor,
    spec: ExperimentSpec,
    ping_amp: torch.Tensor,
    stsp_mode: str = "dynamic",
) -> Dict[str, object]:
    batch_size, t_sample, channels, height, width = sample_spikes.shape
    if t_sample != spec.sample_steps:
        raise ValueError(f"Sample step mismatch: {t_sample} vs {spec.sample_steps}")

    prepare_network_state(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), device=sample_spikes.device)
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

    for t_step in range(spec.sample_steps):
        step_network(sample_spikes[:, t_step, ...])
    for _ in range(spec.delay_steps):
        step_network(zero_input)
    reset_l3_decision_window(net)
    for ping_step in range(spec.ping_steps):
        step_network(zero_input, ping_drive_t=ping_drive, force_l3_time=ping_step, record_ping=True)

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
    batch_size, t_sample, channels, height, width = sample_spikes.shape
    if t_sample != spec.sample_steps:
        raise ValueError(f"Sample step mismatch: {t_sample} vs {spec.sample_steps}")
    if probe_spikes.shape[1] != spec.probe_steps:
        raise ValueError(f"Probe step mismatch: {probe_spikes.shape[1]} vs {spec.probe_steps}")

    prepare_network_state(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), device=sample_spikes.device)
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
        ux_timecourse.append(
            {
                "phase": phase,
                "phase_step": int(phase_step),
                "global_time_step": int(max(0, current_time - 1)),
                "time_ms": float(current_time * spec.dt / ms),
                "layer_means": snapshot_ux_component_layer_means(net, batch_size=batch_size),
            }
        )

    for sample_step in range(spec.sample_steps):
        sample_input_t = zero_input if zero_sample else sample_spikes[:, sample_step, ...]
        step_network(sample_input_t, record_sample_l1=True)
        record_ux_timepoint("sample", sample_step)
    for delay_step in range(spec.delay_steps):
        step_network(zero_input)
        record_ux_timepoint("delay1", delay_step)

    pre_ping_ux = snapshot_ux_state(net, batch_size)
    if intervention_fn is not None:
        intervention_record = intervention_fn(net, batch_meta)

    reset_l3_decision_window(net)
    for ping_step in range(spec.ping_steps):
        step_network(zero_input, ping_drive_t=ping_drive, force_l3_time=ping_step, record_ping=True)
        record_ux_timepoint("ping", ping_step)

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

    for post_ping_step in range(spec.post_ping_steps):
        step_network(zero_input, force_l3_time=spec.ping_steps + post_ping_step)
        record_ux_timepoint("delay2", post_ping_step)

    pre_probe_ux = snapshot_ux_state(net, batch_size)
    reset_l3_decision_window(net)
    for probe_step in range(spec.probe_steps):
        step_network(probe_spikes[:, probe_step, ...], force_l3_time=probe_step)
        record_ux_timepoint("probe", probe_step)

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
    chosen_idx: Optional[List[int]] = None
    positive_idx = [idx for idx in range(len(amps)) if amps[idx] > 0.0 and activation[idx] > 0.0]
    if len(positive_idx) >= len(targets) and len(targets) > 0:
        dp = np.full((len(targets), len(positive_idx)), np.inf, dtype=np.float64)
        back = np.full((len(targets), len(positive_idx)), -1, dtype=np.int64)
        for pos_idx, source_idx in enumerate(positive_idx):
            dp[0, pos_idx] = abs(float(activation[source_idx]) - float(targets[0]))
        for target_idx in range(1, len(targets)):
            for pos_idx in range(target_idx, len(positive_idx)):
                prev_scores = dp[target_idx - 1, :pos_idx]
                if prev_scores.size == 0:
                    continue
                best_prev = int(np.argmin(prev_scores))
                best_score = float(prev_scores[best_prev])
                if not np.isfinite(best_score):
                    continue
                current_idx = positive_idx[pos_idx]
                dp[target_idx, pos_idx] = best_score + abs(float(activation[current_idx]) - float(targets[target_idx]))
                back[target_idx, pos_idx] = best_prev
        end_idx = int(np.argmin(dp[len(targets) - 1]))
        if np.isfinite(dp[len(targets) - 1, end_idx]):
            chosen_rev: List[int] = []
            cur_idx = end_idx
            for target_idx in range(len(targets) - 1, -1, -1):
                chosen_rev.append(positive_idx[cur_idx])
                if target_idx > 0:
                    cur_idx = int(back[target_idx, cur_idx])
                    if cur_idx < 0:
                        chosen_rev = []
                        break
            if chosen_rev:
                chosen_idx = list(reversed(chosen_rev))

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
        sample_spikes = encoder.forward(sample_imgs)[:, : spec.sample_steps, ...].contiguous()
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
                d_l1 = l1_spike_count_total / n_l1
                for row_idx, row in enumerate(batch.itertuples(index=False)):
                    item = {
                        "sample_index": int(row.sample_index),
                        "sample_label": int(row.sample_label),
                        "ping_drive_amp": float(ping_amp),
                        "ping_duration_ms": float(spec.ping_ms),
                        "delay2_ms": float(spec.post_ping_ms),
                        "l1_activation_fraction": float(ping_metrics["layer1"]["activation_fraction"][row_idx]),
                        "D_L1": float(d_l1[row_idx]),
                        "l1_spike_count_total": float(l1_spike_count_total[row_idx]),
                    }
                    for layer_key in LAYER_KEYS:
                        item[f"{layer_key}_activation_fraction"] = float(ping_metrics[layer_key]["activation_fraction"][row_idx])
                        item[f"{layer_key}_spike_rate_hz"] = float(ping_metrics[layer_key]["spike_rate_hz"][row_idx])
                        item[f"{layer_key}_active_fraction"] = float(ping_metrics[layer_key]["active_fraction"][row_idx])
                        item[f"{layer_key}_spike_count_total"] = float(ping_metrics[layer_key]["spike_count_total"][row_idx])
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
    condition_defs: Sequence[object],
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
    condition_defs: Optional[Sequence[object]] = None,
    s_sel_topk_frac: float = DEFAULT_S_SEL_TOPK_FRAC,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if condition_defs is None:
        condition_defs = _build_default_condition_defs(
            ping_target_fracs=ping_target_fracs,
            ping_duration_ms=float(spec.ping_ms),
            delay2_ms=float(spec.post_ping_ms),
            include_no_ping=include_no_ping_baseline,
        )
    condition_defs = list(condition_defs)
    if include_no_ping_baseline and not any(str(cond.control_mode) == CONTROL_DYNAMIC_NO_PING for cond in condition_defs):
        raise ValueError("include_no_ping_baseline=True requires a no-ping baseline condition in condition_defs.")

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
        sample_spikes = encoder.forward(sample_imgs)[:, : spec.sample_steps, ...].contiguous()
        probe_spikes = encoder.forward(probe_imgs)[:, : spec.probe_steps, ...].contiguous()

        donor_idx_b, plan_info = build_trial_shuffle_plan(
            sample_labels=sample_labels,
            probe_labels=probe_labels,
            rng=random.Random(seed + start + 991),
        )
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
                calibration_d_l1 = np.zeros(len(batch), dtype=np.float32)
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
                calibration_d_l1 = np.array(
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
            s_sel_metrics = compute_sample_selective_spike_fraction(
                sample_spike_count_vector=sample_l1_spike_count_vector,
                ping_spike_count_vector=l1_ping_spike_count_vector,
                topk_frac=s_sel_topk_frac,
            )
            n_l1 = float(max(1, l1_ping_spike_count_vector.shape[1]))
            d_l1 = (np.asarray(ping_metrics["layer1"]["spike_count_total"], dtype=np.float32) / n_l1).astype(np.float32, copy=False)
            s_l1 = np.asarray(s_sel_metrics["S_L1"], dtype=np.float32)
            e_sel_dose = (d_l1 * s_l1).astype(np.float32, copy=False)
            selectivity = compute_sample_aligned_selectivity(np.asarray(ping_metrics["layer3_class_counts"], dtype=np.float32), sample_labels)

            phase_state_map = {
                "pre_ping": out["pre_ping_ux"],
                "post_ping_immediate": out["post_ping_immediate_ux"],
                "pre_probe": out["pre_probe_ux"],
            }
            for layer_key in PHASE_LATENT_LAYERS:
                for phase, state_dict in phase_state_map.items():
                    gain = np.asarray(state_dict[layer_key]["gain"], dtype=np.float32)
                    for row_idx in range(len(batch)):
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
                                "trial_id": int(trial_ids[row_idx]),
                                "sample_label": int(sample_labels[row_idx]),
                                "feature": gain[row_idx].astype(np.float32, copy=False),
                            }
                        )

            for timepoint in out["ux_timecourse"]:
                for layer_key in LAYER_KEYS:
                    for row_idx in range(len(batch)):
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
                                "trial_id": int(trial_ids[row_idx]),
                                "layer": layer_key,
                                "phase": str(timepoint["phase"]),
                                "phase_step": int(timepoint["phase_step"]),
                                "time_step": int(timepoint["global_time_step"]),
                                "time_ms": float(timepoint["time_ms"]),
                                "u_value": float(timepoint["layer_means"][layer_key]["u"][row_idx]),
                                "x_value": float(timepoint["layer_means"][layer_key]["x"][row_idx]),
                                "ux_value": float(timepoint["layer_means"][layer_key]["ux"][row_idx]),
                            }
                        )

            for layer_key in LAYER_KEYS:
                active_fraction_t = np.asarray(ping_metrics[layer_key]["active_fraction_t"], dtype=np.float32)
                per_step_peak = np.zeros(len(batch), dtype=np.float32) if active_fraction_t.size == 0 else active_fraction_t.max(axis=0).astype(np.float32, copy=False)
                for row_idx in range(len(batch)):
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
                            "trial_id": int(trial_ids[row_idx]),
                            "sample_index": int(sample_indices[row_idx]),
                            "layer": layer_key,
                            "activation_fraction": float(ping_metrics[layer_key]["activation_fraction"][row_idx]),
                            "spike_rate_hz": float(ping_metrics[layer_key]["spike_rate_hz"][row_idx]),
                            "active_fraction": float(ping_metrics[layer_key]["active_fraction"][row_idx]),
                            "spike_count": float(ping_metrics[layer_key]["spike_count_total"][row_idx]),
                            "integrated_active_fraction_ms": float(ping_metrics[layer_key]["integrated_active_fraction_ms"][row_idx]),
                            "per_step_active_peak": float(per_step_peak[row_idx]),
                        }
                    )

            for row_idx in range(len(batch)):
                achieved_l1_activation = float(ping_metrics["layer1"]["activation_fraction"][row_idx])
                integrated_current = float(ping_amp_np[row_idx] * cond.ping_duration_ms)
                intervention_record = out.get("intervention_record", {"applied": 0})
                trial_rows.append(
                    {
                        "seed": int(seed),
                        "trial_id": int(trial_ids[row_idx]),
                        "sample_index": int(sample_indices[row_idx]),
                        "sample_label": int(sample_labels[row_idx]),
                        "probe_label": int(probe_labels[row_idx]),
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
                        "ping_drive_amp": float(ping_amp_np[row_idx]),
                        "selection_method": selection_method,
                        "selected_calibration_achieved_activation_frac": float(calibration_achieved[row_idx]),
                        "selected_calibration_D_L1": float(calibration_d_l1[row_idx]),
                        "selected_activation_residual": float(activation_residual[row_idx]),
                        "prediction_ping": int(pred_ping[row_idx]),
                        "first_fire_t_ping": int(fire_ping[row_idx]),
                        "prediction_probe": int(pred_probe[row_idx]),
                        "first_fire_t_probe": int(fire_probe[row_idx]),
                        "is_correct_ping": int(pred_ping[row_idx] == sample_labels[row_idx]),
                        "is_silent_ping": int(pred_ping[row_idx] == -1),
                        "is_correct_probe": int(pred_probe[row_idx] == probe_labels[row_idx]),
                        "is_silent_probe": int(pred_probe[row_idx] == -1),
                        "sample_aligned_selectivity": float(selectivity[row_idx]),
                        "D_L1": float(d_l1[row_idx]),
                        "S_L1": float(s_l1[row_idx]),
                        "E_sel_dose": float(e_sel_dose[row_idx]),
                        "d_int": float(d_l1[row_idx]),
                        "s_sel": float(s_l1[row_idx]),
                        "s_sel_topk": int(s_sel_metrics["topk_count"][row_idx]),
                        "ping_spikes_topk": float(s_sel_metrics["ping_topk_spikes"][row_idx]),
                        "ping_spikes_non_topk": float(s_sel_metrics["ping_non_topk_spikes"][row_idx]),
                        "sample_relevance_topk_mean": float(s_sel_metrics["topk_relevance_mean"][row_idx]),
                        "sample_relevance_non_topk_mean": float(s_sel_metrics["non_topk_relevance_mean"][row_idx]),
                        "l1_ping_activation_fraction": achieved_l1_activation,
                        "achieved_activation_frac": achieved_l1_activation,
                        "achieved_l1_activation_fraction": achieved_l1_activation,
                        "dose_l1_activation_ms": achieved_l1_activation * cond.ping_duration_ms,
                        "integrated_perturbation_current": integrated_current,
                        "l1_ping_spike_rate_hz": float(ping_metrics["layer1"]["spike_rate_hz"][row_idx]),
                        "l2_ping_spike_rate_hz": float(ping_metrics["layer2"]["spike_rate_hz"][row_idx]),
                        "l3_ping_spike_rate_hz": float(ping_metrics["layer3"]["spike_rate_hz"][row_idx]),
                        "l1_ping_active_fraction": float(ping_metrics["layer1"]["active_fraction"][row_idx]),
                        "l2_ping_active_fraction": float(ping_metrics["layer2"]["active_fraction"][row_idx]),
                        "l3_ping_active_fraction": float(ping_metrics["layer3"]["active_fraction"][row_idx]),
                        "l1_ping_spike_count": float(ping_metrics["layer1"]["spike_count_total"][row_idx]),
                        "l2_ping_spike_count": float(ping_metrics["layer2"]["spike_count_total"][row_idx]),
                        "l3_ping_spike_count": float(ping_metrics["layer3"]["spike_count_total"][row_idx]),
                        "l2_integrated_active_fraction_ms": float(ping_metrics["layer2"]["integrated_active_fraction_ms"][row_idx]),
                        "l3_integrated_active_fraction_ms": float(ping_metrics["layer3"]["integrated_active_fraction_ms"][row_idx]),
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
    base = df_metrics_seed[df_metrics_seed["control_mode"] == CONTROL_DYNAMIC_NO_PING][["seed"] + list(metric_cols)].rename(columns={col: f"{col}_baseline" for col in metric_cols}).drop_duplicates(subset=["seed"]).reset_index(drop=True)
    merged = df_metrics_seed.merge(base, on="seed", how="left")
    for col in metric_cols:
        merged[f"delta_{col}"] = merged[col] - merged[f"{col}_baseline"]
    return merged


def compute_delta_summary(df_metrics_seed: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["decode_ping_acc", "sample_bias_minus_noise", "probe_accuracy", "error_to_sample_fraction", "D_L1", "S_L1", "E_sel_dose", "l2_ping_spike_count", "l2_ping_active_fraction", "l3_ping_spike_count", "l3_ping_active_fraction"]
    delta_cols = [f"delta_{col}" for col in metric_cols]
    merged = df_metrics_seed.copy()
    if any(col not in merged.columns for col in delta_cols):
        merged = add_baseline_deltas(merged, metric_cols=metric_cols)
    group_cols = ["condition", "condition_label", "condition_family", "regime", "control_mode", "analysis_role", "ping_duration_ms", "delay2_ms", "ping_target_label", "ping_target_frac"]
    return merged.groupby(group_cols, as_index=False).agg(
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
        delta_D_L1_mean=("delta_D_L1", "mean"),
        delta_D_L1_sem=("delta_D_L1", "sem"),
        delta_S_L1_mean=("delta_S_L1", "mean"),
        delta_S_L1_sem=("delta_S_L1", "sem"),
        delta_E_sel_dose_mean=("delta_E_sel_dose", "mean"),
        delta_E_sel_dose_sem=("delta_E_sel_dose", "sem"),
        delta_l2_ping_spike_count_mean=("delta_l2_ping_spike_count", "mean"),
        delta_l2_ping_spike_count_sem=("delta_l2_ping_spike_count", "sem"),
        delta_l2_ping_active_fraction_mean=("delta_l2_ping_active_fraction", "mean"),
        delta_l2_ping_active_fraction_sem=("delta_l2_ping_active_fraction", "sem"),
        delta_l3_ping_spike_count_mean=("delta_l3_ping_spike_count", "mean"),
        delta_l3_ping_spike_count_sem=("delta_l3_ping_spike_count", "sem"),
        delta_l3_ping_active_fraction_mean=("delta_l3_ping_active_fraction", "mean"),
        delta_l3_ping_active_fraction_sem=("delta_l3_ping_active_fraction", "sem"),
    ).sort_values(["control_mode", "regime", "ping_duration_ms", "ping_target_frac"], kind="stable").reset_index(drop=True)


__all__ = ["DEFAULT_TAU_U_MS", "calibrate_ping_per_example", "compute_delta_summary", "run_seed_experiment"]
