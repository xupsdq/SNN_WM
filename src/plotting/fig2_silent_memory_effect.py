import argparse
import math
import os
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
from tqdm import tqdm

from src.experiments.ping_memory.shared.ping_api import compute_sample_and_noise_bias
from src.experiments.silent_memory.shared.population_dms import (
    ExperimentSpec,
    LAYER_KEYS,
    STSP_MODES,
    bootstrap_mean_ci,
    build_class_index,
    build_trial_phase_rate_table,
    encode_images,
    generate_balanced_dms_trial_specs,
    load_model_and_encoder,
    seed_everything,
    validate_trial_specs,
)
from src.experiments.silent_memory.shared.raster_utils import flatten_single_trial_spikes
from figure_utils_common import (
    COLOR_DYNAMIC,
    COLOR_NOISE,
    COLOR_PING,
    COLOR_STATIC,
    PUBLICATION_ANNOTATION_FONT_SIZE,
    PUBLICATION_ERRORBAR_CAPSIZE,
    PUBLICATION_LINE_WIDTH,
    PUBLICATION_MARKER_SIZE,
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
    select_representative_trial,
    validate_required_columns,
)
from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from paper_plot_style import FIGURE2_SUBPLOT_ADJUST, PANEL_LABEL_FONT_SIZE, apply_paper_style
from src.platform.legacy_adapters.units import ms


DEFAULT_DELAY_SWEEP_MS = [100, 300, 600, 1000, 1500, 2000, 3000, 4000]
PHASE_ORDER = ["sample", "delay", "probe"]
PHASE_LABELS = {"sample": "Sample", "delay": "Delay", "probe": "Probe"}
PHASE_COLORS = {
    "sample": COLOR_DYNAMIC,
    "delay": COLOR_STATIC,
    "probe": COLOR_PING if COLOR_PING != COLOR_STATIC else COLOR_NOISE,
}
MODE_COLORS = {
    "dynamic": COLOR_DYNAMIC,
    "static_frozen": COLOR_STATIC,
}
MODE_LABELS = {
    "dynamic": "Dynamic",
    "static_frozen": "Static",
}


def format_compact_count(value: float, _pos: float) -> str:
    value = float(value)
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:.1f}k"
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:g}"


def format_small_limit(value: float, _pos: float) -> str:
    value = float(value)
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "0"
    if abs(value) < 0.1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.2g}"


def format_scaled_ticks(scale: float):
    def _formatter(value: float, _pos: float) -> str:
        scaled = float(value) / float(scale)
        if math.isclose(scaled, round(scaled), abs_tol=1e-9):
            return str(int(round(scaled)))
        return f"{scaled:g}"

    return _formatter


def parse_delay_list(raw_value: str) -> List[float]:
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("delay list is empty")
    parsed = [float(item) for item in values]
    return sorted(dict.fromkeys(parsed))


def format_delay_ms(delay_ms: float) -> str:
    rounded = round(float(delay_ms))
    if math.isclose(float(delay_ms), float(rounded), abs_tol=1e-9):
        return str(int(rounded))
    return f"{float(delay_ms):g}"


def delay_to_steps(delay_ms: float, dt: float) -> int:
    steps = int(round((float(delay_ms) * ms) / dt))
    if steps <= 0:
        raise ValueError(f"delay_ms must map to positive steps, got delay_ms={delay_ms}")
    return steps


def exponential_decay(time_ms: np.ndarray, amp: float, tau: float, offset: float) -> np.ndarray:
    return amp * np.exp(-time_ms / tau) + offset


def _memory_effect_from_arrays(
    pred_label: np.ndarray,
    sample_label: np.ndarray,
    probe_label: np.ndarray,
    num_classes: int,
) -> float:
    error_mask = pred_label != probe_label
    if not np.any(error_mask):
        return 0.0

    pred_err = pred_label[error_mask]
    sample_err = sample_label[error_mask]
    probe_err = probe_label[error_mask]
    bias_sample = float(np.mean(pred_err == sample_err))

    valid = (pred_err >= 0) & (pred_err < num_classes)
    k = num_classes - 2
    if k <= 0:
        raise ValueError("num_classes is too small for memory-effect noise baseline")
    noise_hit = valid & (pred_err != sample_err) & (pred_err != probe_err)
    bias_noise = float(noise_hit.sum() / float(len(pred_err) * k))
    return 100.0 * float(bias_sample - bias_noise)


def compute_memory_effect(df_subset: pd.DataFrame, num_classes: int) -> float:
    required_cols = ["sample_label", "probe_label", "prediction_probe"]
    validate_required_columns(df_subset, required_cols)

    bias_sample, bias_noise = compute_sample_and_noise_bias(df_subset[required_cols].copy(), num_classes=num_classes)
    memory_effect = 100.0 * float(bias_sample - bias_noise)

    pred = df_subset["prediction_probe"].to_numpy(dtype=np.int64)
    sample = df_subset["sample_label"].to_numpy(dtype=np.int64)
    probe = df_subset["probe_label"].to_numpy(dtype=np.int64)
    fast_value = _memory_effect_from_arrays(pred, sample, probe, num_classes=num_classes)
    if not math.isclose(memory_effect, fast_value, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Memory-effect helper drifted from shared sample/noise bias definition")
    return memory_effect


def bootstrap_memory_effect_ci(
    df_subset: pd.DataFrame,
    num_classes: int,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    validate_required_columns(df_subset, ["sample_label", "probe_label", "prediction_probe"])
    if len(df_subset) == 0:
        raise ValueError("bootstrap_memory_effect_ci received empty subset")

    pred = df_subset["prediction_probe"].to_numpy(dtype=np.int64)
    sample = df_subset["sample_label"].to_numpy(dtype=np.int64)
    probe = df_subset["probe_label"].to_numpy(dtype=np.int64)
    n = len(df_subset)
    rng = np.random.default_rng(seed)
    values = np.zeros(n_boot, dtype=np.float64)
    for boot_idx in range(n_boot):
        resample_idx = rng.integers(0, n, size=n)
        values[boot_idx] = _memory_effect_from_arrays(
            pred_label=pred[resample_idx],
            sample_label=sample[resample_idx],
            probe_label=probe[resample_idx],
            num_classes=num_classes,
        )
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def run_delay_sweep_memory_effect(
    net,
    encoder,
    dataset,
    device: torch.device,
    df_specs: pd.DataFrame,
    delay_values_ms: Sequence[float],
    canonical_delay_ms: float,
    batch_size: int,
    spec: ExperimentSpec,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    executed_delays = sorted(dict.fromkeys([float(v) for v in delay_values_ms] + [float(canonical_delay_ms)]))
    trial_rows: List[Dict[str, float]] = []
    phase_tables: List[pd.DataFrame] = []
    batch_starts = range(0, len(df_specs), batch_size)

    for start in tqdm(batch_starts, desc="Figure 2 delay sweep"):
        batch_df = df_specs.iloc[start:start + batch_size].copy().reset_index(drop=True)
        sample_imgs = torch.stack([dataset[int(i)][0] for i in batch_df["sample_index"].tolist()], dim=0).to(device)
        probe_imgs = torch.stack([dataset[int(i)][0] for i in batch_df["probe_index"].tolist()], dim=0).to(device)
        sample_spikes = encode_images(encoder, sample_imgs, spec.sample_steps)
        probe_spikes = encode_images(encoder, probe_imgs, spec.probe_steps)

        for delay_ms in executed_delays:
            delay_steps = delay_to_steps(delay_ms, dt=spec.dt)
            is_canonical = math.isclose(float(delay_ms), float(canonical_delay_ms), abs_tol=1e-9)

            for stsp_mode in STSP_MODES:
                with torch.no_grad():
                    trace = net.forward_dms_spike_trace_session(
                        sample_spikes=sample_spikes,
                        probe_spikes=probe_spikes,
                        delay_steps=delay_steps,
                        stsp_mode=stsp_mode,
                        phase_reset=spec.phase_reset,
                    )

                pred = trace["predictions"]
                pred_labels = pred["prediction_probe"].cpu().numpy().astype(np.int64, copy=False)
                first_fire = pred["first_fire_t_probe"].cpu().numpy().astype(np.int64, copy=False)
                sample_labels = batch_df["sample_label"].to_numpy(dtype=np.int64)
                probe_labels = batch_df["probe_label"].to_numpy(dtype=np.int64)
                sample_indices = batch_df["sample_index"].to_numpy(dtype=np.int64)
                probe_indices = batch_df["probe_index"].to_numpy(dtype=np.int64)
                trial_ids = batch_df["trial_id"].to_numpy(dtype=np.int64)

                for idx in range(len(batch_df)):
                    pred_i = int(pred_labels[idx])
                    ff_i = int(first_fire[idx])
                    sample_i = int(sample_labels[idx])
                    probe_i = int(probe_labels[idx])
                    trial_rows.append(
                        {
                            "trial_id": int(trial_ids[idx]),
                            "stsp_mode": stsp_mode,
                            "sample_label": sample_i,
                            "probe_label": probe_i,
                            "sample_index": int(sample_indices[idx]),
                            "probe_index": int(probe_indices[idx]),
                            "delay_ms": float(delay_ms),
                            "pred_label": pred_i,
                            "prediction_probe": pred_i,
                            "is_correct": int(pred_i == probe_i),
                            "is_silent": int(pred_i < 0 or ff_i < 0),
                            "first_fire_t_probe": ff_i,
                            "is_canonical_delay": int(is_canonical),
                        }
                    )

                if is_canonical:
                    phase_slices = trace["phase_slices"]
                    for layer_name in LAYER_KEYS:
                        phase_tables.append(
                            build_trial_phase_rate_table(
                                spikes=trace[f"{layer_name}_spikes"],
                                phase_slices=phase_slices,
                                layer_name=layer_name,
                                batch_df=batch_df,
                                stsp_mode=stsp_mode,
                            )
                        )

    df_trial_level = pd.DataFrame(trial_rows)
    df_phase_rates = pd.concat(phase_tables, axis=0, ignore_index=True) if phase_tables else pd.DataFrame()
    return df_trial_level, df_phase_rates


def summarize_memory_effect_vs_delay(
    df_trial_level: pd.DataFrame,
    delay_values_ms: Sequence[float],
    num_classes: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    required_cols = ["delay_ms", "stsp_mode", "sample_label", "probe_label", "prediction_probe"]
    validate_required_columns(df_trial_level, required_cols)
    sweep_values = np.array([float(v) for v in delay_values_ms], dtype=np.float64)
    mask = np.isclose(
        df_trial_level["delay_ms"].to_numpy(dtype=np.float64)[:, None],
        sweep_values[None, :],
        atol=1e-9,
    ).any(axis=1)
    df_sweep = df_trial_level.loc[mask].copy()
    if df_sweep.empty:
        raise ValueError("No delay-sweep rows remain after filtering")

    rows: List[Dict[str, float]] = []
    grouped = df_sweep.groupby(["delay_ms", "stsp_mode"], sort=True)
    for group_idx, ((delay_ms, stsp_mode), sub) in enumerate(grouped):
        effect_mean = compute_memory_effect(sub, num_classes=num_classes)
        ci_lower, ci_upper = bootstrap_memory_effect_ci(
            df_subset=sub,
            num_classes=num_classes,
            n_boot=n_boot,
            seed=seed + 1000 + group_idx * 37,
        )
        rows.append(
            {
                "delay_ms": float(delay_ms),
                "stsp_mode": str(stsp_mode),
                "memory_effect_mean": float(effect_mean),
                "memory_effect_ci95_lower": float(ci_lower),
                "memory_effect_ci95_upper": float(ci_upper),
                "fit_y": float("nan"),
                "fit_tau": float("nan"),
            }
        )

    df_metrics = pd.DataFrame(rows).sort_values(["stsp_mode", "delay_ms"], kind="stable").reset_index(drop=True)
    if df_metrics.empty:
        raise ValueError("Failed to summarize memory effect")

    for stsp_mode in STSP_MODES:
        mode_mask = df_metrics["stsp_mode"] == stsp_mode
        sub = df_metrics.loc[mode_mask].copy().sort_values("delay_ms")
        x = sub["delay_ms"].to_numpy(dtype=np.float64)
        y = sub["memory_effect_mean"].to_numpy(dtype=np.float64)
        fit_y = np.full_like(y, np.nan, dtype=np.float64)
        fit_tau = float("nan")
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if finite_mask.sum() >= 3:
            x_fit = x[finite_mask]
            y_fit = y[finite_mask]
            amp0 = float(max(np.max(y_fit) - np.min(y_fit), 1e-3))
            tau0 = float(np.median(x_fit))
            offset0 = float(np.min(y_fit))
            try:
                popt, _ = curve_fit(
                    exponential_decay,
                    xdata=x_fit,
                    ydata=y_fit,
                    p0=[amp0, tau0, offset0],
                    bounds=([-np.inf, 1e-3, -np.inf], [np.inf, 1e7, np.inf]),
                    maxfev=10000,
                )
                fit_y = exponential_decay(x, *popt)
                fit_tau = float(popt[1])
            except Exception:
                fit_y = np.full_like(y, np.nan, dtype=np.float64)
                fit_tau = float("nan")

        df_metrics.loc[mode_mask, "fit_y"] = fit_y
        df_metrics.loc[mode_mask, "fit_tau"] = fit_tau

    return df_metrics


def summarize_phase_firing(
    df_phase_rates: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    required_cols = ["layer", "stsp_mode", "phase", "rate_spikes_per_neuron_step"]
    validate_required_columns(df_phase_rates, required_cols)

    rows: List[Dict[str, float]] = []
    for layer_idx, layer_name in enumerate(LAYER_KEYS):
        for mode_idx, stsp_mode in enumerate(STSP_MODES):
            sub = df_phase_rates[
                (df_phase_rates["layer"] == layer_name) & (df_phase_rates["stsp_mode"] == stsp_mode)
            ].copy()
            if sub.empty:
                continue

            phase_values: Dict[str, np.ndarray] = {}
            phase_ci: Dict[str, Tuple[float, float]] = {}
            for phase_idx, phase in enumerate(PHASE_ORDER):
                values = sub[sub["phase"] == phase]["rate_spikes_per_neuron_step"].to_numpy(dtype=np.float64)
                if len(values) == 0:
                    raise ValueError(f"Missing phase values for {layer_name} / {stsp_mode} / {phase}")
                phase_values[phase] = values
                phase_ci[phase] = bootstrap_mean_ci(
                    values,
                    n_boot=n_boot,
                    seed=seed + 2000 + layer_idx * 100 + mode_idx * 17 + phase_idx,
                )

            rows.append(
                {
                    "layer": layer_name,
                    "stsp_mode": stsp_mode,
                    "sample_rate_mean": float(phase_values["sample"].mean()),
                    "sample_rate_ci95_lower": float(phase_ci["sample"][0]),
                    "sample_rate_ci95_upper": float(phase_ci["sample"][1]),
                    "delay_rate_mean": float(phase_values["delay"].mean()),
                    "delay_rate_ci95_lower": float(phase_ci["delay"][0]),
                    "delay_rate_ci95_upper": float(phase_ci["delay"][1]),
                    "probe_rate_mean": float(phase_values["probe"].mean()),
                    "probe_rate_ci95_lower": float(phase_ci["probe"][0]),
                    "probe_rate_ci95_upper": float(phase_ci["probe"][1]),
                }
            )
    return pd.DataFrame(rows).sort_values(["layer", "stsp_mode"], kind="stable").reset_index(drop=True)


def infer_phase_name(t_step: int, phase_slices: Mapping[str, Sequence[int]]) -> str:
    for phase in PHASE_ORDER:
        start, end = phase_slices[phase]
        if int(start) <= int(t_step) < int(end):
            return phase
    return "unknown"


def build_representative_raster_data(
    net,
    encoder,
    dataset,
    device: torch.device,
    spec: ExperimentSpec,
    df_trial_level: pd.DataFrame,
    df_specs: pd.DataFrame,
    canonical_delay_ms: float,
    stsp_mode: str = "dynamic",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    validate_required_columns(
        df_trial_level,
        ["trial_id", "stsp_mode", "delay_ms", "sample_label", "probe_label", "is_correct", "is_silent", "first_fire_t_probe"],
    )
    validate_required_columns(df_specs, ["trial_id", "sample_index", "probe_index", "sample_label", "probe_label"])

    delay_mask = np.isclose(df_trial_level["delay_ms"].to_numpy(dtype=np.float64), float(canonical_delay_ms), atol=1e-9)
    sub = df_trial_level[(df_trial_level["stsp_mode"] == stsp_mode) & delay_mask].copy()
    if sub.empty:
        raise ValueError("No representative-trial candidates found for canonical delay")

    sub["condition_key"] = f"{stsp_mode}|{format_delay_ms(canonical_delay_ms)}"
    selected_trial_id = select_representative_trial(
        sub,
        condition_col="condition_key",
        correct_col="is_correct",
        silent_col="is_silent",
        first_fire_col="first_fire_t_probe",
    )
    row = sub[sub["trial_id"] == selected_trial_id].sort_values("trial_id").iloc[0]

    spec_row = df_specs[df_specs["trial_id"] == selected_trial_id]
    if len(spec_row) != 1:
        raise ValueError(f"Representative trial_id={selected_trial_id} is missing or duplicated in df_specs")
    spec_row = spec_row.iloc[0]

    sample_img = dataset[int(spec_row["sample_index"])][0].unsqueeze(0).to(device)
    probe_img = dataset[int(spec_row["probe_index"])][0].unsqueeze(0).to(device)
    sample_spikes = encode_images(encoder, sample_img, spec.sample_steps)
    probe_spikes = encode_images(encoder, probe_img, spec.probe_steps)

    with torch.no_grad():
        trace = net.forward_dms_spike_trace_session(
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=delay_to_steps(canonical_delay_ms, dt=spec.dt),
            stsp_mode=stsp_mode,
            phase_reset=spec.phase_reset,
        )

    phase_slices = trace["phase_slices"]
    point_rows: List[Dict[str, object]] = []
    rate_rows: List[Dict[str, object]] = []
    layer_neuron_counts: Dict[str, int] = {}
    layer_counts_by_time: Dict[str, np.ndarray] = {}
    layer_rates_by_time: Dict[str, np.ndarray] = {}
    pooled_counts: Optional[np.ndarray] = None
    total_neurons = 0

    for layer_name in LAYER_KEYS:
        flat_spikes = flatten_single_trial_spikes(trace[f"{layer_name}_spikes"])
        t_idx, neuron_idx = np.where(flat_spikes)
        neuron_count = int(flat_spikes.shape[1])
        spike_counts = flat_spikes.sum(axis=1).astype(np.int64, copy=False)
        rate_per_neuron = spike_counts.astype(np.float64) / float(max(1, neuron_count))
        layer_neuron_counts[layer_name] = neuron_count
        layer_counts_by_time[layer_name] = spike_counts
        layer_rates_by_time[layer_name] = rate_per_neuron
        pooled_counts = spike_counts.astype(np.float64) if pooled_counts is None else pooled_counts + spike_counts
        total_neurons += neuron_count
        for t_step, neuron_index in zip(t_idx.tolist(), neuron_idx.tolist()):
            point_rows.append(
                {
                    "trial_id": int(selected_trial_id),
                    "stsp_mode": stsp_mode,
                    "delay_ms": float(canonical_delay_ms),
                    "layer": layer_name,
                    "t_step": int(t_step),
                    "neuron_index": int(neuron_index),
                    "phase": infer_phase_name(int(t_step), phase_slices=phase_slices),
                }
            )

    if pooled_counts is None:
        raise ValueError("Representative trace returned no layers for pooled firing-rate computation")
    pooled_rate = pooled_counts / float(max(1, total_neurons))
    for t_step in range(len(pooled_rate)):
        row_dict: Dict[str, object] = {
            "trial_id": int(selected_trial_id),
            "stsp_mode": stsp_mode,
            "delay_ms": float(canonical_delay_ms),
            "t_step": int(t_step),
            "time_ms": float(t_step) * float(spec.dt / ms),
            "phase": infer_phase_name(int(t_step), phase_slices=phase_slices),
            "pooled_spike_count": float(pooled_counts[t_step]),
            "pooled_rate_spikes_per_neuron_step": float(pooled_rate[t_step]),
            "total_neurons": int(total_neurons),
        }
        for layer_name in LAYER_KEYS:
            row_dict[f"{layer_name}_spike_count"] = int(layer_counts_by_time[layer_name][t_step])
            row_dict[f"{layer_name}_rate_spikes_per_neuron_step"] = float(layer_rates_by_time[layer_name][t_step])
        rate_rows.append(row_dict)

    selection_reason = (
        "Selected by figure_utils_common.select_representative_trial on dynamic canonical delay; "
        "eligible pool constrained to correct and non-silent trials, then chosen by median first-fire proximity "
        "with label-frequency and trial-id tie-breaks."
    )
    df_metrics = pd.DataFrame(
        [
            {
                "trial_id": int(selected_trial_id),
                "stsp_mode": stsp_mode,
                "sample_label": int(row["sample_label"]),
                "probe_label": int(row["probe_label"]),
                "delay_ms": float(canonical_delay_ms),
                "first_fire_t_probe": int(row["first_fire_t_probe"]),
                "selection_reason": selection_reason,
            }
        ]
    )
    df_points = pd.DataFrame(point_rows)
    df_population_rate = pd.DataFrame(rate_rows)
    return df_metrics, df_points, df_population_rate, {
        "phase_slices": phase_slices,
        "layer_neuron_counts": layer_neuron_counts,
        "total_neurons": int(total_neurons),
    }


def plot_memory_effect_vs_delay(ax: plt.Axes, df_metrics: pd.DataFrame) -> None:
    validate_required_columns(
        df_metrics,
        ["delay_ms", "stsp_mode", "memory_effect_mean", "memory_effect_ci95_lower", "memory_effect_ci95_upper", "fit_y", "fit_tau"],
    )
    for stsp_mode in STSP_MODES:
        sub = df_metrics[df_metrics["stsp_mode"] == stsp_mode].copy().sort_values("delay_ms")
        if sub.empty:
            continue

        x = sub["delay_ms"].to_numpy(dtype=np.float64)
        y = sub["memory_effect_mean"].to_numpy(dtype=np.float64)
        lo = sub["memory_effect_ci95_lower"].to_numpy(dtype=np.float64)
        hi = sub["memory_effect_ci95_upper"].to_numpy(dtype=np.float64)
        color = MODE_COLORS[stsp_mode]
        label = MODE_LABELS[stsp_mode]
        ax.plot(x, y, marker="o", color=color, linewidth=PUBLICATION_LINE_WIDTH + 0.7, label=label)

        fit_y = sub["fit_y"].to_numpy(dtype=np.float64)
        if np.isfinite(fit_y).all():
            ax.plot(x, fit_y, linestyle="--", color=color, linewidth=1.2, alpha=0.45, label=f"{label} fit")

    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Memory effect (pp)")
    ax.legend(loc="upper right")


def add_phase_guides(ax: plt.Axes, phase_slices: Mapping[str, Sequence[int]], dt_ms: float) -> None:
    phase_alpha = {"sample": 0.10, "delay": 0.11, "probe": 0.10}
    for phase in PHASE_ORDER:
        start, end = phase_slices[phase]
        ax.axvspan(start * dt_ms, end * dt_ms, color=PHASE_COLORS[phase], alpha=phase_alpha[phase], linewidth=0)
        ax.axvline(start * dt_ms, color=PHASE_COLORS[phase], linewidth=1.1, linestyle="--", alpha=0.85)


def plot_representative_population_rate(
    ax: plt.Axes,
    df_population_rate: pd.DataFrame,
    phase_slices: Mapping[str, Sequence[int]],
    dt_ms: float,
) -> None:
    validate_required_columns(
        df_population_rate,
        ["t_step", "time_ms", "phase", "pooled_rate_spikes_per_neuron_step"],
    )
    sub = df_population_rate.sort_values("t_step", kind="stable").reset_index(drop=True)
    add_phase_guides(ax, phase_slices=phase_slices, dt_ms=dt_ms)
    x = sub["time_ms"].to_numpy(dtype=np.float64, copy=False)
    y = sub["pooled_rate_spikes_per_neuron_step"].to_numpy(dtype=np.float64, copy=False)
    ax.plot(x, y, color=COLOR_DYNAMIC, linewidth=PUBLICATION_LINE_WIDTH)
    ax.fill_between(x, 0.0, y, color=COLOR_DYNAMIC, alpha=0.14, linewidth=0)
    ax.set_ylabel("")
    ax.text(
        -0.02,
        0.5,
        "Pooled\nrate",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=PUBLICATION_ANNOTATION_FONT_SIZE + 1,
        fontweight="semibold",
        color="#222222",
    )
    ax.set_xlabel("Time (ms)")
    ax.tick_params(axis="x", labelbottom=True)
    ymax = max(float(np.max(y)) if len(y) > 0 else 0.0, 1e-6)
    ylim_top = ymax * 1.25
    x_max = float(np.max(x)) if len(x) > 0 else 1.0
    left_margin = max(5.0 * dt_ms, 0.01 * x_max)
    ax.set_xlim(-left_margin, x_max)
    ax.set_ylim(0.0, ylim_top)
    ax.set_yticks([0.0, ylim_top], labels=["", format_small_limit(ylim_top, 0.0)])
    ax.tick_params(axis="y", pad=12, labelsize=10)
    ax.tick_params(axis="x", pad=6)
    xticks = ax.get_xticks()
    xlabels = ["0" if math.isclose(float(tick), 0.0, abs_tol=1e-9) else f"{tick:g}" for tick in xticks]
    ax.set_xticks(xticks, xlabels)
    delay_start, delay_end = phase_slices["delay"]
    delay_mid_ms = 0.5 * float(delay_start + delay_end) * dt_ms
    ax.text(
        delay_mid_ms,
        ymax * 1.10,
        "Delay period (no sustained firing)",
        ha="center",
        va="center",
        fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
        bbox={"facecolor": "white", "edgecolor": COLOR_STATIC, "boxstyle": "round,pad=0.18", "alpha": 0.9},
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_representative_raster(
    axes: Sequence[plt.Axes],
    df_points: pd.DataFrame,
    phase_slices: Mapping[str, Sequence[int]],
    layer_neuron_counts: Mapping[str, int],
    dt_ms: float,
    layers: Optional[Sequence[str]] = None,
) -> None:
    validate_required_columns(df_points, ["layer", "t_step", "neuron_index"])
    layers = list(LAYER_KEYS if layers is None else layers)
    if len(axes) != len(layers):
        raise ValueError("Representative raster plot expects one axis per requested layer")

    total_steps = int(phase_slices["probe"][1])
    total_ms = total_steps * dt_ms
    for idx, (ax, layer_name) in enumerate(zip(axes, layers)):
        add_phase_guides(ax, phase_slices=phase_slices, dt_ms=dt_ms)
        sub = df_points[df_points["layer"] == layer_name].copy()
        if not sub.empty:
            ax.scatter(
                sub["t_step"].to_numpy(dtype=np.float64) * dt_ms,
                sub["neuron_index"].to_numpy(dtype=np.float64),
                s=PUBLICATION_MARKER_SIZE,
                c="black",
                marker=".",
                linewidths=0,
                alpha=0.9,
            )
        max_neuron = float(max(1, int(layer_neuron_counts.get(layer_name, 1))))
        ax.set_xlim(0.0, total_ms)
        ax.set_ylim(0.0, max_neuron)
        ax.set_ylabel("")
        ax.text(
            -0.02,
            0.5,
            f"Layer {idx + 1}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=PUBLICATION_ANNOTATION_FONT_SIZE + 1,
            fontweight="semibold",
            color="#222222",
        )
        ax.set_yticks([])
        ax.tick_params(axis="y", left=False, labelleft=False)
        if idx != len(layers):
            ax.tick_params(axis="x", labelbottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def plot_phase_firing_summary(axes: Sequence[plt.Axes], df_phase_summary: pd.DataFrame) -> None:
    validate_required_columns(
        df_phase_summary,
        [
            "layer",
            "stsp_mode",
            "sample_rate_mean",
            "sample_rate_ci95_lower",
            "sample_rate_ci95_upper",
            "delay_rate_mean",
            "delay_rate_ci95_lower",
            "delay_rate_ci95_upper",
            "probe_rate_mean",
            "probe_rate_ci95_lower",
            "probe_rate_ci95_upper",
        ],
    )
    if len(axes) != len(LAYER_KEYS):
        raise ValueError("Phase-firing summary expects one axis per layer")

    width = 0.34
    x = np.arange(len(PHASE_ORDER), dtype=np.float64)
    delay_idx = PHASE_ORDER.index("delay")
    for ax, layer_name in zip(axes, LAYER_KEYS):
        sub = df_phase_summary[df_phase_summary["layer"] == layer_name].copy()
        sub["mode_order"] = sub["stsp_mode"].map({mode: idx for idx, mode in enumerate(STSP_MODES)})
        sub = sub.sort_values("mode_order")
        max_upper = 0.0
        for mode_idx, stsp_mode in enumerate(STSP_MODES):
            row = sub[sub["stsp_mode"] == stsp_mode]
            if row.empty:
                continue
            row = row.iloc[0]
            vals = np.array(
                [
                    float(row["sample_rate_mean"]),
                    float(row["delay_rate_mean"]),
                    float(row["probe_rate_mean"]),
                ],
                dtype=np.float64,
            )
            lower = np.array(
                [
                    float(row["sample_rate_ci95_lower"]),
                    float(row["delay_rate_ci95_lower"]),
                    float(row["probe_rate_ci95_lower"]),
                ],
                dtype=np.float64,
            )
            upper = np.array(
                [
                    float(row["sample_rate_ci95_upper"]),
                    float(row["delay_rate_ci95_upper"]),
                    float(row["probe_rate_ci95_upper"]),
                ],
                dtype=np.float64,
            )
            max_upper = max(max_upper, float(np.max(upper)))
            pos = x + (mode_idx - 0.5) * width + width / 2.0
            yerr = np.vstack([vals - lower, upper - vals])
            ax.bar(
                pos,
                vals,
                width=width,
                yerr=yerr,
                color=MODE_COLORS[stsp_mode],
                edgecolor="black",
                alpha=0.92,
                capsize=PUBLICATION_ERRORBAR_CAPSIZE,
                label=MODE_LABELS[stsp_mode],
            )
            delay_y = vals[delay_idx]
        ax.set_xticks(x, [PHASE_LABELS[p] for p in PHASE_ORDER])
        ax.set_ylabel("Spike rate / neuron / step")
        ylim_top = max_upper * 1.2 if max_upper > 0.0 else 1.0
        ax.set_ylim(0.0, ylim_top)
        scale = 1e-4
        tick_max = max(1, int(math.ceil(ylim_top / scale)))
        ax.set_yticks(np.arange(0, tick_max + 1, dtype=np.float64) * scale)
        ax.yaxis.set_major_formatter(FuncFormatter(format_scaled_ticks(scale)))
        ax.text(
            0.00,
            1.01,
            r"$\times 10^{-4}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PUBLICATION_ANNOTATION_FONT_SIZE + 1,
            fontweight="bold",
            color="#444444",
        )
        for mode_idx, stsp_mode in enumerate(STSP_MODES):
            row = sub[sub["stsp_mode"] == stsp_mode]
            if row.empty:
                continue
            row = row.iloc[0]
            delay_y = float(row["delay_rate_mean"])
            if abs(delay_y) < 5e-4:
                continue
            pos = x + (mode_idx - 0.5) * width + width / 2.0
            text_y = min(delay_y + max(0.015 * ylim_top, 0.0002), 0.94 * ylim_top)
            ax.text(
                pos[delay_idx],
                text_y,
                f"{delay_y:.3f}",
                ha="center",
                va="bottom",
                fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
                color="#222222",
            )
    legend_handles = [
        Patch(facecolor=MODE_COLORS["dynamic"], edgecolor="#222222", label=MODE_LABELS["dynamic"]),
        Patch(facecolor=MODE_COLORS["static_frozen"], edgecolor="#222222", label=MODE_LABELS["static_frozen"]),
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.4,
    )


def plot_representative_activity_panel(
    fig: plt.Figure,
    subplot_spec,
    df_raster_points: pd.DataFrame,
    df_population_rate: pd.DataFrame,
    representative_meta: Mapping[str, object],
    dt_ms: float,
) -> tuple[list[plt.Axes], plt.Axes]:
    panel_grid = subplot_spec.subgridspec(3, 1, height_ratios=[0.95, 0.95, 1.0], hspace=0.08)
    raster_axes = [fig.add_subplot(panel_grid[i, 0]) for i in range(2)]
    ax_rate = fig.add_subplot(panel_grid[2, 0], sharex=raster_axes[0])
    plot_representative_population_rate(
        ax_rate,
        df_population_rate=df_population_rate,
        phase_slices=representative_meta["phase_slices"],
        dt_ms=dt_ms,
    )
    plot_representative_raster(
        raster_axes,
        df_points=df_raster_points,
        phase_slices=representative_meta["phase_slices"],
        layer_neuron_counts=representative_meta["layer_neuron_counts"],
        dt_ms=dt_ms,
        layers=LAYER_KEYS[:2],
    )
    return raster_axes, ax_rate


def assemble_figure_main(
    df_memory_effect: pd.DataFrame,
    df_phase_firing: pd.DataFrame,
    df_raster_points: pd.DataFrame,
    df_population_rate: pd.DataFrame,
    representative_meta: Mapping[str, object],
    representative_row: pd.Series,
    dt_ms: float,
) -> plt.Figure:
    apply_paper_style()
    fig = plt.figure(figsize=(12.0, 4.8))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.18], wspace=0.28)
    ax_memory = fig.add_subplot(outer[0, 0])
    raster_axes, ax_rate = plot_representative_activity_panel(
        fig=fig,
        subplot_spec=outer[0, 1],
        df_raster_points=df_raster_points,
        df_population_rate=df_population_rate,
        representative_meta=representative_meta,
        dt_ms=dt_ms,
    )

    plot_memory_effect_vs_delay(ax_memory, df_memory_effect)

    ax_memory.text(-0.15, 1.05, "A", transform=ax_memory.transAxes, fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold")
    raster_axes[0].text(-0.15, 1.05, "B", transform=raster_axes[0].transAxes, fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.92, wspace=0.28)
    return fig


def validate_output_tables(
    df_trial_level: pd.DataFrame,
    df_memory_effect: pd.DataFrame,
    df_phase_firing: pd.DataFrame,
    df_representative: pd.DataFrame,
    df_raster_points: pd.DataFrame,
    df_population_rate: pd.DataFrame,
) -> None:
    validate_required_columns(
        df_trial_level,
        [
            "trial_id",
            "stsp_mode",
            "sample_label",
            "probe_label",
            "delay_ms",
            "pred_label",
            "is_correct",
            "is_silent",
            "first_fire_t_probe",
        ],
    )
    validate_required_columns(
        df_memory_effect,
        [
            "delay_ms",
            "stsp_mode",
            "memory_effect_mean",
            "memory_effect_ci95_lower",
            "memory_effect_ci95_upper",
            "fit_y",
            "fit_tau",
        ],
    )
    validate_required_columns(
        df_phase_firing,
        [
            "layer",
            "stsp_mode",
            "sample_rate_mean",
            "sample_rate_ci95_lower",
            "sample_rate_ci95_upper",
            "delay_rate_mean",
            "delay_rate_ci95_lower",
            "delay_rate_ci95_upper",
            "probe_rate_mean",
            "probe_rate_ci95_lower",
            "probe_rate_ci95_upper",
        ],
    )
    validate_required_columns(
        df_representative,
        [
            "trial_id",
            "stsp_mode",
            "sample_label",
            "probe_label",
            "delay_ms",
            "first_fire_t_probe",
            "selection_reason",
        ],
    )
    validate_required_columns(
        df_raster_points,
        ["trial_id", "stsp_mode", "delay_ms", "layer", "t_step", "neuron_index", "phase"],
    )
    validate_required_columns(
        df_population_rate,
        [
            "trial_id",
            "stsp_mode",
            "delay_ms",
            "t_step",
            "time_ms",
            "phase",
            "pooled_spike_count",
            "pooled_rate_spikes_per_neuron_step",
            "total_neurons",
        ],
    )

def build_metrics_summary(
    df_memory_effect: pd.DataFrame,
    df_phase_firing: pd.DataFrame,
    df_representative: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in df_memory_effect.iterrows():
        group = f"{str(row['stsp_mode'])}|delay_{format_delay_ms(float(row['delay_ms']))}ms"
        for metric in [
            "memory_effect_mean",
            "memory_effect_ci95_lower",
            "memory_effect_ci95_upper",
            "fit_y",
            "fit_tau",
        ]:
            rows.append(
                {
                    "section": "memory_effect_vs_delay",
                    "group": group,
                    "metric": metric,
                    "value": float(row[metric]),
                }
            )
    for _, row in df_phase_firing.iterrows():
        group = f"{str(row['layer'])}|{str(row['stsp_mode'])}"
        for metric in [
            "sample_rate_mean",
            "sample_rate_ci95_lower",
            "sample_rate_ci95_upper",
            "delay_rate_mean",
            "delay_rate_ci95_lower",
            "delay_rate_ci95_upper",
            "probe_rate_mean",
            "probe_rate_ci95_lower",
            "probe_rate_ci95_upper",
        ]:
            rows.append(
                {
                    "section": "phase_firing",
                    "group": group,
                    "metric": metric,
                    "value": float(row[metric]),
                }
            )
    rep_row = df_representative.iloc[0]
    for metric in ["trial_id", "sample_label", "probe_label", "delay_ms", "first_fire_t_probe"]:
        rows.append(
            {
                "section": "representative_trial",
                "group": f"{str(rep_row['stsp_mode'])}|canonical_delay",
                "metric": metric,
                "value": float(rep_row[metric]),
            }
        )
    return pd.DataFrame(rows)


def run_self_checks(
    df_memory_effect: pd.DataFrame,
    df_phase_firing: pd.DataFrame,
    df_representative: pd.DataFrame,
    canonical_delay_ms: float,
) -> None:
    dynamic = df_memory_effect[df_memory_effect["stsp_mode"] == "dynamic"].copy().sort_values("delay_ms")
    if len(dynamic) >= 2:
        short_delay = float(dynamic.iloc[0]["memory_effect_mean"])
        long_delay = float(dynamic.iloc[-1]["memory_effect_mean"])
        if short_delay <= long_delay:
            print(
                "[Warn] Dynamic memory effect does not decay from the first to the last sweep point: "
                f"{short_delay:.3f} vs {long_delay:.3f}"
            )
        else:
            print(f"[Check] Dynamic memory effect decays across the sweep: {short_delay:.3f} -> {long_delay:.3f}")
        tau = float(dynamic["fit_tau"].iloc[0])
        if np.isfinite(tau) and tau > 0:
            print(f"[Check] Dynamic decay fit tau is positive: {tau:.2f} ms")
        else:
            print("[Warn] Dynamic decay fit tau is missing or non-positive")

    for _, row in df_phase_firing.iterrows():
        layer = str(row["layer"])
        stsp_mode = str(row["stsp_mode"])
        delay_rate = float(row["delay_rate_mean"])
        sample_rate = float(row["sample_rate_mean"])
        probe_rate = float(row["probe_rate_mean"])
        if not (delay_rate < sample_rate and delay_rate < probe_rate):
            print(
                "[Warn] Delay firing is not below both sample and probe for "
                f"{layer}/{stsp_mode}: sample={sample_rate:.6f}, delay={delay_rate:.6f}, probe={probe_rate:.6f}"
            )

    rep = df_representative.iloc[0]
    if rep["stsp_mode"] != "dynamic":
        raise ValueError("Representative trial is not dynamic")
    if not math.isclose(float(rep["delay_ms"]), float(canonical_delay_ms), abs_tol=1e-9):
        raise ValueError("Representative trial is not from canonical delay")
    print(
        "[Check] Representative trial is reproducible and valid: "
        f"trial_id={int(rep['trial_id'])}, first_fire_t_probe={int(rep['first_fire_t_probe'])}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Figure 2: silent-memory-effect summary figure.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--save-dir", type=str, default="results/fig2_silent_memory_effect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-boot", type=int, default=1000)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--probe-ms", type=float, default=60.0)
    parser.add_argument("--canonical-delay-ms", type=float, default=400.0)
    parser.add_argument(
        "--delay-ms-list",
        type=str,
        default=",".join(str(v) for v in DEFAULT_DELAY_SWEEP_MS),
    )
    parser.add_argument("--no-phase-reset", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_boot <= 0:
        raise ValueError("--num-boot must be positive")

    delay_values_ms = parse_delay_list(args.delay_ms_list)
    seed_everything(args.seed)
    apply_paper_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=args.sample_ms,
        delay_ms=args.canonical_delay_ms,
        probe_ms=args.probe_ms,
        phase_reset=(not args.no_phase_reset),
    )
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[Init] Device: {device}")
    print(f"[Init] Save dir: {args.save_dir}")
    print(f"[Init] Delay sweep (ms): {', '.join(format_delay_ms(v) for v in delay_values_ms)}")
    print(f"[Init] Canonical delay (ms): {format_delay_ms(args.canonical_delay_ms)}")

    net, encoder = load_model_and_encoder(args.model_path, device, spec)
    _, _, test_loader = build_mnist_skeleton_loader(batch_size=1)
    dataset = test_loader.dataset
    class_index = build_class_index(dataset, num_classes=args.num_classes)
    df_specs = generate_balanced_dms_trial_specs(
        class_index=class_index,
        num_trials=args.trials,
        num_classes=args.num_classes,
        rng=random.Random(args.seed),
    )
    validate_trial_specs(df_specs, num_classes=args.num_classes)

    df_trial_level, df_phase_rates = run_delay_sweep_memory_effect(
        net=net,
        encoder=encoder,
        dataset=dataset,
        device=device,
        df_specs=df_specs,
        delay_values_ms=delay_values_ms,
        canonical_delay_ms=args.canonical_delay_ms,
        batch_size=args.batch_size,
        spec=spec,
    )
    df_memory_effect = summarize_memory_effect_vs_delay(
        df_trial_level=df_trial_level,
        delay_values_ms=delay_values_ms,
        num_classes=args.num_classes,
        n_boot=args.num_boot,
        seed=args.seed,
    )
    df_phase_firing = summarize_phase_firing(df_phase_rates=df_phase_rates, n_boot=args.num_boot, seed=args.seed)
    df_representative, df_raster_points, df_population_rate, representative_meta = build_representative_raster_data(
        net=net,
        encoder=encoder,
        dataset=dataset,
        device=device,
        spec=spec,
        df_trial_level=df_trial_level,
        df_specs=df_specs,
        canonical_delay_ms=args.canonical_delay_ms,
        stsp_mode="dynamic",
    )

    validate_output_tables(
        df_trial_level=df_trial_level,
        df_memory_effect=df_memory_effect,
        df_phase_firing=df_phase_firing,
        df_representative=df_representative,
        df_raster_points=df_raster_points,
        df_population_rate=df_population_rate,
    )
    df_metrics_summary = build_metrics_summary(
        df_memory_effect=df_memory_effect,
        df_phase_firing=df_phase_firing,
        df_representative=df_representative,
    )

    trial_csv = save_tidy_csv(
        df_trial_level,
        os.path.join(args.save_dir, "trial_level.csv"),
        sort_by=["delay_ms", "stsp_mode", "trial_id"],
    )
    metrics_summary_csv = save_tidy_csv(
        df_metrics_summary,
        os.path.join(args.save_dir, "metrics_summary.csv"),
        sort_by=["section", "group", "metric"],
    )
    memory_csv = save_tidy_csv(
        df_memory_effect,
        os.path.join(args.save_dir, "metrics_memory_effect_vs_delay.csv"),
        sort_by=["stsp_mode", "delay_ms"],
    )
    phase_csv = save_tidy_csv(
        df_phase_firing,
        os.path.join(args.save_dir, "metrics_phase_firing.csv"),
        sort_by=["layer", "stsp_mode"],
    )
    rep_csv = save_tidy_csv(
        df_representative,
        os.path.join(args.save_dir, "metrics_representative_trial.csv"),
        sort_by=["stsp_mode", "delay_ms", "trial_id"],
    )
    raster_csv = save_tidy_csv(
        df_raster_points,
        os.path.join(args.save_dir, "representative_raster_points.csv"),
        sort_by=["layer", "t_step", "neuron_index"],
    )
    population_rate_csv = save_tidy_csv(
        df_population_rate,
        os.path.join(args.save_dir, "representative_population_rate.csv"),
        sort_by=["t_step"],
    )

    rep_row = df_representative.iloc[0]
    fig = assemble_figure_main(
        df_memory_effect=df_memory_effect,
        df_phase_firing=df_phase_firing,
        df_raster_points=df_raster_points,
        df_population_rate=df_population_rate,
        representative_meta=representative_meta,
        representative_row=rep_row,
        dt_ms=float(spec.dt / ms),
    )
    figure_paths = save_figure_all_formats(fig, os.path.join(args.save_dir, "figure_main"))
    plt.close(fig)

    run_config_path = save_run_config(
        {
            "model_path": args.model_path,
            "device": str(device),
            "seed": int(args.seed),
            "num_classes": int(args.num_classes),
            "trials": int(args.trials),
            "batch_size": int(args.batch_size),
            "num_boot": int(args.num_boot),
            "timing_ms": {
                "sample": float(args.sample_ms),
                "canonical_delay": float(args.canonical_delay_ms),
                "probe": float(args.probe_ms),
                "delay_sweep": [float(v) for v in delay_values_ms],
            },
            "phase_reset": bool(spec.phase_reset),
            "executed_stsp_modes": list(STSP_MODES),
            "executed_layers": list(LAYER_KEYS),
            "outputs": {
                "trial_level": trial_csv,
                "metrics_summary": metrics_summary_csv,
                "metrics_memory_effect_vs_delay": memory_csv,
                "metrics_phase_firing": phase_csv,
                "metrics_representative_trial": rep_csv,
                "representative_raster_points": raster_csv,
                "representative_population_rate": population_rate_csv,
                "figure_main_png": figure_paths["png"],
                "figure_main_pdf": figure_paths["pdf"],
                "figure_main_svg": figure_paths["svg"],
            },
        },
        args.save_dir,
    )

    run_self_checks(
        df_memory_effect=df_memory_effect,
        df_phase_firing=df_phase_firing,
        df_representative=df_representative,
        canonical_delay_ms=args.canonical_delay_ms,
    )

    print("\n=== Figure 2 Silent Memory Effect Summary ===")
    print(f"Saved: {trial_csv}")
    print(f"Saved: {metrics_summary_csv}")
    print(f"Saved: {memory_csv}")
    print(f"Saved: {phase_csv}")
    print(f"Saved: {rep_csv}")
    print(f"Saved: {raster_csv}")
    print(f"Saved: {population_rate_csv}")
    print(f"Saved: {figure_paths['png']}")
    print(f"Saved: {figure_paths['pdf']}")
    print(f"Saved: {figure_paths['svg']}")
    print(f"Saved: {run_config_path}")


if __name__ == "__main__":
    main()
