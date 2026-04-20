from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.plotting.common.external_input_runtime import (
    DEFAULT_TAU_U_MS,
    DistractorExperimentSpec,
    NO_PING_LABEL,
    PingExperimentSpec,
    build_distractor_class_index,
    build_ping_class_index,
    calibrate_ping_per_example,
    compute_delta_summary,
    format_ping_target_label,
    generate_balanced_trial_specs,
    generate_distractor_trial_specs,
    load_distractor_model_and_encoder,
    load_ping_model_and_encoder,
    override_tau_u_ms,
    parse_float_list,
    parse_seed_list,
    run_distractor_experiment,
    run_distractor_interface_check,
    run_seed_experiment,
    seed_everything_distractor,
    seed_everything_ping,
    validate_distractor_pairing,
    validate_distractor_trial_specs,
    validate_ping_trial_specs,
)
from figure_utils_common import (
    COLOR_DISTRACTOR,
    COLOR_DYNAMIC,
    COLOR_NOISE,
    COLOR_PING,
    COLOR_SAMPLE_ALIGNED,
    COLOR_STATIC,
    PUBLICATION_ANNOTATION_FONT_SIZE,
    PUBLICATION_ERRORBAR_CAPSIZE,
    PUBLICATION_LINE_WIDTH,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
    validate_required_columns,
)
from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from paper_plot_style import DEFAULT_SUBPLOT_ADJUST, PANEL_LABEL_FONT_SIZE, apply_paper_style
from src.platform.legacy_adapters.units import ms

NEUTRAL_GRAY = "#7F7F7F"


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def _bootstrap_mean_ci(values: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boot = np.empty(max(1, int(n_boot)), dtype=np.float64)
    n = arr.size
    for idx in range(len(boot)):
        sample_idx = rng.integers(0, n, size=n)
        boot[idx] = float(arr[sample_idx].mean())
    return float(arr.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _bootstrap_p_mean_gt(values: Sequence[float], null_value: float, n_boot: int, seed: int) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")

    centered = arr - float(null_value)
    rng = np.random.default_rng(seed)
    boot = np.empty(max(1, int(n_boot)), dtype=np.float64)
    n = centered.size
    for idx in range(len(boot)):
        sample_idx = rng.integers(0, n, size=n)
        boot[idx] = float(centered[sample_idx].mean())
    return float((np.sum(boot <= 0.0) + 1.0) / (len(boot) + 1.0))


def _metric_summary_dict(
    metric_name: str,
    values: Sequence[float],
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    mean, ci_lo, ci_hi = _bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
    return {
        metric_name: mean,
        f"{metric_name}_ci95_lower": ci_lo,
        f"{metric_name}_ci95_upper": ci_hi,
    }


def _compute_biases(
    df_subset: pd.DataFrame,
    num_classes: int,
    has_distractor: bool,
) -> Tuple[float, float]:
    errors = df_subset[df_subset["prediction_probe"] != df_subset["probe_label"]].copy()
    if len(errors) == 0:
        return 0.0, 0.0

    pred = errors["prediction_probe"].to_numpy(dtype=np.int64)
    sample = errors["sample_label"].to_numpy(dtype=np.int64)
    probe = errors["probe_label"].to_numpy(dtype=np.int64)
    valid = (pred >= 0) & (pred < num_classes)
    sample_bias = float(np.mean(pred == sample))

    if has_distractor:
        distractor = errors["distractor_label"].to_numpy(dtype=np.int64)
        noise_mask = valid & (pred != sample) & (pred != probe) & (pred != distractor)
        denom = len(errors) * max(1, num_classes - 3)
    else:
        noise_mask = valid & (pred != sample) & (pred != probe)
        denom = len(errors) * max(1, num_classes - 2)
    noise_bias = float(noise_mask.sum() / denom) if denom > 0 else 0.0
    return sample_bias, noise_bias


def _rename_stsp_mode(mode: str) -> str:
    return "static" if mode == "static_frozen" else str(mode)


def run_dual_task_distractor_branch(
    model_path: str,
    dataset_root: str,
    device: torch.device,
    seeds: Sequence[int],
    num_trials: int,
    batch_size: int,
    num_classes: int,
    spec: DistractorExperimentSpec,
    run_interface_check_once: bool = True,
) -> pd.DataFrame:
    net, encoder = load_distractor_model_and_encoder(model_path, device, spec)
    if run_interface_check_once:
        run_distractor_interface_check(net, device)

    _, _, test_loader = build_mnist_skeleton_loader(
        root=dataset_root,
        batch_size=64,
        input_size=28,
        num_workers=0,
    )
    dataset = test_loader.dataset
    class_index = build_distractor_class_index(dataset, num_classes=num_classes)

    trial_frames: List[pd.DataFrame] = []
    for seed in seeds:
        seed_everything_distractor(int(seed))
        rng = random.Random(int(seed))
        df_specs = generate_distractor_trial_specs(
            class_index=class_index,
            num_trials=num_trials,
            num_classes=num_classes,
            rng=rng,
        )
        validate_distractor_trial_specs(df_specs, num_classes=num_classes)
        df_trials_seed = run_distractor_experiment(
            net=net,
            encoder=encoder,
            dataset=dataset,
            df_specs=df_specs,
            spec=spec,
            batch_size=batch_size,
            device=device,
        )
        validate_distractor_pairing(df_trials_seed)
        df_trials_seed = df_trials_seed[
            (df_trials_seed["paradigm"] == "distracted")
            | ((df_trials_seed["paradigm"] == "clean") & (df_trials_seed["stsp_mode"] == "dynamic"))
        ].copy()
        df_trials_seed["seed"] = int(seed)
        df_trials_seed["stsp_mode"] = df_trials_seed["stsp_mode"].map(_rename_stsp_mode)
        df_trials_seed["pred_label"] = df_trials_seed["prediction_probe"].astype(np.int64)
        df_trials_seed["pred_distractor_label"] = df_trials_seed["prediction_distractor"].astype(np.int64)
        df_trials_seed["is_correct"] = df_trials_seed["is_correct_probe"].astype(np.int64)
        trial_frames.append(df_trials_seed)

    df_trials = pd.concat(trial_frames, ignore_index=True)
    return df_trials.sort_values(["seed", "trial_id", "paradigm", "stsp_mode"], kind="stable").reset_index(drop=True)

def summarize_distractor_retention(
    df_trials: pd.DataFrame,
    num_classes: int,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = [
        "seed",
        "trial_id",
        "paradigm",
        "stsp_mode",
        "sample_label",
        "distractor_label",
        "probe_label",
        "prediction_distractor",
        "prediction_probe",
        "is_correct_distractor",
        "is_silent_distractor",
    ]
    validate_required_columns(df_trials, required_cols)

    seed_rows: List[Dict[str, float]] = []
    for seed_id, df_seed in df_trials.groupby("seed", sort=True):
        sub_clean = df_seed[(df_seed["paradigm"] == "clean") & (df_seed["stsp_mode"] == "dynamic")].copy()
        sub_dyn = df_seed[(df_seed["paradigm"] == "distracted") & (df_seed["stsp_mode"] == "dynamic")].copy()
        sub_stat = df_seed[(df_seed["paradigm"] == "distracted") & (df_seed["stsp_mode"] == "static")].copy()
        if len(sub_clean) == 0 or len(sub_dyn) == 0 or len(sub_stat) == 0:
            raise ValueError(f"Missing distractor subsets for seed={seed_id}")

        clean_sample_bias, _ = _compute_biases(sub_clean, num_classes=num_classes, has_distractor=False)
        distracted_sample_bias, distracted_noise_bias = _compute_biases(
            sub_dyn,
            num_classes=num_classes,
            has_distractor=True,
        )
        retention_pct = (
            100.0 * distracted_sample_bias / clean_sample_bias if clean_sample_bias > 0.0 else float("nan")
        )
        seed_rows.append(
            {
                "seed": int(seed_id),
                "acc_distractor_dynamic": 100.0 * float(sub_dyn["is_correct_distractor"].mean()),
                "acc_distractor_static": 100.0 * float(sub_stat["is_correct_distractor"].mean()),
                "silent_rate_distractor_dynamic": 100.0 * float(sub_dyn["is_silent_distractor"].mean()),
                "silent_rate_distractor_static": 100.0 * float(sub_stat["is_silent_distractor"].mean()),
                "sample_bias_clean": float(clean_sample_bias),
                "sample_bias_distracted": float(distracted_sample_bias),
                "noise_bias_distracted": float(distracted_noise_bias),
                "sample_bias_retention_pct": float(retention_pct),
                "sample_minus_noise_distracted": float(distracted_sample_bias - distracted_noise_bias),
            }
        )

    df_seed_metrics = pd.DataFrame(seed_rows).sort_values("seed").reset_index(drop=True)
    chance_acc = 100.0 / float(num_classes)
    summary: Dict[str, float] = {
        "chance_acc_distractor": chance_acc,
        "n_seeds": int(df_seed_metrics["seed"].nunique()),
    }
    for idx, metric_name in enumerate(
        [
            "acc_distractor_dynamic",
            "acc_distractor_static",
            "silent_rate_distractor_dynamic",
            "silent_rate_distractor_static",
            "sample_bias_clean",
            "sample_bias_distracted",
            "noise_bias_distracted",
            "sample_bias_retention_pct",
        ],
        start=1,
    ):
        summary.update(
            _metric_summary_dict(
                metric_name,
                df_seed_metrics[metric_name].to_numpy(dtype=np.float64),
                n_boot=n_boot,
                seed=seed + idx,
            )
        )

    summary["p_one_sided_sample_gt_noise_distracted"] = _bootstrap_p_mean_gt(
        df_seed_metrics["sample_minus_noise_distracted"].to_numpy(dtype=np.float64),
        null_value=0.0,
        n_boot=n_boot,
        seed=seed + 101,
    )
    summary["p_one_sided_acc_distractor_dynamic_gt_chance"] = _bootstrap_p_mean_gt(
        df_seed_metrics["acc_distractor_dynamic"].to_numpy(dtype=np.float64),
        null_value=chance_acc,
        n_boot=n_boot,
        seed=seed + 102,
    )
    summary["p_one_sided_acc_distractor_static_gt_chance"] = _bootstrap_p_mean_gt(
        df_seed_metrics["acc_distractor_static"].to_numpy(dtype=np.float64),
        null_value=chance_acc,
        n_boot=n_boot,
        seed=seed + 103,
    )
    return pd.DataFrame([summary]), df_seed_metrics


def run_ping_branch(
    model_path: str,
    dataset_root: str,
    device: torch.device,
    seeds: Sequence[int],
    num_trials: int,
    batch_size: int,
    num_classes: int,
    spec: PingExperimentSpec,
    ping_drive_candidates: Sequence[float],
    ping_target_fracs: Sequence[float],
    decode_splits: int,
    tau_u_ms: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    net, encoder = load_ping_model_and_encoder(model_path, device, spec)
    override_tau_u_ms(net, tau_u_ms=tau_u_ms, dt=spec.dt)

    _, _, test_loader = build_mnist_skeleton_loader(
        root=dataset_root,
        batch_size=64,
        input_size=28,
        num_workers=0,
    )
    dataset = test_loader.dataset
    class_index = build_ping_class_index(dataset, num_classes=num_classes)

    trial_frames: List[pd.DataFrame] = []
    metrics_frames: List[pd.DataFrame] = []
    for seed in seeds:
        seed_everything_ping(int(seed))
        rng = random.Random(int(seed))
        df_specs = generate_balanced_trial_specs(
            class_index=class_index,
            num_trials=num_trials,
            num_classes=num_classes,
            rng=rng,
        )
        validate_ping_trial_specs(df_specs, num_classes=num_classes)

        df_calibration_seed, _, ping_lookup = calibrate_ping_per_example(
            net=net,
            encoder=encoder,
            dataset=dataset,
            df_specs=df_specs,
            spec=spec,
            ping_amp_candidates=ping_drive_candidates,
            ping_target_fracs=ping_target_fracs,
            batch_size=batch_size,
            device=device,
        )
        if len(df_calibration_seed) == 0:
            raise RuntimeError(f"Ping calibration produced no rows for seed={seed}")

        override_tau_u_ms(net, tau_u_ms=tau_u_ms, dt=spec.dt)
        df_trials_seed, df_metrics_seed, _, _, _ = run_seed_experiment(
            net=net,
            encoder=encoder,
            dataset=dataset,
            df_specs=df_specs,
            spec=spec,
            ping_lookup=ping_lookup,
            ping_target_fracs=ping_target_fracs,
            batch_size=batch_size,
            decode_splits=decode_splits,
            seed=int(seed),
            device=device,
        )
        df_trials_seed = df_trials_seed.copy()
        df_trials_seed["activation_target"] = df_trials_seed["ping_target_frac"].astype(np.float64)
        df_trials_seed["achieved_activation_frac"] = df_trials_seed["l1_ping_activation_fraction"].astype(np.float64)
        df_trials_seed["pred_label"] = df_trials_seed["prediction_probe"].astype(np.int64)
        df_trials_seed["is_correct"] = df_trials_seed["is_correct_probe"].astype(np.int64)
        df_trials_seed["stsp_mode"] = "dynamic"
        df_trials_seed["branch"] = "ping"
        trial_frames.append(df_trials_seed)
        metrics_frames.append(df_metrics_seed.copy())

    df_trials = pd.concat(trial_frames, ignore_index=True)
    df_metrics_seed = pd.concat(metrics_frames, ignore_index=True)
    df_trials = df_trials.sort_values(["seed", "trial_id", "ping_target_frac"], kind="stable").reset_index(drop=True)
    df_metrics_seed = df_metrics_seed.sort_values(["seed", "ping_target_frac"], kind="stable").reset_index(drop=True)
    return df_trials, df_metrics_seed


def summarize_ping_decode_vs_activation(
    df_metrics_seed: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    required_cols = [
        "seed",
        "ping_target_label",
        "ping_target_frac",
        "decode_ping_acc",
        "achieved_activation_frac",
    ]
    validate_required_columns(df_metrics_seed, required_cols)

    base = (
        df_metrics_seed[df_metrics_seed["ping_target_label"] == NO_PING_LABEL][["seed", "decode_ping_acc"]]
        .rename(columns={"decode_ping_acc": "decode_ping_acc_no_ping"})
        .reset_index(drop=True)
    )
    if len(base) == 0:
        raise ValueError("Missing no-ping rows for ping summary")

    merged = df_metrics_seed.merge(base, on="seed", how="left")
    merged["activation_target"] = merged["ping_target_frac"].astype(np.float64)
    merged["delta_decode_ping_acc"] = (
        merged["decode_ping_acc"].astype(np.float64) - merged["decode_ping_acc_no_ping"].astype(np.float64)
    )

    rows: List[Dict[str, float]] = []
    for row_idx, ((target_label, activation_target), sub) in enumerate(
        merged.groupby(["ping_target_label", "activation_target"], sort=True),
        start=1,
    ):
        delta_values = sub["delta_decode_ping_acc"].to_numpy(dtype=np.float64)
        act_values = sub["achieved_activation_frac"].to_numpy(dtype=np.float64)
        delta_mean, delta_lo, delta_hi = _bootstrap_mean_ci(delta_values, n_boot=n_boot, seed=seed + row_idx)
        act_mean, act_lo, act_hi = _bootstrap_mean_ci(act_values, n_boot=n_boot, seed=seed + 500 + row_idx)
        rows.append(
            {
                "ping_target_label": str(target_label),
                "activation_target": float(activation_target),
                "achieved_activation_frac": float(act_mean),
                "achieved_activation_frac_ci95_lower": float(act_lo),
                "achieved_activation_frac_ci95_upper": float(act_hi),
                "delta_decode_ping_acc_mean": float(delta_mean),
                "delta_decode_ping_acc_ci95_lower": float(delta_lo),
                "delta_decode_ping_acc_ci95_upper": float(delta_hi),
                "n_seeds": int(sub["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["achieved_activation_frac", "activation_target"], kind="stable").reset_index(
        drop=True
    )


def summarize_ping_selectivity(
    df_trials: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    required_cols = [
        "seed",
        "ping_target_label",
        "activation_target",
        "achieved_activation_frac",
        "sample_aligned_selectivity",
    ]
    validate_required_columns(df_trials, required_cols)

    seed_level = (
        df_trials.groupby(["seed", "ping_target_label", "activation_target"], as_index=False)
        .agg(
            achieved_activation_frac=("achieved_activation_frac", "mean"),
            sample_aligned_selectivity=("sample_aligned_selectivity", "mean"),
        )
        .sort_values(["seed", "activation_target"], kind="stable")
        .reset_index(drop=True)
    )

    rows: List[Dict[str, float]] = []
    for row_idx, ((target_label, activation_target), sub) in enumerate(
        seed_level.groupby(["ping_target_label", "activation_target"], sort=True),
        start=1,
    ):
        sel_values = sub["sample_aligned_selectivity"].to_numpy(dtype=np.float64)
        act_values = sub["achieved_activation_frac"].to_numpy(dtype=np.float64)
        sel_mean, sel_lo, sel_hi = _bootstrap_mean_ci(sel_values, n_boot=n_boot, seed=seed + row_idx)
        act_mean, act_lo, act_hi = _bootstrap_mean_ci(act_values, n_boot=n_boot, seed=seed + 500 + row_idx)
        rows.append(
            {
                "ping_target_label": str(target_label),
                "activation_target": float(activation_target),
                "achieved_activation_frac": float(act_mean),
                "achieved_activation_frac_ci95_lower": float(act_lo),
                "achieved_activation_frac_ci95_upper": float(act_hi),
                "sample_aligned_selectivity_mean": float(sel_mean),
                "sample_aligned_selectivity_ci95_lower": float(sel_lo),
                "sample_aligned_selectivity_ci95_upper": float(sel_hi),
                "n_seeds": int(sub["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["achieved_activation_frac", "activation_target"], kind="stable").reset_index(
        drop=True
    )

def plot_distractor_branch_main(
    axes: Sequence[plt.Axes],
    metrics_distractor_summary: pd.DataFrame,
    num_classes: int,
) -> None:
    if len(axes) != 2:
        raise ValueError("plot_distractor_branch_main expects exactly two axes")
    row = metrics_distractor_summary.iloc[0]
    chance_acc = float(row.get("chance_acc_distractor", 100.0 / num_classes))

    ax_acc, ax_ret = axes
    bar_labels = ["Dynamic", "Static"]
    acc_values = np.array(
        [float(row["acc_distractor_dynamic"]), float(row["acc_distractor_static"])],
        dtype=np.float64,
    )
    acc_lower = np.array(
        [
            float(row["acc_distractor_dynamic_ci95_lower"]),
            float(row["acc_distractor_static_ci95_lower"]),
        ],
        dtype=np.float64,
    )
    acc_upper = np.array(
        [
            float(row["acc_distractor_dynamic_ci95_upper"]),
            float(row["acc_distractor_static_ci95_upper"]),
        ],
        dtype=np.float64,
    )
    yerr = np.vstack([acc_values - acc_lower, acc_upper - acc_values])
    bars = ax_acc.bar(
        np.arange(2),
        acc_values,
        yerr=yerr,
        capsize=PUBLICATION_ERRORBAR_CAPSIZE,
        color=[COLOR_DYNAMIC, NEUTRAL_GRAY],
        edgecolor="#222222",
        width=0.62,
    )
    ax_acc.axhline(chance_acc, color=COLOR_NOISE, linestyle="--", linewidth=1.2)
    ax_acc.set_xticks(np.arange(2), bar_labels)
    ax_acc.set_ylabel("Distractor decode (%)")
    ax_acc.set_ylim(0.0, max(100.0, float(acc_upper.max()) + 12.0))
    chance_y = chance_acc + 1.5
    ax_acc.text(
        0.02,
        chance_y,
        "Chance",
        transform=ax_acc.get_yaxis_transform(),
        color=COLOR_NOISE,
        fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
        va="bottom",
    )
    for idx, bar in enumerate(bars):
        acc_val = acc_values[idx]
        silent_key = "silent_rate_distractor_dynamic" if idx == 0 else "silent_rate_distractor_static"
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2.0,
            acc_val + 2.0,
            f"{acc_val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
        )
    p_dyn = float(row["p_one_sided_acc_distractor_dynamic_gt_chance"])
    p_stat = float(row["p_one_sided_acc_distractor_static_gt_chance"])
    for idx, p_value in enumerate([p_dyn, p_stat]):
        stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        if stars:
            ax_acc.text(idx, acc_upper[idx] + 4.0, stars, ha="center", va="bottom", fontsize=PUBLICATION_ANNOTATION_FONT_SIZE)

    retention_values = 100.0 * np.array(
        [
            float(row["sample_bias_clean"]),
            float(row["sample_bias_distracted"]),
            float(row["noise_bias_distracted"]),
        ],
        dtype=np.float64,
    )
    retention_lower = 100.0 * np.array(
        [
            float(row["sample_bias_clean_ci95_lower"]),
            float(row["sample_bias_distracted_ci95_lower"]),
            float(row["noise_bias_distracted_ci95_lower"]),
        ],
        dtype=np.float64,
    )
    retention_upper = 100.0 * np.array(
        [
            float(row["sample_bias_clean_ci95_upper"]),
            float(row["sample_bias_distracted_ci95_upper"]),
            float(row["noise_bias_distracted_ci95_upper"]),
        ],
        dtype=np.float64,
    )
    ret_yerr = np.vstack([retention_values - retention_lower, retention_upper - retention_values])
    ax_ret.bar(
        np.arange(3),
        retention_values,
        yerr=ret_yerr,
        capsize=PUBLICATION_ERRORBAR_CAPSIZE,
        color=[COLOR_STATIC, COLOR_DYNAMIC, NEUTRAL_GRAY],
        edgecolor="#222222",
        width=0.62,
    )
    for idx, value in enumerate(retention_values):
        upper = float(retention_upper[idx])
        ax_ret.text(
            idx,
            upper + 1.4,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
            color="#222222",
        )
        ax_ret.text(
            idx,
            upper + 3.2,
            "**",
            ha="center",
            va="bottom",
            fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
            color="#222222",
        )
    ax_ret.set_xticks(np.arange(3), ["Clean\nsample", "Distracted\nsample", "Distracted\nnoise"])
    ax_ret.set_ylabel("Probe-error bias (%)")
    ax_ret.set_ylim(0.0, max(5.0, float(retention_upper.max()) + 8.0))
def plot_ping_selectivity_overlay(
    ax: plt.Axes,
    metrics_ping_summary: pd.DataFrame,
    metrics_ping_selectivity: pd.DataFrame,
) -> None:
    ping = metrics_ping_summary.sort_values(["achieved_activation_frac", "activation_target"], kind="stable").reset_index(drop=True)
    sel = metrics_ping_selectivity.sort_values(["achieved_activation_frac", "activation_target"], kind="stable").reset_index(drop=True)

    x_ping = ping["achieved_activation_frac"].to_numpy(dtype=np.float64)
    y_ping = 100.0 * ping["delta_decode_ping_acc_mean"].to_numpy(dtype=np.float64)
    y_ping_lo = 100.0 * ping["delta_decode_ping_acc_ci95_lower"].to_numpy(dtype=np.float64)
    y_ping_hi = 100.0 * ping["delta_decode_ping_acc_ci95_upper"].to_numpy(dtype=np.float64)

    x_sel = sel["achieved_activation_frac"].to_numpy(dtype=np.float64)
    y_sel = 100.0 * sel["sample_aligned_selectivity_mean"].to_numpy(dtype=np.float64)
    y_sel_lo = 100.0 * sel["sample_aligned_selectivity_ci95_lower"].to_numpy(dtype=np.float64)
    y_sel_hi = 100.0 * sel["sample_aligned_selectivity_ci95_upper"].to_numpy(dtype=np.float64)

    ping_line = ax.plot(
        x_ping,
        y_ping,
        color=COLOR_DYNAMIC,
        marker="o",
        linewidth=PUBLICATION_LINE_WIDTH,
        label="Delta ping decode",
        zorder=4,
    )[0]
    ax.fill_between(x_ping, y_ping_lo, y_ping_hi, color=COLOR_DYNAMIC, alpha=0.10, linewidth=0, zorder=2)
    ax.axhline(0.0, color="#222222", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Achieved activation fraction")
    ax.set_xlim(float(np.min(x_ping)) - 0.001, 0.069)
    ax.set_xticks([tick for tick in ax.get_xticks() if tick < 0.07])
    ax.set_ylabel("Delta ping decode (%)", color=COLOR_DYNAMIC)
    ax.tick_params(axis="y", colors=COLOR_DYNAMIC)
    ax.spines["left"].set_color("#222222")
    ax.spines["left"].set_linewidth(0.9)

    ax_sel = ax.twinx()
    ax.set_zorder(3)
    ax.patch.set_alpha(0.0)
    ax_sel.set_zorder(2)
    sel_line = ax_sel.plot(
        x_sel,
        y_sel,
        color=COLOR_STATIC,
        marker="s",
        linewidth=PUBLICATION_LINE_WIDTH,
        label="Sample-aligned selectivity",
        zorder=1,
    )[0]
    right_axis_color = "#3F8FC3"
    ax_sel.fill_between(x_sel, y_sel_lo, y_sel_hi, color=COLOR_STATIC, alpha=0.08, linewidth=0, zorder=0)
    ax_sel.set_ylabel("Sample-aligned selectivity", color=right_axis_color)
    ax_sel.tick_params(axis="y", colors=right_axis_color)
    ax_sel.spines["right"].set_visible(True)
    ax_sel.spines["right"].set_color("#222222")
    ax_sel.spines["right"].set_linewidth(0.9)

    ax.legend(
        [ping_line, sel_line],
        [ping_line.get_label(), sel_line.get_label()],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
    )


def plot_ping_decode_probe_coupling(
    ax: plt.Axes,
    df_ping_delta_summary: pd.DataFrame,
) -> None:
    sub = df_ping_delta_summary.sort_values("ping_target_frac").reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError("plot_ping_decode_probe_coupling received an empty summary table")

    x = sub["delta_decode_ping_acc_mean"].to_numpy(dtype=np.float64)
    y = sub["delta_probe_accuracy_mean"].to_numpy(dtype=np.float64)
    ax.plot(
        x,
        y,
        linewidth=PUBLICATION_LINE_WIDTH,
        color="#34495e",
        alpha=0.9,
        zorder=2,
    )
    ax.scatter(
        x,
        y,
        s=70,
        color=COLOR_PING,
        edgecolors="#222222",
        linewidths=0.8,
        zorder=3,
    )
    for _, row in sub.iterrows():
        frac = float(row["ping_target_frac"])
        label = NO_PING_LABEL if frac <= 0.0 else f"{frac:.3f}"
        ax.annotate(
            label,
            (float(row["delta_decode_ping_acc_mean"]), float(row["delta_probe_accuracy_mean"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
        )

    ax.axhline(0.0, color="#222222", linestyle=":", linewidth=1.0)
    ax.axvline(0.0, color="#222222", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Delta ping decode")
    ax.set_ylabel("Delta probe accuracy (%)")
    ax.set_title("Decode-probe coupling under u*x-gated ping")


def build_unified_external_input_summary(
    metrics_distractor_summary: pd.DataFrame,
    metrics_ping_summary: pd.DataFrame,
    metrics_ping_selectivity: pd.DataFrame,
    metrics_ping_delta_summary: pd.DataFrame,
    num_classes: int,
) -> plt.Figure:
    apply_paper_style()
    fig = plt.figure(figsize=(12.0, 8.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], wspace=0.38, hspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_distractor_branch_main(
        axes=[ax_a, ax_b],
        metrics_distractor_summary=metrics_distractor_summary,
        num_classes=num_classes,
    )
    plot_ping_selectivity_overlay(ax_c, metrics_ping_summary=metrics_ping_summary, metrics_ping_selectivity=metrics_ping_selectivity)
    plot_ping_decode_probe_coupling(ax_d, df_ping_delta_summary=metrics_ping_delta_summary)

    for label, axis in [("A", ax_a), ("B", ax_b), ("C", ax_c), ("D", ax_d)]:
        axis.text(
            -0.15,
            1.05,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_FONT_SIZE,
            fontweight="bold",
        )

    fig.subplots_adjust(**DEFAULT_SUBPLOT_ADJUST)
    return fig


def _prepare_distractor_output_table(df_trials: pd.DataFrame) -> pd.DataFrame:
    out = df_trials.copy()
    ordered_cols = [
        "seed",
        "trial_id",
        "paradigm",
        "stsp_mode",
        "sample_label",
        "distractor_label",
        "probe_label",
        "pred_label",
        "pred_distractor_label",
        "is_correct",
        "is_silent_distractor",
        "first_fire_t_probe",
        "first_fire_t_distractor",
        "prediction_probe",
        "prediction_distractor",
        "is_correct_probe",
        "is_correct_distractor",
        "is_silent_probe",
    ]
    validate_required_columns(out, ordered_cols[:12])
    return out[ordered_cols]


def _prepare_ping_output_table(df_trials: pd.DataFrame) -> pd.DataFrame:
    out = df_trials.copy()
    ordered_cols = [
        "seed",
        "trial_id",
        "activation_target",
        "achieved_activation_frac",
        "sample_label",
        "probe_label",
        "pred_label",
        "is_correct",
        "ping_target_label",
        "ping_drive_amp",
        "selection_method",
        "first_fire_t_probe",
        "sample_aligned_selectivity",
        "prediction_probe",
        "is_correct_probe",
    ]
    validate_required_columns(out, ordered_cols[:8])
    return out[ordered_cols]

def _build_unified_trial_level(
    df_distractor_output: pd.DataFrame,
    df_ping_output: pd.DataFrame,
) -> pd.DataFrame:
    distractor = df_distractor_output.copy()
    distractor.insert(0, "branch", "distractor")
    ping = df_ping_output.copy()
    ping.insert(0, "branch", "ping")
    return pd.concat([distractor, ping], ignore_index=True, sort=False)


def build_metrics_summary(
    metrics_distractor_summary: pd.DataFrame,
    metrics_ping_summary: pd.DataFrame,
    metrics_ping_selectivity: pd.DataFrame,
    metrics_ping_delta_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    distractor_row = metrics_distractor_summary.iloc[0]
    for metric_name, metric_value in distractor_row.items():
        rows.append(
            {
                "section": "distractor_branch",
                "group": "aggregate",
                "metric": str(metric_name),
                "value": float(metric_value),   
            }
        )
    for _, row in metrics_ping_summary.iterrows():
        group = str(row["activation_target"])
        for metric in [
            "achieved_activation_frac",
            "delta_decode_ping_acc_mean",
            "delta_decode_ping_acc_ci95_lower",
            "delta_decode_ping_acc_ci95_upper",
        ]:
            rows.append(
                {
                    "section": "ping_decode_vs_activation",
                    "group": group,
                    "metric": metric,
                    "value": float(row[metric]),
                }
            )
    for _, row in metrics_ping_selectivity.iterrows():
        group = str(row["activation_target"])
        for metric in [
            "achieved_activation_frac",
            "sample_aligned_selectivity_mean",
            "sample_aligned_selectivity_ci95_lower",
            "sample_aligned_selectivity_ci95_upper",
        ]:
            rows.append(
                {
                    "section": "ping_selectivity",
                    "group": group,
                    "metric": metric,
                    "value": float(row[metric]),
                }
            )
    for _, row in metrics_ping_delta_summary.iterrows():
        group = str(row["ping_target_frac"])
        for metric in [
            "achieved_activation_frac",
            "delta_decode_ping_acc_mean",
            "delta_decode_ping_acc_sem",
            "delta_probe_accuracy_mean",
            "delta_probe_accuracy_sem",
        ]:
            rows.append(
                {
                    "section": "ping_decode_probe_coupling",
                    "group": group,
                    "metric": metric,
                    "value": float(row[metric]),
                }
            )
    return pd.DataFrame(rows)

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Figure 4 external input interrogation pipeline.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--save-dir", type=str, default="results/fig4_external_input_interrogation")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed-list", type=str, default="42")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-boot", type=int, default=5000)
    parser.add_argument("--skip-interface-check", action="store_true")

    parser.add_argument("--distractor-trials", type=int, default=1000)
    parser.add_argument("--distractor-sample-ms", type=float, default=200.0)
    parser.add_argument("--distractor-delay1-ms", type=float, default=400.0)
    parser.add_argument("--distractor-ms", type=float, default=200.0)
    parser.add_argument("--distractor-delay2-ms", type=float, default=400.0)
    parser.add_argument("--distractor-probe-ms", type=float, default=100.0)
    parser.add_argument("--no-distractor-phase-reset", action="store_true")

    parser.add_argument("--ping-trials", type=int, default=1000)
    parser.add_argument("--ping-sample-ms", type=float, default=200.0)
    parser.add_argument("--ping-delay1-ms", type=float, default=500.0)
    parser.add_argument("--ping-ms", type=float, default=30.0)
    parser.add_argument("--ping-delay2-ms", type=float, default=150.0)
    parser.add_argument("--ping-probe-ms", type=float, default=100.0)
    parser.add_argument("--tau-u-ms", type=float, default=DEFAULT_TAU_U_MS)
    parser.add_argument(
        "--ping-drive-candidates",
        type=str,
        default="0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,4.0,5.0,6.0",
    )
    parser.add_argument(
        "--ping-target-fracs",
        type=str,
        default="0.010,0.020,0.025,0.030,0.035,0.040,0.050,0.060,0.070,0.080,0.090,0.100,0.110,0.120",
    )
    parser.add_argument("--decode-splits", type=int, default=5)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.num_classes < 4:
        raise ValueError("--num-classes must be >= 4")
    if args.distractor_trials <= 0 or args.ping_trials <= 0:
        raise ValueError("trial counts must be positive")
    if args.batch_size <= 0 or args.num_boot <= 0 or args.decode_splits <= 0:
        raise ValueError("batch-size, num-boot, and decode-splits must be positive")

    apply_paper_style()

    seeds = parse_seed_list(args.seed_list)
    ping_target_fracs = sorted(set(parse_float_list(args.ping_target_fracs)))
    ping_drive_candidates = parse_float_list(args.ping_drive_candidates)
    if any(float(value) < 0.0 for value in ping_target_fracs):
        raise ValueError("ping-target-fracs must be non-negative")

    device = _resolve_device(args.device)
    distractor_spec = DistractorExperimentSpec(
        dt=1.0 * ms,
        sample_ms=float(args.distractor_sample_ms),
        delay1_ms=float(args.distractor_delay1_ms),
        distractor_ms=float(args.distractor_ms),
        delay2_ms=float(args.distractor_delay2_ms),
        probe_ms=float(args.distractor_probe_ms),
        phase_reset=(not args.no_distractor_phase_reset),
    )
    ping_spec = PingExperimentSpec(
        dt=1.0 * ms,
        sample_ms=float(args.ping_sample_ms),
        delay_ms=float(args.ping_delay1_ms),
        ping_ms=float(args.ping_ms),
        post_ping_ms=float(args.ping_delay2_ms),
        probe_ms=float(args.ping_probe_ms),
    )
    if min(
        distractor_spec.sample_steps,
        distractor_spec.delay1_steps,
        distractor_spec.distractor_steps,
        distractor_spec.delay2_steps,
        distractor_spec.probe_steps,
        ping_spec.sample_steps,
        ping_spec.delay_steps,
        ping_spec.ping_steps,
        ping_spec.post_ping_steps,
        ping_spec.probe_steps,
    ) <= 0:
        raise ValueError("All experiment phases must have positive duration")

    print(
        f"[Init] device={device} | seeds={seeds} | save_dir={save_dir}\n"
        f"[Init] distractor_trials={args.distractor_trials} | ping_trials={args.ping_trials} | batch_size={args.batch_size}"
    )

    df_distractor_trials = run_dual_task_distractor_branch(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        device=device,
        seeds=seeds,
        num_trials=args.distractor_trials,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        spec=distractor_spec,
        run_interface_check_once=(not args.skip_interface_check),
    )
    metrics_distractor_summary, metrics_distractor_seed = summarize_distractor_retention(
        df_distractor_trials,
        num_classes=args.num_classes,
        n_boot=args.num_boot,
        seed=173,
    )

    df_ping_trials, df_ping_metrics_seed = run_ping_branch(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        device=device,
        seeds=seeds,
        num_trials=args.ping_trials,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        spec=ping_spec,
        ping_drive_candidates=ping_drive_candidates,
        ping_target_fracs=ping_target_fracs,
        decode_splits=args.decode_splits,
        tau_u_ms=float(args.tau_u_ms),
    )
    metrics_ping_summary = summarize_ping_decode_vs_activation(
        df_ping_metrics_seed,
        n_boot=args.num_boot,
        seed=271,
    )
    metrics_ping_delta_summary = compute_delta_summary(df_ping_metrics_seed)
    metrics_ping_selectivity = summarize_ping_selectivity(
        df_ping_trials,
        n_boot=args.num_boot,
        seed=389,
    )

    df_distractor_output = _prepare_distractor_output_table(df_distractor_trials)
    df_ping_output = _prepare_ping_output_table(df_ping_trials)
    df_trial_level = _build_unified_trial_level(
        df_distractor_output=df_distractor_output,
        df_ping_output=df_ping_output,
    )
    df_metrics_summary = build_metrics_summary(
        metrics_distractor_summary=metrics_distractor_summary,
        metrics_ping_summary=metrics_ping_summary,
        metrics_ping_selectivity=metrics_ping_selectivity,
        metrics_ping_delta_summary=metrics_ping_delta_summary,
    )

    validate_required_columns(
        df_distractor_output,
        [
            "trial_id",
            "stsp_mode",
            "sample_label",
            "distractor_label",
            "probe_label",
            "pred_label",
            "pred_distractor_label",
            "is_correct",
            "is_silent_distractor",
            "first_fire_t_probe",
        ],
    )
    validate_required_columns(
        metrics_distractor_summary,
        [
            "acc_distractor_dynamic",
            "acc_distractor_static",
            "silent_rate_distractor_dynamic",
            "silent_rate_distractor_static",
            "sample_bias_clean",
            "sample_bias_distracted",
            "noise_bias_distracted",
            "sample_bias_retention_pct",
            "p_one_sided_sample_gt_noise_distracted",
        ],
    )
    validate_required_columns(
        df_ping_output,
        [
            "trial_id",
            "activation_target",
            "achieved_activation_frac",
            "sample_label",
            "probe_label",
            "pred_label",
            "is_correct",
        ],
    )
    validate_required_columns(
        metrics_ping_summary,
        [
            "achieved_activation_frac",
            "delta_decode_ping_acc_mean",
            "delta_decode_ping_acc_ci95_lower",
            "delta_decode_ping_acc_ci95_upper",
        ],
    )
    validate_required_columns(
        metrics_ping_selectivity,
        [
            "achieved_activation_frac",
            "sample_aligned_selectivity_mean",
            "sample_aligned_selectivity_ci95_lower",
            "sample_aligned_selectivity_ci95_upper",
        ],
    )
    validate_required_columns(
        metrics_ping_delta_summary,
        [
            "ping_target_label",
            "ping_target_frac",
            "achieved_activation_frac",
            "delta_decode_ping_acc_mean",
            "delta_decode_ping_acc_sem",
            "delta_probe_accuracy_mean",
            "delta_probe_accuracy_sem",
        ],
    )
    validate_required_columns(df_trial_level, ["branch", "trial_id"])
    validate_required_columns(df_metrics_summary, ["section", "group", "metric", "value"])

    trial_level_csv = save_tidy_csv(
        df_trial_level,
        save_dir / "trial_level.csv",
        sort_by=["branch", "seed", "trial_id"],
    )
    distractor_csv = save_tidy_csv(
        df_distractor_output,
        save_dir / "trial_level_distractor.csv",
        sort_by=["seed", "trial_id", "paradigm", "stsp_mode"],
    )
    metrics_summary_csv = save_tidy_csv(
        df_metrics_summary,
        save_dir / "metrics_summary.csv",
        sort_by=["section", "group", "metric"],
    )
    distractor_summary_csv = save_tidy_csv(
        metrics_distractor_summary,
        save_dir / "metrics_distractor_summary.csv",
    )
    ping_csv = save_tidy_csv(
        df_ping_output,
        save_dir / "trial_level_ping.csv",
        sort_by=["seed", "trial_id", "activation_target"],
    )
    ping_summary_csv = save_tidy_csv(
        metrics_ping_summary,
        save_dir / "metrics_ping_summary.csv",
        sort_by=["achieved_activation_frac", "activation_target"],
    )
    ping_delta_summary_csv = save_tidy_csv(
        metrics_ping_delta_summary,
        save_dir / "metrics_ping_delta_summary.csv",
        sort_by=["ping_target_frac"],
    )
    ping_selectivity_csv = save_tidy_csv(
        metrics_ping_selectivity,
        save_dir / "metrics_ping_selectivity.csv",
        sort_by=["achieved_activation_frac", "activation_target"],
    )

    fig = build_unified_external_input_summary(
        metrics_distractor_summary=metrics_distractor_summary,
        metrics_ping_summary=metrics_ping_summary,
        metrics_ping_selectivity=metrics_ping_selectivity,
        metrics_ping_delta_summary=metrics_ping_delta_summary,
        num_classes=args.num_classes,
    )
    figure_paths = save_figure_all_formats(fig, save_dir / "figure_main")
    plt.close(fig)

    run_config_path = save_run_config(
        {
            "model_path": args.model_path,
            "dataset_root": args.dataset_root,
            "device": str(device),
            "seed_list": list(seeds),
            "num_classes": int(args.num_classes),
            "batch_size": int(args.batch_size),
            "num_boot": int(args.num_boot),
            "distractor_branch": {
                "num_trials": int(args.distractor_trials),
                "timing_ms": {
                    "sample": float(args.distractor_sample_ms),
                    "delay1": float(args.distractor_delay1_ms),
                    "distractor": float(args.distractor_ms),
                    "delay2": float(args.distractor_delay2_ms),
                    "probe": float(args.distractor_probe_ms),
                },
                "phase_reset": bool(not args.no_distractor_phase_reset),
                "comparison_modes": ["dynamic", "static"],
            },
            "ping_branch": {
                "num_trials": int(args.ping_trials),
                "timing_ms": {
                    "sample": float(args.ping_sample_ms),
                    "delay1": float(args.ping_delay1_ms),
                    "ping": float(args.ping_ms),
                    "delay2": float(args.ping_delay2_ms),
                    "probe": float(args.ping_probe_ms),
                },
                "tau_u_ms": float(args.tau_u_ms),
                "decode_splits": int(args.decode_splits),
                "ping_drive_candidates": [float(value) for value in ping_drive_candidates],
                "ping_target_fracs": [float(value) for value in ping_target_fracs],
                "activation_target_labels": [NO_PING_LABEL]
                + [format_ping_target_label(float(value)) for value in ping_target_fracs],
                "x_axis": "achieved_activation_frac",
                "calibration_scope": "per_example",
                "delta_summary_columns": [
                    "ping_target_label",
                    "ping_target_frac",
                    "achieved_activation_frac",
                    "delta_decode_ping_acc_mean",
                    "delta_decode_ping_acc_sem",
                    "delta_probe_accuracy_mean",
                    "delta_probe_accuracy_sem",
                ],
            },
            "outputs": {
                "trial_level": trial_level_csv,
                "metrics_summary": metrics_summary_csv,
                "trial_level_distractor": distractor_csv,
                "metrics_distractor_summary": distractor_summary_csv,
                "trial_level_ping": ping_csv,
                "metrics_ping_summary": ping_summary_csv,
                "metrics_ping_delta_summary": ping_delta_summary_csv,
                "metrics_ping_selectivity": ping_selectivity_csv,
                "figure_main": figure_paths,
            },
            "figure_main_panels": {
                "A": "Distractor decode",
                "B": "Distractor probe-error bias",
                "C": "Ping decode and sample-aligned selectivity",
                "D": "Delta ping decode vs delta probe accuracy",
            },
            "internal_audit_tables": {
                "metrics_distractor_seed": metrics_distractor_seed.to_dict(orient="records"),
                "metrics_ping_seed_rows": int(len(df_ping_metrics_seed)),
            },
        },
        save_dir,
    )

    print(f"[Done] unified trials     -> {trial_level_csv}")
    print(f"[Done] metrics summary    -> {metrics_summary_csv}")
    print(f"[Done] distractor trials  -> {distractor_csv}")
    print(f"[Done] distractor summary -> {distractor_summary_csv}")
    print(f"[Done] ping trials        -> {ping_csv}")
    print(f"[Done] ping summary       -> {ping_summary_csv}")
    print(f"[Done] ping delta summary -> {ping_delta_summary_csv}")
    print(f"[Done] ping selectivity   -> {ping_selectivity_csv}")
    print(f"[Done] figure main        -> {figure_paths}")
    print(f"[Done] run config         -> {run_config_path}")


if __name__ == "__main__":
    main()
