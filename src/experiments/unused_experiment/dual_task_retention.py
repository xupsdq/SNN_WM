import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from src.experiments.retention.shared.dual_task_retention_api import (
    ExperimentSpec,
    build_class_index,
    generate_trial_specs,
    load_model_and_encoder,
    run_experiment,
    run_interface_check,
    seed_everything,
    validate_pairing,
    validate_trial_specs,
)
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_tidy_csv
from src.plotting.common.theme_tokens import (
    ALPHA_ANNOTATION_BOX,
    ALPHA_BAR,
    FIGSIZE_SINGLE_PANEL_TALL,
    FIGSIZE_TWO_PANEL,
    GRID_ALPHA,
    LINE_WIDTH_REFERENCE,
    RETENTION_ACCURACY_COLORS,
    RETENTION_BIAS_COLORS,
    apply_standard_legend,
)
from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.platform.legacy_adapters.units import ms


def _accuracy(df_trials: pd.DataFrame, paradigm: str, stsp_mode: str, target: str) -> float:
    if target not in {"probe", "distractor"}:
        raise ValueError(f"Unknown target: {target}")

    sub = df_trials[(df_trials["paradigm"] == paradigm) & (df_trials["stsp_mode"] == stsp_mode)]
    if len(sub) == 0:
        return float("nan")

    if target == "probe":
        return 100.0 * float((sub["prediction_probe"] == sub["probe_label"]).mean())

    sub = sub[sub["paradigm"] == "distracted"]
    if len(sub) == 0:
        return float("nan")
    return 100.0 * float((sub["prediction_distractor"] == sub["distractor_label"]).mean())


def compute_accuracy_mri(df_trials: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    acc_base = _accuracy(df_trials, paradigm="clean", stsp_mode="static_frozen", target="probe")
    acc_clean = _accuracy(df_trials, paradigm="clean", stsp_mode="dynamic", target="probe")
    acc_distracted = _accuracy(df_trials, paradigm="distracted", stsp_mode="dynamic", target="probe")

    acc_distractor_dynamic = _accuracy(df_trials, paradigm="distracted", stsp_mode="dynamic", target="distractor")
    acc_distractor_static = _accuracy(df_trials, paradigm="distracted", stsp_mode="static_frozen", target="distractor")

    delta_clean = acc_base - acc_clean
    delta_distracted = acc_base - acc_distracted
    mri = 100.0 * delta_distracted / max(delta_clean, eps)

    return pd.DataFrame(
        [
            {
                "acc_base": acc_base,
                "acc_clean": acc_clean,
                "acc_distracted": acc_distracted,
                "delta_m_clean": delta_clean,
                "delta_m_distracted": delta_distracted,
                "mri_percent": mri,
                "acc_distractor_dynamic": acc_distractor_dynamic,
                "acc_distractor_static": acc_distractor_static,
            }
        ]
    )


def _bootstrap_mean_ci_from_values(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        raise ValueError("bootstrap mean CI received empty values")

    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot, dtype=np.float64)
    n = vals.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(vals[idx].mean())
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _paired_bootstrap_diff_summary(values_a: np.ndarray, values_b: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap arrays must have the same shape")
    if a.size == 0:
        raise ValueError("paired bootstrap received empty arrays")

    obs = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot, dtype=np.float64)
    n = a.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(a[idx].mean() - b[idx].mean())

    p_le_0 = float((np.sum(boot <= 0.0) + 1.0) / (len(boot) + 1.0))
    p_ge_0 = float((np.sum(boot >= 0.0) + 1.0) / (len(boot) + 1.0))
    p_two = float(min(1.0, 2.0 * min(p_le_0, p_ge_0)))
    return {
        "observed_diff": obs,
        "ci95_lower": float(np.percentile(boot, 2.5)),
        "ci95_upper": float(np.percentile(boot, 97.5)),
        "p_one_sided_gt0": p_le_0,
        "p_one_sided_lt0": p_ge_0,
        "p_two_sided_ne0": p_two,
        "n_boot": int(len(boot)),
    }


def compute_distractor_sensory_readout(df_trials: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    dyn = df_trials[(df_trials["paradigm"] == "distracted") & (df_trials["stsp_mode"] == "dynamic")].sort_values("trial_id")
    stat = df_trials[(df_trials["paradigm"] == "distracted") & (df_trials["stsp_mode"] == "static_frozen")].sort_values("trial_id")

    if len(dyn) == 0 or len(stat) == 0:
        raise ValueError("Missing distracted dynamic/static subsets for distractor sensory readout")
    if len(dyn) != len(stat):
        raise ValueError("Dynamic and static distracted subsets must have the same length")
    if not np.array_equal(dyn["trial_id"].to_numpy(), stat["trial_id"].to_numpy()):
        raise ValueError("trial_id mismatch between dynamic and static distracted subsets")

    acc_dyn = dyn["is_correct_distractor"].to_numpy(dtype=np.float64)
    acc_stat = stat["is_correct_distractor"].to_numpy(dtype=np.float64)
    silent_dyn = dyn["is_silent_distractor"].to_numpy(dtype=np.float64)
    silent_stat = stat["is_silent_distractor"].to_numpy(dtype=np.float64)

    acc_dyn_ci = _bootstrap_mean_ci_from_values(acc_dyn, n_boot=n_boot, seed=seed + 11)
    acc_stat_ci = _bootstrap_mean_ci_from_values(acc_stat, n_boot=n_boot, seed=seed + 12)
    silent_dyn_ci = _bootstrap_mean_ci_from_values(silent_dyn, n_boot=n_boot, seed=seed + 13)
    silent_stat_ci = _bootstrap_mean_ci_from_values(silent_stat, n_boot=n_boot, seed=seed + 14)
    acc_gap = _paired_bootstrap_diff_summary(acc_dyn, acc_stat, n_boot=n_boot, seed=seed + 21)
    silent_gap = _paired_bootstrap_diff_summary(silent_dyn, silent_stat, n_boot=n_boot, seed=seed + 22)

    return pd.DataFrame(
        [
            {
                "n_trials_distracted_dynamic": int(len(dyn)),
                "n_trials_distracted_static": int(len(stat)),
                "acc_distractor_dynamic": 100.0 * float(acc_dyn.mean()),
                "acc_distractor_dynamic_ci95_lower": 100.0 * acc_dyn_ci[0],
                "acc_distractor_dynamic_ci95_upper": 100.0 * acc_dyn_ci[1],
                "acc_distractor_static": 100.0 * float(acc_stat.mean()),
                "acc_distractor_static_ci95_lower": 100.0 * acc_stat_ci[0],
                "acc_distractor_static_ci95_upper": 100.0 * acc_stat_ci[1],
                "acc_distractor_gap_dynamic_minus_static": 100.0 * float(acc_gap["observed_diff"]),
                "acc_distractor_gap_ci95_lower": 100.0 * float(acc_gap["ci95_lower"]),
                "acc_distractor_gap_ci95_upper": 100.0 * float(acc_gap["ci95_upper"]),
                "acc_distractor_gap_p_one_sided_gt0": float(acc_gap["p_one_sided_gt0"]),
                "acc_distractor_gap_p_two_sided_ne0": float(acc_gap["p_two_sided_ne0"]),
                "silent_rate_distractor_dynamic": 100.0 * float(silent_dyn.mean()),
                "silent_rate_distractor_dynamic_ci95_lower": 100.0 * silent_dyn_ci[0],
                "silent_rate_distractor_dynamic_ci95_upper": 100.0 * silent_dyn_ci[1],
                "silent_rate_distractor_static": 100.0 * float(silent_stat.mean()),
                "silent_rate_distractor_static_ci95_lower": 100.0 * silent_stat_ci[0],
                "silent_rate_distractor_static_ci95_upper": 100.0 * silent_stat_ci[1],
                "silent_rate_distractor_gap_dynamic_minus_static": 100.0 * float(silent_gap["observed_diff"]),
                "silent_rate_distractor_gap_ci95_lower": 100.0 * float(silent_gap["ci95_lower"]),
                "silent_rate_distractor_gap_ci95_upper": 100.0 * float(silent_gap["ci95_upper"]),
                "silent_rate_distractor_gap_p_one_sided_lt0": float(silent_gap["p_one_sided_lt0"]),
                "silent_rate_distractor_gap_p_two_sided_ne0": float(silent_gap["p_two_sided_ne0"]),
                "n_boot": int(n_boot),
            }
        ]
    )


def compute_bias_components(df_subset: pd.DataFrame, num_classes: int) -> Dict[str, float]:
    n_total = len(df_subset)
    if n_total == 0:
        raise ValueError("Bias computation received empty subset")

    errors = df_subset[df_subset["prediction_probe"] != df_subset["probe_label"]]
    n_error = len(errors)
    if n_error == 0:
        return {
            "n_total": int(n_total),
            "n_error": 0,
            "error_rate": 0.0,
            "bias_sample": 0.0,
            "bias_distractor": 0.0,
            "bias_noise": 0.0,
            "bias_silent": 0.0,
        }

    pred = errors["prediction_probe"].to_numpy()
    sample_lbl = errors["sample_label"].to_numpy()
    distractor_lbl = errors["distractor_label"].to_numpy()
    probe_lbl = errors["probe_label"].to_numpy()

    bias_sample = float(np.mean(pred == sample_lbl))
    bias_distractor = float(np.mean(pred == distractor_lbl))
    bias_silent = float(np.mean(pred == -1))

    remaining_classes = num_classes - 3
    if remaining_classes <= 0:
        raise ValueError("num_classes must be >= 4 for noise-bias definition")

    valid = (pred >= 0) & (pred < num_classes)
    noise_hit = valid & (pred != sample_lbl) & (pred != distractor_lbl) & (pred != probe_lbl)
    bias_noise = float(noise_hit.sum() / float(n_error * remaining_classes))

    return {
        "n_total": int(n_total),
        "n_error": int(n_error),
        "error_rate": 100.0 * float(n_error) / float(n_total),
        "bias_sample": bias_sample,
        "bias_distractor": bias_distractor,
        "bias_noise": bias_noise,
        "bias_silent": bias_silent,
    }


def compute_bias_table(df_trials: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for stsp_mode in ["dynamic", "static_frozen"]:
        sub = df_trials[(df_trials["paradigm"] == "distracted") & (df_trials["stsp_mode"] == stsp_mode)]
        row = compute_bias_components(sub, num_classes=num_classes)
        row["paradigm"] = "distracted"
        row["stsp_mode"] = stsp_mode
        rows.append(row)
    return pd.DataFrame(rows)


def _bias_from_numpy(
    pred_probe: np.ndarray,
    sample_lbl: np.ndarray,
    distractor_lbl: np.ndarray,
    probe_lbl: np.ndarray,
    num_classes: int,
) -> Tuple[float, float]:
    err = pred_probe != probe_lbl
    n_err = int(err.sum())
    if n_err == 0:
        return 0.0, 0.0

    pred_e = pred_probe[err]
    sample_e = sample_lbl[err]
    distractor_e = distractor_lbl[err]
    probe_e = probe_lbl[err]

    bias_sample = float(np.mean(pred_e == sample_e))
    valid = (pred_e >= 0) & (pred_e < num_classes)
    noise_hit = valid & (pred_e != sample_e) & (pred_e != distractor_e) & (pred_e != probe_e)
    bias_noise = float(noise_hit.sum() / float(n_err * (num_classes - 3)))
    return bias_sample, bias_noise


def paired_bootstrap_tests(
    df_trials: pd.DataFrame,
    num_classes: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    dyn = df_trials[(df_trials["paradigm"] == "distracted") & (df_trials["stsp_mode"] == "dynamic")].sort_values("trial_id")
    stat = df_trials[(df_trials["paradigm"] == "distracted") & (df_trials["stsp_mode"] == "static_frozen")].sort_values("trial_id")

    if len(dyn) != len(stat):
        raise ValueError("Dynamic and static distracted subsets must have same length for paired bootstrap")
    if not np.array_equal(dyn["trial_id"].to_numpy(), stat["trial_id"].to_numpy()):
        raise ValueError("trial_id mismatch between dynamic and static subsets")

    pred_dyn = dyn["prediction_probe"].to_numpy()
    pred_stat = stat["prediction_probe"].to_numpy()
    sample_lbl = dyn["sample_label"].to_numpy()
    distractor_lbl = dyn["distractor_label"].to_numpy()
    probe_lbl = dyn["probe_label"].to_numpy()

    obs_bias_sample_dyn, obs_bias_noise_dyn = _bias_from_numpy(
        pred_dyn, sample_lbl, distractor_lbl, probe_lbl, num_classes
    )
    obs_bias_sample_stat, _ = _bias_from_numpy(
        pred_stat, sample_lbl, distractor_lbl, probe_lbl, num_classes
    )

    obs_diff_1 = obs_bias_sample_dyn - obs_bias_noise_dyn
    obs_diff_2 = obs_bias_sample_dyn - obs_bias_sample_stat

    n = len(dyn)
    rng = np.random.default_rng(seed)
    boot_diff_1 = np.zeros(n_boot, dtype=np.float64)
    boot_diff_2 = np.zeros(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)

        b_pred_dyn = pred_dyn[idx]
        b_pred_stat = pred_stat[idx]
        b_sample = sample_lbl[idx]
        b_distractor = distractor_lbl[idx]
        b_probe = probe_lbl[idx]

        b_bias_sample_dyn, b_bias_noise_dyn = _bias_from_numpy(
            b_pred_dyn, b_sample, b_distractor, b_probe, num_classes
        )
        b_bias_sample_stat, _ = _bias_from_numpy(
            b_pred_stat, b_sample, b_distractor, b_probe, num_classes
        )

        boot_diff_1[b] = b_bias_sample_dyn - b_bias_noise_dyn
        boot_diff_2[b] = b_bias_sample_dyn - b_bias_sample_stat

    def summarize(test_name: str, observed: float, samples: np.ndarray) -> Dict[str, float]:
        ci_lower = float(np.percentile(samples, 2.5))
        ci_upper = float(np.percentile(samples, 97.5))
        p_one_sided = float((np.sum(samples <= 0.0) + 1.0) / (len(samples) + 1.0))
        return {
            "test_name": test_name,
            "observed_diff": observed,
            "ci95_lower": ci_lower,
            "ci95_upper": ci_upper,
            "p_one_sided_gt0": p_one_sided,
            "n_boot": int(len(samples)),
        }

    return pd.DataFrame(
        [
            summarize(
                "bias_sample_dynamic_minus_bias_noise_dynamic",
                obs_diff_1,
                boot_diff_1,
            ),
            summarize(
                "bias_sample_dynamic_minus_bias_sample_static",
                obs_diff_2,
                boot_diff_2,
            ),
        ]
    )


def bootstrap_ci_scalar(
    sub: pd.DataFrame,
    metric_fn,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    rec = sub.to_records(index=False)
    n = len(rec)
    if n <= 0:
        raise ValueError("bootstrap_ci_scalar received empty subset")

    rng = np.random.default_rng(seed)
    vals = np.zeros(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b = pd.DataFrame.from_records(rec[idx])
        vals[i] = float(metric_fn(b))

    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def compute_sample_readout_survival(
    df_trials: pd.DataFrame,
    num_classes: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    clean_dyn = df_trials[(df_trials["paradigm"] == "clean") & (df_trials["stsp_mode"] == "dynamic")].copy()
    dist_dyn = df_trials[(df_trials["paradigm"] == "distracted") & (df_trials["stsp_mode"] == "dynamic")].copy()

    if len(clean_dyn) == 0 or len(dist_dyn) == 0:
        raise ValueError("Missing clean/distracted dynamic subset for sample readout survival")

    def _bias_sample(sub: pd.DataFrame) -> float:
        err = sub[sub["prediction_probe"] != sub["probe_label"]]
        if len(err) == 0:
            return 0.0
        pred = err["prediction_probe"].to_numpy()
        sample = err["sample_label"].to_numpy()
        return float(np.mean(pred == sample))

    def _bias_noise(sub: pd.DataFrame, has_distractor: bool) -> float:
        err = sub[sub["prediction_probe"] != sub["probe_label"]]
        if len(err) == 0:
            return 0.0
        pred = err["prediction_probe"].to_numpy()
        sample = err["sample_label"].to_numpy()
        probe = err["probe_label"].to_numpy()
        valid = (pred >= 0) & (pred < num_classes)
        noise_hit = valid & (pred != sample) & (pred != probe)
        if has_distractor:
            distractor = err["distractor_label"].to_numpy()
            noise_hit = noise_hit & (pred != distractor)
            k = num_classes - 3
        else:
            k = num_classes - 2
        return float(noise_hit.sum() / float(len(err) * k))

    clean_sample = _bias_sample(clean_dyn)
    distracted_sample = _bias_sample(dist_dyn)
    distracted_noise = _bias_noise(dist_dyn, has_distractor=True)

    clean_sample_ci = bootstrap_ci_scalar(clean_dyn, _bias_sample, n_boot=n_boot, seed=seed + 11)
    distracted_sample_ci = bootstrap_ci_scalar(dist_dyn, _bias_sample, n_boot=n_boot, seed=seed + 12)
    distracted_noise_ci = bootstrap_ci_scalar(
        dist_dyn,
        lambda x: _bias_noise(x, has_distractor=True),
        n_boot=n_boot,
        seed=seed + 13,
    )

    # One-sided p for distracted(sample-noise) from paired bootstrap.
    tests = paired_bootstrap_tests(df_trials=df_trials, num_classes=num_classes, n_boot=n_boot, seed=seed + 21)
    row_test = tests[tests["test_name"] == "bias_sample_dynamic_minus_bias_noise_dynamic"]
    p_one = float(row_test.iloc[0]["p_one_sided_gt0"]) if len(row_test) == 1 else float("nan")

    sample_drop_abs_pp = (clean_sample - distracted_sample) * 100.0
    sample_drop_rel_pct = (
        (clean_sample - distracted_sample) / clean_sample * 100.0 if clean_sample > 0.0 else float("nan")
    )
    sample_retention_pct = (distracted_sample / clean_sample * 100.0) if clean_sample > 0.0 else float("nan")

    n_err_clean = int((clean_dyn["prediction_probe"] != clean_dyn["probe_label"]).sum())
    n_err_dist = int((dist_dyn["prediction_probe"] != dist_dyn["probe_label"]).sum())

    return pd.DataFrame(
        [
            {
                "clean_sample_bias": clean_sample,
                "clean_sample_bias_ci95_lower": clean_sample_ci[0],
                "clean_sample_bias_ci95_upper": clean_sample_ci[1],
                "distracted_sample_bias": distracted_sample,
                "distracted_sample_bias_ci95_lower": distracted_sample_ci[0],
                "distracted_sample_bias_ci95_upper": distracted_sample_ci[1],
                "distracted_noise_bias": distracted_noise,
                "distracted_noise_bias_ci95_lower": distracted_noise_ci[0],
                "distracted_noise_bias_ci95_upper": distracted_noise_ci[1],
                "sample_bias_drop_abs_pp": sample_drop_abs_pp,
                "sample_bias_drop_rel_pct": sample_drop_rel_pct,
                "sample_bias_retention_pct": sample_retention_pct,
                "p_one_sided_sample_gt_noise_distracted": p_one,
                "n_trials_clean_dynamic": int(len(clean_dyn)),
                "n_trials_distracted_dynamic": int(len(dist_dyn)),
                "n_error_clean_dynamic": n_err_clean,
                "n_error_distracted_dynamic": n_err_dist,
                "n_boot": int(n_boot),
            }
        ]
    )


def plot_accuracy_and_mri(metrics_acc: pd.DataFrame) -> plt.Figure:
    row = metrics_acc.iloc[0]
    labels = ["Acc_base", "Acc_clean", "Acc_distracted"]
    values = [row["acc_base"], row["acc_clean"], row["acc_distracted"]]
    colors = [
        RETENTION_ACCURACY_COLORS["base"],
        RETENTION_ACCURACY_COLORS["clean"],
        RETENTION_ACCURACY_COLORS["distracted"],
    ]

    apply_publication_style()
    plt.figure(figsize=FIGSIZE_SINGLE_PANEL_TALL)
    bars = plt.bar(labels, values, color=colors, edgecolor="black", alpha=ALPHA_BAR)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.8, f"{height:.1f}%", ha="center", va="bottom", fontsize=11)

    plt.ylim(0, 100)
    plt.ylabel("Probe Accuracy (%)")
    plt.title("Dual-task Retention: Accuracy & MRI")
    txt = (
        f"Delta_M_clean={row['delta_m_clean']:.2f} pp\n"
        f"Delta_M_distracted={row['delta_m_distracted']:.2f} pp\n"
        f"MRI={row['mri_percent']:.2f}%"
    )
    plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes, va="top", ha="left", fontsize=10,
             bbox=dict(facecolor="white", alpha=ALPHA_ANNOTATION_BOX, edgecolor="gray"))
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_distractor_sensory_readout_on_axes(axes: Sequence[plt.Axes], metrics_dist: pd.DataFrame) -> None:
    row = metrics_dist.iloc[0]
    if len(axes) != 2:
        raise ValueError("plot_distractor_sensory_readout_on_axes expects exactly two axes")

    panel_specs = [
        {
            "title": "Panel A: Distractor classification",
            "ylabel": "Accuracy (%)",
            "cols": [
                ("Dynamic", "acc_distractor_dynamic", "acc_distractor_dynamic_ci95_lower", "acc_distractor_dynamic_ci95_upper"),
                ("Static", "acc_distractor_static", "acc_distractor_static_ci95_lower", "acc_distractor_static_ci95_upper"),
            ],
            "ylim": (0.0, 100.0),
            "gap_text": (
                f"dyn-static = {float(row['acc_distractor_gap_dynamic_minus_static']):.2f} pp\n"
                f"p(two-sided) = {float(row['acc_distractor_gap_p_two_sided_ne0']):.4g}"
            ),
        },
        {
            "title": "Panel B: Distractor silent proportion",
            "ylabel": "Silent trials (%)",
            "cols": [
                ("Dynamic", "silent_rate_distractor_dynamic", "silent_rate_distractor_dynamic_ci95_lower", "silent_rate_distractor_dynamic_ci95_upper"),
                ("Static", "silent_rate_distractor_static", "silent_rate_distractor_static_ci95_lower", "silent_rate_distractor_static_ci95_upper"),
            ],
            "ylim": (0.0, 100.0),
            "gap_text": (
                f"dyn-static = {float(row['silent_rate_distractor_gap_dynamic_minus_static']):.2f} pp\n"
                f"p(two-sided) = {float(row['silent_rate_distractor_gap_p_two_sided_ne0']):.4g}"
            ),
        },
    ]
    colors = ["#d62728", "#7f7f7f"]

    for ax, spec in zip(axes, panel_specs):
        labels = [x[0] for x in spec["cols"]]
        vals = np.array([float(row[x[1]]) for x in spec["cols"]], dtype=np.float64)
        lower = np.array([float(row[x[2]]) for x in spec["cols"]], dtype=np.float64)
        upper = np.array([float(row[x[3]]) for x in spec["cols"]], dtype=np.float64)
        yerr = np.vstack([vals - lower, upper - vals])
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", alpha=ALPHA_BAR, yerr=yerr, capsize=5)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 1.2,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_ylim(*spec["ylim"])
        ax.set_ylabel(spec["ylabel"])
        ax.set_title(spec["title"])
        ax.text(
            0.03,
            0.95,
            spec["gap_text"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=ALPHA_ANNOTATION_BOX),
        )


def plot_distractor_sensory_readout(metrics_dist: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL)
    plot_distractor_sensory_readout_on_axes(axes, metrics_dist)
    fig.suptitle("Distractor sensory readout under STSP-on vs STSP-off", y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_bias_attribution(metrics_bias: pd.DataFrame) -> plt.Figure:
    rows = []
    for _, r in metrics_bias.iterrows():
        cond = "Dynamic" if r["stsp_mode"] == "dynamic" else "Static"
        rows.append({"condition": cond, "bias_type": "Bias_sample", "value": r["bias_sample"]})
        rows.append({"condition": cond, "bias_type": "Bias_distractor", "value": r["bias_distractor"]})
        rows.append({"condition": cond, "bias_type": "Bias_noise", "value": r["bias_noise"]})
        rows.append({"condition": cond, "bias_type": "Bias_silent", "value": r["bias_silent"]})
    df_plot = pd.DataFrame(rows)

    apply_publication_style()
    plt.figure(figsize=(9.5, 5.8))
    ax = sns.barplot(
        data=df_plot,
        x="bias_type",
        y="value",
        hue="condition",
        palette=[RETENTION_BIAS_COLORS["dynamic"], RETENTION_BIAS_COLORS["static_frozen"]],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Probability in Probe Error Trials")
    ax.set_title("Error Attribution Bias (Distracted Condition)")
    ax.set_ylim(0, 1.0)
    apply_standard_legend(ax)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_sample_readout_survival_on_axis(ax: plt.Axes, metrics_survival: pd.DataFrame) -> None:
    row = metrics_survival.iloc[0]
    clean_sample = float(row["clean_sample_bias"])
    dist_sample = float(row["distracted_sample_bias"])
    dist_noise = float(row["distracted_noise_bias"])

    ci_clean = (
        float(row["clean_sample_bias_ci95_lower"]),
        float(row["clean_sample_bias_ci95_upper"]),
    )
    ci_dist_sample = (
        float(row["distracted_sample_bias_ci95_lower"]),
        float(row["distracted_sample_bias_ci95_upper"]),
    )
    ci_dist_noise = (
        float(row["distracted_noise_bias_ci95_lower"]),
        float(row["distracted_noise_bias_ci95_upper"]),
    )
    retention_pct = float(row["sample_bias_retention_pct"])
    p_one = float(row["p_one_sided_sample_gt_noise_distracted"])

    labels = ["Clean\nSample Bias", "Distracted\nSample Bias", "Distracted\nNoise Bias"]
    vals = [clean_sample, dist_sample, dist_noise]
    colors = ["#1f77b4", "#d62728", "#7f7f7f"]

    yerr = np.array(
        [
            [clean_sample - ci_clean[0], dist_sample - ci_dist_sample[0], dist_noise - ci_dist_noise[0]],
            [ci_clean[1] - clean_sample, ci_dist_sample[1] - dist_sample, ci_dist_noise[1] - dist_noise],
        ]
    )

    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, edgecolor="black", alpha=ALPHA_BAR, yerr=yerr, capsize=6)

    for i, v in enumerate(vals):
        ax.text(i, v + 0.0045, f"{v * 100:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.annotate(
        f"Retained sample-readout = {retention_pct:.2f}%",
        xy=(1, dist_sample),
        xytext=(0.2, max(vals) + 0.06),
        arrowprops=dict(arrowstyle="->", lw=1.6, color="black"),
        fontsize=11,
    )

    smoke_txt = (
        f"Distracted Sample Bias > Noise Bias\n(one-sided paired bootstrap p = {p_one:.4g})"
        if np.isfinite(p_one)
        else "Distracted Sample Bias > Noise Bias"
    )
    ax.text(
        1.0,
        max(vals) + 0.02,
        smoke_txt,
        ha="center",
        va="bottom",
        fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=ALPHA_ANNOTATION_BOX),
    )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Probability in Probe Error Trials")
    ax.set_ylim(0, max(vals) + 0.17)
    ax.set_title("Sample Memory Remains Readable After Distractor Washout")


def plot_sample_readout_survival(metrics_survival: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    plot_sample_readout_survival_on_axis(ax, metrics_survival)
    fig.tight_layout()
    return fig


def validate_metrics_consistency(df_trials: pd.DataFrame, metrics_acc: pd.DataFrame, eps: float = 1e-6) -> None:
    row = metrics_acc.iloc[0]
    acc_base = _accuracy(df_trials, "clean", "static_frozen", "probe")
    acc_clean = _accuracy(df_trials, "clean", "dynamic", "probe")
    acc_distracted = _accuracy(df_trials, "distracted", "dynamic", "probe")
    delta_clean = acc_base - acc_clean
    delta_distracted = acc_base - acc_distracted
    mri = 100.0 * delta_distracted / max(delta_clean, eps)

    check_pairs = [
        ("acc_base", acc_base),
        ("acc_clean", acc_clean),
        ("acc_distracted", acc_distracted),
        ("delta_m_clean", delta_clean),
        ("delta_m_distracted", delta_distracted),
        ("mri_percent", mri),
    ]
    for key, val in check_pairs:
        if not np.isclose(float(row[key]), float(val), atol=1e-8):
            raise ValueError(f"Metrics consistency check failed at {key}: {row[key]} vs {val}")

    acc_dist_dyn = _accuracy(df_trials, "distracted", "dynamic", "distractor")
    acc_dist_stat = _accuracy(df_trials, "distracted", "static_frozen", "distractor")
    for key, val in [
        ("acc_distractor_dynamic", acc_dist_dyn),
        ("acc_distractor_static", acc_dist_stat),
    ]:
        if not np.isclose(float(row[key]), float(val), atol=1e-8):
            raise ValueError(f"Metrics consistency check failed at {key}: {row[key]} vs {val}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-task anti-interference retention experiment")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--save-dir", type=str, default="results/dual_task_retention_experiment")
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)

    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay1-ms", type=float, default=400.0)
    parser.add_argument("--distractor-ms", type=float, default=200.0)
    parser.add_argument("--delay2-ms", type=float, default=400.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)

    parser.add_argument("--num-boot", type=int, default=10000)
    parser.add_argument("--skip-interface-check", action="store_true")
    parser.add_argument("--no-phase-reset", action="store_true")
    return parser

def main() -> None:
    args = build_argparser().parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_classes < 4:
        raise ValueError("--num-classes must be >= 4")
    if args.num_boot <= 0:
        raise ValueError("--num-boot must be positive")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=args.sample_ms,
        delay1_ms=args.delay1_ms,
        distractor_ms=args.distractor_ms,
        delay2_ms=args.delay2_ms,
        probe_ms=args.probe_ms,
        phase_reset=(not args.no_phase_reset),
    )

    for name, steps in [
        ("sample", spec.sample_steps),
        ("delay1", spec.delay1_steps),
        ("distractor", spec.distractor_steps),
        ("delay2", spec.delay2_steps),
        ("probe", spec.probe_steps),
    ]:
        if steps <= 0:
            raise ValueError(f"{name} steps must be positive")

    layout = prepare_result_layout(args.save_dir)

    print(f"[Init] Device: {device}")
    print(f"[Init] Save dir: {layout.root}")
    print(
        f"[Init] Timing steps | sample={spec.sample_steps}, delay1={spec.delay1_steps}, "
        f"distractor={spec.distractor_steps}, delay2={spec.delay2_steps}, probe={spec.probe_steps}"
    )
    clean_gap = spec.clean_delay_steps
    distracted_gap = spec.delay1_steps + spec.distractor_steps + spec.delay2_steps
    if clean_gap != distracted_gap:
        raise ValueError(
            f"Time mismatch: clean sample->probe gap {clean_gap} != distracted gap {distracted_gap} steps"
        )
    print(f"[Init] Time-matched sample->probe gap: clean={clean_gap} | distracted={distracted_gap} steps")
    print(f"[Init] Phase reset for dual task windows: {spec.phase_reset}")

    net, encoder = load_model_and_encoder(args.model_path, device, spec)
    if not args.skip_interface_check:
        run_interface_check(net, device)
        print("[Check] forward_dual_task_session interface check passed.")

    _, _, test_loader = build_mnist_skeleton_loader(batch_size=1)
    dataset = test_loader.dataset
    class_index = build_class_index(dataset, num_classes=args.num_classes)

    rng = random.Random(args.seed)
    df_specs = generate_trial_specs(class_index, num_trials=args.trials, num_classes=args.num_classes, rng=rng)
    validate_trial_specs(df_specs, num_classes=args.num_classes)
    trial_specs_csv = save_tidy_csv(df_specs, layout.data_file("trial_specs.csv"), sort_by=["trial_id", "paradigm", "stsp_mode"])

    df_trials = run_experiment(
        net=net,
        encoder=encoder,
        dataset=dataset,
        df_specs=df_specs,
        spec=spec,
        batch_size=args.batch_size,
        device=device,
    )
    validate_pairing(df_trials)

    trial_pred_csv = save_tidy_csv(
        df_trials,
        layout.data_file("trial_predictions.csv"),
        sort_by=["trial_id", "paradigm", "stsp_mode"],
    )

    metrics_acc = compute_accuracy_mri(df_trials)
    validate_metrics_consistency(df_trials, metrics_acc)
    metrics_acc_csv = save_tidy_csv(metrics_acc, layout.data_file("metrics_accuracy_mri.csv"))

    metrics_distractor = compute_distractor_sensory_readout(
        df_trials=df_trials,
        n_boot=args.num_boot,
        seed=args.seed + 303,
    )
    metrics_distractor_csv = save_tidy_csv(metrics_distractor, layout.data_file("metrics_distractor_sensory_readout.csv"))

    metrics_bias = compute_bias_table(df_trials, num_classes=args.num_classes)
    metrics_bias_csv = save_tidy_csv(metrics_bias, layout.data_file("metrics_bias_attribution.csv"), sort_by=["stsp_mode"])

    metrics_boot = paired_bootstrap_tests(
        df_trials=df_trials,
        num_classes=args.num_classes,
        n_boot=args.num_boot,
        seed=args.seed + 101,
    )
    metrics_boot_csv = save_tidy_csv(metrics_boot, layout.data_file("metrics_bootstrap_tests.csv"))

    metrics_survival = compute_sample_readout_survival(
        df_trials=df_trials,
        num_classes=args.num_classes,
        n_boot=args.num_boot,
        seed=args.seed + 202,
    )
    metrics_survival_csv = save_tidy_csv(metrics_survival, layout.data_file("metrics_sample_readout_survival.csv"))

    fig_acc = plot_accuracy_and_mri(metrics_acc)
    fig_acc_paths = save_figure_all_formats(fig_acc, layout.figure_base("accuracy_mri"))
    plt.close(fig_acc)
    fig_dist = plot_distractor_sensory_readout(metrics_distractor)
    fig_dist_paths = save_figure_all_formats(fig_dist, layout.figure_base("distractor_sensory_readout"))
    plt.close(fig_dist)
    fig_bias = plot_bias_attribution(metrics_bias)
    fig_bias_paths = save_figure_all_formats(fig_bias, layout.figure_base("bias_attribution"))
    plt.close(fig_bias)
    fig_survival = plot_sample_readout_survival(metrics_survival)
    fig_survival_paths = save_figure_all_formats(fig_survival, layout.figure_base("sample_readout_survival"))
    plt.close(fig_survival)

    print("\n=== Dual-task Retention Summary ===")
    row = metrics_acc.iloc[0]
    row_dist = metrics_distractor.iloc[0]
    print(f"Acc_base (clean+static): {row['acc_base']:.2f}%")
    print(f"Acc_clean (clean+dynamic): {row['acc_clean']:.2f}%")
    print(f"Acc_distracted (distracted+dynamic): {row['acc_distracted']:.2f}%")
    print(
        "Distractor Acc dynamic/static: "
        f"{row_dist['acc_distractor_dynamic']:.2f}% / {row_dist['acc_distractor_static']:.2f}% "
        f"(gap={row_dist['acc_distractor_gap_dynamic_minus_static']:.2f} pp, "
        f"p={row_dist['acc_distractor_gap_p_two_sided_ne0']:.4g})"
    )
    print(
        "Distractor Silent dynamic/static: "
        f"{row_dist['silent_rate_distractor_dynamic']:.2f}% / {row_dist['silent_rate_distractor_static']:.2f}% "
        f"(gap={row_dist['silent_rate_distractor_gap_dynamic_minus_static']:.2f} pp, "
        f"p={row_dist['silent_rate_distractor_gap_p_two_sided_ne0']:.4g})"
    )
    print(f"Delta_M_clean: {row['delta_m_clean']:.2f} pp")
    print(f"Delta_M_distracted: {row['delta_m_distracted']:.2f} pp")
    print(f"MRI: {row['mri_percent']:.2f}%")
    surv = metrics_survival.iloc[0]
    print(f"Sample-bias retention (distracted/clean): {surv['sample_bias_retention_pct']:.2f}%")
    print(f"Distracted sample>noise (one-sided p): {surv['p_one_sided_sample_gt_noise_distracted']:.4g}")
    print(f"Saved: {trial_specs_csv}")
    print(f"Saved: {trial_pred_csv}")
    summary_path = save_summary_json(
        {
            "experiment": "dual_task_retention",
            "key_metrics": {
                "acc_base": float(row["acc_base"]),
                "acc_clean": float(row["acc_clean"]),
                "acc_distracted": float(row["acc_distracted"]),
                "mri_percent": float(row["mri_percent"]),
                "acc_distractor_dynamic": float(row_dist["acc_distractor_dynamic"]),
                "acc_distractor_static": float(row_dist["acc_distractor_static"]),
                "sample_bias_retention_pct": float(surv["sample_bias_retention_pct"]),
            },
            "outputs": {
                "trial_specs_csv": str(trial_specs_csv),
                "trial_predictions_csv": str(trial_pred_csv),
                "metrics_accuracy_mri_csv": str(metrics_acc_csv),
                "metrics_distractor_sensory_readout_csv": str(metrics_distractor_csv),
                "metrics_bias_attribution_csv": str(metrics_bias_csv),
                "metrics_bootstrap_tests_csv": str(metrics_boot_csv),
                "metrics_sample_readout_survival_csv": str(metrics_survival_csv),
                "figure_accuracy_png": fig_acc_paths["png"],
                "figure_distractor_png": fig_dist_paths["png"],
                "figure_bias_png": fig_bias_paths["png"],
                "figure_survival_png": fig_survival_paths["png"],
            },
        },
        layout.root,
    )
    run_config_path = save_run_config(
        {
            "model_path": args.model_path,
            "seed": int(args.seed),
            "trials": int(args.trials),
            "batch_size": int(args.batch_size),
            "num_classes": int(args.num_classes),
            "num_boot": int(args.num_boot),
            "timing_ms": {
                "sample": float(args.sample_ms),
                "delay1": float(args.delay1_ms),
                "distractor": float(args.distractor_ms),
                "delay2": float(args.delay2_ms),
                "probe": float(args.probe_ms),
            },
            "phase_reset": bool(spec.phase_reset),
        },
        layout.root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=dual_task_retention",
            f"save_dir={layout.root}",
            f"trial_specs_csv={trial_specs_csv}",
            f"trial_predictions_csv={trial_pred_csv}",
            f"metrics_accuracy_mri_csv={metrics_acc_csv}",
            f"metrics_distractor_sensory_readout_csv={metrics_distractor_csv}",
            f"metrics_bias_attribution_csv={metrics_bias_csv}",
            f"metrics_bootstrap_tests_csv={metrics_boot_csv}",
            f"metrics_sample_readout_survival_csv={metrics_survival_csv}",
            f"figure_accuracy_png={fig_acc_paths['png']}",
            f"figure_distractor_png={fig_dist_paths['png']}",
            f"figure_bias_png={fig_bias_paths['png']}",
            f"figure_survival_png={fig_survival_paths['png']}",
            f"summary_json={summary_path}",
            f"run_config_json={run_config_path}",
        ],
        layout.log_dir,
    )

    print(f"Saved: {metrics_acc_csv}")
    print(f"Saved: {metrics_distractor_csv}")
    print(f"Saved: {metrics_bias_csv}")
    print(f"Saved: {metrics_boot_csv}")
    print(f"Saved: {metrics_survival_csv}")
    print(f"Saved: {fig_acc_paths['png']}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {run_config_path}")
    print(f"Saved: {run_log_path}")


if __name__ == "__main__":
    main()
