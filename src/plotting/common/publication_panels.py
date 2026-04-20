from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


_UX_CONDITION_A_DYNAMIC_BASE = "A_dynamic_base"
_UX_CONDITION_B_TRIAL_SHUFFLE_UX = "B_trial_shuffle_ux"
_UX_CONDITION_C_STATIC_FROZEN = "C_static_frozen"
_UX_CONDITION_ORDER = [
    _UX_CONDITION_A_DYNAMIC_BASE,
    _UX_CONDITION_B_TRIAL_SHUFFLE_UX,
    _UX_CONDITION_C_STATIC_FROZEN,
]


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
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", alpha=0.9, yerr=yerr, capsize=5)
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
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85),
        )


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
    ax.bar(x, vals, color=colors, edgecolor="black", alpha=0.9, yerr=yerr, capsize=6)

    for idx, value in enumerate(vals):
        ax.text(idx, value + 0.0045, f"{value * 100:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

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
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Probability in Probe Error Trials")
    ax.set_ylim(0, max(vals) + 0.17)
    ax.set_title("Sample Memory Remains Readable After Distractor Washout")


def plot_input_similarity_summary_on_axes(axes: Sequence[plt.Axes], df_summary: pd.DataFrame, source_name: str) -> None:
    del source_name
    metric_df = df_summary.drop_duplicates(subset=["bucket_name", "stsp_mode"]).copy()
    order = [x for x in ["low", "medium", "high"] if x in metric_df["bucket_name"].tolist()]
    flat_axes = np.asarray(list(axes), dtype=object).reshape(-1)

    if len(flat_axes) != 4:
        raise ValueError("plot_input_similarity_summary_on_axes expects exactly four axes")
    metric_specs = [
        ("probe_accuracy", "Probe accuracy (%)"),
        ("distractor_accuracy", "Distractor accuracy (%)"),
        ("sample_bias_retention_pct", "Sample-bias retention (%)"),
        ("sample_bias_minus_noise", "Sample bias - noise bias (pp)"),
    ]
    palette = {"dynamic": "#d62728", "static_frozen": "#7f7f7f"}

    for ax, (metric, title) in zip(flat_axes, metric_specs):
        sns.barplot(data=metric_df, x="bucket_name", y=metric, hue="stsp_mode", order=order, palette=palette, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        if metric != "sample_bias_minus_noise":
            ax.set_ylim(0, 100)
        ax.legend(title="")


def plot_overview_accuracy_panel(ax: plt.Axes, metrics_condition: pd.DataFrame) -> None:
    label_map = {
        _UX_CONDITION_A_DYNAMIC_BASE: "A: dynamic",
        _UX_CONDITION_B_TRIAL_SHUFFLE_UX: "B: trial-shuffle u/x",
        _UX_CONDITION_C_STATIC_FROZEN: "C: static frozen",
    }
    m = metrics_condition.set_index("condition").loc[_UX_CONDITION_ORDER].reset_index()
    x = np.arange(len(_UX_CONDITION_ORDER))

    colors = ["#d62728", "#1f77b4", "#7f7f7f"]
    ax.bar(x, m["acc_probe"].to_numpy(), color=colors, edgecolor="black", alpha=0.9)
    ax.set_xticks(x, [label_map[o] for o in _UX_CONDITION_ORDER], rotation=10)
    ax.set_ylabel("Probe Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Current Input Classification")


__all__ = [
    "plot_distractor_sensory_readout_on_axes",
    "plot_input_similarity_summary_on_axes",
    "plot_overview_accuracy_panel",
    "plot_sample_readout_survival_on_axis",
]
