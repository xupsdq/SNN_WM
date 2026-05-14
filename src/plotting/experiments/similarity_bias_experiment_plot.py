from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotting.common.colors import get_plot_color
from src.plotting.experiments._common import load_bundle_json, main_for, read_bundle_csv
from src.plotting.experiments._plot_builders import bar_figure


def draw_accuracy_drop_on_ax(ax: plt.Axes, df, *, title: str | None = None) -> None:
    """Draw the similarity-bin memory-bias bar plot on an existing axes."""
    plot_df = df.sort_values("bin_index", kind="stable").reset_index(drop=True)
    x = np.arange(len(plot_df), dtype=float)
    values = plot_df["acc_drop"].to_numpy(dtype=float)
    yerr = plot_df["sem_acc_drop"].to_numpy(dtype=float) if "sem_acc_drop" in plot_df.columns else None
    labels = plot_df["similarity_bin"].astype(str).tolist()
    bars = ax.bar(
        x,
        values,
        yerr=yerr,
        color=get_plot_color("sample_probe_overlap"),
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
        capsize=4,
    )
    ax.axhline(0.0, color=get_plot_color("other_residual"), linewidth=1.0, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("Similarity bin")
    ax.set_ylabel("Memory biase")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    y_padding = max(0.02, float(np.nanmax(np.abs(values))) * 0.04) if values.size else 0.02
    err_values = yerr if yerr is not None else np.zeros_like(values)
    for bar, value, err in zip(bars, values, err_values):
        xpos = bar.get_x() + bar.get_width() / 2.0
        if value >= 0:
            ypos = value + float(err) + y_padding
            va = "bottom"
        else:
            ypos = value - float(err) - y_padding
            va = "top"
        ax.text(xpos, ypos, f"{value:.3f}", ha="center", va=va, fontsize=11)

    ax.annotate(
        "",
        xy=(0.88, 1.08),
        xytext=(0.12, 1.08),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.1, "color": "black"},
        annotation_clip=False,
    )
    ax.text(
        0.5,
        1.14,
        "increasing sample-probe similarity",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=10,
    )


def _plot_accuracy_drop(df):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    draw_accuracy_drop_on_ax(ax, df)
    fig.subplots_adjust(top=0.82)
    return fig


def draw_overlap_bridge_on_ax(ax: plt.Axes, summary, *, title: str | None = "") -> None:
    """Draw the low/high overlap bridge bar plot on an existing axes."""
    low = summary.get("acc_drop_low")
    high = summary.get("acc_drop_high")
    values = [0.0 if low is None else float(low), 0.0 if high is None else float(high)]
    yerr = [
        0.0 if summary.get("sem_acc_drop_low") is None else float(summary["sem_acc_drop_low"]),
        0.0 if summary.get("sem_acc_drop_high") is None else float(summary["sem_acc_drop_high"]),
    ]
    x = np.arange(2, dtype=float)
    ax.bar(
        x,
        values,
        yerr=yerr,
        color=[get_plot_color("low_overlap"), get_plot_color("high_overlap")],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.8,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Low-overlap", "High-overlap"])
    ax.set_ylabel("Memory biase")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    y_padding = max(0.02, float(np.nanmax(np.abs(values))) * 0.04)
    for xpos, value, err in zip(x, values, yerr):
        if value >= 0:
            ypos = value + float(err) + y_padding
            va = "bottom"
        else:
            ypos = value - float(err) - y_padding
            va = "top"
        ax.text(xpos, ypos, f"{value:.3f}", ha="center", va=va, fontsize=11)


def _plot_overlap_bridge(summary):
    fig = bar_figure(
        ["Low-overlap", "High-overlap"],
        [0, 0],
        title="",
        ylabel="Memory biase",
        rotation=0,
    )
    ax = fig.axes[0]
    ax.clear()
    draw_overlap_bridge_on_ax(ax, summary)
    return fig


def plot_bundle(input_dir):
    df_accuracy = read_bundle_csv(input_dir, "bin_accuracy_summary.csv", ["bin_index", "similarity_bin", "acc_dynamic", "acc_static", "acc_drop"])
    overlap_summary = load_bundle_json(input_dir, "within_bin_overlap_summary.json")
    return {
        "figure_2_memory_bias_vs_similarity": _plot_accuracy_drop(df_accuracy),
        "figure_6_memory_bias_overlap_bridge": _plot_overlap_bridge(overlap_summary),
    }


if __name__ == "__main__":
    raise SystemExit(main_for("similarity_bias_experiment", plot_bundle, title="Similarity Bias Experiment"))
