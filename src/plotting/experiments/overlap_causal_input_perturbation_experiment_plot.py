from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotting.common.colors import get_plot_color
from src.plotting.common.theme_tokens import ALPHA_FILL, GRID_ALPHA_SOFT, LINE_WIDTH_PRIMARY
from src.plotting.experiments._common import load_bundle_npz, main_for
from src.plotting.experiments._plot_builders import trace_mean_sem


OVERLAP_CONDITIONS = (
    ("sample_keep_overlap_only_dynamic", "Overlap", "sample_probe_overlap"),
    ("sample_keep_nonoverlap_only_dynamic", "Non-overlap", "non_overlap_control"),
)


def draw_dpi_l3_trace_on_ax(ax: plt.Axes, trace_payload, *, title: str | None = None) -> None:
    """Draw the overlap-vs-nonoverlap DPI trace on an existing axes."""
    ax.axvspan(0, 20, color=get_plot_color("sample_window"), alpha=0.45, linewidth=0)
    ax.axvline(20, color=get_plot_color("other_residual"), linewidth=1.2, linestyle="--")
    all_y_values = []
    for condition, label, color_key in OVERLAP_CONDITIONS:
        time_axis, mean, err = trace_mean_sem(trace_payload, condition, "DPI_L3")
        time_axis = time_axis[:60]
        mean = mean[:60]
        err = err[:60]
        color = get_plot_color(color_key)
        ax.plot(time_axis, mean, color=color, linewidth=LINE_WIDTH_PRIMARY, label=label)
        ax.fill_between(time_axis, mean - err, mean + err, color=color, alpha=ALPHA_FILL)
        all_y_values.extend((mean - err).tolist())
        all_y_values.extend((mean + err).tolist())
        if mean.size:
            peak_index = int(np.nanargmax(mean))
            peak_x = float(time_axis[peak_index])
            peak_y = float(mean[peak_index])
            ax.scatter([peak_x], [peak_y], color=color, edgecolor="white", linewidth=0.8, zorder=4)
            ax.annotate(
                f"{peak_y:.3f}",
                xy=(peak_x, peak_y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=color,
                fontsize=10,
                fontweight="bold",
            )
    if all_y_values:
        finite = np.asarray(all_y_values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            ymin = float(finite.min())
            ymax = float(finite.max())
            pad = max(0.02, (ymax - ymin) * 0.18)
            ax.set_ylim(ymin - pad * 0.25, ymax + pad)
    ax.axhline(0.0, color=get_plot_color("other_residual"), linewidth=1.0, linestyle=":")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("dynamic like fraction")
    if title:
        ax.set_title(title)
    ax.grid(alpha=GRID_ALPHA_SOFT)
    ax.legend(frameon=False)
    ax.text(
        10,
        0.98,
        "probe input section",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
        color="black",
    )


def _plot_dpi_l3_trace(trace_payload):
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    draw_dpi_l3_trace_on_ax(ax, trace_payload)
    fig.tight_layout()
    return fig


def plot_bundle(input_dir):
    trace_payload = load_bundle_npz(input_dir, "pair_trace_similarity.npz")
    return {
        "dpi_l3_trace_overlap_vs_nonoverlap": _plot_dpi_l3_trace(trace_payload),
    }


if __name__ == "__main__":
    raise SystemExit(
        main_for(
            "overlap_causal_input_perturbation_experiment",
            plot_bundle,
            title="Overlap Causal Input Perturbation",
        )
    )
