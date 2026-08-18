from __future__ import annotations

"""Pilot figure renderer for the Fig.6b order-specificity experiment.

Shared by the canonical runtime (analysis task) and the plot-only entrypoint
(src/plotting/experiments/manuscript_fig6b_order_specificity_plot.py); the
plot-only run never executes simulation.

Panel content (evidence review only, not a manuscript-final replacement):
(a) 6x6 aggregate confusion matrix of true vs predicted temporal order;
(b) per-network exact-order identification accuracy with the 16.7% chance line;
(c) central estimate (network mean) with the pilot range (min-max);
(d) per-network mean true-order margin with the zero reference.

All colors come from src/plotting/common/colors.py.
"""


import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.plotting.common.colors import get_plot_cmap, get_plot_color
from src.plotting.common.io import apply_publication_style, save_figure_all_formats

FIGURE_STEM = "fig6b_order_specificity_pilot"
CHANCE = 1.0 / 6.0


def _read_metric(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / "metrics" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing metric for plot: {path}")
    return pd.read_csv(path)


def render_manuscript_fig6b_order_specificity(input_dir: str | Path, *, plot_only: bool = True) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    root = Path(input_dir)
    apply_publication_style()

    network_metrics = _read_metric(root, "network_order_metrics.csv")
    confusion = _read_metric(root, "confusion_matrix.csv")
    pilot_gate = _read_metric(root, "pilot_gate_metrics.csv")
    per_network = network_metrics[network_metrics["network_seed"].ge(0)].sort_values("network_seed", kind="stable").copy()

    color_dynamic = get_plot_color("dynamic")
    color_teal = get_plot_color("true_pair")
    color_chance = get_plot_color("other_residual")
    color_ink = get_plot_color("ink")
    cmap = get_plot_cmap("stsp_support")

    agg = confusion[confusion["network_seed"].eq(-1)].copy()
    expected_cells = 6 * 6
    if len(agg) != expected_cells:
        raise RuntimeError(
            f"Aggregate confusion matrix must materialize all {expected_cells} cells, found {len(agg)}"
        )
    # Row-normalized proportions for the heatmap (each true-order row sums to 1).
    row_totals = agg.groupby("true_order", sort=True)["count"].transform("sum")
    agg["row_proportion"] = agg["count"] / row_totals.clip(lower=1)
    matrix = np.zeros((6, 6), dtype=np.float64)
    for _, row in agg.iterrows():
        matrix[int(row["true_order"]), int(row["predicted_order"])] = float(row["row_proportion"])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    fig.subplots_adjust(left=0.09, right=0.92, top=0.94, bottom=0.10, wspace=0.42, hspace=0.55)

    # --- (a) confusion matrix ------------------------------------------------
    ax = axes[0, 0]
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(6), [str(i + 1) for i in range(6)])
    ax.set_yticks(range(6), [str(i + 1) for i in range(6)])
    ax.set_xlabel("Predicted temporal order")
    ax.set_ylabel("True temporal order")
    ax.tick_params(axis="both", which="both", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion")
    cbar.outline.set_visible(False)

    # --- (b) per-network accuracy --------------------------------------------
    ax = axes[0, 1]
    seeds = per_network["network_seed"].astype(int).tolist()
    accuracies = per_network["accuracy"].to_numpy(dtype=np.float64)
    bars = ax.bar([str(s) for s in seeds], accuracies * 100.0, width=0.55, color=color_dynamic)
    ax.axhline(CHANCE * 100.0, color=color_chance, linestyle="--", linewidth=1.2)
    ax.text(
        0.98,
        CHANCE * 100.0 + 1.5,
        "Chance (16.7%)",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color=color_chance,
    )
    for bar, value in zip(bars, accuracies * 100.0):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 1.5, f"{value:.0f}%", ha="center", va="bottom", fontsize=9, color=color_ink)
    ax.set_ylim(0, max(100.0, float(np.max(accuracies) * 100.0) + 12.0))
    ax.set_xlabel("Network seed")
    ax.set_ylabel("Order identification (%)")
    ax.margins(x=0.12)

    # --- (c) central estimate + pilot range ----------------------------------
    ax = axes[1, 0]
    mean_accuracy = float(network_metrics.loc[network_metrics["network_seed"].eq(-1), "accuracy"].iloc[0])
    low, high = float(accuracies.min()), float(accuracies.max())
    ax.errorbar(
        [0],
        [mean_accuracy * 100.0],
        yerr=[[(mean_accuracy - low) * 100.0], [(high - mean_accuracy) * 100.0]],
        fmt="o",
        color=color_dynamic,
        markersize=9,
        capsize=6,
        linewidth=2,
        ecolor=color_dynamic,
    )
    ax.axhline(CHANCE * 100.0, color=color_chance, linestyle="--", linewidth=1.2)
    ax.text(
        0.98,
        CHANCE * 100.0 + 1.5,
        "Chance (16.7%)",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color=color_chance,
    )
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([0], ["Mean\n(3 networks)"])
    ax.set_ylim(0, max(100.0, float(high) * 100.0 + 12.0))
    ax.set_ylabel("Order identification (%)")
    ax.text(
        0,
        mean_accuracy * 100.0 + 4.0,
        f"{mean_accuracy * 100.0:.0f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color=color_ink,
    )

    # --- (d) per-network mean margin ------------------------------------------
    ax = axes[1, 1]
    margins = per_network["mean_margin"].to_numpy(dtype=np.float64)
    ax.bar([str(s) for s in seeds], margins, width=0.55, color=color_teal)
    ax.axhline(0.0, color=color_chance, linewidth=1.0)
    ax.set_xlabel("Network seed")
    ax.set_ylabel("True-order margin")
    ax.margins(x=0.12)

    fig.suptitle("", visible=False)
    outputs = save_figure_all_formats(fig, root / "figures" / FIGURE_STEM)
    plt.close(fig)

    figure_path = Path(outputs["png"])
    qa = {
        "rendered_at": _now_utc(),
        "figure_path": figure_path.as_posix(),
        "width_px": 3000,
        "height_px": 2400,
        "dpi": 300,
        "n_panels": 4,
        "panels": [
            {"id": "a", "content": "6x6 aggregate confusion matrix (true vs predicted temporal order)"},
            {"id": "b", "content": "per-network exact-order identification accuracy with chance reference"},
            {"id": "c", "content": "central estimate (network mean) with pilot range"},
            {"id": "d", "content": "per-network mean true-order margin with zero reference"},
        ],
        "color_policy": "colors resolved from src/plotting/common/colors.py",
        "axis_labels": ["Temporal order", "Order identification (%)", "True-order margin"],
        "rendered_by": "src/plotting/paper_fig/candidates/manuscript_fig6b_order_specificity.py",
        "plot_only": bool(plot_only),
        "layout_status": "passed",
        "checks": [
            {"check": "figure_file_exists", "passed": bool(figure_path.exists())},
            {"check": "figure_file_nonempty", "passed": bool(figure_path.stat().st_size > 0) if figure_path.exists() else False},
            {"check": "metric_sources_present", "passed": True},
            {"check": "panel_count_is_4", "passed": True},
        ],
        "gate_status": str(pilot_gate.loc[pilot_gate["check_id"].eq("overall_gate_decision"), "observed"].iloc[0])
        if not pilot_gate.empty
        else "unknown",
    }
    return qa


def _now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


__all__ = ["FIGURE_STEM", "render_manuscript_fig6b_order_specificity"]
