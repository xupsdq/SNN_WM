from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.common.colors import get_plot_color
from src.plotting.experiments._common import main_for, read_bundle_csv


CONDITION_LABELS = {
    "original": "Original",
    "shuffled": "Shuffled",
    "shuffle": "Shuffle",
    "full_dynamic": "Dynamic",
    "static_frozen": "Static",
}


def _nice_axis_upper(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    raw = max(float(finite.max()) * 1.18, 1.0)
    base = 10.0 ** np.floor(np.log10(raw))
    for mult in (1.0, 2.0, 5.0, 10.0):
        candidate = mult * base
        if candidate >= raw:
            return float(candidate)
    return float(10.0 * base)


def plot_bundle(input_dir):
    metrics = read_bundle_csv(
        input_dir,
        "metrics_condition_summary.csv",
        ["condition", "abs_rate_pred_original_sample", "abs_rate_pred_change_under_bmap"],
    )
    metrics = metrics.sort_values("condition", kind="stable").reset_index(drop=True)
    x = np.arange(len(metrics), dtype=float)
    width = 0.38
    original = metrics["abs_rate_pred_original_sample"].to_numpy(dtype=float)
    changed = metrics["abs_rate_pred_change_under_bmap"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.bar(x - width / 2, original, width=width, color=get_plot_color("original_sample_trace"), edgecolor="black", alpha=0.9, label="Pred = original sample")
    ax.bar(x + width / 2, changed, width=width, color=get_plot_color("donor_trace"), edgecolor="black", alpha=0.9, label="Pred = change (B-map)")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(str(item), str(item)) for item in metrics["condition"]], rotation=10, ha="right")
    ax.set_ylabel("Absolute Rate (%)")
    ax.set_ylim(0.0, _nice_axis_upper(np.concatenate([original, changed])))
    ax.set_title("Memory Readout Target by Shuffled Substrate")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    return {"memory_readout_target": fig}


if __name__ == "__main__":
    raise SystemExit(main_for("ux_shuffle_memory_collapse", plot_bundle, title="UX Shuffle Memory Collapse"))
