from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotting.common.colors import get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL, GRID_ALPHA_SOFT
from src.plotting.experiments._common import main_for, read_bundle_csv


LAYER_ORDER = ("layer1", "layer2", "layer3")


def _finish_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=GRID_ALPHA_SOFT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_NEUTRAL)
    ax.spines["bottom"].set_color(COLOR_NEUTRAL)
    ax.tick_params(color=COLOR_NEUTRAL)


def _layer_summary(df, *, value: str) -> list[float]:
    if "record_type" not in df.columns:
        raise ValueError("Required column missing: record_type")
    if "layer" not in df.columns:
        raise ValueError("Required column missing: layer")
    if value not in df.columns:
        raise ValueError(f"Required column missing: {value}")
    summary = df[df["record_type"].astype(str) == "layer_summary"].copy()
    if summary.empty:
        raise ValueError("No layer_summary rows found.")
    out = []
    for layer in LAYER_ORDER:
        sub = summary[summary["layer"].astype(str) == layer]
        out.append(float(sub[value].mean()) if not sub.empty else np.nan)
    return out


def _plot_dominance_index(similarity) -> plt.Figure:
    di_vals = _layer_summary(similarity, value="DI")
    x = np.arange(len(LAYER_ORDER), dtype=float)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.bar(x, di_vals, color=get_plot_color("first_item_reference"), width=0.58)
    ax.axhline(0.0, color=COLOR_NEUTRAL, linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_ORDER)
    ax.set_title("Dominance Index")
    ax.set_ylabel("sample-dominance score")
    _finish_axis(ax)
    fig.tight_layout()
    return fig


def _plot_di_vs_sample_first(coupling) -> plt.Figure:
    for column in ("record_type", "layer", "DI_mean", "sample_first_prob"):
        if column not in coupling.columns:
            raise ValueError(f"Required column missing: {column}")
    binned = coupling[coupling["record_type"].astype(str) == "binned_summary"].copy()
    if binned.empty:
        raise ValueError("No binned_summary rows found in ping_coupling_metrics.csv.")

    layer_colors = {
        "layer1": get_plot_color("whole_pair_representation"),
        "layer2": get_plot_color("donor_trace"),
        "layer3": get_plot_color("sample_probe_overlap"),
    }

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for layer in LAYER_ORDER:
        sub = binned[binned["layer"].astype(str) == layer].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("DI_mean", kind="stable")
        ax.plot(
            sub["DI_mean"].to_numpy(dtype=np.float64),
            sub["sample_first_prob"].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=2.0,
            color=layer_colors[layer],
            label=layer,
        )
    ax.set_title("DI vs Sample-First")
    ax.set_xlabel("DI bin mean")
    ax.set_ylabel("Sample-first prob.")
    ax.legend(frameon=False, fontsize=9)
    _finish_axis(ax)
    fig.tight_layout()
    return fig


def plot_bundle(input_dir: Path):
    similarity = read_bundle_csv(input_dir, "state_similarity_metrics.csv", ["record_type", "layer", "DI"])
    coupling = read_bundle_csv(input_dir, "ping_coupling_metrics.csv", ["record_type", "layer", "DI_mean", "sample_first_prob"])
    return {
        "dominance_index": _plot_dominance_index(similarity),
        "di_vs_sample_first": _plot_di_vs_sample_first(coupling),
    }


if __name__ == "__main__":
    raise SystemExit(main_for("chunk_stsp_state_taxonomy", plot_bundle, title="Chunk STSP State Taxonomy"))
