from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotting.common.colors import get_plot_cmap, get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL, GRID_ALPHA_SOFT
from src.plotting.experiments._common import main_for, read_bundle_csv


TARGET_SEQ_LEN = 10
TARGET_LAYER = "layer3"
LAYER_COLORS = {
    "layer1": "whole_pair_representation",
    "layer2": "donor_trace",
    "layer3": "sample_probe_overlap",
}


def _finish_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, alpha=GRID_ALPHA_SOFT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_NEUTRAL)
    ax.spines["bottom"].set_color(COLOR_NEUTRAL)
    ax.tick_params(color=COLOR_NEUTRAL)


def _require_columns(df, columns: tuple[str, ...], source: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {', '.join(missing)}")


def _available_target_seq_len(df, *, source: str) -> int:
    _require_columns(df, ("seq_len",), source)
    seq_lens = sorted(df["seq_len"].dropna().astype(int).unique().tolist())
    if not seq_lens:
        raise ValueError(f"{source} has no sequence-length rows.")
    return int(TARGET_SEQ_LEN) if int(TARGET_SEQ_LEN) in seq_lens else int(seq_lens[-1])


def _seq_len_subset(df, *, source: str, layer: str | None = None):
    target_seq_len = _available_target_seq_len(df, source=source)
    sub = df[df["seq_len"].astype(int) == target_seq_len].copy()
    if layer is not None:
        _require_columns(sub, ("layer",), source)
        sub = sub[sub["layer"].astype(str) == layer].copy()
    if sub.empty:
        layer_msg = f" and layer={layer}" if layer is not None else ""
        raise ValueError(f"{source} has no rows for seq_len={target_seq_len}{layer_msg}.")
    return sub


def _plot_item_similarity_heatmap(item) -> plt.Figure:
    _require_columns(item, ("stage_k", "item_index", "layer", "similarity_weight_nonnegative"), "item_similarity_metrics.csv")
    sub = _seq_len_subset(item, source="item_similarity_metrics.csv", layer=TARGET_LAYER)
    target_seq_len = int(sub["seq_len"].iloc[0])
    matrix = (
        sub.groupby(["stage_k", "item_index"], sort=True)["similarity_weight_nonnegative"]
        .mean()
        .unstack("item_index")
        .reindex(index=range(1, target_seq_len + 1), columns=range(1, target_seq_len + 1))
        .to_numpy(dtype=np.float64)
    )
    vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(matrix, cmap=get_plot_cmap("item_contribution"), vmin=0.0, vmax=max(vmax, 1e-6), aspect="auto")
    ax.set_title(f"{TARGET_LAYER} seq_len={target_seq_len}")
    ax.set_xlabel("item position")
    ax.set_ylabel("stage k")
    ax.set_xticks(np.arange(target_seq_len))
    ax.set_xticklabels(np.arange(1, target_seq_len + 1))
    ax.set_yticks(np.arange(target_seq_len))
    ax.set_yticklabels(np.arange(1, target_seq_len + 1))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def _plot_anchor_position_vs_stage(summary) -> plt.Figure:
    _require_columns(summary, ("stage_k", "layer", "com_sim"), "similarity_summary_metrics.csv")
    sub = _seq_len_subset(summary, source="similarity_summary_metrics.csv", layer=TARGET_LAYER)
    target_seq_len = int(sub["seq_len"].iloc[0])
    grouped = sub.groupby("stage_k", as_index=False)["com_sim"].mean().sort_values("stage_k")

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(
        grouped["stage_k"].to_numpy(dtype=np.float64),
        grouped["com_sim"].to_numpy(dtype=np.float64),
        marker="o",
        linewidth=2.0,
        color=get_plot_color("anchor"),
        label=f"seq_len={target_seq_len}",
    )
    ax.set_xlabel("stage k")
    ax.set_ylabel("anchor position")
    ax.set_title("Anchor Drift vs Stage")
    ax.legend(frameon=False)
    _finish_axis(ax)
    fig.tight_layout()
    return fig


def _plot_ping_retrieval_profile(ping) -> plt.Figure:
    _require_columns(ping, ("seq_len", "stage_k", "item_index", "ping_weight"), "ping_retrieval_metrics.csv")
    final_ping = ping[ping["stage_k"].astype(int) == ping["seq_len"].astype(int)].copy()
    if final_ping.empty:
        raise ValueError("ping_retrieval_metrics.csv has no final-stage rows.")
    grouped = final_ping.groupby(["seq_len", "item_index"], as_index=False)["ping_weight"].mean().sort_values(["seq_len", "item_index"])

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    seq_colors = ("whole_pair_representation", "donor_trace", "sample_probe_overlap", "probe_only_region", "non_overlap_control")
    for idx, (seq_len, sub) in enumerate(grouped.groupby("seq_len", sort=True)):
        ax.plot(
            sub["item_index"].to_numpy(dtype=np.float64),
            sub["ping_weight"].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=2.0,
            color=get_plot_color(seq_colors[idx % len(seq_colors)]),
            label=f"seq_len={int(seq_len)}",
        )
    ax.set_xlabel("item position")
    ax.set_ylabel("retrieval probability")
    ax.set_title("Ping Retrieval Profile")
    ax.legend(frameon=False)
    _finish_axis(ax)
    fig.tight_layout()
    return fig


def _plot_stepwise_update_ratio(update) -> plt.Figure:
    _require_columns(update, ("layer", "stage_k", "stepwise_update_ratio"), "stepwise_update_metrics.csv")
    grouped = update.groupby(["layer", "stage_k"], as_index=False)["stepwise_update_ratio"].mean().sort_values(["layer", "stage_k"])

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for layer_key, sub in grouped.groupby("layer", sort=True):
        color_key = LAYER_COLORS.get(str(layer_key), "other_residual")
        ax.plot(
            sub["stage_k"].to_numpy(dtype=np.float64),
            sub["stepwise_update_ratio"].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=2.0,
            color=get_plot_color(color_key),
            label=str(layer_key),
        )
    ax.set_xlabel("stage k")
    ax.set_ylabel("STSP update fraction")
    ax.set_title("Stepwise Update Ratio")
    ax.legend(frameon=False)
    _finish_axis(ax)
    fig.tight_layout()
    return fig


def plot_bundle(input_dir: Path):
    item = read_bundle_csv(input_dir, "item_similarity_metrics.csv")
    summary = read_bundle_csv(input_dir, "similarity_summary_metrics.csv")
    ping = read_bundle_csv(input_dir, "ping_retrieval_metrics.csv")
    update = read_bundle_csv(input_dir, "stepwise_update_metrics.csv")
    return {
        "anchor_position_vs_stage": _plot_anchor_position_vs_stage(summary),
        "item_similarity_heatmap": _plot_item_similarity_heatmap(item),
        "ping_retrieval_profile": _plot_ping_retrieval_profile(ping),
        "stepwise_update_ratio": _plot_stepwise_update_ratio(update),
    }


if __name__ == "__main__":
    raise SystemExit(main_for("chunk_stsp_multiitem_sequence", plot_bundle, title="Chunk STSP Multiitem Sequence"))
