from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotting.common.colors import get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL
from src.plotting.experiments._common import load_bundle_npz, main_for, optional_bundle_file, read_bundle_csv


EPISODE_TIMELINE_NPZ = "episode_timeline_example.npz"


def _scalar(payload: dict[str, np.ndarray], key: str) -> float:
    if key not in payload:
        raise KeyError(f"{EPISODE_TIMELINE_NPZ} missing required array: {key}")
    arr = np.asarray(payload[key], dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{EPISODE_TIMELINE_NPZ}:{key} is empty")
    return float(arr[0])


def _image(payload: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"{EPISODE_TIMELINE_NPZ} missing required array: {key}")
    arr = np.squeeze(np.asarray(payload[key], dtype=float))
    if arr.ndim != 2:
        raise ValueError(f"{EPISODE_TIMELINE_NPZ}:{key} must be a 2D image")
    return arr


def _format_ms(value: float) -> str:
    return f"{value:.0f} ms"


def _plot_episode_timeline(payload: dict[str, np.ndarray]) -> plt.Figure:
    item1 = _image(payload, "item1_image")
    item2 = _image(payload, "item2_image")
    durations = [
        _scalar(payload, "sample_ms"),
        _scalar(payload, "delay1_ms"),
        _scalar(payload, "item2_ms"),
        _scalar(payload, "delay2_ms"),
    ]
    labels = ["Item 1", "Delay 1", "Item 2", "Delay 2"]
    images = [item1, None, item2, None]
    fills = [
        get_plot_color("sample_window"),
        get_plot_color("background_shade"),
        get_plot_color("probe_window"),
        get_plot_color("background_shade"),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    ax.set_axis_off()
    frame_y = 0.28
    frame_h = 0.48
    input_w = 0.145
    reference_ms = max(durations[0], 1.0)
    widths = [input_w * max(duration, 1.0) / reference_ms for duration in durations]
    total_w = sum(widths)
    start_x = (1.0 - total_w) / 2.0
    x = start_x
    capture_x = start_x + total_w

    for label, duration, image, fill, width in zip(labels, durations, images, fills, widths):
        ax.text(x + width / 2.0, frame_y + frame_h + 0.07, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=11, color=COLOR_NEUTRAL)
        ax.text(x + width / 2.0, frame_y - 0.08, _format_ms(duration), transform=ax.transAxes, ha="center", va="top", fontsize=10, color=COLOR_NEUTRAL)
        if image is None:
            rect = Rectangle((x, frame_y), width, frame_h, transform=ax.transAxes, facecolor=fill, edgecolor=COLOR_NEUTRAL, linewidth=1.0)
            ax.add_patch(rect)
        else:
            image_ax = ax.inset_axes(
                [x, frame_y, width, frame_h],
                transform=ax.transAxes,
            )
            image_ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            image_ax.set_xticks([])
            image_ax.set_yticks([])
            for spine in image_ax.spines.values():
                spine.set_color(COLOR_NEUTRAL)
                spine.set_linewidth(1.0)
        x += width

    ax.plot([capture_x, capture_x], [frame_y, frame_y + frame_h], transform=ax.transAxes, color=get_plot_color("anchor"), linewidth=2.0)
    ax.text(
        capture_x,
        frame_y + frame_h + 0.07,
        "STSP\ncapture",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        linespacing=0.95,
        color=get_plot_color("anchor"),
    )
    fig.tight_layout()
    return fig


def _finish_reference_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_NEUTRAL)
    ax.spines["bottom"].set_color(COLOR_NEUTRAL)
    ax.tick_params(labelsize=8, color=COLOR_NEUTRAL)


def _plot_layer_violin(df, *, value: str, ylabel: str) -> plt.Figure:
    if "layer" not in df.columns:
        raise ValueError("Required column missing: layer")
    if value not in df.columns:
        raise ValueError(f"Required column missing: {value}")
    grouped = [(str(layer), values.to_numpy(dtype=float)) for layer, values in df.groupby("layer", sort=True)[value]]
    grouped = [(label, arr[np.isfinite(arr)]) for label, arr in grouped]
    if not grouped or any(arr.size == 0 for _, arr in grouped):
        raise ValueError(f"{value} has no finite values to plot")

    color = get_plot_color("fused_state", context="fig4_fusion")
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    positions = np.arange(1, len(grouped) + 1, dtype=float)
    violins = ax.violinplot([arr for _, arr in grouped], positions=positions, widths=0.82, showextrema=False)
    for body in violins["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(get_plot_color("donor_trace"))
        body.set_alpha(0.96)
        body.set_linewidth(0.8)

    rng = np.random.default_rng(17)
    for xpos, (_, arr) in zip(positions, grouped):
        jitter = rng.normal(0.0, 0.065, size=arr.size)
        ax.scatter(np.full(arr.size, xpos) + jitter, arr, s=4.5, color=COLOR_NEUTRAL, alpha=0.58, linewidths=0)

    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _ in grouped])
    ax.set_ylabel(ylabel)
    _finish_reference_axis(ax)
    fig.tight_layout()
    return fig


def _plot_wpri_distribution(df):
    value_col = "WPRI" if "WPRI" in df.columns else "whole_part_ratio_index"
    if value_col not in df.columns:
        raise ValueError("whole_over_part_metrics.csv missing required WPRI column")
    values = df[value_col].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("WPRI has no finite values to plot")

    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    ax.hist(values, bins=22, density=True, color=get_plot_color("fused_state", context="fig4_fusion"), edgecolor=COLOR_NEUTRAL, linewidth=0.8, alpha=0.72)
    ax.axvline(0.0, color=COLOR_NEUTRAL, linestyle=":", linewidth=1.1)
    ax.set_xlabel("WPRI")
    ax.set_ylabel("Density")
    xmin = min(0.0, float(np.nanmin(values)) - 0.01)
    xmax = float(np.nanmax(values)) + 0.01
    ax.set_xlim(xmin, xmax)
    _finish_reference_axis(ax)
    fig.tight_layout()
    return fig


def _plot_true_vs_shuffled_pair_score(df) -> plt.Figure:
    required = ("true_pair_score", "shuffled_pair_score")
    for column in required:
        if column not in df.columns:
            raise ValueError(f"fusion_specificity_metrics.csv missing required column: {column}")
    values = [df[column].to_numpy(dtype=float) for column in required]
    values = [arr[np.isfinite(arr)] for arr in values]
    if any(arr.size == 0 for arr in values):
        raise ValueError("Pair specificity score columns have no finite values to plot")

    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    box = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.58,
        tick_labels=["True pair", "Shuffled pair"],
        medianprops={"color": COLOR_NEUTRAL, "linewidth": 1.2},
        boxprops={"edgecolor": COLOR_NEUTRAL, "linewidth": 0.9},
        whiskerprops={"color": COLOR_NEUTRAL, "linewidth": 0.9},
        capprops={"color": COLOR_NEUTRAL, "linewidth": 0.9},
        flierprops={"marker": "o", "markersize": 3.5, "markerfacecolor": "white", "markeredgecolor": COLOR_NEUTRAL, "markeredgewidth": 0.7},
    )
    colors = [get_plot_color("true_pair"), get_plot_color("shuffled_pair")]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_ylabel("Pair score")
    _finish_reference_axis(ax)
    fig.tight_layout()
    return fig


def _require_episode_artifact(input_dir: Path) -> None:
    if optional_bundle_file(input_dir, EPISODE_TIMELINE_NPZ) is None:
        raise FileNotFoundError(
            f"{EPISODE_TIMELINE_NPZ} is required for panel_a_episode_timeline. "
            "Re-run the chunk_step2_fused_state_experiment computation to refresh the result bundle."
        )


def plot_bundle(input_dir: Path):
    _require_episode_artifact(input_dir)
    episode = load_bundle_npz(input_dir, EPISODE_TIMELINE_NPZ)
    fusion = read_bundle_csv(input_dir, "preprobe_fusion_metrics.csv", ["layer", "fusion_dual_score", "fusion_imbalance"])
    specificity = read_bundle_csv(input_dir, "fusion_specificity_metrics.csv", ["layer", "true_pair_score", "shuffled_pair_score"])
    whole = read_bundle_csv(input_dir, "whole_over_part_metrics.csv", ["layer"])
    return {
        "panel_a_episode_timeline": _plot_episode_timeline(episode),
        "panel_c_fusion_dual_score": _plot_layer_violin(fusion, value="fusion_dual_score", ylabel="Fusion dual score"),
        "panel_c_fusion_imbalance": _plot_layer_violin(fusion, value="fusion_imbalance", ylabel="Fusion imbalance"),
        "panel_e_wpri_distribution": _plot_wpri_distribution(whole),
        "panel_f_true_vs_shuffled_pair_score": _plot_true_vs_shuffled_pair_score(specificity),
    }


if __name__ == "__main__":
    raise SystemExit(main_for("chunk_step2_fused_state_experiment", plot_bundle, title="Chunk Step2 Fused State"))
