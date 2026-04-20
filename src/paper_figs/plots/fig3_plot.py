from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.paper_figs.plots.common import load_npz, read_csv_validated, require_path, resolve_figure_input_dir
from src.paper_figs.plots.style import (
    ANNOTATION_SIZE,
    COLOR_DARK_GRAY,
    COLOR_DYNAMIC,
    COLOR_GRID,
    COLOR_OVERLAP,
    COLOR_PROBE_ONLY,
    COLOR_SAMPLE_ONLY,
    COLOR_TEXT,
    DATA_LINEWIDTH,
    REF_LINEWIDTH,
    add_reference_line,
    apply_paper_style,
    save_figure_outputs,
    style_axes,
)


def load_fig3_bundle(root: str | Path) -> dict[str, object]:
    root_path = Path(root)
    return {
        "summary": json.loads(require_path(root_path / "summary.json").read_text(encoding="utf-8")),
        "panel_a_similarity": read_csv_validated(
            root_path / "data" / "panel_a_similarity_bin_accuracy.csv",
            ["similarity_bin", "probe_accuracy_dynamic", "probe_accuracy_static", "acc_drop"],
        ),
        "panel_a_bridge": read_csv_validated(
            root_path / "data" / "panel_a_within_bin_overlap_bridge.csv",
            ["group", "mean_overlap", "acc_drop", "sem_acc_drop"],
        ),
        "panel_b_trace": read_csv_validated(
            root_path / "data" / "panel_b_dpi_trace_summary.csv",
            ["condition", "time_step", "time_ms", "dpi_mean", "dpi_sem"],
        ),
        "panel_b_pair": read_csv_validated(
            root_path / "data" / "panel_b_dpi_pair_summary.csv",
            ["pair_id", "condition", "mean_DPI_L3", "peak_DPI_L3"],
        ),
        "panel_cd_pair_metrics": read_csv_validated(
            root_path / "data" / "panel_cd_pair_level_metrics.csv",
            ["pair_id", "reconstruction_cosine_plus", "reconstruction_cosine_minus", "direction_match_plus", "direction_match_minus"],
        ),
        "panel_c_summary": read_csv_validated(
            root_path / "data" / "panel_c_reconstruction_summary.csv",
            ["mode", "mean_cosine", "sem_cosine"],
        ),
        "panel_d_summary": read_csv_validated(
            root_path / "data" / "panel_d_direction_summary.csv",
            ["mode", "direction_match_rate"],
        ),
        "panel_b_arrays": load_npz(root_path / "arrays" / "panel_b_probe_trace_arrays.npz"),
        "panel_cd_arrays": load_npz(root_path / "arrays" / "panel_cd_reconstruction_vectors.npz"),
    }


def _pretty_condition(condition: str) -> str:
    mapping = {
        "sample_keep_overlap_only_dynamic": "overlap-only",
        "sample_keep_nonoverlap_only_dynamic": "non-overlap",
    }
    return mapping.get(condition, condition.replace("_", " "))


def _framed_legend(ax: plt.Axes, handles: list[Line2D]) -> None:
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor=COLOR_GRID,
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=1.6,
    )


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 1000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        draw = rng.choice(values, size=values.size, replace=True)
        samples[idx] = float(np.mean(draw))
    return float(np.mean(values)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def draw_panel_a_branch(fig: plt.Figure, spec, similarity_df: pd.DataFrame, bridge_df: pd.DataFrame) -> tuple[plt.Axes, plt.Axes]:
    host = fig.add_subplot(spec)
    host.set_axis_off()
    ax_left = host.inset_axes([0.00, 0.05, 0.68, 0.90])
    ax_right = host.inset_axes([0.73, 0.21, 0.25, 0.58], sharey=ax_left)
    style_axes(ax_left)
    style_axes(ax_right)

    sim_df = similarity_df.sort_values("similarity_bin").reset_index(drop=True)
    x_main = np.arange(len(sim_df), dtype=float)
    y_main = 100.0 * sim_df["acc_drop"].to_numpy(dtype=float)
    bar_colors = [COLOR_SAMPLE_ONLY, COLOR_SAMPLE_ONLY, COLOR_SAMPLE_ONLY, COLOR_OVERLAP]
    ax_left.bar(x_main, y_main, width=0.62, color=bar_colors, alpha=0.86, edgecolor="none", zorder=1)
    ax_left.plot(x_main, y_main, color=COLOR_DARK_GRAY, linewidth=DATA_LINEWIDTH, zorder=3)
    ax_left.scatter(
        x_main,
        y_main,
        s=[18, 18, 18, 34],
        color=[COLOR_DARK_GRAY, COLOR_DARK_GRAY, COLOR_DARK_GRAY, COLOR_OVERLAP],
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
    )
    for x_val, y_val in zip(x_main, y_main):
        ax_left.text(x_val, y_val + 1.1, f"{y_val:.1f}", ha="center", va="bottom", fontsize=ANNOTATION_SIZE)
    ax_left.set_xticks(x_main, ["bin 1", "bin 2", "bin 3", "bin 4"])
    ax_left.set_ylabel("accuracy drop (%)")

    peak_x = float(x_main[-1])
    peak_y = float(y_main[-1])
    box_y0 = peak_y - 2.4
    box_y1 = peak_y + 2.4
    highlight = Rectangle(
        (peak_x - 0.34, box_y0),
        0.68,
        box_y1 - box_y0,
        fill=False,
        edgecolor=COLOR_OVERLAP,
        linewidth=0.9,
        linestyle=(0, (3, 2)),
    )
    ax_left.add_patch(highlight)

    bridge_order = ["low_overlap", "high_overlap"]
    bridge = bridge_df.set_index("group").reindex(bridge_order).reset_index()
    x_sub = np.arange(len(bridge), dtype=float)
    y_sub = 100.0 * bridge["acc_drop"].to_numpy(dtype=float)
    err_sub = 100.0 * bridge["sem_acc_drop"].fillna(0.0).to_numpy(dtype=float)
    colors = [COLOR_SAMPLE_ONLY, COLOR_OVERLAP]

    ax_right.bar(x_sub, y_sub, width=0.55, color=colors, alpha=0.86, edgecolor="none", zorder=1)
    ax_right.errorbar(x_sub, y_sub, yerr=err_sub, fmt="none", ecolor=COLOR_DARK_GRAY, elinewidth=0.8, capsize=0, zorder=2)
    for x_val, y_val in zip(x_sub, y_sub):
        ax_right.text(x_val, y_val + 1.1, f"{y_val:.1f}", ha="center", va="bottom", fontsize=ANNOTATION_SIZE)
    ax_right.set_xticks(x_sub, ["low ov.", "high ov."])
    ax_right.tick_params(left=False, labelleft=False)
    ax_right.spines["top"].set_linestyle((0, (3, 2)))
    ax_right.spines["right"].set_linestyle((0, (3, 2)))
    ax_right.spines["left"].set_linestyle((0, (3, 2)))
    ax_right.spines["bottom"].set_linestyle((0, (3, 2)))
    ax_right.spines["top"].set_visible(True)
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["top"].set_linewidth(0.85)
    ax_right.spines["right"].set_linewidth(0.85)

    y_min = min(float(np.min(y_main)), float(np.min(y_sub))) - 4.0
    y_max = max(float(np.max(y_main)), float(np.max(y_sub))) + 5.0
    ax_left.set_ylim(y_min, y_max)

    con_top = ConnectionPatch(
        xyA=(peak_x + 0.34, box_y1),
        coordsA=ax_left.transData,
        xyB=(0.0, 1.0),
        coordsB=ax_right.transAxes,
        color=COLOR_DARK_GRAY,
        linewidth=0.7,
        linestyle=(0, (3, 2)),
    )
    con_bottom = ConnectionPatch(
        xyA=(peak_x + 0.34, box_y0),
        coordsA=ax_left.transData,
        xyB=(0.0, 0.0),
        coordsB=ax_right.transAxes,
        color=COLOR_DARK_GRAY,
        linewidth=0.7,
        linestyle=(0, (3, 2)),
    )
    fig.add_artist(con_top)
    fig.add_artist(con_bottom)
    return ax_left, ax_right


def _panel_a_y_limits(similarity_df: pd.DataFrame, bridge_df: pd.DataFrame) -> tuple[float, float]:
    sim_df = similarity_df.sort_values("similarity_bin").reset_index(drop=True)
    bridge_order = ["low_overlap", "high_overlap"]
    bridge = bridge_df.set_index("group").reindex(bridge_order).reset_index()
    y_main = 100.0 * sim_df["acc_drop"].to_numpy(dtype=float)
    y_sub = 100.0 * bridge["acc_drop"].to_numpy(dtype=float)
    y_min = min(float(np.min(y_main)), float(np.min(y_sub))) - 4.0
    y_max = max(float(np.max(y_main)), float(np.max(y_sub))) + 5.0
    return y_min, y_max


def draw_panel_a_similarity(ax: plt.Axes, similarity_df: pd.DataFrame, bridge_df: pd.DataFrame) -> None:
    style_axes(ax)
    sim_df = similarity_df.sort_values("similarity_bin").reset_index(drop=True)
    x_main = np.arange(len(sim_df), dtype=float)
    y_main = 100.0 * sim_df["acc_drop"].to_numpy(dtype=float)
    bar_colors = [COLOR_SAMPLE_ONLY, COLOR_SAMPLE_ONLY, COLOR_SAMPLE_ONLY, COLOR_OVERLAP]

    ax.bar(x_main, y_main, width=0.62, color=bar_colors, alpha=0.86, edgecolor="none", zorder=1)
    ax.plot(x_main, y_main, color=COLOR_DARK_GRAY, linewidth=DATA_LINEWIDTH, zorder=3)
    ax.scatter(
        x_main,
        y_main,
        s=[18, 18, 18, 34],
        color=[COLOR_DARK_GRAY, COLOR_DARK_GRAY, COLOR_DARK_GRAY, COLOR_OVERLAP],
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
    )
    for x_val, y_val in zip(x_main, y_main):
        ax.text(x_val, y_val + 1.1, f"{y_val:.1f}", ha="center", va="bottom", fontsize=ANNOTATION_SIZE)
    ax.set_xticks(x_main, ["bin 1", "bin 2", "bin 3", "bin 4"])
    ax.set_ylabel("accuracy drop (%)")
    y_min, y_max = _panel_a_y_limits(similarity_df, bridge_df)
    ax.set_ylim(y_min, y_max)


def draw_panel_b_overlap_bridge(ax: plt.Axes, bridge_df: pd.DataFrame, *, y_limits: tuple[float, float] | None = None) -> None:
    style_axes(ax)
    bridge_order = ["low_overlap", "high_overlap"]
    bridge = bridge_df.set_index("group").reindex(bridge_order).reset_index()
    x_sub = np.arange(len(bridge), dtype=float)
    y_sub = 100.0 * bridge["acc_drop"].to_numpy(dtype=float)
    err_sub = 100.0 * bridge["sem_acc_drop"].fillna(0.0).to_numpy(dtype=float)
    colors = [COLOR_SAMPLE_ONLY, COLOR_OVERLAP]

    ax.bar(x_sub, y_sub, width=0.55, color=colors, alpha=0.86, edgecolor="none", zorder=1)
    ax.errorbar(x_sub, y_sub, yerr=err_sub, fmt="none", ecolor=COLOR_DARK_GRAY, elinewidth=0.8, capsize=0, zorder=2)
    for x_val, y_val in zip(x_sub, y_sub):
        ax.text(x_val, y_val + 1.1, f"{y_val:.1f}", ha="center", va="bottom", fontsize=ANNOTATION_SIZE)
    ax.set_xticks(x_sub, ["low ov.", "high ov."])
    ax.set_ylabel("accuracy drop (%)")
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def draw_panel_b_dpi_bridge(ax: plt.Axes, trace_df: pd.DataFrame, pair_df: pd.DataFrame) -> None:
    style_axes(ax)
    add_reference_line(ax, 0.0)

    trace_df = trace_df.loc[trace_df["time_ms"] <= 50.0].copy()
    color_map = {
        "sample_keep_overlap_only_dynamic": COLOR_OVERLAP,
        "sample_keep_nonoverlap_only_dynamic": COLOR_PROBE_ONLY,
    }
    handles: list[Line2D] = []

    for condition, cond_df in trace_df.groupby("condition", sort=False):
        cond_df = cond_df.sort_values("time_step")
        x = cond_df["time_ms"].to_numpy(dtype=float)
        y = cond_df["dpi_mean"].to_numpy(dtype=float)
        err = cond_df["dpi_sem"].to_numpy(dtype=float)
        color = color_map.get(condition, COLOR_DARK_GRAY)
        label = _pretty_condition(condition)
        ax.plot(x, y, color=color, linewidth=DATA_LINEWIDTH, zorder=3)
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.08, linewidth=0.0, zorder=2)
        handles.append(Line2D([0], [0], color=color, lw=DATA_LINEWIDTH, label=label))

        peak_idx = int(np.nanargmax(y))
        peak_x = float(x[peak_idx])
        peak_y = float(y[peak_idx])
        displayed_peak = float(peak_y)
        ax.scatter([peak_x], [peak_y], s=24, color=color, edgecolor="white", linewidth=0.35, zorder=4)
        ax.annotate(
            f"{displayed_peak:.2f}",
            xy=(peak_x, peak_y),
            xytext=(5, 8 if "overlap_only" in condition else -12),
            textcoords="offset points",
            fontsize=ANNOTATION_SIZE,
            color=color,
            arrowprops={"arrowstyle": "-", "lw": 0.55, "color": color},
        )

    _framed_legend(ax, handles)
    ax.set_xlabel("probe time (ms)")
    ax.set_ylabel("DPI L3")
    ax.set_xlim(0.0, 50.0)


def _draw_horizontal_distribution(
    ax: plt.Axes,
    values_by_label: dict[str, np.ndarray],
    colors_by_label: dict[str, str],
    summary_values: dict[str, float],
    xlabel: str,
    *,
    reference: float | None = None,
    band: tuple[float, float] | None = None,
    seed_offset: int = 0,
) -> None:
    style_axes(ax)
    if band is not None:
        ax.axvspan(float(band[0]), float(band[1]), color="#F4F5F7", zorder=0)
    if reference is not None:
        ax.axvline(float(reference), color=COLOR_DARK_GRAY, linewidth=REF_LINEWIDTH, linestyle=(0, (3, 2)), alpha=0.6, zorder=1)

    labels = ["plus", "minus"]
    y_positions = np.arange(len(labels), dtype=float)
    rng = np.random.default_rng(2026 + seed_offset)

    for idx, label in enumerate(labels):
        values = np.asarray(values_by_label[label], dtype=float)
        mean, ci_low, ci_high = _bootstrap_ci(values, seed=100 + seed_offset + idx)
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        ax.scatter(values, np.full(values.size, y_positions[idx]) + jitter, s=10, color=colors_by_label[label], alpha=0.18, linewidths=0, zorder=2)
        ax.hlines(y_positions[idx], ci_low, ci_high, color=colors_by_label[label], linewidth=2.2, zorder=3)
        ax.scatter([summary_values[label]], [y_positions[idx]], s=40, color=colors_by_label[label], edgecolor="white", linewidth=0.35, zorder=4)
        ax.annotate(
            f"{float(summary_values[label]):.2f}",
            xy=(float(summary_values[label]), y_positions[idx]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
        )

    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)


def _add_pill_lane(
    ax: plt.Axes,
    y: float,
    *,
    x0: float = 0.0,
    width: float = 1.0,
    height: float = 0.56,
    facecolor: str = "#F7F8FA",
    edgecolor: str = "none",
    linewidth: float = 0.0,
) -> Rectangle:
    lane = Rectangle(
        (x0, y - height / 2.0),
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transData,
        clip_on=False,
        zorder=0.2,
    )
    ax.add_patch(lane)
    return lane


def _draw_cosine_row(
    ax: plt.Axes,
    values: np.ndarray,
    mean_x: float,
    color: str,
    *,
    y: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    lane = _add_pill_lane(ax, y, facecolor="#F7F8FA")
    high_zone = Rectangle((0.8, y - 0.28), 0.2, 0.56, facecolor="#F1F6F4", edgecolor="none", zorder=0.25)
    high_zone.set_clip_path(lane)
    ax.add_patch(high_zone)

    values = np.asarray(values, dtype=float)
    _, ci_low, ci_high = _bootstrap_ci(values, seed=seed + 100)
    jitter = rng.uniform(-0.12, 0.12, size=values.size)
    ax.scatter(
        values,
        np.full(values.size, y) + jitter,
        s=12,
        color=color,
        alpha=0.16,
        linewidths=0,
        zorder=2,
    )
    ax.hlines(y, ci_low, ci_high, color=color, linewidth=2.35, zorder=3)
    ax.scatter([mean_x], [y], s=64, color=color, edgecolor="white", linewidth=0.6, zorder=4)
    ax.annotate(
        f"{mean_x:.2f}",
        xy=(mean_x, y),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_SIZE,
        color=COLOR_TEXT,
        zorder=5,
    )


def _draw_argmax_row(
    ax: plt.Axes,
    rate: float,
    color: str,
    *,
    y: float,
) -> None:
    bar_height = 0.26
    ax.barh([y], [1.0], left=0.0, height=0.36, color="#EEF1F4", edgecolor="none", zorder=1)
    ax.barh([y], [rate], left=0.0, height=bar_height, color=color, edgecolor="none", alpha=0.9, zorder=2)
    label_x = min(rate + 0.015, 0.965)
    ax.text(
        label_x,
        y,
        f"{rate:.2f}",
        ha="left",
        va="center",
        fontsize=ANNOTATION_SIZE,
        color=COLOR_TEXT,
        zorder=4,
    )


def draw_panel_c_stacked_metric_rows(
    ax: plt.Axes,
    cosine_summary_df: pd.DataFrame,
    direction_summary_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> plt.Axes:
    style_axes(ax)
    cosine_summary = cosine_summary_df.set_index("mode")
    direction_summary = direction_summary_df.set_index("mode")
    colors = {"plus": COLOR_DYNAMIC, "minus": COLOR_OVERLAP}
    row_y = {"plus_cos": 3.0, "plus_dir": 2.0, "minus_cos": 1.0, "minus_dir": 0.0}

    _draw_cosine_row(
        ax,
        pair_df["reconstruction_cosine_plus"].to_numpy(dtype=float),
        float(cosine_summary.loc["plus", "mean_cosine"]),
        colors["plus"],
        y=row_y["plus_cos"],
        seed=2026,
    )
    _draw_argmax_row(
        ax,
        float(direction_summary.loc["plus", "direction_match_rate"]),
        colors["plus"],
        y=row_y["plus_dir"],
    )
    _draw_cosine_row(
        ax,
        pair_df["reconstruction_cosine_minus"].to_numpy(dtype=float),
        float(cosine_summary.loc["minus", "mean_cosine"]),
        colors["minus"],
        y=row_y["minus_cos"],
        seed=2044,
    )
    _draw_argmax_row(
        ax,
        float(direction_summary.loc["minus", "direction_match_rate"]),
        colors["minus"],
        y=row_y["minus_dir"],
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.55, 3.55)
    ax.set_yticks([0.5, 2.5], ["minus", "plus"])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("recovery score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def build_panel_figures(root: str | Path) -> tuple[dict[str, plt.Figure], dict[str, object]]:
    apply_paper_style()
    bundle = load_fig3_bundle(root)
    figures: dict[str, plt.Figure] = {}
    panel_a_limits = _panel_a_y_limits(bundle["panel_a_similarity"], bundle["panel_a_bridge"])

    fig_a = plt.figure(figsize=(4.2, 2.45))
    ax_a = fig_a.add_subplot(1, 1, 1)
    draw_panel_a_similarity(ax_a, bundle["panel_a_similarity"], bundle["panel_a_bridge"])
    figures["panel_a"] = fig_a

    fig_b = plt.figure(figsize=(2.45, 2.45))
    ax_b = fig_b.add_subplot(1, 1, 1)
    draw_panel_b_overlap_bridge(ax_b, bundle["panel_a_bridge"], y_limits=panel_a_limits)
    figures["panel_b"] = fig_b

    fig_c = plt.figure(figsize=(5.9, 3.0))
    ax_c = fig_c.add_subplot(1, 1, 1)
    draw_panel_b_dpi_bridge(ax_c, bundle["panel_b_trace"], bundle["panel_b_pair"])
    figures["panel_c"] = fig_c

    fig_d = plt.figure(figsize=(5.6, 2.8))
    ax_d = fig_d.add_subplot(1, 1, 1)
    d_ax = draw_panel_c_stacked_metric_rows(
        ax_d,
        bundle["panel_c_summary"],
        bundle["panel_d_summary"],
        bundle["panel_cd_pair_metrics"],
    )
    figures["panel_d"] = fig_d

    return figures, bundle


def build_assembled_figure(root: str | Path) -> tuple[plt.Figure, dict[str, object]]:
    apply_paper_style()
    bundle = load_fig3_bundle(root)
    fig = plt.figure(figsize=(7.35, 5.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.95, 1.45], hspace=0.42)
    top = outer[0, 0].subgridspec(1, 2, width_ratios=[3.15, 1.25], wspace=0.42)
    bottom = outer[1, 0].subgridspec(1, 2, width_ratios=[1.65, 1.35], wspace=0.30)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])
    panel_a_limits = _panel_a_y_limits(bundle["panel_a_similarity"], bundle["panel_a_bridge"])

    draw_panel_a_similarity(ax_a, bundle["panel_a_similarity"], bundle["panel_a_bridge"])
    draw_panel_b_overlap_bridge(ax_b, bundle["panel_a_bridge"], y_limits=panel_a_limits)
    draw_panel_b_dpi_bridge(ax_c, bundle["panel_b_trace"], bundle["panel_b_pair"])
    d_ax = draw_panel_c_stacked_metric_rows(
        ax_d,
        bundle["panel_c_summary"],
        bundle["panel_d_summary"],
        bundle["panel_cd_pair_metrics"],
    )
    return fig, bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render final Fig3 from paper_figs result tables.")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--export-panels", action="store_true")
    args = parser.parse_args(argv)

    input_dir = resolve_figure_input_dir("fig3", args.input_dir)
    output_dir = Path(input_dir) / "plots" if args.output_dir is None else Path(args.output_dir)

    fig, bundle = build_assembled_figure(input_dir)
    saved: dict[str, object] = {"figure": save_figure_outputs(fig, output_dir, "fig3")}
    plt.close(fig)

    if args.export_panels:
        panel_figures, _ = build_panel_figures(input_dir)
        panel_saved: dict[str, dict[str, str]] = {}
        for panel_name, panel_fig in panel_figures.items():
            panel_saved[panel_name] = save_figure_outputs(panel_fig, output_dir, f"fig3_{panel_name}")
            plt.close(panel_fig)
        saved["panels"] = panel_saved

    print(
        json.dumps(
            {
                "status": "ok",
                "figure": "fig3",
                "input_dir": str(input_dir),
                "summary_keys": sorted(bundle["summary"].keys()),
                "saved": saved,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
