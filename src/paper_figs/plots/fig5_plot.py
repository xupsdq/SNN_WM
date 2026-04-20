from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.paper_figs.plots.common import load_npz, read_csv_validated, require_path, resolve_figure_input_dir
from src.paper_figs.plots.style import (
    COLOR_DARK_GRAY,
    COLOR_DYNAMIC,
    COLOR_LIGHT_GRAY,
    COLOR_OVERLAP,
    COLOR_PROBE_ONLY,
    COLOR_SAMPLE_ONLY,
    COLOR_TEXT,
    DATA_LINEWIDTH,
    add_reference_line,
    apply_paper_style,
    legend_outside,
    save_figure_outputs,
    style_axes,
)


def load_fig5_bundle(root: str | Path) -> dict[str, object]:
    root_path = Path(root)
    required_new_files = [
        root_path / "data" / "panel_c_sample_induced_rewriting_timeseries.csv",
        root_path / "data" / "panel_c_sample_induced_rewriting_summary.csv",
        root_path / "data" / "panel_e_formation_intervention_comparison.csv",
    ]
    missing_new_files = [path for path in required_new_files if not path.exists()]
    if missing_new_files:
        missing_text = "\n".join(str(path) for path in missing_new_files)
        raise FileNotFoundError(
            "Fig5 plotting now expects the regenerated fusion/rewriting result package.\n"
            "The current input directory does not contain the new Panel C/E artifacts:\n"
            f"{missing_text}\n"
            "Re-run:\n"
            "python -m src.paper_figs.experiments.fig5_experiment\n"
            "Then run:\n"
            "python -m src.paper_figs.plots.fig5_plot"
        )
    return {
        "summary": json.loads(require_path(root_path / "summary.json").read_text(encoding="utf-8")),
        "panel_a_triplets": read_csv_validated(
            root_path / "data" / "panel_a_triplet_definition.csv",
            ["triplet_id", "sample_id", "distractor_id", "probe_id"],
        ),
        "panel_b_fusion": read_csv_validated(
            root_path / "data" / "panel_b_preprobe_fusion_metrics.csv",
            [
                "triplet_id",
                "sim_to_sample_L3",
                "sim_to_distractor_L3",
                "fusion_dual_score_L3",
                "fusion_imbalance_L3",
            ],
        ),
        "panel_b_specificity": read_csv_validated(
            root_path / "data" / "panel_b_fusion_specificity.csv",
            [
                "triplet_id",
                "true_pair_percentile_L3",
                "true_pair_z_L3",
                "true_pair_top1_L3",
            ],
        ),
        "panel_c_rewriting_timeseries": read_csv_validated(
            root_path / "data" / "panel_c_sample_induced_rewriting_timeseries.csv",
            [
                "triplet_id",
                "layer",
                "distractor_step",
                "rewriting_t",
                "sim_removed_to_donly_t",
                "sim_intact_to_donly_t",
            ],
        ),
        "panel_c_rewriting_summary": read_csv_validated(
            root_path / "data" / "panel_c_sample_induced_rewriting_summary.csv",
            [
                "triplet_id",
                "rewrite_mean_L3",
                "rewrite_peak_L3",
                "rewrite_auc_L3",
                "rewrite_early_L3",
                "rewrite_early_L3_intact",
                "rewrite_early_L3_removed",
                "delta_rewrite_early_L3",
            ],
        ),
        "panel_c_pull_summary": read_csv_validated(
            root_path / "data" / "panel_c_distractor_pull_summary.csv",
            ["triplet_id", "barP_L3", "peakP_L3", "earlyP_L3"],
        ),
        "panel_d_bridge": read_csv_validated(
            root_path / "data" / "panel_d_rewriting_to_fusion_bridge.csv",
            ["table", "analysis"],
        ),
        "panel_e_intervention": read_csv_validated(
            root_path / "data" / "panel_e_formation_intervention_comparison.csv",
            [
                "triplet_id",
                "rewrite_early_L3_intact",
                "rewrite_early_L3_removed",
                "fusion_dual_score_L3",
                "formation_fusion_dual_score_L3",
                "true_pair_z_L3",
                "formation_true_pair_z_L3",
                "delta_rewrite_early_L3",
                "delta_barP_L3",
                "delta_fusion_dual_score_L3",
                "delta_true_pair_z_L3",
            ],
        ),
        "panel_a_arrays": load_npz(root_path / "arrays" / "panel_a_example_regions.npz"),
        "panel_b_arrays": load_npz(root_path / "arrays" / "panel_b_example_preprobe_fusion_state.npz"),
        "panel_c_arrays": load_npz(root_path / "arrays" / "panel_c_example_distractor_pull_trace.npz"),
    }


def _sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _finite_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    return xx[mask], yy[mask]


def _paired_values(df: pd.DataFrame, left_col: str, right_col: str) -> tuple[np.ndarray, np.ndarray]:
    left = df[left_col].to_numpy(dtype=float)
    right = df[right_col].to_numpy(dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def _draw_paired_comparison(
    ax: plt.Axes,
    *,
    left_values: np.ndarray,
    right_values: np.ndarray,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
    ylabel: str,
    title: str,
) -> None:
    style_axes(ax, show_grid_y=True)
    x_positions = np.asarray([0.0, 1.0], dtype=float)
    for left_value, right_value in zip(left_values.tolist(), right_values.tolist()):
        ax.plot(x_positions, [left_value, right_value], color=COLOR_LIGHT_GRAY, linewidth=0.8, alpha=0.55, zorder=1)
    ax.scatter(
        np.full(left_values.shape, x_positions[0], dtype=float),
        left_values,
        s=16,
        color=left_color,
        alpha=0.55,
        zorder=2,
    )
    ax.scatter(
        np.full(right_values.shape, x_positions[1], dtype=float),
        right_values,
        s=16,
        color=right_color,
        alpha=0.65,
        zorder=2,
    )
    ax.errorbar(
        [x_positions[0]],
        [float(np.nanmean(left_values)) if left_values.size else float("nan")],
        yerr=[_sem(left_values)],
        fmt="o",
        markersize=6.8,
        linewidth=1.3,
        capsize=3.0,
        color=left_color,
        zorder=4,
    )
    ax.errorbar(
        [x_positions[1]],
        [float(np.nanmean(right_values)) if right_values.size else float("nan")],
        yerr=[_sem(right_values)],
        fmt="o",
        markersize=6.8,
        linewidth=1.3,
        capsize=3.0,
        color=right_color,
        zorder=4,
    )
    ax.set_xticks(x_positions, [left_label, right_label])
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _region_overlay(mask_arrays: dict[str, np.ndarray]) -> np.ndarray:
    sample_only = np.asarray(mask_arrays["sample_only_mask"], dtype=float)
    distractor_only = np.asarray(mask_arrays["distractor_only_mask"], dtype=float)
    shared = np.asarray(mask_arrays["shared_mask"], dtype=float)
    canvas = np.ones(sample_only.shape + (3,), dtype=float)
    sample_rgb = np.array([0x4C, 0x56, 0x6A], dtype=float) / 255.0
    distractor_rgb = np.array([0xCC, 0x79, 0xA7], dtype=float) / 255.0
    shared_rgb = np.array([0x00, 0x9E, 0x73], dtype=float) / 255.0
    canvas[sample_only > 0] = sample_rgb
    canvas[distractor_only > 0] = distractor_rgb
    canvas[shared > 0] = shared_rgb
    return canvas


def _draw_panel_a(ax: plt.Axes, bundle: dict[str, object]) -> None:
    style_axes(ax)
    arrays = bundle["panel_a_arrays"]
    ax.imshow(_region_overlay(arrays), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Triplet Definition")
    ax.text(0.02, -0.1, "sample-only", color=COLOR_SAMPLE_ONLY, transform=ax.transAxes)
    ax.text(0.40, -0.1, "shared", color=COLOR_OVERLAP, transform=ax.transAxes)
    ax.text(0.66, -0.1, "distractor-only", color=COLOR_PROBE_ONLY, transform=ax.transAxes)


def _draw_panel_b1(ax: plt.Axes, bundle: dict[str, object]) -> None:
    fusion_df = bundle["panel_b_fusion"]
    style_axes(ax, show_grid_x=True, show_grid_y=True)
    xx, yy = _finite_pair(fusion_df["sim_to_sample_L3"], fusion_df["sim_to_distractor_L3"])
    ax.scatter(xx, yy, s=18, color=COLOR_DYNAMIC, alpha=0.72)
    low = min(float(np.nanmin(xx)) if xx.size else -1.0, float(np.nanmin(yy)) if yy.size else -1.0)
    high = max(float(np.nanmax(xx)) if xx.size else 1.0, float(np.nanmax(yy)) if yy.size else 1.0)
    ax.plot([low, high], [low, high], linestyle="--", linewidth=0.8, color=COLOR_TEXT)
    ax.set_xlabel("sim_to_sample_L3")
    ax.set_ylabel("sim_to_distractor_L3")
    ax.set_title("Fusion Form")


def _draw_panel_b2(ax: plt.Axes, bundle: dict[str, object]) -> None:
    spec_df = bundle["panel_b_specificity"]
    style_axes(ax, show_grid_y=True)
    percentiles = spec_df["true_pair_percentile_L3"].to_numpy(dtype=float)
    z_scores = spec_df["true_pair_z_L3"].to_numpy(dtype=float)
    ax.hist(
        percentiles[np.isfinite(percentiles)],
        bins=np.linspace(0.0, 1.0, 13),
        color=COLOR_OVERLAP,
        alpha=0.75,
        label="percentile",
    )
    twin = ax.twiny()
    twin.hist(
        z_scores[np.isfinite(z_scores)],
        bins=12,
        histtype="step",
        color=COLOR_PROBE_ONLY,
        linewidth=1.2,
        label="z-score",
    )
    ax.set_xlabel("true_pair_percentile_L3")
    twin.set_xlabel("true_pair_z_L3")
    ax.set_ylabel("count")
    ax.set_title("Fusion Specificity")


def _draw_panel_b(fig: plt.Figure, bundle: dict[str, object]) -> None:
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    _draw_panel_b1(ax1, bundle)
    _draw_panel_b2(ax2, bundle)


def _draw_panel_c_axis(ax: plt.Axes, bundle: dict[str, object]) -> None:
    rewriting_ts = bundle["panel_c_rewriting_timeseries"]
    style_axes(ax, show_grid_y=True)
    layer_df = rewriting_ts[rewriting_ts["layer"] == "layer3"].copy()
    mean_df = layer_df.groupby("distractor_step", as_index=False)["rewriting_t"].mean()
    sem_df = layer_df.groupby("distractor_step", as_index=False)["rewriting_t"].agg(_sem).rename(columns={"rewriting_t": "sem"})
    merged = mean_df.merge(sem_df, on="distractor_step", how="left")
    x = merged["distractor_step"].to_numpy(dtype=float)
    y = merged["rewriting_t"].to_numpy(dtype=float)
    sem = merged["sem"].to_numpy(dtype=float)
    ax.plot(x, y, linewidth=DATA_LINEWIDTH, color=COLOR_PROBE_ONLY)
    ax.fill_between(x, y - sem, y + sem, color=COLOR_PROBE_ONLY, alpha=0.18)
    add_reference_line(ax, 0.0)
    ax.set_xlabel("Distractor step")
    ax.set_ylabel("Sample-driven rewriting index")
    ax.set_title("Sample-driven rewriting of L3 distractor activity")


def _draw_panel_c(fig: plt.Figure, bundle: dict[str, object]) -> None:
    ax = fig.add_subplot(1, 1, 1)
    _draw_panel_c_axis(ax, bundle)


def _draw_supplement_bridge_panel(fig: plt.Figure, bundle: dict[str, object]) -> None:
    bridge_df = bundle["panel_d_bridge"]
    pull_summary = bundle["panel_c_pull_summary"]
    fusion_df = bundle["panel_b_fusion"]
    spec_df = bundle["panel_b_specificity"]
    merged = pull_summary.merge(fusion_df, on="triplet_id", how="inner").merge(spec_df, on="triplet_id", how="inner")
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    style_axes(ax1, show_grid_x=True, show_grid_y=True)
    style_axes(ax2, show_grid_x=True, show_grid_y=True)
    for ax, y_col, color, title in (
        (ax1, "fusion_dual_score_L3", COLOR_DYNAMIC, "barP_L3 vs fusion_dual_score_L3"),
        (ax2, "true_pair_z_L3", COLOR_OVERLAP, "barP_L3 vs true_pair_z_L3"),
    ):
        xx, yy = _finite_pair(merged["barP_L3"], merged[y_col])
        ax.scatter(xx, yy, s=18, color=color, alpha=0.72)
        if xx.size >= 2 and float(np.std(xx)) > 1e-12 and float(np.std(yy)) > 1e-12:
            fit = np.polyfit(xx, yy, deg=1)
            x_line = np.linspace(float(xx.min()), float(xx.max()), 100)
            ax.plot(x_line, fit[0] * x_line + fit[1], color=COLOR_TEXT, linewidth=0.8)
        ax.set_xlabel("barP_L3")
        ax.set_ylabel(y_col)
        ax.set_title(title)
    corr_subset = bridge_df[(bridge_df["analysis"] == "correlation") & (bridge_df["x"] == "barP_L3")]
    if not corr_subset.empty:
        row = corr_subset.iloc[0]
        ax1.text(0.04, 0.96, f"r={row['pearson_r']:.2f}" if pd.notna(row["pearson_r"]) else "r=nan", transform=ax1.transAxes, va="top")


def _draw_panel_d_axes(axes: list[plt.Axes], bundle: dict[str, object]) -> None:
    intervention_df = bundle["panel_e_intervention"]
    metrics = [
        (
            "rewrite_early_L3_intact",
            "rewrite_early_L3_removed",
            "E1. Removing sample trace weakens distractor-period rewriting",
            "Early sample-driven rewriting",
            COLOR_PROBE_ONLY,
        ),
        (
            "fusion_dual_score_L3",
            "formation_fusion_dual_score_L3",
            "E2. Removing sample trace weakens the fused pre-probe state",
            "Similarity to both sample and distractor memories",
            COLOR_DYNAMIC,
        ),
        (
            "true_pair_z_L3",
            "formation_true_pair_z_L3",
            "E3. Removing sample trace weakens true-pair specificity",
            "Specificity for the correct sample-distractor pairing (z-score)",
            COLOR_OVERLAP,
        ),
    ]
    for ax, (left_col, right_col, title, ylabel, color) in zip(axes, metrics):
        left_values, right_values = _paired_values(intervention_df, left_col, right_col)
        _draw_paired_comparison(
            ax,
            left_values=left_values,
            right_values=right_values,
            left_label="intact",
            right_label="sample-trace removed",
            left_color=COLOR_DARK_GRAY,
            right_color=color,
            ylabel=ylabel,
            title=title,
        )


def _draw_panel_d(fig: plt.Figure, bundle: dict[str, object]) -> None:
    subgrid = fig.add_gridspec(1, 3, wspace=0.42)
    axes = [fig.add_subplot(subgrid[0, idx]) for idx in range(3)]
    _draw_panel_d_axes(axes, bundle)
    fig.suptitle("Removing the sample trace weakens rewriting and fused memory", y=1.02)


def build_figure(root: str | Path) -> tuple[plt.Figure, dict[str, object]]:
    apply_paper_style()
    bundle = load_fig5_bundle(root)
    fig = plt.figure(figsize=(11.2, 11.4))
    grid = fig.add_gridspec(3, 2, hspace=0.72, wspace=0.5, height_ratios=[1.0, 1.0, 1.15])

    ax_a = fig.add_subplot(grid[0, :])
    _draw_panel_a(ax_a, bundle)

    ax_b1 = fig.add_subplot(grid[1, 0])
    _draw_panel_b1(ax_b1, bundle)

    ax_b2 = fig.add_subplot(grid[1, 1])
    _draw_panel_b2(ax_b2, bundle)

    ax_c = fig.add_subplot(grid[2, 0])
    _draw_panel_c_axis(ax_c, bundle)

    d_grid = grid[2, 1].subgridspec(1, 3, wspace=0.42)
    d_axes = [fig.add_subplot(d_grid[0, idx]) for idx in range(3)]
    _draw_panel_d_axes(d_axes, bundle)
    return fig, bundle


def build_panel_figures(root: str | Path) -> tuple[dict[str, plt.Figure], dict[str, object]]:
    apply_paper_style()
    bundle = load_fig5_bundle(root)
    figures: dict[str, plt.Figure] = {}

    fig_a = plt.figure(figsize=(4.6, 2.9))
    ax_a = fig_a.add_subplot(1, 1, 1)
    _draw_panel_a(ax_a, bundle)
    figures["panel_a"] = fig_a

    fig_b1 = plt.figure(figsize=(3.4, 2.8))
    ax_b1 = fig_b1.add_subplot(1, 1, 1)
    _draw_panel_b1(ax_b1, bundle)
    figures["panel_b1"] = fig_b1

    fig_b2 = plt.figure(figsize=(3.4, 2.8))
    ax_b2 = fig_b2.add_subplot(1, 1, 1)
    _draw_panel_b2(ax_b2, bundle)
    figures["panel_b2"] = fig_b2

    fig_c = plt.figure(figsize=(4.2, 3.0))
    _draw_panel_c(fig_c, bundle)
    figures["panel_c"] = fig_c

    fig_d = plt.figure(figsize=(8.8, 3.2))
    _draw_panel_d(fig_d, bundle)
    figures["panel_d"] = fig_d

    fig_supp_d = plt.figure(figsize=(7.0, 2.8))
    _draw_supplement_bridge_panel(fig_supp_d, bundle)
    figures["supp_panel_d_bridge"] = fig_supp_d

    return figures, bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render Fig5 from paper_figs result tables.")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--full-figure", action="store_true")
    parser.add_argument("--panels-only", action="store_true")
    args = parser.parse_args(argv)

    input_dir = resolve_figure_input_dir("fig5", args.input_dir)
    output_dir = Path(input_dir) / "plots" if args.output_dir is None else Path(args.output_dir)
    saved: dict[str, dict[str, str]] = {}
    render_full_figure = True
    if bool(args.panels_only):
        render_full_figure = False

    if render_full_figure:
        fig, bundle = build_figure(input_dir)
        saved["main"] = save_figure_outputs(fig, output_dir, "fig5_main")
        plt.close(fig)
        if bool(args.full_figure):
            pass
    else:
        figures, bundle = build_panel_figures(input_dir)
        for panel_name, fig in figures.items():
            saved[panel_name] = save_figure_outputs(fig, output_dir, f"fig5_{panel_name}")
            plt.close(fig)

    print(
        json.dumps(
            {
                "status": "ok",
                "figure": "fig5",
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "render_mode": "full_figure" if render_full_figure else "panels_only",
                "summary_keys": sorted(bundle["summary"].keys()),
                "saved": saved,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
