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
from matplotlib.patches import Patch, Rectangle

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
    COLOR_STATIC,
    COLOR_TEXT,
    DATA_LINEWIDTH,
    REF_LINEWIDTH,
    add_reference_line,
    apply_paper_style,
    save_figure_outputs,
    style_axes,
)

GROUP_DISPLAY_LABELS = {
    "all_units": "all receiving\nunits",
    "overlap_dominant": "overlap-\nbiased",
    "probe_only_dominant": "probe-only-\nbiased",
    "probe_dominant": "probe-only-\nbiased",
}

GROUP_DISPLAY_LABELS_SHORT = {
    "overlap_dominant": "overlap-biased",
    "probe_only_dominant": "probe-only-biased",
}


def load_fig4_bundle(root: str | Path) -> dict[str, object]:
    root_path = Path(root)
    panel_f_trace_npz = root_path / "arrays" / "panel_f_local_competition_trace.npz"
    return {
        "summary": json.loads(require_path(root_path / "summary.json").read_text(encoding="utf-8")),
        "panel_a_trial_definition": read_csv_validated(
            root_path / "data" / "panel_a_trial_definition.csv",
            ["trial_id", "overlap_area", "probe_only_area"],
        ),
        "panel_b_preprobe": read_csv_validated(
            root_path / "data" / "panel_b_preprobe_support.csv",
            ["trial_id", "model_type", "ux_overlap_pre", "ux_probe_only_pre"],
        ),
        "panel_b_transition": read_csv_validated(
            root_path / "data" / "panel_c_transition_summary.csv",
            ["trial_id", "unit_group", "n_units", "n_advance", "n_recruit", "n_loss", "n_unchanged", "P_advance", "P_recruit", "P_loss", "P_unchanged"],
        ),
        "panel_b_changed": read_csv_validated(
            root_path / "data" / "panel_b_changed_only_composition.csv",
            [
                "trial_id",
                "unit_group",
                "n_units",
                "n_advance",
                "n_recruit",
                "n_loss",
                "n_unchanged",
                "changed_count",
                "changed_prevalence",
                "P_changed_advance",
                "P_changed_recruit",
                "P_changed_loss",
            ],
        ),
        "panel_c_event_time": load_npz(root_path / "arrays" / "panel_c_event_time_alignment.npz"),
        "panel_d_chain": read_csv_validated(
            root_path / "data" / "panel_d_causal_chain_events.csv",
            [
                "trial_id",
                "winner_pre_spike_boost",
                "winner_spikes_earlier",
                "loser_post_winner_suppressed",
                "full_chain_satisfied",
            ],
        ),
        "panel_d_input_gain": read_csv_validated(
            root_path / "data" / "panel_d_input_gain_summary.csv",
            ["trial_id", "unit_group", "overlap_input_gain", "probe_only_input_gain", "input_selectivity_gain"],
        ),
        "panel_e_loss_inh": read_csv_validated(
            root_path / "data" / "panel_e_loss_inhibition_summary.csv",
            ["trial_id", "unit_group", "lost_spike_delta_inh"],
        ),
        "panel_f_pairs": read_csv_validated(
            root_path / "data" / "panel_f_local_winner_loser_pairs.csv",
            ["trial_id", "winner_group", "winner_overlap_input_gain", "winner_loser_contrast_shift", "contrast_dynamic", "contrast_static"],
        ),
        "panel_g_support": read_csv_validated(
            root_path / "data" / "panel_g_local_winner_support_summary.csv",
            ["trial_id", "local_winner_support_rate"],
        ),
        "panel_a_arrays": load_npz(root_path / "arrays" / "panel_a_example_masks.npz"),
        "panel_f_arrays": load_npz(root_path / "arrays" / "panel_f_local_competition_exemplar.npz"),
        "panel_f_trace": load_npz(panel_f_trace_npz) if panel_f_trace_npz.exists() else None,
    }


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
        handlelength=1.5,
    )


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 1000) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        draws[idx] = float(np.mean(rng.choice(vals, size=vals.size, replace=True)))
    return float(np.mean(vals)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _nanmean_sem(arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arrays, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array for aligned event-time summary.")
    mean = np.nanmean(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0).astype(float)
    std = np.nanstd(arr, axis=0, ddof=1)
    sem = np.divide(std, np.sqrt(n), out=np.zeros_like(std), where=n > 1.0)
    sem[n <= 1.0] = 0.0
    return mean, sem


def _group_label(name: str) -> str:
    return GROUP_DISPLAY_LABELS.get(name, name.replace("_", "\n"))


def draw_panel_a_definition(fig: plt.Figure, spec, mask_arrays: dict[str, np.ndarray]) -> plt.Axes:
    host = fig.add_subplot(spec)
    host.set_axis_off()
    left_margin = 0.02
    image_w = 0.22
    image_h = 0.68
    bottom = 0.12
    gap = 0.035
    legend_w = 0.27
    legend_h = 0.14
    legend_y = bottom + image_h + 0.04
    cbar_w = 0.024
    cbar_x = left_margin + 3.0 * image_w + 3.0 * gap
    ax_sample = host.inset_axes([left_margin, bottom, image_w, image_h])
    ax_probe = host.inset_axes([left_margin + image_w + gap, bottom, image_w, image_h])
    ax_heat = host.inset_axes([left_margin + 2.0 * (image_w + gap), bottom, image_w, image_h])
    ax_legend = host.inset_axes([cbar_x - 0.11, legend_y, legend_w, legend_h])
    ax_cbar = host.inset_axes([cbar_x, bottom, cbar_w, image_h])

    for ax, label, key in (
        (ax_sample, "sample", "sample_mask"),
        (ax_probe, "probe", "probe_mask"),
    ):
        ax.imshow(np.asarray(mask_arrays[key], dtype=float), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor(COLOR_TEXT)
        ax.set_title(label, fontsize=ANNOTATION_SIZE + 0.15, pad=2.0)
    heat_key = "ux_map_pre_dynamic" if "ux_map_pre_dynamic" in mask_arrays else "ux_map_pre_static"
    heatmap = np.asarray(mask_arrays.get(heat_key, np.zeros_like(mask_arrays["sample_mask"], dtype=float)), dtype=float)
    overlap_mask = np.asarray(mask_arrays["overlap_mask"], dtype=float)
    probe_only_mask = np.asarray(mask_arrays["probe_only_mask"], dtype=float)
    im = ax_heat.imshow(heatmap, cmap="magma", interpolation="nearest")
    ax_heat.contour(overlap_mask, levels=[0.5], colors=[COLOR_OVERLAP], linewidths=1.25)
    ax_heat.contour(probe_only_mask, levels=[0.5], colors=[COLOR_PROBE_ONLY], linewidths=1.25)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    ax_heat.set_title("pre-probe u*x", fontsize=ANNOTATION_SIZE + 0.15, pad=2.0)
    overlap_mean = float(np.mean(heatmap[overlap_mask.astype(bool)])) if bool(overlap_mask.astype(bool).any()) else float("nan")
    probe_only_mean = float(np.mean(heatmap[probe_only_mask.astype(bool)])) if bool(probe_only_mask.astype(bool).any()) else float("nan")
    ax_legend.set_axis_off()
    legend_box = Rectangle(
        (0.02, 0.06),
        0.96,
        0.88,
        fill=False,
        edgecolor=COLOR_DARK_GRAY,
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        transform=ax_legend.transAxes,
        clip_on=False,
    )
    ax_legend.add_patch(legend_box)
    legend_entries = [
        (COLOR_OVERLAP, f"overlap  {overlap_mean:.3f}"),
        (COLOR_PROBE_ONLY, f"probe-only  {probe_only_mean:.3f}"),
    ]
    for idx, (color, label) in enumerate(legend_entries):
        y = 0.69 - 0.31 * idx
        ax_legend.plot([0.08, 0.22], [y, y], color=color, linewidth=1.5, transform=ax_legend.transAxes, solid_capstyle="round")
        ax_legend.text(0.27, y, label, transform=ax_legend.transAxes, ha="left", va="center", fontsize=ANNOTATION_SIZE, color=COLOR_TEXT)
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("u*x")
    return ax_sample


def _draw_panel_b_donut(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    order: list[str],
    colors: dict[str, str],
    count_key_by_name: dict[str, str],
    center_text: str,
    prevalence_lines: list[str] | None = None,
) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    work_df = df.copy()
    if "aggregation_scope" in work_df.columns:
        work_df = work_df[work_df["aggregation_scope"] == "per_trial"].copy()
    handles = [Patch(facecolor=colors[name], edgecolor="none", label=name) for name in order]
    startangle = 90.0
    groups = [
        ("overlap_dominant", -0.72, 0.0, 0.43, 0.19, "overlap-\nbiased"),
        ("probe_only_dominant", 0.72, 0.0, 0.43, 0.19, "probe-only\nbiased"),
    ]
    for idx, (group, cx, cy, radius, width, label) in enumerate(groups):
        sub = work_df[work_df["unit_group"] == group]
        counts = np.asarray([float(sub[count_key_by_name[name]].sum()) for name in order], dtype=float)
        if not np.isfinite(counts).all() or float(counts.sum()) <= 0.0:
            counts = np.asarray([1.0] + [0.0] * (len(order) - 1), dtype=float)
        total = float(counts.sum())
        wedges, _ = ax.pie(
            counts,
            radius=radius,
            center=(cx, cy),
            startangle=startangle,
            counterclock=False,
            colors=[colors[name] for name in order],
            wedgeprops={"width": width, "edgecolor": "white", "linewidth": 0.9},
        )
        for wedge, count in zip(wedges, counts):
            frac = count / total if total > 0.0 else 0.0
            if frac < 0.12:
                continue
            theta = np.deg2rad(0.5 * (wedge.theta1 + wedge.theta2))
            r = radius - 0.50 * width
            ax.text(cx + r * np.cos(theta), cy + r * np.sin(theta), f"{frac * 100.0:.0f}%", ha="center", va="center", fontsize=ANNOTATION_SIZE, color=COLOR_TEXT)
        ax.text(cx, cy + 0.01, label, ha="center", va="center", fontsize=ANNOTATION_SIZE - 0.05, linespacing=0.95)
        if prevalence_lines and idx < len(prevalence_lines):
            ax.text(cx, cy - 0.53, prevalence_lines[idx], ha="center", va="top", fontsize=ANNOTATION_SIZE, color=COLOR_DARK_GRAY)
    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-0.84, 0.76)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=len(order), frameon=False, columnspacing=1.0, handletextpad=0.35)


def draw_panel_b_transition_composition(ax: plt.Axes, transition_changed_df: pd.DataFrame) -> None:
    order = ["advance", "recruit", "loss"]
    colors = {
        "advance": COLOR_OVERLAP,
        "recruit": "#24B2A7",
        "loss": COLOR_PROBE_ONLY,
    }
    prevalence_lines: list[str] = []
    for group in ("overlap_dominant", "probe_only_dominant"):
        sub = transition_changed_df[transition_changed_df["unit_group"] == group]
        n_units = float(pd.to_numeric(sub["n_units"], errors="coerce").sum())
        changed_count = float(pd.to_numeric(sub["changed_count"], errors="coerce").sum())
        changed_prev = changed_count / n_units if n_units > 0.0 else float("nan")
        prevalence_lines.append(f"{100.0 * changed_prev:.1f}%")
    _draw_panel_b_donut(
        ax,
        transition_changed_df,
        order=order,
        colors=colors,
        count_key_by_name={name: f"n_{name}" for name in order},
        center_text="",
        prevalence_lines=prevalence_lines,
    )


def draw_panel_b_single_group(ax: plt.Axes, transition_changed_df: pd.DataFrame, group: str, *, show_legend: bool = True) -> None:
    order = ["advance", "recruit", "loss"]
    colors = {
        "advance": COLOR_OVERLAP,
        "recruit": "#24B2A7",
        "loss": COLOR_PROBE_ONLY,
    }
    display_label = {
        "overlap_dominant": "overlap-\nbiased",
        "probe_only_dominant": "probe-only\nbiased",
    }.get(group, group.replace("_", "\n"))
    sub = transition_changed_df[transition_changed_df["unit_group"] == group].copy()
    n_units = float(pd.to_numeric(sub["n_units"], errors="coerce").sum())
    changed_count = float(pd.to_numeric(sub["changed_count"], errors="coerce").sum())
    changed_prev = changed_count / n_units if n_units > 0.0 else float("nan")
    counts = np.asarray([float(pd.to_numeric(sub[f"n_{name}"], errors="coerce").sum()) for name in order], dtype=float)
    if not np.isfinite(counts).all() or float(counts.sum()) <= 0.0:
        counts = np.asarray([1.0, 0.0, 0.0], dtype=float)

    ax.set_aspect("equal")
    ax.axis("off")
    wedges, _ = ax.pie(
        counts,
        radius=0.54,
        center=(0.0, 0.0),
        startangle=90.0,
        counterclock=False,
        colors=[colors[name] for name in order],
        wedgeprops={"width": 0.23, "edgecolor": "white", "linewidth": 0.9},
    )
    total = float(counts.sum())
    for wedge, count in zip(wedges, counts):
        frac = count / total if total > 0.0 else 0.0
        if frac < 0.12:
            continue
        theta = np.deg2rad(0.5 * (wedge.theta1 + wedge.theta2))
        r = 0.54 - 0.5 * 0.23
        ax.text(r * np.cos(theta), r * np.sin(theta), f"{frac * 100.0:.0f}%", ha="center", va="center", fontsize=ANNOTATION_SIZE, color=COLOR_TEXT)
    ax.text(0.0, 0.01, display_label, ha="center", va="center", fontsize=ANNOTATION_SIZE + 0.05, linespacing=0.95)
    ax.text(0.0, -0.66, f"{100.0 * changed_prev:.1f}%", ha="center", va="top", fontsize=ANNOTATION_SIZE, color=COLOR_DARK_GRAY)
    ax.set_xlim(-0.86, 0.86)
    ax.set_ylim(-0.86, 0.76)
    if show_legend:
        handles = [Patch(facecolor=colors[name], edgecolor="none", label=name) for name in order]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=len(order), frameon=False, columnspacing=1.0, handletextpad=0.35)


def draw_panel_b_full_denominator_composition(ax: plt.Axes, transition_df: pd.DataFrame) -> None:
    order = ["advance", "recruit", "unchanged", "loss"]
    colors = {
        "advance": COLOR_OVERLAP,
        "recruit": "#24B2A7",
        "unchanged": "#D6DADF",
        "loss": COLOR_PROBE_ONLY,
    }
    _draw_panel_b_donut(
        ax,
        transition_df,
        order=order,
        colors=colors,
        count_key_by_name={name: f"n_{name}" for name in order},
        center_text="full denominator\nreceiving-input units",
        prevalence_lines=None,
    )


def draw_panel_c_event_time(fig: plt.Figure, spec, aligned_arrays: dict[str, np.ndarray]) -> plt.Axes:
    inner = spec.subgridspec(2, 1, hspace=0.10)
    ax_top = fig.add_subplot(inner[0, 0])
    ax_bottom = fig.add_subplot(inner[1, 0], sharex=ax_top)
    rel_t = np.asarray(aligned_arrays["relative_time"], dtype=float)
    winner = np.asarray(aligned_arrays["winner_delta_v_aligned"], dtype=float)
    loser = np.asarray(aligned_arrays["loser_delta_v_aligned"], dtype=float)
    loser_inh_before = np.asarray(aligned_arrays["loser_inh_before_aligned"], dtype=float)
    if winner.shape[0] <= 0:
        for ax in (ax_top, ax_bottom):
            ax.text(0.5, 0.5, "No aligned local events", ha="center", va="center", transform=ax.transAxes)
            style_axes(ax)
            ax.set_xticks([])
            ax.set_yticks([])
        return ax_top

    winner_mean, winner_sem = _nanmean_sem(winner)
    loser_mean, loser_sem = _nanmean_sem(loser)
    loser_inh_before_mean, loser_inh_before_sem = _nanmean_sem(loser_inh_before)

    style_axes(ax_top)
    ax_top.axvline(0.0, color=COLOR_DARK_GRAY, linewidth=REF_LINEWIDTH, linestyle=(0, (3, 2)))
    ax_top.axhline(0.0, color=COLOR_GRID, linewidth=0.7)
    ax_top.plot(rel_t, 1000.0 * winner_mean, color=COLOR_OVERLAP, linewidth=DATA_LINEWIDTH, label="winner delta V")
    ax_top.fill_between(rel_t, 1000.0 * (winner_mean - winner_sem), 1000.0 * (winner_mean + winner_sem), color=COLOR_OVERLAP, alpha=0.18, linewidth=0)
    ax_top.plot(rel_t, 1000.0 * loser_mean, color=COLOR_PROBE_ONLY, linewidth=DATA_LINEWIDTH, label="loser delta V")
    ax_top.fill_between(rel_t, 1000.0 * (loser_mean - loser_sem), 1000.0 * (loser_mean + loser_sem), color=COLOR_PROBE_ONLY, alpha=0.18, linewidth=0)
    ax_top.set_ylabel("ΔV_effective (mV)")
    _framed_legend(
        ax_top,
        [
            Line2D([0], [0], color=COLOR_OVERLAP, lw=DATA_LINEWIDTH, label="winner delta V"),
            Line2D([0], [0], color=COLOR_PROBE_ONLY, lw=DATA_LINEWIDTH, label="loser delta V"),
            Line2D([0], [0], color=COLOR_DARK_GRAY, lw=REF_LINEWIDTH, linestyle=(0, (3, 2)), label="winner spike"),
        ],
    )
    ax_top.tick_params(labelbottom=False)

    style_axes(ax_bottom)
    ax_bottom.axvline(0.0, color=COLOR_DARK_GRAY, linewidth=REF_LINEWIDTH, linestyle=(0, (3, 2)))
    ax_bottom.axhline(0.0, color=COLOR_GRID, linewidth=0.7)
    ax_bottom.plot(rel_t, 1000.0 * loser_inh_before_mean, color=COLOR_DYNAMIC, linewidth=DATA_LINEWIDTH)
    ax_bottom.fill_between(
        rel_t,
        1000.0 * (loser_inh_before_mean - loser_inh_before_sem),
        1000.0 * (loser_inh_before_mean + loser_inh_before_sem),
        color=COLOR_DYNAMIC,
        alpha=0.18,
        linewidth=0,
    )
    ax_bottom.set_xlabel("relative time to winner dynamic first spike")
    ax_bottom.set_ylabel("loser inhibition (mV)")
    return ax_top


def _draw_two_group_metric(
    ax: plt.Axes,
    values_by_group: dict[str, np.ndarray],
    *,
    ylabel: str,
    colors_by_group: dict[str, str],
    scale: float = 1.0,
    reference: float | None = None,
) -> None:
    style_axes(ax)
    if reference is not None:
        add_reference_line(ax, reference)
    groups = ["overlap_dominant", "probe_only_dominant"]
    xpos = np.arange(len(groups), dtype=float)
    rng = np.random.default_rng(20260410)
    for idx, group in enumerate(groups):
        vals = np.asarray(values_by_group[group], dtype=float) * scale
        vals = vals[np.isfinite(vals)]
        mean, lo, hi = _bootstrap_ci(vals, seed=300 + idx)
        jitter = rng.uniform(-0.08, 0.08, size=vals.size)
        ax.scatter(np.full(vals.size, xpos[idx]) + jitter, vals, s=11, color=colors_by_group[group], alpha=0.18, linewidths=0, zorder=2)
        ax.vlines(xpos[idx], lo, hi, color=colors_by_group[group], linewidth=2.2, zorder=3)
        ax.scatter([xpos[idx]], [mean], s=34, color=colors_by_group[group], edgecolor="white", linewidth=0.35, zorder=4)
    ax.set_xticks(xpos, [_group_label(g) for g in groups])
    ax.set_ylabel(ylabel)


def draw_panel_c_transitions(ax: plt.Axes, transition_df: pd.DataFrame, stage_df: pd.DataFrame | None) -> None:
    style_axes(ax)
    groups = ["overlap_dominant", "probe_only_dominant"]
    metrics = [("P_advance", "advance"), ("P_loss", "loss")] if stage_df is not None and "P_loss" in stage_df.columns else [("P_advance", "advance"), ("P_recruit", "recruit")]
    work_df = stage_df.copy() if stage_df is not None else transition_df.copy()
    work_df = work_df[work_df["unit_group"].isin(groups)].copy()
    xpos = np.arange(len(metrics), dtype=float)
    offsets = {"overlap_dominant": -0.12, "probe_only_dominant": 0.12}
    colors = {"overlap_dominant": COLOR_OVERLAP, "probe_only_dominant": COLOR_PROBE_ONLY}
    rng = np.random.default_rng(20260411)

    for group in groups:
        group_df = work_df[work_df["unit_group"] == group]
        for metric_idx, (metric, _) in enumerate(metrics):
            vals = pd.to_numeric(group_df[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            x_center = xpos[metric_idx] + offsets[group]
            jitter = rng.uniform(-0.03, 0.03, size=vals.size)
            mean, lo, hi = _bootstrap_ci(vals, seed=1000 + metric_idx * 10 + (0 if group == "overlap_dominant" else 1))
            ax.scatter(np.full(vals.size, x_center) + jitter, vals, s=10, color=colors[group], alpha=0.18, linewidths=0, zorder=2)
            ax.vlines(x_center, lo, hi, color=colors[group], linewidth=2.2, zorder=3)
            ax.scatter([x_center], [mean], s=32, color=colors[group], edgecolor="white", linewidth=0.35, zorder=4)

    ax.set_xticks(xpos, [label for _, label in metrics])
    ax.set_ylabel("probability")
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=5, markerfacecolor=COLOR_OVERLAP, markeredgecolor="white", label="overlap-biased"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=5, markerfacecolor=COLOR_PROBE_ONLY, markeredgecolor="white", label="probe-only-biased"),
    ]
    _framed_legend(ax, handles)


def draw_panel_d_overlap_gain(ax: plt.Axes, input_gain_df: pd.DataFrame) -> None:
    groups = ["overlap_dominant", "probe_only_dominant"]
    gain_df = input_gain_df[input_gain_df["unit_group"].isin(groups)].copy()
    colors = {"overlap_dominant": COLOR_OVERLAP, "probe_only_dominant": COLOR_PROBE_ONLY}
    _draw_two_group_metric(
        ax,
        values_by_group={
            group: gain_df.loc[gain_df["unit_group"] == group, "overlap_input_gain"].to_numpy(dtype=float)
            for group in groups
        },
        ylabel="overlap input gain\n($\\times 10^{-10}$)",
        colors_by_group=colors,
        scale=1e10,
    )


def draw_panel_e_loss_inhibition(ax: plt.Axes, loss_df: pd.DataFrame) -> None:
    groups = ["overlap_dominant", "probe_only_dominant"]
    loss_df = loss_df[loss_df["unit_group"].isin(groups)].copy()
    colors = {"overlap_dominant": COLOR_OVERLAP, "probe_only_dominant": COLOR_PROBE_ONLY}
    _draw_two_group_metric(
        ax,
        values_by_group={
            group: loss_df.loc[loss_df["unit_group"] == group, "lost_spike_delta_inh"].to_numpy(dtype=float)
            for group in groups
        },
        ylabel="$\\Delta$ inhibition",
        colors_by_group=colors,
        reference=0.0,
    )


def _draw_panel_f_fallback(ax: plt.Axes, pair_df: pd.DataFrame) -> None:
    style_axes(ax)
    ax.scatter(
        pair_df["contrast_static"].to_numpy(dtype=float),
        pair_df["contrast_dynamic"].to_numpy(dtype=float),
        s=14,
        color=COLOR_OVERLAP,
        alpha=0.7,
    )
    lims = [
        float(np.nanmin([pair_df["contrast_static"].min(), pair_df["contrast_dynamic"].min()])),
        float(np.nanmax([pair_df["contrast_static"].max(), pair_df["contrast_dynamic"].max()])),
    ]
    ax.plot(lims, lims, color=COLOR_DARK_GRAY, linewidth=REF_LINEWIDTH, linestyle=(0, (3, 2)))
    ax.set_xlabel("static contrast")
    ax.set_ylabel("dynamic contrast")


def draw_panel_g_summary(ax: plt.Axes, support_df: pd.DataFrame) -> None:
    style_axes(ax)
    support = support_df["local_winner_support_rate"].to_numpy(dtype=float)
    support = support[np.isfinite(support)]
    rng = np.random.default_rng(20260412)
    jitter = rng.uniform(-0.05, 0.05, size=support.size)
    ax.boxplot(
        [support],
        positions=[0.0],
        widths=0.26,
        vert=True,
        patch_artist=True,
        boxprops={"facecolor": "#D8F0E6", "edgecolor": COLOR_OVERLAP, "linewidth": 0.9},
        medianprops={"color": COLOR_OVERLAP, "linewidth": 1.2},
        whiskerprops={"color": COLOR_OVERLAP, "linewidth": 0.9},
        capprops={"color": COLOR_OVERLAP, "linewidth": 0.9},
    )
    ax.scatter(np.full(support.size, 0.0) + jitter, support, s=10, color=COLOR_OVERLAP, alpha=0.28, linewidths=0, zorder=3)
    mean_val = float(np.mean(support))
    ax.scatter([0.0], [mean_val], s=28, color=COLOR_TEXT, zorder=4)
    ax.text(0.0, mean_val + 0.03, f"{mean_val:.2f}", ha="center", va="bottom", fontsize=ANNOTATION_SIZE)
    ax.set_xlim(-0.28, 0.28)
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks([])
    ax.set_ylabel("support rate")


def draw_panel_d_voltage(ax: plt.Axes, trace_arrays: dict[str, np.ndarray] | None) -> None:
    style_axes(ax)
    if trace_arrays is None:
        ax.text(0.5, 0.5, "No exemplar trace", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    t_axis = np.asarray(trace_arrays["t_axis"], dtype=float)
    traces = {
        "winner_dynamic": 1000.0 * np.asarray(trace_arrays["winner_v_effective_dynamic"], dtype=float),
        "winner_static": 1000.0 * np.asarray(trace_arrays["winner_v_effective_static"], dtype=float),
        "loser_dynamic": 1000.0 * np.asarray(trace_arrays["loser_v_effective_dynamic"], dtype=float),
        "loser_static": 1000.0 * np.asarray(trace_arrays["loser_v_effective_static"], dtype=float),
    }
    ax.plot(t_axis, traces["winner_dynamic"], color=COLOR_OVERLAP, linewidth=DATA_LINEWIDTH, label="winner, dynamic")
    ax.plot(t_axis, traces["winner_static"], color=COLOR_OVERLAP, linewidth=1.15, linestyle=(0, (4, 2)), alpha=0.92, label="winner, static")
    ax.plot(t_axis, traces["loser_dynamic"], color=COLOR_PROBE_ONLY, linewidth=DATA_LINEWIDTH, label="loser, dynamic")
    ax.plot(t_axis, traces["loser_static"], color=COLOR_PROBE_ONLY, linewidth=1.15, linestyle=(0, (4, 2)), alpha=0.92, label="loser, static")
    ax.axhline(-60.0, color=COLOR_DARK_GRAY, linewidth=REF_LINEWIDTH, linestyle=(0, (3, 2)), zorder=1)
    for key, trace_key, color, filled in (
        ("winner_first_spike_dynamic", "winner_dynamic", COLOR_OVERLAP, True),
        ("winner_first_spike_static", "winner_static", COLOR_OVERLAP, False),
        ("loser_first_spike_dynamic", "loser_dynamic", COLOR_PROBE_ONLY, True),
        ("loser_first_spike_static", "loser_static", COLOR_PROBE_ONLY, False),
    ):
        spike_t = int(np.asarray(trace_arrays[key]).reshape(-1)[0] if np.asarray(trace_arrays[key]).ndim > 0 else trace_arrays[key])
        if 0 <= spike_t < traces[trace_key].shape[0]:
            ax.scatter(
                [t_axis[spike_t]],
                [traces[trace_key][spike_t]],
                s=18,
                facecolor=color if filled else "white",
                edgecolor=color,
                linewidth=0.8,
                zorder=4,
            )
    y_all = np.concatenate([traces["winner_dynamic"], traces["winner_static"], traces["loser_dynamic"], traces["loser_static"], np.asarray([-60.0], dtype=float)])
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    y_pad = max((y_max - y_min) * 0.08, 1.5)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    handles = [
        Line2D([0], [0], color=COLOR_OVERLAP, lw=DATA_LINEWIDTH, label="winner, dynamic"),
        Line2D([0], [0], color=COLOR_OVERLAP, lw=1.15, linestyle=(0, (4, 2)), label="winner, static"),
        Line2D([0], [0], color=COLOR_PROBE_ONLY, lw=DATA_LINEWIDTH, label="loser, dynamic"),
        Line2D([0], [0], color=COLOR_PROBE_ONLY, lw=1.15, linestyle=(0, (4, 2)), label="loser, static"),
        Line2D([0], [0], color=COLOR_DARK_GRAY, lw=REF_LINEWIDTH, linestyle=(0, (3, 2)), label="-60 mV"),
    ]
    _framed_legend(ax, handles)
    ax.set_xlabel("probe step")
    ax.set_ylabel("$V_{effective}$ (mV)")


def draw_panel_d_contrast_shift(ax: plt.Axes, pair_df: pd.DataFrame) -> None:
    style_axes(ax)
    add_reference_line(ax, 0.0)
    values = 1000.0 * pair_df["winner_loser_contrast_shift"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 0:
        ax.text(0.5, 0.5, "No local pairs", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    rng = np.random.default_rng(20260414)
    ax.boxplot(
        [values],
        positions=[0.0],
        widths=0.30,
        patch_artist=True,
        boxprops={"facecolor": "#E8F5EF", "edgecolor": COLOR_OVERLAP, "linewidth": 0.95},
        whiskerprops={"color": COLOR_OVERLAP, "linewidth": 0.9},
        capprops={"color": COLOR_OVERLAP, "linewidth": 0.9},
        medianprops={"color": COLOR_OVERLAP, "linewidth": 1.25},
        flierprops={"markersize": 0},
    )
    jitter = rng.uniform(-0.075, 0.075, size=values.size)
    ax.scatter(np.full(values.size, 0.0) + jitter, values, s=13, color=COLOR_OVERLAP, alpha=0.28, linewidths=0, zorder=3)
    mean_val, lo, hi = _bootstrap_ci(values, seed=20260414)
    ax.vlines(0.0, lo, hi, color=COLOR_TEXT, linewidth=1.5, zorder=4)
    ax.scatter([0.0], [mean_val], s=28, color=COLOR_TEXT, zorder=5)
    positive_frac = 100.0 * float(np.mean(values > 0.0))
    ax.text(0.0, 0.98, f"mean {mean_val:.2f} mV\npositive {positive_frac:.0f}%", ha="center", va="top", transform=ax.transAxes, fontsize=ANNOTATION_SIZE)
    ax.set_xlim(-0.28, 0.28)
    ax.set_xticks([0.0], ["local pairs"])
    ax.set_ylabel("contrast shift (mV)")


def draw_panel_d_chain_prevalence(ax: plt.Axes, chain_df: pd.DataFrame) -> None:
    style_axes(ax)
    if chain_df.empty:
        ax.text(0.5, 0.5, "No local chain events", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    metrics = [
        ("winner_pre_spike_boost", "winner\nboosted"),
        ("loser_post_winner_suppressed", "loser\nsuppressed after"),
        ("full_chain_satisfied", "full\nchain"),
    ]
    xpos = np.arange(len(metrics), dtype=float)
    bar_color = "#8CCFC5"
    for idx, (metric, _) in enumerate(metrics):
        vals = pd.to_numeric(chain_df[metric], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        mean = float(np.mean(vals)) if vals.size > 0 else float("nan")
        _, lo, hi = _bootstrap_ci(vals, seed=1700 + idx)
        ax.bar(
            xpos[idx],
            100.0 * mean,
            width=0.56,
            color=bar_color,
            edgecolor="none",
            linewidth=0.0,
            zorder=2,
        )
        ax.vlines(xpos[idx], 100.0 * lo, 100.0 * hi, color=COLOR_DYNAMIC, linewidth=1.8, zorder=3)
        ax.hlines([100.0 * lo, 100.0 * hi], xpos[idx] - 0.08, xpos[idx] + 0.08, color=COLOR_DYNAMIC, linewidth=1.6, zorder=3)
        ax.text(
            xpos[idx],
            100.0 * mean + 3.0,
            f"{100.0 * mean:.0f}%",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            color=COLOR_TEXT,
        )
    ax.set_xticks(xpos, [label for _, label in metrics])
    ax.set_ylabel("prevalence (%)")
    ax.set_ylim(-2.0, 105.0)


def build_assembled_figure(root: str | Path) -> tuple[plt.Figure, dict[str, object]]:
    apply_paper_style()
    bundle = load_fig4_bundle(root)
    fig = plt.figure(figsize=(8.8, 8.9))
    outer = fig.add_gridspec(3, 2, height_ratios=[0.95, 0.82, 1.05], hspace=0.42, wspace=0.30)

    spec_a = outer[0, :].subgridspec(1, 1)
    row_bc = outer[1, :].subgridspec(2, 2, height_ratios=[0.18, 0.82], hspace=0.02, wspace=0.24)
    row_de = outer[2, :].subgridspec(1, 2, wspace=0.30)

    ax_a = draw_panel_a_definition(fig, spec_a[0, 0], bundle["panel_a_arrays"])

    ax_bc_legend = fig.add_subplot(row_bc[0, :])
    ax_bc_legend.axis("off")
    legend_handles = [
        Patch(facecolor=COLOR_OVERLAP, edgecolor="none", label="advance"),
        Patch(facecolor="#24B2A7", edgecolor="none", label="recruit"),
        Patch(facecolor=COLOR_PROBE_ONLY, edgecolor="none", label="loss"),
    ]
    ax_bc_legend.legend(
        handles=legend_handles,
        loc="center",
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

    ax_b = fig.add_subplot(row_bc[1, 0])
    draw_panel_b_single_group(ax_b, bundle["panel_b_changed"], "overlap_dominant", show_legend=False)
    ax_c = fig.add_subplot(row_bc[1, 1])
    draw_panel_b_single_group(ax_c, bundle["panel_b_changed"], "probe_only_dominant", show_legend=False)

    ax_d = draw_panel_c_event_time(fig, row_de[0, 0], bundle["panel_c_event_time"])
    ax_e = fig.add_subplot(row_de[0, 1])
    draw_panel_d_chain_prevalence(ax_e, bundle["panel_d_chain"])
    return fig, bundle


def build_panel_figures(root: str | Path) -> tuple[dict[str, plt.Figure], dict[str, object]]:
    apply_paper_style()
    bundle = load_fig4_bundle(root)
    figures: dict[str, plt.Figure] = {}

    fig_a = plt.figure(figsize=(5.2, 2.2))
    ax_a = draw_panel_a_definition(fig_a, fig_a.add_gridspec(1, 1)[0, 0], bundle["panel_a_arrays"])
    figures["panel_a"] = fig_a

    fig_b = plt.figure(figsize=(3.5, 2.8))
    ax_b = fig_b.add_subplot(1, 1, 1)
    draw_panel_b_transition_composition(ax_b, bundle["panel_b_changed"])
    figures["panel_b"] = fig_b

    fig_b_overlap = plt.figure(figsize=(2.2, 2.6))
    ax_b_overlap = fig_b_overlap.add_subplot(1, 1, 1)
    draw_panel_b_single_group(ax_b_overlap, bundle["panel_b_changed"], "overlap_dominant")
    figures["panel_b_overlap_biased"] = fig_b_overlap

    fig_b_probe = plt.figure(figsize=(2.2, 2.6))
    ax_b_probe = fig_b_probe.add_subplot(1, 1, 1)
    draw_panel_b_single_group(ax_b_probe, bundle["panel_b_changed"], "probe_only_dominant")
    figures["panel_b_probe_only_biased"] = fig_b_probe

    fig_b_full = plt.figure(figsize=(3.5, 2.8))
    ax_b_full = fig_b_full.add_subplot(1, 1, 1)
    draw_panel_b_full_denominator_composition(ax_b_full, bundle["panel_b_transition"])
    figures["panel_b_full_denominator"] = fig_b_full

    fig_c = plt.figure(figsize=(4.8, 3.5))
    ax_c = draw_panel_c_event_time(fig_c, fig_c.add_gridspec(1, 1)[0, 0], bundle["panel_c_event_time"])
    figures["panel_c"] = fig_c

    fig_d = plt.figure(figsize=(4.1, 2.8))
    ax_d = fig_d.add_subplot(1, 1, 1)
    draw_panel_d_chain_prevalence(ax_d, bundle["panel_d_chain"])
    figures["panel_d"] = fig_d

    fig_exemplar = plt.figure(figsize=(4.5, 2.8))
    ax_exemplar = fig_exemplar.add_subplot(1, 1, 1)
    draw_panel_d_voltage(ax_exemplar, bundle["panel_f_trace"])
    figures["panel_exemplar"] = fig_exemplar

    fig_support = plt.figure(figsize=(2.1, 2.0))
    ax_support = fig_support.add_subplot(1, 1, 1)
    draw_panel_g_summary(ax_support, bundle["panel_g_support"])
    figures["panel_support"] = fig_support

    return figures, bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render Fig4 from paper_figs result tables.")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--export-panels", action="store_true")
    args = parser.parse_args(argv)

    input_dir = resolve_figure_input_dir("fig4", args.input_dir)
    output_dir = Path(input_dir) / "plots" if args.output_dir is None else Path(args.output_dir)

    fig, bundle = build_assembled_figure(input_dir)
    saved: dict[str, object] = {"figure": save_figure_outputs(fig, output_dir, "fig4")}
    plt.close(fig)

    panel_figures, _ = build_panel_figures(input_dir)
    panel_saved: dict[str, dict[str, str]] = {}
    for panel_name, panel_fig in panel_figures.items():
        panel_saved[panel_name] = save_figure_outputs(panel_fig, output_dir, f"fig4_{panel_name}")
        plt.close(panel_fig)
    saved["panels"] = panel_saved
    saved["panels_export_mode"] = "always"

    print(
        json.dumps(
            {
                "status": "ok",
                "figure": "fig4",
                "input_dir": str(input_dir),
                "summary_keys": sorted(bundle["summary"].keys()),
                "saved": saved,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
