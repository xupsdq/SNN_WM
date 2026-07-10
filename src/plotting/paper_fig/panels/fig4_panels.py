from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import t as student_t

from src.plotting.common.colors import get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL, GRID_ALPHA_SOFT
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder
from src.plotting.paper_fig.svg_assets import render_svg_asset_panel


def render_fig4_reentry_schematic(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data, stats, style
    render_svg_asset_panel(ax, spec)


def render_fig4_similarity_entry(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    df = df.sort_values(["similarity_bin_order", "seed_id"], kind="stable")
    summary = _mean_sem(df, ["similarity_bin_order", "similarity_bin"], "value").sort_values("similarity_bin_order")
    x = np.arange(len(summary), dtype=float)
    y = summary["mean"].to_numpy(dtype=float) * 100.0
    sem = summary["sem"].to_numpy(dtype=float) * 100.0
    ax.errorbar(x, y, yerr=sem, fmt="o-", color="#009E73", linewidth=1.15, markersize=3.6, capsize=2.0)
    if _run_mode(stats) == "single_network_draft":
        ax.scatter(np.arange(len(df)), np.interp(df["similarity_bin_order"], summary["similarity_bin_order"], y), s=4, color=COLOR_NEUTRAL, alpha=0.18)
    ax.set_xticks(x, [""] * len(x))
    ax.annotate("", xy=(0.78, -0.04), xytext=(0.22, -0.04), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "linewidth": 0.65, "color": COLOR_NEUTRAL}, annotation_clip=False)
    ax.set_xlabel("")
    ax.set_ylabel(str(spec.get("y_axis", "Accuracy drop")))
    if spec.get("y_label_x") is not None:
        ax.yaxis.set_label_coords(float(spec.get("y_label_x")), float(spec.get("y_label_y", 0.5)))
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig4_similarity_entry"
    ax.paper_fig_similarity_direction_arrow = True
    ax.paper_fig_literal_bin_xticklabels = False
    ax.paper_fig_similarity_bar_order_preserved = True


def render_fig4_overlap_localization(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = ["Low overlap", "High overlap"]
    colors = [get_plot_color("shuffled_pair"), get_plot_color("true_pair")]
    x = np.arange(len(order), dtype=float)
    means: list[float] = []
    sems: list[float] = []
    for condition in order:
        vals = pd.to_numeric(df.loc[df["condition"].eq(condition), "value"], errors="coerce").dropna()
        means.append(float(vals.mean()) if len(vals) else np.nan)
        sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)
    ax.bar(x, means, yerr=sems, color=colors, edgecolor=COLOR_NEUTRAL, linewidth=0.55, alpha=0.72, capsize=2.0, zorder=2)
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.7, zorder=1)
    ax.set_xticks(x, order)
    ax.set_xlabel("")
    ax.set_ylabel(str(spec.get("y_axis", "Accuracy drop")))
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig4_overlap_localization_bar"
    ax.paper_fig_raw_points = False
    ax.paper_fig_paired_lines = False


def render_fig4_overlap_excess(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    metric = str(spec.get("plot_metric", "mean_acc_drop"))
    df = df[df["metric"].astype(str).eq(metric)] if not df.empty and "metric" in df.columns else df
    if df.empty or not {"iso_similarity_bin", "condition"}.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    work = df.copy()
    work["iso_similarity_bin_order"] = pd.to_numeric(work.get("iso_similarity_bin_order", pd.Series(np.nan, index=work.index)), errors="coerce")
    grouped = work.groupby(["iso_similarity_bin_order", "iso_similarity_bin", "condition"], dropna=False)["value"].mean().reset_index()
    pivot = grouped.pivot_table(index=["iso_similarity_bin_order", "iso_similarity_bin"], columns="condition", values="value").reset_index()
    if {"High overlap excess", "Low overlap excess"}.issubset(pivot.columns):
        pivot = pivot.sort_values("iso_similarity_bin_order")
        low_vals = pd.to_numeric(pivot["Low overlap excess"], errors="coerce").dropna() * 100.0
        high_vals = pd.to_numeric(pivot["High overlap excess"], errors="coerce").dropna() * 100.0
        means = [float(low_vals.mean()) if len(low_vals) else np.nan, float(high_vals.mean()) if len(high_vals) else np.nan]
        sems = [float(low_vals.sem()) if len(low_vals) > 1 else 0.0, float(high_vals.sem()) if len(high_vals) > 1 else 0.0]
        ax.bar([0, 1], means, yerr=sems, color=["#8A8A8A", "#007A5A"], edgecolor=COLOR_NEUTRAL, linewidth=0.45, alpha=0.76, capsize=1.8, zorder=2)
        ax.set_xticks([0, 1], ["Low", "High"])
    else:
        order = [c for c in ["Low overlap excess", "High overlap excess"] if c in set(df["condition"].astype(str))]
        x = np.arange(len(order), dtype=float)
        means, sems = [], []
        for condition in order:
            vals = pd.to_numeric(df.loc[df["condition"].astype(str).eq(condition), "value"], errors="coerce").dropna() * 100.0
            means.append(float(vals.mean()) if len(vals) else np.nan)
            sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)
        ax.bar(x, means, yerr=sems, color=["#8A8A8A", "#007A5A"][: len(order)], edgecolor=COLOR_NEUTRAL, linewidth=0.45, alpha=0.76, capsize=1.8, zorder=2)
        ax.set_xticks(x, ["Low", "High"][: len(order)])
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.6)
    ax.set_xlabel(str(spec.get("x_axis", "Overlap beyond similarity")))
    ax.set_ylabel(str(spec.get("y_axis", "Probe bias (%)")), labelpad=float(spec.get("y_labelpad", 1.0)))
    if spec.get("y_label_x") is not None:
        ax.yaxis.set_label_coords(float(spec.get("y_label_x")), float(spec.get("y_label_y", 0.5)))
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "s7_overlap_excess"
    ax.paper_fig_raw_points = False
    ax.paper_fig_paired_lines = False


def render_fig4_overlap_accuracy_identification(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    rates = df[df["metric"].isin(["drop_rate_low_overlap", "drop_rate_high_overlap"])].copy()
    order = ["low_overlap", "high_overlap"]
    label_map = {"low_overlap": "Low\noverlap", "high_overlap": "High\noverlap"}
    color_map = {"low_overlap": get_plot_color("shuffled_pair"), "high_overlap": get_plot_color("true_pair")}
    for idx, condition in enumerate(order):
        vals = pd.to_numeric(rates.loc[rates["condition"].eq(condition), "value"], errors="coerce").dropna()
        if vals.empty:
            continue
        jitter = np.linspace(-0.05, 0.05, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), idx) + jitter, vals, s=12, color=color_map[condition], alpha=0.45, zorder=3)
        ax.bar([idx], [float(vals.mean())], color=color_map[condition], alpha=0.55, edgecolor=COLOR_NEUTRAL, linewidth=0.55, zorder=2)
        ax.errorbar([idx], [float(vals.mean())], yerr=[float(vals.sem()) if len(vals) > 1 else 0.0], fmt="none", ecolor=COLOR_NEUTRAL, capsize=2.0, linewidth=0.8, zorder=4)
    delta = pd.to_numeric(df.loc[df["metric"].eq("delta_drop_rate"), "value"], errors="coerce").dropna()
    if not delta.empty:
        ax.text(0.03, 0.95, f"Delta drop {float(delta.mean()):.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=5.4, color=COLOR_NEUTRAL)
    balance = pd.to_numeric(df.loc[df["metric"].eq("mean_similarity_difference"), "value"], errors="coerce").dropna()
    if not balance.empty:
        ax.text(0.03, 0.82, f"Mean sim diff {float(balance.mean()):.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=5.0, color=COLOR_NEUTRAL)
    ax.set_xticks([0, 1], [label_map[k] for k in order])
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel(str(spec.get("y_axis", "Drop event rate")))
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig4_overlap_accuracy_identification"


def render_fig4_decision_spike_displacement(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = style
    df = _clean(panel_data)
    if "metric" in df.columns:
        df = df[df["metric"].astype(str).eq("DPI_L3_t")]
    if "analysis_role" in df.columns:
        df = df[df["analysis_role"].astype(str).eq("network_time_mean")]
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if "time_ms" not in df.columns:
        render_fig4_overlap_localization(ax, panel_data, stats, spec, style)
        return
    max_time_ms = float((stats or {}).get("max_time_ms_used", spec.get("max_time_ms", 60)))
    time_values = pd.to_numeric(df["time_ms"], errors="coerce")
    df = df[(time_values >= 0.0) & (time_values <= max_time_ms)].copy()
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    colors = {"Overlap support": "#007A5A", "Non-overlap support": "#CC79A7"}
    display_labels = dict(spec.get("display_labels_short") or {})
    network_col = "network_id" if "network_id" in df.columns else "seed_id" if "seed_id" in df.columns else None
    rendered_network_traces = 0
    for condition in ["Overlap support", "Non-overlap support"]:
        part = df[df["condition"].eq(condition)].dropna(subset=["time_ms", "value"])
        if part.empty:
            continue
        color = colors[condition]
        if network_col is not None:
            for _, trace in part.groupby(network_col, dropna=False, sort=True):
                trace = trace.sort_values("time_ms", kind="stable")
                ax.plot(
                    trace["time_ms"].to_numpy(dtype=float),
                    trace["value"].to_numpy(dtype=float),
                    color=color,
                    linewidth=0.34,
                    alpha=0.12,
                    zorder=1,
                )
                rendered_network_traces += 1
        summary = _mean_t_ci(part, ["time_ms"], "value").sort_values("time_ms")
        x = summary["time_ms"].to_numpy(dtype=float)
        y = summary["mean"].to_numpy(dtype=float)
        ci_low = summary["ci95_low"].to_numpy(dtype=float)
        ci_high = summary["ci95_high"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, linewidth=1.15, label=display_labels.get(condition, condition), zorder=3)
        ax.fill_between(x, ci_low, ci_high, color=color, alpha=0.16, linewidth=0, zorder=2)
    for line in spec.get("reference_lines") or []:
        ax.axhline(float(line.get("value", 0.0)), color="0.45", linestyle="--", linewidth=0.7)
    ax.set_xlim(0.0, max_time_ms)
    ax.set_xlabel(str(spec.get("x_axis", "Probe time (ms)")))
    ax.set_ylabel(str(spec.get("y_axis", "DPI")))
    legend = ax.legend(frameon=False, fontsize=5.7, loc="upper right", handlelength=0.9, labelspacing=0.2, borderaxespad=0.2)
    ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
    ax.paper_fig_legend_overlaps_data = False
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig4_decision_spike_displacement"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(df))
    ax.paper_fig_raw_point_alpha = 0.12
    ax.paper_fig_individual_traces = True
    ax.paper_fig_network_trace_count = rendered_network_traces
    ax.paper_fig_error_bar = "two_sided_t_95_ci_across_networks"
    ax.paper_fig_inference_unit = "independently_trained_network"
    ax.paper_fig_renderer_summarizes_row_level = False


def render_fig4_decision_deflection(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty or not {"x_value", "y_value"}.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    focus = df[df["condition"].isin(["Static", "Overlap support", "Non-overlap support", "Random matched", "Dynamic"])]
    colors = {
        "Static": "0.65",
        "Dynamic": get_plot_color("dynamic"),
        "Overlap support": get_plot_color("true_pair"),
        "Non-overlap support": get_plot_color("shuffled_pair"),
        "Random matched": get_plot_color("other_residual"),
    }
    for condition, part in focus.groupby("condition", sort=False):
        ax.scatter(part["x_value"], part["y_value"], s=10, alpha=0.32, color=colors.get(condition, COLOR_NEUTRAL), label=condition)
        mean_x = float(pd.to_numeric(part["x_value"], errors="coerce").mean())
        mean_y = float(pd.to_numeric(part["y_value"], errors="coerce").mean())
        ax.scatter([mean_x], [mean_y], s=38, color=colors.get(condition, COLOR_NEUTRAL), edgecolor="white", linewidth=0.5, zorder=5)
    ax.axhline(0, color="0.55", linestyle=":", linewidth=0.7)
    ax.axvline(0, color="0.55", linestyle=":", linewidth=0.7)
    ax.annotate("", xy=(0.82, 0.18), xytext=(0.18, 0.18), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "linewidth": 0.75, "color": COLOR_NEUTRAL})
    ax.set_xlabel(str(spec.get("x_axis", "Static-to-dynamic push")))
    ax.set_ylabel(str(spec.get("y_axis", "Dynamic-like recovery")))
    _set_padded_limits(ax, focus, "x_value", "y_value")
    xmin, xmax = ax.get_xlim()
    if xmax > xmin:
        ax.set_xticks(np.linspace(xmin, xmax, 5)[1:-1])
    legend = ax.legend(frameon=False, fontsize=4.6, loc="best", handlelength=0.8, labelspacing=0.15)
    ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
    ax.paper_fig_legend_overlaps_data = False
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig4_decision_deflection"


def render_fig4_l3_accumulator_process(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if set(df.get("metric", pd.Series(dtype=str)).astype(str)) == {"decision_deflection_fallback"}:
        ax.paper_fig_plot_form = "decision_deflection_fallback"
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        ax.text(0.5, 0.12, "decision-deflection fallback", transform=ax.transAxes, ha="center", va="center", fontsize=5.4, color=COLOR_NEUTRAL)
        return
    required = {"group", "x0", "y0", "x1", "y1"}
    if not required.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    for col in ("x0", "y0", "x1", "y1"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["x0", "y0", "x1", "y1"])
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    plus_color = get_plot_color("dynamic")
    minus_color = get_plot_color("static_frozen")
    guide_color = get_plot_color("other_residual")
    ax.plot([-1.1, 1.1], [-1.1, 1.1], color=guide_color, linewidth=0.8, zorder=1)
    ax.axhline(0.0, color=guide_color, linewidth=0.6, linestyle=":", zorder=1)
    ax.axvline(0.0, color=guide_color, linewidth=0.6, linestyle=":", zorder=1)
    _draw_process_group(ax, df[df["group"].astype(str).eq("plus")], color=plus_color, marker="o", max_individual=120)
    _draw_process_group(ax, df[df["group"].astype(str).eq("minus")], color=minus_color, marker="^", max_individual=120)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.annotate("", xy=(0.82, -0.045), xytext=(0.18, -0.045), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "linewidth": 0.75, "color": COLOR_NEUTRAL}, annotation_clip=False)
    ax.text(0.5, -0.098, "Dynamic-like firing", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=COLOR_NEUTRAL, clip_on=False)
    ax.annotate("", xy=(-0.015, 0.82), xytext=(-0.015, 0.18), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "linewidth": 0.75, "color": COLOR_NEUTRAL}, annotation_clip=False)
    ax.text(-0.03, 0.5, "Dynamic-like decision", transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=5.8, color=COLOR_NEUTRAL, clip_on=False)
    handles = [
        Line2D([0], [0], color=plus_color, marker="o", markersize=5.0, linewidth=1.6, markerfacecolor=plus_color, markeredgecolor="white", label="Static to dynamic"),
        Line2D([0], [0], color=minus_color, marker="^", markersize=5.0, linewidth=1.6, markerfacecolor=minus_color, markeredgecolor="white", label="Dynamic to static"),
    ]
    legend = ax.legend(handles=handles, frameon=False, fontsize=5.0, loc="upper center", bbox_to_anchor=(0.49, 0.985), ncol=2, handlelength=0.9, columnspacing=0.35, borderaxespad=0.0)
    ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = 2
    _tidy(ax)
    ax.grid(False)
    ax.paper_fig_plot_form = "l3_accumulator_process"
    ax.paper_fig_static_dynamic_trajectory = True
    ax.paper_fig_trajectory_logic_source = "l3_accumulator_mechanism_experiment_plot"
    ax.paper_fig_mean_arrows = True
    ax.paper_fig_individual_traces = True
    ax.paper_fig_axis_direction_annotations = True
    ax.paper_fig_is_two_category_paired_recovery = False
    ax.paper_fig_forced_equal_aspect = False
    ax.paper_fig_normal_rectangular_panel = True


def render_fig4_overlap_perturbation(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if "metric" in df.columns:
        df = df[df["metric"].astype(str).eq("accuracy_drop_vs_static")]
    if "analysis_role" in df.columns:
        df = df[df["analysis_role"].astype(str).eq("condition_mean")]
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = [c for c in ["Random matched", "Non-overlap support", "Overlap support", "Dynamic"] if c in set(df["condition"].astype(str))]
    df = df[df["condition"].isin(order)].copy()
    colors = ["#8A8A8A", "#CC79A7", "#007A5A", "#009E73"][: len(order)]
    x = np.arange(len(order), dtype=float)
    means = []
    ci_half_widths = []
    for condition in order:
        vals = pd.to_numeric(df.loc[df["condition"].eq(condition), "value"], errors="coerce").dropna() * 100.0
        means.append(float(vals.mean()) if len(vals) else np.nan)
        ci_half_widths.append(_t_ci_half_width(vals))
    ax.bar(x, means, yerr=ci_half_widths, color=colors, edgecolor=COLOR_NEUTRAL, linewidth=0.55, alpha=0.72, capsize=2.0, zorder=2)
    for index, (condition, color) in enumerate(zip(order, colors)):
        vals = pd.to_numeric(df.loc[df["condition"].eq(condition), "value"], errors="coerce").dropna().to_numpy(dtype=float) * 100.0
        if vals.size == 0:
            continue
        jitter = np.linspace(-0.13, 0.13, vals.size) if vals.size > 1 else np.zeros(1)
        ax.scatter(np.full(vals.size, x[index]) + jitter, vals, s=8.0, color=color, edgecolors="white", linewidths=0.22, alpha=0.72, zorder=3)
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.7, zorder=1)
    label_map = dict(spec.get("display_labels_short") or {})
    ax.set_xticks(x, [label_map.get(item, item) for item in order], rotation=0)
    ax.tick_params(axis="x", labelsize=5.1, pad=0.4)
    ax.set_xlabel("")
    ax.set_ylabel(str(spec.get("y_axis", "Accuracy drop vs static")), labelpad=float(spec.get("y_labelpad", 1.0)))
    if spec.get("y_label_x") is not None:
        ax.yaxis.set_label_coords(float(spec.get("y_label_x")), float(spec.get("y_label_y", 0.5)))
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig4_overlap_perturbation_bar"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(df))
    ax.paper_fig_raw_point_alpha = 0.72
    ax.paper_fig_jitter_points = True
    ax.paper_fig_error_bar = "two_sided_t_95_ci_across_networks"
    ax.paper_fig_inference_unit = "independently_trained_network"
    ax.paper_fig_renderer_summarizes_row_level = False


def _draw_process_group(ax, part: pd.DataFrame, *, color: str, marker: str, max_individual: int | None = None) -> None:
    if part.empty:
        return
    if max_individual is not None and len(part) > max_individual:
        part = part.sort_values(["x0", "y0", "x1", "y1"], kind="stable").iloc[np.linspace(0, len(part) - 1, max_individual, dtype=int)]
    x0 = part["x0"].to_numpy(dtype=float)
    y0 = part["y0"].to_numpy(dtype=float)
    x1 = part["x1"].to_numpy(dtype=float)
    y1 = part["y1"].to_numpy(dtype=float)
    for sx, sy, ex, ey in zip(x0, y0, x1, y1):
        ax.plot([sx, ex], [sy, ey], color=color, alpha=0.035, linewidth=0.28, zorder=2)
    ax.scatter(x0, y0, s=5.5, marker=marker, facecolors="white", edgecolors=color, linewidths=0.3, alpha=0.16, zorder=3)
    ax.scatter(x1, y1, s=6.0, marker=marker, facecolors=color, edgecolors="white", linewidths=0.25, alpha=0.16, zorder=4)
    mean_start = (float(np.nanmean(x0)), float(np.nanmean(y0)))
    mean_end = (float(np.nanmean(x1)), float(np.nanmean(y1)))
    ax.add_patch(FancyArrowPatch(mean_start, mean_end, arrowstyle="-|>", color=color, linewidth=2.0, mutation_scale=14, alpha=0.95, shrinkA=2, shrinkB=2, zorder=6))
    ax.scatter([mean_start[0]], [mean_start[1]], s=38, marker=marker, facecolors="white", edgecolors=color, linewidths=1.1, zorder=7)
    ax.scatter([mean_end[0]], [mean_end[1]], s=44, marker=marker, facecolors=color, edgecolors="white", linewidths=0.7, zorder=8)


def _paired_or_dots(ax, df: pd.DataFrame, order: list[str], colors: list[str]) -> None:
    x_map = {name: idx for idx, name in enumerate(order)}
    if "matched_group_id" in df.columns:
        for _, part in df.groupby("matched_group_id", dropna=False):
            vals = part[part["condition"].isin(order)]
            if vals["condition"].nunique() == len(order):
                vals = vals.groupby("condition", dropna=False)["value"].mean().reindex(order)
                ax.plot([0, 1], vals.to_numpy(dtype=float), color="0.72", alpha=0.45, linewidth=0.6, zorder=1)
    for idx, condition in enumerate(order):
        vals = pd.to_numeric(df.loc[df["condition"].eq(condition), "value"], errors="coerce").dropna()
        if vals.empty:
            continue
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), idx) + jitter, vals, s=9, color=colors[idx], alpha=0.35, zorder=3)
        ax.errorbar([idx], [float(vals.mean())], yerr=[float(vals.sem()) if len(vals) > 1 else 0.0], fmt="o", color=COLOR_NEUTRAL, markerfacecolor=colors[idx], markeredgecolor="white", markersize=4.5, capsize=2.0, zorder=5)


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    for col in ("value", "x_value", "y_value", "time_ms", "similarity_bin_order"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["value"])


def _set_padded_limits(ax, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    xs = pd.to_numeric(df.get(x_col, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    ys = pd.to_numeric(df.get(y_col, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    if xs.size:
        xmin = min(0.0, float(np.nanmin(xs)))
        xmax = max(0.0, float(np.nanmax(xs)))
        pad = max(0.08, 0.15 * (xmax - xmin if xmax > xmin else 1.0))
        ax.set_xlim(xmin - pad, xmax + pad)
    if ys.size:
        ymin = min(0.0, float(np.nanmin(ys)))
        ymax = max(0.0, float(np.nanmax(ys)))
        pad = max(0.08, 0.15 * (ymax - ymin if ymax > ymin else 1.0))
        ax.set_ylim(ymin - pad, ymax + pad)


def _mean_sem(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False, sort=True)[value_col].agg(["mean", "count", "std"]).reset_index()
    grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    return grouped


def _mean_t_ci(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False, sort=True)[value_col].agg(["mean", "count", "std"]).reset_index()
    counts = grouped["count"].clip(lower=1).to_numpy(dtype=float)
    standard_error = grouped["std"].fillna(0.0).to_numpy(dtype=float) / np.sqrt(counts)
    degrees_of_freedom = np.maximum(counts.astype(int) - 1, 0)
    critical = np.where(
        degrees_of_freedom > 0,
        student_t.ppf(0.975, degrees_of_freedom),
        0.0,
    )
    ci_half_width = critical * standard_error
    grouped["ci95_low"] = grouped["mean"].to_numpy(dtype=float) - ci_half_width
    grouped["ci95_high"] = grouped["mean"].to_numpy(dtype=float) + ci_half_width
    grouped["n_networks"] = counts.astype(int)
    return grouped


def _t_ci_half_width(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size <= 1:
        return 0.0
    standard_error = float(np.std(numeric, ddof=1) / np.sqrt(numeric.size))
    return float(student_t.ppf(0.975, numeric.size - 1) * standard_error)


def _run_mode(stats: Mapping[str, Any] | None) -> str:
    return str((stats or {}).get("run_mode", ""))


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=GRID_ALPHA_SOFT, linewidth=0.35)


def _compact(ax) -> None:
    ax.tick_params(axis="both", labelsize=6.0, pad=1.0, length=1.9, width=0.58, color=COLOR_NEUTRAL)
    ax.xaxis.label.set_size(6.4)
    ax.yaxis.label.set_size(6.4)
    ax.xaxis.labelpad = 0.8
    ax.yaxis.labelpad = 0.5
