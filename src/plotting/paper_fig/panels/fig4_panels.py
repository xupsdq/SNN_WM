from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

from src.plotting.common.colors import get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL, GRID_ALPHA_SOFT
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder


def render_fig4_reentry_schematic(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data, stats, spec, style
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 0.54
    boxes = [
        (0.05, "Sample", "writes prior STSP"),
        (0.34, "Delay", "history persists"),
        (0.63, "Probe", "re-enters overlap"),
    ]
    colors = [get_plot_color("dynamic"), "#f8fafc", get_plot_color("true_pair")]
    for idx, (x, title, subtitle) in enumerate(boxes):
        ax.add_patch(Rectangle((x, y - 0.17), 0.20, 0.34, facecolor=colors[idx], edgecolor=COLOR_NEUTRAL, linewidth=0.7, alpha=0.16 if idx != 1 else 1.0))
        ax.text(x + 0.10, y + 0.05, title, ha="center", va="center", fontsize=7.2, fontweight="bold", color=COLOR_NEUTRAL)
        ax.text(x + 0.10, y - 0.07, subtitle, ha="center", va="center", fontsize=5.6, color=COLOR_NEUTRAL)
    for x0, x1 in ((0.25, 0.34), (0.54, 0.63)):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color=COLOR_NEUTRAL))
    ax.add_patch(FancyArrowPatch((0.15, 0.25), (0.73, 0.25), connectionstyle="arc3,rad=-0.12", arrowstyle="-|>", mutation_scale=9, linewidth=1.0, color=get_plot_color("true_pair")))
    ax.text(0.44, 0.13, "overlap-aligned sample support deflects later probe processing", ha="center", va="center", fontsize=6.2, color=COLOR_NEUTRAL)
    ax.add_patch(Rectangle((0.82, y - 0.13), 0.07, 0.26, facecolor=get_plot_color("true_pair"), edgecolor="none", alpha=0.22))
    ax.add_patch(Rectangle((0.88, y - 0.13), 0.07, 0.26, facecolor=get_plot_color("shuffled_pair"), edgecolor="none", alpha=0.22))
    ax.text(0.885, y + 0.18, "overlap", ha="center", va="bottom", fontsize=5.2, color=COLOR_NEUTRAL)
    ax.paper_fig_plot_form = "fig4_reentry_schematic"


def render_fig4_similarity_entry(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    df = df.sort_values(["similarity_bin_order", "seed_id"], kind="stable")
    summary = _mean_sem(df, ["similarity_bin_order", "similarity_bin"], "value").sort_values("similarity_bin_order")
    x = np.arange(len(summary), dtype=float)
    y = summary["mean"].to_numpy(dtype=float)
    sem = summary["sem"].to_numpy(dtype=float)
    ax.errorbar(x, y, yerr=sem, fmt="o-", color=get_plot_color("true_pair"), linewidth=1.15, markersize=3.6, capsize=2.0)
    if _run_mode(stats) == "single_network_draft":
        ax.scatter(np.arange(len(df)), np.interp(df["similarity_bin_order"], summary["similarity_bin_order"], y), s=4, color=COLOR_NEUTRAL, alpha=0.18)
    ax.set_xticks(x, [""] * len(x))
    ax.annotate("", xy=(0.86, -0.16), xytext=(0.14, -0.16), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": COLOR_NEUTRAL}, annotation_clip=False)
    ax.text(0.50, -0.25, "increasing similarity", transform=ax.transAxes, ha="center", va="top", fontsize=5.2, color=COLOR_NEUTRAL)
    ax.set_xlabel("")
    ax.set_ylabel(str(spec.get("y_axis", "Accuracy drop")))
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig4_similarity_entry"
    ax.paper_fig_similarity_direction_arrow = True
    ax.paper_fig_literal_bin_xticklabels = False


def render_fig4_overlap_localization(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = ["Low overlap", "High overlap"]
    colors = [get_plot_color("shuffled_pair"), get_plot_color("true_pair")]
    _paired_or_dots(ax, df, order, colors)
    ax.set_xticks([0, 1], ["Low\noverlap", "High\noverlap"])
    ax.set_xlabel("")
    ax.set_ylabel(str(spec.get("y_axis", "Accuracy drop")))
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig4_overlap_localization"


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
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if "time_ms" not in df.columns:
        render_fig4_overlap_localization(ax, panel_data, stats, spec, style)
        return
    colors = {"Overlap support": get_plot_color("true_pair"), "Non-overlap support": get_plot_color("shuffled_pair")}
    for condition in ["Overlap support", "Non-overlap support"]:
        part = df[df["condition"].eq(condition)].dropna(subset=["time_ms", "value"])
        if part.empty:
            continue
        summary = _mean_sem(part, ["time_ms"], "value").sort_values("time_ms")
        x = summary["time_ms"].to_numpy(dtype=float)
        y = summary["mean"].to_numpy(dtype=float)
        sem = summary["sem"].to_numpy(dtype=float)
        color = colors[condition]
        ax.plot(x, y, color=color, linewidth=1.15, label=condition.replace(" support", ""))
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.12, linewidth=0)
    for line in spec.get("reference_lines") or []:
        ax.axhline(float(line.get("value", 0.0)), color="0.45", linestyle="--", linewidth=0.7)
    max_time_ms = float((stats or {}).get("max_time_ms_used", spec.get("max_time_ms", 60)))
    ax.set_xlim(0.0, max_time_ms)
    ax.set_xlabel(str(spec.get("x_axis", "Probe time (ms)")))
    ax.set_ylabel(str(spec.get("y_axis", "DPI")))
    legend = ax.legend(frameon=False, fontsize=5.0, loc="best", handlelength=1.1)
    ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
    ax.paper_fig_legend_overlaps_data = False
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig4_decision_spike_displacement"


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
    _draw_process_group(ax, df[df["group"].astype(str).eq("plus")], color=plus_color, marker="o")
    _draw_process_group(ax, df[df["group"].astype(str).eq("minus")], color=minus_color, marker="^")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.annotate("", xy=(0.82, 0.04), xytext=(0.18, 0.04), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "linewidth": 0.65, "color": COLOR_NEUTRAL})
    ax.text(0.5, 0.075, "more dynamic-like firing pattern", transform=ax.transAxes, ha="center", va="bottom", fontsize=4.9, color=COLOR_NEUTRAL)
    ax.annotate("", xy=(0.04, 0.82), xytext=(0.04, 0.18), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "linewidth": 0.65, "color": COLOR_NEUTRAL})
    ax.text(0.075, 0.5, "more dynamic-like decision", transform=ax.transAxes, ha="left", va="center", rotation=90, fontsize=4.9, color=COLOR_NEUTRAL)
    ax.text(0.04, 0.95, "plus: static -> dynamic", transform=ax.transAxes, ha="left", va="top", fontsize=5.2, color=plus_color)
    ax.text(0.04, 0.86, "minus: dynamic -> static", transform=ax.transAxes, ha="left", va="top", fontsize=5.2, color=minus_color)
    ax.text(0.96, 0.08, "open before\nfilled after", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0, color=COLOR_NEUTRAL)
    _tidy(ax)
    ax.grid(False)
    ax.paper_fig_plot_form = "l3_accumulator_process"
    ax.paper_fig_static_dynamic_trajectory = True
    ax.paper_fig_trajectory_logic_source = "l3_accumulator_mechanism_experiment_plot"
    ax.paper_fig_mean_arrows = True
    ax.paper_fig_individual_traces = True
    ax.paper_fig_axis_direction_annotations = True
    ax.paper_fig_is_two_category_paired_recovery = False
    ax.paper_fig_forced_equal_aspect = True


def render_fig4_overlap_perturbation(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = ["Dynamic", "Overlap support", "Non-overlap support", "Random matched"]
    if df["condition"].astype(str).eq("Static baseline").any():
        order.append("Static baseline")
    elif df["condition"].astype(str).eq("Static").any():
        order.append("Static")
    colors = [get_plot_color("dynamic"), get_plot_color("true_pair"), get_plot_color("shuffled_pair"), get_plot_color("other_residual"), "0.70"][: len(order)]
    x = np.arange(len(order), dtype=float)
    means = []
    sems = []
    for condition in order:
        vals = pd.to_numeric(df.loc[df["condition"].eq(condition), "value"], errors="coerce").dropna()
        means.append(float(vals.mean()) if len(vals) else np.nan)
        sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)
        if len(vals):
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), x[order.index(condition)]) + jitter, vals, s=7, color=colors[order.index(condition)], alpha=0.28, zorder=3)
    ax.bar(x, means, yerr=sems, color=colors, edgecolor=COLOR_NEUTRAL, linewidth=0.55, alpha=0.72, capsize=2.0, zorder=2)
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.7, zorder=1)
    label_map = {"Dynamic": "Dyn", "Overlap support": "Overlap", "Non-overlap support": "Non", "Random matched": "Random", "Static": "Static", "Static baseline": "Static"}
    ax.set_xticks(x, [label_map.get(item, item) for item in order], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel(str(spec.get("y_axis", "Accuracy drop vs static")))
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig4_overlap_perturbation"


def _draw_process_group(ax, part: pd.DataFrame, *, color: str, marker: str) -> None:
    if part.empty:
        return
    x0 = part["x0"].to_numpy(dtype=float)
    y0 = part["y0"].to_numpy(dtype=float)
    x1 = part["x1"].to_numpy(dtype=float)
    y1 = part["y1"].to_numpy(dtype=float)
    for sx, sy, ex, ey in zip(x0, y0, x1, y1):
        ax.plot([sx, ex], [sy, ey], color=color, alpha=0.10, linewidth=0.45, zorder=2)
    ax.scatter(x0, y0, s=10, marker=marker, facecolors="white", edgecolors=color, linewidths=0.5, alpha=0.35, zorder=3)
    ax.scatter(x1, y1, s=12, marker=marker, facecolors=color, edgecolors="white", linewidths=0.35, alpha=0.36, zorder=4)
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


def _run_mode(stats: Mapping[str, Any] | None) -> str:
    return str((stats or {}).get("run_mode", ""))


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=GRID_ALPHA_SOFT, linewidth=0.35)


def _compact(ax) -> None:
    ax.tick_params(axis="both", labelsize=5.2, pad=0.8, length=1.8, width=0.55, color=COLOR_NEUTRAL)
    ax.xaxis.label.set_size(5.7)
    ax.yaxis.label.set_size(5.7)
    ax.xaxis.labelpad = 0.5
    ax.yaxis.labelpad = 0.7
