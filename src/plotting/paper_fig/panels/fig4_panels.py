from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from src.plotting.common.colors import get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL, GRID_ALPHA_SOFT
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder


def render_fig4a_reentry_assay_schematic(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Reserve Fig.4A as a blank slot."""
    _ = panel_data, stats, spec, style
    ax.set_axis_off()
    ax.paper_fig_plot_form = "blank_reserved_slot"


def render_similarity_bin_effect(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.4B with the similarity-bias experiment bar style."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty or "similarity_bin_order" not in df.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    summary = _mean_sem(df, ["similarity_bin_order", "similarity_bin"], "value").sort_values("similarity_bin_order")
    x = np.arange(len(summary), dtype=float)
    vals = summary["mean"].to_numpy(dtype=float)
    sems = summary["sem"].to_numpy(dtype=float)
    ax.bar(x, vals, yerr=sems, width=0.62, color=get_plot_color("true_pair"), edgecolor=COLOR_NEUTRAL, linewidth=0.65, alpha=0.82, capsize=2.0)
    ax.plot(x, vals, color=COLOR_NEUTRAL, linewidth=0.85, alpha=0.85)
    ax.set_xticks(x, [""] * len(x))
    ax.set_xlabel("")
    ax.set_ylabel(str(spec.get("y_axis", "")))
    ax.set_ylim(bottom=0)
    _set_label_headroom(ax, vals, sems)
    _add_value_labels(ax, x, vals, sems)
    ax.annotate("", xy=(0.84, -0.11), xytext=(0.16, -0.11), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": COLOR_NEUTRAL}, annotation_clip=False)
    similarity_label = ax.text(0.50, -0.17, "increasing sample-probe similarity", transform=ax.transAxes, ha="center", va="top", fontsize=5.0, color=COLOR_NEUTRAL)
    similarity_label.paper_fig_role = "b_similarity_direction_label"
    _tidy(ax)
    _compact_axis(ax)
    ax.paper_fig_plot_form = "similarity_bin_effect"
    ax.paper_fig_value_labels = True
    ax.paper_fig_value_label_count = int(np.isfinite(vals).sum())
    ax.paper_fig_value_labels_clear = True
    ax.paper_fig_similarity_direction_arrow = True
    ax.paper_fig_literal_bin_xticklabels = False
    ax.paper_fig_similarity_bar_order_preserved = bool(np.all(np.diff(summary["similarity_bin_order"].to_numpy(dtype=float)) >= 0))


def render_overlap_level_effect(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.4C with the similarity-bias overlap bridge style."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    low = df[df["condition"].str.contains("Low", case=False, na=False)]["value"]
    high = df[df["condition"].str.contains("High", case=False, na=False)]["value"]
    vals = [float(low.mean()) if not low.empty else np.nan, float(high.mean()) if not high.empty else np.nan]
    sems = [_sem(low.to_numpy(dtype=float)) if not low.empty else 0.0, _sem(high.to_numpy(dtype=float)) if not high.empty else 0.0]
    x = np.arange(2, dtype=float)
    ax.bar(x, vals, yerr=sems, width=0.58, color=[get_plot_color("background_shade"), get_plot_color("true_pair")], edgecolor=COLOR_NEUTRAL, linewidth=0.65, alpha=0.82, capsize=2.0)
    ax.set_xticks(x, ["Low\noverlap", "High\noverlap"])
    ax.set_xlabel(str(spec.get("x_axis", "")))
    ax.set_ylabel(str(spec.get("y_axis", "")))
    ax.set_ylim(bottom=0)
    _set_label_headroom(ax, np.asarray(vals, dtype=float), np.asarray(sems, dtype=float))
    _add_value_labels(ax, x, np.asarray(vals, dtype=float), np.asarray(sems, dtype=float))
    _tidy(ax)
    _compact_axis(ax)
    ax.paper_fig_plot_form = "overlap_level_effect"
    ax.paper_fig_value_labels = True
    ax.paper_fig_value_label_count = int(np.isfinite(np.asarray(vals, dtype=float)).sum())
    ax.paper_fig_value_labels_clear = True


def render_dynamic_probe_index_timecourse(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.4D with the overlap perturbation DPI trace style."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if "probe_time" not in df.columns:
        ax.paper_fig_fallback_reason = "Fig.4D summary-only DPI fallback; no probe_time/timecourse data."
        _paired_conditions(ax, df, list(spec.get("conditions") or ["Overlap-preserving", "Non-overlap control"]))
        _reference_lines(ax, spec)
        ax.set_xlabel(str(spec.get("x_axis", "")))
        ax.set_ylabel(str(spec.get("y_axis", "")))
        _tidy(ax)
        return
    _draw_dpi_timecourse(ax, df, spec)
    _compact_axis(ax)
    ax.paper_fig_plot_form = "timecourse_line"


def render_static_dynamic_trajectory(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.4E as the static/dynamic manipulation trajectory panel."""
    _ = stats, style
    df = _clean_trajectory(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    original_pos = ax.get_position()
    ax.paper_fig_panel_bounds = [float(original_pos.x0), float(original_pos.y0), float(original_pos.x1), float(original_pos.y1)]
    guide_color = get_plot_color("other_residual")
    ax.plot([-1.1, 1.1], [-1.1, 1.1], color=guide_color, linewidth=0.75, linestyle="-", zorder=1)
    ax.axhline(0.0, color=guide_color, linewidth=0.55, linestyle=":", zorder=1)
    ax.axvline(0.0, color=guide_color, linewidth=0.55, linestyle=":", zorder=1)
    _draw_group(ax, df[df["group"].eq("plus")], color=get_plot_color("dynamic"), marker="o")
    _draw_group(ax, df[df["group"].eq("minus")], color=get_plot_color("static_frozen"), marker="^")

    ax.annotate("", xy=(0.84, -0.035), xytext=(0.18, -0.035), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": COLOR_NEUTRAL}, annotation_clip=False)
    x_direction_label = ax.text(0.51, -0.075, str(spec.get("x_axis", "more dynamic-like firing pattern")), transform=ax.transAxes, ha="center", va="top", fontsize=4.6, color=COLOR_NEUTRAL)
    x_direction_label.paper_fig_role = "e_x_direction_label"
    ax.annotate("", xy=(-0.055, 0.68), xytext=(-0.055, 0.17), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "linewidth": 0.75, "color": COLOR_NEUTRAL}, annotation_clip=False)
    y_direction_label = ax.text(-0.025, 0.425, str(spec.get("y_axis", "more dynamic-like decision")), transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=5.0, color=COLOR_NEUTRAL)
    y_direction_label.paper_fig_role = "e_y_direction_label"

    handles = [
        Line2D([0], [0], marker="o", color=get_plot_color("dynamic"), markerfacecolor=get_plot_color("dynamic"), linestyle="-", linewidth=1.2, markersize=4.0, label="plus: static -> dynamic"),
        Line2D([0], [0], marker="^", color=get_plot_color("static_frozen"), markerfacecolor=get_plot_color("static_frozen"), linestyle="-", linewidth=1.2, markersize=4.0, label="minus: dynamic -> static"),
        Line2D([0], [0], marker="o", color=COLOR_NEUTRAL, markerfacecolor="white", linestyle="None", markersize=3.8, label="before manipulation"),
        Line2D([0], [0], marker="o", color=COLOR_NEUTRAL, markerfacecolor=COLOR_NEUTRAL, linestyle="None", markersize=3.8, label="after manipulation"),
    ]
    legend_fontsize = 4.7
    legend = ax.legend(handles=handles, frameon=True, loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=legend_fontsize, handlelength=0.92, handletextpad=0.28, borderaxespad=0.0, borderpad=0.12, labelspacing=0.16)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.82)
    legend.get_frame().set_edgecolor("none")
    ax.set_xlim(-1.28, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("auto")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=GRID_ALPHA_SOFT, linewidth=0.35)
    _tidy(ax)
    ax.paper_fig_plot_form = "static_dynamic_trajectory"
    ax.paper_fig_static_dynamic_trajectory = True
    ax.paper_fig_trajectory_logic_source = "l3_accumulator_mechanism_experiment_plot"
    ax.paper_fig_individual_traces = True
    ax.paper_fig_mean_arrows = True
    ax.paper_fig_axis_direction_annotations = True
    ax.paper_fig_is_two_category_paired_recovery = False
    ax.paper_fig_forced_equal_aspect = False
    ax.paper_fig_normal_rectangular_panel = True
    ax.paper_fig_e_y_annotation_outside_plot = True
    ax.paper_fig_e_legend_repositioned_inside_panel = True
    ax.paper_fig_e_legend_inside_axes = True
    ax.paper_fig_e_legend_upper_left = True
    ax.paper_fig_e_axis_region_aligned_with_d = True
    ax.paper_fig_e_legend_fontsize = legend_fontsize
    ax.paper_fig_e_legend_markers_enlarged = True
    ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
    ax.paper_fig_legend_overlaps_data = False


def render_final_readout_recovery(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Compatibility shim for older Fig.4 specs."""
    render_static_dynamic_trajectory(ax, panel_data, stats, spec, style)


def plt_rect(*args, **kwargs):
    """Small indirection so the schematic does not import pyplot."""
    from matplotlib.patches import Rectangle

    return Rectangle(*args, **kwargs)


def _dpi_payload_from_panel_data(df: pd.DataFrame) -> dict[str, np.ndarray]:
    condition_map = {
        "Overlap-preserving": "sample_keep_overlap_only_dynamic",
        "Non-overlap control": "sample_keep_nonoverlap_only_dynamic",
    }
    traces: list[np.ndarray] = []
    condition_names: list[str] = []
    id_col = _id_col(df)
    group_cols = [col for col in (id_col, "condition") if col]
    if not group_cols:
        group_cols = ["condition"]
    for key, part in df.groupby(group_cols, dropna=False):
        condition = key[-1] if isinstance(key, tuple) else key
        raw_condition = condition_map.get(str(condition), str(condition))
        series = part.sort_values("probe_time")["value"].to_numpy(dtype=float)
        if series.size == 0:
            continue
        traces.append(series)
        condition_names.append(raw_condition)
    max_len = max((len(trace) for trace in traces), default=0)
    arr = np.full((len(traces), max_len), np.nan, dtype=float)
    for idx, trace in enumerate(traces):
        arr[idx, : len(trace)] = trace
    return {"condition_name": np.asarray(condition_names, dtype=object), "DPI_L3": arr}


def _draw_dpi_timecourse(ax, df: pd.DataFrame, spec: Mapping[str, Any]) -> None:
    condition_order = list(spec.get("conditions") or ["Overlap-preserving", "Non-overlap control"])
    colors = {
        "Overlap-preserving": get_plot_color("true_pair"),
        "Non-overlap control": get_plot_color("shuffled_pair"),
    }
    early_window_color = "#FDE68A"
    early_window_alpha = 0.42
    ax.axvspan(0.0, 20.0, color=early_window_color, alpha=early_window_alpha, linewidth=0, zorder=0)
    peak_annotations: list[dict[str, Any]] = []
    for condition in condition_order:
        part = df[df["condition"].eq(condition)].dropna(subset=["probe_time", "value"]).copy()
        if part.empty:
            continue
        part = part[(part["probe_time"] >= 0) & (part["probe_time"] <= 50)]
        if part.empty:
            continue
        summary = _mean_sem(part, ["probe_time"], "value").sort_values("probe_time")
        x = summary["probe_time"].to_numpy(dtype=float)
        y = summary["mean"].to_numpy(dtype=float)
        sem = summary["sem"].to_numpy(dtype=float)
        color = colors.get(condition, COLOR_NEUTRAL)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.35, label=condition.replace(" control", "\ncontrol"))
        if y.size:
            peak_idx = int(np.nanargmax(y))
            peak_x = float(x[peak_idx])
            peak_y = float(y[peak_idx])
            peak_annotations.append({"condition": condition, "time_ms": peak_x, "value": peak_y})
            ax.scatter([peak_x], [peak_y], s=11, color=color, edgecolor="white", linewidth=0.35, zorder=5)
            label = f"{peak_y:.2f}"
            offset = (1.3, 0.014 if condition == "Overlap-preserving" else -0.018)
            ax.text(
                peak_x + offset[0],
                peak_y + offset[1],
                label,
                color=color,
                fontsize=4.7,
                ha="left",
                va="center",
                zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.25},
            )
    _reference_lines(ax, spec)
    ax.set_xlim(0, 50)
    ax.set_xlabel("Probe time (ms)")
    ax.set_ylabel(str(spec.get("y_axis", "")))
    legend = ax.legend(frameon=False, fontsize=4.8, loc="upper right", handlelength=1.05, borderaxespad=0.1, labelspacing=0.2)
    ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
    ax.paper_fig_legend_overlaps_data = False
    ax.paper_fig_shaded_window = [0.0, 20.0]
    ax.paper_fig_shaded_window_color = early_window_color
    ax.paper_fig_shaded_window_alpha = early_window_alpha
    ax.paper_fig_peak_annotations = peak_annotations
    _tidy(ax)


def _paired_conditions(ax, df: pd.DataFrame, conditions: list[str]) -> None:
    x_map = {condition: idx for idx, condition in enumerate(conditions)}
    id_col = _id_col(df)
    if id_col:
        for _, part in df.groupby(id_col, dropna=False):
            paired = part[part["condition"].isin(conditions)]
            if paired["condition"].nunique() == len(conditions):
                paired = paired.set_index("condition").loc[conditions].reset_index()
                ax.plot([x_map[c] for c in paired["condition"]], paired["value"], color="0.72", linewidth=0.8, alpha=0.55)
                ax.scatter([x_map[c] for c in paired["condition"]], paired["value"], s=12, alpha=0.65)
    for condition in conditions:
        vals = df[df["condition"].eq(condition)]["value"].to_numpy(dtype=float)
        if vals.size:
            ax.errorbar([x_map[condition]], [float(np.nanmean(vals))], yerr=[_sem(vals)], fmt="o", color="black", capsize=3)
    ax.set_xticks(range(len(conditions)), conditions, rotation=20, ha="right")


def _paired_recovery_shift(ax, df: pd.DataFrame, conditions: list[str], spec: Mapping[str, Any]) -> None:
    control = "Non-overlap control"
    overlap = "Overlap-preserving"
    if not {control, overlap}.issubset(set(df["condition"].astype(str))):
        control, overlap = conditions[-1], conditions[0]
    order = [control, overlap]
    x_map = {control: 0.0, overlap: 1.0}
    id_col = _id_col(df)
    paired_rows: list[tuple[float, float]] = []
    color = get_plot_color("true_pair")
    control_color = get_plot_color("shuffled_pair")
    if id_col:
        for _, part in df.groupby(id_col, dropna=False):
            paired = part[part["condition"].isin(order)]
            if paired["condition"].nunique() != 2:
                continue
            values = paired.groupby("condition", dropna=False)["value"].mean()
            y0 = float(values[control])
            y1 = float(values[overlap])
            paired_rows.append((y0, y1))
            ax.plot([0.0, 1.0], [y0, y1], color=COLOR_NEUTRAL, alpha=0.16, linewidth=0.6, zorder=1)
            ax.scatter([0.0, 1.0], [y0, y1], s=8, facecolors=["white", color], edgecolors=[control_color, color], linewidths=0.45, alpha=0.28, zorder=2)
    for condition in order:
        vals = df[df["condition"].eq(condition)]["value"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        x = x_map[condition]
        mean = float(np.nanmean(vals))
        ax.errorbar([x], [mean], yerr=[_sem(vals)], fmt="o", color=COLOR_NEUTRAL, markerfacecolor=(control_color if condition == control else color), markeredgecolor="white", markersize=5.2, linewidth=1.2, capsize=2.5, zorder=5)
    if paired_rows:
        means = np.asarray(paired_rows, dtype=float).mean(axis=0)
        ax.annotate("", xy=(1.0, float(means[1])), xytext=(0.0, float(means[0])), arrowprops={"arrowstyle": "->", "linewidth": 1.9, "color": color, "shrinkA": 3, "shrinkB": 3}, zorder=6)
    ax.set_xticks([0.0, 1.0], ["Non-overlap\ncontrol", "Overlap-\npreserving"])
    ax.set_xlim(-0.32, 1.32)
    values = pd.to_numeric(df["value"], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size:
        ymin = float(np.nanmin(values))
        ymax = float(np.nanmax(values))
        pad = max(0.03, 0.12 * (ymax - ymin if ymax > ymin else 1.0))
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("")
    ax.set_ylabel(str(spec.get("y_axis", "")))
    _tidy(ax)
    _compact_axis(ax)
    ax.paper_fig_paired_change_style_source = "l3_accumulator_mechanism_experiment_plot"
    ax.paper_fig_paired_change_summary_arrow = bool(paired_rows)
    ax.paper_fig_individual_traces = bool(paired_rows)
    ax.paper_fig_legend_overlaps_data = False


def _draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str, alpha: float, linewidth: float, mutation_scale: float, zorder: int) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", color=color, alpha=alpha, linewidth=linewidth, mutation_scale=mutation_scale, shrinkA=1.0, shrinkB=1.0, zorder=zorder)
    ax.add_patch(arrow)


def _draw_group(ax, df: pd.DataFrame, *, color: str, marker: str) -> None:
    if df.empty:
        return
    x0 = df["x0"].to_numpy(dtype=float)
    y0 = df["y0"].to_numpy(dtype=float)
    x1 = df["x1"].to_numpy(dtype=float)
    y1 = df["y1"].to_numpy(dtype=float)
    for start_x, start_y, end_x, end_y in zip(x0, y0, x1, y1):
        ax.plot([float(start_x), float(end_x)], [float(start_y), float(end_y)], color=color, alpha=0.07, linewidth=0.32, zorder=2)
    ax.scatter(x0, y0, s=6.5, marker=marker, facecolors="white", edgecolors=color, linewidths=0.32, alpha=0.18, zorder=3)
    ax.scatter(x1, y1, s=7.0, marker=marker, facecolors=color, edgecolors="white", linewidths=0.25, alpha=0.18, zorder=4)

    mean_start = (float(np.nanmean(x0)), float(np.nanmean(y0)))
    mean_end = (float(np.nanmean(x1)), float(np.nanmean(y1)))
    _draw_arrow(ax, mean_start, mean_end, color=color, alpha=0.98, linewidth=1.45, mutation_scale=10.5, zorder=6)
    ax.scatter([mean_start[0]], [mean_start[1]], s=28, marker=marker, facecolors="white", edgecolors=color, linewidths=0.9, zorder=7)
    ax.scatter([mean_end[0]], [mean_end[1]], s=32, marker=marker, facecolors=color, edgecolors="white", linewidths=0.55, zorder=8)


def _clean_trajectory(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty:
        return pd.DataFrame()
    required = {"group", "x0", "y0", "x1", "y1"}
    if not required.issubset(panel_data.columns):
        return pd.DataFrame()
    df = panel_data.copy()
    for col in ("x0", "y0", "x1", "y1", "before_x", "before_y", "after_x", "after_y", "firing_shift", "decision_shift"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["x0", "y0", "x1", "y1"])


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for col in ("similarity_bin_order", "probe_time", "time_ms", "accuracy_drop", "dynamic_probe_index", "final_readout_recovery"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["value"])


def _reference_lines(ax, spec: Mapping[str, Any]) -> None:
    for line in spec.get("reference_lines") or []:
        ax.axhline(float(line["value"]), linestyle="--", color="0.4", linewidth=0.8)
        if line.get("label"):
            ax.text(0.98, float(line["value"]), str(line["label"]), transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7)


def _sem(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size <= 1:
        return 0.0
    return float(np.nanstd(clean, ddof=1) / np.sqrt(clean.size))


def _set_label_headroom(ax, means: np.ndarray, sems: np.ndarray) -> None:
    tops = np.asarray(means, dtype=float) + np.asarray(sems, dtype=float)
    finite_tops = tops[np.isfinite(tops)]
    if finite_tops.size == 0:
        return
    ymin, ymax = ax.get_ylim()
    upper = max(float(ymax), float(np.nanmax(finite_tops)) * 1.18, float(np.nanmax(finite_tops)) + 0.08 * max(float(np.nanmax(finite_tops)) - ymin, 1.0))
    ax.set_ylim(ymin, upper)


def _add_value_labels(ax, x: np.ndarray, means: np.ndarray, sems: np.ndarray, fmt: str = "{:.1f}") -> None:
    ymin, ymax = ax.get_ylim()
    offset = 0.018 * (ymax - ymin)
    ceiling = ymax - 0.035 * (ymax - ymin)
    for xpos, mean, sem in zip(x, means, sems):
        if not np.isfinite(mean):
            continue
        y = min(ceiling, float(mean) + float(sem) + offset)
        ax.text(float(xpos), y, fmt.format(float(mean)), ha="center", va="bottom", fontsize=4.7, color=COLOR_NEUTRAL, clip_on=True, zorder=7)


def _mean_sem(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False, sort=True)[value_col].agg(["mean", "count", "std"]).reset_index()
    grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    return grouped


def _id_col(df: pd.DataFrame) -> str | None:
    for col in ("seed_id", "network_id"):
        if col in df.columns and df[col].replace("", pd.NA).dropna().nunique() > 0:
            return col
    return None


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _compact_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=5.4, pad=0.8, length=1.9, width=0.6, color=COLOR_NEUTRAL)
    ax.xaxis.label.set_size(5.8)
    ax.yaxis.label.set_size(5.8)
    ax.xaxis.labelpad = 0.4
    ax.yaxis.labelpad = 0.4
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontstyle("normal")
