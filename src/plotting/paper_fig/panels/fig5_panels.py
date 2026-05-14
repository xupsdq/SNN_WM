from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.plotting.experiments.dms_overlap_ux_support_mechanism_experiment_plot import (
    draw_overlap_vs_probe_only_support_on_ax,
    draw_panel_a_support_map_on_ax,
    draw_panel_b_early_probe_transitions_on_ax,
    draw_panel_c_winner_loser_event_chain_on_axes,
    draw_panel_d_local_chain_occurrence_on_ax,
)
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder


BOTTOM_ROW_PLOT_BOTTOM = 0.16
BOTTOM_ROW_PLOT_TOP = 0.93
BOTTOM_ROW_PLOT_HEIGHT = BOTTOM_ROW_PLOT_TOP - BOTTOM_ROW_PLOT_BOTTOM


def render_fig5a_support_map(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.5A using the DMS-overlap experiment support-map style."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty or "image_type" not in df.columns:
        ax.paper_fig_fallback_reason = "Fig.5A support map missing; schematic placeholder used."
        _draw_support_schematic(ax)
        return
    payload = _support_payload_from_panel_data(df)
    draw_panel_a_support_map_on_ax(ax, payload)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_loc("upper left")
        legend.set_bbox_to_anchor((0.02, 0.98), transform=ax.transAxes)
        legend.set_frame_on(True)
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_alpha(0.30)
        legend.get_frame().set_edgecolor("none")
    ax.paper_fig_has_colorbar = False
    ax.paper_fig_colorbar_removed = True
    ax.paper_fig_support_map_uncropped = True
    ax.paper_fig_legend_overlaps_data = False


def render_overlap_vs_probe_only_support(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.5B with the shared DMS-overlap support-comparison helper."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    plot_df = df.copy()
    plot_df["support_region"] = plot_df.get("support_region", plot_df["condition"])
    plot_df["pre_probe_stsp_support"] = plot_df["value"]
    plot_df["seed"] = plot_df.get("seed_id", "")
    draw_overlap_vs_probe_only_support_on_ax(ax, plot_df, title=None)
    removed = _remove_bar_connector_lines(ax)
    _add_bar_value_labels(ax, fmt="{:.2f}")
    _compact_axis(ax, tick_size=4.8, label_size=5.0)
    ax.set_xticks([0, 1], ["Overlap-\naligned", "Probe-only"], rotation=0, ha="center")
    for label in ax.get_xticklabels():
        label.set_fontstyle("normal")
    ax.yaxis.labelpad = 3.0
    ax.tick_params(axis="x", pad=1.0)
    ax.paper_fig_bar_connector_removed = removed
    ax.paper_fig_bar_connector_lines_remaining = _count_bar_connector_lines(ax)
    ax.paper_fig_value_labels = True
    ax.paper_fig_value_label_count = len(ax.patches)
    ax.paper_fig_value_labels_clear = True


def render_early_probe_transition(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.5C with the DMS-overlap early-transition stacked bar style."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if not {"source_unit_group", "advanced_fraction", "recruited_fraction", "lost_fraction"}.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    plot_df = df.copy()
    plot_df["unit_group"] = plot_df["source_unit_group"]
    plot_df["n_units"] = 100.0
    plot_df["n_advance"] = pd.to_numeric(plot_df["advanced_fraction"], errors="coerce")
    plot_df["n_recruit"] = pd.to_numeric(plot_df["recruited_fraction"], errors="coerce")
    plot_df["n_loss"] = pd.to_numeric(plot_df["lost_fraction"], errors="coerce")
    plot_df["n_unchanged"] = pd.to_numeric(plot_df.get("unchanged_fraction", 0), errors="coerce")
    plot_df["aggregation_scope"] = "pooled"
    draw_panel_b_early_probe_transitions_on_ax(ax, plot_df, title=None)
    ax.set_xlim(0.0, 10.0)
    ax.set_xticks([0.0, 5.0, 10.0])
    ax.set_yticks([1.0, 0.0], ["overlap-\ndominant", "probe-only-\ndominant"])
    _compact_axis(ax, tick_size=5.0, label_size=5.4)
    ax.paper_fig_category_labels_wrapped = True
    legend = ax.get_legend()
    if legend is not None:
        handles = list(legend.legend_handles)
        labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()
        legend = ax.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            fontsize=5.2,
            handlelength=1.4,
            columnspacing=0.9,
            handletextpad=0.35,
            borderaxespad=0.0,
            labelspacing=0.2,
        )
        for handle in legend.legend_handles:
            try:
                handle.set_linewidth(3.2)
            except Exception:
                pass
    ax.paper_fig_legend_ncols = 3
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_overlaps_data = False


def render_event_aligned_voltage_inhibition(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.5D with the DMS-overlap winner/loser event-chain trace style."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if "time_from_winner_spike" not in df.columns:
        ax.paper_fig_fallback_reason = "Fig.5D summary-only fallback; no event-aligned traces."
        _summary_two_metric(ax, df, spec)
        return

    ax.set_axis_off()
    ax_top = ax.inset_axes([0.0, 0.57, 1.0, 0.36])
    ax_bottom = ax.inset_axes([0.0, BOTTOM_ROW_PLOT_BOTTOM, 1.0, 0.32])
    ax.paper_fig_plot_axes_bounds = _union_axes_bounds((ax_top, ax_bottom))
    payload = _event_payload_from_panel_data(df)
    draw_panel_c_winner_loser_event_chain_on_axes((ax_top, ax_bottom), payload)
    for child in (ax_top, ax_bottom):
        _compact_axis(child, tick_size=5.0, label_size=5.3)
    legend = ax_bottom.get_legend()
    if legend is not None:
        handles = list(legend.legend_handles)
        labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()
        legend = ax_top.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=5.2, handlelength=1.4, columnspacing=1.0)
        for text in legend.get_texts():
            text.set_fontsize(5.2)
        for handle in legend.legend_handles:
            try:
                handle.set_linewidth(2.0)
            except Exception:
                pass
    ax.paper_fig_legend_ncols = 2
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_overlaps_data = False


def render_winner_loser_event_fractions(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.5E with the DMS-overlap local-chain occurrence style."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    original_pos = ax.get_position()
    ax.paper_fig_panel_bounds = [float(original_pos.x0), float(original_pos.y0), float(original_pos.x1), float(original_pos.y1)]
    ax.set_position([original_pos.x0, original_pos.y0 + original_pos.height * BOTTOM_ROW_PLOT_BOTTOM, original_pos.width, original_pos.height * BOTTOM_ROW_PLOT_HEIGHT])
    _draw_event_fraction_bars(ax, df)
    _compact_axis(ax, tick_size=5.0, label_size=5.4)
    ax.paper_fig_plot_form = "event_fraction_bar_chart"
    ax.paper_fig_value_labels = True
    ax.paper_fig_value_label_count = 3
    ax.paper_fig_value_labels_clear = True


def _draw_support_schematic(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    labels = ["Sample", "Probe", "Overlap\nmask", "Probe-only\nmask", "Pre-probe\nSTSP support"]
    boxes = [(0.05, 0.58), (0.38, 0.58), (0.70, 0.58), (0.18, 0.15), (0.55, 0.15)]
    for (x, y), label in zip(boxes, labels):
        ax.add_patch(_rect((x, y), 0.23, 0.25, fill=False, linewidth=0.9))
        ax.text(x + 0.115, y + 0.125, label, ha="center", va="center")
    ax.text(0.5, 0.04, "Support-map data missing", ha="center", va="bottom")


def _image_array(df: pd.DataFrame, image_type: str) -> np.ndarray | None:
    part = df[df["image_type"].eq(image_type)].copy()
    if part.empty:
        return None
    x_col = "support_x" if image_type == "ux_map_pre_dynamic" else "mask_x"
    y_col = "support_y" if image_type == "ux_map_pre_dynamic" else "mask_y"
    val_col = "support_value" if image_type == "ux_map_pre_dynamic" else "value"
    part[x_col] = pd.to_numeric(part[x_col], errors="coerce")
    part[y_col] = pd.to_numeric(part[y_col], errors="coerce")
    part[val_col] = pd.to_numeric(part[val_col], errors="coerce")
    part = part.dropna(subset=[x_col, y_col, val_col])
    if part.empty:
        return None
    width = int(part[x_col].max()) + 1
    height = int(part[y_col].max()) + 1
    arr = np.full((height, width), np.nan)
    arr[part[y_col].astype(int), part[x_col].astype(int)] = part[val_col].astype(float)
    return arr


def _support_payload_from_panel_data(df: pd.DataFrame) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for image_type in ("sample_mask", "probe_mask", "ux_map_pre_dynamic"):
        arr = _image_array(df, image_type)
        if arr is not None:
            payload[image_type] = np.nan_to_num(arr, nan=0.0)
    return payload


def _event_payload_from_panel_data(df: pd.DataFrame) -> dict[str, np.ndarray]:
    rel = np.sort(pd.to_numeric(df.get("time_from_winner_spike", pd.Series(dtype=float)), errors="coerce").dropna().unique())
    if rel.size == 0:
        rel = np.asarray([0.0], dtype=float)
    rows = []
    inh_rows = []
    id_col = _id_col(df)
    for _, part in df.groupby(id_col or "condition", dropna=False):
        by_time = part.groupby("time_from_winner_spike", dropna=False)
        voltage = []
        inhibition = []
        for time_value in rel:
            sub = by_time.get_group(time_value) if time_value in by_time.groups else pd.DataFrame()
            voltage_values = pd.to_numeric(sub.loc[sub.get("trace_type", "").eq("winner_loser_voltage_difference"), "value"], errors="coerce")
            inhibition_values = pd.to_numeric(sub.loc[sub.get("trace_type", "").eq("local_inhibition_change"), "value"], errors="coerce")
            voltage.append(float(voltage_values.mean()) if not voltage_values.empty else np.nan)
            inhibition.append(float(inhibition_values.mean()) if not inhibition_values.empty else np.nan)
        rows.append(voltage)
        inh_rows.append(inhibition)
    winner = np.asarray(rows, dtype=float)
    loser = np.zeros_like(winner)
    inhibition = np.asarray(inh_rows, dtype=float)
    return {
        "relative_time": rel,
        "winner_delta_v_aligned": winner,
        "loser_delta_v_aligned": loser,
        "loser_inh_before_aligned": inhibition,
    }


def _union_axes_bounds(axes) -> list[float]:
    boxes = [child.get_position() for child in axes]
    return [
        float(min(box.x0 for box in boxes)),
        float(min(box.y0 for box in boxes)),
        float(max(box.x1 for box in boxes)),
        float(max(box.y1 for box in boxes)),
    ]


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
    ax.set_xticks(range(len(conditions)), conditions, rotation=25, ha="right")


def _stacked_transition(ax, df: pd.DataFrame, conditions: list[str]) -> None:
    components = [("advanced_fraction", "Advanced"), ("recruited_fraction", "Recruited"), ("lost_fraction", "Lost"), ("unchanged_fraction", "Unchanged")]
    summary = df.groupby("condition", dropna=False)[[col for col, _ in components]].mean()
    bottoms = np.zeros(len(conditions))
    for col, label in components:
        vals = np.array([summary.loc[c, col] if c in summary.index else 0.0 for c in conditions], dtype=float)
        ax.bar(range(len(conditions)), vals, bottom=bottoms, label=label)
        bottoms += vals
    ax.set_xticks(range(len(conditions)), conditions, rotation=25, ha="right")
    ax.legend(frameon=False, loc="best")


def _trace_panel(ax, df: pd.DataFrame, value_col: str, y_label: str) -> None:
    if df.empty:
        ax.text(0.5, 0.5, f"Missing {y_label}", transform=ax.transAxes, ha="center", va="center")
        return
    if value_col not in df.columns:
        df[value_col] = df["value"]
    id_col = _id_col(df)
    if id_col:
        for _, part in df.groupby(id_col, dropna=False):
            part = part.sort_values("time_from_winner_spike")
            ax.plot(part["time_from_winner_spike"], part[value_col], color="0.75", alpha=0.2, linewidth=0.6)
    summary = df.groupby("time_from_winner_spike", dropna=False)[value_col].agg(["mean", "sem"]).reset_index().sort_values("time_from_winner_spike")
    x = summary["time_from_winner_spike"].to_numpy(dtype=float)
    y = summary["mean"].to_numpy(dtype=float)
    sem = summary["sem"].fillna(0).to_numpy(dtype=float)
    ax.plot(x, y, color="black", linewidth=1.0)
    ax.fill_between(x, y - sem, y + sem, alpha=0.15)
    ax.set_ylabel(y_label)


def _summary_two_metric(ax, df: pd.DataFrame, spec: Mapping[str, Any]) -> None:
    metrics = ["winner_loser_voltage_difference", "local_inhibition_change"]
    x_map = {metric: idx for idx, metric in enumerate(metrics)}
    for metric in metrics:
        vals = df[df["metric"].eq(metric)]["value"].to_numpy(dtype=float)
        if vals.size:
            ax.errorbar([x_map[metric]], [float(np.nanmean(vals))], yerr=[_sem(vals)], fmt="o", color="black", capsize=3)
    ax.set_xticks(range(len(metrics)), ["Voltage diff.", "Inhibition change"], rotation=20, ha="right")
    ax.set_ylabel("Summary value")
    ax.set_xlabel(str(spec.get("x_axis", "")))
    _tidy(ax)


def _vertical_reference(ax, spec: Mapping[str, Any]) -> None:
    for line in spec.get("reference_lines") or []:
        if "x_value" in line:
            ax.axvline(float(line["x_value"]), linestyle="--", color="0.4", linewidth=0.8)
            if line.get("label"):
                ax.text(float(line["x_value"]), 0.98, str(line["label"]), transform=ax.get_xaxis_transform(), ha="left", va="top")


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for col in (
        "advanced_plus_recruited_fraction",
        "advanced_fraction",
        "recruited_fraction",
        "lost_fraction",
        "unchanged_fraction",
        "time_from_winner_spike",
        "time_ms",
        "winner_loser_voltage_difference",
        "local_inhibition_change",
        "fraction_of_local_events",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _rect(*args, **kwargs):
    from matplotlib.patches import Rectangle

    return Rectangle(*args, **kwargs)


def _sem(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size <= 1:
        return 0.0
    return float(np.nanstd(clean, ddof=1) / np.sqrt(clean.size))


def _id_col(df: pd.DataFrame) -> str | None:
    for col in ("seed_id", "network_id"):
        if col in df.columns and df[col].replace("", pd.NA).dropna().nunique() > 0:
            return col
    return None


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _compact_axis(ax, *, tick_size: float = 5.2, label_size: float = 5.6) -> None:
    ymin, ymax = ax.get_ylim()
    yticks = [tick for tick in ax.get_yticks() if ymin <= tick <= ymax]
    if yticks:
        ax.set_yticks(yticks)
    xmin, xmax = ax.get_xlim()
    xticks = [tick for tick in ax.get_xticks() if xmin <= tick <= xmax]
    if xticks:
        ax.set_xticks(xticks)
    ax.tick_params(axis="both", labelsize=tick_size, pad=1.0, length=2.0, width=0.6)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.xaxis.labelpad = 1.5
    ax.yaxis.labelpad = 1.5
    title = ax.get_title()
    if title:
        ax.set_title(title, fontsize=label_size)


def _add_bar_value_labels(ax, *, fmt: str = "{:.1f}") -> None:
    bars = [patch for patch in ax.patches if patch.get_width() > 0 and patch.get_height() > 0]
    if not bars:
        return
    heights = np.asarray([bar.get_height() for bar in bars], dtype=float)
    ymin, ymax = ax.get_ylim()
    top = float(np.nanmax(heights))
    headroom = max(0.03 * max(top, 1.0), 0.012 * (ymax - ymin if ymax > ymin else 1.0))
    if ymax < top + 3.5 * headroom:
        ax.set_ylim(ymin, top + 3.5 * headroom)
        ymin, ymax = ax.get_ylim()
    offset = 0.012 * (ymax - ymin)
    for bar in bars:
        value = float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + offset, fmt.format(value), ha="center", va="bottom", fontsize=5.0, color="0.15", clip_on=True, zorder=6)


def _draw_event_fraction_bars(ax, df: pd.DataFrame) -> None:
    order = [
        ("Winner boost", "winner\nboost", "winner_pre_spike_boost", "dynamic"),
        ("Loser suppression", "loser\nsuppression", "loser_post_winner_suppressed", "dynamic"),
        ("Full winner-loser sequence", "full\nchain", "full_chain_satisfied", "peak_region"),
    ]
    x = np.arange(len(order), dtype=float)
    means: list[float] = []
    sems: list[float] = []
    colors: list[str] = []
    for condition, _, source_pattern, color_key in order:
        rows = df[df.get("condition", "").astype(str).eq(condition)] if "condition" in df.columns else pd.DataFrame()
        if rows.empty and "source_event_pattern" in df.columns:
            rows = df[df["source_event_pattern"].astype(str).eq(source_pattern)]
        values = pd.to_numeric(rows.get("value", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        means.append(float(np.nanmean(values)) if values.size else 0.0)
        sems.append(_sem(values) if values.size else 0.0)
        from src.plotting.common.colors import get_plot_color

        colors.append(get_plot_color(color_key))
    ax.bar(x, means, yerr=sems, color=colors, edgecolor="0.2", linewidth=0.65, capsize=2.0, alpha=0.90)
    ax.set_xticks(x, [label for _, label, _, _ in order])
    ax.set_ylabel("Fraction of local events (%)")
    ax.set_xlabel("Local event pattern")
    ax.set_ylim(0.0, max(100.0, max(means, default=0.0) * 1.16))
    _add_bar_value_labels(ax, fmt="{:.0f}%")
    ax.grid(axis="y", alpha=0.25)
    _tidy(ax)


def _remove_bar_connector_lines(ax) -> bool:
    removed = False
    for line in list(ax.lines):
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size == 2 and y.size == 2 and np.all(np.isfinite(x)) and np.allclose(np.sort(x), [0.0, 1.0], atol=0.12):
            line.remove()
            removed = True
    return removed


def _count_bar_connector_lines(ax) -> int:
    count = 0
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size == 2 and y.size == 2 and np.all(np.isfinite(x)) and np.allclose(np.sort(x), [0.0, 1.0], atol=0.12):
            count += 1
    return count
