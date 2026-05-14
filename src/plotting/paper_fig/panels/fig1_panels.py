from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.plotting.common.colors import get_plot_color
from src.plotting.paper_fig.utils import paper_fig_root


SMALL_PANEL_STYLE = {
    "axis_labelsize": 6.0,
    "tick_labelsize": 5.2,
    "legend_fontsize": 5.0,
    "annotation_fontsize": 5.3,
    "marker_size": 7.0,
    "line_marker_size": 4.2,
    "mean_marker_size": 18.0,
    "line_width": 0.65,
    "tick_width": 0.55,
    "tick_length": 1.8,
    "bar_edge_width": 0.45,
    "error_line_width": 0.7,
    "capsize": 2.1,
    "labelpad": 1.2,
    "jitter_width": 0.075,
}

LAYER_COLORS = {
    "Layer 1": "#4C78A8",
    "Layer 2": "#F58518",
    "Layer 3": "#54A24B",
}

CONDITION_COLORS = {
    "STSP-SNN": "#4C78A8",
    "Dynamic STSP": "#4C78A8",
    "u/x-shuffled": "#B279A2",
    "Static-frozen": "#6B7280",
}


def render_manual_svg_panel(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render a manual schematic slot; keep the slot blank when the asset is absent."""
    _ = panel_data, stats, style
    ax.set_axis_off()
    asset = spec.get("source") or (spec.get("source_mapping") or {}).get("manual_asset")
    asset_path = paper_fig_root() / str(asset) if asset else None
    if asset_path and asset_path.exists():
        ax.text(0.5, 0.54, f"Manual asset\n{asset}", ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.18, "SVG present; final embedding/style pass pending", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return
    ax.paper_fig_plot_form = "blank_manual_slot"


def render_dot_summary(ax, panel_data: pd.DataFrame, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.1B as a compact recall fluctuation line across networks."""
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder_box(ax, spec, "No panel data available")
        return
    ax.paper_fig_plot_form = "recall_fluctuation_line"
    df = df.sort_values([col for col in ("network_id", "seed_id") if col in df.columns], kind="stable")
    values = df["value"].to_numpy(dtype=float)
    x = np.arange(1, len(values) + 1, dtype=float)
    ax.axhspan(88.0, 95.0, color=CONDITION_COLORS["STSP-SNN"], alpha=0.10, linewidth=0, zorder=0)
    ax.paper_fig_has_shaded_band = True
    ax.paper_fig_shaded_band = [88.0, 95.0]
    ax.paper_fig_line_emphasis = "line_over_points"
    ax.plot(x, values, color=CONDITION_COLORS["STSP-SNN"], linewidth=1.15, alpha=0.98, zorder=2)
    ax.scatter(x, values, s=st["line_marker_size"], facecolor="white", edgecolor="0.35", linewidth=0.28, alpha=0.72, zorder=3)
    ax.paper_fig_has_mean_marker = False
    ax.paper_fig_has_mean_annotation = False
    ax.paper_fig_y_label_inside = False
    _reference_lines(ax, spec)
    ax.set_xticks([1, max(1, int(np.ceil(len(values) / 2))), len(values)], ["1", str(max(1, int(np.ceil(len(values) / 2)))), str(len(values))])
    ax.set_xlabel("Network")
    ax.set_ylabel(str(spec.get("y_axis", "Overall recall (%)")))
    ax.set_xlim(0.5, len(values) + 0.5)
    ax.set_ylim(0.0, 102.0)
    _tidy(ax, st)


def render_layerwise_decoding(ax, panel_data: pd.DataFrame, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.1C as a layer-wise magnitude summary, not a delay curve."""
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder_box(ax, spec, "No panel data available")
        return
    ax.paper_fig_plot_form = "layer_bar_summary"
    ax.paper_fig_raw_points = False
    ax.paper_fig_value_labels = True
    order = [label for label in (spec.get("conditions") or ["Layer 1", "Layer 2", "Layer 3"]) if label in set(df["layer"].astype(str))]
    if not order:
        order = sorted(df["layer"].astype(str).unique())
    means, sems = _draw_bar_summary(
        ax,
        df,
        group_col="layer",
        order=order,
        colors=[LAYER_COLORS.get(label, "#4C78A8") for label in order],
        st=st,
        bar_width=0.58,
    )
    _reference_lines(ax, spec)
    display_labels = [str((spec.get("display_labels") or {}).get(label, label)) for label in order]
    ax.set_xticks(range(len(order)), display_labels)
    ax.set_xlabel(str(spec.get("x_axis", "")))
    ax.set_ylabel(str(spec.get("y_axis", "")))
    _set_percent_ylim(ax, df["value"].to_numpy(dtype=float), include_reference=10, top_room=8, min_upper=100)
    _add_value_labels(ax, np.arange(len(order), dtype=float), means, sems, st=st, fmt="{:.1f}")
    _tidy(ax, st)


def render_outcome_profile(ax, panel_data: pd.DataFrame, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.1D as an error-rate summary in the requested condition order."""
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or set(df["metric"].astype(str)) != {"error_rate"}:
        _placeholder_box(ax, spec, "Error-rate data unavailable")
        return
    ax.paper_fig_plot_form = "point_range"
    order = [label for label in (spec.get("conditions") or []) if label in set(df["condition"].astype(str))]
    if not order:
        order = ["Dynamic STSP", "u/x-shuffled", "Static-frozen"]
    _draw_point_range_summary(
        ax,
        df,
        group_col="condition",
        order=order,
        st=st,
    )
    display_labels = [str((spec.get("display_labels") or {}).get(label, label)) for label in order]
    ax.set_xticks(range(len(order)), display_labels, rotation=0, ha="center")
    ax.set_xlabel(str(spec.get("x_axis", "")))
    ax.set_ylabel(str(spec.get("y_axis", "")))
    _set_percent_ylim(ax, df["value"].to_numpy(dtype=float), include_reference=None, top_room=5)
    _tidy(ax, st)


def render_paired_attribution_shift(ax, panel_data: pd.DataFrame, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.1E as vertical stacked error-composition bars."""
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "trace" not in df.columns:
        _placeholder_box(ax, spec, "Error-composition data unavailable")
        return
    conditions = [label for label in (spec.get("conditions") or ["Dynamic baseline", "shuffle"]) if label in set(df["condition"].astype(str))]
    traces = [label for label in (spec.get("traces") or ["Original", "Donor", "Others"]) if label in set(df["trace"].astype(str))]
    if not conditions or not {"Original", "Donor", "Others"}.issubset(set(traces)):
        _placeholder_box(ax, spec, "Error-composition traces unavailable")
        return
    ax.paper_fig_plot_form = "vertical_stacked_error_composition"
    ax.paper_fig_raw_points = False
    ax.paper_fig_value_labels = True
    trace_colors = {
        "Original": get_plot_color("original_sample_trace"),
        "Donor": get_plot_color("donor_trace"),
        "Others": get_plot_color("other_residual"),
    }
    x = np.arange(len(conditions), dtype=float)
    bottom = np.zeros(len(conditions), dtype=float)
    label_count = 0
    for trace in traces:
        means = []
        for condition in conditions:
            vals = pd.to_numeric(df.loc[(df["condition"] == condition) & (df["trace"] == trace), "value"], errors="coerce").dropna().to_numpy(dtype=float)
            means.append(float(np.nanmean(vals)) if vals.size else 0.0)
        heights = np.asarray(means, dtype=float)
        ax.bar(
            x,
            heights,
            bottom=bottom,
            width=0.58,
            color=trace_colors.get(trace, "#D9D9D9"),
            edgecolor="black",
            linewidth=st["bar_edge_width"],
            alpha=0.90,
            label=trace,
            zorder=1,
        )
        for xi, y0, height_value in zip(x, bottom, heights):
            if not np.isfinite(height_value) or height_value < 7.0:
                continue
            color = "white" if trace in {"Original", "Donor"} else "0.25"
            ax.text(
                float(xi),
                float(y0 + height_value / 2.0),
                f"{height_value:.0f}%",
                ha="center",
                va="center",
                fontsize=st["annotation_fontsize"],
                color=color,
                zorder=3,
            )
            label_count += 1
        bottom = bottom + heights
    display_labels = [str((spec.get("display_labels") or {}).get(label, label)) for label in conditions]
    ax.set_xticks(x, display_labels, rotation=0, ha="center")
    ax.set_xlim(-0.55, len(conditions) - 0.45)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks([0, 50, 100], ["0", "50", "100"])
    ax.set_xlabel(str(spec.get("x_axis", "")))
    ax.set_ylabel(str(spec.get("y_axis", "Fraction within error trials (%)")))
    ax.paper_fig_value_label_count = label_count
    ax.paper_fig_value_labels_clear = True
    ax.legend(
        frameon=False,
        fontsize=st["legend_fontsize"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        handlelength=0.8,
        columnspacing=0.55,
        handletextpad=0.25,
        borderaxespad=0.0,
    )
    _tidy(ax, st)


def render_grouped_dot_bar(ax, panel_data: pd.DataFrame, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Placeholder-compatible grouped dot/bar renderer."""
    render_generic_placeholder(ax, panel_data, stats, spec, style)


def render_line_with_summary(ax, panel_data: pd.DataFrame, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Placeholder-compatible line renderer."""
    render_generic_placeholder(ax, panel_data, stats, spec, style)


def render_heatmap(ax, panel_data: pd.DataFrame, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Placeholder-compatible heatmap renderer."""
    render_generic_placeholder(ax, panel_data, stats, spec, style)


def render_scatter_regression(ax, panel_data: pd.DataFrame, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Placeholder-compatible scatter renderer."""
    render_generic_placeholder(ax, panel_data, stats, spec, style)


def render_generic_placeholder(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render a clear placeholder for panels not implemented yet."""
    _ = panel_data, stats, style
    reason = "placeholder renderer; adapter/renderer not implemented for this skeleton"
    ax.paper_fig_placeholder_reason = reason
    _placeholder_box(ax, spec, reason)


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def _reference_lines(ax, spec: Mapping[str, Any]) -> None:
    st = _style(None)
    for line in spec.get("reference_lines") or []:
        ax.axhline(float(line["value"]), linestyle="--", color="0.4", linewidth=st["line_width"])
        if line.get("label"):
            ax.text(0.98, float(line["value"]), str(line["label"]), ha="right", va="bottom", transform=ax.get_yaxis_transform(), fontsize=st["annotation_fontsize"])


def _sem(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size <= 1:
        return 0.0
    return float(np.nanstd(clean, ddof=1) / np.sqrt(clean.size))


def _placeholder_box(ax, spec: Mapping[str, Any], reason: str) -> None:
    panel_id = spec.get("panel_id", "?")
    claim = str(spec.get("claim", "No claim specified"))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.7")
    ax.text(0.5, 0.66, f"Panel {panel_id}", ha="center", va="center", transform=ax.transAxes, fontweight="bold")
    ax.text(0.5, 0.46, claim, ha="center", va="center", transform=ax.transAxes, wrap=True, fontsize=8)
    ax.text(0.5, 0.18, reason, ha="center", va="center", transform=ax.transAxes, fontsize=7, color="0.35", wrap=True)


def _style(style: Mapping[str, Any] | None) -> dict[str, float]:
    merged = dict(SMALL_PANEL_STYLE)
    merged.update(dict(style or {}))
    return merged


def _jitter(n: int, width: float) -> np.ndarray:
    if n <= 1:
        return np.zeros(max(n, 1), dtype=float)
    return np.linspace(-width, width, n, dtype=float)


def _draw_bar_summary(ax, df: pd.DataFrame, group_col: str, order: list[str], colors: list[str], st: Mapping[str, float], bar_width: float) -> tuple[list[float], list[float]]:
    x = np.arange(len(order), dtype=float)
    means, sems = [], []
    for label in order:
        values = pd.to_numeric(df.loc[df[group_col].astype(str) == label, "value"], errors="coerce").dropna().to_numpy(dtype=float)
        means.append(float(np.nanmean(values)) if values.size else np.nan)
        sems.append(_sem(values))
    ax.bar(x, means, width=bar_width, color=colors, edgecolor="black", linewidth=st["bar_edge_width"], alpha=0.86, zorder=1)
    ax.errorbar(x, means, yerr=sems, fmt="none", color="black", linewidth=st["error_line_width"], capsize=st["capsize"], zorder=3)
    return means, sems


def _add_value_labels(ax, x: np.ndarray, means: list[float], sems: list[float], st: Mapping[str, float], fmt: str = "{:.1f}") -> None:
    ymin, ymax = ax.get_ylim()
    offset = 0.018 * (ymax - ymin)
    for xpos, mean, sem in zip(x, means, sems):
        if not np.isfinite(mean):
            continue
        y = min(ymax - 0.035 * (ymax - ymin), mean + sem + offset)
        ax.text(float(xpos), float(y), fmt.format(mean), ha="center", va="bottom", fontsize=st["annotation_fontsize"], clip_on=True)


def _draw_point_range_summary(ax, df: pd.DataFrame, group_col: str, order: list[str], st: Mapping[str, float]) -> None:
    x = np.arange(len(order), dtype=float)
    means, sems = [], []
    for label in order:
        values = pd.to_numeric(df.loc[df[group_col].astype(str) == label, "value"], errors="coerce").dropna().to_numpy(dtype=float)
        means.append(float(np.nanmean(values)) if values.size else np.nan)
        sems.append(_sem(values))
    ax.errorbar(
        x,
        means,
        yerr=sems,
        fmt="o",
        markersize=4.0,
        mfc="black",
        mec="black",
        color="black",
        linewidth=st["error_line_width"],
        capsize=st["capsize"],
        zorder=5,
    )
    for i, label in enumerate(order):
        values = pd.to_numeric(df.loc[df[group_col].astype(str) == label, "value"], errors="coerce").dropna().to_numpy(dtype=float)
        ax.scatter(
            x[i] + _jitter(len(values), width=st["jitter_width"]),
            values,
            s=st["marker_size"],
            facecolor=CONDITION_COLORS.get(label, "white"),
            edgecolor="0.25",
            linewidth=0.3,
            alpha=0.55,
            zorder=3,
        )
    ax.set_xlim(-0.45, len(order) - 0.55)


def _set_percent_ylim(ax, values: np.ndarray, include_reference: float | None, top_room: float, min_upper: float = 0.0) -> None:
    finite = values[np.isfinite(values)]
    max_value = float(finite.max()) if finite.size else 1.0
    if include_reference is not None:
        max_value = max(max_value, float(include_reference))
    if max_value <= 100 and min_upper >= 100:
        upper = 100.0
    else:
        raw = max_value + top_room
        step = 5.0 if raw <= 50 else 10.0
        upper = max(min_upper, float(np.ceil(raw / step) * step))
    ax.set_ylim(0.0, upper)


def _tidy(ax, st: Mapping[str, float] | None = None) -> None:
    st = _style(st)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(st["line_width"])
    ax.tick_params(axis="both", labelsize=st["tick_labelsize"], width=st["tick_width"], length=st["tick_length"], pad=1.5)
    ax.xaxis.label.set_size(st["axis_labelsize"])
    ax.yaxis.label.set_size(st["axis_labelsize"])
    ax.xaxis.labelpad = st["labelpad"]
    ax.yaxis.labelpad = st["labelpad"]
    ax.grid(axis="y", color="0.88", linewidth=0.45)
    ax.set_axisbelow(True)
