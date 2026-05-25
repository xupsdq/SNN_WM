from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.plotting.paper_fig.panels.fig2_panels import (
    STATE_COLORS,
    STATE_LABELS,
    STATE_ORDER,
    _autoscale_y,
    _bar_scatter,
    _clean,
    _placeholder,
    _style,
    _tidy,
)


LAYER_ORDER = ["layer1", "layer2", "layer3"]
LAYER_LABELS = {"layer1": "L1", "layer2": "L2", "layer3": "L3"}
MODEL_ORDER = ["A_only", "B_only", "mean_AB", "sum_AB", "unconstrained_AB", "convex_AB"]
MODEL_LABELS = {"A_only": "A", "B_only": "B", "mean_AB": "Mean", "sum_AB": "Sum", "unconstrained_AB": "Unc.", "convex_AB": "Convex"}


def render_s3_wpri_across_layers(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    _layerwise_bar(ax, panel_data, spec, style, plot_form="s3_wpri_across_layers", placeholder="Layerwise WPRI unavailable")


def render_s3_residual_across_layers(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    _layerwise_bar(ax, panel_data, spec, style, plot_form="s3_residual_across_layers", placeholder="Layerwise residual unavailable")


def render_s3_linear_model_comparison(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, "Linear model comparison unavailable")
        return
    ax.paper_fig_plot_form = "s3_linear_model_comparison"
    order = [m for m in spec.get("model_order", MODEL_ORDER) if m in set(df["condition"].astype(str))]
    if not order:
        order = [m for m in MODEL_ORDER if m in set(df.get("model_name", pd.Series(dtype=str)).astype(str))]
    if not order:
        _placeholder(ax, spec, "Linear model labels unavailable")
        return
    colors = ["#C7C7C7", "#C7C7C7", "#9ECAE1", "#9ECAE1", "#54A24B", "#72B7B2"][: len(order)]
    _bar_scatter(ax, df.assign(condition=df["condition"].astype(str)), "condition", order, colors=colors, st=st, alpha=0.82)
    ax.set_xticks(np.arange(len(order)), [MODEL_LABELS.get(m, m) for m in order], rotation=25, ha="right")
    ax.set_ylabel(str(spec.get("y_axis", "Fit R2")))
    ax.set_xlabel("")
    ax.paper_fig_model_labels_readable = True
    _autoscale_y(ax, df["value"], include_zero=True)
    _tidy(ax, st)


def render_s4_ping_amplitude_sweep(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _line_by_state(ax, panel_data, spec, style, plot_form="s4_ping_amplitude_sweep", placeholder="Ping amplitude sweep unavailable", show_legend=False)


def render_s4_ping_duration_sweep(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _line_by_state(ax, panel_data, spec, style, plot_form="s4_ping_duration_sweep", placeholder="Ping duration sweep unavailable", show_legend=True)


def render_s4_completion_delay_gain(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "x_value" not in df.columns:
        _placeholder(ax, spec, "Completion delay gain unavailable")
        return
    ax.paper_fig_plot_form = "s4_completion_delay_gain"
    ax.axhline(0, color="0.45", linewidth=0.65, linestyle="--")
    ax.paper_fig_has_zero_reference = True
    df = df.copy()
    df["x_value"] = pd.to_numeric(df["x_value"], errors="coerce")
    conditions = df["condition"].astype(str).drop_duplicates().tolist()
    colors = {"S_AB_minus_relevant_single": "#54A24B", "target_A": STATE_COLORS["S_A"], "target_B": STATE_COLORS["S_B"]}
    for condition in conditions:
        part = df[df["condition"].astype(str).eq(condition)].dropna(subset=["x_value"])
        grouped = part.groupby("x_value", as_index=False)["value"].agg(["mean", "sem"]).reset_index()
        if grouped.empty:
            continue
        x = grouped["x_value"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        sem = grouped["sem"].fillna(0).to_numpy(dtype=float)
        color = colors.get(condition, "#54A24B")
        label = "S_AB - single" if condition == "S_AB_minus_relevant_single" else condition.replace("_", " ")
        ax.plot(x, y, marker="o", markersize=2.7, linewidth=0.9, color=color, label=label)
        if part["seed_id"].replace("", pd.NA).dropna().nunique() > 1:
            ax.errorbar(x, y, yerr=sem, fmt="none", ecolor=color, elinewidth=0.45, capsize=1.4, alpha=0.45)
    ax.set_xlabel(str(spec.get("x_axis", "Post-pair delay (ms)")))
    ax.set_ylabel(str(spec.get("y_axis", "Completion gain (%)")))
    if len(conditions) > 1:
        ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="best", handlelength=1.0)
    _autoscale_y(ax, df["value"], include_zero=True)
    _tidy(ax, st)


def _layerwise_bar(ax, panel_data: pd.DataFrame | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None, *, plot_form: str, placeholder: str) -> None:
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, placeholder)
        return
    ax.paper_fig_plot_form = plot_form
    order = [layer for layer in LAYER_ORDER if layer in set(df.get("layer", df["condition"]).astype(str))]
    if not order:
        order = [layer for layer in LAYER_ORDER if layer in set(df["condition"].astype(str))]
    if not order:
        _placeholder(ax, spec, "Layer labels unavailable")
        return
    plot_df = df.copy()
    if "layer" in plot_df.columns:
        plot_df["condition"] = plot_df["layer"].astype(str)
    _bar_scatter(ax, plot_df, "condition", order, colors=["#4C78A8", "#F58518", "#54A24B"], st=st, alpha=0.82)
    ax.axhline(0, color="0.45", linewidth=0.65, linestyle="--")
    ax.set_xticks(np.arange(len(order)), [LAYER_LABELS.get(layer, layer) for layer in order])
    ax.set_ylabel(str(spec.get("y_axis", "Score")))
    ax.set_xlabel(str(spec.get("x_axis", "Layer")))
    _autoscale_y(ax, plot_df["value"], include_zero=True)
    _tidy(ax, st)


def _line_by_state(ax, panel_data: pd.DataFrame | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None, *, plot_form: str, placeholder: str, show_legend: bool) -> None:
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "x_value" not in df.columns:
        _placeholder(ax, spec, placeholder)
        return
    ax.paper_fig_plot_form = plot_form
    ax.paper_fig_x_metric = str(spec.get("sweep_parameter", ""))
    df = df.copy()
    df["x_value"] = pd.to_numeric(df["x_value"], errors="coerce")
    for condition in [c for c in STATE_ORDER if c in set(df["condition"].astype(str))]:
        part = df[df["condition"].astype(str).eq(condition)].dropna(subset=["x_value"])
        grouped = part.groupby("x_value", as_index=False)["value"].agg(["mean", "sem"]).reset_index()
        if grouped.empty:
            continue
        x = grouped["x_value"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        sem = grouped["sem"].fillna(0).to_numpy(dtype=float)
        color = STATE_COLORS.get(condition, "0.3")
        ax.plot(x, y, marker="o", markersize=2.6, linewidth=0.88, color=color, label=STATE_LABELS.get(condition, condition).replace("\n", " "))
        if part["seed_id"].replace("", pd.NA).dropna().nunique() > 1:
            ax.errorbar(x, y, yerr=sem, fmt="none", ecolor=color, elinewidth=0.45, capsize=1.4, alpha=0.45)
    ax.set_ylim(0, 100)
    ax.set_xlabel(str(spec.get("x_axis", "")))
    ax.set_ylabel(str(spec.get("y_axis", "Pair-member readout (%)")))
    if show_legend:
        legend = ax.legend(frameon=False, fontsize=st["legend_fontsize"], ncol=2, loc="best", handlelength=1.0, columnspacing=0.7)
        ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
        ax.paper_fig_legend_ncols = 2
    _tidy(ax, st)
