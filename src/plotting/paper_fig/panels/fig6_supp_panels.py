from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.plotting.common.colors import get_plot_color
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder
from src.plotting.paper_fig.panels.fig6_panels import (
    _clean,
    _dot_bar,
    _fit_line,
    _line_by_x,
    _ordered_unique,
    _paired_low_high,
    _tidy,
)


def render_s11_peak_update_history(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    use = df[df["metric"].astype(str).isin(["mean_update_count", "mean_recent_update_count", "P_multi_recent_w3"])].copy()
    if use.empty:
        use = df
    order = _ordered_unique(use["condition"], ["Nonpeak control", "Peak", "Prior-updated nonpeak", "Single old", "Single recent", "Multi old", "Multi recent"])
    _dot_bar(ax, use, order, ylabel="Update history")
    ax.set_xlabel("")
    ax.paper_fig_plot_form = "s11_peak_update_history"
    _tidy(ax)


def render_s11_update_recency_model_detail(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if df["metric"].astype(str).eq("coefficient").any():
        _coefficient_plot(ax, df, "Coefficient")
    else:
        order = _ordered_unique(df["condition"], ["baseline_only", "update_only", "recency_only", "overlap_only", "update_plus_recency", "update_times_recency"])
        _dot_bar(ax, df, order, ylabel="CV R2")
    ax.paper_fig_plot_form = "s11_update_recency_model_detail"
    _tidy(ax)


def render_s11_peak_source_attribution(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    _line_by_x(ax, df, "relative_position_from_end", "Loss fraction")
    ax.set_xlabel("Positions from end")
    ax.paper_fig_plot_form = "s11_peak_source_attribution"
    _tidy(ax)


def render_s11_peak_input_overlap_origin(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if "recent_k" in df.columns and pd.to_numeric(df["recent_k"], errors="coerce").notna().any():
        _line_by_x(ax, df, "recent_k", "Peak-overlap Dice")
    else:
        order = _ordered_unique(df["condition"], ["all", "recent_2", "recent_3", "recent_4", "recent_5"])
        _dot_bar(ax, df, order, ylabel="Peak-overlap Dice")
    ax.set_xlabel("Overlap window")
    ax.paper_fig_plot_form = "s11_peak_input_overlap_origin"
    _tidy(ax)


def render_s11_alternative_peak_definitions(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _generic_dot(ax, panel_data, stats, spec, "s11_alternative_peak_definitions", "Stability")


def render_s11_visual_energy_classpair_controls(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _generic_dot(ax, panel_data, stats, spec, "s11_visual_energy_classpair_controls", "Diagnostic")


def render_s12_raw_overlap_matched_reentry(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if {"Low peak overlap", "High peak overlap"}.issubset(set(df["condition"].astype(str))):
        _paired_low_high(ax, df, "Low peak overlap", "High peak overlap", ylabel="Re-entry")
    else:
        _generic_dot(ax, panel_data, stats, spec, "s12_raw_overlap_matched_reentry", "Re-entry")
        return
    ax.paper_fig_plot_form = "s12_raw_overlap_matched_reentry"
    _tidy(ax)


def render_s12_peak_overlap_regression_controls(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    _coefficient_plot(ax, df, "Estimate")
    ax.paper_fig_plot_form = "s12_peak_overlap_regression_controls"
    _tidy(ax)


def render_s12_real_rollout_proxy_audit(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = panel_data if panel_data is not None else pd.DataFrame()
    ax.axis("off")
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    rows = []
    for _, r in df.iterrows():
        rows.append((str(r.get("condition", "")), str(r.get("status_text", r.get("value", "")))))
    y = 0.92
    for key, value in rows[:6]:
        color = get_plot_color("peak_region") if key == "proxy_mode" and value.lower() in {"true", "1"} else "0.15"
        ax.text(0.02, y, key.replace("_", " "), transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color=color)
        ax.text(0.98, y, value, transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=color)
        y -= 0.15
    if any(key == "proxy_mode" and value.lower() in {"true", "1"} for key, value in rows):
        ax.text(0.50, 0.04, "proxy mode active", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.2, color=get_plot_color("peak_region"))
    ax.paper_fig_plot_form = "s12_real_rollout_proxy_audit"


def render_s12_downstream_metric_breakdown(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _generic_dot(ax, panel_data, stats, spec, "s12_downstream_metric_breakdown", "Effect")


def render_s12_global_support_spike_controls(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    _coefficient_plot(ax, df, "Control estimate")
    ax.paper_fig_plot_form = "s12_global_support_spike_controls"
    _tidy(ax)


def render_s12_peak_perturbation(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if df["metric"].astype(str).eq("optional_placeholder").any():
        ax.axis("off")
        ax.text(0.5, 0.56, "Optional peak perturbation\nnot available", transform=ax.transAxes, ha="center", va="center", fontsize=7.0, color="0.25")
        ax.text(0.5, 0.30, "main claim remains predictive\npeak-amplified", transform=ax.transAxes, ha="center", va="center", fontsize=6.0, color="0.38")
        ax.paper_fig_plot_form = "s12_peak_perturbation_placeholder"
        return
    _generic_dot(ax, panel_data, stats, spec, "s12_peak_perturbation", "Perturbation effect")


def _generic_dot(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], plot_form: str, ylabel: str) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, None)
        return
    key_col = "condition" if "condition" in df.columns else "metric"
    order = _ordered_unique(df[key_col], [])
    _dot_bar(ax, df, order[:6], ylabel=ylabel)
    ax.set_xlabel("")
    ax.paper_fig_plot_form = plot_form
    _tidy(ax)


def _coefficient_plot(ax, df: pd.DataFrame, ylabel: str) -> None:
    order = _ordered_unique(df["condition"], ["peak_weighted_overlap", "raw_overlap", "visual_similarity", "input_energy", "global_support", "total_spike_count", "nonpeak_support"])
    if not order:
        order = _ordered_unique(df["metric"], [])
    xs = np.arange(len(order))
    for idx, condition in enumerate(order):
        part = df[df["condition"].astype(str).eq(condition)]
        if part.empty:
            part = df[df["metric"].astype(str).eq(condition)]
        vals = pd.to_numeric(part["value"], errors="coerce").dropna()
        if vals.empty:
            continue
        ax.errorbar([idx], [float(vals.mean())], yerr=[float(vals.sem()) if len(vals) > 1 else 0.0], fmt="o", color=get_plot_color("peak_region"), capsize=2.0, markersize=3.5)
        ax.scatter(np.full(len(vals), idx), vals, s=8, alpha=0.28, color=get_plot_color("peak_region"), linewidths=0)
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.6)
    ax.set_xticks(xs, [_short(v) for v in order], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))


def _short(value: Any) -> str:
    return (
        str(value)
        .replace("peak_weighted_overlap", "peak\nweighted")
        .replace("raw_overlap", "raw\noverlap")
        .replace("visual_similarity", "visual\nsim")
        .replace("input_energy", "input\nenergy")
        .replace("total_spike_count", "spike\ncount")
        .replace("global_support", "global\nsupport")
        .replace("nonpeak_support", "nonpeak\nsupport")
        .replace("_", "\n")
    )
