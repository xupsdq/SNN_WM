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


def render_s11_score_input_ping_audit(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    ax.axis("off")
    items = [
        ("nonfinite_raw_count", "nonfinite", "count"),
        ("baseline_floor_count", "baseline floor", "count"),
        ("clipped_ratio_max", "clip max", "ratio"),
        ("mean_valid_site_count", "entry sites", "sites"),
        ("mean_entry_area", "entry area", "px"),
        ("ping_active_sites", "ping sites", "sites"),
        ("total_ping_current", "ping current", "current"),
    ]
    y = 0.92
    for metric, label, unit in items:
        vals = pd.to_numeric(df.loc[df["metric"].astype(str).eq(metric), "value"], errors="coerce").dropna()
        if vals.empty:
            continue
        value = float(vals.mean())
        color = get_plot_color("peak_region", context="fig6") if metric in {"clipped_ratio_max", "ping_active_sites", "total_ping_current"} else "0.18"
        ax.text(0.02, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=5.0, color="0.20")
        ax.text(0.92, y, f"{_format_compact(value)} {unit}", transform=ax.transAxes, ha="right", va="center", fontsize=5.0, color=color)
        ax.plot([0.02, 0.92], [y - 0.055, y - 0.055], transform=ax.transAxes, color="0.86", linewidth=0.35)
        y -= 0.125
        if y < 0.08:
            break
    ax.paper_fig_plot_form = "s11_score_input_ping_audit"


def render_s11_global_ping_count_endpoint(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("mean_early_spike_count")] if not df.empty else df
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    _line_by_x(ax, use, "x_value", "Early spike count", show_points=False)
    ax.set_xlabel("STSP score quantile")
    ax.paper_fig_plot_form = "s11_global_ping_count_endpoint"
    _tidy(ax)


def render_s11_real_probe_window_robustness(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _window_line_panel(ax, panel_data, stats, spec, style, "q5_minus_q1_delta_spike_probability", "Q5 - Q1 deflection", "s11_real_probe_window_robustness")


def render_s11_overlap_interaction_window_robustness(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _window_line_panel(ax, panel_data, stats, spec, style, "interaction_delta", "Interaction", "s11_overlap_interaction_window_robustness")


def render_s11_overlap_site_availability(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("mean_sites")] if not df.empty else df
    if use.empty or not {"stsp_group", "overlap_group"}.issubset(use.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    stsp_order = _ordered_unique(use["stsp_group"], ["high", "high_stsp", "low", "low_stsp"])
    overlap_order = _ordered_unique(use["overlap_group"], ["overlap", "no_overlap", "nonoverlap"])
    matrix = np.full((len(stsp_order), len(overlap_order)), np.nan)
    nonzero = np.full_like(matrix, np.nan, dtype=float)
    for i, stsp in enumerate(stsp_order):
        for j, overlap in enumerate(overlap_order):
            part = use[use["stsp_group"].astype(str).eq(str(stsp)) & use["overlap_group"].astype(str).eq(str(overlap))]
            if part.empty:
                continue
            matrix[i, j] = float(pd.to_numeric(part["value"], errors="coerce").mean())
            nonzero[i, j] = float(pd.to_numeric(part.get("nonzero_fraction", pd.Series(dtype=float)), errors="coerce").mean())
    vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    im = ax.imshow(matrix, cmap="Greens", vmin=0.0, vmax=max(vmax, 1e-6), aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isfinite(matrix[i, j]):
                continue
            ax.text(j, i, f"{matrix[i, j]:.1f}\n{nonzero[i, j]:.2f}", ha="center", va="center", fontsize=5.0, color="0.08")
    ax.set_xticks(np.arange(len(overlap_order)), [_short_s11(v) for v in overlap_order])
    ax.set_yticks(np.arange(len(stsp_order)), [_short_s11(v) for v in stsp_order])
    ax.set_xlabel("Probe overlap")
    ax.set_ylabel("STSP group")
    ax.paper_fig_plot_form = "s11_overlap_site_availability"
    ax.paper_fig_colorbar_needed = False
    _tidy(ax)


def render_s11_high_stsp_ablation_paired_difference(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("high_stsp_overlap_minus_matched_loss")] if not df.empty else df
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    vals = pd.to_numeric(use["value"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    mean = float(np.mean(vals))
    err = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    ax.axhline(0, color="0.42", linestyle="--", linewidth=0.65, zorder=1)
    ax.bar([0], [mean], width=0.55, color=get_plot_color("peak_region", context="fig6"), edgecolor="0.20", linewidth=0.55, zorder=2)
    ax.errorbar([0], [mean], yerr=[err], fmt="none", color="0.12", linewidth=0.65, capsize=2.0, zorder=3)
    frac = float(np.mean(vals > 0)) if vals.size else np.nan
    ax.text(0.95, 0.94, f"n={vals.size}\nP>0={frac:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.4, color="0.28")
    ax.set_xticks([0], ["high-STSP\nminus matched"])
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylabel("Loss difference")
    ax.paper_fig_plot_form = "s11_high_stsp_ablation_paired_difference"
    _tidy(ax)


def render_s11_score_shuffle_null(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _optional_extension_panel(ax, panel_data, stats, spec, style, "score-shuffle\nnot run", "s11_score_shuffle_null")


def render_s11_threshold_sensitivity(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if _is_optional_placeholder(df):
        _optional_extension_panel(ax, panel_data, stats, spec, style, "threshold sweep\nnot run", "s11_threshold_sensitivity")
        return
    if not df.empty and {"stsp_group_quantile", "overlap_threshold"}.issubset(df.columns):
        q = sorted(pd.to_numeric(df["stsp_group_quantile"], errors="coerce").dropna().unique())
        t = sorted(pd.to_numeric(df["overlap_threshold"], errors="coerce").dropna().unique())
        if q and t:
            matrix = np.full((len(q), len(t)), np.nan)
            for i, qv in enumerate(q):
                for j, tv in enumerate(t):
                    part = df[np.isclose(pd.to_numeric(df["stsp_group_quantile"], errors="coerce"), qv) & np.isclose(pd.to_numeric(df["overlap_threshold"], errors="coerce"), tv)]
                    if not part.empty:
                        matrix[i, j] = float(pd.to_numeric(part["value"], errors="coerce").mean())
            vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
            ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isfinite(matrix[i, j]):
                        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5.0)
            ax.set_xticks(np.arange(len(t)), [f"{v:.2f}" for v in t])
            ax.set_yticks(np.arange(len(q)), [f"{v:.2f}" for v in q])
            ax.set_xlabel("Overlap threshold")
            ax.set_ylabel("STSP quantile")
            ax.paper_fig_plot_form = "s11_threshold_sensitivity"
            _tidy(ax)
            return
    _generic_dot(ax, panel_data, stats, spec, "s11_threshold_sensitivity", "Effect")


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


def _window_line_panel(
    ax,
    panel_data: pd.DataFrame | None,
    stats: Mapping[str, Any] | None,
    spec: Mapping[str, Any],
    style: Mapping[str, Any] | None,
    metric: str,
    ylabel: str,
    plot_form: str,
) -> None:
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq(metric)] if not df.empty else df
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    ax.axhline(0, color="0.42", linestyle="--", linewidth=0.65, zorder=1)
    _line_by_x(ax, use, "early_window_ms", ylabel, show_points=False)
    ax.set_xlabel("Early window (ms)")
    ax.paper_fig_plot_form = plot_form
    _tidy(ax)


def _optional_extension_panel(
    ax,
    panel_data: pd.DataFrame | None,
    stats: Mapping[str, Any] | None,
    spec: Mapping[str, Any],
    style: Mapping[str, Any] | None,
    placeholder_text: str,
    plot_form: str,
) -> None:
    df = _clean(panel_data)
    if _is_optional_placeholder(df) or df.empty:
        ax.axis("off")
        ax.text(0.5, 0.58, placeholder_text, transform=ax.transAxes, ha="center", va="center", fontsize=7.0, color="0.28")
        ax.text(0.5, 0.34, "optional extension", transform=ax.transAxes, ha="center", va="center", fontsize=5.8, color="0.45")
        ax.paper_fig_plot_form = f"{plot_form}_placeholder"
        ax.paper_fig_optional_placeholder = True
        return
    _generic_dot(ax, panel_data, stats, spec, plot_form, "Observed - null")


def _is_optional_placeholder(df: pd.DataFrame) -> bool:
    if df.empty or "metric" not in df.columns:
        return False
    return df["metric"].astype(str).eq("optional_placeholder").any()


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


def _short_s11(value: Any) -> str:
    return (
        str(value)
        .replace("high_stsp", "high\nSTSP")
        .replace("low_stsp", "low\nSTSP")
        .replace("no_overlap", "no\nentry")
        .replace("nonoverlap", "no\nentry")
        .replace("overlap", "entry")
        .replace("_", "\n")
    )


def _format_compact(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"
