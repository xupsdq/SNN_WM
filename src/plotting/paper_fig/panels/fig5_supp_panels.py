from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.plotting.common.colors import get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL, GRID_ALPHA_SOFT
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder
from src.plotting.paper_fig.panels.fig5_panels import CONDITION_COLORS, GROUP_COLORS


STYLE = {
    "axis_labelsize": 5.4,
    "tick_labelsize": 5.0,
    "legend_fontsize": 4.8,
    "line_width": 0.85,
    "marker_size": 8.0,
}
TRANSITION_COLORS = {
    "P_advance": "#4C78A8",
    "P_recruit": "#F58518",
    "P_loss": "#E45756",
    "P_unchanged": "#BAB0AC",
    "P_same_winner_preserved": "#54A24B",
    "P_same_winner_lost": "#E45756",
    "P_same_winner_delayed": "#F58518",
    "P_same_winner_lost_or_delayed": "#B279A2",
}


def render_s9_early_window_robustness(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "early_window_ms" not in df.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    for group in _ordered_unique(df["condition"], ["Overlap-dominant", "Probe-only-dominant", "Random matched", "Balanced"]):
        part = df[df["condition"].eq(group)].copy()
        summary = _mean_sem(part, ["early_window_ms"]).sort_values("early_window_ms")
        ax.errorbar(summary["early_window_ms"], summary["mean"], yerr=summary["sem"], fmt="o-", markersize=2.4, linewidth=st["line_width"], capsize=1.4, color=GROUP_COLORS.get(group, COLOR_NEUTRAL), label=_short_group(group))
    ax.set_xlabel(str(spec.get("x_axis", "Early window (ms)")))
    ax.set_ylabel(str(spec.get("y_axis", "Advance + recruit probability")))
    ax.set_ylim(0, max(0.05, min(1.0, _finite_max(df["value"]) * 1.2)))
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="best", handlelength=1.0)
    ax.paper_fig_plot_form = "s9_early_window_robustness"
    _tidy(ax, st)


def render_s9_transition_composition(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    groups = _ordered_unique(df["condition"], ["Overlap-dominant", "Probe-only-dominant", "Random matched", "Balanced"])
    x = np.arange(len(groups), dtype=float)
    bottom = np.zeros(len(groups), dtype=float)
    for metric in ("P_advance", "P_recruit", "P_loss", "P_unchanged"):
        means, _sems = _group_means(df[df["metric"].eq(metric)], "condition", groups)
        ax.bar(x, means, bottom=bottom, width=0.58, color=TRANSITION_COLORS[metric], edgecolor="black", linewidth=0.25, label=_metric_label(metric))
        bottom += means
    ax.set_xticks(x, [_wrap(_short_group(g)) for g in groups], rotation=0)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel(str(spec.get("y_axis", "Transition probability")))
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.20), handlelength=0.8, columnspacing=0.5)
    ax.paper_fig_plot_form = "s9_transition_composition"
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = 4
    _tidy(ax, st)


def render_s9_trialwise_transition_advantage(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    metric = str(spec.get("metric", "delta_P_advance_plus_recruit"))
    plot_df = df[df["metric"].astype(str).eq(metric)].copy()
    if plot_df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = _ordered_unique(plot_df["condition"], ["vs probe-only", "vs random", "vs balanced"])
    xs = np.arange(len(order), dtype=float)
    means, sems = _group_means(plot_df, "condition", order)
    colors = ["#4C78A8", "#6B6B6B", "#7F6DBA"][: len(order)]
    ax.bar(xs, means, yerr=sems, capsize=1.4, width=0.58, color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="0.35", linewidth=0.6)
    ymin = min(0.0, float(np.nanmin(means - sems)) if len(means) else 0.0)
    ymax = max(0.05, float(np.nanmax(means + sems)) if len(means) else 0.05)
    ax.set_ylim(ymin - 0.05 * max(0.05, ymax - ymin), ymax + 0.24 * max(0.05, ymax - ymin))
    for x, label, mean, sem in zip(xs, order, means, sems):
        vals = pd.to_numeric(plot_df.loc[plot_df["condition"].astype(str).eq(str(label)), "value"], errors="coerce").dropna()
        if vals.empty:
            continue
        frac = float((vals > 0).mean())
        ax.text(x, mean + sem + 0.06 * max(0.05, ymax - ymin), f"{frac:.0%}>0", ha="center", va="bottom", fontsize=st["tick_labelsize"], color="0.25")
    ax.set_xticks(xs, [_comparison_short(label) for label in order], rotation=0)
    ax.set_ylabel(str(spec.get("y_axis", "Delta P(advance + recruit)")))
    ax.paper_fig_plot_form = "s9_trialwise_transition_advantage_bar_only"
    ax.paper_fig_raw_points = False
    ax.paper_fig_value_labels = True
    ax.paper_fig_value_label_count = len(order)
    ax.paper_fig_value_labels_clear = True
    _tidy(ax, st)


def render_s9_event_trace(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _trace_plot(ax, panel_data, stats, spec, style, "s9_event_trace")


def render_s9_event_chain_null(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    metric_order = _ordered_unique(df["metric"], ["full_chain_satisfied_fraction", "winner_pre_spike_boost_fraction", "winner_spikes_earlier_fraction", "loser_post_winner_suppressed_fraction"])
    conditions = _ordered_unique(df["condition"], ["Observed"])[:4]
    if len(conditions) == 1:
        conditions = _ordered_unique(df["condition"], ["Observed", "Null shuffle", "Null temporal", "Null null"])
    x = np.arange(len(metric_order), dtype=float)
    width = min(0.26, 0.72 / max(1, len(conditions)))
    for i, condition in enumerate(conditions):
        part = df[df["condition"].astype(str).eq(condition)]
        means, sems = _group_means(part, "metric", metric_order)
        offset = (i - (len(conditions) - 1) / 2.0) * width
        color = get_plot_color("dynamic") if condition == "Observed" else COLOR_NEUTRAL
        ax.bar(x + offset, means, yerr=sems, capsize=1.2, width=width * 0.92, color=color, edgecolor="black", linewidth=0.25, alpha=0.75, label=_event_null_label(condition))
    ax.set_xticks(x, [_short_metric(m) for m in metric_order], rotation=20, ha="right")
    ax.set_ylabel(str(spec.get("y_axis", "Fraction")))
    if len(conditions) > 1:
        ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="upper left", ncols=2, handlelength=0.8, columnspacing=0.7)
    ax.paper_fig_plot_form = "s9_event_chain_null"
    _tidy(ax, st)


def render_s9_neighborhood_event_audit(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    radius = df[df.get("source_level", pd.Series(dtype=str)).astype(str).eq("radius_robustness")].copy()
    if not radius.empty and "neighborhood_radius" in radius.columns:
        for metric, color in (
            ("winner_pre_spike_delta_v_mean", "#D95F02"),
            ("loser_post_winner_inh_rise", "#7570B3"),
            ("loser_post_winner_suppressed", "#1B9E77"),
        ):
            part = radius[radius["metric"].eq(metric)].copy()
            if part.empty:
                continue
            summary = _mean_sem(part, ["neighborhood_radius"]).sort_values("neighborhood_radius")
            ax.errorbar(summary["neighborhood_radius"], summary["mean"], yerr=summary["sem"], fmt="o-", markersize=2.2, linewidth=st["line_width"], capsize=1.2, color=color, label=_short_metric(metric))
    audit = df[df["metric"].eq("event_selection_count")]
    if not audit.empty:
        total = pd.to_numeric(audit["value"], errors="coerce").sum()
        included = pd.to_numeric(audit.loc[audit["condition"].eq("included"), "value"], errors="coerce").sum()
        ax.text(0.99, 0.96, f"events kept {included:.0f}/{total:.0f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.0, color=COLOR_NEUTRAL)
    ax.set_xlabel(str(spec.get("x_axis", "Neighborhood radius")))
    ax.set_ylabel(str(spec.get("y_axis", "Event-chain metric")))
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="best", handlelength=0.9)
    ax.paper_fig_plot_form = "s9_neighborhood_event_audit"
    _tidy(ax, st)


def render_s10_perturbation_ux_audit(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    metrics = _ordered_unique(df["metric"], ["u_delta_mean", "x_delta_mean", "g_delta_mean"])
    conditions = _ordered_unique(df["condition"], ["Attenuate overlap support", "Reset overlap support", "Sham perturbation"])
    x = np.arange(len(conditions), dtype=float)
    width = 0.22
    for i, metric in enumerate(metrics):
        means, sems = _group_means(df[df["metric"].eq(metric)], "condition", conditions)
        ax.bar(x + (i - (len(metrics) - 1) / 2.0) * width, means, yerr=sems, width=width * 0.9, capsize=1.2, color=TRANSITION_COLORS.get(metric, f"0.{i+4}"), edgecolor="black", linewidth=0.25, label=metric.replace("_delta_mean", ""))
    ax.axhline(0, color="0.45", linewidth=0.55)
    ax.set_xticks(x, [_condition_short(c) for c in conditions], rotation=0)
    ax.set_ylabel(str(spec.get("y_axis", "Mean variable delta")))
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="best", handlelength=0.8)
    ax.paper_fig_plot_form = "s10_perturbation_ux_audit"
    _tidy(ax, st)


def render_s9_perturbation_ux_audit(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    render_s10_perturbation_ux_audit(ax, panel_data, stats, spec, style)
    ax.paper_fig_plot_form = "s9_perturbation_ux_audit"


def render_s10_perturbation_transition_contrast(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    df = df[df["metric"].isin(["delta_P_advance_plus_recruit", "delta_P_loss", "delta_P_same_winner_lost_or_delayed"])] if not df.empty else df
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    groups = _ordered_unique(df.get("unit_group_label", df["condition"]), ["Overlap-dominant", "Probe-only-dominant", "Random matched", "Balanced"])
    conditions = _ordered_unique(df["condition"], ["Attenuate overlap support", "Reset overlap support"])
    metric = "delta_P_advance_plus_recruit" if "delta_P_advance_plus_recruit" in set(df["metric"]) else str(df["metric"].iloc[0])
    plot_df = df[df["metric"].eq(metric)]
    x = np.arange(len(groups), dtype=float)
    width = 0.28
    group_col = "unit_group_label" if "unit_group_label" in plot_df.columns else "condition"
    for i, condition in enumerate(conditions):
        part = plot_df[plot_df["condition"].eq(condition)]
        means, sems = _group_means(part, group_col, groups)
        ax.bar(x + (i - (len(conditions) - 1) / 2.0) * width, means, yerr=sems, width=width * 0.9, capsize=1.2, color=CONDITION_COLORS.get(condition, COLOR_NEUTRAL), edgecolor="black", linewidth=0.25, label=_condition_short(condition))
    ax.axhline(0, color="0.45", linewidth=0.55)
    ax.set_xticks(x, [_wrap(_short_group(g)) for g in groups], rotation=0)
    ax.set_ylabel(str(spec.get("y_axis", "Delta transition probability")))
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="best", handlelength=0.8)
    ax.paper_fig_plot_form = "s10_perturbation_transition_contrast"
    _tidy(ax, st)


def render_s9_perturbation_transition_contrast(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    render_s10_perturbation_transition_contrast(ax, panel_data, stats, spec, style)
    ax.paper_fig_plot_form = "s9_perturbation_transition_contrast"


def render_s10_same_winner_lost_delayed(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    conditions = _ordered_unique(df["condition"], ["Dynamic intact", "Attenuate overlap support", "Reset overlap support"])
    metrics = [m for m in ("P_same_winner_preserved", "P_same_winner_lost", "P_same_winner_delayed") if m in set(df["metric"])]
    x = np.arange(len(conditions), dtype=float)
    bottom = np.zeros(len(conditions), dtype=float)
    for metric in metrics:
        means, _sems = _group_means(df[df["metric"].eq(metric)], "condition", conditions)
        ax.bar(x, means, bottom=bottom, width=0.58, color=TRANSITION_COLORS.get(metric, COLOR_NEUTRAL), edgecolor="black", linewidth=0.25, label=_short_metric(metric))
        bottom += means
    ax.set_xticks(x, [_condition_short(c) for c in conditions], rotation=0)
    ax.set_ylim(0, max(1.0, float(np.nanmax(bottom)) if len(bottom) else 1.0) * 1.04)
    ax.set_ylabel(str(spec.get("y_axis", "Same-winner probability")))
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=3, handlelength=0.8, columnspacing=0.8)
    ax.paper_fig_plot_form = "s10_same_winner_lost_delayed"
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = 3
    _tidy(ax, st)


def render_s9_same_winner_lost_delayed(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    render_s10_same_winner_lost_delayed(ax, panel_data, stats, spec, style)
    ax.paper_fig_plot_form = "s9_same_winner_lost_delayed"


def render_s10_dynamic_like_recovery(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    conditions = _ordered_unique(df["condition"], ["Dynamic intact", "Attenuate overlap support", "Reset overlap support", "Sham perturbation", "Static frozen"])
    metrics = _ordered_unique(df["metric"], ["dynamic_like_spike_similarity", "dynamic_like_readout_recovery", "decision_deflection_score"])[:3]
    x = np.arange(len(conditions), dtype=float)
    width = min(0.22, 0.72 / max(1, len(metrics)))
    for i, metric in enumerate(metrics):
        means, sems = _group_means(df[df["metric"].eq(metric)], "condition", conditions)
        ax.bar(x + (i - (len(metrics) - 1) / 2.0) * width, means, yerr=sems, width=width * 0.9, capsize=1.2, color=TRANSITION_COLORS.get(metric, f"0.{i+4}"), edgecolor="black", linewidth=0.25, label=_short_metric(metric))
    ax.set_xticks(x, [_condition_short(c) for c in conditions], rotation=0)
    ax.set_ylabel(str(spec.get("y_axis", "Dynamic-like recovery")))
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="best", handlelength=0.8)
    ax.paper_fig_plot_form = "s10_dynamic_like_recovery"
    _tidy(ax, st)


def render_s10_sham_matching_controls(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    controls = df[~df["metric"].astype(str).str.startswith("matching_error")]
    matching = df[df["metric"].astype(str).str.startswith("matching_error")]
    plot_df = controls if not controls.empty else matching
    metrics = _ordered_unique(plot_df["metric"], list(plot_df["metric"].dropna().unique()))[:6]
    _dot_bar(ax, plot_df, metrics, group_col="metric", color=get_plot_color("other_residual"))
    if not matching.empty:
        mean_error = pd.to_numeric(matching["value"], errors="coerce").dropna().mean()
        if np.isfinite(mean_error):
            ax.text(0.99, 0.94, f"mean match err={mean_error:.3g}", transform=ax.transAxes, ha="right", va="top", fontsize=5.0, color=COLOR_NEUTRAL)
    ax.axhline(0, color="0.45", linewidth=0.55)
    ax.set_xticks(np.arange(len(metrics)), [_short_metric(m) for m in metrics], rotation=18, ha="right")
    ax.set_ylabel(str(spec.get("y_axis", "Value / matching error")))
    ax.paper_fig_plot_form = "s10_sham_matching_controls"
    _tidy(ax, st)


def _trace_plot(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None, plot_form: str) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "time_ms" not in df.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    colors = {"winner_delta_v": "#D95F02", "loser_delta_v": "#1B9E77", "loser_inhibition": "#7570B3"}
    for metric in ("winner_delta_v", "loser_delta_v", "loser_inhibition"):
        part = df[df["metric"].eq(metric)].copy()
        if part.empty:
            continue
        summary = _mean_sem(part, ["time_ms"]).sort_values("time_ms")
        ax.plot(summary["time_ms"], summary["mean"], linewidth=st["line_width"], color=colors.get(metric, COLOR_NEUTRAL), label=_short_metric(metric))
        if len(part["seed_id"].dropna().unique()) > 1:
            ax.fill_between(summary["time_ms"], summary["mean"] - summary["sem"], summary["mean"] + summary["sem"], color=colors.get(metric, COLOR_NEUTRAL), alpha=0.18, linewidth=0)
    ax.axvline(0, color="0.25", linewidth=0.55)
    ax.axhline(0, color="0.82", linewidth=0.5)
    ax.set_xlabel(str(spec.get("x_axis", "Time from winner spike (ms)")))
    ax.set_ylabel(str(spec.get("y_axis", "Dynamic minus static")))
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="best", handlelength=0.9)
    ax.paper_fig_plot_form = plot_form
    _tidy(ax, st)


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    for col in ("value", "time_ms", "early_window_ms", "neighborhood_radius"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["value"])


def _dot_bar(ax, df: pd.DataFrame, order: Sequence[str], *, group_col: str = "condition", color: str | None = None) -> None:
    xs = np.arange(len(order), dtype=float)
    means, sems = _group_means(df, group_col, list(order))
    ax.bar(xs, means, yerr=sems, color=color or COLOR_NEUTRAL, edgecolor="black", linewidth=0.35, alpha=0.68, capsize=1.5, width=0.58)
    for i, item in enumerate(order):
        vals = pd.to_numeric(df[df[group_col].astype(str).eq(str(item))]["value"], errors="coerce").dropna().to_numpy(dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.zeros(len(vals))
        ax.scatter(np.full(len(vals), xs[i]) + jitter, vals, s=7, color="white", edgecolor="0.25", linewidth=0.25, zorder=3)


def _mean_sem(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols), dropna=False, sort=True)["value"].agg(["mean", "count", "std"]).reset_index()
    grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    return grouped


def _group_means(df: pd.DataFrame, group_col: str, order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means = []
    sems = []
    for item in order:
        vals = pd.to_numeric(df[df[group_col].astype(str).eq(str(item))]["value"], errors="coerce").dropna()
        means.append(float(vals.mean()) if not vals.empty else 0.0)
        sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)
    return np.asarray(means, dtype=float), np.asarray(sems, dtype=float)


def _ordered_unique(series: pd.Series, preferred: Sequence[str]) -> list[str]:
    present = [str(v) for v in series.dropna().astype(str).unique()]
    out = [item for item in preferred if item in present]
    out.extend(item for item in present if item not in out)
    return out


def _style(style: Mapping[str, Any] | None) -> dict[str, float]:
    out = dict(STYLE)
    out.update({k: v for k, v in dict(style or {}).items() if k in out})
    return out


def _tidy(ax, st: Mapping[str, float]) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=GRID_ALPHA_SOFT, linewidth=0.32)
    ax.tick_params(axis="both", labelsize=st["tick_labelsize"], pad=0.8, length=1.7, width=0.5, color=COLOR_NEUTRAL)
    ax.xaxis.label.set_size(st["axis_labelsize"])
    ax.yaxis.label.set_size(st["axis_labelsize"])
    ax.xaxis.labelpad = 0.5
    ax.yaxis.labelpad = 0.7


def _wrap(label: str) -> str:
    return str(label).replace("Overlap-", "Overlap-\n").replace("Probe-only-", "Probe-only-\n").replace("Random ", "Random\n")


def _short_group(label: str) -> str:
    return str(label).replace("-dominant", "").replace(" matched", "")


def _metric_label(metric: str) -> str:
    return {"P_advance": "Advance", "P_recruit": "Recruit", "P_loss": "Loss", "P_unchanged": "Unchanged"}.get(metric, metric)


def _short_metric(metric: Any) -> str:
    text = str(metric)
    return (
        text.replace("P_same_winner_", "")
        .replace("P_advance_plus_recruit", "Adv+rec")
        .replace("winner_pre_spike_delta_v_mean", "winner V")
        .replace("loser_post_winner_inh_rise", "loser inh.")
        .replace("loser_post_winner_suppressed", "loser supp.")
        .replace("full_chain_satisfied_fraction", "full chain")
        .replace("winner_pre_spike_boost_fraction", "winner boost")
        .replace("winner_spikes_earlier_fraction", "early winner")
        .replace("loser_post_winner_suppressed_fraction", "loser supp.")
        .replace("dynamic_like_spike_similarity", "spike sim.")
        .replace("dynamic_like_readout_recovery", "readout rec.")
        .replace("decision_deflection_score", "decision")
        .replace("_", "\n")
    )


def _condition_short(condition: str) -> str:
    return {
        "Dynamic intact": "Dynamic",
        "Attenuate overlap support": "Atten.",
        "Reset overlap support": "Reset",
        "Attenuate L1 STSP": "Atten.",
        "Reset L1 STSP": "Reset",
        "Attenuate STSP": "Atten.",
        "Reset STSP": "Reset",
        "Sham perturbation": "Sham",
        "Static frozen": "Static",
    }.get(condition, condition)


def _comparison_short(label: Any) -> str:
    return {
        "vs probe-only": "vs\nprobe-only",
        "vs random": "vs\nrandom",
        "vs balanced": "vs\nbalanced",
    }.get(str(label), str(label))


def _event_null_label(condition: Any) -> str:
    text = str(condition)
    return {
        "Observed": "Observed",
        "Null event_time_shuffle": "time shuffle",
        "Null winner_loser_pairing_shuffle": "pair shuffle",
        "Null neighborhood_shuffle": "radius shuffle",
        "Null trial_shuffle": "trial shuffle",
        "Null label_shuffle": "label shuffle",
    }.get(text, text.replace("Null ", "").replace("_", " "))


def _finite_max(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.max()) if not vals.empty else 0.0
