from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.plotting.common.colors import get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL, GRID_ALPHA_SOFT
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder
from src.plotting.paper_fig.panels.fig4_panels import render_fig4_decision_deflection


def render_s7_similarity_full_trend(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if "similarity_bin_order" in df.columns:
        df = df.sort_values(["similarity_bin_order", "seed_id"], kind="stable")
        group_cols = ["similarity_bin_order", "similarity_bin"] if "similarity_bin" in df.columns else ["similarity_bin_order"]
        summary = _mean_sem(df, group_cols)
        x = np.arange(len(summary))
        ax.errorbar(x, summary["mean"], yerr=summary["sem"], fmt="o-", color=get_plot_color("true_pair"), markersize=2.8, linewidth=0.9, capsize=1.5)
        ax.set_xticks(x, [str(v).replace("bin_", "b") for v in summary.get("similarity_bin", summary["similarity_bin_order"])], rotation=0)
    else:
        ax.scatter(np.arange(len(df)), df["value"], s=8, color=get_plot_color("true_pair"), alpha=0.35)
    ax.set_xlabel(str(spec.get("x_axis", "Similarity bin")))
    ax.set_ylabel(str(spec.get("y_axis", "Dynamic-induced drop")))
    _tidy(ax)
    ax.paper_fig_plot_form = "s7_similarity_full_trend"


def render_s7_matching_diagnostics(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    metrics = _ordered_unique(df["metric"], ["similarity_difference", "sample_energy_rel_difference", "probe_energy_rel_difference", "dice_overlap_difference", "overlap_difference", "mean_similarity_difference", "mean_sample_energy_rel_difference", "mean_probe_energy_rel_difference"])
    _dot_bar(ax, df, metrics, color=get_plot_color("other_residual"))
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.6)
    ax.set_xticks(np.arange(len(metrics)), [_short_metric(m) for m in metrics], rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(str(spec.get("y_axis", "Difference")))
    _tidy(ax)
    ax.paper_fig_plot_form = "s7_matching_diagnostics"


def render_s7_iso_similarity_matching(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    match = df[df["metric"].astype(str).eq("paired_delta_drop_event")].copy()
    rates = df[df["metric"].astype(str).isin(["drop_rate_low_overlap", "drop_rate_high_overlap"])]
    if not match.empty and {"drop_event_high", "drop_event_low"}.issubset(match.columns):
        for idx, row in match.reset_index(drop=True).iterrows():
            low = _num(row.get("drop_event_low"))
            high = _num(row.get("drop_event_high"))
            if np.isfinite(low) and np.isfinite(high):
                ax.plot([0, 1], [low, high], color="0.75", linewidth=0.45, alpha=0.45, zorder=1)
        for x, col, color in ((0, "drop_event_low", get_plot_color("low_overlap")), (1, "drop_event_high", get_plot_color("high_overlap"))):
            vals = pd.to_numeric(match[col], errors="coerce").dropna()
            if vals.empty:
                continue
            ax.scatter(np.full(len(vals), x), vals, s=9, color=color, alpha=0.30, zorder=2)
            ax.errorbar([x], [float(vals.mean())], yerr=[float(vals.sem()) if len(vals) > 1 else 0.0], fmt="o", color=color, markeredgecolor="white", markersize=4.5, capsize=2.0, zorder=4)
        delta_vals = pd.to_numeric(match["value"], errors="coerce").dropna()
        if not delta_vals.empty:
            ax.bar([2.15], [float(delta_vals.mean())], yerr=[float(delta_vals.sem()) if len(delta_vals) > 1 else 0.0], color=get_plot_color("sample_probe_overlap"), alpha=0.65, width=0.45, capsize=2.0)
    elif not rates.empty:
        _dot_bar(ax, rates, ["drop_rate_low_overlap", "drop_rate_high_overlap", "delta_drop_rate"], color=get_plot_color("sample_probe_overlap"))
    p_one = pd.to_numeric(df.loc[df["metric"].eq("permutation_p_one_sided"), "value"], errors="coerce").dropna()
    p_two = pd.to_numeric(df.loc[df["metric"].eq("permutation_p_two_sided"), "value"], errors="coerce").dropna()
    if not p_one.empty or not p_two.empty:
        text = []
        if not p_one.empty:
            text.append(f"p1={float(p_one.mean()):.3g}")
        if not p_two.empty:
            text.append(f"p2={float(p_two.mean()):.3g}")
        ax.text(0.98, 0.94, ", ".join(text), transform=ax.transAxes, ha="right", va="top", fontsize=5.3, color=COLOR_NEUTRAL)
    ax.axhline(0, color="0.45", linestyle=":", linewidth=0.6)
    ax.set_xticks([0, 1, 2.15], ["Low", "High", "High-Low"])
    ax.set_xlabel(str(spec.get("x_axis", "Matched overlap group")))
    ax.set_ylabel(str(spec.get("y_axis", "Drop event rate")))
    _tidy(ax)
    ax.paper_fig_plot_form = "s7_iso_similarity_matching"


def render_s7_overlap_regression(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    _coefficient_plot(ax, panel_data, spec, "s7_overlap_regression")


def render_s7_random_nonoverlap_perturbation(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = ["Overlap support", "Non-overlap support", "Random matched"]
    _dot_bar(ax, df, order, colors=[get_plot_color("true_pair"), get_plot_color("shuffled_pair"), get_plot_color("other_residual")])
    ax.set_xticks(np.arange(len(order)), ["Overlap", "Non", "Random"])
    ax.set_ylabel(str(spec.get("y_axis", "Dynamic-like recovery")))
    _tidy(ax)
    ax.paper_fig_plot_form = "s7_random_nonoverlap_perturbation"


def render_s8_time_resolved_l3(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty or "time_ms" not in df.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    colors = {
        "Dynamic": get_plot_color("dynamic"),
        "Static": get_plot_color("static_frozen"),
        "Overlap support": get_plot_color("true_pair"),
        "Non-overlap support": get_plot_color("shuffled_pair"),
        "Random matched": get_plot_color("other_residual"),
    }
    for condition, part in df.groupby("condition", sort=False):
        part = part.dropna(subset=["time_ms"])
        if part.empty:
            continue
        summary = _mean_sem(part, ["time_ms"]).sort_values("time_ms")
        color = colors.get(condition, COLOR_NEUTRAL)
        ax.plot(summary["time_ms"], summary["mean"], linewidth=0.85, color=color, label=str(condition).replace(" support", ""))
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.6)
    ax.set_xlabel(str(spec.get("x_axis", "Probe time")))
    ax.set_ylabel(str(spec.get("y_axis", "L3 dynamic-pattern index")))
    ax.legend(frameon=False, fontsize=4.7, loc="best", handlelength=0.9)
    _tidy(ax)
    ax.paper_fig_plot_form = "s8_time_resolved_l3"


def render_s8_decision_spike_summary(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = _ordered_unique(df["condition"], ["Overlap support", "Non-overlap support", "Random matched", "Dynamic", "Static"])
    _dot_bar(ax, df, order, color=get_plot_color("true_pair"))
    ax.set_xticks(np.arange(len(order)), [str(v).replace(" support", "").replace("Non-overlap", "Non") for v in order], rotation=20, ha="right")
    ax.set_ylabel(str(spec.get("y_axis", "Decision-step displacement")))
    _tidy(ax)
    ax.paper_fig_plot_form = "s8_decision_spike_summary"


def render_s8_l3_accumulator_replay_detail(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    metrics = _ordered_unique(df["metric"], ["replacement_push_kstar", "replacement_pullback_kstar", "deletion_dynamic_minus_static_kstar", "reconstruction_cosine_plus", "reconstruction_cosine_minus", "mean_static_to_dynamic_push", "mean_dynamic_to_static_pullback"])
    _dot_bar(ax, df, metrics, color=get_plot_color("sample_probe_overlap"))
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.6)
    ax.set_xticks(np.arange(len(metrics)), [_short_metric(m) for m in metrics], rotation=25, ha="right")
    ax.set_ylabel(str(spec.get("y_axis", "Effect")))
    _tidy(ax)
    ax.paper_fig_plot_form = "s8_l3_accumulator_replay_detail"


def render_s8_decision_deflection_summary(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if {"x_value", "y_value"}.issubset(df.columns):
        render_fig4_decision_deflection(ax, panel_data, stats, spec, style)
        ax.paper_fig_plot_form = "s8_decision_deflection_summary"
        return
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    metrics = _ordered_unique(df["metric"], ["static_to_dynamic_push", "dynamic_like_recovery", "decision_deflection_score", "mean_static_to_dynamic_push", "mean_dynamic_to_static_pullback"])
    _dot_bar(ax, df, metrics, color=get_plot_color("other_residual"))
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.6)
    ax.set_xticks(np.arange(len(metrics)), [_short_metric(m) for m in metrics], rotation=25, ha="right")
    ax.set_ylabel(str(spec.get("y_axis", "Dynamic-like recovery")))
    _tidy(ax)
    ax.paper_fig_plot_form = "s8_decision_deflection_summary"


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    for col in ("value", "time_ms", "similarity_bin_order"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["value"])


def _dot_bar(ax, df: pd.DataFrame, order: Sequence[str], *, color: str | None = None, colors: Sequence[str] | None = None) -> None:
    xs = np.arange(len(order), dtype=float)
    means = []
    sems = []
    for idx, key in enumerate(order):
        if key in set(df.get("condition", pd.Series(dtype=str)).astype(str)):
            vals = pd.to_numeric(df.loc[df["condition"].astype(str).eq(key), "value"], errors="coerce").dropna()
        else:
            vals = pd.to_numeric(df.loc[df["metric"].astype(str).eq(key), "value"], errors="coerce").dropna()
        means.append(float(vals.mean()) if len(vals) else np.nan)
        sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)
        if len(vals):
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
            c = (colors[idx] if colors else color) or COLOR_NEUTRAL
            ax.scatter(np.full(len(vals), xs[idx]) + jitter, vals, s=8, color=c, alpha=0.35, zorder=3)
    bar_colors = list(colors) if colors else [color or COLOR_NEUTRAL] * len(order)
    ax.bar(xs, means, yerr=sems, color=bar_colors, edgecolor=COLOR_NEUTRAL, linewidth=0.45, alpha=0.68, capsize=1.8, zorder=2)


def _coefficient_plot(ax, panel_data: pd.DataFrame | None, spec: Mapping[str, Any], plot_form: str) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, None, spec, None)
        return
    order = _ordered_unique(df["condition"], ["overlap", "similarity", "sample_energy", "probe_energy", "beta_overlap", "beta_similarity", "beta_sample_energy", "beta_probe_energy"])
    _dot_bar(ax, df, order, color=get_plot_color("sample_probe_overlap"))
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.6)
    ax.set_xticks(np.arange(len(order)), [_short_metric(v) for v in order], rotation=25, ha="right")
    ax.set_ylabel(str(spec.get("y_axis", "Estimate")))
    _tidy(ax)
    ax.paper_fig_plot_form = plot_form


def _mean_sem(df: pd.DataFrame, group_cols: Sequence[str] | None = None) -> pd.DataFrame:
    group_cols = list(group_cols or ["condition"])
    grouped = df.groupby(group_cols, dropna=False, sort=True)["value"].agg(["mean", "count", "std"]).reset_index()
    grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    return grouped


def _ordered_unique(series: pd.Series, preferred: Sequence[str]) -> list[str]:
    present = [str(v) for v in series.dropna().astype(str).unique()]
    out = [item for item in preferred if item in present]
    out.extend(item for item in present if item not in out)
    return out


def _short_metric(metric: Any) -> str:
    text = str(metric)
    return (
        text.replace("sample_keep_", "")
        .replace("_only_dynamic", "")
        .replace("mean_", "")
        .replace("replacement_", "repl_")
        .replace("reconstruction_", "recon_")
        .replace("similarity_difference", "sim diff")
        .replace("sample_energy_rel_difference", "sample energy")
        .replace("probe_energy_rel_difference", "probe energy")
        .replace("dice_overlap_difference", "overlap")
        .replace("_", "\n")
    )


def _num(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float("nan") if pd.isna(numeric) else float(numeric)


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=GRID_ALPHA_SOFT, linewidth=0.35)
    ax.tick_params(axis="both", labelsize=5.0, pad=0.8, length=1.6, width=0.5, color=COLOR_NEUTRAL)
    ax.xaxis.label.set_size(5.4)
    ax.yaxis.label.set_size(5.4)
    ax.xaxis.labelpad = 0.5
    ax.yaxis.labelpad = 0.7

