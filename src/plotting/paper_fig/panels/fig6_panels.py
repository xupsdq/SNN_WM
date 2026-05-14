from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder


def render_anchor_peak_linkage(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Keep Fig.6A as an intentionally blank panel slot."""
    ax.set_axis_off()
    ax.paper_fig_plot_form = "blank_panel"
    ax.paper_fig_blank_reason = "Fig.6A intentionally blank; direct support-loss/anchor-retreat data unavailable."


def render_peak_membership_by_update_history(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.6B peak fraction by update-history group."""
    _render_group_summary(ax, panel_data, stats, spec, list(spec.get("conditions") or []))


def render_repetition_recency_gain(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.6C final STSP gain by repetition x recency group."""
    _render_group_summary(ax, panel_data, stats, spec, list(spec.get("preferred_groups") or spec.get("conditions") or []))


def render_update_recency_model_comparison(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.6D paired model-comparison R2 plot."""
    _render_paired(ax, panel_data, stats, spec, list(spec.get("conditions") or []))
    delta = (stats or {}).get("model_comparison", {}).get("delta_r2_mean")
    if delta is not None:
        ax.text(0.04, 0.96, f"delta R2={float(delta):.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=7)


def render_peak_manipulation_effect(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.6E peak-flattened/intact/boosted spike enrichment."""
    _render_distribution_summary(ax, panel_data, stats, spec, list(spec.get("conditions") or []))


def render_probe_peak_overlap_dependency(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.6F probe-peak overlap versus intact-over-flattened benefit."""
    df = _clean(panel_data)
    if df.empty or not {"x_value", "y_value"}.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    _scatter_regression(ax, df, spec, stats, trend="quadratic")


def _scatter_regression(ax, df: pd.DataFrame, spec: Mapping[str, Any], stats: Mapping[str, Any] | None, *, trend: str = "linear") -> None:
    x = pd.to_numeric(df["x_value"], errors="coerce")
    y = pd.to_numeric(df["y_value"], errors="coerce")
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if data.empty:
        render_generic_placeholder(ax, df, stats, spec)
        return
    alpha = 0.24 if len(data) <= 1500 else 0.10
    size = 10 if len(data) <= 1500 else 6
    plot_data = data
    if len(plot_data) > 3000:
        plot_data = plot_data.sample(n=3000, random_state=608)
    ax.scatter(plot_data["x"], plot_data["y"], s=size, alpha=alpha, linewidths=0)
    x_values = data["x"].to_numpy(dtype=float)
    y_values = data["y"].to_numpy(dtype=float)
    fit_summary = None
    if trend == "quadratic" and len(data) >= 3 and np.unique(x_values).size >= 3:
        coefficients = np.polyfit(x_values, y_values, 2)
        xs = np.linspace(float(data["x"].min()), float(data["x"].max()), 100)
        ax.plot(xs, np.polyval(coefficients, xs), color="black", linewidth=1.0)
        y_hat = np.polyval(coefficients, x_values)
        fit_summary = _r2_score(y_values, y_hat)
    elif trend == "linear" and len(data) >= 3:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        xs = np.linspace(float(data["x"].min()), float(data["x"].max()), 100)
        ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.0)
    corr = (stats or {}).get("correlations") or {}
    if trend == "quadratic" and fit_summary is not None:
        ax.text(0.04, 0.94, f"quadratic R2={fit_summary:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6)
    elif corr.get("pearson_r") is not None:
        ax.text(0.04, 0.94, f"r={corr['pearson_r']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6)
    if data["x"].between(0, 1).all():
        ax.set_xlim(0, 1)
    ax.set_xlabel(_short_axis_label(spec, "x"))
    ax.set_ylabel(_short_axis_label(spec, "y"))
    ax.paper_fig_plot_form = "scatter_quadratic_fit" if trend == "quadratic" else "scatter_regression"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(data))
    ax.paper_fig_raw_point_alpha = float(alpha)
    _tidy(ax)


def _render_group_summary(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], conditions: list[str]) -> None:
    _ = stats
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec)
        return
    x_map = {condition: idx for idx, condition in enumerate(conditions)}
    rng = np.random.default_rng(606)
    point_count = 0
    for condition in conditions:
        vals = df[df["condition"].eq(condition)]["value"].dropna().to_numpy(dtype=float)
        if vals.size:
            plot_vals = vals
            if vals.size > 1000:
                plot_vals = vals[rng.choice(vals.size, size=1000, replace=False)]
            jitter = rng.uniform(-0.18, 0.18, size=plot_vals.size)
            alpha = 0.24 if vals.size <= 700 else 0.08
            ax.scatter(np.full(plot_vals.size, x_map[condition], dtype=float) + jitter, plot_vals, s=5, alpha=alpha, linewidths=0)
            point_count += int(vals.size)
    for condition in conditions:
        vals = df[df["condition"].eq(condition)]["value"].to_numpy(dtype=float)
        if vals.size:
            ax.errorbar([x_map[condition]], [float(np.nanmean(vals))], yerr=[_sem(vals)], fmt="o", color="black", capsize=3)
    ax.set_xlim(-0.75, max(len(conditions) - 0.25, 0.25))
    ax.set_xticks(range(len(conditions)), [_wrap_condition_label(c) for c in conditions], rotation=0, ha="center")
    if str(spec.get("metric", "")) == "peak_fraction":
        ax.set_ylim(bottom=0)
    ax.set_xlabel(_short_axis_label(spec, "x"))
    ax.set_ylabel(_short_axis_label(spec, "y"))
    ax.paper_fig_plot_form = "row_jitter_with_mean_sem"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(point_count)
    ax.paper_fig_rows_before_renderer_aggregation = int(len(df))
    ax.paper_fig_renderer_summarizes_row_level = True
    _tidy(ax)


def _render_paired(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], conditions: list[str]) -> None:
    _ = stats
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec)
        return
    x_map = {condition: idx for idx, condition in enumerate(conditions)}
    values = [df[df["condition"].eq(condition)]["value"].dropna().to_numpy(dtype=float) for condition in conditions]
    nonempty = [(x_map[condition], vals) for condition, vals in zip(conditions, values) if vals.size]
    if nonempty:
        box = ax.boxplot(
            [vals for _, vals in nonempty],
            positions=[pos for pos, _ in nonempty],
            widths=0.54,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.9},
            boxprops={"facecolor": "0.82", "edgecolor": "0.35", "linewidth": 0.8},
            whiskerprops={"color": "0.35", "linewidth": 0.75},
            capprops={"color": "0.35", "linewidth": 0.75},
        )
        for patch in box["boxes"]:
            patch.set_alpha(0.88)
    rng = np.random.default_rng(609)
    point_count = 0
    for condition, vals in zip(conditions, values):
        if not vals.size:
            continue
        sample = vals
        if vals.size > 800:
            sample = vals[rng.choice(vals.size, size=800, replace=False)]
        jitter = rng.uniform(-0.12, 0.12, size=sample.size)
        ax.scatter(np.full(sample.size, x_map[condition], dtype=float) + jitter, sample, s=4, alpha=0.08, linewidths=0, color="0.25")
        point_count += int(sample.size)
    for condition in conditions:
        vals = df[df["condition"].eq(condition)]["value"].to_numpy(dtype=float)
        if vals.size:
            ax.scatter([x_map[condition]], [float(np.nanmean(vals))], marker="D", s=12, color="black", zorder=4)
    ax.set_xlim(-0.60, max(len(conditions) - 0.40, 0.40))
    ax.set_xticks(range(len(conditions)), [_wrap_condition_label(c) for c in conditions], rotation=0, ha="center")
    ax.set_xlabel(_short_axis_label(spec, "x"))
    ax.set_ylabel(_short_axis_label(spec, "y"))
    ax.paper_fig_plot_form = "boxplot_with_mean_marker"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(point_count)
    ax.paper_fig_rows_before_renderer_aggregation = int(len(df))
    ax.paper_fig_renderer_summarizes_row_level = True
    _tidy(ax)


def _render_distribution_summary(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], conditions: list[str]) -> None:
    _ = stats
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec)
        return
    values = [df[df["condition"].eq(condition)]["value"].dropna().to_numpy(dtype=float) for condition in conditions]
    positions = np.arange(len(conditions), dtype=float)
    nonempty = [(pos, vals) for pos, vals in zip(positions, values) if vals.size]
    if nonempty:
        violin = ax.violinplot([vals for _, vals in nonempty], positions=[pos for pos, _ in nonempty], widths=0.68, showmeans=False, showextrema=False, showmedians=False)
        for body in violin["bodies"]:
            body.set_facecolor("0.70")
            body.set_edgecolor("0.35")
            body.set_alpha(0.34)
    rng = np.random.default_rng(607)
    plotted_points = 0
    for pos, vals in zip(positions, values):
        if not vals.size:
            continue
        sample = vals
        if vals.size > 1200:
            sample = vals[rng.choice(vals.size, size=1200, replace=False)]
        jitter = rng.uniform(-0.22, 0.22, size=sample.size)
        ax.scatter(np.full(sample.size, pos, dtype=float) + jitter, sample, s=4, alpha=0.10, linewidths=0)
        plotted_points += int(sample.size)
        ax.errorbar([pos], [float(np.nanmean(vals))], yerr=[_sem(vals)], fmt="o", color="black", capsize=3, markersize=3.2)
    ax.set_xlim(-0.60, max(len(conditions) - 0.40, 0.40))
    ax.set_xticks(range(len(conditions)), [_wrap_condition_label(c) for c in conditions], rotation=0, ha="center")
    ax.set_xlabel(_short_axis_label(spec, "x"))
    ax.set_ylabel(_short_axis_label(spec, "y"))
    ax.paper_fig_plot_form = "row_distribution_violin_with_mean_sem"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(df))
    ax.paper_fig_raw_point_alpha = 0.10
    ax.paper_fig_rows_before_renderer_aggregation = int(len(df))
    ax.paper_fig_renderer_summarizes_row_level = True
    _tidy(ax)


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for col in ("x_value", "y_value", "probe_peak_overlap", "intact_over_flattened_benefit", "peak_fraction", "final_stsp_gain", "prediction_r2"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["value"])


def _sem(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size <= 1:
        return 0.0
    return float(np.nanstd(clean, ddof=1) / np.sqrt(clean.size))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    clean = np.isfinite(y_true) & np.isfinite(y_pred)
    if clean.sum() <= 1:
        return None
    y = y_true[clean]
    pred = y_pred[clean]
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 0:
        return None
    return float(1.0 - np.sum((y - pred) ** 2) / total)


def _id_col(df: pd.DataFrame) -> str | None:
    for col in ("source_row_id", "row_id", "trial_id", "seed_id", "network_id"):
        if col in df.columns and df[col].replace("", pd.NA).dropna().nunique() > 0:
            return col
    return None


def _wrap_condition_label(label: str) -> str:
    return (
        str(label)
        .replace("Multi-recent", "Multi\nrecent")
        .replace("Single-recent", "Single\nrecent")
        .replace("Multi-old", "Multi\nold")
        .replace("Single-old", "Single\nold")
        .replace("Overlap-only", "Overlap\nonly")
        .replace("Update + recency", "Update +\nrecency")
        .replace("Peak-flattened", "Peak\nflattened")
        .replace("Intact-final", "Intact\nfinal")
        .replace("Peak-boosted", "Peak\nboosted")
    )


def _short_axis_label(spec: Mapping[str, Any], axis: str) -> str:
    raw = str(spec.get(f"{axis}_axis", ""))
    x_replacements = {
        "Update-history group": "Update group",
        "Recency / update-history group": "Update group",
        "Predictor model": "Model",
        "Peak manipulation": "",
    }
    y_replacements = {
        "Intact-state benefit over peak-flattened state": "Intact benefit",
    }
    if axis == "x":
        replacements = x_replacements
    else:
        replacements = y_replacements
    return replacements.get(raw, raw)


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=5.2, pad=1.0, length=2)
    ax.xaxis.label.set_size(6.2)
    ax.yaxis.label.set_size(5.8)
    ax.xaxis.labelpad = 1.0
    ax.yaxis.labelpad = -3.5
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.margins(x=0.06)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
