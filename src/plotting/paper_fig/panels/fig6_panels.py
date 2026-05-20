from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator

from src.plotting.common.colors import get_plot_color
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder


def render_fig6_peak_source_attribution(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    x_col = "position_from_end" if "position_from_end" in df.columns else "x_value"
    _line_by_x(ax, df, x_col, "Peak loss fraction", show_points=False)
    ax.set_xlabel("Position from end")
    ax.set_title("Late updates source final peaks", fontsize=6.2, pad=1.5)
    ax.paper_fig_plot_form = "peak_source_attribution"
    _tidy(ax)


def render_fig6_peak_update_history(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    use = df[df["metric"].astype(str).eq("P_peak")].copy()
    if use.empty:
        use = df
    order = _ordered_by_x(use, "condition")
    _bar_summary(ax, use, order, ylabel="P(peak)", emphasize="")
    ax.set_xlabel("Update count")
    ax.set_title("Repeated updates enrich peaks", fontsize=6.2, pad=1.5)
    ax.paper_fig_plot_form = "peak_update_history"
    _tidy(ax)


def render_fig6_peak_input_overlap_origin(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = _ordered_by_x(df, "condition")
    _bar_summary(ax, df, order, ylabel="P(high-overlap | peak)", emphasize="Recent-3")
    ax.set_xlabel("")
    ax.set_title("Peaks align with recent overlap routes", fontsize=6.2, pad=1.5)
    ax.paper_fig_plot_form = "peak_overlap_alignment"
    _tidy(ax)


def render_fig6_real_peak_overlap_reentry(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    render_fig6_route_peak_reentry_loss(ax, panel_data, stats, spec, style)


def render_fig6_real_peak_overlap_downstream(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    render_fig6_route_peak_downstream(ax, panel_data, stats, spec, style)


def render_fig6_route_peak_reentry_loss(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    raw = panel_data if panel_data is not None else pd.DataFrame()
    df = _clean(panel_data)
    use = df[df.get("metric", pd.Series(dtype=str)).astype(str).eq("normalized_reentry_loss")].copy() if not df.empty else df
    if use.empty:
        if raw is not None and not raw.empty and raw.get("metric", pd.Series(dtype=str)).astype(str).eq("normalized_reentry_loss").any():
            _diagnostic_panel(ax, "Route-peak perturbation\nnot cleared", "claim remains predictive")
            ax.paper_fig_plot_form = "route_peak_reentry_loss_diagnostic"
        else:
            render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = _ordered_unique(use["condition"], ["Route peak", "Route non-peak", "Non-route peak", "Random"])
    _bar_summary(ax, use, order, ylabel="Normalized re-entry loss", emphasize="Route peak")
    ax.axhline(0, color="0.45", linewidth=0.55)
    ax.set_xlabel("")
    ax.set_title("Route-peak perturbation reduces re-entry", fontsize=6.2, pad=1.5)
    ax.paper_fig_plot_form = "route_peak_reentry_loss"
    _annotate_claim_boundary(ax, use)
    _tidy(ax)


def render_fig6_route_peak_downstream(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    raw = panel_data if panel_data is not None else pd.DataFrame()
    df = _clean(panel_data)
    use = df[df.get("metric", pd.Series(dtype=str)).astype(str).eq("P_output_switch")].copy() if not df.empty else df
    if use.empty:
        if raw is not None and not raw.empty and raw.get("metric", pd.Series(dtype=str)).astype(str).eq("P_output_switch").any():
            _diagnostic_panel(ax, "Downstream perturbation\nnot cleared", "claim remains predictive")
            ax.paper_fig_plot_form = "route_peak_downstream_diagnostic"
        else:
            render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = _ordered_unique(use["condition"], ["Route peak", "Route non-peak", "Non-route peak", "Random"])
    _bar_summary(ax, use, order, ylabel="Output switch probability", emphasize="Route peak")
    ax.set_xlabel("")
    ax.set_title("Route-peak perturbation changes downstream output", fontsize=6.2, pad=1.5)
    ax.paper_fig_plot_form = "route_peak_downstream_output_switch"
    _annotate_claim_boundary(ax, use)
    _tidy(ax)


def render_fig6_multi_recent_peak_enrichment(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = list(spec.get("conditions") or ["Single old", "Single recent", "Multi old", "Multi recent"])
    _dot_bar(ax, df, order, ylabel="P(peak)", emphasize=str(spec.get("emphasize_condition", "Multi recent")))
    ax.set_xlabel("Update history")
    ax.set_ylabel("P(peak)")
    ax.paper_fig_plot_form = "multi_recent_peak_enrichment"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(df))
    _tidy(ax)


def render_fig6_update_recency_model(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = list(spec.get("conditions") or ["Base", "Update", "Recency", "Overlap", "Upd+Rec", "Upd x Rec"])
    _dot_bar(ax, df, order, ylabel="CV R2", emphasize="Upd+Rec")
    ax.set_xlabel("Model")
    ax.set_ylabel("CV R2")
    ax.paper_fig_plot_form = "update_recency_model_comparison"
    _tidy(ax)


def render_fig6_peak_weighted_overlap_interface(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty or not {"raw_overlap", "peak_weighted_overlap"}.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    cols = ["network_id", "sequence_id", "probe_id", "raw_overlap", "peak_weighted_overlap"]
    color_col = "peak_overlap_fraction" if "peak_overlap_fraction" in df.columns else None
    if color_col:
        cols.append(color_col)
    data = df.drop_duplicates([col for col in ["network_id", "sequence_id", "probe_id"] if col in df.columns])[cols].copy()
    for col in ("raw_overlap", "peak_weighted_overlap", "peak_overlap_fraction"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["raw_overlap", "peak_weighted_overlap"])
    if data.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    colors = data[color_col] if color_col and data[color_col].notna().any() else get_plot_color("peak_region")
    ax.scatter(data["raw_overlap"], data["peak_weighted_overlap"], c=colors, s=12, cmap="viridis", alpha=0.72, linewidths=0)
    lo = float(np.nanmin([data["raw_overlap"].min(), data["peak_weighted_overlap"].min()]))
    hi = float(np.nanmax([data["raw_overlap"].max(), data["peak_weighted_overlap"].max()]))
    if hi > lo:
        ax.plot([lo, hi], [lo, hi], color="0.25", linewidth=0.7, linestyle="--")
    ax.text(0.04, 0.96, "route: raw overlap", transform=ax.transAxes, ha="left", va="top", fontsize=6.0)
    ax.text(0.04, 0.84, "gain: peak-weighted overlap", transform=ax.transAxes, ha="left", va="top", fontsize=6.0)
    ax.set_xlabel("Raw overlap")
    ax.set_ylabel("Peak-weighted overlap")
    ax.paper_fig_has_colorbar = False
    ax.paper_fig_colorbar_removed = True
    ax.paper_fig_plot_form = "route_gain_scatter"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(data))
    ax.paper_fig_x_metric = "raw_overlap"
    ax.paper_fig_y_metric = "peak_weighted_overlap"
    _tidy(ax)


def render_fig6_peak_weighted_reentry(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    render_fig6_peak_weighted_real_reentry(ax, panel_data, stats, spec, style)


def render_fig6_peak_weighted_real_reentry(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    conditions = set(df["condition"].astype(str))
    if {"Low peak overlap", "High peak overlap"}.issubset(conditions):
        _paired_low_high(ax, df, "Low peak overlap", "High peak overlap", ylabel="Real re-entry strength")
        ax.set_xlabel("Raw-overlap matched group")
        ax.paper_fig_plot_form = "real_matched_peak_reentry"
    else:
        x = pd.to_numeric(df.get("peak_weighted_overlap"), errors="coerce")
        y = pd.to_numeric(df.get("value"), errors="coerce")
        data = pd.DataFrame({"x": x, "y": y}).dropna()
        if data.empty:
            render_generic_placeholder(ax, panel_data, stats, spec, style)
            return
        ax.scatter(data["x"], data["y"], s=10, alpha=0.65, color=get_plot_color("peak_region"), linewidths=0)
        _fit_line(ax, data["x"].to_numpy(), data["y"].to_numpy())
        ax.set_xlabel("Peak-weighted overlap")
        ax.set_ylabel("Real re-entry strength")
        ax.paper_fig_plot_form = "real_regression_peak_reentry"
    _annotate_audit(ax, df)
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(df))
    _tidy(ax)


def render_fig6_peak_weighted_downstream(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    render_fig6_peak_weighted_real_downstream(ax, panel_data, stats, spec, style)


def render_fig6_peak_weighted_real_downstream(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    nodes = ["Early recruitment", "Response reshaping", "Decision deflection"]
    _dot_bar(ax, df, nodes, ylabel="Real rollout effect", emphasize="")
    ax.set_xlabel("Downstream node")
    ax.set_ylabel("Real rollout effect")
    _annotate_audit(ax, df)
    ax.paper_fig_plot_form = "real_downstream_node_summary"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(df))
    _tidy(ax)


def render_fig6_global_mechanism(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = panel_data if panel_data is not None else pd.DataFrame()
    ax.axis("off")
    chain = list(spec.get("figure_chain") or [])
    if not chain:
        chain = [
            "Fig.1 functional STSP substrate",
            "Fig.2 fused two-item state",
            "Fig.3 multi-item peak landscape",
            "Fig.4 overlap re-entry route",
            "Fig.5 local support/competition conversion",
            "Fig.6 peak-amplified re-entry",
        ]
    xs = np.linspace(0.02, 0.82, len(chain))
    box_w = 0.14
    for idx, (x, label) in enumerate(zip(xs, chain)):
        ax.add_patch(Rectangle((x, 0.58), box_w, 0.25, transform=ax.transAxes, facecolor="0.96", edgecolor="0.25", linewidth=0.65))
        ax.text(x + box_w / 2, 0.705, _short_chain_label(label), transform=ax.transAxes, ha="center", va="center", fontsize=5.5)
        if idx < len(chain) - 1:
            ax.add_patch(FancyArrowPatch((x + box_w, 0.705), (xs[idx + 1], 0.705), transform=ax.transAxes, arrowstyle="->", mutation_scale=7, linewidth=0.75, color="0.25"))
    ax.text(0.18, 0.30, "Overlap = route", transform=ax.transAxes, ha="center", va="center", fontsize=8.4, fontweight="bold", color=get_plot_color("sample_probe_overlap"))
    ax.text(0.52, 0.30, "Peaks = gain", transform=ax.transAxes, ha="center", va="center", fontsize=8.4, fontweight="bold", color=get_plot_color("peak_region"))
    ax.text(0.35, 0.30, "+", transform=ax.transAxes, ha="center", va="center", fontsize=10.5, fontweight="bold", color="0.20")
    causal_ok = False
    if not df.empty:
        perturb = df.get("peak_perturbation_implemented", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"}).any()
        success = df.get("peak_perturbation_successful", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"}).any()
        allowed = " ".join(map(str, df.get("allowed_claim_strength", []))).lower()
        causal_ok = perturb and success and "causal" in allowed
    statement = "causal route-peak gain supported" if causal_ok else "Peak-amplified overlap-aligned re-entry"
    ax.text(0.98, 0.10, statement, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4, color="0.25")
    ax.paper_fig_plot_form = "global_mechanism_schematic"
    ax.paper_fig_route_gain_statement = True
    ax.paper_fig_peaks_replace_overlap = False
    ax.paper_fig_causal_claim_allowed = bool(causal_ok)


def _dot_bar(ax, df: pd.DataFrame, order: Sequence[str], *, ylabel: str, emphasize: str = "") -> None:
    order = [str(item) for item in order if str(item) in set(df.get("condition", pd.Series(dtype=str)).astype(str))]
    if not order:
        order = list(dict.fromkeys(map(str, df["condition"].tolist())))
    rng = np.random.default_rng(606)
    xs = np.arange(len(order), dtype=float)
    for idx, condition in enumerate(order):
        vals = pd.to_numeric(df.loc[df["condition"].astype(str).eq(condition), "value"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        color = get_plot_color("peak_region" if condition == emphasize else "other_residual", context="fig6")
        sample = vals if vals.size <= 800 else vals[rng.choice(vals.size, 800, replace=False)]
        ax.scatter(np.full(sample.size, xs[idx]) + rng.uniform(-0.12, 0.12, size=sample.size), sample, s=8, alpha=0.32, color=color, linewidths=0, zorder=2)
        ax.errorbar([xs[idx]], [float(np.mean(vals))], yerr=[_sem(vals)], fmt="o", color="black", markerfacecolor=color if condition == emphasize else "white", markeredgecolor="black", capsize=2.0, markersize=3.6, zorder=4)
    ax.set_xticks(xs, [_wrap_label(v) for v in order])
    ax.set_xlim(-0.6, max(0.6, len(order) - 0.4))
    ax.set_ylabel(ylabel)


def _paired_low_high(ax, df: pd.DataFrame, low_label: str, high_label: str, *, ylabel: str) -> None:
    low = df[df["condition"].astype(str).eq(low_label)].copy()
    high = df[df["condition"].astype(str).eq(high_label)].copy()
    if "matched_group_id" in df.columns:
        for _, part in df[df["condition"].astype(str).isin([low_label, high_label])].groupby("matched_group_id", dropna=False):
            vals = []
            for label in (low_label, high_label):
                v = pd.to_numeric(part.loc[part["condition"].astype(str).eq(label), "value"], errors="coerce").dropna()
                vals.append(float(v.iloc[0]) if not v.empty else np.nan)
            if np.isfinite(vals).all():
                ax.plot([0, 1], vals, color="0.72", linewidth=0.45, alpha=0.55, zorder=1)
    for x, part, color in ((0, low, get_plot_color("low_overlap")), (1, high, get_plot_color("peak_region"))):
        vals = pd.to_numeric(part["value"], errors="coerce").dropna()
        if vals.empty:
            continue
        jitter = np.linspace(-0.10, 0.10, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=9, alpha=0.35, color=color, linewidths=0, zorder=2)
        ax.errorbar([x], [float(vals.mean())], yerr=[float(vals.sem()) if len(vals) > 1 else 0.0], fmt="o", color="black", markerfacecolor=color, capsize=2.0, markersize=4.0, zorder=4)
    ax.set_xticks([0, 1], ["Low\npeak", "High\npeak"])
    ax.set_ylabel(ylabel)


def _line_by_x(ax, df: pd.DataFrame, x_col: str, ylabel: str, *, show_points: bool = True) -> None:
    if x_col not in df.columns:
        render_generic_placeholder(ax, df, None, {"panel_id": ""}, None)
        return
    data = df[[x_col, "value"]].copy()
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna()
    if data.empty:
        return
    summary = data.groupby(x_col, as_index=False)["value"].mean().sort_values(x_col)
    if show_points:
        ax.scatter(data[x_col], data["value"], s=8, alpha=0.25, color=get_plot_color("peak_region"), linewidths=0)
    ax.plot(summary[x_col], summary["value"], marker="o", markersize=2.8, linewidth=0.9, color="0.15")
    ax.set_ylabel(ylabel)


def _bar_summary(ax, df: pd.DataFrame, order: Sequence[str], *, ylabel: str, emphasize: str = "") -> None:
    order = [str(item) for item in order if str(item) in set(df.get("condition", pd.Series(dtype=str)).astype(str))]
    if not order:
        order = list(dict.fromkeys(map(str, df.get("condition", pd.Series(dtype=str)).tolist())))
    xs = np.arange(len(order), dtype=float)
    for idx, condition in enumerate(order):
        part = df[df["condition"].astype(str).eq(condition)].copy()
        vals = pd.to_numeric(part.get("value", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        value = float(vals.mean())
        sem_vals = pd.to_numeric(part.get("sem", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        err = float(np.nanmean(sem_vals)) if sem_vals.size and np.isfinite(sem_vals).any() else _sem(vals)
        color = get_plot_color("peak_region" if condition == emphasize else "other_residual", context="fig6")
        ax.bar([xs[idx]], [value], width=0.62, color=color, edgecolor="0.20", linewidth=0.55, alpha=0.88, zorder=2)
        ax.errorbar([xs[idx]], [value], yerr=[err], fmt="none", color="0.12", linewidth=0.65, capsize=2.0, zorder=3)
    ax.set_xticks(xs, [_wrap_label(v) for v in order])
    ax.set_xlim(-0.6, max(0.6, len(order) - 0.4))
    ax.set_ylabel(ylabel)


def _ordered_by_x(df: pd.DataFrame, label_col: str) -> list[str]:
    if "x_value" not in df.columns:
        return _ordered_unique(df[label_col], [])
    work = df[[label_col, "x_value"]].copy()
    work["x_value"] = pd.to_numeric(work["x_value"], errors="coerce")
    work = work.dropna(subset=["x_value"]).sort_values("x_value")
    out: list[str] = []
    for label in work[label_col].astype(str).tolist():
        if label not in out:
            out.append(label)
    return out or _ordered_unique(df[label_col], [])


def _annotate_claim_boundary(ax, df: pd.DataFrame) -> None:
    if "final_scientific_use" in df.columns and not df["final_scientific_use"].astype(str).str.lower().isin({"true", "1", "yes"}).any():
        ax.text(0.98, 0.96, "diagnostic only", transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color="0.35")


def _diagnostic_panel(ax, headline: str, subline: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.58, headline, transform=ax.transAxes, ha="center", va="center", fontsize=7.0, color="0.22")
    ax.text(0.5, 0.34, subline, transform=ax.transAxes, ha="center", va="center", fontsize=6.0, color="0.38")


def _fit_line(ax, x: np.ndarray, y: np.ndarray) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return
    coef = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(float(np.min(x[mask])), float(np.max(x[mask])), 100)
    ax.plot(xs, coef[0] * xs + coef[1], color="black", linewidth=0.8)


def _annotate_audit(ax, df: pd.DataFrame) -> None:
    y = 0.96
    if df.get("proxy_mode", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"}).all():
        ax.text(0.98, y, "proxy only", transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color="0.35")
        ax.paper_fig_proxy_only = True
        y -= 0.11
    if df.get("final_scientific_use", pd.Series(dtype=str)).astype(str).str.lower().isin({"false", "0"}).all():
        ax.text(0.98, y, "not final-use", transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color="0.35")
        ax.paper_fig_not_final_use = True


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for col in ("raw_overlap", "peak_weighted_overlap", "peak_overlap_fraction", "nonpeak_overlap_fraction", "x_value", "y_value", "relative_position_from_end", "position_from_end", "sem"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["value"])


def _ordered_unique(series: pd.Series, preferred: Sequence[str]) -> list[str]:
    present = [str(v) for v in series.dropna().astype(str).unique()]
    out = [str(item) for item in preferred if str(item) in present]
    out.extend(item for item in present if item not in out)
    return out


def _sem(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size <= 1:
        return 0.0
    return float(np.std(clean, ddof=1) / np.sqrt(clean.size))


def _wrap_label(label: str) -> str:
    return (
        str(label)
        .replace("Single old", "Single\nold")
        .replace("Single recent", "Single\nrecent")
        .replace("Multi old", "Multi\nold")
        .replace("Multi recent", "Multi\nrecent")
        .replace("Baseline only", "Base")
        .replace("Update only", "Update")
        .replace("Recency only", "Recency")
        .replace("Overlap only", "Overlap")
        .replace("Upd x Rec", "Upd x\nRec")
        .replace("Update x recency", "Upd x\nRec")
        .replace("Low peak overlap", "Low\npeak")
        .replace("High peak overlap", "High\npeak")
        .replace("Nonpeak control", "Nonpeak\ncontrol")
        .replace("Prior-updated nonpeak", "Prior-updated\nnonpeak")
        .replace("Early recruitment", "Early\nrecruit.")
        .replace("Response reshaping", "Response\nreshape")
        .replace("Decision deflection", "Decision\ndeflect")
        .replace("Route non-peak", "Route\nnon-peak")
        .replace("Non-route peak", "Non-route\npeak")
        .replace("Route peak", "Route\npeak")
    )


def _short_chain_label(label: str) -> str:
    return (
        label.replace(" functional STSP substrate", "\nfunctional STSP")
        .replace(" fused two-item state", "\nfused state")
        .replace(" multi-item peak landscape", "\npeak landscape")
        .replace(" overlap re-entry route", "\noverlap route")
        .replace(" local support/competition conversion", "\nsupport/competition")
        .replace(" local support / competition conversion", "\nsupport/competition")
        .replace(" peak-amplified re-entry", "\npeak-amplified")
    )


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=5.0, pad=0.8, length=1.8, width=0.5)
    ax.xaxis.label.set_size(5.8)
    ax.yaxis.label.set_size(5.8)
    ax.xaxis.labelpad = 0.7
    ax.yaxis.labelpad = 0.8
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.grid(axis="y", alpha=0.16, linewidth=0.45)
