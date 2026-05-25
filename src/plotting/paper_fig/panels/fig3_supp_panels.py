from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder
from src.plotting.paper_fig.panels.fig3_panels import PEAK, RANDOM, VALLEY, _infer_seq_len, _region_ping_categories


MAIN = "#009E73"
GRID = "0.88"


def render_s5_peak_valley_contrast(ax, panel_data, stats, spec, style=None):
    _region_plot(ax, panel_data, stats, spec, metric="support", ylabel="Support")
    ax.paper_fig_plot_form = "s5_peak_valley_contrast"


def render_s5_landscape_nonflatness(ax, panel_data, stats, spec, style=None):
    _metric_bar(ax, panel_data, stats, spec, metrics=["top_q_mass_fraction", "support_gini", "support_cv"], ylabel="Value")
    ax.paper_fig_plot_form = "s5_landscape_nonflatness"


def render_s5_peak_valley_null(ax, panel_data, stats, spec, style=None):
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = ["observed_peak_valley_delta", "null_peak_valley_delta_p95", "is_structured", "fraction_structured_sequences"]
    label_map = {
        "observed_peak_valley_delta": "Observed",
        "null_peak_valley_delta_p95": "Null p95",
        "is_structured": "Structured",
        "fraction_structured_sequences": "Fraction",
    }
    present = [metric for metric in order if metric in set(df["metric"].astype(str))]
    _metric_bar(ax, df, stats, spec, metrics=present or order, labels=[label_map.get(metric, metric) for metric in (present or order)], ylabel="Value")
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.55)
    ax.paper_fig_plot_form = "s5_peak_valley_null"


def render_s5_anchor_dynamics(ax, panel_data, stats, spec, style=None):
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("anchor_COM")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty or "stage_k" not in use.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    use["stage_k"] = pd.to_numeric(use["stage_k"], errors="coerce")
    summary = _summary(use, "stage_k", "value")
    ax.plot(summary["x"], summary["mean"], color=MAIN, marker="o", markersize=2.5, linewidth=1.2)
    ax.fill_between(summary["x"], summary["mean"] - summary["sem"], summary["mean"] + summary["sem"], color=MAIN, alpha=0.14, linewidth=0)
    ax.set_xlabel("Sequence stage")
    ax.set_ylabel("Anchor COM")
    _finish_axes(ax, spec)
    ax.paper_fig_plot_form = "s5_anchor_dynamics"


def render_s5_ping_recency_decomposition(ax, panel_data, stats, spec, style=None):
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("readout_mass")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty or "readout_class" not in use.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = [name for name in ["latest", "recent", "earlier", "silent"] if name in set(use["readout_class"].astype(str))]
    colors = {"latest": PEAK, "recent": "#E69F00", "earlier": MAIN, "silent": "0.78"}
    xs = np.arange(len(order))
    means = []
    sems = []
    for readout_class in order:
        vals = pd.to_numeric(use.loc[use["readout_class"].astype(str).eq(readout_class), "value"], errors="coerce").dropna().to_numpy(dtype=float)
        means.append(float(vals.mean()) if vals.size else 0.0)
        sems.append(_sem(vals) if vals.size else 0.0)
    ax.bar(xs, means, yerr=sems, width=0.64, color=[colors.get(name, MAIN) for name in order], edgecolor="white", linewidth=0.35, error_kw={"linewidth": 0.6, "capsize": 2.0})
    ax.set_xticks(xs, [name.title() for name in order], rotation=18, ha="right")
    ax.set_ylabel("Readout mass")
    upper = max(0.08, max([m + s for m, s in zip(means, sems)] or [0.0]) * 1.25)
    ax.set_ylim(0, min(1.05, upper if upper > 0 else 1.0))
    _finish_axes(ax, spec)
    ax.paper_fig_plot_form = "s5_ping_recency_decomposition"
    ax.paper_fig_readout_classes = order


def render_s5_weak_probe_recency_gain(ax, panel_data, stats, spec, style=None):
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("target_recovery_gain")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty or "target_position_bin" not in use.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = [name for name in ["early", "middle", "recent", "latest"] if name in set(use["target_position_bin"].astype(str))]
    xs = np.arange(len(order))
    means = []
    sems = []
    for target_bin in order:
        vals = pd.to_numeric(use.loc[use["target_position_bin"].astype(str).eq(target_bin), "value"], errors="coerce").dropna().to_numpy(dtype=float)
        means.append(float(vals.mean()) if vals.size else 0.0)
        sems.append(_sem(vals) if vals.size else 0.0)
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.55)
    ax.bar(xs, means, yerr=sems, width=0.64, color=MAIN, edgecolor="white", linewidth=0.35, error_kw={"linewidth": 0.6, "capsize": 2.0})
    ax.set_xticks(xs, [name.title() for name in order], rotation=18, ha="right")
    ax.set_ylabel("Recovery gain (pp)")
    if means:
        lo = min(m - s for m, s in zip(means, sems))
        hi = max(m + s for m, s in zip(means, sems))
        pad = max(2.0, (hi - lo) * 0.18)
        ax.set_ylim(min(0.0, lo - pad), max(0.0, hi + pad))
    _finish_axes(ax, spec)
    ax.paper_fig_plot_form = "s5_weak_probe_recency_gain"
    ax.paper_fig_y_metric = "target_recovery_gain"


def render_s6_ping_recency_diagnostics(ax, panel_data, stats, spec, style=None):
    _metric_bar(ax, panel_data, stats, spec, metrics=["latest_item_mass", "recent_item_mass", "earlier_item_residual_mass", "P_silent"], labels=["Latest", "Recent", "Earlier", "Silent"], ylabel="Readout mass")
    ax.paper_fig_plot_form = "s6_ping_recency_diagnostics"


def render_s6_weak_probe_target_source(ax, panel_data, stats, spec, style=None):
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    metric = "memory_gain" if df["metric"].astype(str).eq("memory_gain").any() else "P_target"
    use = df[df["metric"].astype(str).eq(metric)].copy()
    x_col = "target_source" if "target_source" in use.columns else "condition"
    _category_points(ax, use, x_col, "value", ylabel="Gain (%)" if metric == "memory_gain" else "Recovery (%)")
    _finish_axes(ax, spec)
    ax.paper_fig_plot_form = "s6_weak_probe_target_source"


def render_s6_peak_cue_matching(ax, panel_data, stats, spec, style=None):
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    cue_order = ["valley", "random", "peak"]
    metrics = [m for m in ["cue_pixel_count", "cue_energy", "encoded_spike_count"] if m in set(df["metric"].astype(str))]
    width = 0.22
    xs = np.arange(len(cue_order))
    for idx, metric in enumerate(metrics[:3]):
        vals = []
        for cue in cue_order:
            part = df[df["metric"].astype(str).eq(metric) & df["cue_condition"].astype(str).eq(cue)]
            vals.append(float(part["value"].mean()) if not part.empty else np.nan)
        norm = np.nanmax(np.abs(vals)) or 1.0
        ax.bar(xs + (idx - 1) * width, np.asarray(vals, dtype=float) / norm, width=width, label=metric.replace("_", " "), color=["#9CC9E2", "0.68", "#F0A37A"][idx])
    ax.set_xticks(xs, ["Valley", "Random", "Peak"])
    ax.set_ylabel("Normalized")
    ax.legend(frameon=False, fontsize=4.8, loc="upper right", handlelength=0.8)
    _finish_axes(ax, spec)
    ax.paper_fig_plot_form = "s6_peak_cue_matching"


def render_s6_peak_cue_state_vs_cue_only(ax, panel_data, stats, spec, style=None):
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    curve = df[df["metric"].astype(str).eq("P_target")].copy()
    if not curve.empty and "keep_fraction" in curve.columns:
        curve["keep_fraction"] = pd.to_numeric(curve["keep_fraction"], errors="coerce")
        colors = {"sequence_state": MAIN, "cue_only": "0.45"}
        for memory, part in curve.groupby("memory_condition", dropna=False):
            summary = _summary(part, "keep_fraction", "value")
            ax.plot(summary["x"], summary["mean"], marker="o", markersize=2.4, linewidth=1.1, color=colors.get(str(memory), "0.2"), label=str(memory).replace("_", " "))
        ax.set_xlabel("Keep fraction")
        ax.set_ylabel("Target recovery (%)")
        ax.legend(frameon=False, fontsize=5.0, loc="best", handlelength=0.9)
    else:
        gain = df[df["metric"].astype(str).eq("memory_gain")]
        _category_points(ax, gain, "cue_condition", "value", ylabel="Memory gain (%)")
    _finish_axes(ax, spec)
    ax.paper_fig_plot_form = "s6_peak_cue_state_vs_cue_only"


def render_s6_peak_cue_serial_position(ax, panel_data, stats, spec, style=None):
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("memory_gain")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    x_col = "target_position" if "target_position" in use.columns and pd.to_numeric(use["target_position"], errors="coerce").notna().any() else "target_position_bin"
    if x_col == "target_position":
        use[x_col] = pd.to_numeric(use[x_col], errors="coerce")
        summary = _summary(use, x_col, "value")
        ax.plot(summary["x"], summary["mean"], color=PEAK, marker="o", markersize=2.5, linewidth=1.1)
        ax.set_xlabel("Target position")
    else:
        _category_points(ax, use, x_col, "value", ylabel="Memory gain (%)")
        ax.set_xlabel("Position bin")
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.55)
    ax.set_ylabel("Memory gain (%)")
    _finish_axes(ax, spec)
    ax.paper_fig_plot_form = "s6_peak_cue_serial_position"


def render_s6_weak_probe_position_stratified(ax, panel_data, stats, spec, style=None):
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("P_target")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    bins = [b for b in ["early", "middle", "recent", "latest"] if b in set(use.get("target_position_bin", pd.Series(dtype=str)).astype(str))]
    memories = [m for m in ["cue_only", "single_item_memory", "sequence_state"] if m in set(use.get("memory_condition", pd.Series(dtype=str)).astype(str))]
    colors = {"cue_only": "0.45", "single_item_memory": "#CC79A7", "sequence_state": MAIN}
    labels = {"cue_only": "No memory", "single_item_memory": "Single item", "sequence_state": "Sequence"}
    xs = np.arange(len(bins))
    width = 0.22 if len(memories) > 1 else 0.5
    for idx, memory in enumerate(memories):
        vals = []
        for bin_name in bins:
            part = use[use["target_position_bin"].astype(str).eq(bin_name) & use["memory_condition"].astype(str).eq(memory)]
            vals.append(float(part["value"].mean()) if not part.empty else np.nan)
        offset = (idx - (len(memories) - 1) / 2) * width
        ax.bar(xs + offset, vals, width=width, color=colors.get(memory, "0.3"), label=labels.get(memory, memory))
    ax.set_xticks(xs, [b.title() for b in bins], rotation=12, ha="right")
    ax.set_ylabel("Target recovery (%)")
    ax.legend(frameon=False, fontsize=4.5, loc="upper left", handlelength=0.8)
    _finish_axes(ax, spec)
    ax.paper_fig_plot_form = "s6_weak_probe_position_stratified"


def render_s6_region_ping_current_matching(ax, panel_data, stats, spec, style=None):
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    metrics = [m for m in ["active_unit_count_mean", "total_ping_current_mean"] if m in set(df["metric"].astype(str))]
    regions = [r for r in ["peak", "valley", "random"] if r in set(df.get("region_condition", pd.Series(dtype=str)).astype(str))]
    xs = np.arange(len(regions))
    width = 0.32
    for idx, metric in enumerate(metrics):
        vals = []
        for region in regions:
            part = df[df["region_condition"].astype(str).eq(region) & df["metric"].astype(str).eq(metric)]
            vals.append(float(part["value"].mean()) if not part.empty else np.nan)
        norm = np.nanmax(np.abs(vals)) or 1.0
        ax.bar(xs + (idx - 0.5) * width, np.asarray(vals, dtype=float) / norm, width=width, label=metric.replace("_mean", "").replace("_", " "), color=["#8FBBD9", "#E7B66B"][idx])
    ax.set_xticks(xs, [r.title() for r in regions])
    ax.set_ylabel("Normalized")
    ax.legend(frameon=False, fontsize=4.5, loc="upper right", handlelength=0.8)
    _finish_axes(ax, spec)
    ax.paper_fig_plot_form = "s6_region_ping_current_matching"


def render_s6_region_ping_s0_control(ax, panel_data, stats, spec, style=None):
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if df["metric"].astype(str).eq("readout_mass").any() and "serial_bin" in df.columns:
        use = df[df["metric"].astype(str).eq("readout_mass")].copy()
        regions = [r for r in ["peak", "valley", "random"] if r in set(use["region_condition"].astype(str))]
        cats = ["latest", "recent", "earlier", "other", "silent"]
        colors = {"latest": PEAK, "recent": "#E69F00", "earlier": MAIN, "other": "0.68", "silent": "0.86"}
        agg = _region_ping_categories(use, regions, _infer_seq_len(use))
        xs = np.arange(len(regions))
        bottom = np.zeros(len(regions), dtype=float)
        for cat in cats:
            vals = np.asarray([agg.get(region, {}).get(cat, 0.0) for region in regions], dtype=float)
            ax.bar(xs, vals, bottom=bottom, width=0.62, color=colors[cat], edgecolor="white", linewidth=0.35, label=cat.title())
            bottom += vals
        ax.set_xticks(xs, [r.title() for r in regions])
        ax.set_ylabel("S0 readout mass")
        ax.legend(frameon=False, fontsize=4.2, loc="upper right", handlelength=0.7)
    else:
        _category_points(ax, df, "region_condition", "value", ylabel="S0 probability")
    _finish_axes(ax, spec)
    ax.paper_fig_plot_form = "s6_region_ping_s0_control"


def render_s6_region_ping_latency(ax, panel_data, stats, spec, style=None):
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    metric = "P_seen_item" if "ping_amp" in df.columns and pd.to_numeric(df["ping_amp"], errors="coerce").notna().any() else "median_first_fire_time_ms"
    use = df[df["metric"].astype(str).eq(metric)].copy()
    if use.empty:
        use = df.copy()
    if "ping_amp" in use.columns and pd.to_numeric(use["ping_amp"], errors="coerce").notna().any():
        use["ping_amp"] = pd.to_numeric(use["ping_amp"], errors="coerce")
        colors = {"peak": PEAK, "valley": VALLEY, "random": RANDOM}
        for region, part in use.groupby("region_condition", dropna=False):
            summary = _summary(part, "ping_amp", "value")
            ax.plot(summary["x"], summary["mean"], marker="o", markersize=2.4, linewidth=1.1, color=colors.get(str(region), "0.25"), label=str(region).title())
        ax.set_xlabel("Ping amplitude")
        ax.set_ylabel("P(seen)" if metric == "P_seen_item" else "Latency (ms)")
        ax.legend(frameon=False, fontsize=4.6, loc="best", handlelength=0.8)
    else:
        _category_points(ax, use, "region_condition", "value", ylabel="Latency (ms)")
    _finish_axes(ax, spec)
    ax.paper_fig_plot_form = "s6_region_ping_latency"


render_s6_region_ping_amp_sweep = render_s6_region_ping_latency


def _metric_bar(ax, panel_data, stats, spec, *, metrics, labels=None, ylabel="Value"):
    _ = stats
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, None)
        return
    labels = labels or [m.replace("_", " ") for m in metrics]
    xs = np.arange(len(metrics))
    for x, metric in zip(xs, metrics):
        vals = pd.to_numeric(df.loc[df["metric"].astype(str).eq(metric), "value"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size > 1 else np.array([0.0])
        ax.scatter(np.full(vals.size, x) + jitter, vals, s=8, color="0.35", alpha=0.5, linewidth=0)
        ax.errorbar(x, vals.mean(), yerr=_sem(vals), fmt="o", color=MAIN, markersize=3.0, capsize=2.0, linewidth=0.75)
    ax.set_xticks(xs, labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    _finish_axes(ax, spec)


def _region_plot(ax, panel_data, stats, spec, *, metric, ylabel):
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq(metric)].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty:
        use = df[df["metric"].astype(str).eq("peak_valley_delta")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, None)
        return
    x_col = "region" if "region" in use.columns else "condition"
    _category_points(ax, use, x_col, "value", ylabel=ylabel)
    _finish_axes(ax, spec)


def _category_points(ax, df, x_col, y_col, *, ylabel):
    if df.empty or x_col not in df.columns:
        return
    raw_order = ["valley", "random", "peak"]
    seen = [str(v) for v in df[x_col].dropna().unique()]
    order = [v for v in raw_order if v in seen] + [v for v in seen if v not in raw_order]
    colors = {"valley": VALLEY, "random": RANDOM, "peak": PEAK}
    xs = np.arange(len(order))
    for x, key in zip(xs, order):
        vals = pd.to_numeric(df.loc[df[x_col].astype(str).eq(key), y_col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size > 1 else np.array([0.0])
        ax.scatter(np.full(vals.size, x) + jitter, vals, s=9, color=colors.get(key, MAIN), alpha=0.55, linewidth=0)
        ax.errorbar(x, vals.mean(), yerr=_sem(vals), fmt="o", color="0.12", markersize=3.0, capsize=2.0, linewidth=0.75)
    ax.set_xticks(xs, [key.replace("_", " ").title() for key in order], rotation=18, ha="right")
    ax.set_ylabel(ylabel)


def _clean(panel_data):
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def _summary(df, x_col, y_col):
    rows = []
    for x, part in df.groupby(x_col, sort=True):
        vals = pd.to_numeric(part[y_col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size:
            rows.append({"x": float(x), "mean": float(vals.mean()), "sem": _sem(vals)})
    return pd.DataFrame(rows).sort_values("x") if rows else pd.DataFrame(columns=["x", "mean", "sem"])


def _sem(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return 0.0 if arr.size <= 1 else float(arr.std(ddof=1) / np.sqrt(arr.size))


def _finish_axes(ax, spec):
    title = str(spec.get("title", "")).strip()
    if title:
        ax.set_title(title, fontsize=6.4, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.35, alpha=0.65)
    ax.tick_params(axis="both", labelsize=5.0, pad=0.8, length=1.8, width=0.5)
    ax.xaxis.label.set_size(5.7)
    ax.yaxis.label.set_size(5.7)
    ax.xaxis.labelpad = 0.8
    ax.yaxis.labelpad = 0.8
