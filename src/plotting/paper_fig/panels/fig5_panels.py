from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from src.plotting.common.colors import get_plot_color
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder


STYLE = {
    "axis_labelsize": 6.1,
    "tick_labelsize": 5.2,
    "legend_fontsize": 5.1,
    "line_width": 0.85,
    "marker_size": 10.0,
    "bar_width": 0.62,
}
CONDITION_COLORS = {
    "Dynamic": get_plot_color("dynamic"),
    "Dynamic intact": get_plot_color("dynamic"),
    "Static frozen": get_plot_color("static_frozen"),
    "Attenuate L1 STSP": "#E45756",
    "Reset L1 STSP": "#4C78A8",
    "Attenuate STSP": "#E45756",
    "Reset STSP": "#4C78A8",
    "Attenuate overlap support": "#E45756",
    "Reset overlap support": "#4C78A8",
    "Sham perturbation": "#A0A0A0",
}
GROUP_COLORS = {
    "Overlap-dominant": "#D95F02",
    "Probe-only-dominant": "#1B9E77",
    "Balanced": "#7570B3",
    "Random matched": "#666666",
}
UNIT_LABELS_FALLBACK = {
    "overlap_dominant": "Overlap-dominant",
    "probe_only_dominant": "Probe-only-dominant",
    "balanced": "Balanced",
    "random_matched": "Random matched",
}


def render_fig5_preprobe_support(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    ax.paper_fig_plot_form = "fig5_preprobe_support"
    order = [g for g in ("Overlap-dominant", "Probe-only-dominant", "Random matched", "Balanced") if g in set(df["condition"])]
    _bar_with_points(ax, df, "condition", order, colors=[GROUP_COLORS.get(g, "0.6") for g in order], st=st)
    ax.set_xticks(np.arange(len(order)), [_wrap(g) for g in order], rotation=0)
    ax.set_ylabel(str(spec.get("y_axis", "Pre-probe STSP support")))
    ax.set_xlabel("")
    _tidy(ax, st)


def render_fig5_early_firing(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    ax.paper_fig_plot_form = "fig5_early_firing_transition"
    groups = [g for g in ("Overlap-dominant", "Probe-only-dominant", "Random matched") if g in set(df["condition"])]
    metrics = [m for m in ("P_advance", "P_recruit", "P_advance_plus_recruit") if m in set(df["metric"])]
    width = 0.22
    x = np.arange(len(groups), dtype=float)
    metric_colors = {"P_advance": "#4C78A8", "P_recruit": "#F58518", "P_advance_plus_recruit": "#54A24B"}
    for i, metric in enumerate(metrics):
        subset = df[df["metric"].eq(metric)]
        means, sems = _group_means(subset, "condition", groups)
        offset = (i - (len(metrics) - 1) / 2.0) * width
        ax.bar(x + offset, means, yerr=sems, width=width * 0.92, capsize=1.8, color=metric_colors.get(metric, "0.6"), edgecolor="black", linewidth=0.35, label=_metric_label(metric))
    ax.set_xticks(x, [_wrap(g) for g in groups], rotation=0)
    ax.set_ylabel(str(spec.get("y_axis", "Early transition probability")))
    ax.set_ylim(0, max(0.05, min(1.0, _finite_max(df["value"]) * 1.25)))
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="upper center", bbox_to_anchor=(0.5, 1.22), ncols=max(1, len(metrics)), handlelength=0.9, columnspacing=0.7)
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = len(metrics)
    _tidy(ax, st)


def render_fig5_early_firing_headline(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    df = df[df["metric"].astype(str).eq("transition_fraction")] if not df.empty else df
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    ax.paper_fig_plot_form = "fig5_early_transition_composition"
    order = [g for g in ("Overlap", "Probe-only", "Random") if g in set(df["condition"].astype(str))]
    x = np.arange(len(order), dtype=float)
    _draw_stacked_transition_bars(ax, df, "condition", order, x, st=st)
    ax.set_xticks(x, [_wrap(g) for g in order], rotation=0)
    ax.set_ylabel(str(spec.get("y_axis", "Transition proportion")))
    ax.set_xlabel("")
    ymax = max(0.08, min(1.05, _finite_max(df.get("total_transition_mass", df["value"])) * 1.18))
    ax.set_ylim(0, ymax)
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="lower center", bbox_to_anchor=(0.5, 1.015), ncols=3, handlelength=0.9, columnspacing=0.65)
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = 3
    ax.paper_fig_raw_points = False
    ax.paper_fig_raw_point_count = 0
    ax.paper_fig_y_metric = "transition_fraction"
    _tidy(ax, st)


def render_fig5_winner_loser_events(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "time_ms" not in df.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    ax.paper_fig_plot_form = "fig5_winner_loser_event_traces"
    colors = {"winner_delta_v": "#D95F02", "loser_delta_v": "#1B9E77", "loser_inhibition": "#7570B3"}
    default_metrics = ("winner_delta_v", "loser_delta_v", "loser_inhibition")
    metric_order = tuple(str(metric) for metric in (spec.get("metrics") or default_metrics) if str(metric) in colors)
    trace_labels = spec.get("trace_labels") or {}
    plotted_metrics: list[str] = []
    for metric in metric_order:
        part = df[df["metric"].eq(metric)].copy()
        if part.empty:
            continue
        plotted_metrics.append(metric)
        grouped = part.groupby("time_ms", as_index=False)["value"].agg(["mean", "sem"]).reset_index()
        x = grouped["time_ms"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        sem = grouped["sem"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, y, linewidth=st["line_width"], color=colors.get(metric, "0.2"), label=str(trace_labels.get(metric, _trace_label(metric))))
        if part["seed_id"].nunique() > 1:
            ax.fill_between(x, y - sem, y + sem, color=colors.get(metric, "0.2"), alpha=0.18, linewidth=0)
    ax.axvline(0, color="0.25", linewidth=0.6)
    ax.axhline(0, color="0.82", linewidth=0.5)
    times = pd.to_numeric(df["time_ms"], errors="coerce").dropna()
    if not times.empty:
        ax.set_xlim(float(times.min()) - 0.5, float(times.max()) + 0.5)
        ax.set_xticks([tick for tick in (-5, 0, 5, 10) if float(times.min()) <= tick <= float(times.max())])
    ax.set_xlabel(str(spec.get("x_axis", "Time from winner spike (ms)")))
    default_ylabel = "Dynamic minus static (baseline-corrected)" if bool((stats or {}).get("baseline_corrected")) else "Dynamic minus static"
    ax.set_ylabel(str(spec.get("y_axis", default_ylabel)))
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="upper left", bbox_to_anchor=(0.01, 0.99), ncols=1, handlelength=1.0, columnspacing=0.7)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.paper_fig_legend_above_plot = False
    ax.paper_fig_legend_ncols = 1
    ax.paper_fig_trace_metrics = plotted_metrics
    ax.paper_fig_raw_points = False
    ax.paper_fig_raw_point_count = 0
    _tidy(ax, st)
    ax.tick_params(axis="y", pad=-1.8)
    for label in ax.get_yticklabels():
        label.set_fontsize(max(4.0, st["tick_labelsize"] - 1.1))


def render_fig5_support_perturbation(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    ax.paper_fig_plot_form = "fig5_support_perturbation_nodes"
    nodes = [n for n in ("early_recruitment", "loser_inhibition", "spike_similarity", "decision_deflection") if n in set(df["metric"])]
    conditions = [c for c in CONDITION_COLORS if c in set(df["condition"])]
    width = min(0.16, 0.75 / max(1, len(conditions)))
    x = np.arange(len(nodes), dtype=float)
    for i, condition in enumerate(conditions):
        part = df[df["condition"].eq(condition)]
        means, sems = _group_means(part, "metric", nodes)
        offset = (i - (len(conditions) - 1) / 2.0) * width
        ax.bar(x + offset, means, yerr=sems, capsize=1.4, width=width * 0.92, color=CONDITION_COLORS.get(condition, "0.7"), edgecolor="black", linewidth=0.3, label=_condition_short(condition))
    ax.set_xticks(x, [_node_short(n) for n in nodes], rotation=0)
    ax.set_ylabel(str(spec.get("y_axis", "Condition value")))
    ax.yaxis.labelpad = 0.0
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="upper center", bbox_to_anchor=(0.5, 1.17), ncols=3, handlelength=0.9, columnspacing=0.55)
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = 3
    _tidy(ax, st)


def render_fig5_causal_perturbation_summary(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    df = df[df["metric"].astype(str).eq("transition_fraction")].copy()
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    ax.paper_fig_plot_form = "fig5_l1_stsp_transition_composition"
    conditions = [c for c in ("Dynamic", "Attenuate L1 STSP", "Reset L1 STSP") if c in set(df["condition"].astype(str))]
    x_positions = np.arange(len(conditions), dtype=float)
    _draw_stacked_transition_bars(ax, df, "condition", conditions, x_positions, st=st, width=0.62, hatch_by_condition=False)
    ax.set_xticks(x_positions, [_wrap(c) for c in conditions], rotation=0)
    ymax = max(0.08, min(1.05, _finite_max(df.get("total_transition_mass", df["value"])) * 1.16))
    ax.set_ylim(0, ymax)
    ax.set_ylabel(str(spec.get("y_axis", "Transition proportion")))
    ax.set_xlabel(str(spec.get("x_axis", "")))
    title = str(spec.get("title", "")).strip()
    if title:
        ax.set_title(title, fontsize=st["axis_labelsize"], pad=2.0)
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="lower center", bbox_to_anchor=(0.5, 1.015), ncols=3, handlelength=0.9, columnspacing=0.65)
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = 3
    ax.paper_fig_raw_points = False
    ax.paper_fig_raw_point_count = 0
    ax.paper_fig_y_metric = "transition_fraction"
    _tidy(ax, st)


def render_fig5_perturbation_transition_distribution(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    ax.paper_fig_plot_form = "fig5_perturbation_transition_distribution"
    base_metrics = ["P_advance", "P_recruit", "P_loss", "P_unchanged"]
    plot_df = df[df["metric"].isin(base_metrics)].copy()
    if plot_df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    conditions = [c for c in ("Dynamic intact", "Attenuate overlap support", "Reset overlap support") if c in set(plot_df["condition"])]
    groups = [g for g in ("Overlap-dominant", "Probe-only-dominant") if g in set(plot_df.get("unit_group_label", plot_df.get("condition", pd.Series(dtype=str))))]
    if not groups and "unit_group" in plot_df.columns:
        groups = [g for g in ("overlap_dominant", "probe_only_dominant") if g in set(plot_df["unit_group"])]
    metric_colors = {
        "P_advance": "#4C78A8",
        "P_recruit": "#F58518",
        "P_loss": "#E45756",
        "P_unchanged": "#BAB0AC",
    }
    x_positions: list[float] = []
    tick_labels: list[str] = []
    bar_width = 0.58
    xpos = 0.0
    group_col = "unit_group_label" if "unit_group_label" in plot_df.columns else "unit_group"
    for group in groups:
        for condition in conditions:
            subset = plot_df[plot_df["condition"].eq(condition) & plot_df[group_col].eq(group)]
            bottom = 0.0
            for metric in base_metrics:
                vals = pd.to_numeric(subset[subset["metric"].eq(metric)]["value"], errors="coerce").dropna()
                value = float(vals.mean()) if not vals.empty else 0.0
                ax.bar(xpos, value, bottom=bottom, width=bar_width, color=metric_colors[metric], edgecolor="black", linewidth=0.25, label=_metric_label(metric) if xpos == 0.0 else None)
                bottom += value
            x_positions.append(xpos)
            tick_labels.append(_condition_tiny(condition))
            xpos += 1.0
        if group != groups[-1]:
            xpos += 0.65
    ax.set_xticks(x_positions, tick_labels, rotation=0)
    if groups:
        centers = []
        start = 0.0
        for group in groups:
            centers.append(start + (len(conditions) - 1) / 2.0)
            start += len(conditions) + 0.65
        for center, group in zip(centers, groups):
            ax.text(center, 1.055, _wrap(UNIT_LABELS_FALLBACK.get(group, group)), ha="center", va="bottom", fontsize=st["tick_labelsize"], transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(str(spec.get("y_axis", "Transition probability")))
    ax.set_xlabel(str(spec.get("x_axis", "Condition")))
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="upper center", bbox_to_anchor=(0.5, 1.23), ncols=4, handlelength=0.9, columnspacing=0.7)
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = 4
    _tidy(ax, st)


def render_fig5_perturbation_summary(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    ax.paper_fig_plot_form = "fig5_perturbation_effect_summary"
    order = [label for label in [_node_label(n) for n in spec.get("nodes", [])] if label in set(df["condition"])]
    if not order:
        order = list(df["condition"].dropna().unique())
    means, sems = _group_means(df, "condition", order)
    y = np.arange(len(order), dtype=float)
    ax.barh(y, means, xerr=sems, height=0.58, color="#D95F02", edgecolor="black", linewidth=0.35, alpha=0.86)
    ax.axvline(0, color="0.25", linewidth=0.6)
    ax.set_yticks(y, [_wrap(label) for label in order])
    ax.invert_yaxis()
    ax.set_xlabel(str(spec.get("y_axis", "Normalized overlap disruption")))
    ax.set_ylabel("")
    vmax = max(0.1, float(np.nanmax(np.abs(means))) if len(means) else 0.1)
    ax.set_xlim(min(-0.05, -0.08 * vmax), vmax * 1.18)
    _tidy(ax, st)


def _bar_with_points(ax, df: pd.DataFrame, group_col: str, order: list[str], *, colors: list[str], st: Mapping[str, float]) -> None:
    x = np.arange(len(order), dtype=float)
    means, sems = _group_means(df, group_col, order)
    ax.bar(x, means, yerr=sems, capsize=1.8, width=st["bar_width"], color=colors, edgecolor="black", linewidth=0.35, alpha=0.88)
    ax.paper_fig_raw_points = False
    ax.paper_fig_raw_point_count = 0


def _draw_stacked_transition_bars(
    ax,
    df: pd.DataFrame,
    group_keys: str | list[str],
    order: list[Any],
    x_positions: list[float] | np.ndarray,
    *,
    st: Mapping[str, float],
    width: float | None = None,
    hatch_by_condition: bool = False,
) -> None:
    transition_order = ["advance", "recruit", "loss"]
    transition_colors = {"advance": "#4C78A8", "recruit": "#F58518", "loss": "#E45756"}
    hatches = {"Dynamic": "", "Attenuate": "///", "Reset": "\\\\\\", "Static": "..."}
    width = float(width if width is not None else st["bar_width"])
    first_bar = True
    for xpos, item in zip(x_positions, order):
        subset = _subset_for_item(df, group_keys, item)
        bottom = 0.0
        condition_label = item[-1] if isinstance(item, tuple) else item
        for transition in transition_order:
            vals = pd.to_numeric(subset[subset.get("transition_type", pd.Series(dtype=str)).astype(str).eq(transition)]["value"], errors="coerce").dropna()
            value = float(vals.mean()) if not vals.empty else 0.0
            ax.bar(
                float(xpos),
                value,
                bottom=bottom,
                width=width,
                color=transition_colors[transition],
                edgecolor="black",
                linewidth=0.28,
                alpha=0.90,
                hatch=hatches.get(str(condition_label), "") if hatch_by_condition else "",
                label=_transition_label(transition) if first_bar else None,
            )
            bottom += value
        first_bar = False
        total_vals = _total_mass_values(subset)
        if len(total_vals) > 1:
            mean_total = float(total_vals.mean())
            sem_total = float(total_vals.sem())
            if np.isfinite(sem_total) and sem_total > 0:
                ax.errorbar(float(xpos), mean_total, yerr=sem_total, fmt="none", ecolor="black", elinewidth=0.55, capsize=1.8, capthick=0.55, zorder=4)


def _subset_for_item(df: pd.DataFrame, group_keys: str | list[str], item: Any) -> pd.DataFrame:
    if isinstance(group_keys, str):
        return df[df[group_keys].astype(str).eq(str(item))]
    values = item if isinstance(item, tuple) else (item,)
    mask = pd.Series(True, index=df.index)
    for key, value in zip(group_keys, values):
        mask &= df[key].astype(str).eq(str(value))
    return df[mask]


def _total_mass_values(df: pd.DataFrame) -> pd.Series:
    if df.empty or "total_transition_mass" not in df.columns:
        return pd.Series(dtype=float)
    id_cols = [col for col in ("network_id", "seed_id", "trial_id", "unit_group", "perturbation_condition", "condition") if col in df.columns]
    unique = df[id_cols + ["total_transition_mass"]].drop_duplicates()
    return pd.to_numeric(unique["total_transition_mass"], errors="coerce").dropna()


def _group_means(df: pd.DataFrame, group_col: str, order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means = []
    sems = []
    for item in order:
        vals = pd.to_numeric(df[df[group_col].eq(item)]["value"], errors="coerce").dropna()
        means.append(float(vals.mean()) if not vals.empty else 0.0)
        sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)
    return np.asarray(means, dtype=float), np.asarray(sems, dtype=float)


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df


def _style(style: Mapping[str, Any] | None) -> dict[str, float]:
    out = dict(STYLE)
    out.update({k: v for k, v in dict(style or {}).items() if k in out})
    return out


def _tidy(ax, st: Mapping[str, float]) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=st["tick_labelsize"], width=0.55, length=2.2, pad=1.5)
    ax.xaxis.label.set_size(st["axis_labelsize"])
    ax.yaxis.label.set_size(st["axis_labelsize"])


def _wrap(label: str) -> str:
    return str(label).replace("Overlap-", "Overlap-\n").replace("Probe-only-", "Probe-only-\n").replace("Random ", "Random\n").replace("Early ", "Early\n").replace("Decision ", "Decision\n").replace("Spike ", "Spike\n")


def _metric_label(metric: str) -> str:
    return {"P_advance": "Advance", "P_recruit": "Recruit", "P_loss": "Loss", "P_advance_plus_recruit": "Advance+recruit"}.get(metric, metric)


def _trace_label(metric: str) -> str:
    return {"winner_delta_v": "Winner ΔV", "loser_delta_v": "Loser ΔV", "loser_inhibition": "Inhibition received by loser"}.get(metric, metric)


def _transition_label(transition_type: str) -> str:
    return {"advance": "Advance", "recruit": "Recruit", "loss": "Loss"}.get(transition_type, transition_type)


def _condition_short(condition: str) -> str:
    return {
        "Dynamic intact": "Dynamic",
        "Static frozen": "Static",
        "Attenuate overlap support": "Attenuate",
        "Reset overlap support": "Reset",
        "Sham perturbation": "Sham",
    }.get(condition, condition)


def _condition_tiny(condition: str) -> str:
    return {
        "Dynamic": "Dyn.",
        "Attenuate": "Atten.",
        "Reset": "Reset",
        "Static": "Static",
        "Dynamic intact": "Same",
        "Attenuate overlap support": "Atten.",
        "Reset overlap support": "Reset",
        "Sham perturbation": "Sham",
    }.get(condition, condition)


def _node_short(node: str) -> str:
    return {
        "early_recruitment": "Early\nrecruit",
        "loser_inhibition": "Loser\ninh.",
        "spike_similarity": "Spike\nsim.",
        "decision_deflection": "Decision\ndeflect",
    }.get(node, node)


def _node_label(node: str) -> str:
    return {
        "early_recruitment": "Early recruitment",
        "winner_voltage_advantage": "Winner voltage",
        "loser_inhibition": "Loser inhibition",
        "spike_pattern_displacement": "Spike pattern",
        "decision_deflection": "Decision deflection",
    }.get(node, node.replace("_", " "))


def _finite_max(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.max()) if not vals.empty else 0.0
