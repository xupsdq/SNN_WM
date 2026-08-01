from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.plotting.common.colors import NATURE_COMPATIBLE_PALETTE as PALETTE, get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL, GRID_ALPHA_SOFT
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder
from src.plotting.paper_fig.panels.fig5_panels import CONDITION_COLORS, GROUP_COLORS


STYLE = {
    "axis_labelsize": 9.0,
    "tick_labelsize": 8.5,
    "legend_fontsize": 7.8,
    "line_width": 1.15,
    "marker_size": 4.2,
}
TRANSITION_COLORS = {
    "P_advance": get_plot_color("transition_advance"),
    "P_recruit": get_plot_color("transition_recruit"),
    "P_loss": get_plot_color("transition_loss"),
    "P_unchanged": get_plot_color("transition_unchanged"),
    "P_same_winner_preserved": get_plot_color("transition_combined"),
    "P_same_winner_lost": get_plot_color("transition_loss"),
    "P_same_winner_delayed": get_plot_color("transition_recruit"),
    "P_same_winner_lost_or_delayed": get_plot_color("negative_result"),
}
FROZEN_GROUP_COLORS = {
    "overlap_dominant": get_plot_color("sample_probe_overlap"),
    "probe_only_dominant": get_plot_color("probe_only_region"),
    "random_matched": get_plot_color("random_control"),
    "balanced": get_plot_color("balanced_support"),
}
FROZEN_PERTURBATION_COLORS = {
    "Attenuate L1 STSP": get_plot_color("perturb_attenuate"),
    "Reset L1 STSP": get_plot_color("perturb_reset"),
    "Attenuate overlap support": get_plot_color("perturb_attenuate"),
    "Reset overlap support": get_plot_color("perturb_reset"),
}
FROZEN_PERTURBATION_HATCHES = {
    "Attenuate L1 STSP": "//",
    "Reset L1 STSP": "xx",
    "Attenuate overlap support": "//",
    "Reset overlap support": "xx",
}


def render_s9_early_window_robustness(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data
    st = _style(style)
    rows = _frozen_summary_rows(stats, "A")
    windows = [5.0, 10.0, 15.0, 20.0, 30.0]
    for group, label, marker in (
        ("overlap_dominant", "Overlap-dominant", "o"),
        ("probe_only_dominant", "Probe-only-dominant", "s"),
        ("random_matched", "Random matched", "D"),
        ("balanced", "Balanced", "^"),
    ):
        values = [_frozen_summary_value(rows, early_window_ms=window, unit_group=group) for window in windows]
        means = [value[0] for value in values]
        sems = [value[1] for value in values]
        ax.errorbar(
            windows,
            means,
            yerr=sems,
            fmt=f"{marker}-",
            markersize=st["marker_size"],
            linewidth=st["line_width"],
            elinewidth=st["line_width"],
            markeredgewidth=0.8,
            capsize=2.0,
            color=FROZEN_GROUP_COLORS[group],
            label=label,
        )
    ax.set_xlabel(str(spec.get("x_axis", "Early window (ms)")))
    ax.set_ylabel(str(spec.get("y_axis", "P(advance OR recruit) (proportion)")))
    ax.set_xticks(windows)
    ax.set_ylim(0.0, 0.62)
    ax.legend(
        frameon=False,
        fontsize=st["legend_fontsize"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=2,
        handlelength=1.2,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    ax.paper_fig_plot_form = "s9_frozen_early_window_robustness"
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = 2
    ax.paper_fig_no_recompute = True
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
    _ = panel_data
    st = _style(style)
    rows = _frozen_summary_rows(stats, "B")
    order = ["vs probe-only", "vs random", "vs balanced"]
    controls = ["probe_only_dominant", "random_matched", "balanced"]
    xs = np.arange(len(order), dtype=float)
    values = [
        _frozen_summary_value(
            rows,
            condition=condition,
            control_group=control,
            metric="delta_P_advance_plus_recruit",
        )
        for condition, control in zip(order, controls)
    ]
    means = np.asarray([value[0] for value in values], dtype=float)
    sems = np.asarray([value[1] for value in values], dtype=float)
    colors = [
        get_plot_color("probe_only_region"),
        get_plot_color("random_control"),
        get_plot_color("balanced_support"),
    ]
    ax.bar(
        xs,
        means,
        yerr=sems,
        capsize=2.0,
        width=0.58,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        error_kw={"elinewidth": st["line_width"], "capthick": st["line_width"]},
    )
    ax.axhline(0, color="0.35", linewidth=st["line_width"])
    ax.set_ylim(-0.02, 0.52)
    ax.set_xticks(xs, [_comparison_short(label) for label in order], rotation=0)
    ax.set_ylabel(str(spec.get("y_axis", "ΔP(advance OR recruit) (proportion)")))
    ax.paper_fig_plot_form = "s9_frozen_trialwise_transition_advantage"
    ax.paper_fig_raw_points = False
    ax.paper_fig_value_labels = False
    ax.paper_fig_value_label_count = 0
    ax.paper_fig_value_labels_clear = True
    ax.paper_fig_fraction_positive_annotations = False
    ax.paper_fig_no_recompute = True
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
            ("winner_pre_spike_delta_v_mean", get_plot_color("winner")),
            ("loser_post_winner_inh_rise", get_plot_color("inhibition")),
            ("loser_post_winner_suppressed", get_plot_color("loser")),
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
    _ = panel_data
    st = _style(style)
    rows = _frozen_summary_rows(stats, "C")
    ax.set_axis_off()
    upper = ax.inset_axes([0.0, 20.0 / 34.0, 1.0, 14.0 / 34.0])
    lower = ax.inset_axes([0.0, 0.0, 1.0, 14.0 / 34.0])
    conditions = ["Attenuate L1 STSP", "Reset L1 STSP"]
    xs = np.arange(2, dtype=float)
    for subax, metric, ylabel, ylim in (
        (upper, "u_delta_mean", "Δu", (-0.043, 0.004)),
        (lower, "x_delta_mean", "Δx", (-0.0001, 0.00105)),
    ):
        values = [_frozen_summary_value(rows, condition=condition, metric=metric) for condition in conditions]
        means = [value[0] for value in values]
        sems = [value[1] for value in values]
        bars = subax.bar(
            xs,
            means,
            yerr=sems,
            capsize=1.8,
            width=0.58,
            color=[FROZEN_PERTURBATION_COLORS[condition] for condition in conditions],
            edgecolor="black",
            linewidth=0.8,
            error_kw={"elinewidth": st["line_width"], "capthick": st["line_width"]},
        )
        for bar, condition in zip(bars, conditions):
            bar.set_hatch(FROZEN_PERTURBATION_HATCHES[condition])
        subax.axhline(0, color="0.35", linewidth=st["line_width"])
        subax.set_ylabel(ylabel)
        subax.set_ylim(*ylim)
        subax.set_xticks(xs)
        _tidy(subax, st)
    upper.set_xticklabels([])
    upper.tick_params(axis="x", length=0)
    lower.set_xticklabels(["Attenuate", "Reset"])
    lower.set_xlabel(str(spec.get("x_axis", "L1 STSP perturbation")))
    for subax in (upper, lower):
        subax.tick_params(axis="both", labelsize=6.8, pad=0.8)
        subax.yaxis.label.set_size(8.0)
    lower.xaxis.label.set_size(7.5)
    ax.paper_fig_plot_form = "s9_frozen_perturbation_ux_split_scale"
    ax.paper_fig_internal_axes = (upper, lower)
    ax.paper_fig_split_scale = True
    ax.paper_fig_no_recompute = True


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
    _ = panel_data
    st = _style(style)
    rows = _frozen_summary_rows(stats, "D")
    groups = ["overlap_dominant", "probe_only_dominant"]
    conditions = ["Attenuate overlap support", "Reset overlap support"]
    x = np.arange(len(groups), dtype=float)
    width = 0.30
    for index, condition in enumerate(conditions):
        values = [
            _frozen_summary_value(
                rows,
                condition=condition,
                metric="delta_P_advance_plus_recruit",
                unit_group=group,
            )
            for group in groups
        ]
        means = [value[0] for value in values]
        sems = [value[1] for value in values]
        bars = ax.bar(
            x + (index - 0.5) * width,
            means,
            yerr=sems,
            width=width * 0.90,
            capsize=1.8,
            color=FROZEN_PERTURBATION_COLORS[condition],
            edgecolor="black",
            linewidth=0.8,
            label="Attenuate" if index == 0 else "Reset",
            error_kw={"elinewidth": st["line_width"], "capthick": st["line_width"]},
        )
        for bar in bars:
            bar.set_hatch(FROZEN_PERTURBATION_HATCHES[condition])
    ax.axhline(0, color="0.35", linewidth=st["line_width"])
    ax.set_xticks(x, ["Overlap-\ndominant", "Probe-only-\ndominant"])
    ax.set_xlabel(str(spec.get("x_axis", "Unit group")))
    ax.set_ylabel(str(spec.get("y_axis", "Perturbed − dynamic\nΔP(advance OR recruit)\n(proportion)")))
    ax.set_ylim(-0.14, 0.012)
    ax.set_title("P(advance OR recruit) (proportion)", fontsize=6.0, pad=1.5)
    ax.legend(
        frameon=False,
        fontsize=st["legend_fontsize"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.16),
        ncols=2,
        handlelength=1.1,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    ax.paper_fig_plot_form = "s9_frozen_perturbation_transition_contrast"
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = 2
    ax.paper_fig_condition_order = tuple(conditions)
    ax.paper_fig_unit_group_order = tuple(groups)
    ax.paper_fig_no_recompute = True
    _tidy(ax, st)
    ax.tick_params(axis="both", labelsize=7.0, pad=0.8)
    ax.xaxis.label.set_size(7.5)
    ax.yaxis.label.set_size(7.0)


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
    _ = panel_data
    st = _style(style)
    rows = _frozen_summary_rows(stats, "E")
    groups = ["probe_only_dominant", "overlap_dominant"]
    conditions = ["Dynamic", "Attenuate overlap support", "Reset overlap support"]
    metrics = ["P_same_winner_preserved", "P_same_winner_lost", "P_same_winner_delayed"]
    x_identities = [(group, condition) for group in groups for condition in conditions]
    x = np.arange(len(x_identities), dtype=float)
    bottom = np.zeros(len(x), dtype=float)
    for metric in metrics:
        means = np.asarray(
            [
                _frozen_summary_value(rows, condition=condition, metric=metric, unit_group=group)[0]
                for group, condition in x_identities
            ],
            dtype=float,
        )
        ax.bar(
            x,
            means,
            bottom=bottom,
            width=0.72,
            color=TRANSITION_COLORS[metric],
            edgecolor="black",
            linewidth=0.8,
            label={
                "P_same_winner_preserved": "Preserved",
                "P_same_winner_lost": "Lost",
                "P_same_winner_delayed": "Delayed",
            }[metric],
        )
        bottom += means
    ax.set_xticks(x, ["Dyn.", "Atten.", "Reset", "Dyn.", "Atten.", "Reset"], rotation=38, ha="right", rotation_mode="anchor")
    ax.text(1.0, -0.25, "Probe-only", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.8, clip_on=False)
    ax.text(4.0, -0.25, "Overlap", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.8, clip_on=False)
    ax.set_xlabel("")
    ax.set_ylabel(str(spec.get("y_axis", "Same-winner outcome\nprobability (proportion)")))
    ax.set_ylim(0.0, 1.03)
    ax.legend(
        frameon=False,
        fontsize=st["legend_fontsize"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=3,
        handlelength=0.9,
        columnspacing=0.55,
        borderaxespad=0.0,
    )
    ax.paper_fig_plot_form = "s9_frozen_same_winner_three_component"
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = 3
    ax.paper_fig_unit_group_order = tuple(groups)
    ax.paper_fig_condition_order = tuple(conditions)
    ax.paper_fig_metric_order = tuple(metrics)
    ax.paper_fig_excluded_pooled_metric = "P_same_winner_lost_or_delayed"
    ax.paper_fig_no_recompute = True
    _tidy(ax, st)
    ax.tick_params(axis="both", labelsize=6.8, pad=0.6)
    ax.yaxis.label.set_size(6.8)


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
    colors = {"winner_delta_v": get_plot_color("winner"), "loser_delta_v": get_plot_color("loser"), "loser_inhibition": get_plot_color("inhibition")}
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


def _frozen_summary_rows(stats: Mapping[str, Any] | None, expected_panel_id: str) -> list[Mapping[str, Any]]:
    if not isinstance(stats, Mapping):
        raise RuntimeError(f"S5{expected_panel_id} requires its fixed statistics payload; none was supplied.")
    if stats.get("figure_id") != "supp_fig_s5" or stats.get("panel_id") != expected_panel_id:
        raise RuntimeError(f"S5{expected_panel_id} fixed statistics identity mismatch.")
    if stats.get("status") != "ok" or stats.get("run_mode") != "multi_network_final":
        raise RuntimeError(f"S5{expected_panel_id} fixed statistics status/run-mode mismatch.")
    rows = stats.get("summaries")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"S5{expected_panel_id} fixed statistics contain no summary rows.")
    return rows


def _frozen_summary_value(rows: Sequence[Mapping[str, Any]], **identity: Any) -> tuple[float, float]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in identity.items())]
    if len(matches) != 1:
        raise RuntimeError(f"Frozen S5 summary identity must match exactly once: {identity!r}; matches={len(matches)}.")
    row = matches[0]
    if "mean" not in row or "sem" not in row:
        raise RuntimeError(f"Frozen S5 summary row lacks mean/sem: {identity!r}.")
    return float(row["mean"]), float(row["sem"])


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
    ax.spines["left"].set_linewidth(st["line_width"])
    ax.spines["bottom"].set_linewidth(st["line_width"])
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=st["tick_labelsize"], pad=1.2, length=2.4, width=st["line_width"], color=COLOR_NEUTRAL)
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
