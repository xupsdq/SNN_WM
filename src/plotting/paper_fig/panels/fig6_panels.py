from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator
from scipy import stats as scipy_stats

from src.plotting.common.colors import get_plot_color
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder


def render_fig6_entry_gated_score_schematic(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    ax.axis("off")
    boxes = [
        (0.03, 0.58, 0.21, 0.25, "multi-item\nsequence", "0.96"),
        (0.31, 0.58, 0.24, 0.25, "STSP gain field\nrho = G_final/G_base", "0.94"),
        (0.63, 0.58, 0.22, 0.25, "entry mask\nping or probe", "0.96"),
        (0.64, 0.16, 0.25, 0.20, "Layer 1\nrecruitment", "0.94"),
    ]
    for x, y, w, h, label, face in boxes:
        ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=face, edgecolor="0.25", linewidth=0.65))
        ax.text(x + w / 2, y + h / 2, label, transform=ax.transAxes, ha="center", va="center", fontsize=6.2)
    arrows = [((0.24, 0.705), (0.31, 0.705)), ((0.55, 0.705), (0.63, 0.705)), ((0.74, 0.58), (0.74, 0.36))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="->", mutation_scale=7, linewidth=0.75, color="0.25"))
    formula = "S_p(E) = sum_RF E(q) rho(q) / [sum_RF E(q) + eps]"
    ax.text(0.04, 0.29, formula, transform=ax.transAxes, ha="left", va="center", fontsize=7.0, fontweight="bold")
    ax.text(0.04, 0.13, "mean gain ratio over entry-active sites", transform=ax.transAxes, ha="left", va="center", fontsize=5.8, color="0.25")
    ax.paper_fig_plot_form = "entry_gated_score_schematic"
    ax.paper_fig_score_name = "entry_gated_stsp_gain_score"
    ax.paper_fig_score_excludes = ["connection_weights", "inhibition", "voltage", "threshold", "WTA", "final_label"]


def render_fig6_high_stsp_overlap_ablation(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    use = df[df.get("metric", pd.Series(dtype=str)).astype(str).eq("loss_delta_spike_probability")].copy() if not df.empty else df
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = [cond for cond in ["high_stsp_overlap", "matched_removal"] if cond in set(use["condition"].astype(str))]
    if not order:
        order = _ordered_unique(use["condition"], [])
    labels = {
        "high_stsp_overlap": "High-STSP\noverlap",
        "matched_removal": "Matched\nremoval",
    }
    colors = {
        "high_stsp_overlap": "#B2182B",
        "matched_removal": "#8A8A8A",
    }
    use = _percent_copy(use)
    xs = np.arange(len(order), dtype=float)
    for idx, condition in enumerate(order):
        vals = pd.to_numeric(use.loc[use["condition"].astype(str).eq(condition), "value"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        ax.bar([xs[idx]], [float(vals.mean())], width=0.62, color=colors.get(condition, "#4c78a8"), edgecolor="0.25", linewidth=0.5)
        ax.errorbar([xs[idx]], [float(vals.mean())], yerr=[_t95_half_width(vals)], fmt="none", color="0.15", linewidth=0.7, capsize=2.0)
    ax.axhline(0, color="0.35", linewidth=0.65, linestyle="--")
    ax.set_xticks(xs, [labels.get(cond, cond.replace("_", "\n")) for cond in order])
    ax.set_ylabel(str(spec.get("y_axis", "Layer 1 recruitment loss (%)")))
    ax.set_xlabel("Removed sites")
    ax.paper_fig_plot_form = "high_stsp_overlap_ablation_bar"
    ax.paper_fig_primary_metric = "loss_delta_spike_probability"
    ax.paper_fig_interval_definition = "two-sided 95% Student-t CI across independent networks"
    ax.paper_fig_final_label_claim = False
    ax.paper_fig_high_stsp_alone_sufficient = False
    _tidy(ax)


def render_fig6_region_ping_readout_bias(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    default_metrics = ["old_mass", "middle_mass", "recent_mass", "other_mass", "silent_rate"]
    metrics = [str(metric) for metric in (spec.get("metrics") or default_metrics)]
    if not df.empty and "other_mass" not in set(df.get("metric", pd.Series(dtype=str)).astype(str)):
        ax.paper_fig_missing_other_mass = True
    use = df[df.get("metric", pd.Series(dtype=str)).astype(str).isin(metrics)].copy() if not df.empty else df
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = [cond for cond in list(spec.get("conditions") or ["Peak ping", "Valley ping", "Random ping"]) if cond in set(use["condition"].astype(str))]
    if not order:
        order = _ordered_unique(use["condition"], [])
    use = _percent_copy(use)
    summary = use.groupby(["condition", "metric"], as_index=False)["value"].mean()
    bottom = np.zeros(len(order), dtype=float)
    colors = {
        "old_mass": "#4C78A8",
        "middle_mass": "#59A14F",
        "recent_mass": "#F28E2B",
        "other_mass": "#b07aa1",
        "silent_rate": "#9c9c9c",
    }
    labels = {
        "old_mass": "Old item mass",
        "middle_mass": "Middle item mass",
        "recent_mass": "Recent item mass",
        "other_mass": "Other readout",
        "silent_rate": "Silent rate",
    }
    labels.update({str(key): str(value) for key, value in (spec.get("metric_labels") or {}).items()})
    for metric in metrics:
        vals = [
            float(summary.loc[summary["condition"].astype(str).eq(cond) & summary["metric"].astype(str).eq(metric), "value"].mean())
            if not summary.loc[summary["condition"].astype(str).eq(cond) & summary["metric"].astype(str).eq(metric), "value"].empty
            else 0.0
            for cond in order
        ]
        ax.bar(np.arange(len(order)), vals, bottom=bottom, color=colors[metric], edgecolor="white", linewidth=0.4, label=labels[metric])
        bottom += np.asarray(vals, dtype=float)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([_wrap_label(cond) for cond in order], rotation=0)
    ax.set_ylabel(str(spec.get("y_axis", "Readout composition (%)")))
    ax.set_xlabel("Ping entry")
    ax.set_ylim(0, max(100.0, float(np.nanmax(bottom)) if len(bottom) else 100.0) * 1.08)
    legend = ax.legend(
        frameon=False,
        fontsize=5.0,
        ncol=min(3, max(1, len(metrics))),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
        handlelength=0.9,
        columnspacing=0.7,
        handletextpad=0.35,
    )
    legend.set_in_layout(False)
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = min(3, max(1, len(metrics)))
    ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
    ax.paper_fig_stack_metrics = metrics
    ax.paper_fig_plot_form = "region_gated_ping_readout_bias"
    _tidy(ax)


def render_fig6_global_ping_score_spike_prediction(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    use = df[df.get("metric", pd.Series(dtype=str)).astype(str).eq("spike_probability")].copy() if not df.empty else df
    if not use.empty:
        use = use[use.get("condition", pd.Series(dtype=str)).astype(str).eq("Global ping")].copy()
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    primary_window = float(spec.get("primary_early_window_ms", spec.get("primary_score_early_window_ms", 15)))
    used_fallback_window = False
    if "early_window_ms" in use.columns:
        windows = pd.to_numeric(use["early_window_ms"], errors="coerce")
        primary = use[windows.sub(primary_window).abs().le(1e-6)].copy()
        if not primary.empty:
            use = primary
        else:
            used_fallback_window = True
            ax.paper_fig_window_filter_fallback = True
    use = _percent_copy(use)
    _score_quantile_lines(ax, use, preferred=["Global ping"])
    ax.set_xlabel("STSP score quantile")
    ax.set_ylabel(str(spec.get("y_axis", "Layer 1 spike probability (%)")))
    ymax = pd.to_numeric(use.get("value", pd.Series(dtype=float)), errors="coerce").max()
    ax.set_ylim(0, max(100.0, float(ymax) * 1.06 if pd.notna(ymax) else 100.0))
    ax.paper_fig_plot_form = "global_ping_score_quantile_spike_probability"
    ax.paper_fig_entry_type = "global_ping"
    ax.paper_fig_primary_endpoint = "Layer 1 spike recruitment"
    ax.paper_fig_primary_early_window_ms = primary_window
    ax.paper_fig_used_fallback_early_window = used_fallback_window
    _tidy(ax)


def render_fig6_ping_score_spike_prediction(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    use = df[df.get("metric", pd.Series(dtype=str)).astype(str).eq("spike_probability")].copy() if not df.empty else df
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    primary_window = float(spec.get("primary_early_window_ms", spec.get("primary_score_early_window_ms", 15)))
    used_fallback_window = False
    if "early_window_ms" in use.columns:
        windows = pd.to_numeric(use["early_window_ms"], errors="coerce")
        primary = use[windows.sub(primary_window).abs().le(1e-6)].copy()
        if not primary.empty:
            use = primary
        else:
            used_fallback_window = True
    _score_quantile_lines(ax, use, preferred=list(spec.get("conditions") or ["Peak ping", "Valley ping", "Random ping"]))
    ax.set_xlabel("STSP score quantile")
    ax.set_ylabel("Early L1 spike probability")
    ax.set_ylim(0, 1)
    ax.paper_fig_plot_form = "score_quantile_spike_probability"
    ax.paper_fig_primary_endpoint = "Layer 1 spike recruitment"
    ax.paper_fig_primary_early_window_ms = primary_window
    ax.paper_fig_baseline_removed = True
    ax.paper_fig_used_fallback_early_window = used_fallback_window
    _tidy(ax)


def render_fig6_real_probe_score_spike_deflection(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    use = df[df.get("metric", pd.Series(dtype=str)).astype(str).eq("delta_spike_probability")].copy() if not df.empty else df
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    primary_window = float(spec.get("primary_early_window_ms", spec.get("primary_score_early_window_ms", 10)))
    used_fallback_window = False
    if "early_window_ms" in use.columns:
        windows = pd.to_numeric(use["early_window_ms"], errors="coerce")
        primary = use[windows.sub(primary_window).abs().le(1e-6)].copy()
        if not primary.empty:
            use = primary
        else:
            used_fallback_window = True
            ax.paper_fig_window_filter_fallback = True
    use = _percent_copy(use)
    _score_quantile_lines(ax, use, preferred=["Real probe"])
    ax.axhline(0, color="0.35", linewidth=0.65, linestyle="--")
    ax.set_xlabel("STSP score quantile")
    ax.set_ylabel(str(spec.get("y_axis", "Layer 1 firing change (%)")))
    ax.paper_fig_plot_form = "real_probe_score_quantile_spike_deflection"
    ax.paper_fig_primary_early_window_ms = primary_window
    ax.paper_fig_used_fallback_early_window = used_fallback_window
    _tidy(ax)


def render_fig6_score_basin_sparsification(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    hit = df[df.get("metric", pd.Series(dtype=str)).astype(str).eq("fired_site_score_percentile_mean")].copy() if not df.empty else df
    if hit.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        ax.text(0.5, 0.08, "fired-site percentile metric unavailable", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6, color="0.35")
        ax.paper_fig_plot_form = "fired_site_score_percentile"
        ax.paper_fig_primary_metric = "fired_site_score_percentile_mean"
        return
    primary_radius = float(spec.get("primary_basin_radius", spec.get("basin_radius", 2)))
    used_fallback_radius = False
    if "basin_radius" in hit.columns:
        radii = pd.to_numeric(hit["basin_radius"], errors="coerce")
        primary = hit[radii.sub(primary_radius).abs().le(1e-6)].copy()
        if not primary.empty:
            hit = primary
        else:
            used_fallback_radius = True
    order = _ordered_unique(hit["condition"], ["Ping", "Real probe", "Peak ping", "Valley ping", "Random ping"])
    summary = hit.groupby("condition", as_index=False).agg(value=("value", "mean"), sem=("fired_site_score_percentile_sem", "mean"))
    vals = [float(summary.loc[summary["condition"].astype(str).eq(cond), "value"].mean()) for cond in order]
    sem_vals = [float(summary.loc[summary["condition"].astype(str).eq(cond), "sem"].mean()) for cond in order] if "sem" in summary.columns else None
    percentile_scale = 100.0 if vals and np.nanmax(vals) > 1.0 else 1.0
    vals = [val / percentile_scale for val in vals]
    yerr = None
    if sem_vals and np.isfinite(sem_vals).any():
        yerr = [0.0 if not np.isfinite(val) else val / percentile_scale for val in sem_vals]
    ax.bar(np.arange(len(order)), vals, yerr=yerr, color="#4c78a8", edgecolor="0.25", linewidth=0.45, capsize=2)
    ax.axhline(0.5, color="0.45", linewidth=0.65, linestyle="--")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([_wrap_label(cond) for cond in order], rotation=0)
    ax.set_ylabel("Fired-site score percentile")
    ax.set_xlabel("Entry")
    ax.set_ylim(0, 1)
    ax.paper_fig_plot_form = "fired_site_score_percentile"
    ax.paper_fig_score_interpretation = "spike enrichment in high-score percentile, not deterministic one-to-one firing"
    ax.paper_fig_primary_metric = "fired_site_score_percentile_mean"
    ax.paper_fig_reference = "0.5 random percentile expectation"
    ax.paper_fig_primary_basin_radius = primary_radius
    ax.paper_fig_used_fallback_basin_radius = used_fallback_radius
    _tidy(ax)


def render_fig6_overlap_gated_stsp_recruitment(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    df = _clean(panel_data)
    use = df[df.get("metric", pd.Series(dtype=str)).astype(str).eq("delta_spike_probability")].copy() if not df.empty else df
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    primary_window = float(spec.get("primary_early_window_ms", spec.get("primary_score_early_window_ms", 10)))
    used_fallback_window = False
    if "early_window_ms" in use.columns:
        windows = pd.to_numeric(use["early_window_ms"], errors="coerce")
        primary = use[windows.sub(primary_window).abs().le(1e-6)].copy()
        if not primary.empty:
            use = primary
        else:
            used_fallback_window = True
            ax.paper_fig_window_filter_fallback = True
    for col in ("stsp_group", "hue_group", "overlap_group", "x_group"):
        if col in use.columns:
            use[col] = use[col].astype(str).str.lower().str.replace("-", "_").str.replace(" ", "_")
    if "stsp_group" not in use.columns and "hue_group" in use.columns:
        use["stsp_group"] = use["hue_group"]
    if "overlap_group" not in use.columns and "x_group" in use.columns:
        use["overlap_group"] = use["x_group"]
    x_order = ["no_overlap", "overlap"]
    hue_order = ["low", "high"]
    use = _percent_copy(use)
    summary = use.groupby(["overlap_group", "stsp_group"], as_index=False)["value"].mean()
    x = np.arange(len(x_order), dtype=float)
    width = 0.32
    colors = {"low": "#6C7A89", "high": "#B2182B"}
    for idx, hue in enumerate(hue_order):
        vals = []
        for group in x_order:
            part = summary[summary["overlap_group"].astype(str).eq(group) & summary["stsp_group"].astype(str).eq(hue)]
            vals.append(float(part["value"].mean()) if not part.empty else np.nan)
        positions = x + (idx - 0.5) * width
        ax.bar(positions, vals, width=width, color=colors[hue], edgecolor="0.25", linewidth=0.45, label=f"{hue.title()} STSP")
    ax.axhline(0, color="0.35", linewidth=0.65, linestyle="--")
    ax.set_xticks(x, ["No overlap", "Overlap"])
    ax.set_ylabel(str(spec.get("y_axis", "Layer 1 firing change (%)")))
    ax.set_xlabel("Probe overlap")
    legend = ax.legend(
        frameon=False,
        fontsize=5.2,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
        handlelength=0.9,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    legend.set_in_layout(False)
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_interaction_annotation = False
    interaction = df[df.get("metric", pd.Series(dtype=str)).astype(str).eq("interaction_delta")].copy() if not df.empty else pd.DataFrame()
    show_interaction_annotation = bool(spec.get("show_interaction_annotation", False))
    if show_interaction_annotation and not interaction.empty and "early_window_ms" in interaction.columns:
        windows = pd.to_numeric(interaction["early_window_ms"], errors="coerce")
        primary_interaction = interaction[windows.sub(primary_window).abs().le(1e-6)].copy()
        if not primary_interaction.empty:
            interaction = primary_interaction
    if show_interaction_annotation and not interaction.empty:
        val = pd.to_numeric(interaction.get("value"), errors="coerce").dropna()
        if not val.empty:
            mean_val = float(val.mean())
            label = "interaction > 0" if mean_val > 0 else f"interaction = {mean_val:.3f}"
            ax.text(0.98, 0.96, label, transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color="0.25")
            ax.paper_fig_interaction_annotation = True
    ax.paper_fig_plot_form = "overlap_gated_stsp_recruitment_2x2"
    ax.paper_fig_primary_metric = "delta_spike_probability"
    ax.paper_fig_interaction_metric = "interaction_delta"
    ax.paper_fig_claim = "probe_overlap_gates_high_stsp_expression"
    ax.paper_fig_primary_early_window_ms = primary_window
    ax.paper_fig_used_fallback_early_window = used_fallback_window
    _tidy(ax)


def render_fig6_blank_mechanism_placeholder(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data, stats, spec, style
    ax.axis("off")
    ax.paper_fig_plot_form = "fig6_blank_mechanism_placeholder"
    ax.paper_fig_optional_placeholder = True
    ax.paper_fig_pure_mechanism_schematic = True
    ax.paper_fig_has_summary_inset = False
    ax.paper_fig_final_label_claim = False
    ax.paper_fig_high_stsp_alone_sufficient = False
    ax.paper_fig_primary_endpoint = "Layer 1 spike recruitment"


def render_fig6_stsp_overlap_gated_recruitment_synthesis(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data, stats, spec, style
    ax.axis("off")
    labels = ["STSP\ngain", "Probe\nentry", "Overlap\ngate", "L1\nbias"]
    xs = np.linspace(0.04, 0.79, len(labels))
    box_w = 0.15
    for idx, (x, label) in enumerate(zip(xs, labels)):
        ax.add_patch(Rectangle((x, 0.55), box_w, 0.25, transform=ax.transAxes, facecolor="0.96", edgecolor="0.25", linewidth=0.65))
        ax.text(x + box_w / 2, 0.675, label, transform=ax.transAxes, ha="center", va="center", fontsize=5.4)
        if idx < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + box_w, 0.675), (xs[idx + 1], 0.675), transform=ax.transAxes, arrowstyle="->", mutation_scale=7, linewidth=0.75, color="0.25"))
    ax.text(0.5, 0.34, "Entry gates STSP expression", transform=ax.transAxes, ha="center", va="center", fontsize=6.2, fontweight="bold")
    ax.text(0.5, 0.18, "Endpoint: early L1 recruitment", transform=ax.transAxes, ha="center", va="center", fontsize=5.5, color="0.25")
    ax.paper_fig_plot_form = "overlap_gated_stsp_recruitment_synthesis"
    ax.paper_fig_pure_mechanism_schematic = True
    ax.paper_fig_has_summary_inset = False
    ax.paper_fig_final_label_claim = False
    ax.paper_fig_high_stsp_alone_sufficient = False
    ax.paper_fig_primary_endpoint = "Layer 1 spike recruitment"


def render_fig6_stsp_field_recruitment_synthesis(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    ax.axis("off")
    labels = [
        "Multi-item\nSTSP field",
        "Entry overlap\nping / probe",
        "High-score\nL1 regions",
        "Local competition\nsparse expression",
        "Possible\nreadout bias",
    ]
    xs = np.linspace(0.03, 0.78, len(labels))
    box_w = 0.16
    for idx, (x, label) in enumerate(zip(xs, labels)):
        ax.add_patch(Rectangle((x, 0.52), box_w, 0.26, transform=ax.transAxes, facecolor="0.96", edgecolor="0.25", linewidth=0.65))
        ax.text(x + box_w / 2, 0.65, label, transform=ax.transAxes, ha="center", va="center", fontsize=5.7)
        if idx < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + box_w, 0.65), (xs[idx + 1], 0.65), transform=ax.transAxes, arrowstyle="->", mutation_scale=7, linewidth=0.75, color="0.25"))
    ax.text(0.05, 0.27, "STSP field + entry -> L1 recruitment bias", transform=ax.transAxes, ha="left", va="center", fontsize=7.2, fontweight="bold")
    ax.text(0.05, 0.12, "local competition -> sparse expression; endpoint is early Layer 1 recruitment", transform=ax.transAxes, ha="left", va="center", fontsize=5.8, color="0.25")
    ax.paper_fig_plot_form = "stsp_field_recruitment_synthesis"
    ax.paper_fig_final_label_claim = False


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
        raise RuntimeError("Fig.6D/E required route-peak perturbation data missing or invalid.")
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
        raise RuntimeError("Fig.6D/E required route-peak perturbation data missing or invalid.")
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


def _score_quantile_lines(ax, df: pd.DataFrame, preferred: Sequence[str]) -> None:
    use = df.copy()
    use["x_value"] = pd.to_numeric(use.get("x_value"), errors="coerce")
    use["value"] = pd.to_numeric(use.get("value"), errors="coerce")
    use = use.dropna(subset=["x_value", "value"])
    if use.empty:
        return
    colors = ["#4c78a8", "#f28e2b", "#59a14f", "#b07aa1", "#e15759"]
    order = _ordered_unique(use["condition"], preferred)
    for idx, condition in enumerate(order):
        part = use[use["condition"].astype(str).eq(condition)].copy()
        if part.empty:
            continue
        summary = part.groupby("x_value", as_index=False)["value"].mean().sort_values("x_value")
        color = colors[idx % len(colors)]
        ax.plot(summary["x_value"], summary["value"], marker="o", markersize=2.8, linewidth=1.0, color=color, label=condition)
    if len(order) > 1:
        ax.legend(frameon=False, fontsize=5.4, loc="best")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(use))
    ax.paper_fig_x_metric = "entry_gated_stsp_gain_score_quantile"


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
    for col in (
        "raw_overlap",
        "peak_weighted_overlap",
        "peak_overlap_fraction",
        "nonpeak_overlap_fraction",
        "x_value",
        "y_value",
        "relative_position_from_end",
        "position_from_end",
        "sem",
        "early_window_ms",
        "basin_radius",
        "fired_site_score_percentile_sem",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["value"])


def _percent_copy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce") * 100.0
    return out


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


def _t95_half_width(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    return float(scipy_stats.t.ppf(0.975, clean.size - 1) * _sem(clean)) if clean.size > 1 else 0.0


def _wrap_label(label: str) -> str:
    return (
        str(label)
        .replace("Single old", "Single\nold")
        .replace("Single recent", "Single\nrecent")
        .replace("Peak ping", "Peak\nping")
        .replace("Valley ping", "Valley\nping")
        .replace("Random ping", "Random\nping")
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
