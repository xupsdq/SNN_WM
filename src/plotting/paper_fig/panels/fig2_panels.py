from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.plotting.common.colors import get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL
from src.plotting.experiments.fig4_chunk_interaction_assay_plot import (
    draw_weak_probe_completion_curve_on_ax,
)
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder


def render_fig2a_episode_schematic(ax, panel_data, stats, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Reserve Fig.2A as a blank panel slot."""
    _ = panel_data, stats, spec, style
    ax.set_axis_off()
    ax.paper_fig_plot_form = "blank_reserved_slot"


def render_one_sample_dot_summary(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render one-sample summaries with the fused-state experiment violin/histogram style."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    metric = str(df["metric"].iloc[0]) if "metric" in df.columns else ""
    if "whole_pair_representation_index" in metric:
        plot_df = pd.DataFrame({"WPRI": df["value"]})
        _draw_wpri_density(ax, plot_df["WPRI"].to_numpy(dtype=float), spec=spec)
        return
    value_col = "fusion_imbalance" if "imbalance" in metric else "fusion_dual_score"
    _draw_fusion_distribution(ax, df["value"].to_numpy(dtype=float), ylabel=str(spec.get("y_axis", value_col)))
    _reference_lines(ax, spec)
    if bool(spec.get("hide_x_tick_labels", False)):
        ax.set_xticks([])
        ax.set_xlabel("")
    _compact_axis(ax, y_pad=1.8, y_labelpad=1.8)
    ax.paper_fig_plot_form = "one_sample_distribution"
    ax.paper_fig_y_range_mode = "data_tight"
    ax.paper_fig_raw_point_count = 240
    ax.paper_fig_raw_point_alpha = 0.08
    ax.paper_fig_raw_points = True


def render_wpri_density(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render row-level WPRI values as a density plot."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    _draw_wpri_density(ax, df["value"].to_numpy(dtype=float), spec=spec)
    _reference_vlines(ax, spec)


def render_paired_condition_plot(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render paired true-vs-shuffled comparison with the experiment boxplot style."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    plot_df = pd.DataFrame(
        {
            "true_pair_score": df[df["condition"].eq("True pair")]["value"].to_numpy(dtype=float),
            "shuffled_pair_score": df[df["condition"].eq("Shuffled pair")]["value"].to_numpy(dtype=float),
        }
    )
    if plot_df["true_pair_score"].dropna().empty or plot_df["shuffled_pair_score"].dropna().empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    _draw_true_shuffled_boxplot(ax, plot_df, ylabel=str(spec.get("y_axis", "Pair-specificity score")))
    ax.paper_fig_plot_form = "box_plot"


def render_neutral_ping_functional_access(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.2E with the fig4 interaction neutral-ping experiment plot style."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    plot_df = df.drop_duplicates(subset=[c for c in ("seed_id", "state_condition") if c in df.columns]).copy()
    if "seed" not in plot_df.columns:
        plot_df["seed"] = plot_df.get("seed_id", "")
    _draw_neutral_ping_stacked(ax, plot_df)
    ax.paper_fig_plot_form = "stacked_readout"


def render_partial_cue_completion(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.2F with the fig4 interaction weak-probe completion curve style."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    curve = df[df.get("curve_or_summary", "").eq("curve")] if "curve_or_summary" in df.columns else df
    if curve.empty or not {"state_condition", "keep_prob", "P_A"}.issubset(curve.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if "seed" not in curve.columns:
        curve = curve.copy()
        curve["seed"] = curve.get("seed_id", "")
    draw_weak_probe_completion_curve_on_ax(ax, curve, title=None)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    _compact_axis(ax)
    ax.paper_fig_plot_form = "partial_cue_completion_curve"


def _draw_true_shuffled_boxplot(ax, df: pd.DataFrame, *, ylabel: str) -> None:
    columns = [("true_pair_score", "True pair"), ("shuffled_pair_score", "Shuffled pair")]
    values = []
    labels = []
    for column, label in columns:
        arr = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
        if arr.size:
            values.append(arr)
            labels.append(label)
    if len(values) != 2:
        raise ValueError("Fig.2C requires finite true_pair_score and shuffled_pair_score values.")
    box = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.55,
        tick_labels=labels,
        showfliers=False,
        medianprops={"color": COLOR_NEUTRAL, "linewidth": 0.9},
        boxprops={"edgecolor": COLOR_NEUTRAL, "linewidth": 0.75},
        whiskerprops={"color": COLOR_NEUTRAL, "linewidth": 0.75},
        capprops={"color": COLOR_NEUTRAL, "linewidth": 0.75},
    )
    colors = [get_plot_color("true_pair"), get_plot_color("shuffled_pair")]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_ylabel(ylabel)
    _compact_axis(ax, y_pad=1.8, y_labelpad=1.8)
    ax.paper_fig_raw_points = False
    ax.paper_fig_showfliers = False


def _draw_fusion_distribution(ax, values: np.ndarray, *, ylabel: str) -> None:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("Fig.2B fusion values are not finite.")
    color = get_plot_color("fused_state", context="fig4_fusion")
    violins = ax.violinplot([clean], positions=[1.0], widths=0.72, showextrema=False)
    for body in violins["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(COLOR_NEUTRAL)
        body.set_alpha(0.86)
        body.set_linewidth(0.7)
    rng = np.random.default_rng(17)
    n_show = min(240, clean.size)
    shown = rng.choice(clean, size=n_show, replace=False) if clean.size > n_show else clean
    jitter = rng.normal(0.0, 0.045, size=shown.size)
    ax.scatter(np.full(shown.size, 1.0) + jitter, shown, s=1.2, color=COLOR_NEUTRAL, alpha=0.08, linewidths=0, rasterized=True)
    ax.hlines(float(np.nanmedian(clean)), 0.72, 1.28, color=COLOR_NEUTRAL, linewidth=0.9)
    ymin = float(np.nanmin(clean))
    ymax = float(np.nanmax(clean))
    pad = max(0.008, 0.05 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(0.55, 1.45)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)


def _draw_one_condition_bar(ax, values: np.ndarray, *, ylabel: str, xtick_label: str) -> None:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("Fig.2D WPRI values are not finite.")
    mean = float(np.nanmean(clean))
    sem = _sem(clean)
    ax.bar([0], [mean], yerr=[sem], width=0.55, color=get_plot_color("fused_state", context="fig4_fusion"), edgecolor=COLOR_NEUTRAL, linewidth=0.75, alpha=0.82, capsize=2.0)
    ax.set_xticks([0], [xtick_label])
    ax.set_ylabel(ylabel)
    ymin = min(0.0, float(np.nanmin(clean)), mean - sem)
    ymax = max(0.0, float(np.nanmax(clean)), mean + sem)
    pad = max(0.02, 0.12 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)


def _draw_wpri_density(ax, values: np.ndarray, *, spec: Mapping[str, Any]) -> None:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("Fig.2D WPRI values are not finite.")
    bins = min(36, max(18, int(np.sqrt(clean.size))))
    color = get_plot_color("fused_state", context="fig4_fusion")
    ax.hist(clean, bins=bins, density=True, color=color, edgecolor=COLOR_NEUTRAL, linewidth=0.45, alpha=0.58)
    counts, edges = np.histogram(clean, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.plot(centers, counts, color=COLOR_NEUTRAL, linewidth=0.8)
    xmin = min(0.0, float(np.nanmin(clean)))
    xmax = float(np.nanmax(clean))
    pad = max(0.01, 0.06 * (xmax - xmin if xmax > xmin else 1.0))
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_xticks([0.0, 0.1, 0.2, 0.25])
    ax.set_xlabel(str(spec.get("x_axis", "WPRI")))
    ax.set_ylabel(str(spec.get("y_axis", "Density")))
    _compact_axis(ax, y_pad=1.8, y_labelpad=1.8)
    ax.paper_fig_plot_form = "density_plot"
    ax.paper_fig_x_metric = "WPRI"
    ax.paper_fig_y_metric = "Density"


def _draw_neutral_ping_stacked(ax, df: pd.DataFrame) -> None:
    states = ["baseline", "S_B", "S_AB"]
    labels = ["No\nmemory", "Item 2\nonly", "Both\nmemories"]
    value_cols = [("P_A", "A"), ("P_B", "B"), ("P_other", "other"), ("P_silent", "silent")]
    summary = {col: _seed_first_mean(df, ["state_condition"], col) for col, _ in value_cols if col in df.columns}
    x = np.arange(len(states), dtype=float)
    bottom = np.zeros(len(states), dtype=float)
    colors = [get_plot_color("true_pair"), get_plot_color("anchor"), get_plot_color("shuffled_pair"), get_plot_color("background_shade")]
    for idx, (col, label) in enumerate(value_cols):
        if col not in summary:
            continue
        vals = []
        for state in states:
            row = summary[col][summary[col]["state_condition"].astype(str).eq(state)]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x, vals, bottom=bottom, label=label, color=colors[idx], edgecolor=COLOR_NEUTRAL, linewidth=0.5, alpha=0.86)
        bottom += np.asarray(vals, dtype=float)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Readout proportion")
    ax.set_ylim(0.0, max(1.18, float(np.nanmax(bottom)) + 0.12))
    legend = ax.legend(
        frameon=False,
        fontsize=4.8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        borderaxespad=0.0,
        handlelength=0.72,
        handletextpad=0.25,
        columnspacing=0.42,
    )
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")
        label.set_fontstyle("normal")
    ax.paper_fig_legend_overlaps_data = False
    ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
    ax.paper_fig_legend_ncols = 4
    ax.paper_fig_third_condition_label = "Both memories"
    _compact_axis(ax, y_pad=1.8, y_labelpad=1.8)


def _auc_only(ax, auc: pd.DataFrame, spec: Mapping[str, Any]) -> None:
    conditions = list(spec.get("conditions") or sorted(auc["condition"].unique()))
    vals = []
    sems = []
    for condition in conditions:
        arr = auc[auc["condition"].eq(condition)]["value"].to_numpy(dtype=float)
        vals.append(float(np.nanmean(arr)) if arr.size else np.nan)
        sems.append(_sem(arr) if arr.size else 0.0)
    ax.bar(range(len(conditions)), vals, alpha=0.75)
    ax.errorbar(range(len(conditions)), vals, yerr=sems, fmt="none", color="black", capsize=2)
    ax.set_xticks(range(len(conditions)), [_display(c) for c in conditions], rotation=30, ha="right")
    ax.set_ylabel(str(spec.get("y_axis", "")))
    _tidy(ax)


def _stacked_fallback(ax, df: pd.DataFrame, spec: Mapping[str, Any]) -> None:
    conditions = list(spec.get("conditions") or sorted(df["condition"].unique()))
    categories = [c for c in ["Pair-member readout", "Item 1 accessibility"] if c in set(df.get("readout_category", []))]
    bottoms = np.zeros(len(conditions))
    for category in categories:
        vals = [df[(df["condition"].eq(condition)) & (df["readout_category"].eq(category))]["value"].mean() for condition in conditions]
        ax.bar(range(len(conditions)), vals, bottom=bottoms, label=category)
        bottoms += np.asarray(vals, dtype=float)
    ax.set_xticks(range(len(conditions)), [_display(c) for c in conditions], rotation=20, ha="right")
    ax.set_ylabel(str(spec.get("y_axis", "")))
    ax.legend(frameon=False, fontsize=7)
    _tidy(ax)


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "dropout_level" in df.columns:
        df["dropout_level"] = pd.to_numeric(df["dropout_level"], errors="coerce")
    return df.dropna(subset=["value"])


def _reference_lines(ax, spec: Mapping[str, Any]) -> None:
    for line in spec.get("reference_lines") or []:
        ax.axhline(float(line["value"]), linestyle="--", color="0.4", linewidth=0.8)
        if line.get("label"):
            ax.text(0.98, float(line["value"]), str(line["label"]), transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7)


def _reference_vlines(ax, spec: Mapping[str, Any]) -> None:
    for line in spec.get("reference_lines") or []:
        ax.axvline(float(line["value"]), linestyle="--", color="0.4", linewidth=0.8)


def _sem(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size <= 1:
        return 0.0
    return float(np.nanstd(clean, ddof=1) / np.sqrt(clean.size))


def _seed_first_mean(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame(columns=[*group_cols, "mean"])
    seed_cols = ["seed", *group_cols] if "seed" in df.columns else group_cols
    seed_level = df.groupby(seed_cols, sort=True)[value_col].mean().reset_index()
    return seed_level.groupby(group_cols, sort=True)[value_col].mean().reset_index(name="mean")


def _id_col(df: pd.DataFrame) -> str | None:
    for col in ("seed_id", "network_id"):
        if col in df.columns and df[col].replace("", pd.NA).dropna().nunique() > 0:
            return col
    return None


def _display(label: str) -> str:
    return str(label).replace("->", "→")


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _compact_axis(ax, *, y_pad: float = 1.5, y_labelpad: float = 1.8) -> None:
    _tidy(ax)
    ax.tick_params(axis="x", labelsize=6.0, pad=0.6, length=2.2, width=0.6, color=COLOR_NEUTRAL)
    ax.tick_params(axis="y", labelsize=6.0, pad=y_pad, length=2.2, width=0.6, color=COLOR_NEUTRAL)
    ax.xaxis.label.set_size(6.4)
    ax.yaxis.label.set_size(6.4)
    ax.yaxis.labelpad = y_labelpad
    ax.xaxis.labelpad = 1.0
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontstyle("normal")
