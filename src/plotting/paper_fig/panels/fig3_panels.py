from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.plotting.common.colors import get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder


def render_one_sample_dot_summary(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.3A with the fused-state experiment violin style."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    _draw_single_distribution(ax, df["value"].to_numpy(dtype=float), ylabel=str(spec.get("y_axis", "Fusion imbalance")))
    _reference_lines(ax, spec)
    ax.paper_fig_plot_form = "one_sample_distribution"


def render_latent_bias_readout_preference(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.3B with the state-taxonomy DI-vs-sample-first plot style."""
    _ = style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    plot_df = df.copy()
    if "DI_mean" not in plot_df.columns and "x_value" in plot_df.columns:
        plot_df["DI_mean"] = plot_df["x_value"]
    if "sample_first_prob" not in plot_df.columns:
        plot_df["sample_first_prob"] = plot_df.get("y_value", plot_df["value"])
    if not {"layer", "DI_mean", "sample_first_prob"}.issubset(plot_df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    _draw_panel_b_aggregated_lines(ax, plot_df, spec)


def render_latest_vs_earlier_mass(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.3C latest-item versus earlier-items mass."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    _paired_conditions(ax, df, list(spec.get("conditions") or ["Latest item", "Earlier items"]))
    ax.set_xlabel(str(spec.get("x_axis", "")))
    ax.set_ylabel(str(spec.get("y_axis", "")))
    _tidy(ax)
    _compact_ticks(ax)


def render_seen_item_ping_access(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.3D neutral-ping seen-item hit-rate summary."""
    _ = stats, style
    df = _clean(panel_data)
    if df.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    values = df["value"].to_numpy(dtype=float)
    ax.bar([0], [float(np.nanmean(values))], yerr=[_sem(values)], width=0.45, color=get_plot_color("true_pair"), edgecolor=COLOR_NEUTRAL, linewidth=0.7, alpha=0.78, capsize=2.5)
    ax.set_xticks([0], [str(spec.get("x_axis", "Neutral ping"))])
    ax.set_ylabel(str(spec.get("y_axis", "")))
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(bottom=0)
    _tidy(ax)
    _compact_ticks(ax)


def render_state_com_shift(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.3E state center-of-mass trajectory."""
    _render_com_shift(ax, panel_data, stats, spec, style, value_col="state_center_of_mass")


def render_ping_com_shift(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    """Render Fig.3F ping center-of-mass trajectory."""
    _render_com_shift(ax, panel_data, stats, spec, style, value_col="ping_center_of_mass")


def _render_com_shift(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None, *, value_col: str) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty or "sequence_stage" not in df.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if value_col not in df.columns:
        df[value_col] = df["value"]
    _line_summary(ax, df, x_col="sequence_stage", y_col=value_col, ylabel=str(spec.get("y_axis", ax.get_ylabel())))


def _paired_conditions(ax, df: pd.DataFrame, conditions: list[str]) -> None:
    x_map = {condition: idx for idx, condition in enumerate(conditions)}
    for condition in conditions:
        vals = df[df["condition"].eq(condition)]["value"].to_numpy(dtype=float)
        if vals.size:
            ax.bar([x_map[condition]], [float(np.nanmean(vals))], yerr=[_sem(vals)], width=0.55, color=get_plot_color("true_pair" if condition == conditions[0] else "shuffled_pair"), edgecolor=COLOR_NEUTRAL, linewidth=0.7, alpha=0.75, capsize=2.5)
    display = [str(condition).replace("Latest item", "Latest\nitem").replace("Earlier items", "Earlier\nitems") for condition in conditions]
    ax.set_xticks(range(len(conditions)), display, rotation=0, ha="center")


def _draw_panel_b_aggregated_lines(ax, df: pd.DataFrame, spec: Mapping[str, Any]) -> None:
    use = df.dropna(subset=["DI_mean", "sample_first_prob"]).copy()
    if use.empty:
        raise ValueError("Fig.3B has no finite DI/readout values.")
    use["x_round"] = pd.to_numeric(use["DI_mean"], errors="coerce").round(6)
    use["y"] = pd.to_numeric(use["sample_first_prob"], errors="coerce")
    grouped = use.groupby(["layer", "x_round"], dropna=False)["y"].agg(["mean", "count", "std"]).reset_index()
    grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    layers = [str(v) for v in grouped["layer"].dropna().unique()]
    colors = [get_plot_color("true_pair"), get_plot_color("anchor"), get_plot_color("shuffled_pair"), COLOR_NEUTRAL]
    plotted_counts: dict[str, int] = {}
    repeated = bool((grouped["count"] > 1).any() or len(grouped) < len(use))
    for idx, layer in enumerate(layers):
        part = grouped[grouped["layer"].astype(str).eq(layer)].sort_values("x_round")
        if part.empty:
            continue
        x = part["x_round"].to_numpy(dtype=float)
        y = part["mean"].to_numpy(dtype=float)
        sem = part["sem"].fillna(0.0).to_numpy(dtype=float)
        color = colors[idx % len(colors)]
        if np.any(sem > 0):
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.13, linewidth=0)
        else:
            ax.axhspan(float(np.nanmin(y)), float(np.nanmax(y)), color=color, alpha=0.035, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.8, label=_display_layer(layer), zorder=3)
        ax.scatter(x, y, s=8, color=color, edgecolor="white", linewidth=0.25, alpha=0.55, zorder=4)
        plotted_counts[_display_layer(layer)] = int(len(part))
    ax.set_xlabel(str(spec.get("x_axis", "Latent state bias")))
    ax.set_ylabel(str(spec.get("y_axis", "Readout preference")))
    ax.legend(frameon=False, fontsize=5.5, loc="best", handlelength=1.1)
    _tidy(ax)
    _compact_ticks(ax)
    ax.paper_fig_plot_form = "aggregated_line"
    ax.paper_fig_rows_before_renderer_aggregation = int(len(use))
    ax.paper_fig_plotted_x_positions_by_layer = plotted_counts
    ax.paper_fig_repeated_x_positions_averaged = repeated
    ax.paper_fig_line_emphasis = "line_over_points"
    ax.paper_fig_raw_points = False
    ax.paper_fig_has_shaded_band = True


def _draw_single_distribution(ax, values: np.ndarray, *, ylabel: str) -> None:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("No finite distribution values.")
    color = get_plot_color("fused_state", context="fig4_fusion")
    violin = ax.violinplot([clean], positions=[0], widths=0.65, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(COLOR_NEUTRAL)
        body.set_alpha(0.82)
        body.set_linewidth(0.7)
    ax.hlines(float(np.nanmedian(clean)), -0.28, 0.28, color=COLOR_NEUTRAL, linewidth=1.0)
    ax.set_xticks([0], ["Layer 3"])
    ax.set_ylabel(ylabel)
    ymin = min(0.0, float(np.nanmin(clean)))
    ymax = float(np.nanmax(clean))
    pad = max(0.02, 0.08 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)
    _tidy(ax)
    _compact_ticks(ax)


def _line_summary(ax, df: pd.DataFrame, *, x_col: str, y_col: str, ylabel: str) -> None:
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    summary = plot_df.groupby(x_col, sort=True)[y_col].agg(["mean", "count", "std"]).reset_index()
    if summary.empty:
        raise ValueError("No finite COM summary values.")
    summary["sem"] = summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
    x = summary[x_col].to_numpy(dtype=float)
    y = summary["mean"].to_numpy(dtype=float)
    sem = summary["sem"].to_numpy(dtype=float)
    color = get_plot_color("true_pair")
    ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.14, linewidth=0)
    ax.plot(x, y, color=color, linewidth=1.7)
    ax.scatter(x, y, s=10, color=color, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xlabel("Sequence stage")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(set(x.astype(int))))
    ymin = float(np.nanmin(y - sem))
    ymax = float(np.nanmax(y + sem))
    pad = max(0.1, 0.05 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)
    tick_min = int(np.floor(ymin))
    tick_max = int(np.ceil(ymax))
    ax.set_yticks([v for v in range(tick_min, tick_max + 1) if v >= 1])
    _tidy(ax)
    _compact_ticks(ax)
    ax.paper_fig_renderer_summarizes_row_level = True


def _display_layer(layer: str) -> str:
    text = str(layer).replace("layer", "Layer ")
    return text if "Layer" in text else str(layer)


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for col in ("x_value", "y_value", "sequence_stage", "state_center_of_mass", "ping_center_of_mass"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["value"])


def _reference_lines(ax, spec: Mapping[str, Any]) -> None:
    for line in spec.get("reference_lines") or []:
        ax.axhline(float(line["value"]), linestyle="--", color="0.4", linewidth=0.8)
        if line.get("label"):
            ax.text(0.98, float(line["value"]), str(line["label"]), transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7)


def _sem(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size <= 1:
        return 0.0
    return float(np.nanstd(clean, ddof=1) / np.sqrt(clean.size))


def _id_col(df: pd.DataFrame) -> str | None:
    for col in ("seed_id", "network_id"):
        if col in df.columns and df[col].replace("", pd.NA).dropna().nunique() > 0:
            return col
    return None


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _compact_ticks(ax) -> None:
    ax.tick_params(axis="both", labelsize=5.8, pad=1.0, length=2.0, width=0.6, color=COLOR_NEUTRAL)
    ax.xaxis.label.set_size(6.4)
    ax.yaxis.label.set_size(6.4)
    ax.xaxis.labelpad = 0.6
    ax.yaxis.labelpad = 0.6
