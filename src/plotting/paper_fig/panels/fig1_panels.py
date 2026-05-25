from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.plotting.common.colors import get_plot_color
from src.plotting.paper_fig.utils import paper_fig_root


STYLE = {
    "axis_labelsize": 6.3,
    "tick_labelsize": 5.4,
    "legend_fontsize": 5.3,
    "annotation_fontsize": 5.2,
    "line_width": 0.75,
    "marker_size": 12.0,
    "bar_width": 0.62,
    "capsize": 2.0,
}

LAYER_COLORS = {"layer1": "#4C78A8", "layer2": "#F58518", "layer3": "#54A24B"}
CONDITION_COLORS = {
    "dynamic_intact": get_plot_color("dynamic"),
    "ux_trial_shuffle": get_plot_color("trial_shuffled_ux"),
    "static_frozen": get_plot_color("static_frozen"),
}


def render_fig1_architecture_schematic(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data, stats, style
    ax.set_axis_off()
    asset = spec.get("source") or (spec.get("source_mapping") or {}).get("manual_asset")
    asset_path = paper_fig_root() / str(asset) if asset else None
    if asset_path and asset_path.exists():
        ax.text(0.5, 0.5, f"Manual schematic asset\n{asset}", ha="center", va="center", fontsize=7.0, transform=ax.transAxes)
        ax.paper_fig_plot_form = "manual_schematic_asset_slot"
        return

    ax.paper_fig_plot_form = "blank_manual_slot"


def render_fig1_baseline_recall(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, "Baseline data unavailable")
        return
    ax.paper_fig_plot_form = "baseline_recall_network_points"
    values = df["value"].to_numpy(dtype=float)
    x = np.arange(len(values), dtype=float)
    mean = float(np.nanmean(values))
    sem = float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    if values.size > 1:
        ax.axhspan(mean - sem, mean + sem, color="#9ECAE1", alpha=0.32, linewidth=0, zorder=0)
        ax.paper_fig_has_shaded_band = True
        ax.paper_fig_shaded_band = [mean - sem, mean + sem]
    ax.plot(x, values, color="#4C78A8", linewidth=0.85, marker="o", markersize=2.7, zorder=3)
    ax.axhline(mean, color="#2F6EA3", linewidth=0.75, zorder=2)
    _reference_lines(ax, spec)
    tick_idx = _network_tick_indices(len(values))
    tick_labels = [str(v) for v in df["seed_id"].astype(str).tolist()]
    ax.set_xticks(x[tick_idx], [tick_labels[i] for i in tick_idx])
    ax.set_xlim(-0.5, max(0.5, len(values) - 0.5))
    ax.set_xlabel(str(spec.get("x_axis", "Network")))
    ax.set_ylabel(str(spec.get("y_axis", "Recall (%)")))
    ax.set_ylim(0, max(100, float(np.nanmax(values) + 8)))
    ax.paper_fig_line_emphasis = "one_network_one_point_connected"
    ax.paper_fig_has_mean_marker = False
    _tidy(ax, st)


def render_fig1_delay_decode(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "delay_ms" not in df.columns:
        _placeholder(ax, spec, "Delay decoding data unavailable")
        return
    ax.paper_fig_plot_form = "delay_decode_curve"
    for layer in [v for v in ("layer1", "layer2", "layer3") if v in set(df["layer"].astype(str))]:
        part = df[df["layer"].astype(str) == layer].copy()
        grouped = part.groupby("delay_ms", as_index=False)["value"].agg(["mean", "sem"]).reset_index()
        x = grouped["delay_ms"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        sem = grouped["sem"].fillna(0).to_numpy(dtype=float)
        label = (spec.get("display_labels") or {}).get(layer, layer)
        color = LAYER_COLORS.get(layer, "0.2")
        ax.plot(x, y, marker="o", markersize=3.0, linewidth=1.0, color=color, label=label)
        if len(part["seed_id"].dropna().unique()) > 1:
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.18, linewidth=0)
    _reference_lines(ax, spec)
    ax.set_xlabel(str(spec.get("x_axis", "Delay (ms)")))
    ax.set_ylabel(str(spec.get("y_axis", "Decoding accuracy (%)")))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="best", handlelength=1.1)
    _tidy(ax, st)


def render_fig1_delay_decode_summary(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "layer" not in df.columns:
        _placeholder(ax, spec, "Delay decoding summary unavailable")
        return
    ax.paper_fig_plot_form = "delay_decode_layer_summary"
    order = [layer for layer in spec.get("conditions", ["layer1", "layer2", "layer3"]) if layer in set(df["layer"].astype(str))]
    if not order:
        _placeholder(ax, spec, "Delay decoding layers unavailable")
        return
    x = np.arange(len(order), dtype=float)
    means, sems = _group_means(df, "layer", order)
    colors = [LAYER_COLORS.get(layer, "0.6") for layer in order]
    ax.bar(x, means, yerr=sems, capsize=st["capsize"], width=st["bar_width"], color=colors, edgecolor="black", linewidth=0.45, alpha=0.86)
    _annotate_bar_values(ax, x, means, sems, suffix="%")
    _reference_lines(ax, spec)
    labels = [(spec.get("display_labels") or {}).get(layer, layer) for layer in order]
    ax.set_xticks(x, labels)
    ax.set_xlabel(str(spec.get("x_axis", "Layer")))
    ax.set_ylabel(str(spec.get("y_axis", "Decoding accuracy (%)")))
    ax.set_ylim(0, 100)
    _tidy(ax, st)


def render_fig1_condition_comparison(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, "Condition metrics unavailable")
        return
    ax.paper_fig_plot_form = "condition_error_rate"
    order = [c for c in spec.get("conditions", ["dynamic_intact", "ux_trial_shuffle", "static_frozen"]) if c in set(df["condition"].astype(str))]
    x = np.arange(len(order), dtype=float)
    means, sems = _group_means(df, "condition", order)
    colors = [CONDITION_COLORS.get(c, "0.7") for c in order]
    ax.bar(x, means, yerr=sems, capsize=st["capsize"], width=st["bar_width"], color=colors, edgecolor="black", linewidth=0.45, alpha=0.86)
    _annotate_bar_values(ax, x, means, sems, suffix="%")
    labels = [(spec.get("display_labels") or {}).get(c, c) for c in order]
    ax.set_xticks(x, labels, rotation=0)
    ax.set_xlabel(str(spec.get("x_axis", "")))
    ax.set_ylabel(str(spec.get("y_axis", "Error rate (%)")))
    ax.set_ylim(0, max(20, min(100, float(np.nanmax(means + sems) + 12) if len(means) else 20)))
    _tidy(ax, st)


def render_fig1_donor_attribution(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "trace" not in df.columns:
        _placeholder(ax, spec, "Attribution data unavailable")
        return
    ax.paper_fig_plot_form = "donor_attribution_grouped"
    conditions = [c for c in spec.get("conditions", ["dynamic_intact", "ux_trial_shuffle"]) if c in set(df["condition"].astype(str))]
    traces = [t for t in spec.get("traces", ["Original", "Donor"]) if t in set(df["trace"].astype(str))]
    width = 0.32
    x = np.arange(len(conditions), dtype=float)
    trace_colors = {"Original": get_plot_color("original_sample_trace"), "Donor": get_plot_color("donor_trace")}
    for i, trace in enumerate(traces):
        offset = (i - (len(traces) - 1) / 2.0) * width
        subset = df[df["trace"].astype(str) == trace]
        means, sems = _group_means(subset, "condition", conditions)
        ax.bar(x + offset, means, yerr=sems, capsize=st["capsize"], width=width * 0.92, color=trace_colors.get(trace, "0.7"), edgecolor="black", linewidth=0.45, alpha=0.88, label=trace)
    labels = [(spec.get("display_labels") or {}).get(c, c) for c in conditions]
    ax.set_xticks(x, labels, rotation=0)
    ax.set_xlabel(str(spec.get("x_axis", "")))
    ax.set_ylabel(str(spec.get("y_axis", "Attribution (%)")))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="upper left", handlelength=0.9)
    _tidy(ax, st)


def render_fig1_error_composition(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or not {"condition", "category"}.issubset(df.columns):
        _placeholder(ax, spec, "Error composition unavailable")
        return
    max_dev = float((stats or {}).get("max_abs_sum_deviation_from_100", 0.0) or 0.0)
    if max_dev > 5.0:
        _placeholder(ax, spec, "Error composition sums invalid")
        return
    ax.paper_fig_plot_form = "error_composition_stacked"
    conditions = [c for c in spec.get("conditions", ["dynamic_intact", "ux_trial_shuffle"]) if c in set(df["condition"].astype(str))]
    categories = [c for c in spec.get("categories", ["Original", "Donor", "Other"]) if c in set(df["category"].astype(str))]
    if not conditions or not categories:
        _placeholder(ax, spec, "Error composition categories unavailable")
        return
    normalized = _normalize_composition(df)
    summary = normalized.groupby(["condition", "category"], as_index=False)["value"].mean()
    x = np.arange(len(conditions), dtype=float)
    bottom = np.zeros(len(conditions), dtype=float)
    colors = {"Original": get_plot_color("original_sample_trace", default="#4C78A8"), "Donor": get_plot_color("donor_trace", default="#F58518"), "Other": get_plot_color("other_residual", default="#BDBDBD")}
    segment_centers: dict[tuple[str, str], float] = {}
    segment_values: dict[tuple[str, str], float] = {}
    for category in categories:
        values = []
        for condition in conditions:
            part = summary[(summary["condition"].astype(str) == condition) & (summary["category"].astype(str) == category)]
            values.append(float(part["value"].iloc[0]) if not part.empty else 0.0)
        values_array = np.asarray(values, dtype=float)
        ax.bar(x, values_array, bottom=bottom, width=st["bar_width"], color=colors.get(category, "0.7"), edgecolor="black", linewidth=0.35, label=category)
        for idx, condition in enumerate(conditions):
            segment_centers[(condition, category)] = float(bottom[idx] + values_array[idx] / 2.0)
            segment_values[(condition, category)] = float(values_array[idx])
        bottom += values_array
    for condition_index, condition in enumerate(conditions):
        for category in ("Original", "Donor"):
            value = segment_values.get((condition, category), 0.0)
            if value <= 0:
                continue
            y = segment_centers[(condition, category)]
            color = "white" if category in {"Original", "Donor"} else "0.15"
            ax.text(
                x[condition_index],
                y,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=st["annotation_fontsize"],
                color=color,
                fontweight="bold",
                clip_on=True,
            )
    labels = [(spec.get("display_labels") or {}).get(c, c) for c in conditions]
    ax.set_xticks(x, labels)
    ax.set_xlabel(str(spec.get("x_axis", "")))
    ax.set_ylabel(str(spec.get("y_axis", "Error composition (% of errors)")))
    ax.set_ylim(0, 100)
    ax.legend(
        frameon=False,
        fontsize=st["legend_fontsize"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=min(3, len(categories)),
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    ax.paper_fig_legend_above_plot = True
    ax.paper_fig_legend_ncols = min(3, len(categories))
    ax.paper_fig_direct_labels = ["Original", "Donor"]
    _tidy(ax, st)


# Backward-compatible renderer names.
render_manual_svg_panel = render_fig1_architecture_schematic
render_dot_summary = render_fig1_baseline_recall
render_layerwise_decoding = render_fig1_delay_decode
render_layerwise_decoding_summary = render_fig1_delay_decode_summary
render_outcome_profile = render_fig1_condition_comparison
render_paired_attribution_shift = render_fig1_donor_attribution


def render_generic_placeholder(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data, stats, style
    _placeholder(ax, spec, "Renderer unavailable")


def plt_rect(ax, xy, width, height, **kwargs):
    from matplotlib.patches import Rectangle

    return Rectangle(xy, width, height, transform=ax.transAxes, **kwargs)


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def _style(style: Mapping[str, Any] | None) -> dict[str, float]:
    out = dict(STYLE)
    out.update({k: float(v) for k, v in dict(style or {}).items() if isinstance(v, (int, float))})
    return out


def _reference_lines(ax, spec: Mapping[str, Any]) -> None:
    for line in spec.get("reference_lines") or []:
        y = float(line.get("value", 0.0))
        ax.axhline(y, color="0.45", linestyle="--", linewidth=0.65)
        if line.get("label"):
            ax.text(0.98, y, str(line["label"]), ha="right", va="bottom", fontsize=5.0, transform=ax.get_yaxis_transform())


def _mean_sem(ax, x: np.ndarray, values: np.ndarray) -> None:
    if values.size == 0:
        return
    mean = float(np.nanmean(values))
    sem = float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    ax.errorbar([float(np.mean(x))], [mean], yerr=[sem], fmt="D", color="black", markersize=3.0, linewidth=0.7, capsize=2.0, zorder=4)


def _annotate_bar_values(ax, x: np.ndarray, means: np.ndarray, sems: np.ndarray, *, suffix: str = "") -> None:
    if len(means) == 0:
        return
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    for xi, mean, sem in zip(x, means, sems):
        y = min(float(mean + sem) + span * 0.035, ymax - span * 0.035)
        ax.text(float(xi), y, f"{float(mean):.1f}{suffix}", ha="center", va="bottom", fontsize=5.1, color="0.15")


def _network_tick_indices(n_values: int) -> list[int]:
    if n_values <= 0:
        return []
    if n_values <= 12:
        return list(range(n_values))
    step = 5 if n_values > 16 else 2
    idx = list(range(0, n_values, step))
    if idx[-1] != n_values - 1:
        idx.append(n_values - 1)
    return idx


def _group_means(df: pd.DataFrame, group_col: str, order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means, sems = [], []
    for label in order:
        values = pd.to_numeric(df.loc[df[group_col].astype(str) == label, "value"], errors="coerce").dropna().to_numpy(dtype=float)
        means.append(float(np.nanmean(values)) if values.size else 0.0)
        sems.append(float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0)
    return np.asarray(means, dtype=float), np.asarray(sems, dtype=float)


def _scatter_points(ax, df: pd.DataFrame, group_col: str, order: list[str], x: np.ndarray) -> None:
    for i, label in enumerate(order):
        values = pd.to_numeric(df.loc[df[group_col].astype(str) == label, "value"], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        jitter = np.linspace(-0.055, 0.055, values.size) if values.size > 1 else np.zeros(1)
        ax.scatter(np.full(values.size, x[i]) + jitter, values, s=8, facecolor="white", edgecolor="0.25", linewidth=0.35, zorder=3)


def _normalize_composition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = [col for col in ("seed_id", "condition") if col in out.columns]
    if not group_cols:
        return out
    totals = out.groupby(group_cols)["value"].transform("sum")
    mask = totals.gt(0)
    out.loc[mask, "value"] = out.loc[mask, "value"] * 100.0 / totals.loc[mask]
    return out


def _tidy(ax, st: Mapping[str, float]) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(st["line_width"])
    ax.tick_params(axis="both", labelsize=st["tick_labelsize"], width=0.55, length=1.8, pad=1.4)
    ax.xaxis.label.set_size(st["axis_labelsize"])
    ax.yaxis.label.set_size(st["axis_labelsize"])
    ax.grid(axis="y", color="0.9", linewidth=0.45)
    ax.set_axisbelow(True)


def _placeholder(ax, spec: Mapping[str, Any], reason: str) -> None:
    ax.paper_fig_placeholder_reason = reason
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.75")
    ax.text(0.5, 0.58, f"Panel {spec.get('panel_id', '?')}", ha="center", va="center", transform=ax.transAxes, fontweight="bold", fontsize=8)
    ax.text(0.5, 0.36, reason, ha="center", va="center", transform=ax.transAxes, fontsize=7, color="0.35", wrap=True)
