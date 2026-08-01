from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from src.plotting.common.colors import NATURE_COMPATIBLE_PALETTE as PALETTE, get_plot_color

from src.plotting.paper_fig.typography import mark_relative_text_size
from src.plotting.paper_fig.svg_assets import (
    load_embedded_square_pngs,
    schematic_arrow,
    schematic_box,
    schematic_digit,
    schematic_text,
    setup_programmatic_schematic,
)


STYLE = {
    "axis_labelsize": 6.3,
    "tick_labelsize": 5.4,
    "legend_fontsize": 5.1,
    "line_width": 0.75,
    "marker_size": 8.0,
    "bar_width": 0.58,
    "capsize": 1.8,
}

STATE_ORDER = ["S0", "S_A", "S_B", "S_AB"]
STATE_LABELS = {"S0": "No\nmemory", "S_A": "Item 1", "S_B": "Item 2", "S_AB": "Fused\npair"}
# Fig.2 treats the fused/captured state as the integrated mechanism-bearing
# outcome.  Reuse the paper's teal mechanism root so this main figure stays in
# the shared blue-orange-teal-neutral language without changing other figures.
FIG2_FUSED_COLOR = PALETTE["mechanism_teal"]
FIG2_FUSED_TINT = PALETTE["mechanism_tint"]
STATE_COLORS = {"S0": get_plot_color("baseline_control"), "S_A": get_plot_color("first_item_reference"), "S_B": get_plot_color("second_item_reference"), "S_AB": FIG2_FUSED_COLOR}
READOUT_COLORS = {"A": get_plot_color("first_item_reference"), "B": get_plot_color("second_item_reference"), "Other": get_plot_color("other_residual"), "Silent": get_plot_color("silent_state")}
COMPOSITION_LABELS = {"Other": "Other", "A": "Item 1", "B": "Item 2", "Silent": "Silent"}


def render_fig2_episode_schematic(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data, stats, style
    width, _ = setup_programmatic_schematic(ax, spec)
    content = spec.get("content") or {}
    timing_raw = content.get("timing_ms") or {}
    timing_keys = ("item_a", "delay1", "item_b", "delay2")
    missing_timing = [key for key in timing_keys if key not in timing_raw]
    if missing_timing:
        raise ValueError(f"Fig.2A timing source-of-truth is missing keys: {missing_timing}")
    timing = {key: int(timing_raw[key]) for key in timing_keys}
    if any(value <= 0 for value in timing.values()):
        raise ValueError(f"Fig.2A timing values must be positive milliseconds, found {timing}")
    item_a, item_b = load_embedded_square_pngs(spec)
    blue, orange = get_plot_color("first_item_reference"), get_plot_color("second_item_reference")
    fused, captured = FIG2_FUSED_COLOR, FIG2_FUSED_COLOR
    neutral, ink = get_plot_color("guide"), get_plot_color("ink")

    schematic_box(ax, 0.5, 25.2, width - 1.0, 19.2, facecolor=PALETTE["white"], edgecolor=PALETTE["white"], radius=1.8, role="top_band")
    schematic_box(ax, 0.5, 1.0, width - 1.0, 23.2, facecolor=PALETTE["white"], edgecolor=PALETTE["white"], radius=1.8, role="bottom_band")

    schematic_text(ax, 11.25, 46.6, "Item A", color=blue, role="header_item_a")
    schematic_text(ax, 74.25, 46.6, "Item B", color=orange, role="header_item_b")
    schematic_digit(ax, item_a, 3.0, 27.0, 16.5, edgecolor=blue, role="digit_item_a")
    schematic_box(ax, 31.0, 29.3, 24.0, 11.8, facecolor=PALETTE["neutral_pale"], edgecolor=PALETTE["neutral_light"], text="Delay", text_color=ink, linestyle=(0, (3, 2)), role="delay_a")
    schematic_digit(ax, item_b, 66.0, 27.0, 16.5, edgecolor=orange, role="digit_item_b")
    schematic_box(ax, 94.0, 29.3, 24.0, 11.8, facecolor=PALETTE["neutral_pale"], edgecolor=PALETTE["neutral_light"], text="Delay", text_color=ink, linestyle=(0, (3, 2)), role="delay_b")
    schematic_box(ax, 130.5, 28.2, 29.0, 14.0, facecolor=FIG2_FUSED_TINT, edgecolor=captured, text="Capture", text_color=PALETTE["ink"], linewidth=0.9, role="capture")
    for start, end in (((19.5, 35.25), (31.0, 35.25)), ((55.0, 35.25), (66.0, 35.25)), ((82.5, 35.25), (94.0, 35.25)), ((118.0, 35.25), (130.5, 35.25))):
        schematic_arrow(ax, start, end, color=neutral)

    schematic_box(ax, 1.5, 3.0, 21.0, 17.5, facecolor=PALETTE["primary_tint"], edgecolor=blue, text="A written", text_color=PALETTE["ink"], role="a_written")
    schematic_box(ax, 29.0, 3.0, 27.0, 17.5, facecolor=PALETTE["primary_tint"], edgecolor=PALETTE["primary_cyan"], text="Retained A trace", text_color=PALETTE["ink"], linestyle=(0, (3, 2)), role="retained_a")
    schematic_box(ax, 63.0, 3.0, 21.0, 17.5, facecolor=PALETTE["comparison_tint"], edgecolor=orange, text="B written", text_color=PALETTE["ink"], role="b_written")
    schematic_box(ax, 91.0, 3.0, 27.0, 17.5, facecolor=FIG2_FUSED_TINT, edgecolor=fused, text="Fused state\nevolves", text_color=PALETTE["ink"], linestyle=(0, (3, 2)), role="fused_state")
    schematic_box(ax, 129.5, 3.0, 30.0, 17.5, facecolor=FIG2_FUSED_TINT, edgecolor=captured, text="Captured\nfused trace", text_color=PALETTE["ink"], linewidth=0.9, role="captured_trace")
    for start, end in (((22.5, 11.75), (29.0, 11.75)), ((56.0, 11.75), (63.0, 11.75)), ((84.0, 11.75), (91.0, 11.75)), ((118.0, 11.75), (129.5, 11.75))):
        schematic_arrow(ax, start, end, color=PALETTE["neutral_mid"])
    for start, end, color in (
        ((11.25, 27.0), (11.25, 20.5), blue),
        ((43.0, 29.3), (43.0, 20.5), PALETTE["primary_cyan"]),
        ((74.25, 27.0), (74.25, 20.5), orange),
        ((106.0, 29.3), (106.0, 20.5), fused),
        ((145.0, 28.2), (145.0, 20.5), captured),
    ):
        schematic_arrow(ax, start, end, color=color)

    ax.paper_fig_plot_form = "programmatic_two_item_episode_schematic"
    ax.paper_fig_episode_timing_ms = timing
    ax.paper_fig_timing_source = str(content.get("timing_source", ""))
    ax.paper_fig_timing_labels_visible = False


def render_fig2_dual_retention_constituents(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, "Dual retention data unavailable")
        return
    ax.paper_fig_plot_form = "dual_retention_constituent_similarity"
    order = ["S_A", "S_B"]
    _bar_summary(ax, df, "condition", order, colors=[STATE_COLORS["S_A"], STATE_COLORS["S_B"]], st=st, alpha=0.82)
    ax.set_xticks(np.arange(len(order)), [_display(spec, v) for v in order])
    ax.set_ylabel(str(spec.get("y_axis", "Similarity to S_AB")))
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.paper_fig_visual_categories = order
    ax.paper_fig_raw_points = False
    _tidy(ax, st)


def render_fig2_pair_specificity(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, "Pair specificity data unavailable")
        return
    ax.paper_fig_plot_form = "true_vs_shuffled_pair_boxplot"
    order = ["True pair", "Shuffled pair"]
    _boxplot_by_condition(ax, df, "condition", order, colors=[FIG2_FUSED_COLOR, get_plot_color("baseline_control")], st=st)
    ax.set_xticks(np.arange(len(order)), ["Experienced\npair", "Shuffled\npair"])
    ax.set_ylabel(str(spec.get("y_axis", "Pair specificity")))
    ax.set_xlabel("")
    ax.paper_fig_raw_points = False
    _autoscale_y(ax, df["value"])
    ax.set_ylim(0.44, 1.05)
    ax.set_yticks([0.5, 0.7, 0.9])
    _tidy(ax, st)


def render_fig2_morphology_closure(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, "Morphology closure data unavailable")
        return
    required = {
        "WPRI": "WPRI",
        "delta_r2_interaction_beyond_bounded_saturation": "Cross-fit ΔR²",
    }
    if not set(required).issubset(set(df["metric"].astype(str))):
        _placeholder(ax, spec, "Cross-fitted Fig.2C metrics unavailable")
        return

    ax.set_axis_off()
    plot_axis = ax.inset_axes([0.09, 0.0, 0.72, 1.0])
    right_axis = plot_axis.twinx()
    colors = {"WPRI": get_plot_color("sequence_state"), "delta_r2_interaction_beyond_bounded_saturation": FIG2_FUSED_COLOR}
    limits = spec.get("metric_y_limits") or {}
    positions = {
        "WPRI": 0.0,
        "delta_r2_interaction_beyond_bounded_saturation": 1.0,
    }
    axes = {
        "WPRI": plot_axis,
        "delta_r2_interaction_beyond_bounded_saturation": right_axis,
    }
    for metric, label in required.items():
        target_axis = axes[metric]
        position = positions[metric]
        values = pd.to_numeric(df.loc[df["metric"].astype(str).eq(metric), "value"], errors="coerce").dropna().to_numpy(dtype=float)
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        target_axis.bar(
            [position],
            [mean],
            yerr=[sem],
            width=0.52,
            capsize=st["capsize"],
            color=colors[metric],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.84,
            zorder=2,
        )
        metric_limits = limits.get(metric)
        if isinstance(metric_limits, (list, tuple)) and len(metric_limits) == 2:
            target_axis.set_ylim(float(metric_limits[0]), float(metric_limits[1]))

    plot_axis.set_xlim(-0.58, 1.58)
    right_axis.set_xlim(plot_axis.get_xlim())
    plot_axis.set_xticks(
        [positions["WPRI"], positions["delta_r2_interaction_beyond_bounded_saturation"]],
        [required["WPRI"], required["delta_r2_interaction_beyond_bounded_saturation"]],
    )
    plot_axis.set_xlabel("")
    plot_axis.set_ylabel("WPRI", color="black")
    right_axis.set_ylabel("Held-out ΔR²", color="black")
    plot_axis.tick_params(axis="y", colors="black")
    right_axis.tick_params(axis="y", colors="black")
    plot_axis.set_yticks([0.00, 0.04, 0.08, 0.12, 0.16])
    right_axis.set_yticks([0.001, 0.002, 0.003], ["1", "2", "3"])

    _tidy(plot_axis, st)
    plot_axis.grid(False)
    right_axis.grid(False)
    right_axis.spines["top"].set_visible(False)
    right_axis.spines["left"].set_visible(False)
    right_axis.spines["bottom"].set_visible(False)
    right_axis.spines["right"].set_visible(True)
    right_axis.spines["right"].set_linewidth(st["line_width"])
    right_axis.tick_params(
        axis="y",
        labelsize=st["tick_labelsize"],
        width=0.55,
        length=1.8,
        pad=0.8,
    )
    right_axis.yaxis.label.set_size(st["axis_labelsize"])
    right_axis.yaxis.labelpad = 1.0
    # The shared semantic-gap fitter is left-axis oriented.  Keep the twin
    # axis on its explicitly tuned right-side spacing so its tick labels stay
    # visually attached to the spine.
    right_axis.paper_fig_skip_semantic_gap_fit = True
    multiplier_artist = right_axis.text(
        1.01,
        0.99,
        "×10⁻³",
        transform=right_axis.transAxes,
        ha="left",
        va="top",
        fontsize=st["tick_labelsize"] * 0.75,
        color="black",
    )
    mark_relative_text_size(multiplier_artist, 0.75)

    ax.paper_fig_plot_form = "wpri_and_crossfit_interaction_dual_axis_bar"
    ax.paper_fig_x_metric = "Metric"
    ax.paper_fig_y_metric = "WPRI (left) and held-out ΔR² (right)"
    ax.paper_fig_raw_points = False
    ax.paper_fig_raw_point_count = 0
    ax.paper_fig_value_labels = False
    ax.paper_fig_value_label_count = 0
    ax.paper_fig_dual_y_axes = True
    ax.paper_fig_secondary_y_label = "Held-out ΔR²"
    ax.paper_fig_secondary_y_multiplier = "×10⁻³"
    ax.paper_fig_secondary_y_multiplier_size_scale = 0.75
    ax.paper_fig_measure_right_stack = True
    ax.paper_fig_composite_child_axes = True
    ax.paper_fig_child_axes = [plot_axis, right_axis]
    ax.paper_fig_inner_axes_bounds = [[0.09, 0.0, 0.72, 1.0], [0.09, 0.0, 0.72, 1.0]]
    ax.paper_fig_inner_axes_aligned = True


def render_fig2_neutral_ping_composition(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or not {"condition", "category"}.issubset(df.columns):
        _placeholder(ax, spec, "Neutral ping composition unavailable")
        return
    ax.paper_fig_plot_form = "neutral_ping_readout_composition_stacked"
    conditions = [c for c in STATE_ORDER if c in set(df["condition"].astype(str))]
    requested_categories = [str(c) for c in spec.get("categories", ["Other", "A", "B", "Silent"])]
    categories = [c for c in requested_categories if c in set(df["category"].astype(str))]
    if not conditions or not categories:
        _placeholder(ax, spec, "Neutral ping categories unavailable")
        return
    summary = df.groupby(["condition", "category"], as_index=False)["value"].mean()
    x = np.arange(len(conditions), dtype=float)
    bottom = np.zeros(len(conditions), dtype=float)
    for category in categories:
        values = []
        for condition in conditions:
            part = summary[(summary["condition"].astype(str) == condition) & (summary["category"].astype(str) == category)]
            values.append(float(part["value"].iloc[0]) if not part.empty else 0.0)
        color = READOUT_COLORS.get(category, "0.7")
        edge = "0.55" if category == "Silent" else "black"
        ax.bar(x, values, bottom=bottom, width=st["bar_width"], color=color, edgecolor=edge, linewidth=0.35, label=COMPOSITION_LABELS.get(category, category))
        bottom += np.asarray(values, dtype=float)
    ax.set_xticks(x, [_display(spec, c) for c in conditions])
    ax.set_ylim(0, 100)
    ax.set_ylabel(str(spec.get("y_axis", "Readout composition (%)")))
    ax.set_xlabel("")
    legend = ax.legend(frameon=False, fontsize=st["legend_fontsize"], ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.04), handlelength=0.8, handletextpad=0.28, columnspacing=0.55, borderaxespad=0.0)
    ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
    ax.paper_fig_legend_ncols = 4
    ax.paper_fig_legend_above_plot = True
    _tidy(ax, st)


def render_fig2_partial_cue_target(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    target_item = str(spec.get("target_item", "")).upper()
    if target_item in {"A", "B"} and "target_item" in df.columns:
        df = df[df["target_item"].astype(str).str.upper().eq(target_item)].copy()
    if df.empty:
        _placeholder(ax, spec, f"Target {target_item or '?'} partial-cue data unavailable")
        return
    ax.paper_fig_plot_form = "partial_cue_target_recovery_curve"
    curve = df[df.get("curve_or_summary", pd.Series("curve", index=df.index)).astype(str).eq("curve")].copy() if "curve_or_summary" in df.columns else df.copy()
    if curve.empty or "keep_prob" not in curve.columns:
        _placeholder(ax, spec, f"Target {target_item or '?'} partial-cue curve unavailable")
        return
    ax.paper_fig_target_item = target_item
    for condition in [c for c in STATE_ORDER if c in set(curve["condition"].astype(str))]:
        part = curve[curve["condition"].astype(str).eq(condition)].copy()
        part["keep_prob"] = pd.to_numeric(part["keep_prob"], errors="coerce")
        grouped = part.dropna(subset=["keep_prob"]).groupby("keep_prob", as_index=False)["value"].agg(["mean", "sem"]).reset_index()
        if grouped.empty:
            continue
        x = grouped["keep_prob"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        sem = grouped["sem"].fillna(0).to_numpy(dtype=float)
        color = STATE_COLORS.get(condition, "0.3")
        ax.plot(x, y, marker="o", markersize=2.4, linewidth=0.82, color=color, label=STATE_LABELS.get(condition, condition).replace("\n", " "))
        if part["seed_id"].replace("", pd.NA).dropna().nunique() > 1:
            _fill_sem_band(ax, x, y, sem, color)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel(str(spec.get("x_axis", "Keep probability")))
    ax.set_ylabel(str(spec.get("y_axis", "Target recovery (%)")))
    ax.set_title(f"Target {target_item}", fontsize=6.5, pad=1.6)
    show_legend = bool(spec.get("show_legend", target_item == "B"))
    if show_legend:
        legend = ax.legend(frameon=False, fontsize=st["legend_fontsize"], ncol=2, loc="lower right", handlelength=1.0, columnspacing=0.7)
        ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
        ax.paper_fig_legend_ncols = 2
    _tidy(ax, st)


def render_fig2_partial_cue_targets_combined(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "target_item" not in df.columns:
        _placeholder(ax, spec, "Partial-cue target data unavailable")
        return
    ax.set_axis_off()
    ax.paper_fig_plot_form = "partial_cue_target_recovery_combined"
    ax.paper_fig_target_items = ["A", "B"]
    ax.paper_fig_composite_child_axes = True
    span_w_mm = float((spec.get("size_mm") or {}).get("width", (spec.get("position_mm") or {}).get("w", 106.667)))
    left_margin_mm = 7.1
    right_margin_mm = 0.0
    gap_w_mm = 8.0
    col_w_mm = max(30.0, (span_w_mm - left_margin_mm - right_margin_mm - gap_w_mm) / 2.0)
    bottom_mm = 5.5
    height_mm = 39.333
    left_bounds = [left_margin_mm / span_w_mm, bottom_mm / 48.0, col_w_mm / span_w_mm, height_mm / 48.0]
    right_bounds = [(left_margin_mm + col_w_mm + gap_w_mm) / span_w_mm, bottom_mm / 48.0, col_w_mm / span_w_mm, height_mm / 48.0]
    left = ax.inset_axes(left_bounds)
    right = ax.inset_axes(right_bounds)
    right.sharey(left)
    ax.paper_fig_child_axes = [left, right]
    ax.paper_fig_inner_axes_bounds = [left_bounds, right_bounds]
    ax.paper_fig_inner_axes_aligned = True
    handles: list[Any] = []
    labels: list[str] = []
    for child, target, show_ylabel in ((left, "A", True), (right, "B", False)):
        child_handles, child_labels = _plot_partial_cue_axis(child, df, target, spec, st, show_ylabel=show_ylabel)
        if child_handles and not handles:
            handles, labels = child_handles, child_labels
    if handles:
        legend = ax.legend(
            handles,
            labels,
            frameon=False,
            fontsize=st["legend_fontsize"],
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            handlelength=1.0,
            handletextpad=0.35,
            columnspacing=0.7,
            borderaxespad=0.0,
        )
        ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
        ax.paper_fig_legend_ncols = 4
        ax.paper_fig_legend_above_plot = True


# Backward-compatible renderer names from old Fig.2 specs.
render_fig2a_episode_schematic = render_fig2_episode_schematic
render_fig2_dual_retention = render_fig2_dual_retention_constituents
render_one_sample_dot_summary = render_fig2_dual_retention_constituents
render_paired_condition_plot = render_fig2_pair_specificity
render_fig2_wpri_and_linear_residual = render_fig2_morphology_closure
render_wpri_density = render_fig2_morphology_closure
render_fig2_neutral_ping = render_fig2_neutral_ping_composition
render_neutral_ping_functional_access = render_fig2_neutral_ping_composition
render_fig2_partial_cue = render_fig2_partial_cue_target
render_partial_cue_completion = render_fig2_partial_cue_target


def _bar_summary(ax, df: pd.DataFrame, group_col: str, order: list[str], *, colors: list[str], st: Mapping[str, float], alpha: float = 0.86) -> None:
    x = np.arange(len(order), dtype=float)
    means, sems = [], []
    for label in order:
        values = pd.to_numeric(df.loc[df[group_col].astype(str) == label, "value"], errors="coerce").dropna().to_numpy(dtype=float)
        means.append(float(np.nanmean(values)) if values.size else 0.0)
        sems.append(float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0)
    ax.bar(x, means, yerr=sems, width=st["bar_width"], capsize=st["capsize"], color=colors[: len(order)], edgecolor="black", linewidth=0.45, alpha=alpha)


def _boxplot_by_condition(ax, df: pd.DataFrame, group_col: str, order: list[str], *, colors: list[str], st: Mapping[str, float]) -> None:
    data = [
        pd.to_numeric(df.loc[df[group_col].astype(str) == label, "value"], errors="coerce").dropna().to_numpy(dtype=float)
        for label in order
    ]
    box = ax.boxplot(
        data,
        positions=np.arange(len(order), dtype=float),
        widths=st["bar_width"],
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.65},
        capprops={"color": "black", "linewidth": 0.65},
        boxprops={"edgecolor": "black", "linewidth": 0.55},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.62)


def _bar_scatter(ax, df: pd.DataFrame, group_col: str, order: list[str], *, colors: list[str], st: Mapping[str, float], alpha: float = 0.86) -> None:
    x = np.arange(len(order), dtype=float)
    means, sems = [], []
    for label in order:
        values = pd.to_numeric(df.loc[df[group_col].astype(str) == label, "value"], errors="coerce").dropna().to_numpy(dtype=float)
        means.append(float(np.nanmean(values)) if values.size else 0.0)
        sems.append(float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0)
    ax.bar(x, means, yerr=sems, width=st["bar_width"], capsize=st["capsize"], color=colors[: len(order)], edgecolor="black", linewidth=0.45, alpha=alpha)
    for i, label in enumerate(order):
        values = pd.to_numeric(df.loc[df[group_col].astype(str) == label, "value"], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        if values.size > 220:
            step = max(1, values.size // 220)
            values = values[::step]
        jitter = np.linspace(-0.07, 0.07, values.size) if values.size > 1 else np.zeros(1)
        ax.scatter(np.full(values.size, x[i]) + jitter, values, s=st["marker_size"], facecolor="white", edgecolor="0.25", linewidth=0.28, zorder=3, alpha=0.62)
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(min(len(df), 750))
    ax.paper_fig_raw_point_alpha = 0.62


def _fill_sem_band(ax, x: np.ndarray, y: np.ndarray, sem: np.ndarray, color: str, *, alpha: float = 0.16) -> None:
    if x.size == 0 or y.size == 0 or sem.size == 0:
        return
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    errs = np.nan_to_num(sem[order], nan=0.0)
    ax.fill_between(xs, ys - errs, ys + errs, color=color, alpha=alpha, linewidth=0, zorder=1)
    ax.paper_fig_has_shaded_band = True
    bands = list(getattr(ax, "paper_fig_shaded_band", []) or [])
    bands.append({"color": color, "alpha": alpha, "kind": "mean_sem"})
    ax.paper_fig_shaded_band = bands


def _plot_partial_cue_axis(ax, df: pd.DataFrame, target_item: str, spec: Mapping[str, Any], st: Mapping[str, float], *, show_ylabel: bool) -> tuple[list[Any], list[str]]:
    target_df = df[df["target_item"].astype(str).str.upper().eq(target_item)].copy()
    curve = target_df[target_df.get("curve_or_summary", pd.Series("curve", index=target_df.index)).astype(str).eq("curve")].copy() if "curve_or_summary" in target_df.columns else target_df
    handles: list[Any] = []
    labels: list[str] = []
    if curve.empty or "keep_prob" not in curve.columns:
        _placeholder(ax, spec, f"Target {target_item} partial-cue curve unavailable")
        return handles, labels
    for condition in [c for c in STATE_ORDER if c in set(curve["condition"].astype(str))]:
        part = curve[curve["condition"].astype(str).eq(condition)].copy()
        part["keep_prob"] = pd.to_numeric(part["keep_prob"], errors="coerce")
        grouped = part.dropna(subset=["keep_prob"]).groupby("keep_prob", as_index=False)["value"].agg(["mean", "sem"]).reset_index()
        if grouped.empty:
            continue
        x = grouped["keep_prob"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        sem = grouped["sem"].fillna(0).to_numpy(dtype=float)
        color = STATE_COLORS.get(condition, "0.3")
        (line,) = ax.plot(x, y, marker="o", markersize=2.1, linewidth=0.78, color=color, label=STATE_LABELS.get(condition, condition).replace("\n", " "))
        if not handles:
            pass
        if part["seed_id"].replace("", pd.NA).dropna().nunique() > 1:
            _fill_sem_band(ax, x, y, sem, color)
        handles.append(line)
        labels.append(STATE_LABELS.get(condition, condition).replace("\n", " "))
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_xlabel(str(spec.get("x_axis", "Keep probability")))
    if show_ylabel:
        ax.set_ylabel(str(spec.get("y_axis", "Target recovery (%)")))
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    ax.text(0.5, 0.98, f"Target {target_item}", ha="center", va="top", fontsize=6.1, transform=ax.transAxes)
    _tidy(ax, st)
    return handles, labels


def _paired_lines(ax, df: pd.DataFrame, order: list[str], *, key_col: str, color: str) -> None:
    if key_col not in df.columns or "seed_id" not in df.columns:
        return
    x_lookup = {label: i for i, label in enumerate(order)}
    for _, part in df.groupby(["seed_id", key_col], dropna=False):
        vals = []
        xs = []
        for label in order:
            row = part[part["condition"].astype(str) == label]
            if row.empty:
                continue
            xs.append(x_lookup[label])
            vals.append(float(pd.to_numeric(row["value"], errors="coerce").iloc[0]))
        if len(vals) == len(order):
            ax.plot(xs, vals, color=color, alpha=0.13, linewidth=0.42, zorder=1)


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


def _autoscale_y(ax, values: pd.Series, *, include_zero: bool = False) -> None:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    pad = max(0.05, 0.12 * (hi - lo if hi > lo else 1.0))
    ax.set_ylim(lo - pad, hi + pad)


def _tidy(ax, st: Mapping[str, float]) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(st["line_width"])
    ax.tick_params(axis="both", labelsize=st["tick_labelsize"], width=0.55, length=1.8, pad=1.4)
    ax.xaxis.label.set_size(st["axis_labelsize"])
    ax.yaxis.label.set_size(st["axis_labelsize"])
    ax.yaxis.labelpad = 1.0
    ax.grid(False)


def _display(spec: Mapping[str, Any], value: str) -> str:
    label = (spec.get("display_labels") or {}).get(value, STATE_LABELS.get(value, value))
    return str(label)


def _placeholder(ax, spec: Mapping[str, Any], reason: str) -> None:
    ax.paper_fig_placeholder_reason = reason
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.75")
    ax.text(0.5, 0.58, f"Panel {spec.get('panel_id', '?')}", ha="center", va="center", transform=ax.transAxes, fontweight="bold", fontsize=8)
    ax.text(0.5, 0.36, reason, ha="center", va="center", transform=ax.transAxes, fontsize=7, color="0.35", wrap=True)


def _rect(ax, x: float, y: float, w: float, h: float, **kwargs):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), w, h, transform=ax.transAxes, **kwargs)
