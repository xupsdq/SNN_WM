from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from src.plotting.common.colors import NATURE_COMPATIBLE_PALETTE as PALETTE, get_plot_color

from src.plotting.paper_fig.panels.fig2_panels import (
    FIG2_FUSED_COLOR,
    STATE_COLORS,
    STATE_LABELS,
    STATE_ORDER,
    _autoscale_y,
    _bar_summary,
    _clean,
    _fill_sem_band,
    _placeholder,
    _style,
    _tidy,
)


LAYER_ORDER = ["layer1", "layer2", "layer3"]
LAYER_LABELS = {"layer1": "L1", "layer2": "L2", "layer3": "L3"}
MODEL_ORDER = ["A_only", "B_only", "mean_AB", "sum_AB", "unconstrained_AB", "convex_AB"]
MODEL_LABELS = {"A_only": "A", "B_only": "B", "mean_AB": "Mean", "sum_AB": "Sum", "unconstrained_AB": "Unc.", "convex_AB": "Convex"}
S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT = 1.15
S2_SOURCE_MARKER_SIZE_PT2 = 16.0


def render_s3_wpri_across_layers(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    st = _style(style)
    order = ["layer1", "layer2", "layer3"]
    df, summary_by_layer = _fixed_s2_rows_and_summaries(panel_data, stats, order, summary_key="layer")
    colors = [PALETTE["primary_navy"], PALETTE["comparison_coral"], PALETTE["mechanism_teal"]]
    markers = ["o", "s", "^"]
    x = np.arange(len(order), dtype=float)
    for index, (layer, color, marker) in enumerate(zip(order, colors, markers)):
        summary = summary_by_layer[layer]
        ax.bar(
            [x[index]],
            [float(summary["mean"])],
            yerr=[float(summary["sem"])],
            width=0.58,
            capsize=3.0,
            color=color,
            edgecolor="black",
            linewidth=S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT,
            alpha=0.82,
            error_kw={
                "elinewidth": S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT,
                "capthick": S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT,
            },
        )
        values = df.loc[df["condition"].astype(str).eq(layer), "value"].astype(float).to_numpy()
        jitter = np.linspace(-0.10, 0.10, values.size)
        ax.scatter(
            np.full(values.size, x[index]) + jitter,
            values,
            s=S2_SOURCE_MARKER_SIZE_PT2,
            marker=marker,
            facecolor="white",
            edgecolor="0.20",
            linewidth=S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT,
            zorder=3,
        )
    ax.axhline(0, color="0.35", linewidth=S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT, linestyle="--")
    ax.set_xticks(x, ["L1", "L2", "L3"])
    ax.set_xlabel(str(spec.get("x_axis", "Layer")))
    ax.set_ylabel(str(spec.get("y_axis", "Whole-pair representation index (dimensionless)")))
    ax.set_ylim(bottom=0.0)
    ax.paper_fig_plot_form = "s3_wpri_across_layers_fixed_persisted"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(df))
    ax.paper_fig_scientific_recomputation = False
    _tidy(ax, st)


def render_s3_residual_across_layers(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    _layerwise_bar(ax, panel_data, spec, style, plot_form="s3_residual_across_layers", placeholder="Layerwise residual unavailable")


def render_s3_linear_model_comparison(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, "Linear model comparison unavailable")
        return
    ax.paper_fig_plot_form = "s3_linear_model_comparison"
    order = [m for m in spec.get("model_order", MODEL_ORDER) if m in set(df["condition"].astype(str))]
    if not order:
        order = [m for m in MODEL_ORDER if m in set(df.get("model_name", pd.Series(dtype=str)).astype(str))]
    if not order:
        _placeholder(ax, spec, "Linear model labels unavailable")
        return
    colors = [PALETTE["neutral_light"], PALETTE["neutral_light"], PALETTE["primary_pale"], PALETTE["primary_pale"], PALETTE["mechanism_teal"], PALETTE["mechanism_mint"]][: len(order)]
    _bar_summary(ax, df.assign(condition=df["condition"].astype(str)), "condition", order, colors=colors, st=st, alpha=0.82)
    ax.set_xticks(np.arange(len(order)), [MODEL_LABELS.get(m, m) for m in order], rotation=25, ha="right")
    ax.set_ylabel(str(spec.get("y_axis", "Fit R2")))
    ax.set_xlabel("")
    ax.paper_fig_model_labels_readable = True
    ax.paper_fig_raw_points = False
    _autoscale_y(ax, df["value"], include_zero=True)
    _tidy(ax, st)


def render_s3_crossfit_interaction(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    st = _style(style)
    order = ["Linear additive", "Quadratic marginals", "Bounded saturation"]
    df, summary_by_condition = _fixed_s2_rows_and_summaries(panel_data, stats, order, summary_key="condition")
    colors = [PALETTE["primary_pale"], PALETTE["primary_cyan"], FIG2_FUSED_COLOR]
    markers = ["o", "s", "^"]
    x = np.arange(len(order), dtype=float)
    for index, (condition, color, marker) in enumerate(zip(order, colors, markers)):
        summary = summary_by_condition[condition]
        ax.bar(
            [x[index]],
            [float(summary["mean"])],
            yerr=[float(summary["sem"])],
            width=0.58,
            capsize=3.0,
            color=color,
            edgecolor="black",
            linewidth=S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT,
            alpha=0.84,
            error_kw={
                "elinewidth": S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT,
                "capthick": S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT,
            },
        )
        values = df.loc[df["condition"].astype(str).eq(condition), "value"].astype(float).to_numpy()
        jitter = np.linspace(-0.10, 0.10, values.size)
        ax.scatter(
            np.full(values.size, x[index]) + jitter,
            values,
            s=S2_SOURCE_MARKER_SIZE_PT2,
            marker=marker,
            facecolor="white",
            edgecolor="0.20",
            linewidth=S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT,
            zorder=3,
        )
    ax.axhline(0, color="0.35", linewidth=S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT, linestyle="--")
    ax.set_xticks(x, ["Linear", "Quadratic\nmarginals", "Bounded\nsaturation"])
    ax.set_ylabel(str(spec.get("y_axis", r"Held-out interaction $\Delta R^2$ (dimensionless)")))
    ax.set_xlabel(str(spec.get("x_axis", "Marginal baseline")))
    ax.paper_fig_plot_form = "crossfit_interaction_model_sensitivity_fixed_persisted"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(df))
    ax.paper_fig_scientific_recomputation = False
    ax.set_ylim(bottom=0.0)
    _tidy(ax, st)


def render_s3_crossfit_null_calibration(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    st = _style(style)
    order = ["Linear + noise", "Bounded saturation", "Sequence/marginal matched"]
    df, summary_by_condition = _fixed_s2_rows_and_summaries(panel_data, stats, order, summary_key="condition")
    calibration = dict((stats or {}).get("calibration_summary") or {})
    null_order = [
        "strict_linear_iid_noise",
        "bounded_separable_saturation",
        "sequence_marginal_matched_interaction_permutation",
    ]
    if set(calibration) != set(null_order):
        raise RuntimeError("S2C frozen calibration summaries are missing, duplicate, or extra.")
    colors = [PALETTE["primary_pale"], PALETTE["comparison_tint"], PALETTE["neutral_light"]]
    markers = ["o", "s", "^"]
    x = np.arange(len(order), dtype=float)
    for index, (condition, null_model, color, marker) in enumerate(zip(order, null_order, colors, markers)):
        values = df.loc[df["condition"].astype(str).eq(condition), "value"].astype(float).to_numpy()
        jitter = np.linspace(-0.18, 0.18, values.size)
        ax.scatter(
            np.full(values.size, index) + jitter,
            values,
            s=S2_SOURCE_MARKER_SIZE_PT2,
            marker=marker,
            facecolor=color,
            edgecolor="0.20",
            linewidth=S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT,
            alpha=0.88,
            zorder=3,
        )
        summary = summary_by_condition[condition]
        frozen = dict(calibration[null_model])
        ax.hlines(
            float(summary["mean"]),
            index - 0.27,
            index + 0.27,
            colors="black",
            linewidth=S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT,
            zorder=4,
        )
        ax.hlines(
            float(frozen["median_null_delta_r2"]),
            index - 0.22,
            index + 0.22,
            colors="0.25",
            linestyles="--",
            linewidth=S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT,
            zorder=4,
        )
        ax.scatter(
            [index],
            [float(frozen["observed_reference_delta_r2"])],
            marker="D",
            s=25.0,
            facecolor="white",
            edgecolor="black",
            linewidth=S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT,
            zorder=5,
        )
    ax.axhline(0, color="0.25", linewidth=S2_SOURCE_SCIENTIFIC_LINEWIDTH_PT, linestyle="--")
    ax.set_xticks(x, ["Linear +\nnoise", "Bounded\nsaturation", "Sequence/\nmarginal matched"])
    for tick_label in ax.get_xticklabels():
        tick_label.set_linespacing(0.75)
    ax.tick_params(axis="x", pad=0.5)
    ax.set_ylabel(str(spec.get("y_axis", r"Null held-out interaction $\Delta R^2$ (dimensionless)")))
    ax.set_xlabel(str(spec.get("x_axis", "Null model")), labelpad=1.0)
    calibration = dict((stats or {}).get("calibration_summary") or {})
    annotation_positions = [
        (1.0 / 6.0, 1.005, "center"),
        (0.5, 1.005, "center"),
        (5.0 / 6.0, 1.005, "center"),
    ]
    for index, null_model in enumerate(null_order):
        payload = dict(calibration.get(null_model) or {})
        if index < 2:
            if payload.get("false_positive_count_one_sided_alpha_0_05") != 0:
                raise RuntimeError(f"S2C frozen FPR count mismatch for {null_model}.")
            if payload.get("n_dataset_replicates") != 100:
                raise RuntimeError(f"S2C frozen FPR denominator mismatch for {null_model}.")
            if payload.get("false_positive_rate_exact_95_ci") != [0.0, 0.03621669264517641]:
                raise RuntimeError(f"S2C frozen exact confidence interval mismatch for {null_model}.")
            label = "FPR 0/100;\nexact 95% CI\n0–0.03621669264517641"
        else:
            if payload.get("empirical_p_observed_vs_null") != 0.009900990099009901:
                raise RuntimeError("S2C frozen empirical P mismatch.")
            label = "empirical P =\n0.009900990099009901"
        ax.text(
            annotation_positions[index][0],
            annotation_positions[index][1],
            label,
            transform=ax.transAxes,
            ha=annotation_positions[index][2],
            va="bottom",
            fontsize=9.1,
            linespacing=0.75,
            clip_on=False,
        )
    ax.paper_fig_plot_form = "crossfit_three_null_calibration_fixed_persisted"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(df))
    ax.paper_fig_has_zero_reference = True
    ax.paper_fig_scientific_recomputation = False
    ax.margins(y=0.08)
    _tidy(ax, st)


def _fixed_s2_rows_and_summaries(
    panel_data: pd.DataFrame | None,
    stats: Mapping[str, Any] | None,
    expected_order: list[str],
    *,
    summary_key: str,
) -> tuple[pd.DataFrame, dict[str, Mapping[str, Any]]]:
    """Validate the frozen plotting rows and return frozen summaries verbatim."""
    if panel_data is None or stats is None:
        raise RuntimeError("Fixed S2 panel rows and statistics are required; no placeholder fallback is allowed.")
    df = panel_data.copy()
    required = {"figure_id", "panel_id", "condition", "network_id", "value"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"Fixed S2 render payload is missing columns: {missing}")
    if len(df) != 60:
        raise RuntimeError(f"Fixed S2 render payload must contain exactly 60 rows, got {len(df)}.")
    if df[["condition", "network_id", "value"]].isna().any().any():
        raise RuntimeError("Fixed S2 render payload contains missing identities or values.")
    if not np.isfinite(df["value"].astype(float).to_numpy()).all():
        raise RuntimeError("Fixed S2 render payload contains non-finite values.")
    observed_order = list(dict.fromkeys(df["condition"].astype(str).tolist()))
    if observed_order != expected_order:
        raise RuntimeError(f"Fixed S2 render condition order mismatch: {observed_order}")
    if df.duplicated(subset=["condition", "network_id"], keep=False).any():
        raise RuntimeError("Fixed S2 render payload contains duplicate condition/network rows.")
    for condition in expected_order:
        if len(df.loc[df["condition"].astype(str).eq(condition)]) != 20:
            raise RuntimeError(f"Fixed S2 render payload has missing or extra rows for {condition}.")

    summaries = list(stats.get("summaries") or [])
    summary_by_condition: dict[str, Mapping[str, Any]] = {}
    for summary in summaries:
        key = str(summary.get(summary_key))
        if key in summary_by_condition:
            raise RuntimeError(f"Fixed S2 statistics contain duplicate summary key {key!r}.")
        summary_by_condition[key] = summary
    if set(summary_by_condition) != set(expected_order):
        raise RuntimeError("Fixed S2 statistics contain missing or extra summaries.")
    frozen_global_values = list(stats.get("values_used_for_plotting") or [])
    if len(frozen_global_values) != len(df):
        raise RuntimeError("Fixed S2 frozen value vector length does not match the validated row identities.")
    offset = 0
    for condition in expected_order:
        summary = summary_by_condition[condition]
        if summary.get("n") != 20 or "mean" not in summary or "sem" not in summary:
            raise RuntimeError(f"Fixed S2 statistics summary is incomplete for {condition}.")
        frozen_condition_values = list(summary.get("values_used_for_plotting") or [])
        if len(frozen_condition_values) != 20:
            raise RuntimeError(f"Fixed S2 frozen value count is not 20 for {condition}.")
        if frozen_global_values[offset : offset + 20] != frozen_condition_values:
            raise RuntimeError(f"Fixed S2 global and condition value order disagree for {condition}.")
        offset += 20
    # The adapter has already hash-validated and round-trip-parsed the CSV, and
    # emitted it byte-identically.  The build resolver subsequently parses that
    # CSV with pandas' legacy float parser, which can alter one binary ULP.  Use
    # the exact frozen JSON vector for display without tolerance or recomputation.
    df["value"] = np.asarray(frozen_global_values, dtype=float)
    return df, summary_by_condition


def _set_tight_zero_limits(ax, values: pd.Series, *, upper_headroom: float = 0.12) -> None:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return
    low = min(0.0, float(np.min(numeric)))
    high = max(0.0, float(np.max(numeric)))
    span = max(high - low, max(abs(high), abs(low), 1e-12) * 0.2)
    ax.set_ylim(low - 0.10 * span, high + float(upper_headroom) * span)


def render_s4_ping_amplitude_sweep(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _line_by_state(ax, panel_data, spec, style, plot_form="s4_ping_amplitude_sweep", placeholder="Ping amplitude sweep unavailable", show_legend=False)


def render_s4_ping_duration_sweep(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _line_by_state(ax, panel_data, spec, style, plot_form="s4_ping_duration_sweep", placeholder="Ping duration sweep unavailable", show_legend=True)


def render_s4_completion_delay_gain(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "x_value" not in df.columns:
        _placeholder(ax, spec, "Completion delay gain unavailable")
        return
    ax.paper_fig_plot_form = "s4_completion_delay_gain"
    ax.axhline(0, color="0.45", linewidth=0.65, linestyle="--")
    ax.paper_fig_has_zero_reference = True
    df = df.copy()
    df["x_value"] = pd.to_numeric(df["x_value"], errors="coerce")
    conditions = df["condition"].astype(str).drop_duplicates().tolist()
    colors = {"S_AB_minus_relevant_single": PALETTE["mechanism_teal"], "target_A": STATE_COLORS["S_A"], "target_B": STATE_COLORS["S_B"]}
    for condition in conditions:
        part = df[df["condition"].astype(str).eq(condition)].dropna(subset=["x_value"])
        grouped = part.groupby("x_value", as_index=False)["value"].agg(["mean", "sem"]).reset_index()
        if grouped.empty:
            continue
        x = grouped["x_value"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        sem = grouped["sem"].fillna(0).to_numpy(dtype=float)
        color = colors.get(condition, PALETTE["mechanism_teal"])
        label = "S_AB - single" if condition == "S_AB_minus_relevant_single" else condition.replace("_", " ")
        ax.plot(x, y, marker="o", markersize=2.7, linewidth=0.9, color=color, label=label)
        if part["seed_id"].replace("", pd.NA).dropna().nunique() > 1:
            _fill_sem_band(ax, x, y, sem, color)
    ax.set_xlabel(str(spec.get("x_axis", "Post-pair delay (ms)")))
    ax.set_ylabel(str(spec.get("y_axis", "Completion gain (%)")))
    if len(conditions) > 1:
        ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="best", handlelength=1.0)
    _autoscale_y(ax, df["value"], include_zero=True)
    _tidy(ax, st)


def _layerwise_bar(ax, panel_data: pd.DataFrame | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None, *, plot_form: str, placeholder: str) -> None:
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, placeholder)
        return
    ax.paper_fig_plot_form = plot_form
    order = [layer for layer in LAYER_ORDER if layer in set(df.get("layer", df["condition"]).astype(str))]
    if not order:
        order = [layer for layer in LAYER_ORDER if layer in set(df["condition"].astype(str))]
    if not order:
        _placeholder(ax, spec, "Layer labels unavailable")
        return
    plot_df = df.copy()
    if "layer" in plot_df.columns:
        plot_df["condition"] = plot_df["layer"].astype(str)
    _bar_summary(ax, plot_df, "condition", order, colors=[PALETTE["primary_navy"], PALETTE["comparison_coral"], PALETTE["mechanism_teal"]], st=st, alpha=0.82)
    ax.axhline(0, color="0.45", linewidth=0.65, linestyle="--")
    ax.set_xticks(np.arange(len(order)), [LAYER_LABELS.get(layer, layer) for layer in order])
    ax.set_ylabel(str(spec.get("y_axis", "Score")))
    ax.set_xlabel(str(spec.get("x_axis", "Layer")))
    ax.paper_fig_raw_points = False
    _autoscale_y(ax, plot_df["value"], include_zero=True)
    _tidy(ax, st)


def _line_by_state(ax, panel_data: pd.DataFrame | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None, *, plot_form: str, placeholder: str, show_legend: bool) -> None:
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "x_value" not in df.columns:
        _placeholder(ax, spec, placeholder)
        return
    ax.paper_fig_plot_form = plot_form
    ax.paper_fig_x_metric = str(spec.get("sweep_parameter", ""))
    df = df.copy()
    df["x_value"] = pd.to_numeric(df["x_value"], errors="coerce")
    for condition in [c for c in STATE_ORDER if c in set(df["condition"].astype(str))]:
        part = df[df["condition"].astype(str).eq(condition)].dropna(subset=["x_value"])
        grouped = part.groupby("x_value", as_index=False)["value"].agg(["mean", "sem"]).reset_index()
        if grouped.empty:
            continue
        x = grouped["x_value"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        sem = grouped["sem"].fillna(0).to_numpy(dtype=float)
        color = STATE_COLORS.get(condition, "0.3")
        ax.plot(x, y, marker="o", markersize=2.6, linewidth=0.88, color=color, label=STATE_LABELS.get(condition, condition).replace("\n", " "))
        if part["seed_id"].replace("", pd.NA).dropna().nunique() > 1:
            _fill_sem_band(ax, x, y, sem, color)
    ax.set_ylim(0, 100)
    ax.set_xlabel(str(spec.get("x_axis", "")))
    ax.set_ylabel(str(spec.get("y_axis", "Pair-member readout (%)")))
    if show_legend:
        legend = ax.legend(frameon=False, fontsize=st["legend_fontsize"], ncol=2, loc="best", handlelength=1.0, columnspacing=0.7)
        ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
        ax.paper_fig_legend_ncols = 2
    _tidy(ax, st)
