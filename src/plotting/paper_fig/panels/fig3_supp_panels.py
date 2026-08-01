from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.plotting.common.colors import NATURE_COMPATIBLE_PALETTE as PALETTE, get_plot_color, get_plot_cmap
from scipy import stats as scipy_stats

from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder
from src.plotting.paper_fig.panels.fig3_panels import PEAK, RANDOM, VALLEY, _infer_seq_len, _region_ping_categories, render_fig3_morphology_serial_profile


MAIN = get_plot_color("dynamic")
GRID = "0.88"
_S3A_FROZEN_P_DISPLAY = r"$\mathrm{P} = 1.12 \times 10^{−16}$"


def render_s3_frozen_fit_comparison(ax, panel_data, stats, spec, style=None):
    """Render S3A from persisted observations and frozen summaries only."""
    _ = style
    summary = _s3_frozen_summary_map(stats, spec)
    df = _clean(panel_data)
    required = {"linear_sse", "saturating_sse"}
    if df.empty or set(df["metric"].astype(str)) != required:
        raise ValueError("supp_fig_s3A: frozen paired observations are incomplete")
    linear = df[df["metric"].astype(str).eq("linear_sse")].set_index("network_id")["value"]
    saturating = df[df["metric"].astype(str).eq("saturating_sse")].set_index("network_id")["value"]
    paired = pd.concat([linear.rename("Linear"), saturating.rename("Saturating")], axis=1).dropna()
    if len(paired) != int(stats["n_networks"]):
        raise ValueError("supp_fig_s3A: frozen observations do not form 20 exact pairs")
    for _, row in paired.iterrows():
        ax.plot([0, 1], row.to_numpy(dtype=float), color="0.72", linewidth=1.15, alpha=0.55, zorder=1)
    colors = [get_plot_color("other_residual"), get_plot_color("sequence_state")]
    for x, metric, color in zip((0, 1), ("linear_sse", "saturating_sse"), colors):
        values = pd.to_numeric(df.loc[df["metric"].astype(str).eq(metric), "value"], errors="raise").to_numpy(dtype=float)
        jitter = np.linspace(-0.055, 0.055, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=10, color=color, alpha=0.48, linewidth=0, zorder=2)
        frozen = summary[(metric,)]
        center = frozen["mean"]
        yerr = np.asarray([[center - frozen["ci95_low"]], [frozen["ci95_high"] - center]])
        ax.errorbar(
            x,
            center,
            yerr=yerr,
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.15,
            markersize=5.0,
            capsize=2.4,
            linewidth=1.15,
            zorder=3,
        )
    delta = summary[("linear_minus_saturating_sse",)]
    ax.text(
        0.0,
        1.035,
        "Linear − saturating SSE = "
        f"{delta['mean']:.3f}; n = {int(delta['n_networks'])}\n"
        f"95% CI [{delta['ci95_low']:.3f}, {delta['ci95_high']:.3f}]; {_S3A_FROZEN_P_DISPLAY}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.6,
        linespacing=1.05,
        color="0.18",
    )
    ax.set_xticks([0, 1], ["Linear", "Saturating"])
    ax.set_xlabel("Fit model")
    ax.set_ylabel("Fit SSE")
    ax.set_ylim(0.0, 2.45)
    _finish_s3_axes(ax)
    ax.paper_fig_plot_form = "s3_frozen_fit_comparison"
    ax.paper_fig_raw_points = True
    ax.paper_fig_no_recompute = True


def render_s3_frozen_peak_valley_null(ax, panel_data, stats, spec, style=None):
    """Render S3B on two explicit scales from three frozen summary rows."""
    _ = panel_data, style
    summary = _s3_frozen_summary_map(stats, spec)
    ax.set_axis_off()
    statistic_ax = ax.inset_axes([0.0, 0.0, 34.0 / 60.0, 1.0])
    fraction_ax = ax.inset_axes([40.0 / 60.0, 0.0, 20.0 / 60.0, 1.0])
    statistic_metrics = ("observed_peak_valley_delta", "null_peak_valley_delta_p95")
    statistic_colors = [get_plot_color("sequence_state"), get_plot_color("random_control")]
    for x, metric, color in zip((0, 1), statistic_metrics, statistic_colors):
        frozen = summary[(metric,)]
        center = frozen["mean"]
        yerr = np.asarray([[center - frozen["ci95_low"]], [frozen["ci95_high"] - center]])
        statistic_ax.bar(x, center, width=0.62, color=color, edgecolor="white", linewidth=1.15)
        statistic_ax.errorbar(x, center, yerr=yerr, fmt="none", ecolor="0.15", capsize=2.2, linewidth=1.15)
    statistic_ax.set_xticks([0, 1], ["Observed", "Null p95"], rotation=18, ha="right")
    statistic_ax.set_ylabel("Peak-valley statistic")
    statistic_ax.set_ylim(0.0, 0.30)
    statistic_ax.set_title("Structure vs null", fontsize=6.8, pad=2.0)
    structured = summary[("is_structured",)]
    fraction_ax.bar(0, structured["mean"], width=0.58, color=get_plot_color("mechanism_teal"), edgecolor="white", linewidth=1.15)
    fraction_ax.errorbar(0, structured["mean"], yerr=structured["sem"], fmt="none", ecolor="0.15", capsize=2.2, linewidth=1.15)
    fraction_ax.set_xticks([0], ["Structured"], rotation=18, ha="right")
    fraction_ax.set_ylabel("Structured fraction")
    fraction_ax.set_ylim(0.0, 1.05)
    fraction_ax.set_yticks([0.0, 0.5, 1.0])
    fraction_ax.set_title("Sequence fraction", fontsize=6.8, pad=2.0)
    _finish_s3_axes(statistic_ax)
    _finish_s3_axes(fraction_ax)
    ax.paper_fig_plot_form = "s3_frozen_peak_valley_null_two_scale"
    ax.paper_fig_no_recompute = True


def render_s3_frozen_morphology_serial_profile(ax, panel_data, stats, spec, style=None):
    """Render S3C in the declared serial-position order from frozen rows."""
    _ = panel_data, style
    summary = _s3_frozen_summary_map(stats, spec)
    positions = list(range(1, 11))
    centers = [summary[(position,)]["mean"] for position in positions]
    sems = [summary[(position,)]["sem"] for position in positions]
    ax.errorbar(
        positions,
        centers,
        yerr=sems,
        color=get_plot_color("sequence_state"),
        marker="o",
        markersize=3.8,
        markeredgecolor="white",
        markeredgewidth=1.15,
        linewidth=1.3,
        elinewidth=1.15,
        capsize=2.0,
    )
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xlabel("Serial position")
    ax.set_ylabel("Layer 1 STSP support mass\n(proportion)")
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(0.0, 0.35)
    _finish_s3_axes(ax)
    ax.paper_fig_plot_form = "s3_frozen_morphology_serial_profile"
    ax.paper_fig_no_recompute = True


def render_s3_frozen_ping_recency(ax, panel_data, stats, spec, style=None):
    """Render S3D from four frozen readout-class summaries."""
    _ = panel_data, style
    summary = _s3_frozen_summary_map(stats, spec)
    classes = ["latest", "recent", "earlier", "silent"]
    labels = ["Latest", "Recent", "Earlier", "Silent"]
    colors = [
        get_plot_color("recent_input"),
        get_plot_color("middle_input"),
        get_plot_color("old_input"),
        get_plot_color("silent_state"),
    ]
    centers = [summary[(name, "readout_mass", name)]["mean"] for name in classes]
    sems = [summary[(name, "readout_mass", name)]["sem"] for name in classes]
    xs = np.arange(4)
    ax.bar(xs, centers, yerr=sems, width=0.64, color=colors, edgecolor="white", linewidth=1.15, error_kw={"elinewidth": 1.15, "capsize": 2.2})
    ax.set_xticks(xs, labels, rotation=15, ha="right")
    ax.set_xlabel("Readout class")
    ax.set_ylabel("Readout mass (proportion)")
    ax.set_ylim(0.0, 0.60)
    _finish_s3_axes(ax)
    ax.paper_fig_plot_form = "s3_frozen_ping_recency"
    ax.paper_fig_no_recompute = True


def render_s3_frozen_weak_probe_recency(ax, panel_data, stats, spec, style=None):
    """Render S3E from four frozen target-position summaries."""
    _ = panel_data, style
    summary = _s3_frozen_summary_map(stats, spec)
    bins = ["early", "middle", "recent", "latest"]
    labels = ["Early", "Middle", "Recent", "Latest"]
    colors = [
        get_plot_color("old_input"),
        get_plot_color("middle_input"),
        get_plot_color("middle_input"),
        get_plot_color("recent_input"),
    ]
    centers = [summary[(name, "target_recovery_gain", name)]["mean"] for name in bins]
    sems = [summary[(name, "target_recovery_gain", name)]["sem"] for name in bins]
    xs = np.arange(4)
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.15, zorder=0)
    ax.bar(xs, centers, yerr=sems, width=0.64, color=colors, edgecolor="white", linewidth=1.15, error_kw={"elinewidth": 1.15, "capsize": 2.2})
    ax.set_xticks(xs, labels, rotation=15, ha="right")
    ax.set_xlabel("Target-position bin")
    ax.set_ylabel("Recovery gain\n(percentage points)")
    ax.set_ylim(-4.0, 31.0)
    _finish_s3_axes(ax)
    ax.paper_fig_plot_form = "s3_frozen_weak_probe_recency"
    ax.paper_fig_no_recompute = True


def render_s3_frozen_boundary_pair(ax, panel_data, stats, spec, style=None):
    """Render S3F by placing one frozen row in each declared matrix cell."""
    _ = panel_data, style
    summary = _s3_frozen_summary_map(stats, spec)
    sequence_order = [3, 5, 7, 10]
    delay_order = [100, 200, 400, 800]
    metrics = ["N_eff_fraction", "rescued_fraction"]
    titles = ["Morphology", "Rescue"]
    ax.set_axis_off()
    heatmap_axes = [
        ax.inset_axes([0.0, 7.833333 / 31.5, 25.0 / 60.0, 23.0 / 31.5]),
        ax.inset_axes([29.0 / 60.0, 7.833333 / 31.5, 25.0 / 60.0, 23.0 / 31.5]),
    ]
    for index, (heatmap_ax, metric, title) in enumerate(zip(heatmap_axes, metrics, titles)):
        matrix = np.zeros((4, 4), dtype=float)
        for row_index, delay in enumerate(delay_order):
            for column_index, sequence_length in enumerate(sequence_order):
                matrix[row_index, column_index] = summary[(delay, metric, sequence_length)]["mean"]
        image = heatmap_ax.pcolormesh(
            np.arange(5, dtype=float) - 0.5,
            np.arange(5, dtype=float) - 0.5,
            matrix,
            cmap=get_plot_cmap("stsp_support"),
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            edgecolors="none",
        )
        heatmap_ax.set_xlim(-0.5, 3.5)
        heatmap_ax.set_ylim(-0.5, 3.5)
        heatmap_ax.set_xticks(np.arange(4), [str(value) for value in sequence_order])
        heatmap_ax.set_yticks(np.arange(4), [str(value) for value in delay_order] if index == 0 else [])
        heatmap_ax.set_xlabel("Sequence length K")
        heatmap_ax.set_ylabel("Delay (ms)" if index == 0 else "")
        heatmap_ax.set_title(title, fontsize=6.8, pad=2.0)
        _finish_s3_heatmap_axes(heatmap_ax)
    colorbar_ax = ax.inset_axes([57.0 / 60.0, 7.833333 / 31.5, 2.5 / 60.0, 23.0 / 31.5])
    color_steps = np.linspace(0.0, 1.0, 65)
    color_values = 0.5 * (color_steps[:-1] + color_steps[1:])
    colorbar_ax.pcolormesh(
        [0.0, 1.0],
        color_steps,
        color_values[:, np.newaxis],
        cmap=get_plot_cmap("stsp_support"),
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        edgecolors="none",
    )
    colorbar_ax.set_xlim(0.0, 1.0)
    colorbar_ax.set_ylim(0.0, 1.0)
    colorbar_ax.set_xticks([])
    colorbar_ax.set_yticks([0.0, 0.5, 1.0])
    colorbar_ax.yaxis.tick_right()
    colorbar_ax.yaxis.set_label_position("right")
    colorbar_ax.set_ylabel("Fraction", fontsize=6.4, labelpad=1.5)
    colorbar_ax.tick_params(axis="y", labelsize=5.8, length=2.0, width=1.15, pad=1.0)
    for spine in colorbar_ax.spines.values():
        spine.set_linewidth(1.15)
    ax.paper_fig_plot_form = "s3_frozen_boundary_pair"
    ax.paper_fig_has_colorbar = True
    ax.paper_fig_no_recompute = True


def _s3_frozen_summary_map(stats, spec):
    if not isinstance(stats, Mapping) or stats.get("plot_only_no_recompute") is not True:
        raise ValueError("supp_fig_s3: renderer requires a validated frozen statistic payload")
    contract = spec.get("frozen_statistics")
    if not isinstance(contract, Mapping):
        raise ValueError("supp_fig_s3: frozen_statistics contract is missing")
    if stats.get("frozen_source_sha256") != contract.get("sha256"):
        raise ValueError("supp_fig_s3: renderer/source hash contract mismatch")
    fields = tuple(map(str, contract.get("identity_fields") or ()))
    rows = stats.get("network_summaries")
    if not fields or not isinstance(rows, list) or len(rows) != int(contract.get("rows", -1)):
        raise ValueError("supp_fig_s3: frozen summary schema mismatch")
    summary = {tuple(row[field] for field in fields): row for row in rows}
    if len(summary) != len(rows):
        raise ValueError("supp_fig_s3: frozen summary identities are duplicated")
    expected_order = [tuple(item) for item in contract.get("network_summary_identity_order") or ()]
    observed_order = [tuple(row[field] for field in fields) for row in rows]
    if observed_order != expected_order:
        raise ValueError("supp_fig_s3: frozen summary identity/order changed after adapter validation")
    return summary


def _finish_s3_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.15)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=6.0, pad=1.1, length=2.2, width=1.15)
    ax.xaxis.label.set_size(6.8)
    ax.yaxis.label.set_size(6.8)
    ax.xaxis.labelpad = 1.4
    ax.yaxis.labelpad = 1.6


def _finish_s3_heatmap_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
    ax.tick_params(axis="both", labelsize=5.8, pad=1.0, length=2.0, width=1.15)
    ax.xaxis.label.set_size(6.2)
    ax.yaxis.label.set_size(6.2)
    ax.xaxis.labelpad = 1.2
    ax.yaxis.labelpad = 1.2


def render_part2_fit_comparison(ax, panel_data, stats, spec, style=None):
    _ = style
    df = _clean(panel_data)
    linear = df[df["metric"].astype(str).eq("linear_sse")].set_index("network_id")["value"]
    saturating = df[df["metric"].astype(str).eq("saturating_sse")].set_index("network_id")["value"]
    paired = pd.concat([linear.rename("Linear"), saturating.rename("Saturating")], axis=1).dropna()
    if paired.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    for _, row in paired.iterrows():
        ax.plot([0, 1], row.to_numpy(dtype=float), color="0.75", linewidth=0.45, alpha=0.65, zorder=1)
    colors = ["0.55", MAIN]
    for x, column, color in zip([0, 1], paired.columns, colors):
        values = paired[column].to_numpy(dtype=float)
        jitter = np.linspace(-0.045, 0.045, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=8, color=color, alpha=0.45, linewidth=0, zorder=2)
        mean, half = _part2_ci(values)
        ax.errorbar(x, mean, yerr=half, fmt="o", color=color, markeredgecolor="white", markersize=4.2, capsize=2.0, linewidth=0.8, zorder=3)
    delta = pd.to_numeric(df.loc[df["metric"].astype(str).eq("linear_minus_saturating_sse"), "value"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(delta):
        mean, half = _part2_ci(delta)
        p_value = float(scipy_stats.ttest_1samp(delta, 0.0).pvalue) if len(delta) > 1 else np.nan
        ax.text(0.03, 0.97, f"Delta SSE {mean:.2f} [{mean-half:.2f}, {mean+half:.2f}]\np={p_value:.2g}, n={len(delta)}", transform=ax.transAxes, ha="left", va="top", fontsize=4.7, color="0.22")
    ax.set_xticks([0, 1], ["Linear", "Saturating"])
    ax.set_ylabel("Fit SSE")
    _finish_axes(ax, spec)
    ax.paper_fig_plot_form = "part2_fit_comparison"
    ax.paper_fig_raw_points = True
    ax.paper_fig_y_metric = "fit_sse"


def render_part2_boundary_pair(ax, panel_data, stats, spec, style=None):
    _ = stats, style
    df = _clean(panel_data)
    metrics = list(map(str, spec.get("metrics") or ["N_eff_fraction", "rescued_fraction"]))
    if df.empty or not {"metric", "seq_len", "delay_ms"}.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    ax.set_axis_off()
    images = []
    for index, (bounds, metric, title) in enumerate(zip(([0.03, 0.16, 0.42, 0.74], [0.55, 0.16, 0.42, 0.74]), metrics, ["Morphology", "Rescue"])):
        inset = ax.inset_axes(bounds)
        image = _part2_heatmap(inset, df[df["metric"].astype(str).eq(metric)], title, show_y_axis=index == 0)
        if image is not None:
            images.append(image)
    if not images:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    cax = ax.inset_axes([0.35, 0.035, 0.30, 0.045])
    cbar = ax.figure.colorbar(images[0], cax=cax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=4.0, length=1.0, pad=0.5)
    cbar.set_label("Fraction", fontsize=4.5, labelpad=0.5)
    ax.paper_fig_plot_form = "part2_boundary_pair"
    ax.paper_fig_has_colorbar = True
    ax.paper_fig_y_metric = ";".join(metrics)


def _part2_heatmap(ax, frame, title, *, show_y_axis=True):
    use = frame.copy()
    use["seq_len"] = pd.to_numeric(use["seq_len"], errors="coerce")
    use["delay_ms"] = pd.to_numeric(use["delay_ms"], errors="coerce")
    use = use.dropna(subset=["seq_len", "delay_ms", "value"])
    if use.empty:
        ax.set_axis_off()
        return None
    xs = np.sort(use["seq_len"].unique())
    ys = np.sort(use["delay_ms"].unique())
    matrix = np.full((len(ys), len(xs)), np.nan)
    for yi, delay in enumerate(ys):
        for xi, seq_len in enumerate(xs):
            values = pd.to_numeric(use.loc[use["seq_len"].eq(seq_len) & use["delay_ms"].eq(delay), "value"], errors="coerce").dropna()
            if not values.empty:
                matrix[yi, xi] = float(values.mean())
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=get_plot_cmap("stsp_support"), vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(np.arange(len(xs)), [str(int(value)) for value in xs])
    ax.set_yticks(np.arange(len(ys)), [str(int(value)) for value in ys] if show_y_axis else [])
    ax.set_xlabel("K", fontsize=4.7, labelpad=0.5)
    ax.set_ylabel("Delay" if show_y_axis else "", fontsize=4.7, labelpad=0.5)
    ax.set_title(title, fontsize=5.0, pad=1.0)
    ax.tick_params(labelsize=4.0, length=1.2, pad=0.5)
    return image


def _part2_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, 0.0
    sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, float(scipy_stats.t.ppf(0.975, len(values) - 1) * sem)


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
    colors = {"latest": get_plot_color("recent_input"), "recent": get_plot_color("middle_input"), "earlier": get_plot_color("old_input"), "silent": get_plot_color("silent_state")}
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
        ax.bar(xs + (idx - 1) * width, np.asarray(vals, dtype=float) / norm, width=width, label=metric.replace("_", " "), color=[PALETTE["primary_pale"], "0.68", PALETTE["comparison_salmon"]][idx])
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
    colors = {"cue_only": get_plot_color("cue_only"), "single_item_memory": get_plot_color("single_item_memory"), "sequence_state": MAIN}
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
        ax.bar(xs + (idx - 0.5) * width, np.asarray(vals, dtype=float) / norm, width=width, label=metric.replace("_mean", "").replace("_", " "), color=[PALETTE["primary_cyan"], PALETTE["comparison_salmon"]][idx])
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
        colors = {"latest": get_plot_color("recent_input"), "recent": get_plot_color("middle_input"), "earlier": get_plot_color("old_input"), "other": get_plot_color("other_residual"), "silent": get_plot_color("silent_state")}
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
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=5.0, pad=0.8, length=1.8, width=0.5)
    ax.xaxis.label.set_size(5.7)
    ax.yaxis.label.set_size(5.7)
    ax.xaxis.labelpad = 0.8
    ax.yaxis.labelpad = 0.8
