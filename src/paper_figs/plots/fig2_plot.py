from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.paper_figs.plots.common import load_npz, read_csv_validated, require_path, resolve_figure_input_dir
from src.paper_figs.plots.style import (
    ANNOTATION_SIZE,
    COLOR_DARK_GRAY,
    COLOR_DYNAMIC,
    COLOR_LIGHT_GRAY,
    COLOR_OVERLAP,
    COLOR_PROBE_ONLY,
    COLOR_SAMPLE_ONLY,
    COLOR_STATIC,
    COLOR_TEXT,
    DATA_LINEWIDTH,
    REF_LINEWIDTH,
    apply_paper_style,
    save_figure_outputs,
    style_axes,
)

CHANCE_LEVEL = 0.10
PANEL_E_BIAS_COLUMNS = [
    "n_total",
    "n_error",
    "error_rate",
    "bias_original_sample",
    "bias_donor_shifted_memory",
    "bias_silent",
    "bias_other_classes",
    "condition",
]
PANEL_E_CONDITION_ORDER = ["A_dynamic_base", "D_trial_shuffle_ux", "E_static_frozen"]
PANEL_E_DISPLAY_NAMES = {
    "A_dynamic_base": "Dynamic baseline",
    "D_trial_shuffle_ux": "trial-shuffle u/x",
    "E_static_frozen": "static frozen",
}
PANEL_E_START_ANGLE = 315.0


def load_fig2_bundle(root: str | Path) -> dict[str, object]:
    root_path = Path(root)
    summary = json.loads(require_path(root_path / "summary.json").read_text(encoding="utf-8"))
    panel_e_bias_path = root_path / "data" / "metrics_error_bias.csv"
    if not panel_e_bias_path.exists():
        raise FileNotFoundError(f"Panel E requires metrics_error_bias.csv at: {panel_e_bias_path}")
    return {
        "summary": summary,
        "panel_b_memory": read_csv_validated(
            root_path / "data" / "panel_b_memory_effect_vs_delay.csv",
            ["delay_ms", "acc_static", "acc_dynamic", "acc_drop"],
        ),
        "panel_b_fit": read_csv_validated(
            root_path / "data" / "panel_b_fit_summary.csv",
            ["metric_name", "tau_ms", "offset", "fit_success"],
        ),
        "panel_c_raster": read_csv_validated(
            root_path / "data" / "panel_c_raster_points.csv",
            ["trial_id", "layer", "t_step", "time_ms", "neuron_index", "phase"],
        ),
        "panel_c_rate": read_csv_validated(
            root_path / "data" / "panel_c_population_rate.csv",
            ["trial_id", "t_step", "time_ms", "phase", "population_rate_smoothed"],
        ),
        "panel_d_decode": read_csv_validated(
            root_path / "data" / "panel_d_decode_metrics.csv",
            ["layer", "delay_ms", "acc", "acc_ci_low", "acc_ci_high"],
        ),
        "panel_e_condition": read_csv_validated(
            root_path / "data" / "panel_e_condition_summary.csv",
            ["condition", "acc_probe", "abs_rate_pred_original_sample", "abs_rate_pred_change_under_bmap"],
        ),
        "panel_e_collapse": read_csv_validated(
            root_path / "data" / "panel_e_collapse_summary.csv",
            ["substrate", "collapse_toward_static_improvement_pp"],
        ),
        "panel_e_bootstrap": read_csv_validated(
            root_path / "data" / "panel_e_bootstrap_tests.csv",
            ["substrate", "test_name", "obs_diff_rate"],
        ),
        "panel_e_bias": read_csv_validated(panel_e_bias_path, PANEL_E_BIAS_COLUMNS),
        "panel_c_arrays": load_npz(root_path / "arrays" / "panel_c_representative_trial.npz"),
    }


def _phase_bounds(rate_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for phase, phase_df in rate_df.groupby("phase", sort=False):
        bounds[str(phase)] = (float(phase_df["time_ms"].min()), float(phase_df["time_ms"].max()))
    return bounds


def _normalize_phase_time(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    result = df.copy()
    result["phase_time_ms"] = result.apply(lambda row: float(row["time_ms"]) - bounds[str(row["phase"])][0], axis=1)
    return result


def _infer_phase_slices(df_population_rate: pd.DataFrame) -> dict[str, tuple[int, int]]:
    phase_slices: dict[str, tuple[int, int]] = {}
    for phase_name in ["sample", "delay", "probe"]:
        sub = df_population_rate[df_population_rate["phase"] == phase_name].copy()
        if sub.empty:
            continue
        t_steps = sub["t_step"].to_numpy(dtype=int)
        phase_slices[phase_name] = (int(t_steps.min()), int(t_steps.max()) + 1)
    return phase_slices


def _infer_layer_neuron_counts(df_raster_points: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for layer_name, sub in df_raster_points.groupby("layer", sort=False):
        counts[str(layer_name)] = int(sub["neuron_index"].to_numpy(dtype=int).max()) + 1
    return counts


def _infer_dt_ms(df_population_rate: pd.DataFrame) -> float:
    ordered = df_population_rate.sort_values("t_step", kind="stable").reset_index(drop=True)
    if len(ordered) < 2:
        return 1.0
    time_ms = ordered["time_ms"].to_numpy(dtype=float)
    t_steps = ordered["t_step"].to_numpy(dtype=float)
    delta_t = t_steps[1:] - t_steps[:-1]
    delta_ms = time_ms[1:] - time_ms[:-1]
    mask = delta_t > 0
    if not mask.any():
        return 1.0
    return float((delta_ms[mask] / delta_t[mask]).mean())


def _format_compact_count(value: float, _pos: float) -> str:
    value = float(value)
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:.1f}k"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def _format_small_limit(value: float, _pos: float) -> str:
    value = float(value)
    if abs(value) < 1e-12:
        return "0"
    if abs(value) < 0.1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.2g}"


def _add_phase_guides(ax: plt.Axes, phase_slices: dict[str, tuple[int, int]], dt_ms: float) -> None:
    phase_colors = {"sample": COLOR_DYNAMIC, "delay": COLOR_STATIC, "probe": COLOR_PROBE_ONLY}
    phase_alpha = {"sample": 0.10, "delay": 0.11, "probe": 0.10}
    for phase in ["sample", "delay", "probe"]:
        start, end = phase_slices[phase]
        ax.axvspan(start * dt_ms, end * dt_ms, color=phase_colors[phase], alpha=phase_alpha[phase], linewidth=0)
        ax.axvline(start * dt_ms, color=phase_colors[phase], linewidth=1.1, linestyle="--", alpha=0.85)


def _plot_representative_population_rate(ax: plt.Axes, df_population_rate: pd.DataFrame, phase_slices: dict[str, tuple[int, int]], dt_ms: float) -> None:
    sub = df_population_rate.sort_values("t_step", kind="stable").reset_index(drop=True)
    _add_phase_guides(ax, phase_slices=phase_slices, dt_ms=dt_ms)
    x = sub["time_ms"].to_numpy(dtype=np.float64, copy=False)
    y = sub["population_rate_smoothed"].to_numpy(dtype=np.float64, copy=False)
    ax.plot(x, y, color=COLOR_DYNAMIC, linewidth=1.6)
    ax.fill_between(x, 0.0, y, color=COLOR_DYNAMIC, alpha=0.14, linewidth=0)
    ax.set_ylabel("")
    ax.text(
        -0.02,
        0.5,
        "Pooled\nrate",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=ANNOTATION_SIZE + 1,
        fontweight="semibold",
        color="#222222",
    )
    ax.set_xlabel("Time (ms)")
    ymax = max(float(np.max(y)) if len(y) > 0 else 0.0, 1e-6)
    ylim_top = ymax * 1.25
    x_max = float(np.max(x)) if len(x) > 0 else 1.0
    left_margin = max(5.0 * dt_ms, 0.01 * x_max)
    ax.set_xlim(-left_margin, x_max)
    ax.set_ylim(0.0, ylim_top)
    ax.set_yticks([0.0, ylim_top], labels=["", _format_small_limit(ylim_top, 0.0)])
    ax.tick_params(axis="y", pad=12, labelsize=10)
    ax.tick_params(axis="x", pad=6)
    xticks = ax.get_xticks()
    xlabels = ["0" if abs(float(tick)) < 1e-9 else f"{tick:g}" for tick in xticks]
    ax.set_xticks(xticks, xlabels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_representative_raster(
    axes: list[plt.Axes],
    df_points: pd.DataFrame,
    phase_slices: dict[str, tuple[int, int]],
    layer_neuron_counts: dict[str, int],
    dt_ms: float,
) -> None:
    layers = ["layer1", "layer2"]
    total_steps = int(phase_slices["probe"][1])
    total_ms = total_steps * dt_ms
    for idx, (ax, layer_name) in enumerate(zip(axes, layers)):
        _add_phase_guides(ax, phase_slices=phase_slices, dt_ms=dt_ms)
        sub = df_points[df_points["layer"] == layer_name].copy()
        if not sub.empty:
            ax.scatter(
                sub["t_step"].to_numpy(dtype=np.float64) * dt_ms,
                sub["neuron_index"].to_numpy(dtype=np.float64),
                s=2.2,
                c="black",
                marker=".",
                linewidths=0,
                alpha=0.9,
            )
        max_neuron = float(max(1, int(layer_neuron_counts.get(layer_name, 1))))
        ax.set_xlim(0.0, total_ms)
        ax.set_ylim(0.0, max_neuron)
        ax.set_ylabel("")
        ax.text(
            -0.02,
            0.5,
            f"Layer {idx + 1}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=ANNOTATION_SIZE + 1,
            fontweight="semibold",
            color="#222222",
        )
        ax.set_yticks([])
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.tick_params(axis="x", labelbottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].xaxis.set_major_formatter(FuncFormatter(_format_compact_count))


def _infer_full_decay_curve(memory_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = memory_df.sort_values("delay_ms").copy()
    x = df["delay_ms"].to_numpy(dtype=float)
    y = df["acc_drop"].to_numpy(dtype=float)
    floor = max(0.0, float(np.nanmin(y)) * 0.8)
    positive = np.clip(y - floor, 1e-6, None)
    slope, intercept = np.polyfit(x, np.log(positive), 1)
    if slope >= 0:
        x_curve = np.linspace(float(x.min()), float(x.max()), 200)
        return x_curve, np.interp(x_curve, x, y)
    tau = -1.0 / slope
    x_curve = np.linspace(float(x.min()), float(x.max()), 300)
    y_curve = floor + np.exp(intercept) * np.exp(-(x_curve - float(x.min())) / tau)
    return x_curve, y_curve


def _panel_b_values(summary: dict[str, object], memory_df: pd.DataFrame, fit_df: pd.DataFrame) -> tuple[float, float]:
    peak = float(memory_df["acc_drop"].max())
    tau = float("nan")
    panel_summary = summary.get("panel_b", {}) if isinstance(summary.get("panel_b"), dict) else {}
    if panel_summary:
        peak = float(panel_summary.get("max_acc_drop_pp", peak))
        tau = float(panel_summary.get("accuracy_drop_tau_ms", np.nan))
    fit_row = fit_df.loc[fit_df["metric_name"] == "acc_drop"]
    if not fit_row.empty and (not np.isfinite(tau) or tau > 1e5):
        tau = float(fit_row["tau_ms"].iloc[0])
    return peak, tau


def _panel_e_bias_display_frame(bias_df: pd.DataFrame) -> pd.DataFrame:
    df = bias_df[bias_df["condition"].isin(PANEL_E_CONDITION_ORDER)].copy()
    if df.empty:
        raise ValueError("Panel E bias table is empty after filtering the required conditions.")
    duplicate_conditions = df["condition"][df["condition"].duplicated()].tolist()
    if duplicate_conditions:
        duplicate_text = ", ".join(map(str, sorted(set(duplicate_conditions))))
        raise ValueError(f"Panel E bias table contains duplicate rows for: {duplicate_text}")
    missing_conditions = [condition for condition in PANEL_E_CONDITION_ORDER if condition not in set(df["condition"])]
    if missing_conditions:
        raise ValueError(f"Panel E bias table is missing required conditions: {', '.join(missing_conditions)}")

    df["condition"] = pd.Categorical(df["condition"], categories=PANEL_E_CONDITION_ORDER, ordered=True)
    df = df.sort_values("condition").reset_index(drop=True)

    original = pd.to_numeric(df["bias_original_sample"], errors="coerce").to_numpy(dtype=float)
    donor = pd.to_numeric(df["bias_donor_shifted_memory"], errors="coerce").to_numpy(dtype=float)
    silent = pd.to_numeric(df["bias_silent"], errors="coerce").to_numpy(dtype=float)
    other = pd.to_numeric(df["bias_other_classes"], errors="coerce").to_numpy(dtype=float) + silent
    composition = np.column_stack([original, donor, other])
    if np.isnan(composition).any():
        raise ValueError("Panel E bias table contains NaN composition values.")
    if np.min(composition) < -1e-6:
        raise ValueError("Panel E bias table contains negative composition weights.")
    composition = np.clip(composition, 0.0, None)
    totals = composition.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("Panel E bias table contains rows with zero total composition.")
    composition = composition / totals[:, None]

    error_rate = pd.to_numeric(df["error_rate"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(error_rate).any():
        raise ValueError("Panel E bias table contains NaN error_rate values.")

    display_df = pd.DataFrame(
        {
            "condition": df["condition"].astype(str),
            "display_name": [PANEL_E_DISPLAY_NAMES[str(item)] for item in df["condition"]],
            "n_total": pd.to_numeric(df["n_total"], errors="coerce").to_numpy(dtype=float),
            "n_error": pd.to_numeric(df["n_error"], errors="coerce").to_numpy(dtype=float),
            "error_rate": error_rate,
            "original": composition[:, 0],
            "donor": composition[:, 1],
            "other": composition[:, 2],
        }
    )
    if display_df[["n_total", "n_error"]].isna().any().any():
        raise ValueError("Panel E bias table contains NaN counts.")
    return display_df


def _polar_to_cartesian(radius: float, angle_deg: float) -> tuple[float, float]:
    theta = np.deg2rad(angle_deg)
    return float(radius * np.cos(theta)), float(radius * np.sin(theta))


def _draw_error_donut(ax: plt.Axes, row: pd.Series, palette: dict[str, str], *, start_angle: float = PANEL_E_START_ANGLE) -> None:
    ax.set_aspect("equal")
    ax.set_axis_off()

    composition_radius = 1.00
    composition_width = 0.34
    center_radius = composition_radius - composition_width + 0.02
    center_text_size = 9.6

    segment_keys = ["donor", "original", "other"]
    segment_spans: dict[str, tuple[float, float]] = {}
    theta = start_angle
    for key in segment_keys:
        share = float(row[key])
        next_theta = theta + share * 360.0
        ax.add_patch(
            patches.Wedge(
                (0.0, 0.0),
                composition_radius,
                theta,
                next_theta,
                width=composition_width,
                facecolor=palette[key],
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
        )
        segment_spans[key] = (theta, next_theta)
        theta = next_theta

    ax.add_patch(
        patches.Circle(
            (0.0, 0.0),
            radius=center_radius,
            facecolor="white",
            edgecolor="none",
            zorder=4,
        )
    )

    ax.text(
        0.0,
        0.12,
        "error rate",
        ha="center",
        va="center",
        fontsize=center_text_size,
        fontweight="semibold",
        color=COLOR_DARK_GRAY,
        zorder=5,
    )
    ax.text(
        0.0,
        -0.12,
        f"{float(row['error_rate']):.1f}%",
        ha="center",
        va="center",
        fontsize=center_text_size,
        fontweight="bold",
        color=COLOR_TEXT,
        zorder=5,
    )
    ax.text(
        0.0,
        1.06,
        str(row["display_name"]),
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_SIZE + 1.0,
        fontweight="semibold",
        color=COLOR_TEXT if str(row["condition"]) != "E_static_frozen" else COLOR_DARK_GRAY,
        clip_on=False,
    )

    for key in ["original", "donor", "other"]:
        theta_start, theta_end = segment_spans[key]
        theta_mid = 0.5 * (theta_start + theta_end)
        text_x, text_y = _polar_to_cartesian(composition_radius + 0.09, theta_mid)
        share = float(row[key])
        ax.text(
            text_x,
            text_y,
            f"{100.0 * share:.1f}%",
            ha="left" if text_x >= 0.0 else "right",
            va="center",
            fontsize=ANNOTATION_SIZE + (0.15 if share >= 0.18 else -0.15),
            fontweight="semibold",
            color=COLOR_TEXT,
            clip_on=False,
            zorder=5,
        )

    ax.set_xlim(-1.18, 1.50)
    ax.set_ylim(-1.22, 1.30)


def _panel_e_palette() -> dict[str, str]:
    return {
        "original": COLOR_DYNAMIC,
        "donor": COLOR_OVERLAP,
        "other": "#B7C4D1",
    }


def _add_panel_e_legend(ax: plt.Axes, palette: dict[str, str], *, y_anchor: float = 1.12) -> None:
    legend_handles = [
        patches.Patch(facecolor=palette["original"], edgecolor="none", label="original"),
        patches.Patch(facecolor=palette["donor"], edgecolor="none", label="donor"),
        patches.Patch(facecolor=palette["other"], edgecolor="none", label="other"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.1, y_anchor),
        bbox_transform=ax.transData,
        ncol=3,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0.0,
    )


def draw_panel_e_single_donut(ax: plt.Axes, row: pd.Series, *, include_legend: bool = False) -> None:
    palette = _panel_e_palette()
    _draw_error_donut(ax, row, palette)
    if include_legend:
        _add_panel_e_legend(ax, palette, y_anchor=1.20)


def draw_panel_e_error_donuts(fig: plt.Figure, spec, bias_df: pd.DataFrame) -> tuple[plt.Axes, list[plt.Axes]]:
    container_ax = fig.add_subplot(spec)
    container_ax.set_axis_off()
    container_ax.patch.set_alpha(0.0)

    panel_grid = spec.subgridspec(3, 1, hspace=0.52)
    display_df = _panel_e_bias_display_frame(bias_df)

    axes: list[plt.Axes] = []
    for idx, (_, row) in enumerate(display_df.iterrows()):
        ax = fig.add_subplot(panel_grid[idx, 0])
        draw_panel_e_single_donut(ax, row, include_legend=(idx == 0))
        axes.append(ax)

    return container_ax, axes


def draw_panel_a_task_schematic(ax: plt.Axes) -> None:
    ax.set_axis_off()
    y0 = 0.5
    track_x0, track_x1 = 0.08, 0.93
    ax.plot([track_x0, track_x1], [y0, y0], color=COLOR_DARK_GRAY, linewidth=1.1, transform=ax.transAxes, clip_on=False)
    blocks = [
        ("sample", 0.12, 0.16, COLOR_SAMPLE_ONLY, "200 ms"),
        ("delay", 0.38, 0.28, COLOR_LIGHT_GRAY, "delay"),
        ("probe", 0.76, 0.12, COLOR_PROBE_ONLY, "100 ms"),
    ]
    for label, x0, width, color, duration in blocks:
        rect = patches.FancyBboxPatch(
            (x0, y0 - 0.09),
            width,
            0.18,
            boxstyle="round,pad=0.01,rounding_size=0.025",
            linewidth=0.8,
            edgecolor=COLOR_DARK_GRAY,
            facecolor=color if label != "delay" else "white",
            alpha=0.16 if label != "delay" else 1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x0 + width / 2, y0 + 0.01, label, ha="center", va="center", transform=ax.transAxes)
        ax.text(x0 + width / 2, y0 - 0.16, duration, ha="center", va="center", transform=ax.transAxes, fontsize=ANNOTATION_SIZE)
    for start, end in ((0.28, 0.38), (0.66, 0.76), (0.88, 0.93)):
        ax.annotate(
            "",
            xy=(end, y0),
            xytext=(start, y0),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_DARK_GRAY},
        )


def draw_panel_b_peak_decay(ax: plt.Axes, summary: dict[str, object], memory_df: pd.DataFrame, fit_df: pd.DataFrame) -> None:
    style_axes(ax)
    df = memory_df.sort_values("delay_ms").copy()
    x = df["delay_ms"].to_numpy(dtype=float)
    y = df["acc_drop"].to_numpy(dtype=float)
    x_fit, y_fit = _infer_full_decay_curve(df)
    peak_idx = int(np.nanargmax(y))
    peak_x = float(x[peak_idx])
    peak_y = float(y[peak_idx])
    peak_value, tau_value = _panel_b_values(summary, df, fit_df)

    ax.plot(x, y, color=COLOR_DYNAMIC, linewidth=DATA_LINEWIDTH, zorder=3, label="memory loss")
    ax.scatter(x, y, s=12, color=COLOR_DYNAMIC, edgecolor="white", linewidth=0.35, zorder=4)
    ax.plot(x_fit, y_fit, color="#B7A37A", linewidth=1.15, linestyle=(0, (4, 3)), zorder=2, label="exp fit")
    ax.scatter([peak_x], [peak_y], s=42, color=COLOR_DYNAMIC, edgecolor=COLOR_TEXT, linewidth=0.55, zorder=5)

    legend_handles = [
        Line2D([0], [0], color=COLOR_DYNAMIC, lw=DATA_LINEWIDTH, marker="o", markersize=4.0, label="memory loss"),
        Line2D([0], [0], color="#B7A37A", lw=1.15, linestyle=(0, (4, 3)), label="exp fit"),
        Line2D([0], [0], color="none", lw=0.0, label=f"Peak loss = {peak_value:.1f}%"),
    ]
    if np.isfinite(tau_value):
        legend_handles.append(Line2D([0], [0], color="none", lw=0.0, label=f"Decay τ = {tau_value:.0f} ms"))
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor=COLOR_DARK_GRAY,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.35,
    )
    ax.set_xlabel("delay (ms)")
    ax.set_ylabel("memory loss (%)")
    ax.set_xlim(float(x.min()) - 40.0, float(x.max()) + 40.0)
    ax.set_ylim(max(0.0, float(y.min()) - 2.5), float(y.max()) + 5.0)


def draw_panel_c_original(fig: plt.Figure, spec, raster_df: pd.DataFrame, rate_df: pd.DataFrame) -> tuple[list[plt.Axes], plt.Axes]:
    representative_meta = {
        "phase_slices": _infer_phase_slices(rate_df),
        "layer_neuron_counts": _infer_layer_neuron_counts(raster_df),
    }
    dt_ms = _infer_dt_ms(rate_df)
    panel_grid = spec.subgridspec(3, 1, height_ratios=[0.95, 0.95, 1.0], hspace=0.08)
    raster_axes = [fig.add_subplot(panel_grid[i, 0]) for i in range(2)]
    ax_rate = fig.add_subplot(panel_grid[2, 0], sharex=raster_axes[0])
    _plot_representative_population_rate(
        ax_rate,
        df_population_rate=rate_df,
        phase_slices=representative_meta["phase_slices"],
        dt_ms=dt_ms,
    )
    _plot_representative_raster(
        raster_axes,
        df_points=raster_df,
        phase_slices=representative_meta["phase_slices"],
        layer_neuron_counts=representative_meta["layer_neuron_counts"],
        dt_ms=dt_ms,
    )
    return raster_axes, ax_rate


def draw_panel_d_decode_bars(ax: plt.Axes, decode_df: pd.DataFrame) -> None:
    style_axes(ax)
    order = ["layer1", "layer2", "layer3"]
    label_map = {"layer1": "Layer 1", "layer2": "Layer 2", "layer3": "Layer 3"}
    palette = {"layer1": "#7A8798", "layer2": "#2E8F67", "layer3": "#0D6F4F"}

    summary_df = (
        decode_df.groupby("layer", as_index=False)
        .agg(acc=("acc", "mean"))
    )
    summary_df["layer"] = pd.Categorical(summary_df["layer"], categories=order, ordered=True)
    summary_df = summary_df.sort_values("layer").reset_index(drop=True)

    x = np.arange(len(summary_df), dtype=float)
    ax.axhline(CHANCE_LEVEL, color=COLOR_DARK_GRAY, linewidth=REF_LINEWIDTH, linestyle=(0, (3, 2)), alpha=0.55, zorder=1)
    ax.bar(
        x,
        summary_df["acc"].to_numpy(dtype=float),
        color=[palette[str(item)] for item in summary_df["layer"]],
        edgecolor="none",
        width=0.58,
        zorder=2,
    )

    for idx, layer in enumerate(order):
        ax.text(
            x[idx],
            float(summary_df.loc[summary_df["layer"] == layer, "acc"].iloc[0]) + 0.03,
            f"{100.0 * float(summary_df.loc[summary_df['layer'] == layer, 'acc'].iloc[0]):.1f}%",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
        )

    ax.set_xticks(x, [label_map[str(item)] for item in summary_df["layer"]])
    ax.set_ylabel("decoding accuracy")
    ax.set_ylim(0.0, 1.06)


def build_panel_figures(root: str | Path) -> tuple[dict[str, plt.Figure], dict[str, object]]:
    apply_paper_style()
    bundle = load_fig2_bundle(root)
    figures: dict[str, plt.Figure] = {}

    fig_a = plt.figure(figsize=(5.0, 1.35))
    ax_a = fig_a.add_subplot(1, 1, 1)
    draw_panel_a_task_schematic(ax_a)
    figures["panel_a"] = fig_a

    fig_b = plt.figure(figsize=(3.5, 2.8))
    ax_b = fig_b.add_subplot(1, 1, 1)
    draw_panel_b_peak_decay(ax_b, bundle["summary"], bundle["panel_b_memory"], bundle["panel_b_fit"])
    figures["panel_b"] = fig_b

    fig_c = plt.figure(figsize=(5.9, 3.0))
    spec_c = fig_c.add_gridspec(1, 1)
    c_axes = draw_panel_c_original(fig_c, spec_c[0, 0], bundle["panel_c_raster"], bundle["panel_c_rate"])
    figures["panel_c"] = fig_c

    fig_d = plt.figure(figsize=(3.05, 2.45))
    ax_d = fig_d.add_subplot(1, 1, 1)
    draw_panel_d_decode_bars(ax_d, bundle["panel_d_decode"])
    figures["panel_d"] = fig_d

    panel_e_df = _panel_e_bias_display_frame(bundle["panel_e_bias"])
    for idx, row in panel_e_df.iterrows():
        fig_e = plt.figure(figsize=(3.3, 2.55 if idx == 0 else 2.15))
        ax_e = fig_e.add_subplot(1, 1, 1)
        draw_panel_e_single_donut(ax_e, row, include_legend=(idx == 0))
        figures[f"panel_e_{str(row['condition'])}"] = fig_e
    return figures, bundle


def build_assembled_figure(root: str | Path) -> tuple[plt.Figure, dict[str, object]]:
    apply_paper_style()
    bundle = load_fig2_bundle(root)
    fig = plt.figure(figsize=(7.4, 9.6))
    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.48, 1.55, 2.1],
        width_ratios=[0.95, 1.35],
        hspace=0.42,
        wspace=0.34,
    )

    ax_a = fig.add_subplot(outer[0, :])
    ax_b = fig.add_subplot(outer[1, 0])
    spec_c = outer[1, 1]
    ax_d = fig.add_subplot(outer[2, 0])
    draw_panel_a_task_schematic(ax_a)
    draw_panel_b_peak_decay(ax_b, bundle["summary"], bundle["panel_b_memory"], bundle["panel_b_fit"])
    c_axes = draw_panel_c_original(fig, spec_c, bundle["panel_c_raster"], bundle["panel_c_rate"])
    draw_panel_d_decode_bars(ax_d, bundle["panel_d_decode"])
    draw_panel_e_error_donuts(fig, outer[2, 1], bundle["panel_e_bias"])
    return fig, bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render final Fig2 from paper_figs result tables.")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--export-panels", action="store_true")
    args = parser.parse_args(argv)

    input_dir = resolve_figure_input_dir("fig2", args.input_dir)
    output_dir = Path(input_dir) / "plots" if args.output_dir is None else Path(args.output_dir)

    fig, bundle = build_assembled_figure(input_dir)
    saved: dict[str, object] = {"figure": save_figure_outputs(fig, output_dir, "fig2")}
    plt.close(fig)

    if args.export_panels:
        panel_figures, _ = build_panel_figures(input_dir)
        panel_saved: dict[str, dict[str, str]] = {}
        for panel_name, panel_fig in panel_figures.items():
            panel_saved[panel_name] = save_figure_outputs(panel_fig, output_dir, f"fig2_{panel_name}")
            plt.close(panel_fig)
        saved["panels"] = panel_saved

    print(
        json.dumps(
            {
                "status": "ok",
                "figure": "fig2",
                "input_dir": str(input_dir),
                "summary_keys": sorted(bundle["summary"].keys()),
                "saved": saved,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
