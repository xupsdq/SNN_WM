from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from src.plotting.common.colors import get_plot_cmap, get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder


MAIN = get_plot_color("true_pair")
PEAK = "#D55E00"
RANDOM = "#6A6A6A"
VALLEY = "#0072B2"
GRID = "0.86"
STATE_ORDER = ("S_final", "S0", "S0_ping_null")
ACCESS_COLORS = {
    "cue_only": "0.62",
    "single_item_memory": "#D55E00",
    "sequence_state": "#0072B2",
    "singleton_access_fraction": "#D55E00",
    "sequence_access_fraction": "#0072B2",
    "rescued_fraction": "#009E73",
    "morphology_N_eff": "#6A6A6A",
    "single_item_access_count": "#D55E00",
    "sequence_state_access_count": "#0072B2",
    "rescued_count": "#009E73",
}
ACCESS_LABELS = {
    "cue_only": "Cue only",
    "single_item_memory": "Slot singleton",
    "sequence_state": "Full sequence",
    "singleton_access_fraction": "Singleton",
    "sequence_access_fraction": "Sequence",
    "rescued_fraction": "Rescued",
    "morphology_N_eff": "L1 N_eff",
    "single_item_access_count": "Singleton",
    "sequence_state_access_count": "Sequence",
    "rescued_count": "Rescued",
}
CUE_SPECIFICITY_COLORS = {
    "matched": "#0072B2",
    "mismatched": "#D55E00",
    "unseen": "#6A6A6A",
}
CUE_SPECIFICITY_LABELS = {
    "matched": "Matched",
    "mismatched": "Mismatched",
    "unseen": "Unseen",
}


def render_fig3_sequence_schematic(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data, stats, style
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    item_x = np.linspace(0.13, 0.55, 5)
    y = 0.62
    ax.text(0.05, y, "S0", ha="center", va="center", fontsize=7, fontweight="bold")
    ax.annotate("", xy=(0.09, y), xytext=(0.065, y), arrowprops={"arrowstyle": "->", "linewidth": 0.75, "color": "0.3"})
    for idx, x in enumerate(item_x, start=1):
        ax.add_patch(plt.Rectangle((x - 0.024, y - 0.11), 0.048, 0.16, facecolor=plt.cm.tab10((idx - 1) / 8), edgecolor="white", linewidth=0.7))
        label = str(idx) if idx < len(item_x) else "K"
        ax.text(x, y - 0.18, f"item {label}", ha="center", va="top", fontsize=5.8)
        if idx < len(item_x):
            ax.annotate("", xy=(x + 0.065, y - 0.03), xytext=(x + 0.027, y - 0.03), arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "0.35"})
    ax.text(0.64, y, "S_final", ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.annotate("", xy=(0.61, y), xytext=(0.58, y), arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "0.3"})
    branches = [("morphology", "B update", "C landscape", 0.78), ("function", "D neutral ping", "E weak probe  F region ping", 0.38)]
    for name, first, second, by in branches:
        ax.annotate("", xy=(0.73, by), xytext=(0.665, y - 0.02), arrowprops={"arrowstyle": "->", "linewidth": 0.75, "color": "0.25"})
        ax.text(0.75, by + 0.08, name, ha="left", va="center", fontsize=6.2, color="0.2")
        ax.text(0.75, by, first, ha="left", va="center", fontsize=5.7)
        ax.text(0.75, by - 0.09, second, ha="left", va="center", fontsize=5.7)
    ax.paper_fig_plot_form = "programmatic_multiitem_sequence_schematic"


def render_fig3_progressive_update(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("stepwise_update_ratio")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty or "stage_k" not in use.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    use["stage_k"] = pd.to_numeric(use["stage_k"], errors="coerce")
    summary = _summary(use, "stage_k", "value")
    if not summary.empty:
        ax.fill_between(summary["x"], summary["mean"] - summary["sem"], summary["mean"] + summary["sem"], color=MAIN, alpha=0.18, linewidth=0)
        ax.plot(summary["x"], summary["mean"], color=MAIN, linewidth=1.35, marker="o", markersize=2.6)
    ax.set_xlabel("Sequence stage")
    ax.set_ylabel("Update ratio")
    ax.set_xlim(1, max(3, float(np.nanmax(use["stage_k"]))) + 0.25)
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_progressive_update"
    ax.paper_fig_individual_traces = False
    ax.paper_fig_has_shaded_band = True


def render_fig3_3d_landscape(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = style
    df = _clean(panel_data)
    if df.empty or not {"row", "col", "value"}.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    try:
        if not hasattr(ax, "plot_surface"):
            raise TypeError("axis is not a 3D projection")
        mat, rows, cols = _matrix(df)
        xx, yy = np.meshgrid(cols, rows)
        finite = mat[np.isfinite(mat)]
        if finite.size == 0:
            raise ValueError("landscape matrix contains no finite values")
        diverging = float(np.nanmin(finite)) < 0 < float(np.nanmax(finite)) or (float(np.nanmin(finite)) < 0 and str((stats or {}).get("landscape_metric_used", "")).startswith("delta"))
        if diverging:
            vmax = float(np.nanpercentile(np.abs(finite), 98)) or 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            cmap = get_plot_cmap("difference")
            cbar_label = "Δ STSP support"
        else:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanpercentile(finite, 92)) or float(np.nanmax(finite)) or 1.0
            if not np.isfinite(vmax) or vmax <= vmin:
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite)) or 1.0
            norm = PowerNorm(gamma=0.38, vmin=vmin, vmax=vmax, clip=True)
            cmap = cm.turbo
            cbar_label = "Support"
        colors = cmap(norm(mat))
        surface = ax.plot_surface(xx, yy, mat, facecolors=colors, rstride=1, cstride=1, linewidth=0.04, edgecolor=(1, 1, 1, 0.16), antialiased=True, shade=False, alpha=1.0)
        surface.set_array(finite)
        surface.set_cmap(cmap)
        surface.set_norm(norm)
        zmin = float(np.nanmin(finite))
        zmax = float(np.nanmax(finite))
        zspan = max(zmax - zmin, 1e-9)
        try:
            floor = np.full_like(mat, zmin - 0.16 * zspan)
            ax.plot_surface(xx, yy, floor, facecolors=cmap(norm(mat)), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.78)
        except Exception:
            pass
        for role, color, marker in (("peak", PEAK, "o"), ("valley", VALLEY, "^"), ("random", RANDOM, "s")):
            pts = df[df.get("mask_role", pd.Series(dtype=str)).astype(str).eq(role)].copy()
            if not pts.empty:
                pts["row"] = pd.to_numeric(pts["row"], errors="coerce")
                pts["col"] = pd.to_numeric(pts["col"], errors="coerce")
                pts["value"] = pd.to_numeric(pts["value"], errors="coerce")
                pts = pts.iloc[:: max(1, int(len(pts) // 32))]
                ax.scatter(pts["col"], pts["row"], pts["value"] + 0.012 * (np.nanmax(finite) - np.nanmin(finite) + 1e-9), s=3.6, color=color, marker=marker, depthshade=False, alpha=0.50)
        ax.view_init(elev=56, azim=-50)
        try:
            ax.set_box_aspect((1.08, 0.95, 0.18), zoom=0.72)
        except TypeError:
            try:
                ax.set_box_aspect((1.08, 0.95, 0.18))
                ax.dist = 12
            except Exception:
                pass
        except Exception:
            pass
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
        ax.set_xlim(float(cols.min()), float(cols.max()))
        ax.set_ylim(float(rows.min()), float(rows.max()))
        ax.set_zlim(zmin - 0.16 * zspan, zmax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("Δ STSP support" if cbar_label.startswith("Δ") else "STSP support", fontsize=5.2, labelpad=-1.5)
        ax.set_xticks([0, mat.shape[1] - 1])
        ax.set_yticks([0, mat.shape[0] - 1])
        ax.tick_params(axis="x", labelsize=4.8, pad=-2.0)
        ax.tick_params(axis="y", labelsize=4.8, pad=-2.0)
        ax.tick_params(axis="z", labelsize=4.8, pad=-0.5)
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.pane.set_edgecolor("1.0")
                axis.pane.set_facecolor((1, 1, 1, 0))
                axis.line.set_color((0, 0, 0, 0.45 if axis is ax.zaxis else 0.0))
            except Exception:
                pass
        cax = getattr(ax, "paper_fig_colorbar_ax", None)
        if cax is not None:
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = ax.figure.colorbar(mappable, cax=cax)
            if hasattr(norm, "vmin") and hasattr(norm, "vmax"):
                cbar.set_ticks([norm.vmin, norm.vmax])
                cbar.ax.set_yticklabels([f"{float(norm.vmin):.2f}", f"{float(norm.vmax):.2f}"])
            cbar.ax.tick_params(labelsize=4.6, length=1.2, width=0.4, pad=1.0)
            cbar.set_label(cbar_label, fontsize=5.2, labelpad=1.2)
            ax.paper_fig_has_colorbar = True
            ax.paper_fig_colorbar_label = cbar_label
        ax.paper_fig_plot_form = "fig3_3d_surface_landscape"
        ax.paper_fig_is_3d_surface = True
        ax.paper_fig_has_summary_inset = False
    except Exception as exc:
        if hasattr(ax, "text2D"):
            ax.text2D(0.02, 0.98, "2D fallback", transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color="0.35")
        _render_landscape_heatmap(ax, df, stats, spec, style)
        ax.paper_fig_plot_form = "fig3_2d_landscape_fallback"
        ax.paper_fig_3d_fallback_reason = str(exc)
        ax.paper_fig_has_summary_inset = False


def render_fig3_landscape_heatmap(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty or not {"row", "col", "value"}.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    mat, rows, cols = _matrix(df)
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanpercentile(finite, 94)) or float(np.nanmax(finite)) or 1.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite)) or 1.0
    norm = PowerNorm(gamma=0.48, vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.turbo
    image = ax.imshow(mat, origin="lower", cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    contour_levels = [float(np.nanpercentile(finite, q)) for q in (60, 78, 90)]
    contour_levels = sorted({level for level in contour_levels if np.isfinite(level) and vmin < level < vmax})
    if contour_levels:
        ax.contour(mat, levels=contour_levels, colors="white", linewidths=0.28, alpha=0.42, origin="lower")
    for role, color, marker, alpha in (("peak", PEAK, "o", 0.72), ("valley", VALLEY, "^", 0.58), ("random", RANDOM, "s", 0.42)):
        pts = df[df.get("mask_role", pd.Series(dtype=str)).astype(str).eq(role)].copy()
        if pts.empty:
            continue
        pts = pts.iloc[:: max(1, int(len(pts) // 80))]
        ax.scatter(pts["col"], pts["row"], s=8.0 if role == "peak" else 6.0, facecolors="none", edgecolors=color, linewidths=0.55, marker=marker, alpha=alpha)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xticks([0, len(cols) - 1])
    ax.set_yticks([0, len(rows) - 1])
    ax.set_xlabel("Spatial x")
    ax.set_ylabel("Spatial y")
    cax = getattr(ax, "paper_fig_colorbar_ax", None)
    if cax is not None:
        cbar = ax.figure.colorbar(image, cax=cax)
        cbar.set_ticks([vmin, vmax])
        cbar.ax.set_yticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
        cbar.ax.tick_params(labelsize=5.0, length=1.4, width=0.45, pad=1.0)
        cbar.set_label("STSP support", fontsize=5.6, labelpad=1.4)
        ax.paper_fig_has_colorbar = True
        ax.paper_fig_colorbar_label = "STSP support"
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_support_landscape_heatmap"
    ax.paper_fig_is_3d_surface = False
    ax.paper_fig_has_summary_inset = False


def render_fig3_neutral_ping_serial_profile(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = style
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("readout_mass")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty or "serial_position" not in use.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    serial_numeric = pd.to_numeric(use["serial_position"], errors="coerce")
    numeric_positions = serial_numeric.dropna()
    if numeric_positions.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    max_pos = int(np.nanmax(numeric_positions))
    serial_bin = use.get("serial_bin", pd.Series("", index=use.index)).astype(str).str.lower()
    nonserial_mask = serial_numeric.isna() & serial_bin.isin({"silent", "other", "no_readout", "none"})
    use["serial_position_plot"] = serial_numeric
    silent_values = pd.to_numeric(use.loc[serial_bin.eq("silent"), "value"], errors="coerce").fillna(0.0)
    has_silent_bin = bool(nonserial_mask.any() and float(silent_values.sum()) > 1e-12)
    if has_silent_bin:
        use.loc[serial_bin.eq("silent"), "serial_position_plot"] = max_pos + 1
    use = use.dropna(subset=["serial_position_plot"])
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    colors = {"S_final": MAIN, "S0": "0.45", "S0_ping_null": "0.55"}
    states = [s for s in ("S_final", "S0") if s in set(use.get("state_condition", use.get("condition")).astype(str))]
    for state in states:
        part = use[use.get("state_condition", use.get("condition")).astype(str).eq(state)]
        if part.empty:
            continue
        summary = _summary(part, "serial_position_plot", "value")
        label = {"S_final": "S final", "S0": "S0"}.get(state, state.replace("_", " "))
        ax.plot(summary["x"], summary["mean"], color=colors.get(state, "0.25"), linewidth=1.3, marker="o", markersize=2.6, label=label)
        ax.fill_between(summary["x"], summary["mean"] - summary["sem"], summary["mean"] + summary["sem"], color=colors.get(state, "0.25"), alpha=0.12, linewidth=0)
    ticks = list(range(1, max_pos + 1))
    if has_silent_bin:
        ticks = ticks + [max_pos + 1]
    if has_silent_bin and max_pos >= 8:
        shown_ticks = list(range(1, max_pos + 1, 2)) + [max_pos + 1]
    else:
        shown_ticks = ticks if max_pos <= 10 else ticks[::2]
        if has_silent_bin and (max_pos + 1) not in shown_ticks:
            shown_ticks = shown_ticks + [max_pos + 1]
    ax.set_xticks(shown_ticks)
    ax.set_xticklabels(["Silent" if tick == max_pos + 1 and has_silent_bin else str(int(tick)) for tick in shown_ticks])
    ax.set_xlim(0.75, max_pos + (1.28 if has_silent_bin else 0.25))
    ymax = min(1.0, max(0.25, float(use["value"].max()) * 1.2))
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Serial position")
    ax.set_ylabel("Readout mass")
    legend = ax.legend(frameon=False, fontsize=5.1, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, handlelength=1.0, borderaxespad=0.0, columnspacing=0.8)
    ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
    ax.paper_fig_legend_ncols = 2
    ax.paper_fig_legend_above_plot = True
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_neutral_ping_serial_profile"
    ax.paper_fig_x_metric = "serial_position"
    ax.paper_fig_y_metric = "readout_mass"


def render_fig3_weak_probe_completion(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = style
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("P_target")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty or "keep_prob" not in use.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if "target_source" in use.columns and use["target_source"].astype(str).str.contains("sequence_member", na=False).any():
        use = use[use["target_source"].astype(str).str.contains("sequence_member", na=False)].copy()
    use["keep_prob"] = pd.to_numeric(use["keep_prob"], errors="coerce")
    colors = {"cue_only": "0.45", "single_item_memory": "#009E73", "sequence_state": MAIN}
    labels = {"cue_only": "No memory", "single_item_memory": "Single-item memory", "sequence_state": "Sequence state"}
    markers = {"cue_only": "o", "single_item_memory": "s", "sequence_state": "^"}
    linestyles = {"cue_only": "-", "single_item_memory": "-", "sequence_state": "-"}
    zorders = {"cue_only": 3, "sequence_state": 4, "single_item_memory": 5}
    legend_handles = {}
    for condition in ("cue_only", "sequence_state", "single_item_memory"):
        part = use[use["memory_condition"].astype(str).eq(condition)] if "memory_condition" in use.columns else pd.DataFrame()
        if part.empty:
            continue
        summary = _summary(part, "keep_prob", "value")
        (line,) = ax.plot(
            summary["x"],
            summary["mean"],
            color=colors[condition],
            linewidth=1.35,
            linestyle=linestyles[condition],
            marker=markers[condition],
            markersize=2.8,
            markerfacecolor="white" if condition == "single_item_memory" else colors[condition],
            markeredgewidth=0.7,
            label=labels[condition],
            zorder=zorders[condition],
        )
        legend_handles[condition] = line
        ax.fill_between(summary["x"], summary["mean"] - summary["sem"], summary["mean"] + summary["sem"], color=colors[condition], alpha=0.16, linewidth=0)
    ax.set_xlabel("Weak-probe keep probability")
    ax.set_ylabel("Target recovery (%)")
    xmax = float(pd.to_numeric(use["keep_prob"], errors="coerce").max())
    keep_ticks = sorted(pd.to_numeric(use["keep_prob"], errors="coerce").dropna().unique().tolist())
    if 1 <= len(keep_ticks) <= 5:
        ax.set_xticks(keep_ticks)
    elif keep_ticks:
        shown = [keep_ticks[0], 0.3, 0.5, 0.7, keep_ticks[-1]]
        shown = sorted({float(v) for v in shown if min(keep_ticks) <= float(v) <= max(keep_ticks)})
        ax.set_xticks(shown)
    ax.set_xlim(0, max(0.35, xmax * 1.14))
    ymax = max(100.0, float(pd.to_numeric(use["value"], errors="coerce").max()) * 1.08)
    ymin = -4.0 if float(pd.to_numeric(use["value"], errors="coerce").min()) <= 0 else 0.0
    ax.set_ylim(ymin, ymax)
    if ymin < 0:
        ax.axhline(0, color="0.62", linewidth=0.55, linestyle="--", zorder=0)
    ordered = [key for key in ("cue_only", "single_item_memory", "sequence_state") if key in legend_handles]
    legend = ax.legend(
        [legend_handles[key] for key in ordered],
        [labels[key] for key in ordered],
        frameon=False,
        fontsize=4.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=0.9,
        borderaxespad=0.0,
        columnspacing=0.45,
        handletextpad=0.28,
    )
    ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
    ax.paper_fig_legend_ncols = 2
    ax.paper_fig_legend_above_plot = True
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_weak_probe_completion"
    ax.paper_fig_y_metric = "P_target"


def render_fig3_peak_cue_memory_gain(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("memory_gain")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    use["cue_condition"] = use.get("cue_condition", use.get("condition")).astype(str).map(_norm_cue)
    order = ["valley", "random", "peak"]
    colors = {"valley": VALLEY, "random": RANDOM, "peak": PEAK}
    xs = np.arange(len(order))
    for x, cue in zip(xs, order):
        vals = pd.to_numeric(use.loc[use["cue_condition"].eq(cue), "value"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        jitter = np.linspace(-0.10, 0.10, vals.size) if vals.size > 1 else np.array([0.0])
        ax.scatter(np.full(vals.size, x) + jitter, vals, s=12, color=colors[cue], alpha=0.58, linewidth=0)
        ax.errorbar(x, vals.mean(), yerr=_sem(vals), fmt="o", color="0.12", markersize=3.2, capsize=2.0, linewidth=0.8)
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.6)
    ax.set_xticks(xs, ["Valley", "Random", "Peak"])
    ax.set_ylabel("Memory gain (%)")
    ax.set_xlabel("")
    _title(ax, spec)
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_peak_cue_memory_gain"
    ax.paper_fig_y_metric = "memory_gain"
    ax.paper_fig_raw_points = True
    ax.paper_fig_raw_point_count = int(len(use))


def render_fig3_region_ping_readout(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = style
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("readout_mass")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty or "region_condition" not in use.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    region_order = [r for r in ("peak", "valley", "random") if r in set(use["region_condition"].astype(str))]
    if not {"peak", "valley"}.issubset(set(region_order)):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    if "random" not in region_order:
        ax.text(0.98, 0.96, "random missing", transform=ax.transAxes, ha="right", va="top", fontsize=5.0, color="0.35")
    cats = ["old", "recent", "silent"]
    colors = {"old": MAIN, "recent": "#E69F00", "silent": "0.84"}
    agg = _region_ping_categories(use, region_order)
    xs = np.arange(len(region_order))
    bottom = np.zeros(len(region_order), dtype=float)
    for cat in cats:
        vals = np.asarray([agg.get(region, {}).get(cat, 0.0) for region in region_order], dtype=float)
        ax.bar(xs, vals, bottom=bottom, width=0.62, color=colors[cat], edgecolor="white", linewidth=0.35, label=cat.title())
        bottom += vals
    ax.set_xticks(xs, [r.title() for r in region_order])
    ax.set_ylim(0, max(1.0, float(np.nanmax(bottom)) * 1.16 if bottom.size else 1.0))
    ax.set_ylabel("Evoked readout probability")
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=4.7, loc="upper right", handlelength=0.8, borderaxespad=0.2, ncol=1)
    _title(ax, spec)
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_region_ping_stacked_readout_mass"
    ax.paper_fig_y_metric = "readout_mass"
    ax.paper_fig_region_ping_categories = cats
    ax.paper_fig_stacked_bars_not_normalized = True


def render_fig3_access_serial_profile(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("target_probability")].copy() if "metric" in df.columns else pd.DataFrame()
    required = {"memory_condition", "serial_position", "value"}
    if use.empty or not required.issubset(use.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    use["serial_position"] = pd.to_numeric(use["serial_position"], errors="coerce")
    use = use.dropna(subset=["serial_position", "value"])
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    order = ["cue_only", "single_item_memory", "sequence_state"]
    for memory in order:
        part = use[use["memory_condition"].astype(str).eq(memory)]
        if part.empty:
            continue
        summary = _summary(part, "serial_position", "value")
        if summary.empty:
            continue
        xs = summary["x"].to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        sem = summary["sem"].to_numpy(dtype=float)
        color = ACCESS_COLORS.get(memory, MAIN)
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.13, linewidth=0)
        ax.plot(xs, mean, color=color, linewidth=1.25, marker="o", markersize=2.5, label=ACCESS_LABELS.get(memory, memory))
    serial = pd.to_numeric(use["serial_position"], errors="coerce").dropna().to_numpy(dtype=float)
    if serial.size == 0:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    ticks = np.sort(np.unique(serial))
    ax.set_xticks(ticks if ticks.size <= 10 else np.linspace(float(ticks.min()), float(ticks.max()), 5))
    ax.set_xlim(max(0.55, float(np.nanmin(serial)) - 0.45), float(np.nanmax(serial)) + 0.45)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("Serial position")
    ax.set_ylabel("Target readout probability")
    ax.legend(frameon=False, fontsize=5.6, loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.8, borderaxespad=0.25)
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_access_serial_profile"
    ax.paper_fig_y_metric = "target_probability"


def render_fig3_cue_specificity_target_profile(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("target_probability")].copy() if "metric" in df.columns else pd.DataFrame()
    required = {"cue_type", "serial_position", "value"}
    if use.empty or not required.issubset(use.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    use["serial_position"] = pd.to_numeric(use["serial_position"], errors="coerce")
    use["value"] = pd.to_numeric(use["value"], errors="coerce")
    if "sem" in use.columns:
        use["sem"] = pd.to_numeric(use["sem"], errors="coerce")
    else:
        use["sem"] = np.nan
    use = use.dropna(subset=["serial_position", "value"])
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    cue_order = [str(v) for v in (spec.get("cue_types") or ["matched", "mismatched", "unseen"])]
    for cue in cue_order:
        part = use[use["cue_type"].astype(str).eq(cue)].copy()
        if part.empty:
            continue
        rows = []
        for serial, serial_part in part.groupby("serial_position", sort=True):
            vals = pd.to_numeric(serial_part["value"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            if vals.size > 1:
                sem = _sem(vals)
            else:
                sem_values = pd.to_numeric(serial_part.get("sem", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
                sem = float(sem_values.mean()) if sem_values.size else 0.0
            rows.append({"x": float(serial), "mean": float(vals.mean()), "sem": sem})
        summary = pd.DataFrame(rows).sort_values("x") if rows else pd.DataFrame()
        if summary.empty:
            continue
        xs = summary["x"].to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        sem = summary["sem"].to_numpy(dtype=float)
        color = CUE_SPECIFICITY_COLORS.get(cue, MAIN)
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.13, linewidth=0)
        ax.plot(xs, mean, color=color, linewidth=1.25, marker="o", markersize=2.5, label=CUE_SPECIFICITY_LABELS.get(cue, cue))
    serial = pd.to_numeric(use["serial_position"], errors="coerce").dropna().to_numpy(dtype=float)
    ticks = np.sort(np.unique(serial))
    ax.set_xticks(ticks if ticks.size <= 10 else np.linspace(float(ticks.min()), float(ticks.max()), 5))
    ax.set_xlim(max(0.55, float(np.nanmin(serial)) - 0.45), float(np.nanmax(serial)) + 0.45)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("Serial position")
    ax.set_ylabel("Target readout probability")
    ax.legend(frameon=False, fontsize=5.6, loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.8, borderaxespad=0.25)
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_cue_specificity_target_profile"
    ax.paper_fig_y_metric = "target_probability"
    ax.paper_fig_state_condition = str(spec.get("state_condition") or "S_final")


def render_fig3_rescue_fraction(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty or not {"seq_len", "metric", "value"}.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    requested_metrics = spec.get("metrics_to_plot")
    if requested_metrics in (None, ""):
        requested_metrics = [str(spec.get("metric") or "rescued_fraction")]
    if isinstance(requested_metrics, str):
        metrics = [requested_metrics]
    else:
        metrics = [str(item) for item in requested_metrics]
    if not metrics:
        metrics = ["rescued_fraction"]
    work = df[df["metric"].astype(str).isin(metrics)].copy()
    work["seq_len"] = pd.to_numeric(work["seq_len"], errors="coerce")
    work = work.dropna(subset=["seq_len", "value"])
    if work.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    seqs = np.sort(work["seq_len"].unique())
    xs = np.arange(len(seqs), dtype=float)
    if len(metrics) == 1:
        metric = metrics[0]
        means: list[float] = []
        sems: list[float] = []
        counts: list[float] = []
        for seq_len in seqs:
            mask = work["seq_len"].eq(seq_len) & work["metric"].astype(str).eq(metric)
            vals = pd.to_numeric(work.loc[mask, "value"], errors="coerce").dropna().to_numpy(dtype=float)
            means.append(float(vals.mean()) if vals.size else np.nan)
            sems.append(_sem(vals))
            if "item_count" in work.columns:
                item_counts = pd.to_numeric(work.loc[mask, "item_count"], errors="coerce").dropna().to_numpy(dtype=float)
                counts.append(float(item_counts.mean()) if item_counts.size else np.nan)
            else:
                counts.append(np.nan)
        color = ACCESS_COLORS.get(metric, MAIN)
        mean_arr = np.asarray(means, dtype=float)
        ax.vlines(xs, 0, mean_arr, color=color, linewidth=2.1, alpha=0.62, zorder=2)
        ax.plot(xs, mean_arr, color="0.16", linewidth=0.85, zorder=3)
        ax.scatter(xs, mean_arr, s=22, facecolor=color, edgecolor="0.16", linewidth=0.45, zorder=4)
        ax.errorbar(xs, mean_arr, yerr=sems, fmt="none", ecolor="0.25", elinewidth=0.55, capsize=1.2, capthick=0.55, zorder=5)
        if str(spec.get("annotate_counts", "true")).lower() not in {"false", "0", "no"}:
            for x, mean, count, seq_len in zip(xs, means, counts, seqs):
                if np.isfinite(mean) and np.isfinite(count):
                    ax.text(x, min(1.07, mean + 0.065), f"{count:.1f}/{int(seq_len)}", ha="center", va="bottom", fontsize=4.8, color="0.18")
        ax.paper_fig_plot_form = "fig3_rescue_fraction_lollipop"
    else:
        width = 0.23
        offsets = np.linspace(-width, width, len(metrics))
        for metric, offset in zip(metrics, offsets):
            means = []
            sems = []
            for seq_len in seqs:
                mask = work["seq_len"].eq(seq_len) & work["metric"].astype(str).eq(metric)
                vals = pd.to_numeric(work.loc[mask, "value"], errors="coerce").dropna().to_numpy(dtype=float)
                means.append(float(vals.mean()) if vals.size else np.nan)
                sems.append(_sem(vals))
            ax.bar(xs + offset, means, width=width, color=ACCESS_COLORS.get(metric, MAIN), edgecolor="white", linewidth=0.35, label=ACCESS_LABELS.get(metric, metric))
            ax.errorbar(xs + offset, means, yerr=sems, fmt="none", ecolor="0.25", elinewidth=0.45, capsize=1.0, capthick=0.45)
        ax.legend(frameon=False, fontsize=4.7, loc="upper left", handlelength=0.9, borderaxespad=0.15)
        ax.paper_fig_plot_form = "fig3_rescue_fraction"
    ax.set_xticks(xs, [str(int(v)) for v in seqs])
    ax.set_xlabel("K")
    ax.set_ylabel("Rescued item fraction" if metrics == ["rescued_fraction"] else "Fraction")
    ax.set_ylim(0, 1.12)
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_y_metric = metrics[0] if len(metrics) == 1 else "rescued_fraction"


def render_fig3_morphology_capacity(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("N_eff")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty or not {"seq_len", "value"}.issubset(use.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    use["seq_len"] = pd.to_numeric(use["seq_len"], errors="coerce")
    use = use.dropna(subset=["seq_len", "value"])
    summary = _summary(use, "seq_len", "value")
    if summary.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    xs = summary["x"].to_numpy(dtype=float)
    mean = summary["mean"].to_numpy(dtype=float)
    sem = summary["sem"].to_numpy(dtype=float)
    ax.plot(xs, xs, color="0.72", linewidth=0.85, linestyle="--", label="Independent")
    ax.fill_between(xs, mean - sem, mean + sem, color=MAIN, alpha=0.15, linewidth=0)
    ax.plot(xs, mean, color=MAIN, linewidth=1.25, marker="o", markersize=2.6, label="Observed")
    ax.set_xticks(xs, [str(int(v)) for v in xs])
    ax.set_xlim(float(xs.min()) - 0.6, float(xs.max()) + 0.6)
    ax.set_ylim(0, max(float(xs.max()), float(np.nanmax(mean + sem))) * 1.08)
    ax.set_xlabel("K")
    ax.set_ylabel("Effective L1 items")
    ax.legend(frameon=False, fontsize=4.8, loc="upper left", handlelength=1.1, borderaxespad=0.15)
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_morphology_capacity"
    ax.paper_fig_y_metric = "N_eff"


def render_fig3_access_capacity(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    metrics = ["morphology_N_eff", "single_item_access_count", "sequence_state_access_count", "rescued_count"]
    if df.empty or not {"seq_len", "metric", "value"}.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    work = df[df["metric"].astype(str).isin(metrics)].copy()
    work["seq_len"] = pd.to_numeric(work["seq_len"], errors="coerce")
    work = work.dropna(subset=["seq_len", "value"])
    if work.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    seqs = np.sort(work["seq_len"].unique())
    xs = np.arange(len(seqs), dtype=float)
    width = 0.18
    offsets = np.linspace(-0.27, 0.27, len(metrics))
    ymax = 0.0
    for metric, offset in zip(metrics, offsets):
        means: list[float] = []
        sems: list[float] = []
        for seq_len in seqs:
            vals = pd.to_numeric(work.loc[work["seq_len"].eq(seq_len) & work["metric"].astype(str).eq(metric), "value"], errors="coerce").dropna().to_numpy(dtype=float)
            means.append(float(vals.mean()) if vals.size else np.nan)
            sems.append(_sem(vals))
        finite = np.asarray(means, dtype=float)
        if np.isfinite(finite).any():
            ymax = max(ymax, float(np.nanmax(finite)))
        ax.bar(xs + offset, means, width=width, color=ACCESS_COLORS.get(metric, MAIN), edgecolor="white", linewidth=0.35, label=ACCESS_LABELS.get(metric, metric))
        ax.errorbar(xs + offset, means, yerr=sems, fmt="none", ecolor="0.25", elinewidth=0.45, capsize=1.0, capthick=0.45)
    ax.set_xticks(xs, [str(int(v)) for v in seqs])
    ax.set_xlabel("K")
    ax.set_ylabel("Items")
    ax.set_ylim(0, max(1.0, ymax * 1.18))
    ax.legend(frameon=False, fontsize=4.15, loc="upper left", handlelength=0.75, borderaxespad=0.1, labelspacing=0.18)
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_access_capacity"
    ax.paper_fig_y_metric = "sequence_state_access_count"


def render_fig3_morphology_serial_profile(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("morphology_support_mass")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty or "serial_position" not in use.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    use["serial_position"] = pd.to_numeric(use["serial_position"], errors="coerce")
    use = use.dropna(subset=["serial_position", "value"])
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    summary = _summary(use, "serial_position", "value")
    if summary.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    xs = summary["x"].to_numpy(dtype=float)
    mean = summary["mean"].to_numpy(dtype=float)
    sem = summary["sem"].to_numpy(dtype=float)
    inferred_k = _infer_seq_len(use)
    if inferred_k > 0:
        ax.axhline(1.0 / inferred_k, color="0.68", linewidth=0.8, linestyle="--", label="Uniform")
    ax.fill_between(xs, mean - sem, mean + sem, color=MAIN, alpha=0.16, linewidth=0)
    ax.plot(xs, mean, color=MAIN, linewidth=1.35, marker="o", markersize=2.8, label="Observed")
    raw_x = pd.to_numeric(use["serial_position"], errors="coerce").to_numpy(dtype=float)
    raw_y = pd.to_numeric(use["value"], errors="coerce").to_numpy(dtype=float)
    if raw_x.size <= 40:
        jitter = np.linspace(-0.045, 0.045, raw_x.size) if raw_x.size > 1 else np.array([0.0])
        ax.scatter(raw_x + jitter, raw_y, s=7.5, color=MAIN, alpha=0.32, linewidth=0)
    ax.set_xlabel("Serial position")
    ax.set_ylabel("L1 STSP support")
    xmin = max(0.6, float(np.nanmin(xs)) - 0.45)
    xmax = float(np.nanmax(xs)) + 0.45
    ax.set_xlim(xmin, xmax)
    unique_positions = np.sort(np.unique(raw_x[np.isfinite(raw_x)]))
    if unique_positions.size <= 10:
        ticks = unique_positions
    else:
        ticks = np.linspace(float(unique_positions.min()), float(unique_positions.max()), 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(round(tick))) for tick in ticks])
    ymax = float(np.nanmax(mean + sem)) if mean.size else 1.0
    ax.set_ylim(0, max(0.05, ymax * 1.18))
    ax.legend(frameon=False, fontsize=4.8, loc="upper right", handlelength=1.0, borderaxespad=0.15)
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_morphology_serial_profile"
    ax.paper_fig_y_metric = "morphology_support_mass"


def render_fig3_boundary_heatmap(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    metric = str(spec.get("metric") or "")
    use = df[df["metric"].astype(str).eq(metric)].copy() if metric and "metric" in df.columns else df.copy()
    if use.empty or not {"seq_len", "delay_ms", "value"}.issubset(use.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    use["seq_len"] = pd.to_numeric(use["seq_len"], errors="coerce")
    use["delay_ms"] = pd.to_numeric(use["delay_ms"], errors="coerce")
    use = use.dropna(subset=["seq_len", "delay_ms", "value"])
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    xs = np.sort(use["seq_len"].unique())
    ys = np.sort(use["delay_ms"].unique())
    mat = np.full((len(ys), len(xs)), np.nan)
    for yi, delay in enumerate(ys):
        for xi, seq_len in enumerate(xs):
            vals = pd.to_numeric(use.loc[use["seq_len"].eq(seq_len) & use["delay_ms"].eq(delay), "value"], errors="coerce").dropna()
            if not vals.empty:
                mat[yi, xi] = float(vals.mean())
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if spec.get("vmin", spec.get("color_vmin", None)) is not None:
        try:
            vmin = float(spec.get("vmin", spec.get("color_vmin")))
        except Exception:
            pass
    if spec.get("vmax", spec.get("color_vmax", None)) is not None:
        try:
            vmax = float(spec.get("vmax", spec.get("color_vmax")))
        except Exception:
            pass
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    image = ax.imshow(mat, origin="lower", aspect="auto", cmap=get_plot_cmap("sequential"), vmin=vmin, vmax=vmax, interpolation="nearest")
    for yi in range(len(ys)):
        for xi in range(len(xs)):
            val = mat[yi, xi]
            if np.isfinite(val):
                ax.text(xi, yi, f"{val:.2g}", ha="center", va="center", fontsize=5.0, color="white" if val > (vmin + vmax) / 2 else "0.15")
    ax.set_xticks(np.arange(len(xs)), [str(int(x)) for x in xs])
    ax.set_yticks(np.arange(len(ys)), [str(int(y)) for y in ys])
    ax.set_xlabel("K")
    ax.set_ylabel("Delay (ms)")
    cax = getattr(ax, "paper_fig_colorbar_ax", None)
    if cax is not None:
        cbar = ax.figure.colorbar(image, cax=cax)
        cbar.ax.tick_params(labelsize=5.0, length=1.4, width=0.45, pad=1.0)
        cbar.set_label(metric.replace("_", " "), fontsize=5.4, labelpad=1.2)
        ax.paper_fig_has_colorbar = True
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_boundary_heatmap"
    ax.paper_fig_y_metric = metric


def render_fig3_morphology_function_coupling(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty or not {"morphology_support_p", "functional_gain_norm"}.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    work = df.copy()
    work["morphology_support_p"] = pd.to_numeric(work["morphology_support_p"], errors="coerce")
    work["functional_gain_norm"] = pd.to_numeric(work["functional_gain_norm"], errors="coerce")
    work = work.dropna(subset=["morphology_support_p", "functional_gain_norm"])
    if work.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    x = work["morphology_support_p"].to_numpy(dtype=float)
    y = work["functional_gain_norm"].to_numpy(dtype=float)
    pos = pd.to_numeric(work.get("serial_position", pd.Series(np.arange(len(work)))), errors="coerce").fillna(0).to_numpy(dtype=float)
    sc = ax.scatter(x, y, c=pos, cmap=get_plot_cmap("sequential"), s=14, alpha=0.72, edgecolors="white", linewidths=0.25)
    if x.size >= 2 and float(np.nanstd(x)) > 1e-12:
        coef = np.polyfit(x, y, deg=1)
        xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 50)
        ax.plot(xx, coef[0] * xx + coef[1], color="0.18", linewidth=0.9, alpha=0.82)
    ax.set_xlabel("Morphology support")
    ax.set_ylabel("Access gain")
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    xpad = max(0.03, (xmax - xmin) * 0.12)
    ypad = max(0.03, (ymax - ymin) * 0.12)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    cax = getattr(ax, "paper_fig_colorbar_ax", None)
    if cax is not None:
        cbar = ax.figure.colorbar(sc, cax=cax)
        cbar.ax.tick_params(labelsize=5.0, length=1.4, width=0.45, pad=1.0)
        cbar.set_label("Serial position", fontsize=5.4, labelpad=1.2)
        ax.paper_fig_has_colorbar = True
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_morphology_function_coupling"
    ax.paper_fig_y_metric = "G_i_norm"


def _render_landscape_heatmap(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats, style
    df = _clean(panel_data)
    if df.empty or not {"row", "col", "value"}.issubset(df.columns):
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    mat, _, _ = _matrix(df)
    finite = mat[np.isfinite(mat)]
    vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    image = ax.imshow(mat, origin="upper", cmap=get_plot_cmap("difference"), vmin=-vmax, vmax=vmax, interpolation="nearest")
    for role, color, marker in (("peak", PEAK, "o"), ("valley", VALLEY, "s")):
        pts = df[df.get("mask_role", pd.Series(dtype=str)).astype(str).eq(role)]
        if not pts.empty:
            ax.scatter(pts["col"], pts["row"], s=4.5, facecolors="none", edgecolors=color, linewidths=0.35, marker=marker)
    ax.set_xticks([])
    ax.set_yticks([])
    cax = getattr(ax, "paper_fig_colorbar_ax", None)
    if cax is not None:
        cbar = ax.figure.colorbar(image, cax=cax)
        cbar.set_ticks([])
        cbar.ax.set_yticklabels([])
        cbar.ax.tick_params(labelsize=0, length=1.2, width=0.4, pad=0)
        ax.paper_fig_has_colorbar = True
    _title(ax, spec)


def _clean(panel_data: pd.DataFrame | None) -> pd.DataFrame:
    if panel_data is None or panel_data.empty or "value" not in panel_data.columns:
        return pd.DataFrame()
    df = panel_data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def _matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = df.copy()
    work["row"] = pd.to_numeric(work["row"], errors="coerce").astype(int)
    work["col"] = pd.to_numeric(work["col"], errors="coerce").astype(int)
    rows = np.arange(int(work["row"].max()) + 1)
    cols = np.arange(int(work["col"].max()) + 1)
    mat = np.full((len(rows), len(cols)), np.nan)
    for _, row in work.iterrows():
        mat[int(row["row"]), int(row["col"])] = float(row["value"])
    return mat, rows, cols


def _summary(df: pd.DataFrame, x_col: str, value_col: str) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for x, part in df.groupby(x_col, sort=True):
        vals = pd.to_numeric(part[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size:
            rows.append({"x": float(x), "mean": float(vals.mean()), "sem": _sem(vals)})
    return pd.DataFrame(rows).sort_values("x") if rows else pd.DataFrame(columns=["x", "mean", "sem"])


def _infer_seq_len(df: pd.DataFrame) -> int:
    vals = pd.to_numeric(df.get("seq_len", pd.Series(dtype=float)), errors="coerce").dropna()
    if not vals.empty:
        return int(vals.max())
    serial = pd.to_numeric(df.get("serial_position", pd.Series(dtype=float)), errors="coerce").dropna()
    return int(serial.max()) if not serial.empty else 10


def _region_ping_categories(df: pd.DataFrame, region_order: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for region in region_order:
        part = df[df["region_condition"].astype(str).eq(region)].copy()
        cat_mass = {key: 0.0 for key in ["old", "recent", "silent"]}
        for _, row in part.iterrows():
            mass = float(row.get("value", 0.0))
            cat = str(row.get("readout_category", "")).strip().lower()
            if cat not in cat_mass:
                cat = "silent"
            cat_mass[cat] += mass
        out[region] = cat_mass
    return out


def _serial_position(value: Any) -> int | None:
    text = str(value).strip()
    if text.startswith("pos_"):
        text = text.split("_", 1)[1]
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return int(numeric) if pd.notna(numeric) else None


def _sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def _norm_cue(value: Any) -> str:
    text = str(value).strip().lower()
    return {"valley_aligned": "valley", "random_matched": "random", "peak_aligned": "peak"}.get(text, text)


def _title(ax, spec: Mapping[str, Any]) -> None:
    title = str(spec.get("title", "")).strip()
    if title:
        ax.set_title(title, fontsize=6.7, pad=1.8)


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.35, alpha=0.65)


def _compact(ax) -> None:
    ax.tick_params(axis="both", labelsize=5.4, pad=1.0, length=2.0, width=0.55, color=COLOR_NEUTRAL)
    ax.xaxis.label.set_size(6.0)
    ax.yaxis.label.set_size(6.0)
    ax.xaxis.labelpad = 1.0
    ax.yaxis.labelpad = 1.0


# Compatibility aliases for older renderer names.
render_fig3_example_landscape_3d = render_fig3_3d_landscape
render_fig3_peak_valley_landscape = render_fig3_3d_landscape
render_fig3_morphology_inset = render_fig3_3d_landscape
render_fig3_neutral_ping_distribution = render_fig3_neutral_ping_serial_profile
render_fig3_neutral_ping = render_fig3_neutral_ping_serial_profile
render_fig3_structural_weak_cue = render_fig3_peak_cue_memory_gain
render_fig3_region_ping_readout_adapter = render_fig3_region_ping_readout
render_fig3_peak_aligned_completion = render_fig3_peak_cue_memory_gain
render_two_item_morphology = render_fig3_progressive_update
render_two_item_readout = render_fig3_neutral_ping_serial_profile
render_multiitem_profile = render_fig3_3d_landscape
render_progressive_update = render_fig3_progressive_update
render_center_migration = render_fig3_neutral_ping_serial_profile
render_fig3_structural_weak_cue_classification = render_fig3_peak_cue_memory_gain
render_fig3_neutral_ping_readout_distribution = render_fig3_neutral_ping_serial_profile
