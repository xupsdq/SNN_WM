from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm

from src.plotting.common.colors import get_plot_cmap, get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL
from src.plotting.paper_fig.panels.fig1_panels import render_generic_placeholder


MAIN = get_plot_color("true_pair")
PEAK = "#D55E00"
RANDOM = "#6A6A6A"
VALLEY = "#0072B2"
GRID = "0.86"
STATE_ORDER = ("S_final", "S0", "S0_ping_null")


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
    for _, part in use.groupby("sequence_id", sort=False):
        part = part.sort_values("stage_k")
        ax.plot(part["stage_k"], part["value"], color=MAIN, alpha=0.12, linewidth=0.55)
    summary = _summary(use, "stage_k", "value")
    if not summary.empty:
        ax.fill_between(summary["x"], summary["mean"] - summary["sem"], summary["mean"] + summary["sem"], color=MAIN, alpha=0.16, linewidth=0)
        ax.plot(summary["x"], summary["mean"], color=MAIN, linewidth=1.35, marker="o", markersize=2.6)
    ax.set_xlabel("Sequence stage")
    ax.set_ylabel("Update ratio")
    ax.set_xlim(1, max(3, float(np.nanmax(use["stage_k"]))) + 0.25)
    _title(ax, spec)
    _tidy(ax)
    _compact(ax)
    ax.paper_fig_plot_form = "fig3_progressive_update"


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
        diverging = float(np.nanmin(finite)) < 0 < float(np.nanmax(finite)) or str((stats or {}).get("landscape_metric_used", "")).startswith("delta")
        if diverging:
            vmax = float(np.nanpercentile(np.abs(finite), 98)) or 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            cmap = get_plot_cmap("difference")
            cbar_label = "Δ STSP support"
        else:
            norm = Normalize(vmin=float(np.nanmin(finite)), vmax=float(np.nanmax(finite)) or 1.0)
            cmap = cm.viridis
            cbar_label = "Support"
        colors = cmap(norm(mat))
        surface = ax.plot_surface(xx, yy, mat, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False, alpha=0.96)
        surface.set_array(finite)
        surface.set_cmap(cmap)
        surface.set_norm(norm)
        for role, color, marker in (("peak", PEAK, "o"), ("valley", VALLEY, "^"), ("random", RANDOM, "s")):
            pts = df[df.get("mask_role", pd.Series(dtype=str)).astype(str).eq(role)].copy()
            if not pts.empty:
                pts["row"] = pd.to_numeric(pts["row"], errors="coerce")
                pts["col"] = pd.to_numeric(pts["col"], errors="coerce")
                pts["value"] = pd.to_numeric(pts["value"], errors="coerce")
                pts = pts.iloc[:: max(1, int(len(pts) // 32))]
                ax.scatter(pts["col"], pts["row"], pts["value"] + 0.012 * (np.nanmax(finite) - np.nanmin(finite) + 1e-9), s=3.6, color=color, marker=marker, depthshade=False, alpha=0.50)
        ax.view_init(elev=34, azim=-48)
        try:
            ax.set_box_aspect((1.05, 0.95, 0.20), zoom=0.58)
        except TypeError:
            try:
                ax.set_box_aspect((1.05, 0.95, 0.20))
                ax.dist = 14
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
        ax.set_zlim(float(np.nanmin(finite)), float(np.nanmax(finite)))
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
            cbar.set_ticks([])
            cbar.ax.set_yticklabels([])
            cbar.ax.tick_params(labelsize=0, length=1.2, width=0.4, pad=0)
            cbar.set_label(cbar_label, fontsize=4.8, labelpad=-1.0)
            ax.paper_fig_has_colorbar = True
            ax.paper_fig_colorbar_label = cbar_label
        title = str(spec.get("title", "")).strip()
        if title:
            pos = ax.get_position()
            ax.figure.text((pos.x0 + pos.x1) / 2.0, pos.y1 + 0.008, title, ha="center", va="bottom", fontsize=6.7)
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


def render_fig3_neutral_ping_serial_profile(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = style
    df = _clean(panel_data)
    use = df[df["metric"].astype(str).eq("readout_mass")].copy() if "metric" in df.columns else pd.DataFrame()
    if use.empty or "serial_position" not in use.columns:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    use["serial_position"] = pd.to_numeric(use["serial_position"], errors="coerce")
    use = use.dropna(subset=["serial_position"])
    if use.empty:
        render_generic_placeholder(ax, panel_data, stats, spec, style)
        return
    colors = {"S_final": MAIN, "S0": "0.45", "S0_ping_null": "0.55"}
    states = [s for s in ("S_final", "S0") if s in set(use.get("state_condition", use.get("condition")).astype(str))]
    for state in states:
        part = use[use.get("state_condition", use.get("condition")).astype(str).eq(state)]
        if part.empty:
            continue
        summary = _summary(part, "serial_position", "value")
        label = {"S_final": "S final", "S0": "S0"}.get(state, state.replace("_", " "))
        ax.plot(summary["x"], summary["mean"], color=colors.get(state, "0.25"), linewidth=1.3, marker="o", markersize=2.6, label=label)
        ax.fill_between(summary["x"], summary["mean"] - summary["sem"], summary["mean"] + summary["sem"], color=colors.get(state, "0.25"), alpha=0.12, linewidth=0)
    max_pos = int(np.nanmax(use["serial_position"]))
    ticks = list(range(1, max_pos + 1))
    ax.set_xticks(ticks if max_pos <= 10 else ticks[::2])
    ax.set_xlim(0.75, max_pos + 0.25)
    ymax = min(1.0, max(0.25, float(use["value"].max()) * 1.2))
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Serial position")
    ax.set_ylabel("Readout mass")
    ax.legend(frameon=False, fontsize=5.1, loc="upper right", handlelength=1.0, borderaxespad=0.2)
    _title(ax, spec)
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
    colors = {"cue_only": "0.45", "single_item_memory": "#CC79A7", "sequence_state": MAIN}
    labels = {"cue_only": "No memory", "single_item_memory": "Single-item memory", "sequence_state": "Sequence state"}
    for condition in ("cue_only", "single_item_memory", "sequence_state"):
        part = use[use["memory_condition"].astype(str).eq(condition)] if "memory_condition" in use.columns else pd.DataFrame()
        if part.empty:
            continue
        summary = _summary(part, "keep_prob", "value")
        ax.plot(summary["x"], summary["mean"], color=colors[condition], linewidth=1.35, marker="o", markersize=2.6, label=labels[condition])
        ax.fill_between(summary["x"], summary["mean"] - summary["sem"], summary["mean"] + summary["sem"], color=colors[condition], alpha=0.12, linewidth=0)
    ax.set_xlabel("Weak-probe keep probability")
    ax.set_ylabel("Target recovery (%)")
    xmax = float(pd.to_numeric(use["keep_prob"], errors="coerce").max())
    keep_ticks = sorted(pd.to_numeric(use["keep_prob"], errors="coerce").dropna().unique().tolist())
    if 1 <= len(keep_ticks) <= 8:
        ax.set_xticks(keep_ticks)
    ax.set_xlim(0, max(0.35, xmax * 1.14))
    ymax = max(100.0, float(pd.to_numeric(use["value"], errors="coerce").max()) * 1.08)
    ax.set_ylim(0, ymax)
    ax.legend(frameon=False, fontsize=4.9, loc="lower right", handlelength=1.0, borderaxespad=0.2)
    _title(ax, spec)
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
