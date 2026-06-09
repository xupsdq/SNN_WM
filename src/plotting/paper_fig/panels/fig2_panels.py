from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


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
STATE_LABELS = {"S0": "No\nmemory", "S_A": "S_A", "S_B": "S_B", "S_AB": "S_AB"}
STATE_COLORS = {"S0": "#8C8C8C", "S_A": "#4C78A8", "S_B": "#F58518", "S_AB": "#54A24B"}
READOUT_COLORS = {"A": "#4C78A8", "B": "#F58518", "Other": "#B8B8B8", "Silent": "#F4F4F4"}
COMPOSITION_LABELS = {"Other": "others", "A": "A", "B": "B", "Silent": "silent"}


def render_fig2_episode_schematic(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data, stats, style
    ax.set_axis_off()
    ax.paper_fig_plot_form = "blank_manual_slot"


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
    _boxplot_by_condition(ax, df, "condition", order, colors=["#4C78A8", "#BAB0AC"], st=st)
    ax.set_xticks(np.arange(len(order)), ["True", "Shuffled"])
    ax.set_ylabel(str(spec.get("y_axis", "Pair specificity")))
    ax.set_xlabel("")
    ax.paper_fig_raw_points = False
    _autoscale_y(ax, df["value"])
    _tidy(ax, st)


def render_fig2_morphology_closure(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, "Morphology closure data unavailable")
        return
    ax.paper_fig_plot_form = "positive_effect_bar_summary_against_zero"
    order = ["WPRI", "Beyond-linear"]
    plot_df = df[df["condition"].astype(str).isin(order)].copy()
    if plot_df.empty:
        _placeholder(ax, spec, "Morphology closure metrics unavailable")
        return
    _bar_summary(ax, plot_df, "condition", order, colors=["#54A24B", "#B279A2"], st=st, alpha=0.83)
    ax.axhline(0, color="0.35", linestyle="--", linewidth=0.72)
    ax.set_xticks(np.arange(len(order)), ["WPRI", "Beyond\nlinear"])
    limits = spec.get("y_limits") or spec.get("y_axis_limits") or [-0.4, 0.4]
    if isinstance(limits, (list, tuple)) and len(limits) == 2:
        ax.set_ylim(float(limits[0]), float(limits[1]))
    else:
        ax.set_ylim(-0.4, 0.4)
    ax.set_ylabel(str(spec.get("y_axis", "Score")))
    ax.set_xlabel("")
    ax.paper_fig_x_metric = "Metric"
    ax.paper_fig_y_metric = "Score"
    ax.paper_fig_raw_points = False
    _tidy(ax, st)


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
    span_w_mm = float((spec.get("size_mm") or {}).get("width", (spec.get("position_mm") or {}).get("w", 94.67)))
    col_w_mm = 42.33
    gap_w_mm = 10.0
    col_frac = col_w_mm / span_w_mm
    gap_frac = gap_w_mm / span_w_mm
    left_bounds = [0.0, 0.0, col_frac, 1.0]
    right_bounds = [col_frac + gap_frac, 0.0, col_frac, 1.0]
    left = ax.inset_axes(left_bounds)
    right = ax.inset_axes(right_bounds)
    right.sharey(left)
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
            loc="lower center",
            bbox_to_anchor=(0.5, 1.04),
            handlelength=1.0,
            handletextpad=0.35,
            columnspacing=0.7,
            borderaxespad=0.0,
        )
        ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
        ax.paper_fig_legend_ncols = 4


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
    ax.set_xticks([0.0, 0.5, 1.0], ["0", ".5", "1"])
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
    ax.grid(axis="y", color="0.9", linewidth=0.45)
    ax.set_axisbelow(True)


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
