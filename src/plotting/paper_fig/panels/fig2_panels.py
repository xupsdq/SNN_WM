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


def render_fig2_episode_schematic(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data, stats, style
    ax.set_axis_off()
    ax.paper_fig_plot_form = "programmatic_two_item_episode_schematic"
    xs = [0.08, 0.24, 0.39, 0.55, 0.70]
    labels = ["Item A", "delay", "Item B", "delay", "S_AB"]
    colors = ["#E8F1FB", "#F5F5F5", "#FFF0DE", "#F5F5F5", "#EAF6EA"]
    for i, (x, label, color) in enumerate(zip(xs, labels, colors)):
        ax.add_patch(_rect(ax, x - 0.045, 0.50, 0.09, 0.24, facecolor=color, edgecolor="0.25", linewidth=0.7))
        ax.text(x, 0.62, label, ha="center", va="center", fontsize=7.0, transform=ax.transAxes)
        if i < len(xs) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.052, 0.62), xytext=(x + 0.052, 0.62), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "0.25"})
    ref_xs = [0.18, 0.31, 0.44, 0.57]
    for x, label, color in zip(ref_xs, ["S0", "S_A", "S_B", "S_AB"], ["#F4F4F4", STATE_COLORS["S_A"], STATE_COLORS["S_B"], STATE_COLORS["S_AB"]]):
        ax.add_patch(_rect(ax, x - 0.037, 0.20, 0.074, 0.18, facecolor=color, edgecolor="0.35", linewidth=0.55, alpha=0.78))
        ax.text(x, 0.29, label, ha="center", va="center", fontsize=6.3, transform=ax.transAxes)
    branch_x = 0.84
    ax.text(branch_x, 0.68, "Morphology", ha="center", va="center", fontsize=6.5, transform=ax.transAxes)
    ax.text(branch_x, 0.42, "Functional\naccess", ha="center", va="center", fontsize=6.5, transform=ax.transAxes)
    ax.annotate("", xy=(branch_x - 0.08, 0.68), xytext=(0.75, 0.62), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "linewidth": 0.75, "color": "0.3"})
    ax.annotate("", xy=(branch_x - 0.08, 0.42), xytext=(0.75, 0.58), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "linewidth": 0.75, "color": "0.3"})


def render_fig2_dual_retention_constituents(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, "Dual retention data unavailable")
        return
    ax.paper_fig_plot_form = "dual_retention_constituent_similarity"
    order = ["S_A", "S_B"]
    _paired_lines(ax, df, order, key_col="pair_id", color="0.3")
    _bar_scatter(ax, df, "condition", order, colors=[STATE_COLORS["S_A"], STATE_COLORS["S_B"]], st=st, alpha=0.82)
    ax.set_xticks(np.arange(len(order)), [_display(spec, v) for v in order])
    ax.set_ylabel(str(spec.get("y_axis", "Similarity to S_AB")))
    ax.set_xlabel("")
    ax.paper_fig_visual_categories = order
    _autoscale_y(ax, df["value"])
    _tidy(ax, st)


def render_fig2_pair_specificity(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, "Pair specificity data unavailable")
        return
    ax.paper_fig_plot_form = "true_vs_shuffled_pair_plot"
    order = ["True pair", "Shuffled pair"]
    _paired_lines(ax, df, order, key_col="pair_id", color="#4C78A8")
    _bar_scatter(ax, df, "condition", order, colors=["#4C78A8", "#BAB0AC"], st=st, alpha=0.55)
    ax.set_xticks(np.arange(len(order)), ["True", "Shuffled"])
    ax.set_ylabel(str(spec.get("y_axis", "Pair specificity")))
    ax.set_xlabel("")
    _autoscale_y(ax, df["value"])
    _tidy(ax, st)


def render_fig2_morphology_closure(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, "Morphology closure data unavailable")
        return
    ax.paper_fig_plot_form = "morphology_closure_summary"
    order = ["WPRI", "Beyond-linear"]
    plot_df = df[df["condition"].astype(str).isin(order)].copy()
    if plot_df.empty:
        _placeholder(ax, spec, "Morphology closure metrics unavailable")
        return
    _bar_scatter(ax, plot_df, "condition", order, colors=["#54A24B", "#B279A2"], st=st, alpha=0.83)
    ax.axhline(0, color="0.35", linestyle="--", linewidth=0.65)
    ax.set_xticks(np.arange(len(order)), ["WPRI", "Beyond\nlinear"])
    ax.set_ylabel(str(spec.get("y_axis", "Score")))
    ax.set_xlabel("")
    ax.paper_fig_x_metric = "Metric"
    ax.paper_fig_y_metric = "Score"
    _autoscale_y(ax, plot_df["value"], include_zero=True)
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
    categories = [c for c in ["A", "B", "Other", "Silent"] if c in set(df["category"].astype(str))]
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
        ax.bar(x, values, bottom=bottom, width=st["bar_width"], color=color, edgecolor=edge, linewidth=0.35, label=category)
        bottom += np.asarray(values, dtype=float)
    ax.set_xticks(x, [_display(spec, c) for c in conditions])
    ax.set_ylim(0, 100)
    ax.set_ylabel(str(spec.get("y_axis", "Readout composition (%)")))
    ax.set_xlabel("")
    legend = ax.legend(frameon=False, fontsize=st["legend_fontsize"], ncol=2, loc="upper left", bbox_to_anchor=(0.00, 1.00), handlelength=0.9, handletextpad=0.35, columnspacing=0.75, borderaxespad=0.0)
    ax.paper_fig_legend_texts = [text.get_text() for text in legend.get_texts()]
    ax.paper_fig_legend_ncols = 2
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
            ax.errorbar(x, y, yerr=sem, fmt="none", ecolor=color, elinewidth=0.45, capsize=1.4, alpha=0.45)
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
