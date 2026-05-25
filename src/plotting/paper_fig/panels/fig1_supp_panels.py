from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.plotting.common.colors import get_plot_color
from src.plotting.paper_fig.panels.fig1_panels import (
    CONDITION_COLORS,
    LAYER_COLORS,
    STYLE,
    _clean,
    _group_means,
    _placeholder,
    _reference_lines,
    _scatter_points,
    _style,
    _tidy,
    plt_rect,
)
from src.plotting.paper_fig.utils import paper_fig_root


PHASE_ORDER = ["stimulus", "sample", "early_delay", "late_delay", "delay", "probe"]


def render_supp_temporal_encoding_schematic(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data, stats, style
    ax.set_axis_off()
    asset = spec.get("source") or (spec.get("source_mapping") or {}).get("manual_asset")
    if _asset_exists(asset):
        ax.text(0.5, 0.5, f"Manual schematic asset\n{asset}", ha="center", va="center", fontsize=7.0, transform=ax.transAxes)
        ax.paper_fig_plot_form = "manual_schematic_asset_slot"
        return
    ax.paper_fig_plot_form = "programmatic_temporal_encoding_schematic"
    _draw_flow(ax, ["MNIST", "DoG\nON/OFF", "Latency\nspikes", "Gamma\nwindow"], ["#F2F2F2", "#DDEEFF", "#FFF2B2", "#E8DDF5"])


def render_supp_stsp_update_schematic(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = panel_data, stats, style
    ax.set_axis_off()
    asset = spec.get("source") or (spec.get("source_mapping") or {}).get("manual_asset")
    if _asset_exists(asset):
        ax.text(0.5, 0.5, f"Manual schematic asset\n{asset}", ha="center", va="center", fontsize=7.0, transform=ax.transAxes)
        ax.paper_fig_plot_form = "manual_schematic_asset_slot"
        return
    ax.paper_fig_plot_form = "programmatic_stsp_update_schematic"
    _draw_flow(ax, ["Layer\ninput", "u update", "x update", "g = u*x", "Delay\nstate"], ["#F2F2F2", "#DCECC9", "#DDEEFF", "#FFF2B2", "#E8DDF5"])
    ax.text(0.5, 0.14, "trial-specific u/x retained across delay", ha="center", va="center", fontsize=5.8, transform=ax.transAxes, color="0.25")


def render_class_recall_by_digit(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "digit_class" not in df.columns:
        _placeholder(ax, spec, "Class recall data unavailable")
        return
    ax.paper_fig_plot_form = "supp_class_recall_by_digit"
    order = [str(i) for i in range(10) if str(i) in set(df["digit_class"].astype(str))]
    if not order:
        order = sorted(df["digit_class"].astype(str).unique().tolist())
    x = np.arange(len(order), dtype=float)
    means, sems = _group_means(df, "digit_class", order)
    ax.bar(x, means, yerr=sems, capsize=st["capsize"], width=0.72, color="#4C78A8", edgecolor="black", linewidth=0.4, alpha=0.86)
    _scatter_points(ax, df, "digit_class", order, x)
    _reference_lines(ax, spec)
    ax.set_xticks(x, order)
    ax.set_xlabel(str(spec.get("x_axis", "Digit class")))
    ax.set_ylabel(str(spec.get("y_axis", "Recall (%)")))
    ax.set_ylim(0, 100)
    _tidy(ax, st)


def render_confusion_matrix(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = style
    df = _clean(panel_data)
    if df.empty or not {"true_label", "pred_label"}.issubset(df.columns):
        _placeholder(ax, spec, "Confusion matrix data unavailable")
        return
    ax.paper_fig_plot_form = "supp_confusion_matrix"
    labels = list((stats or {}).get("labels") or _sorted_labels(set(df["true_label"].astype(str)).union(df["pred_label"].astype(str))))
    matrix = pd.DataFrame(0.0, index=labels, columns=labels)
    summary = df.groupby(["true_label", "pred_label"], as_index=False)["value"].mean()
    for _, row in summary.iterrows():
        true_label = str(row["true_label"])
        pred_label = str(row["pred_label"])
        if true_label in matrix.index and pred_label in matrix.columns:
            matrix.loc[true_label, pred_label] = float(row["value"])
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=max(100, float(matrix.to_numpy().max() if matrix.size else 100)), aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=90)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel(str(spec.get("x_axis", "Predicted class")))
    ax.set_ylabel(str(spec.get("y_axis", "True class")))
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=5.0, width=0.4, length=1.6)
    cbar.set_label("%", fontsize=5.2)
    ax.paper_fig_has_colorbar = True
    ax.paper_fig_colorbar_ax = cbar.ax
    _tidy(ax, _style(style))


def render_phase_firing_rates(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or not {"layer", "phase"}.issubset(df.columns):
        _placeholder(ax, spec, "Phase firing data unavailable")
        return
    ax.paper_fig_plot_form = "supp_phase_firing_rates"
    phases = [phase for phase in PHASE_ORDER if phase in set(df["phase"].astype(str))]
    if not phases:
        phases = df["phase"].astype(str).drop_duplicates().tolist()
    x = np.arange(len(phases), dtype=float)
    for layer in [layer for layer in ("layer1", "layer2", "layer3") if layer in set(df["layer"].astype(str))]:
        part = df[df["layer"].astype(str).eq(layer)]
        means, sems = _group_means(part, "phase", phases)
        ax.plot(x, means, marker="o", markersize=2.7, linewidth=0.9, color=LAYER_COLORS.get(layer, "0.4"), label=(spec.get("display_labels") or {}).get(layer, layer))
        if len(part["seed_id"].dropna().unique()) > 1:
            ax.fill_between(x, means - sems, means + sems, color=LAYER_COLORS.get(layer, "0.4"), alpha=0.16, linewidth=0)
    ax.set_xticks(x, [_short_phase_label(p) for p in phases], rotation=25, ha="right")
    ax.set_xlabel(str(spec.get("x_axis", "Phase")))
    ax.set_ylabel(str(spec.get("y_axis", "Spike rate (Hz)")))
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="best", handlelength=1.0)
    _tidy(ax, st)


def render_delay_decode_timecourse(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "delay_ms" not in df.columns:
        _placeholder(ax, spec, "Delay decoding curve unavailable")
        return
    ax.paper_fig_plot_form = "supp_delay_decode_timecourse"
    df = df.copy()
    df["delay_ms"] = pd.to_numeric(df["delay_ms"], errors="coerce")
    for layer in [layer for layer in ("layer1", "layer2", "layer3") if layer in set(df["layer"].astype(str))]:
        part = df[df["layer"].astype(str).eq(layer)].dropna(subset=["delay_ms"])
        grouped = part.groupby("delay_ms", as_index=False)["value"].agg(["mean", "sem"]).reset_index()
        x = grouped["delay_ms"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        sem = grouped["sem"].fillna(0).to_numpy(dtype=float)
        color = LAYER_COLORS.get(layer, "0.4")
        ax.plot(x, y, marker="o", markersize=2.7, linewidth=0.9, color=color, label=(spec.get("display_labels") or {}).get(layer, layer))
        if len(part["seed_id"].dropna().unique()) > 1:
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.16, linewidth=0)
    _reference_lines(ax, spec)
    ax.set_xlabel(str(spec.get("x_axis", "Delay (ms)")))
    ax.set_ylabel(str(spec.get("y_axis", "Decoding accuracy (%)")))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="best", handlelength=1.0)
    _tidy(ax, st)


def render_dms_delay_probe_accuracy(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    _line_by_condition(ax, panel_data, spec, style, y_default="Probe accuracy (%)", plot_form="supp_dms_delay_probe_accuracy")


def render_stsp_interference_delay(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "delay_ms" not in df.columns:
        _placeholder(ax, spec, "STSP interference data unavailable")
        return
    ax.paper_fig_plot_form = "supp_static_minus_dynamic_delay"
    df = df.copy()
    df["delay_ms"] = pd.to_numeric(df["delay_ms"], errors="coerce")
    grouped = df.dropna(subset=["delay_ms"]).groupby("delay_ms", as_index=False)["value"].agg(["mean", "sem"]).reset_index()
    x = grouped["delay_ms"].to_numpy(dtype=float)
    y = grouped["mean"].to_numpy(dtype=float)
    sem = grouped["sem"].fillna(0).to_numpy(dtype=float)
    ax.axhline(0, color="0.45", linewidth=0.65, linestyle="--")
    ax.plot(x, y, marker="o", markersize=2.8, linewidth=0.95, color=get_plot_color("trial_shuffled_ux"))
    if len(df["seed_id"].dropna().unique()) > 1:
        ax.fill_between(x, y - sem, y + sem, color=get_plot_color("trial_shuffled_ux"), alpha=0.16, linewidth=0)
    ax.set_xlabel(str(spec.get("x_axis", "Delay (ms)")))
    ax.set_ylabel(str(spec.get("y_axis", "Static - dynamic accuracy (%)")))
    _tidy(ax, st)


def render_substrate_shuffle_specificity(ax, panel_data: pd.DataFrame | None, stats: Mapping[str, Any] | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None = None) -> None:
    _ = stats
    st = _style(style)
    df = _clean(panel_data)
    if df.empty:
        _placeholder(ax, spec, "Substrate specificity data unavailable")
        return
    ax.paper_fig_plot_form = "supp_substrate_shuffle_specificity"
    order = [condition for condition in spec.get("conditions", []) if condition in set(df["condition"].astype(str))]
    if not order:
        order = df["condition"].astype(str).drop_duplicates().tolist()
    x = np.arange(len(order), dtype=float)
    means, sems = _group_means(df, "condition", order)
    colors = [CONDITION_COLORS.get(c, get_plot_color(c, default="0.65")) for c in order]
    ax.axhline(0, color="0.45", linewidth=0.65, linestyle="--")
    ax.bar(x, means, yerr=sems, capsize=st["capsize"], width=0.68, color=colors, edgecolor="black", linewidth=0.4, alpha=0.86)
    _scatter_points(ax, df, "condition", order, x)
    labels = [(spec.get("display_labels") or {}).get(c, c) for c in order]
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_xlabel(str(spec.get("x_axis", "")))
    ax.set_ylabel(str(spec.get("y_axis", "Donor attribution gain (%)")))
    _tidy(ax, st)


def _line_by_condition(ax, panel_data: pd.DataFrame | None, spec: Mapping[str, Any], style: Mapping[str, Any] | None, *, y_default: str, plot_form: str) -> None:
    st = _style(style)
    df = _clean(panel_data)
    if df.empty or "delay_ms" not in df.columns:
        _placeholder(ax, spec, "Delay sweep data unavailable")
        return
    ax.paper_fig_plot_form = plot_form
    df = df.copy()
    df["delay_ms"] = pd.to_numeric(df["delay_ms"], errors="coerce")
    conditions = [condition for condition in spec.get("conditions", []) if condition in set(df["condition"].astype(str))]
    for condition in conditions:
        part = df[df["condition"].astype(str).eq(condition)].dropna(subset=["delay_ms"])
        grouped = part.groupby("delay_ms", as_index=False)["value"].agg(["mean", "sem"]).reset_index()
        x = grouped["delay_ms"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        sem = grouped["sem"].fillna(0).to_numpy(dtype=float)
        color = CONDITION_COLORS.get(condition, get_plot_color(condition, default="0.45"))
        ax.plot(x, y, marker="o", markersize=2.7, linewidth=0.95, color=color, label=(spec.get("display_labels") or {}).get(condition, condition))
        if len(part["seed_id"].dropna().unique()) > 1:
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel(str(spec.get("x_axis", "Delay (ms)")))
    ax.set_ylabel(str(spec.get("y_axis", y_default)))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=st["legend_fontsize"], loc="best", handlelength=1.0)
    _tidy(ax, st)


def _asset_exists(asset: Any) -> bool:
    if not asset:
        return False
    path = paper_fig_root() / str(asset)
    return path.exists()


def _draw_flow(ax, labels: Sequence[str], colors: Sequence[str]) -> None:
    xs = np.linspace(0.11, 0.89, len(labels))
    for idx, (x, label) in enumerate(zip(xs, labels)):
        ax.add_patch(plt_rect(ax, (x - 0.072, 0.43), 0.144, 0.24, facecolor=colors[idx % len(colors)], edgecolor="0.25", linewidth=0.65))
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=6.5, transform=ax.transAxes)
        if idx < len(labels) - 1:
            ax.annotate("", xy=(xs[idx + 1] - 0.085, 0.55), xytext=(x + 0.085, 0.55), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "linewidth": 0.75, "color": "0.25"})


def _short_phase_label(phase: str) -> str:
    return {
        "stimulus": "stim",
        "sample": "sample",
        "early_delay": "early",
        "late_delay": "late",
        "delay": "delay",
        "probe": "probe",
    }.get(phase, phase)


def _sorted_labels(labels: set[str]) -> list[str]:
    def key(label: str) -> tuple[int, Any]:
        if label == "silent":
            return (1, 99)
        numeric = pd.to_numeric(pd.Series([label]), errors="coerce").iloc[0]
        return (0, int(numeric)) if pd.notna(numeric) else (1, label)

    return sorted(labels, key=key)
