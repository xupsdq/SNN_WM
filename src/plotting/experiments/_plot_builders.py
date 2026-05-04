from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.plotting.common.colors import get_plot_cmap, get_plot_color, infer_plot_cmap_kind, resolve_plot_color
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    ALPHA_FILL,
    ALPHA_SCATTER,
    GRID_ALPHA_SOFT,
    LINE_WIDTH_PRIMARY,
    LINE_WIDTH_REFERENCE,
    MARKER_CIRCLE,
)


def sem(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def color_for(label: object, *, context: str | None = None) -> str:
    return get_plot_color(label, context=context)


def grouped_mean_sem(df: pd.DataFrame, group_cols: Sequence[str], value_col: str) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols), sort=True)[value_col]
    out = grouped.agg(["mean", "count", "std"]).reset_index()
    out["sem"] = out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    return out


def line_figure(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    hue: str | None = None,
    yerr: str | None = None,
    context: str | None = None,
    color_key: object | None = None,
    figsize: tuple[float, float] = (6.4, 4.6),
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    elif hue and hue in df.columns:
        for label, sub in df.groupby(hue, sort=True):
            sub = sub.sort_values(x)
            err = sub[yerr].to_numpy(dtype=np.float64) if yerr and yerr in sub.columns else None
            ax.errorbar(
                sub[x].to_numpy(),
                sub[y].to_numpy(dtype=np.float64),
                yerr=err,
                marker=MARKER_CIRCLE,
                linewidth=LINE_WIDTH_PRIMARY,
                color=resolve_plot_color(label, hue, context=context),
                label=str(label),
            )
        ax.legend(frameon=False)
    else:
        plot_df = df.sort_values(x)
        err = plot_df[yerr].to_numpy(dtype=np.float64) if yerr and yerr in plot_df.columns else None
        ax.errorbar(
            plot_df[x].to_numpy(),
            plot_df[y].to_numpy(dtype=np.float64),
            yerr=err,
            marker=MARKER_CIRCLE,
            linewidth=LINE_WIDTH_PRIMARY,
            color=resolve_plot_color(color_key or y, context=context),
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def bar_figure(
    labels: Sequence[object],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    yerr: Sequence[float] | None = None,
    color_keys: Sequence[object] | None = None,
    context: str | None = None,
    figsize: tuple[float, float] = (6.0, 4.4),
    rotation: float = 15.0,
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels), dtype=np.float64)
    keys = list(color_keys) if color_keys is not None else list(labels)
    color_values = [resolve_plot_color(key, context=context) for key in keys]
    ax.bar(x, values, yerr=yerr, color=color_values, edgecolor="black", linewidth=0.8, alpha=ALPHA_BAR, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def grouped_bar_figure(
    df: pd.DataFrame,
    *,
    group: str,
    value: str,
    title: str,
    ylabel: str,
    context: str | None = None,
    figsize: tuple[float, float] = (6.2, 4.5),
) -> Figure:
    if df.empty:
        return bar_figure([], [], title=title, ylabel=ylabel, context=context, figsize=figsize)
    grouped = grouped_mean_sem(df, [group], value)
    return bar_figure(
        grouped[group].astype(str).tolist(),
        grouped["mean"].to_numpy(dtype=np.float64),
        yerr=grouped["sem"].to_numpy(dtype=np.float64),
        title=title,
        ylabel=ylabel,
        context=context,
        figsize=figsize,
    )


def scatter_figure(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    hue: str | None = None,
    figsize: tuple[float, float] = (6.0, 4.8),
    trend: bool = False,
    context: str | None = None,
    color_key: object | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    elif hue and hue in df.columns:
        for label, sub in df.groupby(hue, sort=True):
            ax.scatter(sub[x], sub[y], s=24, alpha=ALPHA_SCATTER, color=resolve_plot_color(label, hue, context=context), edgecolors="none", label=str(label))
        ax.legend(frameon=False)
    else:
        ax.scatter(df[x], df[y], s=24, alpha=ALPHA_SCATTER, color=resolve_plot_color(color_key or y, context=context), edgecolors="none")
    if trend and not df.empty and pd.to_numeric(df[x], errors="coerce").nunique() >= 2:
        x_values = pd.to_numeric(df[x], errors="coerce").to_numpy(dtype=np.float64)
        y_values = pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        if mask.sum() >= 3:
            slope, intercept = np.polyfit(x_values[mask], y_values[mask], deg=1)
            x_line = np.linspace(float(np.nanmin(x_values[mask])), float(np.nanmax(x_values[mask])), 200)
            ax.plot(x_line, slope * x_line + intercept, color=get_plot_color("peak_region"), linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def heatmap_from_long(
    df: pd.DataFrame,
    *,
    row: str,
    col: str,
    value: str,
    title: str,
    cmap_kind: str | None = None,
    figsize: tuple[float, float] = (6.2, 5.2),
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    matrix = df.pivot_table(index=row, columns=col, values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=figsize)
    if matrix.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    cmap_obj = get_plot_cmap(cmap_kind or infer_plot_cmap_kind(value)).copy()
    cmap_obj.set_bad(color=get_plot_color("other_residual"))
    im = ax.imshow(matrix.to_numpy(dtype=np.float64), origin="upper", cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(item) for item in matrix.columns])
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([str(item) for item in matrix.index])
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def trace_mean_sem(trace_payload: Mapping[str, np.ndarray], condition: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    condition_vector = np.asarray(trace_payload["condition_name"]).astype(str)
    arr = np.asarray(trace_payload[key], dtype=np.float64)
    selected = arr[condition_vector == str(condition)]
    time_axis = np.arange(arr.shape[1], dtype=np.int64) if arr.ndim == 2 else np.arange(0, dtype=np.int64)
    if selected.size == 0:
        return time_axis, np.zeros_like(time_axis, dtype=np.float64), np.zeros_like(time_axis, dtype=np.float64)
    mean = selected.mean(axis=0)
    err = selected.std(axis=0, ddof=1) / np.sqrt(selected.shape[0]) if selected.shape[0] > 1 else np.zeros_like(mean)
    return time_axis, mean, err


def trace_figure(
    trace_payload: Mapping[str, np.ndarray],
    *,
    conditions: Sequence[tuple[str, str]],
    key: str,
    title: str,
    ylabel: str,
    max_steps: int | None = None,
    figsize: tuple[float, float] = (6.8, 4.8),
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    for condition, label in conditions:
        time_axis, mean, err = trace_mean_sem(trace_payload, condition, key)
        if max_steps is not None:
            time_axis = time_axis[:max_steps]
            mean = mean[:max_steps]
            err = err[:max_steps]
        color = resolve_plot_color(condition, context=title)
        ax.plot(time_axis, mean, color=color, linewidth=LINE_WIDTH_PRIMARY, label=label)
        ax.fill_between(time_axis, mean - err, mean + err, color=color, alpha=ALPHA_FILL)
    ax.axhline(0.0, color=get_plot_color("other_residual"), linewidth=LINE_WIDTH_REFERENCE, linestyle=":")
    ax.set_xlabel("Probe time step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=GRID_ALPHA_SOFT)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def trace_keys_figure(
    trace_payload: Mapping[str, np.ndarray],
    *,
    condition: str,
    key_specs: Sequence[tuple[str, str, str]],
    title: str,
    ylabel: str,
    figsize: tuple[float, float] = (6.8, 4.8),
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    for key, label, color in key_specs:
        time_axis, mean, err = trace_mean_sem(trace_payload, condition, key)
        line_color = resolve_plot_color(color, label, key, context=title)
        ax.plot(time_axis, mean, color=line_color, linewidth=LINE_WIDTH_PRIMARY, label=label)
        ax.fill_between(time_axis, mean - err, mean + err, color=line_color, alpha=ALPHA_FILL)
    ax.set_xlabel("Probe time step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=GRID_ALPHA_SOFT)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def jitter_distribution_figure(
    df: pd.DataFrame,
    *,
    condition_col: str,
    value_col: str,
    conditions: Sequence[tuple[str, str]],
    title: str,
    ylabel: str,
    figsize: tuple[float, float] = (6.2, 4.8),
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)
    for xpos, (condition, label) in enumerate(conditions):
        values = df.loc[df[condition_col].astype(str) == condition, value_col].to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=values.size)
        ax.scatter(np.full(values.size, xpos) + jitter, values, color=resolve_plot_color(condition, context=title), alpha=ALPHA_SCATTER, s=28, edgecolors="none")
        ax.errorbar([xpos], [float(values.mean())], yerr=[sem(values)], fmt="o", color="black", capsize=4)
    ax.axhline(0.0, color=get_plot_color("other_residual"), linewidth=LINE_WIDTH_REFERENCE, linestyle=":")
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([label for _, label in conditions], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=GRID_ALPHA_SOFT)
    fig.tight_layout()
    return fig


def first_numeric_columns(df: pd.DataFrame, *, exclude_ids: bool = True) -> list[str]:
    out: list[str] = []
    for column in df.columns:
        if exclude_ids and (column.endswith("_id") or column in {"trial_id", "pair_id", "record_id"}):
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            out.append(column)
    return out


def generic_csv_overview(df: pd.DataFrame, *, title: str, preferred_x: str | None = None, preferred_y: str | None = None) -> Figure:
    numeric = first_numeric_columns(df)
    if df.empty or not numeric:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.text(0.5, 0.5, "No plottable numeric data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    y = preferred_y if preferred_y in df.columns else numeric[0]
    x = preferred_x if preferred_x in df.columns else None
    if x is not None:
        return line_figure(df, x=x, y=y, title=title, ylabel=y)
    categorical = next((col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and 1 < df[col].nunique() <= 20), None)
    if categorical:
        return grouped_bar_figure(df, group=categorical, value=y, title=title, ylabel=y)
    return bar_figure(np.arange(min(12, len(df))), df[y].head(12), title=title, ylabel=y, rotation=0)
