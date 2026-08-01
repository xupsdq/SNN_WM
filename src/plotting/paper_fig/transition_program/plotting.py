from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

from src.plotting.paper_fig.typography import (
    FIGURE_TEXT_SIZE_PT,
    mark_panel_label,
)

from .contracts import FigureContract
from .style import (
    BLUE_TINT,
    CORAL,
    CORAL_TINT,
    GRAY,
    GRAY_DARK,
    GRAY_LIGHT,
    GRAY_PALE,
    INK,
    NAVY,
    TEAL,
    TEAL_TINT,
    WHITE,
)


def make_figure(
    contract: FigureContract,
) -> tuple[Figure, dict[str, Axes]]:
    width_mm, height_mm = contract.canvas_mm
    fig = plt.figure(
        figsize=(width_mm / 25.4, height_mm / 25.4),
        facecolor=WHITE,
    )
    slots: dict[str, Axes] = {}
    for panel in contract.panels:
        x_mm, y_top_mm, panel_width_mm, panel_height_mm = panel.position_mm
        left = x_mm / width_mm
        bottom = (height_mm - y_top_mm - panel_height_mm) / height_mm
        width = panel_width_mm / width_mm
        height = panel_height_mm / height_mm
        slot = fig.add_axes([left, bottom, width, height])
        slot.set_axis_off()
        slot.set_xlim(0.0, 1.0)
        slot.set_ylim(0.0, 1.0)
        slots[panel.panel_id] = slot
        add_panel_header(
            slot,
            panel.panel_id,
            panel.title,
            panel_width_mm=panel_width_mm,
        )
    fig.transition_panel_slots = slots
    return fig, slots


def add_panel_header(
    slot: Axes,
    panel_id: str,
    title: str,
    *,
    panel_width_mm: float,
) -> None:
    label = slot.text(
        0.0,
        1.0,
        panel_id.lower(),
        transform=slot.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    mark_panel_label(label)


def data_axis(
    slot: Axes,
    *,
    left: float = 0.14,
    right: float = 0.03,
    bottom: float = 0.18,
    top: float = 0.22,
) -> Axes:
    top = _title_free_top(top)
    return slot.inset_axes(
        [left, bottom, 1.0 - left - right, 1.0 - bottom - top]
    )


def split_axes(
    slot: Axes,
    n: int,
    *,
    orientation: str = "horizontal",
    left: float = 0.12,
    right: float = 0.03,
    bottom: float = 0.18,
    top: float = 0.24,
    gap: float = 0.08,
) -> list[Axes]:
    if n < 1:
        raise ValueError("n must be positive")
    top = _title_free_top(top)
    available_width = 1.0 - left - right
    available_height = 1.0 - bottom - top
    axes: list[Axes] = []
    if orientation == "horizontal":
        width = (available_width - gap * (n - 1)) / n
        for index in range(n):
            axes.append(
                slot.inset_axes(
                    [
                        left + index * (width + gap),
                        bottom,
                        width,
                        available_height,
                    ]
                )
            )
    elif orientation == "vertical":
        height = (available_height - gap * (n - 1)) / n
        for index in range(n):
            y = bottom + (n - index - 1) * (height + gap)
            axes.append(slot.inset_axes([left, y, available_width, height]))
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")
    return axes


def clean_axis(
    ax: Axes,
    *,
    grid_axis: str | None = None,
) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.65)
    ax.tick_params(length=2.4, width=0.6, pad=1.5)
    if grid_axis is not None:
        ax.grid(
            axis=grid_axis,
            color=GRAY_LIGHT,
            linewidth=0.5,
            alpha=0.55,
            zorder=0,
        )


def mean_ci(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(array.mean())
    if array.size == 1:
        return mean, mean, mean
    sem = float(stats.sem(array, nan_policy="omit"))
    half = float(stats.t.ppf(0.975, df=array.size - 1) * sem)
    return mean, mean - half, mean + half


def network_means(
    frame: pd.DataFrame,
    groups: Sequence[str],
    value: str,
    *,
    network: str = "network_seed",
) -> pd.DataFrame:
    required = {network, value, *groups}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for network means: {sorted(missing)}")
    data = frame.loc[:, [network, *groups, value]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna(subset=[value])
    return data.groupby([network, *groups], as_index=False, observed=True)[value].mean()


def estimation_plot(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    category: str,
    value: str,
    order: Sequence[Any],
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    null: float | None = 0.0,
    xlabel: str = "",
    network: str = "network_seed",
    connect_pairs: bool = False,
) -> pd.DataFrame:
    network_frame = network_means(frame, [category], value, network=network)
    order_list = list(order)
    if labels is None:
        labels = [str(item) for item in order_list]
    if colors is None:
        colors = [NAVY] * len(order_list)
    if connect_pairs:
        wide = network_frame.pivot(
            index=network,
            columns=category,
            values=value,
        )
        for _, row in wide.iterrows():
            xs = [row.get(item, np.nan) for item in order_list]
            mask = np.isfinite(np.asarray(xs, dtype=float))
            if mask.sum() > 1:
                ax.plot(
                    np.asarray(xs, dtype=float)[mask],
                    np.arange(len(order_list))[mask],
                    color=GRAY_LIGHT,
                    linewidth=0.55,
                    zorder=1,
                )
    for index, (item, color) in enumerate(zip(order_list, colors)):
        values = (
            network_frame.loc[network_frame[category].eq(item), value]
            .to_numpy(dtype=float)
        )
        if values.size == 0:
            raise ValueError(f"No values for {category}={item!r}")
        jitter = _jitter(values.size, width=0.14)
        ax.scatter(
            values,
            np.full(values.size, index, dtype=float) + jitter,
            s=9,
            color=color,
            edgecolor=WHITE,
            linewidth=0.3,
            alpha=0.55,
            zorder=2,
        )
        mean, low, high = mean_ci(values)
        ax.plot([low, high], [index, index], color=INK, linewidth=1.2, zorder=4)
        ax.scatter(
            [mean],
            [index],
            s=24,
            color=color,
            edgecolor=INK,
            linewidth=0.55,
            zorder=5,
        )
    ax.set_yticks(np.arange(len(order_list)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    if null is not None:
        ax.axvline(null, color=INK, linestyle=":", linewidth=0.75, zorder=0)
    clean_axis(ax, grid_axis="x")
    return network_frame


def network_line(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    x: str,
    value: str,
    group: str | None = None,
    group_order: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    network: str = "network_seed",
    show_networks: bool = True,
    show_legend: bool = True,
    null: float | None = None,
) -> pd.DataFrame:
    groups = [group] if group is not None else []
    network_frame = network_means(frame, [x, *groups], value, network=network)
    if group is None:
        group_order = (None,)
    elif group_order is None:
        group_order = tuple(network_frame[group].drop_duplicates().tolist())
    if labels is None:
        labels = [
            "" if item is None else str(item)
            for item in group_order
        ]
    if colors is None:
        colors = [NAVY] * len(tuple(group_order))
    if linestyles is None:
        linestyles = ["-"] * len(tuple(group_order))
    for item, label, color, linestyle in zip(
        group_order,
        labels,
        colors,
        linestyles,
    ):
        part = (
            network_frame
            if item is None
            else network_frame.loc[network_frame[group].eq(item)]
        )
        if part.empty:
            raise ValueError(f"No line data for {group}={item!r}")
        if show_networks:
            for _, network_part in part.groupby(network, sort=True):
                ordered = network_part.sort_values(x)
                ax.plot(
                    ordered[x],
                    ordered[value],
                    color=color,
                    linewidth=0.45,
                    alpha=0.13,
                    zorder=1,
                )
        summary_rows = []
        for x_value, x_part in part.groupby(x, sort=True):
            mean, low, high = mean_ci(x_part[value].to_numpy(float))
            summary_rows.append(
                {"x": x_value, "mean": mean, "low": low, "high": high}
            )
        summary = pd.DataFrame(summary_rows).sort_values("x")
        x_values = summary["x"].to_numpy()
        ax.fill_between(
            x_values,
            summary["low"].to_numpy(float),
            summary["high"].to_numpy(float),
            color=color,
            alpha=0.16,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(
            x_values,
            summary["mean"].to_numpy(float),
            color=color,
            linestyle=linestyle,
            marker="o",
            markerfacecolor=color if linestyle == "-" else WHITE,
            markeredgecolor=color,
            markeredgewidth=0.6,
            linewidth=1.25,
            label=label,
            zorder=3,
        )
    if null is not None:
        ax.axhline(null, color=INK, linestyle=":", linewidth=0.75, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    clean_axis(ax, grid_axis="y")
    if (
        show_legend
        and group is not None
        and len(tuple(group_order)) > 1
    ):
        ax.legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=min(2, len(tuple(group_order))),
            handlelength=1.6,
            borderaxespad=0.0,
            labelspacing=0.25,
            columnspacing=0.8,
        )
    return network_frame


def bivariate_quantile_trajectory(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    quantile: str,
    x_value: str,
    y_value: str,
    order: Sequence[Any],
    colors: Sequence[Any],
    xlabel: str,
    ylabel: str,
    network: str = "network_seed",
) -> pd.DataFrame:
    """Fuse two quantile endpoints into one network-level relationship plot."""
    required = {network, quantile, x_value, y_value}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing columns for bivariate quantile plot: {sorted(missing)}"
        )
    network_frame = (
        frame.loc[:, [network, quantile, x_value, y_value]]
        .dropna()
        .groupby([network, quantile], as_index=False, observed=True)[
            [x_value, y_value]
        ]
        .mean()
    )
    mean_points: list[tuple[float, float]] = []
    order_list = list(order)
    for item, color in zip(order_list, colors):
        part = network_frame.loc[network_frame[quantile].eq(item)]
        if part.empty:
            raise ValueError(f"No bivariate data for {quantile}={item!r}")
        ax.scatter(
            part[x_value],
            part[y_value],
            s=8,
            color=color,
            alpha=0.22,
            edgecolor="none",
            zorder=1,
        )
        x_mean, x_low, x_high = mean_ci(part[x_value].to_numpy(float))
        y_mean, y_low, y_high = mean_ci(part[y_value].to_numpy(float))
        mean_points.append((x_mean, y_mean))
        ax.errorbar(
            [x_mean],
            [y_mean],
            xerr=[[x_mean - x_low], [x_high - x_mean]],
            yerr=[[y_mean - y_low], [y_high - y_mean]],
            color=color,
            marker="o",
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.55,
            linewidth=0.8,
            capsize=1.8,
            zorder=4,
        )
    means = np.asarray(mean_points, dtype=float)
    ax.plot(
        means[:, 0],
        means[:, 1],
        color=GRAY_DARK,
        linewidth=0.8,
        zorder=2,
    )
    label_indices = sorted({0, len(order_list) // 2, len(order_list) - 1})
    for index in label_indices:
        ax.annotate(
            _format_label(order_list[index]),
            mean_points[index],
            xytext=(3, 2),
            textcoords="offset points",
            color=INK,
            ha="left",
            va="bottom",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    clean_axis(ax, grid_axis="both")
    return network_frame


def paired_dumbbell(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    left_value: str,
    right_value: str,
    left_label: str,
    right_label: str,
    ylabel: str,
    network: str = "network_seed",
    left_color: str = GRAY,
    right_color: str = NAVY,
    null: float | None = None,
) -> pd.DataFrame:
    data = (
        frame.groupby(network, as_index=False)[[left_value, right_value]]
        .mean()
        .dropna()
    )
    for _, row in data.iterrows():
        ax.plot(
            [0, 1],
            [row[left_value], row[right_value]],
            color=GRAY_LIGHT,
            linewidth=0.6,
            zorder=1,
        )
    ax.scatter(
        np.zeros(len(data)),
        data[left_value],
        s=11,
        facecolor=WHITE,
        edgecolor=left_color,
        linewidth=0.75,
        zorder=2,
    )
    ax.scatter(
        np.ones(len(data)),
        data[right_value],
        s=11,
        color=right_color,
        edgecolor=WHITE,
        linewidth=0.3,
        zorder=2,
    )
    for x_position, column, color in (
        (0.0, left_value, left_color),
        (1.0, right_value, right_color),
    ):
        mean, low, high = mean_ci(data[column].to_numpy(float))
        ax.errorbar(
            [x_position],
            [mean],
            yerr=[[mean - low], [high - mean]],
            color=INK,
            marker="o",
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.55,
            linewidth=1.0,
            capsize=2,
            zorder=4,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([left_label, right_label])
    ax.set_ylabel(ylabel)
    if null is not None:
        ax.axhline(null, color=INK, linestyle=":", linewidth=0.75)
    clean_axis(ax, grid_axis="y")
    return data


def heatmap(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    row: str,
    column: str,
    value: str,
    cmap: Any,
    row_order: Sequence[Any] | None = None,
    column_order: Sequence[Any] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    colorbar_label: str = "",
    annotate: bool = False,
) -> pd.DataFrame:
    matrix = frame.pivot_table(
        index=row,
        columns=column,
        values=value,
        aggfunc="mean",
        observed=True,
    )
    if row_order is not None:
        matrix = matrix.reindex(list(row_order))
    else:
        matrix = matrix.sort_index()
    if column_order is not None:
        matrix = matrix.reindex(columns=list(column_order))
    else:
        matrix = matrix.sort_index(axis=1)
    data = matrix.to_numpy(dtype=float)
    if center is not None:
        max_abs = np.nanmax(np.abs(data - center))
        vmin = center - max_abs if vmin is None else vmin
        vmax = center + max_abs if vmax is None else vmax
    image = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([_format_label(item) for item in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([_format_label(item) for item in matrix.index])
    ax.tick_params(length=0, pad=1.5)
    if annotate and data.size <= 64:
        threshold = np.nanmean(data)
        for row_index in range(data.shape[0]):
            for column_index in range(data.shape[1]):
                number = data[row_index, column_index]
                if not np.isfinite(number):
                    continue
                ax.text(
                    column_index,
                    row_index,
                    f"{number:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=WHITE if number > threshold else INK,
                )
    host_position = ax.get_position()
    colorbar_width = min(0.010, host_position.width * 0.04)
    colorbar_gap = min(0.010, host_position.width * 0.025)
    text_reserve = min(
        0.035 if colorbar_label else 0.022,
        host_position.width * 0.12,
    )
    plot_width = (
        host_position.width
        - colorbar_width
        - colorbar_gap
        - text_reserve
    )
    ax.set_axes_locator(None)
    ax.set_position(
        (
            host_position.x0,
            host_position.y0,
            plot_width,
            host_position.height,
        )
    )
    colorbar_axis = ax.figure.add_axes(
        (
            host_position.x0 + plot_width + colorbar_gap,
            host_position.y0,
            colorbar_width,
            host_position.height,
        )
    )
    colorbar = ax.figure.colorbar(image, cax=colorbar_axis)
    colorbar.ax.tick_params(length=2, width=0.5, pad=1)
    colorbar.set_label(colorbar_label)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return matrix.reset_index().melt(
        id_vars=row,
        var_name=column,
        value_name=value,
    )


def stacked_composition(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    category: str,
    components: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
    category_order: Sequence[Any],
    category_labels: Sequence[str] | None = None,
    xlabel: str = "Fraction",
    network: str = "network_seed",
) -> pd.DataFrame:
    grouped = (
        frame.groupby([network, category], as_index=False, observed=True)[
            list(components)
        ]
        .mean()
    )
    summary = (
        grouped.groupby(category, as_index=False, observed=True)[
            list(components)
        ]
        .mean()
        .set_index(category)
        .reindex(list(category_order))
    )
    totals = summary.sum(axis=1).replace(0.0, np.nan)
    normalized = summary.div(totals, axis=0).fillna(0.0)
    y = np.arange(len(normalized))
    left = np.zeros(len(normalized), dtype=float)
    for component, label, color in zip(components, labels, colors):
        values = normalized[component].to_numpy(float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.58,
            color=color,
            edgecolor=WHITE,
            linewidth=0.6,
            label=label,
        )
        left += values
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(
        category_labels
        if category_labels is not None
        else [_format_label(item) for item in normalized.index]
    )
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    clean_axis(ax, grid_axis=None)
    ax.legend(
        frameon=False,
        ncol=min(len(components), 2),
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
        handlelength=1.1,
        columnspacing=0.8,
    )
    return grouped


def scatter_relationship(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str = NAVY,
    group: str | None = None,
    group_colors: Mapping[Any, str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    max_points: int = 5000,
    identity: bool = False,
    zero_lines: bool = False,
) -> pd.DataFrame:
    columns = [x, y] + ([group] if group is not None else [])
    data = frame.loc[:, columns].copy()
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna(subset=[x, y])
    if len(data) > max_points:
        data = data.iloc[
            np.linspace(0, len(data) - 1, max_points).astype(int)
        ]
    if group is None:
        ax.scatter(
            data[x],
            data[y],
            s=8,
            color=color,
            alpha=0.25,
            edgecolor="none",
        )
    else:
        if group_colors is None:
            group_colors = {}
        for item, part in data.groupby(group, observed=True):
            item_color = group_colors.get(item, color)
            ax.scatter(
                part[x],
                part[y],
                s=8,
                color=item_color,
                alpha=0.25,
                edgecolor="none",
                label=str(item),
            )
        ax.legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=min(2, data[group].nunique()),
            handletextpad=0.3,
            borderaxespad=0.0,
            columnspacing=0.8,
        )
    if len(data) >= 8 and data[x].nunique() > 1:
        slope, intercept = np.polyfit(data[x].to_numpy(), data[y].to_numpy(), 1)
        line_x = np.linspace(data[x].min(), data[x].max(), 100)
        ax.plot(
            line_x,
            slope * line_x + intercept,
            color=INK,
            linewidth=0.9,
            linestyle="--",
        )
    if identity:
        low = min(float(data[x].min()), float(data[y].min()))
        high = max(float(data[x].max()), float(data[y].max()))
        ax.plot([low, high], [low, high], color=GRAY, linestyle=":", linewidth=0.75)
    if zero_lines:
        ax.axhline(0.0, color=INK, linestyle=":", linewidth=0.65)
        ax.axvline(0.0, color=INK, linestyle=":", linewidth=0.65)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    clean_axis(ax, grid_axis=None)
    return data


def schematic_chain(
    ax: Axes,
    labels: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    loop: bool = False,
    subtitle: str | None = None,
) -> pd.DataFrame:
    ax.set_axis_off()
    n = len(labels)
    if colors is None:
        colors = [BLUE_TINT, TEAL_TINT, CORAL_TINT, GRAY_PALE, BLUE_TINT][:n]
    left_margin = 0.035
    right_margin = 0.035
    gap = 0.035
    width = (
        1.0 - left_margin - right_margin - gap * (n - 1)
    ) / n
    y = 0.31
    height = 0.34
    rows: list[dict[str, object]] = []
    for index, (label, fill) in enumerate(zip(labels, colors)):
        x = left_margin + index * (width + gap)
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=fill,
            edgecolor=INK,
            linewidth=0.7,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2.0,
            y + height / 2.0,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FIGURE_TEXT_SIZE_PT,
            color=INK,
        )
        rows.append({"node": index, "label": label, "x": x})
        if index < n - 1:
            arrow = FancyArrowPatch(
                (x + width + 0.004, y + height / 2.0),
                (x + width + gap - 0.004, y + height / 2.0),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.75,
                color=GRAY_DARK,
                transform=ax.transAxes,
            )
            ax.add_patch(arrow)
    if loop:
        first_x = left_margin + width / 2.0
        last_x = left_margin + (n - 1) * (width + gap) + width / 2.0
        arrow = FancyArrowPatch(
            (last_x, y + 0.01),
            (first_x, y + 0.01),
            connectionstyle="arc3,rad=-0.38",
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.75,
            color=TEAL,
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)
    return pd.DataFrame(rows)


def _jitter(n: int, *, width: float) -> np.ndarray:
    if n <= 1:
        return np.zeros(n, dtype=float)
    return np.linspace(-width, width, n)


def _format_label(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).replace("_", " ")


def _title_free_top(top: float) -> float:
    """Reclaim the former per-panel title band while preserving label clearance."""
    return max(0.10, float(top) - 0.10)
