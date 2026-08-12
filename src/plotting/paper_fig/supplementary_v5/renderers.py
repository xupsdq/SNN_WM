from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from src.plotting.paper_fig.svg_icons import draw_tabler_icon

from .common import (
    BundleReader,
    CORAL,
    CYAN,
    INK,
    NAVY,
    NEUTRAL_DARK,
    NEUTRAL_LIGHT,
    NEUTRAL_MID,
    PALE_BLUE,
    PURPLE,
    TEAL,
    WHITE,
    add_panel_labels,
    add_plot_axis,
    add_top_colorbar,
    apply_axis_spec,
    color_for_role,
    deterministic_jitter,
    draw_matrix,
    draw_reference,
    figure_from_spec,
    horizontal_mean_ci,
    statistic_row,
    style_axis,
    vertical_mean_ci,
)


def _panel_data(reader: BundleReader, figure_id: str, panel_id: str, suffix: str = "") -> pd.DataFrame:
    relative = f"data/source_data/{figure_id}_{panel_id}{suffix}.csv"
    return reader.read_csv(relative, f"{figure_id}{panel_id} frozen panel data")


def _line_summary(
    axis,
    statistics: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    x_order: list[Any],
    *,
    x_filter: str,
    color: str,
    marker: str,
    linestyle: str = "-",
    extra_filters: Mapping[str, Any] | None = None,
) -> None:
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    filters = dict(extra_filters or {})
    for value in x_order:
        row = statistic_row(statistics, figure_id, panel_id, **filters, **{x_filter: value})
        means.append(float(row["mean"]))
        lows.append(float(row["ci95_low"]))
        highs.append(float(row["ci95_high"]))
    positions = np.arange(len(x_order), dtype=float)
    mean_values = np.asarray(means)
    axis.plot(
        positions,
        mean_values,
        color=color,
        linewidth=1.25,
        marker=marker,
        markersize=4.5,
        markerfacecolor=WHITE if linestyle != "-" else color,
        markeredgecolor=color,
        markeredgewidth=0.8,
        linestyle=linestyle,
        zorder=5,
    )
    axis.errorbar(
        positions,
        mean_values,
        yerr=[mean_values - np.asarray(lows), np.asarray(highs) - mean_values],
        fmt="none",
        ecolor=color,
        elinewidth=0.85,
        capsize=2.2,
        capthick=0.85,
        zorder=4,
    )


def _network_trajectory(
    fig: Figure,
    spec: Mapping[str, Any],
    statistics: pd.DataFrame,
    data: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    *,
    x_column: str,
    raw_lines: bool = True,
) -> None:
    panel = spec["panels"][panel_id]
    axis = add_plot_axis(fig, spec, panel_id)
    x_order = list(panel["x_order"])
    position = {value: index for index, value in enumerate(x_order)}
    if raw_lines:
        for _, part in data.groupby("network_seed", sort=True):
            ordered = part.sort_values(x_column)
            axis.plot(
                ordered[x_column].map(position),
                ordered["value"],
                color=PALE_BLUE,
                linewidth=0.43,
                alpha=0.26,
                zorder=1,
            )
    _line_summary(
        axis,
        statistics,
        figure_id,
        panel_id,
        x_order,
        x_filter=x_column,
        color=color_for_role(str(panel.get("color_role", "dynamic"))),
        marker=str(panel.get("marker", "o")),
    )
    axis.set_xticks(np.arange(len(x_order)))
    axis.set_xticklabels([str(value) for value in x_order])
    axis.set_xlim(-0.3, len(x_order) - 0.7)
    apply_axis_spec(axis, panel)
    if "reference" in panel:
        draw_reference(axis, float(panel["reference"]))
    style_axis(axis)


def render_s1(input_dir: BundleReader, spec: Mapping[str, Any], statistics: pd.DataFrame) -> Figure:
    fig = figure_from_spec(spec)
    _network_trajectory(fig, spec, statistics, _panel_data(input_dir, "s1", "a"), "s1", "a", x_column="delay_ms")
    _network_trajectory(fig, spec, statistics, _panel_data(input_dir, "s1", "b"), "s1", "b", x_column="delay_ms")

    panel = spec["panels"]["c"]
    data = _panel_data(input_dir, "s1", "c")
    axis = add_plot_axis(fig, spec, "c")
    order = list(panel["x_order"])
    pivot = data.pivot(index="network_seed", columns="endpoint", values="value")
    for seed, row in pivot.iterrows():
        jitter = float(deterministic_jitter([seed], width=0.055, salt=11)[0])
        axis.plot(
            [0 + jitter, 1 + jitter],
            [row["Inflow"], row["Outflow"]],
            color=NEUTRAL_LIGHT,
            linewidth=0.55,
            alpha=0.65,
            zorder=1,
        )
    raw_styles = {
        "Inflow": (CORAL, "o", CORAL),
        "Outflow": (NEUTRAL_DARK, "o", WHITE),
        "Net": (CORAL, "D", WHITE),
    }
    for x, endpoint in enumerate(order):
        part = data.loc[data["endpoint"].eq(endpoint)]
        edge, marker, face = raw_styles[endpoint]
        jitter = deterministic_jitter(part["network_seed"], width=0.055, salt=13 + x)
        axis.scatter(
            x + jitter,
            part["value"],
            s=8.0,
            marker=marker,
            facecolors=face,
            edgecolors=edge,
            linewidths=0.55,
            alpha=0.58,
            zorder=2,
        )
        row = statistic_row(statistics, "s1", "c", endpoint=endpoint)
        vertical_mean_ci(axis, x, row, color=edge, marker="D" if endpoint == "Net" else "o", markerfacecolor=face)
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(order)
    axis.set_xlim(-0.45, 2.45)
    apply_axis_spec(axis, panel)
    style_axis(axis)

    panel = spec["panels"]["d"]
    data = _panel_data(input_dir, "s1", "d")
    axis = add_plot_axis(fig, spec, "d")
    jitter = deterministic_jitter(data["network_seed"], width=0.075, salt=31)
    axis.scatter(
        jitter,
        data["value"],
        s=10.0,
        marker="o",
        facecolors=WHITE,
        edgecolors=CORAL,
        linewidths=0.65,
        alpha=0.62,
        zorder=2,
    )
    row = statistic_row(statistics, "s1", "d", role="display")
    vertical_mean_ci(axis, 0.0, row, color=CORAL, marker="D")
    axis.set_xticks([0.0])
    axis.set_xticklabels([panel["xlabel"]])
    axis.set_xlabel("")
    axis.set_xlim(-0.45, 0.45)
    axis.set_ylabel(panel["ylabel"])
    axis.set_ylim(*panel["ylim"])
    axis.set_yticks(panel["yticks"])
    draw_reference(axis, float(panel["reference"]))
    style_axis(axis)
    add_panel_labels(fig, spec)
    return fig


def _draw_s2_donor_component(axis, bounds: list[float]) -> None:
    x, y, width, height = [float(value) for value in bounds]
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=to_rgba(CORAL, 0.18),
            edgecolor=CORAL,
            linewidth=0.9,
            zorder=5,
        )
    )
    axis.text(
        x + width * 0.16,
        y + height / 2.0,
        "L1",
        ha="center",
        va="center",
        color=CORAL,
        fontweight="bold",
        fontsize=8.5,
        zorder=7,
    )
    axis.text(
        x + width * 0.61,
        y + height / 2.0,
        "donor u/x",
        ha="center",
        va="center",
        color=INK,
        fontsize=8.5,
        zorder=7,
    )


def _draw_s2_layer_stack(
    axis,
    bounds: list[float],
    *,
    title: str,
    donor_l1: bool,
) -> None:
    x, y, width, height = [float(value) for value in bounds]
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            facecolor=to_rgba(NAVY, 0.045),
            edgecolor=NAVY,
            linewidth=0.85,
            zorder=4,
        )
    )
    axis.text(
        x + width / 2.0,
        y + height + 0.03,
        title,
        ha="center",
        va="bottom",
        color=INK,
        fontsize=8.5,
        zorder=7,
    )
    row_height = height * 0.15
    for index, layer in enumerate(("L1", "L2", "L3")):
        row_center_y = y + height * (0.72 - 0.22 * index)
        row_y = row_center_y - row_height / 2.0
        is_donor = donor_l1 and index == 0
        row_color = CORAL if is_donor else NAVY
        axis.text(
            x + width * 0.13,
            row_y + row_height / 2.0,
            layer,
            ha="center",
            va="center",
            color=row_color,
            fontweight="bold" if index == 0 else "normal",
            fontsize=7.5,
            zorder=7,
        )
        axis.add_patch(
            FancyBboxPatch(
                (x + width * 0.24, row_y),
                width * 0.69,
                row_height,
                boxstyle="round,pad=0.002,rounding_size=0.006",
                facecolor=to_rgba(row_color, 0.18 if is_donor else 0.08),
                edgecolor=row_color,
                linewidth=0.8 if is_donor else 0.6,
                zorder=5,
            )
        )
        axis.text(
            x + width * 0.585,
            row_y + row_height / 2.0,
            "donor u/x" if is_donor else "receiver u/x",
            ha="center",
            va="center",
            color=INK,
            fontsize=7.2,
            zorder=7,
        )


def _draw_s2_exchange_schematic(
    fig: Figure,
    spec: Mapping[str, Any],
    data: pd.DataFrame,
) -> None:
    required_columns = {
        "element_id",
        "component",
        "owner_before",
        "owner_after",
        "operation",
    }
    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise ValueError(f"S2 exchange protocol is missing columns: {missing}")
    observed = {
        (
            str(row.element_id),
            str(row.component),
            str(row.owner_before),
            str(row.owner_after),
            str(row.operation),
        )
        for row in data.itertuples(index=False)
    }
    expected = {
        ("donor_component", "Layer 1 u/x", "donor", "receiver", "transfer"),
        (
            "receiver_component",
            "Layer 1 u/x",
            "receiver",
            "displaced",
            "replace",
        ),
        (
            "receiver_after",
            "receiver carrier + Layer 2/3 state",
            "receiver",
            "receiver",
            "retain_carrier",
        ),
    }
    if observed != expected:
        raise ValueError(f"S2 exchange protocol changed: {sorted(observed)}")

    axis = add_plot_axis(fig, spec, "a")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    layout = spec["panels"]["a"]["exchange_layout"]
    before_bounds = [float(value) for value in layout["before_group"]]
    before_x, before_y, before_width, before_height = before_bounds
    axis.add_patch(
        FancyBboxPatch(
            (before_x, before_y),
            before_width,
            before_height,
            boxstyle="round,pad=0.006,rounding_size=0.014",
            facecolor=WHITE,
            edgecolor=NEUTRAL_LIGHT,
            linewidth=0.75,
            zorder=2,
        )
    )
    axis.text(
        before_x + before_width / 2.0,
        before_y + before_height - 0.055,
        "Before exchange",
        ha="center",
        va="center",
        color=INK,
        zorder=7,
    )
    _draw_s2_donor_component(
        axis, [float(value) for value in layout["donor_component"]]
    )
    _draw_s2_layer_stack(
        axis,
        [float(value) for value in layout["receiver_before"]],
        title="Receiver before",
        donor_l1=False,
    )

    operation_bounds = [
        float(value) for value in layout["operation_icon"]
    ]
    operation_x, operation_y, operation_width, operation_height = (
        operation_bounds
    )
    draw_tabler_icon(
        axis,
        "replace",
        operation_bounds,
        color=CORAL,
        linewidth=0.9,
        zorder=6,
    )
    axis.text(
        operation_x + operation_width / 2.0,
        operation_y + operation_height + 0.075,
        "Layer-1 only",
        ha="center",
        va="bottom",
        color=CORAL,
        fontweight="bold",
        zorder=7,
    )
    axis.text(
        operation_x + operation_width / 2.0,
        operation_y - 0.065,
        "replace receiver u/x",
        ha="center",
        va="top",
        color=INK,
        zorder=7,
    )

    after_bounds = [float(value) for value in layout["receiver_after"]]
    _draw_s2_layer_stack(
        axis,
        after_bounds,
        title="Receiver after",
        donor_l1=True,
    )
    after_x, after_y, after_width, after_height = after_bounds
    arrow_y = operation_y + operation_height / 2.0
    for start, end in (
        (
            (before_x + before_width + 0.012, arrow_y),
            (operation_x - 0.012, arrow_y),
        ),
        (
            (operation_x + operation_width + 0.012, arrow_y),
            (after_x - 0.012, arrow_y),
        ),
    ):
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=7.5,
                linewidth=0.9,
                color=NEUTRAL_DARK,
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=3,
            )
        )


def _draw_s2_primary_transfer(
    fig: Figure,
    spec: Mapping[str, Any],
    statistics: pd.DataFrame,
    data: pd.DataFrame,
) -> None:
    panel = spec["panels"]["b"]
    axis = add_plot_axis(fig, spec, "b")
    row_order = list(panel["row_order"])
    layer_styles = {"L2 update": (NAVY, "o"), "Early L3": (TEAL, "s")}
    for index, layer in enumerate(row_order):
        y = len(row_order) - 1 - index
        color, marker = layer_styles[layer]
        part = data.loc[data["layer"].eq(layer)].sort_values("network_seed")
        if len(part) != 20:
            raise ValueError(f"S2b expected 20 network rows for {layer}, found {len(part)}")
        jitter = deterministic_jitter(part["network_seed"], width=0.10, salt=120 + index)
        axis.scatter(
            part["value"],
            y + jitter,
            s=9.0,
            marker=marker,
            facecolors=color,
            edgecolors=color,
            linewidths=0.45,
            alpha=0.52,
            zorder=2,
        )
        row = statistic_row(statistics, "s2", "b", layer=layer)
        horizontal_mean_ci(axis, y, row, color=color, marker=marker)
    axis.set_yticks(range(len(row_order)))
    axis.set_yticklabels(list(reversed(panel.get("row_labels", row_order))))
    axis.set_ylim(-0.45, len(row_order) - 0.55)
    apply_axis_spec(axis, panel)
    for index, reference in enumerate(panel.get("references", [])):
        draw_reference(axis, float(reference), orientation="vertical", linestyle="--" if index == 0 else ":")
    style_axis(axis)


def render_s2(input_dir: BundleReader, spec: Mapping[str, Any], statistics: pd.DataFrame) -> Figure:
    fig = figure_from_spec(spec)
    _draw_s2_exchange_schematic(fig, spec, _panel_data(input_dir, "s2", "a"))
    _draw_s2_primary_transfer(fig, spec, statistics, _panel_data(input_dir, "s2", "b"))

    panel = spec["panels"]["c"]
    data = _panel_data(input_dir, "s2", "c")
    axis = add_plot_axis(fig, spec, "c")
    rows = list(panel["row_order"])
    layers = ["L2 update", "Early L3"]
    layer_style = {"L2 update": (NAVY, "o", -0.11), "Early L3": (TEAL, "s", 0.11)}
    for row_index, metric in enumerate(rows):
        y_base = len(rows) - 1 - row_index
        for layer in layers:
            color, marker, offset = layer_style[layer]
            part = data.loc[data["metric"].eq(metric) & data["layer"].eq(layer)]
            jitter = deterministic_jitter(part["network_seed"], width=0.055, salt=200 + row_index)
            axis.scatter(
                part["value"],
                y_base + offset + jitter,
                s=7.5,
                marker=marker,
                color=color,
                alpha=0.35,
                linewidths=0,
                zorder=1,
            )
            row = statistic_row(statistics, "s2", "c", layer=layer, metric=metric)
            horizontal_mean_ci(axis, y_base + offset, row, color=color, marker=marker)
    axis.set_yticks(range(len(rows)))
    display_rows = list(panel.get("row_labels", rows))
    axis.set_yticklabels(list(reversed(display_rows)))
    axis.set_ylim(-0.45, len(rows) - 0.55)
    apply_axis_spec(axis, panel)
    draw_reference(axis, float(panel["reference"]), orientation="vertical")
    style_axis(axis)
    axis.legend(
        handles=[
            Line2D([0], [0], color=NAVY, marker="o", linewidth=0, markersize=4.5, label="L2"),
            Line2D([0], [0], color=TEAL, marker="s", linewidth=0, markersize=4.5, label="L3"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.7,
        borderaxespad=0.0,
    )

    panel = spec["panels"]["d"]
    data = _panel_data(input_dir, "s2", "d")
    axis = add_plot_axis(fig, spec, "d")
    row_order = list(panel["row_order"])
    layer_colors = {"L2 update": (NAVY, "o"), "Early L3": (TEAL, "s")}
    for index, layer in enumerate(row_order):
        y = len(row_order) - 1 - index
        color, marker = layer_colors[layer]
        untouched = data.loc[data["layer"].eq(layer) & data["cohort"].eq("Untouched 19")]
        jitter = deterministic_jitter(untouched["network_seed"], width=0.10, salt=310 + index)
        axis.scatter(untouched["value"], y + jitter, s=9.0, marker=marker, color=color, alpha=0.48, linewidths=0)
        untouched_stats = statistic_row(statistics, "s2", "d", layer=layer, cohort="Untouched 19")
        horizontal_mean_ci(axis, y, untouched_stats, color=color, marker=marker)
        full_stats = statistic_row(statistics, "s2", "d", layer=layer, cohort="Full 20")
        axis.plot(
            [float(full_stats["mean"])],
            [y],
            marker=marker,
            markersize=6.0,
            markerfacecolor=WHITE,
            markeredgecolor=NEUTRAL_MID,
            markeredgewidth=0.9,
            linestyle="none",
            zorder=6,
        )
    axis.set_yticks(range(len(row_order)))
    display_rows = list(panel.get("row_labels", row_order))
    axis.set_yticklabels(list(reversed(display_rows)))
    axis.set_ylim(-0.45, len(row_order) - 0.55)
    apply_axis_spec(axis, panel)
    draw_reference(axis, float(panel["reference"]), orientation="vertical")
    style_axis(axis)
    axis.legend(
        handles=[
            Line2D([0], [0], color=INK, marker="o", markerfacecolor=INK, linewidth=0, markersize=4.5, label="Untouched"),
            Line2D([0], [0], color=NEUTRAL_MID, marker="o", markerfacecolor=WHITE, linewidth=0, markersize=4.5, label="Full 20"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.7,
        borderaxespad=0.0,
    )
    add_panel_labels(fig, spec)
    return fig


def _s3_point_range(
    fig: Figure,
    spec: Mapping[str, Any],
    statistics: pd.DataFrame,
    data: pd.DataFrame,
    panel_id: str,
    *,
    key: str,
    labels: Mapping[Any, str],
) -> None:
    panel = spec["panels"][panel_id]
    axis = add_plot_axis(fig, spec, panel_id)
    order = list(labels)
    for index, value in enumerate(order):
        y = len(order) - 1 - index
        part = data.loc[np.isclose(pd.to_numeric(data[key]), float(value))]
        jitter = deterministic_jitter(part["network_seed"], width=0.10, salt=410 + index)
        axis.scatter(part["value"], y + jitter, s=8.0, color=CYAN, alpha=0.48, linewidths=0)
        row = statistic_row(statistics, "s3", panel_id, **{key: value})
        horizontal_mean_ci(axis, y, row, color=NAVY, marker="D")
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels([labels[value] for value in reversed(order)])
    axis.set_ylim(-0.45, len(order) - 0.55)
    apply_axis_spec(axis, panel)
    draw_reference(axis, float(panel["reference"]), orientation="vertical")
    style_axis(axis)


def render_s3(input_dir: BundleReader, spec: Mapping[str, Any], statistics: pd.DataFrame) -> Figure:
    fig = figure_from_spec(spec)
    panel = spec["panels"]["a"]
    axis = add_plot_axis(fig, spec, "a")
    order = list(panel["x_order"])
    series_styles = {
        "Probe-only": (NAVY, "o", "-"),
        "Random": (NEUTRAL_DARK, "s", "--"),
    }
    for series in panel["series_order"]:
        color, marker, linestyle = series_styles[series]
        _line_summary(
            axis,
            statistics,
            "s3",
            "a",
            order,
            x_filter="window_ms",
            color=color,
            marker=marker,
            linestyle=linestyle,
            extra_filters={"comparator": series},
        )
    axis.set_xticks(np.arange(len(order)))
    axis.set_xticklabels([str(value) for value in order])
    axis.set_xlim(-0.3, len(order) - 0.7)
    apply_axis_spec(axis, panel)
    draw_reference(axis, float(panel["reference"]))
    style_axis(axis)
    axis.legend(
        handles=[
            Line2D([0], [0], color=NAVY, marker="o", linewidth=1.2, markersize=4.2, label="Probe-only"),
            Line2D([0], [0], color=NEUTRAL_DARK, marker="s", markerfacecolor=WHITE, linestyle="--", linewidth=1.2, markersize=4.2, label="Random"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    _s3_point_range(fig, spec, statistics, _panel_data(input_dir, "s3", "b"), "b", key="cap", labels={1: "Top 1", 2: "Top 2", 3: "Top 3"})
    _s3_point_range(fig, spec, statistics, _panel_data(input_dir, "s3", "c"), "c", key="distance_limit", labels={2: "≤2", 4: "≤4", 6: "≤6"})

    panel = spec["panels"]["d"]
    data = _panel_data(input_dir, "s3", "d")
    axis = add_plot_axis(fig, spec, "d")
    conditions = list(panel["x_order"])
    fates = list(panel["fate_order"])
    fate_colors = {"Lost": NEUTRAL_LIGHT, "Delayed": CORAL, "Preserved": NAVY}
    bottoms = np.zeros(len(conditions), dtype=float)
    for fate in fates:
        means = [float(statistic_row(statistics, "s3", "d", condition=condition, fate=fate)["mean"]) for condition in conditions]
        axis.bar(
            np.arange(len(conditions)),
            means,
            bottom=bottoms,
            width=0.52,
            color=fate_colors[fate],
            edgecolor=INK,
            linewidth=0.5,
            label=fate,
            zorder=2,
        )
        bottoms += np.asarray(means)
    for x, condition in enumerate(conditions):
        row = statistic_row(statistics, "s3", "d", role="cumulative", condition=condition, fate="Disrupted")
        vertical_mean_ci(axis, x, row, color=INK, marker="none", zorder=6)
    axis.set_xticks(range(len(conditions)))
    axis.set_xticklabels(conditions)
    axis.set_xlim(-0.55, len(conditions) - 0.45)
    apply_axis_spec(axis, panel)
    style_axis(axis)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    add_panel_labels(fig, spec)
    return fig


def render_s4(input_dir: BundleReader, spec: Mapping[str, Any], statistics: pd.DataFrame) -> Figure:
    fig = figure_from_spec(spec)

    for panel_id, salt in (("a", 520), ("b", 540)):
        panel = spec["panels"][panel_id]
        data = _panel_data(input_dir, "s4", panel_id)
        axis = add_plot_axis(fig, spec, panel_id)
        order = list(panel["x_order"])
        color = color_for_role(str(panel["color_role"]))
        marker = str(panel["marker"])
        endpoint = str(data["endpoint"].iloc[0])
        for x, prefix_k in enumerate(order):
            part = data.loc[data["prefix_k"].eq(prefix_k)].sort_values("network_seed")
            if len(part) != 20:
                raise ValueError(f"S4{panel_id} expected 20 networks at K{prefix_k}")
            jitter = deterministic_jitter(part["network_seed"], width=0.13, salt=salt + x)
            axis.scatter(
                x + jitter,
                part["value"],
                s=8.5,
                marker=marker,
                facecolors=color if panel_id == "a" else WHITE,
                edgecolors=color,
                linewidths=0.55,
                alpha=0.52,
                zorder=2,
            )
            row = statistic_row(statistics, "s4", panel_id, prefix_k=prefix_k, endpoint=endpoint)
            vertical_mean_ci(
                axis,
                x,
                row,
                color=color,
                marker=marker,
                markerfacecolor=color if panel_id == "a" else WHITE,
            )
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels([f"K{value}" for value in order])
        axis.set_xlim(-0.45, len(order) - 0.55)
        apply_axis_spec(axis, panel)
        style_axis(axis)

    panel = spec["panels"]["c"]
    data = _panel_data(input_dir, "s4", "c")
    axis = add_plot_axis(fig, spec, "c")
    row_specs = (("L2", 1), ("L2", 5), ("L3", 1), ("L3", 5))
    endpoint_style = {"L2": (NAVY, "o"), "L3": (TEAL, "s")}
    for index, (endpoint, prefix_k) in enumerate(row_specs):
        y = len(row_specs) - 1 - index
        color, marker = endpoint_style[endpoint]
        confirm = data.loc[
            data["endpoint"].eq(endpoint)
            & data["prefix_k"].eq(prefix_k)
            & data["cohort"].eq("Confirm. 19")
        ].sort_values("network_seed")
        if len(confirm) != 19:
            raise ValueError(f"S4c expected 19 confirmatory networks for {endpoint} K{prefix_k}")
        jitter = deterministic_jitter(confirm["network_seed"], width=0.10, salt=570 + index)
        axis.scatter(
            confirm["value"],
            y + jitter,
            s=8.0,
            marker=marker,
            facecolors=color,
            edgecolors=color,
            linewidths=0.45,
            alpha=0.48,
            zorder=2,
        )
        confirm_stats = statistic_row(
            statistics,
            "s4",
            "c",
            endpoint=endpoint,
            prefix_k=prefix_k,
            cohort="Confirm. 19",
            role="display",
        )
        horizontal_mean_ci(axis, y, confirm_stats, color=color, marker=marker)
        full_stats = statistic_row(
            statistics,
            "s4",
            "c",
            endpoint=endpoint,
            prefix_k=prefix_k,
            cohort="Full 20",
            role="reference",
        )
        axis.plot(
            [float(full_stats["mean"])],
            [y],
            marker=marker,
            markersize=6.0,
            markerfacecolor=WHITE,
            markeredgecolor=NEUTRAL_MID,
            markeredgewidth=0.9,
            linestyle="none",
            zorder=6,
        )
    axis.set_yticks(range(len(row_specs)))
    axis.set_yticklabels(list(reversed(panel["row_order"])))
    axis.set_ylim(-0.45, len(row_specs) - 0.55)
    apply_axis_spec(axis, panel)
    style_axis(axis)
    axis.legend(
        handles=[
            Line2D([0], [0], color=INK, marker="o", markerfacecolor=INK, linewidth=0, markersize=4.5, label="Confirm. 19"),
            Line2D([0], [0], color=NEUTRAL_MID, marker="o", markerfacecolor=WHITE, linewidth=0, markersize=4.5, label="Full 20"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.7,
        borderaxespad=0.0,
    )

    panel = spec["panels"]["d"]
    data = _panel_data(input_dir, "s4", "d")
    rows = list(panel["row_order"])
    columns = list(panel["column_order"])
    matrix = (
        data.pivot_table(index="prefix_k", columns="gate", values="value", aggfunc="mean")
        .reindex(index=rows, columns=columns)
        .to_numpy(dtype=float)
    )
    if matrix.shape != (2, 6) or not np.allclose(matrix, 100.0):
        raise ValueError("S4d identity gate matrix must contain twelve 100% pass cells")
    axis = add_plot_axis(fig, spec, "d")
    image = draw_matrix(
        axis,
        matrix,
        cmap_role=str(panel["cmap_role"]),
        vmin=float(panel["vmin"]),
        vmax=float(panel["vmax"]),
        xlabels=list(panel.get("column_labels", columns)),
        ylabels=[f"K{value}" for value in rows],
        annotate_decimals=int(panel["decimals"]),
    )
    axis.set_xlabel(panel["xlabel"])
    axis.set_ylabel(panel["ylabel"])
    add_top_colorbar(fig, spec, panel, image, ticks=[0.0, 50.0, 100.0])

    add_panel_labels(fig, spec)
    return fig


def render_s5(input_dir: BundleReader, spec: Mapping[str, Any], statistics: pd.DataFrame) -> Figure:
    fig = figure_from_spec(spec)
    panel = spec["panels"]["a"]
    data = _panel_data(input_dir, "s5", "a")
    networks = list(range(1000, 1020))
    stages = list(range(2, 11))
    matrix = (
        data.pivot(index="stage_k", columns="network_seed", values="value")
        .reindex(index=stages, columns=networks)
        .to_numpy(dtype=float)
    )
    axis = add_plot_axis(fig, spec, "a")
    image = draw_matrix(
        axis,
        matrix,
        cmap_role=str(panel["cmap_role"]),
        vmin=float(panel["vmin"]),
        vmax=float(panel["vmax"]),
        xlabels=networks,
        ylabels=stages,
    )
    shown = [0, 5, 10, 15, 19]
    axis.set_xticks(shown)
    axis.set_xticklabels([str(networks[index]) for index in shown])
    axis.set_xlabel(panel["xlabel"])
    axis.set_ylabel(panel["ylabel"])
    add_top_colorbar(fig, spec, panel, image, ticks=[0.0, 0.3, 0.6])

    panel = spec["panels"]["b"]
    data = _panel_data(input_dir, "s5", "b")
    axis = add_plot_axis(fig, spec, "b")
    jitter = deterministic_jitter(data["network_seed"], width=0.14, salt=610)
    axis.scatter(data["value"], jitter, s=9.0, color=CYAN, alpha=0.55, linewidths=0)
    row = statistic_row(statistics, "s5", "b")
    horizontal_mean_ci(axis, 0.0, row, color=NAVY, marker="D")
    axis.set_yticks([0.0])
    axis.set_yticklabels([str(panel["row_label"])])
    axis.set_ylim(-0.45, 0.45)
    apply_axis_spec(axis, panel)
    draw_reference(axis, float(panel["reference"]), orientation="vertical")
    style_axis(axis)

    _network_trajectory(fig, spec, statistics, _panel_data(input_dir, "s5", "c"), "s5", "c", x_column="stage_k")
    _network_trajectory(fig, spec, statistics, _panel_data(input_dir, "s5", "d"), "s5", "d", x_column="stage_k")
    add_panel_labels(fig, spec)
    return fig


def _summary_heatmap(
    fig: Figure,
    spec: Mapping[str, Any],
    statistics: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    *,
    row_key: str,
    column_key: str,
) -> Any:
    panel = spec["panels"][panel_id]
    rows = list(panel["row_order"])
    columns = list(panel["column_order"])
    matrix = np.empty((len(rows), len(columns)), dtype=float)
    for row_index, row_value in enumerate(rows):
        for column_index, column_value in enumerate(columns):
            matrix[row_index, column_index] = float(
                statistic_row(statistics, figure_id, panel_id, **{row_key: row_value, column_key: column_value})["mean"]
            )
    axis = add_plot_axis(fig, spec, panel_id)
    image = draw_matrix(
        axis,
        matrix,
        cmap_role=str(panel["cmap_role"]),
        vmin=float(panel["vmin"]),
        vmax=float(panel["vmax"]),
        xlabels=columns,
        ylabels=rows,
        annotate_decimals=int(panel["decimals"]),
    )
    axis.set_xlabel(panel["xlabel"])
    axis.set_ylabel(panel["ylabel"])
    return axis, image, matrix


def render_s6(input_dir: BundleReader, spec: Mapping[str, Any], statistics: pd.DataFrame) -> Figure:
    fig = figure_from_spec(spec)
    definition_styles = {
        "NNLS": (NAVY, "o", "-", PALE_BLUE),
        "Similarity": (PURPLE, "s", "--", NEUTRAL_LIGHT),
    }
    for panel_index, panel_id in enumerate(("a", "b")):
        panel = spec["panels"][panel_id]
        data = _panel_data(input_dir, "s6", panel_id)
        axis = add_plot_axis(fig, spec, panel_id)
        order = list(panel["x_order"])
        positions = {value: index for index, value in enumerate(order)}
        for definition in panel["series_order"]:
            color, marker, linestyle, raw_color = definition_styles[definition]
            subset = data.loc[data["definition"].eq(definition)]
            for _, network in subset.groupby("network_seed", sort=True):
                network = network.sort_values("seq_len")
                axis.plot(
                    network["seq_len"].map(positions),
                    network["value"],
                    color=raw_color,
                    linewidth=0.40,
                    linestyle=linestyle,
                    alpha=0.24,
                    zorder=1,
                )
            _line_summary(
                axis,
                statistics,
                "s6",
                panel_id,
                order,
                x_filter="seq_len",
                color=color,
                marker=marker,
                linestyle=linestyle,
                extra_filters={"definition": definition},
            )
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels([f"K{value}" for value in order])
        axis.set_xlim(-0.3, len(order) - 0.7)
        apply_axis_spec(axis, panel)
        draw_reference(axis, float(panel["reference"]), linestyle=":" if panel_index else "--")
        style_axis(axis)
        axis.legend(
            handles=[
                Line2D([0], [0], color=NAVY, marker="o", linewidth=1.2, markersize=4.2, label="NNLS"),
                Line2D([0], [0], color=PURPLE, marker="s", markerfacecolor=WHITE, linestyle="--", linewidth=1.2, markersize=4.2, label="Similarity"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            handlelength=1.5,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.0,
        )

    for panel_id, ticks in (("c", [0.0, 0.06, 0.12]), ("e", [0.0, 0.2, 0.4])):
        axis, image, _ = _summary_heatmap(
            fig,
            spec,
            statistics,
            "s6",
            panel_id,
            row_key="seq_len",
            column_key="delay_ms",
        )
        axis.set_yticklabels([f"K{value}" for value in spec["panels"][panel_id]["row_order"]])
        add_top_colorbar(fig, spec, spec["panels"][panel_id], image, ticks=ticks)

    _network_trajectory(
        fig,
        spec,
        statistics,
        _panel_data(input_dir, "s6", "d"),
        "s6",
        "d",
        x_column="delay_ms",
    )

    panel = spec["panels"]["f"]
    data = _panel_data(input_dir, "s6", "f")
    axis = add_plot_axis(fig, spec, "f")
    jitter = deterministic_jitter(data["network_seed"], width=0.14, salt=810)
    axis.scatter(data["value"], jitter, s=9.0, color=TEAL, alpha=0.52, linewidths=0)
    row = statistic_row(statistics, "s6", "f")
    horizontal_mean_ci(axis, 0.0, row, color=TEAL, marker="D")
    axis.set_yticks([0.0])
    axis.set_yticklabels([str(panel["row_label"])])
    axis.set_ylim(-0.45, 0.45)
    apply_axis_spec(axis, panel)
    draw_reference(axis, float(panel["reference"]), orientation="vertical")
    style_axis(axis)
    add_panel_labels(fig, spec)
    return fig


def _anchor_heatmap(axis, panel: Mapping[str, Any]) -> None:
    column_order = list(panel["column_order"])
    row_order = list(panel["row_order"])
    anchor_q, anchor_threshold = panel["anchor"]
    column = next(index for index, value in enumerate(column_order) if np.isclose(value, anchor_q))
    row = next(index for index, value in enumerate(row_order) if np.isclose(value, anchor_threshold))
    axis.add_patch(Rectangle((column - 0.48, row - 0.48), 0.96, 0.96, fill=False, edgecolor=INK, linewidth=0.9, zorder=6))


def render_s7(input_dir: BundleReader, spec: Mapping[str, Any], statistics: pd.DataFrame) -> Figure:
    fig = figure_from_spec(spec)

    panel = spec["panels"]["a"]
    data = _panel_data(input_dir, "s7", "a")
    axis = add_plot_axis(fig, spec, "a")
    order = list(panel["x_order"])
    pivot = data.pivot(index="network_seed", columns="subset", values="value").reindex(columns=order)
    for seed, values in pivot.iterrows():
        jitter = float(deterministic_jitter([seed], width=0.045, salt=910)[0])
        axis.plot(
            [0 + jitter, 1 + jitter],
            [values[order[0]], values[order[1]]],
            color=NEUTRAL_LIGHT,
            linewidth=0.55,
            alpha=0.65,
            zorder=1,
        )
    subset_styles = {
        "All trials": (NEUTRAL_DARK, "o", WHITE),
        "Exact match": (NAVY, "D", NAVY),
    }
    for x, subset in enumerate(order):
        part = data.loc[data["subset"].eq(subset)].sort_values("network_seed")
        color, marker, face = subset_styles[subset]
        jitter = deterministic_jitter(part["network_seed"], width=0.075, salt=920 + x)
        axis.scatter(
            x + jitter,
            part["value"],
            s=8.5,
            marker=marker,
            facecolors=face,
            edgecolors=color,
            linewidths=0.55,
            alpha=0.52,
            zorder=2,
        )
        row = statistic_row(statistics, "s7", "a", subset=subset)
        vertical_mean_ci(axis, x, row, color=color, marker=marker, markerfacecolor=face)
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(order)
    axis.set_xlim(-0.45, len(order) - 0.55)
    apply_axis_spec(axis, panel)
    draw_reference(axis, float(panel["reference"]))
    style_axis(axis)

    panel = spec["panels"]["b"]
    data = _panel_data(input_dir, "s7", "b")
    axis = add_plot_axis(fig, spec, "b")
    jitter = deterministic_jitter(data["network_seed"], width=0.075, salt=940)
    axis.scatter(
        jitter,
        data["value"],
        s=9.0,
        marker="o",
        facecolors=WHITE,
        edgecolors=TEAL,
        linewidths=0.65,
        alpha=0.58,
        zorder=2,
    )
    row = statistic_row(statistics, "s7", "b")
    vertical_mean_ci(axis, 0.0, row, color=TEAL, marker="D")
    axis.set_xticks([0.0])
    axis.set_xticklabels([panel["xlabel"]])
    axis.set_xlabel("")
    axis.set_xlim(-0.45, 0.45)
    axis.set_ylabel(panel["ylabel"])
    axis.set_ylim(*panel["ylim"])
    axis.set_yticks(panel["yticks"])
    style_axis(axis)

    panel = spec["panels"]["c"]
    data = _panel_data(input_dir, "s7", "c")
    _network_trajectory(fig, spec, statistics, data, "s7", "c", x_column="window_ms")
    axis = fig.axes[-1]
    anchor_index = list(panel["x_order"]).index(panel["anchor"])
    anchor_row = statistic_row(statistics, "s7", "c", window_ms=panel["anchor"])
    axis.plot(
        [anchor_index],
        [float(anchor_row["mean"])],
        marker="o",
        markersize=7.0,
        markerfacecolor="none",
        markeredgecolor=INK,
        markeredgewidth=0.7,
        linestyle="none",
        zorder=7,
    )

    for panel_id, ticks in (("d", [0.0, 10.0, 20.0]), ("e", [0.0, 50.0, 100.0])):
        axis, image, _ = _summary_heatmap(
            fig,
            spec,
            statistics,
            "s7",
            panel_id,
            row_key="overlap_threshold",
            column_key="stsp_group_quantile",
        )
        axis.set_xticklabels([f"{value:.2f}" for value in spec["panels"][panel_id]["column_order"]])
        axis.set_yticklabels([f"{value:.2f}" for value in spec["panels"][panel_id]["row_order"]])
        _anchor_heatmap(axis, spec["panels"][panel_id])
        add_top_colorbar(fig, spec, spec["panels"][panel_id], image, ticks=ticks)

    panel = spec["panels"]["f"]
    data = _panel_data(input_dir, "s7", "f")
    axis = add_plot_axis(fig, spec, "f")
    order = list(panel["x_order"])
    endpoint_styles = {
        "Observed": (NAVY, "o", NAVY),
        "Shuffled": (NEUTRAL_DARK, "s", WHITE),
        "Difference": (TEAL, "D", TEAL),
    }
    for x, endpoint in enumerate(order):
        part = data.loc[data["endpoint"].eq(endpoint)]
        color, marker, face = endpoint_styles[endpoint]
        jitter = deterministic_jitter(part["network_seed"], width=0.075, salt=1010 + x)
        axis.scatter(
            x + jitter,
            part["value"],
            s=8.5,
            marker=marker,
            facecolors=face,
            edgecolors=color,
            linewidths=0.55,
            alpha=0.52,
        )
        row = statistic_row(statistics, "s7", "f", endpoint=endpoint)
        vertical_mean_ci(axis, x, row, color=color, marker=marker, markerfacecolor=face)
        if endpoint == panel["anchor"]:
            axis.plot(
                [x],
                [float(row["mean"])],
                marker="o",
                markersize=7.0,
                markerfacecolor="none",
                markeredgecolor=INK,
                markeredgewidth=0.7,
                linestyle="none",
                zorder=7,
            )
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(order)
    axis.set_xlim(-0.45, len(order) - 0.55)
    apply_axis_spec(axis, panel)
    draw_reference(axis, float(panel["reference"]))
    style_axis(axis)
    add_panel_labels(fig, spec)
    return fig


FIGURE_RENDERERS: dict[str, Callable[[BundleReader, Mapping[str, Any], pd.DataFrame], Figure]] = {
    "s1": render_s1,
    "s2": render_s2,
    "s3": render_s3,
    "s4": render_s4,
    "s5": render_s5,
    "s6": render_s6,
    "s7": render_s7,
}


__all__ = ["FIGURE_RENDERERS"]
