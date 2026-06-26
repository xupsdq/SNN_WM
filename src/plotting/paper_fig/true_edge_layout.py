from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from matplotlib.transforms import Bbox

from src.plotting.paper_fig.utils import write_json


SCHEMATIC_TYPES = {
    "manual_schematic",
    "manual_or_programmatic_schematic",
    "programmatic_or_manual_schematic",
    "two_item_episode_schematic",
    "multi_item_sequence_schematic",
}

LOCAL_GAP_MM = 0.5
X_GAP_MM = 0.5
X_LABEL_GAP_MM = 0.5
Y_TICK_AXIS_GAP_MM = 1.0
Y_LABEL_GAP_MM = 0.5
LEGEND_GAP_MM = 0.5
MIN_AXIS_MM = 8.0


DEFAULT_SEMANTIC_LAYOUTS: dict[str, dict[str, Any]] = {
    "fig1": {
        "row_groups": [["A"], ["B", "C"], ["D", "E"]],
        "column_groups": [["A"], ["B", "D"], ["C", "E"]],
    },
    "fig2": {
        "row_groups": [["A"], ["B", "C", "D"], ["E", "F"]],
        "column_groups": [["A"], ["B", "E"], ["C"], ["D"], ["F"]],
    },
    "fig3": {
        "row_groups": [["A", "B"], ["C", "D", "E"]],
        "column_groups": [["A", "C"], ["B", "E"], ["D"]],
    },
    "fig4": {
        "row_groups": [["A", "B"], ["C", "D"], ["E", "F"]],
        "column_groups": [["A", "C", "E"], ["B", "D", "F"]],
    },
    "fig5": {
        "row_groups": [["A", "B"], ["C", "D", "E"]],
        "column_groups": [["A", "C"], ["B", "E"], ["D"]],
    },
    "fig6": {
        "row_groups": [["A", "B", "C"], ["D", "E"]],
        "column_groups": [["A", "D"], ["C", "E"], ["B"]],
    },
}


def semantic_layout_for_figure(fig_id: str) -> dict[str, Any]:
    layout = dict(DEFAULT_SEMANTIC_LAYOUTS.get(str(fig_id).lower(), {}))
    if not layout:
        return {}
    layout.update(
        {
            "x_gap_mm": X_GAP_MM,
            "x_label_gap_mm": X_LABEL_GAP_MM,
            "y_tick_axis_gap_mm": Y_TICK_AXIS_GAP_MM,
            "y_label_gap_mm": Y_LABEL_GAP_MM,
            "legend_gap_mm": LEGEND_GAP_MM,
            "local_gap_mm": LOCAL_GAP_MM,
        }
    )
    return layout


def install_true_edge_callbacks(fig, spec: Mapping[str, Any]) -> None:
    fig.paper_fig_finalize_layout = finalize_true_edge_layout
    fig.paper_fig_write_layout_report = write_true_edge_report
    fig.paper_fig_semantic_layout = semantic_layout_for_figure(str(spec.get("figure_id", "")))


def seed_axes_box(panel_id: str, panel: Mapping[str, Any]) -> dict[str, float]:
    target = _box(panel.get("position_mm") or {})
    if str(panel.get("panel_type", "")).lower() in SCHEMATIC_TYPES:
        return target
    w = target["w"]
    h = target["h"]
    # ponytail: one seed, measured solve below owns the real layout.
    left = min(13.0, max(7.0, w * 0.20))
    right = min(3.0, max(1.2, w * 0.04))
    top = min(7.0, max(5.5, h * 0.15))
    bottom = min(12.0, max(8.0, h * 0.22))
    return {
        "x": target["x"] + left,
        "y": target["y"] + top,
        "w": max(8.0, w - left - right),
        "h": max(8.0, h - top - bottom),
    }


def finalize_true_edge_layout(fig, axes: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    panels = spec.get("panels") or {}
    canvas = spec.get("canvas_mm") or {}
    if not canvas:
        return
    layout = _semantic_layout(fig, spec)
    if layout.get("row_groups") or layout.get("column_groups"):
        _finalize_semantic_layout(fig, axes, panels, canvas, spec)
        return
    fig.canvas.draw()
    for panel_id, ax in axes.items():
        panel = panels.get(panel_id) or {}
        if str(panel.get("panel_type", "")).lower() in SCHEMATIC_TYPES:
            continue
        target = _box(panel.get("position_mm") or {})
        _fit_ylabel_gaps(fig, ax)
        for _ in range(5):
            fig.canvas.draw()
            measured = measure_panel_bbox(fig, ax)
            if measured is None:
                break
            axes_box = bbox_to_mm(fig, ax.bbox)
            left = axes_box["left"] - measured["left"]
            right = measured["right"] - axes_box["right"]
            next_axes = {
                "x": target["x"] + left,
                "y": axes_box["y"],
                "w": max(6.0, target["w"] - left - right),
                "h": axes_box["h"],
            }
            _set_axes_mm(fig, ax, next_axes, canvas)
        _fit_ylabel_gaps(fig, ax)
    if str(spec.get("figure_id", "")).lower() == "fig2":
        _finalize_fig2_semantic_boxes(fig, axes, panels, canvas)
    if str(spec.get("figure_id", "")).lower() == "fig3":
        _finalize_fig3_panel_targets(fig, axes, panels, canvas)
    fig.canvas.draw()


def write_true_edge_report(fig, axes: Mapping[str, Any], spec: Mapping[str, Any], output_dir: str | Path) -> None:
    fig.canvas.draw()
    canvas = spec.get("canvas_mm") or {}
    panels = spec.get("panels") or {}
    payload: dict[str, Any] = {
        "figure_id": spec.get("figure_id"),
        "canvas_mm": canvas,
        "method": "measured visible artists with matplotlib renderer; raw axes rectangles are reported separately",
        "target_y_title_tick_gap_mm": Y_LABEL_GAP_MM,
        "gap_targets_mm": {
            "x_tick_to_axis": X_GAP_MM,
            "x_label_to_x_tick": X_LABEL_GAP_MM,
            "y_tick_to_axis": Y_TICK_AXIS_GAP_MM,
            "y_label_to_y_tick": Y_LABEL_GAP_MM,
            "top_legend_text_to_axis_top": LEGEND_GAP_MM,
        },
        "panels": {},
    }
    for panel_id, ax in axes.items():
        panel = panels.get(panel_id) or {}
        measured = measure_panel_bbox(fig, ax)
        slot_box = target_bbox_bottom_left(panel.get("position_mm") or {}, canvas)
        semantic = _semantic_panel_report(fig, ax)
        child_boxes = [bbox_to_mm(fig, child.bbox) for child in _owned_axes(fig, ax)[1:]]
        plot_box = _plot_area_bbox(fig, ax)
        axes_box = bbox_to_mm(fig, ax.bbox)
        residual = _residual(slot_box, measured) if measured else None
        payload["panels"][panel_id] = {
            "slot_bbox_mm": slot_box,
            "target_full_bbox_mm": slot_box,
            "measured_full_bbox_mm": measured,
            "axis_rect_mm": plot_box,
            "axes_bbox_mm": axes_box,
            "parent_axes_bbox_mm": axes_box,
            "child_axes_bbox_mm": child_boxes,
            "plot_area_bbox_mm": plot_box,
            "legend_bbox_mm": legend_bbox_mm(fig, ax),
            "legend_text_bbox_mm": legend_text_bbox_mm(fig, ax),
            "legend_to_plot_gap_mm": legend_to_plot_gap_mm(fig, ax, plot_box),
            "residual_mm": residual,
            "y_title_tick_gap_mm": ylabel_tick_gaps_mm(fig, ax),
            "left_stack_mm": semantic["left_stack_mm"],
            "bottom_stack_mm": semantic["bottom_stack_mm"],
            "top_stack_mm": semantic["top_stack_mm"],
            "legend_stack_mm": semantic["legend_stack_mm"],
            "panel_top_stack_mm": semantic["panel_top_stack_mm"],
            "row_global_top_stack_mm": semantic["row_global_top_stack_mm"],
            "right_stack_mm": semantic["right_stack_mm"],
            "gap_measurements_mm": semantic["gap_measurements_mm"],
            "legend_center_delta_mm": semantic["legend_center_delta_mm"],
            "top_content_items": semantic["top_content_items"],
            "declared_extra_artists": semantic["declared_extra_artists"],
        }
    layout = _semantic_layout(fig, spec)
    if layout.get("row_groups") or layout.get("column_groups"):
        payload["layout_contract"] = _semantic_contract_report(fig, axes, panels, canvas, spec)
    write_json(payload, Path(output_dir) / "true_edge_measurements.json")


def measure_panel_bbox(fig, ax) -> dict[str, float] | None:
    renderer = fig.canvas.get_renderer()
    boxes: list[Bbox] = []
    for owned in _owned_axes(fig, ax):
        if getattr(owned, "axison", False):
            boxes.append(owned.bbox)
        boxes.extend(_artist_boxes(renderer, owned))
    if not boxes:
        return None
    return bbox_to_mm(fig, Bbox.union(boxes))


def bbox_to_mm(fig, bbox) -> dict[str, float]:
    width_mm = fig.get_figwidth() * 25.4
    height_mm = fig.get_figheight() * 25.4
    left = bbox.x0 / fig.bbox.width * width_mm
    right = bbox.x1 / fig.bbox.width * width_mm
    bottom = bbox.y0 / fig.bbox.height * height_mm
    top = bbox.y1 / fig.bbox.height * height_mm
    return {
        "left": float(left),
        "bottom": float(bottom),
        "right": float(right),
        "top": float(top),
        "width": float(right - left),
        "height": float(top - bottom),
        "x": float(left),
        "y": float(height_mm - top),
        "w": float(right - left),
        "h": float(top - bottom),
    }


def target_bbox_bottom_left(box: Mapping[str, Any], canvas: Mapping[str, Any]) -> dict[str, float]:
    x = float(box.get("x", 0.0))
    y = float(box.get("y", 0.0))
    w = float(box.get("w", box.get("width", 0.0)))
    h = float(box.get("h", box.get("height", 0.0)))
    canvas_h = float(canvas.get("height", 0.0))
    top = canvas_h - y
    bottom = top - h
    return {"left": x, "bottom": bottom, "right": x + w, "top": top, "width": w, "height": h, "x": x, "y": y, "w": w, "h": h}


def ylabel_tick_gaps_mm(fig, ax) -> list[float]:
    gaps: list[float] = []
    for owned in _owned_axes(fig, ax):
        gap = _ylabel_tick_gap_mm(fig, owned)
        if gap is not None:
            gaps.append(float(gap))
    return gaps


def legend_bbox_mm(fig, ax) -> dict[str, float] | None:
    legend = ax.get_legend()
    if legend is None or not legend.get_visible():
        return None
    return bbox_to_mm(fig, legend.get_window_extent(fig.canvas.get_renderer()))


def legend_text_bbox_mm(fig, ax) -> dict[str, float] | None:
    legend = ax.get_legend()
    if legend is None or not legend.get_visible():
        return None
    renderer = fig.canvas.get_renderer()
    boxes = []
    for text in legend.get_texts():
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.width > 0 and box.height > 0:
                boxes.append(box)
    if not boxes:
        return None
    return bbox_to_mm(fig, Bbox.union(boxes))


def legend_to_plot_gap_mm(fig, ax, plot_box: Mapping[str, float] | None = None) -> float | None:
    text_box = legend_text_bbox_mm(fig, ax)
    if text_box is None:
        return None
    if plot_box is None:
        plot_box = _plot_area_bbox(fig, ax)
    return float(text_box["bottom"] - plot_box["top"])


def _fit_ylabel_gaps(fig, ax, target_mm: float = 0.5) -> None:
    for owned in _owned_axes(fig, ax):
        for _ in range(5):
            fig.canvas.draw()
            gap = _ylabel_tick_gap_mm(fig, owned)
            if gap is None or abs(gap - target_mm) <= 0.08:
                break
            if getattr(owned.yaxis, "_autolabelpos", True):
                delta_pt = (gap - target_mm) / 25.4 * 72.0
                owned.yaxis.labelpad = max(-20.0, min(8.0, float(owned.yaxis.labelpad) - delta_pt))
            else:
                x, y = owned.yaxis.label.get_position()
                axes_w_mm = owned.bbox.width / fig.dpi * 25.4
                if axes_w_mm <= 0:
                    break
                owned.yaxis.set_label_coords(float(x) + (gap - target_mm) / axes_w_mm, float(y))


def _ylabel_tick_gap_mm(fig, ax) -> float | None:
    renderer = fig.canvas.get_renderer()
    label = ax.yaxis.label
    if not label.get_visible() or not label.get_text():
        return None
    label_box = label.get_window_extent(renderer)
    tick_boxes = []
    for tick in ax.get_yticklabels():
        if tick.get_visible() and tick.get_text():
            box = tick.get_window_extent(renderer)
            if box.width > 0 and box.height > 0:
                tick_boxes.append(box)
    if not tick_boxes:
        return None
    return (min(box.x0 for box in tick_boxes) - label_box.x1) / fig.dpi * 25.4


def _owned_axes(fig, ax) -> list[Any]:
    owned = [ax]
    for child in getattr(ax, "child_axes", []):
        if child not in owned:
            owned.append(child)
    for child in getattr(ax, "paper_fig_child_axes", []):
        if child not in owned:
            owned.append(child)
    parent = ax.bbox
    for other in fig.axes:
        if other is ax:
            continue
        if parent.contains(other.bbox.x0, other.bbox.y0) and parent.contains(other.bbox.x1, other.bbox.y1):
            owned.append(other)
    cbar = getattr(ax, "paper_fig_colorbar_ax", None)
    if cbar is not None and cbar not in owned:
        owned.append(cbar)
    return owned


def _plot_area_bbox(fig, ax) -> dict[str, float]:
    plotted = [owned_ax.bbox for owned_ax in _primary_plot_axes(ax)]
    if not plotted:
        plotted = [ax.bbox]
    return bbox_to_mm(fig, Bbox.union(plotted))


def _primary_plot_axes(ax) -> list[Any]:
    children = [child for child in getattr(ax, "paper_fig_child_axes", []) if getattr(child, "axison", False)]
    if children:
        return children
    return [ax] if getattr(ax, "axison", False) else []


def _primary_plot_bounds_in_parent(fig, ax) -> tuple[float, float, float, float] | None:
    children = _primary_plot_axes(ax)
    if not children or children == [ax]:
        return None
    declared = getattr(ax, "paper_fig_inner_axes_bounds", None)
    if declared:
        left = min(float(box[0]) for box in declared)
        bottom = min(float(box[1]) for box in declared)
        right = max(float(box[0]) + float(box[2]) for box in declared)
        top = max(float(box[1]) + float(box[3]) for box in declared)
        return left, bottom, right, top
    fig.canvas.draw()
    parent = ax.bbox
    if parent.width <= 0 or parent.height <= 0:
        return None
    left = min((child.bbox.x0 - parent.x0) / parent.width for child in children)
    bottom = min((child.bbox.y0 - parent.y0) / parent.height for child in children)
    right = max((child.bbox.x1 - parent.x0) / parent.width for child in children)
    top = max((child.bbox.y1 - parent.y0) / parent.height for child in children)
    return float(left), float(bottom), float(right), float(top)


def _has_top_legend(ax) -> bool:
    return bool(getattr(ax, "paper_fig_legend_above_plot", False))


def _legend_counts_as_top_stack(fig, ax, axis_box: Mapping[str, float] | None = None, legend_box: Mapping[str, float] | None = None) -> bool:
    if _has_top_legend(ax):
        return True
    if legend_box is None:
        legend_box = legend_bbox_mm(fig, ax)
    if legend_box is None:
        return False
    if axis_box is None:
        axis_box = _plot_area_bbox(fig, ax)
    return float(legend_box["bottom"]) >= float(axis_box["top"]) - 0.05


def _declared_right_stack_mm(fig, ax, axis_box: Mapping[str, float], renderer) -> tuple[float, list[str]]:
    stack = float(getattr(ax, "paper_fig_panel_right_stack_mm", getattr(ax, "paper_fig_panel_right_extra_mm", 0.0)) or 0.0)
    extras: list[str] = []
    if stack > 0.0:
        extras.append("declared_right")

    cbar = getattr(ax, "paper_fig_colorbar_ax", None)
    if cbar is not None:
        boxes = [cbar.bbox, *_artist_boxes(renderer, cbar, include_legend=True)]
        cbar_box = bbox_to_mm(fig, Bbox.union(boxes))
        stack = max(stack, max(0.0, float(cbar_box["right"]) - float(axis_box["right"])))
        extras.append("colorbar")

    legend_box = legend_bbox_mm(fig, ax)
    if bool(getattr(ax, "paper_fig_legend_right_of_plot", False)) and legend_box is not None:
        stack = max(stack, max(0.0, float(legend_box["right"]) - float(axis_box["right"])))
        extras.append("right_legend")

    return stack, extras


def _artist_boxes(renderer, ax, *, include_legend: bool = True) -> list[Bbox]:
    artists: list[Any] = []
    if ax.axison:
        artists.extend(ax.get_xticklabels())
        artists.extend(ax.get_yticklabels())
        artists.extend([ax.xaxis.label, ax.yaxis.label, ax.title])
    artists.extend(ax.texts)
    artists.extend(ax.images)
    legend = ax.get_legend()
    if include_legend and legend is not None:
        artists.append(legend)
    boxes: list[Bbox] = []
    for artist in artists:
        if not artist.get_visible():
            continue
        try:
            box = artist.get_window_extent(renderer)
        except Exception:
            continue
        if box.width > 0 and box.height > 0:
            boxes.append(box)
    return boxes


def _finalize_fig2_semantic_boxes(fig, axes: Mapping[str, Any], panels: Mapping[str, Any], canvas: Mapping[str, Any]) -> None:
    for panel_id in ("B", "C", "D"):
        ax = axes.get(panel_id)
        panel = panels.get(panel_id) or {}
        if ax is None:
            continue
        target = target_bbox_bottom_left(panel.get("position_mm") or {}, canvas)
        axes_box = bbox_to_mm(fig, ax.bbox)
        next_box = {
            "x": axes_box["x"],
            "y": float(panel.get("position_mm", {}).get("y", 0.0)),
            "w": axes_box["w"],
            "h": target["top"] - (target["bottom"] + 5.5),
        }
        _set_axes_mm(fig, ax, next_box, canvas)
        _fit_xtick_bottom(fig, ax, target["bottom"])
        _fit_ylabel_gaps(fig, ax)

    e_ax = axes.get("E")
    e_panel = panels.get("E") or {}
    if e_ax is not None:
        target = target_bbox_bottom_left(e_panel.get("position_mm") or {}, canvas)
        axes_box = bbox_to_mm(fig, e_ax.bbox)
        top_inset = 3.1
        next_box = {
            "x": axes_box["x"],
            "y": float(e_panel.get("position_mm", {}).get("y", 0.0)) + top_inset,
            "w": axes_box["w"],
            "h": max(20.0, target["top"] - top_inset - (target["bottom"] + 5.5)),
        }
        _set_axes_mm(fig, e_ax, next_box, canvas)
        _fit_xtick_bottom(fig, e_ax, target["bottom"])
        _fit_ylabel_gaps(fig, e_ax)

    f_ax = axes.get("F")
    f_panel = panels.get("F") or {}
    if f_ax is not None:
        pos = _box(f_panel.get("position_mm") or {})
        _set_axes_mm(fig, f_ax, pos, canvas)
        for child in _owned_axes(fig, f_ax)[1:]:
            _fit_xtick_bottom(fig, child, target_bbox_bottom_left(f_panel.get("position_mm") or {}, canvas)["bottom"])
        _fit_ylabel_gaps(fig, f_ax)


def _finalize_fig3_panel_targets(fig, axes: Mapping[str, Any], panels: Mapping[str, Any], canvas: Mapping[str, Any]) -> None:
    _finalize_semantic_layout(fig, axes, panels, canvas, {"figure_id": "fig3"})


def _finalize_semantic_layout(fig, axes: Mapping[str, Any], panels: Mapping[str, Any], canvas: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    layout = _semantic_layout(fig, spec)
    row_groups = layout["row_groups"]
    column_groups = layout["column_groups"]
    panel_ids = [pid for group in row_groups for pid in group if pid in axes and not _is_schematic_panel(panels.get(pid) or {})]
    canvas_h = float(canvas["height"])
    row_edges: dict[str, tuple[float, float]] = {}
    left_edges: dict[str, float] = {}
    right_edges: dict[str, float] = {}

    for _ in range(4):
        fig.canvas.draw()
        for panel_id in panel_ids:
            _fit_semantic_gaps(fig, axes[panel_id], layout)
        fig.canvas.draw()
        for panel_id in panel_ids:
            _fit_top_legend_gap(fig, axes[panel_id], float(layout["legend_gap_mm"]))
        fig.canvas.draw()
        stacks = {pid: _semantic_panel_report(fig, axes[pid]) for pid in panel_ids}

        for group in row_groups:
            members = [pid for pid in group if pid in axes and pid in panel_ids]
            if not members:
                continue
            row_slot = _union_mm_boxes([target_bbox_bottom_left((panels.get(pid) or {}).get("position_mm") or {}, canvas) for pid in members])
            bottom = row_slot["bottom"] + max(float(stacks[pid]["bottom_stack_mm"]) for pid in members)
            top_base = row_slot["top"] - max(float(stacks[pid]["row_global_top_stack_mm"]) for pid in members)
            for pid in members:
                slot = target_bbox_bottom_left((panels.get(pid) or {}).get("position_mm") or {}, canvas)
                top = min(top_base, slot["top"] - float(stacks[pid]["panel_top_stack_mm"]))
                row_edges[pid] = (bottom, top)

        for group in column_groups:
            members = [pid for pid in group if pid in axes and pid in panel_ids]
            if not members:
                continue
            slots = {pid: target_bbox_bottom_left((panels.get(pid) or {}).get("position_mm") or {}, canvas) for pid in members}
            align_left = max(slot["left"] for slot in slots.values()) - min(slot["left"] for slot in slots.values()) <= 0.25
            align_right = max(slot["right"] for slot in slots.values()) - min(slot["right"] for slot in slots.values()) <= 0.25
            if align_left:
                left = min(slot["left"] for slot in slots.values()) + max(float(stacks[pid]["left_stack_mm"]) for pid in members)
                left_edges.update({pid: left for pid in members})
            if align_right:
                right = max(slot["right"] for slot in slots.values()) - max(float(stacks[pid]["right_stack_mm"]) for pid in members)
                right_edges.update({pid: right for pid in members})

        for panel_id in panel_ids:
            slot = target_bbox_bottom_left((panels.get(panel_id) or {}).get("position_mm") or {}, canvas)
            stack = stacks[panel_id]
            bottom, top = row_edges.get(
                panel_id,
                (
                    slot["bottom"] + stack["bottom_stack_mm"],
                    slot["top"] - stack["row_global_top_stack_mm"] - stack["panel_top_stack_mm"],
                ),
            )
            left = left_edges.get(panel_id, slot["left"] + stack["left_stack_mm"])
            right = right_edges.get(panel_id, slot["right"] - stack["right_stack_mm"])
            _set_panel_plot_area_mm(
                fig,
                axes[panel_id],
                {
                    "x": left,
                    "y": canvas_h - top,
                    "w": max(MIN_AXIS_MM, right - left),
                    "h": max(MIN_AXIS_MM, top - bottom),
                },
                canvas,
            )
    fig.canvas.draw()
    for panel_id in panel_ids:
        _fit_semantic_gaps(fig, axes[panel_id], layout)
        _fit_top_legend_gap(fig, axes[panel_id], float(layout["legend_gap_mm"]))
    fig.canvas.draw()


def _is_schematic_panel(panel: Mapping[str, Any]) -> bool:
    return str(panel.get("panel_type", "")).lower() in SCHEMATIC_TYPES


def _semantic_layout(fig, spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    layout = dict(getattr(fig, "paper_fig_semantic_layout", None) or {})
    if not layout and spec is not None:
        layout = semantic_layout_for_figure(str(spec.get("figure_id", "")))
    return {
        "row_groups": layout.get("row_groups") or [],
        "column_groups": layout.get("column_groups") or [],
        "local_gap_mm": float(layout.get("local_gap_mm", LOCAL_GAP_MM)),
        "x_gap_mm": float(layout.get("x_gap_mm", X_GAP_MM)),
        "x_label_gap_mm": float(layout.get("x_label_gap_mm", X_LABEL_GAP_MM)),
        "y_tick_axis_gap_mm": float(layout.get("y_tick_axis_gap_mm", Y_TICK_AXIS_GAP_MM)),
        "y_label_gap_mm": float(layout.get("y_label_gap_mm", Y_LABEL_GAP_MM)),
        "legend_gap_mm": float(layout.get("legend_gap_mm", LEGEND_GAP_MM)),
    }


def _fit_semantic_gaps(fig, ax, layout: Mapping[str, Any] | None = None) -> None:
    layout = layout or {}
    x_gap = float(layout.get("x_gap_mm", X_GAP_MM))
    x_label_gap = float(layout.get("x_label_gap_mm", X_LABEL_GAP_MM))
    y_tick_gap = float(layout.get("y_tick_axis_gap_mm", Y_TICK_AXIS_GAP_MM))
    y_label_gap = float(layout.get("y_label_gap_mm", Y_LABEL_GAP_MM))
    for owned in _owned_axes(fig, ax):
        for _ in range(4):
            fig.canvas.draw()
            changed = False
            changed = _fit_tick_axis_gap(fig, owned, "x", x_gap) or changed
            changed = _fit_tick_axis_gap(fig, owned, "y", y_tick_gap) or changed
            changed = _fit_xlabel_tick_gap(fig, owned, x_label_gap) or changed
            _fit_ylabel_gaps(fig, owned, y_label_gap)
            if not changed:
                break


def _fit_tick_axis_gap(fig, ax, axis_name: str, target_mm: float) -> bool:
    gap = _tick_axis_gap_mm(fig, ax, axis_name)
    if gap is None or abs(gap - target_mm) <= 0.08:
        return False
    attr = f"paper_fig_{axis_name}tick_pad_pt"
    current = float(getattr(ax, attr, 1.0))
    next_pad = max(-8.0, min(18.0, current + (target_mm - gap) / 25.4 * 72.0))
    ax.tick_params(axis=axis_name, pad=next_pad)
    setattr(ax, attr, next_pad)
    return True


def _fit_xlabel_tick_gap(fig, ax, target_mm: float) -> bool:
    gap = _xlabel_tick_gap_mm(fig, ax)
    if gap is None or abs(gap - target_mm) <= 0.08:
        return False
    ax.xaxis.labelpad = max(-12.0, min(18.0, float(ax.xaxis.labelpad) + (target_mm - gap) / 25.4 * 72.0))
    return True


def _fit_top_legend_gap(fig, ax, target_mm: float = LOCAL_GAP_MM) -> None:
    legend = ax.get_legend()
    if legend is None or not legend.get_visible() or not _legend_counts_as_top_stack(fig, ax):
        return
    axes_h_mm = bbox_to_mm(fig, ax.bbox)["height"]
    if axes_h_mm <= 0:
        return
    plot_bounds = _primary_plot_bounds_in_parent(fig, ax)
    if plot_bounds is None:
        anchor_x = 0.5
        plot_top = 1.0
    else:
        anchor_x = (plot_bounds[0] + plot_bounds[2]) / 2.0
        plot_top = plot_bounds[3]
    if hasattr(legend, "set_loc"):
        legend.set_loc("lower center")
    else:
        legend._loc = 8
    current = float(getattr(legend, "paper_fig_y_offset_mm", target_mm))
    for _ in range(4):
        legend.set_bbox_to_anchor((anchor_x, plot_top + current / axes_h_mm), transform=ax.transAxes)
        fig.canvas.draw()
        gap = legend_to_plot_gap_mm(fig, ax, _plot_area_bbox(fig, ax))
        if gap is None or abs(gap - target_mm) <= 0.08:
            break
        current = max(-4.0, min(14.0, current + target_mm - gap))
        legend.paper_fig_y_offset_mm = current


def _gap_axis(fig, ax, kind: str):
    fig.canvas.draw()
    candidates = _primary_plot_axes(ax) or [ax]
    for candidate in candidates:
        if kind == "x_tick" and _tick_label_bbox_mm(fig, candidate, "x") is not None:
            return candidate
        if kind == "x_label" and _axis_label_bbox_mm(fig, candidate, "x") is not None:
            return candidate
        if kind == "y_tick" and _tick_label_bbox_mm(fig, candidate, "y") is not None:
            return candidate
        if kind == "y_label" and _axis_label_bbox_mm(fig, candidate, "y") is not None:
            return candidate
    return ax


def _semantic_panel_report(fig, ax) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_box = _plot_area_bbox(fig, ax)
    owned = _owned_axes(fig, ax)
    legend_box = legend_bbox_mm(fig, ax)
    legend_text_box = legend_text_bbox_mm(fig, ax)
    top_legend = _legend_counts_as_top_stack(fig, ax, axis_box, legend_box)
    semantic_boxes: list[Bbox] = []
    for owned_ax in owned:
        if getattr(owned_ax, "axison", False):
            semantic_boxes.append(owned_ax.bbox)
        semantic_boxes.extend(_artist_boxes(renderer, owned_ax, include_legend=not top_legend))
    semantic_box = bbox_to_mm(fig, Bbox.union(semantic_boxes)) if semantic_boxes else axis_box
    legend_stack = 0.0
    legend_center_delta = None
    top_items: list[dict[str, Any]] = []
    if top_legend and legend_box is not None and legend_text_box is not None:
        legend_stack = max(0.0, LEGEND_GAP_MM + float(legend_box["top"] - legend_text_box["bottom"]))
        legend_center_delta = abs(((legend_box["left"] + legend_box["right"]) / 2.0) - ((axis_box["left"] + axis_box["right"]) / 2.0))
        top_items.append({
            "kind": "top_legend",
            "scope": "panel_local",
            "bbox_mm": legend_box,
            "gap_to_axis_mm": legend_to_plot_gap_mm(fig, ax, axis_box),
        })
    panel_top_extra = float(getattr(ax, "paper_fig_panel_top_stack_mm", getattr(ax, "paper_fig_panel_top_extra_mm", 0.0)) or 0.0)
    if panel_top_extra > 0.0:
        top_items.append({
            "kind": "declared_panel_top",
            "scope": "panel_local",
            "bbox_mm": None,
            "gap_to_axis_mm": None,
        })
    row_global_top_stack = float(getattr(ax, "paper_fig_row_global_top_stack_mm", 0.0))
    if row_global_top_stack > 0.0:
        top_items.append({
            "kind": "declared_row_global_top",
            "scope": "row_global",
            "bbox_mm": None,
            "gap_to_axis_mm": None,
        })
    panel_top_stack = legend_stack + panel_top_extra
    right_stack, declared_extra = _declared_right_stack_mm(fig, ax, axis_box, renderer)
    return {
        "left_stack_mm": max(0.0, axis_box["left"] - semantic_box["left"]),
        "bottom_stack_mm": max(0.0, axis_box["bottom"] - semantic_box["bottom"]),
        "top_stack_mm": row_global_top_stack,
        "legend_stack_mm": legend_stack,
        "panel_top_stack_mm": panel_top_stack,
        "row_global_top_stack_mm": row_global_top_stack,
        "right_stack_mm": right_stack,
        "gap_measurements_mm": {
            "x_tick_to_axis": _tick_axis_gap_mm(fig, _gap_axis(fig, ax, "x_tick"), "x"),
            "x_label_to_x_tick": _xlabel_tick_gap_mm(fig, _gap_axis(fig, ax, "x_label")),
            "y_tick_to_axis": _tick_axis_gap_mm(fig, _gap_axis(fig, ax, "y_tick"), "y"),
            "y_label_to_y_tick": _ylabel_tick_gap_mm(fig, _gap_axis(fig, ax, "y_label")),
            "top_legend_text_to_axis_top": legend_to_plot_gap_mm(fig, ax, axis_box) if top_legend else None,
        },
        "legend_center_delta_mm": legend_center_delta,
        "top_content_items": top_items,
        "declared_extra_artists": declared_extra,
    }


def _semantic_contract_report(fig, axes: Mapping[str, Any], panels: Mapping[str, Any], canvas: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    layout = _semantic_layout(fig, spec)
    panel_reports = {pid: _semantic_panel_report(fig, ax) for pid, ax in axes.items()}
    row_reports = {}
    for index, group in enumerate(layout["row_groups"], start=1):
        members = [pid for pid in group if pid in axes and not _is_schematic_panel(panels.get(pid) or {})]
        if not members:
            continue
        slots = [target_bbox_bottom_left((panels.get(pid) or {}).get("position_mm") or {}, canvas) for pid in members]
        measured = [measure_panel_bbox(fig, axes[pid]) for pid in members]
        axis_boxes = [_plot_area_bbox(fig, axes[pid]) for pid in members]
        row_bottom_stack = max(float(panel_reports[pid]["bottom_stack_mm"]) for pid in members)
        row_top_stack = max(float(panel_reports[pid]["row_global_top_stack_mm"]) for pid in members)
        row_slot = _union_mm_boxes(slots)
        row_reports[f"row_{index}_{''.join(members)}"] = {
            "panel_ids": members,
            "row_slot_bbox_mm": row_slot,
            "row_union_measured_bbox_mm": _union_mm_boxes([box for box in measured if box is not None]),
            "axis_bottom_range_mm": _range_mm(axis_boxes, "bottom"),
            "axis_top_range_mm": _range_mm(axis_boxes, "top"),
            "axis_top_base_mm": row_slot["top"] - row_top_stack,
            "row_bottom_stack_mm": row_bottom_stack,
            "row_top_stack_mm": row_top_stack,
            "panel_top_stack_mm": {pid: panel_reports[pid]["panel_top_stack_mm"] for pid in members},
            "row_global_top_stack_mm": {pid: panel_reports[pid]["row_global_top_stack_mm"] for pid in members},
        }
    column_reports = {}
    for index, group in enumerate(layout["column_groups"], start=1):
        members = [pid for pid in group if pid in axes and not _is_schematic_panel(panels.get(pid) or {})]
        if not members:
            continue
        slots = [target_bbox_bottom_left((panels.get(pid) or {}).get("position_mm") or {}, canvas) for pid in members]
        measured = [measure_panel_bbox(fig, axes[pid]) for pid in members]
        axis_boxes = [_plot_area_bbox(fig, axes[pid]) for pid in members]
        align_left = _range_mm(slots, "left") <= 0.25
        align_right = _range_mm(slots, "right") <= 0.25
        column_reports[f"column_{index}_{''.join(members)}"] = {
            "panel_ids": members,
            "aligned_edges": [edge for edge, enabled in (("left", align_left), ("right", align_right)) if enabled],
            "column_slot_horizontal_bbox_mm": _horizontal_union_mm(slots),
            "column_union_measured_horizontal_bbox_mm": _horizontal_union_mm([box for box in measured if box is not None]),
            "axis_left_range_mm": _range_mm(axis_boxes, "left"),
            "axis_right_range_mm": _range_mm(axis_boxes, "right"),
            "col_left_stack_mm": max(float(panel_reports[pid]["left_stack_mm"]) for pid in members),
            "col_right_stack_mm": max(float(panel_reports[pid]["right_stack_mm"]) for pid in members),
        }
    return {
        "contract": "semantic_layout_contract_v1",
        "row_groups": layout["row_groups"],
        "column_groups": layout["column_groups"],
        "gap_targets_mm": {
            "x_gap": layout["x_gap_mm"],
            "x_label_gap": layout["x_label_gap_mm"],
            "y_tick_axis_gap": layout["y_tick_axis_gap_mm"],
            "y_label_gap": layout["y_label_gap_mm"],
            "legend_gap": layout["legend_gap_mm"],
        },
        "row_reports": row_reports,
        "column_reports": column_reports,
    }


def _tick_label_bbox_mm(fig, ax, axis_name: str) -> dict[str, float] | None:
    renderer = fig.canvas.get_renderer()
    labels = ax.get_xticklabels() if axis_name == "x" else ax.get_yticklabels()
    boxes = []
    for label in labels:
        if label.get_visible() and label.get_text():
            box = label.get_window_extent(renderer)
            if box.width > 0 and box.height > 0:
                boxes.append(box)
    return bbox_to_mm(fig, Bbox.union(boxes)) if boxes else None


def _axis_label_bbox_mm(fig, ax, axis_name: str) -> dict[str, float] | None:
    label = ax.xaxis.label if axis_name == "x" else ax.yaxis.label
    if not label.get_visible() or not label.get_text():
        return None
    box = label.get_window_extent(fig.canvas.get_renderer())
    return bbox_to_mm(fig, box) if box.width > 0 and box.height > 0 else None


def _tick_axis_gap_mm(fig, ax, axis_name: str) -> float | None:
    ticks = _tick_label_bbox_mm(fig, ax, axis_name)
    if ticks is None:
        return None
    axis_box = bbox_to_mm(fig, ax.bbox)
    if axis_name == "x":
        return float(axis_box["bottom"] - ticks["top"])
    return float(axis_box["left"] - ticks["right"])


def _xlabel_tick_gap_mm(fig, ax) -> float | None:
    ticks = _tick_label_bbox_mm(fig, ax, "x")
    label = _axis_label_bbox_mm(fig, ax, "x")
    if ticks is None or label is None:
        return None
    return float(ticks["bottom"] - label["top"])


def _union_mm_boxes(boxes: list[Mapping[str, float]]) -> dict[str, float]:
    if not boxes:
        return {"left": 0.0, "bottom": 0.0, "right": 0.0, "top": 0.0, "width": 0.0, "height": 0.0, "x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    left = min(float(box["left"]) for box in boxes)
    bottom = min(float(box["bottom"]) for box in boxes)
    right = max(float(box["right"]) for box in boxes)
    top = max(float(box["top"]) for box in boxes)
    return {"left": left, "bottom": bottom, "right": right, "top": top, "width": right - left, "height": top - bottom, "x": left, "y": bottom, "w": right - left, "h": top - bottom}


def _horizontal_union_mm(boxes: list[Mapping[str, float]]) -> dict[str, float]:
    union = _union_mm_boxes(boxes)
    return {"left": union["left"], "right": union["right"], "width": union["width"]}


def _range_mm(boxes: list[Mapping[str, float]], key: str) -> float:
    values = [float(box[key]) for box in boxes]
    return max(values) - min(values) if values else 0.0


def _fit_xtick_bottom(fig, ax, target_bottom_mm: float) -> None:
    current = float(getattr(ax, "paper_fig_xtick_pad_pt", 3.5))
    label_current = float(getattr(ax.xaxis, "labelpad", 4.0))
    for _ in range(6):
        fig.canvas.draw()
        bottom = _xcontent_bottom_mm(fig, ax)
        if bottom is None:
            break
        delta = bottom - target_bottom_mm
        if abs(delta) <= 0.2:
            break
        if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text():
            label_current = max(-12.0, min(12.0, label_current + delta / 25.4 * 72.0))
            ax.xaxis.labelpad = label_current
        else:
            current = max(-8.0, min(24.0, current + delta / 25.4 * 72.0))
            ax.tick_params(axis="x", pad=current)
            ax.paper_fig_xtick_pad_pt = current


def _xcontent_bottom_mm(fig, ax) -> float | None:
    renderer = fig.canvas.get_renderer()
    boxes = []
    for tick in ax.get_xticklabels():
        if tick.get_visible() and tick.get_text():
            box = tick.get_window_extent(renderer)
            if box.width > 0 and box.height > 0:
                boxes.append(box)
    label = ax.xaxis.label
    if label.get_visible() and label.get_text():
        box = label.get_window_extent(renderer)
        if box.width > 0 and box.height > 0:
            boxes.append(box)
    if not boxes:
        return None
    return min(bbox_to_mm(fig, box)["bottom"] for box in boxes)


def _set_axes_mm(fig, ax, box: Mapping[str, float], canvas: Mapping[str, Any]) -> None:
    canvas_w = float(canvas["width"])
    canvas_h = float(canvas["height"])
    ax.set_position([
        float(box["x"]) / canvas_w,
        (canvas_h - float(box["y"]) - float(box["h"])) / canvas_h,
        float(box["w"]) / canvas_w,
        float(box["h"]) / canvas_h,
    ])


def _set_panel_plot_area_mm(fig, ax, box: Mapping[str, float], canvas: Mapping[str, Any]) -> None:
    bounds = _primary_plot_bounds_in_parent(fig, ax)
    if bounds is None:
        _set_axes_mm(fig, ax, box, canvas)
        return
    rel_left, rel_bottom, rel_right, rel_top = bounds
    rel_w = rel_right - rel_left
    rel_h = rel_top - rel_bottom
    if rel_w <= 0 or rel_h <= 0:
        _set_axes_mm(fig, ax, box, canvas)
        return
    canvas_h = float(canvas["height"])
    desired_left = float(box["x"])
    desired_top = canvas_h - float(box["y"])
    desired_bottom = desired_top - float(box["h"])
    parent_w = float(box["w"]) / rel_w
    parent_h = float(box["h"]) / rel_h
    parent_left = desired_left - rel_left * parent_w
    parent_bottom = desired_bottom - rel_bottom * parent_h
    _set_axes_mm(
        fig,
        ax,
        {
            "x": parent_left,
            "y": canvas_h - parent_bottom - parent_h,
            "w": parent_w,
            "h": parent_h,
        },
        canvas,
    )


def _box(box: Mapping[str, Any]) -> dict[str, float]:
    return {
        "x": float(box.get("x", 0.0)),
        "y": float(box.get("y", 0.0)),
        "w": float(box.get("w", box.get("width", 0.0))),
        "h": float(box.get("h", box.get("height", 0.0))),
    }


def _residual(target: Mapping[str, float], measured: Mapping[str, float]) -> dict[str, float]:
    keys = ("left", "bottom", "right", "top", "width", "height")
    return {key: float(measured[key] - target[key]) for key in keys}
