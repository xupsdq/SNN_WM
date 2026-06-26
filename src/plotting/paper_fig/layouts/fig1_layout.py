from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from src.plotting.paper_fig.true_edge_layout import install_true_edge_callbacks, seed_axes_box, semantic_layout_for_figure
from src.plotting.paper_fig.utils import add_axes_mm, mm_to_inch
from src.plotting.paper_fig.utils import write_json


def create_layout(spec: Mapping[str, Any], selected_panels: set[str] | None = None):
    """Create the Fig.1 canvas and panel axes using spec millimeter positions."""
    canvas = spec["canvas_mm"]
    fig = plt.figure(figsize=(mm_to_inch(canvas["width"]), mm_to_inch(canvas["height"])), dpi=300)
    install_true_edge_callbacks(fig, spec)
    fig.paper_fig_semantic_layout = semantic_layout_for_figure("fig1")
    axes = {}
    for panel_id, panel in (spec.get("panels") or {}).items():
        if selected_panels is not None and panel_id not in selected_panels:
            continue
        pos = seed_axes_box(panel_id, panel)
        ax = add_axes_mm(
            fig,
            pos["x"],
            pos["y"],
            pos["w"],
            pos["h"],
            canvas_h_mm=canvas["height"],
            canvas_w_mm=canvas["width"],
        )
        axes[panel_id] = ax
    return fig, axes


def _initial_axes_from_full_bbox(panel_id: str, pos: dict[str, Any]) -> dict[str, Any]:
    """Seed axes inside Fig.1 full-panel boxes before the measured pass."""
    if panel_id == "A":
        return pos
    # ponytail: seed only; _finalize_true_edge_layout measures real text edges.
    left = 11.0
    top = 6.5 if panel_id == "E" else 4.8
    right = 2.5
    bottom = 11.0
    return {
        "x": float(pos["x"]) + left,
        "y": float(pos["y"]) + top,
        "w": float(pos["w"]) - left - right,
        "h": float(pos["h"]) - top - bottom,
    }


def _finalize_true_edge_layout(fig, axes: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    """Fit axes so measured artist bounds land inside the target full-panel boxes."""
    panels = spec.get("panels") or {}
    canvas = spec.get("canvas_mm") or {}
    if not canvas:
        return
    fig.canvas.draw()
    for panel_id in ("B", "C", "D", "E"):
        ax = axes.get(panel_id)
        panel = panels.get(panel_id) or {}
        target = panel.get("position_mm") or {}
        if ax is None or not target:
            continue
        _fit_ylabel_gap(fig, ax)
        for _ in range(4):
            fig.canvas.draw()
            measured = _measure_panel(fig, ax, include_label=False)
            if measured is None:
                break
            axes_mm = _axes_mm(fig, ax, canvas)
            left = axes_mm["x"] - measured["x"]
            top = axes_mm["y"] - measured["y"]
            right = measured["right"] - axes_mm["right"]
            bottom = measured["bottom"] - axes_mm["bottom"]
            next_axes = {
                "x": float(target["x"]) + left,
                "y": float(target["y"]) + top,
                "w": max(12.0, float(target["w"]) - left - right),
                "h": max(12.0, float(target["h"]) - top - bottom),
            }
            _set_axes_mm(fig, ax, next_axes, canvas)
        _fit_ylabel_gap(fig, ax)
    fig.canvas.draw()


def _write_true_edge_report(fig, axes: Mapping[str, Any], spec: Mapping[str, Any], output_dir: str | Path) -> None:
    """Write Fig.1 v3 true-edge measurements next to the generated figure."""
    panels = spec.get("panels") or {}
    canvas = spec.get("canvas_mm") or {}
    fig.canvas.draw()
    data: dict[str, Any] = {
        "figure_id": spec.get("figure_id"),
        "canvas_mm": canvas,
        "method": "measured full-panel bbox includes axes, tick labels, axis labels, annotations, and legends; panel labels excluded",
        "target_y_title_tick_gap_mm": 0.5,
        "panels": {},
    }
    for panel_id in ("B", "C", "D", "E"):
        ax = axes.get(panel_id)
        panel = panels.get(panel_id) or {}
        if ax is None:
            continue
        measured = _measure_panel(fig, ax, include_label=False)
        axes_box = _axes_mm(fig, ax, canvas)
        data["panels"][panel_id] = {
            "target_full_bbox_mm": _box_with_edges(panel.get("position_mm") or {}),
            "measured_full_bbox_mm": measured,
            "axes_bbox_mm": axes_box,
            "y_title_tick_gap_mm": _ylabel_tick_gap_mm(fig, ax),
        }
    write_json(data, Path(output_dir) / "true_edge_measurements.json")


def _fit_ylabel_gap(fig, ax, target_mm: float = 0.5) -> None:
    """Move the y label near tick numbers without overlap."""
    for _ in range(5):
        fig.canvas.draw()
        gap = _ylabel_tick_gap_mm(fig, ax)
        if gap is None or abs(gap - target_mm) <= 0.08:
            return
        delta_pt = (gap - target_mm) / 25.4 * 72.0
        ax.yaxis.labelpad = max(-20.0, min(8.0, float(ax.yaxis.labelpad) - delta_pt))


def _ylabel_tick_gap_mm(fig, ax) -> float | None:
    renderer = fig.canvas.get_renderer()
    y_label = ax.yaxis.label
    if not y_label.get_visible() or not y_label.get_text():
        return None
    try:
        label_box = y_label.get_window_extent(renderer)
    except Exception:
        return None
    tick_boxes = []
    for label in ax.get_yticklabels():
        if not label.get_visible() or not label.get_text():
            continue
        try:
            box = label.get_window_extent(renderer)
        except Exception:
            continue
        if box.width > 0 and box.height > 0:
            tick_boxes.append(box)
    if not tick_boxes or label_box.width <= 0:
        return None
    tick_left = min(box.x0 for box in tick_boxes)
    return (tick_left - label_box.x1) / fig.dpi * 25.4


def _measure_panel(fig, ax, *, include_label: bool) -> dict[str, float] | None:
    renderer = fig.canvas.get_renderer()
    boxes = [ax.bbox]
    if ax.axison:
        artists = list(ax.get_xticklabels()) + list(ax.get_yticklabels()) + [ax.xaxis.label, ax.yaxis.label]
        artists.extend(ax.texts)
        legend = ax.get_legend()
        if legend is not None:
            artists.append(legend)
        for artist in artists:
            if not artist.get_visible():
                continue
            if not include_label and getattr(artist, "paper_fig_panel_label_gap_mm", None) is not None:
                continue
            try:
                box = artist.get_window_extent(renderer)
            except Exception:
                continue
            if box.width > 0 and box.height > 0:
                boxes.append(box)
    return _bbox_to_mm(fig, Bbox.union(boxes))


def _axes_mm(fig, ax, canvas: Mapping[str, Any]) -> dict[str, float]:
    return _bbox_to_mm(fig, ax.bbox)


def _bbox_to_mm(fig, bbox) -> dict[str, float]:
    width_mm = fig.get_figwidth() * 25.4
    height_mm = fig.get_figheight() * 25.4
    x = bbox.x0 / fig.bbox.width * width_mm
    right = bbox.x1 / fig.bbox.width * width_mm
    y = (fig.bbox.height - bbox.y1) / fig.bbox.height * height_mm
    bottom = (fig.bbox.height - bbox.y0) / fig.bbox.height * height_mm
    return {"x": x, "y": y, "w": right - x, "h": bottom - y, "right": right, "bottom": bottom}


def _set_axes_mm(fig, ax, box: Mapping[str, float], canvas: Mapping[str, Any]) -> None:
    canvas_w = float(canvas["width"])
    canvas_h = float(canvas["height"])
    left = float(box["x"]) / canvas_w
    bottom = (canvas_h - float(box["y"]) - float(box["h"])) / canvas_h
    ax.set_position([left, bottom, float(box["w"]) / canvas_w, float(box["h"]) / canvas_h])


def _box_with_edges(box: Mapping[str, Any]) -> dict[str, float]:
    x = float(box.get("x", 0.0))
    y = float(box.get("y", 0.0))
    w = float(box.get("w", 0.0))
    h = float(box.get("h", 0.0))
    return {"x": x, "y": y, "w": w, "h": h, "right": x + w, "bottom": y + h}

