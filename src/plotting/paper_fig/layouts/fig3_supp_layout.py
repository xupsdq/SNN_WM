from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt

from src.plotting.paper_fig.utils import add_axes_mm, mm_to_inch


def create_layout(spec: Mapping[str, Any], selected_panels: set[str] | None = None):
    """Create the Fig.3 supplement canvas and panel axes from spec mm positions."""
    canvas = spec["canvas_mm"]
    fig = plt.figure(figsize=(mm_to_inch(canvas["width"]), mm_to_inch(canvas["height"])), dpi=300)
    for group in spec.get("group_labels") or []:
        fig.text(
            float(group.get("x_mm", 12)) / float(canvas["width"]),
            1.0 - (float(group.get("y_mm", 8)) / float(canvas["height"])),
            str(group.get("label", "")),
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
        )
    axes = {}
    for panel_id, panel in (spec.get("panels") or {}).items():
        if selected_panels is not None and panel_id not in selected_panels:
            continue
        pos = panel.get("axes_mm") or panel.get("position_mm") or {}
        ax = add_axes_mm(
            fig,
            pos["x"],
            pos["y"],
            pos["w"],
            pos["h"],
            canvas_h_mm=canvas["height"],
            canvas_w_mm=canvas["width"],
        )
        ax.paper_fig_axes_mm = dict(pos)
        ax.paper_fig_panel_bounds = _bounds_fraction(panel.get("position_mm") or pos, canvas)
        ax.paper_fig_plot_axes_bounds = _bounds_fraction(pos, canvas)
        axes[panel_id] = ax
    return fig, axes


def _bounds_fraction(pos: Mapping[str, Any], canvas: Mapping[str, Any]) -> list[float]:
    canvas_w = float(canvas["width"])
    canvas_h = float(canvas["height"])
    x0 = float(pos["x"]) / canvas_w
    y0 = (canvas_h - float(pos["y"]) - float(pos["h"])) / canvas_h
    x1 = (float(pos["x"]) + float(pos["w"])) / canvas_w
    y1 = (canvas_h - float(pos["y"])) / canvas_h
    return [x0, y0, x1, y1]
