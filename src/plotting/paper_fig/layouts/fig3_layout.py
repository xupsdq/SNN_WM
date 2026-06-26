from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt

from src.plotting.paper_fig.true_edge_layout import install_true_edge_callbacks, seed_axes_box, semantic_layout_for_figure
from src.plotting.paper_fig.utils import add_axes_mm, mm_to_inch


def create_layout(spec: Mapping[str, Any], selected_panels: set[str] | None = None):
    """Create Fig.3 from explicit mm boxes, including the enlarged 3D landscape panel C."""
    canvas = spec["canvas_mm"]
    fig = plt.figure(figsize=(mm_to_inch(canvas["width"]), mm_to_inch(canvas["height"])), dpi=300)
    install_true_edge_callbacks(fig, spec)
    fig.paper_fig_semantic_layout = semantic_layout_for_figure("fig3")
    axes = {}
    for panel_id, panel in (spec.get("panels") or {}).items():
        if selected_panels is not None and panel_id not in selected_panels:
            continue
        axes_pos = seed_axes_box(panel_id, panel)
        if _needs_3d_axis(panel):
            ax = fig.add_axes(_bounds_fraction(axes_pos, canvas), projection="3d")
        else:
            ax = add_axes_mm(
                fig,
                axes_pos["x"],
                axes_pos["y"],
                axes_pos["w"],
                axes_pos["h"],
                canvas_h_mm=canvas["height"],
                canvas_w_mm=canvas["width"],
            )
        ax.paper_fig_axes_mm = dict(axes_pos)
        ax.paper_fig_panel_bounds = _bounds_fraction(panel.get("position_mm") or axes_pos, canvas)
        ax.paper_fig_plot_axes_bounds = _bounds_fraction(axes_pos, canvas)
        if panel.get("colorbar_mm"):
            cbar_pos = panel["colorbar_mm"]
            cax = add_axes_mm(
                fig,
                cbar_pos["x"],
                cbar_pos["y"],
                cbar_pos["w"],
                cbar_pos["h"],
                canvas_h_mm=canvas["height"],
                canvas_w_mm=canvas["width"],
            )
            ax.paper_fig_colorbar_ax = cax
            ax.paper_fig_colorbar_axes_mm = dict(cbar_pos)
        axes[panel_id] = ax
    return fig, axes


def _needs_3d_axis(panel: Mapping[str, Any]) -> bool:
    return str(panel.get("projection", "")).lower() == "3d" or str(panel.get("panel_type", "")).lower() in {"3d_surface", "surface_3d"}


def _bounds_fraction(pos: Mapping[str, Any], canvas: Mapping[str, Any]) -> list[float]:
    canvas_w = float(canvas["width"])
    canvas_h = float(canvas["height"])
    x0 = float(pos["x"]) / canvas_w
    y0 = (canvas_h - float(pos["y"]) - float(pos["h"])) / canvas_h
    x1 = (float(pos["x"]) + float(pos["w"])) / canvas_w
    y1 = (canvas_h - float(pos["y"])) / canvas_h
    return [x0, y0, x1, y1]
