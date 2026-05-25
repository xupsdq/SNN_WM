from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt

from src.plotting.paper_fig.utils import add_axes_mm, mm_to_inch


def create_layout(spec: Mapping[str, Any], selected_panels: set[str] | None = None):
    """Create the Fig.6 overlap-gated STSP recruitment canvas using top-left millimeter positions."""
    canvas = spec["canvas_mm"]
    fig = plt.figure(figsize=(mm_to_inch(canvas["width"]), mm_to_inch(canvas["height"])), dpi=300)
    axes = {}
    for panel_id, panel in (spec.get("panels") or {}).items():
        if selected_panels is not None and panel_id not in selected_panels:
            continue
        pos = panel.get("position_mm") or {}
        ax = add_axes_mm(
            fig,
            pos["x"],
            pos["y"],
            pos["w"],
            pos["h"],
            canvas_h_mm=canvas["height"],
            canvas_w_mm=canvas["width"],
        )
        ax.paper_fig_axes_mm = {"x": pos["x"], "y": pos["y"], "w": pos["w"], "h": pos["h"]}
        axes[panel_id] = ax
    return fig, axes
