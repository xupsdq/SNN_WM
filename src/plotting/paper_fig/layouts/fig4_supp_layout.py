from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt

from src.plotting.paper_fig.utils import add_axes_mm, mm_to_inch


def create_layout(spec: Mapping[str, Any], selected_panels: set[str] | None = None):
    """Create the Fig.4 supplement canvas and panel axes from spec millimeter positions."""
    canvas = spec["canvas_mm"]
    fig = plt.figure(figsize=(mm_to_inch(canvas["width"]), mm_to_inch(canvas["height"])), dpi=300)
    axes = {}
    for group in spec.get("group_labels") or []:
        fig.text(
            float(group.get("x_mm", 12.0)) / float(canvas["width"]),
            1.0 - (float(group.get("y_mm", 8.0)) / float(canvas["height"])),
            str(group.get("label", "")),
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
        )
    for panel_id, panel in (spec.get("panels") or {}).items():
        if selected_panels is not None and panel_id not in selected_panels:
            continue
        pos = panel.get("axes_mm") or panel.get("position_mm") or {}
        axes[panel_id] = add_axes_mm(
            fig,
            pos["x"],
            pos["y"],
            pos["w"],
            pos["h"],
            canvas_h_mm=canvas["height"],
            canvas_w_mm=canvas["width"],
        )
    return fig, axes

