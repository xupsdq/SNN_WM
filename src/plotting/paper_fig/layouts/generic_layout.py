from __future__ import annotations

import math
from typing import Any, Mapping

import matplotlib.pyplot as plt

from src.plotting.paper_fig.utils import add_axes_mm, mm_to_inch


def create_layout(spec: Mapping[str, Any], selected_panels: set[str] | None = None):
    """Create a simple mm-based placeholder grid for scaffolded figures."""
    canvas = spec.get("canvas_mm") or {"width": 165, "height": 115}
    fig = plt.figure(figsize=(mm_to_inch(canvas["width"]), mm_to_inch(canvas["height"])), dpi=300)
    panel_ids = list((spec.get("panels") or {}).keys())
    if selected_panels is not None:
        panel_ids = [panel_id for panel_id in panel_ids if panel_id in selected_panels]
    if not panel_ids:
        return fig, {}
    n_cols = min(3, max(1, len(panel_ids)))
    n_rows = int(math.ceil(len(panel_ids) / n_cols))
    gutter = 5.0
    panel_w = (float(canvas["width"]) - gutter * (n_cols - 1)) / n_cols
    panel_h = (float(canvas["height"]) - gutter * (n_rows - 1)) / n_rows
    axes = {}
    for idx, panel_id in enumerate(panel_ids):
        row = idx // n_cols
        col = idx % n_cols
        axes[panel_id] = add_axes_mm(
            fig,
            col * (panel_w + gutter),
            row * (panel_h + gutter),
            panel_w,
            panel_h,
            canvas_h_mm=canvas["height"],
            canvas_w_mm=canvas["width"],
        )
    return fig, axes

