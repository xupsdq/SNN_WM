from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from PIL import Image

from src.plotting.paper_fig.utils import add_axes_mm, mm_to_inch


def create_layout(spec: Mapping[str, Any], selected_panels: set[str] | None = None):
    """Create the Fig.4 supplement canvas and panel axes from spec millimeter positions."""
    canvas = spec["canvas_mm"]
    if (float(canvas.get("width", 0.0)), float(canvas.get("height", 0.0))) != (165.0, 110.0):
        raise ValueError("Supplementary Figure S4 requires the frozen 165 x 110 mm canvas.")
    fig = plt.figure(figsize=(mm_to_inch(canvas["width"]), mm_to_inch(canvas["height"])), dpi=300)
    original_savefig = fig.savefig

    def savefig_with_rgb_png(filename, *args, **kwargs):
        original_savefig(filename, *args, **kwargs)
        path = Path(filename)
        if path.suffix.lower() != ".png":
            return
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (300.0, 300.0))
            rgb = image.convert("RGB")
        target_size = (
            round(float(canvas["width"]) / 25.4 * float(dpi[0])),
            round(float(canvas["height"]) / 25.4 * float(dpi[1])),
        )
        if rgb.size != target_size:
            exact_canvas = Image.new("RGB", target_size, "white")
            exact_canvas.paste(rgb, (0, 0))
            rgb = exact_canvas
        rgb.save(path, format="PNG", dpi=dpi)

    fig.savefig = savefig_with_rgb_png
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
        if (
            float(pos["x"]) < 0.0
            or float(pos["y"]) < 0.0
            or float(pos["x"]) + float(pos["w"]) > float(canvas["width"])
            or float(pos["y"]) + float(pos["h"]) > float(canvas["height"])
        ):
            raise ValueError(f"Supplementary Figure S4 panel {panel_id} axes exceed the frozen canvas.")
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

