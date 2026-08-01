from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from PIL import Image

from src.plotting.paper_fig.utils import add_axes_mm, mm_to_inch


EXPECTED_S5_AXES_MM = {
    "A": {"x": 18.0, "y": 14.0, "w": 60.0, "h": 34.0},
    "B": {"x": 97.0, "y": 14.0, "w": 60.0, "h": 34.0},
    "C": {"x": 15.0, "y": 64.0, "w": 38.5, "h": 34.0},
    "D": {"x": 67.333333, "y": 66.0, "w": 38.5, "h": 34.0},
    "E": {"x": 119.666667, "y": 66.0, "w": 38.5, "h": 34.0},
}


def create_layout(spec: Mapping[str, Any], selected_panels: set[str] | None = None):
    """Create the frozen 165 x 110 mm S5 two-row (2+3) canvas."""
    canvas = spec["canvas_mm"]
    if abs(float(canvas["width"]) - 165.0) > 1e-9 or abs(float(canvas["height"]) - 110.0) > 1e-9:
        raise ValueError(f"Supplementary Fig. S5 requires a 165 x 110 mm source canvas; received {canvas!r}.")
    panels = spec.get("panels") or {}
    if tuple(panels) != ("A", "B", "C", "D", "E"):
        raise ValueError(f"Supplementary Fig. S5 requires exact A-E reading order; received {tuple(panels)!r}.")
    for panel_id, expected in EXPECTED_S5_AXES_MM.items():
        actual = panels[panel_id].get("axes_mm") or {}
        if any(abs(float(actual.get(key, float("nan"))) - value) > 1e-6 for key, value in expected.items()):
            raise ValueError(f"Supplementary Fig. S5{panel_id} axes geometry mismatch: {actual!r}.")
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
    for panel_id, panel in panels.items():
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
    _attach_rgb_png_export_guard(fig, enforce_full_canvas_pixels=selected_panels is None)
    return fig, axes


def _attach_rgb_png_export_guard(fig, *, enforce_full_canvas_pixels: bool) -> None:
    """Make every S5 PNG RGB and the full master exactly 1949 x 1299 pixels."""
    original_savefig = fig.savefig

    def savefig_rgb(*args, **kwargs):
        result = original_savefig(*args, **kwargs)
        target_arg = args[0] if args else kwargs.get("fname")
        if not isinstance(target_arg, (str, Path)):
            return result
        target_path = Path(target_arg)
        if target_path.suffix.lower() != ".png" or not target_path.is_file():
            return result
        with Image.open(target_path) as image:
            rgb = image.convert("RGB")
            if enforce_full_canvas_pixels:
                expected_pixels = (round(165.0 / 25.4 * 300.0), round(110.0 / 25.4 * 300.0))
                if rgb.size != expected_pixels:
                    if rgb.width > expected_pixels[0] or rgb.height > expected_pixels[1]:
                        raise RuntimeError(
                            f"S5 full PNG exceeds frozen pixel gate: expected {expected_pixels}, received {rgb.size}."
                        )
                    canvas = Image.new("RGB", expected_pixels, "white")
                    canvas.paste(rgb, (0, 0))
                    rgb = canvas
            rgb.save(target_path, format="PNG", dpi=(300.0, 300.0))
        return result

    fig.savefig = savefig_rgb
