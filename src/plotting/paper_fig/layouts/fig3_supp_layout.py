from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from PIL import Image

from src.plotting.paper_fig.utils import add_axes_mm, mm_to_inch


_S3_CANVAS_MM = {"width": 165.0, "height": 152.0}
_S3_AXES_MM = {
    "A": {"x": 18.0, "y": 13.0, "w": 60.0, "h": 31.5},
    "B": {"x": 97.0, "y": 13.0, "w": 60.0, "h": 31.5},
    "C": {"x": 18.0, "y": 61.666667, "w": 60.0, "h": 31.5},
    "D": {"x": 97.0, "y": 61.666667, "w": 60.0, "h": 31.5},
    "E": {"x": 18.0, "y": 110.333333, "w": 60.0, "h": 31.5},
    "F": {"x": 97.0, "y": 110.333333, "w": 60.0, "h": 31.5},
}


def create_layout(spec: Mapping[str, Any], selected_panels: set[str] | None = None):
    """Create the frozen 165 x 152 mm, three-row S3 layout."""
    if str(spec.get("figure_id", "")) != "supp_fig_s3":
        raise ValueError("fig3_supp_layout is restricted to supp_fig_s3")
    canvas = spec["canvas_mm"]
    for key, expected in _S3_CANVAS_MM.items():
        if abs(float(canvas.get(key, -1.0)) - expected) > 1e-6:
            raise ValueError(f"supp_fig_s3 canvas {key} must be {expected} mm")
    panel_specs = spec.get("panels") or {}
    if set(panel_specs) != set(_S3_AXES_MM):
        raise ValueError("supp_fig_s3 must contain exactly panels A-F")
    for panel_id, expected in _S3_AXES_MM.items():
        observed = panel_specs[panel_id].get("axes_mm") or {}
        for key, value in expected.items():
            if abs(float(observed.get(key, -1.0)) - value) > 1e-6:
                raise ValueError(f"supp_fig_s3{panel_id} axes {key} must be {value} mm")
    plt.rcParams.update(
        {
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )
    fig = plt.figure(
        figsize=(mm_to_inch(canvas["width"]), mm_to_inch(canvas["height"])),
        dpi=300,
        facecolor="white",
        constrained_layout=False,
    )
    _enforce_rgb_png_export(fig)
    for group in spec.get("group_labels") or []:
        fig.text(
            float(group.get("x_mm", 12)) / float(canvas["width"]),
            1.0 - (float(group.get("y_mm", 8)) / float(canvas["height"])),
            str(group.get("label", "")),
            ha="left",
            va="top",
            fontsize=float(group.get("fontsize", 8.0)),
            fontweight="bold",
        )
    axes = {}
    for panel_id, panel in panel_specs.items():
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


def _enforce_rgb_png_export(fig) -> None:
    """Composite S3 PNG exports onto white so the final mode is RGB, not RGBA."""
    original_savefig = fig.savefig

    def savefig_rgb_png(*args, **kwargs):
        result = original_savefig(*args, **kwargs)
        target = args[0] if args else kwargs.get("fname")
        if isinstance(target, (str, Path)) and Path(target).suffix.lower() == ".png":
            png_path = Path(target)
            with Image.open(png_path) as image:
                dpi = image.info.get("dpi", (300.0, 300.0))
                rgba = image.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, "white")
                rgb.paste(rgba, mask=rgba.getchannel("A"))
                if png_path.stem == "supp_fig_s3":
                    expected_size = (
                        round((_S3_CANVAS_MM["width"] / 25.4) * 300.0),
                        round((_S3_CANVAS_MM["height"] / 25.4) * 300.0),
                    )
                    if any(abs(observed - expected) > 1 for observed, expected in zip(rgb.size, expected_size)):
                        raise ValueError(
                            f"supp_fig_s3 PNG raster size {rgb.size} cannot be safely normalized to {expected_size}"
                        )
                    if rgb.size != expected_size:
                        normalized = Image.new("RGB", expected_size, "white")
                        normalized.paste(rgb, (0, 0))
                        rgb = normalized
                rgb.save(png_path, format="PNG", dpi=dpi)
        return result

    fig.savefig = savefig_rgb_png


def _bounds_fraction(pos: Mapping[str, Any], canvas: Mapping[str, Any]) -> list[float]:
    canvas_w = float(canvas["width"])
    canvas_h = float(canvas["height"])
    x0 = float(pos["x"]) / canvas_w
    y0 = (canvas_h - float(pos["y"]) - float(pos["h"])) / canvas_h
    x1 = (float(pos["x"]) + float(pos["w"])) / canvas_w
    y1 = (canvas_h - float(pos["y"])) / canvas_h
    return [x0, y0, x1, y1]
