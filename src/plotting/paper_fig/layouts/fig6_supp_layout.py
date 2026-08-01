from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from PIL import Image

from src.plotting.paper_fig.utils import add_axes_mm, mm_to_inch


def create_layout(spec: Mapping[str, Any], selected_panels: set[str] | None = None):
    """Create the frozen 165 x 110 mm S6 two-row canvas."""
    canvas = spec["canvas_mm"]
    canvas_width = float(canvas["width"])
    canvas_height = float(canvas["height"])
    if (canvas_width, canvas_height) != (165.0, 110.0):
        raise ValueError(f"Supplementary Fig. S6 requires a 165 x 110 mm canvas, got {canvas_width} x {canvas_height} mm.")
    if list(spec.get("reading_order") or []) != ["A", "B", "C", "D", "E"]:
        raise ValueError("Supplementary Fig. S6 reading order must be A-E.")
    fig = plt.figure(figsize=(mm_to_inch(canvas["width"]), mm_to_inch(canvas["height"])), dpi=300)
    _install_master_png_guard(fig)
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
        x, y, width, height = (float(pos[key]) for key in ("x", "y", "w", "h"))
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > canvas_width or y + height > canvas_height:
            raise ValueError(f"Supplementary Fig. S6 panel {panel_id} axes fall outside the frozen canvas: {pos!r}")
        ax = add_axes_mm(
            fig,
            x,
            y,
            width,
            height,
            canvas_h_mm=canvas["height"],
            canvas_w_mm=canvas["width"],
        )
        ax.paper_fig_axes_mm = {"x": x, "y": y, "w": width, "h": height}
        axes[panel_id] = ax
    return fig, axes


def _install_master_png_guard(fig) -> None:
    """Enforce the frozen RGB/300-dpi/rounded-pixel master contract locally."""
    original_savefig = fig.savefig

    def guarded_savefig(filename, *args, **kwargs):
        path = Path(filename)
        if path.suffix.lower() != ".png":
            return original_savefig(filename, *args, **kwargs)

        requested_dpi = float(kwargs.get("dpi", 300.0))
        target_size = (
            round(float(fig.get_figwidth()) * requested_dpi),
            round(float(fig.get_figheight()) * requested_dpi),
        )
        render_dpi = max(
            target_size[0] / float(fig.get_figwidth()),
            target_size[1] / float(fig.get_figheight()),
        ) + 1e-4
        render_kwargs = dict(kwargs)
        render_kwargs["dpi"] = render_dpi
        result = original_savefig(filename, *args, **render_kwargs)

        with Image.open(path) as source:
            if source.size != target_size:
                raise ValueError(f"S6 PNG raster size mismatch before RGB conversion: {source.size} != {target_size}")
            if source.mode == "RGBA":
                rgb = Image.new("RGB", source.size, "white")
                rgb.paste(source, mask=source.getchannel("A"))
            else:
                rgb = source.convert("RGB")
            rgb.save(path, format="PNG", dpi=(requested_dpi, requested_dpi))

        with Image.open(path) as verified:
            if verified.mode != "RGB" or verified.size != target_size:
                raise ValueError(f"S6 PNG master guard failed: mode={verified.mode}, size={verified.size}")
        return result

    fig.savefig = guarded_savefig
    fig.paper_fig_rgb_png_guard = True
