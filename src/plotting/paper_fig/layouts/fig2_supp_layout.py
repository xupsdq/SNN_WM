from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

from src.plotting.paper_fig.utils import add_axes_mm, mm_to_inch


def create_layout(spec: Mapping[str, Any], selected_panels: set[str] | None = None):
    """Create the fixed 165 x 110 mm compact S2 2+1 canvas."""
    canvas = spec["canvas_mm"]
    canvas_identity = (float(canvas["width"]), float(canvas["height"]))
    if canvas_identity != (165.0, 110.0):
        raise RuntimeError(f"Supplementary Fig. S2 canvas must be exactly 165 x 110 mm, got {canvas_identity}.")
    font_family = _require_s2_font_family()
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 9.1,
            "axes.labelsize": 9.1,
            "xtick.labelsize": 9.1,
            "ytick.labelsize": 9.1,
            "legend.fontsize": 9.1,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(mm_to_inch(canvas["width"]), mm_to_inch(canvas["height"])), dpi=300)
    fig.set_facecolor("white")
    fig.paper_fig_canvas_mm = canvas_identity
    fig.paper_fig_required_font_family = font_family
    _install_s2_master_export_guard(fig)
    axes = {}
    for group in spec.get("group_labels") or []:
        fig.text(
            float(group.get("x_mm", 12)) / float(canvas["width"]),
            1.0 - (float(group.get("y_mm", 8)) / float(canvas["height"])),
            str(group.get("label", "")),
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
        )
    for panel_id, panel in (spec.get("panels") or {}).items():
        if selected_panels is not None and panel_id not in selected_panels:
            continue
        pos = panel.get("axes_mm") or panel.get("position_mm") or {}
        expected = {
            "A": {"x": 18.0, "y": 14.0, "w": 60.0, "h": 34.0},
            "B": {"x": 97.0, "y": 14.0, "w": 60.0, "h": 34.0},
            "C": {"x": 18.0, "y": 66.0, "w": 139.0, "h": 34.0},
        }.get(str(panel_id))
        if expected is None or any(float(pos.get(key, float("nan"))) != value for key, value in expected.items()):
            raise RuntimeError(f"Supplementary Fig. S2 panel {panel_id} axes do not match the frozen compact 2+1 geometry.")
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


def _require_s2_font_family() -> str:
    for family in ("Arial", "Helvetica"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        return family
    raise RuntimeError("Supplementary Fig. S2 requires Arial or Helvetica; silent font substitution is forbidden.")


def _install_s2_master_export_guard(fig) -> None:
    """Enforce the frozen rounded-pixel RGB contract for the full PNG master."""
    original_savefig = fig.savefig

    def guarded_savefig(fname, *args, **kwargs):
        path = Path(fname) if isinstance(fname, (str, Path)) else None
        is_full_s2_png = path is not None and path.suffix.lower() == ".png" and path.stem == "supp_fig_s2"
        original_size = fig.get_size_inches().copy()
        try:
            if is_full_s2_png:
                # 165 x 110 mm at 300 dpi rounds to 1949 x 1299 pixels.  Use
                # those rounded raster dimensions while retaining the exact
                # physical source size for the vector siblings.
                fig.set_size_inches(1949.0 / 300.0, 1299.0 / 300.0, forward=False)
                kwargs["dpi"] = 300
                kwargs["transparent"] = False
                kwargs["facecolor"] = "white"
            result = original_savefig(fname, *args, **kwargs)
        finally:
            fig.set_size_inches(original_size, forward=False)
        if is_full_s2_png:
            with Image.open(path) as source:
                if source.size != (1949, 1299):
                    raise RuntimeError(f"Supplementary Fig. S2 PNG dimensions must be 1949 x 1299, got {source.size}.")
                if source.mode == "RGB":
                    rgb = source.copy()
                else:
                    rgba = source.convert("RGBA")
                    rgb = Image.new("RGB", rgba.size, "white")
                    rgb.paste(rgba, mask=rgba.getchannel("A"))
            rgb.save(path, format="PNG", dpi=(300, 300))
        return result

    fig.savefig = guarded_savefig
