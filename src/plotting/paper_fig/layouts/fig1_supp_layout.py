from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.plotting.paper_fig.utils import add_axes_mm, mm_to_inch


def create_layout(spec: Mapping[str, Any], selected_panels: set[str] | None = None):
    """Create the Fig.1 supplement canvas and panel axes from spec millimeter positions."""
    canvas = spec["canvas_mm"]
    if (float(canvas["width"]), float(canvas["height"])) != (165.0, 110.0):
        raise ValueError(f"Supplementary Figure S1 requires a 165 x 110 mm source canvas, found {canvas}")
    if spec.get("group_labels"):
        raise ValueError("Supplementary Figure S1 must not reserve a title/group-label row")
    approved_font = _configure_frozen_typography()
    fig = plt.figure(figsize=(mm_to_inch(canvas["width"]), mm_to_inch(canvas["height"])), dpi=300)
    fig.patch.set_facecolor("white")
    _install_rgb_png_export(fig)
    fig.paper_fig_s1_font_family = approved_font
    fig.paper_fig_finalize_layout = _finalize_frozen_typography
    axes = {}
    for group in spec.get("group_labels") or []:
        fig.text(
            float(group.get("x_mm", 12)) / float(canvas["width"]),
            1.0 - (float(group.get("y_mm", 6)) / float(canvas["height"])),
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
        ax = add_axes_mm(
            fig,
            pos["x"],
            pos["y"],
            pos["w"],
            pos["h"],
            canvas_h_mm=canvas["height"],
            canvas_w_mm=canvas["width"],
        )
        axes[panel_id] = ax
    return fig, axes


def _configure_frozen_typography() -> str:
    approved_font = ""
    for candidate in ("Arial", "Helvetica"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=candidate), fallback_to_default=False)
        except ValueError:
            continue
        approved_font = candidate
        break
    if not approved_font:
        raise RuntimeError("Neither Arial nor Helvetica is installed; S1 export is disabled")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [approved_font],
            "mathtext.fontset": "custom",
            "mathtext.rm": approved_font,
            "mathtext.it": f"{approved_font}:italic",
            "mathtext.bf": f"{approved_font}:bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    from src.plotting.paper_fig import typography

    typography.PANEL_LABEL_SIZE_PT = 10.5
    typography.FIGURE_TEXT_SIZE_PT = 9.1
    typography.FONT_FAMILY = [approved_font]
    typography.VECTOR_TEXT_RCPARAMS["font.family"] = "sans-serif"
    typography.VECTOR_TEXT_RCPARAMS["font.sans-serif"] = [approved_font]
    typography.VECTOR_TEXT_RCPARAMS["svg.fonttype"] = "none"
    typography.VECTOR_TEXT_RCPARAMS["pdf.fonttype"] = 42
    return approved_font


def _finalize_frozen_typography(fig, axes, spec) -> None:
    _ = axes, spec
    approved_font = str(getattr(fig, "paper_fig_s1_font_family", ""))
    if approved_font not in {"Arial", "Helvetica"}:
        raise RuntimeError("S1 approved font identity was lost before export")
    for text in fig.texts:
        text.set_fontfamily([approved_font])


def _install_rgb_png_export(fig) -> None:
    original_savefig = fig.savefig

    def savefig_with_rgb_png(*args, **kwargs):
        result = original_savefig(*args, **kwargs)
        raw_path = args[0] if args else kwargs.get("fname")
        path = Path(raw_path) if raw_path is not None else None
        if path is not None and path.suffix.lower() == ".png":
            from PIL import Image

            dpi = float(kwargs.get("dpi", 300))
            with Image.open(path) as image:
                if image.mode == "RGBA":
                    rgb = Image.new("RGB", image.size, "white")
                    rgb.paste(image, mask=image.getchannel("A"))
                else:
                    rgb = image.convert("RGB")
                target_size = (
                    round(float(fig.get_figwidth()) * dpi),
                    round(float(fig.get_figheight()) * dpi),
                )
                if rgb.size != target_size:
                    rgb = rgb.resize(target_size, Image.Resampling.LANCZOS)
                rgb.save(path, format="PNG", dpi=(dpi, dpi))
        return result

    fig.savefig = savefig_with_rgb_png
