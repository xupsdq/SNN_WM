from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FONT_FAMILY = "Arial"

COLOR_DYNAMIC = "#E69F00"
COLOR_STATIC = "#56B4E9"
COLOR_OVERLAP = "#009E73"
COLOR_PROBE_ONLY = "#CC79A7"
COLOR_SAMPLE_ONLY = "#4C566A"
COLOR_TEXT = "#1F2933"
COLOR_DARK_GRAY = "#4B5563"
COLOR_LIGHT_GRAY = "#D1D5DB"
COLOR_GRID = "#E5E7EB"

PANEL_LABEL_SIZE = 8
AXIS_TITLE_SIZE = 7
TICK_SIZE = 6
ANNOTATION_SIZE = 5.75

AXIS_LINEWIDTH = 0.95
DATA_LINEWIDTH = 1.7
REF_LINEWIDTH = 1.15
FRAME_LINEWIDTH = 0.75


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "DejaVu Sans", "Liberation Sans"],
            "font.size": TICK_SIZE,
            "axes.titlesize": AXIS_TITLE_SIZE,
            "axes.labelsize": AXIS_TITLE_SIZE,
            "axes.linewidth": AXIS_LINEWIDTH,
            "axes.edgecolor": COLOR_TEXT,
            "axes.labelcolor": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": TICK_SIZE,
            "legend.title_fontsize": TICK_SIZE,
            "text.color": COLOR_TEXT,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axes(ax: plt.Axes, *, show_grid_y: bool = False, show_grid_x: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(width=AXIS_LINEWIDTH, length=2.8, pad=2)
    if show_grid_y:
        ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.45 * REF_LINEWIDTH, alpha=0.45)
    if show_grid_x:
        ax.grid(True, axis="x", color=COLOR_GRID, linewidth=0.45 * REF_LINEWIDTH, alpha=0.45)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.04) -> None:
    # Panel letter tags are intentionally disabled for exported paper figures.
    return None


def add_reference_line(ax: plt.Axes, y: float = 0.0) -> None:
    ax.axhline(y, color=COLOR_DARK_GRAY, linewidth=REF_LINEWIDTH, linestyle="--", zorder=0)


def legend_outside(ax: plt.Axes, *, title: str | None = None, ncol: int = 1) -> None:
    ax.legend(
        frameon=False,
        title=title,
        loc="upper right",
        borderaxespad=0.2,
        ncol=ncol,
    )


def save_figure_outputs(fig: plt.Figure, output_dir: str | Path, stem: str) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    svg_path = out_dir / f"{stem}.svg"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    return {"pdf": str(pdf_path), "svg": str(svg_path), "png": str(png_path)}


__all__ = [
    "ANNOTATION_SIZE",
    "AXIS_LINEWIDTH",
    "AXIS_TITLE_SIZE",
    "COLOR_DARK_GRAY",
    "COLOR_DYNAMIC",
    "COLOR_GRID",
    "COLOR_LIGHT_GRAY",
    "COLOR_OVERLAP",
    "COLOR_PROBE_ONLY",
    "COLOR_SAMPLE_ONLY",
    "COLOR_STATIC",
    "COLOR_TEXT",
    "DATA_LINEWIDTH",
    "FRAME_LINEWIDTH",
    "PANEL_LABEL_SIZE",
    "REF_LINEWIDTH",
    "TICK_SIZE",
    "add_panel_label",
    "add_reference_line",
    "apply_paper_style",
    "legend_outside",
    "save_figure_outputs",
    "style_axes",
]
