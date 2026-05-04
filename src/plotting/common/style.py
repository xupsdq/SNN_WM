from __future__ import annotations

import matplotlib as mpl

SKY_BLUE = "#56B4E9"
VERMILION = "#D55E00"
BLUISH_GREEN = "#009E73"
ORANGE = "#E69F00"
BLUE = "#0072B2"
PURPLE_MAGENTA = "#CC79A7"
LIGHT_GRAY = "#D9D9D9"
PALE_GRAY = "#F2F2F2"
PALE_YELLOW = "#FFF2B2"
PALE_BLUE = "#DDEEFF"
PALE_LAVENDER = "#E8DDF5"
NEUTRAL_GRAY = "#8A8A8A"

DYNAMIC_COLOR = BLUISH_GREEN
STATIC_COLOR = NEUTRAL_GRAY
SHUFFLE_COLOR = ORANGE
NOISE_COLOR = LIGHT_GRAY
SAMPLE_COLOR = ORANGE
PING_DECODE_COLOR = SKY_BLUE
SELECTIVITY_COLOR = VERMILION
NONOVERLAP_COLOR = PURPLE_MAGENTA
EARLIER_ITEM_COLOR = BLUE
BACKGROUND_SHADE_COLOR = PALE_GRAY
SAMPLE_WINDOW_COLOR = PALE_YELLOW
PROBE_WINDOW_COLOR = PALE_BLUE
PING_WINDOW_COLOR = PALE_LAVENDER

LINE_WIDTH = 1.8
MARKER_SIZE = 5.5
ERRORBAR_CAPSIZE = 3.5
ANNOTATION_FONT_SIZE = 11
PANEL_LABEL_FONT_SIZE = 15

DEFAULT_SUBPLOT_ADJUST = {
    "left": 0.08,
    "right": 0.97,
    "bottom": 0.11,
    "top": 0.94,
    "wspace": 0.32,
    "hspace": 0.38,
}

FIGURE2_SUBPLOT_ADJUST = {
    "left": 0.08,
    "right": 0.97,
    "bottom": 0.11,
    "top": 0.94,
    "wspace": 0.32,
    "hspace": 0.38,
}


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 13,
            "axes.labelweight": "regular",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "errorbar.capsize": ERRORBAR_CAPSIZE,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
