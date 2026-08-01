from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl

from src.plotting.common.colors import (
    NATURE_COMPATIBLE_PALETTE,
    get_plot_cmap,
    get_plot_color,
)
from src.plotting.paper_fig.typography import (
    FIGURE_TEXT_SIZE_PT,
    FONT_FAMILY,
    PANEL_LABEL_SIZE_PT,
    VECTOR_TEXT_RCPARAMS,
)


INK = get_plot_color("ink")
WHITE = NATURE_COMPATIBLE_PALETTE["white"]
NAVY = get_plot_color("dynamic")
CYAN = get_plot_color("layer1")
TEAL = get_plot_color("sample_probe_overlap")
CORAL = get_plot_color("donor_trace")
ORANGE = NATURE_COMPATIBLE_PALETTE["comparison_salmon"]
PURPLE = NATURE_COMPATIBLE_PALETTE["fused_slate"]
GRAY_DARK = NATURE_COMPATIBLE_PALETTE["neutral_dark"]
GRAY = NATURE_COMPATIBLE_PALETTE["neutral_mid"]
GRAY_LIGHT = NATURE_COMPATIBLE_PALETTE["neutral_light"]
GRAY_PALE = NATURE_COMPATIBLE_PALETTE["neutral_pale"]
BLUE_PALE = NATURE_COMPATIBLE_PALETTE["primary_pale"]
BLUE_TINT = NATURE_COMPATIBLE_PALETTE["primary_tint"]
TEAL_MINT = NATURE_COMPATIBLE_PALETTE["mechanism_mint"]
TEAL_TINT = NATURE_COMPATIBLE_PALETTE["mechanism_tint"]
CORAL_TINT = NATURE_COMPATIBLE_PALETTE["comparison_tint"]
PURPLE_TINT = NATURE_COMPATIBLE_PALETTE["fused_tint"]

LAYER_COLORS = {
    "layer1": CYAN,
    "layer2": NAVY,
    "layer3": TEAL,
    "Layer 1": CYAN,
    "Layer 2": NAVY,
    "Layer 3": TEAL,
}
SEQUENCE_COLORS = {
    "old": CYAN,
    "middle": NAVY,
    "recent": TEAL,
    "old_mass": CYAN,
    "middle_mass": NAVY,
    "recent_mass": TEAL,
}
TRANSITION_COLORS = {
    "advance": NAVY,
    "recruit": CORAL,
    "loss": GRAY_DARK,
    "unchanged": GRAY_LIGHT,
}


@dataclass(frozen=True)
class MarkStyle:
    color: str
    marker: str
    linestyle: str
    markerfacecolor: str


CONDITION_STYLES = {
    "observed": MarkStyle(NAVY, "o", "-", NAVY),
    "dynamic": MarkStyle(NAVY, "o", "-", NAVY),
    "receiver": MarkStyle(NAVY, "o", "-", NAVY),
    "passive": MarkStyle(GRAY, "o", "--", WHITE),
    "static": MarkStyle(GRAY, "o", "--", WHITE),
    "baseline": MarkStyle(GRAY, "o", "--", WHITE),
    "donor": MarkStyle(CORAL, "D", "-", CORAL),
    "shuffle": MarkStyle(CORAL, "x", ":", WHITE),
    "random": MarkStyle(GRAY_DARK, "x", ":", WHITE),
    "overlap": MarkStyle(TEAL, "s", "-", TEAL),
    "reset": MarkStyle(GRAY_DARK, "s", "--", WHITE),
    "attenuate": MarkStyle(CORAL, "^", "-.", CORAL_TINT),
}


def apply_transition_style() -> None:
    mpl.rcParams.update(VECTOR_TEXT_RCPARAMS)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": FIGURE_TEXT_SIZE_PT,
            "axes.titlesize": FIGURE_TEXT_SIZE_PT,
            "axes.labelsize": FIGURE_TEXT_SIZE_PT,
            "xtick.labelsize": FIGURE_TEXT_SIZE_PT,
            "ytick.labelsize": FIGURE_TEXT_SIZE_PT,
            "legend.fontsize": FIGURE_TEXT_SIZE_PT,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.6,
            "patch.linewidth": 0.65,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": None,
        }
    )


def sequential_cmap():
    return get_plot_cmap("stsp_support")


def count_cmap():
    return get_plot_cmap("update_count")


def strength_cmap():
    return get_plot_cmap("peak_strength")


def signed_cmap():
    return get_plot_cmap("signed_effect")


__all__ = [
    "BLUE_PALE",
    "BLUE_TINT",
    "CORAL",
    "CORAL_TINT",
    "CYAN",
    "GRAY",
    "GRAY_DARK",
    "GRAY_LIGHT",
    "GRAY_PALE",
    "INK",
    "LAYER_COLORS",
    "NAVY",
    "ORANGE",
    "PANEL_LABEL_SIZE_PT",
    "PURPLE",
    "PURPLE_TINT",
    "SEQUENCE_COLORS",
    "TEAL",
    "TEAL_MINT",
    "TEAL_TINT",
    "TRANSITION_COLORS",
    "WHITE",
    "apply_transition_style",
    "count_cmap",
    "sequential_cmap",
    "signed_cmap",
    "strength_cmap",
]
