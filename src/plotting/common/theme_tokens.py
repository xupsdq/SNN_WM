from __future__ import annotations

from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend

from src.plotting.common.io import (
    COLOR_DYNAMIC,
    COLOR_NOISE,
    COLOR_STATIC,
    PUBLICATION_LEGEND_FONT_SIZE,
    PUBLICATION_SINGLE_COLUMN_FIGSIZE,
    PUBLICATION_TWO_COLUMN_FIGSIZE,
)

COLOR_CONGRUENT = "#009E73"
COLOR_INCONGRUENT = "#D55E00"
COLOR_ACCENT_BLUE = "#4C78A8"
COLOR_ACCENT_BLUE_DARK = "#1F77B4"
COLOR_ACCENT_BLUE_SOFT = "#4C72B0"
COLOR_ACCENT_RED = "#E45756"
COLOR_ACCENT_ORANGE = "#E69F00"
COLOR_ACCENT_ORANGE_ALT = "#F58518"
COLOR_ACCENT_GREEN = "#54A24B"
COLOR_ACCENT_GREEN_SOFT = "#55A868"
COLOR_ACCENT_BROWN = "#9C755F"
COLOR_ACCENT_PURPLE = "#6A3D9A"
COLOR_ACCENT_PURPLE_LIGHT = "#B279A2"
COLOR_ACCENT_NAVY = "#2F4B7C"
COLOR_ACCENT_PINK = "#CC79A7"
COLOR_ACCENT_SKY = "#0072B2"
COLOR_ACCENT_STEEL = "#6C8EBF"
COLOR_ACCENT_TEAL = "#72B7B2"
COLOR_ACCENT_TAN = "#DD8452"
COLOR_ACCENT_TAUPE = "#9D755D"
COLOR_ACCENT_VIOLET = "#8172B3"
COLOR_ACCENT_ROSE_LIGHT = "#FF9DA6"
COLOR_ACCENT_SILVER = "#BAB0AC"
COLOR_NEUTRAL = "#333333"
COLOR_NEUTRAL_MID = "#555555"
COLOR_NEUTRAL_GRAY = "#9E9E9E"
COLOR_NEUTRAL_LIGHT = "#B0B0B0"
COLOR_GUIDE_LIGHT = "#DDDDDD"
COLOR_OFFWHITE = "#F5F5F5"
COLOR_BOX_FILL = "#F0F0F0"
COLOR_BOX_EDGE = "#555555"
COLOR_NATIVE = "#222222"
COLOR_DEST_OTHER = "#B9B9B9"

MODE_COLORS_DYNAMIC_STATIC = {
    "dynamic": COLOR_DYNAMIC,
    "static": COLOR_STATIC,
    "static_frozen": COLOR_STATIC,
}

BEHAVIOR_CONDITION_COLORS = {
    "congruent": COLOR_CONGRUENT,
    "baseline": COLOR_NOISE,
    "incongruent": COLOR_INCONGRUENT,
}

LAYER_ROUTE_COLORS = {
    "L1": COLOR_ACCENT_ORANGE,
    "L3": COLOR_CONGRUENT,
}

LAYER_STATE_COLORS = {
    "layer1": COLOR_ACCENT_SKY,
    "layer2": COLOR_ACCENT_ORANGE,
    "layer3": COLOR_CONGRUENT,
    "L1": COLOR_ACCENT_SKY,
    "L2": COLOR_ACCENT_ORANGE,
    "L3": COLOR_CONGRUENT,
}

DESTINATION_OUTCOME_COLORS = {
    "probe": COLOR_ACCENT_BLUE,
    "sample": COLOR_ACCENT_RED,
    "other": COLOR_DEST_OTHER,
}

GEOMETRY_CONDITION_COLORS = {
    "compact": COLOR_ACCENT_BLUE,
    "fragmented": COLOR_ACCENT_RED,
    "shuffled_random": COLOR_ACCENT_GREEN,
    "native": COLOR_NATIVE,
}

DISTRACTOR_MAIN_CONDITION_COLORS = {
    "full_dynamic": COLOR_ACCENT_BLUE,
    "sample_remove_SPonly": COLOR_ACCENT_RED,
    "distractor_remove_DPonly": COLOR_ACCENT_ORANGE_ALT,
    "sample_remove_SDP": COLOR_ACCENT_PURPLE_LIGHT,
    "distractor_remove_SDP": COLOR_ACCENT_TAUPE,
    "both_remove_SDP": COLOR_ACCENT_NAVY,
}

DISTRACTOR_CONTROL_CONDITION_COLORS = {
    "sample_remove_SPonly_control": COLOR_ACCENT_TEAL,
    "distractor_remove_DPonly_control": COLOR_ACCENT_GREEN,
    "sample_remove_SDP_control": COLOR_ACCENT_ROSE_LIGHT,
    "distractor_remove_SDP_control": COLOR_ACCENT_SILVER,
    "both_remove_SDP_control": COLOR_ACCENT_STEEL,
}

DISTRACTOR_MEDIATION_SWAP_COLORS = {
    "onset_only": COLOR_ACCENT_BLUE,
    "trace_only": COLOR_ACCENT_ORANGE_ALT,
    "onset_and_trace": COLOR_ACCENT_GREEN,
}

DISTRACTOR_REGION_CONDITION_COLORS = {
    "full_dynamic": COLOR_ACCENT_BLUE,
    "full_static": COLOR_NEUTRAL_GRAY,
    "only_SP": COLOR_ACCENT_RED,
    "only_DP": COLOR_ACCENT_ORANGE_ALT,
    "only_SDP": COLOR_ACCENT_NAVY,
    "ux_intact_dynamic": COLOR_ACCENT_BLUE_DARK,
    "ux_ablated_global": "#D62728",
    "union_reweight_intact": COLOR_ACCENT_BLUE_DARK,
    "union_ux_ablated": "#D62728",
    "complement_ux_ablated": COLOR_ACCENT_ORANGE_ALT,
}

OVERLAP_CONDITION_COLORS = {
    "full_dynamic": COLOR_ACCENT_BLUE,
    "full_static": COLOR_NEUTRAL_GRAY,
    "sample_remove_overlap_dynamic": COLOR_ACCENT_RED,
    "sample_remove_nonoverlap_control_dynamic": COLOR_ACCENT_TEAL,
    "sample_keep_overlap_only_dynamic": COLOR_ACCENT_GREEN,
    "sample_remove_all_foreground_dynamic": COLOR_ACCENT_ORANGE_ALT,
}

FREEZE_SAMPLE_TYPE_COLORS = {
    "diagnostic_overlap": COLOR_CONGRUENT,
    "baseline": COLOR_NOISE,
    "nondiagnostic_overlap": COLOR_INCONGRUENT,
}

FREEZE_CONDITION_COLORS = {
    "full_dynamic": COLOR_DYNAMIC,
    "freeze_L1": COLOR_ACCENT_SKY,
    "freeze_L1_L2": COLOR_ACCENT_ORANGE,
    "full_frozen": COLOR_STATIC,
    "freeze_L2_only": "#56B4E9",
    "freeze_L3_only": COLOR_ACCENT_PINK,
}

DELAY_SWEEP_COLORS = [
    COLOR_ACCENT_SKY,
    COLOR_ACCENT_ORANGE,
    COLOR_CONGRUENT,
    COLOR_INCONGRUENT,
    COLOR_ACCENT_PINK,
]

SILENT_MEMORY_MODE_COLORS = {
    "dynamic": "#d62728",
    "static_frozen": "#7f7f7f",
}

SUBSTRATE_COLORS = {
    "spike": COLOR_INCONGRUENT,
    "membrane": COLOR_ACCENT_SKY,
    "stsp": COLOR_CONGRUENT,
}

RETENTION_ACCURACY_COLORS = {
    "base": "#7f7f7f",
    "clean": COLOR_ACCENT_BLUE_DARK,
    "distracted": "#d62728",
}

RETENTION_BIAS_COLORS = {
    "dynamic": "#d62728",
    "static_frozen": "#7f7f7f",
    "sample": COLOR_ACCENT_BLUE_DARK,
    "change": "#2ca02c",
}

ERROR_DESTINATION_COLORS = {
    "original_sample": COLOR_ACCENT_BLUE_SOFT,
    "donor_sample": COLOR_ACCENT_GREEN_SOFT,
    "probe": "#C44E52",
    "silent": COLOR_ACCENT_VIOLET,
    "other": "#937860",
}

PART_SIMILARITY_BIN_COLORS = {
    "different-part": COLOR_ACCENT_RED,
    "middle-part": COLOR_NEUTRAL_GRAY,
    "same-part": COLOR_ACCENT_BLUE,
}

PAIR_HIGHLIGHT_COLORS = {
    "pair_i": COLOR_ACCENT_RED,
    "pair_j": COLOR_ACCENT_BLUE,
}

FIGSIZE_SINGLE_PANEL_COMPACT = PUBLICATION_SINGLE_COLUMN_FIGSIZE
FIGSIZE_TWO_PANEL = PUBLICATION_TWO_COLUMN_FIGSIZE
FIGSIZE_FOUR_PANEL = (12.0, 8.0)
FIGSIZE_THREE_PANEL = (15.5, 4.8)
FIGSIZE_THREE_PANEL_WIDE = (16.0, 5.0)
FIGSIZE_THREE_PANEL_HEATMAP = (15.8, 5.2)
FIGSIZE_TWO_PANEL_TALL = (8.2, 5.0)
FIGSIZE_REPRESENTATIVE_GRID = (16.5, 8.5)
FIGSIZE_SINGLE_PANEL_WIDE = (7.6, 4.8)
FIGSIZE_SINGLE_PANEL_HEATMAP = (6.4, 5.8)
FIGSIZE_SINGLE_PANEL_TALL = (8.0, 5.5)
FIGSIZE_SINGLE_PANEL_MEDIUM = (8.8, 4.6)
FIGSIZE_SINGLE_PANEL_MEDIUM_TALL = (8.8, 5.1)
FIGSIZE_TWO_PANEL_WIDE = (11.5, 4.8)
FIGSIZE_THREE_PANEL_COMPACT = (13.0, 4.6)
FIGSIZE_THREE_PANEL_MEDIUM = (14.0, 4.8)
FIGSIZE_THREE_PANEL_RELATION = (12.8, 5.0)
FIGSIZE_THREE_PANEL_SUMMARY = (15.0, 4.8)
FIGSIZE_HEATMAP_CASE_GRID = (17.2, 8.4)
FIGSIZE_TWO_BY_TWO = (12.0, 9.0)
FIGSIZE_TWO_BY_THREE = (15.0, 8.0)

CMAP_IMAGE_GRAY = "gray"
CMAP_ACTIVATION = "magma"
CMAP_OVERLAP = "viridis"
CMAP_MASK_OVERLAY = "Reds"
CMAP_DIVERGING = "coolwarm"
CMAP_SEQUENTIAL = "viridis"
CMAP_SEQUENTIAL_ALT = "plasma"
CMAP_SEQUENTIAL_CONTRAST = "cividis"

MARKER_CIRCLE = "o"
LINE_WIDTH_PRIMARY = 2.0
LINE_WIDTH_SECONDARY = 1.8
LINE_WIDTH_REFERENCE = 1.0
LINE_WIDTH_GUIDE = 0.8
GRID_ALPHA = 0.2
GRID_ALPHA_SOFT = 0.25
ALPHA_BAR = 0.85
ALPHA_BAR_SOFT = 0.28
ALPHA_FILL = 0.16
ALPHA_FILL_LIGHT = 0.08
ALPHA_SCATTER = 0.75
ALPHA_SCATTER_LIGHT = 0.55
ALPHA_MASK_OVERLAY = 0.65
ALPHA_GUIDE = 0.7
ALPHA_ANNOTATION_BOX = 0.85

__all__ = [
    "ALPHA_BAR",
    "ALPHA_BAR_SOFT",
    "ALPHA_FILL",
    "ALPHA_FILL_LIGHT",
    "ALPHA_MASK_OVERLAY",
    "ALPHA_SCATTER",
    "ALPHA_SCATTER_LIGHT",
    "BEHAVIOR_CONDITION_COLORS",
    "CMAP_ACTIVATION",
    "CMAP_DIVERGING",
    "CMAP_IMAGE_GRAY",
    "CMAP_MASK_OVERLAY",
    "CMAP_OVERLAP",
    "CMAP_SEQUENTIAL",
    "CMAP_SEQUENTIAL_ALT",
    "CMAP_SEQUENTIAL_CONTRAST",
    "COLOR_ACCENT_BLUE",
    "COLOR_ACCENT_BLUE_DARK",
    "COLOR_ACCENT_BLUE_SOFT",
    "COLOR_ACCENT_BROWN",
    "COLOR_ACCENT_GREEN",
    "COLOR_ACCENT_GREEN_SOFT",
    "COLOR_ACCENT_NAVY",
    "COLOR_ACCENT_ORANGE",
    "COLOR_ACCENT_ORANGE_ALT",
    "COLOR_ACCENT_PINK",
    "COLOR_ACCENT_PURPLE",
    "COLOR_ACCENT_PURPLE_LIGHT",
    "COLOR_ACCENT_RED",
    "COLOR_ACCENT_ROSE_LIGHT",
    "COLOR_ACCENT_SKY",
    "COLOR_ACCENT_STEEL",
    "COLOR_ACCENT_TAN",
    "COLOR_ACCENT_TAUPE",
    "COLOR_ACCENT_TEAL",
    "COLOR_ACCENT_VIOLET",
    "COLOR_ACCENT_SILVER",
    "COLOR_CONGRUENT",
    "COLOR_BOX_EDGE",
    "COLOR_BOX_FILL",
    "COLOR_DEST_OTHER",
    "COLOR_GUIDE_LIGHT",
    "COLOR_INCONGRUENT",
    "COLOR_NATIVE",
    "COLOR_NEUTRAL",
    "COLOR_NEUTRAL_GRAY",
    "COLOR_NEUTRAL_LIGHT",
    "COLOR_NEUTRAL_MID",
    "COLOR_OFFWHITE",
    "DELAY_SWEEP_COLORS",
    "DESTINATION_OUTCOME_COLORS",
    "DISTRACTOR_CONTROL_CONDITION_COLORS",
    "DISTRACTOR_MAIN_CONDITION_COLORS",
    "DISTRACTOR_MEDIATION_SWAP_COLORS",
    "DISTRACTOR_REGION_CONDITION_COLORS",
    "ERROR_DESTINATION_COLORS",
    "FIGSIZE_HEATMAP_CASE_GRID",
    "FIGSIZE_FOUR_PANEL",
    "FIGSIZE_REPRESENTATIVE_GRID",
    "FIGSIZE_SINGLE_PANEL_COMPACT",
    "FIGSIZE_SINGLE_PANEL_HEATMAP",
    "FIGSIZE_SINGLE_PANEL_MEDIUM",
    "FIGSIZE_SINGLE_PANEL_MEDIUM_TALL",
    "FIGSIZE_SINGLE_PANEL_TALL",
    "FIGSIZE_SINGLE_PANEL_WIDE",
    "FIGSIZE_THREE_PANEL",
    "FIGSIZE_THREE_PANEL_COMPACT",
    "FIGSIZE_THREE_PANEL_HEATMAP",
    "FIGSIZE_THREE_PANEL_MEDIUM",
    "FIGSIZE_THREE_PANEL_RELATION",
    "FIGSIZE_THREE_PANEL_SUMMARY",
    "FIGSIZE_THREE_PANEL_WIDE",
    "FIGSIZE_TWO_BY_THREE",
    "FIGSIZE_TWO_BY_TWO",
    "FIGSIZE_TWO_PANEL",
    "FIGSIZE_TWO_PANEL_WIDE",
    "FIGSIZE_TWO_PANEL_TALL",
    "FREEZE_CONDITION_COLORS",
    "FREEZE_SAMPLE_TYPE_COLORS",
    "GEOMETRY_CONDITION_COLORS",
    "GRID_ALPHA",
    "GRID_ALPHA_SOFT",
    "LAYER_STATE_COLORS",
    "LAYER_ROUTE_COLORS",
    "OVERLAP_CONDITION_COLORS",
    "LINE_WIDTH_GUIDE",
    "LINE_WIDTH_PRIMARY",
    "LINE_WIDTH_REFERENCE",
    "LINE_WIDTH_SECONDARY",
    "MARKER_CIRCLE",
    "MODE_COLORS_DYNAMIC_STATIC",
    "PAIR_HIGHLIGHT_COLORS",
    "PART_SIMILARITY_BIN_COLORS",
    "RETENTION_ACCURACY_COLORS",
    "RETENTION_BIAS_COLORS",
    "SILENT_MEMORY_MODE_COLORS",
    "SUBSTRATE_COLORS",
    "ALPHA_GUIDE",
    "ALPHA_ANNOTATION_BOX",
    "apply_standard_figure_legend",
    "apply_standard_legend",
    "case_grid_figsize",
    "horizontal_panel_figsize",
]


def horizontal_panel_figsize(
    n_panels: int,
    *,
    panel_width: float = 4.8,
    height: float = 4.8,
) -> tuple[float, float]:
    return (panel_width * max(1, int(n_panels)), height)


def case_grid_figsize(
    n_rows: int,
    *,
    width: float,
    row_height: float,
) -> tuple[float, float]:
    return (width, row_height * max(1, int(n_rows)))


def apply_standard_legend(
    ax: Axes,
    *,
    handles: Sequence | None = None,
    labels: Sequence[str] | None = None,
    loc: str | None = None,
    ncol: int = 1,
    bbox_to_anchor: tuple[float, float] | None = None,
    compact: bool = False,
    title: str | None = None,
    fontsize: float | None = None,
) -> Legend | None:
    legend_size = fontsize if fontsize is not None else (8 if compact else PUBLICATION_LEGEND_FONT_SIZE)
    kwargs = {
        "frameon": False,
        "fontsize": legend_size,
        "ncol": int(ncol),
        "title": title,
    }
    if loc is not None:
        kwargs["loc"] = loc
    if bbox_to_anchor is not None:
        kwargs["bbox_to_anchor"] = bbox_to_anchor
    if handles is None:
        legend = ax.legend(**kwargs)
    else:
        if labels is None:
            legend = ax.legend(handles=handles, **kwargs)
        else:
            legend = ax.legend(handles, labels, **kwargs)
    if legend is not None and title in (None, ""):
        legend.set_title("")
    return legend


def apply_standard_figure_legend(
    fig: Figure,
    handles,
    labels,
    *,
    loc: str = "upper center",
    ncol: int = 1,
    bbox_to_anchor: tuple[float, float] | None = None,
    fontsize: float | None = None,
    title: str | None = None,
    frameon: bool = False,
) -> Legend:
    legend_size = fontsize if fontsize is not None else PUBLICATION_LEGEND_FONT_SIZE
    kwargs = {
        "loc": loc,
        "ncol": int(ncol),
        "frameon": bool(frameon),
        "fontsize": legend_size,
        "title": title,
    }
    if bbox_to_anchor is not None:
        kwargs["bbox_to_anchor"] = bbox_to_anchor
    legend = fig.legend(handles, labels, **kwargs)
    if title in (None, ""):
        legend.set_title("")
    return legend
