from __future__ import annotations


FIGURE_ID = "fig3_multiitem_peak_landscape"
NUM_CLASSES = 10
PRIMARY_LAYER = "layer1"
PRIMARY_STATE_VARIABLE = "g"
STATE_VARIABLES = ("g", "u", "x")
CUE_CONDITIONS = ("peak", "valley", "random")
MEMORY_CONDITIONS = ("cue_only", "single_item_memory", "sequence_state")
SINGLE_NETWORK_MODE = "single_network"
FIG3_DESIGN_VERSION = "multiitem_peak_landscape_structured_readable_stsp"


__all__ = [
    "CUE_CONDITIONS",
    "FIG3_DESIGN_VERSION",
    "FIGURE_ID",
    "MEMORY_CONDITIONS",
    "NUM_CLASSES",
    "PRIMARY_LAYER",
    "PRIMARY_STATE_VARIABLE",
    "SINGLE_NETWORK_MODE",
    "STATE_VARIABLES",
]
