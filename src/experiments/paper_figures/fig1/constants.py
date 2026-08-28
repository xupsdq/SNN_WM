from __future__ import annotations


FIGURE_ID = "fig1_functional_stsp_substrate"
NUM_CLASSES = 10

MAIN_CONDITIONS = ("dynamic_intact", "ux_trial_shuffle", "static_frozen")
SUPP_CONDITIONS = (
    "dynamic_intact",
    "spike_state_shuffle",
    "membrane_state_shuffle",
    "ux_trial_shuffle",
    "static_frozen",
)
DMS_DELAY_SWEEP_CONDITIONS = ("dynamic_intact", "static_frozen")
SHUFFLE_CONDITIONS = (
    "spike_state_shuffle",
    "membrane_state_shuffle",
    "ux_trial_shuffle",
)
SUBSTRATE_BY_CONDITION = {
    "dynamic_intact": "dynamic",
    "ux_trial_shuffle": "ux",
    "spike_state_shuffle": "spike",
    "membrane_state_shuffle": "membrane",
    "static_frozen": "static",
}
CONDITION_TO_SUBSTRATE = {
    "spike_state_shuffle": "spike",
    "membrane_state_shuffle": "membrane",
    "ux_trial_shuffle": "ux",
}


__all__ = [
    "CONDITION_TO_SUBSTRATE",
    "DMS_DELAY_SWEEP_CONDITIONS",
    "FIGURE_ID",
    "MAIN_CONDITIONS",
    "NUM_CLASSES",
    "SHUFFLE_CONDITIONS",
    "SUBSTRATE_BY_CONDITION",
    "SUPP_CONDITIONS",
]
