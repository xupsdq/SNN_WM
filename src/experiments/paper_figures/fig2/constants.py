from __future__ import annotations


FIGURE_ID = "fig2_pair_fused_stsp_state"
NUM_CLASSES = 10
STATE_CONDITIONS = ("S0", "S_A", "S_B", "S_AB")
STATE_VARIABLES = ("g", "u", "x", "ux_concat")
MIXTURE_MODELS = ("A_only", "B_only", "mean_AB", "sum_AB", "unconstrained_AB", "convex_AB")
SINGLE_NETWORK_MODE = "single_network"
RESIDUAL_TEMPLATE_DEFINITION = (
    "residual_true=y_AB-yhat_unconstrained; true_template=y_AB-0.5*(x_A+x_B); "
    "shuffled_template=y_AB-0.5*(x_A+x_B_j), j!=i"
)


__all__ = [
    "FIGURE_ID",
    "MIXTURE_MODELS",
    "NUM_CLASSES",
    "RESIDUAL_TEMPLATE_DEFINITION",
    "SINGLE_NETWORK_MODE",
    "STATE_CONDITIONS",
    "STATE_VARIABLES",
]
