FIGURE_ID = "fig4"
EXPERIMENT_ID = "fig4_overlap_reentry"
LEGACY_MODULE = "src.experiments.paper_figures.fig4_overlap_reentry_experiment"
SUBEXPERIMENT_FLAGS = {
    "pair_sampling": ("--run-pair-sampling",),
    "rollouts": ("--run-rollouts",),
    "similarity_entry": ("--run-similarity-entry",),
    "overlap_localization": ("--run-overlap-localization",),
    "overlap_accuracy_identification": ("--run-overlap-accuracy-identification",),
    "decision_spike_displacement": ("--run-decision-spike-displacement",),
    "decision_deflection": ("--run-decision-deflection",),
    "overlap_perturbation": ("--run-overlap-perturbation",),
    "supplement": ("--run-supplement",),
}

MAIN_SUBEXPERIMENTS = (
    "pair_sampling",
    "rollouts",
    "similarity_entry",
    "overlap_localization",
    "overlap_accuracy_identification",
    "decision_spike_displacement",
    "decision_deflection",
    "overlap_perturbation",
)
SUPPLEMENT_SUBEXPERIMENTS = (
    *MAIN_SUBEXPERIMENTS,
    "supplement",
)
