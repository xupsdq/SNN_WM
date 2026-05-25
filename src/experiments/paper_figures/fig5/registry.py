FIGURE_ID = "fig5"
EXPERIMENT_ID = "fig5_local_support_competition"
LEGACY_MODULE = "src.experiments.paper_figures.fig5_local_support_competition_experiment"
SUBEXPERIMENT_FLAGS = {
    "trial_sampling": ("--run-trial-sampling",),
    "preprobe_support": ("--run-preprobe-support",),
    "early_firing": ("--run-early-firing",),
    "local_events": ("--run-local-events",),
    "support_perturbation": ("--run-support-perturbation",),
    "supplement": ("--run-supplement",),
}

MAIN_SUBEXPERIMENTS = (
    "trial_sampling",
    "preprobe_support",
    "early_firing",
    "local_events",
    "support_perturbation",
)
SUPPLEMENT_SUBEXPERIMENTS = (
    *MAIN_SUBEXPERIMENTS,
    "supplement",
)
