FIGURE_ID = "fig5"
EXPERIMENT_ID = "fig5_local_support_competition"
RUNNER_MODULE = "src.experiments.paper_figures.fig5.run_task"
COMPATIBILITY_MODULE = "src.experiments.paper_figures.fig5_local_support_competition_experiment"
SCOPE_TASKS = {"main": "main_scope", "supplement": "supplement_scope", "both": "both_scope"}
SUBEXPERIMENT_FLAGS = {
    "trial_sampling": ("--run-trial-sampling",),
    "preprobe_support": ("--run-preprobe-support",),
    "early_firing": ("--run-early-firing",),
    "local_events": ("--run-local-events",),
    "support_perturbation": ("--run-support-perturbation",),
    "supplement": ("--run-supplement",),
}
SUBEXPERIMENT_TASKS = {
    "trial_sampling": "trial_sampling",
    "preprobe_support": "preprobe_support",
    "early_firing": "early_firing",
    "local_events": "local_events",
    "support_perturbation": "support_perturbation",
    "supplement": "supplement",
}
ARCHIVED_SUBEXPERIMENTS = ()

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
