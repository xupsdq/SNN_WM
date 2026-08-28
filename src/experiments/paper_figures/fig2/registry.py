FIGURE_ID = "fig2"
EXPERIMENT_ID = "fig2_pair_fused_stsp_state"
RUNNER_MODULE = "src.experiments.paper_figures.fig2.run_task"
COMPATIBILITY_MODULE = "src.experiments.paper_figures.fig2_pair_fused_stsp_state_experiment"
SCOPE_TASKS = {"main": "main_scope", "supplement": "supplement_scope", "both": "both_scope"}
SUBEXPERIMENT_FLAGS = {
    "state_bank": ("--run-state-bank",),
    "morphology": ("--run-morphology",),
    "linear_mixture": ("--run-linear-mixture",),
    "neutral_ping": ("--run-neutral-ping",),
    "partial_cue": ("--run-partial-cue",),
    "supplement": ("--run-supplement",),
    "ping_sweep": ("--run-ping-sweep",),
    "completion_delay_sweep": ("--run-completion-delay-sweep",),
}
SUBEXPERIMENT_TASKS = {
    "state_bank": "state_bank",
    "morphology": "morphology",
    "linear_mixture": "linear_mixture",
    "neutral_ping": "neutral_ping",
    "partial_cue": "partial_cue",
    "supplement": "supplement",
    "ping_sweep": "ping_sweep",
    "completion_delay_sweep": "completion_delay_sweep",
}
ARCHIVED_SUBEXPERIMENTS = ()

MAIN_SUBEXPERIMENTS = (
    "state_bank",
    "morphology",
    "linear_mixture",
    "neutral_ping",
    "partial_cue",
)
SUPPLEMENT_SUBEXPERIMENTS = (
    *MAIN_SUBEXPERIMENTS,
    "supplement",
)
