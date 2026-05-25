FIGURE_ID = "fig2"
EXPERIMENT_ID = "fig2_pair_fused_stsp_state"
LEGACY_MODULE = "src.experiments.paper_figures.fig2_pair_fused_stsp_state_experiment"
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
