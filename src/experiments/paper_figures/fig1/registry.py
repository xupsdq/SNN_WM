FIGURE_ID = "fig1"
EXPERIMENT_ID = "fig1_functional_stsp_substrate"
LEGACY_MODULE = "src.experiments.paper_figures.fig1_functional_stsp_substrate_experiment"
SUBEXPERIMENT_FLAGS = {
    "baseline": ("--run-baseline",),
    "delay_decode": ("--run-delay-decode",),
    "dms_delay_sweep": ("--run-dms-delay-sweep",),
    "dms_shuffle": ("--run-dms-shuffle",),
    "firing_rate_control": ("--run-firing-rate-control",),
}

MAIN_SUBEXPERIMENTS = (
    "baseline",
    "delay_decode",
    "dms_shuffle",
    "firing_rate_control",
)
SUPPLEMENT_SUBEXPERIMENTS = (
    "baseline",
    "delay_decode",
    "dms_delay_sweep",
    "dms_shuffle",
    "firing_rate_control",
)
BOTH_SCOPE_SUBEXPERIMENTS = SUPPLEMENT_SUBEXPERIMENTS
BOTH_SCOPE_FLAGS = tuple(
    flag
    for subexperiment in BOTH_SCOPE_SUBEXPERIMENTS
    for flag in SUBEXPERIMENT_FLAGS[subexperiment]
)
