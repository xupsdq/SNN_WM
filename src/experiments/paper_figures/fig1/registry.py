FIGURE_ID = "fig1"
EXPERIMENT_ID = "fig1_functional_stsp_substrate"
RUNNER_MODULE = "src.experiments.paper_figures.fig1.run_task"
COMPATIBILITY_MODULE = "src.experiments.paper_figures.fig1_functional_stsp_substrate_experiment"
SCOPE_TASKS = {"main": "main_scope", "supplement": "supplement_scope", "both": "both_scope"}
SUBEXPERIMENT_FLAGS = {
    "baseline": ("--run-baseline",),
    "delay_decode": ("--run-delay-decode",),
    "dms_delay_sweep": ("--run-dms-delay-sweep",),
    "dms_shuffle": ("--run-dms-shuffle",),
    "firing_rate_control": ("--run-firing-rate-control",),
}
SUBEXPERIMENT_TASKS = {
    "baseline": "baseline",
    "delay_decode": "delay_decoder",
    "dms_delay_sweep": "dms_delay_sweep_readout",
    "dms_shuffle": "dms_shuffle_readout",
    "firing_rate_control": "firing_rate_control",
}
ARCHIVED_SUBEXPERIMENTS = ()

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
