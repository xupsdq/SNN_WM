FIGURE_ID = "fig3"
EXPERIMENT_ID = "fig3_multiitem_peak_landscape"
LEGACY_MODULE = "src.experiments.paper_figures.fig3_multiitem_peak_landscape_experiment"
SUBEXPERIMENT_FLAGS = {
    "state_bank": ("--run-state-bank",),
    "progressive_update": ("--run-progressive-update",),
    "peak_valley_landscape": ("--run-peak-valley-landscape",),
    "neutral_ping": ("--run-neutral-ping",),
    "weak_probe": ("--run-weak-probe",),
    "region_ping": ("--run-region-ping",),
    "region_ping_s0_control": ("--run-region-ping-s0-control",),
    "region_ping_amp_sweep": ("--run-region-ping-amp-sweep",),
    "peak_aligned_completion": ("--run-peak-aligned-completion",),
    "peak_cue_main": ("--run-peak-cue-main",),
    "population_morphology_supplement": ("--run-population-morphology-supplement",),
    "structural_weak_cue": ("--run-structural-weak-cue",),
    "structural_weak_cue_supplement": ("--run-structural-weak-cue-supplement",),
    "supplement": ("--run-supplement",),
}

MAIN_SUBEXPERIMENTS = (
    "state_bank",
    "progressive_update",
    "peak_valley_landscape",
    "neutral_ping",
    "weak_probe",
    "region_ping",
)
SUPPLEMENT_SUBEXPERIMENTS = (
    "state_bank",
    "progressive_update",
    "peak_valley_landscape",
    "neutral_ping",
    "structural_weak_cue",
    "population_morphology_supplement",
    "supplement",
)
