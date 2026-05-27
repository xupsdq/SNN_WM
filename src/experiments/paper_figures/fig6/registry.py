FIGURE_ID = "fig6"
EXPERIMENT_ID = "fig6_peak_amplified_reentry"
LEGACY_MODULE = "src.experiments.paper_figures.fig6_peak_amplified_reentry_experiment"
SUBEXPERIMENT_FLAGS = {
    "sequence_bank": ("--run-sequence-bank",),
    "peak_source_attribution": ("--run-peak-source-attribution",),
    "peak_update_history": ("--run-peak-update-history",),
    "peak_input_overlap_origin": ("--run-peak-input-overlap-origin",),
    "real_reentry_rollout": ("--run-real-reentry-rollout",),
    "real_downstream_metrics": ("--run-real-downstream-metrics",),
    "peak_enrichment": ("--run-peak-enrichment",),
    "update_recency_model": ("--run-update-recency-model",),
    "peak_weighted_overlap": ("--run-peak-weighted-overlap",),
    "reentry_prediction": ("--run-reentry-prediction",),
    "downstream_prediction": ("--run-downstream-prediction",),
    "peak_perturbation": ("--run-peak-perturbation", "--force-main-outputs"),
    "field_ping_readout": ("--run-field-ping-readout",),
    "global_ping_score_spike_prediction": ("--run-global-ping-score-spike-prediction",),
    "ping_score_spike_prediction": ("--run-ping-score-spike-prediction",),
    "real_probe_score_spike_deflection": ("--run-real-probe-score-spike-deflection",),
    "overlap_gated_stsp_recruitment": ("--run-overlap-gated-stsp-recruitment",),
    "high_stsp_overlap_ablation": ("--run-high-stsp-overlap-ablation",),
    "score_basin_sparsification": ("--run-score-basin-sparsification",),
    "fig6_downstream_exploratory": ("--run-fig6-downstream-exploratory",),
    "supplement": ("--run-supplement",),
    "score_shuffle_null": ("--run-score-shuffle-null",),
    "overlap_threshold_sensitivity": ("--run-overlap-threshold-sensitivity",),
}

MAIN_SUBEXPERIMENTS = (
    "sequence_bank",
    "field_ping_readout",
    "global_ping_score_spike_prediction",
    "real_probe_score_spike_deflection",
    "overlap_gated_stsp_recruitment",
    "high_stsp_overlap_ablation",
)
SUPPLEMENT_SUBEXPERIMENTS = (
    "sequence_bank",
    "field_ping_readout",
    "global_ping_score_spike_prediction",
    "real_probe_score_spike_deflection",
    "overlap_gated_stsp_recruitment",
    "high_stsp_overlap_ablation",
    "supplement",
    "score_shuffle_null",
    "overlap_threshold_sensitivity",
)
BOTH_SCOPE_FLAGS = (
    "--run-sequence-bank",
    "--run-field-ping-readout",
    "--run-global-ping-score-spike-prediction",
    "--run-real-probe-score-spike-deflection",
    "--run-overlap-gated-stsp-recruitment",
    "--run-high-stsp-overlap-ablation",
    "--run-supplement",
    "--run-score-shuffle-null",
    "--run-overlap-threshold-sensitivity",
    "--force-main-outputs",
)
