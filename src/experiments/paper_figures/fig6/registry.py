FIGURE_ID = "fig6"
EXPERIMENT_ID = "fig6_peak_amplified_reentry"
RUNNER_MODULE = "src.experiments.paper_figures.fig6.run_task"
COMPATIBILITY_MODULE = "src.experiments.paper_figures.fig6_peak_amplified_reentry_experiment"
SCOPE_TASKS = {"main": "main_scope", "supplement": "supplement_scope", "both": "both_scope"}
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
SUBEXPERIMENT_TASKS = {
    "sequence_bank": "sequence_bank",
    "field_ping_readout": "field_ping_readout",
    "global_ping_score_spike_prediction": "global_ping_score_spike_prediction",
    "ping_score_spike_prediction": "global_ping_score_spike_prediction",
    "real_probe_score_spike_deflection": "real_probe_score_spike_deflection",
    "overlap_gated_stsp_recruitment": "overlap_gated_stsp_recruitment",
    "high_stsp_overlap_ablation": "high_stsp_overlap_ablation",
    "supplement": "supplement",
    "score_shuffle_null": "score_shuffle_null",
    "overlap_threshold_sensitivity": "overlap_threshold_sensitivity",
}
ARCHIVED_SUBEXPERIMENTS = (
    "peak_source_attribution",
    "peak_update_history",
    "peak_input_overlap_origin",
    "real_reentry_rollout",
    "real_downstream_metrics",
    "peak_enrichment",
    "update_recency_model",
    "peak_weighted_overlap",
    "reentry_prediction",
    "downstream_prediction",
    "peak_perturbation",
    "score_basin_sparsification",
    "fig6_downstream_exploratory",
)

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
