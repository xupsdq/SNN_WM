from __future__ import annotations

from src.experiments.paper_figures.common.artifact_runtime import REUSE_MODES, normalize_reuse_mode

SCHEMA_NAME = "fig4_runtime_artifacts"
SCHEMA_VERSION = 1

TASK_ALL = "all"
TASK_MAIN_SCOPE = "main_scope"
TASK_SUPPLEMENT_SCOPE = "supplement_scope"
TASK_BOTH_SCOPE = "both_scope"
TASK_PAIR_SAMPLING = "pair_sampling"
TASK_SIMILARITY_ENTRY = "similarity_entry"
TASK_ROLLOUTS = "rollouts"
TASK_OVERLAP_LOCALIZATION = "overlap_localization"
TASK_OVERLAP_ACCURACY_IDENTIFICATION = "overlap_accuracy_identification"
TASK_DECISION_SPIKE_DISPLACEMENT = "decision_spike_displacement"
TASK_DECISION_DEFLECTION = "decision_deflection"
TASK_OVERLAP_PERTURBATION = "overlap_perturbation"
TASK_SUPPLEMENT = "supplement"

TASK_IDS = (
    TASK_ALL,
    TASK_MAIN_SCOPE,
    TASK_SUPPLEMENT_SCOPE,
    TASK_BOTH_SCOPE,
    TASK_PAIR_SAMPLING,
    TASK_SIMILARITY_ENTRY,
    TASK_ROLLOUTS,
    TASK_OVERLAP_LOCALIZATION,
    TASK_OVERLAP_ACCURACY_IDENTIFICATION,
    TASK_DECISION_SPIKE_DISPLACEMENT,
    TASK_DECISION_DEFLECTION,
    TASK_OVERLAP_PERTURBATION,
    TASK_SUPPLEMENT,
)

TABLE_MANIFEST_COLUMNS = (
    "name",
    "filename",
    "rows",
    "columns",
    "sha256",
    "table_digest",
)

PAIR_SAMPLING_FILES = {
    "pair_trials": "pair_trials.csv",
    "pair_candidate_pool": "pair_candidate_pool.csv",
    "overlap_matched_pairs": "overlap_matched_pairs.csv",
    "perturbation_masks": "perturbation_masks.csv",
}

PAIR_MASK_MANIFEST_COLUMNS = (
    "network_seed",
    "pair_id",
    "mask_name",
    "shape",
    "storage_file",
    "storage_key",
    "sha256",
)

SIMILARITY_ENTRY_FILES = {
    "trial_metrics": "trial_metrics.csv",
    "repeat_metrics": "repeat_metrics.csv",
}

ROLLOUT_BANK_FILES = {
    "rollout_manifest": "rollout_manifest.csv",
    "condition_metrics": "condition_metrics.csv",
    "perturbation_masks": "perturbation_masks.csv",
    "l3_replay_capture_manifest": "l3_replay_capture_manifest.csv",
}

ARRAY_MANIFEST_COLUMNS = (
    "name",
    "storage_file",
    "storage_key",
    "shape",
    "dtype",
    "sha256",
)


__all__ = [
    "ARRAY_MANIFEST_COLUMNS",
    "PAIR_MASK_MANIFEST_COLUMNS",
    "PAIR_SAMPLING_FILES",
    "REUSE_MODES",
    "ROLLOUT_BANK_FILES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SIMILARITY_ENTRY_FILES",
    "TABLE_MANIFEST_COLUMNS",
    "TASK_ALL",
    "TASK_BOTH_SCOPE",
    "TASK_DECISION_DEFLECTION",
    "TASK_DECISION_SPIKE_DISPLACEMENT",
    "TASK_IDS",
    "TASK_MAIN_SCOPE",
    "TASK_OVERLAP_ACCURACY_IDENTIFICATION",
    "TASK_OVERLAP_LOCALIZATION",
    "TASK_OVERLAP_PERTURBATION",
    "TASK_PAIR_SAMPLING",
    "TASK_ROLLOUTS",
    "TASK_SIMILARITY_ENTRY",
    "TASK_SUPPLEMENT",
    "TASK_SUPPLEMENT_SCOPE",
    "normalize_reuse_mode",
]
