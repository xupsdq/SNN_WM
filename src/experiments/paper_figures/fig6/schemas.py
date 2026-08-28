from __future__ import annotations

SCHEMA_NAME = "fig6_runtime_artifacts"
SCHEMA_VERSION = 1

TASK_ALL = "all"
TASK_MAIN_SCOPE = "main_scope"
TASK_SUPPLEMENT_SCOPE = "supplement_scope"
TASK_BOTH_SCOPE = "both_scope"
TASK_SEQUENCE_TRIALS = "sequence_trials"
TASK_SEQUENCE_BANK = "sequence_bank"
TASK_FIELD_PING_READOUT = "field_ping_readout"
TASK_GLOBAL_PING_SCORE_SPIKE_PREDICTION = "global_ping_score_spike_prediction"
TASK_REAL_PROBE_SCORE_SPIKE_DEFLECTION = "real_probe_score_spike_deflection"
TASK_OVERLAP_GATED_STSP_RECRUITMENT = "overlap_gated_stsp_recruitment"
TASK_HIGH_STSP_OVERLAP_ABLATION = "high_stsp_overlap_ablation"
TASK_SUPPLEMENT = "supplement"
TASK_SCORE_SHUFFLE_NULL = "score_shuffle_null"
TASK_OVERLAP_THRESHOLD_SENSITIVITY = "overlap_threshold_sensitivity"

TASK_IDS = (
    TASK_ALL,
    TASK_MAIN_SCOPE,
    TASK_SUPPLEMENT_SCOPE,
    TASK_BOTH_SCOPE,
    TASK_SEQUENCE_TRIALS,
    TASK_SEQUENCE_BANK,
    TASK_FIELD_PING_READOUT,
    TASK_GLOBAL_PING_SCORE_SPIKE_PREDICTION,
    TASK_REAL_PROBE_SCORE_SPIKE_DEFLECTION,
    TASK_OVERLAP_GATED_STSP_RECRUITMENT,
    TASK_HIGH_STSP_OVERLAP_ABLATION,
    TASK_SUPPLEMENT,
    TASK_SCORE_SHUFFLE_NULL,
    TASK_OVERLAP_THRESHOLD_SENSITIVITY,
)

REUSE_MODES = ("off", "auto", "require", "force")

TABLE_MANIFEST_COLUMNS = (
    "name",
    "filename",
    "rows",
    "columns",
    "sha256",
    "table_digest",
)

ARRAY_MANIFEST_COLUMNS = (
    "name",
    "storage_file",
    "storage_key",
    "shape",
    "dtype",
    "sha256",
)

BOUNDARY_MANIFEST_COLUMNS = (
    "network_seed",
    "sequence_id",
    "seq_len",
    "layer",
    "state_key",
    "shape",
    "storage_file",
    "storage_key",
    "sha256",
)

SEQUENCE_TRIAL_FILES = {
    "sequence_trials": "sequence_trials.csv",
}

SEQUENCE_BANK_TABLE_FILES = {
    "sequence_meta": "sequence_meta.csv",
    "state_bank_manifest": "state_bank_manifest.csv",
}

SEQUENCE_BANK_ARRAY_FILES = (
    "update_history_matrix.npz",
    "final_support_maps.npz",
)


def normalize_reuse_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in REUSE_MODES:
        choices = ", ".join(REUSE_MODES)
        raise ValueError(f"Unsupported reuse-artifacts mode: {value!r}. Expected one of: {choices}")
    return mode


__all__ = [
    "ARRAY_MANIFEST_COLUMNS",
    "BOUNDARY_MANIFEST_COLUMNS",
    "REUSE_MODES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SEQUENCE_BANK_ARRAY_FILES",
    "SEQUENCE_BANK_TABLE_FILES",
    "SEQUENCE_TRIAL_FILES",
    "TABLE_MANIFEST_COLUMNS",
    "TASK_ALL",
    "TASK_BOTH_SCOPE",
    "TASK_FIELD_PING_READOUT",
    "TASK_GLOBAL_PING_SCORE_SPIKE_PREDICTION",
    "TASK_HIGH_STSP_OVERLAP_ABLATION",
    "TASK_IDS",
    "TASK_MAIN_SCOPE",
    "TASK_OVERLAP_GATED_STSP_RECRUITMENT",
    "TASK_OVERLAP_THRESHOLD_SENSITIVITY",
    "TASK_REAL_PROBE_SCORE_SPIKE_DEFLECTION",
    "TASK_SCORE_SHUFFLE_NULL",
    "TASK_SEQUENCE_BANK",
    "TASK_SEQUENCE_TRIALS",
    "TASK_SUPPLEMENT",
    "TASK_SUPPLEMENT_SCOPE",
    "normalize_reuse_mode",
]
