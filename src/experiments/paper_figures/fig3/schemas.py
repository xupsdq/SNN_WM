from __future__ import annotations

SCHEMA_NAME = "fig3_runtime_artifacts"
SCHEMA_VERSION = 4
CUE_SPECIFICITY_MISMATCHED_SELECTION_POLICY = "same_label_different_image_excluding_sequence_v1"

TASK_ALL = "all"
TASK_SEQUENCE_TRIAL_SPECS = "sequence_trial_specs"
TASK_STATE_BANK = "state_bank"
TASK_BOUNDARY_CONDITION_SPECS = "boundary_condition_specs"
TASK_ACCESS_JOB_SPECS = "access_job_specs"
TASK_BOUNDARY_STATE_BANK = "boundary_state_bank"
TASK_MORPHOLOGY_DECOMPOSITION = "morphology_decomposition"
TASK_NEUTRAL_PING_ACCESS = "neutral_ping_access"
TASK_WEAK_CUE_ACCESS = "weak_cue_access"
TASK_CUE_SPECIFICITY_SPECS = "cue_specificity_specs"
TASK_CUE_SPECIFICITY_ACCESS = "cue_specificity_access"
TASK_EXEMPLAR_DECODER_SPECS = "exemplar_decoder_specs"
TASK_EXEMPLAR_DECODER_STATE_BANK = "exemplar_decoder_state_bank"
TASK_EXEMPLAR_DECODER = "exemplar_decoder"
TASK_EXEMPLAR_DECODER_SUMMARY = "exemplar_decoder_summary"
TASK_MORPHOLOGY_FUNCTION_COUPLING = "morphology_function_coupling"
TASK_BOUNDARY_SUMMARY = "boundary_summary"
TASK_PROGRESSIVE_UPDATE = "progressive_update"
TASK_PEAK_VALLEY_LANDSCAPE = "peak_valley_landscape"
TASK_NEUTRAL_PING = "neutral_ping"
TASK_WEAK_PROBE = "weak_probe"
TASK_SUPPLEMENT = "supplement"

TASK_IDS = (
    TASK_ALL,
    TASK_SEQUENCE_TRIAL_SPECS,
    TASK_STATE_BANK,
    TASK_BOUNDARY_CONDITION_SPECS,
    TASK_ACCESS_JOB_SPECS,
    TASK_BOUNDARY_STATE_BANK,
    TASK_MORPHOLOGY_DECOMPOSITION,
    TASK_NEUTRAL_PING_ACCESS,
    TASK_WEAK_CUE_ACCESS,
    TASK_CUE_SPECIFICITY_SPECS,
    TASK_CUE_SPECIFICITY_ACCESS,
    TASK_EXEMPLAR_DECODER_SPECS,
    TASK_EXEMPLAR_DECODER_STATE_BANK,
    TASK_EXEMPLAR_DECODER,
    TASK_EXEMPLAR_DECODER_SUMMARY,
    TASK_MORPHOLOGY_FUNCTION_COUPLING,
    TASK_BOUNDARY_SUMMARY,
    TASK_PROGRESSIVE_UPDATE,
    TASK_PEAK_VALLEY_LANDSCAPE,
    TASK_NEUTRAL_PING,
    TASK_WEAK_PROBE,
    TASK_SUPPLEMENT,
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

SEQUENCE_SPEC_FILES = {
    "sequence_trials": "sequence_trials.csv",
    "singleton_reference_trials": "singleton_reference_trials.csv",
    "partial_cue_trials": "partial_cue_trials.csv",
}

SEQUENCE_SPEC_MANIFEST_COLUMNS = TABLE_MANIFEST_COLUMNS

TABLE_ARTIFACT_MANIFEST_COLUMNS = TABLE_MANIFEST_COLUMNS

MORPHOLOGY_BOUNDARY_REQUIRED_COLUMNS = (
    "network_seed",
    "condition_id",
    "sequence_id",
    "seq_len",
    "delay_ms",
    "layer",
    "state_variable",
    "N_eff",
    "N_eff_fraction",
    "multi_item_retention_index",
    "latest_collapse_index",
)

FUNCTIONAL_BOUNDARY_REQUIRED_COLUMNS = (
    "network_seed",
    "condition_id",
    "sequence_id",
    "seq_len",
    "delay_ms",
    "keep_prob",
    "accessible_item_count",
    "singleton_access_count",
    "sequence_access_count",
    "rescued_count",
    "singleton_access_fraction",
    "sequence_access_fraction",
    "rescued_fraction",
    "functional_retention_index",
    "mean_G_i",
    "mean_U_i",
    "mean_G_i_norm",
)

BOUNDARY_SUMMARY_REQUIRED_COLUMNS = (
    "network_seed",
    "condition_id",
    "seq_len",
    "delay_ms",
    "N_eff",
    "N_eff_fraction",
    "multi_item_retention_index",
    "latest_collapse_index",
    "accessible_item_count",
    "singleton_access_count",
    "sequence_access_count",
    "rescued_count",
    "singleton_access_fraction",
    "sequence_access_fraction",
    "rescued_fraction",
    "functional_retention_index",
)

CUE_SPECIFICITY_SPECS_REQUIRED_COLUMNS = (
    "cue_specificity_trial_id",
    "job_id",
    "condition_id",
    "sequence_id",
    "seq_len",
    "delay_ms",
    "target_position",
    "target_image_id",
    "target_label",
    "cue_type",
    "cue_position",
    "cue_image_id",
    "cue_label",
    "state_condition",
    "memory_condition",
    "keep_prob",
    "repeat_id",
    "mask_seed",
    "mask_group_id",
    "ordered_item_ids",
    "ordered_item_labels",
    "unseen_labels",
    "cue_selection_policy",
    "cue_is_sequence_member",
    "cue_is_same_label_foil",
    "mismatched_selection_policy",
)

CUE_SPECIFICITY_METRICS_REQUIRED_COLUMNS = (
    "network_seed",
    "condition_id",
    "sequence_id",
    "seq_len",
    "delay_ms",
    "target_position",
    "cue_type",
    "state_condition",
    "memory_condition",
    "keep_prob",
    "P_target",
    "P_cue_label",
    "P_seen_item",
    "P_other_seen_item",
    "P_latest_item",
    "P_unseen",
    "P_silent",
)

EXEMPLAR_DECODER_EPISODE_SPECS_REQUIRED_COLUMNS = (
    "network_seed",
    "sequence_id",
    "digit_label",
    "exemplar_index",
    "target_image_id",
    "episode_id",
    "target_position",
    "seq_len",
    "context_seed",
)

EXEMPLAR_DECODER_SEQUENCE_SPECS_REQUIRED_COLUMNS = (
    *EXEMPLAR_DECODER_EPISODE_SPECS_REQUIRED_COLUMNS,
    "stage_k",
    "item_image_id",
    "item_label",
)

EXEMPLAR_DECODER_STATE_MANIFEST_REQUIRED_COLUMNS = (
    "network_seed",
    "sequence_id",
    "digit_label",
    "exemplar_index",
    "target_image_id",
    "episode_id",
    "target_position",
    "condition",
    "feature_name",
    "feature_shape",
    "storage_file",
    "storage_key",
    "storage_sha256",
    "state_hash",
)

EXEMPLAR_DECODER_METRICS_REQUIRED_COLUMNS = (
    "network_seed",
    "condition",
    "balanced_accuracy",
    "n_predictions",
    "n_folds",
    "n_digit_labels",
    "hash_validation_pass",
)

EXEMPLAR_DECODER_HASH_VALIDATION_REQUIRED_COLUMNS = (
    "network_seed",
    "condition",
    "digit_label",
    "fold_id",
    "train_episode_ids",
    "test_episode_id",
    "train_n",
    "test_n",
    "state_hash_overlap_count",
    "state_hash_overlap",
    "scaler_fit_scope",
    "model_fit_scope",
    "decoder_family",
    "decoder_penalty",
    "decoder_C",
    "decoder_solver",
    "passed",
)

STATE_BANK_MANIFEST_COLUMNS = (
    "artifact_kind",
    "network_seed",
    "sequence_id",
    "seq_len",
    "state_condition",
    "stage_k",
    "layer",
    "state_variable",
    "shape",
    "storage_file",
    "storage_key",
    "sha256",
)

BOUNDARY_MANIFEST_COLUMNS = (
    "artifact_kind",
    "network_seed",
    "sequence_id",
    "seq_len",
    "state_condition",
    "stage_k",
    "layer",
    "state_key",
    "shape",
    "storage_file",
    "storage_key",
    "sha256",
)

LANDSCAPE_MANIFEST_COLUMNS = (
    "artifact_kind",
    "network_seed",
    "sequence_id",
    "landscape_key",
    "shape",
    "storage_file",
    "storage_key",
    "sha256",
)


def normalize_reuse_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in REUSE_MODES:
        choices = ", ".join(REUSE_MODES)
        raise ValueError(f"Unsupported reuse-artifacts mode: {value!r}. Expected one of: {choices}")
    return mode


__all__ = [
    "BOUNDARY_SUMMARY_REQUIRED_COLUMNS",
    "BOUNDARY_MANIFEST_COLUMNS",
    "FUNCTIONAL_BOUNDARY_REQUIRED_COLUMNS",
    "CUE_SPECIFICITY_METRICS_REQUIRED_COLUMNS",
    "CUE_SPECIFICITY_MISMATCHED_SELECTION_POLICY",
    "CUE_SPECIFICITY_SPECS_REQUIRED_COLUMNS",
    "EXEMPLAR_DECODER_EPISODE_SPECS_REQUIRED_COLUMNS",
    "EXEMPLAR_DECODER_HASH_VALIDATION_REQUIRED_COLUMNS",
    "EXEMPLAR_DECODER_METRICS_REQUIRED_COLUMNS",
    "EXEMPLAR_DECODER_SEQUENCE_SPECS_REQUIRED_COLUMNS",
    "EXEMPLAR_DECODER_STATE_MANIFEST_REQUIRED_COLUMNS",
    "LANDSCAPE_MANIFEST_COLUMNS",
    "MORPHOLOGY_BOUNDARY_REQUIRED_COLUMNS",
    "REUSE_MODES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SEQUENCE_SPEC_FILES",
    "SEQUENCE_SPEC_MANIFEST_COLUMNS",
    "STATE_BANK_MANIFEST_COLUMNS",
    "TABLE_ARTIFACT_MANIFEST_COLUMNS",
    "TABLE_MANIFEST_COLUMNS",
    "TASK_ACCESS_JOB_SPECS",
    "TASK_ALL",
    "TASK_BOUNDARY_CONDITION_SPECS",
    "TASK_BOUNDARY_STATE_BANK",
    "TASK_BOUNDARY_SUMMARY",
    "TASK_CUE_SPECIFICITY_ACCESS",
    "TASK_CUE_SPECIFICITY_SPECS",
    "TASK_EXEMPLAR_DECODER",
    "TASK_EXEMPLAR_DECODER_SPECS",
    "TASK_EXEMPLAR_DECODER_STATE_BANK",
    "TASK_EXEMPLAR_DECODER_SUMMARY",
    "TASK_MORPHOLOGY_DECOMPOSITION",
    "TASK_MORPHOLOGY_FUNCTION_COUPLING",
    "TASK_IDS",
    "TASK_NEUTRAL_PING",
    "TASK_NEUTRAL_PING_ACCESS",
    "TASK_PEAK_VALLEY_LANDSCAPE",
    "TASK_PROGRESSIVE_UPDATE",
    "TASK_SEQUENCE_TRIAL_SPECS",
    "TASK_STATE_BANK",
    "TASK_SUPPLEMENT",
    "TASK_WEAK_CUE_ACCESS",
    "TASK_WEAK_PROBE",
    "normalize_reuse_mode",
]
