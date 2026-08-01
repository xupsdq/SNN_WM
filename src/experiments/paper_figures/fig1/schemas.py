from __future__ import annotations

SCHEMA_VERSION = 1
SCHEMA_NAME = "fig1_runtime_artifacts"

TASK_TRIAL_SPECS = "trial_specs"
TASK_BASELINE = "baseline"
TASK_DELAY_FEATURE_BANK = "delay_feature_bank"
TASK_DELAY_DECODER = "delay_decoder"
TASK_DMS_BOUNDARY_BANK = "dms_boundary_bank"
TASK_DMS_SHUFFLE_READOUT = "dms_shuffle_readout"
TASK_DMS_DELAY_SWEEP_READOUT = "dms_delay_sweep_readout"
TASK_FIRING_RATE_CONTROL = "firing_rate_control"
TASK_TIME_BINNED_FIRING_RATE_CONTROL = "time_binned_firing_rate_control"

TASK_IDS = (
    TASK_TRIAL_SPECS,
    TASK_BASELINE,
    TASK_DELAY_FEATURE_BANK,
    TASK_DELAY_DECODER,
    TASK_DMS_BOUNDARY_BANK,
    TASK_DMS_SHUFFLE_READOUT,
    TASK_DMS_DELAY_SWEEP_READOUT,
    TASK_FIRING_RATE_CONTROL,
    TASK_TIME_BINNED_FIRING_RATE_CONTROL,
)

REUSE_MODES = ("off", "auto", "require", "force")
TRIAL_SPEC_FILES = {
    "baseline": "baseline_eval_trials.csv",
    "delay_train": "delay_decode_train_trials.csv",
    "delay_test": "delay_decode_test_trials.csv",
    "dms": "dms_shuffle_trials.csv",
}
TRIAL_SPEC_MANIFEST_COLUMNS = (
    "name",
    "filename",
    "rows",
    "columns",
    "sha256",
    "trial_specs_digest",
)

DELAY_FEATURE_NPZ_KEYS = ("x", "y", "trial_ids")
BOUNDARY_STATE_KEYS = ("v_mem", "g_e", "res", "inh_trace", "u", "x")
DMS_ROW_HASH_COLUMNS = (
    "batch_id",
    "row_index",
    "trial_id",
    "sample_image_id",
    "sample_label",
    "probe_image_id",
    "probe_label",
)
DMS_MANIFEST_COLUMNS = (*DMS_ROW_HASH_COLUMNS, "batch_row_hash")


def normalize_reuse_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in REUSE_MODES:
        choices = ", ".join(REUSE_MODES)
        raise ValueError(f"Unsupported reuse-artifacts mode: {value!r}. Expected one of: {choices}")
    return mode


__all__ = [
    "BOUNDARY_STATE_KEYS",
    "DMS_MANIFEST_COLUMNS",
    "DMS_ROW_HASH_COLUMNS",
    "DELAY_FEATURE_NPZ_KEYS",
    "REUSE_MODES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "TASK_BASELINE",
    "TASK_DELAY_DECODER",
    "TASK_DELAY_FEATURE_BANK",
    "TASK_DMS_BOUNDARY_BANK",
    "TASK_DMS_DELAY_SWEEP_READOUT",
    "TASK_DMS_SHUFFLE_READOUT",
    "TASK_FIRING_RATE_CONTROL",
    "TASK_TIME_BINNED_FIRING_RATE_CONTROL",
    "TASK_IDS",
    "TASK_TRIAL_SPECS",
    "TRIAL_SPEC_FILES",
    "TRIAL_SPEC_MANIFEST_COLUMNS",
    "normalize_reuse_mode",
]
