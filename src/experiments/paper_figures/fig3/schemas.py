from __future__ import annotations

SCHEMA_NAME = "fig3_runtime_artifacts"
SCHEMA_VERSION = 1

TASK_ALL = "all"
TASK_SEQUENCE_TRIAL_SPECS = "sequence_trial_specs"
TASK_STATE_BANK = "state_bank"
TASK_PROGRESSIVE_UPDATE = "progressive_update"
TASK_PEAK_VALLEY_LANDSCAPE = "peak_valley_landscape"
TASK_NEUTRAL_PING = "neutral_ping"
TASK_WEAK_PROBE = "weak_probe"
TASK_SUPPLEMENT = "supplement"

TASK_IDS = (
    TASK_ALL,
    TASK_SEQUENCE_TRIAL_SPECS,
    TASK_STATE_BANK,
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
    "BOUNDARY_MANIFEST_COLUMNS",
    "LANDSCAPE_MANIFEST_COLUMNS",
    "REUSE_MODES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SEQUENCE_SPEC_FILES",
    "SEQUENCE_SPEC_MANIFEST_COLUMNS",
    "STATE_BANK_MANIFEST_COLUMNS",
    "TABLE_MANIFEST_COLUMNS",
    "TASK_ALL",
    "TASK_IDS",
    "TASK_NEUTRAL_PING",
    "TASK_PEAK_VALLEY_LANDSCAPE",
    "TASK_PROGRESSIVE_UPDATE",
    "TASK_SEQUENCE_TRIAL_SPECS",
    "TASK_STATE_BANK",
    "TASK_SUPPLEMENT",
    "TASK_WEAK_PROBE",
    "normalize_reuse_mode",
]
