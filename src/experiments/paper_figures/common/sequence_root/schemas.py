from __future__ import annotations

from src.experiments.paper_figures.common.artifact_runtime import REUSE_MODES, normalize_reuse_mode

SCHEMA_NAME = "fig3_fig6_shared_sequence_root"
SCHEMA_VERSION = 1

TASK_SHARED_SEQUENCE_SPECS = "shared_sequence_specs"
TASK_SHARED_SEQUENCE_ROOT_BANK = "shared_sequence_root_bank"
TASK_FIG3_STATE_BANK_VIEW = "fig3_state_bank_view"
TASK_FIG6_SEQUENCE_BANK_VIEW = "fig6_sequence_bank_view"
TASK_ALL = "all"

TASK_IDS = (
    TASK_ALL,
    TASK_SHARED_SEQUENCE_SPECS,
    TASK_SHARED_SEQUENCE_ROOT_BANK,
    TASK_FIG3_STATE_BANK_VIEW,
    TASK_FIG6_SEQUENCE_BANK_VIEW,
)

SEQUENCE_SPEC_FILES = {
    "sequence_trials": "sequence_trials.csv",
    "singleton_reference_trials": "singleton_reference_trials.csv",
    "partial_cue_trials": "partial_cue_trials.csv",
}

TABLE_MANIFEST_COLUMNS = (
    "name",
    "filename",
    "rows",
    "columns",
    "sha256",
    "table_digest",
)

ROOT_BANK_MANIFEST_COLUMNS = (
    "name",
    "relative_path",
    "cache_key_digest",
    "artifact_digest",
)


__all__ = [
    "REUSE_MODES",
    "ROOT_BANK_MANIFEST_COLUMNS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SEQUENCE_SPEC_FILES",
    "TABLE_MANIFEST_COLUMNS",
    "TASK_ALL",
    "TASK_FIG3_STATE_BANK_VIEW",
    "TASK_FIG6_SEQUENCE_BANK_VIEW",
    "TASK_IDS",
    "TASK_SHARED_SEQUENCE_ROOT_BANK",
    "TASK_SHARED_SEQUENCE_SPECS",
    "normalize_reuse_mode",
]
