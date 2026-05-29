from __future__ import annotations

from enum import Enum

SCHEMA_NAME = "paper_figure_input_spec_identity"
SCHEMA_VERSION = 1


class SpecFamily(str, Enum):
    DMS_TRIAL = "dms_trial"
    PAIR_EPISODE = "pair_episode"
    SEQUENCE = "sequence"
    MASK = "mask"
    PERTURBATION = "perturbation"
    PROBE_TARGET = "probe_target"


class SpecRole(str, Enum):
    SOURCE = "source"
    VIEW = "view"


MANIFEST_COLUMNS = (
    "name",
    "filename",
    "rows",
    "columns",
    "sha256",
    "table_digest",
)

PROVENANCE_REQUIRED_KEYS = (
    "schema_name",
    "schema_version",
    "spec_family",
    "producer_task",
    "consumer_figures",
    "consumer_tasks",
    "dataset_root",
    "dataset_split",
    "network_seed",
    "model_fingerprint",
    "sampling_config",
    "rng_policy",
    "parent_spec_digests",
    "row_identity_columns",
)

SPEC_PROVENANCE_FILE = "spec_provenance.json"
CACHE_KEY_FILE = "cache_key.json"
ARTIFACT_DIGEST_FILE = "artifact_digest.json"
MANIFEST_FILE = "manifest.csv"
SPEC_VIEW_LINK_FILE = "spec_view_link.json"

__all__ = [
    "ARTIFACT_DIGEST_FILE",
    "CACHE_KEY_FILE",
    "MANIFEST_COLUMNS",
    "MANIFEST_FILE",
    "PROVENANCE_REQUIRED_KEYS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SPEC_PROVENANCE_FILE",
    "SPEC_VIEW_LINK_FILE",
    "SpecFamily",
    "SpecRole",
]
