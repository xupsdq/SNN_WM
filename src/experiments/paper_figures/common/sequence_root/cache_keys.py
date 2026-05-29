from __future__ import annotations

from typing import Any

import pandas as pd

from src.experiments.paper_figures.common.sequence_root.schemas import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TASK_SHARED_SEQUENCE_ROOT_BANK,
    TASK_SHARED_SEQUENCE_SPECS,
)
from src.experiments.paper_figures.fig3.cache_keys import (
    cache_key_digest,
    model_fingerprint,
    sha256_file,
    table_digest,
)


def sequence_specs_hash(
    sequence_trials: pd.DataFrame,
    singleton_reference_trials: pd.DataFrame,
    partial_cue_trials: pd.DataFrame,
) -> str:
    return table_digest(
        {
            "partial_cue_trials": partial_cue_trials,
            "sequence_trials": sequence_trials,
            "singleton_reference_trials": singleton_reference_trials,
        }
    )


def build_shared_sequence_specs_cache_key(cfg: Any) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_SHARED_SEQUENCE_SPECS,
        "network_seed": int(getattr(cfg, "network_seed")),
        "dataset_root": str(getattr(cfg, "dataset_root")),
        "dataset_split": str(getattr(cfg, "split")),
        "sequence_lengths": [int(v) for v in getattr(cfg, "sequence_lengths")],
        "num_sequences": int(getattr(cfg, "num_sequences")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "delay_ms": int(getattr(cfg, "delay_ms")),
        "dt": float(getattr(cfg, "dt")),
        "partial_cue_keep_fraction": float(getattr(cfg, "partial_cue_keep_fraction")),
        "target_position": str(getattr(cfg, "target_position")),
        "smoke": bool(getattr(cfg, "smoke", False)),
    }


def build_shared_sequence_root_bank_cache_key(
    cfg: Any,
    *,
    specs_hash: str,
    fig3_state_bank_key_digest: str,
    fig6_sequence_bank_key_digest: str,
) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_SHARED_SEQUENCE_ROOT_BANK,
        "network_seed": int(getattr(cfg, "network_seed")),
        "dataset_split": str(getattr(cfg, "split")),
        "sequence_specs_hash": str(specs_hash),
        "model": model_fingerprint(getattr(cfg, "model_path")),
        "dt": float(getattr(cfg, "dt")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "delay_ms": int(getattr(cfg, "delay_ms")),
        "fig3_state_bank_cache_key_digest": str(fig3_state_bank_key_digest),
        "fig6_sequence_bank_cache_key_digest": str(fig6_sequence_bank_key_digest),
        "smoke": bool(getattr(cfg, "smoke", False)),
    }


__all__ = [
    "build_shared_sequence_root_bank_cache_key",
    "build_shared_sequence_specs_cache_key",
    "cache_key_digest",
    "sequence_specs_hash",
    "sha256_file",
    "table_digest",
]

