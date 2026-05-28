from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.experiments.paper_figures.fig3.schemas import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TASK_SEQUENCE_TRIAL_SPECS,
    TASK_STATE_BANK,
)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        return {
            "path": str(resolved),
            "exists": False,
            "sha256": "",
            "size_bytes": 0,
            "mtime_ns": 0,
        }
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "sha256": sha256_file(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def dataframe_hash(df: pd.DataFrame, *, columns: list[str] | tuple[str, ...] | None = None) -> str:
    selected = df.loc[:, list(columns)] if columns is not None else df
    csv_text = selected.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def table_digest(tables: Mapping[str, pd.DataFrame]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(tables):
        df = tables[name]
        hasher.update(str(name).encode("utf-8"))
        hasher.update(b"\n")
        hasher.update(",".join(str(col) for col in df.columns).encode("utf-8"))
        hasher.update(b"\n")
        hasher.update(dataframe_hash(df).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


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


def build_sequence_specs_cache_key(cfg: Any) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_SEQUENCE_TRIAL_SPECS,
        "network_seed": int(getattr(cfg, "network_seed")),
        "dataset_root": str(Path(getattr(cfg, "dataset_root")).resolve()),
        "dataset_split": str(getattr(cfg, "split")),
        "sequence_lengths": [int(v) for v in getattr(cfg, "sequence_lengths")],
        "primary_sequence_length": int(getattr(cfg, "primary_sequence_length")),
        "main_sequence_length": int(getattr(cfg, "main_sequence_length")),
        "main_only_seq_len_10": bool(getattr(cfg, "main_only_seq_len_10")),
        "num_sequences": int(getattr(cfg, "num_sequences")),
        "partial_cue_keep_fraction": float(getattr(cfg, "partial_cue_keep_fraction")),
        "target_position": str(getattr(cfg, "target_position")),
        "smoke": bool(getattr(cfg, "smoke", False)),
    }


def build_state_bank_cache_key(cfg: Any, *, specs_hash: str) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_STATE_BANK,
        "network_seed": int(getattr(cfg, "network_seed")),
        "dataset_split": str(getattr(cfg, "split")),
        "sequence_specs_hash": str(specs_hash),
        "model": model_fingerprint(getattr(cfg, "model_path")),
        "dt": float(getattr(cfg, "dt")),
        "batch_size": int(getattr(cfg, "batch_size")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "delay_ms": int(getattr(cfg, "delay_ms")),
        "sample_steps": int(getattr(cfg, "sample_steps")),
        "delay_steps": int(getattr(cfg, "delay_steps")),
        "state_variables": ["g", "u", "x"],
        "state_conditions": ["S0", "S_1..S_K", "S_final", "singleton_reference", "singleton_boundary"],
        "functional_restore_mode": str(getattr(cfg, "functional_restore_mode")),
        "sequence_lengths": [int(v) for v in getattr(cfg, "sequence_lengths")],
        "num_sequences": int(getattr(cfg, "num_sequences")),
        "smoke": bool(getattr(cfg, "smoke", False)),
    }


def cache_key_digest(cache_key: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(cache_key).encode("utf-8")).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


__all__ = [
    "build_sequence_specs_cache_key",
    "build_state_bank_cache_key",
    "cache_key_digest",
    "dataframe_hash",
    "model_fingerprint",
    "sequence_specs_hash",
    "sha256_file",
    "table_digest",
]
