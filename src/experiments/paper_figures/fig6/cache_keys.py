from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.paper_figures.common.artifact_runtime import cache_key_digest
from src.experiments.paper_figures.fig6.schemas import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TASK_SEQUENCE_BANK,
    TASK_SEQUENCE_TRIALS,
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


def dataframe_hash(df: pd.DataFrame) -> str:
    csv_text = df.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
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


def sequence_trials_hash(sequence_trials: pd.DataFrame) -> str:
    return table_digest({"sequence_trials": sequence_trials})


def build_sequence_trials_cache_key(cfg: Any) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_SEQUENCE_TRIALS,
        "network_seed": int(getattr(cfg, "network_seed")),
        "dataset_root": str(Path(getattr(cfg, "dataset_root")).resolve()),
        "dataset_split": str(getattr(cfg, "split")),
        "sequence_lengths": tuple(int(v) for v in getattr(cfg, "sequence_lengths")),
        "num_sequences": int(getattr(cfg, "num_sequences")),
        "smoke": bool(getattr(cfg, "smoke", False)),
    }


def build_sequence_bank_cache_key(cfg: Any, *, sequence_trials_hash_value: str) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_SEQUENCE_BANK,
        "network_seed": int(getattr(cfg, "network_seed")),
        "sequence_trials_hash": str(sequence_trials_hash_value),
        "model": model_fingerprint(getattr(cfg, "model_path")),
        "dt": float(getattr(cfg, "dt")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "delay_ms": int(getattr(cfg, "delay_ms")),
        "sample_steps": int(getattr(cfg, "sample_steps")),
        "delay_steps": int(getattr(cfg, "delay_steps")),
        "peak_q": float(getattr(cfg, "peak_q")),
        "foreground_threshold": float(getattr(cfg, "foreground_threshold")),
        "real_probe_entry_mode": str(getattr(cfg, "real_probe_entry_mode")),
        "functional_restore_mode": str(getattr(cfg, "functional_restore_mode")),
        "batch_size": int(getattr(cfg, "batch_size")),
        "enable_sequence_bank_batch": bool(getattr(cfg, "enable_sequence_bank_batch")),
        "use_encode_cache": bool(getattr(cfg, "use_encode_cache")),
        "smoke": bool(getattr(cfg, "smoke", False)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "build_sequence_bank_cache_key",
    "build_sequence_trials_cache_key",
    "cache_key_digest",
    "dataframe_hash",
    "model_fingerprint",
    "sequence_trials_hash",
    "sha256_file",
    "table_digest",
]
