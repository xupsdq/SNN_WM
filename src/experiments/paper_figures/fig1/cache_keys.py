from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.experiments.paper_figures.fig1.schemas import SCHEMA_NAME, SCHEMA_VERSION, TASK_TRIAL_SPECS


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def dataframe_hash(df: pd.DataFrame, *, columns: list[str] | tuple[str, ...] | None = None) -> str:
    selected = df.loc[:, list(columns)] if columns is not None else df
    csv_text = selected.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def trial_specs_hash(specs: Mapping[str, pd.DataFrame]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(specs):
        df = specs[name]
        hasher.update(str(name).encode("utf-8"))
        hasher.update(b"\n")
        hasher.update(",".join(str(col) for col in df.columns).encode("utf-8"))
        hasher.update(b"\n")
        hasher.update(dataframe_hash(df).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def build_cache_key(
    cfg: Any,
    *,
    task_id: str,
    trial_hash: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": str(task_id),
        "model": model_fingerprint(getattr(cfg, "model_path")),
        "network_seed": int(getattr(cfg, "network_seed")),
        "dataset_split": str(getattr(cfg, "split")),
        "trial_spec_hash": str(trial_hash),
        "dt": float(getattr(cfg, "dt")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "delay_ms": int(getattr(cfg, "delay_ms")),
        "delay_points_ms": [int(v) for v in getattr(cfg, "delay_points_ms")],
        "dms_sample_ms": int(getattr(cfg, "dms_sample_ms")),
        "dms_delay_ms": int(getattr(cfg, "dms_delay_ms")),
        "dms_delay_sweep_ms": [int(v) for v in getattr(cfg, "dms_delay_sweep_ms")],
        "probe_ms": int(getattr(cfg, "probe_ms")),
        "batch_size": int(getattr(cfg, "batch_size")),
        "dms_batch_size": int(getattr(cfg, "dms_batch_size")),
    }
    if extra:
        payload["extra"] = _json_safe(extra)
    return payload


def build_trial_specs_cache_key(cfg: Any) -> dict[str, Any]:
    """Build a config-only cache key for trial sampling.

    The trial-spec artifact validates DataFrame contents through manifest hashes;
    this key intentionally captures only inputs that control which rows are sampled.
    """
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_TRIAL_SPECS,
        "network_seed": int(getattr(cfg, "network_seed")),
        "dataset_root": str(Path(getattr(cfg, "dataset_root")).resolve()),
        "dataset_split": str(getattr(cfg, "split")),
        "baseline_eval_per_class": int(getattr(cfg, "baseline_eval_per_class")),
        "delay_decode_train_per_class": int(getattr(cfg, "delay_decode_train_per_class")),
        "delay_decode_test_per_class": int(getattr(cfg, "delay_decode_test_per_class")),
        "dms_num_trials": int(getattr(cfg, "dms_num_trials")),
        "delay_points_ms": [int(v) for v in getattr(cfg, "delay_points_ms")],
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "delay_ms": int(getattr(cfg, "delay_ms")),
        "dms_sample_ms": int(getattr(cfg, "dms_sample_ms")),
        "dms_delay_ms": int(getattr(cfg, "dms_delay_ms")),
        "dms_delay_sweep_ms": [int(v) for v in getattr(cfg, "dms_delay_sweep_ms")],
        "probe_ms": int(getattr(cfg, "probe_ms")),
        "batch_size": int(getattr(cfg, "batch_size")),
        "dms_batch_size": int(getattr(cfg, "dms_batch_size")),
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
    "build_cache_key",
    "build_trial_specs_cache_key",
    "cache_key_digest",
    "dataframe_hash",
    "model_fingerprint",
    "sha256_file",
    "trial_specs_hash",
]
