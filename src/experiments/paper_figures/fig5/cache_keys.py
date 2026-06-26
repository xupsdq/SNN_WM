from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig5.schemas import (
    POSTPROBE_L2_WRITEBACK_SCHEMA_VERSION,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TASK_PROBE_STSP_UPDATE_BANK,
    TASK_SUPPORT_BANK,
    TASK_TRIAL_SAMPLING,
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


def trials_hash(trials: pd.DataFrame) -> str:
    return table_digest({"local_competition_trials": trials})


def build_trial_sampling_cache_key(cfg: Any) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_TRIAL_SAMPLING,
        "network_seed": int(getattr(cfg, "network_seed")),
        "dataset_root": str(Path(getattr(cfg, "dataset_root")).resolve()),
        "dataset_split": str(getattr(cfg, "split")),
        "model": model_fingerprint(getattr(cfg, "model_path")),
        "dt": float(getattr(cfg, "dt")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "probe_ms": int(getattr(cfg, "probe_ms")),
        "sample_steps": int(getattr(cfg, "sample_steps")),
        "probe_steps": int(getattr(cfg, "probe_steps")),
        "max_trials": int(getattr(cfg, "max_trials")),
        "overlap_mask_mode": str(getattr(cfg, "overlap_mask_mode")),
        "foreground_threshold": float(getattr(cfg, "foreground_threshold")),
        "min_overlap_area": int(getattr(cfg, "min_overlap_area")),
        "min_probe_only_area": int(getattr(cfg, "min_probe_only_area")),
        "medium_q_low": float(getattr(cfg, "medium_q_low")),
        "medium_q_high": float(getattr(cfg, "medium_q_high")),
        "smoke": bool(getattr(cfg, "smoke", False)),
    }


def build_support_bank_cache_key(cfg: Any, *, trial_hash: str) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_SUPPORT_BANK,
        "network_seed": int(getattr(cfg, "network_seed")),
        "trial_sampling_hash": str(trial_hash),
        "model": model_fingerprint(getattr(cfg, "model_path")),
        "dt": float(getattr(cfg, "dt")),
        "batch_size": int(getattr(cfg, "batch_size")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "delay_ms": int(getattr(cfg, "delay_ms")),
        "probe_ms": int(getattr(cfg, "probe_ms")),
        "sample_steps": int(getattr(cfg, "sample_steps")),
        "delay_steps": int(getattr(cfg, "delay_steps")),
        "probe_steps": int(getattr(cfg, "probe_steps")),
        "early_window_ms": int(getattr(cfg, "early_window_ms")),
        "drive_score_threshold": float(getattr(cfg, "drive_score_threshold")),
        "local_kernel_radius": int(getattr(cfg, "local_kernel_radius")),
        "peak_support_q": float(getattr(cfg, "peak_support_q")),
        "perturbation_mode": str(getattr(cfg, "perturbation_mode")),
        "perturbation_attenuation_factor": float(getattr(cfg, "perturbation_attenuation_factor")),
        "fig5d_include_balanced": bool(getattr(cfg, "fig5d_include_balanced")),
        "event_align_pre_steps": int(getattr(cfg, "event_align_pre_steps")),
        "event_align_post_steps": int(getattr(cfg, "event_align_post_steps")),
        "enable_branch_batch": bool(getattr(cfg, "enable_branch_batch")),
        "smoke": bool(getattr(cfg, "smoke", False)),
    }


def build_probe_stsp_update_bank_cache_key(
    cfg: Any,
    *,
    trial_hash: str,
    support_bank_digest: str,
    support_bank_cache_key_digest: str,
    unit_group_digest: str,
    conditions: tuple[str, ...],
    variable_sets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "l2_writeback_schema_version": int(POSTPROBE_L2_WRITEBACK_SCHEMA_VERSION),
        "task_id": TASK_PROBE_STSP_UPDATE_BANK,
        "network_seed": int(getattr(cfg, "network_seed")),
        "dataset_root": str(Path(getattr(cfg, "dataset_root")).resolve()),
        "dataset_split": str(getattr(cfg, "split")),
        "model": model_fingerprint(getattr(cfg, "model_path")),
        "trial_sampling_hash": str(trial_hash),
        "parent_support_bank_digest": str(support_bank_digest),
        "parent_support_bank_cache_key_digest": str(support_bank_cache_key_digest),
        "unit_group_digest": str(unit_group_digest),
        "conditions": [str(condition) for condition in conditions],
        "snapshot_variable_sets": [str(value) for value in (variable_sets or ())],
        "dt": float(getattr(cfg, "dt")),
        "batch_size": int(getattr(cfg, "batch_size")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "delay_ms": int(getattr(cfg, "delay_ms")),
        "probe_ms": int(getattr(cfg, "probe_ms")),
        "sample_steps": int(getattr(cfg, "sample_steps")),
        "delay_steps": int(getattr(cfg, "delay_steps")),
        "probe_steps": int(getattr(cfg, "probe_steps")),
        "early_window_ms": int(getattr(cfg, "early_window_ms")),
        "perturbation_mode": str(getattr(cfg, "perturbation_mode")),
        "perturbation_attenuation_factor": float(getattr(cfg, "perturbation_attenuation_factor")),
        "fig5d_include_balanced": bool(getattr(cfg, "fig5d_include_balanced")),
        "enable_branch_batch": bool(getattr(cfg, "enable_branch_batch")),
        "smoke": bool(getattr(cfg, "smoke", False)),
        "trial_chunk_size": int(getattr(cfg, "batch_size")),
        "snapshot_shard_strategy": "trial_chunk",
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
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "build_probe_stsp_update_bank_cache_key",
    "build_support_bank_cache_key",
    "build_trial_sampling_cache_key",
    "cache_key_digest",
    "dataframe_hash",
    "model_fingerprint",
    "sha256_file",
    "table_digest",
    "trials_hash",
]
