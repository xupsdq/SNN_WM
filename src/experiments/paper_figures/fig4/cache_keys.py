from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.paper_figures.common.artifact_runtime import cache_key_digest
from src.experiments.paper_figures.fig4.schemas import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TASK_PAIR_SAMPLING,
    TASK_ROLLOUTS,
    TASK_SIMILARITY_ENTRY,
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


def mask_bank_digest(mask_bank: Mapping[int, Mapping[str, np.ndarray]]) -> str:
    hasher = hashlib.sha256()
    for pair_id in sorted(int(key) for key in mask_bank):
        for mask_name in sorted(str(key) for key in mask_bank[pair_id]):
            arr = np.asarray(mask_bank[pair_id][mask_name], dtype=bool)
            hasher.update(str(pair_id).encode("utf-8"))
            hasher.update(b"\n")
            hasher.update(mask_name.encode("utf-8"))
            hasher.update(b"\n")
            hasher.update(",".join(str(int(v)) for v in arr.shape).encode("utf-8"))
            hasher.update(b"\n")
            hasher.update(np.ascontiguousarray(arr).tobytes())
            hasher.update(b"\n")
    return hasher.hexdigest()


def pair_sampling_hash(
    pair_trials: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    overlap_matched_pairs: pd.DataFrame,
    perturbation_masks: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(
        table_digest(
            {
                "overlap_matched_pairs": overlap_matched_pairs,
                "pair_candidate_pool": candidate_pool,
                "pair_trials": pair_trials,
                "perturbation_masks": perturbation_masks,
            }
        ).encode("utf-8")
    )
    hasher.update(b"\n")
    hasher.update(mask_bank_digest(mask_bank).encode("utf-8"))
    return hasher.hexdigest()


def build_pair_sampling_cache_key(cfg: Any) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_PAIR_SAMPLING,
        "network_seed": int(getattr(cfg, "network_seed")),
        "dataset_root": str(Path(getattr(cfg, "dataset_root")).resolve()),
        "dataset_split": str(getattr(cfg, "split")),
        "model": model_fingerprint(getattr(cfg, "model_path")),
        "dt": float(getattr(cfg, "dt")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "probe_ms": int(getattr(cfg, "probe_ms")),
        "sample_steps": int(getattr(cfg, "sample_steps")),
        "probe_steps": int(getattr(cfg, "probe_steps")),
        "max_pairs": int(getattr(cfg, "max_pairs")),
        "num_similarity_bins": int(getattr(cfg, "num_similarity_bins")),
        "num_overlap_bins": int(getattr(cfg, "num_overlap_bins")),
        "overlap_mask_mode": str(getattr(cfg, "overlap_mask_mode")),
        "foreground_threshold": float(getattr(cfg, "foreground_threshold")),
        "dilation_radius": int(getattr(cfg, "dilation_radius")),
        "random_mask_candidates": int(getattr(cfg, "random_mask_candidates")),
        "require_distinct_pair_labels": bool(getattr(cfg, "require_distinct_pair_labels")),
        "smoke": bool(getattr(cfg, "smoke", False)),
    }


def build_similarity_entry_cache_key(cfg: Any, *, pair_hash: str) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_SIMILARITY_ENTRY,
        "network_seed": int(getattr(cfg, "network_seed")),
        "pair_sampling_hash": str(pair_hash),
        "model": model_fingerprint(getattr(cfg, "model_path")),
        "dt": float(getattr(cfg, "dt")),
        "batch_size": int(getattr(cfg, "batch_size")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "delay_ms": int(getattr(cfg, "delay_ms")),
        "probe_ms": int(getattr(cfg, "probe_ms")),
        "sample_steps": int(getattr(cfg, "sample_steps")),
        "delay_steps": int(getattr(cfg, "delay_steps")),
        "probe_steps": int(getattr(cfg, "probe_steps")),
        "legacy_exact_mode": bool(getattr(cfg, "legacy_exact_mode")),
        "readout": "layer3_v_mem_top_m_mean_m1",
        "smoke": bool(getattr(cfg, "smoke", False)),
    }


def build_rollouts_cache_key(cfg: Any, *, pair_hash: str) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_ROLLOUTS,
        "network_seed": int(getattr(cfg, "network_seed")),
        "pair_sampling_hash": str(pair_hash),
        "model": model_fingerprint(getattr(cfg, "model_path")),
        "dt": float(getattr(cfg, "dt")),
        "batch_size": int(getattr(cfg, "batch_size")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "delay_ms": int(getattr(cfg, "delay_ms")),
        "probe_ms": int(getattr(cfg, "probe_ms")),
        "sample_steps": int(getattr(cfg, "sample_steps")),
        "delay_steps": int(getattr(cfg, "delay_steps")),
        "probe_steps": int(getattr(cfg, "probe_steps")),
        "conditions": [
            "full_dynamic",
            "full_static",
            "sample_keep_overlap_only_dynamic",
            "sample_keep_nonoverlap_only_dynamic",
            "sample_random_matched_dynamic",
        ],
        "l3_replay_capture_version": 1,
        "l3_replay_capture_conditions": ["full_dynamic", "full_static"],
        "save_l3_trace": bool(getattr(cfg, "save_l3_trace")),
        "save_full_trace": bool(getattr(cfg, "save_full_trace")),
        "legacy_exact_mode": bool(getattr(cfg, "legacy_exact_mode")),
        "overlap_mask_mode": str(getattr(cfg, "overlap_mask_mode")),
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
    "build_pair_sampling_cache_key",
    "build_rollouts_cache_key",
    "build_similarity_entry_cache_key",
    "cache_key_digest",
    "dataframe_hash",
    "mask_bank_digest",
    "model_fingerprint",
    "pair_sampling_hash",
    "sha256_file",
    "table_digest",
]
