from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.experiments.paper_figures.fig2.schemas import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TASK_COMPLETION_DELAY_BOUNDARY_BANK,
    TASK_COMPLETION_DELAY_MASK_SPECS,
    TASK_PAIR_TRIAL_SPECS,
    TASK_PARTIAL_CUE_MASK_SPECS,
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


def pair_specs_hash(pair_trials: pd.DataFrame) -> str:
    return table_digest({"pair_trials": pair_trials})


def build_pair_specs_cache_key(cfg: Any) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": TASK_PAIR_TRIAL_SPECS,
        "network_seed": int(getattr(cfg, "network_seed")),
        "dataset_root": str(Path(getattr(cfg, "dataset_root")).resolve()),
        "dataset_split": str(getattr(cfg, "split")),
        "num_pairs": int(getattr(cfg, "num_pairs")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "delay1_ms": int(getattr(cfg, "delay1_ms")),
        "second_item_ms": int(getattr(cfg, "second_item_ms")),
        "delay2_ms": int(getattr(cfg, "delay2_ms")),
        "smoke": bool(getattr(cfg, "smoke", False)),
    }


def build_state_bank_cache_key(cfg: Any, *, pair_hash: str) -> dict[str, Any]:
    return _base_model_key(
        cfg,
        task_id=TASK_STATE_BANK,
        pair_hash=pair_hash,
        extra={
            "boundary_state": "full_boundary_all_layers",
            "state_conditions": ["S0", "S_A", "S_B", "S_AB"],
            "state_variables": ["u", "x", "g"],
            "episode_end_step": int(
                getattr(cfg, "sample_steps")
                + getattr(cfg, "delay1_steps")
                + getattr(cfg, "second_item_steps")
                + getattr(cfg, "delay2_steps")
            ),
        },
    )


def build_partial_cue_mask_cache_key(cfg: Any, *, pair_hash: str) -> dict[str, Any]:
    return _base_config_key(
        cfg,
        task_id=TASK_PARTIAL_CUE_MASK_SPECS,
        pair_hash=pair_hash,
        extra={
            "weak_probe_ms": int(getattr(cfg, "weak_probe_ms")),
            "weak_probe_steps": int(getattr(cfg, "weak_probe_steps")),
            "weak_probe_keep_probs": [float(v) for v in getattr(cfg, "weak_probe_keep_probs")],
            "weak_probe_repeats": int(getattr(cfg, "weak_probe_repeats")),
            "weak_probe_mask_space": str(getattr(cfg, "weak_probe_mask_space")),
            "weak_probe_use_same_mask_across_states": bool(getattr(cfg, "weak_probe_use_same_mask_across_states")),
            "weak_probe_scale": float(getattr(cfg, "weak_probe_scale")),
            "weak_probe_noise": float(getattr(cfg, "weak_probe_noise")),
            "foreground_threshold": float(getattr(cfg, "foreground_threshold")),
            "mask_seed_offset": 404,
        },
    )


def build_completion_boundary_cache_key(cfg: Any, *, pair_hash: str) -> dict[str, Any]:
    return _base_model_key(
        cfg,
        task_id=TASK_COMPLETION_DELAY_BOUNDARY_BANK,
        pair_hash=pair_hash,
        extra={
            "boundary_state": "full_boundary_all_layers",
            "state_conditions": ["S0", "S_B", "S_AB"],
            "completion_delay_sweep_ms": [int(v) for v in getattr(cfg, "completion_delay_sweep_ms")],
            "sample_ms": int(getattr(cfg, "sample_ms")),
            "delay1_ms": int(getattr(cfg, "delay1_ms")),
            "second_item_ms": int(getattr(cfg, "second_item_ms")),
        },
    )


def build_completion_mask_cache_key(cfg: Any, *, pair_hash: str) -> dict[str, Any]:
    return _base_config_key(
        cfg,
        task_id=TASK_COMPLETION_DELAY_MASK_SPECS,
        pair_hash=pair_hash,
        extra={
            "completion_delay_sweep_ms": [int(v) for v in getattr(cfg, "completion_delay_sweep_ms")],
            "completion_delay_keep_prob": float(getattr(cfg, "completion_delay_keep_prob")),
            "completion_delay_repeats": int(getattr(cfg, "completion_delay_repeats")),
            "weak_probe_ms": int(getattr(cfg, "weak_probe_ms")),
            "weak_probe_steps": int(getattr(cfg, "weak_probe_steps")),
            "weak_probe_mask_space": str(getattr(cfg, "weak_probe_mask_space")),
            "weak_probe_scale": float(getattr(cfg, "weak_probe_scale")),
            "weak_probe_noise": float(getattr(cfg, "weak_probe_noise")),
            "foreground_threshold": float(getattr(cfg, "foreground_threshold")),
            "mask_seed_offset": 909,
        },
    )


def _base_model_key(cfg: Any, *, task_id: str, pair_hash: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    payload = _base_config_key(cfg, task_id=task_id, pair_hash=pair_hash, extra=extra)
    payload["model"] = model_fingerprint(getattr(cfg, "model_path"))
    return payload


def _base_config_key(cfg: Any, *, task_id: str, pair_hash: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "task_id": str(task_id),
        "network_seed": int(getattr(cfg, "network_seed")),
        "dataset_split": str(getattr(cfg, "split")),
        "pair_specs_hash": str(pair_hash),
        "dt": float(getattr(cfg, "dt")),
        "batch_size": int(getattr(cfg, "batch_size")),
        "sample_ms": int(getattr(cfg, "sample_ms")),
        "delay1_ms": int(getattr(cfg, "delay1_ms")),
        "second_item_ms": int(getattr(cfg, "second_item_ms")),
        "delay2_ms": int(getattr(cfg, "delay2_ms")),
        "extra": _json_safe(extra),
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
    "build_completion_boundary_cache_key",
    "build_completion_mask_cache_key",
    "build_pair_specs_cache_key",
    "build_partial_cue_mask_cache_key",
    "build_state_bank_cache_key",
    "cache_key_digest",
    "dataframe_hash",
    "model_fingerprint",
    "pair_specs_hash",
    "sha256_file",
    "table_digest",
]
