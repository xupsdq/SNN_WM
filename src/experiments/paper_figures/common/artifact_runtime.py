from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

import numpy as np


CACHE_KEY_FILE = "cache_key.json"
REUSE_MODES = ("off", "auto", "require", "force")

ArtifactT = TypeVar("ArtifactT")


def normalize_reuse_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in REUSE_MODES:
        choices = ", ".join(REUSE_MODES)
        raise ValueError(f"Unsupported reuse-artifacts mode: {value!r}. Expected one of: {choices}")
    return mode


def default_artifact_root(seed_dir: str | Path) -> Path:
    return Path(seed_dir) / "data" / "intermediates"


def task_artifact_dir(artifact_root: str | Path, task_id: str) -> Path:
    return Path(artifact_root) / str(task_id)


def reset_task_artifact_dir(path: str | Path) -> None:
    task_dir = Path(path)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)


def cache_key_digest(cache_key: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _json_safe(cache_key),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_cache_key(task_dir: str | Path, cache_key: Mapping[str, Any]) -> None:
    path = Path(task_dir) / CACHE_KEY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": _json_safe(cache_key),
        "cache_key_digest": cache_key_digest(cache_key),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def read_cache_key(task_dir: str | Path) -> dict[str, Any]:
    path = Path(task_dir) / CACHE_KEY_FILE
    if not path.exists():
        raise FileNotFoundError(f"Artifact cache key is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "cache_key" not in payload or "cache_key_digest" not in payload:
        raise ValueError(f"Malformed artifact cache key file: {path}")
    return payload


def cache_key_matches(task_dir: str | Path, expected_key: Mapping[str, Any]) -> bool:
    try:
        payload = read_cache_key(task_dir)
    except (FileNotFoundError, ValueError):
        return False
    return str(payload.get("cache_key_digest")) == cache_key_digest(expected_key)


def require_cache_key_match(
    task_dir: str | Path,
    expected_key: Mapping[str, Any],
    *,
    task_id: str,
    mismatch_hint: str = "Use --reuse-artifacts force to rebuild.",
) -> None:
    payload = read_cache_key(task_dir)
    expected_digest = cache_key_digest(expected_key)
    found_digest = str(payload.get("cache_key_digest"))
    if found_digest != expected_digest:
        raise RuntimeError(
            f"{task_id} artifact cache key mismatch: expected {expected_digest}, found {found_digest}. "
            f"{mismatch_hint}"
        )


def validate_cache_key_integrity(
    task_dir: str | Path,
    *,
    task_id: str | None = None,
) -> dict[str, Any]:
    payload = read_cache_key(task_dir)
    cache_key = payload["cache_key"]
    cache_path = Path(task_dir) / CACHE_KEY_FILE
    if not isinstance(cache_key, dict):
        raise ValueError(f"Malformed artifact cache key payload: {cache_path}")
    found_digest = str(payload.get("cache_key_digest"))
    computed_digest = cache_key_digest(cache_key)
    if found_digest != computed_digest:
        raise RuntimeError(
            f"Artifact cache key digest mismatch: stored {found_digest}, computed {computed_digest}: "
            f"{cache_path}"
        )
    if task_id is not None and str(cache_key.get("task_id")) != str(task_id):
        raise RuntimeError(
            f"Artifact task id mismatch: expected {task_id!r}, found {cache_key.get('task_id')!r}: "
            f"{cache_path}"
        )
    return cache_key


def materialize_artifact(
    *,
    mode: str,
    task_dir: str | Path,
    expected_key: Mapping[str, Any],
    load: Callable[[], ArtifactT],
    build: Callable[[], ArtifactT],
    fresh: Callable[[], ArtifactT] | None = None,
    force_load_existing: bool = False,
    cache_mismatch_hint: str = "Use --reuse-artifacts force to rebuild.",
    recover_auto_load_errors: bool = True,
    cache_is_reusable: Callable[[], bool] | None = None,
    require_reusable: Callable[[], None] | None = None,
) -> ArtifactT:
    normalized_mode = normalize_reuse_mode(mode)
    task_dir = Path(task_dir)
    task_id = str(expected_key.get("task_id", task_dir.name))

    def load_checked() -> ArtifactT:
        if require_reusable is not None:
            require_reusable()
        else:
            require_cache_key_match(
                task_dir,
                expected_key,
                task_id=task_id,
                mismatch_hint=cache_mismatch_hint,
            )
        return load()

    if normalized_mode == "off":
        return fresh() if fresh is not None else build()
    if normalized_mode == "require":
        return load_checked()
    if normalized_mode == "auto":
        reusable = cache_is_reusable() if cache_is_reusable is not None else cache_key_matches(task_dir, expected_key)
        if reusable:
            if not recover_auto_load_errors:
                return load_checked()
            try:
                return load_checked()
            except Exception:
                pass
        return build()
    if force_load_existing and task_dir.exists():
        return load_checked()
    return build()


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
    "CACHE_KEY_FILE",
    "REUSE_MODES",
    "cache_key_digest",
    "cache_key_matches",
    "default_artifact_root",
    "materialize_artifact",
    "normalize_reuse_mode",
    "read_cache_key",
    "require_cache_key_match",
    "reset_task_artifact_dir",
    "task_artifact_dir",
    "validate_cache_key_integrity",
    "write_cache_key",
]
