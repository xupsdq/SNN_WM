from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig2.artifacts import (
    cache_key_matches,
    read_json,
    require_cache_key_match,
    reset_task_artifact_dir,
    validate_cache_key_integrity,
    write_cache_key,
    write_json,
)
from src.experiments.paper_figures.fig2.cache_keys import cache_key_digest, dataframe_hash, sha256_file


MANIFEST_COLUMNS = (
    "kind",
    "name",
    "filename",
    "rows",
    "columns",
    "shape",
    "dtype",
    "sha256",
    "content_sha256",
)


@dataclass(frozen=True)
class FixedBArtifact:
    root: Path
    tables: dict[str, pd.DataFrame]
    arrays: dict[str, np.ndarray]
    payloads: dict[str, dict[str, Any]]
    manifest: pd.DataFrame
    digest: str


def array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(b"\n")
    hasher.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    hasher.update(b"\n")
    hasher.update(arr.tobytes(order="C"))
    return hasher.hexdigest()


def save_fixed_b_artifact(
    task_dir: Path,
    cache_key: Mapping[str, Any],
    *,
    tables: Mapping[str, pd.DataFrame] | None = None,
    arrays: Mapping[str, np.ndarray] | None = None,
    payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> FixedBArtifact:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    table_map = {str(name): table.copy() for name, table in (tables or {}).items()}
    array_map = {str(name): np.asarray(value) for name, value in (arrays or {}).items()}
    payload_map = {str(name): dict(value) for name, value in (payloads or {}).items()}
    manifest_rows: list[dict[str, Any]] = []

    for name, table in sorted(table_map.items()):
        filename = f"{name}.csv"
        path = task_dir / filename
        table.to_csv(path, index=False, lineterminator="\n")
        persisted = pd.read_csv(path, keep_default_na=False)
        manifest_rows.append(
            {
                "kind": "table",
                "name": name,
                "filename": filename,
                "rows": int(len(persisted)),
                "columns": json.dumps([str(col) for col in persisted.columns], separators=(",", ":")),
                "shape": "",
                "dtype": "",
                "sha256": sha256_file(path),
                "content_sha256": dataframe_hash(persisted),
            }
        )

    if array_map:
        for name, array in array_map.items():
            if array.dtype == object:
                raise TypeError(f"Object arrays are not allowed in fixed-B artifacts: {name}")
        arrays_path = task_dir / "arrays.npz"
        np.savez_compressed(arrays_path, **array_map)
        arrays_file_hash = sha256_file(arrays_path)
        for name, array in sorted(array_map.items()):
            manifest_rows.append(
                {
                    "kind": "array",
                    "name": name,
                    "filename": "arrays.npz",
                    "rows": int(array.shape[0]) if array.ndim else 1,
                    "columns": "",
                    "shape": "x".join(str(v) for v in array.shape),
                    "dtype": str(array.dtype),
                    "sha256": arrays_file_hash,
                    "content_sha256": array_hash(array),
                }
            )

    for name, payload in sorted(payload_map.items()):
        filename = f"{name}.json"
        path = task_dir / filename
        write_json(payload, path)
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        manifest_rows.append(
            {
                "kind": "json",
                "name": name,
                "filename": filename,
                "rows": 1,
                "columns": "",
                "shape": "",
                "dtype": "",
                "sha256": sha256_file(path),
                "content_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            }
        )

    manifest = pd.DataFrame(manifest_rows, columns=list(MANIFEST_COLUMNS))
    manifest.to_csv(task_dir / "manifest.csv", index=False, lineterminator="\n")
    write_cache_key(task_dir, cache_key)
    return load_fixed_b_artifact(task_dir, cache_key, task_id=str(cache_key.get("task_id", task_dir.name)))


def load_fixed_b_artifact(task_dir: Path, expected_key: Mapping[str, Any], *, task_id: str) -> FixedBArtifact:
    task_dir = Path(task_dir)
    require_cache_key_match(task_dir, expected_key, task_id=task_id)
    validate_cache_key_integrity(task_dir, task_id=task_id)
    manifest_path = task_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing fixed-B manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    missing = [col for col in MANIFEST_COLUMNS if col not in manifest.columns]
    if missing:
        raise ValueError(f"{manifest_path} is missing columns {missing}")
    if manifest.empty:
        raise RuntimeError(f"Fixed-B artifact manifest is empty: {manifest_path}")

    for filename, part in manifest.groupby("filename", sort=False):
        path = task_dir / str(filename)
        if not path.exists():
            raise FileNotFoundError(f"Missing fixed-B artifact file: {path}")
        expected_hashes = set(str(v) for v in part["sha256"])
        found = sha256_file(path)
        if expected_hashes != {found}:
            raise RuntimeError(f"Fixed-B artifact hash mismatch for {path}: expected={sorted(expected_hashes)}, found={found}")

    tables: dict[str, pd.DataFrame] = {}
    arrays: dict[str, np.ndarray] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for row in manifest.itertuples(index=False):
        kind = str(row.kind)
        name = str(row.name)
        path = task_dir / str(row.filename)
        if kind == "table":
            table = pd.read_csv(path, keep_default_na=False)
            expected_columns = json.loads(str(row.columns))
            if list(table.columns) != expected_columns:
                raise ValueError(f"Fixed-B table columns mismatch for {path}")
            if len(table) != int(row.rows):
                raise ValueError(f"Fixed-B table row-count mismatch for {path}")
            if dataframe_hash(table) != str(row.content_sha256):
                raise RuntimeError(f"Fixed-B table content hash mismatch for {path}")
            tables[name] = table
        elif kind == "json":
            payload = read_json(path)
            canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != str(row.content_sha256):
                raise RuntimeError(f"Fixed-B JSON content hash mismatch for {path}")
            payloads[name] = payload

    array_rows = manifest.loc[manifest["kind"].eq("array")]
    if not array_rows.empty:
        arrays_path = task_dir / str(array_rows.iloc[0]["filename"])
        with np.load(arrays_path, allow_pickle=False) as loaded:
            for row in array_rows.itertuples(index=False):
                name = str(row.name)
                if name not in loaded.files:
                    raise KeyError(f"Fixed-B array {name!r} missing from {arrays_path}")
                array = np.asarray(loaded[name])
                expected_shape = tuple(int(v) for v in str(row.shape).split("x") if str(v))
                if array.shape != expected_shape or str(array.dtype) != str(row.dtype):
                    raise ValueError(
                        f"Fixed-B array schema mismatch for {name}: expected {expected_shape}/{row.dtype}, "
                        f"found {array.shape}/{array.dtype}"
                    )
                if array_hash(array) != str(row.content_sha256):
                    raise RuntimeError(f"Fixed-B array content hash mismatch for {name} in {arrays_path}")
                arrays[name] = array

    return FixedBArtifact(
        root=task_dir,
        tables=tables,
        arrays=arrays,
        payloads=payloads,
        manifest=manifest,
        digest=cache_key_digest(expected_key),
    )


def artifact_exists_and_matches(task_dir: Path, expected_key: Mapping[str, Any]) -> bool:
    task_dir = Path(task_dir)
    return task_dir.exists() and cache_key_matches(task_dir, expected_key)


__all__ = [
    "FixedBArtifact",
    "artifact_exists_and_matches",
    "array_hash",
    "load_fixed_b_artifact",
    "save_fixed_b_artifact",
]
