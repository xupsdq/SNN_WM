from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.paper_figures.common.specs.cache_keys import (
    cache_key_digest,
    spec_digest,
    table_digest,
)
from src.experiments.paper_figures.common.specs.schemas import (
    ARTIFACT_DIGEST_FILE,
    CACHE_KEY_FILE,
    MANIFEST_COLUMNS,
    MANIFEST_FILE,
    PROVENANCE_REQUIRED_KEYS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SPEC_PROVENANCE_FILE,
    SPEC_VIEW_LINK_FILE,
)
from src.experiments.paper_figures.common.specs.types import SpecArtifact, SpecProvenance, SpecViewLink


def save_spec_artifact(
    task_dir: str | Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    files: Mapping[str, str],
    cache_key: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> SpecArtifact:
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    _require_same_keys("tables/files", tables, files)

    persisted: dict[str, pd.DataFrame] = {}
    rows = []
    for name, filename in files.items():
        df = tables[name].reset_index(drop=True).copy()
        path = task_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        read_back = pd.read_csv(path) if list(df.columns) else pd.DataFrame()
        persisted[name] = read_back
        rows.append(_manifest_row(name, filename, path, read_back))

    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    table_digest_value = table_digest(persisted)
    manifest["table_digest"] = table_digest_value
    manifest.to_csv(task_dir / MANIFEST_FILE, index=False, encoding="utf-8")
    normalized_provenance = _normalize_provenance(provenance)
    _validate_provenance_payload(normalized_provenance)
    _validate_row_identity(persisted, normalized_provenance)
    _write_json(normalized_provenance, task_dir / SPEC_PROVENANCE_FILE)

    key_digest = cache_key_digest(cache_key)
    _write_json({"cache_key": _json_safe(cache_key), "cache_key_digest": key_digest}, task_dir / CACHE_KEY_FILE)
    artifact_digest = _artifact_digest(manifest, normalized_provenance)
    _write_json(
        {
            "artifact_digest": artifact_digest,
            "table_digest": table_digest_value,
            "provenance_digest": spec_digest(normalized_provenance),
        },
        task_dir / ARTIFACT_DIGEST_FILE,
    )
    return SpecArtifact(
        root=task_dir,
        tables=persisted,
        manifest=manifest,
        provenance=SpecProvenance(normalized_provenance),
        cache_key=dict(cache_key),
        cache_key_digest=key_digest,
        table_digest=table_digest_value,
        artifact_digest=artifact_digest,
    )


def load_spec_artifact(
    task_dir: str | Path,
    *,
    files: Mapping[str, str] | None = None,
    expected_key: Mapping[str, Any] | None = None,
) -> SpecArtifact:
    task_dir = Path(task_dir)
    artifact = validate_spec_artifact(task_dir, files=files, expected_key=expected_key)
    return artifact


def validate_spec_artifact(
    task_dir: str | Path,
    *,
    files: Mapping[str, str] | None = None,
    expected_key: Mapping[str, Any] | None = None,
) -> SpecArtifact:
    task_dir = Path(task_dir)
    manifest_path = task_dir / MANIFEST_FILE
    provenance_path = task_dir / SPEC_PROVENANCE_FILE
    cache_key_path = task_dir / CACHE_KEY_FILE
    artifact_digest_path = task_dir / ARTIFACT_DIGEST_FILE
    for path in (manifest_path, provenance_path, cache_key_path, artifact_digest_path):
        if not path.exists():
            raise FileNotFoundError(f"Spec artifact required file is missing: {path}")

    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, MANIFEST_COLUMNS, manifest_path)
    cache_payload = _read_json(cache_key_path)
    if not isinstance(cache_payload, dict) or "cache_key" not in cache_payload or "cache_key_digest" not in cache_payload:
        raise ValueError(f"Malformed spec cache key file: {cache_key_path}")
    if expected_key is not None:
        expected_digest = cache_key_digest(expected_key)
        found_digest = str(cache_payload["cache_key_digest"])
        if found_digest != expected_digest:
            raise ValueError(f"Spec artifact cache key mismatch for {task_dir}: expected {expected_digest}, found {found_digest}")

    expected_files = files or {str(row["name"]): str(row["filename"]) for row in manifest.to_dict("records")}
    tables: dict[str, pd.DataFrame] = {}
    found_names = set(str(value) for value in manifest["name"].tolist())
    if set(expected_files) != found_names:
        raise ValueError(f"Spec manifest names mismatch in {manifest_path}: expected={sorted(expected_files)}, found={sorted(found_names)}")
    for name, filename in expected_files.items():
        row = _manifest_record(manifest, name, filename, manifest_path)
        path = task_dir / filename
        _require_file_hash(path, str(row["sha256"]))
        columns = _parse_columns(row["columns"], manifest_path)
        df = pd.read_csv(path) if columns else pd.DataFrame()
        if list(df.columns) != columns:
            raise ValueError(f"Spec table column mismatch for {path}: expected={columns}, found={list(df.columns)}")
        if len(df) != int(row["rows"]):
            raise ValueError(f"Spec table row count mismatch for {path}: expected={row['rows']}, found={len(df)}")
        tables[name] = df
    manifest_table_digests = {str(value) for value in manifest["table_digest"].astype(str).tolist()}
    computed_table_digest = table_digest(tables)
    if manifest_table_digests != {computed_table_digest}:
        raise ValueError(
            f"Spec table digest mismatch in {manifest_path}: manifest={sorted(manifest_table_digests)}, computed={computed_table_digest}"
        )

    provenance = _read_json(provenance_path)
    _validate_provenance_payload(provenance)
    _validate_parent_specs(provenance)
    _validate_row_identity(tables, provenance)
    computed_artifact_digest = _artifact_digest(manifest, provenance)
    recorded = _read_json(artifact_digest_path)
    if str(recorded.get("artifact_digest")) != computed_artifact_digest:
        raise ValueError(
            f"Spec artifact digest mismatch for {task_dir}: expected {recorded.get('artifact_digest')}, "
            f"found {computed_artifact_digest}"
        )
    return SpecArtifact(
        root=task_dir,
        tables=tables,
        manifest=manifest,
        provenance=SpecProvenance(provenance),
        cache_key=dict(cache_payload["cache_key"]),
        cache_key_digest=str(cache_payload["cache_key_digest"]),
        table_digest=computed_table_digest,
        artifact_digest=computed_artifact_digest,
    )


def materialize_spec_view(
    source: SpecArtifact,
    view_dir: str | Path,
    *,
    view_figure: str,
    view_task: str,
    view_artifact_digest: str,
    view_cache_key_digest: str,
) -> SpecViewLink:
    view_dir = Path(view_dir)
    view_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(SCHEMA_VERSION),
        "source_spec_root": str(Path(source.root).resolve()),
        "source_spec_family": str(source.provenance.payload.get("spec_family", "")),
        "source_producer_task": str(source.provenance.payload.get("producer_task", "")),
        "source_producer_figure": str(source.provenance.payload.get("producer_figure", "")),
        "source_shared_root": str(source.provenance.payload.get("shared_root", "")),
        "source_artifact_digest": str(source.artifact_digest),
        "source_cache_key_digest": str(source.cache_key_digest),
        "view_figure": str(view_figure),
        "view_task": str(view_task),
        "view_artifact_digest": str(view_artifact_digest),
        "view_cache_key_digest": str(view_cache_key_digest),
    }
    _write_json(payload, view_dir / SPEC_VIEW_LINK_FILE)
    return SpecViewLink(root=view_dir, payload=payload)


def copy_spec_artifact(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Cannot copy missing spec artifact: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _manifest_row(name: str, filename: str, path: Path, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "name": str(name),
        "filename": str(filename),
        "rows": int(len(df)),
        "columns": json.dumps([str(col) for col in df.columns], ensure_ascii=False, separators=(",", ":")),
        "sha256": _sha256_file(path),
        "table_digest": "",
    }


def _artifact_digest(manifest: pd.DataFrame, provenance: Mapping[str, Any]) -> str:
    manifest_payload = manifest.loc[:, list(MANIFEST_COLUMNS)].copy()
    return spec_digest(
        {
            "manifest_digest": table_digest({"manifest": manifest_payload}),
            "provenance_digest": spec_digest(provenance),
        }
    )


def _normalize_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_safe(dict(provenance))
    payload.setdefault("schema_name", SCHEMA_NAME)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("model_fingerprint", {})
    payload.setdefault("parent_spec_digests", {})
    payload.setdefault("row_identity_columns", {})
    return payload


def _validate_provenance_payload(payload: Mapping[str, Any]) -> None:
    missing = [key for key in PROVENANCE_REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Spec provenance missing required keys: {missing}")
    if "producer_figure" not in payload and "shared_root" not in payload:
        raise ValueError("Spec provenance must include producer_figure or shared_root")
    if not isinstance(payload.get("consumer_figures"), list):
        raise ValueError("Spec provenance consumer_figures must be a list")
    if not isinstance(payload.get("consumer_tasks"), list):
        raise ValueError("Spec provenance consumer_tasks must be a list")
    if not isinstance(payload.get("sampling_config"), Mapping):
        raise ValueError("Spec provenance sampling_config must be an object")
    if not isinstance(payload.get("rng_policy"), Mapping):
        raise ValueError("Spec provenance rng_policy must be an object")
    if not isinstance(payload.get("row_identity_columns"), Mapping):
        raise ValueError("Spec provenance row_identity_columns must be an object")


def _validate_parent_specs(provenance: Mapping[str, Any]) -> None:
    parents = provenance.get("parent_spec_digests", {})
    if isinstance(parents, Mapping):
        iterable = parents.values()
    elif isinstance(parents, list):
        iterable = parents
    else:
        raise ValueError("Spec provenance parent_spec_digests must be an object or list")
    for item in iterable:
        if not isinstance(item, Mapping):
            continue
        artifact_dir = item.get("artifact_dir") or item.get("path")
        expected = item.get("artifact_digest") or item.get("digest")
        if not artifact_dir or not expected:
            continue
        parent_digest_path = Path(str(artifact_dir)) / ARTIFACT_DIGEST_FILE
        if not parent_digest_path.exists():
            raise FileNotFoundError(f"Parent spec digest target is missing: {parent_digest_path}")
        found = _read_json(parent_digest_path).get("artifact_digest")
        if str(found) != str(expected):
            raise ValueError(f"Parent spec digest mismatch for {parent_digest_path}: expected {expected}, found {found}")


def _validate_row_identity(tables: Mapping[str, pd.DataFrame], provenance: Mapping[str, Any]) -> None:
    identity_map = provenance.get("row_identity_columns", {})
    if not isinstance(identity_map, Mapping):
        raise ValueError("row_identity_columns must be an object")
    for name, columns_value in identity_map.items():
        if name not in tables:
            raise ValueError(f"row_identity_columns references unknown table: {name}")
        columns = [str(col) for col in columns_value]
        df = tables[name]
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Spec table {name} missing row identity columns: {missing}")
        if columns and df.duplicated(subset=columns).any():
            dupes = int(df.duplicated(subset=columns).sum())
            raise ValueError(f"Spec table {name} has duplicate row identities over {columns}: {dupes}")


def _require_same_keys(label: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    if set(left) != set(right):
        raise KeyError(f"{label} key mismatch: left={sorted(left)}, right={sorted(right)}")


def _manifest_record(manifest: pd.DataFrame, name: str, filename: str, manifest_path: Path) -> dict[str, Any]:
    rows = manifest[manifest["name"].astype(str).eq(str(name)) & manifest["filename"].astype(str).eq(str(filename))]
    if len(rows) != 1:
        raise ValueError(f"Expected one spec manifest row for {name}/{filename}, found {len(rows)}: {manifest_path}")
    return rows.iloc[0].to_dict()


def _parse_columns(value: Any, manifest_path: Path) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed manifest columns JSON in {manifest_path}: {value!r}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Manifest columns must be a JSON list of strings in {manifest_path}: {value!r}")
    return parsed


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Spec manifest missing columns {missing} in {path}")


def _require_file_hash(path: Path, expected_sha: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Spec artifact table file is missing: {path}")
    found = _sha256_file(path)
    if found != expected_sha:
        raise ValueError(f"Spec artifact hash mismatch for {path}: expected {expected_sha}, found {found}")


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    "copy_spec_artifact",
    "load_spec_artifact",
    "materialize_spec_view",
    "save_spec_artifact",
    "validate_spec_artifact",
]
