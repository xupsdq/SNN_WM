from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.paper_figures.common.artifact_runtime import (
    CACHE_KEY_FILE,
    default_artifact_root,
    task_artifact_dir,
    write_cache_key,
)
from src.experiments.paper_figures.common.sequence_root.cache_keys import (
    cache_key_digest,
    sequence_specs_hash,
    sha256_file,
    table_digest,
)
from src.experiments.paper_figures.common.sequence_root.schemas import (
    ROOT_BANK_MANIFEST_COLUMNS,
    SEQUENCE_SPEC_FILES,
    TABLE_MANIFEST_COLUMNS,
    TASK_SHARED_SEQUENCE_ROOT_BANK,
    TASK_SHARED_SEQUENCE_SPECS,
)
from src.experiments.paper_figures.common.sequence_root.types import (
    SharedSequenceRootBank,
    SharedSequenceSpecBundle,
)
from src.experiments.paper_figures.common.specs.artifacts import load_spec_artifact, save_spec_artifact
from src.experiments.paper_figures.common.specs.cache_keys import rng_namespace_seed
from src.experiments.paper_figures.common.specs.schemas import SCHEMA_VERSION as SPEC_SCHEMA_VERSION
from src.experiments.paper_figures.common.specs.schemas import SpecFamily


def resolve_root_bank_dir(path: str | Path) -> Path:
    root = Path(path)
    direct = root / CACHE_KEY_FILE
    nested = root / TASK_SHARED_SEQUENCE_ROOT_BANK / CACHE_KEY_FILE
    if direct.exists():
        return root
    if nested.exists():
        return root / TASK_SHARED_SEQUENCE_ROOT_BANK
    raise FileNotFoundError(
        f"Shared sequence root bank not found at {root}. "
        f"Expected {CACHE_KEY_FILE} or {TASK_SHARED_SEQUENCE_ROOT_BANK}/{CACHE_KEY_FILE}."
    )


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_cache_key(task_dir: Path) -> dict[str, Any]:
    path = Path(task_dir) / CACHE_KEY_FILE
    if not path.exists():
        raise FileNotFoundError(f"Shared sequence-root cache key is missing: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict) or "cache_key" not in payload or "cache_key_digest" not in payload:
        raise ValueError(f"Malformed shared sequence-root cache key file: {path}")
    return payload


def cache_key_matches(task_dir: Path, expected_key: Mapping[str, Any]) -> bool:
    try:
        payload = read_cache_key(task_dir)
    except Exception:
        return False
    return str(payload.get("cache_key_digest")) == cache_key_digest(expected_key)


def require_cache_key_match(task_dir: Path, expected_key: Mapping[str, Any], *, task_id: str) -> None:
    payload = read_cache_key(task_dir)
    expected = cache_key_digest(expected_key)
    found = str(payload.get("cache_key_digest"))
    if found != expected:
        raise ValueError(
            f"Shared sequence-root artifact cache key mismatch for {task_id}: expected {expected}, found {found}. "
            "Rebuild the shared producer before using --reuse-artifacts require."
        )


def save_sequence_specs_artifact(
    task_dir: Path,
    *,
    sequence_trials: pd.DataFrame,
    singleton_reference_trials: pd.DataFrame,
    partial_cue_trials: pd.DataFrame,
    cache_key: Mapping[str, Any],
) -> SharedSequenceSpecBundle:
    task_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "sequence_trials": sequence_trials.reset_index(drop=True).copy(),
        "singleton_reference_trials": singleton_reference_trials.reset_index(drop=True).copy(),
        "partial_cue_trials": partial_cue_trials.reset_index(drop=True).copy(),
    }
    provenance = _shared_sequence_spec_provenance(cache_key)
    spec_artifact = save_spec_artifact(
        task_dir,
        tables=tables,
        files=SEQUENCE_SPEC_FILES,
        cache_key=cache_key,
        provenance=provenance,
    )
    persisted = spec_artifact.tables
    manifest = spec_artifact.manifest
    digest = spec_artifact.artifact_digest
    return SharedSequenceSpecBundle(
        root=task_dir,
        sequence_trials=persisted["sequence_trials"],
        singleton_reference_trials=persisted["singleton_reference_trials"],
        partial_cue_trials=persisted["partial_cue_trials"],
        manifest=manifest,
        digest=digest,
        spec_artifact=spec_artifact,
    )


def load_sequence_specs_artifact(task_dir: Path, *, expected_key: Mapping[str, Any] | None = None) -> SharedSequenceSpecBundle:
    task_dir = Path(task_dir)
    spec_artifact = load_spec_artifact(task_dir, files=SEQUENCE_SPEC_FILES, expected_key=expected_key)
    tables = spec_artifact.tables
    manifest = spec_artifact.manifest
    digest = spec_artifact.artifact_digest
    return SharedSequenceSpecBundle(
        root=task_dir,
        sequence_trials=tables["sequence_trials"],
        singleton_reference_trials=tables["singleton_reference_trials"],
        partial_cue_trials=tables["partial_cue_trials"],
        manifest=manifest,
        digest=digest,
        spec_artifact=spec_artifact,
    )


def save_root_bank_artifact(
    task_dir: Path,
    *,
    specs: SharedSequenceSpecBundle,
    fig3_state_bank_dir: Path,
    fig6_sequence_bank_dir: Path,
    cache_key: Mapping[str, Any],
    fig3_digest: str,
    fig6_digest: str,
    fig3_cache_key_digest: str,
    fig6_cache_key_digest: str,
) -> SharedSequenceRootBank:
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    specs_dst = task_dir / TASK_SHARED_SEQUENCE_SPECS
    fig3_dst = task_dir / "fig3_state_bank"
    fig6_dst = task_dir / "fig6_sequence_bank"
    _copy_tree(Path(specs.root), specs_dst)
    _copy_tree(Path(fig3_state_bank_dir), fig3_dst)
    _copy_tree(Path(fig6_sequence_bank_dir), fig6_dst)
    rows = [
        _root_manifest_row("shared_sequence_specs", specs_dst, task_dir, cache_key_digest(read_cache_key(specs_dst)["cache_key"]), specs.digest),
        _root_manifest_row("fig3_state_bank", fig3_dst, task_dir, fig3_cache_key_digest, fig3_digest),
        _root_manifest_row("fig6_sequence_bank", fig6_dst, task_dir, fig6_cache_key_digest, fig6_digest),
    ]
    manifest = pd.DataFrame(rows, columns=list(ROOT_BANK_MANIFEST_COLUMNS))
    manifest.to_csv(task_dir / "manifest.csv", index=False, encoding="utf-8")
    digest = table_digest({"manifest": manifest})
    write_json({"artifact_digest": digest}, task_dir / "artifact_digest.json")
    write_cache_key(task_dir, cache_key)
    return SharedSequenceRootBank(task_dir, specs, fig3_dst, fig6_dst, manifest, digest)


def load_root_bank_artifact(task_dir: Path, *, expected_key: Mapping[str, Any] | None = None) -> SharedSequenceRootBank:
    task_dir = resolve_root_bank_dir(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id=TASK_SHARED_SEQUENCE_ROOT_BANK)
    manifest_path = task_dir / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, ROOT_BANK_MANIFEST_COLUMNS, manifest_path)
    specs_dir = _manifest_path(manifest, "shared_sequence_specs", task_dir, manifest_path)
    fig3_dir = _manifest_path(manifest, "fig3_state_bank", task_dir, manifest_path)
    fig6_dir = _manifest_path(manifest, "fig6_sequence_bank", task_dir, manifest_path)
    specs = load_sequence_specs_artifact(specs_dir)
    digest = table_digest({"manifest": manifest})
    recorded = read_json(task_dir / "artifact_digest.json").get("artifact_digest")
    if str(recorded) != digest:
        raise ValueError(f"Shared sequence root bank digest mismatch: expected {recorded}, found {digest}")
    return SharedSequenceRootBank(task_dir, specs, fig3_dir, fig6_dir, manifest, digest)


def copy_artifact_tree(src: str | Path, dst: str | Path) -> None:
    _copy_tree(Path(src), Path(dst))


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Cannot copy missing shared sequence-root artifact directory: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _root_manifest_row(name: str, path: Path, root: Path, key_digest: str, artifact_digest: str) -> dict[str, Any]:
    return {
        "name": str(name),
        "relative_path": str(path.relative_to(root).as_posix()),
        "cache_key_digest": str(key_digest),
        "artifact_digest": str(artifact_digest),
    }


def _manifest_path(manifest: pd.DataFrame, name: str, root: Path, manifest_path: Path) -> Path:
    rows = manifest[manifest["name"].astype(str).eq(str(name))]
    if len(rows) != 1:
        raise ValueError(f"Expected one shared root manifest row named {name!r} in {manifest_path}, found {len(rows)}")
    return root / str(rows.iloc[0]["relative_path"])


def _table_manifest_row(name: str, filename: str, path: Path, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "name": name,
        "filename": filename,
        "rows": int(len(df)),
        "columns": json.dumps([str(col) for col in df.columns], ensure_ascii=False, separators=(",", ":")),
        "sha256": sha256_file(path),
        "table_digest": table_digest({name: df}),
    }


def _shared_sequence_spec_provenance(cache_key: Mapping[str, Any]) -> dict[str, Any]:
    sampling_config = {
        key: value
        for key, value in cache_key.items()
        if key
        not in {
            "schema_name",
            "schema_version",
            "task_id",
            "network_seed",
            "dataset_root",
            "dataset_split",
        }
    }
    network_seed = int(cache_key.get("network_seed"))
    producer_namespace = "common.sequence_root.shared_sequence_specs"
    return {
        "spec_family": SpecFamily.SEQUENCE.value,
        "producer_task": TASK_SHARED_SEQUENCE_SPECS,
        "shared_root": "fig3_fig6_sequence_root",
        "consumer_figures": ["fig3", "fig6"],
        "consumer_tasks": ["fig3.sequence_trial_specs", "fig6.sequence_trials"],
        "dataset_root": str(cache_key.get("dataset_root")),
        "dataset_split": str(cache_key.get("dataset_split")),
        "network_seed": network_seed,
        "model_fingerprint": {},
        "sampling_config": sampling_config,
        "rng_policy": {
            "mode": "legacy_network_seed",
            "seed": network_seed,
            "recommended_namespace_seed": rng_namespace_seed(
                network_seed,
                SpecFamily.SEQUENCE.value,
                producer_namespace,
                SPEC_SCHEMA_VERSION,
            ),
            "producer_namespace": producer_namespace,
        },
        "parent_spec_digests": {},
        "row_identity_columns": {
            "sequence_trials": ["sequence_id", "stage_k"],
            "singleton_reference_trials": ["sequence_id", "reference_position"],
            "partial_cue_trials": ["sequence_id"],
        },
    }


def _table_manifest_record(manifest: pd.DataFrame, name: str, filename: str, manifest_path: Path) -> dict[str, Any]:
    rows = manifest[manifest["name"].astype(str).eq(str(name)) & manifest["filename"].astype(str).eq(str(filename))]
    if len(rows) != 1:
        raise ValueError(f"Expected one shared spec manifest row for {name}/{filename}, found {len(rows)}: {manifest_path}")
    return rows.iloc[0].to_dict()


def _require_file_hash(path: Path, expected_sha: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Shared sequence-root required file is missing: {path}")
    found = sha256_file(path)
    if found != expected_sha:
        raise ValueError(f"Shared sequence-root hash mismatch for {path}: expected {expected_sha}, found {found}")


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Shared sequence-root manifest missing columns {missing} in {path}")


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
    "cache_key_matches",
    "copy_artifact_tree",
    "default_artifact_root",
    "load_root_bank_artifact",
    "load_sequence_specs_artifact",
    "read_cache_key",
    "resolve_root_bank_dir",
    "save_root_bank_artifact",
    "save_sequence_specs_artifact",
    "task_artifact_dir",
    "write_cache_key",
    "write_json",
]
