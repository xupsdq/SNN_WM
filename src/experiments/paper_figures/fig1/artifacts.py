from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.experiments.common.ping_common import LAYER_KEYS
from src.experiments.paper_figures.common.artifact_runtime import (
    CACHE_KEY_FILE as _CACHE_KEY_FILE,
    cache_key_matches,
    default_artifact_root,
    read_cache_key,
    require_cache_key_match,
    reset_task_artifact_dir,
    task_artifact_dir,
    write_cache_key,
)
from src.experiments.paper_figures.fig1.cache_keys import sha256_file, trial_specs_hash
from src.experiments.paper_figures.fig1.schemas import (
    BOUNDARY_STATE_KEYS,
    DELAY_FEATURE_NPZ_KEYS,
    DMS_MANIFEST_COLUMNS,
    DMS_ROW_HASH_COLUMNS,
    TRIAL_SPEC_FILES,
    TRIAL_SPEC_MANIFEST_COLUMNS,
)


CACHE_KEY_FILE = _CACHE_KEY_FILE


@dataclass(frozen=True)
class TrialSpecsArtifact:
    root: Path
    specs: dict[str, pd.DataFrame]
    manifest: pd.DataFrame
    digest: str


@dataclass(frozen=True)
class DelayFeatureBank:
    root: Path
    features: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]]
    manifest: pd.DataFrame


@dataclass(frozen=True)
class DmsBoundaryBank:
    root: Path
    manifest: pd.DataFrame
    phase_rates: pd.DataFrame
    layer_input_shapes_by_batch: dict[int, dict[str, tuple[int, ...]]]

    def layer_input_shapes_for_batch(self, batch_id: int) -> dict[str, tuple[int, ...]]:
        batch_id = int(batch_id)
        if batch_id not in self.layer_input_shapes_by_batch:
            raise FileNotFoundError(f"DMS boundary bank is missing layer input shapes for batch_id={batch_id}")
        return self.layer_input_shapes_by_batch[batch_id]

    def load_boundary(self, batch_id: int, delay_ms: int) -> dict[str, dict[str, torch.Tensor]]:
        return load_boundary_shard(boundary_shard_path(self.root, int(batch_id), int(delay_ms)))

    def phase_rate_rows(self) -> list[dict[str, Any]]:
        if self.phase_rates.empty:
            return []
        return self.phase_rates.to_dict("records")


def save_trial_specs_artifact(
    task_dir: Path,
    specs: Mapping[str, pd.DataFrame],
    *,
    cache_key: Mapping[str, Any],
) -> TrialSpecsArtifact:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    missing = sorted(set(TRIAL_SPEC_FILES).difference(specs))
    if missing:
        raise KeyError(f"Cannot save trial_specs artifact; missing specs: {missing}")
    digest = trial_specs_hash({name: specs[name] for name in TRIAL_SPEC_FILES})
    rows: list[dict[str, Any]] = []
    for name, filename in TRIAL_SPEC_FILES.items():
        path = task_dir / filename
        df = specs[name].copy()
        df.to_csv(path, index=False, encoding="utf-8")
        rows.append(
            {
                "name": str(name),
                "filename": str(filename),
                "rows": int(len(df)),
                "columns": json.dumps([str(col) for col in df.columns], ensure_ascii=False, separators=(",", ":")),
                "sha256": sha256_file(path),
                "trial_specs_digest": digest,
            }
        )
    manifest = pd.DataFrame(rows, columns=list(TRIAL_SPEC_MANIFEST_COLUMNS))
    manifest.to_csv(task_dir / "manifest.csv", index=False, encoding="utf-8")
    write_cache_key(task_dir, cache_key)
    return TrialSpecsArtifact(task_dir, {name: specs[name].copy() for name in TRIAL_SPEC_FILES}, manifest, digest)


def load_trial_specs_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
) -> TrialSpecsArtifact:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id="trial_specs")
    manifest_path = task_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Trial specs manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    _validate_trial_specs_manifest_shape(manifest, manifest_path)
    specs: dict[str, pd.DataFrame] = {}
    for name, filename in TRIAL_SPEC_FILES.items():
        rec = _trial_manifest_record(manifest, name, filename, manifest_path)
        path = task_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Trial specs artifact file is missing: {path}")
        found_sha = sha256_file(path)
        expected_sha = str(rec["sha256"])
        if found_sha != expected_sha:
            raise RuntimeError(f"Trial specs file hash mismatch for {filename}: expected {expected_sha}, found {found_sha}")
        df = pd.read_csv(path)
        expected_rows = int(rec["rows"])
        if len(df) != expected_rows:
            raise RuntimeError(f"Trial specs row count mismatch for {filename}: expected {expected_rows}, found {len(df)}")
        expected_columns = _parse_manifest_columns(rec["columns"], manifest_path)
        if list(df.columns) != expected_columns:
            raise RuntimeError(
                f"Trial specs column mismatch for {filename}: expected {expected_columns}, found {list(df.columns)}"
            )
        specs[name] = df
    digest = trial_specs_hash(specs)
    manifest_digests = {str(value) for value in manifest["trial_specs_digest"].dropna().astype(str).tolist()}
    if manifest_digests != {digest}:
        raise RuntimeError(f"Trial specs digest mismatch: manifest={sorted(manifest_digests)}, computed={digest}")
    return TrialSpecsArtifact(task_dir, specs, manifest, digest)


def _validate_trial_specs_manifest_shape(manifest: pd.DataFrame, manifest_path: Path) -> None:
    missing_cols = [col for col in TRIAL_SPEC_MANIFEST_COLUMNS if col not in manifest.columns]
    if missing_cols:
        raise ValueError(f"Trial specs manifest is missing columns {missing_cols}: {manifest_path}")
    found_names = set(str(value) for value in manifest["name"].tolist())
    expected_names = set(TRIAL_SPEC_FILES)
    if found_names != expected_names:
        raise RuntimeError(
            f"Trial specs manifest names mismatch: expected={sorted(expected_names)}, found={sorted(found_names)}"
        )


def _trial_manifest_record(manifest: pd.DataFrame, name: str, filename: str, manifest_path: Path) -> dict[str, Any]:
    rows = manifest[
        manifest["name"].astype(str).eq(str(name))
        & manifest["filename"].astype(str).eq(str(filename))
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Trial specs manifest expected one row for {name}/{filename}, found {len(rows)}: {manifest_path}")
    return rows.iloc[0].to_dict()


def _parse_manifest_columns(value: Any, manifest_path: Path) -> list[str]:
    try:
        columns = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Trial specs manifest has malformed columns JSON in {manifest_path}: {value!r}") from exc
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise ValueError(f"Trial specs manifest columns must be a JSON string list in {manifest_path}: {value!r}")
    return columns


def save_delay_feature_bank(
    task_dir: Path,
    features: Mapping[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    cache_key: Mapping[str, Any],
) -> DelayFeatureBank:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    rows: list[dict[str, Any]] = []
    for (layer, delay_ms, set_name), (x, y, trial_ids) in sorted(features.items()):
        path = task_dir / f"{layer}_delay_{int(delay_ms)}ms_{set_name}.npz"
        np.savez_compressed(
            path,
            x=np.asarray(x, dtype=np.float32),
            y=np.asarray(y, dtype=np.int64),
            trial_ids=np.asarray(trial_ids, dtype=np.int64),
        )
        rows.append(
            {
                "layer": str(layer),
                "delay_ms": int(delay_ms),
                "set": str(set_name),
                "n_trials": int(np.asarray(x).shape[0]),
                "n_features": int(np.asarray(x).shape[1]) if np.asarray(x).ndim == 2 else 0,
                "path": path.relative_to(task_dir).as_posix(),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(task_dir / "manifest.csv", index=False, encoding="utf-8")
    write_cache_key(task_dir, cache_key)
    return DelayFeatureBank(task_dir, dict(features), manifest)


def load_delay_feature_bank(task_dir: Path, *, expected_key: Mapping[str, Any] | None = None) -> DelayFeatureBank:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id="delay_feature_bank")
    manifest_path = task_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Delay feature bank manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    features: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for rec in manifest.to_dict("records"):
        path = task_dir / str(rec["path"])
        with np.load(path, allow_pickle=False) as payload:
            keys = tuple(payload.files)
            if set(keys) != set(DELAY_FEATURE_NPZ_KEYS):
                raise ValueError(f"{path} has NPZ keys {keys}, expected {DELAY_FEATURE_NPZ_KEYS}")
            features[(str(rec["layer"]), int(rec["delay_ms"]), str(rec["set"]))] = (
                np.asarray(payload["x"], dtype=np.float32),
                np.asarray(payload["y"], dtype=np.int64),
                np.asarray(payload["trial_ids"], dtype=np.int64),
            )
    return DelayFeatureBank(task_dir, features, manifest)


def boundary_shard_path(task_dir: Path, batch_id: int, delay_ms: int) -> Path:
    return Path(task_dir) / "shards" / f"batch_{int(batch_id):04d}_delay_{int(delay_ms)}ms.npz"


def save_boundary_shard(path: Path, boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for layer_key in LAYER_KEYS:
        if layer_key not in boundary:
            continue
        for state_key in BOUNDARY_STATE_KEYS:
            if state_key not in boundary[layer_key]:
                continue
            tensor = boundary[layer_key][state_key]
            arrays[f"{layer_key}__{state_key}"] = tensor.detach().cpu().numpy()
    if not arrays:
        raise ValueError(f"No boundary state arrays to save for shard: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_boundary_shard(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DMS boundary shard is missing: {path}")
    boundary: dict[str, dict[str, torch.Tensor]] = {}
    with np.load(path, allow_pickle=False) as payload:
        for key in payload.files:
            if "__" not in key:
                raise ValueError(f"Malformed boundary shard key {key!r} in {path}")
            layer_key, state_key = key.split("__", 1)
            boundary.setdefault(layer_key, {})[state_key] = torch.from_numpy(np.array(payload[key]))
    return boundary


def dms_manifest_rows_for_batch(batch: pd.DataFrame, batch_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, rec in enumerate(batch.to_dict("records")):
        row = {
            "batch_id": int(batch_id),
            "row_index": int(row_index),
            "trial_id": int(rec["trial_id"]),
            "sample_image_id": int(rec["sample_image_id"]),
            "sample_label": int(rec["sample_label"]),
            "probe_image_id": int(rec["probe_image_id"]),
            "probe_label": int(rec["probe_label"]),
        }
        row["batch_row_hash"] = dms_batch_row_hash(row)
        rows.append(row)
    return rows


def expected_dms_manifest(dms_trials: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch_id, start in enumerate(range(0, len(dms_trials), int(batch_size))):
        batch = dms_trials.iloc[start : start + int(batch_size)].reset_index(drop=True)
        rows.extend(dms_manifest_rows_for_batch(batch, batch_id))
    return pd.DataFrame(rows, columns=list(DMS_MANIFEST_COLUMNS))


def dms_batch_row_hash(row: Mapping[str, Any]) -> str:
    payload = {column: int(row[column]) for column in DMS_ROW_HASH_COLUMNS}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_dms_boundary_bank_files(
    task_dir: Path,
    *,
    manifest_rows: list[dict[str, Any]],
    shard_rows: list[dict[str, Any]],
    phase_rate_rows: list[dict[str, Any]],
    layer_input_shapes_by_batch: Mapping[int, Mapping[str, tuple[int, ...]]],
    cache_key: Mapping[str, Any],
) -> DmsBoundaryBank:
    task_dir = Path(task_dir)
    manifest = pd.DataFrame(manifest_rows, columns=list(DMS_MANIFEST_COLUMNS))
    manifest.to_csv(task_dir / "manifest.csv", index=False, encoding="utf-8")
    pd.DataFrame(shard_rows).to_csv(task_dir / "shards.csv", index=False, encoding="utf-8")
    phase_rates = pd.DataFrame(phase_rate_rows)
    phase_rates.to_csv(task_dir / "phase_rates.csv", index=False, encoding="utf-8")
    shape_payload = {
        str(int(batch_id)): {layer: [int(v) for v in shape] for layer, shape in shapes.items()}
        for batch_id, shapes in sorted(layer_input_shapes_by_batch.items())
    }
    write_json(shape_payload, task_dir / "layer_input_shapes.json")
    write_cache_key(task_dir, cache_key)
    return DmsBoundaryBank(
        root=task_dir,
        manifest=manifest,
        phase_rates=phase_rates,
        layer_input_shapes_by_batch={
            int(batch_id): {layer: tuple(int(v) for v in shape) for layer, shape in shapes.items()}
            for batch_id, shapes in layer_input_shapes_by_batch.items()
        },
    )


def load_dms_boundary_bank(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
    dms_trials: pd.DataFrame | None = None,
    batch_size: int | None = None,
) -> DmsBoundaryBank:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id="dms_boundary_bank")
    manifest_path = task_dir / "manifest.csv"
    phase_path = task_dir / "phase_rates.csv"
    shapes_path = task_dir / "layer_input_shapes.json"
    for path in (manifest_path, phase_path, shapes_path):
        if not path.exists():
            raise FileNotFoundError(f"DMS boundary bank required file is missing: {path}")
    manifest = pd.read_csv(manifest_path)
    phase_rates = pd.read_csv(phase_path)
    shapes_payload = read_json(shapes_path)
    shapes_by_batch = {
        int(batch_id): {layer: tuple(int(v) for v in shape) for layer, shape in shapes.items()}
        for batch_id, shapes in shapes_payload.items()
    }
    if dms_trials is not None and batch_size is not None:
        _validate_dms_manifest(manifest, expected_dms_manifest(dms_trials, int(batch_size)))
    return DmsBoundaryBank(task_dir, manifest, phase_rates, shapes_by_batch)


def _validate_dms_manifest(manifest: pd.DataFrame, expected: pd.DataFrame) -> None:
    missing_cols = [col for col in DMS_MANIFEST_COLUMNS if col not in manifest.columns]
    if missing_cols:
        raise ValueError(f"DMS boundary manifest is missing columns: {missing_cols}")
    found = manifest.loc[:, list(DMS_MANIFEST_COLUMNS)].astype(str).reset_index(drop=True)
    want = expected.loc[:, list(DMS_MANIFEST_COLUMNS)].astype(str).reset_index(drop=True)
    if len(found) != len(want):
        raise RuntimeError(f"DMS boundary manifest row count mismatch: expected {len(want)}, found {len(found)}")
    if not found.equals(want):
        mismatch_idx = int(np.flatnonzero((found != want).any(axis=1).to_numpy())[0])
        raise RuntimeError(
            "DMS boundary manifest batch_row_hash mismatch at row "
            f"{mismatch_idx}: expected={want.iloc[mismatch_idx].to_dict()} found={found.iloc[mismatch_idx].to_dict()}"
        )


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
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
    "DmsBoundaryBank",
    "DelayFeatureBank",
    "TrialSpecsArtifact",
    "boundary_shard_path",
    "cache_key_matches",
    "default_artifact_root",
    "dms_batch_row_hash",
    "dms_manifest_rows_for_batch",
    "expected_dms_manifest",
    "load_boundary_shard",
    "load_delay_feature_bank",
    "load_dms_boundary_bank",
    "load_trial_specs_artifact",
    "read_cache_key",
    "reset_task_artifact_dir",
    "save_boundary_shard",
    "save_delay_feature_bank",
    "save_trial_specs_artifact",
    "task_artifact_dir",
    "write_cache_key",
    "write_dms_boundary_bank_files",
]
