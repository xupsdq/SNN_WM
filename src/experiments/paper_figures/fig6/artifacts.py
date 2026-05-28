from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.experiments.paper_figures.fig6.cache_keys import (
    cache_key_digest,
    sequence_trials_hash,
    sha256_file,
    table_digest,
)
from src.experiments.paper_figures.fig6.schemas import (
    ARRAY_MANIFEST_COLUMNS,
    BOUNDARY_MANIFEST_COLUMNS,
    SEQUENCE_BANK_ARRAY_FILES,
    SEQUENCE_BANK_TABLE_FILES,
    SEQUENCE_TRIAL_FILES,
    TABLE_MANIFEST_COLUMNS,
)
from src.experiments.paper_figures.fig6.types import PeakAmplifiedReentryBank


CACHE_KEY_FILE = "cache_key.json"


@dataclass(frozen=True)
class SequenceTrialsArtifact:
    root: Path
    sequence_trials: pd.DataFrame
    table_manifest: pd.DataFrame
    digest: str


@dataclass(frozen=True)
class SequenceBankArtifact:
    root: Path
    bank: PeakAmplifiedReentryBank
    table_manifest: pd.DataFrame
    array_manifest: pd.DataFrame
    boundary_manifest: pd.DataFrame
    digest: str


@dataclass(frozen=True)
class TableBundle:
    root: Path
    tables: dict[str, pd.DataFrame]
    manifest: pd.DataFrame
    digest: str


def default_artifact_root(seed_dir: Path) -> Path:
    return Path(seed_dir) / "data" / "intermediates"


def task_artifact_dir(artifact_root: Path, task_id: str) -> Path:
    return Path(artifact_root) / str(task_id)


def reset_task_artifact_dir(path: Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_cache_key(task_dir: Path, cache_key: Mapping[str, Any]) -> None:
    write_json(
        {
            "cache_key": _json_safe(cache_key),
            "cache_key_digest": cache_key_digest(cache_key),
        },
        Path(task_dir) / CACHE_KEY_FILE,
    )


def read_cache_key(task_dir: Path) -> dict[str, Any]:
    path = Path(task_dir) / CACHE_KEY_FILE
    if not path.exists():
        raise FileNotFoundError(f"Artifact cache key is missing: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict) or "cache_key" not in payload or "cache_key_digest" not in payload:
        raise ValueError(f"Malformed artifact cache key file: {path}")
    return payload


def cache_key_matches(task_dir: Path, expected_key: Mapping[str, Any]) -> bool:
    try:
        payload = read_cache_key(task_dir)
    except (FileNotFoundError, ValueError):
        return False
    return str(payload.get("cache_key_digest")) == cache_key_digest(expected_key)


def require_cache_key_match(task_dir: Path, expected_key: Mapping[str, Any], *, task_id: str) -> None:
    payload = read_cache_key(task_dir)
    expected_digest = cache_key_digest(expected_key)
    found_digest = str(payload.get("cache_key_digest"))
    if found_digest != expected_digest:
        raise RuntimeError(
            f"Fig.6 {task_id} artifact cache key mismatch: expected {expected_digest}, found {found_digest}. "
            "Rebuild the producer task before using --reuse-artifacts require."
        )


def save_sequence_trials_artifact(
    task_dir: Path,
    *,
    sequence_trials: pd.DataFrame,
    cache_key: Mapping[str, Any],
) -> SequenceTrialsArtifact:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    tables = _save_table_bundle(
        task_dir,
        tables={"sequence_trials": sequence_trials.reset_index(drop=True).copy()},
        files=SEQUENCE_TRIAL_FILES,
    )
    digest = sequence_trials_hash(tables.tables["sequence_trials"])
    write_json({"artifact_digest": digest}, task_dir / "artifact_digest.json")
    write_cache_key(task_dir, cache_key)
    return SequenceTrialsArtifact(task_dir, tables.tables["sequence_trials"], tables.manifest, digest)


def load_sequence_trials_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
) -> SequenceTrialsArtifact:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id="sequence_trials")
    tables = _load_table_bundle(task_dir, files=SEQUENCE_TRIAL_FILES, task_id="sequence_trials")
    digest = sequence_trials_hash(tables.tables["sequence_trials"])
    recorded = read_json(task_dir / "artifact_digest.json").get("artifact_digest")
    if str(recorded) != digest:
        raise RuntimeError(f"Fig.6 sequence_trials artifact digest mismatch: expected {recorded}, found {digest}")
    return SequenceTrialsArtifact(task_dir, tables.tables["sequence_trials"], tables.manifest, digest)


def save_sequence_bank_artifact(
    task_dir: Path,
    bank: PeakAmplifiedReentryBank,
    *,
    raw_dir: Path,
    cache_key: Mapping[str, Any],
    network_seed: int,
) -> SequenceBankArtifact:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    table_bundle = _save_table_bundle(
        task_dir,
        tables={
            "sequence_meta": bank.sequence_meta.reset_index(drop=True).copy(),
            "state_bank_manifest": _read_csv_required(Path(raw_dir) / "state_bank_manifest.csv"),
        },
        files=SEQUENCE_BANK_TABLE_FILES,
    )
    array_manifest = _copy_npz_artifacts(task_dir, Path(raw_dir), SEQUENCE_BANK_ARRAY_FILES)
    boundary_manifest = _save_boundaries(
        task_dir,
        bank.boundaries,
        sequence_meta=table_bundle.tables["sequence_meta"],
        network_seed=int(network_seed),
    )
    _validate_sequence_bank_payload(table_bundle.tables["sequence_meta"], array_manifest, boundary_manifest, task_dir)
    digest = _artifact_digest(table_bundle.digest, array_manifest, boundary_manifest)
    write_json({"artifact_digest": digest}, task_dir / "artifact_digest.json")
    write_cache_key(task_dir, cache_key)
    artifact_bank = _bank_from_artifact_tables(
        sequence_trials=bank.sequence_trials.reset_index(drop=True).copy(),
        sequence_meta=table_bundle.tables["sequence_meta"],
        task_dir=task_dir,
        boundary_manifest=boundary_manifest,
    )
    return SequenceBankArtifact(task_dir, artifact_bank, table_bundle.manifest, array_manifest, boundary_manifest, digest)


def load_sequence_bank_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
    sequence_trials: pd.DataFrame,
) -> SequenceBankArtifact:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id="sequence_bank")
    table_bundle = _load_table_bundle(task_dir, files=SEQUENCE_BANK_TABLE_FILES, task_id="sequence_bank")
    array_manifest = _read_array_manifest(task_dir)
    _validate_manifest_hashes(task_dir, array_manifest, path_column="storage_file")
    boundary_manifest = _read_boundary_manifest(task_dir)
    _validate_manifest_hashes(task_dir, boundary_manifest, path_column="storage_file")
    _validate_sequence_bank_payload(table_bundle.tables["sequence_meta"], array_manifest, boundary_manifest, task_dir)
    digest = _artifact_digest(table_bundle.digest, array_manifest, boundary_manifest)
    recorded = read_json(task_dir / "artifact_digest.json").get("artifact_digest")
    if str(recorded) != digest:
        raise RuntimeError(f"Fig.6 sequence_bank artifact digest mismatch: expected {recorded}, found {digest}")
    bank = _bank_from_artifact_tables(
        sequence_trials=sequence_trials.reset_index(drop=True).copy(),
        sequence_meta=table_bundle.tables["sequence_meta"],
        task_dir=task_dir,
        boundary_manifest=boundary_manifest,
    )
    return SequenceBankArtifact(task_dir, bank, table_bundle.manifest, array_manifest, boundary_manifest, digest)


def copy_sequence_bank_artifacts_to_raw(task_dir: Path, raw_dir: Path) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("state_bank_manifest.csv", *SEQUENCE_BANK_ARRAY_FILES):
        src = Path(task_dir) / filename
        if src.exists():
            shutil.copy2(src, raw_dir / filename)


def _bank_from_artifact_tables(
    *,
    sequence_trials: pd.DataFrame,
    sequence_meta: pd.DataFrame,
    task_dir: Path,
    boundary_manifest: pd.DataFrame,
) -> PeakAmplifiedReentryBank:
    update_payload = _load_npz_payload(Path(task_dir) / "update_history_matrix.npz")
    support_payload = _load_npz_payload(Path(task_dir) / "final_support_maps.npz")
    required_update = {
        "update_count",
        "last_update_position",
        "time_since_last_update",
        "update_exposure_by_item",
        "item_activation_history",
    }
    required_support = {"G_baseline", "G_final", "delta_support", "peak_mask", "nonpeak_mask"}
    missing_update = sorted(required_update.difference(update_payload))
    missing_support = sorted(required_support.difference(support_payload))
    if missing_update or missing_support:
        raise RuntimeError(f"Fig.6 sequence_bank arrays missing keys update={missing_update} support={missing_support}")
    update_count = np.asarray(update_payload["update_count"], dtype=np.float32)
    boundaries = _load_boundaries(Path(task_dir), boundary_manifest)
    return PeakAmplifiedReentryBank(
        sequence_trials=sequence_trials.reset_index(drop=True).copy(),
        sequence_meta=sequence_meta.reset_index(drop=True).copy(),
        probe_trials=pd.DataFrame(),
        matched_groups=pd.DataFrame(),
        update_count=update_count,
        last_update_position=np.asarray(update_payload["last_update_position"]),
        time_since_last_update=np.asarray(update_payload["time_since_last_update"]),
        update_exposure_by_item=np.asarray(update_payload["update_exposure_by_item"], dtype=np.float32),
        item_activation_history=np.asarray(update_payload["item_activation_history"], dtype=np.float32),
        g_baseline=np.asarray(support_payload["G_baseline"], dtype=np.float32),
        g_final=np.asarray(support_payload["G_final"], dtype=np.float32),
        delta_support=np.asarray(support_payload["delta_support"], dtype=np.float32),
        peak_mask=np.asarray(support_payload["peak_mask"]).astype(bool),
        nonpeak_mask=np.asarray(support_payload["nonpeak_mask"]).astype(bool),
        prior_updated_mask=update_count > 0,
        boundaries=boundaries,
        reentry_metrics=pd.DataFrame(),
        downstream_metrics=pd.DataFrame(),
    )


def _save_boundaries(
    task_dir: Path,
    boundaries: Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]],
    *,
    sequence_meta: pd.DataFrame,
    network_seed: int,
) -> pd.DataFrame:
    expected_ids = {int(value) for value in sequence_meta["sequence_id"].tolist()}
    found_ids = {int(value) for value in boundaries}
    if found_ids != expected_ids:
        raise RuntimeError(f"Fig.6 boundary sequence mismatch: expected={sorted(expected_ids)}, found={sorted(found_ids)}")
    payload: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for sequence_id, layer_map in sorted(boundaries.items()):
        seq_len = _seq_len_for(sequence_meta, int(sequence_id))
        for layer, state_map in sorted(layer_map.items()):
            for state_key, value in sorted(state_map.items()):
                arr = _tensor_to_array(value)
                storage_key = f"sequence_{int(sequence_id)}__{layer}__{state_key}"
                payload[storage_key] = arr
                rows.append(
                    {
                        "network_seed": int(network_seed),
                        "sequence_id": int(sequence_id),
                        "seq_len": int(seq_len),
                        "layer": str(layer),
                        "state_key": str(state_key),
                        "shape": _shape_text(arr.shape),
                        "storage_file": "boundaries.npz",
                        "storage_key": storage_key,
                        "sha256": "",
                    }
                )
    np.savez_compressed(Path(task_dir) / "boundaries.npz", **payload)
    file_hash = sha256_file(Path(task_dir) / "boundaries.npz")
    for row in rows:
        row["sha256"] = file_hash
    manifest = pd.DataFrame(rows, columns=list(BOUNDARY_MANIFEST_COLUMNS))
    manifest.to_csv(Path(task_dir) / "boundary_manifest.csv", index=False, encoding="utf-8")
    return manifest


def _load_boundaries(task_dir: Path, manifest: pd.DataFrame) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    out: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    with np.load(Path(task_dir) / "boundaries.npz", allow_pickle=False) as payload:
        expected_keys = set(str(value) for value in manifest["storage_key"].tolist())
        if set(payload.files) != expected_keys:
            raise RuntimeError(f"Fig.6 boundary NPZ key mismatch: expected={sorted(expected_keys)}, found={sorted(payload.files)}")
        for row in manifest.to_dict("records"):
            key = str(row["storage_key"])
            arr = np.asarray(payload[key])
            if _shape_text(arr.shape) != str(row["shape"]):
                raise RuntimeError(f"Fig.6 boundary shape mismatch for {key}: expected={row['shape']}, found={_shape_text(arr.shape)}")
            sequence_id = int(row["sequence_id"])
            layer = str(row["layer"])
            state_key = str(row["state_key"])
            out.setdefault(sequence_id, {}).setdefault(layer, {})[state_key] = torch.from_numpy(arr)
    return out


def _save_table_bundle(
    task_dir: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    files: Mapping[str, str],
) -> TableBundle:
    task_dir = Path(task_dir)
    missing = sorted(set(files).difference(tables))
    if missing:
        raise KeyError(f"Cannot save table artifact; missing tables: {missing}")
    written: dict[str, tuple[Path, pd.DataFrame, str]] = {}
    for name, filename in files.items():
        df = tables[name].copy()
        path = task_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        read_back = pd.read_csv(path) if list(df.columns) else pd.DataFrame()
        written[name] = (path, read_back, sha256_file(path))
    digest = table_digest({name: written[name][1] for name in files})
    rows: list[dict[str, Any]] = []
    for name, filename in files.items():
        _path, df, file_hash = written[name]
        rows.append(
            {
                "name": str(name),
                "filename": str(filename),
                "rows": int(len(df)),
                "columns": json.dumps([str(col) for col in df.columns], ensure_ascii=False, separators=(",", ":")),
                "sha256": file_hash,
                "table_digest": digest,
            }
        )
    manifest = pd.DataFrame(rows, columns=list(TABLE_MANIFEST_COLUMNS))
    manifest.to_csv(task_dir / "manifest.csv", index=False, encoding="utf-8")
    return TableBundle(task_dir, {name: written[name][1] for name in files}, manifest, digest)


def _load_table_bundle(task_dir: Path, *, files: Mapping[str, str], task_id: str) -> TableBundle:
    task_dir = Path(task_dir)
    manifest_path = task_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fig.6 {task_id} manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, TABLE_MANIFEST_COLUMNS, manifest_path)
    found_names = set(str(value) for value in manifest["name"].tolist())
    expected_names = set(files)
    if found_names != expected_names:
        raise RuntimeError(f"Fig.6 {task_id} manifest names mismatch: expected={sorted(expected_names)}, found={sorted(found_names)}")
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in files.items():
        rec = _table_manifest_record(manifest, name, filename, manifest_path)
        path = task_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Fig.6 {task_id} artifact file is missing: {path}")
        found_sha = sha256_file(path)
        expected_sha = str(rec["sha256"])
        if found_sha != expected_sha:
            raise RuntimeError(f"Fig.6 {task_id} file hash mismatch for {filename}: expected {expected_sha}, found {found_sha}")
        columns = _parse_manifest_columns(rec["columns"], manifest_path)
        df = pd.read_csv(path) if columns else pd.DataFrame()
        if len(df) != int(rec["rows"]):
            raise RuntimeError(f"Fig.6 {task_id} row count mismatch for {filename}: expected {rec['rows']}, found {len(df)}")
        if list(df.columns) != columns:
            raise RuntimeError(f"Fig.6 {task_id} column mismatch for {filename}: expected {columns}, found {list(df.columns)}")
        tables[name] = df
    digest = table_digest(tables)
    manifest_digests = {str(value) for value in manifest["table_digest"].dropna().astype(str).tolist()}
    if manifest_digests != {digest}:
        raise RuntimeError(f"Fig.6 {task_id} table digest mismatch: manifest={sorted(manifest_digests)}, computed={digest}")
    return TableBundle(task_dir, tables, manifest, digest)


def _copy_npz_artifacts(task_dir: Path, raw_dir: Path, filenames: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for filename in filenames:
        src = Path(raw_dir) / filename
        if not src.exists():
            raise FileNotFoundError(f"Fig.6 sequence bank source NPZ is missing: {src}")
        dst = Path(task_dir) / filename
        shutil.copy2(src, dst)
        file_hash = sha256_file(dst)
        with np.load(dst, allow_pickle=False) as payload:
            for key in payload.files:
                arr = np.asarray(payload[key])
                rows.append(
                    {
                        "name": Path(filename).stem,
                        "storage_file": str(filename),
                        "storage_key": str(key),
                        "shape": _shape_text(arr.shape),
                        "dtype": str(arr.dtype),
                        "sha256": file_hash,
                    }
                )
    manifest = pd.DataFrame(rows, columns=list(ARRAY_MANIFEST_COLUMNS))
    manifest.to_csv(Path(task_dir) / "array_manifest.csv", index=False, encoding="utf-8")
    return manifest


def _read_array_manifest(task_dir: Path) -> pd.DataFrame:
    path = Path(task_dir) / "array_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fig.6 array manifest is missing: {path}")
    manifest = pd.read_csv(path)
    _require_columns(manifest, ARRAY_MANIFEST_COLUMNS, path)
    return manifest


def _read_boundary_manifest(task_dir: Path) -> pd.DataFrame:
    path = Path(task_dir) / "boundary_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fig.6 boundary manifest is missing: {path}")
    manifest = pd.read_csv(path)
    _require_columns(manifest, BOUNDARY_MANIFEST_COLUMNS, path)
    return manifest


def _load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Fig.6 NPZ artifact file is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {str(key): np.asarray(payload[key]) for key in payload.files}


def _validate_sequence_bank_payload(
    sequence_meta: pd.DataFrame,
    array_manifest: pd.DataFrame,
    boundary_manifest: pd.DataFrame,
    task_dir: Path,
) -> None:
    sequence_ids = {int(value) for value in sequence_meta["sequence_id"].tolist()}
    boundary_ids = {int(value) for value in boundary_manifest["sequence_id"].tolist()}
    if boundary_ids != sequence_ids:
        raise RuntimeError(f"Fig.6 boundary sequence mismatch: expected={sorted(sequence_ids)}, found={sorted(boundary_ids)}")
    found_files = set(str(value) for value in array_manifest["storage_file"].tolist())
    missing_files = sorted(set(SEQUENCE_BANK_ARRAY_FILES).difference(found_files))
    if missing_files:
        raise RuntimeError(f"Fig.6 sequence_bank array manifest missing files: {missing_files}")
    update_payload = _load_npz_payload(Path(task_dir) / "update_history_matrix.npz")
    support_payload = _load_npz_payload(Path(task_dir) / "final_support_maps.npz")
    n_sequences = len(sequence_meta)
    for key, payload in (
        ("update_count", update_payload),
        ("G_baseline", support_payload),
        ("G_final", support_payload),
        ("delta_support", support_payload),
        ("peak_mask", support_payload),
        ("nonpeak_mask", support_payload),
    ):
        if key not in payload:
            raise RuntimeError(f"Fig.6 sequence_bank missing array key: {key}")
        if np.asarray(payload[key]).shape[0] != n_sequences:
            raise RuntimeError(f"Fig.6 sequence_bank {key} row count mismatch: expected={n_sequences}, found={np.asarray(payload[key]).shape[0]}")


def _artifact_digest(table_digest_value: str, array_manifest: pd.DataFrame, boundary_manifest: pd.DataFrame) -> str:
    return table_digest(
        {
            "digest": pd.DataFrame(
                [
                    {
                        "tables": table_digest_value,
                        "arrays": table_digest({"array_manifest": array_manifest.loc[:, list(ARRAY_MANIFEST_COLUMNS)].copy()}),
                        "boundaries": table_digest({"boundary_manifest": boundary_manifest.loc[:, list(BOUNDARY_MANIFEST_COLUMNS)].copy()}),
                    }
                ]
            )
        }
    )


def _validate_manifest_hashes(task_dir: Path, manifest: pd.DataFrame, *, path_column: str) -> None:
    for rel_path, part in manifest.groupby(path_column, sort=False):
        path = Path(task_dir) / str(rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Fig.6 artifact file is missing: {path}")
        found = sha256_file(path)
        expected = {str(value) for value in part["sha256"].astype(str).tolist()}
        if expected != {found}:
            raise RuntimeError(f"Fig.6 artifact hash mismatch for {path}: expected={sorted(expected)}, found={found}")


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fig.6 required CSV is missing: {path}")
    return pd.read_csv(path)


def _table_manifest_record(manifest: pd.DataFrame, name: str, filename: str, manifest_path: Path) -> dict[str, Any]:
    rows = manifest[
        manifest["name"].astype(str).eq(str(name))
        & manifest["filename"].astype(str).eq(str(filename))
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Fig.6 table manifest expected one row for {name}/{filename}, found {len(rows)}: {manifest_path}")
    return rows.iloc[0].to_dict()


def _parse_manifest_columns(value: Any, manifest_path: Path) -> list[str]:
    try:
        columns = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Fig.6 manifest has malformed columns JSON in {manifest_path}: {value!r}") from exc
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise ValueError(f"Fig.6 manifest columns must be a JSON string list in {manifest_path}: {value!r}")
    return columns


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")


def _seq_len_for(sequence_meta: pd.DataFrame, sequence_id: int) -> int:
    rows = sequence_meta[sequence_meta["sequence_id"].astype(int).eq(int(sequence_id))]
    if rows.empty:
        raise KeyError(f"Fig.6 sequence_meta has no sequence_id={sequence_id}")
    return int(rows.iloc[0]["seq_len"])


def _tensor_to_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _shape_text(shape: tuple[int, ...] | Any) -> str:
    return "x".join(str(int(value)) for value in tuple(shape))


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
    "SequenceBankArtifact",
    "SequenceTrialsArtifact",
    "cache_key_matches",
    "copy_sequence_bank_artifacts_to_raw",
    "default_artifact_root",
    "load_sequence_bank_artifact",
    "load_sequence_trials_artifact",
    "read_cache_key",
    "save_sequence_bank_artifact",
    "save_sequence_trials_artifact",
    "task_artifact_dir",
    "write_cache_key",
    "write_json",
]
