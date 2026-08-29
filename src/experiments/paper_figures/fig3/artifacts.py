from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.experiments.paper_figures.common.artifact_runtime import (
    CACHE_KEY_FILE,
    default_artifact_root,
    task_artifact_dir,
    write_cache_key,
)
from src.experiments.paper_figures.fig3.cache_keys import cache_key_digest, sha256_file, table_digest
from src.experiments.paper_figures.fig3.schemas import (
    BOUNDARY_MANIFEST_COLUMNS,
    LANDSCAPE_MANIFEST_COLUMNS,
    SEQUENCE_SPEC_FILES,
    SEQUENCE_SPEC_MANIFEST_COLUMNS,
    STATE_BANK_MANIFEST_COLUMNS,
    TABLE_ARTIFACT_MANIFEST_COLUMNS,
    TASK_STATE_BANK,
)
from src.experiments.paper_figures.fig3.types import MultiItemSequenceLandscapeBank


MANIFEST_FILE = "manifest.csv"


@dataclass
class SequenceSpecArtifact:
    path: Path
    sequence_trials: pd.DataFrame
    singleton_reference_trials: pd.DataFrame
    partial_cue_trials: pd.DataFrame
    manifest: pd.DataFrame
    digest: str


@dataclass
class StateBankArtifact:
    path: Path
    bank: MultiItemSequenceLandscapeBank
    manifest: pd.DataFrame
    boundary_manifest: pd.DataFrame
    landscape_manifest: pd.DataFrame
    digest: str


@dataclass
class TableBundleArtifact:
    path: Path
    tables: dict[str, pd.DataFrame]
    manifest: pd.DataFrame
    digest: str


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def read_cache_key(task_dir: Path) -> dict[str, Any]:
    path = task_dir / CACHE_KEY_FILE
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "cache_key" not in payload or "cache_key_digest" not in payload:
        raise ValueError(f"Malformed cache key file: {path}")
    return payload


def cache_key_matches(task_dir: Path, expected_key: Mapping[str, Any]) -> bool:
    try:
        payload = read_cache_key(task_dir)
    except Exception:
        return False
    return str(payload.get("cache_key_digest")) == cache_key_digest(expected_key)


def require_cache_key_match(task_dir: Path, expected_key: Mapping[str, Any], *, task_id: str) -> None:
    payload = read_cache_key(task_dir)
    expected_digest = cache_key_digest(expected_key)
    found_digest = str(payload.get("cache_key_digest"))
    if found_digest != expected_digest:
        raise ValueError(
            f"Fig.3 artifact cache key mismatch for {task_id}: expected {expected_digest}, found {found_digest}. "
            "Rebuild the producer task before using --reuse-artifacts require."
        )


def save_sequence_specs_artifact(
    task_dir: Path,
    *,
    sequence_trials: pd.DataFrame,
    singleton_reference_trials: pd.DataFrame,
    partial_cue_trials: pd.DataFrame,
    cache_key: Mapping[str, Any],
) -> SequenceSpecArtifact:
    task_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "sequence_trials": sequence_trials.reset_index(drop=True).copy(),
        "singleton_reference_trials": singleton_reference_trials.reset_index(drop=True).copy(),
        "partial_cue_trials": partial_cue_trials.reset_index(drop=True).copy(),
    }
    persisted_tables: dict[str, pd.DataFrame] = {}
    manifest_rows = []
    for name, df in tables.items():
        filename = SEQUENCE_SPEC_FILES[name]
        path = task_dir / filename
        df.to_csv(path, index=False, encoding="utf-8")
        persisted = pd.read_csv(path)
        persisted_tables[name] = persisted
        manifest_rows.append(_table_manifest_row(name, filename, path, persisted))
    manifest = pd.DataFrame(manifest_rows, columns=list(SEQUENCE_SPEC_MANIFEST_COLUMNS))
    manifest.to_csv(task_dir / MANIFEST_FILE, index=False, encoding="utf-8")
    write_cache_key(task_dir, cache_key)
    return SequenceSpecArtifact(
        path=task_dir,
        sequence_trials=persisted_tables["sequence_trials"],
        singleton_reference_trials=persisted_tables["singleton_reference_trials"],
        partial_cue_trials=persisted_tables["partial_cue_trials"],
        manifest=manifest,
        digest=table_digest(persisted_tables),
    )


def load_sequence_specs_artifact(task_dir: Path, *, expected_key: Mapping[str, Any]) -> SequenceSpecArtifact:
    require_cache_key_match(task_dir, expected_key, task_id="sequence_trial_specs")
    manifest_path = task_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fig.3 sequence specs manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, SEQUENCE_SPEC_MANIFEST_COLUMNS, manifest_path)
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in SEQUENCE_SPEC_FILES.items():
        row = _single_manifest_row(manifest, name=name, path=manifest_path)
        if str(row["filename"]) != filename:
            raise ValueError(f"Fig.3 sequence specs manifest filename mismatch for {name}: {row['filename']} != {filename}")
        path = task_dir / filename
        _require_file_hash(path, str(row["sha256"]))
        df = pd.read_csv(path)
        if int(row["rows"]) != len(df):
            raise ValueError(f"Fig.3 sequence specs row count mismatch for {path}: {len(df)} != {row['rows']}")
        tables[name] = df
    return SequenceSpecArtifact(
        path=task_dir,
        sequence_trials=tables["sequence_trials"],
        singleton_reference_trials=tables["singleton_reference_trials"],
        partial_cue_trials=tables["partial_cue_trials"],
        manifest=manifest,
        digest=table_digest(tables),
    )


def save_table_bundle_artifact(
    task_dir: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    filenames: Mapping[str, str] | None = None,
    cache_key: Mapping[str, Any],
) -> TableBundleArtifact:
    task_dir.mkdir(parents=True, exist_ok=True)
    filenames = dict(filenames or {})
    persisted_tables: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, Any]] = []
    for name, df in sorted(tables.items()):
        filename = filenames.get(name, f"{name}.csv")
        path = task_dir / filename
        clean = df.reset_index(drop=True).copy()
        clean.to_csv(path, index=False, encoding="utf-8")
        persisted = pd.read_csv(path)
        persisted_tables[str(name)] = persisted
        manifest_rows.append(_table_manifest_row(str(name), filename, path, persisted))
    manifest = pd.DataFrame(manifest_rows, columns=list(TABLE_ARTIFACT_MANIFEST_COLUMNS))
    manifest.to_csv(task_dir / MANIFEST_FILE, index=False, encoding="utf-8")
    write_cache_key(task_dir, cache_key)
    return TableBundleArtifact(
        path=task_dir,
        tables=persisted_tables,
        manifest=manifest,
        digest=table_digest(persisted_tables),
    )


def load_table_bundle_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any],
    expected_names: Sequence[str] | None = None,
    expected_columns: Mapping[str, Sequence[str]] | None = None,
) -> TableBundleArtifact:
    require_cache_key_match(task_dir, expected_key, task_id=str(expected_key.get("task_id", "table_bundle")))
    manifest_path = task_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fig.3 table artifact manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, TABLE_ARTIFACT_MANIFEST_COLUMNS, manifest_path)
    expected = set(str(name) for name in (expected_names or manifest["name"].astype(str).tolist()))
    found = set(manifest["name"].astype(str).tolist())
    missing = sorted(expected - found)
    if missing:
        raise ValueError(f"Fig.3 table artifact missing expected tables {missing} in {manifest_path}")
    tables: dict[str, pd.DataFrame] = {}
    for name in sorted(expected):
        row = _single_manifest_row(manifest, name=name, path=manifest_path)
        filename = str(row["filename"])
        path = task_dir / filename
        _require_file_hash(path, str(row["sha256"]))
        df = pd.read_csv(path)
        if int(row["rows"]) != len(df):
            raise ValueError(f"Fig.3 table artifact row count mismatch for {path}: {len(df)} != {row['rows']}")
        columns = ",".join(str(col) for col in df.columns)
        if str(row["columns"]) != columns:
            raise ValueError(f"Fig.3 table artifact columns mismatch for {path}: {columns} != {row['columns']}")
        if expected_columns and name in expected_columns:
            _require_data_columns(df, expected_columns[name], path)
        tables[name] = df
    return TableBundleArtifact(path=task_dir, tables=tables, manifest=manifest, digest=table_digest(tables))


def save_state_bank_artifact(
    task_dir: Path,
    bank: MultiItemSequenceLandscapeBank,
    *,
    cache_key: Mapping[str, Any],
    network_seed: int,
) -> StateBankArtifact:
    task_dir.mkdir(parents=True, exist_ok=True)
    sequence_meta = bank.sequence_meta.reset_index(drop=True).copy()
    sequence_meta.to_csv(task_dir / "sequence_meta.csv", index=False, encoding="utf-8")

    layer_payloads: dict[str, dict[str, np.ndarray]] = {
        "state_bank_layer1.npz": {},
        "state_bank_layer2.npz": {},
        "state_bank_layer3.npz": {},
    }
    manifest_rows: list[dict[str, Any]] = []
    for seq_id, state_map in sorted(bank.arrays.items()):
        seq_len = _seq_len_for(sequence_meta, int(seq_id))
        for state, layer_map in sorted(state_map.items()):
            stage_k = 0 if state == "S0" else (seq_len if state == "S_final" else int(str(state).split("_")[1]))
            for layer, variable_map in sorted(layer_map.items()):
                storage_file = _layer_storage_file(layer)
                for variable, value in sorted(variable_map.items()):
                    arr = np.asarray(value, dtype=np.float32)
                    storage_key = f"sequence_{int(seq_id)}_{str(state).replace('_', '')}_{variable}"
                    layer_payloads[storage_file][storage_key] = arr
                    manifest_rows.append(_array_manifest_row("state_array", network_seed, int(seq_id), seq_len, state, stage_k, layer, variable, storage_file, storage_key, arr))
    for seq_id, ref_map in sorted(bank.singleton_refs.items()):
        seq_len = _seq_len_for(sequence_meta, int(seq_id))
        for pos, layer_map in sorted(ref_map.items()):
            for layer, variable_map in sorted(layer_map.items()):
                storage_file = _layer_storage_file(layer)
                for variable, value in sorted(variable_map.items()):
                    arr = np.asarray(value, dtype=np.float32)
                    storage_key = f"sequence_{int(seq_id)}_singleton_reference_{int(pos)}_{variable}"
                    layer_payloads[storage_file][storage_key] = arr
                    manifest_rows.append(_array_manifest_row("singleton_reference", network_seed, int(seq_id), seq_len, "singleton_reference", int(pos), layer, variable, storage_file, storage_key, arr))
    for filename, payload in layer_payloads.items():
        np.savez_compressed(task_dir / filename, **payload)
    state_manifest = pd.DataFrame(manifest_rows, columns=list(STATE_BANK_MANIFEST_COLUMNS))
    _fill_storage_hashes(state_manifest, task_dir)
    state_manifest.to_csv(task_dir / MANIFEST_FILE, index=False, encoding="utf-8")

    boundary_payload: dict[str, np.ndarray] = {}
    boundary_rows: list[dict[str, Any]] = []
    _append_boundary_payload(
        boundary_payload,
        boundary_rows,
        "boundary",
        bank.boundaries,
        network_seed=network_seed,
        sequence_meta=sequence_meta,
    )
    _append_singleton_boundary_payload(
        boundary_payload,
        boundary_rows,
        bank.singleton_boundaries,
        network_seed=network_seed,
        sequence_meta=sequence_meta,
    )
    np.savez_compressed(task_dir / "boundaries.npz", **boundary_payload)
    boundary_manifest = pd.DataFrame(boundary_rows, columns=list(BOUNDARY_MANIFEST_COLUMNS))
    _fill_storage_hashes(boundary_manifest, task_dir)
    boundary_manifest.to_csv(task_dir / "boundary_manifest.csv", index=False, encoding="utf-8")

    landscape_payload: dict[str, np.ndarray] = {}
    landscape_rows: list[dict[str, Any]] = []
    for seq_id, landscape in sorted(bank.landscapes.items()):
        for key, value in sorted(landscape.items()):
            arr = np.asarray(value)
            storage_key = f"sequence_{int(seq_id)}_{key}"
            landscape_payload[storage_key] = arr
            landscape_rows.append(
                {
                    "artifact_kind": "landscape",
                    "network_seed": int(network_seed),
                    "sequence_id": int(seq_id),
                    "landscape_key": str(key),
                    "shape": _shape_text(arr.shape),
                    "storage_file": "landscapes.npz",
                    "storage_key": storage_key,
                    "sha256": "",
                }
            )
    np.savez_compressed(task_dir / "landscapes.npz", **landscape_payload)
    landscape_manifest = pd.DataFrame(landscape_rows, columns=list(LANDSCAPE_MANIFEST_COLUMNS))
    _fill_storage_hashes(landscape_manifest, task_dir)
    landscape_manifest.to_csv(task_dir / "landscape_manifest.csv", index=False, encoding="utf-8")

    write_cache_key(task_dir, cache_key)
    digest = table_digest({"manifest": state_manifest, "boundary_manifest": boundary_manifest, "landscape_manifest": landscape_manifest})
    return StateBankArtifact(task_dir, bank, state_manifest, boundary_manifest, landscape_manifest, digest)


def load_state_bank_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any],
    sequence_trials: pd.DataFrame,
) -> StateBankArtifact:
    require_cache_key_match(task_dir, expected_key, task_id=TASK_STATE_BANK)
    sequence_meta_path = task_dir / "sequence_meta.csv"
    manifest_path = task_dir / MANIFEST_FILE
    boundary_manifest_path = task_dir / "boundary_manifest.csv"
    landscape_manifest_path = task_dir / "landscape_manifest.csv"
    for path in (sequence_meta_path, manifest_path, boundary_manifest_path, landscape_manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Fig.3 state-bank artifact required file is missing: {path}")
    sequence_meta = pd.read_csv(sequence_meta_path)
    manifest = pd.read_csv(manifest_path)
    boundary_manifest = pd.read_csv(boundary_manifest_path)
    landscape_manifest = pd.read_csv(landscape_manifest_path)
    _require_columns(manifest, STATE_BANK_MANIFEST_COLUMNS, manifest_path)
    _require_columns(boundary_manifest, BOUNDARY_MANIFEST_COLUMNS, boundary_manifest_path)
    _require_columns(landscape_manifest, LANDSCAPE_MANIFEST_COLUMNS, landscape_manifest_path)
    _validate_manifest_files(manifest, task_dir)
    _validate_manifest_files(boundary_manifest, task_dir)
    _validate_manifest_files(landscape_manifest, task_dir)

    arrays: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    singleton_refs: dict[int, dict[int, dict[str, dict[str, np.ndarray]]]] = {}
    npz_cache: dict[str, Any] = {}
    try:
        for _, row in manifest.iterrows():
            storage_file = str(row["storage_file"])
            storage_key = str(row["storage_key"])
            npz = _open_npz(npz_cache, task_dir / storage_file)
            if storage_key not in npz:
                raise KeyError(f"Fig.3 state-bank artifact missing storage key {storage_key!r} in {storage_file}")
            arr = np.asarray(npz[storage_key])
            _require_shape(arr, str(row["shape"]), storage_key)
            seq_id = int(row["sequence_id"])
            layer = str(row["layer"])
            variable = str(row["state_variable"])
            if str(row["artifact_kind"]) == "singleton_reference":
                pos = int(row["stage_k"])
                singleton_refs.setdefault(seq_id, {}).setdefault(pos, {}).setdefault(layer, {})[variable] = arr
            else:
                state = str(row["state_condition"])
                arrays.setdefault(seq_id, {}).setdefault(state, {}).setdefault(layer, {})[variable] = arr
    finally:
        _close_npz_cache(npz_cache)

    boundaries: dict[int, dict[str, Mapping[str, Mapping[str, torch.Tensor]]]] = {}
    singleton_boundaries: dict[int, dict[int, Mapping[str, Mapping[str, torch.Tensor]]]] = {}
    with np.load(task_dir / "boundaries.npz", allow_pickle=False) as npz:
        for _, row in boundary_manifest.iterrows():
            storage_key = str(row["storage_key"])
            if storage_key not in npz:
                raise KeyError(f"Fig.3 boundary artifact missing storage key {storage_key!r}")
            arr = np.asarray(npz[storage_key])
            _require_shape(arr, str(row["shape"]), storage_key)
            tensor = torch.from_numpy(arr)
            seq_id = int(row["sequence_id"])
            layer = str(row["layer"])
            state_key = str(row["state_key"])
            if str(row["artifact_kind"]) == "singleton_boundary":
                pos = int(row["stage_k"])
                singleton_boundaries.setdefault(seq_id, {}).setdefault(pos, {}).setdefault(layer, {})[state_key] = tensor
            else:
                state = str(row["state_condition"])
                boundaries.setdefault(seq_id, {}).setdefault(state, {}).setdefault(layer, {})[state_key] = tensor

    landscapes: dict[int, dict[str, np.ndarray]] = {}
    with np.load(task_dir / "landscapes.npz", allow_pickle=False) as npz:
        for _, row in landscape_manifest.iterrows():
            storage_key = str(row["storage_key"])
            if storage_key not in npz:
                raise KeyError(f"Fig.3 landscape artifact missing storage key {storage_key!r}")
            arr = np.asarray(npz[storage_key])
            _require_shape(arr, str(row["shape"]), storage_key)
            landscapes.setdefault(int(row["sequence_id"]), {})[str(row["landscape_key"])] = arr

    meta_sequences = set(int(v) for v in sequence_meta["sequence_id"].dropna().unique())
    if set(arrays) != meta_sequences:
        raise ValueError(f"Fig.3 state-bank sequence membership mismatch: arrays={sorted(arrays)} sequence_meta={sorted(meta_sequences)}")
    expected_sequences = set(int(v) for v in sequence_trials["sequence_id"].dropna().unique())
    if "source_sequence_id" in sequence_meta.columns:
        source_sequences = set(int(v) for v in sequence_meta["source_sequence_id"].dropna().unique())
        unknown_sources = sorted(source_sequences - expected_sequences)
        if unknown_sources:
            raise ValueError(f"Fig.3 boundary state-bank source sequence ids not present in specs: {unknown_sources}")
    elif set(arrays) != expected_sequences:
        raise ValueError(f"Fig.3 state-bank sequence membership mismatch: arrays={sorted(arrays)} expected={sorted(expected_sequences)}")
    bank = MultiItemSequenceLandscapeBank(
        sequence_trials=sequence_trials.reset_index(drop=True).copy(),
        sequence_meta=sequence_meta,
        arrays=arrays,
        singleton_refs=singleton_refs,
        singleton_boundaries=singleton_boundaries,
        boundaries=boundaries,
        landscapes=landscapes,
    )
    digest = table_digest({"manifest": manifest, "boundary_manifest": boundary_manifest, "landscape_manifest": landscape_manifest})
    return StateBankArtifact(task_dir, bank, manifest, boundary_manifest, landscape_manifest, digest)


def _table_manifest_row(name: str, filename: str, path: Path, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "name": name,
        "filename": filename,
        "rows": int(len(df)),
        "columns": ",".join(str(col) for col in df.columns),
        "sha256": sha256_file(path),
        "table_digest": table_digest({name: df}),
    }


def _array_manifest_row(
    artifact_kind: str,
    network_seed: int,
    sequence_id: int,
    seq_len: int,
    state_condition: str,
    stage_k: int,
    layer: str,
    state_variable: str,
    storage_file: str,
    storage_key: str,
    arr: np.ndarray,
) -> dict[str, Any]:
    return {
        "artifact_kind": artifact_kind,
        "network_seed": int(network_seed),
        "sequence_id": int(sequence_id),
        "seq_len": int(seq_len),
        "state_condition": str(state_condition),
        "stage_k": int(stage_k),
        "layer": str(layer),
        "state_variable": str(state_variable),
        "shape": _shape_text(arr.shape),
        "storage_file": str(storage_file),
        "storage_key": str(storage_key),
        "sha256": "",
    }


def _append_boundary_payload(
    payload: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    artifact_kind: str,
    boundaries: Mapping[int, Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]]],
    *,
    network_seed: int,
    sequence_meta: pd.DataFrame,
) -> None:
    for seq_id, state_map in sorted(boundaries.items()):
        seq_len = _seq_len_for(sequence_meta, int(seq_id))
        for state, layer_map in sorted(state_map.items()):
            stage_k = 0 if state == "S0" else (seq_len if state == "S_final" else int(str(state).split("_")[1]))
            for layer, state_values in sorted(layer_map.items()):
                for state_key, value in sorted(state_values.items()):
                    arr = _tensor_to_array(value)
                    storage_key = f"sequence_{int(seq_id)}_{str(state).replace('_', '')}_{layer}_{state_key}"
                    payload[storage_key] = arr
                    rows.append(_boundary_manifest_row(artifact_kind, network_seed, int(seq_id), seq_len, state, stage_k, layer, state_key, storage_key, arr))


def _append_singleton_boundary_payload(
    payload: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    singleton_boundaries: Mapping[int, Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]]],
    *,
    network_seed: int,
    sequence_meta: pd.DataFrame,
) -> None:
    for seq_id, pos_map in sorted(singleton_boundaries.items()):
        seq_len = _seq_len_for(sequence_meta, int(seq_id))
        for pos, layer_map in sorted(pos_map.items()):
            for layer, state_values in sorted(layer_map.items()):
                for state_key, value in sorted(state_values.items()):
                    arr = _tensor_to_array(value)
                    storage_key = f"sequence_{int(seq_id)}_singleton_boundary_{int(pos)}_{layer}_{state_key}"
                    payload[storage_key] = arr
                    rows.append(_boundary_manifest_row("singleton_boundary", network_seed, int(seq_id), seq_len, "singleton_boundary", int(pos), layer, state_key, storage_key, arr))


def _boundary_manifest_row(
    artifact_kind: str,
    network_seed: int,
    sequence_id: int,
    seq_len: int,
    state_condition: str,
    stage_k: int,
    layer: str,
    state_key: str,
    storage_key: str,
    arr: np.ndarray,
) -> dict[str, Any]:
    return {
        "artifact_kind": str(artifact_kind),
        "network_seed": int(network_seed),
        "sequence_id": int(sequence_id),
        "seq_len": int(seq_len),
        "state_condition": str(state_condition),
        "stage_k": int(stage_k),
        "layer": str(layer),
        "state_key": str(state_key),
        "shape": _shape_text(arr.shape),
        "storage_file": "boundaries.npz",
        "storage_key": str(storage_key),
        "sha256": "",
    }


def _fill_storage_hashes(manifest: pd.DataFrame, task_dir: Path) -> None:
    hashes: dict[str, str] = {}
    for storage_file in sorted(set(str(v) for v in manifest["storage_file"].dropna().unique())):
        if not storage_file:
            continue
        hashes[storage_file] = sha256_file(task_dir / storage_file)
    manifest["sha256"] = manifest["storage_file"].map(hashes).fillna("")


def _validate_manifest_files(manifest: pd.DataFrame, task_dir: Path) -> None:
    for storage_file, part in manifest.groupby("storage_file", sort=True):
        path = task_dir / str(storage_file)
        expected = str(part["sha256"].iloc[0])
        _require_file_hash(path, expected)


def _require_file_hash(path: Path, expected_sha: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fig.3 artifact required file is missing: {path}")
    found = sha256_file(path)
    if str(found) != str(expected_sha):
        raise ValueError(f"Fig.3 artifact hash mismatch for {path}: expected {expected_sha}, found {found}")


def _single_manifest_row(manifest: pd.DataFrame, *, name: str, path: Path) -> pd.Series:
    part = manifest[manifest["name"].astype(str).eq(str(name))]
    if len(part) != 1:
        raise ValueError(f"Expected exactly one manifest row named {name!r} in {path}, found {len(part)}")
    return part.iloc[0]


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Fig.3 artifact manifest missing columns {missing} in {path}")


def _require_data_columns(df: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    missing = [str(col) for col in columns if str(col) not in df.columns]
    if missing:
        raise ValueError(
            f"Fig.3 table artifact missing required data columns {missing} in {path}. "
            "This artifact predates the current Fig.3 schema; rebuild the producer task."
        )


def _require_shape(arr: np.ndarray, expected_shape: str, storage_key: str) -> None:
    found = _shape_text(arr.shape)
    if found != str(expected_shape):
        raise ValueError(f"Fig.3 artifact shape mismatch for {storage_key}: expected {expected_shape}, found {found}")


def _open_npz(cache: dict[str, Any], path: Path) -> Any:
    key = str(path)
    if key not in cache:
        cache[key] = np.load(path, allow_pickle=False)
    return cache[key]


def _close_npz_cache(cache: dict[str, Any]) -> None:
    for value in cache.values():
        value.close()


def _seq_len_for(sequence_meta: pd.DataFrame, sequence_id: int) -> int:
    part = sequence_meta[sequence_meta["sequence_id"].astype(int).eq(int(sequence_id))]
    if part.empty:
        raise ValueError(f"Missing Fig.3 sequence_meta row for sequence_id={sequence_id}")
    return int(part["seq_len"].iloc[0])


def _layer_storage_file(layer: str) -> str:
    if str(layer) == "layer1":
        return "state_bank_layer1.npz"
    if str(layer) == "layer2":
        return "state_bank_layer2.npz"
    if str(layer) == "layer3":
        return "state_bank_layer3.npz"
    raise ValueError(f"Unsupported Fig.3 state-bank artifact layer: {layer!r}")


def _tensor_to_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().to("cpu").numpy()


def _shape_text(shape: Any) -> str:
    return "x".join(str(int(v)) for v in tuple(shape))


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
        return _json_safe(value.item())
    return value


__all__ = [
    "SequenceSpecArtifact",
    "StateBankArtifact",
    "TableBundleArtifact",
    "cache_key_matches",
    "default_artifact_root",
    "load_sequence_specs_artifact",
    "load_state_bank_artifact",
    "load_table_bundle_artifact",
    "read_cache_key",
    "save_sequence_specs_artifact",
    "save_state_bank_artifact",
    "save_table_bundle_artifact",
    "task_artifact_dir",
    "write_json",
]
