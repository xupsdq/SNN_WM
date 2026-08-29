from __future__ import annotations

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
    require_cache_key_match as _require_cache_key_match,
    reset_task_artifact_dir,
    task_artifact_dir,
    validate_cache_key_integrity,
    write_cache_key,
)
from src.experiments.paper_figures.fig2.cache_keys import sha256_file, table_digest
from src.experiments.paper_figures.fig2.schemas import (
    BOUNDARY_MANIFEST_COLUMNS,
    BOUNDARY_STATE_KEYS,
    COMPLETION_BOUNDARY_MANIFEST_COLUMNS,
    COMPLETION_CONDITIONS,
    COMPLETION_DELAY_MASK_COLUMNS,
    CROSSFIT_SPLIT_COLUMNS,
    CROSSFIT_SPLIT_FILE,
    CROSSFIT_NULL_SPEC_COLUMNS,
    CROSSFIT_NULL_SPEC_FILE,
    PAIR_SPEC_FILES,
    PAIR_SPEC_MANIFEST_COLUMNS,
    STATE_BANK_ARRAY_VARIABLES,
    STATE_BANK_MANIFEST_COLUMNS,
    TABLE_MANIFEST_COLUMNS,
    WEAK_PROBE_MASK_COLUMNS,
)
from src.experiments.paper_figures.fig2.types import PairEpisodeStateBank


CACHE_KEY_FILE = _CACHE_KEY_FILE


@dataclass(frozen=True)
class PairTrialSpecsArtifact:
    root: Path
    pair_trials: pd.DataFrame
    candidate_pool: pd.DataFrame
    manifest: pd.DataFrame
    digest: str


@dataclass(frozen=True)
class StateBankArtifact:
    root: Path
    bank: PairEpisodeStateBank
    state_manifest: pd.DataFrame
    boundary_manifest: pd.DataFrame


@dataclass(frozen=True)
class CompletionDelayBoundaryBank:
    root: Path
    boundary_states_by_delay: dict[int, dict[str, Mapping[str, Mapping[str, torch.Tensor]]]]
    layer_input_shapes_by_delay: dict[int, dict[str, tuple[int, ...]]]
    manifest: pd.DataFrame

    def boundary_states_for_delay(self, delay2_ms: int) -> dict[str, Mapping[str, Mapping[str, torch.Tensor]]]:
        delay = int(delay2_ms)
        if delay not in self.boundary_states_by_delay:
            raise FileNotFoundError(f"Completion-delay boundary artifact is missing delay2_ms={delay}")
        return self.boundary_states_by_delay[delay]


@dataclass(frozen=True)
class TableArtifact:
    root: Path
    table: Any
    manifest: pd.DataFrame
    digest: str


def require_cache_key_match(task_dir: Path, expected_key: Mapping[str, Any], *, task_id: str) -> None:
    _require_cache_key_match(
        task_dir,
        expected_key,
        task_id=task_id,
        mismatch_hint="Rebuild the producer task before using --reuse-artifacts require.",
    )


def save_pair_trial_specs_artifact(
    task_dir: Path,
    *,
    pair_trials: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    cache_key: Mapping[str, Any],
) -> PairTrialSpecsArtifact:
    tables = {"pair_trials": pair_trials.copy(), "candidate_pool": candidate_pool.copy()}
    artifact = _save_table_bundle(
        task_dir,
        tables=tables,
        files=PAIR_SPEC_FILES,
        manifest_columns=PAIR_SPEC_MANIFEST_COLUMNS,
        cache_key=cache_key,
    )
    return PairTrialSpecsArtifact(
        root=artifact.root,
        pair_trials=artifact.table["pair_trials"].copy(),
        candidate_pool=artifact.table["candidate_pool"].copy(),
        manifest=artifact.manifest,
        digest=artifact.digest,
    )


def load_pair_trial_specs_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
) -> PairTrialSpecsArtifact:
    artifact = _load_table_bundle(
        task_dir,
        files=PAIR_SPEC_FILES,
        manifest_columns=PAIR_SPEC_MANIFEST_COLUMNS,
        expected_key=expected_key,
        task_id="pair_trial_specs",
    )
    return PairTrialSpecsArtifact(
        root=artifact.root,
        pair_trials=artifact.table["pair_trials"].copy(),
        candidate_pool=artifact.table["candidate_pool"].copy(),
        manifest=artifact.manifest,
        digest=artifact.digest,
    )


def save_crossfit_split_specs_artifact(
    task_dir: Path,
    table: pd.DataFrame,
    *,
    cache_key: Mapping[str, Any],
) -> TableArtifact:
    return _save_single_table_artifact(
        task_dir,
        table=table,
        filename=CROSSFIT_SPLIT_FILE,
        name="crossfit_split_specs",
        columns=CROSSFIT_SPLIT_COLUMNS,
        cache_key=cache_key,
    )


def load_crossfit_split_specs_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
) -> TableArtifact:
    return _load_single_table_artifact(
        task_dir,
        filename=CROSSFIT_SPLIT_FILE,
        name="crossfit_split_specs",
        columns=CROSSFIT_SPLIT_COLUMNS,
        expected_key=expected_key,
        task_id="crossfit_split_specs",
    )


def save_crossfit_null_specs_artifact(
    task_dir: Path,
    table: pd.DataFrame,
    *,
    cache_key: Mapping[str, Any],
) -> TableArtifact:
    return _save_single_table_artifact(
        task_dir,
        table=table,
        filename=CROSSFIT_NULL_SPEC_FILE,
        name="crossfit_null_specs",
        columns=CROSSFIT_NULL_SPEC_COLUMNS,
        cache_key=cache_key,
    )


def load_crossfit_null_specs_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
) -> TableArtifact:
    return _load_single_table_artifact(
        task_dir,
        filename=CROSSFIT_NULL_SPEC_FILE,
        name="crossfit_null_specs",
        columns=CROSSFIT_NULL_SPEC_COLUMNS,
        expected_key=expected_key,
        task_id="crossfit_null_specs",
    )


def save_state_bank_artifact(
    task_dir: Path,
    bank: PairEpisodeStateBank,
    *,
    cache_key: Mapping[str, Any],
    network_seed: int,
) -> StateBankArtifact:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    row_count = len(bank.pair_trials)
    l3_payload: dict[str, np.ndarray] = {}
    all_payload: dict[str, np.ndarray] = {}
    state_rows: list[dict[str, Any]] = []
    for condition in sorted(bank.arrays):
        for layer in LAYER_KEYS:
            for variable in STATE_BANK_ARRAY_VARIABLES:
                arr = np.asarray(bank.arrays[condition][layer][variable], dtype=np.float32)
                if arr.shape[0] != row_count:
                    raise RuntimeError(
                        f"State-bank array row count mismatch for {condition}/{layer}/{variable}: "
                        f"expected {row_count}, found {arr.shape[0]}"
                    )
                if layer == "layer3":
                    l3_key = f"{condition}_{variable}"
                    l3_payload[l3_key] = arr
                    state_rows.append(_state_manifest_row(network_seed, condition, layer, variable, arr, "state_bank_l3.npz", l3_key, ""))
                all_key = f"{condition}_{layer}_{variable}"
                all_payload[all_key] = arr
                state_rows.append(_state_manifest_row(network_seed, condition, layer, variable, arr, "state_bank_all_layers.npz", all_key, ""))
    l3_path = task_dir / "state_bank_l3.npz"
    all_path = task_dir / "state_bank_all_layers.npz"
    np.savez_compressed(l3_path, **l3_payload)
    np.savez_compressed(all_path, **all_payload)
    file_hashes = {
        "state_bank_l3.npz": sha256_file(l3_path),
        "state_bank_all_layers.npz": sha256_file(all_path),
    }
    for row in state_rows:
        row["sha256"] = file_hashes[str(row["storage_file"])]
    state_manifest = pd.DataFrame(state_rows, columns=list(STATE_BANK_MANIFEST_COLUMNS))
    state_manifest.to_csv(task_dir / "state_bank_manifest.csv", index=False, encoding="utf-8")

    boundary_rows = _save_condition_boundary_shards(
        task_dir / "boundary_states",
        bank.boundary_states,
        network_seed=network_seed,
        row_count=row_count,
    )
    boundary_manifest = pd.DataFrame(boundary_rows, columns=list(BOUNDARY_MANIFEST_COLUMNS))
    boundary_manifest.to_csv(task_dir / "boundary_manifest.csv", index=False, encoding="utf-8")
    write_json(_shape_payload(bank.layer_input_shapes), task_dir / "layer_input_shapes.json")
    write_cache_key(task_dir, cache_key)
    return StateBankArtifact(task_dir, bank, state_manifest, boundary_manifest)


def load_state_bank_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
    pair_trials: pd.DataFrame,
) -> StateBankArtifact:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id="state_bank")
    else:
        validate_cache_key_integrity(task_dir, task_id="state_bank")
    row_count = len(pair_trials)
    state_manifest_path = task_dir / "state_bank_manifest.csv"
    boundary_manifest_path = task_dir / "boundary_manifest.csv"
    shapes_path = task_dir / "layer_input_shapes.json"
    for path in (state_manifest_path, boundary_manifest_path, shapes_path):
        if not path.exists():
            raise FileNotFoundError(f"State-bank artifact required file is missing: {path}")
    state_manifest = pd.read_csv(state_manifest_path)
    _require_columns(state_manifest, STATE_BANK_MANIFEST_COLUMNS, state_manifest_path)
    _validate_manifest_hashes(task_dir, state_manifest, path_column="storage_file")
    arrays = _load_state_arrays(task_dir, state_manifest, row_count=row_count)
    boundary_manifest = pd.read_csv(boundary_manifest_path)
    _require_columns(boundary_manifest, BOUNDARY_MANIFEST_COLUMNS, boundary_manifest_path)
    _validate_manifest_hashes(task_dir, boundary_manifest, path_column="path")
    boundary_states = _load_condition_boundary_shards(task_dir, boundary_manifest, row_count=row_count)
    layer_input_shapes = {
        str(layer): tuple(int(v) for v in shape)
        for layer, shape in read_json(shapes_path).items()
    }
    bank = PairEpisodeStateBank(
        pair_trials=pair_trials.reset_index(drop=True).copy(),
        arrays=arrays,
        boundary_states=boundary_states,
        layer_input_shapes=layer_input_shapes,
        restore_mode="artifact_full_boundary",
        episode_end_step=0,
    )
    return StateBankArtifact(task_dir, bank, state_manifest, boundary_manifest)


def save_completion_boundary_bank_artifact(
    task_dir: Path,
    *,
    boundary_states_by_delay: Mapping[int, Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]]],
    layer_input_shapes_by_delay: Mapping[int, Mapping[str, tuple[int, ...]]],
    cache_key: Mapping[str, Any],
    network_seed: int,
    row_count: int,
) -> CompletionDelayBoundaryBank:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    rows: list[dict[str, Any]] = []
    for delay2_ms, states_by_condition in sorted(boundary_states_by_delay.items()):
        delay_root = task_dir / f"delay_{int(delay2_ms)}" / "boundary_states"
        boundary_rows = _save_condition_boundary_shards(
            delay_root,
            states_by_condition,
            network_seed=network_seed,
            row_count=row_count,
            delay2_ms=int(delay2_ms),
        )
        for row in boundary_rows:
            rows.append(
                {
                    "network_seed": row["network_seed"],
                    "delay2_ms": int(delay2_ms),
                    "state_condition": row["state_condition"],
                    "layer": row["layer"],
                    "state_key": row["state_key"],
                    "shape": row["shape"],
                    "path": str((Path(f"delay_{int(delay2_ms)}") / "boundary_states" / Path(row["path"]).name).as_posix()),
                    "sha256": row["sha256"],
                }
            )
    manifest = pd.DataFrame(rows, columns=list(COMPLETION_BOUNDARY_MANIFEST_COLUMNS))
    manifest.to_csv(task_dir / "completion_boundary_manifest.csv", index=False, encoding="utf-8")
    shape_payload = {
        str(int(delay)): _shape_payload(shapes)
        for delay, shapes in sorted(layer_input_shapes_by_delay.items())
    }
    write_json(shape_payload, task_dir / "layer_input_shapes.json")
    write_cache_key(task_dir, cache_key)
    return CompletionDelayBoundaryBank(
        root=task_dir,
        boundary_states_by_delay={
            int(delay): {str(cond): states for cond, states in by_condition.items()}
            for delay, by_condition in boundary_states_by_delay.items()
        },
        layer_input_shapes_by_delay={
            int(delay): {str(layer): tuple(int(v) for v in shape) for layer, shape in shapes.items()}
            for delay, shapes in layer_input_shapes_by_delay.items()
        },
        manifest=manifest,
    )


def load_completion_boundary_bank_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
    pair_trials: pd.DataFrame,
    expected_delays: list[int] | tuple[int, ...] | None = None,
) -> CompletionDelayBoundaryBank:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id="completion_delay_boundary_bank")
    manifest_path = task_dir / "completion_boundary_manifest.csv"
    shapes_path = task_dir / "layer_input_shapes.json"
    for path in (manifest_path, shapes_path):
        if not path.exists():
            raise FileNotFoundError(f"Completion boundary artifact required file is missing: {path}")
    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, COMPLETION_BOUNDARY_MANIFEST_COLUMNS, manifest_path)
    _validate_manifest_hashes(task_dir, manifest, path_column="path")
    found_delays = sorted(int(v) for v in manifest["delay2_ms"].dropna().unique().tolist())
    if expected_delays is not None:
        wanted = sorted(int(v) for v in expected_delays)
        if found_delays != wanted:
            raise RuntimeError(f"Completion boundary delays mismatch: expected={wanted}, found={found_delays}")
    row_count = len(pair_trials)
    boundary_states_by_delay: dict[int, dict[str, Mapping[str, Mapping[str, torch.Tensor]]]] = {}
    for delay2_ms in found_delays:
        part = manifest[manifest["delay2_ms"].astype(int).eq(int(delay2_ms))]
        boundary_states_by_delay[int(delay2_ms)] = _load_condition_boundary_shards(
            task_dir,
            part,
            row_count=row_count,
            conditions=COMPLETION_CONDITIONS,
        )
    shapes_payload = read_json(shapes_path)
    layer_input_shapes_by_delay = {
        int(delay): {str(layer): tuple(int(v) for v in shape) for layer, shape in shapes.items()}
        for delay, shapes in shapes_payload.items()
    }
    return CompletionDelayBoundaryBank(task_dir, boundary_states_by_delay, layer_input_shapes_by_delay, manifest)


def save_partial_cue_mask_specs_artifact(
    task_dir: Path,
    mask_specs: pd.DataFrame,
    *,
    cache_key: Mapping[str, Any],
) -> TableArtifact:
    return _save_single_table_artifact(
        task_dir,
        table=mask_specs.loc[:, list(WEAK_PROBE_MASK_COLUMNS)].copy(),
        filename="weak_probe_masks.csv",
        name="weak_probe_masks",
        columns=WEAK_PROBE_MASK_COLUMNS,
        cache_key=cache_key,
    )


def load_partial_cue_mask_specs_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
) -> TableArtifact:
    return _load_single_table_artifact(
        task_dir,
        filename="weak_probe_masks.csv",
        name="weak_probe_masks",
        columns=WEAK_PROBE_MASK_COLUMNS,
        expected_key=expected_key,
        task_id="partial_cue_mask_specs",
    )


def save_completion_delay_mask_specs_artifact(
    task_dir: Path,
    mask_specs: pd.DataFrame,
    *,
    cache_key: Mapping[str, Any],
) -> TableArtifact:
    return _save_single_table_artifact(
        task_dir,
        table=mask_specs.loc[:, list(COMPLETION_DELAY_MASK_COLUMNS)].copy(),
        filename="completion_delay_mask_specs.csv",
        name="completion_delay_mask_specs",
        columns=COMPLETION_DELAY_MASK_COLUMNS,
        cache_key=cache_key,
    )


def load_completion_delay_mask_specs_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
) -> TableArtifact:
    return _load_single_table_artifact(
        task_dir,
        filename="completion_delay_mask_specs.csv",
        name="completion_delay_mask_specs",
        columns=COMPLETION_DELAY_MASK_COLUMNS,
        expected_key=expected_key,
        task_id="completion_delay_mask_specs",
    )


def _save_single_table_artifact(
    task_dir: Path,
    *,
    table: pd.DataFrame,
    filename: str,
    name: str,
    columns: tuple[str, ...],
    cache_key: Mapping[str, Any],
) -> TableArtifact:
    table = table.loc[:, list(columns)].copy()
    artifact = _save_table_bundle(
        task_dir,
        tables={name: table},
        files={name: filename},
        manifest_columns=TABLE_MANIFEST_COLUMNS,
        cache_key=cache_key,
    )
    return TableArtifact(artifact.root, table, artifact.manifest, artifact.digest)


def _load_single_table_artifact(
    task_dir: Path,
    *,
    filename: str,
    name: str,
    columns: tuple[str, ...],
    expected_key: Mapping[str, Any] | None,
    task_id: str,
) -> TableArtifact:
    artifact = _load_table_bundle(
        task_dir,
        files={name: filename},
        manifest_columns=TABLE_MANIFEST_COLUMNS,
        expected_key=expected_key,
        task_id=task_id,
    )
    table = artifact.table[name]
    if list(table.columns) != list(columns):
        raise RuntimeError(f"{task_id} columns mismatch: expected={list(columns)}, found={list(table.columns)}")
    return TableArtifact(artifact.root, table, artifact.manifest, artifact.digest)


def _save_table_bundle(
    task_dir: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    files: Mapping[str, str],
    manifest_columns: tuple[str, ...],
    cache_key: Mapping[str, Any],
) -> TableArtifact:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    missing = sorted(set(files).difference(tables))
    if missing:
        raise KeyError(f"Cannot save table artifact; missing tables: {missing}")
    written: dict[str, tuple[Path, pd.DataFrame, str]] = {}
    for name, filename in files.items():
        df = tables[name].copy()
        path = task_dir / filename
        df.to_csv(path, index=False, encoding="utf-8")
        written[name] = (path, pd.read_csv(path), sha256_file(path))
    digest = table_digest({name: written[name][1] for name in files})
    rows: list[dict[str, Any]] = []
    for name, filename in files.items():
        path, df, file_hash = written[name]
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
    manifest = pd.DataFrame(rows, columns=list(manifest_columns))
    manifest.to_csv(task_dir / "manifest.csv", index=False, encoding="utf-8")
    write_cache_key(task_dir, cache_key)
    table_payload = {name: written[name][1].copy() for name in files}
    return TableArtifact(task_dir, table_payload, manifest, digest)  # type: ignore[arg-type]


def _load_table_bundle(
    task_dir: Path,
    *,
    files: Mapping[str, str],
    manifest_columns: tuple[str, ...],
    expected_key: Mapping[str, Any] | None,
    task_id: str,
) -> TableArtifact:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id=task_id)
    else:
        validate_cache_key_integrity(task_dir, task_id=task_id)
    manifest_path = task_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{task_id} manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, manifest_columns, manifest_path)
    found_names = set(str(value) for value in manifest["name"].tolist())
    expected_names = set(files)
    if found_names != expected_names:
        raise RuntimeError(f"{task_id} manifest names mismatch: expected={sorted(expected_names)}, found={sorted(found_names)}")
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in files.items():
        rec = _table_manifest_record(manifest, name, filename, manifest_path)
        path = task_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{task_id} artifact file is missing: {path}")
        found_sha = sha256_file(path)
        expected_sha = str(rec["sha256"])
        if found_sha != expected_sha:
            raise RuntimeError(f"{task_id} file hash mismatch for {filename}: expected {expected_sha}, found {found_sha}")
        df = pd.read_csv(path)
        expected_rows = int(rec["rows"])
        if len(df) != expected_rows:
            raise RuntimeError(f"{task_id} row count mismatch for {filename}: expected {expected_rows}, found {len(df)}")
        expected_columns = _parse_manifest_columns(rec["columns"], manifest_path)
        if list(df.columns) != expected_columns:
            raise RuntimeError(f"{task_id} column mismatch for {filename}: expected {expected_columns}, found {list(df.columns)}")
        tables[name] = df
    digest = table_digest(tables)
    manifest_digests = {str(value) for value in manifest["table_digest"].dropna().astype(str).tolist()}
    if manifest_digests != {digest}:
        raise RuntimeError(f"{task_id} table digest mismatch: manifest={sorted(manifest_digests)}, computed={digest}")
    return TableArtifact(task_dir, tables, manifest, digest)  # type: ignore[arg-type]


def _state_manifest_row(
    network_seed: int,
    condition: str,
    layer: str,
    variable: str,
    arr: np.ndarray,
    storage_file: str,
    storage_key: str,
    sha256: str,
) -> dict[str, Any]:
    return {
        "network_seed": int(network_seed),
        "state_condition": str(condition),
        "layer": str(layer),
        "state_variable": str(variable),
        "shape": "x".join(str(int(v)) for v in arr.shape),
        "storage_file": str(storage_file),
        "storage_key": str(storage_key),
        "sha256": str(sha256),
    }


def _save_condition_boundary_shards(
    boundary_root: Path,
    boundary_states: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    *,
    network_seed: int,
    row_count: int,
    delay2_ms: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, boundary in sorted(boundary_states.items()):
        path = boundary_root / f"condition={condition}.npz"
        arrays: dict[str, np.ndarray] = {}
        for layer in LAYER_KEYS:
            if layer not in boundary:
                continue
            for state_key in BOUNDARY_STATE_KEYS:
                if state_key not in boundary[layer]:
                    continue
                arr = boundary[layer][state_key].detach().cpu().numpy()
                if arr.shape[0] != int(row_count):
                    raise RuntimeError(
                        f"Boundary row count mismatch for {condition}/{layer}/{state_key}: "
                        f"expected {row_count}, found {arr.shape[0]}"
                    )
                arrays[f"{layer}__{state_key}"] = arr
        if not arrays:
            raise ValueError(f"No boundary arrays to save for condition={condition}")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        file_hash = sha256_file(path)
        rel = path.relative_to(boundary_root.parent).as_posix() if delay2_ms is None else path.name
        for key, arr in arrays.items():
            layer, state_key = key.split("__", 1)
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "state_condition": str(condition),
                    "layer": str(layer),
                    "state_key": str(state_key),
                    "shape": "x".join(str(int(v)) for v in arr.shape),
                    "path": rel,
                    "sha256": file_hash,
                }
            )
    return rows


def _load_condition_boundary_shards(
    task_dir: Path,
    manifest: pd.DataFrame,
    *,
    row_count: int,
    conditions: tuple[str, ...] | None = None,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    expected_conditions = set(conditions or tuple(str(v) for v in manifest["state_condition"].unique()))
    found_conditions = set(str(v) for v in manifest["state_condition"].unique())
    if found_conditions != expected_conditions:
        raise RuntimeError(f"Boundary condition mismatch: expected={sorted(expected_conditions)}, found={sorted(found_conditions)}")
    out: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for condition in sorted(expected_conditions):
        part = manifest[manifest["state_condition"].astype(str).eq(condition)]
        paths = sorted(set(str(v) for v in part["path"].tolist()))
        if len(paths) != 1:
            raise RuntimeError(f"Boundary manifest expected one shard for {condition}, found {paths}")
        path = Path(task_dir) / paths[0]
        if not path.exists():
            raise FileNotFoundError(f"Boundary shard is missing: {path}")
        condition_state: dict[str, dict[str, torch.Tensor]] = {}
        with np.load(path, allow_pickle=False) as payload:
            expected_keys = {f"{row['layer']}__{row['state_key']}" for row in part.to_dict("records")}
            if set(payload.files) != expected_keys:
                raise RuntimeError(f"Boundary shard keys mismatch for {path}: expected={sorted(expected_keys)}, found={sorted(payload.files)}")
            for key in payload.files:
                layer, state_key = key.split("__", 1)
                arr = np.array(payload[key])
                if arr.shape[0] != int(row_count):
                    raise RuntimeError(f"Boundary shard row count mismatch for {path}/{key}: expected {row_count}, found {arr.shape[0]}")
                condition_state.setdefault(layer, {})[state_key] = torch.from_numpy(arr)
        out[condition] = condition_state
    return out


def _load_state_arrays(task_dir: Path, manifest: pd.DataFrame, *, row_count: int) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    npz_cache: dict[str, Any] = {}
    try:
        for row in manifest.to_dict("records"):
            condition = str(row["state_condition"])
            layer = str(row["layer"])
            variable = str(row["state_variable"])
            storage_file = str(row["storage_file"])
            storage_key = str(row["storage_key"])
            if storage_file not in npz_cache:
                path = Path(task_dir) / storage_file
                if not path.exists():
                    raise FileNotFoundError(f"State-bank NPZ is missing: {path}")
                npz_cache[storage_file] = np.load(path, allow_pickle=False)
            payload = npz_cache[storage_file]
            if storage_key not in payload.files:
                raise RuntimeError(f"State-bank NPZ key {storage_key!r} is missing from {storage_file}")
            arr = np.asarray(payload[storage_key], dtype=np.float32)
            if arr.shape[0] != int(row_count):
                raise RuntimeError(f"State-bank row count mismatch for {storage_file}/{storage_key}: expected {row_count}, found {arr.shape[0]}")
            arrays.setdefault(condition, {}).setdefault(layer, {})[variable] = arr
    finally:
        for payload in npz_cache.values():
            payload.close()
    for condition, by_layer in arrays.items():
        for layer in LAYER_KEYS:
            if layer not in by_layer:
                raise RuntimeError(f"State-bank artifact is missing layer {layer} for condition {condition}")
            layer_arrays = by_layer[layer]
            if not set(STATE_BANK_ARRAY_VARIABLES).issubset(layer_arrays):
                missing = sorted(set(STATE_BANK_ARRAY_VARIABLES).difference(layer_arrays))
                raise RuntimeError(f"State-bank artifact missing variables for {condition}/{layer}: {missing}")
            layer_arrays["ux_concat"] = np.concatenate([layer_arrays["u"], layer_arrays["x"]], axis=1).astype(np.float32, copy=False)
    return arrays


def _validate_manifest_hashes(task_dir: Path, manifest: pd.DataFrame, *, path_column: str) -> None:
    for rel_path, part in manifest.groupby(path_column, sort=False):
        path = Path(task_dir) / str(rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact file is missing: {path}")
        found = sha256_file(path)
        expected = {str(v) for v in part["sha256"].astype(str).tolist()}
        if expected != {found}:
            raise RuntimeError(f"Artifact hash mismatch for {path}: expected={sorted(expected)}, found={found}")


def _table_manifest_record(manifest: pd.DataFrame, name: str, filename: str, manifest_path: Path) -> dict[str, Any]:
    rows = manifest[
        manifest["name"].astype(str).eq(str(name))
        & manifest["filename"].astype(str).eq(str(filename))
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Table manifest expected one row for {name}/{filename}, found {len(rows)}: {manifest_path}")
    return rows.iloc[0].to_dict()


def _parse_manifest_columns(value: Any, manifest_path: Path) -> list[str]:
    try:
        columns = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest has malformed columns JSON in {manifest_path}: {value!r}") from exc
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise ValueError(f"Manifest columns must be a JSON string list in {manifest_path}: {value!r}")
    return columns


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")


def _shape_payload(shapes: Mapping[str, tuple[int, ...]]) -> dict[str, list[int]]:
    return {str(layer): [int(v) for v in shape] for layer, shape in shapes.items()}


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
    "CompletionDelayBoundaryBank",
    "PairTrialSpecsArtifact",
    "StateBankArtifact",
    "TableArtifact",
    "cache_key_matches",
    "default_artifact_root",
    "load_completion_boundary_bank_artifact",
    "load_completion_delay_mask_specs_artifact",
    "load_crossfit_split_specs_artifact",
    "load_crossfit_null_specs_artifact",
    "load_pair_trial_specs_artifact",
    "load_partial_cue_mask_specs_artifact",
    "load_state_bank_artifact",
    "read_cache_key",
    "read_json",
    "reset_task_artifact_dir",
    "save_completion_boundary_bank_artifact",
    "save_completion_delay_mask_specs_artifact",
    "save_crossfit_split_specs_artifact",
    "save_crossfit_null_specs_artifact",
    "save_pair_trial_specs_artifact",
    "save_partial_cue_mask_specs_artifact",
    "save_state_bank_artifact",
    "task_artifact_dir",
    "validate_cache_key_integrity",
    "write_cache_key",
    "write_json",
]
