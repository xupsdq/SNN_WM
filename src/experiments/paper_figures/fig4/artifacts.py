from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig4.cache_keys import (
    cache_key_digest,
    pair_sampling_hash,
    sha256_file,
    table_digest,
)
from src.experiments.paper_figures.fig4.schemas import (
    ARRAY_MANIFEST_COLUMNS,
    PAIR_MASK_MANIFEST_COLUMNS,
    PAIR_SAMPLING_FILES,
    ROLLOUT_BANK_FILES,
    SIMILARITY_ENTRY_FILES,
    TABLE_MANIFEST_COLUMNS,
)
from src.experiments.paper_figures.fig4.types import (
    OverlapPerturbationCompatibleBank,
    SimilarityBiasCompatibleBank,
)


CACHE_KEY_FILE = "cache_key.json"


@dataclass(frozen=True)
class PairSamplingArtifact:
    root: Path
    pair_trials: pd.DataFrame
    candidate_pool: pd.DataFrame
    overlap_matched_pairs: pd.DataFrame
    perturbation_masks: pd.DataFrame
    mask_bank: dict[int, dict[str, np.ndarray]]
    table_manifest: pd.DataFrame
    mask_manifest: pd.DataFrame
    digest: str


@dataclass(frozen=True)
class SimilarityEntryArtifact:
    root: Path
    bank: SimilarityBiasCompatibleBank
    table_manifest: pd.DataFrame
    array_manifest: pd.DataFrame
    digest: str


@dataclass(frozen=True)
class RolloutBankArtifact:
    root: Path
    bank: OverlapPerturbationCompatibleBank
    table_manifest: pd.DataFrame
    array_manifest: pd.DataFrame
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
    payload = {
        "cache_key": _json_safe(cache_key),
        "cache_key_digest": cache_key_digest(cache_key),
    }
    write_json(payload, Path(task_dir) / CACHE_KEY_FILE)


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
            f"Fig.4 {task_id} artifact cache key mismatch: expected {expected_digest}, found {found_digest}. "
            "Rebuild the producer task before using --reuse-artifacts require."
        )


def save_pair_sampling_artifact(
    task_dir: Path,
    *,
    pair_trials: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    overlap_matched_pairs: pd.DataFrame,
    perturbation_masks: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
    cache_key: Mapping[str, Any],
    network_seed: int,
) -> PairSamplingArtifact:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    table_bundle = _save_table_bundle(
        task_dir,
        tables={
            "pair_trials": pair_trials.copy(),
            "pair_candidate_pool": candidate_pool.copy(),
            "overlap_matched_pairs": overlap_matched_pairs.copy(),
            "perturbation_masks": perturbation_masks.copy(),
        },
        files=PAIR_SAMPLING_FILES,
        keep_default_na=False,
    )
    persisted_pair_trials = table_bundle.tables["pair_trials"]
    persisted_candidate_pool = table_bundle.tables["pair_candidate_pool"]
    persisted_overlap_matched = table_bundle.tables["overlap_matched_pairs"]
    persisted_perturbation_masks = table_bundle.tables["perturbation_masks"]
    persisted_masks, mask_manifest = _save_mask_bank(
        task_dir,
        mask_bank,
        network_seed=network_seed,
        pair_trials=persisted_pair_trials,
        perturbation_masks=persisted_perturbation_masks,
    )
    digest = pair_sampling_hash(
        persisted_pair_trials,
        persisted_candidate_pool,
        persisted_overlap_matched,
        persisted_perturbation_masks,
        persisted_masks,
    )
    write_json({"artifact_digest": digest}, task_dir / "artifact_digest.json")
    write_cache_key(task_dir, cache_key)
    return PairSamplingArtifact(
        root=task_dir,
        pair_trials=persisted_pair_trials,
        candidate_pool=persisted_candidate_pool,
        overlap_matched_pairs=persisted_overlap_matched,
        perturbation_masks=persisted_perturbation_masks,
        mask_bank=persisted_masks,
        table_manifest=table_bundle.manifest,
        mask_manifest=mask_manifest,
        digest=digest,
    )


def load_pair_sampling_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
) -> PairSamplingArtifact:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id="pair_sampling")
    table_bundle = _load_table_bundle(
        task_dir,
        files=PAIR_SAMPLING_FILES,
        task_id="pair_sampling",
        keep_default_na=False,
    )
    pair_trials = table_bundle.tables["pair_trials"]
    candidate_pool = table_bundle.tables["pair_candidate_pool"]
    overlap_matched = table_bundle.tables["overlap_matched_pairs"]
    perturbation_masks = table_bundle.tables["perturbation_masks"]
    mask_bank, mask_manifest = _load_mask_bank(task_dir, pair_trials=pair_trials, perturbation_masks=perturbation_masks)
    digest = pair_sampling_hash(pair_trials, candidate_pool, overlap_matched, perturbation_masks, mask_bank)
    recorded = read_json(task_dir / "artifact_digest.json").get("artifact_digest")
    if str(recorded) != digest:
        raise RuntimeError(f"Fig.4 pair_sampling artifact digest mismatch: expected {recorded}, found {digest}")
    return PairSamplingArtifact(
        root=task_dir,
        pair_trials=pair_trials,
        candidate_pool=candidate_pool,
        overlap_matched_pairs=overlap_matched,
        perturbation_masks=perturbation_masks,
        mask_bank=mask_bank,
        table_manifest=table_bundle.manifest,
        mask_manifest=mask_manifest,
        digest=digest,
    )


def save_similarity_entry_artifact(
    task_dir: Path,
    bank: SimilarityBiasCompatibleBank,
    *,
    cache_key: Mapping[str, Any],
) -> SimilarityEntryArtifact:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    table_bundle = _save_table_bundle(
        task_dir,
        tables={
            "trial_metrics": bank.trial_metrics.copy(),
            "repeat_metrics": bank.repeat_metrics.copy(),
        },
        files=SIMILARITY_ENTRY_FILES,
        keep_default_na=True,
    )
    voltage_vectors = {str(key): np.asarray(value) for key, value in bank.voltage_vectors.items()}
    array_manifest = _save_npz_files(task_dir, {"voltage_vectors.npz": voltage_vectors})
    _validate_similarity_payload(table_bundle.tables["trial_metrics"], voltage_vectors)
    digest = _artifact_digest(table_bundle.digest, array_manifest)
    write_json({"artifact_digest": digest}, task_dir / "artifact_digest.json")
    write_cache_key(task_dir, cache_key)
    artifact_bank = SimilarityBiasCompatibleBank(
        bank.pair_trials.reset_index(drop=True).copy(),
        table_bundle.tables["trial_metrics"],
        table_bundle.tables["repeat_metrics"],
        voltage_vectors,
    )
    return SimilarityEntryArtifact(task_dir, artifact_bank, table_bundle.manifest, array_manifest, digest)


def load_similarity_entry_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
    pair_trials: pd.DataFrame,
) -> SimilarityEntryArtifact:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id="similarity_entry")
    table_bundle = _load_table_bundle(
        task_dir,
        files=SIMILARITY_ENTRY_FILES,
        task_id="similarity_entry",
        keep_default_na=True,
    )
    arrays = _load_npz_files(task_dir, task_id="similarity_entry", required_files=("voltage_vectors.npz",))
    voltage_vectors = arrays["voltage_vectors.npz"]
    _validate_similarity_payload(table_bundle.tables["trial_metrics"], voltage_vectors)
    _require_pair_membership(table_bundle.tables["trial_metrics"], pair_trials, task_id="similarity_entry")
    digest = _artifact_digest(table_bundle.digest, _read_array_manifest(task_dir))
    recorded = read_json(task_dir / "artifact_digest.json").get("artifact_digest")
    if str(recorded) != digest:
        raise RuntimeError(f"Fig.4 similarity_entry artifact digest mismatch: expected {recorded}, found {digest}")
    bank = SimilarityBiasCompatibleBank(
        pair_trials.reset_index(drop=True).copy(),
        table_bundle.tables["trial_metrics"],
        table_bundle.tables["repeat_metrics"],
        voltage_vectors,
    )
    return SimilarityEntryArtifact(task_dir, bank, table_bundle.manifest, _read_array_manifest(task_dir), digest)


def save_rollout_bank_artifact(
    task_dir: Path,
    bank: OverlapPerturbationCompatibleBank,
    *,
    raw_dir: Path,
    cache_key: Mapping[str, Any],
) -> RolloutBankArtifact:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    table_bundle = _save_table_bundle(
        task_dir,
        tables={
            "rollout_manifest": bank.rollout_manifest.copy(),
            "condition_metrics": bank.condition_metrics.copy(),
            "perturbation_masks": bank.perturbation_masks.copy(),
            "l3_replay_capture_manifest": bank.l3_replay_capture_manifest.copy(),
        },
        files=ROLLOUT_BANK_FILES,
        keep_default_na=True,
    )
    npz_payloads = _copy_rollout_npz_payloads(Path(raw_dir))
    array_manifest = _save_npz_files(task_dir, npz_payloads)
    _validate_rollout_payload(
        table_bundle.tables["condition_metrics"],
        npz_payloads,
        capture_manifest=table_bundle.tables["l3_replay_capture_manifest"],
    )
    digest = _artifact_digest(table_bundle.digest, array_manifest)
    write_json({"artifact_digest": digest}, task_dir / "artifact_digest.json")
    write_cache_key(task_dir, cache_key)
    artifact_bank = OverlapPerturbationCompatibleBank(
        bank.pair_trials.reset_index(drop=True).copy(),
        table_bundle.tables["perturbation_masks"],
        table_bundle.tables["rollout_manifest"],
        table_bundle.tables["condition_metrics"],
        npz_payloads.get("probe_trace_arrays_l3.npz", {}),
        npz_payloads.get("readout_trajectory_vectors.npz", {}),
        table_bundle.tables["l3_replay_capture_manifest"],
        npz_payloads.get("l3_replay_capture_arrays.npz", {}),
    )
    return RolloutBankArtifact(task_dir, artifact_bank, table_bundle.manifest, array_manifest, digest)


def load_rollout_bank_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
    pair_trials: pd.DataFrame,
) -> RolloutBankArtifact:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id="rollouts")
    table_bundle = _load_table_bundle(
        task_dir,
        files=ROLLOUT_BANK_FILES,
        task_id="rollouts",
        keep_default_na=True,
    )
    arrays = _load_npz_files(
        task_dir,
        task_id="rollouts",
        required_files=("probe_trace_arrays_l3.npz", "readout_trajectory_vectors.npz", "l3_replay_capture_arrays.npz"),
    )
    condition_metrics = table_bundle.tables["condition_metrics"]
    _validate_rollout_payload(
        condition_metrics,
        arrays,
        capture_manifest=table_bundle.tables["l3_replay_capture_manifest"],
    )
    _require_pair_membership(condition_metrics, pair_trials, task_id="rollouts")
    digest = _artifact_digest(table_bundle.digest, _read_array_manifest(task_dir))
    recorded = read_json(task_dir / "artifact_digest.json").get("artifact_digest")
    if str(recorded) != digest:
        raise RuntimeError(f"Fig.4 rollouts artifact digest mismatch: expected {recorded}, found {digest}")
    bank = OverlapPerturbationCompatibleBank(
        pair_trials.reset_index(drop=True).copy(),
        table_bundle.tables["perturbation_masks"],
        table_bundle.tables["rollout_manifest"],
        condition_metrics,
        arrays.get("probe_trace_arrays_l3.npz", {}),
        arrays.get("readout_trajectory_vectors.npz", {}),
        table_bundle.tables["l3_replay_capture_manifest"],
        arrays.get("l3_replay_capture_arrays.npz", {}),
    )
    return RolloutBankArtifact(task_dir, bank, table_bundle.manifest, _read_array_manifest(task_dir), digest)


def copy_rollout_artifact_npz_to_raw(task_dir: Path, raw_dir: Path) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "probe_trace_arrays_l1.npz",
        "probe_trace_arrays_l2.npz",
        "probe_trace_arrays_l3.npz",
        "readout_trajectory_vectors.npz",
        "l3_replay_capture_arrays.npz",
    ):
        src = Path(task_dir) / filename
        if src.exists():
            shutil.copy2(src, raw_dir / filename)


def _save_mask_bank(
    task_dir: Path,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
    *,
    network_seed: int,
    pair_trials: pd.DataFrame,
    perturbation_masks: pd.DataFrame,
) -> tuple[dict[int, dict[str, np.ndarray]], pd.DataFrame]:
    expected_pair_ids = {int(value) for value in pair_trials["pair_id"].tolist()}
    found_pair_ids = {int(value) for value in mask_bank}
    if found_pair_ids != expected_pair_ids:
        raise RuntimeError(f"Fig.4 mask bank pair mismatch: expected={sorted(expected_pair_ids)}, found={sorted(found_pair_ids)}")
    payload: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    persisted: dict[int, dict[str, np.ndarray]] = {}
    for pair_id in sorted(found_pair_ids):
        persisted[pair_id] = {}
        for mask_name in sorted(str(value) for value in mask_bank[pair_id]):
            arr = np.asarray(mask_bank[pair_id][mask_name], dtype=bool)
            key = _mask_storage_key(pair_id, mask_name)
            payload[key] = arr
            persisted[pair_id][mask_name] = arr
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "pair_id": int(pair_id),
                    "mask_name": str(mask_name),
                    "shape": _shape_text(arr.shape),
                    "storage_file": "masks.npz",
                    "storage_key": key,
                    "sha256": "",
                }
            )
    path = task_dir / "masks.npz"
    np.savez_compressed(path, **payload)
    file_hash = sha256_file(path)
    for row in rows:
        row["sha256"] = file_hash
    manifest = pd.DataFrame(rows, columns=list(PAIR_MASK_MANIFEST_COLUMNS))
    manifest.to_csv(task_dir / "mask_manifest.csv", index=False, encoding="utf-8")
    _validate_mask_membership(manifest, perturbation_masks)
    return persisted, manifest


def _load_mask_bank(
    task_dir: Path,
    *,
    pair_trials: pd.DataFrame,
    perturbation_masks: pd.DataFrame,
) -> tuple[dict[int, dict[str, np.ndarray]], pd.DataFrame]:
    manifest_path = Path(task_dir) / "mask_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fig.4 pair_sampling mask manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, PAIR_MASK_MANIFEST_COLUMNS, manifest_path)
    _validate_manifest_hashes(task_dir, manifest, path_column="storage_file")
    _validate_mask_membership(manifest, perturbation_masks)
    expected_pair_ids = {int(value) for value in pair_trials["pair_id"].tolist()}
    bank: dict[int, dict[str, np.ndarray]] = {}
    npz_cache: dict[str, Any] = {}
    try:
        for row in manifest.to_dict("records"):
            pair_id = int(row["pair_id"])
            if pair_id not in expected_pair_ids:
                raise RuntimeError(f"Fig.4 mask manifest contains unknown pair_id={pair_id}")
            filename = str(row["storage_file"])
            key = str(row["storage_key"])
            if filename not in npz_cache:
                path = Path(task_dir) / filename
                if not path.exists():
                    raise FileNotFoundError(f"Fig.4 mask artifact file is missing: {path}")
                npz_cache[filename] = np.load(path, allow_pickle=False)
            payload = npz_cache[filename]
            if key not in payload.files:
                raise RuntimeError(f"Fig.4 mask key {key!r} is missing from {filename}")
            arr = np.asarray(payload[key], dtype=bool)
            if _shape_text(arr.shape) != str(row["shape"]):
                raise RuntimeError(f"Fig.4 mask shape mismatch for {key}: expected={row['shape']}, found={_shape_text(arr.shape)}")
            bank.setdefault(pair_id, {})[str(row["mask_name"])] = arr
    finally:
        for payload in npz_cache.values():
            payload.close()
    if set(bank) != expected_pair_ids:
        raise RuntimeError(f"Fig.4 loaded mask pair mismatch: expected={sorted(expected_pair_ids)}, found={sorted(bank)}")
    return bank, manifest


def _validate_mask_membership(manifest: pd.DataFrame, perturbation_masks: pd.DataFrame) -> None:
    expected = {
        (int(row["pair_id"]), str(row["mask_name"]))
        for row in perturbation_masks.loc[:, ["pair_id", "mask_name"]].to_dict("records")
    }
    found = {
        (int(row["pair_id"]), str(row["mask_name"]))
        for row in manifest.loc[:, ["pair_id", "mask_name"]].to_dict("records")
    }
    if not expected.issubset(found):
        missing = sorted(expected.difference(found))[:10]
        raise RuntimeError(f"Fig.4 mask manifest missing perturbation-mask entries: {missing}")


def _save_table_bundle(
    task_dir: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    files: Mapping[str, str],
    keep_default_na: bool,
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
        read_back = pd.read_csv(path, keep_default_na=keep_default_na)
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


def _load_table_bundle(
    task_dir: Path,
    *,
    files: Mapping[str, str],
    task_id: str,
    keep_default_na: bool,
) -> TableBundle:
    task_dir = Path(task_dir)
    manifest_path = task_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fig.4 {task_id} manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, TABLE_MANIFEST_COLUMNS, manifest_path)
    found_names = set(str(value) for value in manifest["name"].tolist())
    expected_names = set(files)
    if found_names != expected_names:
        raise RuntimeError(f"Fig.4 {task_id} manifest names mismatch: expected={sorted(expected_names)}, found={sorted(found_names)}")
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in files.items():
        rec = _table_manifest_record(manifest, name, filename, manifest_path)
        path = task_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Fig.4 {task_id} artifact file is missing: {path}")
        found_sha = sha256_file(path)
        expected_sha = str(rec["sha256"])
        if found_sha != expected_sha:
            raise RuntimeError(f"Fig.4 {task_id} file hash mismatch for {filename}: expected {expected_sha}, found {found_sha}")
        df = pd.read_csv(path, keep_default_na=keep_default_na)
        expected_rows = int(rec["rows"])
        if len(df) != expected_rows:
            raise RuntimeError(f"Fig.4 {task_id} row count mismatch for {filename}: expected {expected_rows}, found {len(df)}")
        expected_columns = _parse_manifest_columns(rec["columns"], manifest_path)
        if list(df.columns) != expected_columns:
            raise RuntimeError(f"Fig.4 {task_id} column mismatch for {filename}: expected {expected_columns}, found {list(df.columns)}")
        tables[name] = df
    digest = table_digest(tables)
    manifest_digests = {str(value) for value in manifest["table_digest"].dropna().astype(str).tolist()}
    if manifest_digests != {digest}:
        raise RuntimeError(f"Fig.4 {task_id} table digest mismatch: manifest={sorted(manifest_digests)}, computed={digest}")
    return TableBundle(task_dir, tables, manifest, digest)


def _save_npz_files(task_dir: Path, files: Mapping[str, Mapping[str, np.ndarray]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for filename, payload in files.items():
        path = Path(task_dir) / filename
        safe_payload = {str(key): np.asarray(value) for key, value in payload.items()}
        np.savez_compressed(path, **safe_payload)
        file_hash = sha256_file(path)
        for key, arr in safe_payload.items():
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


def _load_npz_files(task_dir: Path, *, task_id: str, required_files: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    manifest = _read_array_manifest(task_dir)
    _validate_manifest_hashes(task_dir, manifest, path_column="storage_file")
    found_files = set(str(value) for value in manifest["storage_file"].tolist())
    missing = sorted(set(required_files).difference(found_files))
    if missing:
        raise RuntimeError(f"Fig.4 {task_id} array manifest missing required files: {missing}")
    out: dict[str, dict[str, np.ndarray]] = {}
    for filename in sorted(found_files):
        path = Path(task_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"Fig.4 {task_id} NPZ artifact file is missing: {path}")
        part = manifest[manifest["storage_file"].astype(str).eq(filename)]
        with np.load(path, allow_pickle=False) as payload:
            expected_keys = set(str(value) for value in part["storage_key"].tolist())
            if set(payload.files) != expected_keys:
                raise RuntimeError(f"Fig.4 {task_id} NPZ key mismatch for {filename}: expected={sorted(expected_keys)}, found={sorted(payload.files)}")
            out[filename] = {}
            for row in part.to_dict("records"):
                key = str(row["storage_key"])
                arr = np.array(payload[key])
                if _shape_text(arr.shape) != str(row["shape"]):
                    raise RuntimeError(f"Fig.4 {task_id} shape mismatch for {filename}/{key}: expected={row['shape']}, found={_shape_text(arr.shape)}")
                out[filename][key] = arr
    return out


def _read_array_manifest(task_dir: Path) -> pd.DataFrame:
    manifest_path = Path(task_dir) / "array_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fig.4 array manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, ARRAY_MANIFEST_COLUMNS, manifest_path)
    return manifest


def _copy_rollout_npz_payloads(raw_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    payloads: dict[str, dict[str, np.ndarray]] = {}
    for filename in (
        "probe_trace_arrays_l1.npz",
        "probe_trace_arrays_l2.npz",
        "probe_trace_arrays_l3.npz",
        "readout_trajectory_vectors.npz",
        "l3_replay_capture_arrays.npz",
    ):
        path = Path(raw_dir) / filename
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as payload:
            payloads[filename] = {str(key): np.array(payload[key]) for key in payload.files}
    if "probe_trace_arrays_l3.npz" not in payloads:
        raise FileNotFoundError(f"Fig.4 rollout artifact cannot find probe_trace_arrays_l3.npz in {raw_dir}")
    if "readout_trajectory_vectors.npz" not in payloads:
        raise FileNotFoundError(f"Fig.4 rollout artifact cannot find readout_trajectory_vectors.npz in {raw_dir}")
    return payloads


def _validate_similarity_payload(trial_metrics: pd.DataFrame, voltage_vectors: Mapping[str, np.ndarray]) -> None:
    required = {"pair_id", "voltage_dynamic", "voltage_static"}
    missing = sorted(required.difference(voltage_vectors))
    if missing:
        raise RuntimeError(f"Fig.4 similarity_entry voltage vectors missing keys: {missing}")
    pair_ids = np.asarray(voltage_vectors["pair_id"], dtype=np.int64)
    if len(trial_metrics) != len(pair_ids):
        raise RuntimeError(f"Fig.4 similarity_entry row count mismatch: table={len(trial_metrics)}, vectors={len(pair_ids)}")
    table_pair_ids = trial_metrics["pair_id"].to_numpy(dtype=np.int64, copy=False) if "pair_id" in trial_metrics.columns else np.zeros(0, dtype=np.int64)
    if not np.array_equal(table_pair_ids, pair_ids):
        raise RuntimeError("Fig.4 similarity_entry pair_id order mismatch between table and voltage vectors.")
    for key in ("voltage_dynamic", "voltage_static"):
        arr = np.asarray(voltage_vectors[key])
        if arr.shape[0] != len(pair_ids):
            raise RuntimeError(f"Fig.4 similarity_entry {key} row count mismatch: expected={len(pair_ids)}, found={arr.shape[0]}")


def _validate_rollout_payload(
    condition_metrics: pd.DataFrame,
    payloads: Mapping[str, Mapping[str, np.ndarray]],
    *,
    capture_manifest: pd.DataFrame | None = None,
) -> None:
    if condition_metrics.empty:
        raise RuntimeError("Fig.4 rollouts artifact condition_metrics is empty.")
    pair_ids = sorted(int(value) for value in condition_metrics["pair_id"].dropna().unique().tolist())
    conditions = sorted(str(value) for value in condition_metrics["condition"].dropna().unique().tolist())
    l3 = payloads.get("probe_trace_arrays_l3.npz", {})
    vectors = payloads.get("readout_trajectory_vectors.npz", {})
    for pair_id in pair_ids:
        for condition in conditions:
            trace_key = f"pair_{int(pair_id)}_{condition}_l3_trace"
            vector_key = f"pair_{int(pair_id)}_{condition}_grouped_voltage"
            if trace_key not in l3:
                raise RuntimeError(f"Fig.4 rollout L3 trace missing key: {trace_key}")
            if vector_key not in vectors:
                raise RuntimeError(f"Fig.4 rollout vector missing key: {vector_key}")
    capture_arrays = payloads.get("l3_replay_capture_arrays.npz", {})
    if capture_manifest is None or capture_manifest.empty:
        raise RuntimeError("Fig.4 rollout L3 replay capture manifest is missing or empty.")
    _validate_l3_replay_capture_payload(condition_metrics, capture_manifest, capture_arrays)


def _validate_l3_replay_capture_payload(
    condition_metrics: pd.DataFrame,
    capture_manifest: pd.DataFrame,
    capture_arrays: Mapping[str, np.ndarray],
) -> None:
    required_columns = ("network_seed", "pair_id", "condition", "field", "storage_file", "storage_key", "shape", "dtype")
    _require_columns(capture_manifest, required_columns, Path("l3_replay_capture_manifest.csv"))
    if set(str(value) for value in capture_manifest["storage_file"].dropna().unique().tolist()) != {"l3_replay_capture_arrays.npz"}:
        raise RuntimeError("Fig.4 L3 replay capture manifest must reference only l3_replay_capture_arrays.npz.")
    pair_ids = sorted(int(value) for value in condition_metrics["pair_id"].dropna().unique().tolist())
    required_conditions = ("full_dynamic", "full_static")
    fields = (
        "probe_s2p_trace",
        "grouped_voltage",
        "readout_snapshot",
        "prediction_probe",
        "first_fire_t_probe",
        "readout_step",
        "probe_onset_v_mem",
        "probe_onset_g_e",
        "probe_onset_res",
        "probe_onset_inh_trace",
        "probe_onset_u_pre",
        "probe_onset_x_pre",
        "probe_onset_input_trace",
        "probe_onset_eligibility_trace",
        "probe_onset_firing_times",
        "probe_onset_input_shape",
        "probe_onset_output_shape",
    )
    if not capture_arrays:
        raise RuntimeError("Fig.4 rollout L3 replay capture arrays are missing.")
    manifest_keys = set(str(value) for value in capture_manifest["storage_key"].tolist())
    array_keys = set(str(key) for key in capture_arrays)
    if manifest_keys != array_keys:
        raise RuntimeError(
            f"Fig.4 L3 replay capture key mismatch: manifest={sorted(manifest_keys)[:10]}, arrays={sorted(array_keys)[:10]}"
        )
    for row in capture_manifest.to_dict("records"):
        key = str(row["storage_key"])
        arr = np.asarray(capture_arrays[key])
        if _shape_text(arr.shape) != str(row["shape"]):
            raise RuntimeError(f"Fig.4 L3 replay capture shape mismatch for {key}: expected={row['shape']}, found={_shape_text(arr.shape)}")
        if str(arr.dtype) != str(row["dtype"]):
            raise RuntimeError(f"Fig.4 L3 replay capture dtype mismatch for {key}: expected={row['dtype']}, found={arr.dtype}")
    for pair_id in pair_ids:
        for condition in required_conditions:
            for field in fields:
                rows = capture_manifest[
                    capture_manifest["pair_id"].astype(int).eq(int(pair_id))
                    & capture_manifest["condition"].astype(str).eq(str(condition))
                    & capture_manifest["field"].astype(str).eq(str(field))
                ]
                if len(rows) != 1:
                    raise RuntimeError(
                        f"Fig.4 L3 replay capture expected one row for pair={pair_id}, condition={condition}, field={field}; found {len(rows)}"
                    )


def _require_pair_membership(df: pd.DataFrame, pair_trials: pd.DataFrame, *, task_id: str) -> None:
    if "pair_id" not in df.columns:
        raise RuntimeError(f"Fig.4 {task_id} artifact table has no pair_id column.")
    expected = {int(value) for value in pair_trials["pair_id"].tolist()}
    found = {int(value) for value in df["pair_id"].dropna().unique().tolist()}
    if not found.issubset(expected):
        raise RuntimeError(f"Fig.4 {task_id} artifact contains unknown pair IDs: {sorted(found.difference(expected))}")


def _artifact_digest(table_digest_value: str, array_manifest: pd.DataFrame) -> str:
    array_part = table_digest({"array_manifest": array_manifest.loc[:, list(ARRAY_MANIFEST_COLUMNS)].copy()})
    return table_digest({"digest": pd.DataFrame([{"tables": table_digest_value, "arrays": array_part}])})


def _validate_manifest_hashes(task_dir: Path, manifest: pd.DataFrame, *, path_column: str) -> None:
    for rel_path, part in manifest.groupby(path_column, sort=False):
        path = Path(task_dir) / str(rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Fig.4 artifact file is missing: {path}")
        found = sha256_file(path)
        expected = {str(value) for value in part["sha256"].astype(str).tolist()}
        if expected != {found}:
            raise RuntimeError(f"Fig.4 artifact hash mismatch for {path}: expected={sorted(expected)}, found={found}")


def _table_manifest_record(manifest: pd.DataFrame, name: str, filename: str, manifest_path: Path) -> dict[str, Any]:
    rows = manifest[
        manifest["name"].astype(str).eq(str(name))
        & manifest["filename"].astype(str).eq(str(filename))
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Fig.4 table manifest expected one row for {name}/{filename}, found {len(rows)}: {manifest_path}")
    return rows.iloc[0].to_dict()


def _parse_manifest_columns(value: Any, manifest_path: Path) -> list[str]:
    try:
        columns = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Fig.4 manifest has malformed columns JSON in {manifest_path}: {value!r}") from exc
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise ValueError(f"Fig.4 manifest columns must be a JSON string list in {manifest_path}: {value!r}")
    return columns


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")


def _mask_storage_key(pair_id: int, mask_name: str) -> str:
    return f"pair_{int(pair_id)}__{str(mask_name)}"


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(int(value)) for value in shape)


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
    "PairSamplingArtifact",
    "RolloutBankArtifact",
    "SimilarityEntryArtifact",
    "cache_key_matches",
    "copy_rollout_artifact_npz_to_raw",
    "default_artifact_root",
    "load_pair_sampling_artifact",
    "load_rollout_bank_artifact",
    "load_similarity_entry_artifact",
    "read_cache_key",
    "read_json",
    "reset_task_artifact_dir",
    "save_pair_sampling_artifact",
    "save_rollout_bank_artifact",
    "save_similarity_entry_artifact",
    "task_artifact_dir",
    "write_cache_key",
    "write_json",
]
