from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig5.cache_keys import (
    cache_key_digest,
    sha256_file,
    table_digest,
    trials_hash,
)
from src.experiments.paper_figures.fig5.schemas import (
    ARRAY_MANIFEST_COLUMNS,
    SUPPORT_BANK_FILES,
    TABLE_MANIFEST_COLUMNS,
    TRIAL_SAMPLING_FILES,
)
from src.experiments.paper_figures.fig5.types import BranchTrace, LocalSupportCompetitionBank


CACHE_KEY_FILE = "cache_key.json"


@dataclass(frozen=True)
class TrialSamplingArtifact:
    root: Path
    trials: pd.DataFrame
    audit: pd.DataFrame
    table_manifest: pd.DataFrame
    array_manifest: pd.DataFrame
    digest: str


@dataclass(frozen=True)
class SupportBankArtifact:
    root: Path
    bank: LocalSupportCompetitionBank
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
            f"Fig.5 {task_id} artifact cache key mismatch: expected {expected_digest}, found {found_digest}. "
            "Rebuild the producer task before using --reuse-artifacts require."
        )


def save_trial_sampling_artifact(
    task_dir: Path,
    *,
    trials: pd.DataFrame,
    audit: pd.DataFrame,
    raw_dir: Path,
    cache_key: Mapping[str, Any],
) -> TrialSamplingArtifact:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    tables = _save_table_bundle(
        task_dir,
        tables={"trials": trials.copy(), "trial_condition_audit": audit.copy()},
        files=TRIAL_SAMPLING_FILES,
    )
    array_manifest = _copy_npz_artifacts(task_dir, raw_dir, ("trial_masks.npz",))
    digest = _artifact_digest(tables.digest, array_manifest)
    write_json({"artifact_digest": digest}, task_dir / "artifact_digest.json")
    write_cache_key(task_dir, cache_key)
    return TrialSamplingArtifact(task_dir, tables.tables["trials"], tables.tables["trial_condition_audit"], tables.manifest, array_manifest, digest)


def load_trial_sampling_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
) -> TrialSamplingArtifact:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id="trial_sampling")
    tables = _load_table_bundle(task_dir, files=TRIAL_SAMPLING_FILES, task_id="trial_sampling")
    arrays = _read_array_manifest(task_dir)
    _validate_manifest_hashes(task_dir, arrays, path_column="storage_file")
    digest = _artifact_digest(tables.digest, arrays)
    recorded = read_json(task_dir / "artifact_digest.json").get("artifact_digest")
    if str(recorded) != digest:
        raise RuntimeError(f"Fig.5 trial_sampling artifact digest mismatch: expected {recorded}, found {digest}")
    return TrialSamplingArtifact(task_dir, tables.tables["trials"], tables.tables["trial_condition_audit"], tables.manifest, arrays, digest)


def save_support_bank_artifact(
    task_dir: Path,
    bank: LocalSupportCompetitionBank,
    *,
    raw_dir: Path,
    cache_key: Mapping[str, Any],
) -> SupportBankArtifact:
    task_dir = Path(task_dir)
    reset_task_artifact_dir(task_dir)
    table_bundle = _save_table_bundle(
        task_dir,
        tables={
            "unit_groups": bank.unit_groups.copy(),
            "perturbation_sets": bank.perturbation_sets.copy(),
            "perturbation_ux_audit": bank.perturbation_ux_audit.copy(),
            "l1_stsp_perturbation_audit": bank.l1_stsp_perturbation_audit.copy(),
            "rollout_manifest": _read_csv_or_empty(Path(raw_dir) / "rollout_manifest.csv"),
            "trace_manifest": _read_csv_or_empty(Path(raw_dir) / "layer1_probe_trace_manifest.csv"),
        },
        files=SUPPORT_BANK_FILES,
    )
    payloads = {
        "support_maps.npz": _support_map_payload(bank.support_maps),
        "branch_traces.npz": _branch_trace_payload(bank.branch_traces),
    }
    array_manifest = _save_npz_files(task_dir, payloads)
    _validate_support_bank_payload(bank.trials, payloads)
    digest = _artifact_digest(table_bundle.digest, array_manifest)
    write_json({"artifact_digest": digest}, task_dir / "artifact_digest.json")
    write_cache_key(task_dir, cache_key)
    artifact_bank = LocalSupportCompetitionBank(
        trials=bank.trials.reset_index(drop=True).copy(),
        support_maps={int(k): np.asarray(v) for k, v in bank.support_maps.items()},
        branch_traces=_branch_traces_from_payload(payloads["branch_traces.npz"]),
        boundary_states={},
        unit_groups=table_bundle.tables["unit_groups"],
        perturbation_sets=table_bundle.tables["perturbation_sets"],
        perturbation_ux_audit=table_bundle.tables["perturbation_ux_audit"],
        l1_stsp_perturbation_audit=table_bundle.tables["l1_stsp_perturbation_audit"],
    )
    return SupportBankArtifact(task_dir, artifact_bank, table_bundle.manifest, array_manifest, digest)


def load_support_bank_artifact(
    task_dir: Path,
    *,
    expected_key: Mapping[str, Any] | None = None,
    trials: pd.DataFrame,
) -> SupportBankArtifact:
    task_dir = Path(task_dir)
    if expected_key is not None:
        require_cache_key_match(task_dir, expected_key, task_id="preprobe_support_bank")
    table_bundle = _load_table_bundle(task_dir, files=SUPPORT_BANK_FILES, task_id="preprobe_support_bank")
    arrays = _load_npz_files(task_dir, task_id="preprobe_support_bank", required_files=("support_maps.npz", "branch_traces.npz"))
    _validate_support_bank_payload(trials, arrays)
    digest = _artifact_digest(table_bundle.digest, _read_array_manifest(task_dir))
    recorded = read_json(task_dir / "artifact_digest.json").get("artifact_digest")
    if str(recorded) != digest:
        raise RuntimeError(f"Fig.5 support bank artifact digest mismatch: expected {recorded}, found {digest}")
    bank = LocalSupportCompetitionBank(
        trials=trials.reset_index(drop=True).copy(),
        support_maps=_support_maps_from_payload(arrays["support_maps.npz"]),
        branch_traces=_branch_traces_from_payload(arrays["branch_traces.npz"]),
        boundary_states={},
        unit_groups=table_bundle.tables["unit_groups"],
        perturbation_sets=table_bundle.tables["perturbation_sets"],
        perturbation_ux_audit=table_bundle.tables["perturbation_ux_audit"],
        l1_stsp_perturbation_audit=table_bundle.tables["l1_stsp_perturbation_audit"],
    )
    return SupportBankArtifact(task_dir, bank, table_bundle.manifest, _read_array_manifest(task_dir), digest)


def copy_trial_npz_to_raw(task_dir: Path, raw_dir: Path) -> None:
    _copy_files(task_dir, raw_dir, ("trial_masks.npz",))


def copy_support_bank_tables_to_bundle(task_dir: Path, raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rollout = pd.read_csv(Path(task_dir) / "rollout_manifest.csv")
    trace = pd.read_csv(Path(task_dir) / "layer1_probe_trace_manifest.csv")
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rollout.to_csv(raw_dir / "rollout_manifest.csv", index=False, encoding="utf-8")
    trace.to_csv(raw_dir / "layer1_probe_trace_manifest.csv", index=False, encoding="utf-8")
    return rollout, trace


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
        raise FileNotFoundError(f"Fig.5 {task_id} manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, TABLE_MANIFEST_COLUMNS, manifest_path)
    found_names = set(str(value) for value in manifest["name"].tolist())
    expected_names = set(files)
    if found_names != expected_names:
        raise RuntimeError(f"Fig.5 {task_id} manifest names mismatch: expected={sorted(expected_names)}, found={sorted(found_names)}")
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in files.items():
        rec = _table_manifest_record(manifest, name, filename, manifest_path)
        path = task_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Fig.5 {task_id} artifact file is missing: {path}")
        found_sha = sha256_file(path)
        expected_sha = str(rec["sha256"])
        if found_sha != expected_sha:
            raise RuntimeError(f"Fig.5 {task_id} file hash mismatch for {filename}: expected {expected_sha}, found {found_sha}")
        columns = _parse_manifest_columns(rec["columns"], manifest_path)
        df = pd.read_csv(path) if columns else pd.DataFrame()
        if len(df) != int(rec["rows"]):
            raise RuntimeError(f"Fig.5 {task_id} row count mismatch for {filename}: expected {rec['rows']}, found {len(df)}")
        if list(df.columns) != columns:
            raise RuntimeError(f"Fig.5 {task_id} column mismatch for {filename}: expected {columns}, found {list(df.columns)}")
        tables[name] = df
    digest = table_digest(tables)
    manifest_digests = {str(value) for value in manifest["table_digest"].dropna().astype(str).tolist()}
    if manifest_digests != {digest}:
        raise RuntimeError(f"Fig.5 {task_id} table digest mismatch: manifest={sorted(manifest_digests)}, computed={digest}")
    return TableBundle(task_dir, tables, manifest, digest)


def _copy_npz_artifacts(task_dir: Path, raw_dir: Path, filenames: tuple[str, ...]) -> pd.DataFrame:
    files: dict[str, dict[str, np.ndarray]] = {}
    for filename in filenames:
        path = Path(raw_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"Fig.5 artifact source NPZ is missing: {path}")
        with np.load(path, allow_pickle=False) as payload:
            files[filename] = {str(key): np.array(payload[key]) for key in payload.files}
    return _save_npz_files(task_dir, files)


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
        raise RuntimeError(f"Fig.5 {task_id} array manifest missing required files: {missing}")
    out: dict[str, dict[str, np.ndarray]] = {}
    for filename in sorted(found_files):
        path = Path(task_dir) / filename
        part = manifest[manifest["storage_file"].astype(str).eq(filename)]
        with np.load(path, allow_pickle=False) as payload:
            expected_keys = set(str(value) for value in part["storage_key"].tolist())
            if set(payload.files) != expected_keys:
                raise RuntimeError(f"Fig.5 {task_id} NPZ key mismatch for {filename}: expected={sorted(expected_keys)}, found={sorted(payload.files)}")
            out[filename] = {}
            for row in part.to_dict("records"):
                key = str(row["storage_key"])
                arr = np.array(payload[key])
                if _shape_text(arr.shape) != str(row["shape"]):
                    raise RuntimeError(f"Fig.5 {task_id} shape mismatch for {filename}/{key}: expected={row['shape']}, found={_shape_text(arr.shape)}")
                out[filename][key] = arr
    return out


def _read_array_manifest(task_dir: Path) -> pd.DataFrame:
    manifest_path = Path(task_dir) / "array_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fig.5 array manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    _require_columns(manifest, ARRAY_MANIFEST_COLUMNS, manifest_path)
    return manifest


def _support_map_payload(support_maps: Mapping[int, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"trial_{int(trial_id)}": np.asarray(value, dtype=np.float32) for trial_id, value in sorted(support_maps.items())}


def _support_maps_from_payload(payload: Mapping[str, np.ndarray]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for key, arr in payload.items():
        if not str(key).startswith("trial_"):
            continue
        out[int(str(key).split("_", 1)[1])] = np.asarray(arr, dtype=np.float32)
    return out


def _branch_trace_payload(branch_traces: Mapping[int, Mapping[str, BranchTrace]]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for trial_id, traces in sorted(branch_traces.items()):
        for condition, trace in sorted(traces.items()):
            prefix = f"trial_{int(trial_id)}__{condition}"
            payload[f"{prefix}__spikes"] = np.asarray(trace.spikes)
            payload[f"{prefix}__v_effective"] = np.asarray(trace.v_effective)
            payload[f"{prefix}__inhibition"] = np.asarray(trace.inhibition)
            payload[f"{prefix}__layer3_spikes"] = np.asarray(trace.layer3_spikes)
            payload[f"{prefix}__prediction"] = np.asarray([int(trace.prediction)], dtype=np.int64)
            payload[f"{prefix}__first_fire_time"] = np.asarray([int(trace.first_fire_time)], dtype=np.int64)
    return payload


def _branch_traces_from_payload(payload: Mapping[str, np.ndarray]) -> dict[int, dict[str, BranchTrace]]:
    grouped: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for key, arr in payload.items():
        parts = str(key).split("__")
        if len(parts) != 3 or not parts[0].startswith("trial_"):
            continue
        trial_id = int(parts[0].split("_", 1)[1])
        condition = parts[1]
        field = parts[2]
        grouped.setdefault((trial_id, condition), {})[field] = np.asarray(arr)
    out: dict[int, dict[str, BranchTrace]] = {}
    for (trial_id, condition), fields in grouped.items():
        required = {"spikes", "v_effective", "inhibition", "layer3_spikes", "prediction", "first_fire_time"}
        missing = sorted(required.difference(fields))
        if missing:
            raise RuntimeError(f"Fig.5 branch trace artifact missing fields for trial={trial_id} condition={condition}: {missing}")
        out.setdefault(trial_id, {})[condition] = BranchTrace(
            spikes=np.asarray(fields["spikes"]),
            v_effective=np.asarray(fields["v_effective"]),
            inhibition=np.asarray(fields["inhibition"]),
            layer3_spikes=np.asarray(fields["layer3_spikes"]),
            prediction=int(np.asarray(fields["prediction"]).reshape(-1)[0]),
            first_fire_time=int(np.asarray(fields["first_fire_time"]).reshape(-1)[0]),
        )
    return out


def _validate_support_bank_payload(trials: pd.DataFrame, payloads: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    trial_ids = {int(value) for value in trials["trial_id"].tolist()}
    support_ids = set(_support_maps_from_payload(payloads["support_maps.npz"]))
    if support_ids != trial_ids:
        raise RuntimeError(f"Fig.5 support map trial mismatch: expected={sorted(trial_ids)}, found={sorted(support_ids)}")
    trace_ids = set(_branch_traces_from_payload(payloads["branch_traces.npz"]))
    if trace_ids != trial_ids:
        raise RuntimeError(f"Fig.5 branch trace trial mismatch: expected={sorted(trial_ids)}, found={sorted(trace_ids)}")


def _artifact_digest(table_digest_value: str, array_manifest: pd.DataFrame) -> str:
    array_part = table_digest({"array_manifest": array_manifest.loc[:, list(ARRAY_MANIFEST_COLUMNS)].copy()})
    return table_digest({"digest": pd.DataFrame([{"tables": table_digest_value, "arrays": array_part}])})


def _validate_manifest_hashes(task_dir: Path, manifest: pd.DataFrame, *, path_column: str) -> None:
    for rel_path, part in manifest.groupby(path_column, sort=False):
        path = Path(task_dir) / str(rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Fig.5 artifact file is missing: {path}")
        found = sha256_file(path)
        expected = {str(value) for value in part["sha256"].astype(str).tolist()}
        if expected != {found}:
            raise RuntimeError(f"Fig.5 artifact hash mismatch for {path}: expected={sorted(expected)}, found={found}")


def _copy_files(src_dir: Path, dst_dir: Path, filenames: tuple[str, ...]) -> None:
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        src = Path(src_dir) / filename
        if src.exists():
            shutil.copy2(src, dst_dir / filename)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _table_manifest_record(manifest: pd.DataFrame, name: str, filename: str, manifest_path: Path) -> dict[str, Any]:
    rows = manifest[
        manifest["name"].astype(str).eq(str(name))
        & manifest["filename"].astype(str).eq(str(filename))
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Fig.5 table manifest expected one row for {name}/{filename}, found {len(rows)}: {manifest_path}")
    return rows.iloc[0].to_dict()


def _parse_manifest_columns(value: Any, manifest_path: Path) -> list[str]:
    try:
        columns = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Fig.5 manifest has malformed columns JSON in {manifest_path}: {value!r}") from exc
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise ValueError(f"Fig.5 manifest columns must be a JSON string list in {manifest_path}: {value!r}")
    return columns


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")


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
    "SupportBankArtifact",
    "TrialSamplingArtifact",
    "cache_key_matches",
    "copy_support_bank_tables_to_bundle",
    "copy_trial_npz_to_raw",
    "default_artifact_root",
    "load_support_bank_artifact",
    "load_trial_sampling_artifact",
    "read_cache_key",
    "save_support_bank_artifact",
    "save_trial_sampling_artifact",
    "task_artifact_dir",
    "trials_hash",
    "write_cache_key",
    "write_json",
]
