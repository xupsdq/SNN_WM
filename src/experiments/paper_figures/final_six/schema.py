from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


EXPECTED_SEEDS = tuple(range(1000, 1020))
PLOT_BASE_COLUMNS = (
    "figure_id",
    "panel_id",
    "network_seed",
    "record_type",
    "endpoint",
    "condition",
    "value",
    "unit",
)
STATISTICS_COLUMNS = (
    "figure_id",
    "panel_id",
    "endpoint",
    "contrast",
    "group",
    "n_networks",
    "estimate",
    "mean",
    "sd",
    "sem",
    "ci95_low",
    "ci95_high",
    "median",
    "q1",
    "q3",
    "min",
    "max",
    "null_value",
    "test_name",
    "statistic",
    "df",
    "p_value",
    "p_adjust_method",
    "p_adjusted",
    "alternative",
    "unit",
    "statistics_status",
)
SOURCE_MANIFEST_COLUMNS = (
    "figure_id",
    "panel_id",
    "source_path",
    "source_sha256",
    "source_bytes",
    "source_level",
    "producer_task",
    "filters",
    "held_fixed",
    "input_rows",
    "output_rows",
    "aggregation_path",
    "independent_unit",
    "included_seeds",
    "excluded_rows",
    "exclusion_reason",
    "output_csv",
    "builder_module",
    "builder_version",
)


@dataclass(frozen=True)
class SourceDescriptor:
    key: str
    pattern: str
    source_level: str
    producer_task: str
    filters: str
    held_fixed: str
    aggregation_path: str
    independent_unit: str = "network_seed"
    seeded: bool = True
    required_columns: Sequence[str] = ()


@dataclass
class LoadedSource:
    descriptor: SourceDescriptor
    frame: pd.DataFrame
    records: list[dict[str, Any]]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _path_seed(path: Path) -> Optional[int]:
    match = re.search(r"seed_(\d+)", path.as_posix())
    return int(match.group(1)) if match else None


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _source_record(
    *,
    figure_id: str,
    panel_id: str,
    repo_root: Path,
    descriptor: SourceDescriptor,
    path: Path,
    input_rows: int,
    source_sha256: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "source_path": _relative(path, repo_root),
        "source_sha256": source_sha256 or sha256_file(path),
        "source_bytes": int(path.stat().st_size),
        "source_level": descriptor.source_level,
        "producer_task": descriptor.producer_task,
        "filters": descriptor.filters,
        "held_fixed": descriptor.held_fixed,
        "input_rows": int(input_rows),
        "output_rows": pd.NA,
        "aggregation_path": descriptor.aggregation_path,
        "independent_unit": descriptor.independent_unit,
        "included_seeds": "1000-1019",
        "excluded_rows": pd.NA,
        "exclusion_reason": "",
        "output_csv": "",
        "builder_module": "",
        "builder_version": "",
    }


def record_file_source(
    *,
    repo_root: Path,
    figure_id: str,
    panel_id: str,
    descriptor: SourceDescriptor,
    path: Path,
    input_rows: int = 1,
) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{descriptor.key}: required parent is missing: {path}")
    return _source_record(
        figure_id=figure_id,
        panel_id=panel_id,
        repo_root=repo_root,
        descriptor=descriptor,
        path=path,
        input_rows=input_rows,
    )


def resolve_source_paths(
    repo_root: Path,
    descriptor: SourceDescriptor,
) -> list[Path]:
    paths = sorted(repo_root.glob(descriptor.pattern))
    if descriptor.seeded:
        by_seed: dict[int, list[Path]] = {}
        for path in paths:
            seed = _path_seed(path)
            if seed is None:
                raise ValueError(f"{descriptor.key}: seeded source has no seed directory: {path}")
            by_seed.setdefault(seed, []).append(path)
        missing = sorted(set(EXPECTED_SEEDS) - set(by_seed))
        extra = sorted(set(by_seed) - set(EXPECTED_SEEDS))
        duplicates = {seed: values for seed, values in by_seed.items() if len(values) != 1}
        if missing or extra or duplicates:
            raise RuntimeError(
                f"{descriptor.key}: require-mode cohort failure; "
                f"missing={missing}, extra={extra}, duplicates={duplicates}"
            )
        return [by_seed[seed][0] for seed in EXPECTED_SEEDS]
    if len(paths) != 1:
        raise RuntimeError(
            f"{descriptor.key}: require-mode expected exactly one parent, found {len(paths)} "
            f"for {descriptor.pattern}"
        )
    return paths


def load_source(
    *,
    repo_root: Path,
    figure_id: str,
    panel_id: str,
    descriptor: SourceDescriptor,
    usecols: Optional[Sequence[str]] = None,
) -> LoadedSource:
    paths = resolve_source_paths(repo_root, descriptor)
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False, usecols=usecols)
        required = set(descriptor.required_columns)
        missing_columns = sorted(required - set(frame.columns))
        if missing_columns:
            raise ValueError(f"{descriptor.key}: missing columns {missing_columns} in {path}")
        path_seed = _path_seed(path)
        if descriptor.seeded and path_seed is not None:
            if "network_seed" not in frame.columns:
                frame.insert(0, "network_seed", path_seed)
            observed = sorted(
                int(value)
                for value in pd.to_numeric(frame["network_seed"], errors="coerce")
                .dropna()
                .unique()
            )
            if observed != [path_seed]:
                raise ValueError(
                    f"{descriptor.key}: network_seed/path mismatch in {path}; observed={observed}"
                )
        frames.append(frame)
        records.append(
            _source_record(
                figure_id=figure_id,
                panel_id=panel_id,
                repo_root=repo_root,
                descriptor=descriptor,
                path=path,
                input_rows=len(frame),
            )
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return LoadedSource(descriptor=descriptor, frame=combined, records=records)


def inspect_source_without_loading(
    *,
    repo_root: Path,
    figure_id: str,
    panel_id: str,
    descriptor: SourceDescriptor,
) -> LoadedSource:
    paths = resolve_source_paths(repo_root, descriptor)
    records: list[dict[str, Any]] = []
    headers: list[pd.DataFrame] = []
    for path in paths:
        header = pd.read_csv(path, nrows=0)
        missing_columns = sorted(set(descriptor.required_columns) - set(header.columns))
        if missing_columns:
            raise ValueError(f"{descriptor.key}: missing columns {missing_columns} in {path}")
        row_count = _count_csv_rows(path)
        headers.append(header)
        records.append(
            _source_record(
                figure_id=figure_id,
                panel_id=panel_id,
                repo_root=repo_root,
                descriptor=descriptor,
                path=path,
                input_rows=row_count,
            )
        )
    columns = list(headers[0].columns) if headers else []
    return LoadedSource(
        descriptor=descriptor,
        frame=pd.DataFrame(columns=columns),
        records=records,
    )


def _count_csv_rows(path: Path, chunk_size: int = 1024 * 1024) -> int:
    lines = 0
    last_byte = b""
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            lines += block.count(b"\n")
            last_byte = block[-1:]
    if path.stat().st_size and last_byte != b"\n":
        lines += 1
    return max(0, lines - 1)


def finalize_source_records(
    records: Iterable[dict[str, Any]],
    *,
    output_rows: int,
    excluded_rows: int,
    exclusion_reason: str,
    output_csv: str,
    builder_module: str,
    builder_version: str,
) -> pd.DataFrame:
    finalized: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        record.update(
            {
                "output_rows": int(output_rows),
                "excluded_rows": int(excluded_rows),
                "exclusion_reason": exclusion_reason,
                "output_csv": output_csv,
                "builder_module": builder_module,
                "builder_version": builder_version,
            }
        )
        finalized.append(record)
    return pd.DataFrame(finalized, columns=SOURCE_MANIFEST_COLUMNS)


def make_plot_data(
    frame: pd.DataFrame,
    *,
    figure_id: str,
    panel_id: str,
    record_type: str,
    endpoint: str,
    condition: str,
    value: str,
    unit: str,
    dimensions: Sequence[str] = (),
) -> pd.DataFrame:
    columns = ["network_seed", *dimensions]
    missing = [column for column in [*columns, value] if column not in frame.columns]
    if missing:
        raise ValueError(f"{figure_id}{panel_id}: plot-data source missing {missing}")
    output = frame.loc[:, columns].copy()
    output.insert(0, "figure_id", figure_id)
    output.insert(1, "panel_id", panel_id)
    output.insert(3, "record_type", record_type)
    output.insert(4, "endpoint", endpoint)
    output.insert(5, "condition", condition)
    output.insert(6, "value", pd.to_numeric(frame[value], errors="coerce"))
    output.insert(7, "unit", unit)
    return output


def validate_plot_data(
    frame: pd.DataFrame,
    *,
    figure_id: str,
    panel_id: str,
    unique_key: Sequence[str],
) -> dict[str, Any]:
    missing = sorted(set(PLOT_BASE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{figure_id}{panel_id}: plot-data missing base columns {missing}")
    if frame.empty:
        raise ValueError(f"{figure_id}{panel_id}: plot-data is empty")
    seeds = sorted(
        int(value)
        for value in pd.to_numeric(frame["network_seed"], errors="coerce").dropna().unique()
    )
    expected = list(EXPECTED_SEEDS)
    if seeds != expected:
        raise ValueError(f"{figure_id}{panel_id}: expected seeds {expected}, observed {seeds}")
    duplicated = int(frame.duplicated(list(unique_key), keep=False).sum())
    if duplicated:
        raise ValueError(
            f"{figure_id}{panel_id}: duplicate plot-data key rows={duplicated}, key={list(unique_key)}"
        )
    numeric = pd.to_numeric(frame["value"], errors="coerce")
    nonnumeric = int(frame["value"].notna().sum() - numeric.notna().sum())
    if nonnumeric:
        raise ValueError(f"{figure_id}{panel_id}: value contains {nonnumeric} nonnumeric entries")
    return {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "expected_seeds": "1000-1019",
        "observed_seeds": ",".join(str(seed) for seed in seeds),
        "missing_seeds": "",
        "extra_seeds": "",
        "duplicate_key_count": 0,
        "row_count": int(len(frame)),
        "status": "pass",
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {
            "n_networks": 0,
            "estimate": math.nan,
            "mean": math.nan,
            "sd": math.nan,
            "sem": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "median": math.nan,
            "q1": math.nan,
            "q3": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else math.nan
    sem = sd / math.sqrt(n) if n > 1 else math.nan
    half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else math.nan
    return {
        "n_networks": n,
        "estimate": mean,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_low": mean - half if n > 1 else math.nan,
        "ci95_high": mean + half if n > 1 else math.nan,
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_statistics(
    values: pd.DataFrame,
    *,
    figure_id: str,
    panel_id: str,
    dimension_columns: Sequence[str] = (),
) -> pd.DataFrame:
    required = {
        "network_seed",
        "endpoint",
        "contrast",
        "group",
        "value",
        "null_value",
        "unit",
        "statistics_status",
    }
    missing = sorted(required - set(values.columns))
    if missing:
        raise ValueError(f"{figure_id}{panel_id}: statistic values missing {missing}")
    work = values.copy()
    for column, default in (
        ("test_name", "one_sample_t"),
        ("alternative", "two-sided"),
        ("p_adjust_family", ""),
    ):
        if column not in work:
            work[column] = default
    group_columns = [
        *dimension_columns,
        "endpoint",
        "contrast",
        "group",
        "null_value",
        "test_name",
        "alternative",
        "unit",
        "statistics_status",
        "p_adjust_family",
    ]
    records: list[dict[str, Any]] = []
    for keys, part in work.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metadata = dict(zip(group_columns, keys))
        by_seed = (
            part.assign(value=pd.to_numeric(part["value"], errors="coerce"))
            .groupby("network_seed", as_index=False)["value"]
            .mean()
        )
        observed_seeds = sorted(
            int(value)
            for value in pd.to_numeric(by_seed["network_seed"], errors="coerce")
            .dropna()
            .unique()
        )
        if observed_seeds != list(EXPECTED_SEEDS):
            raise ValueError(
                f"{figure_id}{panel_id}: statistics group {metadata.get('group')} "
                f"has seed set {observed_seeds}"
            )
        numeric = pd.to_numeric(by_seed["value"], errors="coerce").to_numpy(dtype=float)
        summary = _summary(numeric)
        status = str(metadata["statistics_status"])
        null_value = pd.to_numeric(pd.Series([metadata["null_value"]]), errors="coerce").iloc[0]
        statistic = math.nan
        p_value = math.nan
        df = math.nan
        test_name = ""
        alternative = ""
        if status in {"predeclared_recomputed", "supplied"} and np.isfinite(null_value):
            finite = numeric[np.isfinite(numeric)]
            if len(finite) != len(EXPECTED_SEEDS):
                raise ValueError(
                    f"{figure_id}{panel_id}: inferential group {metadata.get('group')} "
                    "does not contain 20 finite network values"
                )
            result = stats.ttest_1samp(finite, popmean=float(null_value), alternative="two-sided")
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            df = float(len(finite) - 1)
            test_name = "one_sample_t"
            alternative = "two-sided"
        record = {
            "figure_id": figure_id,
            "panel_id": panel_id,
            **{column: metadata[column] for column in dimension_columns},
            "endpoint": metadata["endpoint"],
            "contrast": metadata["contrast"],
            "group": metadata["group"],
            **summary,
            "null_value": float(null_value) if np.isfinite(null_value) else math.nan,
            "test_name": test_name,
            "statistic": statistic,
            "df": df,
            "p_value": p_value,
            "p_adjust_method": "",
            "p_adjusted": math.nan,
            "alternative": alternative,
            "unit": metadata["unit"],
            "statistics_status": status,
            "_p_adjust_family": str(metadata["p_adjust_family"]),
        }
        records.append(record)
    output = pd.DataFrame(records)
    _apply_p_adjustment(output)
    output = output.drop(columns=["_p_adjust_family"])
    ordered = [*STATISTICS_COLUMNS, *[c for c in dimension_columns if c not in STATISTICS_COLUMNS]]
    return output.loc[:, ordered]


def _apply_p_adjustment(frame: pd.DataFrame) -> None:
    for family, indices in frame.groupby("_p_adjust_family", dropna=False).groups.items():
        index = list(indices)
        finite_index = [i for i in index if np.isfinite(float(frame.at[i, "p_value"]))]
        if not finite_index:
            continue
        if str(family):
            adjusted = _benjamini_hochberg(
                np.asarray([float(frame.at[i, "p_value"]) for i in finite_index], dtype=float)
            )
            for row_index, value in zip(finite_index, adjusted):
                frame.at[row_index, "p_adjust_method"] = "BH"
                frame.at[row_index, "p_adjusted"] = float(value)
        else:
            for row_index in finite_index:
                frame.at[row_index, "p_adjust_method"] = "none"
                frame.at[row_index, "p_adjusted"] = float(frame.at[row_index, "p_value"])


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def statistics_values_from_plot(
    plot_data: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    status: str = "descriptive_only",
    null_values: Optional[Mapping[str, float]] = None,
    contrast: str = "",
    p_adjust_family: str = "",
) -> pd.DataFrame:
    null_values = dict(null_values or {})
    columns = ["network_seed", "endpoint", "value", "unit", *group_columns]
    values = plot_data.loc[:, columns].copy()
    values["contrast"] = contrast
    values["group"] = values.apply(
        lambda row: "|".join(str(row[column]) for column in group_columns),
        axis=1,
    )
    values["null_value"] = values["endpoint"].map(null_values).astype(float)
    values["statistics_status"] = status
    values["p_adjust_family"] = p_adjust_family
    return values


def schematic_statistics(figure_id: str, panel_id: str) -> pd.DataFrame:
    row = {column: pd.NA for column in STATISTICS_COLUMNS}
    row.update(
        {
            "figure_id": figure_id,
            "panel_id": panel_id,
            "endpoint": "schematic",
            "contrast": "",
            "group": "schematic",
            "unit": "not_applicable",
            "statistics_status": "not_applicable",
            "panel_type": "schematic",
        }
    )
    return pd.DataFrame([row])


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        frame.to_csv(temp_path, index=False, encoding="utf-8", lineterminator="\n")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
