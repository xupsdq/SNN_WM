from __future__ import annotations

import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.catalog import ExperimentSpec
from src.experiments.common.results import prepare_result_layout, save_summary_json
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info


BuildSeedCommand = Callable[[int, Path, str], list[str]]
NormalizeBundle = Callable[[Path, ExperimentSpec], None]

ID_COLUMNS = {
    "index",
    "seed",
    "run_dir",
    "trial_id",
    "pair_id",
    "sample_id",
    "probe_id",
    "record_id",
    "trial_index",
    "record_index",
    "voltage_vector_index",
    "selection_seed",
    "repeat_seed",
    "rng_seed",
    "eval_seed",
}
METADATA_COLUMNS = {"run_dir", "command", "status", "returncode", "elapsed_seconds", "error", "resumed"}
PAIR_ID_OFFSET = 1_000_000


@dataclass(frozen=True)
class SeedRunResult:
    seed: int
    run_dir: Path
    status: str
    returncode: int | None
    elapsed_seconds: float
    command: str
    device: str
    resumed: bool = False
    error: str = ""


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def relativize_files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def run_info_status(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    return str(status) if status is not None else None


def is_successful_run_dir(run_dir: Path) -> bool:
    return (run_dir / "summary.json").is_file() and run_info_status(run_dir / "meta" / "run_info.json") == "success"


def resolve_seed_list(seed: int, seeds: Sequence[int] | None) -> tuple[int, ...]:
    if seeds is not None:
        out = tuple(int(item) for item in seeds)
        if not out:
            raise SystemExit("--seeds was provided but no seed values were listed.")
        return out
    return (int(seed),)


def resolve_devices(device: str, devices: Sequence[str] | None) -> tuple[str, ...]:
    out = tuple(str(item) for item in (devices or ()) if str(item).strip())
    return out if out else (str(device),)


def _device_for_index(devices: Sequence[str], index: int) -> str:
    return str(devices[int(index) % len(devices)])


def run_seed_once(
    *,
    spec: ExperimentSpec,
    seed: int,
    run_dir: Path,
    device: str,
    model_path: str,
    dataset_root: str,
    build_command: BuildSeedCommand,
    normalize_bundle: NormalizeBundle,
    resume: bool,
) -> SeedRunResult:
    command = build_command(int(seed), run_dir, str(device))
    command_text = subprocess.list2cmdline(command)
    if resume and is_successful_run_dir(run_dir):
        return SeedRunResult(
            seed=int(seed),
            run_dir=run_dir,
            status="success",
            returncode=0,
            elapsed_seconds=0.0,
            command=command_text,
            device=str(device),
            resumed=True,
        )

    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_info_payload = build_run_info(
        experiment_name=spec.experiment_id,
        output_dir=run_dir,
        entry_script=f"python -m {spec.legacy_module}",
        seed=int(seed),
        dataset=str(dataset_root),
        command=command_text,
        model_path=str(model_path),
    )
    write_run_info(run_dir / "meta", run_info_payload)
    status = "failed"
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    elapsed = time.perf_counter() - started
    log_path.write_text(
        "\n".join(
            [
                f"command={command_text}",
                f"returncode={completed.returncode}",
                "[stdout]",
                completed.stdout.rstrip(),
                "[stderr]",
                completed.stderr.rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    if completed.returncode == 0:
        try:
            normalize_bundle(run_dir, spec)
            status = "success"
        except Exception as exc:
            finalize_run_info(run_dir / "meta", run_info_payload, status="failed")
            return SeedRunResult(
                seed=int(seed),
                run_dir=run_dir,
                status="failed",
                returncode=completed.returncode,
                elapsed_seconds=elapsed,
                command=command_text,
                device=str(device),
                error=str(exc),
            )
    finalize_run_info(run_dir / "meta", run_info_payload, status=status)
    ok = completed.returncode == 0 and is_successful_run_dir(run_dir)
    error = "" if ok else f"returncode={completed.returncode}; see {log_path}"
    return SeedRunResult(
        seed=int(seed),
        run_dir=run_dir,
        status="success" if ok else "failed",
        returncode=completed.returncode,
        elapsed_seconds=elapsed,
        command=command_text,
        device=str(device),
        error=error,
    )


def seed_result_record(result: SeedRunResult, root: Path) -> dict[str, Any]:
    try:
        rel_run_dir = result.run_dir.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel_run_dir = str(result.run_dir)
    return {
        "seed": int(result.seed),
        "run_dir": rel_run_dir,
        "device": str(result.device),
        "status": result.status,
        "returncode": result.returncode,
        "elapsed_seconds": float(result.elapsed_seconds),
        "command": result.command,
        "resumed": bool(result.resumed),
        "error": result.error,
    }


def _csv_sources_for_run(run_dir: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for base in (run_dir / "data", run_dir / "metrics", run_dir):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.csv") if base.name in {"data", "metrics"} else base.glob("*.csv")):
            sources.setdefault(path.name, path)
    return sources


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_id_column(column: str) -> bool:
    lower = column.lower()
    return lower in ID_COLUMNS or lower.endswith("_id") or lower.endswith("_seed")


def _infer_group_columns(df: pd.DataFrame, spec: ExperimentSpec) -> list[str]:
    group_cols: list[str] = []
    for column in (spec.csv_group, spec.csv_x):
        if column and column in df.columns and column not in group_cols:
            group_cols.append(column)
    for column in df.columns:
        if column in group_cols or column in METADATA_COLUMNS or _is_id_column(column):
            continue
        if not _is_numeric(df[column]):
            group_cols.append(column)
    numeric_condition_names = {
        "delay_ms",
        "bin_index",
        "stage_k",
        "sequence_length",
        "seq_len",
        "boost_level",
        "input_peak_overlap_fraction",
        "repeat_count",
        "rank",
        "layer_index",
        "keep_prob",
    }
    for column in df.columns:
        lower = column.lower()
        if column in group_cols or column in METADATA_COLUMNS or _is_id_column(column):
            continue
        if lower in numeric_condition_names or lower.endswith("_bin") or lower.endswith("_level"):
            group_cols.append(column)
    return group_cols


def _metric_columns(df: pd.DataFrame, group_cols: Sequence[str]) -> list[str]:
    excluded = set(group_cols) | set(METADATA_COLUMNS)
    metrics: list[str] = []
    for column in df.columns:
        if column in excluded or _is_id_column(column):
            continue
        if _is_numeric(df[column]):
            metrics.append(column)
    return metrics


def _seed_level_frame(df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    cols = ["seed", *group_cols]
    if "seed" not in df.columns or not metric_cols:
        return pd.DataFrame(columns=cols)
    return df.groupby(cols, dropna=False, sort=True)[list(metric_cols)].mean(numeric_only=True).reset_index()


def _t_ci(values: np.ndarray) -> tuple[float | None, float | None]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return None, None
    mean = float(vals.mean())
    if vals.size <= 1:
        return mean, mean
    sem = float(vals.std(ddof=1) / math.sqrt(vals.size))
    radius = float(stats.t.ppf(0.975, vals.size - 1) * sem)
    return mean - radius, mean + radius


def _safe_ttest_1samp(values: np.ndarray) -> tuple[float | None, float | None, str]:
    vals = values[np.isfinite(values)]
    if vals.size < 2:
        return None, None, "insufficient_n"
    if np.allclose(vals, vals[0]):
        statistic = math.inf if not np.isclose(float(vals[0]), 0.0) else 0.0
        p_value = 0.0 if not np.isclose(float(vals[0]), 0.0) else 1.0
        return statistic, p_value, "constant_values"
    test = stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
    return float(test.statistic), float(test.pvalue), "ok"


def _safe_ttest_rel(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None, str]:
    mask = np.isfinite(a) & np.isfinite(b)
    vals_a = a[mask]
    vals_b = b[mask]
    if vals_a.size < 2:
        return None, None, "insufficient_n"
    diff = vals_a - vals_b
    if np.allclose(diff, diff[0]):
        statistic = math.inf if not np.isclose(float(diff[0]), 0.0) else 0.0
        p_value = 0.0 if not np.isclose(float(diff[0]), 0.0) else 1.0
        return statistic, p_value, "constant_values"
    test = stats.ttest_rel(vals_a, vals_b, nan_policy="omit")
    return float(test.statistic), float(test.pvalue), "ok"


def _safe_wilcoxon(values: np.ndarray) -> tuple[float | None, float | None, str]:
    vals = values[np.isfinite(values)]
    vals = vals[~np.isclose(vals, 0.0)]
    if vals.size < 1:
        return 0.0, 1.0, "all_zero"
    try:
        test = stats.wilcoxon(vals, zero_method="wilcox", alternative="two-sided")
    except ValueError as exc:
        return None, None, f"error:{exc}"
    return float(test.statistic), float(test.pvalue), "ok"


def _is_one_sample_metric(metric: str) -> bool:
    lower = metric.lower()
    return (
        lower == "acc_drop"
        or lower.startswith("delta_")
        or lower.endswith("_difference")
        or lower.endswith("_effect")
        or lower.endswith("_effect_size")
        or lower.endswith("_gain")
    )


def _bh_fdr(p_values: Sequence[float | None]) -> list[float | None]:
    indexed = [(idx, float(p)) for idx, p in enumerate(p_values) if p is not None and np.isfinite(float(p))]
    out: list[float | None] = [None] * len(p_values)
    if not indexed:
        return out
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    adjusted = [0.0] * m
    running = 1.0
    for rank_from_end, (_idx, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        adjusted[rank - 1] = running
    for (idx, _), value in zip(indexed, adjusted):
        out[idx] = min(1.0, float(value))
    return out


def summarize_seed_metrics(
    df: pd.DataFrame,
    *,
    spec: ExperimentSpec,
    expected_seeds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    group_cols = _infer_group_columns(df, spec)
    metric_cols = _metric_columns(df, group_cols)
    seed_df = _seed_level_frame(df, group_cols, metric_cols)
    rows: list[dict[str, Any]] = []
    if group_cols:
        group_iter: Iterable[tuple[Any, pd.DataFrame]] = seed_df.groupby(group_cols, dropna=False, sort=True)
    else:
        group_iter = [((), seed_df)]
    for group_key, sub in group_iter:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_payload = dict(zip(group_cols, key_values))
        for metric in metric_cols:
            values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            n_seeds = int(values.size)
            sd = float(values.std(ddof=1)) if n_seeds > 1 else 0.0 if n_seeds == 1 else None
            ci_low, ci_high = _t_ci(values)
            rows.append(
                {
                    **group_payload,
                    "metric": metric,
                    "n_seeds": n_seeds,
                    "mean": float(values.mean()) if n_seeds else None,
                    "sd": sd,
                    "sem": float(sd / math.sqrt(n_seeds)) if sd is not None and n_seeds > 0 else None,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "status": "ok" if n_seeds == expected_seeds else "incomplete_n",
                }
            )
    return pd.DataFrame(rows), seed_df, group_cols, metric_cols


def compute_seed_tests(
    seed_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
    expected_seeds: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if group_cols:
        group_iter: Iterable[tuple[Any, pd.DataFrame]] = seed_df.groupby(list(group_cols), dropna=False, sort=True)
    else:
        group_iter = [((), seed_df)]
    for group_key, sub in group_iter:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_payload = dict(zip(group_cols, key_values))
        if {"acc_static", "acc_dynamic"}.issubset(sub.columns):
            paired = sub[["seed", "acc_static", "acc_dynamic"]].dropna()
            a = paired["acc_static"].to_numpy(dtype=np.float64)
            b = paired["acc_dynamic"].to_numpy(dtype=np.float64)
            diff = a - b
            t_stat, t_p, t_status = _safe_ttest_rel(a, b)
            w_stat, w_p, w_status = _safe_wilcoxon(diff)
            rows.append(
                {
                    **group_payload,
                    "test_family": "paired",
                    "metric": "acc_static_minus_acc_dynamic",
                    "comparison": "acc_static_vs_acc_dynamic",
                    "n_seeds": int(len(paired)),
                    "effect_mean": float(diff.mean()) if diff.size else None,
                    "ttest_statistic": t_stat,
                    "ttest_p_value": t_p,
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_p_value": w_p,
                    "status": "ok" if len(paired) == expected_seeds and t_status != "insufficient_n" else "incomplete_n",
                    "ttest_status": t_status,
                    "wilcoxon_status": w_status,
                }
            )
        for metric in metric_cols:
            if not _is_one_sample_metric(metric):
                continue
            vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            t_stat, t_p, t_status = _safe_ttest_1samp(vals)
            w_stat, w_p, w_status = _safe_wilcoxon(vals)
            rows.append(
                {
                    **group_payload,
                    "test_family": "one_sample_zero",
                    "metric": metric,
                    "comparison": f"{metric}_vs_zero",
                    "n_seeds": int(vals.size),
                    "effect_mean": float(vals.mean()) if vals.size else None,
                    "ttest_statistic": t_stat,
                    "ttest_p_value": t_p,
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_p_value": w_p,
                    "status": "ok" if vals.size == expected_seeds and t_status != "insufficient_n" else "incomplete_n",
                    "ttest_status": t_status,
                    "wilcoxon_status": w_status,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ttest_p_fdr"] = _bh_fdr(out["ttest_p_value"].tolist())
    out["wilcoxon_p_fdr"] = _bh_fdr(out["wilcoxon_p_value"].tolist())
    return out


def _global_pair_id(seed: int, pair_id: Any) -> int:
    return int(seed) * PAIR_ID_OFFSET + int(pair_id)


def _prepare_frame_for_root(frame: pd.DataFrame, *, csv_name: str, result: SeedRunResult, root: Path) -> pd.DataFrame:
    out = frame.copy()
    if csv_name == "pair_results.csv" and "pair_id" in out.columns:
        out["source_pair_id"] = out["pair_id"].astype(int)
        out["pair_id"] = [_global_pair_id(result.seed, value) for value in out["source_pair_id"].tolist()]
    if "seed" not in out.columns:
        out.insert(0, "seed", int(result.seed))
    else:
        out["seed"] = pd.to_numeric(out["seed"], errors="coerce").fillna(int(result.seed)).astype(int)
    try:
        rel_run_dir = result.run_dir.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel_run_dir = str(result.run_dir)
    if "run_dir" not in out.columns:
        out.insert(1, "run_dir", rel_run_dir)
    return out


def aggregate_csv_outputs(
    *,
    spec: ExperimentSpec,
    root: Path,
    results: Sequence[SeedRunResult],
    expected_seeds: int,
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    layout = prepare_result_layout(root)
    success_results = [result for result in results if result.status == "success"]
    csv_names: set[str] = set()
    sources_by_result: dict[int, dict[str, Path]] = {}
    for result in success_results:
        sources = _csv_sources_for_run(result.run_dir)
        sources_by_result[int(result.seed)] = sources
        csv_names.update(sources)

    aggregate_outputs: dict[str, dict[str, str]] = {}
    csv_summaries: list[dict[str, Any]] = []
    for csv_name in sorted(csv_names):
        frames: list[pd.DataFrame] = []
        for result in success_results:
            path = sources_by_result.get(int(result.seed), {}).get(csv_name)
            if path is None:
                continue
            frames.append(_prepare_frame_for_root(pd.read_csv(path), csv_name=csv_name, result=result, root=root))
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        sort_cols = [col for col in ("seed", "pair_id", "record_id", "trial_id", "overlap_group", "state_condition", "keep_prob") if col in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols, kind="stable").reset_index(drop=True)

        root_data_path = layout.data_dir / csv_name
        root_metrics_copy = layout.metrics_dir / csv_name
        root_data_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(root_data_path, index=False)
        combined.to_csv(root_metrics_copy, index=False)

        stem = Path(csv_name).stem
        summary_df, seed_df, group_cols, metric_cols = summarize_seed_metrics(
            combined,
            spec=spec,
            expected_seeds=expected_seeds,
        )
        tests_df = compute_seed_tests(seed_df, group_cols=group_cols, metric_cols=metric_cols, expected_seeds=expected_seeds)
        by_seed_path = layout.metrics_dir / f"{stem}__by_seed.csv"
        seed_summary_path = layout.metrics_dir / f"{stem}__seed_summary.csv"
        seed_tests_path = layout.metrics_dir / f"{stem}__seed_tests.csv"
        seed_df.to_csv(by_seed_path, index=False)
        summary_df.to_csv(seed_summary_path, index=False)
        tests_df.to_csv(seed_tests_path, index=False)
        aggregate_outputs[csv_name] = {
            "combined_csv": root_data_path.relative_to(root).as_posix(),
            "metrics_copy_csv": root_metrics_copy.relative_to(root).as_posix(),
            "by_seed_csv": by_seed_path.relative_to(root).as_posix(),
            "seed_summary_csv": seed_summary_path.relative_to(root).as_posix(),
            "seed_tests_csv": seed_tests_path.relative_to(root).as_posix(),
        }
        csv_summaries.append(
            {
                "csv_name": csv_name,
                "status": "ok",
                "n_seeds": int(combined["seed"].nunique()) if "seed" in combined.columns else 0,
                "n_rows": int(len(combined)),
                "n_seed_rows": int(len(seed_df)),
                "n_summary_rows": int(len(summary_df)),
                "n_test_rows": int(len(tests_df)),
                "group_columns": list(group_cols),
                "metric_columns": list(metric_cols),
            }
        )
    return aggregate_outputs, csv_summaries


def _copy_first_existing(success_results: Sequence[SeedRunResult], source_name: str, target_path: Path) -> int | None:
    for result in success_results:
        for base in (result.run_dir / "data", result.run_dir / "metrics", result.run_dir):
            candidate = base / source_name
            if candidate.is_file():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, target_path)
                return int(result.seed)
    return None


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _save_npz(path: Path, payload: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dict(payload))
    return path.as_posix()


def _aggregate_overlap_trace(root: Path, success_results: Sequence[SeedRunResult]) -> dict[str, Any]:
    condition_parts: list[np.ndarray] = []
    dpi_parts: list[np.ndarray] = []
    seeds: list[int] = []
    for result in success_results:
        path = result.run_dir / "data" / "pair_trace_similarity.npz"
        if not path.is_file():
            continue
        payload = _load_npz(path)
        if "condition_name" in payload and "DPI_L3" in payload:
            condition_parts.append(np.asarray(payload["condition_name"]).astype(str))
            dpi_parts.append(np.asarray(payload["DPI_L3"]))
            seeds.append(int(result.seed))
    if not dpi_parts:
        return {}
    target = root / "data" / "pair_trace_similarity.npz"
    payload = {
        "condition_name": np.concatenate(condition_parts, axis=0),
        "DPI_L3": np.concatenate(dpi_parts, axis=0),
    }
    _save_npz(target, payload)
    shutil.copy2(target, root / "metrics" / target.name)
    return {"pair_trace_similarity_npz": target.relative_to(root).as_posix(), "seeds": seeds}


def _aggregate_l3_vectors(root: Path, success_results: Sequence[SeedRunResult]) -> dict[str, Any]:
    parts: dict[str, list[np.ndarray]] = {}
    seeds: list[int] = []
    for result in success_results:
        path = result.run_dir / "data" / "pair_vectors.npz"
        if not path.is_file():
            continue
        payload = _load_npz(path)
        if "pair_id" not in payload:
            continue
        original_ids = np.asarray(payload["pair_id"], dtype=np.int64)
        global_ids = np.asarray([_global_pair_id(result.seed, item) for item in original_ids], dtype=np.int64)
        for key, value in payload.items():
            arr = np.asarray(value)
            parts.setdefault(key, []).append(global_ids if key == "pair_id" else arr)
        seeds.append(int(result.seed))
    if not parts:
        return {}
    target = root / "data" / "pair_vectors.npz"
    combined = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    _save_npz(target, combined)
    shutil.copy2(target, root / "metrics" / target.name)
    return {"pair_vectors_npz": target.relative_to(root).as_posix(), "seeds": seeds}


def _aggregate_dms_event_alignment(root: Path, success_results: Sequence[SeedRunResult]) -> dict[str, Any]:
    parts: dict[str, list[np.ndarray]] = {}
    first_1d: dict[str, np.ndarray] = {}
    seeds: list[int] = []
    for result in success_results:
        path = result.run_dir / "data" / "l1_local_event_time_alignment.npz"
        if not path.is_file():
            continue
        payload = _load_npz(path)
        for key, value in payload.items():
            arr = np.asarray(value)
            if arr.ndim >= 2:
                parts.setdefault(key, []).append(arr)
            else:
                first_1d.setdefault(key, arr)
        seeds.append(int(result.seed))
    if not parts and not first_1d:
        return {}
    target = root / "data" / "l1_local_event_time_alignment.npz"
    combined = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    combined.update(first_1d)
    _save_npz(target, combined)
    shutil.copy2(target, root / "metrics" / target.name)
    representative = _copy_first_existing(success_results, "l1_panel_a_preprobe_gain_map.npz", root / "data" / "l1_panel_a_preprobe_gain_map.npz")
    if representative is not None:
        shutil.copy2(root / "data" / "l1_panel_a_preprobe_gain_map.npz", root / "metrics" / "l1_panel_a_preprobe_gain_map.npz")
    return {
        "l1_local_event_time_alignment_npz": target.relative_to(root).as_posix(),
        "panel_a_representative_seed": representative,
        "seeds": seeds,
    }


def _aggregate_chunk_step2_representative(root: Path, success_results: Sequence[SeedRunResult]) -> dict[str, Any]:
    representative = _copy_first_existing(success_results, "episode_timeline_example.npz", root / "data" / "episode_timeline_example.npz")
    if representative is None:
        return {}
    shutil.copy2(root / "data" / "episode_timeline_example.npz", root / "metrics" / "episode_timeline_example.npz")
    return {
        "episode_timeline_example_npz": "data/episode_timeline_example.npz",
        "episode_timeline_representative_seed": representative,
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _sem(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return 0.0
    return float(vals.std(ddof=1) / math.sqrt(vals.size))


def _write_similarity_overlap_summary(root: Path) -> dict[str, Any]:
    matched_path = root / "data" / "within_bin_overlap_matched_pairs.csv"
    if not matched_path.is_file():
        return {}
    df = pd.read_csv(matched_path)
    n_pairs = int(len(df))
    summary = {
        "target_bin_label": "combined_seed_sweep",
        "target_bin_index": None,
        "n_total_in_bin": n_pairs,
        "n_high_overlap": n_pairs,
        "n_low_overlap": n_pairs,
        "n_matched_pairs": n_pairs,
        "mean_similarity_high": _safe_float(df["pixel_similarity_high"].mean()) if n_pairs and "pixel_similarity_high" in df else None,
        "mean_similarity_low": _safe_float(df["pixel_similarity_low"].mean()) if n_pairs and "pixel_similarity_low" in df else None,
        "mean_overlap_high": _safe_float(df["dice_overlap_high"].mean()) if n_pairs and "dice_overlap_high" in df else None,
        "mean_overlap_low": _safe_float(df["dice_overlap_low"].mean()) if n_pairs and "dice_overlap_low" in df else None,
        "mean_bvec_high": _safe_float(df["b_vec_high"].mean()) if n_pairs and "b_vec_high" in df else None,
        "mean_bvec_low": _safe_float(df["b_vec_low"].mean()) if n_pairs and "b_vec_low" in df else None,
        "mean_delta_bvec": _safe_float(df["delta_b_vec"].mean()) if n_pairs and "delta_b_vec" in df else None,
        "sem_delta_bvec": _safe_float(_sem(df["delta_b_vec"].to_numpy(dtype=np.float64))) if n_pairs and "delta_b_vec" in df else None,
        "acc_dynamic_high": _safe_float(df["correct_dynamic_high"].mean()) if n_pairs and "correct_dynamic_high" in df else None,
        "acc_dynamic_low": _safe_float(df["correct_dynamic_low"].mean()) if n_pairs and "correct_dynamic_low" in df else None,
        "acc_static_high": _safe_float(df["correct_static_high"].mean()) if n_pairs and "correct_static_high" in df else None,
        "acc_static_low": _safe_float(df["correct_static_low"].mean()) if n_pairs and "correct_static_low" in df else None,
        "status": "ok" if n_pairs >= 8 else "low_sample_size",
        "interpretation": "Seed-sweep aggregate recomputed from root within-bin matched pairs.",
    }
    if n_pairs and {"correct_static_high", "correct_dynamic_high", "correct_static_low", "correct_dynamic_low"}.issubset(df.columns):
        acc_drop_high = df["correct_static_high"].to_numpy(dtype=np.float64) - df["correct_dynamic_high"].to_numpy(dtype=np.float64)
        acc_drop_low = df["correct_static_low"].to_numpy(dtype=np.float64) - df["correct_dynamic_low"].to_numpy(dtype=np.float64)
        summary.update(
            {
                "acc_drop_high": _safe_float(acc_drop_high.mean()),
                "acc_drop_low": _safe_float(acc_drop_low.mean()),
                "sem_acc_drop_high": _safe_float(_sem(acc_drop_high)),
                "sem_acc_drop_low": _safe_float(_sem(acc_drop_low)),
            }
        )
    if n_pairs and {"b_vec_high", "b_vec_low"}.issubset(df.columns):
        high = df["b_vec_high"].to_numpy(dtype=np.float64)
        low = df["b_vec_low"].to_numpy(dtype=np.float64)
        delta = high - low
        if np.allclose(delta, 0.0):
            summary["wilcoxon_statistic"] = 0.0
            summary["wilcoxon_p_value"] = 1.0
        else:
            test = stats.wilcoxon(high, low, alternative="two-sided")
            summary["wilcoxon_statistic"] = _safe_float(test.statistic)
            summary["wilcoxon_p_value"] = _safe_float(test.pvalue)
    path = root / "metrics" / "within_bin_overlap_summary.json"
    save_summary_json(summary, path.parent, filename=path.name)
    return {"within_bin_overlap_summary_json": path.relative_to(root).as_posix()}


def aggregate_special_artifacts(root: Path, spec: ExperimentSpec, success_results: Sequence[SeedRunResult]) -> dict[str, Any]:
    if not success_results:
        return {}
    if spec.experiment_id == "overlap_causal_input_perturbation_experiment":
        return _aggregate_overlap_trace(root, success_results)
    if spec.experiment_id == "l3_accumulator_mechanism_experiment":
        return _aggregate_l3_vectors(root, success_results)
    if spec.experiment_id == "dms_overlap_ux_support_mechanism_experiment":
        return _aggregate_dms_event_alignment(root, success_results)
    if spec.experiment_id == "chunk_step2_fused_state_experiment":
        return _aggregate_chunk_step2_representative(root, success_results)
    if spec.experiment_id == "similarity_bias_experiment":
        return _write_similarity_overlap_summary(root)
    return {}


def aggregate_seed_sweep(
    *,
    spec: ExperimentSpec,
    root: Path,
    seeds: Sequence[int],
    results: Sequence[SeedRunResult],
    args: Any,
    normalize_bundle: NormalizeBundle,
) -> dict[str, Any]:
    layout = prepare_result_layout(root)
    sorted_results = sorted(results, key=lambda item: item.seed)
    records = [seed_result_record(result, root) for result in sorted_results]
    pd.DataFrame(records).to_csv(layout.data_dir / "seed_runs.csv", index=False)

    success_results = [result for result in sorted_results if result.status == "success"]
    aggregate_outputs, csv_summaries = aggregate_csv_outputs(
        spec=spec,
        root=root,
        results=sorted_results,
        expected_seeds=len(seeds),
    )
    special_outputs = aggregate_special_artifacts(root, spec, success_results)
    status = "success" if len(success_results) == len(seeds) else "incomplete_n"
    summary_payload = {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "status": status,
        "mode": "seed_sweep",
        "statistical_unit": "seed",
        "expected_seeds": int(len(seeds)),
        "successful_seeds": int(len(success_results)),
        "failed_seeds": int(len(seeds) - len(success_results)),
        "requested_seed_values": [int(seed) for seed in seeds],
        "successful_seed_values": [int(result.seed) for result in success_results],
        "failed_seed_values": [int(result.seed) for result in sorted_results if result.status != "success"],
        "jobs": int(args.jobs),
        "devices": list(resolve_devices(str(args.device), getattr(args, "devices", None))),
        "smoke": bool(args.smoke),
        "resume": bool(args.resume),
        "continue_on_error": bool(args.continue_on_error),
        "primary_csv": spec.primary_csv,
        "csv_summaries": csv_summaries,
        "aggregate_outputs": aggregate_outputs,
        "special_outputs": special_outputs,
        "seed_runs": records,
    }
    save_summary_json(summary_payload, layout.root)
    manifest = {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "mode": "seed_sweep",
        "files": relativize_files(root),
        "aggregate_outputs": aggregate_outputs,
        "special_outputs": special_outputs,
    }
    (layout.root / "artifact_manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    normalize_bundle(root, spec)
    manifest["files"] = relativize_files(root)
    (layout.root / "artifact_manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_payload


def run_seed_sweep(
    *,
    spec: ExperimentSpec,
    args: Any,
    runtime_python: Path,
    build_command: BuildSeedCommand,
    normalize_bundle: NormalizeBundle,
) -> int:
    if int(args.jobs) < 1:
        raise SystemExit("--jobs must be >= 1")
    seeds = resolve_seed_list(int(args.seed), args.seeds)
    devices = resolve_devices(str(args.device), args.devices)
    root = Path(args.output_dir).resolve()
    layout = prepare_result_layout(root)
    run_config = {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "mode": "seed_sweep",
        "seeds": [int(seed) for seed in seeds],
        "jobs": int(args.jobs),
        "devices": list(devices),
        "resume": bool(args.resume),
        "continue_on_error": bool(args.continue_on_error),
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "device": str(args.device),
        "smoke": bool(args.smoke),
        "config_file": str(Path(args.config).resolve()) if args.config else None,
    }
    (layout.root / "run_config.json").write_text(
        json.dumps(json_safe(run_config), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    run_info_payload = build_run_info(
        experiment_name=spec.experiment_id,
        output_dir=root,
        entry_script=f"python -m src.experiments.runners.{spec.experiment_id}",
        seed=None,
        dataset=str(args.dataset_root),
        command=subprocess.list2cmdline(os.sys.argv),
        model_path=str(args.model_path),
        config_file=str(Path(args.config).resolve()) if args.config else None,
    )
    write_run_info(layout.meta_dir, run_info_payload)

    log_path = layout.logs_dir / "seed_sweep.log"
    tasks = [
        (int(seed), root / "runs" / f"seed_{int(seed):04d}", _device_for_index(devices, index))
        for index, seed in enumerate(seeds)
    ]
    results: list[SeedRunResult] = []
    status = "failed"
    aggregated = False
    try:
        if int(args.jobs) <= 1:
            for seed, run_dir, device in tasks:
                result = run_seed_once(
                    spec=spec,
                    seed=seed,
                    run_dir=run_dir,
                    device=device,
                    model_path=str(args.model_path),
                    dataset_root=str(args.dataset_root),
                    build_command=build_command,
                    normalize_bundle=normalize_bundle,
                    resume=bool(args.resume),
                )
                results.append(result)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(json_safe(seed_result_record(result, root)), ensure_ascii=False, sort_keys=True) + "\n")
                if result.status != "success" and not bool(args.continue_on_error):
                    raise RuntimeError(f"{spec.experiment_id} failed for seed {seed}: {result.error}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.jobs)) as executor:
                future_map = {
                    executor.submit(
                        run_seed_once,
                        spec=spec,
                        seed=seed,
                        run_dir=run_dir,
                        device=device,
                        model_path=str(args.model_path),
                        dataset_root=str(args.dataset_root),
                        build_command=build_command,
                        normalize_bundle=normalize_bundle,
                        resume=bool(args.resume),
                    ): seed
                    for seed, run_dir, device in tasks
                }
                for future in concurrent.futures.as_completed(future_map):
                    seed = future_map[future]
                    result = future.result()
                    results.append(result)
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(json_safe(seed_result_record(result, root)), ensure_ascii=False, sort_keys=True) + "\n")
                    if result.status != "success" and not bool(args.continue_on_error):
                        raise RuntimeError(f"{spec.experiment_id} failed for seed {seed}: {result.error}")
            results.sort(key=lambda item: item.seed)

        summary_payload = aggregate_seed_sweep(
            spec=spec,
            root=root,
            seeds=seeds,
            results=results,
            args=args,
            normalize_bundle=normalize_bundle,
        )
        aggregated = True
        status = str(summary_payload.get("status", "success"))
        if status != "success" and not bool(args.continue_on_error):
            raise RuntimeError(f"{spec.experiment_id} seed sweep incomplete: see {root / 'summary.json'}")
        return 0
    finally:
        if results and not aggregated:
            try:
                aggregate_seed_sweep(
                    spec=spec,
                    root=root,
                    seeds=seeds,
                    results=results,
                    args=args,
                    normalize_bundle=normalize_bundle,
                )
            except Exception:
                pass
        finalize_run_info(layout.meta_dir, run_info_payload, status=status)


__all__ = [
    "SeedRunResult",
    "aggregate_seed_sweep",
    "is_successful_run_dir",
    "json_safe",
    "relativize_files",
    "resolve_devices",
    "resolve_seed_list",
    "run_seed_sweep",
]
