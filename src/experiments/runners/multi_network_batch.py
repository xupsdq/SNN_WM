from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS
from src.experiments.catalog import EXPERIMENT_SPECS, ExperimentSpec, get_experiment_spec
from src.experiments.common.results import prepare_result_layout, save_run_config, save_summary_json
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.runners._common import _resolve_runtime_python


DEFAULT_MODEL_PATH_GLOB = "results/multi_snn/sdnn_ensemble_20/sdnn_ensemble_20/seed_*/net_final.pth"
DEFAULT_OUTPUT_ROOT = "results/multi_network"
DEFAULT_EVAL_SEED = 42

METADATA_COLUMNS = ("network_seed", "network_index", "model_path", "eval_seed", "run_dir")
NETWORK_SEED_RE = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class NetworkCheckpoint:
    index: int
    seed: int
    model_path: Path


@dataclass(frozen=True)
class NetworkRunResult:
    experiment_id: str
    network_index: int
    network_seed: int
    model_path: Path
    run_dir: Path
    status: str
    returncode: int | None
    elapsed_seconds: float
    command: str
    error: str = ""
    resumed: bool = False


def _json_write(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json_ready(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _to_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_to_json_ready(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _relativize_files(root: Path) -> list[str]:
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
    return files


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (DEFAULT_PROJECT_DEFAULTS.paths.repo_root / path).resolve()


def _extract_network_seed(model_path: Path, fallback: int) -> int:
    for part in reversed(model_path.parts):
        match = NETWORK_SEED_RE.search(part)
        if match:
            return int(match.group(1))
    return int(fallback)


def discover_checkpoints(model_path_glob: str) -> list[NetworkCheckpoint]:
    pattern = str(_resolve_repo_path(model_path_glob)) if not Path(model_path_glob).is_absolute() else model_path_glob
    import glob

    paths = sorted(Path(path).resolve() for path in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No model checkpoints matched --model-path-glob: {model_path_glob}")
    checkpoints: list[NetworkCheckpoint] = []
    for index, model_path in enumerate(paths):
        if not model_path.is_file():
            continue
        checkpoints.append(
            NetworkCheckpoint(
                index=index,
                seed=_extract_network_seed(model_path, fallback=index),
                model_path=model_path,
            )
        )
    checkpoints = sorted(checkpoints, key=lambda item: (item.seed, str(item.model_path)))
    checkpoints = [NetworkCheckpoint(index=idx, seed=item.seed, model_path=item.model_path) for idx, item in enumerate(checkpoints)]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files matched --model-path-glob: {model_path_glob}")
    return checkpoints


def parse_experiment_ids(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(EXPERIMENT_SPECS.keys())
    experiment_ids = [item.strip() for item in raw.split(",") if item.strip()]
    if not experiment_ids:
        raise ValueError("--experiments must be 'all' or a comma-separated list of experiment ids.")
    for experiment_id in experiment_ids:
        get_experiment_spec(experiment_id)
    return experiment_ids


def _run_info_status(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    return str(status) if status is not None else None


def _is_successful_run_dir(run_dir: Path) -> bool:
    return (run_dir / "summary.json").is_file() and _run_info_status(run_dir / "meta" / "run_info.json") == "success"


def _build_single_run_command(
    *,
    runtime_python: Path,
    experiment_id: str,
    checkpoint: NetworkCheckpoint,
    run_dir: Path,
    dataset_root: Path,
    device: str,
    eval_seed: int,
    smoke: bool,
) -> list[str]:
    command = [
        str(runtime_python),
        "-m",
        f"src.experiments.runners.{experiment_id}",
        "--output-dir",
        str(run_dir),
        "--model-path",
        str(checkpoint.model_path),
        "--dataset-root",
        str(dataset_root),
        "--device",
        str(device),
        "--seed",
        str(int(eval_seed)),
    ]
    if smoke:
        command.append("--smoke")
    return command


def run_network_once(
    *,
    experiment_id: str,
    checkpoint: NetworkCheckpoint,
    run_dir: Path,
    runtime_python: Path,
    dataset_root: Path,
    device: str,
    eval_seed: int,
    smoke: bool,
    resume: bool,
) -> NetworkRunResult:
    command = _build_single_run_command(
        runtime_python=runtime_python,
        experiment_id=experiment_id,
        checkpoint=checkpoint,
        run_dir=run_dir,
        dataset_root=dataset_root,
        device=device,
        eval_seed=eval_seed,
        smoke=smoke,
    )
    command_text = subprocess.list2cmdline(command)
    if resume and _is_successful_run_dir(run_dir):
        return NetworkRunResult(
            experiment_id=experiment_id,
            network_index=checkpoint.index,
            network_seed=checkpoint.seed,
            model_path=checkpoint.model_path,
            run_dir=run_dir,
            status="success",
            returncode=0,
            elapsed_seconds=0.0,
            command=command_text,
            resumed=True,
        )

    started = time.time()
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    elapsed = time.time() - started
    status = "success" if completed.returncode == 0 and _is_successful_run_dir(run_dir) else "failed"
    error = ""
    if status != "success":
        error = f"returncode={completed.returncode}; see {run_dir / 'logs' / 'runner.log'}"
    return NetworkRunResult(
        experiment_id=experiment_id,
        network_index=checkpoint.index,
        network_seed=checkpoint.seed,
        model_path=checkpoint.model_path,
        run_dir=run_dir,
        status=status,
        returncode=int(completed.returncode),
        elapsed_seconds=elapsed,
        command=command_text,
        error=error,
    )


def _run_result_record(result: NetworkRunResult, root: Path) -> dict[str, Any]:
    try:
        rel_run_dir = result.run_dir.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel_run_dir = str(result.run_dir)
    return {
        "experiment_id": result.experiment_id,
        "network_index": int(result.network_index),
        "network_seed": int(result.network_seed),
        "model_path": str(result.model_path),
        "run_dir": rel_run_dir,
        "status": result.status,
        "returncode": result.returncode,
        "elapsed_seconds": float(result.elapsed_seconds),
        "resumed": bool(result.resumed),
        "error": result.error,
    }


def _complete_results(
    results: Sequence[NetworkRunResult],
    *,
    checkpoints: Sequence[NetworkCheckpoint],
    root: Path,
    experiment_id: str,
) -> list[NetworkRunResult]:
    by_seed = {int(result.network_seed): result for result in results}
    completed = list(results)
    for checkpoint in checkpoints:
        if int(checkpoint.seed) in by_seed:
            continue
        run_dir = root / "runs" / f"seed_{checkpoint.seed:04d}"
        completed.append(
            NetworkRunResult(
                experiment_id=experiment_id,
                network_index=checkpoint.index,
                network_seed=checkpoint.seed,
                model_path=checkpoint.model_path,
                run_dir=run_dir,
                status="not_run",
                returncode=None,
                elapsed_seconds=0.0,
                command="",
                error="not_run",
            )
        )
    return sorted(completed, key=lambda item: item.network_index)


def _csv_artifact_names(spec: ExperimentSpec) -> list[str]:
    names: list[str] = []
    if spec.primary_csv:
        names.append(Path(spec.primary_csv).name)
    for artifact in spec.expected_artifacts:
        path = Path(artifact)
        if path.suffix.lower() == ".csv":
            names.append(path.name)
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        out.append(name)
        seen.add(name)
    return out


def _resolve_run_csv(run_dir: Path, csv_name: str) -> Path | None:
    candidates = [
        run_dir / csv_name,
        run_dir / "data" / csv_name,
        run_dir / "metrics" / csv_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_id_column(column: str) -> bool:
    lower = column.lower()
    explicit_ids = {
        "index",
        "seed",
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
        "network_index",
        "network_seed",
        "eval_seed",
    }
    if lower in explicit_ids:
        return True
    return lower.endswith("_id") or lower.endswith("_seed")


def _infer_group_columns(df: pd.DataFrame, spec: ExperimentSpec) -> list[str]:
    group_cols: list[str] = []
    explicit = [item for item in (spec.csv_group, spec.csv_x) if item]
    for column in explicit:
        if column in df.columns and column not in group_cols:
            group_cols.append(column)
    for column in df.columns:
        if column in METADATA_COLUMNS or column in group_cols:
            continue
        if _is_id_column(column):
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
    }
    for column in df.columns:
        lower = column.lower()
        if column in METADATA_COLUMNS or column in group_cols or _is_id_column(column):
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


def _network_level_frame(df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    cols = ["network_seed", *group_cols]
    if not metric_cols:
        return pd.DataFrame(columns=cols)
    return (
        df.groupby(cols, dropna=False, sort=True)[list(metric_cols)]
        .mean(numeric_only=True)
        .reset_index()
    )


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


def summarize_network_metrics(
    df: pd.DataFrame,
    *,
    spec: ExperimentSpec,
    expected_networks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    group_cols = _infer_group_columns(df, spec)
    metric_cols = _metric_columns(df, group_cols)
    network_df = _network_level_frame(df, group_cols, metric_cols)
    rows: list[dict[str, Any]] = []
    group_iter: Iterable[tuple[Any, pd.DataFrame]]
    if group_cols:
        group_iter = network_df.groupby(group_cols, dropna=False, sort=True)
    else:
        group_iter = [((), network_df)]
    for group_key, sub in group_iter:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_payload = dict(zip(group_cols, key_values))
        for metric in metric_cols:
            values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            n_networks = int(values.size)
            mean = float(values.mean()) if n_networks else None
            sd = float(values.std(ddof=1)) if n_networks > 1 else 0.0 if n_networks == 1 else None
            sem = float(sd / math.sqrt(n_networks)) if sd is not None and n_networks > 0 else None
            ci_low, ci_high = _t_ci(values)
            status = "ok" if n_networks == expected_networks else "incomplete_n"
            rows.append(
                {
                    **group_payload,
                    "metric": metric,
                    "n_networks": n_networks,
                    "mean": mean,
                    "sd": sd,
                    "sem": sem,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "status": status,
                }
            )
    return pd.DataFrame(rows), network_df, group_cols, metric_cols


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
    for rank_from_end, (idx, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        adjusted[rank - 1] = running
    for (idx, _), value in zip(indexed, adjusted):
        out[idx] = min(1.0, float(value))
    return out


def compute_network_tests(
    network_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
    expected_networks: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_iter: Iterable[tuple[Any, pd.DataFrame]]
    if group_cols:
        group_iter = network_df.groupby(list(group_cols), dropna=False, sort=True)
    else:
        group_iter = [((), network_df)]
    for group_key, sub in group_iter:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_payload = dict(zip(group_cols, key_values))
        if {"acc_static", "acc_dynamic"}.issubset(sub.columns):
            paired = sub[["network_seed", "acc_static", "acc_dynamic"]].dropna()
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
                    "n_networks": int(len(paired)),
                    "effect_mean": float(diff.mean()) if diff.size else None,
                    "ttest_statistic": t_stat,
                    "ttest_p_value": t_p,
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_p_value": w_p,
                    "status": "ok" if len(paired) == expected_networks and t_status != "insufficient_n" else "incomplete_n",
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
                    "n_networks": int(vals.size),
                    "effect_mean": float(vals.mean()) if vals.size else None,
                    "ttest_statistic": t_stat,
                    "ttest_p_value": t_p,
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_p_value": w_p,
                    "status": "ok" if vals.size == expected_networks and t_status != "insufficient_n" else "incomplete_n",
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


def aggregate_experiment_outputs(
    *,
    spec: ExperimentSpec,
    root: Path,
    results: Sequence[NetworkRunResult],
    eval_seed: int,
    expected_networks: int,
) -> dict[str, Any]:
    layout = prepare_result_layout(root)
    records = [_run_result_record(result, root) for result in results]
    df_runs = pd.DataFrame(records)
    df_runs.to_csv(layout.data_dir / "network_runs.csv", index=False)

    success_results = [result for result in results if result.status == "success"]
    aggregate_outputs: dict[str, dict[str, str]] = {}
    csv_summaries: list[dict[str, Any]] = []
    for csv_name in _csv_artifact_names(spec):
        frames: list[pd.DataFrame] = []
        for result in success_results:
            csv_path = _resolve_run_csv(result.run_dir, csv_name)
            if csv_path is None:
                continue
            frame = pd.read_csv(csv_path)
            frame.insert(0, "run_dir", result.run_dir.resolve().relative_to(root.resolve()).as_posix())
            frame.insert(0, "eval_seed", int(eval_seed))
            frame.insert(0, "model_path", str(result.model_path))
            frame.insert(0, "network_index", int(result.network_index))
            frame.insert(0, "network_seed", int(result.network_seed))
            frames.append(frame)
        if not frames:
            csv_summaries.append({"csv_name": csv_name, "status": "missing", "n_networks": 0})
            continue
        by_network = pd.concat(frames, ignore_index=True)
        stem = Path(csv_name).stem
        compat_path = layout.data_dir / csv_name
        compat = by_network.copy()
        if "seed" not in compat.columns and "network_seed" in compat.columns:
            compat.insert(0, "seed", compat["network_seed"].astype(int))
        compat.to_csv(compat_path, index=False)
        by_network_path = layout.data_dir / f"{stem}__by_network.csv"
        by_network.to_csv(by_network_path, index=False)
        summary_df, network_df, group_cols, metric_cols = summarize_network_metrics(
            by_network,
            spec=spec,
            expected_networks=expected_networks,
        )
        tests_df = compute_network_tests(
            network_df,
            group_cols=group_cols,
            metric_cols=metric_cols,
            expected_networks=expected_networks,
        )
        summary_path = layout.metrics_dir / f"{stem}__network_summary.csv"
        tests_path = layout.metrics_dir / f"{stem}__network_tests.csv"
        summary_df.to_csv(summary_path, index=False)
        tests_df.to_csv(tests_path, index=False)
        aggregate_outputs[csv_name] = {
            "compat_csv": compat_path.relative_to(root).as_posix(),
            "by_network_csv": by_network_path.relative_to(root).as_posix(),
            "network_summary_csv": summary_path.relative_to(root).as_posix(),
            "network_tests_csv": tests_path.relative_to(root).as_posix(),
        }
        csv_summaries.append(
            {
                "csv_name": csv_name,
                "status": "ok",
                "n_networks": int(by_network["network_seed"].nunique()),
                "n_rows": int(len(by_network)),
                "n_summary_rows": int(len(summary_df)),
                "n_test_rows": int(len(tests_df)),
                "group_columns": list(group_cols),
                "metric_columns": list(metric_cols),
            }
        )

    success_count = int(sum(1 for result in results if result.status == "success"))
    status = "complete" if success_count == expected_networks else "incomplete_n"
    summary_payload = {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "status": status,
        "expected_networks": int(expected_networks),
        "successful_networks": success_count,
        "failed_networks": int(expected_networks - success_count),
        "eval_seed": int(eval_seed),
        "statistical_unit": "network",
        "primary_csv": spec.primary_csv,
        "csv_summaries": csv_summaries,
        "aggregate_outputs": aggregate_outputs,
    }
    save_summary_json(summary_payload, layout.root)
    manifest = {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "files": _relativize_files(root),
        "aggregate_outputs": aggregate_outputs,
    }
    _json_write(layout.root / "artifact_manifest.json", manifest)
    return summary_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mainline experiments across multiple trained SDNN checkpoints.")
    parser.add_argument("--experiments", type=str, default="all", help="'all' or comma-separated experiment ids.")
    parser.add_argument("--model-path-glob", type=str, default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PROJECT_DEFAULTS.paths.dataset_root))
    parser.add_argument("--device", type=str, default=DEFAULT_PROJECT_DEFAULTS.runtime.device)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def run_experiment_batch(
    *,
    spec: ExperimentSpec,
    checkpoints: Sequence[NetworkCheckpoint],
    output_root: Path,
    runtime_python: Path,
    dataset_root: Path,
    device: str,
    eval_seed: int,
    smoke: bool,
    jobs: int,
    resume: bool,
    continue_on_error: bool,
) -> dict[str, Any]:
    root = output_root / spec.experiment_id
    layout = prepare_result_layout(root)
    run_config = {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "model_path_glob_count": int(len(checkpoints)),
        "model_paths": [str(checkpoint.model_path) for checkpoint in checkpoints],
        "network_seeds": [int(checkpoint.seed) for checkpoint in checkpoints],
        "output_root": str(output_root),
        "eval_seed": int(eval_seed),
        "dataset_root": str(dataset_root),
        "device": str(device),
        "smoke": bool(smoke),
        "jobs": int(jobs),
        "resume": bool(resume),
        "continue_on_error": bool(continue_on_error),
    }
    save_run_config(run_config, layout.root)
    run_info_payload = build_run_info(
        experiment_name=f"multi_network.{spec.experiment_id}",
        output_dir=root,
        entry_script="python -m src.experiments.runners.multi_network_batch",
        seed=int(eval_seed),
        dataset=str(dataset_root),
        command=subprocess.list2cmdline(os.sys.argv),
        model_path=None,
    )
    write_run_info(layout.meta_dir, run_info_payload)

    log_path = layout.logs_dir / "multi_network_batch.log"
    results: list[NetworkRunResult] = []
    status = "failed"
    aggregated = False
    try:
        tasks = []
        for checkpoint in checkpoints:
            run_dir = root / "runs" / f"seed_{checkpoint.seed:04d}"
            tasks.append((checkpoint, run_dir))
        if jobs <= 1:
            for checkpoint, run_dir in tasks:
                print(f"[{spec.experiment_id}] network_seed={checkpoint.seed} -> {run_dir}")
                result = run_network_once(
                    experiment_id=spec.experiment_id,
                    checkpoint=checkpoint,
                    run_dir=run_dir,
                    runtime_python=runtime_python,
                    dataset_root=dataset_root,
                    device=device,
                    eval_seed=eval_seed,
                    smoke=smoke,
                    resume=resume,
                )
                results.append(result)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(_to_json_ready(_run_result_record(result, root)), ensure_ascii=False, sort_keys=True) + "\n")
                if result.status != "success" and not continue_on_error:
                    raise RuntimeError(f"{spec.experiment_id} failed for network seed {checkpoint.seed}: {result.error}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(jobs)) as executor:
                future_map = {
                    executor.submit(
                        run_network_once,
                        experiment_id=spec.experiment_id,
                        checkpoint=checkpoint,
                        run_dir=run_dir,
                        runtime_python=runtime_python,
                        dataset_root=dataset_root,
                        device=device,
                        eval_seed=eval_seed,
                        smoke=smoke,
                        resume=resume,
                    ): checkpoint
                    for checkpoint, run_dir in tasks
                }
                for future in concurrent.futures.as_completed(future_map):
                    checkpoint = future_map[future]
                    result = future.result()
                    results.append(result)
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(_to_json_ready(_run_result_record(result, root)), ensure_ascii=False, sort_keys=True) + "\n")
                    if result.status != "success" and not continue_on_error:
                        raise RuntimeError(f"{spec.experiment_id} failed for network seed {checkpoint.seed}: {result.error}")
            results.sort(key=lambda item: item.network_index)
        summary_payload = aggregate_experiment_outputs(
            spec=spec,
            root=root,
            results=_complete_results(results, checkpoints=checkpoints, root=root, experiment_id=spec.experiment_id),
            eval_seed=eval_seed,
            expected_networks=len(checkpoints),
        )
        aggregated = True
        status = "success" if summary_payload["status"] == "complete" else "incomplete_n"
        return summary_payload
    finally:
        if not aggregated:
            aggregate_experiment_outputs(
                spec=spec,
                root=root,
                results=_complete_results(results, checkpoints=checkpoints, root=root, experiment_id=spec.experiment_id),
                eval_seed=eval_seed,
                expected_networks=len(checkpoints),
            )
        finalize_run_info(layout.meta_dir, run_info_payload, status=status)


def main() -> int:
    args = build_parser().parse_args()
    if int(args.jobs) < 1:
        raise SystemExit("--jobs must be >= 1")
    experiment_ids = parse_experiment_ids(args.experiments)
    checkpoints = discover_checkpoints(args.model_path_glob)
    output_root = _resolve_repo_path(args.output_root)
    dataset_root = _resolve_repo_path(args.dataset_root)
    runtime_python = _resolve_runtime_python()
    print(f"[Setup] experiments={len(experiment_ids)} checkpoints={len(checkpoints)} eval_seed={args.eval_seed}")
    print(f"[Setup] runtime_python={runtime_python}")
    failures: list[str] = []
    for experiment_id in experiment_ids:
        spec = get_experiment_spec(experiment_id)
        try:
            run_experiment_batch(
                spec=spec,
                checkpoints=checkpoints,
                output_root=output_root,
                runtime_python=runtime_python,
                dataset_root=dataset_root,
                device=str(args.device),
                eval_seed=int(args.eval_seed),
                smoke=bool(args.smoke),
                jobs=int(args.jobs),
                resume=bool(args.resume),
                continue_on_error=bool(args.continue_on_error),
            )
        except Exception as exc:
            failures.append(f"{experiment_id}: {exc}")
            if not args.continue_on_error:
                raise
    if failures:
        print("[Done] completed with failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("[Done] multi-network batch complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
