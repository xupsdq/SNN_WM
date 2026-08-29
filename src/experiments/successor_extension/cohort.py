"""Resumable sequential cohort driver for the 20-seed confirmatory successor-extension run.

Runs the portable K10 specs/input bank once, then for each seed 1000..1019 the
k10 history bank and experiments A/B/C. GPU work is strictly sequential: every
task executes in its own subprocess and only one runs at a time.

Resume semantics: a task is skipped only if it passes the explicit
completeness/identity check in ``check_task``; anything else is re-run with
bounded retries. Frozen parents are verified against pinned SHA256 both before
and after the cohort.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.experiments.paper_figures.fig2.fixed_b_substrate import resolve_fixed_b_model_path
from src.experiments.paper_figures.run_paper_figures import DEFAULT_MODEL_PATH_GLOB
from src.experiments.successor_extension.aggregate import run_aggregate
from src.experiments.successor_extension.core import (
    FROZEN_PROTOCOL_DIR,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TASK_EXP_A,
    TASK_EXP_B,
    TASK_EXP_C,
    TASK_K10_HISTORY,
    TASK_K10_INPUT,
    TASK_K10_SPECS,
)
from src.experiments.successor_extension.runtime import (
    repository_root,
    resolve_repo_path,
    sha256_file,
    write_json,
)

DEFAULT_OUTPUT_ROOT = "results/successor_extension_v1_confirmatory_20seed"
COHORT_SEEDS = tuple(range(1000, 1020))
SENSITIVITY_SEEDS = tuple(range(1001, 1020))
PER_SEED_TASKS = (TASK_K10_HISTORY, TASK_EXP_A, TASK_EXP_B, TASK_EXP_C)

# Pinned provenance: frozen parents and prior C5 evidence must not change.
BASELINE_HASHES = (
    (
        "c5_l2_successor_closure.py",
        "src/experiments/c5_l2_successor_closure.py",
        "98e201a6c128b82da192eecdc36aee754f2479501c687fc2841d3d8fa5da27fe",
    ),
    (
        "fixed_b_specs.py",
        "src/experiments/paper_figures/fig2/subexperiments/fixed_b_specs.py",
        "ea044bb57b176dbad62d4b3cc3fe212215fca135c3d5b8d744dc3d6f27d0fffb",
    ),
    (
        "frozen_protocol_cache_key",
        f"{FROZEN_PROTOCOL_DIR}/cache_key.json",
        "c9b2bac5ae0ce80a3fe31cfe1bd2574860984a28a5c5f05df0b1ca30be22ca0b",
    ),
    (
        "c5_population_inference",
        "results/causal_closure_multi_seed_20260803/c5_l2_successor/aggregate/data/metrics/c5_population_inference.csv",
        "1b412cd9bf4ed9e21c603e7b770541126c779ece39c5a9ba7b7ccee89c397927",
    ),
)

EXPECTED_METRIC_FILES = {
    TASK_EXP_A: (
        "c5_k10_cell_metrics.csv",
        "c5_k10_endpoint_summary.csv",
        "c5_k10_identity_audit.csv",
        "summary.json",
        "task_manifest.json",
    ),
    TASK_EXP_B: (
        "exp_b_cell_metrics.csv",
        "exp_b_mask_audit.csv",
        "exp_b_network_summary.csv",
        "summary.json",
        "task_manifest.json",
    ),
    TASK_EXP_C: (
        "exp_c_cell_metrics.csv",
        "exp_c_identity_audit.csv",
        "exp_c_network_summary.csv",
        "summary.json",
        "task_manifest.json",
    ),
}

FINITE_ENDPOINT_FIELDS = {
    TASK_EXP_A: ("mean_transfer",),
    TASK_EXP_B: ("mean_overlap_specific_margin",),
    TASK_EXP_C: ("mean_donor_transfer",),
}


@dataclass(frozen=True)
class CohortConfig:
    output_root: str = DEFAULT_OUTPUT_ROOT
    seeds: tuple[int, ...] = COHORT_SEEDS
    sensitivity_seeds: tuple[int, ...] = SENSITIVITY_SEEDS
    device: str = "cuda"
    families: int = 6
    anchors: int = 20
    anchors_per_chunk: int = 5
    bootstrap_draws: int = 5000
    aggregate_bootstrap_draws: int = 20_000
    max_retries: int = 2
    retry_wait_seconds: float = 10.0
    dry_run: bool = False


def _log(log_path: Path, message: str) -> None:
    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "message": str(message),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[cohort] {message}", flush=True)


def _verify_baseline_hashes(repo_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    failures: list[str] = []
    for name, relative, pinned in BASELINE_HASHES:
        actual = sha256_file(repo_root / relative)
        observed[name] = actual
        if actual != pinned:
            failures.append(f"{name}: expected {pinned}, observed {actual}")
    if failures:
        raise RuntimeError("Frozen baseline hash mismatch: " + "; ".join(failures))
    return observed


def _specs_check(root: Path) -> tuple[bool, list[str]]:
    task_dir = root / TASK_K10_SPECS
    required = ("history_specs.csv", "cache_key.json", "task_manifest.json")
    missing = [name for name in required if not (task_dir / name).exists()]
    return (not missing), missing


def _input_bank_check(root: Path) -> tuple[bool, list[str]]:
    task_dir = root / TASK_K10_INPUT
    required = ("cache_key.json", "task_manifest.json", "manifest.csv", "arrays.npz")
    missing = [name for name in required if not (task_dir / name).exists()]
    return (not missing), missing


def _history_check(root: Path, seed: int) -> tuple[bool, list[str]]:
    artifact_dir = root / f"seed_{int(seed)}" / "data" / "intermediates" / TASK_K10_HISTORY
    problems: list[str] = []
    for name in ("cache_key.json", "task_manifest.json"):
        if not (artifact_dir / name).exists():
            problems.append(f"{artifact_dir.name}/{name} missing")
    audit_csv = root / f"seed_{int(seed)}" / "data" / "metrics" / "k10_history_bank_k5_identity_audit.csv"
    if not audit_csv.exists():
        problems.append("k10_history_bank_k5_identity_audit.csv missing")
    else:
        audit = pd.read_csv(audit_csv)
        if "bitwise_equal" not in audit.columns or not audit["bitwise_equal"].eq(1).all():
            problems.append("k5 checkpoint identity audit not all bitwise_equal")
    return (not problems), problems


def _experiment_check(root: Path, seed: int, task: str) -> tuple[bool, list[str]]:
    out_dir = root / f"seed_{int(seed)}" / "data" / "metrics" / task
    problems: list[str] = []
    for name in EXPECTED_METRIC_FILES[task]:
        if not (out_dir / name).exists():
            problems.append(f"{task}/{name} missing")
            continue
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            problems.append(f"{task}/summary.json status={summary.get('status')!r}")
        if int(summary.get("network_seed", -1)) != int(seed):
            problems.append(f"{task}/summary.json network_seed mismatch")
        endpoints = summary.get("endpoints")
        if not isinstance(endpoints, dict) or not endpoints:
            problems.append(f"{task}/summary.json has no endpoints")
        else:
            for endpoint, payload in endpoints.items():
                for field in FINITE_ENDPOINT_FIELDS[task]:
                    value = payload.get(field)
                    if value is None or not math.isfinite(float(value)):
                        problems.append(f"{task}/{endpoint}.{field} missing or non-finite")
    manifest_path = out_dir / "task_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("task_id") != task:
            problems.append(f"{task}/task_manifest.json task_id mismatch")
    identity_audit = {
        TASK_EXP_A: out_dir / "c5_k10_identity_audit.csv",
        TASK_EXP_C: out_dir / "exp_c_identity_audit.csv",
    }.get(task)
    if identity_audit is not None and identity_audit.exists():
        audit = pd.read_csv(identity_audit)
        if "identity_pass" not in audit.columns or not audit["identity_pass"].eq(1).all():
            problems.append(f"{task} identity audit not all identity_pass=1")
    return (not problems), problems


def check_task(root: Path, seed: int, task: str) -> tuple[bool, list[str]]:
    """Explicit completeness/identity gate for one (seed, task) output."""
    if task == TASK_K10_HISTORY:
        return _history_check(root, int(seed))
    if task in EXPECTED_METRIC_FILES:
        return _experiment_check(root, int(seed), task)
    raise ValueError(f"Unknown cohort task: {task!r}")


def _task_command(cfg: CohortConfig, seed: int, task: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.experiments.successor_extension.runner",
        "--task",
        task,
        "--output-root",
        str(cfg.output_root),
        "--network-seed",
        str(int(seed)),
        "--device",
        str(cfg.device),
        "--families",
        str(int(cfg.families)),
        "--anchors",
        str(int(cfg.anchors)),
        "--anchors-per-chunk",
        str(int(cfg.anchors_per_chunk)),
        "--bootstrap-draws",
        str(int(cfg.bootstrap_draws)),
    ]


def _run_one_task(
    cfg: CohortConfig,
    repo_root: Path,
    root: Path,
    seed: int,
    task: str,
    logs_dir: Path,
    log_path: Path,
) -> bool:
    """Run one task in a subprocess with bounded retries. Sequential by construction."""
    for attempt in range(int(cfg.max_retries) + 1):
        label = f"seed_{seed} task={task} attempt={attempt + 1}/{cfg.max_retries + 1}"
        _log(log_path, f"run {label}")
        task_log = logs_dir / f"seed_{int(seed)}_{task}_attempt{attempt + 1}.log"
        command = _task_command(cfg, int(seed), task)
        with task_log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command, cwd=str(repo_root), stdout=handle, stderr=subprocess.STDOUT
            )
        if result.returncode != 0:
            _log(log_path, f"failed rc={result.returncode} {label} (log: {task_log.name})")
        else:
            complete, problems = (
                check_task(root, int(seed), task)
                if task != TASK_K10_INPUT and task != TASK_K10_SPECS
                else (_specs_check(root) if task == TASK_K10_SPECS else _input_bank_check(root))
            )
            if complete:
                _log(log_path, f"completed {label}")
                return True
            _log(
                log_path,
                f"post-run completeness check failed {label}: {problems}",
            )
        if attempt < int(cfg.max_retries):
            _log(log_path, f"retrying {label} in {cfg.retry_wait_seconds}s")
            time.sleep(float(cfg.retry_wait_seconds))
    _log(log_path, f"gave up {label} after {int(cfg.max_retries) + 1} attempts")
    return False


def _write_seed_manifest(cfg: CohortConfig, root: Path, seed: int) -> None:
    """Generic per-seed metadata: identity, config, task status, key hashes."""
    model_path = resolve_fixed_b_model_path(
        None, DEFAULT_MODEL_PATH_GLOB, int(seed), smoke=False,
    )
    tasks: dict[str, dict[str, Any]] = {}
    for task in PER_SEED_TASKS:
        complete, problems = check_task(root, int(seed), task)
        summary_path = (
            root / f"seed_{int(seed)}" / "data" / "metrics" / task / "summary.json"
            if task in EXPECTED_METRIC_FILES
            else None
        )
        tasks[task] = {
            "status": "completed" if complete else "incomplete",
            "completion_check_problems": problems,
            "summary_sha256": (
                sha256_file(summary_path) if summary_path is not None and summary_path.exists() else ""
            ),
        }
    write_json(
        root / f"seed_{int(seed)}" / "seed_manifest.json",
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "network_seed": int(seed),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path) if model_path.exists() else "missing",
            "dataset_root": "test split of the shared MNIST skeleton dataset",
            "device": str(cfg.device),
            "families": int(cfg.families),
            "anchors": int(cfg.anchors),
            "anchors_per_chunk": int(cfg.anchors_per_chunk),
            "bootstrap_draws": int(cfg.bootstrap_draws),
            "tasks": tasks,
        },
    )


def _plan(cfg: CohortConfig, root: Path) -> dict[str, Any]:
    plan: dict[str, Any] = {"specs": None, "input_bank": None, "seeds": {}}
    specs_ok, specs_missing = _specs_check(root)
    plan["specs"] = {"complete": specs_ok, "missing": specs_missing}
    input_ok, input_missing = _input_bank_check(root)
    plan["input_bank"] = {"complete": input_ok, "missing": input_missing}
    for seed in cfg.seeds:
        plan["seeds"][int(seed)] = {}
        for task in PER_SEED_TASKS:
            complete, problems = check_task(root, int(seed), task)
            plan["seeds"][int(seed)][task] = {
                "complete": complete,
                "problems": problems,
            }
    return plan


def run_cohort(cfg: CohortConfig) -> dict[str, Any]:
    repo_root = repository_root()
    root = resolve_repo_path(cfg.output_root)
    root.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "cohort_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = root / "cohort_log.jsonl"

    _log(log_path, f"cohort start output_root={root} seeds={list(cfg.seeds)} device={cfg.device}")
    baseline = _verify_baseline_hashes(repo_root)
    _log(log_path, f"baseline hashes verified: {sorted(baseline)}")

    plan = _plan(cfg, root)
    if cfg.dry_run:
        _log(log_path, "dry-run plan computed; not executing any GPU work")
        write_json(root / "cohort_dry_run_plan.json", plan)
        return {"status": "dry_run", "plan": plan}

    failures: list[str] = []

    specs_ok, _ = _specs_check(root)
    if not specs_ok:
        _log(log_path, "building portable K10 specs")
        if not _run_one_task(cfg, repo_root, root, 0, TASK_K10_SPECS, logs_dir, log_path):
            failures.append(TASK_K10_SPECS)
    else:
        _log(log_path, "K10 specs already complete; skipping")

    input_ok, _ = _input_bank_check(root)
    if not input_ok:
        _log(log_path, "building portable K10 input bank (encoded once with the first cohort model)")
        if not _run_one_task(cfg, repo_root, root, cfg.seeds[0], TASK_K10_INPUT, logs_dir, log_path):
            failures.append(TASK_K10_INPUT)
    else:
        _log(log_path, "K10 input bank already complete; skipping")

    for seed in cfg.seeds:
        for task in PER_SEED_TASKS:
            complete, problems = check_task(root, int(seed), task)
            if complete:
                _log(log_path, f"seed_{seed} task={task} already complete; skipping")
                continue
            _log(log_path, f"seed_{seed} task={task} incomplete ({problems}); running")
            if not _run_one_task(cfg, repo_root, root, int(seed), task, logs_dir, log_path):
                failures.append(f"seed_{seed}/{task}")
        _write_seed_manifest(cfg, root, int(seed))
        _log(log_path, f"seed_{seed} manifest written")

    # frozen parents must be untouched
    after = _verify_baseline_hashes(repo_root)
    _log(log_path, f"baseline hashes re-verified after cohort: {sorted(after)}")

    coverage: dict[int, list[str]] = {}
    all_complete = True
    for seed in cfg.seeds:
        missing = []
        for task in PER_SEED_TASKS:
            complete, _ = check_task(root, int(seed), task)
            if not complete:
                missing.append(task)
        coverage[int(seed)] = missing
        if missing:
            all_complete = False

    aggregate_status = "not_run"
    if all_complete and not failures:
        _log(log_path, "exact 20-seed coverage reached; running population aggregate")
        try:
            run_aggregate(
                output_root=root,
                seeds=cfg.seeds,
                sensitivity_seeds=cfg.sensitivity_seeds,
                bootstrap_draws=int(cfg.aggregate_bootstrap_draws),
            )
            aggregate_status = "completed"
            _log(log_path, "aggregate completed")
        except Exception as error:  # aggregate bug must not erase the cohort evidence
            aggregate_status = f"failed: {error}"
            _log(log_path, f"aggregate failed: {error}")
            failures.append(f"aggregate: {error}")
    else:
        _log(
            log_path,
            f"aggregate skipped: coverage complete={all_complete} failures={failures}",
        )

    status = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if (all_complete and not failures and aggregate_status == "completed") else "failed",
        "output_root": str(root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds_requested": [int(seed) for seed in cfg.seeds],
        "seeds_fully_complete": [int(seed) for seed, missing in coverage.items() if not missing],
        "coverage_missing_tasks": coverage,
        "failures": failures,
        "aggregate_status": aggregate_status,
        "baseline_hashes_before": baseline,
        "baseline_hashes_after": after,
        "baseline_hashes_unchanged": baseline == after,
        "device": str(cfg.device),
    }
    write_json(root / "cohort_status.json", status)
    _log(log_path, f"cohort finished status={status['status']} aggregate={aggregate_status}")
    return status


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    defaults = CohortConfig()
    parser = argparse.ArgumentParser(
        description="Resumable sequential 20-seed confirmatory successor-extension cohort.",
        allow_abbrev=False,
    )
    parser.add_argument("--output-root", default=defaults.output_root)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(defaults.seeds))
    parser.add_argument("--sensitivity-seeds", type=int, nargs="+", default=list(defaults.sensitivity_seeds))
    parser.add_argument("--device", default=defaults.device, choices=("auto", "cpu", "cuda"))
    parser.add_argument("--families", type=int, default=defaults.families)
    parser.add_argument("--anchors", type=int, default=defaults.anchors)
    parser.add_argument("--anchors-per-chunk", type=int, default=defaults.anchors_per_chunk)
    parser.add_argument("--bootstrap-draws", type=int, default=defaults.bootstrap_draws)
    parser.add_argument("--aggregate-bootstrap-draws", type=int, default=defaults.aggregate_bootstrap_draws)
    parser.add_argument("--max-retries", type=int, default=defaults.max_retries)
    parser.add_argument("--retry-wait-seconds", type=float, default=defaults.retry_wait_seconds)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = CohortConfig(
        output_root=str(args.output_root),
        seeds=tuple(int(value) for value in args.seeds),
        sensitivity_seeds=tuple(int(value) for value in args.sensitivity_seeds),
        device=str(args.device),
        families=max(1, int(args.families)),
        anchors=max(1, int(args.anchors)),
        anchors_per_chunk=max(1, int(args.anchors_per_chunk)),
        bootstrap_draws=max(100, int(args.bootstrap_draws)),
        aggregate_bootstrap_draws=max(100, int(args.aggregate_bootstrap_draws)),
        max_retries=max(0, int(args.max_retries)),
        retry_wait_seconds=max(0.0, float(args.retry_wait_seconds)),
        dry_run=bool(args.dry_run),
    )
    status = run_cohort(cfg)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status["status"] in ("completed", "dry_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
