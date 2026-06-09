from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT_TASK = "shared_sequence_root_bank"
FIG3_EXPERIMENT = "fig3_multiitem_peak_landscape"
FIG6_EXPERIMENT = "fig6_peak_amplified_reentry"
METADATA_COLUMNS = {"artifact_digest", "cache_key_digest", "sha256", "table_digest"}
SPEC_FILES = ("sequence_trials.csv", "singleton_reference_trials.csv", "partial_cue_trials.csv")


@dataclass
class CommandResult:
    label: str
    command: list[str]
    cwd: str
    returncode: int
    wall_seconds: float
    started_at: str
    finished_at: str
    log_path: str
    output_root: str
    artifact_root: str
    task_artifact_dir: str
    run_info: dict[str, Any] | None
    summary: dict[str, Any] | None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bench_root = _resolve_path(args.output_root) / run_id
    logs_dir = bench_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    baseline_root = bench_root / "baseline_root"
    optimized_root = bench_root / "optimized_root"
    fig3_baseline_root = bench_root / "fig3_baseline_consumer"
    fig3_optimized_root = bench_root / "fig3_optimized_consumer"
    fig6_baseline_root = bench_root / "fig6_baseline_consumer"
    fig6_optimized_root = bench_root / "fig6_optimized_consumer"
    baseline_artifact_root = baseline_root / "data" / "intermediates"
    optimized_artifact_root = optimized_root / "data" / "intermediates"
    baseline_shared_root = baseline_artifact_root / ROOT_TASK
    optimized_shared_root = optimized_artifact_root / ROOT_TASK

    results: list[CommandResult] = []
    try:
        results.append(
            _run_command(
                "baseline_root",
                _shared_root_command(args, baseline_root, baseline_artifact_root, enable_state_bank_batch=False),
                repo_root=repo_root,
                logs_dir=logs_dir,
                output_root=baseline_root,
                artifact_root=baseline_artifact_root,
                task_artifact_dir=baseline_shared_root,
            )
        )
        results.append(
            _run_command(
                "optimized_root",
                _shared_root_command(args, optimized_root, optimized_artifact_root, enable_state_bank_batch=True),
                repo_root=repo_root,
                logs_dir=logs_dir,
                output_root=optimized_root,
                artifact_root=optimized_artifact_root,
                task_artifact_dir=optimized_shared_root,
            )
        )
        _compare_shared_specs(baseline_shared_root, optimized_shared_root)
        root_compare = _compare_root_outputs(baseline_shared_root, optimized_shared_root, atol=float(args.atol), rtol=float(args.rtol))

        fig3_baseline_artifact = fig3_baseline_root / "data" / "intermediates"
        fig3_optimized_artifact = fig3_optimized_root / "data" / "intermediates"
        fig6_baseline_artifact = fig6_baseline_root / "data" / "intermediates"
        fig6_optimized_artifact = fig6_optimized_root / "data" / "intermediates"
        results.append(
            _run_command(
                "fig3_baseline_consumer",
                _fig_consumer_command(args, "fig3", fig3_baseline_root, fig3_baseline_artifact, baseline_shared_root),
                repo_root=repo_root,
                logs_dir=logs_dir,
                output_root=fig3_baseline_root,
                artifact_root=fig3_baseline_artifact,
                task_artifact_dir=fig3_baseline_root / FIG3_EXPERIMENT / f"seed_{int(args.network_seed)}",
            )
        )
        results.append(
            _run_command(
                "fig3_optimized_consumer",
                _fig_consumer_command(args, "fig3", fig3_optimized_root, fig3_optimized_artifact, optimized_shared_root),
                repo_root=repo_root,
                logs_dir=logs_dir,
                output_root=fig3_optimized_root,
                artifact_root=fig3_optimized_artifact,
                task_artifact_dir=fig3_optimized_root / FIG3_EXPERIMENT / f"seed_{int(args.network_seed)}",
            )
        )
        results.append(
            _run_command(
                "fig6_baseline_consumer",
                _fig_consumer_command(args, "fig6", fig6_baseline_root, fig6_baseline_artifact, baseline_shared_root),
                repo_root=repo_root,
                logs_dir=logs_dir,
                output_root=fig6_baseline_root,
                artifact_root=fig6_baseline_artifact,
                task_artifact_dir=fig6_baseline_root / FIG6_EXPERIMENT / f"seed_{int(args.network_seed)}",
            )
        )
        results.append(
            _run_command(
                "fig6_optimized_consumer",
                _fig_consumer_command(args, "fig6", fig6_optimized_root, fig6_optimized_artifact, optimized_shared_root),
                repo_root=repo_root,
                logs_dir=logs_dir,
                output_root=fig6_optimized_root,
                artifact_root=fig6_optimized_artifact,
                task_artifact_dir=fig6_optimized_root / FIG6_EXPERIMENT / f"seed_{int(args.network_seed)}",
            )
        )

        fig3_baseline_seed = fig3_baseline_root / FIG3_EXPERIMENT / f"seed_{int(args.network_seed)}"
        fig3_optimized_seed = fig3_optimized_root / FIG3_EXPERIMENT / f"seed_{int(args.network_seed)}"
        fig6_baseline_seed = fig6_baseline_root / FIG6_EXPERIMENT / f"seed_{int(args.network_seed)}"
        fig6_optimized_seed = fig6_optimized_root / FIG6_EXPERIMENT / f"seed_{int(args.network_seed)}"
        fig3_regression = _run_check(
            "fig3_regression",
            [
                str(args.python),
                "scripts/regression_compare_fig_outputs.py",
                "--old-root",
                str(fig3_baseline_seed),
                "--new-root",
                str(fig3_optimized_seed),
                "--atol",
                str(float(args.atol)),
                "--rtol",
                str(float(args.rtol)),
                "--ignore-rel-prefix",
                "data/intermediates",
            ],
            repo_root=repo_root,
            logs_dir=logs_dir,
        )
        fig6_regression = _run_check(
            "fig6_regression",
            [
                str(args.python),
                "scripts/regression_compare_fig_outputs.py",
                "--old-root",
                str(fig6_baseline_seed),
                "--new-root",
                str(fig6_optimized_seed),
                "--atol",
                str(float(args.atol)),
                "--rtol",
                str(float(args.rtol)),
                "--ignore-rel-prefix",
                "data/intermediates",
            ],
            repo_root=repo_root,
            logs_dir=logs_dir,
        )
        fig3_layout = _run_check(
            "fig3_optimized_layout",
            [str(args.python), "scripts/validate_results_layout.py", "--input-dir", str(fig3_optimized_seed)],
            repo_root=repo_root,
            logs_dir=logs_dir,
        )
        fig6_layout = _run_check(
            "fig6_optimized_layout",
            [str(args.python), "scripts/validate_results_layout.py", "--input-dir", str(fig6_optimized_seed)],
            repo_root=repo_root,
            logs_dir=logs_dir,
        )
        checks = [root_compare, fig3_regression, fig6_regression, fig3_layout, fig6_layout]
        all_ok = all(item["status"] == "pass" for item in checks)
    except Exception as exc:
        checks = [{"label": "benchmark_exception", "status": "fail", "seconds": 0.0, "log_path": "", "notes": str(exc)}]
        all_ok = False

    payload = _summary_payload(args, bench_root, run_id, results, checks)
    _write_outputs(bench_root, payload)
    _write_verification_report(bench_root, payload)
    return 0 if all_ok else 1


def _shared_root_command(args: argparse.Namespace, output_root: Path, artifact_root: Path, *, enable_state_bank_batch: bool) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "src.experiments.paper_figures.common.sequence_root.run_task",
        "--task",
        ROOT_TASK,
        "--reuse-artifacts",
        "force",
        "--output-dir",
        str(output_root),
        "--artifact-root",
        str(artifact_root),
        "--network-seed",
        str(int(args.network_seed)),
        "--device",
        str(args.device),
        "--dataset-root",
        str(args.dataset_root),
        "--model-path-glob",
        str(args.model_path_glob),
        "--sequence-lengths",
        str(args.sequence_lengths),
        "--num-sequences",
        str(int(args.num_sequences)),
        "--sample-ms",
        str(int(args.sample_ms)),
        "--delay-ms",
        str(int(args.delay_ms)),
        "--batch-size",
        str(int(args.batch_size)),
        "--state-bank-singleton-batch-size",
        str(max(1, int(args.state_bank_singleton_batch_size))),
        "--no-progress",
    ]
    if bool(args.smoke):
        command.append("--smoke")
    if args.model_path:
        command.extend(["--model-path", str(args.model_path)])
    if enable_state_bank_batch:
        command.append("--enable-state-bank-batch")
    return command


def _fig_consumer_command(args: argparse.Namespace, fig_id: str, output_root: Path, artifact_root: Path, shared_root: Path) -> list[str]:
    module = f"src.experiments.paper_figures.{fig_id}.run_task"
    task = "state_bank" if fig_id == "fig3" else "sequence_bank"
    command = [
        str(args.python),
        "-m",
        module,
        "--task",
        task,
        "--reuse-artifacts",
        "require",
        "--output-dir",
        str(output_root),
        "--artifact-root",
        str(artifact_root),
        "--shared-sequence-root",
        str(shared_root),
        "--network-seed",
        str(int(args.network_seed)),
        "--device",
        str(args.device),
        "--dataset-root",
        str(args.dataset_root),
        "--model-path-glob",
        str(args.model_path_glob),
        "--sequence-lengths",
        str(args.sequence_lengths),
        "--num-sequences",
        str(int(args.num_sequences)),
        "--sample-ms",
        str(int(args.sample_ms)),
        "--delay-ms",
        str(int(args.delay_ms)),
        "--batch-size",
        str(int(args.batch_size)),
        "--no-progress",
    ]
    if bool(args.smoke):
        command.append("--smoke")
    if args.model_path:
        command.extend(["--model-path", str(args.model_path)])
    if fig_id == "fig3":
        command.extend(
            [
                "--state-bank-singleton-batch-size",
                str(max(1, int(args.state_bank_singleton_batch_size))),
            ]
        )
    return command


def _run_command(
    label: str,
    command: Sequence[str],
    *,
    repo_root: Path,
    logs_dir: Path,
    output_root: Path,
    artifact_root: Path,
    task_artifact_dir: Path,
) -> CommandResult:
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    completed = subprocess.run(list(command), cwd=str(repo_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall_seconds = time.perf_counter() - start
    finished_at = datetime.now(timezone.utc)
    log_path = logs_dir / f"{label}.log"
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    run_info = _find_run_info(output_root)
    summary = _find_shared_summary(output_root)
    result = CommandResult(
        label=label,
        command=list(command),
        cwd=str(repo_root),
        returncode=int(completed.returncode),
        wall_seconds=float(wall_seconds),
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        log_path=str(log_path),
        output_root=str(output_root),
        artifact_root=str(artifact_root),
        task_artifact_dir=str(task_artifact_dir),
        run_info=run_info,
        summary=summary,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}; see {log_path}")
    return result


def _run_check(label: str, command: Sequence[str], *, repo_root: Path, logs_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    completed = subprocess.run(list(command), cwd=str(repo_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - start
    log_path = logs_dir / f"{label}.log"
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    return {
        "label": label,
        "status": "pass" if completed.returncode == 0 else "fail",
        "seconds": float(seconds),
        "command": list(command),
        "log_path": str(log_path),
        "notes": "" if completed.returncode == 0 else f"exit_code={completed.returncode}",
    }


def _compare_shared_specs(old_root: Path, new_root: Path) -> None:
    for filename in SPEC_FILES:
        old = pd.read_csv(old_root / "shared_sequence_specs" / filename)
        new = pd.read_csv(new_root / "shared_sequence_specs" / filename)
        pd.testing.assert_frame_equal(old, new, check_dtype=False, check_exact=True)


def _compare_root_outputs(old_root: Path, new_root: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    start = time.perf_counter()
    errors: list[str] = []
    _compare_csv_tree(old_root, new_root, errors, atol=atol, rtol=rtol)
    _compare_npz_tree(old_root, new_root, errors, atol=atol, rtol=rtol)
    return {
        "label": "root_scientific_outputs",
        "status": "pass" if not errors else "fail",
        "seconds": float(time.perf_counter() - start),
        "command": [],
        "log_path": "",
        "notes": "; ".join(errors[:20]),
    }


def _compare_csv_tree(old_root: Path, new_root: Path, errors: list[str], *, atol: float, rtol: float) -> None:
    old = _files_by_rel(old_root, ".csv")
    new = _files_by_rel(new_root, ".csv")
    for rel in sorted(set(old) | set(new)):
        if rel not in old:
            errors.append(f"csv only in optimized: {rel}")
            continue
        if rel not in new:
            errors.append(f"csv missing in optimized: {rel}")
            continue
        _compare_csv(rel, old[rel], new[rel], errors, atol=atol, rtol=rtol)


def _compare_csv(rel: str, old_path: Path, new_path: Path, errors: list[str], *, atol: float, rtol: float) -> None:
    old = pd.read_csv(old_path).drop(columns=list(METADATA_COLUMNS & set(pd.read_csv(old_path, nrows=0).columns)), errors="ignore")
    new = pd.read_csv(new_path).drop(columns=list(METADATA_COLUMNS & set(pd.read_csv(new_path, nrows=0).columns)), errors="ignore")
    if list(old.columns) != list(new.columns):
        errors.append(f"{rel}: columns differ")
        return
    if len(old) != len(new):
        errors.append(f"{rel}: rows differ old={len(old)} new={len(new)}")
        return
    if old.empty:
        return
    sort_cols = list(old.columns)
    old = old.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)
    new = new.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)
    for col in old.columns:
        a = old[col]
        b = new[col]
        if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
            if not np.allclose(pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce"), atol=atol, rtol=rtol, equal_nan=True):
                errors.append(f"{rel}: float column differs {col}")
        elif pd.api.types.is_integer_dtype(a) or pd.api.types.is_bool_dtype(a):
            if not np.array_equal(a.to_numpy(), b.to_numpy()):
                errors.append(f"{rel}: exact column differs {col}")
        else:
            if not np.array_equal(a.fillna("<NA>").astype(str).to_numpy(), b.fillna("<NA>").astype(str).to_numpy()):
                errors.append(f"{rel}: string column differs {col}")


def _compare_npz_tree(old_root: Path, new_root: Path, errors: list[str], *, atol: float, rtol: float) -> None:
    old = _files_by_rel(old_root, ".npz")
    new = _files_by_rel(new_root, ".npz")
    for rel in sorted(set(old) | set(new)):
        if rel not in old:
            errors.append(f"npz only in optimized: {rel}")
            continue
        if rel not in new:
            errors.append(f"npz missing in optimized: {rel}")
            continue
        with np.load(old[rel], allow_pickle=False) as old_npz, np.load(new[rel], allow_pickle=False) as new_npz:
            if set(old_npz.files) != set(new_npz.files):
                errors.append(f"{rel}: npz keys differ")
                continue
            for key in sorted(old_npz.files):
                a = old_npz[key]
                b = new_npz[key]
                if a.shape != b.shape:
                    errors.append(f"{rel}:{key}: shape differs")
                elif np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
                    if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
                        errors.append(f"{rel}:{key}: float array differs")
                elif not np.array_equal(a, b):
                    errors.append(f"{rel}:{key}: array differs")


def _files_by_rel(root: Path, suffix: str) -> dict[str, Path]:
    return {path.relative_to(root).as_posix(): path for path in root.rglob(f"*{suffix}") if path.is_file()}


def _summary_payload(
    args: argparse.Namespace,
    bench_root: Path,
    run_id: str,
    results: Sequence[CommandResult],
    checks: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result_by_label = {item.label: item for item in results}
    baseline = result_by_label.get("baseline_root")
    optimized = result_by_label.get("optimized_root")
    baseline_seconds = 0.0 if baseline is None else float(baseline.wall_seconds)
    optimized_seconds = 0.0 if optimized is None else float(optimized.wall_seconds)
    speedup = baseline_seconds / optimized_seconds if baseline_seconds > 0 and optimized_seconds > 0 else 0.0
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "benchmark_root": str(bench_root),
        "parameters": {
            "network_seed": int(args.network_seed),
            "device": str(args.device),
            "smoke": bool(args.smoke),
            "sequence_lengths": str(args.sequence_lengths),
            "num_sequences": int(args.num_sequences),
            "sample_ms": int(args.sample_ms),
            "delay_ms": int(args.delay_ms),
            "batch_size": int(args.batch_size),
            "state_bank_singleton_batch_size": max(1, int(args.state_bank_singleton_batch_size)),
            "atol": float(args.atol),
            "rtol": float(args.rtol),
        },
        "root_runtime": {
            "baseline_seconds": baseline_seconds,
            "optimized_seconds": optimized_seconds,
            "speedup": speedup,
            "speed_evidence": "positive" if speedup > 1.0 else "inconclusive",
        },
        "root_phase_timings": {
            "baseline": _phase_timings(baseline),
            "optimized": _phase_timings(optimized),
        },
        "runs": [item.__dict__ for item in results],
        "checks": list(checks),
        "status": "pass" if all(item["status"] == "pass" for item in checks) and all(item.returncode == 0 for item in results) else "fail",
    }


def _write_outputs(bench_root: Path, payload: dict[str, Any]) -> None:
    bench_root.mkdir(parents=True, exist_ok=True)
    (bench_root / "benchmark_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (bench_root / "runtime_compare.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "returncode", "wall_seconds", "started_at", "finished_at", "output_root", "task_artifact_dir"])
        writer.writeheader()
        for row in payload["runs"]:
            writer.writerow(
                {
                    "label": row["label"],
                    "returncode": row["returncode"],
                    "wall_seconds": f"{float(row['wall_seconds']):.6f}",
                    "started_at": row["started_at"],
                    "finished_at": row["finished_at"],
                    "output_root": row["output_root"],
                    "task_artifact_dir": row["task_artifact_dir"],
                }
            )
    lines = [
        "# Shared Sequence Root Batch Benchmark",
        "",
        f"Status: {payload['status']}",
        f"Benchmark root: `{payload['benchmark_root']}`",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Baseline root seconds | {payload['root_runtime']['baseline_seconds']:.6f} |",
        f"| Optimized root seconds | {payload['root_runtime']['optimized_seconds']:.6f} |",
        f"| Root speedup | {payload['root_runtime']['speedup']:.6f} |",
        f"| Speed evidence | {payload['root_runtime']['speed_evidence']} |",
        f"| Smoke mode | {payload['parameters']['smoke']} |",
        f"| Singleton-boundary batch cap | {payload['parameters']['state_bank_singleton_batch_size']} |",
        "",
        "| Phase | Baseline seconds | Optimized seconds |",
        "| --- | ---: | ---: |",
    ]
    phases = sorted(set(payload.get("root_phase_timings", {}).get("baseline", {})) | set(payload.get("root_phase_timings", {}).get("optimized", {})))
    for phase in phases:
        baseline = payload["root_phase_timings"]["baseline"].get(phase, "")
        optimized = payload["root_phase_timings"]["optimized"].get(phase, "")
        baseline_text = "" if baseline == "" else f"{float(baseline):.6f}"
        optimized_text = "" if optimized == "" else f"{float(optimized):.6f}"
        lines.append(f"| {phase} | {baseline_text} | {optimized_text} |")
    lines.extend(
        [
            "",
            "| Check | Status | Notes |",
            "| --- | --- | --- |",
        ]
    )
    for check in payload["checks"]:
        lines.append(f"| {check['label']} | {check['status']} | {check.get('notes', '')} |")
    (bench_root / "benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_verification_report(bench_root: Path, payload: dict[str, Any]) -> None:
    report_dir = Path("docs") / "verification" / "shared"
    report_dir.mkdir(parents=True, exist_ok=True)
    mode = "smoke" if payload["parameters"]["smoke"] else "nonsmoke"
    report_path = report_dir / f"shared_sequence_root_batch_{mode}_{payload['run_id']}.md"
    lines = [
        "# Shared Sequence Root Batch Verification",
        "",
        f"Created: {payload['created_at']}",
        f"Status: {payload['status']}",
        f"Benchmark root: `{bench_root}`",
        "",
        "## Runtime",
        "",
        f"- Baseline root seconds: `{payload['root_runtime']['baseline_seconds']:.6f}`",
        f"- Optimized root seconds: `{payload['root_runtime']['optimized_seconds']:.6f}`",
        f"- Root speedup: `{payload['root_runtime']['speedup']:.6f}`",
        f"- Speed evidence: `{payload['root_runtime']['speed_evidence']}`",
        f"- Smoke mode: `{payload['parameters']['smoke']}`",
        f"- Singleton-boundary batch cap: `{payload['parameters']['state_bank_singleton_batch_size']}`",
        "",
        "## Phase Timings",
        "",
        "| Phase | Baseline seconds | Optimized seconds |",
        "| --- | ---: | ---: |",
    ]
    phases = sorted(set(payload.get("root_phase_timings", {}).get("baseline", {})) | set(payload.get("root_phase_timings", {}).get("optimized", {})))
    for phase in phases:
        baseline = payload["root_phase_timings"]["baseline"].get(phase, "")
        optimized = payload["root_phase_timings"]["optimized"].get(phase, "")
        baseline_text = "" if baseline == "" else f"{float(baseline):.6f}"
        optimized_text = "" if optimized == "" else f"{float(optimized):.6f}"
        lines.append(f"| {phase} | {baseline_text} | {optimized_text} |")
    if not bool(payload["parameters"]["smoke"]):
        previous = 31.249006
        current = float(payload["root_runtime"]["optimized_seconds"])
        lines.extend(
            [
                "",
                "## Previous Diagnostic Context",
                "",
                f"- Previous optimized N20/B4 root seconds: `{previous:.6f}`",
                f"- Current optimized N20/B4 root seconds: `{current:.6f}`",
                f"- Current / previous ratio: `{(current / previous):.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
        ]
    )
    for check in payload["checks"]:
        lines.append(f"- `{check['label']}`: {check['status']} {check.get('notes', '')}".rstrip())
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_run_info(root: Path) -> dict[str, Any] | None:
    candidates = sorted(root.rglob("run_info.json"))
    if not candidates:
        return None
    with candidates[0].open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _find_shared_summary(root: Path) -> dict[str, Any] | None:
    path = root / "shared_sequence_root_summary.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _phase_timings(result: CommandResult | None) -> dict[str, float]:
    if result is None or not isinstance(result.summary, dict):
        return {}
    timings = result.summary.get("phase_timings")
    if not isinstance(timings, dict):
        return {}
    return {str(key): float(value) for key, value in timings.items()}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path)


def _default_python() -> str:
    torch_env = Path(r"S:\pycharm\Anaconda\envs\torch_env\python.exe")
    return str(torch_env) if torch_env.exists() else sys.executable


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs batched shared Fig.3/Fig.6 sequence-root.")
    parser.add_argument("--python", default=_default_python())
    parser.add_argument("--output-root", default="results/_bench_shared_sequence_root_batch_smoke")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--network-seed", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset-root", default="MNIST")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-path-glob", default="results/multi_snn/sdnn_ensemble_20/sdnn_ensemble_20/seed_*/net_final.pth")
    parser.add_argument("--sequence-lengths", default="3,5")
    parser.add_argument("--num-sequences", type=int, default=4)
    parser.add_argument("--sample-ms", type=int, default=5)
    parser.add_argument("--delay-ms", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--state-bank-singleton-batch-size", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--no-smoke", action="store_true", help="Run without forwarding --smoke to root or consumer tasks.")
    parsed = parser.parse_args(argv)
    parsed.smoke = not bool(parsed.no_smoke)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
