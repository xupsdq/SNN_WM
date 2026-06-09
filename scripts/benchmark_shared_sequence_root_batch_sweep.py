from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


ROOT_TASK = "shared_sequence_root_bank"


@dataclass
class SweepRun:
    label: str
    batch_size: int
    effective_max_batch: int
    command: list[str]
    returncode: int
    wall_seconds: float
    output_root: str
    shared_root: str
    log_path: str
    summary: dict[str, Any] | None
    compare_status: str
    compare_notes: str


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    bench_helpers = _load_benchmark_helpers()
    repo_root = Path.cwd()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_root = _resolve_path(args.output_root) / run_id
    logs_dir = sweep_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    baseline_root = sweep_root / "baseline_serial"
    baseline_artifact_root = baseline_root / "data" / "intermediates"
    baseline_shared_root = baseline_artifact_root / ROOT_TASK
    baseline = _run_root(
        args,
        label="baseline_serial",
        output_root=baseline_root,
        artifact_root=baseline_artifact_root,
        batch_size=int(args.baseline_batch_size),
        enable_batch=False,
        repo_root=repo_root,
        logs_dir=logs_dir,
    )
    if baseline.returncode != 0:
        payload = _payload(args, run_id, sweep_root, [baseline], status="fail")
        _write_outputs(sweep_root, payload)
        _write_report(payload)
        return 1

    runs: list[SweepRun] = [baseline]
    for batch_size in _batch_sizes(args.batch_sizes):
        label = f"optimized_b{batch_size}"
        output_root = sweep_root / label
        artifact_root = output_root / "data" / "intermediates"
        run = _run_root(
            args,
            label=label,
            output_root=output_root,
            artifact_root=artifact_root,
            batch_size=batch_size,
            enable_batch=True,
            repo_root=repo_root,
            logs_dir=logs_dir,
        )
        if run.returncode == 0:
            try:
                bench_helpers._compare_shared_specs(baseline_shared_root, Path(run.shared_root))
                compare = bench_helpers._compare_root_outputs(
                    baseline_shared_root,
                    Path(run.shared_root),
                    atol=float(args.atol),
                    rtol=float(args.rtol),
                )
                run.compare_status = str(compare["status"])
                run.compare_notes = str(compare.get("notes", ""))
            except Exception as exc:
                run.compare_status = "fail"
                run.compare_notes = str(exc)
        runs.append(run)
        if run.returncode != 0 and bool(args.stop_on_failure):
            break

    optimized_runs = [run for run in runs if run.label != "baseline_serial"]
    passing_optimized = [run for run in optimized_runs if run.returncode == 0 and run.compare_status == "pass"]
    if baseline.returncode != 0 or not passing_optimized:
        status = "fail"
    elif all(run.returncode == 0 and run.compare_status == "pass" for run in optimized_runs):
        status = "pass"
    else:
        status = "partial"
    payload = _payload(args, run_id, sweep_root, runs, status=status)
    _write_outputs(sweep_root, payload)
    _write_report(payload)
    return 0 if status in {"pass", "partial"} else 1


def _run_root(
    args: argparse.Namespace,
    *,
    label: str,
    output_root: Path,
    artifact_root: Path,
    batch_size: int,
    enable_batch: bool,
    repo_root: Path,
    logs_dir: Path,
) -> SweepRun:
    shared_root = artifact_root / ROOT_TASK
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
        str(int(batch_size)),
        "--state-bank-singleton-batch-size",
        str(max(1, int(args.state_bank_singleton_batch_size))),
        "--no-progress",
    ]
    if args.model_path:
        command.extend(["--model-path", str(args.model_path)])
    if bool(args.smoke):
        command.append("--smoke")
    if enable_batch:
        command.append("--enable-state-bank-batch")

    start = time.perf_counter()
    completed = subprocess.run(command, cwd=str(repo_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall_seconds = time.perf_counter() - start
    log_path = logs_dir / f"{label}.log"
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    return SweepRun(
        label=label,
        batch_size=int(batch_size),
        effective_max_batch=0,
        command=command,
        returncode=int(completed.returncode),
        wall_seconds=float(wall_seconds),
        output_root=str(output_root),
        shared_root=str(shared_root),
        log_path=str(log_path),
        summary=_read_json(output_root / "shared_sequence_root_summary.json"),
        compare_status="baseline" if not enable_batch else ("not_run" if completed.returncode == 0 else "failed_run"),
        compare_notes="" if completed.returncode == 0 else f"exit_code={completed.returncode}",
    )


def _payload(args: argparse.Namespace, run_id: str, sweep_root: Path, runs: Sequence[SweepRun], *, status: str) -> dict[str, Any]:
    baseline = runs[0] if runs else None
    sequence_counts = _sequence_counts(Path(baseline.shared_root) if baseline else None)
    max_group_size = max(sequence_counts.values()) if sequence_counts else 0
    serialized: list[dict[str, Any]] = []
    for run in runs:
        row = run.__dict__.copy()
        row["effective_max_batch"] = min(int(run.batch_size), int(max_group_size)) if run.label != "baseline_serial" else 1
        if baseline and run.label != "baseline_serial" and run.wall_seconds > 0:
            row["speedup_vs_baseline"] = float(baseline.wall_seconds) / float(run.wall_seconds)
        else:
            row["speedup_vs_baseline"] = 1.0
        serialized.append(row)
    passing = [row for row in serialized if row["label"] != "baseline_serial" and row["returncode"] == 0 and row["compare_status"] == "pass"]
    fastest = min(passing, key=lambda row: float(row["wall_seconds"])) if passing else None
    max_passed = max(passing, key=lambda row: int(row["effective_max_batch"])) if passing else None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "status": str(status),
        "sweep_root": str(sweep_root),
        "parameters": {
            "network_seed": int(args.network_seed),
            "device": str(args.device),
            "smoke": bool(args.smoke),
            "sequence_lengths": str(args.sequence_lengths),
            "num_sequences": int(args.num_sequences),
            "sample_ms": int(args.sample_ms),
            "delay_ms": int(args.delay_ms),
            "baseline_batch_size": int(args.baseline_batch_size),
            "batch_sizes": [int(value) for value in _batch_sizes(args.batch_sizes)],
            "state_bank_singleton_batch_size": max(1, int(args.state_bank_singleton_batch_size)),
            "atol": float(args.atol),
            "rtol": float(args.rtol),
        },
        "sequence_counts_by_len": sequence_counts,
        "fastest_passed": fastest,
        "max_effective_batch_passed": max_passed,
        "runs": serialized,
    }


def _write_outputs(sweep_root: Path, payload: dict[str, Any]) -> None:
    sweep_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / "sweep_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (sweep_root / "sweep_runtime.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "label",
            "batch_size",
            "effective_max_batch",
            "returncode",
            "wall_seconds",
            "speedup_vs_baseline",
            "compare_status",
            "compare_notes",
            "shared_root",
            "log_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["runs"]:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = _summary_lines(payload)
    (sweep_root / "sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(payload: dict[str, Any]) -> None:
    report_dir = Path("docs") / "verification" / "shared"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"shared_sequence_root_batch_sweep_{payload['run_id']}.md"
    report_path.write_text("\n".join(_summary_lines(payload)) + "\n", encoding="utf-8")


def _summary_lines(payload: dict[str, Any]) -> list[str]:
    fastest = payload.get("fastest_passed") or {}
    max_passed = payload.get("max_effective_batch_passed") or {}
    lines = [
        "# Shared Sequence Root Batch Sweep",
        "",
        f"Created: {payload['created_at']}",
        f"Status: {payload['status']}",
        f"Sweep root: `{payload['sweep_root']}`",
        "",
        "## Parameters",
        "",
        f"- Smoke mode: `{payload['parameters']['smoke']}`",
        f"- Sequence lengths: `{payload['parameters']['sequence_lengths']}`",
        f"- Num sequences: `{payload['parameters']['num_sequences']}`",
        f"- Sample/delay ms: `{payload['parameters']['sample_ms']}/{payload['parameters']['delay_ms']}`",
        f"- Batch sizes: `{payload['parameters']['batch_sizes']}`",
        f"- Singleton-boundary batch cap: `{payload['parameters']['state_bank_singleton_batch_size']}`",
        f"- Sequence counts by K: `{payload['sequence_counts_by_len']}`",
        "",
        "## Summary",
        "",
        f"- Fastest passing batch: `{fastest.get('batch_size', '')}`",
        f"- Fastest optimized seconds: `{_fmt(fastest.get('wall_seconds'))}`",
        f"- Fastest speedup vs serial baseline: `{_fmt(fastest.get('speedup_vs_baseline'))}`",
        f"- Max effective batch passed: `{max_passed.get('effective_max_batch', '')}`",
        "",
        "## Runs",
        "",
        "| Label | Batch | Effective batch | Seconds | Speedup | Compare | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["runs"]:
        lines.append(
            f"| {row['label']} | {row['batch_size']} | {row['effective_max_batch']} | "
            f"{float(row['wall_seconds']):.6f} | {float(row['speedup_vs_baseline']):.6f} | "
            f"{row['compare_status']} | {row.get('compare_notes', '')} |"
        )
    lines.extend(["", "## Phase Timings", "", "| Label | Phase timings |", "| --- | --- |"])
    for row in payload["runs"]:
        timings = {}
        summary = row.get("summary")
        if isinstance(summary, dict) and isinstance(summary.get("phase_timings"), dict):
            timings = {key: round(float(value), 6) for key, value in summary["phase_timings"].items()}
        lines.append(f"| {row['label']} | `{json.dumps(timings, ensure_ascii=False)}` |")
    return lines


def _sequence_counts(shared_root: Path | None) -> dict[str, int]:
    if shared_root is None:
        return {}
    path = shared_root / "shared_sequence_specs" / "sequence_trials.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(key): int(value) for key, value in df.groupby("seq_len")["sequence_id"].nunique().to_dict().items()}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _load_benchmark_helpers():
    path = Path("scripts") / "benchmark_shared_sequence_root_batch_smoke.py"
    spec = importlib.util.spec_from_file_location("shared_root_batch_benchmark_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import benchmark helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def _batch_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not sizes:
        raise ValueError("--batch-sizes must contain at least one integer")
    return sizes


def _fmt(value: Any) -> str:
    if value in {None, ""}:
        return ""
    return f"{float(value):.6f}"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path)


def _default_python() -> str:
    torch_env = Path(r"S:\pycharm\Anaconda\envs\torch_env\python.exe")
    return str(torch_env) if torch_env.exists() else sys.executable


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Fig.3/Fig.6 shared-root state-bank batch sizes.")
    parser.add_argument("--python", default=_default_python())
    parser.add_argument("--output-root", default="results/_bench_shared_sequence_root_batch_sweep")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--network-seed", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset-root", default="MNIST")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-path-glob", default="results/multi_snn/sdnn_ensemble_20/sdnn_ensemble_20/seed_*/net_final.pth")
    parser.add_argument("--sequence-lengths", default="3,5,7,10")
    parser.add_argument("--num-sequences", type=int, default=128)
    parser.add_argument("--sample-ms", type=int, default=5)
    parser.add_argument("--delay-ms", type=int, default=5)
    parser.add_argument("--baseline-batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", default="2,4,8,16,32")
    parser.add_argument("--state-bank-singleton-batch-size", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
