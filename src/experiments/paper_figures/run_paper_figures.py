from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS
from src.experiments.runners._common import _resolve_runtime_python


DEFAULT_MODEL_PATH_GLOB = "results/multi_snn/sdnn_ensemble_20/sdnn_ensemble_20/seed_*/net_final.pth"
DEFAULT_OUTPUT_ROOT = "results/paper_experiments"
DEFAULT_DATASET_ROOT = str(DEFAULT_PROJECT_DEFAULTS.paths.dataset_root)
NETWORK_SEED_RE = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)
SCOPES = {"main", "supplement", "both"}


@dataclass(frozen=True)
class FigureRunSpec:
    fig_id: str
    experiment_id: str
    module: str
    main_flags: tuple[str, ...]
    supplement_flags: tuple[str, ...]

    def flags_for_scope(self, scope: str) -> tuple[str, ...]:
        if scope == "both":
            return ("--run-all",)
        if scope == "main":
            return self.main_flags
        if scope == "supplement":
            return self.supplement_flags
        raise ValueError(f"Unsupported scope: {scope}")


@dataclass(frozen=True)
class NetworkCheckpoint:
    index: int
    seed: int
    model_path: Path


@dataclass(frozen=True)
class RunResult:
    fig_id: str
    experiment_id: str
    network_index: int
    network_seed: int
    model_path: Path
    run_dir: Path
    status: str
    returncode: int | None
    elapsed_seconds: float
    command: str
    resumed: bool = False
    dry_run: bool = False
    error: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildResult:
    fig_id: str
    experiment_id: str
    experiment_root: Path
    status: str
    returncode: int | None
    elapsed_seconds: float
    command: str
    skipped_reason: str = ""
    error: str = ""
    dry_run: bool = False


FIGURE_SPECS: tuple[FigureRunSpec, ...] = (
    FigureRunSpec(
        fig_id="fig1",
        experiment_id="fig1_functional_stsp_substrate",
        module="src.experiments.paper_figures.fig1_functional_stsp_substrate_experiment",
        main_flags=("--run-baseline", "--run-delay-decode", "--run-dms-shuffle", "--run-firing-rate-control"),
        supplement_flags=("--run-baseline", "--run-delay-decode", "--run-dms-shuffle", "--run-firing-rate-control"),
    ),
    FigureRunSpec(
        fig_id="fig2",
        experiment_id="fig2_pair_fused_stsp_state",
        module="src.experiments.paper_figures.fig2_pair_fused_stsp_state_experiment",
        main_flags=("--run-state-bank", "--run-morphology", "--run-linear-mixture", "--run-neutral-ping", "--run-partial-cue"),
        supplement_flags=("--run-state-bank", "--run-morphology", "--run-linear-mixture", "--run-neutral-ping", "--run-partial-cue", "--run-supplement"),
    ),
    FigureRunSpec(
        fig_id="fig3",
        experiment_id="fig3_multiitem_peak_landscape",
        module="src.experiments.paper_figures.fig3_multiitem_peak_landscape_experiment",
        main_flags=("--run-state-bank", "--run-progressive-update", "--run-peak-valley-landscape", "--run-neutral-ping", "--run-structural-weak-cue"),
        supplement_flags=(
            "--run-state-bank",
            "--run-progressive-update",
            "--run-peak-valley-landscape",
            "--run-neutral-ping",
            "--run-structural-weak-cue",
            "--run-population-morphology-supplement",
            "--run-supplement",
        ),
    ),
    FigureRunSpec(
        fig_id="fig4",
        experiment_id="fig4_overlap_reentry",
        module="src.experiments.paper_figures.fig4_overlap_reentry_experiment",
        main_flags=(
            "--run-pair-sampling",
            "--run-rollouts",
            "--run-similarity-entry",
            "--run-overlap-localization",
            "--run-overlap-accuracy-identification",
            "--run-decision-spike-displacement",
            "--run-decision-deflection",
        ),
        supplement_flags=(
            "--run-pair-sampling",
            "--run-rollouts",
            "--run-similarity-entry",
            "--run-overlap-localization",
            "--run-overlap-accuracy-identification",
            "--run-decision-spike-displacement",
            "--run-decision-deflection",
            "--run-overlap-perturbation",
            "--run-supplement",
        ),
    ),
    FigureRunSpec(
        fig_id="fig5",
        experiment_id="fig5_local_support_competition",
        module="src.experiments.paper_figures.fig5_local_support_competition_experiment",
        main_flags=("--run-trial-sampling", "--run-preprobe-support", "--run-early-firing", "--run-local-events", "--run-support-perturbation"),
        supplement_flags=("--run-trial-sampling", "--run-preprobe-support", "--run-early-firing", "--run-local-events", "--run-support-perturbation", "--run-supplement"),
    ),
    FigureRunSpec(
        fig_id="fig6",
        experiment_id="fig6_peak_amplified_reentry",
        module="src.experiments.paper_figures.fig6_peak_amplified_reentry_experiment",
        main_flags=(
            "--run-sequence-bank",
            "--run-peak-source-attribution",
            "--run-peak-update-history",
            "--run-peak-input-overlap-origin",
            "--run-real-reentry-rollout",
            "--run-real-downstream-metrics",
        ),
        supplement_flags=(
            "--run-sequence-bank",
            "--run-peak-source-attribution",
            "--run-peak-update-history",
            "--run-peak-input-overlap-origin",
            "--run-real-reentry-rollout",
            "--run-real-downstream-metrics",
            "--run-peak-enrichment",
            "--run-update-recency-model",
            "--run-peak-weighted-overlap",
            "--run-reentry-prediction",
            "--run-downstream-prediction",
            "--run-supplement",
        ),
    ),
)
FIGURE_SPECS_BY_ID = {spec.fig_id: spec for spec in FIGURE_SPECS}
FIGURE_SPECS_BY_NUMBER = {spec.fig_id.replace("fig", ""): spec for spec in FIGURE_SPECS}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


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
    pattern = str(_resolve_repo_path(model_path_glob)) if not Path(model_path_glob).is_absolute() else str(model_path_glob)
    paths = sorted(Path(path).resolve() for path in glob.glob(pattern))
    checkpoints = [
        NetworkCheckpoint(index=index, seed=_extract_network_seed(path, fallback=index), model_path=path)
        for index, path in enumerate(paths)
        if path.is_file()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files matched --model-path-glob: {model_path_glob}")
    checkpoints = sorted(checkpoints, key=lambda item: (item.seed, str(item.model_path)))
    return [NetworkCheckpoint(index=index, seed=item.seed, model_path=item.model_path) for index, item in enumerate(checkpoints)]


def parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    seen: set[int] = set()
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise ValueError(f"Invalid descending seed range: {token}")
            values = range(start, end + 1)
        else:
            values = (int(token),)
        for seed in values:
            if seed in seen:
                continue
            seeds.append(seed)
            seen.add(seed)
    if not seeds:
        raise ValueError("--seeds must contain at least one seed unless --all-seeds is used.")
    return seeds


def select_checkpoints(checkpoints: Sequence[NetworkCheckpoint], *, seeds: str, all_seeds: bool) -> list[NetworkCheckpoint]:
    if all_seeds:
        return list(checkpoints)
    requested = parse_seed_list(seeds)
    by_seed = {int(item.seed): item for item in checkpoints}
    missing = [seed for seed in requested if seed not in by_seed]
    if missing:
        raise ValueError(f"Requested seed(s) not found in checkpoint glob: {missing}")
    return [by_seed[seed] for seed in requested]


def parse_figs(raw: str) -> list[FigureRunSpec]:
    if str(raw).strip().lower() == "all":
        return list(FIGURE_SPECS)
    specs: list[FigureRunSpec] = []
    seen: set[str] = set()
    for item in str(raw).split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in FIGURE_SPECS_BY_NUMBER:
            spec = FIGURE_SPECS_BY_NUMBER[token]
        elif token in FIGURE_SPECS_BY_ID:
            spec = FIGURE_SPECS_BY_ID[token]
        else:
            raise ValueError(f"Unknown figure id: {item}")
        if spec.fig_id in seen:
            continue
        specs.append(spec)
        seen.add(spec.fig_id)
    if not specs:
        raise ValueError("--figs must be 'all' or a comma-separated list like fig1,fig3 or 1,3.")
    return specs


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _run_info_status(run_dir: Path) -> str | None:
    payload = _load_json(run_dir / "meta" / "run_info.json")
    if payload is None:
        return None
    status = payload.get("status")
    return str(status) if status is not None else None


def validate_run_dir(run_dir: Path, *, scope: str) -> tuple[bool, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not (run_dir / "summary.json").is_file():
        errors.append("summary.json missing")
        return False, errors, warnings
    if _run_info_status(run_dir) != "success":
        errors.append("meta/run_info.json missing or status is not success")
    summary = _load_json(run_dir / "summary.json")
    if summary is None:
        errors.append("summary.json unreadable")
        return False, errors, warnings

    if scope in {"main", "both"}:
        missing_main = summary.get("missing_for_main_figure") or []
        if missing_main:
            errors.append(f"missing_for_main_figure is not empty: {missing_main}")
        if bool(summary.get("proxy_mode")):
            errors.append("proxy_mode is true for a main-claim scope")
        if summary.get("final_scientific_use") is False:
            errors.append("final_scientific_use is false for a main-claim scope")
    if scope in {"supplement", "both"}:
        missing_supp = summary.get("missing_for_supplementary") or []
        if missing_supp:
            errors.append(f"missing_for_supplementary is not empty: {missing_supp}")

    raw_warnings = summary.get("warnings") or []
    if isinstance(raw_warnings, list):
        warnings.extend(str(item) for item in raw_warnings)
    return not errors, errors, warnings


def build_experiment_command(
    *,
    runtime_python: Path,
    spec: FigureRunSpec,
    checkpoint: NetworkCheckpoint,
    fig_root: Path,
    dataset_root: Path,
    device: str,
    split: str,
    scope: str,
    smoke: bool,
    save_debug_figures: bool,
    no_progress: bool,
) -> list[str]:
    command = [
        str(runtime_python),
        "-m",
        spec.module,
        "--model-path",
        str(checkpoint.model_path),
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(fig_root),
        "--network-seed",
        str(int(checkpoint.seed)),
        "--device",
        str(device),
        "--split",
        str(split),
        *spec.flags_for_scope(scope),
    ]
    if smoke:
        command.append("--smoke")
    if save_debug_figures:
        command.append("--save-debug-figures")
    if no_progress:
        command.append("--no-progress")
    return command


def _copy_stream(pipe: Any, console: Any, log_handle: Any) -> None:
    while True:
        chunk = pipe.read(1)
        if not chunk:
            break
        log_handle.write(chunk)
        log_handle.flush()
        console.write(chunk)
        console.flush()


def _run_streaming_command(command: Sequence[str], *, env: Mapping[str, str], stdout_path: Path, stderr_path: Path) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with stdout_path.open("w", encoding="utf-8", newline="") as stdout_log, stderr_path.open("w", encoding="utf-8", newline="") as stderr_log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=0,
            env=dict(env),
        )
        threads: list[threading.Thread] = []
        if process.stdout is not None:
            threads.append(threading.Thread(target=_copy_stream, args=(process.stdout, sys.stdout, stdout_log)))
        if process.stderr is not None:
            threads.append(threading.Thread(target=_copy_stream, args=(process.stderr, sys.stderr, stderr_log)))
        for thread in threads:
            thread.start()
        returncode = process.wait()
        for thread in threads:
            thread.join()
    return int(returncode), time.time() - started


def run_one_experiment(
    *,
    runtime_python: Path,
    spec: FigureRunSpec,
    checkpoint: NetworkCheckpoint,
    output_root: Path,
    dataset_root: Path,
    device: str,
    split: str,
    scope: str,
    smoke: bool,
    resume: bool,
    save_debug_figures: bool,
    no_progress: bool,
    dry_run: bool,
) -> RunResult:
    fig_root = output_root / spec.experiment_id
    run_dir = fig_root / f"seed_{int(checkpoint.seed):03d}"
    command = build_experiment_command(
        runtime_python=runtime_python,
        spec=spec,
        checkpoint=checkpoint,
        fig_root=fig_root,
        dataset_root=dataset_root,
        device=device,
        split=split,
        scope=scope,
        smoke=smoke,
        save_debug_figures=save_debug_figures,
        no_progress=no_progress,
    )
    command_text = subprocess.list2cmdline(command)
    if dry_run:
        return RunResult(
            fig_id=spec.fig_id,
            experiment_id=spec.experiment_id,
            network_index=checkpoint.index,
            network_seed=checkpoint.seed,
            model_path=checkpoint.model_path,
            run_dir=run_dir,
            status="dry_run",
            returncode=None,
            elapsed_seconds=0.0,
            command=command_text,
            dry_run=True,
        )

    if resume:
        ok, errors, warnings = validate_run_dir(run_dir, scope=scope)
        if ok:
            return RunResult(
                fig_id=spec.fig_id,
                experiment_id=spec.experiment_id,
                network_index=checkpoint.index,
                network_seed=checkpoint.seed,
                model_path=checkpoint.model_path,
                run_dir=run_dir,
                status="success",
                returncode=0,
                elapsed_seconds=0.0,
                command=command_text,
                resumed=True,
                warnings=tuple(warnings),
            )
        if run_dir.exists() and errors:
            print(f"[Resume] {spec.fig_id} seed={checkpoint.seed} exists but does not satisfy scope={scope}; rerunning ({'; '.join(errors)})")

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    returncode, elapsed = _run_streaming_command(
        command,
        env=env,
        stdout_path=run_dir / "meta" / "controller_stdout.log",
        stderr_path=run_dir / "meta" / "controller_stderr.log",
    )

    ok, errors, warnings = validate_run_dir(run_dir, scope=scope)
    status = "success" if returncode == 0 and ok else "failed"
    error = ""
    if returncode != 0:
        error = f"returncode={returncode}"
    if errors:
        error = "; ".join([item for item in (error, *errors) if item])
    return RunResult(
        fig_id=spec.fig_id,
        experiment_id=spec.experiment_id,
        network_index=checkpoint.index,
        network_seed=checkpoint.seed,
        model_path=checkpoint.model_path,
        run_dir=run_dir,
        status=status,
        returncode=int(returncode),
        elapsed_seconds=elapsed,
        command=command_text,
        error=error,
        warnings=tuple(warnings),
    )


def build_paper_figure_command(
    *,
    runtime_python: Path,
    spec: FigureRunSpec,
    fig_root: Path,
    check_only: bool,
) -> list[str]:
    command = [
        str(runtime_python),
        "-m",
        "src.plotting.paper_fig.build",
        "--fig",
        spec.fig_id,
        "--experiment-root",
        str(fig_root),
    ]
    if check_only:
        command.append("--check-only")
    return command


def run_build(
    *,
    runtime_python: Path,
    spec: FigureRunSpec,
    output_root: Path,
    scope: str,
    no_build_paper_figures: bool,
    check_only_build: bool,
    dry_run: bool,
    fig_runs: Sequence[RunResult],
) -> BuildResult:
    fig_root = output_root / spec.experiment_id
    command = build_paper_figure_command(runtime_python=runtime_python, spec=spec, fig_root=fig_root, check_only=check_only_build)
    command_text = subprocess.list2cmdline(command)
    if scope == "supplement":
        return BuildResult(spec.fig_id, spec.experiment_id, fig_root, "skipped", None, 0.0, command_text, skipped_reason="scope=supplement")
    if no_build_paper_figures:
        return BuildResult(spec.fig_id, spec.experiment_id, fig_root, "skipped", None, 0.0, command_text, skipped_reason="--no-build-paper-figures")
    if dry_run:
        return BuildResult(spec.fig_id, spec.experiment_id, fig_root, "dry_run", None, 0.0, command_text, dry_run=True)
    failed_runs = [result for result in fig_runs if result.status != "success"]
    if failed_runs:
        return BuildResult(spec.fig_id, spec.experiment_id, fig_root, "skipped", None, 0.0, command_text, skipped_reason="one or more selected runs failed")

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    log_dir = output_root / "_build_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    returncode, elapsed = _run_streaming_command(
        command,
        env=env,
        stdout_path=log_dir / f"{spec.fig_id}_stdout.log",
        stderr_path=log_dir / f"{spec.fig_id}_stderr.log",
    )
    status = "success" if returncode == 0 else "failed"
    error = "" if status == "success" else f"returncode={returncode}; see {log_dir / (spec.fig_id + '_stderr.log')}"
    return BuildResult(spec.fig_id, spec.experiment_id, fig_root, status, int(returncode), elapsed, command_text, error=error)


def _run_record(result: RunResult, batch_root: Path) -> dict[str, Any]:
    try:
        rel_run_dir = result.run_dir.resolve().relative_to(batch_root.resolve()).as_posix()
    except ValueError:
        rel_run_dir = str(result.run_dir)
    return {
        "fig_id": result.fig_id,
        "experiment_id": result.experiment_id,
        "network_index": int(result.network_index),
        "network_seed": int(result.network_seed),
        "model_path": str(result.model_path),
        "run_dir": rel_run_dir,
        "status": result.status,
        "returncode": result.returncode,
        "elapsed_seconds": round(float(result.elapsed_seconds), 6),
        "resumed": bool(result.resumed),
        "dry_run": bool(result.dry_run),
        "warnings": " | ".join(result.warnings),
        "error": result.error,
        "command": result.command,
    }


def _build_record(result: BuildResult, batch_root: Path) -> dict[str, Any]:
    try:
        rel_root = result.experiment_root.resolve().relative_to(batch_root.resolve()).as_posix()
    except ValueError:
        rel_root = str(result.experiment_root)
    return {
        "fig_id": result.fig_id,
        "experiment_id": result.experiment_id,
        "experiment_root": rel_root,
        "status": result.status,
        "returncode": result.returncode,
        "elapsed_seconds": round(float(result.elapsed_seconds), 6),
        "skipped_reason": result.skipped_reason,
        "dry_run": bool(result.dry_run),
        "error": result.error,
        "command": result.command,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_reports(
    *,
    report_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
    runtime_python: Path,
    specs: Sequence[FigureRunSpec],
    checkpoints: Sequence[NetworkCheckpoint],
    run_results: Sequence[RunResult],
    build_results: Sequence[BuildResult],
) -> dict[str, Any]:
    run_records = [_run_record(result, output_root) for result in sorted(run_results, key=lambda item: (item.fig_id, item.network_seed))]
    build_records = [_build_record(result, output_root) for result in sorted(build_results, key=lambda item: item.fig_id)]
    failed_run_records = [row for row in run_records if row["status"] != "success"]

    run_fields = [
        "fig_id",
        "experiment_id",
        "network_index",
        "network_seed",
        "model_path",
        "run_dir",
        "status",
        "returncode",
        "elapsed_seconds",
        "resumed",
        "dry_run",
        "warnings",
        "error",
        "command",
    ]
    build_fields = ["fig_id", "experiment_id", "experiment_root", "status", "returncode", "elapsed_seconds", "skipped_reason", "dry_run", "error", "command"]
    _write_csv(report_dir / "run_manifest.csv", run_records, run_fields)
    _write_csv(report_dir / "failed_runs.csv", failed_run_records, run_fields)
    _write_csv(report_dir / "build_manifest.csv", build_records, build_fields)
    _write_json(report_dir / "run_manifest.json", {"runs": run_records, "builds": build_records})

    summary = {
        "status": "success" if not failed_run_records and all(row["status"] in {"success", "skipped"} for row in build_records) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_python": str(runtime_python),
        "figs": [spec.fig_id for spec in specs],
        "scope": str(args.scope),
        "network_seeds": [int(checkpoint.seed) for checkpoint in checkpoints],
        "n_runs": int(len(run_records)),
        "n_run_success": int(sum(1 for row in run_records if row["status"] == "success")),
        "n_run_failed": int(len(failed_run_records)),
        "n_run_resumed": int(sum(1 for row in run_records if row["resumed"])),
        "n_builds": int(len(build_records)),
        "n_build_success": int(sum(1 for row in build_records if row["status"] == "success")),
        "n_build_failed": int(sum(1 for row in build_records if row["status"] == "failed")),
        "n_build_skipped": int(sum(1 for row in build_records if row["status"] == "skipped")),
        "output_root": str(output_root),
        "dataset_root": str(_resolve_repo_path(args.dataset_root)),
        "model_path_glob": str(args.model_path_glob),
        "device": str(args.device),
        "smoke": bool(args.smoke),
        "jobs": int(args.jobs),
        "continue_on_error": bool(args.continue_on_error),
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
    }
    _write_json(report_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone paper-figure experiments across selected figures and SDNN checkpoints.")
    parser.add_argument("--figs", type=str, default="all", help="'all' or comma-separated figure ids like fig1,fig3 or 1,3.")
    parser.add_argument("--scope", choices=sorted(SCOPES), default="both")
    parser.add_argument("--seeds", type=str, default="1000", help="Comma-separated seeds and ranges, e.g. 1000,1001,1005-1007.")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--model-path-glob", type=str, default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_PROJECT_DEFAULTS.runtime.device)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--force", dest="resume", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-build-paper-figures", action="store_true")
    parser.add_argument("--check-only-build", action="store_true")
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.jobs) < 1:
        raise SystemExit("--jobs must be >= 1")

    specs = parse_figs(args.figs)
    checkpoints = select_checkpoints(discover_checkpoints(args.model_path_glob), seeds=str(args.seeds), all_seeds=bool(args.all_seeds))
    output_root = _resolve_repo_path(args.output_root)
    dataset_root = _resolve_repo_path(args.dataset_root)
    runtime_python = _resolve_runtime_python()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_root / "_batch_runs" / timestamp

    print(f"[Setup] figs={','.join(spec.fig_id for spec in specs)} scope={args.scope} seeds={','.join(str(c.seed) for c in checkpoints)}")
    print(f"[Setup] runtime_python={runtime_python}")
    print(f"[Setup] output_root={output_root}")

    tasks = [(spec, checkpoint) for spec in specs for checkpoint in checkpoints]
    run_results: list[RunResult] = []
    failures: list[str] = []

    def submit_one(spec: FigureRunSpec, checkpoint: NetworkCheckpoint) -> RunResult:
        return run_one_experiment(
            runtime_python=runtime_python,
            spec=spec,
            checkpoint=checkpoint,
            output_root=output_root,
            dataset_root=dataset_root,
            device=str(args.device),
            split=str(args.split),
            scope=str(args.scope),
            smoke=bool(args.smoke),
            resume=bool(args.resume),
            save_debug_figures=bool(args.save_debug_figures),
            no_progress=bool(args.no_progress),
            dry_run=bool(args.dry_run),
        )

    if int(args.jobs) <= 1:
        for spec, checkpoint in tasks:
            print(f"[Run] {spec.fig_id} seed={checkpoint.seed}")
            result = submit_one(spec, checkpoint)
            run_results.append(result)
            print(f"[Run] {spec.fig_id} seed={checkpoint.seed} -> {result.status}")
            if result.status not in {"success", "dry_run"}:
                failures.append(f"{spec.fig_id} seed={checkpoint.seed}: {result.error}")
                if not args.continue_on_error:
                    break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.jobs)) as executor:
            future_map = {executor.submit(submit_one, spec, checkpoint): (spec, checkpoint) for spec, checkpoint in tasks}
            for future in concurrent.futures.as_completed(future_map):
                spec, checkpoint = future_map[future]
                result = future.result()
                run_results.append(result)
                print(f"[Run] {spec.fig_id} seed={checkpoint.seed} -> {result.status}")
                if result.status not in {"success", "dry_run"}:
                    failures.append(f"{spec.fig_id} seed={checkpoint.seed}: {result.error}")
        run_results.sort(key=lambda item: (item.fig_id, item.network_seed))

    build_results: list[BuildResult] = []
    if failures and not args.continue_on_error:
        for spec in specs:
            build_results.append(
                BuildResult(spec.fig_id, spec.experiment_id, output_root / spec.experiment_id, "skipped", None, 0.0, "", skipped_reason="run phase failed")
            )
    else:
        for spec in specs:
            fig_runs = [result for result in run_results if result.fig_id == spec.fig_id]
            result = run_build(
                runtime_python=runtime_python,
                spec=spec,
                output_root=output_root,
                scope=str(args.scope),
                no_build_paper_figures=bool(args.no_build_paper_figures),
                check_only_build=bool(args.check_only_build),
                dry_run=bool(args.dry_run),
                fig_runs=fig_runs,
            )
            build_results.append(result)
            print(f"[Build] {spec.fig_id} -> {result.status}{' (' + result.skipped_reason + ')' if result.skipped_reason else ''}")
            if result.status == "failed":
                failures.append(f"{spec.fig_id} build: {result.error}")
                if not args.continue_on_error:
                    break

    if args.dry_run:
        for result in run_results:
            print(f"[DryRun] {result.command}")
        for result in build_results:
            if result.status == "dry_run":
                print(f"[DryRun] {result.command}")
        return 0 if not failures else 1

    summary = _write_reports(
        report_dir=report_dir,
        output_root=output_root,
        args=args,
        runtime_python=runtime_python,
        specs=specs,
        checkpoints=checkpoints,
        run_results=run_results,
        build_results=build_results,
    )
    print(f"[Report] {report_dir}")
    if failures or summary["status"] != "success":
        print("[Done] completed with failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("[Done] paper-figure batch complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
