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
from src.experiments.paper_figures.common.progress import PROGRESS_EVENT_PREFIX
from src.experiments.paper_figures.common.registry import FIGURE_PACKAGE_IDS, load_figure_registry
from src.experiments.paper_figures.common.resources import PARALLEL_AXES, ResourcePlan, resolve_resource_plan
from src.experiments.runners._common import _resolve_runtime_python


DEFAULT_MODEL_PATH_GLOB = "results/multi_snn/sdnn_ensemble_20/sdnn_ensemble_20/seed_*/net_final.pth"
DEFAULT_OUTPUT_ROOT = "results/paper_experiments"
DEFAULT_DATASET_ROOT = str(DEFAULT_PROJECT_DEFAULTS.paths.dataset_root)
NETWORK_SEED_RE = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)
SCOPES = {"main", "supplement", "both"}
PROGRESS_MODES = {"auto", "detailed", "compact", "off"}
_CONSOLE_LOCK = threading.Lock()
GPU_BATCH_FLAGS_BY_FIG = {
    "fig1": ("--enable-condition-batch",),
    "fig2": ("--enable-partial-cue-batch",),
    "fig3": ("--enable-condition-batch",),
    "fig4": ("--enable-condition-batch",),
    "fig5": ("--enable-branch-batch",),
    # Fig.6 probe condition batch is output-equivalent, but medium validation
    # showed negative wall-clock speedup; keep that specific flag opt-in via
    # the Fig.6 CLI. Sequence-bank, high-STSP ablation, and leave-one-out
    # support batches have positive medium evidence and are safe to forward.
    "fig6": (
        "--enable-sequence-bank-batch",
        "--enable-high-stsp-ablation-batch",
        "--enable-leave-one-out-batch",
    ),
}
GPU_METRIC_FLAGS_BY_FIG = {
    "fig1": ("--enable-gpu-metrics",),
}
BENCHMARK_PROFILE_NONE = "none"
BENCHMARK_PROFILE_MEDIUM = "medium"
BENCHMARK_PROFILE_ARGS_BY_FIG: dict[str, dict[str, tuple[str, ...]]] = {
    BENCHMARK_PROFILE_NONE: {},
    BENCHMARK_PROFILE_MEDIUM: {
        "fig1": (
            "--baseline-eval-per-class",
            "30",
            "--delay-decode-train-per-class",
            "20",
            "--delay-decode-test-per-class",
            "20",
            "--dms-num-trials",
            "30",
            "--shuffle-num-boot",
            "200",
        ),
        "fig2": (
            "--num-pairs",
            "50",
            "--weak-probe-repeats",
            "4",
            "--n-shuffle",
            "20",
            "--completion-delay-repeats",
            "2",
        ),
        "fig3": (
            "--num-sequences",
            "25",
            "--weak-probe-repeats",
            "4",
            "--region-ping-repeats",
            "2",
            "--n-null",
            "20",
        ),
        "fig4": (
            "--max-pairs",
            "80",
            "--random-mask-candidates",
            "12",
            "--n-null",
            "20",
            "--n-match-permutations",
            "200",
        ),
        "fig5": (
            "--max-trials",
            "80",
            "--n-null",
            "20",
        ),
        "fig6": (
            "--num-sequences",
            "15",
            "--num-probe-candidates-per-sequence",
            "4",
            "--n-null",
            "20",
            "--n-matched-groups",
            "20",
        ),
    },
}
BENCHMARK_PROFILES = tuple(BENCHMARK_PROFILE_ARGS_BY_FIG)


@dataclass(frozen=True)
class FigureRunSpec:
    fig_id: str
    experiment_id: str
    module: str
    main_subexperiments: tuple[str, ...]
    supplement_subexperiments: tuple[str, ...]
    main_flags: tuple[str, ...]
    supplement_flags: tuple[str, ...]
    both_flags: tuple[str, ...]

    def subexperiments_for_scope(self, scope: str) -> tuple[str, ...]:
        if scope == "main":
            return self.main_subexperiments
        if scope == "supplement":
            return self.supplement_subexperiments
        if scope == "both":
            out: list[str] = []
            for name in (*self.main_subexperiments, *self.supplement_subexperiments):
                if name not in out:
                    out.append(name)
            return tuple(out)
        raise ValueError(f"Unsupported scope: {scope}")

    def flags_for_scope(self, scope: str) -> tuple[str, ...]:
        if scope == "both":
            return self.both_flags
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
    benchmark_profile: str = BENCHMARK_PROFILE_NONE
    profile_args: tuple[str, ...] = ()
    resumed: bool = False
    dry_run: bool = False
    error: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildResult:
    fig_id: str
    build_fig_id: str
    experiment_id: str
    experiment_root: Path
    status: str
    returncode: int | None
    elapsed_seconds: float
    command: str
    skipped_reason: str = ""
    error: str = ""
    dry_run: bool = False


def _figure_spec_from_registry(fig_id: str) -> FigureRunSpec:
    registry = load_figure_registry(fig_id)
    return FigureRunSpec(
        fig_id=registry.fig_id,
        experiment_id=registry.experiment_id,
        module=registry.legacy_module,
        main_subexperiments=registry.main_subexperiments,
        supplement_subexperiments=registry.supplement_subexperiments,
        main_flags=registry.flags_for_scope("main"),
        supplement_flags=registry.flags_for_scope("supplement"),
        both_flags=registry.flags_for_scope("both"),
    )


FIGURE_SPECS: tuple[FigureRunSpec, ...] = tuple(_figure_spec_from_registry(fig_id) for fig_id in FIGURE_PACKAGE_IDS)
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


def benchmark_profile_args(profile: str, fig_id: str) -> tuple[str, ...]:
    profile_map = BENCHMARK_PROFILE_ARGS_BY_FIG.get(str(profile))
    if profile_map is None:
        raise ValueError(f"Unsupported benchmark profile: {profile}")
    return tuple(profile_map.get(str(fig_id), ()))


def benchmark_profile_args_by_fig(profile: str, specs: Sequence[FigureRunSpec]) -> dict[str, list[str]]:
    return {spec.fig_id: list(benchmark_profile_args(profile, spec.fig_id)) for spec in specs}


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
        missing_optional_supp = summary.get("missing_for_optional_supplementary") or []
        if bool(summary.get("supplement_extensions_run_by_default")) and missing_optional_supp:
            errors.append(f"missing_for_optional_supplementary is not empty: {missing_optional_supp}")
        if str(summary.get("figure")) == "fig6_peak_amplified_reentry":
            required_fig6_extensions = [
                run_dir / "data" / "metrics" / "supp_s11g_score_shuffle_null.csv",
                run_dir / "data" / "metrics" / "supp_s11h_threshold_sensitivity.csv",
            ]
            missing_fig6_extensions = [str(path.relative_to(run_dir)) for path in required_fig6_extensions if not path.exists()]
            if missing_fig6_extensions:
                errors.append(f"missing Fig.6 default supplement extensions: {missing_fig6_extensions}")

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
    benchmark_profile: str,
    save_debug_figures: bool,
    no_progress: bool,
    experiment_batch_size: int | None,
    fig1_dms_batch_size: int | None,
    fig1_delay_decode_backend: str | None,
    fig2_functional_readout_batch_size: int | None,
    fig4_l3_region_batch_size: int | None,
    enable_gpu_batching: bool,
    enable_gpu_metrics: bool,
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
    command.extend(benchmark_profile_args(benchmark_profile, spec.fig_id))
    if smoke:
        command.append("--smoke")
    if save_debug_figures:
        command.append("--save-debug-figures")
    if experiment_batch_size is not None:
        command.extend(["--batch-size", str(int(experiment_batch_size))])
    if spec.fig_id == "fig1" and fig1_dms_batch_size is not None:
        command.extend(["--dms-batch-size", str(int(fig1_dms_batch_size))])
    if spec.fig_id == "fig1" and fig1_delay_decode_backend is not None:
        command.extend(["--delay-decode-backend", str(fig1_delay_decode_backend)])
    if spec.fig_id == "fig2" and fig2_functional_readout_batch_size is not None:
        command.extend(["--functional-readout-batch-size", str(int(fig2_functional_readout_batch_size))])
    if spec.fig_id == "fig4" and fig4_l3_region_batch_size is not None:
        command.extend(["--l3-region-batch-size", str(int(fig4_l3_region_batch_size))])
    if enable_gpu_batching:
        command.extend(GPU_BATCH_FLAGS_BY_FIG.get(spec.fig_id, ()))
    if enable_gpu_metrics:
        command.extend(GPU_METRIC_FLAGS_BY_FIG.get(spec.fig_id, ()))
    if no_progress:
        command.append("--no-progress")
    return command


def _write_console(text: str, console: Any = sys.stdout) -> None:
    with _CONSOLE_LOCK:
        console.write(text)
        console.flush()


def _render_progress_event(payload: Mapping[str, Any], *, fallback_fig_id: str, fallback_seed: int) -> str:
    fig_id = str(payload.get("fig_id") or fallback_fig_id)
    network_seed = int(payload.get("network_seed") or fallback_seed)
    phase = str(payload.get("phase") or "phase")
    status = str(payload.get("status") or "progress")
    phase_index = payload.get("phase_index")
    total_phases = payload.get("total_phases")
    if phase_index is not None and total_phases is not None:
        prefix = f"[{fig_id} seed={network_seed} {int(phase_index)}/{int(total_phases)}]"
    else:
        prefix = f"[{fig_id} seed={network_seed}]"
    suffix = ""
    if "elapsed_seconds" in payload and status in {"done", "failed"}:
        suffix = f" elapsed={float(payload['elapsed_seconds']):.1f}s"
    detail = str(payload.get("detail") or "")
    if detail:
        suffix = f"{suffix} {detail}".rstrip()
    return f"{prefix} {status} {phase}{suffix}\n"


def _copy_stdout_stream(
    pipe: Any,
    log_handle: Any,
    *,
    progress_mode: str,
    fallback_fig_id: str,
    fallback_seed: int,
) -> None:
    for line in pipe:
        log_handle.write(line)
        log_handle.flush()
        if line.startswith(PROGRESS_EVENT_PREFIX):
            if progress_mode == "detailed":
                raw = line[len(PROGRESS_EVENT_PREFIX) :].strip()
                try:
                    payload = json.loads(raw)
                except Exception:
                    _write_console(line)
                else:
                    _write_console(_render_progress_event(payload, fallback_fig_id=fallback_fig_id, fallback_seed=fallback_seed))
            continue
        if progress_mode == "detailed":
            _write_console(line)


def _copy_raw_stream(pipe: Any, console: Any, log_handle: Any, *, progress_mode: str) -> None:
    while True:
        chunk = pipe.read(1)
        if not chunk:
            break
        log_handle.write(chunk)
        log_handle.flush()
        if progress_mode == "detailed":
            _write_console(chunk, console=console)


def _run_streaming_command(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    progress_mode: str,
    fallback_fig_id: str = "",
    fallback_seed: int = 0,
) -> tuple[int, float]:
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
            threads.append(
                threading.Thread(
                    target=_copy_stdout_stream,
                    args=(process.stdout, stdout_log),
                    kwargs={
                        "progress_mode": progress_mode,
                        "fallback_fig_id": fallback_fig_id,
                        "fallback_seed": int(fallback_seed),
                    },
                )
            )
        if process.stderr is not None:
            threads.append(threading.Thread(target=_copy_raw_stream, args=(process.stderr, sys.stderr, stderr_log), kwargs={"progress_mode": progress_mode}))
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
    benchmark_profile: str,
    resume: bool,
    save_debug_figures: bool,
    no_progress: bool,
    experiment_batch_size: int | None,
    fig1_dms_batch_size: int | None,
    fig1_delay_decode_backend: str | None,
    fig2_functional_readout_batch_size: int | None,
    fig4_l3_region_batch_size: int | None,
    enable_gpu_batching: bool,
    enable_gpu_metrics: bool,
    progress_mode: str,
    resource_plan: ResourcePlan,
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
        benchmark_profile=benchmark_profile,
        save_debug_figures=save_debug_figures,
        no_progress=no_progress,
        experiment_batch_size=experiment_batch_size,
        fig1_dms_batch_size=fig1_dms_batch_size,
        fig1_delay_decode_backend=fig1_delay_decode_backend,
        fig2_functional_readout_batch_size=fig2_functional_readout_batch_size,
        fig4_l3_region_batch_size=fig4_l3_region_batch_size,
        enable_gpu_batching=enable_gpu_batching,
        enable_gpu_metrics=enable_gpu_metrics,
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
            benchmark_profile=str(benchmark_profile),
            profile_args=benchmark_profile_args(benchmark_profile, spec.fig_id),
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
                benchmark_profile=str(benchmark_profile),
                profile_args=benchmark_profile_args(benchmark_profile, spec.fig_id),
                resumed=True,
                warnings=tuple(warnings),
            )
        if run_dir.exists() and errors:
            print(f"[Resume] {spec.fig_id} seed={checkpoint.seed} exists but does not satisfy scope={scope}; rerunning ({'; '.join(errors)})")

    env = resource_plan.apply_to_env(os.environ)
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
        progress_mode=progress_mode,
        fallback_fig_id=spec.fig_id,
        fallback_seed=checkpoint.seed,
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
        benchmark_profile=str(benchmark_profile),
        profile_args=benchmark_profile_args(benchmark_profile, spec.fig_id),
        error=error,
        warnings=tuple(warnings),
    )


def build_paper_figure_command(
    *,
    runtime_python: Path,
    build_fig_id: str,
    fig_root: Path,
    check_only: bool,
) -> list[str]:
    command = [
        str(runtime_python),
        "-m",
        "src.plotting.paper_fig.build",
        "--fig",
        str(build_fig_id),
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
    build_fig_id: str,
    output_root: Path,
    scope: str,
    no_build_paper_figures: bool,
    check_only_build: bool,
    dry_run: bool,
    fig_runs: Sequence[RunResult],
    progress_mode: str,
    resource_plan: ResourcePlan,
) -> BuildResult:
    fig_root = output_root / spec.experiment_id
    command = build_paper_figure_command(runtime_python=runtime_python, build_fig_id=build_fig_id, fig_root=fig_root, check_only=check_only_build)
    command_text = subprocess.list2cmdline(command)
    if no_build_paper_figures:
        return BuildResult(spec.fig_id, build_fig_id, spec.experiment_id, fig_root, "skipped", None, 0.0, command_text, skipped_reason="--no-build-paper-figures")
    if dry_run:
        return BuildResult(spec.fig_id, build_fig_id, spec.experiment_id, fig_root, "dry_run", None, 0.0, command_text, dry_run=True)
    failed_runs = [result for result in fig_runs if result.status != "success"]
    if failed_runs:
        return BuildResult(spec.fig_id, build_fig_id, spec.experiment_id, fig_root, "skipped", None, 0.0, command_text, skipped_reason="one or more selected runs failed")

    env = resource_plan.apply_to_env(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    log_dir = output_root / "_build_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    returncode, elapsed = _run_streaming_command(
        command,
        env=env,
        stdout_path=log_dir / f"{build_fig_id}_stdout.log",
        stderr_path=log_dir / f"{build_fig_id}_stderr.log",
        progress_mode=progress_mode,
        fallback_fig_id=build_fig_id,
    )
    status = "success" if returncode == 0 else "failed"
    error = "" if status == "success" else f"returncode={returncode}; see {log_dir / (build_fig_id + '_stderr.log')}"
    return BuildResult(spec.fig_id, build_fig_id, spec.experiment_id, fig_root, status, int(returncode), elapsed, command_text, error=error)


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
        "benchmark_profile": result.benchmark_profile,
        "profile_args": subprocess.list2cmdline(list(result.profile_args)),
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
        "build_fig_id": result.build_fig_id,
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
    resource_plan: ResourcePlan,
    specs: Sequence[FigureRunSpec],
    checkpoints: Sequence[NetworkCheckpoint],
    run_results: Sequence[RunResult],
    build_results: Sequence[BuildResult],
) -> dict[str, Any]:
    run_records = [_run_record(result, output_root) for result in sorted(run_results, key=lambda item: (item.fig_id, item.network_seed))]
    build_records = [_build_record(result, output_root) for result in sorted(build_results, key=lambda item: (item.fig_id, item.build_fig_id))]
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
        "benchmark_profile",
        "profile_args",
        "resumed",
        "dry_run",
        "warnings",
        "error",
        "command",
    ]
    build_fields = ["fig_id", "build_fig_id", "experiment_id", "experiment_root", "status", "returncode", "elapsed_seconds", "skipped_reason", "dry_run", "error", "command"]
    _write_csv(report_dir / "run_manifest.csv", run_records, run_fields)
    _write_csv(report_dir / "failed_runs.csv", failed_run_records, run_fields)
    _write_csv(report_dir / "build_manifest.csv", build_records, build_fields)
    _write_json(report_dir / "resource_plan.json", resource_plan.as_dict())
    profile_args_by_fig = benchmark_profile_args_by_fig(str(args.benchmark_profile), specs)
    _write_json(
        report_dir / "run_manifest.json",
        {
            "runs": run_records,
            "builds": build_records,
            "resource_plan": resource_plan.as_dict(),
            "benchmark_profile": str(args.benchmark_profile),
            "benchmark_profile_args_by_fig": profile_args_by_fig,
        },
    )

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
        "benchmark_profile": str(args.benchmark_profile),
        "benchmark_profile_args_by_fig": profile_args_by_fig,
        "jobs": int(getattr(args, "effective_jobs", args.jobs)),
        "requested_jobs": int(args.jobs),
        "seed_jobs": None if args.seed_jobs is None else int(args.seed_jobs),
        "parallel_axis": str(args.parallel_axis),
        "cpu_workers": int(resource_plan.cpu_workers),
        "cpu_threads_per_worker": resource_plan.cpu_threads_per_worker,
        "available_cpu_count": int(resource_plan.available_cpu_count),
        "max_build_workers": int(resource_plan.max_build_workers),
        "resource_plan_notes": list(resource_plan.notes),
        "experiment_batch_size": None if args.experiment_batch_size is None else int(args.experiment_batch_size),
        "fig1_dms_batch_size": None if args.fig1_dms_batch_size is None else int(args.fig1_dms_batch_size),
        "fig1_delay_decode_backend": None if args.fig1_delay_decode_backend is None else str(args.fig1_delay_decode_backend),
        "fig2_functional_readout_batch_size": None if args.fig2_functional_readout_batch_size is None else int(args.fig2_functional_readout_batch_size),
        "fig4_l3_region_batch_size": None if args.fig4_l3_region_batch_size is None else int(args.fig4_l3_region_batch_size),
        "enable_gpu_batching": bool(args.enable_gpu_batching),
        "enable_gpu_metrics": bool(args.enable_gpu_metrics),
        "continue_on_error": bool(args.continue_on_error),
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "progress_mode": str(getattr(args, "effective_progress_mode", args.progress_mode)),
        "requested_progress_mode": str(args.progress_mode),
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
    parser.add_argument(
        "--benchmark-profile",
        choices=BENCHMARK_PROFILES,
        default=BENCHMARK_PROFILE_NONE,
        help="Apply a predefined medium-size benchmark workload. Cannot be combined with --smoke.",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed-jobs", type=int, default=None, help="Alias for process-level concurrency across selected fig/seed runs. Overrides --jobs when set.")
    parser.add_argument("--cpu-workers", type=int, default=None, help="CPU worker budget recorded and forwarded to child runs.")
    parser.add_argument("--cpu-threads-per-worker", type=int, default=None, help="Set BLAS/OpenMP thread env vars for each child run.")
    parser.add_argument("--parallel-axis", choices=list(PARALLEL_AXES), default="auto", help="Resource-planning label for how concurrency is being used.")
    parser.add_argument("--max-build-workers", type=int, default=1, help="Maximum concurrent paper-figure build/check jobs.")
    parser.add_argument("--experiment-batch-size", type=int, default=None, help="Forward a larger --batch-size to each selected paper-figure experiment.")
    parser.add_argument("--fig1-dms-batch-size", type=int, default=None, help="Forward --dms-batch-size to Fig.1 when selected.")
    parser.add_argument(
        "--fig1-delay-decode-backend",
        choices=["torch_linear_probe", "sklearn_linear_svc"],
        default=None,
        help="Forward Fig.1 delay decoder backend selection.",
    )
    parser.add_argument("--fig4-l3-region-batch-size", type=int, default=None, help="Forward --l3-region-batch-size to Fig.4 when selected.")
    parser.add_argument("--fig2-functional-readout-batch-size", type=int, default=None, help="Forward --functional-readout-batch-size to Fig.2 when selected.")
    parser.add_argument("--enable-gpu-batching", action="store_true", help="Forward figure-specific GPU batching flags where supported.")
    parser.add_argument("--enable-gpu-metrics", action="store_true", help="Forward GPU metric helpers where supported.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--force", dest="resume", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-build-paper-figures", action="store_true")
    parser.add_argument("--check-only-build", action="store_true")
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--progress-mode", choices=sorted(PROGRESS_MODES), default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _effective_progress_mode(args: argparse.Namespace, *, checkpoints: Sequence[NetworkCheckpoint]) -> str:
    if bool(args.no_progress):
        return "off"
    requested = str(args.progress_mode)
    if requested != "auto":
        return requested
    if int(getattr(args, "effective_jobs", args.jobs)) == 1 and len(checkpoints) == 1:
        return "detailed"
    return "compact"


def _effective_jobs(args: argparse.Namespace) -> int:
    return int(args.seed_jobs) if args.seed_jobs is not None else int(args.jobs)


def _format_elapsed(seconds: float) -> str:
    return f"{float(seconds):.1f}s"


def build_fig_ids_for_scope(spec: FigureRunSpec, scope: str) -> tuple[str, ...]:
    supplement_id = f"{spec.fig_id}_supp"
    if scope == "main":
        return (spec.fig_id,)
    if scope == "supplement":
        return (supplement_id,)
    if scope == "both":
        return (spec.fig_id, supplement_id)
    raise ValueError(f"Unsupported scope: {scope}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.smoke) and str(args.benchmark_profile) != BENCHMARK_PROFILE_NONE:
        raise SystemExit("--smoke cannot be combined with --benchmark-profile; choose one workload size.")
    try:
        resource_plan = resolve_resource_plan(
            device=str(args.device),
            jobs=int(args.jobs),
            seed_jobs=None if args.seed_jobs is None else int(args.seed_jobs),
            cpu_workers=args.cpu_workers,
            cpu_threads_per_worker=args.cpu_threads_per_worker,
            parallel_axis=str(args.parallel_axis),
            max_build_workers=int(args.max_build_workers),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    effective_jobs = int(resource_plan.effective_jobs)
    setattr(args, "effective_jobs", effective_jobs)

    specs = parse_figs(args.figs)
    checkpoints = select_checkpoints(discover_checkpoints(args.model_path_glob), seeds=str(args.seeds), all_seeds=bool(args.all_seeds))
    effective_progress_mode = _effective_progress_mode(args, checkpoints=checkpoints)
    setattr(args, "effective_progress_mode", effective_progress_mode)
    output_root = _resolve_repo_path(args.output_root)
    dataset_root = _resolve_repo_path(args.dataset_root)
    runtime_python = _resolve_runtime_python()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_root / "_batch_runs" / timestamp

    print(f"[Setup] figs={','.join(spec.fig_id for spec in specs)} scope={args.scope} seeds={','.join(str(c.seed) for c in checkpoints)}")
    print(f"[Setup] runtime_python={runtime_python}")
    print(f"[Setup] output_root={output_root}")
    print(f"[Setup] progress_mode={effective_progress_mode}{' (requested=' + str(args.progress_mode) + ')' if effective_progress_mode != args.progress_mode else ''}")
    print(f"[Setup] benchmark_profile={args.benchmark_profile}")
    print(
        f"[Setup] jobs={effective_jobs}"
        f"{' (seed_jobs=' + str(args.seed_jobs) + ')' if args.seed_jobs is not None else ''}"
        f"{' experiment_batch_size=' + str(args.experiment_batch_size) if args.experiment_batch_size is not None else ''}"
        f"{' enable_gpu_batching' if args.enable_gpu_batching else ''}"
        f"{' enable_gpu_metrics' if args.enable_gpu_metrics else ''}"
    )
    print(
        f"[Setup] resource_plan parallel_axis={resource_plan.parallel_axis} "
        f"cpu_workers={resource_plan.cpu_workers} "
        f"cpu_threads_per_worker={resource_plan.cpu_threads_per_worker} "
        f"max_build_workers={resource_plan.max_build_workers}"
    )
    for note in resource_plan.notes:
        print(f"[Setup] resource_note={note}")

    tasks = [(spec, checkpoint) for spec in specs for checkpoint in checkpoints]
    total_tasks = len(tasks)
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
            benchmark_profile=str(args.benchmark_profile),
            resume=bool(args.resume),
            save_debug_figures=bool(args.save_debug_figures),
            no_progress=effective_progress_mode in {"compact", "off"},
            experiment_batch_size=args.experiment_batch_size,
            fig1_dms_batch_size=args.fig1_dms_batch_size,
            fig1_delay_decode_backend=args.fig1_delay_decode_backend,
            fig2_functional_readout_batch_size=args.fig2_functional_readout_batch_size,
            fig4_l3_region_batch_size=args.fig4_l3_region_batch_size,
            enable_gpu_batching=bool(args.enable_gpu_batching),
            enable_gpu_metrics=bool(args.enable_gpu_metrics),
            progress_mode=effective_progress_mode,
            resource_plan=resource_plan,
            dry_run=bool(args.dry_run),
        )

    if effective_jobs <= 1:
        for task_index, (spec, checkpoint) in enumerate(tasks, start=1):
            print(f"[Run {task_index}/{total_tasks}] seed={checkpoint.seed} {spec.fig_id} running")
            result = submit_one(spec, checkpoint)
            run_results.append(result)
            resumed = " resumed" if result.resumed else ""
            print(f"[Run {task_index}/{total_tasks}] seed={checkpoint.seed} {spec.fig_id} done status={result.status}{resumed} elapsed={_format_elapsed(result.elapsed_seconds)}")
            if result.status not in {"success", "dry_run"}:
                failures.append(f"{spec.fig_id} seed={checkpoint.seed}: {result.error}")
                if not args.continue_on_error:
                    break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_jobs) as executor:
            future_map = {}
            for task_index, (spec, checkpoint) in enumerate(tasks, start=1):
                print(f"[Run {task_index}/{total_tasks}] seed={checkpoint.seed} {spec.fig_id} queued")
                future_map[executor.submit(submit_one, spec, checkpoint)] = (task_index, spec, checkpoint)
            for future in concurrent.futures.as_completed(future_map):
                task_index, spec, checkpoint = future_map[future]
                result = future.result()
                run_results.append(result)
                resumed = " resumed" if result.resumed else ""
                print(f"[Run {task_index}/{total_tasks}] seed={checkpoint.seed} {spec.fig_id} done status={result.status}{resumed} elapsed={_format_elapsed(result.elapsed_seconds)}")
                if result.status not in {"success", "dry_run"}:
                    failures.append(f"{spec.fig_id} seed={checkpoint.seed}: {result.error}")
        run_results.sort(key=lambda item: (item.fig_id, item.network_seed))

    build_results: list[BuildResult] = []
    if failures and not args.continue_on_error:
        for spec in specs:
            for build_fig_id in build_fig_ids_for_scope(spec, str(args.scope)):
                build_results.append(
                    BuildResult(spec.fig_id, build_fig_id, spec.experiment_id, output_root / spec.experiment_id, "skipped", None, 0.0, "", skipped_reason="run phase failed")
                )
    else:
        build_targets = [(spec, build_fig_id) for spec in specs for build_fig_id in build_fig_ids_for_scope(spec, str(args.scope))]
        for build_index, (spec, build_fig_id) in enumerate(build_targets, start=1):
            fig_runs = [result for result in run_results if result.fig_id == spec.fig_id]
            print(f"[Build {build_index}/{len(build_targets)}] {build_fig_id} running")
            result = run_build(
                runtime_python=runtime_python,
                spec=spec,
                build_fig_id=build_fig_id,
                output_root=output_root,
                scope=str(args.scope),
                no_build_paper_figures=bool(args.no_build_paper_figures),
                check_only_build=bool(args.check_only_build),
                dry_run=bool(args.dry_run),
                fig_runs=fig_runs,
                progress_mode=effective_progress_mode,
                resource_plan=resource_plan,
            )
            build_results.append(result)
            print(
                f"[Build {build_index}/{len(build_targets)}] {build_fig_id} done status={result.status} "
                f"elapsed={_format_elapsed(result.elapsed_seconds)}"
                f"{' (' + result.skipped_reason + ')' if result.skipped_reason else ''}"
            )
            if result.status == "failed":
                failures.append(f"{build_fig_id} build: {result.error}")
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
        resource_plan=resource_plan,
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
