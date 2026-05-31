from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


FIGURE_IDS = {
    "fig1": "fig1_functional_stsp_substrate",
    "fig2": "fig2_pair_fused_stsp_state",
    "fig3": "fig3_multiitem_peak_landscape",
    "fig4": "fig4_overlap_reentry",
    "fig5": "fig5_local_support_competition",
    "fig6": "fig6_peak_amplified_reentry",
}

FIGURE_MODULES = {
    "fig1": "src.experiments.paper_figures.fig1.run_task",
    "fig2": "src.experiments.paper_figures.fig2.run_task",
    "fig3": "src.experiments.paper_figures.fig3.run_task",
    "fig4": "src.experiments.paper_figures.fig4.run_task",
    "fig5": "src.experiments.paper_figures.fig5.run_task",
    "fig6": "src.experiments.paper_figures.fig6.run_task",
}

SHARED_SEQUENCE_MODULE = "src.experiments.paper_figures.common.sequence_root.run_task"
SHARED_SEQUENCE_TASK = "shared_sequence_root_bank"

PRESETS = ("core", "extended", "fig3_fig6", "fig4")
REUSE_MODES = ("auto", "force")


@dataclass(frozen=True)
class WarmupTask:
    kind: str
    task_id: str


CORE_TASKS = (
    WarmupTask("fig1", "dms_boundary_bank"),
    WarmupTask("fig2", "state_bank"),
    WarmupTask("fig2", "completion_delay_boundary_bank"),
    WarmupTask("shared_sequence_root", SHARED_SEQUENCE_TASK),
    WarmupTask("fig3", "state_bank"),
    WarmupTask("fig6", "sequence_bank"),
    WarmupTask("fig4", "rollouts"),
    WarmupTask("fig5", "preprobe_support_bank"),
)

EXTENDED_EXTRA_TASKS = (
    WarmupTask("fig1", "delay_feature_bank"),
    WarmupTask("fig2", "partial_cue_mask_specs"),
    WarmupTask("fig2", "completion_delay_mask_specs"),
    WarmupTask("fig4", "similarity_entry"),
)

FIG3_FIG6_TASKS = (
    WarmupTask("shared_sequence_root", SHARED_SEQUENCE_TASK),
    WarmupTask("fig3", "state_bank"),
    WarmupTask("fig6", "sequence_bank"),
)

FIG4_TASKS = (
    WarmupTask("fig4", "rollouts"),
    WarmupTask("fig4", "similarity_entry"),
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = _resolve_path(args.output_root)
    seeds = _parse_seeds(args.seeds)
    tasks = _preset_tasks(args.preset)
    manifest_path = output_root / "warmup_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "entrypoint": "src.experiments.paper_figures.run_upstream_artifact_warmup",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "preset": str(args.preset),
        "reuse_artifacts": str(args.reuse_artifacts),
        "batch_size": None if args.batch_size is None else int(args.batch_size),
        "output_root": str(output_root),
        "seeds": seeds,
        "dry_run": bool(args.dry_run),
        "continue_on_error": bool(args.continue_on_error),
        "tasks": [],
        "downstream_require_examples": _require_examples(args, output_root, seeds),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    planned = _build_plan(args, output_root=output_root, seeds=seeds, tasks=tasks)
    manifest["tasks"] = [
        _manifest_entry(item, status="planned" if args.dry_run else "pending")
        for item in planned
    ]
    _write_manifest(manifest_path, manifest)

    if args.dry_run:
        for item in planned:
            print(_display_command(item["command"]))
        return 0

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    had_failure = False
    total = len(planned)
    overall_start = time.perf_counter()
    if not args.no_progress:
        print(
            f"[warmup] planned {total} task(s) "
            f"preset={args.preset} seeds={','.join(str(seed) for seed in seeds)} output_root={output_root}"
        )
    for index, item in enumerate(planned):
        progress = _progress_prefix(index + 1, total)
        if not args.no_progress:
            print(f"{progress} start seed={item['seed']} target={_task_label(item)}")
        entry = manifest["tasks"][index]
        entry["start_time"] = _utc_now()
        entry["status"] = "running"
        manifest["updated_at"] = _utc_now()
        _write_manifest(manifest_path, manifest)

        start = time.perf_counter()
        proc = subprocess.run(item["command"], env=env)
        entry["end_time"] = _utc_now()
        entry["duration_seconds"] = round(time.perf_counter() - start, 3)
        entry["return_code"] = int(proc.returncode)
        entry["status"] = "success" if proc.returncode == 0 else "failed"
        manifest["updated_at"] = _utc_now()
        _write_manifest(manifest_path, manifest)

        if proc.returncode != 0:
            had_failure = True
            if not args.no_progress:
                print(
                    f"{progress} failed seed={item['seed']} target={_task_label(item)} "
                    f"rc={proc.returncode} elapsed={entry['duration_seconds']:.1f}s "
                    f"total_elapsed={time.perf_counter() - overall_start:.1f}s"
                )
            if not args.continue_on_error:
                break
        elif not args.no_progress:
            print(
                f"{progress} success seed={item['seed']} target={_task_label(item)} "
                f"elapsed={entry['duration_seconds']:.1f}s "
                f"total_elapsed={time.perf_counter() - overall_start:.1f}s"
            )

    if not args.no_progress:
        completed = sum(1 for task in manifest["tasks"] if task["status"] == "success")
        failed = sum(1 for task in manifest["tasks"] if task["status"] == "failed")
        pending = sum(1 for task in manifest["tasks"] if task["status"] == "pending")
        print(
            f"[warmup] finished success={completed} failed={failed} pending={pending} "
            f"elapsed={time.perf_counter() - overall_start:.1f}s manifest={manifest_path}"
        )

    return 1 if had_failure else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm up reusable upstream artifacts for the paper-figure DAG.",
        allow_abbrev=False,
    )
    parser.add_argument("--seeds", nargs="+", default=["1000"], help="Network seeds, as space- or comma-separated integers.")
    parser.add_argument("--output-root", default="results/paper_figure_warm_cache")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset-root", default="MNIST")
    parser.add_argument("--model-path-glob", default="results/multi_snn/sdnn_ensemble_20/sdnn_ensemble_20/seed_*/net_final.pth")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size forwarded to figure-local and shared sequence-root tasks.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preset", default="core", choices=PRESETS)
    parser.add_argument("--reuse-artifacts", default="auto", choices=REUSE_MODES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def _parse_seeds(values: Sequence[str]) -> list[int]:
    seeds: list[int] = []
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if item:
                seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _preset_tasks(preset: str) -> tuple[WarmupTask, ...]:
    if preset == "core":
        return CORE_TASKS
    if preset == "extended":
        return (*CORE_TASKS, *EXTENDED_EXTRA_TASKS)
    if preset == "fig3_fig6":
        return FIG3_FIG6_TASKS
    if preset == "fig4":
        return FIG4_TASKS
    raise ValueError(f"Unsupported warmup preset: {preset}")


def _build_plan(
    args: argparse.Namespace,
    *,
    output_root: Path,
    seeds: Sequence[int],
    tasks: Sequence[WarmupTask],
) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for seed in seeds:
        shared_root = _shared_root_bank_dir(output_root, seed)
        for task in tasks:
            if task.kind == "shared_sequence_root":
                planned.append(_shared_sequence_item(args, output_root=output_root, seed=seed))
            else:
                planned.append(_figure_item(args, output_root=output_root, seed=seed, task=task, shared_root=shared_root))
    return planned


def _shared_sequence_item(args: argparse.Namespace, *, output_root: Path, seed: int) -> dict[str, Any]:
    output_dir = output_root / "shared_sequence_root" / f"seed_{seed}"
    artifact_root = _shared_artifact_root(output_root, seed)
    task_dir = artifact_root / SHARED_SEQUENCE_TASK
    command = [
        sys.executable,
        "-m",
        SHARED_SEQUENCE_MODULE,
        "--task",
        SHARED_SEQUENCE_TASK,
        "--reuse-artifacts",
        str(args.reuse_artifacts),
        "--output-dir",
        str(output_dir),
        "--artifact-root",
        str(artifact_root),
        "--network-seed",
        str(seed),
        "--device",
        str(args.device),
        "--dataset-root",
        str(args.dataset_root),
        "--model-path-glob",
        str(args.model_path_glob),
        "--split",
        str(args.split),
    ]
    _append_common_flags(command, args)
    return {
        "seed": int(seed),
        "kind": "shared_sequence_root",
        "figure": "",
        "task": SHARED_SEQUENCE_TASK,
        "module": SHARED_SEQUENCE_MODULE,
        "command": command,
        "output_root": str(output_dir),
        "artifact_root": str(artifact_root),
        "task_artifact_dir": str(task_dir),
        "shared_sequence_root_path": str(task_dir),
    }


def _figure_item(
    args: argparse.Namespace,
    *,
    output_root: Path,
    seed: int,
    task: WarmupTask,
    shared_root: Path,
) -> dict[str, Any]:
    figure = task.kind
    output_dir = output_root / figure
    artifact_root = output_dir / FIGURE_IDS[figure] / f"seed_{seed}" / "data" / "intermediates"
    task_dir = artifact_root / task.task_id
    command = [
        sys.executable,
        "-m",
        FIGURE_MODULES[figure],
        "--task",
        task.task_id,
        "--reuse-artifacts",
        str(args.reuse_artifacts),
        "--output-dir",
        str(output_dir),
        "--network-seed",
        str(seed),
        "--device",
        str(args.device),
        "--dataset-root",
        str(args.dataset_root),
        "--model-path-glob",
        str(args.model_path_glob),
        "--split",
        str(args.split),
    ]
    if figure in {"fig3", "fig6"}:
        command.extend(["--shared-sequence-root", str(shared_root)])
    _append_common_flags(command, args)
    return {
        "seed": int(seed),
        "kind": "figure",
        "figure": figure,
        "task": task.task_id,
        "module": FIGURE_MODULES[figure],
        "command": command,
        "output_root": str(output_dir),
        "artifact_root": str(artifact_root),
        "task_artifact_dir": str(task_dir),
        "shared_sequence_root_path": str(shared_root) if figure in {"fig3", "fig6"} else "",
    }


def _append_common_flags(command: list[str], args: argparse.Namespace) -> None:
    if args.batch_size is not None:
        command.extend(["--batch-size", str(int(args.batch_size))])
    if args.smoke:
        command.append("--smoke")
    if args.no_progress:
        command.append("--no-progress")


def _task_label(item: Mapping[str, Any]) -> str:
    figure = str(item.get("figure") or item.get("kind") or "task")
    return f"{figure}:{item.get('task')}"


def _progress_prefix(index: int, total: int) -> str:
    total = max(1, int(total))
    index = min(max(1, int(index)), total)
    width = 20
    fraction = index / total
    filled = min(width, max(0, round(width * fraction)))
    bar = "#" * filled + "-" * (width - filled)
    return f"[warmup {index}/{total} {fraction * 100:5.1f}% {bar}]"


def _manifest_entry(item: dict[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "seed": item["seed"],
        "figure": item["figure"],
        "kind": item["kind"],
        "task": item["task"],
        "module": item["module"],
        "command": item["command"],
        "command_display": _display_command(item["command"]),
        "status": status,
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "return_code": None,
        "output_root": item["output_root"],
        "artifact_root": item["artifact_root"],
        "task_artifact_dir": item["task_artifact_dir"],
        "shared_sequence_root_path": item["shared_sequence_root_path"],
    }


def _require_examples(args: argparse.Namespace, output_root: Path, seeds: Sequence[int]) -> dict[str, str]:
    seed = int(seeds[0])
    shared_root = _shared_root_bank_dir(output_root, seed)
    return {
        "fig1_dms_shuffle_readout": _display_command(
            _example_figure_command(args, output_root, seed, "fig1", "dms_shuffle_readout")
        ),
        "fig2_partial_cue": _display_command(_example_figure_command(args, output_root, seed, "fig2", "partial_cue")),
        "fig3_progressive_update": _display_command(
            _example_figure_command(
                args,
                output_root,
                seed,
                "fig3",
                "progressive_update",
                shared_sequence_root=shared_root,
            )
        ),
        "fig4_decision_deflection": _display_command(
            _example_figure_command(args, output_root, seed, "fig4", "decision_deflection")
        ),
        "fig5_preprobe_support": _display_command(
            _example_figure_command(args, output_root, seed, "fig5", "preprobe_support")
        ),
        "fig6_field_ping_readout": _display_command(
            _example_figure_command(
                args,
                output_root,
                seed,
                "fig6",
                "field_ping_readout",
                shared_sequence_root=shared_root,
            )
        ),
    }


def _example_figure_command(
    args: argparse.Namespace,
    output_root: Path,
    seed: int,
    figure: str,
    task: str,
    *,
    shared_sequence_root: Path | None = None,
) -> list[str]:
    command = [
        "python",
        "-m",
        FIGURE_MODULES[figure],
        "--task",
        task,
        "--reuse-artifacts",
        "require",
        "--output-dir",
        str(output_root / figure),
        "--network-seed",
        str(seed),
        "--device",
        str(args.device),
        "--dataset-root",
        str(args.dataset_root),
        "--model-path-glob",
        str(args.model_path_glob),
        "--split",
        str(args.split),
    ]
    if shared_sequence_root is not None:
        command.extend(["--shared-sequence-root", str(shared_sequence_root)])
    _append_common_flags(command, args)
    return command


def _shared_artifact_root(output_root: Path, seed: int) -> Path:
    return output_root / "shared_sequence_root" / f"seed_{seed}" / "data" / "intermediates"


def _shared_root_bank_dir(output_root: Path, seed: int) -> Path:
    return _shared_artifact_root(output_root, seed) / SHARED_SEQUENCE_TASK


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _display_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
