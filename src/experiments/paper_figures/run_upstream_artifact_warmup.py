from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS


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

DEFAULT_WARMUP_OUTPUT_ROOT = Path("results/paper_figure_warm_cache")
MULTI_SEED_ROLLOUT_OUTPUT_ROOT = Path("results/multi_seed_rollout")
MULTI_SEED_ROLLOUT_SEEDS = tuple(range(1000, 1020))
DEFAULT_DATASET_ROOT = str(DEFAULT_PROJECT_DEFAULTS.paths.dataset_root)

PRESETS = ("core", "extended", "fig3_fig6", "fig4", "multi_seed_rollout")
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

MULTI_SEED_ROLLOUT_TASKS = (
    WarmupTask("fig1", "dms_boundary_bank"),
    WarmupTask("fig2", "state_bank"),
    WarmupTask("fig2", "completion_delay_boundary_bank"),
    WarmupTask("shared_sequence_root", SHARED_SEQUENCE_TASK),
    WarmupTask("fig3", "state_bank"),
    WarmupTask("fig3", "boundary_condition_specs"),
    WarmupTask("fig4", "rollouts"),
    WarmupTask("fig6", "sequence_bank"),
    WarmupTask("fig5", "preprobe_support_bank"),
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = _resolve_path(_default_output_root(args))
    seeds = _parse_seeds(args.seeds, default=_default_seeds(args.preset))
    tasks = _preset_tasks(args.preset)
    skip_existing = _skip_existing(args)
    manifest_path = output_root / "warmup_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "entrypoint": "src.experiments.paper_figures.run_upstream_artifact_warmup",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "preset": str(args.preset),
        "preset_description": _preset_description(args.preset),
        "reuse_artifacts": str(args.reuse_artifacts),
        "batch_size": None if args.batch_size is None else int(args.batch_size),
        "output_root": str(output_root),
        "seeds": seeds,
        "skip_existing": bool(skip_existing),
        "skip_existing_check": "task artifact exists and the seed bundle has a successful summary/run_info marker",
        "dry_run": bool(args.dry_run),
        "continue_on_error": bool(args.continue_on_error),
        "tasks": [],
        "downstream_require_examples": _require_examples(args, output_root, seeds),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    planned = _build_plan(args, output_root=output_root, seeds=seeds, tasks=tasks)
    for item in planned:
        item["existing_artifact"] = _check_existing_artifact(Path(item["task_artifact_dir"]))
        item["existing_bundle"] = _check_existing_bundle(item)
        item["skip_existing"] = bool(skip_existing and item["existing_artifact"]["ok"] and item["existing_bundle"]["ok"])
    manifest["tasks"] = [
        _manifest_entry(item, status=_initial_status(item, dry_run=bool(args.dry_run)))
        for item in planned
    ]
    manifest["planned_count"] = len(planned)
    manifest["skipped_count"] = sum(1 for item in planned if item["skip_existing"])
    manifest["executable_count"] = len(planned) - int(manifest["skipped_count"])
    _write_manifest(manifest_path, manifest)

    if args.dry_run:
        for item in planned:
            if item["skip_existing"]:
                print(f"[skip-existing] seed={item['seed']} target={_task_label(item)} artifact={item['task_artifact_dir']}")
            else:
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
            f"execute={manifest['executable_count']} skip={manifest['skipped_count']} "
            f"preset={args.preset} seeds={','.join(str(seed) for seed in seeds)} output_root={output_root}"
        )
    for index, item in enumerate(planned):
        progress = _progress_prefix(index + 1, total)
        entry = manifest["tasks"][index]
        if item["skip_existing"]:
            entry["start_time"] = _utc_now()
            entry["end_time"] = entry["start_time"]
            entry["duration_seconds"] = 0.0
            entry["return_code"] = 0
            entry["status"] = "skipped"
            manifest["updated_at"] = _utc_now()
            _write_manifest(manifest_path, manifest)
            if not args.no_progress:
                print(f"{progress} skip seed={item['seed']} target={_task_label(item)} reason=existing_artifact")
            continue

        if not args.no_progress:
            print(f"{progress} start seed={item['seed']} target={_task_label(item)}")
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
        skipped = sum(1 for task in manifest["tasks"] if task["status"] == "skipped")
        pending = sum(1 for task in manifest["tasks"] if task["status"] == "pending")
        print(
            f"[warmup] finished success={completed} skipped={skipped} failed={failed} pending={pending} "
            f"elapsed={time.perf_counter() - overall_start:.1f}s manifest={manifest_path}"
        )

    return 1 if had_failure else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm up reusable upstream artifacts for the paper-figure DAG.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Network seeds, as space-, comma-, or range-separated integers. Example: 1000-1019.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root. Defaults to results/multi_seed_rollout for --preset multi_seed_rollout, otherwise results/paper_figure_warm_cache.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-path-glob", default="results/multi_snn/sdnn_ensemble_20/sdnn_ensemble_20/seed_*/net_final.pth")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size forwarded to figure-local and shared sequence-root tasks.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preset", default="core", choices=PRESETS)
    parser.add_argument("--reuse-artifacts", default="auto", choices=REUSE_MODES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a task only when its artifact directory and seed bundle success markers are already present.",
    )
    skip_group.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Disable the multi_seed_rollout preset's default skip-existing behavior.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _parse_seeds(values: Sequence[str] | None, *, default: Sequence[int]) -> list[int]:
    if values is None:
        return [int(seed) for seed in default]
    seeds: list[int] = []
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if not item:
                continue
            range_sep = ".." if ".." in item else "-" if "-" in item and not item.startswith("-") else ""
            if range_sep:
                start_text, end_text = item.split(range_sep, 1)
                start = int(start_text)
                end = int(end_text)
                step = 1 if end >= start else -1
                seeds.extend(range(start, end + step, step))
            else:
                seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return sorted(dict.fromkeys(seeds))


def _default_output_root(args: argparse.Namespace) -> str | Path:
    if args.output_root is not None:
        return args.output_root
    if args.preset == "multi_seed_rollout":
        return MULTI_SEED_ROLLOUT_OUTPUT_ROOT
    return DEFAULT_WARMUP_OUTPUT_ROOT


def _default_seeds(preset: str) -> tuple[int, ...]:
    if preset == "multi_seed_rollout":
        return MULTI_SEED_ROLLOUT_SEEDS
    return (1000,)


def _skip_existing(args: argparse.Namespace) -> bool:
    if args.reuse_artifacts == "force":
        return False
    if args.rerun_existing:
        return False
    return bool(args.skip_existing or args.preset == "multi_seed_rollout")


def _preset_tasks(preset: str) -> tuple[WarmupTask, ...]:
    if preset == "core":
        return CORE_TASKS
    if preset == "extended":
        return (*CORE_TASKS, *EXTENDED_EXTRA_TASKS)
    if preset == "fig3_fig6":
        return FIG3_FIG6_TASKS
    if preset == "fig4":
        return FIG4_TASKS
    if preset == "multi_seed_rollout":
        return MULTI_SEED_ROLLOUT_TASKS
    raise ValueError(f"Unsupported warmup preset: {preset}")


def _preset_description(preset: str) -> str:
    if preset == "multi_seed_rollout":
        return (
            "Backfill the current 20-seed multi_seed_rollout target: fig1/fig2/fig6 upstream tail seeds, "
            "fig3 state_bank tail seeds plus boundary_condition_specs for all seeds, fig4 rollouts, shared sequence roots, and fig5 support banks."
        )
    if preset == "core":
        return "Warm up the core upstream artifact banks for active paper figures."
    if preset == "extended":
        return "Warm up core upstream banks plus selected extended reusable specs."
    if preset == "fig3_fig6":
        return "Warm up shared sequence-root, Fig.3 state-bank, and Fig.6 sequence-bank artifacts."
    if preset == "fig4":
        return "Warm up Fig.4 rollout and similarity-entry artifacts."
    return ""


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
    _append_common_flags(command, args, figure=figure)
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


def _append_common_flags(command: list[str], args: argparse.Namespace, *, figure: str | None = None) -> None:
    batch_size = _effective_batch_size(args, figure=figure)
    if batch_size is not None:
        command.extend(["--batch-size", str(int(batch_size))])
    if figure == "fig3":
        command.append("--enable-state-bank-batch")
    if args.smoke:
        command.append("--smoke")
    if args.no_progress:
        command.append("--no-progress")


def _effective_batch_size(args: argparse.Namespace, *, figure: str | None = None) -> int | None:
    if args.batch_size is not None:
        return int(args.batch_size)
    if args.preset == "multi_seed_rollout" and figure == "fig6":
        return 4
    return None


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
        "existing_artifact": item["existing_artifact"],
        "existing_bundle": item["existing_bundle"],
        "skip_existing": item["skip_existing"],
    }


def _initial_status(item: Mapping[str, Any], *, dry_run: bool) -> str:
    if item.get("skip_existing"):
        return "skipped" if not dry_run else "planned_skip"
    return "planned" if dry_run else "pending"


def _check_existing_artifact(task_dir: Path) -> dict[str, Any]:
    missing: list[str] = []
    if not task_dir.is_dir():
        return {"ok": False, "task_artifact_dir": str(task_dir), "missing": [str(task_dir)]}

    cache_key = task_dir / "cache_key.json"
    if not _is_nonempty_file(cache_key):
        missing.append(str(cache_key))

    manifest_candidates = sorted(task_dir.glob("*manifest*.csv"))
    usable_manifests = [path for path in manifest_candidates if _is_nonempty_csv(path)]
    if not usable_manifests:
        missing.append(f"{task_dir}/*manifest*.csv")

    return {
        "ok": not missing,
        "task_artifact_dir": str(task_dir),
        "missing": missing,
        "manifest_candidates": [str(path) for path in manifest_candidates],
        "usable_manifests": [str(path) for path in usable_manifests],
    }


def _check_existing_bundle(item: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(str(item["output_root"]))
    if item.get("kind") == "shared_sequence_root":
        summary_path = output_root / "shared_sequence_root_summary.json"
        ok = _is_nonempty_file(summary_path)
        return {
            "ok": ok,
            "summary_path": str(summary_path),
            "missing": [] if ok else [str(summary_path)],
        }

    seed_root = Path(str(item["artifact_root"])).parents[1]
    summary_path = seed_root / "summary.json"
    run_info_path = seed_root / "meta" / "run_info.json"
    missing: list[str] = []
    if not _is_nonempty_file(summary_path):
        missing.append(str(summary_path))
    status = ""
    if not _is_nonempty_file(run_info_path):
        missing.append(str(run_info_path))
    else:
        try:
            status = str(_read_json(run_info_path).get("status", ""))
        except Exception as exc:
            missing.append(f"{run_info_path}: {exc}")
    if status and status != "success":
        missing.append(f"{run_info_path}: status={status}")
    return {
        "ok": not missing,
        "seed_root": str(seed_root),
        "summary_path": str(summary_path),
        "run_info_path": str(run_info_path),
        "run_info_status": status,
        "missing": missing,
    }


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _is_nonempty_csv(path: Path) -> bool:
    if not _is_nonempty_file(path):
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            rows = 0
            for row in reader:
                if any(str(cell).strip() for cell in row):
                    rows += 1
                if rows > 1:
                    return True
    except Exception:
        return False
    return False


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
    _append_common_flags(command, args, figure=figure)
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _display_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
