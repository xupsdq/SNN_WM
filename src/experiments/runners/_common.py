from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS
from src.config.yaml_loader import load_yaml_file, nested_get
from src.experiments.common.results import prepare_result_layout
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.catalog import ExperimentSpec, get_experiment_spec
from src.pipelines.common import candidate_has_required_modules, discover_python_candidates, python_has_required_modules

REQUIRED_RUNTIME_MODULES = ("torch", "numpy", "pandas", "matplotlib", "scipy", "sklearn", "tqdm")


def build_runner_parser(spec: ExperimentSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Computation-only runner for {spec.title}.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _relativize_files(root: Path) -> list[str]:
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
    return files


def _copy_tree_if_present(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    _ensure_dir(dst)
    for path in src.rglob("*"):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _copy_file_if_present(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_metrics_candidates(output_dir: Path, spec: ExperimentSpec) -> None:
    metrics_dir = _ensure_dir(output_dir / "metrics")
    primary_name = Path(spec.primary_csv).name if spec.primary_csv else None
    for base_dir in (output_dir, output_dir / "data"):
        if not base_dir.exists():
            continue
        for path in sorted(base_dir.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in {".csv", ".json"}:
                continue
            name_lower = path.name.lower()
            should_copy = (
                name_lower.startswith("metrics")
                or "summary" in name_lower
                or path.name == primary_name
            )
            if should_copy:
                _copy_file_if_present(path, metrics_dir / path.name)


def _copy_meta_snapshots(output_dir: Path) -> None:
    meta_dir = _ensure_dir(output_dir / "meta")
    for source_name, target_name in (
        ("run_config.json", "run_config.snapshot.json"),
        ("artifact_manifest.json", "artifact_manifest.snapshot.json"),
    ):
        _copy_file_if_present(output_dir / source_name, meta_dir / target_name)


def normalize_result_bundle(output_dir: Path, spec: ExperimentSpec) -> None:
    layout = prepare_result_layout(output_dir)
    nested_smoke_dir = output_dir / "smoke"
    if nested_smoke_dir.is_dir() and (nested_smoke_dir / "summary.json").exists():
        for path in nested_smoke_dir.iterdir():
            target = output_dir / path.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(path), str(target))
        nested_smoke_dir.rmdir()
    _ensure_dir(layout.data_dir)
    _ensure_dir(output_dir / "arrays")
    logs_dir = _ensure_dir(layout.logs_dir)
    legacy_log_dir = output_dir / "log"
    if legacy_log_dir.exists():
        _copy_tree_if_present(legacy_log_dir, logs_dir)
    _copy_file_if_present(output_dir / "runner.log", logs_dir / "runner.log")
    figures_dir = _ensure_dir(layout.figures_dir)
    legacy_figure_dir = output_dir / "figure"
    if legacy_figure_dir.exists():
        _copy_tree_if_present(legacy_figure_dir, figures_dir)
    if not any(logs_dir.iterdir()):
        (logs_dir / "run.log").write_text("", encoding="utf-8")
    if not (output_dir / "summary.json").exists():
        raise FileNotFoundError(f"{spec.experiment_id}: summary.json missing in {output_dir}")
    if not (output_dir / "run_config.json").exists():
        raise FileNotFoundError(f"{spec.experiment_id}: run_config.json missing in {output_dir}")
    _copy_metrics_candidates(output_dir, spec)
    _copy_meta_snapshots(output_dir)
    manifest = {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "files": _relativize_files(output_dir),
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(_to_json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _resolve_from_config(config: dict[str, Any] | None, *path: str, default: Any = None) -> Any:
    if not config:
        return default
    value = nested_get(config, *path, default=None)
    if value is not None:
        return value
    if len(path) == 1:
        return config.get(path[0], default)
    return default


def _resolve_path_value(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((DEFAULT_PROJECT_DEFAULTS.paths.repo_root / path).resolve())


def _apply_runner_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config_payload = load_yaml_file(args.config) if args.config else {}
    defaults = DEFAULT_PROJECT_DEFAULTS
    args.output_dir = args.output_dir or _resolve_from_config(config_payload, "output_dir")
    args.model_path = args.model_path or _resolve_from_config(
        config_payload,
        "model",
        "path",
        default=_resolve_from_config(config_payload, "model_path"),
    )
    args.dataset_root = args.dataset_root or _resolve_from_config(
        config_payload,
        "data",
        "dataset_root",
        default=_resolve_from_config(config_payload, "dataset_root"),
    )
    args.device = args.device or _resolve_from_config(
        config_payload,
        "runtime",
        "device",
        default=_resolve_from_config(config_payload, "device"),
    )
    args.seed = args.seed if args.seed is not None else _resolve_from_config(
        config_payload,
        "runtime",
        "seed",
        default=_resolve_from_config(config_payload, "seed"),
    )
    args.output_dir = _resolve_path_value(args.output_dir) if args.output_dir else None
    args.model_path = _resolve_path_value(args.model_path) if args.model_path else str(defaults.paths.model_path)
    args.dataset_root = _resolve_path_value(args.dataset_root) if args.dataset_root else str(defaults.paths.dataset_root)
    args.device = args.device or defaults.runtime.device
    args.seed = int(args.seed if args.seed is not None else defaults.runtime.seed)
    if not args.output_dir:
        raise SystemExit("--output-dir is required (or provide it via --config).")
    return args


def _resolve_runtime_python() -> Path:
    current_python = Path(sys.executable).resolve()
    if python_has_required_modules(REQUIRED_RUNTIME_MODULES):
        return current_python
    candidates = discover_python_candidates(current_python)
    preferred = sorted(
        candidates,
        key=lambda path: (
            0 if path.parent.name.lower() == "torch_env" else 1,
            0 if "torch" in str(path).lower() else 1,
            str(path).lower(),
        ),
    )
    for candidate in preferred:
        if candidate_has_required_modules(candidate, REQUIRED_RUNTIME_MODULES):
            return candidate
    raise RuntimeError("No compatible Python interpreter with required runtime modules was found.")


def run_legacy_experiment(spec: ExperimentSpec, args: argparse.Namespace) -> int:
    args = _apply_runner_config_defaults(args)
    output_dir = Path(args.output_dir).resolve()
    layout = prepare_result_layout(output_dir)
    runtime_python = _resolve_runtime_python()
    command = [str(runtime_python), "-m", spec.legacy_module, spec.output_flag, str(output_dir)]
    if spec.supports_model_path:
        command.extend(["--model-path", str(args.model_path)])
    if spec.supports_dataset_root:
        command.extend(["--dataset-root", str(args.dataset_root)])
    if spec.supports_device:
        command.extend(["--device", str(args.device)])
    if spec.supports_seed:
        command.extend(["--seed", str(int(args.seed))])
    if spec.supports_skip_figures:
        command.append("--skip-figures")
    if bool(args.smoke):
        command.extend(spec.smoke_args)
    run_info_payload = build_run_info(
        experiment_name=spec.experiment_id,
        output_dir=output_dir,
        entry_script=f"python -m src.experiments.runners.{spec.experiment_id}",
        seed=int(args.seed) if args.seed is not None else None,
        dataset=str(args.dataset_root),
        command=subprocess.list2cmdline(command),
        model_path=str(args.model_path),
        config_file=str(Path(args.config).resolve()) if args.config else None,
    )
    write_run_info(layout.meta_dir, run_info_payload)
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    log_path = layout.logs_dir / "runner.log"
    status = "failed"
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
        log_path.write_text(
            "\n".join(
                [
                    f"command={subprocess.list2cmdline(command)}",
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
        if completed.returncode != 0:
            raise RuntimeError(f"{spec.experiment_id} failed: see {log_path}")
        normalize_result_bundle(output_dir, spec)
        status = "success"
        return 0
    finally:
        finalize_run_info(layout.meta_dir, run_info_payload, status=status)


def main_for(experiment_id: str) -> int:
    spec = get_experiment_spec(experiment_id)
    parser = build_runner_parser(spec)
    args = parser.parse_args()
    return run_legacy_experiment(spec, args)


__all__ = ["build_runner_parser", "main_for", "normalize_result_bundle", "run_legacy_experiment"]
