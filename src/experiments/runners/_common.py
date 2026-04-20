from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.experiments.catalog import ExperimentSpec, get_experiment_spec
from src.pipelines.common import candidate_has_required_modules, discover_python_candidates, python_has_required_modules

REQUIRED_RUNTIME_MODULES = ("torch", "numpy", "pandas", "matplotlib", "scipy", "sklearn", "tqdm")


def build_runner_parser(spec: ExperimentSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Computation-only runner for {spec.title}.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--dataset-root", type=str, default="MNIST")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
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


def normalize_result_bundle(output_dir: Path, spec: ExperimentSpec) -> None:
    _ensure_dir(output_dir)
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
    _ensure_dir(output_dir / "data")
    _ensure_dir(output_dir / "arrays")
    logs_dir = _ensure_dir(output_dir / "logs")
    legacy_log_dir = output_dir / "log"
    if legacy_log_dir.exists():
        _copy_tree_if_present(legacy_log_dir, logs_dir)
    legacy_figure_dir = output_dir / "figure"
    if legacy_figure_dir.exists():
        shutil.rmtree(legacy_figure_dir)
    if not any(logs_dir.iterdir()):
        (logs_dir / "run.log").write_text("", encoding="utf-8")
    if not (output_dir / "summary.json").exists():
        raise FileNotFoundError(f"{spec.experiment_id}: summary.json missing in {output_dir}")
    if not (output_dir / "run_config.json").exists():
        raise FileNotFoundError(f"{spec.experiment_id}: run_config.json missing in {output_dir}")
    manifest = {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "files": _relativize_files(output_dir),
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(_to_json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


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
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    log_path = output_dir / "runner.log"
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
    return 0


def main_for(experiment_id: str) -> int:
    spec = get_experiment_spec(experiment_id)
    parser = build_runner_parser(spec)
    args = parser.parse_args()
    return run_legacy_experiment(spec, args)


__all__ = ["build_runner_parser", "main_for", "normalize_result_bundle", "run_legacy_experiment"]
