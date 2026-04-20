from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import io
import logging
import os
import runpy
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.common import candidate_has_required_modules as shared_candidate_has_required_modules
from src.pipelines.common import discover_python_candidates as shared_discover_python_candidates
from src.pipelines.common import python_has_required_modules as shared_python_has_required_modules

REQUIRED_RUNTIME_MODULES = ("torch", "numpy", "pandas", "matplotlib", "scipy", "sklearn", "tqdm")
REQUIRED_PUBLICATION_THRESHOLDS = {
    "PUBLICATION_TITLE_FONT_SIZE": 20,
    "PUBLICATION_AXIS_LABEL_FONT_SIZE": 18,
    "PUBLICATION_TICK_LABEL_FONT_SIZE": 16,
    "PUBLICATION_LEGEND_FONT_SIZE": 16,
    "PUBLICATION_ANNOTATION_FONT_SIZE": 16,
    "PUBLICATION_LINE_WIDTH": 2.5,
    "PUBLICATION_MARKER_SIZE": 8.0,
    "PUBLICATION_ERRORBAR_CAPSIZE": 4.0,
}
REQUIRED_DPI = 300


@dataclass(frozen=True)
class FigureTask:
    name: str
    script_path: Path
    save_dir: Path
    argv: tuple[str, ...]
    expected_exports: tuple[str, ...] = ("figure_main.png", "figure_main.pdf", "figure_main.svg")
    required_inputs: tuple[Path, ...] = ()


def current_python_has_required_modules() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(module_name) is not None for module_name in REQUIRED_RUNTIME_MODULES)


def discover_python_candidates(current_python: Path) -> list[Path]:
    candidates: list[Path] = [current_python]
    for parent in current_python.parents:
        env_root = parent / "envs"
        if not env_root.exists():
            continue
        for python_path in sorted(env_root.glob("*/python.exe")):
            resolved = python_path.resolve()
            if resolved not in candidates:
                candidates.append(resolved)
    return candidates


def candidate_has_required_modules(python_path: Path) -> bool:
    probe = (
        "import importlib.util; "
        f"mods={list(REQUIRED_RUNTIME_MODULES)!r}; "
        "print(int(all(importlib.util.find_spec(m) is not None for m in mods)))"
    )
    completed = subprocess.run(
        [str(python_path), "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "1"


def maybe_reexec_with_compatible_python() -> int | None:
    if os.environ.get("FIGURE_PIPELINE_REEXEC") == "1":
        return None
    if current_python_has_required_modules():
        return None

    current_python = Path(sys.executable).resolve()
    for candidate in discover_python_candidates(current_python):
        if candidate == current_python:
            continue
        if not candidate_has_required_modules(candidate):
            continue

        print(f"Re-executing figure pipeline with compatible interpreter: {candidate}")
        env = os.environ.copy()
        env["FIGURE_PIPELINE_REEXEC"] = "1"
        completed = subprocess.run(
            [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            check=False,
        )
        return int(completed.returncode)

    print("No compatible Python interpreter with required modules was found for the figure pipeline.")
    return 1


def current_python_has_required_modules() -> bool:
    return shared_python_has_required_modules(REQUIRED_RUNTIME_MODULES)


def discover_python_candidates(current_python: Path) -> list[Path]:
    return shared_discover_python_candidates(current_python)


def candidate_has_required_modules(python_path: Path) -> bool:
    return shared_candidate_has_required_modules(python_path, REQUIRED_RUNTIME_MODULES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full publication-figure generation pipeline.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--log-path", type=str, default="figure_generation_pipeline.log")
    parser.add_argument("--dry-run", action="store_true", help="Log planned actions without executing figure scripts.")
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("figure_generation_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def resolve_path(workspace_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


def locate_existing_path(preferred_path: Path, workspace_root: Path) -> Path:
    if preferred_path.exists():
        return preferred_path

    matches = sorted(path.resolve() for path in workspace_root.rglob(preferred_path.name))
    if len(matches) == 1:
        return matches[0]
    return preferred_path


def configure_matplotlib_runtime(workspace_root: Path, logger: logging.Logger) -> None:
    mpl_config_dir = workspace_root / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    plt.close("all")
    mpl.rcParams["font.family"] = ["DejaVu Sans"]
    logger.info("Configured matplotlib headless runtime with Agg backend and DejaVu Sans.")


def enforce_publication_thresholds(workspace_root: Path, logger: logging.Logger) -> Dict[str, float]:
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    figure_utils_common = importlib.import_module("figure_utils_common")
    overrides_applied: Dict[str, float] = {}
    for attr_name, minimum in REQUIRED_PUBLICATION_THRESHOLDS.items():
        current = float(getattr(figure_utils_common, attr_name, 0.0))
        if current < minimum:
            setattr(figure_utils_common, attr_name, minimum)
            overrides_applied[attr_name] = minimum

    figure_utils_common.apply_publication_style()

    import matplotlib as mpl

    if float(mpl.rcParams["figure.dpi"]) < REQUIRED_DPI:
        mpl.rcParams["figure.dpi"] = REQUIRED_DPI
        overrides_applied["figure.dpi"] = REQUIRED_DPI
    if float(mpl.rcParams["savefig.dpi"]) < REQUIRED_DPI:
        mpl.rcParams["savefig.dpi"] = REQUIRED_DPI
        overrides_applied["savefig.dpi"] = REQUIRED_DPI

    threshold_report = {
        "title_font_size": float(getattr(figure_utils_common, "PUBLICATION_TITLE_FONT_SIZE", 0.0)),
        "axis_label_font_size": float(getattr(figure_utils_common, "PUBLICATION_AXIS_LABEL_FONT_SIZE", 0.0)),
        "tick_label_font_size": float(getattr(figure_utils_common, "PUBLICATION_TICK_LABEL_FONT_SIZE", 0.0)),
        "legend_font_size": float(getattr(figure_utils_common, "PUBLICATION_LEGEND_FONT_SIZE", 0.0)),
        "annotation_font_size": float(getattr(figure_utils_common, "PUBLICATION_ANNOTATION_FONT_SIZE", 0.0)),
        "figure_dpi": float(mpl.rcParams["figure.dpi"]),
        "savefig_dpi": float(mpl.rcParams["savefig.dpi"]),
    }

    if overrides_applied:
        logger.info("Applied publication overrides: %s", overrides_applied)
    logger.info("Publication threshold check: %s", threshold_report)
    return threshold_report


def build_tasks(workspace_root: Path, model_path: Path, dataset_root: Path, results_root: Path) -> list[FigureTask]:
    paper_dir = (results_root / "paper_figures").resolve()

    return [
        FigureTask(
            name="Fig1",
            script_path=(workspace_root / "figure1.py").resolve(),
            save_dir=paper_dir,
            argv=(
                "--model-path",
                str(model_path),
                "--dataset-root",
                str(dataset_root),
                "--results-root",
                str(results_root),
                "--output-dir",
                str(paper_dir),
            ),
            expected_exports=("figure1.png", "figure1.pdf", "figure1.svg"),
        ),
        FigureTask(
            name="Fig2",
            script_path=(workspace_root / "figure2.py").resolve(),
            save_dir=paper_dir,
            argv=(
                "--model-path",
                str(model_path),
                "--results-root",
                str(results_root),
                "--output-dir",
                str(paper_dir),
            ),
            expected_exports=("figure2.png", "figure2.pdf", "figure2.svg"),
        ),
        FigureTask(
            name="Fig3",
            script_path=(workspace_root / "figure3.py").resolve(),
            save_dir=paper_dir,
            argv=(
                "--model-path",
                str(model_path),
                "--results-root",
                str(results_root),
                "--output-dir",
                str(paper_dir),
            ),
            expected_exports=("figure3.png", "figure3.pdf", "figure3.svg"),
        ),
        FigureTask(
            name="Fig4",
            script_path=(workspace_root / "figure4.py").resolve(),
            save_dir=paper_dir,
            argv=(
                "--model-path",
                str(model_path),
                "--dataset-root",
                str(dataset_root),
                "--results-root",
                str(results_root),
                "--output-dir",
                str(paper_dir),
            ),
            expected_exports=("figure4.png", "figure4.pdf", "figure4.svg"),
        ),
        FigureTask(
            name="Fig5",
            script_path=(workspace_root / "figure5.py").resolve(),
            save_dir=paper_dir,
            argv=(
                "--model-path",
                str(model_path),
                "--results-root",
                str(results_root),
                "--output-dir",
                str(paper_dir),
            ),
            expected_exports=("figure5.png", "figure5.pdf", "figure5.svg"),
        ),
    ]


def validate_exports(task: FigureTask) -> list[Path]:
    return [task.save_dir / name for name in task.expected_exports if not (task.save_dir / name).exists()]


def run_script_once(task: FigureTask, workspace_root: Path) -> str:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    old_argv = list(sys.argv)
    old_cwd = Path.cwd()

    try:
        os.chdir(workspace_root)
        sys.argv = [str(task.script_path), *task.argv]
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                runpy.run_path(str(task.script_path), run_name="__main__")
            except SystemExit as exc:
                exit_code = 0 if exc.code is None else int(exc.code)
                if exit_code != 0:
                    raise RuntimeError(f"{task.name} exited with status {exit_code}") from exc
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv

    return _format_captured_output(stdout_buffer.getvalue(), stderr_buffer.getvalue())


def _format_captured_output(stdout_text: str, stderr_text: str) -> str:
    parts: list[str] = []
    if stdout_text.strip():
        parts.append("[stdout]\n" + stdout_text.strip())
    if stderr_text.strip():
        parts.append("[stderr]\n" + stderr_text.strip())
    return "\n\n".join(parts)


def cleanup_runtime() -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    gc.collect()


def repair_missing_required_input(path: Path, workspace_root: Path, logger: logging.Logger) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return True

    candidates = sorted(
        candidate.resolve()
        for candidate in workspace_root.rglob(path.name)
        if candidate.resolve() != path and candidate.is_file()
    )
    if candidates:
        shutil.copy2(candidates[0], path)
        logger.info("Restored missing intermediate file %s from %s", path, candidates[0])
        return True

    if path.name == "engram_decode_metrics.csv":
        helper_script = workspace_root / "engram_decode.py"
        if helper_script.exists():
            logger.info("Attempting to regenerate %s via %s", path.name, helper_script.name)
            helper_stdout = io.StringIO()
            helper_stderr = io.StringIO()
            old_cwd = Path.cwd()
            try:
                os.chdir(workspace_root)
                with contextlib.redirect_stdout(helper_stdout), contextlib.redirect_stderr(helper_stderr):
                    runpy.run_path(str(helper_script), run_name="__main__")
            except Exception as exc:
                logger.error("Automatic engram decode regeneration failed: %s", exc)
                logger.info(_format_captured_output(helper_stdout.getvalue(), helper_stderr.getvalue()))
            finally:
                os.chdir(old_cwd)
            if path.exists():
                logger.info("Regenerated missing intermediate file %s via %s", path, helper_script.name)
                return True

    return False


def apply_common_repairs(task: FigureTask, workspace_root: Path, logger: logging.Logger) -> None:
    task.save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Ensured output directory exists: %s", task.save_dir)
    configure_matplotlib_runtime(workspace_root, logger)
    enforce_publication_thresholds(workspace_root, logger)

    for required_input in task.required_inputs:
        if not required_input.exists():
            repaired = repair_missing_required_input(required_input, workspace_root, logger)
            if not repaired:
                logger.warning("Required intermediate file is still missing: %s", required_input)


def log_output_paths(task: FigureTask, logger: logging.Logger) -> None:
    for export_name in task.expected_exports:
        export_path = (task.save_dir / export_name).resolve()
        logger.info("%s output: %s", task.name, export_path)


def execute_task(task: FigureTask, workspace_root: Path, logger: logging.Logger, dry_run: bool = False) -> bool:
    logger.info("Starting %s with script %s", task.name, task.script_path)
    logger.info("%s save directory: %s", task.name, task.save_dir)
    apply_common_repairs(task, workspace_root, logger)

    if dry_run:
        logger.info("Dry run enabled; skipped execution for %s", task.name)
        return True

    for attempt_index in range(1, 3):
        try:
            captured_output = run_script_once(task, workspace_root)
            if captured_output:
                logger.info("%s runtime output:\n%s", task.name, captured_output)
        except Exception:
            error_text = traceback.format_exc()
            logger.error("%s failed on attempt %d.\n%s", task.name, attempt_index, error_text)
            if attempt_index == 1:
                logger.info("Applying automatic repairs before retrying %s", task.name)
                apply_common_repairs(task, workspace_root, logger)
                cleanup_runtime()
                continue
            print(f"{task.name} failed after retry. See figure_generation_pipeline.log for diagnostics.")
            return False

        missing_exports = validate_exports(task)
        if not missing_exports:
            logger.info("%s completed successfully.", task.name)
            log_output_paths(task, logger)
            cleanup_runtime()
            return True

        logger.error("%s is missing exports after attempt %d: %s", task.name, attempt_index, missing_exports)
        if attempt_index == 1:
            logger.info("Re-running %s to regenerate missing figure exports.", task.name)
            apply_common_repairs(task, workspace_root, logger)
            cleanup_runtime()
            continue

        print(f"{task.name} failed after retry. See figure_generation_pipeline.log for diagnostics.")
        return False

    return False


def validate_prerequisites(model_path: Path, dataset_root: Path, logger: logging.Logger) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    logger.info("Validated prerequisite paths: model=%s | dataset=%s", model_path, dataset_root)


def main() -> int:
    reexec_exit_code = maybe_reexec_with_compatible_python()
    if reexec_exit_code is not None:
        return reexec_exit_code

    args = parse_args()
    workspace_root = Path(__file__).resolve().parents[2]
    log_path = resolve_path(workspace_root, args.log_path)
    logger = setup_logging(log_path)

    logger.info("Figure pipeline workspace: %s", workspace_root)
    configure_matplotlib_runtime(workspace_root, logger)

    model_path = locate_existing_path(resolve_path(workspace_root, args.model_path), workspace_root)
    dataset_root = locate_existing_path(resolve_path(workspace_root, args.dataset_root), workspace_root)
    results_root = resolve_path(workspace_root, args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    try:
        validate_prerequisites(model_path, dataset_root, logger)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        print(str(exc))
        return 1

    tasks = build_tasks(
        workspace_root=workspace_root,
        model_path=model_path,
        dataset_root=dataset_root,
        results_root=results_root,
    )

    logger.info("Execution order: %s", " -> ".join(task.name for task in tasks))
    for task in tasks:
        success = execute_task(task, workspace_root=workspace_root, logger=logger, dry_run=bool(args.dry_run))
        if not success:
            logger.error("Pipeline halted at %s", task.name)
            return 1

    if args.dry_run:
        message = "Dry run completed. Figure pipeline order and publication checks passed."
    else:
        message = "All figures generated successfully with publication-quality parameters."
    logger.info(message)
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
