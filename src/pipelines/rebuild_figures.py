from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.pipelines.common import candidate_has_required_modules as shared_candidate_has_required_modules
from src.pipelines.common import discover_python_candidates as shared_discover_python_candidates
from src.pipelines.common import python_has_required_modules as shared_python_has_required_modules

REQUIRED_MODULES = ("torch", "numpy", "pandas", "matplotlib", "scipy", "sklearn", "tqdm")


def _current_python_has_required_modules() -> bool:
    return shared_python_has_required_modules(REQUIRED_MODULES)


def _discover_python_candidates(current_python: Path) -> list[Path]:
    return shared_discover_python_candidates(current_python)


def _candidate_has_required_modules(python_path: Path) -> bool:
    return shared_candidate_has_required_modules(python_path, REQUIRED_MODULES)


def _maybe_reexec_with_compatible_python() -> int | None:
    if os.environ.get("PAPER_FIGURES_REEXEC") == "1":
        return None
    if _current_python_has_required_modules():
        return None

    current_python = Path(sys.executable).resolve()
    for candidate in _discover_python_candidates(current_python):
        if candidate == current_python:
            continue
        if not _candidate_has_required_modules(candidate):
            continue

        print(f"Re-executing figure rebuild with compatible interpreter: {candidate}")
        env = os.environ.copy()
        env["PAPER_FIGURES_REEXEC"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        completed = subprocess.run(
            [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            check=False,
        )
        return int(completed.returncode)

    print("No compatible Python interpreter with required modules was found.")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild final Figures 1-5 from result records.")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/paper_figures")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--dataset-root", type=str, default="MNIST")
    return parser.parse_args()


def main() -> int:
    reexec_exit_code = _maybe_reexec_with_compatible_python()
    if reexec_exit_code is not None:
        return reexec_exit_code

    from src.plotting.final_figures import save_figure

    args = parse_args()
    workspace_root = Path(__file__).resolve().parents[2]
    results_root = (workspace_root / args.results_root).resolve()
    output_dir = (workspace_root / args.output_dir).resolve()
    model_path = (workspace_root / args.model_path).resolve()
    dataset_root = (workspace_root / args.dataset_root).resolve()

    for figure_id in ("figure1", "figure2", "figure3", "figure4", "figure5"):
        save_figure(
            figure_id,
            results_root=results_root,
            output_dir=output_dir,
            model_path=model_path,
            dataset_root=dataset_root,
        )
        print(f"Saved {figure_id} to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
