from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.pipelines.common import candidate_has_required_modules as shared_candidate_has_required_modules
from src.pipelines.common import discover_python_candidates as shared_discover_python_candidates
from src.pipelines.common import python_has_required_modules as shared_python_has_required_modules

REQUIRED_MODULES = ("torch", "numpy", "pandas", "matplotlib", "scipy", "sklearn", "tqdm")


def run_figure_cli(figure_id: str) -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    reexec_exit_code = _maybe_reexec_with_compatible_python(figure_id)
    if reexec_exit_code is not None:
        raise SystemExit(reexec_exit_code)

    from src.plotting.final_figures import (
        main_figure1,
        main_figure2,
        main_figure3,
        main_figure4,
        main_figure5,
    )

    main_map = {
        "figure1": main_figure1,
        "figure2": main_figure2,
        "figure3": main_figure3,
        "figure4": main_figure4,
        "figure5": main_figure5,
    }
    main_map[figure_id]()


def _maybe_reexec_with_compatible_python(figure_id: str) -> int | None:
    if os.environ.get("PAPER_FIGURE_SINGLE_REEXEC") == "1":
        return None
    if shared_python_has_required_modules(REQUIRED_MODULES):
        return None

    repo_root = Path(__file__).resolve().parents[2]
    current_python = Path(sys.executable).resolve()
    for candidate in shared_discover_python_candidates(current_python):
        if candidate == current_python:
            continue
        if not shared_candidate_has_required_modules(candidate, REQUIRED_MODULES):
            continue

        env = os.environ.copy()
        env["PAPER_FIGURE_SINGLE_REEXEC"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        completed = subprocess.run(
            [str(candidate), str(repo_root / f"{figure_id}.py"), *sys.argv[1:]],
            cwd=str(repo_root),
            env=env,
            check=False,
        )
        return int(completed.returncode)
    return None
