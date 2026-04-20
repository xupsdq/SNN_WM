from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Iterable


def python_has_required_modules(required_modules: Iterable[str]) -> bool:
    return all(importlib.util.find_spec(module_name) is not None for module_name in required_modules)


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


def candidate_has_required_modules(python_path: Path, required_modules: Iterable[str]) -> bool:
    probe = (
        "import importlib.util; "
        f"mods={list(required_modules)!r}; "
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

