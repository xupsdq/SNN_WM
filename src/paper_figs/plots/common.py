from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def require_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Required artifact not found: {path_obj}")
    return path_obj


def read_csv_validated(path: str | Path, required_columns: Sequence[str]) -> pd.DataFrame:
    csv_path = require_path(path)
    df = pd.read_csv(csv_path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
    return df


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    npz_path = require_path(path)
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_figure_input_dir(figure_id: str, input_dir: str | Path | None = None) -> Path:
    if input_dir is not None:
        return require_path(input_dir)
    results_root = project_root() / "results"
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    candidates: list[Path] = []
    for candidate in results_root.rglob(figure_id):
        if not candidate.is_dir():
            continue
        if not (candidate / "summary.json").exists():
            continue
        if not (candidate / "data").is_dir():
            continue
        candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"No result bundle found automatically for {figure_id} under {results_root}")
    candidates.sort(key=lambda item: (item / "summary.json").stat().st_mtime, reverse=True)
    return candidates[0]


__all__ = ["load_npz", "project_root", "read_csv_validated", "require_path", "resolve_figure_input_dir"]
