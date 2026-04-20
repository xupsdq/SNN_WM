from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PaperFigureLayout:
    root: Path
    data_dir: Path
    arrays_dir: Path
    logs_dir: Path
    staging_dir: Path

    def data_file(self, filename: str) -> Path:
        return self.data_dir / filename

    def array_file(self, filename: str) -> Path:
        return self.arrays_dir / filename

    def log_file(self, filename: str = "run.log") -> Path:
        return self.logs_dir / filename

    def root_file(self, filename: str) -> Path:
        return self.root / filename

    def staging_path(self, name: str) -> Path:
        path = self.staging_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def prepare_layout(root: str | Path) -> PaperFigureLayout:
    root_path = ensure_dir(root)
    return PaperFigureLayout(
        root=root_path,
        data_dir=ensure_dir(root_path / "data"),
        arrays_dir=ensure_dir(root_path / "arrays"),
        logs_dir=ensure_dir(root_path / "logs"),
        staging_dir=ensure_dir(root_path / "_staging"),
    )


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def save_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_to_json_safe(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out_path


def save_csv(df: pd.DataFrame, path: str | Path, sort_by: str | list[str] | tuple[str, ...] | None = None) -> Path:
    out_df = df.copy()
    if sort_by is not None:
        columns = [sort_by] if isinstance(sort_by, str) else list(sort_by)
        out_df = out_df.sort_values(by=columns, kind="stable").reset_index(drop=True)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def save_npz(path: str | Path, **arrays: Any) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path


def copy_artifact(src: str | Path, dst: str | Path) -> Path:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return dst_path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_artifact_manifest(layout: PaperFigureLayout, artifacts: Mapping[str, Any]) -> Path:
    return save_json({"artifacts": _to_json_safe(dict(artifacts))}, layout.root_file("artifact_manifest.json"))


__all__ = [
    "PaperFigureLayout",
    "copy_artifact",
    "ensure_dir",
    "load_json",
    "prepare_layout",
    "save_csv",
    "save_json",
    "save_npz",
    "write_artifact_manifest",
]

