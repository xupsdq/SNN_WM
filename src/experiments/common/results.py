from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.config.snapshots import save_config_snapshot


@dataclass(frozen=True)
class ResultLayout:
    root: Path
    figure_dir: Path
    data_dir: Path
    log_dir: Path

    def figure_base(self, stem: str) -> Path:
        return self.figure_dir / stem

    def data_file(self, filename: str) -> Path:
        return self.data_dir / filename

    def root_file(self, filename: str) -> Path:
        return self.root / filename

    def log_file(self, filename: str = "run.log") -> Path:
        return self.log_dir / filename


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def prepare_result_layout(save_dir: str | Path) -> ResultLayout:
    root = ensure_dir(save_dir)
    figure_dir = ensure_dir(root / "figure")
    data_dir = ensure_dir(root / "data")
    log_dir = ensure_dir(root / "log")
    return ResultLayout(root=root, figure_dir=figure_dir, data_dir=data_dir, log_dir=log_dir)


def save_run_config(config_dict: Any, save_dir: str | Path, filename: str = "run_config.json") -> Path:
    return save_config_snapshot(save_dir=save_dir, config=config_dict, filename=filename)


def save_summary_json(summary_dict: Mapping[str, Any], save_dir: str | Path, filename: str = "summary.json") -> Path:
    save_dir_path = ensure_dir(save_dir)
    out_path = save_dir_path / filename
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_dict, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return out_path


def save_log_lines(
    lines: Iterable[str],
    save_dir: str | Path,
    filename: str = "run.log",
) -> Path:
    save_dir_path = ensure_dir(save_dir)
    out_path = save_dir_path / filename
    text = "\n".join(str(line) for line in lines).rstrip() + "\n"
    out_path.write_text(text, encoding="utf-8")
    return out_path


__all__ = [
    "ResultLayout",
    "ensure_dir",
    "prepare_result_layout",
    "save_log_lines",
    "save_run_config",
    "save_summary_json",
]
