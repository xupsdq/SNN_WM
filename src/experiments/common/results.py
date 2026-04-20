from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.config.snapshots import save_config_snapshot


@dataclass(frozen=True)
class ResultLayout:
    root: Path
    figures_dir: Path
    data_dir: Path
    logs_dir: Path
    metrics_dir: Path
    meta_dir: Path

    @property
    def figure_dir(self) -> Path:
        return self.figures_dir

    @property
    def log_dir(self) -> Path:
        return self.logs_dir

    def figure_base(self, stem: str) -> Path:
        return self.figures_dir / stem

    def data_file(self, filename: str) -> Path:
        return self.data_dir / filename

    def metrics_file(self, filename: str) -> Path:
        return self.metrics_dir / filename

    def meta_file(self, filename: str) -> Path:
        return self.meta_dir / filename

    def root_file(self, filename: str) -> Path:
        return self.root / filename

    def log_file(self, filename: str = "run.log") -> Path:
        return self.logs_dir / filename


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def prepare_result_layout(save_dir: str | Path) -> ResultLayout:
    root = ensure_dir(save_dir)
    figures_dir = ensure_dir(root / "figures")
    data_dir = ensure_dir(root / "data")
    logs_dir = ensure_dir(root / "logs")
    metrics_dir = ensure_dir(root / "metrics")
    meta_dir = ensure_dir(root / "meta")
    return ResultLayout(
        root=root,
        figures_dir=figures_dir,
        data_dir=data_dir,
        logs_dir=logs_dir,
        metrics_dir=metrics_dir,
        meta_dir=meta_dir,
    )


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
