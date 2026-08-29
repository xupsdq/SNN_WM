"""Public runtime and artifact-identity seam for successor-extension workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.experiments.paper_figures.fig2.fixed_b_substrate import (
    build_fixed_b_context,
    resolve_fixed_b_model_path,
)
from src.experiments.paper_figures.fig2.types import Fig2Config


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root() / path).resolve()


def seed_root(cfg: Any) -> Path:
    return resolve_repo_path(cfg.output_root) / f"seed_{int(cfg.network_seed)}"


def build_context(cfg: Any, *, load_model: bool) -> Any:
    model_path = resolve_fixed_b_model_path(
        None,
        str(cfg.model_path_glob),
        int(cfg.network_seed),
        smoke=bool(cfg.smoke),
    )
    fig_cfg = Fig2Config(
        model_path=str(model_path),
        dataset_root=str(resolve_repo_path(cfg.dataset_root)),
        output_root=str(resolve_repo_path(cfg.output_root)),
        network_seed=int(cfg.network_seed),
        device=str(cfg.device),
        smoke=False,
    )
    return build_fixed_b_context(fig_cfg, load_model=load_model)


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parent_entry(task_dir: str | Path) -> dict[str, Any]:
    task_dir = Path(task_dir)
    cache_path = task_dir / "cache_key.json"
    return {
        "path": str(task_dir.resolve()),
        "cache_key_sha256": sha256_file(cache_path) if cache_path.exists() else "missing",
    }


__all__ = [
    "build_context",
    "parent_entry",
    "repository_root",
    "resolve_repo_path",
    "seed_root",
    "sha256_file",
    "write_json",
]
