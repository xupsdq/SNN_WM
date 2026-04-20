from __future__ import annotations

import json
import platform as platform_lib
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.paths import DEFAULT_PATH_CONFIG

from .results import ensure_dir


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=DEFAULT_PATH_CONFIG.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def build_run_info(
    *,
    experiment_name: str,
    output_dir: str | Path,
    entry_script: str,
    seed: int | None,
    dataset: str | None,
    command: str | None,
    model_path: str | None = None,
    config_file: str | None = None,
    status: str = "running",
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "git_commit": _read_git_commit(),
        "seed": seed,
        "dataset": dataset or "",
        "output_dir": str(Path(output_dir).resolve()),
        "entry_script": entry_script,
        "started_at": started_at or _timestamp_now(),
        "finished_at": finished_at,
        "status": status,
        "python_version": sys.version.split()[0],
        "platform": platform_lib.platform(),
        "command": command,
        "hostname": socket.gethostname(),
        "model_path": model_path,
        "config_file": config_file,
    }


def write_run_info(meta_dir: str | Path, payload: dict[str, Any], filename: str = "run_info.json") -> Path:
    meta_path = ensure_dir(meta_dir)
    out_path = meta_path / filename
    out_path.write_text(
        json.dumps(_to_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return out_path


def finalize_run_info(
    meta_dir: str | Path,
    payload: dict[str, Any],
    *,
    status: str,
    filename: str = "run_info.json",
) -> Path:
    final_payload = dict(payload)
    final_payload["status"] = status
    final_payload["finished_at"] = _timestamp_now()
    return write_run_info(meta_dir, final_payload, filename=filename)


__all__ = ["build_run_info", "finalize_run_info", "write_run_info"]
