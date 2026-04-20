from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def safe_json_float(value: object) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        return None
    return scalar


def to_json_ready(value):
    if isinstance(value, dict):
        return {str(key): to_json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [to_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return safe_json_float(value)
    return value


def save_json_payload(payload: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_json_ready(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
