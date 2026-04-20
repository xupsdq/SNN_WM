from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def _to_json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _to_json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    return value


def save_config_snapshot(save_dir: str | Path, config: Any, filename: str = "config_snapshot.json") -> Path:
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = save_dir_path / filename
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_json_safe(config), handle, indent=2, ensure_ascii=False, sort_keys=True)
    return out_path

