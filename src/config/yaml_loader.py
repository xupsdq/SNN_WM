from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def load_yaml_file(path: str | Path) -> Mapping[str, Any]:
    yaml_path = Path(path)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised when dependency is missing
        raise RuntimeError("YAML config support requires PyYAML. Install it before using --config.") from exc
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {yaml_path}")
    return payload


def nested_get(mapping: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


__all__ = ["load_yaml_file", "nested_get"]
