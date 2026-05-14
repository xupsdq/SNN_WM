from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from matplotlib.figure import Figure


MM_PER_INCH = 25.4


def mm_to_inch(mm: float) -> float:
    """Convert millimeters to inches."""
    return float(mm) / MM_PER_INCH


def add_axes_mm(
    fig: Figure,
    x_mm: float,
    y_mm: float,
    w_mm: float,
    h_mm: float,
    canvas_h_mm: float,
    canvas_w_mm: float | None = None,
):
    """Add axes using top-left millimeter coordinates from the figure spec."""
    width_in, height_in = fig.get_size_inches()
    if canvas_w_mm is None:
        canvas_w_mm = width_in * MM_PER_INCH
    left = float(x_mm) / float(canvas_w_mm)
    bottom = (float(canvas_h_mm) - float(y_mm) - float(h_mm)) / float(canvas_h_mm)
    width = float(w_mm) / float(canvas_w_mm)
    height = float(h_mm) / float(canvas_h_mm)
    return fig.add_axes([left, bottom, width, height])


def paper_fig_root() -> Path:
    """Return the package root directory."""
    return Path(__file__).resolve().parent


def repo_root_from_here() -> Path:
    """Find the repository root by walking up from this file."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists() and (parent / "src").is_dir():
            return parent
    return current.parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping with a clear error message."""
    yaml_path = Path(path)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("paper_fig YAML specs require PyYAML.") from exc
    try:
        with yaml_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception as exc:
        raise RuntimeError(f"Failed to parse YAML spec {yaml_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"YAML spec must contain a mapping: {yaml_path}")
    return payload


def write_yaml(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a YAML mapping."""
    yaml_path = Path(path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("paper_fig YAML export requires PyYAML.") from exc
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False, allow_unicode=True)
    return yaml_path


def write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write JSON with stable formatting."""
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
    return json_path


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object."""
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {json_path}")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value

