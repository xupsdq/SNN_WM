from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.plotting.paper_fig.layout_contract import validate_layout_contract


SPEC_ROOT = Path(__file__).resolve().parents[1] / "specs" / "supplementary_v5"


def load_spec(figure_id: str) -> dict[str, Any]:
    normalized = str(figure_id).lower()
    path = SPEC_ROOT / f"{normalized}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict):
        raise ValueError(f"Supplementary visual spec must be an object: {path}")
    report = validate_layout_contract(spec)
    if not report.ok:
        raise ValueError(f"Invalid layout contract in {path}: {report.failures}")
    spec["_spec_path"] = str(path)
    spec["_layout_validation"] = {"passes": report.passes, "warnings": report.warnings}
    return spec


__all__ = ["SPEC_ROOT", "load_spec"]
