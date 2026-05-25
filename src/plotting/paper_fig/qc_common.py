from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.plotting.paper_fig.data_resolver import panel_output_paths
from src.plotting.paper_fig.utils import read_json


def _load_panel_data_map(output_dir: Path, figure_id: str, panels: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for panel_id, panel in panels.items():
        if panel.get("data_adapter") in (None, "", "none"):
            continue
        path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        if path.exists():
            out[panel_id] = pd.read_csv(path)
    return out


def _read_panel_data(output_dir: Path, figure_id: str, panel_id: str) -> pd.DataFrame:
    path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_panel_stats(output_dir: Path, figure_id: str, panel_id: str) -> dict[str, Any]:
    path = panel_output_paths(output_dir, figure_id, panel_id)["stats"]
    if not path.exists():
        return {}
    return read_json(path)


def _x(pos: Mapping[str, Any]) -> float:
    return float(pos.get("x", 0.0))


def _y(pos: Mapping[str, Any]) -> float:
    return float(pos.get("y", 0.0))


def _w(pos: Mapping[str, Any]) -> float:
    return float(pos.get("w", pos.get("width", 0.0)))


def _h(pos: Mapping[str, Any]) -> float:
    return float(pos.get("h", pos.get("height", 0.0)))


def _right(pos: Mapping[str, Any]) -> float:
    return _x(pos) + _w(pos)


def _bottom(pos: Mapping[str, Any]) -> float:
    return _y(pos) + _h(pos)


def _near(left: float, right: float, *, tol: float = 0.05) -> bool:
    return abs(float(left) - float(right)) <= tol


def _boxes_overlap(left: Any, right: Any) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != 4 or len(right) != 4:
        return False
    l0, b0, l1, t0 = [float(v) for v in left]
    r0, rb0, r1, rt0 = [float(v) for v in right]
    return l0 < r1 and l1 > r0 and b0 < rt0 and t0 > rb0


def _box_inside(inner: Any, outer: Any, *, tol: float = 0.003) -> bool:
    if not isinstance(inner, (list, tuple)) or not isinstance(outer, (list, tuple)) or len(inner) != 4 or len(outer) != 4:
        return False
    i0, ib0, i1, it0 = [float(v) for v in inner]
    o0, ob0, o1, ot0 = [float(v) for v in outer]
    return i0 >= o0 - tol and ib0 >= ob0 - tol and i1 <= o1 + tol and it0 <= ot0 + tol


def _box_in_upper_left(inner: Any, outer: Any) -> bool:
    if not _box_inside(inner, outer):
        return False
    i0, _ib0, _i1, it0 = [float(v) for v in inner]
    o0, ob0, _o1, ot0 = [float(v) for v in outer]
    width = max(_o1 - o0, 1e-9)
    height = max(ot0 - ob0, 1e-9)
    return i0 <= o0 + 0.20 * width and it0 >= ot0 - 0.20 * height


def _box_w(box: Any) -> float:
    return float(box[2]) - float(box[0]) if isinstance(box, (list, tuple)) and len(box) == 4 else 0.0


def _box_h(box: Any) -> float:
    return float(box[3]) - float(box[1]) if isinstance(box, (list, tuple)) and len(box) == 4 else 0.0


def _check_exports(
    figure_id: str,
    spec: Mapping[str, Any],
    output_dir: Path,
    full_export_paths: Mapping[str, str] | None,
    check_only: bool,
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    if check_only:
        warnings.append("check-only mode skipped full figure export checks")
    else:
        for ext in ("pdf", "svg", "png"):
            path = Path((full_export_paths or {}).get(ext, output_dir / f"{figure_id}.{ext}"))
            if path.exists():
                passes.append(f"full figure {ext} exists")
            else:
                failures.append(f"full figure {ext} missing")
    for filename in (f"{figure_id}_resolved_spec.yaml", f"{figure_id}_source_manifest.json"):
        if (output_dir / filename).exists():
            passes.append(f"{filename} exists")
        else:
            warnings.append(f"{filename} missing")
    manifest_path = output_dir / f"{figure_id}_source_manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            missing = [s for s in manifest.get("sources", []) if s.get("status") == "missing_source"]
            if missing:
                warnings.append(f"aggregate source manifest contains {len(missing)} missing source entries")
            else:
                passes.append("aggregate source manifest has no missing source entries")
        except Exception as exc:
            failures.append(f"source manifest unreadable: {exc}")
    legacy_named = [
        path.name
        for path in output_dir.glob("fig*_panel*.*")
        if not path.name.lower().startswith(f"{figure_id}_panel")
    ]
    if legacy_named:
        warnings.append(f"legacy experiment stem-like outputs detected: {legacy_named}")
    else:
        passes.append("no legacy experiment stem-like outputs detected")
    canvas = spec.get("canvas_mm") or {}
    if canvas.get("width") and canvas.get("height") and not check_only:
        passes.append(f"full figure target size recorded as {canvas['width']} x {canvas['height']} mm")


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {report['figure_id']} QC report",
        "",
        f"- check_only: {report['check_only']}",
        f"- result: {'PASS' if report['ok'] else 'FAIL'}",
        "",
        "## Passes",
    ]
    lines.extend(f"- {msg}" for msg in report["passes"])
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {msg}" for msg in report["warnings"])
    lines.extend(["", "## Failures"])
    lines.extend(f"- {msg}" for msg in report["failures"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_summary_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows = [row for row in rows if row.get("figure_id") != report["figure_id"]]
    rows.append(
        {
            "figure_id": report["figure_id"],
            "ok": str(bool(report["ok"])),
            "n_passes": len(report["passes"]),
            "n_warnings": len(report["warnings"]),
            "n_failures": len(report["failures"]),
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure_id", "ok", "n_passes", "n_warnings", "n_failures"])
        writer.writeheader()
        writer.writerows(rows)
