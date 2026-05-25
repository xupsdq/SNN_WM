from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.plotting.paper_fig.utils import write_json, write_yaml


def export_full_figure(fig: Figure, output_dir: Path, figure_id: str, *, panel_id: str | None = None) -> dict[str, str]:
    """Export the full paper figure without tight bounding boxes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_id if panel_id is None else f"{figure_id}_panel_{panel_id.lower()}"
    paths: dict[str, str] = {}
    for ext in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300)
        paths[ext] = str(path)
    return paths


def export_individual_panels(
    render_jobs: Mapping[str, tuple[Any, Any, Mapping[str, Any], Any]],
    output_dir: Path,
    figure_id: str,
    renderer_style: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Export individual panels as standalone SVG/PNG artifacts."""
    panel_dir = output_dir / "individual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, dict[str, str]] = {}
    for panel_id, (renderer, panel_data, panel_spec, stats) in render_jobs.items():
        width_mm, height_mm = _size_mm(panel_spec.get("size_mm"))
        width = width_mm / 25.4
        height = height_mm / 25.4
        fig = plt.figure(figsize=(width, height), dpi=300)
        if _needs_3d_axis(panel_spec):
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
        renderer(ax, panel_data, stats, panel_spec, style=renderer_style)
        fig.text(0.01, 0.99, panel_id, ha="left", va="top", fontweight="bold")
        panel_paths: dict[str, str] = {}
        for ext in ("svg", "png"):
            path = panel_dir / f"{figure_id}{panel_id.lower()}.{ext}"
            fig.savefig(path, dpi=300)
            panel_paths[ext] = str(path)
        plt.close(fig)
        exported[panel_id] = panel_paths
    return exported


def _size_mm(size_spec: Any) -> tuple[float, float]:
    """Return panel width and height in millimeters from list or mapping specs."""
    if isinstance(size_spec, Mapping):
        return float(size_spec.get("width", size_spec.get("w", 50))), float(size_spec.get("height", size_spec.get("h", 50)))
    if isinstance(size_spec, (list, tuple)) and len(size_spec) >= 2:
        return float(size_spec[0]), float(size_spec[1])
    return 50.0, 50.0


def _needs_3d_axis(panel_spec: Mapping[str, Any]) -> bool:
    return str(panel_spec.get("projection", "")).lower() == "3d" or str(panel_spec.get("panel_type", "")).lower() in {"3d_surface", "surface_3d"}


def export_resolved_spec(spec: Mapping[str, Any], output_dir: Path, figure_id: str) -> Path:
    """Write the resolved figure spec YAML."""
    return write_yaml(spec, output_dir / f"{figure_id}_resolved_spec.yaml")


def export_source_manifest(manifest: Mapping[str, Any], output_dir: Path, figure_id: str) -> Path:
    """Write the aggregate source manifest JSON."""
    return write_json(manifest, output_dir / f"{figure_id}_source_manifest.json")
