from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.plotting.paper_fig.typography import (
    PANEL_LABEL_SIZE_PT,
    VECTOR_TEXT_RCPARAMS,
    apply_paper_figure_typography,
    mark_panel_label,
)
from src.plotting.paper_fig.utils import write_json, write_yaml


def export_full_figure(fig: Figure, output_dir: Path, figure_id: str, *, panel_id: str | None = None, dpi: int = 300) -> dict[str, str]:
    """Export the full paper figure without tight bounding boxes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_id if panel_id is None else f"{figure_id}_panel_{panel_id.lower()}"
    paths: dict[str, str] = {}
    apply_paper_figure_typography(fig)
    for ext in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{ext}"
        with plt.rc_context(VECTOR_TEXT_RCPARAMS):
            fig.savefig(path, dpi=dpi)
        paths[ext] = str(path)
    return paths


def export_individual_panels(
    render_jobs: Mapping[str, tuple[Any, Any, Mapping[str, Any], Any]],
    output_dir: Path,
    figure_id: str,
    renderer_style: Mapping[str, Any] | None = None,
    dpi: int = 300,
) -> dict[str, dict[str, str]]:
    """Export individual panels as standalone SVG/PNG artifacts."""
    panel_dir = output_dir / "individual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, dict[str, str]] = {}
    for panel_id, render_job in render_jobs.items():
        panel_spec = render_job[2]
        exported[panel_id] = _export_panel_artifact(
            render_job,
            panel_dir,
            f"{figure_id}{panel_id.lower()}",
            panel_id,
            formats=("svg", "png"),
            renderer_style=renderer_style,
            dpi=int(panel_spec.get("png_dpi", dpi)),
        )
    return exported


def export_selected_panel(
    render_job: tuple[Any, Any, Mapping[str, Any], Any],
    output_dir: Path,
    figure_id: str,
    panel_id: str,
    renderer_style: Mapping[str, Any] | None = None,
    *,
    dpi: int = 300,
) -> dict[str, str]:
    """Export one selected panel on its fixed physical panel canvas."""
    panel_spec = render_job[2]
    return _export_panel_artifact(
        render_job,
        output_dir,
        f"{figure_id}_panel_{panel_id.lower()}",
        panel_id,
        formats=("pdf", "svg", "png"),
        renderer_style=renderer_style,
        dpi=int(panel_spec.get("png_dpi", dpi)),
    )


def _export_panel_artifact(
    render_job: tuple[Any, Any, Mapping[str, Any], Any],
    output_dir: Path,
    stem: str,
    panel_id: str,
    *,
    formats: tuple[str, ...],
    renderer_style: Mapping[str, Any] | None,
    dpi: int,
) -> dict[str, str]:
    renderer, panel_data, panel_spec, stats = render_job
    width_mm, height_mm = _size_mm(panel_spec.get("size_mm"))
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d") if _needs_3d_axis(panel_spec) else fig.add_subplot(111)
    individual_axes = panel_spec.get("individual_axes") or {}
    if individual_axes:
        fig.subplots_adjust(
            left=float(individual_axes.get("left", 0.125)),
            right=float(individual_axes.get("right", 0.90)),
            bottom=float(individual_axes.get("bottom", 0.11)),
            top=float(individual_axes.get("top", 0.88)),
        )
    elif "schematic" in str(panel_spec.get("panel_type", "")).lower():
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    renderer(ax, panel_data, stats, panel_spec, style=renderer_style)
    mark_panel_label(
        fig.text(
            0.01,
            0.99,
            panel_id.lower(),
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=PANEL_LABEL_SIZE_PT,
        )
    )
    apply_paper_figure_typography(fig)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for ext in formats:
        path = output_dir / f"{stem}.{ext}"
        with plt.rc_context(VECTOR_TEXT_RCPARAMS):
            fig.savefig(path, dpi=dpi)
        paths[ext] = str(path)
    plt.close(fig)
    return paths


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
