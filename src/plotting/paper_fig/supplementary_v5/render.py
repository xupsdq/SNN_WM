from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from PIL import Image
from pypdf import PdfReader

from .common import BundleReader, save_figure, sha256_file, write_json
from .renderers import FIGURE_RENDERERS
from .specs import load_spec

_PLOT_OWNED_PARENT_PATHS = frozenset({"metrics/plot_qa.json"})


def _parent_hashes(input_dir: Path) -> dict[str, str]:
    paths = sorted(
        path
        for root in (input_dir / "data", input_dir / "metrics")
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(input_dir).as_posix()
        not in _PLOT_OWNED_PARENT_PATHS
    )
    return {path.relative_to(input_dir).as_posix(): sha256_file(path) for path in paths}


def _text_clipping_report(fig) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    failures: list[str] = []
    for text in fig.findobj(match=lambda artist: hasattr(artist, "get_text")):
        if not text.get_visible() or not str(text.get_text()).strip():
            continue
        try:
            bbox = text.get_window_extent(renderer=renderer)
        except Exception:
            continue
        tolerance = 1.5
        if (
            bbox.x0 < canvas.x0 - tolerance
            or bbox.y0 < canvas.y0 - tolerance
            or bbox.x1 > canvas.x1 + tolerance
            or bbox.y1 > canvas.y1 + tolerance
        ):
            failures.append(str(text.get_text()))
    return failures


def _bbox_inside(inner: list[float], outer: list[float], *, tolerance_mm: float) -> bool:
    inner_left, inner_top, inner_width, inner_height = inner
    outer_left, outer_top, outer_width, outer_height = outer
    return (
        inner_left >= outer_left - tolerance_mm
        and inner_top >= outer_top - tolerance_mm
        and inner_left + inner_width <= outer_left + outer_width + tolerance_mm
        and inner_top + inner_height <= outer_top + outer_height + tolerance_mm
    )


def _plot_geometry_report(spec: dict[str, Any], *, tolerance_mm: float = 0.01) -> dict[str, object]:
    """Validate visible data regions, not only their enclosing panel slots."""

    panels = spec["panels"]
    equal_slot_groups: dict[str, dict[str, float]] = {}
    equal_row_groups: dict[str, dict[str, float]] = {}
    panel_records: dict[str, dict[str, object]] = {}
    failures: list[str] = []

    for panel_id, panel in panels.items():
        slot = [float(value) for value in panel["slot_bbox_mm"]]
        plot = [float(value) for value in panel["plot_bbox_mm"]]
        if not _bbox_inside(plot, slot, tolerance_mm=tolerance_mm):
            failures.append(f"panel {panel_id} plot area lies outside its slot")

        canonical_slot_width = 52.3335 if abs(slot[2] - 52.3335) <= tolerance_mm else slot[2]
        slot_group = f"{canonical_slot_width:.4f}x{slot[3]:.3f}"
        equal_slot_groups.setdefault(slot_group, {})[panel_id] = plot[2]
        row_group = f"top={slot[1]:.3f},height={slot[3]:.3f}"
        equal_row_groups.setdefault(row_group, {})[panel_id] = plot[3]

        record: dict[str, object] = {
            "slot_bbox_mm": slot,
            "plot_bbox_mm": plot,
            "visible_plot_size_mm": [plot[2], plot[3]],
        }
        if abs(slot[2] - 79.5) <= tolerance_mm:
            standard_width_ok = plot[2] >= 63.5 - tolerance_mm
            record["standard_two_column_width_ok"] = standard_width_ok
            if not standard_width_ok:
                failures.append(
                    f"panel {panel_id} uses only {plot[2]:g} mm of a standard 79.5 mm slot; "
                    "expected at least 63.5 mm or an approved exception"
                )
        if abs(slot[2] - 52.3335) <= tolerance_mm:
            standard_width_ok = plot[2] >= 37.0 - tolerance_mm
            record["standard_three_column_width_ok"] = standard_width_ok
            if not standard_width_ok:
                failures.append(
                    f"panel {panel_id} uses only {plot[2]:g} mm of a standard 52.333 mm slot; "
                    "expected at least 37.0 mm or an approved exception"
                )
        if "colorbar_bbox_mm" in panel:
            colorbar = [float(value) for value in panel["colorbar_bbox_mm"]]
            record["colorbar_bbox_mm"] = colorbar
            if not _bbox_inside(colorbar, slot, tolerance_mm=tolerance_mm):
                failures.append(f"panel {panel_id} colorbar lies outside its slot")
            if abs(colorbar[0] - plot[0]) > tolerance_mm or abs(colorbar[2] - plot[2]) > tolerance_mm:
                failures.append(
                    f"panel {panel_id} colorbar span {colorbar[0]:g}/{colorbar[2]:g} mm "
                    f"does not match plot span {plot[0]:g}/{plot[2]:g} mm"
                )
        panel_records[panel_id] = record

    width_checks: list[dict[str, object]] = []
    for slot_group, widths in equal_slot_groups.items():
        if len(widths) < 2:
            continue
        span = max(widths.values()) - min(widths.values())
        width_checks.append({"slot_size_mm": slot_group, "panel_widths_mm": widths, "span_mm": span})
        if span > tolerance_mm:
            failures.append(f"equal slots {slot_group} have unequal visible plot widths: {widths}")

    height_checks: list[dict[str, object]] = []
    for row_group, heights in equal_row_groups.items():
        if len(heights) < 2:
            continue
        span = max(heights.values()) - min(heights.values())
        height_checks.append({"row": row_group, "panel_heights_mm": heights, "span_mm": span})
        if span > tolerance_mm:
            failures.append(f"panels in {row_group} have unequal visible plot heights: {heights}")

    if failures:
        raise ValueError(f"{spec['figure_id']}: visible plot geometry contract failed: {'; '.join(failures)}")
    return {
        "status": "passed",
        "tolerance_mm": tolerance_mm,
        "panels": panel_records,
        "equal_slot_width_checks": width_checks,
        "equal_row_height_checks": height_checks,
    }


def _axis_anchor_report(fig, spec: dict[str, Any]) -> dict[str, object]:
    """Require visible bottom/left anchors on every non-schematic data axis."""

    schematic_types = {"causal_identity_schematic", "exchange_schematic"}
    axes_by_label = {axis.get_label(): axis for axis in fig.axes}
    panel_records: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for panel_id, panel in spec["panels"].items():
        axis = axes_by_label.get(f"panel_{panel_id}_plot")
        if axis is None:
            failures.append(f"panel {panel_id} has no registered plot axis")
            continue
        panel_type = str(panel["panel_type"])
        if panel_type in schematic_types:
            panel_records[panel_id] = {
                "panel_type": panel_type,
                "axis_exception": "true schematic",
                "axis_on": bool(axis.axison),
            }
            continue
        bottom_visible = bool(axis.axison and axis.spines["bottom"].get_visible())
        left_visible = bool(axis.axison and axis.spines["left"].get_visible())
        panel_records[panel_id] = {
            "panel_type": panel_type,
            "axis_on": bool(axis.axison),
            "bottom_spine_visible": bottom_visible,
            "left_spine_visible": left_visible,
        }
        if not bottom_visible or not left_visible:
            failures.append(
                f"panel {panel_id} is missing quantitative axis anchors "
                f"(bottom={bottom_visible}, left={left_visible})"
            )
    if failures:
        raise ValueError(f"{spec['figure_id']}: axis anchor contract failed: {'; '.join(failures)}")
    return {"status": "passed", "panels": panel_records}


def _vector_output_report(outputs: dict[str, Path], spec: dict[str, object]) -> dict[str, object]:
    expected_width = float(spec["canvas_mm"][0]) / 25.4 * 72.0
    expected_height = float(spec["canvas_mm"][1]) / 25.4 * 72.0

    pdf = PdfReader(outputs["pdf"])
    if len(pdf.pages) != 1:
        raise ValueError(f"Expected a one-page PDF, found {len(pdf.pages)} pages: {outputs['pdf']}")
    page = pdf.pages[0]
    pdf_width = float(page.mediabox.width)
    pdf_height = float(page.mediabox.height)
    if abs(pdf_width - expected_width) > 0.02 or abs(pdf_height - expected_height) > 0.02:
        raise ValueError(
            f"Unexpected PDF page size {(pdf_width, pdf_height)} versus "
            f"{(expected_width, expected_height)}"
        )
    pdf_text = page.extract_text() or ""
    pdf_fonts = page["/Resources"]["/Font"].get_object()
    if not pdf_text.strip() or not pdf_fonts:
        raise ValueError(f"PDF text or font resources are missing: {outputs['pdf']}")

    svg_root = ET.parse(outputs["svg"]).getroot()
    try:
        svg_width = float(str(svg_root.attrib["width"]).removesuffix("pt"))
        svg_height = float(str(svg_root.attrib["height"]).removesuffix("pt"))
        view_box = [float(value) for value in str(svg_root.attrib["viewBox"]).split()]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SVG canvas declaration: {outputs['svg']}") from exc
    if (
        abs(svg_width - expected_width) > 0.02
        or abs(svg_height - expected_height) > 0.02
        or len(view_box) != 4
        or any(abs(value) > 0.02 for value in view_box[:2])
        or abs(view_box[2] - expected_width) > 0.02
        or abs(view_box[3] - expected_height) > 0.02
    ):
        raise ValueError(
            f"Unexpected SVG canvas {(svg_width, svg_height, view_box)} versus "
            f"{(expected_width, expected_height)}"
        )
    svg_text_count = len(svg_root.findall(".//{http://www.w3.org/2000/svg}text"))
    if svg_text_count == 0:
        raise ValueError(f"SVG contains no editable text elements: {outputs['svg']}")
    return {
        "expected_points": [expected_width, expected_height],
        "pdf_points": [pdf_width, pdf_height],
        "pdf_pages": len(pdf.pages),
        "pdf_font_resources": len(pdf_fonts),
        "pdf_text_characters": len(pdf_text),
        "svg_points": [svg_width, svg_height],
        "svg_text_elements": svg_text_count,
    }


def render_supplementary_v5(
    *,
    input_dir: Path,
    output_dir: Path | None = None,
    figures: Sequence[str] = tuple(f"s{index}" for index in range(1, 8)),
    dpi: int = 300,
) -> dict[str, object]:
    input_root = input_dir.resolve()
    output_root = (output_dir or input_root).resolve()
    reader = BundleReader(input_root)
    statistics = reader.read_csv("metrics/panel_statistics.csv", "frozen panel statistics")
    source_manifest = reader.read_csv("data/source_data_manifest.csv", "frozen Source Data manifest")
    requested = tuple(str(figure).lower() for figure in figures)
    unknown = sorted(set(requested) - set(FIGURE_RENDERERS))
    if unknown:
        raise ValueError(f"Unknown supplementary figure ids: {unknown}")

    before = _parent_hashes(input_root)
    figure_dir = output_root / "figures"
    resolved_spec_dir = output_root / "meta" / "resolved_specs"
    render_records: list[dict[str, object]] = []
    clipping_records: list[dict[str, object]] = []
    for figure_id in requested:
        spec = load_spec(figure_id)
        geometry_report = _plot_geometry_report(spec)
        manifest_panels = set(
            source_manifest.loc[source_manifest["figure_id"].eq(figure_id), "panel_id"].astype(str)
        )
        missing_panels = sorted(set(spec["panels"]) - manifest_panels)
        if missing_panels:
            raise ValueError(f"{figure_id}: Source Data manifest is missing panels {missing_panels}")
        resolved_spec = {key: value for key, value in spec.items() if not key.startswith("_")}
        write_json(resolved_spec_dir / f"{figure_id}.json", resolved_spec)
        fig = FIGURE_RENDERERS[figure_id](reader, spec, statistics)
        axis_anchor_report = _axis_anchor_report(fig, spec)
        clipped_text = _text_clipping_report(fig)
        if clipped_text:
            raise ValueError(f"Text clipping detected in {figure_id}: {clipped_text}")
        outputs = save_figure(fig, figure_dir, f"supp_fig_{figure_id}", dpi=dpi)
        plt.close(fig)
        vector_report = _vector_output_report(outputs, spec)
        with Image.open(outputs["png"]) as image:
            width_px, height_px = image.size
        expected_width = int(float(spec["canvas_mm"][0]) / 25.4 * dpi)
        expected_height = int(float(spec["canvas_mm"][1]) / 25.4 * dpi)
        if abs(width_px - expected_width) > 1 or abs(height_px - expected_height) > 1:
            raise ValueError(
                f"Unexpected PNG dimensions for {figure_id}: {(width_px, height_px)} versus "
                f"{(expected_width, expected_height)}"
            )
        render_records.append(
            {
                "figure_id": figure_id,
                "canvas_mm": spec["canvas_mm"],
                "png_pixels": [width_px, height_px],
                "expected_png_pixels": [expected_width, expected_height],
                "visible_plot_geometry": geometry_report,
                "axis_anchors": axis_anchor_report,
                "vector_outputs": vector_report,
                "outputs": {
                    key: Path(value).relative_to(output_root).as_posix()
                    for key, value in outputs.items()
                },
            }
        )
        clipping_records.append({"figure_id": figure_id, "clipped_text": clipped_text})


    output_files = sorted(
        path
        for path in (output_root / "figures").rglob("*")
        if path.is_file()
    ) + sorted(path for path in resolved_spec_dir.glob("*.json") if path.is_file())
    plot_manifest = {
        "schema_version": 1,
        "producer": "src.plotting.experiments.supplementary_v5_plot",
        "plot_only": True,
        "figures": list(requested),
        "parent_hashes_unchanged": True,
        "source_accesses": reader.accesses,
        "artifacts": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in output_files
        ],
    }
    write_json(output_root / "meta" / "plot_artifact_manifest.json", plot_manifest)
    qa = {
        "status": "passed",
        "plot_only": True,
        "parent_hashes_unchanged": True,
        "tight_layout_used": False,
        "constrained_layout_used": False,
        "bbox_inches_tight_used": False,
        "visible_plot_geometry_passed": all(
            record["visible_plot_geometry"]["status"] == "passed" for record in render_records
        ),
        "axis_anchors_passed": all(
            record["axis_anchors"]["status"] == "passed" for record in render_records
        ),
        "render_records": render_records,
        "text_clipping": clipping_records,
        "grayscale_outputs_present": all(
            (output_root / record["outputs"]["grayscale"]).is_file() for record in render_records
        ),
    }
    write_json(output_root / "metrics" / "plot_qa.json", qa)
    write_json(
        output_root / "meta" / "plot_run_config.json",
        {
            "input_dir": str(input_root),
            "output_dir": str(output_root),
            "figures": list(requested),
            "dpi": int(dpi),
            "experiment_rerun": False,
            "inference_recomputed": False,
        },
    )
    after = _parent_hashes(input_root)
    if before != after:
        changed = sorted(set(before) | set(after))
        changed = [path for path in changed if before.get(path) != after.get(path)]
        raise RuntimeError(
            f"Plot-only replay changed parent Source Data or metrics: {changed}"
        )
    return {
        "status": "complete",
        "figures": list(requested),
        "figure_dir": str(figure_dir),
        "parent_hashes_unchanged": True,
    }


__all__ = ["render_supplementary_v5"]
