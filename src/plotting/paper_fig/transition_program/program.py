from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.text import Text
from PIL import Image

from src.plotting.paper_fig.typography import (
    FIGURE_TEXT_SIZE_PT,
    PANEL_LABEL_SIZE_PT,
    apply_paper_figure_typography,
)

from .context import BuildContext
from .contracts import (
    FIGURE_BY_ID,
    FIGURE_CONTRACTS,
    FigureContract,
    validate_contracts,
)
from .main_builders import MAIN_BUILDERS
from .sources import SourceStore, all_contract_dataset_keys
from .style import (
    CORAL,
    CYAN,
    GRAY,
    GRAY_DARK,
    NAVY,
    PURPLE,
    TEAL,
    apply_transition_style,
)
from .supplementary_builders import SUPPLEMENTARY_BUILDERS


DEFAULT_PAPER_ROOT = Path("results/paper_figure_multi_seed")
DEFAULT_OUTPUT_ROOT = Path(
    "results/paper_figures/results_state_transition_figure_pack"
)
ALLOWED_FORMATS = ("png", "pdf", "svg")
BUILDERS = {**MAIN_BUILDERS, **SUPPLEMENTARY_BUILDERS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot-only builder for the v5 state-transition main and "
            "supplementary figure pack. It never runs simulations."
        )
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=DEFAULT_PAPER_ROOT,
        help=(
            "Root containing the persisted paper_figure_multi_seed "
            "artifacts (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Normalized output bundle (default: %(default)s).",
    )
    parser.add_argument(
        "--figure",
        action="append",
        default=[],
        help=(
            "Figure id to build (fig1..fig7 or s1..s8). Repeat to select "
            "multiple figures. The default is the full pack."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--main-only",
        action="store_true",
        help="Build Fig. 1-Fig. 7 only.",
    )
    group.add_argument(
        "--supplements-only",
        action="store_true",
        help="Build Fig. S1-Fig. S8 only.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf,svg",
        help="Comma-separated output formats from png,pdf,svg.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output resolution (default: %(default)s).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate contracts, required columns, cohorts, and parent "
            "hash readability without rendering or writing outputs."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace files for selected figures if they already exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = _repo_root()
    paper_root = _resolve(repo_root, args.paper_root)
    output_root = _resolve(repo_root, args.output_dir)
    formats = _parse_formats(args.formats)
    contracts = _select_contracts(args)
    validate_contracts()
    _validate_builder_coverage()
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")

    apply_transition_style()
    store = SourceStore(repo_root=repo_root, paper_root=paper_root)
    if args.check_only:
        _validate_sources_without_retaining(store, contracts)
        changed = store.verify_unchanged()
        if changed:
            raise RuntimeError(
                "Parent artifacts changed during check-only validation:\n"
                + "\n".join(changed)
            )
        print(
            json.dumps(
                {
                    "status": "validated",
                    "figures": [contract.figure_id for contract in contracts],
                    "panels": sum(len(contract.panels) for contract in contracts),
                    "datasets": len(all_contract_dataset_keys(contracts)),
                    "source_files": len(store.source_records),
                    "paper_root": str(paper_root),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    _prepare_output_dirs(output_root)
    _guard_existing_outputs(
        output_root,
        contracts,
        formats,
        force=args.force,
    )
    started_at = datetime.now(timezone.utc)
    log_lines = [
        f"{started_at.isoformat()} build started",
        f"repo_root={repo_root}",
        f"paper_root={paper_root}",
        f"output_root={output_root}",
        f"figures={','.join(contract.figure_id for contract in contracts)}",
        f"formats={','.join(formats)} dpi={args.dpi}",
        "mode=plot-only (no simulation entrypoint imported or executed)",
    ]
    context = BuildContext(store=store, output_root=output_root)
    rendered: list[dict[str, object]] = []
    try:
        for contract in contracts:
            keys = sorted(
                {
                    key
                    for panel in contract.panels
                    for key in panel.datasets
                }
            )
            store.validate(keys)
            figure_record = _render_figure(
                context,
                contract,
                output_root,
                formats=formats,
                dpi=args.dpi,
            )
            rendered.append(figure_record)
            store.cache.clear()
            log_lines.append(
                f"{datetime.now(timezone.utc).isoformat()} "
                f"rendered {contract.figure_id}"
            )

        changed = store.verify_unchanged()
        if changed:
            raise RuntimeError(
                "Parent artifacts changed during plotting:\n"
                + "\n".join(changed)
            )
        context.add_qc(
            "pack",
            "parent_hash_immutability",
            "pass",
            (
                f"{len(store.source_records)} persisted source files retained "
                "their pre-render SHA256 hashes."
            ),
        )
        _write_contract_manifest(output_root, contracts)
        _write_source_manifest(output_root, store)
        _write_color_manifest(output_root)
        context.write_tables()
        finished_at = datetime.now(timezone.utc)
        _write_run_config(
            output_root,
            args,
            repo_root=repo_root,
            paper_root=paper_root,
            contracts=contracts,
            formats=formats,
            started_at=started_at,
            finished_at=finished_at,
        )
        _write_summary(
            output_root,
            contracts=contracts,
            rendered=rendered,
            store=store,
            context=context,
            started_at=started_at,
            finished_at=finished_at,
        )
        log_lines.append(f"{finished_at.isoformat()} build completed")
        _write_text(
            output_root / "logs" / "build.log",
            "\n".join(log_lines) + "\n",
        )
        _write_artifact_manifest(output_root)
    except Exception as error:
        log_lines.append(
            f"{datetime.now(timezone.utc).isoformat()} build failed: "
            f"{type(error).__name__}: {error}"
        )
        _write_text(
            output_root / "logs" / "build.log",
            "\n".join(log_lines) + "\n",
        )
        raise

    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": str(output_root),
                "figures": len(rendered),
                "panels": sum(len(contract.panels) for contract in contracts),
                "source_files": len(store.source_records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _render_figure(
    context: BuildContext,
    contract: FigureContract,
    output_root: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> dict[str, object]:
    builder = BUILDERS[contract.figure_id]
    fig = builder(context, contract)
    try:
        apply_paper_figure_typography(fig)
        _validate_canvas(fig, contract)
        fig.canvas.draw()
        _validate_title_free_panels(context, fig, contract)
        _validate_single_chart_panels(context, fig, contract)
        _record_text_qc(context, fig, contract)
        _record_panel_layout_qc(context, fig, contract)
        paths: dict[str, str] = {}
        for suffix in formats:
            path = output_root / "figures" / suffix / (
                f"{contract.figure_id}.{suffix}"
            )
            fig.savefig(
                path,
                format=suffix,
                dpi=dpi,
                bbox_inches=None,
                pad_inches=0.0,
                facecolor="white",
            )
            paths[suffix] = path.relative_to(output_root).as_posix()
        if "png" in formats:
            png_path = output_root / paths["png"]
            _validate_png_dimensions(context, contract, png_path, dpi=dpi)
            gray_path = (
                output_root
                / "figures"
                / "grayscale"
                / f"{contract.figure_id}_grayscale.png"
            )
            with Image.open(png_path) as image:
                image.convert("L").convert("RGB").save(
                    gray_path,
                    dpi=(dpi, dpi),
                )
            paths["grayscale"] = gray_path.relative_to(output_root).as_posix()
        if contract.figure_id == "fig1":
            context.add_qc(
                "fig1",
                "source_bound_decoder_representation",
                "documented",
                (
                    "The persisted decoder source exposes ux_concat only; "
                    "panel c therefore shows the combined u/x state and does "
                    "not fabricate separate u and x decoders."
                ),
            )
        context.add_qc(
            contract.figure_id,
            "declared_block_layout",
            "pass",
            (
                f"{len(contract.panels)} blocks fill the fixed "
                f"{contract.canvas_mm[0]:g} x {contract.canvas_mm[1]:g} mm "
                "canvas using 2 mm margins and gutters."
            ),
        )
        return {
            "figure_id": contract.figure_id,
            "display_id": contract.display_id,
            "canvas_width_mm": contract.canvas_mm[0],
            "canvas_height_mm": contract.canvas_mm[1],
            "panels": len(contract.panels),
            "paths": paths,
        }
    finally:
        plt.close(fig)


def _validate_canvas(fig: Figure, contract: FigureContract) -> None:
    observed = np.asarray(fig.get_size_inches(), dtype=float) * 25.4
    expected = np.asarray(contract.canvas_mm, dtype=float)
    if not np.allclose(observed, expected, atol=1e-7, rtol=0.0):
        raise ValueError(
            f"{contract.figure_id}: canvas {tuple(observed)} mm does not "
            f"match {tuple(expected)} mm"
        )


def _inactive_tick_text_ids(fig: Figure) -> set[int]:
    inactive: set[int] = set()
    renderer = fig.canvas.get_renderer()
    for axis in fig.axes:
        x_tick_labels = [
            label
            for tick in (
                *axis.xaxis.get_major_ticks(),
                *axis.xaxis.get_minor_ticks(),
            )
            for label in (tick.label1, tick.label2)
        ]
        y_tick_labels = [
            label
            for tick in (
                *axis.yaxis.get_major_ticks(),
                *axis.yaxis.get_minor_ticks(),
            )
            for label in (tick.label1, tick.label2)
        ]
        if not axis.axison:
            inactive.update(id(label) for label in x_tick_labels)
            inactive.update(id(label) for label in y_tick_labels)
            inactive.add(id(axis.xaxis.label))
            inactive.add(id(axis.yaxis.label))
            continue
        x_low, x_high = sorted(axis.get_xlim())
        y_low, y_high = sorted(axis.get_ylim())
        x_tolerance = max((x_high - x_low) * 1e-9, 1e-12)
        y_tolerance = max((y_high - y_low) * 1e-9, 1e-12)
        for location, label in zip(
            axis.get_xticks(),
            axis.get_xticklabels(),
        ):
            if not x_low - x_tolerance <= location <= x_high + x_tolerance:
                inactive.add(id(label))
        for location, label in zip(
            axis.get_yticks(),
            axis.get_yticklabels(),
        ):
            if not y_low - y_tolerance <= location <= y_high + y_tolerance:
                inactive.add(id(label))
        axis_bbox = axis.get_window_extent(renderer=renderer)
        for label in x_tick_labels:
            if not label.get_visible() or not label.get_text().strip():
                continue
            bbox = label.get_window_extent(renderer=renderer)
            center_x = (bbox.x0 + bbox.x1) / 2.0
            if center_x < axis_bbox.x0 - 2.0 or center_x > axis_bbox.x1 + 2.0:
                inactive.add(id(label))
        for label in y_tick_labels:
            if not label.get_visible() or not label.get_text().strip():
                continue
            bbox = label.get_window_extent(renderer=renderer)
            center_y = (bbox.y0 + bbox.y1) / 2.0
            if center_y < axis_bbox.y0 - 2.0 or center_y > axis_bbox.y1 + 2.0:
                inactive.add(id(label))
    return inactive


def _record_text_qc(
    context: BuildContext,
    fig: Figure,
    contract: FigureContract,
) -> None:
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    inactive_tick_ids = _inactive_tick_text_ids(fig)
    clipped: list[str] = []
    wrong_sizes: list[str] = []
    forbidden_sample_sizes: list[str] = []
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        if id(text) in inactive_tick_ids:
            continue
        if re.search(
            r"\bn\s*=\s*\d+(?:\s*(?:network|networks))?\b",
            text.get_text(),
            flags=re.IGNORECASE,
        ):
            forbidden_sample_sizes.append(text.get_text())
        size = float(text.get_fontsize())
        expected_size = (
            PANEL_LABEL_SIZE_PT
            if getattr(text, "paper_fig_text_role", "") == "panel_label"
            else FIGURE_TEXT_SIZE_PT
        )
        if abs(size - expected_size) > 0.05:
            wrong_sizes.append(f"{text.get_text()[:24]!r}: {size:g} pt")
        try:
            bbox = text.get_window_extent(renderer=renderer)
        except (RuntimeError, ValueError):
            continue
        owner = getattr(text, "axes", None)
        if text.get_clip_on() and owner is not None:
            owner_bbox = owner.get_window_extent(renderer=renderer)
            if (
                min(bbox.x1, owner_bbox.x1) <= max(bbox.x0, owner_bbox.x0)
                or min(bbox.y1, owner_bbox.y1) <= max(bbox.y0, owner_bbox.y0)
            ):
                continue
        tolerance = 4.0
        if (
            bbox.x0 < figure_bbox.x0 - tolerance
            or bbox.y0 < figure_bbox.y0 - tolerance
            or bbox.x1 > figure_bbox.x1 + tolerance
            or bbox.y1 > figure_bbox.y1 + tolerance
        ):
            clipped.append(text.get_text().replace("\n", " ")[:48])
    if wrong_sizes:
        raise ValueError(
            f"{contract.figure_id}: typography normalization failed: "
            + "; ".join(wrong_sizes[:8])
        )
    if forbidden_sample_sizes:
        raise ValueError(
            f"{contract.figure_id}: repeated sample-size labels are forbidden: "
            + "; ".join(forbidden_sample_sizes[:8])
        )
    context.add_qc(
        contract.figure_id,
        "typography",
        "pass",
        (
            f"All visible text normalized to {FIGURE_TEXT_SIZE_PT:g} pt; "
            f"panel labels normalized to {PANEL_LABEL_SIZE_PT:g} pt."
        ),
    )
    context.add_qc(
        contract.figure_id,
        "sample_size_annotations",
        "pass",
        "No n=... sample-size labels are rendered inside figure panels.",
    )
    context.add_qc(
        contract.figure_id,
        "figure_edge_text_clipping",
        "pass" if not clipped else "warning",
        (
            "No visible text extends beyond the canvas."
            if not clipped
            else "Potential edge text: " + "; ".join(clipped[:8])
        ),
    )


def _validate_single_chart_panels(
    context: BuildContext,
    fig: Figure,
    contract: FigureContract,
) -> None:
    slots = getattr(fig, "transition_panel_slots", {})
    violations: list[str] = []
    for panel in contract.panels:
        slot = slots.get(panel.panel_id)
        if slot is None:
            continue
        visible_children = [
            axis for axis in slot.child_axes if axis.get_visible()
        ]
        if len(visible_children) > 1:
            violations.append(
                f"{panel.panel_id}:{len(visible_children)} data axes"
            )
    if violations:
        raise ValueError(
            f"{contract.figure_id}: micro-chart panels are forbidden: "
            + "; ".join(violations)
        )
    context.add_qc(
        contract.figure_id,
        "single_integrated_chart_per_panel",
        "pass",
        "Every panel contains at most one visible data axis.",
    )


def _validate_title_free_panels(
    context: BuildContext,
    fig: Figure,
    contract: FigureContract,
) -> None:
    slots = getattr(fig, "transition_panel_slots", {})
    leaked = []
    for panel in contract.panels:
        slot = slots.get(panel.panel_id)
        if slot is None:
            raise ValueError(
                f"{contract.figure_id}: missing slot for panel {panel.panel_id}"
            )
        leaked.extend(
            text.get_text()
            for text in slot.texts
            if getattr(text, "paper_fig_text_role", "") != "panel_label"
        )
    if leaked:
        raise ValueError(
            f"{contract.figure_id}: panel titles were rendered: {leaked}"
        )
    context.add_qc(
        contract.figure_id,
        "title_free_panels",
        "pass",
        (
            "All verbal panel titles are omitted; only lowercase panel "
            "identifiers remain in the title band, and the data region "
            "reclaims the released space."
        ),
    )


def _validate_png_dimensions(
    context: BuildContext,
    contract: FigureContract,
    path: Path,
    *,
    dpi: int,
) -> None:
    expected = tuple(
        int(round(value / 25.4 * dpi)) for value in contract.canvas_mm
    )
    with Image.open(path) as image:
        observed = image.size
    if any(abs(a - b) > 1 for a, b in zip(observed, expected)):
        raise ValueError(
            f"{contract.figure_id}: PNG is {observed} px, expected "
            f"{expected} px at {dpi} dpi"
        )
    context.add_qc(
        contract.figure_id,
        "raster_canvas_dimensions",
        "pass",
        (
            f"{observed[0]} x {observed[1]} px at {dpi} dpi "
            f"for {contract.canvas_mm[0]:g} x "
            f"{contract.canvas_mm[1]:g} mm."
        ),
    )


def _record_panel_layout_qc(
    context: BuildContext,
    fig: Figure,
    contract: FigureContract,
) -> None:
    renderer = fig.canvas.get_renderer()
    inactive_tick_ids = _inactive_tick_text_ids(fig)
    slots = getattr(fig, "transition_panel_slots", {})
    slot_boxes = {
        panel_id: slot.get_window_extent(renderer=renderer)
        for panel_id, slot in slots.items()
    }
    text_records: list[tuple[str, str, object, int, str]] = []
    cross_block: list[str] = []
    for axis in fig.axes:
        axis_box = axis.get_window_extent(renderer=renderer)
        center_x = (axis_box.x0 + axis_box.x1) / 2.0
        center_y = (axis_box.y0 + axis_box.y1) / 2.0
        candidates = [
            (panel_id, slot_box)
            for panel_id, slot_box in slot_boxes.items()
            if slot_box.contains(center_x, center_y)
        ]
        if not candidates:
            continue
        panel_id, slot_box = min(
            candidates,
            key=lambda item: item[1].width * item[1].height,
        )
        x_tick_ids = {
            id(label)
            for tick in (
                *axis.xaxis.get_major_ticks(),
                *axis.xaxis.get_minor_ticks(),
            )
            for label in (tick.label1, tick.label2)
        }
        y_tick_ids = {
            id(label)
            for tick in (
                *axis.yaxis.get_major_ticks(),
                *axis.yaxis.get_minor_ticks(),
            )
            for label in (tick.label1, tick.label2)
        }
        legend = axis.get_legend()
        legend_text_ids = (
            {id(text) for text in legend.get_texts()}
            if legend is not None
            else set()
        )
        for text in axis.findobj(match=Text):
            if not text.get_visible() or not text.get_text().strip():
                continue
            if id(text) in inactive_tick_ids:
                continue
            try:
                bbox = text.get_window_extent(renderer=renderer)
            except (RuntimeError, ValueError):
                continue
            if bbox.width <= 0.0 or bbox.height <= 0.0:
                continue
            if text.get_clip_on():
                intersection_width = min(bbox.x1, axis_box.x1) - max(
                    bbox.x0,
                    axis_box.x0,
                )
                intersection_height = min(bbox.y1, axis_box.y1) - max(
                    bbox.y0,
                    axis_box.y0,
                )
                if intersection_width <= 0.0 or intersection_height <= 0.0:
                    continue
            label = text.get_text().replace("\n", " ")[:32]
            if id(text) in x_tick_ids:
                role = "x_tick"
            elif id(text) in y_tick_ids:
                role = "y_tick"
            elif id(text) in legend_text_ids:
                role = "legend"
            elif text is axis.xaxis.label:
                role = "x_label"
            elif text is axis.yaxis.label:
                role = "y_label"
            elif getattr(text, "paper_fig_text_role", "") == "panel_label":
                role = "panel_label"
            else:
                role = "other"
            text_records.append((panel_id, label, bbox, id(axis), role))
            text_area = bbox.width * bbox.height
            for other_panel, other_box in slot_boxes.items():
                if other_panel == panel_id:
                    continue
                intersection_width = min(bbox.x1, other_box.x1) - max(
                    bbox.x0,
                    other_box.x0,
                )
                intersection_height = min(bbox.y1, other_box.y1) - max(
                    bbox.y0,
                    other_box.y0,
                )
                if intersection_width <= 2.0 or intersection_height <= 2.0:
                    continue
                overlap_area = intersection_width * intersection_height
                if overlap_area / max(text_area, 1.0) >= 0.08:
                    cross_block.append(
                        f"{panel_id}:{label} -> {other_panel}"
                    )
                    break
    overlaps: list[str] = []
    for index, (
        panel_a,
        label_a,
        box_a,
        axis_a,
        role_a,
    ) in enumerate(text_records):
        for (
            panel_b,
            label_b,
            box_b,
            axis_b,
            role_b,
        ) in text_records[index + 1 :]:
            if (
                panel_a == panel_b
                and {role_a, role_b} == {"x_tick", "y_tick"}
            ):
                continue
            intersection_width = min(box_a.x1, box_b.x1) - max(
                box_a.x0,
                box_b.x0,
            )
            intersection_height = min(box_a.y1, box_b.y1) - max(
                box_a.y0,
                box_b.y0,
            )
            if intersection_width <= 2.0 or intersection_height <= 2.0:
                continue
            if label_a == label_b and panel_a == panel_b:
                continue
            intersection_area = intersection_width * intersection_height
            smaller_area = min(
                box_a.width * box_a.height,
                box_b.width * box_b.height,
            )
            if intersection_area / max(smaller_area, 1.0) < 0.12:
                continue
            overlaps.append(
                f"{panel_a}:{label_a} <> {panel_b}:{label_b}"
            )
    context.add_qc(
        contract.figure_id,
        "panel_text_containment",
        "pass" if not cross_block else "warning",
        (
            "No visible text enters another declared panel block."
            if not cross_block
            else "Cross-block text: " + "; ".join(cross_block[:12])
        ),
    )
    context.add_qc(
        contract.figure_id,
        "text_text_overlap",
        "pass" if not overlaps else "warning",
        (
            "No visible text bounding boxes overlap."
            if not overlaps
            else "Potential text overlap: " + "; ".join(overlaps[:12])
        ),
    )


def _prepare_output_dirs(output_root: Path) -> None:
    for relative in (
        "data/panel_data",
        "figures/png",
        "figures/pdf",
        "figures/svg",
        "figures/grayscale",
        "logs",
        "metrics",
        "meta",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def _guard_existing_outputs(
    output_root: Path,
    contracts: Sequence[FigureContract],
    formats: Sequence[str],
    *,
    force: bool,
) -> None:
    if force:
        return
    existing = [
        output_root / "figures" / suffix / f"{contract.figure_id}.{suffix}"
        for contract in contracts
        for suffix in formats
        if (
            output_root
            / "figures"
            / suffix
            / f"{contract.figure_id}.{suffix}"
        ).exists()
    ]
    if existing:
        displayed = "\n".join(str(path) for path in existing[:10])
        raise FileExistsError(
            "Selected rendered files already exist. Re-run with --force "
            f"to replace them:\n{displayed}"
        )


def _write_contract_manifest(
    output_root: Path,
    contracts: Sequence[FigureContract],
) -> None:
    rows: list[dict[str, object]] = []
    for figure_order, contract in enumerate(contracts, start=1):
        for panel_order, panel in enumerate(contract.panels, start=1):
            x, y, width, height = panel.position_mm
            rows.append(
                {
                    "figure_order": figure_order,
                    "figure_id": contract.figure_id,
                    "display_id": contract.display_id,
                    "figure_title": contract.title,
                    "takeaway": contract.takeaway,
                    "kind": contract.kind,
                    "canvas_width_mm": contract.canvas_mm[0],
                    "canvas_height_mm": contract.canvas_mm[1],
                    "panel_order": panel_order,
                    "panel_id": panel.panel_id,
                    "panel_title": panel.title,
                    "chart_family": panel.chart_family,
                    "datasets": ";".join(panel.datasets),
                    "statistic": panel.statistic,
                    "renderer": panel.renderer,
                    "x_mm": x,
                    "y_top_mm": y,
                    "width_mm": width,
                    "height_mm": height,
                }
            )
    pd.DataFrame(rows).to_csv(
        output_root / "data" / "panel_manifest.csv",
        index=False,
        encoding="utf-8",
    )
    program_contract = {
        "canvas_width_mm": 165.0,
        "canvas_heights_mm": [102.0, 152.0],
        "outer_margin_mm": 2.0,
        "gutter_mm": 2.0,
        "row_height_mm": 48.0,
        "rendered_panel_titles": False,
        "rendered_panel_labels": True,
        "figure_count": len(contracts),
        "panel_count": sum(len(contract.panels) for contract in contracts),
        "figure_ids": [contract.figure_id for contract in contracts],
        "chart_family_counts": dict(
            sorted(
                Counter(
                    panel.chart_family
                    for contract in contracts
                    for panel in contract.panels
                ).items()
            )
        ),
    }
    _write_json(
        output_root / "meta" / "program_contract.json",
        program_contract,
    )


def _write_source_manifest(
    output_root: Path,
    store: SourceStore,
) -> None:
    records = sorted(
        store.source_records.values(),
        key=lambda record: (
            str(record["dataset_key"]),
            str(record["relative_path"]),
        ),
    )
    pd.DataFrame(records).to_csv(
        output_root / "data" / "source_manifest.csv",
        index=False,
        encoding="utf-8",
    )


def _write_color_manifest(output_root: Path) -> None:
    semantics = {
        "observed_dynamic_receiver": NAVY,
        "layer1_context": CYAN,
        "overlap_support_mechanism": TEAL,
        "donor_perturbation_rewrite": CORAL,
        "pair_residual_geometry": PURPLE,
        "passive_static_baseline": GRAY,
        "neutral_text_axes": GRAY_DARK,
    }
    _write_json(
        output_root / "meta" / "color_semantics.json",
        {
            "palette_basis": (
                "Existing src.plotting.common.colors "
                "Nature-compatible/Okabe-Ito semantics"
            ),
            "semantic_mapping": semantics,
            "continuous_maps": {
                "retained_support": "stsp_support",
                "counts": "update_count",
                "nonnegative_strength": "peak_strength",
                "signed_effect": "signed_effect",
            },
        },
    )


def _write_run_config(
    output_root: Path,
    args: argparse.Namespace,
    *,
    repo_root: Path,
    paper_root: Path,
    contracts: Sequence[FigureContract],
    formats: Sequence[str],
    started_at: datetime,
    finished_at: datetime,
) -> None:
    _write_json(
        output_root / "run_config.json",
        {
            "entrypoint": (
                "python -m "
                "src.plotting.experiments.results_state_transition_figures"
            ),
            "mode": "plot-only",
            "simulation_executed": False,
            "repo_root": str(repo_root),
            "paper_root": str(paper_root),
            "output_root": str(output_root),
            "figure_ids": [contract.figure_id for contract in contracts],
            "formats": list(formats),
            "dpi": int(args.dpi),
            "force": bool(args.force),
            "argv": list(sys.argv),
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "matplotlib": matplotlib.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    )


def _write_summary(
    output_root: Path,
    *,
    contracts: Sequence[FigureContract],
    rendered: Sequence[dict[str, object]],
    store: SourceStore,
    context: BuildContext,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    warnings = [
        record
        for record in context.qc_records
        if str(record["status"]).lower() == "warning"
    ]
    _write_json(
        output_root / "summary.json",
        {
            "status": "complete",
            "mode": "plot-only",
            "figure_count": len(rendered),
            "panel_count": sum(len(contract.panels) for contract in contracts),
            "source_file_count": len(store.source_records),
            "source_rows": int(
                sum(
                    int(record["rows"])
                    for record in store.source_records.values()
                )
            ),
            "parent_hashes_unchanged": True,
            "qc_warning_count": len(warnings),
            "qc_warnings": warnings,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "figures": list(rendered),
        },
    )


def _write_artifact_manifest(output_root: Path) -> None:
    files: list[dict[str, object]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        output_root / "artifact_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "file_count": len(files),
            "files": files,
        },
    )


def _validate_sources_without_retaining(
    store: SourceStore,
    contracts: Sequence[FigureContract],
) -> None:
    for key in sorted(all_contract_dataset_keys(contracts)):
        store.read(key)
        store.cache.pop(key, None)


def _select_contracts(args: argparse.Namespace) -> tuple[FigureContract, ...]:
    if args.figure:
        requested = tuple(dict.fromkeys(value.lower() for value in args.figure))
        unknown = [value for value in requested if value not in FIGURE_BY_ID]
        if unknown:
            raise ValueError(f"Unknown figure ids: {unknown}")
        return tuple(FIGURE_BY_ID[value] for value in requested)
    if args.main_only:
        return tuple(
            contract for contract in FIGURE_CONTRACTS if contract.kind == "main"
        )
    if args.supplements_only:
        return tuple(
            contract
            for contract in FIGURE_CONTRACTS
            if contract.kind == "supplement"
        )
    return FIGURE_CONTRACTS


def _validate_builder_coverage() -> None:
    expected = set(FIGURE_BY_ID)
    observed = set(BUILDERS)
    if observed != expected:
        raise ValueError(
            "Builder coverage mismatch: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(
        dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip())
    )
    if not formats:
        raise ValueError("--formats cannot be empty")
    invalid = [suffix for suffix in formats if suffix not in ALLOWED_FORMATS]
    if invalid:
        raise ValueError(
            f"Unsupported formats {invalid}; choose from {ALLOWED_FORMATS}"
        )
    return formats


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            default=_json_scalar,
        )
        handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_scalar(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


__all__ = [
    "BUILDERS",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PAPER_ROOT",
    "build_parser",
    "main",
]
