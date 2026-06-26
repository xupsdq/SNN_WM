from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult, missing_adapter_result, panel_stem
from src.plotting.paper_fig.export import (
    export_full_figure,
    export_individual_panels,
    export_resolved_spec,
    export_source_manifest,
)
from src.plotting.paper_fig.qc import run_qc
from src.plotting.paper_fig.registry import get_figure_index, get_figure_spec, validate_registry
from src.plotting.paper_fig.typography import (
    PANEL_LABEL_SIZE_PT,
    apply_paper_figure_typography,
    mark_panel_label,
)
from src.plotting.paper_fig.utils import paper_fig_output_root, paper_fig_root, read_json, repo_root_from_here


def build_figure(
    fig_id: str,
    *,
    panel_id: str | None = None,
    check_only: bool = False,
    experiment_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build or check one paper figure."""
    repo_root = repo_root_from_here()
    fig_id = fig_id.lower()
    selected = {panel_id.upper()} if panel_id else None
    spec = get_figure_spec(fig_id)
    if experiment_root is not None:
        spec["experiment_root"] = str(experiment_root)
    if output_root is not None:
        spec["output_root"] = str(output_root)
    implementation_id = _implementation_id(spec, fig_id)
    output_root = spec.get("output_root")
    if output_root:
        output_dir = repo_root / str(output_root) / fig_id
    else:
        output_dir = paper_fig_output_root(repo_root) / fig_id
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_report = validate_registry(repo_root)
    if registry_report["failures"]:
        raise RuntimeError("Registry validation failed: " + "; ".join(registry_report["failures"]))

    panels = _selected_panels(spec, selected)
    if selected is None:
        _clean_inactive_panel_outputs(output_dir, fig_id, panels)
    adapter_results: dict[str, AdapterResult] = {}
    for pid, panel in panels.items():
        adapter_name = panel.get("data_adapter")
        if adapter_name in (None, "", "none") or panel.get("panel_type") in _schematic_panel_types():
            continue
        panel_spec = _panel_spec(spec, pid, panel)
        adapter_results[pid] = _run_adapter(implementation_id, str(adapter_name), panel_spec, repo_root, output_dir)

    aggregate_manifest = _aggregate_source_manifest(spec, panels, adapter_results)
    export_resolved_spec(spec, output_dir, fig_id)
    export_source_manifest(aggregate_manifest, output_dir, fig_id)

    full_export_paths: dict[str, str] | None = None
    placeholder_reasons: dict[str, str] = {}
    render_metadata: dict[str, dict[str, Any]] = {}
    if not check_only:
        fig, axes = _create_layout(implementation_id, spec, selected)
        render_jobs: dict[str, tuple[Any, Any, Mapping[str, Any], Any]] = {}
        for pid, ax in axes.items():
            panel = panels[pid]
            panel_spec = _panel_spec(spec, pid, panel)
            renderer = _resolve_renderer(implementation_id, str(panel.get("renderer") or "render_generic_placeholder"))
            panel_data, stats = _load_adapter_payload(adapter_results.get(pid))
            renderer(ax, panel_data, stats, panel_spec, style={})
            render_metadata[pid] = _collect_render_metadata(ax)
            reason = getattr(ax, "paper_fig_placeholder_reason", None)
            if reason:
                placeholder_reasons[pid] = str(reason)
            render_jobs[pid] = (renderer, panel_data, panel_spec, stats)
        apply_paper_figure_typography(fig)
        finalize_layout = getattr(fig, "paper_fig_finalize_layout", None)
        if callable(finalize_layout):
            finalize_layout(fig, axes, spec)
        panel_label_artists = _draw_panel_labels(fig, spec, panels, selected)
        apply_paper_figure_typography(fig)
        if callable(finalize_layout):
            finalize_layout(fig, axes, spec)
        fig.canvas.draw()
        for pid, ax in axes.items():
            render_metadata[pid].update(_collect_render_metadata(ax, fig=fig, panel_label_artist=panel_label_artists.get(pid)))
        write_layout_report = getattr(fig, "paper_fig_write_layout_report", None)
        if callable(write_layout_report):
            write_layout_report(fig, axes, spec, output_dir)
        full_export_paths = export_full_figure(fig, output_dir, fig_id, panel_id=panel_id)
        export_individual_panels(render_jobs, output_dir, fig_id, renderer_style={})
        plt.close(fig)

    qc_report = run_qc(
        spec,
        output_dir,
        adapter_results,
        full_export_paths,
        selected_panels=selected,
        check_only=check_only,
        placeholder_reasons=placeholder_reasons,
        render_metadata=render_metadata,
    )
    return {"figure_id": fig_id, "output_dir": str(output_dir), "qc": qc_report}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for paper figure generation."""
    parser = argparse.ArgumentParser(description="Build paper-specific manuscript figures from existing results.")
    parser.add_argument("--fig", type=str, default=None, help="Figure id, e.g. fig1.")
    parser.add_argument("--panel", type=str, default=None, help="Optional panel id, e.g. C.")
    parser.add_argument("--all", action="store_true", help="Build all indexed figures.")
    parser.add_argument("--check-only", action="store_true", help="Validate specs/sources and write QC without exporting figures.")
    parser.add_argument("--experiment-root", type=str, default=None, help="Optional experiment result root for figures that support standalone paper experiments.")
    parser.add_argument("--output-root", type=str, default=None, help="Repository-relative output root for generated paper figures.")
    args = parser.parse_args(argv)

    fig_ids = _requested_figures(args.fig, args.all, args.check_only)
    exit_code = 0
    for fig_id in fig_ids:
        result = build_figure(
            fig_id,
            panel_id=args.panel,
            check_only=bool(args.check_only),
            experiment_root=args.experiment_root,
            output_root=args.output_root,
        )
        qc = result["qc"]
        print(f"{fig_id}: {'PASS' if qc['ok'] else 'FAIL'} -> {result['output_dir']}")
        for warning in qc["warnings"]:
            print(f"WARN: {warning}")
        for failure in qc["failures"]:
            print(f"FAIL: {failure}")
        if qc["failures"]:
            exit_code = 1
    return exit_code


def _requested_figures(fig: str | None, build_all: bool, check_only: bool) -> list[str]:
    if build_all or (check_only and fig is None):
        figures = get_figure_index().get("figures") or {}
        return [str(k) for k in figures.keys()]
    if fig:
        return [fig]
    return ["fig1"]


def _selected_panels(spec: Mapping[str, Any], selected: set[str] | None) -> dict[str, Mapping[str, Any]]:
    panels = dict(spec.get("panels") or {})
    if selected is None:
        return panels
    missing = selected.difference(panels)
    if missing:
        raise KeyError(f"Unknown panel(s) for {spec.get('figure_id')}: {sorted(missing)}")
    return {pid: panels[pid] for pid in panels if pid in selected}


def _panel_spec(spec: Mapping[str, Any], panel_id: str, panel: Mapping[str, Any]) -> dict[str, Any]:
    panel_spec = dict(panel)
    panel_spec.setdefault("figure_id", spec.get("figure_id"))
    panel_spec.setdefault("panel_id", panel_id)
    if spec.get("experiment_root") is not None:
        panel_spec.setdefault("experiment_root", spec.get("experiment_root"))
    if spec.get("experiment_root_default") is not None:
        panel_spec.setdefault("experiment_root_default", spec.get("experiment_root_default"))
    return panel_spec


def _implementation_id(spec: Mapping[str, Any], fig_id: str) -> str:
    """Return the implementation module id for specs that intentionally reuse code."""
    return str(spec.get("implementation_id") or fig_id).lower()


def _run_adapter(implementation_id: str, adapter_name: str, panel_spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    func_name = f"build_{adapter_name}"
    try:
        module = importlib.import_module(f"src.plotting.paper_fig.adapters.{implementation_id}_adapters")
        func = getattr(module, func_name)
    except Exception as exc:
        return missing_adapter_result(panel_spec, repo_root, output_dir, f"Adapter {func_name} unavailable: {exc}")
    try:
        return func(panel_spec, repo_root, output_dir)
    except Exception as exc:
        return missing_adapter_result(panel_spec, repo_root, output_dir, f"Adapter {func_name} failed: {exc}")


def _resolve_renderer(implementation_id: str, renderer_name: str):
    try:
        module = importlib.import_module(f"src.plotting.paper_fig.panels.{implementation_id}_panels")
        return getattr(module, renderer_name)
    except Exception:
        module = importlib.import_module("src.plotting.paper_fig.panels.fig1_panels")
        return getattr(module, "render_generic_placeholder")


def _create_layout(implementation_id: str, spec: Mapping[str, Any], selected: set[str] | None):
    try:
        module = importlib.import_module(f"src.plotting.paper_fig.layouts.{implementation_id}_layout")
    except Exception:
        module = importlib.import_module("src.plotting.paper_fig.layouts.generic_layout")
    return module.create_layout(spec, selected_panels=selected)


def _load_adapter_payload(result: AdapterResult | None) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if result is None:
        return None, {}
    panel_data = pd.read_csv(result.panel_data_path) if result.panel_data_path.exists() else None
    stats = read_json(result.stats_manifest_path) if result.stats_manifest_path.exists() else {}
    return panel_data, stats


def _collect_render_metadata(ax, *, fig=None, panel_label_artist=None) -> dict[str, Any]:
    """Collect renderer metadata used by figure-specific QC."""
    pos = ax.get_position()
    metadata: dict[str, Any] = {
        "plot_form": getattr(ax, "paper_fig_plot_form", ""),
        "has_shaded_band": bool(getattr(ax, "paper_fig_has_shaded_band", False)),
        "shaded_band": getattr(ax, "paper_fig_shaded_band", []),
        "line_emphasis": getattr(ax, "paper_fig_line_emphasis", ""),
        "has_mean_marker": bool(getattr(ax, "paper_fig_has_mean_marker", False)),
        "has_mean_annotation": bool(getattr(ax, "paper_fig_has_mean_annotation", False)),
        "raw_points": bool(getattr(ax, "paper_fig_raw_points", False)),
        "value_labels": bool(getattr(ax, "paper_fig_value_labels", False)),
        "value_label_count": int(getattr(ax, "paper_fig_value_label_count", 0) or 0),
        "value_labels_clear": bool(getattr(ax, "paper_fig_value_labels_clear", False)),
        "similarity_direction_arrow": bool(getattr(ax, "paper_fig_similarity_direction_arrow", False)),
        "literal_bin_xticklabels": bool(getattr(ax, "paper_fig_literal_bin_xticklabels", False)),
        "similarity_bar_order_preserved": bool(getattr(ax, "paper_fig_similarity_bar_order_preserved", False)),
        "stack_metrics": getattr(ax, "paper_fig_stack_metrics", []),
        "y_label_inside": bool(getattr(ax, "paper_fig_y_label_inside", False)),
        "x_label": str(ax.get_xlabel()),
        "y_label": str(ax.get_ylabel()),
        "title": str(ax.get_title()),
        "x_tick_rotations": [float(label.get_rotation()) for label in ax.get_xticklabels()],
        "x_tick_labels": [str(label.get_text()) for label in ax.get_xticklabels()],
        "x_tick_fontstyles": [str(label.get_fontstyle()) for label in ax.get_xticklabels()],
        "axes_bounds": [float(pos.x0), float(pos.y0), float(pos.x1), float(pos.y1)],
        "plot_axes_bounds": getattr(ax, "paper_fig_plot_axes_bounds", [float(pos.x0), float(pos.y0), float(pos.x1), float(pos.y1)]),
        "inner_axes_bounds": getattr(ax, "paper_fig_inner_axes_bounds", []),
        "inner_axes_aligned": bool(getattr(ax, "paper_fig_inner_axes_aligned", False)),
        "axes_mm": getattr(ax, "paper_fig_axes_mm", {}),
        "panel_bounds": getattr(ax, "paper_fig_panel_bounds", [float(pos.x0), float(pos.y0), float(pos.x1), float(pos.y1)]),
        "legend_overlaps_axes_bbox": False,
        "legend_overlaps_data": bool(getattr(ax, "paper_fig_legend_overlaps_data", False)),
        "legend_texts": getattr(ax, "paper_fig_legend_texts", []),
        "legend_ncols": int(getattr(ax, "paper_fig_legend_ncols", 0) or 0),
        "legend_above_plot": bool(getattr(ax, "paper_fig_legend_above_plot", False)),
        "x_metric": str(getattr(ax, "paper_fig_x_metric", "")),
        "y_metric": str(getattr(ax, "paper_fig_y_metric", "")),
        "score_name": str(getattr(ax, "paper_fig_score_name", "")),
        "score_excludes": getattr(ax, "paper_fig_score_excludes", []),
        "primary_endpoint": str(getattr(ax, "paper_fig_primary_endpoint", "")),
        "score_interpretation": str(getattr(ax, "paper_fig_score_interpretation", "")),
        "final_label_claim": getattr(ax, "paper_fig_final_label_claim", None),
        "pure_mechanism_schematic": bool(getattr(ax, "paper_fig_pure_mechanism_schematic", False)),
        "optional_placeholder": bool(getattr(ax, "paper_fig_optional_placeholder", False)),
        "y_range_mode": str(getattr(ax, "paper_fig_y_range_mode", "")),
        "raw_point_count": int(getattr(ax, "paper_fig_raw_point_count", 0) or 0),
        "raw_point_alpha": float(getattr(ax, "paper_fig_raw_point_alpha", 0.0) or 0.0),
        "showfliers": bool(getattr(ax, "paper_fig_showfliers", True)),
        "third_condition_label": str(getattr(ax, "paper_fig_third_condition_label", "")),
        "rows_before_renderer_aggregation": int(getattr(ax, "paper_fig_rows_before_renderer_aggregation", 0) or 0),
        "plotted_x_positions_by_layer": getattr(ax, "paper_fig_plotted_x_positions_by_layer", {}),
        "repeated_x_positions_averaged": bool(getattr(ax, "paper_fig_repeated_x_positions_averaged", False)),
        "renderer_summarizes_row_level": bool(getattr(ax, "paper_fig_renderer_summarizes_row_level", False)),
        "paired_change_style_source": str(getattr(ax, "paper_fig_paired_change_style_source", "")),
        "paired_change_summary_arrow": bool(getattr(ax, "paper_fig_paired_change_summary_arrow", False)),
        "individual_traces": bool(getattr(ax, "paper_fig_individual_traces", False)),
        "preserves_fig4e_labels": bool(getattr(ax, "paper_fig_preserves_fig4e_labels", False)),
        "shaded_window": getattr(ax, "paper_fig_shaded_window", []),
        "shaded_window_color": str(getattr(ax, "paper_fig_shaded_window_color", "")),
        "shaded_window_alpha": float(getattr(ax, "paper_fig_shaded_window_alpha", 0.0) or 0.0),
        "peak_annotations": getattr(ax, "paper_fig_peak_annotations", []),
        "static_dynamic_trajectory": bool(getattr(ax, "paper_fig_static_dynamic_trajectory", False)),
        "trajectory_logic_source": str(getattr(ax, "paper_fig_trajectory_logic_source", "")),
        "trace_metrics": getattr(ax, "paper_fig_trace_metrics", []),
        "mean_arrows": bool(getattr(ax, "paper_fig_mean_arrows", False)),
        "axis_direction_annotations": bool(getattr(ax, "paper_fig_axis_direction_annotations", False)),
        "interaction_annotation": bool(getattr(ax, "paper_fig_interaction_annotation", False)),
        "is_two_category_paired_recovery": bool(getattr(ax, "paper_fig_is_two_category_paired_recovery", False)),
        "aspect": str(ax.get_aspect()),
        "forced_equal_aspect": bool(getattr(ax, "paper_fig_forced_equal_aspect", False)),
        "normal_rectangular_panel": bool(getattr(ax, "paper_fig_normal_rectangular_panel", False)),
        "e_y_annotation_outside_plot": bool(getattr(ax, "paper_fig_e_y_annotation_outside_plot", False)),
        "e_legend_repositioned_inside_panel": bool(getattr(ax, "paper_fig_e_legend_repositioned_inside_panel", False)),
        "e_legend_inside_axes": bool(getattr(ax, "paper_fig_e_legend_inside_axes", False)),
        "e_legend_upper_left": bool(getattr(ax, "paper_fig_e_legend_upper_left", False)),
        "e_axis_region_aligned_with_d": bool(getattr(ax, "paper_fig_e_axis_region_aligned_with_d", False)),
        "e_legend_fontsize": float(getattr(ax, "paper_fig_e_legend_fontsize", 0.0) or 0.0),
        "e_legend_markers_enlarged": bool(getattr(ax, "paper_fig_e_legend_markers_enlarged", False)),
        "xlim": [float(v) for v in ax.get_xlim()],
        "ylim": [float(v) for v in ax.get_ylim()],
        "legend_bbox": [],
        "has_colorbar": bool(getattr(ax, "paper_fig_has_colorbar", False)),
        "colorbar_removed": bool(getattr(ax, "paper_fig_colorbar_removed", False)),
        "colorbar_bbox": [],
        "colorbar_axes_mm": getattr(ax, "paper_fig_colorbar_axes_mm", {}),
        "colorbar_label": str(getattr(ax, "paper_fig_colorbar_label", "")),
        "is_3d_surface": bool(getattr(ax, "paper_fig_is_3d_surface", False)),
        "3d_fallback_reason": str(getattr(ax, "paper_fig_3d_fallback_reason", "")),
        "has_summary_inset": bool(getattr(ax, "paper_fig_has_summary_inset", False)),
        "heatmap_square": bool(getattr(ax, "paper_fig_heatmap_square", False)),
        "colorbar_does_not_resize_axes": bool(getattr(ax, "paper_fig_colorbar_does_not_resize_axes", False)),
        "has_y1_reference": bool(getattr(ax, "paper_fig_has_y1_reference", False)),
        "merged_center_panel": bool(getattr(ax, "paper_fig_merged_center_panel", False)),
        "center_line_colors": getattr(ax, "paper_fig_center_line_colors", {}),
        "endpoint_text_labels": bool(getattr(ax, "paper_fig_endpoint_text_labels", False)),
        "support_map_uncropped": bool(getattr(ax, "paper_fig_support_map_uncropped", False)),
        "bar_connector_removed": bool(getattr(ax, "paper_fig_bar_connector_removed", False)),
        "bar_connector_lines_remaining": int(getattr(ax, "paper_fig_bar_connector_lines_remaining", 0) or 0),
        "category_labels_wrapped": bool(getattr(ax, "paper_fig_category_labels_wrapped", False)),
        "role_bboxes": {},
        "x_tick_bboxes": [],
        "y_tick_bboxes": [],
        "clipped_artists": [],
        "panel_label_clipped": False,
        "y_tick_labels_inside_axes": False,
        "y_label_inside_axes": False,
        "panel_label_gap_mm": None,
        "panel_label_bbox": [],
        "svg_asset": str(getattr(ax, "paper_fig_svg_asset", "")),
        "svg_viewbox": getattr(ax, "paper_fig_svg_viewbox", {}),
        "svg_aspect_ratio": float(getattr(ax, "paper_fig_svg_aspect_ratio", 0.0) or 0.0),
        "svg_rendered_size_mm": getattr(ax, "paper_fig_svg_rendered_size_mm", {}),
        "svg_raster_cache": str(getattr(ax, "paper_fig_svg_raster_cache", "")),
    }
    legend = ax.get_legend()
    if legend is not None:
        metadata["legend_texts"] = [text.get_text() for text in legend.get_texts()]
    if fig is None:
        return metadata
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    clipped: list[str] = []
    artists = []
    if ax.axison:
        artists.extend((f"x_tick:{label.get_text()}", label) for label in ax.get_xticklabels())
        artists.extend((f"y_tick:{label.get_text()}", label) for label in ax.get_yticklabels())
        artists.extend([("x_label", ax.xaxis.label), ("y_label", ax.yaxis.label)])
        artists.extend((f"text:{text.get_text()[:24]}", text) for text in ax.texts)
    if legend is not None:
        legend_bbox = legend.get_window_extent(renderer)
        metadata["legend_overlaps_axes_bbox"] = bool(legend_bbox.overlaps(ax.bbox))
        metadata["legend_bbox"] = _bbox_fig_bounds(fig, legend_bbox)
        artists.append(("legend", legend))
    colorbar_ax = getattr(ax, "paper_fig_colorbar_ax", None)
    if colorbar_ax is not None:
        metadata["colorbar_bbox"] = _bbox_fig_bounds(fig, colorbar_ax.bbox)
        artists.extend((f"colorbar_tick:{label.get_text()}", label) for label in colorbar_ax.get_yticklabels())
        artists.append(("colorbar_label", colorbar_ax.yaxis.label))
    if panel_label_artist is not None:
        artists.append(("panel_label", panel_label_artist))
        metadata["panel_label_gap_mm"] = getattr(panel_label_artist, "paper_fig_panel_label_gap_mm", None)
    for name, artist in artists:
        if not artist.get_visible():
            continue
        try:
            bbox = artist.get_window_extent(renderer)
        except Exception:
            continue
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        if not _bbox_inside(fig_bbox, bbox, pad_px=1.0):
            if name == "panel_label":
                metadata["panel_label_clipped"] = True
            else:
                clipped.append(name)
        if name == "panel_label":
            metadata["panel_label_bbox"] = _bbox_fig_bounds(fig, bbox)
        if name.startswith("x_tick:"):
            metadata["x_tick_bboxes"].append(_bbox_fig_bounds(fig, bbox))
        if name.startswith("y_tick:"):
            metadata["y_tick_bboxes"].append(_bbox_fig_bounds(fig, bbox))
        if name.startswith("y_tick:") and bbox.overlaps(ax.bbox):
            metadata["y_tick_labels_inside_axes"] = True
        if name == "y_label" and bbox.overlaps(ax.bbox):
            metadata["y_label_inside_axes"] = True
        role = getattr(artist, "paper_fig_role", None)
        if role:
            metadata["role_bboxes"][str(role)] = _bbox_fig_bounds(fig, bbox)
    metadata["clipped_artists"] = clipped
    return metadata


def _bbox_inside(outer, inner, *, pad_px: float = 0.0) -> bool:
    return (
        inner.x0 >= outer.x0 - pad_px
        and inner.y0 >= outer.y0 - pad_px
        and inner.x1 <= outer.x1 + pad_px
        and inner.y1 <= outer.y1 + pad_px
    )


def _bbox_fig_bounds(fig, bbox) -> list[float]:
    fig_bbox = bbox.transformed(fig.transFigure.inverted())
    return [float(fig_bbox.x0), float(fig_bbox.y0), float(fig_bbox.x1), float(fig_bbox.y1)]


def _draw_panel_labels(fig, spec: Mapping[str, Any], panels: Mapping[str, Mapping[str, Any]], selected: set[str] | None) -> dict[str, Any]:
    """Place panel labels in figure coordinates so rows align consistently."""
    canvas = spec.get("canvas_mm") or {}
    canvas_w = float(canvas.get("width", 1.0))
    canvas_h = float(canvas.get("height", 1.0))
    artists: dict[str, Any] = {}
    for panel_id, panel in panels.items():
        if selected is not None and panel_id not in selected:
            continue
        pos = panel.get("position_mm") or {}
        label_pos = panel.get("letter_mm") or panel.get("panel_label_mm")
        if label_pos:
            x_mm = float(label_pos.get("x", pos.get("x", 0.0)))
            y_mm = float(label_pos.get("y", pos.get("y", 0.0)))
        else:
            x_mm = max(1.5, float(pos.get("x", 0.0)) - 3.0)
            y_pos = float(pos.get("y", 0.0))
            if str(spec.get("figure_id", "")).lower() in {"fig2", "fig3", "fig4"}:
                y_mm = min(float(pos.get("y", 0.0)) + 2.0, 4.0) if y_pos <= 3.0 else max(2.0, y_pos - 4.0)
            elif str(spec.get("figure_id", "")).lower() == "fig6" and y_pos >= 90.0:
                y_mm = y_pos
            else:
                y_mm = 2.0 if y_pos <= 3.0 else max(2.0, y_pos - 8.0)
        if not label_pos and str(spec.get("figure_id", "")).lower() in {"fig2", "fig3", "fig4"}:
            y_mm = min(float(pos.get("y", 0.0)) + 2.0, 4.0) if y_pos <= 3.0 else max(2.0, y_pos - 4.0)
        artists[panel_id] = fig.text(
            x_mm / canvas_w,
            1.0 - (y_mm / canvas_h),
            panel_id.lower(),
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=PANEL_LABEL_SIZE_PT,
        )
        mark_panel_label(artists[panel_id])
        artists[panel_id].paper_fig_panel_label_gap_mm = float(pos.get("y", 0.0)) - y_mm
    return artists


def _aggregate_source_manifest(
    spec: Mapping[str, Any],
    panels: Mapping[str, Mapping[str, Any]],
    adapter_results: Mapping[str, AdapterResult],
) -> dict[str, Any]:
    figure_id = str(spec.get("figure_id"))
    sources: list[dict[str, Any]] = []
    for panel_id, panel in panels.items():
        producer_task = panel.get("producer_task")
        if panel_id in adapter_results:
            manifest = adapter_results[panel_id].source_manifest
            entry = {"panel_id": panel_id, "status": manifest.get("status", "unknown"), "manifest": manifest}
            if producer_task:
                entry["producer_task"] = producer_task
            sources.append(entry)
            continue
        if panel.get("panel_type") in _schematic_panel_types():
            raw_asset = panel.get("source") or (panel.get("source_mapping") or {}).get("manual_asset")
            if raw_asset:
                asset_path = Path(str(raw_asset))
                if not asset_path.is_absolute():
                    asset_path = repo_root_from_here() / asset_path
                if not asset_path.exists():
                    asset_path = paper_fig_root() / str(raw_asset)
                entry = {
                    "panel_id": panel_id,
                    "status": "ok" if asset_path.exists() else "missing_source",
                    "source_type": "manual_asset",
                    "path": str(raw_asset),
                    "exists": asset_path.exists(),
                }
                if producer_task:
                    entry["producer_task"] = producer_task
                sources.append(entry)
            else:
                entry = {
                    "panel_id": panel_id,
                    "status": "ok",
                    "source_type": "programmatic_schematic",
                    "path": None,
                    "exists": True,
                }
                if producer_task:
                    entry["producer_task"] = producer_task
                sources.append(entry)
        else:
            entry = {
                "panel_id": panel_id,
                "status": "no_adapter",
                "source_mapping": panel.get("source_mapping") or {},
            }
            if producer_task:
                entry["producer_task"] = producer_task
            sources.append(entry)
    panel_task_dependencies = {
        panel_id: panel.get("producer_task")
        for panel_id, panel in panels.items()
        if panel.get("producer_task")
    }
    return {
        "figure_id": figure_id,
        "active_panels": list(panels.keys()),
        "panel_task_dependencies": panel_task_dependencies,
        "sources": sources,
    }


def _clean_inactive_panel_outputs(output_dir: Path, figure_id: str, panels: Mapping[str, Mapping[str, Any]]) -> None:
    """Remove generated per-panel artifacts whose panel ids are no longer active."""
    active_stems = {panel_stem(figure_id, panel_id) for panel_id in panels}
    active_individual_stems = {f"{figure_id}{panel_id.lower()}" for panel_id in panels}
    suffix_by_subdir = {
        "panel_data": "_panel_data.csv",
        "stats": "_stats.json",
        "source_manifests": "_sources.json",
    }
    for subdir, suffix in suffix_by_subdir.items():
        folder = output_dir / subdir
        if not folder.exists():
            continue
        for path in folder.glob(f"{figure_id}*"):
            stem = path.name.removesuffix(suffix)
            if path.is_file() and stem not in active_stems:
                path.unlink()
    panel_dir = output_dir / "individual_panels"
    if panel_dir.exists():
        for path in panel_dir.glob(f"{figure_id}*"):
            if path.is_file() and path.stem not in active_individual_stems:
                path.unlink()


def _schematic_panel_types() -> set[str]:
    """Return panel types that are rendered programmatically or manually without data adapters."""
    return {
        "manual_schematic",
        "manual_or_programmatic_schematic",
        "programmatic_or_manual_schematic",
        "two_item_episode_schematic",
        "multi_item_sequence_schematic",
    }


if __name__ == "__main__":
    raise SystemExit(main())
