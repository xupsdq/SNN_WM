from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult, CANONICAL_COLUMNS, panel_output_paths
from src.plotting.paper_fig.utils import paper_fig_root, read_json, repo_root_from_here


def run_qc(
    spec: Mapping[str, Any],
    output_dir: Path,
    adapter_results: Mapping[str, AdapterResult],
    full_export_paths: Mapping[str, str] | None,
    *,
    selected_panels: set[str] | None = None,
    check_only: bool = False,
    placeholder_reasons: Mapping[str, str] | None = None,
    render_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run paper-figure QC and write per-figure plus aggregate reports."""
    figure_id = str(spec.get("figure_id"))
    panels = spec.get("panels") or {}
    if selected_panels is not None:
        panels = {k: v for k, v in panels.items() if k in selected_panels}
    passes: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []
    placeholders = dict(placeholder_reasons or {})

    _check_spec_basics(spec, panels, passes, warnings, failures)
    _check_adapter_outputs(figure_id, panels, output_dir, adapter_results, passes, warnings, failures)
    _check_fig1_specifics(figure_id, spec, panels, output_dir, adapter_results, full_export_paths, render_metadata or {}, passes, warnings, failures)
    _check_fig2_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig3_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig4_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig5_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig6_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_exports(figure_id, spec, output_dir, full_export_paths, check_only, passes, warnings, failures)

    for panel_id, reason in placeholders.items():
        warnings.append(f"{figure_id}{panel_id}: renderer placeholder present: {reason}")

    report = {
        "figure_id": figure_id,
        "check_only": check_only,
        "passes": passes,
        "warnings": warnings,
        "failures": failures,
        "ok": not failures,
    }
    _write_report(output_dir / f"{figure_id}_qc_report.md", report)
    _update_summary_csv(paper_fig_root() / "outputs" / "all_figures_qc_summary.csv", report)
    return report


def _check_spec_basics(spec: Mapping[str, Any], panels: Mapping[str, Any], passes: list[str], warnings: list[str], failures: list[str]) -> None:
    canvas = spec.get("canvas_mm") or {}
    if "width" in canvas and "height" in canvas:
        passes.append("canvas_mm defines width and height")
    else:
        failures.append("canvas_mm must define width and height")
    for panel_id, panel in panels.items():
        if panel.get("claim"):
            passes.append(f"{panel_id}: claim present")
        else:
            failures.append(f"{panel_id}: claim missing")
        if panel.get("panel_type") not in ("manual_schematic", "manual_or_programmatic_schematic", "programmatic_or_manual_schematic") and panel.get("data_adapter") in (None, "", "none"):
            warnings.append(f"{panel_id}: data-driven panel has no adapter")
        if not panel.get("renderer"):
            warnings.append(f"{panel_id}: renderer missing")
        if panel.get("x_axis") is None:
            warnings.append(f"{panel_id}: x_axis missing from spec")
        if panel.get("y_axis") is None:
            warnings.append(f"{panel_id}: y_axis missing from spec")
        if (spec.get("qc_requirements") or {}).get("require_position_mm"):
            if panel.get("position_mm"):
                passes.append(f"{panel_id}: position_mm present")
            else:
                failures.append(f"{panel_id}: position_mm missing")
        if (spec.get("qc_requirements") or {}).get("require_size_mm"):
            if panel.get("size_mm"):
                passes.append(f"{panel_id}: size_mm present")
            else:
                failures.append(f"{panel_id}: size_mm missing")


def _check_adapter_outputs(
    figure_id: str,
    panels: Mapping[str, Any],
    output_dir: Path,
    adapter_results: Mapping[str, AdapterResult],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    for panel_id, panel in panels.items():
        adapter = panel.get("data_adapter")
        if adapter in (None, "", "none") or panel.get("panel_type") in ("manual_schematic", "manual_or_programmatic_schematic", "programmatic_or_manual_schematic"):
            continue
        result = adapter_results.get(panel_id)
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        if result is None:
            warnings.append(f"{figure_id}{panel_id}: adapter did not run")
            continue
        for label, path in paths.items():
            if path.exists():
                passes.append(f"{figure_id}{panel_id}: {label} exists")
            else:
                failures.append(f"{figure_id}{panel_id}: {label} missing: {path}")
        if paths["panel_data"].exists():
            df = pd.read_csv(paths["panel_data"])
            missing_cols = [col for col in ("figure_id", "panel_id", "metric", "value", "source_file") if col not in df.columns]
            if missing_cols:
                failures.append(f"{figure_id}{panel_id}: panel data missing canonical columns {missing_cols}")
            else:
                passes.append(f"{figure_id}{panel_id}: canonical panel data columns present")
        if result.warnings:
            warnings.extend(f"{figure_id}{panel_id}: {message}" for message in result.warnings)
        status = result.source_manifest.get("status")
        if status == "missing_source":
            warnings.append(f"{figure_id}{panel_id}: missing source")


def _check_fig1_specifics(
    figure_id: str,
    spec: Mapping[str, Any],
    panels: Mapping[str, Any],
    output_dir: Path,
    adapter_results: Mapping[str, AdapterResult],
    full_export_paths: Mapping[str, str] | None,
    render_metadata: Mapping[str, Mapping[str, Any]],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    if figure_id != "fig1":
        return
    for panel_id in ("B", "C"):
        panel = panels.get(panel_id)
        if not panel:
            continue
        refs = panel.get("reference_lines") or []
        if any(float(ref.get("value")) == 10 for ref in refs):
            passes.append(f"Fig.1{panel_id}: 10% chance reference line present")
        else:
            failures.append(f"Fig.1{panel_id}: missing 10% chance reference line")

    b_path = panel_output_paths(output_dir, figure_id, "B")["panel_data"]
    if "B" in panels and b_path.exists():
        b_df = pd.read_csv(b_path)
        metrics = set(b_df.get("metric", pd.Series(dtype=str)).astype(str))
        if metrics == {"overall_recall"}:
            passes.append("Fig.1B uses overall recall only")
        else:
            failures.append(f"Fig.1B must use overall_recall only, found {sorted(metrics)}")
        if not any("class" in str(col).lower() for col in b_df.columns):
            passes.append("Fig.1B does not include class-specific recall columns")
        else:
            failures.append("Fig.1B must not include class-specific recall")
        b_form = str(render_metadata.get("B", {}).get("plot_form", ""))
        if b_form:
            if b_form == "recall_fluctuation_line":
                passes.append("Fig.1B renderer is recall fluctuation line")
            else:
                failures.append(f"Fig.1B must render as recall fluctuation line, found {b_form}")
            if bool(render_metadata.get("B", {}).get("has_shaded_band", False)):
                passes.append("Fig.1B has shaded fluctuation band")
            else:
                failures.append("Fig.1B must include a shaded fluctuation band")
            band = render_metadata.get("B", {}).get("shaded_band", [])
            if [float(v) for v in band] == [88.0, 95.0]:
                passes.append("Fig.1B shaded band spans 88% to 95%")
            else:
                failures.append(f"Fig.1B shaded band must span 88% to 95%, found {band}")
            if str(render_metadata.get("B", {}).get("line_emphasis", "")) == "line_over_points":
                passes.append("Fig.1B emphasizes line over points")
            else:
                failures.append("Fig.1B must emphasize the line over the points")
            if bool(render_metadata.get("B", {}).get("has_mean_marker", False)) or bool(render_metadata.get("B", {}).get("has_mean_annotation", False)):
                failures.append("Fig.1B must not include a special mean marker or numeric mean annotation")
            else:
                passes.append("Fig.1B has no special mean marker or numeric mean annotation")
            if bool(render_metadata.get("B", {}).get("y_label_inside", False)):
                failures.append("Fig.1B y-axis label must be outside the plot area")
            else:
                passes.append("Fig.1B y-axis label is outside the plot area")
        else:
            warnings.append("Fig.1B plot-form render metadata unavailable in check-only mode")

    c_path = panel_output_paths(output_dir, figure_id, "C")["panel_data"]
    if "C" in panels and c_path.exists():
        c_df = pd.read_csv(c_path)
        layers = list(dict.fromkeys(str(v) for v in c_df.get("layer", pd.Series(dtype=str)).dropna().tolist()))
        expected_layers = ["Layer 1", "Layer 2", "Layer 3"]
        if set(expected_layers).issubset(set(layers)):
            passes.append("Fig.1C has distinct Layer 1/2/3 labels")
        else:
            failures.append(f"Fig.1C must expose Layer 1/2/3 labels, found {layers}")
        if "delay_ms" not in c_df.columns:
            passes.append("Fig.1C panel data is layer-wise magnitude summary, not delay timecourse")
        else:
            failures.append("Fig.1C panel data must not include delay_ms timecourse values")
        if len(set(layers)) >= 3:
            passes.append("Fig.1C layer identities are distinguishable in panel data/spec")
        else:
            failures.append("Fig.1C layers are not clearly distinguishable")
        c_form = str(render_metadata.get("C", {}).get("plot_form", ""))
        if c_form:
            if c_form == "layer_bar_summary":
                passes.append("Fig.1C renderer is the intended layer-wise bar summary")
            else:
                failures.append(f"Fig.1C must render as layer-wise magnitude summary, found {c_form}")
            if bool(render_metadata.get("C", {}).get("raw_points", False)):
                failures.append("Fig.1C must not show raw-point overlays")
            else:
                passes.append("Fig.1C has no raw-point overlays")
            if bool(render_metadata.get("C", {}).get("value_labels", False)):
                passes.append("Fig.1C has value labels")
            else:
                failures.append("Fig.1C must show value labels")

    d_path = panel_output_paths(output_dir, figure_id, "D")["panel_data"]
    if "D" in panels and d_path.exists():
        d_df = pd.read_csv(d_path)
        metrics = set(d_df.get("metric", pd.Series(dtype=str)).astype(str))
        if metrics == {"error_rate"}:
            passes.append("Fig.1D uses error rate")
        else:
            failures.append(f"Fig.1D must use error_rate, found {sorted(metrics)}")
        if "outcome_category" in d_df.columns and d_df["outcome_category"].replace("", pd.NA).dropna().any():
            failures.append("Fig.1D must not use correct/error/silent stacked composition")
        else:
            passes.append("Fig.1D does not expose stacked outcome categories")
        observed_order = list(dict.fromkeys(str(v) for v in d_df.get("condition", pd.Series(dtype=str)).dropna().tolist()))
        expected_order = ["Dynamic STSP", "u/x-shuffled", "Static-frozen"]
        if observed_order[:3] == expected_order:
            passes.append("Fig.1D condition order is Dynamic STSP -> u/x-shuffled -> Static-frozen")
        else:
            failures.append(f"Fig.1D condition order must be {expected_order}, found {observed_order[:3]}")
        d_form = str(render_metadata.get("D", {}).get("plot_form", ""))
        if d_form:
            if d_form == "point_range":
                passes.append("Fig.1D renderer is point-range, not a filled or stacked bar")
            else:
                failures.append(f"Fig.1D must render as point-range, found {d_form}")
        else:
            warnings.append("Fig.1D plot-form render metadata unavailable in check-only mode")
        if render_metadata.get("D"):
            rotations = [float(v) for v in render_metadata.get("D", {}).get("x_tick_rotations", [])]
            if rotations and all(abs(v) < 0.01 for v in rotations):
                passes.append("Fig.1D x-axis tick labels are upright")
            else:
                failures.append(f"Fig.1D x-axis tick labels must be upright, found rotations {rotations}")
            if not str(render_metadata.get("D", {}).get("x_label", "")).strip():
                passes.append("Fig.1D x-axis title is removed")
            else:
                failures.append(f"Fig.1D x-axis title must be removed, found {render_metadata.get('D', {}).get('x_label')}")
            d_ticks = set(str(v) for v in render_metadata.get("D", {}).get("x_tick_labels", []))
            if {"Dynamic", "shuffle", "Static"}.issubset(d_ticks):
                passes.append("Fig.1D uses Dynamic/shuffle/Static labels")
            else:
                failures.append(f"Fig.1D must use Dynamic/shuffle/Static labels, found {sorted(d_ticks)}")

    e_path = panel_output_paths(output_dir, figure_id, "E")["panel_data"]
    if "E" in panels and e_path.exists():
        e_df = pd.read_csv(e_path)
        if not e_df.empty and pd.to_numeric(e_df.get("value", pd.Series(dtype=float)), errors="coerce").notna().any():
            passes.append("Fig.1E error-composition data remains present")
        else:
            failures.append("Fig.1E error-composition data missing")
        visible_terms = " ".join(str(v) for col in ("condition", "trace", "metric") if col in e_df.columns for v in e_df[col].dropna().unique())
        internal_terms = ("Pred = original sample", "Pred = change (B-map)", "B-map", "Donor sample")
        if any(term in visible_terms for term in internal_terms):
            failures.append("Fig.1E exposes old/internal attribution labels")
        else:
            passes.append("Fig.1E labels avoid old/internal attribution terms")
        traces = set(str(v) for v in e_df.get("trace", pd.Series(dtype=str)).dropna().unique())
        if {"Original", "Donor", "Others"}.issubset(traces):
            passes.append("Fig.1E decomposes error trials into Original, Donor, and Others")
        else:
            failures.append(f"Fig.1E must use Original/Donor/Others traces, found {sorted(traces)}")
        conditions = set(str(v) for v in e_df.get("condition", pd.Series(dtype=str)).dropna().unique())
        if {"Dynamic baseline", "shuffle"}.issubset(conditions):
            passes.append("Fig.1E uses Dynamic baseline and shuffle conditions")
        else:
            failures.append(f"Fig.1E must use Dynamic baseline and shuffle conditions, found {sorted(conditions)}")
        metric_terms = set(str(v) for v in e_df.get("metric", pd.Series(dtype=str)).dropna().unique())
        if metric_terms == {"error_conditional_fraction"}:
            passes.append("Fig.1E metric is error-conditional fraction")
        else:
            failures.append(f"Fig.1E must not use old all-trial attribution metric, found {sorted(metric_terms)}")
        if "denominator" in e_df.columns and set(str(v) for v in e_df["denominator"].dropna().unique()) == {"error_trials"}:
            passes.append("Fig.1E denominator is explicitly error trials only")
        else:
            failures.append("Fig.1E must record error_trials as the denominator")
        if {"condition", "seed_id", "network_id", "value"}.issubset(e_df.columns):
            grouped = e_df.groupby(["condition", "seed_id", "network_id"], dropna=False)["value"].sum()
            max_dev = float((grouped - 100.0).abs().max()) if not grouped.empty else 100.0
            if max_dev <= 1e-6:
                passes.append("Fig.1E Original + Donor + Others sums to 100% within each network/condition")
            else:
                failures.append(f"Fig.1E component fractions must sum to 100% within each network/condition; max deviation={max_dev:.6g} pp")
        stats_path = panel_output_paths(output_dir, figure_id, "E")["stats"]
        if stats_path.exists():
            e_stats = read_json(stats_path)
            if str(e_stats.get("denominator", "")) == "error trials only" and str(e_stats.get("error_definition", "")) == "prediction_probe != probe_label":
                passes.append("Fig.1E stats confirm error-trial denominator and prediction correctness definition")
            else:
                failures.append("Fig.1E stats must confirm denominator=error trials only and error_definition=prediction_probe != probe_label")
            if int(e_stats.get("n_raw_source_files", 0) or 0) > 0 and e_stats.get("raw_source_used"):
                passes.append("Fig.1E stats record raw trial_predictions.csv sources")
            else:
                failures.append("Fig.1E stats must record raw trial_predictions.csv sources")
            if e_stats.get("error_trial_counts"):
                passes.append("Fig.1E stats record error-trial counts per condition")
            else:
                failures.append("Fig.1E stats must record error-trial counts per condition")
            if str(e_stats.get("donor_derivation_note", "")).strip():
                passes.append("Fig.1E stats record donor/change-target derivation")
            else:
                failures.append("Fig.1E stats must record donor/change-target derivation")
        e_form = str(render_metadata.get("E", {}).get("plot_form", ""))
        if e_form:
            if e_form == "vertical_stacked_error_composition":
                passes.append("Fig.1E renders as vertical stacked error-composition bars")
            else:
                failures.append(f"Fig.1E must render as vertical stacked error-composition bars, found {e_form}")
            if bool(render_metadata.get("E", {}).get("raw_points", False)):
                failures.append("Fig.1E must not show raw-point overlays")
            else:
                passes.append("Fig.1E has no raw-point overlays")
            if bool(render_metadata.get("E", {}).get("value_labels", False)):
                passes.append("Fig.1E has value labels")
            else:
                failures.append("Fig.1E must show value labels")
            y_label = str(render_metadata.get("E", {}).get("y_label", ""))
            if "Fraction within error trials" in y_label:
                passes.append("Fig.1E y-axis labels the error-conditional fraction")
            else:
                failures.append(f"Fig.1E y-axis must label Fraction within error trials, found {y_label}")
            e_ticks = set(str(v) for v in render_metadata.get("E", {}).get("x_tick_labels", []))
            if {"Dynamic baseline", "shuffle"}.issubset(e_ticks):
                passes.append("Fig.1E x-axis shows Dynamic baseline and shuffle")
            else:
                failures.append(f"Fig.1E x-axis must show Dynamic baseline and shuffle, found {sorted(e_ticks)}")

    panel_a = panels.get("A")
    if panel_a:
        raw_asset = panel_a.get("source") or (panel_a.get("source_mapping") or {}).get("manual_asset")
        asset_path = paper_fig_root() / str(raw_asset)
        if asset_path.exists():
            passes.append("Fig.1A manual asset exists")
        else:
            warnings.append(f"Fig.1A manual asset missing: {raw_asset}")
            a_form = str(render_metadata.get("A", {}).get("plot_form", ""))
            if a_form:
                if a_form == "blank_manual_slot":
                    passes.append("Fig.1A missing manual asset renders as blank reserved slot")
                else:
                    failures.append(f"Fig.1A missing manual asset must render blank, found {a_form}")
    for panel_id, required_n in ((spec.get("qc_requirements") or {}).get("require_n_networks") or {}).items():
        if panel_id not in panels:
            continue
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        seed_col = "seed_id" if "seed_id" in df.columns else "network_id"
        n = df[seed_col].replace("", pd.NA).dropna().nunique() if seed_col in df.columns else 0
        if n >= int(required_n):
            passes.append(f"Fig.1{panel_id}: n={n} networks/seeds available")
        else:
            warnings.append(f"Fig.1{panel_id}: expected n={required_n}, found n={n}")

    if not full_export_paths:
        warnings.append("Fig.1 visual clipping/legend-overlap checks require rendered export; check-only verifies data/spec contracts only")
    else:
        passes.append("Fig.1 rendered export available for visual readability checks")
        if "E" in render_metadata and len(panels) > 1:
            passes.append("Fig.1E is rendered in the full Fig.1 canvas")
        _check_fig1_geometry(panels, passes, warnings, failures)
        svg_path = Path(full_export_paths.get("svg", ""))
        if svg_path.exists():
            svg_text = svg_path.read_text(encoding="utf-8", errors="ignore")
            forbidden = ["Pred = original sample", "Pred = change (B-map)", "Donor sample", "Static-frozen STSP"]
            leaked = [term for term in forbidden if term in svg_text]
            if leaked:
                failures.append(f"Fig.1 SVG exposes old/internal labels: {leaked}")
            else:
                passes.append("Fig.1 SVG does not expose old/internal labels")
            if "u/x-shuf" in svg_text or "STSP mode" in svg_text:
                failures.append("Fig.1 SVG exposes forbidden abbreviated label or removed D x-axis title")
            else:
                passes.append("Fig.1 SVG does not expose u/x-shuf abbreviation or STSP mode title")
            placeholder_terms = ["Missing manual asset", "The STSP-SNN defines a direct visual-to-decision"]
            placeholder_leaked = [term for term in placeholder_terms if term in svg_text]
            if placeholder_leaked:
                failures.append(f"Fig.1A missing-asset placeholder text is visible: {placeholder_leaked}")
            else:
                passes.append("Fig.1A missing-asset slot has no placeholder text")
        passes.append("Fig.1 B-E renderers use small-panel tick, axis-label, marker, and legend controls")


def _check_fig1_geometry(panels: Mapping[str, Any], passes: list[str], warnings: list[str], failures: list[str]) -> None:
    bottom = {pid: (panels.get(pid) or {}).get("position_mm") or {} for pid in ("C", "D", "E") if pid in panels}
    if len(bottom) != 3:
        return
    def _edge(pid: str, key: str) -> float:
        return float(bottom[pid].get(key, 0.0))
    gaps = [_edge("D", "x") - (_edge("C", "x") + _edge("C", "w")), _edge("E", "x") - (_edge("D", "x") + _edge("D", "w"))]
    if min(gaps) >= 10.0:
        passes.append(f"Fig.1 C-D-E horizontal gutters are >=10 mm ({gaps[0]:.1f}, {gaps[1]:.1f})")
    else:
        failures.append(f"Fig.1 C-D-E gutters too small for label separation: {gaps}")
    heights = [_edge(pid, "h") for pid in ("C", "D", "E")]
    if max(heights) <= 50.0:
        passes.append("Fig.1 bottom-row panels are not vertically stretched")
    else:
        failures.append(f"Fig.1 bottom-row panels are too tall: {heights}")
    bottom_margin = min(115.0 - (_edge(pid, "y") + _edge(pid, "h")) for pid in ("C", "D", "E"))
    if bottom_margin >= 8.0:
        passes.append(f"Fig.1 bottom margin supports x tick labels ({bottom_margin:.1f} mm)")
    else:
        failures.append(f"Fig.1 bottom margin too small for x tick labels ({bottom_margin:.1f} mm)")
    if _edge("E", "w") > _edge("C", "w") and _edge("E", "w") > _edge("D", "w"):
        passes.append("Fig.1E is wider than C/D for grouped comparison and legend")
    else:
        warnings.append("Fig.1E is not wider than both C and D")
    b_pos = (panels.get("B") or {}).get("position_mm") or {}
    e_pos = (panels.get("E") or {}).get("position_mm") or {}
    b_size = (float(b_pos.get("w", 0.0)), float(b_pos.get("h", 0.0)))
    e_size = (float(e_pos.get("w", 0.0)), float(e_pos.get("h", 0.0)))
    if b_size == e_size and b_size != (0.0, 0.0):
        passes.append(f"Fig.1B and Fig.1E have identical panel size {b_size[0]:.1f} x {b_size[1]:.1f} mm")
    else:
        failures.append(f"Fig.1B and Fig.1E must have identical panel size, found B={b_size}, E={e_size}")
    if float(b_pos.get("x", -1.0)) == float(e_pos.get("x", -2.0)):
        passes.append("Fig.1B and Fig.1E are vertically aligned by x-position")
    else:
        failures.append(f"Fig.1B and Fig.1E must align vertically, found x={b_pos.get('x')} and x={e_pos.get('x')}")
    top_y = [float(((panels.get(pid) or {}).get("position_mm") or {}).get("y", -1.0)) for pid in ("A", "B") if pid in panels]
    bottom_y = [float(((panels.get(pid) or {}).get("position_mm") or {}).get("y", -1.0)) for pid in ("C", "D", "E") if pid in panels]
    if len(set(top_y)) <= 1 and len(set(bottom_y)) <= 1:
        passes.append("Fig.1 panel labels share clean row-wise alignment anchors")
    else:
        warnings.append("Fig.1 panel row anchors are not perfectly aligned")


def _check_fig2_specifics(
    figure_id: str,
    spec: Mapping[str, Any],
    panels: Mapping[str, Any],
    output_dir: Path,
    adapter_results: Mapping[str, AdapterResult],
    render_metadata: Mapping[str, Mapping[str, Any]],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    if figure_id != "fig2":
        return
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 135:
        passes.append("Fig.2 canvas is 165 x 135 mm")
    else:
        failures.append(f"Fig.2 canvas must be 165 x 135 mm, found {canvas}")

    _check_fig2_geometry(panels, passes, failures)

    panel_a = panels.get("A") or {}
    if panel_a.get("panel_type") == "manual_or_programmatic_schematic" and panel_a.get("data_adapter") in (None, "", "none"):
        passes.append("Fig.2A is schematic/programmatic and does not require adapter")
    else:
        failures.append("Fig.2A must be schematic/programmatic with no data adapter")
    if panel_a.get("renderer"):
        passes.append("Fig.2A renderer present")
    else:
        failures.append("Fig.2A renderer missing")
    if (panel_a.get("content") or {}).get("blank") is True:
        passes.append("Fig.2A spec reserves a blank panel area")
    else:
        failures.append("Fig.2A must be blank for this patch")
    a_form = str(render_metadata.get("A", {}).get("plot_form", ""))
    if a_form:
        if a_form == "blank_reserved_slot":
            passes.append("Fig.2A renderer leaves the panel blank")
        else:
            failures.append(f"Fig.2A must render a blank reserved slot, found {a_form}")

    for panel_id in ("B", "D"):
        panel = panels.get(panel_id) or {}
        refs = panel.get("reference_lines") or []
        if any(float(ref.get("value")) == 0 for ref in refs):
            passes.append(f"Fig.2{panel_id}: zero reference line present")
        else:
            failures.append(f"Fig.2{panel_id}: zero reference line missing")
    if "Fusion dual score" in str((panels.get("B") or {}).get("y_axis", "")):
        passes.append("Fig.2B y-axis names Fusion dual score")
    else:
        warnings.append("Fig.2B y-axis should be Fusion dual score")
    if str((panels.get("D") or {}).get("x_axis", "")) == "WPRI" and str((panels.get("D") or {}).get("y_axis", "")) == "Density":
        passes.append("Fig.2D spec names density axes as x=WPRI and y=Density")
    else:
        failures.append("Fig.2D spec must name density axes as x=WPRI and y=Density")
    if bool((panels.get("B") or {}).get("hide_x_tick_labels", False)) or str((panels.get("B") or {}).get("x_axis", "")).lower() == "none":
        passes.append("Fig.2B spec removes the Layer 3 x-axis category label")
    else:
        failures.append("Fig.2B must remove the Layer 3 x-axis category label")
    b_ticks = [str(v) for v in render_metadata.get("B", {}).get("x_tick_labels", []) if str(v).strip()]
    if b_ticks:
        if "Layer 3" in b_ticks:
            failures.append("Fig.2B render still shows Layer 3 as an x-axis category label")
        else:
            passes.append("Fig.2B render does not expose Layer 3 as an x-axis category label")
    b_meta = render_metadata.get("B", {})
    if b_meta:
        ylim = [float(v) for v in b_meta.get("ylim", [0.0, 1.0])]
        if ylim[0] <= 0.01 and ylim[1] >= 0.99:
            failures.append(f"Fig.2B must not display the unnecessary full 0-1 y-range, found {ylim}")
        else:
            passes.append(f"Fig.2B uses a tightened y-range {ylim}")
        raw_count = int(b_meta.get("raw_point_count", 0) or 0)
        raw_alpha = float(b_meta.get("raw_point_alpha", 1.0) or 1.0)
        if raw_count <= 500 or raw_alpha <= 0.1:
            passes.append(f"Fig.2B raw points are reduced/de-emphasized (count={raw_count}, alpha={raw_alpha})")
        else:
            failures.append(f"Fig.2B raw points must be reduced or de-emphasized, found count={raw_count}, alpha={raw_alpha}")
        if bool(b_meta.get("y_tick_labels_inside_axes", False)) or bool(b_meta.get("y_label_inside_axes", False)):
            failures.append("Fig.2B y-axis tick labels/label are inside the plotting region")
        else:
            passes.append("Fig.2B y-axis tick labels and label are outside the plotting region")

    for panel_id, required_n in ((spec.get("qc_requirements") or {}).get("require_n_networks") or {}).items():
        if panel_id not in panels:
            continue
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        n = _panel_n(df)
        if n >= int(required_n):
            passes.append(f"Fig.2{panel_id}: n={n} networks/seeds available")
        else:
            warnings.append(f"Fig.2{panel_id}: expected n={required_n}, found n={n}")

    _check_fig2_bcd_granularity(figure_id, output_dir, panels, passes, warnings, failures)

    panel_c = panels.get("C") or {}
    c_conditions = set(panel_c.get("conditions") or [])
    if {"True pair", "Shuffled pair"}.issubset(c_conditions):
        passes.append("Fig.2C spec includes True pair and Shuffled pair")
    else:
        failures.append("Fig.2C must include True pair and Shuffled pair conditions")
    c_form = str(render_metadata.get("C", {}).get("plot_form", ""))
    if c_form:
        if c_form == "box_plot":
            passes.append("Fig.2C renderer is box plot")
        else:
            failures.append(f"Fig.2C renderer must be box plot, found {c_form}")
    elif str(panel_c.get("renderer", "")) == "render_paired_condition_plot":
        passes.append("Fig.2C renderer is configured for box plot")
    c_meta = render_metadata.get("C", {})
    if c_meta:
        if bool(c_meta.get("raw_points", True)) or bool(c_meta.get("showfliers", True)):
            failures.append("Fig.2C must be a clean box plot without raw-point/flyer overlay")
        else:
            passes.append("Fig.2C is a clean box plot without raw-point/flyer overlay")
    c_path = panel_output_paths(output_dir, figure_id, "C")["panel_data"]
    if c_path.exists():
        c_df = pd.read_csv(c_path)
        if _panel_n(c_df) > 0:
            passes.append("Fig.2C paired network identifiers available")
        else:
            warnings.append("Fig.2C paired network identifiers unavailable")

    d_form = str(render_metadata.get("D", {}).get("plot_form", ""))
    if d_form:
        if d_form == "density_plot":
            passes.append("Fig.2D renderer is density plot")
        else:
            failures.append(f"Fig.2D renderer must be density plot, found {d_form}")
    elif str((panels.get("D") or {}).get("renderer", "")) == "render_wpri_density":
        passes.append("Fig.2D renderer is configured as a density plot")
    d_meta = render_metadata.get("D", {})
    if d_meta:
        if str(d_meta.get("x_metric", "")) == "WPRI" and str(d_meta.get("y_metric", "")) == "Density":
            passes.append("Fig.2D density axes are x=WPRI and y=Density")
        else:
            failures.append(f"Fig.2D density axes must be x=WPRI and y=Density, found x={d_meta.get('x_metric')} y={d_meta.get('y_metric')}")

    d_stats = adapter_results.get("D")
    if d_stats and any("trial-level" in msg for msg in d_stats.warnings):
        warnings.append("Fig.2D used trial-level fallback")

    e_path = panel_output_paths(output_dir, figure_id, "E")["panel_data"]
    if e_path.exists():
        e_df = pd.read_csv(e_path)
        expected = {"No memory", "Item 2 only", "Item 1->Item 2"}
        if expected.issubset(set(e_df.get("condition", []))):
            passes.append("Fig.2E panel data uses manuscript memory-state labels")
        else:
            warnings.append("Fig.2E panel data missing one or more expected memory-state labels")
        raw_labels = set(str(v) for v in e_df.get("condition", []))
        if {"S_B", "S_AB"}.intersection(raw_labels):
            warnings.append("Fig.2E final condition labels contain raw S_B/S_AB labels")
        else:
            passes.append("Fig.2E final condition labels do not expose S_B/S_AB")
        if "functional_metric" in e_df.columns and {"Pair-member readout", "Item 1 accessibility"}.issubset(set(e_df["functional_metric"])):
            passes.append("Fig.2E preferred two-metric design available")
        else:
            warnings.append("Fig.2E preferred two-metric design unavailable; fallback composition may be used")
    e_meta = render_metadata.get("E", {})
    rotations = [float(v) for v in e_meta.get("x_tick_rotations", [])]
    fontstyles = [str(v) for v in e_meta.get("x_tick_fontstyles", [])]
    if rotations or fontstyles:
        if all(abs(v) < 0.01 for v in rotations) and all(style in ("normal", "") for style in fontstyles):
            passes.append("Fig.2E x tick labels are upright/plain")
        else:
            failures.append(f"Fig.2E x tick labels must be upright/plain, found rotations={rotations}, styles={fontstyles}")
    if e_meta:
        if bool(e_meta.get("legend_overlaps_data", False)):
            failures.append("Fig.2E legend overlaps the plotted data area")
        else:
            passes.append("Fig.2E legend does not overlap the plotted data area")
        legend_texts = [str(v) for v in e_meta.get("legend_texts", [])]
        ncols = int(e_meta.get("legend_ncols", 0) or 0)
        if legend_texts and ncols >= len(legend_texts):
            passes.append("Fig.2E legend is arranged in one horizontal row")
        else:
            failures.append(f"Fig.2E legend must be one horizontal row, found ncols={ncols}, labels={legend_texts}")
        e_ticks = [str(v) for v in e_meta.get("x_tick_labels", [])]
        if any("Both" in label and "memories" in label for label in e_ticks) or "Both memories" in str(e_meta.get("third_condition_label", "")):
            passes.append("Fig.2E third condition label uses a both-memories label")
        else:
            failures.append(f"Fig.2E third condition label must indicate both memories, found {e_ticks}")
        if bool(e_meta.get("y_tick_labels_inside_axes", False)) or bool(e_meta.get("y_label_inside_axes", False)):
            failures.append("Fig.2E y-axis tick labels/label are inside the plotting region")
        else:
            passes.append("Fig.2E y-axis tick labels and label are outside the plotting region")

    f_path = panel_output_paths(output_dir, figure_id, "F")["panel_data"]
    if f_path.exists():
        f_df = pd.read_csv(f_path)
        curve = f_df[f_df.get("curve_or_summary", "").eq("curve")] if "curve_or_summary" in f_df.columns else pd.DataFrame()
        summary = f_df[f_df.get("curve_or_summary", "").eq("summary")] if "curve_or_summary" in f_df.columns else pd.DataFrame()
        if not curve.empty and "dropout_level" in curve.columns and pd.to_numeric(curve["dropout_level"], errors="coerce").notna().any():
            passes.append("Fig.2F dropout_level curve data available")
        else:
            warnings.append("Fig.2F dropout_level curve data unavailable")
        if not summary.empty:
            passes.append("Fig.2F AUC summary available")
        else:
            warnings.append("Fig.2F AUC summary unavailable")
        if curve.empty and not summary.empty:
            warnings.append("Fig.2F only AUC summary available; no dropout curve found")
        axis_metric_text = f"{(panels.get('F') or {}).get('y_axis', '')} {(panels.get('F') or {}).get('main_metric', '')}"
        if "P(pred=A)" in axis_metric_text or "Item A" in axis_metric_text or "Item B" in axis_metric_text:
            warnings.append("Fig.2F label uses A/B terminology instead of Item 1/Item 2")
        else:
            passes.append("Fig.2F labels use Item 1 terminology")
    if "F" in panels:
        f_meta = render_metadata.get("F", {})
        if f_meta:
            clipped = list(f_meta.get("clipped_artists", []))
            if clipped:
                failures.append(f"Fig.2F has clipped rendered artists: {clipped}")
            else:
                passes.append("Fig.2F remains present and not clipped")

    image_sources: list[str] = []
    for panel_id in panels:
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        if "source_file" in df.columns:
            image_sources.extend([str(v) for v in df["source_file"].dropna().unique() if str(v).lower().endswith((".png", ".pdf", ".svg"))])
    if image_sources:
        warnings.append(f"Fig.2 panel data references old source figure images: {image_sources}")
    else:
        passes.append("Fig.2 panel data does not use old source figure images")

    if render_metadata:
        label_gaps = {
            panel_id: meta.get("panel_label_gap_mm")
            for panel_id, meta in render_metadata.items()
            if meta.get("panel_label_gap_mm") is not None
        }
        if label_gaps and all(float(gap) <= 4.5 for gap in label_gaps.values()):
            passes.append("Fig.2 panel letters are positioned close to their corresponding panels")
        else:
            failures.append(f"Fig.2 panel letters must sit closer to panels, found gaps {label_gaps}")
        clipped_panels = {
            panel_id: list(meta.get("clipped_artists", []))
            for panel_id, meta in render_metadata.items()
            if meta.get("clipped_artists") or meta.get("panel_label_clipped")
        }
        if clipped_panels:
            failures.append(f"Fig.2 labels/ticks/legends/panel letters clipped: {clipped_panels}")
        else:
            passes.append("Fig.2 has no clipped labels, ticks, legends, or panel letters")
    else:
        warnings.append("Fig.2 visual clipping and legend-overlap checks require rendered export; check-only verifies spec/data contracts only")


def _check_fig2_geometry(panels: Mapping[str, Any], passes: list[str], failures: list[str]) -> None:
    pos = {panel_id: (panels.get(panel_id) or {}).get("position_mm") or {} for panel_id in ("A", "B", "C", "D", "E", "F")}
    missing = [panel_id for panel_id, value in pos.items() if not value]
    if missing:
        failures.append(f"Fig.2 missing position_mm for panels {missing}")
        return
    if _near(_w(pos["B"]), _w(pos["C"])) and _near(_w(pos["B"]), _w(pos["D"])) and _near(_w(pos["B"]), _w(pos["E"])) and _near(_h(pos["B"]), _h(pos["C"])) and _near(_h(pos["B"]), _h(pos["D"])) and _near(_h(pos["B"]), _h(pos["E"])):
        passes.append("Fig.2 B/C/D/E have identical width and height")
    else:
        failures.append("Fig.2 B/C/D/E must have identical width and height")
    if _near(_y(pos["B"]), _y(pos["C"])) and _near(_y(pos["B"]), _y(pos["D"])) and _near(_bottom(pos["B"]), _bottom(pos["C"])) and _near(_bottom(pos["B"]), _bottom(pos["D"])):
        passes.append("Fig.2 B/C/D share y position and bottom edge")
    else:
        failures.append("Fig.2 B/C/D must share y position and bottom edge")
    gap_bc = _x(pos["C"]) - _right(pos["B"])
    gap_cd = _x(pos["D"]) - _right(pos["C"])
    if _near(gap_bc, gap_cd, tol=0.15):
        passes.append("Fig.2 B-C and C-D gaps are equal")
    else:
        failures.append(f"Fig.2 B-C and C-D gaps must be equal, found {gap_bc:.3f} and {gap_cd:.3f} mm")
    if _near(_x(pos["F"]), _x(pos["C"])) and _near(_right(pos["F"]), _right(pos["D"])):
        passes.append("Fig.2 F aligns from C.left to D.right")
    else:
        failures.append("Fig.2 F.left must equal C.left and F.right must equal D.right")
    if _near(_x(pos["A"]), _x(pos["B"])) and _near(_right(pos["A"]), _right(pos["D"])):
        passes.append("Fig.2 A aligns from B.left to D.right")
    else:
        failures.append("Fig.2 A.left must equal B.left and A.right must equal D.right")
    if _near(_y(pos["E"]), _y(pos["F"])) and _near(_bottom(pos["E"]), _bottom(pos["F"])):
        passes.append("Fig.2 E/F share y position and bottom edge")
    else:
        failures.append("Fig.2 E/F must share y position and bottom edge")


def _check_fig2_bcd_granularity(
    figure_id: str,
    output_dir: Path,
    panels: Mapping[str, Any],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    for panel_id in ("B", "C", "D"):
        if panel_id not in panels:
            continue
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        if not paths["stats"].exists() or not paths["panel_data"].exists():
            continue
        stats = read_json(paths["stats"])
        df = pd.read_csv(paths["panel_data"])
        n_networks = int(stats.get("n_networks", _panel_n(df)) or 0)
        n_source_files = int(stats.get("n_source_files", 0) or 0)
        raw_rows = int(stats.get("raw_rows_read", 0) or 0)
        layer3_rows = int(stats.get("layer3_rows_before_aggregation", 0) or 0)
        rows_written = int(stats.get("rows_written_to_panel_data", len(df)) or 0)
        averaging = bool(stats.get("averaging_performed", True))
        preaggregated = bool(stats.get("source_appeared_preaggregated", False))

        passes.append(
            f"Fig.2{panel_id}: granularity stats n_networks={n_networks}, source_files={n_source_files}, "
            f"raw_rows={raw_rows}, layer3_rows={layer3_rows}, rows_written={rows_written}"
        )
        if n_source_files > 0:
            passes.append(f"Fig.2{panel_id}: source file count recorded")
        else:
            failures.append(f"Fig.2{panel_id}: stats must record number of source files used")
        if raw_rows > 0 and layer3_rows > 0 and rows_written > 0:
            passes.append(f"Fig.2{panel_id}: raw, Layer 3, and written row counts are recorded")
        else:
            failures.append(f"Fig.2{panel_id}: stats must record nonzero raw, Layer 3, and written row counts")
        if averaging:
            failures.append(f"Fig.2{panel_id}: adapter must not average before writing panel_data")
        else:
            passes.append(f"Fig.2{panel_id}: adapter reports no averaging performed")
        if preaggregated:
            warnings.append(f"Fig.2{panel_id}: source appeared already pre-aggregated; inspect source granularity")
        else:
            passes.append(f"Fig.2{panel_id}: source does not appear pre-aggregated")

        if panel_id == "C":
            expected_min = layer3_rows * 2
            if rows_written >= expected_min and expected_min > 0:
                passes.append("Fig.2C preserves both True pair and Shuffled pair rows for each Layer 3 source row")
            else:
                failures.append(f"Fig.2C must write two condition rows per Layer 3 source row, found {rows_written} for {layer3_rows} Layer 3 rows")
            for condition in ("True pair", "Shuffled pair"):
                n_condition = int(df[df.get("condition", pd.Series(dtype=str)).eq(condition)].shape[0])
                if n_condition > max(n_networks, 20):
                    passes.append(f"Fig.2C {condition}: row-level distribution preserved ({n_condition} rows)")
                else:
                    failures.append(f"Fig.2C {condition}: only {n_condition} rows; this looks network-averaged")
        else:
            if rows_written >= layer3_rows and layer3_rows > max(n_networks, 20):
                passes.append(f"Fig.2{panel_id}: Layer 3 row-level values preserved in panel_data")
            else:
                failures.append(f"Fig.2{panel_id}: panel_data does not show row-level preservation beyond network count")


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
    o0, ob0, o1, ot0 = [float(v) for v in outer]
    width = max(o1 - o0, 1e-9)
    height = max(ot0 - ob0, 1e-9)
    return i0 <= o0 + 0.20 * width and it0 >= ot0 - 0.20 * height


def _box_w(box: Any) -> float:
    return float(box[2]) - float(box[0]) if isinstance(box, (list, tuple)) and len(box) == 4 else 0.0


def _box_h(box: Any) -> float:
    return float(box[3]) - float(box[1]) if isinstance(box, (list, tuple)) and len(box) == 4 else 0.0


def _check_fig3_specifics(
    figure_id: str,
    spec: Mapping[str, Any],
    panels: Mapping[str, Any],
    output_dir: Path,
    adapter_results: Mapping[str, AdapterResult],
    render_metadata: Mapping[str, Mapping[str, Any]],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    if figure_id != "fig3":
        return
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 140:
        passes.append("Fig.3 canvas is 165 x 140 mm")
    else:
        failures.append(f"Fig.3 canvas must be 165 x 140 mm, found {canvas}")

    _check_fig3_geometry(panels, passes, failures)

    for panel_id, required_n in ((spec.get("qc_requirements") or {}).get("require_n_networks") or {}).items():
        if panel_id not in panels:
            continue
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        n = _panel_n(df)
        if n >= int(required_n):
            passes.append(f"Fig.3{panel_id}: n={n} networks/seeds available")
        else:
            warnings.append(f"Fig.3{panel_id}: expected n={required_n}, found n={n}")

    panel_a = panels.get("A") or {}
    refs = panel_a.get("reference_lines") or []
    if any(float(ref.get("value")) == 0 for ref in refs):
        passes.append("Fig.3A: zero reference line present")
    else:
        failures.append("Fig.3A: zero reference line missing")
    if "Fusion imbalance score" in str(panel_a.get("y_axis", "")):
        passes.append("Fig.3A y-axis names Fusion imbalance score")
    else:
        warnings.append("Fig.3A y-axis should be Fusion imbalance score")

    panel_b = panels.get("B") or {}
    if panel_b.get("panel_type") == "scatter_regression":
        passes.append("Fig.3B is specified as a relationship plot")
    else:
        failures.append("Fig.3B must be scatter_regression")
    b_path = panel_output_paths(output_dir, figure_id, "B")["panel_data"]
    if b_path.exists():
        b_df = pd.read_csv(b_path)
        has_xy = {"latent_state_bias", "readout_preference"}.issubset(b_df.columns) or {"x_value", "y_value"}.issubset(b_df.columns)
        if has_xy:
            passes.append("Fig.3B latent/readout relationship columns present")
        else:
            failures.append("Fig.3B must include latent_state_bias/readout_preference or x_value/y_value")
        if "source_record_type" in b_df.columns:
            passes.append("Fig.3B source record type documents pooling level")
        else:
            warnings.append("Fig.3B pooling/source record type not documented")
    b_stats_path = panel_output_paths(output_dir, figure_id, "B")["stats"]
    if b_stats_path.exists():
        try:
            b_stats = read_json(b_stats_path)
            if b_stats.get("correlations"):
                passes.append("Fig.3B correlation/regression stats available")
            else:
                warnings.append("Fig.3B no correlation/regression stats available")
        except Exception as exc:
            warnings.append(f"Fig.3B stats unreadable for correlation check: {exc}")
    b_meta = render_metadata.get("B", {})
    if b_meta:
        rows_before = int(b_meta.get("rows_before_renderer_aggregation", 0) or 0)
        plotted = dict(b_meta.get("plotted_x_positions_by_layer", {}) or {})
        if rows_before > 0 and plotted:
            passes.append(f"Fig.3B renderer aggregated {rows_before} rows into plotted x positions by layer: {plotted}")
        else:
            failures.append("Fig.3B must report rows before renderer aggregation and plotted x positions per layer")
        if bool(b_meta.get("repeated_x_positions_averaged", False)):
            passes.append("Fig.3B repeated x positions were averaged before plotting")
        else:
            warnings.append("Fig.3B did not detect repeated x positions to average")
        if str(b_meta.get("line_emphasis", "")) == "line_over_points" and bool(b_meta.get("has_shaded_band", False)) and not bool(b_meta.get("raw_points", True)):
            passes.append("Fig.3B emphasizes lines over points with a subtle shaded band")
        else:
            failures.append("Fig.3B must emphasize lines over points and include subtle shading")

    c_path = panel_output_paths(output_dir, figure_id, "C")["panel_data"]
    if c_path.exists():
        c_df = pd.read_csv(c_path)
        conditions = set(c_df.get("condition", []))
        if {"Latest item", "Earlier items"}.issubset(conditions):
            passes.append("Fig.3C includes Latest item and Earlier items")
        elif "Latest item" in conditions:
            warnings.append("Fig.3C only latest item is available")
        else:
            failures.append("Fig.3C must include Latest item and Earlier items")
        if c_df.get("unit", pd.Series(dtype=str)).astype(str).str.contains("normalized_mass|fraction", regex=True).any():
            passes.append("Fig.3C item-similarity mass has interpretable normalized unit")
        else:
            warnings.append("Fig.3C item-similarity mass unit is not normalized_mass/fraction")

    d_path = panel_output_paths(output_dir, figure_id, "D")["panel_data"]
    if d_path.exists():
        d_df = pd.read_csv(d_path)
        if "seen_item_hit_rate" in d_df.columns or set(d_df.get("metric", [])) == {"seen_item_hit_rate"}:
            passes.append("Fig.3D seen-item hit-rate summary available")
        else:
            warnings.append("Fig.3D only raw ping retrieval profile may be available")
    d_axis = f"{(panels.get('D') or {}).get('y_axis', '')} {(panels.get('D') or {}).get('x_axis', '')}".lower()
    if "seen" in d_axis or "neutral ping" in d_axis:
        passes.append("Fig.3D labels use seen item / neutral ping terminology")
    else:
        warnings.append("Fig.3D label should use seen item or neutral ping terminology")

    e_path = panel_output_paths(output_dir, figure_id, "E")["panel_data"]
    f_path = panel_output_paths(output_dir, figure_id, "F")["panel_data"]
    e_stages: set[Any] = set()
    f_stages: set[Any] = set()
    if e_path.exists():
        e_df = pd.read_csv(e_path)
        if {"sequence_stage", "state_center_of_mass"}.issubset(e_df.columns):
            passes.append("Fig.3E sequence_stage and state_center_of_mass available")
            e_stages = set(pd.to_numeric(e_df["sequence_stage"], errors="coerce").dropna().astype(int))
        else:
            warnings.append("Fig.3E COM trajectory absent; heatmap-only fallback may be in use")
        if "anchor drift" in str((panels.get("E") or {}).get("source_mapping", {})).lower():
            passes.append("Fig.3E anchor-drift source is documented as state COM mapping")
    if f_path.exists():
        f_df = pd.read_csv(f_path)
        lower_cols = " ".join(map(str, f_df.columns)).lower()
        lower_metrics = " ".join(map(str, f_df.get("metric", []))).lower()
        lower_sources = " ".join(map(str, f_df.get("source_file", []))).lower()
        if {"sequence_stage", "ping_center_of_mass"}.issubset(f_df.columns):
            passes.append("Fig.3F sequence_stage and ping_center_of_mass available")
            f_stages = set(pd.to_numeric(f_df["sequence_stage"], errors="coerce").dropna().astype(int))
        else:
            warnings.append("Fig.3F COM trajectory absent; retrieval-profile-only fallback may be in use")
        if any(token in (lower_cols + " " + lower_metrics + " " + lower_sources) for token in ("stepwise", "sur", "update_ratio")):
            warnings.append("Fig.3F final panel data appears to contain Stepwise update ratio / SUR")
        else:
            passes.append("Fig.3F final panel data does not contain Stepwise update ratio / SUR")
    f_mapping_text = str((panels.get("F") or {}).get("source_mapping", {})).lower()
    if "stepwise_update_ratio" in f_mapping_text:
        passes.append("Fig.3F source mapping records stepwise_update_ratio as explicitly unused")
    if e_stages and f_stages:
        if e_stages == f_stages:
            passes.append("Fig.3E/F use matched sequence-stage semantics")
        else:
            warnings.append(f"Fig.3E/F sequence stages differ: E={sorted(e_stages)}, F={sorted(f_stages)}")

    image_sources: list[str] = []
    for panel_id in panels:
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        if "source_file" in df.columns:
            image_sources.extend([str(v) for v in df["source_file"].dropna().unique() if str(v).lower().endswith((".png", ".pdf", ".svg"))])
    if image_sources:
        warnings.append(f"Fig.3 panel data references old source figure images: {image_sources}")
    else:
        passes.append("Fig.3 panel data does not use old source figure images")

    _check_fig3_row_level_granularity(figure_id, output_dir, panels, passes, warnings, failures)

    if render_metadata:
        inside = [pid for pid, meta in render_metadata.items() if meta.get("y_tick_labels_inside_axes") or meta.get("y_label_inside_axes")]
        if inside:
            failures.append(f"Fig.3 y-axis tick labels/labels inside plot areas: {inside}")
        else:
            passes.append("Fig.3 y-axis tick labels and labels are outside plot areas")
        clipped_panels = {
            panel_id: list(meta.get("clipped_artists", []))
            for panel_id, meta in render_metadata.items()
            if meta.get("clipped_artists") or meta.get("panel_label_clipped")
        }
        if clipped_panels:
            failures.append(f"Fig.3 labels/ticks/legends/panel letters clipped: {clipped_panels}")
        else:
            passes.append("Fig.3 has no clipped labels, ticks, legends, or panel letters")


def _check_fig3_geometry(panels: Mapping[str, Any], passes: list[str], failures: list[str]) -> None:
    ids = ("A", "B", "C", "D", "E", "F")
    pos = {panel_id: (panels.get(panel_id) or {}).get("position_mm") or {} for panel_id in ids}
    if any(not value for value in pos.values()):
        failures.append("Fig.3 all panels must define position_mm")
        return
    widths = [_w(pos[pid]) for pid in ids]
    heights = [_h(pos[pid]) for pid in ids]
    if max(widths) - min(widths) <= 0.05 and max(heights) - min(heights) <= 0.05:
        passes.append("Fig.3 A-F have identical width and height")
    else:
        failures.append(f"Fig.3 A-F must have identical sizes, found widths={widths}, heights={heights}")
    if _near(_y(pos["A"]), _y(pos["B"])) and _near(_y(pos["A"]), _y(pos["C"])) and _near(_bottom(pos["A"]), _bottom(pos["B"])) and _near(_bottom(pos["A"]), _bottom(pos["C"])):
        passes.append("Fig.3 A/B/C share row-1 top and bottom")
    else:
        failures.append("Fig.3 A/B/C must share top and bottom")
    if _near(_y(pos["D"]), _y(pos["E"])) and _near(_y(pos["D"]), _y(pos["F"])) and _near(_bottom(pos["D"]), _bottom(pos["E"])) and _near(_bottom(pos["D"]), _bottom(pos["F"])):
        passes.append("Fig.3 D/E/F share row-2 top and bottom")
    else:
        failures.append("Fig.3 D/E/F must share top and bottom")
    if _near(_x(pos["A"]), _x(pos["D"])) and _near(_x(pos["B"]), _x(pos["E"])) and _near(_x(pos["C"]), _x(pos["F"])):
        passes.append("Fig.3 columns align A/D, B/E, and C/F")
    else:
        failures.append("Fig.3 column left boundaries must align")
    gaps = [_x(pos["B"]) - _right(pos["A"]), _x(pos["C"]) - _right(pos["B"])]
    if _near(gaps[0], gaps[1], tol=0.15):
        passes.append("Fig.3 column gaps are equal")
    else:
        failures.append(f"Fig.3 column gaps must be equal, found {gaps}")


def _check_fig3_row_level_granularity(
    figure_id: str,
    output_dir: Path,
    panels: Mapping[str, Any],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    for panel_id in ("A", "C", "D", "E", "F"):
        if panel_id not in panels:
            continue
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        if not paths["stats"].exists() or not paths["panel_data"].exists():
            continue
        stats = read_json(paths["stats"])
        df = pd.read_csv(paths["panel_data"])
        n_networks = int(stats.get("n_networks", _panel_n(df)) or 0)
        n_source_files = int(stats.get("n_source_files", 0) or 0)
        raw_rows = int(stats.get("raw_rows_read", 0) or 0)
        layer3_rows = int(stats.get("layer3_rows_before_aggregation", 0) or 0)
        rows_written = int(stats.get("rows_written_to_panel_data", len(df)) or 0)
        averaging = bool(stats.get("averaging_performed", True))
        preaggregated = bool(stats.get("source_appeared_preaggregated", False))
        passes.append(
            f"Fig.3{panel_id}: granularity stats n_networks={n_networks}, source_files={n_source_files}, "
            f"raw_rows={raw_rows}, layer3_rows={layer3_rows}, rows_written={rows_written}"
        )
        if n_source_files <= 0 or raw_rows <= 0 or layer3_rows <= 0 or rows_written <= 0:
            failures.append(f"Fig.3{panel_id}: stats must record nonzero source, raw, Layer 3, and written row counts")
        if averaging:
            failures.append(f"Fig.3{panel_id}: adapter must not average before writing panel_data")
        else:
            passes.append(f"Fig.3{panel_id}: adapter reports no averaging performed")
        if preaggregated:
            warnings.append(f"Fig.3{panel_id}: source appeared already pre-aggregated; inspect source granularity")
        else:
            passes.append(f"Fig.3{panel_id}: source does not appear pre-aggregated")
        if panel_id == "C":
            final_rows = int(stats.get("final_stage_layer3_rows", 0) or 0)
            if rows_written > max(n_networks, 20) and final_rows > 0:
                passes.append("Fig.3C preserves trial/sequence-level latest and earlier mass rows")
            else:
                failures.append("Fig.3C does not show row-level latest/earlier mass preservation")
        elif rows_written > max(n_networks, 20):
            passes.append(f"Fig.3{panel_id}: row-level values preserved beyond network count")
        else:
            failures.append(f"Fig.3{panel_id}: panel_data does not show row-level preservation beyond network count")


def _check_fig4_specifics(
    figure_id: str,
    spec: Mapping[str, Any],
    panels: Mapping[str, Any],
    output_dir: Path,
    adapter_results: Mapping[str, AdapterResult],
    render_metadata: Mapping[str, Mapping[str, Any]],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    if figure_id != "fig4":
        return
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 130:
        passes.append("Fig.4 canvas is 165 x 130 mm")
    else:
        failures.append(f"Fig.4 canvas must be 165 x 130 mm, found {canvas}")
    _check_fig4_geometry(panels, passes, failures)

    if "A" in panels:
        panel_a = panels.get("A") or {}
        if panel_a.get("panel_type") in ("programmatic_or_manual_schematic", "manual_or_programmatic_schematic") and panel_a.get("data_adapter") in (None, "", "none"):
            passes.append("Fig.4A is schematic/programmatic and does not require adapter")
        else:
            failures.append("Fig.4A must be schematic/programmatic with no data adapter")
        if (panel_a.get("content") or {}).get("blank") is True:
            passes.append("Fig.4A spec reserves a blank slot")
        else:
            failures.append("Fig.4A must be blank for this patch")
        a_form = str(render_metadata.get("A", {}).get("plot_form", ""))
        if a_form:
            if a_form == "blank_reserved_slot":
                passes.append("Fig.4A renderer leaves the panel blank")
            else:
                failures.append(f"Fig.4A must render blank without placeholder text, found {a_form}")

    for panel_id, required_n in ((spec.get("qc_requirements") or {}).get("require_n_networks") or {}).items():
        if panel_id not in panels:
            continue
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        n = _panel_n(df)
        if n >= int(required_n):
            passes.append(f"Fig.4{panel_id}: n={n} networks/seeds available")
        else:
            warnings.append(f"Fig.4{panel_id}: expected n={required_n}, found n={n}")

    b_path = panel_output_paths(output_dir, figure_id, "B")["panel_data"]
    if "B" in panels and b_path.exists():
        b_df = pd.read_csv(b_path)
        if "similarity_bin" in b_df.columns or "similarity_bin_order" in b_df.columns:
            passes.append("Fig.4B includes sample-probe similarity bins")
        else:
            failures.append("Fig.4B must include similarity_bin or similarity_bin_order")
        if "accuracy_drop" in b_df.columns or set(b_df.get("metric", [])) == {"probe_accuracy_drop"}:
            passes.append("Fig.4B includes probe accuracy drop")
        else:
            failures.append("Fig.4B must include probe accuracy drop")
        orders = pd.to_numeric(b_df.get("similarity_bin_order", pd.Series(dtype=float)), errors="coerce").dropna().unique()
        if len(orders) == 0:
            warnings.append("Fig.4B similarity bins are unordered")
        elif len(orders) <= 2:
            warnings.append("Fig.4B has only two similarity bins; graded trend is hard to assess")
        else:
            passes.append("Fig.4B has ordered similarity bins for graded trend")
        b_meta = render_metadata.get("B", {})
        if b_meta:
            tick_labels = [str(v) for v in b_meta.get("x_tick_labels", []) if str(v).strip()]
            if bool(b_meta.get("similarity_direction_arrow", False)) and not bool(b_meta.get("literal_bin_xticklabels", True)):
                passes.append("Fig.4B uses a directional similarity arrow instead of literal bin labels")
            else:
                failures.append("Fig.4B must use a directional increasing-similarity arrow")
            if any(label.lower().startswith("bin_") for label in tick_labels):
                failures.append(f"Fig.4B must not emphasize literal bin labels on the x-axis, found {tick_labels}")
            else:
                passes.append("Fig.4B x-axis no longer emphasizes bin_1/bin_2/bin_3/bin_4 labels")
            if bool(b_meta.get("similarity_bar_order_preserved", False)):
                passes.append("Fig.4B preserves ordered bar positions for increasing similarity")
            else:
                failures.append("Fig.4B must preserve ordered similarity bar positions")
            expected_labels = int(len(orders)) if len(orders) else int(b_df["similarity_bin"].replace("", pd.NA).dropna().nunique())
            if bool(b_meta.get("value_labels", False)) and int(b_meta.get("value_label_count", 0) or 0) >= expected_labels:
                passes.append("Fig.4B has numeric value labels for each similarity bin")
            else:
                failures.append("Fig.4B must show numeric value labels above each bar/bin")
            if bool(b_meta.get("value_labels_clear", False)):
                passes.append("Fig.4B value labels are placed above error bars and the connecting line")
            else:
                failures.append("Fig.4B value labels must avoid error bars, axes, and panel labels")

    c_path = panel_output_paths(output_dir, figure_id, "C")["panel_data"]
    if "C" in panels and c_path.exists():
        c_df = pd.read_csv(c_path)
        conditions = set(c_df.get("condition", []))
        if {"Low overlap", "High overlap"}.issubset(conditions):
            passes.append("Fig.4C includes Low overlap and High overlap conditions")
        else:
            failures.append("Fig.4C must include Low overlap and High overlap conditions")
        if "accuracy_drop" in c_df.columns or set(c_df.get("metric", [])) == {"probe_accuracy_drop"}:
            passes.append("Fig.4C uses probe accuracy drop")
        else:
            failures.append("Fig.4C must use probe accuracy drop or equivalent metric")
        if "similarity_regime" in c_df.columns and c_df["similarity_regime"].astype(str).str.contains("high", case=False, na=False).any():
            passes.append("Fig.4C documents high-similarity regime")
        else:
            warnings.append("Fig.4C does not document restriction to the high-similarity regime")
        if _panel_n(c_df) > 0:
            passes.append("Fig.4C paired network identifiers available")
        else:
            warnings.append("Fig.4C paired network identifiers unavailable")
        c_meta = render_metadata.get("C", {})
        if c_meta:
            expected_labels = int(c_df["condition"].replace("", pd.NA).dropna().nunique())
            if bool(c_meta.get("value_labels", False)) and int(c_meta.get("value_label_count", 0) or 0) >= expected_labels:
                passes.append("Fig.4C has numeric value labels for each bar")
            else:
                failures.append("Fig.4C must show numeric value labels above each bar")
            if bool(c_meta.get("value_labels_clear", False)):
                passes.append("Fig.4C value labels are placed above error bars")
            else:
                failures.append("Fig.4C value labels must avoid error bars, axes, and panel labels")

    if "D" in panels:
        panel_d = panels.get("D") or {}
        refs = panel_d.get("reference_lines") or []
        if any(float(ref.get("value")) == 0 for ref in refs):
            passes.append("Fig.4D zero reference line present")
        else:
            failures.append("Fig.4D must include reference line at 0")
    d_path = panel_output_paths(output_dir, figure_id, "D")["panel_data"]
    if "D" in panels and d_path.exists():
        d_df = pd.read_csv(d_path)
        d_conditions = set(d_df.get("condition", []))
        if {"Overlap-preserving", "Non-overlap control"}.issubset(d_conditions):
            passes.append("Fig.4D includes overlap-preserving and non-overlap control conditions")
        else:
            failures.append("Fig.4D must include Overlap-preserving and Non-overlap control")
        if "probe_time" in d_df.columns or "time_ms" in d_df.columns:
            passes.append("Fig.4D includes probe_time/time_ms timecourse data")
        else:
            warnings.append("Fig.4D only summary DPI data are available; no timecourse found")
        if "dynamic_probe_index" in d_df.columns or set(d_df.get("metric", [])) == {"dynamic_probe_index"}:
            passes.append("Fig.4D includes dynamic-probe index")
        else:
            failures.append("Fig.4D must include dynamic_probe_index or equivalent")
        if d_df.get("unit", pd.Series(dtype=str)).astype(str).str.lower().eq("percent").any():
            warnings.append("Fig.4D dynamic-probe index unit is mislabeled as percent")
        d_meta = render_metadata.get("D", {})
        if d_meta:
            xlim = [float(v) for v in d_meta.get("xlim", [])]
            if len(xlim) == 2 and abs(xlim[0] - 0.0) <= 0.25 and abs(xlim[1] - 50.0) <= 0.25:
                passes.append("Fig.4D displays only 0-50 ms")
            else:
                failures.append(f"Fig.4D must display 0-50 ms, found xlim={xlim}")
            shaded = [float(v) for v in d_meta.get("shaded_window", [])]
            if shaded == [0.0, 20.0]:
                passes.append("Fig.4D shades the 0-20 ms early window")
            else:
                failures.append(f"Fig.4D must shade 0-20 ms, found {shaded}")
            shade_color = str(d_meta.get("shaded_window_color", "")).lower()
            shade_alpha = float(d_meta.get("shaded_window_alpha", 0.0) or 0.0)
            if shade_color in {"#fde68a", "#ffe08a", "#ffec99"} and 0.30 <= shade_alpha <= 0.55:
                passes.append("Fig.4D early window uses clearly visible light-yellow transparent shading")
            else:
                failures.append(f"Fig.4D early-window shading must be visible light yellow, found color={shade_color}, alpha={shade_alpha}")
            if d_meta.get("peak_annotations"):
                passes.append("Fig.4D annotates peak values within the displayed window")
            else:
                failures.append("Fig.4D must annotate peak values within the 0-50 ms window")

    e_path = panel_output_paths(output_dir, figure_id, "E")["panel_data"]
    if "E" in panels and e_path.exists():
        e_df = pd.read_csv(e_path)
        e_conditions = set(str(v) for v in e_df.get("condition", []))
        old_conditions = {"Overlap-preserving", "Non-overlap control"}
        if old_conditions.isdisjoint(e_conditions):
            passes.append("Fig.4E is no longer the old two-category recovery plot")
        else:
            failures.append("Fig.4E must not remain the old overlap-preserving vs non-overlap recovery plot")
        if set(e_df.get("metric", [])) == {"static_dynamic_manipulation_trajectory"}:
            passes.append("Fig.4E uses the static/dynamic manipulation trajectory metric")
        else:
            failures.append(f"Fig.4E must use static_dynamic_manipulation_trajectory, found {sorted(set(e_df.get('metric', [])))}")
        if {"plus", "minus"}.issubset(set(str(v) for v in e_df.get("group", []))):
            passes.append("Fig.4E panel_data includes plus/minus trajectory groups")
        else:
            failures.append("Fig.4E panel_data must include plus and minus groups")
        required_cols = {"pair_id", "x0", "y0", "x1", "y1", "before_x", "before_y", "after_x", "after_y"}
        missing_cols = sorted(required_cols - set(e_df.columns))
        if not missing_cols:
            passes.append("Fig.4E panel_data includes static/dynamic trajectory coordinates")
        else:
            failures.append(f"Fig.4E panel_data missing trajectory coordinate columns {missing_cols}")
        if e_df.get("source_file", pd.Series(dtype=str)).astype(str).str.lower().str.endswith((".png", ".pdf", ".svg")).any():
            failures.append("Fig.4E must not use old rendered source images")
        sources_path = panel_output_paths(output_dir, figure_id, "E")["sources"]
        if sources_path.exists():
            e_sources = read_json(sources_path)
            source_text = " ".join(str(src.get("path", "")) for src in e_sources.get("sources", []))
            if "pair_results.csv" in source_text and "pair_vectors.npz" in source_text:
                passes.append("Fig.4E source manifest includes pair_results.csv and pair_vectors.npz")
            else:
                failures.append("Fig.4E source manifest must include pair_results.csv and pair_vectors.npz")
    e_meta = render_metadata.get("E", {})
    if e_meta:
        if (
            str(e_meta.get("plot_form", "")) == "static_dynamic_trajectory"
            and bool(e_meta.get("static_dynamic_trajectory", False))
            and "l3_accumulator_mechanism_experiment_plot" in str(e_meta.get("trajectory_logic_source", ""))
            and bool(e_meta.get("mean_arrows", False))
            and bool(e_meta.get("individual_traces", False))
            and bool(e_meta.get("axis_direction_annotations", False))
            and not bool(e_meta.get("is_two_category_paired_recovery", True))
        ):
            passes.append("Fig.4E uses the static/dynamic trajectory renderer based on l3 accumulator mechanism logic")
        else:
            failures.append("Fig.4E must use the intended static/dynamic trajectory renderer")
        if str(e_meta.get("aspect", "")).lower() == "auto" and not bool(e_meta.get("forced_equal_aspect", True)):
            passes.append("Fig.4E is not forced to a square/equal-aspect plotting region")
        else:
            failures.append(f"Fig.4E must not use forced equal aspect, found aspect={e_meta.get('aspect')}")
        if bool(e_meta.get("normal_rectangular_panel", False)):
            passes.append("Fig.4E uses the normal rectangular panel box style")
        else:
            failures.append("Fig.4E must render as a normal rectangular panel, not a square inset")
        axes_bbox = e_meta.get("axes_bounds", [])
        d_axes = render_metadata.get("D", {}).get("axes_bounds", [])
        if axes_bbox and d_axes and _near(float(axes_bbox[1]), float(d_axes[1]), tol=0.002) and _near(float(axes_bbox[3]), float(d_axes[3]), tol=0.002):
            passes.append("Fig.4E plotting axes top and bottom align with Fig.4D axes")
        else:
            failures.append(f"Fig.4E axes region must align with Fig.4D axes region, found E={axes_bbox}, D={d_axes}")
        if axes_bbox and d_axes and _near(_box_h(axes_bbox), _box_h(d_axes), tol=0.002):
            passes.append("Fig.4E plotting axes height matches Fig.4D")
        else:
            failures.append(f"Fig.4E plotting axes height must match Fig.4D, found E={axes_bbox}, D={d_axes}")
        e_pos = (panels.get("E") or {}).get("position_mm") or {}
        d_pos = (panels.get("D") or {}).get("position_mm") or {}
        if e_pos and d_pos and _near(_w(e_pos), _w(d_pos)) and _near(_h(e_pos), _h(d_pos)):
            passes.append("Fig.4E panel box size matches the other normal panels")
        else:
            failures.append("Fig.4E panel box must match the B/C/D panel size")
        role_bboxes = e_meta.get("role_bboxes", {}) if isinstance(e_meta.get("role_bboxes", {}), dict) else {}
        y_dir_bbox = role_bboxes.get("e_y_direction_label", [])
        x_dir_bbox = role_bboxes.get("e_x_direction_label", [])
        legend_bbox = e_meta.get("legend_bbox", [])
        panel_label_bbox = e_meta.get("panel_label_bbox", [])
        if bool(e_meta.get("e_y_annotation_outside_plot", False)) and y_dir_bbox and axes_bbox and not _boxes_overlap(y_dir_bbox, axes_bbox):
            passes.append("Fig.4E y-direction annotation is outside the plotting area")
        else:
            failures.append("Fig.4E y-direction annotation must sit outside the plotting area")
        if y_dir_bbox and d_axes and not _boxes_overlap(y_dir_bbox, d_axes):
            passes.append("Fig.4E y-direction annotation does not overlap Panel D")
        else:
            failures.append("Fig.4E y-direction annotation must not overlap Panel D")
        if float(e_meta.get("e_legend_fontsize", 0.0) or 0.0) >= 4.6 and bool(e_meta.get("e_legend_markers_enlarged", False)):
            passes.append("Fig.4E legend uses enlarged, readable type and markers")
        else:
            failures.append("Fig.4E legend must be enlarged for readability")
        if bool(e_meta.get("e_legend_repositioned_inside_panel", False)) and bool(e_meta.get("e_legend_inside_axes", False)) and legend_bbox and axes_bbox and _box_inside(legend_bbox, axes_bbox):
            passes.append("Fig.4E legend is inside the plotting axes")
        else:
            failures.append("Fig.4E legend must be inside the plotting axes")
        if bool(e_meta.get("e_legend_upper_left", False)) and legend_bbox and axes_bbox and _box_in_upper_left(legend_bbox, axes_bbox):
            passes.append("Fig.4E legend is in the upper-left of the axes")
        else:
            failures.append("Fig.4E legend must sit in the upper-left corner of the axes")
        if legend_bbox and y_dir_bbox and not _boxes_overlap(legend_bbox, y_dir_bbox):
            passes.append("Fig.4E legend does not overlap the y-direction annotation")
        else:
            failures.append("Fig.4E legend must not overlap the y-direction annotation")
        if legend_bbox and x_dir_bbox and not _boxes_overlap(legend_bbox, x_dir_bbox):
            passes.append("Fig.4E legend does not overlap the x-direction annotation")
        else:
            failures.append("Fig.4E legend must not overlap the x-direction annotation")
        if legend_bbox and panel_label_bbox and not _boxes_overlap(legend_bbox, panel_label_bbox):
            passes.append("Fig.4E legend does not overlap the panel label")
        else:
            failures.append("Fig.4E legend must not overlap panel label E")

    forbidden = ("early recruitment", "winner", "loser", "inhibition", "local competition")
    claim_text = " ".join(str((panel or {}).get("claim", "")) for panel in panels.values()).lower()
    leaked = [term for term in forbidden if term in claim_text]
    if leaked:
        warnings.append(f"Fig.4 claims include Fig.5-specific terms: {leaked}")
    else:
        passes.append("Fig.4 claims do not include Fig.5 spike-recruitment/local-competition terms")

    raw_condition_tokens = ("sample_keep_", "full_dynamic", "full_static")
    raw_label_panels: list[str] = []
    image_sources: list[str] = []
    for panel_id in panels:
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        condition_text = " ".join(map(str, df.get("condition", []))).lower()
        if any(token in condition_text for token in raw_condition_tokens):
            raw_label_panels.append(panel_id)
        if "source_file" in df.columns:
            image_sources.extend([str(v) for v in df["source_file"].dropna().unique() if str(v).lower().endswith((".png", ".pdf", ".svg"))])
    if raw_label_panels:
        warnings.append(f"Fig.4 final condition labels expose internal code names in panels {raw_label_panels}")
    else:
        passes.append("Fig.4 final condition labels do not expose internal code names")
    if image_sources:
        warnings.append(f"Fig.4 panel data references old source figure images: {image_sources}")
    else:
        passes.append("Fig.4 panel data does not use old source figure images")

    if render_metadata:
        inside = [pid for pid, meta in render_metadata.items() if meta.get("y_tick_labels_inside_axes") or meta.get("y_label_inside_axes")]
        if inside:
            failures.append(f"Fig.4 y-axis tick labels/labels inside plot areas: {inside}")
        else:
            passes.append("Fig.4 y-axis tick labels and labels are outside plot areas")
        legend_overlap = [pid for pid, meta in render_metadata.items() if meta.get("legend_overlaps_data")]
        if legend_overlap:
            failures.append(f"Fig.4 legends overlap data in panels {legend_overlap}")
        else:
            passes.append("Fig.4 legends do not overlap data")
        clipped_panels = {
            panel_id: list(meta.get("clipped_artists", []))
            for panel_id, meta in render_metadata.items()
            if meta.get("clipped_artists") or meta.get("panel_label_clipped")
        }
        if clipped_panels:
            failures.append(f"Fig.4 labels/ticks/legends/panel letters clipped: {clipped_panels}")
        else:
            passes.append("Fig.4 has no clipped labels, ticks, legends, or panel letters")


def _check_fig4_geometry(panels: Mapping[str, Any], passes: list[str], failures: list[str]) -> None:
    ids = ("B", "C", "D", "E")
    pos = {panel_id: (panels.get(panel_id) or {}).get("position_mm") or {} for panel_id in ("A", *ids)}
    if any(not pos[panel_id] for panel_id in ids):
        failures.append("Fig.4 B/C/D/E must define position_mm")
        return
    widths = [_w(pos[pid]) for pid in ids]
    heights = [_h(pos[pid]) for pid in ids]
    if pos.get("A") and _near(_x(pos["A"]), _x(pos["B"])) and _near(_x(pos["A"]), _x(pos["D"])) and _near(_right(pos["A"]), _right(pos["C"])) and _near(_right(pos["A"]), _right(pos["E"])):
        passes.append("Fig.4 A/B/D left edges and A/C/E right edges align")
    elif pos.get("A"):
        failures.append("Fig.4 A.left must align with B/D and A.right must align with C/E")
    if pos.get("A") and _h(pos["A"]) >= 30.0:
        passes.append("Fig.4A has an adequately tall blank top slot")
    elif pos.get("A"):
        failures.append("Fig.4A blank top slot must be tall enough for the three-row layout")
    if max(widths) - min(widths) <= 0.05 and max(heights) - min(heights) <= 0.05:
        passes.append("Fig.4 B/C/D/E have identical width and height")
    else:
        failures.append(f"Fig.4 B/C/D/E must have identical sizes, found widths={widths}, heights={heights}")
    if _near(_x(pos["B"]), _x(pos["D"])):
        passes.append("Fig.4 B and D share the same left boundary")
    else:
        failures.append("Fig.4 B.left must equal D.left")
    if _near(_right(pos["C"]), _right(pos["E"])):
        passes.append("Fig.4 C and E share the same right boundary")
    else:
        failures.append("Fig.4 C.right must equal E.right")
    if _near(_y(pos["B"]), _y(pos["C"])) and _near(_bottom(pos["B"]), _bottom(pos["C"])):
        passes.append("Fig.4 B/C row alignment is correct")
    else:
        failures.append("Fig.4 B/C must share row top and bottom")
    if _near(_y(pos["D"]), _y(pos["E"])) and _near(_bottom(pos["D"]), _bottom(pos["E"])):
        passes.append("Fig.4 D/E row alignment is correct")
    else:
        failures.append("Fig.4 D/E must share row top and bottom")
    gap_top = _x(pos["C"]) - _right(pos["B"])
    gap_bottom = _x(pos["E"]) - _right(pos["D"])
    if _near(gap_top, gap_bottom, tol=0.15):
        passes.append("Fig.4 B/C and D/E gaps are equal")
    else:
        failures.append(f"Fig.4 row gaps must match, found {gap_top:.3f} and {gap_bottom:.3f} mm")


def _check_fig5_specifics(
    figure_id: str,
    spec: Mapping[str, Any],
    panels: Mapping[str, Any],
    output_dir: Path,
    adapter_results: Mapping[str, AdapterResult],
    render_metadata: Mapping[str, Mapping[str, Any]],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    if figure_id != "fig5":
        return
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 130:
        passes.append("Fig.5 canvas is 165 x 130 mm")
    else:
        failures.append(f"Fig.5 canvas must be 165 x 130 mm, found {canvas}")
    _check_fig5_geometry(panels, render_metadata, passes, failures)

    for panel_id, required_n in ((spec.get("qc_requirements") or {}).get("require_n_networks") or {}).items():
        if panel_id not in panels:
            continue
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        n = _panel_n(df)
        if n >= int(required_n):
            passes.append(f"Fig.5{panel_id}: n={n} networks/seeds available")
        else:
            warnings.append(f"Fig.5{panel_id}: expected n={required_n}, found n={n}")

    a_path = panel_output_paths(output_dir, figure_id, "A")["panel_data"]
    if "A" in panels and a_path.exists():
        a_df = pd.read_csv(a_path)
        image_types = set(a_df.get("image_type", []))
        required = {"sample_mask", "probe_mask", "overlap_mask", "probe_only_mask", "ux_map_pre_dynamic"}
        if required.issubset(image_types):
            passes.append("Fig.5A includes sample/probe/overlap/probe-only/support-map context")
        elif "ux_map_pre_dynamic" in image_types:
            warnings.append("Fig.5A appears to include only a heatmap without complete sample/probe/overlap context")
        else:
            warnings.append("Fig.5A support map missing; schematic placeholder may be used")
        if "placeholder_reason" in a_df.columns and a_df["placeholder_reason"].astype(str).str.len().gt(0).any():
            warnings.append("Fig.5A schematic placeholder used because support-map data are missing")
        else:
            passes.append("Fig.5A support map data available")
        a_meta = render_metadata.get("A", {})
        if a_meta:
            if bool(a_meta.get("colorbar_removed", False)) and not bool(a_meta.get("has_colorbar", False)):
                passes.append("Fig.5A has no colorbar")
            else:
                failures.append("Fig.5A colorbar must be removed")
            if bool(a_meta.get("support_map_uncropped", False)):
                passes.append("Fig.5A support map is rendered as a complete uncropped image")
            else:
                failures.append("Fig.5A support map must remain fully visible and uncropped")

    b_path = panel_output_paths(output_dir, figure_id, "B")["panel_data"]
    if "B" in panels and b_path.exists():
        b_df = pd.read_csv(b_path)
        conditions = set(b_df.get("condition", []))
        if {"Overlap-aligned", "Probe-only"}.issubset(conditions):
            passes.append("Fig.5B includes Overlap-aligned and Probe-only conditions")
        else:
            failures.append("Fig.5B must include Overlap-aligned and Probe-only conditions")
        if "pre_probe_stsp_support" in set(b_df.get("metric", [])) or "support_region" in b_df.columns:
            passes.append("Fig.5B includes pre-probe STSP support")
        else:
            failures.append("Fig.5B must include pre-probe STSP support or equivalent")
        if _panel_n(b_df) > 0:
            passes.append("Fig.5B paired network identifiers available")
        else:
            warnings.append("Fig.5B paired network identifiers unavailable")
        units = set(b_df.get("unit", pd.Series(dtype=str)).astype(str).str.lower())
        if units.intersection({"support", "score"}):
            passes.append("Fig.5B support unit is explicit")
        else:
            warnings.append("Fig.5B support unit is ambiguous")
        b_meta = render_metadata.get("B", {})
        if b_meta:
            if bool(b_meta.get("bar_connector_removed", False)) and int(b_meta.get("bar_connector_lines_remaining", 0) or 0) == 0:
                passes.append("Fig.5B has no connecting line between bar tops")
            else:
                failures.append("Fig.5B must not show a connecting line between the two bars")
            if bool(b_meta.get("value_labels", False)) and int(b_meta.get("value_label_count", 0) or 0) >= 2:
                passes.append("Fig.5B has numeric labels above both bars")
            else:
                failures.append("Fig.5B must show numeric value labels above both bars")
            rotations = [float(v) for v in b_meta.get("x_tick_rotations", [])]
            fontstyles = [str(v).lower() for v in b_meta.get("x_tick_fontstyles", [])]
            if rotations and all(abs(v) < 0.01 for v in rotations) and fontstyles and all(v == "normal" for v in fontstyles):
                passes.append("Fig.5B x-axis tick labels are upright normal text")
            else:
                failures.append(f"Fig.5B x-axis tick labels must be upright normal text, found rotations={rotations}, fontstyles={fontstyles}")

    c_path = panel_output_paths(output_dir, figure_id, "C")["panel_data"]
    if "C" in panels and c_path.exists():
        c_df = pd.read_csv(c_path)
        conditions = set(c_df.get("condition", []))
        if {"Overlap-dominant units", "Probe-only-dominant units"}.issubset(conditions):
            passes.append("Fig.5C includes overlap/probe-only dominant unit groups")
        else:
            failures.append("Fig.5C must include Overlap-dominant units and Probe-only-dominant units")
        if "advanced_plus_recruited_fraction" in c_df.columns or "advanced_plus_recruited_fraction" in set(c_df.get("metric", [])):
            passes.append("Fig.5C preferred advanced_plus_recruited_fraction metric available")
        else:
            warnings.append("Fig.5C preferred metric unavailable; composition fallback may be used")
        if "donut" in " ".join(map(str, c_df.get("source_file", []))).lower() or "composition_fallback" in " ".join(map(str, c_df.columns)).lower():
            warnings.append("Fig.5C only donut/composition fallback may be available")
        raw_conditions = " ".join(map(str, c_df.get("condition", []))).lower()
        if any(token in raw_conditions for token in ("overlap_dominant", "probe_only_dominant", "advance", "recruit")):
            warnings.append("Fig.5C final condition labels expose internal transition labels")
        else:
            passes.append("Fig.5C final condition labels are manuscript-friendly")
        units = set(c_df.get("unit", pd.Series(dtype=str)).astype(str).str.lower())
        if units.intersection({"percent", "fraction"}):
            passes.append("Fig.5C fraction/percent unit is explicit")
        else:
            warnings.append("Fig.5C fraction/percent unit is ambiguous")
        c_meta = render_metadata.get("C", {})
        b_meta = render_metadata.get("B", {})
        if c_meta:
            if bool(c_meta.get("category_labels_wrapped", False)):
                passes.append("Fig.5C category labels are wrapped instead of intruding into Panel B")
            else:
                failures.append("Fig.5C category labels must be wrapped or otherwise reflowed")
            b_axes = b_meta.get("axes_bounds", []) if b_meta else []
            c_label_boxes = c_meta.get("y_tick_bboxes", [])
            if b_axes and c_label_boxes and not any(_boxes_overlap(box, b_axes) for box in c_label_boxes):
                passes.append("Fig.5C category labels do not overlap Panel B")
            else:
                failures.append("Fig.5C category labels must not overlap Panel B")
            if bool(c_meta.get("legend_above_plot", False)) and int(c_meta.get("legend_ncols", 0) or 0) >= 3:
                passes.append("Fig.5C legend is a single row above the plot area")
            else:
                failures.append("Fig.5C legend must be arranged as a single row above the plot area")
            c_xlim = [float(v) for v in c_meta.get("xlim", [])]
            if len(c_xlim) == 2 and abs(c_xlim[0]) <= 0.01 and 9.5 <= c_xlim[1] <= 10.5:
                passes.append("Fig.5C x-axis is compressed to about 0-10%")
            else:
                failures.append(f"Fig.5C x-axis must be displayed at about 0-10%, found xlim={c_xlim}")

    panel_d = panels.get("D") or {}
    if "D" in panels:
        refs = panel_d.get("reference_lines") or []
        if any(float(ref.get("x_value", 9999)) == 0 for ref in refs):
            passes.append("Fig.5D winner-spike vertical reference line specified")
        else:
            failures.append("Fig.5D must include vertical reference line at winner spike time")
    d_path = panel_output_paths(output_dir, figure_id, "D")["panel_data"]
    if "D" in panels and d_path.exists():
        d_df = pd.read_csv(d_path)
        if "time_from_winner_spike" in d_df.columns or "time_ms" in d_df.columns:
            passes.append("Fig.5D includes event-aligned time columns")
        else:
            warnings.append("Fig.5D only summary data are available; no event-aligned traces found")
        has_voltage = "winner_loser_voltage_difference" in d_df.columns or "winner_loser_voltage_difference" in set(d_df.get("metric", []))
        has_inhibition = "local_inhibition_change" in d_df.columns or "local_inhibition_change" in set(d_df.get("metric", []))
        if has_voltage and has_inhibition:
            passes.append("Fig.5D includes voltage-difference and inhibition-change data")
        else:
            failures.append("Fig.5D must include winner_loser_voltage_difference and local_inhibition_change")
        if "fallback_summary_only" in d_df.columns and d_df["fallback_summary_only"].astype(str).str.lower().eq("true").any():
            warnings.append("Fig.5D rendered from summary-only fallback")
        elif "time_from_winner_spike" in d_df.columns:
            passes.append("Fig.5D has timecourse data, not a single-bar-only fallback")
        d_meta = render_metadata.get("D", {})
        if d_meta:
            if bool(d_meta.get("legend_above_plot", False)) and int(d_meta.get("legend_ncols", 0) or 0) >= 2:
                passes.append("Fig.5D legend is a single row above the plot area")
            else:
                failures.append("Fig.5D legend must be a single row above the plot area")

    e_path = panel_output_paths(output_dir, figure_id, "E")["panel_data"]
    if "E" in panels and e_path.exists():
        e_df = pd.read_csv(e_path)
        categories = set(e_df.get("condition", []))
        expected = {"Winner boost", "Loser suppression", "Full winner-loser sequence"}
        if expected.issubset(categories):
            passes.append("Fig.5E includes winner boost, loser suppression, and full sequence fractions")
        else:
            failures.append("Fig.5E must include Winner boost, Loser suppression, and Full winner-loser sequence")
        if "Full winner-loser sequence" not in categories:
            warnings.append("Fig.5E full winner-loser sequence category is missing")
        units = set(e_df.get("unit", pd.Series(dtype=str)).astype(str).str.lower())
        if units.intersection({"percent", "fraction"}):
            passes.append("Fig.5E event fraction unit is explicit")
        else:
            warnings.append("Fig.5E event fraction unit is ambiguous")
        e_meta = render_metadata.get("E", {})
        if e_meta:
            if str(e_meta.get("plot_form", "")) == "event_fraction_bar_chart":
                passes.append("Fig.5E renders as a bar-chart summary")
            else:
                failures.append(f"Fig.5E must render as a bar chart, found {e_meta.get('plot_form')}")
            if bool(e_meta.get("value_labels", False)) and int(e_meta.get("value_label_count", 0) or 0) >= 3:
                passes.append("Fig.5E has numeric labels above bars")
            else:
                failures.append("Fig.5E must show numeric value labels above bars")

    fig4_terms = ("similarity bin", "dynamic-probe index", "final readout recovery")
    fig6_terms = ("peak membership", "recency", "anchor prediction", "peak flattening", "peak boosting")
    claim_text = " ".join(str((panel or {}).get("claim", "")) for panel in panels.values()).lower()
    leaked_fig4 = [term for term in fig4_terms if term in claim_text]
    leaked_fig6 = [term for term in fig6_terms if term in claim_text]
    if leaked_fig4:
        warnings.append(f"Fig.5 claims include Fig.4-specific terms: {leaked_fig4}")
    else:
        passes.append("Fig.5 claims do not repeat Fig.4 similarity/DPI/readout-recovery claims")
    if leaked_fig6:
        warnings.append(f"Fig.5 claims include Fig.6-specific terms: {leaked_fig6}")
    else:
        passes.append("Fig.5 claims do not include Fig.6 peak/recency/anchor terms")

    raw_label_tokens = ("overlap_dominant", "probe_only_dominant", "fig4_panel", "winner_pre_spike_boost", "full_chain_satisfied")
    raw_label_panels: list[str] = []
    image_sources: list[str] = []
    for panel_id in panels:
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        condition_text = " ".join(map(str, df.get("condition", []))).lower()
        if any(token in condition_text for token in raw_label_tokens):
            raw_label_panels.append(panel_id)
        if "source_file" in df.columns:
            image_sources.extend([str(v) for v in df["source_file"].dropna().unique() if str(v).lower().endswith((".png", ".pdf", ".svg"))])
    if raw_label_panels:
        warnings.append(f"Fig.5 final condition labels expose old/internal code names in panels {raw_label_panels}")
    else:
        passes.append("Fig.5 final condition labels do not expose old code names")
    if image_sources:
        warnings.append(f"Fig.5 panel data references old source figure images: {image_sources}")
    else:
        passes.append("Fig.5 panel data does not use old source figure images")

    if render_metadata:
        inside = [pid for pid, meta in render_metadata.items() if meta.get("y_tick_labels_inside_axes") or meta.get("y_label_inside_axes")]
        if inside:
            failures.append(f"Fig.5 y-axis tick labels/labels inside plot areas: {inside}")
        else:
            passes.append("Fig.5 y-axis tick labels and labels are outside plot areas")
        legend_overlap = [pid for pid, meta in render_metadata.items() if meta.get("legend_overlaps_data")]
        if legend_overlap:
            failures.append(f"Fig.5 legends overlap data in panels {legend_overlap}")
        else:
            passes.append("Fig.5 legends do not overlap data")
        clipped_panels = {
            panel_id: list(meta.get("clipped_artists", []))
            for panel_id, meta in render_metadata.items()
            if meta.get("clipped_artists") or meta.get("panel_label_clipped")
        }
        if clipped_panels:
            failures.append(f"Fig.5 labels/ticks/legends/panel letters clipped: {clipped_panels}")
        else:
            passes.append("Fig.5 has no clipped labels, ticks, legends, or panel letters")


def _check_fig5_geometry(panels: Mapping[str, Any], render_metadata: Mapping[str, Mapping[str, Any]], passes: list[str], failures: list[str]) -> None:
    ids = ("A", "B", "C", "D", "E")
    pos = {panel_id: (panels.get(panel_id) or {}).get("position_mm") or {} for panel_id in ids}
    if any(not pos[panel_id] for panel_id in ids):
        failures.append("Fig.5 A-E must define position_mm")
        return
    if _near(_y(pos["A"]), _y(pos["B"]), tol=0.15) and _near(_y(pos["A"]), _y(pos["C"]), tol=0.15):
        passes.append("Fig.5 A/B/C share a common top edge")
    else:
        failures.append("Fig.5 A/B/C must share a common top edge")
    if _near(_bottom(pos["A"]), _bottom(pos["B"]), tol=0.15) and _near(_bottom(pos["A"]), _bottom(pos["C"]), tol=0.15):
        passes.append("Fig.5 A/B/C share a common bottom edge")
    else:
        failures.append("Fig.5 A/B/C must share a common bottom edge")
    top_widths = [_w(pos[pid]) for pid in ("A", "B", "C")]
    top_heights = [_h(pos[pid]) for pid in ("A", "B", "C")]
    if max(top_widths) - min(top_widths) <= 0.15 and max(top_heights) - min(top_heights) <= 0.15:
        passes.append("Fig.5 A/B/C share the same top-row panel size")
    else:
        failures.append(f"Fig.5 A/B/C must share the same panel size, found widths={top_widths}, heights={top_heights}")
    if _near(_y(pos["D"]), _y(pos["E"]), tol=0.15):
        passes.append("Fig.5 D/E share a common top edge")
    else:
        failures.append("Fig.5 D/E must share a common top edge")
    if _near(_bottom(pos["D"]), _bottom(pos["E"]), tol=0.15):
        passes.append("Fig.5 D/E share a common bottom edge")
    else:
        failures.append("Fig.5 D/E must share a common bottom edge")
    if _near(_x(pos["A"]), _x(pos["D"]), tol=0.15):
        passes.append("Fig.5 A.left aligns with D.left")
    else:
        failures.append("Fig.5 A.left must align with D.left")
    if _near(_right(pos["C"]), _right(pos["E"]), tol=0.15):
        passes.append("Fig.5 C.right aligns with E.right")
    else:
        failures.append("Fig.5 C.right must align with E.right")
    if _near(_x(pos["D"]), _x(pos["A"]), tol=0.15) and _near(_right(pos["D"]), _right(pos["B"]), tol=0.15) and _near(_x(pos["E"]), _x(pos["C"]), tol=0.15):
        passes.append("Fig.5 bottom row follows the 3-column scaffold with D spanning columns 1-2 and E in column 3")
    else:
        failures.append("Fig.5 bottom row must align to the 3-column scaffold: D spans A+B, E aligns with C")
    if _w(pos["D"]) > _w(pos["E"]):
        passes.append("Fig.5 D remains wider than E")
    else:
        failures.append("Fig.5 D must be wider than E")
    row_gap = _y(pos["D"]) - _bottom(pos["A"])
    top_gap_ab = _x(pos["B"]) - _right(pos["A"])
    top_gap_bc = _x(pos["C"]) - _right(pos["B"])
    bottom_gap = _x(pos["E"]) - _right(pos["D"])
    if row_gap >= 6.0 and min(top_gap_ab, top_gap_bc, bottom_gap) >= 5.0:
        passes.append("Fig.5 row and column gaps are clean and positive")
    else:
        failures.append(f"Fig.5 gaps are too tight: row={row_gap:.2f}, top=({top_gap_ab:.2f},{top_gap_bc:.2f}), bottom={bottom_gap:.2f}")
    left_margin = min(_x(pos[pid]) for pid in ids)
    top_margin = min(_y(pos[pid]) for pid in ids)
    right_margin = 165.0 - max(_right(pos[pid]) for pid in ids)
    bottom_margin = 130.0 - max(_bottom(pos[pid]) for pid in ids)
    if left_margin >= 6.0 and top_margin >= 5.0 and right_margin >= 5.0 and bottom_margin >= 6.0:
        passes.append("Fig.5 uses stable outer margins on all sides")
    else:
        failures.append(f"Fig.5 outer margins too small: left={left_margin:.2f}, top={top_margin:.2f}, right={right_margin:.2f}, bottom={bottom_margin:.2f}")
    axes = {pid: render_metadata.get(pid, {}).get("plot_axes_bounds", render_metadata.get(pid, {}).get("axes_bounds", [])) for pid in ids}
    if all(isinstance(axes[pid], list) and len(axes[pid]) == 4 for pid in ids):
        passes.append("Fig.5 alignment checks use actual plotting axes boxes rather than panel-label or outer-panel extents")
        top_widths_axes = [_box_w(axes[pid]) for pid in ("A", "B", "C")]
        top_heights_axes = [_box_h(axes[pid]) for pid in ("A", "B", "C")]
        if max(top_widths_axes) - min(top_widths_axes) <= 0.004 and max(top_heights_axes) - min(top_heights_axes) <= 0.004:
            passes.append("Fig.5 A/B/C plotting axes share top-row height")
        else:
            failures.append(f"Fig.5 A/B/C plotting boxes must match, found widths={top_widths_axes}, heights={top_heights_axes}")
        if _near(axes["D"][1], axes["E"][1], tol=0.004) and _near(axes["D"][3], axes["E"][3], tol=0.004):
            passes.append("Fig.5 D/E plotting axes share bottom-row top and bottom")
        else:
            failures.append("Fig.5 D/E plotting boxes must align vertically")
        if _near(axes["D"][1], axes["E"][1], tol=0.004):
            passes.append("Fig.5D bottom x-axis aligns with Fig.5E bottom x-axis")
        else:
            failures.append("Fig.5D bottom x-axis must align with Fig.5E bottom x-axis")
        if _near(axes["D"][0], axes["A"][0], tol=0.004):
            passes.append("Fig.5D plotting left boundary aligns with Fig.5A")
        else:
            failures.append("Fig.5D plotting left boundary must align with Fig.5A")
        if _near(axes["D"][2], axes["B"][2], tol=0.004):
            passes.append("Fig.5D plotting right boundary aligns with Fig.5B")
        else:
            failures.append("Fig.5D plotting right boundary must align with Fig.5B")
        if _near(axes["D"][0], axes["A"][0], tol=0.004) and _near(axes["D"][2], axes["B"][2], tol=0.004) and _near(axes["E"][0], axes["C"][0], tol=0.004) and _near(axes["E"][2], axes["C"][2], tol=0.004):
            passes.append("Fig.5 bottom-row plotting axes align to the top-row column boundaries")
        else:
            failures.append("Fig.5 bottom-row axes must align to the top-row column boundaries")


def _check_fig6_specifics(
    figure_id: str,
    spec: Mapping[str, Any],
    panels: Mapping[str, Any],
    output_dir: Path,
    adapter_results: Mapping[str, AdapterResult],
    render_metadata: Mapping[str, Mapping[str, Any]],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    if figure_id != "fig6":
        return
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 145:
        passes.append("Fig.6 canvas is 165 x 145 mm")
    else:
        failures.append(f"Fig.6 canvas must be 165 x 145 mm, found {canvas}")

    if all(pid in panels for pid in ("A", "B", "C", "D", "E", "F")):
        pos = {pid: _fig6_pos(panels[pid]) for pid in ("A", "B", "C", "D", "E", "F")}
        b, c, d, f, e, a = pos["B"], pos["C"], pos["D"], pos["F"], pos["E"], pos["A"]
        if _fig6_close(b["w"], c["w"]) and _fig6_close(b["w"], d["w"]) and _fig6_close(b["w"], f["w"]) and _fig6_close(b["h"], c["h"]) and _fig6_close(b["h"], d["h"]) and _fig6_close(b["h"], f["h"]):
            passes.append("Fig.6B/C/D/F have identical panel size")
        else:
            failures.append("Fig.6B/C/D/F must have identical width and height")
        if _fig6_close(b["y"], c["y"]) and _fig6_close(b["y"], d["y"]) and _fig6_close(b["bottom"], c["bottom"]) and _fig6_close(b["bottom"], d["bottom"]):
            passes.append("Fig.6B/C/D share y position and bottom edge")
        else:
            failures.append("Fig.6B/C/D must share y position and bottom edge")
        if _fig6_close(c["x"] - b["right"], d["x"] - c["right"]):
            passes.append("Fig.6B-C and Fig.6C-D gaps are identical")
        else:
            failures.append("Fig.6B-C and Fig.6C-D gaps must match")
        if _fig6_close(e["x"], b["x"]) and _fig6_close(e["right"], c["right"]) and _fig6_close(e["w"], b["w"] + (c["x"] - b["right"]) + c["w"]):
            passes.append("Fig.6E spans the first two anchor columns")
        else:
            failures.append("Fig.6E must span B+C and align to B.left/C.right")
        if _fig6_close(f["x"], d["x"]) and _fig6_close(f["right"], d["right"]) and _fig6_close(e["y"], f["y"]) and _fig6_close(e["bottom"], f["bottom"]):
            passes.append("Fig.6F aligns with Fig.6D and shares the Fig.6E row baseline")
        else:
            failures.append("Fig.6F must align with D and share E's y position and bottom edge")
        if _fig6_close(a["x"], b["x"]) and _fig6_close(a["right"], d["right"]):
            passes.append("Fig.6A spans the full B+C+D row width")
        else:
            failures.append("Fig.6A must align to B.left and D.right")

    for panel_id, required_n in ((spec.get("qc_requirements") or {}).get("require_n_networks") or {}).items():
        if panel_id not in panels:
            continue
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        n = _panel_n(df)
        if n >= int(required_n):
            passes.append(f"Fig.6{panel_id}: n={n} networks/seeds available")
        else:
            warnings.append(f"Fig.6{panel_id}: expected n={required_n}, found n={n}")

    a_path = panel_output_paths(output_dir, figure_id, "A")["panel_data"]
    if "A" in panels and a_path.exists():
        a_df = pd.read_csv(a_path)
        has_xy = {"support_loss_in_final_peak_region", "anchor_retreat"}.issubset(a_df.columns) or {"x_value", "y_value"}.issubset(a_df.columns)
        if has_xy and pd.to_numeric(a_df.get("value", pd.Series(dtype=float)), errors="coerce").notna().any():
            passes.append("Fig.6A includes support-loss/anchor-retreat mapping columns")
        else:
            warnings.append("Fig.6A direct support-loss/anchor-retreat data unavailable; blank panel expected")
        units = set(a_df.get("unit", pd.Series(dtype=str)).astype(str).str.lower())
        if "position" in units:
            passes.append("Fig.6A anchor-retreat unit is explicit")
        elif a_df.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any():
            passes.append("Fig.6A omits anchor-retreat units because the panel is intentionally blank")
        else:
            warnings.append("Fig.6A anchor retreat unit is ambiguous")
        if a_df.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any():
            passes.append("Fig.6A missing direct source remains explicit")
    a_mapping = str((panels.get("A") or {}).get("source_mapping", {})).lower()
    a_notes = f"{(panels.get('A') or {}).get('claim', '')} {(panels.get('A') or {}).get('optional_schematic', '')} {a_mapping}".lower()
    if "final" in a_notes and "peak" in a_notes:
        passes.append("Fig.6A spec/source manifest documents final peak-region linkage")
    else:
        warnings.append("Fig.6A lacks documented conceptual link to final peak region")

    b_path = panel_output_paths(output_dir, figure_id, "B")["panel_data"]
    if "B" in panels and b_path.exists():
        b_df = pd.read_csv(b_path)
        expected = {"Multi-recent", "Single-recent", "Multi-old", "Single-old"}
        conditions = set(b_df.get("condition", []))
        if expected.issubset(conditions):
            passes.append("Fig.6B includes all update-history groups")
        else:
            failures.append("Fig.6B must include Multi-recent, Single-recent, Multi-old, and Single-old")
        if "peak_fraction" in b_df.columns or "peak_fraction" in set(b_df.get("metric", [])):
            passes.append("Fig.6B includes peak_fraction")
        else:
            failures.append("Fig.6B must include peak_fraction or equivalent")
        if "Multi-recent" not in conditions:
            warnings.append("Fig.6B Multi-recent group is missing")
        if _contains_internal_label(conditions):
            warnings.append("Fig.6B group labels retain internal source names")
        _check_fig6_granularity("B", output_dir, passes, warnings, failures)

    c_path = panel_output_paths(output_dir, figure_id, "C")["panel_data"]
    if "C" in panels and c_path.exists():
        c_df = pd.read_csv(c_path)
        if {"repetition", "recency"}.issubset(c_df.columns) or "update_history_group" in c_df.columns:
            passes.append("Fig.6C includes repetition/recency or mapped update-history groups")
        else:
            failures.append("Fig.6C must include repetition/recency information")
        if "final_stsp_gain" in c_df.columns or "final_stsp_gain" in set(c_df.get("metric", [])):
            passes.append("Fig.6C includes final_stsp_gain")
        else:
            failures.append("Fig.6C must include final_stsp_gain or equivalent")
        groups = set(c_df.get("condition", []))
        if len(groups.intersection({"Single-old", "Multi-old", "Single-recent", "Multi-recent"})) < 4:
            warnings.append("Fig.6C groups are collapsed in a way that may limit interaction interpretation")
        units = set(c_df.get("unit", pd.Series(dtype=str)).astype(str).str.lower())
        if "gain" in units:
            passes.append("Fig.6C final STSP gain unit is explicit")
        else:
            warnings.append("Fig.6C final STSP gain unit is ambiguous")
        _check_fig6_granularity("C", output_dir, passes, warnings, failures)

    d_path = panel_output_paths(output_dir, figure_id, "D")["panel_data"]
    if "D" in panels and d_path.exists():
        d_df = pd.read_csv(d_path)
        conditions = set(d_df.get("condition", []))
        if {"Overlap-only", "Update + recency"}.issubset(conditions):
            passes.append("Fig.6D includes Overlap-only and Update + recency conditions")
        else:
            failures.append("Fig.6D must include Overlap-only and Update + recency")
        if "prediction_r2" in d_df.columns or "prediction_r2" in set(d_df.get("metric", [])):
            passes.append("Fig.6D includes prediction_r2")
        else:
            failures.append("Fig.6D must include prediction_r2 or equivalent")
        if _panel_n(d_df) > 0:
            passes.append("Fig.6D paired network identifiers available")
        else:
            warnings.append("Fig.6D paired network identifiers unavailable")
        stats_path = panel_output_paths(output_dir, figure_id, "D")["stats"]
        if stats_path.exists():
            d_stats = read_json(stats_path)
            if "delta_r2" in d_stats or "model_comparison" in d_stats:
                passes.append("Fig.6D delta R2 is derived in stats/data")
            else:
                warnings.append("Fig.6D delta R2 not found in stats")
        _check_fig6_granularity("D", output_dir, passes, warnings, failures)

    e_path = panel_output_paths(output_dir, figure_id, "E")["panel_data"]
    if "E" in panels and e_path.exists():
        e_df = pd.read_csv(e_path)
        expected = {"Peak-flattened", "Intact-final", "Peak-boosted"}
        conditions = set(e_df.get("condition", []))
        if expected.issubset(conditions):
            passes.append("Fig.6E includes peak-flattened, intact-final, and peak-boosted")
        else:
            failures.append("Fig.6E must include Peak-flattened, Intact-final, and Peak-boosted")
        if "peak_associated_spike_enrichment" in e_df.columns or "peak_associated_spike_enrichment" in set(e_df.get("metric", [])):
            passes.append("Fig.6E includes peak-associated spike enrichment")
        else:
            failures.append("Fig.6E must include peak-associated spike enrichment or equivalent")
        if any(str(v).lower() in {"e2", "g"} for v in e_df.get("condition", [])):
            warnings.append("Fig.6E old source panel labels E2/G appear in final condition labels")
        means = e_df.groupby("condition", dropna=False)["value"].mean() if "value" in e_df.columns else pd.Series(dtype=float)
        if {"Peak-flattened", "Intact-final", "Peak-boosted"}.issubset(set(means.index)):
            if means["Peak-flattened"] <= means["Intact-final"] <= means["Peak-boosted"]:
                passes.append("Fig.6E flatten/boost direction is interpretable from data")
            else:
                warnings.append("Fig.6E flatten/boost direction is not monotonic in current data")
        units = set(e_df.get("unit", pd.Series(dtype=str)).astype(str).str.lower())
        if units.intersection({"enrichment", "score"}):
            passes.append("Fig.6E spike enrichment unit is explicit")
        else:
            warnings.append("Fig.6E spike enrichment unit is ambiguous")
        _check_fig6_granularity("E", output_dir, passes, warnings, failures)

    f_path = panel_output_paths(output_dir, figure_id, "F")["panel_data"]
    if "F" in panels and f_path.exists():
        f_df = pd.read_csv(f_path)
        has_xy = {"probe_peak_overlap", "intact_over_flattened_benefit"}.issubset(f_df.columns) or {"x_value", "y_value"}.issubset(f_df.columns)
        if has_xy:
            passes.append("Fig.6F includes probe-peak overlap and intact-over-flattened benefit")
        else:
            failures.append("Fig.6F must include probe_peak_overlap and intact_over_flattened_benefit")
        stats_path = panel_output_paths(output_dir, figure_id, "F")["stats"]
        if stats_path.exists():
            f_stats = read_json(stats_path)
            quadratic = f_stats.get("quadratic_fit") or f_stats.get("regression_summary") or {}
            if quadratic.get("fit") == "quadratic" and quadratic.get("r2") is not None:
                passes.append("Fig.6F quadratic fit stats available")
            else:
                warnings.append("Fig.6F quadratic fit stats unavailable")
        if "overlap_unit" in f_df.columns or f_df.get("unit", pd.Series(dtype=str)).astype(str).str.lower().eq("fraction").any():
            passes.append("Fig.6F overlap metric unit is documented")
        else:
            warnings.append("Fig.6F overlap metric unit is ambiguous")
        if "intact_over_flattened_benefit" in f_df.columns or "intact_over_flattened" in " ".join(map(str, f_df.get("metric", []))).lower():
            passes.append("Fig.6F benefit metric is explicitly intact-over-flattened")
        else:
            warnings.append("Fig.6F benefit metric is not explicitly intact-over-flattened")
        _check_fig6_granularity("F", output_dir, passes, warnings, failures)
        f_meta = render_metadata.get("F") or {}
        if f_meta:
            if f_meta.get("plot_form") == "scatter_quadratic_fit":
                passes.append("Fig.6F renders as scatter plus quadratic fit")
            else:
                failures.append(f"Fig.6F must render as scatter plus quadratic fit, found {f_meta.get('plot_form')}")

    if render_metadata:
        clipped_panels = {
            panel_id: list(meta.get("clipped_artists", []))
            for panel_id, meta in render_metadata.items()
            if meta.get("clipped_artists") or meta.get("panel_label_clipped")
        }
        if clipped_panels:
            failures.append(f"Fig.6 labels/ticks/legends/annotations/panel letters clipped: {clipped_panels}")
        else:
            passes.append("Fig.6 has no clipped labels, ticks, legends, annotations, or panel letters")
        inside = [pid for pid, meta in render_metadata.items() if meta.get("y_tick_labels_inside_axes") or meta.get("y_label_inside_axes")]
        if inside:
            failures.append(f"Fig.6 y-axis labels/ticks fall inside plotting regions: {inside}")
        else:
            passes.append("Fig.6 y-axis labels/ticks stay outside plotting regions")
        if (render_metadata.get("A") or {}).get("plot_form") == "blank_panel":
            passes.append("Fig.6A missing-source slot renders blank")

    old_tokens = ("e2", "old fig6b", "old fig6c", "fig6b_", "fig6c_", "fig6d_", "fig6e_")
    fig4_terms = ("similarity bin", "dynamic-probe index", "overlap-preserving perturbation", "final readout recovery")
    fig5_terms = ("early recruitment", "winner/loser", "winner-loser", "local inhibition", "local competition")
    claim_text = " ".join(str((panel or {}).get("claim", "")) for panel in panels.values()).lower()
    leaked_fig4 = [term for term in fig4_terms if term in claim_text]
    leaked_fig5 = [term for term in fig5_terms if term in claim_text]
    if leaked_fig4:
        warnings.append(f"Fig.6 claims include Fig.4-specific terms: {leaked_fig4}")
    else:
        passes.append("Fig.6 claims do not repeat Fig.4 similarity/DPI/perturbation/readout content")
    if leaked_fig5:
        warnings.append(f"Fig.6 claims include Fig.5-specific terms: {leaked_fig5}")
    else:
        passes.append("Fig.6 claims do not repeat Fig.5 recruitment/local-competition content")

    raw_label_panels: list[str] = []
    image_sources: list[str] = []
    for panel_id in panels:
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        label_text = " ".join(map(str, df.get("condition", []))).lower()
        if any(token in label_text for token in old_tokens) or _contains_internal_label(set(df.get("condition", []))):
            raw_label_panels.append(panel_id)
        if "source_file" in df.columns:
            image_sources.extend([str(v) for v in df["source_file"].dropna().unique() if str(v).lower().endswith((".png", ".pdf", ".svg"))])
    if raw_label_panels:
        warnings.append(f"Fig.6 final condition labels expose old/internal names in panels {raw_label_panels}")
    else:
        passes.append("Fig.6 final condition labels do not expose old code names")
    if image_sources:
        warnings.append(f"Fig.6 panel data references old source figure images: {image_sources}")
    else:
        passes.append("Fig.6 panel data does not use old source figure images")


def _fig6_pos(panel: Mapping[str, Any]) -> dict[str, float]:
    raw = panel.get("position_mm") or {}
    x = float(raw.get("x", 0.0))
    y = float(raw.get("y", 0.0))
    w = float(raw.get("w", 0.0))
    h = float(raw.get("h", 0.0))
    return {"x": x, "y": y, "w": w, "h": h, "right": x + w, "bottom": y + h}


def _fig6_close(left: float, right: float, *, tol: float = 0.04) -> bool:
    return abs(float(left) - float(right)) <= tol


def _check_fig6_granularity(panel_id: str, output_dir: Path, passes: list[str], warnings: list[str], failures: list[str]) -> None:
    stats_path = panel_output_paths(output_dir, "fig6", panel_id)["stats"]
    if not stats_path.exists():
        warnings.append(f"Fig.6{panel_id}: granularity stats unavailable")
        return
    stats = read_json(stats_path)
    required = {
        "n_networks",
        "source_files_used",
        "raw_rows_read",
        "rows_after_source_filtering",
        "rows_written_to_panel_data",
        "adapter_performed_network_level_averaging",
        "source_appeared_preaggregated",
    }
    missing = sorted(required.difference(stats))
    if missing:
        failures.append(f"Fig.6{panel_id}: missing granularity stats {missing}")
        return
    if bool(stats.get("adapter_performed_network_level_averaging")):
        failures.append(f"Fig.6{panel_id}: adapter performed network-level averaging")
    else:
        passes.append(f"Fig.6{panel_id}: adapter did not perform network-level averaging")
    written = int(stats.get("rows_written_to_panel_data") or 0)
    after_filter = int(stats.get("rows_after_source_filtering") or 0)
    if panel_id == "D":
        enough_rows = written >= after_filter * 2 if after_filter else written == 0
    else:
        enough_rows = written >= after_filter if after_filter else written == 0
    if enough_rows:
        passes.append(f"Fig.6{panel_id}: panel_data rows preserve source-row granularity")
    else:
        failures.append(f"Fig.6{panel_id}: panel_data rows are fewer than filtered source rows")
    if bool(stats.get("source_appeared_preaggregated")):
        warnings.append(f"Fig.6{panel_id}: source appears pre-aggregated; row-level data unavailable.")
    else:
        passes.append(f"Fig.6{panel_id}: source row-level structure is available")
    if written <= max(int(stats.get("n_networks") or 0), 20) and panel_id in {"B", "C", "D", "E"} and not bool(stats.get("source_appeared_preaggregated")):
        failures.append(f"Fig.6{panel_id}: row-level source available but panel_data looks network-level")


def _contains_internal_label(values: set[Any]) -> bool:
    text = " ".join(str(v).lower() for v in values)
    return any(token in text for token in ("multi_recent", "single_recent", "multi_old", "single_old", "peak_flattened", "peak_boosted", "intact_final", "fig6b", "fig6c", "fig6d", "fig6e"))


def _panel_n(df: pd.DataFrame) -> int:
    for col in ("seed_id", "network_id"):
        if col in df.columns:
            return int(df[col].replace("", pd.NA).dropna().nunique())
    return 0


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
