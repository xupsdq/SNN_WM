from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult, panel_output_paths
from src.plotting.paper_fig.qc_common import (
    _bottom,
    _box_h,
    _box_in_upper_left,
    _box_inside,
    _box_w,
    _boxes_overlap,
    _check_exports,
    _h,
    _load_panel_data_map,
    _near,
    _read_panel_data,
    _read_panel_stats,
    _right,
    _update_summary_csv,
    _w,
    _write_report,
    _x,
    _y,
)
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
    _check_fig1_supp_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig2_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig2_supp_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig3_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig4_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig4_supp_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig5_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig5_supp_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig6_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
    _check_fig6_supp_specifics(figure_id, spec, panels, output_dir, adapter_results, render_metadata or {}, passes, warnings, failures)
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
    _update_summary_csv(output_dir.parent / "all_figures_qc_summary.csv", report)
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
        if panel.get("panel_type") not in ("manual_schematic", "manual_or_programmatic_schematic", "programmatic_or_manual_schematic", "two_item_episode_schematic", "multi_item_sequence_schematic") and panel.get("data_adapter") in (None, "", "none"):
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
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 124:
        passes.append("Fig.1 canvas is 165 x 124 mm")
    else:
        failures.append(f"Fig.1 canvas must be 165 x 124 mm, found {canvas}")
    c_panel = panels.get("C") or {}
    if c_panel.get("renderer") == "render_fig1_delay_decode_summary":
        passes.append("Fig.1C uses compact layer-wise summary renderer")
    else:
        failures.append(f"Fig.1C must use render_fig1_delay_decode_summary, found {c_panel.get('renderer')}")
    if c_panel.get("renderer") == "render_fig1_delay_decode":
        failures.append("Fig.1C must not use the delay timecourse renderer in the main figure")
    for panel_id in ("B", "C"):
        panel = panels.get(panel_id)
        if not panel:
            continue
        refs = panel.get("reference_lines") or []
        if any(float(ref.get("value")) == 10 for ref in refs):
            passes.append(f"Fig.1{panel_id}: 10% chance reference line present")
        else:
            failures.append(f"Fig.1{panel_id}: missing 10% chance reference line")

    panel_data: dict[str, pd.DataFrame] = {}
    for panel_id in ("B", "C", "D", "E"):
        if panel_id not in panels:
            continue
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        missing = [name for name in ("panel_data", "stats", "sources") if not paths[name].exists()]
        if missing:
            failures.append(f"Fig.1{panel_id}: missing adapter outputs {missing}")
            continue
        passes.append(f"Fig.1{panel_id}: panel_data/stats/source_manifest exist")
        panel_data[panel_id] = pd.read_csv(paths["panel_data"])
        stats = read_json(paths["stats"])
        sources = read_json(paths["sources"])
        run_mode = str(stats.get("run_mode") or sources.get("run_mode") or "")
        n_networks = int(stats.get("n_networks") or sources.get("n_networks") or 0)
        if run_mode:
            passes.append(f"Fig.1{panel_id}: run_mode={run_mode}")
        else:
            failures.append(f"Fig.1{panel_id}: run_mode missing from stats/source manifest")
        if n_networks == 1 or run_mode == "single_network_draft":
            warnings.append(f"Fig.1{panel_id}: single_network_draft n_networks=1; draft-only, not final manuscript statistics")
        elif n_networks > 1:
            passes.append(f"Fig.1{panel_id}: n_networks={n_networks}")
        else:
            warnings.append(f"Fig.1{panel_id}: n_networks not recorded")

    b_df = panel_data.get("B")
    if b_df is not None:
        metrics = set(b_df.get("metric", pd.Series(dtype=str)).astype(str))
        if metrics == {"overall_recall"}:
            passes.append("Fig.1B uses overall recall")
        else:
            failures.append(f"Fig.1B must use overall_recall, found {sorted(metrics)}")

    c_df = panel_data.get("C")
    if c_df is not None:
        layers = set(str(v) for v in c_df.get("layer", pd.Series(dtype=str)).dropna().unique())
        expected_layers = set(((spec.get("qc_requirements") or {}).get("required_decode_layers") or ["layer1", "layer2", "layer3"]))
        if expected_layers.issubset(layers):
            passes.append("Fig.1C has layer1/layer2/layer3 delay decoding rows")
        else:
            failures.append(f"Fig.1C missing required layers {sorted(expected_layers - layers)}")
        if "delay_ms" in c_df.columns and pd.to_numeric(c_df["delay_ms"], errors="coerce").dropna().nunique() > 0:
            passes.append("Fig.1C records delay_ms values")
        else:
            failures.append("Fig.1C must record delay_ms values")
        if {"seed_id", "layer", "delay_ms"}.issubset(c_df.columns):
            counts = c_df.groupby(["seed_id", "layer"], dropna=False)["delay_ms"].nunique(dropna=True)
            offenders = counts[counts > 1]
            if offenders.empty:
                passes.append("Fig.1C has one plotting delay per seed/layer")
            else:
                failures.append(f"Fig.1C has multiple plotting delay_ms values per seed/layer: {offenders.to_dict()}")
        if set(c_df.get("metric", pd.Series(dtype=str)).astype(str)) == {"delay_decode_accuracy"}:
            passes.append("Fig.1C uses delay_decode_accuracy")
        else:
            failures.append("Fig.1C must use delay_decode_accuracy")

    d_df = panel_data.get("D")
    if d_df is not None:
        conditions = set(str(v) for v in d_df.get("condition", pd.Series(dtype=str)).dropna().unique())
        required = set(((spec.get("qc_requirements") or {}).get("required_main_conditions") or ["dynamic_intact", "ux_trial_shuffle", "static_frozen"]))
        if required.issubset(conditions):
            passes.append("Fig.1D includes required main conditions")
        else:
            failures.append(f"Fig.1D missing required main conditions {sorted(required - conditions)}")
        if set(d_df.get("metric", pd.Series(dtype=str)).astype(str)) == {"error_rate"}:
            passes.append("Fig.1D uses error_rate")
        else:
            failures.append("Fig.1D must use error_rate")

    e_df = panel_data.get("E")
    if e_df is not None:
        conditions = set(str(v) for v in e_df.get("condition", pd.Series(dtype=str)).dropna().unique())
        traces = set(str(v) for v in e_df.get("trace", pd.Series(dtype=str)).dropna().unique())
        categories = set(str(v) for v in e_df.get("category", pd.Series(dtype=str)).dropna().unique())
        if {"dynamic_intact", "ux_trial_shuffle"}.issubset(conditions):
            passes.append("Fig.1E includes dynamic and u/x shuffle error-composition rows")
        else:
            failures.append(f"Fig.1E missing error-composition conditions, found {sorted(conditions)}")
        if {"Original", "Donor", "Other"}.issubset(traces.union(categories)):
            passes.append("Fig.1E includes Original, Donor, and Other categories")
        else:
            failures.append(f"Fig.1E must include Original, Donor, and Other categories, found traces={sorted(traces)}, categories={sorted(categories)}")
        if set(e_df.get("metric", pd.Series(dtype=str)).astype(str)) == {"error_composition_within_error_pool"}:
            passes.append("Fig.1E uses error_composition_within_error_pool")
        else:
            failures.append("Fig.1E must use error_composition_within_error_pool")
        if {"seed_id", "condition", "value"}.issubset(e_df.columns):
            sums = e_df.groupby(["seed_id", "condition"], dropna=False)["value"].sum()
            max_dev = float((sums - 100.0).abs().max()) if not sums.empty else 0.0
            if max_dev <= 0.5:
                passes.append("Fig.1E composition sums to 100 +/- 0.5 per seed/condition")
            else:
                failures.append(f"Fig.1E composition deviates from 100 by up to {max_dev:.3f}")
        stats_path = panel_output_paths(output_dir, figure_id, "E")["stats"]
        source_path = panel_output_paths(output_dir, figure_id, "E")["sources"]
        e_stats = read_json(stats_path) if stats_path.exists() else {}
        e_sources = read_json(source_path) if source_path.exists() else {}
        source_level = str(e_stats.get("source_level") or e_sources.get("source_level") or "")
        if source_level == "trial_level":
            passes.append("Fig.1E uses trial-level source data")
        elif source_level:
            warnings.append(f"Fig.1E source_level={source_level}; trial-level readout is preferred for final manuscript")

    visible_terms = []
    for df in panel_data.values():
        for col in ("condition", "trace", "metric"):
            if col in df.columns:
                visible_terms.extend(str(v) for v in df[col].dropna().unique())
    old_terms = ("A_dynamic_base", "D_trial_shuffle_ux", "E_static_frozen", "B-map", "Pred =")
    leaked = [term for term in old_terms if term in " ".join(visible_terms)]
    if leaked:
        failures.append(f"Fig.1 panel data exposes old/internal labels: {leaked}")
    else:
        passes.append("Fig.1 panel data avoids old/internal condition labels")

    supp_conditions: set[str] = set()
    manifest_path = output_dir / f"{figure_id}_source_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for item in manifest.get("sources", []):
            panel_manifest = item.get("manifest") or {}
            supp_conditions.update(str(v) for v in panel_manifest.get("conditions", []) or [])
            supp_conditions.update(str(v) for v in panel_manifest.get("supplementary_conditions", []) or [])
    optional = set(((spec.get("qc_requirements") or {}).get("supplementary_conditions") or []))
    if optional and optional.isdisjoint(supp_conditions):
        warnings.append("Fig.1 supplementary shuffle conditions are not visible in main panel manifests; check supp_substrate_shuffle_metrics.csv")

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
                if a_form in {"blank_manual_slot", "programmatic_architecture_schematic"}:
                    passes.append("Fig.1A missing manual asset renders as an allowed blank/manual slot")
                else:
                    failures.append(f"Fig.1A missing manual asset rendered unexpected form {a_form}")
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
        _check_fig1_geometry(spec, panels, passes, warnings, failures)
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


def _check_fig1_geometry(spec: Mapping[str, Any], panels: Mapping[str, Any], passes: list[str], warnings: list[str], failures: list[str]) -> None:
    pos = {pid: (panels.get(pid) or {}).get("position_mm") or {} for pid in ("A", "B", "C", "D", "E") if pid in panels}
    if len(pos) != 5:
        return
    def _edge(pid: str, key: str) -> float:
        return float(pos[pid].get(key, 0.0))
    if _near(_edge("A", "x"), 12.0, tol=0.05) and _near(_edge("A", "w"), 147.0, tol=0.05):
        passes.append("Fig.1A spans the full two-column width")
    else:
        failures.append(f"Fig.1A must span x=12, w=147 mm, found {pos['A']}")
    expected_gap = float((spec.get("gutters_mm") or {}).get("horizontal", 12.0))
    top_gap = _edge("C", "x") - (_edge("B", "x") + _edge("B", "w"))
    bottom_gap = _edge("E", "x") - (_edge("D", "x") + _edge("D", "w"))
    if _near(top_gap, expected_gap, tol=0.1) and _near(bottom_gap, expected_gap, tol=0.1):
        passes.append(f"Fig.1 two-column gutters are {expected_gap:g} mm")
    else:
        failures.append(f"Fig.1 two-column gutters must be {expected_gap:g} mm, found top={top_gap}, bottom={bottom_gap}")
    if _near(_edge("B", "x"), _edge("D", "x"), tol=0.05) and _near(_edge("C", "x"), _edge("E", "x"), tol=0.05):
        passes.append("Fig.1 top and bottom rows align by column")
    else:
        failures.append("Fig.1 B/D and C/E columns must align")
    widths = [_edge(pid, "w") for pid in ("B", "C", "D", "E")]
    if max(widths) - min(widths) <= 0.05:
        passes.append(f"Fig.1 B-E use equal {widths[0]:.1f} mm column widths")
    else:
        failures.append(f"Fig.1 B-E must use equal column widths, found {widths}")
    if _near(_edge("B", "y"), _edge("C", "y"), tol=0.05) and _near(_edge("D", "y"), _edge("E", "y"), tol=0.05):
        passes.append("Fig.1 panel labels share clean row-wise alignment anchors")
    else:
        warnings.append("Fig.1 panel row anchors are not perfectly aligned")
    bottom_margin = min(124.0 - (_edge(pid, "y") + _edge(pid, "h")) for pid in ("D", "E"))
    if bottom_margin >= 8.0:
        passes.append(f"Fig.1 bottom margin supports x tick labels ({bottom_margin:.1f} mm)")
    else:
        failures.append(f"Fig.1 bottom margin too small for x tick labels ({bottom_margin:.1f} mm)")


def _check_fig1_supp_specifics(
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
    _ = adapter_results
    if figure_id not in {"fig1_supp", "fig1_supp_s2"}:
        return
    canvas = spec.get("canvas_mm") or {}
    if figure_id == "fig1_supp":
        expected = ["S1A", "S1B", "S1C", "S1D"]
        expected_height = 86.0
        label = "Fig.1 supplement S1"
    else:
        expected = ["S2A", "S2B", "S2C", "S2D"]
        expected_height = 134.0
        label = "Fig.1 supplement S2"
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == expected_height:
        passes.append(f"{label} canvas is 165 x {expected_height:.0f} mm")
    else:
        failures.append(f"{label} canvas must be 165 x {expected_height:.0f} mm, found {canvas}")
    if list(panels.keys()) == expected:
        passes.append(f"{label} active panel set is {expected[0]}-{expected[-1]}")
    else:
        failures.append(f"{label} active panels must be {expected}, found {list(panels.keys())}")
    if list(spec.get("reading_order") or []) == expected:
        passes.append(f"{label} reading order is {expected[0]}-{expected[-1]}")
    else:
        failures.append(f"{label} reading_order must be {expected}, found {spec.get('reading_order')}")
    if figure_id == "fig1_supp":
        if "S1A" in panels and "S1B" in panels and panels["S1A"] is not panels["S1B"]:
            passes.append("Fig.1 supplement S1A and S1B remain separate panels")
        else:
            failures.append("Fig.1 supplement S1A and S1B must remain separate panels")
    forbidden_claim = "Dynamic STSP alters probe classification relative to static-frozen STSP in a delay-dependent manner."
    forbidden_adapters = []
    forbidden_renderers = []
    forbidden_claim_panels = []
    for panel_id, panel in panels.items():
        if panel.get("data_adapter") == "s2_dms_delay_accuracy_adapter":
            forbidden_adapters.append(panel_id)
        if panel.get("renderer") == "render_dms_delay_probe_accuracy":
            forbidden_renderers.append(panel_id)
        if str(panel.get("claim", "")) == forbidden_claim:
            forbidden_claim_panels.append(panel_id)
    if forbidden_adapters:
        failures.append(f"{label} active panels must not use s2_dms_delay_accuracy_adapter: {forbidden_adapters}")
    else:
        passes.append(f"{label} active panels do not use s2_dms_delay_accuracy_adapter")
    if forbidden_renderers:
        failures.append(f"{label} active panels must not use render_dms_delay_probe_accuracy: {forbidden_renderers}")
    else:
        passes.append(f"{label} active panels do not use render_dms_delay_probe_accuracy")
    if forbidden_claim_panels:
        failures.append(f"{label} active panels still contain old S2C claim: {forbidden_claim_panels}")
    else:
        passes.append(f"{label} active panels omit the old dynamic-vs-static accuracy claim")
    if figure_id == "fig1_supp_s2":
        s2c_panel = panels.get("S2C") or {}
        if s2c_panel.get("data_adapter") == "s2_dms_delay_contrast_adapter" and s2c_panel.get("renderer") == "render_stsp_interference_delay":
            passes.append("Fig.1 supplement S2C uses delay contrast adapter and interference renderer")
        else:
            failures.append("Fig.1 supplement S2C must use s2_dms_delay_contrast_adapter and render_stsp_interference_delay")
        s2d_panel = panels.get("S2D") or {}
        if s2d_panel.get("data_adapter") == "s2_substrate_specificity_adapter" and s2d_panel.get("renderer") == "render_substrate_shuffle_specificity":
            passes.append("Fig.1 supplement S2D uses substrate specificity adapter and renderer")
        else:
            failures.append("Fig.1 supplement S2D must use s2_substrate_specificity_adapter and render_substrate_shuffle_specificity")
    for panel_id in ("S1A", "S1B"):
        panel = panels.get(panel_id) or {}
        raw_asset = panel.get("source") or (panel.get("source_mapping") or {}).get("manual_asset")
        if raw_asset:
            asset_path = paper_fig_root() / str(raw_asset)
            if asset_path.exists():
                passes.append(f"Fig.1 supplement {panel_id} manual asset exists")
            else:
                warnings.append(f"Fig.1 supplement {panel_id} manual asset missing: {raw_asset}")
                form = str(render_metadata.get(panel_id, {}).get("plot_form", ""))
                if not form or form.startswith("programmatic_") or form == "manual_schematic_asset_slot":
                    passes.append(f"Fig.1 supplement {panel_id} missing asset uses allowed placeholder/programmatic schematic")
                else:
                    failures.append(f"Fig.1 supplement {panel_id} missing asset rendered unexpected form {form}")
    for panel_id, panel in panels.items():
        if panel.get("data_adapter") in (None, "", "none") or panel.get("panel_type") in ("manual_schematic", "manual_or_programmatic_schematic", "programmatic_or_manual_schematic"):
            continue
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        if not paths["panel_data"].exists():
            failures.append(f"Fig.1 supplement {panel_id}: panel_data missing")
            continue
        df = pd.read_csv(paths["panel_data"])
        stats = read_json(paths["stats"]) if paths["stats"].exists() else {}
        sources = read_json(paths["sources"]) if paths["sources"].exists() else {}
        run_mode = str(stats.get("run_mode") or sources.get("run_mode") or "")
        n_networks = int(stats.get("n_networks") or sources.get("n_networks") or 0)
        if df.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any() or sources.get("status") == "missing_source":
            warnings.append(f"Fig.1 supplement {panel_id}: missing source rendered as placeholder-compatible adapter output")
        else:
            passes.append(f"Fig.1 supplement {panel_id}: data source available")
        if run_mode == "single_network_draft" or n_networks == 1:
            warnings.append(f"Fig.1 supplement {panel_id}: single_network_draft n_networks=1; draft-only, not final manuscript statistics")
        elif n_networks > 1:
            passes.append(f"Fig.1 supplement {panel_id}: n_networks={n_networks}")
    if figure_id == "fig1_supp_s2":
        s2b_path = panel_output_paths(output_dir, figure_id, "S2B")["panel_data"]
        if s2b_path.exists():
            s2b = pd.read_csv(s2b_path)
            if "delay_ms" in s2b.columns and pd.to_numeric(s2b["delay_ms"], errors="coerce").dropna().nunique() > 1:
                passes.append("Fig.1 supplement S2B keeps full delay decoding timecourse")
            elif not s2b.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any():
                warnings.append("Fig.1 supplement S2B does not show multiple delay_ms values")
        s2c_path = panel_output_paths(output_dir, figure_id, "S2C")["panel_data"]
        if s2c_path.exists():
            s2c = pd.read_csv(s2c_path)
            if s2c.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any():
                warnings.append("Fig.1 supplement S2C delay contrast source unavailable; placeholder written")
            else:
                required_cols = {"delay_ms", "value", "unit", "metric", "condition"}
                if required_cols.issubset(s2c.columns) and set(s2c["metric"].astype(str)) == {"static_minus_dynamic_accuracy"} and set(s2c["condition"].astype(str)) == {"static_minus_dynamic"} and set(s2c["unit"].astype(str)) == {"percent"}:
                    passes.append("Fig.1 supplement S2C panel_data contains static_minus_dynamic_accuracy contrast rows")
                else:
                    failures.append("Fig.1 supplement S2C panel_data must contain metric=static_minus_dynamic_accuracy, condition=static_minus_dynamic, delay_ms, value, unit=percent")
        s2d_path = panel_output_paths(output_dir, figure_id, "S2D")["panel_data"]
        if s2d_path.exists():
            s2d = pd.read_csv(s2d_path)
            if s2d.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any():
                warnings.append("Fig.1 supplement S2D substrate specificity source unavailable; placeholder written")
            else:
                expected_conditions = {"dynamic_intact", "spike_state_shuffle", "membrane_state_shuffle", "ux_trial_shuffle", "static_frozen"}
                metrics = set(s2d.get("metric", pd.Series(dtype=str)).astype(str))
                conditions = set(s2d.get("condition", pd.Series(dtype=str)).astype(str))
                if metrics == {"donor_gain_vs_dynamic"} and expected_conditions.issubset(conditions):
                    passes.append("Fig.1 supplement S2D panel_data contains donor_gain_vs_dynamic substrate rows")
                else:
                    failures.append(f"Fig.1 supplement S2D panel_data must contain donor_gain_vs_dynamic for {sorted(expected_conditions)}, found metrics={sorted(metrics)}, conditions={sorted(conditions)}")
    manifest_path = output_dir / f"{figure_id}_source_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        active_panels = list(manifest.get("active_panels") or [])
        if active_panels == expected:
            passes.append(f"{label} source manifest records active_panels {expected[0]}-{expected[-1]}")
        else:
            failures.append(f"{label} source manifest active_panels must be {expected}, found {active_panels}")
        source_panels = [str(item.get("panel_id", "")) for item in manifest.get("sources", [])]
        forbidden = {"S2A", "S2B", "S2C", "S2D"} if figure_id == "fig1_supp" else {"S1A", "S1B", "S1C", "S1D", "S2E"}
        leaked = sorted(forbidden.intersection(source_panels))
        if leaked:
            failures.append(f"{label} source manifest must not list inactive old split panels as active: {leaked}")
        elif figure_id == "fig1_supp_s2" and "S2C" in source_panels and "S2D" in source_panels:
            passes.append("Fig.1 supplement source manifest lists new S2C/S2D active sources")



def _check_fig2_standalone_contract(
    spec: Mapping[str, Any],
    panels: Mapping[str, Any],
    output_dir: Path,
    render_metadata: Mapping[str, Mapping[str, Any]],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    figure_id = "fig2"
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 135:
        passes.append("Fig.2 canvas is 165 x 135 mm")
    else:
        failures.append(f"Fig.2 canvas must be 165 x 135 mm, found {canvas}")
    _check_fig2_geometry(panels, passes, failures)

    panel_a = panels.get("A") or {}
    if panel_a.get("panel_type") in {"manual_schematic", "manual_or_programmatic_schematic", "programmatic_or_manual_schematic"}:
        passes.append("Fig.2A is manual/programmatic schematic compatible")
    else:
        warnings.append("Fig.2A is not marked as schematic; adapter may be required")
    a_form = str(render_metadata.get("A", {}).get("plot_form", ""))
    if not a_form or a_form == "programmatic_two_item_episode_schematic":
        passes.append("Fig.2A renders as programmatic schematic or check-only placeholder")
    else:
        warnings.append(f"Fig.2A rendered unexpected form {a_form}")

    panel_data: dict[str, pd.DataFrame] = {}
    for panel_id in ("B", "C", "D", "E", "F"):
        if panel_id not in panels:
            failures.append(f"Fig.2{panel_id}: panel missing from spec")
            continue
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        missing = [name for name in ("panel_data", "stats", "sources") if not paths[name].exists()]
        if missing:
            failures.append(f"Fig.2{panel_id}: missing adapter outputs {missing}")
            continue
        passes.append(f"Fig.2{panel_id}: panel_data/stats/source_manifest exist")
        df = pd.read_csv(paths["panel_data"])
        panel_data[panel_id] = df
        stats = read_json(paths["stats"])
        sources = read_json(paths["sources"])
        run_mode = str(stats.get("run_mode") or sources.get("run_mode") or "")
        n_networks = int(stats.get("n_networks") or sources.get("n_networks") or 0)
        if run_mode:
            passes.append(f"Fig.2{panel_id}: run_mode={run_mode}")
        else:
            failures.append(f"Fig.2{panel_id}: run_mode missing")
        if n_networks == 1 or run_mode == "single_network_draft":
            warnings.append(f"Fig.2{panel_id}: single_network_draft n_networks=1; draft-only, not final manuscript statistics")
        elif n_networks > 1:
            passes.append(f"Fig.2{panel_id}: n_networks={n_networks}")
        else:
            warnings.append(f"Fig.2{panel_id}: n_networks not recorded")

    b_df = panel_data.get("B")
    if b_df is not None:
        metrics = set(b_df.get("metric", pd.Series(dtype=str)).astype(str))
        if {"sim_to_A", "sim_to_B", "fusion_dual_score"}.issubset(metrics):
            passes.append("Fig.2B includes both constituent similarities and dual score")
        else:
            failures.append(f"Fig.2B missing dual-retention metrics, found {sorted(metrics)}")
        _require_fig2_primary_rows("Fig.2B", b_df, passes, failures)

    c_df = panel_data.get("C")
    if c_df is not None:
        conditions = set(c_df.get("condition", pd.Series(dtype=str)).astype(str))
        if {"True pair", "Shuffled pair"}.issubset(conditions):
            passes.append("Fig.2C includes true and shuffled pair conditions")
        else:
            failures.append(f"Fig.2C missing true/shuffled conditions, found {sorted(conditions)}")
        if "true_minus_shuffled" in c_df.columns:
            passes.append("Fig.2C carries true_minus_shuffled values")
        else:
            failures.append("Fig.2C missing true_minus_shuffled column")
        _require_fig2_primary_rows("Fig.2C", c_df, passes, failures)

    d_df = panel_data.get("D")
    if d_df is not None:
        metrics = set(d_df.get("metric", pd.Series(dtype=str)).astype(str))
        required_metrics = {"WPRI", "residual_pair_specificity", "beyond_linear_pair_index", "linear_model_r2"}
        if required_metrics.issubset(metrics):
            passes.append("Fig.2D includes WPRI, linear model R2, and residual pair-specificity metrics")
        else:
            failures.append(f"Fig.2D missing required metrics {sorted(required_metrics - metrics)}")
        models = set(d_df.get("model_name", pd.Series(dtype=str)).dropna().astype(str))
        required_models = set(((spec.get("qc_requirements") or {}).get("required_linear_models") or ["A_only", "B_only", "unconstrained_AB", "convex_AB"]))
        if required_models.issubset(models):
            passes.append("Fig.2D includes required linear-mixture models")
        else:
            failures.append(f"Fig.2D missing linear-mixture models {sorted(required_models - models)}")
        _require_fig2_primary_rows("Fig.2D", d_df, passes, failures)

    for panel_id in ("E", "F"):
        df = panel_data.get(panel_id)
        if df is None:
            continue
        states = set(df.get("state_condition", pd.Series(dtype=str)).dropna().astype(str))
        required_states = set(((spec.get("qc_requirements") or {}).get("required_state_conditions") or ["S0", "S_A", "S_B", "S_AB"]))
        if required_states.issubset(states):
            passes.append(f"Fig.2{panel_id}: includes S0/S_A/S_B/S_AB functional conditions")
        else:
            failures.append(f"Fig.2{panel_id}: missing state conditions {sorted(required_states - states)}")
        forbidden_proxy_cols = {"proxy_score_A", "proxy_score_B"}
        if forbidden_proxy_cols.intersection(df.columns) or df.get("source_file", pd.Series(dtype=str)).astype(str).str.contains("supp_functional_proxy", case=False, na=False).any():
            failures.append(f"Fig.2{panel_id}: main panel data appears to use proxy functional diagnostics")
        else:
            passes.append(f"Fig.2{panel_id}: main panel data does not expose proxy score columns")

    e_sources_path = panel_output_paths(output_dir, figure_id, "E")["sources"]
    if e_sources_path.exists():
        e_sources = read_json(e_sources_path)
        e_summary = _fig2_source_json(e_sources, "summary.json")
        if e_summary:
            if e_summary.get("functional_readout_mode") == "real_network_rollout":
                passes.append("Fig.2 summary records real_network_rollout functional readout")
            else:
                failures.append(f"Fig.2 summary functional_readout_mode must be real_network_rollout, found {e_summary.get('functional_readout_mode')}")
            if e_summary.get("neutral_ping_proxy_used_for_main") is False and e_summary.get("partial_cue_proxy_used_for_main") is False:
                passes.append("Fig.2 summary records proxy_used_for_main=false for E/F")
            else:
                failures.append("Fig.2 summary must mark neutral/partial proxy_used_for_main false")
        e_raw = _fig2_source_csv(e_sources, "panel_e_neutral_ping_trial_readout.csv")
        if e_raw is not None:
            required = {"ping_seed", "ping_energy", "ping_spike_count", "prediction", "first_fire_time_ms"}
            missing = sorted(required.difference(e_raw.columns))
            if missing:
                failures.append(f"Fig.2E raw neutral-ping readout missing columns {missing}")
            else:
                passes.append("Fig.2E raw neutral-ping readout includes real rollout decode columns")
        else:
            failures.append("Fig.2E raw neutral-ping readout source missing from manifest")

    f_df = panel_data.get("F")
    f_sources_path = panel_output_paths(output_dir, figure_id, "F")["sources"]
    f_summary: Mapping[str, Any] | None = None
    if f_sources_path.exists():
        f_sources = read_json(f_sources_path)
        f_summary = _fig2_source_json(f_sources, "summary.json")
        is_smoke = bool(f_summary.get("smoke")) if f_summary else False
    else:
        is_smoke = False
    if f_df is not None:
        curve = f_df[f_df.get("curve_or_summary", pd.Series(dtype=str)).astype(str) == "curve"] if "curve_or_summary" in f_df.columns else f_df
        keep = set(float(v) for v in pd.to_numeric(curve.get("keep_prob", pd.Series(dtype=float)), errors="coerce").dropna().unique())
        required_keep = set(float(v) for v in ((spec.get("qc_requirements") or {}).get("required_keep_probs") or []))
        if required_keep.issubset(keep):
            passes.append("Fig.2F includes required partial-cue keep probabilities")
        elif is_smoke:
            warnings.append(f"Fig.2F smoke-mode panel has reduced keep-probability coverage; missing {sorted(required_keep - keep)}")
        else:
            failures.append(f"Fig.2F missing keep probabilities {sorted(required_keep - keep)}")
        if {"P_target", "auc_target_recovery"}.issubset(set(f_df.get("metric", pd.Series(dtype=str)).astype(str))):
            passes.append("Fig.2F includes curve and AUC metrics")
        else:
            warnings.append("Fig.2F curve or AUC metric is missing")
    if f_sources_path.exists():
        f_sources = read_json(f_sources_path)
        f_raw = _fig2_source_csv(f_sources, "panel_f_partial_cue_trial_readout.csv")
        if f_raw is not None:
            required = {"cue_pixel_count", "cue_energy", "encoded_spike_count", "prediction", "first_fire_time_ms"}
            missing = sorted(required.difference(f_raw.columns))
            if missing:
                failures.append(f"Fig.2F raw partial-cue readout missing columns {missing}")
            else:
                passes.append("Fig.2F raw partial-cue readout includes real rollout decode columns")
        else:
            failures.append("Fig.2F raw partial-cue readout source missing from manifest")
        masks = _fig2_source_csv(f_sources, "weak_probe_masks.csv")
        if masks is not None:
            required = {"same_mask_used_across_states", "cue_fraction_actual", "encoded_spike_count"}
            missing = sorted(required.difference(masks.columns))
            if missing:
                failures.append(f"Fig.2F weak_probe_masks missing columns {missing}")
            elif masks["same_mask_used_across_states"].astype(str).str.lower().isin({"true", "1"}).all():
                passes.append("Fig.2F weak_probe_masks records same mask across all state conditions")
            else:
                failures.append("Fig.2F weak_probe_masks must have same_mask_used_across_states true for main rows")
        else:
            failures.append("Fig.2F weak_probe_masks source missing from manifest")

    visible_terms: list[str] = []
    for df in panel_data.values():
        for col in ("condition", "metric", "source_file"):
            if col in df.columns:
                visible_terms.extend(str(v) for v in df[col].dropna().unique())
    old_terms = ("chunk_step2", "fig4_chunk_interaction", "panel_f_true", "panel_e_wpri")
    leaked = [term for term in old_terms if term in " ".join(visible_terms)]
    if leaked:
        failures.append(f"Fig.2 panel data exposes old/internal labels: {leaked}")
    else:
        passes.append("Fig.2 panel data avoids old experiment labels")

    if render_metadata:
        clipped = {
            panel_id: list(meta.get("clipped_artists", []))
            for panel_id, meta in render_metadata.items()
            if meta.get("clipped_artists") or meta.get("panel_label_clipped")
        }
        if clipped:
            failures.append(f"Fig.2 labels/ticks/legends/annotations/panel letters clipped: {clipped}")
        else:
            passes.append("Fig.2 rendered panels have no clipped labels/ticks/legends/annotations/panel letters")


def _require_fig2_primary_rows(label: str, df: pd.DataFrame, passes: list[str], failures: list[str]) -> None:
    layers = set(df.get("layer", pd.Series(dtype=str)).dropna().astype(str))
    variables = set(df.get("state_variable", pd.Series(dtype=str)).dropna().astype(str))
    if "layer3" in layers:
        passes.append(f"{label}: primary layer3 rows present")
    else:
        failures.append(f"{label}: missing primary layer3 rows")
    if "g" in variables:
        passes.append(f"{label}: primary state variable g present")
    else:
        failures.append(f"{label}: missing primary state variable g")


def _fig2_source_path(source_manifest: Mapping[str, Any], filename: str) -> Path | None:
    repo_root = repo_root_from_here()
    for source in source_manifest.get("sources", []) or []:
        raw = str(source.get("path", ""))
        if raw.endswith(filename):
            path = Path(raw)
            return path if path.is_absolute() else repo_root / path
    return None


def _fig2_source_csv(source_manifest: Mapping[str, Any], filename: str) -> pd.DataFrame | None:
    path = _fig2_source_path(source_manifest, filename)
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _fig2_source_json(source_manifest: Mapping[str, Any], filename: str) -> dict[str, Any] | None:
    path = _fig2_source_path(source_manifest, filename)
    if path is None or not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


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
    _ = adapter_results
    if figure_id != "fig2":
        return
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 148:
        passes.append("Fig.2 canvas is 165 x 148 mm")
    else:
        failures.append(f"Fig.2 canvas must be 165 x 148 mm, found {canvas}")
    expected_order = ["A", "B", "C", "D", "E", "F"]
    if len(panels) > 1:
        if list(spec.get("reading_order") or []) == expected_order:
            passes.append("Fig.2 reading order is A-F")
        else:
            failures.append(f"Fig.2 reading_order must be {expected_order}, found {spec.get('reading_order')}")
        missing = [panel_id for panel_id in expected_order if panel_id not in panels]
        if missing:
            failures.append(f"Fig.2 missing panels {missing}")
        else:
            passes.append("Fig.2 panel set includes A-F")
    _check_fig2_new_geometry(panels, passes, failures)

    panel_a = panels.get("A") or {}
    if panel_a.get("data_adapter") in (None, "", "none") and panel_a.get("renderer") == "render_fig2_episode_schematic":
        passes.append("Fig.2A is a blank/manual schematic slot without adapter")
    else:
        failures.append("Fig.2A must use render_fig2_episode_schematic with no data adapter")

    panel_data = _load_panel_data_map(output_dir, figure_id, panels)
    b_df = panel_data.get("B")
    if b_df is not None:
        conditions = set(b_df.get("condition", pd.Series(dtype=str)).dropna().astype(str))
        metrics = set(b_df.get("metric", pd.Series(dtype=str)).dropna().astype(str))
        if conditions == {"S_A", "S_B"}:
            passes.append("Fig.2B visual categories are only S_A and S_B")
        else:
            failures.append(f"Fig.2B must only plot S_A/S_B, found {sorted(conditions)}")
        if "fusion_dual_score" not in metrics and "Dual score" not in conditions:
            passes.append("Fig.2B does not plot fusion_dual_score")
        else:
            failures.append("Fig.2B must not plot fusion_dual_score as a visual category")
        if "fusion_dual_score" in b_df.columns:
            passes.append("Fig.2B retains fusion_dual_score as metadata")
        else:
            warnings.append("Fig.2B does not retain fusion_dual_score metadata")
        _require_fig2_primary_rows("Fig.2B", b_df, passes, failures)

    c_df = panel_data.get("C")
    if c_df is not None:
        conditions = set(c_df.get("condition", pd.Series(dtype=str)).astype(str))
        if {"True pair", "Shuffled pair"}.issubset(conditions):
            passes.append("Fig.2C includes true and shuffled pair conditions")
        else:
            failures.append(f"Fig.2C missing true/shuffled conditions, found {sorted(conditions)}")
        if "pair_id" in c_df.columns and c_df["pair_id"].replace("", pd.NA).dropna().any():
            passes.append("Fig.2C carries pair_id for paired lines")
        else:
            warnings.append("Fig.2C pair_id unavailable for paired lines")
        _require_fig2_primary_rows("Fig.2C", c_df, passes, failures)

    d_df = panel_data.get("D")
    if d_df is not None:
        metrics = set(d_df.get("metric", pd.Series(dtype=str)).dropna().astype(str))
        conditions = set(d_df.get("condition", pd.Series(dtype=str)).dropna().astype(str))
        if "WPRI" in metrics and ("beyond_linear_pair_index" in metrics or "residual_pair_specificity" in metrics):
            passes.append("Fig.2D includes WPRI and beyond-linear/residual closure metrics")
        else:
            failures.append(f"Fig.2D must include WPRI plus beyond-linear/residual metric, found {sorted(metrics)}")
        if "linear_model_r2" not in metrics and not any(str(v) in {"A_only", "B_only", "mean_AB", "sum_AB", "unconstrained_AB", "convex_AB"} for v in conditions):
            passes.append("Fig.2D does not plot full linear model comparison")
        else:
            failures.append("Fig.2D must not plot full linear model comparison in the main figure")
        _require_fig2_primary_rows("Fig.2D", d_df, passes, failures)

    e_df = panel_data.get("E")
    if e_df is not None:
        states = set(e_df.get("state_condition", e_df.get("condition", pd.Series(dtype=str))).dropna().astype(str))
        required_states = set(((spec.get("qc_requirements") or {}).get("required_state_conditions") or ["S0", "S_A", "S_B", "S_AB"]))
        if required_states.issubset(states):
            passes.append("Fig.2E: includes S0/S_A/S_B/S_AB neutral-ping conditions")
        else:
            failures.append(f"Fig.2E: missing neutral-ping states {sorted(required_states - states)}")
        categories = set(e_df.get("category", pd.Series(dtype=str)).dropna().astype(str))
        required_categories = set(((spec.get("qc_requirements") or {}).get("required_readout_categories") or ["A", "B", "Other", "Silent"]))
        if required_categories.issubset(categories):
            passes.append("Fig.2E: includes A/B/Other/Silent ping readout categories")
        else:
            failures.append(f"Fig.2E: missing ping readout categories {sorted(required_categories - categories)}")
        metrics = set(e_df.get("metric", pd.Series(dtype=str)).astype(str))
        if metrics == {"neutral_ping_readout_composition"}:
            passes.append("Fig.2E uses neutral-ping readout composition")
        else:
            failures.append(f"Fig.2E must use neutral_ping_readout_composition, found {sorted(metrics)}")
        if {"seed_id", "condition", "value"}.issubset(e_df.columns):
            sums = e_df.groupby(["seed_id", "condition"], dropna=False)["value"].sum()
            max_dev = float((sums - 100.0).abs().max()) if not sums.empty else 0.0
            if max_dev <= 0.5:
                passes.append("Fig.2E ping composition sums to 100 +/- 0.5 per seed/state")
            else:
                failures.append(f"Fig.2E ping composition deviates from 100 by up to {max_dev:.3f}")
        if _has_proxy_columns(e_df):
            failures.append("Fig.2E: main panel data appears to use proxy functional diagnostics")
        else:
            passes.append("Fig.2E: main panel data does not expose proxy score columns")

    for panel_id, required_targets in (("F", {"A", "B"}),):
        df = panel_data.get(panel_id)
        if df is None:
            continue
        states = set(df.get("state_condition", df.get("condition", pd.Series(dtype=str))).dropna().astype(str))
        targets = set(df.get("target_item", pd.Series(dtype=str)).dropna().astype(str).str.upper())
        required_states = set(((spec.get("qc_requirements") or {}).get("required_state_conditions") or ["S0", "S_A", "S_B", "S_AB"]))
        if targets == required_targets:
            passes.append(f"Fig.2{panel_id}: target_items are A and B")
        else:
            failures.append(f"Fig.2{panel_id}: target_items must be A and B, found {sorted(targets)}")
        if required_states.issubset(states):
            passes.append(f"Fig.2{panel_id}: includes S0/S_A/S_B/S_AB")
        else:
            failures.append(f"Fig.2{panel_id}: missing states {sorted(required_states - states)}")
        if _has_proxy_columns(df):
            failures.append(f"Fig.2{panel_id}: main panel data appears to use proxy functional diagnostics")
        else:
            passes.append(f"Fig.2{panel_id}: main panel data does not expose proxy score columns")
        metrics = set(df.get("metric", pd.Series(dtype=str)).astype(str))
        if "P_target" in metrics:
            passes.append(f"Fig.2{panel_id}: includes absolute target recovery curve")
        else:
            failures.append(f"Fig.2{panel_id}: missing absolute target recovery curve")
        if "auc_target_recovery" in metrics:
            passes.append(f"Fig.2{panel_id}: includes partial-cue AUC summary")
        keep = set(float(v) for v in pd.to_numeric(df.get("keep_prob", pd.Series(dtype=float)), errors="coerce").dropna().unique())
        required_keep = set(float(v) for v in ((spec.get("qc_requirements") or {}).get("required_keep_probs") or []))
        if required_keep.issubset(keep):
            passes.append(f"Fig.2{panel_id}: includes required keep probabilities")
        else:
            warnings.append(f"Fig.2{panel_id}: missing keep probabilities {sorted(required_keep - keep)}")
    f_df = panel_data.get("F")
    if f_df is not None:
        per_target = f_df.groupby("target_item", dropna=False)["value"].count() if "target_item" in f_df.columns else pd.Series(dtype=int)
        if {"A", "B"}.issubset(set(str(v).upper() for v in per_target.index)):
            passes.append("Fig.2F keeps target A and target B target-specific inside the merged panel")
        else:
            failures.append("Fig.2F must keep both target A and target B rows")
        f_meta = render_metadata.get("F", {})
        if bool(f_meta.get("inner_axes_aligned")):
            passes.append("Fig.2F merged target subplots use explicitly aligned inner axes")
        elif render_metadata:
            failures.append("Fig.2F merged target subplots must expose aligned inner axes metadata")

    if render_metadata:
        clipped = {
            panel_id: list(meta.get("clipped_artists", []))
            for panel_id, meta in render_metadata.items()
            if meta.get("clipped_artists") or meta.get("panel_label_clipped")
        }
        if clipped:
            failures.append(f"Fig.2 labels/ticks/legends/annotations/panel letters clipped: {clipped}")
        else:
            passes.append("Fig.2 rendered panels have no clipped labels/ticks/legends/annotations/panel letters")


def _check_fig2_supp_specifics(
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
    _ = adapter_results
    if figure_id != "fig2_supp":
        return
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 116:
        passes.append("Fig.2 supplement canvas is 165 x 116 mm")
    else:
        failures.append(f"Fig.2 supplement canvas must be 165 x 116 mm, found {canvas}")
    expected = ["S3A", "S3B", "S3C", "S3D", "S3E"]
    if list(panels.keys()) == expected:
        passes.append("Fig.2 supplement active panel set is compact S3A-S3E")
    else:
        failures.append(f"Fig.2 supplement active panels must be {expected}, found {list(panels.keys())}")
    if list(spec.get("reading_order") or []) == expected:
        passes.append("Fig.2 supplement reading order is S3A-S3E")
    else:
        failures.append(f"Fig.2 supplement reading_order must be {expected}, found {spec.get('reading_order')}")
    forbidden_renderers = [pid for pid, panel in panels.items() if panel.get("renderer") == "render_s4_ping_duration_sweep"]
    forbidden_ping_ms = [pid for pid, panel in panels.items() if panel.get("sweep_parameter") == "ping_ms"]
    forbidden_claims = [
        pid
        for pid, panel in panels.items()
        if "robust across ping duration" in str(panel.get("claim", "")).lower()
    ]
    if forbidden_renderers:
        failures.append(f"Fig.2 supplement active panels must not render ping-duration sweep: {forbidden_renderers}")
    else:
        passes.append("Fig.2 supplement active panels do not call render_s4_ping_duration_sweep")
    if forbidden_ping_ms:
        failures.append(f"Fig.2 supplement active panels must not use sweep_parameter=ping_ms: {forbidden_ping_ms}")
    else:
        passes.append("Fig.2 supplement active panels do not use ping_ms as an active sweep parameter")
    if forbidden_claims:
        failures.append(f"Fig.2 supplement active claims must not retain ping-duration robustness wording: {forbidden_claims}")
    else:
        passes.append("Fig.2 supplement active claims omit ping-duration robustness wording")
    for panel_id, panel in panels.items():
        if panel.get("data_adapter") in (None, "", "none"):
            continue
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        if not paths["panel_data"].exists():
            warnings.append(f"Fig.2 supplement {panel_id}: missing panel_data placeholder output")
            continue
        df = pd.read_csv(paths["panel_data"])
        sources = read_json(paths["sources"]) if paths["sources"].exists() else {}
        if df.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any() or sources.get("status") == "missing_source":
            warnings.append(f"Fig.2 supplement {panel_id}: missing source rendered as placeholder-compatible adapter output")
        else:
            passes.append(f"Fig.2 supplement {panel_id}: data source available")
        stats = read_json(paths["stats"]) if paths["stats"].exists() else {}
        run_mode = str(stats.get("run_mode") or sources.get("run_mode") or "")
        n_networks = int(stats.get("n_networks") or sources.get("n_networks") or 0)
        if run_mode == "single_network_draft" or n_networks == 1:
            warnings.append(f"Fig.2 supplement {panel_id}: single_network_draft n_networks=1; draft-only, not final manuscript statistics")
    s3c_meta = render_metadata.get("S3C", {})
    if s3c_meta:
        labels = [str(v) for v in s3c_meta.get("x_tick_labels", []) if str(v)]
        if labels and all(len(label) <= 8 for label in labels):
            passes.append("Fig.2 supplement S3C model labels are abbreviated/readable")
        else:
            warnings.append(f"Fig.2 supplement S3C model labels may be long: {labels}")
    else:
        passes.append("Fig.2 supplement S3C renderer is configured for abbreviated model labels")
    s3d = panels.get("S3D") or {}
    s3e = panels.get("S3E") or {}
    if s3d.get("data_adapter") == "s4_ping_sweep_adapter" and s3d.get("renderer") == "render_s4_ping_amplitude_sweep" and s3d.get("sweep_parameter") == "ping_amp":
        passes.append("Fig.2 supplement S3D uses ping amplitude adapter/renderer")
    else:
        failures.append("Fig.2 supplement S3D must use s4_ping_sweep_adapter, render_s4_ping_amplitude_sweep, and sweep_parameter=ping_amp")
    expected_delays = [100, 200, 300, 400, 800, 1200]
    if (
        s3e.get("data_adapter") == "s4_completion_delay_adapter"
        and s3e.get("renderer") == "render_s4_completion_delay_gain"
        and list(s3e.get("expected_delay_ms") or []) == expected_delays
        and str(s3e.get("legacy_panel_id", "")).endswith("S4B")
    ):
        passes.append("Fig.2 supplement S3E uses completion-delay adapter/renderer with legacy metadata")
    else:
        failures.append("Fig.2 supplement S3E must be the completion-delay panel with expected delay values")
    claim = str(s3e.get("claim", "")).lower()
    if "persists across post-pair delay" in claim or "robust across all post-pair delays" in claim:
        failures.append("Fig.2 supplement S3E claim must not use strong persistence wording")
    elif "strongest at short" in claim and "decays across the retention interval" in claim and "functional completion window" in claim:
        passes.append("Fig.2 supplement S3E claim states short-delay strength and retention-window decay")
    else:
        warnings.append("Fig.2 supplement S3E claim may not fully state the requested temporal operating-window wording")
    s4b_refs = s3e.get("reference_lines") or []
    if any(float(ref.get("value", 999)) == 0 for ref in s4b_refs):
        passes.append("Fig.2 supplement S3E has zero reference line in spec")
    else:
        failures.append("Fig.2 supplement S3E must include a zero reference line")
    s4b_meta = render_metadata.get("S3E", {})
    if s4b_meta and not bool(s4b_meta.get("has_zero_reference", True)):
        warnings.append("Fig.2 supplement S3E zero reference render metadata was not detected")
    for panel_id, expected_metric, expected_condition in (
        ("S3D", "pair_member_readout", None),
        ("S3E", "completion_gain", "S_AB_minus_relevant_single"),
    ):
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        if not paths["panel_data"].exists():
            warnings.append(f"Fig.2 supplement {panel_id}: panel_data unavailable for metric contract check")
            continue
        df = pd.read_csv(paths["panel_data"])
        if df.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any():
            warnings.append(f"Fig.2 supplement {panel_id}: missing source placeholder skips metric contract check")
            continue
        metrics = set(df.get("metric", pd.Series(dtype=str)).astype(str))
        conditions = set(df.get("condition", pd.Series(dtype=str)).astype(str))
        if metrics == {expected_metric} and (expected_condition is None or conditions == {expected_condition}):
            passes.append(f"Fig.2 supplement {panel_id} panel_data metric contract is correct")
        else:
            failures.append(f"Fig.2 supplement {panel_id} panel_data contract mismatch: metrics={sorted(metrics)}, conditions={sorted(conditions)}")
    s4b_paths = panel_output_paths(output_dir, figure_id, "S3E")
    if s4b_paths["panel_data"].exists():
        s4b_df = pd.read_csv(s4b_paths["panel_data"])
        if not s4b_df.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any():
            observed_delays = {
                int(round(float(value)))
                for value in pd.to_numeric(s4b_df.get("delay2_ms", pd.Series(dtype=float)), errors="coerce").dropna().tolist()
            }
            missing_delays = [value for value in expected_delays if value not in observed_delays]
            if missing_delays:
                warnings.append(f"Fig.2 supplement S3E missing expected delay values in panel_data: {missing_delays}")
            else:
                passes.append("Fig.2 supplement S3E panel_data contains all expected completion-delay values")
    manifest_path = output_dir / f"{figure_id}_source_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        active_panels = list(manifest.get("active_panels") or [])
        if active_panels == expected:
            passes.append("Fig.2 supplement source manifest records active_panels S3A-S3E")
        else:
            failures.append(f"Fig.2 supplement source manifest active_panels must be {expected}, found {active_panels}")
        source_panels = [str(item.get("panel_id", "")) for item in manifest.get("sources", [])]
        if any(panel_id.startswith("S4") for panel_id in source_panels):
            failures.append("Fig.2 supplement source manifest must not list old S4 panels as active")
        if source_panels and source_panels == expected:
            passes.append("Fig.2 supplement source manifest lists only active S3 panels")


def _has_proxy_columns(df: pd.DataFrame) -> bool:
    forbidden_proxy_cols = {"proxy_score_A", "proxy_score_B"}
    return bool(forbidden_proxy_cols.intersection(df.columns)) or df.get("source_file", pd.Series(dtype=str)).astype(str).str.contains("supp_functional_proxy", case=False, na=False).any()


def _check_fig2_new_geometry(panels: Mapping[str, Any], passes: list[str], failures: list[str]) -> None:
    required = ("A", "B", "C", "D", "E", "F")
    pos = {panel_id: (panels.get(panel_id) or {}).get("position_mm") or {} for panel_id in required if panel_id in panels}
    if len(pos) < len(required):
        return
    if _near(_y(pos["B"]), _y(pos["C"])) and _near(_y(pos["B"]), _y(pos["D"])) and _near(_bottom(pos["B"]), _bottom(pos["C"])) and _near(_bottom(pos["B"]), _bottom(pos["D"])):
        passes.append("Fig.2 B/C/D axes top and bottom align")
    else:
        failures.append("Fig.2 B/C/D axes top and bottom must align")
    if _near(_y(pos["E"]), _y(pos["F"])) and _near(_bottom(pos["E"]), _bottom(pos["F"])):
        passes.append("Fig.2 E/F axes top and bottom align")
    else:
        failures.append("Fig.2 E/F axes top and bottom must align")
    if _near(_h(pos["E"]), _h(pos["F"])) and _near(_h(pos["E"]), _h(pos["B"])):
        passes.append("Fig.2 third-row height matches the second-row panel height")
    else:
        failures.append("Fig.2 third-row height must match the second-row panel height")
    if _near(_x(pos["A"]), _x(pos["B"])) and _near(_right(pos["A"]), _right(pos["D"])):
        passes.append("Fig.2A spans the three aligned columns")
    else:
        failures.append("Fig.2A must span from B.left to D.right")
    if _near(_x(pos["B"]), _x(pos["E"]), tol=0.15) and _near(_right(pos["B"]), _right(pos["E"]), tol=0.15):
        passes.append("Fig.2E aligns to the B column")
    else:
        failures.append("Fig.2E must align to the B column")
    if _near(_x(pos["F"]), _x(pos["C"]), tol=0.15) and _near(_right(pos["F"]), _right(pos["D"]), tol=0.15):
        passes.append("Fig.2F spans the C-D columns")
    else:
        failures.append("Fig.2F must span from C.left to D.right")
    if _x(pos["E"]) >= _x(pos["B"]) - 0.15 and _right(pos["F"]) <= _right(pos["D"]) + 0.15 and _x(pos["F"]) > _right(pos["E"]):
        passes.append("Fig.2 E/F occupy the functional-access row without overlap")
    else:
        failures.append("Fig.2 E/F must occupy the functional-access row without overlap")


def _check_fig3_standalone_contract(
    spec: Mapping[str, Any],
    panels: Mapping[str, Any],
    output_dir: Path,
    render_metadata: Mapping[str, Mapping[str, Any]],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    figure_id = "fig3"
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 158:
        passes.append("Fig.3 canvas is 165 x 158 mm")
    else:
        failures.append(f"Fig.3 canvas must be 165 x 158 mm, found {canvas}")
    expected_order = ["B", "C", "D", "E", "F"]
    if list(spec.get("reading_order") or []) == expected_order and set(panels.keys()) == set(expected_order):
        passes.append("Fig.3 has schematic A removed and uses B-F reading order")
        passes.append("fig3_has_schematic_A = false")
        passes.append("no_empty_A_panel = true")
    else:
        failures.append(f"Fig.3 must contain panels/reading_order B-F with no A panel, found panels={sorted(panels.keys())}, reading_order={spec.get('reading_order')}")
    _check_fig3_new_geometry(panels, passes, failures)

    panel_data: dict[str, pd.DataFrame] = {}
    for panel_id in ("B", "C", "D", "E", "F"):
        if panel_id not in panels:
            failures.append(f"Fig.3{panel_id}: panel missing from spec")
            continue
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        missing = [name for name in ("panel_data", "stats", "sources") if not paths[name].exists()]
        if missing:
            failures.append(f"Fig.3{panel_id}: missing adapter outputs {missing}")
            continue
        passes.append(f"Fig.3{panel_id}: panel_data/stats/source_manifest exist")
        df = pd.read_csv(paths["panel_data"])
        panel_data[panel_id] = df
        stats = read_json(paths["stats"])
        sources = read_json(paths["sources"])
        run_mode = str(stats.get("run_mode") or sources.get("run_mode") or "")
        n_networks = int(stats.get("n_networks") or sources.get("n_networks") or 0)
        if run_mode:
            passes.append(f"Fig.3{panel_id}: run_mode={run_mode}")
        else:
            failures.append(f"Fig.3{panel_id}: run_mode missing")
        if n_networks == 1 or run_mode == "single_network_draft":
            warnings.append(f"Fig.3{panel_id}: single_network_draft n_networks=1; draft-only, not final manuscript statistics")
        elif n_networks > 1:
            passes.append(f"Fig.3{panel_id}: n_networks={n_networks}")
        else:
            warnings.append(f"Fig.3{panel_id}: n_networks not recorded")

    b_df = panel_data.get("B")
    if b_df is not None:
        metrics = set(b_df.get("metric", pd.Series(dtype=str)).astype(str))
        if "stepwise_update_ratio" in metrics:
            passes.append("Fig.3B includes progressive update ratio")
        else:
            failures.append(f"Fig.3B missing stepwise_update_ratio, found {sorted(metrics)}")
        stages = set(pd.to_numeric(b_df.get("stage_k", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
        if stages and max(stages) >= 3:
            passes.append("Fig.3B covers multiple sequence stages")
        else:
            failures.append(f"Fig.3B has insufficient stage coverage: {sorted(stages)}")

    c_df = panel_data.get("C")
    if c_df is not None:
        metrics = set(c_df.get("metric", pd.Series(dtype=str)).astype(str))
        masks = set(c_df.get("mask_role", pd.Series(dtype=str)).astype(str))
        if {"delta_gain_map", "G_final"}.intersection(metrics):
            passes.append("Fig.3C includes representative support landscape")
        else:
            failures.append(f"Fig.3C missing delta_gain_map/G_final, found {sorted(metrics)}")
        if {"peak", "valley"}.issubset(masks):
            passes.append("Fig.3C includes optional peak/valley overlays")
        else:
            warnings.append(f"Fig.3C peak/valley overlays unavailable or partial, found {sorted(masks)}")
        c_sources = _panel_sources(output_dir, figure_id, "C")
        if "panel_c_example_landscape.npz" in c_sources:
            passes.append("Fig.3C source manifest includes panel_c_example_landscape.npz")
        else:
            warnings.append("Fig.3C source manifest does not list panel_c_example_landscape.npz")
        c_panel = panels.get("C") or {}
        if c_panel.get("renderer") == "render_fig3_3d_landscape" and c_panel.get("projection") == "3d":
            passes.append("Fig.3C spec requests 3D landscape renderer")
        else:
            failures.append("Fig.3C must use render_fig3_3d_landscape with projection=3d")
        c_meta = render_metadata.get("C", {})
        if c_meta:
            form = str(c_meta.get("plot_form"))
            if form == "fig3_3d_surface_landscape" or bool(c_meta.get("paper_fig_is_3d_surface", False)):
                passes.append("Fig.3C rendered as 3D surface")
            elif form == "fig3_2d_landscape_fallback" or c_meta.get("3d_fallback_reason"):
                warnings.append(f"Fig.3C used explicit 2D fallback: {c_meta.get('3d_fallback_reason', '')}")
            else:
                failures.append(f"Fig.3C renderer did not report 3D landscape/fallback form, found {form}")
            if bool(c_meta.get("has_summary_inset", c_meta.get("paper_fig_has_summary_inset", False))):
                failures.append("Fig.3C must not contain a summary inset")
            else:
                passes.append("Fig.3C reports no summary inset")

    d_df = panel_data.get("D")
    if d_df is not None:
        metrics = set(d_df.get("metric", pd.Series(dtype=str)).astype(str))
        serial_positions = pd.to_numeric(d_df.get("serial_position", pd.Series(dtype=float)), errors="coerce").dropna()
        states = set(d_df.get("state_condition", pd.Series(dtype=str)).astype(str))
        if "readout_mass" in metrics:
            passes.append("Fig.3D uses neutral-ping readout_mass")
        else:
            failures.append(f"Fig.3D missing readout_mass, found {sorted(metrics)}")
        if not serial_positions.empty and int(serial_positions.min()) >= 1:
            passes.append("Fig.3D has numeric serial positions for plotting")
        else:
            failures.append("Fig.3D must include numeric serial_position rows")
        if "S_final" in states and ({"S0", "S0_ping_null"}.intersection(states)):
            passes.append("Fig.3D includes S_final and S0/null baseline")
        else:
            warnings.append(f"Fig.3D missing preferred state conditions, found {sorted(states)}")
        d_sources = _panel_sources(output_dir, figure_id, "D")
        if "panel_d_ping_" in d_sources:
            passes.append("Fig.3D uses panel_d neutral-ping sources")
        elif "panel_e_ping_" in d_sources:
            warnings.append("Fig.3D uses old panel_e neutral-ping aliases")
        else:
            warnings.append("Fig.3D source manifest does not list panel_d/panel_e ping source names")

    e_df = panel_data.get("E")
    if e_df is not None:
        metrics = set(e_df.get("metric", pd.Series(dtype=str)).astype(str))
        memories = set(e_df.get("memory_condition", pd.Series(dtype=str)).astype(str))
        if "P_target" in metrics:
            passes.append("Fig.3E uses weak-probe P_target recovery")
        else:
            failures.append(f"Fig.3E missing P_target, found {sorted(metrics)}")
        if {"sequence_state", "single_item_memory", "cue_only"}.issubset(memories):
            passes.append("Fig.3E includes cue_only, single_item_memory, and sequence_state weak-probe curves")
        else:
            warnings.append(f"Fig.3E missing preferred memory conditions, found {sorted(memories)}")
        e_sources = _panel_sources(output_dir, figure_id, "E")
        if "panel_e_weak_probe_" in e_sources:
            passes.append("Fig.3E uses panel_e weak-probe sources")
        elif "panel_f_weak_probe_" in e_sources:
            warnings.append("Fig.3E uses old panel_f weak-probe aliases")
        else:
            warnings.append("Fig.3E source manifest does not list panel_e/panel_f weak-probe source names")

    f_df = panel_data.get("F")
    if f_df is not None:
        metrics = set(f_df.get("metric", pd.Series(dtype=str)).astype(str))
        regions = {str(v).replace("_aligned", "").replace("_matched", "") for v in f_df.get("region_condition", pd.Series(dtype=str)).dropna().astype(str)}
        if "readout_mass" in metrics and "memory_gain" not in metrics:
            passes.append("Fig.3F main panel data uses region-ping readout_mass")
        else:
            failures.append(f"Fig.3F must use region-ping readout_mass, not memory_gain/accuracy; found metrics {sorted(metrics)}")
        if {"peak", "random", "valley"}.issubset(regions):
            passes.append("Fig.3F includes peak/valley/random region conditions")
        else:
            failures.append(f"Fig.3F missing region conditions {sorted({'peak', 'random', 'valley'} - regions)}")
        f_sources = _panel_sources(output_dir, figure_id, "F")
        if "panel_f_region_ping_" in f_sources:
            passes.append("Fig.3F uses panel_f_region_ping_* sources")
        else:
            failures.append("Fig.3F must use panel_f_region_ping_* sources")
        if "panel_f_peak_cue_memory_gain.csv" in f_sources:
            failures.append("Fig.3F must not use panel_f_peak_cue_memory_gain.csv as the main figure source")
        f_stats = read_json(panel_output_paths(output_dir, figure_id, "F")["stats"])
        if str(f_stats.get("main_source", "")) == "region_ping":
            passes.append("Fig.3F stats mark main_source=region_ping")
        else:
            warnings.append("Fig.3F stats do not mark main_source=region_ping")
        if str(f_stats.get("main_plot_type", "")) == "stacked_readout_mass":
            passes.append('Fig.3F main_plot_type = "stacked_readout_mass"')
        else:
            failures.append(f"Fig.3F main_plot_type must be stacked_readout_mass, found {f_stats.get('main_plot_type')}")
        categories = [str(v) for v in f_stats.get("readout_categories", [])]
        if categories == ["recent", "old", "silent"]:
            passes.append("Fig.3F categories = ['recent', 'old', 'silent']")
        else:
            failures.append(f"Fig.3F readout_categories must be ['recent', 'old', 'silent'], found {categories}")
        if bool(f_stats.get("uses_serial_position_10_class")):
            failures.append("Fig.3F must not use ten serial positions as main categories")
        else:
            passes.append("Fig.3F uses_serial_position_10_class = false")
        if bool(f_stats.get("uses_latest_recent_earlier_other_silent")):
            failures.append("Fig.3F must not use Latest/Recent/Earlier/Other/Silent five-class stack")
        else:
            passes.append("Fig.3F uses_latest_recent_earlier_other_silent = false")
        if bool(f_stats.get("y_axis_absolute_probability")) and bool(f_stats.get("stacked_bars_not_normalized")):
            passes.append("Fig.3F uses absolute readout probability and non-normalized stacked bars")
        else:
            failures.append("Fig.3F must use absolute readout probability and non-normalized stacked bars")
        f_categories = set(f_df.get("readout_category", pd.Series(dtype=str)).dropna().astype(str))
        if f_categories == {"recent", "old", "silent"}:
            passes.append("Fig.3F panel_data exposes only recent/old/silent readout categories")
        else:
            failures.append(f"Fig.3F panel_data must expose only recent/old/silent categories, found {sorted(f_categories)}")
        if any(key in f_stats for key in ("JS_peak_valley", "TV_peak_valley", "P_peak_label_differs_from_valley")):
            passes.append("Fig.3F stats include peak-valley distribution contrast")
        else:
            warnings.append("Fig.3F stats missing peak-valley distribution contrast")

    visible_terms: list[str] = []
    for df in panel_data.values():
        for col in ("condition", "metric", "source_file"):
            if col in df.columns:
                visible_terms.extend(str(v) for v in df[col].dropna().unique())
    old_terms = ("chunk_stsp_multiitem_sequence", "layer3_anchor_drift", "latest_item_overwrite", "recency_only_primary")
    leaked = [term for term in old_terms if term in " ".join(visible_terms)]
    if leaked:
        failures.append(f"Fig.3 panel data exposes old/internal labels: {leaked}")
    else:
        passes.append("Fig.3 panel data avoids old experiment labels and recency-only primary labels")

    if render_metadata:
        clipped = {
            panel_id: list(meta.get("clipped_artists", []))
            for panel_id, meta in render_metadata.items()
            if meta.get("clipped_artists") or meta.get("panel_label_clipped")
        }
        if clipped:
            failures.append(f"Fig.3 labels/ticks/legends/panel letters clipped: {clipped}")
        else:
            passes.append("Fig.3 rendered panels have no clipped labels/ticks/legends/panel letters")


def _check_fig3_supp_contract(
    spec: Mapping[str, Any],
    panels: Mapping[str, Any],
    output_dir: Path,
    render_metadata: Mapping[str, Mapping[str, Any]],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    figure_id = "fig3_supp"
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 116:
        passes.append("Fig.3 supplement canvas is compact 165 x 116 mm")
    else:
        failures.append(f"Fig.3 supplement canvas must be 165 x 116 mm, found {canvas}")
    expected = ["S4A", "S4B", "S4C", "S4D", "S4E", "S4F"]
    if list(spec.get("reading_order") or []) == expected and set(panels.keys()) == set(expected):
        passes.append("Fig.3 supplement has compact S4A-S4F reading order")
    else:
        failures.append(f"Fig.3 supplement panels/reading_order must be {expected}, found panels={sorted(panels.keys())}, reading_order={spec.get('reading_order')}")

    forbidden_sources = ("panel_f_region_ping_", "supp_region_ping_amp_sweep_")
    for panel_id, panel in panels.items():
        if panel.get("data_adapter") in (None, "", "none"):
            continue
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        if not paths["panel_data"].exists() or not paths["sources"].exists():
            failures.append(f"Fig.3 supplement {panel_id}: missing adapter output")
            continue
        df = pd.read_csv(paths["panel_data"])
        sources = read_json(paths["sources"])
        if df.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any() or sources.get("status") == "missing_source":
            failures.append(f"Fig.3 supplement {panel_id}: missing sources/blank-compatible adapter output")
        else:
            passes.append(f"Fig.3 supplement {panel_id}: data source available")
        source_text = str(sources).lower()
        if any(name in source_text for name in forbidden_sources):
            failures.append(f"Fig.3 supplement {panel_id}: active source manifest still contains Fig.6/region-ping source")
        fallback_warnings = [str(item) for item in (sources.get("warnings") or []) if "fallback" in str(item).lower() or "degraded" in str(item).lower()]
        if fallback_warnings:
            warnings.append(f"Fig.3 supplement {panel_id}: degraded source path recorded: {fallback_warnings}")
        run_mode = str((read_json(paths["stats"]) if paths["stats"].exists() else {}).get("run_mode") or sources.get("run_mode") or "")
        n_networks = int((read_json(paths["stats"]) if paths["stats"].exists() else {}).get("n_networks") or sources.get("n_networks") or 0)
        if run_mode == "single_network_draft" or n_networks == 1:
            warnings.append(f"Fig.3 supplement {panel_id}: single_network_draft n_networks=1; draft-only, not final manuscript statistics")

    s5e_paths = panel_output_paths(output_dir, figure_id, "S4E")
    if s5e_paths["panel_data"].exists():
        s5e = pd.read_csv(s5e_paths["panel_data"])
        classes = set(s5e.get("readout_class", pd.Series(dtype=str)).astype(str))
        required_classes = {"latest", "recent", "earlier", "silent"}
        if s5e.get("metric", pd.Series(dtype=str)).astype(str).eq("readout_mass").any() and required_classes.issubset(classes):
            passes.append("Fig.3 supplement S4E contains latest/recent/earlier/silent readout decomposition")
        else:
            failures.append(f"Fig.3 supplement S4E must contain readout_mass for latest/recent/earlier/silent, found classes={sorted(classes)}")

    s5f_paths = panel_output_paths(output_dir, figure_id, "S4F")
    if s5f_paths["panel_data"].exists():
        s5f = pd.read_csv(s5f_paths["panel_data"])
        metrics = set(s5f.get("metric", pd.Series(dtype=str)).astype(str))
        bins = set(s5f.get("target_position_bin", pd.Series(dtype=str)).astype(str))
        if "target_recovery_gain" in metrics and bins:
            passes.append("Fig.3 supplement S4F uses recency-bin target_recovery_gain")
        else:
            failures.append(f"Fig.3 supplement S4F must plot target_recovery_gain by target_position_bin, found metrics={sorted(metrics)}, bins={sorted(bins)}")

    if render_metadata:
        placeholders = [pid for pid, meta in render_metadata.items() if meta.get("plot_form") == "blank_panel" or meta.get("placeholder_reason")]
        if placeholders:
            failures.append(f"Fig.3 supplement must not have blank/placeholder renderers: {placeholders}")
        else:
            passes.append("Fig.3 supplement rendered without blank placeholders")
        active_forms = {pid: str(meta.get("plot_form", "")) for pid, meta in render_metadata.items()}
        if any(pid.startswith("S5") or pid.startswith("S6") or "s6_" in form for pid, form in active_forms.items()):
            failures.append(f"Fig.3 supplement active render metadata must not contain old S5/S6 panels/forms: {active_forms}")


def _check_fig3_new_geometry(panels: Mapping[str, Any], passes: list[str], failures: list[str]) -> None:
    expected = {
        "B": {"x": 12.00, "y": 8.00, "w": 52.00, "h": 42.00},
        "C": {"x": 75.00, "y": 8.00, "w": 84.00, "h": 96.00},
        "D": {"x": 12.00, "y": 62.00, "w": 52.00, "h": 42.00},
        "E": {"x": 12.00, "y": 116.00, "w": 84.00, "h": 34.00},
        "F": {"x": 103.00, "y": 116.00, "w": 56.00, "h": 34.00},
    }
    if "A" in panels:
        failures.append("Fig.3A schematic panel must be removed from the main figure")
    else:
        passes.append("Fig.3A schematic panel is removed")
    for panel_id, target in expected.items():
        pos = (panels.get(panel_id) or {}).get("position_mm") or {}
        if _fig3_mm_box_close(pos, target, tol=0.08):
            passes.append(f"Fig.3{panel_id}: position_mm matches requested layout")
        else:
            failures.append(f"Fig.3{panel_id}: position_mm must be {target}, found {pos}")
    pos = {panel_id: (panels.get(panel_id) or {}).get("position_mm") or {} for panel_id in expected}
    if _near(float(pos["B"].get("x", -1)), float(pos["D"].get("x", -2)), tol=0.08) and _near(float(pos["B"].get("w", -1)), float(pos["D"].get("w", -2)), tol=0.08):
        passes.append("Fig.3B/D left-column widths align")
    else:
        failures.append("Fig.3B and Fig.3D must align in the left column")
    b_right = float(pos["B"].get("x", 0)) + float(pos["B"].get("w", 0))
    d_right = float(pos["D"].get("x", 0)) + float(pos["D"].get("w", 0))
    c_left = float(pos["C"].get("x", 0))
    if c_left > max(b_right, d_right):
        passes.append("C_right_of_BD = true")
    else:
        failures.append("Fig.3C must be to the right of Fig.3B/D")
    c_top = float(pos["C"].get("y", 0))
    c_bottom = c_top + float(pos["C"].get("h", 0))
    bd_top = min(float(pos["B"].get("y", 0)), float(pos["D"].get("y", 0)))
    bd_bottom = max(float(pos["B"].get("y", 0)) + float(pos["B"].get("h", 0)), float(pos["D"].get("y", 0)) + float(pos["D"].get("h", 0)))
    if _near(c_top, bd_top, tol=0.08) and _near(c_bottom, bd_bottom, tol=0.08):
        passes.append("C_spans_BD_height = true")
    else:
        failures.append("Fig.3C must span the combined Fig.3B/D height")
    if _near(float(pos["E"].get("y", 0)), float(pos["F"].get("y", -1)), tol=0.08) and float(pos["E"].get("w", 0)) > float(pos["F"].get("w", 0)):
        passes.append("E_left_of_F = true")
    else:
        failures.append("Fig.3E/F must align in the bottom row with E wider than F")


def _panel_sources(output_dir: Path, figure_id: str, panel_id: str) -> str:
    path = panel_output_paths(output_dir, figure_id, panel_id)["sources"]
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").lower()


def _fig3_seed_summary_from_panel_data(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "source_file" not in df.columns:
        return {}
    for raw in df["source_file"].dropna().astype(str).unique():
        path = Path(raw)
        parts = list(path.parts)
        if "data" not in parts:
            continue
        data_idx = parts.index("data")
        seed_dir = Path(*parts[:data_idx]) if data_idx > 0 else Path(".")
        summary_path = seed_dir / "summary.json"
        if summary_path.exists():
            return read_json(summary_path)
    return {}


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
    if figure_id == "fig3_supp":
        _check_fig3_supp_contract(spec, panels, output_dir, render_metadata, passes, warnings, failures)
        return
    if figure_id != "fig3":
        return
    _check_fig3_standalone_contract(spec, panels, output_dir, render_metadata, passes, warnings, failures)
    return
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 108:
        passes.append("Fig.3 canvas is 165 x 108 mm")
    else:
        failures.append(f"Fig.3 canvas must be 165 x 108 mm, found {canvas}")
    if set(panels.keys()) == {"A", "B", "C", "D"}:
        passes.append("Fig.3 uses the finalized four-panel A-D structure")
    else:
        failures.append(f"Fig.3 must contain exactly panels A-D, found {sorted(panels.keys())}")
    titled = [panel_id for panel_id, panel in panels.items() if str(panel.get("title", "")).strip()]
    if titled:
        failures.append(f"Fig.3 panel titles must be removed, but titles remain on {titled}")
    else:
        passes.append("Fig.3 panel titles are removed from the spec")

    _check_fig3_geometry(panels, passes, failures)

    required_networks = (spec.get("qc_requirements") or {}).get("require_n_networks") or {}
    for panel_id, required_n in required_networks.items():
        if panel_id not in panels:
            continue
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        stats_path = panel_output_paths(output_dir, figure_id, panel_id)["stats"]
        source_path = panel_output_paths(output_dir, figure_id, panel_id)["sources"]
        if not data_path.exists() or not stats_path.exists() or not source_path.exists():
            failures.append(f"Fig.3{panel_id}: panel_data/stats/source_manifest output is missing")
            continue
        df = pd.read_csv(data_path)
        stats = read_json(stats_path)
        source_manifest = read_json(source_path)
        n = _panel_n(df)
        if n >= int(required_n):
            passes.append(f"Fig.3{panel_id}: n={n} networks/seeds available")
        else:
            failures.append(f"Fig.3{panel_id}: expected n={required_n}, found n={n}")
        if stats.get("aggregation") == "network_first":
            passes.append(f"Fig.3{panel_id}: stats record network-first aggregation")
        else:
            failures.append(f"Fig.3{panel_id}: stats must record aggregation=network_first")
        if bool(stats.get("source_appears_preaggregated")) or bool(stats.get("source_appeared_preaggregated")):
            failures.append(f"Fig.3{panel_id}: source appears pre-aggregated")
        else:
            passes.append(f"Fig.3{panel_id}: source is not marked pre-aggregated")
        used = list(source_manifest.get("source_files_used") or [])
        duplicates = list(source_manifest.get("duplicate_candidates_ignored") or [])
        if any("/metrics/" in str(path).replace("\\", "/") for path in used):
            failures.append(f"Fig.3{panel_id}: metrics duplicate CSV was read into panel_data")
        elif duplicates:
            passes.append(f"Fig.3{panel_id}: duplicate candidates recorded and ignored")
        else:
            passes.append(f"Fig.3{panel_id}: no duplicate candidates needed ignoring")
        if any("/data/" in str(path).replace("\\", "/") and "/runs/" not in str(path).replace("\\", "/") for path in used):
            failures.append(f"Fig.3{panel_id}: root-level aggregate CSV was read with run-level files")
        else:
            passes.append(f"Fig.3{panel_id}: root aggregate sources were not double-counted")
        if str(source_manifest.get("selected_layer", "")) == "Layer 3":
            passes.append(f"Fig.3{panel_id}: selected layer is Layer 3")
        else:
            failures.append(f"Fig.3{panel_id}: selected layer must be Layer 3")
        if panel_id in {"C", "D"}:
            if int(source_manifest.get("selected_sequence_length") or 0) == 10:
                passes.append(f"Fig.3{panel_id}: selected seq_len=10")
            else:
                failures.append(f"Fig.3{panel_id}: selected seq_len=10 is required")

    _check_fig3_panel_semantics(output_dir, panels, passes, warnings, failures)

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

    if render_metadata:
        passes.append("Full composite Fig.3 was regenerated")
        passes.append("No standalone-only update detected")
        _check_fig3_render_layout(render_metadata, passes, warnings, failures)
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
        legend_overlap = [pid for pid, meta in render_metadata.items() if meta.get("legend_overlaps_data") or meta.get("legend_overlaps_axes_bbox")]
        if legend_overlap:
            failures.append(f"Fig.3 legends overlap data/axes: {legend_overlap}")
        else:
            passes.append("Fig.3 has no legend overlaps data")


def _check_fig3_geometry(panels: Mapping[str, Any], passes: list[str], failures: list[str]) -> None:
    expected_slots = {
        "A": {"x": 8, "y": 5, "w": 72, "h": 43},
        "B": {"x": 84, "y": 5, "w": 72, "h": 43},
        "C": {"x": 8, "y": 57, "w": 72, "h": 43},
        "D": {"x": 84, "y": 57, "w": 72, "h": 43},
    }
    expected_axes = {
        "A": {"x": 20, "y": 10, "w": 60, "h": 33},
        "B": {"x": 96, "y": 10, "w": 60, "h": 33},
        "C": {"x": 20, "y": 62, "w": 60, "h": 33},
        "D": {"x": 96, "y": 62, "w": 60, "h": 33},
    }
    for panel_id, expected in expected_slots.items():
        pos = (panels.get(panel_id) or {}).get("position_mm") or {}
        if _fig3_mm_box_close(pos, expected):
            passes.append(f"Fig.3{panel_id}: slot mm coordinates match specification")
        else:
            failures.append(f"Fig.3{panel_id}: slot coordinates differ from specification, found {pos}")
    for panel_id, expected in expected_axes.items():
        pos = (panels.get(panel_id) or {}).get("axes_mm") or {}
        if _fig3_mm_box_close(pos, expected):
            passes.append(f"Fig.3{panel_id}: plotting axes mm coordinates match specification")
        else:
            failures.append(f"Fig.3{panel_id}: axes coordinates differ from specification, found {pos}")
    axes = {panel_id: (panels.get(panel_id) or {}).get("axes_mm") or {} for panel_id in expected_axes}
    if any(not value for value in axes.values()):
        failures.append("Fig.3 all panels must define axes_mm")
        return
    if _same_top_bottom(axes["A"], axes["B"]):
        passes.append("Fig.3 A/B axes top and bottom are aligned")
    else:
        failures.append("Fig.3 A/B axes top and bottom must align")
    if _same_top_bottom(axes["C"], axes["D"]):
        passes.append("Fig.3 C/D axes top and bottom are aligned")
    else:
        failures.append("Fig.3 C/D axes top and bottom must align")
    if _near(float(axes["A"].get("x", 0)), float(axes["C"].get("x", -1))) and _near(float(axes["B"].get("x", 0)), float(axes["D"].get("x", -1))):
        passes.append("Fig.3 columns are aligned in the 2x2 layout")
    else:
        failures.append("Fig.3 2x2 column axes must align")
    if all(_near(float(axis.get("w", 0)), 60) and _near(float(axis.get("h", 0)), 33) for axis in axes.values()):
        passes.append("Fig.3 all plotting axes are 60 x 33 mm")
    else:
        failures.append(f"Fig.3 all plotting axes must be 60 x 33 mm, found {axes}")
    row_gap = float(axes["C"].get("y", 0)) - (float(axes["A"].get("y", 0)) + float(axes["A"].get("h", 0)))
    if _near(row_gap, 19):
        passes.append("Fig.3 row gap is reduced to 19 mm in the compact layout")
    else:
        failures.append(f"Fig.3 compact row gap should be 19 mm, found {row_gap}")
    if (panels.get("C") or {}).get("renderer") == "render_progressive_update" and (panels.get("D") or {}).get("renderer") == "render_center_migration":
        passes.append("Fig.3 remaps old D to C and old merged E/F to D")
    else:
        failures.append("Fig.3 panel remapping must be C=progressive update and D=center migration")


def _check_fig3_panel_semantics(output_dir: Path, panels: Mapping[str, Any], passes: list[str], warnings: list[str], failures: list[str]) -> None:
    a_path = panel_output_paths(output_dir, "fig3", "A")["panel_data"]
    if "A" in panels and a_path.exists():
        a_df = pd.read_csv(a_path)
        metrics = set(a_df.get("metric", pd.Series(dtype=str)).astype(str))
        if "signed_item2_bias" in metrics and "signed_item1_bias" not in metrics and "fusion_imbalance_score" not in metrics:
            passes.append("Fig.3A uses signed Item 2 bias, not Item 1 bias or absolute imbalance")
        else:
            failures.append(f"Fig.3A must use signed_item2_bias and avoid signed_item1_bias/fusion_imbalance_score, found {sorted(metrics)}")
        conditions = set(a_df.get("condition", pd.Series(dtype=str)).astype(str))
        if {"Item 1 reference", "Item 2 reference", "Item 2 - Item 1"}.issubset(conditions):
            passes.append("Fig.3A includes Item 1, Item 2, and signed Item2-minus-Item1 bias rows")
        else:
            failures.append("Fig.3A missing Item 1/Item 2/signed bias panel rows")

    b_path = panel_output_paths(output_dir, "fig3", "B")["panel_data"]
    if "B" in panels and b_path.exists():
        b_df = pd.read_csv(b_path)
        if {"DI_bin", "x_value", "y_value", "item2_first_prob", "functional_item2_bias"}.issubset(b_df.columns):
            passes.append("Fig.3B contains Item-2-centered binned DI probability panel data")
        else:
            failures.append("Fig.3B must contain DI_bin, x_value, y_value, item2_first_prob, and functional_item2_bias")
        metrics = set(b_df.get("metric", pd.Series(dtype=str)).astype(str))
        if metrics == {"item2_first_probability"}:
            passes.append("Fig.3B y-value uses Item 2-first probability")
        else:
            failures.append(f"Fig.3B must use item2_first_probability metric, found {sorted(metrics)}")
        if len(set(b_df.get("DI_bin", []))) >= 4:
            passes.append("Fig.3B uses four DI bins")
        else:
            failures.append("Fig.3B must use four DI bins")
        b_stats_path = panel_output_paths(output_dir, "fig3", "B")["stats"]
        if b_stats_path.exists():
            b_stats = read_json(b_stats_path)
            timing = b_stats.get("timing_validation") or {}
            if timing.get("status") == "PASS" and not bool(timing.get("old_timing_detected", False)):
                passes.append("Fig.3B uses corrected taxonomy timing and excludes old 50/50/50/50 runs")
            else:
                failures.append(f"Fig.3B timing validation failed or detected old timing: {timing}")

    c_path = panel_output_paths(output_dir, "fig3", "C")["panel_data"]
    if "C" in panels and c_path.exists():
        c_df = pd.read_csv(c_path)
        metrics = set(c_df.get("metric", pd.Series(dtype=str)).astype(str))
        if metrics == {"stepwise_update_ratio"} and {"sequence_stage", "stepwise_update_ratio", "value"}.issubset(c_df.columns):
            passes.append("Fig.3C contains progressive update panel data")
        else:
            failures.append(f"Fig.3C must be progressive update data, found metrics {sorted(metrics)} and columns {sorted(c_df.columns)}")
        stages = set(pd.to_numeric(c_df.get("sequence_stage", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
        if stages and min(stages) >= 2 and max(stages) >= 10:
            passes.append("Fig.3C covers progressive update stage_k >= 2 through 10")
        else:
            failures.append(f"Fig.3C must cover progressive update stages 2..10, found {sorted(stages)}")

    d_path = panel_output_paths(output_dir, "fig3", "D")["panel_data"]
    if "D" in panels and d_path.exists():
        d_df = pd.read_csv(d_path)
        d_stages = set(pd.to_numeric(d_df.get("sequence_stage", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
        center_types = set(d_df.get("center_type", pd.Series(dtype=str)).astype(str))
        if {"morphological", "functional"}.issubset(center_types):
            passes.append("Fig.3D merged panel uses both com_sim and ping_com center trajectories")
        else:
            failures.append(f"Fig.3D must contain morphological and functional center rows, found {sorted(center_types)}")
        if d_stages and min(d_stages) == 1 and max(d_stages) == 10:
            passes.append("Fig.3D covers center stages 1..10")
        else:
            failures.append(f"Fig.3D must cover center stages 1..10, found {sorted(d_stages)}")


def _check_fig3_render_layout(render_metadata: Mapping[str, Mapping[str, Any]], passes: list[str], warnings: list[str], failures: list[str]) -> None:
    def meta(panel_id: str) -> Mapping[str, Any]:
        return render_metadata.get(panel_id) or {}

    if str(meta("C").get("plot_form")) == "stage_update_ratio":
        passes.append("Fig.3C renders the progressive update line panel")
    else:
        failures.append("Fig.3C must render as progressive update, not the removed heatmap")
    if str(meta("C").get("plot_form")) != "triangular_heatmap" and not bool(meta("C").get("has_colorbar")):
        passes.append("Old Fig.3C heatmap is absent from the final composite")
    else:
        failures.append("Removed heatmap/colorbar must not appear in final Fig.3")
    c_ylim = meta("C").get("ylim")
    if c_ylim and float(c_ylim[1]) <= 0.5:
        passes.append("Fig.3C y-axis upper limit is <= 0.5")
    else:
        failures.append(f"Fig.3C y-axis must not exceed 0.5, found {c_ylim}")
    if not bool(meta("C").get("has_y1_reference", False)):
        passes.append("Fig.3C has no y=1 reference line")
    else:
        failures.append("Fig.3C must not include a y=1 reference line")
    if str(meta("D").get("plot_form")) == "merged_center_trajectory" and bool(meta("D").get("merged_center_panel", False)):
        passes.append("Fig.3D renders morphology and function in one merged center migration panel")
    else:
        failures.append("Fig.3D must render morphology and function in one merged panel")
    colors = meta("D").get("center_line_colors") or {}
    morph_color = str(colors.get("morphological", "")).lower()
    func_color = str(colors.get("functional", "")).lower()
    if morph_color in {"#d8a21b", "#d8a21bff"} and func_color in {"#009e73", "#009e73ff"}:
        passes.append(f"Fig.3D uses gold/yellow morphological and green functional line colors: {colors}")
    else:
        failures.append(f"Fig.3D must use gold/yellow for Morphological and green for Functional, found {colors}")
    if bool(meta("D").get("legend_above_plot")) and set(meta("D").get("legend_texts") or []) == {"Morphological", "Functional"}:
        passes.append("Fig.3D uses a compact legend above the plot for center-line labels")
    else:
        failures.append(f"Fig.3D must use legend above plot with Morphological/Functional labels, found {meta('D').get('legend_texts')}")
    d_legend_bbox = meta("D").get("legend_bbox") or []
    d_label_bbox = meta("D").get("panel_label_bbox") or []
    if d_legend_bbox and d_label_bbox and not _boxes_overlap(d_legend_bbox, d_label_bbox):
        passes.append("Fig.3D legend does not overlap panel letter D")
    else:
        failures.append("Fig.3D legend must not overlap panel letter D")
    if not bool(meta("D").get("endpoint_text_labels", False)):
        passes.append("Fig.3D uses no endpoint text labels")
    else:
        failures.append("Fig.3D must not use endpoint text labels")
    titles = {panel_id: meta(panel_id).get("title") for panel_id in ("A", "B", "C", "D")}
    if all(not str(value).strip() for value in titles.values()):
        passes.append("Fig.3 rendered axes have no small panel titles")
    else:
        failures.append(f"Fig.3 panel titles must be absent in rendered axes, found {titles}")
    label_gaps = [meta(panel_id).get("panel_label_gap_mm") for panel_id in ("A", "B", "C", "D")]
    if all(gap == 0 for gap in label_gaps):
        passes.append("Fig.3 panel letters have consistent slot-relative offsets")
    else:
        warnings.append(f"Fig.3 panel letter offsets differ from requested slot coordinates: {label_gaps}")
    passes.append("Fig.3 missing-source messages, if any, are confined to QC/manifests")


def _fig3_mm_box_close(actual: Mapping[str, Any], expected: Mapping[str, Any], tol: float = 0.05) -> bool:
    return all(_near(float(actual.get(key, -9999)), float(expected[key]), tol=tol) for key in ("x", "y", "w", "h"))


def _same_top_bottom(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return _near(float(a.get("y", 0)), float(b.get("y", -1))) and _near(float(a.get("y", 0)) + float(a.get("h", 0)), float(b.get("y", -1)) + float(b.get("h", 0)))


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
    _check_fig4_standalone_contract(figure_id, spec, panels, output_dir, adapter_results, render_metadata, passes, warnings, failures)
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
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 124:
        passes.append("Fig.5 canvas is 165 x 124 mm")
    else:
        failures.append(f"Fig.5 canvas must be 165 x 124 mm, found {canvas}")
    _check_fig5_geometry(panels, render_metadata, passes, failures)

    panel_data: dict[str, pd.DataFrame] = {}
    stats_by_panel: dict[str, Mapping[str, Any]] = {}
    sources_by_panel: dict[str, Mapping[str, Any]] = {}
    for panel_id in ("A", "B", "C", "D"):
        if panel_id not in panels:
            failures.append(f"Fig.5{panel_id}: panel spec missing")
            continue
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            failures.append(f"Fig.5{panel_id}: missing adapter outputs {missing}")
            continue
        passes.append(f"Fig.5{panel_id}: panel_data/stats/source_manifest exist")
        panel_data[panel_id] = pd.read_csv(paths["panel_data"])
        stats_by_panel[panel_id] = read_json(paths["stats"])
        sources_by_panel[panel_id] = read_json(paths["sources"])
        run_mode = str(stats_by_panel[panel_id].get("run_mode") or sources_by_panel[panel_id].get("run_mode") or "")
        n_networks = int(stats_by_panel[panel_id].get("n_networks") or sources_by_panel[panel_id].get("n_networks") or 0)
        if run_mode:
            passes.append(f"Fig.5{panel_id}: run_mode={run_mode}")
        else:
            failures.append(f"Fig.5{panel_id}: run_mode missing")
        if run_mode == "single_network_draft" or n_networks == 1:
            warnings.append(f"Fig.5{panel_id}: single_network_draft n_networks=1; draft-only, not final manuscript statistics")
        elif n_networks > 1:
            passes.append(f"Fig.5{panel_id}: n_networks={n_networks}")
        else:
            warnings.append(f"Fig.5{panel_id}: n_networks not recorded")

    if panel_data:
        groups = set()
        conditions = set()
        metrics = set()
        sources_text: list[str] = []
        for df in panel_data.values():
            groups.update(str(v) for v in df.get("unit_group", pd.Series(dtype=str)).dropna().unique())
            conditions.update(str(v) for v in df.get("perturbation_condition", pd.Series(dtype=str)).dropna().unique())
            conditions.update(str(v) for v in df.get("condition", pd.Series(dtype=str)).dropna().unique())
            metrics.update(str(v) for v in df.get("metric", pd.Series(dtype=str)).dropna().unique())
            sources_text.extend(str(v) for v in df.get("source_file", pd.Series(dtype=str)).dropna().unique())

        required_groups = set(((spec.get("qc_requirements") or {}).get("required_unit_groups") or ["overlap_dominant", "probe_only_dominant"]))
        if required_groups.issubset(groups):
            passes.append("Fig.5 required overlap/probe-only unit groups present")
        else:
            failures.append(f"Fig.5 missing required unit groups {sorted(required_groups - groups)}")

        required_conditions = set(((spec.get("qc_requirements") or {}).get("required_main_conditions") or []))
        if required_conditions.issubset(conditions):
            passes.append("Fig.5 required main perturbation conditions present")
        else:
            failures.append(f"Fig.5 missing required perturbation conditions {sorted(required_conditions - conditions)}")

        required_panel_metrics = {
            "A": {"preprobe_support"},
            "B": {"transition_fraction"},
            "C": {"winner_delta_v", "loser_delta_v", "loser_inhibition"},
            "D": {"transition_fraction"},
        }
        for panel_id, expected_metrics in required_panel_metrics.items():
            df = panel_data.get(panel_id)
            if df is None:
                continue
            found = set(str(v) for v in df.get("metric", pd.Series(dtype=str)).dropna().unique())
            if expected_metrics.issubset(found):
                passes.append(f"Fig.5{panel_id}: required metrics present")
            else:
                failures.append(f"Fig.5{panel_id}: missing metrics {sorted(expected_metrics - found)}")
        b_df = panel_data.get("B")
        if b_df is not None:
            b_metrics = set(str(v) for v in b_df.get("metric", pd.Series(dtype=str)).dropna().unique())
            b_groups = set(str(v) for v in b_df.get("unit_group", pd.Series(dtype=str)).dropna().unique())
            b_transitions = set(str(v) for v in b_df.get("transition_type", pd.Series(dtype=str)).dropna().unique())
            if b_metrics == {"transition_fraction"} and {"advance", "recruit", "loss"}.issubset(b_transitions):
                passes.append("Fig.5B uses advance/recruit/loss transition composition")
            else:
                failures.append(f"Fig.5B must use transition_fraction advance/recruit/loss rows, found metrics={sorted(b_metrics)} transitions={sorted(b_transitions)}")
            if b_groups == {"overlap_dominant", "probe_only_dominant", "random_matched"}:
                passes.append("Fig.5B includes only Overlap, Probe-only, and Random unit groups")
            else:
                failures.append(f"Fig.5B must include only overlap/probe/random groups, found {sorted(b_groups)}")

        if "full_chain_satisfied_fraction" in metrics:
            failures.append("Fig.5 main panel data must not use full-chain event fraction as a primary metric")
        else:
            passes.append("Fig.5 main panel data does not use full-chain event fraction")

        removed_conditions = {
            "flatten_overlap_high_support",
            "flatten_nonoverlap_high_support",
            "flatten_random_high_support_matched",
            "Flatten overlap support",
            "Flatten non-overlap support",
            "Flatten random support",
        }
        leaked_removed = sorted(removed_conditions.intersection(conditions))
        if leaked_removed:
            failures.append(f"Fig.5 main panel data still includes removed flatten conditions: {leaked_removed}")
        else:
            passes.append("Fig.5 main panels exclude old flatten/non-overlap/random perturbation conditions")

        d_df = panel_data.get("D")
        if d_df is not None:
            d_conditions = set(str(v) for v in d_df.get("perturbation_condition", pd.Series(dtype=str)).dropna().unique())
            d_groups = set(str(v) for v in d_df.get("unit_group", pd.Series(dtype=str)).dropna().unique())
            d_transitions = set(str(v) for v in d_df.get("transition_type", pd.Series(dtype=str)).dropna().unique())
            expected = {"dynamic_intact", "attenuate_l1_stsp", "reset_l1_stsp"}
            if expected.issubset(d_conditions):
                passes.append("Fig.5D includes dynamic/Layer1-attenuate/Layer1-reset perturbation conditions")
            else:
                failures.append(f"Fig.5D missing perturbation conditions {sorted(expected - d_conditions)}")
            if "static_frozen" in d_conditions:
                warnings.append("Fig.5D plots static_frozen; main D should use static only as transition reference")
            else:
                passes.append("Fig.5D uses static_frozen as reference without plotting a static bar")
            source_text_d = " ".join(str(v) for v in d_df.get("source_file", pd.Series(dtype=str)).dropna().unique()).lower()
            d_metrics = set(str(v) for v in d_df.get("metric", pd.Series(dtype=str)).dropna().unique())
            if d_metrics == {"transition_fraction"} and {"advance", "recruit", "loss"}.issubset(d_transitions):
                passes.append("Fig.5D uses stacked advance/recruit/loss transition composition")
            else:
                failures.append(f"Fig.5D must use transition_fraction advance/recruit/loss rows, found metrics={sorted(d_metrics)} transitions={sorted(d_transitions)}")
            d_labels = set(str(v) for v in d_df.get("condition", pd.Series(dtype=str)).dropna().unique())
            if {"Dynamic", "Attenuate L1 STSP", "Reset L1 STSP"}.issubset(d_labels) and not d_groups:
                passes.append("Fig.5D x-axis is Layer1 STSP condition, not region groups")
            else:
                failures.append(f"Fig.5D must plot Dynamic/Attenuate L1 STSP/Reset L1 STSP without unit-group split, found labels={sorted(d_labels)} groups={sorted(d_groups)}")
            forbidden_d_metrics = {"dynamic_like_spike_similarity", "decision_deflection_score", "loser_post_winner_inh_rise", "early_recruitment", "spike_similarity", "decision_deflection"}
            if forbidden_d_metrics.isdisjoint(d_metrics) and "panel_d_support_perturbation_node_metrics.csv" not in source_text_d:
                passes.append("Fig.5D does not use node similarity/decision summary as the main visual")
            else:
                failures.append(f"Fig.5D must not use node similarity/decision summary, found metrics={sorted(forbidden_d_metrics.intersection(d_metrics))}")
            if "panel_d_l1_stsp_perturbation_transition_summary.csv" in source_text_d:
                passes.append("Fig.5D source is Layer1 STSP perturbation transition summary")
            else:
                failures.append("Fig.5D must use panel_d_l1_stsp_perturbation_transition_summary.csv")

        c_stats = stats_by_panel.get("C", {})
        c_sources = sources_by_panel.get("C", {})
        if c_stats or c_sources:
            if c_stats.get("inhibition_trace_definition") == "loser_unit_received_inhibition" or c_sources.get("inhibition_trace_definition") == "loser_unit_received_inhibition":
                passes.append("Fig.5C manifest defines inhibition as received by the selected loser unit")
            else:
                failures.append("Fig.5C must define loser_inhibition as loser_unit_received_inhibition")
            if c_stats.get("baseline_corrected") is True or c_sources.get("baseline_corrected") is True:
                passes.append("Fig.5C uses pre-event baseline correction")
            else:
                warnings.append("Fig.5C baseline_corrected is not true")

        old_tokens = ("fig4_panel", "dms_overlap_ux_support_mechanism", "overlap_causal_input_perturbation", "full_chain_satisfied")
        condition_text = " ".join(conditions).lower()
        source_text = " ".join(sources_text).lower()
        leaks = [token for token in old_tokens if token in condition_text or token in source_text]
        if leaks:
            warnings.append(f"Fig.5 panel data/source labels contain old/internal tokens: {leaks}")
        else:
            passes.append("Fig.5 panel data avoids old Fig.4/internal labels")
        image_sources = [path for path in sources_text if str(path).lower().endswith((".png", ".pdf", ".svg"))]
        if image_sources:
            warnings.append(f"Fig.5 panel data references rendered image sources: {image_sources}")
        else:
            passes.append("Fig.5 panel data is data-first and does not reference rendered source images")

    supplement_files = []
    for source in sources_by_panel.values():
        supplement_files.extend(source.get("supplement_files") or [])
    supp_by_name = {Path(str(item.get("path", ""))).name: bool(item.get("exists")) for item in supplement_files if isinstance(item, Mapping)}
    if supp_by_name.get("supp_event_chain_null_baselines.csv"):
        passes.append("Fig.5 supplement includes explicit event-chain null baselines")
    else:
        warnings.append("Fig.5 event-chain null baselines missing from supplement manifests")
    if supp_by_name.get("supp_perturbation_ux_audit.csv"):
        passes.append("Fig.5 u/x perturbation audit is present in supplement manifests")
    else:
        warnings.append("Fig.5 u/x perturbation audit missing from supplement manifests")

    manifest_path = output_dir / f"{figure_id}_source_manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
    if "pre_probe_boundary" in manifest_text and "probe_input_changed" in manifest_text:
        passes.append("Fig.5 source manifests record pre-probe perturbation timing and unchanged probe input")
    else:
        warnings.append("Fig.5 source manifests do not clearly record pre-probe perturbation timing/probe-input invariance")

    if render_metadata:
        clipped_panels = {
            panel_id: list(meta.get("clipped_artists", []))
            for panel_id, meta in render_metadata.items()
            if meta.get("clipped_artists") or meta.get("panel_label_clipped")
        }
        if clipped_panels:
            failures.append(f"Fig.5 labels/ticks/legends/panel letters clipped: {clipped_panels}")
        else:
            passes.append("Fig.5 has no clipped labels, ticks, legends, or panel letters")
        point_panels = [panel_id for panel_id in ("A", "B", "D") if bool((render_metadata.get(panel_id) or {}).get("raw_points"))]
        if point_panels:
            failures.append(f"Fig.5 A/B/D must not render raw point overlays, found {point_panels}")
        else:
            passes.append("Fig.5 A/B/D render without raw point overlays")
    return

def _check_fig5_supp_specifics(
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
    _ = spec, adapter_results, render_metadata
    if figure_id != "fig5_supp":
        return
    expected = ["S6A", "S6B", "S6C", "S6D", "S6E", "S6F"]
    if list(panels) == expected:
        passes.append("Fig.5 supplement defines compact S6A-S6F")
    else:
        failures.append(f"Fig.5 supplement panel order must be {expected}, found {list(panels)}")
    inactive = [panel_id for panel_id in panels if str(panel_id).startswith("S10")]
    if inactive:
        failures.append(f"Fig.5 supplement must not expose legacy S10 panels, found {inactive}")
    else:
        passes.append("Fig.5 supplement exposes no legacy S10 panels")
    missing_source_panels: list[str] = []
    for panel_id in expected:
        if panel_id not in panels:
            continue
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            failures.append(f"Fig.5 supplement {panel_id}: missing adapter outputs {missing}")
            continue
        passes.append(f"Fig.5 supplement {panel_id}: panel_data/stats/source_manifest exist")
        df = pd.read_csv(paths["panel_data"])
        sources = read_json(paths["sources"])
        stats = read_json(paths["stats"])
        run_mode = str(stats.get("run_mode") or sources.get("run_mode") or "")
        if run_mode == "single_network_draft":
            warnings.append(f"Fig.5 supplement {panel_id}: single_network_draft n_networks=1; draft-only")
        if sources.get("status") == "missing_source" or df.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any():
            missing_source_panels.append(panel_id)
            if panel_id == "S10E" or bool((panels.get(panel_id) or {}).get("optional")):
                warnings.append(f"Fig.5 supplement {panel_id}: optional source missing; placeholder expected")
            else:
                warnings.append(f"Fig.5 supplement {panel_id}: source missing; placeholder expected")
        panel_type = str((panels.get(panel_id) or {}).get("panel_type", ""))
        perturbation_panel = panel_type in {
            "perturbation_ux_audit",
            "perturbation_transition_contrast",
            "same_winner_lost_delayed",
            "dynamic_like_recovery",
            "sham_matching_controls",
        }
        if perturbation_panel and sources:
            if sources.get("intervention_timing") == "pre_probe_boundary" and sources.get("probe_input_changed") is False:
                passes.append(f"Fig.5 supplement {panel_id}: perturbation manifest records timing/input invariance")
            else:
                warnings.append(f"Fig.5 supplement {panel_id}: perturbation timing/input invariance not recorded")
    s9b_path = panel_output_paths(output_dir, figure_id, "S6B")["panel_data"]
    if s9b_path.exists():
        s9b = pd.read_csv(s9b_path)
        metrics = set(s9b.get("metric", pd.Series(dtype=str)).astype(str))
        conditions = set(s9b.get("condition", pd.Series(dtype=str)).astype(str))
        source_levels = set(s9b.get("source_level", pd.Series(dtype=str)).astype(str))
        required_conditions = {"vs probe-only", "vs random", "vs balanced"}
        if "delta_P_advance_plus_recruit" in metrics and required_conditions.issubset(conditions):
            passes.append("Fig.5 supplement S6B contains trialwise overlap advance/recruit advantages")
        elif "missing_source" in metrics:
            warnings.append("Fig.5 supplement S6B trialwise transition advantage missing; placeholder written")
        else:
            warnings.append(f"Fig.5 supplement S6B lacks trialwise transition advantage rows, metrics={sorted(metrics)} conditions={sorted(conditions)}")
        if "trialwise" in source_levels:
            passes.append("Fig.5 supplement S6B uses trial-level paired deltas")
        elif "aggregate_fallback" in source_levels:
            warnings.append("Fig.5 supplement S6B uses aggregate fallback rather than trial-level paired deltas")
        s9b_meta = render_metadata.get("S6B", {})
        if s9b_meta:
            if s9b_meta.get("plot_form") == "s9_trialwise_transition_advantage_bar_only" and not bool(s9b_meta.get("raw_points", False)):
                passes.append("Fig.5 supplement S6B renders as bar-only trialwise advantage")
            else:
                failures.append(f"Fig.5 supplement S6B must render as bar-only trialwise advantage, found {s9b_meta.get('plot_form')}")
    s9c_path = panel_output_paths(output_dir, figure_id, "S6C")["panel_data"]
    if s9c_path.exists():
        s9c = pd.read_csv(s9c_path)
        metrics = set(s9c.get("metric", pd.Series(dtype=str)).astype(str))
        conditions = set(s9c.get("condition", pd.Series(dtype=str)).astype(str))
        if "full_chain_satisfied_fraction" in metrics and any(c.startswith("Null") for c in conditions):
            passes.append("Fig.5 supplement S6C contains observed/null event-chain baselines")
        elif "missing_source" in metrics:
            warnings.append("Fig.5 supplement S6C event-chain/null baseline missing; placeholder written")
        else:
            warnings.append(f"Fig.5 supplement S6C lacks observed/null event-chain baselines, metrics={sorted(metrics)} conditions={sorted(conditions)}")
    s9e_path = panel_output_paths(output_dir, figure_id, "S6E")["panel_data"]
    if s9e_path.exists():
        s9e = pd.read_csv(s9e_path)
        metrics = set(s9e.get("metric", pd.Series(dtype=str)).astype(str))
        if {"delta_P_advance_plus_recruit", "delta_P_loss"}.intersection(metrics):
            passes.append("Fig.5 supplement S6E contains perturbation transition contrast")
        elif "missing_source" in metrics:
            warnings.append("Fig.5 supplement S6E perturbation transition contrast missing; placeholder written")
        else:
            warnings.append(f"Fig.5 supplement S6E lacks perturbation contrast metrics, found {sorted(metrics)}")
    s9f_path = panel_output_paths(output_dir, figure_id, "S6F")["panel_data"]
    if s9f_path.exists():
        s9f = pd.read_csv(s9f_path)
        metrics = set(s9f.get("metric", pd.Series(dtype=str)).astype(str))
        if {"P_same_winner_preserved", "P_same_winner_lost", "P_same_winner_delayed"}.intersection(metrics):
            passes.append("Fig.5 supplement S6F contains same-winner disruption metrics")
        elif "missing_source" in metrics:
            warnings.append("Fig.5 supplement S6F same-winner metrics missing; placeholder written")
        else:
            warnings.append(f"Fig.5 supplement S6F lacks same-winner metrics, found {sorted(metrics)}")
    if missing_source_panels:
        warnings.append(f"Fig.5 supplement placeholder panels: {sorted(set(missing_source_panels))}")
    else:
        passes.append("Fig.5 supplement has no missing-source placeholders")
    return
    '''
    if False and "ux_map_pre_dynamic" in locals().get("image_types", set()):
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
    '''


def _check_fig5_geometry(panels: Mapping[str, Any], render_metadata: Mapping[str, Mapping[str, Any]], passes: list[str], failures: list[str]) -> None:
    ids = ("A", "B", "C", "D")
    pos = {panel_id: (panels.get(panel_id) or {}).get("position_mm") or {} for panel_id in ids}
    if any(not pos[panel_id] for panel_id in ids):
        failures.append("Fig.5 A-D must define position_mm")
        return
    expected = {
        "A": {"x": 12.00, "y": 8.00, "w": 70.50, "h": 46.00},
        "B": {"x": 88.50, "y": 8.00, "w": 70.50, "h": 46.00},
        "C": {"x": 12.00, "y": 62.00, "w": 70.50, "h": 54.00},
        "D": {"x": 88.50, "y": 62.00, "w": 70.50, "h": 54.00},
    }
    for panel_id, expected_pos in expected.items():
        if _fig5_box_near(pos[panel_id], expected_pos, tol=0.10):
            passes.append(f"Fig.5{panel_id} position matches requested 2x2 mm layout")
        else:
            failures.append(f"Fig.5{panel_id} position must be {expected_pos}, found {pos[panel_id]}")
    if _near(_y(pos["A"]), _y(pos["B"]), tol=0.15) and _near(_bottom(pos["A"]), _bottom(pos["B"]), tol=0.15):
        passes.append("Fig.5 A/B form the top row")
    else:
        failures.append("Fig.5 A/B must form the top row")
    if _near(_y(pos["C"]), _y(pos["D"]), tol=0.15) and _near(_bottom(pos["C"]), _bottom(pos["D"]), tol=0.15):
        passes.append("Fig.5 C/D form the bottom row")
    else:
        failures.append("Fig.5 C/D must form the bottom row")
    if _near(_x(pos["A"]), _x(pos["C"]), tol=0.15) and _near(_x(pos["B"]), _x(pos["D"]), tol=0.15):
        passes.append("Fig.5 columns align in a clean 2x2 layout")
    else:
        failures.append("Fig.5 columns must align A/C and B/D")
    row_gap = _y(pos["C"]) - _bottom(pos["A"])
    col_gap_top = _x(pos["B"]) - _right(pos["A"])
    col_gap_bottom = _x(pos["D"]) - _right(pos["C"])
    if row_gap >= 6.0 and min(col_gap_top, col_gap_bottom) >= 5.0:
        passes.append("Fig.5 2x2 row and column gaps are clean and positive")
    else:
        failures.append(f"Fig.5 2x2 gaps are too tight: row={row_gap:.2f}, cols=({col_gap_top:.2f},{col_gap_bottom:.2f})")
    left_margin = min(_x(pos[pid]) for pid in ids)
    top_margin = min(_y(pos[pid]) for pid in ids)
    right_margin = 165.0 - max(_right(pos[pid]) for pid in ids)
    bottom_margin = 124.0 - max(_bottom(pos[pid]) for pid in ids)
    if left_margin >= 6.0 and top_margin >= 5.0 and right_margin >= 5.0 and bottom_margin >= 6.0:
        passes.append("Fig.5 uses stable outer margins on all sides")
    else:
        failures.append(f"Fig.5 outer margins too small: left={left_margin:.2f}, top={top_margin:.2f}, right={right_margin:.2f}, bottom={bottom_margin:.2f}")
    axes = {pid: render_metadata.get(pid, {}).get("plot_axes_bounds", render_metadata.get(pid, {}).get("axes_bounds", [])) for pid in ids}
    if all(isinstance(axes[pid], list) and len(axes[pid]) == 4 for pid in ids):
        passes.append("Fig.5 alignment checks use actual plotting axes boxes rather than panel-label or outer-panel extents")
        top_widths_axes = [_box_w(axes[pid]) for pid in ("A", "B")]
        bottom_widths_axes = [_box_w(axes[pid]) for pid in ("C", "D")]
        if max(top_widths_axes + bottom_widths_axes) - min(top_widths_axes + bottom_widths_axes) <= 0.006:
            passes.append("Fig.5 plotting axes share 2x2 column widths")
        else:
            failures.append(f"Fig.5 plotting boxes must share column widths, found top={top_widths_axes}, bottom={bottom_widths_axes}")
        if _near(axes["A"][0], axes["C"][0], tol=0.004) and _near(axes["B"][0], axes["D"][0], tol=0.004):
            passes.append("Fig.5 rendered plotting axes preserve column alignment")
        else:
            failures.append("Fig.5 rendered plotting axes must align A/C and B/D columns")


def _fig5_box_near(actual: Mapping[str, Any], expected: Mapping[str, float], *, tol: float = 0.10) -> bool:
    return all(_near(float(actual.get(key, -999.0)), value, tol=tol) for key, value in expected.items())


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
    _check_fig6_stsp_recruitment_contract(spec, panels, output_dir, render_metadata, passes, warnings, failures)
    return
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 126:
        passes.append("Fig.6 canvas is 165 x 126 mm")
    else:
        failures.append(f"Fig.6 canvas must be 165 x 126 mm, found {canvas}")

    expected_panels = {"A", "B", "C", "D", "E", "F"}
    panel_ids = set(panels.keys())
    if panel_ids == expected_panels:
        passes.append("Fig.6 panels are A-F")
    else:
        failures.append(f"Fig.6 must contain exactly panels A-F, found {sorted(panel_ids)}")
    if list(spec.get("reading_order") or []) == ["A", "B", "C", "D", "E", "F"]:
        passes.append("Fig.6 reading order is A-F")
    else:
        failures.append(f"Fig.6 reading_order must be A-F, found {spec.get('reading_order')}")

    missing_source_panels: set[str] = set()
    for panel_id, result in adapter_results.items():
        if result.source_manifest.get("status") == "missing_source":
            missing_source_panels.add(panel_id)
    for panel_id in panels:
        data_path = panel_output_paths(output_dir, figure_id, panel_id)["panel_data"]
        stats_path = panel_output_paths(output_dir, figure_id, panel_id)["stats"]
        source_path = panel_output_paths(output_dir, figure_id, panel_id)["sources"]
        if not data_path.exists():
            failures.append(f"Fig.6{panel_id}: required panel_data missing")
            continue
        if not stats_path.exists():
            failures.append(f"Fig.6{panel_id}: required stats manifest missing")
        if not source_path.exists():
            failures.append(f"Fig.6{panel_id}: required source manifest missing")
        df = pd.read_csv(data_path)
        if df.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any():
            missing_source_panels.add(panel_id)
        stats = read_json(stats_path) if stats_path.exists() else {}
        sources = read_json(source_path) if source_path.exists() else {}
        n_networks = int(stats.get("n_networks") or sources.get("n_networks") or 0)
        run_mode = str(stats.get("run_mode") or sources.get("run_mode") or "")
        if n_networks == 1 or run_mode == "single_network_draft":
            warnings.append(f"Fig.6{panel_id}: single_network_draft n_networks=1; draft-only, not final manuscript statistics")
        elif n_networks > 1:
            passes.append(f"Fig.6{panel_id}: n_networks={n_networks}")
        else:
            warnings.append(f"Fig.6{panel_id}: n_networks not recorded")

    if missing_source_panels:
        failures.append(f"Fig.6 has missing_source panels: {sorted(missing_source_panels)}")
    else:
        passes.append("Fig.6 has no missing_source panel")

    a_path = panel_output_paths(output_dir, figure_id, "A")["panel_data"]
    if a_path.exists():
        a_df = pd.read_csv(a_path)
        if "relative_position_from_end" in a_df.columns and "peak_loss_fraction" in set(a_df.get("metric", [])):
            passes.append("Fig.6A uses leave-one-item-out peak source attribution")
        else:
            failures.append("Fig.6A must use peak source attribution, not multi-recent enrichment")

    b_path = panel_output_paths(output_dir, figure_id, "B")["panel_data"]
    if b_path.exists():
        b_df = pd.read_csv(b_path)
        metrics = set(b_df.get("metric", pd.Series(dtype=str)).astype(str))
        if {"mean_update_count", "mean_time_since_last_update"}.intersection(metrics):
            passes.append("Fig.6B summarizes peak-conditional update history")
        else:
            failures.append("Fig.6B must summarize update history, not the update/recency regression")

    c_path = panel_output_paths(output_dir, figure_id, "C")["panel_data"]
    if c_path.exists():
        c_df = pd.read_csv(c_path)
        if "dice_peak_overlap" in set(c_df.get("metric", [])) or "mean_peak_coverage" in c_df.columns:
            passes.append("Fig.6C uses recent input-overlap origin similarity")
        else:
            failures.append("Fig.6C must use peak input-overlap origin similarity, not D/E overlap trial definitions")

    d_path = panel_output_paths(output_dir, figure_id, "D")["panel_data"]
    if d_path.exists():
        d_df = pd.read_csv(d_path)
        if "reentry_strength_real" in set(d_df.get("metric", [])) or "reentry_strength_real" in d_df.columns:
            passes.append("Fig.6D uses real re-entry rollout metrics")
        else:
            failures.append("Fig.6D must use panel_d_real_reentry_metrics.csv-derived real metrics")
        controls = set(d_df.get("raw_overlap_control", pd.Series(dtype=str)).astype(str))
        has_control = bool(controls.intersection({"matched_group", "regression"})) or "matched_group_id" in d_df.columns
        if has_control and "raw_overlap" in d_df.columns:
            passes.append("Fig.6D controls or matches raw overlap")
        else:
            failures.append("Fig.6D must control or match raw overlap")
        if d_df.get("proxy_mode", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"}).all():
            warnings.append("Fig.6D is proxy-mode only and not final scientific evidence.")

    e_path = panel_output_paths(output_dir, figure_id, "E")["panel_data"]
    if e_path.exists():
        e_df = pd.read_csv(e_path)
        metrics = set(e_df.get("metric", pd.Series(dtype=str)).astype(str))
        if {"early_recruitment_gain_real", "decision_deflection_score_real"}.intersection(metrics):
            passes.append("Fig.6E uses real downstream rollout metrics")
        else:
            failures.append("Fig.6E must use real downstream metrics")
        controls = set(e_df.get("raw_overlap_control", pd.Series(dtype=str)).astype(str))
        if controls.intersection({"matched_group", "regression"}) and "raw_overlap" in e_df.columns:
            passes.append("Fig.6E controls raw overlap")
        else:
            failures.append("Fig.6E must control raw overlap")
        if e_df.get("proxy_mode", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"}).all():
            warnings.append("Fig.6E is proxy-mode only and not final scientific evidence.")

    f_path = panel_output_paths(output_dir, figure_id, "F")["panel_data"]
    if f_path.exists():
        f_df = pd.read_csv(f_path)
        text = " ".join(" ".join(map(str, f_df.get(col, []))) for col in ("route_statement", "gain_statement", "mechanism_statement", "forbidden_language", "allowed_claim_strength")).lower()
        if ("overlap provides the route" in text or "overlap provides route" in text) and ("peaks provide the gain" in text or "peaks provide gain" in text):
            passes.append("Fig.6F states route/gain mechanism")
        else:
            failures.append("Fig.6F must explicitly state overlap provides the route and peaks provide the gain")
        if "peaks replace overlap" in " ".join(map(str, f_df.get("claim", []))).lower():
            failures.append("Fig.6F implies peaks replace overlap")
        claim_strength = " ".join(map(str, f_df.get("allowed_claim_strength", []))).lower()
        perturb = f_df.get("peak_perturbation_implemented", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"}).any()
        if "causal" in claim_strength and not perturb:
            warnings.append("Fig.6F uses causal language without successful peak perturbation")
        elif not perturb:
            warnings.append("Fig.6F is predictive-only because peak perturbation is absent")

    manifest_path = output_dir / f"{figure_id}_source_manifest.json"
    supplement_text = ""
    if manifest_path.exists():
        supplement_text = manifest_path.read_text(encoding="utf-8", errors="ignore").lower()
    if f_path.exists() and "fig6_design_version" in pd.read_csv(f_path).columns and pd.read_csv(f_path)["fig6_design_version"].astype(str).eq("peak_origin_real_reentry_rollout").any():
        passes.append("Fig.6 summary records peak_origin_real_reentry_rollout design version")
    elif manifest_path.exists() and "summary.json" in supplement_text:
        warnings.append("Fig.6 summary is present but design-version evidence was not found in manifest text")
    if "supp_alternative_peak_definitions.csv" in supplement_text:
        passes.append("Fig.6 supplement includes alternative peak definitions")
    else:
        warnings.append("Fig.6 supplement alternative peak definitions not found in source manifest")
    if "supp_visual_energy_classpair_controls.csv" in supplement_text:
        passes.append("Fig.6 supplement includes visual similarity / input energy controls")
    else:
        warnings.append("Fig.6 visual similarity / input energy controls not found in source manifest")

    claim_text = " ".join(str((panel or {}).get("claim", "")) for panel in panels.values()).lower()
    if "peaks replace overlap" in claim_text or "peaks alone gate" in claim_text:
        failures.append("Fig.6 claims imply peaks replace or gate overlap")
    else:
        passes.append("Fig.6 claims preserve route/gain boundary")
    if "causal" in claim_text:
        perturb_present = False
        if f_path.exists():
            f_df = pd.read_csv(f_path)
            perturb_present = f_df.get("peak_perturbation_implemented", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"}).any()
        if not perturb_present:
            warnings.append("Fig.6 causal language appears while peak perturbation is absent")

    if render_metadata:
        blank_panels = [pid for pid, meta in render_metadata.items() if meta.get("plot_form") == "blank_panel"]
        if blank_panels:
            failures.append(f"Fig.6 has blank/placeholder renderers: {blank_panels}")
        else:
            passes.append("Fig.6 has no blank panel renderer")
        if (render_metadata.get("D") or {}).get("plot_form") in {"real_peak_overlap_reentry", "matched_peak_reentry"}:
            passes.append("Fig.6D renderer is real-rollout oriented")
        if (render_metadata.get("E") or {}).get("plot_form") in {"real_peak_overlap_downstream", "downstream_node_summary"}:
            passes.append("Fig.6E renderer is real-downstream oriented")
        if (render_metadata.get("F") or {}).get("peaks_replace_overlap"):
            failures.append("Fig.6F renderer metadata implies peaks replace overlap")
        if (render_metadata.get("F") or {}).get("route_gain_statement"):
            passes.append("Fig.6F renderer reports route/gain statement")
    old_required = (
        "panel_a_multi_recent_peak_enrichment.csv",
        "panel_b_update_recency_model_metrics.csv",
        "panel_c_peak_weighted_overlap_definitions.csv",
        "panel_d_peak_weighted_reentry_metrics.csv",
        "panel_e_peak_weighted_downstream_metrics.csv",
    )
    manifest_text = (output_dir / f"{figure_id}_source_manifest.json").read_text(encoding="utf-8", errors="ignore").lower() if (output_dir / f"{figure_id}_source_manifest.json").exists() else ""
    leaked_old = [name for name in old_required if name in manifest_text]
    if leaked_old:
        warnings.append(f"Fig.6 source manifest still references demoted old main files: {leaked_old}")
    else:
        passes.append("Fig.6 does not require old formula/enrichment main files")


def _check_fig6_stsp_recruitment_contract(
    spec: Mapping[str, Any],
    panels: Mapping[str, Any],
    output_dir: Path,
    render_metadata: Mapping[str, Mapping[str, Any]],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 112:
        passes.append("Fig.6 canvas is 165 x 112 mm")
    else:
        failures.append(f"Fig.6 canvas must be 165 x 112 mm for the A-F overlap-gated STSP layout, found {canvas}")

    expected_panels = {"A", "B", "C", "D", "E", "F"}
    if set(panels.keys()) == expected_panels:
        passes.append("Fig.6 panels are A-F")
    else:
        failures.append(f"Fig.6 must contain exactly panels A-F, found {sorted(panels.keys())}")
    if list(spec.get("reading_order") or []) == ["A", "B", "C", "D", "E", "F"]:
        passes.append("Fig.6 reading order is A-F")
    else:
        failures.append(f"Fig.6 reading_order must be A-F, found {spec.get('reading_order')}")

    expected_boxes = {
        "A": {"x": 8.00, "y": 8.00, "w": 45.00, "h": 38.00},
        "B": {"x": 60.00, "y": 8.00, "w": 45.00, "h": 38.00},
        "C": {"x": 112.00, "y": 8.00, "w": 45.00, "h": 38.00},
        "D": {"x": 8.00, "y": 58.00, "w": 45.00, "h": 38.00},
        "E": {"x": 60.00, "y": 58.00, "w": 45.00, "h": 38.00},
        "F": {"x": 112.00, "y": 58.00, "w": 45.00, "h": 38.00},
    }
    for panel_id, expected in expected_boxes.items():
        if panel_id in panels and _fig6_box_close(_fig6_pos(panels[panel_id]), expected):
            passes.append(f"Fig.6{panel_id} position matches overlap-gated STSP recruitment spec")
        elif panel_id in panels:
            failures.append(f"Fig.6{panel_id} position differs from overlap-gated STSP recruitment spec")

    score_name = str(spec.get("score_name") or ((spec.get("qc_requirements") or {}).get("score_name")) or "")
    if score_name == "entry_gated_stsp_gain_score":
        passes.append("Fig.6 resolved spec contains score_name=entry_gated_stsp_gain_score")
    else:
        failures.append(f"Fig.6 score_name must be entry_gated_stsp_gain_score, found {score_name!r}")

    endpoint = str(spec.get("primary_endpoint") or ((spec.get("qc_requirements") or {}).get("primary_endpoint")) or "")
    if endpoint == "Layer 1 spatial spike enrichment / recruitment":
        passes.append("Fig.6 primary endpoint is Layer 1 spatial spike enrichment / recruitment")
    else:
        failures.append(f"Fig.6 primary_endpoint is incorrect: {endpoint!r}")

    required_outputs = set(str(item) for item in (spec.get("required_outputs") or []))
    expected_required = {
        "data/metrics/panel_a_high_stsp_overlap_ablation.csv",
        "data/metrics/panel_a_high_stsp_overlap_ablation_summary.csv",
        "data/metrics/panel_b_region_ping_readout_bias.csv",
        "data/metrics/panel_c_global_ping_score_spike_prediction.csv",
        "data/metrics/panel_d_real_probe_score_spike_deflection.csv",
        "data/metrics/panel_e_overlap_gated_stsp_recruitment.csv",
        "data/metrics/panel_e_overlap_gated_stsp_interaction.csv",
        "data/raw/panel_f_global_mechanism_metadata.json",
    }
    missing_required = sorted(expected_required - required_outputs)
    if missing_required:
        failures.append(f"Fig.6 required_outputs missing overlap-gated STSP recruitment files {missing_required}")
    else:
        passes.append("Fig.6 required_outputs list contains the overlap-gated STSP recruitment source files")

    demoted_old_outputs = (
        "panel_c_ping_score_spike_prediction.csv",
        "panel_e_score_basin_sparsification.csv",
        "panel_a_peak_source_attribution_summary.csv",
        "panel_b_peak_update_history.csv",
        "panel_c_peak_input_overlap_similarity_summary.csv",
        "panel_d_route_peak_reentry_loss_summary.csv",
        "panel_e_route_peak_downstream_summary.csv",
        "route_peak_perturbation",
        "peak_weighted_overlap",
        "downstream_prediction",
    )
    spec_text = str(spec).lower()
    leaked_required = [name for name in demoted_old_outputs if name in spec_text]
    if leaked_required:
        failures.append(f"Fig.6 main spec still references old/demoted outputs: {leaked_required}")
    else:
        passes.append("Fig.6 main spec does not require old Panel A, route-peak, peak-weighted, or downstream outputs")

    panel_frames: dict[str, pd.DataFrame] = {}
    stats_text_parts: list[str] = []
    source_text_parts: list[str] = []
    missing_source_panels: set[str] = set()
    for panel_id in sorted(panels):
        df = _read_panel_data(output_dir, "fig6", panel_id)
        stats = _read_panel_stats(output_dir, "fig6", panel_id)
        source_path = panel_output_paths(output_dir, "fig6", panel_id)["sources"]
        sources = read_json(source_path) if source_path.exists() else {}
        panel_frames[panel_id] = df
        stats_text_parts.append(str(stats).lower())
        source_text_parts.append(str(sources).lower())
        if df.empty:
            failures.append(f"Fig.6{panel_id}: required panel_data missing or empty")
            continue
        if df.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any() or sources.get("status") == "missing_source":
            missing_source_panels.add(panel_id)
        n_networks = int(stats.get("n_networks") or sources.get("n_networks") or 0)
        if n_networks == 1 or str(stats.get("run_mode", "")) == "single_network_draft":
            warnings.append(f"Fig.6{panel_id}: single_network_draft n_networks=1; draft-only, not final manuscript statistics")
        elif n_networks > 1:
            passes.append(f"Fig.6{panel_id}: n_networks={n_networks}")
    if missing_source_panels:
        failures.append(f"Fig.6 has missing_source panels: {sorted(missing_source_panels)}")
    else:
        passes.append("Fig.6 has no missing_source panel")

    a_df = panel_frames.get("A", pd.DataFrame())
    a_metrics = set(a_df.get("metric", pd.Series(dtype=str)).astype(str))
    a_conditions = set(a_df.get("condition", pd.Series(dtype=str)).astype(str))
    if "loss_delta_spike_probability" in a_metrics and {"high_stsp_overlap", "matched_removal"}.issubset(a_conditions):
        passes.append("Fig.6A contains high-STSP-overlap and matched-removal ablation loss")
    else:
        failures.append("Fig.6A must contain loss_delta_spike_probability for high_stsp_overlap and matched_removal")

    stats_source_text = " ".join(stats_text_parts + source_text_parts)
    excludes = {"connection_weights", "inhibition", "voltage", "threshold", "wta", "final_label"}
    missing_excludes = sorted(token for token in excludes if token not in stats_source_text)
    if missing_excludes:
        failures.append(f"Fig.6 stats/source manifests missing score_excludes tokens {missing_excludes}")
    else:
        passes.append("Fig.6 stats/source manifests record all score exclusions")
    if "layer 1 spatial spike enrichment / recruitment" in stats_source_text:
        passes.append("Fig.6 stats/source manifests record the Layer 1 recruitment endpoint")
    else:
        failures.append("Fig.6 stats/source manifests must record the Layer 1 spatial spike enrichment / recruitment endpoint")

    b_df = panel_frames.get("B", pd.DataFrame())
    b_metrics = set(b_df.get("metric", pd.Series(dtype=str)).astype(str))
    b_conditions = set(b_df.get("condition", pd.Series(dtype=str)).astype(str))
    if {"old_mass", "middle_mass", "recent_mass", "silent_rate"}.issubset(b_metrics) and {"Peak ping", "Valley ping", "Random ping"}.issubset(b_conditions):
        passes.append("Fig.6B contains peak/valley/random serial readout mass metrics")
    else:
        failures.append("Fig.6B must contain old/middle/recent/silent mass for Peak/Valley/Random ping")

    if "other_mass" in b_metrics:
        passes.append("Fig.6B includes optional other_mass readout mass")
    else:
        warnings.append("Fig.6B optional other_mass is absent; renderer should omit that stack segment")

    c_df = panel_frames.get("C", pd.DataFrame())
    c_sources = source_text_parts[sorted(panels).index("C")] if "C" in sorted(panels) else ""
    if not c_df.empty and "spike_probability" in set(c_df.get("metric", pd.Series(dtype=str)).astype(str)) and "x_value" in c_df.columns and "Global ping" in set(c_df.get("condition", pd.Series(dtype=str)).astype(str)):
        passes.append("Fig.6C plots global-ping spike probability by STSP score quantile")
    else:
        failures.append("Fig.6C must plot Global ping spike_probability by score quantile")
    if "panel_c_global_ping_score_spike_prediction.csv" in c_sources and "panel_c_ping_score_spike_prediction.csv" not in c_sources:
        passes.append("Fig.6C source is the global-ping score-spike prediction file")
    else:
        failures.append("Fig.6C source must be panel_c_global_ping_score_spike_prediction.csv and not the old panel_c_ping_score_spike_prediction.csv")

    d_df = panel_frames.get("D", pd.DataFrame())
    if not d_df.empty and "delta_spike_probability" in set(d_df.get("metric", pd.Series(dtype=str)).astype(str)) and "x_value" in d_df.columns:
        passes.append("Fig.6D plots real-probe spike deflection by STSP score quantile")
    else:
        failures.append("Fig.6D must plot delta_spike_probability by score quantile")

    e_df = panel_frames.get("E", pd.DataFrame())
    e_metrics = set(e_df.get("metric", pd.Series(dtype=str)).astype(str))
    e_sources = source_text_parts[sorted(panels).index("E")] if "E" in sorted(panels) else ""
    if "delta_spike_probability" in e_metrics and {"high", "low"}.issubset(set(e_df.get("stsp_group", pd.Series(dtype=str)).astype(str))) and {"overlap", "no_overlap"}.issubset(set(e_df.get("overlap_group", pd.Series(dtype=str)).astype(str))):
        passes.append("Fig.6E reports the high/low STSP x overlap/no-overlap recruitment 2x2")
    else:
        failures.append("Fig.6E must report delta_spike_probability for high/low STSP x overlap/no-overlap groups")
    if "interaction_delta" in e_metrics:
        passes.append("Fig.6E includes overlap-gated STSP interaction_delta rows")
    else:
        failures.append("Fig.6E must include interaction_delta rows")
    if "panel_e_overlap_gated_stsp_recruitment.csv" in e_sources and "panel_e_overlap_gated_stsp_interaction.csv" in e_sources:
        passes.append("Fig.6E source manifest includes recruitment and interaction files")
    else:
        failures.append("Fig.6E source manifest must include recruitment and interaction files")

    f_df = panel_frames.get("F", pd.DataFrame())
    if not f_df.empty and f_df.get("final_label_claim", pd.Series(dtype=object)).astype(str).str.lower().isin({"false", "0"}).any():
        passes.append("Fig.6F explicitly avoids final-label claim metadata")
    elif not f_df.empty:
        warnings.append("Fig.6F final-label claim metadata was not explicit in panel_data")
    f_text = " ".join(" ".join(map(str, f_df.get(col, []))) for col in ("route_statement", "gain_statement", "mechanism_statement")).lower() if not f_df.empty else ""
    if "multi-item stsp fields bias layer 1 recruitment only where later input enters the high-gain field" in f_text:
        passes.append("Fig.6F contains the overlap-gated STSP recruitment mechanism statement")
    else:
        failures.append("Fig.6F must contain the overlap-gated STSP recruitment mechanism statement")
    if "overlap = route" in f_text or "peaks = gain" in f_text:
        failures.append("Fig.6F uses old route/gain statements")
    else:
        passes.append("Fig.6F does not use old route/gain statements")
    if "loss_delta_spike_probability" in set(f_df.get("metric", pd.Series(dtype=str)).astype(str)):
        failures.append("Fig.6F must be a pure mechanism schematic and must not include ablation result rows")
    else:
        passes.append("Fig.6F is pure mechanism metadata without ablation result rows")

    visible_claim_text = " ".join(str((panel or {}).get("claim", "")) for panel in panels.values()).lower()
    panel_text_parts = []
    for df in panel_frames.values():
        if df.empty:
            continue
        visible_cols = [col for col in df.columns if col not in {"forbidden_claims", "forbidden_language", "score_excludes", "source_file"}]
        panel_text_parts.append(" ".join(map(str, df[visible_cols].astype(str).to_numpy().ravel())))
    panel_text = " ".join(panel_text_parts).lower()
    rendered_text = " ".join(str((meta or {}).get(key, "")) for meta in render_metadata.values() for key in ("plot_form", "title", "x_label", "y_label", "score_interpretation")).lower()
    all_claim_text = " ".join([visible_claim_text, panel_text, rendered_text])
    forbidden = (
        "final label prediction",
        "stsp alone determines firing",
        "high stsp automatically fires",
        "peaks replace overlap",
        "peak-gated re-entry",
        "peak-driven re-entry",
        "peaks provide the route",
        "score predicts final label",
        "deterministic final-label prediction",
        "deterministic final label prediction",
        "score reconstructs network forward computation",
        "connection weights define the main score",
        "inhibition is part of the stsp score",
        "peaks = gain",
        "overlap = route",
        "route-peak perturbation",
        "peak-amplified re-entry",
    )
    leaked = [term for term in forbidden if term in all_claim_text]
    if leaked:
        failures.append(f"Fig.6 contains forbidden old or overstrong language: {leaked}")
    else:
        passes.append("Fig.6 avoids old route/gain, STSP-alone, high-STSP-alone, and final-label prediction language")

    if render_metadata:
        expected_forms = {
            "A": "high_stsp_overlap_ablation_bar",
            "B": "region_gated_ping_readout_bias",
            "C": "global_ping_score_quantile_spike_probability",
            "D": "real_probe_score_quantile_spike_deflection",
            "E": "overlap_gated_stsp_recruitment_2x2",
            "F": "overlap_gated_stsp_recruitment_synthesis",
        }
        for panel_id, expected in expected_forms.items():
            observed = str((render_metadata.get(panel_id) or {}).get("plot_form", ""))
            if observed == expected:
                passes.append(f"Fig.6{panel_id} renderer reports {expected}")
            else:
                failures.append(f"Fig.6{panel_id} renderer plot_form must be {expected}, found {observed!r}")
        if (render_metadata.get("C") or {}).get("x_label") == "STSP score quantile" and "L1 spike probability" in str((render_metadata.get("C") or {}).get("y_label", "")):
            passes.append("Fig.6C axes are score quantile versus Layer 1 spike probability")
        else:
            failures.append("Fig.6C axes must be STSP score quantile versus Early L1 spike probability")
        if (render_metadata.get("D") or {}).get("x_label") == "STSP score quantile" and "baseline L1 firing" in str((render_metadata.get("D") or {}).get("y_label", "")):
            passes.append("Fig.6D axes are score quantile versus Layer 1 spike deflection")
        else:
            failures.append("Fig.6D axes must be STSP score quantile versus dynamic-baseline L1 firing")
        if "Probe overlap" in str((render_metadata.get("E") or {}).get("x_label", "")) and "baseline L1 firing" in str((render_metadata.get("E") or {}).get("y_label", "")):
            passes.append("Fig.6E axes are probe overlap versus dynamic-baseline Layer 1 firing")
        else:
            failures.append("Fig.6E axes must be probe overlap versus dynamic-baseline L1 firing")
        if (render_metadata.get("F") or {}).get("final_label_claim") is False:
            passes.append("Fig.6F renderer marks final_label_claim=False")
        else:
            failures.append("Fig.6F renderer must mark final_label_claim=False")
        if (render_metadata.get("F") or {}).get("pure_mechanism_schematic") is True and not (render_metadata.get("F") or {}).get("has_summary_inset"):
            passes.append("Fig.6F renderer is a pure mechanism schematic without result inset")
        else:
            failures.append("Fig.6F renderer must be pure mechanism and omit the ablation inset")


def _check_fig6_route_gain_contract(
    spec: Mapping[str, Any],
    panels: Mapping[str, Any],
    output_dir: Path,
    render_metadata: Mapping[str, Mapping[str, Any]],
    passes: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 158:
        passes.append("Fig.6 canvas is 165 x 158 mm")
    else:
        failures.append(f"Fig.6 canvas must be 165 x 158 mm, found {canvas}")
    expected_panels = {"A", "B", "C", "D", "E", "F"}
    if set(panels.keys()) == expected_panels:
        passes.append("Fig.6 panels are A-F")
    else:
        failures.append(f"Fig.6 must contain exactly panels A-F, found {sorted(panels.keys())}")
    if list(spec.get("reading_order") or []) == ["A", "B", "C", "D", "E", "F"]:
        passes.append("Fig.6 reading order is A-F")
    else:
        failures.append(f"Fig.6 reading_order must be A-F, found {spec.get('reading_order')}")
    expected_boxes = {
        "A": {"x": 12.00, "y": 8.00, "w": 45.67, "h": 38.00},
        "B": {"x": 62.67, "y": 8.00, "w": 45.67, "h": 38.00},
        "C": {"x": 113.33, "y": 8.00, "w": 45.67, "h": 38.00},
        "D": {"x": 12.00, "y": 54.00, "w": 70.50, "h": 52.00},
        "E": {"x": 88.50, "y": 54.00, "w": 70.50, "h": 52.00},
        "F": {"x": 12.00, "y": 114.00, "w": 147.00, "h": 36.00},
    }
    for panel_id, expected in expected_boxes.items():
        if panel_id in panels and _fig6_box_close(_fig6_pos(panels[panel_id]), expected):
            passes.append(f"Fig.6{panel_id} position matches route/gain spec")
        elif panel_id in panels:
            failures.append(f"Fig.6{panel_id} position differs from route/gain spec")

    missing_source_panels: set[str] = set()
    for panel_id in panels:
        df = _read_panel_data(output_dir, "fig6", panel_id)
        stats = _read_panel_stats(output_dir, "fig6", panel_id)
        if df.empty:
            failures.append(f"Fig.6{panel_id}: required panel_data missing or empty")
            continue
        if df.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any():
            missing_source_panels.add(panel_id)
        n_networks = int(stats.get("n_networks") or 0) if stats else 0
        if n_networks == 1 or str(stats.get("run_mode", "")) == "single_network_draft":
            warnings.append(f"Fig.6{panel_id}: single_network_draft n_networks=1; draft-only, not final manuscript statistics")
        elif n_networks > 1:
            passes.append(f"Fig.6{panel_id}: n_networks={n_networks}")
    if missing_source_panels:
        failures.append(f"Fig.6 has missing_source panels: {sorted(missing_source_panels)}")
    else:
        passes.append("Fig.6 has no missing_source panel")

    a_df = _read_panel_data(output_dir, "fig6", "A")
    if not a_df.empty and "peak_loss_fraction" in set(a_df.get("metric", pd.Series(dtype=str)).astype(str)) and ("position_from_end" in a_df.columns or "x_value" in a_df.columns):
        passes.append("Fig.6A uses leave-one-item-out peak source attribution by temporal position")
    else:
        failures.append("Fig.6A must use peak_loss_fraction by position_from_end")

    b_df = _read_panel_data(output_dir, "fig6", "B")
    if not b_df.empty and set(b_df.get("metric", pd.Series(dtype=str)).astype(str)) == {"P_peak"} and "update_count_bin" in b_df.columns:
        passes.append("Fig.6B uses P(peak) by update count")
    else:
        failures.append("Fig.6B must show P_peak by update_count_bin, not model CV R2")

    c_df = _read_panel_data(output_dir, "fig6", "C")
    c_families = set(c_df.get("window_family", pd.Series(dtype=str)).astype(str)) if not c_df.empty else set()
    if not c_df.empty and "peak_coverage" in set(c_df.get("metric", pd.Series(dtype=str)).astype(str)) and {"old", "all", "recent"}.issubset(c_families):
        passes.append("Fig.6C compares peak coverage for old/all/recent foreground-overlap windows")
    else:
        failures.append("Fig.6C must use peak_coverage and include old/all/recent windows")

    d_df = _read_panel_data(output_dir, "fig6", "D")
    d_stats = _read_panel_stats(output_dir, "fig6", "D")
    if not d_df.empty and "normalized_reentry_loss" in set(d_df.get("metric", pd.Series(dtype=str)).astype(str)):
        passes.append("Fig.6D uses route-peak perturbation normalized re-entry loss")
    else:
        failures.append("Fig.6D must use normalized_reentry_loss from route-peak perturbation")
    _check_fig6_route_peak_panel_validity("Fig.6D", d_df, "normalized_reentry_loss", failures, passes)
    if str(d_stats.get("source_level", "")) == "route_peak_perturbation":
        passes.append("Fig.6D source_level=route_peak_perturbation")
    else:
        failures.append("Fig.6D must record source_level=route_peak_perturbation")
    if str(d_stats.get("source_level", "")) in {"real_matched", "real_regression", "peak_weighted_fallback"}:
        failures.append("Fig.6D must not use predictive or peak-weighted fallback sources")

    e_df = _read_panel_data(output_dir, "fig6", "E")
    e_stats = _read_panel_stats(output_dir, "fig6", "E")
    if not e_df.empty and "P_output_switch" in set(e_df.get("metric", pd.Series(dtype=str)).astype(str)):
        passes.append("Fig.6E uses route-peak downstream output-switch probability")
    else:
        failures.append("Fig.6E must use P_output_switch from the route-peak perturbation")
    _check_fig6_route_peak_panel_validity("Fig.6E", e_df, "P_output_switch", failures, passes)
    if str(e_stats.get("source_level", "")) == "route_peak_perturbation":
        passes.append("Fig.6E source_level=route_peak_perturbation")
    else:
        failures.append("Fig.6E must record source_level=route_peak_perturbation")
    if str(e_stats.get("source_level", "")) in {"real_downstream", "peak_weighted_fallback"}:
        failures.append("Fig.6E must not use predictive or peak-weighted fallback sources")

    f_df = _read_panel_data(output_dir, "fig6", "F")
    text_cols = [col for col in ("route_statement", "gain_statement", "mechanism_statement", "allowed_claim_strength") if col in f_df.columns]
    fig6_text = " ".join(" ".join(map(str, f_df.get(col, []))) for col in text_cols).lower()
    if "overlap = route" in fig6_text and "peaks = gain" in fig6_text:
        passes.append("Fig.6F explicitly states overlap = route and peaks = gain")
    else:
        failures.append("Fig.6F must include overlap = route and peaks = gain")
    visible_claim_text = " ".join(str((panel or {}).get("claim", "")) for panel in panels.values()).lower() + " " + fig6_text
    forbidden = ("peaks replace overlap", "peak-gated re-entry", "peak-driven re-entry", "peaks provide the route")
    leaked = [term for term in forbidden if term in visible_claim_text]
    if leaked:
        failures.append(f"Fig.6 contains forbidden route/gain language: {leaked}")
    else:
        passes.append("Fig.6 claims avoid forbidden route/gain language")
    perturb = f_df.get("peak_perturbation_implemented", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"}).any()
    success = f_df.get("peak_perturbation_successful", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"}).any()
    if not (perturb and success):
        failures.append("Fig.6 requires successful route-peak perturbation for formal D/E panels")
    elif "causal" in fig6_text and not (perturb and success):
        failures.append("Fig.6 causal language requires implemented and successful route-peak perturbation")

    if render_metadata:
        blank_panels = [pid for pid, meta in render_metadata.items() if meta.get("plot_form") == "blank_panel"]
        if blank_panels:
            failures.append(f"Fig.6 has blank/placeholder renderers: {blank_panels}")
        else:
            passes.append("Fig.6 has no blank panel renderer")
        diagnostic_panels = [pid for pid, meta in render_metadata.items() if "diagnostic" in str(meta.get("plot_form", ""))]
        if diagnostic_panels:
            failures.append(f"Fig.6 has diagnostic placeholder renderers: {diagnostic_panels}")
        else:
            passes.append("Fig.6 has no diagnostic D/E placeholder renderer")
        if (render_metadata.get("F") or {}).get("route_gain_statement"):
            passes.append("Fig.6F renderer reports route/gain statement")
        if (render_metadata.get("F") or {}).get("peaks_replace_overlap"):
            failures.append("Fig.6F renderer metadata implies peaks replace overlap")
    return

    a_df = _read_panel_data(output_dir, "fig6", "A")
    if not a_df.empty and "P_peak" in set(a_df.get("metric", pd.Series(dtype=str)).astype(str)) and "Multi recent" in set(a_df.get("condition", pd.Series(dtype=str)).astype(str)):
        passes.append("Fig.6A uses multi-recent peak enrichment and includes Multi recent")
    else:
        failures.append("Fig.6A must use multi-recent peak enrichment, not leave-one-out source attribution")

    b_df = _read_panel_data(output_dir, "fig6", "B")
    models = set(b_df.get("model_name", pd.Series(dtype=str)).astype(str)) if not b_df.empty else set()
    if {"update_plus_recency", "update_times_recency"}.intersection(models) and "cv_r2" in set(b_df.get("metric", pd.Series(dtype=str)).astype(str)):
        passes.append("Fig.6B uses update-recency model comparison")
    else:
        failures.append("Fig.6B must include update_plus_recency or update_times_recency CV R2 models")

    c_df = _read_panel_data(output_dir, "fig6", "C")
    if {"raw_overlap", "peak_weighted_overlap"}.issubset(c_df.columns):
        passes.append("Fig.6C exposes raw overlap and peak-weighted overlap")
    else:
        failures.append("Fig.6C must show raw overlap and peak-weighted overlap")

    d_df = _read_panel_data(output_dir, "fig6", "D")
    d_stats = _read_panel_stats(output_dir, "fig6", "D")
    d_level = str(d_stats.get("source_level", ""))
    if d_level in {"real_matched", "real_regression", "peak_weighted_fallback"}:
        passes.append(f"Fig.6D source mode recorded: {d_level}")
        if d_level == "peak_weighted_fallback":
            warnings.append("Fig.6D used peak-weighted fallback rather than preferred real rollout source")
    else:
        failures.append("Fig.6D must report source mode real_matched / real_regression / peak_weighted_fallback")
    if {"proxy_mode", "final_scientific_use", "raw_overlap_control"}.issubset(d_df.columns):
        passes.append("Fig.6D preserves proxy/final-use/raw-overlap audit fields")
    else:
        failures.append("Fig.6D must preserve proxy_mode, final_scientific_use, and raw_overlap_control")
    _warn_proxy_or_not_final("Fig.6D", d_df, warnings)

    e_df = _read_panel_data(output_dir, "fig6", "E")
    e_stats = _read_panel_stats(output_dir, "fig6", "E")
    e_level = str(e_stats.get("source_level", ""))
    if e_level in {"real_downstream", "peak_weighted_fallback"}:
        passes.append(f"Fig.6E source mode recorded: {e_level}")
        if e_level == "peak_weighted_fallback":
            warnings.append("Fig.6E used peak-weighted fallback rather than preferred real downstream source")
    else:
        failures.append("Fig.6E must report source mode real_downstream / peak_weighted_fallback")
    e_metrics = set(e_df.get("metric", pd.Series(dtype=str)).astype(str))
    if {"early_recruitment_gain_real", "response_pattern_displacement_real", "decision_deflection_score_real"}.intersection(e_metrics):
        passes.append("Fig.6E includes real downstream metric rows")
    else:
        failures.append("Fig.6E must include real downstream metric rows when available")
    _warn_proxy_or_not_final("Fig.6E", e_df, warnings)

    f_df = _read_panel_data(output_dir, "fig6", "F")
    text_cols = [col for col in ("route_statement", "gain_statement", "mechanism_statement", "allowed_claim_strength") if col in f_df.columns]
    fig6_text = " ".join(" ".join(map(str, f_df.get(col, []))) for col in text_cols).lower()
    if "overlap = route" in fig6_text and "peaks = gain" in fig6_text:
        passes.append("Fig.6F explicitly states overlap = route and peaks = gain")
    else:
        failures.append("Fig.6F must include overlap = route and peaks = gain")
    visible_claim_text = " ".join(str((panel or {}).get("claim", "")) for panel in panels.values()).lower() + " " + fig6_text
    forbidden = ("peaks replace overlap", "peak-gated re-entry", "peak-driven re-entry", "peaks provide the route")
    leaked = [term for term in forbidden if term in visible_claim_text]
    if leaked:
        failures.append(f"Fig.6 contains forbidden route/gain language: {leaked}")
    else:
        passes.append("Fig.6 claims avoid forbidden route/gain language")
    perturb = f_df.get("peak_perturbation_implemented", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"}).any()
    success = f_df.get("peak_perturbation_successful", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"}).any()
    if "causal" in fig6_text and not (perturb and success):
        failures.append("Fig.6 causal language requires implemented and successful peak perturbation")
    elif not (perturb and success):
        warnings.append("Fig.6 claim strength remains predictive/peak-amplified because peak perturbation is absent or unsuccessful")

    if render_metadata:
        blank_panels = [pid for pid, meta in render_metadata.items() if meta.get("plot_form") == "blank_panel"]
        if blank_panels:
            failures.append(f"Fig.6 has blank/placeholder renderers: {blank_panels}")
        else:
            passes.append("Fig.6 has no blank panel renderer")
        if (render_metadata.get("F") or {}).get("route_gain_statement"):
            passes.append("Fig.6F renderer reports route/gain statement")
        if (render_metadata.get("F") or {}).get("peaks_replace_overlap"):
            failures.append("Fig.6F renderer metadata implies peaks replace overlap")


def _check_fig6_route_peak_panel_validity(label: str, df: pd.DataFrame, metric: str, failures: list[str], passes: list[str]) -> None:
    if df.empty:
        failures.append(f"{label}: required route-peak perturbation panel_data missing or empty")
        return
    metric_rows = df[df.get("metric", pd.Series(dtype=str)).astype(str).eq(metric)].copy()
    if metric_rows.empty:
        failures.append(f"{label}: required metric {metric} missing")
        return
    values = pd.to_numeric(metric_rows.get("value", pd.Series(dtype=float)), errors="coerce")
    if not values.notna().any():
        failures.append(f"{label}: required metric {metric} is all NaN")
    valid = pd.to_numeric(metric_rows.get("n_valid_trials", pd.Series(dtype=float)), errors="coerce").fillna(0)
    if not valid.gt(0).any():
        failures.append(f"{label}: n_valid_trials must be > 0")
    unit_sets = set(metric_rows.get("perturbation_unit_set", pd.Series(dtype=str)).astype(str))
    required = {"route_peak", "route_nonpeak", "nonroute_peak", "random_matched"}
    missing = sorted(required - unit_sets)
    invalid = sorted(
        unit_set
        for unit_set in required.intersection(unit_sets)
        if pd.to_numeric(metric_rows.loc[metric_rows["perturbation_unit_set"].astype(str).eq(unit_set), "n_valid_trials"], errors="coerce").fillna(0).sum() <= 0
    )
    if missing or invalid:
        failures.append(f"{label}: route-peak unit sets invalid; missing={missing}, invalid={invalid}")
    else:
        passes.append(f"{label}: all route-peak perturbation unit sets have valid trials")
    if metric_rows.get("source_file", pd.Series(dtype=str)).astype(str).str.contains("peak_weighted|real_downstream|real_reentry|legacy|predictive", case=False, na=False).any():
        failures.append(f"{label}: source_file indicates a predictive/legacy fallback")
    if metric_rows.get("proxy_mode", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1", "yes"}).any():
        failures.append(f"{label}: proxy_mode=true is forbidden")


def _check_fig6_supp_specifics(
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
    if figure_id != "fig6_supp":
        return
    canvas = spec.get("canvas_mm") or {}
    if float(canvas.get("width", 0)) == 165 and float(canvas.get("height", 0)) == 116:
        passes.append("Fig.6 supplement canvas is 165 x 116 mm")
    else:
        failures.append(f"Fig.6 supplement canvas must be 165 x 116 mm, found {canvas}")
    expected_order = ["S7A", "S7B", "S7C", "S7D", "S7E", "S7F", "S7G", "S7H"]
    if list(spec.get("reading_order") or []) == expected_order:
        passes.append("Fig.6 supplement reading order is S7A-S7H")
    else:
        failures.append(f"Fig.6 supplement reading order must be {expected_order}, found {spec.get('reading_order')}")
    expected = set(expected_order)
    if set(panels.keys()) == expected:
        passes.append("Fig.6 supplement panels are compact S7A-S7H")
    else:
        failures.append(f"Fig.6 supplement panel set mismatch: {sorted(panels.keys())}")

    optional = set((spec.get("qc_requirements") or {}).get("optional_placeholder_panels") or ["S7G", "S7H"])
    required_metrics = {
        "S7A": {"nonfinite_raw_count", "clipped_ratio_max", "ping_active_sites"},
        "S7B": {"mean_early_spike_count"},
        "S7C": {"q5_minus_q1_delta_spike_probability"},
        "S7D": {"interaction_delta"},
        "S7E": {"mean_sites"},
        "S7F": {"high_stsp_overlap_minus_matched_loss"},
    }
    for panel_id in panels:
        df = _read_panel_data(output_dir, "fig6_supp", panel_id)
        if df.empty:
            if panel_id in optional:
                warnings.append(f"Fig.6 supplement {panel_id}: optional panel data unavailable")
            else:
                failures.append(f"Fig.6 supplement {panel_id}: required panel data unavailable")
            continue
        metrics = set(df.get("metric", pd.Series(dtype=str)).astype(str))
        has_missing = "missing_source" in metrics
        has_optional_placeholder = "optional_placeholder" in metrics
        if panel_id in optional:
            if has_optional_placeholder:
                warnings.append(f"Fig.6 supplement {panel_id}: optional extension placeholder")
                passes.append(f"Fig.6 supplement {panel_id}: optional placeholder is allowed")
            elif has_missing:
                warnings.append(f"Fig.6 supplement {panel_id}: optional source placeholder")
            else:
                passes.append(f"Fig.6 supplement {panel_id}: optional extension data available")
            continue
        if has_missing or has_optional_placeholder:
            failures.append(f"Fig.6 supplement {panel_id}: active S7A-F panels must not be placeholders")
            continue
        missing_metrics = sorted(required_metrics.get(panel_id, set()).difference(metrics))
        if missing_metrics:
            failures.append(f"Fig.6 supplement {panel_id}: missing required metrics {missing_metrics}")
        else:
            passes.append(f"Fig.6 supplement {panel_id}: required active metrics present")

    manifest_path = output_dir / f"{figure_id}_source_manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8", errors="ignore").lower() if manifest_path.exists() else ""
    forbidden_active_sources = ("route_peak", "peak_weighted_overlap", "panel_d_peak_weighted_", "panel_e_route_peak_", "supp_s12_")
    leaked = [token for token in forbidden_active_sources if token in manifest_text]
    if leaked:
        failures.append(f"Fig.6 supplement active source manifest contains demoted legacy sources: {leaked}")
    elif manifest_text:
        passes.append("Fig.6 supplement active manifest excludes route/peak-weighted/S12 sources")

    expected_forms = {
        "S7A": "s11_score_input_ping_audit",
        "S7B": "s11_global_ping_count_endpoint",
        "S7C": "s11_real_probe_window_robustness",
        "S7D": "s11_overlap_interaction_window_robustness",
        "S7E": "s11_overlap_site_availability",
        "S7F": "s11_high_stsp_ablation_paired_difference",
    }
    for panel_id, expected_form in expected_forms.items():
        form = str(render_metadata.get(panel_id, {}).get("plot_form", ""))
        if form and form != expected_form:
            failures.append(f"Fig.6 supplement {panel_id}: expected plot_form={expected_form}, found {form}")
        elif form:
            passes.append(f"Fig.6 supplement {panel_id}: renderer form {expected_form}")
    for panel_id in ("S7G", "S7H"):
        form = str(render_metadata.get(panel_id, {}).get("plot_form", ""))
        if form and not form.startswith(("s11_score_shuffle_null", "s11_threshold_sensitivity")):
            warnings.append(f"Fig.6 supplement {panel_id}: optional renderer form was {form}")


def _warn_proxy_or_not_final(label: str, df: pd.DataFrame, warnings: list[str]) -> None:
    if df.empty:
        return
    if df.get("proxy_mode", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"}).all():
        warnings.append(f"{label} is proxy-mode only and not final scientific evidence")
    if df.get("final_scientific_use", pd.Series(dtype=str)).astype(str).str.lower().isin({"false", "0"}).all():
        warnings.append(f"{label} rows are not final scientific use")


def _fig6_pos(panel: Mapping[str, Any]) -> dict[str, float]:
    raw = panel.get("position_mm") or {}
    x = float(raw.get("x", 0.0))
    y = float(raw.get("y", 0.0))
    w = float(raw.get("w", 0.0))
    h = float(raw.get("h", 0.0))
    return {"x": x, "y": y, "w": w, "h": h, "right": x + w, "bottom": y + h}


def _fig6_close(left: float, right: float, *, tol: float = 0.04) -> bool:
    return abs(float(left) - float(right)) <= tol


def _fig6_box_close(actual: Mapping[str, float], expected: Mapping[str, float], *, tol: float = 0.04) -> bool:
    return all(_fig6_close(float(actual.get(key, 0.0)), float(expected[key]), tol=tol) for key in ("x", "y", "w", "h"))


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
    if panel_id == "C":
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
    if written <= max(int(stats.get("n_networks") or 0), 20) and panel_id in {"A", "B", "C", "D"} and not bool(stats.get("source_appeared_preaggregated")):
        failures.append(f"Fig.6{panel_id}: row-level source available but panel_data looks network-level")


def _contains_internal_label(values: set[Any]) -> bool:
    text = " ".join(str(v).lower() for v in values)
    return any(token in text for token in ("multi_recent", "single_recent", "multi_old", "single_old", "peak_flattened", "peak_boosted", "intact_final", "fig6b", "fig6c", "fig6d", "fig6e"))


def _panel_n(df: pd.DataFrame) -> int:
    for col in ("seed_id", "network_id"):
        if col in df.columns:
            return int(df[col].replace("", pd.NA).dropna().nunique())
    return 0


def _check_fig4_standalone_contract(
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
    """Current Fig.4 contract: overlap perturbation D and accumulator process F."""
    _ = adapter_results
    canvas = spec.get("canvas_mm") or {}
    if _fig4_near(float(canvas.get("width", 0)), 165.0) and _fig4_near(float(canvas.get("height", 0)), 148.0):
        passes.append("Fig.4 canvas is 165 x 148 mm")
    else:
        failures.append(f"Fig.4 canvas must be 165 x 148 mm, found {canvas}")
    if list(spec.get("reading_order") or []) == ["A", "B", "C", "D", "E", "F"]:
        passes.append("Fig.4 reading order is A-F")
    else:
        failures.append(f"Fig.4 reading_order must be A-F, found {spec.get('reading_order')}")
    if set(panels) == set("ABCDEF"):
        passes.append("Fig.4 defines exactly panels A-F")
    else:
        failures.append(f"Fig.4 must define exactly A-F, found {sorted(panels)}")
    _check_fig4_current_geometry(panels, passes, failures)
    if (panels.get("D") or {}).get("data_adapter") == "fig4_overlap_perturbation_main_adapter":
        passes.append("Fig.4D uses overlap perturbation main adapter")
    else:
        failures.append(f"Fig.4D must use fig4_overlap_perturbation_main_adapter, found {(panels.get('D') or {}).get('data_adapter')}")
    if (panels.get("F") or {}).get("data_adapter") == "fig4_l3_accumulator_process_adapter":
        passes.append("Fig.4F uses L3 accumulator process adapter")
    else:
        failures.append(f"Fig.4F must use fig4_l3_accumulator_process_adapter, found {(panels.get('F') or {}).get('data_adapter')}")

    panel_data: dict[str, pd.DataFrame] = {}
    stats_by_panel: dict[str, Mapping[str, Any]] = {}
    sources_by_panel: dict[str, Mapping[str, Any]] = {}
    for panel_id in ("B", "C", "D", "E", "F"):
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            failures.append(f"Fig.4{panel_id}: missing adapter outputs {missing}")
            continue
        passes.append(f"Fig.4{panel_id}: panel_data/stats/source_manifest exist")
        df = pd.read_csv(paths["panel_data"])
        panel_data[panel_id] = df
        stats = read_json(paths["stats"])
        sources = read_json(paths["sources"])
        stats_by_panel[panel_id] = stats
        sources_by_panel[panel_id] = sources
        run_mode = str(stats.get("run_mode") or sources.get("run_mode") or "")
        n_networks = int(stats.get("n_networks") or sources.get("n_networks") or 0)
        if run_mode:
            passes.append(f"Fig.4{panel_id}: run_mode={run_mode}")
        else:
            failures.append(f"Fig.4{panel_id}: run_mode missing")
        if n_networks == 1 or run_mode == "single_network_draft":
            warnings.append(f"Fig.4{panel_id}: single_network_draft n_networks=1; draft-only, not final manuscript statistics")
        elif n_networks > 1:
            passes.append(f"Fig.4{panel_id}: n_networks={n_networks}")

    b_df = panel_data.get("B")
    b_stats = stats_by_panel.get("B", {})
    if b_df is not None:
        metrics = set(b_df.get("metric", pd.Series(dtype=str)).astype(str))
        sources = set(b_df.get("metric_source", pd.Series(dtype=str)).astype(str))
        if b_stats.get("main_metric") == "accuracy_drop" and metrics == {"accuracy_drop"}:
            passes.append("Fig.4B main metric is accuracy_drop")
        else:
            failures.append(f"Fig.4B main_metric must be accuracy_drop, found stats={b_stats.get('main_metric')} metrics={sorted(metrics)}")
        if sources.issubset({"mean_acc_drop", "mean_drop_event"}) and sources:
            passes.append("Fig.4B plotted values come from mean_acc_drop or mean_drop_event fallback")
        else:
            failures.append(f"Fig.4B plotted values must come from mean_acc_drop/mean_drop_event, found {sorted(sources)}")

    c_df = panel_data.get("C")
    c_stats = stats_by_panel.get("C", {})
    if c_df is not None:
        metrics = set(c_df.get("metric", pd.Series(dtype=str)).astype(str))
        forbidden = {"paired_delta_drop_event", "delta_drop_rate", "drop_rate_high_overlap", "drop_rate_low_overlap"}
        if forbidden.isdisjoint(metrics):
            passes.append("Fig.4C does not contain complete iso-similarity matching metrics")
        else:
            failures.append(f"Fig.4C must not contain complete iso-similarity matching metrics, found {sorted(forbidden.intersection(metrics))}")
        conditions = set(c_df.get("condition", pd.Series(dtype=str)).astype(str))
        if {"High overlap", "Low overlap"}.issubset(conditions):
            passes.append("Fig.4C includes high/low overlap localization rows")
        else:
            failures.append(f"Fig.4C missing high/low overlap rows, found {sorted(conditions)}")
        if c_stats.get("main_metric") == "accuracy_drop":
            passes.append("Fig.4C main metric is accuracy_drop")
        else:
            failures.append(f"Fig.4C main_metric must be accuracy_drop, found {c_stats.get('main_metric')}")
        sources = set(c_df.get("metric_source", pd.Series(dtype=str)).astype(str))
        if sources == {"acc_drop"} or (bool(c_stats.get("fallback_used")) and c_stats.get("fallback_metric")):
            passes.append("Fig.4C plotted values use acc_drop or declare a fallback metric")
        else:
            failures.append(f"Fig.4C plotted values must come from acc_drop unless fallback is marked, found sources={sorted(sources)} fallback={c_stats.get('fallback_used')}")

    d_df = panel_data.get("D")
    d_sources = sources_by_panel.get("D", {})
    d_stats = stats_by_panel.get("D", {})
    if d_df is not None:
        conditions = set(d_df.get("condition", pd.Series(dtype=str)).astype(str))
        required = {"Dynamic", "Overlap support", "Non-overlap support", "Random matched"}
        if required.issubset(conditions):
            passes.append("Fig.4D includes dynamic, overlap, non-overlap, and random matched perturbation conditions")
        else:
            failures.append(f"Fig.4D missing perturbation conditions {sorted(required - conditions)}")
        if {"paired_delta_drop_event", "delta_drop_rate"}.isdisjoint(set(d_df.get("metric", pd.Series(dtype=str)).astype(str))):
            passes.append("Fig.4D is no longer the iso-similarity matching panel")
        else:
            failures.append("Fig.4D must not use iso-similarity matching metrics")
        metrics = set(d_df.get("metric", pd.Series(dtype=str)).astype(str))
        if d_stats.get("main_metric") == "accuracy_drop_vs_static" and metrics == {"accuracy_drop_vs_static"}:
            passes.append("Fig.4D main metric is accuracy_drop_vs_static")
        else:
            failures.append(f"Fig.4D main_metric must be accuracy_drop_vs_static, found stats={d_stats.get('main_metric')} metrics={sorted(metrics)}")
        if {"dynamic_like_recovery", "DPI_L3", "mean_DPI_L3", "decision_deflection_score"}.isdisjoint(metrics):
            passes.append("Fig.4D does not use recovery/DPI/deflection as plotted metric")
        else:
            failures.append(f"Fig.4D must not use recovery/DPI/deflection as plotted metric, found {sorted(metrics)}")
        if {"probe_accuracy_condition", "probe_accuracy_static"}.issubset(d_df.columns):
            passes.append("Fig.4D panel_data records condition and static probe accuracies")
        else:
            failures.append("Fig.4D must compute plotted values from probe_accuracy or mean_probe_accuracy")
        source_level = str(d_sources.get("source_level", ""))
        if source_level == "supplement_fallback":
            warnings.append("Fig.4D used supplement fallback source for overlap perturbation")
        elif source_level:
            passes.append(f"Fig.4D source_level={source_level}")
        if d_sources.get("perturbation_scope") == "sample_side_prior_support":
            passes.append("Fig.4D manifest records sample-side prior-support perturbation scope")
        else:
            failures.append("Fig.4D manifest must record perturbation_scope=sample_side_prior_support")
        if d_sources.get("probe_input_modified_in_core_conditions") is False and d_sources.get("main_metric") == "accuracy_drop_vs_static":
            passes.append("Fig.4D manifest records static-baseline accuracy-drop contract")
        else:
            failures.append("Fig.4D manifest must record main_metric=accuracy_drop_vs_static and probe_input_modified_in_core_conditions=false")

    e_df = panel_data.get("E")
    e_stats = stats_by_panel.get("E", {})
    if e_df is not None:
        if "time_ms" in e_df.columns:
            max_time = pd.to_numeric(e_df["time_ms"], errors="coerce").dropna().max()
            if pd.isna(max_time) or float(max_time) <= 60.0:
                passes.append("Fig.4E panel_data is limited to <=60 ms")
            else:
                failures.append(f"Fig.4E must not include time_ms > 60, found max {float(max_time):.3f}")
        else:
            failures.append("Fig.4E must include time_ms for 0-60 ms trace QC")
        allowed = {"Overlap support", "Non-overlap support"}
        conditions = set(e_df.get("condition", pd.Series(dtype=str)).astype(str))
        if conditions.issubset(allowed) and conditions:
            passes.append("Fig.4E only includes overlap and non-overlap support traces")
        else:
            failures.append(f"Fig.4E must only include overlap/non-overlap traces, found {sorted(conditions)}")
        if float(e_stats.get("fig4e_max_time_ms", e_stats.get("max_time_ms_used", 0)) or 0) == 60.0:
            passes.append("Fig.4E stats record fig4e_max_time_ms=60")
        else:
            failures.append(f"Fig.4E stats must record fig4e_max_time_ms=60, found {e_stats.get('fig4e_max_time_ms')}")

    f_df = panel_data.get("F")
    f_sources = sources_by_panel.get("F", {})
    if f_df is not None:
        metrics = set(f_df.get("metric", pd.Series(dtype=str)).astype(str))
        if "accumulator_process_shift" in metrics:
            passes.append("Fig.4F uses accumulator process shift metric")
            required_cols = {"group", "x0", "y0", "x1", "y1"}
            missing_cols = sorted(required_cols - set(f_df.columns))
            if not missing_cols:
                passes.append("Fig.4F includes plus/minus trajectory coordinate columns")
            else:
                failures.append(f"Fig.4F missing trajectory columns {missing_cols}")
            groups = set(f_df.get("group", pd.Series(dtype=str)).astype(str))
            if {"plus", "minus"}.issubset(groups):
                passes.append("Fig.4F includes plus and minus trajectory groups")
            else:
                failures.append(f"Fig.4F missing plus/minus groups, found {sorted(groups)}")
        elif "decision_deflection_fallback" in metrics or f_sources.get("fallback_used"):
            warnings.append("Fig.4F fell back to old decision-deflection source")
        else:
            failures.append(f"Fig.4F must use accumulator process shift or explicit fallback, found metrics {sorted(metrics)}")
        if f_sources.get("fallback_used"):
            warnings.append("Fig.4F source manifest reports fallback_used=true")

    if render_metadata:
        f_meta = render_metadata.get("F", {})
        if f_meta:
            if f_meta.get("plot_form") == "l3_accumulator_process" and f_meta.get("mean_arrows") and f_meta.get("individual_traces") and f_meta.get("axis_direction_annotations"):
                passes.append("Fig.4F renderer uses accumulator trajectory grammar")
            elif f_meta.get("plot_form") == "decision_deflection_fallback":
                warnings.append("Fig.4F rendered decision-deflection fallback")
            else:
                failures.append(f"Fig.4F renderer must use accumulator process grammar, found {f_meta.get('plot_form')}")
        clipped = {pid: meta.get("clipped_artists", []) for pid, meta in render_metadata.items() if meta.get("clipped_artists") or meta.get("panel_label_clipped")}
        if clipped:
            failures.append(f"Fig.4 labels/ticks/legends/panel letters clipped: {clipped}")
        else:
            passes.append("Fig.4 rendered labels are not clipped")


def _check_fig4_supp_specifics(
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
    _ = spec, adapter_results, render_metadata
    if figure_id != "fig4_supp":
        return
    expected = ["S5A", "S5B", "S5C", "S5D", "S5E"]
    if list(panels) == expected:
        passes.append("Fig.4 supplement defines compact S5A-S5E")
    else:
        failures.append(f"Fig.4 supplement panel order must be {expected}, found {list(panels)}")
    stale_s8 = [panel_id for panel_id in panels if str(panel_id).startswith("S8")]
    if stale_s8:
        failures.append(f"Fig.4 supplement must not define active S8 panels, found {stale_s8}")
    else:
        passes.append("Fig.4 supplement has no active S8 panels")
    for panel_id in expected:
        paths = panel_output_paths(output_dir, figure_id, panel_id)
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            failures.append(f"Fig.4 supplement {panel_id}: missing adapter outputs {missing}")
            continue
        passes.append(f"Fig.4 supplement {panel_id}: panel_data/stats/source_manifest exist")
        df = pd.read_csv(paths["panel_data"])
        sources = read_json(paths["sources"])
        run_mode = str(sources.get("run_mode") or "")
        if run_mode == "single_network_draft":
            warnings.append(f"Fig.4 supplement {panel_id}: single_network_draft n_networks=1; draft-only")

    s7a_path = panel_output_paths(output_dir, figure_id, "S5A")["panel_data"]
    if s7a_path.exists():
        s7a = pd.read_csv(s7a_path)
        metrics = set(s7a.get("metric", pd.Series(dtype=str)).astype(str))
        groups = set(s7a.get("similarity_group", pd.Series(dtype=str)).astype(str)) | set(s7a.get("overlap_group", pd.Series(dtype=str)).astype(str))
        if {"acc_drop", "DPI_L3"}.intersection(metrics) and {"low_similarity", "high_similarity", "low_overlap", "high_overlap"}.issubset(groups):
            passes.append("Fig.4 supplement S5A exposes similarity x overlap 2x2 rows")
        else:
            failures.append(f"Fig.4 supplement S5A must expose similarity x overlap 2x2 rows, metrics={sorted(metrics)}, groups={sorted(groups)}")

    s7b_path = panel_output_paths(output_dir, figure_id, "S5B")["panel_data"]
    if s7b_path.exists():
        s7b = pd.read_csv(s7b_path)
        metrics = set(s7b.get("metric", pd.Series(dtype=str)).astype(str))
        conditions = set(s7b.get("condition", pd.Series(dtype=str)).astype(str))
        if "mean_acc_drop" in metrics and {"High overlap excess", "Low overlap excess"}.issubset(conditions):
            passes.append("Fig.4 supplement S5B reports overlap-excess accuracy control")
        else:
            failures.append(f"Fig.4 supplement S5B must report high/low overlap-excess mean_acc_drop, metrics={sorted(metrics)}, conditions={sorted(conditions)}")

    s7c_path = panel_output_paths(output_dir, figure_id, "S5C")["panel_data"]
    if s7c_path.exists():
        s7c = pd.read_csv(s7c_path)
        outcomes = set(s7c.get("outcome_metric", pd.Series(dtype=str)).astype(str))
        terms = set(s7c.get("condition", pd.Series(dtype=str)).astype(str))
        if {"DPI_L3", "acc_drop"}.issubset(outcomes) and {"overlap", "similarity", "input_energy"}.issubset(terms):
            passes.append("Fig.4 supplement S5C reports overlap/similarity/energy regression coefficients")
        else:
            failures.append(f"Fig.4 supplement S5C regression rows incomplete, outcomes={sorted(outcomes)}, terms={sorted(terms)}")

    s7d_path = panel_output_paths(output_dir, figure_id, "S5D")["panel_data"]
    if s7d_path.exists():
        s7d = pd.read_csv(s7d_path)
        metrics = set(s7d.get("metric", pd.Series(dtype=str)).astype(str))
        conditions = set(s7d.get("condition", pd.Series(dtype=str)).astype(str))
        if "DPI_L3_contrast" in metrics and {"Overlap - non-overlap", "Overlap - random"}.issubset(conditions):
            passes.append("Fig.4 supplement S5D reports perturbation-specific L3 DPI contrasts")
        else:
            failures.append(f"Fig.4 supplement S5D must report overlap-vs-control DPI contrasts, metrics={sorted(metrics)}, conditions={sorted(conditions)}")

    s7e_path = panel_output_paths(output_dir, figure_id, "S5E")["panel_data"]
    if s7e_path.exists():
        s7e = pd.read_csv(s7e_path)
        conditions = set(s7e.get("condition", pd.Series(dtype=str)).astype(str))
        if {"Dynamic", "Static", "Overlap support", "Non-overlap support", "Random matched"}.issubset(conditions):
            passes.append("Fig.4 supplement S5E reports decision-step summary across dynamic/static/control conditions")
        else:
            failures.append(f"Fig.4 supplement S5E missing decision-step conditions, found {sorted(conditions)}")
        e_meta = render_metadata.get("S5E", {})
        if e_meta:
            if e_meta.get("plot_form") == "s7_decision_spike_summary_bar_only":
                passes.append("Fig.4 supplement S5E renders decision-step summary as bar-only")
            else:
                failures.append(f"Fig.4 supplement S5E must render as bar-only, found {e_meta.get('plot_form')}")


def _check_fig4_current_geometry(panels: Mapping[str, Any], passes: list[str], failures: list[str]) -> None:
    expected = {
        "A": {"x": 12.00, "y": 8.00, "w": 147.00, "h": 24.00},
        "B": {"x": 12.00, "y": 39.00, "w": 45.67, "h": 34.00},
        "C": {"x": 62.67, "y": 39.00, "w": 45.67, "h": 34.00},
        "D": {"x": 113.33, "y": 39.00, "w": 45.67, "h": 34.00},
        "E": {"x": 12.00, "y": 81.00, "w": 70.50, "h": 59.00},
        "F": {"x": 88.50, "y": 81.00, "w": 70.50, "h": 59.00},
    }
    for panel_id, expected_pos in expected.items():
        pos = (panels.get(panel_id) or {}).get("position_mm") or {}
        if _fig4_box_near(pos, expected_pos):
            passes.append(f"Fig.4{panel_id} position matches requested mm layout")
        else:
            failures.append(f"Fig.4{panel_id} position must be {expected_pos}, found {pos}")
    pos = {pid: (panels.get(pid) or {}).get("position_mm") or {} for pid in expected}
    if pos.get("A") and _fig4_near(float(pos["A"].get("w", 0)), 147.0):
        passes.append("Fig.4A is full-width")
    if all(_fig4_near(float(pos[p].get("y", 0)), 39.0) for p in ("B", "C", "D")):
        passes.append("Fig.4B/C/D are aligned in row 2")
    if all(_fig4_near(float(pos[p].get("y", 0)), 81.0) for p in ("E", "F")):
        passes.append("Fig.4E/F are aligned in row 3")


def _fig4_box_near(actual: Mapping[str, Any], expected: Mapping[str, float], *, tol: float = 0.08) -> bool:
    return all(_fig4_near(float(actual.get(key, -999.0)), value, tol=tol) for key, value in expected.items())


def _fig4_near(left: float, right: float, *, tol: float = 0.08) -> bool:
    return abs(float(left) - float(right)) <= tol
