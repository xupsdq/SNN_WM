from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult, missing_adapter_result, summarize_values, write_adapter_outputs


DEFAULT_EXPERIMENT_ROOT = "results/paper_experiments/fig5_local_support_competition"
UNIT_LABELS = {
    "overlap_dominant": "Overlap-dominant",
    "probe_only_dominant": "Probe-only-dominant",
    "balanced": "Balanced",
    "random_matched": "Random matched",
}
MAIN_UNIT_LABELS = {
    "overlap_dominant": "Overlap",
    "probe_only_dominant": "Probe-only",
    "random_matched": "Random",
    "balanced": "Balanced",
}
CONDITION_LABELS = {
    "dynamic_intact": "Dynamic intact",
    "static_frozen": "Static frozen",
    "attenuate_overlap_high_support": "Attenuate overlap support",
    "reset_overlap_high_support": "Reset overlap support",
    "sham_perturbation": "Sham perturbation",
}
MAIN_CONDITION_LABELS = {
    "dynamic_intact": "Dynamic",
    "attenuate_overlap_high_support": "Attenuate",
    "reset_overlap_high_support": "Reset",
    "static_frozen": "Static",
    "sham_perturbation": "Sham",
}
TRANSITION_TYPE_LABELS = {
    "advance": "Advance",
    "recruit": "Recruit",
    "loss": "Loss",
}
MAIN_TRANSITION_COLUMNS = {
    "advance": "P_advance",
    "recruit": "P_recruit",
    "loss": "P_loss",
}


def build_fig5_preprobe_support_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds, warnings = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.5 experiment root has no seed directories.")
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_a_preprobe_support_metrics.csv"
        if not path.exists():
            warnings.append(f"Missing Fig.5A source: {path}")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        for r in df.itertuples(index=False):
            group = str(r.unit_group)
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "preprobe_support",
                    UNIT_LABELS.get(group, group),
                    str(getattr(r, "layer", "layer1")),
                    _network_id(seed_dir),
                    _seed_id(seed_dir, getattr(r, "network_seed", "")),
                    float(getattr(r, "mean_support", np.nan)),
                    "support",
                    path,
                    repo_root,
                    trial_id=int(getattr(r, "trial_id", -1)),
                    unit_group=group,
                    n_units=int(getattr(r, "n_units", 0)),
                    run_mode="",
                )
            )
    return _write_result(spec, repo_root, output_dir, panel_id, pd.DataFrame(rows), sources, warnings)


def build_fig5_early_firing_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_b_transition_summary_by_group.csv"
        if not path.exists():
            warnings.append(f"Missing Fig.5B source: {path}")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        for r in df.itertuples(index=False):
            group = str(r.unit_group)
            for metric in ("P_advance", "P_recruit", "P_advance_plus_recruit"):
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        metric,
                        UNIT_LABELS.get(group, group),
                        "layer1",
                        _network_id(seed_dir),
                        _seed_id(seed_dir, getattr(r, "network_seed", "")),
                        float(getattr(r, metric, np.nan)),
                        "probability",
                        path,
                        repo_root,
                        trial_id=int(getattr(r, "trial_id", -1)),
                        unit_group=group,
                        n_units=int(getattr(r, "n_units", 0)),
                        early_window_ms=int(getattr(r, "early_window_ms", 0)),
                        run_mode="",
                    )
                )
    return _write_result(spec, repo_root, output_dir, panel_id, pd.DataFrame(rows), sources, warnings)


def build_fig5_early_firing_headline_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds, warnings = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.5 experiment root has no seed directories.")
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    source_records: list[dict[str, Any]] = []
    main_groups = ["overlap_dominant", "probe_only_dominant", "random_matched"]
    required_cols = {"unit_group", *MAIN_TRANSITION_COLUMNS.values()}
    for seed_dir in seeds:
        candidates = [
            seed_dir / "data" / "metrics" / "panel_b_transition_summary_by_group.csv",
            seed_dir / "data" / "metrics" / "panel_b_early_firing_transition_metrics.csv",
        ]
        source_records.extend(_source_entry(path, repo_root) for path in candidates)
        path: Path | None = None
        df = pd.DataFrame()
        for candidate in candidates:
            if not candidate.exists():
                continue
            candidate_df = pd.read_csv(candidate)
            missing = sorted(required_cols.difference(candidate_df.columns))
            if missing:
                warnings.append(f"{_rel(candidate, repo_root)} lacks headline columns {missing}.")
                continue
            path = candidate
            df = candidate_df[candidate_df["unit_group"].astype(str).isin(main_groups)].copy()
            break
        if path is None:
            warnings.append(f"Missing Fig.5B headline source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        rows.extend(
            _transition_composition_longform(
                df,
                figure_id=figure_id,
                panel_id=panel_id,
                seed_dir=seed_dir,
                source_file=path,
                repo_root=repo_root,
                unit_groups=main_groups,
            )
        )
    panel_df = pd.DataFrame(rows)
    extra_stats = {
        "primary_plot_type": "stacked_transition_composition",
        "plotted_transition_types": list(TRANSITION_TYPE_LABELS),
        "excluded_unit_groups": ["balanced"],
        "point_overlay_enabled": False,
        "error_bar_enabled": _has_total_mass_replicates(panel_df, ["unit_group"]) if not panel_df.empty else False,
        "available_unit_groups": sorted(set(panel_df.get("unit_group", pd.Series(dtype=str)).dropna().astype(str))) if not panel_df.empty else [],
        "available_early_windows": sorted(set(pd.to_numeric(panel_df.get("early_window_ms", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())) if not panel_df.empty and "early_window_ms" in panel_df.columns else [],
        "total_transition_mass": _total_mass_summary(panel_df, ["unit_group"]) if not panel_df.empty else [],
    }
    extra_manifest = {
        "primary_plot_type": "stacked_transition_composition",
        "plotted_transition_types": list(TRANSITION_TYPE_LABELS),
        "excluded_unit_groups": ["balanced"],
        "point_overlay_enabled": False,
        "error_bar_enabled": bool(extra_stats["error_bar_enabled"]),
        "checked_candidates": [record["path"] for record in source_records],
    }
    return _write_result(
        spec,
        repo_root,
        output_dir,
        panel_id,
        panel_df,
        sources,
        warnings,
        group_cols=["condition", "unit_group", "transition_type"],
        extra_stats=extra_stats,
        extra_manifest=extra_manifest,
        source_records=source_records,
    )


def build_fig5_winner_loser_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    baseline_corrected = True
    for seed_dir in seeds:
        trace_path = seed_dir / "data" / "metrics" / "panel_c_event_trace_summary.csv"
        event_path = seed_dir / "data" / "metrics" / "panel_c_winner_loser_event_metrics.csv"
        if not trace_path.exists():
            warnings.append(f"Missing Fig.5C trace source: {trace_path}")
            continue
        sources.append(trace_path)
        if event_path.exists():
            sources.append(event_path)
        df = pd.read_csv(trace_path)
        df = _baseline_correct_trace(df, value_col="mean_value", group_cols=["trace_type"], time_col="time_ms")
        for r in df.itertuples(index=False):
            trace_type = str(getattr(r, "trace_type", ""))
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    trace_type,
                    _trace_label(trace_type),
                    "layer1",
                    _network_id(seed_dir),
                    _seed_id(seed_dir, getattr(r, "network_seed", "")),
                    float(getattr(r, "mean_value", np.nan)),
                    "delta",
                    trace_path,
                    repo_root,
                    time_ms=float(getattr(r, "time_ms", np.nan)),
                    trace_type=trace_type,
                    n_events=int(getattr(r, "n_events", 0)),
                    sem_value=float(getattr(r, "sem_value", 0.0)),
                    baseline_corrected=baseline_corrected,
                    inhibition_trace_definition="loser_unit_received_inhibition" if trace_type == "loser_inhibition" else "",
                    run_mode="",
                )
            )
    extra_stats = {
        "inhibition_trace_definition": "loser_unit_received_inhibition",
        "baseline_corrected": baseline_corrected,
        "baseline_window": "time_ms < 0",
        "trace_types": ["winner_delta_v", "loser_delta_v", "loser_inhibition"],
    }
    extra_manifest = {
        "inhibition_trace_definition": "loser_unit_received_inhibition",
        "baseline_corrected": baseline_corrected,
        "baseline_window": "time_ms < 0",
        "trace_types": ["winner_delta_v", "loser_delta_v", "loser_inhibition"],
        "inhibition_trace_note": "inhibition trace is measured at the same selected loser unit",
        "point_overlay_enabled": False,
    }
    return _write_result(
        spec,
        repo_root,
        output_dir,
        panel_id,
        pd.DataFrame(rows),
        sources,
        warnings,
        group_cols=["metric", "time_ms"],
        extra_stats=extra_stats,
        extra_manifest=extra_manifest,
    )


def build_fig5_support_perturbation_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    metric_map = {
        "early_recruitment": "P_advance_plus_recruit",
        "loser_inhibition": "loser_post_winner_inh_rise",
        "spike_similarity": "dynamic_like_spike_similarity",
        "decision_deflection": "decision_deflection_score",
    }
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_d_support_perturbation_node_metrics.csv"
        if not path.exists():
            warnings.append(f"Missing Fig.5D source: {path}")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        for r in df.itertuples(index=False):
            condition = str(r.condition)
            for node, col in metric_map.items():
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        node,
                        CONDITION_LABELS.get(condition, condition),
                        "layer1",
                        _network_id(seed_dir),
                        _seed_id(seed_dir, getattr(r, "network_seed", "")),
                        float(getattr(r, col, np.nan)),
                        "a.u.",
                        path,
                        repo_root,
                        trial_id=int(getattr(r, "trial_id", -1)),
                        perturbation_condition=condition,
                        perturbed_unit_group=str(getattr(r, "perturbed_unit_group", "")),
                        n_units=int(getattr(r, "n_perturbed_units", 0)),
                        node=node,
                        run_mode="",
                    )
                )
    return _write_result(spec, repo_root, output_dir, panel_id, pd.DataFrame(rows), sources, warnings, group_cols=["metric", "condition"])


def build_fig5_causal_perturbation_summary_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds, warnings = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.5 experiment root has no seed directories.")
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    source_records: list[dict[str, Any]] = []
    plotted_groups = ["overlap_dominant", "probe_only_dominant"]
    plotted_conditions = [
        "dynamic_intact",
        "attenuate_overlap_high_support",
        "reset_overlap_high_support",
        "static_frozen",
    ]
    source_name = "panel_d_perturbation_transition_summary_by_group.csv"
    required_cols = {"condition", "unit_group", *MAIN_TRANSITION_COLUMNS.values()}
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / source_name
        source_records.append(_source_entry(path, repo_root))
        if not path.exists():
            warnings.append(f"Missing Fig.5D causal transition source: {path}")
            continue
        df = pd.read_csv(path)
        missing = sorted(required_cols.difference(df.columns))
        if missing:
            warnings.append(f"{_rel(path, repo_root)} lacks causal transition columns {missing}.")
            continue
        df = df[
            df["unit_group"].astype(str).isin(plotted_groups)
            & df["condition"].astype(str).isin(plotted_conditions)
        ].copy()
        sources.append(path)
        rows.extend(
            _transition_composition_longform(
                df,
                figure_id=figure_id,
                panel_id=panel_id,
                seed_dir=seed_dir,
                source_file=path,
                repo_root=repo_root,
                unit_groups=plotted_groups,
                perturbation_conditions=plotted_conditions,
            )
        )
    panel_df = pd.DataFrame(rows)
    available_conditions = sorted(set(panel_df.get("perturbation_condition", pd.Series(dtype=str)).dropna().astype(str))) if not panel_df.empty else []
    available_groups = sorted(set(panel_df.get("unit_group", pd.Series(dtype=str)).dropna().astype(str))) if not panel_df.empty else []
    missing_plotted_conditions = [condition for condition in plotted_conditions if condition not in set(available_conditions)]
    extra_stats = {
        "primary_plot_type": "grouped_stacked_transition_composition",
        "source_file": source_name,
        "source_level": "transition_summary_by_group" if sources else "missing",
        "plotted_transition_types": list(TRANSITION_TYPE_LABELS),
        "plotted_unit_groups": plotted_groups,
        "plotted_conditions": plotted_conditions,
        "excluded_conditions": ["sham_perturbation"],
        "excluded_unit_groups": ["random_matched", "balanced"],
        "point_overlay_enabled": False,
        "error_bar_enabled": _has_total_mass_replicates(panel_df, ["unit_group", "perturbation_condition"]) if not panel_df.empty else False,
        "available_conditions": available_conditions,
        "missing_plotted_conditions": missing_plotted_conditions,
        "available_unit_groups": available_groups,
        "total_transition_mass": _total_mass_summary(panel_df, ["unit_group", "perturbation_condition"]) if not panel_df.empty else [],
    }
    extra_manifest = {
        "primary_plot_type": "grouped_stacked_transition_composition",
        "source_file": source_name,
        "source_level": "transition_summary_by_group" if sources else "missing",
        "checked_candidates": [record["path"] for record in source_records],
        "plotted_transition_types": list(TRANSITION_TYPE_LABELS),
        "plotted_unit_groups": plotted_groups,
        "plotted_conditions": plotted_conditions,
        "excluded_conditions": ["sham_perturbation"],
        "excluded_unit_groups": ["random_matched", "balanced"],
        "point_overlay_enabled": False,
        "error_bar_enabled": bool(extra_stats["error_bar_enabled"]),
        "available_conditions": available_conditions,
        "missing_plotted_conditions": missing_plotted_conditions,
        "available_unit_groups": available_groups,
        "intervention_timing": "pre_probe_boundary",
        "probe_input_changed": False,
        "perturbed_unit_scope": "overlap_high_support",
    }
    return _write_result(
        spec,
        repo_root,
        output_dir,
        panel_id,
        panel_df,
        sources,
        warnings,
        group_cols=["condition", "unit_group", "perturbation_condition", "transition_type"],
        extra_stats=extra_stats,
        extra_manifest=extra_manifest,
        source_records=source_records,
    )


def build_fig5_perturbation_transition_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    summary_metrics = (
        "P_advance",
        "P_recruit",
        "P_loss",
        "P_unchanged",
        "P_advance_plus_recruit",
        "P_same_winner_lost_or_delayed",
    )
    for seed_dir in seeds:
        summary_path = seed_dir / "data" / "metrics" / "panel_d_perturbation_transition_summary_by_group.csv"
        contrast_path = seed_dir / "data" / "metrics" / "panel_d_perturbation_transition_contrast.csv"
        unit_path = seed_dir / "data" / "metrics" / "panel_d_perturbation_unit_transitions.csv"
        if not summary_path.exists():
            warnings.append(f"Missing Fig.5D transition source: {summary_path}")
            continue
        sources.append(summary_path)
        if contrast_path.exists():
            sources.append(contrast_path)
        if unit_path.exists():
            sources.append(unit_path)
        df = pd.read_csv(summary_path)
        for r in df.itertuples(index=False):
            condition = str(r.condition)
            group = str(r.unit_group)
            for metric in summary_metrics:
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        metric,
                        CONDITION_LABELS.get(condition, condition),
                        "layer1",
                        _network_id(seed_dir),
                        _seed_id(seed_dir, getattr(r, "network_seed", "")),
                        float(getattr(r, metric, np.nan)),
                        "probability",
                        summary_path,
                        repo_root,
                        trial_id=int(getattr(r, "trial_id", -1)),
                        unit_group=group,
                        unit_group_label=UNIT_LABELS.get(group, group),
                        perturbation_condition=condition,
                        n_units=int(getattr(r, "n_units", 0)),
                        n_same_winner_units=int(getattr(r, "n_same_winner_units", 0)),
                        reference_condition="static_frozen",
                        transition_reference="static_frozen",
                        run_mode="",
                    )
                )
        if contrast_path.exists():
            contrast = pd.read_csv(contrast_path)
            for r in contrast.itertuples(index=False):
                group = str(r.unit_group)
                for metric in (
                    "attenuate_delta_P_advance_plus_recruit",
                    "reset_delta_P_advance_plus_recruit",
                    "attenuate_delta_P_loss",
                    "reset_delta_P_loss",
                    "attenuate_delta_P_same_winner_lost_or_delayed",
                    "reset_delta_P_same_winner_lost_or_delayed",
                ):
                    rows.append(
                        _row(
                            figure_id,
                            panel_id,
                            metric,
                            UNIT_LABELS.get(group, group),
                            "layer1",
                            _network_id(seed_dir),
                            _seed_id(seed_dir, getattr(r, "network_seed", "")),
                            float(getattr(r, metric, np.nan)),
                            "delta_probability",
                            contrast_path,
                            repo_root,
                            trial_id=int(getattr(r, "trial_id", -1)),
                            unit_group=group,
                            unit_group_label=UNIT_LABELS.get(group, group),
                            reference_condition="dynamic_intact",
                            transition_reference="static_frozen",
                            run_mode="",
                        )
                    )
    return _write_result(spec, repo_root, output_dir, panel_id, pd.DataFrame(rows), sources, warnings, group_cols=["metric", "condition", "unit_group_label"])


def build_fig5_perturbation_summary_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_e_perturbation_effect_summary.csv"
        if not path.exists():
            warnings.append(f"Missing Fig.5E source: {path}")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        for r in df.itertuples(index=False):
            node = str(r.node)
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "normalized_overlap_disruption",
                    _node_label(node),
                    "layer1",
                    _network_id(seed_dir),
                    _seed_id(seed_dir, getattr(r, "network_seed", "")),
                    float(getattr(r, "normalized_overlap_disruption", np.nan)),
                    "normalized",
                    path,
                    repo_root,
                    node=node,
                    x_value=node,
                    y_value=float(getattr(r, "normalized_overlap_disruption", np.nan)),
                    run_mode="",
                )
            )
    return _write_result(spec, repo_root, output_dir, panel_id, pd.DataFrame(rows), sources, warnings)


def _write_result(
    spec: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    panel_id: str,
    panel_df: pd.DataFrame,
    source_paths: list[Path],
    warnings: list[str],
    *,
    group_cols: list[str] | None = None,
    extra_stats: Mapping[str, Any] | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
    source_records: list[dict[str, Any]] | None = None,
) -> AdapterResult:
    figure_id = str(spec.get("figure_id", "fig5"))
    seed_dirs, _ = _seed_dirs(spec, repo_root)
    run_mode = "multi_network_final" if len(seed_dirs) > 1 else "single_network_draft"
    n_networks = len(seed_dirs)
    if not panel_df.empty:
        panel_df = panel_df.copy()
        panel_df["run_mode"] = run_mode
    if run_mode == "single_network_draft":
        warnings.append("Single-network result. Use for pipeline validation only, not final manuscript statistics.")
    if panel_df.empty:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.5{panel_id} adapter found no plottable rows.")
    group_cols = group_cols or [c for c in ("metric", "condition", "unit_group", "node") if c in panel_df.columns]
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "run_mode": run_mode,
        "n_networks": int(n_networks),
        "summaries": summarize_values(panel_df, group_cols),
        "values_used_for_plotting": _values(panel_df),
        "warning": "Single-network result. Use for pipeline validation only, not final manuscript statistics." if run_mode == "single_network_draft" else "",
    }
    stats.update(dict(extra_stats or {}))
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok",
        "run_mode": run_mode,
        "n_networks": int(n_networks),
        "source_files_used": [_rel(path, repo_root) for path in source_paths],
        "sources": [{"path": _rel(path, repo_root), "exists": path.exists()} for path in source_paths],
        "experiment_root": str(spec.get("experiment_root") or DEFAULT_EXPERIMENT_ROOT),
        "conditions": sorted(set(map(str, panel_df.get("perturbation_condition", panel_df.get("condition", pd.Series(dtype=str))).dropna().unique()))),
        "unit_groups": sorted(set(map(str, panel_df.get("unit_group", pd.Series(dtype=str)).dropna().unique()))),
        "intervention_timing": "pre_probe_boundary",
        "probe_input_changed": False,
        "supplement_files": _supplement_status(seed_dirs, repo_root),
    }
    if source_records is not None:
        manifest["sources"] = source_records
    manifest.update(dict(extra_manifest or {}))
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def _transition_composition_longform(
    df: pd.DataFrame,
    *,
    figure_id: str,
    panel_id: str,
    seed_dir: Path,
    source_file: Path,
    repo_root: Path,
    unit_groups: list[str],
    perturbation_conditions: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    use = df.copy()
    use["unit_group"] = use["unit_group"].astype(str)
    use = use[use["unit_group"].isin(unit_groups)]
    if perturbation_conditions is not None and "condition" in use.columns:
        use["condition"] = use["condition"].astype(str)
        use = use[use["condition"].isin(perturbation_conditions)]
    for _, r in use.iterrows():
        group = str(r.get("unit_group", ""))
        raw_condition = str(r.get("condition", "")) if perturbation_conditions is not None else ""
        group_label = MAIN_UNIT_LABELS.get(group, UNIT_LABELS.get(group, group))
        condition_label = MAIN_CONDITION_LABELS.get(raw_condition, raw_condition) if raw_condition else group_label
        total_mass = sum(_num(r.get(col)) for col in MAIN_TRANSITION_COLUMNS.values())
        for transition_type, source_col in MAIN_TRANSITION_COLUMNS.items():
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "transition_fraction",
                    condition_label,
                    "layer1",
                    _network_id(seed_dir),
                    _seed_id(seed_dir, r.get("network_seed", "")),
                    _num(r.get(source_col)),
                    "proportion",
                    source_file,
                    repo_root,
                    trial_id=r.get("trial_id", ""),
                    unit_group=group,
                    unit_group_label=group_label,
                    transition_type=transition_type,
                    transition_label=TRANSITION_TYPE_LABELS.get(transition_type, transition_type),
                    source_metric=source_col,
                    total_transition_mass=total_mass,
                    condition_label=condition_label,
                    perturbation_condition=raw_condition,
                    raw_condition=raw_condition,
                    n_units=r.get("n_units", ""),
                    early_window_ms=r.get("early_window_ms", ""),
                    n_same_winner_units=r.get("n_same_winner_units", ""),
                    run_mode="",
                )
            )
    return rows


def _baseline_correct_trace(df: pd.DataFrame, *, value_col: str, group_cols: list[str], time_col: str) -> pd.DataFrame:
    if df.empty or value_col not in df.columns or time_col not in df.columns:
        return df
    out = df.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    pre_event = out[out[time_col] < 0]
    if pre_event.empty:
        return out
    baselines = pre_event.groupby(group_cols, dropna=False)[value_col].mean().rename("_pre_event_mean").reset_index()
    out = out.merge(baselines, on=group_cols, how="left")
    out[value_col] = out[value_col] - out["_pre_event_mean"].fillna(0.0)
    return out.drop(columns=["_pre_event_mean"])


def _total_mass_summary(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    if df.empty or "total_transition_mass" not in df.columns:
        return []
    id_cols = [col for col in ("network_id", "seed_id", "trial_id") if col in df.columns]
    keep_cols = [col for col in group_cols + id_cols + ["total_transition_mass"] if col in df.columns]
    unique = df[keep_cols].drop_duplicates()
    out: list[dict[str, Any]] = []
    for keys, part in unique.groupby([col for col in group_cols if col in unique.columns], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(part["total_transition_mass"], errors="coerce").dropna()
        if values.empty:
            continue
        row = {col: str(value) for col, value in zip(group_cols, keys)}
        row.update(
            {
                "mean": float(values.mean()),
                "sem": float(values.sem()) if len(values) > 1 else 0.0,
                "n": int(values.count()),
            }
        )
        out.append(row)
    return out


def _has_total_mass_replicates(df: pd.DataFrame, group_cols: list[str]) -> bool:
    return any(item.get("n", 0) > 1 and float(item.get("sem", 0.0)) > 0 for item in _total_mass_summary(df, group_cols))


def _seed_dirs(spec: Mapping[str, Any], repo_root: Path) -> tuple[list[Path], list[str]]:
    root = Path(str(spec.get("experiment_root") or DEFAULT_EXPERIMENT_ROOT))
    if not root.is_absolute():
        root = repo_root / root
    warnings: list[str] = []
    if root.name.startswith("seed_"):
        return ([root] if root.exists() else []), warnings
    if not root.exists():
        warnings.append(f"Experiment root does not exist: {root}")
        return [], warnings
    seeds = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not seeds and (root / "summary.json").exists():
        seeds = [root]
    return seeds, warnings


def _supplement_status(seed_dirs: list[Path], repo_root: Path) -> list[dict[str, Any]]:
    names = [
        "supp_event_chain_fraction_metrics.csv",
        "supp_event_chain_null_baselines.csv",
        "supp_event_selection_audit.csv",
        "supp_early_window_robustness.csv",
        "supp_neighborhood_radius_robustness.csv",
        "supp_perturbation_ux_audit.csv",
        "supp_layer_delay_local_competition_metrics.csv",
        "supp_trial_condition_audit.csv",
    ]
    rows = []
    for seed_dir in seed_dirs:
        for name in names:
            path = seed_dir / "data" / "metrics" / name
            rows.append({"path": _rel(path, repo_root), "exists": path.exists()})
    return rows


def _row(
    figure_id: str,
    panel_id: str,
    metric: str,
    condition: str,
    layer: str,
    network_id: Any,
    seed_id: Any,
    value: float,
    unit: str,
    source_file: Path,
    repo_root: Path,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "condition": condition,
        "layer": layer,
        "network_id": network_id,
        "seed_id": seed_id,
        "value": value,
        "unit": unit,
        "source_file": _rel(source_file, repo_root),
    }
    row.update(extra)
    return row


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig5")), str(spec.get("panel_id", "")).upper()


def _network_id(seed_dir: Path) -> str:
    return seed_dir.name


def _seed_id(seed_dir: Path, fallback: Any) -> str:
    if seed_dir.name.startswith("seed_"):
        return seed_dir.name.replace("seed_", "")
    return str(fallback)


def _trace_label(trace_type: str) -> str:
    return {
        "winner_delta_v": "Winner ΔV",
        "loser_delta_v": "Loser ΔV",
        "loser_inhibition": "Inhibition received by loser",
    }.get(trace_type, trace_type)


def _node_label(node: str) -> str:
    return {
        "preprobe_support": "Pre-probe support",
        "early_recruitment": "Early recruitment",
        "winner_voltage_advantage": "Winner voltage",
        "loser_inhibition": "Loser inhibition",
        "spike_pattern_displacement": "Spike pattern",
        "decision_deflection": "Decision deflection",
    }.get(node, node.replace("_", " "))


def _effect_metric_to_node(metric: str) -> str:
    return {
        "P_advance_plus_recruit": "early_recruitment",
        "P_loss": "loser_inhibition",
        "P_same_winner_lost_or_delayed": "loser_inhibition",
        "dynamic_like_spike_similarity": "spike_similarity",
        "dynamic_like_readout_recovery": "spike_similarity",
        "decision_deflection_score": "decision_deflection",
        "early_recruitment": "early_recruitment",
        "loser_inhibition": "loser_inhibition",
        "spike_similarity": "spike_similarity",
        "decision_deflection": "decision_deflection",
    }.get(metric, "")


def _source_entry(path: Path, repo_root: Path) -> dict[str, Any]:
    return {"path": _rel(path, repo_root), "exists": path.exists()}


def _num(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float("nan") if pd.isna(numeric) else float(numeric)


def _auxiliary_means(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    for (metric, group), part in df.groupby(["metric", "unit_group"], dropna=False):
        values = pd.to_numeric(part["value"], errors="coerce").dropna()
        if values.empty:
            continue
        out.append(
            {
                "metric": str(metric),
                "unit_group": str(group),
                "mean": float(values.mean()),
                "sem": float(values.sem()) if len(values) > 1 else 0.0,
                "n": int(values.count()),
            }
        )
    return out


def _condition_delta(df: pd.DataFrame, condition: str, reference: str) -> dict[str, float]:
    if df.empty or "node" not in df.columns or "raw_condition" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for node, part in df.groupby("node", dropna=False):
        values = pd.to_numeric(part.loc[part["raw_condition"].astype(str).eq(condition), "value"], errors="coerce").dropna()
        refs = pd.to_numeric(part.loc[part["raw_condition"].astype(str).eq(reference), "value"], errors="coerce").dropna()
        if values.empty or refs.empty:
            continue
        out[str(node)] = float(values.mean() - refs.mean())
    return out


def _values(df: pd.DataFrame) -> list[float]:
    values = pd.to_numeric(df.get("value", pd.Series(dtype=float)), errors="coerce").dropna()
    return [float(v) for v in values.tolist()]


def _rel(path: Path | str, root: Path) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path_obj).replace("\\", "/")
