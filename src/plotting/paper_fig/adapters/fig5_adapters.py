from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.plotting.paper_fig.data_resolver import AdapterResult, missing_adapter_result, summarize_values, write_adapter_outputs
from src.plotting.paper_fig.utils import read_json


DEFAULT_EXPERIMENT_ROOT = (
    "results/paper_figure_multi_seed/fig5_local_support_competition"
)
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
    "dynamic_intact": "Dynamic",
    "static_frozen": "Static frozen",
    "attenuate_l1_stsp": "Attenuate L1 STSP",
    "reset_l1_stsp": "Reset L1 STSP",
    "attenuate_all_stsp": "Attenuate STSP",
    "reset_all_stsp": "Reset STSP",
    "attenuate_overlap_high_support": "Attenuate overlap support",
    "reset_overlap_high_support": "Reset overlap support",
    "sham_perturbation": "Sham perturbation",
}
MAIN_CONDITION_LABELS = {
    "dynamic_intact": "Dynamic",
    "attenuate_l1_stsp": "Attenuate L1 STSP",
    "reset_l1_stsp": "Reset L1 STSP",
    "attenuate_all_stsp": "Attenuate STSP",
    "reset_all_stsp": "Reset STSP",
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
FIG5D_CAUSAL_ANALYSIS_WINDOW_MS = 50.0
FIG5D_CAUSAL_WINDOW_SOURCE_NAME = "panel_d_l1_stsp_perturbation_unit_transitions.csv"
FIG5E_HEADLINE_METRIC = "P_advance_or_recruit_dynamic_minus_condition"
FIG5E_HEADLINE_REFERENCE_CONDITION = "dynamic_intact"
FIG5E_HEADLINE_COMPARISON_CONDITIONS = ("attenuate_l1_stsp", "reset_l1_stsp")
FIG5_L2_WRITEBACK_SOURCE_NAME = "panel_postprobe_l2_reupdate_history_composition.csv"
FIG5_L2_WRITEBACK_LEGACY_SOURCE_NAME = "panel_postprobe_l2_stsp_writeback_summary.csv"


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
        trial_path = seed_dir / "data" / "metrics" / "panel_c_winner_loser_trial_summary.csv"
        network_path = seed_dir / "data" / "metrics" / "panel_c_winner_loser_network_summary.csv"
        if not trace_path.exists():
            warnings.append(f"Missing Fig.5C trace source: {trace_path}")
            continue
        sources.append(trace_path)
        for path in (event_path, trial_path, network_path):
            if path.exists():
                sources.append(path)
            else:
                warnings.append(f"Missing corrected Fig.5C source: {path}")
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
        "analysis_scope": "selected_winner_loser_events",
        "primary_statistical_metric": "winner_minus_loser_full_pre_delta_v_mean",
        "primary_window_ms": [-8.0, -1.0],
        "descriptive_window_ms": [-4.0, -1.0],
        "aggregation": "event_to_trial_to_network",
        "claim_boundary": "plotted baseline-corrected traces are descriptive; inference uses the separate network-level winner-minus-loser full-pre contrast",
        "inference_unit": "independently_trained_network",
        "confidence_interval": "two-sided t-based 95% CI across networks",
        "excluded_misinterpretation": "winner_pre_spike_boost is an event proportion, not a voltage increase",
    }
    extra_manifest = {
        "inhibition_trace_definition": "loser_unit_received_inhibition",
        "baseline_corrected": baseline_corrected,
        "baseline_window": "time_ms < 0",
        "trace_types": ["winner_delta_v", "loser_delta_v", "loser_inhibition"],
        "inhibition_trace_note": "inhibition trace is measured at the same selected loser unit",
        "analysis_scope": "selected_winner_loser_events",
        "primary_statistical_metric": "winner_minus_loser_full_pre_delta_v_mean",
        "primary_window_ms": [-8.0, -1.0],
        "descriptive_window_ms": [-4.0, -1.0],
        "aggregation": "event_to_trial_to_network",
        "claim_boundary": "descriptive trace display plus a predeclared network-level winner-minus-loser scalar endpoint",
        "excluded_misinterpretation": "winner_pre_spike_boost is an event proportion, not a voltage increase",
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
    plotted_conditions = [FIG5E_HEADLINE_REFERENCE_CONDITION, *FIG5E_HEADLINE_COMPARISON_CONDITIONS]
    analysis_window_ms = float(spec.get("analysis_window_ms", FIG5D_CAUSAL_ANALYSIS_WINDOW_MS))
    unit_source_name = FIG5D_CAUSAL_WINDOW_SOURCE_NAME
    unit_required_cols = {
        "network_seed",
        "trial_id",
        "condition",
        "condition_label",
        "unit_group",
        "included_in_main",
        "first_spike_static",
        "first_spike_condition",
        "perturbation_mode",
        "perturbed_layer",
        "perturbed_variables",
    }
    for seed_dir in seeds:
        unit_path = seed_dir / "data" / "metrics" / unit_source_name
        source_records.append(_source_entry(unit_path, repo_root))
        if not unit_path.exists():
            warnings.append(f"Missing Fig.5E canonical 50-ms unit-transition source: {_rel(unit_path, repo_root)}")
            continue
        unit_cols = set(pd.read_csv(unit_path, nrows=0).columns)
        unit_missing = sorted(unit_required_cols.difference(unit_cols))
        if unit_missing:
            warnings.append(f"{_rel(unit_path, repo_root)} lacks canonical Layer1 STSP unit-transition columns {unit_missing}.")
            continue
        unit_df = pd.read_csv(unit_path, usecols=[col for col in unit_required_cols if col in unit_cols])
        window_steps = _analysis_window_steps(seed_dir, analysis_window_ms)
        df = _l1_stsp_windowed_summary_from_unit_transitions(
            unit_df,
            plotted_conditions=plotted_conditions,
            analysis_window_ms=analysis_window_ms,
            analysis_window_steps=window_steps,
        )
        if df.empty:
            warnings.append(f"{_rel(unit_path, repo_root)} produced no canonical 50-ms Layer1 STSP transitions.")
            continue
        sources.append(unit_path)
        rows.extend(
            _l1_stsp_transition_longform(
                df,
                figure_id,
                panel_id,
                seed_dir,
                unit_path,
                repo_root,
                legacy_global_used=False,
                legacy_region_used=False,
            )
        )
    transition_df = pd.DataFrame(rows)
    panel_df, headline_warnings = _l1_stsp_advance_or_recruit_headline_rows(
        transition_df,
        figure_id=figure_id,
        panel_id=panel_id,
        repo_root=repo_root,
        analysis_window_ms=analysis_window_ms,
    )
    warnings.extend(headline_warnings)
    expected_network_ids = {_network_id(seed_dir) for seed_dir in seeds}
    observed_network_ids = set(panel_df.get("network_id", pd.Series(dtype=str)).dropna().astype(str)) if not panel_df.empty else set()
    if observed_network_ids != expected_network_ids:
        missing_network_ids = sorted(expected_network_ids.difference(observed_network_ids))
        return missing_adapter_result(
            spec,
            repo_root,
            output_dir,
            "Fig.5E headline endpoint requires paired 50-ms advance-or-recruit values for every canonical network; "
            f"missing={missing_network_ids or 'none'}, unexpected={sorted(observed_network_ids.difference(expected_network_ids)) or 'none'}.",
        )
    incomplete_contrasts = {
        comparison: sorted(
            expected_network_ids.difference(
                set(
                    panel_df.loc[
                        panel_df.get("comparison_condition", pd.Series(dtype=str)).astype(str).eq(comparison),
                        "network_id",
                    ].dropna().astype(str)
                )
            )
        )
        for comparison in FIG5E_HEADLINE_COMPARISON_CONDITIONS
    }
    incomplete_contrasts = {comparison: missing for comparison, missing in incomplete_contrasts.items() if missing}
    if incomplete_contrasts:
        return missing_adapter_result(
            spec,
            repo_root,
            output_dir,
            f"Fig.5E headline endpoint requires 20 paired networks for each comparison; missing={incomplete_contrasts}.",
        )
    contrast_summaries = _paired_t95_summaries(panel_df, group_cols=["condition", "comparison_condition"])
    available_conditions = sorted(set(panel_df.get("comparison_condition", pd.Series(dtype=str)).dropna().astype(str))) if not panel_df.empty else []
    missing_plotted_conditions = [condition for condition in FIG5E_HEADLINE_COMPARISON_CONDITIONS if condition not in set(available_conditions)]
    extra_stats = {
        "main_metric": FIG5E_HEADLINE_METRIC,
        "plot_type": "paired_network_contrast_dotplot",
        "primary_plot_type": "paired_network_contrast_dotplot",
        "headline_endpoint": "P(advance OR recruit | dynamic_intact, first 50 ms) minus P(advance OR recruit | comparison condition, first 50 ms)",
        "headline_formula": "P_advance + P_recruit; loss is excluded",
        "source_file": unit_source_name,
        "source_level": "l1_stsp_perturbation_unit_transition_windowed",
        "analysis_window_ms": analysis_window_ms,
        "analysis_window_policy": "first_spikes_at_or_after_window_are_treated_as_no_spike",
        "reference_condition": FIG5E_HEADLINE_REFERENCE_CONDITION,
        "comparison_conditions": list(FIG5E_HEADLINE_COMPARISON_CONDITIONS),
        "transition_types_included": ["advance", "recruit"],
        "transition_types_excluded_from_headline": ["loss"],
        "included_unit_groups": _split_unique(transition_df.get("included_unit_groups", pd.Series(dtype=str))) if not transition_df.empty else [],
        "perturbed_layer": "layer1",
        "perturbed_variables": ["u_pre", "x_pre"],
        "legacy_global_perturbation_used": False,
        "legacy_region_perturbation_used": False,
        "point_overlay_enabled": True,
        "point_overlay_count": int(len(panel_df)),
        "inference_unit": "independently_trained_network",
        "confidence_interval": "two-sided paired t-based 95% CI",
        "contrast_summaries": contrast_summaries,
        "neutral_reset_restore_policy": False,
        "boundary_policy": "restore_preprobe_boundary",
        "error_bar_enabled": bool(contrast_summaries),
        "available_conditions": available_conditions,
        "missing_plotted_conditions": missing_plotted_conditions,
    }
    extra_manifest = {
        "primary_plot_type": "paired_network_contrast_dotplot",
        "main_metric": FIG5E_HEADLINE_METRIC,
        "headline_endpoint": extra_stats["headline_endpoint"],
        "headline_formula": extra_stats["headline_formula"],
        "source_file": unit_source_name,
        "source_level": "l1_stsp_perturbation_unit_transition_windowed",
        "analysis_window_ms": analysis_window_ms,
        "analysis_window_policy": "first_spikes_at_or_after_window_are_treated_as_no_spike",
        "checked_candidates": [record["path"] for record in source_records],
        "reference_condition": FIG5E_HEADLINE_REFERENCE_CONDITION,
        "comparison_conditions": list(FIG5E_HEADLINE_COMPARISON_CONDITIONS),
        "transition_types_included": ["advance", "recruit"],
        "transition_types_excluded_from_headline": ["loss"],
        "included_unit_groups": extra_stats["included_unit_groups"],
        "perturbed_layer": "layer1",
        "perturbed_variables": ["u_pre", "x_pre"],
        "legacy_global_perturbation_used": False,
        "legacy_region_perturbation_used": False,
        "point_overlay_enabled": True,
        "point_overlay_count": int(len(panel_df)),
        "inference_unit": "independently_trained_network",
        "confidence_interval": "two-sided paired t-based 95% CI",
        "error_bar_enabled": bool(extra_stats["error_bar_enabled"]),
        "available_conditions": available_conditions,
        "missing_plotted_conditions": missing_plotted_conditions,
        "intervention_timing": "pre_probe_boundary",
        "boundary_policy": "restore_preprobe_boundary",
        "neutral_reset_restore_policy": False,
        "probe_input_changed": False,
        "perturbed_unit_scope": "all_layer1_stsp_sites",
    }
    return _write_result(
        spec,
        repo_root,
        output_dir,
        panel_id,
        panel_df,
        sources,
        warnings,
        group_cols=["metric", "condition", "comparison_condition"],
        extra_stats=extra_stats,
        extra_manifest=extra_manifest,
        source_records=source_records,
    )


def build_fig5_l2_writeback_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds, warnings = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.5 experiment root has no seed directories.")
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    source_records: list[dict[str, Any]] = []
    required_cols = {
        "network_seed",
        "condition",
        "condition_label",
        "condition_order",
        "memory_control_condition",
        "source_condition",
        "layer",
        "history_status",
        "history_label",
        "history_order",
        "n_trials",
        "n_l2_total_elements",
        "n_l2_history_sites",
        "n_l2_updated_sites",
        "n_l2_total_updated_sites",
        "fraction_among_updates",
        "update_probability_given_history",
        "dynamic_minus_static_prior_fraction",
        "dynamic_conditional_prior_minus_nonprior",
        "static_conditional_prior_minus_nonprior",
        "conditional_difference_in_differences",
    }
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / FIG5_L2_WRITEBACK_SOURCE_NAME
        source_records.append(_source_entry(path, repo_root))
        if not path.exists():
            warnings.append(f"Missing Fig.5D Layer2 re-update history source: {_rel(path, repo_root)}")
            continue
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = sorted(required_cols.difference(header))
        if missing:
            warnings.append(f"{_rel(path, repo_root)} lacks Layer2 re-update history columns {missing}.")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        use = df[df["condition"].astype(str).isin(["dynamic_intact", "static_opportunity"])].copy()
        if use.empty:
            warnings.append(f"{_rel(path, repo_root)} has no dynamic/static Layer2 re-update history rows.")
            continue
        use = use.sort_values(["condition_order", "history_order"], kind="mergesort")
        for _, r in use.iterrows():
            condition = str(r.get("condition", ""))
            condition_label = str(r.get("condition_label", "")) or MAIN_CONDITION_LABELS.get(condition, condition)
            history_status = str(r.get("history_status", ""))
            history_label = str(r.get("history_label", "")) or history_status.replace("_", " ")
            value = _num(r.get("update_probability_given_history"))
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "l2_reupdate_probability_given_history",
                    condition_label,
                    str(r.get("layer", "layer2_presynaptic")),
                    _network_id(seed_dir),
                    _seed_id(seed_dir, r.get("network_seed", "")),
                    value,
                    "probability",
                    path,
                    repo_root,
                    condition_label=condition_label,
                    condition_order=int(_num(r.get("condition_order"))),
                    perturbation_condition=condition,
                    raw_condition=condition,
                    source_condition=str(r.get("source_condition", "")),
                    source_metric="update_probability_given_history",
                    history_status=history_status,
                    history_label=history_label,
                    history_order=int(_num(r.get("history_order"))),
                    n_trials=r.get("n_trials", ""),
                    n_l2_total_elements=r.get("n_l2_total_elements", ""),
                    n_l2_history_sites=r.get("n_l2_history_sites", ""),
                    n_l2_updated_sites=r.get("n_l2_updated_sites", ""),
                    n_l2_total_updated_sites=r.get("n_l2_total_updated_sites", ""),
                    fraction_among_updates=_num(r.get("fraction_among_updates")),
                    update_probability_given_history=value,
                    dynamic_minus_static_prior_fraction=_num(r.get("dynamic_minus_static_prior_fraction")),
                    dynamic_conditional_prior_minus_nonprior=_num(r.get("dynamic_conditional_prior_minus_nonprior")),
                    static_conditional_prior_minus_nonprior=_num(r.get("static_conditional_prior_minus_nonprior")),
                    conditional_difference_in_differences=_num(r.get("conditional_difference_in_differences")),
                    denominator_definition=str(r.get("denominator_definition", "")),
                    run_mode="",
                )
            )
    panel_df = pd.DataFrame(rows)
    extra_stats = {
        "main_metric": "l2_reupdate_probability_given_history",
        "plot_type": "grouped_bar",
        "primary_plot_type": "grouped_bar",
        "source_file": FIG5_L2_WRITEBACK_SOURCE_NAME,
        "layer": "layer2_presynaptic",
        "conditions": ["dynamic_intact", "static_opportunity"],
        "condition_labels": ["Dynamic", "Static"],
        "history_segments": ["prior_updated", "not_prior_updated"],
        "history_labels": ["Prior-updated", "Not prior-updated"],
        "comparison": "probe update probability by prior-update history",
        "static_value_is_opportunity": True,
        "static_actual_stsp_mutation_expected_zero": True,
        "annotation_units": "percentage_points",
    }
    extra_manifest = {
        "primary_plot_type": "grouped_bar",
        "main_metric": "l2_reupdate_probability_given_history",
        "source_file": FIG5_L2_WRITEBACK_SOURCE_NAME,
        "checked_candidates": [record["path"] for record in source_records],
        "legacy_traceability_source_file": FIG5_L2_WRITEBACK_LEGACY_SOURCE_NAME,
        "plotted_conditions": ["dynamic_intact", "static_opportunity"],
        "plotted_condition_labels": ["Dynamic", "Static"],
        "plotted_history_segments": ["prior_updated", "not_prior_updated"],
        "plotted_history_labels": ["Prior-updated", "Not prior-updated"],
        "layer": "layer2_presynaptic",
        "static_value_is_opportunity": True,
        "static_actual_stsp_mutation_expected_zero": True,
        "mechanism_claim": "Prior-updated Layer2 sites have a higher probe update probability, especially under dynamic probe processing.",
    }
    return _write_result(
        spec,
        repo_root,
        output_dir,
        panel_id,
        panel_df,
        sources,
        warnings,
        group_cols=["condition", "history_status"],
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
    network_ids = [_seed_id(seed_dir, seed_dir.name) for seed_dir in seed_dirs]
    if not panel_df.empty:
        panel_df = panel_df.copy()
        panel_df["run_mode"] = run_mode
        panel_df["n_networks"] = int(n_networks)
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
        "network_ids": network_ids,
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
        "network_ids": network_ids,
        "source_files_used": [_rel(path, repo_root) for path in source_paths],
        "sources": [_source_entry(path, repo_root) for path in source_paths],
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


def _l1_stsp_transition_longform(
    df: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    seed_dir: Path,
    source_file: Path,
    repo_root: Path,
    *,
    legacy_global_used: bool,
    legacy_region_used: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    condition_order = {"dynamic_intact": 0, "attenuate_l1_stsp": 1, "reset_l1_stsp": 2}
    transition_order = {"advance": 0, "recruit": 1, "loss": 2}
    for _, r in df.iterrows():
        condition = str(r.get("condition", ""))
        label = str(r.get("condition_label", "")) or MAIN_CONDITION_LABELS.get(condition, condition)
        total_mass = sum(_num(r.get(col)) for col in MAIN_TRANSITION_COLUMNS.values())
        for transition_type, source_col in MAIN_TRANSITION_COLUMNS.items():
            value = _num(r.get(source_col))
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "transition_fraction",
                    label,
                    "layer1",
                    _network_id(seed_dir),
                    _seed_id(seed_dir, r.get("network_seed", "")),
                    value,
                    "proportion",
                    source_file,
                    repo_root,
                    condition_label=label,
                    condition_order=condition_order.get(condition, 999),
                    perturbation_condition=condition,
                    raw_condition=condition,
                    transition_type=transition_type,
                    transition_type_order=transition_order.get(transition_type, 999),
                    transition_label=TRANSITION_TYPE_LABELS.get(transition_type, transition_type),
                    y_value=value,
                    source_metric=source_col,
                    transition_mass=_num(r.get("transition_mass", total_mass)),
                    total_transition_mass=_num(r.get("transition_mass", total_mass)),
                    n_units=r.get("n_units", ""),
                    n_trials=r.get("n_trials", ""),
                    included_unit_groups=r.get("included_unit_groups", ""),
                    analysis_window_ms=r.get("analysis_window_ms", ""),
                    analysis_window_steps=r.get("analysis_window_steps", ""),
                    analysis_window_policy=r.get("analysis_window_policy", ""),
                    perturbation_mode=r.get("perturbation_mode", ""),
                    perturbed_layer=r.get("perturbed_layer", "layer1"),
                    perturbed_variables=r.get("perturbed_variables", ""),
                    legacy_global_perturbation_used=bool(legacy_global_used),
                    legacy_region_perturbation_used=bool(legacy_region_used),
                    run_mode="",
                )
            )
    return rows


def _l1_stsp_advance_or_recruit_headline_rows(
    transition_df: pd.DataFrame,
    *,
    figure_id: str,
    panel_id: str,
    repo_root: Path,
    analysis_window_ms: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Derive the pre-registered Fig. 5E paired headline from canonical 50-ms rows."""
    warnings: list[str] = []
    required = {"network_id", "seed_id", "perturbation_condition", "transition_type", "value", "source_file"}
    if transition_df.empty or not required.issubset(transition_df.columns):
        return pd.DataFrame(), ["Fig.5E canonical transition rows are missing fields needed for the headline paired contrast."]

    use = transition_df[
        transition_df["metric"].astype(str).eq("transition_fraction")
        & transition_df["perturbation_condition"].astype(str).isin(
            [FIG5E_HEADLINE_REFERENCE_CONDITION, *FIG5E_HEADLINE_COMPARISON_CONDITIONS]
        )
        & transition_df["transition_type"].astype(str).isin(["advance", "recruit"])
    ].copy()
    if use.empty:
        return pd.DataFrame(), ["Fig.5E has no canonical 50-ms advance/recruit rows for the frozen headline endpoint."]

    per_condition = (
        use.groupby(["network_id", "seed_id", "perturbation_condition"], as_index=False)
        .agg(
            advance_or_recruit=("value", "sum"),
            source_file=("source_file", "first"),
            analysis_window_steps=("analysis_window_steps", "first"),
            analysis_window_policy=("analysis_window_policy", "first"),
            included_unit_groups=("included_unit_groups", "first"),
            n_units=("n_units", "first"),
        )
    )
    values = per_condition.pivot(
        index=["network_id", "seed_id"],
        columns="perturbation_condition",
        values="advance_or_recruit",
    )
    source_paths = per_condition[per_condition["perturbation_condition"].eq(FIG5E_HEADLINE_REFERENCE_CONDITION)].set_index(
        ["network_id", "seed_id"]
    )
    rows: list[dict[str, Any]] = []
    for contrast_order, comparison in enumerate(FIG5E_HEADLINE_COMPARISON_CONDITIONS):
        missing_conditions = [
            condition
            for condition in (FIG5E_HEADLINE_REFERENCE_CONDITION, comparison)
            if condition not in values.columns
        ]
        if missing_conditions:
            warnings.append(f"Fig.5E cannot form {comparison} paired contrast; missing conditions={missing_conditions}.")
            continue
        paired = values[[FIG5E_HEADLINE_REFERENCE_CONDITION, comparison]].dropna()
        comparison_label = MAIN_CONDITION_LABELS.get(comparison, comparison)
        condition_label = f"Dynamic - {comparison_label}"
        for (network_id, seed_id), value_row in paired.iterrows():
            metadata = source_paths.loc[(network_id, seed_id)]
            source_path = repo_root / str(metadata["source_file"])
            dynamic_value = float(value_row[FIG5E_HEADLINE_REFERENCE_CONDITION])
            comparison_value = float(value_row[comparison])
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    FIG5E_HEADLINE_METRIC,
                    condition_label,
                    "layer1",
                    network_id,
                    seed_id,
                    dynamic_value - comparison_value,
                    "probability difference",
                    source_path,
                    repo_root,
                    contrast_order=contrast_order,
                    reference_condition=FIG5E_HEADLINE_REFERENCE_CONDITION,
                    comparison_condition=comparison,
                    comparison_condition_label=comparison_label,
                    p_dynamic_advance_or_recruit=dynamic_value,
                    p_condition_advance_or_recruit=comparison_value,
                    headline_endpoint="P(advance OR recruit | dynamic, first 50 ms) minus P(advance OR recruit | condition, first 50 ms)",
                    transition_types_included="advance;recruit",
                    transition_types_excluded="loss",
                    analysis_window_ms=float(analysis_window_ms),
                    analysis_window_steps=metadata.get("analysis_window_steps", ""),
                    analysis_window_policy=metadata.get("analysis_window_policy", ""),
                    included_unit_groups=metadata.get("included_unit_groups", ""),
                    n_units_dynamic=metadata.get("n_units", ""),
                    n_units_condition=per_condition.loc[
                        (per_condition["network_id"].astype(str).eq(str(network_id)))
                        & (per_condition["seed_id"].astype(str).eq(str(seed_id)))
                        & (per_condition["perturbation_condition"].astype(str).eq(comparison)),
                        "n_units",
                    ].iloc[0],
                    inference_unit="independently_trained_network",
                    confidence_interval="two-sided paired t-based 95% CI",
                    run_mode="",
                )
            )
    return pd.DataFrame(rows), warnings


def _paired_t95_summaries(df: pd.DataFrame, *, group_cols: list[str]) -> list[dict[str, Any]]:
    """Summarize already paired network-level contrasts with two-sided t-based 95% CIs."""
    if df.empty:
        return []
    summaries: list[dict[str, Any]] = []
    for keys, part in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(part["value"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) < 2:
            continue
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        critical = float(student_t.ppf(0.975, df=len(values) - 1))
        ci_half_width = critical * sem
        t_statistic = float(mean / sem) if sem > 0 else (float("inf") if mean != 0 else 0.0)
        p_value = float(2.0 * student_t.sf(abs(t_statistic), df=len(values) - 1)) if np.isfinite(t_statistic) else 0.0
        row = {column: value for column, value in zip(group_cols, keys)}
        row.update(
            {
                "mean": mean,
                "sem": sem,
                "ci95_lower": mean - ci_half_width,
                "ci95_upper": mean + ci_half_width,
                "t_statistic": t_statistic,
                "p_value_two_sided": p_value,
                "n_networks": int(len(values)),
                "values_used_for_plotting": [float(value) for value in values.tolist()],
            }
        )
        summaries.append(row)
    return summaries


def _l1_stsp_windowed_summary_from_unit_transitions(
    unit_df: pd.DataFrame,
    *,
    plotted_conditions: list[str],
    analysis_window_ms: float,
    analysis_window_steps: int,
) -> pd.DataFrame:
    if unit_df.empty:
        return pd.DataFrame()
    included_groups = ["overlap_dominant", "probe_only_dominant", "random_matched"]
    use = unit_df[unit_df["condition"].astype(str).isin(plotted_conditions)].copy()
    if "included_in_main" in use.columns:
        use = use[_bool_series(use["included_in_main"])].copy()
    else:
        use = use[use["unit_group"].astype(str).isin(included_groups)].copy()
    if use.empty:
        return pd.DataFrame()
    use["_first_static_windowed"] = _windowed_first_spikes(use["first_spike_static"], analysis_window_steps)
    use["_first_condition_windowed"] = _windowed_first_spikes(use["first_spike_condition"], analysis_window_steps)
    use["_transition_windowed"] = _windowed_transition_type(
        use["_first_condition_windowed"],
        use["_first_static_windowed"],
    )
    rows: list[dict[str, Any]] = []
    for (network_seed, condition), part in use.groupby(["network_seed", "condition"], sort=False):
        transitions = part["_transition_windowed"].astype(str)
        label_values = part.get("condition_label", pd.Series(dtype=str)).dropna().astype(str)
        label = label_values.iloc[0] if not label_values.empty and label_values.iloc[0] else MAIN_CONDITION_LABELS.get(str(condition), str(condition))
        perturbation_values = part.get("perturbation_mode", pd.Series(dtype=str)).dropna().astype(str)
        layer_values = part.get("perturbed_layer", pd.Series(dtype=str)).dropna().astype(str)
        variable_values = part.get("perturbed_variables", pd.Series(dtype=str)).dropna().astype(str)
        transition_mass = float(transitions.isin(["advance", "recruit", "loss"]).mean())
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": str(condition),
                "condition_label": label,
                "P_advance": float(transitions.eq("advance").mean()),
                "P_recruit": float(transitions.eq("recruit").mean()),
                "P_loss": float(transitions.eq("loss").mean()),
                "P_unchanged": float(transitions.eq("unchanged").mean()),
                "P_advance_plus_recruit": float(transitions.isin(["advance", "recruit"]).mean()),
                "transition_mass": transition_mass,
                "n_units": int(len(part)),
                "n_trials": int(part["trial_id"].nunique()) if "trial_id" in part.columns else int(len(part)),
                "included_unit_groups": ";".join(included_groups),
                "perturbation_mode": perturbation_values.iloc[0] if not perturbation_values.empty else _l1_stsp_mode(str(condition)),
                "perturbed_layer": layer_values.iloc[0] if not layer_values.empty else ("layer1" if str(condition) in {"attenuate_l1_stsp", "reset_l1_stsp"} else "none"),
                "perturbed_variables": variable_values.iloc[0] if not variable_values.empty else ("u_pre;x_pre" if str(condition) in {"attenuate_l1_stsp", "reset_l1_stsp"} else "none"),
                "analysis_window_ms": float(analysis_window_ms),
                "analysis_window_steps": int(analysis_window_steps),
                "analysis_window_policy": "first_spikes_at_or_after_window_are_treated_as_no_spike",
            }
        )
    return pd.DataFrame(rows)


def _analysis_window_steps(seed_dir: Path, analysis_window_ms: float) -> int:
    config_path = seed_dir / "run_config.json"
    dt_seconds = 0.001
    if config_path.exists():
        try:
            cfg = read_json(config_path)
            dt_seconds = float(cfg.get("dt", dt_seconds))
        except Exception:
            dt_seconds = 0.001
    dt_ms = max(dt_seconds * 1000.0, 1e-12)
    return max(1, int(round(float(analysis_window_ms) / dt_ms)))


def _windowed_first_spikes(values: pd.Series, analysis_window_steps: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(-1).astype(int)
    in_window = (numeric >= 0) & (numeric < int(analysis_window_steps))
    return numeric.where(in_window, -1)


def _windowed_transition_type(first_condition: pd.Series, first_static: pd.Series) -> pd.Series:
    out = pd.Series("unchanged", index=first_condition.index, dtype=object)
    out[(first_condition >= 0) & (first_static >= 0) & (first_condition < first_static)] = "advance"
    out[(first_condition >= 0) & (first_static < 0)] = "recruit"
    out[(first_condition < 0) & (first_static >= 0)] = "loss"
    return out


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _l1_stsp_mode(condition: str) -> str:
    if condition == "attenuate_l1_stsp":
        return "attenuate"
    if condition == "reset_l1_stsp":
        return "reset"
    return "none"


def _legacy_global_summary_as_l1_fallback(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mapping = {
        "dynamic_intact": "dynamic_intact",
        "attenuate_all_stsp": "attenuate_l1_stsp",
        "reset_all_stsp": "reset_l1_stsp",
    }
    use = df[df["condition"].astype(str).isin(mapping)].copy()
    if use.empty:
        return pd.DataFrame()
    use["condition"] = use["condition"].astype(str).map(mapping)
    use["condition_label"] = use["condition"].map(MAIN_CONDITION_LABELS).fillna(use["condition"])
    use["perturbation_mode"] = use["condition"].map({"dynamic_intact": "none", "attenuate_l1_stsp": "attenuate", "reset_l1_stsp": "reset"}).fillna("")
    use["perturbed_layer"] = use["condition"].map({"dynamic_intact": "none", "attenuate_l1_stsp": "layer1", "reset_l1_stsp": "layer1"}).fillna("")
    use["perturbed_variables"] = use["condition"].map({"dynamic_intact": "none", "attenuate_l1_stsp": "u_pre;x_pre", "reset_l1_stsp": "u_pre;x_pre"}).fillna("")
    return use


def _legacy_region_summary_as_l1_fallback(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mapping = {
        "dynamic_intact": "dynamic_intact",
        "attenuate_overlap_high_support": "attenuate_l1_stsp",
        "reset_overlap_high_support": "reset_l1_stsp",
    }
    use = df[df["condition"].astype(str).isin(mapping)].copy()
    if use.empty:
        return pd.DataFrame()
    use["condition"] = use["condition"].astype(str).map(mapping)
    rows: list[dict[str, Any]] = []
    for (network_seed, condition), part in use.groupby(["network_seed", "condition"], sort=False):
        row = {"network_seed": network_seed, "condition": condition, "condition_label": MAIN_CONDITION_LABELS.get(str(condition), str(condition))}
        for col in list(MAIN_TRANSITION_COLUMNS.values()) + ["P_unchanged"]:
            row[col] = float(pd.to_numeric(part.get(col, pd.Series(dtype=float)), errors="coerce").mean())
        row["transition_mass"] = float(sum(_num(row.get(col)) for col in MAIN_TRANSITION_COLUMNS.values()))
        row["n_units"] = int(pd.to_numeric(part.get("n_units", pd.Series(dtype=float)), errors="coerce").sum())
        row["n_trials"] = int(part.get("trial_id", pd.Series(dtype=float)).nunique()) if "trial_id" in part.columns else int(len(part))
        row["included_unit_groups"] = ";".join(sorted(set(part.get("unit_group", pd.Series(dtype=str)).dropna().astype(str))))
        row["perturbation_mode"] = "legacy_region_fallback"
        row["perturbed_layer"] = "layer1"
        row["perturbed_variables"] = "u_pre"
        rows.append(row)
    return pd.DataFrame(rows)


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
        "supp_postprobe_l2_writeback_by_network.csv",
        "supp_postprobe_l2_writeback_by_trial.csv",
        "supp_postprobe_l2_writeback_memory_overlap.csv",
        "supp_postprobe_l2_writeback_magnitude_qc.csv",
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
    exists = path.is_file()
    return {
        "path": _rel(path, repo_root),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": _sha256(path) if exists else "",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _split_unique(values: pd.Series) -> list[str]:
    out: set[str] = set()
    for value in values.dropna().astype(str):
        for item in value.replace(",", ";").split(";"):
            item = item.strip()
            if item:
                out.add(item)
    return sorted(out)


def _rel(path: Path | str, root: Path) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path_obj).replace("\\", "/")
