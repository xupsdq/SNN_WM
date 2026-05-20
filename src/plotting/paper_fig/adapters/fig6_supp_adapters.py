from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult, write_adapter_outputs
from src.plotting.paper_fig.adapters.fig6_adapters import (
    DEFAULT_EXPERIMENT_ROOT,
    DOWNSTREAM_LABELS,
    _bool_value,
    _claim_strength,
    _finish,
    _first_col,
    _first_existing,
    _first_nonempty_by_level,
    _forbidden_language_from_seeds,
    _ids,
    _input_overlap_rows,
    _num,
    _reentry_rows,
    _rel,
    _row,
    _seed_dirs,
    _update_history_rows,
)
from src.plotting.paper_fig.utils import read_json


def build_s11_peak_update_history_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, _, candidates = _first_existing(seed_dir, ["data/metrics/panel_b_peak_update_history_summary.csv", "data/metrics/panel_b_peak_update_history.csv", "data/metrics/supp_s11_peak_update_group_enrichment.csv"])
        checked.extend(candidates)
        if path is None:
            warnings.append(f"Missing S11A update-history source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        rows.extend(_update_history_rows(figure_id, panel_id, seed_dir, path, repo_root))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["metric", "condition"], manifest_extra={"source_mode": "s11_peak_update_history"})


def build_s11_update_recency_model_detail_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, _, candidates = _first_existing(
            seed_dir,
            [
                "data/metrics/supp_update_recency_model_coefficients.csv",
                "data/metrics/supp_update_recency_support_model_coefficients.csv",
                "data/metrics/panel_b_update_recency_model_metrics.csv",
                "data/metrics/supp_s11_update_recency_model_comparison.csv",
                "data/metrics/supp_update_recency_support_model_metrics.csv",
                "data/metrics/supp_legacy_panel_b_update_recency_model_metrics.csv",
            ],
        )
        checked.extend(candidates)
        if path is None:
            warnings.append(f"Missing S11B update-recency detail source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        if "term" in df.columns or "predictor" in df.columns or "coefficient_name" in df.columns:
            term_col = _first_col(df, ["term", "predictor", "coefficient_name"])
            value_col = _first_col(df, ["estimate", "beta", "coefficient", "coefficient_value", "value"])
            for _, r in df.iterrows():
                rows.append(_row(figure_id, panel_id, "coefficient", str(r.get(term_col, "")) if term_col else "coefficient", "layer1", seed_dir, r.get("network_seed", ""), _num(r.get(value_col)) if value_col else np.nan, "coefficient", path, repo_root, model_name=r.get("model_name", ""), se=_num(r.get("se", r.get("std_error", np.nan))), p_value=_num(r.get("p_value", r.get("p", np.nan))), standardized_coefficient=_num(r.get("standardized_coefficient", np.nan))))
        elif "model_name" in df.columns:
            value_col = "cv_r2" if "cv_r2" in df.columns else "r2" if "r2" in df.columns else None
            if value_col is None:
                continue
            for _, r in df.iterrows():
                rows.append(_row(figure_id, panel_id, "cv_r2", str(r.get("model_name", "")), "layer1", seed_dir, r.get("network_seed", ""), _num(r.get(value_col)), "r2", path, repo_root, model_name=r.get("model_name", ""), r2=_num(r.get("r2")), cv_r2=_num(r.get("cv_r2"))))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["metric", "condition"], manifest_extra={"source_mode": "s11_update_recency_model_detail"})


def build_s11_peak_source_attribution_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, _, candidates = _first_existing(seed_dir, ["data/metrics/panel_a_peak_source_attribution_summary.csv", "data/metrics/panel_a_peak_source_attribution.csv", "data/metrics/supp_s11_leave_one_out_source_details.csv"])
        checked.extend(candidates)
        if path is None:
            warnings.append(f"Missing S11C leave-one-out source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        value_col = _first_col(df, ["mean_peak_loss_fraction", "peak_loss_fraction", "peak_loss"])
        if value_col is None:
            continue
        for _, r in df.iterrows():
            rows.append(_row(figure_id, panel_id, "peak_loss_fraction", "Peak source contribution", "layer1", seed_dir, r.get("network_seed", ""), _num(r.get(value_col)), "fraction", path, repo_root, sequence_id=r.get("sequence_id", ""), relative_position_from_end=r.get("relative_position_from_end", ""), removed_position=r.get("removed_position", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["metric", "relative_position_from_end"], manifest_extra={"source_mode": "s11_peak_source_attribution"})


def build_s11_peak_input_overlap_origin_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, _, candidates = _first_existing(seed_dir, ["data/metrics/panel_c_peak_input_overlap_similarity_summary.csv", "data/metrics/panel_c_peak_input_overlap_similarity.csv", "data/metrics/supp_s11_recent_overlap_window_robustness.csv", "data/metrics/supp_recent_overlap_window_robustness.csv"])
        checked.extend(candidates)
        if path is None:
            warnings.append(f"Missing S11D input-overlap origin source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        rows.extend(_input_overlap_rows(figure_id, panel_id, seed_dir, path, repo_root))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["metric", "condition"], manifest_extra={"source_mode": "s11_peak_input_overlap_origin"})


def build_s11_alternative_peak_definitions_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_wide_metric_adapter(
        spec,
        repo_root,
        output_dir,
        ["data/metrics/supp_alternative_peak_definitions.csv", "data/metrics/supp_peak_definition_sensitivity.csv", "data/metrics/supp_s11_alternative_peak_definitions.csv"],
        ["P_peak", "multi_recent_enrichment", "peak_overlap_dice", "peak_overlap_coverage", "peak_weighted_overlap_effect"],
        condition_cols=["peak_definition"],
        unit="stability",
        source_mode="s11_alternative_peak_definitions",
    )


def build_s11_visual_energy_classpair_controls_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_wide_metric_adapter(
        spec,
        repo_root,
        output_dir,
        ["data/metrics/supp_visual_energy_classpair_controls.csv", "data/metrics/supp_s11_visual_energy_classpair_controls.csv", "data/metrics/supp_trial_condition_audit.csv"],
        ["energy_difference", "foreground_difference", "visual_similarity_difference", "class_pair_balance_stat", "mean_input_energy", "mean_visual_similarity"],
        condition_cols=["comparison", "group", "condition"],
        unit="diagnostic",
        source_mode="s11_visual_energy_classpair_controls",
    )


def build_s12_raw_overlap_matched_reentry_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        candidates = [
            ("real_matched", "data/metrics/panel_d_raw_overlap_matched_peak_reentry.csv"),
            ("peak_weighted_fallback", "data/metrics/panel_d_matched_raw_overlap_comparison.csv"),
            ("peak_weighted_fallback", "data/metrics/supp_matched_raw_overlap_peak_comparison.csv"),
            ("peak_weighted_fallback", "data/metrics/supp_s12_raw_overlap_matched_peak_overlap_contrast.csv"),
            ("peak_weighted_fallback", "data/metrics/supp_legacy_panel_d_matched_raw_overlap_comparison.csv"),
        ]
        checked.extend(seed_dir / rel for _, rel in candidates)
        level, path, df = _first_nonempty_by_level(seed_dir, candidates)
        if path is None or df is None:
            warnings.append(f"Missing S12A raw-overlap matched source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        rows.extend(_reentry_rows(figure_id, panel_id, seed_dir, path, repo_root, df, level))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["condition"], manifest_extra={"source_mode": "s12_raw_overlap_matched_reentry"})


def build_s12_peak_overlap_regression_controls_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    wanted = ["beta_peak_weighted_overlap", "beta_raw_overlap", "beta_visual_similarity", "beta_input_energy"]
    for seed_dir in seeds:
        path, _, candidates = _first_existing(seed_dir, ["data/metrics/panel_d_peak_overlap_reentry_regression.csv", "data/metrics/panel_d_peak_weighted_overlap_regression.csv", "data/metrics/supp_raw_vs_peak_weighted_overlap_regression.csv", "data/metrics/supp_s12_peak_weighted_regression_controls.csv", "data/metrics/supp_legacy_panel_d_peak_weighted_overlap_regression.csv"])
        checked.extend(candidates)
        if path is None:
            warnings.append(f"Missing S12B regression control source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        if "predictor" in df.columns:
            for _, r in df.iterrows():
                predictor = str(r.get("predictor", ""))
                if predictor not in {"peak_weighted_overlap", "raw_overlap", "visual_similarity", "input_energy"}:
                    continue
                rows.append(_row(figure_id, panel_id, f"beta_{predictor}", predictor, "layer3", seed_dir, r.get("network_seed", ""), _num(r.get("beta")), "coefficient", path, repo_root, se=_num(r.get("se")), p_value=_num(r.get("p_value")), r2=_num(r.get("r2"))))
        else:
            for _, r in df.iterrows():
                for metric in wanted:
                    if metric in df.columns:
                        rows.append(_row(figure_id, panel_id, metric, metric.replace("beta_", ""), "layer3", seed_dir, r.get("network_seed", ""), _num(r.get(metric)), "coefficient", path, repo_root, p_value=_num(r.get("p_peak_weighted")), r2=_num(r.get("r2"))))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["metric", "condition"], manifest_extra={"source_mode": "s12_peak_overlap_regression_controls"})


def build_s12_real_rollout_proxy_audit_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        summary_path = seed_dir / "summary.json"
        audit_path, _, candidates = _first_existing(seed_dir, ["data/metrics/panel_de_route_peak_perturbation_scientific_use_audit.csv", "data/metrics/supp_s12_real_rollout_scientific_use_audit.csv", "data/metrics/panel_de_real_rollout_scientific_use_audit.csv", "data/metrics/supp_trial_condition_audit.csv"])
        checked.extend([summary_path] + candidates)
        summary = read_json(summary_path) if summary_path.exists() else {}
        if summary_path.exists():
            sources.append(summary_path)
        audit = pd.read_csv(audit_path) if audit_path is not None and audit_path.exists() else pd.DataFrame()
        if audit_path is not None:
            sources.append(audit_path)
        fields = {
            "proxy_mode": _bool_value(summary.get("proxy_mode", audit.get("proxy_mode", pd.Series([False])).iloc[0] if not audit.empty and "proxy_mode" in audit else False)),
            "final_scientific_use": _bool_value(summary.get("final_scientific_use", audit.get("final_scientific_use", pd.Series([False])).iloc[0] if not audit.empty and "final_scientific_use" in audit else False)),
            "peak_perturbation_implemented": _bool_value(summary.get("peak_perturbation_implemented", audit.get("route_peak_perturbation_implemented", pd.Series([False])).iloc[0] if not audit.empty and "route_peak_perturbation_implemented" in audit else False)),
            "route_peak_perturbation_success": _bool_value(summary.get("peak_perturbation_successful", audit.get("route_peak_perturbation_success", pd.Series([False])).iloc[0] if not audit.empty and "route_peak_perturbation_success" in audit else False)),
            "formula_proxy_reentry_removed_from_main": _bool_value(summary.get("formula_proxy_reentry_removed_from_main", False)),
            "allowed_claim_strength": str(summary.get("allowed_claim_strength", _claim_strength([seed_dir]))),
        }
        for field, value in fields.items():
            rows.append(_row(figure_id, panel_id, "scientific_use_audit", field, "audit", seed_dir, summary.get("network_seed", ""), 1.0 if _truthy_for_plot(value) else 0.0, "status", summary_path if summary_path.exists() else audit_path or seed_dir, repo_root, status_text=str(value), proxy_mode=fields["proxy_mode"], final_scientific_use=fields["final_scientific_use"], allowed_claim_strength=fields["allowed_claim_strength"], peak_perturbation_implemented=fields["peak_perturbation_implemented"], route_peak_perturbation_success=fields["route_peak_perturbation_success"], formula_proxy_reentry_removed_from_main=fields["formula_proxy_reentry_removed_from_main"]))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["condition"], manifest_extra={"source_mode": "s12_real_rollout_proxy_audit", "claim_strength": _claim_strength(seeds)})


def build_s12_downstream_metric_breakdown_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_wide_metric_adapter(
        spec,
        repo_root,
        output_dir,
        ["data/metrics/supp_s12_downstream_metric_breakdown.csv", "data/metrics/panel_e_downstream_metric_breakdown.csv", "data/metrics/panel_e_real_downstream_metrics.csv", "data/metrics/panel_e_peak_weighted_downstream_metrics.csv", "data/metrics/panel_e_peak_overlap_downstream_regression.csv", "data/metrics/panel_e_downstream_regression.csv"],
        ["high_minus_low", "beta_peak_weighted_overlap", "early_recruitment_gain_real", "response_pattern_displacement_real", "decision_deflection_score_real", "early_recruitment_gain", "response_pattern_displacement", "decision_deflection_score"],
        condition_cols=["downstream_metric", "metric"],
        unit="effect",
        source_mode="s12_downstream_metric_breakdown",
    )


def build_s12_global_support_spike_controls_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_wide_metric_adapter(
        spec,
        repo_root,
        output_dir,
        ["data/metrics/supp_global_support_spike_count_controls.csv", "data/metrics/supp_s12_global_support_spike_count_controls.csv", "data/metrics/supp_nonoverlap_peak_perturbation_controls.csv", "data/metrics/supp_trial_condition_audit.csv"],
        ["beta_peak_weighted_overlap", "beta_global_support", "beta_total_spike_count", "beta_nonpeak_support", "r2", "global_support_balance", "spike_count_balance"],
        condition_cols=["dependent_metric", "metric", "condition"],
        unit="control",
        source_mode="s12_global_support_spike_controls",
    )


def build_s12_peak_perturbation_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, _, candidates = _first_existing(seed_dir, ["data/metrics/panel_d_route_peak_reentry_loss_summary.csv", "data/metrics/panel_e_route_peak_downstream_summary.csv", "data/metrics/supp_s12_peak_perturbation_metrics.csv", "data/metrics/supp_overlap_aligned_peak_perturbation.csv", "data/metrics/supp_nonoverlap_peak_perturbation_controls.csv"])
        summary_path = seed_dir / "summary.json"
        checked.extend(candidates + [summary_path])
        summary = read_json(summary_path) if summary_path.exists() else {}
        if summary_path.exists():
            sources.append(summary_path)
        if path is None:
            warnings.append("No peak perturbation evidence; claim remains predictive_peak_amplified.")
            rows.append(_row(figure_id, panel_id, "optional_placeholder", "No perturbation", "audit", seed_dir, summary.get("network_seed", ""), 0.0, "status", summary_path if summary_path.exists() else seed_dir, repo_root, placeholder_reason="Optional peak perturbation not available; main claim remains predictive peak-amplified.", peak_perturbation_implemented=False, peak_perturbation_successful=False, allowed_claim_strength="predictive_peak_amplified"))
            continue
        sources.append(path)
        df = pd.read_csv(path)
        if "perturbation_unit_set" in df.columns and ("mean_normalized_reentry_loss" in df.columns or "P_output_switch" in df.columns):
            for _, r in df.iterrows():
                unit_set = str(r.get("perturbation_unit_set", ""))
                for metric in ("mean_normalized_reentry_loss", "P_output_switch", "mean_response_displacement_loss", "mean_decision_deflection_loss"):
                    if metric not in df.columns:
                        continue
                    rows.append(_row(figure_id, panel_id, metric, unit_set.replace("_", " ").title(), "layer3", seed_dir, r.get("network_seed", ""), _num(r.get(metric)), "effect", path, repo_root, peak_perturbation_implemented=True, peak_perturbation_successful=_claim_strength([seed_dir]) == "causal_route_peak_gain", allowed_claim_strength=_claim_strength([seed_dir])))
            continue
        value_col = _first_col(df, ["causal_contribution", "dynamic_like_recovery", "decision_deflection_score", "effect", "value"])
        cond_col = _first_col(df, ["condition", "perturbation_condition", "group"])
        for _, r in df.iterrows():
            success = _bool_value(r.get("successful", r.get("peak_perturbation_successful", False)))
            rows.append(_row(figure_id, panel_id, "peak_perturbation_effect", str(r.get(cond_col, "perturbation")) if cond_col else "perturbation", "layer3", seed_dir, r.get("network_seed", ""), _num(r.get(value_col)) if value_col else np.nan, "effect", path, repo_root, peak_perturbation_implemented=True, peak_perturbation_successful=success, allowed_claim_strength="causal_contribution_supported" if success else "predictive_peak_amplified"))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["condition"], manifest_extra={"source_mode": "s12_peak_perturbation", "claim_strength": _claim_strength(seeds), "forbidden_language": _forbidden_language_from_seeds(seeds)})


def _build_wide_metric_adapter(
    spec: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    candidates: Sequence[str],
    metric_cols: Sequence[str],
    *,
    condition_cols: Sequence[str],
    unit: str,
    source_mode: str,
) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, _, paths = _first_existing(seed_dir, candidates)
        checked.extend(paths)
        if path is None:
            warnings.append(f"Missing {figure_id}{panel_id} source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        condition_col = _first_col(df, condition_cols)
        if {"metric", "value"}.issubset(df.columns):
            for _, r in df.iterrows():
                condition = str(r.get(condition_col, r.get("control_variable", r.get("model_or_comparison", r.get("peak_definition", ""))))) if condition_col or "control_variable" in df.columns or "model_or_comparison" in df.columns or "peak_definition" in df.columns else str(r.get("metric", "summary"))
                metric = str(r.get("metric", "value"))
                rows.append(_row(figure_id, panel_id, metric, DOWNSTREAM_LABELS.get(condition, condition), "layer3" if "downstream" in source_mode or "s12" in source_mode else "layer1", seed_dir, r.get("network_seed", ""), _num(r.get("value")), unit, path, repo_root, source_metric=metric, raw_overlap=_num(r.get("raw_overlap")), peak_weighted_overlap=_num(r.get("peak_weighted_overlap")), proxy_mode=_bool_value(r.get("proxy_mode", False)), final_scientific_use=_bool_value(r.get("final_scientific_use", True)), real_rollout=_bool_value(r.get("real_rollout", False))))
            continue
        for _, r in df.iterrows():
            base_condition = str(r.get(condition_col, "")) if condition_col else "summary"
            for metric in metric_cols:
                if metric not in df.columns:
                    continue
                condition = DOWNSTREAM_LABELS.get(base_condition, base_condition)
                rows.append(_row(figure_id, panel_id, metric, condition, "layer3" if "downstream" in source_mode or "s12" in source_mode else "layer1", seed_dir, r.get("network_seed", ""), _num(r.get(metric)), unit, path, repo_root, source_metric=metric, raw_overlap=_num(r.get("raw_overlap")), peak_weighted_overlap=_num(r.get("peak_weighted_overlap")), proxy_mode=_bool_value(r.get("proxy_mode", False)), final_scientific_use=_bool_value(r.get("final_scientific_use", True)), real_rollout=_bool_value(r.get("real_rollout", False))))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["metric", "condition"], manifest_extra={"source_mode": source_mode})


def _truthy_for_plot(value: Any) -> bool:
    if isinstance(value, str) and value not in {"", "False", "false", "0"}:
        return True
    return _bool_value(value)
