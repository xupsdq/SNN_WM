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


def build_s11_score_input_ping_audit_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        gain_path = seed_dir / "data/metrics/fig6_gain_ratio_audit.csv"
        entry_path = seed_dir / "data/metrics/fig6_entry_score_audit.csv"
        ping_path = seed_dir / "data/metrics/panel_b_region_ping_readout_bias.csv"
        checked.extend([gain_path, entry_path, ping_path])
        gain = pd.read_csv(gain_path) if gain_path.exists() else pd.DataFrame()
        entry = pd.read_csv(entry_path) if entry_path.exists() else pd.DataFrame()
        ping = pd.read_csv(ping_path) if ping_path.exists() else pd.DataFrame()
        for path, df in ((gain_path, gain), (entry_path, entry), (ping_path, ping)):
            if not df.empty:
                sources.append(path)
        if not gain.empty:
            rows.append(_row(figure_id, panel_id, "nonfinite_raw_count", "score finite", "audit", seed_dir, _seed_from_frames(gain), float(pd.to_numeric(gain.get("nonfinite_raw_count"), errors="coerce").fillna(0).sum()), "count", gain_path, repo_root))
            rows.append(_row(figure_id, panel_id, "baseline_floor_count", "baseline floor", "audit", seed_dir, _seed_from_frames(gain), float(pd.to_numeric(gain.get("baseline_floor_count"), errors="coerce").fillna(0).sum()), "count", gain_path, repo_root))
            rows.append(_row(figure_id, panel_id, "clipped_ratio_max", "gain ratio clip", "audit", seed_dir, _seed_from_frames(gain), _num(pd.to_numeric(gain.get("clipped_ratio_max"), errors="coerce").max()), "ratio", gain_path, repo_root))
        if not entry.empty:
            for (entry_type, condition), part in entry.groupby(["entry_type", "entry_condition"], dropna=False):
                rows.append(_row(figure_id, panel_id, "mean_valid_site_count", f"{entry_type}:{condition}", "audit", seed_dir, _seed_from_frames(entry), float(pd.to_numeric(part.get("valid_site_count"), errors="coerce").mean()), "sites", entry_path, repo_root, entry_type=entry_type, entry_condition=condition))
                rows.append(_row(figure_id, panel_id, "mean_entry_area", f"{entry_type}:{condition}", "audit", seed_dir, _seed_from_frames(entry), float(pd.to_numeric(part.get("entry_area"), errors="coerce").mean()), "pixels", entry_path, repo_root, entry_type=entry_type, entry_condition=condition))
        if not ping.empty:
            for condition, part in ping.groupby("entry_condition", dropna=False):
                rows.append(_row(figure_id, panel_id, "ping_active_sites", str(condition), "audit", seed_dir, _seed_from_frames(ping), float(pd.to_numeric(part.get("ping_active_sites"), errors="coerce").mean()), "sites", ping_path, repo_root, entry_condition=condition))
                rows.append(_row(figure_id, panel_id, "total_ping_current", str(condition), "audit", seed_dir, _seed_from_frames(ping), float(pd.to_numeric(part.get("total_ping_current"), errors="coerce").mean()), "current", ping_path, repo_root, entry_condition=condition))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["metric", "condition"], manifest_extra={"source_mode": "s11_score_input_ping_audit"})


def build_s11_global_ping_count_endpoint_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, df, candidates = _first_existing(seed_dir, ["data/metrics/supp_s11b_global_ping_count_endpoint.csv", "data/metrics/panel_c_global_ping_score_spike_prediction.csv"])
        checked.extend(candidates)
        if path is None or df is None or df.empty:
            warnings.append(f"Missing S7B global-ping count endpoint source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        if {"metric", "value", "score_quantile_bin"}.issubset(df.columns):
            use = df[df["metric"].astype(str).eq("mean_early_spike_count")]
            for _, r in use.iterrows():
                rows.append(_row(figure_id, panel_id, "mean_early_spike_count", str(r.get("score_quantile_bin", r.get("condition", ""))), "layer1", seed_dir, r.get("network_seed", ""), _num(r.get("value")), "spike count", path, repo_root, score_quantile_bin=r.get("score_quantile_bin", r.get("condition", "")), x_value=_score_x(r.get("score_quantile_bin", r.get("condition", "")))))
            continue
        for quantile, part in df.groupby("score_quantile_bin", dropna=False):
            rows.append(_row(figure_id, panel_id, "mean_early_spike_count", str(quantile), "layer1", seed_dir, _seed_from_frames(df), float(pd.to_numeric(part.get("mean_early_spike_count"), errors="coerce").mean()), "spike count", path, repo_root, score_quantile_bin=str(quantile), x_value=_score_x(quantile)))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["score_quantile_bin"], manifest_extra={"source_mode": "s11_global_ping_count_endpoint"})


def build_s11_real_probe_window_robustness_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _window_delta_adapter(spec, repo_root, output_dir, ["data/metrics/supp_s11c_real_probe_window_robustness.csv", "data/metrics/panel_d_real_probe_score_spike_deflection.csv"], source_mode="s11_real_probe_window_robustness")


def build_s11_overlap_interaction_window_robustness_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, df, candidates = _first_existing(seed_dir, ["data/metrics/supp_s11d_overlap_interaction_window_robustness.csv", "data/metrics/panel_e_overlap_gated_stsp_interaction.csv"])
        checked.extend(candidates)
        if path is None or df is None or df.empty:
            warnings.append(f"Missing S7D overlap interaction source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        if {"metric", "value", "early_window_ms"}.issubset(df.columns):
            use = df[df["metric"].astype(str).eq("interaction_delta")]
            for _, r in use.iterrows():
                rows.append(_row(figure_id, panel_id, "interaction_delta", str(r.get("early_window_ms", r.get("condition", ""))), "layer1", seed_dir, r.get("network_seed", ""), _num(r.get("value")), "probability difference", path, repo_root, early_window_ms=_num(r.get("early_window_ms")), x_value=_num(r.get("early_window_ms")), n_valid=_num(r.get("n_valid")), fraction_positive=_num(r.get("fraction_positive"))))
            continue
        for window, part in df.groupby("early_window_ms", dropna=False):
            vals = pd.to_numeric(part.get("interaction_delta"), errors="coerce").dropna()
            rows.append(_row(figure_id, panel_id, "interaction_delta", str(window), "layer1", seed_dir, _seed_from_frames(df), float(vals.mean()) if len(vals) else np.nan, "probability difference", path, repo_root, early_window_ms=_num(window), x_value=_num(window), n_valid=float(len(vals)), fraction_positive=float((vals > 0).mean()) if len(vals) else np.nan))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["early_window_ms"], manifest_extra={"source_mode": "s11_overlap_interaction_window_robustness"})


def build_s11_overlap_site_availability_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, df, candidates = _first_existing(seed_dir, ["data/metrics/supp_s11e_overlap_site_availability.csv", "data/metrics/panel_e_overlap_gated_stsp_recruitment.csv"])
        checked.extend(candidates)
        if path is None or df is None or df.empty:
            warnings.append(f"Missing S7E site availability source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        if {"metric", "value", "stsp_group", "overlap_group"}.issubset(df.columns):
            use = df[df["metric"].astype(str).eq("mean_sites")]
            for _, r in use.iterrows():
                rows.append(_row(figure_id, panel_id, "mean_sites", str(r.get("condition", "")), "layer1", seed_dir, r.get("network_seed", ""), _num(r.get("value")), "sites", path, repo_root, stsp_group=r.get("stsp_group", ""), overlap_group=r.get("overlap_group", ""), median_sites=_num(r.get("median_sites")), nonzero_fraction=_num(r.get("nonzero_fraction"))))
            continue
        for (stsp, overlap), part in df.groupby(["stsp_group", "overlap_group"], dropna=False):
            sites = pd.to_numeric(part.get("n_sites"), errors="coerce")
            rows.append(_row(figure_id, panel_id, "mean_sites", f"{stsp}:{overlap}", "layer1", seed_dir, _seed_from_frames(df), float(sites.mean()), "sites", path, repo_root, stsp_group=str(stsp), overlap_group=str(overlap), median_sites=float(sites.median()), nonzero_fraction=float((sites > 0).mean())))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["stsp_group", "overlap_group"], manifest_extra={"source_mode": "s11_overlap_site_availability"})


def build_s11_high_stsp_ablation_paired_difference_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, df, candidates = _first_existing(seed_dir, ["data/metrics/supp_s11f_high_stsp_ablation_paired_difference.csv", "data/metrics/panel_a_high_stsp_overlap_ablation_summary.csv", "data/metrics/panel_f_high_stsp_overlap_ablation_summary.csv"])
        checked.extend(candidates)
        if path is None or df is None or df.empty:
            warnings.append(f"Missing S7F paired ablation source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        if {"metric", "value"}.issubset(df.columns):
            for _, r in df.iterrows():
                rows.append(_row(figure_id, panel_id, "high_stsp_overlap_minus_matched_loss", "paired_difference", "layer1", seed_dir, r.get("network_seed", ""), _num(r.get("value")), "probability difference", path, repo_root, sequence_id=r.get("sequence_id", ""), probe_id=r.get("probe_id", ""), high_stsp_overlap=_num(r.get("high_stsp_overlap")), matched_removal=_num(r.get("matched_removal"))))
            continue
        pivot = df.pivot_table(index=["network_seed", "sequence_id", "probe_id"], columns="loss_condition", values="loss_delta_spike_probability", aggfunc="mean")
        if {"high_stsp_overlap", "matched_removal"}.issubset(set(pivot.columns)):
            for (network_seed, sequence_id, probe_id), r in pivot.iterrows():
                diff = _num(r.get("high_stsp_overlap")) - _num(r.get("matched_removal"))
                rows.append(_row(figure_id, panel_id, "high_stsp_overlap_minus_matched_loss", "paired_difference", "layer1", seed_dir, network_seed, diff, "probability difference", path, repo_root, sequence_id=sequence_id, probe_id=probe_id, high_stsp_overlap=_num(r.get("high_stsp_overlap")), matched_removal=_num(r.get("matched_removal"))))
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["metric", "condition"], manifest_extra={"source_mode": "s11_high_stsp_ablation_paired_difference"})


def build_s11_score_shuffle_null_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _optional_extension_adapter(spec, repo_root, output_dir, "data/metrics/supp_s11g_score_shuffle_null.csv", "score_shuffle_null")


def build_s11_threshold_sensitivity_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _optional_extension_adapter(spec, repo_root, output_dir, "data/metrics/supp_s11h_threshold_sensitivity.csv", "threshold_sensitivity")


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
            warnings.append(f"Missing S7A update-history source under {_rel(seed_dir, repo_root)}.")
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
            warnings.append(f"Missing S7B update-recency detail source under {_rel(seed_dir, repo_root)}.")
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
            warnings.append(f"Missing S7C leave-one-out source under {_rel(seed_dir, repo_root)}.")
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
            warnings.append(f"Missing S7D input-overlap origin source under {_rel(seed_dir, repo_root)}.")
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
            "proxy_fallback_removed_from_main": _bool_value(summary.get("proxy_fallback_removed_from_main", False)),
            "allowed_claim_strength": str(summary.get("allowed_claim_strength", _claim_strength([seed_dir]))),
        }
        for field, value in fields.items():
            rows.append(_row(figure_id, panel_id, "scientific_use_audit", field, "audit", seed_dir, summary.get("network_seed", ""), 1.0 if _truthy_for_plot(value) else 0.0, "status", summary_path if summary_path.exists() else audit_path or seed_dir, repo_root, status_text=str(value), proxy_mode=fields["proxy_mode"], final_scientific_use=fields["final_scientific_use"], allowed_claim_strength=fields["allowed_claim_strength"], peak_perturbation_implemented=fields["peak_perturbation_implemented"], route_peak_perturbation_success=fields["route_peak_perturbation_success"], proxy_fallback_removed_from_main=fields["proxy_fallback_removed_from_main"]))
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


def _window_delta_adapter(
    spec: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    candidates: Sequence[str],
    *,
    source_mode: str,
) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, df, paths = _first_existing(seed_dir, candidates)
        checked.extend(paths)
        if path is None or df is None or df.empty:
            warnings.append(f"Missing {figure_id}{panel_id} window-robustness source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        if {"metric", "value", "early_window_ms"}.issubset(df.columns):
            use = df[df["metric"].astype(str).eq("q5_minus_q1_delta_spike_probability")]
            if use.empty:
                use = df[df["metric"].astype(str).eq("delta_spike_probability")]
            for _, r in use.iterrows():
                window = r.get("early_window_ms", r.get("condition", ""))
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        "q5_minus_q1_delta_spike_probability",
                        str(window),
                        "layer1",
                        seed_dir,
                        r.get("network_seed", ""),
                        _num(r.get("value")),
                        "probability difference",
                        path,
                        repo_root,
                        early_window_ms=_num(window),
                        x_value=_num(window),
                        q1_mean=_num(r.get("q1_mean")),
                        q5_mean=_num(r.get("q5_mean")),
                    )
                )
            continue
        if not {"early_window_ms", "score_quantile_bin", "delta_spike_probability"}.issubset(df.columns):
            warnings.append(f"{figure_id}{panel_id}: source lacks early_window_ms/score_quantile_bin/delta_spike_probability.")
            continue
        for window, part in df.groupby("early_window_ms", dropna=False):
            means = part.groupby("score_quantile_bin")["delta_spike_probability"].mean()
            q1 = _num(means.get("Q1"))
            q5 = _num(means.get("Q5"))
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "q5_minus_q1_delta_spike_probability",
                    str(window),
                    "layer1",
                    seed_dir,
                    _seed_from_frames(df),
                    q5 - q1,
                    "probability difference",
                    path,
                    repo_root,
                    early_window_ms=_num(window),
                    x_value=_num(window),
                    q1_mean=q1,
                    q5_mean=q5,
                    n_rows=float(len(part)),
                )
            )
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["early_window_ms"], manifest_extra={"source_mode": source_mode})


def _optional_extension_adapter(
    spec: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    rel_path: str,
    source_mode: str,
) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path = seed_dir / rel_path
        checked.append(path)
        if not path.exists():
            warnings.append(f"{figure_id}{panel_id}: optional extension {source_mode} not run under {_rel(seed_dir, repo_root)}.")
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "optional_placeholder",
                    source_mode,
                    "optional_extension",
                    seed_dir,
                    "",
                    0.0,
                    "status",
                    path,
                    repo_root,
                    placeholder_reason=f"Optional {source_mode} extension not available in this bundle.",
                    optional_extension=True,
                )
            )
            continue
        df = pd.read_csv(path)
        if df.empty:
            warnings.append(f"{figure_id}{panel_id}: optional extension {source_mode} source is empty.")
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "optional_placeholder",
                    source_mode,
                    "optional_extension",
                    seed_dir,
                    "",
                    0.0,
                    "status",
                    path,
                    repo_root,
                    placeholder_reason=f"Optional {source_mode} extension source is empty.",
                    optional_extension=True,
                )
            )
            sources.append(path)
            continue
        sources.append(path)
        metric_col = _first_col(df, ["metric", "endpoint", "dependent_metric"])
        condition_col = _first_col(df, ["condition", "score_quantile_bin", "stsp_group_quantile", "overlap_threshold"])
        value_col = _first_col(df, ["value", "observed_minus_null", "interaction_delta", "delta", "q5_minus_q1"])
        for _, r in df.iterrows():
            metric = str(r.get(metric_col, "value")) if metric_col else "value"
            condition = str(r.get(condition_col, metric)) if condition_col else metric
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    metric,
                    condition,
                    "optional_extension",
                    seed_dir,
                    r.get("network_seed", ""),
                    _num(r.get(value_col)) if value_col else np.nan,
                    "effect",
                    path,
                    repo_root,
                    optional_extension=True,
                    stsp_group_quantile=_num(r.get("stsp_group_quantile")),
                    overlap_threshold=_num(r.get("overlap_threshold")),
                    endpoint=r.get("endpoint", ""),
                )
            )
    return _finish(spec, output_dir, root, seeds, rows, sources, checked, warnings, ["metric", "condition"], manifest_extra={"source_mode": source_mode, "optional_extension": True})


def _seed_from_frames(*frames: pd.DataFrame) -> Any:
    for frame in frames:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        for col in ("network_seed", "seed", "seed_id"):
            if col not in frame.columns:
                continue
            values = frame[col].dropna()
            if not values.empty:
                return values.iloc[0]
    return ""


def _score_x(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.upper().startswith("Q"):
        text = text[1:]
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if not digits:
        return np.nan
    try:
        return float(digits)
    except ValueError:
        return np.nan


def _truthy_for_plot(value: Any) -> bool:
    if isinstance(value, str) and value not in {"", "False", "false", "0"}:
        return True
    return _bool_value(value)
