from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.plotting.paper_fig.data_resolver import AdapterResult, summarize_values, write_adapter_outputs
from src.plotting.paper_fig.utils import read_json


DEFAULT_EXPERIMENT_ROOT = "results/paper_figure_multi_seed/fig6_peak_amplified_reentry"
DRAFT_WARNING = "Single-network result. Use for pipeline validation only, not final manuscript statistics."
FIG6_SCORE_NAME = "entry_gated_stsp_gain_score"
FIG6_SCORE_DEFINITION = "mean g_final/g_baseline over entry-active presynaptic sites in each Layer 1 receptive field"
FIG6_SCORE_EXCLUDES = ["connection_weights", "inhibition", "voltage", "threshold", "WTA", "final_label"]
FIG6_PRIMARY_ENDPOINT = "Layer 1 spatial spike enrichment / recruitment"
FIG6_INTERPRETATION_BOUNDARY = (
    "The score predicts spike enrichment in high-score regions, not one-to-one firing or final-label prediction."
)
FIG6_MECHANISM_STATEMENT = "Multi-item STSP fields bias Layer 1 recruitment only where later input enters the high-gain field."
FIG6_FORBIDDEN_CLAIMS = [
    "score predicts final label",
    "deterministic final-label prediction",
    "STSP alone determines firing",
    "high STSP automatically fires without entry",
    "connection weights define the main score",
    "inhibition is part of the STSP score",
    "peak-gated re-entry",
    "route-peak perturbation",
    "peaks = gain",
    "overlap = route",
]

GROUP_LABELS = {
    "single_old": "Single old",
    "single_recent": "Single recent",
    "multi_old": "Multi old",
    "multi_recent": "Multi recent",
}
MODEL_LABELS = {
    "baseline_only": "Base",
    "update_only": "Update",
    "recency_only": "Recency",
    "overlap_only": "Overlap",
    "update_plus_recency": "Upd+Rec",
    "update_times_recency": "Upd x Rec",
}
DOWNSTREAM_LABELS = {
    "early_recruitment_gain": "Early recruitment",
    "response_pattern_displacement": "Response reshaping",
    "decision_deflection_score": "Decision deflection",
    "early_recruitment_gain_real": "Early recruitment",
    "response_pattern_displacement_real": "Response reshaping",
    "decision_deflection_score_real": "Decision deflection",
}


def build_fig6_entry_score_metadata_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Fig.6A/F metadata for the entry-gated STSP gain score and interpretation boundary."""
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    checked_paths: list[Path] = []
    metric = str(spec.get("metric") or FIG6_SCORE_NAME)
    for seed_dir in seeds:
        gain_path = seed_dir / "data" / "metrics" / "fig6_gain_ratio_audit.csv"
        entry_path = seed_dir / "data" / "metrics" / "fig6_entry_score_audit.csv"
        checked_paths.extend([gain_path, entry_path])
        if not gain_path.exists() or not entry_path.exists():
            missing = [str(path.relative_to(seed_dir)) for path in (gain_path, entry_path) if not path.exists()]
            warnings.append(f"{seed_dir.name}: missing required Fig.6 STSP audit files {missing}")
            continue
        gain_df = _read_csv_with_warning(gain_path, warnings, "Fig.6 gain-ratio audit")
        entry_df = _read_csv_with_warning(entry_path, warnings, "Fig.6 entry-score audit")
        if gain_df is None or entry_df is None:
            continue
        source_paths.extend([gain_path, entry_path])
        rows.append(
            _row(
                figure_id,
                panel_id,
                metric,
                "metadata",
                "layer1",
                seed_dir,
                _audit_seed_fallback(gain_df, entry_df),
                1.0,
                "present",
                entry_path,
                repo_root,
                score_name=FIG6_SCORE_NAME,
                score_definition=FIG6_SCORE_DEFINITION,
                score_excludes="; ".join(FIG6_SCORE_EXCLUDES),
                primary_endpoint=FIG6_PRIMARY_ENDPOINT,
                interpretation_boundary=FIG6_INTERPRETATION_BOUNDARY,
                gain_ratio_audit_rows=int(len(gain_df)),
                entry_score_audit_rows=int(len(entry_df)),
                source_level="entry_gated_stsp_gain_score_metadata",
                final_label_claim=False,
            )
        )
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        source_paths,
        checked_paths,
        warnings,
        ["metric", "condition"],
        stats_extra=_fig6_score_stats(metric, "entry_gated_stsp_gain_score_metadata"),
        manifest_extra=_fig6_score_manifest("entry_gated_stsp_gain_score_metadata"),
    )


def build_fig6_high_stsp_overlap_ablation_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Fig.6D: aggregate high-STSP-overlap ablation to one row per network and condition."""
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, df, candidates = _first_existing(
            seed_dir,
            [
                "data/metrics/panel_a_high_stsp_overlap_ablation_summary.csv",
                "data/metrics/panel_f_high_stsp_overlap_ablation_summary.csv",
            ],
        )
        checked.extend(candidates)
        if path is None or df is None or df.empty:
            warnings.append(f"Missing Fig.6D high-STSP-overlap ablation summary under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        value_col = _first_col(df, ["loss_delta_spike_probability", "delta_spike_probability_loss", "loss"])
        condition_col = _first_col(df, ["loss_condition", "condition", "ablation_condition"])
        if value_col is None or condition_col is None:
            warnings.append(f"Fig.6D ablation source lacks value/condition columns: {_rel(path, repo_root)}")
            continue
        work = df.copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=[value_col])
        if work.empty:
            continue
        for (network_seed, condition), part in work.groupby(["network_seed", condition_col], dropna=False, sort=False):
            values = pd.to_numeric(part[value_col], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "loss_delta_spike_probability",
                    condition,
                    "layer1",
                    seed_dir,
                    network_seed,
                    float(values.mean()),
                    "probability difference",
                    path,
                    repo_root,
                    early_window_ms=_num(part.get("early_window_ms", pd.Series([np.nan])).iloc[0]),
                    removed_active_area=_num(pd.to_numeric(part.get("removed_active_area", pd.Series(dtype=float)), errors="coerce").mean()),
                    removed_input_energy=_num(pd.to_numeric(part.get("removed_input_energy", pd.Series(dtype=float)), errors="coerce").mean()),
                    lower_level_rows=int(len(values)),
                    aggregation="sequence_probe_to_network_condition",
                    analysis_role="network_condition_mean",
                    score_name=FIG6_SCORE_NAME,
                    primary_endpoint=FIG6_PRIMARY_ENDPOINT,
                    final_label_claim=False,
                    high_stsp_alone_sufficient=False,
                    legacy_source_alias=path.name.startswith("panel_f_"),
                )
            )
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        sources,
        checked,
        warnings,
        ["condition", "metric"],
        stats_extra={
            **_fig6_score_stats("loss_delta_spike_probability", "high_stsp_overlap_ablation_vs_matched_removal"),
            "source_level": "high_stsp_overlap_ablation",
            "adapter_performed_network_level_averaging": True,
            "inferential_unit": "independent network",
            "replicate_unit": "network_id",
            "interval_definition": "two-sided 95% Student-t confidence interval across independent networks",
            "aggregation": "sequence_probe_to_network_condition",
        },
        manifest_extra={
            **_fig6_score_manifest("high_stsp_overlap_ablation_vs_matched_removal"),
            "source_mode": "high_stsp_overlap_ablation",
            "canonical_source": "data/metrics/panel_a_high_stsp_overlap_ablation_summary.csv",
            "legacy_alias_source": "data/metrics/panel_f_high_stsp_overlap_ablation_summary.csv",
            "source_aggregation": "sequence_probe_to_network_condition",
        },
    )


def build_fig6_region_ping_readout_bias_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Fig.6B: region-gated ping readout mass moved from the Fig.3 ping result."""
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    checked_paths: list[Path] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_b_region_ping_readout_bias.csv"
        checked_paths.append(path)
        df = _read_required_fig6_csv(path, warnings, "Fig.6B region ping readout bias")
        if df is None:
            continue
        source_paths.append(path)
        for _, r in df.iterrows():
            condition = _entry_condition_label(r.get("entry_condition", r.get("condition", "")))
            if "other_mass" not in df.columns:
                warnings.append(f"{seed_dir.name}: Fig.6B source lacks optional other_mass column.")
            for metric in ("old_mass", "middle_mass", "recent_mass", "other_mass", "silent_rate"):
                if metric not in df.columns:
                    continue
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        metric,
                        condition,
                        "readout",
                        seed_dir,
                        r.get("network_seed", ""),
                        _num(r.get(metric)),
                        "fraction",
                        path,
                        repo_root,
                        sequence_id=r.get("sequence_id", ""),
                        n_trials=_num(r.get("n_trials")),
                        ping_active_sites=_num(r.get("ping_active_sites")),
                        total_ping_current=_num(r.get("total_ping_current")),
                        entry_condition=r.get("entry_condition", condition),
                    )
                )
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        source_paths,
        checked_paths,
        warnings,
        ["condition", "metric"],
        stats_extra={"main_metric": "serial_readout_mass", "claim": "field_entry_topology_exposes_serial_content_bias", "mask_basis": "rho_based"},
        manifest_extra={"main_metric": "serial_readout_mass", "claim": "field_entry_topology_exposes_serial_content_bias", "mask_basis": "rho_based"},
    )


def build_fig6_ping_score_spike_prediction_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Fig.6C: ping score quantiles versus early Layer 1 spike recruitment."""
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    checked_paths: list[Path] = []
    metric_units = {
        "spike_probability": "probability",
        "mean_early_spike_count": "spike count",
        "mean_first_spike_latency_ms": "ms",
        "fired_site_score_percentile_mean": "percentile",
    }
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_c_ping_score_spike_prediction.csv"
        checked_paths.append(path)
        df = _read_required_fig6_csv(path, warnings, "Fig.6C ping score spike prediction")
        if df is None:
            continue
        source_paths.append(path)
        for row_idx, r in df.iterrows():
            condition = _entry_condition_label(r.get("entry_condition", r.get("condition", "")))
            x_value = _score_quantile_x(r.get("score_quantile_bin", row_idx), row_idx)
            for metric, unit in metric_units.items():
                if metric not in df.columns:
                    continue
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        metric,
                        condition,
                        "layer1",
                        seed_dir,
                        r.get("network_seed", ""),
                        _num(r.get(metric)),
                        unit,
                        path,
                        repo_root,
                        sequence_id=r.get("sequence_id", ""),
                        entry_condition=r.get("entry_condition", condition),
                        early_window_ms=_num(r.get("early_window_ms")),
                        score_quantile_bin=r.get("score_quantile_bin", ""),
                        x_value=x_value,
                        mean_score=_num(r.get("mean_score")),
                        n_sites=_num(r.get("n_sites")),
                        fired_site_count=_num(r.get("fired_site_count")),
                        shuffled_baseline_value=_num(r.get("shuffled_baseline_value")),
                        score_name=FIG6_SCORE_NAME,
                        primary_endpoint=FIG6_PRIMARY_ENDPOINT,
                    )
                )
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        source_paths,
        checked_paths,
        warnings,
        ["condition", "metric", "score_quantile_bin"],
        stats_extra=_fig6_score_stats("spike_probability_by_score_quantile", "ping_score_predicts_l1_recruitment"),
        manifest_extra=_fig6_score_manifest("ping_score_predicts_l1_recruitment"),
    )


def build_fig6_global_ping_score_spike_prediction_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Fig.6C: global-ping STSP score quantiles versus early Layer 1 spike recruitment."""
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    checked_paths: list[Path] = []
    metric_units = {
        "spike_probability": "probability",
        "mean_early_spike_count": "spike count",
        "mean_first_spike_latency_ms": "ms",
        "fired_site_score_percentile_mean": "percentile",
    }
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_c_global_ping_score_spike_prediction.csv"
        checked_paths.append(path)
        df = _read_required_fig6_csv(path, warnings, "Fig.6C global-ping score spike prediction")
        if df is None:
            continue
        source_paths.append(path)
        for row_idx, r in df.iterrows():
            x_value = _score_quantile_x(r.get("score_quantile_bin", row_idx), row_idx)
            for metric, unit in metric_units.items():
                if metric not in df.columns:
                    continue
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        metric,
                        "Global ping",
                        "layer1",
                        seed_dir,
                        r.get("network_seed", ""),
                        _num(r.get(metric)),
                        unit,
                        path,
                        repo_root,
                        sequence_id=r.get("sequence_id", ""),
                        early_window_ms=_num(r.get("early_window_ms")),
                        score_quantile_bin=r.get("score_quantile_bin", ""),
                        x_value=x_value,
                        mean_score=_num(r.get("mean_score")),
                        n_sites=_num(r.get("n_sites")),
                        fired_site_count=_num(r.get("fired_site_count")),
                        score_name=FIG6_SCORE_NAME,
                        primary_endpoint=FIG6_PRIMARY_ENDPOINT,
                        primary_endpoint_detail="Layer 1 spatial spike enrichment / recruitment",
                    )
                )
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        source_paths,
        checked_paths,
        warnings,
        ["condition", "metric", "score_quantile_bin"],
        stats_extra={
            **_fig6_score_stats("spike_probability_by_score_quantile", "global_ping_score_predicts_l1_recruitment"),
            "entry_type": "global_ping",
        },
        manifest_extra={
            **_fig6_score_manifest("global_ping_score_predicts_l1_recruitment"),
            "entry_type": "global_ping",
            "source_file_contract": "data/metrics/panel_c_global_ping_score_spike_prediction.csv",
        },
    )


def build_fig6_real_probe_score_spike_deflection_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Fig.6D: real-probe score quantiles versus Layer 1 firing deflection."""
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    checked_paths: list[Path] = []
    raw_rows_read = 0
    metric_units = {
        "delta_spike_probability": "probability difference",
        "mean_delta_spike_count": "spike count difference",
        "recruit_probability": "probability",
        "advance_probability": "probability",
    }
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_d_real_probe_score_spike_deflection.csv"
        checked_paths.append(path)
        df = _read_required_fig6_csv(path, warnings, "Fig.6D real-probe score spike deflection")
        if df is None:
            continue
        source_paths.append(path)
        raw_rows_read += int(len(df))
        df = df.copy()
        if "network_seed" not in df.columns:
            df["network_seed"] = _seed_id(seed_dir)
        if "score_quantile_bin" not in df.columns:
            df["score_quantile_bin"] = ""
        if "early_window_ms" not in df.columns:
            df["early_window_ms"] = math.nan
        numeric_cols = [
            col
            for col in [
                *metric_units.keys(),
                "mean_score",
                "n_sites",
                "dynamic_spike_probability",
                "baseline_spike_probability",
                "valid_site_count",
                "probe_active_area",
                "prior_updated_overlap_area",
            ]
            if col in df.columns
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.groupby(["network_seed", "score_quantile_bin", "early_window_ms"], dropna=False, as_index=False)[numeric_cols].mean()
        for row_idx, r in df.iterrows():
            x_value = _score_quantile_x(r.get("score_quantile_bin", row_idx), row_idx)
            for metric, unit in metric_units.items():
                if metric not in df.columns:
                    continue
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        metric,
                        "Real probe",
                        "layer1",
                        seed_dir,
                        r.get("network_seed", ""),
                        _num(r.get(metric)),
                        unit,
                        path,
                        repo_root,
                        sequence_id=r.get("sequence_id", ""),
                        probe_id=r.get("probe_id", ""),
                        probe_label=r.get("probe_label", ""),
                        early_window_ms=_num(r.get("early_window_ms")),
                        score_quantile_bin=r.get("score_quantile_bin", ""),
                        x_value=x_value,
                        mean_score=_num(r.get("mean_score")),
                        n_sites=_num(r.get("n_sites")),
                        dynamic_spike_probability=_num(r.get("dynamic_spike_probability")),
                        baseline_spike_probability=_num(r.get("baseline_spike_probability")),
                        valid_site_count=_num(r.get("valid_site_count")),
                        probe_active_area=_num(r.get("probe_active_area")),
                        prior_updated_overlap_area=_num(r.get("prior_updated_overlap_area")),
                        score_name=FIG6_SCORE_NAME,
                        primary_endpoint=FIG6_PRIMARY_ENDPOINT,
                        source_aggregation="network_score_quantile_window_mean",
                    )
                )
    stats_extra = _fig6_score_stats("delta_spike_probability_by_score_quantile", "real_probe_score_predicts_l1_spike_deflection")
    stats_extra["baseline"] = "S0 or detected baseline"
    stats_extra["adapter_performed_network_level_averaging"] = True
    stats_extra["raw_source_rows_read_before_seed_grouping"] = raw_rows_read
    manifest_extra = _fig6_score_manifest("real_probe_score_predicts_l1_spike_deflection")
    manifest_extra["baseline"] = "S0 or detected baseline"
    manifest_extra["source_file_contract"] = "data/metrics/panel_d_real_probe_score_spike_deflection.csv"
    manifest_extra["source_aggregation"] = "network_score_quantile_window_mean"
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        source_paths,
        checked_paths,
        warnings,
        ["condition", "metric", "score_quantile_bin"],
        stats_extra=stats_extra,
        manifest_extra=manifest_extra,
    )


def build_fig6_score_basin_sparsification_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Fig.6E: sparse Layer 1 firing remains enriched in high-score basins."""
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    checked_paths: list[Path] = []
    metric_units = {
        "fired_site_score_percentile_mean": "percentile",
        "high_score_basin_hit_rate": "fraction",
        "enrichment_over_shuffle": "fold",
        "shuffled_hit_rate": "fraction",
    }
    basin_radii: list[float] = []
    top_quantiles: list[float] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_e_score_basin_sparsification.csv"
        overlay_path = seed_dir / "data" / "raw" / "panel_e_example_score_spike_overlay.npz"
        checked_paths.extend([path, overlay_path])
        df = _read_required_fig6_csv(path, warnings, "Fig.6E score-basin sparsification")
        if df is None:
            continue
        source_paths.append(path)
        if overlay_path.exists():
            source_paths.append(overlay_path)
        for _, r in df.iterrows():
            condition = _basin_condition_label(r)
            basin_radii.append(_num(r.get("basin_radius")))
            top_quantiles.append(_num(r.get("top_score_quantile")))
            for metric, unit in metric_units.items():
                if metric not in df.columns:
                    continue
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        metric,
                        condition,
                        "layer1",
                        seed_dir,
                        r.get("network_seed", ""),
                        _num(r.get(metric)),
                        unit,
                        path,
                        repo_root,
                        sequence_id=r.get("sequence_id", ""),
                        entry_type=r.get("entry_type", ""),
                        entry_condition=r.get("entry_condition", ""),
                        basin_radius=_num(r.get("basin_radius")),
                        top_score_quantile=_num(r.get("top_score_quantile")),
                        n_fired_sites=_num(r.get("n_fired_sites")),
                        fired_site_score_percentile_sem=_num(r.get("fired_site_score_percentile_sem")),
                        score_name=FIG6_SCORE_NAME,
                        primary_endpoint=FIG6_PRIMARY_ENDPOINT,
                    )
                )
    stats_extra = _fig6_score_stats(
        "fired_site_score_percentile_mean",
        "actual_fired_sites_are_enriched_at_high_stsp_score_percentiles",
    )
    stats_extra["reference_percentile"] = 0.5
    stats_extra["basin_radius"] = _first_finite(basin_radii)
    stats_extra["top_score_quantile"] = _first_finite(top_quantiles)
    manifest_extra = _fig6_score_manifest("actual_fired_sites_are_enriched_at_high_stsp_score_percentiles")
    manifest_extra["main_metric"] = "fired_site_score_percentile_mean"
    manifest_extra["reference_percentile"] = 0.5
    manifest_extra["optional_raw"] = "data/raw/panel_e_example_score_spike_overlay.npz"
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        source_paths,
        checked_paths,
        warnings,
        ["condition", "metric"],
        stats_extra=stats_extra,
        manifest_extra=manifest_extra,
    )


def build_fig6_overlap_gated_stsp_recruitment_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Fig.6E: high/low STSP by probe-overlap 2x2 recruitment test."""
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    checked_paths: list[Path] = []
    quantiles: list[float] = []
    thresholds: list[float] = []
    expected_quantile = float(spec.get("stsp_group_quantile", 0.50))
    expected_overlap_threshold = float(spec.get("overlap_threshold", 0.05))
    expected_windows = set(int(value) for value in (spec.get("robustness_windows_ms") or [5, 10, 15, 20]))
    raw_recruitment_rows = 0
    raw_interaction_rows = 0
    metric_units = {
        "delta_spike_probability": "probability difference",
        "dynamic_spike_probability": "probability",
        "baseline_spike_probability": "probability",
        "mean_delta_spike_count": "spike count difference",
        "recruit_probability": "probability",
    }
    for seed_dir in seeds:
        recruitment_path = seed_dir / "data" / "metrics" / "panel_e_overlap_gated_stsp_recruitment.csv"
        interaction_path = seed_dir / "data" / "metrics" / "panel_e_overlap_gated_stsp_interaction.csv"
        checked_paths.extend([recruitment_path, interaction_path])
        recruitment_df = _read_required_fig6_csv(recruitment_path, warnings, "Fig.6E overlap-gated STSP recruitment")
        interaction_df = _read_required_fig6_csv(interaction_path, warnings, "Fig.6E overlap-gated STSP interaction")
        if recruitment_df is not None:
            source_paths.append(recruitment_path)
            raw_recruitment_rows += int(len(recruitment_df))
            recruitment_df = recruitment_df.copy()
            _require_fig6_constant(
                recruitment_df,
                "stsp_group_quantile",
                expected_quantile,
                recruitment_path,
            )
            _require_fig6_constant(
                recruitment_df,
                "overlap_threshold",
                expected_overlap_threshold,
                recruitment_path,
            )
            if "network_seed" not in recruitment_df.columns:
                recruitment_df["network_seed"] = _seed_id(seed_dir)
            if "early_window_ms" not in recruitment_df.columns:
                recruitment_df["early_window_ms"] = math.nan
            recruitment_df["stsp_group_norm"] = recruitment_df.get("stsp_group", pd.Series([""] * len(recruitment_df))).map(_stsp_group_label)
            recruitment_df["overlap_group_norm"] = recruitment_df.get("overlap_group", pd.Series([""] * len(recruitment_df))).map(_overlap_group_label)
            invalid_groups = recruitment_df["stsp_group_norm"].astype(str).eq("") | recruitment_df["overlap_group_norm"].astype(str).eq("")
            if bool(invalid_groups.any()):
                warnings.append(f"{seed_dir.name}: Fig.6E recruitment excluded {int(invalid_groups.sum())} rows without STSP/overlap groups.")
                recruitment_df = recruitment_df.loc[~invalid_groups].copy()
            numeric_cols = [
                col
                for col in [
                    *metric_units.keys(),
                    "n_sites",
                    "mean_local_stsp_score",
                    "mean_probe_overlap",
                    "stsp_group_quantile",
                    "overlap_threshold",
                ]
                if col in recruitment_df.columns
            ]
            for col in numeric_cols:
                recruitment_df[col] = pd.to_numeric(recruitment_df[col], errors="coerce")
            if not recruitment_df.empty:
                recruitment_df = recruitment_df.groupby(
                    ["network_seed", "stsp_group_norm", "overlap_group_norm", "early_window_ms"],
                    dropna=False,
                    as_index=False,
                )[numeric_cols].mean()
            for _, r in recruitment_df.iterrows():
                stsp_group = str(r.get("stsp_group_norm", ""))
                overlap_group = str(r.get("overlap_group_norm", ""))
                quantiles.append(_num(r.get("stsp_group_quantile")))
                thresholds.append(_num(r.get("overlap_threshold")))
                condition = f"{stsp_group.title()} STSP + {'Overlap' if overlap_group == 'overlap' else 'No overlap'}"
                for metric, unit in metric_units.items():
                    if metric not in recruitment_df.columns:
                        continue
                    rows.append(
                        _row(
                            figure_id,
                            panel_id,
                            metric,
                            condition,
                            "layer1",
                            seed_dir,
                            r.get("network_seed", ""),
                            _num(r.get(metric)),
                            unit,
                            recruitment_path,
                            repo_root,
                            sequence_id=r.get("sequence_id", ""),
                            probe_id=r.get("probe_id", ""),
                            probe_label=r.get("probe_label", ""),
                            early_window_ms=_num(r.get("early_window_ms")),
                            stsp_group=stsp_group,
                            overlap_group=overlap_group,
                            x_group=overlap_group,
                            hue_group=stsp_group,
                            n_sites=_num(r.get("n_sites")),
                            mean_local_stsp_score=_num(r.get("mean_local_stsp_score")),
                            mean_probe_overlap=_num(r.get("mean_probe_overlap")),
                            dynamic_spike_probability=_num(r.get("dynamic_spike_probability")),
                            baseline_spike_probability=_num(r.get("baseline_spike_probability")),
                            mean_delta_spike_count=_num(r.get("mean_delta_spike_count")),
                            recruit_probability=_num(r.get("recruit_probability")),
                            stsp_group_quantile=_num(r.get("stsp_group_quantile")),
                            overlap_threshold=_num(r.get("overlap_threshold")),
                            score_name=FIG6_SCORE_NAME,
                            primary_endpoint=FIG6_PRIMARY_ENDPOINT,
                            source_aggregation="network_stsp_overlap_window_mean",
                        )
                    )
        if interaction_df is None:
            continue
        source_paths.append(interaction_path)
        raw_interaction_rows += int(len(interaction_df))
        interaction_df = interaction_df.copy()
        _require_fig6_constant(
            interaction_df,
            "stsp_group_quantile",
            expected_quantile,
            interaction_path,
        )
        _require_fig6_constant(
            interaction_df,
            "overlap_threshold",
            expected_overlap_threshold,
            interaction_path,
        )
        if "network_seed" not in interaction_df.columns:
            interaction_df["network_seed"] = _seed_id(seed_dir)
        if "early_window_ms" not in interaction_df.columns:
            interaction_df["early_window_ms"] = math.nan
        interaction_numeric_cols = [
            col
            for col in [
                "interaction_delta",
                "stsp_group_quantile",
                "overlap_threshold",
                "stsp_effect_with_overlap",
                "stsp_effect_without_overlap",
                "high_overlap_delta",
                "low_overlap_delta",
                "high_nooverlap_delta",
                "low_nooverlap_delta",
                "n_sites_high_overlap",
                "n_sites_low_overlap",
                "n_sites_high_nooverlap",
                "n_sites_low_nooverlap",
            ]
            if col in interaction_df.columns
        ]
        for col in interaction_numeric_cols:
            interaction_df[col] = pd.to_numeric(interaction_df[col], errors="coerce")
        interaction_df = interaction_df.groupby(["network_seed", "early_window_ms"], dropna=False, as_index=False)[interaction_numeric_cols].mean()
        found_windows = set(
            pd.to_numeric(interaction_df["early_window_ms"], errors="coerce").dropna().astype(int)
        )
        if found_windows != expected_windows:
            raise RuntimeError(
                f"Fig.6E frozen-window mismatch in {interaction_path}: "
                f"expected={sorted(expected_windows)}, found={sorted(found_windows)}"
            )
        for _, r in interaction_df.iterrows():
            quantiles.append(_num(r.get("stsp_group_quantile")))
            thresholds.append(_num(r.get("overlap_threshold")))
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "interaction_delta",
                    "overlap_gated_stsp_interaction",
                    "layer1",
                    seed_dir,
                    r.get("network_seed", ""),
                    _num(r.get("interaction_delta")),
                    "probability difference",
                    interaction_path,
                    repo_root,
                    sequence_id=r.get("sequence_id", ""),
                    probe_id=r.get("probe_id", ""),
                    probe_label=r.get("probe_label", ""),
                    early_window_ms=_num(r.get("early_window_ms")),
                    stsp_group_quantile=_num(r.get("stsp_group_quantile")),
                    overlap_threshold=_num(r.get("overlap_threshold")),
                    stsp_effect_with_overlap=_num(r.get("stsp_effect_with_overlap")),
                    stsp_effect_without_overlap=_num(r.get("stsp_effect_without_overlap")),
                    high_overlap_delta=_num(r.get("high_overlap_delta")),
                    low_overlap_delta=_num(r.get("low_overlap_delta")),
                    high_nooverlap_delta=_num(r.get("high_nooverlap_delta")),
                    low_nooverlap_delta=_num(r.get("low_nooverlap_delta")),
                    n_sites_high_overlap=_num(r.get("n_sites_high_overlap")),
                    n_sites_low_overlap=_num(r.get("n_sites_low_overlap")),
                    n_sites_high_nooverlap=_num(r.get("n_sites_high_nooverlap")),
                    n_sites_low_nooverlap=_num(r.get("n_sites_low_nooverlap")),
                    score_name=FIG6_SCORE_NAME,
                    primary_endpoint=FIG6_PRIMARY_ENDPOINT,
                    source_aggregation="network_interaction_window_mean",
                )
            )
    stats_extra = _fig6_score_stats("overlap_gated_delta_spike_probability", "probe_overlap_gates_high_stsp_expression")
    stats_extra.update(
        {
            "interaction_metric": "interaction_delta",
            "stsp_group_quantile": _first_finite(quantiles),
            "overlap_threshold": _first_finite(thresholds),
            "frozen_protocol_id": str(spec.get("protocol_id", "")),
            "expected_stsp_group_quantile": expected_quantile,
            "expected_overlap_threshold": expected_overlap_threshold,
            "expected_windows_ms": sorted(expected_windows),
            "adapter_performed_network_level_averaging": True,
            "raw_recruitment_rows_read_before_seed_grouping": raw_recruitment_rows,
            "raw_interaction_rows_read_before_seed_grouping": raw_interaction_rows,
        }
    )
    manifest_extra = _fig6_score_manifest("probe_overlap_gates_high_stsp_expression")
    manifest_extra.update(
        {
            "interaction_metric": "interaction_delta",
            "stsp_group_quantile": _first_finite(quantiles),
            "overlap_threshold": _first_finite(thresholds),
            "frozen_protocol_id": str(spec.get("protocol_id", "")),
            "expected_stsp_group_quantile": expected_quantile,
            "expected_overlap_threshold": expected_overlap_threshold,
            "expected_windows_ms": sorted(expected_windows),
            "source_file_contract": [
                "data/metrics/panel_e_overlap_gated_stsp_recruitment.csv",
                "data/metrics/panel_e_overlap_gated_stsp_interaction.csv",
            ],
            "source_aggregation": {
                "recruitment": "network_stsp_overlap_window_mean",
                "interaction": "network_interaction_window_mean",
            },
        }
    )
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        source_paths,
        checked_paths,
        warnings,
        ["metric", "stsp_group", "overlap_group"],
        stats_extra=stats_extra,
        manifest_extra=manifest_extra,
    )


def build_fig6_peak_source_attribution_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Main Fig.6A: leave-one-item-out source attribution by temporal position."""
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, _, candidates = _first_existing(
            seed_dir,
            [
                "data/metrics/panel_a_peak_source_attribution_summary.csv",
                "data/raw/panel_a_peak_source_attribution.csv",
                "data/metrics/panel_a_peak_source_attribution.csv",
                "data/metrics/supp_s11_leave_one_out_source_details.csv",
            ],
        )
        checked.extend(candidates)
        if path is None:
            warnings.append(f"Missing Fig.6 source-attribution source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        value_col = _first_col(df, ["mean_peak_loss_fraction", "peak_loss_fraction", "peak_loss"])
        if value_col is None:
            warnings.append(f"{_rel(path, repo_root)} lacks peak-loss fields.")
            continue
        for _, r in df.iterrows():
            pos = _position_from_end(r)
            sem_value = r.get("sem_peak_loss_fraction", r.get("sem", ""))
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "peak_loss_fraction",
                    "Peak source contribution",
                    "layer1",
                    seed_dir,
                    r.get("network_seed", ""),
                    _num(r.get(value_col)),
                    "fraction",
                    path,
                    repo_root,
                    sequence_id=r.get("sequence_id", ""),
                    seq_len=r.get("seq_len", ""),
                    removed_position=r.get("removed_position", ""),
                    position_from_end=pos,
                    relative_position_from_end=r.get("relative_position_from_end", ""),
                    x_value=pos,
                    y_value=_num(r.get(value_col)),
                    n_sequences=r.get("n_sequences", ""),
                    n_units=r.get("n_units", ""),
                    sem=_num(sem_value),
                )
            )
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        sources,
        checked,
        warnings,
        ["metric", "position_from_end"],
        stats_extra={"main_metric": "peak_loss_fraction", "source_attribution_mode": "leave_one_item_out", "claim": "late_updates_source_final_peaks"},
        manifest_extra={"main_metric": "peak_loss_fraction", "source_attribution_mode": "leave_one_item_out", "claim": "late_updates_source_final_peaks"},
    )


def build_fig6_peak_update_history_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Main Fig.6B: update count -> P(peak)."""
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, _, candidates = _first_existing(
            seed_dir,
            [
                "data/raw/panel_b_peak_update_history.csv",
                "data/metrics/panel_b_peak_update_history.csv",
                "data/metrics/panel_b_peak_update_history_summary.csv",
                "data/metrics/supp_s11_peak_update_group_enrichment.csv",
            ],
        )
        checked.extend(candidates)
        if path is None:
            warnings.append(f"Missing Fig.6 update-history source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        if {"update_count", "is_peak"}.issubset(df.columns):
            rows.extend(_p_peak_by_update_count_rows(figure_id, panel_id, seed_dir, path, repo_root, df))
        else:
            warnings.append(f"{_rel(path, repo_root)} lacks update_count/is_peak rows for main Fig.6B.")
    bins = list(dict.fromkeys(str(row.get("update_count_bin", "")) for row in rows))
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        sources,
        checked,
        warnings,
        ["metric", "update_count_bin"],
        stats_extra={"main_metric": "P_peak_by_update_count", "binned_update_count": any(str(b).endswith("+") for b in bins), "update_count_bins": bins, "claim": "repeated_updates_enrich_final_peaks"},
        manifest_extra={"main_metric": "P_peak_by_update_count", "binned_update_count": any(str(b).endswith("+") for b in bins), "update_count_bins": bins, "claim": "repeated_updates_enrich_final_peaks"},
    )


def build_fig6_peak_input_overlap_origin_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return build_fig6_peak_overlap_alignment_adapter(spec, repo_root, output_dir)


def build_fig6_peak_overlap_alignment_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Main Fig.6C: final peaks aligned to old/all/recent encoded-entry overlap maps."""
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, _, candidates = _first_existing(
            seed_dir,
            [
                "data/metrics/panel_c_peak_input_overlap_similarity_summary.csv",
                "data/metrics/panel_c_peak_input_overlap_similarity.csv",
                "data/metrics/supp_s11_recent_overlap_window_robustness.csv",
                "data/metrics/supp_recent_overlap_window_robustness.csv",
            ],
        )
        checked.extend(candidates)
        if path is None:
            warnings.append(f"Missing Fig.6 peak-origin overlap source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        rows.extend(_peak_overlap_alignment_rows(figure_id, panel_id, seed_dir, path, repo_root, df))
    selected_windows = list(dict.fromkeys(str(row.get("overlap_window", "")) for row in rows))
    recent_k = next((row.get("window_k") for row in rows if row.get("window_family") == "recent"), None)
    old_k = next((row.get("window_k") for row in rows if row.get("window_family") == "old"), None)
    recent_vals = [row["value"] for row in rows if row.get("window_family") == "recent" and np.isfinite(float(row["value"]))]
    all_vals = [row["value"] for row in rows if row.get("window_family") == "all" and np.isfinite(float(row["value"]))]
    old_vals = [row["value"] for row in rows if row.get("window_family") == "old" and np.isfinite(float(row["value"]))]
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        sources,
        checked,
        warnings,
        ["metric", "overlap_window"],
        stats_extra={
            "main_metric": "peak_coverage",
            "selected_windows": selected_windows,
            "selected_recent_k": recent_k,
            "selected_old_k": old_k,
            "recent_minus_all": (float(np.mean(recent_vals) - np.mean(all_vals)) if recent_vals and all_vals else None),
            "recent_minus_old": (float(np.mean(recent_vals) - np.mean(old_vals)) if recent_vals and old_vals else None),
            "peak_alignment_mode": "foreground_overlap",
            "claim": "peaks_align_with_recent_high_foreground_overlap",
        },
        manifest_extra={"main_metric": "peak_coverage", "selected_windows": selected_windows, "peak_alignment_mode": "foreground_overlap", "claim": "peaks_align_with_recent_high_foreground_overlap"},
    )


def build_fig6_real_peak_overlap_reentry_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return build_fig6_peak_weighted_real_reentry_adapter(spec, repo_root, output_dir)


def build_fig6_real_peak_overlap_downstream_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return build_fig6_peak_weighted_real_downstream_adapter(spec, repo_root, output_dir)


def build_fig6_route_peak_reentry_loss_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_d_route_peak_reentry_loss_summary.csv"
        contrast_path = seed_dir / "data" / "metrics" / "panel_d_route_peak_reentry_loss_contrast.csv"
        audit_path = seed_dir / "data" / "metrics" / "panel_d_route_peak_perturbation_audit.csv"
        scientific_audit_path = seed_dir / "data" / "metrics" / "panel_de_route_peak_perturbation_scientific_use_audit.csv"
        checked.extend([path, contrast_path, audit_path, scientific_audit_path])
        df = _read_required_route_peak_csv(path, "Fig.6D route-peak re-entry summary")
        _read_required_route_peak_csv(contrast_path, "Fig.6D route-peak re-entry contrast")
        audit = _read_required_route_peak_csv(audit_path, "Fig.6D route-peak perturbation audit")
        scientific_audit = _read_required_route_peak_csv(scientific_audit_path, "Fig.6D/E route-peak scientific-use audit")
        _validate_route_peak_summary(df, panel="D", value_col="mean_normalized_reentry_loss")
        _validate_route_peak_audit(audit, "Fig.6D route-peak perturbation audit")
        _validate_route_peak_audit(scientific_audit, "Fig.6D/E route-peak scientific-use audit")
        sources.extend([path, contrast_path, audit_path, scientific_audit_path])
        rows.extend(_route_peak_reentry_rows(figure_id, panel_id, seed_dir, path, repo_root, df))
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        sources,
        checked,
        warnings,
        ["metric", "perturbation_unit_set"],
        stats_extra={"main_metric": "normalized_reentry_loss", "source_level": "route_peak_perturbation", "claim": "route_peak_perturbation_reduces_reentry"},
        manifest_extra={"source_mode": "route_peak_perturbation", "main_metric": "normalized_reentry_loss", "claim_strength": _claim_strength(seeds)},
    )


def build_fig6_route_peak_downstream_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_e_route_peak_downstream_summary.csv"
        contrast_path = seed_dir / "data" / "metrics" / "panel_e_route_peak_downstream_contrast.csv"
        dist_path = seed_dir / "data" / "metrics" / "panel_e_route_peak_output_distribution.csv"
        audit_path = seed_dir / "data" / "metrics" / "panel_de_route_peak_perturbation_scientific_use_audit.csv"
        checked.extend([path, contrast_path, dist_path, audit_path])
        df = _read_required_route_peak_csv(path, "Fig.6E route-peak downstream summary")
        _read_required_route_peak_csv(contrast_path, "Fig.6E route-peak downstream contrast")
        _read_required_route_peak_csv(dist_path, "Fig.6E route-peak output distribution")
        audit = _read_required_route_peak_csv(audit_path, "Fig.6D/E route-peak scientific-use audit")
        _validate_route_peak_summary(df, panel="E", value_col="P_output_switch")
        _validate_route_peak_audit(audit, "Fig.6D/E route-peak scientific-use audit")
        sources.extend([path, contrast_path, dist_path, audit_path])
        rows.extend(_route_peak_downstream_rows(figure_id, panel_id, seed_dir, path, repo_root, df))
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        sources,
        checked,
        warnings,
        ["metric", "perturbation_unit_set"],
        stats_extra={"main_metric": "P_output_switch", "source_level": "route_peak_perturbation", "claim": "route_peak_perturbation_changes_downstream_output"},
        manifest_extra={"source_mode": "route_peak_perturbation", "main_metric": "P_output_switch", "claim_strength": _claim_strength(seeds)},
    )


def build_fig6_multi_recent_peak_enrichment_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, _, candidates = _first_existing(
            seed_dir,
            [
                "data/metrics/panel_a_peak_enrichment_summary.csv",
                "data/metrics/panel_a_multi_recent_peak_enrichment.csv",
                "data/metrics/supp_legacy_panel_a_peak_enrichment_summary.csv",
                "data/metrics/supp_legacy_panel_a_multi_recent_peak_enrichment.csv",
                "data/metrics/supp_s11_peak_update_group_enrichment.csv",
            ],
        )
        checked.extend(candidates)
        if path is None:
            warnings.append(f"Missing Fig.6A multi-recent peak enrichment source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        group_col = _first_col(df, ["update_history_group", "update_group"])
        if group_col is None:
            warnings.append(f"{_rel(path, repo_root)} lacks update-history group columns.")
            continue
        if "P_peak" in df.columns:
            for _, r in df.iterrows():
                group = str(r.get(group_col, ""))
                if group not in GROUP_LABELS:
                    continue
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        "P_peak",
                        GROUP_LABELS[group],
                        "layer1",
                        seed_dir,
                        r.get("network_seed", ""),
                        _num(r.get("P_peak")),
                        "fraction",
                        path,
                        repo_root,
                        update_history_group=group,
                        n_units=r.get("n_units", ""),
                        mean_delta_support=_num(r.get("mean_delta_support")),
                        peak_enrichment=_num(r.get("peak_enrichment")),
                    )
                )
        elif {"is_peak", group_col}.issubset(df.columns):
            for group, part in df.groupby(group_col, sort=False):
                group_text = str(group)
                if group_text not in GROUP_LABELS:
                    continue
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        "P_peak",
                        GROUP_LABELS[group_text],
                        "layer1",
                        seed_dir,
                        part.get("network_seed", pd.Series([""])).iloc[0] if len(part) else "",
                        float(pd.to_numeric(part["is_peak"], errors="coerce").mean()),
                        "fraction",
                        path,
                        repo_root,
                        update_history_group=group_text,
                        n_units=int(len(part)),
                    )
                )
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        sources,
        checked,
        warnings,
        ["metric", "condition"],
        stats_extra={"source_level": "multi_recent_peak_enrichment", "required_group": "multi_recent"},
        manifest_extra={"source_mode": "multi_recent_peak_enrichment"},
    )


def build_fig6_update_recency_model_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, _, candidates = _first_existing(
            seed_dir,
            [
                "data/metrics/panel_b_update_recency_model_metrics.csv",
                "data/metrics/supp_legacy_panel_b_update_recency_model_metrics.csv",
                "data/metrics/supp_update_recency_support_model_metrics.csv",
                "data/metrics/supp_s11_update_recency_model_comparison.csv",
            ],
        )
        checked.extend(candidates)
        if path is None:
            warnings.append(f"Missing Fig.6B update-recency model source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        df = pd.read_csv(path)
        target_col = _first_col(df, ["target", "dependent_metric"])
        if target_col is not None:
            preferred = df[df[target_col].astype(str).eq("delta_support")].copy()
            if not preferred.empty:
                df = preferred
        value_col = "cv_r2" if "cv_r2" in df.columns else "r2" if "r2" in df.columns else None
        if value_col is None or "model_name" not in df.columns:
            warnings.append(f"{_rel(path, repo_root)} lacks model_name and CV/R2 columns.")
            continue
        for _, r in df.iterrows():
            model = str(r.get("model_name", ""))
            if model not in MODEL_LABELS:
                continue
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "cv_r2",
                    MODEL_LABELS[model],
                    str(r.get("layer", "layer1")),
                    seed_dir,
                    r.get("network_seed", ""),
                    _num(r.get(value_col)),
                    "r2",
                    path,
                    repo_root,
                    sequence_id=r.get("sequence_id", ""),
                    model_name=model,
                    r2=_num(r.get("r2")),
                    cv_r2=_num(r.get("cv_r2")),
                    target=r.get(target_col, "") if target_col else "",
                    n_units=r.get("n_units", r.get("n_samples", "")),
                )
            )
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        sources,
        checked,
        warnings,
        ["metric", "condition", "model_name"],
        stats_extra={"source_level": "update_recency_model", "preferred_target": "delta_support"},
        manifest_extra={"source_mode": "update_recency_model"},
    )


def build_fig6_peak_weighted_overlap_interface_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        path, _, candidates = _first_existing(
            seed_dir,
            [
                "data/metrics/panel_c_peak_weighted_overlap_definitions.csv",
                "data/metrics/supp_legacy_panel_c_peak_weighted_overlap_definitions.csv",
            ],
        )
        example_path = seed_dir / "data" / "raw" / "panel_c_overlap_peak_interface_example.npz"
        checked.extend(candidates + [example_path])
        if path is None:
            warnings.append(f"Missing Fig.6C route/gain interface source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        if example_path.exists():
            sources.append(example_path)
        df = pd.read_csv(path)
        if not {"raw_overlap", "peak_weighted_overlap"}.issubset(df.columns):
            warnings.append(f"{_rel(path, repo_root)} lacks raw_overlap / peak_weighted_overlap.")
            continue
        for _, r in df.iterrows():
            for metric, unit in (
                ("raw_overlap", "fraction"),
                ("peak_weighted_overlap", "weighted fraction"),
                ("peak_overlap_fraction", "fraction"),
                ("nonpeak_overlap_fraction", "fraction"),
            ):
                if metric not in df.columns:
                    continue
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        metric,
                        _metric_label(metric),
                        "layer1",
                        seed_dir,
                        r.get("network_seed", ""),
                        _num(r.get(metric)),
                        unit,
                        path,
                        repo_root,
                        sequence_id=r.get("sequence_id", ""),
                        probe_id=r.get("probe_id", ""),
                        probe_label=r.get("probe_label", ""),
                        raw_overlap=_num(r.get("raw_overlap")),
                        peak_weighted_overlap=_num(r.get("peak_weighted_overlap")),
                        peak_overlap_fraction=_num(r.get("peak_overlap_fraction")),
                        nonpeak_overlap_fraction=_num(r.get("nonpeak_overlap_fraction")),
                        visual_similarity=_num(r.get("visual_similarity")),
                        input_energy=_num(r.get("input_energy")),
                        example_npz=_rel(example_path, repo_root) if example_path.exists() else "",
                    )
                )
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        sources,
        checked,
        warnings,
        ["metric", "condition"],
        stats_extra={"source_level": "route_gain_interface"},
        manifest_extra={"source_mode": "route_gain_interface", "route_statement": "raw overlap", "gain_statement": "peak-weighted overlap"},
    )


def build_fig6_peak_weighted_reentry_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Backward-compatible name for formula/proxy-era peak-weighted re-entry."""
    return build_fig6_peak_weighted_real_reentry_adapter(spec, repo_root, output_dir)


def build_fig6_peak_weighted_real_reentry_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    source_levels: list[str] = []
    for seed_dir in seeds:
        candidates = [
            ("real_matched", "data/metrics/panel_d_raw_overlap_matched_peak_reentry.csv"),
            ("real_regression", "data/metrics/panel_d_real_reentry_metrics.csv"),
            ("real_regression", "data/metrics/panel_d_peak_overlap_reentry_regression.csv"),
            ("peak_weighted_fallback", "data/metrics/panel_d_peak_weighted_reentry_metrics.csv"),
            ("peak_weighted_fallback", "data/metrics/panel_d_matched_raw_overlap_comparison.csv"),
            ("peak_weighted_fallback", "data/metrics/panel_d_peak_weighted_overlap_regression.csv"),
            ("peak_weighted_fallback", "data/metrics/supp_legacy_panel_d_peak_weighted_reentry_metrics.csv"),
            ("peak_weighted_fallback", "data/metrics/supp_legacy_panel_d_matched_raw_overlap_comparison.csv"),
            ("peak_weighted_fallback", "data/metrics/supp_legacy_panel_d_peak_weighted_overlap_regression.csv"),
            ("peak_weighted_fallback", "data/metrics/supp_matched_raw_overlap_peak_comparison.csv"),
            ("peak_weighted_fallback", "data/metrics/supp_s12_raw_overlap_matched_peak_overlap_contrast.csv"),
        ]
        checked.extend(seed_dir / rel for _, rel in candidates)
        chosen_level, path, df = _first_nonempty_by_level(seed_dir, candidates)
        if path is None or df is None:
            warnings.append(f"Missing Fig.6D peak-weighted real re-entry source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        source_levels.append(chosen_level)
        rows.extend(_reentry_rows(figure_id, panel_id, seed_dir, path, repo_root, df, chosen_level))
    source_level = _preferred_level(source_levels, ["real_matched", "real_regression", "peak_weighted_fallback"])
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        sources,
        checked,
        warnings,
        ["condition"],
        stats_extra={"source_level": source_level, "peak_high_minus_low": _high_minus_low(rows), "regression_slope": _regression_slope(rows)},
        manifest_extra={"source_mode": source_level, "claim_strength": _claim_strength(seeds), "raw_overlap_control": _raw_overlap_control(rows)},
    )


def build_fig6_peak_weighted_downstream_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Backward-compatible name for formula/proxy-era peak-weighted downstream outputs."""
    return build_fig6_peak_weighted_real_downstream_adapter(spec, repo_root, output_dir)


def build_fig6_peak_weighted_real_downstream_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    source_levels: list[str] = []
    for seed_dir in seeds:
        candidates = [
            ("real_downstream", "data/metrics/panel_e_real_downstream_metrics.csv"),
            ("real_downstream", "data/metrics/panel_e_peak_overlap_downstream_regression.csv"),
            ("peak_weighted_fallback", "data/metrics/panel_e_peak_weighted_downstream_metrics.csv"),
            ("peak_weighted_fallback", "data/metrics/panel_e_downstream_regression.csv"),
            ("peak_weighted_fallback", "data/metrics/supp_legacy_panel_e_peak_weighted_downstream_metrics.csv"),
            ("peak_weighted_fallback", "data/metrics/supp_legacy_panel_e_downstream_regression.csv"),
            ("peak_weighted_fallback", "data/metrics/supp_s12_downstream_metric_breakdown.csv"),
        ]
        checked.extend(seed_dir / rel for _, rel in candidates)
        chosen_level, path, df = _first_nonempty_by_level(seed_dir, candidates)
        if path is None or df is None:
            warnings.append(f"Missing Fig.6E peak-weighted real downstream source under {_rel(seed_dir, repo_root)}.")
            continue
        sources.append(path)
        source_levels.append(chosen_level)
        rows.extend(_downstream_rows(figure_id, panel_id, seed_dir, path, repo_root, df, chosen_level))
    source_level = _preferred_level(source_levels, ["real_downstream", "peak_weighted_fallback"])
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        sources,
        checked,
        warnings,
        ["metric", "condition", "downstream_node"],
        stats_extra={"source_level": source_level},
        manifest_extra={"source_mode": source_level, "claim_strength": _claim_strength(seeds), "raw_overlap_control": _raw_overlap_control(rows)},
    )


def build_fig6_global_mechanism_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _seed_dirs(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    checked: list[Path] = []
    for seed_dir in seeds:
        metadata_path = seed_dir / "data" / "raw" / "panel_f_global_mechanism_metadata.json"
        summary_path = seed_dir / "summary.json"
        checked.extend([metadata_path, summary_path])
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        summary = read_json(summary_path) if summary_path.exists() else {}
        if metadata_path.exists():
            sources.append(metadata_path)
        if not metadata:
            warnings.append(f"Missing Fig.6F mechanism metadata under {_rel(seed_dir, repo_root)}.")
            continue
        rows.append(
            _row(
                figure_id,
                panel_id,
                "mechanism_statement",
                "overlap_gated_stsp_recruitment",
                "mechanism",
                seed_dir,
                summary.get("network_seed", ""),
                1.0,
                "present",
                metadata_path,
                repo_root,
                mechanism_statement=FIG6_MECHANISM_STATEMENT,
                route_statement="probe entry gates STSP expression",
                gain_statement="entry-gated high-STSP field biases early Layer 1 recruitment",
                score_name=FIG6_SCORE_NAME,
                score_definition=FIG6_SCORE_DEFINITION,
                score_excludes="; ".join(FIG6_SCORE_EXCLUDES),
                primary_endpoint=FIG6_PRIMARY_ENDPOINT,
                final_label_claim=False,
                high_stsp_alone_sufficient=False,
                forbidden_claims="; ".join(FIG6_FORBIDDEN_CLAIMS),
                proxy_mode=_bool_value(summary.get("proxy_mode", False)),
                final_scientific_use=_bool_value(summary.get("final_scientific_use", True)),
                fig6_design_version=str(summary.get("fig6_design_version", "")),
                figure_chain="; ".join(_figure_chain(metadata)),
            )
        )
    return _finish(
        spec,
        output_dir,
        root,
        seeds,
        rows,
        sources,
        checked,
        warnings,
        ["metric", "condition"],
        stats_extra={
            **_fig6_score_stats("mechanism_statement", "overlap_gated_stsp_recruitment_synthesis"),
            "source_level": "global_mechanism",
            "final_label_claim": False,
            "high_stsp_alone_sufficient": False,
        },
        manifest_extra={
            **_fig6_score_manifest("overlap_gated_stsp_recruitment_synthesis"),
            "source_mode": "global_mechanism",
            "mechanism_statement": FIG6_MECHANISM_STATEMENT,
            "final_label_claim": False,
            "high_stsp_alone_sufficient": False,
            "forbidden_claims": FIG6_FORBIDDEN_CLAIMS,
            "pure_mechanism_schematic": True,
        },
    )


def _fig6_score_stats(main_metric: str, claim: str) -> dict[str, Any]:
    return {
        "main_metric": main_metric,
        "claim": claim,
        "score_name": FIG6_SCORE_NAME,
        "score_definition": FIG6_SCORE_DEFINITION,
        "score_excludes": list(FIG6_SCORE_EXCLUDES),
        "primary_endpoint": FIG6_PRIMARY_ENDPOINT,
        "interpretation_boundary": FIG6_INTERPRETATION_BOUNDARY,
    }


def _fig6_score_manifest(claim: str) -> dict[str, Any]:
    return {
        "score_name": FIG6_SCORE_NAME,
        "score_definition": FIG6_SCORE_DEFINITION,
        "score_excludes": list(FIG6_SCORE_EXCLUDES),
        "primary_endpoint": FIG6_PRIMARY_ENDPOINT,
        "interpretation_boundary": FIG6_INTERPRETATION_BOUNDARY,
        "claim": claim,
    }


def _read_required_fig6_csv(path: Path, warnings: list[str], label: str) -> pd.DataFrame | None:
    if not path.exists():
        warnings.append(f"{label} missing required source: {path}")
        return None
    return _read_csv_with_warning(path, warnings, label)


def _require_fig6_constant(
    table: pd.DataFrame,
    column: str,
    expected: float,
    source_path: Path,
    *,
    atol: float = 1e-12,
) -> None:
    if column not in table.columns:
        raise RuntimeError(f"Fig.6 frozen protocol requires column {column!r}: {source_path}")
    values = pd.to_numeric(table[column], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0 or not np.allclose(values, float(expected), atol=atol, rtol=0.0):
        unique = sorted(set(float(value) for value in values))
        raise RuntimeError(
            f"Fig.6 frozen protocol mismatch for {column} in {source_path}: "
            f"expected={expected}, found={unique}"
        )


def _read_csv_with_warning(path: Path, warnings: list[str], label: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"{label} unreadable at {path}: {exc}")
        return None
    if df.empty:
        warnings.append(f"{label} empty at {path}")
        return None
    return df


def _audit_seed_fallback(*frames: pd.DataFrame) -> Any:
    for df in frames:
        for col in ("network_seed", "seed_id", "seed"):
            if col in df.columns and len(df[col].dropna()):
                return df[col].dropna().iloc[0]
    return ""


def _entry_condition_label(value: Any) -> str:
    raw = str(value or "").strip()
    key = raw.lower().replace("_", " ").replace("-", " ")
    if "peak" in key:
        return "Peak ping"
    if "valley" in key:
        return "Valley ping"
    if "random" in key:
        return "Random ping"
    return raw or "Ping"


def _stsp_group_label(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"high", "high_stsp", "top", "top_stsp"}:
        return "high"
    if raw in {"low", "low_stsp", "bottom", "bottom_stsp"}:
        return "low"
    return raw if raw in {"high", "low"} else ""


def _overlap_group_label(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"overlap", "with_overlap", "high_overlap", "probe_overlap"}:
        return "overlap"
    if raw in {"no_overlap", "without_overlap", "low_overlap", "nonoverlap", "none"}:
        return "no_overlap"
    return raw if raw in {"overlap", "no_overlap"} else ""


def _score_quantile_x(value: Any, fallback_index: int) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float(fallback_index)
    text = str(value).strip()
    parsed = _num(text)
    if math.isfinite(parsed):
        return parsed
    cleaned = text.replace("%", "").replace("q", "").replace("Q", "")
    for sep in ("-", "_", "to", ":"):
        if sep in cleaned:
            parts = [part.strip() for part in cleaned.split(sep) if part.strip()]
            nums = [_num(part) for part in parts]
            nums = [num for num in nums if math.isfinite(num)]
            if nums:
                return float(sum(nums) / len(nums))
    digits = "".join(ch if ch.isdigit() or ch == "." else " " for ch in text).split()
    nums = [_num(part) for part in digits]
    nums = [num for num in nums if math.isfinite(num)]
    if nums:
        return float(sum(nums) / len(nums))
    return float(fallback_index)


def _basin_condition_label(row: Mapping[str, Any]) -> str:
    entry_type = str(row.get("entry_type", "") or "").strip().lower()
    entry_condition = str(row.get("entry_condition", "") or "").strip()
    if entry_type == "ping":
        return "Ping"
    if entry_type in {"real_probe", "probe", "real probe"}:
        return "Real probe"
    if entry_condition:
        return _entry_condition_label(entry_condition) if "ping" in entry_condition.lower() else entry_condition
    return str(row.get("entry_type", "") or "Entry")


def _first_finite(values: Sequence[float]) -> float | None:
    for value in values:
        if math.isfinite(float(value)):
            return float(value)
    return None


def _position_from_end(row: Mapping[str, Any]) -> float:
    seq_len = _num(row.get("seq_len", np.nan))
    removed = _num(row.get("removed_position", np.nan))
    if np.isfinite(seq_len) and np.isfinite(removed):
        return float(seq_len - removed + 1)
    if "position_from_end" in row:
        return _num(row.get("position_from_end"))
    rel = _num(row.get("relative_position_from_end", np.nan))
    if np.isfinite(rel):
        return float(rel + 1.0) if rel <= 0 or rel == math.floor(rel) else float(rel)
    return float("nan")


def _p_peak_by_update_count_rows(figure_id: str, panel_id: str, seed_dir: Path, path: Path, repo_root: Path, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    use = df.copy()
    use["update_count"] = pd.to_numeric(use["update_count"], errors="coerce")
    use["is_peak_num"] = use["is_peak"].astype(str).str.lower().isin({"true", "1", "yes"}).astype(float)
    use = use.dropna(subset=["update_count"])
    if use.empty:
        return rows
    max_count = int(use["update_count"].max())
    positive_min = int(use.loc[use["update_count"] > 0, "update_count"].min()) if (use["update_count"] > 0).any() else 0
    has_zero = bool((use["update_count"] == 0).any())
    bin_long_tail = max_count > 4

    def label_count(value: float) -> tuple[str, int]:
        iv = int(value)
        if bin_long_tail:
            if has_zero:
                return (f"{min(iv, 3)}+" if iv >= 3 else str(iv), 3 if iv >= 3 else iv)
            return (f"{min(iv, 4)}+" if iv >= 4 else str(iv), 4 if iv >= 4 else iv)
        return str(iv), iv

    labels = use["update_count"].map(lambda v: label_count(v)[0])
    orders = use["update_count"].map(lambda v: label_count(v)[1])
    use = use.assign(update_count_bin=labels, x_order=orders)
    for (bin_label, order), part in use.groupby(["update_count_bin", "x_order"], sort=True):
        vals = pd.to_numeric(part["is_peak_num"], errors="coerce").dropna().to_numpy(dtype=float)
        rows.append(
            _row(
                figure_id,
                panel_id,
                "P_peak",
                str(bin_label),
                "layer1",
                seed_dir,
                part.get("network_seed", pd.Series([""])).iloc[0] if len(part) else "",
                float(np.mean(vals)) if vals.size else np.nan,
                "probability",
                path,
                repo_root,
                update_count_bin=str(bin_label),
                x_value=float(order),
                y_value=float(np.mean(vals)) if vals.size else np.nan,
                n_units=int(len(part)),
                sem=_sem(vals),
                binned_update_count=bool(bin_long_tail),
                final_support=_num(part["final_support"].mean()) if "final_support" in part.columns else np.nan,
                delta_support=_num(part["delta_support"].mean()) if "delta_support" in part.columns else np.nan,
            )
        )
    _ = positive_min
    return rows


def _peak_overlap_alignment_rows(figure_id: str, panel_id: str, seed_dir: Path, path: Path, repo_root: Path, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    value_col = _first_col(df, ["mean_peak_coverage", "peak_coverage", "overlap_precision", "dice_peak_overlap", "mean_dice", "jaccard_peak_overlap"])
    window_col = _first_col(df, ["overlap_window", "window_type", "condition"])
    if value_col is None or window_col is None:
        return rows
    work = df.copy()
    parsed = work[window_col].astype(str).map(_parse_overlap_window)
    work["window_family"] = [item[0] for item in parsed]
    work["window_k"] = [item[1] for item in parsed]
    selected: list[pd.DataFrame] = []
    for family, target_k in (("old", 3), ("all", None), ("recent", 3)):
        part = work[work["window_family"].eq(family)].copy()
        if part.empty:
            continue
        if target_k is not None:
            part["_dist"] = (pd.to_numeric(part["window_k"], errors="coerce") - float(target_k)).abs()
            min_dist = part["_dist"].min()
            part = part[part["_dist"].eq(min_dist)].copy()
        selected.append(part)
    if not selected:
        return rows
    use = pd.concat(selected, ignore_index=True)
    order_map = {"old": 0, "all": 1, "recent": 2}
    for _, r in use.iterrows():
        family = str(r.get("window_family", ""))
        k = _num(r.get("window_k", np.nan))
        label = "All" if family == "all" else f"{family.title()}-{int(k)}" if np.isfinite(k) else family.title()
        value = _num(r.get(value_col))
        rows.append(
            _row(
                figure_id,
                panel_id,
                "peak_coverage",
                label,
                "layer1",
                seed_dir,
                r.get("network_seed", ""),
                value,
                "fraction",
                path,
                repo_root,
                overlap_window=label,
                window_family=family,
                window_k=int(k) if np.isfinite(k) else "",
                x_value=float(order_map.get(family, len(order_map))),
                y_value=value,
                n_sequences=r.get("n_sequences", ""),
                sem=_num(r.get("sem_peak_coverage", r.get("sem_dice", r.get("sem", np.nan)))),
                source_metric=value_col,
            )
        )
    return rows


def _route_peak_reentry_rows(figure_id: str, panel_id: str, seed_dir: Path, path: Path, repo_root: Path, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    if "mean_normalized_reentry_loss" in df.columns:
        for _, r in df.iterrows():
            unit_set = str(r.get("perturbation_unit_set", ""))
            value = _num(r.get("mean_normalized_reentry_loss"))
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "normalized_reentry_loss",
                    _unit_set_label(unit_set),
                    "layer3",
                    seed_dir,
                    r.get("network_seed", ""),
                    value,
                    "fraction",
                    path,
                    repo_root,
                    perturbation_unit_set=unit_set,
                    x_value=float(_unit_set_order(unit_set)),
                    y_value=value,
                    sem=_num(r.get("sem_normalized_reentry_loss")),
                    n_trials=r.get("n_trials", ""),
                    n_valid_trials=r.get("n_valid_trials", ""),
                    insufficient_fraction=_num(r.get("insufficient_fraction")),
                    final_scientific_use=_num(r.get("n_valid_trials", 0)) > 0,
                )
            )
        return rows
    for unit_set, part in df.groupby("perturbation_unit_set", sort=False):
        vals = pd.to_numeric(part.get("normalized_reentry_loss", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        rows.append(
            _row(
                figure_id,
                panel_id,
                "normalized_reentry_loss",
                _unit_set_label(str(unit_set)),
                "layer3",
                seed_dir,
                part.get("network_seed", pd.Series([""])).iloc[0] if len(part) else "",
                float(np.mean(vals)) if vals.size else np.nan,
                "fraction",
                path,
                repo_root,
                perturbation_unit_set=str(unit_set),
                x_value=float(_unit_set_order(str(unit_set))),
                y_value=float(np.mean(vals)) if vals.size else np.nan,
                sem=_sem(vals),
                n_trials=int(len(part)),
                n_valid_trials=int((_bool_series(part.get("perturbation_ok", pd.Series(dtype=object))) & ~_bool_series(part.get("insufficient_units", pd.Series(dtype=object)))).sum()),
            )
        )
    return rows


def _route_peak_downstream_rows(figure_id: str, panel_id: str, seed_dir: Path, path: Path, repo_root: Path, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    if "P_output_switch" in df.columns:
        for _, r in df.iterrows():
            unit_set = str(r.get("perturbation_unit_set", ""))
            value = _num(r.get("P_output_switch"))
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "P_output_switch",
                    _unit_set_label(unit_set),
                    "layer3",
                    seed_dir,
                    r.get("network_seed", ""),
                    value,
                    "probability",
                    path,
                    repo_root,
                    perturbation_unit_set=unit_set,
                    x_value=float(_unit_set_order(unit_set)),
                    y_value=value,
                    sem=_num(r.get("sem_output_switch")),
                    n_trials=r.get("n_trials", ""),
                    n_valid_trials=r.get("n_valid_trials", ""),
                    mean_response_displacement_loss=_num(r.get("mean_response_displacement_loss")),
                    mean_decision_deflection_loss=_num(r.get("mean_decision_deflection_loss")),
                    final_scientific_use=_num(r.get("n_valid_trials", 0)) > 0,
                )
            )
        return rows
    for unit_set, part in df.groupby("perturbation_unit_set", sort=False):
        vals = _bool_series(part.get("output_switch", pd.Series(dtype=object))).astype(float).to_numpy(dtype=float)
        rows.append(
            _row(
                figure_id,
                panel_id,
                "P_output_switch",
                _unit_set_label(str(unit_set)),
                "layer3",
                seed_dir,
                part.get("network_seed", pd.Series([""])).iloc[0] if len(part) else "",
                float(np.mean(vals)) if vals.size else np.nan,
                "probability",
                path,
                repo_root,
                perturbation_unit_set=str(unit_set),
                x_value=float(_unit_set_order(str(unit_set))),
                y_value=float(np.mean(vals)) if vals.size else np.nan,
                n_trials=int(len(part)),
                n_valid_trials=int((_bool_series(part.get("perturbation_ok", pd.Series(dtype=object))) & ~_bool_series(part.get("insufficient_units", pd.Series(dtype=object)))).sum()),
            )
        )
    return rows


def _read_required_route_peak_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} is required for formal Fig.6 D/E build: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{label} is empty; formal Fig.6 D/E build requires valid route-peak perturbation data: {path}")
    return df


def _validate_route_peak_summary(df: pd.DataFrame, *, panel: str, value_col: str) -> None:
    if value_col not in df.columns:
        raise RuntimeError(f"Fig.6{panel} route-peak summary missing required column {value_col}.")
    values = pd.to_numeric(df[value_col], errors="coerce")
    if not values.notna().any():
        raise RuntimeError(f"Fig.6{panel} required route-peak perturbation data missing or invalid: {value_col} is all NaN.")
    if "n_valid_trials" not in df.columns:
        raise RuntimeError(f"Fig.6{panel} route-peak summary missing n_valid_trials.")
    valid = pd.to_numeric(df["n_valid_trials"], errors="coerce").fillna(0)
    if not valid.gt(0).any():
        raise RuntimeError("Fig.6D/E required route-peak perturbation data missing or invalid.")
    present = set(df.get("perturbation_unit_set", pd.Series(dtype=str)).astype(str))
    missing = [unit_set for unit_set in ("route_peak", "route_nonpeak", "nonroute_peak", "random_matched") if unit_set not in present]
    invalid = [
        unit_set
        for unit_set in ("route_peak", "route_nonpeak", "nonroute_peak", "random_matched")
        if unit_set in present
        and pd.to_numeric(df.loc[df["perturbation_unit_set"].astype(str).eq(unit_set), "n_valid_trials"], errors="coerce").fillna(0).sum() <= 0
    ]
    if missing or invalid:
        raise RuntimeError(
            "Fig.6D/E required route-peak perturbation data missing or invalid: "
            f"missing_unit_sets={missing or 'none'}, invalid_unit_sets={invalid or 'none'}."
        )


def _validate_route_peak_audit(df: pd.DataFrame, label: str) -> None:
    success = _bool_series(df.get("route_peak_perturbation_success", pd.Series(dtype=object))).any()
    final_use = _bool_series(df.get("final_scientific_use", pd.Series(dtype=object))).any()
    proxy = _bool_series(df.get("proxy_mode", pd.Series(dtype=object))).any()
    claim = set(df.get("allowed_claim_strength", pd.Series(dtype=str)).astype(str))
    if proxy:
        raise RuntimeError(f"{label} reports proxy_mode=true; formal Fig.6 D/E build forbids proxy outputs.")
    if not success or not final_use or "causal_route_peak_gain" not in claim:
        reason = ";".join(str(v) for v in df.get("failure_reason", pd.Series(dtype=str)).dropna().tolist() if str(v).strip())
        raise RuntimeError(f"{label} did not clear formal Fig.6 D/E scientific-use audit. failure_reason={reason or 'not reported'}")


def _parse_overlap_window(value: Any) -> tuple[str, float]:
    text = str(value).lower().replace("-", "_")
    if text == "all":
        return "all", np.nan
    for family in ("recent", "old"):
        if text.startswith(family):
            parts = [p for p in text.split("_") if p.isdigit()]
            return family, float(parts[0]) if parts else np.nan
    return text, np.nan


def _unit_set_label(unit_set: str) -> str:
    return {
        "route_peak": "Route peak",
        "route_nonpeak": "Route non-peak",
        "nonroute_peak": "Non-route peak",
        "random_matched": "Random",
    }.get(str(unit_set), str(unit_set).replace("_", " ").title())


def _unit_set_order(unit_set: str) -> int:
    order = {"route_peak": 0, "route_nonpeak": 1, "nonroute_peak": 2, "random_matched": 3}
    return order.get(str(unit_set), len(order))


def _sem(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size <= 1:
        return 0.0
    return float(np.std(clean, ddof=1) / np.sqrt(clean.size))


def _update_history_rows(figure_id: str, panel_id: str, seed_dir: Path, path: Path, repo_root: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    group_col = _first_col(df, ["group", "update_group", "update_history_group", "region_type"])
    rows: list[dict[str, Any]] = []
    if group_col is None:
        return rows
    labels = {"peak": "Peak", "nonpeak_control": "Nonpeak control", "prior_updated_nonpeak": "Prior-updated nonpeak"}
    for _, r in df.iterrows():
        raw_group = str(r.get(group_col, ""))
        condition = labels.get(raw_group, GROUP_LABELS.get(raw_group, raw_group.replace("_", " ").title()))
        for metric, unit in (("mean_update_count", "count"), ("mean_time_since_last_update", "positions"), ("P_multi_recent_w3", "fraction"), ("mean_recent_update_count", "count"), ("P_peak", "fraction")):
            if metric not in df.columns:
                continue
            rows.append(_row(figure_id, panel_id, metric, condition, "layer1", seed_dir, r.get("network_seed", ""), _num(r.get(metric)), unit, path, repo_root, group=raw_group, n_units=r.get("n_units", "")))
    return rows


def _input_overlap_rows(figure_id: str, panel_id: str, seed_dir: Path, path: Path, repo_root: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    value_col = _first_col(df, ["mean_dice", "dice_peak_overlap", "jaccard_peak_overlap"])
    cond_col = _first_col(df, ["overlap_window", "window_type", "condition"])
    rows: list[dict[str, Any]] = []
    if value_col is None:
        return rows
    for _, r in df.iterrows():
        condition = str(r.get(cond_col, "")) if cond_col else str(r.get("recent_k", "window"))
        rows.append(
            _row(
                figure_id,
                panel_id,
                "dice_peak_overlap",
                condition,
                "layer1",
                seed_dir,
                r.get("network_seed", ""),
                _num(r.get(value_col)),
                "dice",
                path,
                repo_root,
                overlap_window=condition,
                recent_k=r.get("recent_k", ""),
                mean_peak_coverage=_num(r.get("mean_peak_coverage", r.get("peak_coverage", np.nan))),
                mean_cosine=_num(r.get("mean_cosine", r.get("cosine_delta_support_overlap", np.nan))),
                n_sequences=r.get("n_sequences", ""),
            )
        )
    return rows


def _reentry_rows(figure_id: str, panel_id: str, seed_dir: Path, path: Path, repo_root: Path, df: pd.DataFrame, source_level: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if {"reentry_low", "reentry_high"}.issubset(df.columns):
        for _, r in df.iterrows():
            for condition, value_col, pwo_col in (
                ("Low peak overlap", "reentry_low", "peak_weighted_overlap_low"),
                ("High peak overlap", "reentry_high", "peak_weighted_overlap_high"),
            ):
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        "reentry_strength_real",
                        condition,
                        "layer3",
                        seed_dir,
                        r.get("network_seed", ""),
                        _num(r.get(value_col)),
                        "a.u.",
                        path,
                        repo_root,
                        matched_group_id=r.get("matched_group_id", r.get("matched_set_id", "")),
                        raw_overlap=_num(r.get("raw_overlap", r.get("raw_overlap_low", np.nan))),
                        peak_weighted_overlap=_num(r.get(pwo_col)),
                        peak_overlap_group="low_peak_overlap" if condition.startswith("Low") else "high_peak_overlap",
                        visual_similarity=_num(r.get("visual_similarity", np.nan)),
                        input_energy=_num(r.get("input_energy", np.nan)),
                        proxy_mode=_bool_value(r.get("proxy_mode", False)),
                        final_scientific_use=_bool_value(r.get("final_scientific_use", not _bool_value(r.get("proxy_mode", False)))),
                        raw_overlap_control="matched_group",
                        source_level=source_level,
                    )
                )
        return rows
    value_col = _first_col(df, ["reentry_strength_real", "reentry_strength", "l3_trace_delta_norm"])
    if value_col is None:
        return rows
    for _, r in df.iterrows():
        raw_group = str(r.get("peak_overlap_group", ""))
        condition = _peak_group_label(raw_group)
        metric = "reentry_strength_real" if value_col == "reentry_strength_real" else "reentry_strength"
        proxy = _bool_value(r.get("proxy_mode", False))
        rows.append(
            _row(
                figure_id,
                panel_id,
                metric,
                condition,
                "layer3",
                seed_dir,
                r.get("network_seed", ""),
                _num(r.get(value_col)),
                "a.u.",
                path,
                repo_root,
                sequence_id=r.get("sequence_id", ""),
                probe_id=r.get("probe_id", ""),
                matched_group_id=r.get("matched_group_id", ""),
                raw_overlap=_num(r.get("raw_overlap")),
                peak_weighted_overlap=_num(r.get("peak_weighted_overlap")),
                peak_overlap_group=raw_group,
                visual_similarity=_num(r.get("visual_similarity")),
                input_energy=_num(r.get("input_energy")),
                x_value=_num(r.get("peak_weighted_overlap")),
                y_value=_num(r.get(value_col)),
                proxy_mode=proxy,
                final_scientific_use=_bool_value(r.get("final_scientific_use", not proxy)),
                raw_overlap_control="matched_group" if str(r.get("matched_group_id", "")) else "regression",
                source_level=source_level,
            )
        )
    return rows


def _downstream_rows(figure_id: str, panel_id: str, seed_dir: Path, path: Path, repo_root: Path, df: pd.DataFrame, source_level: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "downstream_metric" in df.columns:
        for _, r in df.iterrows():
            metric = str(r.get("downstream_metric", ""))
            value_col = _first_col(pd.DataFrame([r]), ["high_minus_low", "beta_peak_weighted_overlap"])
            if value_col is None:
                continue
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    metric,
                    DOWNSTREAM_LABELS.get(metric, metric.replace("_", " ")),
                    "layer3",
                    seed_dir,
                    r.get("network_seed", ""),
                    _num(r.get(value_col)),
                    "a.u.",
                    path,
                    repo_root,
                    downstream_node=metric,
                    y_value=_num(r.get(value_col)),
                    proxy_mode=_bool_value(r.get("proxy_mode", False)),
                    final_scientific_use=_bool_value(r.get("final_scientific_use", True)),
                    raw_overlap_control="regression",
                    source_level=source_level,
                )
            )
        return rows
    metric_cols = [
        ("early_recruitment_gain_real", "early_recruitment_gain"),
        ("response_pattern_displacement_real", "response_pattern_displacement"),
        ("decision_deflection_score_real", "decision_deflection_score"),
    ]
    for _, r in df.iterrows():
        proxy = _bool_value(r.get("proxy_mode", False))
        for real_col, fallback_col in metric_cols:
            metric = real_col if real_col in df.columns else fallback_col if fallback_col in df.columns else ""
            if not metric:
                continue
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    metric,
                    DOWNSTREAM_LABELS.get(metric, metric),
                    "layer3",
                    seed_dir,
                    r.get("network_seed", ""),
                    _num(r.get(metric)),
                    "a.u.",
                    path,
                    repo_root,
                    sequence_id=r.get("sequence_id", ""),
                    probe_id=r.get("probe_id", ""),
                    matched_group_id=r.get("matched_group_id", ""),
                    raw_overlap=_num(r.get("raw_overlap")),
                    peak_weighted_overlap=_num(r.get("peak_weighted_overlap")),
                    peak_overlap_group=r.get("peak_overlap_group", ""),
                    downstream_node=metric,
                    x_value=_num(r.get("peak_weighted_overlap")),
                    y_value=_num(r.get(metric)),
                    proxy_mode=proxy,
                    final_scientific_use=_bool_value(r.get("final_scientific_use", not proxy)),
                    raw_overlap_control="matched_group" if str(r.get("matched_group_id", "")) else "regression",
                    source_level=source_level,
                )
            )
    return rows


def _finish(
    spec: Mapping[str, Any],
    output_dir: Path,
    root: Path,
    seeds: Sequence[Path],
    rows: list[dict[str, Any]],
    source_paths: list[Path],
    checked_paths: list[Path],
    warnings: list[str],
    group_cols: list[str],
    *,
    stats_extra: Mapping[str, Any] | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    run_mode = _run_mode(seeds)
    if not seeds:
        warnings.append(f"Experiment root does not contain seed results: {root}")
    if run_mode == "single_network_draft":
        warnings.append(DRAFT_WARNING)
    if not rows:
        return _write_missing(spec, output_dir, root, checked_paths, warnings, f"{figure_id}{panel_id} adapter found no plottable rows.")
    panel_df = pd.DataFrame(rows)
    rows_before_network_aggregation = len(panel_df)
    if figure_id.startswith("supp_fig_s"):
        panel_df = _aggregate_part2_network_rows(panel_df, group_cols)
    panel_df["run_mode"] = run_mode
    panel_df["n_networks"] = len(seeds)
    proxy_flags = _bool_series(panel_df.get("proxy_mode", pd.Series(dtype=object)))
    final_flags = _bool_series(panel_df.get("final_scientific_use", pd.Series(dtype=object)))
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "run_mode": run_mode,
        "n_networks": len(seeds),
        "network_ids": [_seed_id(seed) for seed in seeds],
        "summaries": summarize_values(panel_df, [col for col in group_cols if col in panel_df.columns]),
        "network_summaries": _part2_network_summaries(panel_df, group_cols) if figure_id.startswith("supp_fig_s") else [],
        "values_used_for_plotting": _values(panel_df),
        "source_files_used": int(len(set(map(str, source_paths)))),
        "raw_rows_read": int(rows_before_network_aggregation),
        "rows_after_source_filtering": int(len(panel_df)),
        "rows_written_to_panel_data": int(len(panel_df)),
        "rows_before_network_aggregation": int(rows_before_network_aggregation),
        "rows_after_network_aggregation": int(len(panel_df)),
        "adapter_performed_network_level_averaging": bool(figure_id.startswith("supp_fig_s") and rows_before_network_aggregation != len(panel_df)),
        "source_appeared_preaggregated": False,
        "inferential_unit": "independent network" if figure_id.startswith("supp_fig_s") else "legacy panel rows",
        "replicate_unit": "network_id" if figure_id.startswith("supp_fig_s") else "legacy panel rows",
        "interval_definition": "two-sided 95% Student-t confidence interval across independent networks" if figure_id.startswith("supp_fig_s") else "legacy",
        "any_proxy_mode": bool(proxy_flags.any()) if len(proxy_flags) else False,
        "all_proxy_mode": bool(proxy_flags.all()) if len(proxy_flags) else False,
        "n_final_scientific_rows": int(final_flags.sum()) if len(final_flags) else 0,
        "warnings": list(dict.fromkeys(warnings)),
    }
    stats.update(dict(stats_extra or {}))
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok",
        "experiment_root": str(root),
        "run_mode": run_mode,
        "n_networks": len(seeds),
        "network_ids": [_seed_id(seed) for seed in seeds],
        "source_files_used": [_rel(path, root.parent if root.name.startswith("seed_") else root) for path in source_paths],
        "sources": [_source_entry(path, repo_root=root.parent if root.name.startswith("seed_") else root, used=path in source_paths) for path in checked_paths],
        "checked_candidates": [_rel(path, root.parent if root.name.startswith("seed_") else root) for path in checked_paths],
        "proxy_summary": {
            "any_proxy_mode": stats["any_proxy_mode"],
            "all_proxy_mode": stats["all_proxy_mode"],
            "n_final_scientific_rows": stats["n_final_scientific_rows"],
        },
        "claim_strength": _claim_strength(seeds),
        "warnings": list(stats["warnings"]),
    }
    manifest.update(dict(manifest_extra or {}))
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, list(stats["warnings"]))


def _aggregate_part2_network_rows(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    dimensions = [
        *group_cols,
        "metric",
        "condition",
        "layer",
        "score_quantile_bin",
        "early_window_ms",
        "endpoint",
        "stsp_group_quantile",
        "overlap_threshold",
    ]
    keys = ["network_id", *[col for col in dict.fromkeys(dimensions) if col in df.columns and col != "network_id"]]
    drop_cols = {"trial_id", "unit_id", "pair_id", "sequence_id", "probe_id", "event_id", "shuffle_id"}
    work = df.drop(columns=[col for col in drop_cols if col in df.columns]).copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["network_id", "value"])
    aggregations = {col: ("mean" if col == "value" else "first") for col in work.columns if col not in keys}
    grouped = work.groupby(keys, dropna=False, sort=False)
    out = grouped.agg(aggregations).reset_index()
    out["lower_level_rows"] = grouped.size().to_numpy()
    assert not out.duplicated(keys).any()
    return out


def _part2_network_summaries(df: pd.DataFrame, group_cols: Sequence[str]) -> list[dict[str, Any]]:
    groups = [col for col in group_cols if col in df.columns]
    grouped = df.groupby(groups, dropna=False, sort=False) if groups else [((), df)]
    rows: list[dict[str, Any]] = []
    for key, part in grouped:
        values = pd.to_numeric(part["value"], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        if not isinstance(key, tuple):
            key = (key,)
        n = int(len(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        half = float(scipy_stats.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
        row = {col: value for col, value in zip(groups, key)}
        row.update({
            "n_networks": n,
            "mean": float(np.mean(values)),
            "sem": sem,
            "ci95_low": float(np.mean(values) - half),
            "ci95_high": float(np.mean(values) + half),
            "one_sample_p_vs_zero": _part2_one_sample_p(values),
        })
        rows.append(row)
    return rows


def _part2_one_sample_p(values: np.ndarray) -> float | None:
    if len(values) <= 1:
        return None
    if float(np.std(values, ddof=1)) < 1e-15:
        return 1.0 if abs(float(np.mean(values))) < 1e-15 else 0.0
    return float(scipy_stats.ttest_1samp(values, 0.0).pvalue)


def _write_missing(
    spec: Mapping[str, Any],
    output_dir: Path,
    root: Path,
    checked_paths: Sequence[Path],
    warnings: Sequence[str],
    reason: str,
) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    df = pd.DataFrame(
        [
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "metric": "missing_source",
                "condition": "missing",
                "layer": "",
                "network_id": "",
                "seed_id": "",
                "value": np.nan,
                "unit": "",
                "source_file": "",
                "placeholder_reason": reason,
            }
        ]
    )
    all_warnings = list(dict.fromkeys([*warnings, reason]))
    stats = {"figure_id": figure_id, "panel_id": panel_id, "status": "missing_source", "values_used_for_plotting": [], "warnings": all_warnings}
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "missing_source",
        "experiment_root": str(root),
        "sources": [_source_entry(path, repo_root=root.parent if root.name.startswith("seed_") else root, used=False) for path in checked_paths],
        "checked_candidates": [_rel(path, root.parent if root.name.startswith("seed_") else root) for path in checked_paths],
        "warnings": all_warnings,
    }
    return write_adapter_outputs(output_dir, figure_id, panel_id, df, stats, manifest, all_warnings)


def _seed_dirs(spec: Mapping[str, Any], repo_root: Path) -> tuple[Path, list[Path], list[str]]:
    root = Path(str(spec.get("experiment_root") or DEFAULT_EXPERIMENT_ROOT))
    if not root.is_absolute():
        root = repo_root / root
    warnings: list[str] = []
    if root.name.startswith("seed_"):
        return root, ([root] if root.exists() else []), warnings
    if not root.exists():
        warnings.append(f"Experiment root does not exist: {root}")
        return root, [], warnings
    seeds = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not seeds and (root / "summary.json").exists():
        seeds = [root]
    return root, seeds, warnings


def _first_existing(seed_dir: Path, candidates: Sequence[str]) -> tuple[Path | None, pd.DataFrame | None, list[Path]]:
    paths = [seed_dir / candidate for candidate in candidates]
    for path in paths:
        if path.exists():
            if path.suffix.lower() == ".csv":
                try:
                    return path, pd.read_csv(path), paths
                except Exception:
                    return path, None, paths
            return path, None, paths
    return None, None, paths


def _first_nonempty_by_level(seed_dir: Path, candidates: Sequence[tuple[str, str]]) -> tuple[str, Path | None, pd.DataFrame | None]:
    for level, rel in candidates:
        path = seed_dir / rel
        if not path.exists() or path.suffix.lower() != ".csv":
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            return level, path, df
    return "", None, None


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig6")), str(spec.get("panel_id", "")).upper()


def _row(
    figure_id: str,
    panel_id: str,
    metric: str,
    condition: str,
    layer: str,
    seed_dir: Path,
    seed_fallback: Any,
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
        "network_id": seed_dir.name,
        "seed_id": _seed_id(seed_dir, seed_fallback),
        "value": value,
        "unit": unit,
        "source_file": _rel(source_file, repo_root),
    }
    row.update(extra)
    return row


def _run_mode(seeds: Sequence[Path]) -> str:
    return "multi_network_final" if len(seeds) > 1 else "single_network_draft"


def _seed_id(seed_dir: Path, fallback: Any = "") -> str:
    if seed_dir.name.startswith("seed_"):
        return seed_dir.name.replace("seed_", "")
    return str(fallback)


def _values(df: pd.DataFrame) -> list[float]:
    values = pd.to_numeric(df.get("value", pd.Series(dtype=float)), errors="coerce").dropna()
    return [float(v) for v in values.tolist()]


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _num(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float("nan") if pd.isna(numeric) else float(numeric)


def _first_col(df: pd.DataFrame, columns: Sequence[str]) -> str | None:
    for col in columns:
        if col in df.columns:
            return col
    return None


def _rel(path: Path | str, root: Path) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path_obj).replace("\\", "/")


def _source_entry(path: Path, *, repo_root: Path, used: bool) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": _rel(path, repo_root),
        "exists": exists,
        "used": bool(used),
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": _sha256(path) if exists else "",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_label(metric: str) -> str:
    return {
        "raw_overlap": "Raw overlap route",
        "peak_weighted_overlap": "Peak-weighted gain",
        "peak_overlap_fraction": "Peak overlap fraction",
        "nonpeak_overlap_fraction": "Nonpeak overlap fraction",
    }.get(metric, metric.replace("_", " "))


def _peak_group_label(group: str) -> str:
    return {
        "high_peak_overlap": "High peak overlap",
        "low_peak_overlap": "Low peak overlap",
        "high_peak_weighted_overlap": "High peak overlap",
        "low_peak_weighted_overlap": "Low peak overlap",
        "unmatched": "Probe candidates",
        "": "Probe candidates",
    }.get(str(group), str(group).replace("_", " "))


def _preferred_level(levels: Sequence[str], order: Sequence[str]) -> str:
    for level in order:
        if level in levels:
            return level
    return levels[0] if levels else ""


def _high_minus_low(rows: Sequence[Mapping[str, Any]]) -> float | None:
    df = pd.DataFrame(rows)
    if df.empty or "condition" not in df.columns:
        return None
    high = pd.to_numeric(df.loc[df["condition"].astype(str).eq("High peak overlap"), "value"], errors="coerce").dropna()
    low = pd.to_numeric(df.loc[df["condition"].astype(str).eq("Low peak overlap"), "value"], errors="coerce").dropna()
    if high.empty or low.empty:
        return None
    return float(high.mean() - low.mean())


def _regression_slope(rows: Sequence[Mapping[str, Any]]) -> float | None:
    df = pd.DataFrame(rows)
    if df.empty or not {"peak_weighted_overlap", "value"}.issubset(df.columns):
        return None
    x = pd.to_numeric(df["peak_weighted_overlap"], errors="coerce")
    y = pd.to_numeric(df["value"], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def _raw_overlap_control(rows: Sequence[Mapping[str, Any]]) -> str:
    controls = [str(row.get("raw_overlap_control", "")) for row in rows if row.get("raw_overlap_control", "")]
    if "matched_group" in controls:
        return "matched_group"
    if "regression" in controls:
        return "regression"
    return ""


def _claim_strength(seeds: Sequence[Path]) -> str:
    strengths = []
    perturb_success = False
    for seed_dir in seeds:
        path = seed_dir / "summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        route_peak = summary.get("fig6_route_peak_perturbation", {})
        strengths.append(str(summary.get("allowed_claim_strength", route_peak.get("allowed_claim_strength", "predictive_peak_amplified_only") if isinstance(route_peak, Mapping) else "predictive_peak_amplified_only")))
        perturb_success = perturb_success or _bool_value(summary.get("peak_perturbation_successful", route_peak.get("success", False) if isinstance(route_peak, Mapping) else False))
    if perturb_success and any("causal" in item for item in strengths):
        return "causal_route_peak_gain"
    return "predictive_peak_amplified_only"


def _figure_chain(metadata: Mapping[str, Any]) -> list[str]:
    chain = metadata.get("figure_chain")
    if isinstance(chain, list) and chain:
        return [str(item) for item in chain]
    return [
        "Fig.1 functional STSP substrate",
        "Fig.2 two-item fused state",
        "Fig.3 multi-item peak landscape",
        "Fig.4 overlap re-entry route",
        "Fig.5 local support / competition conversion",
        "Fig.6 peak-amplified re-entry",
    ]


def _forbidden_language(metadata: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("forbidden_language", "forbidden_claims"):
        value = metadata.get(key, summary.get(key, []))
        if isinstance(value, list):
            out.extend(str(item) for item in value)
        elif value:
            out.append(str(value))
    out.extend(["peaks replace overlap", "peak-gated re-entry", "peak-driven re-entry", "peaks provide the route"])
    return list(dict.fromkeys(out))


def _forbidden_language_from_seeds(seeds: Sequence[Path]) -> list[str]:
    out: list[str] = []
    for seed_dir in seeds:
        metadata_path = seed_dir / "data" / "raw" / "panel_f_global_mechanism_metadata.json"
        summary_path = seed_dir / "summary.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        summary = read_json(summary_path) if summary_path.exists() else {}
        out.extend(_forbidden_language(metadata, summary))
    return list(dict.fromkeys(out))
