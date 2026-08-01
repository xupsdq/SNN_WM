from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult, missing_adapter_result, summarize_values, write_adapter_outputs


DEFAULT_EXPERIMENT_ROOT = Path("results") / "paper_experiments" / "fig2_pair_fused_stsp_state"
SINGLE_NETWORK_WARNING = "Single-network result. Use for pipeline validation only, not final manuscript statistics."
STATE_CONDITIONS = ("S0", "S_A", "S_B", "S_AB")
STATE_LABELS = {"S0": "No memory", "S_A": "Item 1 only", "S_B": "Item 2 only", "S_AB": "Item 1->Item 2"}
COMPOSITION_CATEGORIES = ("Other", "A", "B", "Silent")


def build_fig2_dual_retention_constituent_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.2B constituent-only S_AB similarity rows."""
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2 experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    raw_rows = 0
    filtered_rows = 0
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_b_dual_retention_metrics.csv"
        if not path.exists():
            warnings.append(f"Missing dual retention metrics: {_display_path(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        raw_rows += len(df)
        part = _primary(df)
        filtered_rows += len(part)
        sources.append(_source_entry(path, repo_root))
        for _, row in part.iterrows():
            metadata = {
                "fusion_dual_score": row.get("fusion_dual_score", ""),
                "min_component_similarity": row.get("min_component_similarity", ""),
                "sim_to_A_minus_B": row.get("sim_to_A_minus_B", ""),
                "pair_id": row.get("pair_id", ""),
                "layer": row.get("layer", "layer3"),
                "state_variable": row.get("state_variable", "g"),
                "run_mode": _run_mode(seeds),
            }
            for metric, condition in (("sim_to_A", "S_A"), ("sim_to_B", "S_B")):
                if metric not in row or pd.isna(row.get(metric)):
                    continue
                rows.append(
                    _canonical_row(
                        figure_id,
                        panel_id,
                        metric="constituent_similarity",
                        condition=condition,
                        value=float(row.get(metric)),
                        unit="similarity",
                        seed_id=row.get("network_seed", _seed_id(seed_dir)),
                        source_file=_display_path(path, repo_root),
                        constituent_metric=metric,
                        **metadata,
                    )
                )
    if not sources:
        return missing_adapter_result(spec, repo_root, output_dir, "Missing panel_b_dual_retention_metrics.csv for Fig.2B.")
    raw_panel_df = _sort_conditions(pd.DataFrame(rows), ["S_A", "S_B"])
    panel_df = _seed_level_summary(raw_panel_df, ["condition"])
    stats = _stats_payload(figure_id, panel_id, panel_df, "constituent_similarity", _run_mode(seeds), ["condition"])
    stats["metadata_metrics_retained_not_plotted"] = ["fusion_dual_score", "min_component_similarity", "sim_to_A_minus_B"]
    _add_processing_stats(stats, sources, raw_rows, filtered_rows, len(panel_df))
    _add_seed_aggregation_stats(stats, raw_panel_df, panel_df)
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["metadata_metrics_retained_not_plotted"] = stats["metadata_metrics_retained_not_plotted"]
    manifest["visual_categories"] = ["S_A", "S_B"]
    manifest["notes"] = ["fusion_dual_score retained as metadata but not plotted"]
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_fig2_pair_specificity_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2 experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    raw_rows = 0
    filtered_rows = 0
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_c_pair_specificity_metrics.csv"
        if not path.exists():
            warnings.append(f"Missing pair specificity metrics: {_display_path(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        raw_rows += len(df)
        part = _primary(df)
        filtered_rows += len(part)
        sources.append(_source_entry(path, repo_root))
        for _, row in part.iterrows():
            for condition, col in (("True pair", "true_pair_score"), ("Shuffled pair", "shuffled_pair_score")):
                if col not in row or pd.isna(row.get(col)):
                    continue
                rows.append(
                    _canonical_row(
                        figure_id,
                        panel_id,
                        metric="pair_specificity_score",
                        condition=condition,
                        value=float(row.get(col)),
                        unit="similarity",
                        seed_id=row.get("network_seed", _seed_id(seed_dir)),
                        source_file=_display_path(path, repo_root),
                        layer=row.get("layer", "layer3"),
                        pair_id=row.get("pair_id", ""),
                        state_variable=row.get("state_variable", "g"),
                        true_minus_shuffled=row.get("true_minus_shuffled", ""),
                        true_pair_percentile=row.get("true_pair_percentile", ""),
                        true_pair_z=row.get("true_pair_z", ""),
                        n_shuffle=row.get("n_shuffle", ""),
                        run_mode=_run_mode(seeds),
                    )
                )
    if not sources:
        return missing_adapter_result(spec, repo_root, output_dir, "Missing panel_c_pair_specificity_metrics.csv for Fig.2D.")
    raw_panel_df = _sort_conditions(pd.DataFrame(rows), ["True pair", "Shuffled pair"])
    panel_df = _seed_level_summary(raw_panel_df, ["condition"])
    stats = _stats_payload(figure_id, panel_id, panel_df, "pair_specificity_score", _run_mode(seeds), ["condition"])
    _add_processing_stats(stats, sources, raw_rows, filtered_rows, len(panel_df))
    _add_seed_aggregation_stats(stats, raw_panel_df, panel_df)
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_fig2_morphology_closure_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.2C WPRI plus leakage-safe cross-fitted interaction rows."""
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2 experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    raw_rows = 0
    filtered_rows = 0
    for seed_dir in seeds:
        metrics_dir = seed_dir / "data" / "metrics"
        org_path = metrics_dir / "panel_d_pair_level_organization_metrics.csv"
        crossfit_path = metrics_dir / "panel_d_crossfit_interaction_network_metrics.csv"
        crossfit_spec_path = metrics_dir / "panel_d_crossfit_interaction_analysis_spec.json"
        if org_path.exists():
            org = pd.read_csv(org_path)
            raw_rows += len(org)
            org = _primary(org)
            filtered_rows += len(org)
            sources.append(_source_entry(org_path, repo_root))
            for _, row in org.iterrows():
                if pd.isna(row.get("WPRI")):
                    continue
                rows.append(
                    _canonical_row(
                        figure_id,
                        panel_id,
                        metric="WPRI",
                        condition="WPRI",
                        value=float(row.get("WPRI")),
                        unit="score",
                        seed_id=row.get("network_seed", _seed_id(seed_dir)),
                        source_file=_display_path(org_path, repo_root),
                        layer=row.get("layer", "layer3"),
                        pair_id=row.get("pair_id", ""),
                        state_variable=row.get("state_variable", "g"),
                        sim_to_true_pair=row.get("sim_to_true_pair", ""),
                        best_constituent_similarity=row.get("best_constituent_similarity", ""),
                        run_mode=_run_mode(seeds),
                    )
                )
        else:
            warnings.append(f"Missing WPRI metrics: {_display_path(org_path, repo_root)}")
        if crossfit_path.exists():
            crossfit = pd.read_csv(crossfit_path)
            raw_rows += len(crossfit)
            crossfit = _primary(crossfit)
            filtered_rows += len(crossfit)
            sources.append(_source_entry(crossfit_path, repo_root))
            metric_col = "delta_r2_interaction_beyond_bounded_saturation"
            if metric_col not in crossfit.columns:
                raise RuntimeError(
                    f"Fig.2C crossfit source is missing the required endpoint {metric_col}: {crossfit_path}"
                )
            for _, row in crossfit.iterrows():
                if pd.isna(row.get(metric_col)):
                    continue
                rows.append(
                    _canonical_row(
                        figure_id,
                        panel_id,
                        metric=metric_col,
                        condition="Cross-fit ΔR²",
                        value=float(row.get(metric_col)),
                        unit="held-out delta R2",
                        seed_id=row.get("network_seed", _seed_id(seed_dir)),
                        source_file=_display_path(crossfit_path, repo_root),
                        layer=row.get("layer", "layer3"),
                        state_variable=row.get("state_variable", "g"),
                        n_pairs=row.get("n_pairs", ""),
                        n_features=row.get("n_features", ""),
                        n_folds=row.get("n_folds", ""),
                        linear_delta_r2=row.get("delta_r2_linear_interaction", ""),
                        quadratic_delta_r2=row.get("delta_r2_interaction_beyond_marginal_nonlinearity", ""),
                        fallback_used=False,
                        run_mode=_run_mode(seeds),
                    )
                )
        else:
            warnings.append(f"Missing crossfit interaction metrics: {_display_path(crossfit_path, repo_root)}")
        if crossfit_spec_path.exists():
            sources.append(_source_entry(crossfit_spec_path, repo_root))
        else:
            warnings.append(f"Missing crossfit analysis spec: {_display_path(crossfit_spec_path, repo_root)}")
    if not any(source["exists"] for source in sources):
        return missing_adapter_result(spec, repo_root, output_dir, "Missing Fig.2C morphology closure sources.")
    raw_panel_df = _sort_conditions(pd.DataFrame(rows), ["WPRI", "Cross-fit ΔR²"])
    panel_df = _seed_level_summary(raw_panel_df, ["condition", "metric"])
    stats = _stats_payload(figure_id, panel_id, panel_df, "morphology_closure", _run_mode(seeds), ["metric", "condition"])
    stats["visual_metrics"] = _unique(panel_df, "metric")
    stats["linear_model_comparison_plotted"] = False
    stats["legacy_residual_template_metric_used"] = False
    _add_processing_stats(stats, sources, raw_rows, filtered_rows, len(panel_df))
    _add_seed_aggregation_stats(stats, raw_panel_df, panel_df)
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["visual_metrics"] = stats["visual_metrics"]
    manifest["linear_model_comparison_plotted"] = False
    manifest["legacy_residual_template_metric_used"] = False
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_fig2_neutral_ping_composition_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2 experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    source_levels: set[str] = set()
    raw_rows = 0
    filtered_rows = 0
    for seed_dir in seeds:
        raw_path = seed_dir / "data" / "raw" / "panel_e_neutral_ping_trial_readout.csv"
        metrics_path = seed_dir / "data" / "metrics" / "panel_e_neutral_ping_metrics.csv"
        summary_path = seed_dir / "summary.json"
        if raw_path.exists():
            df = pd.read_csv(raw_path)
            raw_rows += len(df)
            sources.append(_source_entry(raw_path, repo_root))
            source_levels.add("trial_level")
            rows.extend(_neutral_ping_rows_from_trials(df, figure_id, panel_id, raw_path, repo_root, seed_dir, seeds, warnings))
            filtered_rows += len(df)
        elif metrics_path.exists():
            df = pd.read_csv(metrics_path)
            raw_rows += len(df)
            sources.append(_source_entry(metrics_path, repo_root))
            source_levels.add("metrics_fallback")
            rows.extend(_neutral_ping_rows_from_metrics(df, figure_id, panel_id, metrics_path, repo_root, seed_dir, seeds, warnings))
            filtered_rows += len(df)
        else:
            warnings.append(f"Missing neutral ping composition sources under {_display_path(seed_dir, repo_root)}")
        if summary_path.exists():
            sources.append(_source_entry(summary_path, repo_root))
    if not any(source["exists"] for source in sources):
        return missing_adapter_result(spec, repo_root, output_dir, "Missing Fig.2E neutral ping composition sources.")
    panel_df = _sort_composition(pd.DataFrame(rows))
    stats = _stats_payload(figure_id, panel_id, panel_df, "neutral_ping_readout_composition", _run_mode(seeds), ["condition", "category"])
    sums = _composition_sums(panel_df)
    stats["per_seed_condition_sums"] = sums
    stats["max_abs_sum_deviation_from_100"] = _max_sum_deviation(sums)
    stats["source_level"] = "trial_level" if "trial_level" in source_levels else "metrics_fallback"
    stats["functional_readout_mode"] = _functional_readout_mode(seeds)
    stats["proxy_used_for_main"] = False
    _add_processing_stats(stats, sources, raw_rows, filtered_rows, len(panel_df))
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["state_conditions"] = _unique(panel_df, "state_condition")
    manifest["categories"] = _unique(panel_df, "category")
    manifest["source_level"] = stats["source_level"]
    manifest["functional_readout_mode"] = _functional_readout_mode(seeds)
    manifest["proxy_used_for_main"] = False
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_fig2_partial_cue_target_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    target_items = _partial_target_items(spec)
    if not target_items:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2 partial cue target_items must include A and/or B.")
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2 experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    raw_rows = 0
    filtered_rows = 0
    for seed_dir in seeds:
        curve_path = seed_dir / "data" / "metrics" / "panel_f_partial_cue_metrics.csv"
        raw_path = seed_dir / "data" / "raw" / "panel_f_partial_cue_trial_readout.csv"
        mask_path = seed_dir / "data" / "trial_specs" / "weak_probe_masks.csv"
        summary_path = seed_dir / "summary.json"
        if curve_path.exists():
            df = pd.read_csv(curve_path)
            raw_rows += len(df)
            sources.append(_source_entry(curve_path, repo_root))
            for target_item in target_items:
                part = df[df.get("target_item", pd.Series(dtype=str)).astype(str).str.upper().eq(target_item)].copy()
                filtered_rows += len(part)
                rows.extend(_partial_cue_rows_from_metrics(part, figure_id, panel_id, target_item, curve_path, repo_root, seed_dir, seeds))
            if raw_path.exists():
                sources.append(_source_entry(raw_path, repo_root))
        elif raw_path.exists():
            df = pd.read_csv(raw_path)
            raw_rows += len(df)
            sources.append(_source_entry(raw_path, repo_root))
            for target_item in target_items:
                part = df[df.get("target_item", pd.Series(dtype=str)).astype(str).str.upper().eq(target_item)].copy()
                filtered_rows += len(part)
                rows.extend(_partial_cue_rows_from_trials(part, figure_id, panel_id, target_item, raw_path, repo_root, seed_dir, seeds))
            warnings.append(f"Fig.2{panel_id} used raw partial-cue fallback.")
        else:
            warnings.append(f"Missing partial cue target sources under {_display_path(seed_dir, repo_root)}")
        if mask_path.exists():
            sources.append(_source_entry(mask_path, repo_root))
        if summary_path.exists():
            sources.append(_source_entry(summary_path, repo_root))
    if not any(source["exists"] for source in sources):
        return missing_adapter_result(spec, repo_root, output_dir, "Missing Fig.2 partial cue target sources.")
    panel_df = _sort_partial(pd.DataFrame(rows))
    stats = _stats_payload(figure_id, panel_id, panel_df, "partial_cue_target_recovery", _run_mode(seeds), ["target_item", "condition", "keep_prob"])
    stats["target_item"] = target_items[0] if len(target_items) == 1 else "A+B"
    stats["target_items"] = target_items
    stats["state_conditions"] = _unique(panel_df, "state_condition")
    stats["keep_probs"] = _unique_numeric(panel_df, "keep_prob")
    if len(target_items) == 1:
        target_item = target_items[0]
        stats["relevant_single_condition"] = "S_A" if target_item == "A" else "S_B"
        stats["irrelevant_single_condition"] = "S_B" if target_item == "A" else "S_A"
    stats["functional_readout_mode"] = _functional_readout_mode(seeds)
    stats["proxy_used_for_main"] = False
    _add_processing_stats(stats, sources, raw_rows, filtered_rows, len(panel_df))
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["target_item"] = stats["target_item"]
    manifest["target_items"] = target_items
    manifest["state_conditions"] = stats["state_conditions"]
    manifest["keep_probs"] = stats["keep_probs"]
    manifest["functional_readout_mode"] = _functional_readout_mode(seeds)
    manifest["proxy_used_for_main"] = False
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_fig2_pair_level_and_linear_mixture_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Backward-compatible full Fig.2D adapter for old specs."""
    return build_fig2_morphology_closure_adapter(spec, repo_root, output_dir)


def build_fig2_neutral_ping_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return build_fig2_neutral_ping_composition_adapter(spec, repo_root, output_dir)


def build_fig2_partial_cue_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    panel_spec = dict(spec)
    panel_spec.setdefault("target_item", "A")
    return build_fig2_partial_cue_target_adapter(panel_spec, repo_root, output_dir)


# Backward-compatible adapter names from old Fig.2 specs.
build_fig2_dual_retention_adapter = build_fig2_dual_retention_constituent_adapter
build_fig2_fusion_dual_score = build_fig2_dual_retention_constituent_adapter
build_fig2_pair_specificity = build_fig2_pair_specificity_adapter
build_fig2_wpri = build_fig2_morphology_closure_adapter
build_fig2_neutral_ping_access = build_fig2_neutral_ping_composition_adapter
build_fig2_partial_cue_completion = build_fig2_partial_cue_adapter


def _neutral_ping_rows_from_trials(
    df: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    path: Path,
    repo_root: Path,
    seed_dir: Path,
    seeds: Sequence[Path],
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = df.copy()
    state_col = _first_col(work, ["state_condition", "condition", "memory_state"])
    if state_col is None:
        warnings.append(f"Neutral ping trial source missing state_condition: {_display_path(path, repo_root)}")
        return rows
    seed_col = _first_col(work, ["network_seed", "seed_id", "network_id"])
    if seed_col is None:
        work["network_seed"] = _seed_id(seed_dir)
        seed_col = "network_seed"
    for (seed_value, condition), part in work.groupby([seed_col, state_col], dropna=False):
        cond = str(condition)
        if cond not in STATE_CONDITIONS:
            continue
        total = int(len(part))
        if total <= 0:
            continue
        a_count = int(_bool_count(part, "pred_is_A"))
        b_count = int(_bool_count(part, "pred_is_B"))
        silent_count = int(_silent_count(part))
        other_count = max(total - a_count - b_count - silent_count, 0)
        counts = {"A": a_count, "B": b_count, "Other": other_count, "Silent": silent_count}
        for category in COMPOSITION_CATEGORIES:
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric="neutral_ping_readout_composition",
                    condition=cond,
                    category=category,
                    trace=category,
                    value=float(counts[category]) * 100.0 / float(total),
                    unit="percent",
                    seed_id=seed_value,
                    source_file=_display_path(path, repo_root),
                    state_condition=cond,
                    n_trials=total,
                    source_level="trial_level",
                    run_mode=_run_mode(seeds),
                )
            )
    return rows


def _neutral_ping_rows_from_metrics(
    df: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    path: Path,
    repo_root: Path,
    seed_dir: Path,
    seeds: Sequence[Path],
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    state_col = _first_col(df, ["state_condition", "condition", "memory_state"])
    if state_col is None:
        warnings.append(f"Neutral ping metric source missing state_condition: {_display_path(path, repo_root)}")
        return rows
    seed_col = _first_col(df, ["network_seed", "seed_id", "network_id"])
    work = df.copy()
    if seed_col is None:
        work["network_seed"] = _seed_id(seed_dir)
        seed_col = "network_seed"
    if "P_silent" not in work.columns:
        warnings.append(f"Neutral ping metrics missing P_silent; using 0 and deriving Other: {_display_path(path, repo_root)}")
    for _, row in work.iterrows():
        cond = str(row.get(state_col, ""))
        if cond not in STATE_CONDITIONS:
            continue
        a = _to_percent(row.get("P_A", 0.0))
        b = _to_percent(row.get("P_B", 0.0))
        silent = _to_percent(row.get("P_silent", 0.0)) if "P_silent" in work.columns else 0.0
        other = _to_percent(row.get("P_other", np.nan)) if "P_other" in work.columns else np.nan
        if not np.isfinite(other):
            other = max(100.0 - a - b - silent, 0.0)
        values = {"A": a, "B": b, "Other": other, "Silent": silent}
        total = sum(float(v) for v in values.values() if np.isfinite(v))
        if total > 0 and abs(total - 100.0) > 1e-6:
            values = {key: float(value) * 100.0 / total for key, value in values.items()}
        for category in COMPOSITION_CATEGORIES:
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric="neutral_ping_readout_composition",
                    condition=cond,
                    category=category,
                    trace=category,
                    value=float(values[category]),
                    unit="percent",
                    seed_id=row.get(seed_col, _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    state_condition=cond,
                    n_trials=row.get("n_trials", ""),
                    source_level="metrics_fallback",
                    P_pair=row.get("P_pair", ""),
                    P_A=row.get("P_A", ""),
                    P_B=row.get("P_B", ""),
                    P_other=row.get("P_other", ""),
                    P_silent=row.get("P_silent", ""),
                    run_mode=_run_mode(seeds),
                )
            )
    return rows


def _partial_cue_rows_from_metrics(
    df: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    target_item: str,
    path: Path,
    repo_root: Path,
    seed_dir: Path,
    seeds: Sequence[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        cond = str(row.get("state_condition", ""))
        if cond not in STATE_CONDITIONS or pd.isna(row.get("P_target")):
            continue
        keep_prob = row.get("keep_prob", "")
        value = _to_percent(row.get("P_target", 0.0))
        rows.append(
            _canonical_row(
                figure_id,
                panel_id,
                metric="P_target",
                condition=cond,
                value=value,
                unit="percent",
                seed_id=row.get("network_seed", _seed_id(seed_dir)),
                source_file=_display_path(path, repo_root),
                state_condition=cond,
                display_condition=STATE_LABELS.get(cond, cond),
                target_item=target_item,
                keep_prob=keep_prob,
                x_value=keep_prob,
                y_value=value,
                P_A=row.get("P_A", ""),
                P_B=row.get("P_B", ""),
                P_pair_member=row.get("P_pair_member", ""),
                P_other_pair_member=row.get("P_other_pair_member", ""),
                P_other_class=row.get("P_other_class", ""),
                P_silent=row.get("P_silent", ""),
                n_trials=row.get("n_trials", ""),
                relevant_single_condition=row.get("relevant_single_condition", "S_A" if target_item == "A" else "S_B"),
                irrelevant_single_condition=row.get("irrelevant_single_condition", "S_B" if target_item == "A" else "S_A"),
                curve_or_summary="curve",
                run_mode=_run_mode(seeds),
            )
        )
    return rows


def _partial_cue_rows_from_trials(
    df: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    target_item: str,
    path: Path,
    repo_root: Path,
    seed_dir: Path,
    seeds: Sequence[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {"state_condition", "keep_prob", "pred_is_target"}
    if not required.issubset(df.columns):
        return rows
    seed_col = _first_col(df, ["network_seed", "seed_id", "network_id"])
    work = df.copy()
    if seed_col is None:
        work["network_seed"] = _seed_id(seed_dir)
        seed_col = "network_seed"
    group_cols = [seed_col, "state_condition", "keep_prob"]
    for (seed_value, condition, keep_prob), part in work.groupby(group_cols, dropna=False):
        cond = str(condition)
        if cond not in STATE_CONDITIONS:
            continue
        value = float(pd.to_numeric(part["pred_is_target"], errors="coerce").fillna(0.0).mean()) * 100.0
        rows.append(
            _canonical_row(
                figure_id,
                panel_id,
                metric="P_target",
                condition=cond,
                value=value,
                unit="percent",
                seed_id=seed_value,
                source_file=_display_path(path, repo_root),
                state_condition=cond,
                target_item=target_item,
                keep_prob=keep_prob,
                x_value=keep_prob,
                y_value=value,
                P_A=_optional_rate(part, "pred_is_A"),
                P_B=_optional_rate(part, "pred_is_B"),
                P_pair_member=_optional_rate(part, "pred_is_pair_member"),
                P_other_pair_member=_optional_rate(part, "pred_is_other_pair_member"),
                P_other_class=_optional_rate(part, "pred_is_other_class"),
                P_silent=_optional_rate(part, "silent"),
                n_trials=len(part),
                relevant_single_condition="S_A" if target_item == "A" else "S_B",
                irrelevant_single_condition="S_B" if target_item == "A" else "S_A",
                curve_or_summary="curve",
                source_level="raw_fallback",
                run_mode=_run_mode(seeds),
            )
        )
    return rows


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig2")), str(spec.get("panel_id", "")).upper()


def _partial_target_items(spec: Mapping[str, Any]) -> list[str]:
    raw_items = spec.get("target_items")
    if raw_items is None:
        raw_items = [spec.get("target_item")]
    if isinstance(raw_items, str):
        raw_items = [raw_items]
    out: list[str] = []
    for item in raw_items or []:
        target = str(item).upper()
        if target in {"A", "B"} and target not in out:
            out.append(target)
    return out


def _seed_dirs(spec: Mapping[str, Any], repo_root: Path) -> list[Path]:
    raw_root = spec.get("experiment_root") or spec.get("experiment_root_default") or DEFAULT_EXPERIMENT_ROOT
    root = Path(str(raw_root))
    if not root.is_absolute():
        root = repo_root / root
    if (root / "data" / "metrics").exists() and root.name.startswith("seed_"):
        return [root]
    seeds = sorted(path for path in root.glob("seed_*") if (path / "data" / "metrics").exists())
    if seeds:
        return seeds
    if (root / "data" / "metrics").exists():
        return [root]
    return []


def _run_mode(seeds: Sequence[Path]) -> str:
    return "single_network_draft" if len(seeds) <= 1 else "multi_network_final"


def _run_mode_warnings(seeds: Sequence[Path]) -> list[str]:
    return [SINGLE_NETWORK_WARNING] if _run_mode(seeds) == "single_network_draft" else []


def _seed_id(seed_dir: Path) -> str:
    return seed_dir.name.replace("seed_", "") if seed_dir.name.startswith("seed_") else seed_dir.name


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _source_entry(path: Path, repo_root: Path) -> dict[str, Any]:
    return {"path": _display_path(path, repo_root), "exists": path.exists()}


def _canonical_row(
    figure_id: str,
    panel_id: str,
    *,
    metric: str,
    condition: str,
    value: float,
    unit: str,
    seed_id: Any,
    source_file: str,
    layer: str = "",
    **extra: Any,
) -> dict[str, Any]:
    seed_text = str(seed_id)
    row = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "condition": condition,
        "layer": layer,
        "network_id": seed_text,
        "seed_id": seed_text,
        "value": value,
        "unit": unit,
        "source_file": source_file,
    }
    row.update(extra)
    return row


def _stats_payload(figure_id: str, panel_id: str, panel_df: pd.DataFrame, metric: str, run_mode: str, group_cols: list[str]) -> dict[str, Any]:
    n_networks = _panel_n(panel_df)
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "run_mode": run_mode,
        "n_networks": n_networks,
        "n_networks_observed": n_networks,
        "summaries": summarize_values(panel_df, group_cols),
        "values_used_for_plotting": _values(panel_df),
    }
    if run_mode == "single_network_draft":
        stats["warning"] = SINGLE_NETWORK_WARNING
    return stats


def _add_processing_stats(stats: dict[str, Any], sources: Sequence[Mapping[str, Any]], raw_rows: int, filtered_rows: int, rows_written: int) -> None:
    stats.update(
        {
            "n_source_files": int(sum(1 for source in sources if source.get("exists"))),
            "raw_rows_read": int(raw_rows),
            "rows_after_source_filtering": int(filtered_rows),
            "layer3_rows_before_aggregation": int(filtered_rows),
            "rows_written_to_panel_data": int(rows_written),
            "averaging_performed": False,
            "source_appeared_preaggregated": False,
        }
    )


def _add_seed_aggregation_stats(stats: dict[str, Any], raw_df: pd.DataFrame, panel_df: pd.DataFrame) -> None:
    stats["seed_level_aggregation"] = True
    stats["rows_before_seed_aggregation"] = int(len(raw_df))
    stats["rows_after_seed_aggregation"] = int(len(panel_df))
    stats["replicate_unit"] = "network_seed"
    stats["averaging_performed"] = len(raw_df) != len(panel_df)


def _seed_level_summary(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty or "seed_id" not in df.columns or "value" not in df.columns:
        return df
    keys = ["seed_id", *[col for col in group_cols if col in df.columns]]
    out_rows: list[dict[str, Any]] = []
    for _, part in df.groupby(keys, dropna=False, sort=False):
        values = pd.to_numeric(part["value"], errors="coerce").dropna()
        if values.empty:
            continue
        row = part.iloc[0].to_dict()
        row["value"] = float(values.mean())
        row["seed_level_n"] = int(values.count())
        row["seed_level_sem"] = float(values.sem()) if values.count() > 1 else 0.0
        row["replicate_unit"] = "network_seed"
        if "pair_id" in row:
            row["pair_id"] = "seed_mean"
        out_rows.append(row)
    return pd.DataFrame(out_rows).reset_index(drop=True)


def _manifest(figure_id: str, panel_id: str, sources: list[dict[str, Any]], seeds: Sequence[Path], *, status: str) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": status,
        "run_mode": _run_mode(seeds),
        "n_networks": len(seeds),
        "seed_dirs": [str(path) for path in seeds],
        "sources": sources,
    }


def _functional_readout_mode(seeds: Sequence[Path]) -> str:
    for seed_dir in seeds:
        path = seed_dir / "summary.json"
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                return str(payload.get("functional_readout_mode", "unknown"))
            except Exception:
                return "unknown"
    return "unknown"


def _primary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "layer" in out.columns:
        out = out[out["layer"].astype(str) == "layer3"]
    if "state_variable" in out.columns:
        out = out[out["state_variable"].astype(str) == "g"]
    return out


def _panel_n(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    for col in ("seed_id", "network_id"):
        if col in df.columns:
            return int(df[col].replace("", pd.NA).dropna().nunique())
    return 0


def _values(df: pd.DataFrame) -> list[float]:
    if df.empty or "value" not in df.columns:
        return []
    return [float(v) for v in pd.to_numeric(df["value"], errors="coerce").dropna().tolist()]


def _unique(df: pd.DataFrame, col: str) -> list[Any]:
    if df.empty or col not in df.columns:
        return []
    return [v.item() if hasattr(v, "item") else v for v in pd.Series(df[col]).dropna().drop_duplicates().tolist()]


def _unique_numeric(df: pd.DataFrame, col: str) -> list[float]:
    if df.empty or col not in df.columns:
        return []
    return sorted(set(float(v) for v in pd.to_numeric(df[col], errors="coerce").dropna().tolist()))


def _first_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lower = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _to_number(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _to_percent(value: Any) -> float:
    numeric = _to_number(value)
    if numeric is None:
        return float("nan")
    return float(numeric * 100.0 if abs(numeric) <= 1.5 else numeric)


def _bool_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0.0).gt(0).sum())


def _silent_count(df: pd.DataFrame) -> int:
    silent = pd.Series(False, index=df.index)
    if "silent" in df.columns:
        silent = silent | pd.to_numeric(df["silent"], errors="coerce").fillna(0.0).gt(0)
    pred_col = _first_col(df, ["prediction", "pred"])
    if pred_col is not None:
        silent = silent | pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0).lt(0)
    return int(silent.sum())


def _optional_rate(df: pd.DataFrame, col: str) -> Any:
    if col not in df.columns:
        return ""
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).mean()) * 100.0


def _composition_sums(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or not {"seed_id", "condition", "value"}.issubset(df.columns):
        return []
    sums = df.groupby(["seed_id", "condition"], dropna=False)["value"].sum().reset_index()
    return [{"seed_id": str(row["seed_id"]), "condition": str(row["condition"]), "sum_percent": float(row["value"])} for _, row in sums.iterrows()]


def _max_sum_deviation(sums: Sequence[Mapping[str, Any]]) -> float:
    if not sums:
        return 0.0
    return float(max(abs(float(row.get("sum_percent", 0.0)) - 100.0) for row in sums))


def _sort_conditions(df: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    if df.empty or "condition" not in df.columns:
        return df
    out = df.copy()
    out["_condition_order"] = out["condition"].map({name: i for i, name in enumerate(order)}).fillna(99)
    sort_cols = ["_condition_order"]
    if "seed_id" in out.columns:
        sort_cols.append("seed_id")
    if "pair_id" in out.columns:
        sort_cols.append("pair_id")
    return out.sort_values(sort_cols, kind="stable").drop(columns=["_condition_order"]).reset_index(drop=True)


def _sort_composition(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_condition_order"] = out["condition"].map({name: i for i, name in enumerate(STATE_CONDITIONS)}).fillna(99)
    out["_category_order"] = out["category"].map({name: i for i, name in enumerate(COMPOSITION_CATEGORIES)}).fillna(99)
    return out.sort_values(["_condition_order", "seed_id", "_category_order"], kind="stable").drop(columns=["_condition_order", "_category_order"]).reset_index(drop=True)


def _sort_partial(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_condition_order"] = out["condition"].map({name: i for i, name in enumerate(STATE_CONDITIONS)}).fillna(99)
    out["_keep_prob"] = pd.to_numeric(out.get("keep_prob", pd.Series(dtype=float)), errors="coerce")
    return out.sort_values(["target_item", "_condition_order", "_keep_prob", "seed_id"], kind="stable").drop(columns=["_condition_order", "_keep_prob"]).reset_index(drop=True)
