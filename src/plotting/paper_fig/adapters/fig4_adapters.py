from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult, missing_adapter_result, summarize_values, write_adapter_outputs


DEFAULT_EXPERIMENT_ROOT = Path("results") / "paper_experiments" / "fig4_overlap_reentry"
DRAFT_WARNING = "Single-network result. Use for pipeline validation only, not final manuscript statistics."
RAW_CONDITIONS = (
    "full_dynamic",
    "full_static",
    "sample_keep_overlap_only_dynamic",
    "sample_keep_nonoverlap_only_dynamic",
    "sample_random_matched_dynamic",
)
FIG4D_CONDITION_ORDER = (
    "full_dynamic",
    "sample_keep_overlap_only_dynamic",
    "sample_keep_nonoverlap_only_dynamic",
    "sample_random_matched_dynamic",
    "full_static",
)
CONDITION_LABELS = {
    "full_dynamic": "Dynamic",
    "full_static": "Static",
    "sample_keep_overlap_only_dynamic": "Overlap support",
    "sample_keep_nonoverlap_only_dynamic": "Non-overlap support",
    "sample_random_matched_dynamic": "Random matched",
    "high_overlap": "High overlap",
    "low_overlap": "Low overlap",
}


def build_fig4_similarity_entry_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, f"No Fig.4 seed directories found under {root}.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    metric_sources: set[str] = set()
    fallback_used = False
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_b_similarity_bin_summary.csv"
        pair_path = seed_dir / "data" / "metrics" / "panel_b_similarity_entry_metrics.csv"
        sources.extend([_source(path, seed_dir), _source(pair_path, seed_dir)])
        if not path.exists():
            warnings.append(f"{_display(path, repo_root)} missing.")
            continue
        df = pd.read_csv(path)
        metric_col = _first_existing_col(df, ["mean_acc_drop", "mean_drop_event"])
        if metric_col is None:
            warnings.append(f"{_display(path, repo_root)} lacks mean_acc_drop or mean_drop_event; Fig.4B value missing.")
            continue
        metric_sources.add(str(metric_col))
        if str(metric_col) != "mean_acc_drop":
            fallback_used = True
            warnings.append(f"{_display(path, repo_root)} lacks mean_acc_drop; using mean_drop_event fallback for Fig.4B accuracy drop.")
        for _, r in df.iterrows():
            value = _num(r.get(metric_col, np.nan))
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="accuracy_drop",
                    condition=str(r.get("similarity_bin", "")),
                    layer="L3",
                    seed_id=_seed_id(seed_dir),
                    value=value,
                    unit="probability_delta",
                    source_file=_display(path, repo_root),
                    y_value=value,
                    metric_source=str(metric_col),
                    similarity_bin=str(r.get("similarity_bin", "")),
                    similarity_bin_order=_bin_order(r.get("similarity_bin", "")),
                    x_value=_num(r.get("bin_center", np.nan)),
                    n_pairs=int(_num(r.get("n_pairs", 0))),
                    run_mode=_run_mode(seeds),
                )
            )
    panel_df = pd.DataFrame(rows)
    stats = _stats(figure_id, panel_id, panel_df, _run_mode(seeds), seeds, "accuracy_drop", warnings, ["similarity_bin"])
    stats.update(
        {
            "main_metric": "accuracy_drop",
            "metric_source": _single_or_list(metric_sources),
            "fallback_used": bool(fallback_used),
            "bcd_metric_family": "accuracy_drop",
        }
    )
    manifest = _manifest(figure_id, panel_id, root, seeds, sources, warnings)
    manifest.update({"main_metric": "accuracy_drop", "metric_source": _single_or_list(metric_sources), "fallback_used": bool(fallback_used), "bcd_metric_family": "accuracy_drop"})
    return _write(spec, output_dir, figure_id, panel_id, panel_df, stats, manifest)


def build_fig4_overlap_localization_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, f"No Fig.4 seed directories found under {root}.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    fallback_used = False
    fallback_metrics: set[str] = set()
    for seed_dir in seeds:
        matched_path = seed_dir / "data" / "metrics" / "panel_c_overlap_matched_comparison.csv"
        loc_path = seed_dir / "data" / "metrics" / "panel_c_overlap_localization_metrics.csv"
        reg_path = seed_dir / "data" / "metrics" / "supp_overlap_similarity_regression.csv"
        sources.extend([_source(matched_path, seed_dir), _source(loc_path, seed_dir), _source(reg_path, seed_dir)])
        if matched_path.exists():
            df = pd.read_csv(matched_path)
            metric_col = _first_existing_col(df, ["acc_drop"])
            if metric_col is None:
                metric_col = _first_existing_col(df, ["DPI_L3", "b_vec", "dynamic_effect_metric"])
                fallback_used = True
                if metric_col is not None:
                    fallback_metrics.add(str(metric_col))
                warnings.append("Fig.4C requires acc_drop for accuracy-drop plotting; non-accuracy fallback used.")
            if metric_col is None:
                warnings.append(f"{_display(matched_path, repo_root)} lacks acc_drop and fallback metric columns.")
                continue
            for _, r in df.iterrows():
                raw_group = str(r.get("overlap_group", ""))
                value = _num(r.get(metric_col, np.nan))
                rows.append(
                    _canonical(
                        figure_id,
                        panel_id,
                        metric="accuracy_drop" if str(metric_col) == "acc_drop" else str(metric_col),
                        condition=CONDITION_LABELS.get(raw_group, raw_group.replace("_", " ").title()),
                        layer="readout",
                        seed_id=_seed_id(seed_dir),
                        value=value,
                        unit="probability_delta" if str(metric_col) == "acc_drop" else "fallback_index",
                        source_file=_display(matched_path, repo_root),
                        y_value=value,
                        metric_source=str(metric_col),
                        pair_id=int(_num(r.get("pair_id", -1))),
                        matched_group_id=str(r.get("matched_group_id", "")),
                        overlap_group=raw_group,
                        pixel_similarity=_num(r.get("pixel_similarity", np.nan)),
                        dice_overlap=_num(r.get("dice_overlap", np.nan)),
                        run_mode=_run_mode(seeds),
                    )
                )
        elif loc_path.exists():
            warnings.append(f"{_display(matched_path, repo_root)} missing; using overlap localization distribution fallback.")
            df = pd.read_csv(loc_path)
            metric_col = _first_existing_col(df, ["acc_drop"])
            if metric_col is None:
                metric_col = _first_existing_col(df, ["DPI_L3", "dynamic_effect_metric", "b_vec"])
                fallback_used = True
                if metric_col is not None:
                    fallback_metrics.add(str(metric_col))
                warnings.append("Fig.4C requires acc_drop for accuracy-drop plotting; non-accuracy fallback used.")
            if metric_col is None:
                warnings.append(f"{_display(loc_path, repo_root)} lacks acc_drop and fallback metric columns.")
                continue
            median = float(pd.to_numeric(df.get("dice_overlap", pd.Series(dtype=float)), errors="coerce").median())
            for _, r in df.iterrows():
                raw_group = "high_overlap" if _num(r.get("dice_overlap", 0.0)) >= median else "low_overlap"
                value = _num(r.get(metric_col, np.nan))
                rows.append(
                    _canonical(
                        figure_id,
                        panel_id,
                        metric="accuracy_drop" if str(metric_col) == "acc_drop" else str(metric_col),
                        condition=CONDITION_LABELS[raw_group],
                        layer="readout",
                        seed_id=_seed_id(seed_dir),
                        value=value,
                        unit="probability_delta" if str(metric_col) == "acc_drop" else "fallback_index",
                        source_file=_display(loc_path, repo_root),
                        y_value=value,
                        metric_source=str(metric_col),
                        pair_id=int(_num(r.get("pair_id", -1))),
                        matched_group_id=str(r.get("matched_group_id", "")),
                        overlap_group=raw_group,
                        pixel_similarity=_num(r.get("pixel_similarity", np.nan)),
                        dice_overlap=_num(r.get("dice_overlap", np.nan)),
                        run_mode=_run_mode(seeds),
                    )
                )
        else:
            warnings.append(f"{_display(matched_path, repo_root)} and {_display(loc_path, repo_root)} missing.")
    panel_df = pd.DataFrame(rows)
    manifest = _manifest(figure_id, panel_id, root, seeds, sources, warnings)
    manifest["overlap_specific_controls"] = bool(any(src.get("path", "").endswith("panel_c_overlap_matched_comparison.csv") and src.get("exists") for src in sources))
    manifest["regression_ready"] = bool(any(src.get("path", "").endswith("supp_overlap_similarity_regression.csv") and src.get("exists") for src in sources))
    stats = _stats(figure_id, panel_id, panel_df, _run_mode(seeds), seeds, "accuracy_drop", warnings, ["condition"])
    stats.update(_overlap_accuracy_drop_stats(panel_df))
    stats.update(
        {
            "main_metric": "accuracy_drop",
            "fallback_used": bool(fallback_used),
            "fallback_metric": _single_or_list(fallback_metrics),
            "bcd_metric_family": "accuracy_drop",
        }
    )
    manifest.update({"main_metric": "accuracy_drop", "fallback_used": bool(fallback_used), "fallback_metric": _single_or_list(fallback_metrics), "bcd_metric_family": "accuracy_drop"})
    return _write(spec, output_dir, figure_id, panel_id, panel_df, stats, manifest)


def build_fig4_decision_spike_displacement_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, f"No Fig.4 seed directories found under {root}.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    max_time_ms = float(spec.get("max_time_ms", 60))
    keep = set(spec.get("conditions") or ["sample_keep_overlap_only_dynamic", "sample_keep_nonoverlap_only_dynamic"])
    before_filter = 0
    after_filter = 0
    original_times: list[float] = []
    plotted_times: list[float] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_e_time_resolved_l3_displacement.csv"
        summary_path = seed_dir / "data" / "metrics" / "panel_e_decision_spike_displacement.csv"
        sources.extend([_source(path, seed_dir), _source(summary_path, seed_dir)])
        if path.exists():
            df = pd.read_csv(path)
            df = df[df["condition"].isin(keep)]
            before_filter += int(len(df))
            df = _ensure_time_ms(df, spec, warnings, _display(path, repo_root))
            if "time_ms" in df.columns:
                original_times.extend(pd.to_numeric(df["time_ms"], errors="coerce").dropna().tolist())
                df = df[pd.to_numeric(df["time_ms"], errors="coerce") <= max_time_ms].copy()
                plotted_times.extend(pd.to_numeric(df["time_ms"], errors="coerce").dropna().tolist())
            else:
                warnings.append(f"{_display(path, repo_root)} lacks time_ms and time_step/dt_ms; Fig.4E time filter could not be applied.")
            after_filter += int(len(df))
            for _, r in df.iterrows():
                raw = str(r.get("condition", ""))
                rows.append(
                    _canonical(
                        figure_id,
                        panel_id,
                        metric="DPI_L3_t",
                        condition=CONDITION_LABELS.get(raw, raw),
                        layer="L3",
                        seed_id=_seed_id(seed_dir),
                        value=_num(r.get("DPI_L3_t", np.nan)),
                        unit="index",
                        source_file=_display(path, repo_root),
                        raw_condition=raw,
                        pair_id=int(_num(r.get("pair_id", -1))),
                        time_step=int(_num(r.get("time_step", 0))),
                        time_ms=_num(r.get("time_ms", np.nan)),
                        similarity_bin=str(r.get("similarity_bin", "")),
                        overlap_bin=str(r.get("overlap_bin", "")),
                        run_mode=_run_mode(seeds),
                    )
                )
        elif summary_path.exists():
            warnings.append(f"{_display(path, repo_root)} missing; using summary displacement fallback.")
            df = pd.read_csv(summary_path)
            before_filter += int(len(df))
            df = df[df["condition"].isin(keep)]
            after_filter += int(len(df))
            for _, r in df.iterrows():
                raw = str(r.get("condition", ""))
                rows.append(
                    _canonical(
                        figure_id,
                        panel_id,
                        metric="mean_DPI_L3",
                        condition=CONDITION_LABELS.get(raw, raw),
                        layer="L3",
                        seed_id=_seed_id(seed_dir),
                        value=_num(r.get("mean_DPI_L3", np.nan)),
                        unit="index",
                        source_file=_display(summary_path, repo_root),
                        raw_condition=raw,
                        pair_id=int(_num(r.get("pair_id", -1))),
                        run_mode=_run_mode(seeds),
                    )
                )
        else:
            warnings.append(f"{_display(path, repo_root)} missing.")
    panel_df = pd.DataFrame(rows)
    stats = _stats(figure_id, panel_id, panel_df, _run_mode(seeds), seeds, "DPI_L3_t", warnings, ["condition", "time_ms"])
    stats.update(
        {
            "max_time_ms_used": max_time_ms,
            "original_time_range_ms": _range_or_none(original_times),
            "plotted_time_range_ms": _range_or_none(plotted_times),
            "n_rows_before_time_filter": before_filter,
            "n_rows_after_time_filter": after_filter,
            "fig4e_max_time_ms": max_time_ms,
        }
    )
    manifest = _manifest(figure_id, panel_id, root, seeds, sources, warnings)
    manifest.update({"fig4e_max_time_ms": max_time_ms, "main_conditions": sorted(keep)})
    return _write(spec, output_dir, figure_id, panel_id, panel_df, stats, manifest)


def build_fig4_overlap_accuracy_identification_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, f"No Fig.4 seed directories found under {root}.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        contrast_path = seed_dir / "data" / "metrics" / "panel_d_overlap_accuracy_contrast_by_network.csv"
        matches_path = seed_dir / "data" / "metrics" / "panel_d_iso_similarity_matched_pairs.csv"
        balance_path = seed_dir / "data" / "metrics" / "panel_d_matching_balance_diagnostics.csv"
        sources.extend([_source(contrast_path, seed_dir), _source(matches_path, seed_dir), _source(balance_path, seed_dir)])
        if contrast_path.exists():
            contrast = pd.read_csv(contrast_path)
            for _, r in contrast.iterrows():
                for metric, unit in (("delta_drop_rate", "probability_delta"), ("drop_rate_high_overlap", "probability"), ("drop_rate_low_overlap", "probability"), ("delta_acc_drop", "accuracy_delta")):
                    value = _num(r.get(metric, np.nan))
                    if np.isfinite(value):
                        rows.append(
                            _canonical(
                                figure_id,
                                panel_id,
                                metric=metric,
                                condition="matched_high_minus_low" if metric.startswith("delta") else metric.replace("drop_rate_", ""),
                                layer="readout",
                                seed_id=_seed_id(seed_dir),
                                value=value,
                                unit=unit,
                                source_file=_display(contrast_path, repo_root),
                                n_matched_sets=int(_num(r.get("n_matched_sets", 0))),
                                permutation_p_one_sided=_num(r.get("permutation_p_one_sided", np.nan)),
                                permutation_p_two_sided=_num(r.get("permutation_p_two_sided", np.nan)),
                                run_mode=_run_mode(seeds),
                            )
                        )
        else:
            warnings.append(f"{_display(contrast_path, repo_root)} missing.")
        if matches_path.exists():
            matches = pd.read_csv(matches_path)
            for _, r in matches.iterrows():
                value = _num(r.get("paired_delta_drop_event", np.nan))
                rows.append(
                    _canonical(
                        figure_id,
                        panel_id,
                        metric="paired_delta_drop_event",
                        condition="matched_pair",
                        layer="readout",
                        seed_id=_seed_id(seed_dir),
                        value=value,
                        unit="probability_delta",
                        source_file=_display(matches_path, repo_root),
                        match_id=int(_num(r.get("match_id", -1))),
                        high_pair_id=int(_num(r.get("high_pair_id", -1))),
                        low_pair_id=int(_num(r.get("low_pair_id", -1))),
                        pixel_similarity_high=_num(r.get("pixel_similarity_high", np.nan)),
                        pixel_similarity_low=_num(r.get("pixel_similarity_low", np.nan)),
                        similarity_difference=_num(r.get("similarity_difference", np.nan)),
                        dice_overlap_high=_num(r.get("dice_overlap_high", np.nan)),
                        dice_overlap_low=_num(r.get("dice_overlap_low", np.nan)),
                        sample_energy_rel_difference=_num(r.get("sample_energy_rel_difference", np.nan)),
                        probe_energy_rel_difference=_num(r.get("probe_energy_rel_difference", np.nan)),
                        drop_event_high=_num(r.get("drop_event_high", np.nan)),
                        drop_event_low=_num(r.get("drop_event_low", np.nan)),
                        paired_delta_drop_event=value,
                        n_matched_sets=len(matches),
                        run_mode=_run_mode(seeds),
                    )
                )
        if balance_path.exists():
            balance = pd.read_csv(balance_path)
            for _, r in balance.iterrows():
                for metric in ("mean_similarity_difference", "max_similarity_difference", "mean_sample_energy_rel_difference", "mean_probe_energy_rel_difference"):
                    value = _num(r.get(metric, np.nan))
                    if np.isfinite(value):
                        rows.append(
                            _canonical(
                                figure_id,
                                panel_id,
                                metric=metric,
                                condition="matching_balance",
                                layer="metadata",
                                seed_id=_seed_id(seed_dir),
                                value=value,
                                unit="difference",
                                source_file=_display(balance_path, repo_root),
                                n_matched_sets=int(_num(r.get("n_matched_sets", 0))),
                                run_mode=_run_mode(seeds),
                            )
                        )
    panel_df = pd.DataFrame(rows)
    return _write(spec, output_dir, figure_id, panel_id, panel_df, _stats(figure_id, panel_id, panel_df, _run_mode(seeds), seeds, "delta_drop_rate", warnings, ["condition"]), _manifest(figure_id, panel_id, root, seeds, sources, warnings))


def build_fig4_decision_deflection_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, f"No Fig.4 seed directories found under {root}.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_f_decision_deflection_metrics.csv"
        sources.append(_source(path, seed_dir))
        if not path.exists():
            warnings.append(f"{_display(path, repo_root)} missing.")
            continue
        df = pd.read_csv(path)
        df = df[df["condition"].isin(RAW_CONDITIONS)]
        for _, r in df.iterrows():
            raw = str(r.get("condition", ""))
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="decision_deflection_score",
                    condition=CONDITION_LABELS.get(raw, raw),
                    layer="readout",
                    seed_id=_seed_id(seed_dir),
                    value=_num(r.get("decision_deflection_score", r.get("dynamic_like_recovery", np.nan))),
                    unit="projection",
                    source_file=_display(path, repo_root),
                    raw_condition=raw,
                    pair_id=int(_num(r.get("pair_id", -1))),
                    x_value=_num(r.get("static_to_dynamic_push", r.get("x1", np.nan))),
                    y_value=_num(r.get("dynamic_like_recovery", r.get("y1", np.nan))),
                    static_to_dynamic_push=_num(r.get("static_to_dynamic_push", np.nan)),
                    dynamic_like_recovery=_num(r.get("dynamic_like_recovery", np.nan)),
                    decision_deflection_score=_num(r.get("decision_deflection_score", np.nan)),
                    run_mode=_run_mode(seeds),
                )
            )
    panel_df = pd.DataFrame(rows)
    return _write(spec, output_dir, figure_id, panel_id, panel_df, _stats(figure_id, panel_id, panel_df, _run_mode(seeds), seeds, "decision_deflection_score", warnings, ["condition"]), _manifest(figure_id, panel_id, root, seeds, sources, warnings))


def build_fig4_overlap_perturbation_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, f"No Fig.4 seed directories found under {root}.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "supp_overlap_preserving_perturbation_metrics.csv"
        summary_path = seed_dir / "data" / "metrics" / "supp_overlap_preserving_perturbation_summary.csv"
        mask_path = seed_dir / "data" / "trial_specs" / "perturbation_masks.csv"
        random_path = seed_dir / "data" / "metrics" / "supp_random_mask_perturbation_controls.csv"
        sources.extend([_source(path, seed_dir), _source(summary_path, seed_dir), _source(mask_path, seed_dir), _source(random_path, seed_dir)])
        if not path.exists():
            warnings.append(f"{_display(path, repo_root)} missing.")
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            raw = str(r.get("condition", ""))
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="dynamic_like_recovery",
                    condition=CONDITION_LABELS.get(raw, raw),
                    layer="L3/readout",
                    seed_id=_seed_id(seed_dir),
                    value=_num(r.get("dynamic_like_recovery", r.get("DPI_L3", np.nan))),
                    unit="index",
                    source_file=_display(path, repo_root),
                    raw_condition=raw,
                    pair_id=int(_num(r.get("pair_id", -1))),
                    DPI_L3=_num(r.get("DPI_L3", np.nan)),
                    decision_deflection_score=_num(r.get("decision_deflection_score", np.nan)),
                    probe_accuracy=_num(r.get("probe_accuracy", np.nan)),
                    run_mode=_run_mode(seeds),
                )
            )
    panel_df = pd.DataFrame(rows)
    manifest = _manifest(figure_id, panel_id, root, seeds, sources, warnings)
    manifest["perturbation_scope"] = "sample_side_prior_support"
    manifest["probe_input_modified_in_core_conditions"] = False
    manifest["random_matched_mask_diagnostics_present"] = bool(any(src.get("path", "").endswith("supp_random_mask_perturbation_controls.csv") and src.get("exists") for src in sources))
    return _write(spec, output_dir, figure_id, panel_id, panel_df, _stats(figure_id, panel_id, panel_df, _run_mode(seeds), seeds, "dynamic_like_recovery", warnings, ["condition"]), manifest)


def build_fig4_overlap_perturbation_main_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, f"No Fig.4 seed directories found under {root}.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    checked_candidates: list[str] = []
    used_sources: list[str] = []
    source_levels: set[str] = set()
    fallback_used = False
    keep_conditions = set(RAW_CONDITIONS)
    for seed_dir in seeds:
        candidates = [
            (seed_dir / "data" / "metrics" / "panel_d_overlap_perturbation_metrics.csv", "main_panel_d"),
            (seed_dir / "data" / "metrics" / "panel_d_overlap_perturbation_summary.csv", "main_panel_d_summary"),
            (seed_dir / "data" / "metrics" / "supp_overlap_preserving_perturbation_metrics.csv", "supplement_fallback"),
            (seed_dir / "data" / "metrics" / "supp_overlap_preserving_perturbation_summary.csv", "supplement_fallback_summary"),
        ]
        companion_paths = [
            seed_dir / "data" / "trial_specs" / "panel_d_perturbation_masks.csv",
            seed_dir / "data" / "trial_specs" / "perturbation_masks.csv",
            seed_dir / "data" / "metrics" / "supp_random_mask_perturbation_controls.csv",
        ]
        for path, _level in candidates:
            sources.append(_source(path, seed_dir))
            checked_candidates.append(_display(path, repo_root))
        for path in companion_paths:
            sources.append(_source(path, seed_dir))
            checked_candidates.append(_display(path, repo_root))
        selected = next(((path, level) for path, level in candidates if path.exists()), None)
        if selected is None:
            warnings.append(f"No overlap perturbation source found under {_display(seed_dir, repo_root)}.")
            continue
        path, source_level = selected
        used_sources.append(_display(path, repo_root))
        for companion in companion_paths:
            if companion.exists():
                used_sources.append(_display(companion, repo_root))
        source_levels.add(source_level)
        if source_level.startswith("supplement"):
            fallback_used = True
            warnings.append(f"{_display(path, repo_root)} used as Fig.4D supplement fallback.")
        df = pd.read_csv(path)
        condition_col = _first_existing_col(df, ["condition", "raw_condition", "perturbation_condition", "summary_group"])
        if condition_col is None:
            warnings.append(f"{_display(path, repo_root)} has no condition column.")
            continue
        df = df[df[condition_col].astype(str).isin(keep_conditions)].copy()
        if source_level in {"main_panel_d", "supplement_fallback"}:
            required = {"pair_id", condition_col, "probe_accuracy"}
            missing = sorted(col for col in required if col not in df.columns)
            if missing:
                warnings.append(f"{_display(path, repo_root)} missing pair-level accuracy-drop columns {missing}.")
                continue
            _append_pair_level_accuracy_drop_rows(rows, df, condition_col, figure_id, panel_id, seed_dir, path, repo_root, source_level, seeds, warnings)
        else:
            if "mean_probe_accuracy" not in df.columns:
                warnings.append(f"{_display(path, repo_root)} missing mean_probe_accuracy for summary-level accuracy-drop plotting.")
                continue
            _append_summary_accuracy_drop_rows(rows, df, condition_col, figure_id, panel_id, seed_dir, path, repo_root, source_level, seeds, warnings)
    panel_df = pd.DataFrame(rows)
    stats = _stats(figure_id, panel_id, panel_df, _run_mode(seeds), seeds, "accuracy_drop_vs_static", warnings, ["condition"])
    stats["source_level"] = _source_level_label(source_levels)
    stats.update(_accuracy_drop_condition_contrasts(panel_df))
    stats.update(
        {
            "main_metric": "accuracy_drop_vs_static",
            "fallback_used": bool(fallback_used),
            "bcd_metric_family": "accuracy_drop",
            "fig4d_static_baseline_used": bool("probe_accuracy_static" in panel_df.columns and pd.to_numeric(panel_df["probe_accuracy_static"], errors="coerce").dropna().size),
        }
    )
    manifest = _manifest(figure_id, panel_id, root, seeds, sources, warnings)
    manifest["perturbation_scope"] = "sample_side_prior_support"
    manifest["probe_input_modified_in_core_conditions"] = False
    manifest["main_metric"] = "accuracy_drop_vs_static"
    manifest["previous_metrics_not_used_for_plotting"] = ["DPI_L3", "dynamic_like_recovery", "decision_deflection_score"]
    manifest["fallback_used"] = bool(fallback_used)
    manifest["bcd_metric_family"] = "accuracy_drop"
    manifest["fig4d_static_baseline_used"] = stats["fig4d_static_baseline_used"]
    manifest["random_matched_mask_diagnostics_present"] = bool(any(("random" in src.get("path", "").lower()) and src.get("exists") for src in sources))
    manifest["checked_candidates"] = checked_candidates
    manifest["source_files_used"] = sorted(set(used_sources))
    manifest["source_level"] = stats["source_level"]
    return _write(spec, output_dir, figure_id, panel_id, panel_df, stats, manifest)


def build_fig4_l3_accumulator_process_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, f"No Fig.4 seed directories found under {root}.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    checked_candidates: list[str] = []
    used_sources: list[str] = []
    scale_records: list[dict[str, float]] = []
    fallback_used = False
    required_keys_seen: set[str] = set()
    for seed_dir in seeds:
        result_path, vector_path, source_level, checked = _find_accumulator_sources(seed_dir, repo_root)
        checked_candidates.extend(checked)
        for raw in checked:
            path = Path(raw)
            sources.append({"path": raw.replace("\\", "/"), "seed_id": _seed_id(seed_dir), "exists": path.exists()})
        if result_path is None or vector_path is None:
            fallback = seed_dir / "data" / "metrics" / "panel_f_decision_deflection_metrics.csv"
            sources.append(_source(fallback, seed_dir))
            checked_candidates.append(_display(fallback, repo_root))
            if fallback.exists():
                fallback_used = True
                warnings.append(f"Accumulator process files missing for {_seed_id(seed_dir)}; using decision-deflection fallback.")
                _append_decision_deflection_fallback(rows, pd.read_csv(fallback), figure_id, panel_id, seed_dir, fallback, repo_root, seeds)
            else:
                warnings.append(f"Accumulator process sources missing under {_display(seed_dir, repo_root)}.")
            continue
        used_sources.extend([_display(result_path, repo_root), _display(vector_path, repo_root)])
        try:
            result_df = pd.read_csv(result_path)
            vectors = np.load(vector_path, allow_pickle=False)
            available = set(vectors.files)
            required_keys_seen.update(available)
            table, scales = _build_accumulator_process_rows(result_df, vectors, warnings, _display(result_path, repo_root))
        except Exception as exc:
            warnings.append(f"Accumulator process adapter failed for {_display(seed_dir, repo_root)}: {exc}")
            continue
        scale_records.append(scales)
        if source_level != "panel_f_accumulator":
            fallback_used = True
        for _, r in table.iterrows():
            group = str(r["group"])
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="accumulator_process_shift",
                    condition=group,
                    layer="L3/readout",
                    seed_id=str(result_df.iloc[0].get("network_seed", _seed_id(seed_dir))) if not result_df.empty else _seed_id(seed_dir),
                    value=float(np.hypot(float(r["x1"]) - float(r["x0"]), float(r["y1"]) - float(r["y0"]))),
                    unit="normalized_coordinate",
                    source_file=_display(result_path, repo_root),
                    pair_id=int(r["pair_id"]),
                    group=group,
                    x0=float(r["x0"]),
                    y0=float(r["y0"]),
                    x1=float(r["x1"]),
                    y1=float(r["y1"]),
                    before_x=float(r["x0"]),
                    before_y=float(r["y0"]),
                    after_x=float(r["x1"]),
                    after_y=float(r["y1"]),
                    fire_shift=float(r["fire_shift"]),
                    decision_shift=float(r["decision_shift"]),
                    run_mode=_run_mode(seeds),
                    source_level=source_level,
                )
            )
    panel_df = pd.DataFrame(rows)
    stats = _stats(figure_id, panel_id, panel_df, _run_mode(seeds), seeds, "accumulator_process_shift", warnings, ["group"])
    stats.update(_accumulator_stats(panel_df, scale_records))
    stats["fallback_used"] = fallback_used
    manifest = _manifest(figure_id, panel_id, root, seeds, sources, warnings)
    manifest["checked_candidates"] = checked_candidates
    manifest["source_files_used"] = sorted(set(used_sources))
    manifest["required_vector_keys_present"] = sorted(required_keys_seen)
    manifest["fallback_used"] = fallback_used
    manifest["trajectory_logic_source"] = "src.plotting.experiments.l3_accumulator_mechanism_experiment_plot"
    return _write(spec, output_dir, figure_id, panel_id, panel_df, stats, manifest)


def _first_existing_col(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    lower = {str(col).lower(): str(col) for col in df.columns}
    for name in names:
        if str(name).lower() in lower:
            return lower[str(name).lower()]
    return None


def _source_level_label(levels: set[str]) -> str:
    if not levels:
        return "missing_source"
    if any(level.startswith("main_panel_d") for level in levels) and any(level.startswith("supplement") for level in levels):
        return "mixed_main_and_supplement_fallback"
    if any(level.startswith("main_panel_d") for level in levels):
        return "main_panel_d"
    return "supplement_fallback"


def _single_or_list(values: set[str]) -> str | list[str]:
    clean = sorted(v for v in values if v)
    if len(clean) == 1:
        return clean[0]
    return clean


def _range_or_none(values: Sequence[float]) -> list[float] | None:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    if not finite:
        return None
    return [float(min(finite)), float(max(finite))]


def _ensure_time_ms(df: pd.DataFrame, spec: Mapping[str, Any], warnings: list[str], source_label: str) -> pd.DataFrame:
    out = df.copy()
    if "time_ms" in out.columns and pd.to_numeric(out["time_ms"], errors="coerce").notna().any():
        out["time_ms"] = pd.to_numeric(out["time_ms"], errors="coerce")
        return out
    dt_ms = spec.get("dt_ms")
    if dt_ms is None:
        dt_ms = spec.get("time_step_ms")
    if dt_ms is None and "dt_ms" in out.columns:
        dt_ms = pd.to_numeric(out["dt_ms"], errors="coerce").dropna().iloc[0] if pd.to_numeric(out["dt_ms"], errors="coerce").dropna().size else None
    if "time_step" in out.columns and dt_ms is not None:
        out["time_ms"] = pd.to_numeric(out["time_step"], errors="coerce") * float(dt_ms)
        warnings.append(f"{source_label} lacks usable time_ms; inferred time_ms from time_step x dt_ms.")
        return out
    warnings.append(f"{source_label} lacks usable time_ms; unable to infer from time_step x dt_ms.")
    return out


def _overlap_accuracy_drop_stats(panel_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mean_acc_drop_high_overlap": np.nan,
        "mean_acc_drop_low_overlap": np.nan,
        "high_minus_low_acc_drop": np.nan,
        "n_matched_groups": 0,
        "n_pairs": 0,
        "matched_groups_available": False,
    }
    if panel_df.empty:
        return out
    values = pd.to_numeric(panel_df.get("value", pd.Series(dtype=float)), errors="coerce")
    work = panel_df.copy()
    work["value"] = values
    grouped = work.groupby("overlap_group", dropna=False)["value"].mean() if "overlap_group" in work.columns else pd.Series(dtype=float)
    high = grouped.get("high_overlap", np.nan)
    low = grouped.get("low_overlap", np.nan)
    out["mean_acc_drop_high_overlap"] = float(high) if np.isfinite(high) else np.nan
    out["mean_acc_drop_low_overlap"] = float(low) if np.isfinite(low) else np.nan
    if np.isfinite(high) and np.isfinite(low):
        out["high_minus_low_acc_drop"] = float(high - low)
    if "matched_group_id" in work.columns:
        nonempty = work["matched_group_id"].replace("", pd.NA).dropna()
        out["n_matched_groups"] = int(nonempty.nunique())
        out["matched_groups_available"] = bool(out["n_matched_groups"])
    if "pair_id" in work.columns:
        out["n_pairs"] = int(work["pair_id"].replace("", pd.NA).dropna().nunique())
    return out


def _append_pair_level_accuracy_drop_rows(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    condition_col: str,
    figure_id: str,
    panel_id: str,
    seed_dir: Path,
    source_path: Path,
    repo_root: Path,
    source_level: str,
    seeds: Sequence[Path],
    warnings: list[str],
) -> None:
    work = df.copy()
    network_col = _first_existing_col(work, ["network_seed", "network_id", "seed_id"])
    if network_col is None:
        work["_network_key"] = _seed_id(seed_dir)
        network_col = "_network_key"
    work["_probe_accuracy"] = pd.to_numeric(work["probe_accuracy"], errors="coerce")
    static_rows = work[work[condition_col].astype(str).eq("full_static")]
    if static_rows.empty:
        warnings.append(f"{_display(source_path, repo_root)} lacks full_static probe_accuracy rows for Fig.4D baseline.")
        return
    static_lookup = static_rows.groupby([network_col, "pair_id"], dropna=False)["_probe_accuracy"].mean().to_dict()
    for _, r in work.iterrows():
        raw = str(r.get(condition_col, ""))
        if raw not in RAW_CONDITIONS:
            continue
        key = (r.get(network_col), r.get("pair_id"))
        static_acc = _num(static_lookup.get(key, np.nan))
        condition_acc = _num(r.get("probe_accuracy", np.nan))
        if not np.isfinite(static_acc) or not np.isfinite(condition_acc):
            continue
        value = float(static_acc - condition_acc)
        rows.append(
            _canonical(
                figure_id,
                panel_id,
                metric="accuracy_drop_vs_static",
                condition="Static baseline" if raw == "full_static" else CONDITION_LABELS.get(raw, raw),
                layer="readout",
                seed_id=str(r.get(network_col, _seed_id(seed_dir))),
                value=value,
                unit="probability_delta",
                source_file=_display(source_path, repo_root),
                y_value=value,
                raw_condition=raw,
                condition_order=FIG4D_CONDITION_ORDER.index(raw) if raw in FIG4D_CONDITION_ORDER else 99,
                pair_id=_maybe_int(r.get("pair_id", "")),
                probe_accuracy_condition=condition_acc,
                probe_accuracy_static=static_acc,
                run_mode=_run_mode(seeds),
                source_level=source_level,
            )
        )


def _append_summary_accuracy_drop_rows(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    condition_col: str,
    figure_id: str,
    panel_id: str,
    seed_dir: Path,
    source_path: Path,
    repo_root: Path,
    source_level: str,
    seeds: Sequence[Path],
    warnings: list[str],
) -> None:
    work = df.copy()
    static_values = pd.to_numeric(work.loc[work[condition_col].astype(str).eq("full_static"), "mean_probe_accuracy"], errors="coerce").dropna()
    if static_values.empty:
        warnings.append(f"{_display(source_path, repo_root)} lacks full_static mean_probe_accuracy for Fig.4D baseline.")
        return
    static_acc = float(static_values.mean())
    network_col = _first_existing_col(work, ["network_seed", "network_id", "seed_id"])
    for _, r in work.iterrows():
        raw = str(r.get(condition_col, ""))
        if raw not in RAW_CONDITIONS:
            continue
        condition_acc = _num(r.get("mean_probe_accuracy", np.nan))
        if not np.isfinite(condition_acc):
            continue
        value = float(static_acc - condition_acc)
        rows.append(
            _canonical(
                figure_id,
                panel_id,
                metric="accuracy_drop_vs_static",
                condition="Static baseline" if raw == "full_static" else CONDITION_LABELS.get(raw, raw),
                layer="readout",
                seed_id=str(r.get(network_col, _seed_id(seed_dir))) if network_col else _seed_id(seed_dir),
                value=value,
                unit="probability_delta",
                source_file=_display(source_path, repo_root),
                y_value=value,
                raw_condition=raw,
                condition_order=FIG4D_CONDITION_ORDER.index(raw) if raw in FIG4D_CONDITION_ORDER else 99,
                n_pairs=int(_num(r.get("n_pairs", 0))),
                probe_accuracy_condition=condition_acc,
                probe_accuracy_static=static_acc,
                run_mode=_run_mode(seeds),
                source_level=source_level,
                source_level_detail="summary",
            )
        )


def _accuracy_drop_condition_contrasts(panel_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "static_probe_accuracy": np.nan,
        "accuracy_drop_dynamic": np.nan,
        "accuracy_drop_overlap": np.nan,
        "accuracy_drop_nonoverlap": np.nan,
        "accuracy_drop_random": np.nan,
        "overlap_minus_nonoverlap_accuracy_drop": np.nan,
        "overlap_minus_random_accuracy_drop": np.nan,
        "dynamic_minus_static_accuracy_drop": np.nan,
    }
    if panel_df.empty or "raw_condition" not in panel_df.columns:
        return out
    grouped = panel_df.groupby("raw_condition", dropna=False)["value"].mean()
    static_acc = pd.to_numeric(panel_df.get("probe_accuracy_static", pd.Series(dtype=float)), errors="coerce").dropna()
    if not static_acc.empty:
        out["static_probe_accuracy"] = float(static_acc.mean())
    mapping = {
        "accuracy_drop_dynamic": "full_dynamic",
        "accuracy_drop_overlap": "sample_keep_overlap_only_dynamic",
        "accuracy_drop_nonoverlap": "sample_keep_nonoverlap_only_dynamic",
        "accuracy_drop_random": "sample_random_matched_dynamic",
    }
    for out_key, raw in mapping.items():
        value = grouped.get(raw, np.nan)
        out[out_key] = float(value) if np.isfinite(value) else np.nan
    if np.isfinite(out["accuracy_drop_overlap"]) and np.isfinite(out["accuracy_drop_nonoverlap"]):
        out["overlap_minus_nonoverlap_accuracy_drop"] = float(out["accuracy_drop_overlap"] - out["accuracy_drop_nonoverlap"])
    if np.isfinite(out["accuracy_drop_overlap"]) and np.isfinite(out["accuracy_drop_random"]):
        out["overlap_minus_random_accuracy_drop"] = float(out["accuracy_drop_overlap"] - out["accuracy_drop_random"])
    static_drop = grouped.get("full_static", 0.0)
    if np.isfinite(out["accuracy_drop_dynamic"]) and np.isfinite(static_drop):
        out["dynamic_minus_static_accuracy_drop"] = float(out["accuracy_drop_dynamic"] - static_drop)
    return out


def _condition_contrasts(panel_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if panel_df.empty or "raw_condition" not in panel_df.columns:
        return out
    grouped = panel_df.groupby("raw_condition", dropna=False)["value"].mean()
    overlap = grouped.get("sample_keep_overlap_only_dynamic", np.nan)
    non = grouped.get("sample_keep_nonoverlap_only_dynamic", np.nan)
    random = grouped.get("sample_random_matched_dynamic", np.nan)
    if np.isfinite(overlap) and np.isfinite(non):
        out["overlap_minus_nonoverlap"] = float(overlap - non)
    if np.isfinite(overlap) and np.isfinite(random):
        out["overlap_minus_random"] = float(overlap - random)
    return out


def _find_accumulator_sources(seed_dir: Path, repo_root: Path) -> tuple[Path | None, Path | None, str, list[str]]:
    seed_name = seed_dir.name
    result_candidates = [
        seed_dir / "data" / "metrics" / "panel_f_l3_accumulator_region_replay_metrics.csv",
        seed_dir / "data" / "metrics" / "panel_f_l3_accumulator_summary.csv",
        seed_dir / "pair_results.csv",
        seed_dir / "data" / "pair_results.csv",
        repo_root / "results" / "l3_accumulator_mechanism_experiment" / "pair_results.csv",
        repo_root / "results" / "l3_accumulator_mechanism_experiment" / "data" / "pair_results.csv",
        repo_root / "results" / "multi_network" / "l3_accumulator_mechanism_experiment" / "runs" / seed_name / "data" / "pair_results.csv",
        repo_root / "results" / "multi_network" / "l3_accumulator_mechanism_experiment" / "runs" / seed_name / "metrics" / "pair_results.csv",
    ]
    vector_candidates = [
        seed_dir / "data" / "raw" / "panel_f_l3_accumulator_pair_vectors.npz",
        seed_dir / "pair_vectors.npz",
        seed_dir / "data" / "pair_vectors.npz",
        repo_root / "results" / "l3_accumulator_mechanism_experiment" / "pair_vectors.npz",
        repo_root / "results" / "l3_accumulator_mechanism_experiment" / "data" / "pair_vectors.npz",
        repo_root / "results" / "multi_network" / "l3_accumulator_mechanism_experiment" / "runs" / seed_name / "data" / "pair_vectors.npz",
    ]
    checked = [_display(path, repo_root) for path in result_candidates + vector_candidates]
    result_path = next((path for path in result_candidates if path.exists()), None)
    vector_path = next((path for path in vector_candidates if path.exists()), None)
    if result_path and "panel_f_l3_accumulator" in result_path.name:
        level = "panel_f_accumulator"
    elif result_path:
        level = "bundle_pair_fallback"
    else:
        level = "missing_source"
    return result_path, vector_path, level, checked


def _append_decision_deflection_fallback(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    seed_dir: Path,
    source_path: Path,
    repo_root: Path,
    seeds: Sequence[Path],
) -> None:
    for _, r in df.iterrows():
        raw = str(r.get("condition", ""))
        rows.append(
            _canonical(
                figure_id,
                panel_id,
                metric="decision_deflection_fallback",
                condition=CONDITION_LABELS.get(raw, raw),
                layer="readout",
                seed_id=str(r.get("network_seed", _seed_id(seed_dir))),
                value=_num(r.get("decision_deflection_score", r.get("dynamic_like_recovery", np.nan))),
                unit="projection",
                source_file=_display(source_path, repo_root),
                raw_condition=raw,
                group="decision_deflection_fallback",
                x0=np.nan,
                y0=np.nan,
                x1=_num(r.get("static_to_dynamic_push", r.get("x1", np.nan))),
                y1=_num(r.get("dynamic_like_recovery", r.get("y1", np.nan))),
                fire_shift=_num(r.get("static_to_dynamic_push", np.nan)),
                decision_shift=_num(r.get("dynamic_like_recovery", np.nan)),
                pair_id=int(_num(r.get("pair_id", -1))),
                run_mode=_run_mode(seeds),
                source_level="decision_deflection_fallback",
            )
        )


def _build_accumulator_process_rows(df: pd.DataFrame, vectors: Any, warnings: list[str], source_label: str) -> tuple[pd.DataFrame, dict[str, float]]:
    pair_col = _first_existing_col(df, ["pair_id"])
    push_col = _first_existing_col(df, ["replacement_push_kstar", "mean_static_to_dynamic_push", "static_to_dynamic_push"])
    pull_col = _first_existing_col(df, ["replacement_pullback_kstar", "mean_dynamic_to_static_pullback", "dynamic_to_static_pullback"])
    if pair_col is None or push_col is None or pull_col is None:
        raise ValueError(f"{source_label} missing pair_id/replacement_push_kstar/replacement_pullback_kstar columns")
    vector_pair = np.asarray(vectors["pair_id"], dtype=np.int64)
    delta_v_key = "delta_V" if "delta_V" in vectors.files else "delta_v" if "delta_v" in vectors.files else None
    if delta_v_key is None or "Delta_hat_plus" not in vectors.files or "Delta_hat_minus" not in vectors.files:
        raise KeyError("pair vectors missing delta_V/delta_v, Delta_hat_plus, or Delta_hat_minus")
    order = {int(pair_id): idx for idx, pair_id in enumerate(vector_pair)}
    work = df[[pair_col, push_col, pull_col]].copy()
    work[pair_col] = pd.to_numeric(work[pair_col], errors="coerce")
    work = work.dropna(subset=[pair_col]).drop_duplicates(subset=[pair_col], keep="first")
    work[pair_col] = work[pair_col].astype(int)
    missing = sorted(set(work[pair_col].tolist()) - set(order))
    if missing:
        warnings.append(f"{source_label}: pair vectors missing {len(missing)} pair_id values; dropping unmatched rows.")
        work = work[work[pair_col].isin(order)]
    work = work.sort_values(pair_col, kind="stable").reset_index(drop=True)
    indices = np.asarray([order[int(pair_id)] for pair_id in work[pair_col]], dtype=np.int64)
    delta_v = np.asarray(vectors[delta_v_key], dtype=np.float64)[indices]
    plus_hat = np.asarray(vectors["Delta_hat_plus"], dtype=np.float64)[indices]
    minus_hat = np.asarray(vectors["Delta_hat_minus"], dtype=np.float64)[indices]
    plus_fire_raw = pd.to_numeric(work[push_col], errors="coerce").to_numpy(dtype=np.float64)
    minus_fire_raw = pd.to_numeric(work[pull_col], errors="coerce").to_numpy(dtype=np.float64)
    plus_decision_raw = _decision_projection(plus_hat, delta_v)
    minus_decision_raw = _decision_projection(minus_hat, delta_v)
    plus_fire_scale = _positive_robust_scale(plus_fire_raw)
    minus_fire_scale = _positive_robust_scale(minus_fire_raw)
    plus_decision_scale = _positive_robust_scale(plus_decision_raw)
    minus_decision_scale = _positive_robust_scale(minus_decision_raw)
    plus_fire = _normalized_shift(plus_fire_raw, plus_fire_scale)
    minus_fire = _normalized_shift(minus_fire_raw, minus_fire_scale)
    plus_decision = _normalized_shift(plus_decision_raw, plus_decision_scale)
    minus_decision = _normalized_shift(minus_decision_raw, minus_decision_scale)
    plus = pd.DataFrame(
        {
            "pair_id": work[pair_col].to_numpy(dtype=int),
            "group": "plus",
            "x0": -1.0,
            "y0": -1.0,
            "x1": -1.0 + 2.0 * plus_fire,
            "y1": -1.0 + 2.0 * plus_decision,
            "fire_shift": plus_fire,
            "decision_shift": plus_decision,
        }
    )
    minus = pd.DataFrame(
        {
            "pair_id": work[pair_col].to_numpy(dtype=int),
            "group": "minus",
            "x0": 1.0,
            "y0": 1.0,
            "x1": 1.0 - 2.0 * minus_fire,
            "y1": 1.0 - 2.0 * minus_decision,
            "fire_shift": minus_fire,
            "decision_shift": minus_decision,
        }
    )
    scales = {
        "plus_fire_scale": float(plus_fire_scale),
        "minus_fire_scale": float(minus_fire_scale),
        "plus_decision_scale": float(plus_decision_scale),
        "minus_decision_scale": float(minus_decision_scale),
    }
    return pd.concat([plus, minus], ignore_index=True), scales


def _decision_projection(delta_hat: np.ndarray, delta_v: np.ndarray) -> np.ndarray:
    denom = np.sum(delta_v * delta_v, axis=1)
    numer = np.sum(delta_hat * delta_v, axis=1)
    out = np.zeros_like(numer, dtype=np.float64)
    valid = np.isfinite(denom) & (denom > 1e-16)
    out[valid] = numer[valid] / denom[valid]
    return out


def _positive_robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return 1.0
    scale = float(np.nanpercentile(finite, 90))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _normalized_shift(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64) / max(float(scale), 1e-12), 0.0, 1.0)


def _accumulator_stats(panel_df: pd.DataFrame, scales: list[dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_pairs": 0}
    if panel_df.empty or "group" not in panel_df.columns:
        return out
    out["n_pairs"] = int(panel_df["pair_id"].replace("", pd.NA).dropna().nunique()) if "pair_id" in panel_df.columns else 0
    for group in ("plus", "minus"):
        part = panel_df[panel_df["group"].astype(str).eq(group)]
        if part.empty:
            continue
        start = [float(pd.to_numeric(part["x0"], errors="coerce").mean()), float(pd.to_numeric(part["y0"], errors="coerce").mean())]
        end = [float(pd.to_numeric(part["x1"], errors="coerce").mean()), float(pd.to_numeric(part["y1"], errors="coerce").mean())]
        out[f"mean_{group}_start"] = start
        out[f"mean_{group}_end"] = end
        out[f"{group}_mean_arrow_length"] = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    for key in ("plus_fire_scale", "minus_fire_scale", "plus_decision_scale", "minus_decision_scale"):
        values = [float(item[key]) for item in scales if key in item and np.isfinite(item[key])]
        out[key] = float(np.mean(values)) if values else np.nan
    return out


def _resolve_experiment_root(spec: Mapping[str, Any], repo_root: Path) -> tuple[Path, list[Path], list[str]]:
    raw = spec.get("experiment_root") or spec.get("experiment_root_default") or DEFAULT_EXPERIMENT_ROOT
    root = Path(raw)
    if not root.is_absolute():
        root = repo_root / root
    warnings: list[str] = []
    if root.name.startswith("seed_"):
        seeds = [root] if root.exists() else []
        parent = root.parent
    else:
        parent = root
        seeds = sorted(path for path in root.glob("seed_*") if path.is_dir()) if root.exists() else []
        if not seeds and (root / "data" / "metrics").exists():
            seeds = [root]
    if _run_mode(seeds) == "single_network_draft" and seeds:
        warnings.append(DRAFT_WARNING)
    return parent, seeds, warnings


def _write(
    spec: Mapping[str, Any],
    output_dir: Path,
    figure_id: str,
    panel_id: str,
    panel_df: pd.DataFrame,
    stats: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> AdapterResult:
    warnings = list(manifest.get("warnings") or [])
    for col in ("run_mode", "n_networks"):
        panel_df[col] = stats.get(col)
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def _stats(figure_id: str, panel_id: str, panel_df: pd.DataFrame, run_mode: str, seeds: Sequence[Path], metric: str, warnings: list[str], group_cols: list[str]) -> dict[str, Any]:
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "run_mode": run_mode,
        "n_networks": len(seeds),
        "network_ids": [_seed_id(seed) for seed in seeds],
        "warnings": list(warnings),
        "summaries": summarize_values(panel_df, [c for c in group_cols if c in panel_df.columns]),
        "values_used_for_plotting": _values(panel_df),
        "aggregation": "within_network_then_across_networks" if run_mode == "multi_network_final" else "single_network_pair_distribution",
    }
    if run_mode == "single_network_draft" and DRAFT_WARNING not in stats["warnings"]:
        stats["warnings"].append(DRAFT_WARNING)
    return stats


def _manifest(figure_id: str, panel_id: str, root: Path, seeds: Sequence[Path], sources: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    existing_paths = [src["path"] for src in sources if src.get("exists")]
    return {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok" if existing_paths else "missing_source",
        "experiment_root": str(root),
        "run_mode": _run_mode(seeds),
        "n_networks": len(seeds),
        "network_ids": [_seed_id(seed) for seed in seeds],
        "required_raw_conditions": list(RAW_CONDITIONS),
        "raw_conditions": list(RAW_CONDITIONS),
        "source_files_used": existing_paths,
        "sources": sources,
        "warnings": list(warnings),
    }


def _source(path: Path, seed_dir: Path) -> dict[str, Any]:
    return {"path": str(path).replace("\\", "/"), "seed_id": _seed_id(seed_dir), "exists": path.exists()}


def _canonical(
    figure_id: str,
    panel_id: str,
    *,
    metric: str,
    condition: str,
    layer: str,
    seed_id: str,
    value: Any,
    unit: str,
    source_file: str,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "condition": condition,
        "layer": layer,
        "network_id": seed_id,
        "seed_id": seed_id,
        "value": value,
        "unit": unit,
        "source_file": source_file,
    }
    row.update(extra)
    return row


def _run_mode(seeds: Sequence[Path]) -> str:
    return "single_network_draft" if len(seeds) <= 1 else "multi_network_final"


def _seed_id(seed_dir: Path) -> str:
    return seed_dir.name.replace("seed_", "") if seed_dir.name.startswith("seed_") else seed_dir.name


def _display(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig4")), str(spec.get("panel_id", "")).upper()


def _num(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float("nan") if pd.isna(numeric) else float(numeric)


def _maybe_int(value: Any) -> int | Any:
    numeric = _num(value)
    if np.isfinite(numeric):
        return int(numeric)
    return value


def _bin_order(label: Any) -> int:
    text = str(label)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0


def _values(df: pd.DataFrame) -> list[float]:
    if "value" not in df.columns:
        return []
    return [float(v) for v in pd.to_numeric(df["value"], errors="coerce").dropna().tolist()]
