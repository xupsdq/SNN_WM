from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult, missing_adapter_result, summarize_values, write_adapter_outputs
from src.plotting.paper_fig.adapters.fig4_adapters import (
    CONDITION_LABELS,
    DEFAULT_EXPERIMENT_ROOT,
    DRAFT_WARNING,
    _accumulator_stats,
    _build_accumulator_process_rows,
    _display,
    _find_accumulator_sources,
    _first_existing_col,
    _num,
    _resolve_experiment_root,
    _run_mode,
    _seed_id,
    _source,
)


PERTURBATION_KEEP = (
    "sample_keep_overlap_only_dynamic",
    "sample_keep_nonoverlap_only_dynamic",
    "sample_random_matched_dynamic",
)


def build_s7_similarity_full_trend_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        pair_path = seed_dir / "data" / "metrics" / "panel_b_similarity_entry_metrics.csv"
        bin_path = seed_dir / "data" / "metrics" / "panel_b_similarity_bin_summary.csv"
        sources.extend([_source(pair_path, seed_dir), _source(bin_path, seed_dir)])
        path = pair_path if pair_path.exists() else bin_path if bin_path.exists() else None
        if path is None:
            warnings.append(f"Missing S7A similarity trend sources under {_display(seed_dir, repo_root)}.")
            continue
        df = pd.read_csv(path)
        bin_col = _first_existing_col(df, ["similarity_bin", "bin", "iso_similarity_bin"])
        value_col = _first_existing_col(df, ["mean_acc_drop", "acc_drop", "mean_drop_event", "drop_event", "drop_event_rate"])
        if bin_col is None or value_col is None:
            warnings.append(f"{_display(path, repo_root)} lacks similarity bin/accuracy-drop columns.")
            continue
        for _, r in df.iterrows():
            rows.append(_row(figure_id, panel_id, "accuracy_drop", str(r.get(bin_col, "")), _num(r.get(value_col)), "probability_delta", seed_dir, path, repo_root, y_value=_num(r.get(value_col)), metric_source=str(value_col), similarity_bin=str(r.get(bin_col, "")), similarity_bin_order=_bin_order(r.get(bin_col, "")), pair_id=r.get("pair_id", "")))
    result = _finish(spec, output_dir, root, seeds, rows, sources, warnings, "accuracy_drop", ["similarity_bin"])
    return result


def build_s7_matching_diagnostics_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    metrics = ("similarity_difference", "sample_energy_rel_difference", "probe_energy_rel_difference", "dice_overlap_difference", "overlap_difference")
    summary_metrics = ("mean_similarity_difference", "mean_sample_energy_rel_difference", "mean_probe_energy_rel_difference", "mean_overlap_difference")
    for seed_dir in seeds:
        candidates = [
            seed_dir / "data" / "metrics" / "panel_d_matching_balance_diagnostics.csv",
            seed_dir / "data" / "metrics" / "panel_d_iso_similarity_matched_pairs.csv",
            seed_dir / "data" / "metrics" / "panel_c_overlap_matched_comparison.csv",
        ]
        sources.extend(_source(path, seed_dir) for path in candidates)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            warnings.append(f"Missing S7B matching diagnostic sources under {_display(seed_dir, repo_root)}.")
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            for metric in metrics + summary_metrics:
                if metric not in df.columns:
                    continue
                rows.append(_row(figure_id, panel_id, metric, metric.replace("mean_", ""), _num(r.get(metric)), "difference", seed_dir, path, repo_root, match_id=r.get("match_id", ""), n_matched_sets=r.get("n_matched_sets", "")))
    result = _finish(spec, output_dir, root, seeds, rows, sources, warnings, "matching_diagnostics", ["metric"])
    return result


def build_s7_iso_similarity_matching_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        matches_path = seed_dir / "data" / "metrics" / "panel_d_iso_similarity_matched_pairs.csv"
        contrast_path = seed_dir / "data" / "metrics" / "panel_d_overlap_accuracy_contrast_by_network.csv"
        balance_path = seed_dir / "data" / "metrics" / "panel_d_matching_balance_diagnostics.csv"
        sources.extend([_source(matches_path, seed_dir), _source(contrast_path, seed_dir), _source(balance_path, seed_dir)])
        if matches_path.exists():
            matches = pd.read_csv(matches_path)
            for _, r in matches.iterrows():
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        "paired_delta_drop_event",
                        "matched_pair",
                        _num(r.get("paired_delta_drop_event")),
                        "probability_delta",
                        seed_dir,
                        matches_path,
                        repo_root,
                        match_id=r.get("match_id", ""),
                        high_pair_id=r.get("high_pair_id", ""),
                        low_pair_id=r.get("low_pair_id", ""),
                        similarity_difference=_num(r.get("similarity_difference")),
                        sample_energy_rel_difference=_num(r.get("sample_energy_rel_difference")),
                        probe_energy_rel_difference=_num(r.get("probe_energy_rel_difference")),
                        dice_overlap_high=_num(r.get("dice_overlap_high")),
                        dice_overlap_low=_num(r.get("dice_overlap_low")),
                        drop_event_high=_num(r.get("drop_event_high")),
                        drop_event_low=_num(r.get("drop_event_low")),
                        paired_delta_drop_event=_num(r.get("paired_delta_drop_event")),
                    )
                )
        else:
            warnings.append(f"{_display(matches_path, repo_root)} missing.")
        if contrast_path.exists():
            contrast = pd.read_csv(contrast_path)
            for _, r in contrast.iterrows():
                for metric in ("delta_drop_rate", "drop_rate_high_overlap", "drop_rate_low_overlap", "permutation_p_one_sided", "permutation_p_two_sided"):
                    if metric in contrast.columns:
                        rows.append(_row(figure_id, panel_id, metric, metric, _num(r.get(metric)), "probability", seed_dir, contrast_path, repo_root, n_matched_sets=r.get("n_matched_sets", "")))
        if balance_path.exists():
            balance = pd.read_csv(balance_path)
            for _, r in balance.iterrows():
                for metric in ("mean_similarity_difference", "mean_sample_energy_rel_difference", "mean_probe_energy_rel_difference", "mean_overlap_difference"):
                    if metric in balance.columns:
                        rows.append(_row(figure_id, panel_id, metric, "matching_balance", _num(r.get(metric)), "difference", seed_dir, balance_path, repo_root, n_matched_sets=r.get("n_matched_sets", "")))
    result = _finish(spec, output_dir, root, seeds, rows, sources, warnings, "paired_delta_drop_event", ["metric"])
    if result.stats_manifest_path.exists():
        # _finish already wrote the useful row summaries; QC reads the CSV/manifest.
        pass
    return result


def build_s7_overlap_regression_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    wanted = ("overlap", "similarity", "sample_energy", "probe_energy")
    for seed_dir in seeds:
        candidates = [seed_dir / "data" / "metrics" / "supp_overlap_similarity_regression.csv", seed_dir / "data" / "metrics" / "panel_c_overlap_localization_metrics.csv"]
        sources.extend(_source(path, seed_dir) for path in candidates)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            warnings.append(f"Missing S7D regression sources under {_display(seed_dir, repo_root)}.")
            continue
        df = pd.read_csv(path)
        if "term" in df.columns:
            for _, r in df.iterrows():
                term = str(r.get("term", ""))
                if not any(key in term.lower() for key in wanted):
                    continue
                value_col = _first_existing_col(df, ["estimate", "coef", "coefficient", "beta", "value"])
                rows.append(_row(figure_id, panel_id, "regression_coefficient", term, _num(r.get(value_col)) if value_col else np.nan, "coefficient", seed_dir, path, repo_root, se=_num(r.get("se", r.get("stderr", np.nan))), p_value=_num(r.get("p_value", r.get("p", np.nan)))))
        else:
            for term in ("beta_overlap", "beta_similarity", "beta_sample_energy", "beta_probe_energy"):
                if term in df.columns:
                    for _, r in df.iterrows():
                        rows.append(_row(figure_id, panel_id, "regression_coefficient", term.replace("beta_", ""), _num(r.get(term)), "coefficient", seed_dir, path, repo_root))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, "regression_coefficient", ["condition"])


def build_s7_random_nonoverlap_perturbation_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_perturbation_subset(spec, repo_root, output_dir, PERTURBATION_KEEP)


def build_s8_time_resolved_l3_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_e_time_resolved_l3_displacement.csv"
        sources.append(_source(path, seed_dir))
        if not path.exists():
            warnings.append(f"{_display(path, repo_root)} missing.")
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            raw = str(r.get("condition", ""))
            rows.append(_row(figure_id, panel_id, "DPI_L3_t", CONDITION_LABELS.get(raw, raw), _num(r.get("DPI_L3_t")), "index", seed_dir, path, repo_root, raw_condition=raw, time_ms=_num(r.get("time_ms")), time_step=r.get("time_step", ""), pair_id=r.get("pair_id", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, "DPI_L3_t", ["condition", "time_ms"])


def build_s8_decision_spike_summary_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_e_decision_spike_displacement.csv"
        sources.append(_source(path, seed_dir))
        if not path.exists():
            warnings.append(f"{_display(path, repo_root)} missing.")
            continue
        df = pd.read_csv(path)
        metric_col = _first_existing_col(df, ["mean_DPI_L3", "decision_spike_advance", "readout_step_displacement", "DPI_L3"])
        if metric_col is None:
            warnings.append(f"{_display(path, repo_root)} lacks decision-step metric columns.")
            continue
        for _, r in df.iterrows():
            raw = str(r.get("condition", ""))
            rows.append(_row(figure_id, panel_id, str(metric_col), CONDITION_LABELS.get(raw, raw), _num(r.get(metric_col)), "index", seed_dir, path, repo_root, raw_condition=raw, pair_id=r.get("pair_id", ""), first_fire_time_dynamic=r.get("first_fire_time_dynamic", ""), first_fire_time_static=r.get("first_fire_time_static", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, "decision_spike_summary", ["condition"])


def build_s8_l3_accumulator_replay_detail_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        metrics = seed_dir / "data" / "metrics" / "panel_f_l3_accumulator_region_replay_metrics.csv"
        summary = seed_dir / "data" / "metrics" / "panel_f_l3_accumulator_summary.csv"
        region_npz = seed_dir / "data" / "raw" / "panel_f_l3_accumulator_region_effects.npz"
        vectors = seed_dir / "data" / "raw" / "panel_f_l3_accumulator_pair_vectors.npz"
        sources.extend([_source(metrics, seed_dir), _source(summary, seed_dir), _source(region_npz, seed_dir), _source(vectors, seed_dir)])
        path = metrics if metrics.exists() else summary if summary.exists() else None
        if path is None:
            warnings.append(f"Missing S8C accumulator replay detail sources under {_display(seed_dir, repo_root)}.")
            continue
        df = pd.read_csv(path)
        for metric in ("replacement_push_kstar", "replacement_pullback_kstar", "deletion_dynamic_minus_static_kstar", "reconstruction_cosine_plus", "reconstruction_cosine_minus", "mean_static_to_dynamic_push", "mean_dynamic_to_static_pullback"):
            if metric not in df.columns:
                continue
            for _, r in df.iterrows():
                rows.append(_row(figure_id, panel_id, metric, metric, _num(r.get(metric)), "effect", seed_dir, path, repo_root, pair_id=r.get("pair_id", ""), summary_group=r.get("summary_group", ""), bias_direction=r.get("bias_direction", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, "l3_accumulator_replay_detail", ["metric"])


def build_s8_decision_deflection_summary_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        metrics = seed_dir / "data" / "metrics" / "panel_f_decision_deflection_metrics.csv"
        summary = seed_dir / "data" / "metrics" / "panel_f_l3_accumulator_summary.csv"
        sources.extend([_source(metrics, seed_dir), _source(summary, seed_dir)])
        path = metrics if metrics.exists() else summary if summary.exists() else None
        if path is None:
            warnings.append(f"Missing S8D decision-deflection sources under {_display(seed_dir, repo_root)}.")
            continue
        df = pd.read_csv(path)
        for metric in ("static_to_dynamic_push", "dynamic_like_recovery", "decision_deflection_score", "mean_static_to_dynamic_push", "mean_dynamic_to_static_pullback"):
            if metric not in df.columns:
                continue
            for _, r in df.iterrows():
                raw = str(r.get("condition", r.get("summary_group", metric)))
                rows.append(_row(figure_id, panel_id, metric, CONDITION_LABELS.get(raw, raw), _num(r.get(metric)), "projection", seed_dir, path, repo_root, raw_condition=raw, pair_id=r.get("pair_id", "")))
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, "decision_deflection_summary", ["metric", "condition"])


def _build_perturbation_subset(spec: Mapping[str, Any], repo_root: Path, output_dir: Path, keep: Sequence[str]) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        candidates = [
            seed_dir / "data" / "metrics" / "supp_overlap_preserving_perturbation_metrics.csv",
            seed_dir / "data" / "metrics" / "supp_random_mask_perturbation_controls.csv",
            seed_dir / "data" / "metrics" / "panel_d_overlap_perturbation_metrics.csv",
        ]
        sources.extend(_source(path, seed_dir) for path in candidates)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            warnings.append(f"Missing perturbation subset source under {_display(seed_dir, repo_root)}.")
            continue
        df = pd.read_csv(path)
        if "condition" not in df.columns:
            warnings.append(f"{_display(path, repo_root)} lacks condition/value columns.")
            continue
        if "probe_accuracy" in df.columns and (df["condition"].astype(str) == "full_static").any():
            static_acc = float(pd.to_numeric(df.loc[df["condition"].astype(str).eq("full_static"), "probe_accuracy"], errors="coerce").dropna().mean())
            metric_col = "accuracy_drop_vs_static"
            work = df[df["condition"].astype(str).isin(keep)].copy()
            for _, r in work.iterrows():
                raw = str(r.get("condition", ""))
                value = static_acc - _num(r.get("probe_accuracy"))
                rows.append(_row(figure_id, panel_id, metric_col, CONDITION_LABELS.get(raw, raw), value, "probability_delta", seed_dir, path, repo_root, y_value=value, raw_condition=raw, pair_id=r.get("pair_id", ""), probe_accuracy_condition=_num(r.get("probe_accuracy")), probe_accuracy_static=static_acc))
            continue
        metric_col = _first_existing_col(df, ["dynamic_like_recovery", "DPI_L3", "decision_deflection_score"])
        if metric_col is None:
            warnings.append(f"{_display(path, repo_root)} lacks probe_accuracy or diagnostic fallback metric columns.")
            continue
        warnings.append(f"{_display(path, repo_root)} lacks static probe_accuracy baseline; S7E uses diagnostic fallback {metric_col}.")
        df = df[df["condition"].astype(str).isin(keep)]
        for _, r in df.iterrows():
            raw = str(r.get("condition", ""))
            rows.append(_row(figure_id, panel_id, str(metric_col), CONDITION_LABELS.get(raw, raw), _num(r.get(metric_col)), "index", seed_dir, path, repo_root, raw_condition=raw, pair_id=r.get("pair_id", "")))
    metric = "accuracy_drop_vs_static" if any(row.get("metric") == "accuracy_drop_vs_static" for row in rows) else "dynamic_like_recovery"
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, metric, ["condition"])


def _roots(spec: Mapping[str, Any], repo_root: Path) -> tuple[Path, list[Path], list[str]]:
    local_spec = dict(spec)
    local_spec.setdefault("experiment_root", DEFAULT_EXPERIMENT_ROOT)
    root, seeds, warnings = _resolve_experiment_root(local_spec, repo_root)
    if not seeds:
        warnings.append(f"No Fig.4 supplement seed directories found under {root}.")
    return root, seeds, warnings


def _row(
    figure_id: str,
    panel_id: str,
    metric: str,
    condition: str,
    value: Any,
    unit: str,
    seed_dir: Path,
    source_path: Path,
    repo_root: Path,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "condition": condition,
        "layer": extra.pop("layer", ""),
        "network_id": str(extra.pop("network_seed", _seed_id(seed_dir))),
        "seed_id": str(extra.pop("seed_id", _seed_id(seed_dir))),
        "value": value,
        "unit": unit,
        "source_file": _display(source_path, repo_root),
        "run_mode": extra.pop("run_mode", ""),
    }
    row.update(extra)
    return row


def _finish(
    spec: Mapping[str, Any],
    output_dir: Path,
    root: Path,
    seeds: Sequence[Path],
    rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    warnings: list[str],
    metric: str,
    group_cols: list[str],
) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    if not seeds:
        return missing_adapter_result(spec, root, output_dir, warnings[-1] if warnings else "No Fig.4 supplement seeds found.")
    if not rows:
        result = missing_adapter_result(spec, root, output_dir, warnings[-1] if warnings else f"Missing source data for {figure_id}{panel_id}.")
        result.source_manifest["sources"] = sources
        return result
    panel_df = pd.DataFrame(rows)
    run_mode = _run_mode(seeds)
    panel_df["run_mode"] = run_mode
    panel_df["n_networks"] = len(seeds)
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "run_mode": run_mode,
        "n_networks": len(seeds),
        "network_ids": [_seed_id(seed) for seed in seeds],
        "warnings": list(warnings),
        "summaries": summarize_values(panel_df, [col for col in group_cols if col in panel_df.columns]),
        "values_used_for_plotting": _values(panel_df),
    }
    if run_mode == "single_network_draft" and DRAFT_WARNING not in stats["warnings"]:
        stats["warnings"].append(DRAFT_WARNING)
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok",
        "experiment_root": str(root),
        "run_mode": run_mode,
        "n_networks": len(seeds),
        "network_ids": [_seed_id(seed) for seed in seeds],
        "source_files_used": [src["path"] for src in sources if src.get("exists")],
        "sources": sources,
        "warnings": list(stats["warnings"]),
        "fallback_used": any("fallback" in str(src.get("path", "")).lower() or "supp_" in str(src.get("path", "")).lower() for src in sources if src.get("exists")),
    }
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, list(stats["warnings"]))


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig4_supp")), str(spec.get("panel_id", "")).upper()


def _bin_order(label: Any) -> int:
    text = str(label)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0


def _values(df: pd.DataFrame) -> list[float]:
    if "value" not in df.columns:
        return []
    return [float(v) for v in pd.to_numeric(df["value"], errors="coerce").dropna().tolist()]
