from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult, missing_adapter_result, panel_output_paths, summarize_values, write_adapter_outputs
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

FROZEN_S4_NETWORK_IDS = tuple(str(seed) for seed in range(1000, 1020))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_s4_path(repo_root: Path, frozen: Mapping[str, Any], key: str) -> Path:
    raw = frozen.get(key)
    if not raw:
        raise RuntimeError(f"S4 persisted-input contract lacks {key!r}.")
    path = Path(str(raw))
    path = path if path.is_absolute() else repo_root / path
    if not path.is_file():
        raise RuntimeError(f"S4 persisted input is missing: {path}")
    expected = str(frozen.get(f"{key}_sha256", "")).lower()
    actual = _sha256(path)
    if not expected or actual != expected:
        raise RuntimeError(f"S4 persisted input hash mismatch for {path}: expected {expected or '<unset>'}, got {actual}.")
    return path


def _copy_frozen_s4_adapter_outputs(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Copy one already-persisted S4 plotting payload after strict identity checks.

    This is intentionally a leaf adapter: it reads neither experiment/runtime
    state nor per-seed metric tables, performs no aggregation, and has no
    fallback path.  Any missing or changed persisted payload is a hard error.
    """
    figure_id, panel_id = _ids(spec)
    frozen = spec.get("persisted_input")
    if not isinstance(frozen, Mapping):
        raise RuntimeError(f"{figure_id}{panel_id} lacks a persisted_input mapping.")

    panel_path = _frozen_s4_path(repo_root, frozen, "panel_data")
    stats_path = _frozen_s4_path(repo_root, frozen, "stats")
    manifest_path = _frozen_s4_path(repo_root, frozen, "source_manifest")

    panel_df = pd.read_csv(panel_path)
    expected_rows = int(frozen.get("row_count", -1))
    if len(panel_df) != expected_rows:
        raise RuntimeError(f"{figure_id}{panel_id} row-count mismatch: expected {expected_rows}, got {len(panel_df)}.")
    required = {"figure_id", "panel_id", "network_id", "seed_id", "value", "run_mode", "n_networks"}
    missing = sorted(required.difference(panel_df.columns))
    if missing:
        raise RuntimeError(f"{figure_id}{panel_id} persisted panel data lacks columns {missing}.")
    if set(panel_df["figure_id"].astype(str)) != {figure_id} or set(panel_df["panel_id"].astype(str)) != {panel_id}:
        raise RuntimeError(f"{figure_id}{panel_id} row identity contains a foreign figure or panel id.")
    network_ids = tuple(sorted(set(panel_df["network_id"].astype(str)), key=int))
    seed_ids = tuple(sorted(set(panel_df["seed_id"].astype(str)), key=int))
    if network_ids != FROZEN_S4_NETWORK_IDS or seed_ids != FROZEN_S4_NETWORK_IDS:
        raise RuntimeError(f"{figure_id}{panel_id} network/seed identity differs from the frozen 1000-1019 set.")
    if set(pd.to_numeric(panel_df["n_networks"], errors="raise").astype(int)) != {20}:
        raise RuntimeError(f"{figure_id}{panel_id} does not declare exactly 20 networks on every row.")
    if set(panel_df["run_mode"].astype(str)) != {"multi_network_final"}:
        raise RuntimeError(f"{figure_id}{panel_id} is not a multi_network_final persisted payload.")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_source_count = int(frozen.get("source_count", -1))
    manifest_network_ids = tuple(str(value) for value in manifest.get("network_ids", []))
    if (
        manifest.get("figure_id") != figure_id
        or str(manifest.get("panel_id")) != panel_id
        or manifest.get("status") != "ok"
        or manifest.get("fallback_used") is not False
        or manifest.get("run_mode") != "multi_network_final"
        or int(manifest.get("n_networks", -1)) != 20
        or manifest_network_ids != FROZEN_S4_NETWORK_IDS
        or len(manifest.get("sources", [])) != expected_source_count
        or len(manifest.get("source_files_used", [])) != expected_source_count
        or manifest.get("warnings") not in ([], None)
    ):
        raise RuntimeError(f"{figure_id}{panel_id} source-manifest identity validation failed.")
    if any(source.get("exists") is not True for source in manifest.get("sources", [])):
        raise RuntimeError(f"{figure_id}{panel_id} source manifest contains a missing source.")

    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    stats_network_ids = tuple(str(value) for value in stats.get("network_ids", []))
    if (
        stats.get("figure_id") != figure_id
        or str(stats.get("panel_id")) != panel_id
        or int(stats.get("n_networks", -1)) != 20
        or stats_network_ids != FROZEN_S4_NETWORK_IDS
        or stats.get("run_mode") != "multi_network_final"
        or stats.get("warnings") not in ([], None)
    ):
        raise RuntimeError(f"{figure_id}{panel_id} persisted stats identity validation failed.")
    panel_values = pd.to_numeric(panel_df["value"], errors="coerce").dropna().to_numpy(dtype=float)
    stats_values = np.asarray(stats.get("values_used_for_plotting", []), dtype=float)
    if panel_values.shape != stats_values.shape or not np.array_equal(panel_values, stats_values, equal_nan=True):
        raise RuntimeError(f"{figure_id}{panel_id} persisted stats values are not row-identical to panel data.")

    destinations = panel_output_paths(output_dir, figure_id, panel_id)
    for destination in destinations.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (panel_path, destinations["panel_data"]),
        (stats_path, destinations["stats"]),
        (manifest_path, destinations["sources"]),
    ):
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    return AdapterResult(
        panel_data_path=destinations["panel_data"],
        stats_manifest_path=destinations["stats"],
        source_manifest_path=destinations["sources"],
        source_manifest=manifest,
        warnings=[],
    )


def build_s7_similarity_overlap_2x2_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _copy_frozen_s4_adapter_outputs(spec, repo_root, output_dir)


def build_s7_overlap_excess_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _copy_frozen_s4_adapter_outputs(spec, repo_root, output_dir)


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
            warnings.append(f"Missing S5A similarity trend sources under {_display(seed_dir, repo_root)}.")
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
    return _copy_frozen_s4_adapter_outputs(spec, repo_root, output_dir)


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
    return _copy_frozen_s4_adapter_outputs(spec, repo_root, output_dir)


def build_s7_random_nonoverlap_perturbation_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_perturbation_subset(spec, repo_root, output_dir, PERTURBATION_KEEP)


def build_s7_alternative_overlap_definitions_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "supp_alternative_overlap_definitions.csv"
        sources.append(_source(path, seed_dir))
        if not path.exists():
            warnings.append(f"{_display(path, repo_root)} missing.")
            continue
        df = pd.read_csv(path)
        required = {"overlap_definition", "overlap_value", "dynamic_effect_metric", "metric_value"}
        missing = sorted(required.difference(df.columns))
        if missing:
            warnings.append(f"{_display(path, repo_root)} lacks columns {missing}.")
            continue
        for (definition, effect_metric), part in df.groupby(["overlap_definition", "dynamic_effect_metric"], dropna=False, sort=False):
            x = pd.to_numeric(part["overlap_value"], errors="coerce")
            y = pd.to_numeric(part["metric_value"], errors="coerce")
            valid = x.notna() & y.notna()
            corr = float(x[valid].corr(y[valid])) if int(valid.sum()) > 1 else np.nan
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "overlap_dpi_correlation" if str(effect_metric) == "DPI_L3" else f"overlap_{effect_metric}_correlation",
                    str(definition),
                    corr,
                    "pearson_r",
                    seed_dir,
                    path,
                    repo_root,
                    dynamic_effect_metric=str(effect_metric),
                    n_pairs=int(valid.sum()),
                )
            )
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, "overlap_dpi_correlation", ["metric", "condition"])


def build_s7_perturbation_specificity_contrast_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _roots(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    contrast_specs = (
        ("overlap_minus_nonoverlap_DPI", "Overlap - non-overlap", "DPI_L3_contrast"),
        ("overlap_minus_random_DPI", "Overlap - random", "DPI_L3_contrast"),
        ("overlap_minus_nonoverlap_recovery", "Recovery - non-overlap", "recovery_contrast"),
        ("overlap_minus_random_recovery", "Recovery - random", "recovery_contrast"),
    )
    for seed_dir in seeds:
        contrast_path = seed_dir / "data" / "metrics" / "panel_d_overlap_perturbation_contrast.csv"
        random_path = seed_dir / "data" / "metrics" / "supp_random_mask_perturbation_controls.csv"
        audit_path = seed_dir / "data" / "metrics" / "panel_d_l1_stsp_overlap_perturbation_audit.csv"
        sources.extend([_source(contrast_path, seed_dir), _source(random_path, seed_dir), _source(audit_path, seed_dir)])
        audit: dict[str, Any] = {}
        if audit_path.exists():
            audit_df = pd.read_csv(audit_path)
            if not audit_df.empty:
                audit = audit_df.iloc[0].to_dict()
        if not contrast_path.exists():
            warnings.append(f"{_display(contrast_path, repo_root)} missing.")
            continue
        contrast = pd.read_csv(contrast_path)
        for _, r in contrast.iterrows():
            for source_col, label, metric in contrast_specs:
                if source_col not in contrast.columns:
                    continue
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        metric,
                        label,
                        _num(r.get(source_col)),
                        "index_delta",
                        seed_dir,
                        contrast_path,
                        repo_root,
                        source_metric=source_col,
                        n_pairs=r.get("n_pairs", ""),
                        probe_input_unchanged=bool(audit.get("probe_input_unchanged", False)),
                        sample_input_complete=bool(audit.get("sample_input_complete", False)),
                        perturbed_layer=audit.get("perturbed_layer", ""),
                        perturbed_variables=audit.get("perturbed_variables", ""),
                        l2_stsp_frozen=bool(audit.get("l2_stsp_frozen", False)),
                        l3_stsp_frozen=bool(audit.get("l3_stsp_frozen", False)),
                    )
                )
    return _finish(spec, output_dir, root, seeds, rows, sources, warnings, "DPI_L3_contrast", ["metric", "condition"])


def build_s7_decision_spike_summary_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return build_s8_decision_spike_summary_adapter(spec, repo_root, output_dir)


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
        warnings.append(f"{_display(path, repo_root)} lacks static probe_accuracy baseline; S5E uses diagnostic fallback {metric_col}.")
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
        "fallback_used": any("fallback" in str(src.get("path", "")).lower() for src in sources if src.get("exists")),
    }
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, list(stats["warnings"]))


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig4_supp")), str(spec.get("panel_id", "")).upper()


def _bin_order(label: Any) -> int:
    text = str(label)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0


def _normalise_regression_term(term: Any) -> str:
    text = str(term).strip()
    text = text.replace("beta_", "")
    if text in {"dice_overlap", "overlap_value"} or "overlap" in text:
        return "overlap"
    if "similarity" in text:
        return "similarity"
    if text in {"input_energy", "energy"} or "input_energy" in text:
        return "input_energy"
    if "sample_energy" in text:
        return "sample_energy"
    if "probe_energy" in text:
        return "probe_energy"
    return text


def _values(df: pd.DataFrame) -> list[float]:
    if "value" not in df.columns:
        return []
    return [float(v) for v in pd.to_numeric(df["value"], errors="coerce").dropna().tolist()]
