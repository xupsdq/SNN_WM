from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult, missing_adapter_result, summarize_values, write_adapter_outputs


DEFAULT_EXPERIMENT_ROOT = Path("results") / "paper_experiments" / "fig1_functional_stsp_substrate"
SINGLE_NETWORK_WARNING = "Single-network result. Use for pipeline validation only, not final manuscript statistics."
MAIN_CONDITIONS = ("dynamic_intact", "ux_trial_shuffle", "static_frozen")
LAYER_ORDER = ("layer1", "layer2", "layer3")


def build_fig1_baseline_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 experiment seed directory not found.")
    rows = []
    sources = []
    warnings = _run_mode_warnings(seeds)
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_b_baseline_metrics_by_network.csv"
        if not path.exists():
            warnings.append(f"Missing baseline metrics: {_display_path(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        for _, row in df.iterrows():
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric="overall_recall",
                    condition="STSP-SNN",
                    value=float(row.get("overall_recall", 0.0)) * 100.0,
                    unit="percent",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    n_trials=row.get("n_trials", ""),
                    run_mode=_run_mode(seeds),
                )
            )
    panel_df = pd.DataFrame(rows)
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="overall_recall", run_mode=_run_mode(seeds), group_cols=["condition"])
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_fig1_delay_decode_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 experiment seed directory not found.")
    rows = []
    sources = []
    warnings = _run_mode_warnings(seeds)
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_c_delay_decode_metrics.csv"
        if not path.exists():
            warnings.append(f"Missing delay decode metrics: {_display_path(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        for _, row in df.iterrows():
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric="delay_decode_accuracy",
                    condition=str(row.get("layer", "")),
                    layer=str(row.get("layer", "")),
                    value=float(row.get("acc", 0.0)) * 100.0,
                    unit="percent",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    delay_ms=int(row.get("delay_ms", 0)),
                    feature_type=row.get("feature_type", "ux_concat"),
                    classifier=row.get("classifier", ""),
                    macro_f1=row.get("macro_f1", ""),
                    chance=float(row.get("chance", 0.1)) * 100.0,
                    n_train=row.get("n_train", ""),
                    n_test=row.get("n_test", ""),
                    run_mode=_run_mode(seeds),
                )
            )
    panel_df = pd.DataFrame(rows)
    if not panel_df.empty:
        panel_df["_layer_order"] = panel_df["layer"].map({name: i for i, name in enumerate(LAYER_ORDER)}).fillna(99)
        panel_df = panel_df.sort_values(["_layer_order", "delay_ms", "seed_id"], kind="stable").drop(columns=["_layer_order"]).reset_index(drop=True)
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="delay_decode_accuracy", run_mode=_run_mode(seeds), group_cols=["layer", "delay_ms"])
    stats["delay_ms_values"] = _unique(panel_df, "delay_ms")
    stats["layers"] = _unique(panel_df, "layer")
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["delay_ms_values"] = stats["delay_ms_values"]
    manifest["layers"] = stats["layers"]
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_fig1_delay_decode_summary_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    available_delays: list[float] = []
    selected_delays: list[float] = []
    requested_delay = spec.get("primary_delay_ms")
    requested_delay_num = _to_number(requested_delay)
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_c_delay_decode_metrics.csv"
        if not path.exists():
            warnings.append(f"Missing delay decode metrics: {_display_path(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        if df.empty:
            warnings.append(f"Empty delay decode metrics: {_display_path(path, repo_root)}")
            continue
        if "delay_ms" in df.columns:
            df = df.copy()
            df["_delay_numeric"] = pd.to_numeric(df["delay_ms"], errors="coerce")
            available_delays.extend(float(v) for v in df["_delay_numeric"].dropna().unique().tolist())
            if requested_delay_num is not None:
                selected = df[df["_delay_numeric"].eq(float(requested_delay_num))].copy()
                if selected.empty:
                    warnings.append(f"Requested primary_delay_ms={requested_delay_num:g} not found in {_display_path(path, repo_root)}")
            else:
                selected_parts = []
                for _, part in df.groupby(df.get("layer", pd.Series([""] * len(df))).astype(str), dropna=False):
                    max_delay = part["_delay_numeric"].max()
                    selected_parts.append(part[part["_delay_numeric"].eq(max_delay)])
                selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
            if not selected.empty:
                selected_delays.extend(float(v) for v in selected["_delay_numeric"].dropna().unique().tolist())
            group_cols = ["layer", "_delay_numeric"]
        else:
            selected = df.copy()
            group_cols = ["layer"] if "layer" in selected.columns else []
        if selected.empty:
            continue
        for key, part in selected.groupby(group_cols, dropna=False) if group_cols else [((), selected)]:
            if not isinstance(key, tuple):
                key = (key,)
            first = part.iloc[0]
            layer = str(first.get("layer", ""))
            delay_value = first.get("_delay_numeric", "") if "_delay_numeric" in part.columns else ""
            acc_values = pd.to_numeric(part.get("acc", pd.Series(dtype=float)), errors="coerce").dropna()
            acc_percent = acc_values.map(_to_percent)
            if acc_percent.empty:
                continue
            chance_value = _to_percent(first.get("chance", 0.1))
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric="delay_decode_accuracy",
                    condition=layer,
                    layer=layer,
                    value=float(acc_percent.mean()),
                    unit="percent",
                    seed_id=first.get("network_seed", _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    delay_ms=delay_value,
                    feature_type=first.get("feature_type", ""),
                    classifier=first.get("classifier", ""),
                    chance=chance_value,
                    n_train=first.get("n_train", ""),
                    n_test=first.get("n_test", ""),
                    run_mode=_run_mode(seeds),
                )
            )
    panel_df = pd.DataFrame(rows)
    if not panel_df.empty:
        panel_df["_layer_order"] = panel_df["layer"].map({name: i for i, name in enumerate(LAYER_ORDER)}).fillna(99)
        panel_df = panel_df.sort_values(["_layer_order", "seed_id"], kind="stable").drop(columns=["_layer_order"]).reset_index(drop=True)
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="delay_decode_accuracy", run_mode=_run_mode(seeds), group_cols=["layer"])
    stats["selected_primary_delay_ms"] = _unique_numeric(panel_df, "delay_ms")
    stats["all_available_delay_ms"] = sorted(set(float(v) for v in available_delays))
    stats["layers"] = _unique(panel_df, "layer")
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["selected_primary_delay_ms"] = stats["selected_primary_delay_ms"]
    manifest["all_available_delay_ms"] = stats["all_available_delay_ms"]
    manifest["layers"] = stats["layers"]
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_fig1_condition_metrics_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 experiment seed directory not found.")
    rows = []
    sources = []
    supplementary_conditions: list[str] = []
    warnings = _run_mode_warnings(seeds)
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_d_condition_metrics.csv"
        if not path.exists():
            warnings.append(f"Missing DMS condition metrics: {_display_path(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        supp_path = seed_dir / "data" / "metrics" / "supp_substrate_shuffle_metrics.csv"
        if supp_path.exists():
            supp_df = pd.read_csv(supp_path)
            supplementary_conditions.extend(str(v) for v in supp_df.get("condition", pd.Series(dtype=str)).dropna().unique())
        for _, row in df.iterrows():
            condition = str(row.get("condition", ""))
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric="error_rate",
                    condition=condition,
                    value=float(row.get("error_rate", 0.0)) * 100.0,
                    unit="percent",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    n_trials=row.get("n_trials", ""),
                    acc_probe=float(row.get("acc_probe", 0.0)) * 100.0,
                    sample_attribution_rate=float(row.get("sample_attribution_rate", 0.0)) * 100.0,
                    donor_attribution_rate=float(row.get("donor_attribution_rate", 0.0)) * 100.0,
                    condition_order=MAIN_CONDITIONS.index(condition) if condition in MAIN_CONDITIONS else 99,
                    run_mode=_run_mode(seeds),
                )
            )
    panel_df = pd.DataFrame(rows)
    if not panel_df.empty:
        panel_df = panel_df.sort_values(["condition_order", "seed_id"], kind="stable").reset_index(drop=True)
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="error_rate", run_mode=_run_mode(seeds), group_cols=["condition"])
    stats["conditions"] = _unique(panel_df, "condition")
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["conditions"] = stats["conditions"]
    manifest["supplementary_conditions"] = sorted(set(supplementary_conditions))
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_fig1_attribution_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 experiment seed directory not found.")
    rows = []
    sources = []
    warnings = _run_mode_warnings(seeds)
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_e_attribution_metrics.csv"
        if not path.exists():
            warnings.append(f"Missing attribution metrics: {_display_path(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        for _, row in df.iterrows():
            for trace, col in (("Original", "original_sample_attribution"), ("Donor", "donor_sample_attribution")):
                rows.append(
                    _canonical_row(
                        figure_id,
                        panel_id,
                        metric="attribution_rate",
                        condition=str(row.get("condition", "")),
                        value=float(row.get(col, 0.0)) * 100.0,
                        unit="percent",
                        seed_id=row.get("network_seed", _seed_id(seed_dir)),
                        source_file=_display_path(path, repo_root),
                        trace=trace,
                        donor_shift_gain_vs_dynamic=float(row.get("donor_shift_gain_vs_dynamic", 0.0)) * 100.0,
                        original_drop_vs_dynamic=float(row.get("original_drop_vs_dynamic", 0.0)) * 100.0,
                        run_mode=_run_mode(seeds),
                    )
                )
    panel_df = pd.DataFrame(rows)
    if not panel_df.empty:
        condition_order = {"dynamic_intact": 0, "ux_trial_shuffle": 1}
        trace_order = {"Original": 0, "Donor": 1}
        panel_df["_condition_order"] = panel_df["condition"].map(condition_order).fillna(99)
        panel_df["_trace_order"] = panel_df["trace"].map(trace_order).fillna(99)
        panel_df = panel_df.sort_values(["_condition_order", "_trace_order", "seed_id"], kind="stable").drop(columns=["_condition_order", "_trace_order"]).reset_index(drop=True)
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="attribution_rate", run_mode=_run_mode(seeds), group_cols=["condition", "trace"])
    stats["conditions"] = _unique(panel_df, "condition")
    stats["traces"] = _unique(panel_df, "trace")
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["conditions"] = stats["conditions"]
    manifest["traces"] = stats["traces"]
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_fig1_error_composition_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    source_levels: list[str] = []
    conditions = [str(v) for v in spec.get("conditions", ["dynamic_intact", "ux_trial_shuffle"])]
    for seed_dir in seeds:
        raw_path = seed_dir / "data" / "raw" / "panel_d_dms_condition_trial_readout.csv"
        condition_path = seed_dir / "data" / "metrics" / "panel_d_condition_metrics.csv"
        attribution_path = seed_dir / "data" / "metrics" / "panel_e_attribution_metrics.csv"
        seed_rows: list[dict[str, Any]] = []
        if raw_path.exists():
            df = pd.read_csv(raw_path)
            sources.append(_source_entry(raw_path, repo_root))
            seed_rows = _error_composition_from_trial_rows(df, seed_dir, raw_path, repo_root, figure_id, panel_id, conditions, seeds, warnings)
            source_levels.append("trial_level")
        elif condition_path.exists():
            df = pd.read_csv(condition_path)
            sources.append(_source_entry(condition_path, repo_root))
            seed_rows = _error_composition_from_condition_metrics(df, seed_dir, condition_path, repo_root, figure_id, panel_id, conditions, seeds, warnings)
            source_levels.append("condition_metrics_fallback")
            warnings.append(f"Fig.1E using condition-metrics fallback for {_display_path(condition_path, repo_root)}; trial-level readout is preferred.")
        elif attribution_path.exists():
            df = pd.read_csv(attribution_path)
            sources.append(_source_entry(attribution_path, repo_root))
            seed_rows = _error_composition_from_attribution_metrics(df, seed_dir, attribution_path, repo_root, figure_id, panel_id, conditions, seeds, warnings)
            source_levels.append("attribution_metrics_estimated")
            warnings.append("Other category estimated from panel_e_attribution_metrics; use trial-level readout for final manuscript.")
        else:
            warnings.append(f"Missing Fig.1E composition sources under {_display_path(seed_dir, repo_root)}")
        rows.extend(seed_rows)
    panel_df = pd.DataFrame(rows)
    if not panel_df.empty:
        condition_order = {condition: i for i, condition in enumerate(conditions)}
        category_order = {"Original": 0, "Donor": 1, "Other": 2}
        panel_df["_condition_order"] = panel_df["condition"].map(condition_order).fillna(99)
        panel_df["_category_order"] = panel_df["category"].map(category_order).fillna(99)
        panel_df = panel_df.sort_values(["_condition_order", "_category_order", "seed_id"], kind="stable").drop(columns=["_condition_order", "_category_order"]).reset_index(drop=True)
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="error_composition_within_error_pool", run_mode=_run_mode(seeds), group_cols=["condition", "category"])
    sums = _composition_sums(panel_df)
    stats["per_seed_condition_sums"] = sums
    stats["max_abs_sum_deviation_from_100"] = max((abs(float(item["sum_percent"]) - 100.0) for item in sums), default=0.0)
    stats["source_level"] = _source_level(source_levels)
    stats["conditions"] = _unique(panel_df, "condition")
    stats["categories"] = _unique(panel_df, "category")
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["source_level"] = stats["source_level"]
    manifest["conditions"] = stats["conditions"]
    manifest["categories"] = stats["categories"]
    manifest["per_seed_condition_sums"] = sums
    manifest["max_abs_sum_deviation_from_100"] = stats["max_abs_sum_deviation_from_100"]
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


# Backward-compatible names used by older Fig.1 specs/build artifacts.
build_fig1_overall_recall = build_fig1_baseline_adapter
build_fig1_delay_decoding = build_fig1_delay_decode_adapter
build_fig1_stsp_outcome_profile = build_fig1_condition_metrics_adapter
build_fig1_attribution_transfer = build_fig1_attribution_adapter


def _error_composition_from_trial_rows(
    df: pd.DataFrame,
    seed_dir: Path,
    path: Path,
    repo_root: Path,
    figure_id: str,
    panel_id: str,
    conditions: Sequence[str],
    seeds: Sequence[Path],
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty or "condition" not in df.columns:
        warnings.append(f"Trial-level Fig.1E source has no condition rows: {_display_path(path, repo_root)}")
        return rows
    for condition in conditions:
        part = df[df["condition"].astype(str).eq(condition)].copy()
        if part.empty:
            warnings.append(f"Trial-level Fig.1E source missing condition {condition}: {_display_path(path, repo_root)}")
            continue
        correct = _bool_series(part, "is_correct_probe")
        if correct is None:
            correct = _bool_series(part, "correct_probe")
        silent = _silent_series(part)
        if correct is None:
            error_mask = silent
            warnings.append(f"Trial-level Fig.1E source lacks is_correct_probe; using silent/no-response rows only for {condition}.")
        else:
            error_mask = (~correct) | silent
        errors = part[error_mask].copy()
        n_error = int(len(errors))
        n_trials = int(len(part))
        if n_error:
            original_mask = _bool_series(errors, "pred_is_original_sample", default_false=True)
            donor_col = "pred_is_donor_shifted_memory" if "pred_is_donor_shifted_memory" in errors.columns else "pred_is_donor_sample"
            donor_raw = _bool_series(errors, donor_col, default_false=True) if donor_col in errors.columns else pd.Series(False, index=errors.index)
            donor_mask = donor_raw & ~original_mask
            other_mask = pd.Series(True, index=errors.index) & ~original_mask & ~donor_mask
            original = int(original_mask.sum())
            donor = int(donor_mask.sum())
            other = int(other_mask.sum())
        else:
            original = donor = other = 0
        rows.extend(
            _composition_rows_from_counts(
                figure_id,
                panel_id,
                condition,
                original,
                donor,
                other,
                seed_id=_seed_value(part, seed_dir),
                source_file=_display_path(path, repo_root),
                n_error=n_error,
                n_trials=n_trials,
                run_mode=_run_mode(seeds),
                source_level="trial_level",
                warnings=warnings,
            )
        )
    return rows


def _error_composition_from_condition_metrics(
    df: pd.DataFrame,
    seed_dir: Path,
    path: Path,
    repo_root: Path,
    figure_id: str,
    panel_id: str,
    conditions: Sequence[str],
    seeds: Sequence[Path],
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        part = df[df.get("condition", pd.Series(dtype=str)).astype(str).eq(condition)]
        if part.empty:
            warnings.append(f"Condition metrics Fig.1E fallback missing condition {condition}: {_display_path(path, repo_root)}")
            continue
        row = part.iloc[0]
        error_fraction = max(_fraction(row.get("error_rate", 0.0)), 0.0)
        original_fraction = max(_fraction(row.get("sample_attribution_rate", 0.0)), 0.0)
        donor_fraction = max(_fraction(row.get("donor_attribution_rate", 0.0)), 0.0)
        if "other_attribution_rate" in part.columns or "silent_rate" in part.columns:
            other_fraction = max(_fraction(row.get("other_attribution_rate", 0.0)), 0.0) + max(_fraction(row.get("silent_rate", 0.0)), 0.0)
        else:
            other_fraction = max(error_fraction - original_fraction - donor_fraction, 0.0)
        rows.extend(
            _composition_rows_from_fractions(
                figure_id,
                panel_id,
                condition,
                original_fraction,
                donor_fraction,
                other_fraction,
                denominator=error_fraction,
                seed_id=row.get("network_seed", _seed_id(seed_dir)),
                source_file=_display_path(path, repo_root),
                n_error=_estimated_count(error_fraction, row.get("n_trials", "")),
                n_trials=row.get("n_trials", ""),
                run_mode=_run_mode(seeds),
                source_level="condition_metrics_fallback",
            )
        )
    return rows


def _error_composition_from_attribution_metrics(
    df: pd.DataFrame,
    seed_dir: Path,
    path: Path,
    repo_root: Path,
    figure_id: str,
    panel_id: str,
    conditions: Sequence[str],
    seeds: Sequence[Path],
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _ = warnings
    for condition in conditions:
        part = df[df.get("condition", pd.Series(dtype=str)).astype(str).eq(condition)]
        if part.empty:
            continue
        row = part.iloc[0]
        original_fraction = max(_fraction(row.get("original_sample_attribution", row.get("sample_attribution_rate", 0.0))), 0.0)
        donor_fraction = max(_fraction(row.get("donor_sample_attribution", row.get("donor_attribution_rate", 0.0))), 0.0)
        other_fraction = max(1.0 - original_fraction - donor_fraction, 0.0)
        rows.extend(
            _composition_rows_from_fractions(
                figure_id,
                panel_id,
                condition,
                original_fraction,
                donor_fraction,
                other_fraction,
                denominator=1.0,
                seed_id=row.get("network_seed", _seed_id(seed_dir)),
                source_file=_display_path(path, repo_root),
                n_error="",
                n_trials="",
                run_mode=_run_mode(seeds),
                source_level="attribution_metrics_estimated",
            )
        )
    return rows


def _composition_rows_from_counts(
    figure_id: str,
    panel_id: str,
    condition: str,
    original: int,
    donor: int,
    other: int,
    *,
    seed_id: Any,
    source_file: str,
    n_error: int,
    n_trials: Any,
    run_mode: str,
    source_level: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if n_error <= 0:
        warnings.append(f"Fig.1E {condition} has n_error=0; composition values are undefined and written as 0.")
        denom = 1.0
    else:
        denom = float(n_error)
    total = original + donor + other
    if n_error > 0 and total != n_error:
        warnings.append(f"Fig.1E {condition} category counts sum to {total}, not n_error={n_error}; normalizing plotted composition.")
        denom = float(total) if total > 0 else float(n_error)
    return [
        _canonical_row(
            figure_id,
            panel_id,
            metric="error_composition_within_error_pool",
            condition=condition,
            value=(count / denom) * 100.0 if denom else 0.0,
            unit="percent_of_errors",
            seed_id=seed_id,
            source_file=source_file,
            category=category,
            trace=category,
            n_error=n_error,
            n_trials=n_trials,
            run_mode=run_mode,
            source_level=source_level,
        )
        for category, count in (("Original", original), ("Donor", donor), ("Other", other))
    ]


def _composition_rows_from_fractions(
    figure_id: str,
    panel_id: str,
    condition: str,
    original: float,
    donor: float,
    other: float,
    *,
    denominator: float,
    seed_id: Any,
    source_file: str,
    n_error: Any,
    n_trials: Any,
    run_mode: str,
    source_level: str,
) -> list[dict[str, Any]]:
    total = original + donor + other
    denom = denominator if denominator > 0 else total
    if denom <= 0:
        denom = 1.0
    values = {"Original": original / denom * 100.0, "Donor": donor / denom * 100.0, "Other": other / denom * 100.0}
    total_percent = sum(values.values())
    if total_percent > 0:
        values = {key: value * 100.0 / total_percent for key, value in values.items()}
    return [
        _canonical_row(
            figure_id,
            panel_id,
            metric="error_composition_within_error_pool",
            condition=condition,
            value=float(values[category]),
            unit="percent_of_errors",
            seed_id=seed_id,
            source_file=source_file,
            category=category,
            trace=category,
            n_error=n_error,
            n_trials=n_trials,
            run_mode=run_mode,
            source_level=source_level,
        )
        for category in ("Original", "Donor", "Other")
    ]


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig1")), str(spec.get("panel_id", "")).upper()


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


def _stats_payload(figure_id: str, panel_id: str, panel_df: pd.DataFrame, *, metric: str, run_mode: str, group_cols: list[str]) -> dict[str, Any]:
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


def _fraction(value: Any) -> float:
    numeric = _to_number(value)
    if numeric is None:
        return 0.0
    return float(numeric / 100.0 if abs(numeric) > 1.5 else numeric)


def _seed_value(df: pd.DataFrame, seed_dir: Path) -> Any:
    if "network_seed" in df.columns and not df["network_seed"].dropna().empty:
        return df["network_seed"].dropna().iloc[0]
    return _seed_id(seed_dir)


def _bool_series(df: pd.DataFrame, col: str, *, default_false: bool = False) -> pd.Series | None:
    if col not in df.columns:
        return pd.Series(False, index=df.index) if default_false else None
    values = df[col]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip().str.lower()
    return numeric.eq(1) | text.isin({"true", "t", "yes", "y"})


def _silent_series(df: pd.DataFrame) -> pd.Series:
    silent = pd.Series(False, index=df.index)
    for col in ("is_silent_probe", "silent"):
        flags = _bool_series(df, col)
        if flags is not None:
            silent = silent | flags
    for col in ("prediction_probe", "prediction"):
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            text = df[col].astype(str).str.strip().str.lower()
            silent = silent | numeric.lt(0).fillna(False) | text.isin({"silent", "no_response", "no-response", "none"})
    return silent


def _estimated_count(fraction: float, n_trials: Any) -> Any:
    n = _to_number(n_trials)
    if n is None:
        return ""
    return int(round(max(fraction, 0.0) * n))


def _composition_sums(panel_df: pd.DataFrame) -> list[dict[str, Any]]:
    if panel_df.empty or not {"seed_id", "condition", "value"}.issubset(panel_df.columns):
        return []
    out = []
    grouped = panel_df.groupby(["seed_id", "condition"], dropna=False)
    for (seed_id, condition), part in grouped:
        total = float(pd.to_numeric(part["value"], errors="coerce").dropna().sum())
        out.append({"seed_id": str(seed_id), "condition": str(condition), "sum_percent": total})
    return out


def _source_level(levels: Sequence[str]) -> str:
    unique = sorted(set(levels))
    if not unique:
        return "missing_source"
    if len(unique) == 1:
        return unique[0]
    return "mixed:" + ",".join(unique)
