from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.plotting.paper_fig.adapters.fig2_adapters import (
    STATE_CONDITIONS,
    _canonical_row,
    _display_path,
    _first_col,
    _manifest,
    _run_mode,
    _run_mode_warnings,
    _seed_dirs,
    _seed_id,
    _source_entry,
    _stats_payload,
    _to_percent,
    _unique,
    _unique_numeric,
)
from src.plotting.paper_fig.data_resolver import AdapterResult, missing_adapter_result, write_adapter_outputs


LAYER_ORDER = ("layer1", "layer2", "layer3")
MODEL_ORDER = ("A_only", "B_only", "mean_AB", "sum_AB", "unconstrained_AB", "convex_AB")
EXPECTED_COMPLETION_DELAY_MS = (100, 200, 300, 400, 800, 1200)


def build_s3_wpri_across_layers_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_layerwise_metric(
        spec,
        repo_root,
        output_dir,
        metric_name="WPRI",
        value_candidates=("WPRI", "wpri"),
        default_sources=("supp_layerwise_morphology_metrics.csv", "panel_d_pair_level_organization_metrics.csv"),
        y_unit="score",
    )


def build_s3_residual_across_layers_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_layerwise_metric(
        spec,
        repo_root,
        output_dir,
        metric_name="beyond_linear_pair_index",
        value_candidates=("beyond_linear_pair_index", "residual_pair_specificity"),
        default_sources=("supp_layerwise_morphology_metrics.csv", "panel_d_linear_residual_pair_specificity_metrics.csv"),
        y_unit="score",
        fallback_warning="S3B used residual_pair_specificity fallback because beyond_linear_pair_index was unavailable.",
    )


def build_s3_linear_model_comparison_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2 supplement experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    model_order = [str(v) for v in spec.get("model_order", MODEL_ORDER)]
    for seed_dir in seeds:
        path = _first_existing(seed_dir, spec, ("supp_linear_mixture_model_comparison.csv", "panel_d_linear_mixture_fit_metrics.csv"))
        if path is None:
            warnings.append(f"Missing S3C linear model comparison source under {_display_path(seed_dir, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        model_col = _first_col(df, ["model_name", "mixture_model", "model"])
        if model_col is None:
            warnings.append(f"S3C source missing model column: {_display_path(path, repo_root)}")
            continue
        value_col = _preferred_fit_col(df)
        if value_col is None:
            warnings.append(f"S3C source missing r2/cv_r2/fit_r2 columns: {_display_path(path, repo_root)}")
            continue
        work = df[df[model_col].astype(str).isin(model_order)].copy()
        for _, row in work.iterrows():
            value = pd.to_numeric(pd.Series([row.get(value_col)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            model = str(row.get(model_col, ""))
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric=value_col,
                    condition=model,
                    value=float(value),
                    unit="r2",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    layer=row.get("layer", ""),
                    pair_id=row.get("pair_id", ""),
                    state_variable=row.get("state_variable", ""),
                    model_name=model,
                    cv_r2=row.get("cv_r2", ""),
                    r2=row.get("r2", row.get("fit_r2", "")),
                    run_mode=_run_mode(seeds),
                )
            )
    if not sources:
        return missing_adapter_result(spec, repo_root, output_dir, "Missing S3C linear model comparison source.")
    raw_panel_df = _sort_by_order(pd.DataFrame(rows), "condition", model_order)
    panel_df = _seed_level_summary(raw_panel_df, ["condition", "model_name"])
    if figure_id == "supp_fig_s2":
        panel_df = panel_df.drop(columns=["pair_id"], errors="ignore")
    stats = _stats_payload(figure_id, panel_id, panel_df, "linear_model_comparison", _run_mode(seeds), ["model_name"])
    stats["model_order"] = model_order
    stats["models"] = _unique(panel_df, "model_name")
    _add_seed_aggregation_stats(stats, raw_panel_df, panel_df)
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["model_order"] = model_order
    _add_output_manifest_fields(manifest, panel_df)
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_s4_ping_sweep_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    sweep_parameter = str(spec.get("sweep_parameter", "ping_amp"))
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2 supplement experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "supp_ping_sweep_metrics.csv"
        if not path.exists():
            warnings.append(f"Missing S4 ping sweep source: {_display_path(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        if sweep_parameter not in df.columns:
            warnings.append(f"S4 ping sweep source missing {sweep_parameter}: {_display_path(path, repo_root)}")
            continue
        work = df.copy()
        if "sweep_type" in work.columns:
            expected_type = "amplitude" if sweep_parameter == "ping_amp" else "duration"
            typed = work[work["sweep_type"].astype(str).eq(expected_type)].copy()
            if not typed.empty:
                work = typed
        value_col = _first_col(work, ["pair_member_readout_rate", "P_pair", "pair_readout_rate"])
        if value_col is None:
            warnings.append(f"S4 ping sweep source missing pair-member readout rate: {_display_path(path, repo_root)}")
            continue
        for _, row in work.iterrows():
            condition = str(row.get("state_condition", ""))
            if condition not in STATE_CONDITIONS:
                continue
            x_value = pd.to_numeric(pd.Series([row.get(sweep_parameter)]), errors="coerce").iloc[0]
            if pd.isna(x_value):
                continue
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric="pair_member_readout",
                    condition=condition,
                    value=_to_percent(row.get(value_col, 0.0)),
                    unit="percent",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    state_condition=condition,
                    x_value=float(x_value),
                    sweep_parameter=sweep_parameter,
                    ping_amp=row.get("ping_amp", ""),
                    ping_ms=row.get("ping_ms", ""),
                    P_A=_to_percent(row.get("P_A", row.get("A_readout_rate", np.nan))),
                    P_B=_to_percent(row.get("P_B", row.get("B_readout_rate", np.nan))),
                    P_silent=_to_percent(row.get("P_silent", row.get("silent_rate", np.nan))),
                    n_trials=row.get("n_trials", ""),
                    run_mode=_run_mode(seeds),
                )
            )
    if not sources:
        return missing_adapter_result(spec, repo_root, output_dir, "Missing supp_ping_sweep_metrics.csv.")
    panel_df = _sort_line(pd.DataFrame(rows), "x_value", STATE_CONDITIONS)
    stats = _stats_payload(figure_id, panel_id, panel_df, "pair_member_readout", _run_mode(seeds), ["condition", "x_value"])
    stats["sweep_parameter"] = sweep_parameter
    stats["x_values"] = _unique_numeric(panel_df, "x_value")
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["sweep_parameter"] = sweep_parameter
    _add_output_manifest_fields(manifest, panel_df)
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_s4_completion_delay_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2 supplement experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    for seed_dir in seeds:
        contrast_path = seed_dir / "data" / "metrics" / "supp_completion_delay_sweep_contrast.csv"
        metrics_path = seed_dir / "data" / "metrics" / "supp_completion_delay_sweep_metrics.csv"
        if contrast_path.exists():
            df = pd.read_csv(contrast_path)
            sources.append(_source_entry(contrast_path, repo_root))
            value_col = _first_col(df, ["completion_gain_SAB_minus_relevant_single", "completion_gain_SAB_minus_SB", "completion_gain_SAB_minus_SA"])
            delay_col = _first_col(df, ["delay2_ms", "completion_delay_ms", "post_pair_delay_ms"])
            if value_col is None or delay_col is None:
                warnings.append(f"{panel_id} contrast source missing gain/delay columns: {_display_path(contrast_path, repo_root)}")
                continue
            for _, row in df.iterrows():
                delay = pd.to_numeric(pd.Series([row.get(delay_col)]), errors="coerce").iloc[0]
                if pd.isna(delay):
                    continue
                rows.append(
                    _canonical_row(
                        figure_id,
                        panel_id,
                        metric="completion_gain",
                        condition="S_AB_minus_relevant_single",
                        value=_to_percent(row.get(value_col, 0.0)),
                        unit="percent",
                        seed_id=row.get("network_seed", _seed_id(seed_dir)),
                        source_file=_display_path(contrast_path, repo_root),
                        x_value=float(delay),
                        delay2_ms=float(delay),
                        keep_prob=row.get("keep_prob", ""),
                        gain_column=value_col,
                        n_trials=row.get("n_trials_SAB", row.get("n_trials", "")),
                        run_mode=_run_mode(seeds),
                    )
                )
            continue
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            sources.append(_source_entry(metrics_path, repo_root))
            rows.extend(_completion_rows_from_metrics(df, figure_id, panel_id, metrics_path, repo_root, seed_dir, seeds, warnings))
        else:
            warnings.append(f"Missing S4 completion delay sources under {_display_path(seed_dir, repo_root)}")
    if not sources:
        return missing_adapter_result(spec, repo_root, output_dir, "Missing completion delay sweep sources.")
    panel_df = _sort_line(pd.DataFrame(rows), "x_value", ["S_AB_minus_relevant_single", "target_A", "target_B"])
    stats = _stats_payload(figure_id, panel_id, panel_df, "completion_gain", _run_mode(seeds), ["condition", "x_value"])
    expected_delay_ms = [int(v) for v in spec.get("expected_delay_ms", EXPECTED_COMPLETION_DELAY_MS)]
    missing_delay_ms = _missing_numeric_values(panel_df, "delay2_ms", expected_delay_ms)
    stats["x_values"] = _unique_numeric(panel_df, "x_value")
    stats["expected_delay_ms"] = expected_delay_ms
    stats["missing_expected_delay_ms"] = missing_delay_ms
    stats["metric"] = "completion_gain"
    stats["condition"] = "S_AB_minus_relevant_single"
    if missing_delay_ms and not panel_df.get("metric", pd.Series(dtype=str)).astype(str).eq("missing_source").any():
        warnings.append(f"{panel_id} missing expected completion delay values: {missing_delay_ms}")
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["expected_delay_ms"] = expected_delay_ms
    manifest["missing_expected_delay_ms"] = missing_delay_ms
    _add_output_manifest_fields(manifest, panel_df)
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def _build_layerwise_metric(
    spec: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    *,
    metric_name: str,
    value_candidates: Sequence[str],
    default_sources: Sequence[str],
    y_unit: str,
    fallback_warning: str | None = None,
) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2 supplement experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    for seed_dir in seeds:
        path = _first_existing(seed_dir, spec, default_sources)
        if path is None:
            warnings.append(f"Missing {panel_id} layerwise source under {_display_path(seed_dir, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        value_col = _first_col(df, value_candidates)
        if value_col is None:
            warnings.append(f"{panel_id} source missing {list(value_candidates)}: {_display_path(path, repo_root)}")
            continue
        if fallback_warning and value_col != value_candidates[0] and fallback_warning not in warnings:
            warnings.append(fallback_warning)
        layer_col = _first_col(df, ["layer", "primary_layer"])
        work = df.copy()
        if layer_col is None:
            work["layer"] = "layer3"
            layer_col = "layer"
        for _, row in work.iterrows():
            layer = str(row.get(layer_col, ""))
            value = pd.to_numeric(pd.Series([row.get(value_col)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric=metric_name if value_col == value_candidates[0] else value_col,
                    condition=layer,
                    value=float(value),
                    unit=y_unit,
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    layer=layer,
                    pair_id=row.get("pair_id", ""),
                    state_variable=row.get("state_variable", row.get("primary_state_variable", "")),
                    run_mode=_run_mode(seeds),
                )
            )
    if not sources:
        return missing_adapter_result(spec, repo_root, output_dir, f"Missing {panel_id} layerwise source.")
    raw_panel_df = _sort_by_order(pd.DataFrame(rows), "layer", LAYER_ORDER)
    panel_df = _seed_level_summary(raw_panel_df, ["layer", "condition"])
    if figure_id == "supp_fig_s2":
        panel_df = panel_df.drop(columns=["pair_id"], errors="ignore")
    stats = _stats_payload(figure_id, panel_id, panel_df, metric_name, _run_mode(seeds), ["layer"])
    stats["layers"] = _unique(panel_df, "layer")
    _add_seed_aggregation_stats(stats, raw_panel_df, panel_df)
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["layers"] = stats["layers"]
    _add_output_manifest_fields(manifest, panel_df)
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def _completion_rows_from_metrics(
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
    required = {"state_condition", "target_recovery_rate"}
    delay_col = _first_col(df, ["delay2_ms", "completion_delay_ms", "post_pair_delay_ms"])
    if delay_col is None or not required.issubset(df.columns):
        warnings.append(f"{panel_id} metrics fallback missing state/recovery/delay columns: {_display_path(path, repo_root)}")
        return rows
    work = df.copy()
    seed_col = _first_col(work, ["network_seed", "seed_id", "network_id"])
    if seed_col is None:
        work["network_seed"] = _seed_id(seed_dir)
        seed_col = "network_seed"
    group_cols = [seed_col, delay_col]
    if "keep_prob" in work.columns:
        group_cols.append("keep_prob")
    for keys, part in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        seed_value = keys[0]
        delay = pd.to_numeric(pd.Series([keys[1]]), errors="coerce").iloc[0]
        if pd.isna(delay):
            continue
        keep_prob = keys[2] if len(keys) > 2 else ""
        lookup = {
            str(row["state_condition"]): _to_percent(row.get("target_recovery_rate", 0.0))
            for _, row in part.iterrows()
        }
        if "S_AB" not in lookup:
            continue
        relevant = "S_B" if "S_B" in lookup else "S_A" if "S_A" in lookup else ""
        if not relevant:
            continue
        rows.append(
            _canonical_row(
                figure_id,
                panel_id,
                metric="completion_gain",
                condition="S_AB_minus_relevant_single",
                value=float(lookup["S_AB"] - lookup[relevant]),
                unit="percent",
                seed_id=seed_value,
                source_file=_display_path(path, repo_root),
                x_value=float(delay),
                delay2_ms=float(delay),
                keep_prob=keep_prob,
                relevant_single_condition=relevant,
                source_level="metrics_fallback",
                run_mode=_run_mode(seeds),
            )
        )
    return rows


def _add_output_manifest_fields(manifest: dict[str, Any], panel_df: pd.DataFrame) -> None:
    manifest["source_files_used"] = [str(source.get("path", "")) for source in manifest.get("sources", []) if source.get("exists", True)]
    manifest["rows_written_to_panel_data"] = int(len(panel_df))
    manifest["rows_written"] = int(len(panel_df))
    manifest["n_networks_observed"] = int(manifest.get("n_networks") or 0)


def _missing_numeric_values(panel_df: pd.DataFrame, column: str, expected: Sequence[int]) -> list[int]:
    if panel_df.empty or column not in panel_df.columns:
        return list(expected)
    observed = {
        int(round(float(value)))
        for value in pd.to_numeric(panel_df[column], errors="coerce").dropna().tolist()
    }
    return [int(value) for value in expected if int(value) not in observed]


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig2_supp")), str(spec.get("panel_id", "")).upper()


def _add_seed_aggregation_stats(stats: dict[str, Any], raw_df: pd.DataFrame, panel_df: pd.DataFrame) -> None:
    stats["seed_level_aggregation"] = True
    stats["rows_before_seed_aggregation"] = int(len(raw_df))
    stats["rows_after_seed_aggregation"] = int(len(panel_df))
    stats["replicate_unit"] = "network_seed"


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


def _first_existing(seed_dir: Path, spec: Mapping[str, Any], default_names: Sequence[str]) -> Path | None:
    candidates: list[str] = []
    candidates.extend(str(v) for v in spec.get("source_priority", []) or [])
    if spec.get("source"):
        candidates.append(str(spec.get("source")))
    candidates.extend(f"data/metrics/{name}" for name in default_names)
    for candidate in candidates:
        path = seed_dir / candidate
        if path.exists():
            return path
    return None


def _preferred_fit_col(df: pd.DataFrame) -> str | None:
    for col in ("cv_r2", "r2", "fit_r2"):
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            return col
    return None


def _sort_by_order(df: pd.DataFrame, col: str, order: Sequence[str]) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out["_order"] = out[col].map({name: i for i, name in enumerate(order)}).fillna(99)
    sort_cols = ["_order"]
    if "seed_id" in out.columns:
        sort_cols.append("seed_id")
    if "pair_id" in out.columns:
        sort_cols.append("pair_id")
    return out.sort_values(sort_cols, kind="stable").drop(columns=["_order"]).reset_index(drop=True)


def _sort_line(df: pd.DataFrame, x_col: str, condition_order: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_condition_order"] = out.get("condition", pd.Series(dtype=str)).map({name: i for i, name in enumerate(condition_order)}).fillna(99)
    out["_x"] = pd.to_numeric(out.get(x_col, pd.Series(dtype=float)), errors="coerce")
    return out.sort_values(["_condition_order", "_x", "seed_id"], kind="stable").drop(columns=["_condition_order", "_x"]).reset_index(drop=True)
