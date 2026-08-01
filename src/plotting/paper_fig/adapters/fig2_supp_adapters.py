from __future__ import annotations

import hashlib
import json
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


_S2_PERSISTED_ROOT = Path("results/paper_figures/outputs/supplementary_part2/supp_fig_s2")
_S2_EXPECTED_NETWORK_IDS = tuple(str(seed) for seed in range(1000, 1020))
_S2_PERSISTED_INPUTS: dict[str, dict[str, Any]] = {
    "A": {
        "panel_data": "panel_data/supp_fig_s2a_panel_data.csv",
        "panel_data_sha256": "6866ea1d75d22937a3e2f461b94acd772c526e9e42b6540dd76396bd839b0b6d",
        "stats": "stats/supp_fig_s2a_stats.json",
        "stats_sha256": "de5a1feeea1ab172afb0e01cb71c8794a7620ca09f52c2e90e764b544caa1c63",
        "manifest": "source_manifests/supp_fig_s2a_sources.json",
        "manifest_sha256": "7ba648190db36a1f6b26f5cceda7e3bdf99832e6edd5ee5231d85517307b3517",
        "columns": (
            "figure_id", "panel_id", "metric", "condition", "layer", "network_id", "seed_id", "value",
            "unit", "source_file", "state_variable", "run_mode", "seed_level_n", "seed_level_sem", "replicate_unit",
        ),
        "condition_order": ("layer1", "layer2", "layer3"),
        "metric_order": ("WPRI",),
        "identity_columns": ("network_id", "layer"),
        "summary_key": "layer",
    },
    "B": {
        "panel_data": "panel_data/supp_fig_s2b_panel_data.csv",
        "panel_data_sha256": "cf553a0cf578039d8196d8d5c00a40a27702e9ec3c276691aadd1c077f25bbcb",
        "stats": "stats/supp_fig_s2b_stats.json",
        "stats_sha256": "04fc7aa1c90145f7ef0f78c969aed5d2e90c852f3f6bde4c9c18b4fc658e98f2",
        "manifest": "source_manifests/supp_fig_s2b_sources.json",
        "manifest_sha256": "7eb04f4615cb73feac90dcd848e3d1ec59f77918907949286cbdc480eca5b61f",
        "columns": (
            "figure_id", "panel_id", "metric", "condition", "layer", "network_id", "seed_id", "value", "unit",
            "source_file", "state_variable", "n_pairs", "n_features", "n_folds", "analysis_status", "run_mode",
        ),
        "condition_order": ("Linear additive", "Quadratic marginals", "Bounded saturation"),
        "metric_order": (
            "delta_r2_linear_interaction",
            "delta_r2_interaction_beyond_marginal_nonlinearity",
            "delta_r2_interaction_beyond_bounded_saturation",
        ),
        "identity_columns": ("network_id", "condition", "metric"),
        "summary_key": "condition",
    },
    "C": {
        "panel_data": "panel_data/supp_fig_s2c_panel_data.csv",
        "panel_data_sha256": "f3c6626bdd8b510bda1b4fc0577128f4a327239e98851d6ad8b1bce2e1515fe4",
        "stats": "stats/supp_fig_s2c_stats.json",
        "stats_sha256": "48d2fdb70abb57a4258f372c30736bc6a4fd85c5a8bc033dbfcb2576744fa10c",
        "manifest": "source_manifests/supp_fig_s2c_sources.json",
        "manifest_sha256": "f3309865bf0d1231f17428c3b24327340bcc7da24175abeeb34910aed905f020",
        "columns": (
            "figure_id", "panel_id", "metric", "condition", "layer", "network_id", "seed_id", "value", "unit",
            "source_file", "state_variable", "null_model", "replicate", "endpoint", "calibration_role",
            "aggregation_level", "observed_reference_delta_r2", "feature_count", "noise_scale_ratio", "permutation_rule",
            "run_mode", "n_null_replicates",
        ),
        "condition_order": ("Linear + noise", "Bounded saturation", "Sequence/marginal matched"),
        "metric_order": ("null_delta_r2",),
        "identity_columns": ("network_id", "null_model", "condition"),
        "summary_key": "condition",
    },
}


def build_s3_crossfit_interaction_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _load_fixed_s2_panel(spec, repo_root, output_dir, expected_panel_id="B")


def build_s3_crossfit_null_calibration_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _load_fixed_s2_panel(spec, repo_root, output_dir, expected_panel_id="C")


def build_s3_wpri_across_layers_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _load_fixed_s2_panel(spec, repo_root, output_dir, expected_panel_id="A")


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


def _load_fixed_s2_panel(
    spec: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    *,
    expected_panel_id: str,
) -> AdapterResult:
    """Load one immutable S2 plotting payload without scientific recomputation.

    The three active S2 adapters intentionally terminate here.  They read the
    hash-frozen panel rows, statistics, and source manifest already audited by
    the data-lineage module.  Missing, stale, duplicate, extra, reordered, or
    fallback content is a hard error; there is no experiment-source fallback.
    """
    figure_id, panel_id = _ids(spec)
    if figure_id != "supp_fig_s2" or panel_id != expected_panel_id:
        raise RuntimeError(
            f"Fixed S2 adapter identity mismatch: expected supp_fig_s2/{expected_panel_id}, "
            f"received {figure_id}/{panel_id}."
        )
    contract = _S2_PERSISTED_INPUTS[panel_id]
    paths = {
        kind: repo_root / _S2_PERSISTED_ROOT / str(contract[kind])
        for kind in ("panel_data", "stats", "manifest")
    }
    for kind, path in paths.items():
        expected_sha = str(contract[f"{kind}_sha256"])
        if not path.is_file():
            raise RuntimeError(f"Fixed S2{panel_id} {kind} input is missing: {path}")
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Fixed S2{panel_id} {kind} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}: {path}"
            )

    panel_df = pd.read_csv(paths["panel_data"], float_precision="round_trip")
    expected_columns = list(contract["columns"])
    if list(panel_df.columns) != expected_columns:
        raise RuntimeError(
            f"Fixed S2{panel_id} schema mismatch: expected {expected_columns}, got {list(panel_df.columns)}"
        )
    if len(panel_df) != 60:
        raise RuntimeError(f"Fixed S2{panel_id} row-count mismatch: expected 60, got {len(panel_df)}")
    if panel_df[list(contract["identity_columns"])].isna().any().any():
        raise RuntimeError(f"Fixed S2{panel_id} has missing row-identity fields.")
    if panel_df.duplicated(subset=list(contract["identity_columns"]), keep=False).any():
        raise RuntimeError(f"Fixed S2{panel_id} has duplicate row identities.")
    if panel_df["value"].isna().any() or not np.isfinite(panel_df["value"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"Fixed S2{panel_id} contains missing or non-finite values.")
    if set(panel_df["figure_id"].astype(str)) != {"supp_fig_s2"}:
        raise RuntimeError(f"Fixed S2{panel_id} contains extra figure identities.")
    if set(panel_df["panel_id"].astype(str)) != {panel_id}:
        raise RuntimeError(f"Fixed S2{panel_id} contains extra panel identities.")

    observed_network_ids = tuple(
        dict.fromkeys(pd.to_numeric(panel_df["network_id"], errors="raise").astype(int).astype(str).tolist())
    )
    if observed_network_ids != _S2_EXPECTED_NETWORK_IDS:
        raise RuntimeError(
            f"Fixed S2{panel_id} network identity/order mismatch: "
            f"expected {_S2_EXPECTED_NETWORK_IDS}, got {observed_network_ids}"
        )
    network_counts = panel_df["network_id"].value_counts(sort=False)
    if len(network_counts) != 20 or set(int(value) for value in network_counts.tolist()) != {3}:
        raise RuntimeError(f"Fixed S2{panel_id} network membership is missing or extra: {network_counts.to_dict()}")

    condition_order = tuple(dict.fromkeys(panel_df["condition"].astype(str).tolist()))
    if condition_order != tuple(contract["condition_order"]):
        raise RuntimeError(
            f"Fixed S2{panel_id} condition order mismatch: expected {contract['condition_order']}, got {condition_order}"
        )
    metric_order = tuple(dict.fromkeys(panel_df["metric"].astype(str).tolist()))
    if metric_order != tuple(contract["metric_order"]):
        raise RuntimeError(
            f"Fixed S2{panel_id} metric order mismatch: expected {contract['metric_order']}, got {metric_order}"
        )
    for condition in condition_order:
        part = panel_df.loc[panel_df["condition"].astype(str).eq(condition)]
        condition_networks = tuple(
            pd.to_numeric(part["network_id"], errors="raise").astype(int).astype(str).tolist()
        )
        if condition_networks != _S2_EXPECTED_NETWORK_IDS:
            raise RuntimeError(f"Fixed S2{panel_id}/{condition} row order is missing, extra, or reordered.")

    if panel_id == "A" and not panel_df["layer"].astype(str).eq(panel_df["condition"].astype(str)).all():
        raise RuntimeError("Fixed S2A layer/condition identity mismatch.")
    if panel_id == "B":
        expected_pairs = {
            "Linear additive": "delta_r2_linear_interaction",
            "Quadratic marginals": "delta_r2_interaction_beyond_marginal_nonlinearity",
            "Bounded saturation": "delta_r2_interaction_beyond_bounded_saturation",
        }
        if any(
            set(panel_df.loc[panel_df["condition"].astype(str).eq(condition), "metric"].astype(str)) != {metric}
            for condition, metric in expected_pairs.items()
        ):
            raise RuntimeError("Fixed S2B condition/metric identity mismatch.")
    if panel_id == "C":
        expected_null_models = {
            "Linear + noise": "strict_linear_iid_noise",
            "Bounded saturation": "bounded_separable_saturation",
            "Sequence/marginal matched": "sequence_marginal_matched_interaction_permutation",
        }
        if any(
            set(panel_df.loc[panel_df["condition"].astype(str).eq(condition), "null_model"].astype(str)) != {null_model}
            for condition, null_model in expected_null_models.items()
        ):
            raise RuntimeError("Fixed S2C condition/null-model identity mismatch.")
        if set(panel_df["aggregation_level"].astype(str)) != {"network_null_mean"}:
            raise RuntimeError("Fixed S2C includes an undeclared aggregation level.")
        if set(pd.to_numeric(panel_df["n_null_replicates"], errors="raise").astype(int)) != {100}:
            raise RuntimeError("Fixed S2C null-replicate identity mismatch.")

    stats_payload = json.loads(paths["stats"].read_text(encoding="utf-8"))
    if stats_payload.get("figure_id") != "supp_fig_s2" or stats_payload.get("panel_id") != panel_id:
        raise RuntimeError(f"Fixed S2{panel_id} statistics identity mismatch.")
    if stats_payload.get("n_networks") != 20 or stats_payload.get("n_networks_observed") != 20:
        raise RuntimeError(f"Fixed S2{panel_id} statistics network count mismatch.")
    summaries = list(stats_payload.get("summaries") or [])
    summary_key = str(contract["summary_key"])
    summary_by_condition = {str(item.get(summary_key)): item for item in summaries}
    if len(summary_by_condition) != len(summaries) or set(summary_by_condition) != set(condition_order):
        raise RuntimeError(f"Fixed S2{panel_id} statistics summaries are missing, duplicate, or extra.")
    for condition in condition_order:
        persisted_values = panel_df.loc[
            panel_df["condition"].astype(str).eq(condition), "value"
        ].astype(float).tolist()
        frozen_values = list(summary_by_condition[condition].get("values_used_for_plotting") or [])
        if persisted_values != frozen_values:
            raise RuntimeError(f"Fixed S2{panel_id}/{condition} row/value identity mismatch against frozen statistics.")
        if summary_by_condition[condition].get("n") != 20:
            raise RuntimeError(f"Fixed S2{panel_id}/{condition} frozen summary n mismatch.")
    if panel_df["value"].astype(float).tolist() != list(stats_payload.get("values_used_for_plotting") or []):
        raise RuntimeError(f"Fixed S2{panel_id} global value order mismatch against frozen statistics.")
    if panel_id == "C":
        _validate_fixed_s2c_statistics(stats_payload)

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("figure_id") != "supp_fig_s2" or manifest.get("panel_id") != panel_id:
        raise RuntimeError(f"Fixed S2{panel_id} source-manifest identity mismatch.")
    if manifest.get("status") != "ok" or manifest.get("n_networks") != 20:
        raise RuntimeError(f"Fixed S2{panel_id} source manifest is not an exact 20-network ok payload.")
    if list(manifest.get("warnings") or []):
        raise RuntimeError(f"Fixed S2{panel_id} source manifest contains fallback/warning paths.")
    sources = list(manifest.get("sources") or [])
    if len(sources) not in ({20} if panel_id == "A" else {40}):
        raise RuntimeError(f"Fixed S2{panel_id} source manifest has missing or extra source entries.")
    if any(not bool(source.get("exists")) for source in sources):
        raise RuntimeError(f"Fixed S2{panel_id} source manifest contains a missing source.")
    if len({str(source.get("path")) for source in sources}) != len(sources):
        raise RuntimeError(f"Fixed S2{panel_id} source manifest contains duplicate source entries.")

    manifest["fixed_persisted_input_gate"] = {
        "status": "pass",
        "panel_data_sha256": contract["panel_data_sha256"],
        "stats_sha256": contract["stats_sha256"],
        "source_manifest_sha256": contract["manifest_sha256"],
        "row_count": 60,
        "network_count": 20,
        "condition_order": list(condition_order),
        "scientific_recomputation": False,
        "fallback": False,
    }
    return write_adapter_outputs(
        output_dir,
        figure_id,
        panel_id,
        panel_df,
        stats_payload,
        manifest,
        [],
    )


def _validate_fixed_s2c_statistics(stats_payload: Mapping[str, Any]) -> None:
    if stats_payload.get("condition_order") != [
        "Linear + noise",
        "Bounded saturation",
        "Sequence/marginal matched",
    ]:
        raise RuntimeError("Fixed S2C statistics condition order mismatch.")
    if stats_payload.get("calibration_gate_passed") is not True:
        raise RuntimeError("Fixed S2C calibration gate is not the frozen passing value.")
    calibration = dict(stats_payload.get("calibration_summary") or {})
    expected_models = {
        "strict_linear_iid_noise",
        "bounded_separable_saturation",
        "sequence_marginal_matched_interaction_permutation",
    }
    if set(calibration) != expected_models:
        raise RuntimeError("Fixed S2C calibration summaries are missing, duplicate, or extra.")
    for key in ("strict_linear_iid_noise", "bounded_separable_saturation"):
        payload = dict(calibration[key])
        if payload.get("false_positive_count_one_sided_alpha_0_05") != 0:
            raise RuntimeError(f"Fixed S2C {key} false-positive count mismatch.")
        if payload.get("n_dataset_replicates") != 100:
            raise RuntimeError(f"Fixed S2C {key} calibration replicate count mismatch.")
        if payload.get("false_positive_rate_exact_95_ci") != [0.0, 0.03621669264517641]:
            raise RuntimeError(f"Fixed S2C {key} exact confidence interval mismatch.")
    matched = dict(calibration["sequence_marginal_matched_interaction_permutation"])
    if matched.get("empirical_p_observed_vs_null") != 0.009900990099009901:
        raise RuntimeError("Fixed S2C empirical P mismatch.")
    if stats_payload.get("dataset_null_replicates_used_for_calibration") != 100:
        raise RuntimeError("Fixed S2C dataset null-replicate count mismatch.")
    if stats_payload.get("network_null_replicate_rows_used_for_calibration") != 6000:
        raise RuntimeError("Fixed S2C source-row count mismatch.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
