from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.plotting.paper_fig.data_resolver import (
    AdapterResult,
    first_existing_path,
    missing_adapter_result,
    summarize_values,
    write_adapter_outputs,
)


def build_fig1_overall_recall(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build canonical Fig.1B overall recall data from ensemble summaries."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1B overall recall source not found.")

    warnings: list[str] = ["Class-specific recall is not available in current source files; plotting overall recall only."]
    df = pd.read_csv(path)
    if "final_accuracy" not in df.columns:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.1B source lacks final_accuracy: {path}")
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("success")].copy()
    seed_col = "seed" if "seed" in df.columns else "index"
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "metric": "overall_recall",
                "condition": "STSP-SNN",
                "layer": "",
                "network_id": row.get("index", row.get(seed_col, "")),
                "seed_id": row.get(seed_col, ""),
                "value": float(row["final_accuracy"]) * 100.0,
                "unit": "percent",
                "source_file": str(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path),
            }
        )
    panel_df = pd.DataFrame(rows)
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": "overall_recall",
        "metric_name": "Overall recall (%)",
        "condition_labels": ["STSP-SNN"],
        "reference_value": 10,
        "summaries": summarize_values(panel_df, ["condition"]),
        "n_networks_observed": _panel_n(panel_df),
        "values_used_for_plotting": _values(panel_df),
    }
    source_manifest = _source_manifest(figure_id, panel_id, path, checked, "ok", repo_root)
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, source_manifest, warnings)


def build_fig1_delay_decoding(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build canonical Fig.1C delay decoding data by layer."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1C delay decoding source not found.")

    df = pd.read_csv(path)
    required = {"layer", "delay_ms", "acc"}
    missing = required.difference(df.columns)
    if missing:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.1C source missing columns {sorted(missing)}: {path}")

    layer_labels = {"layer1": "Layer 1", "layer2": "Layer 2", "layer3": "Layer 3"}
    id_cols = [col for col in ("network_seed", "network_index") if col in df.columns]
    grouped = df.groupby(id_cols + ["layer"], dropna=False, as_index=False)["acc"].mean() if id_cols else df.groupby(["layer"], dropna=False, as_index=False)["acc"].mean()
    rows = []
    for _, row in grouped.iterrows():
        layer = str(row["layer"])
        acc = float(row["acc"])
        rows.append(
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "metric": "decoding_accuracy",
                "condition": layer_labels.get(layer, layer),
                "layer": layer_labels.get(layer, layer),
                "network_id": row.get("network_index", ""),
                "seed_id": row.get("network_seed", ""),
                "value": acc * 100.0,
                "unit": "percent",
                "source_file": str(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path),
                "source_layer": layer,
                "aggregation": "mean_across_delay_ms",
                "acc": acc,
            }
        )
    panel_df = pd.DataFrame(rows)
    warnings: list[str] = ["Fig.1C collapsed delay_ms values to one network-level mean per layer; no timecourse plotted."]
    if "network_seed" not in df.columns:
        warnings.append("Fig.1C used single-run delay decoding data; n=20 network QC will warn.")
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": "decoding_accuracy",
        "metric_name": "Decoding accuracy from u/x state (%)",
        "layer_labels": ["Layer 1", "Layer 2", "Layer 3"],
        "reference_value": 10,
        "summaries": summarize_values(panel_df, ["layer"]),
        "n_networks_observed": _panel_n(panel_df),
        "source_delay_ms_values": [float(v) for v in sorted(pd.to_numeric(df["delay_ms"], errors="coerce").dropna().unique())],
        "aggregation": "mean_across_delay_ms_by_network_and_layer",
        "values_used_for_plotting": _values(panel_df),
    }
    source_manifest = _source_manifest(figure_id, panel_id, path, checked, "ok", repo_root)
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, source_manifest, warnings)


def build_fig1_stsp_outcome_profile(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build canonical Fig.1D error-rate data."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1D outcome profile source not found.")

    df = pd.read_csv(path)
    required = {"condition", "error_rate"}
    missing = required.difference(df.columns)
    if missing:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.1D source lacks unambiguous error_rate column: {path}")

    condition_map = {
        "A_dynamic_base": "Dynamic STSP",
        "D_trial_shuffle_ux": "u/x-shuffled",
        "E_static_frozen": "Static-frozen",
    }
    condition_order = ["Dynamic STSP", "u/x-shuffled", "Static-frozen"]
    rows = []
    for _, row in df.iterrows():
        condition = condition_map.get(str(row["condition"]))
        if not condition:
            continue
        rows.append(
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "metric": "error_rate",
                "condition": condition,
                "layer": "",
                "network_id": row.get("network_index", ""),
                "seed_id": row.get("network_seed", ""),
                "value": float(row["error_rate"]),
                "unit": "percent",
                "source_file": str(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path),
                "condition_order": condition_order.index(condition),
            }
        )
    panel_df = pd.DataFrame(rows)
    if not panel_df.empty:
        panel_df = panel_df.sort_values(["condition_order", "seed_id", "network_id"], kind="stable").reset_index(drop=True)
    warnings: list[str] = []
    if "network_seed" not in df.columns:
        warnings.append("Fig.1D used single-run condition summary data; n=20 network QC will warn.")
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": "error_rate",
        "metric_name": "Error rate (%)",
        "condition_labels": condition_order,
        "condition_order": condition_order,
        "summaries": summarize_values(panel_df, ["condition"]),
        "n_networks_observed": _panel_n(panel_df),
        "values_used_for_plotting": _values(panel_df),
    }
    source_manifest = _source_manifest(figure_id, panel_id, path, checked, "ok", repo_root)
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, source_manifest, warnings)


def build_fig1_attribution_transfer(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build canonical Fig.1E error-conditional attribution composition."""
    figure_id, panel_id = _ids(spec)
    summary_path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if summary_path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1E condition-summary source not found.")

    trial_sources = _fig1e_trial_prediction_sources(spec, repo_root, summary_path)
    if not trial_sources:
        result = missing_adapter_result(
            spec,
            repo_root,
            output_dir,
            "Fig.1E raw trial_predictions.csv sources not found; refusing to reuse all-trial absolute-rate summary columns.",
        )
        result.source_manifest.setdefault("checked_candidates", [_display_path(Path(item), repo_root) for item in checked])
        return result

    condition_map = {
        "A_dynamic_base": "Dynamic baseline",
        "D_trial_shuffle_ux": "shuffle",
    }
    condition_order = ["Dynamic baseline", "shuffle"]
    traces = ["Original", "Donor", "Others"]
    rows = []
    component_sum_checks: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    donor_derivation_labels: set[str] = set()
    network_meta = _fig1e_network_meta(summary_path, repo_root)
    for source_index, trial_path in enumerate(trial_sources):
        df = pd.read_csv(trial_path)
        required = {"condition", "prediction_probe", "probe_label", "sample_label", "donor_sample_label"}
        missing = required.difference(df.columns)
        if missing:
            return missing_adapter_result(spec, repo_root, output_dir, f"Fig.1E raw trial source missing columns {sorted(missing)}: {trial_path}")
        seed_id, network_id = _fig1e_source_identity(trial_path, source_index, network_meta)
        source_label = _display_path(trial_path, repo_root)
        pred = pd.to_numeric(df["prediction_probe"], errors="coerce")
        probe = pd.to_numeric(df["probe_label"], errors="coerce")
        sample = pd.to_numeric(df["sample_label"], errors="coerce")
        donor = pd.to_numeric(df["donor_sample_label"], errors="coerce")
        correct_from_labels = pred.eq(probe)
        if "is_correct_probe" in df.columns:
            saved_correct = pd.to_numeric(df["is_correct_probe"], errors="coerce").fillna(0).astype(int).eq(1)
            mismatch = int(saved_correct.ne(correct_from_labels).sum())
            if mismatch:
                warnings.append(f"Fig.1E {source_label}: {mismatch} rows disagree between is_correct_probe and prediction_probe == probe_label.")
        if "donor_is_distinct" in df.columns:
            donor_distinct = pd.to_numeric(df["donor_is_distinct"], errors="coerce").fillna(0).astype(int).eq(1)
            donor_derivation = "trial-level donor_sample_label with donor_is_distinct=1 (change-under-B-map equivalent)"
        elif "pred_is_donor_shifted_memory" in df.columns:
            donor_distinct = None
            donor_derivation = "trial-level pred_is_donor_shifted_memory fallback"
        else:
            return missing_adapter_result(
                spec,
                repo_root,
                output_dir,
                f"Fig.1E raw trial source lacks donor_is_distinct or pred_is_donor_shifted_memory for change-target decomposition: {trial_path}",
            )
        donor_derivation_labels.add(donor_derivation)

        for raw_condition, display_condition in condition_map.items():
            sub_idx = df["condition"].astype(str).eq(raw_condition)
            sub = df.loc[sub_idx].copy()
            if sub.empty:
                warnings.append(f"Fig.1E {source_label}: condition {raw_condition} not found.")
                continue
            sub_pred = pred.loc[sub_idx]
            sub_probe = probe.loc[sub_idx]
            sub_sample = sample.loc[sub_idx]
            sub_donor = donor.loc[sub_idx]
            error_mask = sub_pred.ne(sub_probe)
            error_total = int(error_mask.sum())
            total_trials = int(sub.shape[0])
            if error_total <= 0:
                original_count = 0
                donor_count = 0
                others_count = 0
                fractions = {"Original": 0.0, "Donor": 0.0, "Others": 0.0}
                warnings.append(f"Fig.1E {source_label}: {display_condition} has zero error trials; component fractions set to 0.")
            else:
                original_count = int((error_mask & sub_pred.eq(sub_sample)).sum())
                if donor_distinct is None:
                    shifted = pd.to_numeric(df.loc[sub_idx, "pred_is_donor_shifted_memory"], errors="coerce").fillna(0).astype(int).eq(1)
                    donor_count = int((error_mask & shifted).sum())
                else:
                    donor_count = int((error_mask & sub_pred.eq(sub_donor) & donor_distinct.loc[sub_idx]).sum())
                others_count = int(error_total - original_count - donor_count)
                if others_count < 0:
                    warnings.append(f"Fig.1E {source_label}: negative Others count before clipping for {display_condition}; check overlapping labels.")
                    others_count = 0
                fractions = {
                    "Original": 100.0 * float(original_count) / float(error_total),
                    "Donor": 100.0 * float(donor_count) / float(error_total),
                    "Others": 100.0 * float(others_count) / float(error_total),
                }
            component_counts = {"Original": original_count, "Donor": donor_count, "Others": others_count}
            sum_percent = float(sum(fractions.values()))
            component_sum_checks.append(
                {
                    "condition": display_condition,
                    "network_id": network_id,
                    "seed_id": seed_id,
                    "sum_percent": sum_percent,
                    "deviation_from_100_percent": abs(sum_percent - 100.0) if error_total > 0 else 0.0,
                }
            )
            count_rows.append(
                {
                    "condition": display_condition,
                    "network_id": network_id,
                    "seed_id": seed_id,
                    "total_trials": total_trials,
                    "error_trials": error_total,
                    "original_error_trials": original_count,
                    "donor_error_trials": donor_count,
                    "others_error_trials": others_count,
                }
            )
            for trace in traces:
                rows.append(
                    {
                        "figure_id": figure_id,
                        "panel_id": panel_id,
                        "metric": "error_conditional_fraction",
                        "condition": display_condition,
                        "layer": "",
                        "network_id": network_id,
                        "seed_id": seed_id,
                        "value": fractions[trace],
                        "unit": "percent",
                        "source_file": source_label,
                        "trace": trace,
                        "component_count": component_counts[trace],
                        "error_total": error_total,
                        "total_trials": total_trials,
                        "denominator": "error_trials",
                        "error_definition": "prediction_probe != probe_label",
                    }
                )
    panel_df = pd.DataFrame(rows)
    if not panel_df.empty:
        panel_df["condition_order"] = panel_df["condition"].map({label: i for i, label in enumerate(condition_order)})
        panel_df["trace_order"] = panel_df["trace"].map({label: i for i, label in enumerate(traces)})
        panel_df = panel_df.sort_values(["condition_order", "seed_id", "network_id", "trace_order"], kind="stable").reset_index(drop=True)
    counts_df = pd.DataFrame(count_rows)
    count_summary = []
    if not counts_df.empty:
        for condition, part in counts_df.groupby("condition", sort=False):
            count_summary.append(
                {
                    "condition": str(condition),
                    "n_networks": int(part["seed_id"].replace("", pd.NA).dropna().nunique() or part["network_id"].replace("", pd.NA).dropna().nunique()),
                    "total_trials": int(part["total_trials"].sum()),
                    "total_error_trials": int(part["error_trials"].sum()),
                    "mean_error_trials_per_network": float(part["error_trials"].mean()),
                    "per_network_error_trials": [
                        {
                            "network_id": row["network_id"],
                            "seed_id": row["seed_id"],
                            "error_trials": int(row["error_trials"]),
                            "total_trials": int(row["total_trials"]),
                        }
                        for _, row in part.iterrows()
                    ],
                }
            )
    max_sum_deviation = max((float(item["deviation_from_100_percent"]) for item in component_sum_checks), default=0.0)
    if max_sum_deviation > 1e-6:
        warnings.append(f"Fig.1E error-composition fractions do not sum to 100% within tolerance; max deviation={max_sum_deviation:.6g} pp.")
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": "error_conditional_fraction",
        "metric_name": "Error composition within error trials (%)",
        "condition_labels": condition_order,
        "trace_labels": traces,
        "summaries": summarize_values(panel_df, ["condition", "trace"]),
        "n_networks_observed": _panel_n(panel_df),
        "values_used_for_plotting": _values(panel_df),
        "denominator": "error trials only",
        "error_definition": "prediction_probe != probe_label",
        "raw_source_used": [_display_path(path, repo_root) for path in trial_sources],
        "summary_source_traced_from": _display_path(summary_path, repo_root),
        "n_raw_source_files": len(trial_sources),
        "donor_derivation": sorted(donor_derivation_labels),
        "donor_derived_from_change_under_bmap_field": False,
        "donor_derivation_note": "Donor is computed from trial-level donor_sample_label with donor_is_distinct=1, matching the change-under-B-map definition without reusing all-trial summary rates.",
        "error_trial_counts": count_summary,
        "component_sum_checks": component_sum_checks,
        "max_component_sum_deviation_percent": max_sum_deviation,
    }
    source_manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok",
        "sources": [{"path": _display_path(path, repo_root), "exists": True, "kind": "trial_predictions"} for path in trial_sources],
        "traced_from_summary_source": {"path": _display_path(summary_path, repo_root), "exists": True},
        "checked_candidates": [_display_path(Path(item), repo_root) for item in checked],
        "derivation": {
            "denominator": "error trials only",
            "error_definition": "prediction_probe != probe_label",
            "components": ["Original", "Donor", "Others"],
            "others": "error_total - original_error_trials - donor_error_trials",
            "donor": "prediction_probe == donor_sample_label and donor_is_distinct == 1",
        },
    }
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, source_manifest, warnings)


def _fig1e_trial_prediction_sources(spec: Mapping[str, Any], repo_root: Path, summary_path: Path) -> list[Path]:
    mapping = spec.get("source_mapping") or {}
    sources: list[Path] = []
    for raw_glob in mapping.get("raw_trial_globs") or []:
        raw_pattern = str(raw_glob).replace("\\", "/")
        if Path(raw_pattern).is_absolute():
            absolute_pattern = Path(raw_pattern)
            sources.extend(sorted(absolute_pattern.parent.glob(absolute_pattern.name)))
        else:
            sources.extend(sorted(repo_root.glob(raw_pattern)))
    if not sources:
        for raw_file in mapping.get("fallback_raw_trial_files") or []:
            raw_path = repo_root / str(raw_file)
            if raw_path.exists():
                sources.append(raw_path)
    if not sources and summary_path.name == "metrics_condition_summary__by_network.csv":
        multi_root = summary_path.parents[1]
        sources.extend(sorted((multi_root / "runs").glob("seed_*/data/trial_predictions.csv")))
    if not sources and summary_path.name == "metrics_condition_summary.csv":
        candidate = summary_path.with_name("trial_predictions.csv")
        if candidate.exists():
            sources.append(candidate)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in sources:
        key = str(path.resolve())
        if path.exists() and key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def _fig1e_network_meta(summary_path: Path, repo_root: Path) -> dict[str, dict[str, str]]:
    _ = repo_root
    if summary_path.name != "metrics_condition_summary__by_network.csv":
        return {}
    network_runs = summary_path.parents[1] / "data" / "network_runs.csv"
    if not network_runs.exists():
        return {}
    df = pd.read_csv(network_runs)
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        seed = str(row.get("network_seed", ""))
        if seed:
            out[seed] = {
                "network_id": str(row.get("network_index", "")),
                "seed_id": seed,
            }
    return out


def _fig1e_source_identity(path: Path, source_index: int, network_meta: Mapping[str, Mapping[str, str]]) -> tuple[str, str]:
    run_name = path.parents[1].name if len(path.parents) > 1 else ""
    seed_id = run_name.replace("seed_", "") if run_name.startswith("seed_") else ""
    if seed_id and seed_id in network_meta:
        meta = network_meta[seed_id]
        return str(meta.get("seed_id", seed_id)), str(meta.get("network_id", source_index))
    return seed_id, str(source_index)


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig1")), str(spec.get("panel_id", "")).upper()


def _candidate_files(spec: Mapping[str, Any]) -> list[str]:
    mapping = spec.get("source_mapping") or {}
    return list(mapping.get("preferred_files") or []) + list(mapping.get("fallback_files") or [])


def _values(df: pd.DataFrame) -> list[float]:
    if "value" not in df.columns:
        return []
    return [float(v) for v in pd.to_numeric(df["value"], errors="coerce").dropna().tolist()]


def _panel_n(df: pd.DataFrame) -> int:
    for col in ("seed_id", "network_id"):
        if col in df.columns:
            return int(df[col].replace("", pd.NA).dropna().nunique())
    return 0


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _source_manifest(figure_id: str, panel_id: str, path: Path, checked: list[str], status: str, repo_root: Path) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": status,
        "sources": [{"path": _display_path(path, repo_root), "exists": True}],
        "checked_candidates": [_display_path(Path(item), repo_root) for item in checked],
    }
