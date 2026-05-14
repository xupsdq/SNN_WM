from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.plotting.paper_fig.data_resolver import (
    AdapterResult,
    first_existing_path,
    missing_adapter_result,
    summarize_values,
    write_adapter_outputs,
)


GROUP_LABELS = {
    "multi_recent": "Multi-recent",
    "single_recent": "Single-recent",
    "multi_old": "Multi-old",
    "single_old": "Single-old",
}

MANIPULATION_LABELS = {
    "peak_flattened": "Peak-flattened",
    "intact_final": "Intact-final",
    "peak_boosted": "Peak-boosted",
}

_COMMON_SOURCE_FIELDS = ["seed", "network_seed", "network_index", "model_path", "eval_seed", "run_dir"]


def build_fig6_anchor_peak_linkage(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.6A anchor-peak linkage data when direct columns are available."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.6A anchor-peak linkage source not found.")
    df = pd.read_csv(path)
    x_candidates = ["support_loss_in_final_peak_region", "final_peak_support_loss", "peak_support_loss"]
    y_candidates = ["anchor_retreat", "anchor_retreat_position", "anchor_shift_retreat"]
    x_col = next((col for col in x_candidates if col in df.columns), None)
    y_col = next((col for col in y_candidates if col in df.columns), None)
    if x_col is None or y_col is None:
        return missing_adapter_result(
            spec,
            repo_root,
            output_dir,
            f"Fig.6A direct support-loss/anchor-retreat columns unavailable in {path}; checked x={x_candidates}, y={y_candidates}.",
        )
    rows = []
    for _, r in df.dropna(subset=[x_col, y_col]).iterrows():
        out = _row(figure_id, panel_id, "anchor_retreat", "Final peak support loss", "Layer 1/3", r.get("network_index", ""), r.get("network_seed", ""), float(r[y_col]), "position", path, repo_root)
        out["support_loss_in_final_peak_region"] = float(r[x_col])
        out["anchor_retreat"] = float(r[y_col])
        out["x_value"] = float(r[x_col])
        out["y_value"] = float(r[y_col])
        out["sequence_id"] = r.get("trial_id", "")
        out["peak_region_id"] = r.get("peak_region_id", "")
        rows.append(out)
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, [], correlations=_correlations_from_rows(rows))


def build_fig6_peak_membership(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.6B peak fraction by update-history group."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.6B peak-membership source not found.")
    df = pd.read_csv(path)
    raw_rows = int(len(df))
    if not {"group_name", "peak_fraction_in_group"}.issubset(df.columns):
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.6B source missing group_name/peak_fraction_in_group: {path}")
    use, warnings = _preferred_group_rows(df, panel_id)
    if "network_seed" not in use.columns:
        warnings.append("Fig.6B used single-run fallback; n=20 network summary unavailable.")
        use["network_seed"] = ""
        use["network_index"] = ""
    use = use.dropna(subset=["group_name", "peak_fraction_in_group"]).copy()
    granularity, granularity_warnings = _granularity_report(panel_id, path, df, use, row_id_cols=_row_identifier_columns(panel_id))
    warnings.extend(granularity_warnings)
    rows = []
    for source_row_id, r in use.reset_index(drop=True).iterrows():
        condition = GROUP_LABELS.get(str(r["group_name"]), str(r["group_name"]))
        out = _row(figure_id, panel_id, "peak_fraction", condition, "Layer 1", r.get("network_index", ""), r.get("network_seed", ""), float(r["peak_fraction_in_group"]), "fraction", path, repo_root)
        _copy_available_fields(out, r, _COMMON_SOURCE_FIELDS + ["group_by", "recent_definition", "group_name", "trial_id", "sequence_id", "peak_region_id"])
        out["source_row_id"] = int(source_row_id)
        out["update_history_group"] = condition
        out["peak_fraction"] = float(r["peak_fraction_in_group"])
        out["peak_membership"] = float(r["peak_fraction_in_group"])
        out["update_count"] = float(r.get("mean_update_count", np.nan))
        out["recency_group"] = "Recent" if "recent" in condition else "Old"
        out["source_group_name"] = str(r["group_name"])
        rows.append(out)
    granularity.update({"raw_rows_read": raw_rows, "rows_after_source_filtering": int(len(use)), "rows_written_to_panel_data": int(len(rows))})
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, warnings, granularity=granularity)


def build_fig6_repetition_recency_gain(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.6C final STSP gain by repetition x recency group."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.6C repetition-recency gain source not found.")
    df = pd.read_csv(path)
    raw_rows = int(len(df))
    if not {"group_name", "mean_final_g"}.issubset(df.columns):
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.6C source missing group_name/mean_final_g: {path}")
    use, warnings = _preferred_group_rows(df, panel_id)
    if "network_seed" not in use.columns:
        warnings.append("Fig.6C used single-run fallback; n=20 network summary unavailable.")
        use["network_seed"] = ""
        use["network_index"] = ""
    use = use.dropna(subset=["group_name", "mean_final_g"]).copy()
    granularity, granularity_warnings = _granularity_report(panel_id, path, df, use, row_id_cols=_row_identifier_columns(panel_id))
    warnings.extend(granularity_warnings)
    rows = []
    for source_row_id, r in use.reset_index(drop=True).iterrows():
        condition = GROUP_LABELS.get(str(r["group_name"]), str(r["group_name"]))
        out = _row(figure_id, panel_id, "final_stsp_gain", condition, "Layer 1", r.get("network_index", ""), r.get("network_seed", ""), float(r["mean_final_g"]), "gain", path, repo_root)
        _copy_available_fields(out, r, _COMMON_SOURCE_FIELDS + ["group_by", "recent_definition", "group_name", "trial_id", "sequence_id", "peak_region_id"])
        out["source_row_id"] = int(source_row_id)
        out["repetition"] = "Multi" if condition.startswith("Multi") else "Single"
        out["recency"] = "Recent" if condition.endswith("recent") else "Old"
        out["update_history_group"] = condition
        out["final_stsp_gain"] = float(r["mean_final_g"])
        out["update_count"] = float(r.get("mean_update_count", np.nan))
        out["source_group_name"] = str(r["group_name"])
        rows.append(out)
    granularity.update({"raw_rows_read": raw_rows, "rows_after_source_filtering": int(len(use)), "rows_written_to_panel_data": int(len(rows))})
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, warnings, granularity=granularity)


def build_fig6_model_comparison(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.6D paired overlap-only versus update+recency R2 model comparison."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.6D model-comparison source not found.")
    df = pd.read_csv(path)
    raw_rows = int(len(df))
    if not {"r2_overlap_only", "r2_update_plus_recency"}.issubset(df.columns):
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.6D source missing R2 columns: {path}")
    warnings: list[str] = []
    if "network_seed" not in df.columns:
        warnings.append("Fig.6D used single-run fallback; n=20 network summary unavailable.")
        df["network_seed"] = ""
        df["network_index"] = ""
    use = df.copy()
    if "valid" in use.columns:
        valid = pd.to_numeric(use["valid"], errors="coerce").fillna(0).astype(int).eq(1)
        if valid.any():
            use = use[valid].copy()
    use = use.dropna(subset=["r2_overlap_only", "r2_update_plus_recency"]).copy()
    granularity, granularity_warnings = _granularity_report(panel_id, path, df, use, row_id_cols=_row_identifier_columns(panel_id))
    warnings.extend(granularity_warnings)
    rows = []
    delta_values: list[float] = []
    for source_row_id, r in use.reset_index(drop=True).iterrows():
        delta = float(r["r2_update_plus_recency"] - r["r2_overlap_only"])
        delta_values.append(delta)
        for condition, col in (("Overlap-only", "r2_overlap_only"), ("Update + recency", "r2_update_plus_recency")):
            out = _row(figure_id, panel_id, "prediction_r2", condition, "Layer 1", r.get("network_index", ""), r.get("network_seed", ""), float(r[col]), "r2", path, repo_root)
            _copy_available_fields(out, r, _COMMON_SOURCE_FIELDS + ["trial_id", "sequence_id", "seq_len", "fold_id", "split_id", "valid"])
            out["source_row_id"] = int(source_row_id)
            out["predictor_model"] = condition
            out["prediction_r2"] = float(r[col])
            out["delta_r2"] = delta
            out["source_model_column"] = col
            rows.append(out)
    stats_extra = {
        "model_comparison": {"delta_r2_mean": float(np.nanmean(delta_values)) if delta_values else None},
        "delta_r2": [float(v) for v in delta_values],
    }
    granularity.update({"raw_rows_read": raw_rows, "rows_after_source_filtering": int(len(use)), "rows_written_to_panel_data": int(len(rows))})
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, warnings, extra_stats=stats_extra, granularity=granularity)


def build_fig6_peak_manipulation(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.6E peak manipulation effect on peak-associated spike enrichment."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.6E peak-manipulation source not found.")
    raw_df = pd.read_csv(path)
    raw_rows = int(len(raw_df))
    if not {"condition", "spike_enrichment"}.issubset(raw_df.columns):
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.6E source missing condition/spike_enrichment: {path}")
    warnings: list[str] = []
    df = raw_df.copy()
    if "valid" in df.columns:
        df = df[pd.to_numeric(df["valid"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    if "network_seed" not in df.columns:
        warnings.append("Fig.6E used single-run fallback; n=20 network summary unavailable.")
        df["network_seed"] = ""
        df["network_index"] = ""
    boosted = df[df["condition"].astype(str).eq("peak_boosted")].copy()
    max_boost = pd.to_numeric(boosted.get("boost_level", pd.Series(dtype=float)), errors="coerce").max() if not boosted.empty else np.nan
    if pd.notna(max_boost):
        keep = df["condition"].astype(str).isin(["peak_flattened", "intact_final"]) | (df["condition"].astype(str).eq("peak_boosted") & pd.to_numeric(df.get("boost_level", 0), errors="coerce").eq(max_boost))
        df = df[keep].copy()
    use = df.dropna(subset=["condition", "spike_enrichment"]).copy()
    granularity, granularity_warnings = _granularity_report(panel_id, path, raw_df, use, row_id_cols=_row_identifier_columns(panel_id))
    warnings.extend(granularity_warnings)
    rows = []
    for source_row_id, r in use.reset_index(drop=True).iterrows():
        raw = str(r["condition"])
        if raw not in MANIPULATION_LABELS:
            continue
        condition = MANIPULATION_LABELS[raw]
        out = _row(figure_id, panel_id, "peak_associated_spike_enrichment", condition, "Layer 1", r.get("network_index", ""), r.get("network_seed", ""), float(r["spike_enrichment"]), "enrichment", path, repo_root)
        _copy_available_fields(out, r, _COMMON_SOURCE_FIELDS + ["trial_id", "sequence_id", "seq_len", "probe_image_id", "probe_label", "probe_group", "target_region", "overlap_level", "boost_level", "valid"])
        out["source_row_id"] = int(source_row_id)
        out["peak_manipulation"] = condition
        out["peak_associated_spike_enrichment"] = float(r["spike_enrichment"])
        out["probe_group"] = r.get("probe_group", "")
        out["overlap_group"] = r.get("overlap_level", "")
        out["source_condition"] = raw
        out["boost_level"] = r.get("boost_level", "")
        rows.append(out)
    granularity.update({"raw_rows_read": raw_rows, "rows_after_source_filtering": int(len(use)), "rows_written_to_panel_data": int(len(rows))})
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, warnings, granularity=granularity, manifest_notes={"max_boost_level_used": None if pd.isna(max_boost) else float(max_boost)})


def build_fig6_probe_peak_overlap_dependency(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.6F probe-peak overlap versus intact-over-flattened benefit scatter data."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.6F probe-peak overlap dependency source not found.")
    raw_df = pd.read_csv(path)
    raw_rows = int(len(raw_df))
    if not {"input_peak_overlap_fraction", "delta_spike_enrichment_intact_vs_flattened"}.issubset(raw_df.columns):
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.6F source missing overlap/benefit columns: {path}")
    warnings: list[str] = []
    df = raw_df.copy()
    if "boost_level" in df.columns:
        df = df[pd.to_numeric(df["boost_level"], errors="coerce").fillna(0).eq(0)].copy()
    if "network_seed" not in df.columns:
        warnings.append("Fig.6F used single-run fallback; n=20 network summary unavailable.")
        df["network_seed"] = ""
        df["network_index"] = ""
    use = df.dropna(subset=["input_peak_overlap_fraction", "delta_spike_enrichment_intact_vs_flattened"]).copy()
    granularity, granularity_warnings = _granularity_report(panel_id, path, raw_df, use, row_id_cols=_row_identifier_columns(panel_id))
    warnings.extend(granularity_warnings)
    rows = []
    for source_row_id, r in use.reset_index(drop=True).iterrows():
        x = float(r["input_peak_overlap_fraction"])
        y = float(r["delta_spike_enrichment_intact_vs_flattened"])
        out = _row(figure_id, panel_id, "intact_over_flattened_benefit", "Intact over peak-flattened", "Layer 1", r.get("network_index", ""), r.get("network_seed", ""), y, "enrichment", path, repo_root)
        _copy_available_fields(out, r, _COMMON_SOURCE_FIELDS + ["trial_id", "sequence_id", "seq_len", "probe_image_id", "probe_label", "probe_group", "target_region", "overlap_level", "boost_level"])
        out["source_row_id"] = int(source_row_id)
        out["probe_peak_overlap"] = x
        out["intact_over_flattened_benefit"] = y
        out["spike_enrichment_benefit"] = y
        out["x_value"] = x
        out["y_value"] = y
        out["overlap_unit"] = "fraction"
        rows.append(out)
    granularity.update({"raw_rows_read": raw_rows, "rows_after_source_filtering": int(len(use)), "rows_written_to_panel_data": int(len(rows))})
    quadratic_fit = _quadratic_fit_from_rows(rows)
    return _write_result(
        output_dir,
        figure_id,
        panel_id,
        pd.DataFrame(rows),
        [path],
        checked,
        warnings,
        extra_stats={"quadratic_fit": quadratic_fit, "regression_summary": quadratic_fit},
        granularity=granularity,
    )


def _write_result(
    output_dir: Path,
    figure_id: str,
    panel_id: str,
    panel_df: pd.DataFrame,
    source_paths: list[Path],
    checked: list[str],
    warnings: list[str],
    *,
    correlations: dict[str, Any] | None = None,
    extra_stats: dict[str, Any] | None = None,
    manifest_notes: dict[str, Any] | None = None,
    granularity: dict[str, Any] | None = None,
) -> AdapterResult:
    metric = str(panel_df["metric"].iloc[0]) if not panel_df.empty and "metric" in panel_df.columns else ""
    group_cols = [c for c in ("metric", "condition", "layer") if c in panel_df.columns]
    stats: dict[str, Any] = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "summaries": summarize_values(panel_df, group_cols),
        "values_used_for_plotting": _values(panel_df),
        "n_networks": _n_networks(panel_df),
    }
    if granularity:
        stats.update(granularity)
        stats["data_granularity"] = dict(granularity)
    if correlations is not None:
        stats["correlations"] = correlations
        stats["regression_summary"] = correlations.get("linear_regression", {})
    if extra_stats:
        stats.update(extra_stats)
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok" if not panel_df.empty else "missing_source",
        "sources": [{"path": _rel(path, Path.cwd()), "exists": path.exists()} for path in source_paths],
        "checked_candidates": checked,
        "label_mappings": {"update_history_groups": GROUP_LABELS, "peak_manipulations": MANIPULATION_LABELS},
    }
    if granularity:
        manifest["data_granularity"] = dict(granularity)
    if manifest_notes:
        manifest.update(manifest_notes)
    if panel_df.empty:
        warnings.append(f"Fig.6{panel_id} adapter produced no plottable rows.")
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def _preferred_group_rows(df: pd.DataFrame, panel_id: str) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    use = df.copy()
    if "group_by" in use.columns:
        update = use[use["group_by"].astype(str).eq("update_count")].copy()
        if not update.empty:
            use = update
        else:
            warnings.append(f"Fig.6{panel_id} did not find group_by=update_count; using all group_by values.")
    if "recent_definition" in use.columns:
        recent = use[use["recent_definition"].astype(str).eq("recent_update")].copy()
        if not recent.empty:
            use = recent
        else:
            warnings.append(f"Fig.6{panel_id} did not find recent_definition=recent_update; using all recency definitions.")
    return use, warnings


def _candidate_files(spec: Mapping[str, Any]) -> list[str]:
    mapping = spec.get("source_mapping") or {}
    candidates = list(mapping.get("preferred_files") or []) + list(mapping.get("fallback_files") or [])
    expanded: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        values = [str(candidate)]
        if "__by_network" in str(candidate):
            values.insert(0, str(candidate).replace("__by_network", ""))
        for value in values:
            if value not in seen:
                expanded.append(value)
                seen.add(value)
    return expanded


def _granularity_report(
    panel_id: str,
    path: Path,
    raw_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    *,
    row_id_cols: list[str],
) -> tuple[dict[str, Any], list[str]]:
    source_appeared_preaggregated = _source_appeared_preaggregated(raw_df, path, row_id_cols)
    row_level_data_preserved = not source_appeared_preaggregated and not filtered_df.empty
    warnings: list[str] = []
    if source_appeared_preaggregated:
        warnings.append(f"Fig.6{panel_id} source appears pre-aggregated; row-level data unavailable.")
    available_row_ids = [col for col in row_id_cols if col in raw_df.columns]
    return (
        {
            "source_files_used": 1,
            "n_source_files_used": 1,
            "raw_rows_read": int(len(raw_df)),
            "rows_after_source_filtering": int(len(filtered_df)),
            "rows_written_to_panel_data": 0,
            "adapter_performed_network_level_averaging": False,
            "source_appeared_preaggregated": bool(source_appeared_preaggregated),
            "row_level_data_preserved": bool(row_level_data_preserved),
            "row_identifier_columns_available": available_row_ids,
            "source_file_used": _rel(path, Path.cwd()),
        },
        warnings,
    )


def _source_appeared_preaggregated(df: pd.DataFrame, path: Path, row_id_cols: list[str]) -> bool:
    row_cols = [col for col in row_id_cols if col in df.columns]
    if row_cols:
        return False
    if "__by_network" in path.stem:
        return True
    if "network_seed" in df.columns and "network_index" in df.columns:
        non_network_cols = [col for col in df.columns if col not in {"network_seed", "network_index", "seed", "model_path", "eval_seed", "run_dir"}]
        return len(non_network_cols) <= 4
    return False


def _row_identifier_columns(panel_id: str) -> list[str]:
    common = ["row_id", "trial_id", "sequence_id", "seq_len", "peak_region_id", "fold_id", "split_id"]
    if panel_id in {"E", "F"}:
        return common + ["probe_image_id", "probe_label", "probe_group", "target_region", "overlap_level", "boost_level"]
    return common + ["group_by", "recent_definition", "group_name"]


def _copy_available_fields(out: dict[str, Any], row: pd.Series, fields: list[str]) -> None:
    for field in fields:
        if field in row.index:
            out[field] = row.get(field, "")


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig6")), str(spec.get("panel_id", "")).upper()


def _rel(path: Path | str, repo_root: Path) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.relative_to(repo_root))
    except ValueError:
        return str(path_obj)


def _row(figure_id: str, panel_id: str, metric: str, condition: str, layer: str, network_id: Any, seed_id: Any, value: float, unit: str, source_file: Path | str, repo_root: Path) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "condition": condition,
        "layer": layer,
        "network_id": network_id,
        "seed_id": seed_id,
        "value": value,
        "unit": unit,
        "source_file": _rel(source_file, repo_root),
    }


def _values(df: pd.DataFrame) -> list[float]:
    if "value" not in df.columns:
        return []
    return [float(v) for v in pd.to_numeric(df["value"], errors="coerce").dropna().tolist()]


def _n_networks(df: pd.DataFrame) -> int:
    for col in ("seed_id", "network_id"):
        if col in df.columns:
            return int(df[col].replace("", pd.NA).dropna().nunique())
    return 0


def _correlations_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty or not {"x_value", "y_value"}.issubset(df.columns):
        return {"n": 0, "pearson_r": None, "spearman_rho": None, "linear_regression": {}}
    data = pd.DataFrame({"x": pd.to_numeric(df["x_value"], errors="coerce"), "y": pd.to_numeric(df["y_value"], errors="coerce")}).dropna()
    if len(data) < 3:
        return {"n": int(len(data)), "pearson_r": None, "spearman_rho": None, "linear_regression": {}}
    slope, intercept = np.polyfit(data["x"].to_numpy(dtype=float), data["y"].to_numpy(dtype=float), 1)
    return {
        "n": int(len(data)),
        "pearson_r": float(data["x"].corr(data["y"], method="pearson")),
        "spearman_rho": float(data["x"].corr(data["y"], method="spearman")),
        "linear_regression": {"slope": float(slope), "intercept": float(intercept)},
    }


def _quadratic_fit_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty or not {"x_value", "y_value"}.issubset(df.columns):
        return {"fit": "quadratic", "n": 0, "coefficients": [], "r2": None}
    data = pd.DataFrame({"x": pd.to_numeric(df["x_value"], errors="coerce"), "y": pd.to_numeric(df["y_value"], errors="coerce")}).dropna()
    if len(data) < 3 or data["x"].nunique() < 3:
        return {"fit": "quadratic", "n": int(len(data)), "coefficients": [], "r2": None}
    x = data["x"].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=float)
    coefficients = np.polyfit(x, y, 2)
    y_hat = np.polyval(coefficients, x)
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if total <= 0 else float(1.0 - np.sum((y - y_hat) ** 2) / total)
    return {
        "fit": "quadratic",
        "n": int(len(data)),
        "coefficients": [float(v) for v in coefficients],
        "r2": r2,
    }
