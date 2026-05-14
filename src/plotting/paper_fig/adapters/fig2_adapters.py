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


STATE_LABELS = {
    "baseline": "No memory",
    "S_B": "Item 2 only",
    "S_AB": "Item 1->Item 2",
}

ROW_IDENTIFIER_COLUMNS = (
    "trial_id",
    "pair_id",
    "episode_id",
    "sample_id",
    "triplet_id",
    "network_seed",
    "network_index",
    "eval_seed",
)


def build_fig2_fusion_dual_score(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.2B row-level Layer 3 fusion dual score data."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2B fusion dual score source not found.")
    df = pd.read_csv(path)
    missing = {"layer", "fusion_dual_score"}.difference(df.columns)
    if missing:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.2B source missing columns {sorted(missing)}: {path}")

    l3 = _layer3(df)
    warnings: list[str] = []
    if "network_seed" not in l3.columns:
        warnings.append("Fig.2B source lacks network_seed; row-level values are preserved but n=20 network identity is unavailable.")
    source_preaggregated = _appears_preaggregated(l3, value_cols=["fusion_dual_score"])
    if source_preaggregated:
        warnings.append("Fig.2B source appears already pre-aggregated; adapter preserved available rows and performed no additional averaging.")
    panel_df = pd.DataFrame(
        [
            _row_from_source(
                figure_id,
                panel_id,
                "fusion_dual_score",
                "Layer 3",
                "Layer 3",
                float(row["fusion_dual_score"]),
                "score",
                path,
                repo_root,
                row,
                source_row_index=idx,
            )
            for idx, row in l3.iterrows()
            if pd.notna(row.get("fusion_dual_score"))
        ]
    )
    processing_stats = _processing_stats(
        source_paths=[path],
        raw_rows_read=len(df),
        layer3_rows=len(l3),
        rows_written=len(panel_df),
        averaging_performed=False,
        source_appeared_preaggregated=source_preaggregated,
    )
    return _write_result(output_dir, figure_id, panel_id, panel_df, path, checked, warnings, reference_value=0, processing_stats=processing_stats)


def build_fig2_pair_specificity(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.2C row-level paired true-vs-shuffled pair-specificity data."""
    figure_id, panel_id = _ids(spec)
    paths, checked = _source_paths(repo_root, spec)
    if not paths:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2C pair-specificity sources not found.")
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    raw_rows_read = 0
    layer3_rows = 0
    source_preaggregated_flags: list[bool] = []
    for idx, path in enumerate(paths):
        df = pd.read_csv(path)
        raw_rows_read += len(df)
        missing = {"layer", "true_pair_score", "shuffled_pair_score"}.difference(df.columns)
        if missing:
            warnings.append(f"Skipping {path}: missing columns {sorted(missing)}")
            continue
        l3 = _layer3(df)
        layer3_rows += len(l3)
        seed_id = _seed_from_path(path)
        network_index = idx if seed_id else ""
        source_preaggregated_flags.append(_appears_preaggregated(l3, value_cols=["true_pair_score", "shuffled_pair_score"]))
        if not seed_id:
            warnings.append(f"Fig.2C source lacks seed identity in path; preserving rows with blank seed_id for {path}.")
        for source_row_index, row in l3.iterrows():
            for condition, value_col in (("True pair", "true_pair_score"), ("Shuffled pair", "shuffled_pair_score")):
                if pd.isna(row.get(value_col)):
                    continue
                out = _row_from_source(
                    figure_id,
                    panel_id,
                    "pair_specificity_score",
                    condition,
                    "Layer 3",
                    float(row[value_col]),
                    "score",
                    path,
                    repo_root,
                    row,
                    fallback_network_id=network_index,
                    fallback_seed_id=seed_id,
                    source_row_index=source_row_index,
                )
                out["pair_type"] = condition
                rows.append(out)
    if len(paths) == 1:
        warnings.append("Fig.2C used single-file fallback; row-level values are preserved but n=20 network coverage may be unavailable.")
    source_preaggregated = any(source_preaggregated_flags)
    if source_preaggregated:
        warnings.append("Fig.2C one or more sources appear already pre-aggregated; adapter preserved available rows and performed no additional averaging.")
    panel_df = pd.DataFrame(rows)
    processing_stats = _processing_stats(
        source_paths=paths,
        raw_rows_read=raw_rows_read,
        layer3_rows=layer3_rows,
        rows_written=len(panel_df),
        averaging_performed=False,
        source_appeared_preaggregated=source_preaggregated,
    )
    return _write_result(output_dir, figure_id, panel_id, panel_df, paths[0], checked, warnings, source_paths=paths, processing_stats=processing_stats)


def build_fig2_wpri(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.2D row-level WPRI data."""
    figure_id, panel_id = _ids(spec)
    paths, checked = _source_paths(repo_root, spec)
    if not paths:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2D WPRI sources not found.")
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    raw_rows_read = 0
    layer3_rows = 0
    source_preaggregated_flags: list[bool] = []
    for idx, path in enumerate(paths):
        df = pd.read_csv(path)
        raw_rows_read += len(df)
        missing = {"layer", "WPRI"}.difference(df.columns)
        if missing:
            warnings.append(f"Skipping {path}: missing columns {sorted(missing)}")
            continue
        l3 = _layer3(df)
        layer3_rows += len(l3)
        seed_id = _seed_from_path(path)
        network_index = idx if seed_id else ""
        source_preaggregated_flags.append(_appears_preaggregated(l3, value_cols=["WPRI"]))
        if not seed_id:
            warnings.append(f"Fig.2D source lacks seed identity in path; preserving rows with blank seed_id for {path}.")
        rows.extend(
            _row_from_source(
                figure_id,
                panel_id,
                "whole_pair_representation_index",
                "Layer 3",
                "Layer 3",
                float(row["WPRI"]),
                "score",
                path,
                repo_root,
                row,
                fallback_network_id=network_index,
                fallback_seed_id=seed_id,
                source_row_index=source_row_index,
            )
            for source_row_index, row in l3.iterrows()
            if pd.notna(row.get("WPRI"))
        )
    if len(paths) == 1:
        warnings.append("Fig.2D used single-file fallback; row-level values are preserved but n=20 network coverage may be unavailable.")
    source_preaggregated = any(source_preaggregated_flags)
    if source_preaggregated:
        warnings.append("Fig.2D one or more sources appear already pre-aggregated; adapter preserved available rows and performed no additional averaging.")
    panel_df = pd.DataFrame(rows)
    processing_stats = _processing_stats(
        source_paths=paths,
        raw_rows_read=raw_rows_read,
        layer3_rows=layer3_rows,
        rows_written=len(panel_df),
        averaging_performed=False,
        source_appeared_preaggregated=source_preaggregated,
    )
    return _write_result(output_dir, figure_id, panel_id, panel_df, paths[0], checked, warnings, reference_value=0, source_paths=paths, processing_stats=processing_stats)


def build_fig2_neutral_ping_access(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.2E neutral-ping functional access data."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2E neutral-ping source not found.")
    df = pd.read_csv(path)
    missing = {"state_condition", "P_pair", "P_A"}.difference(df.columns)
    if missing:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.2E source missing columns {sorted(missing)}: {path}")

    warnings: list[str] = []
    seed_cols = ["network_seed", "network_index"] if "network_seed" in df.columns else []
    if not seed_cols:
        warnings.append("Fig.2E used single-run fallback; n=20 network summary unavailable.")
        df = df.copy()
        df["network_seed"] = ""
        df["network_index"] = ""
        seed_cols = ["network_seed", "network_index"]
    prob_cols = [col for col in ("P_pair", "P_A", "P_B", "P_other", "P_silent") if col in df.columns]
    grouped = df[df["state_condition"].isin(STATE_LABELS)].groupby(seed_cols + ["state_condition"], dropna=False)[prob_cols].mean().reset_index()
    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        condition = STATE_LABELS[str(row["state_condition"])]
        for metric_label, col in (("Pair-member readout", "P_pair"), ("Item 1 accessibility", "P_A")):
            out = _row(
                figure_id,
                panel_id,
                metric_label,
                condition,
                "",
                row.get("network_index", ""),
                row.get("network_seed", ""),
                float(row[col]),
                "probability",
                path,
                repo_root,
            )
            out["memory_state"] = condition
            out["readout_category"] = metric_label
            out["functional_metric"] = metric_label
            out["source_state_condition"] = str(row["state_condition"])
            out["state_condition"] = str(row["state_condition"])
            for prob_col in ("P_A", "P_B", "P_other", "P_silent", "P_pair"):
                out[prob_col] = float(row[prob_col]) if prob_col in row.index and pd.notna(row[prob_col]) else ""
            rows.append(out)
    panel_df = pd.DataFrame(rows)
    return _write_result(output_dir, figure_id, panel_id, panel_df, path, checked, warnings)


def build_fig2_partial_cue_completion(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.2F dropout recovery curves and AUC summaries."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.2F partial-cue curve source not found.")
    df = pd.read_csv(path)
    missing = {"state_condition", "keep_prob", "P_A"}.difference(df.columns)
    if missing:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.2F source missing columns {sorted(missing)}: {path}")

    warnings: list[str] = []
    if "network_seed" not in df.columns:
        warnings.append("Fig.2F used single-run curve fallback; n=20 network summary unavailable.")
        df = df.copy()
        df["network_seed"] = ""
        df["network_index"] = ""
    curve = (
        df[df["state_condition"].isin(STATE_LABELS)]
        .groupby(["network_seed", "network_index", "state_condition", "keep_prob"], dropna=False)["P_A"]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for _, row in curve.iterrows():
        condition = STATE_LABELS[str(row["state_condition"])]
        out = _row(
            figure_id,
            panel_id,
            "Item 1 recovery probability",
            condition,
            "",
            row.get("network_index", ""),
            row.get("network_seed", ""),
            float(row["P_A"]),
            "probability",
            path,
            repo_root,
        )
        out["memory_state"] = condition
        out["dropout_level"] = float(1.0 - float(row["keep_prob"]))
        out["keep_prob"] = float(row["keep_prob"])
        out["state_condition"] = str(row["state_condition"])
        out["P_A"] = float(row["P_A"])
        out["curve_or_summary"] = "curve"
        rows.append(out)

    auc_path, auc_checked = first_existing_path(repo_root, list((spec.get("source_mapping") or {}).get("auc_files") or []) + list((spec.get("source_mapping") or {}).get("auc_fallback_files") or []))
    checked.extend(auc_checked)
    source_paths = [path]
    if auc_path is not None:
        source_paths.append(auc_path)
        auc_df = pd.read_csv(auc_path)
        if {"state_condition", "AUC_A"}.issubset(auc_df.columns):
            if "network_seed" not in auc_df.columns:
                auc_df = auc_df.copy()
                auc_df["network_seed"] = ""
                auc_df["network_index"] = ""
            auc = (
                auc_df[auc_df["state_condition"].isin(STATE_LABELS)]
                .groupby(["network_seed", "network_index", "state_condition"], dropna=False)["AUC_A"]
                .mean()
                .reset_index()
            )
            for _, row in auc.iterrows():
                condition = STATE_LABELS[str(row["state_condition"])]
                out = _row(
                    figure_id,
                    panel_id,
                    "Item 1 recovery AUC",
                    condition,
                    "",
                    row.get("network_index", ""),
                    row.get("network_seed", ""),
                    float(row["AUC_A"]),
                    "probability",
                    auc_path,
                    repo_root,
                )
                out["memory_state"] = condition
                out["dropout_level"] = ""
                out["auc_value"] = float(row["AUC_A"])
                out["state_condition"] = str(row["state_condition"])
                out["P_A"] = ""
                out["curve_or_summary"] = "summary"
                rows.append(out)
        else:
            warnings.append(f"Fig.2F AUC source missing state_condition/AUC_A columns: {auc_path}")
    else:
        warnings.append("Fig.2F AUC summary source not found; rendering curve only.")

    panel_df = pd.DataFrame(rows)
    return _write_result(output_dir, figure_id, panel_id, panel_df, path, checked, warnings, source_paths=source_paths)


def _write_result(
    output_dir: Path,
    figure_id: str,
    panel_id: str,
    panel_df: pd.DataFrame,
    primary_path: Path,
    checked: list[str],
    warnings: list[str],
    *,
    reference_value: float | None = None,
    source_paths: list[Path] | None = None,
    processing_stats: Mapping[str, Any] | None = None,
) -> AdapterResult:
    metric = str(panel_df["metric"].iloc[0]) if not panel_df.empty and "metric" in panel_df.columns else ""
    group_cols = [c for c in ("metric", "condition", "layer", "dropout_level", "curve_or_summary") if c in panel_df.columns]
    stats: dict[str, Any] = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "summaries": summarize_values(panel_df, group_cols),
        "values_used_for_plotting": _values(panel_df),
        "n_networks": _n_networks(panel_df),
    }
    if reference_value is not None:
        stats["reference_value"] = reference_value
    if processing_stats:
        stats.update(dict(processing_stats))
    sources = source_paths or [primary_path]
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok" if not panel_df.empty else "missing_source",
        "sources": [{"path": str(path), "exists": path.exists()} for path in sources],
        "checked_candidates": checked,
    }
    if processing_stats:
        manifest["processing"] = dict(processing_stats)
    if panel_df.empty:
        warnings.append(f"Fig.2{panel_id} adapter produced no plottable rows.")
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def _source_paths(repo_root: Path, spec: Mapping[str, Any]) -> tuple[list[Path], list[str]]:
    mapping = spec.get("source_mapping") or {}
    checked: list[str] = []
    paths: list[Path] = []
    for pattern in mapping.get("preferred_globs") or []:
        checked.append(str(repo_root / pattern))
        paths.extend(sorted(repo_root.glob(str(pattern).replace("\\", "/"))))
    if paths:
        return paths, checked
    path, file_checked = first_existing_path(repo_root, list(mapping.get("preferred_files") or []) + list(mapping.get("fallback_files") or []))
    checked.extend(file_checked)
    return ([path] if path is not None else []), checked


def _candidate_files(spec: Mapping[str, Any]) -> list[str]:
    mapping = spec.get("source_mapping") or {}
    return list(mapping.get("preferred_files") or []) + list(mapping.get("fallback_files") or [])


def _layer3(df: pd.DataFrame) -> pd.DataFrame:
    layer = df["layer"].astype(str)
    out = df[layer.isin(["L3", "layer3", "Layer 3"])].copy()
    return out if not out.empty else df.copy()


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig2")), str(spec.get("panel_id", "")).upper()


def _seed_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("seed_"):
            return part.replace("seed_", "")
    return ""


def _rel(path: Path | str, repo_root: Path) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.relative_to(repo_root))
    except ValueError:
        return str(path_obj)


def _row_from_source(
    figure_id: str,
    panel_id: str,
    metric: str,
    condition: str,
    layer: str,
    value: float,
    unit: str,
    source_file: Path | str,
    repo_root: Path,
    source_row: Mapping[str, Any],
    *,
    fallback_network_id: Any = "",
    fallback_seed_id: Any = "",
    source_row_index: Any = "",
) -> dict[str, Any]:
    row = _row(
        figure_id,
        panel_id,
        metric,
        condition,
        layer,
        source_row.get("network_index", fallback_network_id),
        source_row.get("network_seed", fallback_seed_id),
        value,
        unit,
        source_file,
        repo_root,
    )
    for col in ROW_IDENTIFIER_COLUMNS:
        if col in source_row:
            row[col] = source_row.get(col)
    row["source_row_index"] = source_row_index
    return row


def _processing_stats(
    *,
    source_paths: list[Path],
    raw_rows_read: int,
    layer3_rows: int,
    rows_written: int,
    averaging_performed: bool,
    source_appeared_preaggregated: bool,
) -> dict[str, Any]:
    return {
        "n_source_files": len(source_paths),
        "raw_rows_read": int(raw_rows_read),
        "layer3_rows_before_aggregation": int(layer3_rows),
        "rows_written_to_panel_data": int(rows_written),
        "averaging_performed": bool(averaging_performed),
        "source_appeared_preaggregated": bool(source_appeared_preaggregated),
    }


def _appears_preaggregated(df: pd.DataFrame, *, value_cols: list[str]) -> bool:
    if df.empty:
        return False
    if any(col in df.columns for col in ("trial_id", "pair_id", "episode_id", "sample_id", "triplet_id")):
        return False
    n = len(df)
    if "network_seed" in df.columns:
        n_networks = df["network_seed"].replace("", pd.NA).dropna().nunique()
        return n_networks > 0 and n <= n_networks
    return n <= 1


def _row(
    figure_id: str,
    panel_id: str,
    metric: str,
    condition: str,
    layer: str,
    network_id: Any,
    seed_id: Any,
    value: float,
    unit: str,
    source_file: Path | str,
    repo_root: Path,
) -> dict[str, Any]:
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
