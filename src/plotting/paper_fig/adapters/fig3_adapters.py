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

ROW_IDENTIFIER_COLUMNS = (
    "trial_id",
    "seq_len",
    "stage_k",
    "item_index",
    "pair_id",
    "episode_id",
    "triplet_id",
    "DI_bin",
    "network_seed",
    "network_index",
    "eval_seed",
    "run_dir",
)


def build_fig3_fusion_imbalance(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.3A row-level Layer 3 fusion imbalance scores."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.3A fusion imbalance source not found.")
    df = pd.read_csv(path)
    missing = {"layer", "fusion_imbalance"}.difference(df.columns)
    if missing:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.3A source missing columns {sorted(missing)}: {path}")
    l3 = _layer3(df)
    warnings: list[str] = []
    if "network_seed" not in l3.columns:
        warnings.append("Fig.3A source lacks network_seed; row-level values are preserved but n=20 network identity is unavailable.")
    source_preaggregated = _appears_preaggregated(l3)
    if source_preaggregated:
        warnings.append("Fig.3A source appears already pre-aggregated; adapter preserved available rows and performed no additional averaging.")
    rows = [
        _row_from_source(figure_id, panel_id, "fusion_imbalance_score", "Layer 3", "Layer 3", float(r["fusion_imbalance"]), "score", path, repo_root, r, source_row_index=idx)
        for idx, r in l3.iterrows()
        if pd.notna(r.get("fusion_imbalance"))
    ]
    processing_stats = _processing_stats(
        source_paths=[path],
        raw_rows_read=len(df),
        layer3_rows=len(l3),
        rows_written=len(rows),
        averaging_performed=False,
        source_appeared_preaggregated=source_preaggregated,
    )
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, warnings, reference_value=0, processing_stats=processing_stats)


def build_fig3_latent_bias_readout_preference(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.3B latent bias to readout preference relationship data."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.3B latent-bias/readout source not found.")
    df = pd.read_csv(path)
    warnings: list[str] = []
    if {"record_type", "DI_mean", "sample_first_prob", "distractor_first_prob"}.issubset(df.columns):
        use = df[df["record_type"].eq("binned_summary")].copy()
        source_level = "binned_summary"
        x_col = "DI_mean"
        use["readout_preference"] = pd.to_numeric(use["sample_first_prob"], errors="coerce") - pd.to_numeric(use["distractor_first_prob"], errors="coerce")
    elif {"DI", "sample_first", "distractor_first"}.issubset(df.columns):
        use = df.copy()
        source_level = "trial_level"
        x_col = "DI"
        use["readout_preference"] = pd.to_numeric(use["sample_first"], errors="coerce") - pd.to_numeric(use["distractor_first"], errors="coerce")
        warnings.append("Fig.3B used trial-level relationship fallback; binned network summary unavailable.")
    else:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.3B source missing latent/readout columns: {path}")
    use = use.dropna(subset=[x_col, "readout_preference"])
    rows: list[dict[str, Any]] = []
    for _, r in use.iterrows():
        row = _row(
            figure_id,
            panel_id,
            "latent_bias_readout_preference",
            "Latent bias vs readout preference",
            str(r.get("layer", "")),
            r.get("network_index", ""),
            r.get("network_seed", ""),
            float(r["readout_preference"]),
            "score",
            path,
            repo_root,
        )
        row["x_value"] = float(r[x_col])
        row["y_value"] = float(r["readout_preference"])
        row["latent_state_bias"] = float(r[x_col])
        row["readout_preference"] = float(r["readout_preference"])
        row["record_type"] = source_level
        row["DI_mean"] = float(r[x_col])
        row["sample_first_prob"] = float(r["sample_first_prob"]) if "sample_first_prob" in r.index and pd.notna(r["sample_first_prob"]) else ""
        row["distractor_first_prob"] = float(r["distractor_first_prob"]) if "distractor_first_prob" in r.index and pd.notna(r["distractor_first_prob"]) else ""
        row["pair_id"] = r.get("triplet_id", "")
        row["source_record_type"] = source_level
        row["DI_bin"] = r.get("DI_bin", "")
        rows.append(row)
    panel_df = pd.DataFrame(rows)
    correlations = _correlations(panel_df.get("x_value", pd.Series(dtype=float)), panel_df.get("y_value", pd.Series(dtype=float)))
    return _write_result(output_dir, figure_id, panel_id, panel_df, [path], checked, warnings, correlations=correlations)


def build_fig3_latest_vs_earlier_mass(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.3C row-level latest-item versus earlier-items similarity mass."""
    figure_id, panel_id = _ids(spec)
    paths, checked = _source_paths(repo_root, spec)
    if not paths:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.3C item-similarity source not found.")
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    raw_rows_read = 0
    layer3_rows = 0
    final_stage_rows = 0
    source_preaggregated_flags: list[bool] = []
    for idx, path in enumerate(paths):
        df = pd.read_csv(path)
        raw_rows_read += len(df)
        required = {"trial_id", "seq_len", "stage_k", "layer", "item_index", "similarity_weight_nonnegative"}
        missing = required.difference(df.columns)
        if missing:
            warnings.append(f"Skipping {path}: missing columns {sorted(missing)}")
            continue
        l3 = _layer3(df)
        layer3_rows += len(l3)
        source_preaggregated_flags.append(_appears_preaggregated(l3))
        final = l3[(pd.to_numeric(l3["stage_k"], errors="coerce") == pd.to_numeric(l3["seq_len"], errors="coerce")) & (pd.to_numeric(l3["seq_len"], errors="coerce") > 1)].copy()
        final_stage_rows += len(final)
        if final.empty:
            warnings.append(f"Skipping {path}: no final-stage sequence rows")
            continue
        seed = _seed_from_path(path)
        for (trial_id, seq_len), part in final.groupby(["trial_id", "seq_len"], dropna=False):
            weights = pd.to_numeric(part["similarity_weight_nonnegative"], errors="coerce").fillna(0.0)
            latest_mask = pd.to_numeric(part["item_index"], errors="coerce").eq(pd.to_numeric(part["stage_k"], errors="coerce"))
            latest = float(weights[latest_mask].sum())
            earlier = float(weights[~latest_mask].sum())
            total = latest + earlier
            if total > 0:
                latest /= total
                earlier /= total
            stage_k = pd.to_numeric(part["stage_k"], errors="coerce").dropna()
            source_row_index = ";".join(str(v) for v in part.index.tolist())
            for condition, value in (("Latest item", latest), ("Earlier items", earlier)):
                out = _row(
                    figure_id,
                    panel_id,
                    "item_similarity_mass",
                    condition,
                    "Layer 3",
                    idx if seed else "",
                    seed,
                    value,
                    "normalized_mass",
                    path,
                    repo_root,
                )
                out["contribution_type"] = condition
                out["trial_id"] = trial_id
                out["seq_len"] = seq_len
                out["stage_k"] = int(stage_k.iloc[0]) if not stage_k.empty else seq_len
                out["source_row_index"] = source_row_index
                rows.append(out)
    if len(paths) == 1:
        warnings.append("Fig.3C used single-file fallback; row-level values are preserved but n=20 network coverage may be unavailable.")
    source_preaggregated = any(source_preaggregated_flags)
    if source_preaggregated:
        warnings.append("Fig.3C one or more sources appear already pre-aggregated; adapter preserved available rows and performed no additional averaging.")
    processing_stats = _processing_stats(
        source_paths=paths,
        raw_rows_read=raw_rows_read,
        layer3_rows=layer3_rows,
        rows_written=len(rows),
        averaging_performed=False,
        source_appeared_preaggregated=source_preaggregated,
        extra={"final_stage_layer3_rows": int(final_stage_rows)},
    )
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), paths, checked, warnings, processing_stats=processing_stats)


def build_fig3_seen_item_ping_access(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.3D row-level neutral-ping seen-item hit-rate data."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.3D seen-item ping source not found.")
    df = pd.read_csv(path)
    if "ping_seen_item_hit_rate" not in df.columns:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.3D source missing ping_seen_item_hit_rate: {path}")
    use = _layer3(df)
    use = use[pd.to_numeric(use.get("stage_k", 0), errors="coerce") > 1].copy()
    warnings: list[str] = []
    if "network_seed" not in use.columns:
        warnings.append("Fig.3D source lacks network_seed; row-level values are preserved but n=20 network identity is unavailable.")
    source_preaggregated = _appears_preaggregated(use)
    if source_preaggregated:
        warnings.append("Fig.3D source appears already pre-aggregated; adapter preserved available rows and performed no additional averaging.")
    rows = []
    for idx, r in use.iterrows():
        if pd.isna(r.get("ping_seen_item_hit_rate")):
            continue
        out = _row_from_source(figure_id, panel_id, "seen_item_hit_rate", "Neutral ping", "Layer 3", float(r["ping_seen_item_hit_rate"]), "probability", path, repo_root, r, source_row_index=idx)
        out["seen_item_hit_rate"] = float(r["ping_seen_item_hit_rate"])
        rows.append(out)
    processing_stats = _processing_stats(
        source_paths=[path],
        raw_rows_read=len(df),
        layer3_rows=len(_layer3(df)),
        rows_written=len(rows),
        averaging_performed=False,
        source_appeared_preaggregated=source_preaggregated,
        extra={"post_stage_filter_rows": int(len(use))},
    )
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, warnings, processing_stats=processing_stats)


def build_fig3_state_com_shift(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.3E state center-of-mass trajectory data."""
    return _build_com_shift(spec, repo_root, output_dir, metric="state_center_of_mass", source_col="com_sim", unit="position")


def build_fig3_ping_com_shift(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.3F neutral-ping center-of-mass trajectory data."""
    return _build_com_shift(spec, repo_root, output_dir, metric="ping_center_of_mass", source_col="ping_com", unit="position")


def _build_com_shift(spec: Mapping[str, Any], repo_root: Path, output_dir: Path, *, metric: str, source_col: str, unit: str) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.3{panel_id} COM source not found.")
    df = pd.read_csv(path)
    if source_col not in df.columns or "stage_k" not in df.columns:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.3{panel_id} source missing {source_col}/stage_k: {path}")
    use = _layer3(df).dropna(subset=[source_col, "stage_k"]).copy()
    warnings: list[str] = []
    if "network_seed" not in use.columns:
        use["network_seed"] = ""
        use["network_index"] = ""
        warnings.append(f"Fig.3{panel_id} source lacks network_seed; row-level values are preserved but n=20 network identity is unavailable.")
    source_preaggregated = _appears_preaggregated(use)
    if source_preaggregated:
        warnings.append(f"Fig.3{panel_id} source appears already pre-aggregated; adapter preserved available rows and performed no additional averaging.")
    rows = []
    for idx, r in use.iterrows():
        out = _row_from_source(figure_id, panel_id, metric, "Layer 3", "Layer 3", float(r[source_col]), unit, path, repo_root, r, source_row_index=idx)
        out["sequence_stage"] = int(r["stage_k"])
        out["stage_k"] = int(r["stage_k"])
        out["seq_len"] = int(r["seq_len"]) if "seq_len" in r.index and pd.notna(r["seq_len"]) else int(r["stage_k"])
        out[source_col] = float(r[source_col])
        out[metric] = float(r[source_col])
        rows.append(out)
    processing_stats = _processing_stats(
        source_paths=[path],
        raw_rows_read=len(df),
        layer3_rows=len(_layer3(df)),
        rows_written=len(rows),
        averaging_performed=False,
        source_appeared_preaggregated=source_preaggregated,
    )
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, warnings, processing_stats=processing_stats)


def _write_result(
    output_dir: Path,
    figure_id: str,
    panel_id: str,
    panel_df: pd.DataFrame,
    source_paths: list[Path],
    checked: list[str],
    warnings: list[str],
    *,
    reference_value: float | None = None,
    correlations: dict[str, Any] | None = None,
    processing_stats: Mapping[str, Any] | None = None,
) -> AdapterResult:
    metric = str(panel_df["metric"].iloc[0]) if not panel_df.empty and "metric" in panel_df.columns else ""
    group_cols = [c for c in ("metric", "condition", "layer", "sequence_stage") if c in panel_df.columns]
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
    if correlations is not None:
        stats["correlations"] = correlations
        stats["regression_summary"] = correlations.get("linear_regression", {})
    if processing_stats:
        stats.update(dict(processing_stats))
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok" if not panel_df.empty else "missing_source",
        "sources": [{"path": str(path), "exists": path.exists()} for path in source_paths],
        "checked_candidates": checked,
    }
    if processing_stats:
        manifest["processing"] = dict(processing_stats)
    if panel_id == "F":
        manifest["unused_source_stems"] = ["stepwise_update_ratio"]
    if panel_df.empty:
        warnings.append(f"Fig.3{panel_id} adapter produced no plottable rows.")
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
    if "layer" not in df.columns:
        return df.copy()
    layer = df["layer"].astype(str)
    out = df[layer.isin(["L3", "layer3", "Layer 3"])].copy()
    return out if not out.empty else df.copy()


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig3")), str(spec.get("panel_id", "")).upper()


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
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stats = {
        "n_source_files": len(source_paths),
        "raw_rows_read": int(raw_rows_read),
        "layer3_rows_before_aggregation": int(layer3_rows),
        "rows_written_to_panel_data": int(rows_written),
        "averaging_performed": bool(averaging_performed),
        "source_appeared_preaggregated": bool(source_appeared_preaggregated),
    }
    if extra:
        stats.update(dict(extra))
    return stats


def _appears_preaggregated(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    if any(col in df.columns for col in ("trial_id", "triplet_id", "pair_id", "episode_id", "sample_id", "seq_len", "stage_k", "item_index")):
        return False
    n = len(df)
    if "network_seed" in df.columns:
        n_networks = df["network_seed"].replace("", pd.NA).dropna().nunique()
        return n_networks > 0 and n <= n_networks
    return n <= 1


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


def _correlations(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(data) < 3:
        return {"n": int(len(data)), "pearson_r": None, "spearman_rho": None, "linear_regression": {}}
    pearson = float(data["x"].corr(data["y"], method="pearson"))
    spearman = float(data["x"].corr(data["y"], method="spearman"))
    slope, intercept = np.polyfit(data["x"].to_numpy(dtype=float), data["y"].to_numpy(dtype=float), 1)
    return {
        "n": int(len(data)),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "linear_regression": {"slope": float(slope), "intercept": float(intercept)},
    }
