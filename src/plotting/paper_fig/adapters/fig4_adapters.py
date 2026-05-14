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


CONDITION_LABELS = {
    "sample_keep_overlap_only_dynamic": "Overlap-preserving",
    "sample_keep_nonoverlap_only_dynamic": "Non-overlap control",
}
TRAJECTORY_RESULT_COLUMNS = ("pair_id", "replacement_push_kstar", "replacement_pullback_kstar")
TRAJECTORY_VECTOR_KEYS = ("pair_id", "delta_V", "Delta_hat_plus", "Delta_hat_minus")


def build_fig4_similarity_accuracy_drop(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.4B ordered sample-probe similarity-bin accuracy-drop data."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.4B similarity-bin accuracy-drop source not found.")

    df = pd.read_csv(path)
    warnings: list[str] = []
    if "acc_drop" in df.columns:
        value_col = "acc_drop"
    elif "accuracy_drop" in df.columns:
        value_col = "accuracy_drop"
    else:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.4B source missing acc_drop/accuracy_drop: {path}")

    if "similarity_bin" not in df.columns:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.4B source missing similarity_bin: {path}")

    if "network_seed" not in df.columns:
        warnings.append("Fig.4B used single-run/bin summary fallback; n=20 network summary unavailable.")
        df["network_seed"] = ""
        df["network_index"] = ""
    if "bin_index" not in df.columns:
        warnings.append("Fig.4B source has no bin_index; preserving file order as similarity_bin_order.")
        order_map = {label: idx for idx, label in enumerate(pd.unique(df["similarity_bin"]))}
        df["bin_index"] = df["similarity_bin"].map(order_map)

    rows = []
    for _, r in df.iterrows():
        value = _to_percent(r[value_col])
        out = _row(
            figure_id,
            panel_id,
            "probe_accuracy_drop",
            str(r["similarity_bin"]),
            "",
            r.get("network_index", ""),
            r.get("network_seed", ""),
            value,
            "percent",
            path,
            repo_root,
        )
        out["similarity_bin"] = str(r["similarity_bin"])
        out["similarity_bin_order"] = int(r["bin_index"]) if pd.notna(r["bin_index"]) else ""
        out["sample_probe_similarity"] = _safe_float(r.get("bin_center", r.get("sample_probe_similarity", np.nan)))
        out["accuracy_drop"] = value
        rows.append(out)
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, warnings)


def build_fig4_overlap_accuracy_drop(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.4C high-vs-low overlap accuracy-drop comparison in the high-similarity regime."""
    figure_id, panel_id = _ids(spec)
    paths, checked = _source_paths(repo_root, spec)
    if not paths:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.4C overlap accuracy-drop source not found.")

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for idx, path in enumerate(paths):
        if path.suffix.lower() != ".csv":
            warnings.append(f"Skipping non-CSV Fig.4C source: {path}")
            continue
        df = pd.read_csv(path)
        if {"acc_drop_low", "acc_drop_high"}.issubset(df.columns):
            seed = _seed_from_path(path) or _single_value(df, "network_seed")
            network = _single_value(df, "network_index") if "network_index" in df.columns else idx
            values = {
                "Low overlap": _to_percent(pd.to_numeric(df["acc_drop_low"], errors="coerce").mean()),
                "High overlap": _to_percent(pd.to_numeric(df["acc_drop_high"], errors="coerce").mean()),
            }
        elif {"condition", "acc_drop"}.issubset(df.columns):
            seed = _single_value(df, "network_seed")
            network = _single_value(df, "network_index")
            low = df[df["condition"].astype(str).str.contains("low", case=False, na=False)]["acc_drop"]
            high = df[df["condition"].astype(str).str.contains("high", case=False, na=False)]["acc_drop"]
            values = {
                "Low overlap": _to_percent(pd.to_numeric(low, errors="coerce").mean()),
                "High overlap": _to_percent(pd.to_numeric(high, errors="coerce").mean()),
            }
        else:
            warnings.append(f"Skipping Fig.4C source missing overlap columns: {path}")
            continue
        for condition, value in values.items():
            out = _row(figure_id, panel_id, "probe_accuracy_drop", condition, "", network, seed, value, "percent", path, repo_root)
            out["overlap_level"] = condition
            out["similarity_regime"] = "high_similarity"
            out["accuracy_drop"] = value
            rows.append(out)
    if len(paths) == 1:
        warnings.append("Fig.4C used single-file fallback; n=20 network summary may be unavailable.")
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), paths, checked, warnings)


def build_fig4_dynamic_probe_index_timecourse(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.4D Layer 3 dynamic-probe index timecourse data."""
    figure_id, panel_id = _ids(spec)
    paths, checked = _source_paths(repo_root, spec)
    if not paths:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.4D dynamic-probe index source not found.")

    npz_paths = [path for path in paths if path.suffix.lower() == ".npz"]
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    if npz_paths:
        for idx, path in enumerate(npz_paths):
            with np.load(path, allow_pickle=True) as payload:
                if "DPI_L3" not in payload or "condition_name" not in payload:
                    warnings.append(f"Skipping Fig.4D NPZ missing DPI_L3/condition_name: {path}")
                    continue
                dpi = np.asarray(payload["DPI_L3"], dtype=float)
                cond_raw = np.asarray(payload["condition_name"]).astype(str)
            seed = _seed_from_path(path)
            network = idx if seed else ""
            for raw_name, display_name in CONDITION_LABELS.items():
                mask = cond_raw == raw_name
                if not np.any(mask):
                    warnings.append(f"Fig.4D source lacks condition {raw_name}: {path}")
                    continue
                mean_trace = np.nanmean(dpi[mask], axis=0)
                for time_idx, value in enumerate(mean_trace):
                    out = _row(figure_id, panel_id, "dynamic_probe_index", display_name, "Layer 3", network, seed, float(value), "index", path, repo_root)
                    out["probe_time"] = int(time_idx)
                    out["time_ms"] = int(time_idx)
                    out["dynamic_probe_index"] = float(value)
                    out["source_condition"] = raw_name
                    rows.append(out)
        return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), npz_paths, checked, warnings, reference_value=0)

    path = paths[0]
    df = pd.read_csv(path)
    if not {"condition", "DPI_L3"}.issubset(df.columns):
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.4D source missing condition/DPI_L3: {path}")
    warnings.append("Fig.4D used summary-only DPI fallback; no probe-time trajectory was found.")
    use = df[df["condition"].isin(CONDITION_LABELS)].copy()
    grouped = use.groupby(["network_seed", "network_index", "condition"], dropna=False)["DPI_L3"].mean().reset_index()
    for _, r in grouped.iterrows():
        display_name = CONDITION_LABELS.get(str(r["condition"]), str(r["condition"]))
        out = _row(figure_id, panel_id, "dynamic_probe_index", display_name, "Layer 3", r.get("network_index", ""), r.get("network_seed", ""), float(r["DPI_L3"]), "index", path, repo_root)
        out["dynamic_probe_index"] = float(r["DPI_L3"])
        out["source_condition"] = str(r["condition"])
        out["fallback_summary_only"] = True
        rows.append(out)
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, warnings, reference_value=0)


def build_fig4_static_dynamic_trajectory(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.4E static/dynamic manipulation trajectory coordinates."""
    figure_id, panel_id = _ids(spec)
    result_path, vector_path, checked = _trajectory_sources(repo_root, spec)
    if result_path is None or vector_path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.4E static/dynamic trajectory source pair_results.csv or pair_vectors.npz not found.")
    if result_path.suffix.lower() != ".csv" or vector_path.suffix.lower() != ".npz":
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.4E trajectory sources must be CSV plus NPZ: {result_path}, {vector_path}")

    df = pd.read_csv(result_path)
    missing_columns = [col for col in TRAJECTORY_RESULT_COLUMNS if col not in df.columns]
    if missing_columns:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.4E pair_results.csv missing required columns {missing_columns}: {result_path}")

    with np.load(vector_path, allow_pickle=True) as payload:
        vectors = {key: np.asarray(payload[key]) for key in payload.files}
    missing_keys = [key for key in TRAJECTORY_VECTOR_KEYS if key not in vectors]
    if missing_keys:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.4E pair_vectors.npz missing required arrays {missing_keys}: {vector_path}")

    warnings: list[str] = []
    try:
        table, scales = _build_trajectory_table(df, vectors)
    except Exception as exc:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.4E trajectory table could not be built: {exc}")

    rows: list[dict[str, Any]] = []
    labels = {
        "plus": "plus: static -> dynamic",
        "minus": "minus: dynamic -> static",
    }
    for _, r in table.iterrows():
        for prefix in ("plus", "minus"):
            x0 = float(r[f"{prefix}_x0"])
            y0 = float(r[f"{prefix}_y0"])
            x1 = float(r[f"{prefix}_x1"])
            y1 = float(r[f"{prefix}_y1"])
            value = float(np.hypot(x1 - x0, y1 - y0))
            out = _row(figure_id, panel_id, "static_dynamic_manipulation_trajectory", labels[prefix], "", "", "", value, "normalized_shift", result_path, repo_root)
            out["pair_id"] = int(r["pair_id"])
            out["group"] = prefix
            out["prefix"] = prefix
            out["x0"] = x0
            out["y0"] = y0
            out["x1"] = x1
            out["y1"] = y1
            out["before_x"] = x0
            out["before_y"] = y0
            out["after_x"] = x1
            out["after_y"] = y1
            out["firing_shift"] = float(r[f"{prefix}_fire_shift"])
            out["decision_shift"] = float(r[f"{prefix}_decision_shift"])
            out["result_source_file"] = _rel(result_path, repo_root)
            out["vector_source_file"] = _rel(vector_path, repo_root)
            out["logic_reference"] = "l3_accumulator_mechanism_experiment_plot"
            rows.append(out)

    extra_stats = {
        "n_pairs": int(table["pair_id"].nunique()) if "pair_id" in table.columns else 0,
        "robust_scales": scales,
        "mean_plus_start": _mean_point(table, "plus", "0"),
        "mean_plus_end": _mean_point(table, "plus", "1"),
        "mean_minus_start": _mean_point(table, "minus", "0"),
        "mean_minus_end": _mean_point(table, "minus", "1"),
        "trajectory_logic_reference": "src/plotting/experiments/l3_accumulator_mechanism_experiment_plot.py",
        "adapter_performed_averaging": False,
        "source_appeared_already_preaggregated": False,
    }
    manifest_extra = {
        "trajectory_logic_reference": "src/plotting/experiments/l3_accumulator_mechanism_experiment_plot.py",
        "required_result_columns": list(TRAJECTORY_RESULT_COLUMNS),
        "required_vector_keys": list(TRAJECTORY_VECTOR_KEYS),
    }
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [result_path, vector_path], checked, warnings, extra_stats=extra_stats, manifest_extra=manifest_extra)


def build_fig4_final_readout_recovery(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Compatibility shim for older specs; current Fig.4E uses static/dynamic trajectories."""
    return build_fig4_static_dynamic_trajectory(spec, repo_root, output_dir)


def _trajectory_sources(repo_root: Path, spec: Mapping[str, Any]) -> tuple[Path | None, Path | None, list[str]]:
    mapping = spec.get("source_mapping") or {}
    result_path, result_checked = first_existing_path(repo_root, list(mapping.get("preferred_files") or []) + list(mapping.get("fallback_files") or []))
    vector_path, vector_checked = first_existing_path(repo_root, list(mapping.get("vector_files") or []) + list(mapping.get("vector_fallback_files") or []))
    return result_path, vector_path, [*result_checked, *vector_checked]


def _build_trajectory_table(df: pd.DataFrame, vectors: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, float]]:
    ordered_df, ordered_vectors = _aligned_results_and_vectors(df, vectors)
    delta_v = np.asarray(ordered_vectors["delta_V"], dtype=np.float64)
    delta_hat_plus = np.asarray(ordered_vectors["Delta_hat_plus"], dtype=np.float64)
    delta_hat_minus = np.asarray(ordered_vectors["Delta_hat_minus"], dtype=np.float64)

    plus_fire = ordered_df["replacement_push_kstar"].to_numpy(dtype=np.float64)
    minus_fire = ordered_df["replacement_pullback_kstar"].to_numpy(dtype=np.float64)
    plus_decision = _decision_projection(delta_hat_plus, delta_v)
    minus_decision = _decision_projection(delta_hat_minus, delta_v)

    scales = {
        "plus_fire": _positive_robust_scale(plus_fire),
        "minus_fire": _positive_robust_scale(minus_fire),
        "plus_decision": _positive_robust_scale(plus_decision),
        "minus_decision": _positive_robust_scale(minus_decision),
    }
    plus_fire_shift = _normalized_shift(plus_fire, scales["plus_fire"])
    minus_fire_shift = _normalized_shift(minus_fire, scales["minus_fire"])
    plus_decision_shift = _normalized_shift(plus_decision, scales["plus_decision"])
    minus_decision_shift = _normalized_shift(minus_decision, scales["minus_decision"])

    table = pd.DataFrame(
        {
            "pair_id": ordered_df["pair_id"].astype(int),
            "plus_x0": -1.0,
            "plus_y0": -1.0,
            "plus_x1": -1.0 + 2.0 * plus_fire_shift,
            "plus_y1": -1.0 + 2.0 * plus_decision_shift,
            "plus_fire_shift": plus_fire_shift,
            "plus_decision_shift": plus_decision_shift,
            "minus_x0": 1.0,
            "minus_y0": 1.0,
            "minus_x1": 1.0 - 2.0 * minus_fire_shift,
            "minus_y1": 1.0 - 2.0 * minus_decision_shift,
            "minus_fire_shift": minus_fire_shift,
            "minus_decision_shift": minus_decision_shift,
        }
    )
    return table, scales


def _aligned_results_and_vectors(df: pd.DataFrame, vectors: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    vector_pair_ids = np.asarray(vectors["pair_id"], dtype=np.int64)
    vector_order = {int(pair_id): index for index, pair_id in enumerate(vector_pair_ids)}
    missing = sorted(set(df["pair_id"].astype(int)) - set(vector_order))
    if missing:
        preview = ", ".join(str(item) for item in missing[:8])
        raise ValueError(f"pair_vectors.npz missing pair_id values from pair_results.csv: {preview}")
    ordered_df = df.sort_values("pair_id", kind="stable").reset_index(drop=True)
    indices = np.asarray([vector_order[int(pair_id)] for pair_id in ordered_df["pair_id"]], dtype=np.int64)
    ordered_vectors = {key: np.asarray(vectors[key])[indices] for key in TRAJECTORY_VECTOR_KEYS}
    return ordered_df, ordered_vectors


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


def _mean_point(table: pd.DataFrame, prefix: str, suffix: str) -> dict[str, float]:
    return {
        "x": float(np.nanmean(table[f"{prefix}_x{suffix}"].to_numpy(dtype=float))),
        "y": float(np.nanmean(table[f"{prefix}_y{suffix}"].to_numpy(dtype=float))),
    }


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
    extra_stats: Mapping[str, Any] | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
) -> AdapterResult:
    metric = str(panel_df["metric"].iloc[0]) if not panel_df.empty and "metric" in panel_df.columns else ""
    group_cols = [c for c in ("metric", "condition", "layer", "similarity_bin", "probe_time") if c in panel_df.columns]
    stats: dict[str, Any] = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "summaries": summarize_values(panel_df, group_cols),
        "values_used_for_plotting": _values(panel_df),
        "n_networks": _n_networks(panel_df),
    }
    if panel_id == "D" and "probe_time" in panel_df.columns:
        stats["timecourse_summary"] = summarize_values(panel_df, ["condition", "probe_time"])
    if reference_value is not None:
        stats["reference_value"] = reference_value
    if extra_stats:
        stats.update(dict(extra_stats))
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok" if not panel_df.empty else "missing_source",
        "sources": [{"path": _rel(path, Path.cwd()), "exists": path.exists()} for path in source_paths],
        "checked_candidates": checked,
        "condition_label_mapping": CONDITION_LABELS,
    }
    if manifest_extra:
        manifest.update(dict(manifest_extra))
    if panel_df.empty:
        warnings.append(f"Fig.4{panel_id} adapter produced no plottable rows.")
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
    path, file_checked = first_existing_path(repo_root, _candidate_files(spec))
    checked.extend(file_checked)
    return ([path] if path is not None else []), checked


def _candidate_files(spec: Mapping[str, Any]) -> list[str]:
    mapping = spec.get("source_mapping") or {}
    return list(mapping.get("preferred_files") or []) + list(mapping.get("fallback_files") or [])


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig4")), str(spec.get("panel_id", "")).upper()


def _seed_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("seed_"):
            return part.replace("seed_", "")
    return ""


def _single_value(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns or df[column].dropna().empty:
        return ""
    return df[column].dropna().iloc[0]


def _to_percent(value: Any) -> float:
    numeric = float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])
    return numeric * 100.0 if abs(numeric) <= 1.5 else numeric


def _safe_float(value: Any) -> float | str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(numeric) else float(numeric)


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
