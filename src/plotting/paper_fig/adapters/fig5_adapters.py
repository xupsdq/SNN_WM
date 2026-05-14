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


UNIT_GROUP_LABELS = {
    "overlap_dominant": "Overlap-dominant units",
    "probe_only_dominant": "Probe-only-dominant units",
}

EVENT_LABELS = {
    "winner_pre_spike_boost": "Winner boost",
    "loser_post_winner_suppressed": "Loser suppression",
    "full_chain_satisfied": "Full winner-loser sequence",
}


def build_fig5_support_map_example(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.5A example sample/probe/mask/support-map pixels from an existing NPZ."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.5A support-map NPZ source not found.")
    if path.suffix.lower() != ".npz":
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.5A expected NPZ source, found: {path}")

    with np.load(path, allow_pickle=True) as payload:
        required = ("sample_mask", "probe_mask", "overlap_mask", "probe_only_mask", "ux_map_pre_dynamic")
        missing = [key for key in required if key not in payload]
        if missing:
            return missing_adapter_result(spec, repo_root, output_dir, f"Fig.5A NPZ missing arrays {missing}: {path}")
        arrays = {key: np.asarray(payload[key]) for key in required}
        example_id = f"trial_{int(np.ravel(payload.get('trial_id', [0]))[0])}"

    rows: list[dict[str, Any]] = []
    for image_type, arr in arrays.items():
        for y_idx, x_idx in np.ndindex(arr.shape):
            value = float(arr[y_idx, x_idx])
            out = _row(figure_id, panel_id, "pre_probe_stsp_support_map", image_type, "Layer 1", "", _seed_from_path(path), value, "support", path, repo_root)
            out["example_id"] = example_id
            out["image_type"] = image_type
            out["sample_image_path"] = ""
            out["probe_image_path"] = ""
            out["mask_type"] = image_type if "mask" in image_type else ""
            out["mask_x"] = int(x_idx)
            out["mask_y"] = int(y_idx)
            out["support_x"] = int(x_idx)
            out["support_y"] = int(y_idx)
            out["support_value"] = value if image_type == "ux_map_pre_dynamic" else ""
            out["region_type"] = _region_type(image_type)
            rows.append(out)
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, [])


def build_fig5_overlap_vs_probe_only_support(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.5B overlap-aligned versus probe-only pre-probe support data."""
    figure_id, panel_id = _ids(spec)
    path, checked = first_existing_path(repo_root, _candidate_files(spec))
    if path is None:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.5B pre-probe support summary source not found.")
    df = pd.read_csv(path)
    required = {"ux_overlap_pre", "ux_probe_only_pre"}
    missing = required.difference(df.columns)
    if missing:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.5B source missing columns {sorted(missing)}: {path}")

    warnings: list[str] = []
    if "model_type" in df.columns:
        df = df[df["model_type"].astype(str).eq("dynamic")].copy()
    if "network_seed" not in df.columns:
        warnings.append("Fig.5B used single-run fallback; n=20 network summary unavailable.")
        df["network_seed"] = ""
        df["network_index"] = ""
    grouped = df.groupby(["network_seed", "network_index"], dropna=False)[["ux_overlap_pre", "ux_probe_only_pre"]].mean().reset_index()
    rows = []
    for _, r in grouped.iterrows():
        for condition, col in (("Overlap-aligned", "ux_overlap_pre"), ("Probe-only", "ux_probe_only_pre")):
            value = float(r[col])
            out = _row(figure_id, panel_id, "pre_probe_stsp_support", condition, "Layer 1", r.get("network_index", ""), r.get("network_seed", ""), value, "support", path, repo_root)
            out["support_region"] = condition
            out["overlap_aligned_support"] = float(r["ux_overlap_pre"])
            out["probe_only_support"] = float(r["ux_probe_only_pre"])
            out["source_condition"] = col
            rows.append(out)
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, warnings)


def build_fig5_early_probe_transition(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.5C early advance/recruitment transition rates by unit group."""
    figure_id, panel_id = _ids(spec)
    paths, checked = _source_paths(repo_root, spec)
    if not paths:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.5C early transition source not found.")

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for idx, path in enumerate(paths):
        df = pd.read_csv(path)
        required = {"unit_group", "P_advance", "P_recruit", "P_loss", "P_unchanged"}
        missing = required.difference(df.columns)
        if missing:
            warnings.append(f"Skipping Fig.5C source missing columns {sorted(missing)}: {path}")
            continue
        if "aggregation_scope" in df.columns:
            df = df[df["aggregation_scope"].astype(str).eq("per_trial")].copy()
        seed = _seed_from_path(path) or _single_value(df, "network_seed")
        network = idx if seed else _single_value(df, "network_index")
        for source_group, condition in UNIT_GROUP_LABELS.items():
            part = df[df["unit_group"].astype(str).eq(source_group)]
            if part.empty:
                warnings.append(f"Fig.5C source lacks unit_group={source_group}: {path}")
                continue
            p_advance = float(pd.to_numeric(part["P_advance"], errors="coerce").mean()) * 100.0
            p_recruit = float(pd.to_numeric(part["P_recruit"], errors="coerce").mean()) * 100.0
            p_loss = float(pd.to_numeric(part["P_loss"], errors="coerce").mean()) * 100.0
            p_unchanged = float(pd.to_numeric(part["P_unchanged"], errors="coerce").mean()) * 100.0
            value = p_advance + p_recruit
            out = _row(figure_id, panel_id, "advanced_plus_recruited_fraction", condition, "Layer 1", network, seed, value, "percent", path, repo_root)
            out["unit_group"] = condition
            out["source_unit_group"] = source_group
            out["transition_type"] = "advanced_plus_recruited"
            out["advanced_fraction"] = p_advance
            out["recruited_fraction"] = p_recruit
            out["advanced_plus_recruited_fraction"] = value
            out["lost_fraction"] = p_loss
            out["unchanged_fraction"] = p_unchanged
            out["early_window"] = "early_probe"
            rows.append(out)
    if len(paths) == 1:
        warnings.append("Fig.5C used single-file fallback; n=20 network summary may be unavailable.")
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), paths, checked, warnings)


def build_fig5_event_aligned_voltage_inhibition(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.5D event-aligned winner-loser voltage and loser-inhibition traces."""
    figure_id, panel_id = _ids(spec)
    paths, checked = _source_paths(repo_root, spec)
    if not paths:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.5D event-aligned trace source not found.")

    npz_paths = [path for path in paths if path.suffix.lower() == ".npz"]
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    if npz_paths:
        for idx, path in enumerate(npz_paths):
            with np.load(path, allow_pickle=True) as payload:
                required = ("relative_time", "winner_delta_v_aligned", "loser_delta_v_aligned", "loser_inh_before_aligned")
                missing = [key for key in required if key not in payload]
                if missing:
                    warnings.append(f"Skipping Fig.5D NPZ missing arrays {missing}: {path}")
                    continue
                rel = np.asarray(payload["relative_time"], dtype=float)
                winner_v = np.asarray(payload["winner_delta_v_aligned"], dtype=float)
                loser_v = np.asarray(payload["loser_delta_v_aligned"], dtype=float)
                loser_inh = np.asarray(payload["loser_inh_before_aligned"], dtype=float)
            seed = _seed_from_path(path)
            network = idx if seed else ""
            voltage_diff = _nanmean_no_warn(winner_v - loser_v, axis=0)
            pre_mask = rel < 0
            if np.any(pre_mask):
                inh_baseline = _nanmean_no_warn(loser_inh[:, pre_mask], axis=1).reshape(-1, 1)
            else:
                inh_baseline = loser_inh[:, :1]
            first_col = loser_inh[:, :1]
            inh_baseline = np.where(np.isfinite(inh_baseline), inh_baseline, first_col)
            inhibition_change = _nanmean_no_warn(loser_inh - inh_baseline, axis=0)
            for time_value, value in zip(rel, voltage_diff):
                out = _row(figure_id, panel_id, "winner_loser_voltage_difference", "Dynamic winner events", "Layer 1", network, seed, float(value), "voltage", path, repo_root)
                out["time_from_winner_spike"] = float(time_value)
                out["time_ms"] = float(time_value)
                out["winner_loser_voltage_difference"] = float(value)
                out["local_inhibition_change"] = ""
                out["trace_type"] = "winner_loser_voltage_difference"
                out["event_group"] = "local_winner_loser"
                rows.append(out)
            for time_value, value in zip(rel, inhibition_change):
                out = _row(figure_id, panel_id, "local_inhibition_change", "Dynamic winner events", "Layer 1", network, seed, float(value), "voltage", path, repo_root)
                out["time_from_winner_spike"] = float(time_value)
                out["time_ms"] = float(time_value)
                out["winner_loser_voltage_difference"] = ""
                out["local_inhibition_change"] = float(value)
                out["trace_type"] = "local_inhibition_change"
                out["event_group"] = "local_winner_loser"
                rows.append(out)
        return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), npz_paths, checked, warnings, reference_x=0)

    path = paths[0]
    df = pd.read_csv(path)
    required = {"winner_pre_spike_delta_v_mean", "loser_post_winner_inh_rise"}
    missing = required.difference(df.columns)
    if missing:
        return missing_adapter_result(spec, repo_root, output_dir, f"Fig.5D summary fallback missing columns {sorted(missing)}: {path}")
    warnings.append("Fig.5D used summary-only fallback; no event-aligned traces were found.")
    for metric, col in (("winner_loser_voltage_difference", "winner_pre_spike_delta_v_mean"), ("local_inhibition_change", "loser_post_winner_inh_rise")):
        value = float(pd.to_numeric(df[col], errors="coerce").mean())
        out = _row(figure_id, panel_id, metric, "Dynamic winner events", "Layer 1", "", "", value, "voltage", path, repo_root)
        out[metric] = value
        out["trace_type"] = metric
        out["event_group"] = "local_winner_loser"
        out["fallback_summary_only"] = True
        rows.append(out)
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), [path], checked, warnings, reference_x=0)


def build_fig5_winner_loser_event_fractions(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Build Fig.5E winner-boost, loser-suppression, and full-chain event fractions."""
    figure_id, panel_id = _ids(spec)
    paths, checked = _source_paths(repo_root, spec)
    if not paths:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.5E local causal-chain event source not found.")

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for idx, path in enumerate(paths):
        df = pd.read_csv(path)
        missing = [col for col in EVENT_LABELS if col not in df.columns]
        if missing:
            warnings.append(f"Skipping Fig.5E source missing columns {missing}: {path}")
            continue
        seed = _seed_from_path(path) or _single_value(df, "network_seed")
        network = idx if seed else _single_value(df, "network_index")
        for col, label in EVENT_LABELS.items():
            value = float(pd.to_numeric(df[col], errors="coerce").mean()) * 100.0
            out = _row(figure_id, panel_id, "fraction_of_local_events", label, "Layer 1", network, seed, value, "percent", path, repo_root)
            out["event_pattern"] = label
            out["source_event_pattern"] = col
            out["fraction_of_local_events"] = value
            rows.append(out)
    if len(paths) == 1:
        warnings.append("Fig.5E used single-file fallback; n=20 network summary may be unavailable.")
    return _write_result(output_dir, figure_id, panel_id, pd.DataFrame(rows), paths, checked, warnings)


def _write_result(
    output_dir: Path,
    figure_id: str,
    panel_id: str,
    panel_df: pd.DataFrame,
    source_paths: list[Path],
    checked: list[str],
    warnings: list[str],
    *,
    reference_x: float | None = None,
) -> AdapterResult:
    metric = str(panel_df["metric"].iloc[0]) if not panel_df.empty and "metric" in panel_df.columns else ""
    group_cols = [c for c in ("metric", "condition", "layer", "trace_type", "time_from_winner_spike") if c in panel_df.columns]
    stats: dict[str, Any] = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": metric,
        "summaries": summarize_values(panel_df, group_cols),
        "values_used_for_plotting": _values(panel_df),
        "n_networks": _n_networks(panel_df),
    }
    if panel_id == "D" and "time_from_winner_spike" in panel_df.columns:
        stats["timecourse_summary"] = summarize_values(panel_df, ["metric", "trace_type", "time_from_winner_spike"])
    if reference_x is not None:
        stats["reference_x"] = reference_x
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok" if not panel_df.empty else "missing_source",
        "sources": [{"path": _rel(path, Path.cwd()), "exists": path.exists()} for path in source_paths],
        "checked_candidates": checked,
        "label_mappings": {"unit_groups": UNIT_GROUP_LABELS, "event_patterns": EVENT_LABELS},
    }
    if panel_df.empty:
        warnings.append(f"Fig.5{panel_id} adapter produced no plottable rows.")
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
    return str(spec.get("figure_id", "fig5")), str(spec.get("panel_id", "")).upper()


def _seed_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("seed_"):
            return part.replace("seed_", "")
    return ""


def _single_value(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns or df[column].dropna().empty:
        return ""
    return df[column].dropna().iloc[0]


def _nanmean_no_warn(values: np.ndarray, axis: int) -> np.ndarray:
    """Compute nanmean while returning NaN for all-NaN slices without runtime warnings."""
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    counts = valid.sum(axis=axis)
    sums = np.where(valid, arr, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sums / counts
    return np.where(counts > 0, out, np.nan)


def _region_type(image_type: str) -> str:
    if image_type == "overlap_mask":
        return "Overlap-aligned"
    if image_type == "probe_only_mask":
        return "Probe-only"
    if image_type == "ux_map_pre_dynamic":
        return "Pre-probe STSP support"
    return image_type.replace("_", " ")


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
