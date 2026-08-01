from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.plotting.paper_fig.data_resolver import AdapterResult, summarize_values, write_adapter_outputs
from src.plotting.paper_fig.adapters.fig3_adapters import (
    MEMORY_DISPLAY,
    MEMORY_ORDER,
    REGION_ORDER,
    _canonical,
    _cue_order,
    _display,
    _finish,
    _first_existing,
    _float,
    _ids,
    _max_numeric,
    _missing,
    _normalize_cue,
    _normalize_region,
    _resolve_experiment_root,
    _region_order,
    _serial_bin_order,
    _serial_position,
    _scale_delta_to_percent,
    _seed_id,
    _source,
    _to_percent_value,
)


READOUT_CLASS_ORDER = ("latest", "recent", "earlier", "silent")
TARGET_POSITION_BIN_ORDER = ("early", "middle", "recent", "latest")


def build_part2_frozen_statistics_adapter(
    spec: Mapping[str, Any], repo_root: Path, output_dir: Path
) -> AdapterResult:
    """Load one immutable S3 statistic payload without recomputation or fallback."""
    figure_id, panel_id = _ids(spec)
    if figure_id != "supp_fig_s3" or panel_id not in set("ABCDEF"):
        raise ValueError(
            "part2_frozen_statistics_adapter is restricted to supp_fig_s3 panels A-F"
        )

    contract = spec.get("frozen_statistics")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{figure_id}{panel_id}: frozen_statistics contract is required")
    source_rel = str(contract.get("path", "")).strip()
    expected_sha256 = str(contract.get("sha256", "")).strip().lower()
    identity_fields = tuple(map(str, contract.get("identity_fields") or ()))
    expected_rows = int(contract.get("rows", 0))
    if not source_rel or len(expected_sha256) != 64 or not identity_fields or expected_rows <= 0:
        raise ValueError(f"{figure_id}{panel_id}: incomplete frozen_statistics contract")

    source_path = (repo_root / source_rel).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"{figure_id}{panel_id}: frozen statistic payload is missing: {source_rel}"
        )
    source_bytes = source_path.read_bytes()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{figure_id}{panel_id}: frozen statistic SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {actual_sha256}"
        )
    try:
        payload = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{figure_id}{panel_id}: invalid UTF-8 JSON payload") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{figure_id}{panel_id}: frozen statistic payload must be an object")
    if payload.get("figure_id") != figure_id or payload.get("panel_id") != panel_id:
        raise ValueError(f"{figure_id}{panel_id}: frozen statistic figure/panel identity mismatch")
    if payload.get("warnings") != []:
        raise ValueError(f"{figure_id}{panel_id}: frozen statistic payload contains warnings")

    network_ids = payload.get("network_ids")
    network_summaries = payload.get("network_summaries")
    summaries = payload.get("summaries")
    if not isinstance(network_ids, list) or not isinstance(network_summaries, list) or not isinstance(summaries, list):
        raise ValueError(f"{figure_id}{panel_id}: required frozen statistic arrays are missing")
    if len(network_ids) != int(payload.get("n_networks", -1)) or len(network_ids) != int(
        payload.get("n_networks_observed", -1)
    ):
        raise ValueError(f"{figure_id}{panel_id}: frozen network count fields disagree")
    if len(set(map(str, network_ids))) != len(network_ids):
        raise ValueError(f"{figure_id}{panel_id}: frozen network ids are not unique")
    if len(network_summaries) != expected_rows or len(summaries) != expected_rows:
        raise ValueError(
            f"{figure_id}{panel_id}: expected {expected_rows} frozen summary rows; "
            f"observed {len(network_summaries)} network summaries and {len(summaries)} summaries"
        )

    network_order = _frozen_identity_order(network_summaries, identity_fields, figure_id, panel_id)
    summary_order = _frozen_identity_order(summaries, identity_fields, figure_id, panel_id)
    expected_network_order = _contract_identity_order(
        contract, "network_summary_identity_order", identity_fields, figure_id, panel_id
    )
    expected_summary_order = _contract_identity_order(
        contract, "summary_identity_order", identity_fields, figure_id, panel_id
    )
    expected_display_order = _contract_identity_order(
        contract, "display_identity_order", identity_fields, figure_id, panel_id
    )
    if network_order != expected_network_order:
        raise ValueError(f"{figure_id}{panel_id}: frozen network-summary identity/order mismatch")
    if summary_order != expected_summary_order:
        raise ValueError(f"{figure_id}{panel_id}: frozen summary identity/order mismatch")
    if set(expected_display_order) != set(network_order):
        if panel_id == "A" and set(expected_display_order).issubset(set(network_order)):
            pass
        else:
            raise ValueError(f"{figure_id}{panel_id}: display identity set mismatch")

    required_statistic_fields = {
        "mean",
        "sem",
        "ci95_low",
        "ci95_high",
        "n_networks",
        "one_sample_p_vs_zero",
    }
    for summary in network_summaries:
        if not isinstance(summary, Mapping) or not required_statistic_fields.issubset(summary):
            raise ValueError(f"{figure_id}{panel_id}: malformed frozen network summary")
        if int(summary["n_networks"]) != len(network_ids):
            raise ValueError(f"{figure_id}{panel_id}: frozen row network count mismatch")

    summary_by_identity = {identity: row for identity, row in zip(summary_order, summaries)}
    rows: list[dict[str, Any]] = []
    for identity in expected_display_order if panel_id == "A" else summary_order:
        summary = summary_by_identity[identity]
        values = summary.get("values_used_for_plotting")
        if not isinstance(values, list) or len(values) != len(network_ids):
            raise ValueError(f"{figure_id}{panel_id}: frozen plotting values do not align to network ids")
        for network_id, value in zip(network_ids, values):
            row = {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "metric": summary.get("metric", payload.get("metric", "")),
                "condition": summary.get("condition", summary.get("metric", "")),
                "layer": "",
                "network_id": str(network_id),
                "seed_id": str(network_id),
                "value": value,
                "unit": "frozen_display_value",
                "source_file": source_rel,
                "frozen_source_sha256": actual_sha256,
            }
            for field in identity_fields:
                row[field] = summary[field]
            rows.append(row)

    panel_df = pd.DataFrame(rows)
    if panel_df.empty or panel_df.duplicated(["network_id", *identity_fields]).any():
        raise ValueError(f"{figure_id}{panel_id}: frozen panel rows are empty or duplicated")

    output_stats = dict(payload)
    output_stats["frozen_source_path"] = source_rel
    output_stats["frozen_source_sha256"] = actual_sha256
    output_stats["frozen_identity_fields"] = list(identity_fields)
    output_stats["frozen_network_summary_identity_order"] = [list(item) for item in network_order]
    output_stats["frozen_summary_identity_order"] = [list(item) for item in summary_order]
    output_stats["frozen_display_identity_order"] = [list(item) for item in expected_display_order]
    output_stats["plot_only_no_recompute"] = True
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok",
        "run_mode": payload.get("run_mode"),
        "n_networks": len(network_ids),
        "network_ids": list(map(str, network_ids)),
        "inferential_unit": payload.get("inferential_unit"),
        "source_files_used": [source_rel],
        "checked_candidates": [source_rel],
        "sources": [
            {
                "path": source_rel,
                "exists": True,
                "size_bytes": len(source_bytes),
                "sha256": actual_sha256,
                "role": "immutable_frozen_statistics",
            }
        ],
        "frozen_schema_identity_order_validated": True,
        "plot_only_no_recompute": True,
        "warnings": [],
    }
    return write_adapter_outputs(
        output_dir, figure_id, panel_id, panel_df, output_stats, manifest, []
    )


def _contract_identity_order(
    contract: Mapping[str, Any],
    key: str,
    identity_fields: Sequence[str],
    figure_id: str,
    panel_id: str,
) -> list[tuple[Any, ...]]:
    raw_order = contract.get(key)
    if not isinstance(raw_order, list):
        raise ValueError(f"{figure_id}{panel_id}: {key} must be a list")
    order: list[tuple[Any, ...]] = []
    for item in raw_order:
        if not isinstance(item, list) or len(item) != len(identity_fields):
            raise ValueError(f"{figure_id}{panel_id}: malformed {key} entry")
        order.append(tuple(item))
    if len(set(order)) != len(order):
        raise ValueError(f"{figure_id}{panel_id}: duplicate identity in {key}")
    return order


def _frozen_identity_order(
    rows: Sequence[Mapping[str, Any]],
    identity_fields: Sequence[str],
    figure_id: str,
    panel_id: str,
) -> list[tuple[Any, ...]]:
    order: list[tuple[Any, ...]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not set(identity_fields).issubset(row):
            raise ValueError(f"{figure_id}{panel_id}: frozen identity fields are missing")
        order.append(tuple(row[field] for field in identity_fields))
    if len(set(order)) != len(order):
        raise ValueError(f"{figure_id}{panel_id}: frozen identity rows are duplicated")
    return order


def build_part2_fit_comparison_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    _root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    source_rel = str(spec.get("source", "data/metrics/supp_s3a_morphology_fit_comparison.csv"))
    curve_rel = str(spec.get("curve_source", "data/metrics/supp_s3a_morphology_fit_curve_points.csv"))
    sources: list[Path] = []
    required = {"network_seed", "linear_sse", "saturating_sse", "linear_minus_saturating_sse", "saturating_asymptote"}
    rows: list[dict[str, Any]] = []
    metrics = (
        ("linear_sse", "Linear SSE", "sse"),
        ("saturating_sse", "Saturating SSE", "sse"),
        ("linear_minus_saturating_sse", "Linear - saturating SSE", "delta_sse"),
        ("saturating_asymptote", "Saturating asymptote", "items"),
    )
    for seed_dir in seeds:
        fit_path = seed_dir / source_rel
        curve_path = seed_dir / curve_rel
        sources.extend([fit_path, curve_path])
        if not fit_path.is_file():
            warnings.append(f"Missing Fig.3 fit comparison source: {_display(fit_path, repo_root)}")
            continue
        fit = pd.read_csv(fit_path)
        if not required.issubset(fit.columns):
            warnings.append(f"Fit comparison source lacks {sorted(required - set(fit.columns))}: {_display(fit_path, repo_root)}")
            continue
        if not curve_path.is_file():
            warnings.append(f"Missing Fig.3 fit curve source: {_display(curve_path, repo_root)}")
        for _, row in fit.iterrows():
            for metric, condition, unit in metrics:
                value = _float(row.get(metric))
                if np.isfinite(value):
                    rows.append(_canonical(figure_id, panel_id, metric=metric, condition=condition, layer="layer1", seed_id=row.get("network_seed", _seed_id(seed_dir)), value=value, unit=unit, source_file=_display(fit_path, repo_root), lower_level_rows=row.get("n_curve_points", "")))
    return _part2_finish(spec, repo_root, output_dir, pd.DataFrame(rows), sources, warnings, ["metric"])


def build_part2_peak_valley_null_network_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    _root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    source_rel = str(spec.get("source", "data/metrics/supp_network_peak_valley_summary.csv"))
    sources: list[Path] = []
    rows: list[dict[str, Any]] = []
    metrics = (
        ("mean_peak_valley_delta", "observed_peak_valley_delta"),
        ("mean_null_peak_valley_delta_p95", "null_peak_valley_delta_p95"),
        ("fraction_structured_sequences", "fraction_structured_sequences"),
    )
    for seed_dir in seeds:
        path = seed_dir / source_rel
        sources.append(path)
        if not path.is_file():
            warnings.append(f"Missing Fig.3 peak-valley network summary: {_display(path, repo_root)}")
            continue
        frame = pd.read_csv(path)
        if not {"network_seed", "n_sequences"}.issubset(frame.columns):
            warnings.append(f"Peak-valley summary lacks network_seed/n_sequences: {_display(path, repo_root)}")
            continue
        weights = pd.to_numeric(frame["n_sequences"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for source_col, metric in metrics:
            if source_col not in frame.columns:
                warnings.append(f"Peak-valley summary lacks {source_col}: {_display(path, repo_root)}")
                continue
            values = pd.to_numeric(frame[source_col], errors="coerce").to_numpy(dtype=float)
            value = _weighted_mean(values, weights)
            if np.isfinite(value):
                rows.append(_canonical(figure_id, panel_id, metric=metric, condition=metric, layer="layer1", seed_id=frame["network_seed"].iloc[0], value=value, unit="fraction" if metric == "fraction_structured_sequences" else "value", source_file=_display(path, repo_root), lower_level_rows=int(np.nansum(weights)), aggregation="sequence_condition_to_network"))
    return _part2_finish(spec, repo_root, output_dir, pd.DataFrame(rows), sources, warnings, ["metric"])


def build_part2_existing_network_panel_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    panel_type = str(spec.get("panel_type", ""))
    if panel_type == "ping_recency_decomposition":
        return _build_part2_current_ping_recency_adapter(spec, repo_root, output_dir)
    if panel_type == "weak_probe_recency_gain":
        return _build_part2_current_weak_cue_adapter(spec, repo_root, output_dir)
    source_path = _resolve_repo_path(repo_root, spec.get("source"))
    sources = [source_path] if source_path is not None else []
    if source_path is None or not source_path.is_file():
        return _part2_missing(spec, output_dir, sources, ["Existing twenty-network panel data is missing."])
    panel_df = pd.read_csv(source_path)
    if not {"network_id", "metric", "value"}.issubset(panel_df.columns):
        return _part2_missing(spec, output_dir, sources, ["Existing panel data lacks network_id/metric/value."])
    panel_df = panel_df.copy()
    panel_df["figure_id"] = str(spec.get("figure_id"))
    panel_df["panel_id"] = str(spec.get("panel_id"))
    panel_df["source_file"] = _display(source_path, repo_root)
    group_cols = [col for col in ("metric", "readout_class", "target_position_bin", "condition") if col in panel_df.columns]
    return _part2_finish(spec, repo_root, output_dir, panel_df, sources, [], group_cols)


def _build_part2_current_ping_recency_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    _root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    position_rel = str(spec.get("source", "data/metrics/panel_c_neutral_ping_position_distribution.csv"))
    summary_rel = str(spec.get("summary_source", "data/metrics/panel_c_neutral_ping_access_summary.csv"))
    sources: list[Path] = []
    rows: list[dict[str, Any]] = []
    key_cols = ["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms", "state_condition"]
    for seed_dir in seeds:
        position_path = seed_dir / position_rel
        summary_path = seed_dir / summary_rel
        sources.extend([position_path, summary_path])
        if not position_path.is_file() or not summary_path.is_file():
            warnings.append(f"Missing current neutral-ping sources under {_display(seed_dir, repo_root)}")
            continue
        position = pd.read_csv(position_path)
        summary = pd.read_csv(summary_path)
        required_position = set(key_cols) | {"serial_position", "readout_mass", "n_trials"}
        required_summary = set(key_cols) | {"P_silent", "n_trials"}
        if not required_position.issubset(position.columns) or not required_summary.issubset(summary.columns):
            warnings.append(f"Neutral-ping source schema mismatch under {_display(seed_dir, repo_root)}")
            continue
        position = position[position["state_condition"].astype(str).eq("S_final")].copy()
        summary = summary[summary["state_condition"].astype(str).eq("S_final")].copy()
        summary_index = summary.set_index(key_cols)
        sequence_rows: list[dict[str, Any]] = []
        for key, part in position.groupby(key_cols, dropna=False, sort=False):
            seq_len = int(key[key_cols.index("seq_len")])
            serial_position = pd.to_numeric(part["serial_position"], errors="coerce")
            mass = pd.to_numeric(part["readout_mass"], errors="coerce").fillna(0.0)
            latest = float(mass[serial_position.eq(seq_len)].sum())
            recent = float(mass[serial_position.ge(max(1, seq_len - 2)) & serial_position.lt(seq_len)].sum())
            earlier = float(mass[serial_position.lt(max(1, seq_len - 2))].sum())
            silent = float("nan")
            try:
                match = summary_index.loc[key]
                if isinstance(match, pd.DataFrame):
                    match = match.iloc[0]
                silent = _float(match.get("P_silent"))
            except KeyError:
                pass
            if not np.isfinite(silent):
                silent = max(0.0, 1.0 - latest - recent - earlier)
            sequence_rows.append({"latest": latest, "recent": recent, "earlier": earlier, "silent": silent, "n_trials": _max_numeric(part["n_trials"])})
        sequence = pd.DataFrame(sequence_rows)
        if sequence.empty:
            continue
        weights = pd.to_numeric(sequence["n_trials"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        seed = int(pd.to_numeric(position["network_seed"], errors="coerce").dropna().iloc[0])
        for readout_class in READOUT_CLASS_ORDER:
            value = _weighted_mean(pd.to_numeric(sequence[readout_class], errors="coerce").to_numpy(dtype=float), weights)
            rows.append(_canonical(figure_id, panel_id, metric="readout_mass", condition=readout_class, layer="layer3", seed_id=seed, value=value, unit="probability", source_file=_display(position_path, repo_root), readout_class=readout_class, readout_class_order=_readout_class_order(readout_class), state_condition="S_final", lower_level_rows=int(len(sequence)), n_trials=float(weights.sum()), aggregation="sequence_condition_to_network", x_value=_readout_class_order(readout_class), y_value=value))
    return _part2_finish(spec, repo_root, output_dir, pd.DataFrame(rows), sources, warnings, ["readout_class"])


def _build_part2_current_weak_cue_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    _root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    source_rel = str(spec.get("source", "data/metrics/panel_d_weak_cue_item_metrics.csv"))
    sources: list[Path] = []
    rows: list[dict[str, Any]] = []
    required = {"network_seed", "seq_len", "target_position", "memory_condition", "P_target", "n_trials"}
    for seed_dir in seeds:
        path = seed_dir / source_rel
        sources.append(path)
        if not path.is_file():
            warnings.append(f"Missing current weak-cue source: {_display(path, repo_root)}")
            continue
        frame = pd.read_csv(path)
        if not required.issubset(frame.columns):
            warnings.append(f"Weak-cue source lacks {sorted(required - set(frame.columns))}: {_display(path, repo_root)}")
            continue
        frame = frame.copy()
        frame["target_position_bin"] = [
            _current_target_position_bin(position, seq_len)
            for position, seq_len in zip(frame["target_position"], frame["seq_len"])
        ]
        frame["P_target"] = pd.to_numeric(frame["P_target"], errors="coerce")
        frame["n_trials"] = pd.to_numeric(frame["n_trials"], errors="coerce").fillna(0.0)
        seed = int(pd.to_numeric(frame["network_seed"], errors="coerce").dropna().iloc[0])
        for target_bin, part in frame.groupby("target_position_bin", sort=False):
            means: dict[str, float] = {}
            counts: dict[str, float] = {}
            for memory, memory_part in part.groupby("memory_condition", sort=False):
                values = memory_part["P_target"].to_numpy(dtype=float)
                weights = memory_part["n_trials"].to_numpy(dtype=float)
                means[str(memory)] = _weighted_mean(values, weights)
                counts[str(memory)] = float(np.nansum(weights))
            if not {"sequence_state", "single_item_memory"}.issubset(means):
                warnings.append(f"Weak-cue memory conditions incomplete for {target_bin}: {_display(path, repo_root)}")
                continue
            gain = 100.0 * (means["sequence_state"] - means["single_item_memory"])
            rows.append(_canonical(figure_id, panel_id, metric="target_recovery_gain", condition=str(target_bin), layer="layer3", seed_id=seed, value=gain, unit="percent_delta", source_file=_display(path, repo_root), target_position_bin=str(target_bin), target_position_bin_order=_target_position_bin_order(str(target_bin)), memory_condition="sequence_state_minus_single_item_memory", baseline_condition="single_item_memory", sequence_state_P_target=100.0 * means["sequence_state"], baseline_P_target=100.0 * means["single_item_memory"], lower_level_rows=int(len(part)), n_trials_sequence_state=counts["sequence_state"], n_trials_baseline=counts["single_item_memory"], aggregation="trial_weighted_rows_to_network", x_value=_target_position_bin_order(str(target_bin)), y_value=gain))
    return _part2_finish(spec, repo_root, output_dir, pd.DataFrame(rows), sources, warnings, ["target_position_bin"])


def _current_target_position_bin(target_position: Any, seq_len: Any) -> str:
    position = int(target_position)
    length = int(seq_len)
    if position == length:
        return "latest"
    if position >= length - 2:
        return "recent"
    if position <= 2:
        return "early"
    return "middle"


def build_part2_morphology_serial_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    focus_k = int(spec.get("focus_seq_len", 10))
    focus_delay = int(spec.get("focus_delay_ms", 400))
    sources: list[Path] = []
    rows: list[dict[str, Any]] = []
    for seed_dir in seeds:
        path = seed_dir / "data/metrics/panel_b_morphology_serial_profile.csv"
        sources.append(path)
        if not path.is_file():
            warnings.append(f"Missing morphology serial source: {_display(path, repo_root)}")
            continue
        frame = pd.read_csv(path)
        use = frame[(pd.to_numeric(frame.get("seq_len"), errors="coerce") == focus_k) & (pd.to_numeric(frame.get("delay_ms"), errors="coerce") == focus_delay)].copy()
        use["p_i"] = pd.to_numeric(use.get("p_i"), errors="coerce")
        grouped = use.dropna(subset=["p_i"]).groupby("serial_position", dropna=False, sort=True)["p_i"].agg(["mean", "size"]).reset_index()
        for _, row in grouped.iterrows():
            value = float(row["mean"])
            rows.append(_canonical(figure_id, panel_id, metric="morphology_support_mass", condition=f"K{focus_k}_D{focus_delay}", layer="layer1", seed_id=_seed_id(seed_dir), value=value, unit="mass", source_file=_display(path, repo_root), seq_len=focus_k, delay_ms=focus_delay, serial_position=int(row["serial_position"]), x_value=int(row["serial_position"]), y_value=value, lower_level_rows=int(row["size"])))
    return _part2_finish(spec, repo_root, output_dir, pd.DataFrame(rows), sources, warnings, ["serial_position"])


def build_part2_boundary_pair_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    metrics = tuple(map(str, spec.get("metrics") or ["N_eff_fraction", "rescued_fraction"]))
    sources: list[Path] = []
    rows: list[dict[str, Any]] = []
    for seed_dir in seeds:
        path = seed_dir / "data/metrics/panel_f_boundary_summary.csv"
        sources.append(path)
        if not path.is_file():
            warnings.append(f"Missing boundary summary: {_display(path, repo_root)}")
            continue
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            for metric in metrics:
                value = _float(row.get(metric))
                if not np.isfinite(value):
                    continue
                rows.append(_canonical(figure_id, panel_id, metric=metric, condition=str(row.get("condition_id", "")), layer="layer1", seed_id=row.get("network_seed", _seed_id(seed_dir)), value=value, unit="fraction", source_file=_display(path, repo_root), seq_len=int(row.get("seq_len")), delay_ms=int(row.get("delay_ms")), x_value=int(row.get("seq_len")), y_value=int(row.get("delay_ms")), z_value=value))
    return _part2_finish(spec, repo_root, output_dir, pd.DataFrame(rows), sources, warnings, ["metric", "seq_len", "delay_ms"])


def build_s5_peak_valley_contrast_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_region_support(spec, repo_root, output_dir, ["supp_peak_valley_contrast.csv", "panel_d_peak_valley_contrast.csv", "panel_c_example_landscape_summary.csv"])


def build_s5_landscape_nonflatness_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["supp_landscape_nonflatness.csv", "panel_d_landscape_nonflatness.csv"]
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for seed_dir in seeds:
        path = _first_existing(seed_dir / "data" / "metrics", names)
        if path is not None:
            sources.append(_source(path, repo_root, seed_dir))
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                for metric in ("top_q_mass_fraction", "support_gini", "support_cv"):
                    value = _float(row.get(metric))
                    if np.isfinite(value):
                        rows.append(_row(figure_id, panel_id, metric, metric, value, "value", row, seed_dir, path, repo_root, sequence_id=row.get("sequence_id", ""), seq_len=row.get("seq_len", "")))
            continue
        raw_path = seed_dir / "data" / "raw" / "panel_c_example_landscape.npz"
        if not raw_path.exists():
            sources.extend(_source(seed_dir / "data" / "metrics" / name, repo_root, seed_dir) for name in names)
            sources.append(_source(raw_path, repo_root, seed_dir))
            continue
        sources.append(_source(raw_path, repo_root, seed_dir))
        warnings.append(f"S4B degraded morphology fallback used representative panel_c_example_landscape.npz under {_display(seed_dir, repo_root)}.")
        rows.extend(_landscape_nonflatness_from_npz(figure_id, panel_id, seed_dir, raw_path, repo_root))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "top_q_mass_fraction", ["metric"])


def build_s5_peak_valley_null_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["supp_peak_valley_prevalence.csv", "supp_network_peak_valley_summary.csv", "panel_d_peak_valley_prevalence.csv", "panel_d_network_peak_valley_summary.csv", "panel_c_example_landscape_summary.csv"]
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            sources.extend(_source(seed_dir / "data" / "metrics" / name, repo_root, seed_dir) for name in names)
            continue
        sources.append(_source(path, repo_root, seed_dir))
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            if path.name == "panel_c_example_landscape_summary.csv":
                peak = _float(row.get("peak_mean_support"))
                valley = _float(row.get("valley_mean_support"))
                delta = peak - valley if np.isfinite(peak) and np.isfinite(valley) else float("nan")
                if np.isfinite(delta):
                    warnings.append(f"S4C degraded morphology fallback used representative panel_c_example_landscape_summary.csv under {_display(seed_dir, repo_root)}.")
                    rows.append(_row(figure_id, panel_id, "observed_peak_valley_delta", "observed_peak_valley_delta", delta, "value", row, seed_dir, path, repo_root, seq_len=row.get("seq_len", ""), n_trials=1, fallback_scope="representative_landscape"))
                    rows.append(_row(figure_id, panel_id, "is_structured", "is_structured", float(delta > 0), "value", row, seed_dir, path, repo_root, seq_len=row.get("seq_len", ""), n_trials=1, fallback_scope="representative_landscape"))
                continue
            for metric in ("observed_peak_valley_delta", "null_peak_valley_delta_p95", "is_structured", "fraction_structured_sequences"):
                value = _float(row.get(metric))
                if np.isfinite(value):
                    rows.append(_row(figure_id, panel_id, metric, metric, value, "value", row, seed_dir, path, repo_root, seq_len=row.get("seq_len", ""), n_trials=row.get("n_sequences", row.get("n_null", ""))))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "observed_peak_valley_delta", ["metric"])


def build_s5_anchor_dynamics_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["supp_anchor_dynamics_metrics.csv", "supp_recency_only_controls.csv", "panel_b_progressive_update_metrics.csv"]
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names)
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            continue
        if path.name == "panel_b_progressive_update_metrics.csv":
            warnings.append(f"S4D fallback used panel_b_progressive_update_metrics.csv under {_display(seed_dir, repo_root)} because supplement anchor metrics are missing.")
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            for metric in ("anchor_COM", "similarity_entropy", "state_displacement", "earlier_residual_proxy", "earlier_item_residual_mass"):
                value = _float(row.get(metric))
                if np.isfinite(value):
                    rows.append(_row(figure_id, panel_id, metric, metric, value, "value", row, seed_dir, path, repo_root, stage_k=row.get("stage_k", ""), x_value=row.get("stage_k", ""), y_value=value, seq_len=row.get("seq_len", "")))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "anchor_COM", ["stage_k"])


def build_s5_ping_recency_decomposition_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["panel_d_ping_position_distribution.csv", "panel_d_ping_summary.csv", "supp_ping_recency_diagnostics.csv"]
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for seed_dir in seeds:
        metrics_dir = seed_dir / "data" / "metrics"
        path = _first_existing(metrics_dir, names)
        if path is None:
            sources.extend(_source(metrics_dir / name, repo_root, seed_dir) for name in names)
            warnings.append(f"Missing S4E ping recency source under {_display(seed_dir, repo_root)}")
            continue
        sources.append(_source(path, repo_root, seed_dir))
        df = pd.read_csv(path)
        if path.name == "panel_d_ping_position_distribution.csv":
            rows.extend(_ping_recency_rows_from_distribution(df, figure_id, panel_id, seed_dir, path, repo_root, warnings))
        else:
            warnings.append(f"S4E degraded fallback used {path.name}; panel_d_ping_position_distribution.csv is preferred for latest/recent/earlier/silent decomposition.")
            rows.extend(_ping_recency_rows_from_diagnostics(df, figure_id, panel_id, seed_dir, path, repo_root))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "readout_mass", ["readout_class"])


def build_s5_weak_probe_recency_gain_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["panel_e_weak_probe_position_stratified_metrics.csv"]
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for seed_dir in seeds:
        path = _first_existing(seed_dir / "data" / "metrics", names)
        if path is None:
            missing = seed_dir / "data" / "metrics" / names[0]
            sources.append(_source(missing, repo_root, seed_dir))
            warnings.append(f"Missing S4F weak-probe recency source under {_display(seed_dir, repo_root)}")
            continue
        sources.append(_source(path, repo_root, seed_dir))
        rows.extend(_weak_probe_recency_gain_rows(pd.read_csv(path), figure_id, panel_id, seed_dir, path, repo_root, warnings))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "target_recovery_gain", ["target_position_bin"])


def build_s6_ping_recency_diagnostics_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["supp_ping_recency_diagnostics.csv", "panel_d_ping_summary.csv", "panel_d_ping_position_distribution.csv"]
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names)
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            continue
        df = pd.read_csv(path)
        if path.name == "panel_d_ping_position_distribution.csv":
            df = _recency_from_distribution(df)
        for _, row in df.iterrows():
            for metric in ("latest_item_mass", "recent_item_mass", "earlier_item_residual_mass", "ping_COM", "P_seen_item", "P_silent"):
                value = _float(row.get(metric))
                if np.isfinite(value):
                    rows.append(_row(figure_id, panel_id, metric, metric, value, "probability" if metric != "ping_COM" else "position", row, seed_dir, path, repo_root, state_condition=row.get("state_condition", ""), seq_len=row.get("seq_len", "")))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "earlier_item_residual_mass", ["metric"])


def build_s6_weak_probe_target_source_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["supp_weak_probe_target_source_control.csv", "supp_weak_probe_target_source_gain.csv", "panel_e_weak_probe_metrics.csv", "panel_e_weak_probe_memory_gain.csv"]
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names)
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            if "memory_gain" in path.name or "gain" in path.name:
                value = _float(row.get("target_recovery_gain", row.get("memory_gain")))
                metric = "memory_gain"
                condition = str(row.get("target_source", "target_source"))
                unit = "percent_delta"
                value = _scale_delta_to_percent(value) if np.isfinite(value) else value
            else:
                value = _float(row.get("P_target"))
                metric = "P_target"
                condition = str(row.get("target_source", row.get("memory_condition", "")))
                unit = "percent"
                value = _to_percent_value(value) if np.isfinite(value) else value
            if np.isfinite(value):
                rows.append(_row(figure_id, panel_id, metric, condition, value, unit, row, seed_dir, path, repo_root, target_source=row.get("target_source", ""), memory_condition=row.get("memory_condition", ""), keep_prob=row.get("keep_prob", ""), x_value=row.get("keep_prob", ""), y_value=value))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "P_target", ["target_source", "memory_condition"])


def build_s6_peak_cue_matching_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["supp_structural_weak_cue_matching_diagnostics.csv", "panel_f_peak_cue_matching_diagnostics.csv"]
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names)
    rows: list[dict[str, Any]] = []
    aliases = {
        "cue_pixel_count": ["cue_pixel_count", "cue_pixel_count_mean"],
        "cue_energy": ["cue_energy", "cue_energy_mean"],
        "encoded_spike_count": ["encoded_spike_count", "encoded_spike_count_mean"],
        "cue_spike_count": ["cue_spike_count"],
    }
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            cue = _normalize_cue(row.get("cue_condition", ""))
            for metric, cols in aliases.items():
                col = next((name for name in cols if name in row.index), None)
                value = _float(row.get(col)) if col else float("nan")
                if np.isfinite(value):
                    rows.append(_row(figure_id, panel_id, metric, cue, value, "value", row, seed_dir, path, repo_root, cue_condition=cue, cue_condition_order=_cue_order(cue), keep_fraction=row.get("keep_fraction", "")))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "cue_pixel_count", ["cue_condition", "metric"])


def build_s6_peak_cue_state_vs_cue_only_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["supp_structural_weak_cue_accuracy.csv", "supp_structural_weak_cue_memory_gain.csv", "panel_f_peak_cue_accuracy.csv", "panel_f_peak_cue_memory_gain.csv"]
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names)
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            cue = _normalize_cue(row.get("cue_condition", ""))
            if "memory_gain" in path.name:
                value = _float(row.get("memory_gain"))
                metric = "memory_gain"
                condition = cue
                unit = "percent_delta"
                value = _scale_delta_to_percent(value) if np.isfinite(value) else value
                memory_condition = "sequence_state_minus_cue_only"
            else:
                value = _float(row.get("P_target", row.get("accuracy")))
                metric = "P_target"
                condition = str(row.get("memory_condition", ""))
                unit = "percent"
                value = _to_percent_value(value) if np.isfinite(value) else value
                memory_condition = condition
            if np.isfinite(value):
                rows.append(_row(figure_id, panel_id, metric, condition, value, unit, row, seed_dir, path, repo_root, cue_condition=cue, cue_condition_order=_cue_order(cue), memory_condition=memory_condition, keep_fraction=row.get("keep_fraction", ""), x_value=row.get("keep_fraction", _cue_order(cue)), y_value=value))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "P_target", ["cue_condition", "memory_condition"])


def build_s6_peak_cue_serial_position_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["supp_peak_cue_serial_position_metrics.csv", "supp_peak_cue_serial_position_gain.csv"]
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names + ["panel_f_peak_cue_trial_readout.csv"], raw_names=["panel_f_peak_cue_trial_readout.csv"])
    rows: list[dict[str, Any]] = []
    for seed_dir in seeds:
        metrics_dir = seed_dir / "data" / "metrics"
        raw_dir = seed_dir / "data" / "raw"
        path = _first_existing(metrics_dir, names)
        if path is None:
            raw_path = raw_dir / "panel_f_peak_cue_trial_readout.csv"
            if raw_path.exists():
                path = raw_path
                df = _serial_gain_from_raw(pd.read_csv(raw_path))
            else:
                continue
        else:
            df = pd.read_csv(path)
        for _, row in df.iterrows():
            if "gain" in path.name:
                value = _float(row.get("memory_gain"))
                metric = "memory_gain"
                condition = _normalize_cue(row.get("cue_condition", ""))
            else:
                value = _float(row.get("P_target"))
                metric = "P_target"
                condition = str(row.get("memory_condition", ""))
            if np.isfinite(value):
                y = _scale_delta_to_percent(value) if metric == "memory_gain" else _to_percent_value(value)
                rows.append(_row(figure_id, panel_id, metric, condition, y, "percent_delta" if metric == "memory_gain" else "percent", row, seed_dir, path, repo_root, cue_condition=_normalize_cue(row.get("cue_condition", "")), target_position=row.get("target_position", ""), target_position_bin=row.get("target_position_bin", ""), relative_position=row.get("relative_position", ""), keep_fraction=row.get("keep_fraction", ""), x_value=row.get("target_position", row.get("relative_position", "")), y_value=y))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "memory_gain", ["target_position_bin", "cue_condition"])


def build_s6_weak_probe_position_stratified_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["panel_e_weak_probe_position_stratified_metrics.csv"]
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names)
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            warnings.append(f"Missing weak-probe position-stratified source under {_display(seed_dir, repo_root)}")
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            memory = str(row.get("memory_condition", ""))
            if memory not in set(MEMORY_ORDER):
                continue
            value = _float(row.get("P_target"))
            keep = _float(row.get("keep_prob"))
            if np.isfinite(value):
                y = _to_percent_value(value)
                rows.append(
                    _row(
                        figure_id,
                        panel_id,
                        "P_target",
                        memory,
                        y,
                        "percent",
                        row,
                        seed_dir,
                        path,
                        repo_root,
                        memory_condition=memory,
                        memory_condition_order={name: idx for idx, name in enumerate(MEMORY_ORDER, start=1)}.get(memory, 99),
                        display_condition=MEMORY_DISPLAY.get(memory, memory.replace("_", " ").title()),
                        target_position_bin=row.get("target_position_bin", ""),
                        keep_prob=keep if np.isfinite(keep) else "",
                        x_value=keep if np.isfinite(keep) else "",
                        y_value=y,
                        seq_len=row.get("seq_len", ""),
                        n_trials=row.get("n_trials", ""),
                    )
                )
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "P_target", ["target_position_bin", "memory_condition"])


def build_s6_region_ping_current_matching_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["panel_f_region_ping_current_matching.csv"]
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names)
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            warnings.append(f"Missing region-ping current matching source under {_display(seed_dir, repo_root)}")
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            region = _normalize_region(row.get("region_condition", ""))
            for metric in ("active_unit_count_mean", "total_ping_current_mean", "active_unit_count_std", "total_ping_current_std"):
                value = _float(row.get(metric))
                if np.isfinite(value):
                    rows.append(
                        _row(
                            figure_id,
                            panel_id,
                            metric,
                            region,
                            value,
                            "current" if "current" in metric else "units",
                            row,
                            seed_dir,
                            path,
                            repo_root,
                            region_condition=region,
                            region_condition_order=_region_order(region),
                            support_metric=row.get("support_metric", ""),
                            region_q=row.get("region_q", ""),
                            x_value=_region_order(region),
                            y_value=value,
                            n_trials=row.get("n_trials", ""),
                        )
                    )
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "active_unit_count_mean", ["region_condition", "metric"])


def build_s6_region_ping_s0_control_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["panel_f_region_ping_position_distribution.csv", "panel_f_region_ping_summary.csv"]
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names)
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            warnings.append(f"Missing S0 region-ping control source under {_display(seed_dir, repo_root)}")
            continue
        df = pd.read_csv(path)
        if "state_condition" in df.columns:
            df = df[df["state_condition"].astype(str).eq("S0")].copy()
        elif "memory_condition" in df.columns:
            df = df[df["memory_condition"].astype(str).eq("no_memory")].copy()
        if df.empty:
            warnings.append(f"S0/no_memory rows absent in {_display(path, repo_root)}")
            continue
        if "serial_bin" in df.columns and "readout_mass" in df.columns:
            for _, row in df.iterrows():
                region = _normalize_region(row.get("region_condition", ""))
                value = _float(row.get("readout_mass"))
                if np.isfinite(value):
                    serial_bin = str(row.get("serial_bin", ""))
                    rows.append(
                        _row(
                            figure_id,
                            panel_id,
                            "readout_mass",
                            region,
                            value,
                            "probability",
                            row,
                            seed_dir,
                            path,
                            repo_root,
                            region_condition=region,
                            region_condition_order=_region_order(region),
                            serial_bin=serial_bin,
                            serial_position=_serial_position(serial_bin) or "",
                            serial_bin_order=_serial_bin_order(serial_bin, row.get("seq_len", "")),
                            state_condition="S0",
                            memory_condition=row.get("memory_condition", "no_memory"),
                            x_value=_serial_bin_order(serial_bin, row.get("seq_len", "")),
                            y_value=value,
                            seq_len=row.get("seq_len", ""),
                            n_trials=row.get("n_trials", ""),
                        )
                    )
        else:
            for _, row in df.iterrows():
                region = _normalize_region(row.get("region_condition", ""))
                for metric in ("P_seen_item", "P_silent", "P_unseen"):
                    value = _float(row.get(metric))
                    if np.isfinite(value):
                        rows.append(_row(figure_id, panel_id, metric, region, value, "probability", row, seed_dir, path, repo_root, region_condition=region, region_condition_order=_region_order(region), state_condition="S0", x_value=_region_order(region), y_value=value, n_trials=row.get("n_trials", "")))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "readout_mass", ["region_condition", "serial_bin"])


def build_s6_region_ping_latency_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["supp_region_ping_amp_sweep_latency.csv", "panel_f_region_ping_summary.csv"]
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names)
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            warnings.append(f"Missing region-ping latency source under {_display(seed_dir, repo_root)}")
            continue
        if path.name == "panel_f_region_ping_summary.csv":
            warnings.append("Region-ping amp sweep missing; falling back to panel_f_region_ping_summary.csv latency.")
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            region = _normalize_region(row.get("region_condition", ""))
            amp = _float(row.get("ping_amp"))
            for metric in ("median_first_fire_time_ms", "P_fire_by_ping_end", "P_seen_item"):
                value = _float(row.get(metric))
                if np.isfinite(value):
                    rows.append(
                        _row(
                            figure_id,
                            panel_id,
                            metric,
                            region,
                            value,
                            "ms" if "time" in metric else "probability",
                            row,
                            seed_dir,
                            path,
                            repo_root,
                            region_condition=region,
                            region_condition_order=_region_order(region),
                            state_condition=row.get("state_condition", ""),
                            ping_amp=amp if np.isfinite(amp) else "",
                            x_value=amp if np.isfinite(amp) else _region_order(region),
                            y_value=value,
                            n_trials=row.get("n_trials", ""),
                        )
                    )
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "median_first_fire_time_ms", ["region_condition", "ping_amp"])


def build_s6_region_ping_amp_sweep_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["supp_region_ping_amp_sweep_summary.csv"]
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names)
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            warnings.append(f"Missing region-ping amplitude sweep source under {_display(seed_dir, repo_root)}")
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            region = _normalize_region(row.get("region_condition", ""))
            amp = _float(row.get("ping_amp"))
            for metric in ("P_seen_item", "P_silent", "P_latest_item"):
                value = _float(row.get(metric))
                if np.isfinite(value) and np.isfinite(amp):
                    rows.append(_row(figure_id, panel_id, metric, region, value, "probability", row, seed_dir, path, repo_root, region_condition=region, region_condition_order=_region_order(region), ping_amp=amp, x_value=amp, y_value=value, state_condition=row.get("state_condition", ""), n_trials=row.get("n_trials", "")))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "P_seen_item", ["region_condition", "ping_amp"])


def _landscape_nonflatness_from_npz(figure_id: str, panel_id: str, seed_dir: Path, path: Path, repo_root: Path) -> list[dict[str, Any]]:
    data = np.load(path)
    metric_name = "delta_gain_map" if "delta_gain_map" in data.files else "G_final" if "G_final" in data.files else ""
    if not metric_name:
        return []
    values = np.asarray(data[metric_name], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    support = values.copy()
    min_value = float(np.nanmin(support))
    if min_value < 0:
        support = support - min_value
    support = np.maximum(support, 0.0)
    total = float(support.sum())
    if total <= 0:
        support = np.abs(values)
        total = float(support.sum())
    if total <= 0:
        return []
    top_n = max(1, int(np.ceil(0.2 * support.size)))
    top_q_mass_fraction = float(np.sort(support)[-top_n:].sum() / total)
    mean = float(np.mean(support))
    metrics = {
        "top_q_mass_fraction": top_q_mass_fraction,
        "support_gini": _gini(support),
        "support_cv": float(np.std(support) / mean) if mean > 0 else float("nan"),
    }
    rows: list[dict[str, Any]] = []
    for metric, value in metrics.items():
        if np.isfinite(value):
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric=metric,
                    condition=metric,
                    layer="layer1",
                    seed_id=_seed_id(seed_dir),
                    value=value,
                    unit="value",
                    source_file=_display(path, repo_root),
                    fallback_scope="representative_landscape",
                    landscape_metric_used=metric_name,
                )
            )
    return rows


def _ping_recency_rows_from_distribution(
    df: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    seed_dir: Path,
    path: Path,
    repo_root: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if df.empty or "serial_bin" not in df.columns or "readout_mass" not in df.columns:
        return []
    work = df.copy()
    if "network_seed" not in work.columns:
        work["network_seed"] = _seed_id(seed_dir)
    if "state_condition" not in work.columns:
        work["state_condition"] = "S_final"
    elif work["state_condition"].astype(str).eq("S_final").any():
        work = work[work["state_condition"].astype(str).eq("S_final")].copy()
    else:
        warnings.append(f"S4E panel_d_ping_position_distribution.csv under {_display(seed_dir, repo_root)} has no S_final rows; using available states.")
    if "seq_len" not in work.columns:
        positions = [_serial_position(value) for value in work["serial_bin"]]
        positions = [pos for pos in positions if pos is not None]
        work["seq_len"] = max(positions) if positions else ""
    work["readout_mass"] = pd.to_numeric(work["readout_mass"], errors="coerce").fillna(0.0)
    work["serial_position"] = pd.to_numeric(work["serial_bin"].map(_serial_position), errors="coerce")
    rows: list[dict[str, Any]] = []
    group_cols = ["network_seed", "state_condition", "seq_len"]
    for (seed, state, seq_len), part in work.groupby(group_cols, dropna=False, sort=False):
        numeric = part.dropna(subset=["serial_position"]).copy()
        if numeric.empty:
            continue
        max_pos = int(pd.to_numeric(numeric["serial_position"], errors="coerce").max())
        recent_floor = max(1, max_pos - 2)
        latest = float(numeric.loc[numeric["serial_position"].eq(max_pos), "readout_mass"].sum())
        recent = float(numeric.loc[numeric["serial_position"].ge(recent_floor) & numeric["serial_position"].lt(max_pos), "readout_mass"].sum())
        earlier = float(numeric.loc[numeric["serial_position"].lt(recent_floor), "readout_mass"].sum())
        non_numeric = part[part["serial_position"].isna()].copy()
        labels = non_numeric["serial_bin"].astype(str).str.lower()
        silent = float(non_numeric.loc[labels.isin(["silent", "silence", "no_readout", "none"]), "readout_mass"].sum())
        other = float(non_numeric.loc[~labels.isin(["silent", "silence", "no_readout", "none"]), "readout_mass"].sum())
        values = {"latest": latest, "recent": recent, "earlier": earlier, "silent": silent + other}
        n_trials = _max_numeric(part.get("n_trials", pd.Series(dtype=float)))
        for readout_class in READOUT_CLASS_ORDER:
            value = float(values.get(readout_class, 0.0))
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="readout_mass",
                    condition=readout_class,
                    layer="layer3",
                    seed_id=seed,
                    value=value,
                    unit="probability",
                    source_file=_display(path, repo_root),
                    readout_class=readout_class,
                    readout_class_order=_readout_class_order(readout_class),
                    state_condition=str(state),
                    seq_len=seq_len,
                    latest_position=max_pos,
                    recent_positions=",".join(str(pos) for pos in range(recent_floor, max_pos)),
                    earlier_positions=",".join(str(pos) for pos in range(1, recent_floor)),
                    other_mass_included_in_silent=bool(other > 0),
                    n_trials=n_trials if np.isfinite(n_trials) else "",
                    x_value=_readout_class_order(readout_class),
                    y_value=value,
                )
            )
    return rows


def _ping_recency_rows_from_diagnostics(
    df: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    seed_dir: Path,
    path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    if "state_condition" in work.columns and work["state_condition"].astype(str).eq("S_final").any():
        work = work[work["state_condition"].astype(str).eq("S_final")].copy()
    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        latest = _float(row.get("latest_item_mass"))
        recent = _float(row.get("recent_item_mass"))
        earlier = _float(row.get("earlier_item_residual_mass"))
        silent = _float(row.get("P_silent"))
        values = {
            "latest": latest if np.isfinite(latest) else 0.0,
            "recent": recent if np.isfinite(recent) else 0.0,
            "earlier": earlier if np.isfinite(earlier) else 0.0,
            "silent": silent if np.isfinite(silent) else 0.0,
        }
        for readout_class in READOUT_CLASS_ORDER:
            value = float(values[readout_class])
            rows.append(
                _row(
                    figure_id,
                    panel_id,
                    "readout_mass",
                    readout_class,
                    value,
                    "probability",
                    row,
                    seed_dir,
                    path,
                    repo_root,
                    readout_class=readout_class,
                    readout_class_order=_readout_class_order(readout_class),
                    state_condition=row.get("state_condition", ""),
                    seq_len=row.get("seq_len", ""),
                    x_value=_readout_class_order(readout_class),
                    y_value=value,
                    degraded_from_summary=True,
                )
            )
    return rows


def _weak_probe_recency_gain_rows(
    df: pd.DataFrame,
    figure_id: str,
    panel_id: str,
    seed_dir: Path,
    path: Path,
    repo_root: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    required = {"target_position_bin", "memory_condition", "P_target"}
    if df.empty or not required.issubset(df.columns):
        return []
    work = df.copy()
    if "network_seed" not in work.columns:
        work["network_seed"] = _seed_id(seed_dir)
    if "seq_len" not in work.columns:
        work["seq_len"] = ""
    work["P_target"] = pd.to_numeric(work["P_target"], errors="coerce")
    work["n_trials"] = pd.to_numeric(work.get("n_trials", pd.Series(np.ones(len(work)))), errors="coerce").fillna(1.0)
    rows: list[dict[str, Any]] = []
    warned_cue_fallback = False
    group_cols = ["network_seed", "seq_len", "target_position_bin"]
    for (seed, seq_len, target_bin), part in work.groupby(group_cols, dropna=False, sort=False):
        target_bin = str(target_bin)
        if not target_bin or target_bin.lower() == "nan":
            continue
        means: dict[str, float] = {}
        n_by_memory: dict[str, float] = {}
        for memory, memory_part in part.groupby("memory_condition", dropna=False, sort=False):
            memory = str(memory)
            values = pd.to_numeric(memory_part["P_target"], errors="coerce")
            weights = pd.to_numeric(memory_part["n_trials"], errors="coerce").fillna(1.0)
            valid = values.notna()
            if valid.any():
                means[memory] = _weighted_mean(values[valid].to_numpy(dtype=float), weights[valid].to_numpy(dtype=float))
                n_by_memory[memory] = float(weights[valid].sum())
        if "sequence_state" not in means:
            continue
        baseline = "single_item_memory" if "single_item_memory" in means else "cue_only" if "cue_only" in means else ""
        if not baseline:
            continue
        if baseline == "cue_only" and not warned_cue_fallback:
            warnings.append(f"S4F fallback used sequence_state - cue_only under {_display(seed_dir, repo_root)} because single_item_memory rows are missing.")
            warned_cue_fallback = True
        gain = means["sequence_state"] - means[baseline]
        y = _scale_delta_to_percent(gain)
        rows.append(
            _canonical(
                figure_id,
                panel_id,
                metric="target_recovery_gain",
                condition=target_bin,
                layer="layer3",
                seed_id=seed,
                value=y,
                unit="percent_delta",
                source_file=_display(path, repo_root),
                target_position_bin=target_bin,
                target_position_bin_order=_target_position_bin_order(target_bin),
                memory_condition="sequence_state_minus_" + baseline,
                baseline_condition=baseline,
                sequence_state_P_target=_to_percent_value(means["sequence_state"]),
                baseline_P_target=_to_percent_value(means[baseline]),
                x_value=_target_position_bin_order(target_bin),
                y_value=y,
                seq_len=seq_len,
                n_trials_sequence_state=n_by_memory.get("sequence_state", ""),
                n_trials_baseline=n_by_memory.get(baseline, ""),
            )
        )
    return rows


def _gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    arr = np.sort(np.maximum(arr, 0.0))
    total = float(arr.sum())
    if total <= 0:
        return float("nan")
    index = np.arange(1, arr.size + 1, dtype=float)
    return float((2.0 * np.sum(index * arr) / (arr.size * total)) - ((arr.size + 1.0) / arr.size))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _readout_class_order(readout_class: str) -> int:
    return {name: idx for idx, name in enumerate(READOUT_CLASS_ORDER, start=1)}.get(str(readout_class), 99)


def _target_position_bin_order(target_bin: str) -> int:
    return {name: idx for idx, name in enumerate(TARGET_POSITION_BIN_ORDER, start=1)}.get(str(target_bin), 99)


def _part2_finish(
    spec: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    panel_df: pd.DataFrame,
    source_paths: Sequence[Path],
    warnings: Sequence[str],
    group_cols: Sequence[str],
) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    if panel_df.empty:
        return _part2_missing(spec, output_dir, source_paths, list(warnings) + [f"{figure_id}{panel_id}: no usable existing rows."])
    panel_df = panel_df.copy()
    panel_df["network_id"] = panel_df["network_id"].astype(str)
    panel_df["seed_id"] = panel_df.get("seed_id", panel_df["network_id"]).astype(str)
    network_ids = sorted(panel_df["network_id"].dropna().unique().tolist())
    panel_df["run_mode"] = "multi_network_final" if len(network_ids) > 1 else "single_network_draft"
    panel_df["n_networks"] = len(network_ids)
    keys = ["network_id", *[col for col in group_cols if col in panel_df.columns]]
    assert not panel_df.duplicated(keys).any()
    warning_list = list(dict.fromkeys(map(str, warnings)))
    expected = int(spec.get("expected_networks", len(network_ids)))
    if len(network_ids) != expected:
        warning_list.append(f"Expected {expected} networks, observed {len(network_ids)}.")
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": ";".join(map(str, panel_df["metric"].dropna().unique())),
        "run_mode": "multi_network_final" if len(network_ids) > 1 else "single_network_draft",
        "n_networks": len(network_ids),
        "n_networks_observed": len(network_ids),
        "network_ids": network_ids,
        "inferential_unit": "independent network",
        "replicate_unit": "network_id",
        "interval_definition": "two-sided 95% Student-t confidence interval across independent networks",
        "rows_before_network_aggregation": int(pd.to_numeric(panel_df.get("lower_level_rows", pd.Series([1] * len(panel_df))), errors="coerce").fillna(1).sum()),
        "rows_after_network_aggregation": len(panel_df),
        "adapter_performed_network_level_averaging": bool(pd.to_numeric(panel_df.get("lower_level_rows", pd.Series([1])), errors="coerce").fillna(1).max() > 1),
        "summaries": summarize_values(panel_df, [col for col in group_cols if col in panel_df.columns]),
        "network_summaries": _part2_network_summaries(panel_df, group_cols),
        "values_used_for_plotting": [float(value) for value in pd.to_numeric(panel_df["value"], errors="coerce").dropna()],
        "warnings": warning_list,
    }
    source_entries = [_part2_source(path, repo_root) for path in source_paths]
    manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "ok",
        "run_mode": stats["run_mode"],
        "n_networks": len(network_ids),
        "network_ids": network_ids,
        "inferential_unit": "independent network",
        "source_files_used": [entry["path"] for entry in source_entries if entry["exists"]],
        "sources": source_entries,
        "checked_candidates": [entry["path"] for entry in source_entries],
        "warnings": warning_list,
    }
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warning_list)


def _part2_missing(spec: Mapping[str, Any], output_dir: Path, source_paths: Sequence[Path], warnings: Sequence[str]) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    reason = str(warnings[-1]) if warnings else "Missing source"
    panel_df = pd.DataFrame([_canonical(figure_id, panel_id, metric="missing_source", condition="missing", layer="", seed_id="", value=np.nan, unit="", source_file="", placeholder_reason=reason)])
    stats = {"figure_id": figure_id, "panel_id": panel_id, "status": "missing_source", "values_used_for_plotting": [], "warnings": list(warnings)}
    manifest = {"figure_id": figure_id, "panel_id": panel_id, "status": "missing_source", "source_files_used": [], "sources": [{"path": str(path), "exists": path.exists()} for path in source_paths], "warnings": list(warnings)}
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, list(warnings))


def _part2_network_summaries(df: pd.DataFrame, group_cols: Sequence[str]) -> list[dict[str, Any]]:
    groups = [col for col in group_cols if col in df.columns]
    grouped = df.groupby(groups, dropna=False, sort=False) if groups else [((), df)]
    rows: list[dict[str, Any]] = []
    for key, part in grouped:
        values = pd.to_numeric(part["value"], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        if not isinstance(key, tuple):
            key = (key,)
        n = int(len(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        half = float(scipy_stats.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
        row = {col: value for col, value in zip(groups, key)}
        row.update({
            "n_networks": n,
            "mean": float(np.mean(values)),
            "sem": sem,
            "ci95_low": float(np.mean(values) - half),
            "ci95_high": float(np.mean(values) + half),
            "one_sample_p_vs_zero": _one_sample_p(values),
        })
        rows.append(row)
    return rows


def _resolve_repo_path(repo_root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _part2_source(path: Path, repo_root: Path) -> dict[str, Any]:
    exists = path.is_file()
    size_bytes = path.stat().st_size if exists else 0
    return {
        "path": _display(path, repo_root),
        "exists": exists,
        "size_bytes": size_bytes,
        "bytes": size_bytes,
        "sha256": _sha256(path) if exists else "",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_sample_p(values: np.ndarray) -> float | None:
    if len(values) <= 1:
        return None
    if float(np.std(values, ddof=1)) < 1e-15:
        return 1.0 if abs(float(np.mean(values))) < 1e-15 else 0.0
    return float(scipy_stats.ttest_1samp(values, 0.0).pvalue)


def _build_region_support(spec: Mapping[str, Any], repo_root: Path, output_dir: Path, names: Sequence[str]) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            sources.extend(_source(seed_dir / "data" / "metrics" / name, repo_root, seed_dir) for name in names)
            continue
        sources.append(_source(path, repo_root, seed_dir))
        if path.name == "panel_c_example_landscape_summary.csv":
            warnings.append(f"S4A degraded morphology fallback used representative panel_c_example_landscape_summary.csv under {_display(seed_dir, repo_root)}.")
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            made_region = False
            for region, col in (("valley", "valley_mean_support"), ("random", "random_mean_support"), ("peak", "peak_mean_support")):
                value = _float(row.get(col))
                if np.isfinite(value):
                    made_region = True
                    rows.append(_row(figure_id, panel_id, "support", region, value, "support", row, seed_dir, path, repo_root, region=region, cue_condition=region, x_value=_cue_order(region), y_value=value))
            if not made_region:
                value = _float(row.get("peak_valley_delta"))
                if np.isfinite(value):
                    rows.append(_row(figure_id, panel_id, "peak_valley_delta", "peak_minus_valley", value, "support_delta", row, seed_dir, path, repo_root, y_value=value))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "support", ["region"])


def _start(spec: Mapping[str, Any], repo_root: Path, metric_names: Sequence[str], *, raw_names: Sequence[str] = ()) -> tuple[str, str, Path, list[Path], list[str], list[dict[str, Any]]]:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        for name in metric_names:
            base = seed_dir / ("data/raw" if name in raw_names else "data/metrics")
            sources.append(_source(base / name, repo_root, seed_dir))
    return figure_id, panel_id, root, seeds, warnings, sources


def _seed_paths(seeds: Sequence[Path], names: Sequence[str]) -> list[tuple[Path, Path | None]]:
    out: list[tuple[Path, Path | None]] = []
    for seed_dir in seeds:
        out.append((seed_dir, _first_existing(seed_dir / "data" / "metrics", names)))
    return out


def _row(figure_id: str, panel_id: str, metric: str, condition: str, value: float, unit: str, source: Mapping[str, Any], seed_dir: Path, path: Path, repo_root: Path, **extra: Any) -> dict[str, Any]:
    return _canonical(
        figure_id,
        panel_id,
        metric=metric,
        condition=condition,
        layer=str(source.get("layer", "")),
        seed_id=source.get("network_seed", _seed_id(seed_dir)),
        value=value,
        unit=unit,
        source_file=_display(path, repo_root),
        **extra,
    )


def _recency_from_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "serial_bin" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    work = df.copy()
    work["serial_position"] = work["serial_bin"].astype(str).str.extract(r"pos_(\d+)")[0].astype(float)
    for (seed, state, seq_len), part in work.groupby(["network_seed", "state_condition", "seq_len"], dropna=False):
        numeric = part.dropna(subset=["serial_position"])
        if numeric.empty:
            continue
        max_pos = numeric["serial_position"].max()
        masses = pd.to_numeric(numeric["readout_mass"], errors="coerce").fillna(0.0)
        total = masses.sum()
        recent = numeric.loc[numeric["serial_position"].ge(max_pos - 2), "readout_mass"].sum()
        rows.append(
            {
                "network_seed": seed,
                "state_condition": state,
                "seq_len": seq_len,
                "latest_item_mass": numeric.loc[numeric["serial_position"].eq(max_pos), "readout_mass"].mean(),
                "recent_item_mass": recent,
                "earlier_item_residual_mass": total - recent,
                "ping_COM": (numeric["serial_position"] * numeric["readout_mass"]).sum() / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _serial_gain_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    if not {"cue_condition", "memory_condition", "target_position"}.issubset(df.columns):
        return pd.DataFrame()
    value_col = "P_target" if "P_target" in df.columns else "target_hit"
    if value_col not in df.columns:
        return pd.DataFrame()
    group_cols = [col for col in ["network_seed", "target_position", "target_position_bin", "relative_position", "cue_condition", "memory_condition", "keep_fraction"] if col in df.columns]
    agg = df.groupby(group_cols, dropna=False)[value_col].mean().reset_index()
    index_cols = [col for col in group_cols if col != "memory_condition"]
    pivot = agg.pivot_table(index=index_cols, columns="memory_condition", values=value_col, aggfunc="mean").reset_index()
    if {"sequence_state", "cue_only"}.issubset(pivot.columns):
        pivot["memory_gain"] = pivot["sequence_state"] - pivot["cue_only"]
    return pivot
