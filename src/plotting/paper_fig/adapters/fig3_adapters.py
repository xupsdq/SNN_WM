from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult, summarize_values, write_adapter_outputs


FIGURE_ID = "fig3"
DEFAULT_EXPERIMENT_ROOT = Path("results/paper_experiments/fig3_multiitem_peak_landscape")
DRAFT_WARNING = "Single-network result. Use for pipeline validation only, not final manuscript statistics."
STATE_ORDER = ("S_final", "S0", "S0_ping_null")
CUE_ORDER = ("valley", "random", "peak")
MEMORY_ORDER = ("cue_only", "single_item_memory", "sequence_state")
MEMORY_DISPLAY = {
    "cue_only": "No memory",
    "single_item_memory": "Single-item memory",
    "sequence_state": "Sequence state",
}
REGION_ORDER = ("peak", "valley", "random")


def build_fig3_progressive_update_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_b_progressive_update_metrics.csv"
        sources.append(_source(path, repo_root, seed_dir))
        if not path.exists():
            warnings.append(f"Missing progressive-update source: {_display(path, repo_root)}")
            continue
        df = _primary(pd.read_csv(path), layer="layer1", state_variable="g")
        for _, row in df.iterrows():
            value = _float(row.get("stepwise_update_ratio"))
            if not np.isfinite(value):
                continue
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="stepwise_update_ratio",
                    condition="sequence_update",
                    layer=str(row.get("layer", "layer1")),
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    value=value,
                    unit="ratio",
                    source_file=_display(path, repo_root),
                    sequence_id=row.get("sequence_id", ""),
                    seq_len=row.get("seq_len", ""),
                    stage_k=row.get("stage_k", ""),
                    state_variable=row.get("state_variable", "g"),
                    x_value=row.get("stage_k", ""),
                    y_value=value,
                    state_displacement=row.get("state_displacement", ""),
                    anchor_COM=row.get("anchor_COM", ""),
                    similarity_entropy=row.get("similarity_entropy", ""),
                )
            )
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "stepwise_update_ratio", ["stage_k"])


def build_fig3_example_landscape_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    if not seeds:
        return _missing(spec, repo_root, output_dir, f"No Fig.3 seed directories found under {_display(root, repo_root)}.", [], [])
    seed_dir = seeds[0]
    npz_path = seed_dir / "data" / "raw" / "panel_c_example_landscape.npz"
    summary_path = seed_dir / "data" / "metrics" / "panel_c_example_landscape_summary.csv"
    metadata_path = seed_dir / "data" / "raw" / "panel_c_example_landscape_metadata.json"
    sources = [_source(path, repo_root, seed_dir) for path in (npz_path, summary_path, metadata_path)]
    if not npz_path.exists():
        return _missing(spec, repo_root, output_dir, f"Missing representative landscape source: {_display(npz_path, repo_root)}", sources, warnings)
    data = np.load(npz_path)
    used_metric = "delta_gain_map" if "delta_gain_map" in data.files else "G_final"
    values = np.asarray(data[used_metric], dtype=float)
    peak = np.asarray(data["peak_mask"], dtype=bool) if "peak_mask" in data.files else np.zeros_like(values, dtype=bool)
    valley = np.asarray(data["valley_mask"], dtype=bool) if "valley_mask" in data.files else np.zeros_like(values, dtype=bool)
    random = np.asarray(data["random_matched_mask"], dtype=bool) if "random_matched_mask" in data.files else np.zeros_like(values, dtype=bool)
    metadata = _read_json(metadata_path)
    rows: list[dict[str, Any]] = []
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            mask_role = "none"
            if peak[y, x]:
                mask_role = "peak"
            elif valley[y, x]:
                mask_role = "valley"
            elif random[y, x]:
                mask_role = "random"
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric=used_metric,
                    condition="final_support_landscape",
                    layer=str(metadata.get("layer", "layer1")),
                    seed_id=metadata.get("network_seed", _seed_id(seed_dir)),
                    value=float(values[y, x]),
                    unit="support_delta" if used_metric == "delta_gain_map" else "support",
                    source_file=_display(npz_path, repo_root),
                    sequence_id=metadata.get("sequence_id", ""),
                    seq_len=metadata.get("seq_len", ""),
                    state_variable=str(metadata.get("state_variable", "g")),
                    row=y,
                    col=x,
                    x_value=x,
                    y_value=y,
                    z_value=float(values[y, x]),
                    mask_role=mask_role,
                )
            )
    stats_extra = {
        "landscape_metric_used": used_metric,
        "selected_sequence_id": metadata.get("sequence_id"),
        "selected_seq_len": metadata.get("seq_len"),
        "source_priority_satisfied": _display(npz_path, repo_root),
        "has_summary_inset": False,
    }
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, used_metric, ["row", "col"], stats_extra=stats_extra)


def build_fig3_neutral_ping_serial_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    summary_frames: list[pd.DataFrame] = []
    for seed_dir in seeds:
        metrics_dir = seed_dir / "data" / "metrics"
        raw_dir = seed_dir / "data" / "raw"
        position_path = _first_existing(
            metrics_dir,
            [
                "panel_c_neutral_ping_position_distribution.csv",
                "panel_d_ping_position_distribution.csv",
                "panel_e_ping_position_distribution.csv",
            ],
        )
        summary_path = _first_existing(
            metrics_dir,
            [
                "panel_c_neutral_ping_access_summary.csv",
                "panel_d_ping_summary.csv",
                "panel_e_ping_summary.csv",
            ],
        )
        raw_path = _first_existing(
            raw_dir,
            [
                "panel_c_neutral_ping_access_readout.csv",
                "panel_d_neutral_ping_trial_readout.csv",
            ],
        )
        checked = [
            metrics_dir / "panel_c_neutral_ping_position_distribution.csv",
            metrics_dir / "panel_c_neutral_ping_access_summary.csv",
            raw_dir / "panel_c_neutral_ping_access_readout.csv",
            metrics_dir / "panel_d_ping_position_distribution.csv",
            metrics_dir / "panel_d_ping_summary.csv",
            raw_dir / "panel_d_neutral_ping_trial_readout.csv",
            metrics_dir / "panel_e_ping_position_distribution.csv",
            metrics_dir / "panel_e_ping_summary.csv",
        ]
        sources.extend(_source(path, repo_root, seed_dir) for path in checked)
        if position_path is None and raw_path is not None and raw_path.exists():
            position_df = _ping_position_from_raw(pd.read_csv(raw_path))
            position_source = raw_path
        elif position_path is not None:
            position_df = pd.read_csv(position_path)
            position_source = position_path
            if position_path.name.startswith("panel_e_"):
                warnings.append(f"Fig.3D using old panel_e neutral-ping alias: {_display(position_path, repo_root)}")
        else:
            warnings.append(f"Missing neutral-ping position source under {_display(seed_dir, repo_root)}")
            continue
        if summary_path is not None:
            summary_frames.append(pd.read_csv(summary_path))
            if summary_path.name.startswith("panel_e_"):
                warnings.append(f"Fig.3D using old panel_e ping-summary alias: {_display(summary_path, repo_root)}")
        for _, row in position_df.iterrows():
            mass = _float(row.get("readout_mass"))
            if not np.isfinite(mass):
                continue
            serial_bin = str(row.get("serial_bin", row.get("serial_position", "")))
            serial_position = _serial_position(row.get("serial_position", serial_bin))
            state = _normalize_state(row.get("state_condition", row.get("condition", "")))
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="readout_mass",
                    condition=state,
                    layer="layer3",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    value=mass,
                    unit="probability",
                    source_file=_display(position_source, repo_root),
                    state_condition=state,
                    serial_bin=serial_bin,
                    serial_position=serial_position if serial_position is not None else "",
                    x_value=serial_position if serial_position is not None else "",
                    y_value=mass,
                    seq_len=row.get("seq_len", ""),
                    n_trials=row.get("n_trials", ""),
                    plot_include=serial_position is not None,
                )
            )
    panel_df = pd.DataFrame(rows)
    stats_extra = _ping_stats(panel_df, summary_frames)
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "readout_mass", ["state_condition", "serial_position"], stats_extra=stats_extra)


def build_fig3_weak_probe_completion_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    auc_frames: list[pd.DataFrame] = []
    gain_frames: list[pd.DataFrame] = []
    for seed_dir in seeds:
        metrics_dir = seed_dir / "data" / "metrics"
        raw_dir = seed_dir / "data" / "raw"
        metrics_path = metrics_dir / "panel_e_weak_probe_metrics.csv"
        gain_path = metrics_dir / "panel_e_weak_probe_memory_gain.csv"
        auc_path = _first_existing(metrics_dir, ["panel_e_weak_probe_auc_metrics.csv", "panel_f_weak_probe_auc_metrics.csv"])
        raw_path = raw_dir / "panel_e_weak_probe_trial_readout.csv"
        legacy_metrics_path = metrics_dir / "panel_f_weak_probe_metrics.csv"
        checked = [
            metrics_dir / "panel_e_weak_probe_metrics.csv",
            metrics_dir / "panel_e_weak_probe_memory_gain.csv",
            metrics_dir / "panel_e_weak_probe_auc_metrics.csv",
            raw_path,
            metrics_dir / "panel_f_weak_probe_metrics.csv",
            metrics_dir / "panel_f_weak_probe_memory_gain.csv",
            metrics_dir / "panel_f_weak_probe_auc_metrics.csv",
        ]
        sources.extend(_source(path, repo_root, seed_dir) for path in checked)
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            metrics_source = metrics_path
        elif raw_path.exists():
            metrics_df = _weak_probe_from_raw(pd.read_csv(raw_path))
            metrics_source = raw_path
        elif legacy_metrics_path.exists():
            metrics_df = pd.read_csv(legacy_metrics_path)
            metrics_source = legacy_metrics_path
            warnings.append("Fig.3E missing panel_e weak-probe output; using legacy two-condition weak-probe output.")
            warnings.append(f"Fig.3E using old panel_f weak-probe alias: {_display(legacy_metrics_path, repo_root)}")
        else:
            warnings.append(f"Missing weak-probe completion source under {_display(seed_dir, repo_root)}")
            continue
        metrics_df = _prefer_sequence_member_targets(metrics_df, warnings, "Fig.3E")
        memories_present = set(metrics_df.get("memory_condition", pd.Series(dtype=str)).dropna().astype(str))
        if "single_item_memory" not in memories_present:
            warnings.append("Fig.3E missing single_item_memory; using legacy two-condition weak-probe output.")
        for _, row in metrics_df.iterrows():
            memory = str(row.get("memory_condition", ""))
            if memory not in set(MEMORY_ORDER):
                continue
            keep = _float(row.get("keep_prob", row.get("keep_fraction")))
            p_target = _float(row.get("P_target"))
            if not np.isfinite(keep) or not np.isfinite(p_target):
                continue
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="P_target",
                    condition=memory,
                    layer="layer3",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    value=_to_percent_value(p_target),
                    unit="percent",
                    source_file=_display(metrics_source, repo_root),
                    memory_condition=memory,
                    memory_condition_order=_memory_order(memory),
                    display_condition=MEMORY_DISPLAY.get(memory, memory.replace("_", " ").title()),
                    state_condition=_normalize_state(row.get("state_condition", "")),
                    keep_prob=keep,
                    x_value=keep,
                    y_value=_to_percent_value(p_target),
                    target_source=row.get("target_source", ""),
                    seq_len=row.get("seq_len", ""),
                    n_trials=row.get("n_trials", ""),
                    P_target_fraction=p_target,
                )
            )
        if gain_path.exists():
            gain_df = _prefer_sequence_member_targets(pd.read_csv(gain_path), warnings, "Fig.3E memory-gain")
            gain_df["source_file"] = _display(gain_path, repo_root)
            gain_frames.append(gain_df)
        if auc_path is not None:
            auc_frames.append(pd.read_csv(auc_path))
    panel_df = pd.DataFrame(rows)
    stats_extra = _weak_probe_stats(auc_frames, gain_frames, panel_df)
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "P_target", ["memory_condition", "keep_prob"], stats_extra=stats_extra)


def build_fig3_region_ping_readout_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    summary_frames: list[pd.DataFrame] = []
    contrast_frames: list[pd.DataFrame] = []
    current_frames: list[pd.DataFrame] = []
    for seed_dir in seeds:
        metrics_dir = seed_dir / "data" / "metrics"
        raw_dir = seed_dir / "data" / "raw"
        distribution_path = metrics_dir / "panel_f_region_ping_position_distribution.csv"
        summary_path = metrics_dir / "panel_f_region_ping_summary.csv"
        contrast_path = metrics_dir / "panel_f_region_ping_contrast.csv"
        current_path = metrics_dir / "panel_f_region_ping_current_matching.csv"
        raw_path = raw_dir / "panel_f_region_ping_trial_readout.csv"
        checked = [distribution_path, summary_path, contrast_path, current_path, raw_path]
        sources.extend(_source(path, repo_root, seed_dir) for path in checked)
        if distribution_path.exists():
            dist_df = pd.read_csv(distribution_path)
            dist_source = distribution_path
        elif raw_path.exists():
            dist_df = _region_ping_distribution_from_raw(pd.read_csv(raw_path))
            dist_source = raw_path
        else:
            warnings.append("New Fig.3F requires panel_f_region_ping_* outputs. Legacy peak-cue outputs were not used for main Fig.3F.")
            continue
        if summary_path.exists():
            summary_frames.append(pd.read_csv(summary_path))
        if contrast_path.exists():
            contrast_frames.append(pd.read_csv(contrast_path))
        if current_path.exists():
            current_frames.append(pd.read_csv(current_path))
        dist_df = _filter_region_ping_main(dist_df)
        rows.extend(
            _aggregate_region_ping_readout_rows(
                dist_df,
                figure_id=figure_id,
                panel_id=panel_id,
                seed_dir=seed_dir,
                source_file=_display(dist_source, repo_root),
            )
        )
    panel_df = pd.DataFrame(rows)
    stats_extra = _region_ping_stats(panel_df, summary_frames, contrast_frames, current_frames)
    stats_extra["main_source"] = "region_ping"
    other_mass = float(stats_extra.get("other_mass", 0.0) or 0.0)
    total_mass = sum(float(v) for v in (stats_extra.get("total_readout_mass_by_region") or {}).values())
    if other_mass > 0 and total_mass > 0 and other_mass / total_mass > 0.05:
        warnings.append(f"Fig.3F included large other/unseen mass in Silent: {other_mass:.3g} of {total_mass:.3g}.")
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "readout_mass", ["region_condition", "readout_category"], stats_extra=stats_extra)


def build_fig3_boundary_morphology_profile_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    focus_seq_len = _optional_int(spec.get("focus_seq_len", spec.get("seq_len_focus", None)))
    focus_delay_ms = _optional_float(spec.get("focus_delay_ms", spec.get("delay_ms_focus", None)))
    layer = str(spec.get("morphology_layer") or "layer1")
    state_variable = str(spec.get("state_variable") or "g")
    for seed_dir in seeds:
        path = seed_dir / "data" / "raw" / "panel_b_morphology_item_support.csv"
        sources.append(_source(path, repo_root, seed_dir))
        if not path.exists():
            warnings.append(f"Missing morphology item-support source: {_display(path, repo_root)}")
            continue
        df = _primary(pd.read_csv(path), layer=layer, state_variable=state_variable)
        df = _filter_focus(df, seq_len=focus_seq_len, delay_ms=focus_delay_ms)
        for _, row in df.iterrows():
            value = _float(row.get("p_i"))
            if not np.isfinite(value):
                continue
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="morphology_support_mass",
                    condition=str(row.get("condition_id", "")),
                    layer=str(row.get("layer", "layer1")),
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    value=value,
                    unit="mass",
                    source_file=_display(path, repo_root),
                    sequence_id=row.get("sequence_id", ""),
                    seq_len=row.get("seq_len", ""),
                    delay_ms=row.get("delay_ms", ""),
                    serial_position=row.get("serial_position", ""),
                    x_value=row.get("serial_position", ""),
                    y_value=value,
                    beta=row.get("beta", ""),
                    p_i=value,
                    multi_item_retention_index=row.get("multi_item_retention_index", ""),
                    latest_collapse_index=row.get("latest_collapse_index", ""),
                    reconstruction_R2=row.get("reconstruction_R2", ""),
                )
            )
    stats_extra = {"focus_seq_len": focus_seq_len, "focus_delay_ms": focus_delay_ms, "morphology_layer": layer, "state_variable": state_variable}
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "morphology_support_mass", ["seq_len", "delay_ms", "serial_position"], stats_extra=stats_extra)


def build_fig3_morphology_boundary_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_fig3_boundary_metric_adapter(
        spec,
        repo_root,
        output_dir,
        source_name="panel_c_morphology_boundary_metrics.csv",
        metric=str(spec.get("metric") or "multi_item_retention_index"),
        unit="index",
    )


def build_fig3_functional_boundary_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_fig3_boundary_metric_adapter(
        spec,
        repo_root,
        output_dir,
        source_name="panel_d_functional_boundary_metrics.csv",
        metric=str(spec.get("metric") or "accessible_item_count"),
        unit="count",
    )


def build_fig3_morphology_function_coupling_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    summary_frames: list[pd.DataFrame] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_e_morphology_function_coupling.csv"
        summary_path = seed_dir / "data" / "metrics" / "panel_e_coupling_summary.csv"
        sources.extend(_source(item, repo_root, seed_dir) for item in (path, summary_path))
        if summary_path.exists():
            summary_frames.append(pd.read_csv(summary_path))
        if not path.exists():
            warnings.append(f"Missing morphology-function coupling source: {_display(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            x_value = _float(row.get("morphology_support_p", row.get("p_i")))
            y_value = _float(row.get("functional_gain_norm", row.get("G_i_norm")))
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="G_i_norm",
                    condition=str(row.get("condition_id", "")),
                    layer=str(row.get("layer", "layer1")),
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    value=y_value,
                    unit="normalized_gain",
                    source_file=_display(path, repo_root),
                    sequence_id=row.get("sequence_id", ""),
                    seq_len=row.get("seq_len", ""),
                    delay_ms=row.get("delay_ms", ""),
                    serial_position=row.get("serial_position", row.get("target_position", "")),
                    x_value=x_value,
                    y_value=y_value,
                    morphology_support_p=x_value,
                    morphology_support_beta=row.get("morphology_support_beta", row.get("beta", "")),
                    functional_gain=row.get("functional_gain", row.get("G_i", "")),
                    functional_gain_norm=y_value,
                )
            )
    stats_extra: dict[str, Any] = {}
    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        if "support_gain_corr" in summary.columns:
            stats_extra["support_gain_corr_mean"] = float(pd.to_numeric(summary["support_gain_corr"], errors="coerce").mean())
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "G_i_norm", ["seq_len", "delay_ms"], stats_extra=stats_extra)


def build_fig3_access_serial_profile_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    focus_seq_len = _optional_int(spec.get("focus_seq_len", spec.get("seq_len_focus", 10)))
    focus_delay_ms = _optional_float(spec.get("focus_delay_ms", spec.get("delay_ms_focus", 400)))
    memory_cols = [
        ("cue_only", "Cue only", "P_target_cue_only"),
        ("single_item_memory", "Slot singleton", "P_target_single_item_memory"),
        ("sequence_state", "Full sequence", "P_target_sequence_state"),
    ]
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_d_item_functional_gain.csv"
        sources.append(_source(path, repo_root, seed_dir))
        if not path.exists():
            warnings.append(f"Missing weak-cue item-gain source: {_display(path, repo_root)}")
            continue
        df = _numeric_gain_frame(pd.read_csv(path))
        df = _filter_focus(df, seq_len=focus_seq_len, delay_ms=focus_delay_ms)
        if df.empty:
            warnings.append(f"No access serial-profile rows after focus filter in {_display(path, repo_root)}")
            continue
        for _, row in df.iterrows():
            for memory_condition, memory_label, source_col in memory_cols:
                value = _float(row.get(source_col))
                if not np.isfinite(value):
                    continue
                rows.append(
                    _canonical(
                        figure_id,
                        panel_id,
                        metric="target_probability",
                        condition=f"{row.get('condition_id', '')}:{memory_condition}",
                        layer="",
                        seed_id=row.get("network_seed", _seed_id(seed_dir)),
                        value=value,
                        unit="probability",
                        source_file=_display(path, repo_root),
                        sequence_id=row.get("sequence_id", ""),
                        seq_len=row.get("seq_len", ""),
                        delay_ms=row.get("delay_ms", ""),
                        serial_position=row.get("target_position", ""),
                        target_position=row.get("target_position", ""),
                        memory_condition=memory_condition,
                        memory_label=memory_label,
                        x_value=row.get("target_position", ""),
                        y_value=value,
                        G_i=row.get("G_i", ""),
                        U_i=row.get("U_i", ""),
                        G_i_norm=row.get("G_i_norm", ""),
                    )
                )
    stats_extra = {"focus_seq_len": focus_seq_len, "focus_delay_ms": focus_delay_ms}
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "target_probability", ["memory_condition", "serial_position"], stats_extra=stats_extra)


def build_fig3_cue_specificity_target_profile_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    focus_seq_len = _optional_int(spec.get("focus_seq_len", 7))
    focus_delay_ms = _optional_float(spec.get("focus_delay_ms", 400))
    state_condition = str(spec.get("state_condition") or "S_final")
    cue_order = tuple(str(v) for v in (spec.get("cue_types") or ["matched", "mismatched", "unseen"]))
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_c_cue_specificity_serial_summary.csv"
        sources.append(_source(path, repo_root, seed_dir))
        if not path.exists():
            warnings.append(f"Missing cue-specificity serial source: {_display(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        required = {"state_condition", "cue_type", "target_position", "P_target_mean"}
        if not required.issubset(df.columns):
            warnings.append(f"Cue-specificity source missing columns {sorted(required - set(df.columns))}: {_display(path, repo_root)}")
            continue
        df = df[df["state_condition"].astype(str).eq(state_condition)].copy()
        df = df[df["cue_type"].astype(str).isin(cue_order)].copy()
        if df.empty:
            warnings.append(f"No cue-specificity rows for state={state_condition} in {_display(path, repo_root)}")
            continue
        for _, row in df.iterrows():
            value = _float(row.get("P_target_mean"))
            if not np.isfinite(value):
                continue
            cue_type = str(row.get("cue_type", ""))
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="target_probability",
                    condition=cue_type,
                    layer="",
                    seed_id=_seed_id(seed_dir),
                    value=value,
                    unit="probability",
                    source_file=_display(path, repo_root),
                    seq_len=focus_seq_len,
                    delay_ms=focus_delay_ms,
                    serial_position=row.get("target_position", ""),
                    target_position=row.get("target_position", ""),
                    cue_type=cue_type,
                    state_condition=state_condition,
                    memory_condition=row.get("memory_condition", ""),
                    x_value=row.get("target_position", ""),
                    y_value=value,
                    sem=row.get("P_target_sem", ""),
                    P_target_sem=row.get("P_target_sem", ""),
                    n_sequences=row.get("n_sequences", ""),
                )
            )
    stats_extra = {"focus_seq_len": focus_seq_len, "focus_delay_ms": focus_delay_ms, "state_condition": state_condition, "cue_types": list(cue_order)}
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "target_probability", ["cue_type", "serial_position"], stats_extra=stats_extra)


def build_fig3_rescue_fraction_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    focus_delay_ms = _optional_float(spec.get("focus_delay_ms", spec.get("delay_ms_focus", 400)))
    threshold = float(spec.get("access_threshold", 0.0))
    metrics = [
        ("singleton_access_fraction", "Slot singleton", "singleton_count"),
        ("sequence_access_fraction", "Full sequence", "sequence_count"),
        ("rescued_fraction", "Rescued", "rescued_count"),
    ]
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_d_item_functional_gain.csv"
        sources.append(_source(path, repo_root, seed_dir))
        if not path.exists():
            warnings.append(f"Missing weak-cue item-gain source: {_display(path, repo_root)}")
            continue
        df = _numeric_gain_frame(pd.read_csv(path))
        df = _filter_focus(df, delay_ms=focus_delay_ms)
        if df.empty:
            warnings.append(f"No rescue rows after delay filter in {_display(path, repo_root)}")
            continue
        work = df.copy()
        work["singleton_access"] = work["U_i"] > threshold
        work["sequence_access"] = work["G_i"] > threshold
        work["rescued"] = (work["U_i"] <= threshold) & (work["G_i"] > threshold)
        group_cols = ["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms"]
        for keys, part in work.groupby(group_cols, dropna=False, sort=True):
            network_seed, condition_id, sequence_id, seq_len, delay_ms = keys
            denom = float(seq_len) if np.isfinite(_float(seq_len)) and float(seq_len) > 0 else float(len(part))
            counts = {
                "singleton_count": int(part["singleton_access"].sum()),
                "sequence_count": int(part["sequence_access"].sum()),
                "rescued_count": int(part["rescued"].sum()),
            }
            for metric, label, count_key in metrics:
                value = counts[count_key] / denom if denom > 0 else np.nan
                if not np.isfinite(value):
                    continue
                rows.append(
                    _canonical(
                        figure_id,
                        panel_id,
                        metric=metric,
                        condition=str(condition_id),
                        layer="",
                        seed_id=network_seed if str(network_seed) else _seed_id(seed_dir),
                        value=value,
                        unit="fraction",
                        source_file=_display(path, repo_root),
                        sequence_id=sequence_id,
                        seq_len=seq_len,
                        delay_ms=delay_ms,
                        access_label=label,
                        item_count=counts[count_key],
                        item_fraction=value,
                        x_value=seq_len,
                        y_value=value,
                        access_threshold=threshold,
                    )
                )
    stats_extra = {"focus_delay_ms": focus_delay_ms, "access_threshold": threshold}
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "rescued_fraction", ["seq_len"], stats_extra=stats_extra)


def build_fig3_morphology_capacity_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    focus_delay_ms = _optional_float(spec.get("focus_delay_ms", spec.get("delay_ms_focus", 400)))
    layer = str(spec.get("morphology_layer") or "layer1")
    state_variable = str(spec.get("state_variable") or "g")
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_c_morphology_boundary_metrics.csv"
        sources.append(_source(path, repo_root, seed_dir))
        if not path.exists():
            warnings.append(f"Missing morphology boundary source: {_display(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        df = _primary(df, layer=layer, state_variable=state_variable)
        df = _filter_focus(df, delay_ms=focus_delay_ms)
        if df.empty:
            warnings.append(f"No morphology capacity rows after focus filter in {_display(path, repo_root)}")
            continue
        for _, row in df.iterrows():
            value = _float(row.get("N_eff"))
            seq_len = _float(row.get("seq_len"))
            if not np.isfinite(value):
                continue
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="N_eff",
                    condition=str(row.get("condition_id", "")),
                    layer=str(row.get("layer", layer)),
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    value=value,
                    unit="items",
                    source_file=_display(path, repo_root),
                    sequence_id=row.get("sequence_id", ""),
                    seq_len=row.get("seq_len", ""),
                    delay_ms=row.get("delay_ms", ""),
                    x_value=row.get("seq_len", ""),
                    y_value=value,
                    N_eff=value,
                    N_eff_fraction=value / seq_len if np.isfinite(seq_len) and seq_len > 0 else np.nan,
                    multi_item_retention_index=row.get("multi_item_retention_index", ""),
                    latest_collapse_index=row.get("latest_collapse_index", ""),
                    reconstruction_R2=row.get("reconstruction_R2", ""),
                )
            )
    stats_extra = {"focus_delay_ms": focus_delay_ms, "morphology_layer": layer, "state_variable": state_variable}
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "N_eff", ["seq_len"], stats_extra=stats_extra)


def build_fig3_delay_boundary_heatmap_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    metric = str(spec.get("metric") or "").strip()
    if not metric:
        return _missing(spec, repo_root, output_dir, f"Fig.3{panel_id}: delay heatmap requires a metric in the panel spec.", sources, warnings)
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "panel_f_boundary_summary.csv"
        sources.append(_source(path, repo_root, seed_dir))
        if not path.exists():
            warnings.append(f"Missing boundary-summary source: {_display(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        required = {"network_seed", "condition_id", "seq_len", "delay_ms", metric}
        missing = sorted(required - set(df.columns))
        if missing:
            warnings.append(f"Boundary summary missing {missing} in {_display(path, repo_root)}")
            continue
        for _, row in df.iterrows():
            value = _float(row.get(metric))
            seq_len = _float(row.get("seq_len"))
            delay_ms = _float(row.get("delay_ms"))
            if not (np.isfinite(value) and np.isfinite(seq_len) and np.isfinite(delay_ms)):
                continue
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric=metric,
                    condition=str(row.get("condition_id", "")),
                    layer="layer1" if metric.startswith("N_eff") else "",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    value=value,
                    unit="fraction" if metric.endswith("_fraction") else "items",
                    source_file=_display(path, repo_root),
                    seq_len=int(seq_len),
                    delay_ms=int(delay_ms),
                    x_value=int(seq_len),
                    y_value=int(delay_ms),
                    z_value=value,
                    condition_id=str(row.get("condition_id", "")),
                    N_eff=row.get("N_eff", ""),
                    N_eff_fraction=row.get("N_eff_fraction", ""),
                    rescued_count=row.get("rescued_count", ""),
                    rescued_fraction=row.get("rescued_fraction", ""),
                    sequence_access_count=row.get("sequence_access_count", ""),
                    singleton_access_count=row.get("singleton_access_count", ""),
                )
            )
    panel_df = pd.DataFrame(rows)
    stats_extra = {
        "boundary_metric": metric,
        "observed_seq_lengths": sorted({int(v) for v in pd.to_numeric(panel_df.get("seq_len", pd.Series(dtype=float)), errors="coerce").dropna().unique()}) if not panel_df.empty else [],
        "observed_delay_ms": sorted({int(v) for v in pd.to_numeric(panel_df.get("delay_ms", pd.Series(dtype=float)), errors="coerce").dropna().unique()}) if not panel_df.empty else [],
        "source_contract": "data/metrics/panel_f_boundary_summary.csv",
    }
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, metric, ["seq_len", "delay_ms"], stats_extra=stats_extra)


def build_fig3_access_capacity_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    focus_delay_ms = _optional_float(spec.get("focus_delay_ms", spec.get("delay_ms_focus", 400)))
    threshold = float(spec.get("access_threshold", 0.0))
    layer = str(spec.get("morphology_layer") or "layer1")
    state_variable = str(spec.get("state_variable") or "g")
    metric_labels = {
        "morphology_N_eff": "L1 N_eff",
        "single_item_access_count": "Slot singleton",
        "sequence_state_access_count": "Full sequence",
        "rescued_count": "Rescued",
    }
    for seed_dir in seeds:
        morph_path = seed_dir / "data" / "metrics" / "panel_c_morphology_boundary_metrics.csv"
        gain_path = seed_dir / "data" / "metrics" / "panel_d_item_functional_gain.csv"
        sources.extend(_source(item, repo_root, seed_dir) for item in (morph_path, gain_path))
        if morph_path.exists():
            morph = _primary(pd.read_csv(morph_path), layer=layer, state_variable=state_variable)
            morph = _filter_focus(morph, delay_ms=focus_delay_ms)
            for _, row in morph.iterrows():
                value = _float(row.get("N_eff"))
                if not np.isfinite(value):
                    continue
                rows.append(
                    _canonical(
                        figure_id,
                        panel_id,
                        metric="morphology_N_eff",
                        condition=str(row.get("condition_id", "")),
                        layer=str(row.get("layer", layer)),
                        seed_id=row.get("network_seed", _seed_id(seed_dir)),
                        value=value,
                        unit="items",
                        source_file=_display(morph_path, repo_root),
                        sequence_id=row.get("sequence_id", ""),
                        seq_len=row.get("seq_len", ""),
                        delay_ms=row.get("delay_ms", ""),
                        access_label=metric_labels["morphology_N_eff"],
                        x_value=row.get("seq_len", ""),
                        y_value=value,
                    )
                )
        else:
            warnings.append(f"Missing morphology boundary source: {_display(morph_path, repo_root)}")
        if gain_path.exists():
            gain = _numeric_gain_frame(pd.read_csv(gain_path))
            gain = _filter_focus(gain, delay_ms=focus_delay_ms)
            if not gain.empty:
                gain["singleton_access"] = gain["U_i"] > threshold
                gain["sequence_access"] = gain["G_i"] > threshold
                gain["rescued"] = (gain["U_i"] <= threshold) & (gain["G_i"] > threshold)
                for keys, part in gain.groupby(["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms"], dropna=False, sort=True):
                    network_seed, condition_id, sequence_id, seq_len, delay_ms = keys
                    values = {
                        "single_item_access_count": int(part["singleton_access"].sum()),
                        "sequence_state_access_count": int(part["sequence_access"].sum()),
                        "rescued_count": int(part["rescued"].sum()),
                    }
                    for metric, value in values.items():
                        rows.append(
                            _canonical(
                                figure_id,
                                panel_id,
                                metric=metric,
                                condition=str(condition_id),
                                layer="",
                                seed_id=network_seed if str(network_seed) else _seed_id(seed_dir),
                                value=float(value),
                                unit="items",
                                source_file=_display(gain_path, repo_root),
                                sequence_id=sequence_id,
                                seq_len=seq_len,
                                delay_ms=delay_ms,
                                access_label=metric_labels[metric],
                                x_value=seq_len,
                                y_value=float(value),
                                access_threshold=threshold,
                            )
                        )
        else:
            warnings.append(f"Missing weak-cue item-gain source: {_display(gain_path, repo_root)}")
    stats_extra = {"focus_delay_ms": focus_delay_ms, "access_threshold": threshold, "morphology_layer": layer}
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "sequence_state_access_count", ["seq_len", "metric"], stats_extra=stats_extra)


def build_fig3_peak_cue_memory_gain_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        metrics_dir = seed_dir / "data" / "metrics"
        gain_path = _first_existing(metrics_dir, ["panel_f_peak_cue_memory_gain.csv", "panel_f_peak_aligned_completion_metrics.csv"])
        accuracy_path = metrics_dir / "panel_f_peak_cue_accuracy.csv"
        diagnostics_path = metrics_dir / "panel_f_peak_cue_matching_diagnostics.csv"
        checked = [metrics_dir / name for name in ("panel_f_peak_cue_memory_gain.csv", "panel_f_peak_cue_accuracy.csv", "panel_f_peak_cue_matching_diagnostics.csv", "panel_f_peak_aligned_completion_metrics.csv")]
        sources.extend(_source(path, repo_root, seed_dir) for path in checked)
        diagnostics = _diagnostic_summary(diagnostics_path)
        if gain_path is not None:
            gain_df = pd.read_csv(gain_path)
            gain_source = gain_path
        elif accuracy_path.exists():
            gain_df = _peak_gain_from_accuracy(pd.read_csv(accuracy_path))
            gain_source = accuracy_path
        else:
            warnings.append(f"Missing peak-cue memory-gain source under {_display(seed_dir, repo_root)}")
            continue
        for _, row in gain_df.iterrows():
            cue = _normalize_cue(row.get("cue_condition", row.get("condition", "")))
            gain = _float(row.get("memory_gain", row.get("completion_gain")))
            if not np.isfinite(gain):
                seq = _float(row.get("P_target_sequence", row.get("P_target_sequence_state", row.get("accuracy_sequence_state"))))
                cue_only = _float(row.get("P_target_cue_only", row.get("accuracy_cue_only")))
                gain = seq - cue_only if np.isfinite(seq) and np.isfinite(cue_only) else float("nan")
            if cue not in set(CUE_ORDER) or not np.isfinite(gain):
                continue
            keep = _float(row.get("keep_fraction", row.get("keep_prob")))
            diag = diagnostics.get((cue, keep), diagnostics.get((cue, None), {}))
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="memory_gain",
                    condition=cue,
                    layer="layer1",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    value=_scale_delta_to_percent(gain),
                    unit="percent_delta",
                    source_file=_display(gain_source, repo_root),
                    cue_condition=cue,
                    cue_condition_order=_cue_order(cue),
                    keep_fraction=keep if np.isfinite(keep) else "",
                    x_value=_cue_order(cue),
                    y_value=_scale_delta_to_percent(gain),
                    memory_condition="sequence_state_minus_cue_only",
                    n_trials=row.get("n_trials", ""),
                    P_target_sequence=row.get("P_target_sequence", row.get("P_target_sequence_state", row.get("accuracy_sequence_state", ""))),
                    P_target_cue_only=row.get("P_target_cue_only", row.get("accuracy_cue_only", "")),
                    **diag,
                )
            )
    panel_df = pd.DataFrame(rows)
    stats_extra = _peak_gain_stats(panel_df)
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "memory_gain", ["cue_condition"], stats_extra=stats_extra)


def _build_fig3_boundary_metric_adapter(
    spec: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    *,
    source_name: str,
    metric: str,
    unit: str,
) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    root, seeds, warnings = _resolve_experiment_root(spec, repo_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / source_name
        sources.append(_source(path, repo_root, seed_dir))
        if not path.exists():
            warnings.append(f"Missing Fig.3 boundary metric source: {_display(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        if metric not in df.columns:
            warnings.append(f"Metric {metric!r} missing from {_display(path, repo_root)}")
            continue
        for _, row in df.iterrows():
            value = _float(row.get(metric))
            if not np.isfinite(value):
                continue
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric=metric,
                    condition=str(row.get("condition_id", "")),
                    layer=str(row.get("layer", "layer1")),
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    value=value,
                    unit=unit,
                    source_file=_display(path, repo_root),
                    sequence_id=row.get("sequence_id", ""),
                    seq_len=row.get("seq_len", ""),
                    delay_ms=row.get("delay_ms", ""),
                    x_value=row.get("seq_len", ""),
                    y_value=row.get("delay_ms", ""),
                    z_value=value,
                    multi_item_retention_index=row.get("multi_item_retention_index", ""),
                    latest_collapse_index=row.get("latest_collapse_index", ""),
                    accessible_item_count=row.get("accessible_item_count", ""),
                    functional_retention_index=row.get("functional_retention_index", ""),
                )
            )
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, metric, ["seq_len", "delay_ms"])


def build_fig3_neutral_ping_distribution_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return build_fig3_neutral_ping_serial_adapter(spec, repo_root, output_dir)


def build_fig3_structural_weak_cue_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return build_fig3_peak_cue_memory_gain_adapter(spec, repo_root, output_dir)


def build_fig3_peak_valley_prevalence_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return build_fig3_example_landscape_adapter(spec, repo_root, output_dir)


def _resolve_experiment_root(spec: Mapping[str, Any], repo_root: Path) -> tuple[Path, list[Path], list[str]]:
    raw = spec.get("experiment_root") or spec.get("experiment_root_default") or DEFAULT_EXPERIMENT_ROOT
    root = Path(str(raw))
    if not root.is_absolute():
        root = repo_root / root
    if (root / "data" / "metrics").exists() and root.name.startswith("seed_"):
        seeds = [root]
    else:
        seeds = sorted(path for path in root.glob("seed_*") if (path / "data" / "metrics").exists()) if root.exists() else []
        if not seeds and (root / "data" / "metrics").exists():
            seeds = [root]
    warnings = [DRAFT_WARNING] if len(seeds) == 1 else []
    return root, seeds, warnings


def _finish(
    spec: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    figure_id: str,
    panel_id: str,
    root: Path,
    seeds: Sequence[Path],
    sources: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    warnings: list[str],
    primary_metric: str,
    group_cols: list[str],
    *,
    stats_extra: Mapping[str, Any] | None = None,
) -> AdapterResult:
    if not rows:
        return _missing(spec, repo_root, output_dir, f"Fig.3{panel_id}: no usable rows available from checked sources.", sources, warnings)
    panel_df = pd.DataFrame(rows)
    run_mode = _run_mode(seeds)
    panel_df["run_mode"] = run_mode
    panel_df["n_networks"] = len(seeds)
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "metric": primary_metric,
        "run_mode": run_mode,
        "n_networks": len(seeds),
        "network_ids": [_seed_id(seed) for seed in seeds],
        "summaries": summarize_values(panel_df[panel_df["metric"].astype(str).eq(primary_metric)] if "metric" in panel_df.columns else panel_df, group_cols),
        "values_used_for_plotting": _values(panel_df, primary_metric),
        "warnings": list(warnings),
    }
    if stats_extra:
        stats.update(dict(stats_extra))
    manifest = _manifest(figure_id, panel_id, root, seeds, sources, warnings, status="ok")
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def _missing(spec: Mapping[str, Any], repo_root: Path, output_dir: Path, reason: str, sources: list[dict[str, Any]], warnings: Sequence[str]) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    df = pd.DataFrame([_canonical(figure_id, panel_id, metric="missing_source", condition="missing", layer="", seed_id="", value=np.nan, unit="", source_file="", placeholder_reason=reason)])
    stats = {"figure_id": figure_id, "panel_id": panel_id, "status": "missing_source", "values_used_for_plotting": [], "warnings": list(warnings) + [reason]}
    manifest = {"figure_id": figure_id, "panel_id": panel_id, "status": "missing_source", "repo_root": str(repo_root), "sources": sources, "source_files_used": [], "warnings": list(warnings) + [reason]}
    return write_adapter_outputs(output_dir, figure_id, panel_id, df, stats, manifest, list(warnings) + [reason])


def _manifest(figure_id: str, panel_id: str, root: Path, seeds: Sequence[Path], sources: list[dict[str, Any]], warnings: Sequence[str], *, status: str) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": status,
        "experiment_root": str(root).replace("\\", "/"),
        "run_mode": _run_mode(seeds),
        "n_networks": len(seeds),
        "network_ids": [_seed_id(seed) for seed in seeds],
        "sources": sources,
        "source_files_used": [src["path"] for src in sources if src.get("exists")],
        "warnings": list(warnings),
    }


def _canonical(figure_id: str, panel_id: str, *, metric: str, condition: str, layer: str, seed_id: Any, value: Any, unit: str, source_file: str, **extra: Any) -> dict[str, Any]:
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


def _source(path: Path, repo_root: Path, seed_dir: Path | None = None) -> dict[str, Any]:
    out = {"path": _display(path, repo_root), "exists": path.exists()}
    if seed_dir is not None:
        out["seed_id"] = _seed_id(seed_dir)
    return out


def _display(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", FIGURE_ID)).lower(), str(spec.get("panel_id", "")).upper()


def _seed_id(seed_dir: Path) -> str:
    return seed_dir.name if seed_dir.name.startswith("seed_") else str(seed_dir.name)


def _run_mode(seeds: Sequence[Path]) -> str:
    return "single_network_draft" if len(seeds) <= 1 else "multi_network_final"


def _first_existing(base: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    out = _float(value)
    return int(round(out)) if np.isfinite(out) else None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    out = _float(value)
    return float(out) if np.isfinite(out) else None


def _numeric_gain_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "network_seed",
        "sequence_id",
        "seq_len",
        "delay_ms",
        "target_position",
        "P_target_sequence_state",
        "P_target_single_item_memory",
        "P_target_cue_only",
        "G_i",
        "U_i",
        "G_i_norm",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _filter_focus(df: pd.DataFrame, *, seq_len: int | None = None, delay_ms: float | None = None) -> pd.DataFrame:
    out = df.copy()
    if seq_len is not None and "seq_len" in out.columns:
        out = out[pd.to_numeric(out["seq_len"], errors="coerce").eq(seq_len)]
    if delay_ms is not None and "delay_ms" in out.columns:
        delays = pd.to_numeric(out["delay_ms"], errors="coerce")
        out = out[np.isclose(delays, delay_ms, rtol=0.0, atol=1e-9)]
    return out


def _primary(df: pd.DataFrame, *, layer: str, state_variable: str) -> pd.DataFrame:
    use = df.copy()
    if "layer" in use.columns:
        use = use[use["layer"].astype(str).str.lower().eq(layer.lower())]
    if "state_variable" in use.columns:
        use = use[use["state_variable"].astype(str).str.lower().eq(state_variable.lower())]
    return use


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _serial_position(value: Any) -> int | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if text.startswith("pos_"):
        text = text.split("_", 1)[1]
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return int(numeric)
    return None


def _normalize_state(value: Any) -> str:
    text = str(value).strip()
    return text if text else "unknown"


def _normalize_cue(value: Any) -> str:
    text = str(value).strip().lower()
    return {"valley_aligned": "valley", "random_matched": "random", "peak_aligned": "peak"}.get(text, text)


def _cue_order(cue: str) -> int:
    return {name: idx for idx, name in enumerate(CUE_ORDER)}.get(_normalize_cue(cue), 99)


def _memory_order(memory: str) -> int:
    return {name: idx for idx, name in enumerate(MEMORY_ORDER, start=1)}.get(str(memory), 99)


def _normalize_region(value: Any) -> str:
    text = str(value).strip().lower()
    return {"peak_aligned": "peak", "random_matched": "random", "valley_aligned": "valley"}.get(text, text)


def _region_order(region: str) -> int:
    return {name: idx for idx, name in enumerate(REGION_ORDER, start=1)}.get(_normalize_region(region), 99)


def _to_percent_value(value: float) -> float:
    return float(value) * 100.0 if abs(float(value)) <= 1.5 else float(value)


def _scale_delta_to_percent(value: float) -> float:
    return float(value) * 100.0 if abs(float(value)) <= 2.0 else float(value)


def _values(df: pd.DataFrame, metric: str) -> list[float]:
    if df.empty or "value" not in df.columns:
        return []
    use = df[df["metric"].astype(str).eq(metric)] if "metric" in df.columns else df
    return [float(v) for v in pd.to_numeric(use["value"], errors="coerce").dropna().tolist()]


def _ping_position_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    if {"state_condition", "serial_position"}.issubset(df.columns):
        value_col = "readout_mass" if "readout_mass" in df.columns else "P_target"
        if value_col in df.columns:
            grouped = df.groupby(["network_seed", "state_condition", "serial_position"], dropna=False)[value_col].mean().reset_index()
            grouped["serial_bin"] = grouped["serial_position"].map(lambda v: f"pos_{int(v)}" if pd.notna(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]) else str(v))
            grouped["readout_mass"] = grouped[value_col]
            grouped["n_trials"] = df.groupby(["network_seed", "state_condition", "serial_position"], dropna=False)[value_col].size().to_numpy()
            return grouped
    return pd.DataFrame()


def _region_ping_distribution_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    required = {"region_condition", "serial_bin"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    group_cols = [
        col
        for col in ["network_seed", "state_condition", "memory_condition", "region_condition", "seq_len", "serial_bin"]
        if col in df.columns
    ]
    grouped = df.groupby(group_cols, dropna=False).size().reset_index(name="n_bin")
    totals = df.groupby([col for col in group_cols if col != "serial_bin"], dropna=False).size().reset_index(name="n_trials")
    out = grouped.merge(totals, on=[col for col in group_cols if col != "serial_bin"], how="left")
    out["readout_mass"] = out["n_bin"] / out["n_trials"].replace(0, np.nan)
    return out


def _aggregate_region_ping_readout_rows(
    df: pd.DataFrame,
    *,
    figure_id: str,
    panel_id: str,
    seed_dir: Path,
    source_file: str,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    if "region_condition" not in work.columns and "condition" in work.columns:
        work["region_condition"] = work["condition"]
    if "network_seed" not in work.columns:
        work["network_seed"] = _seed_id(seed_dir)
    if "state_condition" not in work.columns:
        work["state_condition"] = "S_final"
    if "memory_condition" not in work.columns:
        work["memory_condition"] = "sequence_state"
    if "seq_len" not in work.columns:
        work["seq_len"] = ""
    group_cols = ["network_seed", "state_condition", "memory_condition", "region_condition", "seq_len"]
    rows: list[dict[str, Any]] = []
    for keys, part in work.groupby(group_cols, dropna=False, sort=False):
        seed_id, state, memory, region_raw, seq_len_raw = keys
        region = _normalize_region(region_raw)
        if region not in set(REGION_ORDER):
            continue
        seq_len = _infer_region_seq_len(part, seq_len_raw)
        split = int(seq_len // 2) if seq_len > 0 else 0
        category_mass = {"old": 0.0, "recent": 0.0, "silent": 0.0}
        other_mass = 0.0
        for _, row in part.iterrows():
            mass = _float(row.get("readout_mass", row.get("value")))
            if not np.isfinite(mass):
                continue
            category = _region_readout_category(row.get("serial_bin", row.get("serial_position", "")), seq_len)
            if category == "other":
                other_mass += mass
                category = "silent"
            category_mass[category] += mass
        n_trials = _max_numeric(part.get("n_trials", pd.Series(dtype=float)))
        for category in ("recent", "old", "silent"):
            mass = float(category_mass[category])
            rows.append(
                _canonical(
                    figure_id,
                    panel_id,
                    metric="readout_mass",
                    condition=region,
                    layer="layer3",
                    seed_id=seed_id,
                    value=mass,
                    unit="probability",
                    source_file=source_file,
                    region_condition=region,
                    region_condition_order=_region_order(region),
                    readout_category=category,
                    readout_category_order=_readout_category_order(category),
                    y_value=mass,
                    x_value=_region_order(region),
                    seq_len=seq_len,
                    state_condition=_normalize_state(state),
                    memory_condition=str(memory),
                    n_trials=n_trials if np.isfinite(n_trials) else "",
                    old_positions=",".join(str(pos) for pos in range(1, split + 1)),
                    recent_positions=",".join(str(pos) for pos in range(split + 1, seq_len + 1)),
                    other_mass_included_in_silent=bool(other_mass > 0),
                    other_mass=other_mass if category == "silent" else 0.0,
                )
            )
    return rows


def _filter_region_ping_main(df: pd.DataFrame) -> pd.DataFrame:
    use = df.copy()
    if "state_condition" in use.columns and use["state_condition"].astype(str).eq("S_final").any():
        use = use[use["state_condition"].astype(str).eq("S_final")].copy()
    if "memory_condition" in use.columns and use["memory_condition"].astype(str).eq("sequence_state").any():
        use = use[use["memory_condition"].astype(str).eq("sequence_state")].copy()
    return use


def _infer_region_seq_len(df: pd.DataFrame, seq_len_raw: Any = "") -> int:
    numeric_seq_len = _float(seq_len_raw)
    if np.isfinite(numeric_seq_len) and numeric_seq_len > 0:
        return int(numeric_seq_len)
    if "seq_len" in df.columns:
        vals = pd.to_numeric(df["seq_len"], errors="coerce").dropna()
        if not vals.empty:
            return int(vals.max())
    positions = [_serial_position(value) for value in df.get("serial_bin", pd.Series(dtype=object))]
    positions += [_serial_position(value) for value in df.get("serial_position", pd.Series(dtype=object))]
    positions = [pos for pos in positions if pos is not None]
    return int(max(positions)) if positions else 10


def _region_readout_category(serial_bin: Any, seq_len: int) -> str:
    text = str(serial_bin).strip().lower()
    if text in {"silent", "silence", "no_readout", "none"}:
        return "silent"
    if text in {"other", "unseen", "unknown", "nan", ""}:
        return "other"
    pos = _serial_position(text)
    if pos is None or pos <= 0:
        return "other"
    split = int(seq_len // 2) if seq_len > 0 else pos
    return "old" if pos <= split else "recent"


def _readout_category_order(category: str) -> int:
    return {"recent": 1, "old": 2, "silent": 3}.get(str(category), 99)


def _max_numeric(values: Any) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else float("nan")


def _serial_bin_order(serial_bin: Any, seq_len: Any = "") -> int:
    text = str(serial_bin).strip()
    pos = _serial_position(text)
    if pos is not None:
        return int(pos)
    length = _float(seq_len)
    offset = int(length) if np.isfinite(length) and length > 0 else 99
    if text == "other":
        return offset + 1
    if text == "silent":
        return offset + 2
    return offset + 3


def _region_ping_stats(
    panel_df: pd.DataFrame,
    summary_frames: Sequence[pd.DataFrame],
    contrast_frames: Sequence[pd.DataFrame],
    current_frames: Sequence[pd.DataFrame],
) -> dict[str, Any]:
    readout_categories = ["recent", "old", "silent"]
    stats: dict[str, Any] = {
        "available_region_conditions": sorted(set(panel_df.get("region_condition", pd.Series(dtype=str)).dropna().astype(str))) if not panel_df.empty else [],
        "available_serial_bins": sorted(set(panel_df.get("serial_bin", pd.Series(dtype=str)).dropna().astype(str)), key=lambda v: _serial_bin_order(v)) if "serial_bin" in panel_df.columns and not panel_df.empty else [],
        "main_plot_type": "stacked_readout_mass",
        "readout_categories": readout_categories,
        "serial_categories": readout_categories,
        "uses_serial_position_10_class": False,
        "uses_latest_recent_earlier_other_silent": False,
        "y_axis_absolute_probability": True,
        "stacked_bars_not_normalized": True,
        "silent_definition": "serial_bin == silent plus other/unseen outputs included in Silent",
        "other_mass": float(pd.to_numeric(panel_df.get("other_mass", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not panel_df.empty else 0.0,
        "other_mass_included_in_silent": bool(pd.to_numeric(panel_df.get("other_mass", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum() > 0) if not panel_df.empty else False,
        "region_order": list(REGION_ORDER),
        "current_matching_status": "missing",
    }
    if not panel_df.empty:
        seq_len = int(pd.to_numeric(panel_df.get("seq_len", pd.Series(dtype=float)), errors="coerce").dropna().max()) if not pd.to_numeric(panel_df.get("seq_len", pd.Series(dtype=float)), errors="coerce").dropna().empty else 10
        split = int(seq_len // 2)
        stats["old_positions"] = list(range(1, split + 1))
        stats["recent_positions"] = list(range(split + 1, seq_len + 1))
        by_region = panel_df.copy()
        by_region["value"] = pd.to_numeric(by_region["value"], errors="coerce").fillna(0.0)
        for category in readout_categories:
            part = by_region[by_region.get("readout_category", pd.Series(dtype=str)).astype(str).eq(category)]
            stats[f"{category}_mass_by_region"] = {
                _normalize_region(region): float(group["value"].sum())
                for region, group in part.groupby("region_condition", dropna=False)
            }
        stats["total_readout_mass_by_region"] = {
            _normalize_region(region): float(group["value"].sum())
            for region, group in by_region.groupby("region_condition", dropna=False)
        }
    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        if "state_condition" in summary.columns and summary["state_condition"].astype(str).eq("S_final").any():
            summary = summary[summary["state_condition"].astype(str).eq("S_final")]
        if "memory_condition" in summary.columns and summary["memory_condition"].astype(str).eq("sequence_state").any():
            summary = summary[summary["memory_condition"].astype(str).eq("sequence_state")]
        for metric in ("P_seen_item", "P_latest_item", "P_silent"):
            if metric in summary.columns and "region_condition" in summary.columns:
                stats[f"{metric}_by_region"] = {
                    _normalize_region(region): float(pd.to_numeric(part[metric], errors="coerce").mean())
                    for region, part in summary.groupby("region_condition", dropna=False)
                }
    if contrast_frames:
        contrast = pd.concat(contrast_frames, ignore_index=True)
        for col in ("JS_peak_valley", "TV_peak_valley", "P_peak_label_differs_from_valley", "latency_peak_minus_valley"):
            if col in contrast.columns:
                stats[col] = float(pd.to_numeric(contrast[col], errors="coerce").dropna().mean())
    if current_frames:
        current = pd.concat(current_frames, ignore_index=True)
        stats["current_matching_status"] = _current_matching_status(current)
    return stats


def _current_matching_status(current: pd.DataFrame) -> str:
    if current.empty or "region_condition" not in current.columns:
        return "missing"
    required = {"peak", "valley", "random"}
    if not required.issubset({_normalize_region(v) for v in current["region_condition"].astype(str)}):
        return "failed"
    active = pd.to_numeric(current.get("active_unit_count_mean", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    total = pd.to_numeric(current.get("total_ping_current_mean", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    if active.size == 0 or total.size == 0:
        return "missing"
    return "passed" if (float(np.max(active) - np.min(active)) <= 1e-9 and float(np.max(total) - np.min(total)) <= 1e-9) else "failed"


def _weak_probe_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    target_col = "pred_is_target" if "pred_is_target" in df.columns else "target_hit"
    required = {"memory_condition", "keep_prob", target_col}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    group_cols = [col for col in ["network_seed", "target_source", "seq_len", "state_condition", "memory_condition", "keep_prob"] if col in df.columns]
    grouped = df.groupby(group_cols, dropna=False)[target_col].mean().reset_index()
    grouped["P_target"] = grouped[target_col]
    grouped["n_trials"] = df.groupby(group_cols, dropna=False)[target_col].size().to_numpy()
    return grouped


def _prefer_sequence_member_targets(df: pd.DataFrame, warnings: list[str], label: str) -> pd.DataFrame:
    if df.empty or "target_source" not in df.columns:
        return df
    sources = set(df["target_source"].dropna().astype(str))
    preferred = [src for src in sources if "sequence_member" in src]
    if not preferred:
        return df
    if len(sources) > len(preferred):
        warnings.append(f"{label}: filtered mixed target_source values to sequence-member targets for main plotting.")
    return df[df["target_source"].astype(str).isin(preferred)].copy()


def _ping_stats(panel_df: pd.DataFrame, summary_frames: Sequence[pd.DataFrame]) -> dict[str, Any]:
    stats: dict[str, Any] = {"available_state_conditions": []}
    if panel_df.empty:
        return stats
    stats["available_state_conditions"] = sorted(set(panel_df.get("state_condition", pd.Series(dtype=str)).dropna().astype(str)))
    numeric = panel_df[pd.to_numeric(panel_df.get("serial_position", pd.Series(dtype=object)), errors="coerce").notna()].copy()
    numeric["serial_position"] = pd.to_numeric(numeric["serial_position"], errors="coerce")
    numeric["value"] = pd.to_numeric(numeric["value"], errors="coerce")
    final = numeric[numeric["state_condition"].astype(str).eq("S_final")]
    if not final.empty:
        max_pos = int(final["serial_position"].max())
        latest = final[final["serial_position"].eq(max_pos)]["value"].mean()
        recent = final[final["serial_position"].ge(max_pos - 2)]["value"].sum()
        total = final["value"].sum()
        com = (final["serial_position"] * final["value"]).sum() / total if total else np.nan
        stats.update(
            {
                "latest_item_mass": float(latest) if np.isfinite(latest) else None,
                "recent_item_mass": float(recent) if np.isfinite(recent) else None,
                "earlier_item_residual_mass": float(total - recent) if np.isfinite(total) and np.isfinite(recent) else None,
                "ping_COM": float(com) if np.isfinite(com) else None,
            }
        )
    if summary_frames:
        joined = pd.concat(summary_frames, ignore_index=True)
        for col in ("P_seen_item", "P_unseen", "P_silent"):
            if col in joined.columns:
                stats[col] = float(pd.to_numeric(joined[col], errors="coerce").mean())
    return stats


def _weak_probe_stats(auc_frames: Sequence[pd.DataFrame], gain_frames: Sequence[pd.DataFrame], panel_df: pd.DataFrame) -> dict[str, Any]:
    raw_memories = set(panel_df.get("memory_condition", pd.Series(dtype=str)).dropna().astype(str)) if not panel_df.empty else set()
    memories = [memory for memory in MEMORY_ORDER if memory in raw_memories] + sorted(memory for memory in raw_memories if memory not in set(MEMORY_ORDER))
    stats: dict[str, Any] = {
        "has_single_item_memory": "single_item_memory" in memories,
        "memory_conditions": memories,
        "main_metric": "P_target",
        "gain_columns_available": [],
    }
    if not auc_frames:
        auc = pd.DataFrame()
    else:
        auc = pd.concat(auc_frames, ignore_index=True)
    if not auc.empty:
        for col in ("normalized_auc_target_recovery", "sequence_vs_S0_auc_gain", "sequence_state_vs_cue_only_auc_gain"):
            if col in auc.columns:
                stats[col if col != "normalized_auc_target_recovery" else "auc_target_recovery"] = float(pd.to_numeric(auc[col], errors="coerce").mean())
    if gain_frames:
        gain = pd.concat(gain_frames, ignore_index=True)
        wanted = ("sequence_minus_S0", "sequence_minus_single_item", "single_item_minus_S0")
        stats["gain_columns_available"] = [col for col in wanted if col in gain.columns]
        for col in wanted:
            if col not in gain.columns:
                continue
            vals = pd.to_numeric(gain[col], errors="coerce").dropna()
            if vals.empty:
                continue
            stats[col] = {
                "mean": _scale_delta_to_percent(float(vals.mean())),
                "max": _scale_delta_to_percent(float(vals.max())),
                "auc_summary": _scale_delta_to_percent(float(vals.mean())),
            }
    return stats


def _diagnostic_summary(path: Path) -> dict[tuple[str, float | None], dict[str, float]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[tuple[str, float | None], dict[str, float]] = {}
    for cue_raw, cue_df in df.groupby("cue_condition", dropna=False):
        cue = _normalize_cue(cue_raw)
        keep_values = cue_df["keep_fraction"].dropna().unique() if "keep_fraction" in cue_df.columns else [None]
        for keep in keep_values:
            part = cue_df[cue_df["keep_fraction"].eq(keep)] if keep is not None and "keep_fraction" in cue_df.columns else cue_df
            out[(cue, float(keep) if keep is not None else None)] = _diagnostic_means(part)
        out[(cue, None)] = _diagnostic_means(cue_df)
    return out


def _diagnostic_means(df: pd.DataFrame) -> dict[str, float]:
    aliases = {
        "cue_pixel_count": ["cue_pixel_count", "cue_pixel_count_mean"],
        "cue_energy": ["cue_energy", "cue_energy_mean", "cue_input_energy"],
        "encoded_spike_count": ["encoded_spike_count", "encoded_spike_count_mean", "cue_spike_count"],
    }
    out: dict[str, float] = {}
    for target, names in aliases.items():
        col = next((name for name in names if name in df.columns), None)
        if col is not None:
            out[target] = float(pd.to_numeric(df[col], errors="coerce").mean())
    return out


def _peak_gain_from_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    if not {"cue_condition", "memory_condition"}.issubset(df.columns):
        return pd.DataFrame()
    value_col = "P_target" if "P_target" in df.columns else "accuracy"
    if value_col not in df.columns:
        return pd.DataFrame()
    index_cols = [col for col in ["network_seed", "cue_condition", "keep_fraction"] if col in df.columns]
    pivot = df.pivot_table(index=index_cols, columns="memory_condition", values=value_col, aggfunc="mean").reset_index()
    if {"sequence_state", "cue_only"}.issubset(pivot.columns):
        pivot["memory_gain"] = pivot["sequence_state"] - pivot["cue_only"]
        pivot["P_target_sequence_state"] = pivot["sequence_state"]
        pivot["P_target_cue_only"] = pivot["cue_only"]
    return pivot


def _peak_gain_stats(panel_df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if panel_df.empty:
        return stats
    summary = panel_df[panel_df["metric"].astype(str).eq("memory_gain")].copy()
    means = summary.groupby("cue_condition")["value"].mean().to_dict() if "cue_condition" in summary.columns else {}
    if {"peak", "random"}.issubset(means):
        stats["peak_minus_random"] = float(means["peak"] - means["random"])
    if {"peak", "valley"}.issubset(means):
        stats["peak_minus_valley"] = float(means["peak"] - means["valley"])
    for col in ("cue_pixel_count", "cue_energy", "encoded_spike_count"):
        if col in summary.columns:
            stats[f"{col}_by_cue"] = {str(k): float(v) for k, v in summary.groupby("cue_condition")[col].mean().items()}
    return stats


# Compatibility aliases for older spec names.
build_fig3_progressive_update = build_fig3_progressive_update_adapter
build_fig3_multiitem_profile = build_fig3_example_landscape_adapter
build_fig3_center_migration = build_fig3_neutral_ping_serial_adapter
build_fig3_two_item_morphology = build_fig3_progressive_update_adapter
build_fig3_two_item_readout = build_fig3_neutral_ping_serial_adapter
build_fig3_neutral_ping_adapter = build_fig3_neutral_ping_serial_adapter
build_fig3_region_ping_readout = build_fig3_region_ping_readout_adapter
build_fig3_peak_aligned_completion_adapter = build_fig3_peak_cue_memory_gain_adapter
build_fig3_structural_weak_cue = build_fig3_structural_weak_cue_adapter
build_fig3_delay_morphology_heatmap_adapter = build_fig3_delay_boundary_heatmap_adapter
build_fig3_delay_rescue_heatmap_adapter = build_fig3_delay_boundary_heatmap_adapter
