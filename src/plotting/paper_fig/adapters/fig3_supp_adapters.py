from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult
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


def build_s5_peak_valley_contrast_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_region_support(spec, repo_root, output_dir, ["supp_peak_valley_contrast.csv", "panel_d_peak_valley_contrast.csv"])


def build_s5_landscape_nonflatness_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, ["supp_landscape_nonflatness.csv", "panel_d_landscape_nonflatness.csv"])
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, ["supp_landscape_nonflatness.csv", "panel_d_landscape_nonflatness.csv"]):
        if path is None:
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            for metric in ("top_q_mass_fraction", "support_gini", "support_cv"):
                value = _float(row.get(metric))
                if np.isfinite(value):
                    rows.append(_row(figure_id, panel_id, metric, metric, value, "value", row, seed_dir, path, repo_root, sequence_id=row.get("sequence_id", ""), seq_len=row.get("seq_len", "")))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "top_q_mass_fraction", ["metric"])


def build_s5_peak_valley_null_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    names = ["supp_peak_valley_prevalence.csv", "supp_network_peak_valley_summary.csv", "panel_d_peak_valley_prevalence.csv", "panel_d_network_peak_valley_summary.csv"]
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names)
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
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
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            for metric in ("anchor_COM", "similarity_entropy", "state_displacement", "earlier_residual_proxy", "earlier_item_residual_mass"):
                value = _float(row.get(metric))
                if np.isfinite(value):
                    rows.append(_row(figure_id, panel_id, metric, metric, value, "value", row, seed_dir, path, repo_root, stage_k=row.get("stage_k", ""), x_value=row.get("stage_k", ""), y_value=value, seq_len=row.get("seq_len", "")))
    return _finish(spec, repo_root, output_dir, figure_id, panel_id, root, seeds, sources, rows, warnings, "anchor_COM", ["stage_k"])


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


def _build_region_support(spec: Mapping[str, Any], repo_root: Path, output_dir: Path, names: Sequence[str]) -> AdapterResult:
    figure_id, panel_id, root, seeds, warnings, sources = _start(spec, repo_root, names)
    rows: list[dict[str, Any]] = []
    for seed_dir, path in _seed_paths(seeds, names):
        if path is None:
            continue
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
