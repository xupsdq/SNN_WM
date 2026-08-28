from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig3.constants import CUE_CONDITIONS, MEMORY_CONDITIONS
from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _first_float, _mean_numeric, _row_float
from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import ensure_structural_weak_cue_outputs
from src.experiments.paper_figures.fig3.types import ExperimentContext, MultiItemSequenceLandscapeBank

def run_peak_cue_main_from_state_bank(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    ensure_structural_weak_cue_outputs(ctx, bank)
    raw_path = ctx.raw_dir / "supp_structural_weak_cue_trial_readout.csv"
    match_path = ctx.metrics_dir / "supp_structural_weak_cue_matching_diagnostics.csv"
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    panel_raw = _filter_peak_cue_main_raw(ctx, raw)
    wanted_raw_columns = [
        "network_seed",
        "sequence_id",
        "seq_len",
        "target_source",
        "target_position",
        "target_image_id",
        "target_label",
        "keep_fraction",
        "cue_condition",
        "repeat_id",
        "mask_id",
        "memory_condition",
        "prediction",
        "correct",
        "pred_is_target",
        "pred_is_seen_item",
        "pred_is_unseen",
        "silent",
        "first_fire_time_ms",
        "cue_pixel_count",
        "cue_fraction_actual",
        "cue_energy",
        "encoded_spike_count",
        "support_mean_selected",
        "support_mean_foreground",
        "support_quantile_mean",
    ]
    missing_optional = [column for column in wanted_raw_columns if column not in panel_raw.columns]
    if missing_optional:
        ctx.warnings.append("Panel F peak-cue optional fields missing from structural raw: " + ",".join(missing_optional))
    raw_columns = [column for column in wanted_raw_columns if column in panel_raw.columns]
    if not raw_columns:
        raw_columns = list(panel_raw.columns)
    _save_csv(ctx, panel_raw.loc[:, raw_columns].copy(), ctx.raw_dir / "panel_f_peak_cue_trial_readout.csv")
    accuracy = _peak_cue_accuracy(ctx.cfg.network_seed, panel_raw)
    gain = _peak_cue_memory_gain(ctx.cfg.network_seed, accuracy)
    matching = _peak_cue_matching_diagnostics(ctx, panel_raw, match_path)
    _save_csv(ctx, accuracy, ctx.metrics_dir / "panel_f_peak_cue_accuracy.csv")
    _save_csv(ctx, gain, ctx.metrics_dir / "panel_f_peak_cue_memory_gain.csv")
    _save_csv(ctx, matching, ctx.metrics_dir / "panel_f_peak_cue_matching_diagnostics.csv")
    compute_peak_cue_serial_position_metrics(ctx)
    ctx.completed_modules["peak_cue_main"] = True

def _filter_peak_cue_main_raw(ctx: ExperimentContext, raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    out = raw.copy()
    if "keep_fraction" in out.columns:
        keep = pd.to_numeric(out["keep_fraction"], errors="coerce").to_numpy(dtype=float)
        out = out[np.isclose(keep, float(ctx.cfg.peak_cue_main_keep_fraction))].copy()
    if "target_source" in out.columns:
        out = out[out["target_source"].astype(str).eq("sequence_member_random")].copy()
    if "cue_condition" in out.columns:
        out = out[out["cue_condition"].astype(str).isin(CUE_CONDITIONS)].copy()
    if "memory_condition" in out.columns:
        out = out[out["memory_condition"].astype(str).isin(MEMORY_CONDITIONS)].copy()
    return out.reset_index(drop=True)

def _peak_cue_accuracy(network_seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "cue_condition",
        "memory_condition",
        "keep_fraction",
        "P_target",
        "P_seen_item",
        "P_unseen",
        "P_silent",
        "n_trials",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    grouped = raw.groupby(["network_seed", "cue_condition", "memory_condition", "keep_fraction"], sort=True)
    for (seed, cue_condition, memory_condition, keep_fraction), part in grouped:
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(seed) if pd.notna(seed) else int(network_seed),
                "cue_condition": str(cue_condition),
                "memory_condition": str(memory_condition),
                "keep_fraction": float(keep_fraction),
                "P_target": float(part["pred_is_target"].sum() / denom) if "pred_is_target" in part else 0.0,
                "P_seen_item": float(part["pred_is_seen_item"].sum() / denom) if "pred_is_seen_item" in part else 0.0,
                "P_unseen": float(part["pred_is_unseen"].sum() / denom) if "pred_is_unseen" in part else 0.0,
                "P_silent": float(part["silent"].sum() / denom) if "silent" in part else 0.0,
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def _peak_cue_memory_gain(network_seed: int, accuracy: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "cue_condition",
        "keep_fraction",
        "P_target_sequence_state",
        "P_target_cue_only",
        "memory_gain",
        "P_seen_sequence_state",
        "P_seen_cue_only",
        "seen_item_gain",
        "P_silent_sequence_state",
        "P_silent_cue_only",
        "silent_reduction",
        "n_trials",
    ]
    if accuracy.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (seed, cue_condition, keep_fraction), part in accuracy.groupby(["network_seed", "cue_condition", "keep_fraction"], sort=True):
        seq = part[part["memory_condition"].astype(str).eq("sequence_state")]
        cue = part[part["memory_condition"].astype(str).eq("cue_only")]
        p_target_seq = _first_float(seq, "P_target")
        p_target_cue = _first_float(cue, "P_target")
        p_seen_seq = _first_float(seq, "P_seen_item")
        p_seen_cue = _first_float(cue, "P_seen_item")
        p_silent_seq = _first_float(seq, "P_silent")
        p_silent_cue = _first_float(cue, "P_silent")
        rows.append(
            {
                "network_seed": int(seed) if pd.notna(seed) else int(network_seed),
                "cue_condition": str(cue_condition),
                "keep_fraction": float(keep_fraction),
                "P_target_sequence_state": p_target_seq,
                "P_target_cue_only": p_target_cue,
                "memory_gain": float(p_target_seq - p_target_cue),
                "P_seen_sequence_state": p_seen_seq,
                "P_seen_cue_only": p_seen_cue,
                "seen_item_gain": float(p_seen_seq - p_seen_cue),
                "P_silent_sequence_state": p_silent_seq,
                "P_silent_cue_only": p_silent_cue,
                "silent_reduction": float(p_silent_cue - p_silent_seq),
                "n_trials": int(min(_first_float(seq, "n_trials"), _first_float(cue, "n_trials"))),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def _peak_cue_matching_diagnostics(ctx: ExperimentContext, panel_raw: pd.DataFrame, match_path: Path) -> pd.DataFrame:
    columns = [
        "network_seed",
        "cue_condition",
        "keep_fraction",
        "cue_pixel_count",
        "cue_fraction_actual",
        "cue_energy",
        "encoded_spike_count",
        "support_mean_selected",
        "support_quantile_mean",
        "n_masks",
    ]
    match = pd.read_csv(match_path) if match_path.exists() else pd.DataFrame()
    if not match.empty and "keep_fraction" in match.columns:
        keep = pd.to_numeric(match["keep_fraction"], errors="coerce").to_numpy(dtype=float)
        match = match[np.isclose(keep, float(ctx.cfg.peak_cue_main_keep_fraction))].copy()
    if not match.empty and "cue_condition" in match.columns:
        match = match[match["cue_condition"].astype(str).isin(CUE_CONDITIONS)].copy()
    quantile_by_condition: dict[str, float] = {}
    fraction_by_condition: dict[str, float] = {}
    if not panel_raw.empty:
        for cue_condition, part in panel_raw.groupby("cue_condition", sort=True):
            quantile_by_condition[str(cue_condition)] = _mean_numeric(part, "support_quantile_mean")
            fraction_by_condition[str(cue_condition)] = _mean_numeric(part, "cue_fraction_actual")
    rows: list[dict[str, Any]] = []
    if not match.empty:
        for _, row in match.iterrows():
            cue_condition = str(row.get("cue_condition", ""))
            rows.append(
                {
                    "network_seed": int(row.get("network_seed", ctx.cfg.network_seed)),
                    "cue_condition": cue_condition,
                    "keep_fraction": float(row.get("keep_fraction", ctx.cfg.peak_cue_main_keep_fraction)),
                    "cue_pixel_count": _row_float(row, "cue_pixel_count", "cue_pixel_count_mean"),
                    "cue_fraction_actual": fraction_by_condition.get(cue_condition, _row_float(row, "cue_fraction_actual", "cue_fraction_actual_mean")),
                    "cue_energy": _row_float(row, "cue_energy", "cue_energy_mean"),
                    "encoded_spike_count": _row_float(row, "encoded_spike_count", "encoded_spike_count_mean"),
                    "support_mean_selected": _row_float(row, "support_mean_selected"),
                    "support_quantile_mean": quantile_by_condition.get(cue_condition, _row_float(row, "support_quantile_mean")),
                    "n_masks": int(row.get("n_masks", 0)),
                }
            )
    elif not panel_raw.empty:
        for cue_condition, part in panel_raw.groupby("cue_condition", sort=True):
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "cue_condition": str(cue_condition),
                    "keep_fraction": float(ctx.cfg.peak_cue_main_keep_fraction),
                    "cue_pixel_count": _mean_numeric(part, "cue_pixel_count"),
                    "cue_fraction_actual": _mean_numeric(part, "cue_fraction_actual"),
                    "cue_energy": _mean_numeric(part, "cue_energy"),
                    "encoded_spike_count": _mean_numeric(part, "encoded_spike_count"),
                    "support_mean_selected": _mean_numeric(part, "support_mean_selected"),
                    "support_quantile_mean": _mean_numeric(part, "support_quantile_mean"),
                    "n_masks": int(part[["sequence_id", "repeat_id", "mask_id"]].drop_duplicates().shape[0]) if {"sequence_id", "repeat_id", "mask_id"}.issubset(part.columns) else int(len(part)),
                }
            )
    return pd.DataFrame(rows, columns=columns)

def compute_peak_cue_serial_position_metrics(ctx: ExperimentContext) -> None:
    raw_path = ctx.raw_dir / "supp_structural_weak_cue_trial_readout.csv"
    metric_columns = [
        "network_seed",
        "seq_len",
        "target_position",
        "target_position_bin",
        "relative_position",
        "cue_condition",
        "memory_condition",
        "keep_fraction",
        "P_target",
        "P_seen_item",
        "P_unseen",
        "P_silent",
        "n_trials",
    ]
    gain_columns = [
        "network_seed",
        "seq_len",
        "target_position",
        "target_position_bin",
        "relative_position",
        "cue_condition",
        "keep_fraction",
        "P_target_sequence_state",
        "P_target_cue_only",
        "memory_gain",
        "n_trials",
    ]
    if not raw_path.exists():
        ctx.warnings.append("Peak-cue serial-position supplement skipped because structural weak-cue raw output is missing.")
        _save_csv(ctx, pd.DataFrame(columns=metric_columns), ctx.metrics_dir / "supp_peak_cue_serial_position_metrics.csv")
        _save_csv(ctx, pd.DataFrame(columns=gain_columns), ctx.metrics_dir / "supp_peak_cue_serial_position_gain.csv")
        return
    raw = pd.read_csv(raw_path)
    if raw.empty:
        _save_csv(ctx, pd.DataFrame(columns=metric_columns), ctx.metrics_dir / "supp_peak_cue_serial_position_metrics.csv")
        _save_csv(ctx, pd.DataFrame(columns=gain_columns), ctx.metrics_dir / "supp_peak_cue_serial_position_gain.csv")
        return
    raw = raw.copy()
    if "target_source" in raw.columns:
        raw = raw[raw["target_source"].astype(str).eq("sequence_member_random")].copy()
    raw = raw[raw["cue_condition"].astype(str).isin(CUE_CONDITIONS)].copy()
    raw = raw[raw["memory_condition"].astype(str).isin(MEMORY_CONDITIONS)].copy()
    raw["target_position_bin"] = raw.apply(lambda row: _target_position_bin(row.get("target_position", -1), row.get("seq_len", 0)), axis=1)
    raw["relative_position"] = pd.to_numeric(raw["target_position"], errors="coerce") / pd.to_numeric(raw["seq_len"], errors="coerce").replace(0, np.nan)
    metric_rows: list[dict[str, Any]] = []
    for keys, part in raw.groupby(["network_seed", "seq_len", "target_position", "target_position_bin", "relative_position", "cue_condition", "memory_condition", "keep_fraction"], sort=True):
        seed, seq_len, target_position, target_position_bin, relative_position, cue_condition, memory_condition, keep_fraction = keys
        denom = max(1, len(part))
        metric_rows.append(
            {
                "network_seed": int(seed),
                "seq_len": int(seq_len),
                "target_position": int(target_position),
                "target_position_bin": str(target_position_bin),
                "relative_position": float(relative_position),
                "cue_condition": str(cue_condition),
                "memory_condition": str(memory_condition),
                "keep_fraction": float(keep_fraction),
                "P_target": float(part["pred_is_target"].sum() / denom),
                "P_seen_item": float(part["pred_is_seen_item"].sum() / denom),
                "P_unseen": float(part["pred_is_unseen"].sum() / denom),
                "P_silent": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    metrics = pd.DataFrame(metric_rows, columns=metric_columns)
    gain_rows: list[dict[str, Any]] = []
    if not metrics.empty:
        for keys, part in metrics.groupby(["network_seed", "seq_len", "target_position", "target_position_bin", "relative_position", "cue_condition", "keep_fraction"], sort=True):
            seed, seq_len, target_position, target_position_bin, relative_position, cue_condition, keep_fraction = keys
            seq = part[part["memory_condition"].astype(str).eq("sequence_state")]
            cue = part[part["memory_condition"].astype(str).eq("cue_only")]
            p_target_seq = _first_float(seq, "P_target")
            p_target_cue = _first_float(cue, "P_target")
            gain_rows.append(
                {
                    "network_seed": int(seed),
                    "seq_len": int(seq_len),
                    "target_position": int(target_position),
                    "target_position_bin": str(target_position_bin),
                    "relative_position": float(relative_position),
                    "cue_condition": str(cue_condition),
                    "keep_fraction": float(keep_fraction),
                    "P_target_sequence_state": p_target_seq,
                    "P_target_cue_only": p_target_cue,
                    "memory_gain": float(p_target_seq - p_target_cue),
                    "n_trials": int(min(_first_float(seq, "n_trials"), _first_float(cue, "n_trials"))),
                }
            )
    gain = pd.DataFrame(gain_rows, columns=gain_columns)
    _save_csv(ctx, metrics, ctx.metrics_dir / "supp_peak_cue_serial_position_metrics.csv")
    _save_csv(ctx, gain, ctx.metrics_dir / "supp_peak_cue_serial_position_gain.csv")
    ctx.completed_modules["peak_cue_serial_position"] = True

def _target_position_bin(target_position: Any, seq_len: Any) -> str:
    try:
        pos = int(target_position)
        length = int(seq_len)
    except (TypeError, ValueError):
        return "unknown"
    if pos < 1 or length < 1:
        return "unknown"
    if pos == length:
        return "latest"
    if pos >= length - 2:
        return "recent"
    if pos <= 2:
        return "early"
    return "middle"

def _fig3f_cue_gain(part: pd.DataFrame, *, min_keep: float = -np.inf, max_keep: float = np.inf) -> float:
    sub = part[(pd.to_numeric(part["keep_prob"], errors="coerce") > float(min_keep)) & (pd.to_numeric(part["keep_prob"], errors="coerce") <= float(max_keep))]
    if sub.empty:
        return float("nan")
    pivot = sub.pivot_table(index="keep_prob", columns="memory_condition", values="P_target", aggfunc="mean")
    if not {"sequence_state", "cue_only"}.issubset(pivot.columns):
        return float("nan")
    return float((pivot["sequence_state"] - pivot["cue_only"]).mean())
