from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig3.constants import NUM_CLASSES
from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _progress, _run_ping_from_boundary
from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _main_sequence_meta
from src.experiments.paper_figures.fig3.types import ExperimentContext, MultiItemSequenceLandscapeBank

def run_neutral_ping_from_final_state(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    run_neutral_ping_readout_distribution(ctx, bank)

def run_neutral_ping_readout_distribution(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    raw_rows: list[dict[str, Any]] = []
    main_meta = _main_sequence_meta(ctx, bank)
    for _, meta in _progress(main_meta.iterrows(), total=len(main_meta), desc="fig3 ping sequences", enabled=ctx.cfg.show_progress):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        labels = [int(v) for v in str(meta["ordered_item_labels"]).split(";")]
        for ping_repeat in _progress(range(int(ctx.cfg.ping_repeats)), total=int(ctx.cfg.ping_repeats), desc="fig3 ping repeats", enabled=ctx.cfg.show_progress):
            ping_seed = int(ctx.cfg.network_seed) * 100000 + seq_id * 100 + ping_repeat
            for state_condition in ctx.cfg.ping_main_state_conditions:
                state_key = "S_final" if str(state_condition) == "S_final" else "S0"
                boundary = bank.boundaries[seq_id][state_key]
                pred, fire, ping_energy, ping_spike_count, restore_info = _run_ping_from_boundary(ctx, boundary)
                position = labels.index(pred) + 1 if pred in labels else -1
                silent = pred < 0
                memory_condition = "sequence_state" if state_key == "S_final" else "cue_only"
                raw_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": seq_id,
                        "seq_len": seq_len,
                        "ordered_item_labels": ";".join(str(v) for v in labels),
                        "ping_repeat": int(ping_repeat),
                        "ping_seed": int(ping_seed),
                        "state_condition": str(state_condition),
                        "memory_condition": memory_condition,
                        "predicted_label": int(pred),
                        "predicted_position": int(position),
                        "pred_is_seen_item": int(position > 0),
                        "pred_is_unseen": int((not silent) and position < 0),
                        "silent": int(silent),
                        "first_fire_time_ms": int(fire),
                        "ping_energy": float(ping_energy),
                        "ping_spike_count": float(ping_spike_count),
                        "restore_mode": str(ctx.cfg.functional_restore_mode),
                        "stsp_only_restore": int(str(ctx.cfg.functional_restore_mode) == "stsp_only"),
                        "fast_state_reset": int(str(ctx.cfg.functional_restore_mode) == "stsp_only"),
                        "restore_ok": int(restore_info.get("restore_ok", 1)),
                    }
                )
    raw_columns = [
        "network_seed",
        "sequence_id",
        "seq_len",
        "ordered_item_labels",
        "ping_repeat",
        "ping_seed",
        "state_condition",
        "memory_condition",
        "predicted_label",
        "predicted_position",
        "pred_is_seen_item",
        "pred_is_unseen",
        "silent",
        "first_fire_time_ms",
        "ping_energy",
        "ping_spike_count",
        "restore_mode",
        "stsp_only_restore",
        "fast_state_reset",
        "restore_ok",
    ]
    raw = pd.DataFrame(raw_rows, columns=raw_columns)
    _save_csv(ctx, raw, ctx.raw_dir / "panel_d_neutral_ping_trial_readout.csv")
    _save_csv(ctx, raw.copy(), ctx.raw_dir / "panel_e_neutral_ping_trial_readout.csv")

    serial_bins = [f"pos_{idx}" for idx in range(1, int(ctx.cfg.main_sequence_length) + 1)] + ["other", "silent"]
    pos_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    recency_rows: list[dict[str, Any]] = []
    if not raw.empty:
        raw = raw.copy()
        raw["serial_bin"] = raw.apply(_serial_bin_for_ping_row, axis=1)
        raw["class_label"] = raw.apply(lambda r: "silent" if int(r.get("silent", 0)) else str(int(r.get("predicted_label", -1))), axis=1)
        for (network_seed, state_condition, seq_len), part in raw.groupby(["network_seed", "state_condition", "seq_len"], sort=True):
            for serial_bin in serial_bins:
                pos_rows.append(
                    {
                        "network_seed": int(network_seed),
                        "state_condition": str(state_condition),
                        "seq_len": int(seq_len),
                        "serial_bin": serial_bin,
                        "readout_mass": float((part["serial_bin"] == serial_bin).mean()),
                        "n_trials": int(len(part)),
                    }
                )
            for class_label in [str(v) for v in range(NUM_CLASSES)] + ["silent"]:
                class_rows.append(
                    {
                        "network_seed": int(network_seed),
                        "state_condition": str(state_condition),
                        "class_label": class_label,
                        "readout_mass": float((part["class_label"] == class_label).mean()),
                        "n_trials": int(len(part)),
                    }
                )
            seen = part[part["pred_is_seen_item"] == 1]
            positions = pd.to_numeric(part["predicted_position"], errors="coerce")
            latest = float((positions == int(seq_len)).mean())
            earlier = float(((positions > 0) & (positions < int(seq_len))).mean())
            summary_rows.append(
                {
                    "network_seed": int(network_seed),
                    "state_condition": str(state_condition),
                    "seq_len": int(seq_len),
                    "P_seen_item": float(part["pred_is_seen_item"].mean()),
                    "P_unseen": float(part["pred_is_unseen"].mean()),
                    "P_silent": float(part["silent"].mean()),
                    "mean_first_fire_time_ms": float(pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan).mean()),
                    "n_trials": int(len(part)),
                }
            )
            recency_rows.append(
                {
                    "network_seed": int(network_seed),
                    "state_condition": str(state_condition),
                    "seq_len": int(seq_len),
                    "ping_COM": float(positions[positions > 0].mean()) if not seen.empty else float("nan"),
                    "latest_item_mass": latest,
                    "earlier_item_residual_mass": earlier,
                    "earlier_item_above_null": float(earlier - ((int(seq_len) - 1) / NUM_CLASSES)),
                    "n_trials": int(len(part)),
                }
            )
    pos_df = pd.DataFrame(pos_rows)
    class_df = pd.DataFrame(class_rows)
    summary_df = pd.DataFrame(summary_rows)
    _save_csv(ctx, pos_df, ctx.metrics_dir / "panel_d_ping_position_distribution.csv")
    _save_csv(ctx, class_df, ctx.metrics_dir / "panel_d_ping_class_distribution.csv")
    _save_csv(ctx, summary_df, ctx.metrics_dir / "panel_d_ping_summary.csv")
    _save_csv(ctx, pos_df.copy(), ctx.metrics_dir / "panel_e_ping_position_distribution.csv")
    _save_csv(ctx, class_df.copy(), ctx.metrics_dir / "panel_e_ping_class_distribution.csv")
    _save_csv(ctx, summary_df.copy(), ctx.metrics_dir / "panel_e_ping_summary.csv")
    _save_csv(ctx, pd.DataFrame(recency_rows), ctx.metrics_dir / "supp_ping_recency_diagnostics.csv")
    ctx.completed_modules["neutral_ping"] = True

def _serial_bin_for_ping_row(row: pd.Series) -> str:
    if int(row.get("silent", 0)):
        return "silent"
    position = int(row.get("predicted_position", -1))
    seq_len = int(row.get("seq_len", 0))
    if 1 <= position <= seq_len:
        return f"pos_{position}"
    return "other"
