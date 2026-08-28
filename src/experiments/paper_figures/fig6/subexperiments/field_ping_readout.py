from __future__ import annotations

from typing import Any

import pandas as pd

from src.experiments.paper_figures.fig6.constants import PANEL_B_REGION_PING_COLUMNS
from src.experiments.paper_figures.fig6.subexperiments.helpers_1 import (
    _gain_ratio_audit_row,
    _make_score_region_ping_masks,
    _progress,
    _record_gain_ratio_audit,
    _run_masked_ping_layer1_capture,
    _save_csv,
    _sequence_labels_from_meta,
    _serial_age_bin,
    _serial_position_for_label,
    compute_gain_ratio_map,
)
from src.experiments.paper_figures.fig6.types import ExperimentContext, PeakAmplifiedReentryBank



def compute_field_ping_readout(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    rows: list[dict[str, Any]] = []
    for seq_idx, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 field ping readout", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        labels = _sequence_labels_from_meta(meta)
        rho = compute_gain_ratio_map(
            bank.g_final[seq_idx].reshape(28, 28),
            bank.g_baseline[seq_idx].reshape(28, 28),
            eps=float(ctx.cfg.score_eps),
            clip_quantiles=tuple(ctx.cfg.gain_ratio_clip_quantiles),
            use_log=bool(ctx.cfg.score_use_log_gain),
        )
        _record_gain_ratio_audit(ctx, _gain_ratio_audit_row(ctx, seq_id, rho, bank.g_final[seq_idx], bank.g_baseline[seq_idx]))
        masks = _make_score_region_ping_masks(rho, float(ctx.cfg.basin_top_q), int(ctx.cfg.network_seed) + seq_id)
        for entry_condition, entry_mask in masks.items():
            pred, fire_ms, total_current, active_sites, _trace = _run_masked_ping_layer1_capture(
                ctx,
                bank.boundaries.get(seq_id),
                entry_mask,
                float(ctx.cfg.ping_amp),
                int(ctx.cfg.ping_steps),
            )
            serial_position = _serial_position_for_label(labels, pred)
            serial_bin = _serial_age_bin(serial_position, len(labels), pred)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": seq_id,
                    "entry_condition": str(entry_condition),
                    "readout_label": int(pred),
                    "readout_serial_position": int(serial_position),
                    "old_mass": float(serial_bin == "old"),
                    "middle_mass": float(serial_bin == "middle"),
                    "recent_mass": float(serial_bin == "recent"),
                    "other_mass": float(serial_bin == "other"),
                    "silent_rate": float(serial_bin == "silent"),
                    "n_trials": 1,
                    "ping_active_sites": float(active_sites),
                    "total_ping_current": float(total_current),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        group_cols = ["network_seed", "sequence_id", "entry_condition"]
        agg = (
            df.groupby(group_cols, as_index=False)
            .agg(
                readout_label=("readout_label", lambda s: "aggregate"),
                readout_serial_position=("readout_serial_position", lambda s: -1),
                old_mass=("old_mass", "mean"),
                middle_mass=("middle_mass", "mean"),
                recent_mass=("recent_mass", "mean"),
                other_mass=("other_mass", "mean"),
                silent_rate=("silent_rate", "mean"),
                n_trials=("n_trials", "sum"),
                ping_active_sites=("ping_active_sites", "mean"),
                total_ping_current=("total_ping_current", "mean"),
            )
        )
    else:
        agg = pd.DataFrame(columns=PANEL_B_REGION_PING_COLUMNS)
    _save_csv(ctx, agg.reindex(columns=PANEL_B_REGION_PING_COLUMNS), ctx.metrics_dir / "panel_b_region_ping_readout_bias.csv")
    ctx.completed_modules["field_ping_readout"] = True
