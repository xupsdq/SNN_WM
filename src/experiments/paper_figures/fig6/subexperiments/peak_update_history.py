from __future__ import annotations

from typing import Any

import pandas as pd

from src.experiments.paper_figures.fig6.constants import (
    PANEL_B_UPDATE_HISTORY_COLUMNS,
    PANEL_B_UPDATE_HISTORY_SUMMARY_COLUMNS,
)
from src.experiments.paper_figures.fig6.subexperiments.helpers_1 import _progress, _save_csv
from src.experiments.paper_figures.fig6.subexperiments.helpers_2 import _mean_bool, _mean_col
from src.experiments.paper_figures.fig6.types import ExperimentContext, PeakAmplifiedReentryBank



def compute_peak_update_history(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    windows = tuple(int(v) for v in ctx.cfg.recent_overlap_windows)
    for seq_idx, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 update history", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        seq_len = int(meta.seq_len)
        for unit_id in _progress(range(bank.update_count.shape[1]), total=bank.update_count.shape[1], desc="fig6 update units", enabled=ctx.cfg.show_progress):
            update_count = int(bank.update_count[seq_idx, unit_id])
            last_pos = int(bank.last_update_position[seq_idx, unit_id])
            time_since = int(bank.time_since_last_update[seq_idx, unit_id])
            row = {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": seq_id,
                "seq_len": seq_len,
                "unit_id": int(unit_id),
                "is_peak": bool(bank.peak_mask[seq_idx, unit_id]),
                "is_nonpeak_control": bool(bank.nonpeak_mask[seq_idx, unit_id]),
                "update_count": update_count,
                "last_update_position": last_pos,
                "time_since_last_update": time_since,
                "is_multi_update": bool(update_count >= int(ctx.cfg.multi_update_threshold)),
                "final_support": float(bank.g_final[seq_idx, unit_id]),
                "delta_support": float(bank.delta_support[seq_idx, unit_id]),
            }
            for w in (2, 3, 4, 5):
                row[f"recent_w{w}"] = bool(time_since < w)
                row[f"is_multi_recent_w{w}"] = bool(row["is_multi_update"] and time_since < w)
            rows.append(row)
    df = pd.DataFrame(rows, columns=PANEL_B_UPDATE_HISTORY_COLUMNS)
    _save_csv(ctx, df, ctx.metrics_dir / "panel_b_peak_update_history.csv")
    _save_csv(ctx, df, ctx.raw_dir / "panel_b_peak_update_history.csv")
    groups = {
        "peak": df[df["is_peak"].astype(bool)] if not df.empty else df,
        "nonpeak_control": df[df["is_nonpeak_control"].astype(bool)] if not df.empty else df,
        "prior_updated_nonpeak": df[(~df["is_peak"].astype(bool)) & (pd.to_numeric(df["update_count"], errors="coerce") > 0)] if not df.empty else df,
    }
    for group, part in groups.items():
        summary_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "group": group,
                "mean_update_count": _mean_col(part, "update_count"),
                "P_update_ge_2": _mean_bool(part, pd.to_numeric(part.get("update_count", pd.Series(dtype=float)), errors="coerce") >= 2),
                "P_update_ge_3": _mean_bool(part, pd.to_numeric(part.get("update_count", pd.Series(dtype=float)), errors="coerce") >= 3),
                "mean_time_since_last_update": _mean_col(part, "time_since_last_update"),
                "P_recent_w2": _mean_col(part, "recent_w2"),
                "P_recent_w3": _mean_col(part, "recent_w3"),
                "P_recent_w4": _mean_col(part, "recent_w4"),
                "P_recent_w5": _mean_col(part, "recent_w5"),
                "P_multi_recent_w2": _mean_col(part, "is_multi_recent_w2"),
                "P_multi_recent_w3": _mean_col(part, "is_multi_recent_w3"),
                "P_multi_recent_w4": _mean_col(part, "is_multi_recent_w4"),
                "P_multi_recent_w5": _mean_col(part, "is_multi_recent_w5"),
                "n_units": int(len(part)),
            }
        )
    _save_csv(ctx, pd.DataFrame(summary_rows, columns=PANEL_B_UPDATE_HISTORY_SUMMARY_COLUMNS), ctx.metrics_dir / "panel_b_peak_update_history_summary.csv")
    ctx.completed_modules["peak_update_history"] = True
