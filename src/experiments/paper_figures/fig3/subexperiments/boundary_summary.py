from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig3.types import ExperimentContext


def compute_morphology_function_coupling(
    ctx: ExperimentContext,
    morphology_tables: dict[str, pd.DataFrame],
    weak_cue_tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    item_support = morphology_tables.get("morphology_item_support", pd.DataFrame())
    gain = weak_cue_tables.get("item_functional_gain", pd.DataFrame())
    if item_support.empty or gain.empty:
        coupling = pd.DataFrame()
    else:
        coupling = item_support.merge(
            gain,
            left_on=["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms", "serial_position"],
            right_on=["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms", "target_position"],
            how="inner",
        )
        if not coupling.empty:
            coupling["morphology_support_p"] = pd.to_numeric(coupling["p_i"], errors="coerce")
            coupling["morphology_support_beta"] = pd.to_numeric(coupling["beta"], errors="coerce")
            coupling["functional_gain_norm"] = pd.to_numeric(coupling["G_i_norm"], errors="coerce")
            coupling["functional_gain"] = pd.to_numeric(coupling["G_i"], errors="coerce")
    summary = _coupling_summary(coupling)
    order_control = _order_specificity_control(item_support)
    _save_csv(ctx, coupling, ctx.metrics_dir / "panel_e_morphology_function_coupling.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_e_coupling_summary.csv")
    _save_csv(ctx, order_control, ctx.metrics_dir / "panel_f_order_specificity_control.csv")
    ctx.completed_modules["morphology_function_coupling"] = True
    return {
        "morphology_function_coupling": coupling,
        "coupling_summary": summary,
        "order_specificity_control": order_control,
    }


def compute_boundary_summary(
    ctx: ExperimentContext,
    morphology_tables: dict[str, pd.DataFrame],
    weak_cue_tables: dict[str, pd.DataFrame],
    neutral_ping_tables: dict[str, pd.DataFrame],
    coupling_tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    morphology = morphology_tables.get("morphology_boundary_metrics", pd.DataFrame())
    functional = weak_cue_tables.get("functional_boundary_metrics", pd.DataFrame())
    coupling = coupling_tables.get("coupling_summary", pd.DataFrame())
    ping = neutral_ping_tables.get("neutral_ping_access_summary", pd.DataFrame())
    rows: list[dict[str, Any]] = []
    if not morphology.empty:
        for keys, part in morphology.groupby(["network_seed", "condition_id", "seq_len", "delay_ms"], sort=True):
            row = dict(zip(["network_seed", "condition_id", "seq_len", "delay_ms"], keys))
            row["N_eff"] = float(pd.to_numeric(part["N_eff"], errors="coerce").mean()) if "N_eff" in part.columns else float("nan")
            if "N_eff_fraction" in part.columns:
                row["N_eff_fraction"] = float(pd.to_numeric(part["N_eff_fraction"], errors="coerce").mean())
            else:
                seq_len = float(row["seq_len"])
                row["N_eff_fraction"] = float(row["N_eff"] / seq_len) if np.isfinite(row["N_eff"]) and seq_len > 0 else float("nan")
            row["multi_item_retention_index"] = float(pd.to_numeric(part["multi_item_retention_index"], errors="coerce").mean())
            row["latest_collapse_index"] = float(pd.to_numeric(part["latest_collapse_index"], errors="coerce").mean())
            fpart = _match(functional, row)
            if not fpart.empty:
                for col in (
                    "accessible_item_count",
                    "singleton_access_count",
                    "sequence_access_count",
                    "rescued_count",
                    "singleton_access_fraction",
                    "sequence_access_fraction",
                    "rescued_fraction",
                    "functional_retention_index",
                    "mean_G_i",
                    "mean_U_i",
                    "mean_G_i_norm",
                ):
                    if col in fpart.columns:
                        row[col] = float(pd.to_numeric(fpart[col], errors="coerce").mean())
            cpart = _match(coupling, row)
            if not cpart.empty and "support_gain_corr" in cpart.columns:
                row["support_gain_corr"] = float(pd.to_numeric(cpart["support_gain_corr"], errors="coerce").mean())
            ppart = _match(ping, row)
            if not ppart.empty and "latest_item_mass" in ppart.columns:
                row["ping_latest_item_mass"] = float(pd.to_numeric(ppart["latest_item_mass"], errors="coerce").mean())
            rows.append(row)
    summary = pd.DataFrame(rows)
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_f_boundary_summary.csv")
    ctx.completed_modules["boundary_summary"] = True
    return {"boundary_summary": summary}


def _coupling_summary(coupling: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "condition_id", "seq_len", "delay_ms", "support_gain_corr", "n_items", "mean_functional_gain_norm"]
    if coupling.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_cols = ["network_seed", "condition_id", "seq_len", "delay_ms"]
    for keys, part in coupling.groupby(group_cols, sort=True):
        x = pd.to_numeric(part["morphology_support_p"], errors="coerce")
        y = pd.to_numeric(part["functional_gain_norm"], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() >= 2 and float(x[valid].std()) > 0 and float(y[valid].std()) > 0:
            corr = float(np.corrcoef(x[valid].to_numpy(dtype=float), y[valid].to_numpy(dtype=float))[0, 1])
        else:
            corr = float("nan")
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "support_gain_corr": corr,
                "n_items": int(valid.sum()),
                "mean_functional_gain_norm": float(y[valid].mean()) if valid.any() else float("nan"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _order_specificity_control(item_support: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms", "condition", "order_specificity_index", "serial_support_corr"]
    if item_support.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_cols = ["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms"]
    for keys, part in item_support.groupby(group_cols, sort=True):
        ordered = part.sort_values("serial_position")
        p = pd.to_numeric(ordered["p_i"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x = np.linspace(0.0, 1.0, num=len(p)) if len(p) else np.asarray([], dtype=float)
        true_corr = _corr(x, p)
        shuffled = []
        if len(p) > 1:
            for shift in range(1, len(p)):
                shuffled.append(_corr(x, np.roll(p, shift)))
            shuffled.append(_corr(x, p[::-1]))
        null_mean = float(np.nanmean(shuffled)) if shuffled else float("nan")
        row_base = dict(zip(group_cols, keys))
        rows.append({**row_base, "condition": "true_order", "order_specificity_index": true_corr - null_mean if np.isfinite(null_mean) else true_corr, "serial_support_corr": true_corr})
        rows.append({**row_base, "condition": "shuffled_order_null", "order_specificity_index": 0.0, "serial_support_corr": null_mean})
    return pd.DataFrame(rows, columns=columns)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _match(df: pd.DataFrame, row: MappingLike) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ("network_seed", "condition_id", "seq_len", "delay_ms"):
        if col in out.columns and col in row:
            out = out[out[col].astype(str).eq(str(row[col]))]
    return out


MappingLike = dict[str, Any]


__all__ = ["compute_boundary_summary", "compute_morphology_function_coupling"]
