from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as legacy
from src.experiments.paper_figures.fig3.types import ExperimentContext, MultiItemSequenceLandscapeBank


def compute_morphology_decomposition(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    condition_specs: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    item_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    layer = str(ctx.cfg.morphology_layer)
    variable = str(ctx.cfg.morphology_variable)
    for _, condition in condition_specs.iterrows():
        condition_id = str(condition["condition_id"])
        seq_len = int(condition["seq_len"])
        delay_ms = int(condition["delay_ms"])
        sequence_meta = bank.sequence_meta[bank.sequence_meta["seq_len"].astype(int).eq(seq_len)]
        if "condition_id" in sequence_meta.columns:
            sequence_meta = sequence_meta[sequence_meta["condition_id"].astype(str).eq(condition_id)]
        for _, meta in sequence_meta.iterrows():
            seq_id = int(meta.get("source_sequence_id", meta["sequence_id"]))
            baseline = np.asarray(bank.get(seq_id, "S0", layer, variable, condition_id=condition_id, delay_ms=delay_ms), dtype=float).ravel()
            final = np.asarray(bank.get(seq_id, "S_final", layer, variable, condition_id=condition_id, delay_ms=delay_ms), dtype=float).ravel()
            target = final - baseline
            refs = []
            for pos in range(1, seq_len + 1):
                ref = np.asarray(bank.singleton_ref(seq_id, pos, layer, variable, condition_id=condition_id, delay_ms=delay_ms), dtype=float).ravel() - baseline
                refs.append(ref)
            design = np.column_stack(refs) if refs else np.empty((target.size, 0))
            beta = _nnls(design, target)
            beta_sum = float(np.sum(beta))
            if beta.size == 0:
                p = np.asarray([], dtype=float)
            elif beta_sum > 1e-12:
                p = beta / beta_sum
            else:
                p = np.full(beta.size, 1.0 / beta.size, dtype=float)
            recon = design @ beta if beta.size else np.zeros_like(target)
            denom = float(np.sum(target * target))
            reconstruction_r2 = 1.0 - float(np.sum((target - recon) ** 2)) / denom if denom > 1e-12 else float("nan")
            n_eff = float(1.0 / np.sum(p * p)) if p.size and np.sum(p * p) > 0 else 0.0
            n_eff_fraction = float(n_eff / seq_len) if seq_len > 0 else float("nan")
            retention = float((n_eff - 1.0) / max(seq_len - 1, 1))
            latest_mass = float(p[-1]) if p.size else float("nan")
            latest_collapse = float((latest_mass - (1.0 / seq_len)) / max(1.0 - (1.0 / seq_len), 1e-12)) if p.size else float("nan")
            serial_positions = np.arange(1, seq_len + 1, dtype=float)
            serial_norm = (serial_positions - 1.0) / max(seq_len - 1, 1)
            serial_com = float(np.sum(serial_norm * p)) if p.size else float("nan")
            labels = [int(v) for v in str(meta.get("ordered_item_labels", "")).split(";") if str(v) != ""]
            image_ids = [int(v) for v in str(meta.get("ordered_item_ids", "")).split(";") if str(v) != ""]
            for idx, pos in enumerate(range(1, seq_len + 1)):
                row = {
                    "network_seed": int(ctx.cfg.network_seed),
                    "condition_id": condition_id,
                    "sequence_id": seq_id,
                    "seq_len": seq_len,
                    "delay_ms": delay_ms,
                    "layer": layer,
                    "state_variable": variable,
                    "serial_position": int(pos),
                    "serial_position_norm": float(serial_norm[idx]),
                    "item_image_id": int(image_ids[idx]) if idx < len(image_ids) else -1,
                    "item_label": int(labels[idx]) if idx < len(labels) else -1,
                    "beta": float(beta[idx]) if idx < len(beta) else 0.0,
                    "p_i": float(p[idx]) if idx < len(p) else 0.0,
                    "is_latest": int(pos == seq_len),
                    "N_eff": n_eff,
                    "N_eff_fraction": n_eff_fraction,
                    "multi_item_retention_index": retention,
                    "latest_collapse_index": latest_collapse,
                    "serial_COM": serial_com,
                    "reconstruction_R2": reconstruction_r2,
                }
                item_rows.append(row)
                profile_rows.append(row)
            boundary_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "condition_id": condition_id,
                    "sequence_id": seq_id,
                    "seq_len": seq_len,
                    "delay_ms": delay_ms,
                    "layer": layer,
                    "state_variable": variable,
                    "N_eff": n_eff,
                    "N_eff_fraction": n_eff_fraction,
                    "multi_item_retention_index": retention,
                    "latest_mass": latest_mass,
                    "latest_collapse_index": latest_collapse,
                    "serial_COM": serial_com,
                    "reconstruction_R2": reconstruction_r2,
                    "beta_sum": beta_sum,
                }
            )
    tables = {
        "morphology_item_support": pd.DataFrame(item_rows),
        "morphology_serial_profile": pd.DataFrame(profile_rows),
        "morphology_boundary_metrics": pd.DataFrame(boundary_rows),
    }
    legacy._save_csv(ctx, tables["morphology_item_support"], ctx.raw_dir / "panel_b_morphology_item_support.csv")
    legacy._save_csv(ctx, tables["morphology_serial_profile"], ctx.metrics_dir / "panel_b_morphology_serial_profile.csv")
    legacy._save_csv(ctx, tables["morphology_boundary_metrics"], ctx.metrics_dir / "panel_c_morphology_boundary_metrics.csv")
    ctx.completed_modules["morphology_decomposition"] = True
    return tables


def _nnls(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    if design.size == 0:
        return np.asarray([], dtype=float)
    try:
        from scipy.optimize import nnls

        beta, _ = nnls(np.asarray(design, dtype=float), np.asarray(target, dtype=float))
        return np.asarray(beta, dtype=float)
    except Exception:
        beta, *_ = np.linalg.lstsq(np.asarray(design, dtype=float), np.asarray(target, dtype=float), rcond=None)
        return np.maximum(np.asarray(beta, dtype=float), 0.0)


__all__ = ["compute_morphology_decomposition"]
