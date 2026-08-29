from __future__ import annotations

import pandas as pd

from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import (
    _matching_diagnostics,
    _overlap_regression,
    _pair_effect_table,
    _panel_c_matched_comparison,
    _two_by_two,
)
from src.experiments.paper_figures.fig4.types import (
    ExperimentContext,
    OverlapPerturbationCompatibleBank,
    OverlapReentryDMSBank,
)

def compute_overlap_localization_metrics(ctx: ExperimentContext, bank: OverlapReentryDMSBank | OverlapPerturbationCompatibleBank) -> None:
    b_path = ctx.metrics_dir / "panel_b_similarity_entry_metrics.csv"
    effect = _pair_effect_table(ctx, bank)
    if b_path.exists():
        b_base = pd.read_csv(b_path)
        keep = [c for c in ("pair_id", "b_vec", "acc_drop") if c in b_base.columns]
        if keep:
            effect = effect.drop(columns=[c for c in ("b_vec", "acc_drop") if c in effect.columns], errors="ignore").merge(
                b_base[keep],
                on="pair_id",
                how="left",
            )
    merged = bank.pair_trials.merge(effect[["pair_id", "b_vec", "DPI_L3", "acc_drop", "decision_deflection"]], on="pair_id", how="left")
    rows = []
    for _, r in merged.iterrows():
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "pair_id": int(r["pair_id"]),
                "similarity_bin": str(r["similarity_bin"]),
                "overlap_bin": str(r["overlap_bin"]),
                "pixel_similarity": float(r["pixel_similarity"]),
                "dice_overlap": float(r["dice_overlap"]),
                "input_energy_sample": float(r["input_energy_sample"]),
                "input_energy_probe": float(r["input_energy_probe"]),
                "dynamic_effect_metric": float(r["b_vec"]),
                "b_vec": float(r["b_vec"]),
                "DPI_L3": float(r["DPI_L3"]),
                "acc_drop": float(r["acc_drop"]),
                "decision_deflection": float(r["decision_deflection"]),
                "matched_group_id": str(r.get("matched_group_id", "")),
            }
        )
    loc = pd.DataFrame(rows)
    matched = _panel_c_matched_comparison(merged)
    reg = _overlap_regression(merged, int(ctx.cfg.network_seed))
    two = _two_by_two(merged, int(ctx.cfg.network_seed))
    diag = _matching_diagnostics(merged, int(ctx.cfg.network_seed))
    _save_csv(ctx, loc, ctx.metrics_dir / "panel_c_overlap_localization_metrics.csv")
    _save_csv(ctx, matched, ctx.metrics_dir / "panel_c_overlap_matched_comparison.csv")
    _save_csv(ctx, reg, ctx.metrics_dir / "supp_overlap_similarity_regression.csv")
    _save_csv(ctx, two, ctx.metrics_dir / "supp_overlap_similarity_2x2.csv")
    _save_csv(ctx, diag, ctx.metrics_dir / "supp_overlap_matching_diagnostics.csv")
    ctx.completed_modules["overlap_localization"] = True
