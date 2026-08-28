from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.common.ping_common import LAYER_KEYS
from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig2.constants import RESIDUAL_TEMPLATE_DEFINITION, STATE_VARIABLES
from src.experiments.paper_figures.fig2.subexperiments.helpers import (
    _centered_cosine,
    _fit_mixture_models,
    _maybe_float,
    _progress,
)
from src.experiments.paper_figures.fig2.subexperiments.morphology import _write_layerwise_morphology_metrics
from src.experiments.paper_figures.fig2.types import ExperimentContext, PairEpisodeStateBank

def compute_linear_mixture_metrics(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows = []
    supp_rows = []
    null_rows = []
    for layer in _progress(LAYER_KEYS, total=len(LAYER_KEYS), desc="fig2 mixture layers", enabled=ctx.cfg.show_progress):
        for variable in STATE_VARIABLES:
            z0 = bank.get("S0", layer, variable)
            x_a = bank.get("S_A", layer, variable) - z0
            x_b = bank.get("S_B", layer, variable) - z0
            y = bank.get("S_AB", layer, variable) - z0
            for idx, pair_id in enumerate(bank.pair_trials["pair_id"].to_numpy(dtype=np.int64)):
                models = _fit_mixture_models(x_a[idx], x_b[idx], y[idx], ctx.cfg.linear_mixture_cv_folds, ctx.cfg.network_seed + idx)
                best_single = max(models["A_only"]["r2"], models["B_only"]["r2"])
                for model_name, metrics in models.items():
                    cosine_to_sab = float(_centered_cosine(y[idx], metrics["prediction"]))
                    row = {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(pair_id),
                        "layer": layer,
                        "state_variable": variable,
                        "model_name": model_name,
                        "r2": float(metrics["r2"]),
                        "cv_r2": float(metrics["cv_r2"]),
                        "residual_norm": float(metrics["residual_norm"]),
                        "target_norm": float(metrics["target_norm"]),
                        "residual_norm_ratio": float(metrics["residual_norm_ratio"]),
                        "beta_A": _maybe_float(metrics.get("beta_A")),
                        "beta_B": _maybe_float(metrics.get("beta_B")),
                        "intercept": _maybe_float(metrics.get("intercept")),
                        "convex_weight_A": _maybe_float(metrics.get("convex_weight_A")),
                        "convex_weight_B": _maybe_float(metrics.get("convex_weight_B")),
                        "best_single_constituent_r2": float(best_single),
                        "linear_mixture_gain": float(models["unconstrained_AB"]["r2"] - best_single),
                    }
                    rows.append(row)
                    supp_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "pair_id": int(pair_id),
                            "layer": layer,
                            "state_variable": variable,
                            "model_name": model_name,
                            "mixture_model": model_name,
                            "r2": float(metrics["r2"]),
                            "fit_r2": float(metrics["r2"]),
                            "cv_r2": float(metrics["cv_r2"]),
                            "fold_id": "",
                            "cv_fold": "",
                            "residual_norm": float(metrics["residual_norm"]),
                            "residual_norm_ratio": float(metrics["residual_norm_ratio"]),
                            "cosine_to_SAB": cosine_to_sab,
                            "n_pairs": int(len(bank.pair_trials)),
                            "notes": "",
                        }
                    )
                for null_name in ("mean_AB", "sum_AB", "unconstrained_AB", "convex_AB"):
                    pred = models[null_name]["prediction"]
                    null_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "pair_id": int(pair_id),
                            "layer": layer,
                            "state_variable": variable,
                            "null_model": null_name.replace("_AB", "").replace("unconstrained", "LS"),
                            "similarity_to_SAB": float(_centered_cosine(y[idx], pred)),
                            "r2_to_SAB": float(models[null_name]["r2"]),
                            "residual_norm_ratio": float(models[null_name]["residual_norm_ratio"]),
                            "notes": "baseline_subtracted_against_S0",
                        }
                    )
    _save_csv(ctx, pd.DataFrame(rows).drop(columns=["prediction"], errors="ignore"), ctx.metrics_dir / "panel_d_linear_mixture_fit_metrics.csv")
    _save_csv(ctx, pd.DataFrame(supp_rows), ctx.metrics_dir / "supp_linear_mixture_model_comparison.csv")
    _save_csv(ctx, pd.DataFrame(null_rows), ctx.metrics_dir / "supp_additive_null_metrics.csv")
    ctx.completed_modules["linear_mixture"] = True

def compute_linear_residual_pair_specificity(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows = []
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 303)
    n = len(bank.pair_trials)
    for layer in LAYER_KEYS:
        for variable in STATE_VARIABLES:
            z0 = bank.get("S0", layer, variable)
            x_a = bank.get("S_A", layer, variable) - z0
            x_b = bank.get("S_B", layer, variable) - z0
            y = bank.get("S_AB", layer, variable) - z0
            for idx, pair_id in enumerate(bank.pair_trials["pair_id"].to_numpy(dtype=np.int64)):
                model = _fit_mixture_models(x_a[idx], x_b[idx], y[idx], ctx.cfg.linear_mixture_cv_folds, ctx.cfg.network_seed + idx)["unconstrained_AB"]
                residual = y[idx] - model["prediction"]
                true_template = y[idx] - 0.5 * (x_a[idx] + x_b[idx])
                true_score = float(_centered_cosine(residual, true_template))
                choices = [j for j in range(n) if j != idx] or [idx]
                sampled = rng.choice(choices, size=int(ctx.cfg.n_shuffle), replace=len(choices) < int(ctx.cfg.n_shuffle))
                scores = [float(_centered_cosine(residual, y[idx] - 0.5 * (x_a[idx] + x_b[int(j)]))) for j in sampled]
                shuf = float(np.mean(scores)) if scores else float("nan")
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(pair_id),
                        "layer": layer,
                        "state_variable": variable,
                        "residual_true_pair_score": true_score,
                        "residual_shuffled_pair_score": shuf,
                        "residual_pair_specificity": float(true_score - shuf),
                        "beyond_linear_pair_index": float(true_score - shuf),
                        "shuffle_id": "mean",
                        "n_shuffle": int(ctx.cfg.n_shuffle),
                        "residual_template_definition": RESIDUAL_TEMPLATE_DEFINITION,
                    }
                )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "panel_d_linear_residual_pair_specificity_metrics.csv")
    _write_layerwise_morphology_metrics(ctx)
    ctx.completed_modules["linear_residual_pair_specificity"] = True
