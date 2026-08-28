from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig6.constants import (
    PANEL_B_COEF_COLUMNS,
    PANEL_B_METRIC_COLUMNS,
    PRIMARY_LAYER,
    STATE_VARIABLE,
)
from src.experiments.paper_figures.fig6.subexperiments.helpers_1 import _save_csv
from src.experiments.paper_figures.fig6.subexperiments.helpers_2 import _cv_r2, _fit_ols, _standardized_coef
from src.experiments.paper_figures.fig6.types import ExperimentContext, PeakAmplifiedReentryBank



def fit_update_recency_support_models(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    unit_df = pd.read_csv(ctx.metrics_dir / "supp_legacy_panel_a_multi_recent_peak_enrichment.csv")
    metric_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    for sequence_id, part in unit_df.groupby("sequence_id", sort=True):
        y_delta = pd.to_numeric(part["delta_support"], errors="coerce").to_numpy(dtype=float)
        y_final = pd.to_numeric(part["final_support"], errors="coerce").to_numpy(dtype=float)
        features = {
            "baseline_support": pd.to_numeric(part["baseline_support"], errors="coerce").to_numpy(dtype=float),
            "update_count": pd.to_numeric(part["update_count"], errors="coerce").to_numpy(dtype=float),
            "recency": -pd.to_numeric(part["time_since_last_update"], errors="coerce").to_numpy(dtype=float),
            "overlap": pd.to_numeric(part["update_count"], errors="coerce").to_numpy(dtype=float) > 0,
        }
        feature_matrix = {
            "baseline_only": ["baseline_support"],
            "update_only": ["update_count"],
            "recency_only": ["recency"],
            "overlap_only": ["overlap"],
            "update_plus_recency": ["update_count", "recency"],
            "update_times_recency": ["update_count", "recency", "update_x_recency"],
        }
        features["update_x_recency"] = features["update_count"] * features["recency"]
        stats_by_model: dict[str, dict[str, float]] = {}
        for target_name, y in (("delta_support", y_delta), ("final_support", y_final)):
            for model_name, cols in feature_matrix.items():
                x = np.column_stack([np.asarray(features[col], dtype=float) for col in cols])
                fit = _fit_ols(x, y)
                cv_r2 = _cv_r2(x, y, n_folds=5)
                stats_by_model[model_name] = {"r2": fit["r2"], "cv_r2": cv_r2}
                metric_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": int(sequence_id),
                        "layer": PRIMARY_LAYER,
                        "state_variable": STATE_VARIABLE,
                        "target": target_name,
                        "model_name": model_name,
                        "r2": float(fit["r2"]),
                        "cv_r2": float(cv_r2),
                        "auc_if_binary": float("nan"),
                        "delta_r2_vs_overlap_only": float(fit["r2"] - stats_by_model.get("overlap_only", {}).get("r2", np.nan)),
                        "delta_r2_vs_update_only": float(fit["r2"] - stats_by_model.get("update_only", {}).get("r2", np.nan)),
                        "delta_r2_vs_recency_only": float(fit["r2"] - stats_by_model.get("recency_only", {}).get("r2", np.nan)),
                        "n_units": int(len(part)),
                    }
                )
                for coef_name, coef_value, se, p in zip(["intercept"] + cols, fit["beta"], fit["se"], fit["p"]):
                    coef_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "model_name": model_name,
                            "coefficient_name": coef_name,
                            "coefficient_value": float(coef_value),
                            "standardized_coefficient": float(_standardized_coef(coef_value, x, y, coef_name, cols)),
                            "p_value": float(p),
                            "notes": f"target={target_name}; sequence_id={int(sequence_id)}; ordinary least squares",
                        }
                    )
    metrics = pd.DataFrame(metric_rows, columns=PANEL_B_METRIC_COLUMNS)
    coefs = pd.DataFrame(coef_rows, columns=PANEL_B_COEF_COLUMNS)
    _save_csv(ctx, metrics, ctx.metrics_dir / "supp_legacy_panel_b_update_recency_model_metrics.csv")
    _save_csv(ctx, coefs, ctx.metrics_dir / "supp_legacy_panel_b_update_recency_model_coefficients.csv")
    _save_csv(ctx, coefs, ctx.metrics_dir / "supp_update_recency_model_coefficients.csv")
    ctx.completed_modules["update_recency_model"] = True

def compute_supp_update_recency_support_model(ctx: ExperimentContext, unit_df: pd.DataFrame) -> None:
    metric_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    if unit_df.empty:
        _save_csv(ctx, pd.DataFrame(columns=PANEL_B_METRIC_COLUMNS), ctx.metrics_dir / "supp_update_recency_support_model_metrics.csv")
        _save_csv(ctx, pd.DataFrame(columns=PANEL_B_COEF_COLUMNS), ctx.metrics_dir / "supp_update_recency_support_model_coefficients.csv")
        return
    for sequence_id, part in unit_df.groupby("sequence_id", sort=True):
        y_delta = pd.to_numeric(part["delta_support"], errors="coerce").to_numpy(dtype=float)
        y_final = pd.to_numeric(part["final_support"], errors="coerce").to_numpy(dtype=float)
        features = {
            "baseline_support": np.zeros(len(part), dtype=float),
            "update_count": pd.to_numeric(part["update_count"], errors="coerce").to_numpy(dtype=float),
            "recency": -pd.to_numeric(part["time_since_last_update"], errors="coerce").to_numpy(dtype=float),
            "overlap": (pd.to_numeric(part["update_count"], errors="coerce").to_numpy(dtype=float) > 0).astype(float),
        }
        features["update_x_recency"] = features["update_count"] * features["recency"]
        feature_matrix = {
            "baseline_only": ["baseline_support"],
            "update_only": ["update_count"],
            "recency_only": ["recency"],
            "overlap_only": ["overlap"],
            "update_plus_recency": ["update_count", "recency"],
            "update_times_recency": ["update_count", "recency", "update_x_recency"],
        }
        for target_name, y in (("delta_support", y_delta), ("final_support", y_final)):
            model_fits: dict[str, float] = {}
            for model_name, cols in feature_matrix.items():
                x = np.column_stack([np.asarray(features[col], dtype=float) for col in cols])
                fit = _fit_ols(x, y)
                cv_r2 = _cv_r2(x, y, n_folds=5)
                model_fits[model_name] = float(fit["r2"])
                metric_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": int(sequence_id),
                        "layer": PRIMARY_LAYER,
                        "state_variable": STATE_VARIABLE,
                        "target": target_name,
                        "model_name": model_name,
                        "r2": float(fit["r2"]),
                        "cv_r2": float(cv_r2),
                        "auc_if_binary": float("nan"),
                        "delta_r2_vs_overlap_only": float(fit["r2"] - model_fits.get("overlap_only", np.nan)),
                        "delta_r2_vs_update_only": float(fit["r2"] - model_fits.get("update_only", np.nan)),
                        "delta_r2_vs_recency_only": float(fit["r2"] - model_fits.get("recency_only", np.nan)),
                        "n_units": int(len(part)),
                    }
                )
                for coef_name, coef_value, se, p in zip(["intercept"] + cols, fit["beta"], fit["se"], fit["p"]):
                    coef_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "model_name": model_name,
                            "coefficient_name": coef_name,
                            "coefficient_value": float(coef_value),
                            "standardized_coefficient": float(_standardized_coef(coef_value, x, y, coef_name, cols)),
                            "p_value": float(p),
                            "notes": f"supplement-only; target={target_name}; sequence_id={int(sequence_id)}",
                        }
                    )
    _save_csv(ctx, pd.DataFrame(metric_rows, columns=PANEL_B_METRIC_COLUMNS), ctx.metrics_dir / "supp_update_recency_support_model_metrics.csv")
    _save_csv(ctx, pd.DataFrame(coef_rows, columns=PANEL_B_COEF_COLUMNS), ctx.metrics_dir / "supp_update_recency_support_model_coefficients.csv")
