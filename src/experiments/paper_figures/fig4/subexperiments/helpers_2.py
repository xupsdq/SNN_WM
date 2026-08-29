from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig4.constants import CORE_CONDITIONS
from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import _assign_bins
from src.experiments.paper_figures.fig4.types import ExperimentContext, Fig4Config, OverlapReentryDMSBank

def _iso_match_row(match_id: int, bin_name: str, high: pd.Series, low: pd.Series) -> dict[str, Any]:
    sim_diff = abs(float(high["pixel_similarity"]) - float(low["pixel_similarity"]))
    sample_energy_diff = _relative_difference(float(high["input_energy_sample"]), float(low["input_energy_sample"]))
    probe_energy_diff = _relative_difference(float(high["input_energy_probe"]), float(low["input_energy_probe"]))
    return {
        "network_seed": int(high["network_seed"]),
        "match_id": int(match_id),
        "iso_similarity_bin": bin_name,
        "high_pair_id": int(high["pair_id"]),
        "low_pair_id": int(low["pair_id"]),
        "pixel_similarity_high": float(high["pixel_similarity"]),
        "pixel_similarity_low": float(low["pixel_similarity"]),
        "similarity_difference": sim_diff,
        "dice_overlap_high": float(high["dice_overlap"]),
        "dice_overlap_low": float(low["dice_overlap"]),
        "overlap_difference": float(high["dice_overlap"]) - float(low["dice_overlap"]),
        "input_energy_sample_high": float(high["input_energy_sample"]),
        "input_energy_sample_low": float(low["input_energy_sample"]),
        "sample_energy_rel_difference": sample_energy_diff,
        "input_energy_probe_high": float(high["input_energy_probe"]),
        "input_energy_probe_low": float(low["input_energy_probe"]),
        "probe_energy_rel_difference": probe_energy_diff,
        "class_pair_high": str(high["class_pair"]),
        "class_pair_low": str(low["class_pair"]),
        "probe_label_high": int(high["probe_label"]),
        "probe_label_low": int(low["probe_label"]),
        "drop_event_high": int(high["drop_event"]),
        "drop_event_low": int(low["drop_event"]),
        "acc_drop_high": int(high["acc_drop"]),
        "acc_drop_low": int(low["acc_drop"]),
        "paired_delta_drop_event": int(high["drop_event"]) - int(low["drop_event"]),
        "paired_delta_acc_drop": int(high["acc_drop"]) - int(low["acc_drop"]),
    }

def _matched_overlap_permutation_test(matches: pd.DataFrame, cfg: Fig4Config, rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, float]]:
    network_seed = int(matches["network_seed"].iloc[0]) if not matches.empty else 0
    n_perm = max(0, int(cfg.n_match_permutations))
    if matches.empty:
        null = pd.DataFrame(columns=["network_seed", "perm_id", "null_delta_drop_event"])
        return null, {"observed_delta_drop_event": float("nan"), "p_one_sided": float("nan"), "p_two_sided": float("nan")}
    deltas = matches["paired_delta_drop_event"].to_numpy(dtype=float)
    observed = float(np.mean(deltas))
    rows = []
    null_values = []
    for perm_id in range(n_perm):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(deltas))
        null_delta = float(np.mean(signs * deltas))
        null_values.append(null_delta)
        rows.append({"network_seed": network_seed, "perm_id": int(perm_id), "null_delta_drop_event": null_delta})
    null_arr = np.asarray(null_values, dtype=float)
    p_one = float((1 + np.sum(null_arr >= observed)) / (1 + n_perm)) if n_perm else float("nan")
    p_two = float((1 + np.sum(np.abs(null_arr) >= abs(observed))) / (1 + n_perm)) if n_perm else float("nan")
    return pd.DataFrame(rows), {"observed_delta_drop_event": observed, "p_one_sided": p_one, "p_two_sided": p_two}

def _overlap_accuracy_contrast_by_network(matches: pd.DataFrame, network_seed: int, perm_stats: Mapping[str, float]) -> pd.DataFrame:
    columns = [
        "network_seed",
        "n_matched_sets",
        "drop_rate_high_overlap",
        "drop_rate_low_overlap",
        "delta_drop_rate",
        "mean_acc_drop_high_overlap",
        "mean_acc_drop_low_overlap",
        "delta_acc_drop",
        "mean_similarity_difference",
        "max_similarity_difference",
        "mean_overlap_difference",
        "mean_sample_energy_rel_difference",
        "mean_probe_energy_rel_difference",
        "permutation_p_one_sided",
        "permutation_p_two_sided",
    ]
    if matches.empty:
        return pd.DataFrame([{col: (int(network_seed) if col == "network_seed" else 0 if col == "n_matched_sets" else float("nan")) for col in columns}], columns=columns)
    row = {
        "network_seed": int(network_seed),
        "n_matched_sets": int(len(matches)),
        "drop_rate_high_overlap": float(matches["drop_event_high"].mean()),
        "drop_rate_low_overlap": float(matches["drop_event_low"].mean()),
        "delta_drop_rate": float(matches["paired_delta_drop_event"].mean()),
        "mean_acc_drop_high_overlap": float(matches["acc_drop_high"].mean()),
        "mean_acc_drop_low_overlap": float(matches["acc_drop_low"].mean()),
        "delta_acc_drop": float(matches["paired_delta_acc_drop"].mean()),
        "mean_similarity_difference": float(matches["similarity_difference"].mean()),
        "max_similarity_difference": float(matches["similarity_difference"].max()),
        "mean_overlap_difference": float(matches["overlap_difference"].mean()),
        "mean_sample_energy_rel_difference": float(matches["sample_energy_rel_difference"].mean()),
        "mean_probe_energy_rel_difference": float(matches["probe_energy_rel_difference"].mean()),
        "permutation_p_one_sided": float(perm_stats.get("p_one_sided", float("nan"))),
        "permutation_p_two_sided": float(perm_stats.get("p_two_sided", float("nan"))),
    }
    return pd.DataFrame([row], columns=columns)

def _matching_balance_diagnostics(matches: pd.DataFrame, network_seed: int) -> pd.DataFrame:
    columns = [
        "network_seed",
        "n_matched_sets",
        "mean_similarity_difference",
        "median_similarity_difference",
        "p95_similarity_difference",
        "max_similarity_difference",
        "mean_sample_energy_rel_difference",
        "mean_probe_energy_rel_difference",
        "mean_overlap_difference",
        "fraction_probe_label_matched",
        "fraction_class_pair_matched",
    ]
    if matches.empty:
        return pd.DataFrame([{col: (int(network_seed) if col == "network_seed" else 0 if col == "n_matched_sets" else float("nan")) for col in columns}], columns=columns)
    row = {
        "network_seed": int(network_seed),
        "n_matched_sets": int(len(matches)),
        "mean_similarity_difference": float(matches["similarity_difference"].mean()),
        "median_similarity_difference": float(matches["similarity_difference"].median()),
        "p95_similarity_difference": float(matches["similarity_difference"].quantile(0.95)),
        "max_similarity_difference": float(matches["similarity_difference"].max()),
        "mean_sample_energy_rel_difference": float(matches["sample_energy_rel_difference"].mean()),
        "mean_probe_energy_rel_difference": float(matches["probe_energy_rel_difference"].mean()),
        "mean_overlap_difference": float(matches["overlap_difference"].mean()),
        "fraction_probe_label_matched": float((matches["probe_label_high"].astype(int) == matches["probe_label_low"].astype(int)).mean()),
        "fraction_class_pair_matched": float((matches["class_pair_high"].astype(str) == matches["class_pair_low"].astype(str)).mean()),
    }
    return pd.DataFrame([row], columns=columns)

def _compute_overlap_excess_accuracy(df: pd.DataFrame, cfg: Fig4Config) -> pd.DataFrame:
    columns = ["network_seed", "iso_similarity_bin", "overlap_excess_group", "n_pairs", "drop_rate", "mean_acc_drop", "mean_pixel_similarity", "mean_dice_overlap", "mean_overlap_excess"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    use = _assign_bins(df.copy(), "pixel_similarity", "iso_similarity_bin", int(cfg.num_iso_similarity_bins))
    rows: list[dict[str, Any]] = []
    for bin_name, part in use.groupby("iso_similarity_bin", sort=True):
        expected = float(part["dice_overlap"].mean())
        part = part.copy()
        part["overlap_excess"] = part["dice_overlap"] - expected
        median = float(part["overlap_excess"].median())
        for group, mask in (("low_overlap_excess", part["overlap_excess"] <= median), ("high_overlap_excess", part["overlap_excess"] > median)):
            sub = part[mask]
            rows.append(
                {
                    "network_seed": int(part["network_seed"].iloc[0]),
                    "iso_similarity_bin": str(bin_name),
                    "overlap_excess_group": group,
                    "n_pairs": int(len(sub)),
                    "drop_rate": float(sub["drop_event"].mean()) if len(sub) else float("nan"),
                    "mean_acc_drop": float(sub["acc_drop"].mean()) if len(sub) else float("nan"),
                    "mean_pixel_similarity": float(sub["pixel_similarity"].mean()) if len(sub) else float("nan"),
                    "mean_dice_overlap": float(sub["dice_overlap"].mean()) if len(sub) else float("nan"),
                    "mean_overlap_excess": float(sub["overlap_excess"].mean()) if len(sub) else float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _overlap_accuracy_regression(df: pd.DataFrame, network_seed: int) -> pd.DataFrame:
    columns = ["network_seed", "metric", "beta_overlap", "beta_similarity", "beta_input_energy_sample", "beta_input_energy_probe", "r2", "n_pairs", "p_overlap", "notes"]
    use = df[["drop_event", "dice_overlap", "pixel_similarity", "input_energy_sample", "input_energy_probe"]].dropna() if not df.empty else pd.DataFrame()
    if len(use) >= 5:
        x = np.column_stack([np.ones(len(use)), use["dice_overlap"], use["pixel_similarity"], use["input_energy_sample"], use["input_energy_probe"]])
        y = use["drop_event"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        pred = x @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
        notes = "OLS probability model for supplementary drop_event sensitivity; main Fig.4D uses matched contrast."
    else:
        beta = [float("nan")] * 5
        r2 = float("nan")
        notes = "insufficient rows for OLS probability model; main Fig.4D uses matched contrast."
    return pd.DataFrame(
        [
            {
                "network_seed": int(network_seed),
                "metric": "drop_event",
                "beta_overlap": float(beta[1]),
                "beta_similarity": float(beta[2]),
                "beta_input_energy_sample": float(beta[3]),
                "beta_input_energy_probe": float(beta[4]),
                "r2": float(r2),
                "n_pairs": int(len(use)),
                "p_overlap": float("nan"),
                "notes": notes,
            }
        ],
        columns=columns,
    )

def _relative_difference(a: float, b: float) -> float:
    return float(abs(a - b) / max((abs(a) + abs(b)) / 2.0, 1e-12))

def _accuracy_pair_columns() -> list[str]:
    return [
        "network_seed",
        "pair_id",
        "sample_image_id",
        "probe_image_id",
        "sample_label",
        "probe_label",
        "class_pair",
        "similarity_bin",
        "overlap_bin",
        "pixel_similarity",
        "dice_overlap",
        "input_energy_sample",
        "input_energy_probe",
        "correct_dynamic",
        "correct_static",
        "acc_drop",
        "static_correct_eligible",
        "drop_event",
        "dynamic_rescue_event",
    ]

def _iso_match_columns() -> list[str]:
    return [
        "network_seed",
        "match_id",
        "iso_similarity_bin",
        "high_pair_id",
        "low_pair_id",
        "pixel_similarity_high",
        "pixel_similarity_low",
        "similarity_difference",
        "dice_overlap_high",
        "dice_overlap_low",
        "overlap_difference",
        "input_energy_sample_high",
        "input_energy_sample_low",
        "sample_energy_rel_difference",
        "input_energy_probe_high",
        "input_energy_probe_low",
        "probe_energy_rel_difference",
        "class_pair_high",
        "class_pair_low",
        "probe_label_high",
        "probe_label_low",
        "drop_event_high",
        "drop_event_low",
        "acc_drop_high",
        "acc_drop_low",
        "paired_delta_drop_event",
        "paired_delta_acc_drop",
    ]

def _random_mask_controls(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> pd.DataFrame:
    f_path = ctx.metrics_dir / "supp_overlap_preserving_perturbation_metrics.csv"
    f_df = pd.read_csv(f_path) if f_path.exists() else pd.DataFrame()
    rows = []
    random_masks = bank.perturbation_masks[bank.perturbation_masks["mask_name"].eq("random_matched_mask")]
    for _, mask in random_masks.iterrows():
        pair_id = int(mask["pair_id"])
        f_row = f_df[(f_df["pair_id"].eq(pair_id)) & (f_df["condition"].eq("sample_random_matched_dynamic"))].head(1) if not f_df.empty else pd.DataFrame()
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "pair_id": pair_id,
                "condition": "sample_random_matched_dynamic",
                "random_mask_id": f"pair_{pair_id}_random_0",
                "mask_pixel_count": int(mask["pixel_count"]),
                "mask_input_energy": float(mask["input_energy"]),
                "mask_spike_count_estimate": float(mask["spike_count_estimate"]),
                "DPI_L3": _from_row(f_row, "DPI_L3", float("nan")),
                "dynamic_like_recovery": _from_row(f_row, "dynamic_like_recovery", float("nan")),
                "decision_deflection_score": _from_row(f_row, "decision_deflection_score", float("nan")),
            }
        )
    return pd.DataFrame(rows)

def _condition_audit(ctx: ExperimentContext, bank: OverlapReentryDMSBank) -> pd.DataFrame:
    rows = []
    for condition in CORE_CONDITIONS:
        completed = int(bank.condition_metrics[bank.condition_metrics["condition"].eq(condition)]["pair_id"].nunique())
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "n_pairs": int(len(bank.pair_trials)),
                "n_similarity_bins": int(bank.pair_trials["similarity_bin"].nunique()),
                "n_overlap_bins": int(bank.pair_trials["overlap_bin"].nunique()),
                "n_matched_groups": int(bank.pair_trials["matched_group_id"].replace("", pd.NA).dropna().nunique()),
                "n_conditions": int(len(CORE_CONDITIONS)),
                "condition": condition,
                "n_completed": completed,
                "n_failed": max(0, int(len(bank.pair_trials)) - completed),
                "notes": "probe unchanged for all core perturbation assays",
            }
        )
    return pd.DataFrame(rows)

def _from_row(df: pd.DataFrame, column: str, default: float) -> float:
    if df.empty or column not in df.columns:
        return float(default)
    value = pd.to_numeric(df[column], errors="coerce").iloc[0]
    return float(default) if pd.isna(value) else float(value)

def _finite_delta(a: float, b: float) -> float:
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")

__all__ = ('_iso_match_row', '_matched_overlap_permutation_test', '_overlap_accuracy_contrast_by_network', '_matching_balance_diagnostics', '_compute_overlap_excess_accuracy', '_overlap_accuracy_regression', '_relative_difference', '_accuracy_pair_columns', '_iso_match_columns', '_random_mask_controls', '_condition_audit', '_from_row', '_finite_delta')
