from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


def compute_condition_metrics(
    df_trials: pd.DataFrame,
    *,
    condition_order: Sequence[str],
    shuffle_condition: str,
    static_condition: str,
) -> pd.DataFrame:
    sub_b = df_trials[df_trials["condition"] == shuffle_condition][
        ["trial_id", "donor_sample_label", "donor_is_distinct"]
    ].copy()
    if len(sub_b) == 0:
        raise ValueError("Shuffle-condition rows are required to build canonical change mapping.")
    sub_b = sub_b.drop_duplicates("trial_id")
    donor_label_map = dict(zip(sub_b["trial_id"].to_numpy(), sub_b["donor_sample_label"].to_numpy()))
    donor_distinct_map = dict(zip(sub_b["trial_id"].to_numpy(), sub_b["donor_is_distinct"].to_numpy()))

    rows = []
    for cond in condition_order:
        sub = df_trials[df_trials["condition"] == cond]
        if len(sub) == 0:
            continue
        trial_ids = sub["trial_id"].to_numpy()
        donor_lbl_canonical = np.array([donor_label_map[int(t)] for t in trial_ids], dtype=np.int64)
        donor_distinct_canonical = np.array([donor_distinct_map[int(t)] for t in trial_ids], dtype=np.int64)
        pred = sub["prediction_probe"].to_numpy(dtype=np.int64)
        rows.append(
            {
                "condition": cond,
                "n_trials": int(len(sub)),
                "acc_probe": 100.0 * float((sub["prediction_probe"] == sub["probe_label"]).mean()),
                "error_rate": 100.0 * float((sub["prediction_probe"] != sub["probe_label"]).mean()),
                "silent_rate": 100.0 * float((sub["prediction_probe"] == -1).mean()),
                "abs_rate_pred_original_sample": 100.0 * float((sub["prediction_probe"] == sub["sample_label"]).mean()),
                "abs_rate_pred_donor_sample": 100.0 * float((sub["prediction_probe"] == sub["donor_sample_label"]).mean()),
                "abs_rate_pred_donor_shifted_memory": 100.0
                * float(((sub["prediction_probe"] == sub["donor_sample_label"]) & (sub["donor_is_distinct"] == 1)).mean()),
                "abs_rate_pred_change_under_bmap": 100.0
                * float(((pred == donor_lbl_canonical) & (donor_distinct_canonical == 1)).mean()),
                "abs_rate_pred_probe": 100.0 * float((sub["prediction_probe"] == sub["probe_label"]).mean()),
                "self_swap_rate": 100.0 * float(sub["is_self_swap"].mean()),
            }
        )

    df_metrics = pd.DataFrame(rows)
    base_row = df_metrics[df_metrics["condition"] == static_condition]
    if len(base_row) != 1:
        raise ValueError("Need exactly one static baseline row in condition metrics.")
    base_sample_rate = float(base_row.iloc[0]["abs_rate_pred_original_sample"])
    df_metrics["ami_abs_vs_static_pp"] = df_metrics["abs_rate_pred_original_sample"] - base_sample_rate
    df_metrics["distance_to_static_sample_rate_pp"] = (
        df_metrics["abs_rate_pred_original_sample"] - base_sample_rate
    ).abs()
    dist_a = float(df_metrics[df_metrics["condition"] == condition_order[0]]["distance_to_static_sample_rate_pp"].iloc[0])
    df_metrics["collapse_gain_pp_vs_A"] = dist_a - df_metrics["distance_to_static_sample_rate_pp"]
    return df_metrics


def compute_bias_components_ux(df_subset: pd.DataFrame, num_classes: int) -> Dict[str, float]:
    n_total = len(df_subset)
    if n_total == 0:
        raise ValueError("Bias computation received empty subset")

    errors = df_subset[df_subset["prediction_probe"] != df_subset["probe_label"]]
    n_error = len(errors)
    if n_error == 0:
        return {
            "n_total": int(n_total),
            "n_error": 0,
            "error_rate": 0.0,
            "bias_original_sample": 0.0,
            "bias_donor_shifted_memory": 0.0,
            "bias_silent": 0.0,
            "bias_other_classes": 0.0,
        }

    pred = errors["prediction_probe"].to_numpy()
    sample_lbl = errors["sample_label"].to_numpy()
    donor_lbl = errors["donor_sample_label"].to_numpy()
    probe_lbl = errors["probe_label"].to_numpy()
    donor_distinct = errors["donor_is_distinct"].to_numpy()

    bias_original_sample = float(np.mean(pred == sample_lbl))
    bias_donor_shift = float(np.mean((pred == donor_lbl) & (donor_distinct == 1)))
    bias_silent = float(np.mean(pred == -1))

    valid = (pred >= 0) & (pred < num_classes)
    noise_hit = valid & (pred != sample_lbl) & (pred != donor_lbl) & (pred != probe_lbl)
    bias_other_classes = float(np.mean(noise_hit))

    return {
        "n_total": int(n_total),
        "n_error": int(n_error),
        "error_rate": 100.0 * float(n_error) / float(n_total),
        "bias_original_sample": bias_original_sample,
        "bias_donor_shifted_memory": bias_donor_shift,
        "bias_silent": bias_silent,
        "bias_other_classes": bias_other_classes,
    }


def compute_bias_table(df_trials: pd.DataFrame, num_classes: int, *, condition_order: Sequence[str]) -> pd.DataFrame:
    rows = []
    for cond in condition_order:
        sub = df_trials[df_trials["condition"] == cond]
        row = compute_bias_components_ux(sub, num_classes=num_classes)
        row["condition"] = cond
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_drop_test(
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    if len(indicator_a) != len(indicator_b):
        raise ValueError("Paired bootstrap input length mismatch.")
    n = len(indicator_a)
    if n == 0:
        raise ValueError("Paired bootstrap received empty input.")

    rng = np.random.default_rng(seed)
    obs_diff = float(indicator_a.mean() - indicator_b.mean())
    boot = np.zeros(n_boot, dtype=np.float64)
    for b_idx in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b_idx] = float(indicator_a[idx].mean() - indicator_b[idx].mean())

    return {
        "obs_diff": obs_diff,
        "ci95_lower": float(np.percentile(boot, 2.5)),
        "ci95_upper": float(np.percentile(boot, 97.5)),
        "p_one_sided_nonpositive": float(np.mean(boot <= 0.0)),
    }


def paired_bootstrap_closeness_to_static_gain(
    indicator_a: np.ndarray,
    indicator_b: np.ndarray,
    indicator_c: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    if not (len(indicator_a) == len(indicator_b) == len(indicator_c)):
        raise ValueError("Paired bootstrap closeness input length mismatch.")
    n = len(indicator_a)
    if n == 0:
        raise ValueError("Paired bootstrap closeness received empty input.")

    rng = np.random.default_rng(seed)
    mean_a = float(indicator_a.mean())
    mean_b = float(indicator_b.mean())
    mean_c = float(indicator_c.mean())
    obs_gain = abs(mean_a - mean_c) - abs(mean_b - mean_c)

    boot = np.zeros(n_boot, dtype=np.float64)
    for b_idx in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a_b = float(indicator_a[idx].mean())
        b_b = float(indicator_b[idx].mean())
        c_b = float(indicator_c[idx].mean())
        boot[b_idx] = abs(a_b - c_b) - abs(b_b - c_b)

    return {
        "obs_gain": obs_gain,
        "ci95_lower": float(np.percentile(boot, 2.5)),
        "ci95_upper": float(np.percentile(boot, 97.5)),
        "p_one_sided_nonpositive": float(np.mean(boot <= 0.0)),
    }


def compute_collapse_summary(
    df_trials: pd.DataFrame,
    metrics_condition: pd.DataFrame,
    metrics_bias: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    dynamic_condition: str,
    shuffle_condition: str,
    static_condition: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cond_tbl = {r["condition"]: r for _, r in metrics_condition.iterrows()}
    bias_tbl = {r["condition"]: r for _, r in metrics_bias.iterrows()}

    row_a = cond_tbl[dynamic_condition]
    row_b = cond_tbl[shuffle_condition]
    row_c = cond_tbl[static_condition]
    bias_a = bias_tbl[dynamic_condition]
    bias_b = bias_tbl[shuffle_condition]

    acc_retention = 100.0 * float(row_b["acc_probe"]) / max(float(row_a["acc_probe"]), 1e-6)
    drop_abs_sample_rate = float(row_a["abs_rate_pred_original_sample"]) - float(row_b["abs_rate_pred_original_sample"])
    drop_ami = float(row_a["ami_abs_vs_static_pp"]) - float(row_b["ami_abs_vs_static_pp"])
    drop_bias_sample_pp = 100.0 * (float(bias_a["bias_original_sample"]) - float(bias_b["bias_original_sample"]))
    gain_bias_donor_shift_pp = 100.0 * (
        float(bias_b["bias_donor_shifted_memory"]) - float(bias_a["bias_donor_shifted_memory"])
    )

    dist_a_static = abs(float(row_a["abs_rate_pred_original_sample"]) - float(row_c["abs_rate_pred_original_sample"]))
    dist_b_static = abs(float(row_b["abs_rate_pred_original_sample"]) - float(row_c["abs_rate_pred_original_sample"]))

    df_a = df_trials[df_trials["condition"] == dynamic_condition].sort_values("trial_id")
    df_b = df_trials[df_trials["condition"] == shuffle_condition].sort_values("trial_id")
    df_c = df_trials[df_trials["condition"] == static_condition].sort_values("trial_id")
    if not (
        np.array_equal(df_a["trial_id"].to_numpy(), df_b["trial_id"].to_numpy())
        and np.array_equal(df_a["trial_id"].to_numpy(), df_c["trial_id"].to_numpy())
    ):
        raise ValueError("trial_id mismatch between A/B/C for paired bootstrap.")

    ind_a_sample = (df_a["prediction_probe"].to_numpy() == df_a["sample_label"].to_numpy()).astype(np.float64)
    ind_b_sample = (df_b["prediction_probe"].to_numpy() == df_b["sample_label"].to_numpy()).astype(np.float64)
    boot_sample_drop = paired_bootstrap_drop_test(ind_a_sample, ind_b_sample, n_boot=n_boot, seed=seed + 11)

    ind_b_donor_shift = (
        (df_b["prediction_probe"].to_numpy() == df_b["donor_sample_label"].to_numpy())
        & (df_b["donor_is_distinct"].to_numpy() == 1)
    ).astype(np.float64)
    ind_a_donor_shift = (
        (df_a["prediction_probe"].to_numpy() == df_a["donor_sample_label"].to_numpy())
        & (df_a["donor_is_distinct"].to_numpy() == 1)
    ).astype(np.float64)
    boot_donor_gain = paired_bootstrap_drop_test(ind_b_donor_shift, ind_a_donor_shift, n_boot=n_boot, seed=seed + 23)
    ind_c_sample = (df_c["prediction_probe"].to_numpy() == df_c["sample_label"].to_numpy()).astype(np.float64)
    boot_closeness_gain = paired_bootstrap_closeness_to_static_gain(
        indicator_a=ind_a_sample,
        indicator_b=ind_b_sample,
        indicator_c=ind_c_sample,
        n_boot=n_boot,
        seed=seed + 37,
    )

    summary = pd.DataFrame(
        [
            {
                "acc_probe_A_dynamic": float(row_a["acc_probe"]),
                "acc_probe_B_shuffle": float(row_b["acc_probe"]),
                "acc_probe_C_static": float(row_c["acc_probe"]),
                "probe_acc_retention_B_over_A_pct": acc_retention,
                "sample_pred_rate_drop_A_minus_B_pp": drop_abs_sample_rate,
                "ami_drop_A_minus_B_pp": drop_ami,
                "error_bias_original_sample_drop_A_minus_B_pp": drop_bias_sample_pp,
                "error_bias_donor_shift_gain_B_minus_A_pp": gain_bias_donor_shift_pp,
                "distance_to_static_sample_rate_A_pp": dist_a_static,
                "distance_to_static_sample_rate_B_pp": dist_b_static,
                "collapse_toward_static_improvement_pp": dist_a_static - dist_b_static,
                "collapse_gain_bootstrap_pp": 100.0 * boot_closeness_gain["obs_gain"],
                "collapse_gain_bootstrap_ci95_lower_pp": 100.0 * boot_closeness_gain["ci95_lower"],
                "collapse_gain_bootstrap_ci95_upper_pp": 100.0 * boot_closeness_gain["ci95_upper"],
                "collapse_gain_bootstrap_p_one_sided_nonpositive": boot_closeness_gain["p_one_sided_nonpositive"],
                "paired_bootstrap_sample_drop_pp": 100.0 * boot_sample_drop["obs_diff"],
                "paired_bootstrap_sample_drop_ci95_lower_pp": 100.0 * boot_sample_drop["ci95_lower"],
                "paired_bootstrap_sample_drop_ci95_upper_pp": 100.0 * boot_sample_drop["ci95_upper"],
                "paired_bootstrap_p_one_sided_nonpositive": boot_sample_drop["p_one_sided_nonpositive"],
                "paired_bootstrap_donor_shift_gain_pp": 100.0 * boot_donor_gain["obs_diff"],
                "paired_bootstrap_donor_shift_gain_ci95_lower_pp": 100.0 * boot_donor_gain["ci95_lower"],
                "paired_bootstrap_donor_shift_gain_ci95_upper_pp": 100.0 * boot_donor_gain["ci95_upper"],
                "paired_bootstrap_p_one_sided_no_donor_gain": boot_donor_gain["p_one_sided_nonpositive"],
            }
        ]
    )

    boot_table = pd.DataFrame(
        [
            {
                "test_name": "A_minus_B_original_sample_prediction_rate",
                "obs_diff_rate": boot_sample_drop["obs_diff"],
                "ci95_lower": boot_sample_drop["ci95_lower"],
                "ci95_upper": boot_sample_drop["ci95_upper"],
                "p_one_sided_nonpositive": boot_sample_drop["p_one_sided_nonpositive"],
            },
            {
                "test_name": "B_minus_A_donor_shifted_memory_rate",
                "obs_diff_rate": boot_donor_gain["obs_diff"],
                "ci95_lower": boot_donor_gain["ci95_lower"],
                "ci95_upper": boot_donor_gain["ci95_upper"],
                "p_one_sided_nonpositive": boot_donor_gain["p_one_sided_nonpositive"],
            },
            {
                "test_name": "collapse_gain_vs_static",
                "obs_diff_rate": boot_closeness_gain["obs_gain"],
                "ci95_lower": boot_closeness_gain["ci95_lower"],
                "ci95_upper": boot_closeness_gain["ci95_upper"],
                "p_one_sided_nonpositive": boot_closeness_gain["p_one_sided_nonpositive"],
            },
        ]
    )
    return summary, boot_table


__all__ = [
    "compute_bias_components_ux",
    "compute_bias_table",
    "compute_collapse_summary",
    "compute_condition_metrics",
    "paired_bootstrap_closeness_to_static_gain",
    "paired_bootstrap_drop_test",
]
