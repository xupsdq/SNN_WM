from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from input_function import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.deterministic_discovery import ModelRuntime, ScanConfig
from src.experiments.common.diagnostic_mask_utils import apply_ablation, apply_preserve_only, connected_component_count
from src.experiments.common.voltage_readout import (
    compute_voltage_margin,
    compute_voltage_margin_fixed_competitor,
    run_dms_voltage_inference_batch,
)
from diagnostic_feature_overlap_voltage_pipeline import build_dataset_arrays
from diagnostic_probe_diagnostics import ProbeDiagnosticRecord, build_probe_diagnostics, load_probe_diagnostics
from src.plotting.common.io import PUBLICATION_TWO_COLUMN_FIGSIZE, apply_publication_style, save_figure_all_formats, save_run_config, save_tidy_csv

DEFAULT_SAVE_DIR = "results/diagnostic_probe_causal_surgery"
CONDITION_ORDER: Tuple[str, ...] = (
    "no-sample",
    "full-sample",
    "D-only",
    "N-only",
    "minus-D",
    "minus-N",
    "random-only",
    "minus-random",
)
ALL_CONDITION_ORDER: Tuple[str, ...] = CONDITION_ORDER


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def delay_steps(self) -> int:
        return int(round((self.delay_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


class VoltageDMSInferenceBackend:
    def __init__(
        self,
        *,
        net,
        encoder,
        spec: ExperimentSpec,
        device: torch.device,
        batch_size: int,
        readout_mode: str,
        readout_step: int | None,
        voltage_pooling: str,
        top_m: int,
    ) -> None:
        self.net = net
        self.encoder = encoder
        self.spec = spec
        self.device = device
        self.batch_size = int(batch_size)
        self.readout_mode = str(readout_mode)
        self.readout_step = readout_step
        self.voltage_pooling = str(voltage_pooling)
        self.top_m = int(top_m)

    def infer(
        self,
        *,
        sample_images: Sequence[torch.Tensor],
        probe_images: Sequence[torch.Tensor],
        probe_labels: Sequence[int],
        fixed_competitor_labels: Sequence[int] | None = None,
        wrong0_labels: Sequence[int] | None = None,
    ) -> List[Dict[str, object]]:
        outputs: List[Dict[str, object]] = []
        active_fixed_labels = fixed_competitor_labels if fixed_competitor_labels is not None else wrong0_labels
        for start in range(0, len(sample_images), self.batch_size):
            batch_sample = torch.stack(list(sample_images[start:start + self.batch_size]), dim=0)
            batch_probe = torch.stack(list(probe_images[start:start + self.batch_size]), dim=0)
            batch_labels = [int(label) for label in probe_labels[start:start + self.batch_size]]
            batch_wrong0 = None if active_fixed_labels is None else [int(label) for label in active_fixed_labels[start:start + self.batch_size]]
            result = run_dms_voltage_inference_batch(
                net=self.net,
                encoder=self.encoder,
                sample_images=batch_sample,
                probe_images=batch_probe,
                spec=self.spec,
                delay_steps=self.spec.delay_steps,
                device=self.device,
                readout_mode=self.readout_mode,
                readout_step=self.readout_step,
                pooling=self.voltage_pooling,
                m=self.top_m,
                stsp_mode="dynamic",
                intervention_plan=None,
                return_full_traces=False,
            )
            for idx, (bundle, probe_label) in enumerate(zip(result.bundles, batch_labels)):
                margin = compute_voltage_margin(bundle, true_label=int(probe_label))
                wrong0_label = -1 if batch_wrong0 is None else int(batch_wrong0[idx])
                if wrong0_label >= 0:
                    fixed_margin = compute_voltage_margin_fixed_competitor(
                        bundle,
                        true_label=int(probe_label),
                        competitor_label=int(wrong0_label),
                    )
                    wrong0_score = float(fixed_margin.competitor_score)
                    margin_fixed_wrong0 = float(fixed_margin.margin)
                else:
                    wrong0_score = float("nan")
                    margin_fixed_wrong0 = float("nan")
                prediction = int(bundle.predicted_label)
                outputs.append(
                    {
                        "prediction": prediction,
                        "is_correct": int(prediction == int(probe_label)),
                        "first_fire_t_probe": int(bundle.first_fire_t_probe),
                        "true_label_score": float(margin.true_score),
                        "best_wrong_score": float(margin.best_wrong_score),
                        "best_wrong_label": int(margin.best_wrong_label),
                        "margin": float(margin.margin),
                        "baseline_wrong0_label": int(wrong0_label),
                        "wrong0_label": int(wrong0_label),
                        "wrong0_score": float(wrong0_score),
                        "margin_fixed_wrong0": float(margin_fixed_wrong0),
                        "delta_dir": float("nan"),
                        "same_wrong0_persist": int(wrong0_label >= 0 and prediction == int(wrong0_label)),
                        "other_wrong_drift": int(
                            wrong0_label >= 0 and prediction != int(probe_label) and prediction != int(wrong0_label)
                        ),
                        "backend": str(bundle.backend),
                        "readout_step": int(bundle.readout_step),
                    }
                )
        return outputs


def mix_seed(base_seed: int, *parts: int) -> int:
    value = int(base_seed) & 0xFFFFFFFF
    for idx, part in enumerate(parts, start=1):
        value = (value * 1664525 + 1013904223 + int(part) * (374761393 + idx * 97)) & 0xFFFFFFFF
    return int(value)


def parse_probe_ids(raw_value: str) -> List[int]:
    values: List[int] = []
    for token in str(raw_value).split(","):
        text = token.strip()
        if text:
            values.append(int(text))
    return list(dict.fromkeys(values))


def probe_partition_from_runtime_baseline(is_correct: int) -> str:
    return "correct" if int(is_correct) == 1 else "wrong"


def analysis_family_from_partition(probe_partition: str) -> str:
    return "dn_directional"


def condition_order_for_partition(probe_partition: str) -> Tuple[str, ...]:
    return CONDITION_ORDER


def build_ranked_dn_masks(
    importance_map_signed: np.ndarray,
    foreground_mask: np.ndarray,
    topk_fraction: float,
) -> dict[str, object]:
    importance = np.asarray(importance_map_signed, dtype=np.float64)
    foreground = np.asarray(foreground_mask, dtype=bool)
    if importance.shape != foreground.shape:
        raise ValueError(f"importance_map_signed shape {importance.shape} does not match foreground {foreground.shape}")
    foreground_indices = np.flatnonzero(foreground.reshape(-1))
    foreground_area = int(foreground_indices.size)
    k_pixels = int(np.floor(float(topk_fraction) * float(foreground_area)))
    D_mask = np.zeros_like(foreground, dtype=bool)
    N_mask = np.zeros_like(foreground, dtype=bool)
    if foreground_area <= 0 or k_pixels < 1:
        return {
            "D_mask": D_mask,
            "N_mask": N_mask,
            "D_threshold": float("nan"),
            "N_threshold": float("nan"),
            "k_pixels": int(k_pixels),
            "foreground_area": int(foreground_area),
        }
    foreground_scores = importance.reshape(-1)[foreground_indices]
    order_desc = np.argsort(-foreground_scores, kind="stable")
    order_asc = np.argsort(foreground_scores, kind="stable")
    d_indices = foreground_indices[order_desc[:k_pixels]]
    n_indices = foreground_indices[order_asc[:k_pixels]]
    D_mask.reshape(-1)[d_indices] = True
    N_mask.reshape(-1)[n_indices] = True
    return {
        "D_mask": D_mask,
        "N_mask": N_mask,
        "D_threshold": float(foreground_scores[order_desc[k_pixels - 1]]),
        "N_threshold": float(foreground_scores[order_asc[k_pixels - 1]]),
        "k_pixels": int(k_pixels),
        "foreground_area": int(foreground_area),
    }


def region_spec_for_partition(
    probe_record: ProbeDiagnosticRecord,
    partition: str,
    *,
    D_mask: np.ndarray | None = None,
    N_mask: np.ndarray | None = None,
) -> Dict[str, object]:
    if D_mask is None or N_mask is None:
        raise ValueError("D_mask and N_mask must be provided")
    return {
        "probe_partition": str(partition),
        "analysis_family": "dn_directional",
        "region_a_name": "D",
        "region_b_name": "N",
        "region_a_mask": np.asarray(D_mask, dtype=bool),
        "region_b_mask": np.asarray(N_mask, dtype=bool),
        "condition_order": CONDITION_ORDER,
    }


def build_foreground_random_matched_mask(
    diagnostic_mask: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    exclude_mask: np.ndarray | None = None,
    max_tries: int = 128,
) -> tuple[np.ndarray, str]:
    """Sample a random control mask inside the probe foreground.

    The first argument is the reference region whose area should be matched.
    The old name `diagnostic_mask` is retained for backward compatibility.
    """

    reference = np.asarray(diagnostic_mask, dtype=bool)
    foreground = np.asarray(foreground_mask, dtype=bool)
    exclude = np.zeros_like(foreground, dtype=bool) if exclude_mask is None else np.asarray(exclude_mask, dtype=bool)
    area = int(reference.sum())
    if area <= 0:
        return np.zeros_like(reference, dtype=bool), "empty_reference"
    target_components = connected_component_count(reference)
    candidates = [
        ("foreground_excluding_reference", foreground & ~exclude),
        ("foreground_fallback", foreground),
        ("whole_image_fallback", np.ones_like(foreground, dtype=bool)),
    ]
    chosen_pool_name = "unavailable"
    available = None
    for pool_name, pool_mask in candidates:
        if int(np.asarray(pool_mask, dtype=bool).sum()) >= area:
            chosen_pool_name = pool_name
            available = np.asarray(pool_mask, dtype=bool)
            break
    if available is None:
        return np.zeros_like(reference, dtype=bool), "insufficient_area"
    pool_index = np.flatnonzero(available.reshape(-1))
    best_mask = None
    best_gap = None
    for _ in range(int(max_tries)):
        picked = rng.choice(pool_index, size=area, replace=False)
        candidate = np.zeros(available.size, dtype=bool)
        candidate[picked] = True
        candidate = candidate.reshape(available.shape)
        gap = abs(connected_component_count(candidate) - target_components)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_mask = candidate
            if gap == 0:
                break
    if best_mask is None:
        return np.zeros_like(reference, dtype=bool), "sampling_failed"
    return np.asarray(best_mask, dtype=bool), chosen_pool_name


def score_samples_for_probe(
    *,
    probe_record: ProbeDiagnosticRecord,
    partition: str | None = None,
    D_mask: np.ndarray,
    N_mask: np.ndarray,
    image_matrix_flat: np.ndarray,
    dataset_labels: np.ndarray,
    overlap_penalty: float = 1.0,
    topk_fraction: float,
) -> pd.DataFrame:
    if partition is None:
        partition = "correct" if int(probe_record.baseline_is_correct) == 1 else "wrong"
    region = region_spec_for_partition(probe_record, partition, D_mask=D_mask, N_mask=N_mask)
    region_a_vec = np.asarray(region["region_a_mask"], dtype=np.float32).reshape(-1)
    region_b_vec = np.asarray(region["region_b_mask"], dtype=np.float32).reshape(-1)
    scores = image_matrix_flat.astype(np.float32, copy=False)
    overlap_a = scores @ region_a_vec
    overlap_b = scores @ region_b_vec
    candidate_ids = np.arange(scores.shape[0], dtype=np.int64)
    keep = candidate_ids != int(probe_record.probe_id)
    df = pd.DataFrame(
        {
            "probe_id": int(probe_record.probe_id),
            "probe_label": int(probe_record.probe_label),
            "record_baseline_is_correct": int(probe_record.baseline_is_correct),
            "runtime_baseline_is_correct": int(str(partition) == "correct"),
            "runtime_probe_partition": str(partition),
            "probe_partition": str(partition),
            "partition_mismatch": int(
                int(probe_record.baseline_is_correct) in {0, 1}
                and int(probe_record.baseline_is_correct) != int(str(partition) == "correct")
            ),
            "analysis_family": str(region["analysis_family"]),
            "sample_id": candidate_ids[keep],
            "sample_label": dataset_labels[keep].astype(np.int64, copy=False),
            "region_a_overlap": overlap_a[keep].astype(np.float64, copy=False),
            "region_b_overlap": overlap_b[keep].astype(np.float64, copy=False),
            "O_D": overlap_a[keep].astype(np.float64, copy=False),
            "O_N": overlap_b[keep].astype(np.float64, copy=False),
            "O_support": np.nan,
            "O_harm": np.nan,
            "gamma_penalty": float(overlap_penalty),
            "rescue_score": np.nan,
            "support_fraction": np.nan,
            "harm_fraction": np.nan,
            "wrong_selection_rank": np.nan,
            "topk_fraction": float(topk_fraction),
        }
    )
    df["O_support"] = df["O_D"]
    df["O_harm"] = df["O_N"]
    df["balanced_score"] = np.minimum(df["O_D"], df["O_N"])
    df["overlap_total"] = df["O_D"] + df["O_N"]
    df["total_overlap"] = df["overlap_total"]
    df["selection_score"] = df["O_D"] - float(overlap_penalty) * df["O_N"]
    df["rescue_score"] = df["selection_score"]
    denom = np.maximum(df["overlap_total"].to_numpy(dtype=np.float64), 1e-8)
    df["support_fraction"] = df["O_D"] / denom
    df["harm_fraction"] = df["O_N"] / denom
    df = df.sort_values(
        ["selection_score", "O_D", "O_N", "sample_id"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    df["selected_rank"] = np.arange(1, len(df) + 1, dtype=np.int64)
    df["wrong_selection_rank"] = df["selected_rank"]
    return df


def build_condition_images(
    *,
    sample_image: torch.Tensor,
    random_mask: np.ndarray,
    partition: str | None = None,
    analysis_family: str | None = None,
    region_a_mask: np.ndarray | None = None,
    region_b_mask: np.ndarray | None = None,
    diagnostic_mask: np.ndarray | None = None,
    nondiagnostic_mask: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    harm_mask: np.ndarray | None = None,
) -> Dict[str, torch.Tensor]:
    if region_a_mask is None:
        region_a_mask = support_mask if support_mask is not None else diagnostic_mask
        region_b_mask = harm_mask if harm_mask is not None else nondiagnostic_mask
    if region_a_mask is None or region_b_mask is None:
        raise ValueError("Both region_a_mask and region_b_mask must be provided")
    region_a = np.asarray(region_a_mask, dtype=bool)
    region_b = np.asarray(region_b_mask, dtype=bool)
    return {
        "full-sample": sample_image.detach().cpu().clone(),
        "D-only": apply_preserve_only(sample_image, region_a, fill_value=0.0).detach().cpu(),
        "N-only": apply_preserve_only(sample_image, region_b, fill_value=0.0).detach().cpu(),
        "minus-D": apply_ablation(sample_image, region_a, fill_value=0.0).detach().cpu(),
        "minus-N": apply_ablation(sample_image, region_b, fill_value=0.0).detach().cpu(),
        "random-only": apply_preserve_only(sample_image, random_mask, fill_value=0.0).detach().cpu(),
        "minus-random": apply_ablation(sample_image, random_mask, fill_value=0.0).detach().cpu(),
    }


def _multiple_regression(y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack([np.ones_like(y), x1, x2]).astype(np.float64, copy=False)
    coef, _, _, _ = np.linalg.lstsq(design, y.astype(np.float64, copy=False), rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def _multiple_regression_many(y: np.ndarray, predictors: Mapping[str, np.ndarray]) -> dict[str, float]:
    keys = list(predictors)
    design = [np.ones_like(y, dtype=np.float64)]
    for key in keys:
        design.append(np.asarray(predictors[key], dtype=np.float64))
    coef, _, _, _ = np.linalg.lstsq(
        np.column_stack(design).astype(np.float64, copy=False),
        np.asarray(y, dtype=np.float64),
        rcond=None,
    )
    out: dict[str, float] = {"intercept": float(coef[0])}
    for idx, key in enumerate(keys, start=1):
        out[f"beta_{key}"] = float(coef[idx])
    return out


def summarize_condition_metrics(df: pd.DataFrame, *, partition: str) -> pd.DataFrame:
    subset_all = df[df["runtime_probe_partition"] == str(partition)].copy()
    if subset_all.empty:
        return pd.DataFrame()
    baseline_acc = float(subset_all[subset_all["condition_name"] == "no-sample"]["is_correct"].mean())
    rows: List[Dict[str, object]] = []
    for condition in condition_order_for_partition(partition):
        subset = subset_all[subset_all["condition_name"] == condition].copy()
        if subset.empty:
            continue
        row = {
            "probe_partition": str(partition),
            "runtime_probe_partition": str(partition),
            "analysis_family": analysis_family_from_partition(partition),
            "condition_name": str(condition),
            "n_rows": int(len(subset)),
            "accuracy": float(subset["is_correct"].mean()),
            "accuracy_delta_vs_no_sample": float(subset["is_correct"].mean() - baseline_acc),
            "mean_margin": float(subset["margin_for_delta"].mean()) if "margin_for_delta" in subset.columns else float(subset["margin"].mean()),
            "mean_delta_dir": float(subset["delta_dir"].mean()),
            "mean_abs_delta_dir": float(subset["delta_dir"].abs().mean()),
            "mean_prediction_flip_vs_no_sample": float(subset["prediction_flip_vs_no_sample"].mean()),
        }
        row["effect_vs_no_sample"] = row["mean_delta_dir"]
        if str(partition) == "wrong":
            row["rescue_rate"] = float(subset["rescued_to_true"].mean())
            row["same_wrong0_persist_rate"] = float(subset["same_wrong0_persist"].mean())
            row["other_wrong_drift_rate"] = float(subset["other_wrong_drift"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_mean_for_condition(df: pd.DataFrame, condition_name: str, value_col: str) -> float:
    subset = df[df["condition_name"] == str(condition_name)]
    if subset.empty:
        return float("nan")
    return float(subset[value_col].mean())


def summarize_probe_metrics(df: pd.DataFrame, *, partition: str) -> pd.DataFrame:
    subset_all = df[df["runtime_probe_partition"] == str(partition)].copy()
    rows: List[Dict[str, object]] = []
    for probe_id, subset in subset_all.groupby("probe_id", sort=True):
        row: Dict[str, object] = {
            "probe_id": int(probe_id),
            "probe_label": int(subset["probe_label"].iloc[0]),
            "record_baseline_is_correct": int(subset["record_baseline_is_correct"].iloc[0]),
            "runtime_baseline_is_correct": int(subset["runtime_baseline_is_correct"].iloc[0]),
            "probe_partition": str(partition),
            "runtime_probe_partition": str(partition),
            "analysis_family": analysis_family_from_partition(partition),
            "n_pairs": int(subset["sample_id"].nunique()),
            "partition_mismatch": int(subset["partition_mismatch"].max()),
        }
        row["delta_dir_full"] = _safe_mean_for_condition(subset, "full-sample", "delta_dir")
        row["delta_dir_D_only"] = _safe_mean_for_condition(subset, "D-only", "delta_dir")
        row["delta_dir_N_only"] = _safe_mean_for_condition(subset, "N-only", "delta_dir")
        row["delta_dir_minus_D"] = _safe_mean_for_condition(subset, "minus-D", "delta_dir")
        row["delta_dir_minus_N"] = _safe_mean_for_condition(subset, "minus-N", "delta_dir")
        row["sufficiency_contrast"] = float(row["delta_dir_D_only"] - row["delta_dir_N_only"])
        row["necessity_contrast"] = float(row["delta_dir_minus_N"] - row["delta_dir_minus_D"])
        row["causal_selectivity"] = float(np.nanmean([row["sufficiency_contrast"], row["necessity_contrast"]]))
        row["suff_success"] = int(row["sufficiency_contrast"] > 0.0)
        row["nec_success"] = int(row["necessity_contrast"] > 0.0)
        row["joint_success"] = int((row["suff_success"] == 1) and (row["nec_success"] == 1))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_causal_selectivity(df_probe: pd.DataFrame, *, partition: str) -> pd.DataFrame:
    if df_probe.empty or "runtime_probe_partition" not in df_probe.columns:
        return pd.DataFrame()
    subset = df_probe[df_probe["runtime_probe_partition"] == str(partition)].copy()
    if subset.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "probe_partition": str(partition),
                "runtime_probe_partition": str(partition),
                "analysis_family": "dn_directional",
                "n_probes": int(len(subset)),
                "suff_success_rate": float(subset["suff_success"].mean()),
                "nec_success_rate": float(subset["nec_success"].mean()),
                "joint_success_rate": float(subset["joint_success"].mean()),
                "mean_sufficiency_contrast": float(subset["sufficiency_contrast"].mean()),
                "mean_necessity_contrast": float(subset["necessity_contrast"].mean()),
            }
        ]
    )


def summarize_regression(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    full_df = df[df["condition_name"] == "full-sample"].copy()
    for partition in ("correct", "wrong"):
        subset = full_df[full_df["runtime_probe_partition"] == partition].copy()
        if subset.empty:
            continue
        predictors = {
            "O_D": subset["O_D"].to_numpy(dtype=np.float64),
            "O_N": subset["O_N"].to_numpy(dtype=np.float64),
            "selection_score": subset["selection_score"].to_numpy(dtype=np.float64),
        }
        for target_name, y in (
            ("signed_delta_dir", subset["delta_dir"].to_numpy(dtype=np.float64)),
            ("abs_delta_dir", subset["delta_dir"].abs().to_numpy(dtype=np.float64)),
        ):
            for predictor, values in predictors.items():
                rows.append(
                    {
                        "probe_partition": partition,
                        "runtime_probe_partition": partition,
                        "analysis": target_name,
                        "predictor": predictor,
                        "correlation": float(np.corrcoef(values, y)[0, 1]) if len(y) > 1 else float("nan"),
                        "slope": float(np.polyfit(values, y, 1)[0]) if len(y) > 1 else float("nan"),
                        "target": target_name,
                        "model_kind": "univariate",
                    }
                )
            multi = _multiple_regression_many(y, predictors)
            rows.append(
                {
                    "probe_partition": partition,
                    "runtime_probe_partition": partition,
                    "analysis": target_name,
                    "predictor": "O_D+O_N+selection_score",
                    "correlation": float("nan"),
                    "slope": float("nan"),
                    "target": target_name,
                    "model_kind": "multivariate",
                    **multi,
                }
            )
    return pd.DataFrame(rows)


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    ordered = list(dict.fromkeys(list(columns) + list(out.columns)))
    return out.loc[:, ordered]


def select_exemplar_pair(df_pairs: pd.DataFrame, df_probe: pd.DataFrame, *, partition: str) -> tuple[int, int] | None:
    subset_pairs = df_pairs[df_pairs["runtime_probe_partition"] == str(partition)].copy()
    if subset_pairs.empty:
        return None
    winning_probes = set(
        df_probe[
            (df_probe["runtime_probe_partition"] == str(partition))
            & (df_probe["joint_success"] == 1)
        ]["probe_id"].astype(np.int64).tolist()
    )
    pool = subset_pairs[subset_pairs["probe_id"].isin(winning_probes)].copy()
    if pool.empty:
        pool = subset_pairs
    pool = pool.sort_values(["selection_score", "probe_id", "sample_id"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
    chosen = pool.iloc[0]
    return int(chosen["probe_id"]), int(chosen["sample_id"])


def make_probe_overview_figure(records: Sequence[ProbeDiagnosticRecord], *, partition: str) -> plt.Figure:
    apply_publication_style()
    subset = list(records)
    take = min(4, len(subset))
    if take <= 0:
        fig, ax = plt.subplots(figsize=(6.0, 2.5))
        ax.text(0.5, 0.5, f"No {partition} probes", ha="center", va="center")
        ax.axis("off")
        return fig
    fig, axes = plt.subplots(take, 3, figsize=(10.0, 2.8 * max(take, 1)))
    axes_arr = np.atleast_2d(axes)
    for row_idx, record in enumerate(subset[:take]):
        image = record.image[0].numpy()
        overlay = np.zeros((*record.foreground_mask.shape, 3), dtype=np.float32)
        d_mask = np.asarray(record.metadata.get("D_mask", record.diagnostic_mask), dtype=bool)
        n_mask = np.asarray(record.metadata.get("N_mask", record.nondiagnostic_mask), dtype=bool)
        overlay[..., 1] = d_mask.astype(np.float32)
        overlay[..., 0] = n_mask.astype(np.float32)
        overlay_title = "D / N overlay"
        axes_arr[row_idx, 0].imshow(image, cmap="gray")
        axes_arr[row_idx, 0].set_title(f"Probe {record.probe_id} ({record.probe_label})")
        axes_arr[row_idx, 1].imshow(record.importance_map_signed, cmap="coolwarm")
        axes_arr[row_idx, 1].set_title("Signed direction map")
        axes_arr[row_idx, 2].imshow(image, cmap="gray")
        axes_arr[row_idx, 2].imshow(overlay, alpha=0.45)
        axes_arr[row_idx, 2].set_title(overlay_title)
        for col_idx in range(3):
            axes_arr[row_idx, col_idx].axis("off")
    fig.tight_layout()
    return fig


def make_overlap_scatter_figure(df_scores: pd.DataFrame, probe_id: int, selected_sample_ids: Sequence[int], *, partition: str) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    subset = df_scores[(df_scores["probe_id"] == int(probe_id)) & (df_scores["runtime_probe_partition"] == str(partition))].copy()
    x_name, y_name = "O_N", "O_D"
    title = f"Probe {probe_id} D/N overlap ranking ({partition})"
    ax.scatter(subset[x_name], subset[y_name], s=22, alpha=0.45, color="#777777")
    chosen = subset[subset["sample_id"].isin([int(v) for v in selected_sample_ids])].copy()
    ax.scatter(chosen[x_name], chosen[y_name], s=52, alpha=0.9, color="#D55E00")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def make_exemplar_surgery_figure(
    *,
    pair_rows: pd.DataFrame,
    probe_record: ProbeDiagnosticRecord,
    sample_image_lookup: Mapping[str, torch.Tensor],
    partition: str,
) -> plt.Figure:
    apply_publication_style()
    condition_order = condition_order_for_partition(partition)
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.5))
    axes_arr = np.asarray(axes)
    for ax, condition in zip(axes_arr.reshape(-1), condition_order):
        row = pair_rows[pair_rows["condition_name"] == condition].iloc[0]
        combined = torch.maximum(probe_record.image[0], sample_image_lookup[condition][0]).numpy()
        margin_value = float(row["delta_dir"])
        ax.imshow(combined, cmap="gray")
        ax.set_title(f"{condition}\npred={int(row['prediction'])} delta={margin_value:.3f} acc={int(row['is_correct'])}")
        ax.axis("off")
    fig.tight_layout()
    return fig


def make_population_pair_figure(df: pd.DataFrame, *, partition: str) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    subset = df[df["runtime_probe_partition"] == str(partition)].copy()
    left_a = subset[subset["condition_name"] == "N-only"][["probe_id", "sample_id", "delta_dir"]].rename(columns={"delta_dir": "x"})
    left_b = subset[subset["condition_name"] == "D-only"][["probe_id", "sample_id", "delta_dir"]].rename(columns={"delta_dir": "y"})
    right_a = subset[subset["condition_name"] == "minus-D"][["probe_id", "sample_id", "delta_dir"]].rename(columns={"delta_dir": "x"})
    right_b = subset[subset["condition_name"] == "minus-N"][["probe_id", "sample_id", "delta_dir"]].rename(columns={"delta_dir": "y"})
    left_xlabel, left_ylabel = "delta_dir(N-only)", "delta_dir(D-only)"
    right_xlabel, right_ylabel = "delta_dir(minus-D)", "delta_dir(minus-N)"
    left_title, right_title = "D/N sufficiency", "D/N necessity"

    left = left_a.merge(left_b, on=["probe_id", "sample_id"], how="inner")
    axes[0].scatter(left["x"], left["y"], s=24, alpha=0.7)
    left_lim = max(1e-6, float(np.nanmax(np.abs(left[["x", "y"]].to_numpy(dtype=np.float64))))) if not left.empty else 1.0
    axes[0].plot([-left_lim, left_lim], [-left_lim, left_lim], linestyle="--", color="black", linewidth=1.0)
    axes[0].set_xlabel(left_xlabel)
    axes[0].set_ylabel(left_ylabel)
    axes[0].set_title(left_title)
    axes[0].grid(alpha=0.2)

    right = right_a.merge(right_b, on=["probe_id", "sample_id"], how="inner")
    axes[1].scatter(right["x"], right["y"], s=24, alpha=0.7)
    right_lim = max(1e-6, float(np.nanmax(np.abs(right[["x", "y"]].to_numpy(dtype=np.float64))))) if not right.empty else 1.0
    axes[1].plot([-right_lim, right_lim], [-right_lim, right_lim], linestyle="--", color="black", linewidth=1.0)
    axes[1].set_xlabel(right_xlabel)
    axes[1].set_ylabel(right_ylabel)
    axes[1].set_title(right_title)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    return fig


def make_accuracy_figure(df_condition: pd.DataFrame, *, partition: str) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    subset = df_condition[df_condition["runtime_probe_partition"] == str(partition)].copy()
    order = list(condition_order_for_partition(partition))
    subset = subset.set_index("condition_name").reindex(order).dropna(how="all").reset_index()
    ax.bar(subset["condition_name"], subset["accuracy"], color="#4C72B0", alpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy across D/N conditions ({partition})")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def make_selectivity_figure(df_probe: pd.DataFrame, *, partition: str) -> plt.Figure:
    apply_publication_style()
    subset = df_probe[df_probe["runtime_probe_partition"] == str(partition)].copy()
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    x_col, y_col = "sufficiency_contrast", "necessity_contrast"
    success = [
        float(subset["suff_success"].mean()) if not subset.empty else 0.0,
        float(subset["nec_success"].mean()) if not subset.empty else 0.0,
        float(subset["joint_success"].mean()) if not subset.empty else 0.0,
    ]
    title = "D/N causal selectivity"
    axes[0].scatter(subset[x_col], subset[y_col], s=34, alpha=0.8)
    axes[0].axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    axes[0].axvline(0.0, linestyle="--", color="black", linewidth=1.0)
    axes[0].set_xlabel(x_col)
    axes[0].set_ylabel(y_col)
    axes[0].set_title(title)
    axes[0].grid(alpha=0.2)
    axes[1].bar(["suff", "necess", "joint"], success, color=["#55A868", "#C44E52", "#8172B2"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Success rate")
    axes[1].set_title(f"Probe-wise success ({partition})")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def make_wrong_rescue_score_vs_effect_figure(df: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    subset = df[(df["runtime_probe_partition"] == "wrong") & (df["condition_name"] == "full-sample")].copy()
    ax.scatter(subset["rescue_score"], subset["effect_vs_no_sample_fixed_wrong0"], s=28, alpha=0.75)
    if len(subset) > 1:
        coef = np.polyfit(
            subset["rescue_score"].to_numpy(dtype=np.float64),
            subset["effect_vs_no_sample_fixed_wrong0"].to_numpy(dtype=np.float64),
            1,
        )
        x_grid = np.linspace(
            float(subset["rescue_score"].min()),
            float(subset["rescue_score"].max()),
            num=100,
        )
        ax.plot(x_grid, coef[0] * x_grid + coef[1], color="#C44E52", linewidth=1.5)
    ax.set_xlabel("rescue_score")
    ax.set_ylabel("effect_vs_no_sample_fixed_wrong0")
    ax.set_title("Wrong-set rescue score vs fixed-margin effect")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def make_wrong_outcome_breakdown_figure(df_condition: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    subset = df_condition[df_condition["runtime_probe_partition"] == "wrong"].copy()
    order = list(CONDITION_ORDER)
    subset = subset.set_index("condition_name").reindex(order).dropna(how="all").reset_index()
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(subset), dtype=np.float64)
    width = 0.26
    ax.bar(x - width, subset["rescue_rate"], width=width, label="rescue_rate", color="#4C72B0")
    ax.bar(x, subset["same_wrong0_persist_rate"], width=width, label="same_wrong0_persist_rate", color="#55A868")
    ax.bar(x + width, subset["other_wrong_drift_rate"], width=width, label="other_wrong_drift_rate", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(subset["condition_name"], rotation=25)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Wrong-set outcome breakdown by condition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def _should_include_partition(partition: str, probe_partition_mode: str) -> tuple[bool, str]:
    if str(probe_partition_mode) == "correct_only" and str(partition) != "correct":
        return False, "filtered_by_probe_partition_mode"
    if str(probe_partition_mode) == "wrong_only" and str(partition) != "wrong":
        return False, "filtered_by_probe_partition_mode"
    return True, ""


def _condition_order_key_map() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(ALL_CONDITION_ORDER)}


def run_causal_surgery_assay(
    *,
    probe_records: Sequence[ProbeDiagnosticRecord],
    provider_inventory: pd.DataFrame,
    dataset,
    dataset_labels: np.ndarray,
    image_matrix_flat: np.ndarray,
    output_dir: str | Path,
    top_k_samples_per_probe: int,
    random_seed: int,
    inference_backend,
    dn_topk_fraction: float = 0.15,
    dn_overlap_penalty: float = 1.0,
    probe_partition_mode: str = "mixed",
    summary_split_by_baseline: bool = True,
    wrong_mask_scheme: str = "support_harm",
    wrong_harm_penalty: float = 1.0,
    wrong_use_fixed_competitor: bool = True,
    make_plots: bool = True,
) -> Dict[str, object]:
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    probe_map = {int(record.probe_id): record for record in probe_records}
    candidate_probe_ids = sorted(probe_map)

    baseline_probe_rows: List[Dict[str, object]] = []
    wrong_fixed_audit_rows: List[Dict[str, object]] = []
    direction_score_inventory_rows: List[Dict[str, object]] = []
    dn_parameter_rows: List[Dict[str, object]] = []
    dn_spec_by_probe_id: Dict[int, Dict[str, object]] = {}
    if candidate_probe_ids:
        zero_samples = [torch.zeros_like(probe_map[probe_id].image) for probe_id in candidate_probe_ids]
        probe_images = [probe_map[probe_id].image for probe_id in candidate_probe_ids]
        probe_labels = [probe_map[probe_id].probe_label for probe_id in candidate_probe_ids]
        fixed_labels = [
            int(probe_map[probe_id].baseline_wrong0_label) if int(probe_map[probe_id].baseline_wrong0_label) >= 0 else -1
            for probe_id in candidate_probe_ids
        ]
        baseline_outputs = inference_backend.infer(
            sample_images=zero_samples,
            probe_images=probe_images,
            probe_labels=probe_labels,
            fixed_competitor_labels=fixed_labels,
        )
        for probe_id, output in zip(candidate_probe_ids, baseline_outputs):
            runtime_baseline_is_correct = int(int(output["prediction"]) == int(probe_map[probe_id].probe_label))
            baseline_probe_rows.append(
                {
                    "probe_id": int(probe_id),
                    "probe_label": int(probe_map[probe_id].probe_label),
                    "baseline_status": "ok",
                    "runtime_baseline_is_correct": int(runtime_baseline_is_correct),
                    **output,
                }
            )
    baseline_probe_df = pd.DataFrame(baseline_probe_rows)
    baseline_map = baseline_probe_df.set_index("probe_id").to_dict("index") if not baseline_probe_df.empty else {}

    included_records: List[ProbeDiagnosticRecord] = []
    runtime_partition_by_probe_id: Dict[int, str] = {}
    inclusion_rows: List[Dict[str, object]] = []
    for row in provider_inventory.to_dict("records"):
        probe_id = int(row["probe_id"])
        record = probe_map.get(probe_id)
        has_record = record is not None
        has_baseline = probe_id in baseline_map
        include = False
        reason = str(row.get("skip_reason", ""))
        runtime_baseline_is_correct = int(baseline_map[probe_id]["runtime_baseline_is_correct"]) if has_baseline else -1
        partition = probe_partition_from_runtime_baseline(runtime_baseline_is_correct) if has_baseline else ""
        partition_mismatch = (
            int(int(record.baseline_is_correct) != int(runtime_baseline_is_correct))
            if has_record and has_baseline and int(record.baseline_is_correct) in {0, 1}
            else 0
        )
        if has_record and has_baseline:
            direction_score_inventory_rows.append(
                {
                    "probe_id": int(probe_id),
                    "probe_label": int(record.probe_label),
                    "baseline_is_correct": int(record.baseline_is_correct),
                    "baseline_wrong0_label": int(record.baseline_wrong0_label),
                    "importance_min": float(np.nanmin(record.importance_map_signed)),
                    "importance_max": float(np.nanmax(record.importance_map_signed)),
                    "importance_mean": float(np.nanmean(record.importance_map_signed)),
                    "importance_std": float(np.nanstd(record.importance_map_signed)),
                }
            )
            dn_spec = build_ranked_dn_masks(record.importance_map_signed, record.foreground_mask, float(dn_topk_fraction))
            dn_spec_by_probe_id[int(probe_id)] = dn_spec
            include, reason = _should_include_partition(str(partition), str(probe_partition_mode))
            if include:
                if int(record.foreground_mask.sum()) <= 0:
                    include, reason = False, "empty_foreground_mask"
                elif record.importance_map_signed is None or int(np.asarray(record.importance_map_signed).size) <= 0:
                    include, reason = False, "missing_importance_map_signed"
                elif int(dn_spec["k_pixels"]) < 1:
                    include, reason = False, "dn_topk_k_lt_1"
                elif str(partition) == "wrong" and bool(wrong_use_fixed_competitor) and int(record.baseline_wrong0_label) < 0:
                    include, reason = False, "missing_baseline_wrong0_label"
                if include:
                    record.metadata["D_mask"] = np.asarray(dn_spec["D_mask"], dtype=bool)
                    record.metadata["N_mask"] = np.asarray(dn_spec["N_mask"], dtype=bool)
                    included_records.append(record)
                    runtime_partition_by_probe_id[int(probe_id)] = str(partition)
                    dn_parameter_rows.append(
                        {
                            "probe_id": int(probe_id),
                            "probe_label": int(record.probe_label),
                            "runtime_probe_partition": str(partition),
                            "topk_fraction": float(dn_topk_fraction),
                            "foreground_area": int(dn_spec["foreground_area"]),
                            "D_area": int(np.asarray(dn_spec["D_mask"], dtype=bool).sum()),
                            "N_area": int(np.asarray(dn_spec["N_mask"], dtype=bool).sum()),
                            "D_threshold": float(dn_spec["D_threshold"]),
                            "N_threshold": float(dn_spec["N_threshold"]),
                            "k_pixels": int(dn_spec["k_pixels"]),
                        }
                    )
            if str(partition) == "wrong":
                wrong_fixed_audit_rows.append(
                    {
                        "probe_id": int(probe_id),
                        "probe_label": int(record.probe_label),
                        "wrong0_label": int(record.baseline_wrong0_label),
                        "baseline_prediction": int(baseline_map[probe_id]["prediction"]),
                        "baseline_true_score": float(baseline_map[probe_id]["true_label_score"]),
                        "baseline_wrong0_score": float(record.baseline_wrong0_score),
                        "baseline_margin_fixed_wrong0": float(record.baseline_margin_fixed_wrong0),
                        "loaded_wrong0_label": int(getattr(record, "baseline_wrong0_label", -1)),
                        "wrong0_mismatch_flag": 0,
                    }
                )
        inclusion_rows.append(
            {
                **row,
                "is_loaded_record": int(has_record),
                "has_baseline_output": int(has_baseline),
                "record_baseline_is_correct": int(record.baseline_is_correct) if has_record else -1,
                "runtime_baseline_is_correct": int(runtime_baseline_is_correct),
                "runtime_probe_partition": str(partition),
                "partition_mismatch": int(partition_mismatch),
                "topk_fraction": float(dn_topk_fraction),
                "is_included": int(include),
                "inclusion_reason": "" if include else str(reason or "missing_record_or_baseline"),
            }
        )

    overlap_rows: List[pd.DataFrame] = []
    selected_pair_rows: List[Dict[str, object]] = []
    pair_condition_rows: List[Dict[str, object]] = []
    condition_sample_lookup: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    pending_samples: List[torch.Tensor] = []
    pending_probes: List[torch.Tensor] = []
    pending_labels: List[int] = []
    pending_wrong0_labels: List[int] = []
    pending_meta: List[Dict[str, object]] = []

    for record in included_records:
        partition = runtime_partition_by_probe_id[int(record.probe_id)]
        runtime_baseline_is_correct = int(partition == "correct")
        partition_mismatch = (
            int(int(record.baseline_is_correct) != int(runtime_baseline_is_correct))
            if int(record.baseline_is_correct) in {0, 1}
            else 0
        )
        dn_spec = dn_spec_by_probe_id[int(record.probe_id)]
        D_mask = np.asarray(dn_spec["D_mask"], dtype=bool)
        N_mask = np.asarray(dn_spec["N_mask"], dtype=bool)
        region = region_spec_for_partition(record, partition, D_mask=D_mask, N_mask=N_mask)
        scored = score_samples_for_probe(
            probe_record=record,
            partition=partition,
            D_mask=D_mask,
            N_mask=N_mask,
            image_matrix_flat=image_matrix_flat,
            dataset_labels=dataset_labels,
            overlap_penalty=float(dn_overlap_penalty),
            topk_fraction=float(dn_topk_fraction),
        )
        overlap_rows.append(scored)
        chosen = scored.head(int(top_k_samples_per_probe)).copy()
        baseline_output = baseline_map[int(record.probe_id)]
        for chosen_row in chosen.itertuples(index=False):
            sample_image = dataset[int(chosen_row.sample_id)][0].detach().cpu().to(torch.float32)
            pair_seed = mix_seed(int(random_seed), int(record.probe_id), int(chosen_row.sample_id))
            random_mask, random_source = build_foreground_random_matched_mask(
                D_mask,
                record.foreground_mask,
                rng=np.random.default_rng(pair_seed),
                exclude_mask=D_mask,
            )
            condition_images = build_condition_images(
                sample_image=sample_image,
                partition=partition,
                region_a_mask=D_mask,
                region_b_mask=N_mask,
                random_mask=random_mask,
            )
            condition_sample_lookup[(int(record.probe_id), int(chosen_row.sample_id))] = {
                "no-sample": torch.zeros_like(sample_image),
                **condition_images,
            }
            selected_pair_rows.append(
                {
                    "probe_id": int(record.probe_id),
                    "probe_label": int(record.probe_label),
                    "sample_id": int(chosen_row.sample_id),
                    "sample_label": int(chosen_row.sample_label),
                    "record_baseline_is_correct": int(record.baseline_is_correct),
                    "runtime_baseline_is_correct": int(runtime_baseline_is_correct),
                    "runtime_probe_partition": str(partition),
                    "probe_partition": str(partition),
                    "partition_mismatch": int(partition_mismatch),
                    "analysis_family": str(region["analysis_family"]),
                    "region_a_name": str(region["region_a_name"]),
                    "region_b_name": str(region["region_b_name"]),
                    "region_a_overlap": float(chosen_row.region_a_overlap),
                    "region_b_overlap": float(chosen_row.region_b_overlap),
                    "O_D": float(chosen_row.O_D),
                    "O_N": float(chosen_row.O_N),
                    "O_support": float(chosen_row.O_support) if pd.notna(chosen_row.O_support) else float("nan"),
                    "O_harm": float(chosen_row.O_harm) if pd.notna(chosen_row.O_harm) else float("nan"),
                    "balanced_score": float(chosen_row.balanced_score),
                    "gamma_penalty": float(dn_overlap_penalty),
                    "selection_score": float(chosen_row.selection_score),
                    "rescue_score": float(chosen_row.rescue_score) if pd.notna(chosen_row.rescue_score) else float("nan"),
                    "support_fraction": float(chosen_row.support_fraction) if pd.notna(chosen_row.support_fraction) else float("nan"),
                    "harm_fraction": float(chosen_row.harm_fraction) if pd.notna(chosen_row.harm_fraction) else float("nan"),
                    "overlap_total": float(chosen_row.overlap_total),
                    "topk_fraction": float(dn_topk_fraction),
                    "selected_rank": int(chosen_row.selected_rank),
                    "wrong_selection_rank": int(chosen_row.wrong_selection_rank) if pd.notna(chosen_row.wrong_selection_rank) else -1,
                    "D_area": int(D_mask.sum()),
                    "N_area": int(N_mask.sum()),
                    "D_threshold": float(dn_spec["D_threshold"]),
                    "N_threshold": float(dn_spec["N_threshold"]),
                    "k_pixels": int(dn_spec["k_pixels"]),
                    "positive_area": int(record.positive_mask.sum()),
                    "negative_area": int(record.negative_mask.sum()),
                    "random_area": int(random_mask.sum()),
                    "random_seed": int(pair_seed),
                    "random_mask_source": str(random_source),
                }
            )
            base_row = {
                "probe_id": int(record.probe_id),
                "probe_label": int(record.probe_label),
                "sample_id": int(chosen_row.sample_id),
                "sample_label": int(chosen_row.sample_label),
                "record_baseline_is_correct": int(record.baseline_is_correct),
                "runtime_baseline_is_correct": int(runtime_baseline_is_correct),
                "runtime_probe_partition": str(partition),
                "probe_partition": str(partition),
                "partition_mismatch": int(partition_mismatch),
                "analysis_family": str(region["analysis_family"]),
                "condition_name": "no-sample",
                "topk_fraction": float(dn_topk_fraction),
                "region_a_name": str(region["region_a_name"]),
                "region_b_name": str(region["region_b_name"]),
                "region_a_overlap": float(chosen_row.region_a_overlap),
                "region_b_overlap": float(chosen_row.region_b_overlap),
                "O_D": float(chosen_row.O_D),
                "O_N": float(chosen_row.O_N),
                "O_support": float(chosen_row.O_support) if pd.notna(chosen_row.O_support) else float("nan"),
                "O_harm": float(chosen_row.O_harm) if pd.notna(chosen_row.O_harm) else float("nan"),
                "balanced_score": float(chosen_row.balanced_score),
                "gamma_penalty": float(dn_overlap_penalty),
                "selection_score": float(chosen_row.selection_score),
                "rescue_score": float(chosen_row.rescue_score) if pd.notna(chosen_row.rescue_score) else float("nan"),
                "support_fraction": float(chosen_row.support_fraction) if pd.notna(chosen_row.support_fraction) else float("nan"),
                "harm_fraction": float(chosen_row.harm_fraction) if pd.notna(chosen_row.harm_fraction) else float("nan"),
                "overlap_total": float(chosen_row.overlap_total),
                "selected_rank": int(chosen_row.selected_rank),
                "wrong_selection_rank": int(chosen_row.wrong_selection_rank) if pd.notna(chosen_row.wrong_selection_rank) else -1,
                "prediction": int(baseline_output["prediction"]),
                "is_correct": int(int(baseline_output["prediction"]) == int(record.probe_label)),
                "true_label_score": float(baseline_output["true_label_score"]),
                "best_wrong_score": float(baseline_output["best_wrong_score"]),
                "best_wrong_label": int(baseline_output["best_wrong_label"]),
                "margin": float(baseline_output["margin"]),
                "baseline_wrong0_label": int(record.baseline_wrong0_label),
                "wrong0_label": int(record.baseline_wrong0_label),
                "wrong0_score": float(baseline_output.get("wrong0_score", float("nan"))),
                "margin_fixed_wrong0": float(baseline_output.get("margin_fixed_wrong0", float("nan"))),
                "same_wrong0_persist": int(baseline_output.get("same_wrong0_persist", int(record.baseline_wrong0_label >= 0))),
                "other_wrong_drift": int(baseline_output.get("other_wrong_drift", 0)),
                "first_fire_t_probe": int(baseline_output["first_fire_t_probe"]),
                "backend": str(baseline_output["backend"]),
                "readout_step": int(baseline_output["readout_step"]),
                "D_area": int(D_mask.sum()),
                "N_area": int(N_mask.sum()),
                "D_threshold": float(dn_spec["D_threshold"]),
                "N_threshold": float(dn_spec["N_threshold"]),
                "k_pixels": int(dn_spec["k_pixels"]),
                "positive_area": int(record.positive_mask.sum()),
                "negative_area": int(record.negative_mask.sum()),
                "random_area": int(random_mask.sum()),
                "random_seed": int(pair_seed),
                "random_mask_source": str(random_source),
                "debug_metadata": "",
            }
            pair_condition_rows.append(base_row)
            for condition_name, condition_image in condition_images.items():
                pending_samples.append(condition_image.detach().cpu())
                pending_probes.append(record.image.detach().cpu())
                pending_labels.append(int(record.probe_label))
                pending_wrong0_labels.append(int(record.baseline_wrong0_label if str(partition) == "wrong" and bool(wrong_use_fixed_competitor) else -1))
                pending_meta.append(
                    {
                        **{
                            key: value
                            for key, value in base_row.items()
                            if key not in {"prediction", "is_correct", "true_label_score", "best_wrong_score", "best_wrong_label", "margin", "margin_fixed_wrong0", "first_fire_t_probe", "backend", "readout_step"}
                        },
                        "condition_name": str(condition_name),
                    }
                )

    if pending_meta:
        outputs = inference_backend.infer(
            sample_images=pending_samples,
            probe_images=pending_probes,
            probe_labels=pending_labels,
            fixed_competitor_labels=pending_wrong0_labels,
        )
        for meta, output in zip(pending_meta, outputs):
            pair_condition_rows.append(
                {
                    **meta,
                    **output,
                    "debug_metadata": "",
                }
            )

    pair_condition_df = pd.DataFrame(pair_condition_rows)
    if not pair_condition_df.empty:
        no_sample_ref = pair_condition_df[pair_condition_df["condition_name"] == "no-sample"][
            [
                "probe_id",
                "sample_id",
                "prediction",
                "margin",
                "is_correct",
                "true_label_score",
                "baseline_wrong0_label",
                "wrong0_score",
                "margin_fixed_wrong0",
            ]
        ].rename(
            columns={
                "prediction": "prediction_no_sample",
                "margin": "margin_no_sample",
                "is_correct": "is_correct_no_sample",
                "true_label_score": "true_label_score_no_sample",
                "baseline_wrong0_label": "baseline_wrong0_label_no_sample",
                "wrong0_score": "wrong0_score_no_sample",
                "margin_fixed_wrong0": "margin_fixed_wrong0_no_sample",
            }
        )
        full_ref = pair_condition_df[pair_condition_df["condition_name"] == "full-sample"][
            ["probe_id", "sample_id", "prediction", "margin", "is_correct"]
        ].rename(
            columns={
                "prediction": "prediction_full_sample",
                "margin": "margin_full_sample",
                "is_correct": "is_correct_full_sample",
            }
        )
        pair_condition_df = pair_condition_df.merge(no_sample_ref, on=["probe_id", "sample_id"], how="left")
        pair_condition_df = pair_condition_df.merge(full_ref, on=["probe_id", "sample_id"], how="left")
        pair_condition_df["margin_for_delta"] = np.where(
            pair_condition_df["runtime_probe_partition"] == "wrong",
            pair_condition_df["margin_fixed_wrong0"],
            pair_condition_df["margin"],
        )
        pair_condition_df["margin_for_delta_no_sample"] = np.where(
            pair_condition_df["runtime_probe_partition"] == "wrong",
            pair_condition_df["margin_fixed_wrong0_no_sample"],
            pair_condition_df["margin_no_sample"],
        )
        pair_condition_df["delta_dir"] = pair_condition_df["margin_for_delta"] - pair_condition_df["margin_for_delta_no_sample"]
        pair_condition_df["effect_vs_no_sample"] = pair_condition_df["delta_dir"]
        pair_condition_df["margin_shift_vs_no_sample"] = pair_condition_df["delta_dir"]
        pair_condition_df["prediction_flip_vs_no_sample"] = (pair_condition_df["prediction"] != pair_condition_df["prediction_no_sample"]).astype(np.int64)
        pair_condition_df["prediction_flip_vs_full_sample"] = (pair_condition_df["prediction"] != pair_condition_df["prediction_full_sample"]).astype(np.int64)
        pair_condition_df["delta_dir_fixed_wrong0"] = (
            pair_condition_df["margin_fixed_wrong0"] - pair_condition_df["margin_fixed_wrong0_no_sample"]
        )
        pair_condition_df["rescued_to_true"] = (pair_condition_df["prediction"] == pair_condition_df["probe_label"]).astype(np.int64)
        pair_condition_df["same_wrong0_persist"] = (
            (pair_condition_df["baseline_wrong0_label"] >= 0)
            & (pair_condition_df["prediction"] == pair_condition_df["baseline_wrong0_label"])
        ).astype(np.int64)
        pair_condition_df["other_wrong_drift"] = (
            (pair_condition_df["baseline_wrong0_label"] >= 0)
            & (pair_condition_df["prediction"] != pair_condition_df["probe_label"])
            & (pair_condition_df["prediction"] != pair_condition_df["baseline_wrong0_label"])
        ).astype(np.int64)
        order_map = _condition_order_key_map()
        pair_condition_df = (
            pair_condition_df.assign(_condition_order=pair_condition_df["condition_name"].map(order_map))
            .sort_values(["probe_partition", "probe_id", "sample_id", "_condition_order"], kind="stable")
            .drop(columns=["_condition_order"])
            .reset_index(drop=True)
        )

    overlap_df = pd.concat(overlap_rows, axis=0, ignore_index=True) if overlap_rows else pd.DataFrame()
    selected_pairs_df = pd.DataFrame(selected_pair_rows)
    selected_pairs_ranked_df = selected_pairs_df.copy()
    inclusion_df = pd.DataFrame(inclusion_rows)
    wrong_selection_audit_df = pd.DataFrame()
    if not overlap_df.empty:
        wrong_selection_audit_df = overlap_df[overlap_df["runtime_probe_partition"] == "wrong"].copy()
        if not wrong_selection_audit_df.empty:
            selected_keys = set(
                zip(
                    selected_pairs_df["probe_id"].astype(np.int64),
                    selected_pairs_df["sample_id"].astype(np.int64),
                )
            )
            wrong_selection_audit_df["selected_flag"] = [
                int((int(probe_id), int(sample_id)) in selected_keys)
                for probe_id, sample_id in zip(
                    wrong_selection_audit_df["probe_id"].astype(np.int64),
                    wrong_selection_audit_df["sample_id"].astype(np.int64),
                )
            ]
    condition_correct_df = summarize_condition_metrics(pair_condition_df, partition="correct") if not pair_condition_df.empty else pd.DataFrame()
    condition_wrong_df = summarize_condition_metrics(pair_condition_df, partition="wrong") if not pair_condition_df.empty else pd.DataFrame()
    condition_all_df = pd.concat([condition_correct_df, condition_wrong_df], axis=0, ignore_index=True)
    probewise_correct_df = summarize_probe_metrics(pair_condition_df, partition="correct") if not pair_condition_df.empty else pd.DataFrame()
    probewise_wrong_df = summarize_probe_metrics(pair_condition_df, partition="wrong") if not pair_condition_df.empty else pd.DataFrame()
    causal_correct_df = summarize_causal_selectivity(probewise_correct_df, partition="correct")
    causal_wrong_df = summarize_causal_selectivity(probewise_wrong_df, partition="wrong")
    regression_df = summarize_regression(pair_condition_df) if not pair_condition_df.empty else pd.DataFrame()
    condition_wrong_fixed_df = condition_wrong_df.copy()
    probewise_wrong_fixed_df = probewise_wrong_df.copy()
    regression_wrong_fixed_df = regression_df[regression_df["probe_partition"] == "wrong"].copy() if not regression_df.empty else pd.DataFrame()

    provider_inventory = _ensure_columns(provider_inventory, ["probe_id"])
    inclusion_df = _ensure_columns(inclusion_df, ["probe_id"])
    overlap_df = _ensure_columns(overlap_df, ["runtime_probe_partition", "probe_partition", "probe_id", "sample_id"])
    selected_pairs_df = _ensure_columns(selected_pairs_df, ["runtime_probe_partition", "probe_partition", "probe_id", "sample_id"])
    selected_pairs_ranked_df = _ensure_columns(
        selected_pairs_ranked_df,
        ["runtime_probe_partition", "probe_partition", "probe_id", "sample_id", "O_D", "O_N", "selection_score", "selected_rank"],
    )
    wrong_selection_audit_df = _ensure_columns(
        wrong_selection_audit_df,
        ["runtime_probe_partition", "probe_partition", "probe_id", "sample_id", "selected_flag"],
    )
    pair_condition_df = _ensure_columns(pair_condition_df, ["runtime_probe_partition", "probe_partition", "probe_id", "sample_id", "condition_name"])
    condition_correct_df = _ensure_columns(condition_correct_df, ["runtime_probe_partition", "probe_partition", "condition_name"])
    condition_wrong_df = _ensure_columns(condition_wrong_df, ["runtime_probe_partition", "probe_partition", "condition_name"])
    condition_all_df = _ensure_columns(condition_all_df, ["runtime_probe_partition", "probe_partition", "condition_name"])
    probewise_correct_df = _ensure_columns(probewise_correct_df, ["runtime_probe_partition", "probe_partition", "probe_id"])
    probewise_wrong_df = _ensure_columns(probewise_wrong_df, ["runtime_probe_partition", "probe_partition", "probe_id"])
    condition_wrong_fixed_df = _ensure_columns(condition_wrong_fixed_df, ["runtime_probe_partition", "probe_partition", "condition_name"])
    probewise_wrong_fixed_df = _ensure_columns(probewise_wrong_fixed_df, ["runtime_probe_partition", "probe_partition", "probe_id"])
    causal_correct_df = _ensure_columns(causal_correct_df, ["n_probes"])
    causal_wrong_df = _ensure_columns(causal_wrong_df, ["n_probes"])
    regression_df = _ensure_columns(regression_df, ["runtime_probe_partition", "probe_partition", "analysis", "predictor"])
    regression_wrong_fixed_df = _ensure_columns(
        regression_wrong_fixed_df,
        ["runtime_probe_partition", "probe_partition", "analysis", "predictor"],
    )
    probe_dn_parameters_df = _ensure_columns(
        pd.DataFrame(dn_parameter_rows),
        ["probe_id", "probe_label", "runtime_probe_partition", "topk_fraction", "foreground_area", "D_area", "N_area", "D_threshold", "N_threshold", "k_pixels"],
    )
    direction_score_inventory_df = _ensure_columns(
        pd.DataFrame(direction_score_inventory_rows),
        ["probe_id", "probe_label", "baseline_is_correct", "baseline_wrong0_label", "importance_min", "importance_max", "importance_mean", "importance_std"],
    )
    wrong_fixed_audit_df = _ensure_columns(
        pd.DataFrame(wrong_fixed_audit_rows),
        ["probe_id", "probe_label", "wrong0_label", "baseline_prediction"],
    )

    inventory_csv = save_tidy_csv(provider_inventory, save_dir / "probe_diagnostic_inventory.csv", sort_by=["probe_id"])
    inclusion_csv = save_tidy_csv(inclusion_df, save_dir / "probe_inclusion_summary.csv", sort_by=["probe_id"])
    overlap_csv = save_tidy_csv(overlap_df, save_dir / "sample_overlap_scores.csv", sort_by=["probe_partition", "probe_id", "sample_id"])
    selected_csv = save_tidy_csv(selected_pairs_df, save_dir / "selected_probe_sample_pairs.csv", sort_by=["probe_partition", "probe_id", "sample_id"])
    selected_ranked_csv = save_tidy_csv(selected_pairs_ranked_df, save_dir / "selected_probe_sample_pairs_ranked.csv", sort_by=["probe_partition", "probe_id", "selected_rank", "sample_id"])
    wrong_selection_audit_csv = save_tidy_csv(
        wrong_selection_audit_df,
        save_dir / "wrong_selection_audit.csv",
        sort_by=["probe_id", "sample_id"],
    )
    probe_dn_parameters_csv = save_tidy_csv(probe_dn_parameters_df, save_dir / "probe_dn_parameters.csv", sort_by=["runtime_probe_partition", "probe_id"])
    direction_score_inventory_csv = save_tidy_csv(direction_score_inventory_df, save_dir / "direction_score_inventory.csv", sort_by=["probe_id"])
    pair_csv = save_tidy_csv(pair_condition_df, save_dir / "pair_condition_results.csv", sort_by=["probe_partition", "probe_id", "sample_id", "condition_name"])
    condition_correct_csv = save_tidy_csv(condition_correct_df, save_dir / "condition_summary_correct.csv", sort_by=["condition_name"])
    condition_wrong_csv = save_tidy_csv(condition_wrong_df, save_dir / "condition_summary_wrong.csv", sort_by=["condition_name"])
    condition_all_csv = save_tidy_csv(condition_all_df, save_dir / "condition_summary_all.csv", sort_by=["probe_partition", "condition_name"])
    probewise_correct_csv = save_tidy_csv(probewise_correct_df, save_dir / "probewise_summary_correct.csv", sort_by=["probe_id"])
    probewise_wrong_csv = save_tidy_csv(probewise_wrong_df, save_dir / "probewise_summary_wrong.csv", sort_by=["probe_id"])
    condition_wrong_fixed_csv = save_tidy_csv(
        condition_wrong_fixed_df,
        save_dir / "condition_summary_wrong_fixed_competitor.csv",
        sort_by=["condition_name"],
    )
    probewise_wrong_fixed_csv = save_tidy_csv(
        probewise_wrong_fixed_df,
        save_dir / "probewise_summary_wrong_fixed_competitor.csv",
        sort_by=["probe_id"],
    )
    causal_correct_csv = save_tidy_csv(causal_correct_df, save_dir / "causal_selectivity_summary_correct.csv")
    causal_wrong_csv = save_tidy_csv(causal_wrong_df, save_dir / "causal_selectivity_summary_wrong.csv")
    regression_csv = save_tidy_csv(regression_df, save_dir / "regression_summary.csv", sort_by=["probe_partition", "analysis", "predictor"])
    regression_wrong_fixed_csv = save_tidy_csv(
        regression_wrong_fixed_df,
        save_dir / "regression_summary_wrong_fixed_competitor.csv",
        sort_by=["analysis", "predictor"],
    )
    wrong_fixed_audit_csv = save_tidy_csv(
        wrong_fixed_audit_df,
        save_dir / "wrong_fixed_competitor_audit.csv",
        sort_by=["probe_id"],
    )

    plot_paths: Dict[str, object] = {}
    if make_plots and included_records:
        for probe_partition, condition_df, probe_df in (
            ("correct", condition_correct_df, probewise_correct_df),
            ("wrong", condition_wrong_df, probewise_wrong_df),
        ):
            if summary_split_by_baseline:
                subset_records = [
                    record
                    for record in included_records
                    if runtime_partition_by_probe_id.get(int(record.probe_id), "") == probe_partition
                ]
                if subset_records:
                    plot_paths[f"probe_overview_{probe_partition}"] = save_figure_all_formats(
                        make_probe_overview_figure(subset_records, partition=probe_partition),
                        save_dir / f"figure_probe_overview_{probe_partition}",
                    )
                    plt.close("all")
                exemplar = select_exemplar_pair(selected_pairs_df, probe_df, partition=probe_partition)
                if exemplar is not None:
                    probe_id, sample_id = exemplar
                    selected_for_probe = selected_pairs_df[
                        (selected_pairs_df["runtime_probe_partition"] == probe_partition) & (selected_pairs_df["probe_id"] == int(probe_id))
                    ]["sample_id"].astype(np.int64).tolist()
                    plot_paths[f"sample_overlap_scatter_{probe_partition}"] = save_figure_all_formats(
                        make_overlap_scatter_figure(
                            overlap_df,
                            probe_id=probe_id,
                            selected_sample_ids=selected_for_probe,
                            partition=probe_partition,
                        ),
                        save_dir / f"figure_sample_overlap_scatter_{probe_partition}",
                    )
                    plt.close("all")
                    exemplar_rows = pair_condition_df[
                        (pair_condition_df["runtime_probe_partition"] == probe_partition)
                        & (pair_condition_df["probe_id"] == int(probe_id))
                        & (pair_condition_df["sample_id"] == int(sample_id))
                    ].copy()
                    plot_paths[f"exemplar_surgery_{probe_partition}"] = save_figure_all_formats(
                        make_exemplar_surgery_figure(
                            pair_rows=exemplar_rows,
                            probe_record=probe_map[int(probe_id)],
                            sample_image_lookup=condition_sample_lookup[(int(probe_id), int(sample_id))],
                            partition=probe_partition,
                        ),
                        save_dir / f"figure_exemplar_surgery_{probe_partition}",
                    )
                    plt.close("all")
                subset_pair_df = pair_condition_df[pair_condition_df["runtime_probe_partition"] == probe_partition].copy()
                if not subset_pair_df.empty:
                    plot_paths[f"population_pairs_{probe_partition}"] = save_figure_all_formats(
                        make_population_pair_figure(subset_pair_df, partition=probe_partition),
                        save_dir / f"figure_population_pairs_{probe_partition}",
                    )
                    plt.close("all")
                if not condition_df.empty:
                    plot_paths[f"accuracy_conditions_{probe_partition}"] = save_figure_all_formats(
                        make_accuracy_figure(condition_df, partition=probe_partition),
                        save_dir / f"figure_accuracy_conditions_{probe_partition}",
                    )
                    plt.close("all")
                if not probe_df.empty:
                    plot_paths[f"causal_selectivity_{probe_partition}"] = save_figure_all_formats(
                        make_selectivity_figure(probe_df, partition=probe_partition),
                        save_dir / f"figure_causal_selectivity_{probe_partition}",
                    )
                    plt.close("all")

    readme_path = save_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Diagnostic Probe Causal Surgery",
                "",
                "This experiment builds D/N regions from saved signed direction scores at causal runtime.",
                "",
                "## D/N Selection",
                "",
                "- `D` = foreground highest top-k signed direction scores.",
                "- `N` = foreground lowest top-k signed direction scores.",
                "- D/N construction does not depend on runtime partition.",
                "",
                "## Ranking and Effect",
                "",
                "- Sample ranking uses `selection_score = O_D - gamma * O_N`.",
                "- Main effect uses `delta_dir`.",
                "- Runtime partition only changes which margin feeds `delta_dir`.",
                "",
                "## Probe diagnostic sources",
                "",
                "- `--probe-source results`: load existing diagnostic outputs from a prior dense-scan result directory.",
                "- `--probe-source compute`: compute fresh probe diagnostics with `run_deterministic_discovery` and optionally cache them.",
                "",
                "## Main outputs",
                "",
                "- `probe_diagnostic_inventory.csv`",
                "- `probe_inclusion_summary.csv`",
                "- `direction_score_inventory.csv`",
                "- `probe_dn_parameters.csv`",
                "- `sample_overlap_scores.csv`",
                "- `selected_probe_sample_pairs.csv`",
                "- `selected_probe_sample_pairs_ranked.csv`",
                "- `wrong_selection_audit.csv`",
                "- `pair_condition_results.csv`",
                "- `condition_summary_correct.csv`",
                "- `condition_summary_wrong.csv`",
                "- `condition_summary_wrong_fixed_competitor.csv`",
                "- `probewise_summary_correct.csv`",
                "- `probewise_summary_wrong.csv`",
                "- `probewise_summary_wrong_fixed_competitor.csv`",
                "- `causal_selectivity_summary_correct.csv`",
                "- `causal_selectivity_summary_wrong.csv`",
                "- `regression_summary.csv`",
                "- `regression_summary_wrong_fixed_competitor.csv`",
                "- `wrong_fixed_competitor_audit.csv`",
                "",
                "## Examples",
                "",
                "```bash",
                "python diagnostic_probe_causal_surgery_experiment.py --probe-source results --probe-results-dir results/diagnostic_feature_overlap_experiment --probe_partition_mode correct_only",
                "python diagnostic_probe_causal_surgery_experiment.py --probe-source results --probe-results-dir results/diagnostic_feature_overlap_experiment --probe_partition_mode wrong_only",
                "python diagnostic_probe_causal_surgery_experiment.py --probe-source compute --probe-cache-dir results/diagnostic_probe_cache --top-k-samples-per-probe 5 --probe_partition_mode mixed",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "inventory_csv": inventory_csv,
        "inclusion_csv": inclusion_csv,
        "overlap_csv": overlap_csv,
        "selected_csv": selected_csv,
        "selected_ranked_csv": selected_ranked_csv,
        "probe_dn_parameters_csv": probe_dn_parameters_csv,
        "direction_score_inventory_csv": direction_score_inventory_csv,
        "wrong_selection_audit_csv": wrong_selection_audit_csv,
        "pair_csv": pair_csv,
        "condition_correct_csv": condition_correct_csv,
        "condition_wrong_csv": condition_wrong_csv,
        "condition_wrong_fixed_csv": condition_wrong_fixed_csv,
        "condition_all_csv": condition_all_csv,
        "probewise_correct_csv": probewise_correct_csv,
        "probewise_wrong_csv": probewise_wrong_csv,
        "probewise_wrong_fixed_csv": probewise_wrong_fixed_csv,
        "causal_correct_csv": causal_correct_csv,
        "causal_wrong_csv": causal_wrong_csv,
        "regression_csv": regression_csv,
        "regression_wrong_fixed_csv": regression_wrong_fixed_csv,
        "wrong_fixed_audit_csv": wrong_fixed_audit_csv,
        "readme_path": str(readme_path),
        "plot_paths": plot_paths,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe-centric causal surgery over runtime D/N regions built from signed direction scores."
    )
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--probe-source", type=str, default="results", choices=["results", "compute"])
    parser.add_argument("--probe-results-dir", type=str, default="results/diagnostic_feature_overlap_experiment")
    parser.add_argument("--probe-cache-dir", type=str, default="")
    parser.add_argument("--probe-ids", type=str, default="")
    parser.add_argument("--max-probes", type=int, default=2000)
    parser.add_argument("--top-k-samples-per-probe", type=int, default=3)
    parser.add_argument("--sample-pool", type=str, default="test", choices=["test"])
    parser.add_argument("--probe_partition_mode", type=str, default="mixed", choices=["correct_only", "wrong_only", "mixed"])
    parser.add_argument("--summary_split_by_baseline", action="store_true", default=True)
    parser.add_argument("--wrong_mask_scheme", type=str, default="support_harm", choices=["support_harm"])
    parser.add_argument("--wrong-harm-penalty", type=float, default=1.0)
    parser.add_argument("--wrong-use-fixed-competitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dn-topk-fraction", type=float, default=0.15)
    parser.add_argument("--dn-overlap-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=31415)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay-ms", type=float, default=500.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--readout-mode", type=str, default="decision_offset", choices=["decision_offset", "probe_last_step", "explicit_step"])
    parser.add_argument("--readout-step", type=int, default=-1)
    parser.add_argument("--voltage-pooling", type=str, default="top_m_mean", choices=["max", "top_m_mean", "full_mean"])
    parser.add_argument("--top-m", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--scan-stride", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--lambda-global", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.top_k_samples_per_probe <= 0 or args.batch_size <= 0 or args.top_m <= 0:
        raise ValueError("top-k-samples-per-probe, batch-size, and top-m must be positive")
    if args.sample_ms <= 0 or args.delay_ms < 0 or args.probe_ms <= 0 or args.dt_ms <= 0:
        raise ValueError("Invalid timing arguments")
    if args.patch_size <= 0 or args.scan_stride <= 0 or args.micro_batch_size <= 0:
        raise ValueError("patch-size, scan-stride, and micro-batch-size must be positive")
    if float(args.wrong_harm_penalty) < 0.0 or float(args.dn_overlap_penalty) < 0.0:
        raise ValueError("--wrong-harm-penalty and --dn-overlap-penalty must be non-negative")
    if float(args.dn_topk_fraction) <= 0.0 or float(args.dn_topk_fraction) >= 1.0:
        raise ValueError("--dn-topk-fraction must be in (0, 1)")

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    spec = ExperimentSpec(
        dt=float(args.dt_ms * ms),
        sample_ms=float(args.sample_ms),
        delay_ms=float(args.delay_ms),
        probe_ms=float(args.probe_ms),
    )
    readout_step = None if int(args.readout_step) < 0 else int(args.readout_step)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(spec.sample_ms, spec.probe_ms),
    )
    _, _, test_loader = build_mnist_skeleton_loader(root=args.dataset_root, batch_size=1, input_size=28, num_workers=0)
    dataset = test_loader.dataset
    _, dataset_labels, image_matrix_flat = build_dataset_arrays(dataset)

    explicit_probe_ids = parse_probe_ids(args.probe_ids) if str(args.probe_ids).strip() else []
    max_probes = int(args.max_probes)
    if args.probe_source == "results":
        probe_records, provider_inventory = load_probe_diagnostics(
            results_dir=args.probe_results_dir,
            dataset=dataset,
            probe_ids=explicit_probe_ids or None,
        )
    else:
        all_probe_ids = explicit_probe_ids if explicit_probe_ids else list(range(len(dataset)))
        if max_probes > 0:
            all_probe_ids = all_probe_ids[:max_probes]
        model_runtime = ModelRuntime(
            net=net,
            encoder=encoder,
            model_path=str(args.model_path),
            spec=spec,
            device=device,
            readout_mode=str(args.readout_mode),
            readout_step=readout_step,
            voltage_pooling=str(args.voltage_pooling),
            top_m=int(args.top_m),
        )
        scan_config = ScanConfig(
            model_path=str(args.model_path),
            patch_size=int(args.patch_size),
            scan_stride=int(args.scan_stride),
            batch_size=int(args.batch_size),
            micro_batch_size=int(args.micro_batch_size),
            lambda_global=float(args.lambda_global),
            debug_full_stride_override=False,
        )
        cache_dir = Path(args.probe_cache_dir) if str(args.probe_cache_dir).strip() else save_dir / "probe_cache"
        probe_records, provider_inventory = build_probe_diagnostics(
            dataset=dataset,
            model_runtime=model_runtime,
            scan_config=scan_config,
            cache_dir=cache_dir,
            probe_ids=all_probe_ids,
        )

    if max_probes > 0:
        keep_ids = {int(record.probe_id) for record in probe_records[:max_probes]}
        probe_records = [record for record in probe_records if int(record.probe_id) in keep_ids]
        provider_inventory = provider_inventory[provider_inventory["probe_id"].isin(sorted(keep_ids))].copy()

    backend = VoltageDMSInferenceBackend(
        net=net,
        encoder=encoder,
        spec=spec,
        device=device,
        batch_size=int(args.batch_size),
        readout_mode=str(args.readout_mode),
        readout_step=readout_step,
        voltage_pooling=str(args.voltage_pooling),
        top_m=int(args.top_m),
    )
    outputs = run_causal_surgery_assay(
        probe_records=probe_records,
        provider_inventory=provider_inventory,
        dataset=dataset,
        dataset_labels=dataset_labels,
        image_matrix_flat=image_matrix_flat,
        output_dir=save_dir,
        top_k_samples_per_probe=int(args.top_k_samples_per_probe),
        random_seed=int(args.random_seed),
        inference_backend=backend,
        dn_topk_fraction=float(args.dn_topk_fraction),
        dn_overlap_penalty=float(args.dn_overlap_penalty),
        probe_partition_mode=str(args.probe_partition_mode),
        summary_split_by_baseline=bool(args.summary_split_by_baseline),
        wrong_mask_scheme=str(args.wrong_mask_scheme),
        wrong_harm_penalty=float(args.wrong_harm_penalty),
        wrong_use_fixed_competitor=bool(args.wrong_use_fixed_competitor),
        make_plots=True,
    )
    run_config = save_run_config(
        {
            "model_path": str(args.model_path),
            "dataset_root": str(args.dataset_root),
            "save_dir": str(save_dir),
            "probe_source": str(args.probe_source),
            "probe_results_dir": str(args.probe_results_dir),
            "probe_cache_dir": str(args.probe_cache_dir),
            "probe_ids": explicit_probe_ids,
            "max_probes": int(args.max_probes),
            "top_k_samples_per_probe": int(args.top_k_samples_per_probe),
            "probe_partition_mode": str(args.probe_partition_mode),
            "summary_split_by_baseline": bool(args.summary_split_by_baseline),
            "wrong_mask_scheme": str(args.wrong_mask_scheme),
            "wrong_harm_penalty": float(args.wrong_harm_penalty),
            "wrong_use_fixed_competitor": bool(args.wrong_use_fixed_competitor),
            "dn_topk_fraction": float(args.dn_topk_fraction),
            "dn_overlap_penalty": float(args.dn_overlap_penalty),
            "seed": int(args.seed),
            "random_seed": int(args.random_seed),
            "device": str(device),
            "sample_ms": float(args.sample_ms),
            "delay_ms": float(args.delay_ms),
            "probe_ms": float(args.probe_ms),
            "dt_ms": float(args.dt_ms),
            "readout_mode": str(args.readout_mode),
            "readout_step": None if readout_step is None else int(readout_step),
            "voltage_pooling": str(args.voltage_pooling),
            "top_m": int(args.top_m),
            "outputs": outputs,
        },
        save_dir,
    )
    print(f"[Done] Saved: {run_config}")


if __name__ == "__main__":
    main()
