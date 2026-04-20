from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import run_monitored_dms_rollout
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.diagnostic_region_utils import (
    ExperimentSpec,
    bootstrap_rate_ci,
    build_dataset_arrays,
    rank_correlation,
    select_sample_types_for_probe,
)
from src.experiments.common.seed import mix_seed
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.statistics import paired_bootstrap_diff_summary, parse_delay_list, wilson_ci
from src.experiments.diagnostic.shared.behavioral_support import prepare_batch_spikes
from src.experiments.diagnostic.shared.region_estimation import estimate_diagnostic_regions
from src.plotting.common.io import (
    COLOR_DYNAMIC,
    COLOR_NOISE,
    COLOR_STATIC,
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
    validate_required_columns,
)
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    ALPHA_SCATTER,
    DELAY_SWEEP_COLORS,
    FIGSIZE_TWO_PANEL,
    FREEZE_CONDITION_COLORS as FREEZE_CONDITION_STYLE_COLORS,
    FREEZE_SAMPLE_TYPE_COLORS as FREEZE_SAMPLE_STYLE_COLORS,
    GRID_ALPHA,
    LAYER_STATE_COLORS,
    LINE_WIDTH_PRIMARY,
    LINE_WIDTH_REFERENCE,
    MARKER_CIRCLE,
    apply_standard_legend,
    case_grid_figsize,
)

SAMPLE_TYPES: Tuple[str, ...] = ("diagnostic_overlap", "baseline", "nondiagnostic_overlap")
LAYER_KEYS: Tuple[str, ...] = ("layer1", "layer2", "layer3")

FREEZE_CONFIGS_MAIN4: Dict[str, Tuple[str, ...]] = {
    "full_dynamic": (),
    "freeze_L1": ("layer1",),
    "freeze_L1_L2": ("layer1", "layer2"),
    "full_frozen": ("layer1", "layer2", "layer3"),
}
FREEZE_CONFIGS_EXTENDED: Dict[str, Tuple[str, ...]] = {
    **FREEZE_CONFIGS_MAIN4,
    "freeze_L2_only": ("layer2",),
    "freeze_L3_only": ("layer3",),
}
FREEZE_SEED_TOKENS: Dict[str, int] = {
    name: idx for idx, name in enumerate(FREEZE_CONFIGS_EXTENDED.keys(), start=1)
}
METRIC_SEED_TOKENS: Dict[str, int] = {
    "BMI": 1,
    "rescue_diff": 2,
    "bias_diff": 3,
    "overlap_harm": 4,
}

SAMPLE_TYPE_COLORS: Dict[str, str] = dict(FREEZE_SAMPLE_STYLE_COLORS)
FREEZE_CONDITION_COLORS: Dict[str, str] = dict(FREEZE_CONDITION_STYLE_COLORS)
LAYER_COLORS: Dict[str, str] = {
    "layer1": LAYER_STATE_COLORS["layer1"],
    "layer2": LAYER_STATE_COLORS["layer2"],
    "layer3": LAYER_STATE_COLORS["layer3"],
}


def _errorbar_from_ci(values: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    lows_arr = np.asarray(lows, dtype=np.float64)
    highs_arr = np.asarray(highs, dtype=np.float64)
    lower = np.minimum(lows_arr, highs_arr)
    upper = np.maximum(lows_arr, highs_arr)
    return np.vstack(
        [
            np.clip(values_arr - lower, a_min=0.0, a_max=None),
            np.clip(upper - values_arr, a_min=0.0, a_max=None),
        ]
    )


def _slope_from_xy(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return float("nan")
    x_use = x_arr[mask]
    y_use = y_arr[mask]
    x_centered = x_use - float(x_use.mean())
    denom = float(np.sum(x_centered ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x_centered * (y_use - float(y_use.mean()))) / denom)


def resolve_freeze_configs(config_set: str) -> Dict[str, Tuple[str, ...]]:
    if str(config_set) == "main4":
        return dict(FREEZE_CONFIGS_MAIN4)
    if str(config_set) == "extended":
        return dict(FREEZE_CONFIGS_EXTENDED)
    raise ValueError(f"Unsupported freeze config set: {config_set}")


def load_or_compute_diagnostic_regions(
    net,
    encoder,
    raw_images: torch.Tensor,
    dataset_labels: np.ndarray,
    spec: ExperimentSpec,
    trial_count: int,
    patch_size: int,
    delay_values_ms: Sequence[int],
    batch_size: int,
    baseline_batch_size: int,
    device: torch.device,
    seed: int,
    save_dir: Path,
    cache_diagnostic_regions: bool,
    probe_pool_limit: int,
    probe_pool_per_class: int,
    early_stop_multiplier: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, np.ndarray | int | str]]]:
    return estimate_diagnostic_regions(
        net=net,
        encoder=encoder,
        raw_images=raw_images,
        dataset_labels=dataset_labels,
        spec=spec,
        trial_count=trial_count,
        patch_size=patch_size,
        diagnostic_method="occlusion",
        delay_values_ms=delay_values_ms,
        batch_size=batch_size,
        baseline_batch_size=baseline_batch_size,
        device=device,
        seed=seed,
        save_dir=save_dir,
        cache_diagnostic_regions=cache_diagnostic_regions,
        probe_pool_limit=probe_pool_limit,
        probe_pool_per_class=probe_pool_per_class,
        early_stop_multiplier=early_stop_multiplier,
    )


def build_probe_sample_pairs(
    probe_region_summary: pd.DataFrame,
    mask_lookup: Mapping[int, Mapping[str, np.ndarray | int | str]],
    image_matrix_flat: np.ndarray,
    dataset_labels: np.ndarray,
    delay_values_ms: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows: List[Dict[str, object]] = []
    for row in probe_region_summary.itertuples(index=False):
        probe_id = int(row.probe_id)
        if int(row.is_region_valid) != 1:
            selection_rows.append(
                {
                    "probe_id": int(probe_id),
                    "probe_label": int(row.probe_label),
                    "selection_status": "excluded",
                    "selection_exclusion_reason": str(row.region_exclusion_reason),
                }
            )
            continue
        selection_rows.append(
            select_sample_types_for_probe(
                probe_id=probe_id,
                probe_label=int(row.probe_label),
                image_matrix_flat=image_matrix_flat,
                dataset_labels=dataset_labels,
                diagnostic_mask=np.asarray(mask_lookup[probe_id]["diagnostic_mask"], dtype=np.bool_),
                nondiagnostic_mask=np.asarray(mask_lookup[probe_id]["nondiagnostic_mask"], dtype=np.bool_),
            )
        )

    selection_df = pd.DataFrame(selection_rows).sort_values(["probe_id"], kind="stable").reset_index(drop=True)
    valid_selection = selection_df[selection_df["selection_status"] == "selected"].copy()
    if valid_selection.empty:
        raise ValueError("No probes survived diagnostic-region estimation and sample pairing.")

    rows: List[Dict[str, int | float | str]] = []
    pair_id = 0
    trial_id = 0
    for row in valid_selection.itertuples(index=False):
        for delay_ms in delay_values_ms:
            rows.append(
                {
                    "pair_id": int(pair_id),
                    "trial_id": int(trial_id),
                    "delay_ms": int(delay_ms),
                    "sample_type": "diagnostic_overlap",
                    "sample_id": int(row.diagnostic_sample_id),
                    "sample_label": int(row.diagnostic_sample_label),
                    "probe_id": int(row.probe_id),
                    "probe_label": int(row.probe_label),
                    "label_relation": str(row.label_relation),
                    "zero_sample": 0,
                    "diagnostic_overlap_score": float(row.diagnostic_overlap_score),
                    "nondiagnostic_overlap_score": float(row.diagnostic_nondiagnostic_overlap_score),
                }
            )
            trial_id += 1
            rows.append(
                {
                    "pair_id": int(pair_id),
                    "trial_id": int(trial_id),
                    "delay_ms": int(delay_ms),
                    "sample_type": "baseline",
                    "sample_id": -1,
                    "sample_label": -1,
                    "probe_id": int(row.probe_id),
                    "probe_label": int(row.probe_label),
                    "label_relation": "no_sample",
                    "zero_sample": 1,
                    "diagnostic_overlap_score": float("nan"),
                    "nondiagnostic_overlap_score": float("nan"),
                }
            )
            trial_id += 1
            rows.append(
                {
                    "pair_id": int(pair_id),
                    "trial_id": int(trial_id),
                    "delay_ms": int(delay_ms),
                    "sample_type": "nondiagnostic_overlap",
                    "sample_id": int(row.nondiagnostic_sample_id),
                    "sample_label": int(row.nondiagnostic_sample_label),
                    "probe_id": int(row.probe_id),
                    "probe_label": int(row.probe_label),
                    "label_relation": str(row.label_relation),
                    "zero_sample": 0,
                    "diagnostic_overlap_score": float(row.nondiagnostic_overlap_score),
                    "nondiagnostic_overlap_score": float(row.nondiagnostic_nondiagnostic_overlap_score),
                }
            )
            trial_id += 1
        pair_id += 1

    df_specs = pd.DataFrame(rows).sort_values(["delay_ms", "pair_id", "trial_id"], kind="stable").reset_index(drop=True)
    return selection_df, df_specs


def _boundary_state_means(
    boundary_state: Mapping[str, Mapping[str, torch.Tensor]],
    batch_size: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for layer_key in LAYER_KEYS:
        layer_state = boundary_state.get(layer_key, {})
        u_tensor = layer_state.get("u")
        x_tensor = layer_state.get("x")
        if not isinstance(u_tensor, torch.Tensor) or not isinstance(x_tensor, torch.Tensor):
            out[f"mean_u_{layer_key}"] = np.full(batch_size, np.nan, dtype=np.float32)
            out[f"mean_x_{layer_key}"] = np.full(batch_size, np.nan, dtype=np.float32)
            out[f"mean_ux_{layer_key}"] = np.full(batch_size, np.nan, dtype=np.float32)
            continue
        u_view = u_tensor.view(batch_size, -1).numpy().astype(np.float32, copy=False)
        x_view = x_tensor.view(batch_size, -1).numpy().astype(np.float32, copy=False)
        out[f"mean_u_{layer_key}"] = u_view.mean(axis=1).astype(np.float32, copy=False)
        out[f"mean_x_{layer_key}"] = x_view.mean(axis=1).astype(np.float32, copy=False)
        out[f"mean_ux_{layer_key}"] = (u_view * x_view).mean(axis=1).astype(np.float32, copy=False)
    return out


def apply_layerwise_ux_freeze(
    frozen_layers: Sequence[str],
) -> Callable[[object, Dict[str, object]], Dict[str, object]]:
    frozen_layer_set = tuple(str(layer_key) for layer_key in frozen_layers)

    def before_probe_fn(net, ctx: Dict[str, object]) -> Dict[str, object]:
        del ctx
        record: Dict[str, object] = {
            "freeze_applied": int(len(frozen_layer_set) > 0),
            "frozen_layers": ",".join(frozen_layer_set),
        }
        with torch.no_grad():
            for layer_key in LAYER_KEYS:
                layer = getattr(net, layer_key, None)
                if layer is None or getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
                    continue
                u_before = layer.u_pre.detach().view(layer.u_pre.shape[0], -1).mean(dim=1)
                x_before = layer.x_pre.detach().view(layer.x_pre.shape[0], -1).mean(dim=1)
                ux_before = (layer.u_pre.detach() * layer.x_pre.detach()).view(layer.u_pre.shape[0], -1).mean(dim=1)
                if layer_key in frozen_layer_set:
                    layer.u_pre.fill_(float(layer.stsp_U))
                    layer.x_pre.fill_(1.0)
                u_after = layer.u_pre.detach().view(layer.u_pre.shape[0], -1).mean(dim=1)
                x_after = layer.x_pre.detach().view(layer.x_pre.shape[0], -1).mean(dim=1)
                ux_after = (layer.u_pre.detach() * layer.x_pre.detach()).view(layer.u_pre.shape[0], -1).mean(dim=1)
                record[f"{layer_key}_was_frozen"] = int(layer_key in frozen_layer_set)
                record[f"{layer_key}_mean_u_before"] = float(u_before.mean().item())
                record[f"{layer_key}_mean_x_before"] = float(x_before.mean().item())
                record[f"{layer_key}_mean_ux_before"] = float(ux_before.mean().item())
                record[f"{layer_key}_mean_u_after"] = float(u_after.mean().item())
                record[f"{layer_key}_mean_x_after"] = float(x_after.mean().item())
                record[f"{layer_key}_mean_ux_after"] = float(ux_after.mean().item())
        return record

    return before_probe_fn


def run_layerwise_freeze_trials(
    net,
    encoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
    seed: int,
    freeze_configs: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    validate_required_columns(
        df_specs,
        [
            "pair_id",
            "trial_id",
            "delay_ms",
            "sample_type",
            "sample_id",
            "sample_label",
            "probe_id",
            "probe_label",
            "label_relation",
            "zero_sample",
            "diagnostic_overlap_score",
            "nondiagnostic_overlap_score",
        ],
    )
    grouped = list(df_specs.groupby(["delay_ms", "sample_type", "zero_sample"], sort=True))
    total_batches = sum(math.ceil(len(group) / batch_size) * len(freeze_configs) for _, group in grouped)
    records: List[Dict[str, int | float | str]] = []

    with tqdm(total=total_batches, desc="LayerwiseUXFreezeTrials") as pbar:
        for (delay_ms, sample_type, _), group in grouped:
            group = group.sort_values(["pair_id", "trial_id"], kind="stable").reset_index(drop=True)
            delay_steps = int(round((float(delay_ms) * ms) / spec.dt))
            for start in range(0, len(group), batch_size):
                batch = group.iloc[start:start + batch_size].copy()
                sample_spikes, probe_spikes = prepare_batch_spikes(
                    dataset=dataset,
                    batch_df=batch,
                    encoder=encoder,
                    spec=spec,
                    device=device,
                )
                for freeze_condition, frozen_layers in freeze_configs.items():
                    intervention_plan = {"before_probe_fn": apply_layerwise_ux_freeze(frozen_layers)}
                    with torch.no_grad():
                        out = run_monitored_dms_rollout(
                            net=net,
                            sample_spikes=sample_spikes,
                            probe_spikes=probe_spikes,
                            delay_steps=delay_steps,
                            stsp_mode="dynamic",
                            phase_reset=True,
                            intervention_plan=intervention_plan,
                            record_state_names=(),
                        )
                    pred = out["predictions"]["prediction_probe"].numpy().astype(np.int64, copy=False)
                    fire_t = out["predictions"]["first_fire_t_probe"].numpy().astype(np.int64, copy=False)
                    layer_means = _boundary_state_means(out["boundary_states"]["post_intervention"], batch_size=len(batch))
                    intervention_record = dict(out.get("intervention_record", {}))
                    frozen_layers_str = ",".join(str(layer) for layer in frozen_layers)
                    for idx_in_batch, row in enumerate(batch.itertuples(index=False)):
                        predicted_label = int(pred[idx_in_batch])
                        probe_label = int(row.probe_label)
                        sample_label = int(row.sample_label)
                        record: Dict[str, int | float | str] = {
                            "seed": int(seed),
                            "freeze_condition": str(freeze_condition),
                            "delay_ms": int(delay_ms),
                            "sample_type": str(sample_type),
                            "sample_id": int(row.sample_id),
                            "sample_label": int(sample_label),
                            "probe_id": int(row.probe_id),
                            "probe_label": int(probe_label),
                            "predicted_label": int(predicted_label),
                            "is_correct": int(predicted_label == probe_label),
                            "pred_equals_sample": int(sample_label >= 0 and predicted_label == sample_label),
                            "pred_equals_probe": int(predicted_label == probe_label),
                            "diagnostic_overlap_score": float(row.diagnostic_overlap_score)
                            if pd.notna(row.diagnostic_overlap_score)
                            else float("nan"),
                            "nondiagnostic_overlap_score": float(row.nondiagnostic_overlap_score)
                            if pd.notna(row.nondiagnostic_overlap_score)
                            else float("nan"),
                            "label_relation": str(row.label_relation),
                            "first_fire_t_probe": int(fire_t[idx_in_batch]),
                            "is_silent": int(predicted_label == -1),
                            "pair_id": int(row.pair_id),
                            "trial_id": int(row.trial_id),
                            "frozen_layers": frozen_layers_str,
                            "freeze_applied": int(intervention_record.get("freeze_applied", int(len(frozen_layers) > 0))),
                        }
                        for layer_key in LAYER_KEYS:
                            record[f"mean_u_{layer_key}"] = float(layer_means[f"mean_u_{layer_key}"][idx_in_batch])
                            record[f"mean_x_{layer_key}"] = float(layer_means[f"mean_x_{layer_key}"][idx_in_batch])
                            record[f"mean_ux_{layer_key}"] = float(layer_means[f"mean_ux_{layer_key}"][idx_in_batch])
                        records.append(record)
                    pbar.update(1)

    df_trials = pd.DataFrame(records)
    order_map = {name: idx for idx, name in enumerate(freeze_configs)}
    sample_order_map = {name: idx for idx, name in enumerate(SAMPLE_TYPES)}
    df_trials["freeze_condition_order"] = df_trials["freeze_condition"].map(order_map).astype(np.int64)
    df_trials["sample_type_order"] = df_trials["sample_type"].map(sample_order_map).astype(np.int64)
    return (
        df_trials.sort_values(
            ["seed", "freeze_condition_order", "delay_ms", "pair_id", "sample_type_order"],
            kind="stable",
        )
        .drop(columns=["freeze_condition_order", "sample_type_order"])
        .reset_index(drop=True)
    )


def build_pairwide_frame(df_trials: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, int | float | str]] = []
    group_cols = ["freeze_condition", "delay_ms", "pair_id"]
    for (freeze_condition, delay_ms, pair_id), subset in df_trials.groupby(group_cols, sort=True):
        row_by_sample = {str(row.sample_type): row for row in subset.itertuples(index=False)}
        if not all(sample_type in row_by_sample for sample_type in SAMPLE_TYPES):
            continue
        baseline = row_by_sample["baseline"]
        diagnostic = row_by_sample["diagnostic_overlap"]
        nondiagnostic = row_by_sample["nondiagnostic_overlap"]
        rows.append(
            {
                "freeze_condition": str(freeze_condition),
                "delay_ms": int(delay_ms),
                "pair_id": int(pair_id),
                "probe_id": int(baseline.probe_id),
                "probe_label": int(baseline.probe_label),
                "label_relation": str(diagnostic.label_relation),
                "baseline_is_correct": int(baseline.is_correct),
                "baseline_predicted_label": int(baseline.predicted_label),
                "diagnostic_is_correct": int(diagnostic.is_correct),
                "diagnostic_predicted_label": int(diagnostic.predicted_label),
                "diagnostic_sample_label": int(diagnostic.sample_label),
                "diagnostic_sample_id": int(diagnostic.sample_id),
                "diagnostic_overlap_score": float(diagnostic.diagnostic_overlap_score),
                "diagnostic_nondiagnostic_overlap_score": float(diagnostic.nondiagnostic_overlap_score),
                "diagnostic_error_rate": int(int(diagnostic.predicted_label) != int(diagnostic.probe_label)),
                "diagnostic_sample_capture": int(
                    int(diagnostic.sample_label) >= 0
                    and int(diagnostic.predicted_label) == int(diagnostic.sample_label)
                    and int(diagnostic.predicted_label) != int(diagnostic.probe_label)
                ),
                "nondiagnostic_is_correct": int(nondiagnostic.is_correct),
                "nondiagnostic_predicted_label": int(nondiagnostic.predicted_label),
                "nondiagnostic_sample_label": int(nondiagnostic.sample_label),
                "nondiagnostic_sample_id": int(nondiagnostic.sample_id),
                "nondiagnostic_overlap_score": float(nondiagnostic.diagnostic_overlap_score),
                "nondiagnostic_nondiagnostic_overlap_score": float(nondiagnostic.nondiagnostic_overlap_score),
                "nondiagnostic_error_rate": int(int(nondiagnostic.predicted_label) != int(nondiagnostic.probe_label)),
                "nondiagnostic_sample_capture": int(
                    int(nondiagnostic.sample_label) >= 0
                    and int(nondiagnostic.predicted_label) == int(nondiagnostic.sample_label)
                    and int(nondiagnostic.predicted_label) != int(nondiagnostic.probe_label)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["freeze_condition", "delay_ms", "pair_id"], kind="stable").reset_index(drop=True)


def compute_accuracy_summary(df_trials: pd.DataFrame, num_boot: int, seed: int) -> pd.DataFrame:
    del num_boot, seed
    rows: List[Dict[str, int | float | str]] = []
    for (freeze_condition, delay_ms, sample_type), subset in df_trials.groupby(
        ["freeze_condition", "delay_ms", "sample_type"],
        sort=True,
    ):
        n_trials = int(len(subset))
        n_correct = int(subset["is_correct"].sum())
        ci_low, ci_high = wilson_ci(n_correct, n_trials)
        rows.append(
            {
                "freeze_condition": str(freeze_condition),
                "delay_ms": int(delay_ms),
                "sample_type": str(sample_type),
                "n_trials": int(n_trials),
                "accuracy": 100.0 * float(n_correct) / float(n_trials) if n_trials > 0 else float("nan"),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
    return pd.DataFrame(rows).sort_values(["freeze_condition", "delay_ms", "sample_type"], kind="stable").reset_index(drop=True)


def compute_beneficial_memory_index(
    df_trials: pd.DataFrame,
    num_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairwide = build_pairwide_frame(df_trials)
    rows: List[Dict[str, int | float | str]] = []
    for (freeze_condition, delay_ms), subset in pairwide.groupby(["freeze_condition", "delay_ms"], sort=True):
        diff_summary = paired_bootstrap_diff_summary(
            subset["diagnostic_is_correct"].to_numpy(dtype=np.float64),
            subset["nondiagnostic_is_correct"].to_numpy(dtype=np.float64),
            n_boot=num_boot,
            seed=mix_seed(seed, 701, int(delay_ms), FREEZE_SEED_TOKENS[str(freeze_condition)]),
        )
        rows.append(
            {
                "freeze_condition": str(freeze_condition),
                "delay_ms": int(delay_ms),
                "n_pairs": int(len(subset)),
                "beneficial_memory_index": float(diff_summary["observed_diff_pp"]),
                "ci_low": float(diff_summary["ci_low"]),
                "ci_high": float(diff_summary["ci_high"]),
            }
        )
    df_bmi = pd.DataFrame(rows).sort_values(["freeze_condition", "delay_ms"], kind="stable").reset_index(drop=True)
    return df_bmi, pairwide


def compute_rescue_and_bias_metrics(
    df_trials: pd.DataFrame,
    num_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairwide = build_pairwide_frame(df_trials)
    rows: List[Dict[str, int | float | str]] = []
    for (freeze_condition, delay_ms), subset in pairwide.groupby(["freeze_condition", "delay_ms"], sort=True):
        baseline_incorrect = subset[subset["baseline_is_correct"] == 0].copy()
        diagnostic_rescue = baseline_incorrect["diagnostic_is_correct"].to_numpy(dtype=np.float64)
        nondiagnostic_rescue = baseline_incorrect["nondiagnostic_is_correct"].to_numpy(dtype=np.float64)

        rescue_diag_ci_low, rescue_diag_ci_high = bootstrap_rate_ci(
            diagnostic_rescue,
            n_boot=num_boot,
            seed=mix_seed(seed, 801, int(delay_ms), FREEZE_SEED_TOKENS[str(freeze_condition)]),
        )
        rescue_nond_ci_low, rescue_nond_ci_high = bootstrap_rate_ci(
            nondiagnostic_rescue,
            n_boot=num_boot,
            seed=mix_seed(seed, 811, int(delay_ms), FREEZE_SEED_TOKENS[str(freeze_condition)]),
        )
        rescue_diff_summary = paired_bootstrap_diff_summary(
            diagnostic_rescue,
            nondiagnostic_rescue,
            n_boot=num_boot,
            seed=mix_seed(seed, 821, int(delay_ms), FREEZE_SEED_TOKENS[str(freeze_condition)]),
        ) if len(baseline_incorrect) > 0 else {
            "observed_diff_pp": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

        diagnostic_capture = subset["diagnostic_sample_capture"].to_numpy(dtype=np.float64)
        nondiagnostic_capture = subset["nondiagnostic_sample_capture"].to_numpy(dtype=np.float64)
        diagnostic_error = subset["diagnostic_error_rate"].to_numpy(dtype=np.float64)
        nondiagnostic_error = subset["nondiagnostic_error_rate"].to_numpy(dtype=np.float64)

        diag_bias_ci_low, diag_bias_ci_high = bootstrap_rate_ci(
            diagnostic_capture,
            n_boot=num_boot,
            seed=mix_seed(seed, 831, int(delay_ms), FREEZE_SEED_TOKENS[str(freeze_condition)]),
        )
        nond_bias_ci_low, nond_bias_ci_high = bootstrap_rate_ci(
            nondiagnostic_capture,
            n_boot=num_boot,
            seed=mix_seed(seed, 841, int(delay_ms), FREEZE_SEED_TOKENS[str(freeze_condition)]),
        )
        bias_diff_summary = paired_bootstrap_diff_summary(
            nondiagnostic_capture,
            diagnostic_capture,
            n_boot=num_boot,
            seed=mix_seed(seed, 851, int(delay_ms), FREEZE_SEED_TOKENS[str(freeze_condition)]),
        )
        diag_error_ci_low, diag_error_ci_high = bootstrap_rate_ci(
            diagnostic_error,
            n_boot=num_boot,
            seed=mix_seed(seed, 861, int(delay_ms), FREEZE_SEED_TOKENS[str(freeze_condition)]),
        )
        nond_error_ci_low, nond_error_ci_high = bootstrap_rate_ci(
            nondiagnostic_error,
            n_boot=num_boot,
            seed=mix_seed(seed, 871, int(delay_ms), FREEZE_SEED_TOKENS[str(freeze_condition)]),
        )

        rows.append(
            {
                "freeze_condition": str(freeze_condition),
                "delay_ms": int(delay_ms),
                "n_pairs": int(len(subset)),
                "n_baseline_incorrect": int(len(baseline_incorrect)),
                "rescue_rate_diagnostic": 100.0 * float(diagnostic_rescue.mean()) if diagnostic_rescue.size > 0 else float("nan"),
                "rescue_rate_diagnostic_ci_low": float(rescue_diag_ci_low),
                "rescue_rate_diagnostic_ci_high": float(rescue_diag_ci_high),
                "rescue_rate_nondiagnostic": 100.0 * float(nondiagnostic_rescue.mean()) if nondiagnostic_rescue.size > 0 else float("nan"),
                "rescue_rate_nondiagnostic_ci_low": float(rescue_nond_ci_low),
                "rescue_rate_nondiagnostic_ci_high": float(rescue_nond_ci_high),
                "rescue_diff": float(rescue_diff_summary["observed_diff_pp"]),
                "rescue_diff_ci_low": float(rescue_diff_summary["ci_low"]),
                "rescue_diff_ci_high": float(rescue_diff_summary["ci_high"]),
                "misleading_bias_diagnostic": 100.0 * float(diagnostic_capture.mean()) if diagnostic_capture.size > 0 else float("nan"),
                "misleading_bias_diagnostic_ci_low": float(diag_bias_ci_low),
                "misleading_bias_diagnostic_ci_high": float(diag_bias_ci_high),
                "misleading_bias_nondiagnostic": 100.0 * float(nondiagnostic_capture.mean()) if nondiagnostic_capture.size > 0 else float("nan"),
                "misleading_bias_nondiagnostic_ci_low": float(nond_bias_ci_low),
                "misleading_bias_nondiagnostic_ci_high": float(nond_bias_ci_high),
                "bias_diff": float(bias_diff_summary["observed_diff_pp"]),
                "bias_diff_ci_low": float(bias_diff_summary["ci_low"]),
                "bias_diff_ci_high": float(bias_diff_summary["ci_high"]),
                "error_rate_diagnostic": 100.0 * float(diagnostic_error.mean()) if diagnostic_error.size > 0 else float("nan"),
                "error_rate_diagnostic_ci_low": float(diag_error_ci_low),
                "error_rate_diagnostic_ci_high": float(diag_error_ci_high),
                "error_rate_nondiagnostic": 100.0 * float(nondiagnostic_error.mean()) if nondiagnostic_error.size > 0 else float("nan"),
                "error_rate_nondiagnostic_ci_low": float(nond_error_ci_low),
                "error_rate_nondiagnostic_ci_high": float(nond_error_ci_high),
            }
        )
    df_summary = pd.DataFrame(rows).sort_values(["freeze_condition", "delay_ms"], kind="stable").reset_index(drop=True)
    return df_summary, pairwide


def compute_overlap_harm(
    df_trials: pd.DataFrame,
    num_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairwide = build_pairwide_frame(df_trials)
    rows: List[Dict[str, int | float | str]] = []
    for (freeze_condition, delay_ms), subset in pairwide.groupby(["freeze_condition", "delay_ms"], sort=True):
        diff_summary = paired_bootstrap_diff_summary(
            subset["baseline_is_correct"].to_numpy(dtype=np.float64),
            subset["nondiagnostic_is_correct"].to_numpy(dtype=np.float64),
            n_boot=num_boot,
            seed=mix_seed(seed, 901, int(delay_ms), FREEZE_SEED_TOKENS[str(freeze_condition)]),
        )
        rows.append(
            {
                "freeze_condition": str(freeze_condition),
                "delay_ms": int(delay_ms),
                "n_pairs": int(len(subset)),
                "overlap_harm": float(diff_summary["observed_diff_pp"]),
                "ci_low": float(diff_summary["ci_low"]),
                "ci_high": float(diff_summary["ci_high"]),
                "baseline_accuracy": 100.0 * float(subset["baseline_is_correct"].mean()),
                "nondiagnostic_accuracy": 100.0 * float(subset["nondiagnostic_is_correct"].mean()),
            }
        )
    df_overlap = pd.DataFrame(rows).sort_values(["freeze_condition", "delay_ms"], kind="stable").reset_index(drop=True)
    return df_overlap, pairwide


def _summarize_layer_state_group(
    subset: pd.DataFrame,
    freeze_condition: str,
    delay_ms: int,
    sample_type: str,
) -> Dict[str, int | float | str]:
    row: Dict[str, int | float | str] = {
        "summary_type": "group_mean",
        "freeze_condition": str(freeze_condition),
        "delay_ms": int(delay_ms),
        "sample_type": str(sample_type),
        "n_trials": int(len(subset)),
    }
    for layer_key in LAYER_KEYS:
        for state_name in ("u", "x", "ux"):
            col = f"mean_{state_name}_{layer_key}"
            values = subset[col].to_numpy(dtype=np.float64)
            row[col] = float(np.nanmean(values)) if values.size > 0 else float("nan")
    return row


def compute_layer_state_summary(
    df_trials: pd.DataFrame,
    df_bmi: pd.DataFrame,
    df_rescue_bias: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, int | float | str]] = []
    for (freeze_condition, delay_ms, sample_type), subset in df_trials.groupby(
        ["freeze_condition", "delay_ms", "sample_type"],
        sort=True,
    ):
        rows.append(_summarize_layer_state_group(subset, str(freeze_condition), int(delay_ms), str(sample_type)))
    for (freeze_condition, delay_ms), subset in df_trials.groupby(["freeze_condition", "delay_ms"], sort=True):
        rows.append(_summarize_layer_state_group(subset, str(freeze_condition), int(delay_ms), "all_samples"))

    df_group = pd.DataFrame(rows).sort_values(
        ["summary_type", "freeze_condition", "delay_ms", "sample_type"],
        kind="stable",
    ).reset_index(drop=True)

    merged = df_group[df_group["sample_type"] == "all_samples"].merge(
        df_bmi[["freeze_condition", "delay_ms", "beneficial_memory_index"]],
        on=["freeze_condition", "delay_ms"],
        how="left",
    ).merge(
        df_rescue_bias[["freeze_condition", "delay_ms", "bias_diff"]],
        on=["freeze_condition", "delay_ms"],
        how="left",
    )
    corr_rows: List[Dict[str, int | float | str]] = []
    for layer_key in LAYER_KEYS:
        ux_values = merged[f"mean_ux_{layer_key}"].to_numpy(dtype=np.float64)
        corr_rows.append(
            {
                "summary_type": "metric_correlation",
                "freeze_condition": "all",
                "delay_ms": -1,
                "sample_type": "all_samples",
                "n_trials": int(len(merged)),
                "target_metric": "beneficial_memory_index",
                "layer": str(layer_key),
                "state_metric": "mean_ux",
                "rank_correlation": float(rank_correlation(ux_values, merged["beneficial_memory_index"].to_numpy(dtype=np.float64))),
            }
        )
        corr_rows.append(
            {
                "summary_type": "metric_correlation",
                "freeze_condition": "all",
                "delay_ms": -1,
                "sample_type": "all_samples",
                "n_trials": int(len(merged)),
                "target_metric": "bias_diff",
                "layer": str(layer_key),
                "state_metric": "mean_ux",
                "rank_correlation": float(rank_correlation(ux_values, merged["bias_diff"].to_numpy(dtype=np.float64))),
            }
        )
    df_corr = pd.DataFrame(corr_rows)
    return pd.concat([df_group, df_corr], ignore_index=True, sort=False)


def _arrays_from_pairwide(subset: pd.DataFrame) -> Dict[str, np.ndarray]:
    numeric_cols = [
        "pair_id",
        "probe_id",
        "probe_label",
        "baseline_is_correct",
        "diagnostic_is_correct",
        "diagnostic_error_rate",
        "diagnostic_sample_capture",
        "nondiagnostic_is_correct",
        "nondiagnostic_error_rate",
        "nondiagnostic_sample_capture",
    ]
    return {column: subset[column].to_numpy(dtype=np.float64) for column in numeric_cols}


def _metric_bmi(arrays: Mapping[str, np.ndarray]) -> float:
    return 100.0 * float(np.mean(arrays["diagnostic_is_correct"]) - np.mean(arrays["nondiagnostic_is_correct"]))


def _metric_rescue_diff(arrays: Mapping[str, np.ndarray]) -> float:
    mask = arrays["baseline_is_correct"] == 0.0
    if int(mask.sum()) == 0:
        return float("nan")
    return 100.0 * float(np.mean(arrays["diagnostic_is_correct"][mask]) - np.mean(arrays["nondiagnostic_is_correct"][mask]))


def _metric_bias_diff(arrays: Mapping[str, np.ndarray]) -> float:
    return 100.0 * float(np.mean(arrays["nondiagnostic_sample_capture"]) - np.mean(arrays["diagnostic_sample_capture"]))


def _metric_overlap_harm(arrays: Mapping[str, np.ndarray]) -> float:
    return 100.0 * float(np.mean(arrays["baseline_is_correct"]) - np.mean(arrays["nondiagnostic_is_correct"]))


def _bootstrap_metric_difference(
    arrays_a: Mapping[str, np.ndarray],
    arrays_b: Mapping[str, np.ndarray],
    metric_fn: Callable[[Mapping[str, np.ndarray]], float],
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    if not arrays_a or not arrays_b:
        return {
            "estimate_pp": float("nan"),
            "ci_low_pp": float("nan"),
            "ci_high_pp": float("nan"),
            "p_two_sided": float("nan"),
            "n_pairs": 0.0,
        }
    n_pairs = int(next(iter(arrays_a.values())).shape[0])
    if n_pairs == 0:
        return {
            "estimate_pp": float("nan"),
            "ci_low_pp": float("nan"),
            "ci_high_pp": float("nan"),
            "p_two_sided": float("nan"),
            "n_pairs": 0.0,
        }
    observed_a = metric_fn(arrays_a)
    observed_b = metric_fn(arrays_b)
    observed = float(observed_a - observed_b) if np.isfinite(observed_a) and np.isfinite(observed_b) else float("nan")
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot, dtype=np.float64)
    boot.fill(np.nan)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n_pairs, size=n_pairs)
        sample_a = {key: value[sample_idx] for key, value in arrays_a.items()}
        sample_b = {key: value[sample_idx] for key, value in arrays_b.items()}
        val_a = metric_fn(sample_a)
        val_b = metric_fn(sample_b)
        if np.isfinite(val_a) and np.isfinite(val_b):
            boot[idx] = float(val_a - val_b)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return {
            "estimate_pp": float(observed),
            "ci_low_pp": float("nan"),
            "ci_high_pp": float("nan"),
            "p_two_sided": float("nan"),
            "n_pairs": float(n_pairs),
        }
    p_two_sided = 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
    return {
        "estimate_pp": float(observed),
        "ci_low_pp": float(np.percentile(boot, 2.5)),
        "ci_high_pp": float(np.percentile(boot, 97.5)),
        "p_two_sided": float(min(1.0, max(0.0, p_two_sided))),
        "n_pairs": float(n_pairs),
    }


def _bootstrap_metric_delay_slope(
    subset: pd.DataFrame,
    metric_fn: Callable[[Mapping[str, np.ndarray]], float],
    delays: Sequence[int],
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    curves: Dict[int, Dict[str, np.ndarray]] = {}
    observed_values: List[float] = []
    for delay_ms in delays:
        delay_subset = subset[subset["delay_ms"] == int(delay_ms)].copy()
        if delay_subset.empty:
            return {
                "estimate_pp": float("nan"),
                "ci_low_pp": float("nan"),
                "ci_high_pp": float("nan"),
                "n_pairs": 0.0,
            }
        arrays = _arrays_from_pairwide(delay_subset)
        curves[int(delay_ms)] = arrays
        observed_values.append(metric_fn(arrays))
    observed_slope = _slope_from_xy(np.asarray(delays, dtype=np.float64), np.asarray(observed_values, dtype=np.float64))
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot, dtype=np.float64)
    boot.fill(np.nan)
    for idx in range(n_boot):
        curve_values: List[float] = []
        for delay_ms in delays:
            arrays = curves[int(delay_ms)]
            n = int(next(iter(arrays.values())).shape[0])
            sample_idx = rng.integers(0, n, size=n)
            sample_arrays = {key: value[sample_idx] for key, value in arrays.items()}
            curve_values.append(metric_fn(sample_arrays))
        slope = _slope_from_xy(np.asarray(delays, dtype=np.float64), np.asarray(curve_values, dtype=np.float64))
        if np.isfinite(slope):
            boot[idx] = float(slope)
    boot = boot[np.isfinite(boot)]
    return {
        "estimate_pp": float(observed_slope),
        "ci_low_pp": float(np.percentile(boot, 2.5)) if boot.size > 0 else float("nan"),
        "ci_high_pp": float(np.percentile(boot, 97.5)) if boot.size > 0 else float("nan"),
        "n_pairs": float(subset["pair_id"].nunique()),
    }


def run_statistical_tests(
    df_trials: pd.DataFrame,
    num_boot: int,
    seed: int,
    freeze_configs: Mapping[str, Sequence[str]],
) -> Tuple[pd.DataFrame, str]:
    pairwide = build_pairwide_frame(df_trials)
    comparison_pairs = [
        ("full_dynamic", "freeze_L1"),
        ("freeze_L1", "freeze_L1_L2"),
        ("freeze_L1_L2", "full_frozen"),
    ]
    metric_specs = [
        ("BMI", _metric_bmi),
        ("rescue_diff", _metric_rescue_diff),
        ("bias_diff", _metric_bias_diff),
        ("overlap_harm", _metric_overlap_harm),
    ]
    rows: List[Dict[str, int | float | str]] = []
    delays = sorted(pd.unique(pairwide["delay_ms"]).tolist())
    for delay_ms in delays:
        for condition_a, condition_b in comparison_pairs:
            if condition_a not in freeze_configs or condition_b not in freeze_configs:
                continue
            subset_a = pairwide[
                (pairwide["freeze_condition"] == str(condition_a)) & (pairwide["delay_ms"] == int(delay_ms))
            ].copy()
            subset_b = pairwide[
                (pairwide["freeze_condition"] == str(condition_b)) & (pairwide["delay_ms"] == int(delay_ms))
            ].copy()
            merged = subset_a.merge(
                subset_b,
                on=["pair_id", "probe_id", "probe_label"],
                suffixes=("_a", "_b"),
                how="inner",
            )
            if merged.empty:
                continue
            metric_cols = [
                "baseline_is_correct",
                "diagnostic_is_correct",
                "diagnostic_error_rate",
                "diagnostic_sample_capture",
                "nondiagnostic_is_correct",
                "nondiagnostic_error_rate",
                "nondiagnostic_sample_capture",
            ]
            arrays_a = {col: merged[f"{col}_a"].to_numpy(dtype=np.float64) for col in metric_cols}
            arrays_b = {col: merged[f"{col}_b"].to_numpy(dtype=np.float64) for col in metric_cols}
            for metric_name, metric_fn in metric_specs:
                summary = _bootstrap_metric_difference(
                    arrays_a=arrays_a,
                    arrays_b=arrays_b,
                    metric_fn=metric_fn,
                    n_boot=num_boot,
                    seed=mix_seed(
                        seed,
                        1001,
                        int(delay_ms),
                        METRIC_SEED_TOKENS[metric_name],
                        FREEZE_SEED_TOKENS[condition_a],
                        FREEZE_SEED_TOKENS[condition_b],
                    ),
                )
                rows.append(
                    {
                        "analysis": "freeze_condition_comparison",
                        "metric": str(metric_name),
                        "delay_ms": int(delay_ms),
                        "condition_a": str(condition_a),
                        "condition_b": str(condition_b),
                        "estimate_pp": float(summary["estimate_pp"]),
                        "ci_low_pp": float(summary["ci_low_pp"]),
                        "ci_high_pp": float(summary["ci_high_pp"]),
                        "p_two_sided": float(summary["p_two_sided"]),
                        "n_pairs": int(summary["n_pairs"]),
                    }
                )

    dynamic_like = [condition for condition in ("full_dynamic", "freeze_L1", "freeze_L1_L2") if condition in freeze_configs]
    for condition in dynamic_like:
        subset = pairwide[pairwide["freeze_condition"] == str(condition)].copy()
        for metric_name, metric_fn in metric_specs:
            slope_summary = _bootstrap_metric_delay_slope(
                subset=subset,
                metric_fn=metric_fn,
                delays=delays,
                n_boot=num_boot,
                seed=mix_seed(seed, 1101, METRIC_SEED_TOKENS[metric_name], FREEZE_SEED_TOKENS[condition]),
            )
            rows.append(
                {
                    "analysis": "delay_trend",
                    "metric": str(metric_name),
                    "delay_ms": -1,
                    "condition_a": str(condition),
                    "condition_b": "delay_slope",
                    "estimate_pp": float(slope_summary["estimate_pp"]),
                    "ci_low_pp": float(slope_summary["ci_low_pp"]),
                    "ci_high_pp": float(slope_summary["ci_high_pp"]),
                    "p_two_sided": float("nan"),
                    "n_pairs": int(slope_summary["n_pairs"]),
                }
            )

    df_stats = pd.DataFrame(rows).sort_values(
        ["analysis", "metric", "delay_ms", "condition_a", "condition_b"],
        kind="stable",
    ).reset_index(drop=True)

    text_lines: List[str] = []
    text_lines.append("Layerwise UX Freeze Experiment Statistics")
    text_lines.append("single_seed_run=true")
    text_lines.append("")
    text_lines.append("1. Freeze-condition comparisons")
    subset_comparisons = df_stats[df_stats["analysis"] == "freeze_condition_comparison"]
    for row in subset_comparisons.itertuples(index=False):
        text_lines.append(
            f"delay={row.delay_ms} ms, metric={row.metric}, {row.condition_a} - {row.condition_b}: "
            f"{row.estimate_pp:.2f} pp [{row.ci_low_pp:.2f}, {row.ci_high_pp:.2f}], p={row.p_two_sided:.4g}"
        )
    text_lines.append("")
    text_lines.append("2. Dynamic-like delay trends")
    subset_trends = df_stats[df_stats["analysis"] == "delay_trend"]
    for row in subset_trends.itertuples(index=False):
        text_lines.append(
            f"condition={row.condition_a}, metric={row.metric}, slope={row.estimate_pp:.4f} "
            f"[{row.ci_low_pp:.4f}, {row.ci_high_pp:.4f}]"
        )
    return df_stats, "\n".join(text_lines) + "\n"


def make_figure_accuracy_by_freeze_condition(
    df_summary: pd.DataFrame,
    freeze_order: Sequence[str],
) -> plt.Figure:
    apply_publication_style()
    delays = sorted(pd.unique(df_summary["delay_ms"]).tolist())
    fig, axes = plt.subplots(1, len(delays), figsize=(4.2 * len(delays), 4.8), sharey=True)
    axes_arr = np.atleast_1d(axes)
    x = np.arange(len(freeze_order), dtype=np.float64)
    width = 0.24
    offsets = {"diagnostic_overlap": -width, "baseline": 0.0, "nondiagnostic_overlap": width}
    for ax, delay_ms in zip(axes_arr, delays):
        panel = df_summary[df_summary["delay_ms"] == int(delay_ms)].copy()
        for sample_type in SAMPLE_TYPES:
            sub = panel[panel["sample_type"] == sample_type].set_index("freeze_condition").reindex(freeze_order).reset_index()
            y = sub["accuracy"].to_numpy(dtype=np.float64)
            lo = sub["ci_low"].to_numpy(dtype=np.float64)
            hi = sub["ci_high"].to_numpy(dtype=np.float64)
            ax.bar(
                x + offsets[sample_type],
                y,
                width=width,
                color=SAMPLE_TYPE_COLORS[sample_type],
                label=sample_type,
                alpha=ALPHA_BAR,
            )
            ax.errorbar(
                x + offsets[sample_type],
                y,
                yerr=_errorbar_from_ci(y, lo, hi),
                fmt="none",
                ecolor="black",
                linewidth=LINE_WIDTH_REFERENCE,
            )
        ax.set_title(f"Delay = {delay_ms} ms")
        ax.set_xticks(x)
        ax.set_xticklabels(freeze_order, rotation=20, ha="right")
        ax.set_ylim(0.0, 100.0)
        ax.set_xlabel("Freeze condition")
        ax.grid(alpha=GRID_ALPHA, axis="y")
    axes_arr[0].set_ylabel("Accuracy (%)")
    apply_standard_legend(axes_arr[0])
    fig.tight_layout()
    return fig


def make_figure_beneficial_memory_index(
    df_bmi: pd.DataFrame,
    freeze_order: Sequence[str],
) -> plt.Figure:
    apply_publication_style()
    delays = sorted(pd.unique(df_bmi["delay_ms"]).tolist())
    fig, axes = plt.subplots(1, len(delays), figsize=(4.2 * len(delays), 4.8), sharey=True)
    axes_arr = np.atleast_1d(axes)
    x = np.arange(len(freeze_order), dtype=np.float64)
    for ax, delay_ms in zip(axes_arr, delays):
        sub = df_bmi[df_bmi["delay_ms"] == int(delay_ms)].set_index("freeze_condition").reindex(freeze_order).reset_index()
        y = sub["beneficial_memory_index"].to_numpy(dtype=np.float64)
        lo = sub["ci_low"].to_numpy(dtype=np.float64)
        hi = sub["ci_high"].to_numpy(dtype=np.float64)
        colors = [FREEZE_CONDITION_COLORS.get(condition, COLOR_STATIC) for condition in freeze_order]
        ax.bar(x, y, color=colors, alpha=ALPHA_BAR)
        ax.errorbar(x, y, yerr=_errorbar_from_ci(y, lo, hi), fmt="none", ecolor="black", linewidth=LINE_WIDTH_REFERENCE)
        ax.axhline(0.0, color="black", linewidth=LINE_WIDTH_REFERENCE, linestyle="--")
        ax.set_title(f"Delay = {delay_ms} ms")
        ax.set_xticks(x)
        ax.set_xticklabels(freeze_order, rotation=20, ha="right")
        ax.set_xlabel("Freeze condition")
        ax.grid(alpha=GRID_ALPHA, axis="y")
    axes_arr[0].set_ylabel("BMI (pp)")
    fig.tight_layout()
    return fig


def make_figure_rescue_vs_bias(
    df_rescue_bias: pd.DataFrame,
    freeze_order: Sequence[str],
) -> plt.Figure:
    apply_publication_style()
    delays = sorted(pd.unique(df_rescue_bias["delay_ms"]).tolist())
    fig, axes = plt.subplots(len(delays), 2, figsize=(11.5, 3.6 * len(delays)), sharex=False)
    axes_arr = np.atleast_2d(axes)
    x = np.arange(len(freeze_order), dtype=np.float64)
    for row_idx, delay_ms in enumerate(delays):
        sub = df_rescue_bias[df_rescue_bias["delay_ms"] == int(delay_ms)].set_index("freeze_condition").reindex(freeze_order).reset_index()
        rescue_ax = axes_arr[row_idx, 0]
        bias_ax = axes_arr[row_idx, 1]

        rescue_diag = sub["rescue_rate_diagnostic"].to_numpy(dtype=np.float64)
        rescue_nond = sub["rescue_rate_nondiagnostic"].to_numpy(dtype=np.float64)
        rescue_ax.plot(x, rescue_diag, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=SAMPLE_TYPE_COLORS["diagnostic_overlap"], label="diagnostic")
        rescue_ax.plot(x, rescue_nond, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=SAMPLE_TYPE_COLORS["nondiagnostic_overlap"], label="nondiagnostic")
        rescue_ax.errorbar(
            x,
            rescue_diag,
            yerr=_errorbar_from_ci(
                rescue_diag,
                sub["rescue_rate_diagnostic_ci_low"].to_numpy(dtype=np.float64),
                sub["rescue_rate_diagnostic_ci_high"].to_numpy(dtype=np.float64),
            ),
            fmt="none",
            ecolor=SAMPLE_TYPE_COLORS["diagnostic_overlap"],
            linewidth=LINE_WIDTH_REFERENCE,
        )
        rescue_ax.errorbar(
            x,
            rescue_nond,
            yerr=_errorbar_from_ci(
                rescue_nond,
                sub["rescue_rate_nondiagnostic_ci_low"].to_numpy(dtype=np.float64),
                sub["rescue_rate_nondiagnostic_ci_high"].to_numpy(dtype=np.float64),
            ),
            fmt="none",
            ecolor=SAMPLE_TYPE_COLORS["nondiagnostic_overlap"],
            linewidth=LINE_WIDTH_REFERENCE,
        )

        bias_diag = sub["misleading_bias_diagnostic"].to_numpy(dtype=np.float64)
        bias_nond = sub["misleading_bias_nondiagnostic"].to_numpy(dtype=np.float64)
        bias_ax.plot(x, bias_diag, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=SAMPLE_TYPE_COLORS["diagnostic_overlap"], label="diagnostic")
        bias_ax.plot(x, bias_nond, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=SAMPLE_TYPE_COLORS["nondiagnostic_overlap"], label="nondiagnostic")
        bias_ax.errorbar(
            x,
            bias_diag,
            yerr=_errorbar_from_ci(
                bias_diag,
                sub["misleading_bias_diagnostic_ci_low"].to_numpy(dtype=np.float64),
                sub["misleading_bias_diagnostic_ci_high"].to_numpy(dtype=np.float64),
            ),
            fmt="none",
            ecolor=SAMPLE_TYPE_COLORS["diagnostic_overlap"],
            linewidth=LINE_WIDTH_REFERENCE,
        )
        bias_ax.errorbar(
            x,
            bias_nond,
            yerr=_errorbar_from_ci(
                bias_nond,
                sub["misleading_bias_nondiagnostic_ci_low"].to_numpy(dtype=np.float64),
                sub["misleading_bias_nondiagnostic_ci_high"].to_numpy(dtype=np.float64),
            ),
            fmt="none",
            ecolor=SAMPLE_TYPE_COLORS["nondiagnostic_overlap"],
            linewidth=LINE_WIDTH_REFERENCE,
        )

        rescue_ax.set_title(f"Delay = {delay_ms} ms | Rescue")
        bias_ax.set_title(f"Delay = {delay_ms} ms | Misleading bias")
        rescue_ax.set_ylabel("Rate (%)")
        bias_ax.set_ylabel("Rate (%)")
        for ax in (rescue_ax, bias_ax):
            ax.set_xticks(x)
            ax.set_xticklabels(freeze_order, rotation=20, ha="right")
            ax.set_xlabel("Freeze condition")
            ax.grid(alpha=GRID_ALPHA, axis="y")
            apply_standard_legend(ax)
    fig.tight_layout()
    return fig


def make_figure_overlap_harm(
    df_overlap_harm: pd.DataFrame,
    df_rescue_bias: pd.DataFrame,
    freeze_order: Sequence[str],
) -> plt.Figure:
    apply_publication_style()
    delays = sorted(pd.unique(df_overlap_harm["delay_ms"]).tolist())
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL, sharex=False)
    x = np.arange(len(freeze_order), dtype=np.float64)
    for delay_idx, delay_ms in enumerate(delays):
        color = plt.get_cmap("tab10")(delay_idx)
        sub_harm = df_overlap_harm[df_overlap_harm["delay_ms"] == int(delay_ms)].set_index("freeze_condition").reindex(freeze_order).reset_index()
        y_harm = sub_harm["overlap_harm"].to_numpy(dtype=np.float64)
        axes[0].plot(x, y_harm, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=color, label=f"{delay_ms} ms")
        axes[0].errorbar(
            x,
            y_harm,
            yerr=_errorbar_from_ci(
                y_harm,
                sub_harm["ci_low"].to_numpy(dtype=np.float64),
                sub_harm["ci_high"].to_numpy(dtype=np.float64),
            ),
            fmt="none",
            ecolor=color,
            linewidth=LINE_WIDTH_REFERENCE,
        )

        sub_bias = df_rescue_bias[df_rescue_bias["delay_ms"] == int(delay_ms)].set_index("freeze_condition").reindex(freeze_order).reset_index()
        y_bias = sub_bias["bias_diff"].to_numpy(dtype=np.float64)
        axes[1].plot(x, y_bias, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=color, label=f"{delay_ms} ms")
        axes[1].errorbar(
            x,
            y_bias,
            yerr=_errorbar_from_ci(
                y_bias,
                sub_bias["bias_diff_ci_low"].to_numpy(dtype=np.float64),
                sub_bias["bias_diff_ci_high"].to_numpy(dtype=np.float64),
            ),
            fmt="none",
            ecolor=color,
            linewidth=LINE_WIDTH_REFERENCE,
        )

    axes[0].axhline(0.0, color="black", linewidth=LINE_WIDTH_REFERENCE, linestyle="--")
    axes[1].axhline(0.0, color="black", linewidth=LINE_WIDTH_REFERENCE, linestyle="--")
    axes[0].set_title("Overlap harm")
    axes[1].set_title("Bias diff")
    axes[0].set_ylabel("pp")
    axes[1].set_ylabel("pp")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(freeze_order, rotation=20, ha="right")
        ax.set_xlabel("Freeze condition")
        ax.grid(alpha=GRID_ALPHA, axis="y")
        apply_standard_legend(ax)
    fig.tight_layout()
    return fig


def make_figure_layer_state_summary(
    df_layer_state: pd.DataFrame,
    df_bmi: pd.DataFrame,
    freeze_order: Sequence[str],
) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL)
    group_df = df_layer_state[
        (df_layer_state["summary_type"] == "group_mean") & (df_layer_state["sample_type"] == "all_samples")
    ].copy()
    avg_df = (
        group_df.groupby("freeze_condition", sort=False)[[f"mean_ux_{layer}" for layer in LAYER_KEYS]]
        .mean()
        .reindex(freeze_order)
        .reset_index()
    )
    x = np.arange(len(freeze_order), dtype=np.float64)
    for layer_key in LAYER_KEYS:
        axes[0].plot(
            x,
            avg_df[f"mean_ux_{layer_key}"].to_numpy(dtype=np.float64),
            marker=MARKER_CIRCLE,
            linewidth=LINE_WIDTH_PRIMARY,
            color=LAYER_COLORS[layer_key],
            label=layer_key,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(freeze_order, rotation=20, ha="right")
    axes[0].set_title("Retained mean u*x by layer")
    axes[0].set_ylabel("Mean u*x")
    axes[0].set_xlabel("Freeze condition")
    axes[0].grid(alpha=GRID_ALPHA, axis="y")
    apply_standard_legend(axes[0])

    merged = group_df.merge(
        df_bmi[["freeze_condition", "delay_ms", "beneficial_memory_index"]],
        on=["freeze_condition", "delay_ms"],
        how="left",
    )
    for layer_key in LAYER_KEYS:
        x_vals = merged[f"mean_ux_{layer_key}"].to_numpy(dtype=np.float64)
        y_vals = merged["beneficial_memory_index"].to_numpy(dtype=np.float64)
        axes[1].scatter(x_vals, y_vals, s=30, alpha=ALPHA_SCATTER, color=LAYER_COLORS[layer_key], label=layer_key)
    axes[1].set_title("Retained mean u*x vs BMI")
    axes[1].set_xlabel("Mean u*x")
    axes[1].set_ylabel("BMI (pp)")
    axes[1].grid(alpha=GRID_ALPHA)
    apply_standard_legend(axes[1])
    fig.tight_layout()
    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layerwise UX freeze diagnostic-overlap experiment.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--save-dir", type=str, default="results/layerwise_ux_freeze_experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--baseline-batch-size", type=int, default=128)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--delay-ms-list", type=str, default="300,600,1000")
    parser.add_argument("--trial-count", type=int, default=200)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--cache-diagnostic-regions", action="store_true")
    parser.add_argument("--probe-pool-limit", type=int, default=2000)
    parser.add_argument("--probe-pool-per-class", type=int, default=200)
    parser.add_argument("--num-boot", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--freeze-config-set", type=str, default="main4", choices=["main4", "extended"])
    parser.add_argument("--baseline-early-stop-multiplier", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.baseline_batch_size <= 0:
        raise ValueError("--baseline-batch-size must be positive.")
    if args.sample_ms <= 0 or args.probe_ms <= 0:
        raise ValueError("--sample-ms and --probe-ms must be positive.")
    if args.trial_count <= 0:
        raise ValueError("--trial-count must be positive.")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if args.probe_pool_limit <= 0:
        raise ValueError("--probe-pool-limit must be positive.")
    if args.probe_pool_per_class <= 0:
        raise ValueError("--probe-pool-per-class must be positive.")
    if args.num_boot <= 0:
        raise ValueError("--num-boot must be positive.")
    if args.baseline_early_stop_multiplier <= 0:
        raise ValueError("--baseline-early-stop-multiplier must be positive.")

    freeze_configs = resolve_freeze_configs(args.freeze_config_set)
    freeze_order = list(freeze_configs.keys())
    delay_values_ms = parse_delay_list(args.delay_ms_list)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=1.0 * ms, sample_ms=args.sample_ms, probe_ms=args.probe_ms)
    if spec.sample_steps <= 0 or spec.probe_steps <= 0:
        raise ValueError("sample/probe duration must resolve to positive step counts.")

    layout = prepare_result_layout(args.save_dir)
    result_root = layout.root
    save_dir = layout.data_dir
    figure_dir = layout.figure_dir
    log_dir = layout.log_dir

    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(spec.sample_ms, spec.probe_ms),
    )
    _, _, test_loader = build_mnist_skeleton_loader(
        root=args.dataset_root,
        batch_size=1,
        input_size=28,
        num_workers=0,
    )
    dataset = test_loader.dataset
    raw_images, dataset_labels, image_matrix_flat = build_dataset_arrays(dataset)

    probe_region_summary, diagnostic_region_table, mask_lookup = load_or_compute_diagnostic_regions(
        net=net,
        encoder=encoder,
        raw_images=raw_images,
        dataset_labels=dataset_labels,
        spec=spec,
        trial_count=args.trial_count,
        patch_size=args.patch_size,
        delay_values_ms=delay_values_ms,
        batch_size=args.batch_size,
        baseline_batch_size=args.baseline_batch_size,
        device=device,
        seed=args.seed,
        save_dir=save_dir,
        cache_diagnostic_regions=args.cache_diagnostic_regions,
        probe_pool_limit=args.probe_pool_limit,
        probe_pool_per_class=args.probe_pool_per_class,
        early_stop_multiplier=args.baseline_early_stop_multiplier,
    )
    selection_df, df_specs = build_probe_sample_pairs(
        probe_region_summary=probe_region_summary,
        mask_lookup=mask_lookup,
        image_matrix_flat=image_matrix_flat,
        dataset_labels=dataset_labels,
        delay_values_ms=delay_values_ms,
    )
    valid_selection = selection_df[selection_df["selection_status"] == "selected"].copy()
    selection_output = selection_df.merge(
        probe_region_summary,
        on=["probe_id", "probe_label"],
        how="left",
        validate="one_to_one",
    ).sort_values(["probe_id"], kind="stable").reset_index(drop=True)

    df_trials = run_layerwise_freeze_trials(
        net=net,
        encoder=encoder,
        dataset=dataset,
        df_specs=df_specs,
        spec=spec,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        freeze_configs=freeze_configs,
    )
    df_accuracy = compute_accuracy_summary(df_trials=df_trials, num_boot=args.num_boot, seed=args.seed)
    df_bmi, _ = compute_beneficial_memory_index(df_trials=df_trials, num_boot=args.num_boot, seed=args.seed)
    df_rescue_bias, _ = compute_rescue_and_bias_metrics(df_trials=df_trials, num_boot=args.num_boot, seed=args.seed)
    df_overlap_harm, _ = compute_overlap_harm(df_trials=df_trials, num_boot=args.num_boot, seed=args.seed)
    df_layer_state = compute_layer_state_summary(df_trials=df_trials, df_bmi=df_bmi, df_rescue_bias=df_rescue_bias)
    df_stats, stats_text = run_statistical_tests(
        df_trials=df_trials,
        num_boot=args.num_boot,
        seed=args.seed,
        freeze_configs=freeze_configs,
    )

    trial_level_csv = save_tidy_csv(
        df_trials,
        save_dir / "trial_level_results.csv",
        sort_by=["seed", "freeze_condition", "delay_ms", "pair_id", "sample_type"],
    )
    diagnostic_region_csv = save_tidy_csv(
        diagnostic_region_table,
        save_dir / "diagnostic_region_table.csv",
        sort_by=["probe_id", "patch_id"],
    )
    selection_csv = save_tidy_csv(selection_output, save_dir / "probe_anchor_selection.csv", sort_by=["probe_id"])
    accuracy_csv = save_tidy_csv(
        df_accuracy,
        save_dir / "accuracy_summary.csv",
        sort_by=["freeze_condition", "delay_ms", "sample_type"],
    )
    bmi_csv = save_tidy_csv(
        df_bmi,
        save_dir / "beneficial_memory_index_summary.csv",
        sort_by=["freeze_condition", "delay_ms"],
    )
    rescue_bias_csv = save_tidy_csv(
        df_rescue_bias,
        save_dir / "rescue_bias_summary.csv",
        sort_by=["freeze_condition", "delay_ms"],
    )
    overlap_harm_csv = save_tidy_csv(
        df_overlap_harm,
        save_dir / "overlap_harm_summary.csv",
        sort_by=["freeze_condition", "delay_ms"],
    )
    layer_state_csv = save_tidy_csv(
        df_layer_state,
        save_dir / "layer_state_summary.csv",
        sort_by=["summary_type", "freeze_condition", "delay_ms", "sample_type"],
    )
    stats_csv = save_tidy_csv(
        df_stats,
        save_dir / "stats_freeze_condition_comparison.csv",
        sort_by=["analysis", "metric", "delay_ms", "condition_a", "condition_b"],
    )
    stats_txt_path = save_dir / "stats_freeze_condition_comparison.txt"
    stats_txt_path.write_text(stats_text, encoding="utf-8")

    fig1 = make_figure_accuracy_by_freeze_condition(df_accuracy, freeze_order=freeze_order)
    fig1_paths = save_figure_all_formats(fig1, figure_dir / "figure_1_accuracy_by_freeze_condition")
    plt.close(fig1)

    fig2 = make_figure_beneficial_memory_index(df_bmi, freeze_order=freeze_order)
    fig2_paths = save_figure_all_formats(fig2, figure_dir / "figure_2_beneficial_memory_index")
    plt.close(fig2)

    fig3 = make_figure_rescue_vs_bias(df_rescue_bias, freeze_order=freeze_order)
    fig3_paths = save_figure_all_formats(fig3, figure_dir / "figure_3_rescue_vs_bias")
    plt.close(fig3)

    fig4 = make_figure_overlap_harm(df_overlap_harm, df_rescue_bias=df_rescue_bias, freeze_order=freeze_order)
    fig4_paths = save_figure_all_formats(fig4, figure_dir / "figure_4_overlap_harm")
    plt.close(fig4)

    fig5 = make_figure_layer_state_summary(df_layer_state, df_bmi=df_bmi, freeze_order=freeze_order)
    fig5_paths = save_figure_all_formats(fig5, figure_dir / "figure_5_layer_state_summary")
    plt.close(fig5)

    run_config_path = save_run_config(
        {
            "model_path": args.model_path,
            "dataset_root": args.dataset_root,
            "save_dir": str(result_root),
            "seed": int(args.seed),
            "device": str(device),
            "sample_ms": float(args.sample_ms),
            "probe_ms": float(args.probe_ms),
            "delay_ms_list": [int(value) for value in delay_values_ms],
            "trial_count_requested": int(args.trial_count),
            "trial_count_final": int(valid_selection.shape[0]),
            "batch_size": int(args.batch_size),
            "baseline_batch_size": int(args.baseline_batch_size),
            "patch_size": int(args.patch_size),
            "cache_diagnostic_regions": bool(args.cache_diagnostic_regions),
            "probe_pool_limit": int(args.probe_pool_limit),
            "probe_pool_per_class": int(args.probe_pool_per_class),
            "baseline_early_stop_multiplier": float(args.baseline_early_stop_multiplier),
            "num_boot": int(args.num_boot),
            "freeze_config_set": str(args.freeze_config_set),
            "freeze_conditions": {name: list(layers) for name, layers in freeze_configs.items()},
            "freeze_mechanism": "restore selected layers to baseline u=U, x=1 immediately before probe",
            "selection_policy": "reuse diagnostic_feature_overlap_experiment sample pairing",
            "outputs": {
                "trial_level_results": str(trial_level_csv),
                "diagnostic_region_table": str(diagnostic_region_csv),
                "probe_anchor_selection": str(selection_csv),
                "accuracy_summary": str(accuracy_csv),
                "beneficial_memory_index_summary": str(bmi_csv),
                "rescue_bias_summary": str(rescue_bias_csv),
                "overlap_harm_summary": str(overlap_harm_csv),
                "layer_state_summary": str(layer_state_csv),
                "stats_csv": str(stats_csv),
                "stats_txt": str(stats_txt_path),
                "figure_1_png": fig1_paths["png"],
                "figure_1_pdf": fig1_paths["pdf"],
                "figure_2_png": fig2_paths["png"],
                "figure_2_pdf": fig2_paths["pdf"],
                "figure_3_png": fig3_paths["png"],
                "figure_3_pdf": fig3_paths["pdf"],
                "figure_4_png": fig4_paths["png"],
                "figure_4_pdf": fig4_paths["pdf"],
                "figure_5_png": fig5_paths["png"],
                "figure_5_pdf": fig5_paths["pdf"],
            },
        },
        result_root,
    )
    summary_path = save_summary_json(
        {
            "experiment": "layerwise_ux_freeze_experiment",
            "selected_probe_anchors": int(valid_selection.shape[0]),
            "delay_ms_list": [int(value) for value in delay_values_ms],
            "freeze_conditions": {name: list(layers) for name, layers in freeze_configs.items()},
            "run_config_json": str(run_config_path.resolve()),
        },
        result_root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=layerwise_ux_freeze_experiment",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"selected_probe_anchors={int(valid_selection.shape[0])}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_path.resolve()}",
        ],
        log_dir,
    )

    print("\n=== Layerwise UX Freeze Experiment Summary ===")
    print(f"Selected probe anchors: {int(valid_selection.shape[0])}")
    print(f"Saved: {trial_level_csv}")
    print(f"Saved: {accuracy_csv}")
    print(f"Saved: {bmi_csv}")
    print(f"Saved: {rescue_bias_csv}")
    print(f"Saved: {overlap_harm_csv}")
    print(f"Saved: {layer_state_csv}")
    print(f"Saved: {stats_csv}")
    print(f"Saved: {stats_txt_path}")
    print(f"Saved: {fig1_paths['png']}")
    print(f"Saved: {fig2_paths['png']}")
    print(f"Saved: {fig3_paths['png']}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {run_log_path}")
    print(f"Saved: {fig4_paths['png']}")
    print(f"Saved: {fig5_paths['png']}")
    print(f"Saved: {run_config_path}")
