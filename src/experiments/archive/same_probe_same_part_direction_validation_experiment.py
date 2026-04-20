from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.common.dataset import build_class_index
from src.experiments.common.json_io import save_json_payload as _save_json
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.seed import mix_seed
from src.experiments.common.voltage_readout import resolve_readout_step
from src.experiments.distractor.shared.analysis import (
    _normalize_projection_for_plot,
    _prepare_single_source_spike_batch,
    _project_vectors_to_2d,
    compute_delta_v,
    compute_spearman_summary,
    run_single_source_preceding_item_task,
)
from src.experiments.distractor.shared.masking import build_overlap_masks_for_pair
from src.experiments.distractor.shared.pair_sampling import (
    assign_bins_from_values as _assign_bins_from_values,
    build_dataset_arrays,
    select_probe_ids_balanced,
    select_probe_samples_from_candidates as _select_probe_samples_from_candidates,
)
from src.plotting.common.io import (
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
)
from src.plotting.common.theme_tokens import (
    ALPHA_BAR_SOFT,
    ALPHA_MASK_OVERLAY,
    ALPHA_SCATTER,
    CMAP_IMAGE_GRAY,
    CMAP_MASK_OVERLAY,
    COLOR_ACCENT_BLUE,
    COLOR_ACCENT_RED,
    COLOR_GUIDE_LIGHT,
    COLOR_NEUTRAL_LIGHT,
    COLOR_NEUTRAL_MID,
    FIGSIZE_REPRESENTATIVE_GRID,
    FIGSIZE_TWO_PANEL_TALL,
    GRID_ALPHA,
    LINE_WIDTH_GUIDE,
    LINE_WIDTH_PRIMARY,
    PART_SIMILARITY_BIN_COLORS,
    PAIR_HIGHLIGHT_COLORS,
    apply_standard_legend,
)

DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_OUTPUT_DIR = "results/same_probe_same_part_direction_validation_experiment"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_DELAY2_MS = 400.0
DEFAULT_PROBE_MS = 100.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_PROBES = 50
DEFAULT_SAMPLES_PER_PROBE = 24
DEFAULT_NUM_SIM_BINS = 5
DEFAULT_FOREGROUND_THRESHOLD = 0.0
DEFAULT_DILATION_RADIUS = 1
DEFAULT_NUM_CONTROL_CANDIDATES = 32
DEFAULT_SAVE_CASE_COUNT = 4
EPS = 1e-12
PART_BIN_ORDER: tuple[str, ...] = ("different-part", "middle-part", "same-part")
PAIR_OF_PAIRS_COLUMNS: tuple[str, ...] = (
    "probe_id",
    "probe_label",
    "pair_i_id",
    "pair_j_id",
    "pair_i_key",
    "pair_j_key",
    "sample_i_id",
    "sample_j_id",
    "sample_i_label",
    "sample_j_label",
    "pair_i_status",
    "pair_j_status",
    "part_similarity",
    "part_similarity_binary_cosine",
    "part_similarity_iou",
    "direction_similarity",
    "direction_angle_deg",
    "direction_distance_l2",
    "delta_argmax_match",
    "pair_pair_status",
)


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay2_ms: float
    probe_ms: float
    phase_reset: bool = True

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def delay2_steps(self) -> int:
        return int(round((self.delay2_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


def _load_dataset(dataset_root: str, split: str):
    train_loader, _, test_loader = build_mnist_skeleton_loader(
        root=dataset_root,
        batch_size=1,
        input_size=28,
        num_workers=0,
    )
    split_name = str(split).strip().lower()
    if split_name == "train":
        return train_loader.dataset
    if split_name == "test":
        return test_loader.dataset
    raise ValueError(f"Unsupported split: {split}")


def _validate_positive(name: str, value: int | float, *, allow_zero: bool = False) -> None:
    if allow_zero:
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        return
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive.")


def safe_cosine(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor, eps: float = EPS) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    norm_a = float(np.linalg.norm(aa))
    norm_b = float(np.linalg.norm(bb))
    if norm_a <= float(eps) or norm_b <= float(eps):
        return float("nan")
    return float(np.dot(aa, bb) / (norm_a * norm_b))


def safe_angle_deg(cosine_value: float) -> float:
    if not np.isfinite(float(cosine_value)):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(cosine_value), -1.0, 1.0))))


def _sem(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _iou_from_binary_signatures(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1) > 0.0
    bb = np.asarray(b, dtype=np.float64).reshape(-1) > 0.0
    union = aa | bb
    if int(union.sum()) <= 0:
        return float("nan")
    return float((aa & bb).sum() / union.sum())


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 2:
        return {"status": "insufficient_samples"}
    if float(np.std(xx)) <= EPS:
        return {"status": "zero_variance"}
    slope, intercept = np.polyfit(xx, yy, deg=1)
    x_fit = np.linspace(float(xx.min()), float(xx.max()), num=100)
    return {
        "status": "ok",
        "slope": float(slope),
        "intercept": float(intercept),
        "x_fit": x_fit,
        "y_fit": slope * x_fit + intercept,
    }


def _assign_tertile_labels_by_rank(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 0:
        return np.asarray([], dtype=object)
    order = np.argsort(arr, kind="stable")
    label_positions = np.linspace(0, len(PART_BIN_ORDER) - 1, num=arr.size)
    assigned = np.empty(arr.size, dtype=object)
    for rank_pos, arr_idx in enumerate(order.tolist()):
        label_idx = int(round(float(label_positions[rank_pos])))
        assigned[int(arr_idx)] = PART_BIN_ORDER[label_idx]
    return assigned


def build_same_probe_dms_pairs(
    images: torch.Tensor,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    *,
    max_probes: int,
    samples_per_probe: int,
    num_bins: int,
    seed: int,
) -> pd.DataFrame:
    del images
    probe_ids = select_probe_ids_balanced(class_index=class_index, max_probes=max_probes, seed=mix_seed(seed, 31))
    all_ids = np.arange(len(labels), dtype=np.int64)
    rows: list[dict[str, object]] = []
    for probe_rank, probe_id in enumerate(probe_ids):
        probe_id_int = int(probe_id)
        probe_label = int(labels[probe_id_int])
        sims_to_probe = flat_normalized @ flat_normalized[probe_id_int]
        mask = (all_ids != probe_id_int) & (labels[all_ids] != probe_label)
        candidate_ids = all_ids[mask]
        if candidate_ids.size <= 0:
            continue
        df_candidates = pd.DataFrame(
            {
                "sample_id": candidate_ids.astype(np.int64, copy=False),
                "sample_label": labels[candidate_ids].astype(np.int64, copy=False),
                "probe_id": probe_id_int,
                "probe_label": probe_label,
                "probe_rank": int(probe_rank),
                "similarity_public_or_initial": sims_to_probe[candidate_ids].astype(np.float64, copy=False),
            }
        )
        selected_samples = _select_probe_samples_from_candidates(
            df_candidates=df_candidates,
            samples_per_probe=int(samples_per_probe),
            num_bins=int(num_bins),
        )
        if selected_samples.empty:
            continue
        selected_samples = selected_samples.rename(columns={"similarity_public_or_initial": "sp_similarity"}).reset_index(drop=True)
        selected_samples["sp_bin"] = _assign_bins_from_values(
            selected_samples["sp_similarity"].to_numpy(dtype=np.float64, copy=False),
            num_bins=int(num_bins),
        )
        sp_label_order = {label: idx for idx, label in enumerate(pd.unique(selected_samples["sp_bin"]).tolist())}
        selected_samples["sp_bin_index"] = selected_samples["sp_bin"].map(sp_label_order).astype(np.int64)
        selected_samples = selected_samples.sort_values(
            ["probe_rank", "sp_bin_index", "sp_similarity", "sample_id"],
            kind="stable",
        ).reset_index(drop=True)
        n_pairs_for_probe = int(len(selected_samples))
        for row in selected_samples.itertuples(index=False):
            rows.append(
                {
                    "probe_id": probe_id_int,
                    "probe_label": probe_label,
                    "probe_rank": int(probe_rank),
                    "sample_id": int(row.sample_id),
                    "sample_label": int(row.sample_label),
                    "sp_similarity": float(row.sp_similarity),
                    "sp_bin": str(row.sp_bin),
                    "sp_bin_index": int(row.sp_bin_index),
                    "n_pairs_for_probe": int(n_pairs_for_probe),
                }
            )
    if not rows:
        raise RuntimeError("No same-probe sample/probe pairs were generated.")
    df_pairs = pd.DataFrame(rows).drop_duplicates(subset=["probe_id", "sample_id"], keep="first").reset_index(drop=True)
    df_pairs = df_pairs.sort_values(
        ["probe_rank", "sp_bin_index", "sp_similarity", "sample_id", "probe_id"],
        kind="stable",
    ).reset_index(drop=True)
    df_pairs["pair_id"] = np.arange(len(df_pairs), dtype=np.int64)
    df_pairs["pair_key"] = df_pairs.apply(lambda row: f"{int(row.probe_id)}::{int(row.sample_id)}", axis=1)
    return df_pairs[
        [
            "pair_id",
            "pair_key",
            "probe_id",
            "probe_label",
            "probe_rank",
            "sample_id",
            "sample_label",
            "sp_similarity",
            "sp_bin",
            "sp_bin_index",
            "n_pairs_for_probe",
        ]
    ].copy()


def build_probe_part_signature_for_pair(
    sample_image: torch.Tensor,
    probe_image: torch.Tensor,
    *,
    foreground_threshold: float,
    use_dilated_overlap: bool,
    dilation_radius: int,
    seed: int,
    num_control_candidates: int,
) -> dict[str, object]:
    overlap_bundle = build_overlap_masks_for_pair(
        sample_image=sample_image,
        probe_image=probe_image,
        foreground_threshold=float(foreground_threshold),
        use_dilated_overlap=bool(use_dilated_overlap),
        dilation_radius=int(dilation_radius),
        seed=int(seed),
        num_control_candidates=int(num_control_candidates),
    )
    probe_overlap_mask = np.asarray(overlap_bundle.probe_overlap_mask, dtype=bool)
    binary_signature = probe_overlap_mask.astype(np.float64, copy=False).reshape(-1)
    probe_abs = probe_image.detach().cpu().to(torch.float32).abs().amax(dim=0).numpy().astype(np.float64, copy=False)
    weighted_signature = (probe_abs * probe_overlap_mask.astype(np.float64, copy=False)).reshape(-1)
    probe_part_area = int(probe_overlap_mask.sum())
    probe_part_energy = float(probe_abs[probe_overlap_mask].sum()) if probe_part_area > 0 else 0.0
    part_status = "ok" if probe_part_area > 0 else "empty_probe_overlap"
    metadata = dict(overlap_bundle.metadata)
    metadata.update(
        {
            "probe_part_area": int(probe_part_area),
            "probe_part_energy": float(probe_part_energy),
            "part_signature_binary_norm": float(np.linalg.norm(binary_signature)),
            "part_signature_weighted_norm": float(np.linalg.norm(weighted_signature)),
            "part_status": str(part_status),
        }
    )
    return {
        "probe_overlap_mask": probe_overlap_mask,
        "part_signature_binary": np.asarray(binary_signature, dtype=np.float64),
        "part_signature_weighted": np.asarray(weighted_signature, dtype=np.float64),
        "metadata": metadata,
    }


def compute_pair_delta_vectors(
    *,
    net,
    images: torch.Tensor,
    encoder,
    device: torch.device,
    readout_step: int,
    spec: ExperimentSpec,
    pair_df: pd.DataFrame,
    batch_size: int,
    eps: float = EPS,
) -> dict[str, object]:
    df_pairs = pair_df.copy().reset_index(drop=True)
    dynamic_voltage_map: dict[int, np.ndarray] = {}
    static_voltage_map: dict[int, np.ndarray] = {}
    delta_vector_map: dict[int, np.ndarray] = {}
    pair_status_map: dict[int, str] = {}
    delta_norm_map: dict[int, float] = {}

    for batch_start in range(0, len(df_pairs), int(batch_size)):
        batch = df_pairs.iloc[batch_start : batch_start + int(batch_size)].copy().reset_index(drop=True)
        batch_for_rollout = batch.rename(columns={"sample_id": "preceding_id", "sample_label": "preceding_label"})
        preceding_spikes, probe_spikes = _prepare_single_source_spike_batch(
            images=images,
            batch_df=batch_for_rollout,
            encoder=encoder,
            preceding_steps=int(spec.sample_steps),
            probe_steps=int(spec.probe_steps),
            device=device,
        )
        dynamic = run_single_source_preceding_item_task(
            net=net,
            preceding_spikes=preceding_spikes,
            probe_spikes=probe_spikes,
            gap_steps=int(spec.delay2_steps),
            stsp_mode="dynamic",
            readout_step=int(readout_step),
            phase_reset=spec.phase_reset,
        )
        static = run_single_source_preceding_item_task(
            net=net,
            preceding_spikes=preceding_spikes,
            probe_spikes=probe_spikes,
            gap_steps=int(spec.delay2_steps),
            stsp_mode="static_frozen",
            readout_step=int(readout_step),
            phase_reset=spec.phase_reset,
        )
        dynamic_grouped = np.asarray(dynamic["grouped_voltage"], dtype=np.float64)
        static_grouped = np.asarray(static["grouped_voltage"], dtype=np.float64)
        delta_v = np.asarray(compute_delta_v(dynamic_grouped, static_grouped), dtype=np.float64)
        for batch_idx, batch_row in enumerate(batch.itertuples(index=False)):
            pair_id = int(batch_row.pair_id)
            dynamic_vec = np.asarray(dynamic_grouped[batch_idx], dtype=np.float64).reshape(-1)
            static_vec = np.asarray(static_grouped[batch_idx], dtype=np.float64).reshape(-1)
            delta_vec = np.asarray(delta_v[batch_idx], dtype=np.float64).reshape(-1)
            dynamic_voltage_map[pair_id] = dynamic_vec
            static_voltage_map[pair_id] = static_vec
            delta_vector_map[pair_id] = delta_vec
            delta_norm = float(np.linalg.norm(delta_vec))
            delta_norm_map[pair_id] = delta_norm
            part_status = str(getattr(batch_row, "part_status", "ok"))
            if part_status != "ok":
                status = part_status
            elif not np.isfinite(delta_norm):
                status = "invalid_numeric"
            elif delta_norm <= float(eps):
                status = "zero_delta_norm"
            else:
                status = "ok"
            pair_status_map[pair_id] = str(status)

    df_pairs["grouped_voltage_dynamic"] = df_pairs["pair_id"].map(dynamic_voltage_map)
    df_pairs["grouped_voltage_static"] = df_pairs["pair_id"].map(static_voltage_map)
    df_pairs["DeltaV_pair"] = df_pairs["pair_id"].map(delta_vector_map)
    df_pairs["delta_norm"] = df_pairs["pair_id"].map(delta_norm_map).astype(np.float64)
    df_pairs["pair_status"] = df_pairs["pair_id"].map(pair_status_map).astype(str)
    df_pairs["delta_valid"] = df_pairs["pair_status"].eq("ok").astype(np.int64)
    return {
        "pair_df": df_pairs,
        "dynamic_voltage_map": dynamic_voltage_map,
        "static_voltage_map": static_voltage_map,
        "delta_vector_map": delta_vector_map,
    }


def build_same_probe_pair_of_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for probe_id, subset in pair_df.groupby("probe_id", sort=True):
        probe_rows = subset.sort_values(["pair_id"], kind="stable").reset_index(drop=True)
        for idx_i in range(len(probe_rows)):
            row_i = probe_rows.iloc[idx_i]
            for idx_j in range(idx_i + 1, len(probe_rows)):
                row_j = probe_rows.iloc[idx_j]
                pair_status_i = str(row_i["pair_status"])
                pair_status_j = str(row_j["pair_status"])
                weighted_i = np.asarray(row_i["part_signature_weighted"], dtype=np.float64)
                weighted_j = np.asarray(row_j["part_signature_weighted"], dtype=np.float64)
                binary_i = np.asarray(row_i["part_signature_binary"], dtype=np.float64)
                binary_j = np.asarray(row_j["part_signature_binary"], dtype=np.float64)
                delta_i = np.asarray(row_i["DeltaV_pair"], dtype=np.float64)
                delta_j = np.asarray(row_j["DeltaV_pair"], dtype=np.float64)
                part_similarity = safe_cosine(weighted_i, weighted_j)
                part_similarity_binary_cosine = safe_cosine(binary_i, binary_j)
                part_similarity_iou = _iou_from_binary_signatures(binary_i, binary_j)
                direction_similarity = safe_cosine(delta_i, delta_j)
                direction_angle_deg = safe_angle_deg(direction_similarity)
                direction_distance_l2 = float(np.linalg.norm(delta_i - delta_j))
                delta_argmax_match = float(int(np.argmax(delta_i) == np.argmax(delta_j)))
                if pair_status_i != "ok" or pair_status_j != "ok":
                    pair_pair_status = "upstream_invalid_pair"
                elif not np.isfinite(part_similarity) or not np.isfinite(direction_similarity):
                    pair_pair_status = "invalid_similarity"
                else:
                    pair_pair_status = "ok"
                rows.append(
                    {
                        "probe_id": int(probe_id),
                        "probe_label": int(row_i["probe_label"]),
                        "pair_i_id": int(row_i["pair_id"]),
                        "pair_j_id": int(row_j["pair_id"]),
                        "pair_i_key": str(row_i["pair_key"]),
                        "pair_j_key": str(row_j["pair_key"]),
                        "sample_i_id": int(row_i["sample_id"]),
                        "sample_j_id": int(row_j["sample_id"]),
                        "sample_i_label": int(row_i["sample_label"]),
                        "sample_j_label": int(row_j["sample_label"]),
                        "pair_i_status": str(pair_status_i),
                        "pair_j_status": str(pair_status_j),
                        "part_similarity": float(part_similarity),
                        "part_similarity_binary_cosine": float(part_similarity_binary_cosine),
                        "part_similarity_iou": float(part_similarity_iou),
                        "direction_similarity": float(direction_similarity),
                        "direction_angle_deg": float(direction_angle_deg),
                        "direction_distance_l2": float(direction_distance_l2),
                        "delta_argmax_match": float(delta_argmax_match),
                        "pair_pair_status": str(pair_pair_status),
                    }
                )
    return pd.DataFrame(rows, columns=list(PAIR_OF_PAIRS_COLUMNS))


def assign_part_similarity_bins(
    pair_of_pairs_df: pd.DataFrame,
    *,
    eps: float = EPS,
) -> dict[str, object]:
    df = pair_of_pairs_df.copy()
    if df.empty:
        df["part_similarity_bin"] = pd.Series(dtype="object")
        df["same_part_bin"] = pd.Series(dtype=np.float64)
        return {
            "pair_of_pairs_df": df,
            "q33": float("nan"),
            "q67": float("nan"),
            "bin_counts": {},
            "invalid_count": 0,
            "warnings": ["no_pair_of_pairs"],
        }

    df["part_similarity_bin"] = "invalid"
    df["same_part_bin"] = np.nan
    valid_mask = df["pair_pair_status"].astype(str).eq("ok") & np.isfinite(df["part_similarity"].to_numpy(dtype=np.float64))
    valid_values = df.loc[valid_mask, "part_similarity"].to_numpy(dtype=np.float64)
    warnings: list[str] = []
    if valid_values.size <= 0:
        q33 = float("nan")
        q67 = float("nan")
        warnings.append("no_valid_part_similarity")
    else:
        q33, q67 = np.quantile(valid_values, [1.0 / 3.0, 2.0 / 3.0])
        q33 = float(q33)
        q67 = float(q67)
        if not np.isfinite(q33) or not np.isfinite(q67) or abs(q67 - q33) <= float(eps):
            warnings.append("collapsed_tertiles_rank_fallback")
            df.loc[valid_mask, "part_similarity_bin"] = _assign_tertile_labels_by_rank(valid_values)
        else:
            values = df["part_similarity"].to_numpy(dtype=np.float64)
            df.loc[valid_mask & (values <= q33), "part_similarity_bin"] = "different-part"
            df.loc[valid_mask & (values > q33) & (values <= q67), "part_similarity_bin"] = "middle-part"
            df.loc[valid_mask & (values > q67), "part_similarity_bin"] = "same-part"

    df.loc[df["part_similarity_bin"] == "different-part", "same_part_bin"] = 0.0
    df.loc[df["part_similarity_bin"] == "same-part", "same_part_bin"] = 1.0
    bin_counts = {bin_name: int((df["part_similarity_bin"] == bin_name).sum()) for bin_name in PART_BIN_ORDER}
    invalid_count = int((df["part_similarity_bin"] == "invalid").sum())
    if any(count < 3 for count in bin_counts.values() if count > 0):
        warnings.append("at_least_one_part_similarity_bin_has_fewer_than_3_pairs")
    return {
        "pair_of_pairs_df": df,
        "q33": float(q33 if "q33" in locals() else float("nan")),
        "q67": float(q67 if "q67" in locals() else float("nan")),
        "bin_counts": bin_counts,
        "invalid_count": int(invalid_count),
        "warnings": warnings,
    }


def summarize_direction_similarity_by_part_bin(pair_of_pairs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bin_name in PART_BIN_ORDER:
        subset = pair_of_pairs_df[pair_of_pairs_df["part_similarity_bin"] == bin_name].copy()
        rows.append(
            {
                "part_similarity_bin": str(bin_name),
                "n_pairs": int(len(subset)),
                "mean_direction_similarity": float(subset["direction_similarity"].mean()) if not subset.empty else float("nan"),
                "sem_direction_similarity": _sem(subset["direction_similarity"].to_numpy(dtype=np.float64)) if not subset.empty else 0.0,
                "mean_direction_angle_deg": float(subset["direction_angle_deg"].mean()) if not subset.empty else float("nan"),
                "sem_direction_angle_deg": _sem(subset["direction_angle_deg"].to_numpy(dtype=np.float64)) if not subset.empty else 0.0,
                "mean_direction_distance_l2": float(subset["direction_distance_l2"].mean()) if not subset.empty else float("nan"),
                "mean_delta_argmax_match": float(subset["delta_argmax_match"].mean()) if not subset.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _select_pair_record_columns(df_pairs: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "pair_id",
        "pair_key",
        "probe_id",
        "probe_label",
        "probe_rank",
        "sample_id",
        "sample_label",
        "sp_similarity",
        "sp_bin",
        "n_pairs_for_probe",
        "delta_norm",
        "delta_valid",
        "pair_status",
        "probe_part_area",
        "probe_part_energy",
        "part_signature_binary_norm",
        "part_signature_weighted_norm",
        "part_status",
        "base_overlap_area",
        "probe_overlap_area",
        "sample_overlap_area",
        "foreground_threshold",
        "use_dilated_overlap",
        "dilation_radius",
    ]
    present = [col for col in keep_cols if col in df_pairs.columns]
    return df_pairs[present].copy()


def _select_pair_of_pairs_columns(df_pair_of_pairs: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "probe_id",
        "probe_label",
        "pair_i_id",
        "pair_j_id",
        "pair_i_key",
        "pair_j_key",
        "sample_i_id",
        "sample_j_id",
        "sample_i_label",
        "sample_j_label",
        "part_similarity",
        "part_similarity_binary_cosine",
        "part_similarity_iou",
        "direction_similarity",
        "direction_angle_deg",
        "direction_distance_l2",
        "delta_argmax_match",
        "part_similarity_bin",
        "same_part_bin",
        "pair_pair_status",
    ]
    present = [col for col in keep_cols if col in df_pair_of_pairs.columns]
    return df_pair_of_pairs[present].copy()


def _save_pair_sidecars(pair_df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    sorted_pairs = pair_df.sort_values(["pair_id"], kind="stable").reset_index(drop=True)
    pair_ids = sorted_pairs["pair_id"].to_numpy(dtype=np.int64, copy=False)
    delta_vectors = np.stack([np.asarray(vec, dtype=np.float64) for vec in sorted_pairs["DeltaV_pair"]], axis=0)
    dynamic_vectors = np.stack([np.asarray(vec, dtype=np.float64) for vec in sorted_pairs["grouped_voltage_dynamic"]], axis=0)
    static_vectors = np.stack([np.asarray(vec, dtype=np.float64) for vec in sorted_pairs["grouped_voltage_static"]], axis=0)
    signature_binary = np.stack([np.asarray(vec, dtype=np.float64) for vec in sorted_pairs["part_signature_binary"]], axis=0)
    signature_weighted = np.stack([np.asarray(vec, dtype=np.float64) for vec in sorted_pairs["part_signature_weighted"]], axis=0)
    probe_overlap_masks = np.stack([np.asarray(mask, dtype=bool) for mask in sorted_pairs["probe_overlap_mask"]], axis=0)

    delta_path = output_dir / "pair_delta_vectors.npz"
    np.savez_compressed(
        delta_path,
        pair_ids=pair_ids,
        delta_vectors=delta_vectors,
        grouped_voltage_dynamic=dynamic_vectors,
        grouped_voltage_static=static_vectors,
    )
    signature_path = output_dir / "pair_part_signatures.npz"
    np.savez_compressed(
        signature_path,
        pair_ids=pair_ids,
        part_signature_binary=signature_binary,
        part_signature_weighted=signature_weighted,
        probe_overlap_masks=probe_overlap_masks,
    )
    return {"pair_delta_vectors": str(delta_path), "pair_part_signatures": str(signature_path)}


def plot_part_similarity_vs_direction_similarity(pair_of_pairs_df: pd.DataFrame) -> tuple[plt.Figure, dict[str, object]]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    valid = pair_of_pairs_df[pair_of_pairs_df["pair_pair_status"] == "ok"].copy()
    x = valid["part_similarity"].to_numpy(dtype=np.float64) if not valid.empty else np.asarray([], dtype=np.float64)
    y = valid["direction_similarity"].to_numpy(dtype=np.float64) if not valid.empty else np.asarray([], dtype=np.float64)
    if x.size <= 0:
        ax.text(0.5, 0.5, "No valid same-probe pair-of-pairs", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig, {"spearman": {"status": "insufficient_samples"}, "linear_fit": {"status": "insufficient_samples"}}

    ax.scatter(x, y, s=22, alpha=0.72, color=COLOR_ACCENT_BLUE)
    fit = _linear_fit(x, y)
    if fit.get("status") == "ok":
        ax.plot(fit["x_fit"], fit["y_fit"], color=COLOR_ACCENT_RED, linewidth=LINE_WIDTH_PRIMARY)
    spearman = compute_spearman_summary(x, y)
    annotation = [
        f"n={int(len(valid))}",
        f"rho={float(spearman['rho']):.3f}" if spearman.get("status") == "ok" else f"rho={spearman.get('status')}",
        f"p={float(spearman['p_value']):.3g}" if spearman.get("status") == "ok" else "",
    ]
    if fit.get("status") == "ok":
        annotation.append(f"slope={float(fit['slope']):.3f}")
    ax.text(
        0.02,
        0.98,
        "\n".join(line for line in annotation if line),
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC"},
    )
    ax.set_xlabel("part similarity")
    ax.set_ylabel("direction similarity")
    ax.set_title("Part similarity vs direction similarity")
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig, {"spearman": spearman, "linear_fit": fit}


def plot_same_part_vs_different_part(
    pair_of_pairs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_TWO_PANEL_TALL)
    color_map = dict(PART_SIMILARITY_BIN_COLORS)
    valid = pair_of_pairs_df[pair_of_pairs_df["pair_pair_status"] == "ok"].copy()
    rng = np.random.default_rng(1234)
    positions = np.arange(len(PART_BIN_ORDER), dtype=np.float64)
    for pos, bin_name in zip(positions, PART_BIN_ORDER):
        subset = valid[valid["part_similarity_bin"] == bin_name]["direction_similarity"].to_numpy(dtype=np.float64)
        if subset.size > 0:
            jitter = rng.normal(0.0, 0.035, size=subset.size)
            ax.scatter(np.full(subset.size, pos, dtype=np.float64) + jitter, subset, s=18, alpha=0.45, color=color_map[bin_name])
        summary_row = summary_df[summary_df["part_similarity_bin"] == bin_name]
        if not summary_row.empty:
            row = summary_row.iloc[0]
            mean_val = float(row["mean_direction_similarity"])
            sem_val = float(row["sem_direction_similarity"])
            if np.isfinite(mean_val):
                ax.bar(pos, mean_val, width=0.58, color=color_map[bin_name], alpha=ALPHA_BAR_SOFT, edgecolor=color_map[bin_name])
                ax.errorbar(pos, mean_val, yerr=sem_val, color=color_map[bin_name], linewidth=LINE_WIDTH_PRIMARY, capsize=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(PART_BIN_ORDER)
    ax.set_ylabel("direction similarity")
    ax.set_title("Direction similarity by probe-part similarity bin")
    ax.grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def _choose_representative_examples(pair_of_pairs_df: pd.DataFrame) -> dict[str, object]:
    valid = pair_of_pairs_df[pair_of_pairs_df["pair_pair_status"] == "ok"].copy()
    if valid.empty:
        return {"probe_id": None, "same_row": None, "different_row": None, "status": "no_valid_pair_of_pairs"}

    candidates: list[dict[str, object]] = []
    for probe_id, subset in valid.groupby("probe_id", sort=True):
        same_subset = subset[subset["part_similarity_bin"] == "same-part"].copy()
        diff_subset = subset[subset["part_similarity_bin"] == "different-part"].copy()
        if same_subset.empty or diff_subset.empty:
            continue
        candidates.append(
            {
                "probe_id": int(probe_id),
                "score": float(same_subset["direction_similarity"].median() - diff_subset["direction_similarity"].median()),
                "same_row": same_subset.sort_values(["direction_similarity", "part_similarity"], ascending=[False, False], kind="stable").iloc[0],
                "different_row": diff_subset.sort_values(["direction_similarity", "part_similarity"], ascending=[True, True], kind="stable").iloc[0],
            }
        )
    if candidates:
        best = max(candidates, key=lambda item: (item["score"], -item["probe_id"]))
        return {
            "probe_id": int(best["probe_id"]),
            "same_row": best["same_row"],
            "different_row": best["different_row"],
            "status": "matched_same_and_different",
        }

    fallback_probe = (
        valid.groupby("probe_id", sort=True).size().reset_index(name="n_rows").sort_values(["n_rows", "probe_id"], ascending=[False, True], kind="stable").iloc[0]
    )
    probe_id = int(fallback_probe["probe_id"])
    subset = valid[valid["probe_id"] == probe_id].copy()
    same_subset = subset[subset["part_similarity_bin"] == "same-part"].copy()
    diff_subset = subset[subset["part_similarity_bin"] == "different-part"].copy()
    return {
        "probe_id": int(probe_id),
        "same_row": same_subset.iloc[0] if not same_subset.empty else None,
        "different_row": diff_subset.iloc[0] if not diff_subset.empty else None,
        "status": "fallback_probe",
    }


def _representative_selection_summary(selection: Mapping[str, object]) -> dict[str, object]:
    def _row_to_summary(row: object) -> dict[str, object] | None:
        if row is None:
            return None
        if not isinstance(row, pd.Series):
            return {"status": "unsupported_row_type"}
        return {
            "probe_id": int(row["probe_id"]),
            "pair_i_id": int(row["pair_i_id"]),
            "pair_j_id": int(row["pair_j_id"]),
            "sample_i_id": int(row["sample_i_id"]),
            "sample_j_id": int(row["sample_j_id"]),
            "sample_i_label": int(row["sample_i_label"]),
            "sample_j_label": int(row["sample_j_label"]),
            "part_similarity": float(row["part_similarity"]),
            "direction_similarity": float(row["direction_similarity"]),
            "part_similarity_bin": str(row.get("part_similarity_bin", "")),
            "pair_pair_status": str(row.get("pair_pair_status", "")),
        }

    return {
        "status": str(selection.get("status", "")),
        "probe_id": None if selection.get("probe_id") is None else int(selection["probe_id"]),
        "same_example": _row_to_summary(selection.get("same_row")),
        "different_example": _row_to_summary(selection.get("different_row")),
    }


def _plot_image(ax, image: torch.Tensor, title: str) -> None:
    arr = image.detach().cpu().to(torch.float32).squeeze(0).numpy()
    ax.imshow(arr, cmap=CMAP_IMAGE_GRAY, vmin=0.0, vmax=max(1.0, float(np.nanmax(arr))))
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_mask_overlay(ax, probe_image: torch.Tensor, mask: np.ndarray, title: str) -> None:
    arr = probe_image.detach().cpu().to(torch.float32).squeeze(0).numpy()
    ax.imshow(arr, cmap=CMAP_IMAGE_GRAY, vmin=0.0, vmax=max(1.0, float(np.nanmax(arr))))
    mask_float = np.asarray(mask, dtype=np.float64)
    ax.imshow(np.ma.masked_where(mask_float <= 0.0, mask_float), cmap=CMAP_MASK_OVERLAY, alpha=ALPHA_MASK_OVERLAY, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_representative_same_probe_examples(
    *,
    pair_df: pd.DataFrame,
    pair_of_pairs_df: pd.DataFrame,
    images: torch.Tensor,
) -> tuple[plt.Figure, dict[str, object]]:
    selection = _choose_representative_examples(pair_of_pairs_df)
    apply_publication_style()
    fig, axes = plt.subplots(2, 6, figsize=FIGSIZE_REPRESENTATIVE_GRID, squeeze=False)
    if selection["probe_id"] is None:
        axes[0, 0].text(0.5, 0.5, "No representative probe available", ha="center", va="center")
        axes[0, 0].axis("off")
        for ax in axes.flat[1:]:
            ax.axis("off")
        fig.tight_layout()
        return fig, selection

    pair_lookup = pair_df.set_index("pair_id", drop=False)
    probe_id = int(selection["probe_id"])
    probe_image = images[probe_id]
    subset_pairs = pair_df[(pair_df["probe_id"] == probe_id) & (pair_df["pair_status"] == "ok")].copy().sort_values(["pair_id"], kind="stable")
    projected = _normalize_projection_for_plot(_project_vectors_to_2d([np.asarray(vec, dtype=np.float64) for vec in subset_pairs["DeltaV_pair"]]))
    pair_proj_lookup = {int(pair_id): projected[idx] for idx, pair_id in enumerate(subset_pairs["pair_id"].astype(int).tolist())}

    row_specs = [
        ("same-part", selection.get("same_row")),
        ("different-part", selection.get("different_row")),
    ]
    for row_idx, (row_label, pair_row) in enumerate(row_specs):
        if pair_row is None:
            axes[row_idx, 0].text(0.5, 0.5, f"No {row_label} exemplar", ha="center", va="center")
            axes[row_idx, 0].axis("off")
            for ax in axes[row_idx, 1:]:
                ax.axis("off")
            continue

        pair_i = pair_lookup.loc[int(pair_row["pair_i_id"])]
        pair_j = pair_lookup.loc[int(pair_row["pair_j_id"])]
        sample_i_image = images[int(pair_i["sample_id"])]
        sample_j_image = images[int(pair_j["sample_id"])]
        _plot_image(axes[row_idx, 0], sample_i_image, f"{row_label}: sample i\nid={int(pair_i['sample_id'])}")
        _plot_image(axes[row_idx, 1], sample_j_image, f"{row_label}: sample j\nid={int(pair_j['sample_id'])}")
        _plot_image(axes[row_idx, 2], probe_image, f"fixed probe\nid={probe_id}")
        _plot_mask_overlay(axes[row_idx, 3], probe_image, np.asarray(pair_i["probe_overlap_mask"], dtype=bool), "probe-side overlap i")
        _plot_mask_overlay(axes[row_idx, 4], probe_image, np.asarray(pair_j["probe_overlap_mask"], dtype=bool), "probe-side overlap j")

        ax_proj = axes[row_idx, 5]
        ax_proj.axhline(0.0, color=COLOR_GUIDE_LIGHT, linewidth=LINE_WIDTH_GUIDE)
        ax_proj.axvline(0.0, color=COLOR_GUIDE_LIGHT, linewidth=LINE_WIDTH_GUIDE)
        for pair_id, point in pair_proj_lookup.items():
            ax_proj.scatter(point[0], point[1], s=24, alpha=0.45, color=COLOR_NEUTRAL_LIGHT)
            ax_proj.text(point[0] + 0.02, point[1] + 0.02, f"{pair_id}", fontsize=7, alpha=0.75, clip_on=True)
        point_i = pair_proj_lookup.get(int(pair_i["pair_id"]), np.zeros(2, dtype=np.float64))
        point_j = pair_proj_lookup.get(int(pair_j["pair_id"]), np.zeros(2, dtype=np.float64))
        ax_proj.scatter(point_i[0], point_i[1], s=70, color=PAIR_HIGHLIGHT_COLORS["pair_i"], label="pair i")
        ax_proj.scatter(point_j[0], point_j[1], s=70, color=PAIR_HIGHLIGHT_COLORS["pair_j"], label="pair j")
        ax_proj.plot([point_i[0], point_j[0]], [point_i[1], point_j[1]], color=COLOR_NEUTRAL_MID, linewidth=1.2, alpha=0.8)
        ax_proj.set_title(
            f"DeltaV projection\ncos={float(pair_row['direction_similarity']):.3f}, part={float(pair_row['part_similarity']):.3f}",
            fontsize=11,
        )
        ax_proj.set_xlabel("PC1")
        ax_proj.set_ylabel("PC2")
        apply_standard_legend(ax_proj, compact=True, loc="best")
        ax_proj.grid(alpha=GRID_ALPHA)
        ax_proj.set_xlim(-1.15, 1.15)
        ax_proj.set_ylim(-1.15, 1.15)

    fig.suptitle(
        "Representative same-probe examples: direction follows the probe-centered part that overlap emphasizes",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    return fig, selection


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Same-probe same-part direction validation experiment.")
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--delay2-ms", type=float, default=DEFAULT_DELAY2_MS)
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--samples-per-probe", type=int, default=DEFAULT_SAMPLES_PER_PROBE)
    parser.add_argument("--num-sim-bins", type=int, default=DEFAULT_NUM_SIM_BINS)
    parser.add_argument("--foreground-threshold", type=float, default=DEFAULT_FOREGROUND_THRESHOLD)
    parser.add_argument("--use-dilated-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dilation-radius", type=int, default=DEFAULT_DILATION_RADIUS)
    parser.add_argument("--num-control-candidates", type=int, default=DEFAULT_NUM_CONTROL_CANDIDATES)
    parser.add_argument("--save-case-count", type=int, default=DEFAULT_SAVE_CASE_COUNT)
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    _validate_positive("--sample-ms", float(args.sample_ms))
    _validate_positive("--delay2-ms", float(args.delay2_ms), allow_zero=True)
    _validate_positive("--probe-ms", float(args.probe_ms))
    _validate_positive("--batch-size", int(args.batch_size))
    _validate_positive("--max-probes", int(args.max_probes))
    _validate_positive("--samples-per-probe", int(args.samples_per_probe))
    _validate_positive("--num-sim-bins", int(args.num_sim_bins))
    _validate_positive("--dilation-radius", int(args.dilation_radius), allow_zero=True)
    _validate_positive("--num-control-candidates", int(args.num_control_candidates))
    _validate_positive("--save-case-count", int(args.save_case_count), allow_zero=True)
    if int(args.samples_per_probe) < 2:
        raise ValueError("--samples-per-probe must be at least 2 for same-probe pair-of-pairs analysis.")

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=1.0 * ms, sample_ms=float(args.sample_ms), delay2_ms=float(args.delay2_ms), probe_ms=float(args.probe_ms))
    if spec.sample_steps <= 0 or spec.probe_steps <= 0:
        raise ValueError("sample/probe duration must resolve to positive steps.")
    if spec.delay2_steps < 0:
        raise ValueError("delay2 must resolve to a non-negative number of steps.")

    layout = prepare_result_layout(args.output_dir)
    result_root = layout.root
    output_dir = layout.data_dir
    figures_dir = layout.figure_dir
    logs_dir = layout.log_dir

    dataset = _load_dataset(args.dataset_root, args.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    num_classes = int(labels.max()) + 1
    class_index = build_class_index(dataset, num_classes=num_classes)
    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(float(spec.sample_ms), float(spec.probe_ms)),
    )
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )

    df_pairs = build_same_probe_dms_pairs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=int(args.max_probes),
        samples_per_probe=int(args.samples_per_probe),
        num_bins=int(args.num_sim_bins),
        seed=int(args.seed),
    )

    signature_rows: list[dict[str, object]] = []
    for row in df_pairs.itertuples(index=False):
        signature_bundle = build_probe_part_signature_for_pair(
            sample_image=images[int(row.sample_id)],
            probe_image=images[int(row.probe_id)],
            foreground_threshold=float(args.foreground_threshold),
            use_dilated_overlap=bool(args.use_dilated_overlap),
            dilation_radius=int(args.dilation_radius),
            seed=mix_seed(int(args.seed), int(row.pair_id), int(row.sample_id), int(row.probe_id)),
            num_control_candidates=int(args.num_control_candidates),
        )
        signature_row = {
            "pair_id": int(row.pair_id),
            "probe_overlap_mask": signature_bundle["probe_overlap_mask"],
            "part_signature_binary": signature_bundle["part_signature_binary"],
            "part_signature_weighted": signature_bundle["part_signature_weighted"],
        }
        signature_row.update(signature_bundle["metadata"])
        signature_rows.append(signature_row)
    df_signatures = pd.DataFrame(signature_rows)
    df_pairs = df_pairs.merge(df_signatures, on="pair_id", how="left", validate="one_to_one")

    pair_rollout = compute_pair_delta_vectors(
        net=net,
        images=images,
        encoder=encoder,
        device=device,
        readout_step=int(readout_step),
        spec=spec,
        pair_df=df_pairs,
        batch_size=int(args.batch_size),
    )
    df_pairs = pair_rollout["pair_df"]
    df_pair_of_pairs = build_same_probe_pair_of_pairs(df_pairs)
    binned = assign_part_similarity_bins(df_pair_of_pairs)
    df_pair_of_pairs = binned["pair_of_pairs_df"]
    summary_df = summarize_direction_similarity_by_part_bin(df_pair_of_pairs)

    pair_csv = save_tidy_csv(_select_pair_record_columns(df_pairs), output_dir / "same_probe_pair_records.csv", sort_by=["probe_id", "pair_id"])
    pair_of_pairs_csv = save_tidy_csv(
        _select_pair_of_pairs_columns(df_pair_of_pairs),
        output_dir / "same_probe_pair_of_pairs.csv",
        sort_by=["probe_id", "pair_i_id", "pair_j_id"],
    )
    sidecars = _save_pair_sidecars(df_pairs, output_dir)

    fig_paths: dict[str, object] = {}
    scatter_stats = {"spearman": {"status": "not_run"}, "linear_fit": {"status": "not_run"}}
    representative_selection = {"status": "skipped"}
    if not bool(args.skip_figures):
        fig1, scatter_stats = plot_part_similarity_vs_direction_similarity(df_pair_of_pairs)
        fig_paths["figure_1_part_similarity_vs_direction_similarity"] = save_figure_all_formats(
            fig1,
            figures_dir / "figure_1_part_similarity_vs_direction_similarity",
        )
        plt.close(fig1)

        fig2 = plot_same_part_vs_different_part(df_pair_of_pairs, summary_df)
        fig_paths["figure_2_same_part_vs_different_part"] = save_figure_all_formats(
            fig2,
            figures_dir / "figure_2_same_part_vs_different_part",
        )
        plt.close(fig2)

        fig3, representative_selection = plot_representative_same_probe_examples(pair_df=df_pairs, pair_of_pairs_df=df_pair_of_pairs, images=images)
        fig_paths["figure_3_representative_same_probe_examples"] = save_figure_all_formats(
            fig3,
            figures_dir / "figure_3_representative_same_probe_examples",
        )
        plt.close(fig3)

    summary_payload = {
        "design_summary": {
            "fixed_delay_ms": float(spec.delay2_ms),
            "same_probe_design": True,
            "task_sequence": "sample -> delay2 -> probe",
            "includes_distractor": False,
            "comparison_scope": "same-probe pair-of-pairs only",
        },
        "scientific_statement": (
            "This experiment tests whether, under a fixed probe and fixed delay2, direction is primarily organized "
            "by which probe-relevant part overlap emphasizes, rather than by preceding source identity alone."
        ),
        "part_signature_definition": {
            "binary": "flattened probe-side overlap mask",
            "weighted": "flattened abs(probe) masked by probe-side overlap support",
        },
        "part_similarity_metric": "cosine(part_signature_weighted_i, part_signature_weighted_j)",
        "direction_similarity_metric": "cosine(DeltaV_i, DeltaV_j)",
        "binning_rule": "global tertiles on valid same-probe pair-of-pairs using weighted part similarity",
        "q33": float(binned["q33"]),
        "q67": float(binned["q67"]),
        "bin_counts": binned["bin_counts"],
        "invalid_count": int(binned["invalid_count"]),
        "warnings": list(binned["warnings"]),
        "summary_by_bin": summary_df.to_dict(orient="records"),
        "pair_counts": {
            "n_pairs": int(len(df_pairs)),
            "n_valid_pairs": int(df_pairs["pair_status"].eq("ok").sum()),
            "n_pair_of_pairs": int(len(df_pair_of_pairs)),
            "n_valid_pair_of_pairs": int(df_pair_of_pairs["pair_pair_status"].eq("ok").sum()),
        },
        "correlation_summary": scatter_stats,
        "representative_selection": _representative_selection_summary(representative_selection),
        "assumptions": {
            "fixed_delay": "All comparisons use the same delay2 value.",
            "source_identity_scope": "Preceding source identity is not the primary object of comparison here.",
            "mechanistic_target": "The target is whether direction is primarily organized by which probe-relevant part is emphasized.",
            "mechanism_chain_context": "This is a direction-interpretation validation layered on top of the overlap -> probe dynamics -> decision vector shift chain.",
            "source_effects_position": "This does not deny source effects; it tests whether source effects are largely expressed through probe-centered overlap part selection.",
        },
        "artifacts": {
            "same_probe_pair_records_csv": str(pair_csv),
            "same_probe_pair_of_pairs_csv": str(pair_of_pairs_csv),
            **sidecars,
            "figure_paths": fig_paths,
        },
    }
    summary_json = _save_json(summary_payload, output_dir / "same_probe_part_summary.json")

    run_config = {
        "model_path": str(args.model_path),
        "config": None if args.config is None else str(args.config),
        "dataset_root": str(args.dataset_root),
        "split": str(args.split),
        "device": str(device),
        "seed": int(args.seed),
        "output_dir": str(result_root),
        "sample_ms": float(spec.sample_ms),
        "delay2_ms": float(spec.delay2_ms),
        "probe_ms": float(spec.probe_ms),
        "sample_steps": int(spec.sample_steps),
        "delay2_steps": int(spec.delay2_steps),
        "probe_steps": int(spec.probe_steps),
        "batch_size": int(args.batch_size),
        "max_probes": int(args.max_probes),
        "samples_per_probe": int(args.samples_per_probe),
        "num_sim_bins": int(args.num_sim_bins),
        "foreground_threshold": float(args.foreground_threshold),
        "use_dilated_overlap": bool(args.use_dilated_overlap),
        "dilation_radius": int(args.dilation_radius),
        "num_control_candidates": int(args.num_control_candidates),
        "save_case_count": int(args.save_case_count),
        "skip_figures": bool(args.skip_figures),
        "readout_step": int(readout_step),
        "fixed_delay_ms": float(spec.delay2_ms),
        "same_probe_design": True,
        "part_signature_definition": "probe-side overlap mask",
        "part_signature_weight_definition": "abs(probe) masked by probe_overlap_mask",
        "part_similarity_metric": "cosine(weighted_signature)",
        "direction_similarity_metric": "cosine(DeltaV_i, DeltaV_j)",
        "binning_rule": "global tertiles on valid same-probe pair-of-pairs",
        "scientific_hypothesis": "Direction is primarily organized by which probe-centered overlap part is emphasized, not by preceding source identity alone.",
        "summary_json": str(summary_json),
    }
    run_config_path = save_run_config(run_config, result_root)
    summary_path = save_summary_json(summary_payload, result_root)
    run_log_path = save_log_lines(
        [
            "experiment=same_probe_same_part_direction_validation_experiment",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"pairs={len(df_pairs)}",
            f"pair_of_pairs={len(df_pair_of_pairs)}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_path.resolve()}",
        ],
        logs_dir,
    )


if __name__ == "__main__":
    main()
