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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.json_io import save_json_payload as _save_json
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.seed import mix_seed
from src.experiments.common.diagnostic_mask_utils import apply_ablation
from src.experiments.common.voltage_readout import resolve_readout_step
from src.experiments.distractor.shared.masking import (
    apply_input_mask_to_spike_batch,
    build_best_energy_matched_control_mask as _build_best_energy_matched_control_mask,
    compute_final_pattern_similarity,
    compute_trace_pattern_similarity,
    dilate_mask as _dilate_mask,
    foreground_mask as _foreground_mask,
    mask_energy as _mask_energy,
)
from src.experiments.distractor.shared.pair_sampling import (
    assign_bins_from_values as _assign_bins_from_values,
    build_dataset_arrays,
    extract_grouped_voltage_vector,
    select_probe_ids_balanced,
    select_probe_samples_from_candidates as _select_probe_samples_from_candidates,
)
from src.experiments.distractor.shared.config import sem as _sem
from src.plotting.common.io import (
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
)
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    ALPHA_FILL,
    ALPHA_FILL_LIGHT,
    ALPHA_SCATTER,
    FIGSIZE_THREE_PANEL,
    FIGSIZE_TWO_PANEL,
    GRID_ALPHA,
    LINE_WIDTH_PRIMARY,
    LINE_WIDTH_REFERENCE,
    apply_standard_legend,
    case_grid_figsize,
)

DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_OUTPUT_DIR = "results/distractor_input_causal_perturbation_experiment"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_DELAY1_MS = 400.0
DEFAULT_DISTRACTOR_MS = 200.0
DEFAULT_DELAY2_MS = 400.0
DEFAULT_PROBE_MS = 100.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_PROBES = 20
DEFAULT_SAMPLES_PER_PROBE = 12
DEFAULT_MAX_TRIPLETS = 240
DEFAULT_NUM_SIM_BINS = 5
DEFAULT_FOREGROUND_THRESHOLD = 0.0
DEFAULT_DILATION_RADIUS = 1
DEFAULT_SAVE_CASE_COUNT = 4
DEFAULT_NUM_CONTROL_CANDIDATES = 32
DEFAULT_TAU_MS = 500.0

MAIN_DYNAMIC_CONDITIONS: tuple[str, ...] = (
    "sample_remove_SPonly_dynamic",
    "distractor_remove_DPonly_dynamic",
    "sample_remove_SDP_dynamic",
    "distractor_remove_SDP_dynamic",
    "both_remove_SDP_dynamic",
)
LAYER_ORDER: tuple[str, ...] = ("L1", "L2", "L3", "final")
CONDITION_COLORS: dict[str, str] = {
    "full_dynamic_distractor": "#4C78A8",
    "full_static_distractor": "#9E9E9E",
    "sample_remove_SPonly_dynamic": "#E45756",
    "sample_remove_SPonly_control_dynamic": "#72B7B2",
    "distractor_remove_DPonly_dynamic": "#F58518",
    "distractor_remove_DPonly_control_dynamic": "#54A24B",
    "sample_remove_SDP_dynamic": "#B279A2",
    "sample_remove_SDP_control_dynamic": "#FF9DA6",
    "distractor_remove_SDP_dynamic": "#9D755D",
    "distractor_remove_SDP_control_dynamic": "#BAB0AC",
    "both_remove_SDP_dynamic": "#2F4B7C",
    "both_remove_SDP_control_dynamic": "#6C8EBF",
    "remove_probe_relevant_union_dynamic": "#C44536",
    "remove_probe_relevant_union_control_dynamic": "#7FB069",
}


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay1_ms: float
    distractor_ms: float
    delay2_ms: float
    probe_ms: float
    phase_reset: bool = True

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def delay1_steps(self) -> int:
        return int(round((self.delay1_ms * ms) / self.dt))

    @property
    def distractor_steps(self) -> int:
        return int(round((self.distractor_ms * ms) / self.dt))

    @property
    def delay2_steps(self) -> int:
        return int(round((self.delay2_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


@dataclass(frozen=True)
class TripletMaskBundle:
    sp_only_mask: np.ndarray
    dp_only_mask: np.ndarray
    sdp_mask: np.ndarray
    probe_relevant_union_mask: np.ndarray
    sample_sp_only_mask: np.ndarray
    sample_sp_only_control_mask: np.ndarray
    distractor_dp_only_mask: np.ndarray
    distractor_dp_only_control_mask: np.ndarray
    sample_sdp_mask: np.ndarray
    sample_sdp_control_mask: np.ndarray
    distractor_sdp_mask: np.ndarray
    distractor_sdp_control_mask: np.ndarray
    sample_union_mask: np.ndarray
    sample_union_control_mask: np.ndarray
    distractor_union_mask: np.ndarray
    distractor_union_control_mask: np.ndarray
    sample_foreground_mask: np.ndarray
    distractor_foreground_mask: np.ndarray
    probe_foreground_mask: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    stsp_mode: str
    sample_mask_key: str | None
    distractor_mask_key: str | None


@dataclass(frozen=True)
class RolloutReadout:
    grouped_voltage: np.ndarray
    probe_l1_trace: torch.Tensor
    probe_l2_trace: torch.Tensor
    probe_l3_trace: torch.Tensor
    prediction_probe: np.ndarray
    first_fire_t_probe: np.ndarray
    readout_step: int


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


def _rebalance_global_triplets(df_triplets: pd.DataFrame, max_triplets: int) -> pd.DataFrame:
    if len(df_triplets) <= int(max_triplets):
        return df_triplets.copy().reset_index(drop=True)
    by_probe = [sub.copy().reset_index(drop=True) for _, sub in df_triplets.groupby("probe_id", sort=True)]
    selected_parts: list[pd.DataFrame] = []
    cursor = 0
    while len(selected_parts) < int(max_triplets):
        made_progress = False
        for group in by_probe:
            if cursor >= len(group):
                continue
            selected_parts.append(group.iloc[[cursor]].copy())
            made_progress = True
            if len(selected_parts) >= int(max_triplets):
                break
        if not made_progress:
            break
        cursor += 1
    return pd.concat(selected_parts, axis=0, ignore_index=True).reset_index(drop=True)


def _select_distractor_for_sample_probe(
    *,
    sample_id: int,
    sample_label: int,
    probe_id: int,
    probe_label: int,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    all_ids: np.ndarray,
    num_bins: int,
    sample_position: int,
    total_samples_for_probe: int,
) -> dict[str, object]:
    sims_to_probe = flat_normalized @ flat_normalized[int(probe_id)]
    sims_to_sample = flat_normalized @ flat_normalized[int(sample_id)]
    mask = (
        (all_ids != int(sample_id))
        & (all_ids != int(probe_id))
        & (labels[all_ids] != int(sample_label))
        & (labels[all_ids] != int(probe_label))
    )
    candidate_ids = all_ids[mask]
    if candidate_ids.size <= 0:
        raise RuntimeError("No valid distractor candidate found for the requested sample/probe pair.")
    df_candidates = pd.DataFrame(
        {
            "distractor_id": candidate_ids.astype(np.int64, copy=False),
            "distractor_label": labels[candidate_ids].astype(np.int64, copy=False),
            "dp_similarity": sims_to_probe[candidate_ids].astype(np.float64, copy=False),
            "sd_similarity": sims_to_sample[candidate_ids].astype(np.float64, copy=False),
        }
    ).sort_values(["dp_similarity", "sd_similarity", "distractor_id"], kind="stable").reset_index(drop=True)
    df_candidates["dp_bin"] = _assign_bins_from_values(
        df_candidates["dp_similarity"].to_numpy(dtype=np.float64, copy=False),
        num_bins=int(num_bins),
    )
    unique_bins = pd.unique(df_candidates["dp_bin"]).tolist()
    target_positions = np.floor(
        np.linspace(0, max(len(unique_bins) - 1, 0), num=max(1, int(total_samples_for_probe)))
    ).astype(np.int64)
    target_position = int(target_positions[min(int(sample_position), len(target_positions) - 1)])
    target_bin = str(unique_bins[target_position])
    sub = df_candidates[df_candidates["dp_bin"] == target_bin].copy().reset_index(drop=True)
    if sub.empty:
        sub = df_candidates.copy().reset_index(drop=True)
        target_bin = str(sub.iloc[0]["dp_bin"])
    chosen = sub.iloc[min(int(sample_position), len(sub) - 1)]
    return {
        "distractor_id": int(chosen["distractor_id"]),
        "distractor_label": int(chosen["distractor_label"]),
        "dp_similarity": float(chosen["dp_similarity"]),
        "sd_similarity": float(chosen["sd_similarity"]),
        "dp_bin": str(target_bin),
    }


def build_triplet_specs(
    images: torch.Tensor,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    *,
    max_probes: int,
    samples_per_probe: int,
    num_bins: int,
    max_triplets: int,
    seed: int,
) -> pd.DataFrame:
    del images
    probe_ids = select_probe_ids_balanced(class_index=class_index, max_probes=max_probes, seed=mix_seed(seed, 31))
    rows: list[dict[str, object]] = []
    all_ids = np.arange(len(labels), dtype=np.int64)
    for probe_rank, probe_id in enumerate(probe_ids):
        probe_id_int = int(probe_id)
        probe_label = int(labels[probe_id_int])
        sims_to_probe = flat_normalized @ flat_normalized[probe_id_int]
        mask = (all_ids != probe_id_int) & (labels[all_ids] != probe_label)
        candidate_ids = all_ids[mask]
        df_sample_candidates = pd.DataFrame(
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
            df_candidates=df_sample_candidates,
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
        for sample_position, row in enumerate(selected_samples.itertuples(index=False)):
            distractor = _select_distractor_for_sample_probe(
                sample_id=int(row.sample_id),
                sample_label=int(row.sample_label),
                probe_id=probe_id_int,
                probe_label=probe_label,
                labels=labels,
                flat_normalized=flat_normalized,
                all_ids=all_ids,
                num_bins=int(num_bins),
                sample_position=int(sample_position),
                total_samples_for_probe=int(len(selected_samples)),
            )
            rows.append(
                {
                    "sample_id": int(row.sample_id),
                    "sample_label": int(row.sample_label),
                    "distractor_id": int(distractor["distractor_id"]),
                    "distractor_label": int(distractor["distractor_label"]),
                    "probe_id": probe_id_int,
                    "probe_label": probe_label,
                    "probe_rank": int(probe_rank),
                    "sp_similarity": float(row.sp_similarity),
                    "dp_similarity": float(distractor["dp_similarity"]),
                    "sd_similarity": float(distractor["sd_similarity"]),
                    "sp_bin": str(row.sp_bin),
                    "dp_bin": str(distractor["dp_bin"]),
                }
            )
    if not rows:
        raise RuntimeError("No distractor triplets were generated.")
    df_triplets = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["sample_id", "distractor_id", "probe_id"], keep="first")
        .reset_index(drop=True)
    )
    df_triplets = _rebalance_global_triplets(df_triplets, max_triplets=int(max_triplets))
    sp_label_order = {label: idx for idx, label in enumerate(pd.unique(df_triplets["sp_bin"]).tolist())}
    dp_label_order = {label: idx for idx, label in enumerate(pd.unique(df_triplets["dp_bin"]).tolist())}
    df_triplets["sp_bin_index"] = df_triplets["sp_bin"].map(sp_label_order).astype(np.int64)
    df_triplets["dp_bin_index"] = df_triplets["dp_bin"].map(dp_label_order).astype(np.int64)
    df_triplets = df_triplets.sort_values(
        ["probe_rank", "sp_bin_index", "dp_bin_index", "sp_similarity", "dp_similarity", "sample_id", "distractor_id"],
        kind="stable",
    ).reset_index(drop=True)
    df_triplets["triplet_id"] = np.arange(len(df_triplets), dtype=np.int64)
    return df_triplets


def _stack_encoded_batch(
    image_ids: Sequence[int],
    *,
    images: torch.Tensor,
    encoder,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    batch_images = images[[int(idx) for idx in image_ids]].to(device=device, dtype=torch.float32)
    return encode_images(encoder, batch_images, steps=int(steps))


def prepare_triplet_spike_batch(
    images: torch.Tensor,
    batch_df: pd.DataFrame,
    encoder,
    spec: ExperimentSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_ids = batch_df["sample_id"].astype(int).tolist()
    distractor_ids = batch_df["distractor_id"].astype(int).tolist()
    probe_ids = batch_df["probe_id"].astype(int).tolist()
    unique_sample_ids = list(dict.fromkeys(sample_ids))
    unique_distractor_ids = list(dict.fromkeys(distractor_ids))
    unique_probe_ids = list(dict.fromkeys(probe_ids))
    sample_encoded = _stack_encoded_batch(unique_sample_ids, images=images, encoder=encoder, steps=spec.sample_steps, device=device)
    distractor_encoded = _stack_encoded_batch(unique_distractor_ids, images=images, encoder=encoder, steps=spec.distractor_steps, device=device)
    probe_encoded = _stack_encoded_batch(unique_probe_ids, images=images, encoder=encoder, steps=spec.probe_steps, device=device)
    sample_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_sample_ids)}
    distractor_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_distractor_ids)}
    probe_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_probe_ids)}
    sample_select = torch.tensor([sample_lookup[int(idx)] for idx in sample_ids], dtype=torch.long, device=device)
    distractor_select = torch.tensor([distractor_lookup[int(idx)] for idx in distractor_ids], dtype=torch.long, device=device)
    probe_select = torch.tensor([probe_lookup[int(idx)] for idx in probe_ids], dtype=torch.long, device=device)
    return (
        sample_encoded.index_select(0, sample_select),
        distractor_encoded.index_select(0, distractor_select),
        probe_encoded.index_select(0, probe_select),
    )


def _project_phase_mask(mask: np.ndarray, phase_foreground: np.ndarray, *, use_dilated_overlap: bool, dilation_radius: int) -> np.ndarray:
    base = np.asarray(mask, dtype=bool)
    if bool(use_dilated_overlap) and int(dilation_radius) > 0 and base.any():
        return _dilate_mask(base, int(dilation_radius)) & np.asarray(phase_foreground, dtype=bool)
    return base & np.asarray(phase_foreground, dtype=bool)


def _build_phase_control_mask(
    *,
    image: torch.Tensor,
    reference_mask: np.ndarray,
    phase_foreground: np.ndarray,
    phase_probe_relevant_union: np.ndarray,
    seed: int,
    num_control_candidates: int,
) -> tuple[np.ndarray, str, float]:
    rng = np.random.default_rng(int(seed))
    return _build_best_energy_matched_control_mask(
        image=image,
        reference_mask=np.asarray(reference_mask, dtype=bool),
        preferred_pool_mask=np.asarray(phase_foreground, dtype=bool) & ~np.asarray(phase_probe_relevant_union, dtype=bool),
        fallback_pool_mask=~np.asarray(phase_probe_relevant_union, dtype=bool),
        rng=rng,
        num_candidates=int(num_control_candidates),
    )


def compute_time_weighted_structures(
    *,
    spec: ExperimentSpec,
    area_sp_only: float,
    area_dp_only: float,
    area_sdp: float,
    tau_ms: float,
) -> dict[str, float]:
    tau = float(tau_ms)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau_ms must be positive and finite.")
    sample_to_probe_gap_ms = float(spec.delay1_ms + spec.distractor_ms + spec.delay2_ms)
    distractor_to_probe_gap_ms = float(spec.delay2_ms)
    w_s = float(np.exp(-sample_to_probe_gap_ms / tau))
    w_d = float(np.exp(-distractor_to_probe_gap_ms / tau))
    return {
        "w_S": w_s,
        "w_D": w_d,
        "X_SP": float(w_s * float(area_sp_only)),
        "X_DP": float(w_d * float(area_dp_only)),
        "X_SDP": float((w_s * w_d) * float(area_sdp)),
    }


def build_probe_relevant_masks_for_triplet(
    sample_image: torch.Tensor,
    distractor_image: torch.Tensor,
    probe_image: torch.Tensor,
    *,
    foreground_threshold: float,
    use_dilated_overlap: bool,
    dilation_radius: int,
    seed: int,
    num_control_candidates: int,
) -> TripletMaskBundle:
    sample_fg = _foreground_mask(sample_image, threshold=foreground_threshold)
    distractor_fg = _foreground_mask(distractor_image, threshold=foreground_threshold)
    probe_fg = _foreground_mask(probe_image, threshold=foreground_threshold)

    sp_only = sample_fg & probe_fg & ~distractor_fg
    dp_only = distractor_fg & probe_fg & ~sample_fg
    sdp = sample_fg & distractor_fg & probe_fg
    union_mask = sp_only | dp_only | sdp

    sample_sp_only = _project_phase_mask(sp_only, sample_fg, use_dilated_overlap=use_dilated_overlap, dilation_radius=dilation_radius)
    distractor_dp_only = _project_phase_mask(dp_only, distractor_fg, use_dilated_overlap=use_dilated_overlap, dilation_radius=dilation_radius)
    sample_sdp = _project_phase_mask(sdp, sample_fg, use_dilated_overlap=use_dilated_overlap, dilation_radius=dilation_radius)
    distractor_sdp = _project_phase_mask(sdp, distractor_fg, use_dilated_overlap=use_dilated_overlap, dilation_radius=dilation_radius)
    sample_union = _project_phase_mask(union_mask, sample_fg, use_dilated_overlap=use_dilated_overlap, dilation_radius=dilation_radius)
    distractor_union = _project_phase_mask(union_mask, distractor_fg, use_dilated_overlap=use_dilated_overlap, dilation_radius=dilation_radius)

    sample_probe_relevant_union = sample_sp_only | sample_sdp
    distractor_probe_relevant_union = distractor_dp_only | distractor_sdp

    sample_sp_control, sample_sp_source, sample_sp_gap = _build_phase_control_mask(
        image=sample_image,
        reference_mask=sample_sp_only,
        phase_foreground=sample_fg,
        phase_probe_relevant_union=sample_probe_relevant_union,
        seed=mix_seed(seed, 101),
        num_control_candidates=int(num_control_candidates),
    )
    distractor_dp_control, distractor_dp_source, distractor_dp_gap = _build_phase_control_mask(
        image=distractor_image,
        reference_mask=distractor_dp_only,
        phase_foreground=distractor_fg,
        phase_probe_relevant_union=distractor_probe_relevant_union,
        seed=mix_seed(seed, 202),
        num_control_candidates=int(num_control_candidates),
    )
    sample_sdp_control, sample_sdp_source, sample_sdp_gap = _build_phase_control_mask(
        image=sample_image,
        reference_mask=sample_sdp,
        phase_foreground=sample_fg,
        phase_probe_relevant_union=sample_probe_relevant_union,
        seed=mix_seed(seed, 303),
        num_control_candidates=int(num_control_candidates),
    )
    distractor_sdp_control, distractor_sdp_source, distractor_sdp_gap = _build_phase_control_mask(
        image=distractor_image,
        reference_mask=distractor_sdp,
        phase_foreground=distractor_fg,
        phase_probe_relevant_union=distractor_probe_relevant_union,
        seed=mix_seed(seed, 404),
        num_control_candidates=int(num_control_candidates),
    )
    sample_union_control, sample_union_source, sample_union_gap = _build_phase_control_mask(
        image=sample_image,
        reference_mask=sample_union,
        phase_foreground=sample_fg,
        phase_probe_relevant_union=sample_probe_relevant_union,
        seed=mix_seed(seed, 505),
        num_control_candidates=int(num_control_candidates),
    )
    distractor_union_control, distractor_union_source, distractor_union_gap = _build_phase_control_mask(
        image=distractor_image,
        reference_mask=distractor_union,
        phase_foreground=distractor_fg,
        phase_probe_relevant_union=distractor_probe_relevant_union,
        seed=mix_seed(seed, 606),
        num_control_candidates=int(num_control_candidates),
    )

    metadata = {
        "foreground_threshold": float(foreground_threshold),
        "use_dilated_overlap": int(bool(use_dilated_overlap)),
        "dilation_radius": int(dilation_radius),
        "sample_foreground_area": int(sample_fg.sum()),
        "distractor_foreground_area": int(distractor_fg.sum()),
        "probe_foreground_area": int(probe_fg.sum()),
        "area_SPonly": int(sp_only.sum()),
        "area_DPonly": int(dp_only.sum()),
        "area_SDP": int(sdp.sum()),
        "area_probe_relevant_union": int(union_mask.sum()),
        "sample_SPonly_area": int(sample_sp_only.sum()),
        "sample_SPonly_control_area": int(sample_sp_control.sum()),
        "sample_SPonly_energy": float(_mask_energy(sample_image, sample_sp_only)),
        "sample_SPonly_control_energy": float(_mask_energy(sample_image, sample_sp_control)),
        "sample_SPonly_control_source": str(sample_sp_source),
        "sample_SPonly_control_energy_gap": float(sample_sp_gap),
        "sample_SPonly_empty": int(int(sample_sp_only.sum()) <= 0),
        "distractor_DPonly_area": int(distractor_dp_only.sum()),
        "distractor_DPonly_control_area": int(distractor_dp_control.sum()),
        "distractor_DPonly_energy": float(_mask_energy(distractor_image, distractor_dp_only)),
        "distractor_DPonly_control_energy": float(_mask_energy(distractor_image, distractor_dp_control)),
        "distractor_DPonly_control_source": str(distractor_dp_source),
        "distractor_DPonly_control_energy_gap": float(distractor_dp_gap),
        "distractor_DPonly_empty": int(int(distractor_dp_only.sum()) <= 0),
        "sample_SDP_area": int(sample_sdp.sum()),
        "sample_SDP_control_area": int(sample_sdp_control.sum()),
        "sample_SDP_energy": float(_mask_energy(sample_image, sample_sdp)),
        "sample_SDP_control_energy": float(_mask_energy(sample_image, sample_sdp_control)),
        "sample_SDP_control_source": str(sample_sdp_source),
        "sample_SDP_control_energy_gap": float(sample_sdp_gap),
        "sample_SDP_empty": int(int(sample_sdp.sum()) <= 0),
        "distractor_SDP_area": int(distractor_sdp.sum()),
        "distractor_SDP_control_area": int(distractor_sdp_control.sum()),
        "distractor_SDP_energy": float(_mask_energy(distractor_image, distractor_sdp)),
        "distractor_SDP_control_energy": float(_mask_energy(distractor_image, distractor_sdp_control)),
        "distractor_SDP_control_source": str(distractor_sdp_source),
        "distractor_SDP_control_energy_gap": float(distractor_sdp_gap),
        "distractor_SDP_empty": int(int(distractor_sdp.sum()) <= 0),
        "sample_union_area": int(sample_union.sum()),
        "sample_union_control_area": int(sample_union_control.sum()),
        "sample_union_energy": float(_mask_energy(sample_image, sample_union)),
        "sample_union_control_energy": float(_mask_energy(sample_image, sample_union_control)),
        "sample_union_control_source": str(sample_union_source),
        "sample_union_control_energy_gap": float(sample_union_gap),
        "distractor_union_area": int(distractor_union.sum()),
        "distractor_union_control_area": int(distractor_union_control.sum()),
        "distractor_union_energy": float(_mask_energy(distractor_image, distractor_union)),
        "distractor_union_control_energy": float(_mask_energy(distractor_image, distractor_union_control)),
        "distractor_union_control_source": str(distractor_union_source),
        "distractor_union_control_energy_gap": float(distractor_union_gap),
    }
    return TripletMaskBundle(
        sp_only_mask=np.asarray(sp_only, dtype=bool),
        dp_only_mask=np.asarray(dp_only, dtype=bool),
        sdp_mask=np.asarray(sdp, dtype=bool),
        probe_relevant_union_mask=np.asarray(union_mask, dtype=bool),
        sample_sp_only_mask=np.asarray(sample_sp_only, dtype=bool),
        sample_sp_only_control_mask=np.asarray(sample_sp_control, dtype=bool),
        distractor_dp_only_mask=np.asarray(distractor_dp_only, dtype=bool),
        distractor_dp_only_control_mask=np.asarray(distractor_dp_control, dtype=bool),
        sample_sdp_mask=np.asarray(sample_sdp, dtype=bool),
        sample_sdp_control_mask=np.asarray(sample_sdp_control, dtype=bool),
        distractor_sdp_mask=np.asarray(distractor_sdp, dtype=bool),
        distractor_sdp_control_mask=np.asarray(distractor_sdp_control, dtype=bool),
        sample_union_mask=np.asarray(sample_union, dtype=bool),
        sample_union_control_mask=np.asarray(sample_union_control, dtype=bool),
        distractor_union_mask=np.asarray(distractor_union, dtype=bool),
        distractor_union_control_mask=np.asarray(distractor_union_control, dtype=bool),
        sample_foreground_mask=np.asarray(sample_fg, dtype=bool),
        distractor_foreground_mask=np.asarray(distractor_fg, dtype=bool),
        probe_foreground_mask=np.asarray(probe_fg, dtype=bool),
        metadata=metadata,
    )


def run_overlap_perturbed_distractor_task(
    net,
    sample_spikes: torch.Tensor,
    distractor_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    *,
    delay1_steps: int,
    delay2_steps: int,
    stsp_mode: str,
    readout_step: int,
    sample_input_mask: torch.Tensor | np.ndarray | None = None,
    distractor_input_mask: torch.Tensor | np.ndarray | None = None,
    phase_reset: bool = True,
) -> RolloutReadout:
    if sample_spikes.ndim != 5 or distractor_spikes.ndim != 5 or probe_spikes.ndim != 5:
        raise ValueError("sample_spikes, distractor_spikes, and probe_spikes must have shape [B, T, C, H, W]")
    batch_size, _, channels, height, width = sample_spikes.shape
    if int(distractor_spikes.shape[0]) != int(batch_size) or int(probe_spikes.shape[0]) != int(batch_size):
        raise ValueError("All spike tensors must share the same batch size.")

    masked_sample_spikes = apply_input_mask_to_spike_batch(sample_spikes, sample_input_mask, mode="remove")
    masked_distractor_spikes = apply_input_mask_to_spike_batch(distractor_spikes, distractor_input_mask, mode="remove")
    prepare_network_state(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), dtype=sample_spikes.dtype, device=sample_spikes.device)
    current_time = 0
    readout_snapshot = None
    probe_l1_frames: list[torch.Tensor] = []
    probe_l2_frames: list[torch.Tensor] = []
    probe_l3_frames: list[torch.Tensor] = []

    def reset_decision_window() -> None:
        net.layer3.reset_decision_state()
        if bool(phase_reset):
            with torch.no_grad():
                net.layer3.v_mem.fill_(net.layer3.V_L)
                net.layer3.lateral_inh.reset_state(net.layer3.output_shape)

    def step_network(input_t: torch.Tensor, *, phase: str, phase_step: int, force_l3_time: int | None = None) -> None:
        nonlocal current_time, readout_snapshot
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())
        l3_time = current_time if force_l3_time is None else force_l3_time
        _, m3 = net.layer3.forward_step(
            s2_p,
            l3_time,
            training=False,
            monitor=(phase == "probe" and int(phase_step) == int(readout_step)),
            stsp_mode=stsp_mode,
        )
        if phase == "probe":
            probe_l1_frames.append(s1_p.detach().cpu().to(torch.float32))
            probe_l2_frames.append(s2.detach().cpu().to(torch.float32))
            probe_l3_frames.append(s2_p.detach().cpu().to(torch.float32))
        if phase == "probe" and int(phase_step) == int(readout_step):
            if "v_mem_snapshot" not in m3:
                raise RuntimeError("Layer-3 readout snapshot was not captured.")
            readout_snapshot = m3["v_mem_snapshot"].detach().cpu().to(torch.float32)
        current_time += 1

    with torch.no_grad():
        for t_step in range(int(masked_sample_spikes.shape[1])):
            step_network(masked_sample_spikes[:, t_step, ...], phase="sample", phase_step=t_step)
        for _ in range(int(delay1_steps)):
            step_network(zero_input, phase="delay1", phase_step=0)
        reset_decision_window()
        for t_step in range(int(masked_distractor_spikes.shape[1])):
            force_t = int(t_step) if bool(phase_reset) else None
            step_network(masked_distractor_spikes[:, t_step, ...], phase="distractor", phase_step=t_step, force_l3_time=force_t)
        for _ in range(int(delay2_steps)):
            step_network(zero_input, phase="delay2", phase_step=0)
        reset_decision_window()
        for t_step in range(int(probe_spikes.shape[1])):
            force_t = int(t_step) if bool(phase_reset) else None
            step_network(probe_spikes[:, t_step, ...], phase="probe", phase_step=t_step, force_l3_time=force_t)

    if readout_snapshot is None:
        raise RuntimeError("Requested probe readout snapshot was not produced.")
    flat_times = net.layer3.firing_times
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    prediction_probe = (min_indices // net.layer3.neurons_per_class).long()
    prediction_probe[~has_fired] = -1
    first_fire_t_probe = min_times.clone()
    first_fire_t_probe[~has_fired] = -1
    first_fire_t_probe = first_fire_t_probe.to(torch.long)

    return RolloutReadout(
        grouped_voltage=extract_grouped_voltage_vector(net, readout_snapshot),
        probe_l1_trace=torch.stack(probe_l1_frames, dim=0),
        probe_l2_trace=torch.stack(probe_l2_frames, dim=0),
        probe_l3_trace=torch.stack(probe_l3_frames, dim=0),
        prediction_probe=prediction_probe.detach().cpu().numpy().astype(np.int64, copy=False),
        first_fire_t_probe=first_fire_t_probe.detach().cpu().numpy().astype(np.int64, copy=False),
        readout_step=int(readout_step),
    )


def _raw_cosine_similarity(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor, eps: float = 1e-8) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), float(eps))
    return float(np.dot(aa, bb) / denom)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.size < 2 or yy.size < 2:
        return float("nan")
    if float(np.std(xx)) <= 1e-12 or float(np.std(yy)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def _condition_spec_table(include_union_condition: bool) -> dict[str, ConditionSpec]:
    table = {
        "full_dynamic_distractor": ConditionSpec("full_dynamic_distractor", "dynamic", None, None),
        "full_static_distractor": ConditionSpec("full_static_distractor", "static_frozen", None, None),
        "sample_remove_SPonly_dynamic": ConditionSpec("sample_remove_SPonly_dynamic", "dynamic", "sample_sp_only_mask", None),
        "sample_remove_SPonly_control_dynamic": ConditionSpec("sample_remove_SPonly_control_dynamic", "dynamic", "sample_sp_only_control_mask", None),
        "distractor_remove_DPonly_dynamic": ConditionSpec("distractor_remove_DPonly_dynamic", "dynamic", None, "distractor_dp_only_mask"),
        "distractor_remove_DPonly_control_dynamic": ConditionSpec("distractor_remove_DPonly_control_dynamic", "dynamic", None, "distractor_dp_only_control_mask"),
        "sample_remove_SDP_dynamic": ConditionSpec("sample_remove_SDP_dynamic", "dynamic", "sample_sdp_mask", None),
        "sample_remove_SDP_control_dynamic": ConditionSpec("sample_remove_SDP_control_dynamic", "dynamic", "sample_sdp_control_mask", None),
        "distractor_remove_SDP_dynamic": ConditionSpec("distractor_remove_SDP_dynamic", "dynamic", None, "distractor_sdp_mask"),
        "distractor_remove_SDP_control_dynamic": ConditionSpec("distractor_remove_SDP_control_dynamic", "dynamic", None, "distractor_sdp_control_mask"),
        "both_remove_SDP_dynamic": ConditionSpec("both_remove_SDP_dynamic", "dynamic", "sample_sdp_mask", "distractor_sdp_mask"),
        "both_remove_SDP_control_dynamic": ConditionSpec("both_remove_SDP_control_dynamic", "dynamic", "sample_sdp_control_mask", "distractor_sdp_control_mask"),
    }
    if bool(include_union_condition):
        table["remove_probe_relevant_union_dynamic"] = ConditionSpec(
            "remove_probe_relevant_union_dynamic",
            "dynamic",
            "sample_union_mask",
            "distractor_union_mask",
        )
        table["remove_probe_relevant_union_control_dynamic"] = ConditionSpec(
            "remove_probe_relevant_union_control_dynamic",
            "dynamic",
            "sample_union_control_mask",
            "distractor_union_control_mask",
        )
    return table


def _condition_order(include_union_condition: bool) -> tuple[str, ...]:
    base = (
        "full_dynamic_distractor",
        "full_static_distractor",
        "sample_remove_SPonly_dynamic",
        "sample_remove_SPonly_control_dynamic",
        "distractor_remove_DPonly_dynamic",
        "distractor_remove_DPonly_control_dynamic",
        "sample_remove_SDP_dynamic",
        "sample_remove_SDP_control_dynamic",
        "distractor_remove_SDP_dynamic",
        "distractor_remove_SDP_control_dynamic",
        "both_remove_SDP_dynamic",
        "both_remove_SDP_control_dynamic",
    )
    if not bool(include_union_condition):
        return base
    return base + ("remove_probe_relevant_union_dynamic", "remove_probe_relevant_union_control_dynamic")


def _build_condition_mask_batch(mask_records: Sequence[TripletMaskBundle], mask_key: str | None) -> torch.Tensor | None:
    if mask_key is None:
        return None
    stacked = np.stack([np.asarray(getattr(record, mask_key), dtype=bool) for record in mask_records], axis=0)
    return torch.as_tensor(stacked, dtype=torch.bool)


def _augment_triplet_specs_with_mask_metadata(
    df_triplets: pd.DataFrame,
    mask_records: Sequence[TripletMaskBundle],
    *,
    spec: ExperimentSpec,
    tau_ms: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for triplet_row in df_triplets.itertuples(index=False):
        mask_bundle = mask_records[int(triplet_row.triplet_id)]
        weighted = compute_time_weighted_structures(
            spec=spec,
            area_sp_only=float(mask_bundle.metadata["area_SPonly"]),
            area_dp_only=float(mask_bundle.metadata["area_DPonly"]),
            area_sdp=float(mask_bundle.metadata["area_SDP"]),
            tau_ms=float(tau_ms),
        )
        row = dict(triplet_row._asdict())
        row.update(mask_bundle.metadata)
        row.update(weighted)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)


def _select_case_triplets(df_triplets: pd.DataFrame, save_case_count: int) -> list[int]:
    if df_triplets.empty or int(save_case_count) <= 0:
        return []
    ordered = df_triplets.sort_values(["area_SDP", "area_probe_relevant_union", "X_SDP"], ascending=[False, False, False], kind="stable").reset_index(drop=True)
    take = min(int(save_case_count), len(ordered))
    positions = np.linspace(0, len(ordered) - 1, num=take).astype(np.int64)
    return ordered.iloc[sorted(dict.fromkeys(int(pos) for pos in positions.tolist()))]["triplet_id"].astype(int).tolist()


def summarize_distractor_pattern_chain_metrics(df_results: pd.DataFrame) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for condition_name, sub in df_results.groupby("condition", sort=False):
        rows.append(
            {
                "condition": str(condition_name),
                "n_records": int(len(sub)),
                "n_triplets": int(sub["triplet_id"].nunique()),
                "mean_DPI_L1": float(sub["DPI_L1"].mean()),
                "mean_DPI_L2": float(sub["DPI_L2"].mean()),
                "mean_DPI_L3": float(sub["DPI_L3"].mean()),
                "mean_DPI_final": float(sub["DPI_final"].mean()),
                "sem_DPI_L1": float(_sem(sub["DPI_L1"].to_numpy(dtype=np.float64))),
                "sem_DPI_L2": float(_sem(sub["DPI_L2"].to_numpy(dtype=np.float64))),
                "sem_DPI_L3": float(_sem(sub["DPI_L3"].to_numpy(dtype=np.float64))),
                "sem_DPI_final": float(_sem(sub["DPI_final"].to_numpy(dtype=np.float64))),
            }
        )

    contrast_pairs = [
        ("sample_remove_SPonly_dynamic", "sample_remove_SPonly_control_dynamic"),
        ("distractor_remove_DPonly_dynamic", "distractor_remove_DPonly_control_dynamic"),
        ("sample_remove_SDP_dynamic", "sample_remove_SDP_control_dynamic"),
        ("distractor_remove_SDP_dynamic", "distractor_remove_SDP_control_dynamic"),
        ("both_remove_SDP_dynamic", "both_remove_SDP_control_dynamic"),
        ("remove_probe_relevant_union_dynamic", "remove_probe_relevant_union_control_dynamic"),
    ]
    perturbation_vs_control: dict[str, object] = {}
    for perturbation_name, control_name in contrast_pairs:
        perturb = df_results[df_results["condition"] == perturbation_name].copy()
        control = df_results[df_results["condition"] == control_name].copy()
        if perturb.empty or control.empty:
            continue
        merged = perturb.merge(control, on="triplet_id", suffixes=("_perturb", "_control"), how="inner")
        if merged.empty:
            continue
        perturbation_vs_control[f"{perturbation_name}__minus__{control_name}"] = {
            metric_name: {
                "mean_difference": float(
                    np.mean(
                        merged[f"{metric_name}_perturb"].to_numpy(dtype=np.float64)
                        - merged[f"{metric_name}_control"].to_numpy(dtype=np.float64)
                    )
                ),
                "n_triplets": int(len(merged)),
            }
            for metric_name in ("DPI_L1", "DPI_L2", "DPI_L3", "DPI_final")
        }

    dynamic_only = df_results[
        df_results["condition"].isin(
            [
                "sample_remove_SPonly_dynamic",
                "distractor_remove_DPonly_dynamic",
                "sample_remove_SDP_dynamic",
                "distractor_remove_SDP_dynamic",
                "both_remove_SDP_dynamic",
                "remove_probe_relevant_union_dynamic",
            ]
        )
    ].copy()
    strength_correlations: dict[str, object] = {}
    if not dynamic_only.empty:
        for strength_name in ("area_SPonly", "area_DPonly", "area_SDP", "X_SP", "X_DP", "X_SDP"):
            x = dynamic_only[strength_name].to_numpy(dtype=np.float64)
            y = dynamic_only["DPI_L3"].to_numpy(dtype=np.float64)
            strength_correlations[strength_name] = {
                "pearson_r_with_DPI_L3": _safe_corr(x, y),
                "n_records": int(len(dynamic_only)),
            }

    scatter_rows: list[dict[str, object]] = []
    for condition_name in MAIN_DYNAMIC_CONDITIONS:
        subset = df_results[df_results["condition"] == condition_name].copy()
        if subset.empty:
            continue
        scatter_rows.append(
            {
                "condition": str(condition_name),
                "n_triplets": int(len(subset)),
                "pearson_r_DPI_L3_vs_DPI_final": _safe_corr(
                    subset["DPI_L3"].to_numpy(dtype=np.float64),
                    subset["DPI_final"].to_numpy(dtype=np.float64),
                ),
            }
        )

    ranking_by_mean_dpi_l3 = sorted(rows, key=lambda item: float(item["mean_DPI_L3"]))
    return {
        "overall": {
            "n_records": int(len(df_results)),
            "n_triplets": int(df_results["triplet_id"].nunique()),
            "n_probes": int(df_results["probe_id"].nunique()),
        },
        "condition_means": rows,
        "ranking_by_mean_DPI_L3": ranking_by_mean_dpi_l3,
        "perturbation_vs_control": perturbation_vs_control,
        "pair_scatter": scatter_rows,
        "strength_correlations": strength_correlations,
        "assumptions": {
            "grouped_voltage": "layer3.get_grouped_voltage(v_mem_snapshot).mean(-1)",
            "pattern_normalization": "(x - mean(x)) / (||x - mean(x)||_2 + eps)",
            "L1_trace": "probe-phase pooled layer1 output s1_p",
            "L2_trace": "probe-phase layer2 output s2",
            "L3_trace": "probe-phase layer3 input s2_p",
            "probe_perturbation": "disabled; only sample-side and distractor-side perturbation are applied",
            "time_weighting": "exponential decay to probe onset with X_SDP=(w_S*w_D)*area_SDP",
            "sample_to_probe_gap_ms": "delay1_ms + distractor_ms + delay2_ms",
            "distractor_to_probe_gap_ms": "delay2_ms",
        },
    }


def plot_input_perturbation_cases(
    *,
    triplet_ids: Sequence[int],
    df_triplets: pd.DataFrame,
    images: torch.Tensor,
    mask_records: Sequence[TripletMaskBundle],
) -> plt.Figure:
    apply_publication_style()
    n_cases = max(1, len(triplet_ids))
    fig, axes = plt.subplots(n_cases, 12, figsize=case_grid_figsize(n_cases, width=24.0, row_height=2.7), squeeze=False)
    for row_idx, triplet_id in enumerate(triplet_ids):
        triplet_row = df_triplets[df_triplets["triplet_id"] == int(triplet_id)].iloc[0]
        record = mask_records[int(triplet_id)]
        sample_image = images[int(triplet_row.sample_id)]
        distractor_image = images[int(triplet_row.distractor_id)]
        probe_image = images[int(triplet_row.probe_id)]
        sample_minus_sp = apply_ablation(sample_image, record.sample_sp_only_mask)
        distractor_minus_dp = apply_ablation(distractor_image, record.distractor_dp_only_mask)
        sample_minus_sdp = apply_ablation(sample_image, record.sample_sdp_mask)
        distractor_minus_sdp = apply_ablation(distractor_image, record.distractor_sdp_mask)
        panels = [
            ("sample", sample_image[0].numpy(), "gray"),
            ("distractor", distractor_image[0].numpy(), "gray"),
            ("probe", probe_image[0].numpy(), "gray"),
            ("SP-only", record.sp_only_mask.astype(np.float32), "magma"),
            ("SP ctrl", record.sample_sp_only_control_mask.astype(np.float32), "viridis"),
            ("sample - SP", sample_minus_sp[0].numpy(), "gray"),
            ("DP-only", record.dp_only_mask.astype(np.float32), "magma"),
            ("DP ctrl", record.distractor_dp_only_control_mask.astype(np.float32), "viridis"),
            ("distr - DP", distractor_minus_dp[0].numpy(), "gray"),
            ("SDP", record.sdp_mask.astype(np.float32), "magma"),
            ("sample - SDP", sample_minus_sdp[0].numpy(), "gray"),
            ("distr - SDP", distractor_minus_sdp[0].numpy(), "gray"),
        ]
        for col_idx, (title, panel, cmap) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(
                    f"triplet {int(triplet_id)}\n"
                    f"s={int(triplet_row.sample_label)} "
                    f"d={int(triplet_row.distractor_label)} "
                    f"p={int(triplet_row.probe_label)}"
                )
    fig.tight_layout()
    return fig


def plot_trace_pattern_similarity(df_results: pd.DataFrame, trace_payload: Mapping[str, np.ndarray]) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL, sharex=True, sharey=True)
    layer_specs = (
        ("L1", "S_dyn_L1", "S_sta_L1"),
        ("L2", "S_dyn_L2", "S_sta_L2"),
        ("L3", "S_dyn_L3", "S_sta_L3"),
    )
    for ax, (layer_name, dyn_key, sta_key) in zip(axes, layer_specs):
        dyn_traces = np.asarray(trace_payload[dyn_key], dtype=np.float64)
        sta_traces = np.asarray(trace_payload[sta_key], dtype=np.float64)
        if dyn_traces.size <= 0:
            continue
        time_axis = np.arange(dyn_traces.shape[1], dtype=np.int64)
        for condition_name in MAIN_DYNAMIC_CONDITIONS:
            cond_ids = df_results[df_results["condition"] == condition_name]["record_id"].to_numpy(dtype=np.int64, copy=False)
            if cond_ids.size <= 0:
                continue
            dyn_sub = dyn_traces[cond_ids]
            sta_sub = sta_traces[cond_ids]
            dyn_mean = dyn_sub.mean(axis=0)
            sta_mean = sta_sub.mean(axis=0)
            dyn_sem = np.zeros_like(dyn_mean) if dyn_sub.shape[0] <= 1 else dyn_sub.std(axis=0, ddof=1) / np.sqrt(dyn_sub.shape[0])
            sta_sem = np.zeros_like(sta_mean) if sta_sub.shape[0] <= 1 else sta_sub.std(axis=0, ddof=1) / np.sqrt(sta_sub.shape[0])
            color = CONDITION_COLORS[condition_name]
            ax.plot(time_axis, dyn_mean, color=color, linewidth=LINE_WIDTH_PRIMARY, label=f"{condition_name} -> dynamic")
            ax.fill_between(time_axis, dyn_mean - dyn_sem, dyn_mean + dyn_sem, color=color, alpha=ALPHA_FILL)
            ax.plot(time_axis, sta_mean, color=color, linewidth=1.6, linestyle="--", label=f"{condition_name} -> static")
            ax.fill_between(time_axis, sta_mean - sta_sem, sta_mean + sta_sem, color=color, alpha=ALPHA_FILL_LIGHT)
        ax.set_title(layer_name)
        ax.set_xlabel("Probe time step")
        ax.grid(alpha=GRID_ALPHA)
    axes[0].set_ylabel("Centered cosine similarity")
    handles, labels = axes[0].get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    apply_standard_legend(axes[0], handles=list(dedup.values()), labels=list(dedup.keys()), compact=True)
    fig.tight_layout()
    return fig


def plot_dpi_by_layer(df_results: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    x = np.arange(len(LAYER_ORDER), dtype=np.float64)
    plotted_conditions = [cond for cond in MAIN_DYNAMIC_CONDITIONS if cond in set(df_results["condition"].tolist())]
    width = 0.82 / max(1, len(plotted_conditions))
    offsets = np.linspace(-0.41 + width / 2.0, 0.41 - width / 2.0, num=max(1, len(plotted_conditions)))
    for offset, condition_name in zip(offsets, plotted_conditions):
        subset = df_results[df_results["condition"] == condition_name].copy()
        if subset.empty:
            continue
        means = np.asarray(
            [
                subset["DPI_L1"].mean(),
                subset["DPI_L2"].mean(),
                subset["DPI_L3"].mean(),
                subset["DPI_final"].mean(),
            ],
            dtype=np.float64,
        )
        sems = np.asarray(
            [
                _sem(subset["DPI_L1"].to_numpy(dtype=np.float64)),
                _sem(subset["DPI_L2"].to_numpy(dtype=np.float64)),
                _sem(subset["DPI_L3"].to_numpy(dtype=np.float64)),
                _sem(subset["DPI_final"].to_numpy(dtype=np.float64)),
            ],
            dtype=np.float64,
        )
        ax.bar(x + float(offset), means, width=width, color=CONDITION_COLORS[condition_name], alpha=ALPHA_BAR, label=condition_name)
        ax.errorbar(x + float(offset), means, yerr=sems, fmt="none", ecolor="black", elinewidth=LINE_WIDTH_REFERENCE, capsize=3)
    ax.axhline(0.0, color="#333333", linewidth=LINE_WIDTH_REFERENCE, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_ORDER)
    ax.set_ylabel("DPI")
    ax.grid(alpha=GRID_ALPHA, axis="y")
    apply_standard_legend(ax, compact=True)
    fig.tight_layout()
    return fig


def plot_triplet_scatter(df_results: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL)
    ax_left, ax_right = axes
    for condition_name in MAIN_DYNAMIC_CONDITIONS:
        subset = df_results[df_results["condition"] == condition_name].copy()
        if subset.empty:
            continue
        color = CONDITION_COLORS[condition_name]
        x = subset["DPI_L3"].to_numpy(dtype=np.float64)
        y = subset["DPI_final"].to_numpy(dtype=np.float64)
        ax_left.scatter(x, y, color=color, alpha=ALPHA_SCATTER, label=condition_name)
        if x.size >= 2 and float(np.std(x)) > 1e-12:
            slope, intercept = np.polyfit(x, y, deg=1)
            x_line = np.linspace(float(np.min(x)), float(np.max(x)), num=64)
            ax_left.plot(x_line, slope * x_line + intercept, color=color, linewidth=1.4)
        xs = subset["X_SDP"].to_numpy(dtype=np.float64)
        ys = subset["DPI_L3"].to_numpy(dtype=np.float64)
        ax_right.scatter(xs, ys, color=color, alpha=ALPHA_SCATTER, label=condition_name)
    ax_left.axhline(0.0, color="#333333", linewidth=LINE_WIDTH_REFERENCE, linestyle=":")
    ax_left.axvline(0.0, color="#333333", linewidth=LINE_WIDTH_REFERENCE, linestyle=":")
    ax_left.set_xlabel("DPI_L3")
    ax_left.set_ylabel("DPI_final")
    ax_left.set_title("DPI_L3 vs DPI_final")
    ax_left.grid(alpha=GRID_ALPHA)
    ax_right.axhline(0.0, color="#333333", linewidth=LINE_WIDTH_REFERENCE, linestyle=":")
    ax_right.set_xlabel("X_SDP")
    ax_right.set_ylabel("DPI_L3")
    ax_right.set_title("Weighted SDP strength vs DPI_L3")
    ax_right.grid(alpha=GRID_ALPHA)
    apply_standard_legend(ax_left, compact=True)
    fig.tight_layout()
    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distractor input causal perturbation experiment.")
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--delay1-ms", type=float, default=DEFAULT_DELAY1_MS)
    parser.add_argument("--distractor-ms", type=float, default=DEFAULT_DISTRACTOR_MS)
    parser.add_argument("--delay2-ms", type=float, default=DEFAULT_DELAY2_MS)
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    parser.add_argument("--tau-ms", type=float, default=DEFAULT_TAU_MS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--samples-per-probe", type=int, default=DEFAULT_SAMPLES_PER_PROBE)
    parser.add_argument("--max-triplets", type=int, default=DEFAULT_MAX_TRIPLETS)
    parser.add_argument("--num-sim-bins", type=int, default=DEFAULT_NUM_SIM_BINS)
    parser.add_argument("--foreground-threshold", type=float, default=DEFAULT_FOREGROUND_THRESHOLD)
    parser.add_argument("--use-dilated-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dilation-radius", type=int, default=DEFAULT_DILATION_RADIUS)
    parser.add_argument("--save-case-count", type=int, default=DEFAULT_SAVE_CASE_COUNT)
    parser.add_argument("--num-control-candidates", type=int, default=DEFAULT_NUM_CONTROL_CANDIDATES)
    parser.add_argument("--include-union-condition", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.max_probes) <= 0:
        raise ValueError("--max-probes must be positive.")
    if int(args.samples_per_probe) <= 0:
        raise ValueError("--samples-per-probe must be positive.")
    if int(args.max_triplets) <= 0:
        raise ValueError("--max-triplets must be positive.")
    if int(args.num_sim_bins) <= 0:
        raise ValueError("--num-sim-bins must be positive.")
    if int(args.save_case_count) < 0:
        raise ValueError("--save-case-count must be non-negative.")
    if int(args.num_control_candidates) <= 0:
        raise ValueError("--num-control-candidates must be positive.")
    if float(args.tau_ms) <= 0.0:
        raise ValueError("--tau-ms must be positive.")

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=float(args.sample_ms),
        delay1_ms=float(args.delay1_ms),
        distractor_ms=float(args.distractor_ms),
        delay2_ms=float(args.delay2_ms),
        probe_ms=float(args.probe_ms),
        phase_reset=True,
    )
    for name, steps in (
        ("sample", spec.sample_steps),
        ("delay1", spec.delay1_steps),
        ("distractor", spec.distractor_steps),
        ("delay2", spec.delay2_steps),
        ("probe", spec.probe_steps),
    ):
        if int(steps) <= 0:
            raise ValueError(f"{name} steps must resolve to a positive integer.")

    layout = prepare_result_layout(args.output_dir)
    result_root = layout.root
    output_dir = layout.data_dir
    figures_dir = layout.figure_dir
    logs_dir = layout.log_dir

    dataset = _load_dataset(dataset_root=args.dataset_root, split=args.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    num_classes = int(len(np.unique(labels)))
    class_index = build_class_index(dataset, num_classes=num_classes)
    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(float(args.sample_ms), float(args.distractor_ms), float(args.probe_ms)),
    )
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )

    df_triplets = build_triplet_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=int(args.max_probes),
        samples_per_probe=int(args.samples_per_probe),
        num_bins=int(args.num_sim_bins),
        max_triplets=int(args.max_triplets),
        seed=int(args.seed),
    )

    mask_records: list[TripletMaskBundle] = []
    for triplet_row in df_triplets.itertuples(index=False):
        triplet_id = int(triplet_row.triplet_id)
        mask_records.append(
            build_probe_relevant_masks_for_triplet(
                sample_image=images[int(triplet_row.sample_id)],
                distractor_image=images[int(triplet_row.distractor_id)],
                probe_image=images[int(triplet_row.probe_id)],
                foreground_threshold=float(args.foreground_threshold),
                use_dilated_overlap=bool(args.use_dilated_overlap),
                dilation_radius=int(args.dilation_radius),
                seed=mix_seed(int(args.seed), triplet_id, int(triplet_row.sample_id), int(triplet_row.distractor_id), int(triplet_row.probe_id)),
                num_control_candidates=int(args.num_control_candidates),
            )
        )
    df_triplets = _augment_triplet_specs_with_mask_metadata(df_triplets, mask_records, spec=spec, tau_ms=float(args.tau_ms))

    condition_specs = _condition_spec_table(include_union_condition=bool(args.include_union_condition))
    condition_order = _condition_order(include_union_condition=bool(args.include_union_condition))

    results_rows: list[dict[str, object]] = []
    trace_records: dict[str, list[np.ndarray | int | str]] = {
        "record_id": [],
        "triplet_id": [],
        "condition_name": [],
        "S_dyn_L1": [],
        "S_sta_L1": [],
        "S_dyn_L2": [],
        "S_sta_L2": [],
        "S_dyn_L3": [],
        "S_sta_L3": [],
    }
    final_records: dict[str, list[np.ndarray | float | int | str]] = {
        "record_id": [],
        "triplet_id": [],
        "condition_name": [],
        "V_cond": [],
        "V_full_dyn": [],
        "V_full_sta": [],
        "S_dyn_final": [],
        "S_sta_final": [],
        "DPI_final": [],
        "S_dyn_final_raw": [],
        "S_sta_final_raw": [],
    }

    batch_starts = range(0, len(df_triplets), int(args.batch_size))
    for batch_start in tqdm(batch_starts, total=math.ceil(len(df_triplets) / int(args.batch_size)), desc="Running distractor pattern chain"):
        batch_df = df_triplets.iloc[batch_start : batch_start + int(args.batch_size)].copy().reset_index(drop=True)
        sample_spikes, distractor_spikes, probe_spikes = prepare_triplet_spike_batch(
            images=images,
            batch_df=batch_df,
            encoder=encoder,
            spec=spec,
            device=device,
        )
        batch_triplet_ids = batch_df["triplet_id"].astype(int).tolist()
        batch_masks = [mask_records[triplet_id] for triplet_id in batch_triplet_ids]
        rollout_outputs: dict[str, RolloutReadout] = {
            "full_dynamic_distractor": run_overlap_perturbed_distractor_task(
                net=net,
                sample_spikes=sample_spikes,
                distractor_spikes=distractor_spikes,
                probe_spikes=probe_spikes,
                delay1_steps=spec.delay1_steps,
                delay2_steps=spec.delay2_steps,
                stsp_mode=condition_specs["full_dynamic_distractor"].stsp_mode,
                readout_step=readout_step,
                phase_reset=spec.phase_reset,
            ),
            "full_static_distractor": run_overlap_perturbed_distractor_task(
                net=net,
                sample_spikes=sample_spikes,
                distractor_spikes=distractor_spikes,
                probe_spikes=probe_spikes,
                delay1_steps=spec.delay1_steps,
                delay2_steps=spec.delay2_steps,
                stsp_mode=condition_specs["full_static_distractor"].stsp_mode,
                readout_step=readout_step,
                phase_reset=spec.phase_reset,
            ),
        }
        for condition_name in condition_order:
            if condition_name in rollout_outputs:
                continue
            spec_row = condition_specs[condition_name]
            sample_mask = _build_condition_mask_batch(batch_masks, spec_row.sample_mask_key)
            distractor_mask = _build_condition_mask_batch(batch_masks, spec_row.distractor_mask_key)
            rollout_outputs[condition_name] = run_overlap_perturbed_distractor_task(
                net=net,
                sample_spikes=sample_spikes,
                distractor_spikes=distractor_spikes,
                probe_spikes=probe_spikes,
                delay1_steps=spec.delay1_steps,
                delay2_steps=spec.delay2_steps,
                stsp_mode=spec_row.stsp_mode,
                readout_step=readout_step,
                sample_input_mask=None if sample_mask is None else sample_mask.to(device=device),
                distractor_input_mask=None if distractor_mask is None else distractor_mask.to(device=device),
                phase_reset=spec.phase_reset,
            )

        full_dynamic = rollout_outputs["full_dynamic_distractor"]
        full_static = rollout_outputs["full_static_distractor"]
        for batch_idx, triplet_row in enumerate(batch_df.itertuples(index=False)):
            triplet_dict = dict(triplet_row._asdict())
            ref_l1_dyn = full_dynamic.probe_l1_trace[:, batch_idx].numpy()
            ref_l2_dyn = full_dynamic.probe_l2_trace[:, batch_idx].numpy()
            ref_l3_dyn = full_dynamic.probe_l3_trace[:, batch_idx].numpy()
            ref_l1_sta = full_static.probe_l1_trace[:, batch_idx].numpy()
            ref_l2_sta = full_static.probe_l2_trace[:, batch_idx].numpy()
            ref_l3_sta = full_static.probe_l3_trace[:, batch_idx].numpy()
            v_full_dyn = np.asarray(full_dynamic.grouped_voltage[batch_idx], dtype=np.float64)
            v_full_sta = np.asarray(full_static.grouped_voltage[batch_idx], dtype=np.float64)
            for condition_name in condition_order:
                rollout = rollout_outputs[condition_name]
                cond_l1 = rollout.probe_l1_trace[:, batch_idx].numpy()
                cond_l2 = rollout.probe_l2_trace[:, batch_idx].numpy()
                cond_l3 = rollout.probe_l3_trace[:, batch_idx].numpy()
                s_dyn_l1, s_sta_l1, dpi_l1 = compute_trace_pattern_similarity(cond_l1, ref_l1_dyn, ref_l1_sta)
                s_dyn_l2, s_sta_l2, dpi_l2 = compute_trace_pattern_similarity(cond_l2, ref_l2_dyn, ref_l2_sta)
                s_dyn_l3, s_sta_l3, dpi_l3 = compute_trace_pattern_similarity(cond_l3, ref_l3_dyn, ref_l3_sta)
                v_cond = np.asarray(rollout.grouped_voltage[batch_idx], dtype=np.float64)
                s_dyn_final, s_sta_final, dpi_final = compute_final_pattern_similarity(v_cond, v_full_dyn, v_full_sta)
                s_dyn_final_raw = _raw_cosine_similarity(v_cond, v_full_dyn)
                s_sta_final_raw = _raw_cosine_similarity(v_cond, v_full_sta)
                record_id = len(results_rows)
                results_rows.append(
                    {
                        "record_id": int(record_id),
                        "condition": str(condition_name),
                        "readout_step": int(rollout.readout_step),
                        "prediction_probe": int(rollout.prediction_probe[batch_idx]),
                        "first_fire_t_probe": int(rollout.first_fire_t_probe[batch_idx]),
                        "DPI_L1": float(dpi_l1),
                        "DPI_L2": float(dpi_l2),
                        "DPI_L3": float(dpi_l3),
                        "DPI_final": float(dpi_final),
                        "S_dyn_final": float(s_dyn_final),
                        "S_sta_final": float(s_sta_final),
                        "S_dyn_final_raw": float(s_dyn_final_raw),
                        "S_sta_final_raw": float(s_sta_final_raw),
                        **triplet_dict,
                    }
                )
                trace_records["record_id"].append(int(record_id))
                trace_records["triplet_id"].append(int(triplet_row.triplet_id))
                trace_records["condition_name"].append(str(condition_name))
                trace_records["S_dyn_L1"].append(np.asarray(s_dyn_l1, dtype=np.float32))
                trace_records["S_sta_L1"].append(np.asarray(s_sta_l1, dtype=np.float32))
                trace_records["S_dyn_L2"].append(np.asarray(s_dyn_l2, dtype=np.float32))
                trace_records["S_sta_L2"].append(np.asarray(s_sta_l2, dtype=np.float32))
                trace_records["S_dyn_L3"].append(np.asarray(s_dyn_l3, dtype=np.float32))
                trace_records["S_sta_L3"].append(np.asarray(s_sta_l3, dtype=np.float32))
                final_records["record_id"].append(int(record_id))
                final_records["triplet_id"].append(int(triplet_row.triplet_id))
                final_records["condition_name"].append(str(condition_name))
                final_records["V_cond"].append(v_cond.astype(np.float32, copy=False))
                final_records["V_full_dyn"].append(v_full_dyn.astype(np.float32, copy=False))
                final_records["V_full_sta"].append(v_full_sta.astype(np.float32, copy=False))
                final_records["S_dyn_final"].append(float(s_dyn_final))
                final_records["S_sta_final"].append(float(s_sta_final))
                final_records["DPI_final"].append(float(dpi_final))
                final_records["S_dyn_final_raw"].append(float(s_dyn_final_raw))
                final_records["S_sta_final_raw"].append(float(s_sta_final_raw))

    df_results = pd.DataFrame(results_rows).sort_values(["record_id"], kind="stable").reset_index(drop=True)
    case_triplet_ids = _select_case_triplets(df_triplets, save_case_count=int(args.save_case_count))
    summary_metrics = summarize_distractor_pattern_chain_metrics(df_results)
    summary_metrics["case_triplet_ids"] = [int(triplet_id) for triplet_id in case_triplet_ids]
    summary_metrics["condition_order"] = list(condition_order)
    summary_metrics["readout_step"] = int(readout_step)
    summary_metrics["tau_ms"] = float(args.tau_ms)
    summary_metrics["time_weights"] = compute_time_weighted_structures(spec=spec, area_sp_only=1.0, area_dp_only=1.0, area_sdp=1.0, tau_ms=float(args.tau_ms))

    triplet_specs_csv = save_tidy_csv(df_triplets, output_dir / "triplet_specs.csv", sort_by=["triplet_id"])
    results_csv = save_tidy_csv(df_results, output_dir / "triplet_condition_pattern_results.csv", sort_by=["triplet_id", "condition"])

    trace_arrays = {
        "record_id": np.asarray(trace_records["record_id"], dtype=np.int64),
        "triplet_id": np.asarray(trace_records["triplet_id"], dtype=np.int64),
        "condition_name": np.asarray(trace_records["condition_name"]),
        "S_dyn_L1": np.stack(trace_records["S_dyn_L1"], axis=0) if trace_records["S_dyn_L1"] else np.zeros((0, spec.probe_steps), dtype=np.float32),
        "S_sta_L1": np.stack(trace_records["S_sta_L1"], axis=0) if trace_records["S_sta_L1"] else np.zeros((0, spec.probe_steps), dtype=np.float32),
        "S_dyn_L2": np.stack(trace_records["S_dyn_L2"], axis=0) if trace_records["S_dyn_L2"] else np.zeros((0, spec.probe_steps), dtype=np.float32),
        "S_sta_L2": np.stack(trace_records["S_sta_L2"], axis=0) if trace_records["S_sta_L2"] else np.zeros((0, spec.probe_steps), dtype=np.float32),
        "S_dyn_L3": np.stack(trace_records["S_dyn_L3"], axis=0) if trace_records["S_dyn_L3"] else np.zeros((0, spec.probe_steps), dtype=np.float32),
        "S_sta_L3": np.stack(trace_records["S_sta_L3"], axis=0) if trace_records["S_sta_L3"] else np.zeros((0, spec.probe_steps), dtype=np.float32),
    }
    trace_npz = output_dir / "triplet_trace_similarity.npz"
    np.savez_compressed(trace_npz, **trace_arrays)

    final_arrays = {
        "record_id": np.asarray(final_records["record_id"], dtype=np.int64),
        "triplet_id": np.asarray(final_records["triplet_id"], dtype=np.int64),
        "condition_name": np.asarray(final_records["condition_name"]),
        "V_cond": np.stack(final_records["V_cond"], axis=0) if final_records["V_cond"] else np.zeros((0, num_classes), dtype=np.float32),
        "V_full_dyn": np.stack(final_records["V_full_dyn"], axis=0) if final_records["V_full_dyn"] else np.zeros((0, num_classes), dtype=np.float32),
        "V_full_sta": np.stack(final_records["V_full_sta"], axis=0) if final_records["V_full_sta"] else np.zeros((0, num_classes), dtype=np.float32),
        "S_dyn_final": np.asarray(final_records["S_dyn_final"], dtype=np.float32),
        "S_sta_final": np.asarray(final_records["S_sta_final"], dtype=np.float32),
        "DPI_final": np.asarray(final_records["DPI_final"], dtype=np.float32),
        "S_dyn_final_raw": np.asarray(final_records["S_dyn_final_raw"], dtype=np.float32),
        "S_sta_final_raw": np.asarray(final_records["S_sta_final_raw"], dtype=np.float32),
    }
    final_npz = output_dir / "triplet_final_vectors.npz"
    np.savez_compressed(final_npz, **final_arrays)

    summary_json = _save_json(summary_metrics, output_dir / "summary_metrics.json")

    fig1 = plot_input_perturbation_cases(triplet_ids=case_triplet_ids, df_triplets=df_triplets, images=images, mask_records=mask_records)
    fig1_paths = save_figure_all_formats(fig1, figures_dir / "figure_1_input_perturbation_cases")
    plt.close(fig1)

    fig2 = plot_trace_pattern_similarity(df_results, trace_arrays)
    fig2_paths = save_figure_all_formats(fig2, figures_dir / "figure_2_trace_pattern_similarity")
    plt.close(fig2)

    fig3 = plot_dpi_by_layer(df_results)
    fig3_paths = save_figure_all_formats(fig3, figures_dir / "figure_3_dpi_by_layer")
    plt.close(fig3)

    fig4 = plot_triplet_scatter(df_results)
    fig4_paths = save_figure_all_formats(fig4, figures_dir / "figure_4_triplet_scatter")
    plt.close(fig4)

    run_config_path = save_run_config(
        {
            "model_path": str(Path(args.model_path).resolve()),
            "config_argument": args.config,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "split": str(args.split),
            "output_dir": str(result_root.resolve()),
            "device": str(device),
            "seed": int(args.seed),
            "sample_ms": float(args.sample_ms),
            "delay1_ms": float(args.delay1_ms),
            "distractor_ms": float(args.distractor_ms),
            "delay2_ms": float(args.delay2_ms),
            "probe_ms": float(args.probe_ms),
            "tau_ms": float(args.tau_ms),
            "batch_size": int(args.batch_size),
            "max_probes": int(args.max_probes),
            "samples_per_probe": int(args.samples_per_probe),
            "max_triplets": int(args.max_triplets),
            "num_sim_bins": int(args.num_sim_bins),
            "foreground_threshold": float(args.foreground_threshold),
            "use_dilated_overlap": bool(args.use_dilated_overlap),
            "dilation_radius": int(args.dilation_radius),
            "save_case_count": int(args.save_case_count),
            "num_control_candidates": int(args.num_control_candidates),
            "include_union_condition": bool(args.include_union_condition),
            "readout_step": int(readout_step),
            "condition_order": list(condition_order),
            "assumptions": summary_metrics["assumptions"],
            "outputs": {
                "triplet_specs_csv": str(Path(triplet_specs_csv).resolve()),
                "triplet_condition_pattern_results_csv": str(Path(results_csv).resolve()),
                "triplet_trace_similarity_npz": str(trace_npz.resolve()),
                "triplet_final_vectors_npz": str(final_npz.resolve()),
                "summary_metrics_json": str(summary_json.resolve()),
                "figure_1_png": fig1_paths["png"],
                "figure_2_png": fig2_paths["png"],
                "figure_3_png": fig3_paths["png"],
                "figure_4_png": fig4_paths["png"],
            },
        },
        result_root,
    )
    summary_path = save_summary_json(
        {
            "experiment": "distractor_input_causal_perturbation_experiment",
            "pair_count": int(len(df_results)),
            "case_pair_count": int(len(case_pair_ids)),
            "mean_dpi_l3": float(df_results["DPI_L3"].mean()),
            "mean_dpi_final": float(df_results["DPI_final"].mean()),
            "artifacts": {
                "data_summary_metrics_json": str(summary_json.resolve()),
                "run_config_json": str(run_config_path.resolve()),
            },
        },
        result_root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=distractor_input_causal_perturbation_experiment",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"pairs={len(df_results)}",
            f"case_pairs={len(case_pair_ids)}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_path.resolve()}",
        ],
        logs_dir,
    )

    print("\n=== Distractor Input Causal Perturbation Experiment Summary ===")
    print(f"Triplets: {int(df_results['triplet_id'].nunique())}")
    print(f"Records: {int(len(df_results))}")
    for condition_name in MAIN_DYNAMIC_CONDITIONS:
        subset = df_results[df_results["condition"] == condition_name]
        if subset.empty:
            continue
        print(
            f"{condition_name}: "
            f"DPI_L1={float(subset['DPI_L1'].mean()):.4f}, "
            f"DPI_L2={float(subset['DPI_L2'].mean()):.4f}, "
            f"DPI_L3={float(subset['DPI_L3'].mean()):.4f}, "
            f"DPI_final={float(subset['DPI_final'].mean()):.4f}"
        )
    print(f"Triplet specs CSV: {triplet_specs_csv}")
    print(f"Triplet-condition CSV: {results_csv}")
    print(f"Trace NPZ: {trace_npz}")
    print(f"Final vectors NPZ: {final_npz}")
    print(f"Summary JSON: {summary_json}")
    print(f"Run config: {run_config_path}")


if __name__ == "__main__":
    main()
