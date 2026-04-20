from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.data.encoding import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.common.dataset import encode_images
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.seed import mix_seed
from src.experiments.distractor.shared.config import DEFAULT_TAU_MS
from src.experiments.distractor.shared.masking import (
    apply_input_mask_to_spike_batch,
    build_best_energy_matched_control_mask,
    dilate_mask,
    foreground_mask,
    mask_energy,
)
from src.experiments.distractor.shared.pair_sampling import (
    assign_bins_from_values,
    extract_grouped_voltage_vector,
    select_probe_ids_balanced,
    select_probe_samples_from_candidates,
)


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
    df_candidates["dp_bin"] = assign_bins_from_values(
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
        selected_samples = select_probe_samples_from_candidates(
            df_candidates=df_sample_candidates,
            samples_per_probe=int(samples_per_probe),
            num_bins=int(num_bins),
        )
        if selected_samples.empty:
            continue
        selected_samples = selected_samples.rename(columns={"similarity_public_or_initial": "sp_similarity"}).reset_index(drop=True)
        selected_samples["sp_bin"] = assign_bins_from_values(
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
        return dilate_mask(base, int(dilation_radius)) & np.asarray(phase_foreground, dtype=bool)
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
    return build_best_energy_matched_control_mask(
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
    sample_fg = foreground_mask(sample_image, threshold=foreground_threshold)
    distractor_fg = foreground_mask(distractor_image, threshold=foreground_threshold)
    probe_fg = foreground_mask(probe_image, threshold=foreground_threshold)

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
        "sample_SPonly_energy": float(mask_energy(sample_image, sample_sp_only)),
        "sample_SPonly_control_energy": float(mask_energy(sample_image, sample_sp_control)),
        "sample_SPonly_control_source": str(sample_sp_source),
        "sample_SPonly_control_energy_gap": float(sample_sp_gap),
        "sample_SPonly_empty": int(int(sample_sp_only.sum()) <= 0),
        "distractor_DPonly_area": int(distractor_dp_only.sum()),
        "distractor_DPonly_control_area": int(distractor_dp_control.sum()),
        "distractor_DPonly_energy": float(mask_energy(distractor_image, distractor_dp_only)),
        "distractor_DPonly_control_energy": float(mask_energy(distractor_image, distractor_dp_control)),
        "distractor_DPonly_control_source": str(distractor_dp_source),
        "distractor_DPonly_control_energy_gap": float(distractor_dp_gap),
        "distractor_DPonly_empty": int(int(distractor_dp_only.sum()) <= 0),
        "sample_SDP_area": int(sample_sdp.sum()),
        "sample_SDP_control_area": int(sample_sdp_control.sum()),
        "sample_SDP_energy": float(mask_energy(sample_image, sample_sdp)),
        "sample_SDP_control_energy": float(mask_energy(sample_image, sample_sdp_control)),
        "sample_SDP_control_source": str(sample_sdp_source),
        "sample_SDP_control_energy_gap": float(sample_sdp_gap),
        "sample_SDP_empty": int(int(sample_sdp.sum()) <= 0),
        "distractor_SDP_area": int(distractor_sdp.sum()),
        "distractor_SDP_control_area": int(distractor_sdp_control.sum()),
        "distractor_SDP_energy": float(mask_energy(distractor_image, distractor_sdp)),
        "distractor_SDP_control_energy": float(mask_energy(distractor_image, distractor_sdp_control)),
        "distractor_SDP_control_source": str(distractor_sdp_source),
        "distractor_SDP_control_energy_gap": float(distractor_sdp_gap),
        "distractor_SDP_empty": int(int(distractor_sdp.sum()) <= 0),
        "sample_union_area": int(sample_union.sum()),
        "sample_union_control_area": int(sample_union_control.sum()),
        "sample_union_energy": float(mask_energy(sample_image, sample_union)),
        "sample_union_control_energy": float(mask_energy(sample_image, sample_union_control)),
        "sample_union_control_source": str(sample_union_source),
        "sample_union_control_energy_gap": float(sample_union_gap),
        "distractor_union_area": int(distractor_union.sum()),
        "distractor_union_control_area": int(distractor_union_control.sum()),
        "distractor_union_energy": float(mask_energy(distractor_image, distractor_union)),
        "distractor_union_control_energy": float(mask_energy(distractor_image, distractor_union_control)),
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


__all__ = [
    "ConditionSpec",
    "DEFAULT_TAU_MS",
    "ExperimentSpec",
    "RolloutReadout",
    "TripletMaskBundle",
    "_augment_triplet_specs_with_mask_metadata",
    "_build_condition_mask_batch",
    "_load_dataset",
    "build_probe_relevant_masks_for_triplet",
    "build_triplet_specs",
    "compute_time_weighted_structures",
    "prepare_triplet_spike_batch",
    "run_overlap_perturbed_distractor_task",
]
