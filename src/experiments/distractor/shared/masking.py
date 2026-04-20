from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.experiments.common.input_masks import foreground_mask_from_image
from src.experiments.common.pattern_metrics import (
    compute_final_pattern_similarity as shared_compute_final_pattern_similarity,
    compute_trace_pattern_similarity as shared_compute_trace_pattern_similarity,
)
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.seed import mix_seed
from src.experiments.distractor.shared.pair_sampling import extract_grouped_voltage_vector


@dataclass(frozen=True)
class OverlapMaskBundle:
    overlap_mask: np.ndarray
    sample_overlap_mask: np.ndarray
    probe_overlap_mask: np.ndarray
    sample_nonoverlap_control_mask: np.ndarray
    probe_nonoverlap_control_mask: np.ndarray
    sample_foreground_mask: np.ndarray
    sample_nonoverlap_mask: np.ndarray
    probe_foreground_mask: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class RolloutReadout:
    grouped_voltage: np.ndarray
    probe_l1_trace: torch.Tensor
    probe_l2_trace: torch.Tensor
    probe_l3_trace: torch.Tensor
    prediction_probe: np.ndarray
    first_fire_t_probe: np.ndarray
    readout_step: int


def foreground_mask(image: torch.Tensor, threshold: float) -> np.ndarray:
    return foreground_mask_from_image(image, threshold=threshold)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any() or int(radius) <= 0:
        return mask_bool
    tensor = torch.as_tensor(mask_bool.astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = 2 * int(radius) + 1
    dilated = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(radius))
    return np.asarray(dilated.squeeze(0).squeeze(0).numpy() > 0.0, dtype=bool)


def mask_energy(image: torch.Tensor, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any():
        return 0.0
    arr = image.detach().cpu().to(torch.float32).abs().sum(dim=0).numpy()
    return float(arr[mask_bool].sum())


def _sample_area_matched_mask(pool_mask: np.ndarray, area: int, rng: np.random.Generator) -> np.ndarray:
    pool_bool = np.asarray(pool_mask, dtype=bool)
    if int(area) <= 0:
        return np.zeros_like(pool_bool, dtype=bool)
    available = np.flatnonzero(pool_bool.reshape(-1))
    if available.size < int(area):
        raise ValueError("Pool does not have enough area to sample a matched mask.")
    chosen = rng.choice(available, size=int(area), replace=False)
    out = np.zeros(pool_bool.size, dtype=bool)
    out[chosen] = True
    return out.reshape(pool_bool.shape)


def build_best_energy_matched_control_mask(
    *,
    image: torch.Tensor,
    reference_mask: np.ndarray,
    preferred_pool_mask: np.ndarray,
    fallback_pool_mask: np.ndarray,
    rng: np.random.Generator,
    num_candidates: int,
) -> tuple[np.ndarray, str, float]:
    reference_bool = np.asarray(reference_mask, dtype=bool)
    area = int(reference_bool.sum())
    if area <= 0:
        return np.zeros_like(reference_bool, dtype=bool), "empty_reference", 0.0
    preferred_pool = np.asarray(preferred_pool_mask, dtype=bool)
    fallback_pool = np.asarray(fallback_pool_mask, dtype=bool)
    if int(preferred_pool.sum()) >= area:
        pool = preferred_pool
        source = "foreground_nonoverlap"
    elif int(fallback_pool.sum()) >= area:
        pool = fallback_pool
        source = "full_nonoverlap_fallback"
    else:
        pool = np.ones_like(reference_bool, dtype=bool)
        source = "whole_image_fallback"

    reference_energy = mask_energy(image, reference_bool)
    best_mask = None
    best_gap = None
    for _ in range(max(1, int(num_candidates))):
        candidate = _sample_area_matched_mask(pool, area=area, rng=rng)
        energy_gap = abs(mask_energy(image, candidate) - reference_energy)
        if best_gap is None or energy_gap < best_gap:
            best_gap = float(energy_gap)
            best_mask = candidate
            if energy_gap <= 0.0:
                break
    if best_mask is None:
        raise RuntimeError("Failed to construct a matched control mask.")
    return np.asarray(best_mask, dtype=bool), str(source), float(best_gap if best_gap is not None else 0.0)


def build_overlap_masks_for_pair(
    sample_image: torch.Tensor,
    probe_image: torch.Tensor,
    *,
    foreground_threshold: float,
    use_dilated_overlap: bool,
    dilation_radius: int,
    seed: int,
    num_control_candidates: int,
) -> OverlapMaskBundle:
    sample_fg = foreground_mask(sample_image, threshold=foreground_threshold)
    probe_fg = foreground_mask(probe_image, threshold=foreground_threshold)
    base_overlap = sample_fg & probe_fg
    if bool(use_dilated_overlap) and int(dilation_radius) > 0 and base_overlap.any():
        sample_overlap = dilate_mask(base_overlap, int(dilation_radius)) & sample_fg
        probe_overlap = dilate_mask(base_overlap, int(dilation_radius)) & probe_fg
    else:
        sample_overlap = base_overlap.copy()
        probe_overlap = base_overlap.copy()
    sample_nonoverlap = sample_fg & ~sample_overlap

    sample_rng = np.random.default_rng(mix_seed(seed, 101, int(sample_image.numel())))
    probe_rng = np.random.default_rng(mix_seed(seed, 202, int(probe_image.numel())))
    sample_control, sample_control_source, sample_gap = build_best_energy_matched_control_mask(
        image=sample_image,
        reference_mask=sample_overlap,
        preferred_pool_mask=sample_fg & ~sample_overlap,
        fallback_pool_mask=~sample_overlap,
        rng=sample_rng,
        num_candidates=int(num_control_candidates),
    )
    probe_control, probe_control_source, probe_gap = build_best_energy_matched_control_mask(
        image=probe_image,
        reference_mask=probe_overlap,
        preferred_pool_mask=probe_fg & ~probe_overlap,
        fallback_pool_mask=~probe_overlap,
        rng=probe_rng,
        num_candidates=int(num_control_candidates),
    )

    metadata = {
        "foreground_threshold": float(foreground_threshold),
        "use_dilated_overlap": int(bool(use_dilated_overlap)),
        "dilation_radius": int(dilation_radius),
        "base_overlap_area": int(base_overlap.sum()),
        "sample_overlap_area": int(sample_overlap.sum()),
        "probe_overlap_area": int(probe_overlap.sum()),
        "sample_control_area": int(sample_control.sum()),
        "probe_control_area": int(probe_control.sum()),
        "sample_foreground_area": int(sample_fg.sum()),
        "sample_nonoverlap_area": int(sample_nonoverlap.sum()),
        "probe_foreground_area": int(probe_fg.sum()),
        "sample_control_source": str(sample_control_source),
        "probe_control_source": str(probe_control_source),
        "sample_overlap_energy": float(mask_energy(sample_image, sample_overlap)),
        "sample_nonoverlap_energy": float(mask_energy(sample_image, sample_nonoverlap)),
        "sample_control_energy": float(mask_energy(sample_image, sample_control)),
        "probe_overlap_energy": float(mask_energy(probe_image, probe_overlap)),
        "probe_control_energy": float(mask_energy(probe_image, probe_control)),
        "sample_control_energy_gap": float(sample_gap),
        "probe_control_energy_gap": float(probe_gap),
    }
    return OverlapMaskBundle(
        overlap_mask=np.asarray(base_overlap, dtype=bool),
        sample_overlap_mask=np.asarray(sample_overlap, dtype=bool),
        probe_overlap_mask=np.asarray(probe_overlap, dtype=bool),
        sample_nonoverlap_control_mask=np.asarray(sample_control, dtype=bool),
        probe_nonoverlap_control_mask=np.asarray(probe_control, dtype=bool),
        sample_foreground_mask=np.asarray(sample_fg, dtype=bool),
        sample_nonoverlap_mask=np.asarray(sample_nonoverlap, dtype=bool),
        probe_foreground_mask=np.asarray(probe_fg, dtype=bool),
        metadata=metadata,
    )


def apply_input_mask_to_spike_batch(
    spike_batch: torch.Tensor,
    spatial_mask: torch.Tensor | np.ndarray | None,
    *,
    mode: str = "remove",
) -> torch.Tensor:
    if spatial_mask is None:
        return spike_batch
    if spike_batch.ndim != 5:
        raise ValueError(f"Expected spike_batch shape [B, T, C, H, W], got {tuple(spike_batch.shape)}")
    if str(mode).strip().lower() != "remove":
        raise ValueError(f"Unsupported mask application mode: {mode}")
    mask = torch.as_tensor(spatial_mask, dtype=torch.bool, device=spike_batch.device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Expected mask shape [H, W] or [B, H, W], got {tuple(mask.shape)}")
    if mask.shape[0] not in {1, spike_batch.shape[0]}:
        raise ValueError("Mask batch dimension must be 1 or match spike batch size.")
    while mask.ndim < spike_batch.ndim:
        mask = mask.unsqueeze(1)
    return spike_batch.masked_fill(mask, 0.0)


def run_overlap_perturbed_dms(
    net,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    *,
    delay_steps: int,
    stsp_mode: str,
    readout_step: int,
    sample_input_mask: torch.Tensor | np.ndarray | None = None,
) -> RolloutReadout:
    if sample_spikes.ndim != 5 or probe_spikes.ndim != 5:
        raise ValueError("sample_spikes and probe_spikes must have shape [B, T, C, H, W]")
    batch_size, _, channels, height, width = sample_spikes.shape
    if int(probe_spikes.shape[0]) != int(batch_size):
        raise ValueError("sample_spikes and probe_spikes must share batch size")

    masked_sample_spikes = apply_input_mask_to_spike_batch(sample_spikes, sample_input_mask, mode="remove")
    prepare_network_state(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), dtype=sample_spikes.dtype, device=sample_spikes.device)
    current_time = 0
    readout_snapshot = None
    probe_l1_frames: list[torch.Tensor] = []
    probe_l2_frames: list[torch.Tensor] = []
    probe_l3_frames: list[torch.Tensor] = []

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
        for _ in range(int(delay_steps)):
            step_network(zero_input, phase="delay", phase_step=0)
        net.layer3.reset_decision_state()
        net.layer3.v_mem.fill_(net.layer3.V_L)
        net.layer3.lateral_inh.reset_state(net.layer3.output_shape)
        for t_step in range(int(probe_spikes.shape[1])):
            step_network(probe_spikes[:, t_step, ...], phase="probe", phase_step=t_step, force_l3_time=t_step)

    if readout_snapshot is None:
        raise RuntimeError("Requested probe readout snapshot was not produced.")
    if not probe_l1_frames or not probe_l2_frames or not probe_l3_frames:
        raise RuntimeError("Probe traces were not recorded.")

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


def normalize_pattern_vector(x: np.ndarray | torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    centered = arr - float(arr.mean())
    norm = float(np.linalg.norm(centered, ord=2))
    return centered / float(norm + float(eps))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)))


def compute_trace_pattern_similarity(
    cond_trace: np.ndarray | torch.Tensor,
    ref_trace_dyn: np.ndarray | torch.Tensor,
    ref_trace_sta: np.ndarray | torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    return shared_compute_trace_pattern_similarity(
        cond_trace=cond_trace,
        ref_trace_dyn=ref_trace_dyn,
        ref_trace_sta=ref_trace_sta,
        eps=eps,
    )


def compute_final_pattern_similarity(
    v_cond: np.ndarray | torch.Tensor,
    v_full_dyn: np.ndarray | torch.Tensor,
    v_full_sta: np.ndarray | torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[float, float, float]:
    return shared_compute_final_pattern_similarity(
        v_cond=v_cond,
        v_full_dyn=v_full_dyn,
        v_full_sta=v_full_sta,
        eps=eps,
    )


__all__ = [
    "OverlapMaskBundle",
    "RolloutReadout",
    "apply_input_mask_to_spike_batch",
    "build_best_energy_matched_control_mask",
    "build_overlap_masks_for_pair",
    "compute_final_pattern_similarity",
    "compute_trace_pattern_similarity",
    "dilate_mask",
    "foreground_mask",
    "mask_energy",
    "normalize_pattern_vector",
    "run_overlap_perturbed_dms",
]
