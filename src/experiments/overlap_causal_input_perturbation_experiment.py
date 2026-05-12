
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.units import ms
from src.experiments.common.dataset import build_class_index
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.pattern_metrics import (
    compute_final_pattern_similarity as shared_compute_final_pattern_similarity,
    compute_trace_pattern_similarity as shared_compute_trace_pattern_similarity,
)
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.results import prepare_result_layout, save_log_lines
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.voltage_readout import resolve_readout_step
from src.experiments.distractor.shared.pair_sampling import (
    PairExperimentSpec,
    build_dataset_arrays,
    build_pair_specs,
    extract_grouped_voltage_vector,
    prepare_pair_spike_batch,
)
from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_tidy_csv
from src.plotting.common.theme_tokens import (
    ALPHA_FILL,
    ALPHA_SCATTER,
    GRID_ALPHA,
    LINE_WIDTH_PRIMARY,
    LINE_WIDTH_REFERENCE,
    OVERLAP_CONDITION_COLORS,
)

EXPERIMENT_NAME = "overlap_causal_input_perturbation_experiment"
PRIMARY_FOCUS = "s2p / L3 probe trace"

DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_OUTPUT_DIR = f"results/{EXPERIMENT_NAME}"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_DELAY_MS = 500.0
DEFAULT_PROBE_MS = 100.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_PROBES = 20
DEFAULT_SAMPLES_PER_PROBE = 12
DEFAULT_MAX_PAIRS = 500
DEFAULT_NUM_SIM_BINS = 5
DEFAULT_FOREGROUND_THRESHOLD = 0.0
DEFAULT_DILATION_RADIUS = 1
DEFAULT_SAVE_CASE_COUNT = 4
DEFAULT_NUM_CONTROL_CANDIDATES = 32

MAIN_CONDITION_ORDER: tuple[str, ...] = (
    "full_dynamic",
    "full_static",
    "sample_keep_overlap_only_dynamic",
    "sample_keep_nonoverlap_only_dynamic",
)
PRIMARY_ANALYSIS_CONDITIONS: tuple[str, ...] = (
    "sample_keep_overlap_only_dynamic",
    "sample_keep_nonoverlap_only_dynamic",
)
LAYER_ORDER: tuple[str, ...] = ("L1", "L2", "L3", "final")

CONDITION_COLORS: dict[str, str] = dict(OVERLAP_CONDITION_COLORS)
CONDITION_COLORS["sample_keep_nonoverlap_only_dynamic"] = "#E45756"
REFERENCE_COLORS = {
    "dynamic": CONDITION_COLORS["full_dynamic"],
    "static": CONDITION_COLORS["full_static"],
}


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


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
class ConditionSpec:
    name: str
    stsp_mode: str
    sample_mask_key: str | None


@dataclass(frozen=True)
class RolloutReadout:
    grouped_voltage: np.ndarray
    probe_l1_trace: torch.Tensor
    probe_l2_trace: torch.Tensor
    probe_l3_trace: torch.Tensor
    prediction_probe: np.ndarray
    first_fire_t_probe: np.ndarray
    readout_step: int


def mix_seed(base_seed: int, *parts: int) -> int:
    value = int(base_seed) & 0xFFFFFFFF
    for idx, part in enumerate(parts, start=1):
        value = (value * 1664525 + 1013904223 + int(part) * (374761393 + idx * 97)) & 0xFFFFFFFF
    return int(value)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        return None
    return scalar


def _sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _safe_ratio(numerator: float, denominator: float, *, eps: float = 1e-8) -> float:
    if abs(float(denominator)) <= float(eps):
        if abs(float(numerator)) <= float(eps):
            return 0.0
        return float("nan")
    return float(numerator) / float(denominator)


def _subtract_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    result = float(left) - float(right)
    if not np.isfinite(result):
        return None
    return result


def _to_json_ready(value):
    if isinstance(value, dict):
        return {str(key): _to_json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _safe_float(value)
    return value


def _save_json(payload: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json_ready(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


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


def _foreground_mask(image: torch.Tensor, threshold: float) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected image shape [C, H, W], got {tuple(image.shape)}")
    arr = image.detach().cpu().to(torch.float32).abs().amax(dim=0).numpy()
    return np.asarray(arr > float(threshold), dtype=bool)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any() or int(radius) <= 0:
        return mask_bool
    tensor = torch.as_tensor(mask_bool.astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = 2 * int(radius) + 1
    dilated = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(radius))
    return np.asarray(dilated.squeeze(0).squeeze(0).numpy() > 0.0, dtype=bool)


def _mask_energy(image: torch.Tensor, mask: np.ndarray) -> float:
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


def _build_best_energy_matched_control_mask(
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

    reference_energy = _mask_energy(image, reference_bool)
    best_mask = None
    best_gap = None
    for _ in range(max(1, int(num_candidates))):
        candidate = _sample_area_matched_mask(pool, area=area, rng=rng)
        energy_gap = abs(_mask_energy(image, candidate) - reference_energy)
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
    sample_fg = _foreground_mask(sample_image, threshold=foreground_threshold)
    probe_fg = _foreground_mask(probe_image, threshold=foreground_threshold)
    base_overlap = sample_fg & probe_fg
    if bool(use_dilated_overlap) and int(dilation_radius) > 0 and base_overlap.any():
        sample_overlap = _dilate_mask(base_overlap, int(dilation_radius)) & sample_fg
        probe_overlap = _dilate_mask(base_overlap, int(dilation_radius)) & probe_fg
    else:
        sample_overlap = base_overlap.copy()
        probe_overlap = base_overlap.copy()
    sample_nonoverlap = sample_fg & ~sample_overlap

    sample_rng = np.random.default_rng(mix_seed(seed, 101, int(sample_image.numel())))
    probe_rng = np.random.default_rng(mix_seed(seed, 202, int(probe_image.numel())))
    sample_control, sample_control_source, sample_gap = _build_best_energy_matched_control_mask(
        image=sample_image,
        reference_mask=sample_overlap,
        preferred_pool_mask=sample_fg & ~sample_overlap,
        fallback_pool_mask=~sample_overlap,
        rng=sample_rng,
        num_candidates=int(num_control_candidates),
    )
    probe_control, probe_control_source, probe_gap = _build_best_energy_matched_control_mask(
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
        "sample_overlap_energy": float(_mask_energy(sample_image, sample_overlap)),
        "sample_nonoverlap_energy": float(_mask_energy(sample_image, sample_nonoverlap)),
        "sample_control_energy": float(_mask_energy(sample_image, sample_control)),
        "probe_overlap_energy": float(_mask_energy(probe_image, probe_overlap)),
        "probe_control_energy": float(_mask_energy(probe_image, probe_control)),
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

def build_main_condition_specs(*, include_legacy_conditions: bool = False) -> dict[str, ConditionSpec]:
    specs: dict[str, ConditionSpec] = {
        "full_dynamic": ConditionSpec("full_dynamic", "dynamic", None),
        "full_static": ConditionSpec("full_static", "static_frozen", None),
        "sample_keep_overlap_only_dynamic": ConditionSpec(
            "sample_keep_overlap_only_dynamic",
            "dynamic",
            "sample_nonoverlap_mask",
        ),
        "sample_keep_nonoverlap_only_dynamic": ConditionSpec(
            "sample_keep_nonoverlap_only_dynamic",
            "dynamic",
            "sample_overlap_mask",
        ),
    }
    if include_legacy_conditions:
        specs.update(
            {
                "sample_remove_overlap_dynamic": ConditionSpec("sample_remove_overlap_dynamic", "dynamic", "sample_overlap_mask"),
                "sample_remove_nonoverlap_control_dynamic": ConditionSpec(
                    "sample_remove_nonoverlap_control_dynamic",
                    "dynamic",
                    "sample_nonoverlap_control_mask",
                ),
                "sample_remove_all_foreground_dynamic": ConditionSpec(
                    "sample_remove_all_foreground_dynamic",
                    "dynamic",
                    "sample_foreground_mask",
                ),
            }
        )
    return specs


def _build_condition_mask_batch(mask_records: Sequence[OverlapMaskBundle], mask_key: str | None) -> torch.Tensor | None:
    if mask_key is None:
        return None
    stacked = np.stack([np.asarray(getattr(record, mask_key), dtype=bool) for record in mask_records], axis=0)
    return torch.as_tensor(stacked, dtype=torch.bool)


def run_single_condition_rollout(
    *,
    net,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    batch_masks: Sequence[OverlapMaskBundle],
    condition_spec: ConditionSpec,
    delay_steps: int,
    readout_step: int,
    device: torch.device,
) -> RolloutReadout:
    sample_mask = _build_condition_mask_batch(batch_masks, condition_spec.sample_mask_key)
    return run_overlap_perturbed_dms(
        net=net,
        sample_spikes=sample_spikes,
        probe_spikes=probe_spikes,
        delay_steps=delay_steps,
        stsp_mode=condition_spec.stsp_mode,
        readout_step=readout_step,
        sample_input_mask=None if sample_mask is None else sample_mask.to(device=device),
    )


def _compute_curve_mean_sem(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(curves, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected [N, T] curve array, got {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros(arr.shape[1], dtype=np.float64), np.zeros(arr.shape[1], dtype=np.float64)
    mean_curve = arr.mean(axis=0)
    if arr.shape[0] == 1:
        sem_curve = np.zeros_like(mean_curve)
    else:
        sem_curve = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean_curve, sem_curve


def extract_and_summarize_s2p_trace_similarity(
    trace_arrays: Mapping[str, np.ndarray],
    condition_name: str,
) -> dict[str, object]:
    condition_vector = np.asarray(trace_arrays["condition_name"])
    selector = condition_vector == str(condition_name)
    dpi_all = np.asarray(trace_arrays["DPI_L3"], dtype=np.float64)
    s_dyn_all = np.asarray(trace_arrays.get("S_dyn_L3", np.zeros_like(dpi_all)), dtype=np.float64)
    s_sta_all = np.asarray(trace_arrays.get("S_sta_L3", np.zeros_like(dpi_all)), dtype=np.float64)
    s_dyn = s_dyn_all[selector]
    s_sta = s_sta_all[selector]
    dpi = dpi_all[selector]
    probe_steps = int(s_dyn_all.shape[1]) if s_dyn_all.ndim == 2 else 0
    if s_dyn.size == 0:
        return {
            "condition": str(condition_name),
            "n_records": 0,
            "time_axis": np.arange(probe_steps, dtype=np.int64),
            "S_dyn_mean": np.zeros(probe_steps, dtype=np.float64),
            "S_dyn_sem": np.zeros(probe_steps, dtype=np.float64),
            "S_sta_mean": np.zeros(probe_steps, dtype=np.float64),
            "S_sta_sem": np.zeros(probe_steps, dtype=np.float64),
            "DPI_mean": np.zeros(probe_steps, dtype=np.float64),
            "DPI_sem": np.zeros(probe_steps, dtype=np.float64),
        }
    s_dyn_mean, s_dyn_sem = _compute_curve_mean_sem(s_dyn)
    s_sta_mean, s_sta_sem = _compute_curve_mean_sem(s_sta)
    dpi_mean, dpi_sem = _compute_curve_mean_sem(dpi)
    return {
        "condition": str(condition_name),
        "n_records": int(s_dyn.shape[0]),
        "time_axis": np.arange(s_dyn.shape[1], dtype=np.int64),
        "S_dyn_mean": s_dyn_mean,
        "S_dyn_sem": s_dyn_sem,
        "S_sta_mean": s_sta_mean,
        "S_sta_sem": s_sta_sem,
        "DPI_mean": dpi_mean,
        "DPI_sem": dpi_sem,
    }


def _plot_s2p_trace_similarity(summary: Mapping[str, object], *, title: str) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    time_axis = np.asarray(summary["time_axis"], dtype=np.int64)
    s_dyn_mean = np.asarray(summary["S_dyn_mean"], dtype=np.float64)
    s_dyn_sem = np.asarray(summary["S_dyn_sem"], dtype=np.float64)
    s_sta_mean = np.asarray(summary["S_sta_mean"], dtype=np.float64)
    s_sta_sem = np.asarray(summary["S_sta_sem"], dtype=np.float64)

    ax.plot(time_axis, s_dyn_mean, color=REFERENCE_COLORS["dynamic"], linewidth=LINE_WIDTH_PRIMARY, label="S_dyn_L3(t)")
    ax.fill_between(
        time_axis,
        s_dyn_mean - s_dyn_sem,
        s_dyn_mean + s_dyn_sem,
        color=REFERENCE_COLORS["dynamic"],
        alpha=ALPHA_FILL,
    )
    ax.plot(time_axis, s_sta_mean, color=REFERENCE_COLORS["static"], linewidth=LINE_WIDTH_PRIMARY, label="S_sta_L3(t)")
    ax.fill_between(
        time_axis,
        s_sta_mean - s_sta_sem,
        s_sta_mean + s_sta_sem,
        color=REFERENCE_COLORS["static"],
        alpha=ALPHA_FILL,
    )
    ax.set_title(title)
    ax.set_xlabel("Probe time step")
    ax.set_ylabel("Similarity")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_s2p_trace_similarity_keep_overlap_only(trace_arrays: Mapping[str, np.ndarray]) -> plt.Figure:
    summary = extract_and_summarize_s2p_trace_similarity(trace_arrays, "sample_keep_overlap_only_dynamic")
    return _plot_s2p_trace_similarity(summary, title="s2p / L3 Trace Pattern Similarity - Keep Overlap Only")


def plot_s2p_trace_similarity_keep_nonoverlap_only(trace_arrays: Mapping[str, np.ndarray]) -> plt.Figure:
    summary = extract_and_summarize_s2p_trace_similarity(trace_arrays, "sample_keep_nonoverlap_only_dynamic")
    return _plot_s2p_trace_similarity(summary, title="s2p / L3 Trace Pattern Similarity - Keep Non-overlap Only")


def plot_dpi_l3_trace_overlap_vs_nonoverlap(trace_arrays: Mapping[str, np.ndarray]) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    max_time_steps = 60
    condition_specs = (
        ("sample_keep_overlap_only_dynamic", "Keep Overlap Only"),
        ("sample_keep_nonoverlap_only_dynamic", "Keep Non-overlap Only"),
    )
    for condition_name, label in condition_specs:
        summary = extract_and_summarize_s2p_trace_similarity(trace_arrays, condition_name)
        time_axis = np.asarray(summary["time_axis"], dtype=np.int64)[:max_time_steps]
        dpi_mean = np.asarray(summary["DPI_mean"], dtype=np.float64)[:max_time_steps]
        dpi_sem = np.asarray(summary["DPI_sem"], dtype=np.float64)[:max_time_steps]
        color = CONDITION_COLORS[condition_name]
        ax.plot(time_axis, dpi_mean, color=color, linewidth=LINE_WIDTH_PRIMARY, label=label)
        ax.fill_between(
            time_axis,
            dpi_mean - dpi_sem,
            dpi_mean + dpi_sem,
            color=color,
            alpha=max(ALPHA_FILL * 0.7, 0.08),
        )
    ax.axhline(0.0, color="#333333", linewidth=LINE_WIDTH_REFERENCE, linestyle=":")
    ax.set_xlabel("Probe time step")
    ax.set_ylabel("DPI_L3(t)")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_s2p_dpi_distribution(df_results: pd.DataFrame, *, condition_name: str, title: str) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    subset = df_results[df_results["condition"] == str(condition_name)].copy()
    values = subset["DPI_L3"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.10, 0.10, size=values.size) if values.size > 0 else np.zeros(0, dtype=np.float64)
    color = CONDITION_COLORS[condition_name]

    if values.size > 0:
        ax.scatter(x, values, color=color, alpha=ALPHA_SCATTER, s=28, edgecolors="none")
        mean_value = float(np.mean(values))
        sem_value = _sem(values)
        ax.errorbar(
            [0.0],
            [mean_value],
            yerr=[sem_value],
            fmt="o",
            color="#222222",
            markersize=6,
            elinewidth=LINE_WIDTH_REFERENCE,
            capsize=4,
            zorder=4,
        )
        ax.hlines(mean_value, -0.22, 0.22, colors="#222222", linewidth=LINE_WIDTH_REFERENCE)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    ax.axhline(0.0, color="#333333", linewidth=LINE_WIDTH_REFERENCE, linestyle=":")
    ax.set_xlim(-0.25, 0.25)
    ax.set_xticks([])
    ax.set_ylabel("DPI_L3")
    ax.set_title(title)
    ax.grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def plot_s2p_dpi_keep_overlap_only(df_results: pd.DataFrame) -> plt.Figure:
    return _plot_s2p_dpi_distribution(
        df_results,
        condition_name="sample_keep_overlap_only_dynamic",
        title="s2p / L3 DPI - Keep Overlap Only",
    )


def plot_s2p_dpi_keep_nonoverlap_only(df_results: pd.DataFrame) -> plt.Figure:
    return _plot_s2p_dpi_distribution(
        df_results,
        condition_name="sample_keep_nonoverlap_only_dynamic",
        title="s2p / L3 DPI - Keep Non-overlap Only",
    )


def plot_dpi_l3_summary_overlap_vs_nonoverlap(df_results: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    condition_specs = (
        ("sample_keep_overlap_only_dynamic", "Overlap"),
        ("sample_keep_nonoverlap_only_dynamic", "Non-overlap"),
    )
    rng = np.random.default_rng(0)
    has_any_data = False

    for xpos, (condition_name, label) in enumerate(condition_specs):
        subset = df_results[df_results["condition"] == str(condition_name)].copy()
        values = subset["DPI_L3"].to_numpy(dtype=np.float64)
        color = CONDITION_COLORS[condition_name]
        if values.size <= 0:
            continue
        has_any_data = True
        jitter = rng.uniform(-0.10, 0.10, size=values.size)
        ax.scatter(
            np.full(values.size, float(xpos), dtype=np.float64) + jitter,
            values,
            color=color,
            alpha=ALPHA_SCATTER,
            s=28,
            edgecolors="none",
        )
        mean_value = float(np.mean(values))
        sem_value = _sem(values)
        ax.errorbar(
            [float(xpos)],
            [mean_value],
            yerr=[sem_value],
            fmt="o",
            color="#222222",
            markersize=6,
            elinewidth=LINE_WIDTH_REFERENCE,
            capsize=4,
            zorder=4,
        )
        ax.hlines(mean_value, xpos - 0.22, xpos + 0.22, colors="#222222", linewidth=LINE_WIDTH_REFERENCE)

    if not has_any_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0.0, color="#333333", linewidth=LINE_WIDTH_REFERENCE, linestyle=":")
    ax.set_xlim(-0.5, len(condition_specs) - 0.5)
    ax.set_xticks(np.arange(len(condition_specs), dtype=np.int64))
    ax.set_xticklabels([label for _, label in condition_specs])
    ax.set_ylabel("DPI_L3")
    ax.grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def _condition_mean_payload(df_results: pd.DataFrame, condition_name: str) -> dict[str, float | None]:
    subset = df_results[df_results["condition"] == str(condition_name)].copy()
    if subset.empty:
        return {
            "mean_S_dyn_L3": None,
            "mean_S_sta_L3": None,
            "mean_DPI_L3": None,
            "sem_DPI_L3": None,
        }
    dpi_values = subset["DPI_L3"].to_numpy(dtype=np.float64)
    return {
        "mean_S_dyn_L3": _safe_float(subset["mean_S_dyn_L3"].mean()),
        "mean_S_sta_L3": _safe_float(subset["mean_S_sta_L3"].mean()),
        "mean_DPI_L3": _safe_float(dpi_values.mean()),
        "sem_DPI_L3": _safe_float(_sem(dpi_values)),
    }


def build_summary_metrics(df_results: pd.DataFrame) -> dict[str, object]:
    overlap_summary = _condition_mean_payload(df_results, "sample_keep_overlap_only_dynamic")
    nonoverlap_summary = _condition_mean_payload(df_results, "sample_keep_nonoverlap_only_dynamic")
    return {
        "overall": {
            "n_records": int(len(df_results)),
            "n_pairs": int(df_results["pair_id"].nunique()) if not df_results.empty else 0,
            "n_probes": int(df_results["probe_id"].nunique()) if not df_results.empty else 0,
        },
        "primary_focus": PRIMARY_FOCUS,
        "condition_means": {
            "sample_keep_overlap_only_dynamic": overlap_summary,
            "sample_keep_nonoverlap_only_dynamic": nonoverlap_summary,
        },
        "comparison": {
            "delta_DPI_L3_overlap_minus_nonoverlap": _subtract_or_none(
                overlap_summary["mean_DPI_L3"],
                nonoverlap_summary["mean_DPI_L3"],
            ),
            "delta_S_dyn_L3_overlap_minus_nonoverlap": _subtract_or_none(
                overlap_summary["mean_S_dyn_L3"],
                nonoverlap_summary["mean_S_dyn_L3"],
            ),
            "delta_S_sta_L3_overlap_minus_nonoverlap": _subtract_or_none(
                overlap_summary["mean_S_sta_L3"],
                nonoverlap_summary["mean_S_sta_L3"],
            ),
        },
        "assumptions": {
            "primary_analysis_object": "probe_l3_trace is interpreted as s2p, i.e. the decision-layer input during probe.",
            "grouped_voltage": "Saved as auxiliary output only via layer3.get_grouped_voltage(v_mem_snapshot).mean(-1).",
            "pattern_normalization": "(x - mean(x)) / (||x - mean(x)||_2 + eps)",
            "S_dyn_L3_definition": "cosine similarity between the condition probe_l3_trace and full_dynamic probe_l3_trace at each probe time step",
            "S_sta_L3_definition": "cosine similarity between the condition probe_l3_trace and full_static probe_l3_trace at each probe time step",
            "DPI_L3_definition": "mean_t [S_dyn_L3(t) - S_sta_L3(t)]",
            "keep_overlap_only_condition": "Remove sample_nonoverlap_mask so only overlap-region sample input remains.",
            "keep_nonoverlap_only_condition": "Remove sample_overlap_mask so only non-overlap sample input remains.",
            "probe_perturbation": "disabled; only sample-side perturbation is applied",
        },
    }


def build_summary_payload(summary_metrics: Mapping[str, object]) -> dict[str, object]:
    condition_means = dict(summary_metrics.get("condition_means", {}))
    comparison = dict(summary_metrics.get("comparison", {}))
    overlap_summary = dict(condition_means.get("sample_keep_overlap_only_dynamic", {}))
    nonoverlap_summary = dict(condition_means.get("sample_keep_nonoverlap_only_dynamic", {}))
    summary_text = (
        "Keeping only overlap regions preserves a more dynamic-like s2p trace, "
        "whereas keeping only non-overlap regions shifts the s2p trace toward static-like similarity. "
        f"Overlap-only: mean_S_dyn_L3={_safe_float(overlap_summary.get('mean_S_dyn_L3'))}, "
        f"mean_S_sta_L3={_safe_float(overlap_summary.get('mean_S_sta_L3'))}, "
        f"mean_DPI_L3={_safe_float(overlap_summary.get('mean_DPI_L3'))}; "
        f"non-overlap-only: mean_S_dyn_L3={_safe_float(nonoverlap_summary.get('mean_S_dyn_L3'))}, "
        f"mean_S_sta_L3={_safe_float(nonoverlap_summary.get('mean_S_sta_L3'))}, "
        f"mean_DPI_L3={_safe_float(nonoverlap_summary.get('mean_DPI_L3'))}."
    )
    return {
        "experiment_name": EXPERIMENT_NAME,
        "primary_focus": PRIMARY_FOCUS,
        "primary_conditions": list(PRIMARY_ANALYSIS_CONDITIONS),
        "overlap_only_mean_S_dyn_L3": overlap_summary.get("mean_S_dyn_L3"),
        "overlap_only_mean_S_sta_L3": overlap_summary.get("mean_S_sta_L3"),
        "overlap_only_mean_DPI_L3": overlap_summary.get("mean_DPI_L3"),
        "nonoverlap_only_mean_S_dyn_L3": nonoverlap_summary.get("mean_S_dyn_L3"),
        "nonoverlap_only_mean_S_sta_L3": nonoverlap_summary.get("mean_S_sta_L3"),
        "nonoverlap_only_mean_DPI_L3": nonoverlap_summary.get("mean_DPI_L3"),
        "delta_DPI_L3_overlap_minus_nonoverlap": comparison.get("delta_DPI_L3_overlap_minus_nonoverlap"),
        "delta_S_dyn_L3_overlap_minus_nonoverlap": comparison.get("delta_S_dyn_L3_overlap_minus_nonoverlap"),
        "delta_S_sta_L3_overlap_minus_nonoverlap": comparison.get("delta_S_sta_L3_overlap_minus_nonoverlap"),
        "summary_text": summary_text,
    }


def _initialize_trace_records() -> dict[str, list[object]]:
    return {
        "condition_name": [],
        "DPI_L3": [],
    }


def _initialize_final_records() -> dict[str, list[object]]:
    return {
        "record_id": [],
        "pair_id": [],
        "condition_name": [],
        "V_cond": [],
        "V_full_dyn": [],
        "V_full_sta": [],
        "S_dyn_final": [],
        "S_sta_final": [],
        "DPI_final": [],
        "Retain_dyn_final": [],
        "Pull_sta_final": [],
    }


def _empty_trace_matrix(probe_steps: int) -> np.ndarray:
    return np.zeros((0, probe_steps), dtype=np.float32)


def finalize_trace_arrays(trace_records: Mapping[str, list[object]], probe_steps: int) -> dict[str, np.ndarray]:
    return {
        "condition_name": np.asarray(trace_records["condition_name"]),
        "DPI_L3": np.stack(trace_records["DPI_L3"], axis=0) if trace_records["DPI_L3"] else _empty_trace_matrix(probe_steps),
    }


def finalize_final_arrays(final_records: Mapping[str, list[object]], num_classes: int) -> dict[str, np.ndarray]:
    return {
        "record_id": np.asarray(final_records["record_id"], dtype=np.int64),
        "pair_id": np.asarray(final_records["pair_id"], dtype=np.int64),
        "condition_name": np.asarray(final_records["condition_name"]),
        "V_cond": np.stack(final_records["V_cond"], axis=0) if final_records["V_cond"] else np.zeros((0, num_classes), dtype=np.float32),
        "V_full_dyn": np.stack(final_records["V_full_dyn"], axis=0)
        if final_records["V_full_dyn"]
        else np.zeros((0, num_classes), dtype=np.float32),
        "V_full_sta": np.stack(final_records["V_full_sta"], axis=0)
        if final_records["V_full_sta"]
        else np.zeros((0, num_classes), dtype=np.float32),
        "S_dyn_final": np.asarray(final_records["S_dyn_final"], dtype=np.float32),
        "S_sta_final": np.asarray(final_records["S_sta_final"], dtype=np.float32),
        "DPI_final": np.asarray(final_records["DPI_final"], dtype=np.float32),
        "Retain_dyn_final": np.asarray(final_records["Retain_dyn_final"], dtype=np.float32),
        "Pull_sta_final": np.asarray(final_records["Pull_sta_final"], dtype=np.float32),
    }

def save_main_figures(figures_dir: Path, df_results: pd.DataFrame, trace_arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, str]]:
    figure_paths: dict[str, dict[str, str]] = {}

    fig_dpi_trace = plot_dpi_l3_trace_overlap_vs_nonoverlap(trace_arrays)
    figure_paths["dpi_l3_trace_overlap_vs_nonoverlap"] = save_figure_all_formats(
        fig_dpi_trace,
        figures_dir / "dpi_l3_trace_overlap_vs_nonoverlap",
    )
    plt.close(fig_dpi_trace)

    return figure_paths


def save_metadata_files(
    *,
    args: argparse.Namespace,
    device: torch.device,
    result_root: Path,
    log_dir: Path,
    readout_step: int,
    condition_order: Sequence[str],
    summary_payload: Mapping[str, object],
    output_paths: Mapping[str, object],
    start_time: str,
    n_pairs: int,
) -> tuple[Path, Path, Path]:
    run_config_payload = {
        "model_path": str(Path(args.model_path).resolve()),
        "config_argument": args.config,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": str(args.split),
        "output_dir": str(result_root.resolve()),
        "device": str(device),
        "seed": int(args.seed),
        "sample_ms": float(args.sample_ms),
        "delay_ms": float(args.delay_ms),
        "probe_ms": float(args.probe_ms),
        "batch_size": int(args.batch_size),
        "max_probes": int(args.max_probes),
        "samples_per_probe": int(args.samples_per_probe),
        "max_pairs": int(args.max_pairs),
        "num_sim_bins": int(args.num_sim_bins),
        "foreground_threshold": float(args.foreground_threshold),
        "use_dilated_overlap": bool(args.use_dilated_overlap),
        "dilation_radius": int(args.dilation_radius),
        "save_case_count": int(args.save_case_count),
        "num_control_candidates": int(args.num_control_candidates),
        "skip_figures": bool(args.skip_figures),
        "readout_step": int(readout_step),
        "condition_order": [str(name) for name in condition_order],
    }
    run_config_path = _save_json(run_config_payload, result_root / "run_config.json")
    summary_path = _save_json(summary_payload, result_root / "summary.json")

    figure_paths = output_paths.get("figure_paths", {})
    dpi_trace_png = figure_paths.get("dpi_l3_trace_overlap_vs_nonoverlap", {}).get("png", "")
    log_lines = [
        f"start_time={start_time}",
        f"experiment_name={EXPERIMENT_NAME}",
        f"model_path={Path(args.model_path).resolve()}",
        f"dataset_root={Path(args.dataset_root).resolve()}",
        f"seed={int(args.seed)}",
        f"device={device}",
        f"pairs={int(n_pairs)}",
        f"primary_focus={PRIMARY_FOCUS}",
        f"primary_conditions={','.join(PRIMARY_ANALYSIS_CONDITIONS)}",
        f"pair_trace_similarity_npz={output_paths['trace_npz']}",
        f"summary_json={summary_path.resolve()}",
        f"run_config_json={run_config_path.resolve()}",
        f"dpi_l3_trace_overlap_vs_nonoverlap_png={dpi_trace_png}",
    ]
    run_log_path = save_log_lines(log_lines, log_dir, filename="run.log")
    return run_config_path, summary_path, run_log_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlap causal input perturbation experiment focused on s2p / L3")
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--delay-ms", type=float, default=DEFAULT_DELAY_MS)
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--samples-per-probe", type=int, default=DEFAULT_SAMPLES_PER_PROBE)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--num-sim-bins", type=int, default=DEFAULT_NUM_SIM_BINS)
    parser.add_argument("--foreground-threshold", type=float, default=DEFAULT_FOREGROUND_THRESHOLD)
    parser.add_argument("--use-dilated-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dilation-radius", type=int, default=DEFAULT_DILATION_RADIUS)
    parser.add_argument("--save-case-count", type=int, default=DEFAULT_SAVE_CASE_COUNT)
    parser.add_argument("--num-control-candidates", type=int, default=DEFAULT_NUM_CONTROL_CANDIDATES)
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.max_probes) <= 0:
        raise ValueError("--max-probes must be positive.")
    if int(args.samples_per_probe) <= 0:
        raise ValueError("--samples-per-probe must be positive.")
    if int(args.max_pairs) <= 0:
        raise ValueError("--max-pairs must be positive.")
    if int(args.num_sim_bins) <= 0:
        raise ValueError("--num-sim-bins must be positive.")
    if int(args.save_case_count) < 0:
        raise ValueError("--save-case-count must be non-negative.")
    if int(args.num_control_candidates) <= 0:
        raise ValueError("--num-control-candidates must be positive.")

    start_time = datetime.now().astimezone().isoformat()
    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=1.0 * ms, sample_ms=float(args.sample_ms), probe_ms=float(args.probe_ms))
    if spec.sample_steps <= 0 or spec.probe_steps <= 0:
        raise ValueError("sample/probe duration must resolve to positive steps.")
    delay_steps = int(round((float(args.delay_ms) * ms) / spec.dt))
    if delay_steps < 0:
        raise ValueError("--delay-ms must resolve to a non-negative number of steps.")

    layout = prepare_result_layout(args.output_dir)
    result_root = layout.root
    figures_dir = layout.figure_dir
    data_dir = layout.data_dir
    log_dir = layout.log_dir

    dataset = _load_dataset(dataset_root=args.dataset_root, split=args.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    num_classes = int(len(np.unique(labels)))
    class_index = build_class_index(dataset, num_classes=num_classes)
    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(float(args.sample_ms), float(args.delay_ms), float(args.probe_ms)),
    )
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )
    df_pairs = build_pair_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=int(args.max_probes),
        samples_per_probe=int(args.samples_per_probe),
        num_bins=int(args.num_sim_bins),
        max_pairs=int(args.max_pairs),
        seed=int(args.seed),
    )

    mask_records: list[OverlapMaskBundle] = []
    for pair_row in df_pairs.itertuples(index=False):
        pair_id = int(pair_row.pair_id)
        mask_records.append(
            build_overlap_masks_for_pair(
                sample_image=images[int(pair_row.sample_id)],
                probe_image=images[int(pair_row.probe_id)],
                foreground_threshold=float(args.foreground_threshold),
                use_dilated_overlap=bool(args.use_dilated_overlap),
                dilation_radius=int(args.dilation_radius),
                seed=mix_seed(int(args.seed), pair_id, int(pair_row.sample_id), int(pair_row.probe_id)),
                num_control_candidates=int(args.num_control_candidates),
            )
        )

    condition_specs = build_main_condition_specs(include_legacy_conditions=False)
    condition_order = tuple(condition_specs.keys())
    results_rows: list[dict[str, object]] = []
    trace_records = _initialize_trace_records()

    batch_starts = range(0, len(df_pairs), int(args.batch_size))
    total_batches = math.ceil(len(df_pairs) / int(args.batch_size)) if len(df_pairs) > 0 else 0
    for batch_start in tqdm(batch_starts, total=total_batches, desc="Running s2p overlap perturbation"):
        batch_df = df_pairs.iloc[batch_start : batch_start + int(args.batch_size)].copy().reset_index(drop=True)
        sample_spikes, probe_spikes = prepare_pair_spike_batch(
            images=images,
            batch_df=batch_df,
            encoder=encoder,
            spec=PairExperimentSpec(dt=spec.dt, sample_ms=spec.sample_ms, probe_ms=spec.probe_ms),
            device=device,
        )
        batch_pair_ids = batch_df["pair_id"].astype(int).tolist()
        batch_masks = [mask_records[pair_id] for pair_id in batch_pair_ids]

        rollout_outputs: dict[str, RolloutReadout] = {}
        for condition_name in condition_order:
            rollout_outputs[condition_name] = run_single_condition_rollout(
                net=net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                batch_masks=batch_masks,
                condition_spec=condition_specs[condition_name],
                delay_steps=delay_steps,
                readout_step=readout_step,
                device=device,
            )

        full_dynamic = rollout_outputs["full_dynamic"]
        full_static = rollout_outputs["full_static"]

        for batch_idx, pair_row in enumerate(batch_df.itertuples(index=False)):
            pair_id = int(pair_row.pair_id)
            mask_bundle = mask_records[pair_id]
            ref_l1_dyn = full_dynamic.probe_l1_trace[:, batch_idx].numpy()
            ref_l2_dyn = full_dynamic.probe_l2_trace[:, batch_idx].numpy()
            ref_l3_dyn = full_dynamic.probe_l3_trace[:, batch_idx].numpy()
            ref_l1_sta = full_static.probe_l1_trace[:, batch_idx].numpy()
            ref_l2_sta = full_static.probe_l2_trace[:, batch_idx].numpy()
            ref_l3_sta = full_static.probe_l3_trace[:, batch_idx].numpy()
            v_full_dyn = np.asarray(full_dynamic.grouped_voltage[batch_idx], dtype=np.float64)
            v_full_sta = np.asarray(full_static.grouped_voltage[batch_idx], dtype=np.float64)

            pair_condition_metrics: dict[str, dict[str, object]] = {}
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
                predicted_label = int(rollout.prediction_probe[batch_idx])
                first_fire_t_probe = int(rollout.first_fire_t_probe[batch_idx])
                pair_condition_metrics[condition_name] = {
                    "rollout": rollout,
                    "predicted_label": predicted_label,
                    "first_fire_t_probe": first_fire_t_probe,
                    "v_cond": v_cond,
                    "S_dyn": {
                        "L1": np.asarray(s_dyn_l1, dtype=np.float64),
                        "L2": np.asarray(s_dyn_l2, dtype=np.float64),
                        "L3": np.asarray(s_dyn_l3, dtype=np.float64),
                        "final": float(s_dyn_final),
                    },
                    "S_sta": {
                        "L1": np.asarray(s_sta_l1, dtype=np.float64),
                        "L2": np.asarray(s_sta_l2, dtype=np.float64),
                        "L3": np.asarray(s_sta_l3, dtype=np.float64),
                        "final": float(s_sta_final),
                    },
                    "DPI": {
                        "L1": float(dpi_l1),
                        "L2": float(dpi_l2),
                        "L3": float(dpi_l3),
                        "final": float(dpi_final),
                    },
                }

            full_dynamic_metric = pair_condition_metrics["full_dynamic"]
            ref_mean_s_dyn = {
                "L1": float(np.mean(full_dynamic_metric["S_dyn"]["L1"])),
                "L2": float(np.mean(full_dynamic_metric["S_dyn"]["L2"])),
                "L3": float(np.mean(full_dynamic_metric["S_dyn"]["L3"])),
                "final": float(full_dynamic_metric["S_dyn"]["final"]),
            }

            for condition_name in condition_order:
                condition_metric = pair_condition_metrics[condition_name]
                rollout = condition_metric["rollout"]
                record_id = len(results_rows)
                mean_s_dyn = {
                    "L1": float(np.mean(condition_metric["S_dyn"]["L1"])),
                    "L2": float(np.mean(condition_metric["S_dyn"]["L2"])),
                    "L3": float(np.mean(condition_metric["S_dyn"]["L3"])),
                    "final": float(condition_metric["S_dyn"]["final"]),
                }
                mean_s_sta = {
                    "L1": float(np.mean(condition_metric["S_sta"]["L1"])),
                    "L2": float(np.mean(condition_metric["S_sta"]["L2"])),
                    "L3": float(np.mean(condition_metric["S_sta"]["L3"])),
                    "final": float(condition_metric["S_sta"]["final"]),
                }
                retain_dyn = {
                    layer_name: _safe_ratio(mean_s_dyn[layer_name], ref_mean_s_dyn[layer_name]) for layer_name in LAYER_ORDER
                }

                results_rows.append(
                    {
                        "record_id": int(record_id),
                        "pair_id": pair_id,
                        "sample_id": int(pair_row.sample_id),
                        "probe_id": int(pair_row.probe_id),
                        "sample_label": int(pair_row.sample_label),
                        "probe_label": int(pair_row.probe_label),
                        "similarity_public_or_initial": float(pair_row.similarity_public_or_initial),
                        "similarity_bin": str(pair_row.similarity_bin),
                        "similarity_bin_index": int(pair_row.similarity_bin_index),
                        "condition": str(condition_name),
                        "overlap_area": int(mask_bundle.metadata["sample_overlap_area"]),
                        "control_area": int(mask_bundle.metadata["sample_control_area"]),
                        "sample_foreground_area": int(mask_bundle.metadata["sample_foreground_area"]),
                        "sample_nonoverlap_area": int(mask_bundle.metadata["sample_nonoverlap_area"]),
                        "overlap_energy": float(mask_bundle.metadata["sample_overlap_energy"]),
                        "control_energy": float(mask_bundle.metadata["sample_control_energy"]),
                        "sample_nonoverlap_energy": float(mask_bundle.metadata["sample_nonoverlap_energy"]),
                        "control_energy_gap": float(mask_bundle.metadata["sample_control_energy_gap"]),
                        "control_source": str(mask_bundle.metadata["sample_control_source"]),
                        "readout_step": int(rollout.readout_step),
                        "prediction_probe": int(condition_metric["predicted_label"]),
                        "first_fire_t_probe": int(condition_metric["first_fire_t_probe"]),
                        "DPI_L1": float(condition_metric["DPI"]["L1"]),
                        "DPI_L2": float(condition_metric["DPI"]["L2"]),
                        "DPI_L3": float(condition_metric["DPI"]["L3"]),
                        "DPI_final": float(condition_metric["DPI"]["final"]),
                        "mean_S_dyn_L1": mean_s_dyn["L1"],
                        "mean_S_dyn_L2": mean_s_dyn["L2"],
                        "mean_S_dyn_L3": mean_s_dyn["L3"],
                        "mean_S_dyn_final": mean_s_dyn["final"],
                        "mean_S_sta_L1": mean_s_sta["L1"],
                        "mean_S_sta_L2": mean_s_sta["L2"],
                        "mean_S_sta_L3": mean_s_sta["L3"],
                        "mean_S_sta_final": mean_s_sta["final"],
                        "Retain_dyn_L1": retain_dyn["L1"],
                        "Retain_dyn_L2": retain_dyn["L2"],
                        "Retain_dyn_L3": retain_dyn["L3"],
                        "Retain_dyn_final": retain_dyn["final"],
                        "Pull_sta_L1": mean_s_sta["L1"],
                        "Pull_sta_L2": mean_s_sta["L2"],
                        "Pull_sta_L3": mean_s_sta["L3"],
                        "Pull_sta_final": mean_s_sta["final"],
                        "S_dyn_final": float(condition_metric["S_dyn"]["final"]),
                        "S_sta_final": float(condition_metric["S_sta"]["final"]),
                    }
                )

                trace_records["condition_name"].append(str(condition_name))
                trace_records["DPI_L3"].append(
                    np.asarray(condition_metric["S_dyn"]["L3"] - condition_metric["S_sta"]["L3"], dtype=np.float32)
                )

    df_results = pd.DataFrame(results_rows).sort_values(["record_id"], kind="stable").reset_index(drop=True)
    trace_arrays = finalize_trace_arrays(trace_records, spec.probe_steps)
    trace_npz = data_dir / "pair_trace_similarity.npz"
    np.savez_compressed(trace_npz, **trace_arrays)

    summary_metrics = build_summary_metrics(df_results)
    summary_payload = build_summary_payload(summary_metrics)
    figure_paths = {} if bool(args.skip_figures) else save_main_figures(figures_dir, df_results, trace_arrays)

    output_paths = {
        "trace_npz": str(trace_npz.resolve()),
        "figure_paths": figure_paths,
    }
    run_config_path, summary_path, run_log_path = save_metadata_files(
        args=args,
        device=device,
        result_root=result_root,
        log_dir=log_dir,
        readout_step=readout_step,
        condition_order=condition_order,
        summary_payload=summary_payload,
        output_paths=output_paths,
        start_time=start_time,
        n_pairs=int(df_results["pair_id"].nunique()) if not df_results.empty else 0,
    )

    print(f"\n=== {EXPERIMENT_NAME} ===")
    print(f"Primary focus: {PRIMARY_FOCUS}")
    print(f"Pairs: {int(df_results['pair_id'].nunique()) if not df_results.empty else 0}")
    for condition_name in PRIMARY_ANALYSIS_CONDITIONS:
        subset = df_results[df_results["condition"] == condition_name]
        if subset.empty:
            continue
        print(
            f"{condition_name}: "
            f"mean_S_dyn_L3={float(subset['mean_S_dyn_L3'].mean()):.4f}, "
            f"mean_S_sta_L3={float(subset['mean_S_sta_L3'].mean()):.4f}, "
            f"mean_DPI_L3={float(subset['DPI_L3'].mean()):.4f}"
        )
    print(f"Trace NPZ: {trace_npz}")
    print(f"Summary JSON: {summary_path}")
    print(f"Run config JSON: {run_config_path}")
    print(f"Run log: {run_log_path}")


if __name__ == "__main__":
    main()
