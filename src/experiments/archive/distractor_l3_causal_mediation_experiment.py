from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

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
from src.experiments.common.dataset import build_class_index
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.voltage_readout import resolve_readout_step
from src.experiments.common.seed import mix_seed
from src.experiments.distractor.shared.l3_replay import (
    Layer3ReplaySnapshot,
    _snapshot_layer3_for_replay,
    replay_layer3_probe_phase,
)
from src.experiments.distractor.shared.masking import apply_input_mask_to_spike_batch
from src.experiments.distractor.shared.pair_sampling import (
    build_dataset_arrays,
    extract_grouped_voltage_vector,
)
from src.experiments.distractor.shared.triplets import (
    ExperimentSpec,
    TripletMaskBundle,
    build_probe_relevant_masks_for_triplet,
    build_triplet_specs,
    prepare_triplet_spike_batch,
)
from src.plotting.common.io import (
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
)
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    ALPHA_SCATTER,
    DISTRACTOR_CONTROL_CONDITION_COLORS,
    DISTRACTOR_MAIN_CONDITION_COLORS,
    DISTRACTOR_MEDIATION_SWAP_COLORS,
    FIGSIZE_THREE_PANEL_SUMMARY,
    FIGSIZE_TWO_PANEL,
    GRID_ALPHA,
    LINE_WIDTH_REFERENCE,
    apply_standard_legend,
)

DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_OUTPUT_DIR = "results/distractor_l3_causal_mediation_experiment"
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
DEFAULT_SAVE_CASE_COUNT = 4
DEFAULT_NUM_SIM_BINS = 5
DEFAULT_FOREGROUND_THRESHOLD = 0.0
DEFAULT_NUM_CONTROL_CANDIDATES = 32

DEFAULT_SWAP_MODES: tuple[str, ...] = ("onset_only", "trace_only", "onset_and_trace")
DEFAULT_CONDITIONS: tuple[str, ...] = (
    "sample_remove_SPonly",
    "distractor_remove_DPonly",
    "sample_remove_SDP",
    "distractor_remove_SDP",
    "both_remove_SDP",
    "sample_remove_SPonly_control",
    "distractor_remove_DPonly_control",
    "sample_remove_SDP_control",
    "distractor_remove_SDP_control",
    "both_remove_SDP_control",
)
FOCUS_CONDITIONS: tuple[str, ...] = (
    "sample_remove_SDP",
    "distractor_remove_SDP",
    "both_remove_SDP",
    "sample_remove_SPonly",
    "distractor_remove_DPonly",
)
SWAP_MODE_COLORS: dict[str, str] = dict(DISTRACTOR_MEDIATION_SWAP_COLORS)
CONDITION_COLORS: dict[str, str] = {
    **DISTRACTOR_MAIN_CONDITION_COLORS,
    **DISTRACTOR_CONTROL_CONDITION_COLORS,
}


@dataclass(frozen=True)
class MediationConditionSpec:
    """Maps a public condition name onto the existing mask bundle keys."""

    name: str
    sample_mask_key: str | None
    distractor_mask_key: str | None


@dataclass(frozen=True)
class DistractorL3TraceCaptureResult:
    """Full distractor rollout with L3 probe-phase replay state."""

    grouped_voltage: np.ndarray
    readout_snapshot: torch.Tensor
    probe_s2p_trace: torch.Tensor
    probe_onset_snapshot: Layer3ReplaySnapshot
    prediction_probe: np.ndarray
    first_fire_t_probe: np.ndarray
    readout_step: int


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        return None
    return scalar


def _to_json_ready(value):
    if isinstance(value, dict):
        return {str(key): _to_json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
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


def _center_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape {arr.shape}")
    return arr - float(np.mean(arr))


def _safe_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64).reshape(-1), ord=2))


def _safe_sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 1:
        return 0.0
    return float(np.std(finite, ddof=1) / np.sqrt(finite.size))


def _safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= float(eps):
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _parse_comma_list(raw_values: Sequence[str] | None, *, default: Sequence[str]) -> list[str]:
    if not raw_values:
        return list(default)
    items: list[str] = []
    for value in raw_values:
        for token in str(value).split(","):
            text = token.strip()
            if text:
                items.append(text)
    return list(dict.fromkeys(items))


def _condition_spec_table() -> dict[str, MediationConditionSpec]:
    return {
        "sample_remove_SPonly": MediationConditionSpec("sample_remove_SPonly", "sample_sp_only_mask", None),
        "distractor_remove_DPonly": MediationConditionSpec("distractor_remove_DPonly", None, "distractor_dp_only_mask"),
        "sample_remove_SDP": MediationConditionSpec("sample_remove_SDP", "sample_sdp_mask", None),
        "distractor_remove_SDP": MediationConditionSpec("distractor_remove_SDP", None, "distractor_sdp_mask"),
        "both_remove_SDP": MediationConditionSpec("both_remove_SDP", "sample_sdp_mask", "distractor_sdp_mask"),
        "sample_remove_SPonly_control": MediationConditionSpec("sample_remove_SPonly_control", "sample_sp_only_control_mask", None),
        "distractor_remove_DPonly_control": MediationConditionSpec("distractor_remove_DPonly_control", None, "distractor_dp_only_control_mask"),
        "sample_remove_SDP_control": MediationConditionSpec("sample_remove_SDP_control", "sample_sdp_control_mask", None),
        "distractor_remove_SDP_control": MediationConditionSpec("distractor_remove_SDP_control", None, "distractor_sdp_control_mask"),
        "both_remove_SDP_control": MediationConditionSpec(
            "both_remove_SDP_control",
            "sample_sdp_control_mask",
            "distractor_sdp_control_mask",
        ),
    }


def _validate_choice_subset(values: Sequence[str], *, allowed: Sequence[str], argument_name: str) -> list[str]:
    allowed_set = set(allowed)
    invalid = [value for value in values if value not in allowed_set]
    if invalid:
        raise ValueError(f"Unsupported {argument_name}: {invalid}. Allowed: {sorted(allowed_set)}")
    return list(values)


def _build_condition_mask_batch(mask_records: Sequence[TripletMaskBundle], mask_key: str | None) -> torch.Tensor | None:
    if mask_key is None:
        return None
    stacked = np.stack([np.asarray(getattr(record, mask_key), dtype=bool) for record in mask_records], axis=0)
    return torch.as_tensor(stacked, dtype=torch.bool)


def _snapshot_field_to_numpy(field: torch.Tensor | None, index: int) -> np.ndarray:
    if field is None:
        return np.zeros((1,), dtype=np.float32)
    if field.shape[0] == 1:
        sliced = field
    else:
        sliced = field[int(index):int(index) + 1]
    return sliced.detach().cpu().numpy().astype(np.float32, copy=False)


def compose_swapped_l3_state(
    *,
    recipient_probe_onset_snapshot: Layer3ReplaySnapshot,
    recipient_probe_s2p_trace: torch.Tensor,
    donor_probe_onset_snapshot: Layer3ReplaySnapshot,
    donor_probe_s2p_trace: torch.Tensor,
    swap_mode: str,
) -> tuple[Layer3ReplaySnapshot, torch.Tensor]:
    """Build probe-onset snapshot and probe trace for a requested swap mode."""

    mode = str(swap_mode)
    if mode == "onset_only":
        return donor_probe_onset_snapshot, recipient_probe_s2p_trace
    if mode == "trace_only":
        return recipient_probe_onset_snapshot, donor_probe_s2p_trace
    if mode == "onset_and_trace":
        return donor_probe_onset_snapshot, donor_probe_s2p_trace
    raise ValueError(f"Unsupported swap_mode: {swap_mode}")


def run_distractor_with_l3_trace_capture(
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
) -> DistractorL3TraceCaptureResult:
    """Run a distractor trial and capture the L3 state needed for replay."""

    if sample_spikes.ndim != 5 or distractor_spikes.ndim != 5 or probe_spikes.ndim != 5:
        raise ValueError("sample_spikes, distractor_spikes, and probe_spikes must have shape [B, T, C, H, W]")
    batch_size, _, channels, height, width = sample_spikes.shape
    if tuple(distractor_spikes.shape[:1]) != (batch_size,) or tuple(probe_spikes.shape[:1]) != (batch_size,):
        raise ValueError("All spike tensors must share the same batch size")
    if int(readout_step) < 0 or int(readout_step) >= int(probe_spikes.shape[1]):
        raise ValueError(f"readout_step={readout_step} must fall within the probe phase")

    device = sample_spikes.device
    zero_input = torch.zeros((batch_size, channels, height, width), dtype=sample_spikes.dtype, device=device)
    masked_sample_spikes = apply_input_mask_to_spike_batch(sample_spikes, sample_input_mask, mode="remove")
    masked_distractor_spikes = apply_input_mask_to_spike_batch(distractor_spikes, distractor_input_mask, mode="remove")
    prepare_network_state(net, batch_size, channels, height, width)
    current_time = 0
    readout_snapshot = None
    probe_s2p_chunks: list[torch.Tensor] = []

    def reset_decision_window() -> None:
        net.layer3.reset_decision_state()
        if bool(phase_reset):
            with torch.no_grad():
                net.layer3.v_mem.fill_(net.layer3.V_L)
                net.layer3.lateral_inh.reset_state(net.layer3.output_shape)

    def step_network(input_t: torch.Tensor, *, capture_probe_trace: bool, probe_phase_step: int | None, force_l3_time: int | None = None) -> None:
        nonlocal current_time, readout_snapshot
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())
        if capture_probe_trace:
            probe_s2p_chunks.append(s2_p.detach().cpu().to(torch.float32))
        l3_time = current_time if force_l3_time is None else force_l3_time
        _, monitor_data = net.layer3.forward_step(
            s2_p,
            l3_time,
            training=False,
            monitor=bool(capture_probe_trace and probe_phase_step is not None and int(probe_phase_step) == int(readout_step)),
            stsp_mode=stsp_mode,
        )
        if capture_probe_trace and probe_phase_step is not None and int(probe_phase_step) == int(readout_step):
            if "v_mem_snapshot" not in monitor_data:
                raise RuntimeError("Layer-3 readout snapshot was not captured during the probe phase")
            readout_snapshot = monitor_data["v_mem_snapshot"].detach().cpu().to(torch.float32)
        current_time += 1

    with torch.no_grad():
        for t_step in range(int(masked_sample_spikes.shape[1])):
            step_network(masked_sample_spikes[:, t_step, ...], capture_probe_trace=False, probe_phase_step=None)
        for _ in range(int(delay1_steps)):
            step_network(zero_input, capture_probe_trace=False, probe_phase_step=None)
        reset_decision_window()
        for t_step in range(int(masked_distractor_spikes.shape[1])):
            force_t = int(t_step) if bool(phase_reset) else None
            step_network(
                masked_distractor_spikes[:, t_step, ...],
                capture_probe_trace=False,
                probe_phase_step=None,
                force_l3_time=force_t,
            )
        for _ in range(int(delay2_steps)):
            step_network(zero_input, capture_probe_trace=False, probe_phase_step=None)
        reset_decision_window()
        probe_onset_snapshot = _snapshot_layer3_for_replay(net, readout_step=int(readout_step))
        for t_step in range(int(probe_spikes.shape[1])):
            force_t = int(t_step) if bool(phase_reset) else None
            step_network(
                probe_spikes[:, t_step, ...],
                capture_probe_trace=True,
                probe_phase_step=int(t_step),
                force_l3_time=force_t,
            )

    if readout_snapshot is None:
        raise RuntimeError("Requested probe readout snapshot was not produced")
    flat_times = net.layer3.firing_times.detach().cpu()
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    prediction_probe = (min_indices // net.layer3.neurons_per_class).long()
    prediction_probe[~has_fired] = -1
    first_fire_t_probe = min_times.clone()
    first_fire_t_probe[~has_fired] = -1
    return DistractorL3TraceCaptureResult(
        grouped_voltage=extract_grouped_voltage_vector(net, readout_snapshot),
        readout_snapshot=readout_snapshot,
        probe_s2p_trace=torch.stack(probe_s2p_chunks, dim=1),
        probe_onset_snapshot=probe_onset_snapshot,
        prediction_probe=prediction_probe.numpy().astype(np.int64, copy=False),
        first_fire_t_probe=first_fire_t_probe.numpy().astype(np.int64, copy=False),
        readout_step=int(readout_step),
    )


def replay_distractor_l3_probe_phase(
    net,
    probe_onset_snapshot: Layer3ReplaySnapshot,
    modified_probe_s2p_trace: torch.Tensor,
    *,
    stsp_mode: str,
) -> Dict[str, object]:
    """Replay the distractor probe phase from an L3 onset snapshot and trace."""

    return replay_layer3_probe_phase(
        net=net,
        probe_onset_snapshot=probe_onset_snapshot,
        modified_probe_s2p_trace=modified_probe_s2p_trace,
        stsp_mode=stsp_mode,
    )


def compute_mediation_metrics(
    *,
    V_full: np.ndarray,
    V_pert: np.ndarray,
    V_push: np.ndarray | None = None,
    V_rescue: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, object]:
    """Compute centered-voltage mediation metrics with numerical protection."""

    v_full = np.asarray(V_full, dtype=np.float64).reshape(-1)
    v_pert = np.asarray(V_pert, dtype=np.float64).reshape(-1)
    if v_full.shape != v_pert.shape:
        raise ValueError(f"V_full shape {v_full.shape} does not match V_pert shape {v_pert.shape}")
    z_full = _center_vector(v_full)
    z_pert = _center_vector(v_pert)
    delta_total = z_pert - z_full
    total_effect_norm = _safe_norm(delta_total)

    if V_push is None:
        z_push = np.full_like(z_full, np.nan)
        delta_push = np.full_like(z_full, np.nan)
        push_effect_norm = float("nan")
        push_cosine = float("nan")
        push_direction_match = float("nan")
        push_magnitude_ratio = float("nan")
    else:
        v_push = np.asarray(V_push, dtype=np.float64).reshape(-1)
        if v_push.shape != v_full.shape:
            raise ValueError(f"V_push shape {v_push.shape} does not match V_full shape {v_full.shape}")
        z_push = _center_vector(v_push)
        delta_push = z_push - z_full
        push_effect_norm = _safe_norm(delta_push)
        push_cosine = _safe_cosine(delta_push, delta_total, eps=eps)
        if push_effect_norm <= float(eps) or total_effect_norm <= float(eps):
            push_direction_match = 0.0
        else:
            push_direction_match = float(int(np.argmax(delta_push) == np.argmax(delta_total)))
        push_magnitude_ratio = float(push_effect_norm / max(total_effect_norm, float(eps)))

    if V_rescue is None:
        z_rescue = np.full_like(z_full, np.nan)
        delta_rescue = np.full_like(z_full, np.nan)
        rescue_effect_norm = float("nan")
        rescue_ratio = float("nan")
        rescue_distance_reduction = float("nan")
    else:
        v_rescue = np.asarray(V_rescue, dtype=np.float64).reshape(-1)
        if v_rescue.shape != v_full.shape:
            raise ValueError(f"V_rescue shape {v_rescue.shape} does not match V_full shape {v_full.shape}")
        z_rescue = _center_vector(v_rescue)
        delta_rescue = z_rescue - z_pert
        rescue_effect_norm = _safe_norm(delta_rescue)
        total_distance = _safe_norm(v_pert - v_full)
        residual_distance = _safe_norm(v_rescue - v_full)
        rescue_ratio = float(1.0 - residual_distance / max(total_distance, float(eps)))
        rescue_distance_reduction = float(total_distance - residual_distance)

    return {
        "z_full": z_full,
        "z_pert": z_pert,
        "z_push": z_push,
        "z_rescue": z_rescue,
        "delta_total": delta_total,
        "delta_push": delta_push,
        "delta_rescue": delta_rescue,
        "total_effect_norm": float(total_effect_norm),
        "push_effect_norm": float(push_effect_norm),
        "rescue_effect_norm": float(rescue_effect_norm),
        "push_cosine": float(push_cosine),
        "push_direction_match": float(push_direction_match),
        "rescue_ratio": float(rescue_ratio),
        "push_magnitude_ratio": float(push_magnitude_ratio),
        "rescue_distance_reduction": float(rescue_distance_reduction),
    }


def build_full_perturbed_push_rescue_records(
    *,
    net,
    sample_spikes: torch.Tensor,
    distractor_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    batch_df: pd.DataFrame,
    batch_masks: Sequence[TripletMaskBundle],
    spec: ExperimentSpec,
    readout_step: int,
    condition_specs: Mapping[str, MediationConditionSpec],
    conditions: Sequence[str],
    swap_modes: Sequence[str],
    stsp_mode: str,
    skip_push: bool,
    skip_rescue: bool,
    trace_case_triplet_ids: set[int] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Build full / perturbed / push / rescue records for one batch."""

    full_capture = run_distractor_with_l3_trace_capture(
        net=net,
        sample_spikes=sample_spikes,
        distractor_spikes=distractor_spikes,
        probe_spikes=probe_spikes,
        delay1_steps=int(spec.delay1_steps),
        delay2_steps=int(spec.delay2_steps),
        stsp_mode=stsp_mode,
        readout_step=int(readout_step),
        phase_reset=bool(spec.phase_reset),
    )

    perturbed_captures: dict[str, DistractorL3TraceCaptureResult] = {}
    push_outputs: dict[tuple[str, str], dict[str, object]] = {}
    rescue_outputs: dict[tuple[str, str], dict[str, object]] = {}

    for condition_name in conditions:
        condition_spec = condition_specs[str(condition_name)]
        sample_mask = _build_condition_mask_batch(batch_masks, condition_spec.sample_mask_key)
        distractor_mask = _build_condition_mask_batch(batch_masks, condition_spec.distractor_mask_key)
        perturbed_captures[condition_name] = run_distractor_with_l3_trace_capture(
            net=net,
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            probe_spikes=probe_spikes,
            delay1_steps=int(spec.delay1_steps),
            delay2_steps=int(spec.delay2_steps),
            stsp_mode=stsp_mode,
            readout_step=int(readout_step),
            sample_input_mask=None if sample_mask is None else sample_mask.to(device=sample_spikes.device),
            distractor_input_mask=None if distractor_mask is None else distractor_mask.to(device=sample_spikes.device),
            phase_reset=bool(spec.phase_reset),
        )
        for swap_mode in swap_modes:
            if not bool(skip_push):
                push_snapshot, push_trace = compose_swapped_l3_state(
                    recipient_probe_onset_snapshot=full_capture.probe_onset_snapshot,
                    recipient_probe_s2p_trace=full_capture.probe_s2p_trace,
                    donor_probe_onset_snapshot=perturbed_captures[condition_name].probe_onset_snapshot,
                    donor_probe_s2p_trace=perturbed_captures[condition_name].probe_s2p_trace,
                    swap_mode=str(swap_mode),
                )
                push_outputs[(condition_name, str(swap_mode))] = replay_distractor_l3_probe_phase(
                    net=net,
                    probe_onset_snapshot=push_snapshot,
                    modified_probe_s2p_trace=push_trace,
                    stsp_mode=stsp_mode,
                )
            if not bool(skip_rescue):
                rescue_snapshot, rescue_trace = compose_swapped_l3_state(
                    recipient_probe_onset_snapshot=perturbed_captures[condition_name].probe_onset_snapshot,
                    recipient_probe_s2p_trace=perturbed_captures[condition_name].probe_s2p_trace,
                    donor_probe_onset_snapshot=full_capture.probe_onset_snapshot,
                    donor_probe_s2p_trace=full_capture.probe_s2p_trace,
                    swap_mode=str(swap_mode),
                )
                rescue_outputs[(condition_name, str(swap_mode))] = replay_distractor_l3_probe_phase(
                    net=net,
                    probe_onset_snapshot=rescue_snapshot,
                    modified_probe_s2p_trace=rescue_trace,
                    stsp_mode=stsp_mode,
                )

    records: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    record_id = 0
    for batch_idx, triplet_row in enumerate(batch_df.itertuples(index=False)):
        triplet_meta = dict(triplet_row._asdict())
        v_full = np.asarray(full_capture.grouped_voltage[batch_idx], dtype=np.float64)
        pred_full = int(full_capture.prediction_probe[batch_idx])
        fire_full = int(full_capture.first_fire_t_probe[batch_idx])
        for condition_name in conditions:
            pert_capture = perturbed_captures[condition_name]
            v_pert = np.asarray(pert_capture.grouped_voltage[batch_idx], dtype=np.float64)
            pred_pert = int(pert_capture.prediction_probe[batch_idx])
            fire_pert = int(pert_capture.first_fire_t_probe[batch_idx])
            for swap_mode in swap_modes:
                push_output = push_outputs.get((condition_name, str(swap_mode)))
                rescue_output = rescue_outputs.get((condition_name, str(swap_mode)))
                v_push = None if push_output is None else np.asarray(push_output["grouped_voltage"][batch_idx], dtype=np.float64)
                v_rescue = None if rescue_output is None else np.asarray(rescue_output["grouped_voltage"][batch_idx], dtype=np.float64)
                metrics = compute_mediation_metrics(V_full=v_full, V_pert=v_pert, V_push=v_push, V_rescue=v_rescue)
                prediction_push = -1 if push_output is None else int(push_output["prediction_probe"][batch_idx])
                prediction_rescue = -1 if rescue_output is None else int(rescue_output["prediction_probe"][batch_idx])
                first_fire_push = -1 if push_output is None else int(push_output["first_fire_t_probe"][batch_idx])
                first_fire_rescue = -1 if rescue_output is None else int(rescue_output["first_fire_t_probe"][batch_idx])
                current_record_id = int(record_id)
                record_id += 1
                records.append(
                    {
                        "record_id": current_record_id,
                        "analysis_family": "distractor_l3_causal_mediation",
                        "triplet_id": int(triplet_row.triplet_id),
                        "condition": str(condition_name),
                        "swap_mode": str(swap_mode),
                        "stsp_mode": str(stsp_mode),
                        "push_available": int(push_output is not None),
                        "rescue_available": int(rescue_output is not None),
                        "sample_id": int(triplet_row.sample_id),
                        "sample_label": int(triplet_row.sample_label),
                        "distractor_id": int(triplet_row.distractor_id),
                        "distractor_label": int(triplet_row.distractor_label),
                        "probe_id": int(triplet_row.probe_id),
                        "probe_label": int(triplet_row.probe_label),
                        "prediction_full": pred_full,
                        "prediction_pert": pred_pert,
                        "prediction_push": prediction_push,
                        "prediction_rescue": prediction_rescue,
                        "first_fire_t_full": fire_full,
                        "first_fire_t_pert": fire_pert,
                        "first_fire_t_push": first_fire_push,
                        "first_fire_t_rescue": first_fire_rescue,
                        "readout_step": int(full_capture.readout_step),
                        "total_effect_norm": float(metrics["total_effect_norm"]),
                        "push_effect_norm": float(metrics["push_effect_norm"]),
                        "rescue_effect_norm": float(metrics["rescue_effect_norm"]),
                        "push_cosine": float(metrics["push_cosine"]),
                        "push_direction_match": float(metrics["push_direction_match"]),
                        "rescue_ratio": float(metrics["rescue_ratio"]),
                        "push_magnitude_ratio": float(metrics["push_magnitude_ratio"]),
                        "rescue_distance_reduction": float(metrics["rescue_distance_reduction"]),
                        **triplet_meta,
                    }
                )
                vector_rows.append(
                    {
                        "record_id": current_record_id,
                        "triplet_id": int(triplet_row.triplet_id),
                        "condition": str(condition_name),
                        "swap_mode": str(swap_mode),
                        "V_full": v_full.astype(np.float32, copy=False),
                        "V_pert": v_pert.astype(np.float32, copy=False),
                        "V_push": (np.full_like(v_full, np.nan) if v_push is None else v_push).astype(np.float32, copy=False),
                        "V_rescue": (np.full_like(v_full, np.nan) if v_rescue is None else v_rescue).astype(np.float32, copy=False),
                        "delta_total": np.asarray(metrics["delta_total"], dtype=np.float32),
                        "delta_push": np.asarray(metrics["delta_push"], dtype=np.float32),
                        "delta_rescue": np.asarray(metrics["delta_rescue"], dtype=np.float32),
                    }
                )
                should_save_trace_case = (
                    trace_case_triplet_ids is not None
                    and int(triplet_row.triplet_id) in trace_case_triplet_ids
                    and str(swap_mode) == "onset_and_trace"
                )
                if should_save_trace_case:
                    trace_rows.append(
                        {
                            "record_id": current_record_id,
                            "triplet_id": int(triplet_row.triplet_id),
                            "condition": str(condition_name),
                            "swap_mode": str(swap_mode),
                            "full_probe_s2p_trace": full_capture.probe_s2p_trace[batch_idx].detach().cpu().numpy().astype(np.float32, copy=False),
                            "perturbed_probe_s2p_trace": pert_capture.probe_s2p_trace[batch_idx].detach().cpu().numpy().astype(np.float32, copy=False),
                            "full_snapshot_v_mem": _snapshot_field_to_numpy(full_capture.probe_onset_snapshot.v_mem, batch_idx),
                            "perturbed_snapshot_v_mem": _snapshot_field_to_numpy(pert_capture.probe_onset_snapshot.v_mem, batch_idx),
                            "full_snapshot_g_e": _snapshot_field_to_numpy(full_capture.probe_onset_snapshot.g_e, batch_idx),
                            "perturbed_snapshot_g_e": _snapshot_field_to_numpy(pert_capture.probe_onset_snapshot.g_e, batch_idx),
                            "full_snapshot_res": _snapshot_field_to_numpy(full_capture.probe_onset_snapshot.res, batch_idx),
                            "perturbed_snapshot_res": _snapshot_field_to_numpy(pert_capture.probe_onset_snapshot.res, batch_idx),
                            "full_snapshot_inh_trace": _snapshot_field_to_numpy(full_capture.probe_onset_snapshot.inh_trace, batch_idx),
                            "perturbed_snapshot_inh_trace": _snapshot_field_to_numpy(pert_capture.probe_onset_snapshot.inh_trace, batch_idx),
                            "full_snapshot_u_pre": _snapshot_field_to_numpy(full_capture.probe_onset_snapshot.u_pre, batch_idx),
                            "perturbed_snapshot_u_pre": _snapshot_field_to_numpy(pert_capture.probe_onset_snapshot.u_pre, batch_idx),
                            "full_snapshot_x_pre": _snapshot_field_to_numpy(full_capture.probe_onset_snapshot.x_pre, batch_idx),
                            "perturbed_snapshot_x_pre": _snapshot_field_to_numpy(pert_capture.probe_onset_snapshot.x_pre, batch_idx),
                            "full_snapshot_firing_times": _snapshot_field_to_numpy(full_capture.probe_onset_snapshot.firing_times, batch_idx),
                            "perturbed_snapshot_firing_times": _snapshot_field_to_numpy(pert_capture.probe_onset_snapshot.firing_times, batch_idx),
                        }
                    )

    return {"records": records, "vector_rows": vector_rows, "trace_rows": trace_rows}


def summarize_mediation_results(
    df_results: pd.DataFrame,
    *,
    conditions: Sequence[str],
    swap_modes: Sequence[str],
) -> dict[str, object]:
    grouped_rows: list[dict[str, object]] = []
    for condition_name in conditions:
        for swap_mode in swap_modes:
            subset = df_results[(df_results["condition"] == str(condition_name)) & (df_results["swap_mode"] == str(swap_mode))].copy()
            if subset.empty:
                continue
            grouped_rows.append(
                {
                    "condition": str(condition_name),
                    "swap_mode": str(swap_mode),
                    "n_records": int(len(subset)),
                    "n_triplets": int(subset["triplet_id"].nunique()),
                    "push_cosine_mean": float(subset["push_cosine"].mean(skipna=True)),
                    "push_cosine_sem": float(_safe_sem(subset["push_cosine"].to_numpy(dtype=np.float64))),
                    "direction_match_rate_mean": float(subset["push_direction_match"].mean(skipna=True)),
                    "direction_match_rate_sem": float(_safe_sem(subset["push_direction_match"].to_numpy(dtype=np.float64))),
                    "rescue_ratio_mean": float(subset["rescue_ratio"].mean(skipna=True)),
                    "rescue_ratio_sem": float(_safe_sem(subset["rescue_ratio"].to_numpy(dtype=np.float64))),
                    "total_effect_norm_mean": float(subset["total_effect_norm"].mean(skipna=True)),
                    "push_effect_norm_mean": float(subset["push_effect_norm"].mean(skipna=True)),
                    "rescue_effect_norm_mean": float(subset["rescue_effect_norm"].mean(skipna=True)),
                }
            )

    focus_rows = [row for row in grouped_rows if str(row["condition"]) in set(FOCUS_CONDITIONS)]
    return {
        "overall": {
            "n_records": int(len(df_results)),
            "n_triplets": int(df_results["triplet_id"].nunique()) if len(df_results) else 0,
            "n_probes": int(df_results["probe_id"].nunique()) if len(df_results) else 0,
            "main_swap_mode": "onset_and_trace",
        },
        "condition_swap_summary": grouped_rows,
        "focus_condition_summary": focus_rows,
        "assumptions": {
            "experiment_scope": "This experiment tests causal mediation only and does not compare input-area contribution, total contribution, or unit-area efficiency.",
            "grouped_voltage": "layer3.get_grouped_voltage(v_mem_snapshot).mean(-1)",
            "centered_grouped_voltage": "z(V) = V - mean(V)",
            "push_definition": "full upstream history with probe-phase L3 replaced by the perturbed donor state",
            "rescue_definition": "perturbed upstream history with probe-phase L3 replaced by the full donor state",
            "trace_snapshot_export": "To stay within workstation memory, trace/snapshot NPZ stores selected case triplets only, using onset_and_trace rows and omitting heavy snapshot fields such as input_trace and eligibility_trace.",
            "swap_modes": {
                "onset_only": "replace probe_onset_snapshot only",
                "trace_only": "replace probe_s2p_trace only",
                "onset_and_trace": "replace both probe_onset_snapshot and probe_s2p_trace",
            },
            "probe_perturbation": "disabled",
        },
    }


def plot_mediation_flow_schematic() -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(12.0, 3.8))
    ax.axis("off")
    x_positions = [0.1, 0.36, 0.62, 0.88]
    titles = ["Full", "Perturbed", "Push", "Rescue"]
    subtitles = [
        "full history\nfull L3\nV_full",
        "perturbed history\nperturbed L3\nV_pert",
        "full history\nperturbed L3\nV_push",
        "perturbed history\nfull L3\nV_rescue",
    ]
    fill_colors = ["#DCEAF7", "#FDE2E4", "#E8F3E8", "#F6E8FF"]
    for xpos, title, subtitle, fill in zip(x_positions, titles, subtitles, fill_colors):
        ax.text(
            xpos,
            0.6,
            f"{title}\n{subtitle}",
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": fill, "edgecolor": "#333333"},
            transform=ax.transAxes,
        )
    for left, right in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate(
            "",
            xy=(right - 0.09, 0.6),
            xytext=(left + 0.09, 0.6),
            arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "#333333"},
            xycoords=ax.transAxes,
        )
    ax.text(0.5, 0.15, "Question: does the output change track the swapped L3 state?", ha="center", va="center", fontsize=12, transform=ax.transAxes)
    fig.tight_layout()
    return fig


def plot_summary_bars(df_results: pd.DataFrame, *, conditions: Sequence[str], swap_modes: Sequence[str]) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL_SUMMARY, sharex=True)
    metrics = [
        ("push_cosine", "Push cosine"),
        ("push_direction_match", "Direction match rate"),
        ("rescue_ratio", "Rescue ratio"),
    ]
    x = np.arange(len(conditions), dtype=np.float64)
    width = 0.24
    offsets = np.linspace(-width, width, num=len(swap_modes))
    for ax, (metric_name, title) in zip(axes, metrics):
        for offset, swap_mode in zip(offsets, swap_modes):
            means = []
            sems = []
            for condition_name in conditions:
                subset = df_results[(df_results["condition"] == str(condition_name)) & (df_results["swap_mode"] == str(swap_mode))]
                values = subset[metric_name].to_numpy(dtype=np.float64, copy=False) if len(subset) else np.asarray([], dtype=np.float64)
                means.append(float(np.nanmean(values)) if values.size else float("nan"))
                sems.append(float(_safe_sem(values)))
            ax.bar(
                x + float(offset),
                np.asarray(means, dtype=np.float64),
                width=width,
                color=SWAP_MODE_COLORS[str(swap_mode)],
                label=str(swap_mode),
                alpha=ALPHA_BAR,
            )
            ax.errorbar(x + float(offset), means, yerr=sems, fmt="none", ecolor="black", capsize=3, elinewidth=LINE_WIDTH_REFERENCE)
        ax.set_title(title)
        ax.grid(alpha=GRID_ALPHA, axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels(list(conditions), rotation=35, ha="right")
    axes[0].set_ylabel("Metric value")
    apply_standard_legend(axes[2], compact=True)
    fig.tight_layout()
    return fig


def plot_triplet_scatter(df_results: pd.DataFrame, *, conditions: Sequence[str]) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL)
    ax_left, ax_right = axes
    subset = df_results[df_results["swap_mode"] == "onset_and_trace"].copy()
    for condition_name in conditions:
        condition_subset = subset[subset["condition"] == str(condition_name)].copy()
        if condition_subset.empty:
            continue
        color = CONDITION_COLORS.get(str(condition_name), "#4C78A8")
        ax_left.scatter(
            condition_subset["total_effect_norm"].to_numpy(dtype=np.float64),
            condition_subset["push_cosine"].to_numpy(dtype=np.float64),
            color=color,
            alpha=ALPHA_SCATTER,
            label=str(condition_name),
        )
        ax_right.scatter(
            condition_subset["total_effect_norm"].to_numpy(dtype=np.float64),
            condition_subset["rescue_ratio"].to_numpy(dtype=np.float64),
            color=color,
            alpha=ALPHA_SCATTER,
            label=str(condition_name),
        )
    ax_left.set_xlabel("Total effect norm")
    ax_left.set_ylabel("Push cosine")
    ax_left.set_title("Total effect vs push alignment")
    ax_left.grid(alpha=GRID_ALPHA)
    ax_right.set_xlabel("Total effect norm")
    ax_right.set_ylabel("Rescue ratio")
    ax_right.set_title("Total effect vs rescue ratio")
    ax_right.grid(alpha=GRID_ALPHA)
    apply_standard_legend(ax_left, compact=True)
    fig.tight_layout()
    return fig


def _stack_vector_rows(rows: Sequence[dict[str, object]], key: str, fallback_shape: tuple[int, ...]) -> np.ndarray:
    if not rows:
        return np.zeros(fallback_shape, dtype=np.float32)
    return np.stack([np.asarray(row[key], dtype=np.float32) for row in rows], axis=0)


def _select_case_triplets(df_triplets: pd.DataFrame, save_case_count: int) -> list[int]:
    if int(save_case_count) <= 0 or df_triplets.empty:
        return []
    ordered = df_triplets.sort_values(["probe_rank", "triplet_id"], kind="stable").reset_index(drop=True)
    take = min(int(save_case_count), len(ordered))
    positions = np.linspace(0, len(ordered) - 1, num=take).astype(np.int64)
    return ordered.iloc[sorted(dict.fromkeys(int(pos) for pos in positions.tolist()))]["triplet_id"].astype(int).tolist()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distractor L3 causal mediation experiment.")
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
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--samples-per-probe", type=int, default=DEFAULT_SAMPLES_PER_PROBE)
    parser.add_argument("--max-triplets", type=int, default=DEFAULT_MAX_TRIPLETS)
    parser.add_argument("--save-case-count", type=int, default=DEFAULT_SAVE_CASE_COUNT)
    parser.add_argument("--swap-modes", nargs="*", default=list(DEFAULT_SWAP_MODES))
    parser.add_argument("--conditions", nargs="*", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--skip-rescue", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.max_probes) <= 0:
        raise ValueError("--max-probes must be positive")
    if int(args.samples_per_probe) <= 0:
        raise ValueError("--samples-per-probe must be positive")
    if int(args.max_triplets) <= 0:
        raise ValueError("--max-triplets must be positive")
    if int(args.save_case_count) < 0:
        raise ValueError("--save-case-count must be non-negative")

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
    for phase_name, steps in (
        ("sample", spec.sample_steps),
        ("delay1", spec.delay1_steps),
        ("distractor", spec.distractor_steps),
        ("delay2", spec.delay2_steps),
        ("probe", spec.probe_steps),
    ):
        if int(steps) <= 0:
            raise ValueError(f"{phase_name} steps must resolve to a positive integer")

    swap_modes = _validate_choice_subset(
        _parse_comma_list(args.swap_modes, default=DEFAULT_SWAP_MODES),
        allowed=DEFAULT_SWAP_MODES,
        argument_name="swap modes",
    )
    conditions = _validate_choice_subset(
        _parse_comma_list(args.conditions, default=DEFAULT_CONDITIONS),
        allowed=DEFAULT_CONDITIONS,
        argument_name="conditions",
    )
    condition_specs = _condition_spec_table()

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
        num_bins=int(DEFAULT_NUM_SIM_BINS),
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
                foreground_threshold=float(DEFAULT_FOREGROUND_THRESHOLD),
                use_dilated_overlap=False,
                dilation_radius=0,
                seed=mix_seed(int(args.seed), triplet_id, int(triplet_row.sample_id), int(triplet_row.distractor_id), int(triplet_row.probe_id)),
                num_control_candidates=int(DEFAULT_NUM_CONTROL_CANDIDATES),
            )
        )

    case_triplet_ids = _select_case_triplets(df_triplets, save_case_count=int(args.save_case_count))
    case_triplet_id_set = set(int(triplet_id) for triplet_id in case_triplet_ids)

    result_rows: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    batch_starts = range(0, len(df_triplets), int(args.batch_size))
    for batch_start in tqdm(batch_starts, total=math.ceil(len(df_triplets) / int(args.batch_size)), desc="DistractorL3Mediation"):
        batch_df = df_triplets.iloc[batch_start:batch_start + int(args.batch_size)].copy().reset_index(drop=True)
        batch_triplet_ids = batch_df["triplet_id"].astype(int).tolist()
        batch_masks = [mask_records[triplet_id] for triplet_id in batch_triplet_ids]
        sample_spikes, distractor_spikes, probe_spikes = prepare_triplet_spike_batch(
            images=images,
            batch_df=batch_df,
            encoder=encoder,
            spec=spec,
            device=device,
        )
        batch_outputs = build_full_perturbed_push_rescue_records(
            net=net,
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            probe_spikes=probe_spikes,
            batch_df=batch_df,
            batch_masks=batch_masks,
            spec=spec,
            readout_step=int(readout_step),
            condition_specs=condition_specs,
            conditions=conditions,
            swap_modes=swap_modes,
            stsp_mode="dynamic",
            skip_push=bool(args.skip_push),
            skip_rescue=bool(args.skip_rescue),
            trace_case_triplet_ids=case_triplet_id_set,
        )
        result_rows.extend(batch_outputs["records"])
        vector_rows.extend(batch_outputs["vector_rows"])
        trace_rows.extend(batch_outputs["trace_rows"])

    if len(result_rows) != len(vector_rows):
        raise RuntimeError("Result and vector row counts must match")
    for global_record_id, (result_row, vector_row) in enumerate(zip(result_rows, vector_rows)):
        result_row["record_id"] = int(global_record_id)
        vector_row["record_id"] = int(global_record_id)
    trace_record_ids = {int(row["record_id"]) for row in trace_rows}
    valid_record_ids = set(range(len(result_rows)))
    if not trace_record_ids.issubset(valid_record_ids):
        raise RuntimeError("Trace row record ids must reference existing analysis rows")

    df_results = pd.DataFrame(result_rows).sort_values(["record_id"], kind="stable").reset_index(drop=True)
    triplet_specs_csv = save_tidy_csv(df_triplets, output_dir / "triplet_specs.csv", sort_by=["triplet_id"])
    results_csv = save_tidy_csv(df_results, output_dir / "triplet_mediation_results.csv", sort_by=["record_id"])

    vectors_npz = output_dir / "triplet_mediation_vectors.npz"
    np.savez_compressed(
        vectors_npz,
        record_id=df_results["record_id"].to_numpy(dtype=np.int64, copy=False),
        triplet_id=df_results["triplet_id"].to_numpy(dtype=np.int64, copy=False),
        condition=df_results["condition"].to_numpy(),
        swap_mode=df_results["swap_mode"].to_numpy(),
        V_full=_stack_vector_rows(vector_rows, "V_full", (0, num_classes)),
        V_pert=_stack_vector_rows(vector_rows, "V_pert", (0, num_classes)),
        V_push=_stack_vector_rows(vector_rows, "V_push", (0, num_classes)),
        V_rescue=_stack_vector_rows(vector_rows, "V_rescue", (0, num_classes)),
        delta_total=_stack_vector_rows(vector_rows, "delta_total", (0, num_classes)),
        delta_push=_stack_vector_rows(vector_rows, "delta_push", (0, num_classes)),
        delta_rescue=_stack_vector_rows(vector_rows, "delta_rescue", (0, num_classes)),
    )

    traces_npz = output_dir / "triplet_mediation_traces_or_snapshots.npz"
    np.savez_compressed(
        traces_npz,
        record_id=np.asarray([int(row["record_id"]) for row in trace_rows], dtype=np.int64),
        triplet_id=np.asarray([int(row["triplet_id"]) for row in trace_rows], dtype=np.int64),
        condition=np.asarray([str(row["condition"]) for row in trace_rows]),
        swap_mode=np.asarray([str(row["swap_mode"]) for row in trace_rows]),
        full_probe_s2p_trace=_stack_vector_rows(trace_rows, "full_probe_s2p_trace", (0, spec.probe_steps, 1, 1, 1)),
        perturbed_probe_s2p_trace=_stack_vector_rows(trace_rows, "perturbed_probe_s2p_trace", (0, spec.probe_steps, 1, 1, 1)),
        full_snapshot_v_mem=_stack_vector_rows(trace_rows, "full_snapshot_v_mem", (0, 1)),
        perturbed_snapshot_v_mem=_stack_vector_rows(trace_rows, "perturbed_snapshot_v_mem", (0, 1)),
        full_snapshot_g_e=_stack_vector_rows(trace_rows, "full_snapshot_g_e", (0, 1)),
        perturbed_snapshot_g_e=_stack_vector_rows(trace_rows, "perturbed_snapshot_g_e", (0, 1)),
        full_snapshot_res=_stack_vector_rows(trace_rows, "full_snapshot_res", (0, 1)),
        perturbed_snapshot_res=_stack_vector_rows(trace_rows, "perturbed_snapshot_res", (0, 1)),
        full_snapshot_inh_trace=_stack_vector_rows(trace_rows, "full_snapshot_inh_trace", (0, 1)),
        perturbed_snapshot_inh_trace=_stack_vector_rows(trace_rows, "perturbed_snapshot_inh_trace", (0, 1)),
        full_snapshot_u_pre=_stack_vector_rows(trace_rows, "full_snapshot_u_pre", (0, 1)),
        perturbed_snapshot_u_pre=_stack_vector_rows(trace_rows, "perturbed_snapshot_u_pre", (0, 1)),
        full_snapshot_x_pre=_stack_vector_rows(trace_rows, "full_snapshot_x_pre", (0, 1)),
        perturbed_snapshot_x_pre=_stack_vector_rows(trace_rows, "perturbed_snapshot_x_pre", (0, 1)),
        full_snapshot_firing_times=_stack_vector_rows(trace_rows, "full_snapshot_firing_times", (0, 1)),
        perturbed_snapshot_firing_times=_stack_vector_rows(trace_rows, "perturbed_snapshot_firing_times", (0, 1)),
        readout_step=np.asarray([int(readout_step)], dtype=np.int64),
    )

    summary_metrics = summarize_mediation_results(df_results, conditions=conditions, swap_modes=swap_modes)
    summary_metrics["condition_order"] = list(conditions)
    summary_metrics["swap_mode_order"] = list(swap_modes)
    summary_metrics["case_triplet_ids"] = list(case_triplet_ids)
    summary_metrics_json = _save_json(summary_metrics, output_dir / "summary_metrics.json")

    figure_outputs: dict[str, str] = {}
    if not bool(args.skip_figures):
        fig1 = plot_mediation_flow_schematic()
        fig1_paths = save_figure_all_formats(fig1, figures_dir / "figure_1_mediation_flow")
        plt.close(fig1)
        fig2 = plot_summary_bars(df_results, conditions=conditions, swap_modes=swap_modes)
        fig2_paths = save_figure_all_formats(fig2, figures_dir / "figure_2_summary_bars")
        plt.close(fig2)
        fig3 = plot_triplet_scatter(df_results, conditions=FOCUS_CONDITIONS)
        fig3_paths = save_figure_all_formats(fig3, figures_dir / "figure_3_triplet_scatter")
        plt.close(fig3)
        figure_outputs = {
            "figure_1_png": fig1_paths["png"],
            "figure_2_png": fig2_paths["png"],
            "figure_3_png": fig3_paths["png"],
        }

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
            "batch_size": int(args.batch_size),
            "max_probes": int(args.max_probes),
            "samples_per_probe": int(args.samples_per_probe),
            "max_triplets": int(args.max_triplets),
            "save_case_count": int(args.save_case_count),
            "case_triplet_ids": list(case_triplet_ids),
            "swap_modes": list(swap_modes),
            "conditions": list(conditions),
            "skip_figures": bool(args.skip_figures),
            "skip_push": bool(args.skip_push),
            "skip_rescue": bool(args.skip_rescue),
            "readout_step": int(readout_step),
            "assumptions": summary_metrics["assumptions"],
            "outputs": {
                "triplet_specs_csv": str(Path(triplet_specs_csv).resolve()),
                "triplet_mediation_results_csv": str(Path(results_csv).resolve()),
                "triplet_mediation_vectors_npz": str(vectors_npz.resolve()),
                "triplet_mediation_traces_or_snapshots_npz": str(traces_npz.resolve()),
                "summary_metrics_json": str(summary_metrics_json.resolve()),
                **figure_outputs,
            },
        },
        result_root,
    )
    summary_path = save_summary_json(
        {
            "experiment": "distractor_l3_causal_mediation_experiment",
            "triplet_count": int(df_results["triplet_id"].nunique()),
            "result_rows": int(len(df_results)),
            "case_triplet_ids": list(case_triplet_ids),
            "artifacts": {
                "data_summary_metrics_json": str(summary_metrics_json.resolve()),
                "run_config_json": str(run_config_path.resolve()),
            },
        },
        result_root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=distractor_l3_causal_mediation_experiment",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"triplet_rows={len(df_results)}",
            f"case_triplets={len(case_triplet_ids)}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_path.resolve()}",
        ],
        logs_dir,
    )

    print("\n=== Distractor L3 Causal Mediation Experiment Summary ===")
    print(f"Triplets analysed: {int(df_results['triplet_id'].nunique()) if len(df_results) else 0}")
    print(f"Records: {int(len(df_results))}")
    if len(df_results):
        onset_and_trace = df_results[df_results["swap_mode"] == "onset_and_trace"].copy()
        if not onset_and_trace.empty:
            print(f"Mean push cosine (onset_and_trace): {float(onset_and_trace['push_cosine'].mean(skipna=True)):.4f}")
            print(f"Mean rescue ratio (onset_and_trace): {float(onset_and_trace['rescue_ratio'].mean(skipna=True)):.4f}")
    print(f"Saved: {results_csv}")
    print(f"Saved: {vectors_npz}")
    print(f"Saved: {traces_npz}")
    print(f"Saved: {summary_metrics_json}")
    print(f"Saved: {run_config_path}")


if __name__ == "__main__":
    main()
