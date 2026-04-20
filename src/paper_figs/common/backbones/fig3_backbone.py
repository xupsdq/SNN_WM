from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.voltage_readout import resolve_readout_step
from src.experiments.distractor.shared.pair_sampling import (
    PairExperimentSpec,
    build_pair_specs,
    extract_grouped_voltage_vector,
    prepare_pair_spike_batch,
)
from src.experiments.l3_accumulator_mechanism_experiment import (
    Layer3ReplaySnapshot,
    _center_vector,
    _nanargmax_with_default,
    _snapshot_layer3_for_replay,
    _vector_similarity,
    make_l3_region_masks,
    run_l3_deletion_analysis_for_pair,
    run_l3_replacement_analysis_for_pair,
    summarize_l3_mechanism_results,
)
from src.experiments.overlap_causal_input_perturbation_experiment import (
    OverlapMaskBundle,
    _initialize_final_records,
    _initialize_trace_records,
    apply_input_mask_to_spike_batch,
    build_overlap_masks_for_pair,
    build_summary_metrics,
    compute_final_pattern_similarity,
    compute_trace_pattern_similarity,
    finalize_final_arrays,
    finalize_trace_arrays,
    mix_seed,
)
from src.paper_figs.common.model_env import (
    DT,
    build_class_index,
    build_dataset_arrays,
    load_mnist_skeleton_dataset,
    load_paper_model_and_encoder,
)
from src.paper_figs.common.stats import sem

OVERLAP_CONDITIONS = ("sample_keep_overlap_only_dynamic", "sample_keep_nonoverlap_only_dynamic")
MECHANISM_CONDITIONS = ("full_dynamic", "full_static", *OVERLAP_CONDITIONS)


@dataclass(frozen=True)
class Fig3MechanismConfig:
    sample_ms: float = 200.0
    delay_ms: float = 500.0
    probe_ms: float = 100.0
    batch_size: int = 8
    max_probes: int = 20
    samples_per_probe: int = 12
    max_pairs: int = 240
    num_sim_bins: int = 5
    foreground_threshold: float = 0.0
    use_dilated_overlap: bool = False
    dilation_radius: int = 1
    num_control_candidates: int = 32
    l3_mask_mode: str = "1x1"


def build_fig3_mechanism_config(smoke: bool) -> Fig3MechanismConfig:
    if not bool(smoke):
        return Fig3MechanismConfig()
    return Fig3MechanismConfig(
        batch_size=8,
        max_probes=4,
        samples_per_probe=2,
        max_pairs=24,
    )


@dataclass(frozen=True)
class SharedMechanismCapture:
    grouped_voltage: np.ndarray
    probe_l1_trace: torch.Tensor
    probe_l2_trace: torch.Tensor
    probe_l3_trace: torch.Tensor
    readout_snapshot: torch.Tensor
    probe_s2p_trace: torch.Tensor
    probe_onset_snapshot: Layer3ReplaySnapshot
    prediction_probe: np.ndarray
    first_fire_t_probe: np.ndarray
    readout_step: int


@dataclass(frozen=True)
class Fig3MechanismResult:
    config: dict[str, Any]
    pair_metadata: pd.DataFrame
    panel_b_trace_summary: pd.DataFrame
    panel_b_pair_summary: pd.DataFrame
    panel_cd_pair_metrics: pd.DataFrame
    panel_c_reconstruction_summary: pd.DataFrame
    panel_d_direction_summary: pd.DataFrame
    panel_b_probe_trace_arrays: dict[str, np.ndarray]
    panel_cd_reconstruction_vectors: dict[str, np.ndarray]
    overlap_summary: dict[str, object]
    l3_summary: dict[str, object]
    stats: dict[str, Any]


def _stack_mask_batch(batch_masks: list[OverlapMaskBundle], attribute_name: str, device: torch.device) -> torch.Tensor:
    stacked = np.stack([np.asarray(getattr(mask_bundle, attribute_name), dtype=bool) for mask_bundle in batch_masks], axis=0)
    return torch.as_tensor(stacked, dtype=torch.bool, device=device)


def _slice_optional_tensor(tensor: torch.Tensor | None, batch_index: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[batch_index : batch_index + 1, ...].detach().cpu().clone()


def _slice_probe_onset_snapshot(snapshot: Layer3ReplaySnapshot, batch_index: int) -> Layer3ReplaySnapshot:
    return Layer3ReplaySnapshot(
        v_mem=snapshot.v_mem[batch_index : batch_index + 1, ...].detach().cpu().clone(),
        g_e=snapshot.g_e[batch_index : batch_index + 1, ...].detach().cpu().clone(),
        res=snapshot.res[batch_index : batch_index + 1, ...].detach().cpu().clone(),
        inh_trace=snapshot.inh_trace[batch_index : batch_index + 1, ...].detach().cpu().clone(),
        u_pre=_slice_optional_tensor(snapshot.u_pre, batch_index),
        x_pre=_slice_optional_tensor(snapshot.x_pre, batch_index),
        input_trace=_slice_optional_tensor(snapshot.input_trace, batch_index),
        eligibility_trace=_slice_optional_tensor(snapshot.eligibility_trace, batch_index),
        firing_times=_slice_optional_tensor(snapshot.firing_times, batch_index),
        input_shape=(1, int(snapshot.input_shape[1]), int(snapshot.input_shape[2]), int(snapshot.input_shape[3])),
        output_shape=(1, int(snapshot.output_shape[1]), int(snapshot.output_shape[2]), int(snapshot.output_shape[3])),
        readout_step=int(snapshot.readout_step),
    )


def _slice_capture(capture: SharedMechanismCapture, batch_index: int) -> SharedMechanismCapture:
    return SharedMechanismCapture(
        grouped_voltage=np.asarray(capture.grouped_voltage[batch_index : batch_index + 1, ...], dtype=np.float64),
        probe_l1_trace=capture.probe_l1_trace[:, batch_index : batch_index + 1, ...].detach().cpu().clone(),
        probe_l2_trace=capture.probe_l2_trace[:, batch_index : batch_index + 1, ...].detach().cpu().clone(),
        probe_l3_trace=capture.probe_l3_trace[:, batch_index : batch_index + 1, ...].detach().cpu().clone(),
        readout_snapshot=capture.readout_snapshot[batch_index : batch_index + 1, ...].detach().cpu().clone(),
        probe_s2p_trace=capture.probe_s2p_trace[batch_index : batch_index + 1, ...].detach().cpu().clone(),
        probe_onset_snapshot=_slice_probe_onset_snapshot(capture.probe_onset_snapshot, batch_index),
        prediction_probe=np.asarray(capture.prediction_probe[batch_index : batch_index + 1], dtype=np.int64),
        first_fire_t_probe=np.asarray(capture.first_fire_t_probe[batch_index : batch_index + 1], dtype=np.int64),
        readout_step=int(capture.readout_step),
    )


def _capture_shared_probe_rollout(
    *,
    net,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    delay_steps: int,
    stsp_mode: str,
    readout_step: int,
    sample_input_mask: torch.Tensor | None = None,
    phase_reset: bool = True,
) -> SharedMechanismCapture:
    if sample_spikes.ndim != 5 or probe_spikes.ndim != 5:
        raise ValueError("sample_spikes and probe_spikes must have shape [B, T, C, H, W]")
    batch_size, _, channels, height, width = sample_spikes.shape
    if int(probe_spikes.shape[0]) != int(batch_size):
        raise ValueError("sample_spikes and probe_spikes must share batch size")

    masked_sample_spikes = sample_spikes
    if sample_input_mask is not None:
        masked_sample_spikes = apply_input_mask_to_spike_batch(sample_spikes, sample_input_mask, mode="remove")

    prepare_network_state(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), dtype=sample_spikes.dtype, device=sample_spikes.device)
    current_time = 0
    probe_l1_frames: list[torch.Tensor] = []
    probe_l2_frames: list[torch.Tensor] = []
    probe_l3_frames: list[torch.Tensor] = []
    readout_snapshot = None

    def step_network(input_t: torch.Tensor, *, phase: str, phase_step: int, force_l3_time: int | None = None) -> None:
        nonlocal current_time, readout_snapshot
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())
        l3_time = current_time if force_l3_time is None else force_l3_time
        _, monitor_data = net.layer3.forward_step(
            s2_p,
            l3_time,
            training=False,
            monitor=bool(phase == "probe" and int(phase_step) == int(readout_step)),
            stsp_mode=stsp_mode,
        )
        if phase == "probe":
            probe_l1_frames.append(s1_p.detach().cpu().to(torch.float32))
            probe_l2_frames.append(s2.detach().cpu().to(torch.float32))
            probe_l3_frames.append(s2_p.detach().cpu().to(torch.float32))
        if phase == "probe" and int(phase_step) == int(readout_step):
            readout_snapshot = monitor_data["v_mem_snapshot"].detach().cpu().to(torch.float32)
        current_time += 1

    with torch.no_grad():
        for t_step in range(int(masked_sample_spikes.shape[1])):
            step_network(masked_sample_spikes[:, t_step, ...], phase="sample", phase_step=t_step)
        for _ in range(int(delay_steps)):
            step_network(zero_input, phase="delay", phase_step=0)

        net.layer3.reset_decision_state()
        if bool(phase_reset):
            net.layer3.v_mem.fill_(net.layer3.V_L)
            net.layer3.lateral_inh.reset_state(net.layer3.output_shape)
        probe_onset_snapshot = _snapshot_layer3_for_replay(net, readout_step=int(readout_step))

        for t_step in range(int(probe_spikes.shape[1])):
            l3_time = int(t_step) if bool(phase_reset) else int(current_time)
            step_network(probe_spikes[:, t_step, ...], phase="probe", phase_step=t_step, force_l3_time=l3_time)

    if readout_snapshot is None:
        raise RuntimeError("Probe readout snapshot was not captured.")

    flat_times = net.layer3.firing_times.detach().cpu()
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    prediction_probe = (min_indices // net.layer3.neurons_per_class).long()
    prediction_probe[~has_fired] = -1
    first_fire_t_probe = min_times.clone()
    first_fire_t_probe[~has_fired] = -1

    probe_l1_trace = torch.stack(probe_l1_frames, dim=0)
    probe_l2_trace = torch.stack(probe_l2_frames, dim=0)
    probe_l3_trace = torch.stack(probe_l3_frames, dim=0)
    probe_s2p_trace = torch.stack(probe_l3_frames, dim=1)
    return SharedMechanismCapture(
        grouped_voltage=extract_grouped_voltage_vector(net, readout_snapshot),
        probe_l1_trace=probe_l1_trace,
        probe_l2_trace=probe_l2_trace,
        probe_l3_trace=probe_l3_trace,
        readout_snapshot=readout_snapshot,
        probe_s2p_trace=probe_s2p_trace,
        probe_onset_snapshot=probe_onset_snapshot,
        prediction_probe=prediction_probe.numpy().astype(np.int64, copy=False),
        first_fire_t_probe=first_fire_t_probe.numpy().astype(np.int64, copy=False),
        readout_step=int(readout_step),
    )


def _build_trace_summary(trace_arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    time_steps = int(trace_arrays["DPI_L3"].shape[1]) if np.asarray(trace_arrays["DPI_L3"]).ndim == 2 else 0
    condition_names = np.asarray(trace_arrays["condition_name"])
    for condition_name in OVERLAP_CONDITIONS:
        selector = condition_names == str(condition_name)
        dpi = np.asarray(trace_arrays["DPI_L3"], dtype=np.float64)[selector]
        s_dyn = np.asarray(trace_arrays["S_dyn_L3"], dtype=np.float64)[selector]
        s_sta = np.asarray(trace_arrays["S_sta_L3"], dtype=np.float64)[selector]
        for t_step in range(time_steps):
            rows.append(
                {
                    "condition": str(condition_name),
                    "time_step": int(t_step),
                    "time_ms": float(t_step),
                    "dpi_mean": float(np.mean(dpi[:, t_step])) if len(dpi) else float("nan"),
                    "dpi_sem": float(sem(dpi[:, t_step])) if len(dpi) else 0.0,
                    "s_dyn_mean": float(np.mean(s_dyn[:, t_step])) if len(s_dyn) else float("nan"),
                    "s_sta_mean": float(np.mean(s_sta[:, t_step])) if len(s_sta) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _build_pair_summary(trace_arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    condition_names = np.asarray(trace_arrays["condition_name"])
    for condition_name in OVERLAP_CONDITIONS:
        selector = condition_names == str(condition_name)
        pair_ids = np.asarray(trace_arrays["pair_id"])[selector]
        mean_dpi = np.asarray(trace_arrays["mean_DPI_L3"], dtype=np.float64)[selector]
        mean_s_dyn = np.asarray(trace_arrays["mean_S_dyn_L3"], dtype=np.float64)[selector]
        mean_s_sta = np.asarray(trace_arrays["mean_S_sta_L3"], dtype=np.float64)[selector]
        peak_dpi = (
            np.asarray(trace_arrays["DPI_L3"], dtype=np.float64)[selector].max(axis=1)
            if selector.any()
            else np.zeros((0,), dtype=np.float64)
        )
        for pair_id, m_dpi, p_dpi, m_dyn, m_sta in zip(pair_ids, mean_dpi, peak_dpi, mean_s_dyn, mean_s_sta):
            rows.append(
                {
                    "pair_id": int(pair_id),
                    "condition": str(condition_name),
                    "mean_DPI_L3": float(m_dpi),
                    "peak_DPI_L3": float(p_dpi),
                    "mean_S_dyn_L3": float(m_dyn),
                    "mean_S_sta_L3": float(m_sta),
                }
            )
    return pd.DataFrame(rows)


def run_fig3_mechanism_backbone(
    *,
    model_path: str,
    dataset_root: str,
    device: torch.device,
    seed: int,
    smoke: bool,
    logger=None,
) -> Fig3MechanismResult:
    config = build_fig3_mechanism_config(bool(smoke))
    dataset = load_mnist_skeleton_dataset(dataset_root, split="test")
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    class_index = build_class_index(dataset, num_classes=int(len(np.unique(labels))))
    pair_spec = PairExperimentSpec(dt=DT, sample_ms=float(config.sample_ms), probe_ms=float(config.probe_ms))
    net, encoder = load_paper_model_and_encoder(
        model_path=model_path,
        device=device,
        max_duration_ms=max(float(config.sample_ms), float(config.delay_ms), float(config.probe_ms)),
    )
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(pair_spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )
    delay_steps = int(round((float(config.delay_ms) * ms) / DT))

    df_pairs = build_pair_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=int(config.max_probes),
        samples_per_probe=int(config.samples_per_probe),
        num_bins=int(config.num_sim_bins),
        max_pairs=int(config.max_pairs),
        seed=int(seed),
    )

    mask_bank: dict[int, OverlapMaskBundle] = {}
    for pair_row in df_pairs.itertuples(index=False):
        pair_id = int(pair_row.pair_id)
        mask_bank[pair_id] = build_overlap_masks_for_pair(
            sample_image=images[int(pair_row.sample_id)],
            probe_image=images[int(pair_row.probe_id)],
            foreground_threshold=float(config.foreground_threshold),
            use_dilated_overlap=bool(config.use_dilated_overlap),
            dilation_radius=int(config.dilation_radius),
            seed=mix_seed(int(seed), pair_id, int(pair_row.sample_id), int(pair_row.probe_id)),
            num_control_candidates=int(config.num_control_candidates),
        )

    overlap_rows: list[dict[str, object]] = []
    trace_records = _initialize_trace_records()
    final_records = _initialize_final_records()
    l3_pair_rows: list[dict[str, object]] = []
    v_dyn_rows: list[np.ndarray] = []
    v_sta_rows: list[np.ndarray] = []
    delta_v_rows: list[np.ndarray] = []
    delta_hat_plus_rows: list[np.ndarray] = []
    delta_hat_minus_rows: list[np.ndarray] = []

    regions = None
    total_batches = max(1, int(math.ceil(len(df_pairs) / max(int(config.batch_size), 1))))
    shared_baseline_rollouts = 0
    shared_condition_rollouts = 0
    l3_replay_pairs = 0

    for batch_index, batch_start in enumerate(range(0, len(df_pairs), int(config.batch_size)), start=1):
        batch_df = df_pairs.iloc[batch_start : batch_start + int(config.batch_size)].copy().reset_index(drop=True)
        if batch_df.empty:
            continue
        batch_pair_ids = batch_df["pair_id"].astype(int).tolist()
        batch_masks = [mask_bank[pair_id] for pair_id in batch_pair_ids]
        sample_spikes, probe_spikes = prepare_pair_spike_batch(
            images=images,
            batch_df=batch_df,
            encoder=encoder,
            spec=pair_spec,
            device=device,
        )
        overlap_only_mask = _stack_mask_batch(batch_masks, "sample_nonoverlap_mask", device)
        nonoverlap_only_mask = _stack_mask_batch(batch_masks, "sample_overlap_mask", device)

        captures = {
            "full_dynamic": _capture_shared_probe_rollout(
                net=net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                delay_steps=delay_steps,
                stsp_mode="dynamic",
                readout_step=readout_step,
            ),
            "full_static": _capture_shared_probe_rollout(
                net=net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                delay_steps=delay_steps,
                stsp_mode="static_frozen",
                readout_step=readout_step,
            ),
            "sample_keep_overlap_only_dynamic": _capture_shared_probe_rollout(
                net=net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                delay_steps=delay_steps,
                stsp_mode="dynamic",
                readout_step=readout_step,
                sample_input_mask=overlap_only_mask,
            ),
            "sample_keep_nonoverlap_only_dynamic": _capture_shared_probe_rollout(
                net=net,
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                delay_steps=delay_steps,
                stsp_mode="dynamic",
                readout_step=readout_step,
                sample_input_mask=nonoverlap_only_mask,
            ),
        }
        shared_baseline_rollouts += 2
        shared_condition_rollouts += len(MECHANISM_CONDITIONS)

        full_dynamic = captures["full_dynamic"]
        full_static = captures["full_static"]
        if regions is None:
            trace_height = int(full_dynamic.probe_s2p_trace.shape[-2])
            trace_width = int(full_dynamic.probe_s2p_trace.shape[-1])
            regions = make_l3_region_masks(trace_height, trace_width, mask_mode=str(config.l3_mask_mode))

        for batch_idx, pair_row in enumerate(batch_df.itertuples(index=False)):
            pair_id = int(pair_row.pair_id)
            mask_bundle = mask_bank[pair_id]
            ref_l1_dyn = full_dynamic.probe_l1_trace[:, batch_idx].numpy()
            ref_l2_dyn = full_dynamic.probe_l2_trace[:, batch_idx].numpy()
            ref_l3_dyn = full_dynamic.probe_l3_trace[:, batch_idx].numpy()
            ref_l1_sta = full_static.probe_l1_trace[:, batch_idx].numpy()
            ref_l2_sta = full_static.probe_l2_trace[:, batch_idx].numpy()
            ref_l3_sta = full_static.probe_l3_trace[:, batch_idx].numpy()
            v_full_dyn = np.asarray(full_dynamic.grouped_voltage[batch_idx], dtype=np.float64)
            v_full_sta = np.asarray(full_static.grouped_voltage[batch_idx], dtype=np.float64)

            pair_condition_metrics: dict[str, dict[str, object]] = {}
            for condition_name in MECHANISM_CONDITIONS:
                rollout = captures[condition_name]
                cond_l1 = rollout.probe_l1_trace[:, batch_idx].numpy()
                cond_l2 = rollout.probe_l2_trace[:, batch_idx].numpy()
                cond_l3 = rollout.probe_l3_trace[:, batch_idx].numpy()
                s_dyn_l1, s_sta_l1, dpi_l1 = compute_trace_pattern_similarity(cond_l1, ref_l1_dyn, ref_l1_sta)
                s_dyn_l2, s_sta_l2, dpi_l2 = compute_trace_pattern_similarity(cond_l2, ref_l2_dyn, ref_l2_sta)
                s_dyn_l3, s_sta_l3, dpi_l3 = compute_trace_pattern_similarity(cond_l3, ref_l3_dyn, ref_l3_sta)
                v_cond = np.asarray(rollout.grouped_voltage[batch_idx], dtype=np.float64)
                s_dyn_final, s_sta_final, dpi_final = compute_final_pattern_similarity(v_cond, v_full_dyn, v_full_sta)
                pair_condition_metrics[condition_name] = {
                    "rollout": rollout,
                    "predicted_label": int(rollout.prediction_probe[batch_idx]),
                    "first_fire_t_probe": int(rollout.first_fire_t_probe[batch_idx]),
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

            for condition_name in MECHANISM_CONDITIONS:
                condition_metric = pair_condition_metrics[condition_name]
                rollout = condition_metric["rollout"]
                record_id = len(overlap_rows)
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
                    layer_name: (mean_s_dyn[layer_name] / ref_mean_s_dyn[layer_name]) if abs(ref_mean_s_dyn[layer_name]) > 1e-8 else float("nan")
                    for layer_name in ("L1", "L2", "L3", "final")
                }
                overlap_rows.append(
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
                trace_records["record_id"].append(int(record_id))
                trace_records["pair_id"].append(pair_id)
                trace_records["condition_name"].append(str(condition_name))
                trace_records["S_dyn_L1"].append(np.asarray(condition_metric["S_dyn"]["L1"], dtype=np.float32))
                trace_records["S_sta_L1"].append(np.asarray(condition_metric["S_sta"]["L1"], dtype=np.float32))
                trace_records["DPI_L1"].append(np.asarray(condition_metric["S_dyn"]["L1"] - condition_metric["S_sta"]["L1"], dtype=np.float32))
                trace_records["S_dyn_L2"].append(np.asarray(condition_metric["S_dyn"]["L2"], dtype=np.float32))
                trace_records["S_sta_L2"].append(np.asarray(condition_metric["S_sta"]["L2"], dtype=np.float32))
                trace_records["DPI_L2"].append(np.asarray(condition_metric["S_dyn"]["L2"] - condition_metric["S_sta"]["L2"], dtype=np.float32))
                trace_records["S_dyn_L3"].append(np.asarray(condition_metric["S_dyn"]["L3"], dtype=np.float32))
                trace_records["S_sta_L3"].append(np.asarray(condition_metric["S_sta"]["L3"], dtype=np.float32))
                trace_records["DPI_L3"].append(np.asarray(condition_metric["S_dyn"]["L3"] - condition_metric["S_sta"]["L3"], dtype=np.float32))
                trace_records["mean_S_dyn_L3"].append(np.float32(mean_s_dyn["L3"]))
                trace_records["mean_S_sta_L3"].append(np.float32(mean_s_sta["L3"]))
                trace_records["mean_DPI_L3"].append(np.float32(condition_metric["DPI"]["L3"]))

                final_records["record_id"].append(int(record_id))
                final_records["pair_id"].append(pair_id)
                final_records["condition_name"].append(str(condition_name))
                final_records["V_cond"].append(np.asarray(condition_metric["v_cond"], dtype=np.float32))
                final_records["V_full_dyn"].append(v_full_dyn.astype(np.float32, copy=False))
                final_records["V_full_sta"].append(v_full_sta.astype(np.float32, copy=False))
                final_records["S_dyn_final"].append(float(condition_metric["S_dyn"]["final"]))
                final_records["S_sta_final"].append(float(condition_metric["S_sta"]["final"]))
                final_records["DPI_final"].append(float(condition_metric["DPI"]["final"]))
                final_records["Retain_dyn_final"].append(float(retain_dyn["final"]))
                final_records["Pull_sta_final"].append(float(mean_s_sta["final"]))

            dynamic_single = _slice_capture(full_dynamic, batch_idx)
            static_single = _slice_capture(full_static, batch_idx)
            deletion = run_l3_deletion_analysis_for_pair(
                net=net,
                dynamic_capture=dynamic_single,
                static_capture=static_single,
                regions=regions,
                batch_size=int(config.batch_size),
            )
            replacement = run_l3_replacement_analysis_for_pair(
                net=net,
                dynamic_capture=dynamic_single,
                static_capture=static_single,
                regions=regions,
                batch_size=int(config.batch_size),
            )
            l3_replay_pairs += 1

            v_dyn = np.asarray(dynamic_single.grouped_voltage[0], dtype=np.float64)
            v_sta = np.asarray(static_single.grouped_voltage[0], dtype=np.float64)
            delta_v = _center_vector(v_dyn) - _center_vector(v_sta)
            bias_magnitude = float(np.linalg.norm(delta_v, ord=2))
            bias_direction = int(np.argmax(delta_v))

            delta_hat_plus = np.nansum(np.asarray(replacement["R_plus_tilde"], dtype=np.float64), axis=0)
            delta_hat_minus = np.nansum(np.asarray(replacement["R_minus_tilde"], dtype=np.float64), axis=0)
            sim_plus = _vector_similarity(delta_hat_plus, delta_v)
            sim_minus = _vector_similarity(delta_hat_minus, delta_v)

            e_sta_k = np.asarray(deletion["E_sta"][:, bias_direction], dtype=np.float64)
            e_dyn_k = np.asarray(deletion["E_dyn"][:, bias_direction], dtype=np.float64)
            r_plus_k = np.asarray(replacement["R_plus_tilde"][:, bias_direction], dtype=np.float64)
            r_minus_k = np.asarray(replacement["R_minus_tilde"][:, bias_direction], dtype=np.float64)

            l3_pair_rows.append(
                {
                    "pair_id": int(pair_row.pair_id),
                    "probe_id": int(pair_row.probe_id),
                    "sample_id": int(pair_row.sample_id),
                    "probe_label": int(pair_row.probe_label),
                    "sample_label": int(pair_row.sample_label),
                    "similarity_public_or_initial": float(pair_row.similarity_public_or_initial),
                    "similarity_bin": str(pair_row.similarity_bin),
                    "similarity_bin_index": int(pair_row.similarity_bin_index),
                    "bias_magnitude": bias_magnitude,
                    "bias_direction": int(bias_direction),
                    "pred_dynamic": int(dynamic_single.prediction_probe[0]),
                    "pred_static": int(static_single.prediction_probe[0]),
                    "first_fire_t_dynamic": int(dynamic_single.first_fire_t_probe[0]),
                    "first_fire_t_static": int(static_single.first_fire_t_probe[0]),
                    "top_dynamic_deletion_region_for_k_star": _nanargmax_with_default(e_dyn_k),
                    "top_static_deletion_region_for_k_star": _nanargmax_with_default(e_sta_k),
                    "deletion_dynamic_minus_static_kstar": float(np.nanmean(e_dyn_k - e_sta_k)),
                    "top_static_to_dynamic_push_region_for_k_star": _nanargmax_with_default(r_plus_k),
                    "top_dynamic_to_static_pullback_region_for_k_star": _nanargmax_with_default(r_minus_k),
                    "top_push_value_kstar": float(np.nanmax(r_plus_k)) if np.isfinite(r_plus_k).any() else None,
                    "top_pullback_value_kstar": float(np.nanmax(r_minus_k)) if np.isfinite(r_minus_k).any() else None,
                    "replacement_push_kstar": float(np.nanmean(r_plus_k)),
                    "replacement_pullback_kstar": float(np.nanmean(r_minus_k)),
                    "reconstruction_cosine_plus": float(sim_plus["cosine"]),
                    "reconstruction_pearson_plus": float(sim_plus["pearson"]),
                    "reconstruction_spearman_plus": float(sim_plus["spearman"]),
                    "reconstruction_cosine_minus": float(sim_minus["cosine"]),
                    "reconstruction_pearson_minus": float(sim_minus["pearson"]),
                    "reconstruction_spearman_minus": float(sim_minus["spearman"]),
                    "direction_match_plus": int(np.argmax(delta_hat_plus) == bias_direction) if np.isfinite(delta_hat_plus).any() else 0,
                    "direction_match_minus": int(np.argmax(delta_hat_minus) == bias_direction) if np.isfinite(delta_hat_minus).any() else 0,
                    "readout_step": int(readout_step),
                }
            )
            v_dyn_rows.append(v_dyn)
            v_sta_rows.append(v_sta)
            delta_v_rows.append(delta_v)
            delta_hat_plus_rows.append(delta_hat_plus)
            delta_hat_minus_rows.append(delta_hat_minus)

        if logger is not None:
            logger.info("[Backbone] batch=%s/%s pairs=%s shared_rollouts=%s", batch_index, total_batches, len(batch_df), len(MECHANISM_CONDITIONS))

    overlap_df = pd.DataFrame(overlap_rows).sort_values(["record_id"], kind="stable").reset_index(drop=True)
    overlap_trace_arrays = finalize_trace_arrays(trace_records, pair_spec.probe_steps)
    _ = finalize_final_arrays(final_records, int(len(np.unique(labels))))
    overlap_summary = build_summary_metrics(overlap_df)

    l3_df = pd.DataFrame(l3_pair_rows).sort_values(["pair_id"], kind="stable").reset_index(drop=True)
    l3_summary = summarize_l3_mechanism_results(l3_df)

    panel_b_trace_summary = _build_trace_summary(overlap_trace_arrays)
    panel_b_pair_summary = _build_pair_summary(overlap_trace_arrays)
    panel_cd_pair_metrics = l3_df[
        [
            "pair_id",
            "bias_magnitude",
            "bias_direction",
            "reconstruction_cosine_plus",
            "reconstruction_cosine_minus",
            "direction_match_plus",
            "direction_match_minus",
        ]
    ].copy()
    panel_c_reconstruction_summary = pd.DataFrame(
        [
            {
                "mode": "plus",
                "mean_cosine": float(l3_summary["overall"]["mean_reconstruction_cosine_plus"]),
                "sem_cosine": float(panel_cd_pair_metrics["reconstruction_cosine_plus"].sem(ddof=1) if len(panel_cd_pair_metrics) > 1 else 0.0),
            },
            {
                "mode": "minus",
                "mean_cosine": float(l3_summary["overall"]["mean_reconstruction_cosine_minus"]),
                "sem_cosine": float(panel_cd_pair_metrics["reconstruction_cosine_minus"].sem(ddof=1) if len(panel_cd_pair_metrics) > 1 else 0.0),
            },
        ]
    )
    panel_d_direction_summary = pd.DataFrame(
        [
            {"mode": "plus", "direction_match_rate": float(l3_summary["overall"]["direction_match_rate_plus"])},
            {"mode": "minus", "direction_match_rate": float(l3_summary["overall"]["direction_match_rate_minus"])},
        ]
    )
    panel_cd_reconstruction_vectors = {
        "pair_id": l3_df["pair_id"].to_numpy(dtype=np.int64, copy=False),
        "v_dyn": np.stack(v_dyn_rows, axis=0) if v_dyn_rows else np.zeros((0, 0), dtype=np.float64),
        "v_sta": np.stack(v_sta_rows, axis=0) if v_sta_rows else np.zeros((0, 0), dtype=np.float64),
        "delta_v": np.stack(delta_v_rows, axis=0) if delta_v_rows else np.zeros((0, 0), dtype=np.float64),
        "delta_hat_plus": np.stack(delta_hat_plus_rows, axis=0) if delta_hat_plus_rows else np.zeros((0, 0), dtype=np.float64),
        "delta_hat_minus": np.stack(delta_hat_minus_rows, axis=0) if delta_hat_minus_rows else np.zeros((0, 0), dtype=np.float64),
    }

    return Fig3MechanismResult(
        config={
            "sample_ms": float(config.sample_ms),
            "delay_ms": float(config.delay_ms),
            "probe_ms": float(config.probe_ms),
            "batch_size": int(config.batch_size),
            "max_probes": int(config.max_probes),
            "samples_per_probe": int(config.samples_per_probe),
            "max_pairs": int(config.max_pairs),
            "num_sim_bins": int(config.num_sim_bins),
            "foreground_threshold": float(config.foreground_threshold),
            "use_dilated_overlap": bool(config.use_dilated_overlap),
            "dilation_radius": int(config.dilation_radius),
            "num_control_candidates": int(config.num_control_candidates),
            "l3_mask_mode": str(config.l3_mask_mode),
        },
        pair_metadata=df_pairs.copy(),
        panel_b_trace_summary=panel_b_trace_summary,
        panel_b_pair_summary=panel_b_pair_summary,
        panel_cd_pair_metrics=panel_cd_pair_metrics,
        panel_c_reconstruction_summary=panel_c_reconstruction_summary,
        panel_d_direction_summary=panel_d_direction_summary,
        panel_b_probe_trace_arrays=overlap_trace_arrays,
        panel_cd_reconstruction_vectors=panel_cd_reconstruction_vectors,
        overlap_summary=overlap_summary,
        l3_summary=l3_summary,
        stats={
            "pair_strategy": "A_separate__BCD_shared",
            "shared_pair_count": int(len(df_pairs)),
            "shared_batch_count": int(total_batches),
            "shared_baseline_rollouts": int(shared_baseline_rollouts),
            "shared_condition_rollouts": int(shared_condition_rollouts),
            "l3_replay_pairs": int(l3_replay_pairs),
            "avoided_stage_modules": [
                "src.experiments.overlap_causal_input_perturbation_experiment",
                "src.experiments.l3_accumulator_mechanism_experiment",
            ],
        },
    )


__all__ = [
    "Fig3MechanismResult",
    "build_fig3_mechanism_config",
    "run_fig3_mechanism_backbone",
]
