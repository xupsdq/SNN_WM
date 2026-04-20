from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.experiments.common.voltage_readout import resolve_readout_step
from src.experiments.distractor_chunk_holistic_invocation_experiment import (
    CUE_CONDITIONS,
    ExperimentSpec as ChunkExperimentSpec,
    build_probe_region_bundle as build_chunk_probe_region_bundle,
    compute_holistic_metrics,
    compute_region_support_summary,
    compute_reshaping_metrics,
    compute_winner_metrics,
    prepare_triplet_batches,
    run_distractor_rollout_capture,
    _mask_spike_batch_keep_region,
)
from src.experiments.distractor_region_ux_mechanism_experiment import (
    BASELINE_CONDITION,
    MAIN_REGION_ORDER,
    REFERENCE_DISTRACTOR_CONDITION,
    REFERENCE_SAMPLE_CONDITION,
    build_layer1_composition_summary,
    build_layer1_composition_trial_rows,
    build_layer1_formula_fit_summary,
    build_probe_region_bundle as build_region_probe_region_bundle,
    simulate_layer1_input_boundary_stsp,
)
from src.paper_figs.common.model_env import (
    DT,
    build_class_index,
    build_dataset_arrays,
    load_mnist_skeleton_dataset,
    load_paper_model_and_encoder,
)


@dataclass(frozen=True)
class Fig5BackboneConfig:
    sample_ms: float = 200.0
    delay1_ms: float = 400.0
    distractor_ms: float = 200.0
    delay2_ms: float = 400.0
    probe_ms: float = 100.0
    batch_size: int = 16
    max_probes: int = 20
    samples_per_probe: int = 12
    max_triplets: int = 240
    num_sim_bins: int = 4
    foreground_threshold: float = 0.0
    dilation_radius: int = 1
    winner_window_frac: float = 0.5
    tie_threshold: float = 0.02


def build_fig5_backbone_config(smoke: bool) -> Fig5BackboneConfig:
    if not bool(smoke):
        return Fig5BackboneConfig()
    return Fig5BackboneConfig(
        batch_size=2,
        max_probes=2,
        samples_per_probe=1,
        max_triplets=4,
    )


@dataclass(frozen=True)
class Fig5BackboneResult:
    config: dict[str, Any]
    triplets: pd.DataFrame
    region_support_condition: pd.DataFrame
    layer1_trial_metrics: pd.DataFrame
    layer1_formula_fit: pd.DataFrame
    reshaping_metrics: pd.DataFrame
    holistic_metrics: pd.DataFrame
    cue_winner_metrics: pd.DataFrame
    stats: dict[str, Any]


def _np_state_to_torch(state: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "u": torch.as_tensor(np.asarray(state["u"]), dtype=torch.float32),
        "x": torch.as_tensor(np.asarray(state["x"]), dtype=torch.float32),
        "ux": torch.as_tensor(np.asarray(state["ux"]), dtype=torch.float32),
    }


def run_fig5_backbone(
    *,
    model_path: str,
    dataset_root: str,
    device: torch.device,
    seed: int,
    smoke: bool,
    logger=None,
) -> Fig5BackboneResult:
    config = build_fig5_backbone_config(bool(smoke))
    dataset = load_mnist_skeleton_dataset(dataset_root, split="test")
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    class_index = build_class_index(dataset, num_classes=int(len(np.unique(labels))))
    spec = ChunkExperimentSpec(
        dt=DT,
        sample_ms=float(config.sample_ms),
        delay1_ms=float(config.delay1_ms),
        distractor_ms=float(config.distractor_ms),
        delay2_ms=float(config.delay2_ms),
        probe_ms=float(config.probe_ms),
    )
    net, encoder = load_paper_model_and_encoder(
        model_path=model_path,
        device=device,
        max_duration_ms=max(float(config.sample_ms), float(config.distractor_ms), float(config.probe_ms)),
    )
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )

    from src.experiments.common.distractor_triplets import build_triplet_specs

    df_triplets = build_triplet_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=int(config.max_probes),
        samples_per_probe=int(config.samples_per_probe),
        num_bins=int(config.num_sim_bins),
        max_triplets=int(config.max_triplets),
        seed=int(seed),
    )

    region_bundle_by_triplet: dict[int, Any] = {}
    chunk_bundle_by_triplet: dict[int, Any] = {}
    triplet_rows: list[dict[str, object]] = []
    for row in df_triplets.itertuples(index=False):
        triplet_id = int(row.triplet_id)
        region_bundle = build_region_probe_region_bundle(
            net=net,
            sample_image=images[int(row.sample_id)],
            distractor_image=images[int(row.distractor_id)],
            probe_image=images[int(row.probe_id)],
            foreground_threshold=float(config.foreground_threshold),
            dilation_radius=int(config.dilation_radius),
        )
        chunk_bundle = build_chunk_probe_region_bundle(
            images[int(row.sample_id)],
            images[int(row.distractor_id)],
            images[int(row.probe_id)],
            net=net,
            foreground_threshold=float(config.foreground_threshold),
            dilation_radius=int(config.dilation_radius),
        )
        region_bundle_by_triplet[triplet_id] = region_bundle
        chunk_bundle_by_triplet[triplet_id] = chunk_bundle
        meta = dict(row._asdict())
        meta.update(
            {
                "sample_only_area": int(region_bundle.metadata["sample_only_area"]),
                "distractor_only_area": int(region_bundle.metadata["distractor_only_area"]),
                "shared_area": int(region_bundle.metadata["shared_area"]),
            }
        )
        triplet_rows.append(meta)
    triplets_aug = pd.DataFrame(triplet_rows).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)

    layer1_baseline_u = float(net.layer1.stsp_U)
    region_support_rows: list[dict[str, object]] = []
    layer1_rows: list[dict[str, object]] = []
    reshaping_tables: list[pd.DataFrame] = []
    holistic_tables: list[pd.DataFrame] = []
    cue_winner_tables: list[pd.DataFrame] = []
    rollout_count = 0

    total_batches = max(1, int(np.ceil(len(triplets_aug) / max(int(config.batch_size), 1))))
    for batch_index, batch_start in enumerate(range(0, len(triplets_aug), int(config.batch_size)), start=1):
        batch_df = triplets_aug.iloc[batch_start : batch_start + int(config.batch_size)].copy().reset_index(drop=True)
        if batch_df.empty:
            continue
        triplet_ids = batch_df["triplet_id"].astype(int).tolist()
        region_bundles = [region_bundle_by_triplet[triplet_id] for triplet_id in triplet_ids]
        chunk_bundles = [chunk_bundle_by_triplet[triplet_id] for triplet_id in triplet_ids]
        batches = prepare_triplet_batches(images, batch_df, encoder=encoder, spec=spec, device=device)
        probe_full = batches["probe"]
        cue_batches = {
            "cue_SP": _mask_spike_batch_keep_region(probe_full, [bundle.probe_region_masks["SP"] for bundle in chunk_bundles]),
            "cue_DP": _mask_spike_batch_keep_region(probe_full, [bundle.probe_region_masks["DP"] for bundle in chunk_bundles]),
            "cue_SDP": _mask_spike_batch_keep_region(probe_full, [bundle.probe_region_masks["SDP"] for bundle in chunk_bundles]),
        }
        captures = {
            BASELINE_CONDITION: run_distractor_rollout_capture(
                net,
                sample_spikes=batches["sample"],
                distractor_spikes=batches["distractor"],
                probe_spikes=probe_full,
                spec=spec,
                readout_step=readout_step,
            ),
            REFERENCE_SAMPLE_CONDITION: run_distractor_rollout_capture(
                net,
                sample_spikes=batches["sample"],
                distractor_spikes=batches["zero_distractor"],
                probe_spikes=probe_full,
                spec=spec,
                readout_step=readout_step,
            ),
            REFERENCE_DISTRACTOR_CONDITION: run_distractor_rollout_capture(
                net,
                sample_spikes=batches["distractor_as_sample"],
                distractor_spikes=batches["zero_distractor"],
                probe_spikes=probe_full,
                spec=spec,
                readout_step=readout_step,
            ),
            "distractor_only_trajectory_reference": run_distractor_rollout_capture(
                net,
                sample_spikes=batches["zero_sample"],
                distractor_spikes=batches["distractor"],
                probe_spikes=batches["zero_probe"],
                spec=spec,
                readout_step=readout_step,
            ),
            "sample_only_trajectory_reference": run_distractor_rollout_capture(
                net,
                sample_spikes=batches["sample"],
                distractor_spikes=batches["zero_distractor"],
                probe_spikes=batches["zero_probe"],
                spec=spec,
                readout_step=readout_step,
            ),
        }
        rollout_count += len(captures)
        for cue_name, cue_probe in cue_batches.items():
            captures[cue_name] = run_distractor_rollout_capture(
                net,
                sample_spikes=batches["sample"],
                distractor_spikes=batches["distractor"],
                probe_spikes=cue_probe,
                spec=spec,
                readout_step=readout_step,
            )
            rollout_count += 1

        for condition_name in (
            BASELINE_CONDITION,
            REFERENCE_SAMPLE_CONDITION,
            REFERENCE_DISTRACTOR_CONDITION,
            *CUE_CONDITIONS,
        ):
            capture = captures[condition_name]
            for local_idx, triplet_id in enumerate(triplet_ids):
                single_state = {
                    layer_key: {
                        key: (
                            np.asarray(value[local_idx : local_idx + 1, ...], dtype=np.float32)
                            if np.asarray(value).ndim > 0
                            else np.asarray(value, dtype=np.float32)
                        )
                        for key, value in capture.preprobe_states[layer_key].items()
                    }
                    for layer_key in ("layer1", "layer2", "layer3")
                }
                region_support_rows.extend(
                    compute_region_support_summary(
                        triplet_id=int(triplet_id),
                        condition=str(condition_name),
                        preprobe_states=single_state,
                        bundle=chunk_bundles[local_idx],
                    )
                )

        predicted_layer1_preprobe = simulate_layer1_input_boundary_stsp(
            layer=net.layer1,
            sample_spikes=batches["sample"],
            distractor_spikes=batches["distractor"],
            delay1_steps=int(spec.delay1_steps),
            delay2_steps=int(spec.delay2_steps),
        )
        observed_layer1_state = _np_state_to_torch(captures[BASELINE_CONDITION].preprobe_states["layer1"])
        layer1_rows.extend(
            build_layer1_composition_trial_rows(
                batch_df=batch_df,
                bundles=region_bundles,
                observed_state=observed_layer1_state,
                predicted_state=predicted_layer1_preprobe,
                baseline_ux=layer1_baseline_u,
            )
        )

        reshaping_tables.append(
            compute_reshaping_metrics(
                triplet_ids=triplet_ids,
                mixed_capture=captures[BASELINE_CONDITION],
                distractor_only_capture=captures["distractor_only_trajectory_reference"],
                sample_only_capture=captures["sample_only_trajectory_reference"],
            )
        )

        batch_winners: dict[str, pd.DataFrame] = {}
        for condition_name in (BASELINE_CONDITION, *CUE_CONDITIONS):
            winner_df = compute_winner_metrics(
                condition_name=condition_name,
                triplet_ids=triplet_ids,
                condition_trace=captures[condition_name].probe_grouped_voltage_trace,
                sample_reference_trace=captures[REFERENCE_SAMPLE_CONDITION].probe_grouped_voltage_trace,
                distractor_reference_trace=captures[REFERENCE_DISTRACTOR_CONDITION].probe_grouped_voltage_trace,
                winner_window_frac=float(config.winner_window_frac),
                tie_threshold=float(config.tie_threshold),
                predictions=captures[condition_name].prediction_probe,
                first_fire=captures[condition_name].first_fire_t_probe,
            )
            batch_winners[condition_name] = winner_df
            if condition_name in CUE_CONDITIONS:
                cue_winner_tables.append(winner_df)

        holistic_tables.append(
            compute_holistic_metrics(
                triplet_ids=triplet_ids,
                full_capture=captures[BASELINE_CONDITION],
                distractor_reference_capture=captures[REFERENCE_DISTRACTOR_CONDITION],
                cue_captures={cue_name: captures[cue_name] for cue_name in CUE_CONDITIONS},
                cue_winners={cue_name: batch_winners[cue_name] for cue_name in CUE_CONDITIONS},
                winner_window_frac=float(config.winner_window_frac),
            )
        )
        if logger is not None:
            logger.info("[Backbone] batch=%s/%s triplets=%s shared_rollouts=%s", batch_index, total_batches, len(batch_df), len(captures))

    layer1_trial_metrics = pd.DataFrame(layer1_rows).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)
    region_support_condition = pd.DataFrame(region_support_rows).sort_values(
        ["triplet_id", "condition", "layer", "region"],
        kind="stable",
    ).reset_index(drop=True)
    reshaping_metrics = pd.concat(reshaping_tables, axis=0, ignore_index=True).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)
    holistic_metrics = pd.concat(holistic_tables, axis=0, ignore_index=True).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)
    cue_winner_metrics = pd.concat(cue_winner_tables, axis=0, ignore_index=True).sort_values(["triplet_id", "condition"], kind="stable").reset_index(drop=True)
    layer1_formula_fit = build_layer1_formula_fit_summary(layer1_trial_metrics)
    _ = build_layer1_composition_summary(layer1_trial_metrics)

    return Fig5BackboneResult(
        config={
            "sample_ms": float(config.sample_ms),
            "delay1_ms": float(config.delay1_ms),
            "distractor_ms": float(config.distractor_ms),
            "delay2_ms": float(config.delay2_ms),
            "probe_ms": float(config.probe_ms),
            "batch_size": int(config.batch_size),
            "max_probes": int(config.max_probes),
            "samples_per_probe": int(config.samples_per_probe),
            "max_triplets": int(config.max_triplets),
            "num_sim_bins": int(config.num_sim_bins),
            "foreground_threshold": float(config.foreground_threshold),
            "dilation_radius": int(config.dilation_radius),
            "winner_window_frac": float(config.winner_window_frac),
            "tie_threshold": float(config.tie_threshold),
        },
        triplets=triplets_aug[
            [
                "triplet_id",
                "sample_id",
                "distractor_id",
                "probe_id",
                "sample_label",
                "distractor_label",
                "probe_label",
                "sample_only_area",
                "distractor_only_area",
                "shared_area",
            ]
        ].copy(),
        region_support_condition=region_support_condition,
        layer1_trial_metrics=layer1_trial_metrics,
        layer1_formula_fit=layer1_formula_fit,
        reshaping_metrics=reshaping_metrics[["triplet_id", "barR_L2", "barR_L3", "barP_L2", "barP_L3"]].copy(),
        holistic_metrics=holistic_metrics[
            [
                "triplet_id",
                "H_full_SP",
                "H_adv_SP",
                "W_probe_cue_SP",
                "winner_label_cue_SP",
                "H_full_DP",
                "H_adv_DP",
                "W_probe_cue_DP",
                "winner_label_cue_DP",
                "H_full_SDP",
                "H_adv_SDP",
                "W_probe_cue_SDP",
                "winner_label_cue_SDP",
            ]
        ].copy(),
        cue_winner_metrics=cue_winner_metrics.copy(),
        stats={
            "triplet_count": int(len(triplets_aug)),
            "batch_count": int(total_batches),
            "shared_rollout_count": int(rollout_count),
            "per_batch_condition_count": int(5 + len(CUE_CONDITIONS)),
            "avoided_stage_modules": ["src.experiments.distractor_region_ux_mechanism_experiment", "src.experiments.distractor_chunk_holistic_invocation_experiment"],
        },
    )


__all__ = [
    "Fig5BackboneResult",
    "build_fig5_backbone_config",
    "run_fig5_backbone",
]
