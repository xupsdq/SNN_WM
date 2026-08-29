from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.paper_figures.common.sequence_root.selection import (
    select_matched_nonpeak_mask,
    select_top_mask,
)
from src.experiments.paper_figures.fig6.constants import (
    PRIMARY_LAYER,
    SEQUENCE_TRIAL_COLUMNS,
    STATE_BANK_MANIFEST_COLUMNS,
    STATE_VARIABLE,
)
from src.experiments.paper_figures.fig6.subexperiments.helpers_1 import _item_entry_mask
from src.experiments.paper_figures.fig6.subexperiments.helpers_1 import (
    _leave_one_out_support_map,
    _leave_one_out_support_maps_batch,
    _progress,
    _save_csv,
    _sequence_support_maps,
    _sequence_support_maps_batch,
)
from src.experiments.paper_figures.fig6.subexperiments.helpers_2 import (
    _is_proxy_mode,
    _pairwise_image_sims,
)
from src.experiments.paper_figures.fig6.types import ExperimentContext, PeakAmplifiedReentryBank



def build_sequence_trials(ctx: ExperimentContext) -> pd.DataFrame:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg.network_seed))
    lengths = list(cfg.sequence_lengths)
    image_ids_by_label = {label: np.asarray(ids, dtype=np.int64) for label, ids in ctx.class_index.items()}
    rows: list[dict[str, Any]] = []
    for sequence_id in _progress(range(int(cfg.num_sequences)), total=int(cfg.num_sequences), desc="fig6 sequence specs", enabled=cfg.show_progress):
        seq_len = int(lengths[sequence_id % len(lengths)])
        labels = rng.choice(np.arange(10), size=seq_len, replace=seq_len > 10)
        image_ids = [int(rng.choice(image_ids_by_label[int(label)])) for label in labels]
        sims = _pairwise_image_sims(ctx.dataset, image_ids)
        sequence_seed = int(rng.integers(0, 2**31 - 1))
        for stage_k, (image_id, label) in enumerate(zip(image_ids, labels), start=1):
            rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "sequence_id": int(sequence_id),
                    "seq_len": int(seq_len),
                    "stage_k": int(stage_k),
                    "item_image_id": int(image_id),
                    "item_label": int(label),
                    "ordered_item_ids": ";".join(map(str, image_ids)),
                    "ordered_item_labels": ";".join(map(str, [int(v) for v in labels])),
                    "sequence_seed": int(sequence_seed),
                    "mean_pairwise_image_similarity": float(np.mean(sims)) if sims else 0.0,
                    "max_pairwise_image_similarity": float(np.max(sims)) if sims else 0.0,
                    "min_pairwise_image_similarity": float(np.min(sims)) if sims else 0.0,
                }
            )
    out = pd.DataFrame(rows, columns=SEQUENCE_TRIAL_COLUMNS)
    _save_csv(ctx, out, ctx.trial_specs_dir / "sequence_trials.csv")
    ctx.n_sequences = int(out["sequence_id"].nunique())
    ctx.completed_modules["sequence_trials"] = True
    return out

def run_sequence_bank(ctx: ExperimentContext, sequence_trials: pd.DataFrame) -> PeakAmplifiedReentryBank:
    seq_ids = sorted(sequence_trials["sequence_id"].unique())
    n_seq = len(seq_ids)
    n_units = 28 * 28
    update_count = np.zeros((n_seq, n_units), dtype=np.float32)
    last_update_position = np.zeros((n_seq, n_units), dtype=np.int16)
    time_since_last_update = np.zeros((n_seq, n_units), dtype=np.int16)
    update_exposure_by_item = np.zeros((n_seq, max(sequence_trials["seq_len"]), n_units), dtype=np.float32)
    item_activation_history = np.zeros_like(update_exposure_by_item)
    g_baseline = np.zeros((n_seq, n_units), dtype=np.float32)
    g_final = np.zeros((n_seq, n_units), dtype=np.float32)
    delta_support = np.zeros((n_seq, n_units), dtype=np.float32)
    peak_mask = np.zeros((n_seq, n_units), dtype=bool)
    nonpeak_mask = np.zeros((n_seq, n_units), dtype=bool)
    prior_updated_mask = np.zeros((n_seq, n_units), dtype=bool)
    boundaries: dict[int, Mapping[str, Mapping[str, Any]]] = {}
    manifest_rows: list[dict[str, Any]] = []
    sequence_meta_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], Any] = {}
    enable_sequence_batch = bool(getattr(ctx.cfg, "enable_sequence_bank_batch", False))
    sequence_batch_size = max(1, int(getattr(ctx.cfg, "batch_size", 1)))
    map_jobs: list[dict[str, Any]] = []

    def flush_map_jobs() -> None:
        if not map_jobs:
            return
        jobs = list(map_jobs)
        map_jobs.clear()
        if enable_sequence_batch and len(jobs) > 1:
            map_results = _sequence_support_maps_batch(
                ctx,
                [job["image_ids"] for job in jobs],
                encode_cache=encode_cache,
            )
        else:
            map_results = [
                _sequence_support_maps(
                    ctx,
                    job["image_ids"],
                    job["masks"],
                    update_count[int(job["row_idx"])],
                    last_update_position[int(job["row_idx"])],
                    int(job["seq_len"]),
                    encode_cache=encode_cache,
                )
                for job in jobs
            ]
        for job, (baseline_map, final_map, boundary) in zip(jobs, map_results):
            row_idx = int(job["row_idx"])
            sequence_id = int(job["sequence_id"])
            seq_len = int(job["seq_len"])
            image_ids = [int(v) for v in job["image_ids"]]
            labels = [int(v) for v in job["labels"]]
            if boundary:
                boundaries[sequence_id] = boundary
            else:
                raise RuntimeError(f"Fig.6 sequence_id={sequence_id} did not produce an S_final boundary.")
            g_baseline[row_idx] = baseline_map.reshape(-1).astype(np.float32)
            g_final[row_idx] = final_map.reshape(-1).astype(np.float32)
            delta_support[row_idx] = g_final[row_idx] - g_baseline[row_idx]
            peaks = select_top_mask(
                delta_support[row_idx].reshape(28, 28),
                ctx.cfg.peak_q,
                positive=delta_support[row_idx].reshape(28, 28) > 0,
            )
            peak_mask[row_idx] = peaks.reshape(-1)
            nonpeak_mask[row_idx] = select_matched_nonpeak_mask(
                peak_mask[row_idx],
                prior_updated_mask[row_idx],
                int(ctx.cfg.network_seed) + sequence_id,
            )
            sequence_meta_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": sequence_id,
                    "seq_len": seq_len,
                    "ordered_item_ids": ";".join(map(str, image_ids)),
                    "ordered_item_labels": ";".join(map(str, labels)),
                }
            )
            for state_condition, stage_k, arrs in (
                ("S0", 0, {"G_baseline": g_baseline[row_idx]}),
                ("S_final", seq_len, {"G_final": g_final[row_idx], "delta_support": delta_support[row_idx]}),
            ):
                for key, arr in arrs.items():
                    manifest_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "sequence_id": sequence_id,
                            "seq_len": seq_len,
                            "state_condition": state_condition,
                            "stage_k": int(stage_k),
                            "layer": PRIMARY_LAYER,
                            "state_variable": STATE_VARIABLE if key != "delta_support" else "delta_support",
                            "shape": "28x28",
                            "storage_file": "final_support_maps.npz",
                            "storage_key": f"{key}_sequence_{sequence_id}",
                            "captured_after": state_condition,
                            "sample_ms": int(ctx.cfg.sample_ms),
                            "delay_ms": int(ctx.cfg.delay_ms),
                        }
                    )

    for row_idx, sequence_id in _progress(enumerate(seq_ids), total=len(seq_ids), desc="fig6 sequence bank", enabled=ctx.cfg.show_progress):
        group = sequence_trials[sequence_trials["sequence_id"].eq(sequence_id)].sort_values("stage_k")
        seq_len = int(group["seq_len"].iloc[0])
        image_ids = [int(v) for v in group["item_image_id"].tolist()]
        labels = [int(v) for v in group["item_label"].tolist()]
        masks = np.stack([_item_entry_mask(ctx, image_id, ctx.cfg.sample_steps, cache=encode_cache) for image_id in image_ids], axis=0)
        exposure = masks.reshape(seq_len, -1).astype(np.float32)
        update_exposure_by_item[row_idx, :seq_len, :] = exposure
        item_activation_history[row_idx, :seq_len, :] = exposure
        update_count[row_idx] = exposure.sum(axis=0)
        for pos in range(seq_len):
            active = exposure[pos] > 0
            last_update_position[row_idx, active] = pos + 1
        time_since_last_update[row_idx] = np.where(last_update_position[row_idx] > 0, seq_len - last_update_position[row_idx], seq_len + 1)
        prior_updated_mask[row_idx] = update_count[row_idx] > 0
        if enable_sequence_batch and map_jobs and (int(map_jobs[0]["seq_len"]) != seq_len or len(map_jobs) >= sequence_batch_size):
            flush_map_jobs()
        map_jobs.append(
            {
                "row_idx": int(row_idx),
                "sequence_id": int(sequence_id),
                "seq_len": int(seq_len),
                "image_ids": image_ids,
                "labels": labels,
                "masks": masks,
                "entry_mask_mode": str(ctx.cfg.real_probe_entry_mode),
            }
        )
        if not enable_sequence_batch:
            flush_map_jobs()

    flush_map_jobs()

    missing_boundaries = [int(seq_id) for seq_id in seq_ids if int(seq_id) not in boundaries]
    if missing_boundaries:
        raise RuntimeError(
            "Fig.6 sequence bank missing S_final boundaries for "
            f"{len(missing_boundaries)}/{len(seq_ids)} sequence trials: {missing_boundaries[:10]}"
        )

    _save_csv(ctx, pd.DataFrame(manifest_rows, columns=STATE_BANK_MANIFEST_COLUMNS), ctx.raw_dir / "state_bank_manifest.csv")
    np.savez_compressed(
        ctx.raw_dir / "update_history_matrix.npz",
        update_count=update_count,
        last_update_position=last_update_position,
        time_since_last_update=time_since_last_update,
        update_exposure_by_item=update_exposure_by_item,
        item_activation_history=item_activation_history,
        unit_ids=np.arange(n_units, dtype=np.int32),
        sequence_ids=np.asarray(seq_ids, dtype=np.int32),
    )
    np.savez_compressed(
        ctx.raw_dir / "final_support_maps.npz",
        G_baseline=g_baseline,
        G_final=g_final,
        delta_support=delta_support,
        peak_mask=peak_mask.astype(np.uint8),
        nonpeak_mask=nonpeak_mask.astype(np.uint8),
        unit_ids=np.arange(n_units, dtype=np.int32),
        sequence_ids=np.asarray(seq_ids, dtype=np.int32),
    )
    ctx.output_files["state_bank_manifest"] = "data/raw/state_bank_manifest.csv"
    ctx.output_files["update_history_matrix"] = "data/raw/update_history_matrix.npz"
    ctx.output_files["final_support_maps"] = "data/raw/final_support_maps.npz"
    ctx.completed_modules["sequence_bank"] = True
    return PeakAmplifiedReentryBank(
        sequence_trials=sequence_trials.reset_index(drop=True),
        sequence_meta=pd.DataFrame(sequence_meta_rows),
        probe_trials=pd.DataFrame(),
        matched_groups=pd.DataFrame(),
        update_count=update_count,
        last_update_position=last_update_position,
        time_since_last_update=time_since_last_update,
        update_exposure_by_item=update_exposure_by_item,
        item_activation_history=item_activation_history,
        g_baseline=g_baseline,
        g_final=g_final,
        delta_support=delta_support,
        peak_mask=peak_mask,
        nonpeak_mask=nonpeak_mask,
        prior_updated_mask=prior_updated_mask,
        boundaries=boundaries,
        reentry_metrics=pd.DataFrame(),
        downstream_metrics=pd.DataFrame(),
    )

def run_leave_one_item_out_support_bank(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> dict[int, list[dict[str, Any]]]:
    proxy_mode = _is_proxy_mode(ctx)
    rows_by_sequence: dict[int, list[dict[str, Any]]] = {}
    raw_payload: dict[str, np.ndarray] = {}
    encode_cache: dict[tuple[Any, ...], Any] = {}
    for seq_idx, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 leave-one-out sequences", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        image_ids = [int(v) for v in str(meta.ordered_item_ids).split(";") if str(v) != ""]
        labels = [int(v) for v in str(meta.ordered_item_labels).split(";") if str(v) != ""]
        seq_len = int(meta.seq_len)
        full = bank.g_final[seq_idx].reshape(28, 28)
        baseline = bank.g_baseline[seq_idx].reshape(28, 28)
        sequence_rows: list[dict[str, Any]] = []
        if bool(getattr(ctx.cfg, "enable_leave_one_out_batch", False)):
            minus_maps = _leave_one_out_support_maps_batch(ctx, image_ids, encode_cache=encode_cache)
        else:
            minus_maps = []
        for removed_idx in _progress(range(seq_len), total=seq_len, desc="fig6 leave-one-out items", enabled=ctx.cfg.show_progress):
            minus_map = (
                minus_maps[int(removed_idx)]
                if minus_maps
                else _leave_one_out_support_map(ctx, image_ids, removed_idx, encode_cache=encode_cache)
            )
            delta_minus = minus_map - baseline
            loss_map = np.maximum(full - minus_map, 0.0).astype(np.float32)
            sequence_rows.append(
                {
                    "removed_position": int(removed_idx + 1),
                    "removed_label": int(labels[removed_idx]) if removed_idx < len(labels) else -1,
                    "removed_image_id": int(image_ids[removed_idx]) if removed_idx < len(image_ids) else -1,
                    "G_minus_i": minus_map.reshape(-1).astype(np.float32),
                    "delta_minus_i": delta_minus.reshape(-1).astype(np.float32),
                    "loss_map_i": loss_map.reshape(-1).astype(np.float32),
                    "proxy_mode": bool(proxy_mode),
                }
            )
            if ctx.cfg.save_full_traces:
                raw_payload[f"sequence_{seq_id}_removed_{removed_idx + 1}_G_minus_i"] = minus_map.astype(np.float32)
                raw_payload[f"sequence_{seq_id}_removed_{removed_idx + 1}_loss_map_i"] = loss_map.astype(np.float32)
        rows_by_sequence[seq_id] = sequence_rows
    if raw_payload:
        np.savez_compressed(ctx.raw_dir / "leave_one_item_out_support_maps.npz", **raw_payload)
        ctx.output_files["leave_one_item_out_support_maps"] = "data/raw/leave_one_item_out_support_maps.npz"
    ctx.completed_modules["peak_source_attribution_replay"] = True
    return rows_by_sequence
