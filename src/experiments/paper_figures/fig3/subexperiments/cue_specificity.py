from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.experiments.common.dataset import encode_images
from src.experiments.paper_figures.fig3.schemas import CUE_SPECIFICITY_MISMATCHED_SELECTION_POLICY
from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import concat_named_boundaries, run_probe_readout_from_boundary
from src.experiments.paper_figures.fig3.subexperiments.weak_probe import _make_weak_probe_spikes_encoded_dropout
from src.experiments.paper_figures.fig3.types import ExperimentContext, MultiItemSequenceLandscapeBank


NUM_CLASSES = 10
DEFAULT_CUE_TYPES = ("matched", "mismatched", "unseen")
DEFAULT_STATE_SPECS = (("S_final", "sequence_state"), ("S0", "cue_only"))
UNSEEN_SELECTION_POLICY = "stable_absent_label_then_stable_class_index_legacy_script_v1"
MATCHED_SELECTION_POLICY = "target_sequence_image_v1"


def build_cue_specificity_specs(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    access_jobs: pd.DataFrame,
    *,
    seq_len: int,
    delay_ms: int,
    keep_prob: float,
    cue_types: Sequence[str] = DEFAULT_CUE_TYPES,
    state_specs: Sequence[tuple[str, str]] = DEFAULT_STATE_SPECS,
) -> pd.DataFrame:
    cue_types = _normalize_cue_types(cue_types)
    state_specs = tuple((str(state), str(memory)) for state, memory in state_specs)
    jobs = select_cue_specificity_jobs(access_jobs, seq_len=seq_len, delay_ms=delay_ms, keep_prob=keep_prob)
    if jobs.empty:
        raise ValueError(f"No Fig.3 weak-cue access jobs for cue specificity: K={seq_len}, delay_ms={delay_ms}, keep_prob={keep_prob}.")
    sequence_lookup = sequence_lookup_for(sequence_trials, seq_len=seq_len)
    rows: list[dict[str, Any]] = []
    condition_id = f"K{int(seq_len)}_D{int(delay_ms)}"
    trial_id = 0
    mask_group_id = 0
    sorted_jobs = jobs.sort_values(["sequence_id", "target_position", "repeat_id", "job_id"]).reset_index(drop=True)
    for _, job in sorted_jobs.iterrows():
        seq_id = int(job["sequence_id"])
        if seq_id not in sequence_lookup:
            raise ValueError(f"Cue specificity job references missing K{seq_len} sequence_id={seq_id}.")
        seq = sequence_lookup[seq_id]
        labels = list(seq["labels"])
        image_ids = list(seq["image_ids"])
        if len(set(labels)) != len(labels):
            raise ValueError(f"Cue specificity requires unique item labels within sequence_id={seq_id}; found {labels}.")
        unseen_labels = [label for label in range(NUM_CLASSES) if label not in set(labels)]
        if not unseen_labels:
            raise ValueError(f"Cue specificity requires at least one unseen class for sequence_id={seq_id}.")
        target_position = int(job["target_position"])
        target_idx = target_position - 1
        if target_idx < 0 or target_idx >= len(image_ids):
            raise ValueError(f"Cue specificity target_position={target_position} outside K={len(image_ids)} for sequence_id={seq_id}.")
        cue_specs = _cue_specs_for_job(
            seq_id=seq_id,
            target_position=target_position,
            repeat_id=int(job["repeat_id"]),
            image_ids=image_ids,
            labels=labels,
            unseen_labels=unseen_labels,
            class_index=ctx.class_index,
        )
        for cue_spec in cue_specs:
            if cue_spec["cue_type"] not in cue_types:
                continue
            mask_group_id += 1
            for state_idx, (state_condition, memory_condition) in enumerate(state_specs):
                trial_id += 1
                rows.append(
                    {
                        "cue_specificity_trial_id": int(trial_id),
                        "job_id": int(job["job_id"]),
                        "condition_id": condition_id,
                        "sequence_id": seq_id,
                        "seq_len": int(seq_len),
                        "delay_ms": int(delay_ms),
                        "target_position": target_position,
                        "target_image_id": int(job["target_image_id"]),
                        "target_label": int(labels[target_idx]),
                        "cue_type": str(cue_spec["cue_type"]),
                        "cue_position": int(cue_spec["cue_position"]),
                        "cue_image_id": int(cue_spec["cue_image_id"]),
                        "cue_label": int(cue_spec["cue_label"]),
                        "state_condition": str(state_condition),
                        "memory_condition": str(memory_condition),
                        "state_index": int(state_idx),
                        "keep_prob": float(job["keep_prob"]),
                        "repeat_id": int(job["repeat_id"]),
                        "mask_seed": int(job["mask_seed"]),
                        "mask_group_id": int(mask_group_id),
                        "ordered_item_ids": ";".join(str(v) for v in image_ids),
                        "ordered_item_labels": ";".join(str(v) for v in labels),
                        "unseen_labels": ";".join(str(v) for v in unseen_labels),
                        "cue_selection_policy": str(cue_spec["cue_selection_policy"]),
                        "cue_is_sequence_member": int(cue_spec["cue_image_id"] in set(image_ids)),
                        "cue_is_same_label_foil": int(cue_spec["cue_type"] == "mismatched"),
                        "mismatched_selection_policy": CUE_SPECIFICITY_MISMATCHED_SELECTION_POLICY,
                        "unseen_selection_policy": UNSEEN_SELECTION_POLICY,
                    }
                )
    if not rows:
        raise ValueError(f"Cue specificity produced no rows after cue-type filter {list(cue_types)}.")
    return pd.DataFrame(rows).sort_values("cue_specificity_trial_id").reset_index(drop=True)


def select_cue_specificity_jobs(access_jobs: pd.DataFrame, *, seq_len: int, delay_ms: int, keep_prob: float) -> pd.DataFrame:
    required = {"job_type", "seq_len", "delay_ms", "keep_prob"}
    missing = required - set(access_jobs.columns)
    if missing:
        raise ValueError(f"access_job_specs missing required columns for cue specificity: {sorted(missing)}")
    jobs = access_jobs[
        access_jobs["job_type"].astype(str).eq("weak_cue")
        & access_jobs["seq_len"].astype(int).eq(int(seq_len))
        & access_jobs["delay_ms"].astype(int).eq(int(delay_ms))
        & np.isclose(pd.to_numeric(access_jobs["keep_prob"], errors="coerce").to_numpy(dtype=float), float(keep_prob))
    ].copy()
    return jobs.reset_index(drop=True)


def run_cue_specificity_readout(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    cue_specs: pd.DataFrame,
    *,
    readout_batch_size: int,
) -> pd.DataFrame:
    if cue_specs.empty:
        raise ValueError("cue_specificity_specs is empty.")
    _require_columns(
        cue_specs,
        [
            "cue_specificity_trial_id",
            "condition_id",
            "sequence_id",
            "seq_len",
            "delay_ms",
            "target_position",
            "target_image_id",
            "target_label",
            "cue_type",
            "cue_position",
            "cue_image_id",
            "cue_label",
            "state_condition",
            "memory_condition",
            "keep_prob",
            "repeat_id",
            "mask_seed",
            "mask_group_id",
            "ordered_item_ids",
            "ordered_item_labels",
            "unseen_labels",
            "cue_selection_policy",
            "cue_is_sequence_member",
            "cue_is_same_label_foil",
            "mismatched_selection_policy",
        ],
        "cue_specificity_specs",
    )
    raw_rows: list[dict[str, Any]] = []
    pending_boundaries: list[Mapping[str, Mapping[str, torch.Tensor]]] = []
    pending_spikes: list[torch.Tensor] = []
    pending_rows: list[dict[str, Any]] = []
    encoded_cache: dict[int, torch.Tensor] = {}
    batch_size = max(1, int(readout_batch_size))
    ordered = cue_specs.sort_values("cue_specificity_trial_id").reset_index(drop=True)
    for _, group in ordered.groupby("mask_group_id", sort=True):
        group = group.sort_values("cue_specificity_trial_id").reset_index(drop=True)
        first = group.iloc[0]
        full_spikes = _encoded_image(ctx, int(first["cue_image_id"]), encoded_cache)
        weak_spikes, mask_info = _make_weak_probe_spikes_encoded_dropout(
            full_spikes,
            float(first["keep_prob"]),
            seed=int(first["mask_seed"]),
            same_mask_count=len(group),
            use_same_mask_across_states=True,
            device=ctx.device,
        )
        for state_idx, (_, spec_row) in enumerate(group.iterrows()):
            seq_id = int(spec_row["sequence_id"])
            delay_ms = int(spec_row["delay_ms"])
            condition_id = str(spec_row["condition_id"])
            state_condition = str(spec_row["state_condition"])
            pending_boundaries.append(bank.boundary_for(seq_id, state_condition, condition_id=condition_id, delay_ms=delay_ms))
            pending_spikes.append(weak_spikes[state_idx : state_idx + 1])
            pending_rows.append(_base_raw_row_from_spec(ctx, spec_row, mask_info))
        if len(pending_rows) >= batch_size:
            raw_rows.extend(_flush_readout_batch(ctx, pending_boundaries, pending_spikes, pending_rows))
            pending_boundaries = []
            pending_spikes = []
            pending_rows = []
    if pending_rows:
        raw_rows.extend(_flush_readout_batch(ctx, pending_boundaries, pending_spikes, pending_rows))
    return pd.DataFrame(raw_rows)


def compute_cue_specificity_tables(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics = compute_cue_specificity_metrics(raw)
    memory_gain = compute_cue_specificity_memory_gain(metrics)
    serial = compute_cue_specificity_serial_summary(metrics)
    contrast = compute_cue_specificity_contrast_summary(serial)
    summary = cue_specificity_summary_table(metrics)
    return {
        "cue_specificity_trial_readout": raw.reset_index(drop=True),
        "cue_specificity_metrics": metrics,
        "cue_specificity_memory_gain": memory_gain,
        "cue_specificity_serial_summary": serial,
        "cue_specificity_contrast_summary": contrast,
        "cue_specificity_summary": summary,
    }


def compute_cue_specificity_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "network_seed",
        "condition_id",
        "sequence_id",
        "seq_len",
        "delay_ms",
        "target_position",
        "cue_type",
        "state_condition",
        "memory_condition",
        "keep_prob",
    ]
    metric_cols = {
        "P_target": "pred_is_target",
        "P_cue_label": "pred_is_cue_label",
        "P_seen_item": "pred_is_seen_item",
        "P_other_seen_item": "pred_is_other_seen_item",
        "P_latest_item": "pred_is_latest_item",
        "P_unseen": "pred_is_unseen",
        "P_silent": "silent",
    }
    rows: list[dict[str, Any]] = []
    for keys, part in raw.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        for out_col, src_col in metric_cols.items():
            row[out_col] = float(pd.to_numeric(part[src_col], errors="coerce").mean())
        row["n_trials"] = int(len(part))
        rows.append(row)
    return pd.DataFrame(rows, columns=[*group_cols, *metric_cols.keys(), "n_trials"])


def compute_cue_specificity_memory_gain(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "condition_id",
        "sequence_id",
        "seq_len",
        "delay_ms",
        "target_position",
        "cue_type",
        "keep_prob",
        "P_target_sequence_state",
        "P_target_cue_only",
        "target_memory_gain",
        "P_seen_item_sequence_state",
        "P_seen_item_cue_only",
        "seen_item_memory_gain",
        "P_silent_sequence_state",
        "P_silent_cue_only",
        "silent_delta",
        "n_trials_sequence_state",
        "n_trials_cue_only",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_cols = [
        "network_seed",
        "condition_id",
        "sequence_id",
        "seq_len",
        "delay_ms",
        "target_position",
        "cue_type",
        "keep_prob",
    ]
    for keys, part in metrics.groupby(group_cols, sort=True):
        seq = part[part["state_condition"].astype(str).eq("S_final")]
        cue = part[part["state_condition"].astype(str).eq("S0")]
        row = dict(zip(group_cols, keys))
        p_target_seq = _first_metric(seq, "P_target")
        p_target_cue = _first_metric(cue, "P_target")
        p_seen_seq = _first_metric(seq, "P_seen_item")
        p_seen_cue = _first_metric(cue, "P_seen_item")
        p_silent_seq = _first_metric(seq, "P_silent")
        p_silent_cue = _first_metric(cue, "P_silent")
        row.update(
            {
                "P_target_sequence_state": p_target_seq,
                "P_target_cue_only": p_target_cue,
                "target_memory_gain": _nan_diff(p_target_seq, p_target_cue),
                "P_seen_item_sequence_state": p_seen_seq,
                "P_seen_item_cue_only": p_seen_cue,
                "seen_item_memory_gain": _nan_diff(p_seen_seq, p_seen_cue),
                "P_silent_sequence_state": p_silent_seq,
                "P_silent_cue_only": p_silent_cue,
                "silent_delta": _nan_diff(p_silent_seq, p_silent_cue),
                "n_trials_sequence_state": int(_first_metric(seq, "n_trials", default=0.0)),
                "n_trials_cue_only": int(_first_metric(cue, "n_trials", default=0.0)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def compute_cue_specificity_serial_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    value_cols = ["P_target", "P_cue_label", "P_seen_item", "P_other_seen_item", "P_latest_item", "P_unseen", "P_silent"]
    group_cols = ["state_condition", "memory_condition", "cue_type", "target_position"]
    for keys, part in metrics.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        for col in value_cols:
            vals = pd.to_numeric(part[col], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{col}_mean"] = float(vals.mean()) if vals.size else float("nan")
            row[f"{col}_sem"] = _sem(vals)
        row["n_sequences"] = int(part["sequence_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def compute_cue_specificity_contrast_summary(serial_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state_condition, state_part in serial_summary.groupby("state_condition", sort=True):
        pivot = state_part.pivot_table(index="target_position", columns="cue_type", values="P_target_mean", aggfunc="first")
        for target_position, row in pivot.iterrows():
            matched = _float(row.get("matched"))
            mismatched = _float(row.get("mismatched"))
            unseen = _float(row.get("unseen"))
            rows.append(
                {
                    "state_condition": state_condition,
                    "target_position": int(target_position),
                    "P_target_matched": matched,
                    "P_target_mismatched": mismatched,
                    "P_target_unseen": unseen,
                    "matched_minus_unseen": _nan_diff(matched, unseen),
                    "matched_minus_mismatched": _nan_diff(matched, mismatched),
                }
            )
    return pd.DataFrame(rows)


def cue_specificity_scientific_checks(metrics: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    s_final = metrics[metrics["state_condition"].astype(str).eq("S_final")]
    s0 = metrics[metrics["state_condition"].astype(str).eq("S0")]
    memory_gain = compute_cue_specificity_memory_gain(metrics)
    out["S_final_P_target_by_cue"] = _mean_by_cue(s_final, "P_target")
    out["S0_P_target_by_cue"] = _mean_by_cue(s0, "P_target")
    matched = out["S_final_P_target_by_cue"].get("matched", float("nan"))
    mismatched = out["S_final_P_target_by_cue"].get("mismatched", float("nan"))
    unseen = out["S_final_P_target_by_cue"].get("unseen", float("nan"))
    out["S_final_matched_minus_unseen_P_target"] = _nan_diff(matched, unseen)
    out["S_final_matched_minus_mismatched_P_target"] = _nan_diff(matched, mismatched)
    out["S_final_matched_gt_unseen_P_target"] = bool(np.isfinite(out["S_final_matched_minus_unseen_P_target"]) and out["S_final_matched_minus_unseen_P_target"] > 0.0)
    out["S_final_matched_gt_mismatched_P_target"] = bool(np.isfinite(out["S_final_matched_minus_mismatched_P_target"]) and out["S_final_matched_minus_mismatched_P_target"] > 0.0)
    s_final_seen = _mean_by_cue(s_final, "P_seen_item")
    s0_seen = _mean_by_cue(s0, "P_seen_item")
    out["S_final_P_seen_item_by_cue"] = s_final_seen
    out["S0_P_seen_item_by_cue"] = s0_seen
    out["generic_seen_item_arousal"] = {
        cue: _nan_diff(s_final_seen.get(cue, float("nan")), s0_seen.get(cue, float("nan")))
        for cue in ("mismatched", "unseen")
    }
    out["target_memory_gain_by_cue"] = _mean_by_cue(memory_gain, "target_memory_gain")
    gain_matched = out["target_memory_gain_by_cue"].get("matched", float("nan"))
    gain_mismatched = out["target_memory_gain_by_cue"].get("mismatched", float("nan"))
    gain_unseen = out["target_memory_gain_by_cue"].get("unseen", float("nan"))
    out["target_memory_gain_matched_minus_mismatched"] = _nan_diff(gain_matched, gain_mismatched)
    out["target_memory_gain_matched_minus_unseen"] = _nan_diff(gain_matched, gain_unseen)
    out["target_memory_gain_matched_gt_mismatched"] = bool(
        np.isfinite(out["target_memory_gain_matched_minus_mismatched"])
        and out["target_memory_gain_matched_minus_mismatched"] > 0.0
    )
    out["target_memory_gain_matched_gt_unseen"] = bool(
        np.isfinite(out["target_memory_gain_matched_minus_unseen"])
        and out["target_memory_gain_matched_minus_unseen"] > 0.0
    )
    return out


def cue_specificity_summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    checks = cue_specificity_scientific_checks(metrics)
    rows = [
        {"metric": "S_final_matched_minus_unseen_P_target", "value": checks["S_final_matched_minus_unseen_P_target"]},
        {"metric": "S_final_matched_minus_mismatched_P_target", "value": checks["S_final_matched_minus_mismatched_P_target"]},
        {"metric": "S_final_matched_gt_unseen_P_target", "value": int(checks["S_final_matched_gt_unseen_P_target"])},
        {"metric": "S_final_matched_gt_mismatched_P_target", "value": int(checks["S_final_matched_gt_mismatched_P_target"])},
    ]
    for cue, value in checks["generic_seen_item_arousal"].items():
        rows.append({"metric": f"generic_seen_item_arousal_{cue}", "value": value})
    for cue, value in checks["S_final_P_target_by_cue"].items():
        rows.append({"metric": f"S_final_P_target_{cue}", "value": value})
    for cue, value in checks["S_final_P_seen_item_by_cue"].items():
        rows.append({"metric": f"S_final_P_seen_item_{cue}", "value": value})
    for cue, value in checks["target_memory_gain_by_cue"].items():
        rows.append({"metric": f"target_memory_gain_{cue}", "value": value})
    rows.extend(
        [
            {
                "metric": "target_memory_gain_matched_minus_mismatched",
                "value": checks["target_memory_gain_matched_minus_mismatched"],
            },
            {
                "metric": "target_memory_gain_matched_minus_unseen",
                "value": checks["target_memory_gain_matched_minus_unseen"],
            },
            {
                "metric": "target_memory_gain_matched_gt_mismatched",
                "value": int(checks["target_memory_gain_matched_gt_mismatched"]),
            },
            {
                "metric": "target_memory_gain_matched_gt_unseen",
                "value": int(checks["target_memory_gain_matched_gt_unseen"]),
            },
        ]
    )
    return pd.DataFrame(rows)


def sequence_lookup_for(sequence_trials: pd.DataFrame, *, seq_len: int) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    use = sequence_trials[sequence_trials["seq_len"].astype(int).eq(int(seq_len))].copy()
    for seq_id, part in use.groupby("sequence_id", sort=True):
        ordered = part.sort_values("stage_k")
        out[int(seq_id)] = {
            "image_ids": [int(v) for v in ordered["item_image_id"].tolist()],
            "labels": [int(v) for v in ordered["item_label"].tolist()],
        }
    return out


def _cue_specs_for_job(
    *,
    seq_id: int,
    target_position: int,
    repeat_id: int,
    image_ids: Sequence[int],
    labels: Sequence[int],
    unseen_labels: Sequence[int],
    class_index: Mapping[int, Sequence[int]],
) -> list[dict[str, Any]]:
    target_idx = int(target_position) - 1
    target_label = int(labels[target_idx])
    target_image_id = int(image_ids[target_idx])
    mismatched_image_id = _same_label_foil_image_id(
        seq_id=seq_id,
        target_position=target_position,
        repeat_id=repeat_id,
        target_label=target_label,
        target_image_id=target_image_id,
        image_ids=image_ids,
        class_index=class_index,
    )
    unseen_label = int(unseen_labels[_stable_index(seq_id, target_position, repeat_id, len(unseen_labels), offset=91)])
    unseen_pool = list(class_index.get(unseen_label, ()))
    if not unseen_pool:
        raise ValueError(f"No dataset images available for unseen label {unseen_label}.")
    return [
        {
            "cue_type": "matched",
            "cue_position": int(target_position),
            "cue_image_id": target_image_id,
            "cue_label": target_label,
            "cue_selection_policy": MATCHED_SELECTION_POLICY,
        },
        {
            "cue_type": "mismatched",
            "cue_position": 0,
            "cue_image_id": mismatched_image_id,
            "cue_label": target_label,
            "cue_selection_policy": CUE_SPECIFICITY_MISMATCHED_SELECTION_POLICY,
        },
        {
            "cue_type": "unseen",
            "cue_position": 0,
            "cue_image_id": int(unseen_pool[_stable_index(seq_id, target_position, repeat_id, len(unseen_pool), offset=193)]),
            "cue_label": unseen_label,
            "cue_selection_policy": UNSEEN_SELECTION_POLICY,
        },
    ]


def _base_raw_row_from_spec(ctx: ExperimentContext, row: pd.Series, mask_info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "job_id": int(row["job_id"]),
        "condition_id": str(row["condition_id"]),
        "sequence_id": int(row["sequence_id"]),
        "seq_len": int(row["seq_len"]),
        "delay_ms": int(row["delay_ms"]),
        "target_position": int(row["target_position"]),
        "target_image_id": int(row["target_image_id"]),
        "target_label": int(row["target_label"]),
        "cue_type": str(row["cue_type"]),
        "cue_position": int(row["cue_position"]),
        "cue_image_id": int(row["cue_image_id"]),
        "cue_label": int(row["cue_label"]),
        "keep_prob": float(row["keep_prob"]),
        "repeat_id": int(row["repeat_id"]),
        "mask_seed": int(row["mask_seed"]),
        "state_condition": str(row["state_condition"]),
        "memory_condition": str(row["memory_condition"]),
        "ordered_item_ids": str(row["ordered_item_ids"]),
        "ordered_item_labels": str(row["ordered_item_labels"]),
        "unseen_labels": str(row["unseen_labels"]),
        "cue_selection_policy": str(row["cue_selection_policy"]),
        "cue_is_sequence_member": int(row["cue_is_sequence_member"]),
        "cue_is_same_label_foil": int(row["cue_is_same_label_foil"]),
        "mismatched_selection_policy": str(row["mismatched_selection_policy"]),
        "mask_space": "encoded_spikes",
        "same_mask_used_across_states": bool(mask_info["same_mask_used_across_states"]),
        "same_mask_used_across_memory_conditions": bool(mask_info["same_mask_used_across_memory_conditions"]),
        "realized_keep_fraction": float(mask_info["realized_keep_fraction"]),
        "full_spike_count": float(mask_info["full_spike_count"]),
        "weak_spike_count": float(mask_info["weak_spike_count"]),
        "weak_spike_fraction": float(mask_info["weak_spike_fraction"]),
    }


def _flush_readout_batch(
    ctx: ExperimentContext,
    boundaries: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    spikes: Sequence[torch.Tensor],
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    batched_boundary = concat_named_boundaries(boundaries, device=ctx.device)
    batched_spikes = torch.cat(list(spikes), dim=0).contiguous()
    predictions, fire_times = run_probe_readout_from_boundary(
        ctx,
        batched_boundary,
        batched_spikes,
        probe_scale=float(ctx.cfg.weak_probe_scale),
        probe_noise=float(ctx.cfg.weak_probe_noise),
        seed=0,
    )
    out: list[dict[str, Any]] = []
    for row, pred_value, fire_value in zip(rows, predictions, fire_times):
        pred = int(pred_value)
        fire = int(fire_value)
        target_label = int(row["target_label"])
        cue_label = int(row["cue_label"])
        labels = _int_list(row["ordered_item_labels"])
        silent = pred < 0
        pred_is_seen = (not silent) and pred in labels
        enriched = dict(row)
        enriched.update(
            {
                "prediction": pred,
                "pred_is_target": int(pred == target_label),
                "pred_is_cue_label": int(pred == cue_label),
                "pred_is_seen_item": int(pred_is_seen),
                "pred_is_other_seen_item": int(pred_is_seen and pred != target_label),
                "pred_is_latest_item": int(pred == int(labels[-1])),
                "pred_is_unseen": int((not silent) and pred not in labels),
                "silent": int(silent),
                "first_fire_time_ms": float(fire * ctx.cfg.dt / ms) if fire >= 0 else -1.0,
            }
        )
        out.append(enriched)
    return out


def _encoded_image(ctx: ExperimentContext, image_id: int, cache: dict[int, torch.Tensor]) -> torch.Tensor:
    image_id = int(image_id)
    if image_id not in cache:
        image = ctx.dataset[image_id][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
        cache[image_id] = encode_images(ctx.encoder, image, ctx.cfg.weak_probe_steps).to(ctx.device)
    return cache[image_id]


def _normalize_cue_types(cue_types: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(v).strip() for v in cue_types if str(v).strip())
    unknown = sorted(set(values) - set(DEFAULT_CUE_TYPES))
    if unknown:
        raise ValueError(f"Unsupported cue specificity cue types: {unknown}. Expected subset of {list(DEFAULT_CUE_TYPES)}.")
    if not values:
        raise ValueError("At least one cue specificity cue type is required.")
    return values


def _mean_by_cue(df: pd.DataFrame, column: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for cue, part in df.groupby("cue_type", sort=True):
        vals = pd.to_numeric(part[column], errors="coerce").dropna().to_numpy(dtype=float)
        out[str(cue)] = float(vals.mean()) if vals.size else float("nan")
    return out


def _same_label_foil_image_id(
    *,
    seq_id: int,
    target_position: int,
    repeat_id: int,
    target_label: int,
    target_image_id: int,
    image_ids: Sequence[int],
    class_index: Mapping[int, Sequence[int]],
) -> int:
    pool = [int(idx) for idx in class_index.get(int(target_label), ()) if int(idx) not in set(int(v) for v in image_ids)]
    if not pool:
        pool = [int(idx) for idx in class_index.get(int(target_label), ()) if int(idx) != int(target_image_id)]
    if not pool:
        raise ValueError(
            "Cue specificity same-label foil requires a non-target image for "
            f"target_label={target_label}, sequence_id={seq_id}, target_position={target_position}."
        )
    return int(pool[_stable_index(seq_id, target_position, repeat_id, len(pool), offset=307)])


def _first_metric(df: pd.DataFrame, column: str, *, default: float = float("nan")) -> float:
    if df.empty or column not in df.columns:
        return float(default)
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return float(default)
    return float(values.iloc[0])


def _sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _nan_diff(a: float, b: float) -> float:
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")


def _stable_index(sequence_id: int, target_position: int, repeat_id: int, length: int, *, offset: int) -> int:
    if length <= 0:
        raise ValueError("Stable index requires a positive length.")
    value = (
        int(sequence_id) * 10_007
        + int(target_position) * 101
        + int(repeat_id) * 37
        + int(offset)
    )
    return int(value % int(length))


def _int_list(value: Any) -> list[int]:
    return [int(part) for part in str(value).split(";") if str(part).strip()]


def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [str(col) for col in columns if str(col) not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
