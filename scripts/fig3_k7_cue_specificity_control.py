from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.paper_figures.fig3.artifacts import load_state_bank_artifact
from src.experiments.paper_figures.fig3.cache_keys import sha256_file
from src.experiments.paper_figures.fig3.subexperiments.cue_specificity import (
    build_cue_specificity_specs,
    compute_cue_specificity_tables,
    cue_specificity_scientific_checks,
    run_cue_specificity_readout,
)
from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import concat_named_boundaries, run_probe_readout_from_boundary
from src.experiments.paper_figures.fig3.subexperiments.weak_probe import _make_weak_probe_spikes_encoded_dropout
from src.experiments.paper_figures.fig3.types import ExperimentContext, Fig3Config
from src.plotting.common.io import apply_publication_style, save_figure_all_formats


DEFAULT_FIG3_SEED_ROOT = "results/fig3_full_seed1000_b32_s4/fig3_multiitem_peak_landscape/seed_1000"
DEFAULT_OUTPUT_DIR = "results/fig3_k7_cue_specificity_control"
NUM_CLASSES = 10
CUE_TYPES = ("matched", "mismatched", "unseen")
STATE_SPECS = (("S_final", "sequence_state"), ("S0", "cue_only"))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    fig3_seed_root = _resolve_path(args.fig3_seed_root)
    output_dir = _resolve_path(args.output_dir)
    _guard_output_not_inside_source(output_dir, fig3_seed_root)

    layout = prepare_result_layout(output_dir)
    data_raw_dir = layout.data_dir / "raw"
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    run_info = build_run_info(
        experiment_name="fig3_k7_cue_specificity_control",
        output_dir=layout.root,
        entry_script="scripts/fig3_k7_cue_specificity_control.py",
        seed=None,
        dataset=None,
        command=" ".join(sys.argv if argv is None else ["fig3_k7_cue_specificity_control.py", *argv]),
        model_path=None,
        status="running",
    )
    write_run_info(layout.meta_dir, run_info)

    log_lines = [f"{_now()} start fig3_k7_cue_specificity_control"]
    try:
        result = run_control(args, fig3_seed_root=fig3_seed_root, output_dir=output_dir, layout=layout, log_lines=log_lines)
        run_info.update(
            {
                "seed": int(result["network_seed"]),
                "dataset": str(result["dataset"]),
                "model_path": str(result["model_path"]),
            }
        )
        finalize_run_info(layout.meta_dir, run_info, status="success")
        save_log_lines([*log_lines, f"{_now()} completed status=success"], layout.logs_dir)
        return 0
    except Exception:
        finalize_run_info(layout.meta_dir, run_info, status="failed")
        save_log_lines([*log_lines, f"{_now()} completed status=failed"], layout.logs_dir)
        raise


def run_control(
    args: argparse.Namespace,
    *,
    fig3_seed_root: Path,
    output_dir: Path,
    layout: Any,
    log_lines: list[str],
) -> dict[str, Any]:
    source_hashes_before = _source_artifact_hashes(fig3_seed_root)
    source_config = _load_json(fig3_seed_root / "run_config.json")
    cfg = _build_fig3_config(source_config, args=args, output_dir=output_dir)
    seed_everything(int(cfg.network_seed))
    device = resolve_device(str(args.device))
    dataset = load_mnist_skeleton_dataset(str(cfg.dataset_root), str(cfg.split))
    class_index = build_class_index(dataset, NUM_CLASSES)
    net, encoder = load_model_and_encoder(str(cfg.model_path), device=device, dt=float(cfg.dt), max_duration_ms=max(int(cfg.sample_ms), int(cfg.weak_probe_ms), 100))
    ctx = ExperimentContext(
        cfg=cfg,
        seed_dir=layout.root,
        config_dir=layout.meta_dir,
        trial_specs_dir=layout.data_dir / "trial_specs",
        raw_dir=layout.data_dir / "raw",
        metrics_dir=layout.metrics_dir,
        debug_dir=layout.figures_dir,
        device=device,
        dataset=dataset,
        class_index=class_index,
        net=net,
        encoder=encoder,
        warnings=[],
        output_files={},
        completed_modules={},
        run_log=[],
    )
    ctx.trial_specs_dir.mkdir(parents=True, exist_ok=True)

    sequence_trials = pd.read_csv(fig3_seed_root / "data" / "trial_specs" / "sequence_trials.csv")
    access_jobs = pd.read_csv(fig3_seed_root / "data" / "trial_specs" / "access_job_specs.csv")
    boundary_dir = fig3_seed_root / "data" / "intermediates" / "boundary_state_bank"
    boundary_cache = _load_json(boundary_dir / "cache_key.json")
    expected_key = boundary_cache["cache_key"]
    bank_artifact = load_state_bank_artifact(boundary_dir, expected_key=expected_key, sequence_trials=sequence_trials)
    bank = bank_artifact.bank
    log_lines.append(f"{_now()} loaded boundary_state_bank digest={bank_artifact.digest}")

    jobs = _select_jobs(access_jobs, seq_len=int(args.seq_len), delay_ms=int(args.delay_ms), keep_prob=float(args.keep_prob))
    if args.max_sequences is not None:
        keep_ids = sorted(int(v) for v in jobs["sequence_id"].dropna().unique())[: int(args.max_sequences)]
        jobs = jobs[jobs["sequence_id"].astype(int).isin(keep_ids)].copy()
    if jobs.empty:
        raise ValueError("No matching weak-cue jobs found for requested K/delay/keep-prob.")

    filtered_access_jobs = access_jobs[access_jobs["job_id"].astype(int).isin(jobs["job_id"].astype(int).tolist())].copy()
    cue_specs = build_cue_specificity_specs(
        ctx,
        sequence_trials,
        filtered_access_jobs,
        seq_len=int(args.seq_len),
        delay_ms=int(args.delay_ms),
        keep_prob=float(args.keep_prob),
        cue_types=CUE_TYPES,
    )
    raw = run_cue_specificity_readout(
        ctx,
        bank,
        cue_specs,
        readout_batch_size=int(args.readout_batch_size),
    )
    skip_counts = {
        "duplicate_label_jobs": 0,
        "missing_unseen_jobs": 0,
        "missing_sequence_jobs": 0,
        "skipped_job_count": 0,
    }
    expected_raw_rows = int(len(cue_specs))
    if len(raw) != expected_raw_rows:
        raise RuntimeError(f"Unexpected raw row count: found {len(raw)}, expected {expected_raw_rows}.")

    tables = compute_cue_specificity_tables(raw)
    metrics = tables["cue_specificity_metrics"]
    serial_summary = tables["cue_specificity_serial_summary"]
    contrast_summary = tables["cue_specificity_contrast_summary"]
    scientific = cue_specificity_scientific_checks(metrics)
    figure_paths = _save_figure(metrics, serial_summary, contrast_summary, layout.figures_dir / "fig3_k7_cue_specificity_control")

    raw_path = layout.data_dir / "raw" / "cue_specificity_trial_readout.csv"
    metrics_path = layout.metrics_dir / "cue_specificity_metrics.csv"
    serial_path = layout.metrics_dir / "cue_specificity_serial_summary.csv"
    contrast_path = layout.metrics_dir / "cue_specificity_contrast_summary.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    serial_summary.to_csv(serial_path, index=False, encoding="utf-8")
    contrast_summary.to_csv(contrast_path, index=False, encoding="utf-8")

    source_hashes_after = _source_artifact_hashes(fig3_seed_root)
    source_hashes_unchanged = source_hashes_before == source_hashes_after
    if not source_hashes_unchanged:
        raise RuntimeError("Source Fig.3 boundary artifact hashes changed during standalone control run.")

    run_config = {
        "fig3_seed_root": str(fig3_seed_root),
        "output_dir": str(output_dir),
        "seq_len": int(args.seq_len),
        "delay_ms": int(args.delay_ms),
        "keep_prob": float(args.keep_prob),
        "device": str(args.device),
        "max_sequences": None if args.max_sequences is None else int(args.max_sequences),
        "readout_batch_size": int(args.readout_batch_size),
        "cue_types": list(CUE_TYPES),
        "state_conditions": [item[0] for item in STATE_SPECS],
        "source_run_config": source_config,
    }
    save_run_config(run_config, layout.root)

    summary = {
        "experiment": "fig3_k7_cue_specificity_control",
        "status": "success",
        "network_seed": int(cfg.network_seed),
        "dataset": f"MNIST:{cfg.split}",
        "model_path": str(cfg.model_path),
        "fig3_seed_root": str(fig3_seed_root),
        "seq_len": int(args.seq_len),
        "delay_ms": int(args.delay_ms),
        "keep_prob": float(args.keep_prob),
        "n_jobs": int(len(jobs)),
        "n_sequences": int(jobs["sequence_id"].nunique()),
        "raw_rows": int(len(raw)),
        "expected_raw_rows": int(expected_raw_rows),
        "skip_counts": skip_counts,
        "scientific_checks": scientific,
        "source_artifact_hashes_before": source_hashes_before,
        "source_artifact_hashes_after": source_hashes_after,
        "source_artifact_hashes_unchanged": bool(source_hashes_unchanged),
        "outputs": {
            "raw": _rel(raw_path, layout.root),
            "metrics": _rel(metrics_path, layout.root),
            "serial_summary": _rel(serial_path, layout.root),
            "contrast_summary": _rel(contrast_path, layout.root),
            "figures": {ext: _rel(Path(path), layout.root) for ext, path in figure_paths.items()},
        },
        "evidence_status": "single_seed_diagnostic_control_not_manuscript_final",
    }
    save_summary_json(summary, layout.root)
    _write_artifact_manifest(layout.root, summary["outputs"])
    log_lines.append(f"{_now()} wrote raw_rows={len(raw)} expected_raw_rows={expected_raw_rows}")
    return {
        "network_seed": int(cfg.network_seed),
        "dataset": f"MNIST:{cfg.split}",
        "model_path": str(cfg.model_path),
    }


def _run_trials(
    ctx: ExperimentContext,
    bank: Any,
    jobs: pd.DataFrame,
    sequence_lookup: Mapping[int, dict[str, Any]],
    class_index: Mapping[int, Sequence[int]],
    *,
    seq_len: int,
    delay_ms: int,
    readout_batch_size: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    raw_rows: list[dict[str, Any]] = []
    encoded_cache: dict[int, torch.Tensor] = {}
    skip_counts = {
        "duplicate_label_jobs": 0,
        "missing_unseen_jobs": 0,
        "missing_sequence_jobs": 0,
        "skipped_job_count": 0,
    }
    condition_id = f"K{int(seq_len)}_D{int(delay_ms)}"
    sorted_jobs = jobs.sort_values(["sequence_id", "target_position", "repeat_id", "job_id"])
    for seq_id_value, seq_jobs in sorted_jobs.groupby("sequence_id", sort=True):
        seq_id = int(seq_id_value)
        if seq_id not in sequence_lookup:
            skip_counts["missing_sequence_jobs"] += int(len(seq_jobs))
            skip_counts["skipped_job_count"] += int(len(seq_jobs))
            continue
        seq = sequence_lookup[seq_id]
        labels = list(seq["labels"])
        image_ids = list(seq["image_ids"])
        if len(set(labels)) != len(labels):
            skip_counts["duplicate_label_jobs"] += int(len(seq_jobs))
            skip_counts["skipped_job_count"] += int(len(seq_jobs))
            continue
        unseen_labels = [label for label in range(NUM_CLASSES) if label not in set(labels)]
        if not unseen_labels:
            skip_counts["missing_unseen_jobs"] += int(len(seq_jobs))
            skip_counts["skipped_job_count"] += int(len(seq_jobs))
            continue

        state_boundaries = [
            bank.boundary_for(seq_id, state, condition_id=condition_id, delay_ms=delay_ms)
            for state, _ in STATE_SPECS
        ]
        pending_boundaries: list[Mapping[str, Mapping[str, torch.Tensor]]] = []
        pending_spikes: list[torch.Tensor] = []
        pending_rows: list[dict[str, Any]] = []
        for _, job in seq_jobs.iterrows():
            target_position = int(job["target_position"])
            target_idx = target_position - 1
            if target_idx < 0 or target_idx >= len(image_ids):
                skip_counts["missing_sequence_jobs"] += 1
                skip_counts["skipped_job_count"] += 1
                continue
            target_label = int(labels[target_idx])
            cue_specs = _cue_specs_for_job(
                seq_id=seq_id,
                target_position=target_position,
                repeat_id=int(job["repeat_id"]),
                image_ids=image_ids,
                labels=labels,
                unseen_labels=unseen_labels,
                class_index=class_index,
            )
            if cue_specs is None:
                skip_counts["missing_unseen_jobs"] += 1
                skip_counts["skipped_job_count"] += 1
                continue
            for cue_spec in cue_specs:
                full_spikes = _encoded_image(ctx, int(cue_spec["cue_image_id"]), encoded_cache)
                weak_spikes, mask_info = _make_weak_probe_spikes_encoded_dropout(
                    full_spikes,
                    float(job["keep_prob"]),
                    seed=int(job["mask_seed"]),
                    same_mask_count=len(STATE_SPECS),
                    use_same_mask_across_states=True,
                    device=ctx.device,
                )
                for row_idx, (state_condition, memory_condition) in enumerate(STATE_SPECS):
                    pending_boundaries.append(state_boundaries[row_idx])
                    pending_spikes.append(weak_spikes[row_idx : row_idx + 1])
                    pending_rows.append(
                        _base_raw_row(
                            ctx,
                            job=job,
                            condition_id=condition_id,
                            seq_id=seq_id,
                            seq_len=seq_len,
                            delay_ms=delay_ms,
                            target_position=target_position,
                            target_label=target_label,
                            cue_spec=cue_spec,
                            state_condition=state_condition,
                            memory_condition=memory_condition,
                            image_ids=image_ids,
                            labels=labels,
                            unseen_labels=unseen_labels,
                            mask_info=mask_info,
                        )
                    )
                if len(pending_rows) >= int(readout_batch_size):
                    raw_rows.extend(_flush_readout_batch(ctx, pending_boundaries, pending_spikes, pending_rows, labels))
                    pending_boundaries = []
                    pending_spikes = []
                    pending_rows = []
        if pending_rows:
            raw_rows.extend(_flush_readout_batch(ctx, pending_boundaries, pending_spikes, pending_rows, labels))
    return pd.DataFrame(raw_rows), skip_counts


def _cue_specs_for_job(
    *,
    seq_id: int,
    target_position: int,
    repeat_id: int,
    image_ids: Sequence[int],
    labels: Sequence[int],
    unseen_labels: Sequence[int],
    class_index: Mapping[int, Sequence[int]],
) -> list[dict[str, Any]] | None:
    target_idx = int(target_position) - 1
    target_label = int(labels[target_idx])
    mismatch_idx = (target_idx + 1) % len(image_ids)
    unseen_label = int(unseen_labels[_stable_index(seq_id, target_position, repeat_id, len(unseen_labels), offset=91)])
    unseen_pool = list(class_index.get(unseen_label, ()))
    if not unseen_pool:
        return None
    return [
        {
            "cue_type": "matched",
            "cue_position": int(target_position),
            "cue_image_id": int(image_ids[target_idx]),
            "cue_label": target_label,
        },
        {
            "cue_type": "mismatched",
            "cue_position": int(mismatch_idx + 1),
            "cue_image_id": int(image_ids[mismatch_idx]),
            "cue_label": int(labels[mismatch_idx]),
        },
        {
            "cue_type": "unseen",
            "cue_position": 0,
            "cue_image_id": int(unseen_pool[_stable_index(seq_id, target_position, repeat_id, len(unseen_pool), offset=193)]),
            "cue_label": unseen_label,
        },
    ]


def _base_raw_row(
    ctx: ExperimentContext,
    *,
    job: pd.Series,
    condition_id: str,
    seq_id: int,
    seq_len: int,
    delay_ms: int,
    target_position: int,
    target_label: int,
    cue_spec: Mapping[str, Any],
    state_condition: str,
    memory_condition: str,
    image_ids: Sequence[int],
    labels: Sequence[int],
    unseen_labels: Sequence[int],
    mask_info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "job_id": int(job["job_id"]),
        "condition_id": condition_id,
        "sequence_id": int(seq_id),
        "seq_len": int(seq_len),
        "delay_ms": int(delay_ms),
        "target_position": int(target_position),
        "target_image_id": int(job["target_image_id"]),
        "target_label": int(target_label),
        "cue_type": str(cue_spec["cue_type"]),
        "cue_position": int(cue_spec["cue_position"]),
        "cue_image_id": int(cue_spec["cue_image_id"]),
        "cue_label": int(cue_spec["cue_label"]),
        "keep_prob": float(job["keep_prob"]),
        "repeat_id": int(job["repeat_id"]),
        "mask_seed": int(job["mask_seed"]),
        "state_condition": str(state_condition),
        "memory_condition": str(memory_condition),
        "ordered_item_ids": ";".join(str(v) for v in image_ids),
        "ordered_item_labels": ";".join(str(v) for v in labels),
        "unseen_labels": ";".join(str(v) for v in unseen_labels),
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
    labels: Sequence[int],
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


def _compute_metrics(raw: pd.DataFrame) -> pd.DataFrame:
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
    columns = [*group_cols, *metric_cols.keys(), "n_trials"]
    return pd.DataFrame(rows, columns=columns)


def _serial_summary(metrics: pd.DataFrame) -> pd.DataFrame:
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


def _contrast_summary(serial_summary: pd.DataFrame) -> pd.DataFrame:
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
                    "matched_minus_unseen": matched - unseen if np.isfinite(matched) and np.isfinite(unseen) else float("nan"),
                    "matched_minus_mismatched": matched - mismatched if np.isfinite(matched) and np.isfinite(mismatched) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _scientific_checks(metrics: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    s_final = metrics[metrics["state_condition"].astype(str).eq("S_final")]
    s0 = metrics[metrics["state_condition"].astype(str).eq("S0")]
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
    return out


def _save_figure(
    metrics: pd.DataFrame,
    serial_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    base_path: Path,
) -> dict[str, str]:
    apply_publication_style()
    colors = {"matched": "#0072B2", "mismatched": "#D55E00", "unseen": "#6A6A6A"}
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), constrained_layout=True)

    ax = axes[0]
    s_final = serial_summary[serial_summary["state_condition"].astype(str).eq("S_final")]
    for cue in CUE_TYPES:
        part = s_final[s_final["cue_type"].astype(str).eq(cue)].sort_values("target_position")
        if part.empty:
            continue
        x = pd.to_numeric(part["target_position"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(part["P_target_mean"], errors="coerce").to_numpy(dtype=float)
        sem = pd.to_numeric(part["P_target_sem"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.4, color=colors[cue], label=cue)
        ax.fill_between(x, y - sem, y + sem, color=colors[cue], alpha=0.14, linewidth=0)
    ax.set_title("S_final target readout")
    ax.set_xlabel("Serial position")
    ax.set_ylabel("P(target)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8)
    _tidy(ax)

    ax = axes[1]
    contrast = contrast_summary[contrast_summary["state_condition"].astype(str).eq("S_final")].sort_values("target_position")
    if not contrast.empty:
        x = pd.to_numeric(contrast["target_position"], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, contrast["matched_minus_unseen"].to_numpy(dtype=float), marker="o", color="#0072B2", linewidth=1.4, label="matched - unseen")
        ax.plot(x, contrast["matched_minus_mismatched"].to_numpy(dtype=float), marker="s", color="#009E73", linewidth=1.4, label="matched - mismatched")
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_title("Cue specificity")
    ax.set_xlabel("Serial position")
    ax.set_ylabel("Delta P(target)")
    ax.set_ylim(-0.25, 1.02)
    ax.legend(frameon=False, fontsize=8)
    _tidy(ax)

    ax = axes[2]
    generic = metrics[metrics["cue_type"].astype(str).isin(["mismatched", "unseen"])].copy()
    generic = generic.groupby(["cue_type", "state_condition"], sort=True)["P_seen_item"].mean().reset_index()
    cue_order = ["mismatched", "unseen"]
    state_order = ["S0", "S_final"]
    x = np.arange(len(cue_order), dtype=float)
    width = 0.34
    for idx, state in enumerate(state_order):
        vals = []
        for cue in cue_order:
            part = generic[generic["cue_type"].astype(str).eq(cue) & generic["state_condition"].astype(str).eq(state)]
            vals.append(float(part["P_seen_item"].iloc[0]) if not part.empty else np.nan)
        offset = (-0.5 + idx) * width
        ax.bar(x + offset, vals, width=width, label=state, color="#B0B0B0" if state == "S0" else "#0072B2", edgecolor="white", linewidth=0.5)
    ax.set_title("Generic seen-item readout")
    ax.set_xticks(x, cue_order)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("P(seen item)")
    ax.legend(frameon=False, fontsize=8)
    _tidy(ax)

    return save_figure_all_formats(fig, base_path)


def _select_jobs(access_jobs: pd.DataFrame, *, seq_len: int, delay_ms: int, keep_prob: float) -> pd.DataFrame:
    jobs = access_jobs[
        access_jobs["job_type"].astype(str).eq("weak_cue")
        & access_jobs["seq_len"].astype(int).eq(int(seq_len))
        & access_jobs["delay_ms"].astype(int).eq(int(delay_ms))
        & np.isclose(pd.to_numeric(access_jobs["keep_prob"], errors="coerce").to_numpy(dtype=float), float(keep_prob))
    ].copy()
    return jobs.reset_index(drop=True)


def _sequence_lookup(sequence_trials: pd.DataFrame, *, seq_len: int) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    use = sequence_trials[sequence_trials["seq_len"].astype(int).eq(int(seq_len))].copy()
    for seq_id, part in use.groupby("sequence_id", sort=True):
        ordered = part.sort_values("stage_k")
        out[int(seq_id)] = {
            "image_ids": [int(v) for v in ordered["item_image_id"].tolist()],
            "labels": [int(v) for v in ordered["item_label"].tolist()],
        }
    return out


def _encoded_image(ctx: ExperimentContext, image_id: int, cache: dict[int, torch.Tensor]) -> torch.Tensor:
    image_id = int(image_id)
    if image_id not in cache:
        image = ctx.dataset[image_id][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
        cache[image_id] = encode_images(ctx.encoder, image, ctx.cfg.weak_probe_steps).to(ctx.device)
    return cache[image_id]


def _build_fig3_config(source_config: Mapping[str, Any], *, args: argparse.Namespace, output_dir: Path) -> Fig3Config:
    kwargs: dict[str, Any] = {}
    field_by_name = {field.name: field for field in fields(Fig3Config)}
    for name, field in field_by_name.items():
        if name not in source_config:
            continue
        value = source_config[name]
        if isinstance(field.default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    kwargs["output_root"] = str(output_dir)
    kwargs["device"] = str(args.device)
    kwargs["sequence_lengths"] = (int(args.seq_len),)
    kwargs["boundary_sequence_lengths"] = (int(args.seq_len),)
    kwargs["boundary_delay_grid_ms"] = (int(args.delay_ms),)
    kwargs["weak_cue_main_keep_prob"] = float(args.keep_prob)
    kwargs["show_progress"] = False
    return Fig3Config(**kwargs)


def _source_artifact_hashes(fig3_seed_root: Path) -> dict[str, str]:
    boundary_dir = fig3_seed_root / "data" / "intermediates" / "boundary_state_bank"
    paths = {
        "boundary_state_bank/cache_key.json": boundary_dir / "cache_key.json",
        "boundary_state_bank/boundary_manifest.csv": boundary_dir / "boundary_manifest.csv",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _write_artifact_manifest(root: Path, outputs: Mapping[str, Any]) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(_rel(path, root))
    payload = {
        "experiment_id": "fig3_k7_cue_specificity_control",
        "created_at": _now(),
        "outputs": outputs,
        "files": files,
    }
    (root / "artifact_manifest.json").write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _mean_by_cue(df: pd.DataFrame, column: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for cue, part in df.groupby("cue_type", sort=True):
        out[str(cue)] = float(pd.to_numeric(part[column], errors="coerce").mean())
    return out


def _sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(values.size))


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _nan_diff(a: float, b: float) -> float:
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")


def _stable_index(sequence_id: int, target_position: int, repeat_id: int, length: int, *, offset: int) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    value = (
        int(sequence_id) * 10_007
        + int(target_position) * 101
        + int(repeat_id) * 37
        + int(offset)
    )
    return int(np.mod(value, int(length)))


def _tidy(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.title.set_size(10)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Fig.3 K7 matched-vs-unseen cue specificity control.")
    parser.add_argument("--fig3-seed-root", default=DEFAULT_FIG3_SEED_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-len", type=int, default=7)
    parser.add_argument("--delay-ms", type=int, default=400)
    parser.add_argument("--keep-prob", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--readout-batch-size", type=int, default=6)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _guard_output_not_inside_source(output_dir: Path, source_root: Path) -> None:
    try:
        output_dir.relative_to(source_root)
    except ValueError:
        return
    raise ValueError(f"Refusing to write output inside source Fig.3 seed root: output_dir={output_dir}, source_root={source_root}")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
