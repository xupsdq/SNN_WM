from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.experiments.common.input_masks import entry_mask_from_image, overlap_mask as build_overlap_mask
from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig5.constants import (
    PANEL_D_L1_STSP_AUDIT_COLUMNS,
    PERTURBATION_UNIT_COLUMNS,
    PERTURBATION_UX_AUDIT_COLUMNS,
    TRIAL_COLUMNS,
    UNIT_GROUP_COLUMNS,
)
from src.experiments.paper_figures.fig5.subexperiments.helpers import (
    _centered_cosine,
    _image_array,
    _iter_batches,
    _perturbation_unit_rows,
    _progress,
    _run_batch_network_checked,
    _save_panel_a_example,
    _save_probe_trace_manifest,
    _save_trial_mask_npz,
    _unit_group_rows,
)
from src.experiments.paper_figures.fig5.types import BranchTrace, ExperimentContext, LocalSupportCompetitionBank

def build_local_competition_trials(ctx: ExperimentContext) -> pd.DataFrame:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg.network_seed))
    entry_cache: dict[tuple[Any, ...], np.ndarray] = {}
    candidates: list[dict[str, Any]] = []
    rejected_overlap = 0
    rejected_probe_only = 0
    per_label_target = max(1, int(math.ceil(cfg.max_trials / 10)))
    selected_by_label = {i: 0 for i in range(10)}
    image_ids_by_label = {label: np.asarray(ids, dtype=np.int64) for label, ids in ctx.class_index.items()}

    attempts = max(int(cfg.max_trials) * 80, 100)
    for attempt in _progress(range(attempts), total=attempts, desc="fig5 trial sampling", enabled=cfg.show_progress):
        if len(candidates) >= int(cfg.max_trials):
            break
        probe_label = int((attempt + cfg.network_seed) % 10)
        if selected_by_label[probe_label] >= per_label_target and len(candidates) < int(cfg.max_trials) - 10:
            continue
        sample_label_choices = [label for label in range(10) if label != probe_label]
        sample_label = int(rng.choice(sample_label_choices))
        sample_id = int(rng.choice(image_ids_by_label[sample_label]))
        probe_id = int(rng.choice(image_ids_by_label[probe_label]))
        if sample_id == probe_id:
            continue
        sample_img = _image_array(ctx.dataset, sample_id)
        probe_img = _image_array(ctx.dataset, probe_id)
        sample_mask = entry_mask_from_image(
            ctx.dataset[sample_id][0],
            mode=str(cfg.overlap_mask_mode),
            encoder=ctx.encoder,
            steps=int(cfg.sample_steps),
            device=ctx.device,
            foreground_threshold=float(cfg.foreground_threshold),
            cache=entry_cache,
            image_id=sample_id,
        )
        probe_mask = entry_mask_from_image(
            ctx.dataset[probe_id][0],
            mode=str(cfg.overlap_mask_mode),
            encoder=ctx.encoder,
            steps=int(cfg.probe_steps),
            device=ctx.device,
            foreground_threshold=float(cfg.foreground_threshold),
            cache=entry_cache,
            image_id=probe_id,
        )
        overlap_mask = build_overlap_mask(sample_mask, probe_mask)
        probe_only_mask = probe_mask & (~sample_mask)
        overlap_area = int(overlap_mask.sum())
        probe_only_area = int(probe_only_mask.sum())
        if overlap_area < int(cfg.min_overlap_area):
            rejected_overlap += 1
            continue
        if probe_only_area < int(cfg.min_probe_only_area):
            rejected_probe_only += 1
            continue
        dice = float(2.0 * overlap_area / max(1.0, float(sample_mask.sum() + probe_mask.sum())))
        sim = _centered_cosine(sample_img.reshape(-1), probe_img.reshape(-1))
        selected_by_label[probe_label] += 1
        candidates.append(
            {
                "network_seed": int(cfg.network_seed),
                "trial_id": int(len(candidates)),
                "sample_image_id": sample_id,
                "sample_label": sample_label,
                "probe_image_id": probe_id,
                "probe_label": probe_label,
                "sample_foreground_area": int(sample_mask.sum()),
                "probe_foreground_area": int(probe_mask.sum()),
                "sample_entry_area": int(sample_mask.sum()),
                "probe_entry_area": int(probe_mask.sum()),
                "overlap_area": overlap_area,
                "probe_only_area": probe_only_area,
                "overlap_quantile": float("nan"),
                "selected_trial_group": "pending",
                "input_energy_sample": float(sample_img.sum()),
                "input_energy_probe": float(probe_img.sum()),
                "pixel_similarity": float(sim),
                "dice_overlap": float(dice),
                "overlap_mask_mode": str(cfg.overlap_mask_mode),
                "class_pair": f"{sample_label}->{probe_label}",
                "trial_seed": int(rng.integers(0, 2**31 - 1)),
            }
        )

    trials = pd.DataFrame(candidates)
    if trials.empty:
        raise RuntimeError("No Fig.5 local-competition trials passed the overlap/probe-only filters.")
    overlap_values = trials["overlap_area"].rank(method="average", pct=True).to_numpy(dtype=float)
    trials["overlap_quantile"] = overlap_values
    trials["selected_trial_group"] = np.where(
        trials["overlap_quantile"].between(cfg.medium_q_low, cfg.medium_q_high),
        "medium_overlap",
        np.where(trials["overlap_quantile"] > cfg.medium_q_high, "overlap_rich", "accepted_low_medium"),
    )
    trials = trials.sort_values(["selected_trial_group", "probe_label", "trial_id"], ascending=[False, True, True]).head(int(cfg.max_trials)).copy()
    trials["trial_id"] = np.arange(len(trials), dtype=int)
    trials = trials[TRIAL_COLUMNS]
    _save_csv(ctx, trials, ctx.trial_specs_dir / "local_competition_trials.csv")

    audit = pd.DataFrame(
        [
            {
                "network_seed": int(cfg.network_seed),
                "n_candidates": int(len(candidates) + rejected_overlap + rejected_probe_only),
                "n_selected": int(len(trials)),
                "n_rejected_low_overlap": int(rejected_overlap),
                "n_rejected_low_probe_only": int(rejected_probe_only),
                "n_by_probe_label": json.dumps({str(k): int(v) for k, v in trials["probe_label"].value_counts().sort_index().to_dict().items()}, sort_keys=True),
                "n_by_overlap_quantile": json.dumps({str(k): int(v) for k, v in trials["selected_trial_group"].value_counts().sort_index().to_dict().items()}, sort_keys=True),
                "notes": "Deterministic DMS-style sample/probe sampling with nontrivial overlap and probe-only regions.",
            }
        ]
    )
    _save_csv(ctx, audit, ctx.metrics_dir / "supp_trial_condition_audit.csv")
    _save_trial_mask_npz(ctx, trials)
    ctx.completed_modules["trial_sampling"] = True
    return trials

def build_local_support_competition_bank(ctx: ExperimentContext, trials: pd.DataFrame) -> LocalSupportCompetitionBank:
    support_maps: dict[int, np.ndarray] = {}
    branch_traces: dict[int, dict[str, BranchTrace]] = {}
    boundary_states: dict[int, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    perturb_audit_rows: list[dict[str, Any]] = []
    l1_perturb_audit_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    perturb_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    batches = _iter_batches(trials, ctx.cfg.batch_size)
    for batch in _progress(batches, total=math.ceil(len(trials) / ctx.cfg.batch_size), desc="fig5 support batches", enabled=ctx.cfg.show_progress):
        batch_results = _run_batch_network_checked(ctx, batch)
        perturb_audit_rows.extend(list(batch_results.get("perturbation_ux_audit", [])))
        l1_perturb_audit_rows.extend(list(batch_results.get("l1_stsp_perturbation_audit", [])))
        for trial_idx, trial in _progress(batch.reset_index(drop=True).iterrows(), total=len(batch), desc="fig5 batch trials", enabled=ctx.cfg.show_progress):
            trial_id = int(trial["trial_id"])
            support = batch_results["support_maps"][trial_id]
            support_maps[trial_id] = support
            branch_traces[trial_id] = batch_results["branch_traces"][trial_id]
            if trial_id in batch_results["boundary_states"]:
                boundary_states[trial_id] = batch_results["boundary_states"][trial_id]
            group_df = _unit_group_rows(ctx, trial, support)
            unit_rows.extend(group_df.to_dict("records"))
            perturb_df = _perturbation_unit_rows(ctx, trial, support, group_df)
            perturb_rows.extend(perturb_df.to_dict("records"))
            manifest_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": trial_id,
                    "phase": "pre_probe",
                    "condition": "dynamic_intact",
                    "saved_boundary_state": str(trial_id in boundary_states),
                    "saved_probe_trace": str(bool(ctx.cfg.save_full_traces)),
                    "sample_ms": int(ctx.cfg.sample_ms),
                    "delay_ms": int(ctx.cfg.delay_ms),
                    "probe_ms": int(ctx.cfg.probe_ms),
                    "notes": "Boundary computed once per trial and reused for branch probe conditions.",
                }
            )

    unit_groups = pd.DataFrame(unit_rows, columns=UNIT_GROUP_COLUMNS)
    perturb_sets = pd.DataFrame(perturb_rows, columns=PERTURBATION_UNIT_COLUMNS)
    perturb_audit = pd.DataFrame(perturb_audit_rows, columns=PERTURBATION_UX_AUDIT_COLUMNS)
    l1_perturb_audit = pd.DataFrame(l1_perturb_audit_rows, columns=PANEL_D_L1_STSP_AUDIT_COLUMNS)
    _save_csv(ctx, unit_groups, ctx.trial_specs_dir / "unit_group_definitions.csv")
    _save_csv(ctx, perturb_sets, ctx.trial_specs_dir / "perturbation_unit_sets.csv")
    _save_csv(ctx, perturb_audit, ctx.metrics_dir / "supp_perturbation_ux_audit.csv")
    _save_csv(ctx, pd.DataFrame(manifest_rows), ctx.raw_dir / "rollout_manifest.csv")
    _save_probe_trace_manifest(ctx, branch_traces)
    _save_panel_a_example(ctx, trials, support_maps, unit_groups)
    ctx.completed_modules["preprobe_support_bank"] = True
    return LocalSupportCompetitionBank(
        trials=trials,
        support_maps=support_maps,
        branch_traces=branch_traces,
        boundary_states=boundary_states,
        unit_groups=unit_groups,
        perturbation_sets=perturb_sets,
        perturbation_ux_audit=perturb_audit,
        l1_stsp_perturbation_audit=l1_perturb_audit,
    )
