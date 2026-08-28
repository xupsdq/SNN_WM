from __future__ import annotations

import sys as _early_sys

_early_args = _early_sys.argv[1:]
if __name__ == "__main__" and (
    "--task=final-statistics" in _early_args
    or any(
        value == "--task"
        and index + 1 < len(_early_args)
        and _early_args[index + 1] == "final-statistics"
        for index, value in enumerate(_early_args)
    )
):
    from src.experiments.paper_figures.final_six.pipeline import (
        canonical_runner_main as _final_statistics_main,
    )

    raise SystemExit(_final_statistics_main("fig2", _early_args))

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS
from src.config.units import ms
from src.core.network import SDNN_Network
from src.data.encoding import DoGSpikeEncoder
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.paper_figures.common.bundle_io import (
    prepare_seed_dirs,
    relative_to_root,
    resolve_seed_dir,
    save_csv_with_registry,
)
from src.experiments.paper_figures.fig2.artifacts import (
    CompletionDelayBoundaryBank,
    cache_key_matches,
    default_artifact_root,
    load_completion_boundary_bank_artifact,
    load_completion_delay_mask_specs_artifact,
    load_crossfit_split_specs_artifact,
    load_crossfit_null_specs_artifact,
    load_pair_trial_specs_artifact,
    load_partial_cue_mask_specs_artifact,
    load_state_bank_artifact,
    save_completion_boundary_bank_artifact,
    save_completion_delay_mask_specs_artifact,
    save_crossfit_split_specs_artifact,
    save_crossfit_null_specs_artifact,
    save_pair_trial_specs_artifact,
    save_partial_cue_mask_specs_artifact,
    save_state_bank_artifact,
    task_artifact_dir,
    validate_cache_key_integrity,
    write_json,
)
from src.experiments.paper_figures.fig2.cache_keys import (
    build_completion_boundary_cache_key,
    build_completion_mask_cache_key,
    build_crossfit_split_cache_key,
    build_crossfit_null_specs_cache_key,
    build_pair_specs_cache_key,
    build_partial_cue_mask_cache_key,
    build_state_bank_cache_key,
    cache_key_digest,
    dataframe_hash,
    model_fingerprint,
    pair_specs_hash,
)
from src.experiments.paper_figures.fig2.constants import FIGURE_ID, NUM_CLASSES, STATE_CONDITIONS
from src.experiments.paper_figures.fig2.output import (
    ms_to_steps,
    prepare_dirs,
    seed_output_dir,
    utc_now,
    write_config_files,
    write_run_log_file,
    write_summary,
)
from src.experiments.paper_figures.fig2.schemas import (
    COMPLETION_CONDITIONS,
    COMPLETION_DELAY_MASK_COLUMNS,
    REUSE_MODES,
    TASK_ALL,
    TASK_BOTH_SCOPE,
    TASK_COMPLETION_DELAY_BOUNDARY_BANK,
    TASK_COMPLETION_DELAY_MASK_SPECS,
    TASK_COMPLETION_DELAY_SWEEP,
    TASK_CROSSFIT_INTERACTION,
    TASK_CROSSFIT_NULL_CALIBRATION,
    TASK_CROSSFIT_NULL_SPECS,
    TASK_CROSSFIT_SPLIT_SPECS,
    TASK_IDS,
    TASK_LINEAR_MIXTURE,
    TASK_MAIN_SCOPE,
    TASK_MORPHOLOGY,
    TASK_NEUTRAL_PING,
    TASK_PAIR_TRIAL_SPECS,
    TASK_PARTIAL_CUE,
    TASK_PARTIAL_CUE_MASK_SPECS,
    TASK_PING_SWEEP,
    TASK_STATE_BANK,
    TASK_SUPPLEMENT,
    TASK_SUPPLEMENT_SCOPE,
    WEAK_PROBE_MASK_COLUMNS,
    normalize_reuse_mode,
)
from src.experiments.paper_figures.fig2.subexperiments.completion_delay_sweep import (
    _make_completion_weak_spikes,
    run_completion_delay_sweep_from_pair_trials,
)
from src.experiments.paper_figures.fig2.subexperiments.crossfit_interaction import (
    build_crossfit_null_specs,
    build_crossfit_split_specs,
    compute_crossfit_interaction_metrics,
    compute_crossfit_null_calibration_metrics,
    validate_crossfit_null_specs,
    validate_crossfit_split_specs,
)
from src.experiments.paper_figures.fig2.subexperiments.linear_mixture import (
    compute_linear_mixture_metrics,
    compute_linear_residual_pair_specificity,
)
from src.experiments.paper_figures.fig2.subexperiments.morphology import (
    compute_dual_retention_metrics,
    compute_pair_level_organization_metrics,
    compute_pair_specificity_metrics,
)
from src.experiments.paper_figures.fig2.subexperiments.neutral_ping import run_neutral_ping_real_rollout_from_state_bank
from src.experiments.paper_figures.fig2.subexperiments.partial_cue import (
    _make_weak_probe_spikes_for_target,
    run_partial_cue_real_rollout_from_state_bank,
)
from src.experiments.paper_figures.fig2.subexperiments.ping_sweep import run_neutral_ping_parameter_sweep
from src.experiments.paper_figures.fig2.subexperiments.supplement import compute_supplementary_metrics
from src.experiments.paper_figures.fig2.subexperiments.trial_specs import build_pair_trial_specs
from src.experiments.paper_figures.fig2.subexperiments.state_bank import run_pair_episode_state_bank
from src.experiments.paper_figures.fig2.subexperiments.helpers import (
    _capture_pair_batch,
    _concat_boundary_states,
    _encode_cached,
    _iter_batches,
    _layer_input_shapes_from_boundary,
    _pair_sampling_audit,
    _progress,
    _trial_condition_audit,
    _weak_probe_mask_row,
)
from src.experiments.paper_figures.fig2.types import ExperimentContext, Fig2Config, PairEpisodeStateBank
from src.experiments.paper_figures.fig2.fixed_b_transition import run_fixed_b_task
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_specs import (
    B_FOLD_MODES,
    CROSSFIT_AXES,
)
from src.experiments.paper_figures.fig2.schemas import (
    FIXED_B_TASK_IDS,
    TASK_FIXED_B_COHORT_AGGREGATE,
    TASK_FIXED_B_SPECS,
)

from src.experiments.paper_figures.run_paper_figures import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH_GLOB,
    DEFAULT_OUTPUT_ROOT,
    discover_checkpoints,
)


PORTABLE_CROSSFIT_PARENT_TASKS = frozenset(
    {
        TASK_CROSSFIT_SPLIT_SPECS,
        TASK_CROSSFIT_INTERACTION,
        TASK_CROSSFIT_NULL_SPECS,
        TASK_CROSSFIT_NULL_CALIBRATION,
    }
)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--task" in raw_argv and "final-statistics" in raw_argv:
        from src.experiments.paper_figures.final_six.pipeline import canonical_runner_main

        return canonical_runner_main("fig2", raw_argv)
    args = _parse_args(argv)
    mode = normalize_reuse_mode(args.reuse_artifacts)
    cfg = _config_from_args(args)
    load_model = not (
        str(args.task)
        in {
            TASK_CROSSFIT_SPLIT_SPECS,
            TASK_CROSSFIT_NULL_SPECS,
            TASK_FIXED_B_SPECS,
            TASK_FIXED_B_COHORT_AGGREGATE,
        }
        or (
            str(args.task) in {TASK_CROSSFIT_INTERACTION, TASK_CROSSFIT_NULL_CALIBRATION}
            and mode == "require"
        )
    )
    ctx = _build_context(cfg, load_model=load_model)
    artifact_root = _artifact_root_from_args(args, ctx.seed_dir)
    run_info = build_run_info(
        experiment_name=f"{FIGURE_ID}.{args.task}",
        output_dir=ctx.seed_dir,
        entry_script="src.experiments.paper_figures.fig2.run_task",
        seed=cfg.network_seed,
        dataset=f"MNIST:{cfg.split}",
        command=" ".join(sys.argv if argv is None else ["run_task", *argv]),
        model_path=cfg.model_path,
        status="running",
    )
    run_info.update(
        {
            "reuse_artifacts": mode,
            "artifact_root": str(artifact_root.resolve()),
            "task": str(args.task),
        }
    )
    write_run_info(ctx.seed_dir / "meta", run_info)
    try:
        write_config_files(ctx)
        if str(args.task) in FIXED_B_TASK_IDS:
            run_fixed_b_task(ctx, task_id=str(args.task), mode=mode, artifact_root=artifact_root)
        else:
            pair_trials = _get_pair_specs(ctx, task_id=str(args.task), mode=mode, artifact_root=artifact_root)
            run_info["pair_specs_hash"] = pair_specs_hash(pair_trials)
            _run_task(ctx, pair_trials, task_id=str(args.task), mode=mode, artifact_root=artifact_root)
        _finalize_bundle(ctx, artifact_root=artifact_root, mode=mode)
        finalize_run_info(ctx.seed_dir / "meta", run_info, status="success")
        return 0
    except Exception:
        finalize_run_info(ctx.seed_dir / "meta", run_info, status="failed")
        raise


def _get_pair_specs(
    ctx: ExperimentContext,
    *,
    task_id: str,
    mode: str,
    artifact_root: Path,
) -> pd.DataFrame:
    task_dir = task_artifact_dir(artifact_root, TASK_PAIR_TRIAL_SPECS)
    expected_key = build_pair_specs_cache_key(ctx.cfg)
    if task_id in PORTABLE_CROSSFIT_PARENT_TASKS and mode in {"auto", "require"}:
        if task_dir.exists():
            parent_key = validate_cache_key_integrity(task_dir, task_id=TASK_PAIR_TRIAL_SPECS)
            artifact = load_pair_trial_specs_artifact(task_dir)
            _validate_portable_pair_parent(ctx, artifact.pair_trials, parent_key)
            _write_pair_specs_to_bundle(ctx, artifact.pair_trials, artifact.candidate_pool)
            _set_artifact_metadata(ctx, "pair_trial_specs", "loaded_portable", task_dir, artifact.digest, parent_key)
            return artifact.pair_trials
        if mode == "require":
            load_pair_trial_specs_artifact(task_dir)
    if mode == "require":
        artifact = load_pair_trial_specs_artifact(task_dir, expected_key=expected_key)
        _write_pair_specs_to_bundle(ctx, artifact.pair_trials, artifact.candidate_pool)
        _set_artifact_metadata(ctx, "pair_trial_specs", "loaded", task_dir, artifact.digest, expected_key)
        return artifact.pair_trials
    if mode == "auto" and cache_key_matches(task_dir, expected_key):
        try:
            artifact = load_pair_trial_specs_artifact(task_dir, expected_key=expected_key)
            _write_pair_specs_to_bundle(ctx, artifact.pair_trials, artifact.candidate_pool)
            _set_artifact_metadata(ctx, "pair_trial_specs", "loaded", task_dir, artifact.digest, expected_key)
            return artifact.pair_trials
        except Exception:
            pass
    pair_trials = build_pair_trial_specs(ctx)
    pool_path = ctx.trial_specs_dir / "pair_candidate_pool.csv"
    if not pool_path.exists():
        raise FileNotFoundError(f"Pair candidate pool was not written by trial spec builder: {pool_path}")
    candidate_pool = pd.read_csv(pool_path)
    artifact = save_pair_trial_specs_artifact(
        task_dir,
        pair_trials=pair_trials,
        candidate_pool=candidate_pool,
        cache_key=expected_key,
    )
    _write_pair_specs_to_bundle(ctx, artifact.pair_trials, artifact.candidate_pool)
    _set_artifact_metadata(ctx, "pair_trial_specs", "built", task_dir, artifact.digest, expected_key)
    ctx.run_log.append(f"{utc_now()} pair_trial_specs source=built artifact={task_dir}")
    return artifact.pair_trials


def _validate_portable_pair_parent(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    parent_key: Mapping[str, Any],
) -> None:
    expected_values = {
        "network_seed": int(ctx.cfg.network_seed),
        "dataset_split": str(ctx.cfg.split),
        "num_pairs": int(ctx.cfg.num_pairs),
        "sample_ms": int(ctx.cfg.sample_ms),
        "delay1_ms": int(ctx.cfg.delay1_ms),
        "second_item_ms": int(ctx.cfg.second_item_ms),
        "delay2_ms": int(ctx.cfg.delay2_ms),
    }
    for name, expected in expected_values.items():
        found = parent_key.get(name)
        if found != expected:
            raise RuntimeError(
                f"Portable pair-spec lineage mismatch for {name}: expected={expected!r}, found={found!r}"
            )
    if len(pair_trials) != int(ctx.cfg.num_pairs):
        raise RuntimeError(
            f"Portable pair-spec row count mismatch: expected={ctx.cfg.num_pairs}, found={len(pair_trials)}"
        )
    required_columns = {"A_image_id", "B_image_id", "A_label", "B_label"}
    missing = sorted(required_columns.difference(pair_trials.columns))
    if missing:
        raise RuntimeError(f"Portable pair-spec validation is missing columns: {missing}")
    expected_labels: dict[int, int] = {}
    for row in pair_trials.itertuples(index=False):
        expected_labels[int(row.A_image_id)] = int(row.A_label)
        expected_labels[int(row.B_image_id)] = int(row.B_label)
    for image_id, expected_label in expected_labels.items():
        found_label = int(ctx.dataset[int(image_id)][1])
        if found_label != expected_label:
            raise RuntimeError(
                f"Local dataset identity mismatch for image {image_id}: "
                f"pair artifact label={expected_label}, local dataset label={found_label}"
            )


def _write_pair_specs_to_bundle(ctx: ExperimentContext, pair_trials: pd.DataFrame, candidate_pool: pd.DataFrame) -> None:
    save_csv_with_registry(ctx, pair_trials, ctx.trial_specs_dir / "pair_trials.csv")
    save_csv_with_registry(ctx, candidate_pool, ctx.trial_specs_dir / "pair_candidate_pool.csv")
    save_csv_with_registry(ctx, _pair_sampling_audit(ctx.cfg.network_seed, pair_trials, candidate_pool), ctx.metrics_dir / "supp_pair_sampling_audit.csv")
    save_csv_with_registry(ctx, _trial_condition_audit(ctx.cfg.network_seed, pair_trials), ctx.metrics_dir / "supp_trial_condition_audit.csv")
    ctx.n_pairs = int(len(pair_trials))
    ctx.completed_modules["pair_trial_specs"] = True


def _run_task(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    *,
    task_id: str,
    mode: str,
    artifact_root: Path,
) -> None:
    if task_id in {TASK_MAIN_SCOPE, TASK_SUPPLEMENT_SCOPE, TASK_BOTH_SCOPE}:
        bank = _get_state_bank(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        _run_morphology(ctx, bank)
        _run_linear_mixture(ctx, bank)
        run_neutral_ping_real_rollout_from_state_bank(ctx, bank)
        if mode != "off":
            setattr(
                ctx,
                "partial_cue_mask_specs",
                _get_partial_cue_mask_specs(ctx, pair_trials, mode=mode, artifact_root=artifact_root),
            )
        run_partial_cue_real_rollout_from_state_bank(ctx, bank)
        if task_id in {TASK_SUPPLEMENT_SCOPE, TASK_BOTH_SCOPE}:
            compute_supplementary_metrics(ctx, bank)
        return
    if task_id == TASK_PAIR_TRIAL_SPECS:
        return
    if task_id == TASK_STATE_BANK:
        _get_state_bank(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        return
    if task_id == TASK_CROSSFIT_SPLIT_SPECS:
        split_specs = _get_crossfit_split_specs(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        _write_crossfit_split_specs_to_bundle(ctx, split_specs)
        return
    if task_id == TASK_CROSSFIT_INTERACTION:
        split_specs = _get_crossfit_split_specs(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        _write_crossfit_split_specs_to_bundle(ctx, split_specs)
        bank = _get_state_bank_for_crossfit(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        compute_crossfit_interaction_metrics(ctx, bank, split_specs)
        return
    if task_id == TASK_CROSSFIT_NULL_SPECS:
        split_specs = _get_crossfit_split_specs(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        null_specs = _get_crossfit_null_specs(
            ctx,
            pair_trials,
            split_specs,
            mode=mode,
            artifact_root=artifact_root,
        )
        _write_crossfit_null_specs_to_bundle(ctx, null_specs)
        return
    if task_id == TASK_CROSSFIT_NULL_CALIBRATION:
        split_specs = _get_crossfit_split_specs(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        null_specs = _get_crossfit_null_specs(
            ctx,
            pair_trials,
            split_specs,
            mode=mode,
            artifact_root=artifact_root,
        )
        _write_crossfit_null_specs_to_bundle(ctx, null_specs)
        bank = _get_state_bank_for_crossfit(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        compute_crossfit_null_calibration_metrics(ctx, bank, split_specs, null_specs)
        return
    if task_id == TASK_MORPHOLOGY:
        _run_morphology(ctx, _get_state_bank(ctx, pair_trials, mode=mode, artifact_root=artifact_root))
        return
    if task_id == TASK_LINEAR_MIXTURE:
        _run_linear_mixture(ctx, _get_state_bank(ctx, pair_trials, mode=mode, artifact_root=artifact_root))
        return
    if task_id == TASK_NEUTRAL_PING:
        run_neutral_ping_real_rollout_from_state_bank(ctx, _get_state_bank(ctx, pair_trials, mode=mode, artifact_root=artifact_root))
        return
    if task_id == TASK_PARTIAL_CUE_MASK_SPECS:
        specs = _get_partial_cue_mask_specs(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        _write_partial_cue_mask_specs_to_bundle(ctx, specs)
        return
    if task_id == TASK_PARTIAL_CUE:
        bank = _get_state_bank(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        if mode != "off":
            setattr(ctx, "partial_cue_mask_specs", _get_partial_cue_mask_specs(ctx, pair_trials, mode=mode, artifact_root=artifact_root))
        run_partial_cue_real_rollout_from_state_bank(ctx, bank)
        return
    if task_id == TASK_PING_SWEEP:
        run_neutral_ping_parameter_sweep(ctx, _get_state_bank(ctx, pair_trials, mode=mode, artifact_root=artifact_root))
        return
    if task_id == TASK_COMPLETION_DELAY_BOUNDARY_BANK:
        _get_completion_boundary_bank(ctx, pair_trials, mode=mode, artifact_root=artifact_root, producer=True)
        return
    if task_id == TASK_COMPLETION_DELAY_MASK_SPECS:
        specs = _get_completion_delay_mask_specs(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        _write_completion_delay_mask_specs_to_bundle(ctx, specs)
        return
    if task_id == TASK_COMPLETION_DELAY_SWEEP:
        if mode == "off":
            run_completion_delay_sweep_from_pair_trials(ctx, pair_trials)
        else:
            setattr(ctx, "completion_delay_boundary_bank", _get_completion_boundary_bank(ctx, pair_trials, mode=mode, artifact_root=artifact_root, producer=False))
            setattr(ctx, "completion_delay_mask_specs", _get_completion_delay_mask_specs(ctx, pair_trials, mode=mode, artifact_root=artifact_root))
            run_completion_delay_sweep_from_pair_trials(ctx, pair_trials)
        return
    if task_id == TASK_SUPPLEMENT:
        compute_supplementary_metrics(ctx, _empty_bank(pair_trials))
        return
    if task_id == TASK_ALL:
        bank = _get_state_bank(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        _run_morphology(ctx, bank)
        _run_linear_mixture(ctx, bank)
        split_specs = _get_crossfit_split_specs(ctx, pair_trials, mode=mode, artifact_root=artifact_root)
        _write_crossfit_split_specs_to_bundle(ctx, split_specs)
        compute_crossfit_interaction_metrics(ctx, bank, split_specs)
        null_specs = _get_crossfit_null_specs(
            ctx,
            pair_trials,
            split_specs,
            mode=mode,
            artifact_root=artifact_root,
        )
        _write_crossfit_null_specs_to_bundle(ctx, null_specs)
        compute_crossfit_null_calibration_metrics(ctx, bank, split_specs, null_specs)
        run_neutral_ping_real_rollout_from_state_bank(ctx, bank)
        if mode != "off":
            setattr(ctx, "partial_cue_mask_specs", _get_partial_cue_mask_specs(ctx, pair_trials, mode=mode, artifact_root=artifact_root))
        run_partial_cue_real_rollout_from_state_bank(ctx, bank)
        run_neutral_ping_parameter_sweep(ctx, bank)
        if mode != "off":
            setattr(ctx, "completion_delay_boundary_bank", _get_completion_boundary_bank(ctx, pair_trials, mode=mode, artifact_root=artifact_root, producer=False))
            setattr(ctx, "completion_delay_mask_specs", _get_completion_delay_mask_specs(ctx, pair_trials, mode=mode, artifact_root=artifact_root))
        run_completion_delay_sweep_from_pair_trials(ctx, pair_trials)
        compute_supplementary_metrics(ctx, bank)
        return
    raise ValueError(f"Unsupported Fig.2 task: {task_id}")


def _run_morphology(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    compute_dual_retention_metrics(ctx, bank)
    compute_pair_specificity_metrics(ctx, bank)
    compute_pair_level_organization_metrics(ctx, bank)
    ctx.completed_modules["morphology"] = True


def _run_linear_mixture(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    compute_linear_mixture_metrics(ctx, bank)
    compute_linear_residual_pair_specificity(ctx, bank)
    ctx.completed_modules["linear_mixture"] = True


def _get_crossfit_split_specs(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
) -> pd.DataFrame:
    task_dir = task_artifact_dir(artifact_root, TASK_CROSSFIT_SPLIT_SPECS)
    parent_digest = str(getattr(ctx, "pair_trial_specs_cache_key_digest", ""))
    if not parent_digest:
        parent_key = validate_cache_key_integrity(
            task_artifact_dir(artifact_root, TASK_PAIR_TRIAL_SPECS),
            task_id=TASK_PAIR_TRIAL_SPECS,
        )
        parent_digest = cache_key_digest(parent_key)
    expected_key = build_crossfit_split_cache_key(
        ctx.cfg,
        pair_hash=pair_specs_hash(pair_trials),
        parent_pair_cache_digest=parent_digest,
    )
    if mode == "require":
        artifact = load_crossfit_split_specs_artifact(task_dir, expected_key=expected_key)
        validate_crossfit_split_specs(
            artifact.table,
            pair_trials,
            network_seed=ctx.cfg.network_seed,
            n_folds=ctx.cfg.crossfit_folds,
        )
        _set_artifact_metadata(ctx, "crossfit_split_specs", "loaded", task_dir, artifact.digest, expected_key)
        ctx.completed_modules["crossfit_split_specs"] = True
        return artifact.table.copy()
    if mode == "auto" and task_dir.exists():
        if cache_key_matches(task_dir, expected_key):
            artifact = load_crossfit_split_specs_artifact(task_dir, expected_key=expected_key)
            validate_crossfit_split_specs(
                artifact.table,
                pair_trials,
                network_seed=ctx.cfg.network_seed,
                n_folds=ctx.cfg.crossfit_folds,
            )
            _set_artifact_metadata(ctx, "crossfit_split_specs", "loaded", task_dir, artifact.digest, expected_key)
            ctx.completed_modules["crossfit_split_specs"] = True
            return artifact.table.copy()
        validate_cache_key_integrity(task_dir, task_id=TASK_CROSSFIT_SPLIT_SPECS)

    split_specs = build_crossfit_split_specs(
        pair_trials,
        network_seed=ctx.cfg.network_seed,
        n_folds=ctx.cfg.crossfit_folds,
    )
    if mode == "off":
        setattr(ctx, "crossfit_split_specs_artifact_source", "built_in_memory")
        setattr(ctx, "crossfit_split_specs_cache_key_digest", cache_key_digest(expected_key))
        ctx.completed_modules["crossfit_split_specs"] = True
        return split_specs
    artifact = save_crossfit_split_specs_artifact(task_dir, split_specs, cache_key=expected_key)
    _set_artifact_metadata(ctx, "crossfit_split_specs", "built", task_dir, artifact.digest, expected_key)
    ctx.completed_modules["crossfit_split_specs"] = True
    return artifact.table.copy()


def _write_crossfit_split_specs_to_bundle(ctx: ExperimentContext, split_specs: pd.DataFrame) -> None:
    save_csv_with_registry(ctx, split_specs, ctx.trial_specs_dir / "crossfit_split_specs.csv")
    ctx.completed_modules["crossfit_split_specs"] = True


def _get_crossfit_null_specs(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    split_specs: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
) -> pd.DataFrame:
    task_dir = task_artifact_dir(artifact_root, TASK_CROSSFIT_NULL_SPECS)
    if mode == "off":
        split_parent_digest = str(getattr(ctx, "crossfit_split_specs_cache_key_digest", ""))
        if not split_parent_digest:
            raise RuntimeError("In-memory cross-fit split specs are missing their cache-key digest")
    else:
        split_parent_dir = task_artifact_dir(artifact_root, TASK_CROSSFIT_SPLIT_SPECS)
        split_parent_key = validate_cache_key_integrity(split_parent_dir, task_id=TASK_CROSSFIT_SPLIT_SPECS)
        split_parent_digest = cache_key_digest(split_parent_key)
    expected_key = build_crossfit_null_specs_cache_key(
        ctx.cfg,
        pair_hash=pair_specs_hash(pair_trials),
        split_specs_hash=dataframe_hash(split_specs),
        parent_split_cache_digest=split_parent_digest,
    )
    if mode == "require":
        artifact = load_crossfit_null_specs_artifact(task_dir, expected_key=expected_key)
        validate_crossfit_null_specs(
            artifact.table,
            network_seed=ctx.cfg.network_seed,
            n_replicates=ctx.cfg.crossfit_null_replicates,
            feature_count=ctx.cfg.crossfit_null_feature_count,
            noise_scale_ratio=ctx.cfg.crossfit_null_noise_scale_ratio,
        )
        _set_artifact_metadata(ctx, "crossfit_null_specs", "loaded", task_dir, artifact.digest, expected_key)
        ctx.completed_modules["crossfit_null_specs"] = True
        return artifact.table.copy()
    if mode == "auto" and task_dir.exists():
        if cache_key_matches(task_dir, expected_key):
            artifact = load_crossfit_null_specs_artifact(task_dir, expected_key=expected_key)
            validate_crossfit_null_specs(
                artifact.table,
                network_seed=ctx.cfg.network_seed,
                n_replicates=ctx.cfg.crossfit_null_replicates,
                feature_count=ctx.cfg.crossfit_null_feature_count,
                noise_scale_ratio=ctx.cfg.crossfit_null_noise_scale_ratio,
            )
            _set_artifact_metadata(ctx, "crossfit_null_specs", "loaded", task_dir, artifact.digest, expected_key)
            ctx.completed_modules["crossfit_null_specs"] = True
            return artifact.table.copy()
        validate_cache_key_integrity(task_dir, task_id=TASK_CROSSFIT_NULL_SPECS)

    table = build_crossfit_null_specs(
        network_seed=ctx.cfg.network_seed,
        n_replicates=ctx.cfg.crossfit_null_replicates,
        feature_count=ctx.cfg.crossfit_null_feature_count,
        noise_scale_ratio=ctx.cfg.crossfit_null_noise_scale_ratio,
    )
    if mode == "off":
        setattr(ctx, "crossfit_null_specs_artifact_source", "built_in_memory")
        setattr(ctx, "crossfit_null_specs_cache_key_digest", cache_key_digest(expected_key))
        ctx.completed_modules["crossfit_null_specs"] = True
        return table
    artifact = save_crossfit_null_specs_artifact(task_dir, table, cache_key=expected_key)
    _set_artifact_metadata(ctx, "crossfit_null_specs", "built", task_dir, artifact.digest, expected_key)
    ctx.completed_modules["crossfit_null_specs"] = True
    return artifact.table.copy()


def _write_crossfit_null_specs_to_bundle(ctx: ExperimentContext, null_specs: pd.DataFrame) -> None:
    save_csv_with_registry(ctx, null_specs, ctx.trial_specs_dir / "crossfit_null_specs.csv")
    ctx.completed_modules["crossfit_null_specs"] = True


def _get_state_bank_for_crossfit(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
) -> PairEpisodeStateBank:
    task_dir = task_artifact_dir(artifact_root, TASK_STATE_BANK)
    if mode in {"auto", "require"} and task_dir.exists():
        parent_key = validate_cache_key_integrity(task_dir, task_id=TASK_STATE_BANK)
        _validate_portable_state_parent(ctx, pair_trials, parent_key)
        artifact = load_state_bank_artifact(task_dir, pair_trials=pair_trials)
        _set_artifact_metadata(ctx, "state_bank", "loaded_portable", task_dir, "", parent_key)
        ctx.completed_modules["state_bank"] = True
        return artifact.bank
    if mode == "require":
        load_state_bank_artifact(task_dir, pair_trials=pair_trials)
    return _get_state_bank(ctx, pair_trials, mode=mode, artifact_root=artifact_root)


def _validate_portable_state_parent(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    parent_key: Mapping[str, Any],
) -> None:
    expected_values = {
        "network_seed": int(ctx.cfg.network_seed),
        "dataset_split": str(ctx.cfg.split),
        "pair_specs_hash": pair_specs_hash(pair_trials),
        "dt": float(ctx.cfg.dt),
        "sample_ms": int(ctx.cfg.sample_ms),
        "delay1_ms": int(ctx.cfg.delay1_ms),
        "second_item_ms": int(ctx.cfg.second_item_ms),
        "delay2_ms": int(ctx.cfg.delay2_ms),
    }
    for name, expected in expected_values.items():
        found = parent_key.get(name)
        if found != expected:
            raise RuntimeError(
                f"Portable state-bank lineage mismatch for {name}: expected={expected!r}, found={found!r}"
            )
    extra = parent_key.get("extra")
    if not isinstance(extra, Mapping):
        raise RuntimeError("Portable state-bank lineage is missing its extra cache-key payload")
    if set(str(value) for value in extra.get("state_conditions", [])) != {"S0", "S_A", "S_B", "S_AB"}:
        raise RuntimeError(f"Portable state-bank conditions are invalid: {extra.get('state_conditions')!r}")
    if set(str(value) for value in extra.get("state_variables", [])) != {"u", "x", "g"}:
        raise RuntimeError(f"Portable state-bank variables are invalid: {extra.get('state_variables')!r}")
    upstream_model = parent_key.get("model")
    if not isinstance(upstream_model, Mapping):
        raise RuntimeError("Portable state-bank lineage is missing its model fingerprint")
    local_model = model_fingerprint(ctx.cfg.model_path)
    for name in ("sha256", "size_bytes"):
        if upstream_model.get(name) != local_model.get(name):
            raise RuntimeError(
                f"Portable state-bank model fingerprint mismatch for {name}: "
                f"artifact={upstream_model.get(name)!r}, local={local_model.get(name)!r}"
            )


def _get_state_bank(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
) -> PairEpisodeStateBank:
    task_dir = task_artifact_dir(artifact_root, TASK_STATE_BANK)
    expected_key = build_state_bank_cache_key(ctx.cfg, pair_hash=pair_specs_hash(pair_trials))
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_state_bank_artifact(task_dir, expected_key=expected_key, pair_trials=pair_trials)
        _set_artifact_metadata(ctx, "state_bank", "loaded", task_dir, "", expected_key)
        ctx.completed_modules["state_bank"] = True
        return artifact.bank
    if mode == "require":
        artifact = load_state_bank_artifact(task_dir, expected_key=expected_key, pair_trials=pair_trials)
        _set_artifact_metadata(ctx, "state_bank", "loaded", task_dir, "", expected_key)
        ctx.completed_modules["state_bank"] = True
        return artifact.bank
    bank = run_pair_episode_state_bank(ctx, pair_trials)
    if mode != "off":
        save_state_bank_artifact(task_dir, bank, cache_key=expected_key, network_seed=ctx.cfg.network_seed)
    _set_artifact_metadata(ctx, "state_bank", "built", task_dir, "", expected_key)
    return bank


def _get_completion_boundary_bank(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
    producer: bool,
) -> CompletionDelayBoundaryBank:
    task_dir = task_artifact_dir(artifact_root, TASK_COMPLETION_DELAY_BOUNDARY_BANK)
    expected_key = build_completion_boundary_cache_key(ctx.cfg, pair_hash=pair_specs_hash(pair_trials))
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        return load_completion_boundary_bank_artifact(
            task_dir,
            expected_key=expected_key,
            pair_trials=pair_trials,
            expected_delays=tuple(ctx.cfg.completion_delay_sweep_ms),
        )
    if mode == "require":
        return load_completion_boundary_bank_artifact(
            task_dir,
            expected_key=expected_key,
            pair_trials=pair_trials,
            expected_delays=tuple(ctx.cfg.completion_delay_sweep_ms),
        )
    if mode == "off" and not producer:
        raise RuntimeError("Internal error: completion boundary bank requested for reuse-artifacts=off.")
    bank = _build_completion_boundary_bank(ctx, pair_trials)
    artifact = save_completion_boundary_bank_artifact(
        task_dir,
        boundary_states_by_delay=bank.boundary_states_by_delay,
        layer_input_shapes_by_delay=bank.layer_input_shapes_by_delay,
        cache_key=expected_key,
        network_seed=ctx.cfg.network_seed,
        row_count=len(pair_trials),
    )
    ctx.completed_modules["completion_delay_boundary_bank"] = True
    _set_artifact_metadata(ctx, "completion_delay_boundary_bank", "built", task_dir, "", expected_key)
    return artifact


def _build_completion_boundary_bank(ctx: ExperimentContext, pair_trials: pd.DataFrame) -> CompletionDelayBoundaryBank:
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    boundary_states_by_delay: dict[int, dict[str, Mapping[str, Mapping[str, torch.Tensor]]]] = {}
    layer_input_shapes_by_delay: dict[int, dict[str, tuple[int, ...]]] = {}
    for delay2_ms in _progress(
        ctx.cfg.completion_delay_sweep_ms,
        total=len(ctx.cfg.completion_delay_sweep_ms),
        desc="fig2 completion boundary bank",
        enabled=ctx.cfg.show_progress,
    ):
        delay2_steps = ms_to_steps(int(delay2_ms), ctx.cfg.dt)
        collected: dict[str, Mapping[str, Mapping[str, torch.Tensor]]] = {}
        for batch in _iter_batches(pair_trials, ctx.cfg.batch_size):
            a_spikes = _encode_cached(ctx, batch["A_image_id"].to_numpy(), ctx.cfg.sample_steps, cache=encode_cache)
            b_spikes = _encode_cached(ctx, batch["B_image_id"].to_numpy(), ctx.cfg.second_item_steps, cache=encode_cache)
            _batch_bank, batch_boundaries = _capture_pair_batch(ctx, a_spikes, b_spikes, delay2_steps=delay2_steps)
            for condition in COMPLETION_CONDITIONS:
                if condition not in collected:
                    collected[condition] = batch_boundaries[condition]
                else:
                    collected[condition] = _concat_boundary_states(collected[condition], batch_boundaries[condition])
        boundary_states_by_delay[int(delay2_ms)] = collected
        layer_input_shapes_by_delay[int(delay2_ms)] = _layer_input_shapes_from_boundary(collected["S0"])
    return CompletionDelayBoundaryBank(Path(), boundary_states_by_delay, layer_input_shapes_by_delay, pd.DataFrame())


def _get_partial_cue_mask_specs(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
) -> pd.DataFrame:
    task_dir = task_artifact_dir(artifact_root, TASK_PARTIAL_CUE_MASK_SPECS)
    expected_key = build_partial_cue_mask_cache_key(ctx.cfg, pair_hash=pair_specs_hash(pair_trials))
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        try:
            return load_partial_cue_mask_specs_artifact(task_dir, expected_key=expected_key).table.copy()
        except Exception:
            if mode == "require":
                raise
    if mode == "require":
        return load_partial_cue_mask_specs_artifact(task_dir, expected_key=expected_key).table.copy()
    specs = _build_partial_cue_mask_specs(ctx, pair_trials)
    artifact = save_partial_cue_mask_specs_artifact(task_dir, specs, cache_key=expected_key)
    _set_artifact_metadata(ctx, "partial_cue_mask_specs", "built", task_dir, artifact.digest, expected_key)
    return artifact.table.copy()


def _build_partial_cue_mask_specs(ctx: ExperimentContext, pair_trials: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 404)
    rows: list[dict[str, Any]] = []
    mask_id = 0
    full_probe_cache: dict[tuple[int, int], torch.Tensor] = {}
    for _, rec in _progress(pair_trials.iterrows(), total=len(pair_trials), desc="fig2 partial cue mask specs", enabled=ctx.cfg.show_progress):
        pair_id = int(rec["pair_id"])
        labels = {"A": int(rec["A_label"]), "B": int(rec["B_label"])}
        image_ids = {"A": int(rec["A_image_id"]), "B": int(rec["B_image_id"])}
        for target_item in ("A", "B"):
            image_id = image_ids[target_item]
            cache_key = (image_id, int(ctx.cfg.weak_probe_steps))
            if cache_key not in full_probe_cache:
                target_image = ctx.dataset[image_id][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
                full_probe_cache[cache_key] = encode_images(ctx.encoder, target_image, ctx.cfg.weak_probe_steps).to(ctx.device)
            for keep_prob in ctx.cfg.weak_probe_keep_probs:
                for repeat_id in range(int(ctx.cfg.weak_probe_repeats)):
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    _weak_spikes, mask_info = _make_weak_probe_spikes_for_target(
                        ctx,
                        full_probe_cache[cache_key],
                        image_id,
                        target_item,
                        float(keep_prob),
                        mask_seed,
                        len(STATE_CONDITIONS),
                        ctx.cfg.weak_probe_use_same_mask_across_states,
                    )
                    rows.append(
                        _weak_probe_mask_row(
                            ctx,
                            mask_id=mask_id,
                            pair_id=pair_id,
                            target_item=target_item,
                            target_label=labels[target_item],
                            keep_prob=float(keep_prob),
                            repeat_id=repeat_id,
                            mask_seed=mask_seed,
                            mask_info=mask_info,
                        )
                    )
                    mask_id += 1
    return pd.DataFrame(rows, columns=list(WEAK_PROBE_MASK_COLUMNS))


def _get_completion_delay_mask_specs(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
) -> pd.DataFrame:
    task_dir = task_artifact_dir(artifact_root, TASK_COMPLETION_DELAY_MASK_SPECS)
    expected_key = build_completion_mask_cache_key(ctx.cfg, pair_hash=pair_specs_hash(pair_trials))
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        try:
            return load_completion_delay_mask_specs_artifact(task_dir, expected_key=expected_key).table.copy()
        except Exception:
            if mode == "require":
                raise
    if mode == "require":
        return load_completion_delay_mask_specs_artifact(task_dir, expected_key=expected_key).table.copy()
    specs = _build_completion_delay_mask_specs(ctx, pair_trials)
    artifact = save_completion_delay_mask_specs_artifact(task_dir, specs, cache_key=expected_key)
    _set_artifact_metadata(ctx, "completion_delay_mask_specs", "built", task_dir, artifact.digest, expected_key)
    return artifact.table.copy()


def _build_completion_delay_mask_specs(ctx: ExperimentContext, pair_trials: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 909)
    rows: list[dict[str, Any]] = []
    mask_id = 0
    full_probe_cache: dict[int, torch.Tensor] = {}
    for delay2_ms in _progress(
        ctx.cfg.completion_delay_sweep_ms,
        total=len(ctx.cfg.completion_delay_sweep_ms),
        desc="fig2 completion delay mask specs",
        enabled=ctx.cfg.show_progress,
    ):
        for batch in _iter_batches(pair_trials, ctx.cfg.batch_size):
            for _batch_idx, rec in batch.reset_index(drop=True).iterrows():
                image_id = int(rec["A_image_id"])
                if image_id not in full_probe_cache:
                    target_image = ctx.dataset[image_id][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
                    full_probe_cache[image_id] = encode_images(ctx.encoder, target_image, ctx.cfg.weak_probe_steps).to(ctx.device)
                for repeat_id in range(int(ctx.cfg.completion_delay_repeats)):
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    _weak_spikes, mask_info = _make_completion_weak_spikes(
                        ctx,
                        full_probe_cache[image_id],
                        image_id,
                        float(ctx.cfg.completion_delay_keep_prob),
                        mask_seed,
                        len(COMPLETION_CONDITIONS),
                    )
                    rows.append(_completion_mask_row(ctx, rec, delay2_ms=int(delay2_ms), repeat_id=repeat_id, mask_id=mask_id, mask_seed=mask_seed, mask_info=mask_info))
                    mask_id += 1
    return pd.DataFrame(rows, columns=list(COMPLETION_DELAY_MASK_COLUMNS))


def _completion_mask_row(
    ctx: ExperimentContext,
    rec: pd.Series,
    *,
    delay2_ms: int,
    repeat_id: int,
    mask_id: int,
    mask_seed: int,
    mask_info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "mask_id": int(mask_id),
        "pair_id": int(rec["pair_id"]),
        "delay2_ms": int(delay2_ms),
        "target_item": "A",
        "target_label": int(rec["A_label"]),
        "A_label": int(rec["A_label"]),
        "B_label": int(rec["B_label"]),
        "keep_prob": float(ctx.cfg.completion_delay_keep_prob),
        "repeat_id": int(repeat_id),
        "mask_seed": int(mask_seed),
        "mask_space": str(mask_info.get("mask_space", ctx.cfg.weak_probe_mask_space)),
        "same_mask_used_across_states": bool(mask_info.get("same_mask_used_across_states", True)),
        "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
        "weak_probe_noise": float(ctx.cfg.weak_probe_noise),
        "realized_keep_fraction": _maybe_float(mask_info.get("realized_keep_fraction")),
        "full_spike_count": _maybe_float(mask_info.get("full_spike_count")),
        "weak_spike_count": _maybe_float(mask_info.get("weak_spike_count")),
        "weak_spike_fraction": _maybe_float(mask_info.get("weak_spike_fraction")),
        "cue_pixel_count": _maybe_int(mask_info.get("cue_pixel_count")),
        "target_foreground_count": _maybe_int(mask_info.get("target_foreground_count")),
        "cue_fraction_actual": _maybe_float(mask_info.get("cue_fraction_actual")),
        "cue_energy": _maybe_float(mask_info.get("cue_energy")),
        "encoded_spike_count": _maybe_float(mask_info.get("encoded_spike_count")),
    }


def _write_partial_cue_mask_specs_to_bundle(ctx: ExperimentContext, specs: pd.DataFrame) -> None:
    save_csv_with_registry(ctx, specs.loc[:, list(WEAK_PROBE_MASK_COLUMNS)].copy(), ctx.trial_specs_dir / "weak_probe_masks.csv")
    ctx.completed_modules["partial_cue_mask_specs"] = True


def _write_completion_delay_mask_specs_to_bundle(ctx: ExperimentContext, specs: pd.DataFrame) -> None:
    save_csv_with_registry(ctx, specs.loc[:, list(COMPLETION_DELAY_MASK_COLUMNS)].copy(), ctx.trial_specs_dir / "completion_delay_mask_specs.csv")
    ctx.completed_modules["completion_delay_mask_specs"] = True


def _empty_bank(pair_trials: pd.DataFrame) -> PairEpisodeStateBank:
    return PairEpisodeStateBank(
        pair_trials=pair_trials.reset_index(drop=True).copy(),
        arrays={},
        boundary_states={},
        layer_input_shapes={},
        restore_mode="not_required",
        episode_end_step=0,
    )


def _build_context(cfg: Fig2Config, *, load_model: bool = True) -> ExperimentContext:
    seed_everything(int(cfg.network_seed))
    seed_dir = seed_output_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = prepare_dirs(seed_dir)
    device = resolve_device(cfg.device)
    dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, NUM_CLASSES)
    max_duration = max(cfg.sample_ms, cfg.second_item_ms, cfg.weak_probe_ms, 100)
    warnings: list[str] = []
    if not load_model:
        net = None
        encoder = None
    elif Path(cfg.model_path).exists():
        net, encoder = load_model_and_encoder(cfg.model_path, device=device, dt=cfg.dt, max_duration_ms=max_duration)
    elif cfg.smoke:
        seed_everything(int(cfg.network_seed))
        net = SDNN_Network(device=str(device)).to(device)
        net.eval()
        encoder = DoGSpikeEncoder(dt=cfg.dt, max_duration=max_duration * ms, device=str(device))
        warnings.append(
            "Model checkpoint missing; smoke mode used an untrained repo SDNN_Network instance. "
            "Functional outputs are real network rollouts, but are not manuscript evidence."
        )
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {cfg.model_path}")
    return ExperimentContext(
        cfg=cfg,
        seed_dir=seed_dir,
        config_dir=dirs["config"],
        trial_specs_dir=dirs["trial_specs"],
        raw_dir=dirs["raw"],
        metrics_dir=dirs["metrics"],
        debug_dir=dirs["debug"],
        device=device,
        dataset=dataset,
        class_index=class_index,
        net=net,
        encoder=encoder,
        warnings=warnings,
        output_files={},
        completed_modules={},
        run_log=[
            f"{utc_now()} start {FIGURE_ID} task runner seed={cfg.network_seed} smoke={cfg.smoke} "
            f"model_loaded={bool(load_model)}"
        ],
    )


def _artifact_root_from_args(args: argparse.Namespace, seed_dir: Path) -> Path:
    if not args.artifact_root:
        root = default_artifact_root(seed_dir)
    else:
        value = Path(args.artifact_root)
        root = value if value.is_absolute() else (Path.cwd() / value)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (DEFAULT_PROJECT_DEFAULTS.paths.repo_root / path).resolve()


def _resolve_model_path(model_path: str | None, model_path_glob: str, network_seed: int, *, smoke: bool) -> Path:
    if model_path:
        return _resolve_repo_path(model_path)
    try:
        checkpoints = discover_checkpoints(str(model_path_glob))
    except FileNotFoundError:
        if smoke:
            return _resolve_repo_path("results/missing_fig2_smoke_model.pth")
        raise
    by_seed = {int(item.seed): item.model_path for item in checkpoints}
    if int(network_seed) not in by_seed:
        if smoke:
            return _resolve_repo_path("results/missing_fig2_smoke_model.pth")
        known = ", ".join(str(seed) for seed in sorted(by_seed))
        raise FileNotFoundError(f"No checkpoint for network seed {network_seed} matched --model-path-glob. Known seeds: {known}")
    return by_seed[int(network_seed)]


def _output_root_from_args(args: argparse.Namespace) -> Path:
    if args.output_dir:
        root = _resolve_repo_path(args.output_dir)
        if root.name.startswith("seed_") or root.name == FIGURE_ID:
            return root
        return root / FIGURE_ID
    return _resolve_repo_path(args.output_root)


def _finalize_bundle(ctx: ExperimentContext, *, artifact_root: Path, mode: str) -> None:
    _mark_completed_from_existing_outputs(ctx)
    write_config_files(ctx)
    _refresh_output_file_registry(ctx)
    summary = write_summary(ctx)
    summary.update(
        {
            "reuse_artifacts": str(mode),
            "runtime_artifact_root": str(Path(artifact_root).resolve()),
        }
    )
    write_json(summary, ctx.seed_dir / "summary.json")
    write_run_log_file(ctx)


def _mark_completed_from_existing_outputs(ctx: ExperimentContext) -> None:
    checks = {
        "pair_trial_specs": [ctx.trial_specs_dir / "pair_trials.csv", ctx.trial_specs_dir / "pair_candidate_pool.csv"],
        "state_bank": [ctx.raw_dir / "state_bank_l3.npz", ctx.raw_dir / "state_bank_manifest.csv"],
        "crossfit_split_specs": [ctx.trial_specs_dir / "crossfit_split_specs.csv"],
        "crossfit_null_specs": [ctx.trial_specs_dir / "crossfit_null_specs.csv"],
        "morphology": [
            ctx.metrics_dir / "panel_b_dual_retention_metrics.csv",
            ctx.metrics_dir / "panel_c_pair_specificity_metrics.csv",
            ctx.metrics_dir / "panel_d_pair_level_organization_metrics.csv",
        ],
        "linear_mixture": [
            ctx.metrics_dir / "panel_d_linear_mixture_fit_metrics.csv",
            ctx.metrics_dir / "panel_d_linear_residual_pair_specificity_metrics.csv",
        ],
        "crossfit_interaction": [
            ctx.metrics_dir / "panel_d_crossfit_interaction_network_metrics.csv",
            ctx.metrics_dir / "panel_d_crossfit_interaction_fold_metrics.csv",
            ctx.metrics_dir / "panel_d_crossfit_interaction_pair_metrics.csv",
            ctx.metrics_dir / "supp_crossfit_interaction_coefficients.csv",
            ctx.metrics_dir / "panel_d_crossfit_interaction_analysis_spec.json",
        ],
        "crossfit_null_calibration": [
            ctx.metrics_dir / "supp_crossfit_null_network_metrics.csv",
            ctx.metrics_dir / "supp_crossfit_null_analysis_spec.json",
        ],
        "neutral_ping": [ctx.raw_dir / "panel_e_neutral_ping_trial_readout.csv", ctx.metrics_dir / "panel_e_neutral_ping_metrics.csv"],
        "partial_cue": [ctx.raw_dir / "panel_f_partial_cue_trial_readout.csv", ctx.metrics_dir / "panel_f_partial_cue_metrics.csv"],
        "ping_sweep": [ctx.raw_dir / "supp_ping_sweep_trial_readout.csv", ctx.metrics_dir / "supp_ping_sweep_metrics.csv"],
        "completion_delay_sweep": [
            ctx.raw_dir / "supp_completion_delay_sweep_trial_readout.csv",
            ctx.metrics_dir / "supp_completion_delay_sweep_metrics.csv",
            ctx.metrics_dir / "supp_completion_delay_sweep_contrast.csv",
        ],
        "supplement": [ctx.metrics_dir / "supp_delay_layer_fused_state_metrics.csv"],
    }
    for name, paths in checks.items():
        if all(path.exists() for path in paths):
            ctx.completed_modules[name] = True


def _refresh_output_file_registry(ctx: ExperimentContext) -> None:
    for path in sorted(ctx.seed_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = relative_to_root(path, ctx.seed_dir)
        if rel.startswith("data/intermediates/"):
            continue
        if path.suffix.lower() in {".csv", ".json", ".txt", ".npz"}:
            ctx.output_files[path.stem] = rel


def _set_artifact_metadata(
    ctx: ExperimentContext,
    name: str,
    source: str,
    artifact_dir: Path,
    digest: str,
    cache_key: Mapping[str, Any],
) -> None:
    setattr(ctx, f"{name}_artifact_source", str(source))
    setattr(ctx, f"{name}_artifact_root", str(Path(artifact_dir).resolve()))
    setattr(ctx, f"{name}_artifact_digest", str(digest))
    setattr(ctx, f"{name}_cache_key_digest", cache_key_digest(cache_key))


def _maybe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except TypeError:
        pass
    return float(value)


def _maybe_int(value: Any) -> int | float:
    if value is None:
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except TypeError:
        pass
    return int(value)


def _config_from_args(args: argparse.Namespace) -> Fig2Config:
    smoke = bool(args.smoke)
    keep_probs = tuple(float(v) for v in str(args.weak_probe_keep_probs).split(",") if str(v).strip())
    ping_amp_sweep = tuple(float(v) for v in str(args.ping_amp_sweep).split(",") if str(v).strip())
    ping_ms_sweep = tuple(int(v) for v in str(args.ping_ms_sweep).split(",") if str(v).strip())
    completion_delay_sweep_ms = tuple(int(v) for v in str(args.completion_delay_sweep_ms).split(",") if str(v).strip())
    completion_delay_repeats = int(args.completion_delay_repeats)
    if smoke:
        keep_probs = (0.2, 0.7)
        ping_amp_sweep = (1.0,)
        ping_ms_sweep = ping_ms_sweep[:2]
        completion_delay_sweep_ms = completion_delay_sweep_ms[:2]
        completion_delay_repeats = 1
    delay_grid = tuple(int(v) for v in str(args.delay_layer_grid).split(",") if str(v).strip())
    crossfit_layers = tuple(str(v).strip() for v in str(args.crossfit_layers).split(",") if str(v).strip())
    crossfit_state_variables = tuple(
        str(v).strip() for v in str(args.crossfit_state_variables).split(",") if str(v).strip()
    )
    fixed_b_prefix_depths = tuple(
        int(v) for v in str(args.fixed_b_prefix_depths).split(",") if str(v).strip()
    )
    fixed_b_ridge_alphas = tuple(
        float(v) for v in str(args.fixed_b_ridge_alphas).split(",") if str(v).strip()
    )
    fixed_b_crossfit_axes = tuple(
        str(value).strip()
        for value in str(args.fixed_b_crossfit_axes).split(",")
        if str(value).strip()
    )
    invalid_fixed_b_axes = sorted(
        set(fixed_b_crossfit_axes) - set(CROSSFIT_AXES)
    )
    if not fixed_b_crossfit_axes or invalid_fixed_b_axes:
        raise ValueError(
            f"Unsupported fixed-B crossfit axes: {invalid_fixed_b_axes}"
        )
    fixed_b_b_fold_mode = str(args.fixed_b_b_fold_mode)
    if not fixed_b_prefix_depths or not fixed_b_ridge_alphas:
        raise ValueError("fixed-B prefix depths and ridge alphas cannot be empty")
    fixed_b_history_families = int(args.fixed_b_history_families)
    fixed_b_candidate_families = int(args.fixed_b_candidate_families)
    fixed_b_anchors = int(args.fixed_b_anchors)
    fixed_b_folds = int(args.fixed_b_folds)
    fixed_b_item_ms = int(args.fixed_b_item_ms)
    fixed_b_inter_delay_ms = int(args.fixed_b_inter_delay_ms)
    fixed_b_stimulus_ms = int(args.fixed_b_stimulus_ms)
    fixed_b_post_ms = int(args.fixed_b_post_ms)
    fixed_b_target_components = int(args.fixed_b_target_components)
    fixed_b_null_replicates = int(args.fixed_b_null_replicates)
    fixed_b_source_match_max_smd = float(args.fixed_b_source_match_max_smd)
    fixed_b_protocol_dir = (
        str(_resolve_repo_path(args.fixed_b_protocol_dir))
        if args.fixed_b_protocol_dir
        else ""
    )
    fixed_b_task_state_path = (
        str(_resolve_repo_path(args.fixed_b_task_state))
        if args.fixed_b_task_state
        else ""
    )
    if smoke:
        fixed_b_history_families = 2
        fixed_b_candidate_families = 20
        fixed_b_anchors = (
            20
            if fixed_b_b_fold_mode == "stratified_within_class"
            else 2
        )
        fixed_b_prefix_depths = (1, 2)
        fixed_b_folds = 2
        fixed_b_item_ms = 20
        fixed_b_inter_delay_ms = 20
        fixed_b_stimulus_ms = 20
        fixed_b_post_ms = 20
        fixed_b_target_components = min(fixed_b_target_components, 8)
        fixed_b_null_replicates = min(fixed_b_null_replicates, 3)
        fixed_b_source_match_max_smd = max(fixed_b_source_match_max_smd, 100.0)
    if not crossfit_layers or not crossfit_state_variables:
        raise ValueError("Crossfit analysis requires at least one layer and one state variable")
    task = str(args.task)
    run_all = task == TASK_ALL
    scope_task = task in {TASK_MAIN_SCOPE, TASK_SUPPLEMENT_SCOPE, TASK_BOTH_SCOPE}
    supplement_scope = task in {TASK_SUPPLEMENT_SCOPE, TASK_BOTH_SCOPE}
    model_path = _resolve_model_path(args.model_path, str(args.model_path_glob), int(args.network_seed), smoke=smoke)
    return Fig2Config(
        model_path=str(model_path),
        dataset_root=str(_resolve_repo_path(args.dataset_root)),
        output_root=str(_output_root_from_args(args)),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sample_ms=int(args.sample_ms),
        delay1_ms=int(args.delay1_ms),
        second_item_ms=int(args.second_item_ms),
        delay2_ms=int(args.delay2_ms),
        ping_ms=int(args.ping_ms),
        ping_amp=float(args.ping_amp),
        ping_repeats=1 if smoke else int(args.ping_repeats),
        ping_mode=str(args.ping_mode),
        ping_noise=float(args.ping_noise),
        ping_amp_sweep=ping_amp_sweep,
        ping_ms_sweep=ping_ms_sweep,
        weak_probe_ms=int(args.weak_probe_ms),
        weak_probe_keep_probs=keep_probs,
        weak_probe_repeats=1 if smoke else int(args.weak_probe_repeats),
        weak_probe_mask_space=str(args.weak_probe_mask_space),
        weak_probe_use_same_mask_across_states=bool(args.weak_probe_use_same_mask_across_states),
        weak_probe_scale=float(args.weak_probe_scale),
        weak_probe_noise=float(args.weak_probe_noise),
        weak_probe_metric_mode=str(args.weak_probe_metric_mode),
        foreground_threshold=float(args.foreground_threshold),
        functional_restore_mode=str(args.functional_restore_mode),
        num_pairs=min(int(args.num_pairs), 20) if smoke else int(args.num_pairs),
        batch_size=min(int(args.batch_size), 4) if smoke else int(args.batch_size),
        n_shuffle=4 if smoke else int(args.n_shuffle),
        delay_layer_grid=delay_grid[:2] if smoke else delay_grid,
        linear_mixture_cv_folds=2 if smoke else int(args.linear_mixture_cv_folds),
        crossfit_folds=2 if smoke else int(args.crossfit_folds),
        crossfit_layers=crossfit_layers,
        crossfit_state_variables=crossfit_state_variables,
        crossfit_null_replicates=2 if smoke else int(args.crossfit_null_replicates),
        crossfit_null_feature_count=64 if smoke else int(args.crossfit_null_feature_count),
        crossfit_null_noise_scale_ratio=float(args.crossfit_null_noise_scale_ratio),
        run_state_bank=run_all or scope_task or task == TASK_STATE_BANK,
        run_morphology=run_all or scope_task or task == TASK_MORPHOLOGY,
        run_linear_mixture=run_all or scope_task or task == TASK_LINEAR_MIXTURE,
        run_crossfit_interaction=run_all or task == TASK_CROSSFIT_INTERACTION,
        run_crossfit_null_calibration=run_all or task == TASK_CROSSFIT_NULL_CALIBRATION,
        run_neutral_ping=run_all or scope_task or task == TASK_NEUTRAL_PING,
        run_partial_cue=run_all or scope_task or task == TASK_PARTIAL_CUE,
        run_supplement=run_all or supplement_scope or task == TASK_SUPPLEMENT,
        run_ping_sweep=run_all or task == TASK_PING_SWEEP,
        run_completion_delay_sweep=run_all or task == TASK_COMPLETION_DELAY_SWEEP,
        completion_delay_sweep_ms=completion_delay_sweep_ms,
        completion_delay_keep_prob=float(args.completion_delay_keep_prob),
        completion_delay_repeats=completion_delay_repeats,
        save_debug_figures=bool(args.save_debug_figures),
        save_spike_cache=bool(args.save_spike_cache),
        save_all_layer_state_bank=True,
        save_functional_traces=bool(args.save_functional_traces),
        save_proxy_functional_debug=bool(args.save_proxy_functional_debug),
        show_progress=not bool(args.no_progress),
        use_encode_cache=not bool(args.no_encode_cache),
        enable_partial_cue_batch=bool(args.enable_partial_cue_batch),
        functional_readout_batch_size=max(1, int(args.functional_readout_batch_size)),
        fixed_b_protocol_seed=int(args.fixed_b_protocol_seed),
        fixed_b_candidate_families=fixed_b_candidate_families,
        fixed_b_history_families=fixed_b_history_families,
        fixed_b_anchors=fixed_b_anchors,
        fixed_b_prefix_depths=fixed_b_prefix_depths,
        fixed_b_item_ms=fixed_b_item_ms,
        fixed_b_inter_delay_ms=fixed_b_inter_delay_ms,
        fixed_b_stimulus_ms=fixed_b_stimulus_ms,
        fixed_b_post_ms=fixed_b_post_ms,
        fixed_b_folds=fixed_b_folds,
        fixed_b_early_window_ms=int(args.fixed_b_early_window_ms),
        fixed_b_trace_window_ms=int(args.fixed_b_trace_window_ms),
        fixed_b_ridge_alphas=fixed_b_ridge_alphas,
        fixed_b_target_components=fixed_b_target_components,
        fixed_b_b_fold_mode=fixed_b_b_fold_mode,
        fixed_b_crossfit_axes=fixed_b_crossfit_axes,
        fixed_b_diagnostic_alpha=float(args.fixed_b_diagnostic_alpha),
        fixed_b_null_replicates=fixed_b_null_replicates,
        fixed_b_source_match_max_smd=fixed_b_source_match_max_smd,
        fixed_b_protocol_dir=fixed_b_protocol_dir,
        fixed_b_task_state_path=fixed_b_task_state_path,
        smoke=smoke,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Fig.2 runtime-artifact DAG task.", allow_abbrev=False)
    parser.add_argument("--task", required=True, choices=TASK_IDS)
    parser.add_argument("--reuse-artifacts", default="auto", choices=REUSE_MODES)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-path-glob", default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=str(Path(DEFAULT_OUTPUT_ROOT) / FIGURE_ID))
    parser.add_argument("--output-dir", default=None, help="Batch output root; the Fig.2 experiment id is appended unless a seed or figure root is supplied.")
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default=DEFAULT_PROJECT_DEFAULTS.runtime.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--save-spike-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-encode-cache", action="store_true")
    parser.add_argument("--enable-partial-cue-batch", action="store_true")
    parser.add_argument("--functional-readout-batch-size", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay1-ms", type=int, default=200)
    parser.add_argument("--second-item-ms", type=int, default=200)
    parser.add_argument("--delay2-ms", type=int, default=400)
    parser.add_argument("--ping-ms", type=int, default=30)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--ping-repeats", type=int, default=1)
    parser.add_argument("--ping-mode", default="constant_drive", choices=["constant_drive", "bernoulli_drive"])
    parser.add_argument("--ping-noise", type=float, default=0.0)
    parser.add_argument("--ping-amp-sweep", default="0.5,1.0,1.5")
    parser.add_argument("--ping-ms-sweep", default="10,30,60")
    parser.add_argument("--weak-probe-ms", type=int, default=100)
    parser.add_argument("--weak-probe-keep-probs", default="0.05,0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    parser.add_argument("--weak-probe-repeats", type=int, default=20)
    parser.add_argument("--weak-probe-mask-space", default="encoded_spikes", choices=["encoded_spikes", "image_foreground"])
    parser.add_argument("--weak-probe-use-same-mask-across-states", dest="weak_probe_use_same_mask_across_states", action="store_true", default=True)
    parser.add_argument("--weak-probe-independent-masks-across-states", dest="weak_probe_use_same_mask_across_states", action="store_false")
    parser.add_argument("--weak-probe-scale", type=float, default=0.35)
    parser.add_argument("--weak-probe-noise", type=float, default=0.0)
    parser.add_argument("--weak-probe-metric-mode", default="fig4_compat", choices=["fig4_compat", "legacy"])
    parser.add_argument("--foreground-threshold", type=float, default=0.1)
    parser.add_argument("--functional-restore-mode", choices=["full_boundary", "stsp_only"], default="stsp_only")
    parser.add_argument("--num-pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-shuffle", type=int, default=50)
    parser.add_argument("--delay-layer-grid", default="200,400,800")
    parser.add_argument("--linear-mixture-cv-folds", type=int, default=5)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--crossfit-layers", default="layer3")
    parser.add_argument("--crossfit-state-variables", default="g")
    parser.add_argument("--crossfit-null-replicates", type=int, default=100)
    parser.add_argument("--crossfit-null-feature-count", type=int, default=1024)
    parser.add_argument("--crossfit-null-noise-scale-ratio", type=float, default=1.0)
    parser.add_argument("--completion-delay-sweep-ms", default="100,200,300,400,800,1200")
    parser.add_argument("--completion-delay-keep-prob", type=float, default=0.2)
    parser.add_argument("--completion-delay-repeats", type=int, default=5)
    parser.add_argument("--save-functional-traces", action="store_true")
    parser.add_argument("--save-proxy-functional-debug", action="store_true")
    parser.add_argument("--fixed-b-protocol-seed", type=int, default=20260724)
    parser.add_argument("--fixed-b-candidate-families", type=int, default=50)
    parser.add_argument("--fixed-b-history-families", type=int, default=10)
    parser.add_argument("--fixed-b-protocol-dir", default=None)
    parser.add_argument("--fixed-b-task-state", default=None)
    parser.add_argument("--fixed-b-anchors", type=int, default=50)
    parser.add_argument("--fixed-b-prefix-depths", default="1,5")
    parser.add_argument("--fixed-b-item-ms", type=int, default=200)
    parser.add_argument("--fixed-b-inter-delay-ms", type=int, default=200)
    parser.add_argument("--fixed-b-stimulus-ms", type=int, default=200)
    parser.add_argument("--fixed-b-post-ms", type=int, default=200)
    parser.add_argument("--fixed-b-folds", type=int, default=5)
    parser.add_argument("--fixed-b-early-window-ms", type=int, default=20)
    parser.add_argument("--fixed-b-trace-window-ms", type=int, default=200)
    parser.add_argument("--fixed-b-ridge-alphas", default="0.1,1.0,10.0")
    parser.add_argument("--fixed-b-target-components", type=int, default=32)
    parser.add_argument(
        "--fixed-b-b-fold-mode",
        default="stratified_within_class",
        choices=B_FOLD_MODES,
    )
    parser.add_argument("--fixed-b-crossfit-axes", default="both")
    parser.add_argument("--fixed-b-diagnostic-alpha", type=float, default=10.0)
    parser.add_argument("--fixed-b-null-replicates", type=int, default=19)
    parser.add_argument("--fixed-b-source-match-max-smd", type=float, default=0.10)
    return parser.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
