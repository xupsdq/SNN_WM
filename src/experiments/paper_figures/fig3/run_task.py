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

    raise SystemExit(_final_statistics_main("fig3", _early_args))

import argparse
import shutil
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
from src.experiments.common.dataset import build_class_index
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.paper_figures.common.bundle_io import (
    json_safe,
    prepare_seed_dirs,
    relative_to_root,
    resolve_seed_dir,
    save_csv_with_registry,
    write_json_file,
)
from src.experiments.paper_figures.fig3.artifacts import (
    StateBankArtifact,
    TableBundleArtifact,
    cache_key_matches,
    default_artifact_root,
    load_sequence_specs_artifact,
    load_state_bank_artifact,
    load_table_bundle_artifact,
    save_sequence_specs_artifact,
    save_state_bank_artifact,
    save_table_bundle_artifact,
    task_artifact_dir,
    write_json,
)
from src.experiments.paper_figures.fig3.cache_keys import (
    build_access_job_specs_cache_key,
    build_boundary_condition_specs_cache_key,
    build_boundary_state_bank_cache_key,
    build_boundary_summary_cache_key,
    build_cue_specificity_access_cache_key,
    build_cue_specificity_specs_cache_key,
    build_formation_intervention_specs_cache_key,
    build_formation_necessity_cache_key,
    build_morphology_decomposition_cache_key,
    build_morphology_function_coupling_cache_key,
    build_neutral_ping_access_cache_key,
    build_sequence_specs_cache_key,
    build_state_bank_cache_key,
    build_weak_cue_access_cache_key,
    cache_key_digest,
    sequence_specs_hash,
    table_digest,
)
from src.experiments.paper_figures.fig3.constants import FIGURE_ID, NUM_CLASSES
from src.experiments.paper_figures.fig3.schemas import (
    BOUNDARY_SUMMARY_REQUIRED_COLUMNS,
    CUE_SPECIFICITY_METRICS_REQUIRED_COLUMNS,
    CUE_SPECIFICITY_SPECS_REQUIRED_COLUMNS,
    FUNCTIONAL_BOUNDARY_REQUIRED_COLUMNS,
    MORPHOLOGY_BOUNDARY_REQUIRED_COLUMNS,
    FORMATION_INTERVENTION_SPEC_FILES,
    FORMATION_INTERVENTION_SPEC_REQUIRED_COLUMNS,
    FORMATION_RESULT_FILES,
    FORMATION_RESULT_REQUIRED_COLUMNS,
    REUSE_MODES,
    TASK_ACCESS_JOB_SPECS,
    TASK_ALL,
    TASK_BOUNDARY_CONDITION_SPECS,
    TASK_BOUNDARY_STATE_BANK,
    TASK_BOUNDARY_SUMMARY,
    TASK_CUE_SPECIFICITY_ACCESS,
    TASK_CUE_SPECIFICITY_SPECS,
    TASK_EXEMPLAR_DECODER,
    TASK_EXEMPLAR_DECODER_SPECS,
    TASK_EXEMPLAR_DECODER_STATE_BANK,
    TASK_EXEMPLAR_DECODER_SUMMARY,
    TASK_FORMATION_INTERVENTION_SPECS,
    TASK_FORMATION_NECESSITY,
    TASK_MORPHOLOGY_DECOMPOSITION,
    TASK_MORPHOLOGY_FUNCTION_COUPLING,
    TASK_IDS,
    TASK_NEUTRAL_PING,
    TASK_NEUTRAL_PING_ACCESS,
    TASK_PEAK_VALLEY_LANDSCAPE,
    TASK_PROGRESSIVE_UPDATE,
    TASK_SEQUENCE_TRIAL_SPECS,
    TASK_STATE_BANK,
    TASK_SUPPLEMENT,
    TASK_WEAK_CUE_ACCESS,
    TASK_WEAK_PROBE,
    normalize_reuse_mode,
)
from src.experiments.paper_figures.fig3.subexperiments.boundary_specs import build_access_job_specs, build_boundary_condition_specs
from src.experiments.paper_figures.fig3.subexperiments.boundary_state_bank import materialize_boundary_state_bank
from src.experiments.paper_figures.fig3.subexperiments.boundary_summary import compute_boundary_summary, compute_morphology_function_coupling
from src.experiments.paper_figures.fig3.subexperiments.cue_specificity import (
    build_cue_specificity_specs,
    compute_cue_specificity_tables,
    cue_specificity_scientific_checks,
    run_cue_specificity_readout,
)
from src.experiments.paper_figures.fig3.subexperiments.exemplar_decoder import (
    DELAY_MS as EXEMPLAR_DECODER_DELAY_MS,
    SEQUENCE_LENGTH as EXEMPLAR_DECODER_SEQUENCE_LENGTH,
    get_exemplar_decoder_results,
    get_exemplar_decoder_specs,
    get_exemplar_decoder_state_bank,
    run_exemplar_decoder_summary,
)
from src.experiments.paper_figures.fig3.subexperiments.formation_necessity import (
    build_formation_intervention_specs,
    run_formation_necessity,
)
from src.experiments.paper_figures.fig3.subexperiments.functional_access import run_neutral_ping_access, run_weak_cue_access
from src.experiments.paper_figures.fig3.subexperiments.morphology_decomposition import compute_morphology_decomposition, write_morphology_fit_outputs
from src.experiments.paper_figures.fig3.subexperiments.neutral_ping import run_neutral_ping_readout_distribution
from src.experiments.paper_figures.fig3.subexperiments.output_contract import (
    _write_config_files,
    _write_summary,
    utc_now,
    write_run_log_file,
)
from src.experiments.paper_figures.fig3.subexperiments.peak_valley_landscape import compute_final_support_landscape
from src.experiments.paper_figures.fig3.subexperiments.progressive_update import compute_progressive_update_metrics
from src.experiments.paper_figures.fig3.subexperiments.state_bank import run_multiitem_sequence_state_bank
from src.experiments.paper_figures.fig3.subexperiments.supplement import compute_supplementary_metrics
from src.experiments.paper_figures.fig3.subexperiments.trial_specs import build_sequence_trial_specs
from src.experiments.paper_figures.fig3.subexperiments.weak_probe import run_sequence_weak_probe_real_rollout_from_state_bank
from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _trial_condition_audit
from src.experiments.paper_figures.fig3.types import ExperimentContext, Fig3Config, MultiItemSequenceLandscapeBank
from src.experiments.paper_figures.common.sequence_root.artifacts import (
    copy_artifact_tree as copy_shared_artifact_tree,
    load_root_bank_artifact as load_shared_root_bank_artifact,
)
from src.experiments.paper_figures.common.specs.artifacts import materialize_spec_view
from src.experiments.paper_figures.run_paper_figures import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH_GLOB,
    DEFAULT_OUTPUT_ROOT,
    discover_checkpoints,
)


EXEMPLAR_DECODER_SEED_TASK_IDS = frozenset(
    {
        TASK_EXEMPLAR_DECODER_SPECS,
        TASK_EXEMPLAR_DECODER_STATE_BANK,
        TASK_EXEMPLAR_DECODER,
    }
)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--task" in raw_argv and "final-statistics" in raw_argv:
        from src.experiments.paper_figures.final_six.pipeline import canonical_runner_main

        return canonical_runner_main("fig3", raw_argv)
    args = _parse_args(argv)
    mode = normalize_reuse_mode(args.reuse_artifacts)
    if args.task == TASK_EXEMPLAR_DECODER_SUMMARY:
        run_exemplar_decoder_summary(_output_root_from_args(args), mode=mode)
        return 0
    cfg = _config_from_args(args)
    ctx = _build_context(cfg)
    artifact_root = _artifact_root_from_args(args, ctx.seed_dir)
    run_info = build_run_info(
        experiment_name=f"{FIGURE_ID}.{args.task}",
        output_dir=ctx.seed_dir,
        entry_script="src.experiments.paper_figures.fig3.run_task",
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
            "shared_sequence_root": str(Path(args.shared_sequence_root).resolve()) if args.shared_sequence_root else "",
            "task": str(args.task),
        }
    )
    write_run_info(ctx.seed_dir / "meta", run_info)
    try:
        _write_config_files(ctx)
        if str(args.task) in EXEMPLAR_DECODER_SEED_TASK_IDS:
            _run_exemplar_decoder_task(ctx, task_id=str(args.task), mode=mode, artifact_root=artifact_root)
        else:
            seq_trials, singleton_trials, partial_trials = _get_sequence_specs(
                ctx,
                task_id=str(args.task),
                mode=mode,
                artifact_root=artifact_root,
                shared_sequence_root=Path(args.shared_sequence_root) if args.shared_sequence_root else None,
            )
            spec_hash = sequence_specs_hash(seq_trials, singleton_trials, partial_trials)
            run_info["sequence_specs_hash"] = spec_hash
            _run_task(
                ctx,
                seq_trials,
                task_id=str(args.task),
                mode=mode,
                artifact_root=artifact_root,
                specs_hash=spec_hash,
                shared_sequence_root=Path(args.shared_sequence_root) if args.shared_sequence_root else None,
            )
        _finalize_bundle(ctx, artifact_root=artifact_root, mode=mode)
        finalize_run_info(ctx.seed_dir / "meta", run_info, status="success")
        return 0
    except Exception:
        finalize_run_info(ctx.seed_dir / "meta", run_info, status="failed")
        raise


def _run_exemplar_decoder_task(
    ctx: ExperimentContext,
    *,
    task_id: str,
    mode: str,
    artifact_root: Path,
) -> None:
    specs = get_exemplar_decoder_specs(ctx, mode=mode, artifact_root=artifact_root)
    if task_id == TASK_EXEMPLAR_DECODER_SPECS:
        return
    state_bank = get_exemplar_decoder_state_bank(ctx, specs, mode=mode, artifact_root=artifact_root)
    if task_id == TASK_EXEMPLAR_DECODER_STATE_BANK:
        return
    get_exemplar_decoder_results(ctx, specs, state_bank, mode=mode, artifact_root=artifact_root)


def _get_sequence_specs(
    ctx: ExperimentContext,
    *,
    task_id: str,
    mode: str,
    artifact_root: Path,
    shared_sequence_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_dir = task_artifact_dir(artifact_root, TASK_SEQUENCE_TRIAL_SPECS)
    expected_key = build_sequence_specs_cache_key(ctx.cfg)
    if shared_sequence_root is not None:
        root_bank = load_shared_root_bank_artifact(shared_sequence_root)
        artifact = save_sequence_specs_artifact(
            task_dir,
            sequence_trials=root_bank.specs.sequence_trials,
            singleton_reference_trials=root_bank.specs.singleton_reference_trials,
            partial_cue_trials=root_bank.specs.partial_cue_trials,
            cache_key=expected_key,
        )
        if root_bank.specs.spec_artifact is not None:
            materialize_spec_view(
                root_bank.specs.spec_artifact,
                task_dir,
                view_figure="fig3",
                view_task=TASK_SEQUENCE_TRIAL_SPECS,
                view_artifact_digest=artifact.digest,
                view_cache_key_digest=cache_key_digest(expected_key),
            )
        _write_sequence_specs_to_bundle(ctx, artifact.sequence_trials, artifact.singleton_reference_trials, artifact.partial_cue_trials)
        _set_artifact_metadata(ctx, "sequence_trial_specs", "shared_sequence_root", task_dir, artifact.digest, expected_key)
        return artifact.sequence_trials, artifact.singleton_reference_trials, artifact.partial_cue_trials
    if mode == "require":
        artifact = load_sequence_specs_artifact(task_dir, expected_key=expected_key)
        _write_sequence_specs_to_bundle(ctx, artifact.sequence_trials, artifact.singleton_reference_trials, artifact.partial_cue_trials)
        _set_artifact_metadata(ctx, "sequence_trial_specs", "loaded", task_dir, artifact.digest, expected_key)
        return artifact.sequence_trials, artifact.singleton_reference_trials, artifact.partial_cue_trials
    if mode == "auto" and cache_key_matches(task_dir, expected_key):
        artifact = load_sequence_specs_artifact(task_dir, expected_key=expected_key)
        _write_sequence_specs_to_bundle(ctx, artifact.sequence_trials, artifact.singleton_reference_trials, artifact.partial_cue_trials)
        _set_artifact_metadata(ctx, "sequence_trial_specs", "loaded", task_dir, artifact.digest, expected_key)
        return artifact.sequence_trials, artifact.singleton_reference_trials, artifact.partial_cue_trials
    seq_trials, singleton_trials, partial_trials = build_sequence_trial_specs(ctx)
    if mode != "off":
        artifact = save_sequence_specs_artifact(
            task_dir,
            sequence_trials=seq_trials,
            singleton_reference_trials=singleton_trials,
            partial_cue_trials=partial_trials,
            cache_key=expected_key,
        )
        _set_artifact_metadata(ctx, "sequence_trial_specs", "built", task_dir, artifact.digest, expected_key)
        seq_trials = artifact.sequence_trials
        singleton_trials = artifact.singleton_reference_trials
        partial_trials = artifact.partial_cue_trials
    ctx.run_log.append(f"{utc_now()} sequence_trial_specs source=built task={task_id}")
    return seq_trials, singleton_trials, partial_trials


def _write_sequence_specs_to_bundle(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    singleton_reference_trials: pd.DataFrame,
    partial_cue_trials: pd.DataFrame,
) -> None:
    save_csv_with_registry(ctx, sequence_trials, ctx.trial_specs_dir / "sequence_trials.csv")
    save_csv_with_registry(ctx, singleton_reference_trials, ctx.trial_specs_dir / "singleton_reference_trials.csv")
    save_csv_with_registry(ctx, partial_cue_trials, ctx.trial_specs_dir / "partial_cue_trials.csv")
    save_csv_with_registry(ctx, _trial_condition_audit(ctx.cfg.network_seed, sequence_trials), ctx.metrics_dir / "supp_trial_condition_audit.csv")
    if not sequence_trials.empty:
        example = sequence_trials[sequence_trials["sequence_id"].astype(int).eq(int(sequence_trials["sequence_id"].iloc[0]))].copy()
        write_json_file(json_safe(example.iloc[0].to_dict()), ctx.raw_dir / "panel_a_example_sequence_metadata.json")
        np.savez_compressed(
            ctx.raw_dir / "panel_a_example_sequence.npz",
            image_ids=example["item_image_id"].to_numpy(dtype=np.int64),
            labels=example["item_label"].to_numpy(dtype=np.int64),
        )
        ctx.output_files["panel_a_example_sequence_metadata"] = relative_to_root(ctx.raw_dir / "panel_a_example_sequence_metadata.json", ctx.seed_dir)
        ctx.output_files["panel_a_example_sequence"] = relative_to_root(ctx.raw_dir / "panel_a_example_sequence.npz", ctx.seed_dir)
    ctx.n_sequences = int(sequence_trials["sequence_id"].nunique()) if "sequence_id" in sequence_trials.columns else 0
    ctx.completed_modules["sequence_trial_specs"] = True


def _run_task(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    *,
    task_id: str,
    mode: str,
    artifact_root: Path,
    specs_hash: str,
    shared_sequence_root: Path | None,
) -> None:
    if task_id == TASK_SEQUENCE_TRIAL_SPECS:
        return
    if task_id == TASK_BOUNDARY_CONDITION_SPECS:
        _get_boundary_condition_specs(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        return
    if task_id == TASK_ACCESS_JOB_SPECS:
        condition_artifact = _get_boundary_condition_specs(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        _get_access_job_specs(ctx, sequence_trials, condition_artifact, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        return
    if task_id == TASK_BOUNDARY_STATE_BANK:
        condition_artifact = _get_boundary_condition_specs(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        _get_boundary_state_bank(
            ctx,
            sequence_trials,
            condition_artifact,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
            shared_sequence_root=shared_sequence_root,
        )
        return
    if task_id == TASK_MORPHOLOGY_DECOMPOSITION:
        condition_artifact = _get_boundary_condition_specs(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        boundary_artifact = _get_boundary_state_bank(
            ctx,
            sequence_trials,
            condition_artifact,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
            shared_sequence_root=shared_sequence_root,
        )
        _get_morphology_decomposition(ctx, condition_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
        return
    if task_id == TASK_NEUTRAL_PING_ACCESS:
        condition_artifact, access_artifact, boundary_artifact = _get_boundary_access_parents(
            ctx,
            sequence_trials,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
            shared_sequence_root=shared_sequence_root,
        )
        _get_neutral_ping_access(ctx, access_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
        return
    if task_id == TASK_WEAK_CUE_ACCESS:
        condition_artifact, access_artifact, boundary_artifact = _get_boundary_access_parents(
            ctx,
            sequence_trials,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
            shared_sequence_root=shared_sequence_root,
        )
        _get_weak_cue_access(ctx, access_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
        return
    if task_id == TASK_CUE_SPECIFICITY_SPECS:
        condition_artifact = _get_boundary_condition_specs(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        access_artifact = _get_access_job_specs(ctx, sequence_trials, condition_artifact, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        _get_cue_specificity_specs(ctx, sequence_trials, access_artifact, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        return
    if task_id == TASK_CUE_SPECIFICITY_ACCESS:
        condition_artifact, access_artifact, boundary_artifact = _get_boundary_access_parents(
            ctx,
            sequence_trials,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
            shared_sequence_root=shared_sequence_root,
        )
        cue_specs_artifact = _get_cue_specificity_specs(ctx, sequence_trials, access_artifact, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        _get_cue_specificity_access(ctx, cue_specs_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
        return
    if task_id == TASK_MORPHOLOGY_FUNCTION_COUPLING:
        condition_artifact, access_artifact, boundary_artifact = _get_boundary_access_parents(
            ctx,
            sequence_trials,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
            shared_sequence_root=shared_sequence_root,
        )
        morphology_artifact = _get_morphology_decomposition(ctx, condition_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
        weak_cue_artifact = _get_weak_cue_access(ctx, access_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
        _get_morphology_function_coupling(ctx, morphology_artifact, weak_cue_artifact, mode=mode, artifact_root=artifact_root)
        return
    if task_id == TASK_BOUNDARY_SUMMARY:
        parents = _get_new_boundary_downstream_parents(
            ctx,
            sequence_trials,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
            shared_sequence_root=shared_sequence_root,
        )
        _get_boundary_summary(ctx, *parents, mode=mode, artifact_root=artifact_root)
        return
    if task_id == TASK_STATE_BANK:
        _get_state_bank(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash, shared_sequence_root=shared_sequence_root)
        return
    if task_id == TASK_PROGRESSIVE_UPDATE:
        condition_artifact = _get_boundary_condition_specs(
            ctx,
            sequence_trials,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
        )
        boundary_artifact = _get_boundary_state_bank(
            ctx,
            sequence_trials,
            condition_artifact,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
            shared_sequence_root=shared_sequence_root,
        )
        compute_progressive_update_metrics(ctx, boundary_artifact.bank)
        return
    if task_id in {TASK_FORMATION_INTERVENTION_SPECS, TASK_FORMATION_NECESSITY}:
        condition_artifact = _get_boundary_condition_specs(
            ctx,
            sequence_trials,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
        )
        boundary_artifact = _get_boundary_state_bank(
            ctx,
            sequence_trials,
            condition_artifact,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
            shared_sequence_root=shared_sequence_root,
        )
        formation_specs = _get_formation_intervention_specs(
            ctx,
            boundary_artifact,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
        )
        if task_id == TASK_FORMATION_NECESSITY:
            _get_formation_necessity(
                ctx,
                boundary_artifact,
                formation_specs,
                mode=mode,
                artifact_root=artifact_root,
            )
        return
    if task_id == TASK_PEAK_VALLEY_LANDSCAPE:
        condition_artifact = _get_boundary_condition_specs(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        boundary_artifact = _get_boundary_state_bank(
            ctx,
            sequence_trials,
            condition_artifact,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
            shared_sequence_root=shared_sequence_root,
        )
        compute_final_support_landscape(ctx, boundary_artifact.bank)
        return
    if task_id == TASK_NEUTRAL_PING:
        run_neutral_ping_readout_distribution(ctx, _get_state_bank(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash, shared_sequence_root=shared_sequence_root))
        return
    if task_id == TASK_WEAK_PROBE:
        run_sequence_weak_probe_real_rollout_from_state_bank(ctx, _get_state_bank(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash, shared_sequence_root=shared_sequence_root))
        return
    if task_id == TASK_SUPPLEMENT:
        compute_supplementary_metrics(ctx, _get_state_bank(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash, shared_sequence_root=shared_sequence_root))
        return
    if task_id == TASK_ALL:
        condition_artifact, access_artifact, boundary_artifact = _get_boundary_access_parents(
            ctx,
            sequence_trials,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
            shared_sequence_root=shared_sequence_root,
        )
        morphology_artifact = _get_morphology_decomposition(ctx, condition_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
        weak_cue_artifact = _get_weak_cue_access(ctx, access_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
        neutral_ping_artifact = _get_neutral_ping_access(ctx, access_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
        coupling_artifact = _get_morphology_function_coupling(ctx, morphology_artifact, weak_cue_artifact, mode=mode, artifact_root=artifact_root)
        _get_boundary_summary(ctx, morphology_artifact, weak_cue_artifact, neutral_ping_artifact, coupling_artifact, mode=mode, artifact_root=artifact_root)
        cue_specs_artifact = _get_cue_specificity_specs(ctx, sequence_trials, access_artifact, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        _get_cue_specificity_access(ctx, cue_specs_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
        compute_progressive_update_metrics(ctx, boundary_artifact.bank)
        return
    raise ValueError(f"Unsupported Fig.3 task: {task_id}")


def _get_state_bank(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
    specs_hash: str,
    shared_sequence_root: Path | None = None,
) -> MultiItemSequenceLandscapeBank:
    if mode == "off" and shared_sequence_root is None:
        return run_multiitem_sequence_state_bank(ctx, sequence_trials)
    return _get_state_bank_artifact(
        ctx,
        sequence_trials,
        mode=mode,
        artifact_root=artifact_root,
        specs_hash=specs_hash,
        shared_sequence_root=shared_sequence_root,
    ).bank


def _get_state_bank_artifact(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
    specs_hash: str,
    shared_sequence_root: Path | None = None,
    write_compat_outputs: bool = True,
) -> StateBankArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_STATE_BANK)
    expected_key = build_state_bank_cache_key(ctx.cfg, specs_hash=specs_hash)
    if shared_sequence_root is not None:
        root_bank = load_shared_root_bank_artifact(shared_sequence_root)
        copy_shared_artifact_tree(root_bank.fig3_state_bank_dir, task_dir)
        artifact = load_state_bank_artifact(task_dir, expected_key=expected_key, sequence_trials=sequence_trials)
        if write_compat_outputs:
            _write_state_bank_compat_outputs(ctx, task_dir)
        _set_artifact_metadata(ctx, "state_bank", "shared_sequence_root", task_dir, artifact.digest, expected_key)
        ctx.completed_modules["state_bank"] = True
        ctx.run_log.append(f"{utc_now()} state_bank source=shared_sequence_root artifact={root_bank.root}")
        return artifact
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_state_bank_artifact(task_dir, expected_key=expected_key, sequence_trials=sequence_trials)
        if write_compat_outputs:
            _write_state_bank_compat_outputs(ctx, task_dir)
        _set_artifact_metadata(ctx, "state_bank", "loaded", task_dir, artifact.digest, expected_key)
        ctx.completed_modules["state_bank"] = True
        return artifact
    if mode == "require":
        artifact = load_state_bank_artifact(task_dir, expected_key=expected_key, sequence_trials=sequence_trials)
        if write_compat_outputs:
            _write_state_bank_compat_outputs(ctx, task_dir)
        _set_artifact_metadata(ctx, "state_bank", "loaded", task_dir, artifact.digest, expected_key)
        ctx.completed_modules["state_bank"] = True
        return artifact
    if mode == "off":
        raise ValueError("Fig.3 state-bank artifact helper cannot return a persisted artifact in reuse mode 'off'.")
    bank = run_multiitem_sequence_state_bank(ctx, sequence_trials, write_compat_outputs=write_compat_outputs)
    artifact = save_state_bank_artifact(task_dir, bank, cache_key=expected_key, network_seed=ctx.cfg.network_seed)
    _set_artifact_metadata(ctx, "state_bank", "built", task_dir, artifact.digest, expected_key)
    return artifact




def _get_formation_intervention_specs(
    ctx: ExperimentContext,
    boundary_artifact: StateBankArtifact,
    *,
    mode: str,
    artifact_root: Path,
    specs_hash: str,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_FORMATION_INTERVENTION_SPECS)
    expected_key = build_formation_intervention_specs_cache_key(
        ctx.cfg,
        boundary_state_digest=boundary_artifact.digest,
        specs_hash=specs_hash,
    )
    expected_columns = {
        "formation_intervention_specs": FORMATION_INTERVENTION_SPEC_REQUIRED_COLUMNS,
    }
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_table_bundle_artifact(
            task_dir,
            expected_key=expected_key,
            expected_names=FORMATION_INTERVENTION_SPEC_FILES.keys(),
            expected_columns=expected_columns,
        )
        _copy_formation_specs_to_bundle(ctx, task_dir)
        _set_artifact_metadata(ctx, TASK_FORMATION_INTERVENTION_SPECS, "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "require":
        artifact = load_table_bundle_artifact(
            task_dir,
            expected_key=expected_key,
            expected_names=FORMATION_INTERVENTION_SPEC_FILES.keys(),
            expected_columns=expected_columns,
        )
        _copy_formation_specs_to_bundle(ctx, task_dir)
        _set_artifact_metadata(ctx, TASK_FORMATION_INTERVENTION_SPECS, "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    table = build_formation_intervention_specs(ctx, boundary_artifact.bank)
    tables = {"formation_intervention_specs": table}
    if mode == "off":
        _write_formation_specs_to_bundle(ctx, table)
        return TableBundleArtifact(task_dir, tables, pd.DataFrame(), table_digest(tables))
    artifact = save_table_bundle_artifact(
        task_dir,
        tables=tables,
        filenames=FORMATION_INTERVENTION_SPEC_FILES,
        cache_key=expected_key,
    )
    _copy_formation_specs_to_bundle(ctx, task_dir)
    _set_artifact_metadata(ctx, TASK_FORMATION_INTERVENTION_SPECS, "built", task_dir, artifact.digest, expected_key)
    return artifact


def _get_formation_necessity(
    ctx: ExperimentContext,
    boundary_artifact: StateBankArtifact,
    specs_artifact: TableBundleArtifact,
    *,
    mode: str,
    artifact_root: Path,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_FORMATION_NECESSITY)
    expected_key = build_formation_necessity_cache_key(
        ctx.cfg,
        boundary_state_digest=boundary_artifact.digest,
        intervention_specs_digest=specs_artifact.digest,
    )
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_table_bundle_artifact(
            task_dir,
            expected_key=expected_key,
            expected_names=FORMATION_RESULT_FILES.keys(),
            expected_columns=FORMATION_RESULT_REQUIRED_COLUMNS,
        )
        _copy_formation_tables_to_bundle(ctx, task_dir)
        _set_artifact_metadata(ctx, TASK_FORMATION_NECESSITY, "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "require":
        artifact = load_table_bundle_artifact(
            task_dir,
            expected_key=expected_key,
            expected_names=FORMATION_RESULT_FILES.keys(),
            expected_columns=FORMATION_RESULT_REQUIRED_COLUMNS,
        )
        _copy_formation_tables_to_bundle(ctx, task_dir)
        _set_artifact_metadata(ctx, TASK_FORMATION_NECESSITY, "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    tables = run_formation_necessity(
        ctx,
        boundary_artifact.bank,
        specs_artifact.tables["formation_intervention_specs"],
    )
    if mode == "off":
        _write_formation_tables_to_bundle(ctx, tables)
        return TableBundleArtifact(task_dir, tables, pd.DataFrame(), table_digest(tables))
    artifact = save_table_bundle_artifact(
        task_dir,
        tables=tables,
        filenames=FORMATION_RESULT_FILES,
        cache_key=expected_key,
    )
    _copy_formation_tables_to_bundle(ctx, task_dir)
    _set_artifact_metadata(ctx, TASK_FORMATION_NECESSITY, "built", task_dir, artifact.digest, expected_key)
    return artifact


def _get_boundary_condition_specs(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
    specs_hash: str,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_BOUNDARY_CONDITION_SPECS)
    expected_key = build_boundary_condition_specs_cache_key(ctx.cfg, specs_hash=specs_hash)
    filenames = {"boundary_condition_specs": "boundary_condition_specs.csv"}
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys())
        _write_boundary_condition_specs_to_bundle(ctx, artifact.tables["boundary_condition_specs"])
        _set_artifact_metadata(ctx, "boundary_condition_specs", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "require":
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys())
        _write_boundary_condition_specs_to_bundle(ctx, artifact.tables["boundary_condition_specs"])
        _set_artifact_metadata(ctx, "boundary_condition_specs", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    table = build_boundary_condition_specs(ctx, sequence_trials)
    if mode == "off":
        return TableBundleArtifact(task_dir, {"boundary_condition_specs": table}, pd.DataFrame(), table_digest({"boundary_condition_specs": table}))
    artifact = save_table_bundle_artifact(task_dir, tables={"boundary_condition_specs": table}, filenames=filenames, cache_key=expected_key)
    _set_artifact_metadata(ctx, "boundary_condition_specs", "built", task_dir, artifact.digest, expected_key)
    return artifact


def _get_access_job_specs(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    condition_artifact: TableBundleArtifact,
    *,
    mode: str,
    artifact_root: Path,
    specs_hash: str,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_ACCESS_JOB_SPECS)
    expected_key = build_access_job_specs_cache_key(
        ctx.cfg,
        specs_hash=specs_hash,
        condition_specs_digest=condition_artifact.digest,
    )
    filenames = {"access_job_specs": "access_job_specs.csv"}
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys())
        _write_access_job_specs_to_bundle(ctx, artifact.tables["access_job_specs"])
        _set_artifact_metadata(ctx, "access_job_specs", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "require":
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys())
        _write_access_job_specs_to_bundle(ctx, artifact.tables["access_job_specs"])
        _set_artifact_metadata(ctx, "access_job_specs", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    table = build_access_job_specs(ctx, sequence_trials, condition_artifact.tables["boundary_condition_specs"])
    if mode == "off":
        return TableBundleArtifact(task_dir, {"access_job_specs": table}, pd.DataFrame(), table_digest({"access_job_specs": table}))
    artifact = save_table_bundle_artifact(task_dir, tables={"access_job_specs": table}, filenames=filenames, cache_key=expected_key)
    _set_artifact_metadata(ctx, "access_job_specs", "built", task_dir, artifact.digest, expected_key)
    return artifact


def _get_boundary_state_bank(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    condition_artifact: TableBundleArtifact,
    *,
    mode: str,
    artifact_root: Path,
    specs_hash: str,
    shared_sequence_root: Path | None,
) -> StateBankArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_BOUNDARY_STATE_BANK)
    expected_key = build_boundary_state_bank_cache_key(
        ctx.cfg,
        specs_hash=specs_hash,
        condition_specs_digest=condition_artifact.digest,
    )
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_state_bank_artifact(task_dir, expected_key=expected_key, sequence_trials=sequence_trials)
        materialize_boundary_state_bank(ctx, artifact.bank, condition_artifact.tables["boundary_condition_specs"], sequence_trials=sequence_trials)
        _write_state_bank_compat_outputs(ctx, task_dir)
        _set_artifact_metadata(ctx, "boundary_state_bank", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "require":
        artifact = load_state_bank_artifact(task_dir, expected_key=expected_key, sequence_trials=sequence_trials)
        materialize_boundary_state_bank(ctx, artifact.bank, condition_artifact.tables["boundary_condition_specs"], sequence_trials=sequence_trials)
        _write_state_bank_compat_outputs(ctx, task_dir)
        _set_artifact_metadata(ctx, "boundary_state_bank", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    condition_specs = condition_artifact.tables["boundary_condition_specs"]
    condition_delays = sorted(int(v) for v in condition_specs["delay_ms"].dropna().unique())
    source_bank = None
    if condition_delays == [int(ctx.cfg.delay_ms)]:
        source_bank = _get_state_bank(
            ctx,
            sequence_trials,
            mode=mode,
            artifact_root=artifact_root,
            specs_hash=specs_hash,
            shared_sequence_root=shared_sequence_root,
        )
    bank = materialize_boundary_state_bank(ctx, source_bank, condition_specs, sequence_trials=sequence_trials)
    if mode == "off":
        return StateBankArtifact(task_dir, bank, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "off")
    artifact = save_state_bank_artifact(task_dir, bank, cache_key=expected_key, network_seed=ctx.cfg.network_seed)
    _write_state_bank_compat_outputs(ctx, task_dir)
    _set_artifact_metadata(ctx, "boundary_state_bank", "built", task_dir, artifact.digest, expected_key)
    return artifact


def _get_boundary_access_parents(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
    specs_hash: str,
    shared_sequence_root: Path | None,
) -> tuple[TableBundleArtifact, TableBundleArtifact, StateBankArtifact]:
    condition_artifact = _get_boundary_condition_specs(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
    access_artifact = _get_access_job_specs(ctx, sequence_trials, condition_artifact, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
    boundary_artifact = _get_boundary_state_bank(
        ctx,
        sequence_trials,
        condition_artifact,
        mode=mode,
        artifact_root=artifact_root,
        specs_hash=specs_hash,
        shared_sequence_root=shared_sequence_root,
    )
    return condition_artifact, access_artifact, boundary_artifact


def _get_morphology_decomposition(
    ctx: ExperimentContext,
    condition_artifact: TableBundleArtifact,
    boundary_artifact: StateBankArtifact,
    *,
    mode: str,
    artifact_root: Path,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_MORPHOLOGY_DECOMPOSITION)
    expected_key = build_morphology_decomposition_cache_key(
        ctx.cfg,
        boundary_state_digest=boundary_artifact.digest,
        condition_specs_digest=condition_artifact.digest,
    )
    filenames = {
        "morphology_item_support": "panel_b_morphology_item_support.csv",
        "morphology_serial_profile": "panel_b_morphology_serial_profile.csv",
        "morphology_boundary_metrics": "panel_c_morphology_boundary_metrics.csv",
    }
    expected_columns = {"morphology_boundary_metrics": MORPHOLOGY_BOUNDARY_REQUIRED_COLUMNS}
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys(), expected_columns=expected_columns)
        _write_morphology_tables_to_bundle(ctx, artifact.tables)
        _set_artifact_metadata(ctx, "morphology_decomposition", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "require":
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys(), expected_columns=expected_columns)
        _write_morphology_tables_to_bundle(ctx, artifact.tables)
        _set_artifact_metadata(ctx, "morphology_decomposition", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    tables = compute_morphology_decomposition(ctx, boundary_artifact.bank, condition_artifact.tables["boundary_condition_specs"])
    if mode == "off":
        return TableBundleArtifact(task_dir, tables, pd.DataFrame(), table_digest(tables))
    artifact = save_table_bundle_artifact(task_dir, tables=tables, filenames=filenames, cache_key=expected_key)
    _set_artifact_metadata(ctx, "morphology_decomposition", "built", task_dir, artifact.digest, expected_key)
    return artifact


def _get_weak_cue_access(
    ctx: ExperimentContext,
    access_artifact: TableBundleArtifact,
    boundary_artifact: StateBankArtifact,
    *,
    mode: str,
    artifact_root: Path,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_WEAK_CUE_ACCESS)
    expected_key = build_weak_cue_access_cache_key(
        ctx.cfg,
        boundary_state_digest=boundary_artifact.digest,
        access_job_specs_digest=access_artifact.digest,
    )
    filenames = {
        "weak_cue_item_readout": "panel_d_weak_cue_item_readout.csv",
        "weak_cue_item_metrics": "panel_d_weak_cue_item_metrics.csv",
        "item_functional_gain": "panel_d_item_functional_gain.csv",
        "functional_boundary_metrics": "panel_d_functional_boundary_metrics.csv",
    }
    expected_columns = {"functional_boundary_metrics": FUNCTIONAL_BOUNDARY_REQUIRED_COLUMNS}
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys(), expected_columns=expected_columns)
        _write_weak_cue_tables_to_bundle(ctx, artifact.tables)
        _set_artifact_metadata(ctx, "weak_cue_access", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "require":
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys(), expected_columns=expected_columns)
        _write_weak_cue_tables_to_bundle(ctx, artifact.tables)
        _set_artifact_metadata(ctx, "weak_cue_access", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    tables = run_weak_cue_access(ctx, boundary_artifact.bank, access_artifact.tables["access_job_specs"])
    if mode == "off":
        return TableBundleArtifact(task_dir, tables, pd.DataFrame(), table_digest(tables))
    artifact = save_table_bundle_artifact(task_dir, tables=tables, filenames=filenames, cache_key=expected_key)
    _set_artifact_metadata(ctx, "weak_cue_access", "built", task_dir, artifact.digest, expected_key)
    return artifact


def _get_cue_specificity_specs(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    access_artifact: TableBundleArtifact,
    *,
    mode: str,
    artifact_root: Path,
    specs_hash: str,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_CUE_SPECIFICITY_SPECS)
    expected_key = build_cue_specificity_specs_cache_key(
        ctx.cfg,
        specs_hash=specs_hash,
        access_job_specs_digest=access_artifact.digest,
    )
    filenames = {"cue_specificity_specs": "cue_specificity_specs.csv"}
    expected_columns = {"cue_specificity_specs": CUE_SPECIFICITY_SPECS_REQUIRED_COLUMNS}
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys(), expected_columns=expected_columns)
        _write_cue_specificity_specs_to_bundle(ctx, artifact.tables["cue_specificity_specs"])
        _set_artifact_metadata(ctx, "cue_specificity_specs", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "require":
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys(), expected_columns=expected_columns)
        _write_cue_specificity_specs_to_bundle(ctx, artifact.tables["cue_specificity_specs"])
        _set_artifact_metadata(ctx, "cue_specificity_specs", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    table = build_cue_specificity_specs(
        ctx,
        sequence_trials,
        access_artifact.tables["access_job_specs"],
        seq_len=int(ctx.cfg.cue_specificity_seq_len),
        delay_ms=int(ctx.cfg.cue_specificity_delay_ms),
        keep_prob=float(ctx.cfg.cue_specificity_keep_prob),
        cue_types=ctx.cfg.cue_specificity_cue_types,
    )
    if mode == "off":
        return TableBundleArtifact(task_dir, {"cue_specificity_specs": table}, pd.DataFrame(), table_digest({"cue_specificity_specs": table}))
    artifact = save_table_bundle_artifact(task_dir, tables={"cue_specificity_specs": table}, filenames=filenames, cache_key=expected_key)
    _write_cue_specificity_specs_to_bundle(ctx, artifact.tables["cue_specificity_specs"])
    _set_artifact_metadata(ctx, "cue_specificity_specs", "built", task_dir, artifact.digest, expected_key)
    return artifact


def _get_cue_specificity_access(
    ctx: ExperimentContext,
    cue_specs_artifact: TableBundleArtifact,
    boundary_artifact: StateBankArtifact,
    *,
    mode: str,
    artifact_root: Path,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_CUE_SPECIFICITY_ACCESS)
    expected_key = build_cue_specificity_access_cache_key(
        ctx.cfg,
        boundary_state_digest=boundary_artifact.digest,
        cue_specificity_specs_digest=cue_specs_artifact.digest,
    )
    filenames = {
        "cue_specificity_trial_readout": "panel_c_cue_specificity_trial_readout.csv",
        "cue_specificity_metrics": "panel_c_cue_specificity_metrics.csv",
        "cue_specificity_memory_gain": "panel_c_cue_specificity_memory_gain.csv",
        "cue_specificity_serial_summary": "panel_c_cue_specificity_serial_summary.csv",
        "cue_specificity_contrast_summary": "panel_c_cue_specificity_contrast_summary.csv",
        "cue_specificity_summary": "panel_c_cue_specificity_summary.csv",
    }
    expected_columns = {"cue_specificity_metrics": CUE_SPECIFICITY_METRICS_REQUIRED_COLUMNS}
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys(), expected_columns=expected_columns)
        _write_cue_specificity_access_tables_to_bundle(ctx, artifact.tables)
        _validate_cue_specificity_science(ctx, artifact.tables)
        _set_artifact_metadata(ctx, "cue_specificity_access", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "require":
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys(), expected_columns=expected_columns)
        _write_cue_specificity_access_tables_to_bundle(ctx, artifact.tables)
        _validate_cue_specificity_science(ctx, artifact.tables)
        _set_artifact_metadata(ctx, "cue_specificity_access", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    cue_specs = cue_specs_artifact.tables["cue_specificity_specs"]
    raw = run_cue_specificity_readout(
        ctx,
        boundary_artifact.bank,
        cue_specs,
        readout_batch_size=int(ctx.cfg.cue_specificity_readout_batch_size),
    )
    if len(raw) != len(cue_specs):
        raise RuntimeError(f"Cue specificity raw row count mismatch: found {len(raw)}, expected {len(cue_specs)}.")
    tables = compute_cue_specificity_tables(raw)
    _validate_cue_specificity_science(ctx, tables)
    if mode == "off":
        return TableBundleArtifact(task_dir, tables, pd.DataFrame(), table_digest(tables))
    artifact = save_table_bundle_artifact(task_dir, tables=tables, filenames=filenames, cache_key=expected_key)
    _write_cue_specificity_access_tables_to_bundle(ctx, artifact.tables)
    _set_artifact_metadata(ctx, "cue_specificity_access", "built", task_dir, artifact.digest, expected_key)
    return artifact


def _get_neutral_ping_access(
    ctx: ExperimentContext,
    access_artifact: TableBundleArtifact,
    boundary_artifact: StateBankArtifact,
    *,
    mode: str,
    artifact_root: Path,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_NEUTRAL_PING_ACCESS)
    expected_key = build_neutral_ping_access_cache_key(
        ctx.cfg,
        boundary_state_digest=boundary_artifact.digest,
        access_job_specs_digest=access_artifact.digest,
    )
    filenames = {
        "neutral_ping_access_readout": "panel_c_neutral_ping_access_readout.csv",
        "neutral_ping_position_distribution": "panel_c_neutral_ping_position_distribution.csv",
        "neutral_ping_access_summary": "panel_c_neutral_ping_access_summary.csv",
    }
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys())
        _write_neutral_ping_tables_to_bundle(ctx, artifact.tables)
        _set_artifact_metadata(ctx, "neutral_ping_access", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "require":
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys())
        _write_neutral_ping_tables_to_bundle(ctx, artifact.tables)
        _set_artifact_metadata(ctx, "neutral_ping_access", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    tables = run_neutral_ping_access(ctx, boundary_artifact.bank, access_artifact.tables["access_job_specs"])
    if mode == "off":
        return TableBundleArtifact(task_dir, tables, pd.DataFrame(), table_digest(tables))
    artifact = save_table_bundle_artifact(task_dir, tables=tables, filenames=filenames, cache_key=expected_key)
    _set_artifact_metadata(ctx, "neutral_ping_access", "built", task_dir, artifact.digest, expected_key)
    return artifact


def _get_morphology_function_coupling(
    ctx: ExperimentContext,
    morphology_artifact: TableBundleArtifact,
    weak_cue_artifact: TableBundleArtifact,
    *,
    mode: str,
    artifact_root: Path,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_MORPHOLOGY_FUNCTION_COUPLING)
    expected_key = build_morphology_function_coupling_cache_key(
        ctx.cfg,
        morphology_digest=morphology_artifact.digest,
        weak_cue_digest=weak_cue_artifact.digest,
    )
    filenames = {
        "morphology_function_coupling": "panel_e_morphology_function_coupling.csv",
        "coupling_summary": "panel_e_coupling_summary.csv",
        "order_specificity_control": "panel_f_order_specificity_control.csv",
    }
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys())
        _write_coupling_tables_to_bundle(ctx, artifact.tables)
        _set_artifact_metadata(ctx, "morphology_function_coupling", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "require":
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys())
        _write_coupling_tables_to_bundle(ctx, artifact.tables)
        _set_artifact_metadata(ctx, "morphology_function_coupling", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    tables = compute_morphology_function_coupling(ctx, morphology_artifact.tables, weak_cue_artifact.tables)
    if mode == "off":
        return TableBundleArtifact(task_dir, tables, pd.DataFrame(), table_digest(tables))
    artifact = save_table_bundle_artifact(task_dir, tables=tables, filenames=filenames, cache_key=expected_key)
    _set_artifact_metadata(ctx, "morphology_function_coupling", "built", task_dir, artifact.digest, expected_key)
    return artifact


def _get_new_boundary_downstream_parents(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
    specs_hash: str,
    shared_sequence_root: Path | None,
) -> tuple[TableBundleArtifact, TableBundleArtifact, TableBundleArtifact, TableBundleArtifact]:
    condition_artifact, access_artifact, boundary_artifact = _get_boundary_access_parents(
        ctx,
        sequence_trials,
        mode=mode,
        artifact_root=artifact_root,
        specs_hash=specs_hash,
        shared_sequence_root=shared_sequence_root,
    )
    morphology_artifact = _get_morphology_decomposition(ctx, condition_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
    weak_cue_artifact = _get_weak_cue_access(ctx, access_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
    neutral_ping_artifact = _get_neutral_ping_access(ctx, access_artifact, boundary_artifact, mode=mode, artifact_root=artifact_root)
    coupling_artifact = _get_morphology_function_coupling(ctx, morphology_artifact, weak_cue_artifact, mode=mode, artifact_root=artifact_root)
    return morphology_artifact, weak_cue_artifact, neutral_ping_artifact, coupling_artifact


def _get_boundary_summary(
    ctx: ExperimentContext,
    morphology_artifact: TableBundleArtifact,
    weak_cue_artifact: TableBundleArtifact,
    neutral_ping_artifact: TableBundleArtifact,
    coupling_artifact: TableBundleArtifact,
    *,
    mode: str,
    artifact_root: Path,
) -> TableBundleArtifact:
    task_dir = task_artifact_dir(artifact_root, TASK_BOUNDARY_SUMMARY)
    expected_key = build_boundary_summary_cache_key(
        ctx.cfg,
        morphology_digest=morphology_artifact.digest,
        weak_cue_digest=weak_cue_artifact.digest,
        neutral_ping_digest=neutral_ping_artifact.digest,
        coupling_digest=coupling_artifact.digest,
    )
    filenames = {"boundary_summary": "panel_f_boundary_summary.csv"}
    expected_columns = {"boundary_summary": BOUNDARY_SUMMARY_REQUIRED_COLUMNS}
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys(), expected_columns=expected_columns)
        _write_boundary_summary_tables_to_bundle(ctx, artifact.tables)
        _set_artifact_metadata(ctx, "boundary_summary", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "require":
        artifact = load_table_bundle_artifact(task_dir, expected_key=expected_key, expected_names=filenames.keys(), expected_columns=expected_columns)
        _write_boundary_summary_tables_to_bundle(ctx, artifact.tables)
        _set_artifact_metadata(ctx, "boundary_summary", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    tables = compute_boundary_summary(ctx, morphology_artifact.tables, weak_cue_artifact.tables, neutral_ping_artifact.tables, coupling_artifact.tables)
    if mode == "off":
        return TableBundleArtifact(task_dir, tables, pd.DataFrame(), table_digest(tables))
    artifact = save_table_bundle_artifact(task_dir, tables=tables, filenames=filenames, cache_key=expected_key)
    _set_artifact_metadata(ctx, "boundary_summary", "built", task_dir, artifact.digest, expected_key)
    return artifact


def _write_state_bank_compat_outputs(ctx: ExperimentContext, task_dir: Path) -> None:
    for filename in ("state_bank_layer1.npz", "state_bank_layer2.npz", "state_bank_layer3.npz"):
        src = task_dir / filename
        dst = ctx.raw_dir / filename
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            ctx.output_files[dst.stem] = relative_to_root(dst, ctx.seed_dir)
    manifest_src = task_dir / "manifest.csv"
    if manifest_src.exists():
        dst = ctx.raw_dir / "state_bank_manifest.csv"
        shutil.copy2(manifest_src, dst)
        ctx.output_files["state_bank_manifest"] = relative_to_root(dst, ctx.seed_dir)


def _copy_formation_specs_to_bundle(
    ctx: ExperimentContext,
    task_dir: Path,
) -> None:
    source = task_dir / FORMATION_INTERVENTION_SPEC_FILES[
        "formation_intervention_specs"
    ]
    destination = ctx.trial_specs_dir / "formation_intervention_specs.csv"
    shutil.copy2(source, destination)
    ctx.output_files[destination.stem] = relative_to_root(destination, ctx.seed_dir)
    ctx.completed_modules[TASK_FORMATION_INTERVENTION_SPECS] = True


def _write_formation_specs_to_bundle(ctx: ExperimentContext, table: pd.DataFrame) -> None:
    save_csv_with_registry(ctx, table, ctx.trial_specs_dir / "formation_intervention_specs.csv")
    ctx.completed_modules[TASK_FORMATION_INTERVENTION_SPECS] = True


def _copy_formation_tables_to_bundle(
    ctx: ExperimentContext,
    task_dir: Path,
) -> None:
    destinations = {
        "formation_stage_readout": ctx.raw_dir / "formation_stage_readout.csv",
        "formation_access_readout": ctx.raw_dir / "formation_access_readout.csv",
        "formation_pair_specificity": (
            ctx.metrics_dir / "formation_pair_specificity.csv"
        ),
        "formation_condition_summary": (
            ctx.metrics_dir / "formation_condition_summary.csv"
        ),
    }
    for name, destination in destinations.items():
        shutil.copy2(task_dir / FORMATION_RESULT_FILES[name], destination)
        ctx.output_files[destination.stem] = relative_to_root(
            destination,
            ctx.seed_dir,
        )
    ctx.completed_modules[TASK_FORMATION_NECESSITY] = True


def _write_formation_tables_to_bundle(
    ctx: ExperimentContext,
    tables: Mapping[str, pd.DataFrame],
) -> None:
    mapping = {
        "formation_stage_readout": ctx.raw_dir / "formation_stage_readout.csv",
        "formation_access_readout": ctx.raw_dir / "formation_access_readout.csv",
        "formation_pair_specificity": ctx.metrics_dir / "formation_pair_specificity.csv",
        "formation_condition_summary": ctx.metrics_dir / "formation_condition_summary.csv",
    }
    for name, path in mapping.items():
        if name in tables:
            save_csv_with_registry(ctx, tables[name], path)
    ctx.completed_modules[TASK_FORMATION_NECESSITY] = True


def _write_boundary_condition_specs_to_bundle(ctx: ExperimentContext, table: pd.DataFrame) -> None:
    save_csv_with_registry(ctx, table, ctx.trial_specs_dir / "boundary_condition_specs.csv")
    ctx.completed_modules["boundary_condition_specs"] = True


def _write_access_job_specs_to_bundle(ctx: ExperimentContext, table: pd.DataFrame) -> None:
    save_csv_with_registry(ctx, table, ctx.trial_specs_dir / "access_job_specs.csv")
    ctx.completed_modules["access_job_specs"] = True


def _write_morphology_tables_to_bundle(ctx: ExperimentContext, tables: Mapping[str, pd.DataFrame]) -> None:
    mapping = {
        "morphology_item_support": ctx.raw_dir / "panel_b_morphology_item_support.csv",
        "morphology_serial_profile": ctx.metrics_dir / "panel_b_morphology_serial_profile.csv",
        "morphology_boundary_metrics": ctx.metrics_dir / "panel_c_morphology_boundary_metrics.csv",
    }
    for name, path in mapping.items():
        if name in tables:
            save_csv_with_registry(ctx, tables[name], path)
    write_morphology_fit_outputs(ctx, tables["morphology_boundary_metrics"])
    ctx.completed_modules["morphology_decomposition"] = True


def _write_weak_cue_tables_to_bundle(ctx: ExperimentContext, tables: Mapping[str, pd.DataFrame]) -> None:
    mapping = {
        "weak_cue_item_readout": ctx.raw_dir / "panel_d_weak_cue_item_readout.csv",
        "weak_cue_item_metrics": ctx.metrics_dir / "panel_d_weak_cue_item_metrics.csv",
        "item_functional_gain": ctx.metrics_dir / "panel_d_item_functional_gain.csv",
        "functional_boundary_metrics": ctx.metrics_dir / "panel_d_functional_boundary_metrics.csv",
    }
    for name, path in mapping.items():
        if name in tables:
            save_csv_with_registry(ctx, tables[name], path)
    ctx.completed_modules["weak_cue_access"] = True


def _write_cue_specificity_specs_to_bundle(ctx: ExperimentContext, table: pd.DataFrame) -> None:
    save_csv_with_registry(ctx, table, ctx.trial_specs_dir / "cue_specificity_specs.csv")
    ctx.completed_modules["cue_specificity_specs"] = True


def _write_cue_specificity_access_tables_to_bundle(ctx: ExperimentContext, tables: Mapping[str, pd.DataFrame]) -> None:
    mapping = {
        "cue_specificity_trial_readout": ctx.raw_dir / "panel_c_cue_specificity_trial_readout.csv",
        "cue_specificity_metrics": ctx.metrics_dir / "panel_c_cue_specificity_metrics.csv",
        "cue_specificity_memory_gain": ctx.metrics_dir / "panel_c_cue_specificity_memory_gain.csv",
        "cue_specificity_serial_summary": ctx.metrics_dir / "panel_c_cue_specificity_serial_summary.csv",
        "cue_specificity_contrast_summary": ctx.metrics_dir / "panel_c_cue_specificity_contrast_summary.csv",
        "cue_specificity_summary": ctx.metrics_dir / "panel_c_cue_specificity_summary.csv",
    }
    for name, path in mapping.items():
        if name in tables:
            save_csv_with_registry(ctx, tables[name], path)
    ctx.completed_modules["cue_specificity_access"] = True


def _validate_cue_specificity_science(ctx: ExperimentContext, tables: Mapping[str, pd.DataFrame]) -> None:
    metrics = tables.get("cue_specificity_metrics")
    if metrics is None or metrics.empty:
        raise ValueError("Cue specificity access produced no metrics.")
    checks = cue_specificity_scientific_checks(metrics)
    setattr(ctx, "cue_specificity_scientific_checks", checks)
    if bool(ctx.cfg.smoke):
        return
    if not bool(checks.get("target_memory_gain_matched_gt_unseen")):
        raise RuntimeError(f"Cue specificity hard gate failed: matched must exceed unseen for target memory gain. Checks={checks}")
    if not bool(checks.get("target_memory_gain_matched_gt_mismatched")):
        raise RuntimeError(f"Cue specificity hard gate failed: matched must exceed same-label mismatched foil for target memory gain. Checks={checks}")


def _write_neutral_ping_tables_to_bundle(ctx: ExperimentContext, tables: Mapping[str, pd.DataFrame]) -> None:
    mapping = {
        "neutral_ping_access_readout": ctx.raw_dir / "panel_c_neutral_ping_access_readout.csv",
        "neutral_ping_position_distribution": ctx.metrics_dir / "panel_c_neutral_ping_position_distribution.csv",
        "neutral_ping_access_summary": ctx.metrics_dir / "panel_c_neutral_ping_access_summary.csv",
    }
    for name, path in mapping.items():
        if name in tables:
            save_csv_with_registry(ctx, tables[name], path)
    ctx.completed_modules["neutral_ping_access"] = True


def _write_coupling_tables_to_bundle(ctx: ExperimentContext, tables: Mapping[str, pd.DataFrame]) -> None:
    mapping = {
        "morphology_function_coupling": ctx.metrics_dir / "panel_e_morphology_function_coupling.csv",
        "coupling_summary": ctx.metrics_dir / "panel_e_coupling_summary.csv",
        "order_specificity_control": ctx.metrics_dir / "panel_f_order_specificity_control.csv",
    }
    for name, path in mapping.items():
        if name in tables:
            save_csv_with_registry(ctx, tables[name], path)
    ctx.completed_modules["morphology_function_coupling"] = True


def _write_boundary_summary_tables_to_bundle(ctx: ExperimentContext, tables: Mapping[str, pd.DataFrame]) -> None:
    if "boundary_summary" in tables:
        save_csv_with_registry(ctx, tables["boundary_summary"], ctx.metrics_dir / "panel_f_boundary_summary.csv")
    ctx.completed_modules["boundary_summary"] = True


def _build_context(cfg: Fig3Config) -> ExperimentContext:
    seed_everything(int(cfg.network_seed))
    seed_dir = resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = prepare_seed_dirs(seed_dir, include_root_layout=True)
    device = resolve_device(cfg.device)
    dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, NUM_CLASSES)
    warnings: list[str] = []
    if Path(cfg.model_path).exists():
        net, encoder = load_model_and_encoder(cfg.model_path, device=device, dt=cfg.dt, max_duration_ms=max(cfg.sample_ms, cfg.weak_probe_ms, 100))
    elif cfg.smoke:
        seed_everything(int(cfg.network_seed))
        net = SDNN_Network(device=str(device)).to(device)
        net.eval()
        encoder = DoGSpikeEncoder(dt=cfg.dt, max_duration=max(cfg.sample_ms, cfg.weak_probe_ms, 100) * ms, device=str(device))
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
        run_log=[f"{utc_now()} start {FIGURE_ID} task runner seed={cfg.network_seed} smoke={cfg.smoke}"],
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
            return _resolve_repo_path("results/missing_fig3_smoke_model.pth")
        raise
    by_seed = {int(item.seed): item.model_path for item in checkpoints}
    if int(network_seed) not in by_seed:
        if smoke:
            return _resolve_repo_path("results/missing_fig3_smoke_model.pth")
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
    _write_config_files(ctx)
    _refresh_output_file_registry(ctx)
    summary = _write_summary(ctx)
    _apply_boundary_summary_contract(ctx, summary)
    _apply_runtime_artifact_metadata(ctx, summary)
    cue_specificity_checks = getattr(ctx, "cue_specificity_scientific_checks", None)
    if cue_specificity_checks is not None:
        summary["cue_specificity_scientific_checks"] = cue_specificity_checks
    summary.update(
        {
            "reuse_artifacts": str(mode),
            "runtime_artifact_root": str(Path(artifact_root).resolve()),
        }
    )
    write_json(summary, ctx.seed_dir / "summary.json")
    write_run_log_file(ctx)


def _apply_boundary_summary_contract(ctx: ExperimentContext, summary: dict[str, Any]) -> None:
    required = {
        "panel_b_morphology_item_support": ctx.raw_dir / "panel_b_morphology_item_support.csv",
        "panel_c_morphology_boundary_metrics": ctx.metrics_dir / "panel_c_morphology_boundary_metrics.csv",
        "panel_d_functional_boundary_metrics": ctx.metrics_dir / "panel_d_functional_boundary_metrics.csv",
        "panel_e_morphology_function_coupling": ctx.metrics_dir / "panel_e_morphology_function_coupling.csv",
        "panel_f_boundary_summary": ctx.metrics_dir / "panel_f_boundary_summary.csv",
    }
    if not any(path.exists() for path in required.values()):
        return
    missing = [str(path.relative_to(ctx.seed_dir)).replace("\\", "/") for path in required.values() if not path.exists()]
    sequence_path = ctx.trial_specs_dir / "sequence_trials.csv"
    condition_path = ctx.trial_specs_dir / "boundary_condition_specs.csv"
    access_path = ctx.trial_specs_dir / "access_job_specs.csv"
    n_sequences = 0
    n_conditions = 0
    n_access_jobs = 0
    if sequence_path.exists():
        n_sequences = int(len(pd.read_csv(sequence_path)))
    if condition_path.exists():
        n_conditions = int(len(pd.read_csv(condition_path)))
    if access_path.exists():
        n_access_jobs = int(len(pd.read_csv(access_path)))
    summary.update(
        {
            "fig3_design_version": "morphology_function_boundary_v1",
            "main_panels": {
                "A": "L1 item-wise STSP morphology profile",
                "B": "K x delay morphology boundary",
                "C": "K x delay weak-cue functional access boundary",
                "D": "item-level morphology-to-function coupling",
            },
            "main_claim_supported_fields_available": not missing,
            "missing_for_main_figure": missing,
            "new_fig3_boundary_dag": True,
            "smoke_only_engineering_validation": bool(ctx.cfg.smoke),
            "manuscript_evidence_status": "smoke_only_not_final_evidence" if ctx.cfg.smoke else "candidate_single_seed_or_full_run",
            "boundary_grid": {
                "sequence_lengths": [int(v) for v in ctx.cfg.boundary_sequence_lengths],
                "delay_ms": [int(v) for v in ctx.cfg.boundary_delay_grid_ms],
            },
            "n_sequences": n_sequences,
            "n_boundary_conditions": n_conditions,
            "n_access_jobs": n_access_jobs,
            "morphology_metrics": [
                "beta",
                "p_i",
                "N_eff",
                "N_eff_fraction",
                "multi_item_retention_index",
                "latest_collapse_index",
            ],
            "functional_metrics": [
                "G_i",
                "G_i_norm",
                "accessible_item_count",
                "singleton_access_count",
                "sequence_access_count",
                "rescued_count",
                "rescued_fraction",
            ],
            "coupling_metrics": [
                "morphology_support_p",
                "functional_gain_norm",
                "support_gain_corr",
            ],
        }
    )


def _apply_runtime_artifact_metadata(ctx: ExperimentContext, summary: dict[str, Any]) -> None:
    task_sources: dict[str, dict[str, str]] = {}
    for task_id in TASK_IDS:
        if task_id == TASK_ALL:
            continue
        source = getattr(ctx, f"{task_id}_artifact_source", None)
        if source is None:
            continue
        task_sources[str(task_id)] = {
            "source": str(source),
            "artifact_root": str(getattr(ctx, f"{task_id}_artifact_root", "")),
            "artifact_digest": str(getattr(ctx, f"{task_id}_artifact_digest", "")),
            "cache_key_digest": str(getattr(ctx, f"{task_id}_cache_key_digest", "")),
        }
    if task_sources:
        summary["runtime_artifact_sources"] = task_sources


def _mark_completed_from_existing_outputs(ctx: ExperimentContext) -> None:
    checks = {
        "sequence_trial_specs": [ctx.trial_specs_dir / "sequence_trials.csv", ctx.trial_specs_dir / "singleton_reference_trials.csv", ctx.trial_specs_dir / "partial_cue_trials.csv"],
        "state_bank": [ctx.raw_dir / "state_bank_layer1.npz", ctx.raw_dir / "state_bank_layer3.npz", ctx.raw_dir / "state_bank_manifest.csv"],
        "boundary_condition_specs": [ctx.trial_specs_dir / "boundary_condition_specs.csv"],
        "access_job_specs": [ctx.trial_specs_dir / "access_job_specs.csv"],
        "boundary_state_bank": [ctx.raw_dir / "boundary_state_bank_manifest.csv"],
        "morphology_decomposition": [ctx.raw_dir / "panel_b_morphology_item_support.csv", ctx.metrics_dir / "panel_c_morphology_boundary_metrics.csv"],
        "weak_cue_access": [ctx.raw_dir / "panel_d_weak_cue_item_readout.csv", ctx.metrics_dir / "panel_d_functional_boundary_metrics.csv"],
        "neutral_ping_access": [ctx.raw_dir / "panel_c_neutral_ping_access_readout.csv", ctx.metrics_dir / "panel_c_neutral_ping_access_summary.csv"],
        "morphology_function_coupling": [ctx.metrics_dir / "panel_e_morphology_function_coupling.csv", ctx.metrics_dir / "panel_f_order_specificity_control.csv"],
        "boundary_summary": [ctx.metrics_dir / "panel_f_boundary_summary.csv"],
        "cue_specificity_specs": [ctx.trial_specs_dir / "cue_specificity_specs.csv"],
        "cue_specificity_access": [
            ctx.raw_dir / "panel_c_cue_specificity_trial_readout.csv",
            ctx.metrics_dir / "panel_c_cue_specificity_memory_gain.csv",
            ctx.metrics_dir / "panel_c_cue_specificity_serial_summary.csv",
        ],
        TASK_FORMATION_INTERVENTION_SPECS: [
            ctx.trial_specs_dir / "formation_intervention_specs.csv",
        ],
        TASK_FORMATION_NECESSITY: [
            ctx.raw_dir / "formation_stage_readout.csv",
            ctx.raw_dir / "formation_access_readout.csv",
            ctx.metrics_dir / "formation_pair_specificity.csv",
            ctx.metrics_dir / "formation_condition_summary.csv",
        ],
        "progressive_update": [
            ctx.metrics_dir / "panel_b_progressive_update_metrics.csv",
            ctx.metrics_dir / "panel_b_prefix_trajectory_metrics.csv",
            ctx.metrics_dir / "panel_b_prefix_trajectory_summary.csv",
            ctx.metrics_dir / "panel_b_prefix_item_weights.csv",
        ],
        "peak_valley_landscape": [ctx.metrics_dir / "panel_c_example_landscape_summary.csv"],
        "neutral_ping": [ctx.raw_dir / "panel_d_neutral_ping_trial_readout.csv", ctx.metrics_dir / "panel_d_ping_summary.csv"],
        "weak_probe": [ctx.raw_dir / "panel_e_weak_probe_trial_readout.csv", ctx.metrics_dir / "panel_e_weak_probe_metrics.csv"],
        "supplement": [ctx.metrics_dir / "supp_anchor_dynamics_metrics.csv", ctx.metrics_dir / "supp_recency_only_controls.csv"],
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


def _config_from_args(args: argparse.Namespace) -> Fig3Config:
    smoke = bool(args.smoke)
    task = str(args.task)
    run_all = task == TASK_ALL
    exemplar_decoder_task = task in EXEMPLAR_DECODER_SEED_TASK_IDS
    seq_lengths = tuple(int(v) for v in str(args.sequence_lengths).split(",") if str(v).strip())
    boundary_sequence_lengths = tuple(int(v) for v in str(args.boundary_sequence_lengths).split(",") if str(v).strip())
    boundary_delay_grid_ms = tuple(int(v) for v in str(args.boundary_delay_grid_ms).split(",") if str(v).strip())
    weak_probe_keep = tuple(float(v) for v in str(args.weak_probe_keep_probs).split(",") if str(v).strip())
    weak_cue_keep = tuple(float(v) for v in str(args.weak_cue_keep_fractions).split(",") if str(v).strip())
    cue_specificity_cue_types = tuple(str(v).strip() for v in str(args.cue_specificity_cue_types).split(",") if str(v).strip())
    peak_cue_main_keep = float(args.peak_cue_main_keep_fraction)
    if not any(np.isclose(float(value), peak_cue_main_keep) for value in weak_cue_keep):
        weak_cue_keep = tuple(sorted([*weak_cue_keep, peak_cue_main_keep]))
    else:
        weak_cue_keep = tuple(sorted({float(value) for value in weak_cue_keep}))
    if smoke:
        weak_probe_keep = (0.2, 0.7)
        weak_cue_keep = (peak_cue_main_keep,)
        boundary_sequence_lengths = seq_lengths
        boundary_delay_grid_ms = (int(args.delay_ms),)
    if exemplar_decoder_task:
        seq_lengths = (EXEMPLAR_DECODER_SEQUENCE_LENGTH,)
        boundary_sequence_lengths = seq_lengths
        boundary_delay_grid_ms = (EXEMPLAR_DECODER_DELAY_MS,)
    model_path = _resolve_model_path(args.model_path, str(args.model_path_glob), int(args.network_seed), smoke=smoke)
    return Fig3Config(
        model_path=str(model_path),
        dataset_root=str(_resolve_repo_path(args.dataset_root)),
        output_root=str(_output_root_from_args(args)),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sequence_lengths=seq_lengths,
        primary_sequence_length=EXEMPLAR_DECODER_SEQUENCE_LENGTH if exemplar_decoder_task else int(args.primary_sequence_length),
        main_sequence_length=EXEMPLAR_DECODER_SEQUENCE_LENGTH if exemplar_decoder_task else int(args.main_sequence_length),
        main_only_seq_len_10=False if exemplar_decoder_task else bool(args.main_only_seq_len_10),
        sample_ms=int(args.sample_ms),
        delay_ms=EXEMPLAR_DECODER_DELAY_MS if exemplar_decoder_task else int(args.delay_ms),
        ping_ms=int(args.ping_ms),
        ping_amp=float(args.ping_amp),
        ping_repeats=1 if smoke else int(args.ping_repeats),
        ping_main_state_conditions=tuple(str(v).strip() for v in str(args.ping_main_state_conditions).split(",") if str(v).strip()),
        weak_probe_ms=int(args.weak_probe_ms),
        weak_probe_keep_probs=weak_probe_keep,
        weak_probe_repeats=min(int(args.weak_probe_repeats), 2) if smoke else int(args.weak_probe_repeats),
        weak_probe_mask_space=str(args.weak_probe_mask_space),
        weak_probe_use_same_mask_across_states=bool(args.weak_probe_use_same_mask_across_states),
        weak_probe_scale=float(args.weak_probe_scale),
        weak_probe_noise=float(args.weak_probe_noise),
        weak_probe_metric_mode=str(args.weak_probe_metric_mode),
        weak_probe_target_source=str(args.weak_probe_target_source),
        weak_probe_memory_scope=str(args.weak_probe_memory_scope),
        num_sequences=4 if smoke else int(args.num_sequences),
        batch_size=min(int(args.batch_size), 2) if smoke else int(args.batch_size),
        progressive_max_sequences=min(int(args.progressive_max_sequences), 4) if smoke else int(args.progressive_max_sequences),
        progressive_natural_decay=bool(args.progressive_natural_decay),
        formation_sequence_length=int(args.formation_sequence_length),
        formation_max_sequences=min(int(args.formation_max_sequences), 4) if smoke else int(args.formation_max_sequences),
        formation_terminal_stage=int(args.formation_terminal_stage),
        formation_mask_mode=str(args.formation_mask_mode),
        formation_attenuation=float(args.formation_attenuation),
        formation_weak_probe_keep_fraction=float(args.formation_weak_probe_keep_fraction),
        formation_weak_probe_repeats=(
            2
            if smoke
            else int(args.formation_weak_probe_repeats)
        ),
        formation_n_shuffle=min(int(args.formation_n_shuffle), 3) if smoke else int(args.formation_n_shuffle),
        peak_q=float(args.peak_q),
        valley_q=float(args.valley_q),
        n_null=8 if smoke else int(args.n_null),
        weak_cue_target_source=str(args.weak_cue_target_source),
        weak_cue_keep_fractions=weak_cue_keep,
        weak_cue_repeats=2 if smoke else int(args.weak_cue_repeats),
        weak_cue_mask_mode=str(args.weak_cue_mask_mode),
        foreground_threshold=float(args.foreground_threshold),
        functional_restore_mode=str(args.functional_restore_mode),
        partial_cue_keep_fraction=float(args.partial_cue_keep_fraction),
        partial_cue_keep_fraction_sweep=tuple(float(v) for v in str(args.partial_cue_keep_fraction_sweep).split(",") if str(v).strip()),
        partial_cue_repeats=2 if smoke else int(args.partial_cue_repeats),
        target_position=str(args.target_position),
        run_state_bank=run_all or task == TASK_STATE_BANK,
        run_progressive_update=run_all or task == TASK_PROGRESSIVE_UPDATE,
        run_peak_valley_landscape=run_all or task == TASK_PEAK_VALLEY_LANDSCAPE,
        run_neutral_ping=run_all or task == TASK_NEUTRAL_PING,
        run_weak_probe=run_all or task == TASK_WEAK_PROBE,
        run_supplement=run_all or task == TASK_SUPPLEMENT,
        save_debug_figures=bool(args.save_debug_figures),
        save_spike_cache=bool(args.save_spike_cache),
        save_all_layer_state_bank=True,
        show_progress=not bool(args.no_progress),
        use_encode_cache=not bool(args.no_encode_cache),
        enable_condition_batch=bool(args.enable_condition_batch),
        enable_state_bank_batch=bool(args.enable_state_bank_batch),
        state_bank_singleton_batch_size=max(1, int(args.state_bank_singleton_batch_size)),
        smoke=smoke,
        peak_cue_main_keep_fraction=peak_cue_main_keep,
        region_ping_q=float(args.region_ping_q),
        region_ping_support_metric=str(args.region_ping_support_metric),
        region_ping_conditions=tuple(str(v).strip() for v in str(args.region_ping_conditions).split(",") if str(v).strip()),
        region_ping_repeats=min(int(args.region_ping_repeats), 2) if smoke else int(args.region_ping_repeats),
        region_ping_amp_sweep=tuple(float(v) for v in str(args.region_ping_amp_sweep).split(",") if str(v).strip()),
        region_ping_use_random_matched=bool(args.region_ping_use_random_matched),
        weak_probe_include_singleton=bool(args.weak_probe_include_singleton),
        boundary_sequence_lengths=boundary_sequence_lengths,
        boundary_delay_grid_ms=boundary_delay_grid_ms,
        morphology_layer=str(args.morphology_layer),
        morphology_variable=str(args.morphology_variable),
        weak_cue_main_keep_prob=float(args.weak_cue_main_keep_prob),
        access_null_quantile=float(args.access_null_quantile),
        cue_specificity_seq_len=int(args.cue_specificity_seq_len),
        cue_specificity_delay_ms=int(args.cue_specificity_delay_ms),
        cue_specificity_keep_prob=float(args.cue_specificity_keep_prob),
        cue_specificity_readout_batch_size=max(1, int(args.cue_specificity_readout_batch_size)),
        cue_specificity_cue_types=cue_specificity_cue_types,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Fig.3 runtime-artifact DAG task.", allow_abbrev=False)
    parser.add_argument("--task", required=True, choices=TASK_IDS)
    parser.add_argument("--reuse-artifacts", default="auto", choices=REUSE_MODES)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--shared-sequence-root", default=None, help="Path to a shared Fig.3/Fig.6 sequence-root bank artifact or artifact root.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-path-glob", default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=str(Path(DEFAULT_OUTPUT_ROOT) / FIGURE_ID))
    parser.add_argument("--output-dir", default=None, help="Batch output root; the Fig.3 experiment id is appended unless a seed or figure root is supplied.")
    parser.add_argument("--network-seed", type=int, default=None)
    parser.add_argument("--device", default=DEFAULT_PROJECT_DEFAULTS.runtime.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--save-spike-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-encode-cache", action="store_true")
    parser.add_argument("--enable-condition-batch", action="store_true")
    parser.add_argument("--enable-state-bank-batch", action="store_true", help="Batch same-length Fig.3 state-bank sequence captures.")
    parser.add_argument(
        "--state-bank-singleton-batch-size",
        type=int,
        default=4,
        help="Maximum row batch for singleton-boundary capture inside batched Fig.3 state-bank capture.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sequence-lengths", default="3,5,7,10")
    parser.add_argument("--primary-sequence-length", type=int, default=7)
    parser.add_argument("--main-sequence-length", type=int, default=10)
    parser.add_argument("--main-only-seq-len-10", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progressive-max-sequences", type=int, default=20)
    parser.add_argument("--progressive-natural-decay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--formation-sequence-length", type=int, default=10)
    parser.add_argument("--formation-max-sequences", type=int, default=20)
    parser.add_argument("--formation-terminal-stage", type=int, default=7)
    parser.add_argument("--formation-mask-mode", default="encoded_spike", choices=["encoded_spike", "foreground"])
    parser.add_argument("--formation-attenuation", type=float, default=0.5)
    parser.add_argument("--formation-weak-probe-keep-fraction", type=float, default=0.20)
    parser.add_argument(
        "--formation-weak-probe-repeats",
        type=int,
        default=20,
    )
    parser.add_argument("--formation-n-shuffle", type=int, default=20)
    parser.add_argument("--boundary-sequence-lengths", default="3,5,7,10")
    parser.add_argument("--boundary-delay-grid-ms", default="100,200,300,400,600,800,1200,1500")
    parser.add_argument("--morphology-layer", default="layer1", choices=["layer1", "layer2", "layer3"])
    parser.add_argument("--morphology-variable", default="g", choices=["g", "u", "x"])
    parser.add_argument("--weak-cue-main-keep-prob", type=float, default=0.5)
    parser.add_argument("--access-null-quantile", type=float, default=0.95)
    parser.add_argument("--cue-specificity-seq-len", type=int, default=7)
    parser.add_argument("--cue-specificity-delay-ms", type=int, default=400)
    parser.add_argument("--cue-specificity-keep-prob", type=float, default=0.5)
    parser.add_argument("--cue-specificity-readout-batch-size", type=int, default=6)
    parser.add_argument("--cue-specificity-cue-types", default="matched,mismatched,unseen")
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--ping-ms", type=int, default=30)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--ping-repeats", type=int, default=1)
    parser.add_argument("--ping-main-state-conditions", default="S_final,S0")
    parser.add_argument("--weak-probe-ms", type=int, default=100)
    parser.add_argument("--weak-probe-keep-probs", default="0.05,0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    parser.add_argument("--weak-probe-repeats", type=int, default=20)
    parser.add_argument("--weak-probe-mask-space", default="encoded_spikes", choices=["encoded_spikes"])
    parser.add_argument("--weak-probe-use-same-mask-across-states", dest="weak_probe_use_same_mask_across_states", action="store_true", default=True)
    parser.add_argument("--weak-probe-independent-masks-across-states", dest="weak_probe_use_same_mask_across_states", action="store_false")
    parser.add_argument("--weak-probe-scale", type=float, default=0.35)
    parser.add_argument("--weak-probe-noise", type=float, default=0.0)
    parser.add_argument("--weak-probe-metric-mode", default="fig2_compat", choices=["fig2_compat", "legacy"])
    parser.add_argument("--weak-probe-target-source", default="sequence_member_random", choices=["sequence_member_random", "unseen_random", "both"])
    parser.add_argument("--weak-probe-memory-scope", default="final_only", choices=["final_only", "all_prefixes"])
    parser.add_argument("--weak-probe-include-singleton", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--region-ping-q", type=float, default=0.20)
    parser.add_argument("--region-ping-support-metric", default="gain_ratio_map", choices=["gain_ratio_map", "delta_gain_map", "G_final"])
    parser.add_argument("--region-ping-conditions", default="peak,valley,random")
    parser.add_argument("--region-ping-repeats", type=int, default=5)
    parser.add_argument("--region-ping-amp-sweep", default="0.25,0.5,1.0,1.5")
    parser.add_argument("--region-ping-use-random-matched", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-sequences", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--peak-q", type=float, default=0.20)
    parser.add_argument("--valley-q", type=float, default=0.20)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--weak-cue-target-source", default="sequence_member_random", choices=["sequence_member_random", "unseen_random", "both"])
    parser.add_argument("--weak-cue-keep-fractions", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--peak-cue-main-keep-fraction", type=float, default=0.10)
    parser.add_argument("--weak-cue-repeats", type=int, default=10)
    parser.add_argument("--weak-cue-mask-mode", default="rank_within_target_foreground", choices=["rank_within_target_foreground"])
    parser.add_argument("--foreground-threshold", type=float, default=0.1)
    parser.add_argument("--functional-restore-mode", choices=["full_boundary", "stsp_only", "stsp_only_legacy_current_ux"], default="stsp_only")
    parser.add_argument("--partial-cue-keep-fraction", type=float, default=0.10)
    parser.add_argument("--partial-cue-keep-fraction-sweep", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--partial-cue-repeats", type=int, default=20)
    parser.add_argument("--target-position", default="K-1")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.task != TASK_EXEMPLAR_DECODER_SUMMARY and args.network_seed is None:
        parser.error("--network-seed is required unless --task exemplar_decoder_summary is selected.")
    if args.task in EXEMPLAR_DECODER_SEED_TASK_IDS and args.device != "cuda":
        parser.error("Fig.3 exemplar decoder state acquisition requires --device cuda; CPU and auto are not permitted.")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
