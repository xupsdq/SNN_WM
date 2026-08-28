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

    raise SystemExit(_final_statistics_main("fig4", _early_args))

import argparse
import sys
from dataclasses import asdict
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
    prepare_seed_dirs,
    relative_to_root,
    resolve_seed_dir,
    save_csv_with_registry,
)
from src.experiments.paper_figures.fig4.artifacts import (
    cache_key_matches,
    copy_rollout_artifact_npz_to_raw,
    default_artifact_root,
    load_pair_sampling_artifact,
    load_rollout_bank_artifact,
    load_similarity_entry_artifact,
    save_pair_sampling_artifact,
    save_rollout_bank_artifact,
    save_similarity_entry_artifact,
    task_artifact_dir,
    write_json,
)
from src.experiments.paper_figures.fig4.cache_keys import (
    build_pair_sampling_cache_key,
    build_rollouts_cache_key,
    build_similarity_entry_cache_key,
    cache_key_digest,
    pair_sampling_hash,
)
from src.experiments.paper_figures.fig4.constants import FIGURE_ID, NUM_CLASSES
from src.experiments.paper_figures.fig4.schemas import (
    REUSE_MODES,
    TASK_ALL,
    TASK_DECISION_DEFLECTION,
    TASK_DECISION_SPIKE_DISPLACEMENT,
    TASK_IDS,
    TASK_OVERLAP_ACCURACY_IDENTIFICATION,
    TASK_OVERLAP_LOCALIZATION,
    TASK_OVERLAP_PERTURBATION,
    TASK_PAIR_SAMPLING,
    TASK_ROLLOUTS,
    TASK_SIMILARITY_ENTRY,
    TASK_SUPPLEMENT,
    normalize_reuse_mode,
)
from src.experiments.paper_figures.fig4.subexperiments.decision_deflection import (
    compute_decision_deflection_metrics,
    compute_l3_accumulator_region_replay_metrics_from_bank,
    compute_l3_accumulator_region_replay_metrics,
)
from src.experiments.paper_figures.fig4.subexperiments.decision_spike_displacement import compute_probe_l3_trace_dpi_metrics
from src.experiments.paper_figures.fig4.subexperiments.overlap_accuracy_identification import compute_overlap_accuracy_identification
from src.experiments.paper_figures.fig4.subexperiments.overlap_localization import compute_overlap_localization_metrics
from src.experiments.paper_figures.fig4.subexperiments.overlap_perturbation import (
    compute_l1_stsp_overlap_perturbation_outputs,
    compute_overlap_preserving_perturbation_metrics,
)
from src.experiments.paper_figures.fig4.subexperiments.pair_sampling import build_pair_trials
from src.experiments.paper_figures.fig4.subexperiments.helpers_1 import (
    _bvec_summary,
    _cti_summary,
    _panel_b_accuracy_drop_summary,
    _summary_by_bin,
    _write_panel_a_example,
)
from src.experiments.paper_figures.fig4.subexperiments.output_contract import (
    _write_config_files,
    _write_summary,
    utc_now,
    write_run_log_file,
)
from src.experiments.paper_figures.fig4.subexperiments.rollouts import (
    run_overlap_perturbation_compatible_rollouts,
    run_similarity_bias_compatible_trials,
)
from src.experiments.paper_figures.fig4.subexperiments.supplement import (
    compute_supplement_outputs,
    write_fig4_panel_aliases_and_supplement_aliases,
)
from src.experiments.paper_figures.fig4.types import (
    ExperimentContext,
    Fig4Config,
    OverlapPerturbationCompatibleBank,
    SimilarityBiasCompatibleBank,
)
from src.experiments.paper_figures.run_paper_figures import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH_GLOB,
    DEFAULT_OUTPUT_ROOT,
    discover_checkpoints,
)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--task" in raw_argv and "final-statistics" in raw_argv:
        from src.experiments.paper_figures.final_six.pipeline import canonical_runner_main

        return canonical_runner_main("fig4", raw_argv)
    args = _parse_args(argv)
    mode = normalize_reuse_mode(args.reuse_artifacts)
    cfg = _config_from_args(args)
    ctx = _build_context(cfg)
    artifact_root = _artifact_root_from_args(args, ctx.seed_dir)
    run_info = build_run_info(
        experiment_name=f"{FIGURE_ID}.{args.task}",
        output_dir=ctx.seed_dir,
        entry_script="src.experiments.paper_figures.fig4.run_task",
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
        _write_config_files(ctx)
        pair_trials, _candidate_pool, _overlap_matched, perturbation_masks, mask_bank, pair_hash = _get_pair_sampling(
            ctx,
            mode=mode,
            artifact_root=artifact_root,
        )
        run_info["pair_sampling_hash"] = pair_hash
        _run_task(
            ctx,
            pair_trials,
            perturbation_masks,
            mask_bank,
            pair_hash=pair_hash,
            task_id=str(args.task),
            mode=mode,
            artifact_root=artifact_root,
        )
        _finalize_bundle(ctx, artifact_root=artifact_root, mode=mode)
        finalize_run_info(ctx.seed_dir / "meta", run_info, status="success")
        return 0
    except Exception:
        finalize_run_info(ctx.seed_dir / "meta", run_info, status="failed")
        raise


def _get_pair_sampling(
    ctx: ExperimentContext,
    *,
    mode: str,
    artifact_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, dict[str, np.ndarray]], str]:
    task_dir = task_artifact_dir(artifact_root, TASK_PAIR_SAMPLING)
    expected_key = build_pair_sampling_cache_key(ctx.cfg)
    if mode == "require":
        artifact = load_pair_sampling_artifact(task_dir, expected_key=expected_key)
        _write_pair_sampling_to_bundle(
            ctx,
            artifact.pair_trials,
            artifact.candidate_pool,
            artifact.overlap_matched_pairs,
            artifact.perturbation_masks,
            artifact.mask_bank,
        )
        _set_artifact_metadata(ctx, "pair_sampling", "loaded", task_dir, artifact.digest, expected_key)
        return (
            artifact.pair_trials,
            artifact.candidate_pool,
            artifact.overlap_matched_pairs,
            artifact.perturbation_masks,
            artifact.mask_bank,
            artifact.digest,
        )
    if mode == "auto" and cache_key_matches(task_dir, expected_key):
        artifact = load_pair_sampling_artifact(task_dir, expected_key=expected_key)
        _write_pair_sampling_to_bundle(
            ctx,
            artifact.pair_trials,
            artifact.candidate_pool,
            artifact.overlap_matched_pairs,
            artifact.perturbation_masks,
            artifact.mask_bank,
        )
        _set_artifact_metadata(ctx, "pair_sampling", "loaded", task_dir, artifact.digest, expected_key)
        return (
            artifact.pair_trials,
            artifact.candidate_pool,
            artifact.overlap_matched_pairs,
            artifact.perturbation_masks,
            artifact.mask_bank,
            artifact.digest,
        )
    pair_trials, candidate_pool, perturbation_masks, mask_bank = build_pair_trials(ctx)
    overlap_path = ctx.trial_specs_dir / "overlap_matched_pairs.csv"
    if not overlap_path.exists():
        raise FileNotFoundError(f"Fig.4 pair sampling did not write overlap matched pairs: {overlap_path}")
    overlap_matched = _read_overlap_matched_pairs(overlap_path)
    if mode != "off":
        artifact = save_pair_sampling_artifact(
            task_dir,
            pair_trials=pair_trials,
            candidate_pool=candidate_pool,
            overlap_matched_pairs=overlap_matched,
            perturbation_masks=perturbation_masks,
            mask_bank=mask_bank,
            cache_key=expected_key,
            network_seed=ctx.cfg.network_seed,
        )
        _write_pair_sampling_to_bundle(
            ctx,
            artifact.pair_trials,
            artifact.candidate_pool,
            artifact.overlap_matched_pairs,
            artifact.perturbation_masks,
            artifact.mask_bank,
        )
        _set_artifact_metadata(ctx, "pair_sampling", "built", task_dir, artifact.digest, expected_key)
        ctx.run_log.append(f"{utc_now()} pair_sampling source=built artifact={task_dir}")
        return (
            artifact.pair_trials,
            artifact.candidate_pool,
            artifact.overlap_matched_pairs,
            artifact.perturbation_masks,
            artifact.mask_bank,
            artifact.digest,
        )
    pair_hash = pair_sampling_hash(pair_trials, candidate_pool, overlap_matched, perturbation_masks, mask_bank)
    _set_artifact_metadata(ctx, "pair_sampling", "fresh", task_dir, pair_hash, expected_key)
    return pair_trials, candidate_pool, overlap_matched, perturbation_masks, mask_bank, pair_hash


def _run_task(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    perturbation_masks: pd.DataFrame,
    mask_bank: dict[int, dict[str, np.ndarray]],
    *,
    pair_hash: str,
    task_id: str,
    mode: str,
    artifact_root: Path,
) -> None:
    if task_id == TASK_PAIR_SAMPLING:
        return
    if task_id == TASK_SIMILARITY_ENTRY:
        _get_similarity_entry(ctx, pair_trials, pair_hash=pair_hash, mode=mode, artifact_root=artifact_root)
        return
    if task_id == TASK_ROLLOUTS:
        _get_rollouts(ctx, pair_trials, perturbation_masks, mask_bank, pair_hash=pair_hash, mode=mode, artifact_root=artifact_root)
        return
    if task_id == TASK_OVERLAP_LOCALIZATION:
        _get_similarity_entry(ctx, pair_trials, pair_hash=pair_hash, mode=mode, artifact_root=artifact_root)
        overlap_bank = _get_rollouts(ctx, pair_trials, perturbation_masks, mask_bank, pair_hash=pair_hash, mode=mode, artifact_root=artifact_root)
        compute_overlap_localization_metrics(ctx, overlap_bank)
        return
    if task_id == TASK_OVERLAP_ACCURACY_IDENTIFICATION:
        similarity_bank = _get_similarity_entry(ctx, pair_trials, pair_hash=pair_hash, mode=mode, artifact_root=artifact_root)
        compute_overlap_accuracy_identification(ctx, similarity_bank)
        return
    if task_id == TASK_DECISION_SPIKE_DISPLACEMENT:
        overlap_bank = _get_rollouts(ctx, pair_trials, perturbation_masks, mask_bank, pair_hash=pair_hash, mode=mode, artifact_root=artifact_root)
        compute_probe_l3_trace_dpi_metrics(ctx, overlap_bank)
        return
    if task_id == TASK_DECISION_DEFLECTION:
        if mode == "off":
            _compute_l3_accumulator_region_replay_fresh(ctx, pair_trials)
            return
        overlap_bank = _get_rollouts(ctx, pair_trials, perturbation_masks, mask_bank, pair_hash=pair_hash, mode=mode, artifact_root=artifact_root)
        compute_l3_accumulator_region_replay_metrics_from_bank(ctx, overlap_bank)
        return
    if task_id == TASK_OVERLAP_PERTURBATION:
        overlap_bank = _get_rollouts(ctx, pair_trials, perturbation_masks, mask_bank, pair_hash=pair_hash, mode=mode, artifact_root=artifact_root)
        compute_decision_deflection_metrics(ctx, overlap_bank)
        compute_overlap_preserving_perturbation_metrics(ctx, overlap_bank)
        compute_l1_stsp_overlap_perturbation_outputs(ctx, pair_trials, mask_bank)
        return
    if task_id == TASK_SUPPLEMENT:
        overlap_bank = _get_rollouts(ctx, pair_trials, perturbation_masks, mask_bank, pair_hash=pair_hash, mode=mode, artifact_root=artifact_root)
        compute_decision_deflection_metrics(ctx, overlap_bank)
        compute_supplement_outputs(ctx, overlap_bank)
        return
    if task_id == TASK_ALL:
        similarity_bank = _get_similarity_entry(ctx, pair_trials, pair_hash=pair_hash, mode=mode, artifact_root=artifact_root)
        overlap_bank = _get_rollouts(ctx, pair_trials, perturbation_masks, mask_bank, pair_hash=pair_hash, mode=mode, artifact_root=artifact_root)
        compute_overlap_localization_metrics(ctx, overlap_bank)
        compute_overlap_accuracy_identification(ctx, similarity_bank)
        compute_probe_l3_trace_dpi_metrics(ctx, overlap_bank)
        if mode == "off":
            _compute_l3_accumulator_region_replay_fresh(ctx, pair_trials)
        else:
            compute_l3_accumulator_region_replay_metrics_from_bank(ctx, overlap_bank)
        compute_decision_deflection_metrics(ctx, overlap_bank)
        compute_overlap_preserving_perturbation_metrics(ctx, overlap_bank)
        compute_l1_stsp_overlap_perturbation_outputs(ctx, pair_trials, mask_bank)
        compute_supplement_outputs(ctx, overlap_bank)
        return
    raise ValueError(f"Unsupported Fig.4 task: {task_id}")


def _compute_l3_accumulator_region_replay_fresh(ctx: ExperimentContext, pair_trials: pd.DataFrame) -> None:
    compute_l3_accumulator_region_replay_metrics(ctx, pair_trials)


def _get_similarity_entry(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    *,
    pair_hash: str,
    mode: str,
    artifact_root: Path,
) -> SimilarityBiasCompatibleBank:
    task_dir = task_artifact_dir(artifact_root, TASK_SIMILARITY_ENTRY)
    expected_key = build_similarity_entry_cache_key(ctx.cfg, pair_hash=pair_hash)
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_similarity_entry_artifact(task_dir, expected_key=expected_key, pair_trials=pair_trials)
        _write_similarity_entry_to_bundle(ctx, artifact.bank)
        _set_artifact_metadata(ctx, "similarity_entry", "loaded", task_dir, artifact.digest, expected_key)
        ctx.completed_modules["similarity_entry"] = True
        return artifact.bank
    if mode == "require":
        artifact = load_similarity_entry_artifact(task_dir, expected_key=expected_key, pair_trials=pair_trials)
        _write_similarity_entry_to_bundle(ctx, artifact.bank)
        _set_artifact_metadata(ctx, "similarity_entry", "loaded", task_dir, artifact.digest, expected_key)
        ctx.completed_modules["similarity_entry"] = True
        return artifact.bank
    bank = run_similarity_bias_compatible_trials(ctx, pair_trials)
    if mode != "off":
        artifact = save_similarity_entry_artifact(task_dir, bank, cache_key=expected_key)
        _write_similarity_entry_to_bundle(ctx, artifact.bank)
        _set_artifact_metadata(ctx, "similarity_entry", "built", task_dir, artifact.digest, expected_key)
        return artifact.bank
    _set_artifact_metadata(ctx, "similarity_entry", "fresh", task_dir, "", expected_key)
    return bank


def _get_rollouts(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    perturbation_masks: pd.DataFrame,
    mask_bank: dict[int, dict[str, np.ndarray]],
    *,
    pair_hash: str,
    mode: str,
    artifact_root: Path,
) -> OverlapPerturbationCompatibleBank:
    task_dir = task_artifact_dir(artifact_root, TASK_ROLLOUTS)
    expected_key = build_rollouts_cache_key(ctx.cfg, pair_hash=pair_hash)
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_rollout_bank_artifact(task_dir, expected_key=expected_key, pair_trials=pair_trials)
        _write_rollouts_to_bundle(ctx, artifact.bank, task_dir=task_dir)
        _set_artifact_metadata(ctx, "rollouts", "loaded", task_dir, artifact.digest, expected_key)
        ctx.completed_modules["rollouts"] = True
        return artifact.bank
    if mode == "require":
        artifact = load_rollout_bank_artifact(task_dir, expected_key=expected_key, pair_trials=pair_trials)
        _write_rollouts_to_bundle(ctx, artifact.bank, task_dir=task_dir)
        _set_artifact_metadata(ctx, "rollouts", "loaded", task_dir, artifact.digest, expected_key)
        ctx.completed_modules["rollouts"] = True
        return artifact.bank
    bank = run_overlap_perturbation_compatible_rollouts(ctx, pair_trials, perturbation_masks, mask_bank)
    if mode != "off":
        artifact = save_rollout_bank_artifact(task_dir, bank, raw_dir=ctx.raw_dir, cache_key=expected_key)
        _write_rollouts_to_bundle(ctx, artifact.bank, task_dir=task_dir)
        _set_artifact_metadata(ctx, "rollouts", "built", task_dir, artifact.digest, expected_key)
        return artifact.bank
    _set_artifact_metadata(ctx, "rollouts", "fresh", task_dir, "", expected_key)
    return bank


def _write_pair_sampling_to_bundle(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    overlap_matched: pd.DataFrame,
    perturbation_masks: pd.DataFrame,
    mask_bank: Mapping[int, Mapping[str, np.ndarray]],
) -> None:
    save_csv_with_registry(ctx, pair_trials, ctx.trial_specs_dir / "pair_trials.csv")
    save_csv_with_registry(ctx, candidate_pool, ctx.trial_specs_dir / "pair_candidate_pool.csv")
    save_csv_with_registry(ctx, overlap_matched, ctx.trial_specs_dir / "overlap_matched_pairs.csv")
    save_csv_with_registry(ctx, perturbation_masks, ctx.trial_specs_dir / "perturbation_masks.csv")
    if not pair_trials.empty:
        image_ids = sorted(
            {
                int(value)
                for value in pair_trials[["sample_image_id", "probe_image_id"]].to_numpy().ravel()
            }
        )
        max_image_id = max(image_ids)
        images = torch.stack(
            [ctx.dataset[idx][0].detach().cpu().to(torch.float32) for idx in range(max_image_id + 1)],
            dim=0,
        )
        _write_panel_a_example(ctx, pair_trials, mask_bank, images)
    ctx.completed_modules["pair_sampling"] = True
    ctx.n_pairs = int(len(pair_trials))


def _read_overlap_matched_pairs(path: Path) -> pd.DataFrame:
    columns = [
        "network_seed",
        "matched_group_id",
        "high_pair_id",
        "low_pair_id",
        "similarity_difference",
        "energy_difference",
        "class_pair_matched",
        "overlap_difference",
    ]
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def _write_similarity_entry_to_bundle(ctx: ExperimentContext, bank: SimilarityBiasCompatibleBank) -> None:
    trial_df = bank.trial_metrics.copy()
    repeat_df = bank.repeat_metrics.copy()
    save_csv_with_registry(ctx, trial_df, ctx.metrics_dir / "panel_b_similarity_entry_metrics.csv")
    save_csv_with_registry(ctx, _summary_by_bin(trial_df, "similarity_bin", "pixel_similarity"), ctx.metrics_dir / "panel_b_similarity_bin_summary.csv")
    save_csv_with_registry(ctx, _panel_b_accuracy_drop_summary(trial_df), ctx.metrics_dir / "panel_b_similarity_accuracy_drop_summary.csv")
    save_csv_with_registry(ctx, _bvec_summary(trial_df), ctx.metrics_dir / "supp_similarity_bvec_summary.csv")
    save_csv_with_registry(ctx, _cti_summary(trial_df), ctx.metrics_dir / "supp_similarity_cti_summary.csv")
    save_csv_with_registry(ctx, repeat_df, ctx.raw_dir / "similarity_bias_repeat_metrics.csv")
    np.savez_compressed(ctx.raw_dir / "similarity_bias_voltage_vectors.npz", **bank.voltage_vectors)
    ctx.output_files["similarity_bias_voltage_vectors"] = "data/raw/similarity_bias_voltage_vectors.npz"
    ctx.completed_modules["similarity_entry"] = True


def _write_rollouts_to_bundle(ctx: ExperimentContext, bank: OverlapPerturbationCompatibleBank, *, task_dir: Path) -> None:
    save_csv_with_registry(ctx, bank.rollout_manifest, ctx.raw_dir / "overlap_perturbation_rollout_manifest.csv")
    save_csv_with_registry(ctx, bank.rollout_manifest, ctx.raw_dir / "rollout_manifest.csv")
    if not bank.l3_replay_capture_manifest.empty:
        save_csv_with_registry(ctx, bank.l3_replay_capture_manifest, ctx.raw_dir / "l3_replay_capture_manifest.csv")
    save_csv_with_registry(ctx, bank.perturbation_masks, ctx.metrics_dir / "supp_overlap_mask_application_audit.csv")
    copy_rollout_artifact_npz_to_raw(task_dir, ctx.raw_dir)
    for stem in ("probe_trace_arrays_l1", "probe_trace_arrays_l2", "probe_trace_arrays_l3", "readout_trajectory_vectors", "l3_replay_capture_arrays"):
        path = ctx.raw_dir / f"{stem}.npz"
        if path.exists():
            ctx.output_files[stem] = relative_to_root(path, ctx.seed_dir)
    ctx.output_files["overlap_perturbation_rollout_manifest"] = "data/raw/overlap_perturbation_rollout_manifest.csv"
    ctx.completed_modules["rollouts"] = True


def _build_context(cfg: Fig4Config) -> ExperimentContext:
    seed_everything(int(cfg.network_seed))
    seed_dir = resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = prepare_seed_dirs(seed_dir, include_root_layout=True)
    device = resolve_device(cfg.device)
    dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, NUM_CLASSES)
    warnings: list[str] = []
    max_duration_ms = max(cfg.sample_ms, cfg.probe_ms, 100)
    if Path(cfg.model_path).exists():
        net, encoder = load_model_and_encoder(
            cfg.model_path,
            device=device,
            dt=cfg.dt,
            max_duration_ms=max_duration_ms,
        )
    elif cfg.smoke:
        seed_everything(int(cfg.network_seed))
        net = SDNN_Network(device=str(device)).to(device)
        net.eval()
        encoder = DoGSpikeEncoder(dt=cfg.dt, max_duration=max_duration_ms * ms, device=str(device))
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
            return _resolve_repo_path("results/missing_fig4_smoke_model.pth")
        raise
    by_seed = {int(item.seed): item.model_path for item in checkpoints}
    if int(network_seed) not in by_seed:
        if smoke:
            return _resolve_repo_path("results/missing_fig4_smoke_model.pth")
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
    write_fig4_panel_aliases_and_supplement_aliases(ctx)
    _normalize_optional_empty_tables(ctx)
    _refresh_output_file_registry(ctx)
    summary = _write_summary(ctx)
    summary.update(
        {
            "reuse_artifacts": str(mode),
            "runtime_artifact_root": str(Path(artifact_root).resolve()),
        }
    )
    write_json(summary, ctx.seed_dir / "summary.json")
    write_run_log_file(ctx)


def _normalize_optional_empty_tables(ctx: ExperimentContext) -> None:
    path = ctx.metrics_dir / "supp_similarity_bin_full_stats.csv"
    if not path.exists():
        return
    try:
        pd.read_csv(path)
    except pd.errors.EmptyDataError:
        save_csv_with_registry(
            ctx,
            pd.DataFrame(
                columns=[
                    "network_seed",
                    "similarity_bin",
                    "mean_pixel_similarity",
                    "mean_acc_drop",
                    "mean_drop_event",
                    "mean_b_vec",
                    "mean_DPI_L3",
                    "n_pairs",
                ]
            ),
            path,
        )


def _mark_completed_from_existing_outputs(ctx: ExperimentContext) -> None:
    checks = {
        "pair_sampling": [ctx.trial_specs_dir / "pair_trials.csv", ctx.trial_specs_dir / "pair_candidate_pool.csv", ctx.trial_specs_dir / "perturbation_masks.csv"],
        "similarity_entry": [
            ctx.metrics_dir / "panel_b_similarity_entry_metrics.csv",
            ctx.metrics_dir / "panel_b_similarity_bin_summary.csv",
            ctx.metrics_dir / "panel_b_similarity_accuracy_drop_summary.csv",
        ],
        "rollouts": [ctx.raw_dir / "overlap_perturbation_rollout_manifest.csv", ctx.raw_dir / "probe_trace_arrays_l3.npz", ctx.raw_dir / "readout_trajectory_vectors.npz"],
        "overlap_localization": [ctx.metrics_dir / "panel_c_overlap_localization_metrics.csv", ctx.metrics_dir / "panel_c_overlap_matched_comparison.csv"],
        "overlap_accuracy_identification": [
            ctx.metrics_dir / "panel_c_high_similarity_overlap_accuracy_drop.csv",
            ctx.metrics_dir / "panel_d_overlap_accuracy_pair_table.csv",
            ctx.metrics_dir / "supp_overlap_excess_accuracy_metrics.csv",
        ],
        "decision_spike_displacement": [ctx.metrics_dir / "panel_e_time_resolved_l3_displacement.csv", ctx.metrics_dir / "panel_e_decision_spike_displacement.csv"],
        "decision_deflection": [ctx.metrics_dir / "panel_f_l3_accumulator_region_replay_metrics.csv", ctx.metrics_dir / "panel_f_l3_accumulator_summary.csv"],
        "overlap_perturbation": [
            ctx.raw_dir / "panel_d_l1_stsp_overlap_perturbation_trial_readout.csv",
            ctx.metrics_dir / "panel_d_overlap_perturbation_summary.csv",
        ],
        "supplement": [ctx.metrics_dir / "supp_alternative_overlap_definitions.csv", ctx.metrics_dir / "supp_class_pair_breakdown.csv"],
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


def _config_from_args(args: argparse.Namespace) -> Fig4Config:
    smoke = bool(args.smoke)
    task = str(args.task)
    run_all = task == TASK_ALL
    delay_ms = int(args.delay_ms)
    if bool(args.legacy_exact_mode):
        delay_ms = 500
    model_path = _resolve_model_path(args.model_path, str(args.model_path_glob), int(args.network_seed), smoke=smoke)
    return Fig4Config(
        model_path=str(model_path),
        dataset_root=str(_resolve_repo_path(args.dataset_root)),
        output_root=str(_output_root_from_args(args)),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sample_ms=int(args.sample_ms),
        delay_ms=delay_ms,
        probe_ms=int(args.probe_ms),
        batch_size=min(int(args.batch_size), 4) if smoke else int(args.batch_size),
        max_pairs=16 if smoke else int(args.max_pairs),
        num_similarity_bins=int(args.num_similarity_bins),
        num_overlap_bins=int(args.num_overlap_bins),
        overlap_mask_mode=str(args.overlap_mask_mode),
        foreground_threshold=float(args.foreground_threshold),
        dilation_radius=int(args.dilation_radius),
        random_mask_candidates=min(int(args.random_mask_candidates), 8) if smoke else int(args.random_mask_candidates),
        n_null=8 if smoke else int(args.n_null),
        save_full_trace=bool(args.save_full_trace),
        save_l3_trace=not bool(args.no_save_l3_trace),
        run_pair_sampling=True,
        run_rollouts=run_all or task in {
            TASK_ROLLOUTS,
            TASK_OVERLAP_LOCALIZATION,
            TASK_DECISION_SPIKE_DISPLACEMENT,
            TASK_DECISION_DEFLECTION,
            TASK_OVERLAP_PERTURBATION,
            TASK_SUPPLEMENT,
        },
        run_similarity_entry=run_all or task in {TASK_SIMILARITY_ENTRY, TASK_OVERLAP_ACCURACY_IDENTIFICATION},
        run_overlap_localization=run_all or task == TASK_OVERLAP_LOCALIZATION,
        run_overlap_accuracy_identification=run_all or task == TASK_OVERLAP_ACCURACY_IDENTIFICATION,
        run_decision_spike_displacement=run_all or task == TASK_DECISION_SPIKE_DISPLACEMENT,
        run_decision_deflection=run_all or task == TASK_DECISION_DEFLECTION,
        run_overlap_perturbation=run_all or task == TASK_OVERLAP_PERTURBATION,
        run_supplement=run_all or task == TASK_SUPPLEMENT,
        num_iso_similarity_bins=5 if smoke else int(args.num_iso_similarity_bins),
        overlap_tail_quantile=float(args.overlap_tail_quantile),
        match_similarity_caliper=float(args.match_similarity_caliper),
        match_energy_caliper=float(args.match_energy_caliper),
        match_require_probe_label=bool(args.match_require_probe_label),
        match_require_class_pair=bool(args.match_require_class_pair),
        require_distinct_pair_labels=bool(args.require_distinct_pair_labels),
        min_matches_per_network=int(args.min_matches_per_network),
        n_match_permutations=100 if smoke else int(args.n_match_permutations),
        save_debug_figures=bool(args.save_debug_figures),
        show_progress=not bool(args.no_progress),
        enable_condition_batch=bool(args.enable_condition_batch),
        use_legacy_similarity_bias_method=bool(args.use_legacy_similarity_bias_method),
        use_legacy_overlap_perturbation_method=bool(args.use_legacy_overlap_perturbation_method),
        use_legacy_l3_accumulator_method=bool(args.use_legacy_l3_accumulator_method),
        legacy_exact_mode=bool(args.legacy_exact_mode),
        l3_mask_mode=str(args.l3_mask_mode),
        l3_region_batch_size=min(int(args.l3_region_batch_size), 8) if smoke else int(args.l3_region_batch_size),
        temporal_pool=str(args.temporal_pool),
        save_case_count=min(int(args.save_case_count), 2) if smoke else int(args.save_case_count),
        run_l3_region_deletion=bool(args.run_l3_region_deletion),
        run_l3_region_replacement=bool(args.run_l3_region_replacement),
        smoke=smoke,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Fig.4 runtime-artifact DAG task.", allow_abbrev=False)
    parser.add_argument("--task", required=True, choices=TASK_IDS)
    parser.add_argument("--reuse-artifacts", default="auto", choices=REUSE_MODES)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-path-glob", default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=str(Path(DEFAULT_OUTPUT_ROOT) / FIGURE_ID))
    parser.add_argument("--output-dir", default=None, help="Batch output root; the Fig.4 experiment id is appended unless a seed or figure root is supplied.")
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default=DEFAULT_PROJECT_DEFAULTS.runtime.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--enable-condition-batch", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=400)
    parser.add_argument("--probe-ms", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--num-similarity-bins", type=int, default=5)
    parser.add_argument("--num-overlap-bins", type=int, default=3)
    parser.add_argument("--overlap-mask-mode", default="encoded_spike", choices=["encoded_spike", "foreground"])
    parser.add_argument("--foreground-threshold", type=float, default=0.1)
    parser.add_argument("--dilation-radius", type=int, default=1)
    parser.add_argument("--random-mask-candidates", type=int, default=32)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--num-iso-similarity-bins", type=int, default=20)
    parser.add_argument("--overlap-tail-quantile", type=float, default=0.33)
    parser.add_argument("--match-similarity-caliper", type=float, default=0.02)
    parser.add_argument("--match-energy-caliper", type=float, default=0.15)
    parser.add_argument("--match-require-probe-label", action="store_true")
    parser.add_argument("--match-require-class-pair", action="store_true")
    parser.add_argument("--require-distinct-pair-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-matches-per-network", type=int, default=20)
    parser.add_argument("--n-match-permutations", type=int, default=2000)
    parser.add_argument("--save-full-trace", action="store_true")
    parser.add_argument("--no-save-l3-trace", action="store_true")
    parser.add_argument("--legacy-exact-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-legacy-similarity-bias-method", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-legacy-overlap-perturbation-method", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-legacy-l3-accumulator-method", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l3-mask-mode", choices=["1x1", "2x2"], default="1x1")
    parser.add_argument("--l3-region-batch-size", type=int, default=16)
    parser.add_argument("--temporal-pool", choices=["mean"], default="mean")
    parser.add_argument("--save-case-count", type=int, default=4)
    parser.add_argument("--run-l3-region-deletion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-l3-region-replacement", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
