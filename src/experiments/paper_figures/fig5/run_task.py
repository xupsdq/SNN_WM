from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS
from src.experiments.common.dataset import build_class_index
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.paper_figures import fig5_local_support_competition_experiment as legacy
from src.experiments.paper_figures.fig5.artifacts import (
    cache_key_matches,
    copy_probe_stsp_update_artifact_to_bundle,
    copy_support_bank_tables_to_bundle,
    copy_trial_npz_to_raw,
    default_artifact_root,
    load_probe_stsp_update_artifact,
    load_support_bank_artifact,
    load_support_bank_metadata_artifact,
    load_trial_sampling_artifact,
    read_cache_key,
    read_probe_stsp_update_unit_groups,
    save_support_bank_artifact,
    save_trial_sampling_artifact,
    task_artifact_dir,
    write_json,
)
from src.experiments.paper_figures.fig5.cache_keys import (
    build_probe_stsp_update_bank_cache_key,
    build_support_bank_cache_key,
    build_trial_sampling_cache_key,
    cache_key_digest,
    trials_hash,
)
from src.experiments.paper_figures.fig5.schemas import (
    REUSE_MODES,
    TASK_ALL,
    TASK_EARLY_FIRING,
    TASK_IDS,
    TASK_LOCAL_EVENTS,
    TASK_POSTPROBE_STSP_UPDATE,
    TASK_PREPROBE_SUPPORT,
    TASK_PROBE_STSP_UPDATE_BANK,
    TASK_SUPPORT_BANK,
    TASK_SUPPORT_PERTURBATION,
    TASK_SUPPLEMENT,
    TASK_TRIAL_SAMPLING,
    normalize_reuse_mode,
)
from src.experiments.paper_figures.fig5.subexperiments.early_firing import compute_early_firing_transition_metrics
from src.experiments.paper_figures.fig5.subexperiments.local_events import compute_event_aligned_metrics
from src.experiments.paper_figures.fig5.subexperiments.postprobe_stsp_writeback import (
    build_and_save_probe_stsp_update_artifact,
    probe_stsp_update_conditions,
    probe_stsp_update_layers,
    probe_stsp_update_variable_sets,
    unit_group_digest,
    write_postprobe_stsp_update_metrics,
)
from src.experiments.paper_figures.fig5.subexperiments.preprobe_support import compute_preprobe_support_metrics
from src.experiments.paper_figures.fig5.subexperiments.supplement import (
    write_fig5_supplement_aliases,
    write_supplement_outputs,
)
from src.experiments.paper_figures.fig5.subexperiments.support_perturbation import (
    compute_perturbation_effect_summary,
    compute_perturbation_transition_metrics,
    compute_support_perturbation_metrics,
)
from src.experiments.paper_figures.fig5.subexperiments.trial_sampling import (
    build_local_competition_trials,
    build_local_support_competition_bank,
)
from src.experiments.paper_figures.fig5.types import ExperimentContext, Fig5Config, LocalSupportCompetitionBank
from src.experiments.paper_figures.run_paper_figures import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH_GLOB,
    DEFAULT_OUTPUT_ROOT,
    discover_checkpoints,
)


FIGURE_ID = legacy.FIGURE_ID
NUM_CLASSES = 10


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = normalize_reuse_mode(args.reuse_artifacts)
    cfg = _config_from_args(args)
    ctx = _build_artifact_only_context(cfg) if _is_postprobe_artifact_only_require(str(args.task), mode) else _build_context(cfg)
    _init_process_rss_tracker(ctx)
    _reset_cuda_peak_memory(ctx.device)
    artifact_root = _artifact_root_from_args(args, ctx.seed_dir)
    _sample_process_rss(ctx, "artifact_root_resolved")
    run_info = build_run_info(
        experiment_name=f"{FIGURE_ID}.{args.task}",
        output_dir=ctx.seed_dir,
        entry_script="src.experiments.paper_figures.fig5.run_task",
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
        legacy._write_config_files(ctx)
        _sample_process_rss(ctx, "config_written")
        if _is_postprobe_artifact_only_require(str(args.task), mode):
            trial_hash = _run_postprobe_artifact_only_require(ctx, artifact_root=artifact_root)
            run_info["trial_sampling_hash"] = trial_hash
            _sample_process_rss(ctx, "postprobe_artifact_only_require_complete")
        else:
            trials, trial_hash = _get_trial_sampling(ctx, mode=mode, artifact_root=artifact_root)
            _sample_process_rss(ctx, "trial_sampling_complete")
            run_info["trial_sampling_hash"] = trial_hash
            _run_task(ctx, trials, trial_hash=trial_hash, task_id=str(args.task), mode=mode, artifact_root=artifact_root)
            _sample_process_rss(ctx, "task_complete")
        _finalize_bundle(ctx, artifact_root=artifact_root, mode=mode, task_id=str(args.task))
        finalize_run_info(ctx.seed_dir / "meta", run_info, status="success")
        return 0
    except Exception:
        finalize_run_info(ctx.seed_dir / "meta", run_info, status="failed")
        raise


def _get_trial_sampling(
    ctx: ExperimentContext,
    *,
    mode: str,
    artifact_root: Path,
) -> tuple[pd.DataFrame, str]:
    task_dir = task_artifact_dir(artifact_root, TASK_TRIAL_SAMPLING)
    expected_key = build_trial_sampling_cache_key(ctx.cfg)
    if mode == "require":
        artifact = _timed_artifact_io(ctx, load_trial_sampling_artifact, task_dir, expected_key=expected_key)
        _write_trial_sampling_to_bundle(ctx, artifact.trials, artifact.audit, task_dir=task_dir)
        trial_hash = trials_hash(artifact.trials)
        _set_artifact_metadata(ctx, "trial_sampling", "loaded", task_dir, artifact.digest, expected_key)
        return artifact.trials, trial_hash
    if mode == "auto" and cache_key_matches(task_dir, expected_key):
        artifact = _timed_artifact_io(ctx, load_trial_sampling_artifact, task_dir, expected_key=expected_key)
        _write_trial_sampling_to_bundle(ctx, artifact.trials, artifact.audit, task_dir=task_dir)
        trial_hash = trials_hash(artifact.trials)
        _set_artifact_metadata(ctx, "trial_sampling", "loaded", task_dir, artifact.digest, expected_key)
        return artifact.trials, trial_hash

    trials = build_local_competition_trials(ctx)
    audit = pd.read_csv(ctx.metrics_dir / "supp_trial_condition_audit.csv")
    if mode != "off":
        artifact = _timed_artifact_io(
            ctx,
            save_trial_sampling_artifact,
            task_dir,
            trials=trials,
            audit=audit,
            raw_dir=ctx.raw_dir,
            cache_key=expected_key,
        )
        _write_trial_sampling_to_bundle(ctx, artifact.trials, artifact.audit, task_dir=task_dir)
        trial_hash = trials_hash(artifact.trials)
        _set_artifact_metadata(ctx, "trial_sampling", "built", task_dir, artifact.digest, expected_key)
        ctx.run_log.append(f"{legacy._now()} trial_sampling source=built artifact={task_dir}")
        return artifact.trials, trial_hash

    trial_hash = trials_hash(trials)
    _set_artifact_metadata(ctx, "trial_sampling", "fresh", task_dir, trial_hash, expected_key)
    ctx.n_trials = int(len(trials))
    return trials, trial_hash


def _get_support_bank(
    ctx: ExperimentContext,
    trials: pd.DataFrame,
    *,
    trial_hash: str,
    mode: str,
    artifact_root: Path,
) -> LocalSupportCompetitionBank:
    task_dir = task_artifact_dir(artifact_root, TASK_SUPPORT_BANK)
    expected_key = build_support_bank_cache_key(ctx.cfg, trial_hash=trial_hash)
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = _timed_artifact_io(ctx, load_support_bank_artifact, task_dir, expected_key=expected_key, trials=trials)
        _write_support_bank_to_bundle(ctx, artifact.bank, task_dir=task_dir)
        _set_artifact_metadata(ctx, "preprobe_support_bank", "loaded", task_dir, artifact.digest, expected_key)
        return artifact.bank
    if mode == "require":
        artifact = _timed_artifact_io(ctx, load_support_bank_artifact, task_dir, expected_key=expected_key, trials=trials)
        _write_support_bank_to_bundle(ctx, artifact.bank, task_dir=task_dir)
        _set_artifact_metadata(ctx, "preprobe_support_bank", "loaded", task_dir, artifact.digest, expected_key)
        return artifact.bank

    bank = build_local_support_competition_bank(ctx, trials)
    if mode != "off":
        artifact = _timed_artifact_io(ctx, save_support_bank_artifact, task_dir, bank, raw_dir=ctx.raw_dir, cache_key=expected_key)
        _write_support_bank_to_bundle(ctx, artifact.bank, task_dir=task_dir)
        _set_artifact_metadata(ctx, "preprobe_support_bank", "built", task_dir, artifact.digest, expected_key)
        ctx.run_log.append(f"{legacy._now()} preprobe_support_bank source=built artifact={task_dir}")
        return artifact.bank

    bank_hash = cache_key_digest(expected_key)
    _set_artifact_metadata(ctx, "preprobe_support_bank", "fresh", task_dir, bank_hash, expected_key)
    ctx.n_trials = int(len(trials))
    return bank


def _run_task(
    ctx: ExperimentContext,
    trials: pd.DataFrame,
    *,
    trial_hash: str,
    task_id: str,
    mode: str,
    artifact_root: Path,
) -> None:
    _sample_process_rss(ctx, f"task_{task_id}_start")
    if task_id == TASK_TRIAL_SAMPLING:
        return

    bank = _get_support_bank(ctx, trials, trial_hash=trial_hash, mode=mode, artifact_root=artifact_root)
    _sample_process_rss(ctx, "preprobe_support_bank_ready")
    if task_id == TASK_SUPPORT_BANK:
        return
    if task_id == TASK_PROBE_STSP_UPDATE_BANK:
        _get_probe_stsp_update_bank(ctx, trials, bank, trial_hash=trial_hash, mode=mode, artifact_root=artifact_root)
        _sample_process_rss(ctx, "probe_stsp_update_bank_ready")
        return
    if task_id == TASK_POSTPROBE_STSP_UPDATE:
        artifact = _get_probe_stsp_update_bank(ctx, trials, bank, trial_hash=trial_hash, mode=mode, artifact_root=artifact_root)
        _sample_process_rss(ctx, "probe_stsp_update_bank_ready")
        write_postprobe_stsp_update_metrics(ctx, artifact)
        _sample_process_rss(ctx, "postprobe_stsp_update_metrics_complete")
        return
    if task_id == TASK_PREPROBE_SUPPORT:
        compute_preprobe_support_metrics(ctx, bank)
        _sample_process_rss(ctx, "preprobe_support_metrics_complete")
        return
    if task_id == TASK_EARLY_FIRING:
        compute_early_firing_transition_metrics(ctx, bank)
        _sample_process_rss(ctx, "early_firing_metrics_complete")
        return
    if task_id == TASK_LOCAL_EVENTS:
        compute_event_aligned_metrics(ctx, bank)
        _sample_process_rss(ctx, "local_events_metrics_complete")
        return
    if task_id == TASK_SUPPORT_PERTURBATION:
        _run_support_perturbation(ctx, bank)
        _sample_process_rss(ctx, "support_perturbation_metrics_complete")
        return
    if task_id == TASK_SUPPLEMENT:
        compute_early_firing_transition_metrics(ctx, bank)
        _sample_process_rss(ctx, "early_firing_metrics_complete")
        compute_event_aligned_metrics(ctx, bank)
        _sample_process_rss(ctx, "local_events_metrics_complete")
        _run_support_perturbation(ctx, bank)
        _sample_process_rss(ctx, "support_perturbation_metrics_complete")
        write_supplement_outputs(ctx)
        _sample_process_rss(ctx, "supplement_outputs_complete")
        return
    if task_id == TASK_ALL:
        compute_preprobe_support_metrics(ctx, bank)
        _sample_process_rss(ctx, "preprobe_support_metrics_complete")
        compute_early_firing_transition_metrics(ctx, bank)
        _sample_process_rss(ctx, "early_firing_metrics_complete")
        compute_event_aligned_metrics(ctx, bank)
        _sample_process_rss(ctx, "local_events_metrics_complete")
        _run_support_perturbation(ctx, bank)
        _sample_process_rss(ctx, "support_perturbation_metrics_complete")
        write_supplement_outputs(ctx)
        _sample_process_rss(ctx, "supplement_outputs_complete")
        return
    raise ValueError(f"Unsupported Fig.5 task: {task_id}")


def _run_postprobe_artifact_only_require(
    ctx: ExperimentContext,
    *,
    artifact_root: Path,
) -> str:
    trial_dir = task_artifact_dir(artifact_root, TASK_TRIAL_SAMPLING)
    trial_key = build_trial_sampling_cache_key(ctx.cfg)
    trial_artifact = _timed_artifact_io(ctx, load_trial_sampling_artifact, trial_dir, expected_key=trial_key)
    _write_trial_sampling_to_bundle(ctx, trial_artifact.trials, trial_artifact.audit, task_dir=trial_dir)
    trial_hash = trials_hash(trial_artifact.trials)
    _set_artifact_metadata(ctx, "trial_sampling", "loaded", trial_dir, trial_artifact.digest, trial_key)

    support_dir = task_artifact_dir(artifact_root, TASK_SUPPORT_BANK)
    support_key = build_support_bank_cache_key(ctx.cfg, trial_hash=trial_hash)
    support_artifact = _timed_artifact_io(ctx, load_support_bank_metadata_artifact, support_dir, expected_key=support_key)
    support_key_digest = cache_key_digest(support_key)
    _set_artifact_metadata(ctx, "preprobe_support_bank", "metadata_loaded", support_dir, support_artifact.digest, support_key)

    probe_dir = task_artifact_dir(artifact_root, TASK_PROBE_STSP_UPDATE_BANK)
    embedded_unit_groups = read_probe_stsp_update_unit_groups(probe_dir)
    support_unit_digest = unit_group_digest(support_artifact.tables["unit_groups"])
    embedded_unit_digest = unit_group_digest(embedded_unit_groups)
    if embedded_unit_digest != support_unit_digest:
        raise RuntimeError(
            "Fig.5 postprobe_stsp_update artifact-only require unit-group digest mismatch: "
            f"support_bank={support_unit_digest}, probe_artifact={embedded_unit_digest}"
        )
    expected_probe_key = build_probe_stsp_update_bank_cache_key(
        ctx.cfg,
        trial_hash=trial_hash,
        support_bank_digest=support_artifact.digest,
        support_bank_cache_key_digest=support_key_digest,
        unit_group_digest=embedded_unit_digest,
        conditions=probe_stsp_update_conditions(),
        variable_sets=probe_stsp_update_variable_sets(),
    )
    artifact = _load_probe_stsp_update_bank(
        ctx,
        probe_dir,
        trial_artifact.trials,
        expected_probe_key,
        trial_hash,
        support_artifact.digest,
    )
    _mirror_probe_stsp_update_artifact(ctx, probe_dir)
    _set_artifact_metadata(ctx, "probe_stsp_update_bank", "loaded", probe_dir, artifact.digest, expected_probe_key)
    setattr(ctx, "postprobe_stsp_update_require_replay_mode", "artifact_only")
    ctx.run_log.append(
        f"{legacy._now()} postprobe_stsp_update require_replay=artifact_only "
        "model_load=false encoder_load=false dataset_sample_load=false "
        "support_maps_npz_load=false branch_traces_npz_load=false"
    )
    _sample_process_rss(ctx, "postprobe_artifact_metadata_loaded")
    write_postprobe_stsp_update_metrics(ctx, artifact)
    _sample_process_rss(ctx, "postprobe_stsp_update_metrics_complete")
    return trial_hash


def _get_probe_stsp_update_bank(
    ctx: ExperimentContext,
    trials: pd.DataFrame,
    bank: LocalSupportCompetitionBank,
    *,
    trial_hash: str,
    mode: str,
    artifact_root: Path,
):
    task_dir = task_artifact_dir(artifact_root, TASK_PROBE_STSP_UPDATE_BANK)
    support_digest = str(getattr(ctx, "preprobe_support_bank_artifact_digest", ""))
    support_key_digest = str(getattr(ctx, "preprobe_support_bank_cache_key_digest", ""))
    if not support_digest:
        raise RuntimeError("Fig.5 probe_stsp_update_bank requires a validated preprobe_support_bank digest.")
    expected_key = build_probe_stsp_update_bank_cache_key(
        ctx.cfg,
        trial_hash=trial_hash,
        support_bank_digest=support_digest,
        support_bank_cache_key_digest=support_key_digest,
        unit_group_digest=unit_group_digest(bank.unit_groups),
        conditions=probe_stsp_update_conditions(),
        variable_sets=probe_stsp_update_variable_sets(),
    )
    if mode == "require":
        artifact = _load_probe_stsp_update_bank(ctx, task_dir, trials, expected_key, trial_hash, support_digest)
        _mirror_probe_stsp_update_artifact(ctx, task_dir)
        _set_artifact_metadata(ctx, "probe_stsp_update_bank", "loaded", task_dir, artifact.digest, expected_key)
        return artifact
    if mode == "auto" and _probe_stsp_cache_state(task_dir, expected_key) == "match":
        artifact = _load_probe_stsp_update_bank(ctx, task_dir, trials, expected_key, trial_hash, support_digest)
        _mirror_probe_stsp_update_artifact(ctx, task_dir)
        _set_artifact_metadata(ctx, "probe_stsp_update_bank", "loaded", task_dir, artifact.digest, expected_key)
        return artifact

    _timed_artifact_io(
        ctx,
        build_and_save_probe_stsp_update_artifact,
        ctx,
        bank,
        task_dir=task_dir,
        cache_key=expected_key,
        trial_hash=trial_hash,
        parent_support_bank_digest=support_digest,
    )
    artifact = _load_probe_stsp_update_bank(ctx, task_dir, trials, expected_key, trial_hash, support_digest)
    _mirror_probe_stsp_update_artifact(ctx, task_dir)
    _set_artifact_metadata(ctx, "probe_stsp_update_bank", "built", task_dir, artifact.digest, expected_key)
    ctx.run_log.append(f"{legacy._now()} probe_stsp_update_bank source=built artifact={task_dir}")
    return artifact


def _load_probe_stsp_update_bank(
    ctx: ExperimentContext,
    task_dir: Path,
    trials: pd.DataFrame,
    expected_key: Mapping[str, Any],
    trial_hash: str,
    support_digest: str,
):
    return _timed_artifact_io(
        ctx,
        load_probe_stsp_update_artifact,
        task_dir,
        expected_key=expected_key,
        expected_trials=trials,
        expected_conditions=probe_stsp_update_conditions(),
        expected_layers=probe_stsp_update_layers(),
        expected_variable_sets=probe_stsp_update_variable_sets(),
        expected_parent_digest=support_digest,
        expected_trial_hash=trial_hash,
        expected_network_seed=int(ctx.cfg.network_seed),
        expected_trial_chunk_size=int(expected_key["trial_chunk_size"]),
    )


def _probe_stsp_cache_state(task_dir: Path, expected_key: Mapping[str, Any]) -> str:
    task_dir = Path(task_dir)
    if not task_dir.exists() or not (task_dir / "cache_key.json").exists():
        return "missing"
    payload = read_cache_key(task_dir)
    return "match" if str(payload.get("cache_key_digest")) == cache_key_digest(expected_key) else "stale"


def _mirror_probe_stsp_update_artifact(ctx: ExperimentContext, task_dir: Path) -> None:
    bundle_dir = default_artifact_root(ctx.seed_dir) / TASK_PROBE_STSP_UPDATE_BANK
    try:
        same_dir = Path(task_dir).resolve() == bundle_dir.resolve()
    except FileNotFoundError:
        same_dir = False
    if not same_dir:
        _timed_artifact_io(ctx, copy_probe_stsp_update_artifact_to_bundle, task_dir, bundle_dir)
    ctx.output_files["probe_stsp_update_bank_manifest"] = legacy._rel(bundle_dir / "manifest.csv", ctx.seed_dir)
    ctx.output_files["probe_stsp_update_bank_snapshot_manifest"] = legacy._rel(bundle_dir / "snapshot_manifest.csv", ctx.seed_dir)
    ctx.completed_modules["probe_stsp_update_bank"] = True


def _run_support_perturbation(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    compute_perturbation_transition_metrics(ctx, bank)
    compute_support_perturbation_metrics(ctx, bank)
    compute_perturbation_effect_summary(ctx)


def _write_trial_sampling_to_bundle(
    ctx: ExperimentContext,
    trials: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    task_dir: Path | None,
) -> None:
    legacy._save_csv(ctx, trials.copy(), ctx.trial_specs_dir / "local_competition_trials.csv")
    legacy._save_csv(ctx, audit.copy(), ctx.metrics_dir / "supp_trial_condition_audit.csv")
    if task_dir is not None:
        copy_trial_npz_to_raw(task_dir, ctx.raw_dir)
    trial_masks = ctx.raw_dir / "trial_masks.npz"
    if trial_masks.exists():
        ctx.output_files["trial_masks"] = legacy._rel(trial_masks, ctx.seed_dir)
    ctx.completed_modules["trial_sampling"] = True
    ctx.n_trials = int(len(trials))


def _write_support_bank_to_bundle(
    ctx: ExperimentContext,
    bank: LocalSupportCompetitionBank,
    *,
    task_dir: Path | None,
) -> None:
    legacy._save_csv(ctx, bank.unit_groups.copy(), ctx.trial_specs_dir / "unit_group_definitions.csv")
    legacy._save_csv(ctx, bank.perturbation_sets.copy(), ctx.trial_specs_dir / "perturbation_unit_sets.csv")
    legacy._save_csv(ctx, bank.perturbation_ux_audit.copy(), ctx.metrics_dir / "supp_perturbation_ux_audit.csv")
    if task_dir is not None:
        copy_support_bank_tables_to_bundle(task_dir, ctx.raw_dir)
        ctx.output_files["rollout_manifest"] = legacy._rel(ctx.raw_dir / "rollout_manifest.csv", ctx.seed_dir)
        ctx.output_files["layer1_probe_trace_manifest"] = legacy._rel(ctx.raw_dir / "layer1_probe_trace_manifest.csv", ctx.seed_dir)
    legacy._save_panel_a_example(ctx, bank.trials, bank.support_maps, bank.unit_groups)
    ctx.completed_modules["preprobe_support_bank"] = True
    ctx.n_trials = int(len(bank.trials))


def _build_context(cfg: Fig5Config) -> ExperimentContext:
    seed_everything(int(cfg.network_seed))
    seed_dir = legacy._resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = legacy._prepare_dirs(seed_dir)
    device = resolve_device(cfg.device)
    dataset = legacy._load_dataset_or_raise(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, NUM_CLASSES)
    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Fig.5 requires a real model checkpoint; not found: {cfg.model_path}")
    try:
        net, encoder = load_model_and_encoder(
            cfg.model_path,
            device=device,
            dt=cfg.dt,
            max_duration_ms=max(cfg.sample_ms, cfg.probe_ms, 100),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load Fig.5 model checkpoint and encoder from {cfg.model_path}") from exc
    if net is None or encoder is None:
        raise RuntimeError("Fig.5 requires a real model and encoder; load_model_and_encoder returned an empty component.")
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
        warnings=[],
        output_files={},
        completed_modules={},
        run_log=[f"{legacy._now()} start {FIGURE_ID} task runner seed={cfg.network_seed} smoke={cfg.smoke}"],
    )


def _build_artifact_only_context(cfg: Fig5Config) -> ExperimentContext:
    seed_everything(int(cfg.network_seed))
    seed_dir = legacy._resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = legacy._prepare_dirs(seed_dir)
    device = resolve_device(cfg.device)
    return ExperimentContext(
        cfg=cfg,
        seed_dir=seed_dir,
        config_dir=dirs["config"],
        trial_specs_dir=dirs["trial_specs"],
        raw_dir=dirs["raw"],
        metrics_dir=dirs["metrics"],
        debug_dir=dirs["debug"],
        device=device,
        dataset=None,
        class_index={},
        net=None,
        encoder=None,
        warnings=[],
        output_files={},
        completed_modules={},
        run_log=[
            f"{legacy._now()} start {FIGURE_ID} task runner seed={cfg.network_seed} smoke={cfg.smoke}",
            f"{legacy._now()} postprobe_stsp_update require_replay=artifact_only context=metadata_only",
        ],
    )


def _is_postprobe_artifact_only_require(task_id: str, mode: str) -> bool:
    return str(task_id) == TASK_POSTPROBE_STSP_UPDATE and str(mode) == "require"


def _artifact_root_from_args(args: argparse.Namespace, seed_dir: Path) -> Path:
    if not args.artifact_root:
        root = default_artifact_root(seed_dir)
    else:
        value = Path(args.artifact_root)
        candidate = value if value.is_absolute() else (Path.cwd() / value)
        root = _resolve_artifact_root_candidate(candidate, seed_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_artifact_root_candidate(candidate: Path, seed_dir: Path) -> Path:
    candidate = Path(candidate).resolve()
    options = (
        candidate,
        candidate / "data" / "intermediates",
        candidate / FIGURE_ID / seed_dir.name / "data" / "intermediates",
        candidate / seed_dir.name / "data" / "intermediates",
    )
    task_names = (TASK_TRIAL_SAMPLING, TASK_SUPPORT_BANK, TASK_PROBE_STSP_UPDATE_BANK)
    for option in options:
        if any((option / task_name).exists() for task_name in task_names):
            return option
    return candidate


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (DEFAULT_PROJECT_DEFAULTS.paths.repo_root / path).resolve()


def _resolve_model_path(model_path: str | None, model_path_glob: str, network_seed: int) -> Path:
    if model_path:
        return _resolve_repo_path(model_path)
    checkpoints = discover_checkpoints(str(model_path_glob))
    by_seed = {int(item.seed): item.model_path for item in checkpoints}
    if int(network_seed) not in by_seed:
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


def _finalize_bundle(ctx: ExperimentContext, *, artifact_root: Path, mode: str, task_id: str) -> None:
    _sample_process_rss(ctx, "finalize_start")
    _mark_completed_from_existing_outputs(ctx)
    legacy._write_config_files(ctx)
    if _task_needs_bank(task_id):
        write_fig5_supplement_aliases(ctx)
        if ctx.cfg.save_debug_figures:
            legacy.save_debug_figures(ctx)
    _mark_completed_from_existing_outputs(ctx)
    _refresh_output_file_registry(ctx)
    _sample_process_rss(ctx, "summary_write_ready")
    summary = legacy._write_summary(ctx)
    profiling = _runtime_profiling_summary(ctx)
    summary.update(
        {
            "reuse_artifacts": str(mode),
            "runtime_artifact_root": str(Path(artifact_root).resolve()),
            "task": str(task_id),
            "postprobe_stsp_update": {
                "enabled": bool(ctx.completed_modules.get("postprobe_stsp_update") or ctx.completed_modules.get("probe_stsp_update_bank")),
                "producer_task": TASK_PROBE_STSP_UPDATE_BANK,
                "downstream_task": TASK_POSTPROBE_STSP_UPDATE,
                "artifact_digest": getattr(ctx, "probe_stsp_update_bank_artifact_digest", None),
                "conditions": list(probe_stsp_update_conditions()),
                "layers": list(probe_stsp_update_layers()),
                "device": str(ctx.device),
                "batch_size": int(ctx.cfg.batch_size),
                "enable_branch_batch": bool(ctx.cfg.enable_branch_batch),
                "enable_branch_batch_requested": bool(ctx.cfg.enable_branch_batch),
                "branch_batch_effective": bool(getattr(ctx, "probe_stsp_update_branch_batch_effective", False)),
                "branch_batch_execution": str(
                    getattr(
                        ctx,
                        "probe_stsp_update_branch_batch_execution",
                        "serial_fallback" if bool(ctx.cfg.enable_branch_batch) else "serial",
                    )
                ),
                "branch_batch_fallback_reason": str(
                    getattr(
                        ctx,
                        "probe_stsp_update_branch_batch_fallback_reason",
                        "Branch-batched post-probe STSP boundary capture is not validated; conditions execute serially."
                        if bool(ctx.cfg.enable_branch_batch)
                        else "",
                    )
                ),
                "full_traces_saved": False,
                "require_replay_mode": getattr(ctx, "postprobe_stsp_update_require_replay_mode", None),
                "artifact_only_require": bool(getattr(ctx, "postprobe_stsp_update_require_replay_mode", "") == "artifact_only"),
                "metric_processing": ctx.availability.get("postprobe_stsp_update_metric_processing"),
                "metric_shards_processed": ctx.availability.get("postprobe_stsp_update_metric_shards_processed"),
            },
            "profiling": {
                "device": str(ctx.device),
                "batch_size": int(ctx.cfg.batch_size),
                "peak_cuda_memory_mb": _peak_cuda_memory_mb(ctx.device),
                "cpu_rss_peak_mb": profiling["cpu_rss_peak_mb"],
                "cpu_rss_current_mb": profiling["cpu_rss_current_mb"],
                "cpu_rss_peak_method": profiling["cpu_rss_peak_method"],
                "cpu_rss_peak_sample_count": profiling["cpu_rss_peak_sample_count"],
                "cpu_rss_peak_phase": profiling["cpu_rss_peak_phase"],
                "cpu_rss_peak_limitation": profiling["cpu_rss_peak_limitation"],
                "artifact_read_write_time_sec": float(getattr(ctx, "runtime_artifact_read_write_time_sec", 0.0)),
                "cache_hits": int(getattr(ctx, "runtime_cache_hits", 0)),
                "cache_misses": int(getattr(ctx, "runtime_cache_misses", 0)),
            },
        }
    )
    write_json(summary, ctx.seed_dir / "summary.json")
    legacy._write_run_log(ctx)


def _task_needs_bank(task_id: str) -> bool:
    return task_id != TASK_TRIAL_SAMPLING


def _mark_completed_from_existing_outputs(ctx: ExperimentContext) -> None:
    checks = {
        "trial_sampling": [
            ctx.trial_specs_dir / "local_competition_trials.csv",
            ctx.metrics_dir / "supp_trial_condition_audit.csv",
            ctx.raw_dir / "trial_masks.npz",
        ],
        "preprobe_support_bank": [
            ctx.trial_specs_dir / "unit_group_definitions.csv",
            ctx.trial_specs_dir / "perturbation_unit_sets.csv",
            ctx.metrics_dir / "supp_perturbation_ux_audit.csv",
            ctx.raw_dir / "rollout_manifest.csv",
            ctx.raw_dir / "layer1_probe_trace_manifest.csv",
        ],
        "preprobe_support": [ctx.metrics_dir / "panel_a_preprobe_support_metrics.csv"],
        "early_firing": [
            ctx.metrics_dir / "panel_b_early_firing_transition_metrics.csv",
            ctx.metrics_dir / "panel_b_transition_summary_by_group.csv",
            ctx.metrics_dir / "supp_early_window_robustness.csv",
        ],
        "local_events": [
            ctx.metrics_dir / "panel_c_winner_loser_event_metrics.csv",
            ctx.metrics_dir / "panel_c_event_trace_summary.csv",
            ctx.metrics_dir / "supp_event_selection_audit.csv",
            ctx.metrics_dir / "supp_neighborhood_radius_robustness.csv",
        ],
        "support_perturbation": [
            ctx.metrics_dir / "panel_d_l1_stsp_perturbation_unit_transitions.csv",
            ctx.metrics_dir / "panel_d_l1_stsp_perturbation_transition_summary.csv",
            ctx.metrics_dir / "panel_d_l1_stsp_perturbation_audit.csv",
            ctx.metrics_dir / "panel_d_l1_stsp_perturbation_contrast.csv",
        ],
        "support_perturbation_downstream": [
            ctx.metrics_dir / "panel_d_support_perturbation_node_metrics.csv",
            ctx.metrics_dir / "panel_d_support_perturbation_trial_metrics.csv",
        ],
        "perturbation_effect_summary": [ctx.metrics_dir / "panel_d_perturbation_effect_summary.csv"],
        "supplement": [
            ctx.metrics_dir / "supp_event_chain_fraction_metrics.csv",
            ctx.metrics_dir / "supp_event_chain_null_baselines.csv",
            ctx.metrics_dir / "supp_layer_delay_local_competition_metrics.csv",
        ],
        "probe_stsp_update_bank": [
            default_artifact_root(ctx.seed_dir) / TASK_PROBE_STSP_UPDATE_BANK / "manifest.csv",
            default_artifact_root(ctx.seed_dir) / TASK_PROBE_STSP_UPDATE_BANK / "snapshot_manifest.csv",
            default_artifact_root(ctx.seed_dir) / TASK_PROBE_STSP_UPDATE_BANK / "cache_key.json",
            default_artifact_root(ctx.seed_dir) / TASK_PROBE_STSP_UPDATE_BANK / "artifact_digest.json",
        ],
        "postprobe_stsp_update": [
            ctx.metrics_dir / "panel_postprobe_l2_stsp_writeback_summary.csv",
            ctx.metrics_dir / "panel_postprobe_l2_reupdate_history_composition.csv",
            ctx.metrics_dir / "supp_postprobe_l2_writeback_by_trial.csv",
            ctx.metrics_dir / "supp_postprobe_l2_reupdate_history_by_trial.csv",
            ctx.metrics_dir / "supp_postprobe_l2_writeback_memory_overlap.csv",
            ctx.metrics_dir / "supp_postprobe_l2_writeback_magnitude_qc.csv",
            ctx.metrics_dir / "supp_postprobe_l1_firing_bridge.csv",
        ],
        "supplement_aliases": [
            ctx.metrics_dir / "supp_s9_transition_composition_by_group.csv",
            ctx.metrics_dir / "supp_s10_perturbation_ux_audit.csv",
            ctx.metrics_dir / "supp_s10_perturbation_transition_contrast.csv",
        ],
    }
    for name, paths in checks.items():
        if all(path.exists() for path in paths):
            ctx.completed_modules[name] = True
    event_path = ctx.metrics_dir / "panel_c_winner_loser_event_metrics.csv"
    if event_path.exists():
        try:
            ctx.n_events = int(len(pd.read_csv(event_path)))
        except pd.errors.EmptyDataError:
            ctx.n_events = 0
    trial_path = ctx.trial_specs_dir / "local_competition_trials.csv"
    if trial_path.exists():
        try:
            ctx.n_trials = int(len(pd.read_csv(trial_path)))
        except pd.errors.EmptyDataError:
            ctx.n_trials = 0


def _refresh_output_file_registry(ctx: ExperimentContext) -> None:
    for path in sorted(ctx.seed_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = legacy._rel(path, ctx.seed_dir)
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
    if str(source) in {"loaded", "metadata_loaded"}:
        setattr(ctx, "runtime_cache_hits", int(getattr(ctx, "runtime_cache_hits", 0)) + 1)
    elif str(source) in {"built", "fresh"}:
        setattr(ctx, "runtime_cache_misses", int(getattr(ctx, "runtime_cache_misses", 0)) + 1)


def _timed_artifact_io(ctx: ExperimentContext, operation: Any, *args: Any, **kwargs: Any) -> Any:
    operation_name = getattr(operation, "__name__", operation.__class__.__name__)
    _sample_process_rss(ctx, f"artifact_io_{operation_name}_start")
    start = time.perf_counter()
    try:
        return operation(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        current = float(getattr(ctx, "runtime_artifact_read_write_time_sec", 0.0))
        setattr(ctx, "runtime_artifact_read_write_time_sec", current + elapsed)
        _sample_process_rss(ctx, f"artifact_io_{operation_name}_end")


def _reset_cuda_peak_memory(device: torch.device) -> None:
    if getattr(device, "type", "") != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        return


def _peak_cuda_memory_mb(device: torch.device) -> float | None:
    if getattr(device, "type", "") != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    except Exception:
        return None


def _init_process_rss_tracker(ctx: ExperimentContext) -> None:
    setattr(ctx, "runtime_cpu_rss_current_mb", None)
    setattr(ctx, "runtime_cpu_rss_peak_mb", None)
    setattr(ctx, "runtime_cpu_rss_peak_phase", None)
    setattr(ctx, "runtime_cpu_rss_sample_count", 0)
    setattr(ctx, "runtime_cpu_rss_available", True)
    _sample_process_rss(ctx, "context_initialized")


def _sample_process_rss(ctx: ExperimentContext, phase: str) -> float | None:
    value = _process_rss_mb()
    if value is None:
        setattr(ctx, "runtime_cpu_rss_available", False)
        return None
    setattr(ctx, "runtime_cpu_rss_current_mb", value)
    sample_count = int(getattr(ctx, "runtime_cpu_rss_sample_count", 0)) + 1
    setattr(ctx, "runtime_cpu_rss_sample_count", sample_count)
    peak = getattr(ctx, "runtime_cpu_rss_peak_mb", None)
    if peak is None or value > float(peak):
        setattr(ctx, "runtime_cpu_rss_peak_mb", value)
        setattr(ctx, "runtime_cpu_rss_peak_phase", str(phase))
    return value


def _runtime_profiling_summary(ctx: ExperimentContext) -> dict[str, Any]:
    _sample_process_rss(ctx, "profiling_summary")
    peak = getattr(ctx, "runtime_cpu_rss_peak_mb", None)
    current = getattr(ctx, "runtime_cpu_rss_current_mb", None)
    count = int(getattr(ctx, "runtime_cpu_rss_sample_count", 0))
    if peak is None:
        return {
            "cpu_rss_peak_mb": None,
            "cpu_rss_current_mb": current,
            "cpu_rss_peak_method": "unavailable",
            "cpu_rss_peak_sample_count": count,
            "cpu_rss_peak_phase": None,
            "cpu_rss_peak_limitation": "psutil process RSS sampling was unavailable or failed.",
        }
    return {
        "cpu_rss_peak_mb": float(peak),
        "cpu_rss_current_mb": float(current) if current is not None else None,
        "cpu_rss_peak_method": "process_rss_sampled_psutil",
        "cpu_rss_peak_sample_count": count,
        "cpu_rss_peak_phase": getattr(ctx, "runtime_cpu_rss_peak_phase", None),
        "cpu_rss_peak_limitation": "Sampled at task, artifact I/O, metric, and summary boundaries; transient intra-operation RSS peaks between samples may be missed.",
    }


def _process_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def _config_from_args(args: argparse.Namespace) -> Fig5Config:
    smoke = bool(args.smoke)
    task = str(args.task)
    run_all = task == TASK_ALL
    run_supplement = task == TASK_SUPPLEMENT
    model_path = _resolve_model_path(args.model_path, str(args.model_path_glob), int(args.network_seed))
    return Fig5Config(
        model_path=str(model_path),
        dataset_root=str(_resolve_repo_path(args.dataset_root)),
        output_root=str(_output_root_from_args(args)),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sample_ms=int(args.sample_ms),
        delay_ms=int(args.delay_ms),
        probe_ms=int(args.probe_ms),
        batch_size=min(int(args.batch_size), 2) if smoke else int(args.batch_size),
        max_trials=8 if smoke else int(args.max_trials),
        overlap_mask_mode=str(args.overlap_mask_mode),
        foreground_threshold=float(args.foreground_threshold),
        min_overlap_area=int(args.min_overlap_area),
        min_probe_only_area=int(args.min_probe_only_area),
        medium_q_low=float(args.medium_q_low),
        medium_q_high=float(args.medium_q_high),
        early_window_ms=int(args.early_window_ms),
        drive_score_threshold=float(args.drive_score_threshold),
        local_kernel_radius=int(args.local_kernel_radius),
        peak_support_q=float(args.peak_support_q),
        perturbation_mode=str(args.perturbation_mode),
        perturbation_attenuation_factor=float(args.perturbation_attenuation_factor),
        fig5d_include_balanced=bool(args.fig5d_include_balanced),
        event_align_pre_steps=int(args.event_align_pre_steps),
        event_align_post_steps=int(args.event_align_post_steps),
        chain_pre_spike_steps=int(args.chain_pre_spike_steps),
        chain_post_spike_steps=int(args.chain_post_spike_steps),
        n_null=8 if smoke else int(args.n_null),
        save_full_traces=bool(args.save_full_traces),
        save_spike_cache=bool(args.save_spike_cache),
        run_trial_sampling=True,
        run_preprobe_support=run_all or task == TASK_PREPROBE_SUPPORT,
        run_early_firing=run_all or task == TASK_EARLY_FIRING or run_supplement,
        run_local_events=run_all or task == TASK_LOCAL_EVENTS or run_supplement,
        run_support_perturbation=run_all or task == TASK_SUPPORT_PERTURBATION or run_supplement,
        run_supplement=run_all or run_supplement,
        save_debug_figures=bool(args.save_debug_figures),
        show_progress=not bool(args.no_progress),
        enable_branch_batch=bool(args.enable_branch_batch),
        smoke=smoke,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Fig.5 runtime-artifact DAG task.", allow_abbrev=False)
    parser.add_argument("--task", required=True, choices=TASK_IDS)
    parser.add_argument("--reuse-artifacts", default="auto", choices=REUSE_MODES)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-path-glob", default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=str(Path(DEFAULT_OUTPUT_ROOT) / FIGURE_ID))
    parser.add_argument("--output-dir", default=None, help="Batch output root; the Fig.5 experiment id is appended unless a seed or figure root is supplied.")
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default=DEFAULT_PROJECT_DEFAULTS.runtime.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--save-spike-cache", action="store_true")
    parser.add_argument("--save-full-traces", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--enable-branch-batch", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=400)
    parser.add_argument("--probe-ms", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=500)
    parser.add_argument("--overlap-mask-mode", default="encoded_spike", choices=["encoded_spike", "foreground"])
    parser.add_argument("--foreground-threshold", type=float, default=0.1)
    parser.add_argument("--min-overlap-area", type=int, default=4)
    parser.add_argument("--min-probe-only-area", type=int, default=4)
    parser.add_argument("--medium-q-low", type=float, default=0.35)
    parser.add_argument("--medium-q-high", type=float, default=0.65)
    parser.add_argument("--early-window-ms", type=int, default=15)
    parser.add_argument("--drive-score-threshold", type=float, default=0.05)
    parser.add_argument("--local-kernel-radius", type=int, default=2)
    parser.add_argument("--peak-support-q", type=float, default=0.20)
    parser.add_argument("--perturbation-mode", default="attenuate_reset", choices=["attenuate_reset", "attenuate", "reset"])
    parser.add_argument("--perturbation-attenuation-factor", type=float, default=0.5)
    parser.add_argument("--fig5d-include-balanced", action="store_true")
    parser.add_argument("--event-align-pre-steps", type=int, default=8)
    parser.add_argument("--event-align-post-steps", type=int, default=12)
    parser.add_argument("--chain-pre-spike-steps", type=int, default=4)
    parser.add_argument("--chain-post-spike-steps", type=int, default=6)
    parser.add_argument("--n-null", type=int, default=100)
    return parser.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
