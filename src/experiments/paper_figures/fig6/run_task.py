from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS
from src.experiments.common.dataset import build_class_index
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as legacy
from src.experiments.paper_figures.fig6.artifacts import (
    cache_key_matches,
    copy_sequence_bank_artifacts_to_raw,
    default_artifact_root,
    load_sequence_bank_artifact,
    load_sequence_trials_artifact,
    save_sequence_bank_artifact,
    save_sequence_trials_artifact,
    task_artifact_dir,
    write_json,
)
from src.experiments.paper_figures.fig6.cache_keys import (
    build_sequence_bank_cache_key,
    build_sequence_trials_cache_key,
    cache_key_digest,
    sequence_trials_hash,
)
from src.experiments.paper_figures.fig6.schemas import (
    REUSE_MODES,
    TASK_ALL,
    TASK_FIELD_PING_READOUT,
    TASK_GLOBAL_PING_SCORE_SPIKE_PREDICTION,
    TASK_HIGH_STSP_OVERLAP_ABLATION,
    TASK_IDS,
    TASK_OVERLAP_GATED_STSP_RECRUITMENT,
    TASK_OVERLAP_THRESHOLD_SENSITIVITY,
    TASK_REAL_PROBE_SCORE_SPIKE_DEFLECTION,
    TASK_SCORE_SHUFFLE_NULL,
    TASK_SEQUENCE_BANK,
    TASK_SEQUENCE_TRIALS,
    TASK_SUPPLEMENT,
    normalize_reuse_mode,
)
from src.experiments.paper_figures.fig6.subexperiments.field_ping_readout import compute_field_ping_readout
from src.experiments.paper_figures.fig6.subexperiments.global_ping_score_spike_prediction import compute_global_ping_score_spike_prediction
from src.experiments.paper_figures.fig6.subexperiments.high_stsp_overlap_ablation import compute_high_stsp_overlap_ablation
from src.experiments.paper_figures.fig6.subexperiments.overlap_gated_stsp_recruitment import compute_overlap_gated_stsp_recruitment
from src.experiments.paper_figures.fig6.subexperiments.real_probe_score_spike_deflection import compute_real_probe_score_spike_deflection
from src.experiments.paper_figures.fig6.subexperiments.sequence_bank import build_sequence_trials, run_sequence_bank
from src.experiments.paper_figures.fig6.subexperiments.supplement import (
    compute_overlap_threshold_sensitivity_extension,
    compute_score_shuffle_null_extension,
    compute_supplement_outputs,
    write_fig6_supplement_aliases,
    write_global_mechanism_metadata,
)
from src.experiments.paper_figures.fig6.types import ExperimentContext, Fig6Config, PeakAmplifiedReentryBank
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


FIGURE_ID = legacy.FIGURE_ID
NUM_CLASSES = 10


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = normalize_reuse_mode(args.reuse_artifacts)
    cfg = _config_from_args(args)
    ctx = _build_context(cfg)
    artifact_root = _artifact_root_from_args(args, ctx.seed_dir)
    run_info = build_run_info(
        experiment_name=f"{FIGURE_ID}.{args.task}",
        output_dir=ctx.seed_dir,
        entry_script="src.experiments.paper_figures.fig6.run_task",
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
    write_run_info(ctx.meta_dir, run_info)
    try:
        legacy._write_config_files(ctx)
        sequence_trials, sequence_hash = _get_sequence_trials(
            ctx,
            mode=mode,
            artifact_root=artifact_root,
            shared_sequence_root=Path(args.shared_sequence_root) if args.shared_sequence_root else None,
        )
        run_info["sequence_trials_hash"] = sequence_hash
        _run_task(
            ctx,
            sequence_trials,
            sequence_hash=sequence_hash,
            task_id=str(args.task),
            mode=mode,
            artifact_root=artifact_root,
            shared_sequence_root=Path(args.shared_sequence_root) if args.shared_sequence_root else None,
        )
        _finalize_bundle(ctx, artifact_root=artifact_root, mode=mode, task_id=str(args.task))
        finalize_run_info(ctx.meta_dir, run_info, status="success")
        return 0
    except Exception:
        finalize_run_info(ctx.meta_dir, run_info, status="failed")
        raise


def _get_sequence_trials(
    ctx: ExperimentContext,
    *,
    mode: str,
    artifact_root: Path,
    shared_sequence_root: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    task_dir = task_artifact_dir(artifact_root, TASK_SEQUENCE_TRIALS)
    expected_key = build_sequence_trials_cache_key(ctx.cfg)
    if shared_sequence_root is not None:
        root_bank = load_shared_root_bank_artifact(shared_sequence_root)
        artifact = save_sequence_trials_artifact(task_dir, sequence_trials=root_bank.specs.sequence_trials, cache_key=expected_key)
        if root_bank.specs.spec_artifact is not None:
            materialize_spec_view(
                root_bank.specs.spec_artifact,
                task_dir,
                view_figure="fig6",
                view_task=TASK_SEQUENCE_TRIALS,
                view_artifact_digest=artifact.digest,
                view_cache_key_digest=cache_key_digest(expected_key),
            )
        _write_sequence_trials_to_bundle(ctx, artifact.sequence_trials)
        _set_artifact_metadata(ctx, "sequence_trials", "shared_sequence_root", task_dir, artifact.digest, expected_key)
        return artifact.sequence_trials, artifact.digest
    if mode == "require":
        artifact = load_sequence_trials_artifact(task_dir, expected_key=expected_key)
        _write_sequence_trials_to_bundle(ctx, artifact.sequence_trials)
        _set_artifact_metadata(ctx, "sequence_trials", "loaded", task_dir, artifact.digest, expected_key)
        return artifact.sequence_trials, artifact.digest
    if mode == "auto" and cache_key_matches(task_dir, expected_key):
        artifact = load_sequence_trials_artifact(task_dir, expected_key=expected_key)
        _write_sequence_trials_to_bundle(ctx, artifact.sequence_trials)
        _set_artifact_metadata(ctx, "sequence_trials", "loaded", task_dir, artifact.digest, expected_key)
        return artifact.sequence_trials, artifact.digest

    sequence_trials = build_sequence_trials(ctx)
    if mode != "off":
        artifact = save_sequence_trials_artifact(task_dir, sequence_trials=sequence_trials, cache_key=expected_key)
        _write_sequence_trials_to_bundle(ctx, artifact.sequence_trials)
        _set_artifact_metadata(ctx, "sequence_trials", "built", task_dir, artifact.digest, expected_key)
        ctx.run_log.append(f"{legacy._now()} sequence_trials source=built artifact={task_dir}")
        return artifact.sequence_trials, artifact.digest

    digest = sequence_trials_hash(sequence_trials)
    _set_artifact_metadata(ctx, "sequence_trials", "fresh", task_dir, digest, expected_key)
    ctx.n_sequences = int(sequence_trials["sequence_id"].nunique()) if "sequence_id" in sequence_trials.columns else 0
    return sequence_trials, digest


def _get_sequence_bank(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    *,
    sequence_hash: str,
    mode: str,
    artifact_root: Path,
    shared_sequence_root: Path | None = None,
) -> PeakAmplifiedReentryBank:
    task_dir = task_artifact_dir(artifact_root, TASK_SEQUENCE_BANK)
    expected_key = build_sequence_bank_cache_key(ctx.cfg, sequence_trials_hash_value=sequence_hash)
    if shared_sequence_root is not None:
        root_bank = load_shared_root_bank_artifact(shared_sequence_root)
        copy_shared_artifact_tree(root_bank.fig6_sequence_bank_dir, task_dir)
        artifact = load_sequence_bank_artifact(task_dir, expected_key=expected_key, sequence_trials=sequence_trials)
        _write_sequence_bank_to_bundle(ctx, artifact.bank, task_dir=task_dir)
        _set_artifact_metadata(ctx, "sequence_bank", "shared_sequence_root", task_dir, artifact.digest, expected_key)
        ctx.run_log.append(f"{legacy._now()} sequence_bank source=shared_sequence_root artifact={root_bank.root}")
        return artifact.bank
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_sequence_bank_artifact(task_dir, expected_key=expected_key, sequence_trials=sequence_trials)
        _write_sequence_bank_to_bundle(ctx, artifact.bank, task_dir=task_dir)
        _set_artifact_metadata(ctx, "sequence_bank", "loaded", task_dir, artifact.digest, expected_key)
        return artifact.bank
    if mode == "require":
        artifact = load_sequence_bank_artifact(task_dir, expected_key=expected_key, sequence_trials=sequence_trials)
        _write_sequence_bank_to_bundle(ctx, artifact.bank, task_dir=task_dir)
        _set_artifact_metadata(ctx, "sequence_bank", "loaded", task_dir, artifact.digest, expected_key)
        return artifact.bank

    bank = run_sequence_bank(ctx, sequence_trials)
    if mode != "off":
        artifact = save_sequence_bank_artifact(
            task_dir,
            bank,
            raw_dir=ctx.raw_dir,
            cache_key=expected_key,
            network_seed=ctx.cfg.network_seed,
        )
        _write_sequence_bank_to_bundle(ctx, artifact.bank, task_dir=task_dir)
        _set_artifact_metadata(ctx, "sequence_bank", "built", task_dir, artifact.digest, expected_key)
        ctx.run_log.append(f"{legacy._now()} sequence_bank source=built artifact={task_dir}")
        return artifact.bank

    bank_hash = cache_key_digest(expected_key)
    _set_artifact_metadata(ctx, "sequence_bank", "fresh", task_dir, bank_hash, expected_key)
    ctx.n_sequences = int(len(bank.sequence_meta))
    return bank


def _run_task(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    *,
    sequence_hash: str,
    task_id: str,
    mode: str,
    artifact_root: Path,
    shared_sequence_root: Path | None,
) -> None:
    if task_id == TASK_SEQUENCE_TRIALS:
        return

    bank = _get_sequence_bank(
        ctx,
        sequence_trials,
        sequence_hash=sequence_hash,
        mode=mode,
        artifact_root=artifact_root,
        shared_sequence_root=shared_sequence_root,
    )
    if task_id == TASK_SEQUENCE_BANK:
        return
    if task_id == TASK_FIELD_PING_READOUT:
        compute_field_ping_readout(ctx, bank)
        return
    if task_id == TASK_GLOBAL_PING_SCORE_SPIKE_PREDICTION:
        compute_global_ping_score_spike_prediction(ctx, bank)
        return
    if task_id == TASK_REAL_PROBE_SCORE_SPIKE_DEFLECTION:
        compute_real_probe_score_spike_deflection(ctx, bank)
        return
    if task_id == TASK_OVERLAP_GATED_STSP_RECRUITMENT:
        compute_overlap_gated_stsp_recruitment(ctx, bank)
        return
    if task_id == TASK_HIGH_STSP_OVERLAP_ABLATION:
        compute_high_stsp_overlap_ablation(ctx, bank)
        return
    if task_id == TASK_SCORE_SHUFFLE_NULL:
        compute_score_shuffle_null_extension(ctx, bank)
        return
    if task_id == TASK_OVERLAP_THRESHOLD_SENSITIVITY:
        compute_overlap_threshold_sensitivity_extension(ctx, bank)
        return
    if task_id == TASK_SUPPLEMENT:
        _run_main_tasks(ctx, bank)
        compute_supplement_outputs(ctx, bank)
        return
    if task_id == TASK_ALL:
        _run_main_tasks(ctx, bank)
        compute_supplement_outputs(ctx, bank)
        if ctx.cfg.run_score_shuffle_null:
            compute_score_shuffle_null_extension(ctx, bank)
        if ctx.cfg.run_overlap_threshold_sensitivity:
            compute_overlap_threshold_sensitivity_extension(ctx, bank)
        return
    raise ValueError(f"Unsupported Fig.6 task: {task_id}")


def _run_main_tasks(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    compute_field_ping_readout(ctx, bank)
    compute_global_ping_score_spike_prediction(ctx, bank)
    compute_real_probe_score_spike_deflection(ctx, bank)
    compute_overlap_gated_stsp_recruitment(ctx, bank)
    compute_high_stsp_overlap_ablation(ctx, bank)


def _write_sequence_trials_to_bundle(ctx: ExperimentContext, sequence_trials: pd.DataFrame) -> None:
    legacy._save_csv(ctx, sequence_trials.copy(), ctx.trial_specs_dir / "sequence_trials.csv")
    ctx.completed_modules["sequence_trials"] = True
    ctx.n_sequences = int(sequence_trials["sequence_id"].nunique()) if "sequence_id" in sequence_trials.columns else 0


def _write_sequence_bank_to_bundle(
    ctx: ExperimentContext,
    bank: PeakAmplifiedReentryBank,
    *,
    task_dir: Path | None,
) -> None:
    if task_dir is not None:
        copy_sequence_bank_artifacts_to_raw(task_dir, ctx.raw_dir)
    for stem in ("state_bank_manifest", "update_history_matrix", "final_support_maps"):
        suffix = ".csv" if stem == "state_bank_manifest" else ".npz"
        path = ctx.raw_dir / f"{stem}{suffix}"
        if path.exists():
            ctx.output_files[stem] = legacy._rel(path, ctx.seed_dir)
    ctx.completed_modules["sequence_bank"] = True
    ctx.n_sequences = int(len(bank.sequence_meta))


def _build_context(cfg: Fig6Config) -> ExperimentContext:
    seed_everything(int(cfg.network_seed))
    seed_dir = legacy._resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = legacy._prepare_dirs(seed_dir)
    device = resolve_device(cfg.device)
    dataset = legacy._load_dataset_required(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, NUM_CLASSES)
    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Fig.6 model checkpoint not found: {model_path}")
    try:
        net, encoder = load_model_and_encoder(
            cfg.model_path,
            device=device,
            dt=cfg.dt,
            max_duration_ms=max(cfg.sample_ms, cfg.probe_ms, 100),
        )
    except Exception as exc:
        raise RuntimeError(f"Fig.6 model load failed for {cfg.model_path}: {exc}") from exc
    return ExperimentContext(
        cfg=cfg,
        seed_dir=seed_dir,
        config_dir=dirs["config"],
        trial_specs_dir=dirs["trial_specs"],
        raw_dir=dirs["raw"],
        metrics_dir=dirs["metrics"],
        debug_dir=dirs["debug"],
        meta_dir=dirs["meta"],
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
    _mark_completed_from_existing_outputs(ctx)
    legacy._write_config_files(ctx)
    write_fig6_supplement_aliases(ctx)
    if task_id != TASK_SEQUENCE_TRIALS:
        write_global_mechanism_metadata(ctx)
    if ctx.cfg.save_debug_figures:
        legacy.save_debug_figures(ctx)
    legacy._flush_score_audits(ctx)
    _mark_completed_from_existing_outputs(ctx)
    _refresh_output_file_registry(ctx)
    summary = legacy._write_summary(ctx)
    summary.update(
        {
            "reuse_artifacts": str(mode),
            "runtime_artifact_root": str(Path(artifact_root).resolve()),
            "task": str(task_id),
        }
    )
    write_json(summary, ctx.seed_dir / "summary.json")
    legacy._write_run_log(ctx)


def _mark_completed_from_existing_outputs(ctx: ExperimentContext) -> None:
    checks = {
        "sequence_trials": [ctx.trial_specs_dir / "sequence_trials.csv"],
        "sequence_bank": [
            ctx.raw_dir / "state_bank_manifest.csv",
            ctx.raw_dir / "update_history_matrix.npz",
            ctx.raw_dir / "final_support_maps.npz",
        ],
        "field_ping_readout": [ctx.metrics_dir / "panel_b_region_ping_readout_bias.csv"],
        "global_ping_score_spike_prediction": [ctx.metrics_dir / "panel_c_global_ping_score_spike_prediction.csv"],
        "real_probe_score_spike_deflection": [ctx.metrics_dir / "panel_d_real_probe_score_spike_deflection.csv"],
        "overlap_gated_stsp_recruitment": [
            ctx.metrics_dir / "panel_e_overlap_gated_stsp_recruitment.csv",
            ctx.metrics_dir / "panel_e_overlap_gated_stsp_interaction.csv",
        ],
        "high_stsp_overlap_ablation": [
            ctx.metrics_dir / "panel_a_high_stsp_overlap_ablation.csv",
            ctx.metrics_dir / "panel_a_high_stsp_overlap_ablation_summary.csv",
        ],
        "supplement": [
            ctx.metrics_dir / "supp_s11a_score_input_ping_audit.csv",
            ctx.metrics_dir / "supp_s11b_global_ping_count_endpoint.csv",
            ctx.metrics_dir / "supp_s11c_real_probe_window_robustness.csv",
            ctx.metrics_dir / "supp_s11d_overlap_interaction_window_robustness.csv",
            ctx.metrics_dir / "supp_s11e_overlap_site_availability.csv",
            ctx.metrics_dir / "supp_s11f_high_stsp_ablation_paired_difference.csv",
        ],
        "score_shuffle_null": [ctx.metrics_dir / "supp_s11g_score_shuffle_null.csv"],
        "overlap_threshold_sensitivity": [ctx.metrics_dir / "supp_s11h_threshold_sensitivity.csv"],
    }
    for name, paths in checks.items():
        if all(path.exists() for path in paths):
            ctx.completed_modules[name] = True
    sequence_path = ctx.trial_specs_dir / "sequence_trials.csv"
    if sequence_path.exists():
        try:
            seq = pd.read_csv(sequence_path)
            ctx.n_sequences = int(seq["sequence_id"].nunique()) if "sequence_id" in seq.columns else int(len(seq))
        except pd.errors.EmptyDataError:
            ctx.n_sequences = 0
    probe_path = ctx.trial_specs_dir / "probe_candidate_trials.csv"
    if probe_path.exists():
        try:
            ctx.n_probe_candidates = int(len(pd.read_csv(probe_path)))
        except pd.errors.EmptyDataError:
            ctx.n_probe_candidates = 0
    matched_path = ctx.trial_specs_dir / "matched_raw_overlap_groups.csv"
    if matched_path.exists():
        try:
            ctx.n_matched_groups = int(len(pd.read_csv(matched_path)))
        except pd.errors.EmptyDataError:
            ctx.n_matched_groups = 0


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


def _config_from_args(args: argparse.Namespace) -> Fig6Config:
    smoke = bool(args.smoke)
    task = str(args.task)
    run_all = task == TASK_ALL
    run_supplement = run_all or task == TASK_SUPPLEMENT
    seq_lengths = tuple(int(v) for v in str(args.sequence_lengths).split(",") if str(v).strip())
    recent_windows = tuple(int(v) for v in str(args.recent_overlap_windows).split(",") if str(v).strip())
    score_windows = tuple(int(v) for v in str(args.score_early_windows_ms).split(",") if str(v).strip())
    clip_quantiles = tuple(float(v) for v in str(args.gain_ratio_clip_quantiles).split(",") if str(v).strip())
    force_main_outputs = bool(run_all) if args.force_main_outputs is None else bool(args.force_main_outputs)
    model_path = _resolve_model_path(args.model_path, str(args.model_path_glob), int(args.network_seed))
    return Fig6Config(
        model_path=str(model_path),
        dataset_root=str(_resolve_repo_path(args.dataset_root)),
        output_root=str(_output_root_from_args(args)),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sequence_lengths=seq_lengths,
        primary_sequence_length=int(args.primary_sequence_length),
        sample_ms=int(args.sample_ms),
        delay_ms=int(args.delay_ms),
        ping_ms=int(args.ping_ms),
        ping_amp=float(args.ping_amp),
        global_ping_ms=int(args.global_ping_ms),
        global_ping_amp=float(args.global_ping_amp),
        probe_ms=int(args.probe_ms),
        batch_size=min(int(args.batch_size), 2) if smoke else int(args.batch_size),
        num_sequences=4 if smoke else int(args.num_sequences),
        num_probe_candidates_per_sequence=2 if smoke else int(args.num_probe_candidates_per_sequence),
        peak_q=float(args.peak_q),
        recent_window=int(args.recent_window),
        multi_update_threshold=int(args.multi_update_threshold),
        n_null=8 if smoke else int(args.n_null),
        n_matched_groups=4 if smoke else int(args.n_matched_groups),
        foreground_threshold=float(args.foreground_threshold),
        functional_restore_mode=str(args.functional_restore_mode),
        save_full_traces=bool(args.save_full_traces),
        save_l3_trace=not bool(args.no_save_l3_trace),
        save_spike_cache=bool(args.save_spike_cache),
        run_sequence_bank=True,
        run_field_ping_readout=run_all or task == TASK_FIELD_PING_READOUT or run_supplement,
        run_global_ping_score_spike_prediction=run_all or task == TASK_GLOBAL_PING_SCORE_SPIKE_PREDICTION or run_supplement,
        run_ping_score_spike_prediction=False,
        run_real_probe_score_spike_deflection=run_all or task == TASK_REAL_PROBE_SCORE_SPIKE_DEFLECTION or run_supplement,
        run_overlap_gated_stsp_recruitment=run_all or task == TASK_OVERLAP_GATED_STSP_RECRUITMENT or run_supplement,
        run_high_stsp_overlap_ablation=run_all or task == TASK_HIGH_STSP_OVERLAP_ABLATION or run_supplement,
        run_supplement=run_supplement,
        run_score_shuffle_null=run_all or task == TASK_SCORE_SHUFFLE_NULL,
        run_overlap_threshold_sensitivity=run_all or task == TASK_OVERLAP_THRESHOLD_SENSITIVITY,
        force_main_outputs=force_main_outputs,
        score_eps=float(args.score_eps),
        score_early_windows_ms=score_windows,
        primary_score_early_window_ms=int(args.primary_score_early_window_ms),
        score_n_bins=int(args.score_n_bins),
        basin_radius=int(args.basin_radius),
        basin_top_q=float(args.basin_top_q),
        stsp_group_quantile=float(args.stsp_group_quantile),
        overlap_threshold=float(args.overlap_threshold),
        gain_ratio_clip_quantiles=clip_quantiles if len(clip_quantiles) == 2 else (0.01, 0.99),
        real_probe_entry_mode=str(args.real_probe_entry_mode),
        score_use_log_gain=bool(args.score_use_log_gain),
        recent_overlap_windows=recent_windows,
        leave_one_out_mode=str(args.leave_one_out_mode),
        real_rollout_required_for_main=bool(args.real_rollout_required_for_main),
        save_debug_figures=bool(args.save_debug_figures),
        show_progress=not bool(args.no_progress),
        use_encode_cache=not bool(args.no_encode_cache),
        enable_probe_batch=bool(args.enable_probe_batch),
        enable_high_stsp_ablation_batch=bool(args.enable_high_stsp_ablation_batch),
        enable_sequence_bank_batch=bool(args.enable_sequence_bank_batch),
        enable_leave_one_out_batch=bool(args.enable_leave_one_out_batch),
        smoke=smoke,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Fig.6 runtime-artifact DAG task.", allow_abbrev=False)
    parser.add_argument("--task", required=True, choices=TASK_IDS)
    parser.add_argument("--reuse-artifacts", default="auto", choices=REUSE_MODES)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--shared-sequence-root", default=None, help="Path to a shared Fig.3/Fig.6 sequence-root bank artifact or artifact root.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-path-glob", default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=str(Path(DEFAULT_OUTPUT_ROOT) / FIGURE_ID))
    parser.add_argument("--output-dir", default=None, help="Batch output root; the Fig.6 experiment id is appended unless a seed or figure root is supplied.")
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default=DEFAULT_PROJECT_DEFAULTS.runtime.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sequence-lengths", default="10")
    parser.add_argument("--primary-sequence-length", type=int, default=7)
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--ping-ms", type=int, default=30)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--global-ping-ms", type=int, default=30)
    parser.add_argument("--global-ping-amp", type=float, default=0.5)
    parser.add_argument("--probe-ms", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--num-probe-candidates-per-sequence", type=int, default=8)
    parser.add_argument("--peak-q", type=float, default=0.20)
    parser.add_argument("--recent-window", type=int, default=2)
    parser.add_argument("--multi-update-threshold", type=int, default=2)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--n-matched-groups", type=int, default=100)
    parser.add_argument("--foreground-threshold", type=float, default=0.1)
    parser.add_argument("--functional-restore-mode", choices=["full_boundary", "stsp_only", "stsp_only_legacy_current_ux"], default="stsp_only")
    parser.add_argument("--recent-overlap-windows", default="2,3,4,5")
    parser.add_argument("--score-eps", type=float, default=1e-6)
    parser.add_argument("--score-early-windows-ms", default="5,10,15,20")
    parser.add_argument("--primary-score-early-window-ms", type=int, default=10)
    parser.add_argument("--score-n-bins", type=int, default=5)
    parser.add_argument("--basin-radius", type=int, default=2)
    parser.add_argument("--basin-top-q", type=float, default=0.20)
    parser.add_argument("--stsp-group-quantile", type=float, default=0.20)
    parser.add_argument("--overlap-threshold", type=float, default=0.05)
    parser.add_argument("--gain-ratio-clip-quantiles", default="0.01,0.99")
    parser.add_argument("--real-probe-entry-mode", default="encoded_spike", choices=["encoded_spike", "foreground"])
    parser.add_argument("--score-use-log-gain", action="store_true")
    parser.add_argument("--leave-one-out-mode", default="blank_same_timing", choices=["blank_same_timing"])
    parser.add_argument("--real-rollout-required-for-main", dest="real_rollout_required_for_main", action="store_true", default=True)
    parser.add_argument("--allow-proxy-main", dest="real_rollout_required_for_main", action="store_false")
    parser.add_argument("--force-main-outputs", dest="force_main_outputs", action="store_true", default=None)
    parser.add_argument("--no-force-main-outputs", dest="force_main_outputs", action="store_false")
    parser.add_argument("--save-full-traces", action="store_true")
    parser.add_argument("--no-save-l3-trace", action="store_true")
    parser.add_argument("--save-spike-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-encode-cache", action="store_true")
    parser.add_argument("--enable-probe-batch", action="store_true")
    parser.add_argument("--enable-high-stsp-ablation-batch", action="store_true")
    parser.add_argument("--enable-sequence-bank-batch", action="store_true")
    parser.add_argument("--enable-leave-one-out-batch", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
