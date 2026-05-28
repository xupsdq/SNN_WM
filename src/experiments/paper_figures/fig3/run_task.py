from __future__ import annotations

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
from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as legacy
from src.experiments.paper_figures.fig3.artifacts import (
    cache_key_matches,
    default_artifact_root,
    load_sequence_specs_artifact,
    load_state_bank_artifact,
    save_sequence_specs_artifact,
    save_state_bank_artifact,
    task_artifact_dir,
    write_json,
)
from src.experiments.paper_figures.fig3.cache_keys import (
    build_sequence_specs_cache_key,
    build_state_bank_cache_key,
    cache_key_digest,
    sequence_specs_hash,
)
from src.experiments.paper_figures.fig3.schemas import (
    REUSE_MODES,
    TASK_ALL,
    TASK_IDS,
    TASK_NEUTRAL_PING,
    TASK_PEAK_VALLEY_LANDSCAPE,
    TASK_PROGRESSIVE_UPDATE,
    TASK_SEQUENCE_TRIAL_SPECS,
    TASK_STATE_BANK,
    TASK_SUPPLEMENT,
    TASK_WEAK_PROBE,
    normalize_reuse_mode,
)
from src.experiments.paper_figures.fig3.subexperiments.neutral_ping import run_neutral_ping_readout_distribution
from src.experiments.paper_figures.fig3.subexperiments.peak_valley_landscape import compute_final_support_landscape
from src.experiments.paper_figures.fig3.subexperiments.progressive_update import compute_progressive_update_metrics
from src.experiments.paper_figures.fig3.subexperiments.state_bank import run_multiitem_sequence_state_bank
from src.experiments.paper_figures.fig3.subexperiments.supplement import compute_supplementary_metrics
from src.experiments.paper_figures.fig3.subexperiments.trial_specs import build_sequence_trial_specs
from src.experiments.paper_figures.fig3.subexperiments.weak_probe import run_sequence_weak_probe_real_rollout_from_state_bank
from src.experiments.paper_figures.fig3.types import ExperimentContext, Fig3Config, MultiItemSequenceLandscapeBank
from src.experiments.paper_figures.run_paper_figures import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH_GLOB,
    DEFAULT_OUTPUT_ROOT,
    discover_checkpoints,
)


FIGURE_ID = legacy.FIGURE_ID
NUM_CLASSES = legacy.NUM_CLASSES


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = normalize_reuse_mode(args.reuse_artifacts)
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
            "task": str(args.task),
        }
    )
    write_run_info(ctx.seed_dir / "meta", run_info)
    try:
        legacy._write_config_files(ctx)
        seq_trials, singleton_trials, partial_trials = _get_sequence_specs(
            ctx,
            task_id=str(args.task),
            mode=mode,
            artifact_root=artifact_root,
        )
        spec_hash = sequence_specs_hash(seq_trials, singleton_trials, partial_trials)
        run_info["sequence_specs_hash"] = spec_hash
        _run_task(ctx, seq_trials, task_id=str(args.task), mode=mode, artifact_root=artifact_root, specs_hash=spec_hash)
        _finalize_bundle(ctx, artifact_root=artifact_root, mode=mode)
        finalize_run_info(ctx.seed_dir / "meta", run_info, status="success")
        return 0
    except Exception:
        finalize_run_info(ctx.seed_dir / "meta", run_info, status="failed")
        raise


def _get_sequence_specs(
    ctx: ExperimentContext,
    *,
    task_id: str,
    mode: str,
    artifact_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_dir = task_artifact_dir(artifact_root, TASK_SEQUENCE_TRIAL_SPECS)
    expected_key = build_sequence_specs_cache_key(ctx.cfg)
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
    ctx.run_log.append(f"{legacy._now()} sequence_trial_specs source=built task={task_id}")
    return seq_trials, singleton_trials, partial_trials


def _write_sequence_specs_to_bundle(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    singleton_reference_trials: pd.DataFrame,
    partial_cue_trials: pd.DataFrame,
) -> None:
    legacy._save_csv(ctx, sequence_trials, ctx.trial_specs_dir / "sequence_trials.csv")
    legacy._save_csv(ctx, singleton_reference_trials, ctx.trial_specs_dir / "singleton_reference_trials.csv")
    legacy._save_csv(ctx, partial_cue_trials, ctx.trial_specs_dir / "partial_cue_trials.csv")
    legacy._save_csv(ctx, legacy._trial_condition_audit(ctx.cfg.network_seed, sequence_trials), ctx.metrics_dir / "supp_trial_condition_audit.csv")
    if not sequence_trials.empty:
        example = sequence_trials[sequence_trials["sequence_id"].astype(int).eq(int(sequence_trials["sequence_id"].iloc[0]))].copy()
        legacy._write_json(legacy._json_safe(example.iloc[0].to_dict()), ctx.raw_dir / "panel_a_example_sequence_metadata.json")
        np.savez_compressed(
            ctx.raw_dir / "panel_a_example_sequence.npz",
            image_ids=example["item_image_id"].to_numpy(dtype=np.int64),
            labels=example["item_label"].to_numpy(dtype=np.int64),
        )
        ctx.output_files["panel_a_example_sequence_metadata"] = legacy._rel(ctx.raw_dir / "panel_a_example_sequence_metadata.json", ctx.seed_dir)
        ctx.output_files["panel_a_example_sequence"] = legacy._rel(ctx.raw_dir / "panel_a_example_sequence.npz", ctx.seed_dir)
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
) -> None:
    if task_id == TASK_SEQUENCE_TRIAL_SPECS:
        return
    if task_id == TASK_STATE_BANK:
        _get_state_bank(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        return
    if task_id == TASK_PROGRESSIVE_UPDATE:
        compute_progressive_update_metrics(ctx, _get_state_bank(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash))
        return
    if task_id == TASK_PEAK_VALLEY_LANDSCAPE:
        compute_final_support_landscape(ctx, _get_state_bank(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash))
        return
    if task_id == TASK_NEUTRAL_PING:
        run_neutral_ping_readout_distribution(ctx, _get_state_bank(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash))
        return
    if task_id == TASK_WEAK_PROBE:
        run_sequence_weak_probe_real_rollout_from_state_bank(ctx, _get_state_bank(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash))
        return
    if task_id == TASK_SUPPLEMENT:
        compute_supplementary_metrics(ctx, _get_state_bank(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash))
        return
    if task_id == TASK_ALL:
        bank = _get_state_bank(ctx, sequence_trials, mode=mode, artifact_root=artifact_root, specs_hash=specs_hash)
        compute_progressive_update_metrics(ctx, bank)
        compute_final_support_landscape(ctx, bank)
        run_neutral_ping_readout_distribution(ctx, bank)
        run_sequence_weak_probe_real_rollout_from_state_bank(ctx, bank)
        compute_supplementary_metrics(ctx, bank)
        return
    raise ValueError(f"Unsupported Fig.3 task: {task_id}")


def _get_state_bank(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    *,
    mode: str,
    artifact_root: Path,
    specs_hash: str,
) -> MultiItemSequenceLandscapeBank:
    task_dir = task_artifact_dir(artifact_root, TASK_STATE_BANK)
    expected_key = build_state_bank_cache_key(ctx.cfg, specs_hash=specs_hash)
    if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
        artifact = load_state_bank_artifact(task_dir, expected_key=expected_key, sequence_trials=sequence_trials)
        _write_state_bank_compat_outputs(ctx, task_dir)
        _set_artifact_metadata(ctx, "state_bank", "loaded", task_dir, artifact.digest, expected_key)
        ctx.completed_modules["state_bank"] = True
        return artifact.bank
    if mode == "require":
        artifact = load_state_bank_artifact(task_dir, expected_key=expected_key, sequence_trials=sequence_trials)
        _write_state_bank_compat_outputs(ctx, task_dir)
        _set_artifact_metadata(ctx, "state_bank", "loaded", task_dir, artifact.digest, expected_key)
        ctx.completed_modules["state_bank"] = True
        return artifact.bank
    bank = run_multiitem_sequence_state_bank(ctx, sequence_trials)
    if mode != "off":
        artifact = save_state_bank_artifact(task_dir, bank, cache_key=expected_key, network_seed=ctx.cfg.network_seed)
        _set_artifact_metadata(ctx, "state_bank", "built", task_dir, artifact.digest, expected_key)
    return bank


def _write_state_bank_compat_outputs(ctx: ExperimentContext, task_dir: Path) -> None:
    for filename in ("state_bank_layer1.npz", "state_bank_layer3.npz"):
        src = task_dir / filename
        dst = ctx.raw_dir / filename
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            ctx.output_files[dst.stem] = legacy._rel(dst, ctx.seed_dir)
    manifest_src = task_dir / "manifest.csv"
    if manifest_src.exists():
        dst = ctx.raw_dir / "state_bank_manifest.csv"
        shutil.copy2(manifest_src, dst)
        ctx.output_files["state_bank_manifest"] = legacy._rel(dst, ctx.seed_dir)


def _build_context(cfg: Fig3Config) -> ExperimentContext:
    seed_everything(int(cfg.network_seed))
    seed_dir = legacy._resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = legacy._prepare_dirs(seed_dir)
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
    legacy._write_config_files(ctx)
    _refresh_output_file_registry(ctx)
    summary = legacy._write_summary(ctx)
    summary.update(
        {
            "reuse_artifacts": str(mode),
            "runtime_artifact_root": str(Path(artifact_root).resolve()),
        }
    )
    write_json(summary, ctx.seed_dir / "summary.json")
    legacy._write_run_log(ctx)


def _mark_completed_from_existing_outputs(ctx: ExperimentContext) -> None:
    checks = {
        "sequence_trial_specs": [ctx.trial_specs_dir / "sequence_trials.csv", ctx.trial_specs_dir / "singleton_reference_trials.csv", ctx.trial_specs_dir / "partial_cue_trials.csv"],
        "state_bank": [ctx.raw_dir / "state_bank_layer1.npz", ctx.raw_dir / "state_bank_layer3.npz", ctx.raw_dir / "state_bank_manifest.csv"],
        "progressive_update": [ctx.metrics_dir / "panel_b_progressive_update_metrics.csv"],
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


def _config_from_args(args: argparse.Namespace) -> Fig3Config:
    smoke = bool(args.smoke)
    task = str(args.task)
    run_all = task == TASK_ALL
    seq_lengths = tuple(int(v) for v in str(args.sequence_lengths).split(",") if str(v).strip())
    weak_probe_keep = tuple(float(v) for v in str(args.weak_probe_keep_probs).split(",") if str(v).strip())
    weak_cue_keep = tuple(float(v) for v in str(args.weak_cue_keep_fractions).split(",") if str(v).strip())
    peak_cue_main_keep = float(args.peak_cue_main_keep_fraction)
    if not any(np.isclose(float(value), peak_cue_main_keep) for value in weak_cue_keep):
        weak_cue_keep = tuple(sorted([*weak_cue_keep, peak_cue_main_keep]))
    else:
        weak_cue_keep = tuple(sorted({float(value) for value in weak_cue_keep}))
    if smoke:
        weak_probe_keep = (0.2, 0.7)
        weak_cue_keep = (peak_cue_main_keep,)
    model_path = _resolve_model_path(args.model_path, str(args.model_path_glob), int(args.network_seed), smoke=smoke)
    return Fig3Config(
        model_path=str(model_path),
        dataset_root=str(_resolve_repo_path(args.dataset_root)),
        output_root=str(_output_root_from_args(args)),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sequence_lengths=seq_lengths,
        primary_sequence_length=int(args.primary_sequence_length),
        main_sequence_length=int(args.main_sequence_length),
        main_only_seq_len_10=bool(args.main_only_seq_len_10),
        sample_ms=int(args.sample_ms),
        delay_ms=int(args.delay_ms),
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
        smoke=smoke,
        peak_cue_main_keep_fraction=peak_cue_main_keep,
        region_ping_q=float(args.region_ping_q),
        region_ping_support_metric=str(args.region_ping_support_metric),
        region_ping_conditions=tuple(str(v).strip() for v in str(args.region_ping_conditions).split(",") if str(v).strip()),
        region_ping_repeats=min(int(args.region_ping_repeats), 2) if smoke else int(args.region_ping_repeats),
        region_ping_amp_sweep=tuple(float(v) for v in str(args.region_ping_amp_sweep).split(",") if str(v).strip()),
        region_ping_use_random_matched=bool(args.region_ping_use_random_matched),
        weak_probe_include_singleton=bool(args.weak_probe_include_singleton),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Fig.3 runtime-artifact DAG task.", allow_abbrev=False)
    parser.add_argument("--task", required=True, choices=TASK_IDS)
    parser.add_argument("--reuse-artifacts", default="auto", choices=REUSE_MODES)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-path-glob", default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=str(Path(DEFAULT_OUTPUT_ROOT) / FIGURE_ID))
    parser.add_argument("--output-dir", default=None, help="Batch output root; the Fig.3 experiment id is appended unless a seed or figure root is supplied.")
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default=DEFAULT_PROJECT_DEFAULTS.runtime.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--save-spike-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-encode-cache", action="store_true")
    parser.add_argument("--enable-condition-batch", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sequence-lengths", default="3,5,7,10")
    parser.add_argument("--primary-sequence-length", type=int, default=7)
    parser.add_argument("--main-sequence-length", type=int, default=10)
    parser.add_argument("--main-only-seq-len-10", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--peak-q", type=float, default=0.20)
    parser.add_argument("--valley-q", type=float, default=0.20)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--weak-cue-target-source", default="sequence_member_random", choices=["sequence_member_random", "unseen_random", "both"])
    parser.add_argument("--weak-cue-keep-fractions", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--peak-cue-main-keep-fraction", type=float, default=0.10)
    parser.add_argument("--weak-cue-repeats", type=int, default=20)
    parser.add_argument("--weak-cue-mask-mode", default="rank_within_target_foreground", choices=["rank_within_target_foreground"])
    parser.add_argument("--foreground-threshold", type=float, default=0.1)
    parser.add_argument("--functional-restore-mode", choices=["full_boundary", "stsp_only", "stsp_only_legacy_current_ux"], default="stsp_only")
    parser.add_argument("--partial-cue-keep-fraction", type=float, default=0.10)
    parser.add_argument("--partial-cue-keep-fraction-sweep", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--partial-cue-repeats", type=int, default=20)
    parser.add_argument("--target-position", default="K-1")
    return parser.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
