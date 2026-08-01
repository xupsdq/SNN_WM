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

    raise SystemExit(_final_statistics_main("fig1", _early_args))

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS
from src.experiments.common.dataset import build_class_index
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.paper_figures import fig1_functional_stsp_substrate_experiment as legacy
from src.experiments.paper_figures.fig1.artifacts import (
    DmsBoundaryBank,
    boundary_shard_path,
    cache_key_matches,
    default_artifact_root,
    load_delay_feature_bank,
    load_dms_boundary_bank,
    load_trial_specs_artifact,
    read_json,
    reset_task_artifact_dir,
    save_boundary_shard,
    save_trial_specs_artifact,
    task_artifact_dir,
    write_json,
    write_dms_boundary_bank_files,
)
from src.experiments.paper_figures.fig1.cache_keys import (
    build_cache_key,
    build_trial_specs_cache_key,
    cache_key_digest,
    trial_specs_hash,
)
from src.experiments.paper_figures.fig1.schemas import (
    REUSE_MODES,
    TASK_BASELINE,
    TASK_DELAY_DECODER,
    TASK_DELAY_FEATURE_BANK,
    TASK_DMS_BOUNDARY_BANK,
    TASK_DMS_DELAY_SWEEP_READOUT,
    TASK_DMS_SHUFFLE_READOUT,
    TASK_FIRING_RATE_CONTROL,
    TASK_TIME_BINNED_FIRING_RATE_CONTROL,
    TASK_IDS,
    TASK_TRIAL_SPECS,
    normalize_reuse_mode,
)
from src.experiments.paper_figures.run_paper_figures import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH_GLOB,
    DEFAULT_OUTPUT_ROOT,
    discover_checkpoints,
)
from src.experiments.paper_figures.fig1.subexperiments.baseline import run_baseline_eval
from src.experiments.paper_figures.fig1.subexperiments.delay_decode import (
    build_delay_feature_bank,
    load_delay_feature_bank_for_decode,
    run_delay_decoder_from_bank,
    run_delay_stsp_decode,
)
from src.experiments.paper_figures.fig1.subexperiments.dms_delay_sweep import run_dms_functional_delay_sweep
from src.experiments.paper_figures.fig1.subexperiments.dms_shuffle import run_dms_substrate_shuffle
from src.experiments.paper_figures.fig1.subexperiments.firing_rate_control import (
    run_phase_firing_rate_control,
    run_phase_firing_rate_control_from_bank,
)
from src.experiments.paper_figures.fig1.subexperiments.time_binned_firing_rate import (
    run_time_binned_firing_rate_control,
)
from src.experiments.paper_figures.fig1.subexperiments.helpers import (
    _encode_cached,
    _iter_batches,
    _run_sample_multi_delay_boundary_capture_with_phase,
)
from src.experiments.paper_figures.fig1.types import ExperimentContext, Fig1Config


FIGURE_ID = legacy.FIGURE_ID
NUM_CLASSES = legacy.NUM_CLASSES


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--task" in raw_argv and "final-statistics" in raw_argv:
        from src.experiments.paper_figures.final_six.pipeline import canonical_runner_main

        return canonical_runner_main("fig1", raw_argv)
    args = _parse_args(argv)
    mode = normalize_reuse_mode(args.reuse_artifacts)
    cfg = _config_from_args(args)
    ctx = _build_context(cfg)
    run_info = build_run_info(
        experiment_name=f"{FIGURE_ID}.{args.task}",
        output_dir=ctx.seed_dir,
        entry_script="src.experiments.paper_figures.fig1.run_task",
        seed=cfg.network_seed,
        dataset=f"MNIST:{cfg.split}",
        command=" ".join(sys.argv if argv is None else ["run_task", *argv]),
        model_path=cfg.model_path,
        status="running",
    )
    write_run_info(ctx.seed_dir / "meta", run_info)
    try:
        legacy._write_config_files(ctx)
        artifact_root = _artifact_root_from_args(args, ctx.seed_dir)
        specs = _get_trial_specs(ctx, task_id=str(args.task), mode=mode, artifact_root=artifact_root)
        _annotate_run_info_with_trial_specs(run_info, ctx)
        _run_task(ctx, specs, task_id=str(args.task), mode=mode, artifact_root=artifact_root)
        _finalize_bundle(ctx)
        finalize_run_info(ctx.seed_dir / "meta", run_info, status="success")
        return 0
    except Exception:
        finalize_run_info(ctx.seed_dir / "meta", run_info, status="failed")
        raise


def _get_trial_specs(
    ctx: ExperimentContext,
    *,
    task_id: str,
    mode: str,
    artifact_root: Path,
) -> dict[str, pd.DataFrame]:
    task_dir = task_artifact_dir(artifact_root, TASK_TRIAL_SPECS)
    expected_key = build_trial_specs_cache_key(ctx.cfg)
    if task_id == TASK_TRIAL_SPECS:
        return _build_and_save_trial_specs(ctx, task_dir=task_dir, cache_key=expected_key)
    if mode == "off":
        specs = legacy.build_trial_specs(ctx)
        _set_trial_specs_metadata(ctx, source="built", artifact_dir=task_dir, digest=trial_specs_hash(specs), cache_key=expected_key)
        return specs
    if mode == "require":
        return _load_trial_specs_for_bundle(ctx, task_dir=task_dir, expected_key=expected_key)
    if mode == "auto":
        if cache_key_matches(task_dir, expected_key):
            try:
                return _load_trial_specs_for_bundle(ctx, task_dir=task_dir, expected_key=expected_key)
            except Exception:
                pass
        return _build_and_save_trial_specs(ctx, task_dir=task_dir, cache_key=expected_key)
    if mode == "force":
        if task_dir.exists():
            return _load_trial_specs_for_bundle(ctx, task_dir=task_dir, expected_key=expected_key)
        return _build_and_save_trial_specs(ctx, task_dir=task_dir, cache_key=expected_key)
    raise ValueError(f"Unsupported reuse-artifacts mode: {mode}")


def _build_and_save_trial_specs(
    ctx: ExperimentContext,
    *,
    task_dir: Path,
    cache_key: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    specs = legacy.build_trial_specs(ctx)
    artifact = save_trial_specs_artifact(task_dir, specs, cache_key=cache_key)
    _set_trial_specs_metadata(ctx, source="built", artifact_dir=task_dir, digest=artifact.digest, cache_key=cache_key)
    ctx.run_log.append(f"{legacy._now()} trial_specs source=built artifact={task_dir}")
    return specs


def _load_trial_specs_for_bundle(
    ctx: ExperimentContext,
    *,
    task_dir: Path,
    expected_key: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    artifact = load_trial_specs_artifact(task_dir, expected_key=expected_key)
    _write_loaded_trial_specs_to_bundle(ctx, artifact.specs)
    _set_trial_specs_metadata(ctx, source="loaded", artifact_dir=task_dir, digest=artifact.digest, cache_key=expected_key)
    ctx.run_log.append(f"{legacy._now()} trial_specs source=loaded artifact={task_dir}")
    return artifact.specs


def _write_loaded_trial_specs_to_bundle(ctx: ExperimentContext, specs: Mapping[str, pd.DataFrame]) -> None:
    legacy._save_csv(ctx, specs["baseline"], ctx.trial_specs_dir / "baseline_eval_trials.csv")
    legacy._save_csv(ctx, specs["delay_train"], ctx.trial_specs_dir / "delay_decode_train_trials.csv")
    legacy._save_csv(ctx, specs["delay_test"], ctx.trial_specs_dir / "delay_decode_test_trials.csv")
    legacy._save_csv(ctx, specs["dms"], ctx.trial_specs_dir / "dms_shuffle_trials.csv")
    legacy._save_csv(ctx, _trial_condition_audit_from_dms(ctx, specs["dms"]), ctx.metrics_dir / "supp_trial_condition_audit.csv")
    ctx.n_trials.update(
        {
            "baseline": len(specs["baseline"]),
            "delay_train": len(specs["delay_train"]),
            "delay_test": len(specs["delay_test"]),
            "dms": len(specs["dms"]),
        }
    )
    ctx.completed_modules["trial_specs"] = True


def _trial_condition_audit_from_dms(ctx: ExperimentContext, dms: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "network_seed": int(ctx.cfg.network_seed),
            "audit_type": "donor_plan",
            "label": "all",
            "count": int(len(dms)),
            "fixed_point_count": 0,
            "fixed_point_rate": 0.0,
            "notes": "Donor mapping is constructed per DMS batch with strict all-three-distinct semantics.",
        }
    ]
    for col in ("sample_label", "probe_label"):
        if col not in dms.columns:
            continue
        for label, count in dms[col].value_counts().sort_index().items():
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "audit_type": f"class_count_{col}",
                    "label": int(label),
                    "count": int(count),
                    "fixed_point_count": 0,
                    "fixed_point_rate": 0.0,
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def _set_trial_specs_metadata(
    ctx: ExperimentContext,
    *,
    source: str,
    artifact_dir: Path,
    digest: str,
    cache_key: Mapping[str, Any],
) -> None:
    setattr(ctx, "trial_specs_source", str(source))
    setattr(ctx, "trial_specs_artifact_root", str(Path(artifact_dir).resolve()))
    setattr(ctx, "trial_specs_digest", str(digest))
    setattr(ctx, "trial_specs_cache_key_digest", cache_key_digest(cache_key))


def _trial_specs_metadata(ctx: ExperimentContext) -> dict[str, Any]:
    return {
        "trial_specs_source": getattr(ctx, "trial_specs_source", ""),
        "trial_specs_artifact_root": getattr(ctx, "trial_specs_artifact_root", ""),
        "trial_specs_digest": getattr(ctx, "trial_specs_digest", ""),
        "trial_specs_cache_key_digest": getattr(ctx, "trial_specs_cache_key_digest", ""),
    }


def _annotate_run_info_with_trial_specs(run_info: dict[str, Any], ctx: ExperimentContext) -> None:
    run_info.update(_trial_specs_metadata(ctx))


def _run_task(
    ctx: ExperimentContext,
    specs: Mapping[str, pd.DataFrame],
    *,
    task_id: str,
    mode: str,
    artifact_root: Path,
) -> None:
    if task_id == TASK_TRIAL_SPECS:
        return
    if task_id == TASK_BASELINE:
        run_baseline_eval(ctx, specs["baseline"])
        return
    if task_id == TASK_DELAY_FEATURE_BANK:
        _get_delay_features(ctx, specs["delay_train"], specs["delay_test"], artifact_root=artifact_root, mode=mode, producer=True)
        return
    if task_id == TASK_DELAY_DECODER:
        if mode == "off":
            run_delay_stsp_decode(ctx, specs["delay_train"], specs["delay_test"])
        else:
            features = _get_delay_features(
                ctx,
                specs["delay_train"],
                specs["delay_test"],
                artifact_root=artifact_root,
                mode=mode,
                producer=False,
            )
            run_delay_decoder_from_bank(ctx, features)
        return
    if task_id == TASK_DMS_BOUNDARY_BANK:
        _get_dms_boundary_bank(ctx, specs["dms"], artifact_root=artifact_root, mode=mode, producer=True)
        return
    if task_id == TASK_DMS_SHUFFLE_READOUT:
        if mode == "off":
            run_dms_substrate_shuffle(ctx, specs["dms"])
        else:
            bank = _get_dms_boundary_bank(ctx, specs["dms"], artifact_root=artifact_root, mode=mode, producer=False)
            run_dms_substrate_shuffle(ctx, specs["dms"], boundary_bank=bank)
        return
    if task_id == TASK_DMS_DELAY_SWEEP_READOUT:
        if mode == "off":
            run_dms_functional_delay_sweep(ctx, specs["dms"])
        else:
            bank = _get_dms_boundary_bank(ctx, specs["dms"], artifact_root=artifact_root, mode=mode, producer=False)
            run_dms_functional_delay_sweep(ctx, specs["dms"], boundary_bank=bank)
        return
    if task_id == TASK_FIRING_RATE_CONTROL:
        if mode == "off":
            legacy._write_empty_phase_rates(ctx)
            ctx.completed_modules["firing_rate_control"] = True
        else:
            bank = _get_dms_boundary_bank(ctx, specs["dms"], artifact_root=artifact_root, mode=mode, producer=False)
            run_phase_firing_rate_control_from_bank(ctx, bank)
        return
    if task_id == TASK_TIME_BINNED_FIRING_RATE_CONTROL:
        run_time_binned_firing_rate_control(
            ctx,
            specs["dms"],
            bin_ms=int(ctx.cfg.firing_bin_ms),
        )
        return
    raise ValueError(f"Unsupported Fig.1 task: {task_id}")


def _get_delay_features(
    ctx: ExperimentContext,
    train_trials: pd.DataFrame,
    test_trials: pd.DataFrame,
    *,
    artifact_root: Path,
    mode: str,
    producer: bool,
) -> dict[tuple[str, int, str], tuple[Any, Any, Any]]:
    task_dir = task_artifact_dir(artifact_root, TASK_DELAY_FEATURE_BANK)
    expected_key = _delay_feature_cache_key(ctx, train_trials, test_trials)
    if mode != "off":
        if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
            try:
                return load_delay_feature_bank_for_decode(task_dir, expected_key=expected_key)
            except Exception:
                if mode == "require":
                    raise
        elif mode == "require":
            load_delay_feature_bank(task_dir, expected_key=expected_key)
        if mode == "require":
            raise RuntimeError("delay_feature_bank require-mode load failed without producing features.")
    if mode == "off" and not producer:
        return build_delay_feature_bank(ctx, train_trials, test_trials)
    return build_delay_feature_bank(ctx, train_trials, test_trials, artifact_dir=task_dir, cache_key=expected_key)


def _get_dms_boundary_bank(
    ctx: ExperimentContext,
    dms_trials: pd.DataFrame,
    *,
    artifact_root: Path,
    mode: str,
    producer: bool,
) -> DmsBoundaryBank:
    task_dir = task_artifact_dir(artifact_root, TASK_DMS_BOUNDARY_BANK)
    expected_key = _dms_boundary_cache_key(ctx, dms_trials)
    if mode != "off":
        if mode in {"auto", "require"} and cache_key_matches(task_dir, expected_key):
            try:
                return load_dms_boundary_bank(
                    task_dir,
                    expected_key=expected_key,
                    dms_trials=dms_trials,
                    batch_size=int(ctx.cfg.dms_batch_size),
                )
            except Exception:
                if mode == "require":
                    raise
        elif mode == "require":
            return load_dms_boundary_bank(
                task_dir,
                expected_key=expected_key,
                dms_trials=dms_trials,
                batch_size=int(ctx.cfg.dms_batch_size),
            )
    if mode == "off" and not producer:
        raise RuntimeError("Internal error: DMS boundary bank requested for reuse-artifacts=off.")
    return _build_dms_boundary_bank(ctx, dms_trials, task_dir=task_dir, cache_key=expected_key)


def _build_dms_boundary_bank(
    ctx: ExperimentContext,
    dms_trials: pd.DataFrame,
    *,
    task_dir: Path,
    cache_key: Mapping[str, Any],
) -> DmsBoundaryBank:
    reset_task_artifact_dir(task_dir)
    delay_points = tuple(sorted({int(ctx.cfg.dms_delay_ms), *(int(v) for v in ctx.cfg.dms_delay_sweep_ms)}))
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    manifest_rows: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    phase_rate_rows: list[dict[str, Any]] = []
    layer_input_shapes_by_batch: dict[int, dict[str, tuple[int, ...]]] = {}
    batches = _iter_batches(dms_trials, ctx.cfg.dms_batch_size)
    total_batches = legacy.math.ceil(len(dms_trials) / max(1, int(ctx.cfg.dms_batch_size)))
    for batch_id, batch in enumerate(
        legacy._progress(
            batches,
            total=total_batches,
            desc="fig1 dms boundary bank batches",
            enabled=ctx.cfg.show_progress,
        )
    ):
        sample_spikes = _encode_cached(ctx, batch["sample_image_id"].to_numpy(), ctx.cfg.dms_sample_steps, cache=encode_cache)
        snapshots_by_delay, phase_rows, layer_input_shapes = _run_sample_multi_delay_boundary_capture_with_phase(
            ctx,
            sample_spikes,
            batch,
            delay_points,
            phase_delay_ms=int(ctx.cfg.dms_delay_ms),
        )
        layer_input_shapes_by_batch[int(batch_id)] = {
            layer: tuple(int(v) for v in shape) for layer, shape in layer_input_shapes.items()
        }
        manifest_rows.extend(legacy_dms_manifest_rows_for_batch(batch, int(batch_id)))
        phase_rate_rows.extend(phase_rows)
        for delay_ms, boundary in snapshots_by_delay.items():
            shard_path = boundary_shard_path(task_dir, int(batch_id), int(delay_ms))
            save_boundary_shard(shard_path, boundary)
            shard_rows.append(
                {
                    "batch_id": int(batch_id),
                    "delay_ms": int(delay_ms),
                    "path": shard_path.relative_to(task_dir).as_posix(),
                }
            )
    return write_dms_boundary_bank_files(
        task_dir,
        manifest_rows=manifest_rows,
        shard_rows=shard_rows,
        phase_rate_rows=phase_rate_rows,
        layer_input_shapes_by_batch=layer_input_shapes_by_batch,
        cache_key=cache_key,
    )


def legacy_dms_manifest_rows_for_batch(batch: pd.DataFrame, batch_id: int) -> list[dict[str, Any]]:
    from src.experiments.paper_figures.fig1.artifacts import dms_manifest_rows_for_batch

    return dms_manifest_rows_for_batch(batch, batch_id)


def _delay_feature_cache_key(ctx: ExperimentContext, train_trials: pd.DataFrame, test_trials: pd.DataFrame) -> dict[str, Any]:
    return build_cache_key(
        ctx.cfg,
        task_id=TASK_DELAY_FEATURE_BANK,
        trial_hash=trial_specs_hash({"delay_train": train_trials, "delay_test": test_trials}),
        extra={"feature_type": "ux_concat", "delay_points_ms": [int(v) for v in ctx.cfg.delay_points_ms]},
    )


def _dms_boundary_cache_key(ctx: ExperimentContext, dms_trials: pd.DataFrame) -> dict[str, Any]:
    return build_cache_key(
        ctx.cfg,
        task_id=TASK_DMS_BOUNDARY_BANK,
        trial_hash=trial_specs_hash({"dms": dms_trials}),
        extra={
            "boundary_delays_ms": sorted({int(ctx.cfg.dms_delay_ms), *(int(v) for v in ctx.cfg.dms_delay_sweep_ms)}),
            "boundary_state": "full_layer_boundary",
            "phase_rates_delay_ms": int(ctx.cfg.dms_delay_ms),
        },
    )


def _build_context(cfg: Fig1Config) -> ExperimentContext:
    seed_everything(int(cfg.network_seed))
    if not Path(cfg.model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {cfg.model_path}")
    seed_dir = legacy._resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = legacy._prepare_dirs(seed_dir)
    device = resolve_device(cfg.device)
    dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, NUM_CLASSES)
    net, encoder = load_model_and_encoder(
        cfg.model_path,
        device=device,
        dt=cfg.dt,
        max_duration_ms=max(cfg.sample_ms, cfg.dms_sample_ms, cfg.probe_ms, 100),
    )
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
        n_trials={},
        donor_constraint_summary={},
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


def _finalize_bundle(ctx: ExperimentContext) -> None:
    _mark_completed_from_existing_outputs(ctx)
    legacy._write_config_files(ctx)
    _refresh_output_file_registry(ctx)
    summary = legacy._write_summary(ctx)
    summary.update(_trial_specs_metadata(ctx))
    write_json(summary, ctx.seed_dir / "summary.json")
    legacy._write_run_log(ctx)


def _mark_completed_from_existing_outputs(ctx: ExperimentContext) -> None:
    checks = {
        "trial_specs": [
            ctx.trial_specs_dir / "baseline_eval_trials.csv",
            ctx.trial_specs_dir / "delay_decode_train_trials.csv",
            ctx.trial_specs_dir / "delay_decode_test_trials.csv",
            ctx.trial_specs_dir / "dms_shuffle_trials.csv",
        ],
        "baseline": [
            ctx.metrics_dir / "panel_b_baseline_metrics_by_network.csv",
            ctx.metrics_dir / "supp_class_recall_by_digit.csv",
            ctx.metrics_dir / "supp_confusion_matrix_long.csv",
        ],
        "delay_decode": [
            ctx.metrics_dir / "panel_c_delay_decode_metrics.csv",
            ctx.metrics_dir / "supp_delay_decode_curve.csv",
            ctx.raw_dir / "panel_c_delay_decode_predictions.csv",
        ],
        "dms_delay_sweep": [
            ctx.metrics_dir / "supp_dms_delay_sweep_metrics.csv",
            ctx.metrics_dir / "supp_dms_delay_sweep_contrast.csv",
            ctx.raw_dir / "supp_dms_delay_sweep_trial_readout.csv",
        ],
        "dms_shuffle": [
            ctx.metrics_dir / "panel_d_condition_metrics.csv",
            ctx.metrics_dir / "panel_e_attribution_metrics.csv",
            ctx.raw_dir / "panel_d_dms_condition_trial_readout.csv",
        ],
        "firing_rate_control": [ctx.metrics_dir / "supp_phase_firing_rates.csv"],
    }
    for name, paths in checks.items():
        if all(path.exists() for path in paths):
            ctx.completed_modules[name] = True
    summary_path = ctx.seed_dir / "summary.json"
    if ctx.completed_modules.get("dms_shuffle") and summary_path.exists():
        existing = read_json(summary_path)
        donor_keys = (
            "donor_constraint_audit_available",
            "strict_all_three_distinct_donor",
            "n_donor_sample_conflict",
            "n_donor_probe_conflict",
            "n_sample_probe_conflict",
            "n_all_three_distinct_fail",
            "n_self_swap",
            "used_relaxed_rule",
            "donor_constraint_status",
        )
        ctx.donor_constraint_summary = {key: existing[key] for key in donor_keys if key in existing}
    ctx.cfg = replace(
        ctx.cfg,
        run_baseline=bool(ctx.completed_modules.get("baseline", False)),
        run_delay_decode=bool(ctx.completed_modules.get("delay_decode", False)),
        run_dms_delay_sweep=bool(ctx.completed_modules.get("dms_delay_sweep", False)),
        run_dms_shuffle=bool(ctx.completed_modules.get("dms_shuffle", False)),
        run_firing_rate_control=bool(ctx.completed_modules.get("firing_rate_control", False)),
    )


def _refresh_output_file_registry(ctx: ExperimentContext) -> None:
    for path in sorted(ctx.seed_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = legacy._rel(path, ctx.seed_dir)
        if rel.startswith("data/intermediates/"):
            continue
        if path.suffix.lower() in {".csv", ".json", ".txt"}:
            ctx.output_files[path.stem] = rel


def _config_from_args(args: argparse.Namespace) -> Fig1Config:
    smoke = bool(args.smoke)
    delay_points_ms = tuple(int(v) for v in str(args.delay_points_ms).split(",") if str(v).strip())
    dms_delay_sweep_ms = tuple(int(v) for v in str(args.dms_delay_sweep_ms).split(",") if str(v).strip())
    if smoke:
        delay_points_ms = delay_points_ms[:2]
        dms_delay_sweep_ms = dms_delay_sweep_ms[:2]
    output_root = _output_root_from_args(args)
    model_path = _resolve_model_path(args.model_path, str(args.model_path_glob), int(args.network_seed))
    dataset_root = _resolve_repo_path(args.dataset_root)
    return Fig1Config(
        model_path=str(model_path),
        dataset_root=str(dataset_root),
        output_root=str(output_root),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        baseline_eval_per_class=2 if smoke else int(args.baseline_eval_per_class),
        delay_decode_train_per_class=2 if smoke else int(args.delay_decode_train_per_class),
        delay_decode_test_per_class=2 if smoke else int(args.delay_decode_test_per_class),
        delay_points_ms=delay_points_ms,
        sample_ms=int(args.sample_ms),
        delay_ms=int(args.delay_ms),
        dms_sample_ms=int(args.dms_sample_ms),
        dms_delay_ms=int(args.dms_delay_ms),
        dms_delay_sweep_ms=dms_delay_sweep_ms,
        probe_ms=int(args.probe_ms),
        batch_size=min(int(args.batch_size), 16) if smoke else int(args.batch_size),
        dms_batch_size=min(int(args.dms_batch_size), 4) if smoke else int(args.dms_batch_size),
        dms_num_trials=min(int(args.dms_num_trials), 20) if smoke else int(args.dms_num_trials),
        firing_bin_ms=int(args.firing_bin_ms),
        delay_decode_backend=str(args.delay_decode_backend),
        delay_decode_torch_ridge_lambda=float(args.delay_decode_torch_ridge_lambda),
        run_baseline=False,
        run_delay_decode=False,
        run_dms_delay_sweep=False,
        run_dms_shuffle=False,
        run_firing_rate_control=False,
        save_debug_figures=bool(args.save_debug_figures),
        save_feature_cache=bool(args.save_feature_cache),
        show_progress=not bool(args.no_progress),
        use_encode_cache=not bool(args.no_encode_cache),
        enable_condition_batch=bool(args.enable_condition_batch),
        enable_gpu_metrics=bool(args.enable_gpu_metrics),
        shuffle_compat_mode=bool(args.shuffle_compat_mode),
        pure_substrate_only=bool(args.pure_substrate_only),
        shuffle_num_boot=int(args.shuffle_num_boot),
        shuffle_rng_offset=int(args.shuffle_rng_offset),
        smoke=smoke,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Fig.1 runtime-artifact task.")
    parser.add_argument("--task", required=True, choices=TASK_IDS)
    parser.add_argument("--reuse-artifacts", default="auto", choices=REUSE_MODES)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-path-glob", default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=str(Path(DEFAULT_OUTPUT_ROOT) / FIGURE_ID))
    parser.add_argument("--output-dir", default=None, help="Batch output root; the Fig.1 experiment id is appended unless a seed or figure root is supplied.")
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default=DEFAULT_PROJECT_DEFAULTS.runtime.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--save-feature-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-encode-cache", action="store_true")
    parser.add_argument("--enable-condition-batch", action="store_true")
    parser.add_argument("--enable-gpu-metrics", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shuffle-compat-mode", action="store_true")
    parser.add_argument("--pure-substrate-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-num-boot", type=int, default=1000)
    parser.add_argument("--shuffle-rng-offset", type=int, default=17)
    parser.add_argument("--baseline-eval-per-class", type=int, default=100)
    parser.add_argument("--delay-decode-train-per-class", type=int, default=50)
    parser.add_argument("--delay-decode-test-per-class", type=int, default=50)
    parser.add_argument("--delay-decode-backend", default="torch_linear_probe", choices=["torch_linear_probe", "sklearn_linear_svc"])
    parser.add_argument("--delay-decode-torch-ridge-lambda", type=float, default=1.0)
    parser.add_argument("--delay-points-ms", default="100,200,400,800,1200")
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=1200)
    parser.add_argument("--dms-sample-ms", type=int, default=200)
    parser.add_argument("--dms-delay-ms", type=int, default=400)
    parser.add_argument("--dms-delay-sweep-ms", default="100,200,400,800,1200")
    parser.add_argument("--probe-ms", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dms-batch-size", type=int, default=16)
    parser.add_argument("--dms-num-trials", type=int, default=100)
    parser.add_argument("--firing-bin-ms", type=int, default=50)
    return parser.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
