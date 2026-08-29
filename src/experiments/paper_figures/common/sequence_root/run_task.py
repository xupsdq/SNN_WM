from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS
from src.experiments.common.input_masks import entry_mask_from_image
from src.experiments.paper_figures.common.artifact_runtime import materialize_artifact
from src.experiments.paper_figures.common.sequence_root.artifacts import (
    cache_key_matches,
    default_artifact_root,
    load_sequence_specs_artifact,
    require_cache_key_match,
    save_root_bank_artifact,
    save_sequence_specs_artifact,
    task_artifact_dir,
    write_json,
)
from src.experiments.paper_figures.common.sequence_root.cache_keys import (
    build_shared_sequence_root_bank_cache_key,
    build_shared_sequence_specs_cache_key,
    cache_key_digest,
)
from src.experiments.paper_figures.common.sequence_root.schemas import (
    REUSE_MODES,
    TASK_ALL,
    TASK_FIG3_STATE_BANK_VIEW,
    TASK_FIG6_SEQUENCE_BANK_VIEW,
    TASK_IDS,
    TASK_SHARED_SEQUENCE_ROOT_BANK,
    TASK_SHARED_SEQUENCE_SPECS,
    normalize_reuse_mode,
)
from src.experiments.paper_figures.common.sequence_root.selection import (
    select_matched_nonpeak_mask,
    select_top_mask,
)
from src.experiments.paper_figures.common.specs.artifacts import materialize_spec_view
from src.experiments.paper_figures.fig3 import run_task as fig3_rt
from src.experiments.paper_figures.fig3.artifacts import save_sequence_specs_artifact as save_fig3_sequence_specs_artifact
from src.experiments.paper_figures.fig3.cache_keys import (
    build_sequence_specs_cache_key as build_fig3_sequence_specs_cache_key,
    build_state_bank_cache_key as build_fig3_state_bank_cache_key,
    cache_key_digest as fig3_cache_key_digest,
    sequence_specs_hash as fig3_sequence_specs_hash,
)
from src.experiments.paper_figures.fig3.schemas import TASK_SEQUENCE_TRIAL_SPECS as FIG3_TASK_SEQUENCE_TRIAL_SPECS
from src.experiments.paper_figures.fig3.schemas import TASK_STATE_BANK as FIG3_TASK_STATE_BANK
from src.experiments.paper_figures.fig3.subexperiments.trial_specs import build_sequence_trial_specs
from src.experiments.paper_figures.fig6 import run_task as fig6_rt
from src.experiments.paper_figures.fig6.artifacts import save_sequence_trials_artifact as save_fig6_sequence_trials_artifact
from src.experiments.paper_figures.fig6.artifacts import save_sequence_bank_artifact as save_fig6_sequence_bank_artifact
from src.experiments.paper_figures.fig6.cache_keys import (
    build_sequence_bank_cache_key as build_fig6_sequence_bank_cache_key,
    build_sequence_trials_cache_key as build_fig6_sequence_trials_cache_key,
    cache_key_digest as fig6_cache_key_digest,
    sequence_trials_hash as fig6_sequence_trials_hash,
)
from src.experiments.paper_figures.fig6.schemas import TASK_SEQUENCE_BANK as FIG6_TASK_SEQUENCE_BANK
from src.experiments.paper_figures.fig6.schemas import TASK_SEQUENCE_TRIALS as FIG6_TASK_SEQUENCE_TRIALS
from src.experiments.paper_figures.fig6.types import PeakAmplifiedReentryBank
from src.experiments.paper_figures.run_paper_figures import DEFAULT_DATASET_ROOT, DEFAULT_MODEL_PATH_GLOB


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = normalize_reuse_mode(args.reuse_artifacts)
    fig3_args = fig3_rt._parse_args(_fig3_argv(args, task=FIG3_TASK_STATE_BANK))
    fig3_cfg = fig3_rt._config_from_args(fig3_args)
    fig3_ctx = fig3_rt._build_context(fig3_cfg)
    artifact_root = _artifact_root_from_args(args, fig3_ctx.seed_dir)
    phase_timings: dict[str, float] = {}
    specs = _get_shared_sequence_specs(fig3_ctx, mode=mode, artifact_root=artifact_root, phase_timings=phase_timings)
    if args.task == TASK_SHARED_SEQUENCE_SPECS:
        _write_summary(args, artifact_root=artifact_root, task_dir=specs.root, digest=specs.digest, phase_timings=phase_timings)
        return 0
    if args.task in {TASK_SHARED_SEQUENCE_ROOT_BANK, TASK_FIG3_STATE_BANK_VIEW, TASK_FIG6_SEQUENCE_BANK_VIEW, TASK_ALL}:
        fig6_args = fig6_rt._parse_args(_fig6_argv(args, task=FIG6_TASK_SEQUENCE_BANK))
        fig6_cfg = fig6_rt._config_from_args(fig6_args)
        root_bank = _get_shared_root_bank(
            fig3_ctx,
            fig6_cfg=fig6_cfg,
            mode=mode,
            artifact_root=artifact_root,
            specs=specs,
            phase_timings=phase_timings,
        )
        _write_summary(args, artifact_root=artifact_root, task_dir=root_bank.root, digest=root_bank.digest, phase_timings=phase_timings)
        return 0
    raise ValueError(f"Unsupported shared sequence-root task: {args.task}")


def _materialize_sequence_root_artifact(
    *,
    mode: str,
    task_dir: Path,
    expected_key: Mapping[str, Any],
    load: Callable[[], Any],
    build: Callable[[], Any],
) -> Any:
    task_id = str(expected_key.get("task_id", task_dir.name))
    return materialize_artifact(
        mode=mode,
        task_dir=task_dir,
        expected_key=expected_key,
        load=load,
        build=build,
        recover_auto_load_errors=False,
        cache_is_reusable=lambda: cache_key_matches(task_dir, expected_key),
        require_reusable=lambda: require_cache_key_match(
            task_dir,
            expected_key,
            task_id=task_id,
        ),
    )


def _get_shared_sequence_specs(fig3_ctx, *, mode: str, artifact_root: Path, phase_timings: dict[str, float] | None = None):
    start = time.perf_counter()
    task_dir = task_artifact_dir(artifact_root, TASK_SHARED_SEQUENCE_SPECS)
    expected_key = build_shared_sequence_specs_cache_key(fig3_ctx.cfg)
    try:
        def build():
            sequence_trials, singleton_trials, partial_trials = build_sequence_trial_specs(fig3_ctx)
            return save_sequence_specs_artifact(
                task_dir,
                sequence_trials=sequence_trials,
                singleton_reference_trials=singleton_trials,
                partial_cue_trials=partial_trials,
                cache_key=expected_key,
            )

        return _materialize_sequence_root_artifact(
            mode=mode,
            task_dir=task_dir,
            expected_key=expected_key,
            load=lambda: load_sequence_specs_artifact(task_dir, expected_key=expected_key),
            build=build,
        )
    finally:
        if phase_timings is not None:
            phase_timings["shared_sequence_specs_seconds"] = float(time.perf_counter() - start)


def _get_shared_root_bank(fig3_ctx, *, fig6_cfg, mode: str, artifact_root: Path, specs, phase_timings: dict[str, float] | None = None):
    task_dir = task_artifact_dir(artifact_root, TASK_SHARED_SEQUENCE_ROOT_BANK)
    fig3_specs_hash = fig3_sequence_specs_hash(specs.sequence_trials, specs.singleton_reference_trials, specs.partial_cue_trials)
    fig6_specs_hash = fig6_sequence_trials_hash(specs.sequence_trials)
    fig3_bank_key = build_fig3_state_bank_cache_key(fig3_ctx.cfg, specs_hash=fig3_specs_hash)
    fig6_bank_key = build_fig6_sequence_bank_cache_key(fig6_cfg, sequence_trials_hash_value=fig6_specs_hash)
    expected_key = build_shared_sequence_root_bank_cache_key(
        fig3_ctx.cfg,
        specs_hash=specs.digest,
        fig3_state_bank_key_digest=fig3_cache_key_digest(fig3_bank_key),
        fig6_sequence_bank_key_digest=fig6_cache_key_digest(fig6_bank_key),
    )
    def load():
        from src.experiments.paper_figures.common.sequence_root.artifacts import load_root_bank_artifact

        start = time.perf_counter()
        artifact = load_root_bank_artifact(task_dir, expected_key=expected_key)
        if phase_timings is not None:
            phase_timings["shared_sequence_root_load_seconds"] = float(time.perf_counter() - start)
        return artifact

    build_requested = object()
    selected = _materialize_sequence_root_artifact(
        mode=mode,
        task_dir=task_dir,
        expected_key=expected_key,
        load=load,
        build=lambda: build_requested,
    )
    if selected is not build_requested:
        return selected

    shared_work = task_artifact_dir(artifact_root, "_shared_sequence_root_work")
    fig3_artifact_root = shared_work / "fig3"
    fig6_artifact_root = shared_work / "fig6"
    fig3_specs_dir = fig3_artifact_root / FIG3_TASK_SEQUENCE_TRIAL_SPECS
    fig6_specs_dir = fig6_artifact_root / FIG6_TASK_SEQUENCE_TRIALS
    fig3_state_dir = fig3_artifact_root / FIG3_TASK_STATE_BANK
    fig6_bank_dir = fig6_artifact_root / FIG6_TASK_SEQUENCE_BANK

    fig3_sequence_key = build_fig3_sequence_specs_cache_key(fig3_ctx.cfg)
    fig3_specs_artifact = save_fig3_sequence_specs_artifact(
        fig3_specs_dir,
        sequence_trials=specs.sequence_trials,
        singleton_reference_trials=specs.singleton_reference_trials,
        partial_cue_trials=specs.partial_cue_trials,
        cache_key=fig3_sequence_key,
    )
    if specs.spec_artifact is not None:
        materialize_spec_view(
            specs.spec_artifact,
            fig3_specs_dir,
            view_figure="fig3",
            view_task=FIG3_TASK_SEQUENCE_TRIAL_SPECS,
            view_artifact_digest=fig3_specs_artifact.digest,
            view_cache_key_digest=fig3_cache_key_digest(fig3_sequence_key),
        )
    fig3_rt._write_sequence_specs_to_bundle(
        fig3_ctx,
        specs.sequence_trials,
        specs.singleton_reference_trials,
        specs.partial_cue_trials,
    )
    start = time.perf_counter()
    fig3_artifact = fig3_rt._get_state_bank_artifact(
        fig3_ctx,
        specs.sequence_trials,
        mode="auto",
        artifact_root=fig3_artifact_root,
        specs_hash=fig3_specs_hash,
        write_compat_outputs=False,
    )
    if phase_timings is not None:
        phase_timings["fig3_state_bank_seconds"] = float(time.perf_counter() - start)

    start = time.perf_counter()
    fig6_ctx = fig6_rt._build_context(fig6_cfg)
    if phase_timings is not None:
        phase_timings["fig6_context_seconds"] = float(time.perf_counter() - start)
    start = time.perf_counter()
    fig6_specs_artifact = save_fig6_sequence_trials_artifact(
        fig6_specs_dir,
        sequence_trials=specs.sequence_trials,
        cache_key=build_fig6_sequence_trials_cache_key(fig6_cfg),
    )
    if specs.spec_artifact is not None:
        fig6_sequence_key = build_fig6_sequence_trials_cache_key(fig6_cfg)
        materialize_spec_view(
            specs.spec_artifact,
            fig6_specs_dir,
            view_figure="fig6",
            view_task=FIG6_TASK_SEQUENCE_TRIALS,
            view_artifact_digest=fig6_specs_artifact.digest,
            view_cache_key_digest=fig6_cache_key_digest(fig6_sequence_key),
        )
    fig6_rt._write_sequence_trials_to_bundle(fig6_ctx, specs.sequence_trials)
    if phase_timings is not None:
        phase_timings["fig6_sequence_specs_seconds"] = float(time.perf_counter() - start)
    start = time.perf_counter()
    fig6_bank = _materialize_fig6_bank_from_fig3(fig6_ctx, specs.sequence_trials, fig3_artifact.bank)
    if phase_timings is not None:
        phase_timings["fig6_materialize_seconds"] = float(time.perf_counter() - start)
    start = time.perf_counter()
    fig6_artifact = save_fig6_sequence_bank_artifact(
        fig6_bank_dir,
        fig6_bank,
        raw_dir=fig6_ctx.raw_dir,
        cache_key=fig6_bank_key,
        network_seed=int(fig6_cfg.network_seed),
    )
    if phase_timings is not None:
        phase_timings["fig6_save_seconds"] = float(time.perf_counter() - start)
    start = time.perf_counter()
    root_artifact = save_root_bank_artifact(
        task_dir,
        specs=specs,
        fig3_state_bank_dir=fig3_state_dir,
        fig6_sequence_bank_dir=fig6_bank_dir,
        cache_key=expected_key,
        fig3_digest=fig3_artifact.digest,
        fig6_digest=fig6_artifact.digest,
        fig3_cache_key_digest=fig3_cache_key_digest(fig3_bank_key),
        fig6_cache_key_digest=fig6_cache_key_digest(fig6_bank_key),
    )
    if phase_timings is not None:
        phase_timings["root_artifact_save_seconds"] = float(time.perf_counter() - start)
    return root_artifact


def _materialize_fig6_bank_from_fig3(fig6_ctx, sequence_trials: pd.DataFrame, fig3_bank) -> PeakAmplifiedReentryBank:
    seq_ids = sorted(int(value) for value in sequence_trials["sequence_id"].unique())
    n_seq = len(seq_ids)
    n_units = 28 * 28
    max_len = int(max(sequence_trials["seq_len"])) if len(sequence_trials) else 0
    update_count = np.zeros((n_seq, n_units), dtype=np.float32)
    last_update_position = np.zeros((n_seq, n_units), dtype=np.int16)
    time_since_last_update = np.zeros((n_seq, n_units), dtype=np.int16)
    update_exposure_by_item = np.zeros((n_seq, max_len, n_units), dtype=np.float32)
    item_activation_history = np.zeros_like(update_exposure_by_item)
    g_baseline = np.zeros((n_seq, n_units), dtype=np.float32)
    g_final = np.zeros((n_seq, n_units), dtype=np.float32)
    delta_support = np.zeros((n_seq, n_units), dtype=np.float32)
    peak_mask = np.zeros((n_seq, n_units), dtype=bool)
    nonpeak_mask = np.zeros((n_seq, n_units), dtype=bool)
    prior_updated_mask = np.zeros((n_seq, n_units), dtype=bool)
    boundaries: dict[int, Mapping[str, Mapping[str, Any]]] = {}
    manifest_rows: list[dict[str, Any]] = []
    sequence_meta_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], Any] = {}

    for row_idx, sequence_id in enumerate(seq_ids):
        group = sequence_trials[sequence_trials["sequence_id"].astype(int).eq(int(sequence_id))].sort_values("stage_k")
        seq_len = int(group["seq_len"].iloc[0])
        image_ids = [int(v) for v in group["item_image_id"].tolist()]
        labels = [int(v) for v in group["item_label"].tolist()]
        masks = np.stack(
            [
                entry_mask_from_image(
                    fig6_ctx.dataset[int(image_id)][0],
                    mode=str(fig6_ctx.cfg.real_probe_entry_mode),
                    encoder=fig6_ctx.encoder,
                    steps=int(fig6_ctx.cfg.sample_steps),
                    device=fig6_ctx.device,
                    foreground_threshold=float(fig6_ctx.cfg.foreground_threshold),
                    cache=encode_cache,
                    image_id=int(image_id),
                )
                for image_id in image_ids
            ],
            axis=0,
        )
        exposure = masks.reshape(seq_len, -1).astype(np.float32)
        update_exposure_by_item[row_idx, :seq_len, :] = exposure
        item_activation_history[row_idx, :seq_len, :] = exposure
        update_count[row_idx] = exposure.sum(axis=0)
        for pos in range(seq_len):
            active = exposure[pos] > 0
            last_update_position[row_idx, active] = pos + 1
        time_since_last_update[row_idx] = np.where(last_update_position[row_idx] > 0, seq_len - last_update_position[row_idx], seq_len + 1)
        prior_updated_mask[row_idx] = update_count[row_idx] > 0

        g_baseline[row_idx] = _fig3_layer1_support(fig3_bank.arrays[int(sequence_id)]["S0"]["layer1"]["g"]).reshape(-1)
        g_final[row_idx] = _fig3_layer1_support(fig3_bank.arrays[int(sequence_id)]["S_final"]["layer1"]["g"]).reshape(-1)
        delta_support[row_idx] = g_final[row_idx] - g_baseline[row_idx]
        peaks = select_top_mask(
            delta_support[row_idx].reshape(28, 28),
            fig6_ctx.cfg.peak_q,
            positive=delta_support[row_idx].reshape(28, 28) > 0,
        )
        peak_mask[row_idx] = peaks.reshape(-1)
        nonpeak_mask[row_idx] = select_matched_nonpeak_mask(
            peak_mask[row_idx],
            prior_updated_mask[row_idx],
            int(fig6_ctx.cfg.network_seed) + sequence_id,
        )
        boundaries[int(sequence_id)] = fig3_bank.boundaries[int(sequence_id)]["S_final"]
        sequence_meta_rows.append(
            {
                "network_seed": int(fig6_ctx.cfg.network_seed),
                "sequence_id": int(sequence_id),
                "seq_len": int(seq_len),
                "ordered_item_ids": ";".join(map(str, image_ids)),
                "ordered_item_labels": ";".join(map(str, labels)),
            }
        )
        for state_condition, stage_k, arrs in (
            ("S0", 0, {"G_baseline": g_baseline[row_idx]}),
            ("S_final", seq_len, {"G_final": g_final[row_idx], "delta_support": delta_support[row_idx]}),
        ):
            for key, arr in arrs.items():
                manifest_rows.append(
                    {
                        "network_seed": int(fig6_ctx.cfg.network_seed),
                        "sequence_id": int(sequence_id),
                        "seq_len": int(seq_len),
                        "state_condition": state_condition,
                        "stage_k": int(stage_k),
                        "layer": "layer1",
                        "state_variable": "g" if key != "delta_support" else "delta_support",
                        "shape": "28x28",
                        "storage_file": "final_support_maps.npz",
                        "storage_key": f"{key}_sequence_{sequence_id}",
                        "captured_after": state_condition,
                        "sample_ms": int(fig6_ctx.cfg.sample_ms),
                        "delay_ms": int(fig6_ctx.cfg.delay_ms),
                    }
                )

    fig6_ctx.raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(fig6_ctx.raw_dir / "state_bank_manifest.csv", index=False, encoding="utf-8")
    np.savez_compressed(
        fig6_ctx.raw_dir / "update_history_matrix.npz",
        update_count=update_count,
        last_update_position=last_update_position,
        time_since_last_update=time_since_last_update,
        update_exposure_by_item=update_exposure_by_item,
        item_activation_history=item_activation_history,
        unit_ids=np.arange(n_units, dtype=np.int32),
        sequence_ids=np.asarray(seq_ids, dtype=np.int32),
    )
    np.savez_compressed(
        fig6_ctx.raw_dir / "final_support_maps.npz",
        G_baseline=g_baseline,
        G_final=g_final,
        delta_support=delta_support,
        peak_mask=peak_mask.astype(np.uint8),
        nonpeak_mask=nonpeak_mask.astype(np.uint8),
        unit_ids=np.arange(n_units, dtype=np.int32),
        sequence_ids=np.asarray(seq_ids, dtype=np.int32),
    )
    fig6_ctx.completed_modules["sequence_bank"] = True
    fig6_ctx.n_sequences = int(n_seq)
    return PeakAmplifiedReentryBank(
        sequence_trials=sequence_trials.reset_index(drop=True).copy(),
        sequence_meta=pd.DataFrame(sequence_meta_rows),
        probe_trials=pd.DataFrame(),
        matched_groups=pd.DataFrame(),
        update_count=update_count,
        last_update_position=last_update_position,
        time_since_last_update=time_since_last_update,
        update_exposure_by_item=update_exposure_by_item,
        item_activation_history=item_activation_history,
        g_baseline=g_baseline,
        g_final=g_final,
        delta_support=delta_support,
        peak_mask=peak_mask,
        nonpeak_mask=nonpeak_mask,
        prior_updated_mask=prior_updated_mask,
        boundaries=boundaries,
        reentry_metrics=pd.DataFrame(),
        downstream_metrics=pd.DataFrame(),
    )


def _fig3_layer1_support(flat_g: np.ndarray) -> np.ndarray:
    return np.asarray(flat_g, dtype=np.float32).reshape(2, 28, 28).mean(axis=0)


def _artifact_root_from_args(args: argparse.Namespace, seed_dir: Path) -> Path:
    if args.artifact_root:
        root = Path(args.artifact_root)
        root = root if root.is_absolute() else (Path.cwd() / root)
    else:
        root = default_artifact_root(seed_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (DEFAULT_PROJECT_DEFAULTS.paths.repo_root / path).resolve()


def _write_summary(args: argparse.Namespace, *, artifact_root: Path, task_dir: Path, digest: str, phase_timings: Mapping[str, float] | None = None) -> None:
    summary_path = Path(args.output_dir) if args.output_dir else Path(args.output_root)
    output_root = _resolve_repo_path(summary_path)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "task": str(args.task),
            "artifact_root": str(Path(artifact_root).resolve()),
            "task_artifact_dir": str(Path(task_dir).resolve()),
            "artifact_digest": str(digest),
            "phase_timings": {str(key): float(value) for key, value in (phase_timings or {}).items()},
            "command": " ".join(sys.argv if args.argv_is_real else ["run_task"]),
        },
        output_root / "shared_sequence_root_summary.json",
    )


def _fig3_argv(args: argparse.Namespace, *, task: str) -> list[str]:
    argv = _common_figure_argv(args, task=task, output_dir=str(_resolve_repo_path(args.output_root) / "fig3"))
    argv.extend(
        [
            "--main-sequence-length",
            str(args.main_sequence_length),
            "--primary-sequence-length",
            str(args.primary_sequence_length),
            "--partial-cue-keep-fraction",
            str(args.partial_cue_keep_fraction),
            "--target-position",
            str(args.target_position),
            "--batch-size",
            str(args.batch_size),
            "--state-bank-singleton-batch-size",
            str(args.state_bank_singleton_batch_size),
        ]
    )
    if args.smoke:
        argv.append("--smoke")
    if args.no_progress:
        argv.append("--no-progress")
    if args.enable_state_bank_batch:
        argv.append("--enable-state-bank-batch")
    return argv


def _fig6_argv(args: argparse.Namespace, *, task: str) -> list[str]:
    argv = _common_figure_argv(args, task=task, output_dir=str(_resolve_repo_path(args.output_root) / "fig6"))
    argv.extend(
        [
            "--primary-sequence-length",
            str(args.primary_sequence_length),
            "--batch-size",
            str(args.batch_size),
            "--peak-q",
            str(args.peak_q),
            "--foreground-threshold",
            str(args.foreground_threshold),
        ]
    )
    if args.smoke:
        argv.append("--smoke")
    if args.no_progress:
        argv.append("--no-progress")
    if args.enable_sequence_bank_batch:
        argv.append("--enable-sequence-bank-batch")
    return argv


def _common_figure_argv(args: argparse.Namespace, *, task: str, output_dir: str) -> list[str]:
    argv = [
        "--task",
        task,
        "--reuse-artifacts",
        "auto",
        "--output-dir",
        output_dir,
        "--network-seed",
        str(args.network_seed),
        "--device",
        str(args.device),
        "--split",
        str(args.split),
        "--dataset-root",
        str(args.dataset_root),
        "--model-path-glob",
        str(args.model_path_glob),
        "--sequence-lengths",
        str(args.sequence_lengths),
        "--num-sequences",
        str(args.num_sequences),
        "--sample-ms",
        str(args.sample_ms),
        "--delay-ms",
        str(args.delay_ms),
    ]
    if args.model_path:
        argv.extend(["--model-path", str(args.model_path)])
    return argv


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared Fig.3/Fig.6 sequence-root producer.", allow_abbrev=False)
    parser.add_argument("--task", required=True, choices=TASK_IDS)
    parser.add_argument("--reuse-artifacts", default="auto", choices=REUSE_MODES)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-path-glob", default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--output-root",
        default="results/multi_seed_rollout/shared_sequence_root",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default=DEFAULT_PROJECT_DEFAULTS.runtime.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sequence-lengths", default="3,5,7,10")
    parser.add_argument("--primary-sequence-length", type=int, default=7)
    parser.add_argument("--main-sequence-length", type=int, default=10)
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--state-bank-singleton-batch-size",
        type=int,
        default=4,
        help="Maximum row batch for Fig.3 singleton-boundary capture while the main sequence capture uses --batch-size.",
    )
    parser.add_argument("--partial-cue-keep-fraction", type=float, default=0.10)
    parser.add_argument("--target-position", default="K-1")
    parser.add_argument("--peak-q", type=float, default=0.20)
    parser.add_argument("--foreground-threshold", type=float, default=0.1)
    parser.add_argument("--enable-state-bank-batch", action="store_true", help="Forward Fig.3 same-length state-bank batching to the shared root producer.")
    parser.add_argument("--enable-sequence-bank-batch", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    parsed.argv_is_real = argv is None
    if parsed.output_dir:
        parsed.output_root = parsed.output_dir
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
