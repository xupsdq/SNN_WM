from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - runtime validation reports missing torch.
    torch = None  # type: ignore[assignment]

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS
from src.config.units import ms
from src.core.network import SDNN_Network
from src.data.encoding import DoGSpikeEncoder
from src.experiments.common.dataset import build_class_index
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as fig3
from src.experiments.paper_figures import fig4_overlap_reentry_experiment as fig4
from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as fig6
from src.experiments.paper_figures.fig6.types import PeakAmplifiedReentryBank
from src.experiments.paper_figures.run_paper_figures import (
    BENCHMARK_PROFILE_ARGS_BY_FIG,
    BENCHMARK_PROFILE_NONE,
    DEFAULT_MODEL_PATH_GLOB,
    NetworkCheckpoint,
    _resolve_repo_path,
    discover_checkpoints,
    select_checkpoints,
)
from src.experiments.runners._common import _resolve_runtime_python


DEFAULT_SOURCE_ROOT = "results/paper_figure_multi_seed"
DEFAULT_OUTPUT_ROOT = "results/paper_figure_issue_patch_shared_optimized_20260526"
DEFAULT_DATASET_ROOT = str(DEFAULT_PROJECT_DEFAULTS.paths.dataset_root)
PATCH_VERSION = "shared_optimized_issue_patch_20260526"
DEFAULT_FIGS = "fig3,fig4,fig6"

EXPERIMENT_IDS = {
    "fig3": "fig3_multiitem_peak_landscape",
    "fig4": "fig4_overlap_reentry",
    "fig6": "fig6_peak_amplified_reentry",
}
MODULES = {
    "fig3": "src.experiments.paper_figures.fig3_multiitem_peak_landscape_experiment",
    "fig4": "src.experiments.paper_figures.fig4_overlap_reentry_experiment",
    "fig6": "src.experiments.paper_figures.fig6_peak_amplified_reentry_experiment",
}
PLOT_FIG_IDS = {
    "fig3": ("fig3", "fig3_supp"),
    "fig4": ("fig4", "fig4_supp"),
    "fig6": ("fig6", "fig6_supp"),
}

FIG3_FLAGS = (
    "--run-state-bank",
    "--run-neutral-ping",
    "--run-weak-probe",
    "--enable-condition-batch",
)
FIG4_FLAGS = (
    "--run-similarity-entry",
    "--run-overlap-localization",
    "--run-overlap-accuracy-identification",
    "--run-overlap-perturbation",
    "--require-distinct-pair-labels",
)
FIG6_FLAGS = (
    "--run-sequence-bank",
    "--run-field-ping-readout",
    "--run-global-ping-score-spike-prediction",
    "--run-real-probe-score-spike-deflection",
    "--run-overlap-gated-stsp-recruitment",
    "--run-high-stsp-overlap-ablation",
    "--run-supplement",
)
FIG6_SAFE_BATCH_FLAGS = (
    "--enable-sequence-bank-batch",
    "--enable-high-stsp-ablation-batch",
    "--enable-leave-one-out-batch",
)

COMMON_PATCH_METADATA = (
    "run_config.json",
    "summary.json",
    "artifact_manifest.json",
    "config/run_config.json",
    "config/figure_requirements.json",
    "meta/run_info.json",
    "run_log.txt",
)

FIG3_OVERLAY_FILES = (
    "data/trial_specs/sequence_trials.csv",
    "data/trial_specs/singleton_reference_trials.csv",
    "data/trial_specs/partial_cue_trials.csv",
    "data/trial_specs/weak_probe_targets.csv",
    "data/trial_specs/weak_probe_masks.csv",
    "data/raw/panel_d_neutral_ping_trial_readout.csv",
    "data/raw/panel_e_neutral_ping_trial_readout.csv",
    "data/raw/panel_e_weak_probe_trial_readout.csv",
    "data/raw/panel_f_weak_probe_trial_readout.csv",
    "data/metrics/panel_d_ping_position_distribution.csv",
    "data/metrics/panel_d_ping_class_distribution.csv",
    "data/metrics/panel_d_ping_summary.csv",
    "data/metrics/panel_e_ping_position_distribution.csv",
    "data/metrics/panel_e_ping_class_distribution.csv",
    "data/metrics/panel_e_ping_summary.csv",
    "data/metrics/supp_ping_recency_diagnostics.csv",
    "data/metrics/panel_e_weak_probe_metrics.csv",
    "data/metrics/panel_e_weak_probe_auc_metrics.csv",
    "data/metrics/panel_e_weak_probe_memory_gain.csv",
    "data/metrics/panel_e_weak_probe_position_stratified_metrics.csv",
    "data/metrics/panel_f_weak_probe_metrics.csv",
    "data/metrics/panel_f_weak_probe_auc_metrics.csv",
    "data/metrics/panel_f_weak_probe_memory_gain.csv",
)
FIG4_OVERLAY_FILES = (
    "data/trial_specs/pair_trials.csv",
    "data/trial_specs/pair_candidate_pool.csv",
    "data/trial_specs/overlap_matched_pairs.csv",
    "data/trial_specs/perturbation_masks.csv",
    "data/raw/rollout_manifest.csv",
    "data/raw/overlap_perturbation_rollout_manifest.csv",
    "data/raw/panel_d_l1_stsp_overlap_perturbation_trial_readout.csv",
    "data/metrics/panel_b_similarity_entry_metrics.csv",
    "data/metrics/panel_b_similarity_bin_summary.csv",
    "data/metrics/panel_b_similarity_accuracy_drop_summary.csv",
    "data/metrics/panel_c_overlap_localization_metrics.csv",
    "data/metrics/panel_c_overlap_matched_comparison.csv",
    "data/metrics/panel_c_high_similarity_overlap_accuracy_drop.csv",
    "data/metrics/panel_c_high_similarity_overlap_accuracy_drop_summary.csv",
    "data/metrics/panel_c_high_similarity_overlap_accuracy_drop_contrast.csv",
    "data/metrics/panel_d_l1_stsp_overlap_perturbation_audit.csv",
    "data/metrics/panel_d_l1_stsp_overlap_perturbation_summary.csv",
    "data/metrics/panel_d_l1_stsp_overlap_perturbation_contrast.csv",
    "data/metrics/panel_d_overlap_perturbation_metrics.csv",
    "data/metrics/panel_d_overlap_perturbation_summary.csv",
    "data/metrics/panel_d_overlap_perturbation_contrast.csv",
    "data/metrics/panel_d_overlap_accuracy_pair_table.csv",
    "data/metrics/panel_d_iso_similarity_matched_pairs.csv",
    "data/metrics/panel_d_overlap_accuracy_permutation_null.csv",
    "data/metrics/panel_d_overlap_accuracy_contrast_by_network.csv",
    "data/metrics/panel_d_matching_balance_diagnostics.csv",
    "data/metrics/supp_s7_similarity_bin_full_trend.csv",
    "data/metrics/supp_s7_overlap_matching_diagnostics.csv",
    "data/metrics/supp_s7_iso_similarity_matched_pairs.csv",
    "data/metrics/supp_s7_iso_similarity_overlap_contrast.csv",
    "data/metrics/supp_s7_iso_similarity_permutation_null.csv",
    "data/metrics/supp_s7_overlap_matching_balance_diagnostics.csv",
    "data/metrics/supp_s7_overlap_regression_controls.csv",
    "data/metrics/supp_s7_random_nonoverlap_perturbation_controls.csv",
    "data/metrics/supp_overlap_similarity_2x2.csv",
    "data/metrics/supp_overlap_excess_accuracy_metrics.csv",
    "data/metrics/supp_overlap_accuracy_regression.csv",
    "data/metrics/supp_overlap_matching_diagnostics.csv",
    "data/metrics/supp_overlap_mask_application_audit.csv",
    "data/metrics/supp_overlap_preserving_perturbation_metrics.csv",
    "data/metrics/supp_overlap_preserving_perturbation_summary.csv",
)
FIG4_ROW_MERGE_FILES = (
    "data/metrics/supp_overlap_similarity_regression.csv",
)
FIG6_OVERLAY_FILES = (
    "data/trial_specs/sequence_trials.csv",
    "data/trial_specs/probe_candidate_trials.csv",
    "data/trial_specs/matched_raw_overlap_groups.csv",
    "data/raw/state_bank_manifest.csv",
    "data/raw/update_history_matrix.npz",
    "data/raw/final_support_maps.npz",
    "data/raw/panel_f_global_mechanism_metadata.json",
    "data/metrics/fig6_gain_ratio_audit.csv",
    "data/metrics/fig6_entry_score_audit.csv",
    "data/metrics/panel_a_high_stsp_overlap_ablation.csv",
    "data/metrics/panel_a_high_stsp_overlap_ablation_summary.csv",
    "data/metrics/panel_b_region_ping_readout_bias.csv",
    "data/metrics/panel_c_global_ping_score_spike_prediction.csv",
    "data/metrics/panel_d_real_probe_score_spike_deflection.csv",
    "data/metrics/panel_e_overlap_gated_stsp_recruitment.csv",
    "data/metrics/panel_e_overlap_gated_stsp_interaction.csv",
    "data/metrics/panel_f_high_stsp_overlap_ablation.csv",
    "data/metrics/panel_f_high_stsp_overlap_ablation_summary.csv",
    "data/metrics/supp_s11a_score_input_ping_audit.csv",
    "data/metrics/supp_s11b_global_ping_count_endpoint.csv",
    "data/metrics/supp_s11c_real_probe_window_robustness.csv",
    "data/metrics/supp_s11d_overlap_interaction_window_robustness.csv",
    "data/metrics/supp_s11e_overlap_site_availability.csv",
    "data/metrics/supp_s11f_high_stsp_ablation_paired_difference.csv",
    "data/metrics/supp_s11g_score_shuffle_null.csv",
    "data/metrics/supp_s11h_threshold_sensitivity.csv",
)
OVERLAY_FILES = {
    "fig3": FIG3_OVERLAY_FILES,
    "fig4": FIG4_OVERLAY_FILES,
    "fig6": FIG6_OVERLAY_FILES,
}
ROW_MERGE_FILES = {"fig4": FIG4_ROW_MERGE_FILES}
COMPARE_FILES = {
    fig_id: tuple(files) + tuple(ROW_MERGE_FILES.get(fig_id, ()))
    for fig_id, files in OVERLAY_FILES.items()
}

KEY_COLUMNS = (
    "network_seed",
    "sequence_id",
    "seq_len",
    "stage_k",
    "pair_id",
    "probe_id",
    "mask_id",
    "condition",
    "state_condition",
    "memory_condition",
    "entry_condition",
    "early_window_ms",
    "score_quantile_bin",
    "stsp_group",
    "overlap_group",
    "metric",
    "outcome",
    "term",
    "loss_condition",
)
DISCRETE_COLUMN_TOKENS = (
    "prediction",
    "predicted_label",
    "correct",
    "correctness",
    "first_fire_time",
    "silent",
    "readout_label",
    "readout_serial_position",
    "acc_drop",
)


@dataclass(frozen=True)
class SharedSeedResources:
    seed: int
    model_path: Path
    device: Any
    dataset: Any
    class_index: dict[int, list[int]]
    net: Any
    encoder: Any
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RunRecord:
    variant: str
    fig_id: str
    seed: int
    command: str
    status: str
    returncode: int | None
    elapsed_seconds: float
    output_root: Path
    stdout_tail: str = ""
    stderr_tail: str = ""
    optimization_notes: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _json_safe(row.get(name, "")) for name in fieldnames})


def _parse_figs(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in str(raw).split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"3", "fig3"}:
            fig_id = "fig3"
        elif token in {"4", "fig4"}:
            fig_id = "fig4"
        elif token in {"6", "fig6"}:
            fig_id = "fig6"
        else:
            raise ValueError(f"Unsupported figure: {item}")
        if fig_id not in seen:
            out.append(fig_id)
            seen.add(fig_id)
    if not out:
        raise ValueError("--figs must contain at least one of fig3, fig4, fig6.")
    return out


def _seed_name(seed: int) -> str:
    return f"seed_{int(seed):03d}"


def _experiment_root(base_root: Path, fig_id: str) -> Path:
    exp_id = EXPERIMENT_IDS[fig_id]
    return base_root / exp_id / exp_id


def _seed_dir(base_root: Path, fig_id: str, seed: int) -> Path:
    return _experiment_root(base_root, fig_id) / _seed_name(seed)


def _source_seed_dir_candidates(source_root: Path, fig_id: str, seed: int) -> list[tuple[str, Path]]:
    exp_id = EXPERIMENT_IDS[fig_id]
    seed_name = _seed_name(seed)
    return [
        ("nested", source_root / exp_id / exp_id / seed_name),
        ("flat", source_root / exp_id / seed_name),
    ]


def _resolve_source_seed_dir(source_root: Path, fig_id: str, seed: int) -> tuple[Path, str, list[tuple[str, Path]]]:
    candidates = _source_seed_dir_candidates(source_root, fig_id, seed)
    for layout, path in candidates:
        if path.is_dir():
            return path, layout, candidates
    return candidates[0][1], "missing", candidates


def _variant_base(output_root: Path, variant: str) -> Path:
    if variant == "final":
        return output_root
    compact_names = {
        "reference_serial": "ref",
        "optimized_staging": "opt",
    }
    if variant in compact_names:
        return output_root / f"_{compact_names[variant]}"
    return output_root / f"_{variant}"


def _profile_args(fig_id: str, profile: str) -> tuple[str, ...]:
    if profile == BENCHMARK_PROFILE_NONE:
        return ()
    by_fig = BENCHMARK_PROFILE_ARGS_BY_FIG.get(profile, {})
    return tuple(by_fig.get(fig_id, ()))


def _batch_size_for_fig(args: argparse.Namespace, fig_id: str) -> int | None:
    fig_value = getattr(args, f"{fig_id}_batch_size", None)
    if fig_value is not None:
        return int(fig_value)
    common_value = getattr(args, "experiment_batch_size", None)
    if common_value is not None:
        return int(common_value)
    return None


def _base_argv(
    *,
    fig_id: str,
    output_base: Path,
    checkpoint: NetworkCheckpoint,
    dataset_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    argv = [
        "--model-path",
        str(checkpoint.model_path),
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(_experiment_root(output_base, fig_id)),
        "--network-seed",
        str(int(checkpoint.seed)),
        "--device",
        str(args.device),
        "--split",
        str(args.split),
        *_profile_args(fig_id, str(args.benchmark_profile)),
    ]
    batch_size = _batch_size_for_fig(args, fig_id)
    if batch_size is not None:
        argv.extend(["--batch-size", str(batch_size)])
    if bool(args.smoke):
        argv.append("--smoke")
    if bool(args.no_progress):
        argv.append("--no-progress")
    if fig_id == "fig3":
        argv.extend(FIG3_FLAGS)
        argv.extend(["--sequence-lengths", str(args.shared_sequence_lengths)])
        argv.extend(["--functional-restore-mode", "stsp_only"])
    elif fig_id == "fig4":
        argv.extend(FIG4_FLAGS)
        if getattr(args, "fig4_l3_region_batch_size", None) is not None:
            argv.extend(["--l3-region-batch-size", str(int(args.fig4_l3_region_batch_size))])
    elif fig_id == "fig6":
        argv.extend(FIG6_FLAGS)
        argv.extend(["--sequence-lengths", str(args.shared_sequence_lengths)])
        argv.extend(["--functional-restore-mode", "stsp_only"])
        if bool(args.fig6_safe_gpu_batching):
            argv.extend(FIG6_SAFE_BATCH_FLAGS)
    else:  # pragma: no cover - parser guards this.
        raise ValueError(fig_id)
    return argv


def _module_for_fig(fig_id: str) -> Any:
    if fig_id == "fig3":
        return fig3
    if fig_id == "fig4":
        return fig4
    if fig_id == "fig6":
        return fig6
    raise ValueError(fig_id)


def _config_from_argv(fig_id: str, argv: Sequence[str]) -> Any:
    module = _module_for_fig(fig_id)
    parsed = module._parse_args(list(argv))
    cfg = module._config_from_args(parsed)
    if fig_id == "fig6":
        cfg = fig6._effective_config(cfg)
    return cfg


def _command_for_fig(runtime_python: Path, fig_id: str, argv: Sequence[str]) -> list[str]:
    return [str(runtime_python), "-m", MODULES[fig_id], *[str(item) for item in argv]]


def _safe_clean_seed_dir(base_root: Path, fig_id: str, seed: int, *, dry_run: bool) -> None:
    seed_dir = _seed_dir(base_root, fig_id, seed)
    if dry_run or not seed_dir.exists():
        return
    resolved = seed_dir.resolve()
    allowed = base_root.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise RuntimeError(f"Refusing to clean seed dir outside variant root: {seed_dir}")
    shutil.rmtree(seed_dir)


def _run_subprocess(command: Sequence[str], *, cwd: Path, dry_run: bool, log_path: Path | None = None) -> dict[str, Any]:
    command_text = subprocess.list2cmdline([str(part) for part in command])
    if dry_run:
        return {
            "command": command_text,
            "status": "dry_run",
            "returncode": None,
            "elapsed_seconds": 0.0,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    started = time.perf_counter()
    proc = subprocess.run([str(part) for part in command], cwd=cwd, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {command_text}\n\n[stdout]\n{proc.stdout}\n\n[stderr]\n{proc.stderr}\n",
            encoding="utf-8",
        )
    return {
        "command": command_text,
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(elapsed),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _run_reference_serial(
    *,
    runtime_python: Path,
    fig_id: str,
    output_base: Path,
    checkpoint: NetworkCheckpoint,
    dataset_root: Path,
    args: argparse.Namespace,
    repo_root: Path,
) -> RunRecord:
    seed = int(checkpoint.seed)
    _safe_clean_seed_dir(output_base, fig_id, seed, dry_run=bool(args.dry_run))
    argv = _base_argv(fig_id=fig_id, output_base=output_base, checkpoint=checkpoint, dataset_root=dataset_root, args=args)
    command = _command_for_fig(runtime_python, fig_id, argv)
    result = _run_subprocess(
        command,
        cwd=repo_root,
        dry_run=bool(args.dry_run),
        log_path=output_base / "logs" / f"{fig_id}_reference_seed_{seed:03d}.log",
    )
    return RunRecord(
        variant="reference_serial",
        fig_id=fig_id,
        seed=seed,
        command=str(result["command"]),
        status=str(result["status"]),
        returncode=result["returncode"],
        elapsed_seconds=float(result["elapsed_seconds"]),
        output_root=_experiment_root(output_base, fig_id),
        stdout_tail=str(result.get("stdout_tail", "")),
        stderr_tail=str(result.get("stderr_tail", "")),
    )


def _clone_seed_bundle(source_root: Path, output_root: Path, fig_id: str, seed: int, *, dry_run: bool) -> dict[str, Any]:
    src, source_layout, candidates = _resolve_source_seed_dir(source_root, fig_id, seed)
    dst = _seed_dir(output_root, fig_id, seed)
    if not src.is_dir():
        checked = "; ".join(f"{layout}={path}" for layout, path in candidates)
        raise FileNotFoundError(f"Missing source seed bundle for {fig_id} {_seed_name(seed)}. Checked: {checked}")
    if dry_run:
        return {"fig_id": fig_id, "seed": int(seed), "source": str(src), "source_layout": source_layout, "destination": str(dst), "status": "dry_run"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return {"fig_id": fig_id, "seed": int(seed), "source": str(src), "source_layout": source_layout, "destination": str(dst), "status": "copied"}


def _load_shared_resources(
    *,
    checkpoint: NetworkCheckpoint,
    configs: Mapping[str, Any],
    dataset_root: Path,
    device_arg: str,
    split: str,
    smoke: bool,
) -> SharedSeedResources:
    seed = int(checkpoint.seed)
    seed_everything(seed)
    device = resolve_device(device_arg)
    dataset = load_mnist_skeleton_dataset(str(dataset_root), split)
    class_index = build_class_index(dataset, 10)
    max_duration_ms = max(
        int(getattr(cfg, "sample_ms", 100))
        for cfg in configs.values()
    )
    for cfg in configs.values():
        for name in ("weak_probe_ms", "probe_ms", "ping_ms", "global_ping_ms"):
            if hasattr(cfg, name):
                max_duration_ms = max(max_duration_ms, int(getattr(cfg, name)))
    warnings: list[str] = []
    model_path = Path(checkpoint.model_path)
    if model_path.exists():
        net, encoder = load_model_and_encoder(str(model_path), device=device, dt=0.001, max_duration_ms=max_duration_ms)
    elif smoke:
        if torch is None:
            raise RuntimeError("PyTorch is required to construct a smoke SDNN_Network fallback.")
        net = SDNN_Network(device=str(device)).to(device)
        net.eval()
        encoder = DoGSpikeEncoder(dt=0.001, max_duration=max_duration_ms * ms, device=str(device))
        warnings.append("Model checkpoint missing; smoke mode used an untrained repo SDNN_Network instance.")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return SharedSeedResources(
        seed=seed,
        model_path=model_path,
        device=device,
        dataset=dataset,
        class_index=class_index,
        net=net,
        encoder=encoder,
        warnings=tuple(warnings),
    )


def _fig_context(fig_id: str, cfg: Any, resources: SharedSeedResources, *, command_text: str) -> tuple[Any, Mapping[str, Any]]:
    module = _module_for_fig(fig_id)
    seed_dir = module._resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = module._prepare_dirs(seed_dir)
    warnings = list(resources.warnings)
    ctx_kwargs = {
        "cfg": cfg,
        "seed_dir": seed_dir,
        "config_dir": dirs["config"],
        "trial_specs_dir": dirs["trial_specs"],
        "raw_dir": dirs["raw"],
        "metrics_dir": dirs["metrics"],
        "debug_dir": dirs["debug"],
        "device": resources.device,
        "dataset": resources.dataset,
        "class_index": resources.class_index,
        "net": resources.net,
        "encoder": resources.encoder,
        "warnings": warnings,
        "output_files": {},
        "completed_modules": {},
        "run_log": [f"{_now()} start shared_optimized {EXPERIMENT_IDS[fig_id]} seed={cfg.network_seed} smoke={cfg.smoke}"],
    }
    if fig_id == "fig6":
        ctx_kwargs["meta_dir"] = dirs["meta"]
    ctx = module.ExperimentContext(**ctx_kwargs)
    meta_dir = ctx.meta_dir if fig_id == "fig6" else seed_dir / "meta"
    run_info = build_run_info(
        experiment_name=EXPERIMENT_IDS[fig_id],
        output_dir=seed_dir,
        entry_script="src.experiments.paper_figures.optimized_issue_patch.run_shared_issue_patch",
        seed=int(cfg.network_seed),
        dataset=f"MNIST:{cfg.split}",
        command=command_text,
        model_path=str(resources.model_path),
        status="running",
    )
    write_run_info(meta_dir, run_info)
    return ctx, run_info


def _finalize_ctx(fig_id: str, ctx: Any, run_info: Mapping[str, Any], status: str) -> None:
    meta_dir = ctx.meta_dir if fig_id == "fig6" else ctx.seed_dir / "meta"
    finalize_run_info(meta_dir, run_info, status=status)


def _run_fig3_shared(cfg: Any, resources: SharedSeedResources, *, command_text: str) -> tuple[Any, Any]:
    seed_everything(int(cfg.network_seed))
    ctx, run_info = _fig_context("fig3", cfg, resources, command_text=command_text)
    try:
        fig3._write_config_files(ctx)
        seq_trials, _singleton_trials, _partial_trials = fig3.build_sequence_trial_specs(ctx)
        bank = fig3.run_multiitem_sequence_state_bank(ctx, seq_trials)
        if cfg.run_neutral_ping:
            fig3.run_neutral_ping_readout_distribution(ctx, bank)
        if cfg.run_weak_probe:
            fig3.run_sequence_weak_probe_real_rollout_from_state_bank(ctx, bank)
        fig3._write_summary(ctx)
        fig3._write_run_log(ctx)
        _finalize_ctx("fig3", ctx, run_info, "success")
        return ctx, bank
    except Exception:
        _finalize_ctx("fig3", ctx, run_info, "failed")
        raise


def _fig6_sequence_meta_from_trials(sequence_trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sequence_id, group in sequence_trials.groupby("sequence_id", sort=True):
        group = group.sort_values("stage_k")
        rows.append(
            {
                "network_seed": int(group["network_seed"].iloc[0]),
                "sequence_id": int(sequence_id),
                "seq_len": int(group["seq_len"].iloc[0]),
                "ordered_item_ids": str(group["ordered_item_ids"].iloc[0]),
                "ordered_item_labels": str(group["ordered_item_labels"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _write_fig6_derived_state_bank(ctx: Any, fig3_bank: Any) -> PeakAmplifiedReentryBank:
    sequence_trials = fig3_bank.sequence_trials.reindex(columns=fig6.SEQUENCE_TRIAL_COLUMNS).copy()
    fig6._save_csv(ctx, sequence_trials, ctx.trial_specs_dir / "sequence_trials.csv")
    ctx.n_sequences = int(sequence_trials["sequence_id"].nunique())
    ctx.completed_modules["sequence_trials"] = True

    seq_ids = sorted(sequence_trials["sequence_id"].unique())
    n_seq = len(seq_ids)
    n_units = 28 * 28
    max_seq_len = int(sequence_trials["seq_len"].max()) if not sequence_trials.empty else 0
    update_count = np.zeros((n_seq, n_units), dtype=np.float32)
    last_update_position = np.zeros((n_seq, n_units), dtype=np.int16)
    time_since_last_update = np.zeros((n_seq, n_units), dtype=np.int16)
    update_exposure_by_item = np.zeros((n_seq, max_seq_len, n_units), dtype=np.float32)
    item_activation_history = np.zeros_like(update_exposure_by_item)
    g_baseline = np.zeros((n_seq, n_units), dtype=np.float32)
    g_final = np.zeros((n_seq, n_units), dtype=np.float32)
    delta_support = np.zeros((n_seq, n_units), dtype=np.float32)
    peak_mask = np.zeros((n_seq, n_units), dtype=bool)
    nonpeak_mask = np.zeros((n_seq, n_units), dtype=bool)
    prior_updated_mask = np.zeros((n_seq, n_units), dtype=bool)
    boundaries: dict[int, Mapping[str, Mapping[str, Any]]] = {}
    manifest_rows: list[dict[str, Any]] = []

    for row_idx, sequence_id in enumerate(seq_ids):
        seq_id = int(sequence_id)
        group = sequence_trials[sequence_trials["sequence_id"].eq(seq_id)].sort_values("stage_k")
        seq_len = int(group["seq_len"].iloc[0])
        image_ids = [int(v) for v in group["item_image_id"].tolist()]
        masks = np.stack(
            [
                fig6._foreground_mask(ctx.dataset, image_id, ctx.cfg.foreground_threshold)
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
        landscape = fig3_bank.landscapes[seq_id]
        g_baseline[row_idx] = np.asarray(landscape["G_baseline"], dtype=np.float32).reshape(-1)
        g_final[row_idx] = np.asarray(landscape["G_final"], dtype=np.float32).reshape(-1)
        delta_support[row_idx] = g_final[row_idx] - g_baseline[row_idx]
        peaks = fig6._top_mask(delta_support[row_idx].reshape(28, 28), ctx.cfg.peak_q, positive=delta_support[row_idx].reshape(28, 28) > 0)
        peak_mask[row_idx] = peaks.reshape(-1)
        nonpeak_mask[row_idx] = fig6._matched_nonpeak_mask(peak_mask[row_idx], prior_updated_mask[row_idx], int(ctx.cfg.network_seed) + seq_id)
        boundaries[seq_id] = fig3_bank.boundaries[seq_id]["S_final"]
        for state_condition, stage_k, arrs in (
            ("S0", 0, {"G_baseline": g_baseline[row_idx]}),
            ("S_final", seq_len, {"G_final": g_final[row_idx], "delta_support": delta_support[row_idx]}),
        ):
            for key, arr in arrs.items():
                manifest_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": seq_id,
                        "seq_len": seq_len,
                        "state_condition": state_condition,
                        "stage_k": int(stage_k),
                        "layer": fig6.PRIMARY_LAYER,
                        "state_variable": fig6.STATE_VARIABLE if key != "delta_support" else "delta_support",
                        "shape": "28x28",
                        "storage_file": "final_support_maps.npz",
                        "storage_key": f"{key}_sequence_{seq_id}",
                        "captured_after": state_condition,
                        "sample_ms": int(ctx.cfg.sample_ms),
                        "delay_ms": int(ctx.cfg.delay_ms),
                    }
                )

    sequence_meta = _fig6_sequence_meta_from_trials(sequence_trials)
    fig6._save_csv(ctx, pd.DataFrame(manifest_rows, columns=fig6.STATE_BANK_MANIFEST_COLUMNS), ctx.raw_dir / "state_bank_manifest.csv")
    np.savez_compressed(
        ctx.raw_dir / "update_history_matrix.npz",
        update_count=update_count,
        last_update_position=last_update_position,
        time_since_last_update=time_since_last_update,
        update_exposure_by_item=update_exposure_by_item,
        item_activation_history=item_activation_history,
        unit_ids=np.arange(n_units, dtype=np.int32),
        sequence_ids=np.asarray(seq_ids, dtype=np.int32),
    )
    np.savez_compressed(
        ctx.raw_dir / "final_support_maps.npz",
        G_baseline=g_baseline,
        G_final=g_final,
        delta_support=delta_support,
        peak_mask=peak_mask.astype(np.uint8),
        nonpeak_mask=nonpeak_mask.astype(np.uint8),
        unit_ids=np.arange(n_units, dtype=np.int32),
        sequence_ids=np.asarray(seq_ids, dtype=np.int32),
    )
    ctx.output_files["state_bank_manifest"] = "data/raw/state_bank_manifest.csv"
    ctx.output_files["update_history_matrix"] = "data/raw/update_history_matrix.npz"
    ctx.output_files["final_support_maps"] = "data/raw/final_support_maps.npz"
    ctx.completed_modules["sequence_bank"] = True
    return PeakAmplifiedReentryBank(
        sequence_trials=sequence_trials.reset_index(drop=True),
        sequence_meta=sequence_meta,
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


@contextmanager
def _cached_fig6_probe_traces(bank: PeakAmplifiedReentryBank) -> Iterator[dict[str, int]]:
    from src.experiments.paper_figures.fig6.subexperiments import overlap_gated_stsp_recruitment as overlap_mod
    from src.experiments.paper_figures.fig6.subexperiments import real_probe_score_spike_deflection as probe_mod

    original_probe = probe_mod._run_real_probe_layer1_capture
    original_overlap = overlap_mod._run_real_probe_layer1_capture
    boundary_ids = {id(boundary): int(seq_id) for seq_id, boundary in bank.boundaries.items()}
    cache: dict[tuple[int, int, str], np.ndarray] = {}
    stats = {"hits": 0, "misses": 0}

    def cached(ctx: Any, probe_image_id: int, boundary: Mapping[str, Mapping[str, Any]] | None, *, probe_spikes: Any | None = None) -> np.ndarray:
        boundary_key = "baseline" if boundary is None else f"sequence_{boundary_ids.get(id(boundary), id(boundary))}"
        key = (id(ctx), int(probe_image_id), boundary_key)
        if key in cache:
            stats["hits"] += 1
            return cache[key].copy()
        stats["misses"] += 1
        trace = original_probe(ctx, probe_image_id, boundary, probe_spikes=probe_spikes)
        cache[key] = np.asarray(trace, dtype=np.float32).copy()
        return cache[key].copy()

    probe_mod._run_real_probe_layer1_capture = cached
    overlap_mod._run_real_probe_layer1_capture = cached
    try:
        yield stats
    finally:
        probe_mod._run_real_probe_layer1_capture = original_probe
        overlap_mod._run_real_probe_layer1_capture = original_overlap


def _run_fig6_shared(cfg: Any, resources: SharedSeedResources, *, command_text: str, fig3_bank: Any | None) -> tuple[Any, PeakAmplifiedReentryBank, dict[str, int]]:
    ctx, run_info = _fig_context("fig6", cfg, resources, command_text=command_text)
    trace_stats = {"hits": 0, "misses": 0}
    try:
        fig6._write_config_files(ctx)
        if fig3_bank is not None and bool(cfg.run_sequence_bank):
            bank = _write_fig6_derived_state_bank(ctx, fig3_bank)
        else:
            sequence_trials = fig6.build_sequence_trials(ctx)
            bank = fig6.run_sequence_bank(ctx, sequence_trials)
        if cfg.run_field_ping_readout:
            fig6.compute_field_ping_readout(ctx, bank)
        if cfg.run_global_ping_score_spike_prediction or cfg.run_ping_score_spike_prediction:
            fig6.compute_global_ping_score_spike_prediction(ctx, bank)
        with _cached_fig6_probe_traces(bank) as stats:
            if cfg.run_real_probe_score_spike_deflection:
                fig6.compute_real_probe_score_spike_deflection(ctx, bank)
            if cfg.run_overlap_gated_stsp_recruitment:
                fig6.compute_overlap_gated_stsp_recruitment(ctx, bank)
            trace_stats.update(stats)
        if cfg.run_high_stsp_overlap_ablation:
            fig6.compute_high_stsp_overlap_ablation(ctx, bank)
        if cfg.run_supplement:
            fig6.compute_supplement_outputs(ctx, bank)
        if cfg.run_score_shuffle_null:
            fig6.compute_score_shuffle_null_extension(ctx, bank)
        if cfg.run_overlap_threshold_sensitivity:
            fig6.compute_overlap_threshold_sensitivity_extension(ctx, bank)
        if cfg.run_legacy_supplement:
            fig6.compute_legacy_supplement_outputs(ctx, bank)
        fig6.write_fig6_supplement_aliases(ctx)
        fig6.write_global_mechanism_metadata(ctx)
        fig6._flush_score_audits(ctx)
        fig6._write_summary(ctx)
        fig6._write_run_log(ctx)
        _finalize_ctx("fig6", ctx, run_info, "success")
        return ctx, bank, trace_stats
    except Exception:
        _finalize_ctx("fig6", ctx, run_info, "failed")
        raise


def _run_fig4_shared(cfg: Any, resources: SharedSeedResources, *, command_text: str) -> Any:
    seed_everything(int(cfg.network_seed))
    ctx, run_info = _fig_context("fig4", cfg, resources, command_text=command_text)
    try:
        fig4._write_config_files(ctx)
        pair_trials, _candidate_pool, perturbation_masks, mask_bank = fig4.build_pair_trials(ctx)
        similarity_bank = None
        overlap_bank = None
        if cfg.run_similarity_entry or cfg.run_overlap_accuracy_identification:
            similarity_bank = fig4.run_similarity_bias_compatible_trials(ctx, pair_trials)
        if cfg.run_rollouts or cfg.run_overlap_localization or cfg.run_decision_spike_displacement or cfg.run_overlap_perturbation or cfg.run_supplement:
            overlap_bank = fig4.run_overlap_perturbation_compatible_rollouts(ctx, pair_trials, perturbation_masks, mask_bank)
        if cfg.run_overlap_localization and overlap_bank is not None:
            fig4.compute_overlap_localization_metrics(ctx, overlap_bank)
        if cfg.run_overlap_accuracy_identification and similarity_bank is not None:
            fig4.compute_overlap_accuracy_identification(ctx, similarity_bank)
        if cfg.run_decision_spike_displacement and overlap_bank is not None:
            fig4.compute_probe_l3_trace_dpi_metrics(ctx, overlap_bank)
        if cfg.run_decision_deflection or ((cfg.run_overlap_perturbation or cfg.run_supplement) and overlap_bank is not None):
            if cfg.run_decision_deflection:
                fig4.compute_l3_accumulator_region_replay_metrics(ctx, pair_trials)
            if (cfg.run_overlap_perturbation or cfg.run_supplement) and overlap_bank is not None:
                fig4.compute_decision_deflection_metrics(ctx, overlap_bank)
        if cfg.run_overlap_perturbation:
            if overlap_bank is not None:
                fig4.compute_overlap_preserving_perturbation_metrics(ctx, overlap_bank)
            fig4.compute_l1_stsp_overlap_perturbation_outputs(ctx, pair_trials, mask_bank)
        if cfg.run_supplement:
            if overlap_bank is not None:
                fig4.compute_supplement_outputs(ctx, overlap_bank)
        fig4.write_fig4_panel_aliases_and_supplement_aliases(ctx)
        fig4._write_summary(ctx)
        fig4._write_run_log(ctx)
        _finalize_ctx("fig4", ctx, run_info, "success")
        return ctx
    except Exception:
        _finalize_ctx("fig4", ctx, run_info, "failed")
        raise


def _run_optimized_seed(
    *,
    figs: Sequence[str],
    output_base: Path,
    checkpoint: NetworkCheckpoint,
    dataset_root: Path,
    args: argparse.Namespace,
) -> list[RunRecord]:
    seed = int(checkpoint.seed)
    for fig_id in figs:
        _safe_clean_seed_dir(output_base, fig_id, seed, dry_run=bool(args.dry_run))
    configs: dict[str, Any] = {}
    command_texts: dict[str, str] = {}
    for fig_id in figs:
        argv = _base_argv(fig_id=fig_id, output_base=output_base, checkpoint=checkpoint, dataset_root=dataset_root, args=args)
        configs[fig_id] = _config_from_argv(fig_id, argv)
        command_texts[fig_id] = f"in-process shared optimized equivalent of: {subprocess.list2cmdline(_command_for_fig(Path(sys.executable), fig_id, argv))}"
    records: list[RunRecord] = []
    if bool(args.dry_run):
        for fig_id in figs:
            records.append(
                RunRecord(
                    variant="optimized_shared",
                    fig_id=fig_id,
                    seed=seed,
                    command=command_texts[fig_id],
                    status="dry_run",
                    returncode=None,
                    elapsed_seconds=0.0,
                    output_root=_experiment_root(output_base, fig_id),
                    optimization_notes="dry_run",
                )
            )
        return records
    resources = _load_shared_resources(
        checkpoint=checkpoint,
        configs=configs,
        dataset_root=dataset_root,
        device_arg=str(args.device),
        split=str(args.split),
        smoke=bool(args.smoke),
    )
    fig3_bank = None
    for fig_id in figs:
        started = time.perf_counter()
        status = "success"
        notes = "shared_model_dataset_load"
        try:
            if fig_id == "fig3":
                _ctx3, fig3_bank = _run_fig3_shared(configs[fig_id], resources, command_text=command_texts[fig_id])
                notes = "shared_model_dataset_load;sequence_bank_source=fig3"
            elif fig_id == "fig6":
                _ctx6, _bank6, trace_stats = _run_fig6_shared(configs[fig_id], resources, command_text=command_texts[fig_id], fig3_bank=fig3_bank)
                source = "fig3_derived" if fig3_bank is not None else "fig6_native"
                notes = f"shared_model_dataset_load;sequence_bank_source={source};probe_trace_cache_hits={trace_stats.get('hits', 0)};probe_trace_cache_misses={trace_stats.get('misses', 0)}"
            elif fig_id == "fig4":
                _run_fig4_shared(configs[fig_id], resources, command_text=command_texts[fig_id])
                notes = "shared_model_dataset_load;fig4_condition_batch=false"
            else:  # pragma: no cover
                raise ValueError(fig_id)
            elapsed = time.perf_counter() - started
            records.append(
                RunRecord(
                    variant="optimized_shared",
                    fig_id=fig_id,
                    seed=seed,
                    command=command_texts[fig_id],
                    status=status,
                    returncode=0,
                    elapsed_seconds=float(elapsed),
                    output_root=_experiment_root(output_base, fig_id),
                    optimization_notes=notes,
                )
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            records.append(
                RunRecord(
                    variant="optimized_shared",
                    fig_id=fig_id,
                    seed=seed,
                    command=command_texts[fig_id],
                    status="failed",
                    returncode=1,
                    elapsed_seconds=float(elapsed),
                    output_root=_experiment_root(output_base, fig_id),
                    stderr_tail=str(exc)[-4000:],
                    optimization_notes=notes,
                )
            )
            break
    return records


def _copy_rel(src_seed: Path, dst_seed: Path, rel: str, *, dry_run: bool) -> dict[str, Any]:
    src = src_seed / rel
    dst = dst_seed / rel
    if dry_run:
        return {"source": str(src), "destination": str(dst), "rel": rel, "status": "dry_run"}
    if not src.exists():
        return {"source": str(src), "destination": str(dst), "rel": rel, "status": "missing_source"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source": str(src), "destination": str(dst), "rel": rel, "status": "copied"}


def _row_selector_column(frame: pd.DataFrame) -> str | None:
    for col in ("outcome", "metric"):
        if col in frame.columns:
            return col
    return None


def _merge_acc_drop_rows(src_path: Path, dst_path: Path, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"source": str(src_path), "destination": str(dst_path), "status": "dry_run"}
    if not src_path.is_file():
        return {"source": str(src_path), "destination": str(dst_path), "status": "missing_source"}
    new_df = pd.read_csv(src_path)
    old_df = pd.read_csv(dst_path) if dst_path.is_file() else pd.DataFrame()
    selector = _row_selector_column(new_df) or _row_selector_column(old_df)
    if selector is None:
        return {"source": str(src_path), "destination": str(dst_path), "status": "failed", "reason": "no_metric_or_outcome_column"}

    def is_acc_drop(frame: pd.DataFrame) -> pd.Series:
        if selector not in frame.columns:
            return pd.Series([False] * len(frame), index=frame.index)
        return frame[selector].astype(str).eq("acc_drop")

    new_acc = new_df[is_acc_drop(new_df)].copy()
    old_keep = old_df[~is_acc_drop(old_df)].copy() if not old_df.empty else pd.DataFrame()
    ordered_cols = list(dict.fromkeys([*old_keep.columns.tolist(), *new_acc.columns.tolist()]))
    merged = pd.concat(
        [old_keep.reindex(columns=ordered_cols), new_acc.reindex(columns=ordered_cols)],
        ignore_index=True,
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(dst_path, index=False, encoding="utf-8")
    return {
        "source": str(src_path),
        "destination": str(dst_path),
        "status": "merged_acc_drop_rows",
        "selector_column": selector,
        "old_rows_kept": int(len(old_keep)),
        "new_acc_drop_rows": int(len(new_acc)),
        "merged_rows": int(len(merged)),
    }


def _overlay_seed(staging_root: Path, final_root: Path, fig_id: str, seed: int, *, dry_run: bool) -> list[dict[str, Any]]:
    src_seed = _seed_dir(staging_root, fig_id, seed)
    dst_seed = _seed_dir(final_root, fig_id, seed)
    records: list[dict[str, Any]] = []
    for rel in (*COMMON_PATCH_METADATA, *OVERLAY_FILES[fig_id]):
        record = _copy_rel(src_seed, dst_seed, rel, dry_run=dry_run)
        record.update({"fig_id": fig_id, "seed": int(seed), "mode": "copy"})
        records.append(record)
    for rel in ROW_MERGE_FILES.get(fig_id, ()):
        record = _merge_acc_drop_rows(src_seed / rel, dst_seed / rel, dry_run=dry_run)
        record.update({"fig_id": fig_id, "seed": int(seed), "rel": rel, "mode": "row_merge_acc_drop"})
        records.append(record)
    if not dry_run:
        _write_json(
            dst_seed / "shared_optimization_seed_manifest.json",
            {
                "patch_version": PATCH_VERSION,
                "fig_id": fig_id,
                "seed": int(seed),
                "created_at": _now(),
                "overlay_policy": "clone_first_allowlist_overlay",
                "overlay_records": records,
            },
        )
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sort_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keys = [col for col in KEY_COLUMNS if col in frame.columns]
    if not keys:
        out = frame.copy()
        out.insert(0, "__row_index__", np.arange(len(out), dtype=np.int64))
        return out, ["__row_index__"]
    return frame.sort_values(keys, kind="mergesort").reset_index(drop=True), keys


def _discrete_column(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in DISCRETE_COLUMN_TOKENS)


def _csv_diff(left_path: Path, right_path: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)
    left, keys = _sort_frame(left)
    right, _ = _sort_frame(right)
    left_cols = list(left.columns)
    right_cols = list(right.columns)
    schema_changed = left_cols != right_cols
    row_count_changed = len(left) != len(right)
    common_cols = [col for col in left_cols if col in right.columns]
    changed_cells = 0
    max_abs = 0.0
    sum_abs = 0.0
    n_abs = 0
    exact_changed = 0
    n = min(len(left), len(right))
    for col in common_cols:
        lvals = left[col].iloc[:n]
        rvals = right[col].iloc[:n]
        if col in keys or _discrete_column(col):
            mismatch = lvals.astype(str).fillna("<NA>") != rvals.astype(str).fillna("<NA>")
            exact_changed += int(mismatch.sum())
            changed_cells += int(mismatch.sum())
            continue
        lnum = pd.to_numeric(lvals, errors="coerce")
        rnum = pd.to_numeric(rvals, errors="coerce")
        numeric_mask = lnum.notna() | rnum.notna()
        if bool(numeric_mask.any()):
            la = lnum[numeric_mask].to_numpy(dtype=float)
            ra = rnum[numeric_mask].to_numpy(dtype=float)
            close = np.isclose(la, ra, atol=float(atol), rtol=float(rtol), equal_nan=True)
            diff = np.abs(la - ra)
            finite = diff[np.isfinite(diff)]
            if finite.size:
                max_abs = max(max_abs, float(finite.max()))
                sum_abs += float(finite.sum())
                n_abs += int(finite.size)
            changed_cells += int((~close).sum())
        string_mask = ~numeric_mask
        if bool(string_mask.any()):
            mismatch = lvals[string_mask].astype(str).fillna("<NA>") != rvals[string_mask].astype(str).fillna("<NA>")
            exact_changed += int(mismatch.sum())
            changed_cells += int(mismatch.sum())
    if schema_changed:
        changed_cells += abs(len(left_cols) - len(right_cols))
    if row_count_changed:
        changed_cells += abs(len(left) - len(right))
    status = "unchanged" if changed_cells == 0 and not schema_changed and not row_count_changed else "changed"
    if status == "changed" and exact_changed == 0 and not schema_changed and not row_count_changed and max_abs <= max(float(atol), float(rtol)):
        status = "within_tolerance"
    return {
        "status": status,
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "changed_cells": int(changed_cells),
        "max_abs_diff": float(max_abs),
        "mean_abs_diff": float(sum_abs / n_abs) if n_abs else 0.0,
        "schema_changed": bool(schema_changed),
        "row_count_changed": bool(row_count_changed),
        "key_columns": ";".join(keys),
        "numeric_columns": ";".join(
            col for col in common_cols if pd.api.types.is_numeric_dtype(left[col]) or pd.api.types.is_numeric_dtype(right[col])
        ),
        "discrete_changes": int(exact_changed),
    }


def _npz_diff(left_path: Path, right_path: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    changed = 0
    max_abs = 0.0
    sum_abs = 0.0
    n_abs = 0
    with np.load(left_path, allow_pickle=False) as left, np.load(right_path, allow_pickle=False) as right:
        left_keys = sorted(left.files)
        right_keys = sorted(right.files)
        if left_keys != right_keys:
            return {
                "status": "changed",
                "changed_cells": abs(len(left_keys) - len(right_keys)),
                "schema_changed": True,
                "row_count_changed": False,
                "max_abs_diff": "",
                "mean_abs_diff": "",
                "key_columns": "",
                "numeric_columns": "",
            }
        for key in left_keys:
            la = left[key]
            ra = right[key]
            if la.shape != ra.shape or la.dtype.kind != ra.dtype.kind:
                changed += 1
                continue
            if la.dtype.kind in {"f", "c"}:
                close = np.isclose(la, ra, atol=float(atol), rtol=float(rtol), equal_nan=True)
                diff = np.abs(la.astype(float) - ra.astype(float))
                finite = diff[np.isfinite(diff)]
                if finite.size:
                    max_abs = max(max_abs, float(finite.max()))
                    sum_abs += float(finite.sum())
                    n_abs += int(finite.size)
                changed += int((~close).sum())
            else:
                changed += int(np.count_nonzero(la != ra))
    status = "unchanged" if changed == 0 else "changed"
    if status == "changed" and max_abs <= max(float(atol), float(rtol)):
        status = "within_tolerance"
    return {
        "status": status,
        "changed_cells": int(changed),
        "max_abs_diff": float(max_abs),
        "mean_abs_diff": float(sum_abs / n_abs) if n_abs else 0.0,
        "schema_changed": False,
        "row_count_changed": False,
        "key_columns": "",
        "numeric_columns": "",
    }


def _file_diff(left_path: Path, right_path: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    if not left_path.is_file() and not right_path.is_file():
        return {"status": "missing_both"}
    if not left_path.is_file():
        return {"status": "missing_left"}
    if not right_path.is_file():
        return {"status": "missing_right"}
    suffix = left_path.suffix.lower()
    if suffix == ".csv":
        return _csv_diff(left_path, right_path, atol=atol, rtol=rtol)
    if suffix == ".npz":
        return _npz_diff(left_path, right_path, atol=atol, rtol=rtol)
    same_hash = _sha256(left_path) == _sha256(right_path)
    return {
        "status": "unchanged" if same_hash else "changed",
        "left_hash": _sha256(left_path),
        "right_hash": _sha256(right_path),
        "changed_cells": "" if same_hash else 1,
        "max_abs_diff": "",
        "mean_abs_diff": "",
        "schema_changed": False,
        "row_count_changed": False,
        "key_columns": "",
        "numeric_columns": "",
    }


def _compare_allowed_files(
    reference_root: Path,
    optimized_root: Path,
    figs: Sequence[str],
    checkpoints: Sequence[NetworkCheckpoint],
    *,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fig_id in figs:
        for checkpoint in checkpoints:
            seed = int(checkpoint.seed)
            ref_seed = _seed_dir(reference_root, fig_id, seed)
            opt_seed = _seed_dir(optimized_root, fig_id, seed)
            for rel in COMPARE_FILES[fig_id]:
                left = ref_seed / rel
                right = opt_seed / rel
                diff = _file_diff(left, right, atol=float(atol), rtol=float(rtol))
                rows.append(
                    {
                        "fig_id": fig_id,
                        "seed": seed,
                        "rel": rel,
                        "reference_path": str(left),
                        "optimized_path": str(right),
                        **diff,
                    }
                )
    return rows


def _equivalence_status(rows: Sequence[Mapping[str, Any]]) -> str:
    bad = {"changed", "missing_left", "missing_right"}
    if any(str(row.get("status")) in bad for row in rows):
        return "failed"
    allowed = {"unchanged", "within_tolerance", "missing_both", "metadata_only"}
    if any(str(row.get("status")) not in allowed for row in rows):
        return "failed"
    return "success"


def _validate_seed_dir(runtime_python: Path, seed_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    command = [str(runtime_python), "scripts/validate_results_layout.py", "--input-dir", str(seed_dir)]
    result = _run_subprocess(command, cwd=DEFAULT_PROJECT_DEFAULTS.paths.repo_root, dry_run=dry_run)
    result.update({"seed_dir": str(seed_dir)})
    return result


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _numeric_nonzero_count(path: Path) -> int:
    count = 0
    for row in _csv_rows(path):
        for value in row.values():
            try:
                number = float(str(value).strip())
            except (TypeError, ValueError):
                continue
            if number != 0.0 and number == number:
                count += 1
    return count


def _audit_nonzero_csv(seed_dir: Path, rel_paths: Sequence[str], fig_id: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for rel in rel_paths:
        path = seed_dir / rel
        if not path.is_file():
            results.append({"fig_id": fig_id, "seed_dir": str(seed_dir), "path": rel, "status": "failed", "reason": "missing"})
            continue
        rows = _csv_rows(path)
        nonzero_count = _numeric_nonzero_count(path)
        status = "success" if rows and nonzero_count > 0 else "failed"
        reason = "" if status == "success" else "empty_or_all_numeric_zero"
        results.append({"fig_id": fig_id, "seed_dir": str(seed_dir), "path": rel, "status": status, "n_rows": len(rows), "numeric_nonzero_count": nonzero_count, "reason": reason})
    return results


def _fig4_pair_audit(seed_dir: Path) -> dict[str, Any]:
    path = seed_dir / "data" / "trial_specs" / "pair_trials.csv"
    if not path.is_file():
        return {"fig_id": "fig4", "seed_dir": str(seed_dir), "path": "data/trial_specs/pair_trials.csv", "status": "failed", "same_label_rows": "", "n_rows": "", "reason": "missing"}
    df = pd.read_csv(path)
    if not {"sample_label", "probe_label"}.issubset(df.columns):
        return {"fig_id": "fig4", "seed_dir": str(seed_dir), "path": "data/trial_specs/pair_trials.csv", "status": "failed", "same_label_rows": "", "n_rows": int(len(df)), "reason": "missing_label_columns"}
    same = int((df["sample_label"].astype(str) == df["probe_label"].astype(str)).sum())
    return {
        "fig_id": "fig4",
        "seed_dir": str(seed_dir),
        "path": "data/trial_specs/pair_trials.csv",
        "status": "success" if same == 0 else "failed",
        "same_label_rows": same,
        "n_rows": int(len(df)),
        "reason": "",
    }


def _run_audits(final_root: Path, figs: Sequence[str], checkpoints: Sequence[NetworkCheckpoint]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fig_id in figs:
        for checkpoint in checkpoints:
            seed_dir = _seed_dir(final_root, fig_id, int(checkpoint.seed))
            if fig_id == "fig3":
                rows.extend(
                    _audit_nonzero_csv(
                        seed_dir,
                        [
                            "data/raw/panel_d_neutral_ping_trial_readout.csv",
                            "data/raw/panel_e_weak_probe_trial_readout.csv",
                        ],
                        fig_id,
                    )
                )
            elif fig_id == "fig4":
                rows.append(_fig4_pair_audit(seed_dir))
            elif fig_id == "fig6":
                rows.extend(
                    _audit_nonzero_csv(
                        seed_dir,
                        [
                            "data/metrics/panel_b_region_ping_readout_bias.csv",
                            "data/metrics/panel_c_global_ping_score_spike_prediction.csv",
                            "data/metrics/panel_d_real_probe_score_spike_deflection.csv",
                            "data/metrics/panel_e_overlap_gated_stsp_recruitment.csv",
                            "data/metrics/panel_a_high_stsp_overlap_ablation.csv",
                        ],
                        fig_id,
                    )
                )
    return rows


def _run_build_checks(runtime_python: Path, output_root: Path, figs: Sequence[str], *, dry_run: bool, build_preview: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repo_root = DEFAULT_PROJECT_DEFAULTS.paths.repo_root
    for fig_id in figs:
        exp_root = _experiment_root(output_root, fig_id)
        for plot_fig_id in PLOT_FIG_IDS[fig_id]:
            command = [
                str(runtime_python),
                "-m",
                "src.plotting.paper_fig.build",
                "--fig",
                plot_fig_id,
                "--experiment-root",
                str(exp_root),
            ]
            if not build_preview:
                command.append("--check-only")
            result = _run_subprocess(command, cwd=repo_root, dry_run=dry_run)
            result.update({"fig_id": fig_id, "plot_fig_id": plot_fig_id, "experiment_root": str(exp_root)})
            rows.append(result)
    return rows


def _run_records(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    return [
        {
            "variant": item.variant,
            "fig_id": item.fig_id,
            "seed": int(item.seed),
            "status": item.status,
            "returncode": item.returncode,
            "elapsed_seconds": item.elapsed_seconds,
            "output_root": str(item.output_root),
            "command": item.command,
            "stdout_tail": item.stdout_tail,
            "stderr_tail": item.stderr_tail,
            "optimization_notes": item.optimization_notes,
        }
        for item in records
    ]


def _write_reports(
    *,
    report_dir: Path,
    output_root: Path,
    source_root: Path,
    reference_root: Path,
    optimized_root: Path,
    runtime_python: Path,
    args: argparse.Namespace,
    figs: Sequence[str],
    checkpoints: Sequence[NetworkCheckpoint],
    clone_records: Sequence[Mapping[str, Any]],
    run_records: Sequence[RunRecord],
    equivalence_rows: Sequence[Mapping[str, Any]],
    overlay_records: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    build_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    run_rows = _run_records(run_records)
    _write_csv(
        report_dir / "run_manifest.csv",
        run_rows,
        [
            "variant",
            "fig_id",
            "seed",
            "status",
            "returncode",
            "elapsed_seconds",
            "output_root",
            "command",
            "stdout_tail",
            "stderr_tail",
            "optimization_notes",
        ],
    )
    _write_csv(
        report_dir / "equivalence_report.csv",
        equivalence_rows,
        [
            "fig_id",
            "seed",
            "rel",
            "status",
            "left_rows",
            "right_rows",
            "changed_cells",
            "max_abs_diff",
            "mean_abs_diff",
            "schema_changed",
            "row_count_changed",
            "discrete_changes",
            "key_columns",
            "numeric_columns",
            "reference_path",
            "optimized_path",
        ],
    )
    _write_csv(report_dir / "overlay_manifest.csv", overlay_records, ["fig_id", "seed", "mode", "rel", "status", "source", "destination", "old_rows_kept", "new_acc_drop_rows", "merged_rows", "reason"])
    _write_csv(report_dir / "clone_manifest.csv", clone_records, ["fig_id", "seed", "source", "source_layout", "destination", "status"])
    _write_csv(report_dir / "validation_manifest.csv", validation_rows, ["fig_id", "seed", "seed_dir", "status", "returncode", "command", "stdout_tail", "stderr_tail"])
    _write_csv(report_dir / "audit_manifest.csv", audit_rows, ["fig_id", "seed_dir", "path", "status", "same_label_rows", "n_rows", "numeric_nonzero_count", "reason"])
    _write_csv(report_dir / "build_manifest.csv", build_rows, ["fig_id", "plot_fig_id", "experiment_root", "status", "returncode", "command", "stdout_tail", "stderr_tail"])

    status = "success"
    allowed_run_status = {"success", "dry_run"}
    if any(row.get("status") not in {"copied", "dry_run"} for row in clone_records):
        status = "failed"
    if any(item.status not in allowed_run_status for item in run_records):
        status = "failed"
    if equivalence_rows and _equivalence_status(equivalence_rows) != "success":
        status = "failed"
    if any(row.get("status") in {"missing_source", "failed"} for row in overlay_records):
        status = "failed"
    if any(row.get("status") not in {"success", "dry_run"} for row in validation_rows):
        status = "failed"
    if any(row.get("status") == "failed" for row in audit_rows):
        status = "failed"
    if any(row.get("status") == "failed" for row in build_rows):
        status = "failed"

    summary = {
        "status": status,
        "patch_version": PATCH_VERSION,
        "created_at": _now(),
        "runtime_python": str(runtime_python),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "reference_root": str(reference_root),
        "optimized_staging_root": str(optimized_root),
        "figs": list(figs),
        "seeds": [int(item.seed) for item in checkpoints],
        "smoke": bool(args.smoke),
        "check_equivalence": bool(args.check_equivalence),
        "check_build": bool(args.check_build),
        "shared_sequence_lengths": str(args.shared_sequence_lengths),
        "experiment_batch_size": getattr(args, "experiment_batch_size", None),
        "fig3_batch_size": getattr(args, "fig3_batch_size", None),
        "fig4_batch_size": getattr(args, "fig4_batch_size", None),
        "fig6_batch_size": getattr(args, "fig6_batch_size", None),
        "fig4_l3_region_batch_size": getattr(args, "fig4_l3_region_batch_size", None),
        "benchmark_profile": str(args.benchmark_profile),
        "equivalence_tolerance": {"atol": float(args.equivalence_atol), "rtol": float(args.equivalence_rtol)},
        "equivalence_status": _equivalence_status(equivalence_rows) if equivalence_rows else "not_run",
        "clone_count": len(clone_records),
        "run_count": len(run_records),
        "overlay_count": len(overlay_records),
        "validation_count": len(validation_rows),
        "audit_count": len(audit_rows),
        "build_count": len(build_rows),
        "optimization_policy": {
            "shared_resource_load": "each seed loads checkpoint, dataset, and class index once",
            "fig3_fig6_sequence_bank": "Fig.6 optimized staging derives sequence_trials and STSP support maps from the Fig.3 sequence/state bank when Fig.3 is selected",
            "fig6_probe_trace_cache": "Fig.6 D/E share intact dynamic and baseline real-probe Layer 1 traces by (probe_image_id, sequence boundary)",
            "fig4": "Fig.4 reuses shared model/data resources but keeps DMS pair rollouts separate and leaves condition batching disabled",
            "overlay": "final bundle is cloned from source first, then allowlisted regenerated files are overlaid",
        },
        "fig4_s5c_policy": "replace acc_drop rows in supp_overlap_similarity_regression.csv and keep old non-acc_drop rows",
    }
    _write_json(report_dir / "summary.json", summary)
    manifest = {
        **summary,
        "clone_records": list(clone_records),
        "run_records": run_rows,
        "equivalence_rows": list(equivalence_rows),
        "overlay_records": list(overlay_records),
        "validation_rows": list(validation_rows),
        "audit_rows": list(audit_rows),
        "build_rows": list(build_rows),
    }
    _write_json(report_dir / "shared_optimization_manifest.json", manifest)
    _write_json(output_root / "shared_optimization_manifest.json", manifest)
    return summary


def run_shared_issue_patch(args: argparse.Namespace) -> int:
    repo_root = DEFAULT_PROJECT_DEFAULTS.paths.repo_root
    runtime_python = Path(args.python).resolve() if args.python else _resolve_runtime_python()
    output_root = _resolve_repo_path(args.output_root)
    source_root = _resolve_repo_path(args.source_root)
    dataset_root = _resolve_repo_path(args.dataset_root)
    figs = _parse_figs(args.figs)
    checkpoints = select_checkpoints(discover_checkpoints(args.model_path_glob), seeds=args.seeds, all_seeds=bool(args.all_seeds))
    reference_root = _variant_base(output_root, "reference_serial")
    optimized_root = _variant_base(output_root, "optimized_staging")
    report_dir = output_root / "shared_optimization_reports"

    clone_records: list[dict[str, Any]] = []
    run_records: list[RunRecord] = []
    overlay_records: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    build_rows: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        for fig_id in figs:
            clone_records.append(_clone_seed_bundle(source_root, output_root, fig_id, int(checkpoint.seed), dry_run=bool(args.dry_run)))

    if bool(args.check_equivalence):
        for checkpoint in checkpoints:
            for fig_id in figs:
                record = _run_reference_serial(
                    runtime_python=runtime_python,
                    fig_id=fig_id,
                    output_base=reference_root,
                    checkpoint=checkpoint,
                    dataset_root=dataset_root,
                    args=args,
                    repo_root=repo_root,
                )
                run_records.append(record)
                if record.status == "failed":
                    summary = _write_reports(
                        report_dir=report_dir,
                        output_root=output_root,
                        source_root=source_root,
                        reference_root=reference_root,
                        optimized_root=optimized_root,
                        runtime_python=runtime_python,
                        args=args,
                        figs=figs,
                        checkpoints=checkpoints,
                        clone_records=clone_records,
                        run_records=run_records,
                        equivalence_rows=[],
                        overlay_records=[],
                        validation_rows=[],
                        audit_rows=[],
                        build_rows=[],
                    )
                    return 0 if summary["status"] == "success" else 1

    for checkpoint in checkpoints:
        seed_records = _run_optimized_seed(figs=figs, output_base=optimized_root, checkpoint=checkpoint, dataset_root=dataset_root, args=args)
        run_records.extend(seed_records)
        if any(record.status == "failed" for record in seed_records):
            summary = _write_reports(
                report_dir=report_dir,
                output_root=output_root,
                source_root=source_root,
                reference_root=reference_root,
                optimized_root=optimized_root,
                runtime_python=runtime_python,
                args=args,
                figs=figs,
                checkpoints=checkpoints,
                clone_records=clone_records,
                run_records=run_records,
                equivalence_rows=[],
                overlay_records=[],
                validation_rows=[],
                audit_rows=[],
                build_rows=[],
            )
            return 0 if summary["status"] == "success" else 1

    equivalence_rows: list[dict[str, Any]] = []
    if bool(args.check_equivalence):
        equivalence_rows = _compare_allowed_files(
            reference_root,
            optimized_root,
            figs,
            checkpoints,
            atol=float(args.equivalence_atol),
            rtol=float(args.equivalence_rtol),
        )
        if _equivalence_status(equivalence_rows) != "success" and not bool(args.overlay_on_equivalence_failure):
            summary = _write_reports(
                report_dir=report_dir,
                output_root=output_root,
                source_root=source_root,
                reference_root=reference_root,
                optimized_root=optimized_root,
                runtime_python=runtime_python,
                args=args,
                figs=figs,
                checkpoints=checkpoints,
                clone_records=clone_records,
                run_records=run_records,
                equivalence_rows=equivalence_rows,
                overlay_records=[],
                validation_rows=[],
                audit_rows=[],
                build_rows=[],
            )
            return 0 if summary["status"] == "success" else 1

    for checkpoint in checkpoints:
        seed = int(checkpoint.seed)
        for fig_id in figs:
            overlay_records.extend(_overlay_seed(optimized_root, output_root, fig_id, seed, dry_run=bool(args.dry_run)))
            validation = _validate_seed_dir(runtime_python, _seed_dir(output_root, fig_id, seed), dry_run=bool(args.dry_run))
            validation.update({"fig_id": fig_id, "seed": seed})
            validation_rows.append(validation)

    audit_rows = _run_audits(output_root, figs, checkpoints) if not bool(args.dry_run) else []

    if bool(args.check_build):
        build_rows = _run_build_checks(runtime_python, output_root, figs, dry_run=bool(args.dry_run), build_preview=bool(args.build_preview))

    summary = _write_reports(
        report_dir=report_dir,
        output_root=output_root,
        source_root=source_root,
        reference_root=reference_root,
        optimized_root=optimized_root,
        runtime_python=runtime_python,
        args=args,
        figs=figs,
        checkpoints=checkpoints,
        clone_records=clone_records,
        run_records=run_records,
        equivalence_rows=equivalence_rows,
        overlay_records=overlay_records,
        validation_rows=validation_rows,
        audit_rows=audit_rows,
        build_rows=build_rows,
    )
    return 0 if summary["status"] == "success" else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared optimized Fig.3/Fig.4/Fig.6 paper-figure issue patch.")
    parser.add_argument("--figs", default=DEFAULT_FIGS, help="Comma-separated subset of fig3,fig4,fig6.")
    parser.add_argument("--seeds", default="1000", help="Seed list/range, e.g. 1000 or 1000-1019.")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--model-path-glob", default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=None, help="Runtime Python. Defaults to the repo runtime resolver.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark-profile", default=BENCHMARK_PROFILE_NONE, choices=tuple(BENCHMARK_PROFILE_ARGS_BY_FIG))
    parser.add_argument("--shared-sequence-lengths", default="3,5,7,10")
    parser.add_argument("--check-equivalence", action="store_true")
    parser.add_argument("--overlay-on-equivalence-failure", action="store_true")
    parser.add_argument("--equivalence-atol", type=float, default=1e-5)
    parser.add_argument("--equivalence-rtol", type=float, default=1e-5)
    parser.add_argument("--experiment-batch-size", type=int, default=None, help="Forward --batch-size to Fig.3/Fig.4/Fig.6 unless a figure-specific override is set.")
    parser.add_argument("--fig3-batch-size", type=int, default=None, help="Forward --batch-size to Fig.3 only.")
    parser.add_argument("--fig4-batch-size", type=int, default=None, help="Forward --batch-size to Fig.4 only.")
    parser.add_argument("--fig6-batch-size", type=int, default=None, help="Forward --batch-size to Fig.6 only.")
    parser.add_argument("--fig4-l3-region-batch-size", type=int, default=None, help="Forward --l3-region-batch-size to Fig.4.")
    parser.add_argument("--fig6-safe-gpu-batching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-build", action="store_true")
    parser.add_argument("--build-preview", action="store_true", help="Build figures instead of check-only when --check-build is set.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_shared_issue_patch(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
