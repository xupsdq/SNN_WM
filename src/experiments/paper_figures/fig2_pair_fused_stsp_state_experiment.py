from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.core.network import SDNN_Network
from src.data.encoding import DoGSpikeEncoder
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import reset_all_state_restore_selected_stsp_in_place, snapshot_boundary_state
from src.experiments.common.ping_common import LAYER_KEYS, prepare_network_state, snapshot_ux_state
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.plotting.common.io import apply_publication_style, save_figure_all_formats

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _progress(iterable, *, total=None, desc: str = "", enabled: bool = True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


FIGURE_ID = "fig2_pair_fused_stsp_state"
NUM_CLASSES = 10
STATE_CONDITIONS = ("S0", "S_A", "S_B", "S_AB")
STATE_VARIABLES = ("g", "u", "x", "ux_concat")
MIXTURE_MODELS = ("A_only", "B_only", "mean_AB", "sum_AB", "unconstrained_AB", "convex_AB")
SINGLE_NETWORK_MODE = "single_network"
RESIDUAL_TEMPLATE_DEFINITION = "residual_true=y_AB-yhat_unconstrained; true_template=y_AB-0.5*(x_A+x_B); shuffled_template=y_AB-0.5*(x_A+x_B_j), j!=i"


@dataclass(frozen=True)
class Fig2Config:
    model_path: str
    dataset_root: str
    output_root: str
    network_seed: int
    device: str = "auto"
    split: str = "test"
    dt: float = 0.001
    sample_ms: int = 200
    delay1_ms: int = 200
    second_item_ms: int = 200
    delay2_ms: int = 400
    ping_ms: int = 30
    ping_amp: float = 1.0
    ping_repeats: int = 1
    ping_mode: str = "constant_drive"
    ping_noise: float = 0.0
    ping_amp_sweep: tuple[float, ...] = (0.5, 1.0, 1.5)
    ping_ms_sweep: tuple[int, ...] = (10, 30, 60)
    weak_probe_ms: int = 30
    weak_probe_keep_probs: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    weak_probe_repeats: int = 5
    weak_probe_mask_space: str = "encoded_spikes"
    weak_probe_use_same_mask_across_states: bool = True
    weak_probe_scale: float = 0.35
    weak_probe_noise: float = 0.0
    weak_probe_metric_mode: str = "fig4_compat"
    foreground_threshold: float = 0.0
    num_pairs: int = 200
    batch_size: int = 16
    n_shuffle: int = 50
    delay_layer_grid: tuple[int, ...] = (200, 400, 800)
    linear_mixture_cv_folds: int = 5
    primary_layer: str = "layer3"
    primary_state_variable: str = "g"
    run_state_bank: bool = False
    run_morphology: bool = False
    run_linear_mixture: bool = False
    run_neutral_ping: bool = False
    run_partial_cue: bool = False
    run_supplement: bool = False
    run_ping_sweep: bool = False
    run_completion_delay_sweep: bool = False
    completion_delay_sweep_ms: tuple[int, ...] = (200, 400, 800, 1200)
    completion_delay_keep_prob: float = 0.2
    completion_delay_repeats: int = 5
    save_debug_figures: bool = False
    save_spike_cache: bool = False
    save_all_layer_state_bank: bool = False
    save_functional_traces: bool = False
    save_proxy_functional_debug: bool = False
    show_progress: bool = True
    use_encode_cache: bool = True
    enable_partial_cue_batch: bool = False
    smoke: bool = False

    @property
    def sample_steps(self) -> int:
        return _ms_to_steps(self.sample_ms, self.dt)

    @property
    def delay1_steps(self) -> int:
        return _ms_to_steps(self.delay1_ms, self.dt)

    @property
    def second_item_steps(self) -> int:
        return _ms_to_steps(self.second_item_ms, self.dt)

    @property
    def delay2_steps(self) -> int:
        return _ms_to_steps(self.delay2_ms, self.dt)

    @property
    def ping_steps(self) -> int:
        return _ms_to_steps(self.ping_ms, self.dt)

    @property
    def weak_probe_steps(self) -> int:
        return _ms_to_steps(self.weak_probe_ms, self.dt)


@dataclass
class ExperimentContext:
    cfg: Fig2Config
    seed_dir: Path
    config_dir: Path
    trial_specs_dir: Path
    raw_dir: Path
    metrics_dir: Path
    debug_dir: Path
    device: torch.device
    dataset: Any
    class_index: dict[int, list[int]]
    net: Any
    encoder: Any
    warnings: list[str]
    output_files: dict[str, str]
    completed_modules: dict[str, bool]
    run_log: list[str]
    n_pairs: int = 0


@dataclass
class PairEpisodeStateBank:
    pair_trials: pd.DataFrame
    arrays: dict[str, dict[str, dict[str, np.ndarray]]]
    boundary_states: dict[str, Mapping[str, Mapping[str, torch.Tensor]]]
    layer_input_shapes: dict[str, tuple[int, ...]]
    restore_mode: str
    episode_end_step: int

    def get(self, condition: str, layer: str, variable: str) -> np.ndarray:
        return self.arrays[condition][layer][variable]


@dataclass
class FunctionalReadout:
    prediction: np.ndarray
    first_fire_time_ms: np.ndarray
    silent: np.ndarray
    readout_margin_A: np.ndarray | None = None
    readout_margin_B: np.ndarray | None = None
    trace: dict[str, np.ndarray] | None = None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    run(cfg)
    return 0


def run(cfg: Fig2Config) -> dict[str, Any]:
    seed_everything(int(cfg.network_seed))
    seed_dir = _resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = _prepare_dirs(seed_dir)
    device = resolve_device(cfg.device)
    dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, NUM_CLASSES)
    max_duration = max(cfg.sample_ms, cfg.second_item_ms, cfg.weak_probe_ms, 100)
    warnings: list[str] = []
    if Path(cfg.model_path).exists():
        net, encoder = load_model_and_encoder(cfg.model_path, device=device, dt=cfg.dt, max_duration_ms=max_duration)
    elif cfg.smoke:
        seed_everything(int(cfg.network_seed))
        net = SDNN_Network(device=str(device)).to(device)
        net.eval()
        encoder = DoGSpikeEncoder(dt=cfg.dt, max_duration=max_duration * ms, device=str(device))
        warnings.append(
            "Model checkpoint missing; smoke mode used an untrained repo SDNN_Network instance. "
            "Functional E/F outputs are still real network rollouts, but are not manuscript evidence."
        )
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {cfg.model_path}")
    ctx = ExperimentContext(
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
        run_log=[f"{_now()} start {FIGURE_ID} seed={cfg.network_seed} smoke={cfg.smoke}"],
    )
    run_info = build_run_info(
        experiment_name=FIGURE_ID,
        output_dir=seed_dir,
        entry_script="src.experiments.paper_figures.fig2_pair_fused_stsp_state_experiment",
        seed=cfg.network_seed,
        dataset=f"MNIST:{cfg.split}",
        command=" ".join(sys.argv),
        model_path=cfg.model_path,
        status="running",
    )
    write_run_info(seed_dir / "meta", run_info)

    try:
        _write_config_files(ctx)
        pair_trials = build_pair_trial_specs(ctx)
        bank: PairEpisodeStateBank | None = None
        needs_bank = any(
            (
                cfg.run_state_bank,
                cfg.run_morphology,
                cfg.run_linear_mixture,
                cfg.run_neutral_ping,
                cfg.run_partial_cue,
                cfg.run_supplement,
                cfg.run_ping_sweep,
            )
        )
        if needs_bank:
            bank = run_pair_episode_state_bank(ctx, pair_trials)
        if bank is not None and cfg.run_morphology:
            compute_dual_retention_metrics(ctx, bank)
            compute_pair_specificity_metrics(ctx, bank)
            compute_pair_level_organization_metrics(ctx, bank)
        if bank is not None and cfg.run_linear_mixture:
            compute_linear_mixture_metrics(ctx, bank)
            compute_linear_residual_pair_specificity(ctx, bank)
        if bank is not None and cfg.run_neutral_ping:
            run_neutral_ping_real_rollout_from_state_bank(ctx, bank)
        if bank is not None and cfg.run_ping_sweep:
            run_neutral_ping_parameter_sweep(ctx, bank)
        if bank is not None and cfg.run_partial_cue:
            run_partial_cue_real_rollout_from_state_bank(ctx, bank)
        if cfg.run_completion_delay_sweep:
            run_completion_delay_sweep_from_pair_trials(ctx, pair_trials)
        if bank is not None and cfg.save_proxy_functional_debug:
            write_functional_proxy_diagnostics(ctx, bank)
        if bank is not None and cfg.run_supplement:
            compute_supplementary_metrics(ctx, bank)
        if cfg.save_debug_figures:
            save_debug_figures(ctx)
        summary = _write_summary(ctx)
        _write_run_log(ctx)
        finalize_run_info(seed_dir / "meta", run_info, status="success")
        return summary
    except Exception:
        finalize_run_info(seed_dir / "meta", run_info, status="failed")
        raise


def build_pair_trial_specs(ctx: ExperimentContext) -> pd.DataFrame:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg.network_seed))
    images_cache: dict[int, torch.Tensor] = {}

    def image_flat(image_id: int) -> np.ndarray:
        if image_id not in images_cache:
            images_cache[image_id] = ctx.dataset[int(image_id)][0].detach().cpu().to(torch.float32)
        return images_cache[image_id].reshape(-1).numpy().astype(np.float64, copy=False)

    class_pairs = [(a, b) for a in range(NUM_CLASSES) for b in range(NUM_CLASSES) if a != b]
    target_pairs = [class_pairs[i % len(class_pairs)] for i in range(int(cfg.num_pairs))]
    rng.shuffle(target_pairs)
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    candidate_id = 0
    for pair_id, (a_label, b_label) in _progress(
        enumerate(target_pairs),
        total=len(target_pairs),
        desc="fig2 pair specs",
        enabled=ctx.cfg.show_progress,
    ):
        a_pool = np.asarray(ctx.class_index[int(a_label)], dtype=np.int64)
        b_pool = np.asarray(ctx.class_index[int(b_label)], dtype=np.int64)
        local_candidates = []
        for _ in range(6):
            a_img = int(rng.choice(a_pool))
            b_img = int(rng.choice(b_pool))
            if a_img == b_img:
                continue
            sim, overlap = _image_similarity_and_overlap(image_flat(a_img), image_flat(b_img))
            local_candidates.append((a_img, b_img, sim, overlap))
            candidate_rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "candidate_id": int(candidate_id),
                    "A_image_id": a_img,
                    "A_label": int(a_label),
                    "B_image_id": b_img,
                    "B_label": int(b_label),
                    "pixel_similarity": sim,
                    "foreground_overlap": overlap,
                    "eligible": 1,
                    "exclusion_reason": "",
                }
            )
            candidate_id += 1
        if not local_candidates:
            a_img = int(rng.choice(a_pool))
            b_img = int(rng.choice(b_pool))
            sim, overlap = _image_similarity_and_overlap(image_flat(a_img), image_flat(b_img))
        else:
            local_candidates.sort(key=lambda item: abs(item[2] - 0.35) + 0.2 * item[3])
            a_img, b_img, sim, overlap = local_candidates[0]
        rows.append(
            {
                "network_seed": int(cfg.network_seed),
                "pair_id": int(pair_id),
                "A_image_id": int(a_img),
                "A_label": int(a_label),
                "B_image_id": int(b_img),
                "B_label": int(b_label),
                "pair_seed": int(rng.integers(0, 2**31 - 1)),
                "pixel_similarity": float(sim),
                "foreground_overlap": float(overlap),
                "class_pair": f"{int(a_label)}->{int(b_label)}",
                "selection_bin": _selection_bin(sim),
            }
        )

    pair_trials = pd.DataFrame(rows)
    pool = pd.DataFrame(candidate_rows)
    audit = _pair_sampling_audit(cfg.network_seed, pair_trials, pool)
    _save_csv(ctx, pair_trials, ctx.trial_specs_dir / "pair_trials.csv")
    _save_csv(ctx, pool, ctx.trial_specs_dir / "pair_candidate_pool.csv")
    _save_csv(ctx, audit, ctx.metrics_dir / "supp_pair_sampling_audit.csv")
    _save_csv(ctx, _trial_condition_audit(cfg.network_seed, pair_trials), ctx.metrics_dir / "supp_trial_condition_audit.csv")
    ctx.n_pairs = int(len(pair_trials))
    ctx.completed_modules["pair_trial_specs"] = True
    return pair_trials


def run_pair_episode_state_bank(ctx: ExperimentContext, pair_trials: pd.DataFrame) -> PairEpisodeStateBank:
    cfg = ctx.cfg
    arrays: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        cond: {layer: {"u": [], "x": [], "g": []} for layer in LAYER_KEYS} for cond in STATE_CONDITIONS
    }
    boundary_states: dict[str, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    manifest_rows: list[dict[str, Any]] = []
    all_layer_manifest_rows: list[dict[str, Any]] = []
    first_episode_saved = False

    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    batches = _iter_batches(pair_trials, cfg.batch_size)
    for batch in _progress(
        batches,
        total=math.ceil(len(pair_trials) / cfg.batch_size),
        desc="fig2 state batches",
        enabled=cfg.show_progress,
    ):
        a_spikes = _encode_cached(ctx, batch["A_image_id"].to_numpy(), cfg.sample_steps, cache=encode_cache)
        b_spikes = _encode_cached(ctx, batch["B_image_id"].to_numpy(), cfg.second_item_steps, cache=encode_cache)
        batch_bank, batch_boundaries = _capture_pair_batch(ctx, a_spikes, b_spikes)
        for cond in STATE_CONDITIONS:
            if cond not in boundary_states:
                boundary_states[cond] = batch_boundaries[cond]
            else:
                boundary_states[cond] = _concat_boundary_states(boundary_states[cond], batch_boundaries[cond])
            for layer in LAYER_KEYS:
                for variable in ("u", "x", "g"):
                    arrays[cond][layer][variable].append(batch_bank[cond][layer][variable])
        if not first_episode_saved and len(batch) > 0:
            first_episode_saved = True
            example = batch.iloc[0].to_dict()
            example_a_image = ctx.dataset[int(example["A_image_id"])][0].detach().cpu().to(torch.float32).numpy()
            example_b_image = ctx.dataset[int(example["B_image_id"])][0].detach().cpu().to(torch.float32).numpy()
            _write_json(_json_safe(example), ctx.raw_dir / "panel_a_example_episode_metadata.json")
            np.savez_compressed(
                ctx.raw_dir / "panel_a_example_episode.npz",
                A_image=example_a_image,
                B_image=example_b_image,
            )
            ctx.output_files["panel_a_example_episode_metadata"] = _rel(ctx.raw_dir / "panel_a_example_episode_metadata.json", ctx.seed_dir)
            ctx.output_files["panel_a_example_episode"] = _rel(ctx.raw_dir / "panel_a_example_episode.npz", ctx.seed_dir)

    final_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {
        cond: {layer: {} for layer in LAYER_KEYS} for cond in STATE_CONDITIONS
    }
    l3_payload: dict[str, np.ndarray] = {}
    all_layer_payload: dict[str, np.ndarray] = {}
    for cond in STATE_CONDITIONS:
        for layer in LAYER_KEYS:
            for variable in ("u", "x", "g"):
                arr = np.vstack(arrays[cond][layer][variable]).astype(np.float32, copy=False)
                final_arrays[cond][layer][variable] = arr
                storage_file = "state_bank_l3.npz" if layer == "layer3" else ("state_bank_all_layers.npz" if cfg.save_all_layer_state_bank else "")
                storage_key = f"{cond}_{variable}" if layer == "layer3" else f"{cond}_{layer}_{variable}"
                row = {
                    "network_seed": int(cfg.network_seed),
                    "pair_id": "all",
                    "state_condition": cond,
                    "layer": layer,
                    "state_variable": variable,
                    "shape": "x".join(str(v) for v in arr.shape),
                    "storage_file": storage_file,
                    "storage_key": storage_key if storage_file else "",
                    "captured_after": "A_delay1_B_delay2",
                    "sample_ms": int(cfg.sample_ms),
                    "delay1_ms": int(cfg.delay1_ms),
                    "second_item_ms": int(cfg.second_item_ms),
                    "delay2_ms": int(cfg.delay2_ms),
                }
                all_layer_manifest_rows.append(row)
                if layer == "layer3":
                    manifest_rows.append(row)
                    l3_payload[f"{cond}_{variable}"] = arr
                elif cfg.save_all_layer_state_bank:
                    all_layer_payload[storage_key] = arr
            final_arrays[cond][layer]["ux_concat"] = np.concatenate(
                [final_arrays[cond][layer]["u"], final_arrays[cond][layer]["x"]],
                axis=1,
            ).astype(np.float32, copy=False)
    np.savez_compressed(ctx.raw_dir / "state_bank_l3.npz", **l3_payload)
    ctx.output_files["state_bank_l3"] = _rel(ctx.raw_dir / "state_bank_l3.npz", ctx.seed_dir)
    if cfg.save_all_layer_state_bank:
        np.savez_compressed(ctx.raw_dir / "state_bank_all_layers.npz", **all_layer_payload)
        ctx.output_files["state_bank_all_layers"] = _rel(ctx.raw_dir / "state_bank_all_layers.npz", ctx.seed_dir)
    _save_csv(ctx, pd.DataFrame(manifest_rows), ctx.raw_dir / "state_bank_manifest.csv")
    _save_csv(ctx, pd.DataFrame(all_layer_manifest_rows), ctx.raw_dir / "state_bank_all_layers_manifest.csv")
    layer_input_shapes = _layer_input_shapes_from_boundary(boundary_states["S0"])
    ctx.completed_modules["state_bank"] = True
    return PairEpisodeStateBank(
        pair_trials=pair_trials.reset_index(drop=True),
        arrays=final_arrays,
        boundary_states=boundary_states,
        layer_input_shapes=layer_input_shapes,
        restore_mode="reset_all_state_restore_selected_stsp",
        episode_end_step=int(cfg.sample_steps + cfg.delay1_steps + cfg.second_item_steps + cfg.delay2_steps),
    )


def compute_dual_retention_metrics(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows = []
    for layer in _progress(LAYER_KEYS, total=len(LAYER_KEYS), desc="fig2 dual layers", enabled=ctx.cfg.show_progress):
        for variable in STATE_VARIABLES:
            s_ab = bank.get("S_AB", layer, variable)
            s_a = bank.get("S_A", layer, variable)
            s_b = bank.get("S_B", layer, variable)
            sim_a = _row_centered_cosine(s_ab, s_a)
            sim_b = _row_centered_cosine(s_ab, s_b)
            for idx, pair_id in enumerate(bank.pair_trials["pair_id"].to_numpy(dtype=np.int64)):
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(pair_id),
                        "layer": layer,
                        "state_variable": variable,
                        "sim_to_A": float(sim_a[idx]),
                        "sim_to_B": float(sim_b[idx]),
                        "fusion_dual_score": float(0.5 * (sim_a[idx] + sim_b[idx])),
                        "min_component_similarity": float(min(sim_a[idx], sim_b[idx])),
                        "sim_to_A_minus_B": float(sim_a[idx] - sim_b[idx]),
                    }
                )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "panel_b_dual_retention_metrics.csv")
    ctx.completed_modules["dual_retention"] = True


def compute_pair_specificity_metrics(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows = []
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 202)
    n = len(bank.pair_trials)
    for layer in _progress(LAYER_KEYS, total=len(LAYER_KEYS), desc="fig2 specificity layers", enabled=ctx.cfg.show_progress):
        for variable in STATE_VARIABLES:
            s_ab = bank.get("S_AB", layer, variable)
            s_a = bank.get("S_A", layer, variable)
            s_b = bank.get("S_B", layer, variable)
            true_comp = 0.5 * (s_a + s_b)
            true_score = _row_centered_cosine(s_ab, true_comp)
            for i, pair_id in enumerate(bank.pair_trials["pair_id"].to_numpy(dtype=np.int64)):
                choices = [j for j in range(n) if j != i] or [i]
                sampled = rng.choice(choices, size=int(ctx.cfg.n_shuffle), replace=len(choices) < int(ctx.cfg.n_shuffle))
                scores = []
                for j in sampled:
                    pseudo = 0.5 * (s_a[i : i + 1] + s_b[int(j) : int(j) + 1])
                    scores.append(float(_row_centered_cosine(s_ab[i : i + 1], pseudo)[0]))
                shuf_mean = float(np.mean(scores)) if scores else float("nan")
                shuf_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                percentile = float(np.mean(np.asarray(scores) <= true_score[i])) if scores else float("nan")
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(pair_id),
                        "layer": layer,
                        "state_variable": variable,
                        "true_pair_score": float(true_score[i]),
                        "shuffled_pair_score": shuf_mean,
                        "pseudo_pair_score": shuf_mean,
                        "true_minus_shuffled": float(true_score[i] - shuf_mean),
                        "true_pair_percentile": percentile,
                        "true_pair_z": float((true_score[i] - shuf_mean) / max(shuf_std, 1e-8)),
                        "n_shuffle": int(ctx.cfg.n_shuffle),
                    }
                )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "panel_c_pair_specificity_metrics.csv")
    ctx.completed_modules["pair_specificity"] = True


def compute_pair_level_organization_metrics(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows = []
    for layer in LAYER_KEYS:
        for variable in STATE_VARIABLES:
            s_ab = bank.get("S_AB", layer, variable)
            s_a = bank.get("S_A", layer, variable)
            s_b = bank.get("S_B", layer, variable)
            comp = 0.5 * (s_a + s_b)
            sim_pair = _row_centered_cosine(s_ab, comp)
            sim_a = _row_centered_cosine(s_ab, s_a)
            sim_b = _row_centered_cosine(s_ab, s_b)
            for idx, pair_id in enumerate(bank.pair_trials["pair_id"].to_numpy(dtype=np.int64)):
                best = max(float(sim_a[idx]), float(sim_b[idx]))
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(pair_id),
                        "layer": layer,
                        "state_variable": variable,
                        "sim_to_true_pair": float(sim_pair[idx]),
                        "sim_to_A": float(sim_a[idx]),
                        "sim_to_B": float(sim_b[idx]),
                        "best_constituent_similarity": best,
                        "WPRI": float(sim_pair[idx] - best),
                    }
                )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "panel_d_pair_level_organization_metrics.csv")
    ctx.completed_modules["pair_level_organization"] = True


def compute_linear_mixture_metrics(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows = []
    supp_rows = []
    null_rows = []
    for layer in _progress(LAYER_KEYS, total=len(LAYER_KEYS), desc="fig2 mixture layers", enabled=ctx.cfg.show_progress):
        for variable in STATE_VARIABLES:
            z0 = bank.get("S0", layer, variable)
            x_a = bank.get("S_A", layer, variable) - z0
            x_b = bank.get("S_B", layer, variable) - z0
            y = bank.get("S_AB", layer, variable) - z0
            for idx, pair_id in enumerate(bank.pair_trials["pair_id"].to_numpy(dtype=np.int64)):
                models = _fit_mixture_models(x_a[idx], x_b[idx], y[idx], ctx.cfg.linear_mixture_cv_folds, ctx.cfg.network_seed + idx)
                best_single = max(models["A_only"]["r2"], models["B_only"]["r2"])
                for model_name, metrics in models.items():
                    cosine_to_sab = float(_centered_cosine(y[idx], metrics["prediction"]))
                    row = {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(pair_id),
                        "layer": layer,
                        "state_variable": variable,
                        "model_name": model_name,
                        "r2": float(metrics["r2"]),
                        "cv_r2": float(metrics["cv_r2"]),
                        "residual_norm": float(metrics["residual_norm"]),
                        "target_norm": float(metrics["target_norm"]),
                        "residual_norm_ratio": float(metrics["residual_norm_ratio"]),
                        "beta_A": _maybe_float(metrics.get("beta_A")),
                        "beta_B": _maybe_float(metrics.get("beta_B")),
                        "intercept": _maybe_float(metrics.get("intercept")),
                        "convex_weight_A": _maybe_float(metrics.get("convex_weight_A")),
                        "convex_weight_B": _maybe_float(metrics.get("convex_weight_B")),
                        "best_single_constituent_r2": float(best_single),
                        "linear_mixture_gain": float(models["unconstrained_AB"]["r2"] - best_single),
                    }
                    rows.append(row)
                    supp_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "pair_id": int(pair_id),
                            "layer": layer,
                            "state_variable": variable,
                            "model_name": model_name,
                            "mixture_model": model_name,
                            "r2": float(metrics["r2"]),
                            "fit_r2": float(metrics["r2"]),
                            "cv_r2": float(metrics["cv_r2"]),
                            "fold_id": "",
                            "cv_fold": "",
                            "residual_norm": float(metrics["residual_norm"]),
                            "residual_norm_ratio": float(metrics["residual_norm_ratio"]),
                            "cosine_to_SAB": cosine_to_sab,
                            "n_pairs": int(len(bank.pair_trials)),
                            "notes": "",
                        }
                    )
                for null_name in ("mean_AB", "sum_AB", "unconstrained_AB", "convex_AB"):
                    pred = models[null_name]["prediction"]
                    null_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "pair_id": int(pair_id),
                            "layer": layer,
                            "state_variable": variable,
                            "null_model": null_name.replace("_AB", "").replace("unconstrained", "LS"),
                            "similarity_to_SAB": float(_centered_cosine(y[idx], pred)),
                            "r2_to_SAB": float(models[null_name]["r2"]),
                            "residual_norm_ratio": float(models[null_name]["residual_norm_ratio"]),
                            "notes": "baseline_subtracted_against_S0",
                        }
                    )
    _save_csv(ctx, pd.DataFrame(rows).drop(columns=["prediction"], errors="ignore"), ctx.metrics_dir / "panel_d_linear_mixture_fit_metrics.csv")
    _save_csv(ctx, pd.DataFrame(supp_rows), ctx.metrics_dir / "supp_linear_mixture_model_comparison.csv")
    _save_csv(ctx, pd.DataFrame(null_rows), ctx.metrics_dir / "supp_additive_null_metrics.csv")
    ctx.completed_modules["linear_mixture"] = True


def compute_linear_residual_pair_specificity(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows = []
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 303)
    n = len(bank.pair_trials)
    for layer in LAYER_KEYS:
        for variable in STATE_VARIABLES:
            z0 = bank.get("S0", layer, variable)
            x_a = bank.get("S_A", layer, variable) - z0
            x_b = bank.get("S_B", layer, variable) - z0
            y = bank.get("S_AB", layer, variable) - z0
            for idx, pair_id in enumerate(bank.pair_trials["pair_id"].to_numpy(dtype=np.int64)):
                model = _fit_mixture_models(x_a[idx], x_b[idx], y[idx], ctx.cfg.linear_mixture_cv_folds, ctx.cfg.network_seed + idx)["unconstrained_AB"]
                residual = y[idx] - model["prediction"]
                true_template = y[idx] - 0.5 * (x_a[idx] + x_b[idx])
                true_score = float(_centered_cosine(residual, true_template))
                choices = [j for j in range(n) if j != idx] or [idx]
                sampled = rng.choice(choices, size=int(ctx.cfg.n_shuffle), replace=len(choices) < int(ctx.cfg.n_shuffle))
                scores = [float(_centered_cosine(residual, y[idx] - 0.5 * (x_a[idx] + x_b[int(j)]))) for j in sampled]
                shuf = float(np.mean(scores)) if scores else float("nan")
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(pair_id),
                        "layer": layer,
                        "state_variable": variable,
                        "residual_true_pair_score": true_score,
                        "residual_shuffled_pair_score": shuf,
                        "residual_pair_specificity": float(true_score - shuf),
                        "beyond_linear_pair_index": float(true_score - shuf),
                        "shuffle_id": "mean",
                        "n_shuffle": int(ctx.cfg.n_shuffle),
                        "residual_template_definition": RESIDUAL_TEMPLATE_DEFINITION,
                    }
                )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "panel_d_linear_residual_pair_specificity_metrics.csv")
    _write_layerwise_morphology_metrics(ctx)
    ctx.completed_modules["linear_residual_pair_specificity"] = True


def run_neutral_ping_real_rollout_from_state_bank(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows: list[dict[str, Any]] = []
    trace_payload: dict[str, np.ndarray] = {}
    for ping_repeat in _progress(range(int(ctx.cfg.ping_repeats)), total=int(ctx.cfg.ping_repeats), desc="fig2 ping repeats", enabled=ctx.cfg.show_progress):
        ping_seed = int(ctx.cfg.network_seed * 1009 + 200 + ping_repeat)
        for condition in _progress(STATE_CONDITIONS, total=len(STATE_CONDITIONS), desc="fig2 ping states", enabled=ctx.cfg.show_progress):
            boundary = bank.boundary_states[condition]
            readout = run_ping_readout_from_boundary(ctx, boundary, ping_seed=ping_seed, record_trace=ctx.cfg.save_functional_traces)
            if readout.trace:
                for key, value in readout.trace.items():
                    trace_payload[f"{condition}_repeat_{ping_repeat}_{key}"] = value
            for idx, rec in bank.pair_trials.reset_index(drop=True).iterrows():
                pred = int(readout.prediction[idx])
                a_label = int(rec["A_label"])
                b_label = int(rec["B_label"])
                silent = bool(readout.silent[idx])
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(rec["pair_id"]),
                        "state_condition": condition,
                        "ping_repeat": int(ping_repeat),
                        "ping_seed": int(ping_seed),
                        "A_label": a_label,
                        "B_label": b_label,
                        "prediction": pred,
                        "pred_is_A": int(pred == a_label),
                        "pred_is_B": int(pred == b_label),
                        "pred_is_pair_member": int(pred in {a_label, b_label}),
                        "pred_is_other": int((not silent) and pred not in {a_label, b_label}),
                        "silent": int(silent),
                        "first_fire_time_ms": float(readout.first_fire_time_ms[idx]),
                        "ping_spike_count": float(_ping_spike_count(ctx, ping_seed)),
                        "ping_energy": float(_ping_energy(ctx, ping_seed)),
                        "readout_margin_A": _readout_margin_value(readout.readout_margin_A, idx),
                        "readout_margin_B": _readout_margin_value(readout.readout_margin_B, idx),
                    }
                )
    trial_df = pd.DataFrame(rows, columns=PANEL_E_RAW_COLUMNS)
    _save_csv(ctx, trial_df, ctx.raw_dir / "panel_e_neutral_ping_trial_readout.csv")
    metrics = _neutral_ping_metrics(ctx.cfg.network_seed, trial_df)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_e_neutral_ping_metrics.csv")
    if ctx.cfg.save_functional_traces:
        np.savez_compressed(ctx.raw_dir / "panel_e_neutral_ping_l3_traces.npz", **trace_payload)
        ctx.output_files["panel_e_neutral_ping_l3_traces"] = _rel(ctx.raw_dir / "panel_e_neutral_ping_l3_traces.npz", ctx.seed_dir)
    ctx.completed_modules["neutral_ping"] = True


def run_neutral_ping_parameter_sweep(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows: list[dict[str, Any]] = []
    sweep_specs: list[tuple[str, float, int]] = []
    sweep_specs.extend(("amplitude", float(amp), int(ctx.cfg.ping_ms)) for amp in ctx.cfg.ping_amp_sweep)
    sweep_specs.extend(("duration", float(ctx.cfg.ping_amp), int(ping_ms)) for ping_ms in ctx.cfg.ping_ms_sweep)
    for sweep_type, ping_amp, ping_ms in _progress(sweep_specs, total=len(sweep_specs), desc="fig2 ping sweep", enabled=ctx.cfg.show_progress):
        for condition in STATE_CONDITIONS:
            ping_seed = _stable_sweep_seed(ctx.cfg.network_seed, sweep_type, ping_amp, ping_ms, condition)
            readout = run_ping_readout_from_boundary(
                ctx,
                bank.boundary_states[condition],
                ping_seed=ping_seed,
                ping_amp=ping_amp,
                ping_steps=_ms_to_steps(ping_ms, ctx.cfg.dt),
                record_trace=False,
            )
            for idx, rec in bank.pair_trials.reset_index(drop=True).iterrows():
                pred = int(readout.prediction[idx])
                a_label = int(rec["A_label"])
                b_label = int(rec["B_label"])
                silent = bool(readout.silent[idx])
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": int(rec["pair_id"]),
                        "state_condition": condition,
                        "sweep_type": sweep_type,
                        "ping_amp": float(ping_amp),
                        "ping_ms": int(ping_ms),
                        "ping_repeat": 0,
                        "A_label": a_label,
                        "B_label": b_label,
                        "prediction": pred,
                        "pred_is_A": int(pred == a_label),
                        "pred_is_B": int(pred == b_label),
                        "pred_is_pair_member": int(pred in {a_label, b_label}),
                        "pred_is_other": int((not silent) and pred not in {a_label, b_label}),
                        "silent": int(silent),
                        "first_fire_time_ms": float(readout.first_fire_time_ms[idx]),
                    }
                )
    trial_df = pd.DataFrame(rows, columns=SUPP_PING_SWEEP_RAW_COLUMNS)
    _save_csv(ctx, trial_df, ctx.raw_dir / "supp_ping_sweep_trial_readout.csv")
    _save_csv(ctx, _ping_sweep_metrics(ctx.cfg.network_seed, trial_df), ctx.metrics_dir / "supp_ping_sweep_metrics.csv")
    ctx.completed_modules["ping_sweep"] = True


def run_partial_cue_real_rollout_from_state_bank(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 404)
    raw_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    trace_payload: dict[str, np.ndarray] = {}
    mask_id = 0
    pair_iter = bank.pair_trials.iterrows()
    for _, rec in _progress(pair_iter, total=len(bank.pair_trials), desc="fig2 partial cue pairs", enabled=ctx.cfg.show_progress):
        pair_id = int(rec["pair_id"])
        labels = {"A": int(rec["A_label"]), "B": int(rec["B_label"])}
        image_ids = {"A": int(rec["A_image_id"]), "B": int(rec["B_image_id"])}
        for target_item in ("A", "B"):
            target_label = labels[target_item]
            other_label = labels["B" if target_item == "A" else "A"]
            target_image = ctx.dataset[image_ids[target_item]][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
            full_target_spikes = encode_images(ctx.encoder, target_image, ctx.cfg.weak_probe_steps).to(ctx.device)
            for keep_prob in ctx.cfg.weak_probe_keep_probs:
                for repeat_id in range(int(ctx.cfg.weak_probe_repeats)):
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    if ctx.cfg.weak_probe_mask_space == "encoded_spikes":
                        weak_spikes, mask_info = _make_weak_probe_spikes_encoded_dropout(
                            full_target_spikes,
                            float(keep_prob),
                            seed=mask_seed,
                            same_mask_count=len(STATE_CONDITIONS),
                            use_same_mask_across_states=ctx.cfg.weak_probe_use_same_mask_across_states,
                            device=ctx.device,
                        )
                    elif ctx.cfg.weak_probe_mask_space == "image_foreground":
                        weak_spikes_1, mask_info = _make_weak_probe_spikes_image_foreground(
                            ctx,
                            image_ids[target_item],
                            target_item,
                            float(keep_prob),
                            seed=mask_seed,
                        )
                        weak_spikes = weak_spikes_1.repeat(len(STATE_CONDITIONS), 1, 1, 1, 1)
                    else:
                        raise ValueError(f"Unsupported weak_probe_mask_space={ctx.cfg.weak_probe_mask_space}")
                    mask_rows.append(
                        _weak_probe_mask_row(
                            ctx,
                            mask_id=mask_id,
                            pair_id=pair_id,
                            target_item=target_item,
                            target_label=target_label,
                            keep_prob=float(keep_prob),
                            repeat_id=repeat_id,
                            mask_seed=mask_seed,
                            mask_info=mask_info,
                        )
                    )
                    boundary = concat_condition_boundaries(bank.boundary_states, STATE_CONDITIONS, [pair_id], ctx.device)
                    readout = run_probe_readout_from_boundary(
                        ctx,
                        boundary,
                        weak_spikes,
                        probe_scale=float(ctx.cfg.weak_probe_scale),
                        probe_noise=float(ctx.cfg.weak_probe_noise),
                        seed=mask_seed + 31,
                        record_trace=ctx.cfg.save_functional_traces,
                    )
                    if readout.trace:
                        for key, value in readout.trace.items():
                            trace_payload[f"mask_{mask_id}_{key}"] = value
                    for condition_index, condition in enumerate(STATE_CONDITIONS):
                        pred = int(readout.prediction[condition_index])
                        silent = bool(readout.silent[condition_index])
                        raw_rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "pair_id": pair_id,
                                "state_condition": condition,
                                "target_item": target_item,
                                "target_label": int(target_label),
                                "other_pair_label": int(other_label),
                                "keep_prob": float(keep_prob),
                                "repeat_id": int(repeat_id),
                                "mask_id": int(mask_id),
                                "prediction": pred,
                                "pred_is_target": int(pred == target_label),
                                "pred_is_A": int(pred == labels["A"]),
                                "pred_is_B": int(pred == labels["B"]),
                                "pred_is_pair_member": int(pred in {labels["A"], labels["B"]}),
                                "pred_is_other_pair_member": int(pred == other_label),
                                "pred_is_other_class": int((not silent) and pred not in {labels["A"], labels["B"]}),
                                "silent": int(silent),
                                "first_fire_time_ms": float(readout.first_fire_time_ms[condition_index]),
                                "mask_space": str(mask_info.get("mask_space", ctx.cfg.weak_probe_mask_space)),
                                "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
                                "weak_probe_noise": float(ctx.cfg.weak_probe_noise),
                                "weak_probe_metric_mode": str(ctx.cfg.weak_probe_metric_mode),
                                "realized_keep_fraction": _maybe_float(mask_info.get("realized_keep_fraction")),
                                "cue_fraction_actual": _maybe_float(mask_info.get("cue_fraction_actual")),
                                "weak_spike_fraction": _maybe_float(mask_info.get("weak_spike_fraction")),
                                "same_mask_used_across_states": bool(mask_info.get("same_mask_used_across_states", ctx.cfg.weak_probe_use_same_mask_across_states)),
                                "cue_pixel_count": _maybe_int(mask_info.get("cue_pixel_count")),
                                "target_foreground_count": _maybe_int(mask_info.get("target_foreground_count")),
                                "cue_energy": _maybe_float(mask_info.get("cue_energy")),
                                "encoded_spike_count": _maybe_float(mask_info.get("encoded_spike_count", mask_info.get("weak_spike_count"))),
                            }
                        )
                    mask_id += 1
    mask_df = pd.DataFrame(mask_rows, columns=WEAK_PROBE_MASK_COLUMNS)
    raw_df = pd.DataFrame(raw_rows, columns=PANEL_F_RAW_COLUMNS)
    _save_csv(ctx, mask_df, ctx.trial_specs_dir / "weak_probe_masks.csv")
    _save_csv(ctx, raw_df, ctx.raw_dir / "panel_f_partial_cue_trial_readout.csv")
    metrics = _partial_cue_metrics(ctx.cfg.network_seed, raw_df)
    auc = _partial_cue_auc_metrics(ctx.cfg.network_seed, metrics)
    pair_metrics = _partial_cue_pair_metrics(ctx.cfg.network_seed, raw_df)
    compat_summary, compat_auc, compat_threshold = _compat_fig4_weak_probe_outputs(ctx.cfg.network_seed, metrics, auc)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_f_partial_cue_metrics.csv")
    _save_csv(ctx, auc, ctx.metrics_dir / "panel_f_partial_cue_auc_metrics.csv")
    _save_csv(ctx, pair_metrics, ctx.metrics_dir / "panel_f_partial_cue_pair_metrics.csv")
    _save_csv(ctx, compat_summary, ctx.metrics_dir / "compat_fig4_weak_probe_summary.csv")
    _save_csv(ctx, compat_auc, ctx.metrics_dir / "compat_fig4_weak_probe_auc.csv")
    _save_csv(ctx, compat_threshold, ctx.metrics_dir / "compat_fig4_weak_probe_threshold.csv")
    _save_csv(ctx, metrics[metrics["target_item"] == "B"].copy(), ctx.metrics_dir / "supp_completion_target_B_metrics.csv")
    if ctx.cfg.save_functional_traces:
        np.savez_compressed(ctx.raw_dir / "panel_f_partial_cue_l3_traces.npz", **trace_payload)
        ctx.output_files["panel_f_partial_cue_l3_traces"] = _rel(ctx.raw_dir / "panel_f_partial_cue_l3_traces.npz", ctx.seed_dir)
    ctx.completed_modules["partial_cue"] = True


def run_completion_delay_sweep_from_pair_trials(ctx: ExperimentContext, pair_trials: pd.DataFrame) -> None:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 909)
    raw_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    conditions = ("S0", "S_B", "S_AB")
    for delay2_ms in _progress(ctx.cfg.completion_delay_sweep_ms, total=len(ctx.cfg.completion_delay_sweep_ms), desc="fig2 completion delay sweep", enabled=ctx.cfg.show_progress):
        delay2_steps = _ms_to_steps(int(delay2_ms), ctx.cfg.dt)
        for batch in _iter_batches(pair_trials, ctx.cfg.batch_size):
            a_spikes = _encode_cached(ctx, batch["A_image_id"].to_numpy(), ctx.cfg.sample_steps, cache=encode_cache)
            b_spikes = _encode_cached(ctx, batch["B_image_id"].to_numpy(), ctx.cfg.second_item_steps, cache=encode_cache)
            _batch_bank, batch_boundaries = _capture_pair_batch(ctx, a_spikes, b_spikes, delay2_steps=delay2_steps)
            for batch_idx, rec in batch.reset_index(drop=True).iterrows():
                pair_id = int(rec["pair_id"])
                a_label = int(rec["A_label"])
                b_label = int(rec["B_label"])
                target_image = ctx.dataset[int(rec["A_image_id"])][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
                full_target_spikes = encode_images(ctx.encoder, target_image, ctx.cfg.weak_probe_steps).to(ctx.device)
                for repeat_id in range(int(ctx.cfg.completion_delay_repeats)):
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    if ctx.cfg.weak_probe_mask_space == "encoded_spikes":
                        weak_spikes, mask_info = _make_weak_probe_spikes_encoded_dropout(
                            full_target_spikes,
                            float(ctx.cfg.completion_delay_keep_prob),
                            seed=mask_seed,
                            same_mask_count=len(conditions),
                            use_same_mask_across_states=True,
                            device=ctx.device,
                        )
                    elif ctx.cfg.weak_probe_mask_space == "image_foreground":
                        weak_spikes_1, mask_info = _make_weak_probe_spikes_image_foreground(
                            ctx,
                            int(rec["A_image_id"]),
                            "A",
                            float(ctx.cfg.completion_delay_keep_prob),
                            seed=mask_seed,
                        )
                        weak_spikes = weak_spikes_1.repeat(len(conditions), 1, 1, 1, 1)
                    else:
                        raise ValueError(f"Unsupported weak_probe_mask_space={ctx.cfg.weak_probe_mask_space}")
                    boundary = concat_condition_boundaries(batch_boundaries, conditions, [int(batch_idx)], ctx.device)
                    readout = run_probe_readout_from_boundary(
                        ctx,
                        boundary,
                        weak_spikes,
                        probe_scale=float(ctx.cfg.weak_probe_scale),
                        probe_noise=float(ctx.cfg.weak_probe_noise),
                        seed=mask_seed + 31,
                        record_trace=False,
                    )
                    weak_spike_count = _maybe_float(mask_info.get("weak_spike_count", mask_info.get("encoded_spike_count")))
                    for condition_index, condition in enumerate(conditions):
                        pred = int(readout.prediction[condition_index])
                        silent = bool(readout.silent[condition_index])
                        raw_rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "pair_id": pair_id,
                                "delay2_ms": int(delay2_ms),
                                "state_condition": condition,
                                "target_item": "A",
                                "target_label": a_label,
                                "A_label": a_label,
                                "B_label": b_label,
                                "keep_prob": float(ctx.cfg.completion_delay_keep_prob),
                                "repeat_id": int(repeat_id),
                                "prediction": pred,
                                "correct_target": int(pred == a_label),
                                "pred_is_A": int(pred == a_label),
                                "pred_is_B": int(pred == b_label),
                                "pred_is_other": int((not silent) and pred not in {a_label, b_label}),
                                "silent": int(silent),
                                "first_fire_time_ms": float(readout.first_fire_time_ms[condition_index]),
                                "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
                                "weak_spike_count": weak_spike_count,
                            }
                        )
    trial_df = pd.DataFrame(raw_rows, columns=SUPP_COMPLETION_DELAY_RAW_COLUMNS)
    metrics = _completion_delay_sweep_metrics(ctx.cfg.network_seed, trial_df)
    contrast = _completion_delay_sweep_contrast(ctx.cfg.network_seed, metrics)
    _save_csv(ctx, trial_df, ctx.raw_dir / "supp_completion_delay_sweep_trial_readout.csv")
    _save_csv(ctx, metrics, ctx.metrics_dir / "supp_completion_delay_sweep_metrics.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "supp_completion_delay_sweep_contrast.csv")
    ctx.completed_modules["completion_delay_sweep"] = True


def run_neutral_ping_accessibility_proxy(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> pd.DataFrame:
    rows = []
    for _, rec in bank.pair_trials.iterrows():
        pair_id = int(rec["pair_id"])
        a_label = int(rec["A_label"])
        b_label = int(rec["B_label"])
        for condition in STATE_CONDITIONS:
            scores = _access_scores(bank, pair_id, condition)
            pred = _prediction_from_scores(scores, a_label, b_label, ctx.cfg.network_seed + pair_id)
            silent = pred < 0
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": pair_id,
                    "state_condition": condition,
                    "ping_repeat": 0,
                    "prediction": int(pred),
                    "pred_is_A": int(pred == a_label),
                    "pred_is_B": int(pred == b_label),
                    "pred_is_pair_member": int(pred in {a_label, b_label}),
                    "pred_is_other": int((not silent) and pred not in {a_label, b_label}),
                    "silent": int(silent),
                    "first_fire_time_ms": -1 if silent else int(ctx.cfg.ping_ms // 2),
                }
            )
    return pd.DataFrame(rows)


def run_partial_cue_accessibility_proxy(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> pd.DataFrame:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 404)
    raw_rows = []
    mask_rows = []
    mask_id = 0
    for _, rec in bank.pair_trials.iterrows():
        pair_id = int(rec["pair_id"])
        labels = {"A": int(rec["A_label"]), "B": int(rec["B_label"])}
        for target_item in ("A", "B"):
            target_label = labels[target_item]
            other_label = labels["B" if target_item == "A" else "A"]
            for keep_prob in ctx.cfg.weak_probe_keep_probs:
                for repeat_id in range(int(ctx.cfg.weak_probe_repeats)):
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    mask_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "mask_id": int(mask_id),
                            "pair_id": pair_id,
                            "target_item": target_item,
                            "keep_prob": float(keep_prob),
                            "repeat_id": int(repeat_id),
                            "mask_seed": mask_seed,
                        }
                    )
                    for condition in STATE_CONDITIONS:
                        scores = _access_scores(bank, pair_id, condition)
                        target_score = scores[target_item] * (0.35 + 0.65 * float(keep_prob))
                        other_score = scores["B" if target_item == "A" else "A"] * 0.45
                        pred = target_label if target_score >= other_score and target_score > 0.08 else (other_label if other_score > 0.12 else -1)
                        silent = pred < 0
                        raw_rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "pair_id": pair_id,
                                "state_condition": condition,
                                "target_item": target_item,
                                "keep_prob": float(keep_prob),
                                "repeat_id": int(repeat_id),
                                "mask_id": int(mask_id),
                                "prediction": int(pred),
                                "pred_is_target": int(pred == target_label),
                                "pred_is_pair_member": int(pred in {target_label, other_label}),
                                "pred_is_other": int((not silent) and pred not in {target_label, other_label}),
                                "silent": int(silent),
                                "first_fire_time_ms": -1 if silent else int(ctx.cfg.weak_probe_ms // 2),
                            }
                        )
                    mask_id += 1
    return pd.DataFrame(raw_rows)


def _make_weak_probe_spikes_encoded_dropout(
    full_probe_spikes: torch.Tensor,
    keep_prob: float,
    *,
    seed: int,
    same_mask_count: int,
    use_same_mask_across_states: bool,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    full = full_probe_spikes.to(device=device, dtype=torch.float32)
    keep_prob = float(np.clip(float(keep_prob), 0.0, 1.0))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    if use_same_mask_across_states:
        mask = (torch.rand(full.shape, generator=gen, device=device) < keep_prob).to(torch.float32)
        weak = (full * mask).repeat(int(same_mask_count), 1, 1, 1, 1)
        realized_keep_fraction = float(mask.mean().detach().cpu().item())
    else:
        expanded = full.repeat(int(same_mask_count), 1, 1, 1, 1)
        mask = (torch.rand(expanded.shape, generator=gen, device=device) < keep_prob).to(torch.float32)
        weak = expanded * mask
        realized_keep_fraction = float(mask.mean().detach().cpu().item())
    full_spike_count = float(full.sum().detach().cpu().item())
    weak_spike_count = float(weak.sum().detach().cpu().item())
    denom = full_spike_count * float(same_mask_count)
    return weak.contiguous(), {
        "keep_prob": keep_prob,
        "mask_space": "encoded_spikes",
        "realized_keep_fraction": realized_keep_fraction,
        "full_spike_count": full_spike_count,
        "weak_spike_count": weak_spike_count,
        "weak_spike_fraction": float(weak_spike_count / denom) if denom > 0.0 else 0.0,
        "same_mask_used_across_states": bool(use_same_mask_across_states),
        "mask_seed": int(seed),
    }


def _make_weak_probe_spikes_image_foreground(
    ctx: ExperimentContext,
    target_image_id: int,
    target_item: str,
    keep_prob: float,
    *,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _ = target_item
    target_image = ctx.dataset[int(target_image_id)][0].detach().cpu().to(torch.float32).squeeze()
    foreground = target_image.numpy() > float(ctx.cfg.foreground_threshold)
    target_foreground_count = int(foreground.sum())
    mask_rng = np.random.default_rng(int(seed))
    keep = mask_rng.random(foreground.shape) < float(keep_prob)
    mask = foreground & keep
    partial = (target_image.numpy() * mask.astype(np.float32)).astype(np.float32)
    partial_tensor = torch.as_tensor(partial, dtype=torch.float32).view(1, 1, *partial.shape)
    partial_spikes = encode_images(ctx.encoder, partial_tensor.to(ctx.device), ctx.cfg.weak_probe_steps).to(ctx.device)
    encoded_spike_count = float(partial_spikes.sum().detach().cpu().item())
    cue_pixel_count = int(mask.sum())
    cue_energy = float(partial.sum())
    cue_fraction_actual = float(cue_pixel_count / max(1, target_foreground_count))
    return partial_spikes.contiguous(), {
        "keep_prob": float(keep_prob),
        "mask_space": "image_foreground",
        "cue_pixel_count": cue_pixel_count,
        "target_foreground_count": target_foreground_count,
        "cue_fraction_actual": cue_fraction_actual,
        "cue_energy": cue_energy,
        "encoded_spike_count": encoded_spike_count,
        "weak_spike_count": encoded_spike_count,
        "weak_spike_fraction": float("nan"),
        "same_mask_used_across_states": True,
        "mask_seed": int(seed),
    }


def _weak_probe_mask_row(
    ctx: ExperimentContext,
    *,
    mask_id: int,
    pair_id: int,
    target_item: str,
    target_label: int,
    keep_prob: float,
    repeat_id: int,
    mask_seed: int,
    mask_info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "mask_id": int(mask_id),
        "pair_id": int(pair_id),
        "target_item": str(target_item),
        "target_label": int(target_label),
        "keep_prob": float(keep_prob),
        "repeat_id": int(repeat_id),
        "mask_seed": int(mask_seed),
        "mask_space": str(mask_info.get("mask_space", ctx.cfg.weak_probe_mask_space)),
        "same_mask_used_across_states": bool(mask_info.get("same_mask_used_across_states", ctx.cfg.weak_probe_use_same_mask_across_states)),
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


def compute_supplementary_metrics(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    rows = []
    primary_var = ctx.cfg.primary_state_variable
    for delay2_ms in ctx.cfg.delay_layer_grid:
        scale = math.exp(-(float(delay2_ms) - float(ctx.cfg.delay2_ms)) / 1000.0)
        for layer in LAYER_KEYS:
            dual = _metric_lookup(ctx.metrics_dir / "panel_b_dual_retention_metrics.csv", layer, primary_var, "fusion_dual_score")
            spec = _metric_lookup(ctx.metrics_dir / "panel_c_pair_specificity_metrics.csv", layer, primary_var, "true_minus_shuffled")
            wpri = _metric_lookup(ctx.metrics_dir / "panel_d_pair_level_organization_metrics.csv", layer, primary_var, "WPRI")
            residual = _metric_lookup(ctx.metrics_dir / "panel_d_linear_residual_pair_specificity_metrics.csv", layer, primary_var, "residual_pair_specificity")
            linear = _linear_metric_lookup(ctx.metrics_dir / "panel_d_linear_mixture_fit_metrics.csv", layer, primary_var, "unconstrained_AB", "r2")
            for pair_id in bank.pair_trials["pair_id"].to_numpy(dtype=np.int64):
                idx = int(pair_id)
                for metric, values in (
                    ("dual_retention", dual),
                    ("pair_specificity", spec),
                    ("WPRI", wpri),
                    ("linear_mixture_r2", linear),
                    ("residual_pair_specificity", residual),
                ):
                    val = float(values[idx]) if idx < len(values) else float("nan")
                    rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "pair_id": int(pair_id),
                            "layer": layer,
                            "delay2_ms": int(delay2_ms),
                            "state_variable": primary_var,
                            "metric": metric,
                            "value": float(val * scale),
                        }
                    )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "supp_delay_layer_fused_state_metrics.csv")
    ctx.completed_modules["supplement"] = True


def _write_layerwise_morphology_metrics(ctx: ExperimentContext) -> None:
    wpri_path = ctx.metrics_dir / "panel_d_pair_level_organization_metrics.csv"
    residual_path = ctx.metrics_dir / "panel_d_linear_residual_pair_specificity_metrics.csv"
    mixture_path = ctx.metrics_dir / "panel_d_linear_mixture_fit_metrics.csv"
    frames: list[pd.DataFrame] = []
    if wpri_path.exists():
        wpri = pd.read_csv(wpri_path)
        cols = ["network_seed", "pair_id", "layer", "state_variable", "WPRI"]
        frames.append(wpri[[col for col in cols if col in wpri.columns]].copy())
    if residual_path.exists():
        residual = pd.read_csv(residual_path)
        residual = residual.rename(columns={"beyond_linear_pair_index": "residual_true_minus_shuffled"})
        cols = ["network_seed", "pair_id", "layer", "state_variable", "residual_pair_specificity", "residual_true_minus_shuffled"]
        frames.append(residual[[col for col in cols if col in residual.columns]].copy())
    base_cols = ["network_seed", "pair_id", "layer", "state_variable"]
    if not frames:
        out = pd.DataFrame(columns=base_cols)
    else:
        out = frames[0]
        for frame in frames[1:]:
            out = out.merge(frame, on=base_cols, how="outer")
    if mixture_path.exists():
        mixture = pd.read_csv(mixture_path)
        if not mixture.empty and {"network_seed", "pair_id", "layer", "state_variable", "model_name", "r2"}.issubset(mixture.columns):
            idx = mixture.groupby(base_cols)["r2"].idxmax()
            best = mixture.loc[idx, base_cols + ["model_name", "r2"]].rename(
                columns={"model_name": "best_linear_model", "r2": "linear_mixture_r2"}
            )
            out = out.merge(best, on=base_cols, how="outer")
    for col in ["WPRI", "residual_pair_specificity", "residual_true_minus_shuffled", "linear_mixture_r2", "best_linear_model"]:
        if col not in out.columns:
            out[col] = np.nan if col != "best_linear_model" else ""
    out["primary_layer"] = str(ctx.cfg.primary_layer)
    out["primary_state_variable"] = str(ctx.cfg.primary_state_variable)
    columns = [
        "network_seed",
        "pair_id",
        "layer",
        "state_variable",
        "WPRI",
        "residual_pair_specificity",
        "residual_true_minus_shuffled",
        "linear_mixture_r2",
        "best_linear_model",
        "primary_layer",
        "primary_state_variable",
    ]
    _save_csv(ctx, out[[col for col in columns if col in out.columns]], ctx.metrics_dir / "supp_layerwise_morphology_metrics.csv")


def _capture_pair_batch(
    ctx: ExperimentContext,
    a_spikes: torch.Tensor,
    b_spikes: torch.Tensor,
    *,
    delay2_steps: int | None = None,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Mapping[str, Mapping[str, torch.Tensor]]]]:
    cfg = ctx.cfg
    n, _, channels, height, width = a_spikes.shape
    conditions = len(STATE_CONDITIONS)
    prepare_network_state(ctx.net, n * conditions, channels, height, width)
    zero = torch.zeros((n * conditions, channels, height, width), device=ctx.device)
    current_time = 0

    def expand_phase(phase: str, t: int) -> torch.Tensor:
        x = zero.clone()
        if phase == "A":
            x[n : 2 * n] = a_spikes[:, t, ...]
            x[3 * n : 4 * n] = a_spikes[:, t, ...]
        elif phase == "B":
            x[2 * n : 3 * n] = b_spikes[:, t, ...]
            x[3 * n : 4 * n] = b_spikes[:, t, ...]
        return x

    with torch.no_grad():
        for t in range(cfg.sample_steps):
            current_time = _step_network_once(ctx.net, expand_phase("A", t), current_time)
        for _ in range(cfg.delay1_steps):
            current_time = _step_network_once(ctx.net, zero, current_time)
        for t in range(cfg.second_item_steps):
            current_time = _step_network_once(ctx.net, expand_phase("B", t), current_time)
        for _ in range(cfg.delay2_steps if delay2_steps is None else int(delay2_steps)):
            current_time = _step_network_once(ctx.net, zero, current_time)
        snapshot = snapshot_ux_state(ctx.net, batch_size=n * conditions)
        boundary = snapshot_boundary_state(ctx.net)

    bank: dict[str, dict[str, dict[str, np.ndarray]]] = {cond: {layer: {} for layer in LAYER_KEYS} for cond in STATE_CONDITIONS}
    boundaries: dict[str, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    for cidx, cond in enumerate(STATE_CONDITIONS):
        sl = slice(cidx * n, (cidx + 1) * n)
        for layer in LAYER_KEYS:
            u = snapshot[layer]["u"][sl].astype(np.float32, copy=False)
            x = snapshot[layer]["x"][sl].astype(np.float32, copy=False)
            bank[cond][layer]["u"] = u
            bank[cond][layer]["x"] = x
            bank[cond][layer]["g"] = (u * x).astype(np.float32, copy=False)
        boundaries[cond] = _slice_boundary_state(boundary, sl)
    return bank, boundaries


def _step_network_once(net, input_t: torch.Tensor, current_time: int, *, stsp_mode: str = "dynamic", ping_drive: torch.Tensor | None = None) -> int:
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode, ping_drive=ping_drive)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s2p = net.pool2(s2.float())
    net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    return current_time + 1


def _fit_mixture_models(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, cv_folds: int, seed: int) -> dict[str, dict[str, Any]]:
    x_a = np.asarray(x_a, dtype=np.float64).reshape(-1)
    x_b = np.asarray(x_b, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    target_norm = float(np.linalg.norm(y))
    models = {
        "A_only": _linear_model_metrics(y, _fit_single(x_a, y)),
        "B_only": _linear_model_metrics(y, _fit_single(x_b, y)),
        "mean_AB": _fixed_model_metrics(y, 0.5 * x_a + 0.5 * x_b),
        "sum_AB": _fixed_model_metrics(y, x_a + x_b),
        "unconstrained_AB": _linear_model_metrics(y, _fit_two(x_a, x_b, y)),
        "convex_AB": _fixed_model_metrics(y, _convex_prediction(x_a, x_b, y)),
    }
    for name, metrics in models.items():
        metrics["target_norm"] = target_norm
        metrics["cv_r2"] = _cv_r2(name, x_a, x_b, y, cv_folds, seed)
        if name == "convex_AB":
            w_a = _convex_weight(x_a, x_b, y)
            metrics["convex_weight_A"] = w_a
            metrics["convex_weight_B"] = 1.0 - w_a
        if name == "unconstrained_AB":
            beta = _fit_two_coeffs(x_a, x_b, y)
            metrics["beta_A"] = beta[0]
            metrics["beta_B"] = beta[1]
            metrics["intercept"] = beta[2]
        if name == "A_only":
            beta, intercept = _fit_single_coeffs(x_a, y)
            metrics["beta_A"] = beta
            metrics["intercept"] = intercept
        if name == "B_only":
            beta, intercept = _fit_single_coeffs(x_b, y)
            metrics["beta_B"] = beta
            metrics["intercept"] = intercept
    return models


def _linear_model_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    return _fixed_model_metrics(y, pred)


def _fixed_model_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    residual = y - pred
    residual_norm = float(np.linalg.norm(residual))
    target_norm = float(np.linalg.norm(y))
    return {
        "prediction": pred,
        "r2": _r2(y, pred),
        "residual_norm": residual_norm,
        "target_norm": target_norm,
        "residual_norm_ratio": float(residual_norm / max(target_norm, 1e-12)),
    }


def _fit_single(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, intercept = _fit_single_coeffs(x, y)
    return beta * x + intercept


def _fit_single_coeffs(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mat = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(mat, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _fit_two(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta_a, beta_b, intercept = _fit_two_coeffs(x_a, x_b, y)
    return beta_a * x_a + beta_b * x_b + intercept


def _fit_two_coeffs(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mat = np.column_stack([x_a, x_b, np.ones_like(x_a)])
    coef, *_ = np.linalg.lstsq(mat, y, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def _convex_weight(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray) -> float:
    d = x_a - x_b
    denom = float(np.dot(d, d))
    if denom <= 1e-12:
        return 0.5
    return float(np.clip(np.dot(y - x_b, d) / denom, 0.0, 1.0))


def _convex_prediction(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = _convex_weight(x_a, x_b, y)
    return w * x_a + (1.0 - w) * x_b


def _cv_r2(model_name: str, x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, cv_folds: int, seed: int) -> float:
    n = len(y)
    k = max(2, min(int(cv_folds), n))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    scores = []
    for fold in np.array_split(perm, k):
        if len(fold) == 0 or len(fold) == n:
            continue
        train = np.setdiff1d(perm, fold, assume_unique=False)
        pred = _predict_model(model_name, x_a, x_b, y, train, fold)
        scores.append(_r2(y[fold], pred))
    return float(np.nanmean(scores)) if scores else float("nan")


def _predict_model(model_name: str, x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    if model_name == "A_only":
        beta, intercept = _fit_single_coeffs(x_a[train], y[train])
        return beta * x_a[test] + intercept
    if model_name == "B_only":
        beta, intercept = _fit_single_coeffs(x_b[train], y[train])
        return beta * x_b[test] + intercept
    if model_name == "mean_AB":
        return 0.5 * x_a[test] + 0.5 * x_b[test]
    if model_name == "sum_AB":
        return x_a[test] + x_b[test]
    if model_name == "convex_AB":
        w = _convex_weight(x_a[train], x_b[train], y[train])
        return w * x_a[test] + (1.0 - w) * x_b[test]
    beta_a, beta_b, intercept = _fit_two_coeffs(x_a[train], x_b[train], y[train])
    return beta_a * x_a[test] + beta_b * x_b[test] + intercept


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def _access_scores(bank: PairEpisodeStateBank, pair_id: int, condition: str) -> dict[str, float]:
    idx = int(pair_id)
    layer = "layer3"
    variable = "g"
    state = bank.get(condition, layer, variable)[idx : idx + 1]
    sim_a = float(_row_centered_cosine(state, bank.get("S_A", layer, variable)[idx : idx + 1])[0])
    sim_b = float(_row_centered_cosine(state, bank.get("S_B", layer, variable)[idx : idx + 1])[0])
    sim0 = float(_row_centered_cosine(state, bank.get("S0", layer, variable)[idx : idx + 1])[0])
    a = max(0.0, (sim_a - sim0 + 1.0) / 2.0)
    b = max(0.0, (sim_b - sim0 + 1.0) / 2.0)
    return {"A": float(np.clip(a, 0.0, 1.0)), "B": float(np.clip(b, 0.0, 1.0))}


def _prediction_from_scores(scores: Mapping[str, float], a_label: int, b_label: int, seed: int) -> int:
    threshold = 0.15
    if max(scores["A"], scores["B"]) < threshold:
        return -1
    if abs(scores["A"] - scores["B"]) < 1e-9:
        return a_label if int(seed) % 2 == 0 else b_label
    return a_label if scores["A"] > scores["B"] else b_label


def _partial_cue_metrics(network_seed: int, raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped_stats: dict[tuple[str, float, str], dict[str, float]] = {}
    for (target, keep, cond), part in raw_df.groupby(["target_item", "keep_prob", "state_condition"], sort=False):
        denom = max(1, len(part))
        grouped_stats[(str(target), float(keep), str(cond))] = {
            "P_target": float(part["pred_is_target"].sum() / denom),
            "P_other_pair_member": float(part["pred_is_other_pair_member"].sum() / denom),
            "P_silent": float(part["silent"].sum() / denom),
        }
    for keys, part in raw_df.groupby(["state_condition", "target_item", "keep_prob"], sort=False):
        cond, target, keep = str(keys[0]), str(keys[1]), float(keys[2])
        denom = max(1, len(part))
        relevant = "S_A" if target == "A" else "S_B"
        irrelevant = "S_B" if target == "A" else "S_A"
        p = grouped_stats
        p_cond = p.get((target, keep, cond), {})
        p_sab = p.get((target, keep, "S_AB"), {})
        p_s0 = p.get((target, keep, "S0"), {})
        p_rel = p.get((target, keep, relevant), {})
        p_irrel = p.get((target, keep, irrelevant), {})
        rows.append(
            {
                "network_seed": int(network_seed),
                "state_condition": cond,
                "target_item": target,
                "keep_prob": keep,
                "P_target": float(part["pred_is_target"].sum() / denom),
                "P_A": float(part["pred_is_A"].sum() / denom),
                "P_B": float(part["pred_is_B"].sum() / denom),
                "P_pair_member": float(part["pred_is_pair_member"].sum() / denom),
                "P_other_pair_member": float(part["pred_is_other_pair_member"].sum() / denom),
                "P_other_class": float(part["pred_is_other_class"].sum() / denom),
                "P_silent": float(part["silent"].sum() / denom),
                "P_relevant_single_target": float(p_rel.get("P_target", np.nan)),
                "P_irrelevant_single_target": float(p_irrel.get("P_target", np.nan)),
                "target_recovery_gain_vs_S0": float(p_sab.get("P_target", np.nan) - p_s0.get("P_target", np.nan)),
                "target_recovery_gain_vs_relevant_single": float(p_sab.get("P_target", np.nan) - p_rel.get("P_target", np.nan)),
                "target_recovery_gain_vs_irrelevant_single": float(p_sab.get("P_target", np.nan) - p_irrel.get("P_target", np.nan)),
                "other_pair_intrusion_change_vs_relevant_single": float(
                    p_sab.get("P_other_pair_member", np.nan) - p_rel.get("P_other_pair_member", np.nan)
                ),
                "silent_reduction_vs_S0": float(p_s0.get("P_silent", np.nan) - p_sab.get("P_silent", np.nan)),
                "relevant_single_condition": relevant,
                "irrelevant_single_condition": irrelevant,
                "weak_probe_metric_mode": _mode_value(part, "weak_probe_metric_mode", "fig4_compat"),
                "weak_probe_mask_space": _mode_value(part, "mask_space", ""),
                "mean_first_fire_time_ms": float(pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan).mean()),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def _ping_sweep_metrics(network_seed: int, trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in trial_df.groupby(["sweep_type", "ping_amp", "ping_ms", "state_condition"], sort=False):
        sweep_type, ping_amp, ping_ms, condition = str(keys[0]), float(keys[1]), int(keys[2]), str(keys[3])
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "sweep_type": sweep_type,
                "ping_amp": ping_amp,
                "ping_ms": ping_ms,
                "state_condition": condition,
                "pair_member_readout_rate": float(part["pred_is_pair_member"].sum() / denom),
                "A_readout_rate": float(part["pred_is_A"].sum() / denom),
                "B_readout_rate": float(part["pred_is_B"].sum() / denom),
                "other_readout_rate": float(part["pred_is_other"].sum() / denom),
                "silent_rate": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "network_seed",
            "sweep_type",
            "ping_amp",
            "ping_ms",
            "state_condition",
            "pair_member_readout_rate",
            "A_readout_rate",
            "B_readout_rate",
            "other_readout_rate",
            "silent_rate",
            "n_trials",
        ],
    )


def _completion_delay_sweep_metrics(network_seed: int, trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in trial_df.groupby(["delay2_ms", "state_condition", "keep_prob"], sort=False):
        delay2_ms, condition, keep_prob = int(keys[0]), str(keys[1]), float(keys[2])
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "delay2_ms": delay2_ms,
                "state_condition": condition,
                "keep_prob": keep_prob,
                "target_recovery_rate": float(part["correct_target"].sum() / denom),
                "A_readout_rate": float(part["pred_is_A"].sum() / denom),
                "B_readout_rate": float(part["pred_is_B"].sum() / denom),
                "other_readout_rate": float(part["pred_is_other"].sum() / denom),
                "silent_rate": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "network_seed",
            "delay2_ms",
            "state_condition",
            "keep_prob",
            "target_recovery_rate",
            "A_readout_rate",
            "B_readout_rate",
            "other_readout_rate",
            "silent_rate",
            "n_trials",
        ],
    )


def _completion_delay_sweep_contrast(network_seed: int, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in metrics.groupby(["delay2_ms", "keep_prob"], sort=True):
        delay2_ms, keep_prob = int(keys[0]), float(keys[1])
        by_cond = {str(row["state_condition"]): row for row in part.to_dict("records")}
        recovery_sab = _maybe_float(by_cond.get("S_AB", {}).get("target_recovery_rate"))
        recovery_sb = _maybe_float(by_cond.get("S_B", {}).get("target_recovery_rate"))
        recovery_s0 = _maybe_float(by_cond.get("S0", {}).get("target_recovery_rate"))
        rows.append(
            {
                "network_seed": int(network_seed),
                "delay2_ms": delay2_ms,
                "keep_prob": keep_prob,
                "recovery_SAB": recovery_sab,
                "recovery_SB": recovery_sb,
                "recovery_S0": recovery_s0,
                "completion_gain_SAB_minus_SB": _nan_diff(recovery_sab, recovery_sb),
                "completion_gain_SAB_minus_S0": _nan_diff(recovery_sab, recovery_s0),
                "n_trials_SAB": _maybe_int(by_cond.get("S_AB", {}).get("n_trials")),
                "n_trials_SB": _maybe_int(by_cond.get("S_B", {}).get("n_trials")),
                "n_trials_S0": _maybe_int(by_cond.get("S0", {}).get("n_trials")),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "network_seed",
            "delay2_ms",
            "keep_prob",
            "recovery_SAB",
            "recovery_SB",
            "recovery_S0",
            "completion_gain_SAB_minus_SB",
            "completion_gain_SAB_minus_S0",
            "n_trials_SAB",
            "n_trials_SB",
            "n_trials_S0",
        ],
    )


def _stable_sweep_seed(network_seed: int, sweep_type: str, ping_amp: float, ping_ms: int, condition: str) -> int:
    token = f"{int(network_seed)}|{sweep_type}|{float(ping_amp):.6f}|{int(ping_ms)}|{condition}"
    value = 2166136261
    for ch in token:
        value = (value ^ ord(ch)) * 16777619
        value &= 0x7FFFFFFF
    return int(value)


def _partial_cue_auc_metrics(network_seed: int, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_item, target_part in metrics.groupby("target_item", sort=False):
        auc_by_condition: dict[str, float] = {}
        p50_by_condition: dict[str, float] = {}
        legacy_threshold_by_condition: dict[str, float] = {}
        for condition, part in target_part.groupby("state_condition", sort=False):
            ordered = part.sort_values("keep_prob")
            x = ordered["keep_prob"].to_numpy(dtype=float)
            y = ordered["P_target"].to_numpy(dtype=float)
            auc = _normalized_auc(x, y)
            auc_by_condition[str(condition)] = auc
            p50_by_condition[str(condition)] = _p50_from_curve(x, y, threshold=0.5)
            threshold_rows = ordered[ordered["P_target"] >= 0.5]
            legacy_threshold_by_condition[str(condition)] = float(threshold_rows["keep_prob"].iloc[0]) if not threshold_rows.empty else float("nan")
        for condition, part in target_part.groupby("state_condition", sort=False):
            ordered = part.sort_values("keep_prob")
            auc = auc_by_condition[str(condition)]
            relevant = "S_A" if target_item == "A" else "S_B"
            irrelevant = "S_B" if target_item == "A" else "S_A"
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "state_condition": str(condition),
                    "target_item": str(target_item),
                    "normalized_auc_target_recovery": auc,
                    "p50_target_recovery_keep_prob": p50_by_condition[str(condition)],
                    "legacy_threshold_keep_prob": legacy_threshold_by_condition[str(condition)],
                    "SAB_vs_S0_auc_gain": float(auc_by_condition.get("S_AB", 0.0) - auc_by_condition.get("S0", 0.0)),
                    "SAB_vs_relevant_single_auc_gain": float(auc_by_condition.get("S_AB", 0.0) - auc_by_condition.get(relevant, 0.0)),
                    "SAB_vs_irrelevant_single_auc_gain": float(auc_by_condition.get("S_AB", 0.0) - auc_by_condition.get(irrelevant, 0.0)),
                    "SAB_vs_relevant_single_upper_bound_gap": float(auc_by_condition.get(relevant, 0.0) - auc_by_condition.get("S_AB", 0.0)),
                    "SAB_vs_relevant_single_p50_shift": _nan_diff(p50_by_condition.get("S_AB"), p50_by_condition.get(relevant)),
                    "SAB_vs_S0_p50_shift": _nan_diff(p50_by_condition.get("S_AB"), p50_by_condition.get("S0")),
                    "low_cue_gain": _cue_gain(target_part, target_item, max_keep=0.1),
                    "mid_cue_gain": _cue_gain(target_part, target_item, min_keep=0.1, max_keep=0.3),
                    "high_cue_gain": _cue_gain(target_part, target_item, min_keep=0.3),
                    "weak_probe_metric_mode": _mode_value(part, "weak_probe_metric_mode", "fig4_compat"),
                    "weak_probe_mask_space": _mode_value(part, "weak_probe_mask_space", ""),
                    "n_trials": int(part["n_trials"].sum()) if "n_trials" in part.columns else int(len(part)),
                }
            )
    return pd.DataFrame(rows)


def _partial_cue_pair_metrics(network_seed: int, raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in raw_df.groupby(["pair_id", "target_item", "state_condition", "keep_prob"], sort=False):
        pair_id, target, cond, keep = int(keys[0]), str(keys[1]), str(keys[2]), float(keys[3])
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "pair_id": pair_id,
                "target_item": target,
                "state_condition": cond,
                "keep_prob": keep,
                "P_target": float(part["pred_is_target"].sum() / denom),
                "P_A": float(part["pred_is_A"].sum() / denom),
                "P_B": float(part["pred_is_B"].sum() / denom),
                "P_pair_member": float(part["pred_is_pair_member"].sum() / denom),
                "P_other_pair_member": float(part["pred_is_other_pair_member"].sum() / denom),
                "P_other_class": float(part["pred_is_other_class"].sum() / denom),
                "P_silent": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def _normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return float("nan")
    if len(x) == 1:
        return float(y.mean())
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])
    if span <= 0.0:
        return float(np.nanmean(y))
    return float(np.trapezoid(y, x) / span)


def _p50_from_curve(x: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return float("nan")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.any(y >= float(threshold)):
        first = int(np.argmax(y >= float(threshold)))
        if first == 0:
            return float(x[0])
        x0, x1 = float(x[first - 1]), float(x[first])
        y0, y1 = float(y[first - 1]), float(y[first])
        if abs(y1 - y0) <= 1e-12:
            return x1
        frac = (float(threshold) - y0) / (y1 - y0)
        return float(x0 + frac * (x1 - x0))
    return float("nan")


def _nan_diff(a: Any, b: Any) -> float:
    aa = float(a) if a is not None else float("nan")
    bb = float(b) if b is not None else float("nan")
    return float(aa - bb) if math.isfinite(aa) and math.isfinite(bb) else float("nan")


def _mode_value(part: pd.DataFrame, column: str, default: str) -> str:
    if column not in part.columns or part.empty:
        return str(default)
    values = part[column].dropna().astype(str).unique()
    return str(values[0]) if len(values) else str(default)


def _compat_fig4_weak_probe_outputs(
    network_seed: int,
    metrics: pd.DataFrame,
    auc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for _, row in metrics.iterrows():
        target = str(row["target_item"])
        summary_rows.append(
            {
                "network_seed": int(network_seed),
                "target_item": target,
                "fig4_mapping": "A_completion" if target == "A" else "mirrored_B_completion",
                "state_condition": str(row["state_condition"]),
                "baseline_condition": "S0",
                "relevant_single_condition": "S_A" if target == "A" else "S_B",
                "irrelevant_single_condition": "S_B" if target == "A" else "S_A",
                "keep_prob": float(row["keep_prob"]),
                "pred_A": float(row["P_A"]),
                "pred_B": float(row["P_B"]),
                "pred_other": float(row["P_other_class"]),
                "pred_silent": float(row["P_silent"]),
                "P_target": float(row["P_target"]),
                "P_pair": float(row["P_pair_member"]),
                "target_recovery_gain_vs_S0": float(row.get("target_recovery_gain_vs_S0", np.nan)),
                "target_recovery_gain_vs_relevant_single": float(row.get("target_recovery_gain_vs_relevant_single", np.nan)),
                "weak_probe_metric_mode": str(row.get("weak_probe_metric_mode", "")),
                "weak_probe_mask_space": str(row.get("weak_probe_mask_space", "")),
            }
        )
    auc_rows = []
    threshold_rows = []
    for _, row in auc.iterrows():
        base = {
            "network_seed": int(network_seed),
            "target_item": str(row["target_item"]),
            "fig4_mapping": "A_completion" if str(row["target_item"]) == "A" else "mirrored_B_completion",
            "state_condition": str(row["state_condition"]),
            "weak_probe_metric_mode": str(row.get("weak_probe_metric_mode", "")),
            "weak_probe_mask_space": str(row.get("weak_probe_mask_space", "")),
        }
        auc_rows.append(
            {
                **base,
                "normalized_auc_target_recovery": float(row.get("normalized_auc_target_recovery", np.nan)),
                "SAB_vs_S0_auc_gain": float(row.get("SAB_vs_S0_auc_gain", np.nan)),
                "SAB_vs_relevant_single_auc_gain": float(row.get("SAB_vs_relevant_single_auc_gain", np.nan)),
                "SAB_vs_irrelevant_single_auc_gain": float(row.get("SAB_vs_irrelevant_single_auc_gain", np.nan)),
                "low_cue_gain": float(row.get("low_cue_gain", np.nan)),
                "mid_cue_gain": float(row.get("mid_cue_gain", np.nan)),
                "high_cue_gain": float(row.get("high_cue_gain", np.nan)),
            }
        )
        threshold_rows.append(
            {
                **base,
                "p50_target_recovery_keep_prob": float(row.get("p50_target_recovery_keep_prob", np.nan)),
                "SAB_vs_relevant_single_p50_shift": float(row.get("SAB_vs_relevant_single_p50_shift", np.nan)),
                "SAB_vs_S0_p50_shift": float(row.get("SAB_vs_S0_p50_shift", np.nan)),
                "legacy_threshold_keep_prob": float(row.get("legacy_threshold_keep_prob", np.nan)),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(auc_rows), pd.DataFrame(threshold_rows)


def _cue_gain(target_part: pd.DataFrame, target_item: str, *, min_keep: float = -np.inf, max_keep: float = np.inf) -> float:
    _ = target_item
    part = target_part[(pd.to_numeric(target_part["keep_prob"], errors="coerce") > float(min_keep)) & (pd.to_numeric(target_part["keep_prob"], errors="coerce") <= float(max_keep))]
    if part.empty:
        return float("nan")
    pivot = part.pivot_table(index="keep_prob", columns="state_condition", values="P_target", aggfunc="mean")
    if not {"S_AB", "S0"}.issubset(pivot.columns):
        return float("nan")
    return float((pivot["S_AB"] - pivot["S0"]).mean())


def run_ping_readout_from_boundary(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    ping_seed: int,
    ping_amp: float | None = None,
    ping_steps: int | None = None,
    record_trace: bool = False,
) -> FunctionalReadout:
    batch_size = int(next(iter(next(iter(boundary.values())).values())).shape[0])
    restore_condition_state_for_functional_readout(ctx, boundary, batch_size)
    input_shape = _layer_input_shapes_for_batch(boundary, batch_size)["layer1"]
    zero_input = torch.zeros(input_shape, dtype=torch.float32, device=ctx.device)
    gen = torch.Generator(device=ctx.device)
    gen.manual_seed(int(ping_seed))
    traces: dict[str, list[torch.Tensor]] = {"layer3_spikes": []}
    amp = float(ctx.cfg.ping_amp if ping_amp is None else ping_amp)
    steps = int(ctx.cfg.ping_steps if ping_steps is None else ping_steps)
    with torch.no_grad():
        for t_idx in range(steps):
            if ctx.cfg.ping_mode == "bernoulli_drive":
                ping_drive = (torch.rand(zero_input.shape, generator=gen, device=ctx.device) < amp).to(torch.float32)
            else:
                ping_drive = torch.full_like(zero_input, amp)
            if float(ctx.cfg.ping_noise) > 0.0:
                ping_drive = torch.clamp(ping_drive + torch.randn(ping_drive.shape, generator=gen, device=ctx.device) * float(ctx.cfg.ping_noise), min=0.0)
            s3 = _forward_three_layers_with_optional_trace(ctx.net, zero_input, t_idx, ping_drive=ping_drive)
            if record_trace:
                traces["layer3_spikes"].append(s3.detach().to(torch.float32).cpu())
    pred, fire_t = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size=batch_size)
    trace = _pack_trace(traces) if record_trace else None
    return FunctionalReadout(
        prediction=pred.numpy().astype(np.int64, copy=False),
        first_fire_time_ms=fire_t.numpy().astype(np.float32, copy=False) * float(ctx.cfg.dt / ms),
        silent=(pred.numpy() < 0),
        readout_margin_A=_readout_margin_for_class(ctx, 0),
        readout_margin_B=_readout_margin_for_class(ctx, 1),
        trace=trace,
    )


def run_probe_readout_from_boundary(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    probe_spikes: torch.Tensor,
    *,
    probe_scale: float = 1.0,
    probe_noise: float = 0.0,
    seed: int = 0,
    record_trace: bool = False,
) -> FunctionalReadout:
    batch_size = int(probe_spikes.shape[0])
    restore_condition_state_for_functional_readout(ctx, boundary, batch_size)
    gen = torch.Generator(device=ctx.device)
    gen.manual_seed(int(seed))
    traces: dict[str, list[torch.Tensor]] = {"layer3_spikes": []}
    with torch.no_grad():
        for t_idx in range(int(probe_spikes.shape[1])):
            input_t = probe_spikes[:, t_idx].to(ctx.device, dtype=torch.float32) * float(probe_scale)
            if float(probe_noise) > 0.0:
                input_t = torch.clamp(
                    input_t + torch.randn(input_t.shape, generator=gen, device=ctx.device) * float(probe_noise),
                    min=0.0,
                )
            s3 = _forward_three_layers_with_optional_trace(ctx.net, input_t, t_idx)
            if record_trace:
                traces["layer3_spikes"].append(s3.detach().to(torch.float32).cpu())
    pred, fire_t = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size=batch_size)
    trace = _pack_trace(traces) if record_trace else None
    return FunctionalReadout(
        prediction=pred.numpy().astype(np.int64, copy=False),
        first_fire_time_ms=fire_t.numpy().astype(np.float32, copy=False) * float(ctx.cfg.dt / ms),
        silent=(pred.numpy() < 0),
        readout_margin_A=_readout_margin_for_class(ctx, 0),
        readout_margin_B=_readout_margin_for_class(ctx, 1),
        trace=trace,
    )


def restore_condition_state_for_functional_readout(ctx: ExperimentContext, boundary: Mapping[str, Mapping[str, torch.Tensor]], batch_size: int) -> dict[str, object]:
    layer_input_shapes = _layer_input_shapes_for_batch(boundary, int(batch_size))
    restore_ux = boundary_state_to_restore_ux_by_layer(boundary, ctx.device)
    info = reset_all_state_restore_selected_stsp_in_place(ctx.net, layer_input_shapes, restore_ux)
    ctx.net.layer3.reset_decision_state()
    if hasattr(ctx.net.layer3, "lateral_inh"):
        ctx.net.layer3.lateral_inh.reset_state(ctx.net.layer3.output_shape)
    return info


def boundary_state_to_restore_ux_by_layer(boundary: Mapping[str, Mapping[str, torch.Tensor]], device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_key, state in boundary.items():
        if "u" in state and "x" in state:
            out[layer_key] = (state["u"].to(device), state["x"].to(device))
    return out


def slice_boundary_state(boundary_state: Mapping[str, Mapping[str, torch.Tensor]], row_indices: Sequence[int], device: torch.device | None = None) -> dict[str, dict[str, torch.Tensor]]:
    idx = torch.as_tensor(list(row_indices), dtype=torch.long)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key, state in boundary_state.items():
        out[layer_key] = {}
        for key, value in state.items():
            selected = value.index_select(0, idx).detach().clone()
            out[layer_key][key] = selected.to(device) if device is not None else selected
    return out


def concat_condition_boundaries(
    boundary_states_by_condition: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    conditions: Sequence[str],
    row_indices: Sequence[int],
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    sliced = [slice_boundary_state(boundary_states_by_condition[condition], row_indices, device) for condition in conditions]
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key in sliced[0]:
        out[layer_key] = {}
        for key in sliced[0][layer_key]:
            out[layer_key][key] = torch.cat([part[layer_key][key] for part in sliced], dim=0)
    return out


def _forward_three_layers_with_optional_trace(net, input_t: torch.Tensor, t_step: int, *, ping_drive: torch.Tensor | None = None) -> torch.Tensor:
    s1, _ = net.layer1.forward_step(input_t, t_step, training=False, monitor=False, stsp_mode="dynamic", ping_drive=ping_drive)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    s2p = net.pool2(s2.float())
    s3, _ = net.layer3.forward_step(s2p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    return s3


def _layer_input_shapes_from_boundary(boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, tuple[int, ...]]:
    return {layer_key: tuple(state["u"].shape) for layer_key, state in boundary.items() if "u" in state}


def _layer_input_shapes_for_batch(boundary: Mapping[str, Mapping[str, torch.Tensor]], batch_size: int) -> dict[str, tuple[int, ...]]:
    shapes = _layer_input_shapes_from_boundary(boundary)
    return {layer_key: (int(batch_size),) + tuple(shape[1:]) for layer_key, shape in shapes.items()}


def _pack_trace(traces: Mapping[str, list[torch.Tensor]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, values in traces.items():
        if values:
            out[key] = torch.stack(values, dim=0).numpy().astype(np.float32, copy=False)
    return out


def _readout_margin_for_class(ctx: ExperimentContext, class_id: int) -> np.ndarray:
    firing = ctx.net.layer3.firing_times.detach().cpu()
    batch_size = firing.shape[0]
    grouped = firing.view(batch_size, ctx.net.layer3.num_classes, ctx.net.layer3.neurons_per_class, -1).reshape(batch_size, ctx.net.layer3.num_classes, -1)
    class_min = grouped.min(dim=2).values
    target = class_min[:, int(class_id)].numpy().astype(np.float32, copy=False)
    others = class_min.clone()
    others[:, int(class_id)] = float("inf")
    other_min = others.min(dim=1).values.numpy().astype(np.float32, copy=False)
    margin = other_min - target
    margin[~np.isfinite(margin)] = np.nan
    return margin


def _readout_margin_value(values: np.ndarray | None, idx: int) -> float:
    if values is None or idx >= len(values):
        return float("nan")
    return float(values[idx])


def _ping_spike_count(ctx: ExperimentContext, ping_seed: int) -> float:
    shape = tuple(ctx.net.layer1.output_shape)
    if ctx.cfg.ping_mode == "bernoulli_drive":
        gen = torch.Generator(device=ctx.device)
        gen.manual_seed(int(ping_seed))
        return float(sum((torch.rand(shape, generator=gen, device=ctx.device) < float(ctx.cfg.ping_amp)).sum().item() for _ in range(int(ctx.cfg.ping_steps))))
    return float(np.prod(shape) * int(ctx.cfg.ping_steps)) if float(ctx.cfg.ping_amp) > 0.0 else 0.0


def _ping_energy(ctx: ExperimentContext, ping_seed: int) -> float:
    _ = ping_seed
    shape = tuple(ctx.net.layer1.output_shape)
    if ctx.cfg.ping_mode == "bernoulli_drive":
        return float(_ping_spike_count(ctx, ping_seed))
    return float(np.prod(shape) * float(ctx.cfg.ping_amp) * int(ctx.cfg.ping_steps))


def _neutral_ping_metrics(network_seed: int, trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lookup: dict[str, dict[str, float]] = {}
    for condition, part in trial_df.groupby("state_condition", sort=False):
        denom = max(1, len(part))
        row = {
            "network_seed": int(network_seed),
            "state_condition": condition,
            "P_A": float(part["pred_is_A"].sum() / denom),
            "P_B": float(part["pred_is_B"].sum() / denom),
            "P_pair": float(part["pred_is_pair_member"].sum() / denom),
            "P_other": float(part["pred_is_other"].sum() / denom),
            "P_silent": float(part["silent"].sum() / denom),
            "P_A_minus_B": float((part["pred_is_A"].sum() - part["pred_is_B"].sum()) / denom),
            "pair_access_gain_SAB_vs_S0": 0.0,
            "old_item_rescue_SAB_vs_SB": 0.0,
            "new_item_rescue_SAB_vs_SA": 0.0,
            "dual_access_balance": 0.0,
            "n_trials": int(len(part)),
        }
        lookup[str(condition)] = row
        rows.append(row)
    p_a_sab = lookup.get("S_AB", {}).get("P_A", 0.0)
    p_b_sab = lookup.get("S_AB", {}).get("P_B", 0.0)
    for row in rows:
        row["pair_access_gain_SAB_vs_S0"] = float(lookup.get("S_AB", {}).get("P_pair", 0.0) - lookup.get("S0", {}).get("P_pair", 0.0))
        row["old_item_rescue_SAB_vs_SB"] = float(p_a_sab - lookup.get("S_B", {}).get("P_A", 0.0))
        row["new_item_rescue_SAB_vs_SA"] = float(p_b_sab - lookup.get("S_A", {}).get("P_B", 0.0))
        row["dual_access_balance"] = float(min(p_a_sab, p_b_sab))
    return pd.DataFrame(rows)


def write_functional_proxy_diagnostics(ctx: ExperimentContext, bank: PairEpisodeStateBank) -> None:
    real_path = ctx.raw_dir / "panel_e_neutral_ping_trial_readout.csv"
    real = pd.read_csv(real_path) if real_path.exists() else pd.DataFrame()
    rows = []
    for _, rec in bank.pair_trials.iterrows():
        pair_id = int(rec["pair_id"])
        for condition in STATE_CONDITIONS:
            scores = _access_scores(bank, pair_id, condition)
            match = real[(real["pair_id"].eq(pair_id)) & (real["state_condition"].eq(condition))].head(1) if not real.empty else pd.DataFrame()
            real_a = int(match["pred_is_A"].iloc[0]) if not match.empty else 0
            real_b = int(match["pred_is_B"].iloc[0]) if not match.empty else 0
            real_pair = int(match["pred_is_pair_member"].iloc[0]) if not match.empty else 0
            proxy_pred_a = int(scores["A"] >= scores["B"] and max(scores.values()) >= 0.15)
            proxy_pred_b = int(scores["B"] > scores["A"] and max(scores.values()) >= 0.15)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "pair_id": pair_id,
                    "state_condition": condition,
                    "proxy_score_A": float(scores["A"]),
                    "proxy_score_B": float(scores["B"]),
                    "real_pred_is_A": real_a,
                    "real_pred_is_B": real_b,
                    "real_pred_is_pair_member": real_pair,
                    "proxy_real_agreement": int((proxy_pred_a == real_a) and (proxy_pred_b == real_b)),
                }
            )
    _save_csv(ctx, pd.DataFrame(rows), ctx.metrics_dir / "supp_functional_proxy_diagnostics.csv")
    ctx.completed_modules["proxy_functional_debug"] = True


def _metric_lookup(path: Path, layer: str, variable: str, metric_col: str) -> np.ndarray:
    if not path.exists():
        return np.asarray([], dtype=float)
    df = pd.read_csv(path)
    part = df[(df["layer"].astype(str) == layer) & (df["state_variable"].astype(str) == variable)].sort_values("pair_id")
    return part[metric_col].to_numpy(dtype=float) if metric_col in part.columns else np.asarray([], dtype=float)


def _linear_metric_lookup(path: Path, layer: str, variable: str, model_name: str, metric_col: str) -> np.ndarray:
    if not path.exists():
        return np.asarray([], dtype=float)
    df = pd.read_csv(path)
    part = df[(df["layer"].astype(str) == layer) & (df["state_variable"].astype(str) == variable) & (df["model_name"].astype(str) == model_name)].sort_values("pair_id")
    return part[metric_col].to_numpy(dtype=float) if metric_col in part.columns else np.asarray([], dtype=float)


def save_debug_figures(ctx: ExperimentContext) -> None:
    apply_publication_style()
    jobs = [
        ("panel_b_dual_retention_metrics.csv", "fusion_dual_score", "fig2_debug_dual_retention"),
        ("panel_c_pair_specificity_metrics.csv", "true_minus_shuffled", "fig2_debug_pair_specificity"),
        ("panel_d_pair_level_organization_metrics.csv", "WPRI", "fig2_debug_wpri"),
        ("panel_d_linear_residual_pair_specificity_metrics.csv", "residual_pair_specificity", "fig2_debug_linear_residual"),
        ("panel_e_neutral_ping_metrics.csv", "P_pair", "fig2_debug_real_neutral_ping"),
        ("panel_f_partial_cue_metrics.csv", "P_target", "fig2_debug_real_partial_cue"),
    ]
    for filename, column, stem in jobs:
        path = ctx.metrics_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        part = df[(df.get("layer", "") == "layer3") & (df.get("state_variable", "") == "g")] if "layer" in df.columns else df
        if column not in part.columns or part.empty:
            continue
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
        ax.hist(pd.to_numeric(part[column], errors="coerce").dropna(), bins=20, color="#4C78A8", alpha=0.8)
        ax.set_title(stem)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        save_figure_all_formats(fig, ctx.debug_dir / stem)
        plt.close(fig)
    ping_path = ctx.metrics_dir / "supp_ping_sweep_metrics.csv"
    if ping_path.exists():
        import matplotlib.pyplot as plt

        ping_df = pd.read_csv(ping_path)
        for sweep_type, x_col, stem in (
            ("amplitude", "ping_amp", "supp_ping_amp_sweep_pair_member_readout"),
            ("duration", "ping_ms", "supp_ping_ms_sweep_pair_member_readout"),
        ):
            part = ping_df[ping_df["sweep_type"].astype(str).eq(sweep_type)] if "sweep_type" in ping_df.columns else pd.DataFrame()
            if part.empty or not {x_col, "state_condition", "pair_member_readout_rate"}.issubset(part.columns):
                continue
            fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
            for condition, cond_part in part.groupby("state_condition", sort=True):
                ordered = cond_part.sort_values(x_col)
                ax.plot(ordered[x_col], ordered["pair_member_readout_rate"], marker="o", label=str(condition))
            ax.set_xlabel(x_col)
            ax.set_ylabel("pair_member_readout_rate")
            ax.legend(frameon=False, fontsize=7)
            save_figure_all_formats(fig, ctx.debug_dir / stem)
            plt.close(fig)
    completion_path = ctx.metrics_dir / "supp_completion_delay_sweep_contrast.csv"
    if completion_path.exists():
        import matplotlib.pyplot as plt

        comp_df = pd.read_csv(completion_path)
        if not comp_df.empty and {"delay2_ms", "completion_gain_SAB_minus_SB"}.issubset(comp_df.columns):
            fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
            ordered = comp_df.sort_values("delay2_ms")
            ax.plot(ordered["delay2_ms"], ordered["completion_gain_SAB_minus_SB"], marker="o")
            ax.axhline(0.0, color="0.5", linewidth=0.8)
            ax.set_xlabel("delay2_ms")
            ax.set_ylabel("completion_gain_SAB_minus_SB")
            save_figure_all_formats(fig, ctx.debug_dir / "supp_completion_delay_gain")
            plt.close(fig)
    ctx.completed_modules["debug_figures"] = True


def _pair_sampling_audit(network_seed: int, pairs: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("pixel_similarity", "foreground_overlap"):
        values = pd.to_numeric(pairs[col], errors="coerce")
        rows.append({"network_seed": int(network_seed), "audit_type": f"{col}_summary", "label": "mean", "count": int(values.count()), "value": float(values.mean())})
        rows.append({"network_seed": int(network_seed), "audit_type": f"{col}_summary", "label": "std", "count": int(values.count()), "value": float(values.std(ddof=1) if values.count() > 1 else 0.0)})
    for class_pair, count in pairs["class_pair"].value_counts().sort_index().items():
        rows.append({"network_seed": int(network_seed), "audit_type": "class_pair_count", "label": str(class_pair), "count": int(count), "value": float(count)})
    rows.append({"network_seed": int(network_seed), "audit_type": "candidate_pool", "label": "eligible", "count": int(pool.get("eligible", pd.Series(dtype=int)).sum() if not pool.empty else 0), "value": float(len(pool))})
    return pd.DataFrame(rows)


def _trial_condition_audit(network_seed: int, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = [{"network_seed": int(network_seed), "audit_type": "n_pairs", "label": "all", "count": int(len(pairs)), "value": float(len(pairs))}]
    same = int((pairs["A_label"] == pairs["B_label"]).sum())
    rows.append({"network_seed": int(network_seed), "audit_type": "same_label_pairs", "label": "A_label_eq_B_label", "count": same, "value": float(same / max(1, len(pairs)))})
    for col in ("A_label", "B_label"):
        for label, count in pairs[col].value_counts().sort_index().items():
            rows.append({"network_seed": int(network_seed), "audit_type": f"{col}_count", "label": int(label), "count": int(count), "value": float(count)})
    return pd.DataFrame(rows)


def _image_similarity_and_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    sim = _centered_cosine(a, b)
    fa = a > 0.1
    fb = b > 0.1
    union = np.logical_or(fa, fb).sum()
    overlap = float(np.logical_and(fa, fb).sum() / max(1, union))
    return float(sim), overlap


def _selection_bin(sim: float) -> str:
    if sim < 0.25:
        return "low_similarity"
    if sim < 0.55:
        return "mid_similarity"
    return "high_similarity"


def _row_centered_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = a.astype(np.float64, copy=False) - a.astype(np.float64, copy=False).mean(axis=1, keepdims=True)
    b2 = b.astype(np.float64, copy=False) - b.astype(np.float64, copy=False).mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1)
    return np.sum(a2 * b2, axis=1) / np.maximum(denom, 1e-12)


def _centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), 1e-12))


def _slice_boundary_state(boundary: Mapping[str, Mapping[str, torch.Tensor]], sl: slice) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key, state in boundary.items():
        out[layer_key] = {key: value[sl].detach().cpu().clone() for key, value in state.items()}
    return out


def _concat_boundary_states(a: Mapping[str, Mapping[str, torch.Tensor]], b: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key in a:
        out[layer_key] = {}
        for key in a[layer_key]:
            out[layer_key][key] = torch.cat([a[layer_key][key], b[layer_key][key]], dim=0)
    return out


def _images_for_ids(dataset, image_ids: Iterable[int]) -> torch.Tensor:
    return torch.stack([dataset[int(idx)][0].detach().to(torch.float32) for idx in image_ids], dim=0)


def _encode_cached(ctx: ExperimentContext, image_ids: Iterable[int], steps: int, *, cache: dict[tuple[Any, ...], torch.Tensor]) -> torch.Tensor:
    ids = tuple(int(v) for v in image_ids)
    key = (ids, int(steps), str(ctx.device))
    if (not ctx.cfg.use_encode_cache) or key not in cache:
        images = _images_for_ids(ctx.dataset, ids).to(ctx.device)
        spikes = encode_images(ctx.encoder, images, int(steps))
        if not ctx.cfg.use_encode_cache:
            return spikes
        cache[key] = spikes
    return cache[key]


def _iter_batches(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), int(batch_size)):
        yield df.iloc[start : start + int(batch_size)].reset_index(drop=True)


def _write_config_files(ctx: ExperimentContext) -> None:
    cfg = ctx.cfg
    _write_json(_json_safe(asdict(cfg)), ctx.config_dir / "run_config.json")
    _write_json(
        {
            "main_panels": ["A", "B", "C", "D", "E", "F"],
            "state_bank_required": True,
            "primary_layer": cfg.primary_layer,
            "primary_state_variable": cfg.primary_state_variable,
            "supplementary_outputs": [
                "supp_layerwise_morphology_metrics.csv",
                "supp_linear_mixture_model_comparison.csv",
                "supp_delay_layer_fused_state_metrics.csv",
                "supp_pair_sampling_audit.csv",
                "supp_additive_null_metrics.csv",
                "supp_completion_target_B_metrics.csv",
                "supp_ping_sweep_metrics.csv",
                "supp_completion_delay_sweep_metrics.csv",
                "supp_completion_delay_sweep_contrast.csv",
                "supp_trial_condition_audit.csv",
            ],
            "supplementary_figures": {
                "S3": {
                    "title": "Morphological controls for pair-specific fused STSP states.",
                    "outputs": ["supp_layerwise_morphology_metrics.csv", "supp_linear_mixture_model_comparison.csv"],
                },
                "S4": {
                    "title": "Functional robustness of fused-state access.",
                    "outputs": [
                        "supp_ping_sweep_metrics.csv",
                        "supp_completion_delay_sweep_metrics.csv",
                        "supp_completion_delay_sweep_contrast.csv",
                    ],
                },
            },
        },
        ctx.config_dir / "figure_requirements.json",
    )
    _write_json(
        {
            "state_conditions": list(STATE_CONDITIONS),
            "primary_representation": {"layer": cfg.primary_layer, "state_variable": cfg.primary_state_variable},
            "similarity": "centered_cosine",
            "pair_composite": "0.5*(S_A + S_B)",
            "neutral_ping": "real network rollout from restored S0/S_A/S_B/S_AB boundary states using class-uninformative ping_drive",
            "partial_cue": "real weak-probe rollout from restored S0/S_A/S_B/S_AB boundary states using Fig.4-compatible encoded-spike dropout by default",
            "functional_readout_source": "decode_prediction_and_fire_time_from_layer3",
            "weak_probe_mask_space": str(cfg.weak_probe_mask_space),
            "weak_probe_use_same_mask_across_states": bool(cfg.weak_probe_use_same_mask_across_states),
            "weak_probe_scale": float(cfg.weak_probe_scale),
            "weak_probe_noise": float(cfg.weak_probe_noise),
            "weak_probe_metric_mode": str(cfg.weak_probe_metric_mode),
            "fig4_weak_probe_compat_enabled": bool(cfg.weak_probe_mask_space == "encoded_spikes" and cfg.weak_probe_metric_mode == "fig4_compat"),
        },
        ctx.config_dir / "condition_spec.json",
    )
    _write_json(
        {
            "restore_mode": "reset_all_state_restore_selected_stsp",
            "restore_convention": "Functional readout restores condition-specific STSP u/x states and resets non-STSP fast activity before ping/probe readout.",
            "ping_mode": str(cfg.ping_mode),
            "ping_amp": float(cfg.ping_amp),
            "ping_repeats": int(cfg.ping_repeats),
            "ping_noise": float(cfg.ping_noise),
            "run_ping_sweep": bool(cfg.run_ping_sweep),
            "ping_amp_sweep": list(cfg.ping_amp_sweep),
            "ping_ms_sweep": list(cfg.ping_ms_sweep),
            "weak_probe_keep_probs": list(cfg.weak_probe_keep_probs),
            "weak_probe_repeats": int(cfg.weak_probe_repeats),
            "weak_probe_mask_space": str(cfg.weak_probe_mask_space),
            "weak_probe_use_same_mask_across_states": bool(cfg.weak_probe_use_same_mask_across_states),
            "weak_probe_scale": float(cfg.weak_probe_scale),
            "weak_probe_noise": float(cfg.weak_probe_noise),
            "weak_probe_metric_mode": str(cfg.weak_probe_metric_mode),
            "fig4_weak_probe_compat_enabled": bool(cfg.weak_probe_mask_space == "encoded_spikes" and cfg.weak_probe_metric_mode == "fig4_compat"),
            "run_completion_delay_sweep": bool(cfg.run_completion_delay_sweep),
            "completion_delay_sweep_ms": list(cfg.completion_delay_sweep_ms),
            "completion_delay_keep_prob": float(cfg.completion_delay_keep_prob),
            "completion_delay_repeats": int(cfg.completion_delay_repeats),
            "foreground_threshold": float(cfg.foreground_threshold),
            "decoder_name": "decode_prediction_and_fire_time_from_layer3",
            "proxy_used_for_main": False,
            "save_functional_traces": bool(cfg.save_functional_traces),
        },
        ctx.config_dir / "functional_readout_spec.json",
    )
    _write_json(
        {
            "models": list(MIXTURE_MODELS),
            "baseline_subtraction": "x_A=S_A-S0; x_B=S_B-S0; y_AB=S_AB-S0",
            "cv": {"folds": int(cfg.linear_mixture_cv_folds), "unit": "feature_dimensions"},
            "residual_template_definition": RESIDUAL_TEMPLATE_DEFINITION,
        },
        ctx.config_dir / "linear_mixture_spec.json",
    )


def _write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main = [
        ctx.metrics_dir / "panel_b_dual_retention_metrics.csv",
        ctx.metrics_dir / "panel_c_pair_specificity_metrics.csv",
        ctx.metrics_dir / "panel_d_pair_level_organization_metrics.csv",
        ctx.metrics_dir / "panel_d_linear_mixture_fit_metrics.csv",
        ctx.metrics_dir / "panel_d_linear_residual_pair_specificity_metrics.csv",
        ctx.metrics_dir / "panel_e_neutral_ping_metrics.csv",
        ctx.metrics_dir / "panel_f_partial_cue_metrics.csv",
        ctx.metrics_dir / "panel_f_partial_cue_auc_metrics.csv",
        ctx.metrics_dir / "compat_fig4_weak_probe_summary.csv",
    ]
    required_supp = [
        ctx.metrics_dir / "supp_pair_sampling_audit.csv",
        ctx.metrics_dir / "supp_trial_condition_audit.csv",
    ]
    if ctx.cfg.run_morphology and ctx.cfg.run_linear_mixture:
        required_supp.append(ctx.metrics_dir / "supp_layerwise_morphology_metrics.csv")
    if ctx.cfg.run_linear_mixture:
        required_supp.extend(
            [
                ctx.metrics_dir / "supp_additive_null_metrics.csv",
                ctx.metrics_dir / "supp_linear_mixture_model_comparison.csv",
            ]
        )
    if ctx.cfg.run_partial_cue:
        required_supp.append(ctx.metrics_dir / "supp_completion_target_B_metrics.csv")
    if ctx.cfg.run_supplement:
        required_supp.append(ctx.metrics_dir / "supp_delay_layer_fused_state_metrics.csv")
    if ctx.cfg.run_ping_sweep:
        required_supp.extend(
            [
                ctx.raw_dir / "supp_ping_sweep_trial_readout.csv",
                ctx.metrics_dir / "supp_ping_sweep_metrics.csv",
            ]
        )
    if ctx.cfg.run_completion_delay_sweep:
        required_supp.extend(
            [
                ctx.raw_dir / "supp_completion_delay_sweep_trial_readout.csv",
                ctx.metrics_dir / "supp_completion_delay_sweep_metrics.csv",
                ctx.metrics_dir / "supp_completion_delay_sweep_contrast.csv",
            ]
        )
    summary = {
        "figure": FIGURE_ID,
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": SINGLE_NETWORK_MODE,
        "smoke": bool(ctx.cfg.smoke),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_pairs": int(ctx.n_pairs),
        "state_conditions": list(STATE_CONDITIONS),
        "linear_mixture_models": list(MIXTURE_MODELS),
        "fig2_supplement_plan": {
            "S3": "Morphological controls: WPRI across layers, residual pair-specificity across layers, linear mixture model comparison.",
            "S4": "Functional robustness: ping amplitude sweep, ping duration sweep, completion gain across post-pair retention delays.",
        },
        "ping_sweep_completed": bool(ctx.completed_modules.get("ping_sweep", False)),
        "completion_delay_sweep_completed": bool(ctx.completed_modules.get("completion_delay_sweep", False)),
        "warnings": ctx.warnings,
        "residual_template_definition": RESIDUAL_TEMPLATE_DEFINITION,
        "functional_readout_mode": "real_network_rollout",
        "neutral_ping_proxy_used_for_main": False,
        "partial_cue_proxy_used_for_main": False,
        "weak_probe_mask_space": str(ctx.cfg.weak_probe_mask_space),
        "weak_probe_use_same_mask_across_states": bool(ctx.cfg.weak_probe_use_same_mask_across_states),
        "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
        "weak_probe_noise": float(ctx.cfg.weak_probe_noise),
        "weak_probe_metric_mode": str(ctx.cfg.weak_probe_metric_mode),
        "fig4_weak_probe_compat_enabled": bool(ctx.cfg.weak_probe_mask_space == "encoded_spikes" and ctx.cfg.weak_probe_metric_mode == "fig4_compat"),
        "proxy_diagnostics_available": bool((ctx.metrics_dir / "supp_functional_proxy_diagnostics.csv").exists()),
        "proxy_used_for_main": False,
        "functional_readout_note": "Panel F uses Fig.4-compatible encoded-spike dropout weak probes by default, extended to S0/S_A/S_B/S_AB and bidirectional A/B target recovery.",
        "main_claim_supported_fields_available": all(path.exists() for path in required_main),
        "missing_for_main_figure": [_rel(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_supplementary": [_rel(path, ctx.seed_dir) for path in required_supp if not path.exists()],
    }
    _write_json(summary, ctx.seed_dir / "summary.json")
    ctx.output_files["summary"] = "summary.json"
    return summary


def _write_run_log(ctx: ExperimentContext) -> None:
    ctx.run_log.append(f"{_now()} completed modules={sorted(k for k, v in ctx.completed_modules.items() if v)}")
    path = ctx.seed_dir / "run_log.txt"
    path.write_text("\n".join(ctx.run_log) + "\n", encoding="utf-8")
    ctx.output_files["run_log"] = "run_log.txt"


def _save_csv(ctx: ExperimentContext, df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    ctx.output_files[path.stem] = _rel(path, ctx.seed_dir)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _prepare_dirs(seed_dir: Path) -> dict[str, Path]:
    paths = {
        "config": seed_dir / "config",
        "trial_specs": seed_dir / "data" / "trial_specs",
        "raw": seed_dir / "data" / "raw",
        "metrics": seed_dir / "data" / "metrics",
        "debug": seed_dir / "debug_figures",
        "meta": seed_dir / "meta",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _resolve_seed_dir(output_root: Path, network_seed: int) -> Path:
    if output_root.name.startswith("seed_"):
        return output_root
    return output_root / f"seed_{int(network_seed):03d}"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


def _maybe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _maybe_int(value: Any) -> int | float:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return float("nan")
    return int(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_from_args(args: argparse.Namespace) -> Fig2Config:
    smoke = bool(args.smoke)
    run_all = bool(args.run_all)
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
    return Fig2Config(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        output_root=str(args.output_root),
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
        num_pairs=min(int(args.num_pairs), 20) if smoke else int(args.num_pairs),
        batch_size=min(int(args.batch_size), 4) if smoke else int(args.batch_size),
        n_shuffle=4 if smoke else int(args.n_shuffle),
        delay_layer_grid=delay_grid[:2] if smoke else delay_grid,
        linear_mixture_cv_folds=2 if smoke else int(args.linear_mixture_cv_folds),
        run_state_bank=run_all or bool(args.run_state_bank),
        run_morphology=run_all or bool(args.run_morphology),
        run_linear_mixture=run_all or bool(args.run_linear_mixture),
        run_neutral_ping=run_all or bool(args.run_neutral_ping),
        run_partial_cue=run_all or bool(args.run_partial_cue),
        run_supplement=run_all or bool(args.run_supplement),
        run_ping_sweep=run_all or bool(args.run_ping_sweep),
        run_completion_delay_sweep=run_all or bool(args.run_completion_delay_sweep),
        completion_delay_sweep_ms=completion_delay_sweep_ms,
        completion_delay_keep_prob=float(args.completion_delay_keep_prob),
        completion_delay_repeats=completion_delay_repeats,
        save_debug_figures=bool(args.save_debug_figures),
        save_spike_cache=bool(args.save_spike_cache),
        save_all_layer_state_bank=bool(args.save_all_layer_state_bank),
        save_functional_traces=bool(args.save_functional_traces),
        save_proxy_functional_debug=bool(args.save_proxy_functional_debug),
        show_progress=not bool(args.no_progress),
        use_encode_cache=not bool(args.no_encode_cache),
        enable_partial_cue_batch=bool(args.enable_partial_cue_batch),
        smoke=smoke,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Fig.2 pair-fused STSP state experiment.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-state-bank", action="store_true")
    parser.add_argument("--run-morphology", action="store_true")
    parser.add_argument("--run-linear-mixture", action="store_true")
    parser.add_argument("--run-neutral-ping", action="store_true")
    parser.add_argument("--run-partial-cue", action="store_true")
    parser.add_argument("--run-supplement", action="store_true")
    parser.add_argument("--run-ping-sweep", action="store_true")
    parser.add_argument("--run-completion-delay-sweep", action="store_true")
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--save-spike-cache", action="store_true")
    parser.add_argument("--save-all-layer-state-bank", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-encode-cache", action="store_true")
    parser.add_argument("--enable-partial-cue-batch", action="store_true")
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
    parser.add_argument("--foreground-threshold", type=float, default=0.0)
    parser.add_argument("--num-pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-shuffle", type=int, default=50)
    parser.add_argument("--delay-layer-grid", default="200,400,800")
    parser.add_argument("--linear-mixture-cv-folds", type=int, default=5)
    parser.add_argument("--completion-delay-sweep-ms", default="200,400,800,1200")
    parser.add_argument("--completion-delay-keep-prob", type=float, default=0.2)
    parser.add_argument("--completion-delay-repeats", type=int, default=5)
    parser.add_argument("--save-functional-traces", action="store_true")
    parser.add_argument("--save-proxy-functional-debug", action="store_true")
    return parser.parse_args(argv)


PANEL_E_RAW_COLUMNS = [
    "network_seed",
    "pair_id",
    "state_condition",
    "ping_repeat",
    "ping_seed",
    "A_label",
    "B_label",
    "prediction",
    "pred_is_A",
    "pred_is_B",
    "pred_is_pair_member",
    "pred_is_other",
    "silent",
    "first_fire_time_ms",
    "ping_spike_count",
    "ping_energy",
    "readout_margin_A",
    "readout_margin_B",
]
SUPP_PING_SWEEP_RAW_COLUMNS = [
    "network_seed",
    "pair_id",
    "state_condition",
    "sweep_type",
    "ping_amp",
    "ping_ms",
    "ping_repeat",
    "A_label",
    "B_label",
    "prediction",
    "pred_is_A",
    "pred_is_B",
    "pred_is_pair_member",
    "pred_is_other",
    "silent",
    "first_fire_time_ms",
]
WEAK_PROBE_MASK_COLUMNS = [
    "network_seed",
    "mask_id",
    "pair_id",
    "target_item",
    "target_label",
    "keep_prob",
    "repeat_id",
    "mask_seed",
    "mask_space",
    "same_mask_used_across_states",
    "weak_probe_scale",
    "weak_probe_noise",
    "realized_keep_fraction",
    "full_spike_count",
    "weak_spike_count",
    "weak_spike_fraction",
    "cue_pixel_count",
    "target_foreground_count",
    "cue_fraction_actual",
    "cue_energy",
    "encoded_spike_count",
]
SUPP_COMPLETION_DELAY_RAW_COLUMNS = [
    "network_seed",
    "pair_id",
    "delay2_ms",
    "state_condition",
    "target_item",
    "target_label",
    "A_label",
    "B_label",
    "keep_prob",
    "repeat_id",
    "prediction",
    "correct_target",
    "pred_is_A",
    "pred_is_B",
    "pred_is_other",
    "silent",
    "first_fire_time_ms",
    "weak_probe_scale",
    "weak_spike_count",
]
PANEL_F_RAW_COLUMNS = [
    "network_seed",
    "pair_id",
    "state_condition",
    "target_item",
    "target_label",
    "other_pair_label",
    "keep_prob",
    "repeat_id",
    "mask_id",
    "prediction",
    "pred_is_target",
    "pred_is_A",
    "pred_is_B",
    "pred_is_pair_member",
    "pred_is_other_pair_member",
    "pred_is_other_class",
    "silent",
    "first_fire_time_ms",
    "mask_space",
    "weak_probe_scale",
    "weak_probe_noise",
    "weak_probe_metric_mode",
    "realized_keep_fraction",
    "cue_fraction_actual",
    "weak_spike_fraction",
    "same_mask_used_across_states",
    "cue_pixel_count",
    "target_foreground_count",
    "cue_energy",
    "encoded_spike_count",
]


if __name__ == "__main__":
    raise SystemExit(main())
