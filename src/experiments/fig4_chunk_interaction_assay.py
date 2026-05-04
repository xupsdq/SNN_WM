from __future__ import annotations

"""Fig. 4 chunk interaction assay.

This experiment tests whether high-overlap A->B pairs create selective Layer 3
shared-feature STSP peaks beyond a mean-mixture additive state. Core comparisons
are true sequential S_AB vs S_mean, high-overlap vs low-overlap pairs, and
functional readout before/after removing only the Layer 3 chunk peak.
"""

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import DEFAULT_PATH_CONFIG
from src.config.units import ms
from src.experiments.common.dataset import build_class_index, build_dataset_arrays, encode_images
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import (
    build_layer_input_shapes,
    reset_all_state_restore_selected_stsp_in_place,
)
from src.experiments.common.results import (
    prepare_result_layout,
    save_log_lines,
    save_run_config,
    save_summary_json,
)
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.seed import mix_seed
from src.experiments.common.statistics import bootstrap_mean_ci, sem
from src.plotting.common.io import save_tidy_csv

EXPERIMENT_ID = "fig4_chunk_interaction_assay"
LAYER_KEYS = ("layer1", "layer2", "layer3")
EPS = 1e-12


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path
    model_path: Path
    dataset_root: Path
    dataset_split: str
    device: str
    seeds: tuple[int, ...]
    num_pairs_per_group: int
    high_overlap_quantile: float
    low_overlap_quantile: float
    candidate_pool_per_class: int
    pair_candidate_limit: int
    batch_size: int
    sample_duration: float
    delay1_duration: float
    delay2_duration: float
    probe_duration: float
    probe_scale: float
    probe_noise: float
    member_probe_scale: float
    num_probe_candidates: int
    num_nonmember_probes: int
    probe_c_selection: str
    baseline_confusion_max: float
    ping_duration: float
    ping_amp: float
    ping_amp_list: tuple[float, ...]
    ping_repeats: int
    ping_noise: float
    mask_top_q: float
    shared_mask_top_q: float
    peak_top_q: float
    morphology_layer: str
    state_name: str
    additive_mode: str
    use_ls_additive: bool
    blockade_mode: str
    save_states: bool
    save_state_debug: bool
    smoke_test: bool

    def steps(self, duration_ms: float) -> int:
        return max(0, int(round((float(duration_ms) * ms) / (1.0 * ms))))

    @property
    def sample_steps(self) -> int:
        return self.steps(self.sample_duration)

    @property
    def delay1_steps(self) -> int:
        return self.steps(self.delay1_duration)

    @property
    def delay2_steps(self) -> int:
        return self.steps(self.delay2_duration)

    @property
    def probe_steps(self) -> int:
        return self.steps(self.probe_duration)

    @property
    def ping_steps(self) -> int:
        return self.steps(self.ping_duration)

    @property
    def encoder_max_duration_ms(self) -> float:
        return max(self.sample_duration, self.probe_duration, self.ping_duration, 100.0)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir.resolve())
        data["model_path"] = str(self.model_path.resolve())
        data["dataset_root"] = str(self.dataset_root.resolve())
        data["seeds"] = list(self.seeds)
        return data


@dataclass
class RunLogger:
    lines: list[str]

    def log(self, message: str) -> None:
        text = str(message)
        print(text, flush=True)
        self.lines.append(text)


@dataclass(frozen=True)
class StspSnapshot:
    layer_input_shapes: dict[str, tuple[int, ...]]
    restore_ux_by_layer: dict[str, tuple[torch.Tensor, torch.Tensor]]
    state_by_layer: dict[str, dict[str, np.ndarray]]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _relativize_files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fig. 4 chunk-specific STSP interaction assay.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_PATH_CONFIG.results_root / EXPERIMENT_ID))
    parser.add_argument("--checkpoint", "--model-path", dest="model_path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--dataset-split", "--split", dest="dataset_split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None, help="Single-seed compatibility alias.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--num-pairs-per-group", type=int, default=20)
    parser.add_argument("--high-overlap-quantile", type=float, default=0.80)
    parser.add_argument("--low-overlap-quantile", type=float, default=0.20)
    parser.add_argument("--candidate-pool-per-class", type=int, default=50)
    parser.add_argument("--pair-candidate-limit", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-duration", type=float, default=100.0)
    parser.add_argument("--delay1-duration", type=float, default=50.0)
    parser.add_argument("--delay2-duration", type=float, default=50.0)
    parser.add_argument("--probe-duration", type=float, default=30.0)
    parser.add_argument("--probe-scale", type=float, default=0.35)
    parser.add_argument("--probe-noise", type=float, default=0.0)
    parser.add_argument("--member-probe-scale", type=float, default=None)
    parser.add_argument("--num-probe-candidates", type=int, default=2)
    parser.add_argument("--num-nonmember-probes", type=int, default=5)
    parser.add_argument("--probe-c-selection", type=str, default="baseline_matched", choices=["random", "baseline_matched"])
    parser.add_argument("--baseline-confusion-max", type=float, default=1.0)
    parser.add_argument("--ping-duration", type=float, default=30.0)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--ping-amp-list", type=float, nargs="*", default=None)
    parser.add_argument("--ping-repeats", type=int, default=20)
    parser.add_argument("--ping-noise", type=float, default=0.0)
    parser.add_argument("--mask-top-q", type=float, default=0.20)
    parser.add_argument("--shared-mask-top-q", type=float, default=0.20)
    parser.add_argument("--peak-top-q", type=float, default=0.10)
    parser.add_argument("--morphology-layer", type=str, default="layer3", choices=["layer3"])
    parser.add_argument("--state-name", type=str, default="g", choices=["g", "u", "x"])
    parser.add_argument("--use-ls-additive", action="store_true")
    parser.add_argument("--additive-mode", type=str, default="mean", choices=["mean", "sum", "ls"])
    parser.add_argument("--blockade-mode", type=str, default="hold", choices=["hold"])
    parser.add_argument("--save-states", action="store_true")
    parser.add_argument("--save-state-debug", action="store_true")
    parser.add_argument("--skip-figures", action="store_true", help="Accepted for shared runner compatibility; computation is plot-free.")
    parser.add_argument("--smoke-test", "--smoke", dest="smoke_test", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    seeds = tuple(int(v) for v in (args.seeds if args.seeds else ()))
    if not seeds:
        seeds = (int(args.seed),) if args.seed is not None else tuple(range(20))
    cfg = ExperimentConfig(
        output_dir=Path(args.output_dir),
        model_path=Path(args.model_path),
        dataset_root=Path(args.dataset_root),
        dataset_split=str(args.dataset_split),
        device=str(args.device),
        seeds=seeds,
        num_pairs_per_group=int(args.num_pairs_per_group),
        high_overlap_quantile=float(args.high_overlap_quantile),
        low_overlap_quantile=float(args.low_overlap_quantile),
        candidate_pool_per_class=int(args.candidate_pool_per_class),
        pair_candidate_limit=int(args.pair_candidate_limit),
        batch_size=int(args.batch_size),
        sample_duration=float(args.sample_duration),
        delay1_duration=float(args.delay1_duration),
        delay2_duration=float(args.delay2_duration),
        probe_duration=float(args.probe_duration),
        probe_scale=float(args.probe_scale),
        probe_noise=float(args.probe_noise),
        member_probe_scale=float(args.member_probe_scale if args.member_probe_scale is not None else args.probe_scale),
        num_probe_candidates=int(args.num_probe_candidates),
        num_nonmember_probes=int(args.num_nonmember_probes),
        probe_c_selection=str(args.probe_c_selection),
        baseline_confusion_max=float(args.baseline_confusion_max),
        ping_duration=float(args.ping_duration),
        ping_amp=float(args.ping_amp),
        ping_amp_list=tuple(float(v) for v in (args.ping_amp_list if args.ping_amp_list else (float(args.ping_amp),))),
        ping_repeats=int(args.ping_repeats),
        ping_noise=float(args.ping_noise),
        mask_top_q=float(args.mask_top_q),
        shared_mask_top_q=float(args.shared_mask_top_q),
        peak_top_q=float(args.peak_top_q),
        morphology_layer=str(args.morphology_layer),
        state_name=str(args.state_name),
        additive_mode=str(args.additive_mode),
        use_ls_additive=bool(args.use_ls_additive or str(args.additive_mode) == "ls"),
        blockade_mode=str(args.blockade_mode),
        save_states=bool(args.save_states),
        save_state_debug=bool(args.save_state_debug),
        smoke_test=bool(args.smoke_test),
    )
    if cfg.smoke_test:
        cfg = ExperimentConfig(
            **{
                **asdict(cfg),
                "seeds": tuple(cfg.seeds[:1]) if cfg.seeds else (0,),
                "num_pairs_per_group": min(cfg.num_pairs_per_group, 1),
                "candidate_pool_per_class": min(cfg.candidate_pool_per_class, 8),
                "pair_candidate_limit": min(cfg.pair_candidate_limit, 800),
                "batch_size": min(cfg.batch_size, 2),
                "num_probe_candidates": min(cfg.num_probe_candidates, 1),
                "num_nonmember_probes": min(cfg.num_nonmember_probes, 1),
                "ping_repeats": min(cfg.ping_repeats, 2),
                "sample_duration": min(cfg.sample_duration, 30.0),
                "delay1_duration": min(cfg.delay1_duration, 5.0),
                "delay2_duration": min(cfg.delay2_duration, 5.0),
                "probe_duration": min(cfg.probe_duration, 12.0),
                "ping_duration": min(cfg.ping_duration, 12.0),
            }
        )
    if cfg.sample_steps <= 0 or cfg.probe_steps <= 0 or cfg.ping_steps <= 0:
        raise ValueError("sample, probe, and ping durations must map to at least one step.")
    if not (0.0 < cfg.low_overlap_quantile < cfg.high_overlap_quantile < 1.0):
        raise ValueError("Require 0 < low-overlap-quantile < high-overlap-quantile < 1.")
    if not (0.0 < cfg.mask_top_q <= 1.0):
        raise ValueError("--mask-top-q must be in (0, 1].")
    if not (0.0 < cfg.shared_mask_top_q <= 1.0):
        raise ValueError("--shared-mask-top-q must be in (0, 1].")
    if not (0.0 < cfg.peak_top_q <= 1.0):
        raise ValueError("--peak-top-q must be in (0, 1].")
    if cfg.ping_repeats <= 0 or cfg.num_nonmember_probes <= 0:
        raise ValueError("--ping-repeats and --num-nonmember-probes must be positive.")
    return cfg


def _tensor_images(images: torch.Tensor, ids: Sequence[int], device: torch.device) -> torch.Tensor:
    return images[torch.as_tensor(list(ids), dtype=torch.long)].to(device=device, dtype=torch.float32)


def _flatten_unit(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)
    norm = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), EPS)
    return arr / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(av, bv) / max(np.linalg.norm(av) * np.linalg.norm(bv), EPS))


def _weighted_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    av = np.maximum(np.asarray(a, dtype=np.float64).reshape(-1), 0.0)
    bv = np.maximum(np.asarray(b, dtype=np.float64).reshape(-1), 0.0)
    return float(np.minimum(av, bv).sum() / max(np.maximum(av, bv).sum(), EPS))


def _foreground_mask(image: torch.Tensor, threshold: float = 0.05) -> np.ndarray:
    arr = image.detach().cpu().numpy().squeeze()
    return np.asarray(arr > float(threshold), dtype=bool)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        return float("nan"), float("nan")
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _foreground_dice(a: np.ndarray, b: np.ndarray) -> float:
    ab = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    return float(2.0 * np.logical_and(ab, bb).sum() / max(ab.sum() + bb.sum(), 1))


def _encode_id_bank(
    ids: Sequence[int],
    images: torch.Tensor,
    encoder: Any,
    *,
    steps: int,
    device: torch.device,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    unique_ids = list(dict.fromkeys(int(v) for v in ids))
    for start in range(0, len(unique_ids), max(1, int(batch_size))):
        batch_ids = unique_ids[start : start + max(1, int(batch_size))]
        batch_images = _tensor_images(images, batch_ids, device)
        spikes = encode_images(encoder, batch_images, steps).detach()
        for idx, image_id in enumerate(batch_ids):
            out[int(image_id)] = spikes[idx].detach().clone()
    return out


def _precompute_pair_features(
    pool_ids: Sequence[int],
    images: torch.Tensor,
    labels: np.ndarray,
    net: Any,
    encoder: Any,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[int, dict[str, Any]]:
    spike_lookup = _encode_id_bank(pool_ids, images, encoder, steps=cfg.sample_steps, device=device, batch_size=cfg.batch_size)
    features: dict[int, dict[str, Any]] = {}
    kernels = net.layer1.kernels.detach().abs()
    for image_id in pool_ids:
        image = images[int(image_id)]
        fg = _foreground_mask(image)
        centroid_y, centroid_x = _centroid(fg)
        spikes = spike_lookup[int(image_id)]
        spike_energy = spikes.sum(dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
        with torch.no_grad():
            drive_t = F.conv2d(spikes.to(device).float(), kernels, stride=net.layer1.stride, padding=net.layer1.padding)
            drive = drive_t.sum(dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
        features[int(image_id)] = {
            "label": int(labels[int(image_id)]),
            "pixel_flat": image.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1),
            "foreground_mask": fg,
            "foreground_area": int(fg.sum()),
            "centroid_y": centroid_y,
            "centroid_x": centroid_x,
            "spike_energy_map": spike_energy,
            "spike_energy": float(spike_energy.sum()),
            "layer1_drive": drive,
            "layer1_drive_energy": float(drive.sum()),
        }
    return features


def _sample_pool_ids(class_index: Mapping[int, Sequence[int]], per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    ids: list[int] = []
    for label in sorted(class_index):
        pool = np.asarray(class_index[int(label)], dtype=np.int64)
        if pool.size == 0:
            continue
        count = min(int(per_class), int(pool.size))
        chosen = rng.choice(pool, size=count, replace=False)
        ids.extend(int(v) for v in chosen.tolist())
    return sorted(dict.fromkeys(ids))


def build_pair_table_for_seed(
    *,
    seed: int,
    images: torch.Tensor,
    labels: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    net: Any,
    encoder: Any,
    cfg: ExperimentConfig,
    device: torch.device,
) -> pd.DataFrame:
    rng = np.random.default_rng(mix_seed(seed, 101))
    pool_ids = _sample_pool_ids(class_index, cfg.candidate_pool_per_class, mix_seed(seed, 103))
    features = _precompute_pair_features(pool_ids, images, labels, net, encoder, cfg, device)
    pair_rows: list[dict[str, Any]] = []
    candidate_pairs = [(a, b) for a in pool_ids for b in pool_ids if a != b and labels[int(a)] != labels[int(b)]]
    if len(candidate_pairs) > int(cfg.pair_candidate_limit):
        keep = rng.choice(np.arange(len(candidate_pairs)), size=int(cfg.pair_candidate_limit), replace=False)
        candidate_pairs = [candidate_pairs[int(i)] for i in keep.tolist()]
    pixel_units = {idx: vec / max(np.linalg.norm(vec), EPS) for idx, vec in ((i, features[i]["pixel_flat"]) for i in pool_ids)}
    for sample_id, second_id in candidate_pairs:
        fa = features[int(sample_id)]
        fb = features[int(second_id)]
        sample_area = float(fa["foreground_area"])
        second_area = float(fb["foreground_area"])
        sample_energy = float(fa["spike_energy"])
        second_energy = float(fb["spike_energy"])
        cy_a, cx_a = float(fa["centroid_y"]), float(fa["centroid_x"])
        cy_b, cx_b = float(fb["centroid_y"]), float(fb["centroid_x"])
        pair_rows.append(
            {
                "sample_idx": int(sample_id),
                "second_idx": int(second_id),
                "sample_label": int(fa["label"]),
                "second_label": int(fb["label"]),
                "pixel_similarity": float(np.dot(pixel_units[int(sample_id)], pixel_units[int(second_id)])),
                "pixel_dice_overlap": _foreground_dice(fa["foreground_mask"], fb["foreground_mask"]),
                "encoded_spikemap_overlap": _weighted_jaccard(fa["spike_energy_map"], fb["spike_energy_map"]),
                "layer1_drive_overlap": _weighted_jaccard(fa["layer1_drive"], fb["layer1_drive"]),
                "feature_overlap": _weighted_jaccard(fa["layer1_drive"], fb["layer1_drive"]),
                "sample_area": sample_area,
                "second_area": second_area,
                "area_abs_delta": abs(sample_area - second_area),
                "sample_energy": sample_energy,
                "second_energy": second_energy,
                "energy_abs_delta": abs(sample_energy - second_energy),
                "sample_centroid_y": cy_a,
                "sample_centroid_x": cx_a,
                "second_centroid_y": cy_b,
                "second_centroid_x": cx_b,
                "centroid_distance": float(math.hypot(cy_a - cy_b, cx_a - cx_b)) if np.isfinite(cy_a + cy_b + cx_a + cx_b) else np.nan,
            }
        )
    candidates = pd.DataFrame(pair_rows)
    if candidates.empty:
        raise RuntimeError("No valid different-label A/B pair candidates were found.")
    lo = float(candidates["feature_overlap"].quantile(cfg.low_overlap_quantile))
    hi = float(candidates["feature_overlap"].quantile(cfg.high_overlap_quantile))
    high_pool = candidates[candidates["feature_overlap"] >= hi].sort_values(
        ["feature_overlap", "area_abs_delta", "energy_abs_delta", "centroid_distance"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    low_pool = candidates[candidates["feature_overlap"] <= lo].sort_values(
        ["feature_overlap", "area_abs_delta", "energy_abs_delta", "centroid_distance"],
        ascending=[True, True, True, True],
        kind="stable",
    )
    selected: list[pd.DataFrame] = []
    used: set[tuple[str, int, int]] = set()
    for group_name, pool in (("high", high_pool), ("low", low_pool)):
        rows: list[pd.Series] = []
        label_pair_counts: dict[tuple[int, int], int] = {}
        for _, row in pool.iterrows():
            if len(rows) >= int(cfg.num_pairs_per_group):
                break
            key = (int(row["sample_label"]), int(row["second_label"]))
            image_key = (group_name, int(row["sample_idx"]), int(row["second_idx"]))
            if image_key in used:
                continue
            max_allowed = 1 + len(rows) // max(1, 90)
            if label_pair_counts.get(key, 0) > max_allowed:
                continue
            rows.append(row)
            used.add(image_key)
            label_pair_counts[key] = label_pair_counts.get(key, 0) + 1
        if len(rows) < int(cfg.num_pairs_per_group):
            for _, row in pool.iterrows():
                if len(rows) >= int(cfg.num_pairs_per_group):
                    break
                image_key = (group_name, int(row["sample_idx"]), int(row["second_idx"]))
                if image_key in used:
                    continue
                rows.append(row)
                used.add(image_key)
        if len(rows) == 0:
            raise RuntimeError(f"No {group_name}-overlap pairs could be selected.")
        group_df = pd.DataFrame(rows).copy()
        group_df["overlap_group"] = group_name
        selected.append(group_df)
    out = pd.concat(selected, ignore_index=True)
    out.insert(0, "seed", int(seed))
    out.insert(1, "pair_id", [f"seed{int(seed):03d}_{g}_{i:04d}" for i, g in enumerate(out["overlap_group"].astype(str))])
    out["overlap_quantile_low_cut"] = lo
    out["overlap_quantile_high_cut"] = hi
    out["selection_rank"] = out.groupby("overlap_group").cumcount().astype(int)
    return out.reset_index(drop=True)


def forward_three_layers(
    net: Any,
    input_t: torch.Tensor,
    t_step: int,
    *,
    ping_drive: torch.Tensor | None = None,
    freeze_mask_by_layer: Mapping[str, torch.Tensor] | None = None,
    freeze_values_by_layer: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    masks = freeze_mask_by_layer or {}
    values = freeze_values_by_layer or {}

    def restore_masked(layer_key: str) -> None:
        if layer_key not in masks:
            return
        layer = getattr(net, layer_key)
        mask = masks[layer_key].to(device=layer.u_pre.device, dtype=torch.bool)
        u0, x0 = values[layer_key]
        while mask.ndim < layer.u_pre.ndim:
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(layer.u_pre)
        layer.u_pre.copy_(torch.where(mask, u0.to(layer.u_pre.device, layer.u_pre.dtype), layer.u_pre))
        layer.x_pre.copy_(torch.where(mask, x0.to(layer.x_pre.device, layer.x_pre.dtype), layer.x_pre))

    s1, _ = net.layer1.forward_step(input_t, t_step, training=False, monitor=False, stsp_mode="dynamic", ping_drive=ping_drive)
    restore_masked("layer1")
    s1_p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1_p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    restore_masked("layer2")
    s2_p = net.pool2(s2.float())
    net.layer3.forward_step(s2_p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    restore_masked("layer3")


def snapshot_stsp_state(net: Any, layer_input_shapes: Mapping[str, tuple[int, ...]], batch_size: int) -> StspSnapshot:
    state_by_layer: dict[str, dict[str, np.ndarray]] = {}
    restore: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        if layer.u_pre is None or layer.x_pre is None:
            raise ValueError(f"{layer_key} is missing STSP state.")
        u = layer.u_pre.detach().cpu().clone()
        x = layer.x_pre.detach().cpu().clone()
        restore[layer_key] = (u, x)
        state_by_layer[layer_key] = {
            "u": u.view(batch_size, -1).numpy().astype(np.float32, copy=False),
            "x": x.view(batch_size, -1).numpy().astype(np.float32, copy=False),
            "g": (u * x).view(batch_size, -1).numpy().astype(np.float32, copy=False),
        }
    return StspSnapshot(
        layer_input_shapes={str(k): tuple(v) for k, v in layer_input_shapes.items()},
        restore_ux_by_layer=restore,
        state_by_layer=state_by_layer,
    )


def run_state_capture(
    net: Any,
    sample_spikes: torch.Tensor,
    second_spikes: torch.Tensor,
    cfg: ExperimentConfig,
    *,
    active_a: bool,
    active_b: bool,
    freeze_mask_by_layer: Mapping[str, torch.Tensor] | None = None,
) -> tuple[StspSnapshot, np.ndarray, np.ndarray]:
    batch_size, _, channels, height, width = sample_spikes.shape
    with torch.no_grad():
        net.layer1.reset_state((batch_size, channels, height, width))
        layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
        net.layer2.reset_state(layer_input_shapes["layer2"])
        net.layer3.reset_state(layer_input_shapes["layer3"])
        zero_input = torch.zeros((batch_size, channels, height, width), dtype=sample_spikes.dtype, device=sample_spikes.device)
        current_time = 0
        for t_idx in range(int(sample_spikes.shape[1])):
            forward_three_layers(net, sample_spikes[:, t_idx] if active_a else zero_input, current_time)
            current_time += 1
        for _ in range(int(cfg.delay1_steps)):
            forward_three_layers(net, zero_input, current_time)
            current_time += 1
        freeze_values: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if freeze_mask_by_layer:
            for layer_key in freeze_mask_by_layer:
                layer = getattr(net, layer_key)
                freeze_values[layer_key] = (layer.u_pre.detach().clone(), layer.x_pre.detach().clone())
        net.layer3.reset_decision_state()
        for t_idx in range(int(second_spikes.shape[1])):
            forward_three_layers(
                net,
                second_spikes[:, t_idx] if active_b else zero_input,
                current_time,
                freeze_mask_by_layer=freeze_mask_by_layer if active_b else None,
                freeze_values_by_layer=freeze_values if active_b else None,
            )
            current_time += 1
        b_pred, b_fire_t = decode_prediction_and_fire_time_from_layer3(net, batch_size=batch_size)
        for _ in range(int(cfg.delay2_steps)):
            forward_three_layers(net, zero_input, current_time)
            current_time += 1
        return snapshot_stsp_state(net, layer_input_shapes, batch_size), b_pred.numpy(), b_fire_t.numpy()


def _combined_state_matrix(snapshot: StspSnapshot, *, state_name: str = "g") -> np.ndarray:
    return np.concatenate([snapshot.state_by_layer[layer][state_name] for layer in LAYER_KEYS], axis=1).astype(np.float32, copy=False)


def _centered_unit_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64)
    centered = arr - arr.mean(axis=1, keepdims=True)
    norms = np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), EPS)
    return centered, centered / norms


def centered_cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac, _ = _centered_unit_rows(a)
    bc, _ = _centered_unit_rows(b)
    denom = np.maximum(np.linalg.norm(ac, axis=1) * np.linalg.norm(bc, axis=1), EPS)
    return (np.sum(ac * bc, axis=1) / denom).astype(np.float32, copy=False)


def pair_composite_similarity(state: np.ndarray, part_a: np.ndarray, part_b: np.ndarray) -> np.ndarray:
    sc, _ = _centered_unit_rows(state)
    ac, _ = _centered_unit_rows(part_a)
    bc, _ = _centered_unit_rows(part_b)
    comp = ac + bc
    comp_norm = np.maximum(np.linalg.norm(comp, axis=1), EPS)
    state_norm = np.maximum(np.linalg.norm(sc, axis=1), EPS)
    return (np.sum(sc * comp, axis=1) / (state_norm * comp_norm)).astype(np.float32, copy=False)


def fit_additive_snapshot(
    baseline: StspSnapshot,
    state_a: StspSnapshot,
    state_b: StspSnapshot,
    state_ab: StspSnapshot,
) -> tuple[StspSnapshot, pd.DataFrame]:
    batch_size = next(iter(baseline.restore_ux_by_layer.values()))[0].shape[0]
    restore: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    state_by_layer: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for layer_key in LAYER_KEYS:
        u0, x0 = baseline.restore_ux_by_layer[layer_key]
        ua, xa = state_a.restore_ux_by_layer[layer_key]
        ub, xb = state_b.restore_ux_by_layer[layer_key]
        uab, xab = state_ab.restore_ux_by_layer[layer_key]
        out_u = torch.empty_like(u0)
        out_x = torch.empty_like(x0)
        for variable, base, a_val, b_val, target, out_tensor in (
            ("u", u0, ua, ub, uab, out_u),
            ("x", x0, xa, xb, xab, out_x),
        ):
            base_flat = base.view(batch_size, -1).numpy().astype(np.float64, copy=False)
            a_flat = a_val.view(batch_size, -1).numpy().astype(np.float64, copy=False)
            b_flat = b_val.view(batch_size, -1).numpy().astype(np.float64, copy=False)
            y_flat = target.view(batch_size, -1).numpy().astype(np.float64, copy=False)
            fitted_rows: list[np.ndarray] = []
            for row_idx in range(batch_size):
                x_design = np.stack([a_flat[row_idx] - base_flat[row_idx], b_flat[row_idx] - base_flat[row_idx]], axis=1)
                y = y_flat[row_idx] - base_flat[row_idx]
                coef, *_ = np.linalg.lstsq(x_design, y, rcond=None)
                fitted = base_flat[row_idx] + x_design @ coef
                clipped = np.clip(fitted, 0.0, 1.0)
                residual = y - x_design @ coef
                denom = float(np.sum((y - y.mean()) ** 2))
                r2 = 1.0 - float(np.sum(residual**2)) / max(denom, EPS)
                clip_fraction = float(np.mean(np.abs(fitted - clipped) > 1e-7))
                rows.append(
                    {
                        "row_idx": int(row_idx),
                        "layer": layer_key,
                        "variable": variable,
                        "alpha": float(coef[0]),
                        "beta": float(coef[1]),
                        "residual_norm": float(np.linalg.norm(residual)),
                        "target_norm": float(np.linalg.norm(y)),
                        "r2": float(r2),
                        "mean_additive": float(clipped.mean()),
                        "mean_target": float(y_flat[row_idx].mean()),
                        "std_additive": float(clipped.std()),
                        "std_target": float(y_flat[row_idx].std()),
                        "norm_additive": float(np.linalg.norm(clipped - base_flat[row_idx])),
                        "norm_target": float(np.linalg.norm(y)),
                        "clipping_fraction": clip_fraction,
                    }
                )
                fitted_rows.append(clipped.astype(np.float32, copy=False))
            out_tensor.copy_(torch.from_numpy(np.stack(fitted_rows, axis=0)).view_as(out_tensor))
        restore[layer_key] = (out_u, out_x)
        state_by_layer[layer_key] = {
            "u": out_u.view(batch_size, -1).numpy().astype(np.float32, copy=False),
            "x": out_x.view(batch_size, -1).numpy().astype(np.float32, copy=False),
            "g": (out_u * out_x).view(batch_size, -1).numpy().astype(np.float32, copy=False),
        }
    return (
        StspSnapshot(
            layer_input_shapes=dict(baseline.layer_input_shapes),
            restore_ux_by_layer=restore,
            state_by_layer=state_by_layer,
        ),
        pd.DataFrame(rows),
    )


def build_mean_mixture_snapshot(
    baseline: StspSnapshot,
    state_a: StspSnapshot,
    state_b: StspSnapshot,
    *,
    mode: str = "mean",
) -> tuple[StspSnapshot, pd.DataFrame]:
    """Construct a baseline-centered mean/sum additive control state."""
    batch_size = next(iter(baseline.restore_ux_by_layer.values()))[0].shape[0]
    restore: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    state_by_layer: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    scale = 1.0 if str(mode) == "sum" else 0.5
    for layer_key in LAYER_KEYS:
        u0, x0 = baseline.restore_ux_by_layer[layer_key]
        ua, xa = state_a.restore_ux_by_layer[layer_key]
        ub, xb = state_b.restore_ux_by_layer[layer_key]
        raw_u = u0 + float(scale) * (ua - u0) + float(scale) * (ub - u0)
        raw_x = x0 + float(scale) * (xa - x0) + float(scale) * (xb - x0)
        out_u = torch.clamp(raw_u, 0.0, 1.0).detach().cpu().clone()
        out_x = torch.clamp(raw_x, 0.0, 1.0).detach().cpu().clone()
        clip_fraction = float(
            torch.cat(
                [
                    (raw_u.detach().cpu() != out_u).reshape(batch_size, -1).float(),
                    (raw_x.detach().cpu() != out_x).reshape(batch_size, -1).float(),
                ],
                dim=1,
            )
            .mean(dim=1)
            .mean()
            .item()
        )
        restore[layer_key] = (out_u, out_x)
        g = out_u * out_x
        state_by_layer[layer_key] = {
            "u": out_u.view(batch_size, -1).numpy().astype(np.float32, copy=False),
            "x": out_x.view(batch_size, -1).numpy().astype(np.float32, copy=False),
            "g": g.view(batch_size, -1).numpy().astype(np.float32, copy=False),
        }
        for row_idx in range(batch_size):
            rows.append(
                {
                    "row_idx": int(row_idx),
                    "layer": layer_key,
                    "additive_mode": str(mode),
                    "g_mean": float(state_by_layer[layer_key]["g"][row_idx].mean()),
                    "g_std": float(state_by_layer[layer_key]["g"][row_idx].std()),
                    "g_norm": float(np.linalg.norm(state_by_layer[layer_key]["g"][row_idx])),
                    "clipping_fraction": clip_fraction,
                }
            )
    return (
        StspSnapshot(
            layer_input_shapes=dict(baseline.layer_input_shapes),
            restore_ux_by_layer=restore,
            state_by_layer=state_by_layer,
        ),
        pd.DataFrame(rows),
    )


def layer_state_matrix(snapshot: StspSnapshot, *, layer: str = "layer3", state_name: str = "g") -> np.ndarray:
    """Return a flattened layer-specific STSP matrix for morphology."""
    if layer not in snapshot.state_by_layer:
        raise ValueError(f"Unsupported morphology layer: {layer}")
    if state_name not in snapshot.state_by_layer[layer]:
        raise ValueError(f"Unsupported state name for {layer}: {state_name}")
    return np.asarray(snapshot.state_by_layer[layer][state_name], dtype=np.float32)


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return (arr - arr.mean(axis=1, keepdims=True)) / np.maximum(arr.std(axis=1, keepdims=True), EPS)


def _top_mask(values: np.ndarray, q: float, *, positive_only: bool = False, within_mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(arr.shape, dtype=bool)
    for row_idx in range(arr.shape[0]):
        allowed = np.ones(arr.shape[1], dtype=bool) if within_mask is None else np.asarray(within_mask[row_idx], dtype=bool).copy()
        if positive_only:
            allowed &= arr[row_idx] > 0.0
        idx = np.flatnonzero(allowed)
        if idx.size == 0:
            continue
        count = max(1, int(math.ceil(float(q) * idx.size)))
        ranked = idx[np.argsort(arr[row_idx, idx], kind="stable")]
        out[row_idx, ranked[-count:]] = True
    return out


def compute_layer3_peak_morphology(
    batch_df: pd.DataFrame,
    baseline: StspSnapshot,
    state_a: StspSnapshot,
    state_b: StspSnapshot,
    state_ab: StspSnapshot,
    state_mean: StspSnapshot,
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute Layer 3 shared-feature peak metrics for each pair."""
    g0 = layer_state_matrix(baseline, layer=cfg.morphology_layer, state_name=cfg.state_name)
    g_a = layer_state_matrix(state_a, layer=cfg.morphology_layer, state_name=cfg.state_name)
    g_b = layer_state_matrix(state_b, layer=cfg.morphology_layer, state_name=cfg.state_name)
    g_ab = layer_state_matrix(state_ab, layer=cfg.morphology_layer, state_name=cfg.state_name)
    g_mean = layer_state_matrix(state_mean, layer=cfg.morphology_layer, state_name=cfg.state_name)
    delta_a = g_a - g0
    delta_b = g_b - g0
    e_ab = g_ab - g_mean
    shared_score_z = np.minimum(_zscore_rows(delta_a), _zscore_rows(delta_b))
    if not np.isfinite(shared_score_z).all():
        shared_score_z = np.minimum(delta_a, delta_b)
    shared_mask = _top_mask(shared_score_z, cfg.shared_mask_top_q)
    peak_mask = _top_mask(e_ab, cfg.peak_top_q, positive_only=True)
    any_empty = np.asarray([not bool(row.any()) for row in peak_mask], dtype=bool)
    if bool(any_empty.any()):
        fallback_peak = _top_mask(e_ab, cfg.peak_top_q, positive_only=False)
        peak_mask[any_empty] = fallback_peak[any_empty]
    rows: list[dict[str, Any]] = []
    n = int(e_ab.shape[1])
    for row_idx, (_, pair_row) in enumerate(batch_df.reset_index(drop=True).iterrows()):
        shared = shared_mask[row_idx]
        peak = peak_mask[row_idx]
        nonshared = ~shared
        nonpeak = ~peak
        shared_count = int(shared.sum())
        peak_count = int(peak.sum())
        shared_peak_fraction = float((peak & shared).sum() / max(peak_count, 1))
        shared_base_fraction = float(shared_count / max(n, 1))
        rows.append(
            {
                "seed": int(pair_row["seed"]),
                "pair_id": pair_row["pair_id"],
                "overlap_group": pair_row["overlap_group"],
                "sample_label": int(pair_row["sample_label"]),
                "second_label": int(pair_row["second_label"]),
                "morphology_layer": cfg.morphology_layer,
                "state_name": cfg.state_name,
                "shared_peak_excess": float(np.mean(e_ab[row_idx, shared])) if shared_count else np.nan,
                "peak_enrichment": float(shared_peak_fraction / max(shared_base_fraction, EPS)) if peak_count and shared_count else np.nan,
                "peak_sharpness": float(np.mean(e_ab[row_idx, peak]) - np.mean(e_ab[row_idx, nonpeak])) if peak_count and int(nonpeak.sum()) else np.nan,
                "supra_mean_fraction": float(np.mean(e_ab[row_idx, shared] > EPS)) if shared_count else np.nan,
                "shared_positive_mass": float(np.maximum(e_ab[row_idx, shared], 0.0).sum()) if shared_count else 0.0,
                "nonshared_positive_mass": float(np.maximum(e_ab[row_idx, nonshared], 0.0).sum()) if int(nonshared.sum()) else 0.0,
                "shared_vs_nonshared_excess": float(np.mean(e_ab[row_idx, shared]) - np.mean(e_ab[row_idx, nonshared]))
                if shared_count and int(nonshared.sum())
                else np.nan,
                "shared_mask_count": shared_count,
                "peak_mask_count": peak_count,
                "shared_mask_mean_delta_A": float(np.mean(delta_a[row_idx, shared])) if shared_count else np.nan,
                "shared_mask_mean_delta_B": float(np.mean(delta_b[row_idx, shared])) if shared_count else np.nan,
                "shared_mask_mean_E_AB": float(np.mean(e_ab[row_idx, shared])) if shared_count else np.nan,
                "shared_mask_fallback": 0,
                "peak_mask_fallback": int(any_empty[row_idx]),
            }
        )
    return pd.DataFrame(rows), {
        "shared_mask": shared_mask,
        "peak_mask": peak_mask,
        "e_ab": e_ab,
        "g_ab": g_ab,
        "g_mean": g_mean,
    }


def replace_layer3_elements(
    source: StspSnapshot,
    donor: StspSnapshot,
    mask: np.ndarray,
) -> StspSnapshot:
    """Return source with only masked Layer 3 u/x replaced by donor u/x."""
    restore: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    state_by_layer: dict[str, dict[str, np.ndarray]] = {}
    batch_size = next(iter(source.restore_ux_by_layer.values()))[0].shape[0]
    for layer_key in LAYER_KEYS:
        src_u, src_x = source.restore_ux_by_layer[layer_key]
        out_u = src_u.detach().cpu().clone()
        out_x = src_x.detach().cpu().clone()
        if layer_key == "layer3":
            donor_u, donor_x = donor.restore_ux_by_layer[layer_key]
            mask_t = torch.as_tensor(mask, dtype=torch.bool).view(batch_size, -1)
            flat_u = out_u.view(batch_size, -1)
            flat_x = out_x.view(batch_size, -1)
            flat_donor_u = donor_u.view(batch_size, -1)
            flat_donor_x = donor_x.view(batch_size, -1)
            flat_u[mask_t] = flat_donor_u[mask_t]
            flat_x[mask_t] = flat_donor_x[mask_t]
        restore[layer_key] = (out_u, out_x)
        state_by_layer[layer_key] = {
            "u": out_u.view(batch_size, -1).numpy().astype(np.float32, copy=False),
            "x": out_x.view(batch_size, -1).numpy().astype(np.float32, copy=False),
            "g": (out_u * out_x).view(batch_size, -1).numpy().astype(np.float32, copy=False),
        }
    return StspSnapshot(layer_input_shapes=dict(source.layer_input_shapes), restore_ux_by_layer=restore, state_by_layer=state_by_layer)


def build_chunk_peak_removal_states(
    batch_df: pd.DataFrame,
    state_ab: StspSnapshot,
    state_mean: StspSnapshot,
    morphology_payload: Mapping[str, np.ndarray],
    cfg: ExperimentConfig,
    *,
    seed: int,
) -> tuple[dict[str, StspSnapshot], pd.DataFrame]:
    """Build chunk/random/nonshared Layer 3 peak-removal causal controls."""
    shared = np.asarray(morphology_payload["shared_mask"], dtype=bool)
    e_ab = np.asarray(morphology_payload["e_ab"], dtype=np.float64)
    g_ab = np.asarray(morphology_payload["g_ab"], dtype=np.float64)
    batch_size, n = e_ab.shape
    rng = np.random.default_rng(seed)
    chunk_mask = np.zeros_like(shared, dtype=bool)
    random_mask = np.zeros_like(shared, dtype=bool)
    nonshared_mask = np.zeros_like(shared, dtype=bool)
    rows: list[dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(batch_df.reset_index(drop=True).iterrows()):
        allowed_chunk = shared[row_idx] & (e_ab[row_idx] > 0.0)
        idx = np.flatnonzero(allowed_chunk)
        if idx.size == 0:
            idx = np.flatnonzero(shared[row_idx])
        if idx.size == 0:
            idx = np.arange(n)
        count = max(1, int(math.ceil(float(cfg.peak_top_q) * idx.size)))
        chunk_idx = idx[np.argsort(e_ab[row_idx, idx], kind="stable")[-count:]]
        chunk_mask[row_idx, chunk_idx] = True

        nonshared_idx = np.flatnonzero((~shared[row_idx]) & (e_ab[row_idx] > 0.0))
        if nonshared_idx.size < chunk_idx.size:
            nonshared_idx = np.flatnonzero(~shared[row_idx])
        if nonshared_idx.size:
            chosen = nonshared_idx[np.argsort(e_ab[row_idx, nonshared_idx], kind="stable")[-min(chunk_idx.size, nonshared_idx.size):]]
            nonshared_mask[row_idx, chosen] = True

        candidate = np.setdiff1d(np.arange(n), chunk_idx, assume_unique=False)
        if candidate.size:
            target_mean = float(np.mean(e_ab[row_idx, chunk_idx])) if chunk_idx.size else 0.0
            ranked = candidate[np.argsort(np.abs(e_ab[row_idx, candidate] - target_mean), kind="stable")]
            if str(cfg.probe_c_selection) == "random":
                chosen_random = rng.choice(candidate, size=min(chunk_idx.size, candidate.size), replace=False)
            else:
                chosen_random = ranked[: min(chunk_idx.size, ranked.size)]
            random_mask[row_idx, chosen_random] = True

        def mask_mean(mask_row: np.ndarray, values: np.ndarray) -> float:
            return float(np.mean(values[mask_row])) if bool(mask_row.any()) else np.nan

        rows.append(
            {
                "seed": int(row["seed"]),
                "pair_id": row["pair_id"],
                "overlap_group": row["overlap_group"],
                "chunk_mask_count": int(chunk_mask[row_idx].sum()),
                "random_mask_count": int(random_mask[row_idx].sum()),
                "nonshared_mask_count": int(nonshared_mask[row_idx].sum()),
                "chunk_mask_mean_E_AB": mask_mean(chunk_mask[row_idx], e_ab[row_idx]),
                "random_mask_mean_E_AB": mask_mean(random_mask[row_idx], e_ab[row_idx]),
                "nonshared_mask_mean_E_AB": mask_mean(nonshared_mask[row_idx], e_ab[row_idx]),
                "chunk_mask_mean_gAB": mask_mean(chunk_mask[row_idx], g_ab[row_idx]),
                "random_mask_mean_gAB": mask_mean(random_mask[row_idx], g_ab[row_idx]),
                "nonshared_mask_mean_gAB": mask_mean(nonshared_mask[row_idx], g_ab[row_idx]),
            }
        )
    return (
        {
            "S_AB_minus_chunk_peak": replace_layer3_elements(state_ab, state_mean, chunk_mask),
            "S_AB_minus_random_peak": replace_layer3_elements(state_ab, state_mean, random_mask),
            "S_AB_minus_nonshared_peak": replace_layer3_elements(state_ab, state_mean, nonshared_mask),
        },
        pd.DataFrame(rows),
    )


def build_morphology_rows(
    batch_df: pd.DataFrame,
    state_a: StspSnapshot,
    state_b: StspSnapshot,
    state_ab: StspSnapshot,
    state_add: StspSnapshot,
    *,
    seed: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    a = _combined_state_matrix(state_a)
    b = _combined_state_matrix(state_b)
    ab = _combined_state_matrix(state_ab)
    add = _combined_state_matrix(state_add)
    rows: list[dict[str, Any]] = []
    n = len(batch_df)
    perm = np.arange(n)
    if n > 1:
        perm = rng.permutation(n)
        if np.any(perm == np.arange(n)):
            perm = np.roll(perm, 1)
    for condition, matrix in (("S_AB", ab), ("S_A_plus_B_ls", add)):
        sim_a = centered_cosine_rows(matrix, a)
        sim_b = centered_cosine_rows(matrix, b)
        true_pair = pair_composite_similarity(matrix, a, b)
        shuffled = pair_composite_similarity(matrix, a, b[perm])
        wpri = true_pair - np.maximum(sim_a, sim_b)
        sim_ab = centered_cosine_rows(matrix, ab)
        for row_idx, (_, pair_row) in enumerate(batch_df.reset_index(drop=True).iterrows()):
            rows.append(
                {
                    "seed": int(seed),
                    "pair_id": pair_row["pair_id"],
                    "overlap_group": pair_row["overlap_group"],
                    "state_condition": condition,
                    "sim_to_A": float(sim_a[row_idx]),
                    "sim_to_B": float(sim_b[row_idx]),
                    "sim_to_AB": float(sim_ab[row_idx]),
                    "true_pair_score": float(true_pair[row_idx]),
                    "shuffled_pair_score": float(shuffled[row_idx]),
                    "true_minus_shuffled_pair_score": float(true_pair[row_idx] - shuffled[row_idx]),
                    "best_part_similarity": float(max(sim_a[row_idx], sim_b[row_idx])),
                    "WPRI": float(wpri[row_idx]),
                }
            )
    return pd.DataFrame(rows)


def run_readout_from_snapshot(
    net: Any,
    snapshot: StspSnapshot,
    *,
    mode: str,
    cfg: ExperimentConfig,
    probe_spikes: torch.Tensor | None = None,
    probe_scale: float = 1.0,
    probe_noise: float = 0.0,
    ping_amp: float | None = None,
    ping_noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = int(snapshot.layer_input_shapes["layer1"][0])
    with torch.no_grad():
        reset_all_state_restore_selected_stsp_in_place(net, snapshot.layer_input_shapes, snapshot.restore_ux_by_layer)
        net.layer3.reset_decision_state()
        zero_input = torch.zeros(snapshot.layer_input_shapes["layer1"], dtype=torch.float32, device=net.layer1.v_mem.device)
        gen = torch.Generator(device=zero_input.device)
        gen.manual_seed(int(seed))
        if mode == "ping":
            amp = float(cfg.ping_amp if ping_amp is None else ping_amp)
            for t_idx in range(int(cfg.ping_steps)):
                ping_drive = torch.full_like(zero_input, amp)
                if float(ping_noise) > 0.0:
                    ping_drive = torch.clamp(
                        ping_drive + torch.randn(ping_drive.shape, generator=gen, device=ping_drive.device) * float(ping_noise),
                        min=0.0,
                    )
                forward_three_layers(net, zero_input, t_idx, ping_drive=ping_drive)
        elif mode == "probe":
            if probe_spikes is None:
                raise ValueError("probe_spikes is required for probe readout.")
            for t_idx in range(int(probe_spikes.shape[1])):
                input_t = probe_spikes[:, t_idx].to(dtype=torch.float32) * float(probe_scale)
                if float(probe_noise) > 0.0:
                    input_t = torch.clamp(input_t + torch.randn(input_t.shape, generator=gen, device=input_t.device) * float(probe_noise), min=0.0)
                forward_three_layers(net, input_t, t_idx)
        else:
            raise ValueError(f"Unsupported readout mode: {mode}")
    pred, fire_t = decode_prediction_and_fire_time_from_layer3(net, batch_size=batch_size)
    return pred.numpy().astype(np.int64, copy=False), fire_t.numpy().astype(np.int64, copy=False)


def _classify_pair_prediction(pred: int, sample_label: int, second_label: int) -> dict[str, int]:
    return {
        "pred_A": int(pred == sample_label),
        "pred_B": int(pred == second_label),
        "pred_other": int(pred >= 0 and pred not in {sample_label, second_label}),
        "pred_silent": int(pred < 0),
    }


def build_ping_rows(batch_df: pd.DataFrame, predictions: Mapping[str, tuple[np.ndarray, np.ndarray]], *, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state_condition, (pred, fire_t) in predictions.items():
        for row_idx, (_, pair_row) in enumerate(batch_df.reset_index(drop=True).iterrows()):
            p = int(pred[row_idx])
            a = int(pair_row["sample_label"])
            b = int(pair_row["second_label"])
            hit_a = int(p == a)
            hit_b = int(p == b)
            member = int(hit_a or hit_b)
            balance = 1.0 if (hit_a + hit_b) == 1 else 0.0
            rows.append(
                {
                    "seed": int(seed),
                    "pair_id": pair_row["pair_id"],
                    "overlap_group": pair_row["overlap_group"],
                    "state_condition": state_condition,
                    "ping_pred": p,
                    "first_fire_t": int(fire_t[row_idx]),
                    "ping_A": hit_a,
                    "ping_B": hit_b,
                    "ping_other": int(p >= 0 and member == 0),
                    "ping_silent": int(p < 0),
                    "pair_member_readout_rate": member,
                    "old_item_readout_rate": hit_a,
                    "second_item_readout_rate": hit_b,
                    "dual_accessibility_index": float(member * balance),
                }
            )
    return pd.DataFrame(rows)


def run_ping_decomposed_rows(
    batch_df: pd.DataFrame,
    snapshots: Mapping[str, StspSnapshot],
    net: Any,
    cfg: ExperimentConfig,
    *,
    seed: int,
) -> pd.DataFrame:
    """Run stochastic neutral ping repeats and decompose A/B/other/silent."""
    rows: list[dict[str, Any]] = []
    for state_condition, snapshot in snapshots.items():
        for amp_idx, amp in enumerate(cfg.ping_amp_list):
            for repeat in range(int(cfg.ping_repeats)):
                pred, fire_t = run_readout_from_snapshot(
                    net,
                    snapshot,
                    mode="ping",
                    cfg=cfg,
                    ping_amp=float(amp),
                    ping_noise=float(cfg.ping_noise),
                    seed=mix_seed(seed, amp_idx, repeat, 811),
                )
                for row_idx, (_, pair_row) in enumerate(batch_df.reset_index(drop=True).iterrows()):
                    p = int(pred[row_idx])
                    a = int(pair_row["sample_label"])
                    b = int(pair_row["second_label"])
                    classified = _classify_pair_prediction(p, a, b)
                    rows.append(
                        {
                            "seed": int(seed),
                            "pair_id": pair_row["pair_id"],
                            "overlap_group": pair_row["overlap_group"],
                            "state_condition": state_condition,
                            "ping_repeat": int(repeat),
                            "ping_amp": float(amp),
                            "ping_pred": p,
                            "first_fire_t": int(fire_t[row_idx]),
                            "ping_A": classified["pred_A"],
                            "ping_B": classified["pred_B"],
                            "ping_other": classified["pred_other"],
                            "ping_silent": classified["pred_silent"],
                        }
                    )
    return pd.DataFrame(rows)


def summarize_ping_decomposed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = df.groupby(["seed", "pair_id", "overlap_group", "state_condition"], as_index=False)[
        ["ping_A", "ping_B", "ping_other", "ping_silent"]
    ].mean()
    summary = summary.rename(columns={"ping_A": "P_A", "ping_B": "P_B", "ping_other": "P_other", "ping_silent": "P_silent"})
    summary["P_pair"] = summary["P_A"] + summary["P_B"]
    summary["pair_balance"] = 1.0 - np.abs(summary["P_A"] - summary["P_B"])
    p_pair = summary["P_pair"].to_numpy(dtype=np.float64)
    p_a = summary["P_A"].to_numpy(dtype=np.float64)
    p_b = summary["P_B"].to_numpy(dtype=np.float64)
    entropy = np.full(len(summary), np.nan, dtype=np.float64)
    valid = p_pair > EPS
    pa_norm = np.zeros_like(p_pair)
    pb_norm = np.zeros_like(p_pair)
    pa_norm[valid] = p_a[valid] / p_pair[valid]
    pb_norm[valid] = p_b[valid] / p_pair[valid]
    for idx in np.flatnonzero(valid):
        probs = np.asarray([pa_norm[idx], pb_norm[idx]], dtype=np.float64)
        probs = probs[probs > 0.0]
        entropy[idx] = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0
    summary["AB_entropy"] = entropy
    piv = summary.pivot_table(index=["seed", "pair_id"], columns="state_condition", values=["P_A", "P_pair", "pair_balance"], aggfunc="mean")

    def lookup(metric: str, condition: str) -> pd.Series:
        if (metric, condition) not in piv:
            return pd.Series(np.nan, index=piv.index)
        return piv[(metric, condition)]

    extras = pd.DataFrame(index=piv.index)
    extras["old_item_rescue"] = lookup("P_A", "S_AB") - lookup("P_A", "S_B")
    extras["beyond_mean_old_rescue"] = lookup("P_A", "S_AB") - lookup("P_A", "S_mean")
    extras["peak_removal_delta_old"] = lookup("P_A", "S_AB") - lookup("P_A", "S_AB_minus_chunk_peak")
    extras["peak_removal_delta_pair"] = lookup("P_pair", "S_AB") - lookup("P_pair", "S_AB_minus_chunk_peak")
    extras["peak_removal_delta_balance"] = lookup("pair_balance", "S_AB") - lookup("pair_balance", "S_AB_minus_chunk_peak")
    return summary.merge(extras.reset_index(), on=["seed", "pair_id"], how="left")


def choose_probe_ids(
    batch_df: pd.DataFrame,
    class_index: Mapping[int, Sequence[int]],
    *,
    seed: int,
    num_probe_candidates: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    out: list[list[int]] = []
    classes = sorted(int(k) for k in class_index)
    for _, row in batch_df.iterrows():
        blocked = {int(row["sample_label"]), int(row["second_label"])}
        candidates: list[int] = []
        for label in rng.permutation([c for c in classes if c not in blocked]).tolist():
            pool = np.asarray(class_index[int(label)], dtype=np.int64)
            if pool.size == 0:
                continue
            candidates.append(int(rng.choice(pool)))
            if len(candidates) >= int(num_probe_candidates):
                break
        out.append(candidates)
    return out


def build_probe_rows(
    batch_df: pd.DataFrame,
    probe_ids_by_row: Sequence[Sequence[int]],
    state_snapshots: Mapping[str, StspSnapshot],
    images: torch.Tensor,
    labels: np.ndarray,
    encoder: Any,
    net: Any,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    flat_probe_ids = [int(v) for ids in probe_ids_by_row for v in ids]
    probe_lookup = _encode_id_bank(flat_probe_ids, images, encoder, steps=cfg.probe_steps, device=device, batch_size=cfg.batch_size)
    base_df = batch_df.reset_index(drop=True)
    for probe_rank in range(max((len(v) for v in probe_ids_by_row), default=0)):
        active_rows = [idx for idx, ids in enumerate(probe_ids_by_row) if probe_rank < len(ids)]
        if not active_rows:
            continue
        probe_batch = torch.zeros((len(base_df), cfg.probe_steps, *next(iter(probe_lookup.values())).shape[1:]), dtype=torch.float32, device=device)
        probe_label = np.full(len(base_df), -1, dtype=np.int64)
        probe_id_arr = np.full(len(base_df), -1, dtype=np.int64)
        for row_idx in active_rows:
            probe_id = int(probe_ids_by_row[row_idx][probe_rank])
            probe_batch[row_idx] = probe_lookup[probe_id]
            probe_label[row_idx] = int(labels[probe_id])
            probe_id_arr[row_idx] = probe_id
        for state_condition, snapshot in state_snapshots.items():
            pred, fire_t = run_readout_from_snapshot(
                net,
                snapshot,
                mode="probe",
                cfg=cfg,
                probe_spikes=probe_batch,
                probe_scale=cfg.probe_scale,
                probe_noise=cfg.probe_noise,
                seed=mix_seed(seed, probe_rank, 211),
            )
            for row_idx in active_rows:
                pair_row = base_df.iloc[row_idx]
                p = int(pred[row_idx])
                a = int(pair_row["sample_label"])
                b = int(pair_row["second_label"])
                c = int(probe_label[row_idx])
                is_error = int(p != c)
                within_pair_error = int(is_error and (p == a or p == b))
                rows.append(
                    {
                        "seed": int(seed),
                        "pair_id": pair_row["pair_id"],
                        "overlap_group": pair_row["overlap_group"],
                        "state_condition": state_condition,
                        "probe_rank": int(probe_rank),
                        "probe_idx": int(probe_id_arr[row_idx]),
                        "probe_label": c,
                        "probe_pred": p,
                        "first_fire_t": int(fire_t[row_idx]),
                        "correct_probe_prediction": int(p == c),
                        "error_to_A": int(p == a),
                        "error_to_B": int(p == b),
                        "error_to_other": int(is_error and p >= 0 and p not in {a, b}),
                        "no_spike": int(p < 0),
                        "is_error": is_error,
                        "within_pair_error": within_pair_error,
                        "within_pair_error_enrichment": float(within_pair_error) if is_error else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def run_probe_decomposed_rows(
    batch_df: pd.DataFrame,
    state_snapshots: Mapping[str, StspSnapshot],
    net: Any,
    cfg: ExperimentConfig,
    probe_spikes_by_rank: Sequence[torch.Tensor],
    probe_labels_by_rank: Sequence[np.ndarray],
    probe_ids_by_rank: Sequence[np.ndarray],
    *,
    probe_type: str,
    probe_scale: float,
    seed: int,
) -> pd.DataFrame:
    """Run weak-probe readout and decompose predictions by A/B/other/silent."""
    rows: list[dict[str, Any]] = []
    base_df = batch_df.reset_index(drop=True)
    for probe_rank, probe_batch in enumerate(probe_spikes_by_rank):
        probe_labels = np.asarray(probe_labels_by_rank[probe_rank], dtype=np.int64)
        probe_ids = np.asarray(probe_ids_by_rank[probe_rank], dtype=np.int64)
        for state_condition, snapshot in state_snapshots.items():
            pred, fire_t = run_readout_from_snapshot(
                net,
                snapshot,
                mode="probe",
                cfg=cfg,
                probe_spikes=probe_batch,
                probe_scale=float(probe_scale),
                probe_noise=float(cfg.probe_noise),
                seed=mix_seed(seed, probe_rank, 821),
            )
            for row_idx, (_, pair_row) in enumerate(base_df.iterrows()):
                p = int(pred[row_idx])
                a = int(pair_row["sample_label"])
                b = int(pair_row["second_label"])
                target = int(probe_labels[row_idx])
                classified = _classify_pair_prediction(p, a, b)
                row: dict[str, Any] = {
                    "seed": int(seed),
                    "pair_id": pair_row["pair_id"],
                    "overlap_group": pair_row["overlap_group"],
                    "state_condition": state_condition,
                    "probe_type": probe_type,
                    "probe_rank": int(probe_rank),
                    "probe_idx": int(probe_ids[row_idx]),
                    "probe_label": target,
                    "probe_pred": p,
                    "first_fire_t": int(fire_t[row_idx]),
                    "pred_A": classified["pred_A"],
                    "pred_B": classified["pred_B"],
                    "pred_other": classified["pred_other"],
                    "pred_silent": classified["pred_silent"],
                }
                if probe_type == "member_A":
                    row.update(
                        {
                            "A_correct": int(p == a),
                            "B_intrusion": int(p == b),
                            "pred_other_nonpair": classified["pred_other"],
                        }
                    )
                else:
                    row.update(
                        {
                            "correct_C": int(p == target),
                            "error_to_A": int(p == a),
                            "error_to_B": int(p == b),
                            "error_to_other": int(p >= 0 and p not in {a, b, target}),
                            "no_spike": int(p < 0),
                            "P_within_pair_error": int(p == a or p == b),
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_member_probe_a(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = df.groupby(["seed", "pair_id", "overlap_group", "state_condition"], as_index=False)[
        ["A_correct", "B_intrusion", "pred_other", "pred_silent"]
    ].mean()
    summary = summary.rename(
        columns={
            "A_correct": "P_A_under_probe_A",
            "B_intrusion": "B_intrusion_rate",
            "pred_other": "P_other",
            "pred_silent": "P_silent",
        }
    )
    piv = summary.pivot_table(index=["seed", "pair_id"], columns="state_condition", values="P_A_under_probe_A", aggfunc="mean")
    extras = pd.DataFrame(index=piv.index)
    extras["A_correct_delta_vs_S_B"] = piv.get("S_AB") - piv.get("S_B")
    extras["A_correct_delta_vs_S_mean"] = piv.get("S_AB") - piv.get("S_mean")
    extras["A_correct_drop_after_peak_removal"] = piv.get("S_AB") - piv.get("S_AB_minus_chunk_peak")
    return summary.merge(extras.reset_index(), on=["seed", "pair_id"], how="left")


def summarize_nonmember_probe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = df.groupby(["seed", "pair_id", "overlap_group", "state_condition"], as_index=False)[
        ["correct_C", "error_to_A", "error_to_B", "error_to_other", "no_spike", "P_within_pair_error"]
    ].mean()
    summary = summary.rename(
        columns={
            "correct_C": "P_C",
            "error_to_A": "P_error_A",
            "error_to_B": "P_error_B",
            "error_to_other": "P_error_other",
            "no_spike": "P_silent",
        }
    )
    baseline = summary[summary["state_condition"] == "baseline"][
        ["seed", "pair_id", "P_error_A", "P_error_B", "P_error_other", "P_within_pair_error"]
    ].rename(
        columns={
            "P_error_A": "baseline_P_error_A",
            "P_error_B": "baseline_P_error_B",
            "P_error_other": "baseline_P_error_other",
            "P_within_pair_error": "baseline_P_within_pair_error",
        }
    )
    out = summary.merge(baseline, on=["seed", "pair_id"], how="left")
    out["excess_error_A"] = out["P_error_A"] - out["baseline_P_error_A"]
    out["excess_error_B"] = out["P_error_B"] - out["baseline_P_error_B"]
    out["excess_error_other"] = out["P_error_other"] - out["baseline_P_error_other"]
    out["excess_within_pair_error"] = out["P_within_pair_error"] - out["baseline_P_within_pair_error"]
    piv = out.pivot_table(index=["seed", "pair_id"], columns="state_condition", values="excess_within_pair_error", aggfunc="mean")
    extras = pd.DataFrame(index=piv.index)
    extras["interaction_vs_mean"] = piv.get("S_AB") - piv.get("S_mean")
    extras["peak_removal_drop"] = piv.get("S_AB") - piv.get("S_AB_minus_chunk_peak")
    extras["random_removal_drop"] = piv.get("S_AB") - piv.get("S_AB_minus_random_peak")
    extras["nonshared_removal_drop"] = piv.get("S_AB") - piv.get("S_AB_minus_nonshared_peak")
    return out.merge(extras.reset_index(), on=["seed", "pair_id"], how="left")


def build_layer1_masks_for_batch(
    batch_df: pd.DataFrame,
    spike_lookup: Mapping[int, torch.Tensor],
    cfg: ExperimentConfig,
    *,
    seed: int,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    masks: dict[str, list[np.ndarray]] = {"overlap_write_blockade": [], "matched_nonoverlap_write_blockade": [], "random_write_blockade": []}
    rows: list[dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(batch_df.reset_index(drop=True).iterrows()):
        a_energy = spike_lookup[int(row["sample_idx"])].sum(dim=0).detach().cpu().numpy().reshape(-1)
        b_energy = spike_lookup[int(row["second_idx"])].sum(dim=0).detach().cpu().numpy().reshape(-1)
        n = int(a_energy.size)
        top_k = max(1, int(math.ceil(n * float(cfg.mask_top_q))))
        top_a = np.zeros(n, dtype=bool)
        top_b = np.zeros(n, dtype=bool)
        top_a[np.argsort(a_energy, kind="stable")[-top_k:]] = True
        top_b[np.argsort(b_energy, kind="stable")[-top_k:]] = True
        shared = top_a & top_b
        fallback = 0
        if int(shared.sum()) <= 0:
            shared[np.argmax(np.minimum(a_energy, b_energy))] = True
            fallback = 1
        k = int(shared.sum())
        shared_idx = np.flatnonzero(shared)
        outside = np.flatnonzero(~shared)
        shared_b_median = float(np.median(b_energy[shared_idx])) if shared_idx.size else 0.0
        matched = np.zeros(n, dtype=bool)
        if outside.size:
            order = outside[np.argsort(np.abs(b_energy[outside] - shared_b_median), kind="stable")]
            matched[order[: min(k, order.size)]] = True
        random_mask = np.zeros(n, dtype=bool)
        if outside.size:
            chosen = rng.choice(outside, size=min(k, outside.size), replace=False)
            random_mask[chosen] = True
        masks["overlap_write_blockade"].append(shared.reshape(spike_lookup[int(row["sample_idx"])].shape[1:]))
        masks["matched_nonoverlap_write_blockade"].append(matched.reshape(spike_lookup[int(row["sample_idx"])].shape[1:]))
        masks["random_write_blockade"].append(random_mask.reshape(spike_lookup[int(row["sample_idx"])].shape[1:]))
        rows.append(
            {
                "pair_id": row["pair_id"],
                "overlap_group": row["overlap_group"],
                "shared_mask_count": k,
                "matched_mask_count": int(matched.sum()),
                "random_mask_count": int(random_mask.sum()),
                "shared_mask_fallback": fallback,
                "shared_mean_A_energy": float(a_energy[shared].mean()) if k else np.nan,
                "shared_mean_B_energy": float(b_energy[shared].mean()) if k else np.nan,
            }
        )
    tensor_masks = {name: torch.as_tensor(np.stack(items, axis=0), dtype=torch.bool) for name, items in masks.items()}
    return tensor_masks, pd.DataFrame(rows)


def build_blockade_rows(
    batch_df: pd.DataFrame,
    state_a: StspSnapshot,
    state_b: StspSnapshot,
    normal_state: StspSnapshot,
    normal_b_pred: np.ndarray,
    normal_b_fire_t: np.ndarray,
    blockade_states: Mapping[str, tuple[StspSnapshot, np.ndarray, np.ndarray]],
    *,
    seed: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    base_rows: list[dict[str, Any]] = []
    a = _combined_state_matrix(state_a)
    b = _combined_state_matrix(state_b)
    normal = _combined_state_matrix(normal_state)
    n = len(batch_df)
    perm = rng.permutation(n) if n > 1 else np.arange(n)
    if n > 1 and np.any(perm == np.arange(n)):
        perm = np.roll(perm, 1)
    states: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {"normal_dynamic": (normal, normal_b_pred, normal_b_fire_t)}
    states.update({name: (_combined_state_matrix(snap), pred, fire_t) for name, (snap, pred, fire_t) in blockade_states.items()})
    for condition, (matrix, b_pred, b_fire_t) in states.items():
        sim_a = centered_cosine_rows(matrix, a)
        sim_b = centered_cosine_rows(matrix, b)
        true_pair = pair_composite_similarity(matrix, a, b)
        shuffled = pair_composite_similarity(matrix, a, b[perm])
        wpri = true_pair - np.maximum(sim_a, sim_b)
        for row_idx, (_, pair_row) in enumerate(batch_df.reset_index(drop=True).iterrows()):
            base_rows.append(
                {
                    "seed": int(seed),
                    "pair_id": pair_row["pair_id"],
                    "overlap_group": pair_row["overlap_group"],
                    "blockade_condition": condition,
                    "sim_to_A": float(sim_a[row_idx]),
                    "sim_to_B": float(sim_b[row_idx]),
                    "true_pair_score": float(true_pair[row_idx]),
                    "shuffled_pair_score": float(shuffled[row_idx]),
                    "WPRI": float(wpri[row_idx]),
                    "B_immediate_pred": int(b_pred[row_idx]) if len(b_pred) == n else -1,
                    "B_immediate_fire_t": int(b_fire_t[row_idx]) if len(b_fire_t) == n else -1,
                    "B_immediate_correct": int(len(b_pred) == n and int(b_pred[row_idx]) == int(pair_row["second_label"])),
                }
            )
    return pd.DataFrame(base_rows)


def _safe_rate(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df:
        return float("nan")
    values = df[column].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def summarize_outputs(
    peak_morphology: pd.DataFrame,
    ping_summary: pd.DataFrame,
    member_summary: pd.DataFrame,
    nonmember_summary: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, Any]:
    seed_peak = peak_morphology.groupby(["seed", "overlap_group"], as_index=False)[
        ["shared_peak_excess", "peak_enrichment", "peak_sharpness", "shared_vs_nonshared_excess"]
    ].mean()
    seed_ping = ping_summary.groupby(["seed", "overlap_group", "state_condition"], as_index=False)[
        ["P_A", "P_B", "P_other", "P_silent", "P_pair", "pair_balance", "AB_entropy"]
    ].mean()
    seed_member = member_summary.groupby(["seed", "overlap_group", "state_condition"], as_index=False)[
        ["P_A_under_probe_A", "B_intrusion_rate"]
    ].mean()
    seed_nonmember = nonmember_summary.groupby(["seed", "overlap_group", "state_condition"], as_index=False)[
        ["excess_within_pair_error", "P_within_pair_error", "excess_error_A", "excess_error_B"]
    ].mean()

    def comparison(table: pd.DataFrame, group_col: str, value_col: str, left: str, right: str, *, overlap_group: str | None = None) -> dict[str, Any]:
        sub = table.copy()
        if overlap_group is not None and "overlap_group" in sub:
            sub = sub[sub["overlap_group"] == overlap_group]
        piv = sub.pivot_table(index="seed", columns=group_col, values=value_col, aggfunc="mean")
        if left not in piv or right not in piv:
            return {"available": False}
        diff = (piv[left] - piv[right]).dropna().to_numpy(dtype=np.float64)
        if diff.size == 0:
            return {"available": False}
        ci = bootstrap_mean_ci(diff, n_boot=min(1000, max(100, 200 * diff.size)), seed=seed) if diff.size > 1 else (float(diff[0]), float(diff[0]))
        return {
            "available": True,
            "n_seeds": int(diff.size),
            "mean_diff": float(diff.mean()),
            "sem_diff": sem(diff),
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
            "cohen_dz": float(diff.mean() / max(diff.std(ddof=1), EPS)) if diff.size > 1 else None,
        }

    def grouped_mean_records(table: pd.DataFrame, group_cols: Sequence[str], value_col: str) -> list[dict[str, Any]]:
        if table.empty:
            return []
        out = table.groupby(list(group_cols), as_index=False)[value_col].mean()
        return out.to_dict(orient="records")

    def high_value_diff_from_pair_summary(table: pd.DataFrame, value_col: str, left: str, right: str) -> dict[str, Any]:
        return comparison(table, "state_condition", value_col, left, right, overlap_group="high")

    return {
        "seed_level": {
            "layer3_peak_morphology": seed_peak.to_dict(orient="records"),
            "ping": seed_ping.to_dict(orient="records"),
            "member_probe_A": seed_member.to_dict(orient="records"),
            "nonmember_probe_C": seed_nonmember.to_dict(orient="records"),
        },
        "comparisons": {
            "shared_peak_excess_high_vs_low": comparison(seed_peak, "overlap_group", "shared_peak_excess", "high", "low"),
            "peak_enrichment_high_vs_low": comparison(seed_peak, "overlap_group", "peak_enrichment", "high", "low"),
            "ping_old_rescue_high": high_value_diff_from_pair_summary(seed_ping, "P_A", "S_AB", "S_B"),
            "ping_beyond_mean_old_rescue_high": high_value_diff_from_pair_summary(seed_ping, "P_A", "S_AB", "S_mean"),
            "ping_peak_removal_drop_high": high_value_diff_from_pair_summary(seed_ping, "P_A", "S_AB", "S_AB_minus_chunk_peak"),
            "member_probe_A_delta_vs_S_mean_high": high_value_diff_from_pair_summary(seed_member, "P_A_under_probe_A", "S_AB", "S_mean"),
            "member_probe_A_peak_removal_drop_high": high_value_diff_from_pair_summary(seed_member, "P_A_under_probe_A", "S_AB", "S_AB_minus_chunk_peak"),
            "nonmember_excess_within_pair_vs_mean_high": high_value_diff_from_pair_summary(seed_nonmember, "excess_within_pair_error", "S_AB", "S_mean"),
            "nonmember_peak_removal_drop_high": high_value_diff_from_pair_summary(seed_nonmember, "excess_within_pair_error", "S_AB", "S_AB_minus_chunk_peak"),
            "random_vs_chunk_peak_removal_control": comparison(
                seed_ping.assign(
                    chunk_drop=np.nan,
                ),
                "state_condition",
                "P_A",
                "S_AB_minus_random_peak",
                "S_AB_minus_chunk_peak",
                overlap_group="high",
            ),
        },
        "across_seed_mean": {
            "shared_peak_excess": grouped_mean_records(seed_peak, ["overlap_group"], "shared_peak_excess"),
            "peak_enrichment": grouped_mean_records(seed_peak, ["overlap_group"], "peak_enrichment"),
            "ping_P_A": grouped_mean_records(seed_ping, ["overlap_group", "state_condition"], "P_A"),
            "ping_P_pair": grouped_mean_records(seed_ping, ["overlap_group", "state_condition"], "P_pair"),
            "member_probe_A": grouped_mean_records(seed_member, ["overlap_group", "state_condition"], "P_A_under_probe_A"),
            "nonmember_excess_within_pair_error": grouped_mean_records(seed_nonmember, ["overlap_group", "state_condition"], "excess_within_pair_error"),
        },
    }


def process_seed(
    *,
    seed: int,
    images: torch.Tensor,
    labels: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    net: Any,
    encoder: Any,
    cfg: ExperimentConfig,
    device: torch.device,
    layout: Any,
    logger: RunLogger,
) -> dict[str, pd.DataFrame]:
    seed_everything(seed)
    logger.log(f"[Seed {seed}] selecting high/low overlap pairs.")
    pair_df = build_pair_table_for_seed(seed=seed, images=images, labels=labels, class_index=class_index, net=net, encoder=encoder, cfg=cfg, device=device)
    all_ids = sorted(set(pair_df["sample_idx"].astype(int)).union(set(pair_df["second_idx"].astype(int))))
    spike_lookup = _encode_id_bank(all_ids, images, encoder, steps=cfg.sample_steps, device=device, batch_size=cfg.batch_size)

    legacy_morphology_rows: list[pd.DataFrame] = []
    additive_rows: list[pd.DataFrame] = []
    ls_additive_rows: list[pd.DataFrame] = []
    peak_morphology_rows: list[pd.DataFrame] = []
    removal_mask_rows: list[pd.DataFrame] = []
    ping_decomposed_rows: list[pd.DataFrame] = []
    ping_summary_rows: list[pd.DataFrame] = []
    member_probe_rows: list[pd.DataFrame] = []
    member_probe_summary_rows: list[pd.DataFrame] = []
    nonmember_probe_rows: list[pd.DataFrame] = []
    nonmember_probe_summary_rows: list[pd.DataFrame] = []
    mask_rows: list[pd.DataFrame] = []
    state_npz_payload: dict[str, np.ndarray] = {}

    for batch_start in range(0, len(pair_df), max(1, int(cfg.batch_size))):
        batch_df = pair_df.iloc[batch_start : batch_start + max(1, int(cfg.batch_size))].reset_index(drop=True)
        sample_spikes = torch.stack([spike_lookup[int(v)] for v in batch_df["sample_idx"].astype(int)], dim=0).to(device)
        second_spikes = torch.stack([spike_lookup[int(v)] for v in batch_df["second_idx"].astype(int)], dim=0).to(device)
        logger.log(f"[Seed {seed}] batch {batch_start // max(1, int(cfg.batch_size)) + 1}: {len(batch_df)} pair(s).")
        state0, _, _ = run_state_capture(net, sample_spikes, second_spikes, cfg, active_a=False, active_b=False)
        state_a, _, _ = run_state_capture(net, sample_spikes, second_spikes, cfg, active_a=True, active_b=False)
        state_b, _, _ = run_state_capture(net, sample_spikes, second_spikes, cfg, active_a=False, active_b=True)
        state_ab, _b_pred_normal, _b_fire_normal = run_state_capture(net, sample_spikes, second_spikes, cfg, active_a=True, active_b=True)
        mean_mode = "sum" if str(cfg.additive_mode) == "sum" else "mean"
        state_mean, additive_diag = build_mean_mixture_snapshot(state0, state_a, state_b, mode=mean_mode)
        additive_diag.insert(0, "seed", int(seed))
        additive_diag.insert(1, "pair_id", additive_diag["row_idx"].map(lambda idx: batch_df["pair_id"].iloc[int(idx)]))
        additive_diag["overlap_group"] = additive_diag["row_idx"].map(lambda idx: batch_df["overlap_group"].iloc[int(idx)])
        a_l3 = layer_state_matrix(state_a)
        b_l3 = layer_state_matrix(state_b)
        ab_l3 = layer_state_matrix(state_ab)
        mean_l3 = layer_state_matrix(state_mean)
        sim_to_a = centered_cosine_rows(mean_l3, a_l3)
        sim_to_b = centered_cosine_rows(mean_l3, b_l3)
        sim_to_ab = centered_cosine_rows(mean_l3, ab_l3)
        additive_diag["sim_to_A"] = additive_diag["row_idx"].map(lambda idx: float(sim_to_a[int(idx)]))
        additive_diag["sim_to_B"] = additive_diag["row_idx"].map(lambda idx: float(sim_to_b[int(idx)]))
        additive_diag["sim_to_AB"] = additive_diag["row_idx"].map(lambda idx: float(sim_to_ab[int(idx)]))
        additive_rows.append(additive_diag)

        if cfg.use_ls_additive:
            state_ls, ls_diag = fit_additive_snapshot(state0, state_a, state_b, state_ab)
            ls_diag.insert(0, "seed", int(seed))
            ls_diag.insert(1, "pair_id", ls_diag["row_idx"].map(lambda idx: batch_df["pair_id"].iloc[int(idx)]))
            ls_additive_rows.append(ls_diag)
            legacy_morphology_rows.append(build_morphology_rows(batch_df, state_a, state_b, state_ab, state_ls, seed=seed, rng=np.random.default_rng(mix_seed(seed, batch_start, 401))))

        peak_df, morph_payload = compute_layer3_peak_morphology(batch_df, state0, state_a, state_b, state_ab, state_mean, cfg)
        peak_morphology_rows.append(peak_df)
        removal_states, removal_df = build_chunk_peak_removal_states(
            batch_df,
            state_ab,
            state_mean,
            morph_payload,
            cfg,
            seed=mix_seed(seed, batch_start, 701),
        )
        removal_mask_rows.append(removal_df)

        state_map = {
            "baseline": state0,
            "S_A": state_a,
            "S_B": state_b,
            "S_mean": state_mean,
            "S_AB": state_ab,
            **removal_states,
        }
        ping_df = run_ping_decomposed_rows(batch_df, state_map, net, cfg, seed=mix_seed(seed, batch_start, 801))
        ping_decomposed_rows.append(ping_df)
        ping_summary_rows.append(summarize_ping_decomposed(ping_df))

        member_states = {name: state_map[name] for name in ("baseline", "S_B", "S_mean", "S_AB", "S_AB_minus_chunk_peak", "S_AB_minus_random_peak", "S_AB_minus_nonshared_peak")}
        member_probe = sample_spikes[:, : cfg.probe_steps].clone()
        if member_probe.shape[1] < cfg.probe_steps:
            pad = torch.zeros(
                (member_probe.shape[0], cfg.probe_steps - member_probe.shape[1], *member_probe.shape[2:]),
                dtype=member_probe.dtype,
                device=member_probe.device,
            )
            member_probe = torch.cat([member_probe, pad], dim=1)
        member_labels = batch_df["sample_label"].to_numpy(dtype=np.int64)
        member_ids = batch_df["sample_idx"].to_numpy(dtype=np.int64)
        member_df = run_probe_decomposed_rows(
            batch_df,
            member_states,
            net,
            cfg,
            [member_probe],
            [member_labels],
            [member_ids],
            probe_type="member_A",
            probe_scale=cfg.member_probe_scale,
            seed=mix_seed(seed, batch_start, 901),
        )
        member_probe_rows.append(member_df)
        member_probe_summary_rows.append(summarize_member_probe_a(member_df))

        probe_ids = choose_probe_ids(batch_df, class_index, seed=mix_seed(seed, batch_start, 501), num_probe_candidates=cfg.num_nonmember_probes)
        flat_probe_ids = [int(v) for ids in probe_ids for v in ids]
        probe_lookup = _encode_id_bank(flat_probe_ids, images, encoder, steps=cfg.probe_steps, device=device, batch_size=cfg.batch_size)
        probe_spikes_by_rank: list[torch.Tensor] = []
        probe_labels_by_rank: list[np.ndarray] = []
        probe_ids_by_rank: list[np.ndarray] = []
        for probe_rank in range(int(cfg.num_nonmember_probes)):
            probe_batch = torch.zeros((len(batch_df), cfg.probe_steps, *member_probe.shape[2:]), dtype=torch.float32, device=device)
            probe_label = np.full(len(batch_df), -1, dtype=np.int64)
            probe_id_arr = np.full(len(batch_df), -1, dtype=np.int64)
            for row_idx, ids in enumerate(probe_ids):
                probe_id = int(ids[min(probe_rank, len(ids) - 1)])
                probe_batch[row_idx] = probe_lookup[probe_id]
                probe_label[row_idx] = int(labels[probe_id])
                probe_id_arr[row_idx] = probe_id
            probe_spikes_by_rank.append(probe_batch)
            probe_labels_by_rank.append(probe_label)
            probe_ids_by_rank.append(probe_id_arr)
        nonmember_states = {name: state_map[name] for name in ("baseline", "S_B", "S_mean", "S_AB", "S_AB_minus_chunk_peak", "S_AB_minus_random_peak", "S_AB_minus_nonshared_peak")}
        nonmember_df = run_probe_decomposed_rows(
            batch_df,
            nonmember_states,
            net,
            cfg,
            probe_spikes_by_rank,
            probe_labels_by_rank,
            probe_ids_by_rank,
            probe_type="nonmember_C",
            probe_scale=cfg.probe_scale,
            seed=mix_seed(seed, batch_start, 1001),
        )
        nonmember_probe_rows.append(nonmember_df)
        nonmember_probe_summary_rows.append(summarize_nonmember_probe(nonmember_df))

        if cfg.save_states:
            for condition, snap in {"S0": state0, "S_A": state_a, "S_B": state_b, "S_mean": state_mean, "S_AB": state_ab, **removal_states}.items():
                state_npz_payload[f"batch{batch_start}_{condition}_layer3_g"] = layer_state_matrix(snap)
            state_npz_payload[f"batch{batch_start}_shared_mask"] = morph_payload["shared_mask"].astype(np.uint8, copy=False)
            state_npz_payload[f"batch{batch_start}_peak_mask"] = morph_payload["peak_mask"].astype(np.uint8, copy=False)

    if cfg.save_states and state_npz_payload:
        np.savez_compressed(layout.data_file(f"states_masks_seed{int(seed)}.npz"), **state_npz_payload)

    return {
        "pair": pair_df,
        "legacy_morphology": pd.concat(legacy_morphology_rows, ignore_index=True) if legacy_morphology_rows else pd.DataFrame(),
        "additive": pd.concat(additive_rows, ignore_index=True) if additive_rows else pd.DataFrame(),
        "ls_additive": pd.concat(ls_additive_rows, ignore_index=True) if ls_additive_rows else pd.DataFrame(),
        "peak_morphology": pd.concat(peak_morphology_rows, ignore_index=True) if peak_morphology_rows else pd.DataFrame(),
        "removal_masks": pd.concat(removal_mask_rows, ignore_index=True) if removal_mask_rows else pd.DataFrame(),
        "ping_decomposed": pd.concat(ping_decomposed_rows, ignore_index=True) if ping_decomposed_rows else pd.DataFrame(),
        "ping_summary": pd.concat(ping_summary_rows, ignore_index=True) if ping_summary_rows else pd.DataFrame(),
        "member_probe": pd.concat(member_probe_rows, ignore_index=True) if member_probe_rows else pd.DataFrame(),
        "member_probe_summary": pd.concat(member_probe_summary_rows, ignore_index=True) if member_probe_summary_rows else pd.DataFrame(),
        "nonmember_probe": pd.concat(nonmember_probe_rows, ignore_index=True) if nonmember_probe_rows else pd.DataFrame(),
        "nonmember_probe_summary": pd.concat(nonmember_probe_summary_rows, ignore_index=True) if nonmember_probe_summary_rows else pd.DataFrame(),
        "mask": pd.concat(mask_rows, ignore_index=True) if mask_rows else pd.DataFrame(),
    }


def write_artifact_manifest(layout: Any, extra: Mapping[str, Any] | None = None) -> Path:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "files": _relativize_files(layout.root),
    }
    if extra:
        payload.update(dict(extra))
    path = layout.root_file("artifact_manifest.json")
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    cfg = build_config(parser.parse_args(argv))
    layout = prepare_result_layout(cfg.output_dir)
    logger = RunLogger(lines=[])
    run_info = build_run_info(
        experiment_name=EXPERIMENT_ID,
        output_dir=layout.root,
        entry_script=f"python -m src.experiments.{EXPERIMENT_ID}",
        seed=int(cfg.seeds[0]) if len(cfg.seeds) == 1 else None,
        dataset=str(cfg.dataset_root),
        command=subprocess.list2cmdline(sys.argv),
        model_path=str(cfg.model_path),
    )
    write_run_info(layout.meta_dir, run_info)
    status = "failed"
    try:
        logger.log(f"[Start] {EXPERIMENT_ID} | seeds={list(cfg.seeds)} | smoke={cfg.smoke_test}")
        device = resolve_device(cfg.device)
        logger.log(f"[Runtime] device={device}")
        seed_everything(int(cfg.seeds[0]))
        dataset = load_mnist_skeleton_dataset(str(cfg.dataset_root), split=cfg.dataset_split)
        images, labels, _ = build_dataset_arrays(dataset)
        class_index = build_class_index(dataset, num_classes=10)
        net, encoder = load_model_and_encoder(cfg.model_path, device=device, dt=1.0 * ms, max_duration_ms=cfg.encoder_max_duration_ms)
        output_names = (
            "pair",
            "legacy_morphology",
            "additive",
            "ls_additive",
            "peak_morphology",
            "removal_masks",
            "ping_decomposed",
            "ping_summary",
            "member_probe",
            "member_probe_summary",
            "nonmember_probe",
            "nonmember_probe_summary",
            "mask",
        )
        all_outputs: dict[str, list[pd.DataFrame]] = {name: [] for name in output_names}
        for seed in cfg.seeds:
            seed_outputs = process_seed(
                seed=int(seed),
                images=images,
                labels=labels,
                class_index=class_index,
                net=net,
                encoder=encoder,
                cfg=cfg,
                device=device,
                layout=layout,
                logger=logger,
            )
            for name, df in seed_outputs.items():
                all_outputs[name].append(df)

        pair_df = pd.concat(all_outputs["pair"], ignore_index=True)
        morphology_df = pd.concat(all_outputs["legacy_morphology"], ignore_index=True) if any(not df.empty for df in all_outputs["legacy_morphology"]) else pd.DataFrame()
        additive_df = pd.concat(all_outputs["additive"], ignore_index=True)
        ls_additive_df = pd.concat(all_outputs["ls_additive"], ignore_index=True) if any(not df.empty for df in all_outputs["ls_additive"]) else pd.DataFrame()
        peak_morphology_df = pd.concat(all_outputs["peak_morphology"], ignore_index=True)
        removal_mask_df = pd.concat(all_outputs["removal_masks"], ignore_index=True)
        ping_decomposed_df = pd.concat(all_outputs["ping_decomposed"], ignore_index=True)
        ping_summary_df = pd.concat(all_outputs["ping_summary"], ignore_index=True)
        member_probe_df = pd.concat(all_outputs["member_probe"], ignore_index=True)
        member_probe_summary_df = pd.concat(all_outputs["member_probe_summary"], ignore_index=True)
        nonmember_probe_df = pd.concat(all_outputs["nonmember_probe"], ignore_index=True)
        nonmember_probe_summary_df = pd.concat(all_outputs["nonmember_probe_summary"], ignore_index=True)
        mask_df = pd.concat(all_outputs["mask"], ignore_index=True)
        exported = {
            "pair_table": save_tidy_csv(pair_df, layout.data_file("pair_table.csv"), sort_by=["seed", "overlap_group", "selection_rank"]),
            "mean_mixture_diagnostics": save_tidy_csv(additive_df, layout.data_file("mean_mixture_diagnostics.csv"), sort_by=["seed", "pair_id", "layer"]),
            "layer3_peak_morphology_metrics": save_tidy_csv(
                peak_morphology_df,
                layout.data_file("layer3_peak_morphology_metrics.csv"),
                sort_by=["seed", "pair_id"],
            ),
            "chunk_peak_removal_mask_stats": save_tidy_csv(
                removal_mask_df,
                layout.data_file("chunk_peak_removal_mask_stats.csv"),
                sort_by=["seed", "pair_id"],
            ),
            "ping_decomposed_metrics": save_tidy_csv(
                ping_decomposed_df,
                layout.data_file("ping_decomposed_metrics.csv"),
                sort_by=["seed", "pair_id", "state_condition", "ping_amp", "ping_repeat"],
            ),
            "ping_decomposed_summary": save_tidy_csv(
                ping_summary_df,
                layout.data_file("ping_decomposed_summary.csv"),
                sort_by=["seed", "pair_id", "state_condition"],
            ),
            "member_probe_A_metrics": save_tidy_csv(
                member_probe_df,
                layout.data_file("member_probe_A_metrics.csv"),
                sort_by=["seed", "pair_id", "state_condition", "probe_rank"],
            ),
            "member_probe_A_summary": save_tidy_csv(
                member_probe_summary_df,
                layout.data_file("member_probe_A_summary.csv"),
                sort_by=["seed", "pair_id", "state_condition"],
            ),
            "nonmember_probe_error_metrics": save_tidy_csv(
                nonmember_probe_df,
                layout.data_file("nonmember_probe_error_metrics.csv"),
                sort_by=["seed", "pair_id", "state_condition", "probe_rank"],
            ),
            "nonmember_probe_error_summary": save_tidy_csv(
                nonmember_probe_summary_df,
                layout.data_file("nonmember_probe_error_summary.csv"),
                sort_by=["seed", "pair_id", "state_condition"],
            ),
        }
        if not morphology_df.empty:
            exported["supplementary_ls_morphology_metrics"] = save_tidy_csv(
                morphology_df,
                layout.data_file("supplementary_ls_morphology_metrics.csv"),
                sort_by=["seed", "pair_id", "state_condition"],
            )
        if not ls_additive_df.empty:
            exported["supplementary_ls_additive_fit_diagnostics"] = save_tidy_csv(
                ls_additive_df,
                layout.data_file("supplementary_ls_additive_fit_diagnostics.csv"),
                sort_by=["seed", "pair_id", "layer", "variable"],
            )
        if not mask_df.empty:
            exported["legacy_layer1_mask_statistics"] = save_tidy_csv(mask_df, layout.data_file("legacy_layer1_mask_statistics.csv"), sort_by=["seed", "pair_id"])
        for name, path in exported.items():
            source = Path(path)
            target = layout.metrics_file(source.name) if source.suffix.lower() == ".csv" else layout.metrics_file(f"{name}.json")
            if source.suffix.lower() == ".csv":
                pd.read_csv(source).to_csv(target, index=False)

        summary_metrics = summarize_outputs(
            peak_morphology_df,
            ping_summary_df,
            member_probe_summary_df,
            nonmember_probe_summary_df,
            seed=mix_seed(int(cfg.seeds[0]), 809),
        )
        summary_metrics_path = save_summary_json(summary_metrics, layout.data_dir, filename="summary_metrics.json")
        save_summary_json(summary_metrics, layout.metrics_dir, filename="summary_metrics.json")
        save_summary_json(summary_metrics, layout.root, filename="summary_metrics.json")
        summary = {
            "experiment_id": EXPERIMENT_ID,
            "status": "success",
            "config": cfg.to_json_dict(),
            "num_pairs": int(len(pair_df)),
            "num_seeds": int(len(cfg.seeds)),
            "exported_files": _json_safe({**exported, "summary_metrics": str(summary_metrics_path)}),
            "key_comparisons": summary_metrics.get("comparisons", {}),
        }
        save_run_config(cfg.to_json_dict(), layout.root)
        save_summary_json(summary, layout.root)
        save_log_lines(logger.lines, layout.logs_dir)
        write_artifact_manifest(layout, extra={"exported_files": exported})
        status = "success"
        logger.log("[Done] Fig. 4 chunk interaction assay completed.")
        return 0
    finally:
        finalize_run_info(layout.meta_dir, run_info, status=status)


if __name__ == "__main__":
    raise SystemExit(main())
