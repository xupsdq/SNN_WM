from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
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
from src.experiments.common.distractor_triplets import build_triplet_specs as shared_build_triplet_specs
from src.experiments.common.distractor_triplets import load_mnist_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import build_layer_input_shapes
from src.experiments.common.monitored_dms import reset_all_state_restore_selected_stsp_in_place
from src.experiments.common.monitored_dms import snapshot_boundary_state
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.experiments.common.runtime import seed_everything
from src.experiments.common.seed import mix_seed
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_tidy_csv
from src.plotting.common.style import DYNAMIC_COLOR, NOISE_COLOR, SAMPLE_COLOR, SHUFFLE_COLOR, STATIC_COLOR

PING_CONDITIONS = ("dynamic_ping", "dynamic_no_ping", "static_ping", "shuffle_ux_ping", "ping_only")
CUE_CONDITIONS = ("cue_SP", "cue_DP", "cue_SDP")
CHANCE_BASELINE = 0.1
LAYER_KEYS = ("layer1", "layer2", "layer3")


@dataclass(frozen=True)
class Step5Config:
    model_path: str
    dataset_root: str
    split: str
    device: str
    seed: int
    output_dir: str
    sample_ms: float
    delay1_ms: float
    distractor_ms: float
    delay2_ms: float
    ping_ms: float
    ping_amp: float
    batch_size: int
    max_probes: int
    samples_per_probe: int
    max_triplets: int
    num_sim_bins: int
    skip_figures: bool
    smoke: bool
    dt: float = 1.0 * ms

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def delay1_steps(self) -> int:
        return int(round((self.delay1_ms * ms) / self.dt))

    @property
    def distractor_steps(self) -> int:
        return int(round((self.distractor_ms * ms) / self.dt))

    @property
    def delay2_steps(self) -> int:
        return int(round((self.delay2_ms * ms) / self.dt))

    @property
    def ping_steps(self) -> int:
        return int(round((self.ping_ms * ms) / self.dt))

    @property
    def max_duration_ms(self) -> float:
        return max(self.sample_ms, self.distractor_ms, self.ping_ms, 100.0)


@dataclass
class TripletBatch:
    batch_id: int
    df: pd.DataFrame
    sample_spikes: torch.Tensor
    distractor_spikes: torch.Tensor
    wrong_distractor_spikes: torch.Tensor
    zero_sample_spikes: torch.Tensor
    zero_distractor_spikes: torch.Tensor
    cue_spikes: Dict[str, torch.Tensor]


@dataclass
class BackboneSnapshot:
    condition: str
    layer_input_shapes: Dict[str, tuple[int, ...]]
    restore_ux_by_layer: Dict[str, tuple[torch.Tensor, torch.Tensor]]
    template_vector: np.ndarray


@dataclass
class BranchReadout:
    condition: str
    branch_name: str
    readout_vectors: np.ndarray
    class_scores_time: np.ndarray
    pair_decode_time: np.ndarray
    top1_member_time: np.ndarray
    first_fire_pred: np.ndarray
    first_fire_t: np.ndarray
    first_fire_hit: np.ndarray
    silent_mask: np.ndarray
    layer1_pattern_vector: Optional[np.ndarray] = None
    layer2_pattern_vector: Optional[np.ndarray] = None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 5 chunk holistic reactivation experiment.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_PATH_CONFIG.results_root / "chunk_step5_holistic_reactivation"))
    parser.add_argument("--sample-ms", type=float, default=50.0)
    parser.add_argument("--delay1-ms", type=float, default=50.0)
    parser.add_argument("--distractor-ms", type=float, default=50.0)
    parser.add_argument("--delay2-ms", type=float, default=50.0)
    parser.add_argument("--ping-ms", type=float, default=30.0)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-probes", type=int, default=20)
    parser.add_argument("--samples-per-probe", type=int, default=25)
    parser.add_argument("--max-triplets", type=int, default=500)
    parser.add_argument("--num-sim-bins", type=int, default=3)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def normalize_config(args: argparse.Namespace) -> Step5Config:
    cfg = Step5Config(**{k: getattr(args, k) for k in Step5Config.__dataclass_fields__.keys() if hasattr(args, k)})
    if cfg.smoke:
        cfg = Step5Config(**{**asdict(cfg), "batch_size": min(cfg.batch_size, 2), "max_probes": min(cfg.max_probes, 2), "samples_per_probe": min(cfg.samples_per_probe, 1), "max_triplets": min(cfg.max_triplets, 4)})
    if min(cfg.sample_steps, cfg.delay1_steps, cfg.distractor_steps, cfg.delay2_steps, cfg.ping_steps) <= 0:
        raise ValueError("All durations must map to at least one step.")
    return cfg


def resolve_device_with_fallback(device_arg: str) -> tuple[torch.device, str]:
    if str(device_arg).strip().lower() == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu"), "[Runtime] CUDA unavailable on 2026-04-16; falling back to CPU."
    return torch.device(str(device_arg)), f"[Runtime] Using device={device_arg}."


def log_and_print(log_lines: list[str], message: str) -> None:
    print(message, flush=True)
    log_lines.append(message)


def assign_wrong_distractors(df_triplets: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df_triplets.copy().reset_index(drop=True)
    pool = df[["distractor_id", "distractor_label"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out_rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        mask = (
            (pool["distractor_id"] != int(row.distractor_id))
            & (pool["distractor_id"] != int(row.sample_id))
            & (pool["distractor_id"] != int(row.probe_id))
            & (pool["distractor_label"] != int(row.sample_label))
            & (pool["distractor_label"] != int(row.probe_label))
        )
        valid = pool[mask].reset_index(drop=True)
        if valid.empty:
            raise RuntimeError("Failed to assign wrong distractor.")
        picked = valid.iloc[int(rng.integers(0, len(valid)))]
        out_rows.append({"wrong_distractor_id": int(picked["distractor_id"]), "wrong_distractor_label": int(picked["distractor_label"])})
    out = pd.concat([df, pd.DataFrame(out_rows)], axis=1)
    out["sample_label_name"] = out["sample_label"].astype(str)
    out["distractor_label_name"] = out["distractor_label"].astype(str)
    out["probe_label_name"] = out["probe_label"].astype(str)
    out["wrong_distractor_label_name"] = out["wrong_distractor_label"].astype(str)
    return out


def build_triplet_specs(images: torch.Tensor, labels: np.ndarray, flat_normalized: np.ndarray, class_index: Mapping[int, Sequence[int]], cfg: Step5Config) -> pd.DataFrame:
    df = shared_build_triplet_specs(images=images, labels=labels, flat_normalized=flat_normalized, class_index=class_index, max_probes=cfg.max_probes, samples_per_probe=cfg.samples_per_probe, num_bins=cfg.num_sim_bins, max_triplets=cfg.max_triplets, seed=cfg.seed)
    df = assign_wrong_distractors(df, seed=mix_seed(cfg.seed, 401))
    wrong_vec = flat_normalized[df["wrong_distractor_id"].to_numpy(dtype=np.int64)]
    probe_vec = flat_normalized[df["probe_id"].to_numpy(dtype=np.int64)]
    sample_vec = flat_normalized[df["sample_id"].to_numpy(dtype=np.int64)]
    df["wrong_dp_similarity"] = np.sum(wrong_vec * probe_vec, axis=1)
    df["wrong_sd_similarity"] = np.sum(wrong_vec * sample_vec, axis=1)
    df["selection_metadata"] = "step5_chunk_triplets"
    return df


def build_spike_lookup(images: torch.Tensor, encoder, ids: Sequence[int], *, steps: int, device: torch.device) -> dict[int, torch.Tensor]:
    unique_ids = sorted({int(idx) for idx in ids})
    spike_bank = encode_images(encoder, images[unique_ids].to(device=device, dtype=torch.float32), steps=steps)
    return {img_id: spike_bank[pos] for pos, img_id in enumerate(unique_ids)}


def build_constituent_region_masks(sample_images: torch.Tensor, distractor_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sample_norm = sample_images / sample_images.flatten(start_dim=1).amax(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
    distractor_norm = distractor_images / distractor_images.flatten(start_dim=1).amax(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
    # Cue masks follow each constituent's own occupied pixel support directly.
    sample_mask = sample_norm > 1e-12
    distractor_mask = distractor_norm > 1e-12
    return sample_mask.to(dtype=torch.float32), distractor_mask.to(dtype=torch.float32)


def build_part_cue_images(sample_images: torch.Tensor, distractor_images: torch.Tensor) -> dict[str, torch.Tensor]:
    sample_mask, distractor_mask = build_constituent_region_masks(sample_images, distractor_images)
    sample_only_mask = sample_mask * (1.0 - distractor_mask)
    distractor_only_mask = distractor_mask * (1.0 - sample_mask)
    shared_mask = sample_mask * distractor_mask
    shared_foreground = torch.maximum(sample_images, distractor_images) * shared_mask

    def _norm(x: torch.Tensor) -> torch.Tensor:
        return x / x.flatten(start_dim=1).amax(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)

    return {
        "cue_SP": _norm(sample_images * sample_only_mask),
        "cue_DP": _norm(distractor_images * distractor_only_mask),
        "cue_SDP": _norm(shared_foreground),
    }


def trim_or_pad_spike_sequence(spikes: torch.Tensor, target_steps: int) -> torch.Tensor:
    if int(spikes.shape[1]) == int(target_steps):
        return spikes
    if int(spikes.shape[1]) > int(target_steps):
        return spikes[:, : int(target_steps), ...].contiguous()
    pad_shape = (int(spikes.shape[0]), int(target_steps) - int(spikes.shape[1]), int(spikes.shape[2]), int(spikes.shape[3]), int(spikes.shape[4]))
    padding = torch.zeros(pad_shape, dtype=spikes.dtype, device=spikes.device)
    return torch.cat([spikes, padding], dim=1)


def combine_spike_sequences(*sequences: torch.Tensor) -> torch.Tensor:
    base = torch.zeros_like(sequences[0])
    for seq in sequences:
        base = torch.maximum(base, seq)
    return base


def pooled_spike_pattern_vector(early_counts: torch.Tensor, late_counts: torch.Tensor, *, output_size: tuple[int, int]) -> np.ndarray:
    early_pooled = F.adaptive_avg_pool2d(early_counts, output_size=output_size)
    late_pooled = F.adaptive_avg_pool2d(late_counts, output_size=output_size)
    vector = torch.cat([early_pooled.flatten(start_dim=1), late_pooled.flatten(start_dim=1)], dim=1)
    return vector.detach().cpu().numpy().astype(np.float32, copy=False)


def build_spike_pattern_summary(
    layer1_steps: Sequence[torch.Tensor],
    layer2_steps: Sequence[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    if len(layer1_steps) == 0 or len(layer2_steps) == 0:
        raise ValueError("Spike pattern summary requires at least one time step.")
    split = max(1, len(layer1_steps) // 2)
    layer1_early = torch.stack([step.float() for step in layer1_steps[:split]], dim=0).sum(dim=0)
    layer1_late = torch.stack([step.float() for step in layer1_steps[split:]], dim=0).sum(dim=0) if len(layer1_steps[split:]) > 0 else torch.zeros_like(layer1_early)
    layer2_early = torch.stack([step.float() for step in layer2_steps[:split]], dim=0).sum(dim=0)
    layer2_late = torch.stack([step.float() for step in layer2_steps[split:]], dim=0).sum(dim=0) if len(layer2_steps[split:]) > 0 else torch.zeros_like(layer2_early)
    l1_vec = pooled_spike_pattern_vector(layer1_early, layer1_late, output_size=(4, 4))
    l2_vec = pooled_spike_pattern_vector(layer2_early, layer2_late, output_size=(2, 2))
    return l1_vec, l2_vec


def prepare_triplet_batches(df_triplets: pd.DataFrame, images: torch.Tensor, encoder, cfg: Step5Config, device: torch.device) -> Iterator[TripletBatch]:
    for batch_id, start in enumerate(range(0, len(df_triplets), cfg.batch_size)):
        batch_df = df_triplets.iloc[start : start + cfg.batch_size].copy().reset_index(drop=True)
        sample_ids = batch_df["sample_id"].astype(int).tolist()
        distractor_ids = batch_df["distractor_id"].astype(int).tolist()
        wrong_ids = batch_df["wrong_distractor_id"].astype(int).tolist()
        sample_lookup = build_spike_lookup(images, encoder, sample_ids, steps=cfg.sample_steps, device=device)
        dist_lookup = build_spike_lookup(images, encoder, distractor_ids + wrong_ids, steps=cfg.distractor_steps, device=device)
        sample_spikes = torch.stack([sample_lookup[idx] for idx in sample_ids], dim=0)
        distractor_spikes = torch.stack([dist_lookup[idx] for idx in distractor_ids], dim=0)
        wrong_spikes = torch.stack([dist_lookup[idx] for idx in wrong_ids], dim=0)
        sample_images = images[sample_ids].to(device=device, dtype=torch.float32)
        distractor_images = images[distractor_ids].to(device=device, dtype=torch.float32)
        cue_images = build_part_cue_images(sample_images, distractor_images)
        cue_spikes = {name: encode_images(encoder, cue_images[name], steps=cfg.ping_steps) for name in CUE_CONDITIONS}
        b, _, c, h, w = sample_spikes.shape
        yield TripletBatch(
            batch_id=batch_id,
            df=batch_df,
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            wrong_distractor_spikes=wrong_spikes,
            zero_sample_spikes=torch.zeros((b, cfg.sample_steps, c, h, w), dtype=sample_spikes.dtype, device=device),
            zero_distractor_spikes=torch.zeros((b, cfg.distractor_steps, c, h, w), dtype=sample_spikes.dtype, device=device),
            cue_spikes=cue_spikes,
        )


def snapshot_stsp_restore_state(net) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
    restore: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        if getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
            continue
        restore[layer_key] = (layer.u_pre.detach().cpu().clone(), layer.x_pre.detach().cpu().clone())
    return restore


def grouped_class_scores(net, tensor: torch.Tensor) -> np.ndarray:
    grouped = net.layer3.get_grouped_voltage(tensor.to(torch.float32)).mean(dim=-1)
    return grouped.detach().cpu().numpy().astype(np.float32, copy=False)


def extract_boundary_template_vector(net, boundary_state: Mapping[str, Mapping[str, torch.Tensor]]) -> np.ndarray:
    layer3_state = boundary_state["layer3"]
    return np.concatenate([grouped_class_scores(net, layer3_state["v_mem"]), grouped_class_scores(net, layer3_state["g_e"])], axis=1)


def forward_three_layers(net, input_t: torch.Tensor, t_step: int, *, stsp_mode: str, ping_drive: torch.Tensor | None = None) -> None:
    s1, _ = net.layer1.forward_step(input_t, t_step, training=False, monitor=False, stsp_mode=stsp_mode, ping_drive=ping_drive)
    s1_p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1_p, t_step, training=False, monitor=False, stsp_mode=stsp_mode)
    s2_p = net.pool2(s2.float())
    net.layer3.forward_step(s2_p, t_step, training=False, monitor=False, stsp_mode=stsp_mode)


def run_to_preprobe_boundary(net, cfg: Step5Config, sample_spikes: torch.Tensor, distractor_spikes: torch.Tensor, *, stsp_mode: str, condition: str) -> BackboneSnapshot:
    batch_size, _, channels, height, width = sample_spikes.shape
    with torch.no_grad():
        net.layer1.reset_state((batch_size, channels, height, width))
        h1 = (height + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
        w1 = (width + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
        net.layer2.reset_state((batch_size, net.layer1.out_channels, h1 // 2, w1 // 2))
        h2 = ((h1 // 2) + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
        w2 = ((w1 // 2) + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
        net.layer3.reset_state((batch_size, net.layer2.out_channels, h2 // 2, w2 // 2))
        layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
        zero_input = torch.zeros((batch_size, channels, height, width), dtype=sample_spikes.dtype, device=sample_spikes.device)
        current_time = 0
        for t_idx in range(int(sample_spikes.shape[1])):
            forward_three_layers(net, sample_spikes[:, t_idx, ...], current_time, stsp_mode=stsp_mode)
            current_time += 1
        for _ in range(cfg.delay1_steps):
            forward_three_layers(net, zero_input, current_time, stsp_mode=stsp_mode)
            current_time += 1
        for t_idx in range(int(distractor_spikes.shape[1])):
            forward_three_layers(net, distractor_spikes[:, t_idx, ...], current_time, stsp_mode=stsp_mode)
            current_time += 1
        for _ in range(cfg.delay2_steps):
            forward_three_layers(net, zero_input, current_time, stsp_mode=stsp_mode)
            current_time += 1
    return BackboneSnapshot(
        condition=condition,
        layer_input_shapes=layer_input_shapes,
        restore_ux_by_layer=snapshot_stsp_restore_state(net),
        template_vector=extract_boundary_template_vector(net, snapshot_boundary_state(net)),
    )


def permute_ux_state(restore_ux_by_layer: Mapping[str, tuple[torch.Tensor, torch.Tensor]], *, roll_steps: int) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_key, (u_state, x_state) in restore_ux_by_layer.items():
        if int(u_state.shape[0]) < 2:
            out[layer_key] = (u_state.clone(), x_state.clone())
            continue
        out[layer_key] = (torch.roll(u_state, shifts=roll_steps, dims=0).clone(), torch.roll(x_state, shifts=roll_steps, dims=0).clone())
    return out


def compute_pair_decode_hits(class_scores_time: np.ndarray, sample_labels: np.ndarray, distractor_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top2 = np.argsort(-class_scores_time, axis=2)[..., :2]
    sample_match = (top2 == sample_labels[:, None, None]).any(axis=2)
    distractor_match = (top2 == distractor_labels[:, None, None]).any(axis=2)
    top1 = np.argmax(class_scores_time, axis=2)
    top1_member = ((top1 == sample_labels[:, None]) | (top1 == distractor_labels[:, None])).astype(np.int64)
    return (sample_match & distractor_match).astype(np.int64), top1_member


def run_branch_readout(
    net,
    backbone: BackboneSnapshot,
    batch_df: pd.DataFrame,
    *,
    branch_name: str,
    condition: str,
    branch_spikes: torch.Tensor,
    stsp_mode: str,
    ping_amp: float = 0.0,
    shuffle_ux: bool = False,
) -> BranchReadout:
    restore_ux = permute_ux_state(backbone.restore_ux_by_layer, roll_steps=1) if shuffle_ux else backbone.restore_ux_by_layer
    with torch.no_grad():
        reset_all_state_restore_selected_stsp_in_place(net, backbone.layer_input_shapes, restore_ux_by_layer=restore_ux)
        net.layer3.reset_decision_state()
        net.layer3.v_mem.fill_(net.layer3.V_L)
        net.layer3.lateral_inh.reset_state(net.layer3.output_shape)
        zero_input = torch.zeros(backbone.layer_input_shapes["layer1"], dtype=torch.float32, device=net.layer1.v_mem.device)
        time_v: list[np.ndarray] = []
        time_g: list[np.ndarray] = []
        layer1_steps: list[torch.Tensor] = []
        layer2_steps: list[torch.Tensor] = []
        current_time = 0
        for t_idx in range(int(branch_spikes.shape[1])):
            input_t = branch_spikes[:, t_idx, ...]
            ping_drive = None
            if float(ping_amp) > 0.0:
                input_t = zero_input
                ping_drive = torch.full_like(zero_input, float(ping_amp))
            s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode, ping_drive=ping_drive)
            s1_p = net.pool1(s1.float())
            s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
            s2_p = net.pool2(s2.float())
            net.layer3.forward_step(s2_p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
            # Use pooled spike maps so cue-pattern comparison follows the effective inter-layer code.
            layer1_steps.append(s1_p.detach())
            layer2_steps.append(s2_p.detach())
            time_v.append(grouped_class_scores(net, net.layer3.v_mem))
            time_g.append(grouped_class_scores(net, net.layer3.g_e))
            current_time += 1
    class_scores_time = np.stack(time_v, axis=1)
    readout_vectors = np.concatenate([np.mean(np.stack(time_v, axis=1), axis=1), np.mean(np.stack(time_g, axis=1), axis=1)], axis=1)
    layer1_pattern_vector, layer2_pattern_vector = build_spike_pattern_summary(layer1_steps=layer1_steps, layer2_steps=layer2_steps)
    sample_labels = batch_df["sample_label"].to_numpy(dtype=np.int64, copy=False)
    distractor_labels = batch_df["distractor_label"].to_numpy(dtype=np.int64, copy=False)
    pair_decode_time, top1_member_time = compute_pair_decode_hits(class_scores_time, sample_labels, distractor_labels)
    pred_first, first_fire_t = decode_prediction_and_fire_time_from_layer3(net, batch_size=len(batch_df))
    pred_first_np = pred_first.detach().cpu().numpy().astype(np.int64, copy=False)
    first_fire_t_np = first_fire_t.detach().cpu().numpy().astype(np.int64, copy=False)
    first_fire_hit = ((pred_first_np == sample_labels) | (pred_first_np == distractor_labels)).astype(np.int64)
    silent_mask = pred_first_np < 0
    return BranchReadout(
        condition=condition,
        branch_name=branch_name,
        readout_vectors=readout_vectors,
        class_scores_time=class_scores_time,
        pair_decode_time=pair_decode_time,
        top1_member_time=top1_member_time,
        first_fire_pred=pred_first_np,
        first_fire_t=first_fire_t_np,
        first_fire_hit=first_fire_hit,
        silent_mask=silent_mask.astype(np.int64, copy=False),
        layer1_pattern_vector=layer1_pattern_vector,
        layer2_pattern_vector=layer2_pattern_vector,
    )


def centered_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    a_centered = a_arr - a_arr.mean(axis=-1, keepdims=True)
    b_centered = b_arr - b_arr.mean(axis=-1, keepdims=True)
    return np.sum(a_centered * b_centered, axis=-1) / np.maximum(np.linalg.norm(a_centered, axis=-1) * np.linalg.norm(b_centered, axis=-1), 1e-12)


def build_cue_template_bank(
    net,
    batch: TripletBatch,
    cfg: Step5Config,
    *,
    reference_backbones: Mapping[str, BackboneSnapshot],
) -> dict[str, dict[str, np.ndarray]]:
    sample_branch = trim_or_pad_spike_sequence(batch.sample_spikes, cfg.ping_steps)
    distractor_branch = trim_or_pad_spike_sequence(batch.distractor_spikes, cfg.ping_steps)
    wrong_branch = trim_or_pad_spike_sequence(batch.wrong_distractor_spikes, cfg.ping_steps)
    template_reads = {
        "sample_only_reference": run_branch_readout(net, reference_backbones["sample_only_reference"], batch.df, branch_name="cue_template", condition="sample_only_reference", branch_spikes=sample_branch, stsp_mode="dynamic"),
        "distractor_only_reference": run_branch_readout(net, reference_backbones["distractor_only_reference"], batch.df, branch_name="cue_template", condition="distractor_only_reference", branch_spikes=distractor_branch, stsp_mode="dynamic"),
        "baseline_intact": run_branch_readout(net, reference_backbones["baseline_intact"], batch.df, branch_name="cue_template", condition="baseline_intact", branch_spikes=combine_spike_sequences(sample_branch, distractor_branch), stsp_mode="dynamic"),
        "shuffled_pair_reference": run_branch_readout(net, reference_backbones["shuffled_pair_reference"], batch.df, branch_name="cue_template", condition="shuffled_pair_reference", branch_spikes=combine_spike_sequences(sample_branch, wrong_branch), stsp_mode="dynamic"),
    }
    bank: dict[str, dict[str, np.ndarray]] = {"L1": {}, "L2": {}}
    for name, readout in template_reads.items():
        bank["L1"][name] = readout.layer1_pattern_vector
        bank["L2"][name] = readout.layer2_pattern_vector
    return bank


def compute_ping_template_metrics(batch_df: pd.DataFrame, readout: BranchReadout, condition_trial_rows: pd.DataFrame) -> pd.DataFrame:
    df = batch_df[["triplet_id", "sample_id", "distractor_id", "probe_id"]].copy()
    trial_rows = condition_trial_rows[condition_trial_rows["condition"] == readout.condition].reset_index(drop=True)
    df["record_type"] = "ping_outcome_trial"
    df["condition"] = readout.condition
    df["ping_outcome"] = trial_rows["ping_outcome"]
    df["is_sample"] = trial_rows["is_sample"]
    df["is_distractor"] = trial_rows["is_distractor"]
    df["is_silent"] = trial_rows["is_silent"]
    df["is_other"] = trial_rows["is_other"]
    df["p_true_member"] = trial_rows["is_sample"].to_numpy(dtype=np.float64, copy=False) + trial_rows["is_distractor"].to_numpy(dtype=np.float64, copy=False)
    df["p_nonmember"] = trial_rows["is_other"]
    df["mean_first_fire_t_non_silent"] = np.where(readout.silent_mask.astype(bool), np.nan, readout.first_fire_t)
    df["PRS"] = df["p_true_member"] - df["p_nonmember"]
    df["WPRI_ping"] = df["p_true_member"] - df["is_silent"]
    return df


def compute_cue_holistic_metrics(
    batch_df: pd.DataFrame,
    cue_readouts: Mapping[str, BranchReadout],
    cue_template_bank: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    per_trial_accumulator: dict[int, dict[str, float]] = {int(tid): {"triplet_id": int(tid)} for tid in batch_df["triplet_id"].tolist()}
    sim_cache: dict[tuple[str, str], np.ndarray] = {}
    layer_field_map = {"L1": "layer1_pattern_vector", "L2": "layer2_pattern_vector"}
    for cue_name in CUE_CONDITIONS:
        readout = cue_readouts[cue_name]
        for layer_name, field_name in layer_field_map.items():
            vector = getattr(readout, field_name)
            template_stack = np.stack(
                [
                    cue_template_bank[layer_name]["sample_only_reference"],
                    cue_template_bank[layer_name]["distractor_only_reference"],
                    cue_template_bank[layer_name]["baseline_intact"],
                    cue_template_bank[layer_name]["shuffled_pair_reference"],
                ],
                axis=1,
            )
            sims = centered_cosine_similarity(vector[:, None, :], template_stack)
            sim_cache[(cue_name, layer_name)] = sims
            global_adv = sims[:, 2] - np.maximum(sims[:, 0], sims[:, 1])
            tps = sims[:, 2] - sims[:, 3]
            dominance = sims[:, 0] - sims[:, 1]
            for idx, row in enumerate(batch_df.itertuples(index=False)):
                long_rows.append(
                    {
                        "triplet_id": int(row.triplet_id),
                        "cue_condition": cue_name,
                        "layer": layer_name,
                        "Sim_sample_only": float(sims[idx, 0]),
                        "Sim_distractor_only": float(sims[idx, 1]),
                        "Sim_true_pair": float(sims[idx, 2]),
                        "Sim_shuffled_pair": float(sims[idx, 3]),
                        "GlobalAdv": float(global_adv[idx]),
                        "TPS": float(tps[idx]),
                        "Dominance_S": float(dominance[idx]),
                        "H_full": float(sims[idx, 2]),
                        "H_adv": float(global_adv[idx]),
                    }
                )
                per_trial_accumulator[int(row.triplet_id)][f"TPS_{cue_name.split('_', 1)[1]}_{layer_name}"] = float(tps[idx])
                per_trial_accumulator[int(row.triplet_id)][f"GlobalAdv_{cue_name.split('_', 1)[1]}_{layer_name}"] = float(global_adv[idx])
                per_trial_accumulator[int(row.triplet_id)][f"Sim_sample_only_{cue_name.split('_', 1)[1]}_{layer_name}"] = float(sims[idx, 0])
                per_trial_accumulator[int(row.triplet_id)][f"Sim_distractor_only_{cue_name.split('_', 1)[1]}_{layer_name}"] = float(sims[idx, 1])
    cue_gain_specs = {
        "SP": {
            "cue_key": "cue_SP",
            "partner_index": 1,
        },
        "DP": {
            "cue_key": "cue_DP",
            "partner_index": 0,
        },
    }
    for idx, row in enumerate(batch_df.itertuples(index=False)):
        entry = per_trial_accumulator[int(row.triplet_id)]
        for short_name, spec in cue_gain_specs.items():
            cue_key = str(spec["cue_key"])
            sims_l1 = sim_cache[(cue_key, "L1")][idx]
            sims_l2 = sim_cache[(cue_key, "L2")][idx]
            partner_gain = float(sims_l2[int(spec["partner_index"])] - sims_l1[int(spec["partner_index"])])
            # Fusion gain tracks how much closer the higher layer gets to the whole true-pair pattern.
            fusion_gain = float(sims_l2[2] - sims_l1[2])
            entry[f"PartnerGain_{short_name}"] = partner_gain
            entry[f"FusionGain_{short_name}"] = fusion_gain
    for triplet_id in sorted(per_trial_accumulator):
        wide_rows.append(per_trial_accumulator[triplet_id])
    df_long = pd.DataFrame(long_rows)
    df_wide = pd.DataFrame(wide_rows)
    if not df_long.empty:
        gain_columns = ["PartnerGain_SP", "PartnerGain_DP", "FusionGain_SP", "FusionGain_DP"]
        df_long = df_long.merge(df_wide[["triplet_id", *gain_columns]], on="triplet_id", how="left")
    return df_long, df_wide


def summarize_ping_trial_rows(batch_df: pd.DataFrame, readout: BranchReadout) -> pd.DataFrame:
    df = batch_df[["triplet_id", "sample_label", "distractor_label", "probe_label"]].copy()
    sample_labels = df["sample_label"].to_numpy(dtype=np.int64, copy=False)
    distractor_labels = df["distractor_label"].to_numpy(dtype=np.int64, copy=False)
    silent_mask = readout.silent_mask.astype(bool)
    is_sample = (~silent_mask) & (readout.first_fire_pred == sample_labels)
    is_distractor = (~silent_mask) & (readout.first_fire_pred == distractor_labels)
    is_other = (~silent_mask) & (~is_sample) & (~is_distractor)
    ping_outcome = np.full(len(df), "other", dtype=object)
    ping_outcome[is_sample] = "sample"
    ping_outcome[is_distractor] = "distractor"
    ping_outcome[silent_mask] = "silent"
    df["record_type"] = "trial"
    df["condition"] = readout.condition
    df["branch_name"] = readout.branch_name
    df["ping_decode_acc"] = readout.pair_decode_time.mean(axis=1)
    df["ping_decode_last"] = readout.pair_decode_time[:, -1]
    df["top1_member_last"] = readout.top1_member_time[:, -1]
    df["ping_first_fire_acc"] = readout.first_fire_hit
    df["first_fire_pred"] = readout.first_fire_pred
    df["first_fire_t"] = readout.first_fire_t
    df["ping_outcome"] = ping_outcome
    df["is_sample"] = is_sample.astype(np.int64)
    df["is_distractor"] = is_distractor.astype(np.int64)
    df["is_silent"] = silent_mask.astype(np.int64)
    df["is_other"] = is_other.astype(np.int64)
    df["p_member"] = df["is_sample"] + df["is_distractor"]
    return df


def summarize_ping_timecourse(readout: BranchReadout) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    silent_mask = readout.silent_mask.astype(bool)
    non_silent_value = 1.0 - silent_mask.mean()
    member_rate = readout.top1_member_time.mean(axis=0)
    for time_idx in range(readout.pair_decode_time.shape[1]):
        rows.append({"record_type": "timecourse_summary", "condition": readout.condition, "branch_name": readout.branch_name, "time_index": int(time_idx), "non_silent_rate": float(non_silent_value), "sample_or_distractor_rate": float(member_rate[time_idx]), "pair_decode_rate": float(readout.pair_decode_time[:, time_idx].mean()), "top1_member_rate": float(member_rate[time_idx])})
    return pd.DataFrame(rows)


def summarize_ping_condition(df_trial_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, sub in df_trial_rows.groupby("condition", sort=False):
        non_silent = sub[sub["is_silent"] == 0]
        rows.append(
            {
                "record_type": "condition_summary",
                "condition": str(condition),
                "branch_name": "neutral_ping",
                "p_sample": float(sub["is_sample"].mean()),
                "p_distractor": float(sub["is_distractor"].mean()),
                "p_silent": float(sub["is_silent"].mean()),
                "p_other": float(sub["is_other"].mean()),
                "p_member": float((sub["is_sample"] + sub["is_distractor"]).mean()),
                "ping_first_fire_acc": float(sub["ping_first_fire_acc"].mean()),
                "mean_first_fire_t_non_silent": float(non_silent["first_fire_t"].mean()) if not non_silent.empty else np.nan,
                "ping_decode_acc": float(sub["ping_decode_acc"].mean()),
                "ping_decode_last": float(sub["ping_decode_last"].mean()),
                "top1_member_last": float(sub["top1_member_last"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_cue_first_fire_trials(batch_df: pd.DataFrame, cue_readout: BranchReadout) -> pd.DataFrame:
    df = batch_df[["triplet_id", "sample_label", "distractor_label"]].copy()
    sample_labels = df["sample_label"].to_numpy(dtype=np.int64, copy=False)
    distractor_labels = df["distractor_label"].to_numpy(dtype=np.int64, copy=False)
    silent_mask = cue_readout.silent_mask.astype(bool)
    is_sample = (~silent_mask) & (cue_readout.first_fire_pred == sample_labels)
    is_distractor = (~silent_mask) & (cue_readout.first_fire_pred == distractor_labels)
    is_other = (~silent_mask) & (~is_sample) & (~is_distractor)
    cue_condition = str(cue_readout.condition)
    if cue_condition == "cue_SP":
        partner_recall: np.ndarray = is_distractor.astype(np.float64)
    elif cue_condition == "cue_DP":
        partner_recall = is_sample.astype(np.float64)
    else:
        partner_recall = np.full(len(df), np.nan, dtype=np.float64)
    df["record_type"] = "trial"
    df["cue_condition"] = cue_condition
    df["first_fire_pred"] = cue_readout.first_fire_pred
    df["first_fire_t"] = cue_readout.first_fire_t
    df["is_sample"] = is_sample.astype(np.int64)
    df["is_distractor"] = is_distractor.astype(np.int64)
    df["is_silent"] = silent_mask.astype(np.int64)
    df["is_other"] = is_other.astype(np.int64)
    df["PartnerRecall"] = partner_recall
    df["chance_baseline"] = CHANCE_BASELINE
    df["partner_above_chance"] = df["PartnerRecall"] - CHANCE_BASELINE
    return df


def summarize_cue_first_fire_condition(df_trial_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cue_condition, sub in df_trial_rows.groupby("cue_condition", sort=False):
        partner_values = pd.to_numeric(sub["PartnerRecall"], errors="coerce")
        partner_recall = float(partner_values.mean()) if partner_values.notna().any() else np.nan
        partner_minus_chance = partner_recall - CHANCE_BASELINE if np.isfinite(partner_recall) else np.nan
        rows.append(
            {
                "record_type": "condition_summary",
                "cue_condition": str(cue_condition),
                "p_sample": float(sub["is_sample"].mean()),
                "p_distractor": float(sub["is_distractor"].mean()),
                "p_silent": float(sub["is_silent"].mean()),
                "p_other": float(sub["is_other"].mean()),
                "PartnerRecall": partner_recall,
                "chance_baseline": CHANCE_BASELINE,
                "partner_minus_chance": partner_minus_chance,
            }
        )
    return pd.DataFrame(rows)


def save_step5_figure(layout, df_ping_trials: pd.DataFrame, df_ping_time: pd.DataFrame, df_ping_templates: pd.DataFrame, df_cue_holistic: pd.DataFrame, df_cue_specificity: pd.DataFrame, df_cue_first_fire: pd.DataFrame) -> dict[str, str]:
    apply_publication_style()
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 12.0))
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.flatten()
    ax_a.axis("off")
    phases = ["sample", "delay1", "distractor", "delay2", "ping / cue"]
    xpos = np.linspace(0.08, 0.88, num=len(phases))
    for idx, (label, x_val) in enumerate(zip(phases, xpos)):
        color = SAMPLE_COLOR if idx == 0 else (SHUFFLE_COLOR if idx == 2 else "#d9d9d9")
        ax_a.add_patch(plt.Rectangle((x_val - 0.07, 0.58), 0.14, 0.18, facecolor=color, edgecolor="black", lw=1.0))
        ax_a.text(x_val, 0.67, label, ha="center", va="center", fontsize=11)
        if idx < len(phases) - 1:
            ax_a.annotate("", xy=(xpos[idx + 1] - 0.08, 0.67), xytext=(x_val + 0.08, 0.67), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax_a.text(0.18, 0.28, "Branch A: neutral ping reactivation", color=DYNAMIC_COLOR, fontsize=11)
    ax_a.text(0.18, 0.16, "Branch B: part-cue holistic completion", color=STATIC_COLOR, fontsize=11)
    ax_a.set_title("Panel A: Step 5 paradigm")
    ping_summary = df_ping_trials.groupby("condition", sort=False)[["is_sample", "is_distractor", "is_silent", "is_other"]].mean().reindex(list(PING_CONDITIONS)).reset_index()
    outcome_colors = {"is_sample": SAMPLE_COLOR, "is_distractor": SHUFFLE_COLOR, "is_silent": NOISE_COLOR, "is_other": STATIC_COLOR}
    bottom = np.zeros(len(ping_summary), dtype=np.float64)
    x_ping = np.arange(len(ping_summary))
    for col_name, label in [("is_sample", "sample"), ("is_distractor", "distractor"), ("is_silent", "silent"), ("is_other", "other")]:
        values = ping_summary[col_name].to_numpy(dtype=np.float64, copy=False)
        ax_b.bar(x_ping, values, bottom=bottom, color=outcome_colors[col_name], edgecolor="black", label=label)
        bottom += values
    ax_b.set_xticks(x_ping, ping_summary["condition"], rotation=20)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel("Outcome proportion")
    ax_b.set_title("Panel B: Neutral ping first-fire outcome")
    ax_b.legend(fontsize=9, ncol=2)
    cue_l2_global = df_cue_holistic[df_cue_holistic["layer"] == "L2"].groupby("cue_condition", sort=False)[["Sim_sample_only", "Sim_distractor_only", "GlobalAdv"]].mean().reindex(list(CUE_CONDITIONS)).reset_index()
    x_cue = np.arange(len(cue_l2_global))
    ax_c.bar(x_cue - 0.26, cue_l2_global["Sim_sample_only"], width=0.26, color=SAMPLE_COLOR, edgecolor="black", label="Sim_sample_only")
    ax_c.bar(x_cue, cue_l2_global["Sim_distractor_only"], width=0.26, color=SHUFFLE_COLOR, edgecolor="black", label="Sim_distractor_only")
    ax_c.bar(x_cue + 0.26, cue_l2_global["GlobalAdv"], width=0.26, color=DYNAMIC_COLOR, edgecolor="black", label="GlobalAdv_L2")
    ax_c.axhline(0.0, color="black", lw=0.8)
    ax_c.set_xticks(x_cue, cue_l2_global["cue_condition"])
    ax_c.set_title("Panel C: Layer2 constituent similarity + GlobalAdv")
    ax_c.legend(fontsize=8)

    cue_l1 = df_cue_holistic[df_cue_holistic["layer"] == "L1"].groupby("cue_condition", sort=False)[["Sim_sample_only", "Sim_distractor_only"]].mean().reindex(list(CUE_CONDITIONS)).reset_index()
    ax_d.bar(x_cue - 0.18, cue_l1["Sim_sample_only"], width=0.36, color=SAMPLE_COLOR, edgecolor="black", label="Sim_sample_only")
    ax_d.bar(x_cue + 0.18, cue_l1["Sim_distractor_only"], width=0.36, color=SHUFFLE_COLOR, edgecolor="black", label="Sim_distractor_only")
    ax_d.set_xticks(x_cue, cue_l1["cue_condition"])
    ax_d.set_title("Panel D: Layer1 constituent similarity")
    ax_d.legend(fontsize=8)

    gain_cols = ["PartnerGain_SP", "PartnerGain_DP", "FusionGain_SP", "FusionGain_DP"]
    gain_summary = df_cue_specificity[gain_cols].mean()
    ax_e.bar(np.arange(len(gain_cols)), gain_summary.to_numpy(dtype=np.float64, copy=False), color=[DYNAMIC_COLOR, STATIC_COLOR, SAMPLE_COLOR, SHUFFLE_COLOR], edgecolor="black")
    ax_e.axhline(0.0, color="black", lw=0.8)
    ax_e.set_xticks(np.arange(len(gain_cols)), gain_cols, rotation=20)
    ax_e.set_title("Panel E: Partner gain / fusion gain")

    cue_ff_summary = df_cue_first_fire[df_cue_first_fire["record_type"] == "condition_summary"].copy().set_index("cue_condition").reindex(list(CUE_CONDITIONS)).reset_index()
    x_ff = np.arange(len(cue_ff_summary))
    ax_f.bar(x_ff - 0.27, cue_ff_summary["p_sample"], width=0.18, color=SAMPLE_COLOR, edgecolor="black", label="p_sample")
    ax_f.bar(x_ff - 0.09, cue_ff_summary["p_distractor"], width=0.18, color=SHUFFLE_COLOR, edgecolor="black", label="p_distractor")
    ax_f.bar(x_ff + 0.09, cue_ff_summary["p_silent"], width=0.18, color=NOISE_COLOR, edgecolor="black", label="p_silent")
    ax_f.bar(x_ff + 0.27, cue_ff_summary["p_other"], width=0.18, color=STATIC_COLOR, edgecolor="black", label="p_other")
    ax_f.axhline(CHANCE_BASELINE, color="black", lw=1.0, linestyle="--", label="chance=0.1")
    ax_f.set_xticks(x_ff, cue_ff_summary["cue_condition"])
    ax_f.set_ylim(0.0, 1.0)
    ax_f.set_title("Panel F: Cue first-fire outcome")
    ax_f.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = save_figure_all_formats(fig, layout.figure_base("chunk_step5_holistic_reactivation"))
    plt.close(fig)
    return out


def save_optional_npz(layout, template_bank_first: Mapping[str, np.ndarray] | None, example_ping: BranchReadout | None, example_cue: BranchReadout | None) -> list[str]:
    saved: list[str] = []
    if template_bank_first is not None:
        path = layout.data_file("template_bank_preprobe.npz")
        np.savez_compressed(path, **{key: value.astype(np.float32, copy=False) for key, value in template_bank_first.items()})
        saved.append(str(path))
    if example_ping is not None:
        path = layout.data_file("example_ping_trace.npz")
        np.savez_compressed(path, class_scores_time=example_ping.class_scores_time, pair_decode_time=example_ping.pair_decode_time, first_fire_pred=example_ping.first_fire_pred, first_fire_t=example_ping.first_fire_t, silent_mask=example_ping.silent_mask)
        saved.append(str(path))
    if example_cue is not None:
        path = layout.data_file("example_cue_trace.npz")
        np.savez_compressed(path, class_scores_time=example_cue.class_scores_time, pair_decode_time=example_cue.pair_decode_time, first_fire_pred=example_cue.first_fire_pred, first_fire_t=example_cue.first_fire_t, layer1_pattern_vector=example_cue.layer1_pattern_vector, layer2_pattern_vector=example_cue.layer2_pattern_vector)
        saved.append(str(path))
    return saved


def run_experiment(cfg: Step5Config) -> dict[str, object]:
    log_lines: list[str] = []
    layout = prepare_result_layout(cfg.output_dir)
    device, device_message = resolve_device_with_fallback(cfg.device)
    log_and_print(log_lines, device_message)
    seed_everything(int(cfg.seed))
    log_and_print(log_lines, f"[Config] split={cfg.split} seed={cfg.seed} batch_size={cfg.batch_size} triplets<={cfg.max_triplets} smoke={cfg.smoke}")
    save_run_config(asdict(cfg), layout.root)
    dataset = load_mnist_dataset(dataset_root=cfg.dataset_root, split=cfg.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    class_index = build_class_index(dataset, num_classes=10)
    log_and_print(log_lines, f"[Data] Loaded {len(dataset)} examples from split={cfg.split}.")
    net, encoder = load_model_and_encoder(model_path=cfg.model_path, device=device, dt=cfg.dt, max_duration_ms=cfg.max_duration_ms)
    log_and_print(log_lines, f"[Model] Loaded checkpoint from {cfg.model_path}.")
    df_triplets = build_triplet_specs(images=images, labels=labels, flat_normalized=flat_normalized, class_index=class_index, cfg=cfg)
    log_and_print(log_lines, f"[Triplets] Generated {len(df_triplets)} triplets.")
    all_ping_trial_rows: list[pd.DataFrame] = []
    all_ping_time_rows: list[pd.DataFrame] = []
    all_ping_template_rows: list[pd.DataFrame] = []
    all_cue_holistic_rows: list[pd.DataFrame] = []
    all_cue_specificity_rows: list[pd.DataFrame] = []
    all_cue_first_fire_rows: list[pd.DataFrame] = []
    template_bank_first: dict[str, np.ndarray] | None = None
    example_ping: BranchReadout | None = None
    example_cue: BranchReadout | None = None
    with torch.no_grad():
        for batch in prepare_triplet_batches(df_triplets=df_triplets, images=images, encoder=encoder, cfg=cfg, device=device):
            log_and_print(log_lines, f"[Batch {batch.batch_id}] size={len(batch.df)} building reference backbones.")
            reference_backbones = {
                "baseline_intact": run_to_preprobe_boundary(net, cfg, batch.sample_spikes, batch.distractor_spikes, stsp_mode="dynamic", condition="baseline_intact"),
                "sample_only_reference": run_to_preprobe_boundary(net, cfg, batch.sample_spikes, batch.zero_distractor_spikes, stsp_mode="dynamic", condition="sample_only_reference"),
                "distractor_only_reference": run_to_preprobe_boundary(net, cfg, batch.zero_sample_spikes, batch.distractor_spikes, stsp_mode="dynamic", condition="distractor_only_reference"),
                "shuffled_pair_reference": run_to_preprobe_boundary(net, cfg, batch.sample_spikes, batch.wrong_distractor_spikes, stsp_mode="dynamic", condition="shuffled_pair_reference"),
            }
            ping_only_backbone = run_to_preprobe_boundary(net, cfg, batch.zero_sample_spikes, batch.zero_distractor_spikes, stsp_mode="dynamic", condition="ping_only_backbone")
            static_backbone = run_to_preprobe_boundary(net, cfg, batch.sample_spikes, batch.distractor_spikes, stsp_mode="static_frozen", condition="static_backbone")
            template_bank = {name: snap.template_vector for name, snap in reference_backbones.items()}
            if template_bank_first is None:
                template_bank_first = {name: value.copy() for name, value in template_bank.items()}
            branch_spikes_none = torch.zeros((len(batch.df), cfg.ping_steps, batch.sample_spikes.shape[2], batch.sample_spikes.shape[3], batch.sample_spikes.shape[4]), dtype=batch.sample_spikes.dtype, device=batch.sample_spikes.device)
            ping_results = {
                "dynamic_ping": run_branch_readout(net, reference_backbones["baseline_intact"], batch.df, branch_name="neutral_ping", condition="dynamic_ping", branch_spikes=branch_spikes_none, stsp_mode="dynamic", ping_amp=cfg.ping_amp, shuffle_ux=False),
                "dynamic_no_ping": run_branch_readout(net, reference_backbones["baseline_intact"], batch.df, branch_name="neutral_ping", condition="dynamic_no_ping", branch_spikes=branch_spikes_none, stsp_mode="dynamic", ping_amp=0.0, shuffle_ux=False),
                "static_ping": run_branch_readout(net, static_backbone, batch.df, branch_name="neutral_ping", condition="static_ping", branch_spikes=branch_spikes_none, stsp_mode="static_frozen", ping_amp=cfg.ping_amp, shuffle_ux=False),
                "shuffle_ux_ping": run_branch_readout(net, reference_backbones["baseline_intact"], batch.df, branch_name="neutral_ping", condition="shuffle_ux_ping", branch_spikes=branch_spikes_none, stsp_mode="dynamic", ping_amp=cfg.ping_amp, shuffle_ux=True),
                "ping_only": run_branch_readout(net, ping_only_backbone, batch.df, branch_name="neutral_ping", condition="ping_only", branch_spikes=branch_spikes_none, stsp_mode="dynamic", ping_amp=cfg.ping_amp, shuffle_ux=False),
            }
            if example_ping is None:
                example_ping = ping_results["dynamic_ping"]
            batch_ping_rows: list[pd.DataFrame] = []
            for condition in PING_CONDITIONS:
                trial_rows = summarize_ping_trial_rows(batch.df, ping_results[condition])
                batch_ping_rows.append(trial_rows)
                all_ping_trial_rows.append(trial_rows)
                all_ping_time_rows.append(summarize_ping_timecourse(ping_results[condition]))
            all_ping_template_rows.append(compute_ping_template_metrics(batch.df, ping_results["dynamic_ping"], pd.concat(batch_ping_rows, axis=0, ignore_index=True)))
            cue_results = {cue_name: run_branch_readout(net, reference_backbones["baseline_intact"], batch.df, branch_name="part_cue", condition=cue_name, branch_spikes=batch.cue_spikes[cue_name], stsp_mode="dynamic", ping_amp=0.0, shuffle_ux=False) for cue_name in CUE_CONDITIONS}
            if example_cue is None:
                example_cue = cue_results["cue_SP"]
            cue_template_bank = build_cue_template_bank(net, batch, cfg, reference_backbones=reference_backbones)
            df_cue_holistic, df_cue_specificity = compute_cue_holistic_metrics(batch.df, cue_results, cue_template_bank)
            batch_cue_first_fire = pd.concat([summarize_cue_first_fire_trials(batch.df, cue_results[cue_name]) for cue_name in CUE_CONDITIONS], axis=0, ignore_index=True)
            all_cue_holistic_rows.append(df_cue_holistic)
            all_cue_specificity_rows.append(df_cue_specificity)
            all_cue_first_fire_rows.append(batch_cue_first_fire)
            log_and_print(log_lines, f"[Batch {batch.batch_id}] pre-probe backbone, ping branch, and cue branch completed.")
    df_ping_trials = pd.concat(all_ping_trial_rows, axis=0, ignore_index=True)
    df_ping_time = pd.concat(all_ping_time_rows, axis=0, ignore_index=True)
    df_ping_summary = summarize_ping_condition(df_ping_trials)
    df_ping_metrics = pd.concat([df_ping_trials, df_ping_summary, df_ping_time], axis=0, ignore_index=True, sort=False)
    df_ping_templates = pd.concat(all_ping_template_rows, axis=0, ignore_index=True)
    df_cue_holistic = pd.concat(all_cue_holistic_rows, axis=0, ignore_index=True)
    df_cue_specificity = pd.concat(all_cue_specificity_rows, axis=0, ignore_index=True)
    df_cue_first_fire_trials = pd.concat(all_cue_first_fire_rows, axis=0, ignore_index=True)
    df_cue_first_fire_summary = summarize_cue_first_fire_condition(df_cue_first_fire_trials)
    df_cue_first_fire = pd.concat([df_cue_first_fire_trials, df_cue_first_fire_summary], axis=0, ignore_index=True, sort=False)
    triplets_csv = save_tidy_csv(df_triplets, layout.data_file("triplets.csv"), sort_by=["triplet_id"])
    ping_csv = save_tidy_csv(df_ping_metrics, layout.data_file("ping_metrics.csv"))
    ping_template_csv = save_tidy_csv(df_ping_templates, layout.data_file("ping_template_similarity.csv"), sort_by=["triplet_id"])
    cue_holistic_csv = save_tidy_csv(df_cue_holistic, layout.data_file("cue_holistic_metrics.csv"), sort_by=["triplet_id", "cue_condition"])
    cue_specificity_csv = save_tidy_csv(df_cue_specificity, layout.data_file("cue_completion_specificity.csv"), sort_by=["triplet_id"])
    cue_first_fire_csv = save_tidy_csv(df_cue_first_fire, layout.data_file("cue_first_fire_metrics.csv"))
    figure_paths: dict[str, str] = {}
    if not cfg.skip_figures:
        figure_paths = save_step5_figure(layout, df_ping_trials, df_ping_time, df_ping_templates, df_cue_holistic, df_cue_specificity, df_cue_first_fire)
        log_and_print(log_lines, f"[Figure] Saved Step 5 figure to {figure_paths.get('png', '')}.")
    optional_npz = save_optional_npz(layout, template_bank_first, example_ping, example_cue)
    summary = {
        "triplet_count": int(len(df_triplets)),
        "ping_decode_acc_dynamic_ping_mean": float(df_ping_trials.loc[df_ping_trials["condition"] == "dynamic_ping", "ping_decode_acc"].mean()),
        "ping_first_fire_acc_dynamic_ping_mean": float(df_ping_trials.loc[df_ping_trials["condition"] == "dynamic_ping", "ping_first_fire_acc"].mean()),
        "dynamic_ping_outcomes": {
            "p_sample": float(df_ping_trials.loc[df_ping_trials["condition"] == "dynamic_ping", "is_sample"].mean()),
            "p_distractor": float(df_ping_trials.loc[df_ping_trials["condition"] == "dynamic_ping", "is_distractor"].mean()),
            "p_silent": float(df_ping_trials.loc[df_ping_trials["condition"] == "dynamic_ping", "is_silent"].mean()),
            "p_other": float(df_ping_trials.loc[df_ping_trials["condition"] == "dynamic_ping", "is_other"].mean()),
        },
        "PRS_mean": float(df_ping_templates["PRS"].mean()),
        "WPRI_ping_mean": float(df_ping_templates["WPRI_ping"].mean()),
        "cue_layer_summary": {
            "GlobalAdv_primary": float(df_cue_holistic.loc[df_cue_holistic["layer"] == "L2", "GlobalAdv"].mean()),
            "GlobalAdv_L1": float(df_cue_holistic.loc[df_cue_holistic["layer"] == "L1", "GlobalAdv"].mean()),
            "GlobalAdv_L2": float(df_cue_holistic.loc[df_cue_holistic["layer"] == "L2", "GlobalAdv"].mean()),
            "TPS_L1": float(df_cue_holistic.loc[df_cue_holistic["layer"] == "L1", "TPS"].mean()),
            "TPS_L2": float(df_cue_holistic.loc[df_cue_holistic["layer"] == "L2", "TPS"].mean()),
        },
        "cue_partner_recall_summary": {
            row["cue_condition"]: {
                "PartnerRecall": (float(row["PartnerRecall"]) if pd.notna(row["PartnerRecall"]) else None),
                "partner_minus_chance": (float(row["partner_minus_chance"]) if pd.notna(row["partner_minus_chance"]) else None),
            }
            for row in df_cue_first_fire_summary.to_dict(orient="records")
        },
        "partner_gain_summary": {
            "PartnerGain_SP": float(df_cue_specificity["PartnerGain_SP"].mean()),
            "PartnerGain_DP": float(df_cue_specificity["PartnerGain_DP"].mean()),
        },
        "fusion_gain_summary": {
            "FusionGain_SP": float(df_cue_specificity["FusionGain_SP"].mean()),
            "FusionGain_DP": float(df_cue_specificity["FusionGain_DP"].mean()),
        },
        "exported_files": {
            "triplets_csv": triplets_csv,
            "ping_metrics_csv": ping_csv,
            "ping_template_similarity_csv": ping_template_csv,
            "cue_holistic_metrics_csv": cue_holistic_csv,
            "cue_completion_specificity_csv": cue_specificity_csv,
            "cue_first_fire_metrics_csv": cue_first_fire_csv,
            "figures": figure_paths,
            "optional_npz": optional_npz,
        },
    }
    save_summary_json(summary, layout.root)
    save_log_lines(log_lines, layout.log_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    cfg = normalize_config(build_argparser().parse_args(argv))
    summary = run_experiment(cfg)
    print(f"[Done] triplets={summary['triplet_count']} dynamic_ping_decode={summary['ping_decode_acc_dynamic_ping_mean']:.4f} PRS={summary['PRS_mean']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
