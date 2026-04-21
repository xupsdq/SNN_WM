from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import DEFAULT_PATH_CONFIG
from src.config.units import ms
from src.experiments.common.dataset import build_class_index, build_dataset_arrays, encode_images
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.common.distractor_triplets import build_triplet_specs as shared_build_triplet_specs
from src.experiments.common.distractor_triplets import load_mnist_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import build_layer_input_shapes, reset_all_state_restore_selected_stsp_in_place
from src.experiments.common.ping_common import LAYER_KEYS, prepare_network_state
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.experiments.common.runtime import seed_everything
from src.experiments.common.seed import mix_seed
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_tidy_csv
from src.plotting.common.style import DYNAMIC_COLOR, NOISE_COLOR, SAMPLE_COLOR, SHUFFLE_COLOR

CONDITION_FULL = "baseline_intact"
CONDITION_SAMPLE = "sample_only_reference"
CONDITION_DISTRACTOR = "distractor_only_reference"
CONDITION_SHUFFLE = "shuffled_pair_reference"
CONDITION_NAMES = (
    CONDITION_FULL,
    CONDITION_SAMPLE,
    CONDITION_DISTRACTOR,
    CONDITION_SHUFFLE,
)
REGION_NAMES = ("SP", "DP", "SDP")
PING_CONDITION = "neutral_ping"


@dataclass(frozen=True)
class TaxonomyConfig:
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
    batch_size: int
    max_probes: int
    samples_per_probe: int
    max_triplets: int
    num_sim_bins: int
    epsilon: float
    skip_figures: bool
    smoke: bool
    dt: float = 1.0 * ms
    ping_ms: float = 30.0
    ping_amp: float = 1.0

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
    sample_images: torch.Tensor
    distractor_images: torch.Tensor


@dataclass
class BackboneSnapshot:
    condition: str
    layer_input_shapes: Dict[str, tuple[int, ...]]
    restore_ux_by_layer: Dict[str, tuple[torch.Tensor, torch.Tensor]]
    state_by_layer: Dict[str, Dict[str, np.ndarray]]


@dataclass
class PingReadout:
    condition: str
    first_fire_pred: np.ndarray
    first_fire_t: np.ndarray
    silent_mask: np.ndarray
    sample_first: np.ndarray
    distractor_first: np.ndarray
    member_first: np.ndarray
    nonmember_first: np.ndarray


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk STSP state taxonomy analysis.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PATH_CONFIG.results_root / "chunk_stsp_state_taxonomy"),
    )
    parser.add_argument("--sample-ms", type=float, default=50.0)
    parser.add_argument("--delay1-ms", type=float, default=50.0)
    parser.add_argument("--distractor-ms", type=float, default=50.0)
    parser.add_argument("--delay2-ms", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-probes", type=int, default=20)
    parser.add_argument("--samples-per-probe", type=int, default=25)
    parser.add_argument("--max-triplets", type=int, default=500)
    parser.add_argument("--num-sim-bins", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--ping-ms", type=float, default=30.0)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def normalize_config(args: argparse.Namespace) -> TaxonomyConfig:
    cfg = TaxonomyConfig(
        **{
            key: getattr(args, key)
            for key in TaxonomyConfig.__dataclass_fields__.keys()
            if hasattr(args, key)
        }
    )
    if cfg.smoke:
        cfg = TaxonomyConfig(
            **{
                **asdict(cfg),
                "batch_size": min(int(cfg.batch_size), 2),
                "max_probes": min(int(cfg.max_probes), 2),
                "samples_per_probe": min(int(cfg.samples_per_probe), 2),
                "max_triplets": min(int(cfg.max_triplets), 4),
                "num_sim_bins": min(int(cfg.num_sim_bins), 3),
            }
        )
    if min(
        cfg.sample_steps,
        cfg.delay1_steps,
        cfg.distractor_steps,
        cfg.delay2_steps,
        cfg.ping_steps,
    ) <= 0:
        raise ValueError("All durations must map to at least one simulation step.")
    return cfg


def resolve_device_with_fallback(device_arg: str) -> tuple[torch.device, str]:
    normalized = str(device_arg).strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "[Runtime] Using device=auto -> cuda."
        return torch.device("cpu"), "[Runtime] Using device=auto -> cpu."
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu"), "[Runtime] CUDA unavailable on 2026-04-17; falling back to CPU."
    return torch.device(str(device_arg)), f"[Runtime] Using device={device_arg}."


def log_and_print(log_lines: list[str], message: str) -> None:
    print(message, flush=True)
    log_lines.append(str(message))


def assign_wrong_distractors(
    df_triplets: pd.DataFrame,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    all_ids = np.arange(len(labels), dtype=np.int64)
    rows: list[dict[str, object]] = []
    for row in df_triplets.itertuples(index=False):
        valid_mask = (
            (all_ids != int(row.sample_id))
            & (all_ids != int(row.probe_id))
            & (all_ids != int(row.distractor_id))
            & (labels[all_ids] != int(row.sample_label))
            & (labels[all_ids] != int(row.probe_label))
            & (labels[all_ids] != int(row.distractor_label))
        )
        valid_ids = all_ids[valid_mask]
        if valid_ids.size <= 0:
            raise RuntimeError("Failed to assign wrong distractors from the dataset pool.")
        wrong_id = int(valid_ids[int(rng.integers(0, valid_ids.size))])
        rows.append(
            {
                "wrong_distractor_id": wrong_id,
                "wrong_distractor_label": int(labels[wrong_id]),
                "wrong_dp_similarity": float(np.dot(flat_normalized[wrong_id], flat_normalized[int(row.probe_id)])),
                "wrong_sd_similarity": float(np.dot(flat_normalized[wrong_id], flat_normalized[int(row.sample_id)])),
            }
        )
    out = pd.concat([df_triplets.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["selection_metadata"] = "chunk_stsp_state_taxonomy"
    return out


def build_triplet_specs(
    images: torch.Tensor,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    cfg: TaxonomyConfig,
) -> pd.DataFrame:
    df = shared_build_triplet_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=int(cfg.max_probes),
        samples_per_probe=int(cfg.samples_per_probe),
        num_bins=int(cfg.num_sim_bins),
        max_triplets=int(cfg.max_triplets),
        seed=int(cfg.seed),
    )
    return assign_wrong_distractors(
        df_triplets=df,
        labels=labels,
        flat_normalized=flat_normalized,
        seed=mix_seed(cfg.seed, 401),
    )


def build_spike_lookup(
    images: torch.Tensor,
    encoder,
    ids: Sequence[int],
    *,
    steps: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    unique_ids = sorted({int(idx) for idx in ids})
    spike_bank = encode_images(
        encoder,
        images[unique_ids].to(device=device, dtype=torch.float32),
        steps=int(steps),
    )
    return {int(image_id): spike_bank[pos] for pos, image_id in enumerate(unique_ids)}


def prepare_triplet_batches(
    df_triplets: pd.DataFrame,
    images: torch.Tensor,
    encoder,
    cfg: TaxonomyConfig,
    device: torch.device,
) -> Iterator[TripletBatch]:
    for batch_id, start in enumerate(range(0, len(df_triplets), int(cfg.batch_size))):
        batch_df = df_triplets.iloc[start : start + int(cfg.batch_size)].copy().reset_index(drop=True)
        sample_ids = batch_df["sample_id"].astype(int).tolist()
        distractor_ids = batch_df["distractor_id"].astype(int).tolist()
        wrong_ids = batch_df["wrong_distractor_id"].astype(int).tolist()
        sample_lookup = build_spike_lookup(images, encoder, sample_ids, steps=cfg.sample_steps, device=device)
        distractor_lookup = build_spike_lookup(
            images,
            encoder,
            distractor_ids + wrong_ids,
            steps=cfg.distractor_steps,
            device=device,
        )
        sample_spikes = torch.stack([sample_lookup[int(idx)] for idx in sample_ids], dim=0)
        distractor_spikes = torch.stack([distractor_lookup[int(idx)] for idx in distractor_ids], dim=0)
        wrong_distractor_spikes = torch.stack([distractor_lookup[int(idx)] for idx in wrong_ids], dim=0)
        sample_images = images[sample_ids].to(device=device, dtype=torch.float32)
        distractor_images = images[distractor_ids].to(device=device, dtype=torch.float32)
        batch_size, _, channels, height, width = sample_spikes.shape
        yield TripletBatch(
            batch_id=int(batch_id),
            df=batch_df,
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            wrong_distractor_spikes=wrong_distractor_spikes,
            zero_sample_spikes=torch.zeros(
                (batch_size, cfg.sample_steps, channels, height, width),
                dtype=sample_spikes.dtype,
                device=device,
            ),
            zero_distractor_spikes=torch.zeros(
                (batch_size, cfg.distractor_steps, channels, height, width),
                dtype=sample_spikes.dtype,
                device=device,
            ),
            sample_images=sample_images,
            distractor_images=distractor_images,
        )


def forward_three_layers(
    net,
    input_t: torch.Tensor,
    t_step: int,
    *,
    stsp_mode: str,
    ping_drive: torch.Tensor | None = None,
) -> None:
    s1, _ = net.layer1.forward_step(
        input_t,
        t_step,
        training=False,
        monitor=False,
        stsp_mode=stsp_mode,
        ping_drive=ping_drive,
    )
    s1_p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1_p, t_step, training=False, monitor=False, stsp_mode=stsp_mode)
    s2_p = net.pool2(s2.float())
    net.layer3.forward_step(s2_p, t_step, training=False, monitor=False, stsp_mode=stsp_mode)


def snapshot_stsp_state_batch(net, batch_size: int) -> tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    state_by_layer: Dict[str, Dict[str, np.ndarray]] = {}
    restore_by_layer: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        if getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
            raise ValueError(f"{layer_key} is missing STSP state at the requested boundary.")
        u = layer.u_pre.detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=False)
        x = layer.x_pre.detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=False)
        g = (layer.u_pre * layer.x_pre).detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=False)
        state_by_layer[str(layer_key)] = {"u_pre": u, "x_pre": x, "g_pre": g}
        restore_by_layer[str(layer_key)] = (
            layer.u_pre.detach().cpu().clone(),
            layer.x_pre.detach().cpu().clone(),
        )
    return state_by_layer, restore_by_layer


def run_to_preprobe_boundary_and_snapshot_stsp(
    net,
    cfg: TaxonomyConfig,
    sample_spikes: torch.Tensor,
    distractor_spikes: torch.Tensor,
    *,
    stsp_mode: str,
    condition: str,
) -> BackboneSnapshot:
    batch_size, _, channels, height, width = sample_spikes.shape
    with torch.no_grad():
        prepare_network_state(net, batch_size, channels, height, width)
        layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
        zero_input = torch.zeros((batch_size, channels, height, width), dtype=sample_spikes.dtype, device=sample_spikes.device)
        current_time = 0
        for t_idx in range(int(sample_spikes.shape[1])):
            forward_three_layers(net, sample_spikes[:, t_idx, ...], current_time, stsp_mode=stsp_mode)
            current_time += 1
        for _ in range(int(cfg.delay1_steps)):
            forward_three_layers(net, zero_input, current_time, stsp_mode=stsp_mode)
            current_time += 1
        for t_idx in range(int(distractor_spikes.shape[1])):
            forward_three_layers(net, distractor_spikes[:, t_idx, ...], current_time, stsp_mode=stsp_mode)
            current_time += 1
        for _ in range(int(cfg.delay2_steps)):
            forward_three_layers(net, zero_input, current_time, stsp_mode=stsp_mode)
            current_time += 1
        state_by_layer, restore_by_layer = snapshot_stsp_state_batch(net, batch_size=batch_size)
    return BackboneSnapshot(
        condition=str(condition),
        layer_input_shapes=layer_input_shapes,
        restore_ux_by_layer=restore_by_layer,
        state_by_layer=state_by_layer,
    )


def run_neutral_ping_branch(
    net,
    cfg: TaxonomyConfig,
    backbone: BackboneSnapshot,
    batch_df: pd.DataFrame,
) -> PingReadout:
    batch_size = len(batch_df)
    sample_labels = batch_df["sample_label"].to_numpy(dtype=np.int64, copy=False)
    distractor_labels = batch_df["distractor_label"].to_numpy(dtype=np.int64, copy=False)
    with torch.no_grad():
        reset_all_state_restore_selected_stsp_in_place(
            net,
            backbone.layer_input_shapes,
            restore_ux_by_layer=backbone.restore_ux_by_layer,
        )
        net.layer3.reset_decision_state()
        net.layer3.v_mem.fill_(net.layer3.V_L)
        net.layer3.lateral_inh.reset_state(net.layer3.output_shape)
        zero_input = torch.zeros(
            backbone.layer_input_shapes["layer1"],
            dtype=torch.float32,
            device=net.layer1.v_mem.device,
        )
        ping_drive = torch.full_like(zero_input, float(cfg.ping_amp))
        for t_idx in range(int(cfg.ping_steps)):
            forward_three_layers(net, zero_input, t_idx, stsp_mode="dynamic", ping_drive=ping_drive)
    first_fire_pred, first_fire_t = decode_prediction_and_fire_time_from_layer3(net, batch_size=batch_size)
    pred_np = first_fire_pred.numpy().astype(np.int64, copy=False)
    fire_t_np = first_fire_t.numpy().astype(np.int64, copy=False)
    silent_mask = pred_np < 0
    sample_first = ((~silent_mask) & (pred_np == sample_labels)).astype(np.int64)
    distractor_first = ((~silent_mask) & (pred_np == distractor_labels)).astype(np.int64)
    member_first = (sample_first | distractor_first).astype(np.int64)
    nonmember_first = ((~silent_mask) & (member_first == 0)).astype(np.int64)
    return PingReadout(
        condition=PING_CONDITION,
        first_fire_pred=pred_np,
        first_fire_t=fire_t_np,
        silent_mask=silent_mask.astype(np.int64, copy=False),
        sample_first=sample_first,
        distractor_first=distractor_first,
        member_first=member_first,
        nonmember_first=nonmember_first,
    )


def centered_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    a_centered = a_arr - a_arr.mean(axis=1, keepdims=True)
    b_centered = b_arr - b_arr.mean(axis=1, keepdims=True)
    numerator = np.sum(a_centered * b_centered, axis=1)
    denominator = np.maximum(
        np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1),
        float(eps),
    )
    return numerator / denominator


def raw_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    numerator = np.sum(a_arr * b_arr, axis=1)
    denominator = np.maximum(
        np.linalg.norm(a_arr, axis=1) * np.linalg.norm(b_arr, axis=1),
        float(eps),
    )
    return numerator / denominator


def assign_bins_from_values(values: np.ndarray, num_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.asarray([], dtype=object)
    q = max(1, min(int(num_bins), int(arr.size)))
    labels = [f"bin_{idx + 1}" for idx in range(q)]
    if q == 1:
        return np.asarray([labels[0]] * int(arr.size), dtype=object)
    ranks = pd.Series(arr).rank(method="first")
    try:
        return pd.qcut(ranks, q=q, labels=labels).astype("object").to_numpy()
    except ValueError:
        order = np.argsort(arr, kind="stable")
        raw = np.linspace(0, q - 1, num=arr.size)
        out = np.empty(arr.size, dtype=object)
        for pos, idx in enumerate(order.tolist()):
            out[int(idx)] = labels[int(round(float(raw[pos])))]
        return out


def compute_similarity_rows(
    batch_df: pd.DataFrame,
    state_bank: Mapping[str, BackboneSnapshot],
    eps: float = 1e-12,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_key in LAYER_KEYS:
        full_g = state_bank[CONDITION_FULL].state_by_layer[layer_key]["g_pre"]
        sample_g = state_bank[CONDITION_SAMPLE].state_by_layer[layer_key]["g_pre"]
        distractor_g = state_bank[CONDITION_DISTRACTOR].state_by_layer[layer_key]["g_pre"]
        shuffle_g = state_bank[CONDITION_SHUFFLE].state_by_layer[layer_key]["g_pre"]
        sim_fs = centered_cosine_similarity(full_g, sample_g, eps=eps)
        sim_fd = centered_cosine_similarity(full_g, distractor_g, eps=eps)
        sim_fshuffle = centered_cosine_similarity(full_g, shuffle_g, eps=eps)
        raw_fs = raw_cosine_similarity(full_g, sample_g, eps=eps)
        raw_fd = raw_cosine_similarity(full_g, distractor_g, eps=eps)
        raw_fshuffle = raw_cosine_similarity(full_g, shuffle_g, eps=eps)
        di = (sim_fs - sim_fd) / np.maximum(sim_fs + sim_fd, eps)
        for row_idx, triplet_row in enumerate(batch_df.itertuples(index=False)):
            rows.append(
                {
                    "record_type": "trial_level",
                    "triplet_id": int(triplet_row.triplet_id),
                    "layer": str(layer_key),
                    "sample_id": int(triplet_row.sample_id),
                    "distractor_id": int(triplet_row.distractor_id),
                    "probe_id": int(triplet_row.probe_id),
                    "sample_label": int(triplet_row.sample_label),
                    "distractor_label": int(triplet_row.distractor_label),
                    "probe_label": int(triplet_row.probe_label),
                    "Sim_FS": float(sim_fs[row_idx]),
                    "Sim_FD": float(sim_fd[row_idx]),
                    "Sim_FShuffle": float(sim_fshuffle[row_idx]),
                    "raw_Sim_FS": float(raw_fs[row_idx]),
                    "raw_Sim_FD": float(raw_fd[row_idx]),
                    "raw_Sim_FShuffle": float(raw_fshuffle[row_idx]),
                    "DI": float(di[row_idx]),
                }
            )
    return rows


def fit_linear_decomposition(
    full_vec: np.ndarray,
    sample_vec: np.ndarray,
    distractor_vec: np.ndarray,
    base_u: float,
    eps: float,
) -> dict[str, float]:
    y = np.asarray(full_vec, dtype=np.float64)
    x_s = np.asarray(sample_vec, dtype=np.float64)
    x_d = np.asarray(distractor_vec, dtype=np.float64)
    design = np.column_stack([x_s, x_d, np.ones_like(x_s)])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ coef
    residual = y - y_hat
    sse = float(np.sum(np.square(residual)))
    tss = float(np.sum(np.square(y - y.mean())))
    r2 = 1.0 - sse / max(tss, eps)
    base_vec = np.full_like(y, float(base_u))
    residual_norm = float(np.linalg.norm(residual) / max(np.linalg.norm(y - base_vec), eps))
    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "c": float(coef[2]),
        "R2": float(r2),
        "residual_norm": residual_norm,
    }


def compute_decomposition_rows(
    batch_df: pd.DataFrame,
    state_bank: Mapping[str, BackboneSnapshot],
    net,
    eps: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_key in LAYER_KEYS:
        full_g = state_bank[CONDITION_FULL].state_by_layer[layer_key]["g_pre"]
        sample_g = state_bank[CONDITION_SAMPLE].state_by_layer[layer_key]["g_pre"]
        distractor_g = state_bank[CONDITION_DISTRACTOR].state_by_layer[layer_key]["g_pre"]
        base_u = float(getattr(net, layer_key).stsp_U)
        for row_idx, triplet_row in enumerate(batch_df.itertuples(index=False)):
            metrics = fit_linear_decomposition(
                full_vec=full_g[row_idx],
                sample_vec=sample_g[row_idx],
                distractor_vec=distractor_g[row_idx],
                base_u=base_u,
                eps=eps,
            )
            rows.append(
                {
                    "record_type": "trial_level",
                    "triplet_id": int(triplet_row.triplet_id),
                    "layer": str(layer_key),
                    "sample_label": int(triplet_row.sample_label),
                    "distractor_label": int(triplet_row.distractor_label),
                    **metrics,
                }
            )
    return rows


def _condition_change_stats(vec: np.ndarray, base_u: float, epsilon: float) -> dict[str, float | np.ndarray]:
    dg = np.asarray(vec, dtype=np.float64) - float(base_u)
    changed = np.abs(dg) > float(epsilon)
    positive = dg > float(epsilon)
    negative = dg < -float(epsilon)
    return {
        "changed": changed,
        "changed_fraction": float(changed.mean()),
        "changed_magnitude_mean": float(np.mean(np.abs(dg[changed]))) if np.any(changed) else 0.0,
        "positive_fraction": float(positive.mean()),
        "negative_fraction": float(negative.mean()),
    }


def compute_changed_synapse_rows(
    batch_df: pd.DataFrame,
    state_bank: Mapping[str, BackboneSnapshot],
    net,
    epsilon: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_key in LAYER_KEYS:
        base_u = float(getattr(net, layer_key).stsp_U)
        full_g = state_bank[CONDITION_FULL].state_by_layer[layer_key]["g_pre"]
        sample_g = state_bank[CONDITION_SAMPLE].state_by_layer[layer_key]["g_pre"]
        distractor_g = state_bank[CONDITION_DISTRACTOR].state_by_layer[layer_key]["g_pre"]
        shuffle_g = state_bank[CONDITION_SHUFFLE].state_by_layer[layer_key]["g_pre"]
        for row_idx, triplet_row in enumerate(batch_df.itertuples(index=False)):
            stats_s = _condition_change_stats(sample_g[row_idx], base_u=base_u, epsilon=epsilon)
            stats_d = _condition_change_stats(distractor_g[row_idx], base_u=base_u, epsilon=epsilon)
            stats_f = _condition_change_stats(full_g[row_idx], base_u=base_u, epsilon=epsilon)
            stats_shuffle = _condition_change_stats(shuffle_g[row_idx], base_u=base_u, epsilon=epsilon)
            s_mask = np.asarray(stats_s["changed"], dtype=bool)
            d_mask = np.asarray(stats_d["changed"], dtype=bool)
            f_mask = np.asarray(stats_f["changed"], dtype=bool)
            shared = s_mask & d_mask
            s_only = s_mask & (~d_mask)
            d_only = d_mask & (~s_mask)
            full_only_novel = f_mask & (~s_mask) & (~d_mask)
            union_supported = f_mask & (s_mask | d_mask)
            full_count = int(f_mask.sum())
            if full_count > 0:
                s_only_in_full = f_mask & s_only
                d_only_in_full = f_mask & d_only
                shared_in_full = f_mask & shared
                novel_in_full = full_only_novel
                p_s_only_given_full = float(s_only_in_full.sum() / full_count)
                p_d_only_given_full = float(d_only_in_full.sum() / full_count)
                p_shared_given_full = float(shared_in_full.sum() / full_count)
                p_novel_given_full = float(novel_in_full.sum() / full_count)
            else:
                p_s_only_given_full = np.nan
                p_d_only_given_full = np.nan
                p_shared_given_full = np.nan
                p_novel_given_full = np.nan
            rows.append(
                {
                    "record_type": "trial_level",
                    "triplet_id": int(triplet_row.triplet_id),
                    "layer": str(layer_key),
                    "epsilon": float(epsilon),
                    "changed_fraction_sample": float(stats_s["changed_fraction"]),
                    "changed_fraction_distractor": float(stats_d["changed_fraction"]),
                    "changed_fraction_full": float(stats_f["changed_fraction"]),
                    "changed_fraction_shuffle": float(stats_shuffle["changed_fraction"]),
                    "changed_magnitude_mean_sample": float(stats_s["changed_magnitude_mean"]),
                    "changed_magnitude_mean_distractor": float(stats_d["changed_magnitude_mean"]),
                    "changed_magnitude_mean_full": float(stats_f["changed_magnitude_mean"]),
                    "positive_fraction_full": float(stats_f["positive_fraction"]),
                    "negative_fraction_full": float(stats_f["negative_fraction"]),
                    "S_only_changed_fraction": float(s_only.mean()),
                    "D_only_changed_fraction": float(d_only.mean()),
                    "Shared_changed_fraction": float(shared.mean()),
                    "Full_only_novel_changed_fraction": float(full_only_novel.mean()),
                    "Full_union_supported_changed_fraction": float(union_supported.mean()),
                    "full_changed_count": int(full_count),
                    "P_S_only_given_full": p_s_only_given_full,
                    "P_D_only_given_full": p_d_only_given_full,
                    "P_Shared_given_full": p_shared_given_full,
                    "P_Novel_given_full": p_novel_given_full,
                }
            )
    return rows


def build_layer1_region_masks(
    sample_images: torch.Tensor,
    distractor_images: torch.Tensor,
    channels: int,
) -> Dict[str, np.ndarray]:
    sample_norm = sample_images / sample_images.flatten(start_dim=1).amax(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
    distractor_norm = distractor_images / distractor_images.flatten(start_dim=1).amax(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
    sample_support = sample_norm > 1e-12
    distractor_support = distractor_norm > 1e-12
    support_masks = {
        "SP": sample_support & (~distractor_support),
        "DP": distractor_support & (~sample_support),
        "SDP": sample_support & distractor_support,
    }
    out: Dict[str, np.ndarray] = {}
    for region_name, mask in support_masks.items():
        expanded = mask.expand(mask.shape[0], int(channels), mask.shape[2], mask.shape[3])
        out[str(region_name)] = expanded.detach().cpu().numpy().astype(bool, copy=False).reshape(mask.shape[0], -1)
    return out


def compute_layer1_region_rows(
    batch_df: pd.DataFrame,
    batch: TripletBatch,
    state_bank: Mapping[str, BackboneSnapshot],
    net,
    epsilon: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    channels = int(batch.sample_spikes.shape[2])
    base_u = float(net.layer1.stsp_U)
    masks = build_layer1_region_masks(batch.sample_images, batch.distractor_images, channels=channels)
    condition_to_state = {
        CONDITION_SAMPLE: state_bank[CONDITION_SAMPLE].state_by_layer["layer1"]["g_pre"],
        CONDITION_DISTRACTOR: state_bank[CONDITION_DISTRACTOR].state_by_layer["layer1"]["g_pre"],
        CONDITION_FULL: state_bank[CONDITION_FULL].state_by_layer["layer1"]["g_pre"],
    }
    for row_idx, triplet_row in enumerate(batch_df.itertuples(index=False)):
        for condition_name, region_state in condition_to_state.items():
            g_vec = np.asarray(region_state[row_idx], dtype=np.float64)
            dg_vec = g_vec - float(base_u)
            for region_name in REGION_NAMES:
                region_mask = np.asarray(masks[region_name][row_idx], dtype=bool)
                if not np.any(region_mask):
                    mean_g = np.nan
                    mean_dg = np.nan
                    changed_fraction = np.nan
                    region_size = 0
                else:
                    region_g = g_vec[region_mask]
                    region_dg = dg_vec[region_mask]
                    mean_g = float(region_g.mean())
                    mean_dg = float(region_dg.mean())
                    changed_fraction = float((np.abs(region_dg) > float(epsilon)).mean())
                    region_size = int(region_mask.sum())
                rows.append(
                    {
                        "record_type": "trial_level",
                        "triplet_id": int(triplet_row.triplet_id),
                        "condition": str(condition_name),
                        "layer": "layer1",
                        "region": str(region_name),
                        "region_size": int(region_size),
                        "mean_g": mean_g,
                        "mean_dg": mean_dg,
                        "changed_fraction": changed_fraction,
                    }
                )
    return rows


def _projection_metrics(full_vec: np.ndarray, template_vec: np.ndarray, eps: float) -> tuple[float, float]:
    full = np.asarray(full_vec, dtype=np.float64)
    template = np.asarray(template_vec, dtype=np.float64)
    full_centered = full - full.mean()
    template_centered = template - template.mean()
    template_norm = max(float(np.linalg.norm(template_centered)), float(eps))
    cosine = float(np.dot(full_centered, template_centered) / max(np.linalg.norm(full_centered) * template_norm, float(eps)))
    projection = float(np.dot(full_centered, template_centered / template_norm))
    return projection, cosine


def compute_projection_rows(
    batch_df: pd.DataFrame,
    state_bank: Mapping[str, BackboneSnapshot],
    eps: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_key in ("layer2", "layer3"):
        full_g = state_bank[CONDITION_FULL].state_by_layer[layer_key]["g_pre"]
        sample_g = state_bank[CONDITION_SAMPLE].state_by_layer[layer_key]["g_pre"]
        distractor_g = state_bank[CONDITION_DISTRACTOR].state_by_layer[layer_key]["g_pre"]
        for row_idx, triplet_row in enumerate(batch_df.itertuples(index=False)):
            proj_s, cos_s = _projection_metrics(full_g[row_idx], sample_g[row_idx], eps=eps)
            proj_d, cos_d = _projection_metrics(full_g[row_idx], distractor_g[row_idx], eps=eps)
            dominance = (proj_s - proj_d) / max(abs(proj_s) + abs(proj_d), float(eps))
            rows.append(
                {
                    "record_type": "trial_level",
                    "triplet_id": int(triplet_row.triplet_id),
                    "layer": str(layer_key),
                    "proj_full_on_S": float(proj_s),
                    "proj_full_on_D": float(proj_d),
                    "cosine_full_S": float(cos_s),
                    "cosine_full_D": float(cos_d),
                    "dominance_in_subspace": float(dominance),
                }
            )
    return rows


def compute_ping_coupling_rows(
    batch_df: pd.DataFrame,
    similarity_rows: Sequence[Mapping[str, object]],
    decomposition_rows: Sequence[Mapping[str, object]],
    ping_readout: PingReadout,
    num_bins: int,
) -> list[dict[str, object]]:
    df_sim = pd.DataFrame(similarity_rows)
    df_decomp = pd.DataFrame(decomposition_rows)[["triplet_id", "layer", "residual_norm", "R2", "alpha", "beta"]]
    merged = df_sim.merge(df_decomp, on=["triplet_id", "layer"], how="left", validate="one_to_one")
    trial_info = batch_df[
        [
            "triplet_id",
            "sample_label",
            "distractor_label",
            "probe_label",
            "sample_id",
            "distractor_id",
            "probe_id",
        ]
    ].copy()
    ping_df = trial_info.copy()
    ping_df["condition"] = PING_CONDITION
    ping_df["first_fire_pred"] = ping_readout.first_fire_pred
    ping_df["first_fire_t"] = ping_readout.first_fire_t
    ping_df["silent"] = ping_readout.silent_mask
    ping_df["sample_first"] = ping_readout.sample_first
    ping_df["distractor_first"] = ping_readout.distractor_first
    ping_df["member_first"] = ping_readout.member_first
    ping_df["nonmember_first"] = ping_readout.nonmember_first
    rows: list[dict[str, object]] = []
    for layer_key, sub in merged.groupby("layer", sort=True):
        di_bins = assign_bins_from_values(sub["DI"].to_numpy(dtype=np.float64, copy=False), num_bins=num_bins)
        sub_layer = sub.copy()
        sub_layer["DI_bin"] = di_bins
        sub_layer = sub_layer.merge(ping_df, on="triplet_id", how="left", validate="one_to_one")
        for record in sub_layer.to_dict(orient="records"):
            record["record_type"] = "trial_level"
            rows.append(record)
        for di_bin, di_sub in sub_layer.groupby("DI_bin", sort=True):
            rows.append(
                {
                    "record_type": "binned_summary",
                    "layer": str(layer_key),
                    "DI_bin": str(di_bin),
                    "trial_count": int(len(di_sub)),
                    "DI_mean": float(di_sub["DI"].mean()),
                    "sample_first_prob": float(di_sub["sample_first"].mean()),
                    "distractor_first_prob": float(di_sub["distractor_first"].mean()),
                    "member_first_prob": float(di_sub["member_first"].mean()),
                    "silent_prob": float(di_sub["silent"].mean()),
                }
            )
        corr = np.nan
        di_np = sub_layer["DI"].to_numpy(dtype=np.float64)
        sample_first_np = sub_layer["sample_first"].to_numpy(dtype=np.float64)
        if len(sub_layer) >= 2 and np.nanstd(di_np) > 0.0 and np.nanstd(sample_first_np) > 0.0:
            corr = float(np.corrcoef(di_np, sample_first_np)[0, 1])
        rows.append(
            {
                "record_type": "layer_summary",
                "layer": str(layer_key),
                "trial_count": int(len(sub_layer)),
                "sample_first_prob": float(sub_layer["sample_first"].mean()),
                "distractor_first_prob": float(sub_layer["distractor_first"].mean()),
                "member_first_prob": float(sub_layer["member_first"].mean()),
                "silent_prob": float(sub_layer["silent"].mean()),
                "DI_sample_first_corr": corr,
            }
        )
    return rows


def summarize_metric_table(
    rows: Sequence[Mapping[str, object]],
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    if len(rows) == 0:
        return pd.DataFrame(columns=["record_type", *group_cols, *value_cols])
    df = pd.DataFrame(rows)
    df_trials = df[df["record_type"] == "trial_level"].copy()
    if df_trials.empty:
        return df
    df_summary = df_trials.groupby(list(group_cols), dropna=False, as_index=False)[list(value_cols)].mean(numeric_only=True)
    df_summary.insert(0, "record_type", "layer_summary")
    return pd.concat([df, df_summary], axis=0, ignore_index=True, sort=False)


def collect_example_state_npz(
    state_bank: Mapping[str, BackboneSnapshot],
    max_trials: int = 2,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for condition_name, snapshot in state_bank.items():
        for layer_key in LAYER_KEYS:
            for state_name in ("u_pre", "x_pre", "g_pre"):
                arr = snapshot.state_by_layer[layer_key][state_name]
                out[f"{condition_name}_{layer_key}_{state_name}"] = arr[:max_trials].astype(np.float32, copy=False)
    return out


def save_overview_figure(
    layout,
    df_similarity: pd.DataFrame,
    df_decomposition: pd.DataFrame,
    df_changed: pd.DataFrame,
    df_coupling: pd.DataFrame,
) -> dict[str, str]:
    apply_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.flatten()

    ax_a.axis("off")
    ax_a.text(0.03, 0.88, "A. Pre-ping taxonomy design", fontsize=13, weight="bold")
    ax_a.text(0.03, 0.64, "sample -> delay1 -> distractor -> delay2 -> snapshot", fontsize=11)
    ax_a.text(0.03, 0.48, "Compare: full / sample-only / distractor-only / shuffled", fontsize=11)
    ax_a.text(0.03, 0.32, "Readout: restore STSP only -> neutral ping -> first-fire", fontsize=11)
    ax_a.text(0.03, 0.16, "Target: coexistence vs biased integration vs pair-specific state", fontsize=11)

    sim_summary = df_similarity[df_similarity["record_type"] == "layer_summary"].copy()
    layer_order = list(LAYER_KEYS)
    x = np.arange(len(layer_order))
    for metric, color, label in (
        ("Sim_FS", SAMPLE_COLOR, "full vs sample"),
        ("Sim_FD", SHUFFLE_COLOR, "full vs distractor"),
        ("Sim_FShuffle", NOISE_COLOR, "full vs shuffled"),
    ):
        vals = [
            float(sim_summary.loc[sim_summary["layer"] == layer_key, metric].mean())
            if np.any(sim_summary["layer"] == layer_key)
            else np.nan
            for layer_key in layer_order
        ]
        ax_b.plot(x, vals, marker="o", linewidth=2.0, color=color, label=label)
    ax_b.set_xticks(x, layer_order)
    ax_b.set_title("B. Similarity")
    ax_b.set_ylabel("Centered cosine")
    ax_b.legend(frameon=False, fontsize=9)

    di_vals = [
        float(sim_summary.loc[sim_summary["layer"] == layer_key, "DI"].mean())
        if np.any(sim_summary["layer"] == layer_key)
        else np.nan
        for layer_key in layer_order
    ]
    ax_c.bar(x, di_vals, color=DYNAMIC_COLOR, width=0.6)
    ax_c.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_c.set_xticks(x, layer_order)
    ax_c.set_title("C. Dominance Index")
    ax_c.set_ylabel("DI")

    decomp_summary = df_decomposition[df_decomposition["record_type"] == "layer_summary"].copy()
    for metric, color in (
        ("alpha", SAMPLE_COLOR),
        ("beta", SHUFFLE_COLOR),
        ("R2", DYNAMIC_COLOR),
        ("residual_norm", NOISE_COLOR),
    ):
        vals = [
            float(decomp_summary.loc[decomp_summary["layer"] == layer_key, metric].mean())
            if np.any(decomp_summary["layer"] == layer_key)
            else np.nan
            for layer_key in layer_order
        ]
        ax_d.plot(x, vals, marker="o", linewidth=2.0, color=color, label=metric)
    ax_d.set_xticks(x, layer_order)
    ax_d.set_title("D. Linear Decomposition")
    ax_d.legend(frameon=False, fontsize=9)

    changed_summary = df_changed[df_changed["record_type"] == "layer_summary"].copy()
    bottom = np.zeros(len(layer_order), dtype=np.float64)
    for metric, color, label in (
        ("S_only_changed_fraction", SAMPLE_COLOR, "S-only"),
        ("D_only_changed_fraction", SHUFFLE_COLOR, "D-only"),
        ("Shared_changed_fraction", DYNAMIC_COLOR, "shared"),
        ("Full_only_novel_changed_fraction", NOISE_COLOR, "full-only novel"),
    ):
        vals = np.asarray(
            [
                float(changed_summary.loc[changed_summary["layer"] == layer_key, metric].mean())
                if np.any(changed_summary["layer"] == layer_key)
                else np.nan
                for layer_key in layer_order
            ],
            dtype=np.float64,
        )
        ax_e.bar(x, vals, bottom=bottom, color=color, width=0.6, label=label)
        bottom = np.nan_to_num(bottom) + np.nan_to_num(vals)
    ax_e.set_xticks(x, layer_order)
    ax_e.set_title("E. Changed-Synapse Taxonomy")
    ax_e.set_ylabel("Fraction")
    ax_e.legend(frameon=False, fontsize=9)

    coupling_bins = df_coupling[df_coupling["record_type"] == "binned_summary"].copy()
    for layer_key, color in zip(layer_order, (SAMPLE_COLOR, DYNAMIC_COLOR, SHUFFLE_COLOR)):
        sub = coupling_bins[coupling_bins["layer"] == layer_key].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("DI_mean", kind="stable")
        ax_f.plot(
            sub["DI_mean"].to_numpy(dtype=np.float64),
            sub["sample_first_prob"].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=2.0,
            color=color,
            label=layer_key,
        )
    ax_f.set_title("F. DI vs Sample-First")
    ax_f.set_xlabel("DI bin mean")
    ax_f.set_ylabel("Sample-first prob.")
    ax_f.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    figure_paths = save_figure_all_formats(fig, layout.figure_base("chunk_stsp_state_taxonomy_overview"))
    plt.close(fig)
    return figure_paths


def save_full_conditioned_changed_figure(
    layout,
    df_changed: pd.DataFrame,
) -> dict[str, str]:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax_a, ax_b = axes
    layer_order = list(LAYER_KEYS)
    x = np.arange(len(layer_order))
    changed_summary = df_changed[df_changed["record_type"] == "layer_summary"].copy()

    changed_fraction_full = np.asarray(
        [
            float(changed_summary.loc[changed_summary["layer"] == layer_key, "changed_fraction_full"].mean())
            if np.any(changed_summary["layer"] == layer_key)
            else np.nan
            for layer_key in layer_order
        ],
        dtype=np.float64,
    )
    ax_a.bar(x, changed_fraction_full, color=DYNAMIC_COLOR, width=0.6)
    ax_a.set_xticks(x, layer_order)
    ax_a.set_ylabel("Fraction")
    ax_a.set_title("A. Full changed fraction")

    bottom = np.zeros(len(layer_order), dtype=np.float64)
    for metric, color, label in (
        ("P_S_only_given_full", SAMPLE_COLOR, "S-only"),
        ("P_D_only_given_full", SHUFFLE_COLOR, "D-only"),
        ("P_Shared_given_full", DYNAMIC_COLOR, "Shared"),
        ("P_Novel_given_full", NOISE_COLOR, "Novel"),
    ):
        vals = np.asarray(
            [
                float(changed_summary.loc[changed_summary["layer"] == layer_key, metric].mean())
                if np.any(changed_summary["layer"] == layer_key)
                else np.nan
                for layer_key in layer_order
            ],
            dtype=np.float64,
        )
        ax_b.bar(x, vals, bottom=bottom, color=color, width=0.6, label=label)
        bottom = np.where(np.isnan(vals), bottom, np.nan_to_num(bottom) + np.nan_to_num(vals))
    ax_b.set_xticks(x, layer_order)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel("Proportion within full-changed")
    ax_b.set_title("B. Composition within full-changed synapses")
    ax_b.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    figure_paths = save_figure_all_formats(fig, layout.figure_base("chunk_stsp_full_conditioned_changed"))
    plt.close(fig)
    return figure_paths


def build_summary_payload(
    cfg: TaxonomyConfig,
    df_triplets: pd.DataFrame,
    df_similarity: pd.DataFrame,
    df_decomposition: pd.DataFrame,
    df_changed: pd.DataFrame,
    df_region: pd.DataFrame,
    df_projection: pd.DataFrame,
    df_coupling: pd.DataFrame,
    exported_files: Mapping[str, object],
) -> dict[str, object]:
    sim_summary = df_similarity[df_similarity["record_type"] == "layer_summary"].copy()
    decomp_summary = df_decomposition[df_decomposition["record_type"] == "layer_summary"].copy()
    changed_summary = df_changed[df_changed["record_type"] == "layer_summary"].copy()
    region_summary = df_region[df_region["record_type"] == "layer_summary"].copy()
    projection_summary = df_projection[df_projection["record_type"] == "layer_summary"].copy()
    coupling_summary = df_coupling[df_coupling["record_type"] == "layer_summary"].copy()

    layer_similarity_summary = {
        str(layer_key): {
            "mean_Sim_FS": float(sim_summary.loc[sim_summary["layer"] == layer_key, "Sim_FS"].mean()),
            "mean_Sim_FD": float(sim_summary.loc[sim_summary["layer"] == layer_key, "Sim_FD"].mean()),
            "mean_Sim_FShuffle": float(sim_summary.loc[sim_summary["layer"] == layer_key, "Sim_FShuffle"].mean()),
            "mean_DI": float(sim_summary.loc[sim_summary["layer"] == layer_key, "DI"].mean()),
        }
        for layer_key in LAYER_KEYS
    }
    layer_decomposition_summary = {
        str(layer_key): {
            "mean_alpha": float(decomp_summary.loc[decomp_summary["layer"] == layer_key, "alpha"].mean()),
            "mean_beta": float(decomp_summary.loc[decomp_summary["layer"] == layer_key, "beta"].mean()),
            "mean_R2": float(decomp_summary.loc[decomp_summary["layer"] == layer_key, "R2"].mean()),
            "mean_residual_norm": float(decomp_summary.loc[decomp_summary["layer"] == layer_key, "residual_norm"].mean()),
        }
        for layer_key in LAYER_KEYS
    }
    layer_changed_summary = {
        str(layer_key): {
            "changed_fraction_sample": float(changed_summary.loc[changed_summary["layer"] == layer_key, "changed_fraction_sample"].mean()),
            "changed_fraction_distractor": float(changed_summary.loc[changed_summary["layer"] == layer_key, "changed_fraction_distractor"].mean()),
            "changed_fraction_full": float(changed_summary.loc[changed_summary["layer"] == layer_key, "changed_fraction_full"].mean()),
            "Full_only_novel_changed_fraction": float(changed_summary.loc[changed_summary["layer"] == layer_key, "Full_only_novel_changed_fraction"].mean()),
        }
        for layer_key in LAYER_KEYS
    }
    layer1_region_summary = {
        str(region_name): {
            condition_name: {
                "mean_g": float(
                    region_summary.loc[
                        (region_summary["region"] == region_name) & (region_summary["condition"] == condition_name),
                        "mean_g",
                    ].mean()
                ),
                "mean_dg": float(
                    region_summary.loc[
                        (region_summary["region"] == region_name) & (region_summary["condition"] == condition_name),
                        "mean_dg",
                    ].mean()
                ),
                "changed_fraction": float(
                    region_summary.loc[
                        (region_summary["region"] == region_name) & (region_summary["condition"] == condition_name),
                        "changed_fraction",
                    ].mean()
                ),
            }
            for condition_name in (CONDITION_SAMPLE, CONDITION_DISTRACTOR, CONDITION_FULL)
        }
        for region_name in REGION_NAMES
    }
    higher_layer_projection_summary = {
        str(layer_key): {
            "mean_proj_full_on_S": float(projection_summary.loc[projection_summary["layer"] == layer_key, "proj_full_on_S"].mean()),
            "mean_proj_full_on_D": float(projection_summary.loc[projection_summary["layer"] == layer_key, "proj_full_on_D"].mean()),
            "mean_dominance_in_subspace": float(projection_summary.loc[projection_summary["layer"] == layer_key, "dominance_in_subspace"].mean()),
        }
        for layer_key in ("layer2", "layer3")
    }
    ping_coupling_summary = {
        str(layer_key): {
            "sample_first_prob": float(coupling_summary.loc[coupling_summary["layer"] == layer_key, "sample_first_prob"].mean()),
            "distractor_first_prob": float(coupling_summary.loc[coupling_summary["layer"] == layer_key, "distractor_first_prob"].mean()),
            "member_first_prob": float(coupling_summary.loc[coupling_summary["layer"] == layer_key, "member_first_prob"].mean()),
            "silent_prob": float(coupling_summary.loc[coupling_summary["layer"] == layer_key, "silent_prob"].mean()),
            "DI_sample_first_corr": float(coupling_summary.loc[coupling_summary["layer"] == layer_key, "DI_sample_first_corr"].mean()),
        }
        for layer_key in LAYER_KEYS
    }
    full_conditioned_changed_summary = {
        str(layer_key): {
            "changed_fraction_full": float(changed_summary.loc[changed_summary["layer"] == layer_key, "changed_fraction_full"].mean()),
            "P_S_only_given_full": float(changed_summary.loc[changed_summary["layer"] == layer_key, "P_S_only_given_full"].mean()),
            "P_D_only_given_full": float(changed_summary.loc[changed_summary["layer"] == layer_key, "P_D_only_given_full"].mean()),
            "P_Shared_given_full": float(changed_summary.loc[changed_summary["layer"] == layer_key, "P_Shared_given_full"].mean()),
            "P_Novel_given_full": float(changed_summary.loc[changed_summary["layer"] == layer_key, "P_Novel_given_full"].mean()),
        }
        for layer_key in LAYER_KEYS
    }
    return {
        "triplet_count": int(len(df_triplets)),
        "epsilon": float(cfg.epsilon),
        "layer_similarity_summary": layer_similarity_summary,
        "layer_decomposition_summary": layer_decomposition_summary,
        "layer_changed_summary": layer_changed_summary,
        "full_conditioned_changed_summary": full_conditioned_changed_summary,
        "layer1_region_summary": layer1_region_summary,
        "higher_layer_projection_summary": higher_layer_projection_summary,
        "ping_coupling_summary": ping_coupling_summary,
        "exported_files": dict(exported_files),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = normalize_config(args)
    layout = prepare_result_layout(cfg.output_dir)
    data_dir = layout.data_dir
    metrics_dir = layout.metrics_dir
    meta_dir = layout.meta_dir
    log_lines: list[str] = []

    device, device_message = resolve_device_with_fallback(cfg.device)
    log_and_print(log_lines, device_message)
    run_config_payload = asdict(cfg)
    save_run_config(run_config_payload, layout.root)
    save_run_config(run_config_payload, meta_dir, filename="run_config.snapshot.json")

    seed_everything(int(cfg.seed))
    log_and_print(log_lines, f"[Setup] seed={cfg.seed}")
    log_and_print(log_lines, f"[Setup] loading dataset split={cfg.split} from {cfg.dataset_root}")
    dataset = load_mnist_dataset(cfg.dataset_root, cfg.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    class_index = build_class_index(dataset, num_classes=10)
    df_triplets = build_triplet_specs(images, labels, flat_normalized, class_index, cfg)
    triplets_csv = save_tidy_csv(df_triplets, data_dir / "triplets.csv", sort_by=["triplet_id"])
    log_and_print(log_lines, f"[Data] triplets={len(df_triplets)} saved={triplets_csv}")

    net, encoder = load_model_and_encoder(
        model_path=cfg.model_path,
        device=device,
        dt=cfg.dt,
        max_duration_ms=cfg.max_duration_ms,
    )
    log_and_print(log_lines, f"[Model] loaded {cfg.model_path}")

    similarity_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    changed_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    ping_rows: list[dict[str, object]] = []
    example_state_dump: dict[str, np.ndarray] | None = None

    for batch in prepare_triplet_batches(df_triplets, images, encoder, cfg, device):
        log_and_print(
            log_lines,
            f"[Batch] id={batch.batch_id} size={len(batch.df)} triplet_range={int(batch.df['triplet_id'].min())}-{int(batch.df['triplet_id'].max())}",
        )
        state_bank = {
            CONDITION_FULL: run_to_preprobe_boundary_and_snapshot_stsp(
                net,
                cfg,
                batch.sample_spikes,
                batch.distractor_spikes,
                stsp_mode="dynamic",
                condition=CONDITION_FULL,
            ),
            CONDITION_SAMPLE: run_to_preprobe_boundary_and_snapshot_stsp(
                net,
                cfg,
                batch.sample_spikes,
                batch.zero_distractor_spikes,
                stsp_mode="dynamic",
                condition=CONDITION_SAMPLE,
            ),
            CONDITION_DISTRACTOR: run_to_preprobe_boundary_and_snapshot_stsp(
                net,
                cfg,
                batch.zero_sample_spikes,
                batch.distractor_spikes,
                stsp_mode="dynamic",
                condition=CONDITION_DISTRACTOR,
            ),
            CONDITION_SHUFFLE: run_to_preprobe_boundary_and_snapshot_stsp(
                net,
                cfg,
                batch.sample_spikes,
                batch.wrong_distractor_spikes,
                stsp_mode="dynamic",
                condition=CONDITION_SHUFFLE,
            ),
        }
        if example_state_dump is None:
            example_state_dump = collect_example_state_npz(state_bank, max_trials=min(2, len(batch.df)))

        batch_similarity = compute_similarity_rows(batch.df, state_bank)
        batch_decomposition = compute_decomposition_rows(batch.df, state_bank, net=net, eps=1e-12)
        batch_changed = compute_changed_synapse_rows(batch.df, state_bank, net=net, epsilon=cfg.epsilon)
        batch_region = compute_layer1_region_rows(batch.df, batch, state_bank, net=net, epsilon=cfg.epsilon)
        batch_projection = compute_projection_rows(batch.df, state_bank, eps=1e-12)
        ping_readout = run_neutral_ping_branch(net, cfg, state_bank[CONDITION_FULL], batch.df)
        batch_ping = compute_ping_coupling_rows(
            batch.df,
            similarity_rows=batch_similarity,
            decomposition_rows=batch_decomposition,
            ping_readout=ping_readout,
            num_bins=cfg.num_sim_bins,
        )

        similarity_rows.extend(batch_similarity)
        decomposition_rows.extend(batch_decomposition)
        changed_rows.extend(batch_changed)
        region_rows.extend(batch_region)
        projection_rows.extend(batch_projection)
        ping_rows.extend(batch_ping)

    df_similarity = summarize_metric_table(
        similarity_rows,
        group_cols=["layer"],
        value_cols=["Sim_FS", "Sim_FD", "Sim_FShuffle", "raw_Sim_FS", "raw_Sim_FD", "raw_Sim_FShuffle", "DI"],
    )
    df_decomposition = summarize_metric_table(
        decomposition_rows,
        group_cols=["layer"],
        value_cols=["alpha", "beta", "c", "R2", "residual_norm"],
    )
    df_changed = summarize_metric_table(
        changed_rows,
        group_cols=["layer"],
        value_cols=[
            "changed_fraction_sample",
            "changed_fraction_distractor",
            "changed_fraction_full",
            "changed_fraction_shuffle",
            "changed_magnitude_mean_sample",
            "changed_magnitude_mean_distractor",
            "changed_magnitude_mean_full",
            "positive_fraction_full",
            "negative_fraction_full",
            "S_only_changed_fraction",
            "D_only_changed_fraction",
            "Shared_changed_fraction",
            "Full_only_novel_changed_fraction",
            "Full_union_supported_changed_fraction",
            "full_changed_count",
            "P_S_only_given_full",
            "P_D_only_given_full",
            "P_Shared_given_full",
            "P_Novel_given_full",
        ],
    )
    df_region = summarize_metric_table(
        region_rows,
        group_cols=["layer", "condition", "region"],
        value_cols=["region_size", "mean_g", "mean_dg", "changed_fraction"],
    )
    df_projection = summarize_metric_table(
        projection_rows,
        group_cols=["layer"],
        value_cols=["proj_full_on_S", "proj_full_on_D", "cosine_full_S", "cosine_full_D", "dominance_in_subspace"],
    )
    df_ping = pd.DataFrame(ping_rows)

    similarity_csv = save_tidy_csv(df_similarity, metrics_dir / "state_similarity_metrics.csv")
    decomposition_csv = save_tidy_csv(df_decomposition, metrics_dir / "state_decomposition_metrics.csv")
    changed_csv = save_tidy_csv(df_changed, metrics_dir / "state_changed_synapse_metrics.csv")
    region_csv = save_tidy_csv(df_region, metrics_dir / "layer1_region_metrics.csv")
    projection_csv = save_tidy_csv(df_projection, metrics_dir / "higher_layer_projection_metrics.csv")
    ping_csv = save_tidy_csv(df_ping, metrics_dir / "ping_coupling_metrics.csv")

    example_npz_path = data_dir / "example_stsp_states.npz"
    if example_state_dump is None:
        example_state_dump = {}
    np.savez_compressed(example_npz_path, **example_state_dump)

    figure_paths: dict[str, str] = {}
    strict_full_changed_figure_paths: dict[str, str] = {}
    if not cfg.skip_figures:
        figure_paths = save_overview_figure(layout, df_similarity, df_decomposition, df_changed, df_ping)
        strict_full_changed_figure_paths = save_full_conditioned_changed_figure(layout, df_changed)

    exported_files: dict[str, object] = {
        "triplets_csv": triplets_csv,
        "state_similarity_metrics_csv": similarity_csv,
        "state_decomposition_metrics_csv": decomposition_csv,
        "state_changed_synapse_metrics_csv": changed_csv,
        "layer1_region_metrics_csv": region_csv,
        "higher_layer_projection_metrics_csv": projection_csv,
        "ping_coupling_metrics_csv": ping_csv,
        "example_stsp_states_npz": str(example_npz_path),
        "figure": figure_paths,
        "full_conditioned_changed_figure": strict_full_changed_figure_paths,
    }
    log_path = str(layout.log_file())
    exported_files["log_file"] = log_path
    summary_path = str(layout.root_file("summary.json"))
    exported_files["summary_json"] = summary_path
    summary_payload = build_summary_payload(
        cfg=cfg,
        df_triplets=df_triplets,
        df_similarity=df_similarity,
        df_decomposition=df_decomposition,
        df_changed=df_changed,
        df_region=df_region,
        df_projection=df_projection,
        df_coupling=df_ping,
        exported_files=exported_files,
    )
    save_summary_json(summary_payload, layout.root)
    save_summary_json(summary_payload, metrics_dir, filename="summary.json")
    save_run_config(
        {
            "experiment_name": "chunk_stsp_state_taxonomy",
            "triplet_count": int(len(df_triplets)),
            "state_similarity_metrics_csv": similarity_csv,
            "state_decomposition_metrics_csv": decomposition_csv,
            "state_changed_synapse_metrics_csv": changed_csv,
            "higher_layer_projection_metrics_csv": projection_csv,
            "ping_coupling_metrics_csv": ping_csv,
        },
        metrics_dir,
        filename="main_metrics.json",
    )

    log_and_print(log_lines, f"[Done] summary={summary_path}")
    save_log_lines(log_lines, layout.log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
