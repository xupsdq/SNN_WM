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

ORDER_A = "order_A_sample_first"
ORDER_B = "order_B_distractor_first"


@dataclass(frozen=True)
class ValidationSuiteConfig:
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
    delay3_ms: float
    ping_ms: float
    ping_amp: float
    batch_size: int
    max_probes: int
    samples_per_probe: int
    max_triplets: int
    num_sim_bins: int
    epsilon: float
    skip_figures: bool
    smoke: bool
    dt: float = 1.0 * ms
    delay_scan_ms: tuple[float, ...] = (0.0, 25.0, 50.0, 100.0, 200.0, 400.0)

    @property
    def sample_steps(self) -> int:
        return ms_to_steps(self.sample_ms, self.dt)

    @property
    def delay1_steps(self) -> int:
        return ms_to_steps(self.delay1_ms, self.dt)

    @property
    def distractor_steps(self) -> int:
        return ms_to_steps(self.distractor_ms, self.dt)

    @property
    def delay2_steps(self) -> int:
        return ms_to_steps(self.delay2_ms, self.dt)

    @property
    def delay3_steps(self) -> int:
        return ms_to_steps(self.delay3_ms, self.dt)

    @property
    def ping_steps(self) -> int:
        return ms_to_steps(self.ping_ms, self.dt)

    @property
    def max_duration_ms(self) -> float:
        return max(self.sample_ms, self.distractor_ms, self.ping_ms, 100.0)


@dataclass
class ValidationBatch:
    batch_id: int
    df: pd.DataFrame
    spikes_by_name: Dict[str, torch.Tensor]


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
    target_first: Dict[str, np.ndarray]


def ms_to_steps(duration_ms: float, dt: float) -> int:
    return int(round((float(duration_ms) * ms) / float(dt)))


def parse_float_list(spec: str) -> tuple[float, ...]:
    values = [float(token.strip()) for token in str(spec).split(",") if token.strip()]
    if not values:
        raise ValueError("delay-scan-ms must include at least one numeric value.")
    return tuple(values)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk STSP validation suite.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PATH_CONFIG.results_root / "chunk_stsp_validation_suite"),
    )
    parser.add_argument("--sample-ms", type=float, default=50.0)
    parser.add_argument("--delay1-ms", type=float, default=50.0)
    parser.add_argument("--distractor-ms", type=float, default=50.0)
    parser.add_argument("--delay2-ms", type=float, default=50.0)
    parser.add_argument("--delay3-ms", type=float, default=50.0)
    parser.add_argument("--ping-ms", type=float, default=30.0)
    parser.add_argument("--ping-amp", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-probes", type=int, default=20)
    parser.add_argument("--samples-per-probe", type=int, default=25)
    parser.add_argument("--max-triplets", type=int, default=500)
    parser.add_argument("--num-sim-bins", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--delay-scan-ms", type=str, default="0,25,50,100,200,400")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def normalize_config(args: argparse.Namespace) -> ValidationSuiteConfig:
    cfg = ValidationSuiteConfig(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        split=str(args.split),
        device=str(args.device),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        sample_ms=float(args.sample_ms),
        delay1_ms=float(args.delay1_ms),
        distractor_ms=float(args.distractor_ms),
        delay2_ms=float(args.delay2_ms),
        delay3_ms=float(args.delay3_ms),
        ping_ms=float(args.ping_ms),
        ping_amp=float(args.ping_amp),
        batch_size=int(args.batch_size),
        max_probes=int(args.max_probes),
        samples_per_probe=int(args.samples_per_probe),
        max_triplets=int(args.max_triplets),
        num_sim_bins=int(args.num_sim_bins),
        epsilon=float(args.epsilon),
        skip_figures=bool(args.skip_figures),
        smoke=bool(args.smoke),
        delay_scan_ms=parse_float_list(args.delay_scan_ms),
    )
    if cfg.smoke:
        reduced_scan = cfg.delay_scan_ms[: min(len(cfg.delay_scan_ms), 3)]
        cfg = ValidationSuiteConfig(
            **{
                **asdict(cfg),
                "batch_size": min(int(cfg.batch_size), 2),
                "max_probes": min(int(cfg.max_probes), 2),
                "samples_per_probe": min(int(cfg.samples_per_probe), 2),
                "max_triplets": min(int(cfg.max_triplets), 4),
                "num_sim_bins": min(int(cfg.num_sim_bins), 3),
                "delay_scan_ms": tuple(reduced_scan),
            }
        )
    if min(cfg.sample_steps, cfg.distractor_steps, cfg.ping_steps) <= 0:
        raise ValueError("Sample, distractor, and ping durations must map to at least one step.")
    if min(cfg.delay1_steps, cfg.delay2_steps, cfg.delay3_steps) < 0:
        raise ValueError("Delay steps must be non-negative.")
    return cfg


def resolve_device_with_fallback(device_arg: str) -> tuple[torch.device, str]:
    device_str = str(device_arg).strip().lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu"), "[Runtime] CUDA unavailable on 2026-04-17; falling back to CPU."
    return torch.device(device_str), f"[Runtime] Using device={device_str}."


def log_and_print(log_lines: list[str], message: str) -> None:
    print(message, flush=True)
    log_lines.append(str(message))


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return safe_float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def assign_wrong_distractors(
    df_triplets: pd.DataFrame,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Add a random non-member distractor for shuffled second-item controls."""

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
                "wrong_distractor_similarity_to_sample": float(
                    np.dot(flat_normalized[wrong_id], flat_normalized[int(row.sample_id)])
                ),
            }
        )
    out = pd.concat([df_triplets.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return out


def assign_three_item_candidates(
    df_triplets: pd.DataFrame,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Add one random valid third item for the three-input validation."""

    rng = np.random.default_rng(int(seed))
    all_ids = np.arange(len(labels), dtype=np.int64)
    rows: list[dict[str, object]] = []
    for row in df_triplets.itertuples(index=False):
        sample_id = int(row.sample_id)
        distractor_id = int(row.distractor_id)
        probe_id = int(row.probe_id)
        sample_label = int(row.sample_label)
        distractor_label = int(row.distractor_label)
        probe_label = int(row.probe_label)
        valid_mask = (
            (all_ids != sample_id)
            & (all_ids != distractor_id)
            & (all_ids != probe_id)
            & (labels[all_ids] != sample_label)
            & (labels[all_ids] != distractor_label)
            & (labels[all_ids] != probe_label)
        )
        candidate_ids = all_ids[valid_mask]
        if candidate_ids.size <= 0:
            raise RuntimeError("No valid third-item candidate remained after filtering.")
        third_item_id = int(candidate_ids[int(rng.integers(0, candidate_ids.size))])
        rows.append(
            {
                "third_item_id": int(third_item_id),
                "third_item_label": int(labels[third_item_id]),
                "third_item_similarity_to_item1": float(
                    np.dot(flat_normalized[third_item_id], flat_normalized[sample_id])
                ),
            }
        )
    return pd.concat([df_triplets.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def build_validation_triplets(
    images: torch.Tensor,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    cfg: ValidationSuiteConfig,
) -> pd.DataFrame:
    """Build the shared triplet pool used across all validation modules."""

    df_triplets = shared_build_triplet_specs(
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
    df_triplets = assign_wrong_distractors(
        df_triplets=df_triplets,
        labels=labels,
        flat_normalized=flat_normalized,
        seed=mix_seed(cfg.seed, 401),
    )
    df_triplets = assign_three_item_candidates(
        df_triplets=df_triplets,
        labels=labels,
        flat_normalized=flat_normalized,
        seed=mix_seed(cfg.seed, 402),
    )
    df_triplets["selection_metadata"] = "chunk_stsp_validation_suite"
    return df_triplets


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


def prepare_validation_batches(
    df_triplets: pd.DataFrame,
    images: torch.Tensor,
    encoder,
    cfg: ValidationSuiteConfig,
    device: torch.device,
) -> Iterator[ValidationBatch]:
    """Yield encoded batches with all item variants required by the suite."""

    for batch_id, start in enumerate(range(0, len(df_triplets), int(cfg.batch_size))):
        batch_df = df_triplets.iloc[start : start + int(cfg.batch_size)].copy().reset_index(drop=True)
        sample_ids = batch_df["sample_id"].astype(int).tolist()
        other_ids = (
            batch_df["distractor_id"].astype(int).tolist()
            + batch_df["wrong_distractor_id"].astype(int).tolist()
            + batch_df["third_item_id"].astype(int).tolist()
        )
        sample_lookup = build_spike_lookup(
            images,
            encoder,
            sample_ids,
            steps=int(cfg.sample_steps),
            device=device,
        )
        other_lookup = build_spike_lookup(
            images,
            encoder,
            other_ids,
            steps=int(cfg.distractor_steps),
            device=device,
        )
        spikes_by_name = {
            "sample": torch.stack([sample_lookup[int(idx)] for idx in sample_ids], dim=0),
            "distractor": torch.stack(
                [other_lookup[int(idx)] for idx in batch_df["distractor_id"].astype(int).tolist()],
                dim=0,
            ),
            "wrong_distractor": torch.stack(
                [other_lookup[int(idx)] for idx in batch_df["wrong_distractor_id"].astype(int).tolist()],
                dim=0,
            ),
            "third_item": torch.stack(
                [other_lookup[int(idx)] for idx in batch_df["third_item_id"].astype(int).tolist()],
                dim=0,
            ),
        }
        yield ValidationBatch(batch_id=int(batch_id), df=batch_df, spikes_by_name=spikes_by_name)


def zeros_like_sequence(sequence: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(sequence)


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


def snapshot_stsp_state_batch(
    net,
    batch_size: int,
) -> tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    state_by_layer: Dict[str, Dict[str, np.ndarray]] = {}
    restore_by_layer: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        if getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
            raise ValueError(f"{layer_key} is missing STSP state at the snapshot boundary.")
        u_pre = layer.u_pre.detach().view(batch_size, -1)
        x_pre = layer.x_pre.detach().view(batch_size, -1)
        g_pre = (layer.u_pre * layer.x_pre).detach().view(batch_size, -1)
        state_by_layer[str(layer_key)] = {
            "u_pre": u_pre.cpu().numpy().astype(np.float32, copy=False),
            "x_pre": x_pre.cpu().numpy().astype(np.float32, copy=False),
            "g_pre": g_pre.cpu().numpy().astype(np.float32, copy=False),
        }
        restore_by_layer[str(layer_key)] = (
            layer.u_pre.detach().cpu().clone(),
            layer.x_pre.detach().cpu().clone(),
        )
    return state_by_layer, restore_by_layer


def run_sequence_to_snapshot(
    net,
    item_sequences: Sequence[torch.Tensor],
    delay_steps: Sequence[int],
    *,
    stsp_mode: str,
    condition: str,
) -> BackboneSnapshot:
    """Roll a 2-item or 3-item sequence to the pre-ping snapshot boundary."""

    if len(item_sequences) == 0:
        raise ValueError("item_sequences must not be empty.")
    if len(item_sequences) != len(delay_steps):
        raise ValueError("delay_steps must match item_sequences length.")
    first_sequence = item_sequences[0]
    batch_size, _, channels, height, width = first_sequence.shape
    with torch.no_grad():
        prepare_network_state(net, batch_size, channels, height, width)
        layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
        zero_input = torch.zeros(
            (batch_size, channels, height, width),
            dtype=first_sequence.dtype,
            device=first_sequence.device,
        )
        current_time = 0
        for sequence, tail_delay in zip(item_sequences, delay_steps):
            for t_idx in range(int(sequence.shape[1])):
                forward_three_layers(net, sequence[:, t_idx, ...], current_time, stsp_mode=stsp_mode)
                current_time += 1
            for _ in range(int(tail_delay)):
                forward_three_layers(net, zero_input, current_time, stsp_mode=stsp_mode)
                current_time += 1
        state_by_layer, restore_by_layer = snapshot_stsp_state_batch(net, batch_size=batch_size)
    return BackboneSnapshot(
        condition=str(condition),
        layer_input_shapes=layer_input_shapes,
        restore_ux_by_layer=restore_by_layer,
        state_by_layer=state_by_layer,
    )


def run_ping_from_snapshot(
    net,
    cfg: ValidationSuiteConfig,
    backbone: BackboneSnapshot,
    *,
    label_targets: Mapping[str, np.ndarray],
) -> PingReadout:
    """Restore STSP-only memory traces and run a neutral ping readout branch."""

    batch_size = int(len(next(iter(label_targets.values()))))
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
    silent_mask = (pred_np < 0).astype(np.int64, copy=False)
    target_first = {
        str(name): (((silent_mask == 0) & (pred_np == labels)).astype(np.int64, copy=False))
        for name, labels in label_targets.items()
    }
    return PingReadout(
        condition="neutral_ping",
        first_fire_pred=pred_np,
        first_fire_t=fire_t_np,
        silent_mask=silent_mask,
        target_first=target_first,
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


def compute_similarity_against_references(
    full_snapshot: BackboneSnapshot,
    reference_snapshots: Mapping[str, BackboneSnapshot],
    *,
    eps: float = 1e-12,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Compute centered and raw cosine similarity from full state to each reference state."""

    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for layer_key in LAYER_KEYS:
        full_g = full_snapshot.state_by_layer[layer_key]["g_pre"]
        out[str(layer_key)] = {}
        for ref_name, ref_snapshot in reference_snapshots.items():
            ref_g = ref_snapshot.state_by_layer[layer_key]["g_pre"]
            out[str(layer_key)][str(ref_name)] = {
                "centered": centered_cosine_similarity(full_g, ref_g, eps=eps),
                "raw": raw_cosine_similarity(full_g, ref_g, eps=eps),
            }
    return out


def fit_linear_decomposition_against_references(
    full_vec: np.ndarray,
    reference_vectors: Mapping[str, np.ndarray],
    *,
    base_u: float,
    eps: float,
) -> dict[str, float]:
    """Fit full-state gain as a linear mixture of reference gains plus a bias term."""

    y = np.asarray(full_vec, dtype=np.float64)
    ref_names = list(reference_vectors.keys())
    design = np.column_stack([np.asarray(reference_vectors[name], dtype=np.float64) for name in ref_names] + [np.ones_like(y)])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ coef
    residual = y - y_hat
    sse = float(np.sum(np.square(residual)))
    tss = float(np.sum(np.square(y - y.mean())))
    base_vec = np.full_like(y, float(base_u))
    residual_norm = float(np.linalg.norm(residual) / max(np.linalg.norm(y - base_vec), float(eps)))
    out = {str(name): float(coef[idx]) for idx, name in enumerate(ref_names)}
    out["c"] = float(coef[len(ref_names)])
    out["R2"] = float(1.0 - sse / max(tss, float(eps)))
    out["residual_norm"] = residual_norm
    return out


def compute_changed_mask(vec: np.ndarray, *, base_u: float, epsilon: float) -> np.ndarray:
    return (np.abs(np.asarray(vec, dtype=np.float64) - float(base_u)) > float(epsilon)).astype(bool, copy=False)


def compute_strict_full_conditioned_composition(
    full_vec: np.ndarray,
    reference_vectors: Mapping[str, np.ndarray],
    *,
    base_u: float,
    epsilon: float,
    eps: float,
) -> dict[str, float]:
    """Partition full-changed synapses into reference-supported and novel sources."""

    full_mask = compute_changed_mask(full_vec, base_u=base_u, epsilon=epsilon)
    full_count = int(full_mask.sum())
    changed_fraction_full = float(full_mask.mean())
    reference_masks = {
        str(name): compute_changed_mask(vec, base_u=base_u, epsilon=epsilon)
        for name, vec in reference_vectors.items()
    }
    out: dict[str, float] = {
        "changed_fraction_full": changed_fraction_full,
        "full_changed_count": float(full_count),
    }
    if len(reference_masks) == 2 and {"item1", "item2"}.issubset(reference_masks):
        item1_mask = reference_masks["item1"]
        item2_mask = reference_masks["item2"]
        item1_only_in_full = full_mask & item1_mask & (~item2_mask)
        item2_only_in_full = full_mask & item2_mask & (~item1_mask)
        shared_in_full = full_mask & item1_mask & item2_mask
        novel_in_full = full_mask & (~item1_mask) & (~item2_mask)
        if full_count > 0:
            p_item1 = float(item1_only_in_full.sum() / full_count)
            p_item2 = float(item2_only_in_full.sum() / full_count)
            p_shared = float(shared_in_full.sum() / full_count)
            p_novel = float(novel_in_full.sum() / full_count)
        else:
            p_item1 = np.nan
            p_item2 = np.nan
            p_shared = np.nan
            p_novel = np.nan
        out.update(
            {
                "P_item1_only_given_full": p_item1,
                "P_item2_only_given_full": p_item2,
                "P_shared_given_full": p_shared,
                "P_novel_given_full": p_novel,
                "Abs_item1_only_in_full": float(changed_fraction_full * p_item1) if np.isfinite(p_item1) else np.nan,
                "Abs_item2_only_in_full": float(changed_fraction_full * p_item2) if np.isfinite(p_item2) else np.nan,
                "Abs_shared_in_full": float(changed_fraction_full * p_shared) if np.isfinite(p_shared) else np.nan,
                "Abs_novel_in_full": float(changed_fraction_full * p_novel) if np.isfinite(p_novel) else np.nan,
                "supported_fraction_given_full": float(p_item1 + p_item2 + p_shared)
                if np.all(np.isfinite([p_item1, p_item2, p_shared]))
                else np.nan,
                "source_asymmetry_given_full": float((p_item1 - p_item2) / max(p_item1 + p_item2, float(eps)))
                if np.all(np.isfinite([p_item1, p_item2]))
                else np.nan,
                "composition_sum_given_full": float(p_item1 + p_item2 + p_shared + p_novel)
                if np.all(np.isfinite([p_item1, p_item2, p_shared, p_novel]))
                else np.nan,
            }
        )
        return out
    union_supported = np.zeros_like(full_mask, dtype=bool)
    for name, mask in reference_masks.items():
        union_supported |= mask
        only_mask = full_mask & mask
        for other_name, other_mask in reference_masks.items():
            if other_name == name:
                continue
            only_mask &= ~other_mask
        p_only = float(only_mask.sum() / full_count) if full_count > 0 else np.nan
        out[f"P_{name}_only_given_full"] = p_only
    supported_in_full = full_mask & union_supported
    novel_in_full = full_mask & (~union_supported)
    out["P_supported_given_full"] = float(supported_in_full.sum() / full_count) if full_count > 0 else np.nan
    out["P_novel_given_full"] = float(novel_in_full.sum() / full_count) if full_count > 0 else np.nan
    return out


def summarize_metric_table(
    rows: Sequence[Mapping[str, object]],
    *,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    if len(rows) == 0:
        return pd.DataFrame(columns=["record_type", *group_cols, *value_cols])
    df = pd.DataFrame(rows)
    if "record_type" not in df.columns:
        df.insert(0, "record_type", "trial_level")
    df_trials = df[df["record_type"] == "trial_level"].copy()
    if df_trials.empty:
        return df
    summary_df = df_trials.groupby(list(group_cols), dropna=False, as_index=False)[list(value_cols)].mean(numeric_only=True)
    summary_df.insert(0, "record_type", "layer_summary")
    return pd.concat([df, summary_df], axis=0, ignore_index=True, sort=False)


def collect_example_state_npz(
    tag: str,
    snapshots: Mapping[str, BackboneSnapshot],
    *,
    max_trials: int = 2,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for snapshot_name, snapshot in snapshots.items():
        for layer_key in LAYER_KEYS:
            for state_name in ("u_pre", "x_pre", "g_pre"):
                key = f"{tag}_{snapshot_name}_{layer_key}_{state_name}"
                out[key] = snapshot.state_by_layer[layer_key][state_name][:max_trials].astype(np.float32, copy=False)
    return out


def run_sequence_reversal_experiment(
    net,
    cfg: ValidationSuiteConfig,
    df_triplets: pd.DataFrame,
    images: torch.Tensor,
    encoder,
    device: torch.device,
    log_lines: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Test whether dominance follows the first input rather than fixed semantic roles."""

    seq_rows: list[dict[str, object]] = []
    strict_rows: list[dict[str, object]] = []
    example_dump: dict[str, np.ndarray] = {}
    for batch in prepare_validation_batches(df_triplets, images, encoder, cfg, device):
        log_and_print(log_lines, f"[SequenceReversal] batch={batch.batch_id} size={len(batch.df)}")
        sample_labels = batch.df["sample_label"].to_numpy(dtype=np.int64, copy=False)
        distractor_labels = batch.df["distractor_label"].to_numpy(dtype=np.int64, copy=False)
        order_specs = {
            ORDER_A: {
                "first_role": "sample",
                "second_role": "distractor",
                "first_sequence": batch.spikes_by_name["sample"],
                "second_sequence": batch.spikes_by_name["distractor"],
                "first_labels": sample_labels,
                "second_labels": distractor_labels,
            },
            ORDER_B: {
                "first_role": "distractor",
                "second_role": "sample",
                "first_sequence": batch.spikes_by_name["distractor"],
                "second_sequence": batch.spikes_by_name["sample"],
                "first_labels": distractor_labels,
                "second_labels": sample_labels,
            },
        }
        for order_condition, spec in order_specs.items():
            first_sequence = spec["first_sequence"]
            second_sequence = spec["second_sequence"]
            state_bank = {
                "full": run_sequence_to_snapshot(
                    net,
                    [first_sequence, second_sequence],
                    [cfg.delay1_steps, cfg.delay2_steps],
                    stsp_mode="dynamic",
                    condition=f"{order_condition}_full",
                ),
                "item1_only": run_sequence_to_snapshot(
                    net,
                    [first_sequence, zeros_like_sequence(second_sequence)],
                    [cfg.delay1_steps, cfg.delay2_steps],
                    stsp_mode="dynamic",
                    condition=f"{order_condition}_item1_only",
                ),
                "item2_only": run_sequence_to_snapshot(
                    net,
                    [zeros_like_sequence(first_sequence), second_sequence],
                    [cfg.delay1_steps, cfg.delay2_steps],
                    stsp_mode="dynamic",
                    condition=f"{order_condition}_item2_only",
                ),
            }
            if not example_dump:
                example_dump.update(collect_example_state_npz("sequence_reversal", state_bank))
            ping = run_ping_from_snapshot(
                net,
                cfg,
                state_bank["full"],
                label_targets={
                    "first_item": np.asarray(spec["first_labels"], dtype=np.int64),
                    "second_item": np.asarray(spec["second_labels"], dtype=np.int64),
                    "sample": sample_labels,
                    "distractor": distractor_labels,
                },
            )
            similarity = compute_similarity_against_references(
                state_bank["full"],
                {"first": state_bank["item1_only"], "second": state_bank["item2_only"]},
            )
            for layer_key in LAYER_KEYS:
                base_u = float(getattr(net, layer_key).stsp_U)
                full_g = state_bank["full"].state_by_layer[layer_key]["g_pre"]
                first_g = state_bank["item1_only"].state_by_layer[layer_key]["g_pre"]
                second_g = state_bank["item2_only"].state_by_layer[layer_key]["g_pre"]
                sim_first = similarity[layer_key]["first"]["centered"]
                sim_second = similarity[layer_key]["second"]["centered"]
                raw_first = similarity[layer_key]["first"]["raw"]
                raw_second = similarity[layer_key]["second"]["raw"]
                di = (sim_first - sim_second) / np.maximum(sim_first + sim_second, 1e-12)
                for row_idx, triplet_row in enumerate(batch.df.itertuples(index=False)):
                    decomp = fit_linear_decomposition_against_references(
                        full_g[row_idx],
                        {"alpha_first": first_g[row_idx], "beta_second": second_g[row_idx]},
                        base_u=base_u,
                        eps=1e-12,
                    )
                    strict = compute_strict_full_conditioned_composition(
                        full_g[row_idx],
                        {"item1": first_g[row_idx], "item2": second_g[row_idx]},
                        base_u=base_u,
                        epsilon=cfg.epsilon,
                        eps=max(cfg.epsilon, 1e-12),
                    )
                    seq_rows.append(
                        {
                            "record_type": "trial_level",
                            "experiment": "sequence_reversal",
                            "order_condition": str(order_condition),
                            "first_role": str(spec["first_role"]),
                            "second_role": str(spec["second_role"]),
                            "triplet_id": int(triplet_row.triplet_id),
                            "layer": str(layer_key),
                            "Sim_F_first": float(sim_first[row_idx]),
                            "Sim_F_second": float(sim_second[row_idx]),
                            "raw_Sim_F_first": float(raw_first[row_idx]),
                            "raw_Sim_F_second": float(raw_second[row_idx]),
                            "DI_first_second": float(di[row_idx]),
                            "alpha_first": float(decomp["alpha_first"]),
                            "beta_second": float(decomp["beta_second"]),
                            "c": float(decomp["c"]),
                            "R2": float(decomp["R2"]),
                            "residual_norm": float(decomp["residual_norm"]),
                            "first_item_first_prob": int(ping.target_first["first_item"][row_idx]),
                            "second_item_first_prob": int(ping.target_first["second_item"][row_idx]),
                            "sample_first_prob": int(ping.target_first["sample"][row_idx]),
                            "distractor_first_prob": int(ping.target_first["distractor"][row_idx]),
                            "silent_prob": int(ping.silent_mask[row_idx]),
                        }
                    )
                    strict_rows.append(
                        {
                            "record_type": "trial_level",
                            "experiment": "strict_full_conditioned_changed",
                            "order_condition": str(order_condition),
                            "first_role": str(spec["first_role"]),
                            "second_role": str(spec["second_role"]),
                            "triplet_id": int(triplet_row.triplet_id),
                            "layer": str(layer_key),
                            **strict,
                        }
                    )
    df_sequence = summarize_metric_table(
        seq_rows,
        group_cols=["experiment", "order_condition", "first_role", "second_role", "layer"],
        value_cols=[
            "Sim_F_first",
            "Sim_F_second",
            "raw_Sim_F_first",
            "raw_Sim_F_second",
            "DI_first_second",
            "alpha_first",
            "beta_second",
            "c",
            "R2",
            "residual_norm",
            "first_item_first_prob",
            "second_item_first_prob",
            "sample_first_prob",
            "distractor_first_prob",
            "silent_prob",
        ],
    )
    df_strict = summarize_metric_table(
        strict_rows,
        group_cols=["experiment", "order_condition", "first_role", "second_role", "layer"],
        value_cols=[
            "changed_fraction_full",
            "full_changed_count",
            "P_item1_only_given_full",
            "P_item2_only_given_full",
            "P_shared_given_full",
            "P_novel_given_full",
            "Abs_item1_only_in_full",
            "Abs_item2_only_in_full",
            "Abs_shared_in_full",
            "Abs_novel_in_full",
            "supported_fraction_given_full",
            "source_asymmetry_given_full",
            "composition_sum_given_full",
        ],
    )
    return df_sequence, df_strict, example_dump


def run_delay_scan_experiment(
    net,
    cfg: ValidationSuiteConfig,
    df_triplets: pd.DataFrame,
    images: torch.Tensor,
    encoder,
    device: torch.device,
    log_lines: list[str],
) -> pd.DataFrame:
    """Scan delay1 and delay2 to test whether bias tracks STSP time scales."""

    rows: list[dict[str, object]] = []
    for batch in prepare_validation_batches(df_triplets, images, encoder, cfg, device):
        log_and_print(log_lines, f"[DelayScan] batch={batch.batch_id} size={len(batch.df)}")
        sample_sequence = batch.spikes_by_name["sample"]
        distractor_sequence = batch.spikes_by_name["distractor"]
        zero_sample = zeros_like_sequence(sample_sequence)
        zero_distractor = zeros_like_sequence(distractor_sequence)
        sample_labels = batch.df["sample_label"].to_numpy(dtype=np.int64, copy=False)
        distractor_labels = batch.df["distractor_label"].to_numpy(dtype=np.int64, copy=False)
        for scan_type in ("delay1_scan", "delay2_scan"):
            for delay_ms in cfg.delay_scan_ms:
                delay_steps = ms_to_steps(delay_ms, cfg.dt)
                delay1_steps = delay_steps if scan_type == "delay1_scan" else cfg.delay1_steps
                delay2_steps = delay_steps if scan_type == "delay2_scan" else cfg.delay2_steps
                full = run_sequence_to_snapshot(
                    net,
                    [sample_sequence, distractor_sequence],
                    [delay1_steps, delay2_steps],
                    stsp_mode="dynamic",
                    condition=f"{scan_type}_{delay_ms:.1f}_full",
                )
                item1_only = run_sequence_to_snapshot(
                    net,
                    [sample_sequence, zero_distractor],
                    [delay1_steps, delay2_steps],
                    stsp_mode="dynamic",
                    condition=f"{scan_type}_{delay_ms:.1f}_item1_only",
                )
                item2_only = run_sequence_to_snapshot(
                    net,
                    [zero_sample, distractor_sequence],
                    [delay1_steps, delay2_steps],
                    stsp_mode="dynamic",
                    condition=f"{scan_type}_{delay_ms:.1f}_item2_only",
                )
                ping = run_ping_from_snapshot(
                    net,
                    cfg,
                    full,
                    label_targets={"item1": sample_labels, "item2": distractor_labels},
                )
                similarity = compute_similarity_against_references(full, {"item1": item1_only, "item2": item2_only})
                for layer_key in LAYER_KEYS:
                    base_u = float(getattr(net, layer_key).stsp_U)
                    full_g = full.state_by_layer[layer_key]["g_pre"]
                    item1_g = item1_only.state_by_layer[layer_key]["g_pre"]
                    item2_g = item2_only.state_by_layer[layer_key]["g_pre"]
                    sim_item1 = similarity[layer_key]["item1"]["centered"]
                    sim_item2 = similarity[layer_key]["item2"]["centered"]
                    di = (sim_item1 - sim_item2) / np.maximum(sim_item1 + sim_item2, 1e-12)
                    for row_idx, triplet_row in enumerate(batch.df.itertuples(index=False)):
                        decomp = fit_linear_decomposition_against_references(
                            full_g[row_idx],
                            {"alpha": item1_g[row_idx], "beta": item2_g[row_idx]},
                            base_u=base_u,
                            eps=1e-12,
                        )
                        strict = compute_strict_full_conditioned_composition(
                            full_g[row_idx],
                            {"item1": item1_g[row_idx], "item2": item2_g[row_idx]},
                            base_u=base_u,
                            epsilon=cfg.epsilon,
                            eps=max(cfg.epsilon, 1e-12),
                        )
                        rows.append(
                            {
                                "record_type": "trial_level",
                                "experiment": "delay_scan",
                                "scan_type": str(scan_type),
                                "delay_ms": float(delay_ms),
                                "triplet_id": int(triplet_row.triplet_id),
                                "layer": str(layer_key),
                                "DI": float(di[row_idx]),
                                "alpha": float(decomp["alpha"]),
                                "beta": float(decomp["beta"]),
                                "R2": float(decomp["R2"]),
                                "residual_norm": float(decomp["residual_norm"]),
                                "changed_fraction_full": float(strict["changed_fraction_full"]),
                                "P_item1_only_given_full": float(strict["P_item1_only_given_full"]),
                                "P_item2_only_given_full": float(strict["P_item2_only_given_full"]),
                                "P_shared_given_full": float(strict["P_shared_given_full"]),
                                "P_novel_given_full": float(strict["P_novel_given_full"]),
                                "supported_fraction_given_full": float(strict["supported_fraction_given_full"]),
                                "source_asymmetry_given_full": float(strict["source_asymmetry_given_full"]),
                                "item1_first_prob": int(ping.target_first["item1"][row_idx]),
                                "item2_first_prob": int(ping.target_first["item2"][row_idx]),
                                "silent_prob": int(ping.silent_mask[row_idx]),
                            }
                        )
    return summarize_metric_table(
        rows,
        group_cols=["experiment", "scan_type", "delay_ms", "layer"],
        value_cols=[
            "DI",
            "alpha",
            "beta",
            "R2",
            "residual_norm",
            "changed_fraction_full",
            "P_item1_only_given_full",
            "P_item2_only_given_full",
            "P_shared_given_full",
            "P_novel_given_full",
            "supported_fraction_given_full",
            "source_asymmetry_given_full",
            "item1_first_prob",
            "item2_first_prob",
            "silent_prob",
        ],
    )


def run_three_item_experiment(
    net,
    cfg: ValidationSuiteConfig,
    df_triplets: pd.DataFrame,
    images: torch.Tensor,
    encoder,
    device: torch.device,
    log_lines: list[str],
) -> pd.DataFrame:
    """Extend the bias analysis to a three-input backbone."""

    rows: list[dict[str, object]] = []
    for batch in prepare_validation_batches(df_triplets, images, encoder, cfg, device):
        log_and_print(log_lines, f"[ThreeItem] batch={batch.batch_id} size={len(batch.df)}")
        sample_sequence = batch.spikes_by_name["sample"]
        distractor_sequence = batch.spikes_by_name["distractor"]
        third_sequence = batch.spikes_by_name["third_item"]
        zero_sample = zeros_like_sequence(sample_sequence)
        zero_distractor = zeros_like_sequence(distractor_sequence)
        zero_third = zeros_like_sequence(third_sequence)
        sample_labels = batch.df["sample_label"].to_numpy(dtype=np.int64, copy=False)
        distractor_labels = batch.df["distractor_label"].to_numpy(dtype=np.int64, copy=False)
        third_labels = batch.df["third_item_label"].to_numpy(dtype=np.int64, copy=False)
        full = run_sequence_to_snapshot(
            net,
            [sample_sequence, distractor_sequence, third_sequence],
            [cfg.delay1_steps, cfg.delay2_steps, cfg.delay3_steps],
            stsp_mode="dynamic",
            condition="three_item_full123",
        )
        item1_only = run_sequence_to_snapshot(
            net,
            [sample_sequence, zero_distractor, zero_third],
            [cfg.delay1_steps, cfg.delay2_steps, cfg.delay3_steps],
            stsp_mode="dynamic",
            condition="three_item_item1_only",
        )
        item2_only = run_sequence_to_snapshot(
            net,
            [zero_sample, distractor_sequence, zero_third],
            [cfg.delay1_steps, cfg.delay2_steps, cfg.delay3_steps],
            stsp_mode="dynamic",
            condition="three_item_item2_only",
        )
        item3_only = run_sequence_to_snapshot(
            net,
            [zero_sample, zero_distractor, third_sequence],
            [cfg.delay1_steps, cfg.delay2_steps, cfg.delay3_steps],
            stsp_mode="dynamic",
            condition="three_item_item3_only",
        )
        ping = run_ping_from_snapshot(
            net,
            cfg,
            full,
            label_targets={"item1": sample_labels, "item2": distractor_labels, "item3": third_labels},
        )
        similarity = compute_similarity_against_references(
            full,
            {"item1": item1_only, "item2": item2_only, "item3": item3_only},
        )
        for layer_key in LAYER_KEYS:
            base_u = float(getattr(net, layer_key).stsp_U)
            full_g = full.state_by_layer[layer_key]["g_pre"]
            item1_g = item1_only.state_by_layer[layer_key]["g_pre"]
            item2_g = item2_only.state_by_layer[layer_key]["g_pre"]
            item3_g = item3_only.state_by_layer[layer_key]["g_pre"]
            sim_item1 = similarity[layer_key]["item1"]["centered"]
            sim_item2 = similarity[layer_key]["item2"]["centered"]
            sim_item3 = similarity[layer_key]["item3"]["centered"]
            for row_idx, triplet_row in enumerate(batch.df.itertuples(index=False)):
                decomp = fit_linear_decomposition_against_references(
                    full_g[row_idx],
                    {"alpha1": item1_g[row_idx], "alpha2": item2_g[row_idx], "alpha3": item3_g[row_idx]},
                    base_u=base_u,
                    eps=1e-12,
                )
                strict = compute_strict_full_conditioned_composition(
                    full_g[row_idx],
                    {"1": item1_g[row_idx], "2": item2_g[row_idx], "3": item3_g[row_idx]},
                    base_u=base_u,
                    epsilon=cfg.epsilon,
                    eps=max(cfg.epsilon, 1e-12),
                )
                rows.append(
                    {
                        "record_type": "trial_level",
                        "experiment": "three_item",
                        "triplet_id": int(triplet_row.triplet_id),
                        "layer": str(layer_key),
                        "Sim_F1": float(sim_item1[row_idx]),
                        "Sim_F2": float(sim_item2[row_idx]),
                        "Sim_F3": float(sim_item3[row_idx]),
                        "alpha1": float(decomp["alpha1"]),
                        "alpha2": float(decomp["alpha2"]),
                        "alpha3": float(decomp["alpha3"]),
                        "R2": float(decomp["R2"]),
                        "residual_norm": float(decomp["residual_norm"]),
                        "changed_fraction_full123": float(strict["changed_fraction_full"]),
                        "item1_first_prob": int(ping.target_first["item1"][row_idx]),
                        "item2_first_prob": int(ping.target_first["item2"][row_idx]),
                        "item3_first_prob": int(ping.target_first["item3"][row_idx]),
                        "silent_prob": int(ping.silent_mask[row_idx]),
                        "P_1only_given_full": float(strict["P_1_only_given_full"]),
                        "P_2only_given_full": float(strict["P_2_only_given_full"]),
                        "P_3only_given_full": float(strict["P_3_only_given_full"]),
                        "P_supported_given_full": float(strict["P_supported_given_full"]),
                        "P_novel_given_full": float(strict["P_novel_given_full"]),
                    }
                )
    return summarize_metric_table(
        rows,
        group_cols=["experiment", "layer"],
        value_cols=[
            "Sim_F1",
            "Sim_F2",
            "Sim_F3",
            "alpha1",
            "alpha2",
            "alpha3",
            "R2",
            "residual_norm",
            "changed_fraction_full123",
            "item1_first_prob",
            "item2_first_prob",
            "item3_first_prob",
            "silent_prob",
            "P_1only_given_full",
            "P_2only_given_full",
            "P_3only_given_full",
            "P_supported_given_full",
            "P_novel_given_full",
        ],
    )


def layer_summary_records(df: pd.DataFrame) -> pd.DataFrame:
    if "record_type" not in df.columns:
        return df.copy()
    out = df[df["record_type"] == "layer_summary"].copy()
    return out if not out.empty else df.copy()


def save_sequence_reversal_figure(layout, df_sequence: pd.DataFrame) -> dict[str, str]:
    apply_publication_style()
    df_summary = layer_summary_records(df_sequence)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    layer_order = list(LAYER_KEYS)
    x = np.arange(len(layer_order))
    ax_a, ax_b = axes
    style_map = {
        ORDER_A: ("sample first", SAMPLE_COLOR, "-"),
        ORDER_B: ("distractor first", SHUFFLE_COLOR, "--"),
    }
    for order_condition, (label, color, linestyle) in style_map.items():
        subset = df_summary[df_summary["order_condition"] == order_condition]
        ax_a.plot(
            x,
            [
                float(subset.loc[subset["layer"] == layer_key, "DI_first_second"].mean())
                if np.any(subset["layer"] == layer_key)
                else np.nan
                for layer_key in layer_order
            ],
            marker="o",
            linewidth=2.0,
            linestyle=linestyle,
            color=color,
            label=label,
        )
        ax_b.plot(
            x,
            [
                float(subset.loc[subset["layer"] == layer_key, "first_item_first_prob"].mean())
                if np.any(subset["layer"] == layer_key)
                else np.nan
                for layer_key in layer_order
            ],
            marker="o",
            linewidth=2.0,
            linestyle=linestyle,
            color=color,
            label=label,
        )
    ax_a.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_a.set_xticks(x, layer_order)
    ax_a.set_ylabel("DI(first, second)")
    ax_a.set_title("A. First-item dominance by order")
    ax_a.legend(frameon=False, fontsize=9)
    ax_b.set_xticks(x, layer_order)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel("First-item first-fire probability")
    ax_b.set_title("B. Neutral ping first-item preference")
    ax_b.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    paths = save_figure_all_formats(fig, layout.figure_base("sequence_reversal_overview"))
    plt.close(fig)
    return paths


def save_delay_scan_figure(layout, df_delay: pd.DataFrame) -> dict[str, str]:
    apply_publication_style()
    df_summary = layer_summary_records(df_delay)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax_a, ax_b = axes
    layer_color = {"layer1": SAMPLE_COLOR, "layer2": DYNAMIC_COLOR, "layer3": SHUFFLE_COLOR}
    scan_style = {"delay1_scan": "-", "delay2_scan": "--"}
    for scan_type, linestyle in scan_style.items():
        for layer_key, color in layer_color.items():
            subset = df_summary[(df_summary["scan_type"] == scan_type) & (df_summary["layer"] == layer_key)].copy()
            subset = subset.sort_values("delay_ms", kind="stable")
            if subset.empty:
                continue
            label = f"{scan_type.replace('_', ' ')} {layer_key}"
            ax_a.plot(
                subset["delay_ms"].to_numpy(dtype=np.float64),
                subset["DI"].to_numpy(dtype=np.float64),
                marker="o",
                linewidth=1.8,
                linestyle=linestyle,
                color=color,
                label=label,
            )
            ax_b.plot(
                subset["delay_ms"].to_numpy(dtype=np.float64),
                subset["item1_first_prob"].to_numpy(dtype=np.float64),
                marker="o",
                linewidth=1.8,
                linestyle=linestyle,
                color=color,
                label=label,
            )
    ax_a.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_a.set_xlabel("Delay (ms)")
    ax_a.set_ylabel("DI")
    ax_a.set_title("A. Delay vs DI")
    ax_b.set_xlabel("Delay (ms)")
    ax_b.set_ylabel("Item1 first-fire probability")
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_title("B. Delay vs item1-first ping")
    ax_b.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    paths = save_figure_all_formats(fig, layout.figure_base("delay_scan_overview"))
    plt.close(fig)
    return paths


def save_strict_full_conditioned_figure(layout, df_strict: pd.DataFrame) -> dict[str, str]:
    apply_publication_style()
    df_summary = layer_summary_records(df_strict)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax_a, ax_b, ax_c = axes
    order_conditions = [ORDER_A, ORDER_B]
    labels = []
    x_positions = np.arange(len(order_conditions) * len(LAYER_KEYS))
    changed_vals = []
    p_item1_vals = []
    p_item2_vals = []
    p_shared_vals = []
    p_novel_vals = []
    abs_item1_vals = []
    abs_item2_vals = []
    abs_shared_vals = []
    abs_novel_vals = []
    for order_condition in order_conditions:
        subset = df_summary[df_summary["order_condition"] == order_condition]
        short = "A" if order_condition == ORDER_A else "B"
        for layer_key in LAYER_KEYS:
            row = subset[subset["layer"] == layer_key]
            labels.append(f"{short}-{layer_key[-1]}")
            changed_vals.append(float(row["changed_fraction_full"].mean()) if not row.empty else np.nan)
            p_item1_vals.append(float(row["P_item1_only_given_full"].mean()) if not row.empty else np.nan)
            p_item2_vals.append(float(row["P_item2_only_given_full"].mean()) if not row.empty else np.nan)
            p_shared_vals.append(float(row["P_shared_given_full"].mean()) if not row.empty else np.nan)
            p_novel_vals.append(float(row["P_novel_given_full"].mean()) if not row.empty else np.nan)
            abs_item1_vals.append(float(row["Abs_item1_only_in_full"].mean()) if not row.empty else np.nan)
            abs_item2_vals.append(float(row["Abs_item2_only_in_full"].mean()) if not row.empty else np.nan)
            abs_shared_vals.append(float(row["Abs_shared_in_full"].mean()) if not row.empty else np.nan)
            abs_novel_vals.append(float(row["Abs_novel_in_full"].mean()) if not row.empty else np.nan)
    ax_a.bar(x_positions, changed_vals, color=DYNAMIC_COLOR, width=0.7)
    ax_a.set_xticks(x_positions, labels)
    ax_a.set_ylabel("Fraction")
    ax_a.set_title("A. changed_fraction_full")
    bottom = np.zeros(len(x_positions), dtype=np.float64)
    for values, color, label in (
        (p_item1_vals, SAMPLE_COLOR, "item1-only"),
        (p_item2_vals, SHUFFLE_COLOR, "item2-only"),
        (p_shared_vals, DYNAMIC_COLOR, "shared"),
        (p_novel_vals, NOISE_COLOR, "novel"),
    ):
        vals = np.asarray(values, dtype=np.float64)
        ax_b.bar(x_positions, vals, bottom=bottom, color=color, width=0.7, label=label)
        bottom = np.where(np.isnan(vals), bottom, np.nan_to_num(bottom) + np.nan_to_num(vals))
    ax_b.set_xticks(x_positions, labels)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel("Within full-changed")
    ax_b.set_title("B. Composition within full-changed")
    ax_b.legend(frameon=False, fontsize=8)
    bottom = np.zeros(len(x_positions), dtype=np.float64)
    for values, color, label in (
        (abs_item1_vals, SAMPLE_COLOR, "Abs item1-only"),
        (abs_item2_vals, SHUFFLE_COLOR, "Abs item2-only"),
        (abs_shared_vals, DYNAMIC_COLOR, "Abs shared"),
        (abs_novel_vals, NOISE_COLOR, "Abs novel"),
    ):
        vals = np.asarray(values, dtype=np.float64)
        ax_c.bar(x_positions, vals, bottom=bottom, color=color, width=0.7, label=label)
        bottom = np.where(np.isnan(vals), bottom, np.nan_to_num(bottom) + np.nan_to_num(vals))
    ax_c.set_xticks(x_positions, labels)
    ax_c.set_ylabel("Absolute fraction of synapses")
    ax_c.set_title("C. Absolute source mass")
    ax_c.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths = save_figure_all_formats(fig, layout.figure_base("strict_full_conditioned_changed"))
    plt.close(fig)
    return paths


def save_three_item_figure(layout, df_three: pd.DataFrame) -> dict[str, str]:
    apply_publication_style()
    df_summary = layer_summary_records(df_three)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax_a, ax_b = axes
    layer_order = list(LAYER_KEYS)
    x = np.arange(len(layer_order))
    metric_specs = (
        ("Sim_F1", SAMPLE_COLOR, "item1 similarity"),
        ("Sim_F2", DYNAMIC_COLOR, "item2 similarity"),
        ("Sim_F3", SHUFFLE_COLOR, "item3 similarity"),
    )
    for metric, color, label in metric_specs:
        ax_a.plot(
            x,
            [
                float(df_summary.loc[df_summary["layer"] == layer_key, metric].mean())
                if np.any(df_summary["layer"] == layer_key)
                else np.nan
                for layer_key in layer_order
            ],
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
    ping_specs = (
        ("item1_first_prob", SAMPLE_COLOR, "item1 first-fire"),
        ("item2_first_prob", DYNAMIC_COLOR, "item2 first-fire"),
        ("item3_first_prob", SHUFFLE_COLOR, "item3 first-fire"),
    )
    for metric, color, label in ping_specs:
        ax_b.plot(
            x,
            [
                float(df_summary.loc[df_summary["layer"] == layer_key, metric].mean())
                if np.any(df_summary["layer"] == layer_key)
                else np.nan
                for layer_key in layer_order
            ],
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
    ax_a.set_xticks(x, layer_order)
    ax_a.set_ylabel("Centered cosine similarity")
    ax_a.set_title("A. Layer-wise three-item similarity")
    ax_a.legend(frameon=False, fontsize=8, ncol=2)
    ax_b.set_xticks(x, layer_order)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel("First-fire probability")
    ax_b.set_title("B. Layer-wise ping first-fire probabilities")
    ax_b.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    paths = save_figure_all_formats(fig, layout.figure_base("three_item_overview"))
    plt.close(fig)
    return paths


def save_validation_suite_figures(
    layout,
    *,
    df_sequence: pd.DataFrame,
    df_delay: pd.DataFrame,
    df_strict: pd.DataFrame,
    df_three: pd.DataFrame,
) -> dict[str, object]:
    """Save all suite figures using a consistent publication style."""

    return {
        "sequence_reversal": save_sequence_reversal_figure(layout, df_sequence),
        "delay_scan": save_delay_scan_figure(layout, df_delay),
        "strict_full_conditioned_changed": save_strict_full_conditioned_figure(layout, df_strict),
        "three_item": save_three_item_figure(layout, df_three),
    }


def build_nested_summary(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> dict[str, object]:
    summary_df = layer_summary_records(df)
    if summary_df.empty:
        return {}
    result: dict[str, object] = {}
    for record in summary_df[list(group_cols) + list(value_cols)].to_dict(orient="records"):
        cursor = result
        for group_col in group_cols[:-1]:
            cursor = cursor.setdefault(str(record[group_col]), {})
        leaf_key = str(record[group_cols[-1]])
        cursor[leaf_key] = {str(col): safe_float(record[col]) for col in value_cols}
    return result


def build_summary_payload(
    cfg: ValidationSuiteConfig,
    *,
    device_requested: str,
    device_resolved: str,
    triplet_count: int,
    df_sequence: pd.DataFrame,
    df_delay: pd.DataFrame,
    df_strict: pd.DataFrame,
    df_three: pd.DataFrame,
    exported_files: Mapping[str, object],
) -> dict[str, object]:
    df_seq_summary = layer_summary_records(df_sequence)
    df_delay_summary = layer_summary_records(df_delay)
    sequence_reversal_summary = {
        "by_order_condition": build_nested_summary(
            df_sequence,
            group_cols=["order_condition", "layer"],
            value_cols=[
                "Sim_F_first",
                "Sim_F_second",
                "DI_first_second",
                "alpha_first",
                "beta_second",
                "first_item_first_prob",
                "second_item_first_prob",
                "sample_first_prob",
                "distractor_first_prob",
            ],
        ),
        "dominance_follows_first": {
            order_condition: safe_float(
                df_seq_summary.loc[df_seq_summary["order_condition"] == order_condition, "DI_first_second"].mean()
            )
            for order_condition in (ORDER_A, ORDER_B)
        },
        "high_layer_minus_low_layer_first_bias": {
            order_condition: safe_float(
                float(
                    df_seq_summary.loc[
                        (df_seq_summary["order_condition"] == order_condition) & (df_seq_summary["layer"] == "layer3"),
                        "DI_first_second",
                    ].mean()
                    - df_seq_summary.loc[
                        (df_seq_summary["order_condition"] == order_condition) & (df_seq_summary["layer"] == "layer1"),
                        "DI_first_second",
                    ].mean()
                )
            )
            for order_condition in (ORDER_A, ORDER_B)
        },
    }
    delay_scan_summary = {
        "by_scan_type": build_nested_summary(
            df_delay,
            group_cols=["scan_type", "delay_ms", "layer"],
            value_cols=[
                "DI",
                "alpha",
                "beta",
                "changed_fraction_full",
                "P_item1_only_given_full",
                "P_item2_only_given_full",
                "P_shared_given_full",
                "P_novel_given_full",
                "item1_first_prob",
                "item2_first_prob",
            ],
        ),
        "delay_range_ms": [float(x) for x in sorted(pd.unique(df_delay_summary["delay_ms"]).tolist())] if not df_delay_summary.empty else [],
    }
    strict_summary = {
        "by_order_condition": build_nested_summary(
            df_strict,
            group_cols=["order_condition", "layer"],
            value_cols=[
                "changed_fraction_full",
                "P_item1_only_given_full",
                "P_item2_only_given_full",
                "P_shared_given_full",
                "P_novel_given_full",
                "Abs_item1_only_in_full",
                "Abs_item2_only_in_full",
                "Abs_shared_in_full",
                "Abs_novel_in_full",
                "supported_fraction_given_full",
                "source_asymmetry_given_full",
            ],
        )
    }
    three_item_summary = {
        "by_layer": build_nested_summary(
            df_three,
            group_cols=["layer"],
            value_cols=[
                "Sim_F1",
                "Sim_F2",
                "Sim_F3",
                "alpha1",
                "alpha2",
                "alpha3",
                "changed_fraction_full123",
                "item1_first_prob",
                "item2_first_prob",
                "item3_first_prob",
                "P_1only_given_full",
                "P_2only_given_full",
                "P_3only_given_full",
                "P_supported_given_full",
                "P_novel_given_full",
            ],
        )
    }
    return {
        "config": json_safe(asdict(cfg)),
        "device_requested": str(device_requested),
        "device_resolved": str(device_resolved),
        "triplet_count": int(triplet_count),
        "sequence_reversal_summary": sequence_reversal_summary,
        "delay_scan_summary": delay_scan_summary,
        "strict_full_conditioned_changed_summary": strict_summary,
        "three_item_summary": three_item_summary,
        "exported_files": json_safe(exported_files),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = normalize_config(args)
    layout = prepare_result_layout(cfg.output_dir)
    log_lines: list[str] = []

    device, device_message = resolve_device_with_fallback(cfg.device)
    log_and_print(log_lines, device_message)
    save_run_config(json_safe(asdict(cfg)), layout.root)

    seed_everything(int(cfg.seed))
    log_and_print(log_lines, f"[Setup] seed={cfg.seed}")
    log_and_print(log_lines, f"[Setup] loading dataset split={cfg.split} from {cfg.dataset_root}")
    dataset = load_mnist_dataset(cfg.dataset_root, cfg.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    class_index = build_class_index(dataset, num_classes=10)
    df_triplets = build_validation_triplets(images, labels, flat_normalized, class_index, cfg)
    triplets_csv = save_tidy_csv(df_triplets, layout.data_file("triplets.csv"), sort_by=["triplet_id"])
    log_and_print(log_lines, f"[Data] triplets={len(df_triplets)} saved={triplets_csv}")

    net, encoder = load_model_and_encoder(
        model_path=cfg.model_path,
        device=device,
        dt=cfg.dt,
        max_duration_ms=cfg.max_duration_ms,
    )
    log_and_print(log_lines, f"[Model] loaded {cfg.model_path}")

    df_sequence, df_strict, example_dump = run_sequence_reversal_experiment(
        net,
        cfg,
        df_triplets,
        images,
        encoder,
        device,
        log_lines,
    )
    df_delay = run_delay_scan_experiment(
        net,
        cfg,
        df_triplets,
        images,
        encoder,
        device,
        log_lines,
    )
    df_three = run_three_item_experiment(
        net,
        cfg,
        df_triplets,
        images,
        encoder,
        device,
        log_lines,
    )

    sequence_csv = save_tidy_csv(df_sequence, layout.data_file("sequence_reversal_metrics.csv"))
    delay_csv = save_tidy_csv(df_delay, layout.data_file("delay_scan_metrics.csv"))
    strict_csv = save_tidy_csv(df_strict, layout.data_file("strict_full_conditioned_changed_metrics.csv"))
    three_csv = save_tidy_csv(df_three, layout.data_file("three_item_metrics.csv"))
    example_npz_path = layout.data_file("example_stsp_states.npz")
    np.savez_compressed(example_npz_path, **example_dump)

    figure_paths: dict[str, object] = {}
    if not cfg.skip_figures:
        figure_paths = save_validation_suite_figures(
            layout,
            df_sequence=df_sequence,
            df_delay=df_delay,
            df_strict=df_strict,
            df_three=df_three,
        )

    log_path = layout.root_file("log.txt")
    summary_path = layout.root_file("summary.json")
    exported_files = {
        "triplets_csv": str(triplets_csv),
        "sequence_reversal_metrics_csv": str(sequence_csv),
        "delay_scan_metrics_csv": str(delay_csv),
        "strict_full_conditioned_changed_metrics_csv": str(strict_csv),
        "three_item_metrics_csv": str(three_csv),
        "example_stsp_states_npz": str(example_npz_path),
        "summary_json": str(summary_path),
        "log_txt": str(log_path),
        "figures": figure_paths,
    }
    summary_payload = build_summary_payload(
        cfg,
        device_requested=cfg.device,
        device_resolved=str(device),
        triplet_count=len(df_triplets),
        df_sequence=df_sequence,
        df_delay=df_delay,
        df_strict=df_strict,
        df_three=df_three,
        exported_files=exported_files,
    )
    log_and_print(log_lines, f"[Done] summary={summary_path}")
    save_summary_json(json_safe(summary_payload), layout.root)
    save_log_lines(log_lines, layout.root, filename="log.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
