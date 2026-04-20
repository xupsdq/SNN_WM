from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import DEFAULT_PATH_CONFIG
from src.config.units import ms
from src.experiments.common.dataset import build_class_index, build_dataset_arrays, encode_images
from src.experiments.common.distractor_triplets import build_triplet_specs as shared_build_triplet_specs
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import (
    prepare_result_layout,
    save_log_lines,
    save_run_config,
    save_summary_json,
)
from src.experiments.common.seed import mix_seed
from src.plotting.common.io import (
    COLOR_DYNAMIC,
    apply_publication_style,
    save_figure_all_formats,
    save_tidy_csv,
)

DT = 1.0 * ms
EPS = 1e-12
TARGET_LAYER = "L3"
CONDITION_NAMES = ("mixed", "sample_only", "distractor_only")


@dataclass(frozen=True)
class TimingConfig:
    dt: float
    sample_ms: float
    delay1_ms: float
    distractor_ms: float
    delay2_ms: float

    def to_steps(self, duration_ms: float) -> int:
        steps = int(round((float(duration_ms) * ms) / float(self.dt)))
        return max(0, steps)

    @property
    def sample_steps(self) -> int:
        return self.to_steps(self.sample_ms)

    @property
    def delay1_steps(self) -> int:
        return self.to_steps(self.delay1_ms)

    @property
    def distractor_steps(self) -> int:
        return self.to_steps(self.distractor_ms)

    @property
    def delay2_steps(self) -> int:
        return self.to_steps(self.delay2_ms)


@dataclass(frozen=True)
class ExperimentConfig:
    model_path: Path
    dataset_root: Path
    split: str
    device: str
    seed: int
    output_dir: Path
    batch_size: int
    max_probes: int
    samples_per_probe: int
    max_triplets: int
    num_sim_bins: int
    skip_figures: bool
    smoke: bool
    timings: TimingConfig

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["model_path"] = str(self.model_path.resolve())
        data["dataset_root"] = str(self.dataset_root.resolve())
        data["output_dir"] = str(self.output_dir.resolve())
        return data


@dataclass
class RunLogger:
    lines: list[str]

    def log(self, message: str) -> None:
        text = str(message)
        print(text, flush=True)
        self.lines.append(text)


@dataclass(frozen=True)
class PreparedBatch:
    batch_df: pd.DataFrame
    sample_spikes: torch.Tensor
    distractor_spikes: torch.Tensor
    zero_sample_spikes: torch.Tensor
    zero_distractor_spikes: torch.Tensor


@dataclass(frozen=True)
class CenteredBank:
    centered: np.ndarray
    norms: np.ndarray
    unit: np.ndarray


def _validate_positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    numeric = float(value)
    if allow_zero:
        if numeric < 0.0:
            raise ValueError(f"{name} must be >= 0, got {numeric}")
    elif numeric <= 0.0:
        raise ValueError(f"{name} must be > 0, got {numeric}")


def _validate_int_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _sampling_scale_name(config: ExperimentConfig) -> str:
    if config.smoke:
        return "smoke"
    if config.max_triplets >= 80:
        return "full"
    if config.max_triplets >= 40:
        return "draft"
    return "custom"


def _apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not bool(args.smoke):
        return args
    args.batch_size = min(int(args.batch_size), 2)
    args.max_probes = min(int(args.max_probes), 2)
    args.samples_per_probe = min(int(args.samples_per_probe), 1)
    args.max_triplets = min(int(args.max_triplets), 4)
    args.num_sim_bins = min(int(args.num_sim_bins), 2)
    return args


def _resolve_runtime_device(device_arg: str) -> tuple[torch.device, bool]:
    raw = str(device_arg).strip().lower()
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), False
    if raw.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu"), True
    return torch.device(raw), False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 2-only chunk fused-state experiment: capture pre-probe STSP state after sample-delay1-distractor-delay2."
    )
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PATH_CONFIG.results_root / "chunk_step2_fused_state_experiment"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay1-ms", type=float, default=400.0)
    parser.add_argument("--distractor-ms", type=float, default=200.0)
    parser.add_argument("--delay2-ms", type=float, default=400.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-probes", type=int, default=10)
    parser.add_argument("--samples-per-probe", type=int, default=100)
    parser.add_argument("--max-triplets", type=int, default=1000)
    parser.add_argument("--num-sim-bins", type=int, default=4)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    args = _apply_smoke_overrides(args)
    _validate_positive("--sample-ms", float(args.sample_ms))
    _validate_positive("--distractor-ms", float(args.distractor_ms))
    _validate_positive("--delay1-ms", float(args.delay1_ms), allow_zero=True)
    _validate_positive("--delay2-ms", float(args.delay2_ms), allow_zero=True)
    _validate_int_positive("--batch-size", int(args.batch_size))
    _validate_int_positive("--max-probes", int(args.max_probes))
    _validate_int_positive("--samples-per-probe", int(args.samples_per_probe))
    _validate_int_positive("--max-triplets", int(args.max_triplets))
    _validate_int_positive("--num-sim-bins", int(args.num_sim_bins))
    return ExperimentConfig(
        model_path=Path(args.model_path).resolve(),
        dataset_root=Path(args.dataset_root).resolve(),
        split=str(args.split).strip().lower(),
        device=str(args.device).strip(),
        seed=int(args.seed),
        output_dir=Path(args.output_dir).resolve(),
        batch_size=int(args.batch_size),
        max_probes=int(args.max_probes),
        samples_per_probe=int(args.samples_per_probe),
        max_triplets=int(args.max_triplets),
        num_sim_bins=int(args.num_sim_bins),
        skip_figures=bool(args.skip_figures),
        smoke=bool(args.smoke),
        timings=TimingConfig(
            dt=DT,
            sample_ms=float(args.sample_ms),
            delay1_ms=float(args.delay1_ms),
            distractor_ms=float(args.distractor_ms),
            delay2_ms=float(args.delay2_ms),
        ),
    )


def log_config_summary(
    logger: RunLogger,
    config: ExperimentConfig,
    runtime_device: torch.device,
    cuda_fallback: bool,
) -> None:
    logger.log(
        "[Config] "
        f"model={config.model_path} | dataset={config.dataset_root} | split={config.split} | "
        f"device_request={config.device} | runtime_device={runtime_device}"
    )
    if cuda_fallback:
        logger.log("[Warn] CUDA was requested but is not available. Falling back to CPU for this run.")
    logger.log(
        "[Config] timings_ms="
        f"(sample={config.timings.sample_ms:.0f}, delay1={config.timings.delay1_ms:.0f}, "
        f"distractor={config.timings.distractor_ms:.0f}, delay2={config.timings.delay2_ms:.0f}) "
        f"steps=(sample={config.timings.sample_steps}, delay1={config.timings.delay1_steps}, "
        f"distractor={config.timings.distractor_steps}, delay2={config.timings.delay2_steps})"
    )
    logger.log(
        "[Config] triplets="
        f"(scale={_sampling_scale_name(config)}, batch_size={config.batch_size}, max_probes={config.max_probes}, "
        f"samples_per_probe={config.samples_per_probe}, max_triplets={config.max_triplets}, "
        f"num_sim_bins={config.num_sim_bins}, smoke={config.smoke})"
    )


def build_triplet_specs(
    images: torch.Tensor,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    class_index: Mapping[int, Iterable[int]],
    *,
    config: ExperimentConfig,
) -> pd.DataFrame:
    triplets = shared_build_triplet_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=config.max_probes,
        samples_per_probe=config.samples_per_probe,
        num_bins=config.num_sim_bins,
        max_triplets=config.max_triplets,
        seed=config.seed,
    ).copy()
    triplets["split"] = config.split
    triplets["selection_scale"] = _sampling_scale_name(config)
    triplets["selection_seed"] = int(config.seed)
    triplets["sample_bin"] = triplets["sp_bin"].astype(str)
    triplets["distractor_bin"] = triplets["dp_bin"].astype(str)
    triplets["sample_to_probe_similarity"] = triplets["sp_similarity"].astype(np.float64)
    triplets["distractor_to_probe_similarity"] = triplets["dp_similarity"].astype(np.float64)
    triplets["sample_to_distractor_similarity"] = triplets["sd_similarity"].astype(np.float64)
    triplets["sample_rank_within_probe"] = triplets.groupby("probe_id").cumcount().astype(np.int64)
    return triplets


def _encode_image_bank(
    image_ids: list[int],
    *,
    images: torch.Tensor,
    encoder,
    steps: int,
    device: torch.device,
    cache: dict[tuple[int, int], torch.Tensor],
) -> tuple[list[int], torch.Tensor]:
    unique_ids = list(dict.fromkeys(int(idx) for idx in image_ids))
    missing = [image_id for image_id in unique_ids if (int(image_id), int(steps)) not in cache]
    if missing:
        batch_images = images[missing].to(device=device, dtype=torch.float32)
        encoded = encode_images(encoder, batch_images, steps=int(steps)).detach().cpu()
        for row_idx, image_id in enumerate(missing):
            cache[(int(image_id), int(steps))] = encoded[row_idx]
    stacked = torch.stack([cache[(int(image_id), int(steps))] for image_id in unique_ids], dim=0)
    return unique_ids, stacked.to(device=device, non_blocking=True)


def prepare_triplet_batches(
    df_triplets: pd.DataFrame,
    images: torch.Tensor,
    encoder,
    *,
    timing: TimingConfig,
    batch_size: int,
    device: torch.device,
) -> Iterator[PreparedBatch]:
    sample_cache: dict[tuple[int, int], torch.Tensor] = {}
    distractor_cache: dict[tuple[int, int], torch.Tensor] = {}
    for start in range(0, len(df_triplets), int(batch_size)):
        batch_df = df_triplets.iloc[start : start + int(batch_size)].copy().reset_index(drop=True)
        sample_ids = batch_df["sample_id"].astype(int).tolist()
        distractor_ids = batch_df["distractor_id"].astype(int).tolist()
        unique_sample_ids, sample_unique = _encode_image_bank(
            sample_ids,
            images=images,
            encoder=encoder,
            steps=timing.sample_steps,
            device=device,
            cache=sample_cache,
        )
        unique_distractor_ids, distractor_unique = _encode_image_bank(
            distractor_ids,
            images=images,
            encoder=encoder,
            steps=timing.distractor_steps,
            device=device,
            cache=distractor_cache,
        )
        sample_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_sample_ids)}
        distractor_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_distractor_ids)}
        sample_select = torch.tensor([sample_lookup[int(idx)] for idx in sample_ids], dtype=torch.long, device=device)
        distractor_select = torch.tensor(
            [distractor_lookup[int(idx)] for idx in distractor_ids],
            dtype=torch.long,
            device=device,
        )
        sample_spikes = sample_unique.index_select(0, sample_select)
        distractor_spikes = distractor_unique.index_select(0, distractor_select)
        zero_sample_spikes = torch.zeros_like(sample_spikes)
        zero_distractor_spikes = torch.zeros_like(distractor_spikes)
        yield PreparedBatch(
            batch_df=batch_df,
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            zero_sample_spikes=zero_sample_spikes,
            zero_distractor_spikes=zero_distractor_spikes,
        )


def _initialize_session_state(net, batch_shape: tuple[int, int, int, int]) -> None:
    batch_size, in_channels, height, width = batch_shape
    net.layer1.reset_state((batch_size, in_channels, height, width))
    h1 = (height + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    w1 = (width + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    h1_p, w1_p = h1 // 2, w1 // 2
    net.layer2.reset_state((batch_size, net.layer1.out_channels, h1_p, w1_p))
    h2 = (h1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    w2 = (w1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    h2_p, w2_p = h2 // 2, w2 // 2
    net.layer3.reset_state((batch_size, net.layer2.out_channels, h2_p, w2_p))


def run_chunk_state_capture(
    net,
    sample_spikes: torch.Tensor,
    distractor_spikes: torch.Tensor,
    *,
    timing: TimingConfig,
) -> torch.Tensor:
    if sample_spikes.ndim != 5 or distractor_spikes.ndim != 5:
        raise ValueError("sample_spikes and distractor_spikes must have shape [B, T, C, H, W].")
    if sample_spikes.shape[0] != distractor_spikes.shape[0]:
        raise ValueError("sample and distractor batches must have the same batch dimension.")
    if sample_spikes.shape[2:] != distractor_spikes.shape[2:]:
        raise ValueError("sample and distractor spikes must share channel/height/width.")
    batch_size, _, in_channels, height, width = sample_spikes.shape
    _initialize_session_state(net, (batch_size, in_channels, height, width))
    zero_input = torch.zeros((batch_size, in_channels, height, width), device=sample_spikes.device, dtype=sample_spikes.dtype)

    def step_network(input_t: torch.Tensor, time_index: int) -> None:
        s1, _ = net.layer1.forward_step(input_t, time_index, training=False, stsp_mode="dynamic")
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, time_index, training=False, stsp_mode="dynamic")
        s2_p = net.pool2(s2.float())
        net.layer3.forward_step(s2_p, time_index, training=False, monitor=False, stsp_mode="dynamic")

    with torch.no_grad():
        current_time = 0
        for t in range(sample_spikes.shape[1]):
            step_network(sample_spikes[:, t, ...], current_time)
            current_time += 1
        for _ in range(timing.delay1_steps):
            step_network(zero_input, current_time)
            current_time += 1
        for t in range(distractor_spikes.shape[1]):
            step_network(distractor_spikes[:, t, ...], current_time)
            current_time += 1
        for _ in range(timing.delay2_steps):
            step_network(zero_input, current_time)
            current_time += 1
        l3_ux = (net.layer3.u_pre * net.layer3.x_pre).detach().to(torch.float32).cpu()
        return l3_ux


def _flatten_tensor_batch(x: torch.Tensor) -> np.ndarray:
    return x.reshape(x.shape[0], -1).numpy().astype(np.float32, copy=False)


def capture_triplet_state_banks(
    df_triplets: pd.DataFrame,
    images: torch.Tensor,
    net,
    encoder,
    *,
    config: ExperimentConfig,
    device: torch.device,
    logger: RunLogger,
) -> dict[str, np.ndarray]:
    accum: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITION_NAMES}
    num_batches = int(math.ceil(len(df_triplets) / max(1, int(config.batch_size))))
    logger.log(f"[Capture] Running {len(df_triplets)} triplets across {num_batches} batch(es).")
    for batch_idx, prepared in enumerate(
        prepare_triplet_batches(
            df_triplets,
            images,
            encoder,
            timing=config.timings,
            batch_size=config.batch_size,
            device=device,
        ),
        start=1,
    ):
        batch_size = len(prepared.batch_df)
        sample_concat = torch.cat(
            [prepared.sample_spikes, prepared.sample_spikes, prepared.zero_sample_spikes],
            dim=0,
        )
        distractor_concat = torch.cat(
            [prepared.distractor_spikes, prepared.zero_distractor_spikes, prepared.distractor_spikes],
            dim=0,
        )
        l3_ux = run_chunk_state_capture(
            net,
            sample_concat,
            distractor_concat,
            timing=config.timings,
        )
        split_lookup = {
            "mixed": slice(0, batch_size),
            "sample_only": slice(batch_size, 2 * batch_size),
            "distractor_only": slice(2 * batch_size, 3 * batch_size),
        }
        flat_ux = _flatten_tensor_batch(l3_ux)
        for condition, row_slice in split_lookup.items():
            accum[condition].append(flat_ux[row_slice])
        logger.log(f"[Capture] batch={batch_idx}/{num_batches} | triplets={batch_size} | done.")

    state_bank: dict[str, np.ndarray] = {}
    for condition in CONDITION_NAMES:
        state_bank[condition] = np.concatenate(accum[condition], axis=0).astype(np.float32, copy=False)
    return state_bank


def _build_state_vector(condition_bank: np.ndarray) -> np.ndarray:
    return np.asarray(condition_bank, dtype=np.float32)


def _build_centered_bank(x: np.ndarray) -> CenteredBank:
    arr = np.asarray(x, dtype=np.float32)
    centered = arr - arr.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    safe_norms = np.maximum(norms, EPS)
    unit = centered / safe_norms[:, None]
    return CenteredBank(centered=centered, norms=safe_norms, unit=unit)


def _rowwise_centered_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_bank = _build_centered_bank(a)
    b_bank = _build_centered_bank(b)
    numer = np.sum(a_bank.centered * b_bank.centered, axis=1)
    denom = np.maximum(a_bank.norms * b_bank.norms, EPS)
    return (numer / denom).astype(np.float32, copy=False)


def compute_preprobe_fusion_metrics(
    df_triplets: pd.DataFrame,
    state_bank: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    mixed_vec = _build_state_vector(state_bank["mixed"])
    sample_vec = _build_state_vector(state_bank["sample_only"])
    distractor_vec = _build_state_vector(state_bank["distractor_only"])
    sim_to_sample = _rowwise_centered_cosine(mixed_vec, sample_vec)
    sim_to_distractor = _rowwise_centered_cosine(mixed_vec, distractor_vec)
    dual_score = 0.5 * (sim_to_sample + sim_to_distractor)
    imbalance = np.abs(sim_to_sample - sim_to_distractor)
    for row_idx, triplet_id in enumerate(df_triplets["triplet_id"].astype(int).tolist()):
        records.append(
            {
                "triplet_id": int(triplet_id),
                "layer": TARGET_LAYER,
                "sim_to_sample": float(sim_to_sample[row_idx]),
                "sim_to_distractor": float(sim_to_distractor[row_idx]),
                "fusion_dual_score": float(dual_score[row_idx]),
                "fusion_imbalance": float(imbalance[row_idx]),
            }
        )
    return pd.DataFrame.from_records(records)


def compute_fusion_specificity_metrics(
    df_triplets: pd.DataFrame,
    state_bank: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> pd.DataFrame:
    triplet_ids = df_triplets["triplet_id"].astype(int).to_numpy(dtype=np.int64, copy=False)
    num_triplets = int(len(triplet_ids))
    records: list[dict[str, object]] = []
    mixed_vec = _build_state_vector(state_bank["mixed"])
    sample_vec = _build_state_vector(state_bank["sample_only"])
    distractor_vec = _build_state_vector(state_bank["distractor_only"])
    mixed_bank = _build_centered_bank(mixed_vec)
    sample_bank = _build_centered_bank(sample_vec)
    distractor_bank = _build_centered_bank(distractor_vec)

    ms_term = mixed_bank.unit @ sample_bank.centered.T
    md_term = mixed_bank.unit @ distractor_bank.centered.T
    sd_term = sample_bank.centered @ distractor_bank.centered.T
    sample_norm_sq = np.sum(sample_bank.centered * sample_bank.centered, axis=1, dtype=np.float32)
    distractor_norm_sq = np.sum(distractor_bank.centered * distractor_bank.centered, axis=1, dtype=np.float32)
    pair_norms = np.sqrt(
        np.maximum(
            sample_norm_sq[:, None] + distractor_norm_sq[None, :] + 2.0 * sd_term,
            EPS,
        )
    ).astype(np.float32, copy=False)
    rng = np.random.default_rng(mix_seed(seed, 503))

    for row_idx, triplet_id in enumerate(triplet_ids.tolist()):
        score_map = (ms_term[row_idx][:, None] + md_term[row_idx][None, :]) / pair_norms
        flat_scores = score_map.reshape(-1)
        true_linear_idx = row_idx * num_triplets + row_idx
        true_score = float(score_map[row_idx, row_idx])
        other_scores = np.delete(flat_scores, true_linear_idx)
        if other_scores.size == 0:
            rank = 1
            percentile = 100.0
            z_score = 0.0
            top1 = 1
            shuffled_score = true_score
            mean_other = true_score
        else:
            rank = 1 + int(np.sum(other_scores > (true_score + 1e-7)))
            percentile = 100.0 * float(np.mean(flat_scores <= (true_score + 1e-7)))
            other_mean = float(other_scores.mean())
            other_std = float(other_scores.std(ddof=0))
            z_score = float((true_score - other_mean) / max(other_std, EPS))
            top1 = int(rank == 1)
            shuffled_score = float(other_scores[int(rng.integers(0, other_scores.size))])
            mean_other = other_mean
        records.append(
            {
                "triplet_id": int(triplet_id),
                "layer": TARGET_LAYER,
                "true_pair_score": true_score,
                "true_pair_rank": int(rank),
                "true_pair_percentile": percentile,
                "true_pair_z": z_score,
                "true_pair_top1": int(top1),
                "shuffled_pair_score": float(shuffled_score),
                "mean_other_pair_score": float(mean_other),
            }
        )
    return pd.DataFrame.from_records(records)


def compute_whole_over_part_metrics(
    fusion_metrics: pd.DataFrame,
    specificity_metrics: pd.DataFrame,
) -> pd.DataFrame:
    merged = fusion_metrics.merge(
        specificity_metrics[["triplet_id", "layer", "true_pair_score"]],
        on=["triplet_id", "layer"],
        how="inner",
        validate="one_to_one",
    ).copy()
    merged["sim_to_true_pair"] = merged["true_pair_score"].astype(np.float64)
    merged["sim_to_sample_only"] = merged["sim_to_sample"].astype(np.float64)
    merged["sim_to_distractor_only"] = merged["sim_to_distractor"].astype(np.float64)
    merged["best_constituent_similarity"] = np.maximum(
        merged["sim_to_sample_only"].to_numpy(dtype=np.float64, copy=False),
        merged["sim_to_distractor_only"].to_numpy(dtype=np.float64, copy=False),
    )
    merged["WPRI"] = merged["sim_to_true_pair"] - merged["best_constituent_similarity"]
    return merged[
        [
            "triplet_id",
            "layer",
            "sim_to_true_pair",
            "sim_to_sample_only",
            "sim_to_distractor_only",
            "best_constituent_similarity",
            "WPRI",
        ]
    ].copy()


def _select_example_triplet(
    df_triplets: pd.DataFrame,
    fusion_metrics: pd.DataFrame,
    specificity_metrics: pd.DataFrame,
) -> pd.Series:
    l3_fusion = fusion_metrics[fusion_metrics["layer"] == TARGET_LAYER][
        ["triplet_id", "fusion_dual_score", "fusion_imbalance"]
    ].copy()
    l3_spec = specificity_metrics[specificity_metrics["layer"] == TARGET_LAYER][
        ["triplet_id", "true_pair_percentile", "true_pair_top1"]
    ].copy()
    scored = (
        df_triplets.merge(l3_fusion, on="triplet_id", how="left")
        .merge(l3_spec, on="triplet_id", how="left")
        .sort_values(
            ["true_pair_top1", "true_pair_percentile", "fusion_dual_score", "fusion_imbalance"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return scored.iloc[0]


def _normalize_image(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy().astype(np.float32, copy=False)
    arr = np.squeeze(arr)
    arr = arr - float(arr.min())
    scale = float(arr.max())
    if scale > 0.0:
        arr = arr / scale
    return arr


def _build_overlap_rgb(sample_image: torch.Tensor, distractor_image: torch.Tensor, *, threshold: float = 0.20) -> np.ndarray:
    sample = _normalize_image(sample_image)
    distractor = _normalize_image(distractor_image)
    sample_mask = sample >= float(threshold)
    distractor_mask = distractor >= float(threshold)
    rgb = np.ones(sample.shape + (3,), dtype=np.float32)
    sample_color = np.asarray([0.93, 0.64, 0.15], dtype=np.float32)
    distractor_color = np.asarray([0.00, 0.62, 0.45], dtype=np.float32)
    shared_color = np.asarray([0.20, 0.20, 0.20], dtype=np.float32)
    rgb[sample_mask & (~distractor_mask)] = sample_color
    rgb[(~sample_mask) & distractor_mask] = distractor_color
    rgb[sample_mask & distractor_mask] = shared_color
    intensity = np.maximum(sample, distractor)[..., None]
    return np.clip(0.35 + 0.65 * rgb * np.maximum(intensity, 0.35), 0.0, 1.0)


def _save_single_axis_figure(
    layout,
    stem: str,
    *,
    figsize: tuple[float, float],
    draw_fn: Callable,
) -> dict[str, str]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(fig, ax)
    saved = save_figure_all_formats(fig, layout.figure_base(stem))
    plt.close(fig)
    return saved


def save_triplet_definition_panels(
    triplet_row: pd.Series,
    images: torch.Tensor,
    layout,
) -> dict[str, dict[str, str]]:
    sample_image = images[int(triplet_row["sample_id"])]
    distractor_image = images[int(triplet_row["distractor_id"])]
    probe_image = images[int(triplet_row["probe_id"])]
    overlap_rgb = _build_overlap_rgb(sample_image, distractor_image)
    return {
        "sample_image": _save_single_axis_figure(
            layout,
            "panel_a_sample_image",
            figsize=(3.0, 3.0),
            draw_fn=lambda _fig, ax: (
                ax.imshow(_normalize_image(sample_image), cmap="gray", vmin=0.0, vmax=1.0),
                ax.axis("off"),
            ),
        ),
        "distractor_image": _save_single_axis_figure(
            layout,
            "panel_a_distractor_image",
            figsize=(3.0, 3.0),
            draw_fn=lambda _fig, ax: (
                ax.imshow(_normalize_image(distractor_image), cmap="gray", vmin=0.0, vmax=1.0),
                ax.axis("off"),
            ),
        ),
        "probe_image": _save_single_axis_figure(
            layout,
            "panel_a_probe_image",
            figsize=(3.0, 3.0),
            draw_fn=lambda _fig, ax: (
                ax.imshow(_normalize_image(probe_image), cmap="gray", vmin=0.0, vmax=1.0),
                ax.axis("off"),
            ),
        ),
        "overlap_support": _save_single_axis_figure(
            layout,
            "panel_a_overlap_support",
            figsize=(3.0, 3.0),
            draw_fn=lambda _fig, ax: (
                ax.imshow(overlap_rgb),
                ax.axis("off"),
            ),
        ),
    }


def save_fusion_form_panel(
    fusion_metrics: pd.DataFrame,
    layout,
) -> dict[str, str]:
    apply_publication_style()
    l3 = fusion_metrics[fusion_metrics["layer"] == TARGET_LAYER].copy()
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(
        l3["sim_to_sample"],
        l3["sim_to_distractor"],
        s=48,
        alpha=0.82,
        color=COLOR_DYNAMIC,
        edgecolors="white",
        linewidths=0.6,
    )
    min_val = float(min(l3["sim_to_sample"].min(), l3["sim_to_distractor"].min(), -1.0))
    max_val = float(max(l3["sim_to_sample"].max(), l3["sim_to_distractor"].max(), 1.0))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1.1)
    ax.axvline(0.0, linestyle=":", color="gray", linewidth=0.9)
    ax.axhline(0.0, linestyle=":", color="gray", linewidth=0.9)
    ax.set_xlabel(f"sim_to_sample ({TARGET_LAYER} centered cosine)")
    ax.set_ylabel(f"sim_to_distractor ({TARGET_LAYER} centered cosine)")
    saved = save_figure_all_formats(fig, layout.figure_base("panel_b_fusion_form_scatter"))
    plt.close(fig)
    return saved


def save_fusion_summary_panels(
    fusion_metrics: pd.DataFrame,
    layout,
) -> dict[str, dict[str, str]]:
    l3_metrics = fusion_metrics[fusion_metrics["layer"] == TARGET_LAYER].copy()
    plot_df = l3_metrics.assign(panel=TARGET_LAYER)

    def _save_violin(metric_name: str, stem: str, ylabel: str) -> dict[str, str]:
        return _save_single_axis_figure(
            layout,
            stem,
            figsize=(5.0, 4.8),
            draw_fn=lambda _fig, ax: (
                sns.violinplot(
                    data=plot_df,
                    x="panel",
                    y=metric_name,
                    cut=0,
                    inner=None,
                    linewidth=0.8,
                    ax=ax,
                    color=COLOR_DYNAMIC,
                ),
                sns.stripplot(
                    data=plot_df,
                    x="panel",
                    y=metric_name,
                    color="black",
                    size=3.2,
                    alpha=0.45,
                    jitter=0.15,
                    ax=ax,
                ),
                ax.set_xlabel(""),
                ax.set_ylabel(ylabel),
            ),
        )

    return {
        "fusion_dual_score": _save_violin("fusion_dual_score", "panel_c_fusion_dual_score", "Fusion dual score"),
        "fusion_imbalance": _save_violin("fusion_imbalance", "panel_c_fusion_imbalance", "Fusion imbalance"),
    }


def save_specificity_panels(
    specificity_metrics: pd.DataFrame,
    layout,
) -> dict[str, dict[str, str]]:
    l3_metrics = specificity_metrics[specificity_metrics["layer"] == TARGET_LAYER].copy()
    top1_rate = float(l3_metrics["true_pair_top1"].mean())
    return {
        "true_pair_percentile": _save_single_axis_figure(
            layout,
            "panel_d_true_pair_percentile",
            figsize=(4.6, 4.6),
            draw_fn=lambda _fig, ax: (
                sns.histplot(
                    data=l3_metrics,
                    x="true_pair_percentile",
                    bins=16,
                    stat="density",
                    color=COLOR_DYNAMIC,
                    alpha=0.55,
                    ax=ax,
                ),
                ax.set_xlabel("Percentile"),
                ax.set_ylabel("Density"),
            ),
        ),
        "true_pair_z_score": _save_single_axis_figure(
            layout,
            "panel_d_true_pair_z_score",
            figsize=(4.6, 4.6),
            draw_fn=lambda _fig, ax: (
                sns.histplot(
                    data=l3_metrics,
                    x="true_pair_z",
                    bins=16,
                    stat="density",
                    color=COLOR_DYNAMIC,
                    alpha=0.55,
                    ax=ax,
                ),
                ax.set_xlabel("True-pair z-score"),
                ax.set_ylabel("Density"),
            ),
        ),
        "true_pair_top1_rate": _save_single_axis_figure(
            layout,
            "panel_d_true_pair_top1_rate",
            figsize=(4.6, 4.6),
            draw_fn=lambda _fig, ax: (
                ax.bar([TARGET_LAYER], [top1_rate], color=COLOR_DYNAMIC, edgecolor="black", alpha=0.9),
                ax.text(0, top1_rate + 0.02, f"{100.0 * top1_rate:.1f}%", ha="center", va="bottom"),
                ax.set_ylim(0.0, 1.0),
                ax.set_ylabel("Top-1 rate"),
            ),
        ),
    }


def save_whole_over_part_panels(
    whole_over_part_metrics: pd.DataFrame,
    layout,
) -> dict[str, dict[str, str]]:
    l3_metrics = whole_over_part_metrics[whole_over_part_metrics["layer"] == TARGET_LAYER].copy()
    min_axis = float(
        min(
            l3_metrics["best_constituent_similarity"].min(),
            l3_metrics["sim_to_true_pair"].min(),
            -1.0,
        )
    )
    max_axis = float(
        max(
            l3_metrics["best_constituent_similarity"].max(),
            l3_metrics["sim_to_true_pair"].max(),
            1.0,
        )
    )
    return {
        "true_pair_vs_best_part": _save_single_axis_figure(
            layout,
            "panel_e_true_pair_vs_best_part",
            figsize=(5.4, 4.8),
            draw_fn=lambda _fig, ax: (
                ax.scatter(
                    l3_metrics["best_constituent_similarity"],
                    l3_metrics["sim_to_true_pair"],
                    s=42,
                    alpha=0.78,
                    color=COLOR_DYNAMIC,
                    edgecolors="white",
                    linewidths=0.5,
                ),
                ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color="black", linewidth=1.0),
                ax.set_xlabel("Best constituent similarity"),
                ax.set_ylabel("True-pair similarity"),
            ),
        ),
        "wpri_distribution": _save_single_axis_figure(
            layout,
            "panel_e_wpri_distribution",
            figsize=(5.4, 4.8),
            draw_fn=lambda _fig, ax: (
                sns.histplot(
                    data=l3_metrics,
                    x="WPRI",
                    bins=16,
                    stat="density",
                    color=COLOR_DYNAMIC,
                    alpha=0.55,
                    ax=ax,
                ),
                ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0),
                ax.set_xlabel("WPRI"),
                ax.set_ylabel("Density"),
            ),
        ),
    }


def save_shuffled_control_panels(
    specificity_metrics: pd.DataFrame,
    layout,
) -> dict[str, dict[str, str]]:
    l3_metrics = specificity_metrics[specificity_metrics["layer"] == TARGET_LAYER].copy()
    plot_df = l3_metrics[
        ["triplet_id", "true_pair_score", "shuffled_pair_score"]
    ].melt(
        id_vars=["triplet_id"],
        value_vars=["true_pair_score", "shuffled_pair_score"],
        var_name="score_type",
        value_name="score_value",
    )
    label_map = {
        "true_pair_score": "True pair",
        "shuffled_pair_score": "Shuffled pair",
    }
    plot_df["score_type"] = plot_df["score_type"].map(label_map)
    delta_df = l3_metrics.copy()
    delta_df["true_minus_shuffled"] = delta_df["true_pair_score"] - delta_df["shuffled_pair_score"]
    return {
        "true_vs_shuffled_pair_score": _save_single_axis_figure(
            layout,
            "panel_f_true_vs_shuffled_pair_score",
            figsize=(5.5, 4.8),
            draw_fn=lambda _fig, ax: (
                sns.boxplot(
                    data=plot_df,
                    x="score_type",
                    y="score_value",
                    color=COLOR_DYNAMIC,
                    ax=ax,
                ),
                ax.set_xlabel(""),
                ax.set_ylabel("Pair score"),
            ),
        ),
        "true_minus_shuffled_control": _save_single_axis_figure(
            layout,
            "panel_f_true_minus_shuffled_control",
            figsize=(5.5, 4.8),
            draw_fn=lambda _fig, ax: (
                sns.histplot(
                    data=delta_df,
                    x="true_minus_shuffled",
                    bins=16,
                    stat="density",
                    color=COLOR_DYNAMIC,
                    alpha=0.55,
                    ax=ax,
                ),
                ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0),
                ax.set_xlabel("True - shuffled"),
                ax.set_ylabel("Density"),
            ),
        ),
    }


def save_state_bank_npz(
    state_bank: Mapping[str, np.ndarray],
    layout,
    *,
    triplet_ids: np.ndarray,
) -> Path:
    out_path = layout.data_file("state_bank_L3.npz")
    payload: dict[str, np.ndarray] = {"triplet_id": np.asarray(triplet_ids, dtype=np.int64)}
    for condition in CONDITION_NAMES:
        payload[f"{TARGET_LAYER}_{condition}_ux"] = np.asarray(state_bank[condition], dtype=np.float32)
    np.savez_compressed(out_path, **payload)
    return out_path


def main() -> None:
    parser = build_arg_parser()
    config = build_config(parser.parse_args())
    layout = prepare_result_layout(config.output_dir)
    logger = RunLogger(lines=[])
    runtime_device, cuda_fallback = _resolve_runtime_device(config.device)
    log_config_summary(logger, config, runtime_device, cuda_fallback)

    if not config.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {config.model_path}")
    if not config.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {config.dataset_root}")

    logger.log("[Init] Loading dataset and model.")
    dataset = load_mnist_skeleton_dataset(str(config.dataset_root), split=config.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    num_classes = int(labels.max()) + 1
    class_index = build_class_index(dataset, num_classes=num_classes)
    max_duration_ms = max(config.timings.sample_ms, config.timings.distractor_ms, 1.0)
    net, encoder = load_model_and_encoder(
        config.model_path,
        device=runtime_device,
        dt=config.timings.dt,
        max_duration_ms=max_duration_ms,
    )
    logger.log(f"[Init] Dataset loaded: n={len(dataset)} | classes={num_classes} | encoder_max_duration_ms={max_duration_ms:.0f}")

    triplets = build_triplet_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        config=config,
    )
    logger.log(
        "[Triplets] "
        f"generated={len(triplets)} | probes={triplets['probe_id'].nunique()} | "
        f"unique_samples={triplets['sample_id'].nunique()} | unique_distractors={triplets['distractor_id'].nunique()}"
    )

    state_bank = capture_triplet_state_banks(
        triplets,
        images,
        net,
        encoder,
        config=config,
        device=runtime_device,
        logger=logger,
    )
    logger.log("[Capture] Pre-probe state capture complete for mixed / sample_only / distractor_only.")

    fusion_metrics = compute_preprobe_fusion_metrics(triplets, state_bank)
    specificity_metrics = compute_fusion_specificity_metrics(triplets, state_bank, seed=config.seed)
    whole_over_part_metrics = compute_whole_over_part_metrics(fusion_metrics, specificity_metrics)
    logger.log("[Metrics] Fusion, true-pair specificity, and WPRI tables computed.")

    triplet_export_columns = [
        "triplet_id",
        "sample_id",
        "distractor_id",
        "probe_id",
        "sample_label",
        "distractor_label",
        "probe_label",
        "sample_to_probe_similarity",
        "distractor_to_probe_similarity",
        "sample_to_distractor_similarity",
        "sample_bin",
        "distractor_bin",
        "probe_rank",
        "sample_rank_within_probe",
        "selection_scale",
        "selection_seed",
        "split",
    ]
    triplets_csv = save_tidy_csv(triplets[triplet_export_columns].copy(), layout.data_file("triplets.csv"), sort_by=["triplet_id"])
    fusion_csv = save_tidy_csv(fusion_metrics, layout.data_file("preprobe_fusion_metrics.csv"), sort_by=["triplet_id", "layer"])
    specificity_csv = save_tidy_csv(
        specificity_metrics[
            [
                "triplet_id",
                "layer",
                "true_pair_score",
                "true_pair_rank",
                "true_pair_percentile",
                "true_pair_z",
                "true_pair_top1",
                "shuffled_pair_score",
                "mean_other_pair_score",
            ]
        ],
        layout.data_file("fusion_specificity_metrics.csv"),
        sort_by=["triplet_id", "layer"],
    )
    whole_over_part_csv = save_tidy_csv(
        whole_over_part_metrics,
        layout.data_file("whole_over_part_metrics.csv"),
        sort_by=["triplet_id", "layer"],
    )
    state_bank_path = save_state_bank_npz(
        state_bank,
        layout,
        triplet_ids=triplets["triplet_id"].to_numpy(dtype=np.int64, copy=False),
    )
    logger.log(
        "[Save] data="
        f"triplets={triplets_csv} | fusion={fusion_csv} | specificity={specificity_csv} | "
        f"whole_over_part={whole_over_part_csv} | state_bank={state_bank_path}"
    )

    figure_paths: dict[str, object] = {}
    if not config.skip_figures:
        example_triplet = _select_example_triplet(triplets, fusion_metrics, specificity_metrics)
        figure_paths["panel_a"] = save_triplet_definition_panels(example_triplet, images, layout)
        figure_paths["panel_b"] = save_fusion_form_panel(fusion_metrics, layout)
        figure_paths["panel_c"] = save_fusion_summary_panels(fusion_metrics, layout)
        figure_paths["panel_d"] = save_specificity_panels(specificity_metrics, layout)
        figure_paths["panel_e"] = save_whole_over_part_panels(whole_over_part_metrics, layout)
        figure_paths["panel_f"] = save_shuffled_control_panels(specificity_metrics, layout)
        logger.log("[Save] figures exported as standalone subplots to PNG/PDF/SVG.")
    else:
        logger.log("[Save] Figures skipped by --skip-figures.")

    summary = {
        "triplet_count": int(len(triplets)),
        "probe_count": int(triplets["probe_id"].nunique()),
        "unique_sample_count": int(triplets["sample_id"].nunique()),
        "unique_distractor_count": int(triplets["distractor_id"].nunique()),
        "runtime_device": str(runtime_device),
        "cuda_fallback": bool(cuda_fallback),
        "sampling_scale": _sampling_scale_name(config),
        "files": {
            "triplets_csv": str(triplets_csv),
            "preprobe_fusion_metrics_csv": str(fusion_csv),
            "fusion_specificity_metrics_csv": str(specificity_csv),
            "whole_over_part_metrics_csv": str(whole_over_part_csv),
            "state_bank_npz": str(state_bank_path),
        },
        "figure_paths": figure_paths,
        "summary_stats": {
            "L3_fusion_dual_score_mean": float(
                fusion_metrics.loc[fusion_metrics["layer"] == TARGET_LAYER, "fusion_dual_score"].mean()
            ),
            "L3_true_pair_top1_rate": float(
                specificity_metrics.loc[specificity_metrics["layer"] == TARGET_LAYER, "true_pair_top1"].mean()
            ),
            "L3_WPRI_mean": float(
                whole_over_part_metrics.loc[whole_over_part_metrics["layer"] == TARGET_LAYER, "WPRI"].mean()
            ),
        },
    }
    save_run_config(config.to_json_dict(), layout.root)
    save_summary_json(summary, layout.root)
    save_log_lines(logger.lines, layout.log_dir)
    logger.log("[Done] Step 2 fused-state experiment completed.")


if __name__ == "__main__":
    main()
