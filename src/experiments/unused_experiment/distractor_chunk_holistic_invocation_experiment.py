from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.units import ms
from src.experiments.common.dataset import build_class_index, build_dataset_arrays, encode_images
from src.experiments.common.json_io import save_json_payload
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.runtime import seed_everything
from src.experiments.common.seed import mix_seed
from src.experiments.common.voltage_readout import resolve_readout_step
from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_run_config, save_tidy_csv
from src.plotting.common.style import BLUISH_GREEN, ORANGE, SKY_BLUE, VERMILION

EXPERIMENT_NAME = "distractor_chunk_holistic_invocation_experiment"
DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_OUTPUT_DIR = f"results/{EXPERIMENT_NAME}"
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_DELAY1_MS = 400.0
DEFAULT_DISTRACTOR_MS = 200.0
DEFAULT_DELAY2_MS = 400.0
DEFAULT_PROBE_MS = 100.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_PROBES = 20
DEFAULT_SAMPLES_PER_PROBE = 12
DEFAULT_MAX_TRIPLETS = 240
DEFAULT_NUM_SIM_BINS = 4
DEFAULT_FOREGROUND_THRESHOLD = 0.0
DEFAULT_DILATION_RADIUS = 1
DEFAULT_WINNER_WINDOW_FRAC = 0.5
DEFAULT_TIE_THRESHOLD = 0.02
DEFAULT_REDISTRIBUTION_FRACTION = 0.5
DEFAULT_EPS = 1e-12
PRIMARY_FUSION_LAYER = "L3"
PRIMARY_REWRITING_METRIC = "barP_L3"
SMOKE_COMMAND = (
    "conda run -n torch_env python "
    "src/experiments/distractor_chunk_holistic_invocation_experiment.py --device cuda --smoke"
)
SMOKE_NOTE = "smoke experiment should be run in torch_env"

LAYER_KEYS = ("layer1", "layer2", "layer3")
REGION_KEYS = ("SP", "DP", "SDP")
CUE_CONDITIONS = ("cue_SP", "cue_DP", "cue_SDP")
INTERVENTION_CONDITIONS = (
    "clamp_SP",
    "clamp_DP",
    "clamp_SDP",
    "redistribute_DP_to_SDP",
    "redistribute_SDP_to_DP",
)
WINNER_CONDITIONS = ("baseline_intact",) + CUE_CONDITIONS + INTERVENTION_CONDITIONS


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay1_ms: float
    distractor_ms: float
    delay2_ms: float
    probe_ms: float
    phase_reset: bool = True

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
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


@dataclass(frozen=True)
class ExperimentConfig:
    model_path: str
    config: str | None
    dataset_root: str
    split: str
    device: str
    seed: int
    output_dir: str
    sample_ms: float
    delay1_ms: float
    distractor_ms: float
    delay2_ms: float
    probe_ms: float
    batch_size: int
    max_probes: int
    samples_per_probe: int
    max_triplets: int
    num_sim_bins: int
    foreground_threshold: float
    dilation_radius: int
    winner_window_frac: float
    tie_threshold: float
    redistribution_fraction: float
    skip_figures: bool
    smoke: bool


@dataclass(frozen=True)
class ProbeRegionBundle:
    probe_region_masks: dict[str, np.ndarray]
    sample_phase_masks: dict[str, np.ndarray]
    distractor_phase_masks: dict[str, np.ndarray]
    layer_region_masks: dict[str, dict[str, np.ndarray]]
    metadata: dict[str, object]


@dataclass(frozen=True)
class RolloutCapture:
    probe_grouped_voltage_trace: np.ndarray
    distractor_grouped_voltage_trace_l3: np.ndarray
    distractor_l2_trace: np.ndarray
    distractor_l3_trace: np.ndarray
    probe_l2_trace: np.ndarray
    probe_l3_trace: np.ndarray
    preprobe_states: dict[str, dict[str, np.ndarray]]
    readout_step: int
    prediction_probe: np.ndarray
    first_fire_t_probe: np.ndarray
    intervention_record: dict[str, object]


@dataclass(frozen=True)
class OLSCoefficient:
    predictor: str
    beta: float | None
    se: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None


@dataclass(frozen=True)
class OLSFitResult:
    model_name: str
    response: str
    predictors: tuple[str, ...]
    n: int
    r2: float | None
    coefficients: tuple[OLSCoefficient, ...]
    status: str
    note: str | None = None


@dataclass(frozen=True)
class Fig5FusionBackboneResult:
    config: dict[str, Any]
    triplets: pd.DataFrame
    preprobe_fusion_metrics: pd.DataFrame
    fusion_specificity_metrics: pd.DataFrame
    distractor_pull_timeseries: pd.DataFrame
    distractor_pull_summary: pd.DataFrame
    sample_induced_rewriting_timeseries: pd.DataFrame
    sample_induced_rewriting_summary: pd.DataFrame
    rewriting_fusion_bridge: pd.DataFrame
    formation_intervention_metrics: pd.DataFrame
    region_support_condition: pd.DataFrame
    layer1_trial_metrics: pd.DataFrame
    layer1_formula_fit: pd.DataFrame
    holistic_metrics: pd.DataFrame
    cue_winner_metrics: pd.DataFrame
    example_triplet_id: int
    example_preprobe_fusion_state: dict[str, np.ndarray]
    example_distractor_pull_trace: dict[str, np.ndarray]
    stats: dict[str, Any]


def _safe_float(value: Any) -> float | None:
    try:
        scalar = float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return None
    return scalar if np.isfinite(scalar) else None


def _sem(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _finite_xy(x: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    return xx[mask], yy[mask]


def _zscore(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    std = float(values.std(ddof=0))
    if values.size <= 0 or std <= DEFAULT_EPS:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _centered_cosine(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor, eps: float = DEFAULT_EPS) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa = aa - float(aa.mean())
    bb = bb - float(bb.mean())
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= eps or nb <= eps:
        return float("nan")
    return float(np.dot(aa, bb) / (na * nb))


def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return float("nan")
    return float(finite.mean())


def _nanmax(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return float("nan")
    return float(finite.max())


def _nansum(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return float("nan")
    return float(finite.sum())


def _center_rows(values: np.ndarray, eps: float = DEFAULT_EPS) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    centered = arr - arr.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    norms = np.where(norms > float(eps), norms, np.nan)
    return centered, norms


def _pairwise_centered_cosine_matrix(a: np.ndarray, b: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    aa, na = _center_rows(np.asarray(a, dtype=np.float64), eps=eps)
    bb, nb = _center_rows(np.asarray(b, dtype=np.float64), eps=eps)
    denom = np.outer(na, nb)
    sims = (aa @ bb.T) / denom
    sims[~np.isfinite(sims)] = np.nan
    return sims


def _preprobe_state_vectors(
    preprobe_states: Mapping[str, Mapping[str, np.ndarray]],
    *,
    layer_key: str,
    state_key: str = "ux",
) -> np.ndarray:
    values = np.asarray(preprobe_states[layer_key][state_key], dtype=np.float64)
    if values.ndim < 2:
        raise ValueError(f"Expected batched preprobe state for {layer_key}.{state_key}.")
    return values.reshape(values.shape[0], -1)


def _append_preprobe_state_buffer(
    buffer: dict[str, dict[str, object]],
    preprobe_states: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    for layer_key in LAYER_KEYS:
        layer_buffer = buffer.setdefault(layer_key, {"baseline_u": None, "u": [], "x": [], "ux": []})
        if layer_buffer["baseline_u"] is None:
            layer_buffer["baseline_u"] = np.asarray(preprobe_states[layer_key]["baseline_u"], dtype=np.float32)
        for state_key in ("u", "x", "ux"):
            layer_buffer[state_key].append(np.asarray(preprobe_states[layer_key][state_key], dtype=np.float32))


def _finalize_preprobe_state_buffer(buffer: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for layer_key in LAYER_KEYS:
        layer_buffer = buffer[layer_key]
        out[layer_key] = {
            "baseline_u": np.asarray(layer_buffer["baseline_u"], dtype=np.float32),
            "u": np.concatenate(layer_buffer["u"], axis=0).astype(np.float32, copy=False),
            "x": np.concatenate(layer_buffer["x"], axis=0).astype(np.float32, copy=False),
            "ux": np.concatenate(layer_buffer["ux"], axis=0).astype(np.float32, copy=False),
        }
    return out


def _save_npz_payload(path: Path, **arrays: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def _read_config_file(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix in {".json", ""}:
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyYAML is required to read YAML config files.") from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config suffix: {config_path.suffix}")
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError("Config file must contain a top-level mapping.")
    return {str(key): value for key, value in payload.items()}


def _parse_args(argv: Sequence[str] | None = None) -> ExperimentConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_defaults = _read_config_file(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Fig.5 fusion+formation backbone: fused latent memory form, distractor-phase rewriting, and formation-level intervention."
    )
    parser.set_defaults(**config_defaults)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--delay1-ms", type=float, default=DEFAULT_DELAY1_MS)
    parser.add_argument("--distractor-ms", type=float, default=DEFAULT_DISTRACTOR_MS)
    parser.add_argument("--delay2-ms", type=float, default=DEFAULT_DELAY2_MS)
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--samples-per-probe", type=int, default=DEFAULT_SAMPLES_PER_PROBE)
    parser.add_argument("--max-triplets", type=int, default=DEFAULT_MAX_TRIPLETS)
    parser.add_argument("--num-sim-bins", type=int, default=DEFAULT_NUM_SIM_BINS)
    parser.add_argument("--foreground-threshold", type=float, default=DEFAULT_FOREGROUND_THRESHOLD)
    parser.add_argument("--dilation-radius", type=int, default=DEFAULT_DILATION_RADIUS)
    parser.add_argument("--winner-window-frac", type=float, default=DEFAULT_WINNER_WINDOW_FRAC)
    parser.add_argument("--tie-threshold", type=float, default=DEFAULT_TIE_THRESHOLD)
    parser.add_argument("--redistribution-fraction", type=float, default=DEFAULT_REDISTRIBUTION_FRACTION)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    raw = parser.parse_args(argv)
    config = ExperimentConfig(
        model_path=str(raw.model_path),
        config=raw.config,
        dataset_root=str(raw.dataset_root),
        split=str(raw.split),
        device=str(raw.device),
        seed=int(raw.seed),
        output_dir=str(raw.output_dir),
        sample_ms=float(raw.sample_ms),
        delay1_ms=float(raw.delay1_ms),
        distractor_ms=float(raw.distractor_ms),
        delay2_ms=float(raw.delay2_ms),
        probe_ms=float(raw.probe_ms),
        batch_size=int(raw.batch_size),
        max_probes=int(raw.max_probes),
        samples_per_probe=int(raw.samples_per_probe),
        max_triplets=int(raw.max_triplets),
        num_sim_bins=int(raw.num_sim_bins),
        foreground_threshold=float(raw.foreground_threshold),
        dilation_radius=int(raw.dilation_radius),
        winner_window_frac=float(raw.winner_window_frac),
        tie_threshold=float(raw.tie_threshold),
        redistribution_fraction=float(raw.redistribution_fraction),
        skip_figures=bool(raw.skip_figures),
        smoke=bool(raw.smoke),
    )
    return _apply_smoke_overrides(config)


def _apply_smoke_overrides(config: ExperimentConfig) -> ExperimentConfig:
    if not config.smoke:
        return config
    return ExperimentConfig(
        **{
            **asdict(config),
            "batch_size": min(int(config.batch_size), 2),
            "max_probes": min(int(config.max_probes), 2),
            "samples_per_probe": min(int(config.samples_per_probe), 1),
            "max_triplets": min(int(config.max_triplets), 4),
        }
    )


def _validate_config(config: ExperimentConfig) -> None:
    for name in (
        "sample_ms",
        "distractor_ms",
        "probe_ms",
        "batch_size",
        "max_probes",
        "samples_per_probe",
        "max_triplets",
        "num_sim_bins",
    ):
        if float(getattr(config, name)) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    if float(config.delay1_ms) < 0.0 or float(config.delay2_ms) < 0.0:
        raise ValueError("delay1_ms and delay2_ms must be non-negative.")
    if not 0.0 < float(config.winner_window_frac) <= 1.0:
        raise ValueError("winner_window_frac must be in (0, 1].")
    if float(config.redistribution_fraction) < 0.0 or float(config.redistribution_fraction) > 1.0:
        raise ValueError("redistribution_fraction must be in [0, 1].")
    if int(config.dilation_radius) < 0:
        raise ValueError("dilation_radius must be non-negative.")


def _resolve_device(device_arg: str, log_lines: list[str]) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        log_lines.append("device_fallback=CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return requested


def _load_dataset(dataset_root: str, split: str):
    train_loader, _, test_loader = build_mnist_skeleton_loader(
        root=dataset_root,
        batch_size=1,
        input_size=28,
        num_workers=0,
    )
    split_name = str(split).strip().lower()
    if split_name == "train":
        return train_loader.dataset
    if split_name == "test":
        return test_loader.dataset
    raise ValueError(f"Unsupported split: {split}")


def _assign_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 0:
        return np.asarray([], dtype=object)
    q = max(1, min(int(num_bins), int(arr.size)))
    labels = [f"bin_{idx + 1}" for idx in range(q)]
    if q == 1:
        return np.asarray([labels[0]] * arr.size, dtype=object)
    ranks = pd.Series(arr).rank(method="first")
    try:
        return pd.qcut(ranks, q=q, labels=labels).astype("object").to_numpy()
    except ValueError:
        order = np.argsort(arr, kind="stable")
        out = np.empty(arr.size, dtype=object)
        raw = np.linspace(0, q - 1, num=arr.size)
        for pos, idx in enumerate(order.tolist()):
            out[idx] = labels[int(round(raw[pos]))]
        return out


def _select_probe_ids_balanced(
    class_index: Mapping[int, Sequence[int]],
    *,
    max_probes: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(int(seed))
    per_class: dict[int, list[int]] = {}
    for label in sorted(class_index):
        ids = np.asarray([int(idx) for idx in class_index[int(label)]], dtype=np.int64)
        per_class[int(label)] = rng.permutation(ids).tolist()
    selected: list[int] = []
    while len(selected) < int(max_probes):
        progress = False
        for label in sorted(per_class):
            if not per_class[label]:
                continue
            selected.append(int(per_class[label].pop(0)))
            progress = True
            if len(selected) >= int(max_probes):
                break
        if not progress:
            break
    return selected


def _take_evenly(df: pd.DataFrame, count: int) -> list[int]:
    if count <= 0 or df.empty:
        return []
    base_index = df["index"].to_numpy(dtype=np.int64) if "index" in df.columns else df.index.to_numpy(dtype=np.int64)
    if len(df) <= int(count):
        return base_index.astype(int).tolist()
    positions = np.linspace(0, len(df) - 1, num=int(count))
    picked = sorted({int(round(pos)) for pos in positions.tolist()})
    while len(picked) < int(count):
        for idx in range(len(df)):
            if idx not in picked:
                picked.append(idx)
            if len(picked) >= int(count):
                break
    return base_index[np.asarray(sorted(picked[: int(count)]), dtype=np.int64)].astype(int).tolist()


def _select_probe_samples(df_candidates: pd.DataFrame, *, samples_per_probe: int, num_bins: int) -> pd.DataFrame:
    if df_candidates.empty:
        return df_candidates.iloc[:0].copy()
    ordered = df_candidates.sort_values(["sp_similarity", "sample_id"], kind="stable").reset_index(drop=True)
    ordered["sp_bin"] = _assign_bins(ordered["sp_similarity"].to_numpy(dtype=np.float64), num_bins=int(num_bins))
    unique_bins = pd.unique(ordered["sp_bin"]).tolist()
    desired_positions = np.floor(
        np.linspace(0, max(len(unique_bins) - 1, 0), num=max(1, int(samples_per_probe)))
    ).astype(np.int64)
    selected_idx: list[int] = []
    for pos, bin_label in enumerate(unique_bins):
        need = int((desired_positions == int(pos)).sum())
        if need <= 0:
            continue
        sub = ordered[ordered["sp_bin"] == bin_label].copy().reset_index()
        selected_idx.extend(_take_evenly(sub, need))
    if len(selected_idx) < int(samples_per_probe):
        leftovers = ordered.drop(index=selected_idx, errors="ignore").reset_index()
        selected_idx.extend(_take_evenly(leftovers, int(samples_per_probe) - len(selected_idx)))
    return ordered.iloc[sorted(dict.fromkeys(selected_idx))].head(int(samples_per_probe)).copy().reset_index(drop=True)


def build_triplet_specs(
    images: torch.Tensor,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    *,
    max_probes: int,
    samples_per_probe: int,
    num_bins: int,
    max_triplets: int,
    seed: int,
) -> pd.DataFrame:
    del images
    all_ids = np.arange(len(labels), dtype=np.int64)
    probe_ids = _select_probe_ids_balanced(class_index, max_probes=int(max_probes), seed=mix_seed(seed, 31))
    rows: list[dict[str, object]] = []
    for probe_rank, probe_id in enumerate(probe_ids):
        probe_label = int(labels[int(probe_id)])
        sims_to_probe = flat_normalized @ flat_normalized[int(probe_id)]
        sample_mask = (all_ids != int(probe_id)) & (labels[all_ids] != probe_label)
        sample_ids = all_ids[sample_mask]
        df_samples = pd.DataFrame(
            {
                "sample_id": sample_ids.astype(np.int64, copy=False),
                "sample_label": labels[sample_ids].astype(np.int64, copy=False),
                "probe_id": int(probe_id),
                "probe_label": probe_label,
                "probe_rank": int(probe_rank),
                "sp_similarity": sims_to_probe[sample_ids].astype(np.float64, copy=False),
            }
        )
        selected_samples = _select_probe_samples(
            df_samples,
            samples_per_probe=int(samples_per_probe),
            num_bins=int(num_bins),
        )
        if selected_samples.empty:
            continue
        for sample_pos, sample_row in enumerate(selected_samples.itertuples(index=False)):
            sims_to_sample = flat_normalized @ flat_normalized[int(sample_row.sample_id)]
            distractor_mask = (
                (all_ids != int(sample_row.sample_id))
                & (all_ids != int(probe_id))
                & (labels[all_ids] != int(sample_row.sample_label))
                & (labels[all_ids] != int(probe_label))
            )
            distractor_ids = all_ids[distractor_mask]
            if distractor_ids.size <= 0:
                continue
            df_distractors = pd.DataFrame(
                {
                    "distractor_id": distractor_ids.astype(np.int64, copy=False),
                    "distractor_label": labels[distractor_ids].astype(np.int64, copy=False),
                    "dp_similarity": sims_to_probe[distractor_ids].astype(np.float64, copy=False),
                    "sd_similarity": sims_to_sample[distractor_ids].astype(np.float64, copy=False),
                }
            ).sort_values(["dp_similarity", "sd_similarity", "distractor_id"], kind="stable")
            df_distractors["dp_bin"] = _assign_bins(
                df_distractors["dp_similarity"].to_numpy(dtype=np.float64),
                num_bins=int(num_bins),
            )
            unique_dp_bins = pd.unique(df_distractors["dp_bin"]).tolist()
            target_positions = np.floor(
                np.linspace(0, max(len(unique_dp_bins) - 1, 0), num=max(1, len(selected_samples)))
            ).astype(np.int64)
            target_bin = str(unique_dp_bins[int(target_positions[min(int(sample_pos), len(target_positions) - 1)])])
            sub = df_distractors[df_distractors["dp_bin"] == target_bin].copy().reset_index(drop=True)
            if sub.empty:
                sub = df_distractors.reset_index(drop=True)
            chosen = sub.iloc[min(int(sample_pos), len(sub) - 1)]
            rows.append(
                {
                    "sample_id": int(sample_row.sample_id),
                    "sample_label": int(sample_row.sample_label),
                    "distractor_id": int(chosen["distractor_id"]),
                    "distractor_label": int(chosen["distractor_label"]),
                    "probe_id": int(probe_id),
                    "probe_label": int(probe_label),
                    "probe_rank": int(probe_rank),
                    "sp_similarity": float(sample_row.sp_similarity),
                    "dp_similarity": float(chosen["dp_similarity"]),
                    "sd_similarity": float(chosen["sd_similarity"]),
                    "sp_bin": str(sample_row.sp_bin),
                    "dp_bin": str(chosen["dp_bin"]),
                }
            )
    if not rows:
        raise RuntimeError("No triplets were generated.")
    df_triplets = pd.DataFrame(rows).drop_duplicates(subset=["sample_id", "distractor_id", "probe_id"]).reset_index(drop=True)
    if len(df_triplets) > int(max_triplets):
        parts: list[pd.DataFrame] = []
        by_probe = [sub.copy().reset_index(drop=True) for _, sub in df_triplets.groupby("probe_id", sort=True)]
        cursor = 0
        while len(parts) < int(max_triplets):
            progress = False
            for group in by_probe:
                if cursor >= len(group):
                    continue
                parts.append(group.iloc[[cursor]].copy())
                progress = True
                if len(parts) >= int(max_triplets):
                    break
            if not progress:
                break
            cursor += 1
        df_triplets = pd.concat(parts, axis=0, ignore_index=True)
    df_triplets = df_triplets.sort_values(
        ["probe_rank", "sp_similarity", "dp_similarity", "sample_id", "distractor_id"],
        kind="stable",
    ).reset_index(drop=True)
    df_triplets["triplet_id"] = np.arange(len(df_triplets), dtype=np.int64)
    return df_triplets


def _foreground_mask(image: torch.Tensor, threshold: float) -> np.ndarray:
    arr = image.detach().cpu().to(torch.float32).abs().amax(dim=0).numpy()
    return np.asarray(arr > float(threshold), dtype=bool)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if int(radius) <= 0 or not mask_bool.any():
        return mask_bool
    tensor = torch.as_tensor(mask_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel = 2 * int(radius) + 1
    out = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(radius))
    return np.asarray(out.squeeze(0).squeeze(0).numpy() > 0.0, dtype=bool)


def _compute_layer_spatial_shapes(net, input_hw: tuple[int, int]) -> dict[str, tuple[int, int]]:
    height, width = int(input_hw[0]), int(input_hw[1])
    h1 = (height + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    w1 = (width + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    h1_p, w1_p = h1 // 2, w1 // 2
    h2 = (h1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    w2 = (w1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    h2_p, w2_p = h2 // 2, w2 // 2
    return {"layer1": (height, width), "layer2": (h1_p, w1_p), "layer3": (h2_p, w2_p)}


def _project_mask_to_shape(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if tuple(mask_bool.shape) == tuple(target_hw):
        return mask_bool.copy()
    tensor = torch.as_tensor(mask_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_max_pool2d(tensor, output_size=tuple(int(v) for v in target_hw))
    return np.asarray(pooled.squeeze(0).squeeze(0).numpy() > 0.0, dtype=bool)


def build_probe_region_bundle(
    sample_image: torch.Tensor,
    distractor_image: torch.Tensor,
    probe_image: torch.Tensor,
    *,
    net,
    foreground_threshold: float,
    dilation_radius: int,
) -> ProbeRegionBundle:
    sample_fg = _foreground_mask(sample_image, threshold=float(foreground_threshold))
    distractor_fg = _foreground_mask(distractor_image, threshold=float(foreground_threshold))
    probe_fg = _foreground_mask(probe_image, threshold=float(foreground_threshold))

    sp = sample_fg & probe_fg & ~distractor_fg
    dp = distractor_fg & probe_fg & ~sample_fg
    sdp = sample_fg & distractor_fg & probe_fg
    full = probe_fg.copy()
    union = sp | dp | sdp

    sample_sp = _dilate_mask(sp, int(dilation_radius)) & sample_fg if sp.any() else sp.copy()
    sample_sdp = _dilate_mask(sdp, int(dilation_radius)) & sample_fg if sdp.any() else sdp.copy()
    distractor_dp = _dilate_mask(dp, int(dilation_radius)) & distractor_fg if dp.any() else dp.copy()
    distractor_sdp = _dilate_mask(sdp, int(dilation_radius)) & distractor_fg if sdp.any() else sdp.copy()

    spatial_shapes = _compute_layer_spatial_shapes(net, input_hw=(probe_image.shape[-2], probe_image.shape[-1]))
    layer_region_masks: dict[str, dict[str, np.ndarray]] = {}
    for layer_key, target_hw in spatial_shapes.items():
        layer_region_masks[layer_key] = {
            "SP": _project_mask_to_shape(sp, target_hw),
            "DP": _project_mask_to_shape(dp, target_hw),
            "SDP": _project_mask_to_shape(sdp, target_hw),
            "FULL": _project_mask_to_shape(full, target_hw),
            "UNION": _project_mask_to_shape(union, target_hw),
        }

    metadata = {
        "probe_foreground_area": int(full.sum()),
        "sample_foreground_area": int(sample_fg.sum()),
        "distractor_foreground_area": int(distractor_fg.sum()),
        "area_SP": int(sp.sum()),
        "area_DP": int(dp.sum()),
        "area_SDP": int(sdp.sum()),
        "area_union": int(union.sum()),
        "empty_SP": int(int(sp.sum()) == 0),
        "empty_DP": int(int(dp.sum()) == 0),
        "empty_SDP": int(int(sdp.sum()) == 0),
        "foreground_threshold": float(foreground_threshold),
        "dilation_radius": int(dilation_radius),
    }
    return ProbeRegionBundle(
        probe_region_masks={"SP": sp, "DP": dp, "SDP": sdp, "FULL": full, "UNION": union},
        sample_phase_masks={"SP": sample_sp, "SDP": sample_sdp, "UNION": sample_sp | sample_sdp},
        distractor_phase_masks={"DP": distractor_dp, "SDP": distractor_sdp, "UNION": distractor_dp | distractor_sdp},
        layer_region_masks=layer_region_masks,
        metadata=metadata,
    )


def _stack_encoded(
    image_ids: Sequence[int],
    *,
    images: torch.Tensor,
    encoder,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    batch_images = images[[int(idx) for idx in image_ids]].to(device=device, dtype=torch.float32)
    return encode_images(encoder, batch_images, steps=int(steps))


def prepare_triplet_batches(
    images: torch.Tensor,
    batch_df: pd.DataFrame,
    *,
    encoder,
    spec: ExperimentSpec,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sample_ids = batch_df["sample_id"].astype(int).tolist()
    distractor_ids = batch_df["distractor_id"].astype(int).tolist()
    probe_ids = batch_df["probe_id"].astype(int).tolist()

    unique_sample_ids = list(dict.fromkeys(sample_ids))
    unique_distractor_ids = list(dict.fromkeys(distractor_ids))
    unique_probe_ids = list(dict.fromkeys(probe_ids))

    sample_encoded = _stack_encoded(unique_sample_ids, images=images, encoder=encoder, steps=spec.sample_steps, device=device)
    distractor_encoded = _stack_encoded(unique_distractor_ids, images=images, encoder=encoder, steps=spec.distractor_steps, device=device)
    distractor_as_sample = _stack_encoded(unique_distractor_ids, images=images, encoder=encoder, steps=spec.sample_steps, device=device)
    probe_encoded = _stack_encoded(unique_probe_ids, images=images, encoder=encoder, steps=spec.probe_steps, device=device)

    sample_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_sample_ids)}
    distractor_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_distractor_ids)}
    probe_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_probe_ids)}

    sample_select = torch.tensor([sample_lookup[int(idx)] for idx in sample_ids], dtype=torch.long, device=device)
    distractor_select = torch.tensor([distractor_lookup[int(idx)] for idx in distractor_ids], dtype=torch.long, device=device)
    probe_select = torch.tensor([probe_lookup[int(idx)] for idx in probe_ids], dtype=torch.long, device=device)

    sample_spikes = sample_encoded.index_select(0, sample_select)
    distractor_spikes = distractor_encoded.index_select(0, distractor_select)
    distractor_sample_spikes = distractor_as_sample.index_select(0, distractor_select)
    probe_spikes = probe_encoded.index_select(0, probe_select)
    return {
        "sample": sample_spikes,
        "distractor": distractor_spikes,
        "distractor_as_sample": distractor_sample_spikes,
        "probe": probe_spikes,
        "zero_sample": torch.zeros_like(sample_spikes),
        "zero_distractor": torch.zeros_like(distractor_spikes),
        "zero_probe": torch.zeros_like(probe_spikes),
    }


def _mask_spike_batch_keep_region(spike_batch: torch.Tensor, keep_masks: Sequence[np.ndarray]) -> torch.Tensor:
    if spike_batch.ndim != 5:
        raise ValueError(f"Expected [B, T, C, H, W], got {tuple(spike_batch.shape)}")
    if len(keep_masks) != int(spike_batch.shape[0]):
        raise ValueError("keep_masks length must match spike batch size.")
    stacked = np.stack([np.asarray(mask, dtype=bool) for mask in keep_masks], axis=0)
    mask_tensor = torch.as_tensor(stacked, dtype=torch.bool, device=spike_batch.device)
    mask_tensor = mask_tensor[:, None, None, :, :]
    return spike_batch * mask_tensor.to(dtype=spike_batch.dtype)


def _extract_grouped_voltage_vector(net, voltage_snapshot: torch.Tensor) -> np.ndarray:
    grouped = net.layer3.get_grouped_voltage(voltage_snapshot.to(torch.float32))
    return grouped.mean(dim=-1).detach().cpu().numpy().astype(np.float64, copy=False)


def _broadcast_batch_spatial_mask(mask_batch: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    mask = mask_batch.to(device=tensor.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError("mask_batch must have shape [B, H, W] or [H, W].")
    if mask.shape[0] == 1 and tensor.shape[0] > 1:
        mask = mask.expand(tensor.shape[0], -1, -1)
    if mask.shape[0] != tensor.shape[0]:
        raise ValueError("mask batch dimension must match tensor batch dimension.")
    return mask.unsqueeze(1).expand(-1, tensor.shape[1], -1, -1)


def _stack_region_masks(
    batch_bundles: Sequence[ProbeRegionBundle],
    *,
    layer_key: str,
    region_key: str,
    device: torch.device,
) -> torch.Tensor:
    masks = np.stack(
        [np.asarray(bundle.layer_region_masks[layer_key][region_key], dtype=bool) for bundle in batch_bundles],
        axis=0,
    )
    return torch.as_tensor(masks, dtype=torch.bool, device=device)


def _positive_excess_mass_tensor(gain: torch.Tensor, baseline_u: float, region_mask: torch.Tensor) -> torch.Tensor:
    broad_mask = _broadcast_batch_spatial_mask(region_mask, gain)
    excess = torch.relu(gain - float(baseline_u)) * broad_mask.to(dtype=gain.dtype)
    return excess.sum(dim=(1, 2, 3))


def snapshot_preprobe_ux(net) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        if getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
            raise RuntimeError(f"{layer_key} does not expose STSP state.")
        u = layer.u_pre.detach().cpu().numpy().astype(np.float32, copy=False)
        x = layer.x_pre.detach().cpu().numpy().astype(np.float32, copy=False)
        ux = (layer.u_pre * layer.x_pre).detach().cpu().numpy().astype(np.float32, copy=False)
        out[layer_key] = {"u": u, "x": x, "ux": ux, "baseline_u": np.asarray(float(layer.stsp_U), dtype=np.float32)}
    return out


def compute_region_support_summary(
    *,
    triplet_id: int,
    condition: str,
    preprobe_states: Mapping[str, Mapping[str, np.ndarray]],
    bundle: ProbeRegionBundle,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_key in LAYER_KEYS:
        layer_state = preprobe_states[layer_key]
        ux = np.asarray(layer_state["ux"], dtype=np.float64)
        baseline_u = float(layer_state["baseline_u"])
        for region_key in (*REGION_KEYS, "FULL", "UNION"):
            mask = np.asarray(bundle.layer_region_masks[layer_key][region_key], dtype=bool)
            channel_mask = np.broadcast_to(mask[None, :, :], ux.shape[1:]).reshape(-1)
            sample_ux = ux[0].reshape(-1)[channel_mask]
            if sample_ux.size <= 0:
                mean_ux = float("nan")
                mean_positive_excess = float("nan")
                support_mass = 0.0
            else:
                positive_excess = np.maximum(sample_ux - baseline_u, 0.0)
                mean_ux = float(sample_ux.mean())
                mean_positive_excess = float(positive_excess.mean())
                support_mass = float(positive_excess.sum())
            rows.append(
                {
                    "triplet_id": int(triplet_id),
                    "condition": str(condition),
                    "layer": str(layer_key),
                    "region": str(region_key),
                    "region_area": int(mask.sum()),
                    "mean_ux": mean_ux,
                    "mean_positive_excess_ux": mean_positive_excess,
                    "support_mass": support_mass,
                }
            )
    return rows


def _slice_state_value(value: np.ndarray, index: int):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr
    return arr[[int(index)]]


def _build_clamp_before_probe_fn(
    *,
    batch_bundles: Sequence[ProbeRegionBundle],
    region_key: str,
) -> Callable[[Any, dict[str, object]], dict[str, object]]:
    def before_probe_fn(net, ctx: dict[str, object]) -> dict[str, object]:
        del ctx
        record: dict[str, object] = {"type": "clamp", "region": str(region_key), "layers": {}}
        with torch.no_grad():
            for layer_key in LAYER_KEYS:
                layer = getattr(net, layer_key)
                mask_batch = _stack_region_masks(
                    batch_bundles,
                    layer_key=layer_key,
                    region_key=region_key,
                    device=layer.u_pre.device,
                )
                gain_before = layer.u_pre * layer.x_pre
                support_before = _positive_excess_mass_tensor(gain_before, float(layer.stsp_U), mask_batch)
                broad_mask = _broadcast_batch_spatial_mask(mask_batch, layer.u_pre)
                baseline_tensor = torch.full_like(layer.u_pre, float(layer.stsp_U))
                layer.u_pre.copy_(torch.where(broad_mask, baseline_tensor, layer.u_pre))
                layer.x_pre.copy_(torch.where(broad_mask, torch.ones_like(layer.x_pre), layer.x_pre))
                gain_after = layer.u_pre * layer.x_pre
                support_after = _positive_excess_mass_tensor(gain_after, float(layer.stsp_U), mask_batch)
                record["layers"][layer_key] = {
                    "support_before": support_before.detach().cpu().numpy().astype(np.float64).tolist(),
                    "support_after": support_after.detach().cpu().numpy().astype(np.float64).tolist(),
                    "support_delta": (support_after - support_before).detach().cpu().numpy().astype(np.float64).tolist(),
                }
        return record

    return before_probe_fn


def _build_redistribution_before_probe_fn(
    *,
    batch_bundles: Sequence[ProbeRegionBundle],
    source_region: str,
    target_region: str,
    fraction: float,
) -> Callable[[Any, dict[str, object]], dict[str, object]]:
    def before_probe_fn(net, ctx: dict[str, object]) -> dict[str, object]:
        del ctx
        record: dict[str, object] = {
            "type": "redistribution",
            "source_region": str(source_region),
            "target_region": str(target_region),
            "fraction": float(fraction),
            "layers": {},
        }
        with torch.no_grad():
            for layer_key in LAYER_KEYS:
                layer = getattr(net, layer_key)
                source_mask = _stack_region_masks(
                    batch_bundles,
                    layer_key=layer_key,
                    region_key=source_region,
                    device=layer.u_pre.device,
                )
                target_mask = _stack_region_masks(
                    batch_bundles,
                    layer_key=layer_key,
                    region_key=target_region,
                    device=layer.u_pre.device,
                )
                gain_before = layer.u_pre * layer.x_pre
                source_broadcast = _broadcast_batch_spatial_mask(source_mask, gain_before).to(dtype=gain_before.dtype)
                target_broadcast = _broadcast_batch_spatial_mask(target_mask, gain_before).to(dtype=gain_before.dtype)
                baseline_u = float(layer.stsp_U)
                baseline_gain = torch.full_like(gain_before, baseline_u)
                source_excess = torch.relu(gain_before - baseline_gain) * source_broadcast
                source_removed = float(fraction) * source_excess
                gain_after_source = gain_before - source_removed

                target_capacity = torch.clamp(1.0 - gain_after_source, min=0.0) * target_broadcast
                requested_transfer = source_removed.sum(dim=(1, 2, 3))
                capacity_total = target_capacity.sum(dim=(1, 2, 3))
                actual_transfer = torch.minimum(requested_transfer, capacity_total)
                weight_den = torch.clamp(capacity_total.view(-1, 1, 1, 1), min=DEFAULT_EPS)
                weights = target_capacity / weight_den
                added = weights * actual_transfer.view(-1, 1, 1, 1)
                gain_after = torch.clamp(gain_after_source + added, min=0.0, max=1.0)

                support_before_source = _positive_excess_mass_tensor(gain_before, baseline_u, source_mask)
                support_before_target = _positive_excess_mass_tensor(gain_before, baseline_u, target_mask)
                total_mask = _stack_region_masks(
                    batch_bundles,
                    layer_key=layer_key,
                    region_key="UNION",
                    device=layer.u_pre.device,
                )
                total_before = _positive_excess_mass_tensor(gain_before, baseline_u, total_mask)

                layer.x_pre.copy_(torch.clamp(gain_after / torch.clamp(layer.u_pre, min=DEFAULT_EPS), min=0.0, max=1.0))
                gain_final = layer.u_pre * layer.x_pre

                support_after_source = _positive_excess_mass_tensor(gain_final, baseline_u, source_mask)
                support_after_target = _positive_excess_mass_tensor(gain_final, baseline_u, target_mask)
                total_after = _positive_excess_mass_tensor(gain_final, baseline_u, total_mask)
                record["layers"][layer_key] = {
                    "requested_transfer": requested_transfer.detach().cpu().numpy().astype(np.float64).tolist(),
                    "actual_transfer": actual_transfer.detach().cpu().numpy().astype(np.float64).tolist(),
                    "source_before": support_before_source.detach().cpu().numpy().astype(np.float64).tolist(),
                    "source_after": support_after_source.detach().cpu().numpy().astype(np.float64).tolist(),
                    "target_before": support_before_target.detach().cpu().numpy().astype(np.float64).tolist(),
                    "target_after": support_after_target.detach().cpu().numpy().astype(np.float64).tolist(),
                    "total_before": total_before.detach().cpu().numpy().astype(np.float64).tolist(),
                    "total_after": total_after.detach().cpu().numpy().astype(np.float64).tolist(),
                    "total_delta": (total_after - total_before).detach().cpu().numpy().astype(np.float64).tolist(),
                }
        return record

    return before_probe_fn


def _build_clamp_sample_trace_before_distractor_fn(
    *,
    layer_keys: Sequence[str] = ("layer2", "layer3"),
) -> Callable[[Any, dict[str, object]], dict[str, object]]:
    def before_distractor_fn(net, ctx: dict[str, object]) -> dict[str, object]:
        del ctx
        record: dict[str, object] = {
            "type": "clamp_sample_trace_before_distractor",
            "target_layers": [str(layer_key) for layer_key in layer_keys],
            "layers": {},
        }
        with torch.no_grad():
            for layer_key in layer_keys:
                layer = getattr(net, str(layer_key))
                gain_before = layer.u_pre * layer.x_pre
                excess_before = torch.relu(gain_before - float(layer.stsp_U)).sum(dim=(1, 2, 3))
                layer.u_pre.fill_(float(layer.stsp_U))
                layer.x_pre.fill_(1.0)
                gain_after = layer.u_pre * layer.x_pre
                excess_after = torch.relu(gain_after - float(layer.stsp_U)).sum(dim=(1, 2, 3))
                record["layers"][str(layer_key)] = {
                    "total_excess_before": excess_before.detach().cpu().numpy().astype(np.float64).tolist(),
                    "total_excess_after": excess_after.detach().cpu().numpy().astype(np.float64).tolist(),
                    "total_excess_delta": (excess_after - excess_before).detach().cpu().numpy().astype(np.float64).tolist(),
                }
        return record

    return before_distractor_fn


def run_distractor_rollout_capture(
    net,
    *,
    sample_spikes: torch.Tensor,
    distractor_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    spec: ExperimentSpec,
    readout_step: int,
    stsp_mode: str = "dynamic",
    before_distractor_fn: Callable[[Any, dict[str, object]], dict[str, object]] | None = None,
    before_probe_fn: Callable[[Any, dict[str, object]], dict[str, object]] | None = None,
) -> RolloutCapture:
    if sample_spikes.ndim != 5 or distractor_spikes.ndim != 5 or probe_spikes.ndim != 5:
        raise ValueError("Expected spike tensors with shape [B, T, C, H, W].")
    batch_size, _, channels, height, width = sample_spikes.shape
    prepare_network_state(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), dtype=sample_spikes.dtype, device=sample_spikes.device)
    current_time = 0
    distractor_l2_frames: list[np.ndarray] = []
    distractor_l3_frames: list[np.ndarray] = []
    probe_l2_frames: list[np.ndarray] = []
    probe_l3_frames: list[np.ndarray] = []
    probe_grouped_trace: list[np.ndarray] = []
    distractor_grouped_trace: list[np.ndarray] = []
    intervention_record: dict[str, object] = {}

    def reset_decision_window() -> None:
        net.layer3.reset_decision_state()
        if bool(spec.phase_reset):
            with torch.no_grad():
                net.layer3.v_mem.fill_(net.layer3.V_L)
                net.layer3.lateral_inh.reset_state(net.layer3.output_shape)

    def step_network(input_t: torch.Tensor, *, phase_name: str, phase_step: int, force_l3_time: int | None = None) -> None:
        nonlocal current_time
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())
        s2, m2 = net.layer2.forward_step(s1_p, current_time, training=False, monitor=True, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())
        l3_time = current_time if force_l3_time is None else force_l3_time
        _, m3 = net.layer3.forward_step(s2_p, l3_time, training=False, monitor=True, stsp_mode=stsp_mode)
        l2_summary = m2["v_mem_snapshot"].detach().mean(dim=(-2, -1)).cpu().numpy().astype(np.float64, copy=False)
        l3_grouped = _extract_grouped_voltage_vector(net, m3["v_mem_snapshot"])
        if phase_name == "distractor":
            distractor_l2_frames.append(l2_summary)
            distractor_l3_frames.append(l3_grouped)
            distractor_grouped_trace.append(l3_grouped)
        elif phase_name == "probe":
            probe_l2_frames.append(l2_summary)
            probe_l3_frames.append(l3_grouped)
            probe_grouped_trace.append(l3_grouped)
        current_time += 1

    with torch.no_grad():
        for t_step in range(int(sample_spikes.shape[1])):
            step_network(sample_spikes[:, t_step, ...], phase_name="sample", phase_step=t_step)
        for t_step in range(int(spec.delay1_steps)):
            step_network(zero_input, phase_name="delay1", phase_step=t_step)
        reset_decision_window()
        if before_distractor_fn is not None:
            intervention_record["before_distractor"] = dict(
                before_distractor_fn(
                    net,
                    {
                        "readout_step": int(readout_step),
                        "current_time": int(current_time),
                        "stage": "before_distractor",
                    },
                )
            )
        for t_step in range(int(distractor_spikes.shape[1])):
            force_t = int(t_step) if bool(spec.phase_reset) else None
            step_network(
                distractor_spikes[:, t_step, ...],
                phase_name="distractor",
                phase_step=t_step,
                force_l3_time=force_t,
            )
        for t_step in range(int(spec.delay2_steps)):
            step_network(zero_input, phase_name="delay2", phase_step=t_step)

        if before_probe_fn is not None:
            intervention_record["before_probe"] = dict(
                before_probe_fn(
                    net,
                    {
                        "readout_step": int(readout_step),
                        "current_time": int(current_time),
                        "stage": "before_probe",
                    },
                )
            )
        preprobe_states = snapshot_preprobe_ux(net)

        reset_decision_window()
        for t_step in range(int(probe_spikes.shape[1])):
            force_t = int(t_step) if bool(spec.phase_reset) else None
            step_network(
                probe_spikes[:, t_step, ...],
                phase_name="probe",
                phase_step=t_step,
                force_l3_time=force_t,
            )

    flat_times = net.layer3.firing_times
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    prediction_probe = (min_indices // net.layer3.neurons_per_class).long()
    prediction_probe[~has_fired] = -1
    first_fire_t_probe = min_times.clone()
    first_fire_t_probe[~has_fired] = -1
    first_fire_t_probe = first_fire_t_probe.to(torch.long)

    probe_trace = np.stack(probe_grouped_trace, axis=1) if probe_grouped_trace else np.zeros((batch_size, 0, net.layer3.num_classes))
    distractor_trace = (
        np.stack(distractor_grouped_trace, axis=1)
        if distractor_grouped_trace
        else np.zeros((batch_size, 0, net.layer3.num_classes))
    )
    return RolloutCapture(
        probe_grouped_voltage_trace=probe_trace,
        distractor_grouped_voltage_trace_l3=distractor_trace,
        distractor_l2_trace=np.stack(distractor_l2_frames, axis=1)
        if distractor_l2_frames
        else np.zeros((batch_size, 0, net.layer2.out_channels)),
        distractor_l3_trace=distractor_trace,
        probe_l2_trace=np.stack(probe_l2_frames, axis=1)
        if probe_l2_frames
        else np.zeros((batch_size, 0, net.layer2.out_channels)),
        probe_l3_trace=probe_trace,
        preprobe_states=preprobe_states,
        readout_step=int(readout_step),
        prediction_probe=prediction_probe.detach().cpu().numpy().astype(np.int64, copy=False),
        first_fire_t_probe=first_fire_t_probe.detach().cpu().numpy().astype(np.int64, copy=False),
        intervention_record=intervention_record,
    )


def _winner_window_indices(trace_len: int, frac: float) -> np.ndarray:
    count = max(1, int(math.ceil(float(trace_len) * float(frac))))
    return np.arange(max(0, int(trace_len) - count), int(trace_len), dtype=np.int64)


def compute_winner_metrics(
    *,
    condition_name: str,
    triplet_ids: Sequence[int],
    condition_trace: np.ndarray,
    sample_reference_trace: np.ndarray,
    distractor_reference_trace: np.ndarray,
    winner_window_frac: float,
    tie_threshold: float,
    predictions: np.ndarray,
    first_fire: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if condition_trace.shape != sample_reference_trace.shape or condition_trace.shape != distractor_reference_trace.shape:
        raise ValueError("Winner traces must share shape [B, T, C].")
    window = _winner_window_indices(condition_trace.shape[1], frac=float(winner_window_frac))
    for batch_idx, triplet_id in enumerate(triplet_ids):
        sample_like = []
        distractor_like = []
        winner_trace = []
        for t_step in range(condition_trace.shape[1]):
            s_sim = _centered_cosine(condition_trace[batch_idx, t_step], sample_reference_trace[batch_idx, t_step])
            d_sim = _centered_cosine(condition_trace[batch_idx, t_step], distractor_reference_trace[batch_idx, t_step])
            sample_like.append(s_sim)
            distractor_like.append(d_sim)
            winner_trace.append(s_sim - d_sim if np.isfinite(s_sim) and np.isfinite(d_sim) else float("nan"))
        window_values = np.asarray([winner_trace[idx] for idx in window], dtype=np.float64)
        w_probe = float(np.nanmean(window_values)) if np.isfinite(window_values).any() else 0.0
        if w_probe > float(tie_threshold):
            winner_label = "sample_win"
        elif w_probe < -float(tie_threshold):
            winner_label = "distractor_win"
        else:
            winner_label = "tie"
        rows.append(
            {
                "triplet_id": int(triplet_id),
                "condition": str(condition_name),
                "W_probe": float(w_probe),
                "sample_like_mean": _nanmean(sample_like),
                "distractor_like_mean": _nanmean(distractor_like),
                "sample_win": int(winner_label == "sample_win"),
                "distractor_win": int(winner_label == "distractor_win"),
                "tie": int(winner_label == "tie"),
                "winner_label": winner_label,
                "prediction_probe": int(predictions[batch_idx]),
                "first_fire_t_probe": int(first_fire[batch_idx]),
            }
        )
    return pd.DataFrame(rows)


def compute_reshaping_metrics(
    *,
    triplet_ids: Sequence[int],
    mixed_capture: RolloutCapture,
    distractor_only_capture: RolloutCapture,
    sample_only_capture: RolloutCapture,
    include_timeseries: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    for batch_idx, triplet_id in enumerate(triplet_ids):
        r_l2 = []
        r_l3 = []
        p_l2 = []
        p_l3 = []
        sim_to_distractor_ref_l3 = []
        sim_to_sample_ref_l3 = []
        for t_step in range(mixed_capture.distractor_l2_trace.shape[1]):
            mix_l2 = mixed_capture.distractor_l2_trace[batch_idx, t_step]
            donly_l2 = distractor_only_capture.distractor_l2_trace[batch_idx, t_step]
            sonly_l2 = sample_only_capture.distractor_l2_trace[batch_idx, t_step]
            mix_l3 = mixed_capture.distractor_l3_trace[batch_idx, t_step]
            donly_l3 = distractor_only_capture.distractor_l3_trace[batch_idx, t_step]
            sonly_l3 = sample_only_capture.distractor_l3_trace[batch_idx, t_step]
            cos_d_l2 = _centered_cosine(mix_l2, donly_l2)
            cos_s_l2 = _centered_cosine(mix_l2, sonly_l2)
            cos_d_l3 = _centered_cosine(mix_l3, donly_l3)
            cos_s_l3 = _centered_cosine(mix_l3, sonly_l3)
            pull_l2 = (cos_s_l2 - cos_d_l2) if np.isfinite(cos_s_l2) and np.isfinite(cos_d_l2) else float("nan")
            pull_l3 = (cos_s_l3 - cos_d_l3) if np.isfinite(cos_s_l3) and np.isfinite(cos_d_l3) else float("nan")
            r_l2.append(1.0 - cos_d_l2 if np.isfinite(cos_d_l2) else float("nan"))
            r_l3.append(1.0 - cos_d_l3 if np.isfinite(cos_d_l3) else float("nan"))
            p_l2.append(pull_l2)
            p_l3.append(pull_l3)
            sim_to_distractor_ref_l3.append(cos_d_l3)
            sim_to_sample_ref_l3.append(cos_s_l3)
            time_rows.append(
                {
                    "triplet_id": int(triplet_id),
                    "layer": "layer2",
                    "distractor_step": int(t_step),
                    "pull_t": float(pull_l2),
                }
            )
            time_rows.append(
                {
                    "triplet_id": int(triplet_id),
                    "layer": "layer3",
                    "distractor_step": int(t_step),
                    "pull_t": float(pull_l3),
                }
            )
        early_count = max(1, int(math.ceil(max(len(p_l3), 1) * 0.25)))
        summary_rows.append(
            {
                "triplet_id": int(triplet_id),
                "barR_L2": _nanmean(r_l2),
                "barR_L3": _nanmean(r_l3),
                "barP_L2": _nanmean(p_l2),
                "barP_L3": _nanmean(p_l3),
                "peakP_L2": _nanmax(p_l2),
                "peakP_L3": _nanmax(p_l3),
                "aucP_L2": _nansum(p_l2),
                "aucP_L3": _nansum(p_l3),
                "earlyP_L2": _nanmean(p_l2[:early_count]),
                "earlyP_L3": _nanmean(p_l3[:early_count]),
                "mean_sim_to_distractor_ref_L3": _nanmean(sim_to_distractor_ref_l3),
                "mean_sim_to_sample_ref_L3": _nanmean(sim_to_sample_ref_l3),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)
    timeseries_df = pd.DataFrame(time_rows).sort_values(
        ["triplet_id", "layer", "distractor_step"],
        kind="stable",
    ).reset_index(drop=True)
    if not include_timeseries:
        return summary_df
    return summary_df, timeseries_df


def compute_preprobe_fusion_metrics(
    *,
    triplet_ids: Sequence[int],
    mixed_preprobe_states: Mapping[str, Mapping[str, np.ndarray]],
    sample_only_preprobe_states: Mapping[str, Mapping[str, np.ndarray]],
    distractor_only_preprobe_states: Mapping[str, Mapping[str, np.ndarray]],
    layers: Sequence[str] = ("layer2", "layer3"),
    state_key: str = "ux",
) -> pd.DataFrame:
    rows = [{"triplet_id": int(triplet_id)} for triplet_id in triplet_ids]
    for layer_key in layers:
        layer_suffix = f"L{int(layer_key[-1])}"
        mixed_vectors = _preprobe_state_vectors(mixed_preprobe_states, layer_key=str(layer_key), state_key=state_key)
        sample_vectors = _preprobe_state_vectors(sample_only_preprobe_states, layer_key=str(layer_key), state_key=state_key)
        distractor_vectors = _preprobe_state_vectors(distractor_only_preprobe_states, layer_key=str(layer_key), state_key=state_key)
        for batch_idx in range(len(rows)):
            sim_to_sample = _centered_cosine(mixed_vectors[batch_idx], sample_vectors[batch_idx])
            sim_to_distractor = _centered_cosine(mixed_vectors[batch_idx], distractor_vectors[batch_idx])
            rows[batch_idx][f"sim_to_sample_{layer_suffix}"] = float(sim_to_sample)
            rows[batch_idx][f"sim_to_distractor_{layer_suffix}"] = float(sim_to_distractor)
            rows[batch_idx][f"fusion_dual_score_{layer_suffix}"] = float(
                _nanmean([sim_to_sample, sim_to_distractor])
            )
            rows[batch_idx][f"fusion_imbalance_{layer_suffix}"] = float(
                abs(sim_to_sample - sim_to_distractor)
            ) if np.isfinite(sim_to_sample) and np.isfinite(sim_to_distractor) else float("nan")
    return pd.DataFrame(rows).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)


def compute_sample_induced_rewriting_metrics(
    *,
    triplet_ids: Sequence[int],
    intact_capture: RolloutCapture,
    removed_capture: RolloutCapture,
    distractor_only_capture: RolloutCapture,
    layers: Sequence[str] = ("layer3",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for batch_idx, triplet_id in enumerate(triplet_ids):
        row: dict[str, object] = {"triplet_id": int(triplet_id)}
        for layer_key in layers:
            if str(layer_key) == "layer2":
                intact_trace = intact_capture.distractor_l2_trace[batch_idx]
                removed_trace = removed_capture.distractor_l2_trace[batch_idx]
                donly_trace = distractor_only_capture.distractor_l2_trace[batch_idx]
            else:
                intact_trace = intact_capture.distractor_l3_trace[batch_idx]
                removed_trace = removed_capture.distractor_l3_trace[batch_idx]
                donly_trace = distractor_only_capture.distractor_l3_trace[batch_idx]
            layer_suffix = f"L{int(str(layer_key)[-1])}"
            rewriting_values = []
            intact_rewrite_values = []
            removed_rewrite_values = []
            for t_step in range(intact_trace.shape[0]):
                sim_removed_to_donly_t = _centered_cosine(removed_trace[t_step], donly_trace[t_step])
                sim_intact_to_donly_t = _centered_cosine(intact_trace[t_step], donly_trace[t_step])
                intact_rewrite_t = (1.0 - sim_intact_to_donly_t) if np.isfinite(sim_intact_to_donly_t) else float("nan")
                removed_rewrite_t = (1.0 - sim_removed_to_donly_t) if np.isfinite(sim_removed_to_donly_t) else float("nan")
                rewriting_t = (
                    sim_removed_to_donly_t - sim_intact_to_donly_t
                    if np.isfinite(sim_removed_to_donly_t) and np.isfinite(sim_intact_to_donly_t)
                    else float("nan")
                )
                rewriting_values.append(rewriting_t)
                intact_rewrite_values.append(intact_rewrite_t)
                removed_rewrite_values.append(removed_rewrite_t)
                time_rows.append(
                    {
                        "triplet_id": int(triplet_id),
                        "layer": str(layer_key),
                        "distractor_step": int(t_step),
                        "rewriting_t": float(rewriting_t),
                        "sim_removed_to_donly_t": float(sim_removed_to_donly_t),
                        "sim_intact_to_donly_t": float(sim_intact_to_donly_t),
                        "rewrite_intact_t": float(intact_rewrite_t),
                        "rewrite_removed_t": float(removed_rewrite_t),
                    }
                )
            early_count = max(1, int(math.ceil(max(len(rewriting_values), 1) * 0.25)))
            row[f"rewrite_mean_{layer_suffix}"] = _nanmean(rewriting_values)
            row[f"rewrite_peak_{layer_suffix}"] = _nanmax(rewriting_values)
            row[f"rewrite_auc_{layer_suffix}"] = _nansum(rewriting_values)
            row[f"rewrite_early_{layer_suffix}"] = _nanmean(rewriting_values[:early_count])
            row[f"rewrite_mean_{layer_suffix}_intact"] = _nanmean(intact_rewrite_values)
            row[f"rewrite_peak_{layer_suffix}_intact"] = _nanmax(intact_rewrite_values)
            row[f"rewrite_auc_{layer_suffix}_intact"] = _nansum(intact_rewrite_values)
            row[f"rewrite_early_{layer_suffix}_intact"] = _nanmean(intact_rewrite_values[:early_count])
            row[f"rewrite_mean_{layer_suffix}_removed"] = _nanmean(removed_rewrite_values)
            row[f"rewrite_peak_{layer_suffix}_removed"] = _nanmax(removed_rewrite_values)
            row[f"rewrite_auc_{layer_suffix}_removed"] = _nansum(removed_rewrite_values)
            row[f"rewrite_early_{layer_suffix}_removed"] = _nanmean(removed_rewrite_values[:early_count])
            row[f"delta_rewrite_mean_{layer_suffix}"] = row[f"rewrite_mean_{layer_suffix}_removed"] - row[f"rewrite_mean_{layer_suffix}_intact"]
            row[f"delta_rewrite_peak_{layer_suffix}"] = row[f"rewrite_peak_{layer_suffix}_removed"] - row[f"rewrite_peak_{layer_suffix}_intact"]
            row[f"delta_rewrite_auc_{layer_suffix}"] = row[f"rewrite_auc_{layer_suffix}_removed"] - row[f"rewrite_auc_{layer_suffix}_intact"]
            row[f"delta_rewrite_early_{layer_suffix}"] = row[f"rewrite_early_{layer_suffix}_removed"] - row[f"rewrite_early_{layer_suffix}_intact"]
        summary_rows.append(row)
    return (
        pd.DataFrame(time_rows).sort_values(["triplet_id", "layer", "distractor_step"], kind="stable").reset_index(drop=True),
        pd.DataFrame(summary_rows).sort_values(["triplet_id"], kind="stable").reset_index(drop=True),
    )


def compute_fusion_specificity_metrics(
    *,
    triplet_ids: Sequence[int],
    mixed_preprobe_states: Mapping[str, Mapping[str, np.ndarray]],
    sample_only_preprobe_states: Mapping[str, Mapping[str, np.ndarray]],
    distractor_only_preprobe_states: Mapping[str, Mapping[str, np.ndarray]],
    layers: Sequence[str] = ("layer2", "layer3"),
    state_key: str = "ux",
) -> pd.DataFrame:
    rows = [{"triplet_id": int(triplet_id)} for triplet_id in triplet_ids]
    for layer_key in layers:
        layer_suffix = f"L{int(layer_key[-1])}"
        mixed_vectors = _preprobe_state_vectors(mixed_preprobe_states, layer_key=str(layer_key), state_key=state_key)
        sample_vectors = _preprobe_state_vectors(sample_only_preprobe_states, layer_key=str(layer_key), state_key=state_key)
        distractor_vectors = _preprobe_state_vectors(distractor_only_preprobe_states, layer_key=str(layer_key), state_key=state_key)
        sim_to_samples = _pairwise_centered_cosine_matrix(mixed_vectors, sample_vectors)
        sim_to_distractors = _pairwise_centered_cosine_matrix(mixed_vectors, distractor_vectors)
        n_triplets = len(rows)
        for batch_idx in range(n_triplets):
            pair_scores = 0.5 * (
                sim_to_samples[batch_idx][:, None] + sim_to_distractors[batch_idx][None, :]
            )
            flat_scores = pair_scores.reshape(-1)
            true_index = int(batch_idx * n_triplets + batch_idx)
            true_score = float(flat_scores[true_index]) if true_index < flat_scores.size else float("nan")
            null_scores = np.delete(flat_scores, true_index)
            finite_all = flat_scores[np.isfinite(flat_scores)]
            finite_null = null_scores[np.isfinite(null_scores)]
            if not np.isfinite(true_score) or finite_all.size <= 0:
                rank = float("nan")
                percentile = float("nan")
                z_score = float("nan")
                top1 = 0
            else:
                rank = 1 + int(np.sum(finite_all > true_score))
                percentile = float(np.mean(finite_all <= true_score))
                if finite_null.size <= 1 or float(np.std(finite_null, ddof=0)) <= DEFAULT_EPS:
                    z_score = float("nan")
                else:
                    z_score = float((true_score - float(finite_null.mean())) / float(finite_null.std(ddof=0)))
                top1 = int(rank == 1)
            rows[batch_idx][f"true_pair_score_{layer_suffix}"] = true_score
            rows[batch_idx][f"true_pair_rank_{layer_suffix}"] = rank
            rows[batch_idx][f"true_pair_percentile_{layer_suffix}"] = percentile
            rows[batch_idx][f"true_pair_z_{layer_suffix}"] = z_score
            rows[batch_idx][f"true_pair_top1_{layer_suffix}"] = top1
    return pd.DataFrame(rows).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)


def compute_rewriting_to_fusion_bridge(
    *,
    distractor_pull_summary: pd.DataFrame,
    preprobe_fusion_metrics: pd.DataFrame,
    fusion_specificity_metrics: pd.DataFrame,
    triplet_metadata: pd.DataFrame,
) -> pd.DataFrame:
    analysis_df = (
        distractor_pull_summary.merge(preprobe_fusion_metrics, on="triplet_id", how="inner")
        .merge(fusion_specificity_metrics, on="triplet_id", how="inner")
        .merge(
            triplet_metadata[["triplet_id", "sp_similarity", "dp_similarity", "sd_similarity"]],
            on="triplet_id",
            how="left",
        )
    )
    rows = [
        _correlation_record(
            table_name="rewriting_to_fusion_bridge",
            x_name="barP_L3",
            y_name="fusion_dual_score_L3",
            x=analysis_df["barP_L3"],
            y=analysis_df["fusion_dual_score_L3"],
        ),
        _correlation_record(
            table_name="rewriting_to_fusion_bridge",
            x_name="barP_L3",
            y_name="true_pair_z_L3",
            x=analysis_df["barP_L3"],
            y=analysis_df["true_pair_z_L3"],
        ),
    ]
    rows.extend(
        _ols_result_to_rows(
            _fit_standardized_ols(
                analysis_df,
                response="fusion_dual_score_L3",
                predictors=("barP_L3", "sp_similarity", "dp_similarity", "sd_similarity"),
                model_name="fusion_dual_score_L3 ~ barP_L3 + sp_similarity + dp_similarity + sd_similarity",
            ),
            table_name="rewriting_to_fusion_bridge",
        )
    )
    rows.extend(
        _ols_result_to_rows(
            _fit_standardized_ols(
                analysis_df,
                response="true_pair_z_L3",
                predictors=("barP_L3", "sp_similarity", "dp_similarity", "sd_similarity"),
                model_name="true_pair_z_L3 ~ barP_L3 + sp_similarity + dp_similarity + sd_similarity",
            ),
            table_name="rewriting_to_fusion_bridge",
        )
    )
    return pd.DataFrame(rows)


def compute_holistic_metrics(
    *,
    triplet_ids: Sequence[int],
    full_capture: RolloutCapture,
    distractor_reference_capture: RolloutCapture,
    cue_captures: Mapping[str, RolloutCapture],
    cue_winners: Mapping[str, pd.DataFrame],
    winner_window_frac: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    window = _winner_window_indices(full_capture.probe_l3_trace.shape[1], frac=float(winner_window_frac))
    for batch_idx, triplet_id in enumerate(triplet_ids):
        row: dict[str, object] = {"triplet_id": int(triplet_id)}
        for cue_name in CUE_CONDITIONS:
            cue_capture = cue_captures[cue_name]
            h_full = []
            h_adv = []
            for t_step in window.tolist():
                cue_vec = cue_capture.probe_l3_trace[batch_idx, t_step]
                full_vec = full_capture.probe_l3_trace[batch_idx, t_step]
                dref_vec = distractor_reference_capture.probe_l3_trace[batch_idx, t_step]
                sim_full = _centered_cosine(cue_vec, full_vec)
                sim_dref = _centered_cosine(cue_vec, dref_vec)
                h_full.append(sim_full)
                h_adv.append(sim_full - sim_dref if np.isfinite(sim_full) and np.isfinite(sim_dref) else float("nan"))
            suffix = cue_name.replace("cue_", "")
            cue_winner_row = cue_winners[cue_name].loc[cue_winners[cue_name]["triplet_id"] == int(triplet_id)].iloc[0]
            row[f"H_full_{suffix}"] = _nanmean(h_full)
            row[f"H_adv_{suffix}"] = _nanmean(h_adv)
            row[f"W_probe_{cue_name}"] = float(cue_winner_row["W_probe"])
            row[f"winner_label_{cue_name}"] = str(cue_winner_row["winner_label"])
        row["H_adv_mean"] = _nanmean([row["H_adv_SP"], row["H_adv_DP"], row["H_adv_SDP"]])
        rows.append(row)
    return pd.DataFrame(rows)


def _build_layer1_summary(support_df: pd.DataFrame, *, condition_name: str) -> pd.DataFrame:
    subset = support_df[(support_df["condition"] == condition_name) & (support_df["layer"] == "layer1")].copy()
    rows: list[dict[str, object]] = []
    for triplet_id, group in subset.groupby("triplet_id", sort=True):
        by_region = {str(row.region): row for row in group.itertuples(index=False)}
        u_sp = float(by_region.get("SP").support_mass) if "SP" in by_region else 0.0
        u_dp = float(by_region.get("DP").support_mass) if "DP" in by_region else 0.0
        u_sdp = float(by_region.get("SDP").support_mass) if "SDP" in by_region else 0.0
        rows.append(
            {
                "triplet_id": int(triplet_id),
                "condition": str(condition_name),
                "U_SP_L1": u_sp,
                "U_DP_L1": u_dp,
                "U_SDP_L1": u_sdp,
                "U_total_L1": float(u_sp + u_dp + u_sdp),
                "mean_ux_SP_L1": float(by_region.get("SP").mean_ux) if "SP" in by_region else float("nan"),
                "mean_ux_DP_L1": float(by_region.get("DP").mean_ux) if "DP" in by_region else float("nan"),
                "mean_ux_SDP_L1": float(by_region.get("SDP").mean_ux) if "SDP" in by_region else float("nan"),
                "PI1_L1": float(u_sp - u_dp),
                "PI2_L1": float(u_sdp - u_dp),
                "PI3_L1": float(u_sp + u_sdp - u_dp),
            }
        )
    return pd.DataFrame(rows).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)


def _correlation_record(
    *,
    table_name: str,
    x_name: str,
    y_name: str,
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
) -> dict[str, object]:
    xx, yy = _finite_xy(x, y)
    if xx.size < 3 or float(np.std(xx)) <= DEFAULT_EPS or float(np.std(yy)) <= DEFAULT_EPS:
        return {"table": str(table_name), "analysis": "correlation", "x": str(x_name), "y": str(y_name), "n": int(xx.size), "pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None, "slope": None, "intercept": None, "r2": None, "status": "insufficient_samples_or_variance"}
    pearson = stats.pearsonr(xx, yy)
    spearman = stats.spearmanr(xx, yy)
    lin = stats.linregress(xx, yy)
    return {"table": str(table_name), "analysis": "correlation", "x": str(x_name), "y": str(y_name), "n": int(xx.size), "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue), "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue), "slope": float(lin.slope), "intercept": float(lin.intercept), "r2": float(lin.rvalue**2), "status": "ok"}


def _fit_standardized_ols(df: pd.DataFrame, *, response: str, predictors: Sequence[str], model_name: str) -> OLSFitResult:
    clean = df[[response, *predictors]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n = int(len(clean))
    if n <= int(len(predictors)) + 1:
        return OLSFitResult(model_name=str(model_name), response=str(response), predictors=tuple(str(name) for name in predictors), n=n, r2=None, coefficients=tuple(), status="insufficient_samples", note="Not enough complete cases for OLS.")
    y = _zscore(clean[response].to_numpy(dtype=np.float64))
    x = np.column_stack([_zscore(clean[name].to_numpy(dtype=np.float64)) for name in predictors])
    design = np.column_stack([np.ones(n, dtype=np.float64), x])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - (design @ beta)
    df_resid = max(n - design.shape[1], 1)
    sigma2 = float(np.sum(resid**2) / df_resid)
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    tcrit = float(stats.t.ppf(0.975, df_resid))
    coeffs = tuple(
        OLSCoefficient(
            predictor=str(predictor),
            beta=float(beta[idx]),
            se=float(se[idx]),
            ci_low=float(beta[idx] - tcrit * se[idx]),
            ci_high=float(beta[idx] + tcrit * se[idx]),
            p_value=float(2.0 * stats.t.sf(abs(beta[idx] / max(float(se[idx]), DEFAULT_EPS)), df_resid)),
        )
        for idx, predictor in enumerate(predictors, start=1)
    )
    tss = float(np.sum((y - float(y.mean())) ** 2))
    rss = float(np.sum(resid**2))
    return OLSFitResult(model_name=str(model_name), response=str(response), predictors=tuple(str(name) for name in predictors), n=n, r2=(1.0 - rss / tss) if tss > DEFAULT_EPS else None, coefficients=coeffs, status="ok")


def _ols_result_to_rows(fit: OLSFitResult, *, table_name: str) -> list[dict[str, object]]:
    if fit.status != "ok":
        return [{"table": str(table_name), "analysis": "standardized_ols", "model_name": str(fit.model_name), "response": str(fit.response), "predictor": None, "n": int(fit.n), "beta": None, "se": None, "ci_low": None, "ci_high": None, "p_value": None, "r2": fit.r2, "status": str(fit.status), "note": fit.note}]
    return [{"table": str(table_name), "analysis": "standardized_ols", "model_name": str(fit.model_name), "response": str(fit.response), "predictor": str(coeff.predictor), "n": int(fit.n), "beta": coeff.beta, "se": coeff.se, "ci_low": coeff.ci_low, "ci_high": coeff.ci_high, "p_value": coeff.p_value, "r2": fit.r2, "status": "ok", "note": fit.note} for coeff in fit.coefficients]


def _flatten_intervention_record_rows(*, triplet_ids: Sequence[int], condition_name: str, record: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not record:
        return rows
    phase_records: list[tuple[str, Mapping[str, object]]] = []
    if "layers" in record and isinstance(record["layers"], Mapping):
        phase_records.append((str(record.get("stage", "intervention")), record))
    else:
        for phase_name, phase_record in record.items():
            if isinstance(phase_record, Mapping) and "layers" in phase_record and isinstance(phase_record["layers"], Mapping):
                phase_records.append((str(phase_name), phase_record))
    for phase_name, phase_record in phase_records:
        for layer_key, payload in phase_record["layers"].items():
            if not isinstance(payload, Mapping):
                continue
            for batch_idx, triplet_id in enumerate(triplet_ids):
                row = {
                    "triplet_id": int(triplet_id),
                    "condition": str(condition_name),
                    "phase": str(phase_name),
                    "layer": str(layer_key),
                    "intervention_type": str(phase_record.get("type")),
                }
                for metric_key, metric_values in payload.items():
                    values = np.asarray(metric_values, dtype=np.float64)
                    row[str(metric_key)] = float(values[batch_idx]) if batch_idx < values.size else float("nan")
                rows.append(row)
    return rows


def _summarize_conditions(winner_df: pd.DataFrame, intervention_df: pd.DataFrame) -> pd.DataFrame:
    merged = winner_df.copy()
    if not intervention_df.empty:
        merged = merged.merge(intervention_df[["triplet_id", "condition", "H_adv_vs_full"]].drop_duplicates(), on=["triplet_id", "condition"], how="left")
    rows: list[dict[str, object]] = []
    for condition_name, group in merged.groupby("condition", sort=True):
        rows.append({"condition": str(condition_name), "n": int(len(group)), "W_probe_mean": float(group["W_probe"].mean()), "W_probe_sem": _sem(group["W_probe"].to_numpy(dtype=np.float64)), "sample_win_rate": float(group["sample_win"].mean()), "distractor_win_rate": float(group["distractor_win"].mean()), "tie_rate": float(group["tie"].mean()), "H_adv_vs_full_mean": float(group["H_adv_vs_full"].mean()) if "H_adv_vs_full" in group else float("nan"), "H_adv_vs_full_sem": _sem(group["H_adv_vs_full"].to_numpy(dtype=np.float64)) if "H_adv_vs_full" in group else float("nan")})
    return pd.DataFrame(rows).sort_values(["condition"], kind="stable").reset_index(drop=True)


def run_fig5_fusion_backbone_from_config(
    config: ExperimentConfig,
    *,
    device: torch.device,
    logger: Any | None = None,
) -> Fig5FusionBackboneResult:
    spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=float(config.sample_ms),
        delay1_ms=float(config.delay1_ms),
        distractor_ms=float(config.distractor_ms),
        delay2_ms=float(config.delay2_ms),
        probe_ms=float(config.probe_ms),
    )
    dataset = _load_dataset(config.dataset_root, config.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    class_index = build_class_index(dataset, num_classes=int(len(np.unique(labels))))
    net, encoder = load_model_and_encoder(
        model_path=config.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(float(config.sample_ms), float(config.distractor_ms), float(config.probe_ms)),
    )
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )
    df_triplets = build_triplet_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=int(config.max_probes),
        samples_per_probe=int(config.samples_per_probe),
        num_bins=int(config.num_sim_bins),
        max_triplets=int(config.max_triplets),
        seed=int(config.seed),
    )

    region_bundles: dict[int, ProbeRegionBundle] = {}
    triplet_export_rows: list[dict[str, object]] = []
    for row in df_triplets.itertuples(index=False):
        bundle = build_probe_region_bundle(
            images[int(row.sample_id)],
            images[int(row.distractor_id)],
            images[int(row.probe_id)],
            net=net,
            foreground_threshold=float(config.foreground_threshold),
            dilation_radius=int(config.dilation_radius),
        )
        region_bundles[int(row.triplet_id)] = bundle
        meta = dict(row._asdict())
        meta.update(bundle.metadata)
        triplet_export_rows.append(meta)
    triplets_aug = pd.DataFrame(triplet_export_rows).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)
    if "sample_only_area" not in triplets_aug.columns and "area_SP" in triplets_aug.columns:
        triplets_aug["sample_only_area"] = triplets_aug["area_SP"]
    if "distractor_only_area" not in triplets_aug.columns and "area_DP" in triplets_aug.columns:
        triplets_aug["distractor_only_area"] = triplets_aug["area_DP"]
    if "shared_area" not in triplets_aug.columns and "area_SDP" in triplets_aug.columns:
        triplets_aug["shared_area"] = triplets_aug["area_SDP"]

    preprobe_buffers: dict[str, dict[str, dict[str, object]]] = {
        "baseline_intact": {},
        "sample_only_trajectory_reference": {},
        "distractor_only_trajectory_reference": {},
        "clamp_sample_trace_before_distractor": {},
    }
    region_support_rows: list[dict[str, object]] = []
    cue_winner_tables: list[pd.DataFrame] = []
    holistic_tables: list[pd.DataFrame] = []
    pull_summary_tables: list[pd.DataFrame] = []
    pull_timeseries_tables: list[pd.DataFrame] = []
    formation_pull_tables: list[pd.DataFrame] = []
    rewriting_timeseries_tables: list[pd.DataFrame] = []
    rewriting_summary_tables: list[pd.DataFrame] = []
    first_batch_captures: dict[str, RolloutCapture] = {}
    first_batch_triplet_ids: list[int] = []
    rollout_count = 0
    total_batches = max(1, int(np.ceil(len(df_triplets) / max(int(config.batch_size), 1))))

    for batch_index, batch_start in enumerate(range(0, len(df_triplets), int(config.batch_size)), start=1):
        batch_df = df_triplets.iloc[batch_start : batch_start + int(config.batch_size)].copy().reset_index(drop=True)
        if batch_df.empty:
            continue
        triplet_ids = batch_df["triplet_id"].astype(int).tolist()
        bundles = [region_bundles[int(triplet_id)] for triplet_id in triplet_ids]
        batches = prepare_triplet_batches(images, batch_df, encoder=encoder, spec=spec, device=device)
        probe_full = batches["probe"]
        cue_batches = {
            "cue_SP": _mask_spike_batch_keep_region(probe_full, [bundle.probe_region_masks["SP"] for bundle in bundles]),
            "cue_DP": _mask_spike_batch_keep_region(probe_full, [bundle.probe_region_masks["DP"] for bundle in bundles]),
            "cue_SDP": _mask_spike_batch_keep_region(probe_full, [bundle.probe_region_masks["SDP"] for bundle in bundles]),
        }
        captures: dict[str, RolloutCapture] = {
            "baseline_intact": run_distractor_rollout_capture(
                net,
                sample_spikes=batches["sample"],
                distractor_spikes=batches["distractor"],
                probe_spikes=probe_full,
                spec=spec,
                readout_step=readout_step,
            ),
            "sample_reference": run_distractor_rollout_capture(
                net,
                sample_spikes=batches["sample"],
                distractor_spikes=batches["zero_distractor"],
                probe_spikes=probe_full,
                spec=spec,
                readout_step=readout_step,
            ),
            "distractor_reference": run_distractor_rollout_capture(
                net,
                sample_spikes=batches["distractor_as_sample"],
                distractor_spikes=batches["zero_distractor"],
                probe_spikes=probe_full,
                spec=spec,
                readout_step=readout_step,
            ),
            "distractor_only_trajectory_reference": run_distractor_rollout_capture(
                net,
                sample_spikes=batches["zero_sample"],
                distractor_spikes=batches["distractor"],
                probe_spikes=batches["zero_probe"],
                spec=spec,
                readout_step=readout_step,
            ),
            "sample_only_trajectory_reference": run_distractor_rollout_capture(
                net,
                sample_spikes=batches["sample"],
                distractor_spikes=batches["zero_distractor"],
                probe_spikes=batches["zero_probe"],
                spec=spec,
                readout_step=readout_step,
            ),
            "clamp_sample_trace_before_distractor": run_distractor_rollout_capture(
                net,
                sample_spikes=batches["sample"],
                distractor_spikes=batches["distractor"],
                probe_spikes=probe_full,
                spec=spec,
                readout_step=readout_step,
                before_distractor_fn=_build_clamp_sample_trace_before_distractor_fn(),
            ),
        }
        rollout_count += int(len(captures))
        for cue_name, cue_probe in cue_batches.items():
            captures[cue_name] = run_distractor_rollout_capture(
                net,
                sample_spikes=batches["sample"],
                distractor_spikes=batches["distractor"],
                probe_spikes=cue_probe,
                spec=spec,
                readout_step=readout_step,
            )
            rollout_count += 1

        for state_name in (
            "baseline_intact",
            "sample_only_trajectory_reference",
            "distractor_only_trajectory_reference",
            "clamp_sample_trace_before_distractor",
        ):
            _append_preprobe_state_buffer(preprobe_buffers[state_name], captures[state_name].preprobe_states)

        for condition_name in (
            "baseline_intact",
            "sample_reference",
            "distractor_reference",
            "sample_only_trajectory_reference",
            "distractor_only_trajectory_reference",
            *CUE_CONDITIONS,
        ):
            capture = captures[condition_name]
            for local_idx, triplet_id in enumerate(triplet_ids):
                single_state = {
                    layer_key: {
                        key: _slice_state_value(value, local_idx)
                        for key, value in capture.preprobe_states[layer_key].items()
                    }
                    for layer_key in LAYER_KEYS
                }
                region_support_rows.extend(
                    compute_region_support_summary(
                        triplet_id=int(triplet_id),
                        condition=str(condition_name),
                        preprobe_states=single_state,
                        bundle=bundles[local_idx],
                    )
                )

        pull_summary_df, pull_timeseries_df = compute_reshaping_metrics(
            triplet_ids=triplet_ids,
            mixed_capture=captures["baseline_intact"],
            distractor_only_capture=captures["distractor_only_trajectory_reference"],
            sample_only_capture=captures["sample_only_trajectory_reference"],
            include_timeseries=True,
        )
        pull_summary_tables.append(pull_summary_df)
        pull_timeseries_tables.append(pull_timeseries_df)
        formation_pull_tables.append(
            compute_reshaping_metrics(
                triplet_ids=triplet_ids,
                mixed_capture=captures["clamp_sample_trace_before_distractor"],
                distractor_only_capture=captures["distractor_only_trajectory_reference"],
                sample_only_capture=captures["sample_only_trajectory_reference"],
            )
        )
        rewriting_timeseries_df, rewriting_summary_df = compute_sample_induced_rewriting_metrics(
            triplet_ids=triplet_ids,
            intact_capture=captures["baseline_intact"],
            removed_capture=captures["clamp_sample_trace_before_distractor"],
            distractor_only_capture=captures["distractor_only_trajectory_reference"],
            layers=("layer3",),
        )
        rewriting_timeseries_tables.append(rewriting_timeseries_df)
        rewriting_summary_tables.append(rewriting_summary_df)

        batch_winners: dict[str, pd.DataFrame] = {}
        for condition_name in ("baseline_intact", *CUE_CONDITIONS):
            winner_df = compute_winner_metrics(
                condition_name=condition_name,
                triplet_ids=triplet_ids,
                condition_trace=captures[condition_name].probe_grouped_voltage_trace,
                sample_reference_trace=captures["sample_reference"].probe_grouped_voltage_trace,
                distractor_reference_trace=captures["distractor_reference"].probe_grouped_voltage_trace,
                winner_window_frac=float(config.winner_window_frac),
                tie_threshold=float(config.tie_threshold),
                predictions=captures[condition_name].prediction_probe,
                first_fire=captures[condition_name].first_fire_t_probe,
            )
            batch_winners[condition_name] = winner_df
            if condition_name in CUE_CONDITIONS:
                cue_winner_tables.append(winner_df)

        holistic_tables.append(
            compute_holistic_metrics(
                triplet_ids=triplet_ids,
                full_capture=captures["baseline_intact"],
                distractor_reference_capture=captures["distractor_reference"],
                cue_captures={cue_name: captures[cue_name] for cue_name in CUE_CONDITIONS},
                cue_winners={cue_name: batch_winners[cue_name] for cue_name in CUE_CONDITIONS},
                winner_window_frac=float(config.winner_window_frac),
            )
        )
        if not first_batch_captures:
            first_batch_captures = captures
            first_batch_triplet_ids = [int(triplet_id) for triplet_id in triplet_ids]
        if logger is not None:
            logger.info(
                "[Backbone] batch=%s/%s triplets=%s rollouts=%s",
                batch_index,
                total_batches,
                len(batch_df),
                len(captures),
            )

    preprobe_baseline = _finalize_preprobe_state_buffer(preprobe_buffers["baseline_intact"])
    preprobe_sample_only = _finalize_preprobe_state_buffer(preprobe_buffers["sample_only_trajectory_reference"])
    preprobe_distractor_only = _finalize_preprobe_state_buffer(preprobe_buffers["distractor_only_trajectory_reference"])
    preprobe_formation = _finalize_preprobe_state_buffer(preprobe_buffers["clamp_sample_trace_before_distractor"])

    pull_summary_df = pd.concat(pull_summary_tables, axis=0, ignore_index=True).sort_values(
        ["triplet_id"],
        kind="stable",
    ).reset_index(drop=True)
    pull_timeseries_df = pd.concat(pull_timeseries_tables, axis=0, ignore_index=True).sort_values(
        ["triplet_id", "layer", "distractor_step"],
        kind="stable",
    ).reset_index(drop=True)
    formation_pull_df = pd.concat(formation_pull_tables, axis=0, ignore_index=True).sort_values(
        ["triplet_id"],
        kind="stable",
    ).reset_index(drop=True)
    rewriting_timeseries_df = pd.concat(rewriting_timeseries_tables, axis=0, ignore_index=True).sort_values(
        ["triplet_id", "layer", "distractor_step"],
        kind="stable",
    ).reset_index(drop=True)
    rewriting_summary_df = pd.concat(rewriting_summary_tables, axis=0, ignore_index=True).sort_values(
        ["triplet_id"],
        kind="stable",
    ).reset_index(drop=True)

    fusion_metrics_df = compute_preprobe_fusion_metrics(
        triplet_ids=triplets_aug["triplet_id"].astype(int).tolist(),
        mixed_preprobe_states=preprobe_baseline,
        sample_only_preprobe_states=preprobe_sample_only,
        distractor_only_preprobe_states=preprobe_distractor_only,
    )
    specificity_df = compute_fusion_specificity_metrics(
        triplet_ids=triplets_aug["triplet_id"].astype(int).tolist(),
        mixed_preprobe_states=preprobe_baseline,
        sample_only_preprobe_states=preprobe_sample_only,
        distractor_only_preprobe_states=preprobe_distractor_only,
    )
    formation_fusion_df = compute_preprobe_fusion_metrics(
        triplet_ids=triplets_aug["triplet_id"].astype(int).tolist(),
        mixed_preprobe_states=preprobe_formation,
        sample_only_preprobe_states=preprobe_sample_only,
        distractor_only_preprobe_states=preprobe_distractor_only,
    )
    formation_specificity_df = compute_fusion_specificity_metrics(
        triplet_ids=triplets_aug["triplet_id"].astype(int).tolist(),
        mixed_preprobe_states=preprobe_formation,
        sample_only_preprobe_states=preprobe_sample_only,
        distractor_only_preprobe_states=preprobe_distractor_only,
    )
    rewriting_bridge_df = compute_rewriting_to_fusion_bridge(
        distractor_pull_summary=pull_summary_df,
        preprobe_fusion_metrics=fusion_metrics_df,
        fusion_specificity_metrics=specificity_df,
        triplet_metadata=triplets_aug,
    )

    formation_intervention_df = (
        pull_summary_df.merge(
            formation_pull_df[["triplet_id", "barP_L2", "barP_L3"]].rename(
                columns={
                    "barP_L2": "formation_barP_L2",
                    "barP_L3": "formation_barP_L3",
                }
            ),
            on="triplet_id",
            how="inner",
        )
        .merge(
            rewriting_summary_df[
                [
                    "triplet_id",
                    "rewrite_mean_L3",
                    "rewrite_peak_L3",
                    "rewrite_auc_L3",
                    "rewrite_early_L3",
                    "rewrite_mean_L3_intact",
                    "rewrite_peak_L3_intact",
                    "rewrite_auc_L3_intact",
                    "rewrite_early_L3_intact",
                    "rewrite_mean_L3_removed",
                    "rewrite_peak_L3_removed",
                    "rewrite_auc_L3_removed",
                    "rewrite_early_L3_removed",
                    "delta_rewrite_mean_L3",
                    "delta_rewrite_peak_L3",
                    "delta_rewrite_auc_L3",
                    "delta_rewrite_early_L3",
                ]
            ],
            on="triplet_id",
            how="inner",
        )
        .merge(
            fusion_metrics_df[["triplet_id", "fusion_dual_score_L2", "fusion_dual_score_L3"]],
            on="triplet_id",
            how="inner",
        )
        .merge(
            formation_fusion_df[["triplet_id", "fusion_dual_score_L2", "fusion_dual_score_L3"]].rename(
                columns={
                    "fusion_dual_score_L2": "formation_fusion_dual_score_L2",
                    "fusion_dual_score_L3": "formation_fusion_dual_score_L3",
                }
            ),
            on="triplet_id",
            how="inner",
        )
        .merge(
            specificity_df[["triplet_id", "true_pair_z_L2", "true_pair_z_L3"]],
            on="triplet_id",
            how="inner",
        )
        .merge(
            formation_specificity_df[["triplet_id", "true_pair_z_L2", "true_pair_z_L3"]].rename(
                columns={
                    "true_pair_z_L2": "formation_true_pair_z_L2",
                    "true_pair_z_L3": "formation_true_pair_z_L3",
                }
            ),
            on="triplet_id",
            how="inner",
        )
    )
    formation_intervention_df["condition"] = "clamp_sample_trace_before_distractor"
    formation_intervention_df["delta_barP_L2"] = formation_intervention_df["formation_barP_L2"] - formation_intervention_df["barP_L2"]
    formation_intervention_df["delta_barP_L3"] = formation_intervention_df["formation_barP_L3"] - formation_intervention_df["barP_L3"]
    formation_intervention_df["delta_fusion_dual_score_L2"] = (
        formation_intervention_df["formation_fusion_dual_score_L2"] - formation_intervention_df["fusion_dual_score_L2"]
    )
    formation_intervention_df["delta_fusion_dual_score_L3"] = (
        formation_intervention_df["formation_fusion_dual_score_L3"] - formation_intervention_df["fusion_dual_score_L3"]
    )
    formation_intervention_df["delta_true_pair_z_L2"] = (
        formation_intervention_df["formation_true_pair_z_L2"] - formation_intervention_df["true_pair_z_L2"]
    )
    formation_intervention_df["delta_true_pair_z_L3"] = (
        formation_intervention_df["formation_true_pair_z_L3"] - formation_intervention_df["true_pair_z_L3"]
    )
    formation_intervention_df = formation_intervention_df.sort_values(
        ["triplet_id"],
        kind="stable",
    ).reset_index(drop=True)

    region_support_df = pd.DataFrame(region_support_rows).sort_values(
        ["triplet_id", "condition", "layer", "region"],
        kind="stable",
    ).reset_index(drop=True)
    layer1_trial_metrics = _build_layer1_summary(region_support_df, condition_name="baseline_intact")
    layer1_formula_fit = pd.DataFrame(
        columns=["metric_name", "region", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "r2", "n"]
    )
    holistic_df = pd.concat(holistic_tables, axis=0, ignore_index=True).sort_values(
        ["triplet_id"],
        kind="stable",
    ).reset_index(drop=True)
    cue_winner_df = pd.concat(cue_winner_tables, axis=0, ignore_index=True).sort_values(
        ["triplet_id", "condition"],
        kind="stable",
    ).reset_index(drop=True)

    example_triplet_id = int(first_batch_triplet_ids[0]) if first_batch_triplet_ids else int(triplets_aug["triplet_id"].iloc[0])
    example_index = int(first_batch_triplet_ids.index(example_triplet_id)) if first_batch_triplet_ids else 0
    example_preprobe_fusion_state = {
        "triplet_id": np.asarray([example_triplet_id], dtype=np.int64),
        "mixed_preprobe_ux_L2": np.asarray(first_batch_captures["baseline_intact"].preprobe_states["layer2"]["ux"][example_index], dtype=np.float32),
        "mixed_preprobe_ux_L3": np.asarray(first_batch_captures["baseline_intact"].preprobe_states["layer3"]["ux"][example_index], dtype=np.float32),
        "sample_only_preprobe_ux_L2": np.asarray(first_batch_captures["sample_only_trajectory_reference"].preprobe_states["layer2"]["ux"][example_index], dtype=np.float32),
        "sample_only_preprobe_ux_L3": np.asarray(first_batch_captures["sample_only_trajectory_reference"].preprobe_states["layer3"]["ux"][example_index], dtype=np.float32),
        "distractor_only_preprobe_ux_L2": np.asarray(first_batch_captures["distractor_only_trajectory_reference"].preprobe_states["layer2"]["ux"][example_index], dtype=np.float32),
        "distractor_only_preprobe_ux_L3": np.asarray(first_batch_captures["distractor_only_trajectory_reference"].preprobe_states["layer3"]["ux"][example_index], dtype=np.float32),
    }
    example_pull_rows = pull_timeseries_df[pull_timeseries_df["triplet_id"] == int(example_triplet_id)].copy()
    example_distractor_pull_trace = {
        "triplet_id": np.asarray([example_triplet_id], dtype=np.int64),
        "mixed_distractor_L2_trace": np.asarray(first_batch_captures["baseline_intact"].distractor_l2_trace[example_index], dtype=np.float32),
        "mixed_distractor_L3_trace": np.asarray(first_batch_captures["baseline_intact"].distractor_l3_trace[example_index], dtype=np.float32),
        "sample_only_reference_L2_trace": np.asarray(first_batch_captures["sample_only_trajectory_reference"].distractor_l2_trace[example_index], dtype=np.float32),
        "sample_only_reference_L3_trace": np.asarray(first_batch_captures["sample_only_trajectory_reference"].distractor_l3_trace[example_index], dtype=np.float32),
        "distractor_only_reference_L2_trace": np.asarray(first_batch_captures["distractor_only_trajectory_reference"].distractor_l2_trace[example_index], dtype=np.float32),
        "distractor_only_reference_L3_trace": np.asarray(first_batch_captures["distractor_only_trajectory_reference"].distractor_l3_trace[example_index], dtype=np.float32),
        "pull_t_L2": example_pull_rows.loc[example_pull_rows["layer"] == "layer2", "pull_t"].to_numpy(dtype=np.float32),
        "pull_t_L3": example_pull_rows.loc[example_pull_rows["layer"] == "layer3", "pull_t"].to_numpy(dtype=np.float32),
    }

    return Fig5FusionBackboneResult(
        config={
            "sample_ms": float(config.sample_ms),
            "delay1_ms": float(config.delay1_ms),
            "distractor_ms": float(config.distractor_ms),
            "delay2_ms": float(config.delay2_ms),
            "probe_ms": float(config.probe_ms),
            "batch_size": int(config.batch_size),
            "max_probes": int(config.max_probes),
            "samples_per_probe": int(config.samples_per_probe),
            "max_triplets": int(config.max_triplets),
            "num_sim_bins": int(config.num_sim_bins),
            "foreground_threshold": float(config.foreground_threshold),
            "dilation_radius": int(config.dilation_radius),
            "winner_window_frac": float(config.winner_window_frac),
            "tie_threshold": float(config.tie_threshold),
            "smoke": bool(config.smoke),
            "smoke_command": SMOKE_COMMAND,
            "smoke_note": SMOKE_NOTE,
        },
        triplets=triplets_aug.copy(),
        preprobe_fusion_metrics=fusion_metrics_df,
        fusion_specificity_metrics=specificity_df,
        distractor_pull_timeseries=pull_timeseries_df,
        distractor_pull_summary=pull_summary_df,
        sample_induced_rewriting_timeseries=rewriting_timeseries_df,
        sample_induced_rewriting_summary=rewriting_summary_df,
        rewriting_fusion_bridge=rewriting_bridge_df,
        formation_intervention_metrics=formation_intervention_df,
        region_support_condition=region_support_df,
        layer1_trial_metrics=layer1_trial_metrics,
        layer1_formula_fit=layer1_formula_fit,
        holistic_metrics=holistic_df,
        cue_winner_metrics=cue_winner_df,
        example_triplet_id=example_triplet_id,
        example_preprobe_fusion_state=example_preprobe_fusion_state,
        example_distractor_pull_trace=example_distractor_pull_trace,
        stats={
            "triplet_count": int(len(triplets_aug)),
            "batch_count": int(total_batches),
            "shared_rollout_count": int(rollout_count),
            "per_batch_condition_count": int(6 + len(CUE_CONDITIONS)),
        },
    )


def _choose_example_triplet(df_metrics: pd.DataFrame) -> int:
    clean = df_metrics[["triplet_id", "barP_L3"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if clean.empty:
        return int(df_metrics["triplet_id"].iloc[0])
    target = float(clean["barP_L3"].median())
    clean["distance"] = (clean["barP_L3"] - target).abs()
    return int(clean.sort_values(["distance", "triplet_id"], kind="stable").iloc[0]["triplet_id"])


def _save_panel_a(
    captures_by_condition: Mapping[str, RolloutCapture],
    example_triplet_id: int,
    triplet_to_batch: Mapping[int, int],
    save_base: Path,
) -> dict[str, str]:
    apply_publication_style()
    idx = int(triplet_to_batch[example_triplet_id])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    mixed_trace = captures_by_condition["baseline_intact"].distractor_l3_trace[idx]
    donly_trace = captures_by_condition["distractor_only_trajectory_reference"].distractor_l3_trace[idx]
    steps = np.arange(mixed_trace.shape[0], dtype=np.int64)
    mixed_norm = np.linalg.norm(mixed_trace - mixed_trace.mean(axis=1, keepdims=True), axis=1)
    donly_norm = np.linalg.norm(donly_trace - donly_trace.mean(axis=1, keepdims=True), axis=1)
    ax.plot(steps, mixed_norm, color=VERMILION, label="mixed")
    ax.plot(steps, donly_norm, color=SKY_BLUE, label="distractor-only")
    ax.set_xlabel("Distractor step")
    ax.set_ylabel("Layer3 trace norm")
    ax.set_title("Panel A: Example distractor-phase trajectory")
    ax.legend(frameon=False)
    out = save_figure_all_formats(fig, save_base)
    plt.close(fig)
    return out


def _save_box_panel(df: pd.DataFrame, *, value_cols: Sequence[str], title: str, ylabel: str, save_base: Path) -> dict[str, str]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.boxplot([df[col].to_numpy(dtype=np.float64) for col in value_cols], tick_labels=list(value_cols), widths=0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    out = save_figure_all_formats(fig, save_base)
    plt.close(fig)
    return out


def _save_partial_cue_panel(df_holistic: pd.DataFrame, save_base: Path) -> dict[str, str]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    labels = ["H_full_SP", "H_full_DP", "H_full_SDP"]
    means = [float(df_holistic[col].mean()) for col in labels]
    sems = [_sem(df_holistic[col].to_numpy(dtype=np.float64)) for col in labels]
    ax.bar(np.arange(len(labels)), means, yerr=sems, color=[SKY_BLUE, ORANGE, BLUISH_GREEN], alpha=0.85)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(["SP", "DP", "SDP"])
    ax.set_ylabel("Mean similarity to full probe")
    ax.set_title("Panel D: Partial-cue holistic invocation")
    out = save_figure_all_formats(fig, save_base)
    plt.close(fig)
    return out


def _save_scatter_panel(
    *,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_base: Path,
) -> dict[str, str]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    xx, yy = _finite_xy(df[x_col], df[y_col])
    ax.scatter(xx, yy, s=26, color=VERMILION, alpha=0.7)
    if xx.size >= 2 and float(np.std(xx)) > DEFAULT_EPS and float(np.std(yy)) > DEFAULT_EPS:
        fit = stats.linregress(xx, yy)
        x_line = np.linspace(float(xx.min()), float(xx.max()), 100)
        ax.plot(x_line, fit.intercept + fit.slope * x_line, color=SKY_BLUE)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    out = save_figure_all_formats(fig, save_base)
    plt.close(fig)
    return out


def _save_intervention_panel(df_intervention: pd.DataFrame, save_base: Path) -> dict[str, str]:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    summary = df_intervention.groupby("condition", as_index=False)[["H_adv_vs_full", "W_probe"]].mean()
    tick_labels = [str(name).replace("redistribute_", "redist_") for name in summary["condition"].tolist()]
    axes[0].bar(np.arange(len(summary)), summary["H_adv_vs_full"], color=ORANGE, alpha=0.85)
    axes[1].bar(np.arange(len(summary)), summary["W_probe"], color=VERMILION, alpha=0.85)
    for ax in axes:
        ax.set_xticks(np.arange(len(summary)))
        ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    axes[0].set_title("Panel G: Intervention effect on H_adv")
    axes[0].set_ylabel("H_adv_vs_full")
    axes[1].set_title("Panel G: Intervention effect on W_probe")
    axes[1].set_ylabel("W_probe")
    out = save_figure_all_formats(fig, save_base)
    plt.close(fig)
    return out


def _mechanistic_interpretation(
    bridge_reshaping_rows: pd.DataFrame,
    bridge_probe_rows: pd.DataFrame,
    bridge_holistic_rows: pd.DataFrame,
) -> str:
    def _find_r(table: pd.DataFrame, x: str, y: str) -> float | None:
        subset = table[(table["analysis"] == "correlation") & (table["x"] == x) & (table["y"] == y)]
        if subset.empty:
            return None
        return _safe_float(subset.iloc[0]["pearson_r"])

    r1 = _find_r(bridge_reshaping_rows, "PI3_L1", "barP_L3")
    r2 = _find_r(bridge_probe_rows, "barP_L3", "W_probe_full")
    r3 = _find_r(bridge_holistic_rows, "PI3_L1", "H_adv_SDP")
    pieces = []
    if r1 is not None:
        pieces.append(f"PI3_L1 to barP_L3 r={r1:.3f}")
    if r2 is not None:
        pieces.append(f"barP_L3 to W_probe_full r={r2:.3f}")
    if r3 is not None:
        pieces.append(f"PI3_L1 to H_adv_SDP r={r3:.3f}")
    if not pieces:
        return "Bridge coefficients were not estimable from the analysed triplets."
    return (
        "Across analysed triplets, stronger Layer1 chunk-priority was associated with more sample-pulled "
        "high-level reshaping and more full-chunk-like partial-cue invocation; "
        + "; ".join(pieces)
        + "."
    )


def _bridge_row_lookup(df_bridge: pd.DataFrame, *, x_name: str, y_name: str) -> dict[str, object]:
    subset = df_bridge[
        (df_bridge["analysis"] == "correlation")
        & (df_bridge["x"] == str(x_name))
        & (df_bridge["y"] == str(y_name))
    ]
    if subset.empty:
        return {"pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None, "n": 0}
    row = subset.iloc[0]
    return {
        "pearson_r": _safe_float(row.get("pearson_r")),
        "pearson_p": _safe_float(row.get("pearson_p")),
        "spearman_rho": _safe_float(row.get("spearman_rho")),
        "spearman_p": _safe_float(row.get("spearman_p")),
        "n": int(row.get("n", 0)),
    }


def build_fig5_fusion_summary(
    *,
    triplets: pd.DataFrame,
    preprobe_fusion_metrics: pd.DataFrame,
    fusion_specificity_metrics: pd.DataFrame,
    distractor_pull_summary: pd.DataFrame,
    rewriting_fusion_bridge: pd.DataFrame,
    formation_intervention_metrics: pd.DataFrame,
    smoke: bool,
    sample_induced_rewriting_summary: pd.DataFrame | None = None,
) -> dict[str, object]:
    rewriting_summary = sample_induced_rewriting_summary
    if rewriting_summary is None:
        rewriting_summary = pd.DataFrame(
            {
                "rewrite_mean_L3": np.asarray([], dtype=np.float64),
                "rewrite_peak_L3": np.asarray([], dtype=np.float64),
                "rewrite_auc_L3": np.asarray([], dtype=np.float64),
                "rewrite_early_L3": np.asarray([], dtype=np.float64),
            }
        )

    def _mean_if_present(df: pd.DataFrame, column: str) -> float:
        if column not in df.columns or df.empty:
            return float("nan")
        return float(df[column].mean())

    return {
        "experiment": EXPERIMENT_NAME,
        "scientific_question": (
            "Does the higher-layer pre-probe latent STSP state form a fused sample+distractor memory, "
            "and is that fused state written by distractor-phase computation that has already been rewritten by the sample trace?"
        ),
        "primary_backbone": "distractor_chunk_holistic_invocation_experiment",
        "panel_b_fusion_form": {
            "mean_sim_to_sample_L3": float(preprobe_fusion_metrics["sim_to_sample_L3"].mean()),
            "mean_sim_to_distractor_L3": float(preprobe_fusion_metrics["sim_to_distractor_L3"].mean()),
            "mean_fusion_dual_score_L3": float(preprobe_fusion_metrics["fusion_dual_score_L3"].mean()),
            "mean_fusion_imbalance_L3": float(preprobe_fusion_metrics["fusion_imbalance_L3"].mean()),
        },
        "panel_b_specificity": {
            "mean_true_pair_percentile_L3": float(fusion_specificity_metrics["true_pair_percentile_L3"].mean()),
            "mean_true_pair_z_L3": float(fusion_specificity_metrics["true_pair_z_L3"].mean()),
            "top1_rate_L3": float(fusion_specificity_metrics["true_pair_top1_L3"].mean()),
        },
        "panel_c_rewriting": {
            "mean_rewrite_mean_L3": _mean_if_present(rewriting_summary, "rewrite_mean_L3"),
            "mean_rewrite_peak_L3": _mean_if_present(rewriting_summary, "rewrite_peak_L3"),
            "mean_rewrite_auc_L3": _mean_if_present(rewriting_summary, "rewrite_auc_L3"),
            "mean_rewrite_early_L3": _mean_if_present(rewriting_summary, "rewrite_early_L3"),
            "mean_barP_L2": float(distractor_pull_summary["barP_L2"].mean()),
            "mean_barP_L3": float(distractor_pull_summary["barP_L3"].mean()),
            "mean_peakP_L3": float(distractor_pull_summary["peakP_L3"].mean()),
            "mean_earlyP_L3": float(distractor_pull_summary["earlyP_L3"].mean()),
            "mean_sim_to_distractor_ref_L3": float(distractor_pull_summary["mean_sim_to_distractor_ref_L3"].mean()),
            "mean_sim_to_sample_ref_L3": float(distractor_pull_summary["mean_sim_to_sample_ref_L3"].mean()),
        },
        "panel_d_bridge": {
            "barP_L3_to_fusion_dual_score_L3": _bridge_row_lookup(
                rewriting_fusion_bridge,
                x_name="barP_L3",
                y_name="fusion_dual_score_L3",
            ),
            "barP_L3_to_true_pair_z_L3": _bridge_row_lookup(
                rewriting_fusion_bridge,
                x_name="barP_L3",
                y_name="true_pair_z_L3",
            ),
        },
        "panel_e_intervention": {
            "mean_rewrite_early_L3_intact": float(formation_intervention_metrics["rewrite_early_L3_intact"].mean()),
            "mean_rewrite_early_L3_removed": float(formation_intervention_metrics["rewrite_early_L3_removed"].mean()),
            "mean_delta_rewrite_early_L3": float(formation_intervention_metrics["delta_rewrite_early_L3"].mean()),
            "mean_delta_barP_L3": float(formation_intervention_metrics["delta_barP_L3"].mean()),
            "mean_delta_fusion_dual_score_L3": float(formation_intervention_metrics["delta_fusion_dual_score_L3"].mean()),
            "mean_delta_true_pair_z_L3": float(formation_intervention_metrics["delta_true_pair_z_L3"].mean()),
            "mean_barP_L3_intact": float(formation_intervention_metrics["barP_L3"].mean()),
            "mean_barP_L3_removed": float(formation_intervention_metrics["formation_barP_L3"].mean()),
            "mean_fusion_dual_score_L3_intact": float(formation_intervention_metrics["fusion_dual_score_L3"].mean()),
            "mean_fusion_dual_score_L3_removed": float(formation_intervention_metrics["formation_fusion_dual_score_L3"].mean()),
            "mean_true_pair_z_L3_intact": float(formation_intervention_metrics["true_pair_z_L3"].mean()),
            "mean_true_pair_z_L3_removed": float(formation_intervention_metrics["formation_true_pair_z_L3"].mean()),
        },
        "triplet_count": int(len(triplets)),
        "smoke": {
            "enabled": bool(smoke),
            "command": SMOKE_COMMAND,
            "note": SMOKE_NOTE,
        },
        "supplement_note": (
            "Layer1 region-support and holistic retrieval exports remain available as supplementary outputs only "
            "and do not define the main Fig.5 conclusion."
        ),
    }


def _save_triplet_definition_panel(
    *,
    images: torch.Tensor,
    df_triplets: pd.DataFrame,
    example_triplet_id: int,
    threshold: float,
    save_base: Path,
) -> dict[str, str]:
    apply_publication_style()
    triplet = df_triplets.loc[df_triplets["triplet_id"] == int(example_triplet_id)].iloc[0]
    sample_img = np.asarray(images[int(triplet["sample_id"])].detach().cpu().numpy()).squeeze()
    distractor_img = np.asarray(images[int(triplet["distractor_id"])].detach().cpu().numpy()).squeeze()
    probe_img = np.asarray(images[int(triplet["probe_id"])].detach().cpu().numpy()).squeeze()
    sample_mask = (sample_img > float(threshold)).astype(np.int64)
    distractor_mask = (distractor_img > float(threshold)).astype(np.int64)
    region_map = np.zeros_like(sample_mask, dtype=np.int64)
    region_map[(sample_mask == 1) & (distractor_mask == 0)] = 1
    region_map[(sample_mask == 0) & (distractor_mask == 1)] = 2
    region_map[(sample_mask == 1) & (distractor_mask == 1)] = 3
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.0))
    panels = [
        ("Sample", sample_img, "gray"),
        ("Distractor", distractor_img, "gray"),
        ("Probe", probe_img, "gray"),
        ("Regions", region_map, "viridis"),
    ]
    for ax, (title, panel, cmap) in zip(axes, panels):
        ax.imshow(panel, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Panel A: triplet definition")
    fig.tight_layout()
    out = save_figure_all_formats(fig, save_base)
    plt.close(fig)
    return out


def _save_fusion_form_panel(
    *,
    df_fusion: pd.DataFrame,
    df_specificity: pd.DataFrame,
    save_base: Path,
) -> dict[str, str]:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    xx, yy = _finite_xy(df_fusion["sim_to_sample_L3"], df_fusion["sim_to_distractor_L3"])
    axes[0].scatter(xx, yy, s=28, color=VERMILION, alpha=0.72)
    lims = [float(np.nanmin(np.concatenate([xx, yy]))), float(np.nanmax(np.concatenate([xx, yy])))] if xx.size and yy.size else [-1.0, 1.0]
    axes[0].plot(lims, lims, color=SKY_BLUE, linewidth=1.0, linestyle="--")
    axes[0].set_xlabel("sim_to_sample_L3")
    axes[0].set_ylabel("sim_to_distractor_L3")
    axes[0].set_title("B1: fused latent form")
    percentile = df_specificity["true_pair_percentile_L3"].to_numpy(dtype=np.float64)
    z_scores = df_specificity["true_pair_z_L3"].to_numpy(dtype=np.float64)
    bins = np.linspace(0.0, 1.0, 16)
    axes[1].hist(percentile[np.isfinite(percentile)], bins=bins, color=BLUISH_GREEN, alpha=0.65, label="percentile")
    ax2 = axes[1].twinx()
    ax2.hist(z_scores[np.isfinite(z_scores)], bins=16, histtype="step", color=ORANGE, linewidth=1.6, label="z-score")
    axes[1].set_xlabel("true-pair percentile / z")
    axes[1].set_ylabel("count")
    ax2.set_ylabel("z-score count")
    axes[1].set_title("B2: fusion specificity")
    out = save_figure_all_formats(fig, save_base)
    plt.close(fig)
    return out


def _save_rewriting_panel(
    *,
    df_rewriting_timeseries: pd.DataFrame,
    df_rewriting_summary: pd.DataFrame,
    save_base: Path,
) -> dict[str, str]:
    apply_publication_style()
    del df_rewriting_summary
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    layer_df = df_rewriting_timeseries[df_rewriting_timeseries["layer"] == "layer3"].copy()
    mean_df = layer_df.groupby("distractor_step", as_index=False)["rewriting_t"].mean()
    sem_df = layer_df.groupby("distractor_step", as_index=False)["rewriting_t"].agg(_sem).rename(columns={"rewriting_t": "sem"})
    merged = mean_df.merge(sem_df, on="distractor_step", how="left")
    x = merged["distractor_step"].to_numpy(dtype=np.float64)
    y = merged["rewriting_t"].to_numpy(dtype=np.float64)
    sem = merged["sem"].to_numpy(dtype=np.float64)
    ax.plot(x, y, color=VERMILION, linewidth=1.8)
    ax.fill_between(x, y - sem, y + sem, color=VERMILION, alpha=0.18)
    ax.axhline(0.0, color="#4B5563", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Distractor step")
    ax.set_ylabel("Sample-driven rewriting index")
    ax.set_title("Panel C: Sample-driven rewriting of L3 distractor activity")
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    out = save_figure_all_formats(fig, save_base)
    plt.close(fig)
    return out


def _save_rewriting_bridge_panel(
    *,
    df_pull_summary: pd.DataFrame,
    df_fusion: pd.DataFrame,
    df_specificity: pd.DataFrame,
    save_base: Path,
) -> dict[str, str]:
    apply_publication_style()
    merged = df_pull_summary.merge(df_fusion, on="triplet_id", how="inner").merge(
        df_specificity,
        on="triplet_id",
        how="inner",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for ax, y_col, title, color in (
        (axes[0], "fusion_dual_score_L3", "D1: barP_L3 vs fusion_dual_score_L3", VERMILION),
        (axes[1], "true_pair_z_L3", "D2: barP_L3 vs true_pair_z_L3", BLUISH_GREEN),
    ):
        xx, yy = _finite_xy(merged["barP_L3"], merged[y_col])
        ax.scatter(xx, yy, s=26, color=color, alpha=0.72)
        if xx.size >= 2 and float(np.std(xx)) > DEFAULT_EPS and float(np.std(yy)) > DEFAULT_EPS:
            fit = stats.linregress(xx, yy)
            x_line = np.linspace(float(xx.min()), float(xx.max()), 100)
            ax.plot(x_line, fit.intercept + fit.slope * x_line, color=SKY_BLUE, linewidth=1.2)
        ax.set_xlabel("barP_L3")
        ax.set_ylabel(y_col)
        ax.set_title(title)
    out = save_figure_all_formats(fig, save_base)
    plt.close(fig)
    return out


def _save_formation_panel(df_intervention: pd.DataFrame, save_base: Path) -> dict[str, str]:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    fig.suptitle("Panel E: Removing the sample trace weakens rewriting and fused memory", y=1.02)
    metrics = [
        (
            "rewrite_early_L3_intact",
            "rewrite_early_L3_removed",
            "E1. Removing sample trace weakens distractor-period rewriting",
            "Early sample-driven rewriting",
            VERMILION,
        ),
        (
            "fusion_dual_score_L3",
            "formation_fusion_dual_score_L3",
            "E2. Removing sample trace weakens the fused pre-probe state",
            "Similarity to both sample and distractor memories",
            SKY_BLUE,
        ),
        (
            "true_pair_z_L3",
            "formation_true_pair_z_L3",
            "E3. Removing sample trace weakens true-pair specificity",
            "Specificity for the correct sample-distractor pairing (z-score)",
            ORANGE,
        ),
    ]
    x_positions = np.asarray([0.0, 1.0], dtype=np.float64)
    for ax, (intact_col, removed_col, title, ylabel, color) in zip(axes, metrics):
        intact_vals = df_intervention[intact_col].to_numpy(dtype=np.float64)
        removed_vals = df_intervention[removed_col].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(intact_vals) & np.isfinite(removed_vals)
        intact_vals = intact_vals[finite_mask]
        removed_vals = removed_vals[finite_mask]
        for intact_value, removed_value in zip(intact_vals.tolist(), removed_vals.tolist()):
            ax.plot(x_positions, [intact_value, removed_value], color="#C7CDD4", linewidth=0.8, alpha=0.5, zorder=1)
        ax.scatter(
            np.full(intact_vals.shape, x_positions[0], dtype=np.float64),
            intact_vals,
            s=18,
            color="#4B5563",
            alpha=0.5,
            zorder=2,
        )
        ax.scatter(
            np.full(removed_vals.shape, x_positions[1], dtype=np.float64),
            removed_vals,
            s=18,
            color=color,
            alpha=0.7,
            zorder=2,
        )
        ax.errorbar(
            [x_positions[0]],
            [float(np.nanmean(intact_vals)) if intact_vals.size else float("nan")],
            yerr=[_sem(intact_vals)],
            fmt="o",
            markersize=7.0,
            linewidth=1.4,
            capsize=3.0,
            color="#4B5563",
            zorder=4,
        )
        ax.errorbar(
            [x_positions[1]],
            [float(np.nanmean(removed_vals)) if removed_vals.size else float("nan")],
            yerr=[_sem(removed_vals)],
            fmt="o",
            markersize=7.0,
            linewidth=1.4,
            capsize=3.0,
            color=color,
            zorder=4,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["intact", "sample-trace removed"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    out = save_figure_all_formats(fig, save_base)
    plt.close(fig)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    config = _parse_args(argv)
    _validate_config(config)
    seed_everything(int(config.seed))

    layout = prepare_result_layout(config.output_dir)
    log_lines: list[str] = [f"experiment={EXPERIMENT_NAME}", f"seed={int(config.seed)}"]
    device = _resolve_device(config.device, log_lines)
    log_lines.append(f"device={device}")
    backbone = run_fig5_fusion_backbone_from_config(config, device=device)
    dataset = _load_dataset(config.dataset_root, config.split)
    images, _, _ = build_dataset_arrays(dataset)

    triplet_csv = save_tidy_csv(backbone.triplets, layout.data_file("panel_a_triplet_definition.csv"), sort_by=["triplet_id"])
    fusion_csv = save_tidy_csv(backbone.preprobe_fusion_metrics, layout.data_file("panel_b_preprobe_fusion_metrics.csv"), sort_by=["triplet_id"])
    specificity_csv = save_tidy_csv(backbone.fusion_specificity_metrics, layout.data_file("panel_b_fusion_specificity.csv"), sort_by=["triplet_id"])
    pull_time_csv = save_tidy_csv(backbone.distractor_pull_timeseries, layout.data_file("panel_c_distractor_pull_timeseries.csv"), sort_by=["triplet_id", "layer", "distractor_step"])
    pull_summary_csv = save_tidy_csv(backbone.distractor_pull_summary, layout.data_file("panel_c_distractor_pull_summary.csv"), sort_by=["triplet_id"])
    rewriting_time_csv = save_tidy_csv(backbone.sample_induced_rewriting_timeseries, layout.data_file("panel_c_sample_induced_rewriting_timeseries.csv"), sort_by=["triplet_id", "layer", "distractor_step"])
    rewriting_summary_csv = save_tidy_csv(backbone.sample_induced_rewriting_summary, layout.data_file("panel_c_sample_induced_rewriting_summary.csv"), sort_by=["triplet_id"])
    bridge_csv = save_tidy_csv(backbone.rewriting_fusion_bridge, layout.data_file("panel_d_rewriting_to_fusion_bridge.csv"))
    formation_csv = save_tidy_csv(backbone.formation_intervention_metrics, layout.data_file("panel_e_formation_intervention.csv"), sort_by=["triplet_id"])
    formation_compare_csv = save_tidy_csv(backbone.formation_intervention_metrics, layout.data_file("panel_e_formation_intervention_comparison.csv"), sort_by=["triplet_id"])
    supp_region_csv = save_tidy_csv(backbone.region_support_condition, layout.data_file("supp_region_support_condition.csv"), sort_by=["triplet_id", "condition", "layer", "region"])
    supp_layer1_csv = save_tidy_csv(backbone.layer1_trial_metrics, layout.data_file("supp_layer1_trial_metrics.csv"), sort_by=["triplet_id"])
    supp_layer1_fit_csv = save_tidy_csv(backbone.layer1_formula_fit, layout.data_file("supp_layer1_formula_fit.csv"))
    supp_holistic_csv = save_tidy_csv(backbone.holistic_metrics, layout.data_file("supp_holistic_metrics.csv"), sort_by=["triplet_id"])
    supp_cue_csv = save_tidy_csv(backbone.cue_winner_metrics, layout.data_file("supp_cue_winner_metrics.csv"), sort_by=["triplet_id", "condition"])
    example_preprobe_npz = _save_npz_payload(
        layout.root_file("panel_b_example_preprobe_fusion_state.npz"),
        **backbone.example_preprobe_fusion_state,
    )
    example_pull_npz = _save_npz_payload(
        layout.root_file("panel_c_example_distractor_pull_trace.npz"),
        **backbone.example_distractor_pull_trace,
    )

    figure_paths: dict[str, object] = {}
    if not config.skip_figures:
        figure_paths["panel_a"] = _save_triplet_definition_panel(
            images=images,
            df_triplets=backbone.triplets,
            example_triplet_id=int(backbone.example_triplet_id),
            threshold=float(config.foreground_threshold),
            save_base=layout.figure_base("panel_a_triplet_definition"),
        )
        figure_paths["panel_b"] = _save_fusion_form_panel(
            df_fusion=backbone.preprobe_fusion_metrics,
            df_specificity=backbone.fusion_specificity_metrics,
            save_base=layout.figure_base("panel_b_fusion_form_and_specificity"),
        )
        figure_paths["panel_c"] = _save_rewriting_panel(
            df_rewriting_timeseries=backbone.sample_induced_rewriting_timeseries,
            df_rewriting_summary=backbone.sample_induced_rewriting_summary,
            save_base=layout.figure_base("panel_c_distractor_rewriting"),
        )
        figure_paths["panel_d"] = _save_rewriting_bridge_panel(
            df_pull_summary=backbone.distractor_pull_summary,
            df_fusion=backbone.preprobe_fusion_metrics,
            df_specificity=backbone.fusion_specificity_metrics,
            save_base=layout.figure_base("panel_d_rewriting_to_fusion_bridge"),
        )
        figure_paths["panel_e"] = _save_formation_panel(
            backbone.formation_intervention_metrics,
            layout.figure_base("panel_e_formation_intervention"),
        )

    summary_payload = build_fig5_fusion_summary(
        triplets=backbone.triplets,
        preprobe_fusion_metrics=backbone.preprobe_fusion_metrics,
        fusion_specificity_metrics=backbone.fusion_specificity_metrics,
        distractor_pull_summary=backbone.distractor_pull_summary,
        sample_induced_rewriting_summary=backbone.sample_induced_rewriting_summary,
        rewriting_fusion_bridge=backbone.rewriting_fusion_bridge,
        formation_intervention_metrics=backbone.formation_intervention_metrics,
        smoke=bool(config.smoke),
    )
    summary_payload["saved_artifact_paths"] = {
        "panel_a_triplet_definition_csv": str(Path(triplet_csv).resolve()),
        "panel_b_preprobe_fusion_metrics_csv": str(Path(fusion_csv).resolve()),
        "panel_b_fusion_specificity_csv": str(Path(specificity_csv).resolve()),
        "panel_c_distractor_pull_timeseries_csv": str(Path(pull_time_csv).resolve()),
        "panel_c_distractor_pull_summary_csv": str(Path(pull_summary_csv).resolve()),
        "panel_c_sample_induced_rewriting_timeseries_csv": str(Path(rewriting_time_csv).resolve()),
        "panel_c_sample_induced_rewriting_summary_csv": str(Path(rewriting_summary_csv).resolve()),
        "panel_d_rewriting_to_fusion_bridge_csv": str(Path(bridge_csv).resolve()),
        "panel_e_formation_intervention_csv": str(Path(formation_csv).resolve()),
        "panel_e_formation_intervention_comparison_csv": str(Path(formation_compare_csv).resolve()),
        "panel_b_example_preprobe_fusion_state_npz": str(Path(example_preprobe_npz).resolve()),
        "panel_c_example_distractor_pull_trace_npz": str(Path(example_pull_npz).resolve()),
        "supp_region_support_condition_csv": str(Path(supp_region_csv).resolve()),
        "supp_layer1_trial_metrics_csv": str(Path(supp_layer1_csv).resolve()),
        "supp_layer1_formula_fit_csv": str(Path(supp_layer1_fit_csv).resolve()),
        "supp_holistic_metrics_csv": str(Path(supp_holistic_csv).resolve()),
        "supp_cue_winner_metrics_csv": str(Path(supp_cue_csv).resolve()),
        "figures": figure_paths,
    }
    summary_payload["runtime"] = {
        "triplets_analysed": int(len(backbone.triplets)),
        "device": str(device),
        "smoke": bool(config.smoke),
        "smoke_command": SMOKE_COMMAND,
        "smoke_note": SMOKE_NOTE,
    }
    summary_path = save_json_payload(summary_payload, layout.root_file("summary.json"))
    save_run_config(
        {
            "experiment": EXPERIMENT_NAME,
            **asdict(config),
            "device_resolved": str(device),
            "smoke_command": SMOKE_COMMAND,
            "smoke_note": SMOKE_NOTE,
        },
        layout.root,
    )
    run_log_path = save_log_lines(
        log_lines
        + [
            f"triplets={int(len(backbone.triplets))}",
            f"smoke={int(bool(config.smoke))}",
            f"smoke_note={SMOKE_NOTE}",
            f"summary_json={summary_path.resolve()}",
        ],
        layout.log_dir,
        filename="run.log",
    )

    print(f"Triplets analysed: {int(len(backbone.triplets))}")
    print(f"Primary fusion layer: {PRIMARY_FUSION_LAYER}")
    print(f"Primary rewriting metric: {PRIMARY_REWRITING_METRIC}")
    print(f"Smoke note: {SMOKE_NOTE}")
    print(f"Summary path: {summary_path}")
    print(f"Run log path: {run_log_path}")


if __name__ == "__main__":
    main()
