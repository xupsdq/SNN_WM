import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.core.network import SDNN_Network
from src.data.encoding import DoGSpikeEncoder
from src.experiments.common.dataset import build_class_index as shared_build_class_index
from src.experiments.common.dataset import encode_images as shared_encode_images
from src.experiments.common.decoding import decode_accuracy_with_splits as shared_decode_accuracy_with_splits
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3 as shared_decode_prediction_and_fire_time_from_layer3
from src.experiments.common.model_io import compensate_stsp_gain as shared_compensate_stsp_gain
from src.experiments.common.model_io import load_model_and_encoder as shared_load_model_and_encoder
from src.experiments.common.runtime import seed_everything as shared_seed_everything
from src.config.units import ms


LAYER_KEYS = ["layer1", "layer2", "layer3"]
NO_PING_LABEL = "no_ping"


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay_ms: float
    ping_ms: float
    post_ping_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def delay_steps(self) -> int:
        return int(round((self.delay_ms * ms) / self.dt))

    @property
    def ping_steps(self) -> int:
        return int(round((self.ping_ms * ms) / self.dt))

    @property
    def post_ping_steps(self) -> int:
        return int(round((self.post_ping_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))

    @property
    def total_pre_probe_steps(self) -> int:
        return self.sample_steps + self.delay_steps + self.ping_steps + self.post_ping_steps


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seed_list(seed_list: str) -> List[int]:
    items = [s.strip() for s in seed_list.split(",") if s.strip()]
    if not items:
        raise ValueError("seed-list is empty")
    return [int(s) for s in items]


def parse_float_list(spec: str) -> List[float]:
    tokens = [tok.strip() for tok in spec.split(",") if tok.strip()]
    if not tokens:
        raise ValueError("empty float list")
    if "..." not in tokens:
        return [float(tok) for tok in tokens]

    if len(tokens) != 4 or tokens[2] != "...":
        raise ValueError(f"Unsupported ellipsis float-list syntax: {spec}")
    start = float(tokens[0])
    second = float(tokens[1])
    stop = float(tokens[3])
    step = second - start
    if step <= 0:
        raise ValueError(f"Ellipsis float-list step must be positive: {spec}")

    values: List[float] = []
    cur = start
    while cur <= stop + 1e-9:
        values.append(round(cur, 10))
        cur += step
    return values


def format_ping_target_label(target_frac: float) -> str:
    return f"target_{int(round(float(target_frac) * 1000.0)):03d}"


def compensate_stsp_gain(net: SDNN_Network, scaling_factor: float) -> None:
    with torch.no_grad():
        if hasattr(net, "layer1"):
            net.layer1.kernels.data *= scaling_factor
        if hasattr(net, "layer2"):
            net.layer2.kernels.data *= scaling_factor
        if hasattr(net, "layer3"):
            net.layer3.kernels.data *= scaling_factor


def load_model_and_encoder(
    model_path: str,
    device: torch.device,
    spec: ExperimentSpec,
) -> Tuple[SDNN_Network, DoGSpikeEncoder]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    net = SDNN_Network(device=str(device)).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    compensate_stsp_gain(net, scaling_factor=1.0 / net.layer3.stsp_U)
    net.eval()

    max_duration_ms = max(spec.sample_ms, spec.probe_ms, 100.0)
    encoder = DoGSpikeEncoder(dt=spec.dt, max_duration=max_duration_ms * ms, device=str(device))
    return net, encoder


def override_tau_u_ms(net: SDNN_Network, tau_u_ms: float, dt: float) -> None:
    tau_u = float(tau_u_ms * ms)
    decay_u = math.exp(-dt / tau_u) if tau_u > 0.0 else 0.0
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key, None)
        if layer is None or not getattr(layer, "enable_stsp", False):
            continue
        layer.stsp_decay_u = decay_u


def build_class_index(dataset, num_classes: int) -> Dict[int, List[int]]:
    class_index: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_index[int(label)].append(idx)
    for cls in range(num_classes):
        if len(class_index[cls]) == 0:
            raise ValueError(f"Class {cls} has no samples in dataset")
    return class_index


def generate_balanced_trial_specs(
    class_index: Dict[int, List[int]],
    num_trials: int,
    num_classes: int,
    rng: random.Random,
) -> pd.DataFrame:
    sample_labels = [i % num_classes for i in range(num_trials)]
    rng.shuffle(sample_labels)
    rows: List[Dict[str, int]] = []
    all_classes = list(range(num_classes))
    for trial_id, sample_label in enumerate(sample_labels):
        probe_candidates = [c for c in all_classes if c != sample_label]
        probe_label = rng.choice(probe_candidates)
        rows.append(
            {
                "trial_id": int(trial_id),
                "sample_label": int(sample_label),
                "probe_label": int(probe_label),
                "sample_index": int(rng.choice(class_index[sample_label])),
                "probe_index": int(rng.choice(class_index[probe_label])),
            }
        )
    return pd.DataFrame(rows)


def validate_trial_specs(df_specs: pd.DataFrame, num_classes: int) -> None:
    if df_specs["trial_id"].nunique() != len(df_specs):
        raise ValueError("trial_id must be unique")
    for col in ["sample_label", "probe_label"]:
        vals = df_specs[col].to_numpy(dtype=np.int64)
        if (vals < 0).any() or (vals >= num_classes).any():
            raise ValueError(f"{col} out of range")
    if not np.all(df_specs["sample_label"].to_numpy(dtype=np.int64) != df_specs["probe_label"].to_numpy(dtype=np.int64)):
        raise ValueError("probe_label must differ from sample_label")


def encode_images(encoder: DoGSpikeEncoder, images: torch.Tensor, steps: int) -> torch.Tensor:
    with torch.no_grad():
        spikes = encoder.forward(images)
    return spikes[:, :steps, ...].contiguous()


def prepare_network_state(net: SDNN_Network, batch_size: int, c: int, h: int, w: int) -> None:
    net.layer1.reset_state((batch_size, c, h, w))

    h1 = (h + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    w1 = (w + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    h1_p, w1_p = h1 // 2, w1 // 2
    net.layer2.reset_state((batch_size, net.layer1.out_channels, h1_p, w1_p))

    h2 = (h1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    w2 = (w1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    h2_p, w2_p = h2 // 2, w2 // 2
    net.layer3.reset_state((batch_size, net.layer2.out_channels, h2_p, w2_p))


def reset_l3_decision_window(net: SDNN_Network) -> None:
    net.layer3.reset_decision_state()
    with torch.no_grad():
        net.layer3.v_mem.fill_(net.layer3.V_L)
        net.layer3.lateral_inh.reset_state(net.layer3.output_shape)


def decode_prediction_and_fire_time_from_layer3(net: SDNN_Network, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_times = net.layer3.firing_times
    if flat_times.shape[0] != batch_size:
        raise ValueError(f"Batch mismatch: firing_times={flat_times.shape[0]}, expected={batch_size}")
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    pred = (min_indices // net.layer3.neurons_per_class).long()
    pred[~has_fired] = -1
    fire_t = min_times.clone()
    fire_t[~has_fired] = -1
    return pred.detach().cpu().long(), fire_t.detach().cpu().long()


def snapshot_ux_state(net: SDNN_Network, batch_size: int) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key, None)
        if layer is None or getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
            raise ValueError(f"{layer_key} is missing STSP state")
        u = layer.u_pre.detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=False)
        x = layer.x_pre.detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=False)
        gain = (layer.u_pre * layer.x_pre).detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=False)
        out[layer_key] = {"u": u, "x": x, "gain": gain}
    return out


def snapshot_ux_layer_means(net: SDNN_Network, batch_size: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key, None)
        if layer is None or getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
            raise ValueError(f"{layer_key} is missing STSP state")
        gain = (layer.u_pre * layer.x_pre).detach().view(batch_size, -1)
        out[layer_key] = gain.mean(dim=1).cpu().numpy().astype(np.float32, copy=False)
    return out


def build_stratified_splits(
    labels: np.ndarray,
    n_splits: int,
    test_ratio: float,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_splits):
        train_idx: List[int] = []
        test_idx: List[int] = []
        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) < 2:
                raise ValueError(f"Class {int(cls)} has <2 trials; increase num-trials")
            perm = rng.permutation(cls_idx)
            n_test = max(1, int(round(len(perm) * test_ratio)))
            n_test = min(n_test, len(perm) - 1)
            test_idx.extend(perm[:n_test].tolist())
            train_idx.extend(perm[n_test:].tolist())
        splits.append((np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)))
    return splits


def decode_accuracy_with_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    device: Optional[torch.device] = None,
) -> float:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape={x.shape}")
    if device is None:
        device = torch.device("cpu")

    x_t = torch.as_tensor(x.astype(np.float32, copy=False), dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y.astype(np.int64, copy=False), dtype=torch.long, device=device)
    accs: List[float] = []
    for train_idx, test_idx in splits:
        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        test_idx_t = torch.as_tensor(test_idx, dtype=torch.long, device=device)
        x_train = x_t.index_select(0, train_idx_t)
        y_train = y_t.index_select(0, train_idx_t)
        x_test = x_t.index_select(0, test_idx_t)
        y_test = y_t.index_select(0, test_idx_t)

        d = x_t.shape[1]
        counts = torch.bincount(y_train, minlength=num_classes).to(torch.float32)
        valid = counts > 0
        if not torch.any(valid):
            accs.append(0.0)
            continue

        centroids = torch.zeros((num_classes, d), dtype=torch.float32, device=device)
        centroids.index_add_(0, y_train, x_train)
        centroids[valid] = centroids[valid] / counts[valid].unsqueeze(1)

        dist = torch.cdist(x_test, centroids, p=2.0) ** 2
        dist[:, ~valid] = float("inf")
        pred = torch.argmin(dist, dim=1)
        accs.append(float((pred == y_test).to(torch.float32).mean().item()))
    return float(np.mean(accs))


seed_everything = shared_seed_everything
build_class_index = shared_build_class_index
encode_images = shared_encode_images
decode_prediction_and_fire_time_from_layer3 = shared_decode_prediction_and_fire_time_from_layer3


def compensate_stsp_gain(net: SDNN_Network, scaling_factor: float) -> None:
    shared_compensate_stsp_gain(net, scaling_factor=scaling_factor)


def load_model_and_encoder(
    model_path: str,
    device: torch.device,
    spec: ExperimentSpec,
) -> Tuple[SDNN_Network, DoGSpikeEncoder]:
    return shared_load_model_and_encoder(
        model_path=model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(spec.sample_ms, spec.probe_ms, 100.0),
    )


def decode_accuracy_with_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    device: Optional[torch.device] = None,
) -> float:
    return shared_decode_accuracy_with_splits(
        x=x,
        y=y,
        splits=splits,
        num_classes=num_classes,
        device=device,
    )


def compute_sample_and_noise_bias(df_subset: pd.DataFrame, num_classes: int) -> Tuple[float, float]:
    err = df_subset[df_subset["prediction_probe"] != df_subset["probe_label"]]
    if len(err) == 0:
        return 0.0, 0.0
    pred = err["prediction_probe"].to_numpy(dtype=np.int64)
    sample = err["sample_label"].to_numpy(dtype=np.int64)
    probe = err["probe_label"].to_numpy(dtype=np.int64)
    valid = (pred >= 0) & (pred < num_classes)
    bias_sample = float(np.mean(pred == sample))
    k = num_classes - 2
    if k <= 0:
        raise ValueError("num_classes is too small for noise-bias definition")
    noise_hit = valid & (pred != sample) & (pred != probe)
    bias_noise = float(noise_hit.sum() / float(len(err) * k))
    return bias_sample, bias_noise


def select_monotonic_ping_indices(
    amps: np.ndarray,
    activation: np.ndarray,
    targets: Sequence[float],
) -> Optional[List[int]]:
    positive_idx = [i for i in range(len(amps)) if amps[i] > 0.0 and activation[i] > 0.0]
    k = len(targets)
    m = len(positive_idx)
    if m < k or k == 0:
        return None

    dp = np.full((k, m), np.inf, dtype=np.float64)
    back = np.full((k, m), -1, dtype=np.int64)

    for i, idx in enumerate(positive_idx):
        dp[0, i] = abs(float(activation[idx]) - float(targets[0]))

    for j in range(1, k):
        for i in range(j, m):
            cost_here = abs(float(activation[positive_idx[i]]) - float(targets[j]))
            prev_scores = dp[j - 1, :i]
            if prev_scores.size == 0:
                continue
            best_prev = int(np.argmin(prev_scores))
            best_score = float(prev_scores[best_prev])
            if not np.isfinite(best_score):
                continue
            dp[j, i] = best_score + cost_here
            back[j, i] = best_prev

    end_i = int(np.argmin(dp[k - 1]))
    if not np.isfinite(dp[k - 1, end_i]):
        return None

    chosen_rev: List[int] = []
    cur_i = end_i
    for j in range(k - 1, -1, -1):
        chosen_rev.append(positive_idx[cur_i])
        if j > 0:
            cur_i = int(back[j, cur_i])
            if cur_i < 0:
                return None
    return list(reversed(chosen_rev))


def summarize_metrics(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=list(group_cols))
    grouped = df.groupby(list(group_cols), as_index=False)
    mean_df = grouped[list(value_cols)].mean()
    sem_df = grouped[list(value_cols)].sem().fillna(0.0)
    stats_df = pd.concat(
        [
            mean_df[list(value_cols)].rename(columns={col: f"{col}_mean" for col in value_cols}),
            sem_df[list(value_cols)].rename(columns={col: f"{col}_sem" for col in value_cols}),
        ],
        axis=1,
    )
    return pd.concat([mean_df[list(group_cols)], stats_df], axis=1)
