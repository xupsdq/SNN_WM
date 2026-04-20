from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.config.units import ms
from src.core.network import SDNN_Network
from src.data.encoding import DoGSpikeEncoder
from src.experiments.common.dataset import build_class_index
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.experiments.common.runtime import resolve_device, seed_everything
from src.plotting.common.io import (
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_tidy_csv,
    validate_required_columns,
)
from src.plotting.common.theme_tokens import (
    BEHAVIOR_CONDITION_COLORS,
    FIGSIZE_FOUR_PANEL,
    LINE_WIDTH_REFERENCE,
    LINE_WIDTH_SECONDARY,
    MARKER_CIRCLE,
    MODE_COLORS_DYNAMIC_STATIC,
    apply_standard_legend,
)

CONDITIONS: Tuple[str, ...] = ("congruent", "baseline", "incongruent")
MEMORY_CONDITIONS: Tuple[str, ...] = ("congruent", "incongruent")
STSP_MODES: Tuple[str, ...] = ("dynamic", "static_frozen")
PROBE_SUBSETS: Tuple[str, ...] = ("baseline_correct", "baseline_error")

CONDITION_COLORS: Dict[str, str] = dict(BEHAVIOR_CONDITION_COLORS)
MODE_COLORS: Dict[str, str] = dict(MODE_COLORS_DYNAMIC_STATIC)
PROBE_SUBSET_TITLES: Dict[str, str] = {
    "baseline_correct": "Baseline-correct probes",
    "baseline_error": "Baseline-error probes",
}
MODE_TITLES: Dict[str, str] = {
    "dynamic": "Dynamic STSP",
    "static_frozen": "Static Frozen STSP",
}
METRIC_SPECS: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("rescue_rate", "Rescue rate", "#009E73", "rescue_ci_low", "rescue_ci_high"),
    ("corruption_rate", "Corruption rate", "#D55E00", "corruption_ci_low", "corruption_ci_high"),
    ("net_benefit", "Net benefit", "#0072B2", "net_ci_low", "net_ci_high"),
)


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


def parse_delay_list(delay_text: str) -> List[int]:
    values: List[int] = []
    for raw in str(delay_text).split(","):
        item = raw.strip()
        if not item:
            continue
        delay = int(float(item))
        if delay <= 0:
            raise ValueError("Delay values must be positive.")
        values.append(delay)
    if not values:
        raise ValueError("At least one delay is required.")
    return sorted(dict.fromkeys(values))


def mix_seed(base_seed: int, *parts: int) -> int:
    value = int(base_seed) & 0xFFFFFFFF
    for idx, part in enumerate(parts, start=1):
        value = (value * 1664525 + 1013904223 + int(part) * (374761393 + idx * 97)) & 0xFFFFFFFF
    return int(value)


def _condition_sort_key(series: pd.Series) -> pd.Series:
    order = {name: idx for idx, name in enumerate(CONDITIONS)}
    return series.map(order).astype(np.int64)


def _probe_subset_sort_key(series: pd.Series) -> pd.Series:
    order = {name: idx for idx, name in enumerate(PROBE_SUBSETS)}
    return series.map(order).astype(np.int64)


def _balanced_label_list(num_trials: int, num_classes: int, rng: random.Random) -> List[int]:
    labels = [idx % num_classes for idx in range(num_trials)]
    rng.shuffle(labels)
    return labels


def _all_probe_pool(class_index: Dict[int, List[int]]) -> pd.DataFrame:
    rows: List[Dict[str, int]] = []
    for probe_label in sorted(class_index):
        for probe_id in sorted(class_index[int(probe_label)]):
            rows.append({"probe_id": int(probe_id), "probe_label": int(probe_label)})
    return pd.DataFrame(rows).sort_values(["probe_label", "probe_id"], kind="stable").reset_index(drop=True)


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p_hat = float(successes) / float(total)
    denom = 1.0 + (z ** 2) / float(total)
    center = (p_hat + (z ** 2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + ((z ** 2) / (4.0 * (total ** 2))))
    return 100.0 * max(0.0, center - margin), 100.0 * min(1.0, center + margin)


def paired_bootstrap_diff_summary(values_a: np.ndarray, values_b: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have the same shape.")
    if a.size == 0:
        raise ValueError("paired_bootstrap_diff_summary received empty arrays.")

    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot, dtype=np.float64)
    n = a.size
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot[idx] = (float(a[sample_idx].mean()) - float(b[sample_idx].mean())) * 100.0

    observed = (float(a.mean()) - float(b.mean())) * 100.0
    return {
        "observed_diff_pp": observed,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_boot": int(n_boot),
    }


def encode_images(encoder: DoGSpikeEncoder, images: torch.Tensor, steps: int) -> torch.Tensor:
    with torch.no_grad():
        spikes = encoder.forward(images)
    return spikes[:, :steps, ...].contiguous()


def extract_prediction_and_fire_time_from_layer3(net: SDNN_Network, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_times = net.layer3.firing_times.detach().cpu()
    if flat_times.shape[0] != batch_size:
        raise ValueError("firing_times batch size mismatch.")

    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    pred = (min_indices // net.layer3.neurons_per_class).long()
    pred[~has_fired] = -1

    fire_t = min_times.clone()
    fire_t[~has_fired] = -1
    return pred, fire_t.to(torch.long)


def _stack_images(dataset, indices: Iterable[int], device: torch.device) -> torch.Tensor:
    images = [dataset[int(index)][0] for index in indices]
    return torch.stack(images, dim=0).to(device)


def prepare_batch_spikes(
    dataset,
    batch_df: pd.DataFrame,
    encoder: DoGSpikeEncoder,
    spec: ExperimentSpec,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probe_images = _stack_images(dataset, batch_df["probe_id"].tolist(), device=device)
    probe_spikes = encode_images(encoder, probe_images, spec.probe_steps)

    if batch_df["zero_sample"].eq(1).all():
        batch_size = len(batch_df)
        channels = int(probe_spikes.shape[2])
        height = int(probe_spikes.shape[3])
        width = int(probe_spikes.shape[4])
        sample_spikes = torch.zeros(
            (batch_size, spec.sample_steps, channels, height, width),
            device=device,
            dtype=probe_spikes.dtype,
        )
        return sample_spikes, probe_spikes

    if not batch_df["zero_sample"].eq(0).all():
        raise ValueError("Batches must not mix zero-sample and non-zero-sample trials.")

    sample_images = _stack_images(dataset, batch_df["sample_id"].tolist(), device=device)
    sample_spikes = encode_images(encoder, sample_images, spec.sample_steps)
    return sample_spikes, probe_spikes


def _validate_specs_for_runtime(df_specs: pd.DataFrame) -> None:
    validate_required_columns(
        df_specs,
        [
            "pair_id",
            "trial_id",
            "delay_ms",
            "condition",
            "probe_id",
            "probe_label",
            "sample_id",
            "sample_label",
            "zero_sample",
        ],
    )
    if df_specs["trial_id"].nunique() != len(df_specs):
        raise ValueError("trial_id must be unique.")
    if not set(pd.unique(df_specs["condition"])).issubset(set(CONDITIONS)):
        raise ValueError("Unknown condition present in trial specs.")


def run_behavioral_trials(
    net: SDNN_Network,
    encoder: DoGSpikeEncoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
    seed: int,
    stsp_modes: Sequence[str] = STSP_MODES,
) -> pd.DataFrame:
    _validate_specs_for_runtime(df_specs)
    records: List[Dict[str, int | str]] = []
    grouped = list(df_specs.groupby(["delay_ms", "condition", "zero_sample"], sort=True))
    total_batches = sum(math.ceil(len(group) / batch_size) for _, group in grouped)

    with tqdm(total=total_batches, desc="Behavioral Signature") as pbar:
        for (delay_ms, condition, _), group in grouped:
            group = group.sort_values(["pair_id", "trial_id"], kind="stable").reset_index(drop=True)
            delay_steps = int(round((float(delay_ms) * ms) / spec.dt))
            for start in range(0, len(group), batch_size):
                batch = group.iloc[start:start + batch_size].copy()
                sample_spikes, probe_spikes = prepare_batch_spikes(
                    dataset=dataset,
                    batch_df=batch,
                    encoder=encoder,
                    spec=spec,
                    device=device,
                )
                bsz = len(batch)

                for stsp_mode in stsp_modes:
                    with torch.no_grad():
                        out = net.forward_classify_session(
                            sample_spikes=sample_spikes,
                            test_spikes=probe_spikes,
                            delay_duration_steps=delay_steps,
                            stsp_mode=str(stsp_mode),
                        )
                    pred, fire_t = extract_prediction_and_fire_time_from_layer3(net, bsz)
                    returned_pred = out["prediction"].detach().cpu().long()
                    if not torch.equal(pred, returned_pred):
                        raise ValueError("Prediction mismatch between returned output and layer3 state.")

                    for idx_in_batch, row in enumerate(batch.itertuples(index=False)):
                        predicted_label = int(pred[idx_in_batch].item())
                        first_fire = int(fire_t[idx_in_batch].item())
                        probe_label = int(row.probe_label)
                        records.append(
                            {
                                "seed": int(seed),
                                "trial_id": int(row.trial_id),
                                "pair_id": int(row.pair_id),
                                "delay_ms": int(delay_ms),
                                "condition": str(condition),
                                "stsp_mode": str(stsp_mode),
                                "probe_id": int(row.probe_id),
                                "probe_label": int(probe_label),
                                "sample_id": int(row.sample_id),
                                "sample_label": int(row.sample_label),
                                "zero_sample": int(row.zero_sample),
                                "predicted_label": int(predicted_label),
                                "prediction_probe": int(predicted_label),
                                "first_fire_t_probe": int(first_fire),
                                "is_correct": int(predicted_label == probe_label),
                                "is_silent": int(predicted_label == -1),
                            }
                        )
                pbar.update(1)

    df_trials = pd.DataFrame(records)
    return (
        df_trials.assign(condition_order=_condition_sort_key(df_trials["condition"]))
        .sort_values(["seed", "delay_ms", "pair_id", "condition_order", "stsp_mode"], kind="stable")
        .drop(columns=["condition_order"])
        .reset_index(drop=True)
    )


def identify_probe_subsets(
    net: SDNN_Network,
    encoder: DoGSpikeEncoder,
    dataset,
    probe_candidates: pd.DataFrame,
    spec: ExperimentSpec,
    delay_values_ms: Sequence[int],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    validate_required_columns(probe_candidates, ["probe_id", "probe_label"])
    probe_pool = (
        probe_candidates[["probe_id", "probe_label"]]
        .drop_duplicates()
        .sort_values(["probe_label", "probe_id"], kind="stable")
        .reset_index(drop=True)
    )
    rows: List[Dict[str, int | str]] = []
    pair_id = 0
    trial_id = 0
    for delay_ms in delay_values_ms:
        for row in probe_pool.itertuples(index=False):
            rows.append(
                {
                    "pair_id": int(pair_id),
                    "trial_id": int(trial_id),
                    "delay_ms": int(delay_ms),
                    "condition": "baseline",
                    "probe_id": int(row.probe_id),
                    "probe_label": int(row.probe_label),
                    "sample_id": -1,
                    "sample_label": -1,
                    "zero_sample": 1,
                }
            )
            pair_id += 1
            trial_id += 1

    df_specs = pd.DataFrame(rows)
    df_trials = run_behavioral_trials(
        net=net,
        encoder=encoder,
        dataset=dataset,
        df_specs=df_specs,
        spec=spec,
        batch_size=batch_size,
        device=device,
        seed=seed,
        stsp_modes=("static_frozen",),
    )

    check = (
        df_trials.groupby(["probe_id", "probe_label"], as_index=False)
        .agg(
            n_reference_trials=("is_correct", "size"),
            subset_reference_correct=("is_correct", "first"),
            n_unique_correct=("is_correct", "nunique"),
            reference_predicted_label=("predicted_label", "first"),
        )
        .copy()
    )
    inconsistent = check[check["n_unique_correct"] != 1].copy()
    if not inconsistent.empty:
        bad_rows = inconsistent.head(10)[["probe_id", "probe_label", "n_unique_correct"]].to_dict("records")
        raise ValueError(f"Baseline probe subsets are not deterministic across delays: {bad_rows}")

    check["probe_subset"] = np.where(
        check["subset_reference_correct"].eq(1),
        "baseline_correct",
        "baseline_error",
    )
    check["reference_stsp_mode"] = "static_frozen"
    check["reference_condition"] = "baseline"
    check["reference_delays_ms"] = ",".join(str(int(v)) for v in delay_values_ms)
    return (
        check[
            [
                "probe_id",
                "probe_label",
                "probe_subset",
                "subset_reference_correct",
                "reference_predicted_label",
                "n_reference_trials",
                "reference_stsp_mode",
                "reference_condition",
                "reference_delays_ms",
            ]
        ]
        .assign(probe_subset_order=lambda x: _probe_subset_sort_key(x["probe_subset"]))
        .sort_values(["probe_subset_order", "probe_label", "probe_id"], kind="stable")
        .drop(columns=["probe_subset_order"])
        .reset_index(drop=True)
    )


def _next_probe_record(
    per_label_records: Mapping[int, List[Dict[str, object]]],
    cursor_state: Dict[int, int],
    shuffle_state: Dict[int, List[int]],
    label: int,
    rng: random.Random,
) -> Dict[str, object]:
    records = per_label_records[int(label)]
    if not records:
        raise ValueError(f"No probe reference rows available for label {label}.")
    if int(label) not in shuffle_state or cursor_state[int(label)] >= len(shuffle_state[int(label)]):
        order = list(range(len(records)))
        rng.shuffle(order)
        shuffle_state[int(label)] = order
        cursor_state[int(label)] = 0
    record = records[shuffle_state[int(label)][cursor_state[int(label)]]]
    cursor_state[int(label)] += 1
    return record


def _build_incongruent_sample_labels(probe_labels: Sequence[int], num_classes: int, rng: random.Random) -> List[int]:
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2.")
    offsets = [1 + (idx % (num_classes - 1)) for idx in range(len(probe_labels))]
    rng.shuffle(offsets)
    return [int((int(probe_label) + int(offset)) % num_classes) for probe_label, offset in zip(probe_labels, offsets)]


def sample_trials_by_condition(
    class_index: Dict[int, List[int]],
    probe_pool: pd.DataFrame,
    delay_values_ms: Sequence[int],
    trials_per_condition: int,
    num_classes: int,
    seed: int,
) -> pd.DataFrame:
    if trials_per_condition <= 0:
        raise ValueError("trials_per_condition must be positive.")

    per_label_records: Dict[int, List[Dict[str, object]]] = {}
    for probe_label, group in probe_pool.groupby("probe_label", sort=True):
        per_label_records[int(probe_label)] = group.to_dict("records")

    all_rows: List[Dict[str, int | str]] = []
    pair_id_offset = 0
    for delay_idx, delay_ms in enumerate(delay_values_ms):
        rng_probe = random.Random(mix_seed(seed, 501, delay_idx, int(delay_ms)))
        rng_congruent = random.Random(mix_seed(seed, 511, delay_idx, int(delay_ms)))
        rng_incongruent = random.Random(mix_seed(seed, 521, delay_idx, int(delay_ms)))

        desired_labels = _balanced_label_list(
            num_trials=trials_per_condition,
            num_classes=num_classes,
            rng=rng_probe,
        )
        cursor_state: Dict[int, int] = {}
        shuffle_state: Dict[int, List[int]] = {}
        base_rows: List[Dict[str, int | str]] = []
        for local_idx, probe_label in enumerate(desired_labels):
            probe_record = _next_probe_record(
                per_label_records=per_label_records,
                cursor_state=cursor_state,
                shuffle_state=shuffle_state,
                label=int(probe_label),
                rng=rng_probe,
            )
            base_rows.append(
                {
                    "pair_id": int(pair_id_offset + local_idx),
                    "delay_ms": int(delay_ms),
                    "probe_id": int(probe_record["probe_id"]),
                    "probe_label": int(probe_record["probe_label"]),
                }
            )
        pair_id_offset += len(base_rows)
        incongruent_labels = _build_incongruent_sample_labels(
            probe_labels=[int(row["probe_label"]) for row in base_rows],
            num_classes=num_classes,
            rng=rng_incongruent,
        )

        for row, incongruent_label in zip(base_rows, incongruent_labels):
            probe_id = int(row["probe_id"])
            probe_label = int(row["probe_label"])
            congruent_candidates = [idx for idx in class_index[probe_label] if int(idx) != probe_id]
            if not congruent_candidates:
                raise ValueError(f"No alternate congruent sample available for class {probe_label}.")
            congruent_sample_id = int(rng_congruent.choice(congruent_candidates))
            incongruent_sample_id = int(rng_incongruent.choice(class_index[int(incongruent_label)]))
            all_rows.extend(
                [
                    {
                        **row,
                        "condition": "congruent",
                        "sample_id": int(congruent_sample_id),
                        "sample_label": int(probe_label),
                        "zero_sample": 0,
                    },
                    {
                        **row,
                        "condition": "baseline",
                        "sample_id": -1,
                        "sample_label": -1,
                        "zero_sample": 1,
                    },
                    {
                        **row,
                        "condition": "incongruent",
                        "sample_id": int(incongruent_sample_id),
                        "sample_label": int(incongruent_label),
                        "zero_sample": 0,
                    },
                ]
            )

    df_specs = pd.DataFrame(all_rows)
    df_specs["condition_order"] = _condition_sort_key(df_specs["condition"])
    df_specs = df_specs.sort_values(["delay_ms", "pair_id", "condition_order"], kind="stable").reset_index(drop=True)
    df_specs["trial_id"] = np.arange(len(df_specs), dtype=np.int64)
    return df_specs.drop(columns=["condition_order"])


def _attach_probe_subset_metadata(df_trials: pd.DataFrame, probe_reference: pd.DataFrame) -> pd.DataFrame:
    merged = df_trials.merge(
        probe_reference[
            [
                "probe_id",
                "probe_label",
                "probe_subset",
                "subset_reference_correct",
                "reference_predicted_label",
            ]
        ],
        on=["probe_id", "probe_label"],
        how="left",
        validate="many_to_one",
    )
    if merged["probe_subset"].isna().any():
        raise ValueError("Missing probe subset metadata for some probe trials.")
    return merged


def compute_accuracy_summary(df_trials: pd.DataFrame) -> pd.DataFrame:
    delays = sorted(pd.unique(df_trials["delay_ms"]).tolist())
    rows: List[Dict[str, float | int | str]] = []
    for stsp_mode in STSP_MODES:
        for delay_ms in delays:
            for condition in CONDITIONS:
                for probe_subset in PROBE_SUBSETS:
                    subset = df_trials[
                        (df_trials["stsp_mode"] == stsp_mode)
                        & (df_trials["delay_ms"] == int(delay_ms))
                        & (df_trials["condition"] == condition)
                        & (df_trials["probe_subset"] == probe_subset)
                    ].copy()
                    n_trials = int(len(subset))
                    n_correct = int(subset["is_correct"].sum()) if n_trials > 0 else 0
                    ci_low, ci_high = wilson_ci(n_correct, n_trials)
                    rows.append(
                        {
                            "stsp_mode": str(stsp_mode),
                            "delay_ms": int(delay_ms),
                            "condition": str(condition),
                            "probe_subset": str(probe_subset),
                            "n_trials": int(n_trials),
                            "accuracy": 100.0 * float(n_correct) / float(n_trials) if n_trials > 0 else float("nan"),
                            "ci_low": float(ci_low),
                            "ci_high": float(ci_high),
                        }
                    )
    return pd.DataFrame(rows)


def _rescue_corruption_components(subset: pd.DataFrame) -> Dict[str, float | int]:
    baseline_error = subset[subset["probe_subset"] == "baseline_error"]
    baseline_correct = subset[subset["probe_subset"] == "baseline_correct"]

    n_error = int(len(baseline_error))
    n_correct = int(len(baseline_correct))
    rescue_success = int(baseline_error["is_correct"].sum())
    corruption_fail = int((1 - baseline_correct["is_correct"]).sum())

    rescue_rate = 100.0 * float(rescue_success) / float(n_error) if n_error > 0 else float("nan")
    corruption_rate = 100.0 * float(corruption_fail) / float(n_correct) if n_correct > 0 else float("nan")
    net_benefit = rescue_rate - corruption_rate if n_error > 0 and n_correct > 0 else float("nan")
    return {
        "n_baseline_error": n_error,
        "n_baseline_correct": n_correct,
        "rescue_success": rescue_success,
        "corruption_fail": corruption_fail,
        "rescue_rate": rescue_rate,
        "corruption_rate": corruption_rate,
        "net_benefit": net_benefit,
    }


def _bootstrap_net_benefit_ci(subset: pd.DataFrame, n_boot: int, seed: int) -> Tuple[float, float]:
    if len(subset) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    correct = subset["is_correct"].to_numpy(dtype=np.int64)
    probe_subset = subset["probe_subset"].to_numpy(dtype=object)
    values = np.full(n_boot, np.nan, dtype=np.float64)
    n = len(subset)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        sampled = pd.DataFrame(
            {
                "is_correct": correct[sample_idx],
                "probe_subset": probe_subset[sample_idx],
            }
        )
        values[idx] = float(_rescue_corruption_components(sampled)["net_benefit"])
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def compute_rescue_corruption_metrics(
    df_trials: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    delays = sorted(pd.unique(df_trials["delay_ms"]).tolist())
    rows: List[Dict[str, float | int | str]] = []
    for mode_idx, stsp_mode in enumerate(STSP_MODES):
        for delay_idx, delay_ms in enumerate(delays):
            for cond_idx, condition in enumerate(MEMORY_CONDITIONS):
                subset = df_trials[
                    (df_trials["stsp_mode"] == stsp_mode)
                    & (df_trials["delay_ms"] == int(delay_ms))
                    & (df_trials["condition"] == condition)
                ].copy()
                stats = _rescue_corruption_components(subset=subset)
                rescue_ci_low, rescue_ci_high = wilson_ci(
                    int(stats["rescue_success"]),
                    int(stats["n_baseline_error"]),
                )
                corruption_ci_low, corruption_ci_high = wilson_ci(
                    int(stats["corruption_fail"]),
                    int(stats["n_baseline_correct"]),
                )
                net_ci_low, net_ci_high = _bootstrap_net_benefit_ci(
                    subset=subset,
                    n_boot=n_boot,
                    seed=mix_seed(seed, 601, mode_idx, delay_idx, cond_idx),
                )
                rows.append(
                    {
                        "stsp_mode": str(stsp_mode),
                        "delay_ms": int(delay_ms),
                        "condition": str(condition),
                        "n_trials": int(len(subset)),
                        "n_baseline_error": int(stats["n_baseline_error"]),
                        "n_baseline_correct": int(stats["n_baseline_correct"]),
                        "rescue_rate": float(stats["rescue_rate"]),
                        "corruption_rate": float(stats["corruption_rate"]),
                        "net_benefit": float(stats["net_benefit"]),
                        "rescue_ci_low": float(rescue_ci_low),
                        "rescue_ci_high": float(rescue_ci_high),
                        "corruption_ci_low": float(corruption_ci_low),
                        "corruption_ci_high": float(corruption_ci_high),
                        "net_ci_low": float(net_ci_low),
                        "net_ci_high": float(net_ci_high),
                        "ci_low": float(net_ci_low),
                        "ci_high": float(net_ci_high),
                    }
                )
    return pd.DataFrame(rows)


def _compute_overall_accuracy_summary(df_accuracy: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    grouped = df_accuracy.groupby(["stsp_mode", "delay_ms", "condition"], sort=False)
    for (stsp_mode, delay_ms, condition), subset in grouped:
        n_trials = int(subset["n_trials"].sum())
        weighted_correct = float(((subset["accuracy"].fillna(0.0) / 100.0) * subset["n_trials"]).sum())
        n_correct = int(round(weighted_correct))
        ci_low, ci_high = wilson_ci(n_correct, n_trials)
        rows.append(
            {
                "stsp_mode": str(stsp_mode),
                "delay_ms": int(delay_ms),
                "condition": str(condition),
                "n_trials": int(n_trials),
                "accuracy": 100.0 * float(n_correct) / float(n_trials) if n_trials > 0 else float("nan"),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
    return pd.DataFrame(rows)


def _paired_condition_frame(
    df_trials: pd.DataFrame,
    *,
    stsp_mode: str,
    delay_ms: int,
    probe_subset: str,
) -> pd.DataFrame:
    subset = df_trials[
        (df_trials["stsp_mode"] == stsp_mode)
        & (df_trials["delay_ms"] == int(delay_ms))
        & (df_trials["probe_subset"] == probe_subset)
        & (df_trials["condition"].isin(MEMORY_CONDITIONS))
    ].copy()
    pivot = subset.pivot_table(index="pair_id", columns="condition", values="is_correct", aggfunc="first")
    if not {"congruent", "incongruent"}.issubset(set(pivot.columns)):
        return pd.DataFrame()
    return pivot[["congruent", "incongruent"]].dropna().sort_index()


def _compute_congruency_effect_table(df_trials: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    delays = sorted(pd.unique(df_trials["delay_ms"]).tolist())
    rows: List[Dict[str, float | int | str]] = []
    for mode_idx, stsp_mode in enumerate(STSP_MODES):
        for delay_idx, delay_ms in enumerate(delays):
            for subset_idx, probe_subset in enumerate(PROBE_SUBSETS):
                pivot = _paired_condition_frame(
                    df_trials=df_trials,
                    stsp_mode=stsp_mode,
                    delay_ms=int(delay_ms),
                    probe_subset=probe_subset,
                )
                if pivot.empty:
                    rows.append(
                        {
                            "stsp_mode": str(stsp_mode),
                            "delay_ms": int(delay_ms),
                            "probe_subset": str(probe_subset),
                            "n_pairs": 0,
                            "congruency_effect": float("nan"),
                            "ci_low": float("nan"),
                            "ci_high": float("nan"),
                        }
                    )
                    continue
                res = paired_bootstrap_diff_summary(
                    pivot["congruent"].to_numpy(dtype=np.float64),
                    pivot["incongruent"].to_numpy(dtype=np.float64),
                    n_boot=n_boot,
                    seed=mix_seed(seed, 701, mode_idx, delay_idx, subset_idx),
                )
                rows.append(
                    {
                        "stsp_mode": str(stsp_mode),
                        "delay_ms": int(delay_ms),
                        "probe_subset": str(probe_subset),
                        "n_pairs": int(len(pivot)),
                        "congruency_effect": float(res["observed_diff_pp"]),
                        "ci_low": float(res["ci_low"]),
                        "ci_high": float(res["ci_high"]),
                    }
                )
    return pd.DataFrame(rows)


def _errorbar_from_ci(values: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    highs = np.asarray(highs, dtype=np.float64)
    lower = np.minimum(lows, highs)
    upper = np.maximum(lows, highs)
    return np.vstack(
        [
            np.clip(values - lower, a_min=0.0, a_max=None),
            np.clip(upper - values, a_min=0.0, a_max=None),
        ]
    )


def plot_accuracy_vs_delay(df_accuracy: pd.DataFrame) -> plt.Figure:
    overall = _compute_overall_accuracy_summary(df_accuracy=df_accuracy)
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE, sharey=True)
    delays = sorted(pd.unique(overall["delay_ms"]).tolist())
    for ax, condition in zip(axes, CONDITIONS):
        subset = overall[overall["condition"] == condition].copy()
        for stsp_mode in STSP_MODES:
            mode_df = subset[subset["stsp_mode"] == stsp_mode].sort_values("delay_ms")
            x = mode_df["delay_ms"].to_numpy(dtype=np.float64)
            y = mode_df["accuracy"].to_numpy(dtype=np.float64)
            lo = mode_df["ci_low"].to_numpy(dtype=np.float64)
            hi = mode_df["ci_high"].to_numpy(dtype=np.float64)
            ax.errorbar(
                x,
                y,
                yerr=_errorbar_from_ci(y, lo, hi),
                marker=MARKER_CIRCLE,
                linewidth=LINE_WIDTH_SECONDARY,
                color=MODE_COLORS[stsp_mode],
                label=MODE_TITLES[stsp_mode],
            )
        ax.set_title(condition.capitalize())
        ax.set_xlabel("Delay (ms)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0.0, 100.0)
        ax.set_xticks(delays)
        apply_standard_legend(ax, title=None)
    fig.tight_layout()
    return fig


def plot_baseline_error_accuracy(df_accuracy: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE, sharey=True)
    subset_df = df_accuracy[df_accuracy["probe_subset"] == "baseline_error"].copy()
    delays = sorted(pd.unique(subset_df["delay_ms"]).tolist())
    for ax, condition in zip(axes, CONDITIONS):
        condition_df = subset_df[subset_df["condition"] == condition].copy()
        for stsp_mode in STSP_MODES:
            mode_df = condition_df[condition_df["stsp_mode"] == stsp_mode].sort_values("delay_ms")
            x = mode_df["delay_ms"].to_numpy(dtype=np.float64)
            y = mode_df["accuracy"].to_numpy(dtype=np.float64)
            lo = mode_df["ci_low"].to_numpy(dtype=np.float64)
            hi = mode_df["ci_high"].to_numpy(dtype=np.float64)
            ax.errorbar(
                x,
                y,
                yerr=_errorbar_from_ci(y, lo, hi),
                marker=MARKER_CIRCLE,
                linewidth=LINE_WIDTH_SECONDARY,
                color=MODE_COLORS[stsp_mode],
                label=MODE_TITLES[stsp_mode],
            )
        ax.set_title(condition.capitalize())
        ax.set_xlabel("Delay (ms)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0.0, 100.0)
        ax.set_xticks(delays)
        apply_standard_legend(ax, title=None)
    fig.tight_layout()
    return fig


def plot_rescue_corruption(df_rescue: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_FOUR_PANEL, sharex=True, sharey=True)
    delays = sorted(pd.unique(df_rescue["delay_ms"]).tolist())
    for row_idx, stsp_mode in enumerate(STSP_MODES):
        for col_idx, condition in enumerate(MEMORY_CONDITIONS):
            ax = axes[row_idx, col_idx]
            subset = df_rescue[
                (df_rescue["stsp_mode"] == stsp_mode)
                & (df_rescue["condition"] == condition)
            ].copy().sort_values("delay_ms")
            for metric, label, color, lo_col, hi_col in METRIC_SPECS:
                x = subset["delay_ms"].to_numpy(dtype=np.float64)
                y = subset[metric].to_numpy(dtype=np.float64)
                lo = subset[lo_col].to_numpy(dtype=np.float64)
                hi = subset[hi_col].to_numpy(dtype=np.float64)
                ax.errorbar(
                    x,
                    y,
                    yerr=_errorbar_from_ci(y, lo, hi),
                    marker=MARKER_CIRCLE,
                    linewidth=LINE_WIDTH_SECONDARY,
                    color=color,
                    label=label,
                )
            ax.axhline(0.0, color="black", linewidth=LINE_WIDTH_REFERENCE, linestyle="--")
            ax.set_title(f"{MODE_TITLES[stsp_mode]} | {condition}")
            ax.set_xlabel("Delay (ms)")
            ax.set_ylabel("Rate (%)")
            ax.set_xticks(delays)
            apply_standard_legend(ax, title=None)
    fig.tight_layout()
    return fig


def plot_congruency_effect(df_congruency: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE, sharey=True)
    delays = sorted(pd.unique(df_congruency["delay_ms"]).tolist())
    for ax, probe_subset in zip(axes, PROBE_SUBSETS):
        subset = df_congruency[df_congruency["probe_subset"] == probe_subset].copy()
        for stsp_mode in STSP_MODES:
            mode_df = subset[subset["stsp_mode"] == stsp_mode].sort_values("delay_ms")
            x = mode_df["delay_ms"].to_numpy(dtype=np.float64)
            y = mode_df["congruency_effect"].to_numpy(dtype=np.float64)
            lo = mode_df["ci_low"].to_numpy(dtype=np.float64)
            hi = mode_df["ci_high"].to_numpy(dtype=np.float64)
            ax.errorbar(
                x,
                y,
                yerr=_errorbar_from_ci(y, lo, hi),
                marker=MARKER_CIRCLE,
                linewidth=LINE_WIDTH_SECONDARY,
                color=MODE_COLORS[stsp_mode],
                label=MODE_TITLES[stsp_mode],
            )
        ax.axhline(0.0, color="black", linewidth=LINE_WIDTH_REFERENCE, linestyle="--")
        ax.set_title(PROBE_SUBSET_TITLES[probe_subset])
        ax.set_xlabel("Delay (ms)")
        ax.set_ylabel("Congruency effect (pp)")
        ax.set_xticks(delays)
        apply_standard_legend(ax, title=None)
    fig.tight_layout()
    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline-defined probe-subset DMS behavioral memory analysis.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--save-dir", type=str, default="results/behavioral_signature")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials-per-condition", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-boot", type=int, default=2000)
    parser.add_argument("--delay-ms-list", type=str, default="300,600,1000")
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.trials_per_condition <= 0:
        raise ValueError("--trials-per-condition must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_boot <= 0:
        raise ValueError("--num-boot must be positive.")
    if args.num_classes < 2:
        raise ValueError("--num-classes must be >= 2.")

    delay_values_ms = parse_delay_list(args.delay_ms_list)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=1.0 * ms, sample_ms=args.sample_ms, probe_ms=args.probe_ms)
    if spec.sample_steps <= 0 or spec.probe_steps <= 0:
        raise ValueError("sample and probe durations must resolve to positive step counts.")

    layout = prepare_result_layout(args.save_dir)

    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(spec.sample_ms, spec.probe_ms),
    )
    _, _, test_loader = build_mnist_skeleton_loader(
        root=args.dataset_root,
        batch_size=1,
        input_size=28,
        num_workers=0,
    )
    dataset = test_loader.dataset
    class_index = build_class_index(dataset, num_classes=args.num_classes)

    probe_pool = _all_probe_pool(class_index=class_index)
    df_specs = sample_trials_by_condition(
        class_index=class_index,
        probe_pool=probe_pool,
        delay_values_ms=delay_values_ms,
        trials_per_condition=args.trials_per_condition,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    probe_reference = identify_probe_subsets(
        net=net,
        encoder=encoder,
        dataset=dataset,
        probe_candidates=df_specs[["probe_id", "probe_label"]],
        spec=spec,
        delay_values_ms=delay_values_ms,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    df_trials = run_behavioral_trials(
        net=net,
        encoder=encoder,
        dataset=dataset,
        df_specs=df_specs,
        spec=spec,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        stsp_modes=STSP_MODES,
    )
    df_trials = _attach_probe_subset_metadata(df_trials=df_trials, probe_reference=probe_reference)
    df_trials = df_trials[
        [
            "seed",
            "stsp_mode",
            "delay_ms",
            "condition",
            "sample_id",
            "sample_label",
            "probe_id",
            "probe_label",
            "probe_subset",
            "subset_reference_correct",
            "reference_predicted_label",
            "is_correct",
            "predicted_label",
            "pair_id",
            "trial_id",
            "zero_sample",
            "prediction_probe",
            "first_fire_t_probe",
            "is_silent",
        ]
    ].copy()

    df_accuracy = compute_accuracy_summary(df_trials=df_trials)
    df_rescue = compute_rescue_corruption_metrics(
        df_trials=df_trials,
        n_boot=args.num_boot,
        seed=args.seed,
    )
    df_congruency = _compute_congruency_effect_table(
        df_trials=df_trials,
        n_boot=args.num_boot,
        seed=args.seed,
    )

    probe_reference_csv = save_tidy_csv(
        probe_reference,
        layout.data_file("probe_subset_reference.csv"),
        sort_by=["probe_subset", "probe_label", "probe_id"],
    )
    trial_level_csv = save_tidy_csv(
        df_trials,
        layout.data_file("trial_level_results.csv"),
        sort_by=["seed", "stsp_mode", "delay_ms", "pair_id", "condition"],
    )
    accuracy_csv = save_tidy_csv(
        df_accuracy,
        layout.data_file("accuracy_summary.csv"),
        sort_by=["stsp_mode", "delay_ms", "condition", "probe_subset"],
    )
    rescue_csv = save_tidy_csv(
        df_rescue[
            [
                "stsp_mode",
                "delay_ms",
                "condition",
                "rescue_rate",
                "corruption_rate",
                "net_benefit",
                "ci_low",
                "ci_high",
            ]
        ],
        layout.data_file("rescue_corruption_summary.csv"),
        sort_by=["stsp_mode", "delay_ms", "condition"],
    )

    fig1 = plot_accuracy_vs_delay(df_accuracy=df_accuracy)
    fig1_paths = save_figure_all_formats(fig1, layout.figure_base("accuracy_vs_delay"))
    plt.close(fig1)

    fig2 = plot_baseline_error_accuracy(df_accuracy=df_accuracy)
    fig2_paths = save_figure_all_formats(fig2, layout.figure_base("baseline_error_accuracy"))
    plt.close(fig2)

    fig3 = plot_rescue_corruption(df_rescue=df_rescue)
    fig3_paths = save_figure_all_formats(fig3, layout.figure_base("rescue_corruption"))
    plt.close(fig3)

    fig4 = plot_congruency_effect(df_congruency=df_congruency)
    fig4_paths = save_figure_all_formats(fig4, layout.figure_base("congruency_effect"))
    plt.close(fig4)

    run_config_path = save_run_config(
        {
            "model_path": args.model_path,
            "dataset_root": args.dataset_root,
            "device": str(device),
            "seed": int(args.seed),
            "delay_ms_list": [int(v) for v in delay_values_ms],
            "trials_per_condition": int(args.trials_per_condition),
            "batch_size": int(args.batch_size),
            "num_boot": int(args.num_boot),
            "num_classes": int(args.num_classes),
            "sample_ms": float(args.sample_ms),
            "probe_ms": float(args.probe_ms),
            "baseline_pairing_note": (
                "Probe subsets are defined only for the unique probe exemplars sampled into this run, "
                "using static_frozen blank-sample baseline across all requested delays. "
                "Rescue and corruption rates are computed from baseline_error and baseline_correct probe subsets, "
                "respectively, and net benefit is rescue minus corruption."
            ),
            "outputs": {
                "probe_subset_reference": str(probe_reference_csv),
                "trial_level_results": str(trial_level_csv),
                "accuracy_summary": str(accuracy_csv),
                "rescue_corruption_summary": str(rescue_csv),
                "figure_1_png": fig1_paths["png"],
                "figure_1_pdf": fig1_paths["pdf"],
                "figure_2_png": fig2_paths["png"],
                "figure_2_pdf": fig2_paths["pdf"],
                "figure_3_png": fig3_paths["png"],
                "figure_3_pdf": fig3_paths["pdf"],
                "figure_4_png": fig4_paths["png"],
                "figure_4_pdf": fig4_paths["pdf"],
            },
        },
        layout.root,
    )
    summary_path = save_summary_json(
        {
            "experiment": "behavioral_signature_experiment",
            "delays_ms": [int(v) for v in delay_values_ms],
            "outputs": {
                "probe_subset_reference_csv": str(probe_reference_csv),
                "trial_level_results_csv": str(trial_level_csv),
                "accuracy_summary_csv": str(accuracy_csv),
                "rescue_corruption_summary_csv": str(rescue_csv),
                "figure_1_png": fig1_paths["png"],
                "figure_2_png": fig2_paths["png"],
                "figure_3_png": fig3_paths["png"],
                "figure_4_png": fig4_paths["png"],
            },
        },
        layout.root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=behavioral_signature_experiment",
            f"save_dir={layout.root}",
            f"probe_subset_reference_csv={probe_reference_csv}",
            f"trial_level_results_csv={trial_level_csv}",
            f"accuracy_summary_csv={accuracy_csv}",
            f"rescue_corruption_summary_csv={rescue_csv}",
            f"figure_1_png={fig1_paths['png']}",
            f"figure_2_png={fig2_paths['png']}",
            f"figure_3_png={fig3_paths['png']}",
            f"figure_4_png={fig4_paths['png']}",
            f"summary_json={summary_path}",
            f"run_config_json={run_config_path}",
        ],
        layout.log_dir,
    )

    print("\n=== Baseline-Defined Behavioral Signature Summary ===")
    print(f"Saved: {probe_reference_csv}")
    print(f"Saved: {trial_level_csv}")
    print(f"Saved: {accuracy_csv}")
    print(f"Saved: {rescue_csv}")
    print(f"Saved: {fig1_paths['png']}")
    print(f"Saved: {fig1_paths['pdf']}")
    print(f"Saved: {fig2_paths['png']}")
    print(f"Saved: {fig2_paths['pdf']}")
    print(f"Saved: {fig3_paths['png']}")
    print(f"Saved: {fig3_paths['pdf']}")
    print(f"Saved: {fig4_paths['png']}")
    print(f"Saved: {fig4_paths['pdf']}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {run_config_path}")
    print(f"Saved: {run_log_path}")


if __name__ == "__main__":
    main()
