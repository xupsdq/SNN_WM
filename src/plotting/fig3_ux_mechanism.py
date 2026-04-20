import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm

from src.experiments.ping_memory.shared.ping_api import (
    LAYER_KEYS,
    build_stratified_splits,
    decode_accuracy_with_splits,
    prepare_network_state,
    snapshot_ux_state,
)
from src.experiments.ping_memory.shared.shuffle_ops import (
    apply_trial_shuffle_ux_in_place,
    build_trial_shuffle_plan,
    paired_bootstrap_closeness_to_static_gain,
    paired_bootstrap_drop_test,
    run_dms_session_with_intervention,
)
from src.experiments.silent_memory.shared.population_dms import (
    bootstrap_decode_accuracy,
    build_class_index,
    encode_images,
    extract_delay_features,
    generate_balanced_dms_trial_specs,
    load_model_and_encoder,
    validate_trial_specs,
)
from figure_utils_common import (
    get_paper_color_map,
    PUBLICATION_ANNOTATION_FONT_SIZE,
    PUBLICATION_LINE_WIDTH,
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
    validate_required_columns,
)
from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from paper_plot_style import DEFAULT_SUBPLOT_ADJUST, PANEL_LABEL_FONT_SIZE, apply_paper_style
CONDITION_A_DYNAMIC_BASE = "A_dynamic_base"
CONDITION_B_TRIAL_SHUFFLE_UX = "B_trial_shuffle_ux"
CONDITION_B_PURE_UX_ONLY_SHUFFLE = "B_pure_ux_only_shuffle"
CONDITION_C_STATIC_FROZEN = "C_static_frozen"

CONDITION_ORDER = [
    CONDITION_A_DYNAMIC_BASE,
    CONDITION_B_TRIAL_SHUFFLE_UX,
    CONDITION_B_PURE_UX_ONLY_SHUFFLE,
    CONDITION_C_STATIC_FROZEN,
]

CONDITION_LABELS = {
    CONDITION_A_DYNAMIC_BASE: "Dynamic",
    CONDITION_B_TRIAL_SHUFFLE_UX: "Shuffle u/x",
    CONDITION_B_PURE_UX_ONLY_SHUFFLE: "Shuffle u/x only",
    CONDITION_C_STATIC_FROZEN: "Static frozen",
}

ENGRAM_DECODE_METRICS_PATH = Path("results/engram_decode_experiment/engram_decode_metrics.csv")
ENGRAM_LAYER_ORDER = ["l1", "l2", "l3"]
ENGRAM_LAYER_LABELS = {"l1": "layer1", "l2": "layer2", "l3": "layer3"}
NEUTRAL_GRAY = "#7F7F7F"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snapshot_ux_delay_state(
    net,
    sample_spikes: torch.Tensor,
    delay_steps: int,
    stsp_mode: str = "dynamic",
) -> Dict[str, Dict[str, np.ndarray]]:
    batch_size, t_sample, c, h, w = sample_spikes.shape
    prepare_network_state(net, batch_size, c, h, w)
    zero_input = torch.zeros((batch_size, c, h, w), device=sample_spikes.device)
    current_time = 0

    def step_network(input_t: torch.Tensor) -> None:
        nonlocal current_time
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())
        net.layer3.forward_step(s2_p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        current_time += 1

    for t in range(t_sample):
        step_network(sample_spikes[:, t, ...])
    for _ in range(delay_steps):
        step_network(zero_input)

    out = snapshot_ux_state(net, batch_size=batch_size)
    for layer_key in LAYER_KEYS:
        out[layer_key]["ux"] = np.concatenate(
            [out[layer_key]["u"], out[layer_key]["x"]],
            axis=1,
        ).astype(np.float32, copy=False)
    return out


def collect_delay_decode_features(
    net,
    encoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    ux_feature_buf: Dict[str, List[np.ndarray]] = {layer_key: [] for layer_key in LAYER_KEYS}
    spike_feature_buf: Dict[str, List[np.ndarray]] = {layer_key: [] for layer_key in LAYER_KEYS}

    for start in tqdm(range(0, len(df_specs), batch_size), desc="Collect delay features"):
        batch = df_specs.iloc[start : start + batch_size].copy()
        sample_imgs = torch.stack([dataset[int(i)][0] for i in batch["sample_index"].tolist()], dim=0).to(device)
        probe_imgs = torch.stack([dataset[int(i)][0] for i in batch["probe_index"].tolist()], dim=0).to(device)
        sample_spikes = encode_images(encoder, sample_imgs, spec.sample_steps)
        probe_spikes = encode_images(encoder, probe_imgs, spec.probe_steps)

        with torch.no_grad():
            ux_snapshot = snapshot_ux_delay_state(
                net=net,
                sample_spikes=sample_spikes,
                delay_steps=spec.delay_steps,
                stsp_mode="dynamic",
            )
            spike_trace = net.forward_dms_spike_trace_session(
                sample_spikes=sample_spikes,
                probe_spikes=probe_spikes,
                delay_steps=spec.delay_steps,
                stsp_mode="dynamic",
                phase_reset=True,
            )

        phase_slices = spike_trace["phase_slices"]
        for layer_key in LAYER_KEYS:
            ux_feature_buf[layer_key].append(ux_snapshot[layer_key]["ux"])
            spike_feature_buf[layer_key].append(
                extract_delay_features(spike_trace[f"{layer_key}_spikes"], phase_slices=phase_slices)
            )

    ux_features = {
        layer_key: np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for layer_key, chunks in ux_feature_buf.items()
    }
    spike_features = {
        layer_key: np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for layer_key, chunks in spike_feature_buf.items()
    }
    return ux_features, spike_features


def _build_delay_decode_table(
    features_by_layer: Mapping[str, np.ndarray],
    labels: np.ndarray,
    delay_ms: float,
    num_classes: int,
    decode_splits: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    chance_level = 1.0 / float(num_classes)
    rows: List[Dict[str, float]] = []

    for layer_idx, layer_key in enumerate(LAYER_KEYS):
        features = np.asarray(features_by_layer[layer_key], dtype=np.float32)
        splits = build_stratified_splits(
            labels=labels,
            n_splits=decode_splits,
            test_ratio=0.3,
            seed=seed + 100 + layer_idx * 17,
        )
        acc = decode_accuracy_with_splits(
            x=features,
            y=labels,
            splits=splits,
            num_classes=num_classes,
        )
        boot = bootstrap_decode_accuracy(
            features=features,
            labels=labels,
            num_classes=num_classes,
            decode_splits=decode_splits,
            n_boot=n_boot,
            seed=seed + 1000 + layer_idx * 37,
        )
        rows.append(
            {
                "layer": layer_key,
                "delay_ms": float(delay_ms),
                "decode_acc": float(acc),
                "decode_acc_ci95_lower": float(boot["ci95_lower"]),
                "decode_acc_ci95_upper": float(boot["ci95_upper"]),
                "chance_level": float(chance_level),
                "p_one_sided_gt_chance": float(boot["p_one_sided_gt_chance"]),
                "n_trials": int(len(labels)),
                "n_boot": int(n_boot),
            }
        )
    return pd.DataFrame(rows)


def build_ux_delay_decode_table(
    ux_features_by_layer: Mapping[str, np.ndarray],
    labels: np.ndarray,
    delay_ms: float,
    num_classes: int,
    decode_splits: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    return _build_delay_decode_table(
        features_by_layer=ux_features_by_layer,
        labels=labels,
        delay_ms=delay_ms,
        num_classes=num_classes,
        decode_splits=decode_splits,
        n_boot=n_boot,
        seed=seed,
    )


def build_spike_delay_decode_table(
    spike_features_by_layer: Mapping[str, np.ndarray],
    labels: np.ndarray,
    delay_ms: float,
    num_classes: int,
    decode_splits: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    return _build_delay_decode_table(
        features_by_layer=spike_features_by_layer,
        labels=labels,
        delay_ms=delay_ms,
        num_classes=num_classes,
        decode_splits=decode_splits,
        n_boot=n_boot,
        seed=seed,
    )

def run_ux_shuffle_causality(
    net,
    encoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
    rng: random.Random,
    include_pure_ux_condition: bool,
) -> pd.DataFrame:
    all_records: List[Dict[str, int]] = []

    for start in tqdm(range(0, len(df_specs), batch_size), desc="Run shuffle causality"):
        batch = df_specs.iloc[start : start + batch_size].copy()
        bsz = len(batch)
        trial_ids = batch["trial_id"].to_numpy(dtype=np.int64)
        sample_lbl_np = batch["sample_label"].to_numpy(dtype=np.int64)
        probe_lbl_np = batch["probe_label"].to_numpy(dtype=np.int64)

        sample_imgs = torch.stack([dataset[int(i)][0] for i in batch["sample_index"].tolist()], dim=0).to(device)
        probe_imgs = torch.stack([dataset[int(i)][0] for i in batch["probe_index"].tolist()], dim=0).to(device)
        sample_spikes = encode_images(encoder, sample_imgs, spec.sample_steps)
        probe_spikes = encode_images(encoder, probe_imgs, spec.probe_steps)

        donor_idx_b, plan_info = build_trial_shuffle_plan(
            sample_labels=sample_lbl_np,
            probe_labels=probe_lbl_np,
            rng=rng,
        )
        batch_meta = {
            "trial_id": trial_ids,
            "sample_label": sample_lbl_np,
            "probe_label": probe_lbl_np,
            "donor_batch_index": donor_idx_b,
        }

        def intervention_trial_shuffle(local_net, meta: Dict[str, np.ndarray]) -> Dict[str, int]:
            donor_idx = np.asarray(meta["donor_batch_index"], dtype=np.int64)
            apply_trial_shuffle_ux_in_place(local_net, donor_idx)
            return {
                "applied": 1,
                "n_self_swap": int(np.sum(donor_idx == np.arange(len(donor_idx), dtype=np.int64))),
            }

        condition_runs = [
            (CONDITION_A_DYNAMIC_BASE, "dynamic", None, False),
            (CONDITION_B_TRIAL_SHUFFLE_UX, "dynamic", intervention_trial_shuffle, False),
            (CONDITION_C_STATIC_FROZEN, "static_frozen", None, False),
        ]
        if include_pure_ux_condition:
            condition_runs.insert(2, (CONDITION_B_PURE_UX_ONLY_SHUFFLE, "dynamic", intervention_trial_shuffle, True))

        for condition_name, stsp_mode, intervention_fn, pure_ux_only in condition_runs:
            with torch.no_grad():
                out = run_dms_session_with_intervention(
                    net=net,
                    sample_spikes=sample_spikes,
                    probe_spikes=probe_spikes,
                    delay_steps=spec.delay_steps,
                    stsp_mode=stsp_mode,
                    intervention_fn=intervention_fn,
                    batch_meta=batch_meta,
                    pure_ux_only=pure_ux_only,
                )

            pred_probe = out["prediction_probe"].detach().cpu().long().numpy()
            fire_t_probe = out["first_fire_t_probe"].detach().cpu().long().numpy()
            ux_restore_ok = int(out["ux_restore_ok"].item()) if "ux_restore_ok" in out else 1
            non_ux_reset = int(out["non_ux_state_reset_applied"].item()) if "non_ux_state_reset_applied" in out else 0

            for i in range(bsz):
                donor_batch_i = int(donor_idx_b[i])
                donor_trial_id = int(trial_ids[donor_batch_i])
                donor_sample_label = int(sample_lbl_np[donor_batch_i])
                sample_label = int(sample_lbl_np[i])
                probe_label = int(probe_lbl_np[i])
                pred_label = int(pred_probe[i])
                donor_is_distinct = int(donor_sample_label != sample_label)
                all_records.append(
                    {
                        "trial_id": int(trial_ids[i]),
                        "condition": condition_name,
                        "stsp_mode": stsp_mode,
                        "sample_label": sample_label,
                        "probe_label": probe_label,
                        "donor_trial_id": donor_trial_id,
                        "donor_sample_label": donor_sample_label,
                        "donor_is_distinct": donor_is_distinct,
                        "is_self_swap": int(donor_batch_i == i),
                        "donor_probe_conflict": int(donor_sample_label == probe_label),
                        "pred_label": pred_label,
                        "is_correct": int(pred_label == probe_label),
                        "is_silent": int(pred_label == -1),
                        "first_fire_t_probe": int(fire_t_probe[i]),
                        "pure_ux_only": int(pure_ux_only),
                        "non_ux_state_reset_applied": int(non_ux_reset if pure_ux_only else 0),
                        "ux_restore_ok": int((not pure_ux_only) or (ux_restore_ok == 1)),
                        "shuffle_ux_applied": int(intervention_fn is not None),
                        "pred_is_original_sample": int(pred_label == sample_label),
                        "pred_is_donor_sample": int(pred_label == donor_sample_label),
                        "pred_is_donor_shifted_memory": int(
                            (pred_label == donor_sample_label) and (donor_is_distinct == 1)
                        ),
                        "used_relaxed_shuffle_rule": int(plan_info["used_relaxed_rule"]),
                    }
                )

        if plan_info["used_relaxed_rule"] == 1:
            print(
                f"[Warn] Batch start={start}: relaxed no-self donor rule; "
                f"self-swaps={plan_info['n_self_swap']}/{bsz}"
            )

    return pd.DataFrame(all_records).sort_values(["trial_id", "condition"]).reset_index(drop=True)


def validate_trial_level_table(
    df_trials: pd.DataFrame,
    include_pure_ux_condition: bool,
) -> None:
    required_columns = [
        "trial_id",
        "condition",
        "sample_label",
        "probe_label",
        "donor_sample_label",
        "pred_label",
        "is_correct",
        "is_silent",
        "first_fire_t_probe",
        "pure_ux_only",
        "non_ux_state_reset_applied",
    ]
    validate_required_columns(df_trials, required_columns)

    expected_conditions = [
        CONDITION_A_DYNAMIC_BASE,
        CONDITION_B_TRIAL_SHUFFLE_UX,
        CONDITION_C_STATIC_FROZEN,
    ]
    if include_pure_ux_condition:
        expected_conditions.insert(2, CONDITION_B_PURE_UX_ONLY_SHUFFLE)

    count_per_trial = df_trials.groupby("trial_id").size()
    if not bool((count_per_trial == len(expected_conditions)).all()):
        raise ValueError("Each trial_id must appear exactly once per formal condition.")

    for col in ["sample_label", "probe_label", "donor_trial_id", "donor_sample_label", "donor_is_distinct", "is_self_swap"]:
        uniq = df_trials.groupby("trial_id")[col].nunique()
        if not bool((uniq == 1).all()):
            raise ValueError(f"{col} is not stable across conditions for some trials.")

    if bool((df_trials["donor_probe_conflict"] == 1).any()):
        raise ValueError("Found donor_sample_label == probe_label in trial_level.csv.")

    sub_shuffle = df_trials[df_trials["condition"] == CONDITION_B_TRIAL_SHUFFLE_UX]
    if len(sub_shuffle) == 0:
        raise ValueError("Missing B_trial_shuffle_ux condition.")
    if bool((sub_shuffle["shuffle_ux_applied"] != 1).any()):
        raise ValueError("B_trial_shuffle_ux must report shuffle_ux_applied=1.")

    if include_pure_ux_condition:
        sub_pure = df_trials[df_trials["condition"] == CONDITION_B_PURE_UX_ONLY_SHUFFLE]
        if len(sub_pure) == 0:
            raise ValueError("Missing B_pure_ux_only_shuffle condition.")
        if bool((sub_pure["pure_ux_only"] != 1).any()):
            raise ValueError("Pure-u/x condition must report pure_ux_only=1.")
        if bool((sub_pure["non_ux_state_reset_applied"] != 1).any()):
            raise ValueError("Pure-u/x condition must report non_ux_state_reset_applied=1.")
        if bool((sub_pure["ux_restore_ok"] != 1).any()):
            raise ValueError("Pure-u/x condition contains ux_restore_ok=0 rows.")


def _condition_indicator(
    df_trials: pd.DataFrame,
    condition: str,
    column: str,
) -> np.ndarray:
    sub = df_trials[df_trials["condition"] == condition].sort_values("trial_id").reset_index(drop=True)
    return sub[column].to_numpy(dtype=np.float64)


def summarize_shuffle_results(
    df_trials: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    for condition in CONDITION_ORDER:
        sub = df_trials[df_trials["condition"] == condition].copy()
        if len(sub) == 0:
            continue
        rows.append(
            {
                "condition": condition,
                "n_trials": int(len(sub)),
                "acc_probe": 100.0 * float(sub["is_correct"].mean()),
                "bias_original_sample": 100.0 * float(sub["pred_is_original_sample"].mean()),
                "bias_donor_shifted_memory": 100.0 * float(sub["pred_is_donor_shifted_memory"].mean()),
                "bias_probe": 100.0 * float((sub["pred_label"] == sub["probe_label"]).mean()),
                "bias_silent": 100.0 * float(sub["is_silent"].mean()),
                "abs_rate_pred_original_sample": 100.0 * float(sub["pred_is_original_sample"].mean()),
                "abs_rate_pred_donor_sample": 100.0 * float(sub["pred_is_donor_sample"].mean()),
                "abs_rate_pred_probe": 100.0 * float((sub["pred_label"] == sub["probe_label"]).mean()),
                "pure_ux_only": int(sub["pure_ux_only"].iloc[0]),
                "non_ux_state_reset_rate": 100.0 * float(sub["non_ux_state_reset_applied"].mean()),
            }
        )

    df_summary = pd.DataFrame(rows)
    summary_lookup = {row["condition"]: row for _, row in df_summary.iterrows()}
    if CONDITION_A_DYNAMIC_BASE not in summary_lookup or CONDITION_C_STATIC_FROZEN not in summary_lookup:
        raise ValueError("Shuffle summary requires A_dynamic_base and C_static_frozen.")

    boot_rows: List[Dict[str, float]] = []
    ind_a_original = _condition_indicator(df_trials, CONDITION_A_DYNAMIC_BASE, "pred_is_original_sample")
    ind_a_donor_shift = _condition_indicator(df_trials, CONDITION_A_DYNAMIC_BASE, "pred_is_donor_shifted_memory")
    ind_c_original = _condition_indicator(df_trials, CONDITION_C_STATIC_FROZEN, "pred_is_original_sample")

    for idx, condition in enumerate([CONDITION_B_TRIAL_SHUFFLE_UX, CONDITION_B_PURE_UX_ONLY_SHUFFLE]):
        if condition not in summary_lookup:
            continue

        ind_b_original = _condition_indicator(df_trials, condition, "pred_is_original_sample")
        ind_b_donor_shift = _condition_indicator(df_trials, condition, "pred_is_donor_shifted_memory")

        boot_original_drop = paired_bootstrap_drop_test(
            indicator_a=ind_a_original,
            indicator_b=ind_b_original,
            n_boot=n_boot,
            seed=seed + 31 + idx * 100,
        )
        boot_donor_gain = paired_bootstrap_drop_test(
            indicator_a=ind_b_donor_shift,
            indicator_b=ind_a_donor_shift,
            n_boot=n_boot,
            seed=seed + 47 + idx * 100,
        )
        boot_collapse_gain = paired_bootstrap_closeness_to_static_gain(
            indicator_a=ind_a_original,
            indicator_b=ind_b_original,
            indicator_c=ind_c_original,
            n_boot=n_boot,
            seed=seed + 61 + idx * 100,
        )

        boot_rows.extend(
            [
                {
                    "condition": condition,
                    "test_name": "A_minus_condition_original_sample_rate",
                    "obs_diff_rate": float(boot_original_drop["obs_diff"]),
                    "ci95_lower": float(boot_original_drop["ci95_lower"]),
                    "ci95_upper": float(boot_original_drop["ci95_upper"]),
                    "p_one_sided_nonpositive": float(boot_original_drop["p_one_sided_nonpositive"]),
                },
                {
                    "condition": condition,
                    "test_name": "condition_minus_A_donor_shifted_memory_rate",
                    "obs_diff_rate": float(boot_donor_gain["obs_diff"]),
                    "ci95_lower": float(boot_donor_gain["ci95_lower"]),
                    "ci95_upper": float(boot_donor_gain["ci95_upper"]),
                    "p_one_sided_nonpositive": float(boot_donor_gain["p_one_sided_nonpositive"]),
                },
                {
                    "condition": condition,
                    "test_name": "collapse_gain_vs_static",
                    "obs_diff_rate": float(boot_collapse_gain["obs_gain"]),
                    "ci95_lower": float(boot_collapse_gain["ci95_lower"]),
                    "ci95_upper": float(boot_collapse_gain["ci95_upper"]),
                    "p_one_sided_nonpositive": float(boot_collapse_gain["p_one_sided_nonpositive"]),
                },
            ]
        )

    return df_summary, pd.DataFrame(boot_rows)


def load_engram_decode_metrics(metrics_path: Path) -> pd.DataFrame:
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(
            "Figure 3 requires the external decode artifact at "
            f"{metrics_path}. Run engram_decode.py first to generate engram_decode_metrics.csv."
        )
    df = pd.read_csv(metrics_path)
    required_columns = ["layer", "delay_ms", "acc", "acc_ci_low", "acc_ci_high"]
    validate_required_columns(df, required_columns)
    layer_values = [str(value) for value in df["layer"].tolist()]
    invalid_layers = sorted(set(layer_values) - set(ENGRAM_LAYER_ORDER))
    if invalid_layers:
        raise ValueError(f"Unexpected engram decode layers: {invalid_layers}")
    df = df[df["layer"].isin(ENGRAM_LAYER_ORDER)].copy()
    df["layer"] = pd.Categorical(df["layer"], categories=ENGRAM_LAYER_ORDER, ordered=True)
    return df.sort_values(["layer", "delay_ms"], kind="stable").reset_index(drop=True)


def build_metrics_summary(
    df_engram_decode: pd.DataFrame,
    df_shuffle_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in df_engram_decode.iterrows():
        for metric in ["acc", "acc_ci_low", "acc_ci_high"]:
            rows.append(
                {
                    "section": "engram_decode",
                    "group": f"{row['layer']}|delay_{int(row['delay_ms'])}ms",
                    "metric": metric,
                    "value": float(row[metric]),
                }
            )
    for _, row in df_shuffle_summary.iterrows():
        for metric in [
            "acc_probe",
            "abs_rate_pred_original_sample",
            "abs_rate_pred_donor_sample",
        ]:
            rows.append(
                {
                    "section": "shuffle_summary",
                    "group": row["condition"],
                    "metric": metric,
                    "value": float(row[metric]),
                }
            )
    return pd.DataFrame(rows)


def p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."

def plot_engram_decode_vs_delay(
    ax: Axes,
    df_engram_decode: pd.DataFrame,
    num_classes: int,
    color_map: Mapping[str, str],
) -> None:
    validate_required_columns(df_engram_decode, ["layer", "delay_ms", "acc", "acc_ci_low", "acc_ci_high"])
    layer_colors = {
        "l1": color_map["sample_aligned"],
        "l2": color_map["ping_branch"],
        "l3": color_map["dynamic"],
    }
    for layer_name in ENGRAM_LAYER_ORDER:
        sub = df_engram_decode[df_engram_decode["layer"] == layer_name].copy().sort_values("delay_ms")
        if sub.empty:
            continue
        x = sub["delay_ms"].to_numpy(dtype=np.float64)
        y = sub["acc"].to_numpy(dtype=np.float64)
        y_lo = sub["acc_ci_low"].to_numpy(dtype=np.float64)
        y_hi = sub["acc_ci_high"].to_numpy(dtype=np.float64)
        color = layer_colors[layer_name]
        ax.plot(x, y, marker="o", linewidth=PUBLICATION_LINE_WIDTH, color=color, label=ENGRAM_LAYER_LABELS[layer_name])
        ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Decoding accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.4)


def plot_shuffle_summary_panel(
    ax_left: Axes,
    ax_right: Axes,
    df_shuffle_summary: pd.DataFrame,
    color_map: Mapping[str, str],
) -> None:
    ordered_conditions = [
        CONDITION_A_DYNAMIC_BASE,
        CONDITION_B_TRIAL_SHUFFLE_UX,
        CONDITION_C_STATIC_FROZEN,
    ]
    summary = df_shuffle_summary.set_index("condition").loc[ordered_conditions].reset_index()
    x = np.arange(len(summary))

    left_colors = [color_map["static"], color_map["dynamic"], NEUTRAL_GRAY]
    for idx, (_, row) in enumerate(summary.iterrows()):
        bar = ax_left.bar(
            x[idx],
            float(row["acc_probe"]),
            width=0.6,
            color=left_colors[idx],
            edgecolor="#222222",
        )
        value = float(row["acc_probe"])
        ax_left.text(
            bar[0].get_x() + bar[0].get_width() / 2.0,
            value + 2.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
            color="#222222",
        )
    ax_left.set_xticks(x, [CONDITION_LABELS[cond] for cond in summary["condition"].tolist()])
    ax_left.set_ylabel("Probe accuracy (%)")
    ax_left.set_ylim(0.0, 100.0)
    ax_left.grid(False)

    width = 0.34
    ax_right.bar(
        x - width / 2.0,
        summary["abs_rate_pred_original_sample"].to_numpy(dtype=np.float64),
        width=width,
        color=color_map["static"],
        edgecolor="#222222",
        label="Pred = original sample",
    )
    ax_right.bar(
        x + width / 2.0,
        summary["abs_rate_pred_donor_sample"].to_numpy(dtype=np.float64),
        width=width,
        color=color_map["dynamic"],
        edgecolor="#222222",
        label="Pred = donor sample",
    )
    ax_right.set_xticks(x, [CONDITION_LABELS[cond] for cond in summary["condition"].tolist()])
    ax_right.set_ylabel("Absolute rate (%)")
    max_target = max(
        float(summary["abs_rate_pred_original_sample"].max()),
        float(summary["abs_rate_pred_donor_sample"].max()),
        0.1,
    )
    ax_right.set_ylim(0.0, min(10.0, max_target + 1.5))
    ax_right.legend(loc="upper right", frameon=False, handlelength=1.5)
    ax_right.grid(False)


def create_figure_main(
    df_engram_decode: pd.DataFrame,
    df_shuffle_summary: pd.DataFrame,
    num_classes: int,
) -> Figure:
    apply_paper_style()
    color_map = get_paper_color_map()
    fig = plt.figure(figsize=(15.0, 5.0))
    outer = fig.add_gridspec(1, 3, wspace=0.35)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b_left = fig.add_subplot(outer[0, 1])
    ax_b_right = fig.add_subplot(outer[0, 2])

    plot_engram_decode_vs_delay(ax_a, df_engram_decode, num_classes=num_classes, color_map=color_map)
    plot_shuffle_summary_panel(ax_b_left, ax_b_right, df_shuffle_summary, color_map=color_map)

    ax_a.text(-0.19, 1.08, "A", transform=ax_a.transAxes, fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold")
    ax_b_left.text(-0.15, 1.05, "B", transform=ax_b_left.transAxes, fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold")
    ax_b_right.text(-0.15, 1.05, "C", transform=ax_b_right.transAxes, fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold")
    fig.subplots_adjust(**DEFAULT_SUBPLOT_ADJUST)
    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Figure 3 mechanism: u/x retention, causality, and donor-shift.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--save-dir", type=str, default="results/fig3_ux_mechanism")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay-ms", type=float, default=500.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--decode-splits", type=int, default=5)
    parser.add_argument("--num-boot", type=int, default=5000)
    parser.add_argument("--include-pure-ux-condition", action=argparse.BooleanOptionalAction, default=True)
    return parser

def main() -> None:
    args = build_argparser().parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_classes < 3:
        raise ValueError("--num-classes must be >= 3")
    if args.decode_splits <= 0:
        raise ValueError("--decode-splits must be positive")
    if args.num_boot <= 0:
        raise ValueError("--num-boot must be positive")

    seed_everything(args.seed)
    apply_paper_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=float(args.sample_ms),
        delay_ms=float(args.delay_ms),
        probe_ms=float(args.probe_ms),
    )
    for name, steps in [
        ("sample", spec.sample_steps),
        ("delay", spec.delay_steps),
        ("probe", spec.probe_steps),
    ]:
        if steps <= 0:
            raise ValueError(f"{name} steps must be positive")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Init] Device: {device}")
    print(f"[Init] Save dir: {save_dir}")
    print(
        f"[Init] Timing | sample={spec.sample_steps} steps ({spec.sample_ms}ms), "
        f"delay={spec.delay_steps} steps ({spec.delay_ms}ms), "
        f"probe={spec.probe_steps} steps ({spec.probe_ms}ms)"
    )

    net, encoder = load_model_and_encoder(args.model_path, device, spec)
    _, _, test_loader = build_mnist_skeleton_loader(batch_size=1)
    dataset = test_loader.dataset
    class_index = build_class_index(dataset, num_classes=args.num_classes)
    rng = random.Random(args.seed)
    df_specs = generate_balanced_dms_trial_specs(
        class_index=class_index,
        num_trials=args.trials,
        num_classes=args.num_classes,
        rng=rng,
    )
    validate_trial_specs(df_specs, num_classes=args.num_classes)

    df_engram_decode = load_engram_decode_metrics(ENGRAM_DECODE_METRICS_PATH)

    df_trials = run_ux_shuffle_causality(
        net=net,
        encoder=encoder,
        dataset=dataset,
        df_specs=df_specs,
        spec=spec,
        batch_size=args.batch_size,
        device=device,
        rng=rng,
        include_pure_ux_condition=bool(args.include_pure_ux_condition),
    )
    validate_trial_level_table(df_trials, include_pure_ux_condition=bool(args.include_pure_ux_condition))
    df_shuffle_summary, df_shuffle_bootstrap = summarize_shuffle_results(
        df_trials=df_trials,
        n_boot=args.num_boot,
        seed=args.seed + 800,
    )
    df_metrics_summary = build_metrics_summary(
        df_engram_decode=df_engram_decode,
        df_shuffle_summary=df_shuffle_summary,
    )

    trial_level_columns = [
        "trial_id",
        "condition",
        "sample_label",
        "probe_label",
        "donor_sample_label",
        "pred_label",
        "is_correct",
        "is_silent",
        "first_fire_t_probe",
        "pure_ux_only",
        "non_ux_state_reset_applied",
        "stsp_mode",
        "donor_trial_id",
        "donor_is_distinct",
        "pred_is_original_sample",
        "pred_is_donor_sample",
        "pred_is_donor_shifted_memory",
        "is_self_swap",
        "shuffle_ux_applied",
        "ux_restore_ok",
    ]
    df_trial_export = df_trials[trial_level_columns].copy()

    trial_level_csv = save_tidy_csv(df_trial_export, save_dir / "trial_level.csv", sort_by=["trial_id", "condition"])
    engram_decode_csv = save_tidy_csv(df_engram_decode, save_dir / "metrics_engram_decode.csv", sort_by=["layer", "delay_ms"])
    shuffle_summary_csv = save_tidy_csv(df_shuffle_summary, save_dir / "metrics_shuffle_summary.csv", sort_by=["condition"])
    metrics_summary_csv = save_tidy_csv(df_metrics_summary, save_dir / "metrics_summary.csv", sort_by=["section", "group", "metric"])
    if len(df_shuffle_bootstrap) > 0:
        shuffle_bootstrap_csv = save_tidy_csv(
            df_shuffle_bootstrap,
            save_dir / "metrics_shuffle_bootstrap.csv",
            sort_by=["condition", "test_name"],
        )
    else:
        shuffle_bootstrap_csv = ""

    fig = create_figure_main(
        df_engram_decode=df_engram_decode,
        df_shuffle_summary=df_shuffle_summary,
        num_classes=args.num_classes,
    )
    figure_paths = save_figure_all_formats(fig, save_dir / "figure_main")
    plt.close(fig)

    run_config = {
        "model_path": str(args.model_path),
        "save_dir": str(save_dir),
        "trials": int(args.trials),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "num_classes": int(args.num_classes),
        "sample_ms": float(args.sample_ms),
        "delay_ms": float(args.delay_ms),
        "probe_ms": float(args.probe_ms),
        "decode_splits": int(args.decode_splits),
        "num_boot": int(args.num_boot),
        "include_pure_ux_condition": bool(args.include_pure_ux_condition),
        "output_files": {
            "engram_decode_source_csv": str(ENGRAM_DECODE_METRICS_PATH),
            "trial_level_csv": trial_level_csv,
            "metrics_engram_decode_csv": engram_decode_csv,
            "metrics_shuffle_summary_csv": shuffle_summary_csv,
            "metrics_summary_csv": metrics_summary_csv,
            "metrics_shuffle_bootstrap_csv": shuffle_bootstrap_csv,
            "figure_main_png": figure_paths["png"],
            "figure_main_pdf": figure_paths["pdf"],
            "figure_main_svg": figure_paths["svg"],
        },
    }
    save_run_config(run_config, save_dir)

    print(f"[Done] Saved: {trial_level_csv}")
    print(f"[Done] Saved: {engram_decode_csv}")
    print(f"[Done] Saved: {shuffle_summary_csv}")
    print(f"[Done] Saved: {metrics_summary_csv}")
    if shuffle_bootstrap_csv:
        print(f"[Done] Saved: {shuffle_bootstrap_csv}")
    print(f"[Done] Saved: {save_dir / 'figure_main.png'}")


if __name__ == "__main__":
    main()

