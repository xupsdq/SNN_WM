from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.ping_memory.shared.shuffle_ops import (
    apply_trial_shuffle_ux_in_place,
    build_trial_shuffle_plan,
    paired_bootstrap_drop_test,
)
from src.experiments.silent_memory.shared.population_dms import (
    build_class_index,
    encode_images,
    generate_balanced_dms_trial_specs,
    validate_trial_specs,
)
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import (
    compare_tensor_dict,
    reset_fast_state_in_place,
    reset_non_ux_state_preserve_current_ux_in_place,
    reset_stsp_to_baseline_in_place,
    run_monitored_dms_rollout,
)
from src.experiments.common.ping_common import LAYER_KEYS
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_run_config, save_summary_json
from src.experiments.common.runtime import resolve_device, seed_everything
from src.plotting.common.io import apply_publication_style, save_figure_all_formats, save_tidy_csv
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    ERROR_DESTINATION_COLORS,
    FIGSIZE_TWO_PANEL_WIDE,
    GRID_ALPHA,
    apply_standard_legend,
)

CONDITION_A_DYNAMIC_BASE = "A_dynamic_base"
CONDITION_B_TRIAL_SHUFFLE_UX = "B_trial_shuffle_ux"
CONDITION_B_PURE_UX_ONLY_SHUFFLE = "B_pure_ux_only_shuffle"
CONDITION_D_SPIKE_SILENCING = "D_spike_silencing"
CONDITION_E_MEMBRANE_RESET = "E_membrane_reset"
CONDITION_F_STSP_BASELINE_RESET = "F_stsp_baseline_reset"

CONDITION_ORDER = [
    CONDITION_A_DYNAMIC_BASE,
    CONDITION_B_TRIAL_SHUFFLE_UX,
    CONDITION_B_PURE_UX_ONLY_SHUFFLE,
    CONDITION_D_SPIKE_SILENCING,
    CONDITION_E_MEMBRANE_RESET,
    CONDITION_F_STSP_BASELINE_RESET,
]
DEFAULT_SAVE_DIR = "results/fig5_causal_substrate_dissociation"


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def delay_steps(self) -> int:
        return int(round((self.delay_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Causal dissociation between spikes, membrane reset, and STSP reset.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay-ms", type=float, default=500.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--num-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _encode_batch_specs(dataset, batch_df: pd.DataFrame, device: torch.device, encoder, spec: ExperimentSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    sample_imgs = torch.stack([dataset[int(i)][0] for i in batch_df["sample_index"].tolist()], dim=0).to(device)
    probe_imgs = torch.stack([dataset[int(i)][0] for i in batch_df["probe_index"].tolist()], dim=0).to(device)
    sample_spikes = encode_images(encoder, sample_imgs, spec.sample_steps)
    probe_spikes = encode_images(encoder, probe_imgs, spec.probe_steps)
    return sample_spikes, probe_spikes


def _is_fast_state_baseline(net, boundary_state: Mapping[str, Mapping[str, torch.Tensor]], atol: float = 1e-6) -> bool:
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        state = boundary_state[layer_key]
        if not torch.allclose(state["v_mem"], torch.full_like(state["v_mem"], layer.V_L), atol=atol, rtol=0.0):
            return False
        if not torch.allclose(state["g_e"], torch.zeros_like(state["g_e"]), atol=atol, rtol=0.0):
            return False
        if not torch.equal(state["res"], torch.zeros_like(state["res"])):
            return False
        if not torch.allclose(state["inh_trace"], torch.zeros_like(state["inh_trace"]), atol=atol, rtol=0.0):
            return False
    return True


def _is_stsp_baseline(net, boundary_state: Mapping[str, Mapping[str, torch.Tensor]], atol: float = 1e-6) -> bool:
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        state = boundary_state[layer_key]
        if "u" not in state or "x" not in state:
            return False
        if not torch.allclose(state["u"], torch.full_like(state["u"], float(layer.stsp_U)), atol=atol, rtol=0.0):
            return False
        if not torch.allclose(state["x"], torch.ones_like(state["x"]), atol=atol, rtol=0.0):
            return False
    return True


def _is_ux_preserved(
    before: Mapping[str, Mapping[str, torch.Tensor]],
    after: Mapping[str, Mapping[str, torch.Tensor]],
) -> bool:
    return all(
        compare_tensor_dict(before[layer_key], after[layer_key], keys=("u", "x"))
        for layer_key in LAYER_KEYS
    )


def _is_fast_state_preserved(
    before: Mapping[str, Mapping[str, torch.Tensor]],
    after: Mapping[str, Mapping[str, torch.Tensor]],
) -> bool:
    return all(
        compare_tensor_dict(before[layer_key], after[layer_key], keys=("v_mem", "g_e", "res", "inh_trace"))
        for layer_key in LAYER_KEYS
    )


def _delay_spikes_are_zero(
    state_traces: Mapping[str, Mapping[str, torch.Tensor]],
    phase_slices: Mapping[str, List[int]],
) -> bool:
    delay_start, delay_end = phase_slices["delay"]
    for layer_key in LAYER_KEYS:
        spikes = state_traces[layer_key]["spikes"][delay_start:delay_end]
        if bool(spikes.any().item()):
            return False
    return True


def _condition_indicator(df_trials: pd.DataFrame, condition: str, column: str) -> np.ndarray:
    sub = df_trials[df_trials["condition"] == condition].sort_values("trial_id").reset_index(drop=True)
    return sub[column].to_numpy(dtype=np.float64)


def _compute_error_bias_components(df_subset: pd.DataFrame) -> Dict[str, float]:
    err = df_subset[df_subset["pred_label"] != df_subset["probe_label"]].copy()
    if len(err) == 0:
        return {
            "n_error": 0,
            "error_rate": 0.0,
            "bias_original_sample": 0.0,
            "bias_donor_shifted_memory": 0.0,
            "bias_silent": 0.0,
            "bias_probe": 0.0,
            "bias_other_classes": 0.0,
        }
    pred = err["pred_label"].to_numpy(dtype=np.int64)
    sample = err["sample_label"].to_numpy(dtype=np.int64)
    probe = err["probe_label"].to_numpy(dtype=np.int64)
    donor = err["donor_sample_label"].to_numpy(dtype=np.int64)
    donor_distinct = err["donor_is_distinct"].to_numpy(dtype=np.int64)
    valid = pred >= 0
    return {
        "n_error": int(len(err)),
        "error_rate": 100.0 * float(len(err)) / float(len(df_subset)),
        "bias_original_sample": 100.0 * float(np.mean(pred == sample)),
        "bias_donor_shifted_memory": 100.0 * float(np.mean((pred == donor) & (donor_distinct == 1))),
        "bias_silent": 100.0 * float(np.mean(pred == -1)),
        "bias_probe": 100.0 * float(np.mean(pred == probe)),
        "bias_other_classes": 100.0 * float(np.mean(valid & (pred != sample) & (pred != donor) & (pred != probe))),
    }


def _build_error_destination_table(df_trials: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for condition in CONDITION_ORDER:
        err = df_trials[(df_trials["condition"] == condition) & (df_trials["pred_label"] != df_trials["probe_label"])].copy()
        denom = max(1, len(err))
        rows.extend(
            [
                {
                    "condition": condition,
                    "destination": "original_sample",
                    "rate_percent": 100.0 * float((err["pred_label"] == err["sample_label"]).mean()) if len(err) > 0 else 0.0,
                    "n_error": int(len(err)),
                },
                {
                    "condition": condition,
                    "destination": "donor_sample",
                    "rate_percent": 100.0
                    * float(((err["pred_label"] == err["donor_sample_label"]) & (err["donor_is_distinct"] == 1)).mean())
                    if len(err) > 0
                    else 0.0,
                    "n_error": int(len(err)),
                },
                {
                    "condition": condition,
                    "destination": "silent",
                    "rate_percent": 100.0 * float((err["pred_label"] == -1).mean()) if len(err) > 0 else 0.0,
                    "n_error": int(len(err)),
                },
                {
                    "condition": condition,
                    "destination": "other",
                    "rate_percent": 100.0
                    * float(
                        (
                            (err["pred_label"] >= 0)
                            & (err["pred_label"] != err["sample_label"])
                            & (err["pred_label"] != err["probe_label"])
                            & (err["pred_label"] != err["donor_sample_label"])
                        ).mean()
                    )
                    if len(err) > 0
                    else 0.0,
                    "n_error": int(len(err)),
                },
            ]
        )
        if len(err) == 0:
            continue
        total = sum(row["rate_percent"] for row in rows[-4:])
        if not np.isclose(total, 100.0, atol=1e-6):
            rows[-1]["rate_percent"] += 100.0 - total
    return pd.DataFrame(rows)


def plot_condition_summary(df_condition: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL_WIDE)
    ordered = df_condition.set_index("condition").reindex(CONDITION_ORDER).reset_index()
    labels = [str(v).replace("_", "\n") for v in ordered["condition"].tolist()]
    x = np.arange(len(ordered), dtype=np.float64)

    axes[0].bar(x, ordered["acc_probe"].to_numpy(dtype=np.float64), color=ERROR_DESTINATION_COLORS["original_sample"], edgecolor="black", alpha=ALPHA_BAR)
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylabel("Probe accuracy (%)")
    axes[0].set_title("Condition-level probe accuracy")

    axes[1].bar(x, ordered["sample_related_bias"].to_numpy(dtype=np.float64), color="#DD8452", edgecolor="black", alpha=ALPHA_BAR)
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel("Sample-related bias (%)")
    axes[1].set_title("Condition-level sample-related bias")

    fig.tight_layout()
    return fig


def plot_error_destination(df_error_dest: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    destinations = list(dict.fromkeys(df_error_dest["destination"].tolist()))
    pivot = (
        df_error_dest.pivot(index="condition", columns="destination", values="rate_percent")
        .reindex(CONDITION_ORDER)
        .fillna(0.0)
    )
    color_map = dict(ERROR_DESTINATION_COLORS)
    x = np.arange(len(pivot), dtype=np.float64)
    bottom = np.zeros(len(pivot), dtype=np.float64)
    for destination in destinations:
        vals = pivot[destination].to_numpy(dtype=np.float64)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=color_map.get(str(destination), "#999999"),
            edgecolor="black",
            alpha=ALPHA_BAR,
            label=str(destination).replace("_", " "),
        )
        bottom += vals
    ax.set_xticks(x, [str(v).replace("_", "\n") for v in pivot.index.tolist()], rotation=15)
    ax.set_ylabel("Error destination rate (%)")
    ax.set_title("Error destination composition by intervention")
    apply_standard_legend(ax, ncol=2)
    fig.tight_layout()
    return fig


def _summarize_condition_metrics(df_trials: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for condition in CONDITION_ORDER:
        sub = df_trials[df_trials["condition"] == condition].copy()
        rows.append(
            {
                "condition": condition,
                "n_trials": int(len(sub)),
                "acc_probe": 100.0 * float(sub["is_correct"].mean()),
                "sample_related_bias": 100.0 * float(sub["pred_is_original_sample"].mean()),
                "donor_shift_bias": 100.0 * float(sub["pred_is_donor_shifted_memory"].mean()),
                "abs_rate_pred_original_sample": 100.0 * float(sub["pred_is_original_sample"].mean()),
                "abs_rate_pred_donor_sample": 100.0 * float(sub["pred_is_donor_sample"].mean()),
                "abs_rate_pred_probe": 100.0 * float((sub["pred_label"] == sub["probe_label"]).mean()),
                "abs_rate_silent": 100.0 * float(sub["is_silent"].mean()),
                "delay_spike_zero_verified_rate": 100.0 * float(sub["delay_spike_zero_verified"].mean()),
                "membrane_reset_faststate_ok_rate": 100.0 * float(sub["membrane_reset_faststate_ok"].mean()),
                "membrane_reset_ux_preserved_rate": 100.0 * float(sub["membrane_reset_ux_preserved"].mean()),
                "stsp_reset_baseline_ok_rate": 100.0 * float(sub["stsp_reset_baseline_ok"].mean()),
                "stsp_reset_faststate_preserved_rate": 100.0 * float(sub["stsp_reset_faststate_preserved"].mean()),
                "pure_ux_only_faststate_ok_rate": 100.0 * float(sub["pure_ux_only_faststate_ok"].mean()),
                "pure_ux_only_ux_restore_ok_rate": 100.0 * float(sub["pure_ux_only_ux_restore_ok"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_bias_metrics(df_trials: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for condition in CONDITION_ORDER:
        row = _compute_error_bias_components(df_trials[df_trials["condition"] == condition].copy())
        row["condition"] = condition
        rows.append(row)
    return pd.DataFrame(rows)


def _build_paired_bootstrap_table(df_trials: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    def add_contrast(condition_a: str, condition_b: str, metric: str, base_seed: int) -> None:
        indicator_a = _condition_indicator(df_trials, condition_a, metric)
        indicator_b = _condition_indicator(df_trials, condition_b, metric)
        boot = paired_bootstrap_drop_test(indicator_a, indicator_b, n_boot=n_boot, seed=base_seed)
        rows.append(
            {
                "contrast": f"{condition_a}_minus_{condition_b}",
                "metric": metric,
                "obs_diff_rate": float(boot["obs_diff"]),
                "ci95_lower": float(boot["ci95_lower"]),
                "ci95_upper": float(boot["ci95_upper"]),
                "p_one_sided_nonpositive": float(boot["p_one_sided_nonpositive"]),
            }
        )

    add_contrast(CONDITION_A_DYNAMIC_BASE, CONDITION_B_TRIAL_SHUFFLE_UX, "pred_is_original_sample", seed + 11)
    add_contrast(CONDITION_B_TRIAL_SHUFFLE_UX, CONDITION_A_DYNAMIC_BASE, "pred_is_donor_shifted_memory", seed + 23)
    add_contrast(CONDITION_A_DYNAMIC_BASE, CONDITION_B_PURE_UX_ONLY_SHUFFLE, "pred_is_original_sample", seed + 31)
    add_contrast(CONDITION_B_PURE_UX_ONLY_SHUFFLE, CONDITION_A_DYNAMIC_BASE, "pred_is_donor_shifted_memory", seed + 43)

    add_contrast(CONDITION_A_DYNAMIC_BASE, CONDITION_D_SPIKE_SILENCING, "is_correct", seed + 101)
    add_contrast(CONDITION_A_DYNAMIC_BASE, CONDITION_D_SPIKE_SILENCING, "pred_is_original_sample", seed + 113)
    add_contrast(CONDITION_A_DYNAMIC_BASE, CONDITION_E_MEMBRANE_RESET, "is_correct", seed + 127)
    add_contrast(CONDITION_A_DYNAMIC_BASE, CONDITION_E_MEMBRANE_RESET, "pred_is_original_sample", seed + 139)
    add_contrast(CONDITION_A_DYNAMIC_BASE, CONDITION_F_STSP_BASELINE_RESET, "is_correct", seed + 151)
    add_contrast(CONDITION_A_DYNAMIC_BASE, CONDITION_F_STSP_BASELINE_RESET, "pred_is_original_sample", seed + 163)
    add_contrast(CONDITION_D_SPIKE_SILENCING, CONDITION_F_STSP_BASELINE_RESET, "is_correct", seed + 177)
    add_contrast(CONDITION_D_SPIKE_SILENCING, CONDITION_F_STSP_BASELINE_RESET, "pred_is_original_sample", seed + 189)
    add_contrast(CONDITION_E_MEMBRANE_RESET, CONDITION_F_STSP_BASELINE_RESET, "is_correct", seed + 201)
    add_contrast(CONDITION_E_MEMBRANE_RESET, CONDITION_F_STSP_BASELINE_RESET, "pred_is_original_sample", seed + 213)
    return pd.DataFrame(rows)


def _validate_trial_level(df_trials: pd.DataFrame) -> None:
    expected_conditions = set(CONDITION_ORDER)
    count_per_trial = df_trials.groupby("trial_id").size()
    if not bool((count_per_trial == len(CONDITION_ORDER)).all()):
        raise ValueError("Each trial_id must appear exactly once per condition.")
    found_conditions = set(df_trials["condition"].unique().tolist())
    if found_conditions != expected_conditions:
        raise ValueError(f"Unexpected conditions: {sorted(found_conditions)}")
    for col in ["sample_label", "probe_label", "donor_trial_id", "donor_sample_label", "donor_is_distinct", "is_self_swap"]:
        uniq = df_trials.groupby("trial_id")[col].nunique()
        if not bool((uniq == 1).all()):
            raise ValueError(f"{col} is not stable across conditions.")
    sub_shuffle = df_trials[df_trials["condition"] == CONDITION_B_TRIAL_SHUFFLE_UX]
    if bool((sub_shuffle["shuffle_ux_applied"] != 1).any()):
        raise ValueError("B_trial_shuffle_ux must report shuffle_ux_applied=1.")
    sub_pure = df_trials[df_trials["condition"] == CONDITION_B_PURE_UX_ONLY_SHUFFLE]
    if bool((sub_pure["pure_ux_only_faststate_ok"] != 1).any()):
        raise ValueError("B_pure_ux_only_shuffle must verify non-u/x fast-state reset.")
    if bool((sub_pure["pure_ux_only_ux_restore_ok"] != 1).any()):
        raise ValueError("B_pure_ux_only_shuffle must preserve shuffled u/x.")
    sub_silence = df_trials[df_trials["condition"] == CONDITION_D_SPIKE_SILENCING]
    if bool((sub_silence["delay_spike_zero_verified"] != 1).any()):
        raise ValueError("D_spike_silencing must keep all delay spikes at zero.")


def main() -> None:
    args = build_argparser().parse_args()
    if args.trials <= 0 or args.batch_size <= 0:
        raise ValueError("trials and batch-size must be positive")
    if args.num_classes < 3:
        raise ValueError("num-classes must be >= 3")
    if args.num_boot <= 0:
        raise ValueError("num-boot must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    spec = ExperimentSpec(
        dt=float(args.dt_ms * ms),
        sample_ms=float(args.sample_ms),
        delay_ms=float(args.delay_ms),
        probe_ms=float(args.probe_ms),
    )
    for name, steps in [("sample", spec.sample_steps), ("delay", spec.delay_steps), ("probe", spec.probe_steps)]:
        if steps <= 0:
            raise ValueError(f"{name} steps must be positive")

    layout = prepare_result_layout(args.save_dir)

    print(f"[Init] Device: {device}")
    print(f"[Init] Save dir: {layout.root}")
    print(
        f"[Init] Timing | sample={spec.sample_steps} steps, delay={spec.delay_steps} steps, "
        f"probe={spec.probe_steps} steps"
    )

    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(spec.sample_ms, spec.probe_ms, spec.delay_ms),
    )
    _, _, test_loader = build_mnist_skeleton_loader(
        root=args.dataset_root,
        batch_size=1,
        input_size=28,
    )
    dataset = test_loader.dataset
    class_index = build_class_index(dataset, num_classes=args.num_classes)
    df_specs = generate_balanced_dms_trial_specs(
        class_index=class_index,
        num_trials=args.trials,
        num_classes=args.num_classes,
        rng=random.Random(args.seed),
    )
    validate_trial_specs(df_specs, num_classes=args.num_classes)

    trial_rows: List[Dict[str, int]] = []
    batch_starts = range(0, len(df_specs), args.batch_size)
    rng = random.Random(args.seed)

    for start in tqdm(batch_starts, desc="Causal dissociation batches"):
        batch_df = df_specs.iloc[start:start + args.batch_size].copy().reset_index(drop=True)
        sample_spikes, probe_spikes = _encode_batch_specs(dataset, batch_df, device, encoder, spec)
        sample_lbl_np = batch_df["sample_label"].to_numpy(dtype=np.int64)
        probe_lbl_np = batch_df["probe_label"].to_numpy(dtype=np.int64)
        trial_ids = batch_df["trial_id"].to_numpy(dtype=np.int64)
        donor_idx, plan_info = build_trial_shuffle_plan(sample_lbl_np, probe_lbl_np, rng=rng)

        def intervention_shuffle(local_net, _ctx: Dict[str, object]) -> Dict[str, object]:
            apply_trial_shuffle_ux_in_place(local_net, donor_idx)
            return {"shuffle_ux_applied": 1}

        def intervention_pure_shuffle(local_net, ctx: Dict[str, object]) -> Dict[str, object]:
            apply_trial_shuffle_ux_in_place(local_net, donor_idx)
            record = reset_non_ux_state_preserve_current_ux_in_place(
                local_net,
                layer_input_shapes=ctx["layer_input_shapes"],
            )
            record["shuffle_ux_applied"] = 1
            return record

        def intervention_membrane_reset(local_net, _ctx: Dict[str, object]) -> Dict[str, object]:
            reset_fast_state_in_place(local_net)
            return {"membrane_reset_applied": 1}

        def intervention_stsp_reset(local_net, _ctx: Dict[str, object]) -> Dict[str, object]:
            reset_stsp_to_baseline_in_place(local_net)
            return {"stsp_reset_applied": 1}

        condition_specs = [
            (CONDITION_A_DYNAMIC_BASE, {"delay_mode": "normal"}),
            (CONDITION_B_TRIAL_SHUFFLE_UX, {"delay_mode": "normal", "before_probe_fn": intervention_shuffle}),
            (CONDITION_B_PURE_UX_ONLY_SHUFFLE, {"delay_mode": "normal", "before_probe_fn": intervention_pure_shuffle}),
            (CONDITION_D_SPIKE_SILENCING, {"delay_mode": "spike_silence"}),
            (CONDITION_E_MEMBRANE_RESET, {"delay_mode": "normal", "before_probe_fn": intervention_membrane_reset}),
            (CONDITION_F_STSP_BASELINE_RESET, {"delay_mode": "normal", "before_probe_fn": intervention_stsp_reset}),
        ]

        for condition_name, plan in condition_specs:
            with torch.no_grad():
                out = run_monitored_dms_rollout(
                    net=net,
                    sample_spikes=sample_spikes,
                    probe_spikes=probe_spikes,
                    delay_steps=spec.delay_steps,
                    stsp_mode="dynamic",
                    intervention_plan=plan,
                    record_state_names=("spikes",),
                )

            pred_probe = out["predictions"]["prediction_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
            fire_t_probe = out["predictions"]["first_fire_t_probe"].detach().cpu().numpy().astype(np.int64, copy=False)
            boundary_pre = out["boundary_states"]["pre_intervention"]
            boundary_post = out["boundary_states"]["post_intervention"]
            intervention_record = dict(out["intervention_record"])

            delay_zero_ok = int(
                condition_name != CONDITION_D_SPIKE_SILENCING
                or _delay_spikes_are_zero(out["state_traces"], out["phase_slices"])
            )
            membrane_fast_ok = int(
                condition_name != CONDITION_E_MEMBRANE_RESET or _is_fast_state_baseline(net, boundary_post)
            )
            membrane_ux_ok = int(
                condition_name != CONDITION_E_MEMBRANE_RESET or _is_ux_preserved(boundary_pre, boundary_post)
            )
            stsp_baseline_ok = int(
                condition_name != CONDITION_F_STSP_BASELINE_RESET or _is_stsp_baseline(net, boundary_post)
            )
            stsp_fast_ok = int(
                condition_name != CONDITION_F_STSP_BASELINE_RESET or _is_fast_state_preserved(boundary_pre, boundary_post)
            )
            pure_fast_ok = int(
                condition_name != CONDITION_B_PURE_UX_ONLY_SHUFFLE or _is_fast_state_baseline(net, boundary_post)
            )
            pure_ux_ok = int(intervention_record.get("ux_restore_ok", 1 if condition_name != CONDITION_B_PURE_UX_ONLY_SHUFFLE else 0))

            for row_idx in range(len(batch_df)):
                donor_batch_idx = int(donor_idx[row_idx])
                donor_trial_id = int(trial_ids[donor_batch_idx])
                donor_sample_label = int(sample_lbl_np[donor_batch_idx])
                sample_label = int(sample_lbl_np[row_idx])
                probe_label = int(probe_lbl_np[row_idx])
                pred_label = int(pred_probe[row_idx])
                donor_is_distinct = int(donor_sample_label != sample_label)
                trial_rows.append(
                    {
                        "trial_id": int(trial_ids[row_idx]),
                        "condition": condition_name,
                        "sample_label": sample_label,
                        "probe_label": probe_label,
                        "donor_trial_id": donor_trial_id,
                        "donor_sample_label": donor_sample_label,
                        "donor_is_distinct": donor_is_distinct,
                        "is_self_swap": int(donor_batch_idx == row_idx),
                        "pred_label": pred_label,
                        "is_correct": int(pred_label == probe_label),
                        "is_silent": int(pred_label == -1),
                        "first_fire_t_probe": int(fire_t_probe[row_idx]),
                        "pred_is_original_sample": int(pred_label == sample_label),
                        "pred_is_donor_sample": int(pred_label == donor_sample_label),
                        "pred_is_donor_shifted_memory": int((pred_label == donor_sample_label) and (donor_is_distinct == 1)),
                        "shuffle_ux_applied": int(intervention_record.get("shuffle_ux_applied", 0)),
                        "used_relaxed_shuffle_rule": int(plan_info["used_relaxed_rule"]),
                        "delay_spike_zero_verified": int(delay_zero_ok),
                        "membrane_reset_faststate_ok": int(membrane_fast_ok),
                        "membrane_reset_ux_preserved": int(membrane_ux_ok),
                        "stsp_reset_baseline_ok": int(stsp_baseline_ok),
                        "stsp_reset_faststate_preserved": int(stsp_fast_ok),
                        "pure_ux_only_faststate_ok": int(pure_fast_ok),
                        "pure_ux_only_ux_restore_ok": int(pure_ux_ok),
                    }
                )

        if plan_info["used_relaxed_rule"] == 1:
            print(
                f"[Warn] Batch start={start}: relaxed donor rule used; "
                f"self-swaps={plan_info['n_self_swap']}/{len(batch_df)}"
            )

    df_trials = pd.DataFrame(trial_rows).sort_values(["trial_id", "condition"], kind="stable").reset_index(drop=True)
    _validate_trial_level(df_trials)
    df_condition = _summarize_condition_metrics(df_trials)
    df_bias = _summarize_bias_metrics(df_trials)
    df_error_dest = _build_error_destination_table(df_trials)
    df_paired = _build_paired_bootstrap_table(df_trials, n_boot=args.num_boot, seed=args.seed + 1000)

    trial_specs_csv = save_tidy_csv(df_specs, layout.data_file("trial_specs.csv"), sort_by=["trial_id"])
    trial_csv = save_tidy_csv(df_trials, layout.data_file("trial_level.csv"), sort_by=["trial_id", "condition"])
    condition_csv = save_tidy_csv(df_condition, layout.data_file("metrics_condition_summary.csv"), sort_by=["condition"])
    bias_csv = save_tidy_csv(df_bias, layout.data_file("metrics_bias_summary.csv"), sort_by=["condition"])
    error_csv = save_tidy_csv(
        df_error_dest,
        layout.data_file("metrics_error_destination.csv"),
        sort_by=["condition", "destination"],
    )
    paired_csv = save_tidy_csv(
        df_paired,
        layout.data_file("metrics_paired_bootstrap.csv"),
        sort_by=["contrast", "metric"],
    )
    fig_condition = plot_condition_summary(df_condition)
    fig_condition_paths = save_figure_all_formats(fig_condition, layout.figure_base("condition_summary"))
    plt.close(fig_condition)
    fig_error = plot_error_destination(df_error_dest)
    fig_error_paths = save_figure_all_formats(fig_error, layout.figure_base("error_destination"))
    plt.close(fig_error)

    run_config_path = save_run_config(
        {
            "model_path": str(args.model_path),
            "dataset_root": str(args.dataset_root),
            "seed": int(args.seed),
            "device": str(device),
            "trials": int(args.trials),
            "batch_size": int(args.batch_size),
            "num_classes": int(args.num_classes),
            "timing_ms": {
                "sample": float(args.sample_ms),
                "delay": float(args.delay_ms),
                "probe": float(args.probe_ms),
                "dt": float(args.dt_ms),
            },
            "output_files": {
                "trial_specs_csv": trial_specs_csv,
                "trial_level_csv": trial_csv,
                "metrics_condition_summary_csv": condition_csv,
                "metrics_bias_summary_csv": bias_csv,
                "metrics_error_destination_csv": error_csv,
                "metrics_paired_bootstrap_csv": paired_csv,
                "figure_condition_summary_png": fig_condition_paths["png"],
                "figure_error_destination_png": fig_error_paths["png"],
            },
        },
        layout.root,
    )
    summary_path = save_summary_json(
        {
            "experiment": "causal_substrate_dissociation",
            "outputs": {
                "trial_specs_csv": str(trial_specs_csv),
                "trial_level_csv": str(trial_csv),
                "metrics_condition_summary_csv": str(condition_csv),
                "metrics_bias_summary_csv": str(bias_csv),
                "metrics_error_destination_csv": str(error_csv),
                "metrics_paired_bootstrap_csv": str(paired_csv),
                "figure_condition_summary_png": fig_condition_paths["png"],
                "figure_error_destination_png": fig_error_paths["png"],
            },
            "condition_order": CONDITION_ORDER,
        },
        layout.root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=causal_substrate_dissociation",
            f"save_dir={layout.root}",
            f"trial_specs_csv={trial_specs_csv}",
            f"trial_level_csv={trial_csv}",
            f"metrics_condition_summary_csv={condition_csv}",
            f"metrics_bias_summary_csv={bias_csv}",
            f"metrics_error_destination_csv={error_csv}",
            f"metrics_paired_bootstrap_csv={paired_csv}",
            f"figure_condition_summary_png={fig_condition_paths['png']}",
            f"figure_error_destination_png={fig_error_paths['png']}",
            f"summary_json={summary_path}",
            f"run_config_json={run_config_path}",
        ],
        layout.log_dir,
    )

    print(f"[Done] Saved: {trial_specs_csv}")
    print(f"[Done] Saved: {trial_csv}")
    print(f"[Done] Saved: {condition_csv}")
    print(f"[Done] Saved: {bias_csv}")
    print(f"[Done] Saved: {error_csv}")
    print(f"[Done] Saved: {paired_csv}")
    print(f"[Done] Saved: {fig_condition_paths['png']}")
    print(f"[Done] Saved: {fig_error_paths['png']}")
    print(f"[Done] Saved: {summary_path}")
    print(f"[Done] Saved: {run_config_path}")
    print(f"[Done] Saved: {run_log_path}")


if __name__ == "__main__":
    main()
