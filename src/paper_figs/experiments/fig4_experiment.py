from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.paper_figs.common.io import load_json, prepare_layout, save_csv, save_json, save_npz, write_artifact_manifest
from src.paper_figs.common.model_env import load_mnist_skeleton_dataset
from src.paper_figs.common.runtime import (
    build_common_parser,
    format_smoke_command,
    resolve_device_strict,
    run_python_module,
    seed_everything,
    setup_logger,
)
from src.paper_figs.common.sampling import coords_to_mask

FIGURE_ID = "fig4"
MODULE_NAME = "src.paper_figs.experiments.fig4_experiment"
DEFAULT_OUTPUT_DIR = str(Path("results") / "paper_figs" / FIGURE_ID)


def build_argparser():
    return build_common_parser(
        description="Fig4 paper experiment: overlap-defined support and Layer1 local competition.",
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )


def build_fig4_config(smoke: bool) -> dict[str, int]:
    # smoke experiment should be run in torch_env
    if smoke:
        return {"max_probes": 4, "max_pairs": 24, "batch_size": 8}
    return {"max_probes": 0, "max_pairs": 0, "batch_size": 0}


def run_overlap_support_backbone(args, layout, config: dict[str, int], logger) -> Path:
    stage_dir = layout.staging_path("dms_overlap_support")
    module_args = [
        "--model-path",
        args.model_path,
        "--output-dir",
        str(stage_dir),
        "--dataset-root",
        args.dataset_root,
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
    ]
    if int(config["max_probes"]) > 0:
        module_args.extend(["--max-probes", str(int(config["max_probes"]))])
    if int(config["max_pairs"]) > 0:
        module_args.extend(["--max-pairs", str(int(config["max_pairs"]))])
    if int(config["batch_size"]) > 0:
        module_args.extend(["--batch-size", str(int(config["batch_size"]))])
    run_python_module(
        "src.experiments.dms_overlap_ux_support_mechanism_experiment",
        module_args,
        logger=logger,
        cwd=Path.cwd(),
    )
    return stage_dir


def _foreground_mask(image_tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy()
    if image.ndim == 3:
        image = image[0]
    return (image > 0).astype(np.uint8)


def _load_stage_panel_a_gain_map(stage_dir: Path) -> dict[str, np.ndarray] | None:
    path = stage_dir / "data" / "l1_panel_a_preprobe_gain_map.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def _build_example_masks_npz(stage_dir: Path, dataset) -> dict[str, np.ndarray]:
    panel_a_gain_map = _load_stage_panel_a_gain_map(stage_dir)
    if panel_a_gain_map is not None:
        trial_id = int(np.asarray(panel_a_gain_map["trial_id"]).reshape(-1)[0])
        sample_id = int(np.asarray(panel_a_gain_map["sample_id"]).reshape(-1)[0])
        probe_id = int(np.asarray(panel_a_gain_map["probe_id"]).reshape(-1)[0])
        overlap_mask = np.asarray(panel_a_gain_map["overlap_mask"], dtype=np.uint8)
        probe_only_mask = np.asarray(panel_a_gain_map["probe_only_mask"], dtype=np.uint8)
    else:
        metadata = load_json(stage_dir / "data" / "pair_mask_metadata.json")
        trial = metadata["trials"][0]
        trial_id = int(trial["trial_id"])
        sample_id = int(trial["sample_id"])
        probe_id = int(trial["probe_id"])
        sample_mask = _foreground_mask(dataset[sample_id][0])
        overlap_mask = coords_to_mask(trial["overlap_coords"], shape=sample_mask.shape)
        probe_only_mask = coords_to_mask(trial["probe_only_coords"], shape=sample_mask.shape)
    sample_mask = _foreground_mask(dataset[sample_id][0])
    probe_mask = _foreground_mask(dataset[probe_id][0])
    arrays = {
        "trial_id": np.asarray([trial_id], dtype=np.int64),
        "sample_id": np.asarray([sample_id], dtype=np.int64),
        "probe_id": np.asarray([probe_id], dtype=np.int64),
        "sample_mask": sample_mask,
        "probe_mask": probe_mask,
        "overlap_mask": overlap_mask,
        "probe_only_mask": probe_only_mask,
    }
    if panel_a_gain_map is not None:
        arrays["ux_map_pre_dynamic"] = np.asarray(panel_a_gain_map["ux_map_pre_dynamic"], dtype=np.float32)
        if "ux_map_pre_static" in panel_a_gain_map:
            arrays["ux_map_pre_static"] = np.asarray(panel_a_gain_map["ux_map_pre_static"], dtype=np.float32)
    return arrays


def _build_local_exemplar_npz(df_pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    exemplar = df_pairs.iloc[df_pairs["winner_loser_contrast_shift"].abs().idxmax()]
    arrays = {}
    for column in [
        "trial_id",
        "winner_unit_idx",
        "loser_unit_idx",
        "winner_overlap_input_gain",
        "winner_probe_only_input_gain",
        "loser_overlap_input_gain",
        "loser_probe_only_input_gain",
        "winner_loser_contrast_shift",
        "contrast_dynamic",
        "contrast_static",
        "contrast_time_index",
    ]:
        arrays[column] = np.asarray([exemplar[column]])
    arrays["winner_group"] = np.asarray([str(exemplar["winner_group"])])
    arrays["loser_group"] = np.asarray([str(exemplar["loser_group"])])
    arrays["winner_transition"] = np.asarray([str(exemplar["winner_transition"])])
    arrays["loser_transition"] = np.asarray([str(exemplar["loser_transition"])])
    return arrays


def _load_stage_local_exemplar_trace(stage_dir: Path) -> dict[str, np.ndarray] | None:
    path = stage_dir / "data" / "l1_local_winner_loser_exemplar_trace.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def _load_stage_event_time_alignment(stage_dir: Path) -> dict[str, np.ndarray] | None:
    path = stage_dir / "data" / "l1_local_event_time_alignment.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def _build_panel_b_changed_only_composition(df_transition: pd.DataFrame) -> pd.DataFrame:
    groups = ["overlap_dominant", "probe_only_dominant"]
    work_df = df_transition.loc[df_transition["unit_group"].isin(groups)].copy()
    if work_df.empty:
        return pd.DataFrame(
            columns=[
                "trial_id",
                "unit_group",
                "n_units",
                "n_advance",
                "n_recruit",
                "n_loss",
                "n_unchanged",
                "changed_count",
                "changed_prevalence",
                "P_changed_advance",
                "P_changed_recruit",
                "P_changed_loss",
            ]
        )
    changed_count = (
        pd.to_numeric(work_df["n_advance"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work_df["n_recruit"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work_df["n_loss"], errors="coerce").fillna(0.0)
    )
    n_units = pd.to_numeric(work_df["n_units"], errors="coerce")
    work_df["changed_count"] = changed_count.astype(int)
    work_df["changed_prevalence"] = np.divide(
        changed_count.to_numpy(dtype=float),
        n_units.to_numpy(dtype=float),
        out=np.full(work_df.shape[0], np.nan, dtype=float),
        where=n_units.to_numpy(dtype=float) > 0.0,
    )
    denom = changed_count.to_numpy(dtype=float)
    for name in ("advance", "recruit", "loss"):
        numer = pd.to_numeric(work_df[f"n_{name}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        work_df[f"P_changed_{name}"] = np.divide(
            numer,
            denom,
            out=np.full(work_df.shape[0], np.nan, dtype=float),
            where=denom > 0.0,
        )
    cols = [
        "trial_id",
        "unit_group",
        "n_units",
        "n_advance",
        "n_recruit",
        "n_loss",
        "n_unchanged",
        "changed_count",
        "changed_prevalence",
        "P_changed_advance",
        "P_changed_recruit",
        "P_changed_loss",
    ]
    return work_df[cols].copy()


def build_fig4_summary(
    df_preprobe: pd.DataFrame,
    df_transition: pd.DataFrame,
    df_transition_changed: pd.DataFrame,
    df_chain: pd.DataFrame,
    df_local_pairs: pd.DataFrame,
    df_local_support: pd.DataFrame,
) -> dict[str, object]:
    dynamic_df = df_preprobe.loc[df_preprobe["model_type"] == "dynamic"].copy()
    support_delta = (dynamic_df["ux_overlap_pre"] - dynamic_df["ux_probe_only_pre"]).mean()
    transition_focus = df_transition.pivot_table(index="trial_id", columns="unit_group", values="P_advance", aggfunc="mean")
    loss_focus = df_transition.pivot_table(index="trial_id", columns="unit_group", values="P_loss", aggfunc="mean")
    changed_focus = df_transition_changed.pivot_table(index="trial_id", columns="unit_group", values="P_changed_advance", aggfunc="mean")
    changed_loss_focus = df_transition_changed.pivot_table(index="trial_id", columns="unit_group", values="P_changed_loss", aggfunc="mean")
    changed_prev_focus = df_transition_changed.pivot_table(index="trial_id", columns="unit_group", values="changed_prevalence", aggfunc="mean")
    contrast_shift = pd.to_numeric(df_local_pairs["winner_loser_contrast_shift"], errors="coerce").to_numpy(dtype=float)
    contrast_shift = contrast_shift[np.isfinite(contrast_shift)]
    chain_rates = {
        key: pd.to_numeric(df_chain[key], errors="coerce").to_numpy(dtype=float)
        for key in ("winner_pre_spike_boost", "winner_spikes_earlier", "loser_post_winner_suppressed", "full_chain_satisfied")
        if key in df_chain.columns
    }

    def _mean_or_nan(df: pd.DataFrame, column: str) -> float:
        if column not in df.columns:
            return float("nan")
        return float(df[column].mean())

    def _array_mean_or_nan(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size > 0 else float("nan")

    return {
        "figure": FIGURE_ID,
        "presentation_note": "Layer1 units are stratified by overlap-biased versus probe-only-biased feedforward drive as an intermediate grouping variable. This bias-conditioned grouping bridges overlap-defined latent support to later winner-loser competition without redefining the main competition actors.",
        "panel_a": {"mean_dynamic_overlap_minus_probeonly_ux_pre": float(support_delta)},
        "panel_b": {
            "description": "Transition composition is summarized among receiving-input units conditioned on whether their Layer1 feedforward drive is overlap-biased or probe-only-biased.",
            "role_note": "These bias-conditioned groups are used to test whether overlap-aligned latent support is preferentially converted into early probe-time advance or recruit outcomes.",
            "mean_P_advance_overlap_dominant": _mean_or_nan(transition_focus, "overlap_dominant"),
            "mean_P_advance_probe_dominant": _mean_or_nan(transition_focus, "probe_only_dominant"),
            "mean_P_loss_overlap_dominant": _mean_or_nan(loss_focus, "overlap_dominant"),
            "mean_P_loss_probe_dominant": _mean_or_nan(loss_focus, "probe_only_dominant"),
            "mean_P_changed_advance_overlap_dominant": _mean_or_nan(changed_focus, "overlap_dominant"),
            "mean_P_changed_advance_probe_dominant": _mean_or_nan(changed_focus, "probe_only_dominant"),
            "mean_P_changed_loss_overlap_dominant": _mean_or_nan(changed_loss_focus, "overlap_dominant"),
            "mean_P_changed_loss_probe_dominant": _mean_or_nan(changed_loss_focus, "probe_only_dominant"),
            "mean_changed_prevalence_overlap_dominant": _mean_or_nan(changed_prev_focus, "overlap_dominant"),
            "mean_changed_prevalence_probe_dominant": _mean_or_nan(changed_prev_focus, "probe_only_dominant"),
            "display_groups": {
                "all_units": "all receiving units",
                "overlap_dominant": "overlap-biased units",
                "probe_only_dominant": "probe-only-biased units",
            },
        },
        "panel_c": {
            "mean_winner_pre_spike_boost": _array_mean_or_nan(chain_rates.get("winner_pre_spike_boost", np.asarray([], dtype=float))),
            "mean_loser_post_winner_suppressed": _array_mean_or_nan(chain_rates.get("loser_post_winner_suppressed", np.asarray([], dtype=float))),
        },
        "panel_d": {
            "mean_full_chain_satisfied": _array_mean_or_nan(chain_rates.get("full_chain_satisfied", np.asarray([], dtype=float))),
            "mean_winner_spikes_earlier": _array_mean_or_nan(chain_rates.get("winner_spikes_earlier", np.asarray([], dtype=float))),
            "mean_winner_loser_contrast_shift": float(np.mean(contrast_shift)) if contrast_shift.size > 0 else float("nan"),
        },
        "panel_g": {"mean_local_winner_support_rate": float(df_local_support["local_winner_support_rate"].mean())},
    }


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    config = build_fig4_config(bool(args.smoke))
    seed_everything(int(args.seed))
    device = resolve_device_strict(args.device)
    layout = prepare_layout(args.output_dir)
    logger = setup_logger(layout.log_file(), f"paper_{FIGURE_ID}")
    smoke_command = format_smoke_command(MODULE_NAME, layout.root)

    logger.info("[Init] figure=%s", FIGURE_ID)
    logger.info("[Init] output_dir=%s", layout.root)
    logger.info("[Init] device=%s", device)
    logger.info("[Init] smoke=%s", bool(args.smoke))

    stage_dir = run_overlap_support_backbone(args, layout, config, logger)
    dataset = load_mnist_skeleton_dataset(args.dataset_root, split="test")

    df_trial_def = pd.read_csv(stage_dir / "data" / "pair_metadata.csv")[
        ["trial_id", "sample_label", "probe_label", "overlap_area", "probe_only_area", "overlap_quantile"]
    ].copy()
    df_preprobe = pd.read_csv(stage_dir / "data" / "preprobe_stsp_summary.csv")[
        ["trial_id", "model_type", "ux_overlap_pre", "ux_probe_only_pre", "mean_ux_on_overlap"]
    ].copy()
    df_transition = pd.read_csv(stage_dir / "data" / "l1_firing_transition_summary.csv")
    df_transition = df_transition.loc[
        df_transition["aggregation_scope"] == "per_trial",
        [
            "trial_id",
            "unit_group",
            "n_units",
            "n_advance",
            "n_recruit",
            "n_loss",
            "n_unchanged",
            "P_advance",
            "P_recruit",
            "P_loss",
            "P_unchanged",
        ],
    ].copy()
    df_transition_changed = _build_panel_b_changed_only_composition(df_transition)
    df_input_gain = pd.read_csv(stage_dir / "data" / "l1_input_source_gain_summary.csv")
    df_input_gain = df_input_gain.loc[
        df_input_gain["aggregation_scope"] == "per_trial",
        ["trial_id", "unit_group", "overlap_input_gain", "probe_only_input_gain", "input_selectivity_gain"],
    ].copy()
    df_loss_inh = pd.read_csv(stage_dir / "data" / "l1_loss_inhibition_summary.csv")
    df_loss_inh = df_loss_inh.loc[
        df_loss_inh["aggregation_scope"] == "per_trial",
        ["trial_id", "unit_group", "lost_spike_delta_inh", "n_lost_spike_units"],
    ].copy()
    df_local_pairs = pd.read_csv(stage_dir / "data" / "l1_local_winner_loser_pairs.csv")[
        [
            "trial_id",
            "winner_unit_idx",
            "loser_unit_idx",
            "winner_group",
            "winner_overlap_input_gain",
            "winner_loser_contrast_shift",
            "loser_group",
            "winner_transition",
            "loser_transition",
            "winner_probe_only_input_gain",
            "loser_overlap_input_gain",
            "loser_probe_only_input_gain",
            "contrast_dynamic",
            "contrast_static",
            "contrast_time_index",
        ]
    ].copy()
    df_chain = pd.read_csv(stage_dir / "data" / "l1_local_causal_chain_events.csv")[
        [
            "trial_id",
            "winner_unit_idx",
            "loser_unit_idx",
            "winner_group",
            "loser_group",
            "align_time_index",
            "winner_pre_spike_boost",
            "winner_spikes_earlier",
            "loser_post_winner_suppressed",
            "full_chain_satisfied",
            "winner_pre_spike_delta_v_mean",
            "loser_post_winner_delta_v_mean",
            "loser_pre_winner_inh_before_mean",
            "loser_post_winner_inh_before_mean",
            "loser_post_winner_inh_rise",
        ]
    ].copy()
    df_local_events = pd.read_csv(stage_dir / "data" / "l1_local_winner_support_summary.csv")
    if "aggregation_scope" in df_local_events.columns:
        df_local_events = df_local_events.loc[df_local_events["aggregation_scope"] == "loser_event"].copy()
    df_local_support = (
        df_local_events.groupby("trial_id", as_index=False)
        .agg(
            local_winner_support_rate=("supported", "mean"),
            n_loser_events=("supported", "size"),
            n_supported_events=("supported", "sum"),
        )
        .astype({"trial_id": int, "n_loser_events": int, "n_supported_events": int})
    )

    example_masks = _build_example_masks_npz(stage_dir, dataset)
    local_exemplar = _build_local_exemplar_npz(df_local_pairs)
    local_exemplar_trace = _load_stage_local_exemplar_trace(stage_dir)
    event_time_alignment = _load_stage_event_time_alignment(stage_dir)

    artifact_paths = {
        "run_config_json": str(
            save_json(
                {
                    "figure": FIGURE_ID,
                    "module_name": MODULE_NAME,
                    "model_path": str(Path(args.model_path).resolve()),
                    "dataset_root": str(Path(args.dataset_root).resolve()),
                    "device_requested": str(args.device),
                    "device_resolved": str(device),
                    "seed": int(args.seed),
                    "smoke": bool(args.smoke),
                    "smoke_note": "smoke experiment should be run in torch_env",
                    "smoke_command": smoke_command,
                    "config": config,
                },
                layout.root_file("run_config.json"),
            )
        ),
        "panel_a_trial_definition_csv": str(save_csv(df_trial_def, layout.data_file("panel_a_trial_definition.csv"), sort_by=["trial_id"])),
        "panel_b_preprobe_support_csv": str(save_csv(df_preprobe, layout.data_file("panel_b_preprobe_support.csv"), sort_by=["trial_id", "model_type"])),
        "panel_c_transition_summary_csv": str(save_csv(df_transition, layout.data_file("panel_c_transition_summary.csv"), sort_by=["trial_id", "unit_group"])),
        "panel_b_changed_only_composition_csv": str(save_csv(df_transition_changed, layout.data_file("panel_b_changed_only_composition.csv"), sort_by=["trial_id", "unit_group"])),
        "panel_d_input_gain_summary_csv": str(save_csv(df_input_gain, layout.data_file("panel_d_input_gain_summary.csv"), sort_by=["trial_id", "unit_group"])),
        "panel_e_loss_inhibition_summary_csv": str(save_csv(df_loss_inh, layout.data_file("panel_e_loss_inhibition_summary.csv"), sort_by=["trial_id", "unit_group"])),
        "panel_f_local_winner_loser_pairs_csv": str(save_csv(df_local_pairs, layout.data_file("panel_f_local_winner_loser_pairs.csv"), sort_by=["trial_id"])),
        "panel_d_causal_chain_events_csv": str(save_csv(df_chain, layout.data_file("panel_d_causal_chain_events.csv"), sort_by=["trial_id"])),
        "panel_g_local_winner_support_summary_csv": str(save_csv(df_local_support, layout.data_file("panel_g_local_winner_support_summary.csv"), sort_by=["trial_id"])),
        "panel_a_example_masks_npz": str(save_npz(layout.array_file("panel_a_example_masks.npz"), **example_masks)),
        "panel_f_local_competition_exemplar_npz": str(save_npz(layout.array_file("panel_f_local_competition_exemplar.npz"), **local_exemplar)),
    }
    if local_exemplar_trace is not None:
        artifact_paths["panel_f_local_competition_trace_npz"] = str(save_npz(layout.array_file("panel_f_local_competition_trace.npz"), **local_exemplar_trace))
    if event_time_alignment is not None:
        artifact_paths["panel_c_event_time_alignment_npz"] = str(save_npz(layout.array_file("panel_c_event_time_alignment.npz"), **event_time_alignment))
    summary = build_fig4_summary(df_preprobe, df_transition, df_transition_changed, df_chain, df_local_pairs, df_local_support)
    artifact_paths["summary_json"] = str(save_json({**summary, "saved_artifacts": artifact_paths}, layout.root_file("summary.json")))
    artifact_paths["artifact_manifest_json"] = str(write_artifact_manifest(layout, artifact_paths))
    logger.info("[Done] Fig4 artifacts saved.")


if __name__ == "__main__":
    main()
