from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.paper_figs.common.backbones.fig3_backbone import run_fig3_mechanism_backbone
from src.paper_figs.common.io import load_json, prepare_layout, save_csv, save_json, save_npz, write_artifact_manifest
from src.paper_figs.common.runtime import (
    build_common_parser,
    format_smoke_command,
    resolve_device_strict,
    run_python_module,
    seed_everything,
    setup_logger,
)

FIGURE_ID = "fig3"
MODULE_NAME = "src.paper_figs.experiments.fig3_experiment"
DEFAULT_OUTPUT_DIR = str(Path("results") / "paper_figs" / FIGURE_ID)


def build_argparser():
    return build_common_parser(
        description="Fig3 paper experiment: similarity, overlap perturbation, and L3 reconstruction.",
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )


def build_fig3_config(smoke: bool) -> dict[str, int]:
    if smoke:
        return {
            "similarity_max_pairs": 200,
            "similarity_max_samples": 120,
            "similarity_batch_size": 32,
        }
    return {
        "similarity_max_pairs": 5000,
        "similarity_max_samples": 0,
        "similarity_batch_size": 128,
    }


def _first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of the expected paths exist: {', '.join(str(item) for item in candidates)}")


def run_similarity_bias_analysis(args, layout, config: dict[str, int], logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage_dir = layout.staging_path("similarity_bias")
    module_args = [
        "--model-path",
        args.model_path,
        "--dataset-root",
        args.dataset_root,
        "--output-dir",
        str(stage_dir),
        "--device",
        str(args.device),
        "--delay-ms",
        "500.0",
        "--sample-ms",
        "200.0",
        "--probe-ms",
        "100.0",
        "--num-bins",
        "4",
        "--max-pairs",
        str(int(config["similarity_max_pairs"])),
        "--batch-size",
        str(int(config["similarity_batch_size"])),
        "--repeats",
        "1",
        "--seed",
        str(int(args.seed)),
        "--skip-figures",
    ]
    if int(config["similarity_max_samples"]) > 0:
        module_args.extend(["--max-samples", str(int(config["similarity_max_samples"]))])
    run_python_module("src.experiments.similarity_bias_experiment", module_args, logger=logger, cwd=Path.cwd())

    bin_df = pd.read_csv(
        _first_existing_path(
            stage_dir / "metrics" / "bin_accuracy_summary.csv",
            stage_dir / "data" / "bin_accuracy_summary.csv",
        )
    ).rename(
        columns={
            "acc_dynamic": "probe_accuracy_dynamic",
            "acc_static": "probe_accuracy_static",
        }
    )
    bridge_json_path = stage_dir / "log" / "within_bin_overlap_summary.json"
    bridge_payload = load_json(bridge_json_path) if bridge_json_path.exists() else {}
    bridge_rows = [
        {
            "group": "low_overlap",
            "mean_similarity": bridge_payload.get("mean_similarity_low"),
            "mean_overlap": bridge_payload.get("mean_overlap_low"),
            "acc_drop": bridge_payload.get("acc_drop_low"),
            "sem_acc_drop": bridge_payload.get("sem_acc_drop_low"),
            "n_pairs": int(bridge_payload.get("n_low_overlap", 0)),
        },
        {
            "group": "high_overlap",
            "mean_similarity": bridge_payload.get("mean_similarity_high"),
            "mean_overlap": bridge_payload.get("mean_overlap_high"),
            "acc_drop": bridge_payload.get("acc_drop_high"),
            "sem_acc_drop": bridge_payload.get("sem_acc_drop_high"),
            "n_pairs": int(bridge_payload.get("n_high_overlap", 0)),
        },
    ]
    return (
        bin_df[["similarity_bin", "probe_accuracy_dynamic", "probe_accuracy_static", "acc_drop"]].copy(),
        pd.DataFrame(bridge_rows),
    )


def run_mechanism_backbone(args, device, logger):
    return run_fig3_mechanism_backbone(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        device=device,
        seed=int(args.seed),
        smoke=bool(args.smoke),
        logger=logger,
    )


def build_fig3_summary(sim_bridge_df: pd.DataFrame, overlap_summary: dict, l3_summary: dict) -> dict[str, object]:
    delta_row = overlap_summary.get("comparison", {})
    plus_row = l3_summary.get("overall", {})
    bridge_high = sim_bridge_df.loc[sim_bridge_df["group"] == "high_overlap"].iloc[0].to_dict()
    bridge_low = sim_bridge_df.loc[sim_bridge_df["group"] == "low_overlap"].iloc[0].to_dict()
    return {
        "figure": FIGURE_ID,
        "panel_a": {
            "high_overlap_acc_drop": bridge_high.get("acc_drop"),
            "low_overlap_acc_drop": bridge_low.get("acc_drop"),
            "high_overlap_mean_overlap": bridge_high.get("mean_overlap"),
            "low_overlap_mean_overlap": bridge_low.get("mean_overlap"),
        },
        "panel_b": {
            "delta_DPI_L3_overlap_minus_nonoverlap": delta_row.get("delta_DPI_L3_overlap_minus_nonoverlap"),
        },
        "panel_cd": {
            "mean_reconstruction_cosine_plus": plus_row.get("mean_reconstruction_cosine_plus"),
            "mean_reconstruction_cosine_minus": plus_row.get("mean_reconstruction_cosine_minus"),
            "direction_match_rate_plus": plus_row.get("direction_match_rate_plus"),
            "direction_match_rate_minus": plus_row.get("direction_match_rate_minus"),
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    config = build_fig3_config(bool(args.smoke))
    seed_everything(int(args.seed))
    device = resolve_device_strict(args.device)
    layout = prepare_layout(args.output_dir)
    logger = setup_logger(layout.log_file(), f"paper_{FIGURE_ID}")
    smoke_command = format_smoke_command(MODULE_NAME, layout.root)

    logger.info("[Init] figure=%s", FIGURE_ID)
    logger.info("[Init] output_dir=%s", layout.root)
    logger.info("[Init] device=%s", device)
    logger.info("[Init] smoke=%s", bool(args.smoke))

    df_similarity_bins, df_overlap_bridge = run_similarity_bias_analysis(args, layout, config, logger)
    mechanism = run_mechanism_backbone(args, device, logger)

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
                    "smoke_command": smoke_command,
                    "config": {
                        "panel_a": config,
                        "mechanism": mechanism.config,
                    },
                    "optimization": mechanism.stats,
                },
                layout.root_file("run_config.json"),
            )
        ),
        "panel_a_similarity_bin_accuracy_csv": str(save_csv(df_similarity_bins, layout.data_file("panel_a_similarity_bin_accuracy.csv"), sort_by=["similarity_bin"])),
        "panel_a_within_bin_overlap_bridge_csv": str(save_csv(df_overlap_bridge, layout.data_file("panel_a_within_bin_overlap_bridge.csv"), sort_by=["group"])),
        "panel_b_dpi_trace_summary_csv": str(save_csv(mechanism.panel_b_trace_summary, layout.data_file("panel_b_dpi_trace_summary.csv"), sort_by=["condition", "time_step"])),
        "panel_b_dpi_pair_summary_csv": str(save_csv(mechanism.panel_b_pair_summary, layout.data_file("panel_b_dpi_pair_summary.csv"), sort_by=["condition", "pair_id"])),
        "panel_cd_pair_level_metrics_csv": str(save_csv(mechanism.panel_cd_pair_metrics, layout.data_file("panel_cd_pair_level_metrics.csv"), sort_by=["pair_id"])),
        "panel_c_reconstruction_summary_csv": str(save_csv(mechanism.panel_c_reconstruction_summary, layout.data_file("panel_c_reconstruction_summary.csv"), sort_by=["mode"])),
        "panel_d_direction_summary_csv": str(save_csv(mechanism.panel_d_direction_summary, layout.data_file("panel_d_direction_summary.csv"), sort_by=["mode"])),
        "panel_b_probe_trace_arrays_npz": str(save_npz(layout.array_file("panel_b_probe_trace_arrays.npz"), **mechanism.panel_b_probe_trace_arrays)),
        "panel_cd_reconstruction_vectors_npz": str(
            save_npz(
                layout.array_file("panel_cd_reconstruction_vectors.npz"),
                pair_id=mechanism.panel_cd_reconstruction_vectors["pair_id"],
                v_dyn=mechanism.panel_cd_reconstruction_vectors["v_dyn"],
                v_sta=mechanism.panel_cd_reconstruction_vectors["v_sta"],
                delta_v=mechanism.panel_cd_reconstruction_vectors["delta_v"],
                delta_hat_plus=mechanism.panel_cd_reconstruction_vectors["delta_hat_plus"],
                delta_hat_minus=mechanism.panel_cd_reconstruction_vectors["delta_hat_minus"],
            )
        ),
    }
    summary = build_fig3_summary(df_overlap_bridge, mechanism.overlap_summary, mechanism.l3_summary)
    artifact_paths["summary_json"] = str(save_json({**summary, "saved_artifacts": artifact_paths}, layout.root_file("summary.json")))
    artifact_paths["artifact_manifest_json"] = str(write_artifact_manifest(layout, artifact_paths))
    logger.info("[Done] Fig3 artifacts saved.")


if __name__ == "__main__":
    main()
