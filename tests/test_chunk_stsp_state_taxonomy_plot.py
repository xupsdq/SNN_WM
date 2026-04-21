from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.plotting.experiments.chunk_stsp_state_taxonomy_plot import main


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_chunk_stsp_state_taxonomy_plot_only_generates_both_groups(tmp_path: Path) -> None:
    result_dir = tmp_path / "bundle"
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (result_dir / name).mkdir(parents=True, exist_ok=True)

    _write_csv(
        result_dir / "metrics" / "state_similarity_metrics.csv",
        [
            {"record_type": "layer_summary", "layer": "layer1", "Sim_FS": 0.7, "Sim_FD": 0.4, "Sim_FShuffle": 0.2, "DI": 0.3},
            {"record_type": "binned_summary", "layer": "layer1", "DI_mean": 0.3, "sample_first_prob": 0.6},
            {"record_type": "layer_summary", "layer": "layer2", "Sim_FS": 0.6, "Sim_FD": 0.35, "Sim_FShuffle": 0.2, "DI": 0.25},
            {"record_type": "binned_summary", "layer": "layer2", "DI_mean": 0.25, "sample_first_prob": 0.5},
            {"record_type": "layer_summary", "layer": "layer3", "Sim_FS": 0.5, "Sim_FD": 0.3, "Sim_FShuffle": 0.2, "DI": 0.2},
            {"record_type": "binned_summary", "layer": "layer3", "DI_mean": 0.2, "sample_first_prob": 0.4},
        ],
    )
    _write_csv(
        result_dir / "metrics" / "state_decomposition_metrics.csv",
        [
            {"record_type": "layer_summary", "layer": "layer1", "alpha": 0.5, "beta": 0.2, "R2": 0.8, "residual_norm": 0.1},
            {"record_type": "layer_summary", "layer": "layer2", "alpha": 0.4, "beta": 0.3, "R2": 0.75, "residual_norm": 0.12},
            {"record_type": "layer_summary", "layer": "layer3", "alpha": 0.3, "beta": 0.35, "R2": 0.7, "residual_norm": 0.15},
        ],
    )
    _write_csv(
        result_dir / "metrics" / "state_changed_synapse_metrics.csv",
        [
            {"record_type": "layer_summary", "layer": "layer1", "S_only_changed_fraction": 0.2, "D_only_changed_fraction": 0.1, "Shared_changed_fraction": 0.3, "Full_only_novel_changed_fraction": 0.4, "changed_fraction_full": 0.5, "P_S_only_given_full": 0.2, "P_D_only_given_full": 0.2, "P_Shared_given_full": 0.3, "P_Novel_given_full": 0.3},
            {"record_type": "layer_summary", "layer": "layer2", "S_only_changed_fraction": 0.25, "D_only_changed_fraction": 0.1, "Shared_changed_fraction": 0.25, "Full_only_novel_changed_fraction": 0.4, "changed_fraction_full": 0.45, "P_S_only_given_full": 0.25, "P_D_only_given_full": 0.15, "P_Shared_given_full": 0.25, "P_Novel_given_full": 0.35},
            {"record_type": "layer_summary", "layer": "layer3", "S_only_changed_fraction": 0.3, "D_only_changed_fraction": 0.1, "Shared_changed_fraction": 0.2, "Full_only_novel_changed_fraction": 0.4, "changed_fraction_full": 0.4, "P_S_only_given_full": 0.3, "P_D_only_given_full": 0.1, "P_Shared_given_full": 0.2, "P_Novel_given_full": 0.4},
        ],
    )
    _write_csv(
        result_dir / "metrics" / "ping_coupling_metrics.csv",
        [
            {"record_type": "binned_summary", "layer": "layer1", "DI_mean": 0.3, "sample_first_prob": 0.6},
            {"record_type": "binned_summary", "layer": "layer2", "DI_mean": 0.2, "sample_first_prob": 0.55},
            {"record_type": "binned_summary", "layer": "layer3", "DI_mean": 0.1, "sample_first_prob": 0.5},
        ],
    )
    (result_dir / "meta" / "plot_bundle_manifest.json").write_text(json.dumps({"version": 1, "experiment_name": "chunk_stsp_state_taxonomy"}), encoding="utf-8")

    output_dir = result_dir / "figures_plot_only"
    assert main(["--input-dir", str(result_dir), "--output-dir", str(output_dir)]) == 0
    assert (output_dir / "chunk_stsp_state_taxonomy_overview.png").exists()
    assert (output_dir / "chunk_stsp_full_conditioned_changed.png").exists()
