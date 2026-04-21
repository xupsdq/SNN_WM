from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.plotting.experiments.chunk_stsp_multiitem_sequence_plot import main


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_chunk_stsp_multiitem_sequence_plot_only_generates_expected_figures(tmp_path: Path) -> None:
    result_dir = tmp_path / "chunk_bundle"
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (result_dir / name).mkdir(parents=True, exist_ok=True)

    _write_csv(
        result_dir / "data" / "item_similarity_metrics.csv",
        [
            {"layer": "layer3", "seq_len": 3, "stage_k": 1, "item_index": 1, "similarity_weight_nonnegative": 1.0},
            {"layer": "layer3", "seq_len": 3, "stage_k": 2, "item_index": 1, "similarity_weight_nonnegative": 0.3},
            {"layer": "layer3", "seq_len": 3, "stage_k": 2, "item_index": 2, "similarity_weight_nonnegative": 0.7},
            {"layer": "layer3", "seq_len": 3, "stage_k": 3, "item_index": 1, "similarity_weight_nonnegative": 0.2},
            {"layer": "layer3", "seq_len": 3, "stage_k": 3, "item_index": 2, "similarity_weight_nonnegative": 0.3},
            {"layer": "layer3", "seq_len": 3, "stage_k": 3, "item_index": 3, "similarity_weight_nonnegative": 0.5},
            {"layer": "layer2", "seq_len": 3, "stage_k": 3, "item_index": 3, "similarity_weight_nonnegative": 0.4},
        ],
    )
    _write_csv(
        result_dir / "metrics" / "similarity_summary_metrics.csv",
        [
            {"layer": "layer1", "seq_len": 3, "stage_k": 1, "com_sim": 1.0, "sim_effective_count": 1.0},
            {"layer": "layer1", "seq_len": 3, "stage_k": 3, "com_sim": 1.8, "sim_effective_count": 1.4},
            {"layer": "layer2", "seq_len": 3, "stage_k": 1, "com_sim": 1.0, "sim_effective_count": 1.0},
            {"layer": "layer2", "seq_len": 3, "stage_k": 3, "com_sim": 2.0, "sim_effective_count": 1.8},
            {"layer": "layer3", "seq_len": 3, "stage_k": 1, "com_sim": 1.0, "sim_effective_count": 1.0},
            {"layer": "layer3", "seq_len": 3, "stage_k": 2, "com_sim": 1.6, "sim_effective_count": 1.5},
            {"layer": "layer3", "seq_len": 3, "stage_k": 3, "com_sim": 2.3, "sim_effective_count": 2.2},
        ],
    )
    _write_csv(
        result_dir / "metrics" / "ping_retrieval_metrics.csv",
        [
            {"seq_len": 3, "stage_k": 3, "item_index": 1, "ping_weight": 0.2},
            {"seq_len": 3, "stage_k": 3, "item_index": 2, "ping_weight": 0.3},
            {"seq_len": 3, "stage_k": 3, "item_index": 3, "ping_weight": 0.5},
        ],
    )
    _write_csv(
        result_dir / "data" / "cluster_participation_metrics.csv",
        [
            {"trial_id": 1, "layer": "layer1", "seq_len": 3, "stage_k": 3, "cluster_similarity_mass": 0.6},
            {"trial_id": 1, "layer": "layer1", "seq_len": 3, "stage_k": 3, "cluster_similarity_mass": 0.4},
            {"trial_id": 1, "layer": "layer3", "seq_len": 3, "stage_k": 3, "cluster_similarity_mass": 0.7},
            {"trial_id": 1, "layer": "layer3", "seq_len": 3, "stage_k": 3, "cluster_similarity_mass": 0.3},
        ],
    )
    _write_csv(
        result_dir / "data" / "stepwise_update_metrics.csv",
        [
            {"layer": "layer1", "stage_k": 2, "stepwise_update_ratio": 0.9},
            {"layer": "layer2", "stage_k": 2, "stepwise_update_ratio": 1.0},
            {"layer": "layer3", "stage_k": 2, "stepwise_update_ratio": 1.1},
        ],
    )
    (result_dir / "summary.json").write_text("{}", encoding="utf-8")
    (result_dir / "meta" / "plot_bundle_manifest.json").write_text(
        json.dumps(
            {
                "bundle_type": "chunk_stsp_multiitem_sequence_plot_inputs",
                "version": 1,
                "files": {
                    "item_similarity_metrics_csv": "data/item_similarity_metrics.csv",
                    "similarity_summary_metrics_csv": "metrics/similarity_summary_metrics.csv",
                    "ping_retrieval_metrics_csv": "metrics/ping_retrieval_metrics.csv",
                    "cluster_participation_metrics_csv": "data/cluster_participation_metrics.csv",
                    "stepwise_update_metrics_csv": "data/stepwise_update_metrics.csv",
                },
            }
        ),
        encoding="utf-8",
    )

    output_dir = result_dir / "figures_plot_only"
    exit_code = main(["--input-dir", str(result_dir), "--output-dir", str(output_dir)])

    assert exit_code == 0
    for stem in (
        "item_similarity_heatmap",
        "anchor_position_vs_stage",
        "similarity_concentration",
        "ping_retrieval_profile",
        "cluster_participation",
        "stepwise_update_ratio",
    ):
        assert (output_dir / f"{stem}.png").exists()
