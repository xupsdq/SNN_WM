from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.plotting.experiments.l3_accumulator_mechanism_experiment_plot import main


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_l3_accumulator_plot_only_generates_retained_figures(tmp_path: Path) -> None:
    result_dir = tmp_path / "bundle"
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (result_dir / name).mkdir(parents=True, exist_ok=True)

    _write_csv(
        result_dir / "data" / "pair_results.csv",
        [
            {
                "reconstruction_cosine_plus": 0.8,
                "reconstruction_cosine_minus": 0.3,
                "direction_match_plus": 1,
                "direction_match_minus": 0,
                "top_push_value_kstar": 0.5,
                "bias_magnitude": 0.2,
            },
            {
                "reconstruction_cosine_plus": 0.7,
                "reconstruction_cosine_minus": 0.4,
                "direction_match_plus": 1,
                "direction_match_minus": 1,
                "top_push_value_kstar": 0.4,
                "bias_magnitude": 0.25,
            },
        ],
    )
    (result_dir / "meta" / "plot_bundle_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "experiment_name": "l3_accumulator_mechanism_experiment",
                "inputs": {
                    "pair_results": {
                        "path": "data/pair_results.csv",
                        "required_columns": [
                            "reconstruction_cosine_plus",
                            "reconstruction_cosine_minus",
                            "direction_match_plus",
                            "direction_match_minus",
                            "top_push_value_kstar",
                            "bias_magnitude",
                        ],
                    }
                },
                "excluded_outputs": [
                    "figures/figure_1_case_deletion_maps.*",
                    "figures/figure_2_case_replacement_maps.*",
                ],
            }
        ),
        encoding="utf-8",
    )

    output_dir = result_dir / "figures_plot_only"
    assert main(["--input-dir", str(result_dir), "--output-dir", str(output_dir)]) == 0
    for stem in ("reconstruction_cosine", "argmax_reconstruction", "figure_4_pair_level_scatter"):
        assert (output_dir / f"{stem}.png").exists()
    assert not (output_dir / "figure_1_case_deletion_maps.png").exists()


def test_l3_accumulator_plot_only_requires_pair_results_csv(tmp_path: Path) -> None:
    result_dir = tmp_path / "bundle_missing"
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (result_dir / name).mkdir(parents=True, exist_ok=True)
    (result_dir / "meta" / "plot_bundle_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "experiment_name": "l3_accumulator_mechanism_experiment",
                "inputs": {"pair_results": {"path": "data/pair_results.csv"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="pair_results.csv"):
        main(["--input-dir", str(result_dir), "--output-dir", str(result_dir / "figures_plot_only")])
