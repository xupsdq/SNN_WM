from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from src.experiments.paper_figures.fig6b_order_specificity.analysis import (
    _confusion_matrix,
    _formal_primary_statistics,
    _latest_only_design_reference,
    _monte_carlo_plus_one_p,
    _network_metrics,
    _validate_formal_runtime_config,
)
from src.experiments.paper_figures.fig6b_order_specificity.formal_spec import (
    FORMAL_SPEC_SCHEMA,
    load_frozen_formal_spec,
)
from src.experiments.paper_figures.fig6b_order_specificity.types import (
    CHANCE_ACCURACY,
    N_ORDERS,
    OrderSpecificityConfig,
)
from src.plotting.paper_fig.candidates.manuscript_fig6b_order_specificity import (
    render_manuscript_fig6b_order_specificity,
)


def test_analysis_module_imports_without_plotting() -> None:
    script = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys

        class RejectPlottingImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "src.plotting" or fullname.startswith("src.plotting."):
                    raise ImportError(f"plotting import rejected: {fullname}")
                return None

        sys.meta_path.insert(0, RejectPlottingImport())
        importlib.import_module(
            "src.experiments.paper_figures.fig6b_order_specificity.analysis"
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_plot_only_replays_sparse_historical_pilot_bundle_without_mutating_inputs(
    tmp_path,
) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    pd.DataFrame(
        [
            {
                "network_seed": seed,
                "accuracy": accuracy,
                "mean_margin": margin,
            }
            for seed, accuracy, margin in (
                (1000, 1.0, 0.04),
                (1001, 71.0 / 72.0, 0.03),
                (1002, 1.0, 0.05),
                (-1, 215.0 / 216.0, 0.04),
            )
        ]
    ).to_csv(metrics_dir / "network_order_metrics.csv", index=False)
    sparse_confusion = [
        {
            "network_seed": -1,
            "true_order": order,
            "predicted_order": order,
            "count": 35 if order == 5 else 36,
            "proportion": (35 if order == 5 else 36) / 216.0,
        }
        for order in range(N_ORDERS)
    ]
    sparse_confusion.append(
        {
            "network_seed": -1,
            "true_order": 5,
            "predicted_order": 3,
            "count": 1,
            "proportion": 1.0 / 216.0,
        }
    )
    pd.DataFrame(sparse_confusion).to_csv(
        metrics_dir / "confusion_matrix.csv", index=False
    )
    pd.DataFrame(
        [{"check_id": "overall_gate_decision", "observed": "GO"}]
    ).to_csv(metrics_dir / "pilot_gate_metrics.csv", index=False)
    inputs_before = {
        path.name: path.read_bytes() for path in sorted(metrics_dir.iterdir())
    }

    qa = render_manuscript_fig6b_order_specificity(tmp_path, plot_only=True)

    assert qa["plot_only"] is True
    assert (tmp_path / "figures" / "fig6b_order_specificity_pilot.png").is_file()
    assert {
        path.name: path.read_bytes() for path in sorted(metrics_dir.iterdir())
    } == inputs_before


def _prediction_rows() -> pd.DataFrame:
    rows = []
    for network_seed in (1000, 1001):
        for true_order in range(N_ORDERS):
            predicted = 3 if network_seed == 1001 and true_order == 5 else true_order
            rows.append(
                {
                    "network_seed": network_seed,
                    "set_id": 0,
                    "order_index": true_order,
                    "predicted_order_index": predicted,
                    "correct": int(predicted == true_order),
                    "margin": 0.01 if predicted == true_order else -0.01,
                    "score_spread": 0.03,
                    "score_tied": 0,
                }
            )
    return pd.DataFrame(rows)


def test_network_metrics_separate_trial_and_correct_counts() -> None:
    predictions = _prediction_rows()
    metrics = _network_metrics(predictions)
    overall = metrics.loc[metrics["network_seed"].eq(-1)].iloc[0]
    assert int(overall["n_trials"]) == 12
    assert int(overall["n_correct"]) == 11
    assert float(overall["accuracy"]) == pytest.approx(11.0 / 12.0)


def test_confusion_matrix_materializes_all_zero_cells_and_row_normalizes() -> None:
    confusion = _confusion_matrix(_prediction_rows())
    aggregate = confusion.loc[confusion["network_seed"].eq(-1)].copy()
    assert len(aggregate) == N_ORDERS * N_ORDERS
    assert int(aggregate["count"].sum()) == 12
    row_sums = aggregate.groupby("true_order", sort=True)["proportion"].sum()
    assert np.allclose(row_sums.to_numpy(dtype=float), 1.0)
    assert int((aggregate["count"] == 0).sum()) > 0
    error_cell = aggregate.loc[
        aggregate["true_order"].eq(5) & aggregate["predicted_order"].eq(3)
    ].iloc[0]
    assert int(error_cell["count"]) == 1
    assert float(error_cell["proportion"]) == pytest.approx(0.5)


def test_monte_carlo_p_uses_plus_one_correction() -> None:
    p_value, exceedances = _monte_carlo_plus_one_p(np.asarray([0.1, 0.2]), 1.0)
    assert exceedances == 0
    assert p_value == pytest.approx(1.0 / 3.0)
    p_value, exceedances = _monte_carlo_plus_one_p(np.asarray([0.1, 0.2]), 0.15)
    assert exceedances == 1
    assert p_value == pytest.approx(2.0 / 3.0)


def test_latest_only_is_an_analytical_design_reference() -> None:
    rows = []
    for network_seed in (1000, 1001):
        for order_index, prefix in enumerate(
            ("1;2;3", "1;3;2", "2;1;3", "2;3;1", "3;1;2", "3;2;1")
        ):
            rows.append(
                {
                    "network_seed": network_seed,
                    "set_id": 0,
                    "order_index": order_index,
                    "ordered_item_ids": f"{prefix};4",
                    "latest_item_id": 4,
                }
            )
    reference = _latest_only_design_reference(pd.DataFrame(rows), (1000, 1001))
    assert reference["reference_type"].eq("design_implied_chance").all()
    assert reference["latest_item_fixed"].all()
    assert np.allclose(reference["expected_accuracy"], CHANCE_ACCURACY)
    assert reference["empirical_accuracy"].isna().all()


def test_formal_scope_and_primary_statistics_use_20_networks() -> None:
    cfg = OrderSpecificityConfig(
        output_dir="unused",
        task="analysis",
        analysis_scope="formal",
        n_permutation_draws=10000,
    )
    assert cfg.expected_network_seeds == tuple(range(1000, 1020))
    spec = load_frozen_formal_spec()
    _validate_formal_runtime_config(cfg, spec)

    rows = []
    for offset, seed in enumerate(range(1000, 1020)):
        rows.append(
            {
                "network_seed": seed,
                "n_trials": 72,
                "n_correct": 71 + (offset % 2),
                "accuracy": (71 + (offset % 2)) / 72.0,
                "mean_margin": 0.004,
                "median_margin": 0.004,
                "mean_score_spread": 0.03,
                "tied_predictions": 0,
            }
        )
    rows.append(
        {
            "network_seed": -1,
            "n_trials": 1440,
            "n_correct": sum(row["n_correct"] for row in rows),
            "accuracy": np.mean([row["accuracy"] for row in rows]),
            "mean_margin": 0.004,
            "median_margin": 0.004,
            "mean_score_spread": 0.03,
            "tied_predictions": 0,
        }
    )
    statistics = _formal_primary_statistics(pd.DataFrame(rows)).iloc[0]
    assert int(statistics["n_networks"]) == 20
    assert int(statistics["df"]) == 19
    assert int(statistics["networks_above_chance"]) == 20
    assert float(statistics["mean_accuracy"]) > 0.98


def test_formal_runtime_rejects_unfrozen_permutation_count() -> None:
    cfg = OrderSpecificityConfig(
        output_dir="unused",
        task="analysis",
        analysis_scope="formal",
        n_permutation_draws=200,
    )
    with pytest.raises(RuntimeError, match="does not match the frozen specification"):
        _validate_formal_runtime_config(cfg, load_frozen_formal_spec())


def test_frozen_formal_analysis_spec_digest_and_scope() -> None:
    spec = load_frozen_formal_spec()
    assert spec["schema"] == FORMAL_SPEC_SCHEMA
    assert spec["status"] == "frozen_before_formal_scoring"
    assert spec["design"]["network_seeds"] == list(range(1000, 1020))
    assert spec["primary_analysis"]["primary_endpoint"] == (
        "exact six-way temporal-order identification accuracy"
    )
    assert spec["secondary_analyses"]["latest_only_reference"]["status"] == (
        "analytical design reference, not an empirical predictor"
    )
    assert "not promoted to primary" in (
        spec["secondary_analyses"]["equal_weight_additive_comparator"]["status"]
    )
    assert "promoting equal-weight performance to the primary endpoint" in (
        spec["prohibited_post_pilot_changes"]
    )
