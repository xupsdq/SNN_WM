from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig2.subexperiments.crossfit_interaction import (
    PRIMARY_ENDPOINT,
    SENSITIVITY_ENDPOINT,
    build_crossfit_null_specs,
    build_crossfit_split_specs,
    fit_crossfit_interaction,
    validate_crossfit_split_specs,
)


def _pair_table(n_pairs: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": np.arange(n_pairs, dtype=np.int64),
            "A_image_id": np.arange(n_pairs, dtype=np.int64),
            "B_image_id": np.arange(10_000, 10_000 + n_pairs, dtype=np.int64),
            "A_label": np.arange(n_pairs, dtype=np.int64) % 10,
            "B_label": (np.arange(n_pairs, dtype=np.int64) + 1) % 10,
        }
    )


def test_crossfit_split_keeps_shared_constituent_images_together() -> None:
    pairs = _pair_table(12)
    pairs.loc[1, "A_image_id"] = int(pairs.loc[0, "A_image_id"])
    pairs.loc[5, "B_image_id"] = int(pairs.loc[4, "A_image_id"])
    split_specs = build_crossfit_split_specs(pairs, network_seed=1000, n_folds=4)

    validate_crossfit_split_specs(split_specs, pairs, network_seed=1000, n_folds=4)
    by_pair = split_specs.set_index("pair_id")
    assert int(by_pair.loc[0, "fold"]) == int(by_pair.loc[1, "fold"])
    assert int(by_pair.loc[4, "fold"]) == int(by_pair.loc[5, "fold"])
    assert int(by_pair.loc[0, "component_size"]) == 2
    assert int(by_pair.loc[4, "component_size"]) == 2


def test_crossfit_additive_null_has_no_structural_positive_gain() -> None:
    pairs = _pair_table(60)
    splits = build_crossfit_split_specs(pairs, network_seed=1000, n_folds=5)
    rng = np.random.default_rng(84)
    x_a = rng.normal(size=(60, 128))
    x_b = rng.normal(size=(60, 128))
    y = 0.43 * x_a + 0.57 * x_b

    result = fit_crossfit_interaction(
        x_a,
        x_b,
        y,
        pairs,
        splits,
        network_seed=1000,
        layer="layer3",
        state_variable="g",
    )
    row = result.network_metrics.iloc[0]
    assert float(row[PRIMARY_ENDPOINT]) <= 0.0
    assert abs(float(row[PRIMARY_ENDPOINT])) < 1e-4
    assert abs(float(row[SENSITIVITY_ENDPOINT])) < 1e-10


def test_crossfit_null_specs_freeze_three_predeclared_nulls() -> None:
    specs = build_crossfit_null_specs(
        network_seed=1000,
        n_replicates=7,
        feature_count=64,
        noise_scale_ratio=1.0,
    )
    assert len(specs) == 21
    assert specs.groupby("null_model")["replicate"].nunique().eq(7).all()
    assert specs["random_seed"].nunique() == 21
    assert set(specs["endpoint"]) == {PRIMARY_ENDPOINT, SENSITIVITY_ENDPOINT}


def test_crossfit_recovers_negative_pair_interaction_out_of_fold() -> None:
    pairs = _pair_table(60)
    splits = build_crossfit_split_specs(pairs, network_seed=1000, n_folds=5)
    rng = np.random.default_rng(91)
    x_a = rng.normal(size=(60, 128))
    x_b = rng.normal(size=(60, 128))
    y = 0.43 * x_a + 0.57 * x_b - 0.18 * x_a * x_b + rng.normal(0.0, 0.01, size=x_a.shape)

    result = fit_crossfit_interaction(
        x_a,
        x_b,
        y,
        pairs,
        splits,
        network_seed=1000,
        layer="layer3",
        state_variable="g",
    )
    row = result.network_metrics.iloc[0]
    gamma = result.coefficients.loc[
        result.coefficients["model"].eq("linear_interaction")
        & result.coefficients["predictor"].eq("z_A_x_z_B"),
        "coefficient",
    ]
    assert float(row[PRIMARY_ENDPOINT]) > 0.02
    assert float(row[SENSITIVITY_ENDPOINT]) > 0.02
    assert len(gamma) == 5
    assert (gamma.astype(float) < 0.0).all()
