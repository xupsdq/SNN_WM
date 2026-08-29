from __future__ import annotations

import numpy as np
import pytest

from src.experiments.common.inference import (
    bootstrap_mean_ci,
    crossed_bootstrap_mean_ci,
    exact_sign_flip_p,
    holm_adjust,
    stable_seed,
)


def test_stable_seed_preserves_existing_inference_namespaces() -> None:
    assert stable_seed(20260731, "panel", "endpoint", "condition") == 1356445041
    assert stable_seed("endpoint", 1) == 3524196352
    assert stable_seed("endpoint") == 3363553206


def test_bootstrap_mean_ci_preserves_frozen_percentile_result() -> None:
    values = np.asarray([0.125, -0.25, 0.5, 1.0], dtype=np.float64)
    assert bootstrap_mean_ci(values, draws=1000, seed=12345) == (-0.0625, 0.875)


def test_exact_sign_flip_preserves_frozen_alternatives_and_limits() -> None:
    positive = np.asarray([1.0, 2.0, 3.0])
    observed = (
        exact_sign_flip_p(positive, alternative="greater"),
        exact_sign_flip_p(-positive, alternative="less"),
        exact_sign_flip_p(positive, alternative="two-sided"),
    )
    assert observed == (0.125, 0.125, 0.25)
    assert np.isnan(exact_sign_flip_p([np.nan, np.inf], alternative="greater"))
    with pytest.raises(ValueError, match="bounded to 24"):
        exact_sign_flip_p(np.ones(25), alternative="greater")


def test_exact_sign_flip_counts_observed_pattern_in_enumeration_order() -> None:
    values = np.asarray(
        [
            -1.2919896640826884,
            -2.3255813953488342,
            2.439024390243901,
            4.629629629629626,
            -7.181571815718154,
            -5.303030303030301,
            -2.971576227390184,
            -12.962962962962962,
            -7.954545454545457,
            -5.808080808080806,
            -1.328502415458935,
            -7.493540051679588,
            -9.09090909090909,
            -8.080808080808083,
            -4.567901234567898,
            -4.567901234567898,
            -10.246913580246911,
            -12.661498708010338,
            -1.481481481481481,
            -3.1746031746031775,
        ]
    )
    assert exact_sign_flip_p(values, alternative="less") == 81.0 / (2**20)


def test_holm_adjust_preserves_frozen_step_down_result() -> None:
    adjusted = holm_adjust(np.asarray([0.001, 0.02, 0.04]))
    assert adjusted.tolist() == [0.003, 0.04, 0.04]


def test_crossed_bootstrap_preserves_frozen_two_axis_result() -> None:
    values = np.asarray([0.1, 0.4, 0.9, 1.2, -0.3, 0.7, 0.0, 1.5, 0.2])
    family_ids = np.asarray([3, 3, 3, 7, 7, 7, 9, 9, 9])
    anchor_ids = np.asarray([2, 5, 8] * 3)
    observed = crossed_bootstrap_mean_ci(
        values,
        family_ids,
        anchor_ids,
        draws=1000,
        seed=456,
    )
    assert observed == (0.06666666666666668, 0.8999999999999999)
