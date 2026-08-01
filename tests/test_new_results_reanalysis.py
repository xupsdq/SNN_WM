from __future__ import annotations

import numpy as np

from src.experiments.paper_figures.new_results_reanalysis import (
    _exact_sign_flip_p,
    _holm_adjust,
    _layer2_ux,
)


def test_exact_sign_flip_one_sided_all_positive() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    assert _exact_sign_flip_p(values, alternative="greater") == 1.0 / 8.0
    assert _exact_sign_flip_p(-values, alternative="less") == 1.0 / 8.0


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    raw = np.asarray([0.04, 0.01, 0.03])
    adjusted = _holm_adjust(raw)
    ordered = adjusted[np.argsort(raw)]
    assert np.all(np.diff(ordered) >= 0)
    assert np.all((adjusted >= raw) & (adjusted <= 1.0))


def test_layer2_ux_concatenates_u_then_x() -> None:
    bank = {
        "sequence_7_Sfinal_u": np.asarray([1.0, 2.0], dtype=np.float32),
        "sequence_7_Sfinal_x": np.asarray([3.0, 4.0], dtype=np.float32),
    }
    observed = _layer2_ux(bank, 7, "Sfinal")
    np.testing.assert_array_equal(observed, np.asarray([1.0, 2.0, 3.0, 4.0]))
