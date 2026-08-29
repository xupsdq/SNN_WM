from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from src.experiments.paper_figures.common.sequence_root.selection import (
    select_matched_nonpeak_mask,
    select_top_mask,
)


def test_sequence_selection_matches_frozen_fig6_masks() -> None:
    values = np.array([[np.nan, 1.0, 5.0], [3.0, -2.0, 4.0]])
    top = select_top_mask(values, 0.5, positive=values > 0)
    np.testing.assert_array_equal(
        top,
        np.array([[False, False, True], [False, False, True]]),
    )

    peak = np.array([True, False, False, False, True, False, False, False])
    pool = np.array([True, True, True, True, True, True, True, False])
    np.testing.assert_array_equal(
        select_matched_nonpeak_mask(peak, pool, seed=17),
        np.array([False, False, False, True, False, False, True, False]),
    )

    undersized_pool = np.array(
        [True, True, False, False, True, False, False, False]
    )
    np.testing.assert_array_equal(
        select_matched_nonpeak_mask(peak, undersized_pool, seed=17),
        np.array([False, False, False, False, False, True, False, True]),
    )


def test_shared_sequence_root_does_not_import_fig6_subexperiment_implementation() -> None:
    path = Path("src/experiments/paper_figures/common/sequence_root/run_task.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith(
            "src.experiments.paper_figures.fig6.subexperiments"
        )
    ]
    assert imports == []
