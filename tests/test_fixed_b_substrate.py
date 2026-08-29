from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.paper_figures.fig2.fixed_b_artifacts import FixedBArtifact
from src.experiments.paper_figures.fig2.fixed_b_substrate import (
    load_fixed_b_parent,
    load_paired_history_slice,
)


def test_paired_history_slice_keeps_rows_boundaries_and_donors_aligned() -> None:
    history_specs = pd.DataFrame(
        [
            {"prefix_k": 5, "history_row_id": 21, "history_family_id": 2, "history_condition": "A", "elapsed_steps": 123},
            {"prefix_k": 5, "history_row_id": 10, "history_family_id": 1, "history_condition": "A", "elapsed_steps": 123},
            {"prefix_k": 5, "history_row_id": 30, "history_family_id": 0, "history_condition": "S0", "elapsed_steps": 123},
            {"prefix_k": 5, "history_row_id": 20, "history_family_id": 2, "history_condition": "C", "elapsed_steps": 123},
            {"prefix_k": 5, "history_row_id": 11, "history_family_id": 1, "history_condition": "C", "elapsed_steps": 123},
        ]
    )
    artifact = FixedBArtifact(
        root=Path("."),
        tables={"history_specs": history_specs},
        arrays={
            "k5__layer1__u": np.arange(10, dtype=np.float32).reshape(5, 2),
            "k5__layer1__x": np.arange(10, 20, dtype=np.float32).reshape(5, 2),
        },
        payloads={},
        manifest=pd.DataFrame(),
        digest="frozen-fixture",
    )

    selected = load_paired_history_slice(artifact, prefix_k=5, max_families=1)

    assert selected.rows["history_row_id"].tolist() == [10, 11]
    assert selected.rows["history_condition"].tolist() == ["A", "C"]
    assert selected.family_ids == (1,)
    assert selected.current_time == 123
    np.testing.assert_array_equal(selected.donor_indices, np.array([1, 0]))
    np.testing.assert_array_equal(
        selected.boundary["layer1"]["u"],
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
    )


def test_fixed_b_parent_rejects_a_mismatched_task_identity(tmp_path: Path) -> None:
    (tmp_path / "cache_key.json").write_text(
        '{"cache_key":{"task_id":"fixed_b_history_bank"}}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Parent task/cache-key mismatch"):
        load_fixed_b_parent(tmp_path, task_id="fixed_b_input_bank")


def test_fixed_b_consumers_do_not_import_private_runtime_interfaces() -> None:
    consumers = (
        Path("src/experiments/paper_figures/fig2/run_task.py"),
        Path("src/experiments/c5_l2_successor_closure.py"),
        Path("src/experiments/successor_extension/core.py"),
        Path("src/experiments/successor_extension/cohort.py"),
    )
    forbidden_modules = {
        "src.experiments.paper_figures.fig2.run_task",
        "src.experiments.paper_figures.fig2.subexperiments.fixed_b_runtime",
    }
    violations: list[str] = []
    substrate_consumers: set[Path] = set()
    for path in consumers:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module == "src.experiments.paper_figures.fig2.fixed_b_substrate":
                substrate_consumers.add(path)
            if node.module in forbidden_modules:
                violations.extend(
                    f"{path}:{node.lineno}:{node.module}:{alias.name}"
                    for alias in node.names
                )
            if node.module == "src.experiments.c5_l2_successor_closure":
                violations.extend(
                    f"{path}:{node.lineno}:{node.module}:{alias.name}"
                    for alias in node.names
                    if alias.name.startswith("_")
                )

    assert violations == []
    assert substrate_consumers == set(consumers)
