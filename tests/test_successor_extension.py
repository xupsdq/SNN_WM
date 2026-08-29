"""Minimal self-checks for the successor-extension logic.

Run directly (no framework needed):
    python tests/test_successor_extension.py
or under pytest. Covers: B->C->D mapping, K=10 suffix-image exclusion,
donor-transfer invariants, and the outcome-blind overlap-mask builder.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.experiments.c5_l2_successor_closure import (
    build_c_anchor_mapping,
    donor_transfer,
)
from src.experiments.successor_extension.core import (
    _array_sha256,
    _extension_label,
    build_d_anchor_mapping,
    build_overlap_masks,
    pick_suffix_images_for_families,
)


def test_experiment_c_array_identity_hash_is_available_and_stable():
    value = np.array([[1, 2], [3, 4]], dtype=np.int16)
    assert _array_sha256(value) == (
        "03fd8197b1e1033acd10db0dc8637d3e44a14f604ed0cd1592282e4462bcb6eb"
    )


def test_successor_extension_runner_entrypoint_imports():
    importlib.import_module("src.experiments.successor_extension.runner")


def test_successor_runtime_owns_paths_json_and_parent_identity(tmp_path):
    runtime = importlib.import_module("src.experiments.successor_extension.runtime")
    expected_repo_root = Path(__file__).resolve().parents[1]
    assert runtime.repository_root() == expected_repo_root
    assert runtime.resolve_repo_path("results/example") == (
        expected_repo_root / "results" / "example"
    ).resolve()

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    cache_bytes = b'{"cache_key":"frozen"}\n'
    (task_dir / "cache_key.json").write_bytes(cache_bytes)

    summary_path = tmp_path / "nested" / "summary.json"
    runtime.write_json(summary_path, {"status": "completed", "seed": 1000})

    summary_text = summary_path.read_text(encoding="utf-8")
    assert summary_text == '{\n  "seed": 1000,\n  "status": "completed"\n}\n'
    assert json.loads(summary_text) == {
        "seed": 1000,
        "status": "completed",
    }
    assert runtime.parent_entry(task_dir) == {
        "path": str(task_dir.resolve()),
        "cache_key_sha256": hashlib.sha256(cache_bytes).hexdigest(),
    }
    cfg = SimpleNamespace(output_root=str(tmp_path / "results"), network_seed=1000)
    assert runtime.seed_root(cfg) == (tmp_path / "results" / "seed_1000").resolve()


def test_successor_callers_cross_the_public_runtime_seam():
    callers = (
        Path("src/experiments/successor_extension/runner.py"),
        Path("src/experiments/successor_extension/cohort.py"),
        Path("src/experiments/successor_extension/aggregate.py"),
    )
    private_core_imports: list[str] = []
    runtime_callers: set[Path] = set()
    for path in callers:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module == "src.experiments.successor_extension.runtime":
                runtime_callers.add(path)
            if node.module == "src.experiments.successor_extension.core":
                private_core_imports.extend(
                    f"{path}:{node.lineno}:{alias.name}"
                    for alias in node.names
                    if alias.name.startswith("_")
                )

    assert private_core_imports == []
    assert runtime_callers == set(callers)

    core_path = Path("src/experiments/successor_extension/core.py")
    core_tree = ast.parse(core_path.read_text(encoding="utf-8"), filename=str(core_path))
    retired_helpers = {
        "_build_ctx",
        "_parent_entry",
        "_repo_root",
        "_resolve",
        "_seed_root",
        "_sha256_file",
        "_write_json",
    }
    defined_names = {
        node.name
        for node in ast.walk(core_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert defined_names.isdisjoint(retired_helpers)


def _synthetic_b_specs() -> pd.DataFrame:
    rows = []
    for anchor_id in range(50):
        rows.append(
            {
                "b_anchor_id": anchor_id,
                "B_image_id": 10_000 + anchor_id,
                "B_label": anchor_id % 10,
                "B_replicate_id": anchor_id // 10,
            }
        )
    return pd.DataFrame(rows)


def test_bcd_anchor_mapping():
    specs = _synthetic_b_specs()
    c_map = build_c_anchor_mapping(specs)
    d_map = build_d_anchor_mapping(specs)
    assert len(c_map) == len(specs) == len(d_map)
    for row in d_map.itertuples(index=False):
        c_row = c_map.loc[c_map["b_anchor_id"].eq(row.b_anchor_id)].iloc[0]
        assert row.D_label == (row.B_label + 2) % 10
        assert c_row.C_label == (row.B_label + 1) % 10
        assert row.D_label != row.B_label and row.D_label != c_row.C_label
        spec = specs.loc[specs["b_anchor_id"].eq(row.b_anchor_id)].iloc[0]
        assert specs.loc[specs["b_anchor_id"].eq(row.d_anchor_id)].iloc[0]["B_replicate_id"] == spec["B_replicate_id"]
    assert d_map["d_anchor_id"].nunique() == len(d_map)
    assert c_map["c_anchor_id"].nunique() == len(c_map)


def test_extension_label_formula_continues_frozen_prefix():
    # frozen family 0 (candidate 30): A=[0,3,6,9,2], C=[5,2,9,6,3]
    family_0 = {"candidate_family_id": 30}
    a_labels = [_extension_label(30, "A", pos) for pos in range(10)]
    c_labels = [_extension_label(30, "C", pos) for pos in range(10)]
    assert a_labels == [0, 3, 6, 9, 2, 5, 8, 1, 4, 7]
    assert c_labels == [5, 2, 9, 6, 3, 0, 7, 4, 1, 8]
    assert a_labels[:5] == [0, 3, 6, 9, 2]
    assert c_labels[:5] == [5, 2, 9, 6, 3]


def test_suffix_image_picking_excludes_used_ids():
    families = pd.DataFrame(
        {
            "history_family_id": [0, 1, 2],
            "candidate_family_id": [30, 1, 32],
            "balance_stratum": [0, 1, 2],
        }
    )
    used = {100, 200, 300}
    pools = {label: list(range(label * 10, label * 10 + 10)) for label in range(10)}
    assignment = pick_suffix_images_for_families(families, used, pools)
    all_suffix = [image for suffix in assignment.values() for image in suffix]
    assert len(all_suffix) == len(set(all_suffix)), "suffix images reused across families"
    assert not (set(all_suffix) & used), "suffix images intersect the frozen exclusion set"
    for (family_id, condition), suffix in assignment.items():
        assert len(suffix) == 5
        family = families.loc[families["history_family_id"].eq(family_id)].iloc[0]
        expected = [_extension_label(int(family["candidate_family_id"]), condition, pos) for pos in range(5, 10)]
        for image_id, label in zip(suffix, expected):
            assert image_id in pools[label], f"image {image_id} has wrong class for label {label}"
    # determinism: same call reproduces identical assignment
    again = pick_suffix_images_for_families(families, used, pools)
    assert again == assignment


def test_donor_transfer_invariants():
    donor = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    receiver = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    values, valid = donor_transfer(donor, receiver, donor)
    assert valid.all()
    assert np.allclose(values, 1.0)
    values, valid = donor_transfer(receiver, receiver, donor)
    assert valid.all()
    assert np.allclose(values, 0.0)
    bad_shape = np.zeros((3, 2), dtype=np.float32)
    try:
        donor_transfer(bad_shape, receiver, donor)
        raise AssertionError("shape mismatch must raise")
    except ValueError:
        pass


def test_overlap_masks_disjoint_and_matched():
    rng = np.random.default_rng(0)
    support = np.zeros((2, 4, 4), dtype=bool)
    support[0, 0, 0] = True
    support[0, 0, 1] = True
    support[1, 3, 3] = True
    u = np.full((2, 4, 4), 0.2, dtype=np.float32)
    x = np.ones((2, 4, 4), dtype=np.float32)
    # history touched: row 0 fully deviated, row 1 only the support site
    u[0] = 0.9
    x[0] = 0.5
    u[1, 3, 3] = 0.8
    x[1, 3, 3] = 0.4
    masks = build_overlap_masks(u, x, support, rng)
    overlap = masks["overlap"]
    nonoverlap = masks["nonoverlap"]
    random_mask = masks["random"]
    deviated = np.abs(u * x - 0.2) > 1e-4
    assert np.all(overlap == (deviated & support))
    assert np.all(nonoverlap == (deviated & ~support))
    assert not (overlap & nonoverlap).any()
    assert not (overlap & random_mask).any()
    assert int(random_mask.sum()) == int(overlap.sum()) == 3
    assert int(masks["insufficient_random"]) == 0
    # insufficient pool: only one deviated site, overlap takes two
    u_small = np.full((2, 4, 4), 0.2, dtype=np.float32)
    x_small = np.ones((2, 4, 4), dtype=np.float32)
    u_small[0, 0, 0] = 0.9
    x_small[0, 0, 0] = 0.5
    support_small = np.zeros((2, 4, 4), dtype=bool)
    support_small[0, 0, 0] = True
    support_small[1, 1, 1] = True  # deviated nowhere on row 1
    masks_small = build_overlap_masks(u_small, x_small, support_small, rng)
    assert int(masks_small["insufficient_random"]) == 1


if __name__ == "__main__":
    test_bcd_anchor_mapping()
    test_extension_label_formula_continues_frozen_prefix()
    test_suffix_image_picking_excludes_used_ids()
    test_donor_transfer_invariants()
    test_overlap_masks_disjoint_and_matched()
    print("all successor_extension self-checks passed")
