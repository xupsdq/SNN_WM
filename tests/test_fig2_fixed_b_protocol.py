from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments.common.inference import exact_sign_flip_p, holm_adjust
from src.core.network import SDNN_Network
from src.experiments.common.monitored_dms import build_layer_input_shapes, snapshot_boundary_state
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.paper_figures.fig2.fixed_b_artifacts import FixedBArtifact, array_hash
from src.experiments.paper_figures.fig2.fixed_b_protocol import (
    CONFIRMATORY_AUTHORIZATION,
    CONFIRMATORY_SEEDS,
    FULL_COHORT_SEEDS,
    select_history_families,
    validate_seed_permission,
)
from src.experiments.paper_figures.fig2.successor_replay import (
    FAST_STATE_KEYS,
    STSP_STATE_KEYS,
    restore_boundary_state,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_analysis import (
    _exact_b_audit,
    _fit_feature_model,
    _two_axis_masks,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_mechanism_analysis import (
    _matched_random_coordinate_mean,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_runtime import (
    LAYER_KEYS,
    _mixed_swap_boundary,
    pack_event_bits,
    unpack_event_bits,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_specs import (
    NULL_PURPOSES,
    _null_specs,
)


def _artifact(*, tables=None, arrays=None) -> FixedBArtifact:
    return FixedBArtifact(
        root=Path("."),
        tables=dict(tables or {}),
        arrays=dict(arrays or {}),
        payloads={},
        manifest=pd.DataFrame(),
        digest="test",
    )


def test_outcome_blind_selector_rejects_post_b_covariates() -> None:
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(
            fixed_b_history_families=2,
            fixed_b_protocol_seed=20260724,
            fixed_b_source_match_max_smd=100.0,
            smoke=True,
        )
    )
    features = []
    overlap = []
    families = []
    for candidate_id in range(4):
        families.append(
            {
                "candidate_family_id": candidate_id,
                "balance_stratum": candidate_id // 2,
            }
        )
        for prefix_k in (1, 5):
            for condition, offset in (("A", 0.0), ("C", 0.01)):
                features.append(
                    {
                        "candidate_family_id": candidate_id,
                        "history_condition": condition,
                        "prefix_k": prefix_k,
                        "layer1_u_mean": candidate_id + prefix_k + offset,
                        "layer2_prior_spike_count": 2 * candidate_id + offset,
                    }
                )
                for anchor_id in range(2):
                    overlap.append(
                        {
                            "candidate_family_id": candidate_id,
                            "history_condition": condition,
                            "prefix_k": prefix_k,
                            "b_anchor_id": anchor_id,
                            "projected_overlap": candidate_id + anchor_id + offset,
                        }
                    )
    selected, audit, balance, summary = select_history_families(
        ctx,
        pd.DataFrame(features),
        pd.DataFrame(overlap),
        pd.DataFrame(families),
    )
    assert len(selected) == 2
    assert audit["selection_uses_outcomes"].eq(0).all()
    assert balance["passed"].eq(1).all()
    assert summary["outcome_columns_accessed"] == []

    contaminated = pd.DataFrame(features)
    contaminated["post_B_update_norm"] = 1.0
    with pytest.raises(ValueError, match="non-prestate covariates"):
        select_history_families(
            ctx,
            contaminated,
            pd.DataFrame(overlap),
            pd.DataFrame(families),
        )


def test_exact_b_audit_detects_any_tensor_hash_change() -> None:
    tensors = np.zeros((2, 3, 1, 2, 2), dtype=np.bool_)
    tensors[1, 0, 0, 0, 0] = True
    hashes = [array_hash(value) for value in tensors]
    inputs = _artifact(
        tables={
            "input_manifest": pd.DataFrame(
                {
                    "b_anchor_id": [0, 1],
                    "tensor_sha256": hashes,
                    "input_energy": [0.0, 1.0],
                }
            )
        }
    )
    rows = pd.DataFrame(
        {
            "b_anchor_id": [0, 0, 1, 1],
            "exact_b_tensor_sha256": [hashes[0], hashes[0], hashes[1], hashes[1]],
        }
    )
    ctx = SimpleNamespace(cfg=SimpleNamespace(network_seed=1000))
    passed = _exact_b_audit(ctx, inputs, _artifact(tables={"rollout_rows": rows}))
    assert passed["passed"].eq(1).all()

    corrupted = rows.copy()
    corrupted.loc[3, "exact_b_tensor_sha256"] = "changed"
    failed = _exact_b_audit(ctx, inputs, _artifact(tables={"rollout_rows": corrupted}))
    assert failed.loc[failed["b_anchor_id"].eq(1), "passed"].eq(0).all()


def test_swap_boundary_changes_only_declared_stsp_scope() -> None:
    boundary = {
        layer: {
            state: np.asarray([[10.0 + index], [20.0 + index]], dtype=np.float32)
            for index, state in enumerate(FAST_STATE_KEYS + STSP_STATE_KEYS)
        }
        for layer in LAYER_KEYS
    }
    specs = pd.DataFrame(
        [
            {
                "history_family_id": 0,
                "receiver_condition": "A",
                "donor_condition": "C",
            }
        ]
    )
    index = {(0, "A"): 0, (0, "C"): 1}
    mixed = _mixed_swap_boundary(boundary, specs, index, scope="layer1_only")
    for layer in LAYER_KEYS:
        for state in FAST_STATE_KEYS:
            assert np.array_equal(mixed[layer][state], boundary[layer][state][[0]])
        expected_row = 1 if layer == "layer1" else 0
        for state in STSP_STATE_KEYS:
            assert np.array_equal(
                mixed[layer][state],
                boundary[layer][state][[expected_row]],
            )


def test_serialized_boundary_restores_every_fast_and_stsp_state_exactly() -> None:
    net = SDNN_Network(device="cpu")
    prepare_network_state(net, 2, 1, 28, 28)
    with torch.no_grad():
        for layer_index, layer_name in enumerate(LAYER_KEYS):
            layer = getattr(net, layer_name)
            layer.v_mem.fill_(-55.0 - layer_index)
            layer.g_e.fill_(0.2 + 0.01 * layer_index)
            layer.res.fill_(1)
            layer.lateral_inh.inh_trace.fill_(0.3 + 0.01 * layer_index)
            layer.u_pre.fill_(0.6 + 0.01 * layer_index)
            layer.x_pre.fill_(0.7 + 0.01 * layer_index)
    expected = snapshot_boundary_state(net)
    with torch.no_grad():
        for layer_name in LAYER_KEYS:
            layer = getattr(net, layer_name)
            layer.v_mem.zero_()
            layer.g_e.zero_()
            layer.res.zero_()
            layer.lateral_inh.inh_trace.zero_()
            layer.u_pre.zero_()
            layer.x_pre.zero_()
    shapes = build_layer_input_shapes(net, 2, 1, 28, 28)
    restore_boundary_state(net, expected, shapes, mode="full_boundary", device=torch.device("cpu"))
    restored = snapshot_boundary_state(net)
    for layer_name in LAYER_KEYS:
        for state_name in FAST_STATE_KEYS + STSP_STATE_KEYS:
            assert torch.equal(restored[layer_name][state_name], expected[layer_name][state_name])


def test_event_bitpacking_and_null_specs_are_frozen_and_recomputable() -> None:
    rng = np.random.default_rng(17)
    events = rng.integers(0, 2, size=(7, 3, 4, 2, 2), dtype=np.uint8).astype(np.bool_)
    packed = pack_event_bits(events)
    unpacked = unpack_event_bits(packed, events.shape[1:])
    assert np.array_equal(unpacked, events)

    first = _null_specs(20260724, (1, 5), 7)
    second = _null_specs(20260724, (1, 5), 7)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["purpose"]) == set(NULL_PURPOSES)
    assert first.groupby(["purpose", "prefix_k"])["replicate"].nunique().eq(7).all()
    assert first["random_seed"].is_unique


def test_two_axis_folds_exclude_shared_history_and_b_identities() -> None:
    rows = pd.DataFrame(
        {
            "history_fold": [0, 0, 1, 1, 2, 2, 0, 1, 2],
            "b_fold": [0, 1, 0, 1, 2, 0, 2, 2, 1],
        }
    )
    train, test, guard = _two_axis_masks(rows, 0)
    assert not np.any(train & test)
    assert not np.any(train & guard)
    assert not np.any(test & guard)
    assert rows.loc[train, "history_fold"].ne(0).all()
    assert rows.loc[train, "b_fold"].ne(0).all()
    assert rows.loc[test, "history_fold"].eq(0).all()
    assert rows.loc[test, "b_fold"].eq(0).all()




def test_remaining_seed_gate_requires_exact_runtime_authorization(tmp_path: Path) -> None:
    state_path = tmp_path / "task_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "4.0",
                "remaining_seeds_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="still forbidden"):
        validate_seed_permission(
            1001,
            task_state_path=state_path,
            protocol=SimpleNamespace(
                payloads={"protocol": {"protocol_digest": "test-digest"}}
            ),
        )

    state_path.write_text(
        json.dumps(
            {
                "schema_version": "4.0",
                "remaining_seeds_allowed": True,
                "protocol_digest": "test-digest",
                "confirmatory_networks": list(CONFIRMATORY_SEEDS),
                "runtime_authorization": {
                    "text": CONFIRMATORY_AUTHORIZATION,
                    "track": "confirmatory_v4",
                },
            }
        ),
        encoding="utf-8",
    )
    assert (
        validate_seed_permission(
            1001,
            task_state_path=state_path,
            protocol=SimpleNamespace(
                payloads={"protocol": {"protocol_digest": "test-digest"}}
            ),
        )
        == "confirmatory_v4"
    )


def test_v4_network_inference_helpers_are_exact_and_deterministic() -> None:
    values = np.ones(len(FULL_COHORT_SEEDS), dtype=np.float64)
    assert exact_sign_flip_p(values, alternative="greater") == pytest.approx(
        1.0 / (2 ** len(FULL_COHORT_SEEDS))
    )
    confirmatory_values = np.ones(
        len(CONFIRMATORY_SEEDS),
        dtype=np.float64,
    )
    assert exact_sign_flip_p(confirmatory_values, alternative="greater") == pytest.approx(
        1.0 / (2 ** len(CONFIRMATORY_SEEDS))
    )
    adjusted = holm_adjust(np.asarray([0.001, 0.02, 0.04]))
    assert np.allclose(adjusted, np.asarray([0.003, 0.04, 0.04]))

    coordinates = np.arange(20, dtype=np.float64)
    first = _matched_random_coordinate_mean(
        coordinates,
        5,
        seed=123,
        replicates=7,
    )
    second = _matched_random_coordinate_mean(
        coordinates,
        5,
        seed=123,
        replicates=7,
    )
    assert first == second
    assert np.isfinite(first)
