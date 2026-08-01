from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.history_rewrite_bridge import (
    build_c_anchor_mapping,
    donor_transfer,
    exact_one_sided_sign_flip_p,
    holm_adjust,
    mix_layer1_stsp,
)


def test_c_mapping_is_one_to_one_and_rotates_class() -> None:
    rows = []
    anchor_id = 0
    for replicate_id in range(2):
        for label in range(10):
            rows.append(
                {
                    "b_anchor_id": anchor_id,
                    "B_image_id": 1000 + anchor_id,
                    "B_label": label,
                    "B_replicate_id": replicate_id,
                }
            )
            anchor_id += 1
    mapping = build_c_anchor_mapping(pd.DataFrame(rows))
    assert mapping["c_anchor_id"].nunique() == len(mapping)
    assert np.array_equal(
        mapping["C_label"].to_numpy(),
        (mapping["B_label"].to_numpy() + 1) % 10,
    )
    assert mapping["B_replicate_id"].tolist() == [
        value // 10 for value in range(20)
    ]


def test_donor_transfer_recovers_known_fraction() -> None:
    receiver = np.zeros((2, 3), dtype=np.float32)
    donor = np.asarray([[1.0, 2.0, 3.0], [2.0, -1.0, 1.0]])
    swap = 0.4 * donor
    values, valid = donor_transfer(swap, receiver, donor)
    assert valid.tolist() == [True, True]
    assert np.allclose(values, 0.4)


def test_donor_transfer_marks_degenerate_donor_contrast_invalid() -> None:
    receiver = np.zeros((1, 2), dtype=np.float32)
    values, valid = donor_transfer(receiver, receiver, receiver)
    assert valid.tolist() == [False]
    assert np.isnan(values[0])


def test_exact_sign_flip_and_holm() -> None:
    assert exact_one_sided_sign_flip_p(np.ones(3)) == 1.0 / 8.0
    varying_positive = np.linspace(0.1, 0.9, 19, dtype=np.float64)
    assert exact_one_sided_sign_flip_p(varying_positive) == 1.0 / (1 << 19)
    adjusted = holm_adjust(np.asarray([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_layer1_mix_uses_only_donor_layer1_ux() -> None:
    donor = _synthetic_boundary(offset=10.0)
    receiver = _synthetic_boundary(offset=1.0)
    mixed = mix_layer1_stsp(donor, receiver)
    for layer in ("layer1", "layer2", "layer3"):
        for state in ("v_mem", "g_e", "res", "inh_trace", "u", "x"):
            expected = (
                donor[layer][state]
                if layer == "layer1" and state in {"u", "x"}
                else receiver[layer][state]
            )
            assert np.array_equal(mixed[layer][state], expected)


def _synthetic_boundary(
    *,
    offset: float,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        layer: {
            state: np.full(
                (2, 3),
                offset + layer_index + state_index / 10.0,
                dtype=np.float32,
            )
            for state_index, state in enumerate(
                ("v_mem", "g_e", "res", "inh_trace", "u", "x")
            )
        }
        for layer_index, layer in enumerate(("layer1", "layer2", "layer3"))
    }
