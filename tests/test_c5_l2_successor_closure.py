from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.c5_l2_successor_closure import (
    C5Config,
    PRIMARY_ENDPOINTS,
    _layer2_mix_is_exact,
    _mix_layer2_stsp_by_index,
    donor_transfer,
    screening_verdict,
    summarize_c5_endpoints,
)


def _boundary() -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for layer_index, layer in enumerate(("layer1", "layer2", "layer3"), start=1):
        out[layer] = {}
        for state_index, state in enumerate(("v_mem", "g_e", "res", "inh_trace", "u", "x"), start=1):
            base = 100 * layer_index + 10 * state_index
            out[layer][state] = np.asarray([[base, base + 1], [base + 2, base + 3]], dtype=np.float32)
    return out


def test_layer2_mix_changes_only_layer2_ux() -> None:
    boundary = _boundary()
    boundary["layer1"]["v_mem"][0, 0] = np.nan
    donor_indices = np.asarray([1, 0], dtype=np.int64)
    mixed = _mix_layer2_stsp_by_index(boundary, donor_indices)
    assert _layer2_mix_is_exact(mixed, boundary, donor_indices)
    assert np.array_equal(mixed["layer2"]["u"], boundary["layer2"]["u"][donor_indices])
    assert np.array_equal(mixed["layer2"]["x"], boundary["layer2"]["x"][donor_indices])
    assert np.array_equal(mixed["layer1"]["u"], boundary["layer1"]["u"])
    assert np.array_equal(mixed["layer3"]["x"], boundary["layer3"]["x"])
    assert np.array_equal(mixed["layer2"]["v_mem"], boundary["layer2"]["v_mem"])


def test_donor_transfer_recovers_known_projection() -> None:
    receiver = np.zeros((2, 3), dtype=np.float32)
    donor = np.asarray([[1.0, 2.0, 0.0], [2.0, 0.0, 2.0]], dtype=np.float32)
    swap = 0.25 * donor
    values, valid = donor_transfer(swap, receiver, donor)
    assert valid.all()
    assert np.allclose(values, 0.25)


def test_summary_and_verdict_require_all_endpoint_depth_gates() -> None:
    rows = []
    for prefix in (1, 5):
        for family in range(3):
            for anchor in range(4):
                for direction in ("A", "C"):
                    rows.append(
                        {
                            "network_seed": 1000,
                            "prefix_k": prefix,
                            "history_family_id": family,
                            "b_anchor_id": anchor,
                            "receiver_history_condition": direction,
                            "early_layer2_event_map_donor_transfer": 0.4,
                            "early_layer2_event_map_transfer_valid": 1,
                            "layer3_successor_ux_donor_transfer": 0.3,
                            "layer3_successor_ux_transfer_valid": 1,
                        }
                    )
    cells = pd.DataFrame(rows)
    cfg = C5Config(bootstrap_draws=200)
    summary = summarize_c5_endpoints(cells, cfg)
    assert len(summary) == 4
    assert set(summary["endpoint"]) == set(PRIMARY_ENDPOINTS)
    assert summary["screening_pass"].eq(1).all()
    identity = pd.DataFrame({"identity_pass": [1, 1]})
    verdict = screening_verdict(summary, identity)
    assert verdict["verdict"] == "supported_in_development_seed"
