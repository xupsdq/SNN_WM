from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.c5_l2_successor_closure import (
    C5Config,
    PRIMARY_ENDPOINTS,
    donor_transfer,
    screening_verdict,
    summarize_c5_endpoints,
)


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
