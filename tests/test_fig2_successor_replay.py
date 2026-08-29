from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from src.core.network import SDNN_Network
from src.experiments.common.monitored_dms import snapshot_boundary_state
from src.experiments.common.ping_common import prepare_network_state


def _boundary() -> dict[str, dict[str, np.ndarray]]:
    boundary: dict[str, dict[str, np.ndarray]] = {}
    states = ("v_mem", "g_e", "res", "inh_trace", "u", "x")
    for layer_index, layer in enumerate(("layer1", "layer2", "layer3"), start=1):
        boundary[layer] = {}
        for state_index, state in enumerate(states, start=1):
            base = 100 * layer_index + 10 * state_index
            boundary[layer][state] = np.asarray(
                [[base, base + 1], [base + 2, base + 3]],
                dtype=np.float32,
            )
    return boundary


def test_prepare_layer2_stsp_transplant_changes_only_layer2_ux() -> None:
    from src.experiments.paper_figures.fig2.successor_replay import (
        prepare_layer2_stsp_transplant,
    )

    source = _boundary()
    source["layer1"]["v_mem"][0, 0] = np.nan
    combined, slices, audit = prepare_layer2_stsp_transplant(
        source,
        donor_indices=np.asarray([1, 0], dtype=np.int64),
    )

    assert slices == {
        "native": slice(0, 2),
        "layer2_swap": slice(2, 4),
        "own_sham": slice(4, 6),
    }
    assert audit == {
        "layer2_only_mix_exact": True,
        "own_sham_boundary_exact": True,
    }
    assert np.array_equal(
        combined["layer2"]["u"],
        np.asarray(
            [
                [250, 251],
                [252, 253],
                [252, 253],
                [250, 251],
                [250, 251],
                [252, 253],
            ],
            dtype=np.float32,
        ),
    )
    assert np.array_equal(
        combined["layer2"]["x"],
        np.asarray(
            [
                [260, 261],
                [262, 263],
                [262, 263],
                [260, 261],
                [260, 261],
                [262, 263],
            ],
            dtype=np.float32,
        ),
    )
    for layer, states in source.items():
        for state, values in states.items():
            if layer == "layer2" and state in {"u", "x"}:
                continue
            np.testing.assert_array_equal(combined[layer][state], np.concatenate([values] * 3))
            assert not np.shares_memory(combined[layer][state], values)


def test_zero_input_has_zero_passive_corrected_successor_effect() -> None:
    from src.experiments.paper_figures.fig2.successor_replay import (
        capture_successor_transition,
        correct_passive_successor_effects,
    )

    net = SDNN_Network(device="cpu")
    prepare_network_state(net, 1, 2, 28, 28)
    boundary = snapshot_boundary_state(net)
    input_seq = torch.zeros((1, 1, 2, 28, 28), dtype=torch.bool)
    ctx = SimpleNamespace(
        net=net,
        device=torch.device("cpu"),
        cfg=SimpleNamespace(fixed_b_post_steps=0, fixed_b_early_window_steps=1),
    )

    active = capture_successor_transition(
        ctx,
        boundary=boundary,
        input_seq=input_seq,
        current_time=9,
        passive=False,
        random_seed=123,
    )
    passive = capture_successor_transition(
        ctx,
        boundary=boundary,
        input_seq=input_seq,
        current_time=9,
        passive=True,
        random_seed=123,
    )
    corrected = correct_passive_successor_effects(active, passive)

    assert set(active) == {"early_layer2_event_map", "layer3_ux_pre", "layer3_ux_post"}
    assert active["early_layer2_event_map"].dtype == np.float32
    assert active["layer3_ux_pre"].dtype == np.float32
    assert np.count_nonzero(corrected["early_layer2_event_map"]) == 0
    assert np.count_nonzero(corrected["layer3_successor_ux"]) == 0
