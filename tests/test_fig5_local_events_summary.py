from types import SimpleNamespace

import pandas as pd

from src.experiments.paper_figures.fig5.subexperiments.local_events import (
    _trial_event_summary,
    _trial_first_trace_summary,
)


def test_fig5c_summaries_weight_trials_not_events() -> None:
    events = pd.DataFrame(
        {
            "network_seed": [1000, 1000, 1000],
            "trial_id": [0, 0, 1],
            "winner_full_pre_delta_v_mean": [1.0, 3.0, 9.0],
            "loser_full_pre_delta_v_mean": [0.0, 0.0, 0.0],
            "winner_minus_loser_full_pre_delta_v_mean": [1.0, 3.0, 9.0],
            "winner_late_pre_delta_v_mean": [1.0, 3.0, 9.0],
            "loser_late_pre_delta_v_mean": [0.0, 0.0, 0.0],
            "winner_minus_loser_late_pre_delta_v_mean": [1.0, 3.0, 9.0],
        }
    )
    trial = _trial_event_summary(events)
    assert trial["winner_minus_loser_full_pre_delta_v_mean"].tolist() == [2.0, 9.0]

    ctx = SimpleNamespace(cfg=SimpleNamespace(network_seed=1000))
    trace = _trial_first_trace_summary(
        ctx,
        [
            {"trial_id": 0, "event_id": 0, "time_ms": -1.0, "trace_type": "winner_delta_v", "value": 1.0},
            {"trial_id": 0, "event_id": 1, "time_ms": -1.0, "trace_type": "winner_delta_v", "value": 3.0},
            {"trial_id": 1, "event_id": 2, "time_ms": -1.0, "trace_type": "winner_delta_v", "value": 9.0},
        ],
    )
    assert trace.loc[0, "mean_value"] == 5.5
    assert trace.loc[0, "n_trials"] == 2
    assert trace.loc[0, "n_events"] == 3
