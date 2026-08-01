from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.plotting.experiments.manuscript_v4_figures import (
    EXPECTED_NETWORKS,
    SourceStore,
)


def test_fixed_b_seed_metric_inserts_and_validates_network_seed(
    tmp_path: Path,
) -> None:
    fixed_root = tmp_path / "fixed_b"
    for seed in EXPECTED_NETWORKS:
        metrics = fixed_root / f"seed_{seed}" / "data" / "metrics"
        metrics.mkdir(parents=True)
        pd.DataFrame({"value": [float(seed)]}).to_csv(
            metrics / "metric.csv",
            index=False,
        )
    store = SourceStore(
        repo_root=tmp_path,
        paper_root=tmp_path / "paper",
        p0_root=tmp_path / "p0",
        fixed_b_root=fixed_root,
    )

    observed = store.read_fixed_b_seed_metric(
        "metric.csv",
        ("network_seed", "value"),
    )

    assert len(observed) == len(EXPECTED_NETWORKS)
    assert observed["network_seed"].astype(int).tolist() == list(
        EXPECTED_NETWORKS
    )
    assert observed["value"].astype(int).tolist() == list(
        EXPECTED_NETWORKS
    )
