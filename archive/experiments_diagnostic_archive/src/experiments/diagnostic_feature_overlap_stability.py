from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_probe_split(
    probe_df: pd.DataFrame,
    *,
    label_col: str,
    discovery_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if discovery_frac <= 0.0 or discovery_frac >= 1.0:
        raise ValueError("discovery_frac must be in (0, 1)")
    rng = np.random.default_rng(int(seed))
    discovery_parts = []
    heldout_parts = []
    for _, group in probe_df.groupby(label_col, sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        shuffled = rng.permutation(indices)
        cutoff = int(np.ceil(len(shuffled) * float(discovery_frac)))
        cutoff = min(max(cutoff, 1), len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)
        discovery_idx = shuffled[:cutoff]
        heldout_idx = shuffled[cutoff:]
        discovery_parts.append(probe_df.loc[discovery_idx].copy())
        if len(heldout_idx) > 0:
            heldout_parts.append(probe_df.loc[heldout_idx].copy())
    discovery = pd.concat(discovery_parts, axis=0).sort_values(["probe_label", "probe_id"], kind="stable").reset_index(drop=True)
    heldout = (
        pd.concat(heldout_parts, axis=0).sort_values(["probe_label", "probe_id"], kind="stable").reset_index(drop=True)
        if heldout_parts
        else probe_df.iloc[0:0].copy()
    )
    if heldout.empty and len(discovery) > 1:
        move_idx = int(rng.integers(0, len(discovery)))
        heldout = discovery.iloc[[move_idx]].copy().reset_index(drop=True)
        discovery = discovery.drop(discovery.index[move_idx]).reset_index(drop=True)
    if discovery.empty and len(heldout) > 1:
        move_idx = int(rng.integers(0, len(heldout)))
        discovery = heldout.iloc[[move_idx]].copy().reset_index(drop=True)
        heldout = heldout.drop(heldout.index[move_idx]).reset_index(drop=True)
    return discovery, heldout
