from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import pandas as pd


def sample_mismatch_pair_specs(
    class_index: Dict[int, List[int]],
    *,
    num_pairs: int,
    num_classes: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(int(seed))
    rows = []
    classes = list(range(int(num_classes)))
    for pair_id in range(int(num_pairs)):
        sample_label = rng.choice(classes)
        probe_candidates = [label for label in classes if label != sample_label]
        probe_label = rng.choice(probe_candidates)
        rows.append(
            {
                "pair_id": int(pair_id),
                "sample_label": int(sample_label),
                "probe_label": int(probe_label),
                "sample_index": int(rng.choice(class_index[int(sample_label)])),
                "probe_index": int(rng.choice(class_index[int(probe_label)])),
            }
        )
    return pd.DataFrame(rows)


def sample_triplet_specs(
    class_index: Dict[int, List[int]],
    *,
    num_triplets: int,
    num_classes: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(int(seed))
    rows = []
    classes = list(range(int(num_classes)))
    for triplet_id in range(int(num_triplets)):
        probe_label = rng.choice(classes)
        sample_label = rng.choice([label for label in classes if label != probe_label])
        distractor_label = rng.choice([label for label in classes if label not in {probe_label, sample_label}])
        rows.append(
            {
                "triplet_id": int(triplet_id),
                "sample_label": int(sample_label),
                "distractor_label": int(distractor_label),
                "probe_label": int(probe_label),
                "sample_index": int(rng.choice(class_index[int(sample_label)])),
                "distractor_index": int(rng.choice(class_index[int(distractor_label)])),
                "probe_index": int(rng.choice(class_index[int(probe_label)])),
            }
        )
    return pd.DataFrame(rows)


def coords_to_mask(coords, shape: tuple[int, int] = (28, 28)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for row, col in coords:
        if 0 <= int(row) < shape[0] and 0 <= int(col) < shape[1]:
            mask[int(row), int(col)] = 1
    return mask


__all__ = [
    "coords_to_mask",
    "sample_mismatch_pair_specs",
    "sample_triplet_specs",
]

