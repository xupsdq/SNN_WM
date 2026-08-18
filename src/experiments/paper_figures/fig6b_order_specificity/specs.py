"""Stimulus specifications for the fixed-set, fixed-latest temporal-order pilot.

Design rules (pre-registered in meta/analysis_spec.json):
- K = 4; each item set contains exactly four fixed items {A, B, C, D}.
- D is always the last input; A, B, C traverse all six orders.
- All six order conditions share the item set, latest item, sequence length,
  delay, stimulus timing and simulation parameters; only the A/B/C -> slot
  assignment varies.
- Each set uses four distinct MNIST image categories; image IDs never overlap
  across sets; the latest-item category is balanced across sets (round-robin).
- Stimulus specifications are generated from a fixed stimulus-spec seed that is
  independent of the network seed, so every network uses identical controlled
  stimulus specifications.
- Every image ID, label, role and seed is persisted.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig6b_order_specificity.types import (
    ITEM_ROLES,
    N_ORDERS,
    NUM_CLASSES,
    ORDER_NAMES,
    ORDER_PERMUTATIONS,
    SEQUENCE_LENGTH,
    OrderSpecificityConfig,
)


def _stable_seed(base: int, *parts: str) -> int:
    payload = ":".join([str(base), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def build_stimulus_specs(
    cfg: OrderSpecificityConfig,
    class_index: dict[int, list[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build sequence and singleton-reference specs shared by all networks."""
    rng = np.random.default_rng(int(cfg.stimulus_spec_seed))
    used_image_ids: set[int] = set()

    set_rows: list[dict[str, Any]] = []  # one row per set (pre-network expansion)
    ref_rows: list[dict[str, Any]] = []  # one row per reference (pre-expansion)

    for set_id in range(int(cfg.num_sets)):
        latest_label = int(set_id % NUM_CLASSES)
        other_labels = [
            int(value)
            for value in rng.choice(
                [label for label in range(NUM_CLASSES) if label != latest_label],
                size=3,
                replace=False,
            )
        ]
        role_labels = {
            "A": other_labels[0],
            "B": other_labels[1],
            "C": other_labels[2],
            "D": latest_label,
        }
        role_image_ids: dict[str, int] = {}
        for role in ITEM_ROLES:
            label = role_labels[role]
            image_id = _sample_unused_image(rng, class_index[label], used_image_ids)
            role_image_ids[role] = image_id
            used_image_ids.add(image_id)

        ordered_ids = [role_image_ids[role] for role in ITEM_ROLES]
        ordered_labels = [role_labels[role] for role in ITEM_ROLES]
        set_rows.append(
            {
                "set_id": int(set_id),
                "item_a_image_id": int(role_image_ids["A"]),
                "item_b_image_id": int(role_image_ids["B"]),
                "item_c_image_id": int(role_image_ids["C"]),
                "latest_image_id": int(role_image_ids["D"]),
                "item_a_label": int(role_labels["A"]),
                "item_b_label": int(role_labels["B"]),
                "item_c_label": int(role_labels["C"]),
                "latest_label": int(role_labels["D"]),
                "ordered_item_ids": ";".join(str(v) for v in ordered_ids),
                "ordered_item_labels": ";".join(str(v) for v in ordered_labels),
            }
        )
        for role, slot in (("A", 1), ("A", 2), ("A", 3), ("B", 1), ("B", 2), ("B", 3), ("C", 1), ("C", 2), ("C", 3), ("D", 4)):
            ref_rows.append(
                {
                    "set_id": int(set_id),
                    "item_role": role,
                    "item_image_id": int(role_image_ids[role]),
                    "item_label": int(role_labels[role]),
                    "temporal_slot": int(slot),
                    "reference_seed": _stable_seed(cfg.stimulus_spec_seed, "ref", str(set_id), role, str(slot)),
                }
            )

    set_frame = pd.DataFrame(set_rows)
    ref_frame = pd.DataFrame(ref_rows)

    # Expand per network seed: every network uses the identical stimulus specs.
    sequence_rows: list[dict[str, Any]] = []
    singleton_rows: list[dict[str, Any]] = []
    for network_seed in cfg.expected_network_seeds:
        for row in set_frame.itertuples(index=False):
            ordered_ids = [int(v) for v in str(row.ordered_item_ids).split(";")]
            ordered_labels = [int(v) for v in str(row.ordered_item_labels).split(";")]
            for order_index, permutation in enumerate(ORDER_PERMUTATIONS):
                ids_by_slot = {role: ordered_ids[ITEM_ROLES.index(role)] for role in ITEM_ROLES}
                order_ids = [ids_by_slot[role] for role in permutation] + [ids_by_slot["D"]]
                order_labels = [ordered_labels[ITEM_ROLES.index(role)] for role in permutation] + [ordered_labels[3]]
                sequence_rows.append(
                    {
                        "network_seed": int(network_seed),
                        "set_id": int(row.set_id),
                        "order_index": int(order_index),
                        "order_name": ORDER_NAMES[order_index],
                        "ordered_item_ids": ";".join(str(v) for v in order_ids),
                        "ordered_item_labels": ";".join(str(v) for v in order_labels),
                        "latest_item_id": int(ids_by_slot["D"]),
                        "latest_item_label": int(ordered_labels[3]),
                        "seq_len": int(cfg.sequence_length),
                        "sample_ms": int(cfg.sample_ms),
                        "delay_ms": int(cfg.delay_ms),
                        "stimulus_spec_seed": int(cfg.stimulus_spec_seed),
                        "sequence_seed": _stable_seed(cfg.stimulus_spec_seed, "order", str(row.set_id), str(order_index)),
                    }
                )
        for row in ref_frame.itertuples(index=False):
            singleton_rows.append(
                {
                    "network_seed": int(network_seed),
                    "set_id": int(row.set_id),
                    "item_role": str(row.item_role),
                    "item_image_id": int(row.item_image_id),
                    "item_label": int(row.item_label),
                    "temporal_slot": int(row.temporal_slot),
                    "reference_seed": int(row.reference_seed),
                    "seq_len": int(cfg.sequence_length),
                    "sample_ms": int(cfg.sample_ms),
                    "delay_ms": int(cfg.delay_ms),
                    "stimulus_spec_seed": int(cfg.stimulus_spec_seed),
                    "purpose": "candidate_shared_reference",
                }
            )

    sequence_specs = pd.DataFrame(sequence_rows)
    reference_specs = pd.DataFrame(singleton_rows)
    return sequence_specs, reference_specs


def _sample_unused_image(
    rng: np.random.Generator,
    class_pool: Iterable[int],
    used_image_ids: set[int],
) -> int:
    pool = [int(v) for v in class_pool if int(v) not in used_image_ids]
    if not pool:
        raise RuntimeError("Exhausted image pool for a class while building stimulus specs.")
    return int(rng.choice(pool))


def validate_stimulus_specs(
    cfg: OrderSpecificityConfig,
    sequence_specs: pd.DataFrame,
    reference_specs: pd.DataFrame,
) -> pd.DataFrame:
    """Structural validation of the fixed-set design; raises on hard failures."""
    rows: list[dict[str, Any]] = []

    def _check(check_id: str, description: str, passed: bool, detail: str = "") -> None:
        rows.append(
            {
                "check_id": check_id,
                "description": description,
                "passed": bool(passed),
                "detail": detail,
            }
        )
        if not passed:
            raise RuntimeError(f"Stimulus-spec check failed: {check_id} ({description}) {detail}")

    n_networks = int(sequence_specs["network_seed"].nunique())
    _check("network_count", "specs exist for every expected network", n_networks == len(cfg.expected_network_seeds),
           f"expected={len(cfg.expected_network_seeds)} observed={n_networks}")

    counts = sequence_specs.groupby(["network_seed", "set_id"], sort=True).size()
    _check("six_orders_per_set", "each set has exactly 6 order conditions", bool((counts == int(cfg.num_orders)).all()),
           f"counts={counts.to_dict()}")

    for (network_seed, set_id), part in sequence_specs.groupby(["network_seed", "set_id"], sort=True):
        _check(
            "latest_is_last",
            "D is always the last input",
            bool((part["ordered_item_ids"].str.split(";").str[-1].astype(int) == part["latest_item_id"]).all())
            and bool((part["ordered_item_labels"].str.split(";").str[-1].astype(int) == part["latest_item_label"]).all()),
            f"network={network_seed} set={set_id}",
        )
        _check(
            "unique_orders",
            "the six historical orders are unique within a set",
            part["order_name"].nunique() == int(cfg.num_orders),
            f"network={network_seed} set={set_id}",
        )
        _check(
            "identical_item_set",
            "item set is identical across all six orders",
            part["ordered_item_ids"].apply(lambda s: frozenset(str(s).split(";"))).nunique() == 1,
            f"network={network_seed} set={set_id}",
        )
        _check(
            "latest_shared_across_orders",
            "latest item is identical across the six orders",
            part["latest_item_id"].nunique() == 1,
            f"network={network_seed} set={set_id}",
        )
        _check(
            "four_distinct_labels",
            "the set uses four distinct MNIST categories",
            part["ordered_item_labels"].apply(lambda s: len(set(str(s).split(";")))).eq(4).all(),
            f"network={network_seed} set={set_id}",
        )

    all_ids = sequence_specs.drop_duplicates("set_id")["ordered_item_ids"].str.split(";").explode().astype(int)
    _check(
        "image_ids_disjoint_across_sets",
        "image IDs never overlap across sets",
        int(all_ids.nunique()) == int(cfg.num_sets) * 4,
        f"expected={int(cfg.num_sets) * 4} unique={int(all_ids.nunique())}",
    )

    latest_counts = sequence_specs.drop_duplicates("set_id").groupby("latest_item_label").size()
    _check(
        "latest_label_balance",
        "latest-item category is balanced across sets",
        bool(len(latest_counts) and int(latest_counts.max() - latest_counts.min()) <= 2),
        f"counts={latest_counts.to_dict()}",
    )

    ref_counts = reference_specs.groupby(["network_seed", "set_id"], sort=True).size()
    _check("ten_refs_per_set", "each set has 10 singleton references", bool((ref_counts == 10).all()),
           f"counts={ref_counts.to_dict()}")
    expected_refs = {
        (role, slot)
        for role in ("A", "B", "C")
        for slot in (1, 2, 3)
    } | {("D", 4)}
    for (network_seed, set_id), part in reference_specs.groupby(["network_seed", "set_id"], sort=True):
        observed_refs = set(zip(part["item_role"].astype(str), part["temporal_slot"].astype(int)))
        _check(
            "ref_slot_coverage",
            "references cover every item x slot used by the candidates",
            observed_refs == expected_refs,
            f"network={network_seed} set={set_id} missing={sorted(expected_refs - observed_refs)}",
        )
        set_row = sequence_specs.loc[
            sequence_specs["network_seed"].eq(network_seed) & sequence_specs["set_id"].eq(set_id)
        ].iloc[0]
        set_ids = {
            "A": int(set_row["ordered_item_ids"].split(";")[0]),
            "B": int(set_row["ordered_item_ids"].split(";")[1]),
            "C": int(set_row["ordered_item_ids"].split(";")[2]),
            "D": int(set_row["ordered_item_ids"].split(";")[3]),
        }
        for ref in part.itertuples(index=False):
            _check(
                "ref_matches_set_item",
                "reference items match the set items by role",
                int(ref.item_image_id) == set_ids[str(ref.item_role)],
                f"network={network_seed} set={set_id} role={ref.item_role}",
            )

    _check(
        "ref_seed_independent",
        "reference seeds are independent of order-trial seeds",
        int(len(set(sequence_specs["sequence_seed"]) & set(reference_specs["reference_seed"]))) == 0,
    )
    return pd.DataFrame(rows)


def specs_digest(
    cfg: OrderSpecificityConfig,
    sequence_specs: pd.DataFrame,
    reference_specs: pd.DataFrame,
) -> str:
    payload = [
        str(cfg.stimulus_spec_seed),
        str(cfg.num_sets),
        str(cfg.num_orders),
        str(cfg.sequence_length),
        str(cfg.sample_ms),
        str(cfg.delay_ms),
    ]
    for frame in (sequence_specs, reference_specs):
        payload.append(
            hashlib.sha256(
                frame.to_csv(index=False).encode("utf-8")
            ).hexdigest()
        )
    return hashlib.sha256(":".join(payload).encode("utf-8")).hexdigest()


def specs_cache_key(cfg: OrderSpecificityConfig) -> dict[str, Any]:
    return {
        "stimulus_spec_seed": int(cfg.stimulus_spec_seed),
        "num_sets": int(cfg.num_sets),
        "num_orders": int(cfg.num_orders),
        "sequence_length": int(cfg.sequence_length),
        "sample_ms": int(cfg.sample_ms),
        "delay_ms": int(cfg.delay_ms),
        "dt": float(cfg.dt),
        "split": str(cfg.split),
        "expected_network_seeds": list(cfg.expected_network_seeds),
    }


def cache_key_digest(key: Mapping[str, Any]) -> str:
    payload = repr(sorted(key.items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_safe(payload: Any) -> Any:
    return _to_json_safe(payload)


__all__ = [
    "build_stimulus_specs",
    "cache_key_digest",
    "json_safe",
    "specs_cache_key",
    "specs_digest",
    "validate_stimulus_specs",
]
