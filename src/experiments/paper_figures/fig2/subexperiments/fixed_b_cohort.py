from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.common.inference import (
    bootstrap_mean_ci,
    exact_sign_flip_p,
    holm_adjust,
    stable_seed,
)
from src.experiments.paper_figures.fig2.artifacts import write_json
from src.experiments.paper_figures.fig2.fixed_b_protocol import (
    CONFIRMATORY_SEEDS,
    DEVELOPMENT_SEEDS,
    FULL_COHORT_SEEDS,
    load_frozen_protocol,
    protocol_digest,
    validate_seed_permission,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_specs import (
    FIXED_B_SCHEMA_VERSION,
)


def aggregate_fixed_b_cohort(
    *,
    figure_root: str | Path,
    protocol_dir: str | Path,
    task_state_path: str | Path,
    aggregate_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Aggregate all 20 frozen v4 endpoints with the network as inference unit.

    Seed 1000 is retained as the protocol-development network and is explicitly
    role-audited. Seeds 1001..1019 remain the untouched confirmatory cohort.
    """

    figure_root = Path(figure_root).resolve()
    protocol = load_frozen_protocol(protocol_dir)
    validate_seed_permission(
        CONFIRMATORY_SEEDS[0],
        task_state_path=task_state_path,
        protocol=protocol,
    )
    expected_digest = protocol_digest(protocol)
    scalar_tables: list[pd.DataFrame] = []
    cohort_rows: list[dict[str, Any]] = []
    for seed in FULL_COHORT_SEEDS:
        seed_dir = figure_root / f"seed_{seed}"
        metrics_dir = seed_dir / "data" / "metrics"
        decision_path = metrics_dir / "fixed_b_single_seed_decision.json"
        scalar_path = metrics_dir / "fixed_b_network_scalars.csv"
        if not decision_path.exists() or not scalar_path.exists():
            raise FileNotFoundError(
                f"Missing fixed-B v4 confirmatory outputs for seed {seed}: "
                f"{decision_path}, {scalar_path}"
            )
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if int(decision.get("network_seed", -1)) != seed:
            raise ValueError(f"Confirmatory seed identity mismatch in {decision_path}")
        if str(decision.get("protocol_digest", "")) != expected_digest:
            raise RuntimeError(f"Confirmatory protocol digest mismatch in {decision_path}")
        if not bool(decision.get("engineering_valid", False)):
            raise RuntimeError(f"Fixed-B seed {seed} is engineering-invalid")
        expected_role = (
            "development_protocol_alignment"
            if seed in set(DEVELOPMENT_SEEDS)
            else "untouched_confirmatory_network"
        )
        scalars = pd.read_csv(
            scalar_path,
            float_precision="round_trip",
        )
        scalars.insert(0, "network_seed", seed)
        scalars.insert(1, "protocol_digest", expected_digest)
        scalar_tables.append(scalars)
        cohort_rows.append(
            {
                "network_seed": seed,
                "seed_role": expected_role,
                "source_seed_role": str(decision.get("seed_role", "")),
                "protocol_digest": expected_digest,
                "engineering_valid": 1,
                "minimum_valid_coverage": float(
                    decision["minimum_valid_coverage"]
                ),
                "decision_path": decision_path.relative_to(
                    figure_root.parent
                ).as_posix(),
            }
        )
    scalar_table = pd.concat(scalar_tables, ignore_index=True)
    found_seeds = set(scalar_table["network_seed"].astype(int))
    if found_seeds != set(FULL_COHORT_SEEDS):
        raise RuntimeError(
            "Fixed-B v4 aggregation input is not exactly the full seeds 1000..1019"
        )
    inference = _network_level_inference(scalar_table)
    confirmatory_scalars = scalar_table.loc[
        scalar_table["network_seed"].astype(int).isin(CONFIRMATORY_SEEDS)
    ].copy()
    confirmatory_inference = _network_level_inference(confirmatory_scalars)
    verdict = _confirmatory_verdict(
        inference,
        scalar_table,
        expected_digest,
        confirmatory_inference=confirmatory_inference,
        confirmatory_scalars=confirmatory_scalars,
    )
    output_dir = (
        Path(aggregate_dir).resolve()
        if aggregate_dir is not None
        else figure_root / "aggregate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cohort_audit": output_dir / "fixed_b_confirmatory_cohort_audit.csv",
        "network_scalars": output_dir / "fixed_b_confirmatory_network_scalars.csv",
        "inference": output_dir / "fixed_b_confirmatory_inference.csv",
        "untouched_confirmatory_network_scalars": (
            output_dir / "fixed_b_untouched_confirmatory_network_scalars.csv"
        ),
        "untouched_confirmatory_inference": (
            output_dir / "fixed_b_untouched_confirmatory_inference.csv"
        ),
        "verdict": output_dir / "fixed_b_confirmatory_verdict.json",
    }
    pd.DataFrame(cohort_rows).to_csv(
        paths["cohort_audit"],
        index=False,
        lineterminator="\n",
    )
    scalar_table.to_csv(
        paths["network_scalars"],
        index=False,
        lineterminator="\n",
    )
    inference.to_csv(paths["inference"], index=False, lineterminator="\n")
    confirmatory_scalars.to_csv(
        paths["untouched_confirmatory_network_scalars"],
        index=False,
        lineterminator="\n",
    )
    confirmatory_inference.to_csv(
        paths["untouched_confirmatory_inference"],
        index=False,
        lineterminator="\n",
    )
    write_json(verdict, paths["verdict"])
    return paths


def _network_level_inference(scalars: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["family", "endpoint", "prefix_k", "role", "threshold"]
    for keys, part in scalars.groupby(group_columns, sort=True):
        family, endpoint, prefix_k, role, threshold = keys
        values = (
            part.sort_values("network_seed")["value"].to_numpy(dtype=np.float64)
        )
        low, high = bootstrap_mean_ci(
            values,
            draws=20_000,
            seed=stable_seed(str(endpoint), int(prefix_k)),
        )
        rows.append(
            {
                "family": str(family),
                "endpoint": str(endpoint),
                "prefix_k": int(prefix_k),
                "role": str(role),
                "threshold": float(threshold),
                "n_networks": int(len(values)),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "ci95_low": low,
                "ci95_high": high,
                "fraction_above_zero": float(np.mean(values > 0)),
                "fraction_meeting_threshold": float(
                    np.mean(values >= float(threshold))
                ),
                "p_one_sided": exact_sign_flip_p(values, alternative="greater"),
                "holm_adjusted_p": float("nan"),
            }
        )
    table = pd.DataFrame(rows)
    core_indices = table.index[table["family"].eq("core_primary")].tolist()
    table.loc[core_indices, "holm_adjusted_p"] = holm_adjust(
        table.loc[core_indices, "p_one_sided"].to_numpy(dtype=np.float64)
    )
    return table


def _confirmatory_verdict(
    inference: pd.DataFrame,
    scalars: pd.DataFrame,
    digest: str,
    *,
    confirmatory_inference: pd.DataFrame | None = None,
    confirmatory_scalars: pd.DataFrame | None = None,
) -> dict[str, Any]:
    full_engineering = bool(
        scalars.loc[
            scalars["endpoint"].eq("all_engineering_gates"),
            "value",
        ].eq(1.0).all()
    )
    full_cohort_complete = bool(
        set(scalars["network_seed"].astype(int)) == set(FULL_COHORT_SEEDS)
    )
    full_common_pass, full_gamma_pass, full_primary_pass = _core_conditions(
        inference
    )
    full_core_pass = bool(
        full_engineering
        and full_cohort_complete
        and full_common_pass
        and full_gamma_pass
        and full_primary_pass
    )

    confirmatory_scalars = (
        confirmatory_scalars
        if confirmatory_scalars is not None
        else scalars.loc[
            scalars["network_seed"].astype(int).isin(CONFIRMATORY_SEEDS)
        ].copy()
    )
    confirmatory_inference = (
        confirmatory_inference
        if confirmatory_inference is not None
        else _network_level_inference(confirmatory_scalars)
    )
    confirmatory_engineering = bool(
        confirmatory_scalars.loc[
            confirmatory_scalars["endpoint"].eq(
                "all_engineering_gates"
            ),
            "value",
        ].eq(1.0).all()
    )
    confirmatory_cohort_complete = bool(
        set(confirmatory_scalars["network_seed"].astype(int))
        == set(CONFIRMATORY_SEEDS)
    )
    common_pass, gamma_sesoi_pass, primary_pass = _core_conditions(
        confirmatory_inference
    )
    core_pass = bool(
        confirmatory_engineering
        and confirmatory_cohort_complete
        and common_pass
        and gamma_sesoi_pass
        and primary_pass
    )
    return {
        "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
        "protocol_digest": digest,
        "full_cohort_seeds": list(FULL_COHORT_SEEDS),
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "n_networks": len(FULL_COHORT_SEEDS),
        "confirmatory_n_networks": len(CONFIRMATORY_SEEDS),
        "development_seed_1000_included": True,
        "development_seed_1000_excluded": False,
        "inference_table_scope": "full_20_network_cohort",
        "seed_roles_audited": True,
        "engineering_valid": full_engineering,
        "cohort_complete": full_cohort_complete,
        "full_cohort_common_update_condition_pass": full_common_pass,
        "full_cohort_gamma_sesoi_pass": full_gamma_pass,
        "full_cohort_all_primary_effects_positive_and_holm_significant": (
            full_primary_pass
        ),
        "full_cohort_core_pass": full_core_pass,
        "confirmatory_engineering_valid": confirmatory_engineering,
        "confirmatory_cohort_complete": confirmatory_cohort_complete,
        "common_update_condition_pass": common_pass,
        "gamma_sesoi_pass": gamma_sesoi_pass,
        "all_primary_effects_positive_and_holm_significant": primary_pass,
        "confirmatory_core_pass": core_pass,
        "verdict": (
            "confirmatory_core_pass"
            if core_pass
            else "confirmatory_core_fail"
        ),
        "inference_unit": "independently_trained_network",
        "test": "exact_one_sided_sign_flip",
        "multiplicity": "Holm_across_8_prespecified_core_effects",
        "optional_stopping": False,
        "outcome_based_exclusions": False,
    }


def _core_conditions(
    inference: pd.DataFrame,
) -> tuple[bool, bool, bool]:
    common = inference.loc[
        inference["endpoint"].eq("same_B_common_update_cosine")
    ]
    primary = inference.loc[inference["family"].eq("core_primary")]
    gamma = primary.loc[
        primary["endpoint"].eq(
            "processing_residual_gamma_energy_fraction"
        )
    ]
    common_pass = bool(
        len(common) == 2
        and common["mean"].ge(common["threshold"]).all()
    )
    gamma_sesoi_pass = bool(
        len(gamma) == 2
        and gamma["mean"].ge(gamma["threshold"]).all()
    )
    primary_pass = bool(
        len(primary) == 8
        and primary["mean"].gt(0).all()
        and primary["holm_adjusted_p"].lt(0.05).all()
    )
    return common_pass, gamma_sesoi_pass, primary_pass


__all__ = [
    "aggregate_fixed_b_cohort",
]
