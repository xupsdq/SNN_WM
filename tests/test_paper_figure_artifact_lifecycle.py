from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from src.experiments.paper_figures.common.artifact_runtime import (
    cache_key_digest,
    materialize_artifact,
    read_cache_key,
    write_cache_key,
)
from src.experiments.paper_figures.fig1 import artifacts as fig1_artifacts
from src.experiments.paper_figures.fig1.cache_keys import cache_key_digest as fig1_cache_key_digest


FIG1_CACHE_KEY = {
    "schema_name": "fig1_runtime_artifacts",
    "schema_version": 1,
    "task_id": "trial_specs",
    "network_seed": 1000,
    "delay_points_ms": [0, 50, 100],
    "extra": {"dataset_id": "mnist-test", "pair": (3, 7)},
}
FIG1_CACHE_KEY_DIGEST = "345219a8b121fd08735a345168ab76ad6400d9e185fc0af2a1851c247b17e56e"
FIGURE_CACHE_KEY_DIGESTS = {
    2: "0d127696eb36871bd34b9c71f6c2d5f123a6fb3b3cf3704b15bae5f6223cc20f",
    3: "d1b3511ceb0d69c09b5a48e4167b55b50293a3d62008e67e0bfc864c2fbff51b",
    4: "ad117ec2150376ef9603bd22a60afcfceaa2673a1bcdba5570de6eafe25f2137",
    5: "64361847c05e7d9f4b16db5dddc2ea6f1039fa55dc2230c160e8b500c8dd5d5f",
    6: "f80f45e25193ddeb5f28b9b5a0416c3bb2e7ee52137ff9be959d2e81e40291fa",
}


def test_fig1_cache_identity_matches_frozen_contract(tmp_path: Path) -> None:
    task_dir = tmp_path / "trial_specs"

    write_cache_key(task_dir, FIG1_CACHE_KEY)

    assert cache_key_digest(FIG1_CACHE_KEY) == FIG1_CACHE_KEY_DIGEST
    assert fig1_cache_key_digest(FIG1_CACHE_KEY) == FIG1_CACHE_KEY_DIGEST
    assert read_cache_key(task_dir) == {
        "cache_key": {
            **FIG1_CACHE_KEY,
            "extra": {"dataset_id": "mnist-test", "pair": [3, 7]},
        },
        "cache_key_digest": FIG1_CACHE_KEY_DIGEST,
    }


@pytest.mark.parametrize(
    ("mode", "has_cache", "expected"),
    (
        ("off", False, "fresh"),
        ("auto", False, "built"),
        ("auto", True, "loaded"),
        ("require", True, "loaded"),
        ("force", True, "built"),
    ),
)
def test_artifact_lifecycle_preserves_fig1_mode_results(
    tmp_path: Path,
    mode: str,
    has_cache: bool,
    expected: str,
) -> None:
    task_dir = tmp_path / "artifact"
    if has_cache:
        write_cache_key(task_dir, FIG1_CACHE_KEY)

    result = materialize_artifact(
        mode=mode,
        task_dir=task_dir,
        expected_key=FIG1_CACHE_KEY,
        load=lambda: "loaded",
        build=lambda: "built",
        fresh=lambda: "fresh",
    )

    assert result == expected


def test_require_rejects_missing_artifact_without_building(tmp_path: Path) -> None:
    def reject_build() -> str:
        raise AssertionError("require mode must not build")

    with pytest.raises(FileNotFoundError, match="Artifact cache key is missing"):
        materialize_artifact(
            mode="require",
            task_dir=tmp_path / "missing",
            expected_key=FIG1_CACHE_KEY,
            load=lambda: "loaded",
            build=reject_build,
        )


def test_auto_rebuilds_when_matching_artifact_payload_is_invalid(tmp_path: Path) -> None:
    task_dir = tmp_path / "artifact"
    write_cache_key(task_dir, FIG1_CACHE_KEY)

    def reject_load() -> str:
        raise ValueError("invalid payload")

    assert materialize_artifact(
        mode="auto",
        task_dir=task_dir,
        expected_key=FIG1_CACHE_KEY,
        load=reject_load,
        build=lambda: "built",
    ) == "built"


def test_force_can_load_an_existing_parent_artifact(tmp_path: Path) -> None:
    task_dir = tmp_path / "trial_specs"
    write_cache_key(task_dir, FIG1_CACHE_KEY)

    assert materialize_artifact(
        mode="force",
        task_dir=task_dir,
        expected_key=FIG1_CACHE_KEY,
        load=lambda: "loaded-parent",
        build=lambda: "built-parent",
        force_load_existing=True,
    ) == "loaded-parent"


def test_fig1_artifact_adapter_keeps_existing_cache_interface(tmp_path: Path) -> None:
    task_dir = tmp_path / "trial_specs"

    fig1_artifacts.write_cache_key(task_dir, FIG1_CACHE_KEY)

    assert fig1_artifacts.cache_key_matches(task_dir, FIG1_CACHE_KEY)
    assert fig1_artifacts.read_cache_key(task_dir)["cache_key_digest"] == FIG1_CACHE_KEY_DIGEST


def test_second_adapter_can_preserve_its_cache_mismatch_guidance(tmp_path: Path) -> None:
    task_dir = tmp_path / "artifact"
    write_cache_key(task_dir, {**FIG1_CACHE_KEY, "network_seed": 1001})

    with pytest.raises(RuntimeError, match="Rebuild the producer task"):
        materialize_artifact(
            mode="require",
            task_dir=task_dir,
            expected_key=FIG1_CACHE_KEY,
            load=lambda: "loaded",
            build=lambda: "built",
            cache_mismatch_hint="Rebuild the producer task before using --reuse-artifacts require.",
        )


def test_auto_can_preserve_fail_fast_payload_validation(tmp_path: Path) -> None:
    task_dir = tmp_path / "artifact"
    write_cache_key(task_dir, FIG1_CACHE_KEY)

    def reject_load() -> str:
        raise ValueError("invalid payload")

    with pytest.raises(ValueError, match="invalid payload"):
        materialize_artifact(
            mode="auto",
            task_dir=task_dir,
            expected_key=FIG1_CACHE_KEY,
            load=reject_load,
            build=lambda: "built",
            recover_auto_load_errors=False,
        )


def test_adapter_can_preserve_figure_specific_cache_validation(tmp_path: Path) -> None:
    task_dir = tmp_path / "artifact"
    events: list[str] = []

    def require_reusable() -> None:
        events.append("require")
        raise ValueError("figure-specific mismatch")

    with pytest.raises(ValueError, match="figure-specific mismatch"):
        materialize_artifact(
            mode="require",
            task_dir=task_dir,
            expected_key=FIG1_CACHE_KEY,
            load=lambda: "loaded",
            build=lambda: "built",
            cache_is_reusable=lambda: True,
            require_reusable=require_reusable,
        )

    assert events == ["require"]


def test_common_cache_identity_preserves_numpy_payloads() -> None:
    key = {
        "task_id": "numpy_fixture",
        "array": np.asarray([1, 2], dtype=np.int64),
        "scalar": np.float32(1.5),
    }

    assert cache_key_digest(key) == "1a611da7c822543597742602b21dcf751c8a4046dac37bd56d840e340d0849d6"


@pytest.mark.parametrize("figure_number", tuple(FIGURE_CACHE_KEY_DIGESTS))
def test_figure_cache_identity_matches_frozen_contract(figure_number: int) -> None:
    schema_version = 4 if figure_number == 3 else 1
    key = {
        "schema_name": f"fig{figure_number}_runtime_artifacts",
        "schema_version": schema_version,
        "task_id": "fixture_specs",
        "network_seed": 1000,
        "delay_points_ms": [0, 50, 100],
        "extra": {"dataset_id": "mnist-test", "pair": (3, 7)},
    }
    module = importlib.import_module(f"src.experiments.paper_figures.fig{figure_number}.cache_keys")

    assert module.cache_key_digest(key) == FIGURE_CACHE_KEY_DIGESTS[figure_number]


@pytest.mark.parametrize("figure_number", tuple(FIGURE_CACHE_KEY_DIGESTS))
def test_figure_artifact_adapters_keep_cache_file_contract(tmp_path: Path, figure_number: int) -> None:
    schema_version = 4 if figure_number == 3 else 1
    key = {
        "schema_name": f"fig{figure_number}_runtime_artifacts",
        "schema_version": schema_version,
        "task_id": "fixture_specs",
        "network_seed": 1000,
        "delay_points_ms": [0, 50, 100],
        "extra": {"dataset_id": "mnist-test", "pair": (3, 7)},
    }
    module = importlib.import_module(f"src.experiments.paper_figures.fig{figure_number}.artifacts")
    task_dir = tmp_path / f"fig{figure_number}"

    module.write_cache_key(task_dir, key)

    assert module.cache_key_matches(task_dir, key)
    assert module.read_cache_key(task_dir)["cache_key_digest"] == FIGURE_CACHE_KEY_DIGESTS[figure_number]
