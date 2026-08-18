from pathlib import Path

from src.data.encoding import _resolve_torchvision_mnist_root


def test_canonical_mnist_root_maps_to_torchvision_parent(tmp_path: Path) -> None:
    canonical = tmp_path / "data" / "MNIST"
    (canonical / "raw").mkdir(parents=True)

    assert Path(_resolve_torchvision_mnist_root(canonical)) == canonical.parent


def test_legacy_nested_mnist_root_remains_supported(tmp_path: Path) -> None:
    legacy = tmp_path / "MNIST"
    (legacy / "MNIST" / "raw").mkdir(parents=True)

    assert Path(_resolve_torchvision_mnist_root(legacy)) == legacy
