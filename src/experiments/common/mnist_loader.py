from __future__ import annotations

from src.data.encoding import build_mnist_skeleton_loader


def load_mnist_skeleton_dataset(
    dataset_root: str,
    split: str,
):
    train_loader, _, test_loader = build_mnist_skeleton_loader(
        root=dataset_root,
        batch_size=1,
        input_size=28,
        num_workers=0,
    )
    split_name = str(split).strip().lower()
    if split_name == "train":
        return train_loader.dataset
    if split_name == "test":
        return test_loader.dataset
    raise ValueError(f"Unsupported split: {split}")


__all__ = ["load_mnist_skeleton_dataset"]
