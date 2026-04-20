"""Compatibility adapter for legacy ``input_function`` imports.

This module provides a stable src-side entrypoint for encoding and skeleton
loader APIs while deferring to ``src.data.encoding`` as the implementation
source.
"""

from src.data.encoding import DoGSpikeEncoder, build_fashionmnist_skeleton_loader, build_mnist_skeleton_loader

__all__ = ["DoGSpikeEncoder", "build_fashionmnist_skeleton_loader", "build_mnist_skeleton_loader"]
