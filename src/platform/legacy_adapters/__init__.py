"""Compatibility adapters for legacy root-level platform modules."""

from .encoding import DoGSpikeEncoder, build_fashionmnist_skeleton_loader, build_mnist_skeleton_loader
from .network import SDNN_Network, lif_dynamics_jit, stsp_dynamics_jit
from .units import mV, mvolt, mm, ms, nF, nS, pA

__all__ = [
    "DoGSpikeEncoder",
    "SDNN_Network",
    "build_fashionmnist_skeleton_loader",
    "build_mnist_skeleton_loader",
    "lif_dynamics_jit",
    "mV",
    "mm",
    "ms",
    "mvolt",
    "nF",
    "nS",
    "pA",
    "stsp_dynamics_jit",
]

