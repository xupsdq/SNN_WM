"""Compatibility adapter for legacy ``unit_setting`` imports.

This module keeps unit constants available from a src-side path while treating
``src.config.units`` as the official source of truth.
"""

from src.config.units import mV, mvolt, mm, ms, nF, nS, pA

__all__ = ["mV", "mvolt", "nF", "nS", "ms", "pA", "mm"]
