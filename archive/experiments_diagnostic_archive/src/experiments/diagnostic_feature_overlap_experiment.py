from __future__ import annotations

import argparse
import warnings

from diagnostic_feature_overlap_voltage_pipeline import build_argparser as build_deterministic_argparser
from diagnostic_feature_overlap_voltage_pipeline import main as deterministic_main


def build_argparser() -> argparse.ArgumentParser:
    warnings.warn(
        "diagnostic_feature_overlap_experiment is deprecated; use the dense-scan diagnostic_feature_overlap_voltage_pipeline instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_deterministic_argparser()


def main() -> None:
    warnings.warn(
        "diagnostic_feature_overlap_experiment is deprecated; forwarding to the dense-scan deterministic pipeline.",
        DeprecationWarning,
        stacklevel=2,
    )
    deterministic_main()


if __name__ == "__main__":
    main()
    
