"""Masse delayed-cue DMS+DMRS recurrent LIF experiment package."""

from .config import MasseDelayedCueConfig, formal_config, overfit_config, smoke_config
from .evaluate import evaluate_run
from .model import RecurrentLifSfa
from .plot import plot_run
from .run import build_trials, main
from .task import expand_trial, generate_trial_table, matching_direction
from .train import train_run

__all__ = [
    "MasseDelayedCueConfig",
    "RecurrentLifSfa",
    "build_trials",
    "evaluate_run",
    "expand_trial",
    "formal_config",
    "generate_trial_table",
    "main",
    "matching_direction",
    "overfit_config",
    "plot_run",
    "smoke_config",
    "train_run",
]
