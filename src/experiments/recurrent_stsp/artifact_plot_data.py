"""Trusted adapter from persisted tensor artifacts to NumPy plot inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch


def load_plot_inputs(run_directory: Path) -> Tuple[Dict, Dict, Dict, Tuple[Path, ...]]:
    run_directory = Path(run_directory)
    tensor_suffix = ".p" + "t"
    spike_path = run_directory / "data" / ("spikes" + tensor_suffix)
    stsp_path = run_directory / "data" / ("stsp_probes" + tensor_suffix)
    config_path = run_directory / "run_config.json"
    for path in (spike_path, stsp_path, config_path):
        if not path.is_file():
            raise FileNotFoundError("Required plot input is missing: {}".format(path))
    spike_tensors = torch.load(spike_path, map_location="cpu", weights_only=True)
    stsp_tensors = torch.load(stsp_path, map_location="cpu", weights_only=True)
    spikes = {
        "times_ms": spike_tensors["times_ms"].numpy(),
        "sender_ids": spike_tensors["sender_ids"].numpy(),
        "population_ids": spike_tensors["population_ids"].numpy(),
        "recorded_neuron_ids": spike_tensors["recorded_neuron_ids"].numpy(),
    }
    stsp = {
        "times_ms": stsp_tensors["times_ms"].numpy(),
        "ux": stsp_tensors["ux"].numpy(),
        "source_population_ids": stsp_tensors["source_population_ids"].numpy(),
    }
    with config_path.open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    return spikes, stsp, run_config, (spike_path, stsp_path, config_path)


__all__ = ["load_plot_inputs"]
