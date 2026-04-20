from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config.units import ms
from src.experiments.common.dataset import build_class_index as shared_build_class_index
from src.experiments.common.dataset import encode_images as shared_encode_images
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.common.model_io import load_model_and_encoder as shared_load_model_and_encoder
from src.experiments.common.runtime import seed_everything as shared_seed_everything
from src.platform.legacy_adapters.encoding import DoGSpikeEncoder
from src.platform.legacy_adapters.network import SDNN_Network


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay1_ms: float
    distractor_ms: float
    delay2_ms: float
    probe_ms: float
    phase_reset: bool

    @property
    def sample_steps(self) -> int:
        return int((self.sample_ms * ms) / self.dt)

    @property
    def delay1_steps(self) -> int:
        return int((self.delay1_ms * ms) / self.dt)

    @property
    def distractor_steps(self) -> int:
        return int((self.distractor_ms * ms) / self.dt)

    @property
    def delay2_steps(self) -> int:
        return int((self.delay2_ms * ms) / self.dt)

    @property
    def probe_steps(self) -> int:
        return int((self.probe_ms * ms) / self.dt)

    @property
    def clean_delay_steps(self) -> int:
        return self.delay1_steps + self.distractor_steps + self.delay2_steps


seed_everything = shared_seed_everything
build_class_index = shared_build_class_index


def load_model_and_encoder(model_path: str, device: torch.device, spec: ExperimentSpec) -> Tuple[SDNN_Network, DoGSpikeEncoder]:
    return shared_load_model_and_encoder(
        model_path=model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(spec.sample_ms, spec.distractor_ms, spec.probe_ms),
    )


def generate_trial_specs(
    class_index: Dict[int, List[int]],
    num_trials: int,
    num_classes: int,
    rng: random.Random,
) -> pd.DataFrame:
    rows: List[Dict[str, int]] = []
    classes = list(range(num_classes))
    for trial_id in range(num_trials):
        sample_label = rng.choice(classes)
        distractor_candidates = [c for c in classes if c != sample_label]
        distractor_label = rng.choice(distractor_candidates)
        probe_candidates = [c for c in classes if c != sample_label and c != distractor_label]
        probe_label = rng.choice(probe_candidates)
        rows.append(
            {
                "trial_id": int(trial_id),
                "sample_index": int(rng.choice(class_index[sample_label])),
                "sample_label": int(sample_label),
                "distractor_index": int(rng.choice(class_index[distractor_label])),
                "distractor_label": int(distractor_label),
                "probe_index": int(rng.choice(class_index[probe_label])),
                "probe_label": int(probe_label),
            }
        )
    return pd.DataFrame(rows)


def validate_trial_specs(df_specs: pd.DataFrame, num_classes: int) -> None:
    if df_specs["trial_id"].nunique() != len(df_specs):
        raise ValueError("trial_id must be unique in trial specs")
    sample_lbl = df_specs["sample_label"].to_numpy()
    distractor_lbl = df_specs["distractor_label"].to_numpy()
    probe_lbl = df_specs["probe_label"].to_numpy()
    if not np.all(sample_lbl != distractor_lbl):
        raise ValueError("Found trial(s) where sample_label == distractor_label")
    if not np.all(sample_lbl != probe_lbl):
        raise ValueError("Found trial(s) where sample_label == probe_label")
    if not np.all(distractor_lbl != probe_lbl):
        raise ValueError("Found trial(s) where distractor_label == probe_label")
    for col in ["sample_label", "distractor_label", "probe_label"]:
        vals = df_specs[col].to_numpy()
        if (vals < 0).any() or (vals >= num_classes).any():
            raise ValueError(f"{col} contains out-of-range class index")


def run_interface_check(net: SDNN_Network, device: torch.device) -> None:
    with torch.no_grad():
        bsz, c, h, w = 2, 2, 28, 28
        sample = (torch.rand((bsz, 20, c, h, w), device=device) > 0.95).float()
        distractor = (torch.rand((bsz, 20, c, h, w), device=device) > 0.95).float()
        probe = (torch.rand((bsz, 20, c, h, w), device=device) > 0.95).float()
        out = net.forward_dual_task_session(
            sample_spikes=sample,
            distractor_spikes=distractor,
            probe_spikes=probe,
            delay1_steps=10,
            delay2_steps=10,
            stsp_mode="static_frozen",
            phase_reset=True,
        )

    required_keys = {
        "prediction_distractor",
        "prediction_probe",
        "first_fire_t_distractor",
        "first_fire_t_probe",
    }
    if set(out.keys()) != required_keys:
        raise ValueError(f"forward_dual_task_session keys mismatch: {set(out.keys())}")
    for key in required_keys:
        tensor = out[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{key} is not a tensor")
        if tensor.shape != (bsz,):
            raise ValueError(f"{key} shape mismatch: {tensor.shape}")
        if tensor.dtype != torch.long:
            raise ValueError(f"{key} dtype mismatch: {tensor.dtype}")


def run_experiment(
    net: SDNN_Network,
    encoder: DoGSpikeEncoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    all_records: List[Dict[str, int]] = []
    for start in tqdm(range(0, len(df_specs), batch_size), desc="DualTask Batches"):
        batch = df_specs.iloc[start : start + batch_size]
        bsz = len(batch)
        sample_imgs = torch.stack([dataset[int(i)][0] for i in batch["sample_index"].tolist()], dim=0).to(device)
        distractor_imgs = torch.stack([dataset[int(i)][0] for i in batch["distractor_index"].tolist()], dim=0).to(device)
        probe_imgs = torch.stack([dataset[int(i)][0] for i in batch["probe_index"].tolist()], dim=0).to(device)

        sample_spikes = shared_encode_images(encoder, sample_imgs, spec.sample_steps)
        distractor_spikes = shared_encode_images(encoder, distractor_imgs, spec.distractor_steps)
        probe_spikes = shared_encode_images(encoder, probe_imgs, spec.probe_steps)

        for paradigm, stsp_mode in [
            ("clean", "static_frozen"),
            ("clean", "dynamic"),
            ("distracted", "static_frozen"),
            ("distracted", "dynamic"),
        ]:
            if paradigm == "clean":
                with torch.no_grad():
                    out = net.forward_classify_session(
                        sample_spikes=sample_spikes,
                        test_spikes=probe_spikes,
                        delay_duration_steps=spec.clean_delay_steps,
                        stsp_mode=stsp_mode,
                    )
                pred_probe = out["prediction"].detach().cpu().long()
                pred_distractor = torch.full((bsz,), -1, dtype=torch.long)
                fire_t_probe = decode_prediction_and_fire_time_from_layer3(net, bsz)[1]
                fire_t_distractor = torch.full((bsz,), -1, dtype=torch.long)
            else:
                with torch.no_grad():
                    out = net.forward_dual_task_session(
                        sample_spikes=sample_spikes,
                        distractor_spikes=distractor_spikes,
                        probe_spikes=probe_spikes,
                        delay1_steps=spec.delay1_steps,
                        delay2_steps=spec.delay2_steps,
                        stsp_mode=stsp_mode,
                        phase_reset=spec.phase_reset,
                    )
                pred_distractor = out["prediction_distractor"].detach().cpu().long()
                pred_probe = out["prediction_probe"].detach().cpu().long()
                fire_t_distractor = out["first_fire_t_distractor"].detach().cpu().long()
                fire_t_probe = out["first_fire_t_probe"].detach().cpu().long()

            sample_lbl = batch["sample_label"].to_numpy()
            distractor_lbl = batch["distractor_label"].to_numpy()
            probe_lbl = batch["probe_label"].to_numpy()
            trial_ids = batch["trial_id"].to_numpy()
            for i in range(bsz):
                pd_i = int(pred_distractor[i].item())
                pp_i = int(pred_probe[i].item())
                y_d = int(distractor_lbl[i])
                y_p = int(probe_lbl[i])
                all_records.append(
                    {
                        "trial_id": int(trial_ids[i]),
                        "paradigm": paradigm,
                        "stsp_mode": stsp_mode,
                        "sample_label": int(sample_lbl[i]),
                        "distractor_label": y_d,
                        "probe_label": y_p,
                        "prediction_distractor": pd_i,
                        "prediction_probe": pp_i,
                        "first_fire_t_distractor": int(fire_t_distractor[i].item()),
                        "first_fire_t_probe": int(fire_t_probe[i].item()),
                        "is_correct_distractor": int(pd_i == y_d) if paradigm == "distracted" else -1,
                        "is_correct_probe": int(pp_i == y_p),
                        "is_silent_distractor": int(pd_i == -1) if paradigm == "distracted" else -1,
                        "is_silent_probe": int(pp_i == -1),
                    }
                )
    return pd.DataFrame(all_records).sort_values(["trial_id", "paradigm", "stsp_mode"]).reset_index(drop=True)


def validate_pairing(df_trials: pd.DataFrame) -> None:
    count_per_trial = df_trials.groupby("trial_id").size()
    if not (count_per_trial == 4).all():
        bad_ids = count_per_trial[count_per_trial != 4].index.tolist()
        raise ValueError(f"Each trial_id must appear exactly 4 times. Bad ids: {bad_ids[:10]}")
    for col in ["sample_label", "distractor_label", "probe_label"]:
        uniq = df_trials.groupby("trial_id")[col].nunique()
        if not (uniq == 1).all():
            bad_ids = uniq[uniq != 1].index.tolist()
            raise ValueError(f"{col} is not paired-identical across conditions for ids: {bad_ids[:10]}")


__all__ = [
    "ExperimentSpec",
    "build_class_index",
    "generate_trial_specs",
    "load_model_and_encoder",
    "run_experiment",
    "run_interface_check",
    "seed_everything",
    "validate_pairing",
    "validate_trial_specs",
]
