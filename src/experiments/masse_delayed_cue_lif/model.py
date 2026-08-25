"""Recurrent LIF / SFA-LIF network with SuperSpike surrogate gradients."""

from __future__ import annotations

from typing import NamedTuple, Tuple

import torch
from torch import nn

from .config import (
    N_INPUT,
    N_OUTPUT,
    REFRACTORY_STEPS,
    SFA_BETA,
    SURROGATE_ALPHA,
    V_RESET,
    V_THRESHOLD,
    MasseDelayedCueConfig,
)


def superspike(voltage: torch.Tensor, threshold: torch.Tensor, alpha: float, active: torch.Tensor) -> torch.Tensor:
    delta = voltage - threshold
    hard = (delta >= 0).to(voltage.dtype) * active
    surrogate = alpha / (alpha * delta.abs() + 1.0) ** 2
    return hard + surrogate * (delta - delta.detach()) * active


@torch.jit.script
def scan_lif(
    inputs: torch.Tensor,
    weight_in: torch.Tensor,
    weight_rec: torch.Tensor,
    bias_rec: torch.Tensor,
    weight_out: torch.Tensor,
    bias_out: torch.Tensor,
    recurrent_mask: torch.Tensor,
    sfa_mask: torch.Tensor,
    mem_decay: float,
    syn_decay: float,
    sfa_decay: float,
    readout_decay: float,
    threshold: float,
    reset: float,
    sfa_beta: float,
    alpha: float,
    refractory_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_steps, batch_size, _ = inputs.shape
    n_hidden = weight_rec.shape[0]
    n_output = weight_out.shape[0]
    voltage = torch.zeros(batch_size, n_hidden, device=inputs.device, dtype=inputs.dtype)
    current = torch.zeros(batch_size, n_hidden, device=inputs.device, dtype=inputs.dtype)
    spikes = torch.zeros(batch_size, n_hidden, device=inputs.device, dtype=inputs.dtype)
    adaptation = torch.zeros(batch_size, n_hidden, device=inputs.device, dtype=inputs.dtype)
    refractory = torch.zeros(batch_size, n_hidden, device=inputs.device, dtype=torch.int64)
    readout = torch.zeros(batch_size, n_output, device=inputs.device, dtype=inputs.dtype)
    logit_steps = []
    rec = weight_rec * recurrent_mask
    for _step_index in range(n_steps):
        drive = torch.nn.functional.linear(inputs[_step_index], weight_in, None) + torch.nn.functional.linear(
            spikes.detach(), rec, bias_rec
        )
        current = current * syn_decay + drive
        active = (refractory == 0).to(inputs.dtype)
        voltage = torch.where(
            refractory == 0,
            mem_decay * voltage + (1.0 - mem_decay) * current,
            torch.full_like(voltage, reset),
        )
        thresh = threshold + sfa_beta * adaptation * sfa_mask
        delta = voltage - thresh
        hard = (delta >= 0).to(inputs.dtype) * active
        surrogate = alpha / (alpha * delta.abs() + 1.0) ** 2
        spikes = hard + surrogate * (delta - delta.detach()) * active
        spiked = spikes > 0
        voltage = torch.where(spiked, torch.full_like(voltage, reset), voltage)
        refractory = torch.where(
            spiked,
            torch.full_like(refractory, refractory_steps),
            torch.clamp(refractory - 1, min=0),
        )
        adaptation = adaptation * sfa_decay + spikes.detach()
        readout = readout_decay * readout + (1.0 - readout_decay) * torch.nn.functional.linear(
            spikes, weight_out, bias_out
        )
        logit_steps.append(readout)
    return torch.stack(logit_steps, dim=0), voltage, current, spikes, adaptation, refractory, readout


class RecurrentState(NamedTuple):
    voltage: torch.Tensor
    current: torch.Tensor
    spikes: torch.Tensor
    adaptation: torch.Tensor
    refractory: torch.Tensor
    readout: torch.Tensor


class RecurrentLifSfa(nn.Module):
    def __init__(self, config: MasseDelayedCueConfig):
        super().__init__()
        self.config = config
        self.n_hidden = config.n_hidden
        self.n_sfa = config.n_sfa
        dt = config.dt_ms / 1000.0
        self.mem_decay = float(torch.exp(torch.tensor(-dt / (config.tau_mem_ms / 1000.0))))
        self.syn_decay = float(torch.exp(torch.tensor(-dt / (config.tau_syn_ms / 1000.0))))
        self.sfa_decay = float(torch.exp(torch.tensor(-dt / (config.tau_sfa_ms / 1000.0))))
        self.readout_decay = float(torch.exp(torch.tensor(-dt / (config.tau_readout_ms / 1000.0))))
        self.threshold = V_THRESHOLD
        self.reset = V_RESET
        self.sfa_beta = SFA_BETA
        self.surrogate_alpha = SURROGATE_ALPHA
        self.refractory_steps = REFRACTORY_STEPS

        self.input_linear = nn.Linear(N_INPUT, config.n_hidden, bias=False)
        self.recurrent_linear = nn.Linear(config.n_hidden, config.n_hidden, bias=True)
        self.readout_linear = nn.Linear(config.n_hidden, N_OUTPUT, bias=True)
        mask = torch.ones(config.n_hidden, config.n_hidden)
        mask.fill_diagonal_(0.0)
        self.register_buffer("recurrent_mask", mask)
        sfa_mask = torch.zeros(config.n_hidden)
        sfa_mask[: config.n_sfa] = 1.0
        self.register_buffer("sfa_mask", sfa_mask)
        self.reset_parameters(config.model_init_seed)

    def reset_parameters(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        self._xavier_uniform_(self.input_linear.weight, generator)
        hidden = torch.randn(self.n_hidden, self.n_hidden, generator=generator)
        hidden.fill_diagonal_(0.0)
        spectral_radius = torch.linalg.eigvals(hidden).abs().max().clamp_min(1e-6)
        self.recurrent_linear.weight.data.copy_(hidden * (0.95 / spectral_radius))
        nn.init.constant_(self.recurrent_linear.bias, 0.5)
        self._xavier_uniform_(self.readout_linear.weight, generator)
        nn.init.zeros_(self.readout_linear.bias)

    @staticmethod
    def _xavier_uniform_(tensor: torch.Tensor, generator: torch.Generator) -> None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        tensor.data.uniform_(-bound, bound, generator=generator)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> RecurrentState:
        zeros = torch.zeros(batch_size, self.n_hidden, device=device, dtype=dtype)
        return RecurrentState(
            voltage=zeros.clone(),
            current=zeros.clone(),
            spikes=zeros.clone(),
            adaptation=zeros.clone(),
            refractory=torch.zeros(batch_size, self.n_hidden, device=device, dtype=torch.int64),
            readout=torch.zeros(batch_size, N_OUTPUT, device=device, dtype=dtype),
        )

    def recurrent_weight(self) -> torch.Tensor:
        return self.recurrent_linear.weight * self.recurrent_mask

    def step(self, inputs: torch.Tensor, state: RecurrentState) -> tuple[torch.Tensor, RecurrentState]:
        drive = self.input_linear(inputs) + torch.nn.functional.linear(
            state.spikes.detach(), self.recurrent_weight(), self.recurrent_linear.bias
        )
        current = state.current * self.syn_decay + drive
        active = state.refractory == 0
        voltage = torch.where(
            active,
            self.mem_decay * state.voltage + (1.0 - self.mem_decay) * current,
            torch.full_like(state.voltage, self.reset),
        )
        threshold = self.threshold + self.sfa_beta * state.adaptation * self.sfa_mask
        spikes = superspike(voltage, threshold, self.surrogate_alpha, active.to(voltage.dtype))
        spiked = spikes > 0
        voltage = torch.where(spiked, torch.full_like(voltage, self.reset), voltage)
        refractory = torch.where(
            spiked,
            torch.full_like(state.refractory, self.refractory_steps),
            torch.clamp(state.refractory - 1, min=0),
        )
        adaptation = state.adaptation * self.sfa_decay + spikes.detach()
        readout = self.readout_decay * state.readout + (1.0 - self.readout_decay) * self.readout_linear(spikes)
        next_state = RecurrentState(
            voltage=voltage,
            current=current,
            spikes=spikes,
            adaptation=adaptation,
            refractory=refractory,
            readout=readout,
        )
        return readout, next_state

    def forward(
        self,
        inputs: torch.Tensor,
        state: RecurrentState | None = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        if inputs.dim() != 3:
            raise ValueError("inputs must have shape [time, batch, channels]")
        _n_steps, batch_size, n_channels = inputs.shape
        if n_channels != N_INPUT:
            raise ValueError(f"expected {N_INPUT} input channels, got {n_channels}")
        if state is None:
            logits, voltage, current, spikes, adaptation, refractory, readout = scan_lif(
                inputs,
                self.input_linear.weight,
                self.recurrent_linear.weight,
                self.recurrent_linear.bias,
                self.readout_linear.weight,
                self.readout_linear.bias,
                self.recurrent_mask,
                self.sfa_mask,
                self.mem_decay,
                self.syn_decay,
                self.sfa_decay,
                self.readout_decay,
                float(self.threshold),
                float(self.reset),
                float(self.sfa_beta),
                float(self.surrogate_alpha),
                int(self.refractory_steps),
            )
            final_state = RecurrentState(
                voltage=voltage,
                current=current,
                spikes=spikes,
                adaptation=adaptation,
                refractory=refractory,
                readout=readout,
            )
            return logits, final_state
        logits = []
        for step_index in range(inputs.shape[0]):
            logit, state = self.step(inputs[step_index], state)
            logits.append(logit)
        return torch.stack(logits, dim=0), state
