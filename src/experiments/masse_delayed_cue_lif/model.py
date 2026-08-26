"""Recurrent LIF / SFA-LIF network with SuperSpike surrogate gradients."""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
from torch import nn

from .config import (
    N_INPUT,
    N_OUTPUT,
    REFRACTORY_STEPS,
    SFA_BETA,
    STSP_TAU_U_DEP_MS,
    STSP_TAU_U_FAC_MS,
    STSP_TAU_X_DEP_MS,
    STSP_TAU_X_FAC_MS,
    STSP_U_DEP,
    STSP_U_FAC,
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


@torch.jit.script
def scan_stripped(
    inputs: torch.Tensor,
    weight_in: torch.Tensor,
    weight_rec: torch.Tensor,
    bias_rec: torch.Tensor,
    weight_out: torch.Tensor,
    bias_out: torch.Tensor,
    recurrent_mask: torch.Tensor,
    mem_decay: float,
    readout_decay: float,
    threshold: float,
    reset: float,
    alpha: float,
    refractory_steps: int,
    use_stsp: bool,
    baseline_u: torch.Tensor,
    decay_x: torch.Tensor,
    decay_u: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    n_steps, batch_size, _ = inputs.shape
    n_hidden = weight_rec.shape[0]
    n_output = weight_out.shape[0]
    voltage = torch.zeros(batch_size, n_hidden, device=inputs.device, dtype=inputs.dtype)
    spikes = torch.zeros(batch_size, n_hidden, device=inputs.device, dtype=inputs.dtype)
    refractory = torch.zeros(batch_size, n_hidden, device=inputs.device, dtype=torch.int64)
    readout = torch.zeros(batch_size, n_output, device=inputs.device, dtype=inputs.dtype)
    syn_x = torch.ones(batch_size, n_hidden, device=inputs.device, dtype=inputs.dtype)
    syn_u = baseline_u.expand(batch_size, n_hidden).clone()
    spike_power_sum = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
    logit_steps = []
    rec = weight_rec * recurrent_mask
    ones = torch.ones_like(syn_x)
    for _step_index in range(n_steps):
        if use_stsp:
            syn_u = baseline_u + (syn_u - baseline_u) * decay_u
            syn_x = ones + (syn_x - ones) * decay_x
            u_plus = syn_u + spikes * baseline_u * (1.0 - syn_u)
            rec_in = u_plus * syn_x * spikes.detach()
            syn_x = torch.clamp(syn_x - spikes * u_plus * syn_x, 0.0, 1.0)
            syn_u = torch.clamp(u_plus, 0.0, 1.0)
        else:
            rec_in = spikes.detach()
        drive = torch.nn.functional.linear(inputs[_step_index], weight_in, None) + torch.nn.functional.linear(
            rec_in, rec, bias_rec
        )
        active = (refractory == 0).to(inputs.dtype)
        voltage = torch.where(
            refractory == 0,
            mem_decay * voltage + (1.0 - mem_decay) * drive,
            torch.full_like(voltage, reset),
        )
        delta = voltage - threshold
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
        spike_power_sum = spike_power_sum + spikes.pow(2).mean()
        readout = readout_decay * readout + (1.0 - readout_decay) * torch.nn.functional.linear(
            spikes, weight_out, bias_out
        )
        logit_steps.append(readout)
    spike_power = spike_power_sum / float(n_steps)
    return torch.stack(logit_steps, dim=0), voltage, spikes, refractory, readout, spike_power, syn_x, syn_u


class RecurrentState(NamedTuple):
    voltage: torch.Tensor
    current: Optional[torch.Tensor]
    spikes: torch.Tensor
    adaptation: Optional[torch.Tensor]
    refractory: torch.Tensor
    readout: torch.Tensor
    syn_x: Optional[torch.Tensor] = None
    syn_u: Optional[torch.Tensor] = None
    spike_power: Optional[torch.Tensor] = None


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
        self.use_synaptic_current = bool(config.use_synaptic_current)
        self.use_stsp = bool(config.use_stsp)
        self.dt_ms = float(config.dt_ms)

        self.input_linear = nn.Linear(N_INPUT, config.n_hidden, bias=False)
        self.recurrent_linear = nn.Linear(config.n_hidden, config.n_hidden, bias=True)
        self.readout_linear = nn.Linear(config.n_hidden, N_OUTPUT, bias=True)
        mask = torch.ones(config.n_hidden, config.n_hidden)
        mask.fill_diagonal_(0.0)
        self.register_buffer("recurrent_mask", mask)
        sfa_mask = torch.zeros(config.n_hidden)
        sfa_mask[: config.n_sfa] = 1.0
        self.register_buffer("sfa_mask", sfa_mask)
        even = torch.arange(config.n_hidden) % 2 == 0
        baseline_u = torch.where(
            even, torch.full((config.n_hidden,), STSP_U_FAC), torch.full((config.n_hidden,), STSP_U_DEP)
        ).to(dtype=torch.float32)
        tau_x = torch.where(
            even,
            torch.full((config.n_hidden,), STSP_TAU_X_FAC_MS),
            torch.full((config.n_hidden,), STSP_TAU_X_DEP_MS),
        ).to(dtype=torch.float32)
        tau_u = torch.where(
            even,
            torch.full((config.n_hidden,), STSP_TAU_U_FAC_MS),
            torch.full((config.n_hidden,), STSP_TAU_U_DEP_MS),
        ).to(dtype=torch.float32)
        if not self.use_synaptic_current:
            self.register_buffer("stsp_baseline_u", baseline_u)
            self.register_buffer("stsp_decay_x", torch.exp(-config.dt_ms / tau_x))
            self.register_buffer("stsp_decay_u", torch.exp(-config.dt_ms / tau_u))
        else:
            self.stsp_baseline_u = baseline_u
            self.stsp_decay_x = torch.exp(-config.dt_ms / tau_x)
            self.stsp_decay_u = torch.exp(-config.dt_ms / tau_u)
        self.reset_parameters(config.model_init_seed, config.recurrent_weight_scale)

    def reset_parameters(self, seed: int, recurrent_weight_scale: float = 1.0) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        self._xavier_uniform_(self.input_linear.weight, generator)
        hidden = torch.randn(self.n_hidden, self.n_hidden, generator=generator)
        hidden.fill_diagonal_(0.0)
        spectral_radius = torch.linalg.eigvals(hidden).abs().max().clamp_min(1e-6)
        scaled = hidden * (0.95 / spectral_radius) * float(recurrent_weight_scale)
        self.recurrent_linear.weight.data.copy_(scaled)
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
        if self.use_synaptic_current:
            return RecurrentState(
                voltage=zeros.clone(),
                current=zeros.clone(),
                spikes=zeros.clone(),
                adaptation=zeros.clone(),
                refractory=torch.zeros(batch_size, self.n_hidden, device=device, dtype=torch.int64),
                readout=torch.zeros(batch_size, N_OUTPUT, device=device, dtype=dtype),
                spike_power=torch.zeros((), device=device, dtype=dtype),
            )
        syn_x = None
        syn_u = None
        if self.use_stsp:
            syn_x = torch.ones(batch_size, self.n_hidden, device=device, dtype=dtype)
            syn_u = self.stsp_baseline_u.to(device=device, dtype=dtype).expand(batch_size, -1).clone()
        return RecurrentState(
            voltage=zeros.clone(),
            current=None,
            spikes=zeros.clone(),
            adaptation=None,
            refractory=torch.zeros(batch_size, self.n_hidden, device=device, dtype=torch.int64),
            readout=torch.zeros(batch_size, N_OUTPUT, device=device, dtype=dtype),
            syn_x=syn_x,
            syn_u=syn_u,
            spike_power=torch.zeros((), device=device, dtype=dtype),
        )

    def recurrent_weight(self) -> torch.Tensor:
        return self.recurrent_linear.weight * self.recurrent_mask

    def _stsp_transmit(
        self, spikes: torch.Tensor, syn_x: torch.Tensor, syn_u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        baseline = self.stsp_baseline_u.to(device=spikes.device, dtype=spikes.dtype)
        syn_u = baseline + (syn_u - baseline) * self.stsp_decay_u.to(dtype=spikes.dtype)
        syn_x = 1.0 + (syn_x - 1.0) * self.stsp_decay_x.to(dtype=spikes.dtype)
        u_plus = syn_u + spikes * baseline * (1.0 - syn_u)
        rec_in = u_plus * syn_x * spikes.detach()
        syn_x = torch.clamp(syn_x - spikes * u_plus * syn_x, 0.0, 1.0)
        syn_u = torch.clamp(u_plus, 0.0, 1.0)
        return rec_in, syn_x, syn_u

    def step(self, inputs: torch.Tensor, state: RecurrentState) -> tuple[torch.Tensor, RecurrentState]:
        if self.use_synaptic_current:
            if state.current is None or state.adaptation is None:
                raise ValueError("legacy dynamics require current and adaptation state")
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
            return readout, RecurrentState(
                voltage=voltage,
                current=current,
                spikes=spikes,
                adaptation=adaptation,
                refractory=refractory,
                readout=readout,
                spike_power=spikes.pow(2).mean(),
            )

        rec_in = state.spikes.detach()
        syn_x = state.syn_x
        syn_u = state.syn_u
        if self.use_stsp:
            if syn_x is None or syn_u is None:
                raise ValueError("STSP dynamics require syn_x and syn_u")
            rec_in, syn_x, syn_u = self._stsp_transmit(state.spikes, syn_x, syn_u)
        drive = self.input_linear(inputs) + torch.nn.functional.linear(
            rec_in, self.recurrent_weight(), self.recurrent_linear.bias
        )
        active = state.refractory == 0
        voltage = torch.where(
            active,
            self.mem_decay * state.voltage + (1.0 - self.mem_decay) * drive,
            torch.full_like(state.voltage, self.reset),
        )
        spikes = superspike(
            voltage,
            torch.full_like(voltage, self.threshold),
            self.surrogate_alpha,
            active.to(voltage.dtype),
        )
        spiked = spikes > 0
        voltage = torch.where(spiked, torch.full_like(voltage, self.reset), voltage)
        refractory = torch.where(
            spiked,
            torch.full_like(state.refractory, self.refractory_steps),
            torch.clamp(state.refractory - 1, min=0),
        )
        readout = self.readout_decay * state.readout + (1.0 - self.readout_decay) * self.readout_linear(spikes)
        return readout, RecurrentState(
            voltage=voltage,
            current=None,
            spikes=spikes,
            adaptation=None,
            refractory=refractory,
            readout=readout,
            syn_x=syn_x,
            syn_u=syn_u,
            spike_power=spikes.pow(2).mean(),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: RecurrentState | None = None,
        *,
        record_traces: bool = False,
        shuffle_stsp_at: int | None = None,
        shuffle_generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, RecurrentState] | tuple[torch.Tensor, RecurrentState, torch.Tensor, torch.Tensor | None]:
        if inputs.dim() != 3:
            raise ValueError("inputs must have shape [time, batch, channels]")
        _n_steps, batch_size, n_channels = inputs.shape
        if n_channels != N_INPUT:
            raise ValueError(f"expected {N_INPUT} input channels, got {n_channels}")
        if state is None and not record_traces and shuffle_stsp_at is None and self.use_synaptic_current:
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
        if (
            state is None
            and not record_traces
            and shuffle_stsp_at is None
            and not self.use_synaptic_current
        ):
            logits, voltage, spikes, refractory, readout, spike_power, syn_x, syn_u = scan_stripped(
                inputs,
                self.input_linear.weight,
                self.recurrent_linear.weight,
                self.recurrent_linear.bias,
                self.readout_linear.weight,
                self.readout_linear.bias,
                self.recurrent_mask,
                self.mem_decay,
                self.readout_decay,
                float(self.threshold),
                float(self.reset),
                float(self.surrogate_alpha),
                int(self.refractory_steps),
                bool(self.use_stsp),
                self.stsp_baseline_u,
                self.stsp_decay_x,
                self.stsp_decay_u,
            )
            final_state = RecurrentState(
                voltage=voltage,
                current=None,
                spikes=spikes,
                adaptation=None,
                refractory=refractory,
                readout=readout,
                syn_x=syn_x if self.use_stsp else None,
                syn_u=syn_u if self.use_stsp else None,
                spike_power=spike_power,
            )
            return logits, final_state
        if state is None:
            state = self.initial_state(batch_size, inputs.device, inputs.dtype)
        logits = []
        spike_steps = []
        efficacy_steps = []
        spike_power_sum = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
        for step_index in range(inputs.shape[0]):
            if shuffle_stsp_at is not None and step_index == int(shuffle_stsp_at) and state.syn_x is not None:
                perm = torch.randperm(self.n_hidden, generator=shuffle_generator).to(state.syn_x.device)
                state = state._replace(syn_x=state.syn_x[:, perm], syn_u=state.syn_u[:, perm] if state.syn_u is not None else None)
            logit, state = self.step(inputs[step_index], state)
            logits.append(logit)
            spike_power_sum = spike_power_sum + state.spikes.pow(2).mean()
            if record_traces:
                spike_steps.append(state.spikes)
                if state.syn_x is not None and state.syn_u is not None:
                    efficacy_steps.append(state.syn_x * state.syn_u)
        state = state._replace(spike_power=spike_power_sum / float(inputs.shape[0]))
        stacked = torch.stack(logits, dim=0)
        if record_traces:
            spikes = torch.stack(spike_steps, dim=0)
            efficacy = torch.stack(efficacy_steps, dim=0) if efficacy_steps else None
            return stacked, state, spikes, efficacy
        return stacked, state
