import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from src.config.units import *


# ==========================================
# 1. JIT 缂栬瘧鐨勫姩鍔涘鏍稿績 (淇濇寔涓嶅彉)
# ==========================================
@torch.jit.script
def lif_dynamics_jit(V_m: torch.Tensor,
                     g_e: torch.Tensor,
                     res: torch.Tensor,
                     total_exc: torch.Tensor,
                     V_E: float, V_L: float,
                     g_m: float, C_m: float, dt: float,
                     alpha_e: float):
    g_e = g_e * alpha_e + total_exc
    I_leak = -g_m * (V_m - V_L)
    I_exc = -g_e * (V_m - V_E)
    dV = (I_leak + I_exc) / C_m * dt

    non_ref_mask = (res == 0)
    V_m_new = torch.where(non_ref_mask, V_m + dV, V_m)
    res = (res - 1).clamp(min=0)
    return V_m_new, g_e, res


@torch.jit.script
def stsp_dynamics_jit(u: torch.Tensor,
                      x: torch.Tensor,
                      input_spikes: torch.Tensor,
                      U: float,
                      decay_x: float,
                      decay_u: float):
    """
    STSP 鍔ㄥ姏瀛︽牳蹇?[Mongillo et al. 2008]
    """
    # 1. 鏃堕棿鑷劧琛板噺/鎭㈠
    x = 1.0 + (x - 1.0) * decay_x
    u = U + (u - U) * decay_u

    # 2. 璁＄畻褰撳墠澧炵泭 (Pulse鍒版潵鏃?
    gain = u * x

    # 3. 鑴夊啿瑙﹀彂鐨勭姸鎬佺獊鍙?
    mask = input_spikes > 0

    # 璧勬簮娑堣€?(Depression)
    x = torch.where(mask, x - gain, x)
    # 閽欑瀛愮Н绱?(Facilitation)
    u_increment = U * (1.0 - u)
    u = torch.where(mask, u + u_increment, u)

    # 鎴柇淇濇姢
    u = torch.clamp(u, 0.0, 1.0)
    x = torch.clamp(x, 0.0, 1.0)

    return u, x, gain


# ==========================================
# 2. 渚у悜鎶戝埗鍥炶矾 (淇濇寔涓嶅彉)
# ==========================================
class LateralInhibition(nn.Module):
    def __init__(self, channels, kernel_size=7, sigma_cross=1.0, strength_cross=100.0, decay_tau=10.0 * ms, dt=1.0 * ms,
                 device='cuda'):
        super(LateralInhibition, self).__init__()
        self.device = device
        self.padding = kernel_size // 2
        self.alpha_inh = math.exp(-float(dt) / float(decay_tau))

        k_cross_base = self._create_gaussian_kernel(kernel_size, sigma_cross)
        k_cross_base = k_cross_base / (k_cross_base.sum() + 1e-12) * strength_cross
        self.register_buffer('kernel_global', k_cross_base)
        self.inh_trace = None

    def _create_gaussian_kernel(self, size, sigma):
        x = torch.linspace(-(size // 2), size // 2, size)
        y = torch.linspace(-(size // 2), size // 2, size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        return kernel.view(1, 1, size, size)

    def reset_state(self, input_shape):
        self.inh_trace = torch.zeros(input_shape, device=self.device)

    def decay_only(self):
        self.inh_trace *= self.alpha_inh

    def forward(self, spikes):
        if not spikes.any():
            self.inh_trace *= self.alpha_inh
            return self.inh_trace

        spikes_f = spikes.float()
        global_activity = spikes_f.sum(dim=1, keepdim=True)
        global_inh = F.conv2d(global_activity, self.kernel_global, padding=self.padding)
        self.inh_trace = self.inh_trace * self.alpha_inh + global_inh
        return self.inh_trace


# ==========================================
# 3. 鐗╃悊灞傚熀绫?(宸蹭慨鏀规敮鎸?STSP Monitor)
# ==========================================
class BaseLIFLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, stride, padding,
                 # --- 鐗╃悊鍙傛暟 ---
                 V_reset=-60.0 * mV,
                 V_L=-70.0 * mV,
                 V_E=0.0 * mV,
                 C_m=0.1 * nF,
                 g_m=10.0 * nS,
                 tau_e=5.0 * ms,
                 tau_rp=20.0 * ms,
                 dt=1.0 * ms,
                 # --- 鎶戝埗涓庡櫔澹?---
                 inh_strength=100.0 * mV,
                 inh_tau=15.0 * ms,
                 sigma_cross=1.0,
                 noise_init_std=0.0 * mV,
                 noise_decay=0.9,
                 init_w=0.6 * nS,
                 # --- 绔炰簤鍙傛暟 ---
                 use_channel_wta=False,
                 k_winners=1,
                 # --- STSP 鍙傛暟 ---
                 enable_stsp=False,
                 stsp_U=0.2,
                 stsp_tau_D=100.0 * ms,
                 stsp_tau_F=1000.0 * ms,
                 device='cuda'):
        super(BaseLIFLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device

        self.V_reset = float(V_reset)
        self.V_L = float(V_L)
        self.V_E = float(V_E)
        self.g_m = float(g_m)
        self.C_m = float(C_m)
        self.dt = float(dt)
        self.tau_rp_step = int(tau_rp / dt)
        self.alpha_e = math.exp(-dt / tau_e)

        # STSP 棰勮绠?
        self.enable_stsp = enable_stsp
        self.stsp_U = stsp_U
        self.stsp_decay_x = math.exp(-dt / stsp_tau_D) if stsp_tau_D > 0 else 0.0
        self.stsp_decay_u = math.exp(-dt / stsp_tau_F) if stsp_tau_F > 0 else 0.0

        self.noise_decay_rate = noise_decay
        self.register_buffer('current_noise_std', torch.tensor(float(noise_init_std)))

        self.use_channel_wta = use_channel_wta
        self.k_winners = k_winners

        self.kernels = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device) * 0.01 * nS + init_w,
            requires_grad=False)

        # ... (LateralInhibition 鍒濆鍖栦繚鎸佷笉鍙? ...
        self.lateral_inh = LateralInhibition(
            channels=out_channels,
            kernel_size=7,
            sigma_cross=sigma_cross,
            strength_cross=inh_strength,
            decay_tau=inh_tau,
            dt=dt,
            device=device
        )

        self.v_mem = None
        self.g_e = None
        self.res = None
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.empty_mask = None

        # STSP 鐘舵€佸彉閲?
        self.u_pre = None
        self.x_pre = None

    def reset_state(self, input_shape):
        B, C_in, H_in, W_in = input_shape
        h_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.output_shape = (B, self.out_channels, h_out, w_out)

        self.v_mem = torch.full(self.output_shape, self.V_L, device=self.device)
        self.g_e = torch.zeros(self.output_shape, device=self.device)
        self.res = torch.zeros(self.output_shape, dtype=torch.long, device=self.device)
        self.empty_mask = torch.zeros(self.output_shape, dtype=torch.bool, device=self.device)
        self.lateral_inh.reset_state(self.output_shape)

        if self.enable_stsp:
            self.u_pre = torch.full(input_shape, self.stsp_U, device=self.device)
            self.x_pre = torch.ones(input_shape, device=self.device)
        else:
            self.u_pre = None
            self.x_pre = None

    def forward_physics(self, input_spikes, effective_thresh, monitor=False, check_firing=True, stsp_mode='dynamic', ping_drive=None):
        """
        stsp_mode:
          - 'dynamic': 姝ｅ父 STSP锛屾洿鏂?u, x
          - 'static_frozen': 鍥哄畾澧炵泭 = U (鍗冲亣璁?u=U, x=1 涓斾笉闅忔椂闂村彉鍖?
        """
        # 1. STSP 杈撳叆璋冨埗
        gain = None
        input_spikes_f = input_spikes.float()
        if ping_drive is None:
            ping_drive_f = torch.zeros_like(input_spikes_f)
        else:
            ping_drive_f = ping_drive.float()
        total_presyn = input_spikes_f + ping_drive_f

        if self.enable_stsp:
            if stsp_mode == 'dynamic':
                # 姝ｅ父鍔ㄦ€佹洿鏂?
                self.u_pre, self.x_pre, gain = stsp_dynamics_jit(
                    self.u_pre, self.x_pre, input_spikes,
                    self.stsp_U, self.stsp_decay_x, self.stsp_decay_u
                )
                effective_input = input_spikes_f * gain
                effective_ping = ping_drive_f * gain
                total_presyn = effective_input + effective_ping
            elif stsp_mode == 'static_frozen':
                # 鍐荤粨妯″紡锛氫娇鐢ㄥ垵濮嬪鐩?U * 1.0
                # 涓嶆洿鏂?u_pre, x_pre
                gain = torch.tensor(self.stsp_U, device=self.device)
                effective_input = input_spikes_f * gain
                effective_ping = ping_drive_f * gain
                total_presyn = effective_input + effective_ping

        # 2. 鍗风Н
        syn_input = F.conv2d(total_presyn, self.kernels, stride=self.stride, padding=self.padding)

        # 3. LIF 绉垎
        self.v_mem, self.g_e, self.res = lif_dynamics_jit(
            self.v_mem, self.g_e, self.res, syn_input,
            self.V_E, self.V_L, self.g_m, self.C_m, self.dt, self.alpha_e
        )

        # 4. 蹇€熼€氶亾 (淇 KeyError 鐨勫叧閿偣)
        if not check_firing:
            self.lateral_inh.decay_only()
            monitor_data = {}
            if monitor:
                # 鍗充娇涓嶅彂鏀撅紝涔熻璁板綍褰撳墠鐨勮啘鐢典綅浠ヤ緵瀵规瘮
                monitor_data['v_raw'] = self.v_mem.detach().clone()
                monitor_data['v_mem_snapshot'] = self.v_mem.detach().clone()
                if self.enable_stsp and stsp_mode == 'dynamic':
                    monitor_data['stsp_u'] = self.u_pre.detach().clone()
                    monitor_data['stsp_x'] = self.x_pre.detach().clone()
                    if gain is not None:
                        monitor_data['stsp_gain'] = gain.detach().clone()
            return self.empty_mask, monitor_data

        # 5. 鍣０涓庡彂鏀?
        inh_val_before = self.lateral_inh.inh_trace.clone()
        noise = torch.randn_like(self.v_mem) * self.current_noise_std
        v_raw_pre_reset = self.v_mem + noise
        v_effective = v_raw_pre_reset - inh_val_before

        candidate_mask = (v_effective >= effective_thresh) & (self.res == 0)

        # 6. WTA
        fire_mask = candidate_mask
        if self.use_channel_wta and candidate_mask.any():
            _, topk_indices = torch.topk(v_effective, k=self.k_winners, dim=1)
            winner_mask = torch.zeros_like(candidate_mask)
            winner_mask.scatter_(1, topk_indices, True)
            fire_mask = candidate_mask & winner_mask
            losers = candidate_mask & (~winner_mask)
            if losers.any():
                self.v_mem.masked_fill_(losers, self.V_L)

        # 7. 鏇存柊
        if fire_mask.any():
            self.lateral_inh(fire_mask)
            self.v_mem.masked_fill_(fire_mask, self.V_reset)
            self.res.masked_fill_(fire_mask, self.tau_rp_step)
        else:
            self.lateral_inh.decay_only()

        monitor_data = {}
        if monitor:
            inh_val_after = self.lateral_inh.inh_trace
            monitor_data = {
                'v_raw': v_raw_pre_reset.detach(),
                'v_effective': v_effective.detach(),
                'v_mem_snapshot': self.v_mem.detach().clone(),
                'g_e': self.g_e.detach().clone(),
                'inh_before': inh_val_before.detach(),
                'inh_after': inh_val_after.detach(),
                'fire_mask': fire_mask.detach()
            }
            if self.enable_stsp and stsp_mode == 'dynamic':
                monitor_data['stsp_u'] = self.u_pre.detach().clone()
                monitor_data['stsp_x'] = self.x_pre.detach().clone()
                if gain is not None:
                    monitor_data['stsp_gain'] = gain.detach().clone()

        return fire_mask, monitor_data

    # ... (batch_end_homeostasis_base 淇濇寔涓嶅彉) ...
    def batch_end_homeostasis_base(self):
        self.current_noise_std *= self.noise_decay_rate


# ==========================================
# 4. 瀛愮被: STDPLayer (淇濇寔涓嶅彉)
# ==========================================
class STDPLayer(BaseLIFLayer):
    # ... (__init__ 淇濇寔涓嶅彉) ...
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2,
                 learning_rate=0.005, tau_plus=20.0 * ms, w_min=0.0 * nS, w_max=1.0 * nS, x_tar=0.5,
                 theta_init=0.0 * mV, theta_decay=0.001, min_thresh=-60.0 * mV, coupling_factor=1.0,
                 **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, **kwargs)

        self.lr = learning_rate
        self.w_min = w_min
        self.w_max = w_max
        self.x_tar = x_tar
        self.alpha_pos = math.exp(-self.dt / tau_plus)

        self.V_thr_base = float(min_thresh)
        self.theta_decay = theta_decay
        self.coupling_factor = coupling_factor

        with torch.no_grad():
            w_flat = self.kernels.view(out_channels, -1)
            self.target_norm = w_flat.sum(dim=1).mean().item()

        self.pre_trace = None
        self.register_buffer('theta', torch.ones(out_channels, device=self.device) * theta_init)

    # ... (reset_state 淇濇寔涓嶅彉) ...
    def reset_state(self, input_shape):
        super().reset_state(input_shape)
        self.pre_trace = torch.zeros(input_shape, device=self.device)

    def forward_step(self, input_spikes, t, training=False, monitor=False, stsp_mode='dynamic', ping_drive=None):
        eff_thresh = self.V_thr_base + self.theta.view(1, -1, 1, 1)
        # Pass stsp_mode to BaseLIFLayer
        fire_mask, monitor_data = self.forward_physics(
            input_spikes, eff_thresh, monitor=monitor, check_firing=True, stsp_mode=stsp_mode, ping_drive=ping_drive
        )

        in_spikes_f = input_spikes.float()

        if training and fire_mask.any():
            self._apply_stdp(in_spikes_f, fire_mask)

        self.pre_trace = self.pre_trace * self.alpha_pos + in_spikes_f
        return fire_mask, monitor_data

    # ... (_apply_stdp 鍜?batch_end_homeostasis 淇濇寔涓嶅彉) ...
    def _apply_stdp(self, in_spikes, post_spikes):
        pre_trace_unfolded = self.unfold(self.pre_trace)
        post_flat = post_spikes.float().view(post_spikes.shape[0], self.out_channels, -1)
        winner_count = post_flat.sum(dim=(0, 2)).view(-1, 1) + 1e-15

        post_perm = post_flat.permute(1, 0, 2).reshape(self.out_channels, -1)
        pre_perm = pre_trace_unfolded.permute(0, 2, 1).reshape(-1, self.kernels[0].numel())
        accumulated_trace = torch.mm(post_perm, pre_perm)
        mean_trace = accumulated_trace / winner_count

        w_flat = self.kernels.view(self.out_channels, -1)
        trace_diff = mean_trace - self.x_tar
        weight_dep = torch.where(trace_diff > 0, self.w_max - w_flat, w_flat - self.w_min)
        update = self.lr * trace_diff * weight_dep

        with torch.no_grad():
            active_channels = (winner_count > 0.1).view(-1, 1, 1, 1)
            self.kernels += update.view_as(self.kernels) * active_channels
            self.kernels.clamp_(min=self.w_min, max=self.w_max)

    def batch_end_homeostasis(self):
        super().batch_end_homeostasis_base()
        with torch.no_grad():
            self.theta *= (1.0 - self.theta_decay)
            k_flat = self.kernels.view(self.out_channels, -1)
            curr_norm = k_flat.sum(dim=1, keepdim=True)
            delta = curr_norm - self.target_norm
            delta_pos = torch.relu(delta)
            if delta_pos.sum() > 1e-12:
                boost = self.coupling_factor * (delta_pos / self.g_m) * (self.V_E - self.V_L)
                self.theta += boost.flatten()
            self.theta.clamp_(min=0.0)
            scale = self.target_norm / (curr_norm + 1e-12)
            self.kernels.data = (k_flat * scale).view_as(self.kernels)


# ==========================================
# 5. 瀛愮被: RSTDPLayer (淇敼: 浼犻€?stsp_mode)
# ==========================================
class RSTDPLayer(BaseLIFLayer):
    # ... (__init__, update_adaptive_rates, reset_state 淇濇寔涓嶅彉) ...
    def __init__(self, in_channels, num_classes, neurons_per_class, input_spatial_size=7,
                 learning_rate=0.005,
                 cycle_length=60,
                 decision_time_offset=20,
                 reward_factor=1.0,
                 punishment_factor=1.0,
                 tau_elig=10.0 * ms,
                 min_thresh=-60.0 * mV,
                 theta_init=0.0 * mV,
                 theta_decay=0.001,
                 coupling_factor=1.0,
                 **kwargs):
        self.num_classes = num_classes
        self.neurons_per_class = neurons_per_class
        out_channels = num_classes * neurons_per_class

        kernel_size = input_spatial_size
        stride = 1
        padding = 0
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         **kwargs)

        self.lr = learning_rate
        self.wmax = 1.0 * nS
        self.wmin = 0.0 * nS
        self.max_reward_val = reward_factor
        self.max_punish_val = punishment_factor
        self.register_buffer('reward_base', torch.tensor(float(reward_factor)))
        self.register_buffer('punishment_base', torch.tensor(float(punishment_factor)))
        self.register_buffer('running_avg_acc', torch.tensor(0.1))

        self.cycle_length = int(cycle_length)
        self.decision_time_offset = int(decision_time_offset)
        self.execution_threshold = self.V_L

        self.tau_elig = tau_elig
        self.alpha_pos = math.exp(-self.dt / (20.0 * ms))
        self.theta_decay = theta_decay
        self.coupling_factor = coupling_factor
        self.V_thr_base = float(min_thresh)
        self.theta = torch.ones(out_channels, device=self.device) * theta_init

        with torch.no_grad():
            w_flat = self.kernels.view(self.out_channels, -1)
            self.target_norm = w_flat.sum(dim=1).mean().item()

        self.eligibility_trace = None
        self.input_trace = None
        self.firing_times = None

    def update_adaptive_rates(self, batch_accuracy=None, alpha_smooth=0.05, epsilon=0.001):
        if batch_accuracy is not None:
            old_acc = self.running_avg_acc.item()
            new_acc = (1 - alpha_smooth) * old_acc + alpha_smooth * batch_accuracy
            self.running_avg_acc.fill_(new_acc)

        acc = max(0.0, min(1.0, self.running_avg_acc.item()))
        new_reward = self.max_reward_val * (1.0 - acc) + epsilon
        new_punish = self.max_punish_val * acc + epsilon

        self.reward_base.fill_(new_reward)
        self.punishment_base.fill_(new_punish)
        return new_reward, new_punish

    def reset_state(self, input_shape):
        super().reset_state(input_shape)
        B = input_shape[0]
        k_shape = self.kernels.shape
        trace_shape = (B,) + k_shape
        self.eligibility_trace = torch.zeros(trace_shape, device=self.device)
        self.input_trace = torch.zeros(input_shape, device=self.device)
        self.firing_times = torch.full((B, self.out_channels), float('inf'), device=self.device)

    def reset_decision_state(self):
        """
        浠呴噸缃喅绛栫姸鎬侊紙棣栨鍙戞斁鏃堕棿锛夛紝淇濈暀鑶滅數浣嶃€佹潈閲嶅拰STSP鐘舵€併€?
        鍔熻兘锛氬湪 DMS 浠诲姟鐨?Test 闃舵寮€濮嬪墠璋冪敤锛屽氨鍍忊€滄瘮璧涜鏃跺櫒褰掗浂鈥濓紝
        纭繚鎴戜滑璁板綍鐨勬槸 Test 闃舵浜х敓鐨勬柊鍐崇瓥锛岃€屼笉鏄?Sample 闃舵鐨勬棫鍐崇瓥銆?
        """
        if self.firing_times is not None:
            self.firing_times.fill_(float('inf'))

    def get_grouped_voltage(self, voltage_tensor=None):
        """Return layer-3 membrane voltages grouped as [B, num_classes, group_size]."""
        if voltage_tensor is None:
            voltage_tensor = self.v_mem
        if voltage_tensor.ndim != 4:
            raise ValueError(f"Expected voltage tensor with shape [B, C, H, W], got {tuple(voltage_tensor.shape)}")
        batch_size, channels, height, width = voltage_tensor.shape
        expected_channels = self.num_classes * self.neurons_per_class
        if channels != expected_channels:
            raise ValueError(
                f"Channel mismatch for grouped voltage: got {channels}, expected {expected_channels}"
            )
        grouped = voltage_tensor.view(batch_size, self.num_classes, self.neurons_per_class, height * width)
        return grouped.reshape(batch_size, self.num_classes, self.neurons_per_class * height * width)

    def forward_step(self, input_spikes, t, labels=None, training=True, monitor=False, stsp_mode='dynamic', ping_drive=None):
        B = input_spikes.shape[0]
        current_step_in_cycle = t % self.cycle_length
        is_decision_time = (current_step_in_cycle == self.decision_time_offset)

        if is_decision_time:
            eff_thresh = self.V_L
            # eff_thresh = self.execution_threshold + self.theta.view(1, -1, 1, 1)
        else:
            eff_thresh = torch.tensor(10000.0, device=self.device)

        # Pass stsp_mode
        raw_fire_mask, monitor_data = self.forward_physics(
            input_spikes, eff_thresh, monitor=monitor, check_firing=is_decision_time, stsp_mode=stsp_mode, ping_drive=ping_drive
        )

        if is_decision_time:
            fire_mask_flat = raw_fire_mask.view(B, -1)
            just_fired = fire_mask_flat & (self.firing_times == float('inf'))
            self.firing_times[just_fired] = float(t)

        self.input_trace = self.input_trace * self.alpha_pos + input_spikes.float()

        if training and is_decision_time and raw_fire_mask.any():
            self._accumulate_eligibility(raw_fire_mask, self.input_trace)
            trigger_indices = torch.where(raw_fire_mask.view(B, -1).sum(dim=1) > 0)[0]
            if len(trigger_indices) > 0:
                self._perform_online_update(trigger_indices, labels, raw_fire_mask)
            self.eligibility_trace.zero_()

        return raw_fire_mask, monitor_data

    # ... (_accumulate_eligibility, _perform_online_update, batch_end_homeostasis 淇濇寔涓嶅彉) ...
    def _accumulate_eligibility(self, fire_mask, input_trace):
        post = fire_mask.float().view(fire_mask.shape[0], self.out_channels, -1).sum(dim=2)
        post = post.unsqueeze(2)
        pre = input_trace.view(input_trace.shape[0], -1).unsqueeze(1)
        delta = torch.bmm(post, pre)
        k_shape = self.kernels.shape
        delta = delta.view((delta.shape[0],) + k_shape)
        self.eligibility_trace += delta

    def _perform_online_update(self, trigger_indices, labels, current_fire_mask):
        subset_traces = self.eligibility_trace[trigger_indices]
        subset_labels = labels[trigger_indices]
        K_trig = len(trigger_indices)

        rewards = torch.zeros(K_trig, self.out_channels, device=self.device)
        for i in range(K_trig):
            lbl = subset_labels[i].item()
            start_ch = lbl * self.neurons_per_class
            end_ch = start_ch + self.neurons_per_class
            rewards[i, :] = -self.punishment_base
            rewards[i, start_ch:end_ch] = self.reward_base

        rewards_view = rewards.view(K_trig, self.out_channels, 1, 1, 1)
        weighted_traces = rewards_view * subset_traces
        total_delta_w = weighted_traces.sum(dim=0)

        w_dist_to_bound = torch.where(total_delta_w > 0,
                                      self.wmax - self.kernels,
                                      self.kernels - self.wmin)

        self.kernels += self.lr * total_delta_w * w_dist_to_bound
        self.kernels.clamp_(min=self.wmin, max=self.wmax)

    def batch_end_homeostasis(self):
        super().batch_end_homeostasis_base()
        with torch.no_grad():
            self.theta *= (1.0 - self.theta_decay)
            k_flat = self.kernels.view(self.out_channels, -1)
            curr_norm = k_flat.sum(dim=1, keepdim=True)
            delta = curr_norm - self.target_norm
            delta_pos = torch.relu(delta)
            if delta_pos.sum() > 1e-12:
                boost = self.coupling_factor * (delta_pos / self.g_m) * (self.V_E - self.V_L)
                self.theta += boost.flatten()
            self.theta.clamp_(min=0.0)
            scale = self.target_norm / (curr_norm + 1e-12)
            self.kernels.data = (k_flat * scale).view_as(self.kernels)


# ==========================================
# 6. 缃戠粶灏佽 (SDNN_Network) [鏇存柊]
# ==========================================
class SDNN_Network(nn.Module):
    # ... (__init__ 鍜?forward 淇濇寔涓嶅彉) ...
    def __init__(self, device='cuda'):
        super(SDNN_Network, self).__init__()
        self.device = device

        # Layer 1
        self.layer1 = STDPLayer(
            in_channels=2,
            out_channels=30,
            kernel_size=5, stride=1, padding=2,
            tau_e=5.0 * ms,
            learning_rate=0.001,
            init_w=0.6 * nS,
            coupling_factor=0.5,
            theta_init=0 * mV,
            inh_strength=20.0 * mV,
            inh_tau=10.0 * ms,
            sigma_cross=2,
            use_channel_wta=True,
            k_winners=5,
            enable_stsp=True,
            stsp_U=0.2,
            device=device
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2
        self.layer2 = STDPLayer(
            in_channels=30,
            out_channels=150,
            kernel_size=3, stride=1, padding=2,
            tau_e=5.0 * ms,
            tau_plus=20.0 * ms,
            learning_rate=0.001,
            init_w=0.6 * nS,
            sigma_cross=1,
            inh_strength=20.0 * mV,
            theta_init=0 * mV,
            min_thresh=-50 * mV,
            coupling_factor=1,
            inh_tau=10.0 * ms,
            use_channel_wta=True,
            k_winners=10,
            enable_stsp=True,
            stsp_U=0.2,
            device=device
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3
        self.layer3 = RSTDPLayer(
            in_channels=150, num_classes=10, neurons_per_class=20,
            input_spatial_size=8,
            tau_e=5.0 * ms,
            tau_elig=20 * ms,
            theta_init=0 * mV,
            min_thresh=0 * mV,
            learning_rate=0.01, reward_factor=1.0,
            init_w=0.8 * nS,
            noise_init_std=0.0 * mV,
            noise_decay=0.995,
            sigma_cross=0,
            inh_strength=10.0 * mV,
            inh_tau=10.0 * ms,
            use_channel_wta=True,
            coupling_factor=0,
            theta_decay=0,
            k_winners=1,
            enable_stsp=True,
            stsp_U=0.2,
            device=device
        )

    def forward(self, spike_train_batch, layer_idx=1, labels=None, monitor=False):
        # 鍘熷 forward 鍑芥暟锛岀敤浜庤缁冨拰鏍囧噯娴嬭瘯 (淇濇寔鑷姩 reset)
        B, T, C, H, W = spike_train_batch.shape
        self.layer1.reset_state((B, C, H, W))

        mon_spikes = []
        mon_pooled_spikes = []
        mon_l3_voltages = []
        last_monitor_data = None

        for t in range(T):
            input_t = spike_train_batch[:, t, ...]
            s1, m_data1 = self.layer1.forward_step(input_t, t, training=(layer_idx == 1), monitor=monitor)

            if layer_idx == 1:
                if monitor:
                    mon_spikes.append(s1.detach())
                    s1_p = self.pool1(s1.float())
                    mon_pooled_spikes.append(s1_p.detach())
                    if s1.any(): last_monitor_data = m_data1
                continue

            s1_p = self.pool1(s1.float())
            if t == 0: self.layer2.reset_state(s1_p.shape)
            s2, m_data2 = self.layer2.forward_step(s1_p, t, training=(layer_idx == 2), monitor=monitor)

            if layer_idx == 2:
                if monitor:
                    mon_spikes.append(s2.detach())
                    if s2.any(): last_monitor_data = m_data2
                continue

            s2_p = self.pool2(s2.float())
            if t == 0: self.layer3.reset_state(s2_p.shape)
            is_training_l3 = (labels is not None) and self.training
            s3, m_data3 = self.layer3.forward_step(s2_p, t, labels=labels, training=is_training_l3, monitor=monitor)

            if monitor:
                mon_spikes.append(s3.detach())
                if s3.any(): last_monitor_data = m_data3
                if layer_idx == 3:
                    v_t_b0 = self.layer3.v_mem[0].detach().cpu().clone()
                    mon_l3_voltages.append(v_t_b0)

        if layer_idx == 1:
            self.layer1.batch_end_homeostasis()
        elif layer_idx == 2:
            self.layer2.batch_end_homeostasis()
        elif layer_idx == 3:
            if self.training:
                self.layer3.batch_end_homeostasis()

        return {
            'out_spikes': torch.stack(mon_spikes, dim=1) if monitor else None,
            'l3_voltage_trace': torch.stack(mon_l3_voltages, dim=0) if (monitor and len(mon_l3_voltages) > 0) else None,
            'monitor_data': last_monitor_data
        }

    def forward_dms_session(self, sample_spikes, test_spikes, delay_duration_steps=200, stsp_mode='dynamic'):
        """
        DMS 涓撶敤鍓嶅悜浼犳挱鍑芥暟 - 澧炲己鐗?
        stsp_mode: 'dynamic' (瀹為獙缁? 鎴?'static_frozen' (瀵圭収缁?
        """
        B, T_sample, C, H, W = sample_spikes.shape
        T_test = test_spikes.shape[1]

        # 1. 鍏ㄥ眬鍒濆鍖?/ 鐘舵€侀噸缃?
        self.layer1.reset_state((B, C, H, W))

        # 璁＄畻褰㈢姸
        h1 = (H + 2 * self.layer1.padding - self.layer1.kernel_size) // self.layer1.stride + 1
        w1 = (W + 2 * self.layer1.padding - self.layer1.kernel_size) // self.layer1.stride + 1
        h1_p, w1_p = h1 // 2, w1 // 2
        self.layer2.reset_state((B, self.layer1.out_channels, h1_p, w1_p))

        h2 = (h1_p + 2 * self.layer2.padding - self.layer2.kernel_size) // self.layer2.stride + 1
        w2 = (w1_p + 2 * self.layer2.padding - self.layer2.kernel_size) // self.layer2.stride + 1
        h2_p, w2_p = h2 // 2, w2 // 2
        self.layer3.reset_state((B, self.layer2.out_channels, h2_p, w2_p))

        # 鏁版嵁鏀堕泦瀹瑰櫒
        traces = {
            'l3_u': [],
            'l3_x': [],
            'l3_gain': [],
            'l2_spikes': [],
            'l3_v': []  # [鏂板] 鑶滅數浣嶈建杩?
        }

        current_time = 0

        # 瀹氫箟鍗曟鎵ц鍑芥暟锛岄€忎紶 stsp_mode
        def step_network(input_t, record=False):
            # Layer 1
            s1, _ = self.layer1.forward_step(input_t, current_time, training=False, stsp_mode=stsp_mode)
            s1_p = self.pool1(s1.float())

            # Layer 2
            s2, _ = self.layer2.forward_step(s1_p, current_time, training=False, stsp_mode=stsp_mode)
            s2_p = self.pool2(s2.float())

            # Layer 3
            # 寮€鍚?monitor=True 浠ユ崟鑾锋暟鎹?
            s3, m3 = self.layer3.forward_step(s2_p, current_time, training=False, monitor=record, stsp_mode=stsp_mode)

            if record:
                # 璁板綍鑶滅數浣?(Core Metric)
                traces['l3_v'].append(m3['v_raw'].detach().cpu())

                if self.layer3.enable_stsp and stsp_mode == 'dynamic':
                    traces['l3_u'].append(m3['stsp_u'].detach().cpu())
                    traces['l3_x'].append(m3['stsp_x'].detach().cpu())
                    gain_snapshot = m3.get('stsp_gain')
                    if gain_snapshot is None:
                        gain_snapshot = m3['stsp_u'] * m3['stsp_x']
                    traces['l3_gain'].append(gain_snapshot.detach().cpu())
                    traces['l2_spikes'].append(s2_p.detach().cpu())

        # ================= Phase 1: Sample =================
        for t in range(T_sample):
            step_network(sample_spikes[:, t, ...], record=True)
            current_time += 1

        # ================= Phase 2: Delay =================
        zero_input = torch.zeros((B, C, H, W), device=self.device)
        for t in range(delay_duration_steps):
            step_network(zero_input, record=True)
            current_time += 1

        # ================= Phase 3: Test =================
        for t in range(T_test):
            step_network(test_spikes[:, t, ...], record=True)
            current_time += 1

        self.layer1.reset_state((B, C, H, W))

        # 鏁寸悊缁撴灉
        res = {
            'v': torch.stack(traces['l3_v'], dim=0)  # [Total_Time, B, C, H, W]
        }
        if len(traces['l3_u']) > 0:
            res['u'] = torch.stack(traces['l3_u'], dim=0)
            res['x'] = torch.stack(traces['l3_x'], dim=0)
            res['gain'] = torch.stack(traces['l3_gain'], dim=0)
            res['spikes'] = torch.stack(traces['l2_spikes'], dim=0)

        return res

    # [鏇挎崲 SDNN_Network 绫讳腑鐨勫師 forward_dms_session 鏂规硶]
    def forward_classify_session(self, sample_spikes, test_spikes, delay_duration_steps=200, stsp_mode='dynamic'):
        """
        DMS 涓撶敤鍓嶅悜浼犳挱鍑芥暟 - 浼樺寲鐗?
        鐗规€э細
        1. 鏀寔 Batch 骞惰鎺ㄧ悊 (B > 1)
        2. 鍖呭惈鐩镐綅閲嶇疆 (Phase Reset)锛岃В鍐?Delay 瀵艰嚧鐨勯闂晥搴?
        """
        B, T_sample, C, H, W = sample_spikes.shape
        T_test = test_spikes.shape[1]

        # --- 1. 鍏ㄥ眬鍒濆鍖?---
        # 閲嶇疆鎵€鏈夊眰鐨勭姸鎬侊紝鍖呮嫭鑶滅數浣嶃€丼TSP鐘舵€佺瓑
        # 杩欓噷鐨?B 浼氳嚜鍔ㄩ€傚簲杈撳叆鐨?Batch Size
        self.layer1.reset_state((B, C, H, W))

        # 璁＄畻鍚勫眰杈撳嚭褰㈢姸骞堕噸缃?
        h1 = (H + 2 * self.layer1.padding - self.layer1.kernel_size) // self.layer1.stride + 1
        w1 = (W + 2 * self.layer1.padding - self.layer1.kernel_size) // self.layer1.stride + 1
        h1_p, w1_p = h1 // 2, w1 // 2
        self.layer2.reset_state((B, self.layer1.out_channels, h1_p, w1_p))

        h2 = (h1_p + 2 * self.layer2.padding - self.layer2.kernel_size) // self.layer2.stride + 1
        w2 = (w1_p + 2 * self.layer2.padding - self.layer2.kernel_size) // self.layer2.stride + 1
        h2_p, w2_p = h2 // 2, w2 // 2
        self.layer3.reset_state((B, self.layer2.out_channels, h2_p, w2_p))

        current_time = 0

        # 瀹氫箟鍗曟鎵ц杈呭姪鍑芥暟
        # force_l3_time: 鐢ㄤ簬寮哄埗鎸囧畾 Layer 3 鐨勬椂闂存锛屽疄鐜扮浉浣嶅榻?
        def step_network(input_t, force_l3_time=None):
            # Layer 1 (鐗╃悊鏃堕棿鑷劧娴侀€?
            s1, _ = self.layer1.forward_step(input_t, current_time, training=False, stsp_mode=stsp_mode)
            s1_p = self.pool1(s1.float())

            # Layer 2
            s2, _ = self.layer2.forward_step(s1_p, current_time, training=False, stsp_mode=stsp_mode)
            s2_p = self.pool2(s2.float())

            # Layer 3
            # 濡傛灉鎸囧畾浜?force_l3_time锛屽垯浣跨敤璇ユ椂闂磋鐩?current_time
            # 杩欑‘淇濅簡 Test 闃舵鏃犺浣曟椂寮€濮嬶紝鍐崇瓥绐楀彛 (Phase) 閮芥槸浠?0 寮€濮嬪榻愮殑
            t_for_l3 = current_time if force_l3_time is None else force_l3_time

            s3, m3 = self.layer3.forward_step(s2_p, t_for_l3, training=False, monitor=False, stsp_mode=stsp_mode)

        # ================= Phase 1: Sample =================
        for t in range(T_sample):
            step_network(sample_spikes[:, t, ...])
            current_time += 1

        # ================= Phase 2: Delay =================
        # 鏋勯€犲叏闆惰緭鍏ワ紝褰㈢姸鑷姩鍖归厤 Batch
        zero_input = torch.zeros((B, C, H, W), device=self.device)
        for t in range(delay_duration_steps):
            step_network(zero_input)
            current_time += 1

        # ================= Critical: Reset Decision State =================
        # 娓呴櫎 Sample 闃舵鍙兘浜х敓鐨勪换浣曞喅绛栬褰曪紙鍙戞斁鏃堕棿锛夛紝鍙繚鐣欒啘鐢典綅/STSP鐘舵€?
        self.layer3.reset_decision_state()

        # 棰濆绋冲仴鎬ф搷浣滐細灏嗚啘鐢典綅鍜屾姂鍒剁姸鎬佸綊浣?
        # 娉ㄦ剰锛氳繖閲屼笉閲嶇疆 STSP (u, x)锛屽洜涓洪偅鏄蹇嗙殑杞戒綋
        with torch.no_grad():
            self.layer3.v_mem.fill_(self.layer3.V_L)
            self.layer3.lateral_inh.reset_state(self.layer3.output_shape)

        # ================= Phase 3: Test (With Phase Reset) =================
        for t in range(T_test):
            # 浣跨敤 force_l3_time=t 杩涜鐩镐綅閲嶇疆
            # Layer 3 浼氳涓虹幇鍦ㄦ槸 Test 浠诲姟鐨勭 t 姣
            step_network(test_spikes[:, t, ...], force_l3_time=t)
            current_time += 1

        # ================= Result Collection =================
        # firing_times: [B, Neurons]
        flat_times = self.layer3.firing_times

        # 妫€鏌ユ槸鍚﹀彂鏀?
        has_fired = (flat_times != float('inf')).any(dim=1)

        # 鎵惧埌鏈€鏃╁彂鏀剧殑绁炵粡鍏冪储寮?
        _, min_indices = torch.min(flat_times, dim=1)
        npc = self.layer3.neurons_per_class
        pred_labels = min_indices // npc

        # 鏈彂鏀剧殑鏍锋湰鏍囪涓?-1
        pred_labels[~has_fired] = -1

        return {'prediction': pred_labels}

    def forward_dual_task_session(
            self,
            sample_spikes,
            distractor_spikes,
            probe_spikes,
            delay1_steps=200,
            delay2_steps=200,
            stsp_mode='dynamic',
            phase_reset=True
    ):
        """
        Dual-task session:
        Sample -> Delay1 -> Distractor -> Delay2 -> Probe

        Returns:
          - prediction_distractor: LongTensor [B]
          - prediction_probe: LongTensor [B]
          - first_fire_t_distractor: LongTensor [B], -1 means silent
          - first_fire_t_probe: LongTensor [B], -1 means silent
        """
        B, T_sample, C, H, W = sample_spikes.shape
        B_d, T_distractor, C_d, H_d, W_d = distractor_spikes.shape
        B_p, T_probe, C_p, H_p, W_p = probe_spikes.shape

        if (B_d, C_d, H_d, W_d) != (B, C, H, W):
            raise ValueError("distractor_spikes shape is incompatible with sample_spikes")
        if (B_p, C_p, H_p, W_p) != (B, C, H, W):
            raise ValueError("probe_spikes shape is incompatible with sample_spikes")

        # Global reset for a fresh session.
        self.layer1.reset_state((B, C, H, W))

        h1 = (H + 2 * self.layer1.padding - self.layer1.kernel_size) // self.layer1.stride + 1
        w1 = (W + 2 * self.layer1.padding - self.layer1.kernel_size) // self.layer1.stride + 1
        h1_p, w1_p = h1 // 2, w1 // 2
        self.layer2.reset_state((B, self.layer1.out_channels, h1_p, w1_p))

        h2 = (h1_p + 2 * self.layer2.padding - self.layer2.kernel_size) // self.layer2.stride + 1
        w2 = (w1_p + 2 * self.layer2.padding - self.layer2.kernel_size) // self.layer2.stride + 1
        h2_p, w2_p = h2 // 2, w2 // 2
        self.layer3.reset_state((B, self.layer2.out_channels, h2_p, w2_p))

        current_time = 0
        zero_input = torch.zeros((B, C, H, W), device=self.device)

        def step_network(input_t, force_l3_time=None):
            nonlocal current_time

            s1, _ = self.layer1.forward_step(input_t, current_time, training=False, stsp_mode=stsp_mode)
            s1_p = self.pool1(s1.float())

            s2, _ = self.layer2.forward_step(s1_p, current_time, training=False, stsp_mode=stsp_mode)
            s2_p = self.pool2(s2.float())

            t_for_l3 = current_time if force_l3_time is None else force_l3_time
            self.layer3.forward_step(s2_p, t_for_l3, training=False, monitor=False, stsp_mode=stsp_mode)
            current_time += 1

        def reset_decision_window():
            self.layer3.reset_decision_state()
            if phase_reset:
                with torch.no_grad():
                    self.layer3.v_mem.fill_(self.layer3.V_L)
                    self.layer3.lateral_inh.reset_state(self.layer3.output_shape)

        def decode_current_prediction():
            flat_times = self.layer3.firing_times
            has_fired = (flat_times != float('inf')).any(dim=1)
            min_times, min_indices = torch.min(flat_times, dim=1)
            pred_labels = min_indices // self.layer3.neurons_per_class
            pred_labels = pred_labels.long()
            pred_labels[~has_fired] = -1

            first_fire_t = min_times.clone()
            first_fire_t[~has_fired] = -1
            first_fire_t = first_fire_t.to(torch.long)
            return pred_labels, first_fire_t

        # Phase A: Sample
        for t in range(T_sample):
            step_network(sample_spikes[:, t, ...])

        # Phase B: Delay1
        for t in range(delay1_steps):
            step_network(zero_input)

        # Phase C: Distractor classification window
        reset_decision_window()
        for t in range(T_distractor):
            force_t = t if phase_reset else None
            step_network(distractor_spikes[:, t, ...], force_l3_time=force_t)
        prediction_distractor, first_fire_t_distractor = decode_current_prediction()

        # Phase D: Delay2
        for t in range(delay2_steps):
            step_network(zero_input)

        # Phase E: Probe classification window
        reset_decision_window()
        for t in range(T_probe):
            force_t = t if phase_reset else None
            step_network(probe_spikes[:, t, ...], force_l3_time=force_t)
        prediction_probe, first_fire_t_probe = decode_current_prediction()

        return {
            'prediction_distractor': prediction_distractor,
            'prediction_probe': prediction_probe,
            'first_fire_t_distractor': first_fire_t_distractor,
            'first_fire_t_probe': first_fire_t_probe,
        }

    def forward_dual_task_spike_trace_session(
            self,
            sample_spikes,
            distractor_spikes,
            probe_spikes,
            delay1_steps=200,
            delay2_steps=200,
            stsp_mode='dynamic',
            phase_reset=True
    ):
        """
        Dual-task session with full-layer spike tracing.

        Returns:
          - layer1_spikes: BoolTensor [T_total, B, C1, H1, W1]
          - layer2_spikes: BoolTensor [T_total, B, C2, H2, W2]
          - layer3_spikes: BoolTensor [T_total, B, C3, H3, W3]
          - phase_slices: dict[str, [start, end)] over the global time axis
          - predictions: dict with distractor/probe predictions and first-fire times
        """
        B, T_sample, C, H, W = sample_spikes.shape
        B_d, T_distractor, C_d, H_d, W_d = distractor_spikes.shape
        B_p, T_probe, C_p, H_p, W_p = probe_spikes.shape

        if (B_d, C_d, H_d, W_d) != (B, C, H, W):
            raise ValueError("distractor_spikes shape is incompatible with sample_spikes")
        if (B_p, C_p, H_p, W_p) != (B, C, H, W):
            raise ValueError("probe_spikes shape is incompatible with sample_spikes")

        self.layer1.reset_state((B, C, H, W))

        h1 = (H + 2 * self.layer1.padding - self.layer1.kernel_size) // self.layer1.stride + 1
        w1 = (W + 2 * self.layer1.padding - self.layer1.kernel_size) // self.layer1.stride + 1
        h1_p, w1_p = h1 // 2, w1 // 2
        self.layer2.reset_state((B, self.layer1.out_channels, h1_p, w1_p))

        h2 = (h1_p + 2 * self.layer2.padding - self.layer2.kernel_size) // self.layer2.stride + 1
        w2 = (w1_p + 2 * self.layer2.padding - self.layer2.kernel_size) // self.layer2.stride + 1
        h2_p, w2_p = h2 // 2, w2 // 2
        self.layer3.reset_state((B, self.layer2.out_channels, h2_p, w2_p))

        current_time = 0
        zero_input = torch.zeros((B, C, H, W), device=self.device)
        phase_slices = {}

        traces_l1 = []
        traces_l2 = []
        traces_l3 = []

        def step_network(input_t, force_l3_time=None):
            nonlocal current_time

            s1, _ = self.layer1.forward_step(input_t, current_time, training=False, stsp_mode=stsp_mode)
            s1_p = self.pool1(s1.float())

            s2, _ = self.layer2.forward_step(s1_p, current_time, training=False, stsp_mode=stsp_mode)
            s2_p = self.pool2(s2.float())

            t_for_l3 = current_time if force_l3_time is None else force_l3_time
            s3, _ = self.layer3.forward_step(s2_p, t_for_l3, training=False, monitor=False, stsp_mode=stsp_mode)

            traces_l1.append(s1.detach().to(torch.bool))
            traces_l2.append(s2.detach().to(torch.bool))
            traces_l3.append(s3.detach().to(torch.bool))
            current_time += 1

        def reset_decision_window():
            self.layer3.reset_decision_state()
            if phase_reset:
                with torch.no_grad():
                    self.layer3.v_mem.fill_(self.layer3.V_L)
                    self.layer3.lateral_inh.reset_state(self.layer3.output_shape)

        def decode_current_prediction():
            flat_times = self.layer3.firing_times
            has_fired = (flat_times != float('inf')).any(dim=1)
            min_times, min_indices = torch.min(flat_times, dim=1)
            pred_labels = (min_indices // self.layer3.neurons_per_class).long()
            pred_labels[~has_fired] = -1

            first_fire_t = min_times.clone()
            first_fire_t[~has_fired] = -1
            first_fire_t = first_fire_t.to(torch.long)
            return pred_labels, first_fire_t

        def run_phase(phase_name, tensor_seq, use_phase_reset_clock=False):
            start_t = current_time
            for t in range(tensor_seq.shape[1]):
                force_t = t if use_phase_reset_clock else None
                step_network(tensor_seq[:, t, ...], force_l3_time=force_t)
            phase_slices[phase_name] = [int(start_t), int(current_time)]

        def run_zero_phase(phase_name, steps):
            start_t = current_time
            for _ in range(int(steps)):
                step_network(zero_input)
            phase_slices[phase_name] = [int(start_t), int(current_time)]

        # A: Sample
        run_phase("sample", sample_spikes, use_phase_reset_clock=False)

        # B: Delay1
        run_zero_phase("delay1", delay1_steps)

        # C: Distractor
        reset_decision_window()
        run_phase("distractor", distractor_spikes, use_phase_reset_clock=phase_reset)
        prediction_distractor, first_fire_t_distractor = decode_current_prediction()

        # D: Delay2
        run_zero_phase("delay2", delay2_steps)

        # E: Probe
        reset_decision_window()
        run_phase("probe", probe_spikes, use_phase_reset_clock=phase_reset)
        prediction_probe, first_fire_t_probe = decode_current_prediction()

        layer1_spikes = torch.stack(traces_l1, dim=0).cpu()
        layer2_spikes = torch.stack(traces_l2, dim=0).cpu()
        layer3_spikes = torch.stack(traces_l3, dim=0).cpu()

        return {
            "layer1_spikes": layer1_spikes,
            "layer2_spikes": layer2_spikes,
            "layer3_spikes": layer3_spikes,
            "phase_slices": phase_slices,
            "predictions": {
                "prediction_distractor": prediction_distractor.detach().cpu(),
                "prediction_probe": prediction_probe.detach().cpu(),
                "first_fire_t_distractor": first_fire_t_distractor.detach().cpu(),
                "first_fire_t_probe": first_fire_t_probe.detach().cpu(),
            },
        }

    def forward_dms_spike_trace_session(
            self,
            sample_spikes,
            probe_spikes,
            delay_steps=200,
            stsp_mode='dynamic',
            phase_reset=True
    ):
        """
        DMS session with full-layer spike tracing.

        Returns:
          - layer1_spikes: BoolTensor [T_total, B, C1, H1, W1]
          - layer2_spikes: BoolTensor [T_total, B, C2, H2, W2]
          - layer3_spikes: BoolTensor [T_total, B, C3, H3, W3]
          - phase_slices: dict[str, [start, end)] over the global time axis
          - predictions: dict with probe prediction and first-fire time
        """
        B, T_sample, C, H, W = sample_spikes.shape
        B_p, T_probe, C_p, H_p, W_p = probe_spikes.shape

        if (B_p, C_p, H_p, W_p) != (B, C, H, W):
            raise ValueError("probe_spikes shape is incompatible with sample_spikes")

        self.layer1.reset_state((B, C, H, W))

        h1 = (H + 2 * self.layer1.padding - self.layer1.kernel_size) // self.layer1.stride + 1
        w1 = (W + 2 * self.layer1.padding - self.layer1.kernel_size) // self.layer1.stride + 1
        h1_p, w1_p = h1 // 2, w1 // 2
        self.layer2.reset_state((B, self.layer1.out_channels, h1_p, w1_p))

        h2 = (h1_p + 2 * self.layer2.padding - self.layer2.kernel_size) // self.layer2.stride + 1
        w2 = (w1_p + 2 * self.layer2.padding - self.layer2.kernel_size) // self.layer2.stride + 1
        h2_p, w2_p = h2 // 2, w2 // 2
        self.layer3.reset_state((B, self.layer2.out_channels, h2_p, w2_p))

        current_time = 0
        zero_input = torch.zeros((B, C, H, W), device=self.device)
        phase_slices = {}

        traces_l1 = []
        traces_l2 = []
        traces_l3 = []

        def step_network(input_t, force_l3_time=None):
            nonlocal current_time

            s1, _ = self.layer1.forward_step(input_t, current_time, training=False, stsp_mode=stsp_mode)
            s1_p = self.pool1(s1.float())

            s2, _ = self.layer2.forward_step(s1_p, current_time, training=False, stsp_mode=stsp_mode)
            s2_p = self.pool2(s2.float())

            t_for_l3 = current_time if force_l3_time is None else force_l3_time
            s3, _ = self.layer3.forward_step(s2_p, t_for_l3, training=False, monitor=False, stsp_mode=stsp_mode)

            traces_l1.append(s1.detach().to(torch.bool))
            traces_l2.append(s2.detach().to(torch.bool))
            traces_l3.append(s3.detach().to(torch.bool))
            current_time += 1

        def reset_decision_window():
            self.layer3.reset_decision_state()
            if phase_reset:
                with torch.no_grad():
                    self.layer3.v_mem.fill_(self.layer3.V_L)
                    self.layer3.lateral_inh.reset_state(self.layer3.output_shape)

        def decode_current_prediction():
            flat_times = self.layer3.firing_times
            has_fired = (flat_times != float('inf')).any(dim=1)
            min_times, min_indices = torch.min(flat_times, dim=1)
            pred_labels = (min_indices // self.layer3.neurons_per_class).long()
            pred_labels[~has_fired] = -1

            first_fire_t = min_times.clone()
            first_fire_t[~has_fired] = -1
            first_fire_t = first_fire_t.to(torch.long)
            return pred_labels, first_fire_t

        def run_phase(phase_name, tensor_seq, use_phase_reset_clock=False):
            start_t = current_time
            for t in range(tensor_seq.shape[1]):
                force_t = t if use_phase_reset_clock else None
                step_network(tensor_seq[:, t, ...], force_l3_time=force_t)
            phase_slices[phase_name] = [int(start_t), int(current_time)]

        def run_zero_phase(phase_name, steps):
            start_t = current_time
            for _ in range(int(steps)):
                step_network(zero_input)
            phase_slices[phase_name] = [int(start_t), int(current_time)]

        # A: Sample
        run_phase("sample", sample_spikes, use_phase_reset_clock=False)

        # B: Delay
        run_zero_phase("delay", delay_steps)

        # C: Probe
        reset_decision_window()
        run_phase("probe", probe_spikes, use_phase_reset_clock=phase_reset)
        prediction_probe, first_fire_t_probe = decode_current_prediction()

        layer1_spikes = torch.stack(traces_l1, dim=0).cpu()
        layer2_spikes = torch.stack(traces_l2, dim=0).cpu()
        layer3_spikes = torch.stack(traces_l3, dim=0).cpu()

        return {
            "layer1_spikes": layer1_spikes,
            "layer2_spikes": layer2_spikes,
            "layer3_spikes": layer3_spikes,
            "phase_slices": phase_slices,
            "predictions": {
                "prediction_probe": prediction_probe.detach().cpu(),
                "first_fire_t_probe": first_fire_t_probe.detach().cpu(),
            },
        }

    def get_kernels(self, layer_idx=1):
        if layer_idx == 1: return self.layer1.kernels.detach().cpu().numpy()
        if layer_idx == 2: return self.layer2.kernels.detach().cpu().numpy()
        if layer_idx == 3: return self.layer3.kernels.detach().cpu().numpy()
        return None
