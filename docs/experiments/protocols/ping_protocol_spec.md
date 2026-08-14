# Paper Figure P4 Ping Protocol Spec

生成日期：2026-05-28

本文档只固化 `src/experiments/paper_figures/` 当前 gold-standard code 中的 Ping 电流注入范式。它不是重构方案，也不改变源码。

## 1. 范式定义

P4 Ping 范式的核心操作定义为：

```text
given restored STSP/network state
  -> use zero external image/spike input
  -> inject same-magnitude current into all or selected Layer 1 input sites
  -> let the injected current pass through the normal STSP gain path
  -> decode the result from Layer 3 decision/readout state
```

关键点：

- Ping 不是正常图像 spike 输入，也不是 weak cue。
- Ping 是对 Layer 1 输入空间中全部或部分神经元/位置施加同大小电流。
- Ping 电流必须进入 `ping_drive` 路径，并接受 STSP gain 调制。
- Ping 之后的结果以 Layer 3 readout / decision / prediction 为主。
- Layer 1 spike trace、active sites、total current 可以作为辅助指标，但不能替代 Layer 3 readout。

## 2. 与相邻范式的边界

| 范式 | 当前用户确认定义 | 与 Ping 的边界 |
|---|---|---|
| P5 Weak cue / partial cue | 在 STSP 状态下，用更少的有效输入恢复判断；从非零像素中随机选择或按区域选择一定比例/数量的输入，观察准确率恢复 | Weak cue 输入仍然来自图像/encoded spikes；Ping 输入是无类别信息的电流驱动 |
| P1/P2/P3 state generation | 一张图输入加 delay 构成一个完整输入单元；输入不同个数的单元后，最终截图作为后续实验的 STSP 状态基础 | state generation 负责产生可恢复状态；Ping 只消费这些状态并做功能读出 |

## 3. Gold-code 证据

### 3.1 STSP 调制证据

`src/core/network.py` 是 Ping 是否受 STSP 调制的核心证据：

- `network.py:212-225`：`forward_physics(..., ping_drive=None)` 将 `input_spikes` 与 `ping_drive` 合并为 presynaptic input。
- `network.py:227-236`：`stsp_mode == "dynamic"` 时，`gain` 来自 `stsp_dynamics_jit(...)`，并且 `effective_ping = ping_drive_f * gain`。
- `network.py:354-359`：Layer 1 `forward_step` 将 `ping_drive` 传入 `forward_physics`。
- `network.py:508-522`：Layer 3 readout 层也走同一个 `forward_physics` / `stsp_mode` 接口；但当前 P4 ping drive 实际注入点在 Layer 1。

结论：当前 gold code 中，只要 ping 通过 `layer1.forward_step(..., stsp_mode="dynamic", ping_drive=ping_drive)` 进入网络，ping current 就是 STSP-modulated current。

### 3.2 Layer 1 注入与 Layer 3 readout 证据

| 位置 | 证据 | 结论 |
|---|---|---|
| `fig2/subexperiments/helpers.py:682-711` | `run_ping_readout_from_boundary()` restore state 后创建 `zero_input`，用 `ping_drive` 做 rollout，最后 `decode_prediction_and_fire_time_from_layer3()` | Fig.2 ping 是 restored state + L1 current drive + L3 readout |
| `fig2/subexperiments/helpers.py:798-803` | `_forward_three_layers_with_optional_trace()` 将 `ping_drive` 传给 Layer 1，并以 dynamic STSP 推进 Layer 1/2/3 | Fig.2 helper 保持 STSP dynamic path |
| `fig3/subexperiments/helpers_1.py:365-377` | `_run_ping_from_boundary()` 使用 `torch.full_like(zero, ping_amp)` 作为 full-field ping，再 L3 decode | Fig.3 neutral ping 是全 L1 current drive |
| `fig3/subexperiments/helpers_1.py:422-427` | `_step_network_once(..., ping_drive=...)` 将 ping drive 传给 Layer 1，Layer 2/3 继续 dynamic STSP | Fig.3 ping helper 的核心路径与定义一致 |
| `fig3/subexperiments/region_ping.py:194-222` | `_run_masked_ping_from_boundary()` 将 `region_mask * ping_amp` 作为 masked L1 ping drive，再 L3 decode | Fig.3 region ping 是局部/区域 L1 current drive |
| `fig3/subexperiments/region_ping.py:224-270` | batch 版本同样构造 `mask_batch * ping_amp`，并记录 active units / total current | 区域比较需要记录 active count 和 total current |
| `fig6/subexperiments/helpers_1.py:698-731` | `_run_masked_ping_layer1_capture()` 用 `region_mask * ping_amp` 注入 L1，保存 L1 trace，最后 L3 decode | Fig.6 ping 是 masked/global L1 current drive，Layer 1 trace 是辅助输出 |

## 4. P4 Task 矩阵

| Task | 当前角色 | State source | Target scope | Amplitude / timing | 主 readout | 输出 artifact | Change gate |
|---|---|---|---|---|---|---|---|
| `fig2.neutral_ping` | pair-state neutral ping | `PairEpisodeStateBank.boundary_states`，遍历 `STATE_CONDITIONS` | 当前 helper 默认全 Layer 1 input sites | `Fig2Config.ping_amp=1.0`, `ping_ms=30`, `ping_repeats=1`; 可选 `ping_mode` / `ping_noise` | `prediction`, `first_fire_time_ms`, pair-member readout | `data/raw/panel_e_neutral_ping_trial_readout.csv`; `data/metrics/panel_e_neutral_ping_metrics.csv`; optional `panel_e_neutral_ping_l3_traces.npz` | R3 |
| `fig2.ping_sweep` | Fig.2 ping 参数扫描 | 同 `fig2.neutral_ping` | 全 Layer 1 input sites | amp sweep `(0.5, 1.0, 1.5)`；duration sweep `(10, 30, 60)` ms | pair-member / A / B / other readout rates | `data/raw/supp_ping_sweep_trial_readout.csv`; `data/metrics/supp_ping_sweep_metrics.csv` | R3 |
| `fig3.neutral_ping` | sequence-state neutral ping | Fig.3 sequence bank boundaries；默认 `S_final` 与 `S0` | 全 Layer 1 input sites | `Fig3Config.ping_amp=1.0`, `ping_ms=30`, `ping_repeats=1` | serial-position distribution from L3 prediction | `data/raw/panel_d_neutral_ping_trial_readout.csv`; `data/metrics/panel_d_ping_position_distribution.csv`; `panel_d/panel_e` summary copies | R3 |
| `fig3.region_ping` | support-map region ping | Fig.3 sequence bank landscape/boundary | `peak`, `valley`, `random` region masks；`region_ping_q=0.20`; count-matched | `ping_amp=1.0`, `ping_ms=30`, `region_ping_repeats=5` | predicted label / serial bin / region contrast | `data/raw/panel_f_region_ping_trial_readout.csv`; `data/metrics/panel_f_region_ping_*` | R3 |
| `fig3.region_ping_s0_control` | region ping control modifier | Same region ping path | Adds S0 condition through `run_region_ping_s0_control` flag | Same as region ping | same schema, state_condition distinguishes control | no separate compute file; thin wrapper to current subexperiment runner | R3 |
| `fig3.region_ping_amp_sweep` | region ping amplitude modifier | Same region ping path | Same masks as region ping | `region_ping_amp_sweep=(0.25, 0.5, 1.0, 1.5)` | sweep summary and latency | `data/raw/supp_region_ping_amp_sweep_trial_readout.csv`; `data/metrics/supp_region_ping_amp_sweep_*` | R3 |
| `fig6.field_ping_readout` | Fig.6 field/local ping | Fig.6 sequence bank final boundary | score-region masks from gain-ratio map; `basin_top_q=0.20` | `Fig6Config.ping_amp=1.0`, `ping_ms=30` | serial age bin / readout label from L3 prediction | `data/metrics/panel_b_region_ping_readout_bias.csv` | R3 |
| `fig6.global_ping_score_spike_prediction` | Fig.6 global ping | Fig.6 sequence bank final boundary | all finite sites in gain-ratio map | `global_ping_amp=0.5`, `global_ping_ms=30` | L3 prediction plus Layer 1 early spike/score quantile metrics | `data/metrics/panel_c_global_ping_score_spike_prediction.csv` | R3 |
| `fig6.ping_score_spike_prediction` | alias/readout wrapper | Same as global ping | Same as global ping | Same as global ping | delegates to `compute_global_ping_score_spike_prediction()` | same as global ping | R3 |

## 5. `PingProtocolSpec` 字段

后续每个 P4 task 都应能被描述为下面的 schema。当前阶段只用于文档和审核，不要求改代码。

```text
task_id
figure_id
state_source:
  state_family: pair_state / sequence_state / score_state
  boundary_key: S0 / S_final / S_A / S_B / S_AB / figure-specific
  restore_mode
target:
  target_layer: layer1
  target_space: input_shape / entry_map / score_region
  target_scope: global / full_field / peak_region / valley_region / random_region / score_region
  mask_generation
  active_site_count_rule
drive:
  drive_type: current_drive
  input_spikes: zero
  amplitude
  duration_ms
  duration_steps
  repeat_count
  seed_rule
  optional_noise
stsp:
  stsp_mode: dynamic
  ping_drive_path: layer1.forward_step(..., ping_drive=...)
  modulation_rule: effective_ping = ping_drive * STSP gain
readout:
  primary_layer: layer3
  decoder: decode_prediction_and_fire_time_from_layer3
  primary_columns
  auxiliary_columns
artifacts:
  raw_outputs
  metric_outputs
  optional_trace_outputs
human_gate:
  change_gate: R3
  protected_fields
```

## 6. Hard Rules

这些规则应成为后续 AI 整理 ping 相关代码时的硬约束：

1. Ping 不能改成图像 spike 输入；图像/encoded spike 属于 weak cue 或 probe。
2. Ping 的主输入应为 `zero_input + ping_drive`。
3. Ping drive 必须进入 Layer 1 `forward_step(..., ping_drive=...)`。
4. Ping drive 必须保留 STSP dynamic 调制，即不能绕过 `effective_ping = ping_drive * gain`。
5. P4 的主结果必须以 Layer 3 decode/readout 为准；Layer 1 trace 只作为辅助解释或 score/spike coupling。
6. 区域/局部 ping 必须记录 mask 生成规则、active site count、total current。
7. 区域比较必须保留 count-matching 或 current-matching 证据；否则不能把差异解释为 STSP/region effect。
8. `S0` control、amp sweep、duration sweep 都是 modifier，不是新的范式。
9. 任何改 `ping_amp`、`ping_ms`、`ping_steps`、mask generation、restore mode、L3 decoder 的行为都必须按 R3 人工审核。
10. 任何尝试 batch 化 ping rollout 都必须先做 equivalence check；当前 Fig.2 已有“condition batching changes threshold-sensitive readout predictions”的警告。

## 7. 当前可 AI 推进的整理任务

这些任务不改变源码，可以继续由 AI 做：

| 任务 | 涉及文件 | 输出 |
|---|---|---|
| 提取 P4 参数矩阵 | `fig2/types.py`, `fig3/types.py`, `fig6/types.py` | `ping_amp`, `ping_ms`, sweep grid, repeats, mode/noise |
| 提取 P4 artifact contract | 9 个 P4 task 文件与 output contract | raw/metrics/trace 输出字段表 |
| 标记 P4 modifier | `region_ping_s0_control`, `region_ping_amp_sweep`, `ping_sweep` | modifier 表，而不是独立范式 |
| 记录 gold-code evidence | helper 文件与 `network.py` | 证据链 checklist |
| 生成 human review checklist | 所有 P4 task | 后续改代码前的人审清单 |

## 8. 不能直接 AI 修改的内容

以下内容虽然已经被 gold code 明确，但仍然不能直接由 AI 修改：

- `src/core/network.py` 中 `ping_drive` 与 STSP gain 的关系。
- `fig2/subexperiments/helpers.py::run_ping_readout_from_boundary` 的 rollout/readout 语义。
- `fig3/subexperiments/helpers_1.py::_run_ping_from_boundary` 与 `_step_network_once`。
- `fig3/subexperiments/region_ping.py` 的 region mask、count matching、S0 control、amp sweep。
- `fig6/subexperiments/helpers_1.py::_run_masked_ping_layer1_capture` 的 trace capture 与 L3 readout。
- Fig.2 condition batching、Fig.3/6 ping batching、或任何改变 threshold-sensitive prediction 的优化。

## 9. 下一步建议

下一步应继续做 P4 的两个只读交付物：

1. `PingProtocolSpec` 实例表：为 9 个 P4 task 各填一行完整字段。
2. P4 artifact schema 表：列出每个 raw/metric CSV 的关键列、是否 panel 必需、是否可作为回归测试基线。

完成这两个表后，P4 的审核等级可以稳定保持为：

```text
Audit level: R2
Change gate: R3
```

也就是：AI 可以继续整理协议、参数和契约；未来改行为时必须人工门控。
