# Paper Figures 优化原则与失败经验总结

本文档总结的是本轮 paper figures 修复与效率优化产生的通用经验。它刻意不按单个 panel 复述细节，而是把成功优化和失败尝试抽象为后续可复用的判断原则。

核心结论：论文图实验优化的目标不是“把更多代码搬到 GPU”或“让 GPU 利用率看起来更高”，而是在保持科学语义、输出契约和可复现性的前提下减少 wall-clock 时间。凡是不能证明结果等价的加速，都不应成为默认路径；失败的 batch 尝试本身是重要证据，应作为以后设计 deterministic rollout 的边界条件。

## 1. 优化首先是语义保持问题，不是性能问题

paper figures 的实验输出不是普通数值程序输出。很多指标来自阈值发放、首次发放时间、离散预测、STSP 状态恢复、条件扰动和 trace 派生量。对这类系统，微小的执行路径变化可能被阈值机制放大为离散输出变化。

因此优化前必须先回答：

- 这个任务是否真的是同构任务，只是 row 数更多？
- 每一行的模型状态、输入 mask、随机 seed、扰动语义是否完全等价？
- batch 后能否按原 row order 无损拆回？
- 输出差异是否只可能来自浮点级别，而不是 prediction、first-fire、accuracy、switch、DPI 等科学变量改变？

如果这些问题不能同时回答清楚，优化就不能直接默认启用。

## 2. 可以默认化的优化通常有三个特征

第一类是纯数据搬运优化。典型模式是把 timestep 循环中的 `.detach().cpu()` 改为 device-side 累积，阶段末统一 `stack().cpu().numpy()`。这种优化只改变 CPU transfer 时机，不改变输入、网络状态、随机流、mask、condition 或输出 row key，因此通常风险最低。

第二类是真正同构的 rollout batch。它要求每个 job 的状态恢复、输入 shape、STSP mode、扰动规则和 readout 规则完全一致，差别只在 batch row。成功 batch 后必须能用 job table 拆回原来的 `sequence_id`、`pair_id`、`condition`、`probe_id`、`mask_id` 等 row key。

第三类是后处理或 null/sweep 的 CPU shard。统计、null sampling、CSV 汇总、排序、绘图构建等不应占用 GPU 主 rollout 窗口。它们更适合 CPU 分片和 deterministic merge，前提是 seed 分配、row key 和最终排序稳定。

## 3. 不能默认化的优化通常暴露四类风险

第一类是把异构 condition 粗暴拼成一个 batch。不同 condition 如果代表不同 state restore、mask、STSP mode、probe reset 或 perturbation 语义，它们不是同一个计算任务的多行，而是多个实验条件。直接拼 batch 可能改变边界状态、执行顺序或阈值轨迹。

第二类是把本来独立的 readout job 合并成一次读出。functional readout 往往依赖干净状态重建、选定 STSP trace、decision state reset、lateral inhibition reset 和特定 probe/ping seed。合并后即使 row order 正确，也可能改变 batch shape、随机流或阈值发放路径。

第三类是把 smoke PASS 当成中等规模等价证明。smoke 只能证明路径可执行和 bundle 结构大体可用，不能覆盖足够多的 near-threshold case。阈值系统的失败常常只在 medium 或更大规模中出现。

第四类是把 layout/build PASS 当成科学等价。layout validation 和 check-only build 只能证明文件存在、字段可读、图能构建；它们不能证明 prediction、trace、DPI、AUC、summary metrics 没有漂移。

## 4. 失败判据要看“变量性质”，不只看差异大小

对 paper figures，失败不是只有脚本报错才算失败。以下差异都应直接判为 unsafe：

- `prediction`、`pred_is_*`、`correctness`、`probe_accuracy`、`output_switch` 等离散列改变。
- `first_fire_time`、`first_fire_time_ms`、decision-spike displacement 等发放时间改变。
- trace NPZ、class evidence、grouped voltage、DPI timecourse 等底层轨迹改变。
- 下游 AUC、p50、summary、contrast、regression controls 因上游离散列或 trace 改变而漂移。
- 加速版本 medium runtime 无收益或负收益，即使 regression PASS，也不应默认启用。

反过来，只有 CSV/NPZ 在稳定 key 对齐后通过严格 regression，layout validation 通过，build check 通过，并且 wall-clock 有可解释收益，才可以考虑接入总控默认 flag。

## 5. 失败经验的正面价值

失败 batch 尝试不应简单删除成“没做成”。它们提供了后续重写 deterministic rollout 的边界：

- 哪些变量对 batch shape 敏感。
- 哪些 condition 不能共享一次 rollout。
- 哪些 readout 必须保持独立 reset 和独立 seed。
- 哪些输出最早暴露科学漂移。
- 哪些优化只是 smoke 可行，但 medium 不安全。

这些信息比一次局部加速更有价值，因为它们能防止后续在全量实验里引入隐蔽的科学输出漂移。

## 6. 默认优化策略

后续优化应按以下优先级推进。

1. 先做不改变语义的同步点优化：延迟 CPU transfer、减少非必要 trace、避免重复编码和重复读取。
2. 再做同构任务 batch：同一模型状态语义、同一 input shape、同一 reset 规则、同一扰动类型。
3. 对 post-hoc 或统计型任务使用 CPU shard，并保证 deterministic merge。
4. 对 threshold-sensitive readout、condition merge、perturbation merge 保持 opt-in 或审计项，除非 medium regression 已证明安全。
5. 对无收益或负收益的安全 batch，不接入默认总控；保留为手动实验 flag 或直接回退 serial。

## 7. 每个 batch helper 必须携带 job table

任何新的 batch helper 都应显式构建 job table，而不是依赖隐式循环顺序。job table 至少应记录：

- 原始 row key，例如 `pair_id`、`sequence_id`、`probe_id`、`condition`、`repeat_id`、`mask_id`。
- 输入来源，例如 sample/probe image id、boundary id、state condition。
- 随机来源，例如 seed、mask seed、null seed。
- 扰动语义，例如 mask name、reset target、removed index、condition group。
- batch 内位置，例如 `batch_index`、`row_offset`、`condition_offset`。

batch 输出必须只通过这个 job table 拆回原始 row order。不能依赖“看起来循环顺序一致”。

## 8. 状态恢复要显式区分完整状态和选择性状态

paper figures 中有两类常见恢复：

- 完整 boundary restore：恢复 `v_mem/g_e/res/inh_trace/u/x` 等完整运行状态。
- 选择性 STSP restore：重建快速状态，只恢复 `u/x`，用于把 STSP 作为 causal memory source。

这两类恢复语义不同，不能在 batch helper 中混用或隐式替代。只恢复 `u/x` 的 readout 尤其容易受到阈值状态、decision reset、lateral inhibition reset 和 batch shape 的影响。后续若要 batch 这类路径，需要先定义独立的 deterministic readout state object，而不是复用普通 condition concat。

## 9. 验证标准必须分层

每次优化都应至少经过以下验证层级：

1. Syntax/import check：只能证明代码可加载，不能证明结果正确。
2. Smoke run：证明路径可执行、bundle 能生成。
3. Medium serial-vs-optimized regression：证明主要科学输出等价。
4. Layout validation：证明 bundle 契约仍然可读。
5. Figure build check：证明绘图层仍能消费优化后的 bundle。
6. Runtime comparison：证明默认启用有实际 wall-clock 收益。

其中第 3 步是科学安全门槛。没有 medium regression PASS 的优化，不应被描述为完成。

## 10. 通过与失败都要写入 manifest 或文档

优化结果应留下机器可读和人工可读两类记录：

- 机器可读：run manifest、summary、validation summary。
- 人工可读：说明优化改了什么、哪些输出通过、哪些输出失败、失败变量是什么、为什么这些变量说明语义改变。

失败路径也应记录原因，例如 `threshold-sensitive spike dynamics`、`readout prediction changed`、`condition merge changes trace`、`negative runtime benefit`。这样后续看到一个 flag 回退 serial 时，不会误以为是实现遗漏。

## 11. 默认 flag 只应启用“安全且正收益”的路径

`--enable-gpu-batching` 这类总控 flag 应代表“启用所有已证明安全且值得默认启用的优化”。它不应盲目转发所有存在的 batch flag。

推荐规则：

- regression PASS 但 runtime 负收益：不默认启用。
- smoke PASS 但 medium FAIL：不默认启用。
- layout/build PASS 但 regression FAIL：不默认启用。
- 只能证明局部输出等价、不能证明主图 contract 等价：不默认启用。
- 安全回退 serial 可以保留，但不应被宣传为真实 batch speedup。

## 12. 成功经验与失败经验的统一解释

本轮修复的根本经验是：优化必须沿着“计算同构性”推进，而不是沿着“代码位置看起来相似”推进。

相似的循环不一定能 batch；相同的函数入口也不一定代表相同的实验语义。真正能 batch 的，是那些状态、输入、扰动、随机性和输出拆分规则都同构的任务。真正不能 batch 的，往往正是论文 claim 最敏感的 causal manipulation 或 threshold readout。

因此后续的工程原则应是：

- 先证明同构，再写 batch。
- 先保护 row key，再追求吞吐。
- 先看 trace/prediction 是否等价，再看 summary 是否接近。
- 先保留科学语义，再讨论 GPU 利用率。
- 失败要保留为边界知识，而不是只留下“已回退 serial”。

这个原则比任何单个 fig 的优化更重要。它决定了后续 paper figures 代码可以在加速的同时保持论文证据链可信。

