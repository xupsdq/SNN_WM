# Masse-LIF 去慢状态 STSP 匹配对照规格

**状态：** IMPLEMENTING  
**核心决定：** 保留已训成的无 STSP Masse delayed-cue LIF 作为历史基线，另从初始化训练一对去掉占位慢变量的匹配循环 LIF：一侧无 STSP，一侧带 Masse 式脉冲触发、按细胞共享的 STSP；本阶段 1 个 seed 交付行为门槛与延迟末机制解码。  
**交付目标：** 在没有 800 ms 突触电流和 SFA 的前提下，检验 STSP 是否成为可训练的静默承载，而不是给已经能做对任务的 formal 网补动力学。  
**前置规格：** [`masse_delayed_cue_lif_task_spec.md`](masse_delayed_cue_lif_task_spec.md)（任务时间合同、试次语义、行为门槛、结果目录习惯）。本文件不重开那些决定。

## Problem Statement

Tracer 4 的 formal 循环 LIF 已经完成 delayed-cue DMS+DMRS（测试试次准确率 1.0，时间点准确率约 94%），但该网**没有**显式 STSP。它用两条与 STSP 同时标的慢状态解题：独立突触电流 \(\tau=800\) ms，以及 25% SFA-LIF（\(\tau=400\) ms）。srnn-2afc 的突触电流是 35–40 ms；800 ms 不是那条参考实现的默认值，而是已经占了 Masse 静默突触记忆的功能位。

因此不能从“formal 无 STSP 仍能做任务”推出“任务不需要 STSP”，也不能在保留 800 ms 电流的同一底盘上把 STSP 加进去再声称 \(x\cdot u\) 是静默承载。需要的对照是：两边都拿掉这些占位慢变量，唯一结构差是循环边上有没有脉冲触发 STSP。

Masse 等（2019）把 Mongillo STSP 加在 **rate RNN** 的循环边上（按突触前细胞共享 \(x,u\)，输入和读出不加），并用发放惩罚逼出 activity-silent 维持。无 STSP 的 rate 网更难训。本阶段把该加法接到已有 Masse-LIF 任务上，但必须用**脉冲触发**更新，不能把 0/1 脉冲直接代入 Masse 的 \(dt\cdot r\) 欧拉式（2 ms 步长下一次脉冲几乎推不动 \(x,u\)）。

Tiddia / `recurrent_stsp` 的逐边 Tsodyks3 保持只读参考，不进入本阶段 BPTT。

## Solution

在 `src/experiments/masse_delayed_cue_lif/` 内增加配对 profile，不新建实验线，不覆盖 `results/masse_delayed_cue_lif/formal/`。

三套网络职责固定：

| 网络 | 路径角色 | 细胞 | 慢状态 | STSP | 本阶段动作 |
|---|---|---|---|---|---|
| 历史 formal | `results/masse_delayed_cue_lif/formal/` | 375 LIF + 125 SFA | 800 ms 电流 + SFA | 无 | **冻结，不重训**；只证明慢电流路线能做对任务 |
| 配对无 STSP | `results/masse_delayed_cue_lif/stripped_no_stsp/` | 500 普通 LIF | 无独立电流、无 SFA | 无 | 从初始化训练；达不到 90% 为合法阴性 |
| 配对有 STSP | `results/masse_delayed_cue_lif/stripped_stsp/` | 500 普通 LIF | 无独立电流、无 SFA | 循环边、按细胞共享 | 从初始化训练；必须过原行为门槛 |

两边配对网使用同一试次表、同一损失、同一发放 L2、同一训练预算。有 STSP 的网在行为评价之后增加只读机制节点：延迟最后 100 ms 从发放和 \(x\cdot u\) 线性解码样本方向，并打乱 \(x\cdot u\) 后测任务正确率。绘图只读已有产物。

## User Stories

1. As a 工作记忆模型研究者, I want 一对除 STSP 外匹配的循环 LIF, so that 我可以把静默承载的差异归因到 STSP 而不是慢电流或 SFA。
2. As a 工作记忆模型研究者, I want 冻结已有 formal 检查点, so that 慢电流解题路线作为历史存在证明被保留，且不被重跑覆盖。
3. As a 工作记忆模型研究者, I want 无 STSP 配对网失败时仍被报告, so that “去掉占位慢变量后任务更依赖 STSP”可以成为合法结论而不是工程事故。
4. As a 工作记忆模型研究者, I want 在延迟末分别从发放和突触效力解码样本, so that 对照符合 Masse 2019 的最小机制合同。
5. As a 工作记忆模型研究者, I want 打乱 \(x\cdot u\) 后看到任务正确率变化, so that 突触效力不只是与样本相关，而是被行为使用。
6. As a 工作记忆模型研究者, I want DMS 与 DMRS90 分开报告解码和打乱, so that 维持与操作不被混成一个静默故事。
7. As a 模型开发者, I want STSP 按脉冲触发、按突触前细胞共享, so that 量纲与 Mongillo/Tsodyks 发放事件一致，且 BPTT 成本仍是 \(O(N)\) 而不是逐边 \(O(N^2)\)。
8. As a 模型开发者, I want 两边相同的发放 L2, so that 静默倾向不被单侧惩罚所解释。
9. As a 模型开发者, I want 无 STSP 循环权重在谱半径 0.95 后再除以 3, so that 静止有效增益与 Masse 对照构造一致。
10. As a 后续 Agent, I want 明确的 DAG、目录和成功/失败口径, so that 实施时不重开设计树。

## Implementation Decisions

1. **科学问题**  
   本阶段回答的是机制对照，不是“formal 网必须加 STSP 才能完成任务”。formal 已经完成任务。配对实验问：在没有 800 ms 电流和 SFA 时，显式 STSP 是否可训，以及延迟末信息是否可从 \(x\cdot u\) 读出。

2. **底盘**  
   保留 Masse-LIF 任务、500 单元、无 Dale、无自连接、30 维模拟电流输入、三分类读出、SuperSpike、BPTT、Adam \(3\times 10^{-4}\)、`dt=2 ms`。不改成 Masse 原版 100 单元 rate 网，不接入 Tiddia 10k 稀疏图。

3. **删除占位慢变量**  
   配对网删除独立突触电流状态：每步 `drive` 直接进入膜电位，只保留 \(\tau_{\mathrm{mem}}=20\) ms。删除 SFA：100% 普通 LIF。读出 20 ms 低通和不应期保留，它们不是工作记忆静默底物。

4. **STSP 放置与参数**  
   只调制循环边。输入权重和读出权重不加 STSP。  
   按突触前细胞共享 \(x,u\)（Masse 的计算效率约定，不是逐边）。细胞下标偶数 facilitation、奇数 depression：  
   - facilitation：\(U=0.15\)，\(\tau_x=200\) ms，\(\tau_u=1500\) ms  
   - depression：\(U=0.45\)，\(\tau_x=1500\) ms，\(\tau_u=200\) ms  
   脉冲触发更新：发放时 \(u \leftarrow u + U(1-u)\)，释放 \(u x\)，\(x \leftarrow x(1-u)\)；发放间 \(u\) 回到 \(U\)、\(x\) 回到 1 的指数恢复。  
   有效循环驱动为 \(W\,x\,u\,s\)。  
   \(x,u\) 留在代理梯度图中，不对 STSP 状态做 stop-gradient。  
   循环脉冲驱动与 formal 相同：`rec_in` 使用 `spikes.detach()`，STSP 状态仍用未截断的脉冲更新。禁止把 SuperSpike Jacobian 沿循环边连乘 1250 步。  
   梯度裁剪按 Masse 官方逐参数 \(\ell_2\) 阈值 0.1，不用全体参数一次全局范数 5。非有限梯度跳过该 batch 的 `optimizer.step()`。  
   禁止：把二进制脉冲当作 Masse 代码里的连续 \(h\) 再乘 `dt_sec`；禁止先把脉冲滤成率再套一层慢滤波。

5. **初始化**  
   两边循环权重先按现有方式缩到谱半径 0.95。无 STSP 侧再将循环权重除以 3。有 STSP 侧不除，静止有效增益为 \(W\cdot U\)。不从 formal 检查点热启动。种子与前置规格一致：`trial_table=0`，`model_init=1`，`train_order=2`；本阶段 1 个 seed。

6. **损失**  
   任务交叉熵、测试窗权重 2、50 ms 宽限期权重 0，与前置规格相同。两边加**同一**发放 L2（对逐步脉冲，系数先取 Masse `spike_cost=2\times 10^{-2}`；smoke 上调节到既不静默塌缩也能下降损失）。禁止只给 STSP 网加惩罚。

7. **行为门槛**  
   有 STSP 网：总体、DMS、DMRS90 时间点准确率 ≥ 90%；四个规则×匹配条件试次准确率 ≥ 85%。  
   无 STSP 网：使用同一指标文件格式；达不到门槛时 `summary` 标记为合法阴性，禁止加回电流、SFA 或增大网络来迫使它过线。

8. **数据**  
   优先复用 formal 的持久化试次表（同一 `trial_id` 与输入 seed）。若必须重建，使用同一生成器与 `trial_table` seed，不得改 8 方向、规则窗或均衡约束。划分仍为 1024 / 256 / 256，batch 64，最多 100 epoch，验证集选检查点。

9. **DAG**  

   ```text
   试次表（复用 formal 或按原生成器重建）
             ↓
   stripped_no_stsp 训练 → best.pt
             ↓
   stripped_no_stsp 行为评价
             ↓
   stripped_no_stsp 机制重放（仅发放）
             ↓
   stripped_no_stsp plot-only

   同一试次表
             ↓
   stripped_stsp 训练 → best.pt
             ↓
   stripped_stsp 行为评价
             ↓
   stripped_stsp 机制重放（发放 + x,u）
             ↓
   stripped_stsp plot-only
   ```

   缺少上游产物时显式失败，不自动重训。plot-only 不得加载训练生成器做新采样，不得重跑仿真。机制节点只读取冻结检查点与测试试次，写出解码与打乱指标；不更新权重。

10. **机制合同**  
    在测试集、延迟最后 100 ms（1500–2000 ms 延迟段的末 100 ms，即规则后延迟结束前；与 Masse “delay 末 100 ms”对齐，对应本任务测试开始前 1900–2000 ms）上：  
    - 线性多类解码样本方向（8 类），特征为 500 维发放或 500 维 \(x\cdot u\)；解码器只在该时间窗的训练折上拟合的约定：用测试试次做时间点解码时，按 Masse 习惯在同一时间点划分 train/test fold，不得用任务训练损失泄漏标签。实施时固定为：仅使用机制节点自己的 held-out split，与任务训练检查点无关。  
    - 有 STSP 网：报告发放解码准确率、\(x\cdot u\) 解码准确率，以及打乱 \(x\cdot u\) 后的任务时间点/试次准确率。  
    - 无 STSP 网：只报告发放解码。  
    - 全部指标分总体、DMS、DMRS90。  
    不做 EXC/INH 分组、tuning similarity index 或 Masse 全套补图（没有 Dale 定律）。

11. **代码与结果落点**  
    包：`src/experiments/masse_delayed_cue_lif/`。  
    入口仍为 `python -m src.experiments.masse_delayed_cue_lif.run {build-trials,train,evaluate,plot}`，增加机制子命令或 `evaluate --decode`，实施时不得把机制计算塞进 plot。  
    新 profile：`stripped_no_stsp`、`stripped_stsp`；smoke 对应缩小单元与试次数，但必须仍无电流状态、无 SFA，STSP 侧仍是脉冲触发按细胞共享。  
    结果目录：  
    `results/masse_delayed_cue_lif/stripped_no_stsp/`  
    `results/masse_delayed_cue_lif/stripped_stsp/`  
    不覆盖 `formal/` 或 `smoke/`。不进入 13 个主线 catalog，不修改 `recurrent_stsp` 或论文图包。

12. **实施顺序**  
    - 先改模型：删除电流状态与 SFA；实现脉冲触发按细胞 STSP；两边同一发放 L2 与权重 ÷3。  
    - smoke DAG（含机制节点写文件、plot-only 不改哈希）。  
    - 单 seed 正式配对训练、行为评价、机制重放、plot。  
    - 多 seed（Masse 的 20 网络）不在本阶段。

## Testing Decisions

1. 任务生成器测试不重写；复用前置规格的 32 条件语义测试。  
2. 配对模型：无 STSP 前向无 \(x,u\) 状态；有 STSP 的 \(x,u\) 形状为 batch×隐藏，不出现边维。单脉冲后 facilitation 细胞的 \(u\) 上升、\(x\) 下降，量级为 \(O(U)\) 而非 \(O(dt\cdot U)\)。  
3. 梯度：循环权重有有限非零梯度。无 STSP 时循环脉冲叶子对后续读出无梯度；有 STSP 时该叶子经 \(x,u\) 仍有梯度。逐参数裁剪后，循环梯度爆炸不得把读出梯度压成 0；非有限总范数判定为不可用。  
4. 无电流状态：模型 state 不含独立 `current`；无 SFA mask。  
5. smoke：CUDA 上两条 profile 都能写出检查点、行为指标和机制 JSON；plot-only 不改变检查点与预测哈希。不要求 90%。  
6. 正式：有 STSP 检查行为门槛；无 STSP 无论是否过线都必须留下完整指标与机制（发放）产物。机制指标必须能从保存的延迟末特征重新计算。

## Execution Contract

本段在实施授权后把规格落到 CLI。

```text
python -m src.experiments.masse_delayed_cue_lif.run build-trials --profile stripped_no_stsp --output-directory results/masse_delayed_cue_lif/stripped_no_stsp --reuse-trials results/masse_delayed_cue_lif/formal/data/trials.csv
python -m src.experiments.masse_delayed_cue_lif.run train --profile stripped_no_stsp --output-directory results/masse_delayed_cue_lif/stripped_no_stsp
python -m src.experiments.masse_delayed_cue_lif.run evaluate --profile stripped_no_stsp --output-directory results/masse_delayed_cue_lif/stripped_no_stsp
python -m src.experiments.masse_delayed_cue_lif.run decode --profile stripped_no_stsp --output-directory results/masse_delayed_cue_lif/stripped_no_stsp
python -m src.experiments.masse_delayed_cue_lif.run plot --profile stripped_no_stsp --output-directory results/masse_delayed_cue_lif/stripped_no_stsp

python -m src.experiments.masse_delayed_cue_lif.run build-trials --profile stripped_stsp --output-directory results/masse_delayed_cue_lif/stripped_stsp --reuse-trials results/masse_delayed_cue_lif/formal/data/trials.csv
python -m src.experiments.masse_delayed_cue_lif.run train --profile stripped_stsp --output-directory results/masse_delayed_cue_lif/stripped_stsp
python -m src.experiments.masse_delayed_cue_lif.run evaluate --profile stripped_stsp --output-directory results/masse_delayed_cue_lif/stripped_stsp
python -m src.experiments.masse_delayed_cue_lif.run decode --profile stripped_stsp --output-directory results/masse_delayed_cue_lif/stripped_stsp
python -m src.experiments.masse_delayed_cue_lif.run plot --profile stripped_stsp --output-directory results/masse_delayed_cue_lif/stripped_stsp
```

`--device` 默认 `cuda`。试次表默认复用 `results/masse_delayed_cue_lif/formal/data/trials.csv`；若该文件不在工作区，用 `build-trials --profile formal` 的同一生成合同重建到配对目录，不得改语义。

## Out of Scope

- 不覆盖、不重训、不热启动 `results/masse_delayed_cue_lif/formal/`。  
- 不把 STSP 接到 Tiddia 逐边 Tsodyks3，不对 25 万边状态做 BPTT。  
- 不引入 Dale 定律、e-prop、BrainTrace 或局部学习。  
- 不在本阶段做 20 seed 科学统计。  
- 不把 smoke 或单 seed 写成 Masse 式多网络结论。  
- 不修改现有论文正文、主图、补图或 Tiddia 结果 bundle。

## Further Notes

- Masse 等（2019）：<https://doi.org/10.1038/s41593-019-0414-3>；官方代码 <https://github.com/nmasse/Short-term-plasticity-RNN>（`synapse_config='full'`，rate 欧拉；本规格用其 \(U,\tau\) 与按细胞共享，不用其 \(dt\cdot h\) 脉冲代入）。  
- Kozachkov 等（2022）表明 rate 网有/无 STSP 都能做工作记忆任务，差异在脑似性与鲁棒性；不能用来主张“脉冲 LIF 去慢状态后无 STSP 也一定能训成”。  
- 设计确认日期：2026-08-25。实施须另一次明确授权。  
- 2026-08-25 单 seed 正式配对已跑完（循环脉冲 `detach` + 逐参数 clip 0.1）。STSP 与 no-STSP 测试试次准确率均为 0.492，未过 90%/85% 门；无 STSP 已标 `legal_negative`。STSP `last.pt` 谱半径 0.94→9.24。延迟末 \(x\cdot u\) 解码 1.0 在未训练初始化上同样出现，不能当成学会了记忆。产物在 gitignored 的 `results/masse_delayed_cue_lif/stripped_stsp/` 与 `stripped_no_stsp/`；`formal/` 未覆盖。
