# Masse 延迟规则线索工作记忆 LIF 网络实施规格

**状态：** TRACER_4_COMPLETE  
**核心决定：** 使用可训练循环 LIF/SFA-LIF 网络完成 Masse 等（2019）的 delayed-cue DMS+DMRS 任务；网络实现参考 `srnn-2afc`，任务语义以 Masse 论文和官方任务生成器为准。  
**交付目标：** 得到可训练、可保存、可重新加载并在独立测试集上达到明确准确率门槛的任务模型，而不是扩展当前固定连接 Tiddia 网络或研究生物合理学习规则。

## Problem Statement

当前项目已经拥有适合研究逐突触 STSP 状态和因果干预的固定连接 Tiddia 循环 LIF 网络，但它没有端到端的监督任务、训练损失和经过训练的行为读出，因此不能直接声称能够完成规则依赖的工作记忆任务。

本阶段需要解决的问题不是学习规则是否具有生物合理性，也不是如何把 e-prop、BrainTrace、Tiddia 和其他模型组合起来，而是建立一个现实、可运行、可复现的循环脉冲网络任务基线。该基线必须使用 LIF 系列循环神经元，能够通过常规代理梯度和 BPTT 学会明确的工作记忆任务，并为后续对网络活动、状态和任务条件进行分析提供稳定检查点。

实验范式不得从 `srnn-2afc` 的视听选择任务照搬，也不得重新发明四方向直接回忆任务。正式范式采用 Masse 等（2019）已经发表并提供官方代码的 delayed-cue DMS+DMRS：网络先接收并维持样本方向，在延迟中段才获知应采用原方向匹配还是顺时针旋转 90° 匹配，最后对测试方向输出匹配或不匹配。

## Solution

实现一个独立的、可训练的循环 LIF 工作记忆实验工作流，其最终行为目标为 delayed-cue DMS+DMRS。

任务使用 8 个等间隔运动方向。每个试次先呈现样本方向，随后进入延迟；规则线索在延迟开始 500 ms 后出现并持续 250 ms，指示本试次采用 DMS 或 DMRS90 规则；延迟结束后呈现测试方向。DMS 的匹配方向等于样本方向，DMRS90 的匹配方向等于样本顺时针旋转 90° 后的方向。网络输出固定注视、匹配或不匹配。

网络实现采用普通 PyTorch，不依赖旧版 TensorFlow 或 Norse 0.03 运行时。默认正式模型包含 500 个循环单元，其中 75% 为普通 LIF、25% 为带阈值适应的 SFA-LIF；训练采用 SuperSpike 类代理梯度、BPTT、Adam 和交叉熵。生物约束不是本阶段目标，因此不要求 Dale 定律、局部学习、在线学习或 STSP。

完整工作流按项目 DAG 规则拆为：

```text
持久化任务配置与试次表
          ↓
训练并选择最佳检查点
          ↓
冻结检查点上的独立测试评价与记录
          ↓
只读取已有结果的绘图
```

最低可交付结果包括任务配置、训练/验证/测试试次身份、最佳检查点、训练历史、逐试次测试预测、分条件指标、运行元数据、artifact manifest 和 plot-only 图。正式检查点重新加载后必须复现保存时的测试预测和指标。

## User Stories

1. As a 工作记忆模型研究者, I want 一个已经训练成功的循环 LIF 网络, so that 我可以先确认网络确实能够完成行为任务。
2. As a 工作记忆模型研究者, I want 使用 Masse 2019 已发表的实验范式, so that 任务定义具有明确文献依据而不是临时设计。
3. As a 工作记忆模型研究者, I want 在同一个网络中随机交替 DMS 和 DMRS90 试次, so that 我可以比较维持条件与规则操作条件。
4. As a 工作记忆模型研究者, I want 规则在线索出现前先保留样本 500 ms, so that 网络不能在样本编码时提前知道应采用哪条规则。
5. As a 工作记忆模型研究者, I want 规则线索只在延迟中段短暂出现, so that 任务遵循论文 delayed-cue 范式。
6. As a 工作记忆模型研究者, I want 样本、规则和匹配状态保持均衡, so that 总体准确率不会掩盖条件不平衡。
7. As a 工作记忆模型研究者, I want DMS 和 DMRS90 分别报告准确率, so that 网络不能只学会其中一种规则。
8. As a 工作记忆模型研究者, I want 匹配和不匹配分别报告准确率, so that 网络不能通过恒定输出获得虚假成功。
9. As a 工作记忆模型研究者, I want 使用独立测试噪声和试次身份, so that 报告的任务性能不是训练样本记忆。
10. As a 模型开发者, I want 任务规则由一个确定性生成器实现, so that 相同配置和 seed 能产生完全相同的试次表。
11. As a 模型开发者, I want 任务生成器显式保存样本、规则、匹配状态和测试方向, so that 每个标签都可以被独立审计。
12. As a 模型开发者, I want 输入方向沿用论文的圆周方向编码, so that 任务几何与 Masse 2019 保持一致。
13. As a 模型开发者, I want 网络直接使用项目现有 PyTorch 环境, so that 不需要引入 TensorFlow 1 或 Norse 0.03。
14. As a 模型开发者, I want 使用常规代理梯度和 BPTT, so that 优先保证任务可训练性而不是学习规则的生物合理性。
15. As a 模型开发者, I want 提供小规模 smoke 配置, so that CUDA 上可以快速验证完整工作流和 artifact 契约。
16. As a 模型开发者, I want 提供正式 500 单元配置, so that 最终模型与参考 SRNN 的容量处于同一量级。
17. As a 模型开发者, I want 按验证集表现选择最佳检查点, so that 测试集不参与模型选择。
18. As a 模型开发者, I want 检查点包含模型、优化器、epoch 和配置身份, so that 中断训练后可以继续，并能追溯模型来源。
19. As a 模型开发者, I want 训练失败时区分数据语义错误、梯度错误和容量不足, so that 不通过盲目增加网络规模来掩盖问题。
20. As a 结果审阅者, I want 看到总体、分规则和分标签的测试指标, so that 我可以判断模型是否真正掌握完整任务。
21. As a 结果审阅者, I want 看到逐试次预测记录, so that 汇总指标可以从原始预测重新计算。
22. As a 结果审阅者, I want 最佳检查点重新加载后得到相同预测, so that 模型交付不是依赖训练进程中的隐式状态。
23. As a 结果审阅者, I want 绘图任务只消费已有训练和评价产物, so that 重绘不会重新训练模型或改变结果。
24. As a 后续 Agent, I want 一个明确的任务、模型、训练、评价和输出合同, so that 我可以直接实施而不重新讨论研究方向。
25. As a 后续 Agent, I want 明确知道哪些内容不属于本阶段, so that 我不会擅自加入 STSP、BrainTrace、e-prop 或 Tiddia 连接迁移。
26. As a 项目维护者, I want 新工作流沿用现有结果目录、manifest 和 run_info 习惯, so that 新实验能够进入当前实验工程体系。
27. As a 项目维护者, I want 训练、评价和绘图声明显式依赖关系, so that 工作流满足项目的 artifact DAG 约束。
28. As a 项目维护者, I want 正式结果与 smoke 结果明确分离, so that 结构验证不会被误认为科学性能证据。

## Implementation Decisions

1. **任务事实来源**
   - 行为范式以 Masse 等（2019）论文及其官方 `Short-term-plasticity-RNN` 任务生成逻辑为准。
   - 循环脉冲网络、SFA-LIF 和代理梯度训练方式参考 `srnn-2afc`。
   - 外部仓库是方法参考，不作为运行时依赖；如实际复制代码，必须保留许可证、来源 URL 和固定提交身份。

2. **正式任务固定为 delayed-cue DMS+DMRS**
   - 运动方向共 8 个：0°、45°、90°、135°、180°、225°、270°、315°。
   - 规则共 2 个：DMS 和 DMRS90。
   - DMS 匹配方向为样本本身。
   - DMRS90 匹配方向为样本顺时针旋转 90°，即方向索引加 2 后模 8。
   - 匹配试次的测试方向等于规则定义的匹配方向。
   - 不匹配试次的测试方向从其余 7 个方向中均匀抽取。
   - 规则、匹配状态和样本方向在每个数据划分内均衡。
   - 输入模式重叠保持为固定编码属性，不作为实验变量。

3. **试次时间合同**

   | 绝对时间 | 阶段 | 输入与目标 |
   |---|---|---|
   | 0–500 ms | 固定注视 | 无运动刺激；输出固定注视 |
   | 500–1000 ms | 样本 | 呈现一个运动方向；输出固定注视 |
   | 1000–1500 ms | 延迟前半段 | 不提供规则；输出固定注视 |
   | 1500–1750 ms | 规则线索 | 呈现 DMS 或 DMRS90 规则；输出固定注视 |
   | 1750–2000 ms | 延迟后半段 | 规则线索消失；输出固定注视 |
   | 2000–2500 ms | 测试 | 呈现测试方向；输出匹配或不匹配 |

   - 测试开始后的前 50 ms 是决策宽限期，不计入损失或准确率。
   - 物理时间遵循论文；数值积分默认采用 2 ms 时间步，以适配循环 LIF 训练。

4. **输入表示**
   - 使用 24 个运动方向调谐输入通道和 6 个规则输入通道。
   - 运动通道使用圆周 von Mises 调谐，方向数、调谐宽度关系沿用论文；实现时将调谐值归一化为网络输入电流尺度。
   - 规则通道分为两组，每组 3 个通道，在偏好规则出现时为高电平，其他时间为零。
   - 第一版直接把 30 维时间序列作为循环 LIF 层的输入电流，不额外训练输入脉冲编码器；这不改变任务语义，并减少无关失败源。
   - smoke 和首个正式验收允许关闭输入噪声；噪声鲁棒性不是首个交付门槛。

5. **网络结构**
   - 默认正式网络包含 500 个全连接循环单元。
   - 375 个单元使用普通 LIF，125 个单元使用带自适应阈值的 SFA-LIF；两者都属于 LIF 系列。
   - 循环连接不包含自连接。
   - 不强制 Dale 定律或 E/I 权重符号，因为本阶段只要求完成任务。
   - 读出层产生固定注视、匹配和不匹配三个 logits。
   - smoke 配置缩小神经元数和数据量，但保持同一任务、同一接口和同一 artifact schema。

6. **训练方法**
   - 使用 SuperSpike 类代理导数处理脉冲阈值。
   - 使用完整 BPTT 和 Adam；不实现 e-prop、RTRL 或其他在线规则。
   - 使用时间点交叉熵：固定注视、样本和延迟阶段目标为固定注视，测试阶段目标为匹配或不匹配。
   - 测试阶段损失权重为其他有效时段的 2 倍；50 ms 决策宽限期权重为零。
   - 首发不使用 firing-rate、spike-count 或 L2 权重正则。
   - 使用验证集准确率选择最佳检查点；测试集只在检查点冻结后运行。
   - 训练支持早停和最大 epoch 上限，正式默认上限为 100 个 epoch。
   - 首先要求模型能过拟合一个包含全部 32 种基本条件的小数据集，再启动正式训练。

7. **数据划分与 seed**
   - 任务生成节点持久化紧凑试次表，而不是预先保存完整时间序列；每行至少包含 split、trial_id、sample_direction、rule、match、test_direction 和输入随机 seed。
   - 训练、验证和测试拥有不重叠的 trial_id 与输入 seed。
   - 每个划分内覆盖并均衡 8×2×2 共 32 种基本条件。
   - 网络初始化 seed、任务表 seed 和训练顺序 seed 分开记录。

8. **主要评价合同**
   - 主要指标是测试期排除前 50 ms 后的时间点分类准确率，与论文评价方式一致。
   - 同时计算每个试次对有效测试窗口 logits 取平均后的试次级准确率。
   - 必须报告总体、DMS、DMRS90、匹配、不匹配，以及规则×匹配状态四个交叉条件的准确率。
   - 正式成功门槛：总体、DMS 和 DMRS90 时间点准确率均不低于 90%；四个规则×匹配状态条件均不低于 85%。
   - 随机水平按匹配/不匹配二分类计，为 50%。
   - 固定注视准确率单独报告，但不用于替代主要任务指标。

9. **DAG 与 artifact 合同**
   - `build-trials` 只生成任务配置和试次表。
   - `train` 只消费任务配置与训练/验证试次，输出训练历史和最佳检查点。
   - `evaluate` 只消费冻结检查点与测试试次，输出逐试次预测和聚合指标。
   - `plot` 只消费训练历史、测试预测和指标，不加载训练数据生成器进行新采样，不重新训练或评价。
   - 所有节点在 manifest 中声明依赖和输出；缺少上游 artifact 时显式失败。
   - 结果 bundle 沿用项目通用结构。一次 run 目录的固定文件名为：`run_config.json`、`summary.json`、`artifact_manifest.json`、`meta/run_info.json`、`data/trials.csv`、`data/checkpoints/best.pt`、`data/checkpoints/last.pt`、`data/train_history.json`、`data/test_predictions.csv`、`metrics/test_metrics.json`，以及 `figures/training_curves.png`、`figures/condition_accuracy.png`、`figures/rule_match_confusion.png`、`figures/example_trial_timeline.png`。
   - `best.pt` 是验证准确率最好的检查点，`evaluate` 只读它；`last.pt` 每个 epoch 结束覆盖，供断点续训。两者都包含模型、优化器、epoch 和配置身份。

10. **实施顺序**
    - Tracer 1–3 一次交付后再停：确定性任务生成器与 32 条件语义测试；小网络前向、非零有限梯度和小数据过拟合；smoke DAG（试次表 → 可重载检查点 → 评价指标 → plot-only 图）。
    - Tracer 4 需要单独授权：500 单元正式训练并达到性能门槛。2026-08-24 已授权；正式结果写入 `results/masse_delayed_cue_lif/formal/`。
    - 在正式任务通过前，不启动 STSP、机制干预、多 seed 科学统计或 Tiddia 迁移。

11. **与当前 Tiddia 模型的边界**
    - 当前 Tiddia 稀疏 10k/20M 固定网络、NEST 等价内核、逐边 STSP 状态和既有论文图实验全部保持不变。
    - 新任务模型是独立的可训练实验，不读取或覆盖 Tiddia 连接产物和检查点。
    - 本阶段不声称新任务模型复现 Tiddia 动力学，也不声称 Tiddia 已经具备任务能力。

## Testing Decisions

1. **主测试缝隙**
   - 选用一个最高层的公开工作流缝隙：从已持久化试次表出发，完成 smoke 训练、检查点重载、测试评价和 plot-only 消费。
   - 该缝隙以 artifact 和指标为断言对象，不检查内部类名、私有状态或具体张量实现。
   - 这是主要集成验收；低层测试只覆盖无法通过该缝隙清楚定位的任务语义和梯度问题。

2. **任务生成器行为测试**
   - 每个划分覆盖全部 32 种基本条件且计数均衡。
   - DMS 匹配测试方向等于样本方向。
   - DMRS90 匹配测试方向等于样本索引加 2 后模 8。
   - 所有不匹配测试方向均不等于当前规则的匹配方向。
   - 样本、规则和测试只在规定时间窗出现。
   - 规则线索在 1500–1750 ms 出现，第一段 500 ms 延迟不泄漏规则。
   - 决策宽限期损失 mask 为零，其他目标和权重符合时间合同。
   - 相同 seed 产生相同试次表，不同 split 不共享 trial_id 或输入 seed。

3. **模型公共行为测试**
   - 给定合法批次时，模型输出形状为时间×批次×3。
   - 时间展开后损失有限，至少一个循环参数和输入参数获得有限非零梯度。
   - 重置状态后重复前向得到确定性结果；不重置状态时不得静默跨试次污染。
   - 保存并重新加载检查点后，对固定输入产生相同 logits 和预测。

4. **最小训练自检**
   - 小模型能够过拟合覆盖全部 32 个基本条件的固定小数据集。
   - 过拟合只作为 CUDA pytest：32 试次、32 个隐藏单元（8 个 SFA）、不写入 `results/`。断言损失有限且相对首个 epoch 下降，有效测试窗试次级训练准确率 ≥ 80%，检查点重载后 logits 一致。不要求 90%，也不看测试集。
   - 该检查用于证明任务标签、损失、时间 mask 和梯度链路闭合，不作为正式性能证据。
   - smoke 工作流只要求结构完整、损失有限且检查点可重载，不要求达到正式 90% 门槛。

5. **正式验收测试**
   - 冻结最佳验证检查点后，只运行一次正式测试评价。
   - 重新从逐试次预测计算汇总指标，并与保存的指标文件一致。
   - 检查总体、DMS、DMRS90 和四个规则×匹配状态门槛。
   - 修改绘图配置并重绘前后，检查点、测试预测和指标文件哈希保持不变。

6. **既有测试先例**
   - 沿用现有循环 STSP 工作流测试的做法：小配置、临时结果目录、规范化 artifact、显式成功条件和 plot-only 叶节点。
   - 沿用现有结果布局和 manifest 测试，而不为训练模型创建第二套结果规范。
   - 不修改或削弱现有 Tiddia、论文图和数据 lineage 测试。

## Execution Contract

本段记录 2026-08-24 已锁定的执行方式。它不重开科学选择，只把规格落到本仓库的包、CLI、剖面和文件名。

1. **代码落点**
   - 独立包 `src/experiments/masse_delayed_cue_lif/`。
   - 入口：`python -m src.experiments.masse_delayed_cue_lif.run {build-trials,train,evaluate,plot}`。
   - 配置为包内 dataclass 默认值 + argparse；`CLI > 代码默认`。不新建根目录 `configs/`，本阶段不加 YAML。
   - 不进入现有 13 个主线实验 catalog，不修改 `recurrent_stsp`、Tiddia 内核或论文图包。
   - 自写普通 PyTorch LIF + SFA-LIF + SuperSpike；不依赖 Norse，不复用 `BaseLIFLayer`。
   - Masse / srnn-2afc 仓库不入库、不当运行时。方程与任务语义按本规格和参考文献实现。
   - 训练与推理默认 `float32`。

2. **CLI 与种子**

```text
python -m src.experiments.masse_delayed_cue_lif.run build-trials --profile smoke --output-directory results/masse_delayed_cue_lif/smoke
python -m src.experiments.masse_delayed_cue_lif.run train          --profile smoke --output-directory results/masse_delayed_cue_lif/smoke
python -m src.experiments.masse_delayed_cue_lif.run evaluate       --profile smoke --output-directory results/masse_delayed_cue_lif/smoke
python -m src.experiments.masse_delayed_cue_lif.run plot           --profile smoke --output-directory results/masse_delayed_cue_lif/smoke
```

   - `--device` 默认 `cuda`。过拟合、smoke 和正式训练全部使用 CUDA。
   - 缺少上游产物时显式失败，不自动重跑或自动生成。
   - 种子：`trial_table=0`，`model_init=1`，`train_order=2`。
   - `--profile {smoke,formal}` 都存在于代码中；Tracer 1–3 只运行 smoke。Tracer 4 运行 `--profile formal`。

3. **数据剖面**

   | 剖面 | 单元 | 试次 train/val/test | batch | 训练 | 设备 | Tracer |
   |---|---|---|---|---|---|---|
   | 过拟合 | 32（8 SFA） | 固定 32 条（每条件 1 条） | 全 batch | 直到可过拟合 | CUDA | 1–3：仅 pytest |
   | smoke | 64（16 SFA） | 128 / 64 / 64 | 64 | 最多 5 epoch，不卡 90% | CUDA | 1–3：CLI + 测试 + `results/` bundle |
   | 正式 | 500（125 SFA） | 1024 / 256 / 256 | 64 | 最多 100 epoch，验证集选检查点 | CUDA | 4：CLI + `results/masse_delayed_cue_lif/formal/` |

   - 试次表持久化，按行在训练时展开；不预存完整时间序列，不在线无限采样。
   - 优化器为 Adam，学习率默认 `3e-4`（过拟合 pytest 可用更高学习率）；细胞时间常数参考 srnn-2afc：`dt=2 ms`，`tau_mem=20 ms`，`tau_sfa=400 ms`。

4. **Tracer 1–3 完成标准**
   - 任务生成器与 32 条件语义测试通过。
   - 小网 CUDA 过拟合 pytest 通过。
   - smoke DAG 在 CUDA 上从头跑通，pytest 绿，并在 `results/masse_delayed_cue_lif/smoke/` 留下真实 bundle。
   - 不要求 90% 正式门槛。达到本标准后停止，等待授权 Tracer 4。Tracer 4 已授权，正式门槛见 Testing Decision 5。

## Out of Scope

- 不把 STSP 加入新任务模型。
- 不训练、改造或迁移当前 Tiddia 固定连接网络。
- 不使用 BrainTrace、e-prop、D-RTRL、FORCE 或局部学习规则。
- 不要求学习过程具有生物合理性或在线性。
- 不要求纯普通 LIF；正式默认允许 25% SFA-LIF 以提高长延迟任务可训练性。
- 不复现 Masse 2019 关于活动静默、突触存储或因果机制的结论。
- 不把输入方向模式重叠作为科学变量。
- 不在首个交付中加入 DMRS45、DMRS180、逆时针 DMRS90、ABBA、ABCA 或 dualDMS。
- 不在首个交付中进行多 seed 科学统计、超参数大规模搜索或网络规模扫描。
- 不修改现有论文正文、主图、补图或当前结果 bundle。
- 不把 smoke 结果称为正式任务性能证据。

## Further Notes

- Masse 等（2019）论文：<https://doi.org/10.1038/s41593-019-0414-3>
- Masse 官方任务代码：<https://github.com/nmasse/Short-term-plasticity-RNN>
- SRNN 架构与训练参考：<https://github.com/Jakexxh/srnn-2afc>
- Masse 论文的主 DMS 与 DMRS90 比较分别训练任务网络；其 delayed-cue 变体让同一个网络在延迟中段接收规则。本规格选择后者作为最终任务，因为它在同一模型中同时包含维持与操作条件。
- 本规格中的“操作”严格指任务关系从样本同向匹配变为顺时针 90° 匹配；任务成功不自动证明网络内部形成了某种特定神经或突触机制。
- 如果正式模型未达到门槛，诊断顺序固定为：先验证 32 条件过拟合，再检查时间 mask 和标签，再检查梯度与状态重置，最后才调整 SFA 比例、隐藏单元数或优化参数。
- 去慢电流/SFA 后的有/无 STSP 匹配对照不在本规格范围内，见 [`masse_delayed_cue_lif_stsp_match_spec.md`](masse_delayed_cue_lif_stsp_match_spec.md)。本规格的 formal 检查点在该对照中只作历史基线，不得覆盖。
