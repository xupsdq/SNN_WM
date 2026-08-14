# Fig.2 面板与布局契约

## 当前权威修订（2026-08-10）

本节覆盖下方 2026-07-30 版五面板契约；下方内容仅保留为设计历史，
不再约束当前主图。完整逐项决定与实施记录见
`docs/archive/paper/figure-revision-notes/fig2_revision_working_notes.md` 的 F17–F24。

- 面板集合：a–d，共四个主图面板；
- 论证链：
  `a exact-B 配对反事实 → b 双向行为改写 → c 共同更新与历史残差并存 → d 事件关联`；
- a 采用一个收束式配对结构：
  `History A/C → no input (200 ms) → one identical B → paired post-B comparison`；
- 三个可见刺激使用 frozen protocol 中三个不同类别的真实 MNIST 图像：
  History A 使用 family 1 的 digit 1，History C 使用 family 1 的 digit 6，
  B 使用唯一 frozen anchor 30 的 digit 0；上下历史共同指向同一张 B 图像，
  不再重复画两个 B；
- no-input 阶段只显示 `No input` 与 `200 ms`，使用实线中性边界；禁止时间尺、
  无解释横虚线、端点刻度或装饰性 Delay 符号；
- B 后只定义两个下游比较对象：
  `Post-B STSP state` 与 `B-choice outcome`。状态使用 Tabler
  `hierarchy-3` 公共图标，行为使用 Tabler `target-arrow` 公共图标；二者下方
  统一标明 `A-history versus C-history`；
- a 不再使用括号点阵状态占位符、空心圆行为占位符或方向不明的 `A vs C`
  bracket；比较对象和比较条件必须在同一个语义区域内直接写明；
- a 只使用蓝色标识 A 历史、洋红色标识 C 历史；B 保持中性，状态和行为图标
  使用其跨图固定语义色；所有连接均为相邻语义单元之间的短箭头；
- b：Mismatched/Aligned × Rescue/Loss 的 `2 × 2` 分组柱状图；
- c：单坐标阈值中心化两柱图；横轴为两个端点，纵轴为各端点相对
  自身预设阈值的差值；
- d：`Matched random` 与 `Changed events` 的 network-level 抖动点和均值标记；
- 原 e 从主图、布局和 panel QA 移除，但其持久化数据、统计和来源保留；
- 画布：`165 mm × 102 mm`；
- slot：
  - a：`[2.000, 2.000, 161.000, 48.000] mm`；
  - b：`[2.000, 52.000, 52.333, 48.000] mm`；
  - c：`[56.333, 52.000, 52.334, 48.000] mm`；
  - d：`[110.667, 52.000, 52.333, 48.000] mm`；
- 科学边界：20 个独立 network（1000–1019）、`prefix_k=1`，
  plot-only 读取既有 final bundle，不新增训练、模拟或 forward replay。

当前状态：`paired_post_b_comparison_validated_plot_ready`。

## 历史状态（2026-07-30 版，已被上节覆盖）

- 科学逻辑：已冻结。
- 面板集合：已冻结为 a–e。
- 布局拓扑：已冻结为与 Fig.1 相同的三排结构；第一排 a，第二排 b–c，第三排 d–e。
- 网络口径：全部使用 `seed_1000`–`seed_1019`，共 20 个独立训练网络。
- 历史深度：主图只显示一步历史，即 `prefix_k = 1`。
- K5：明确排除出 Fig.2，保留给后续递归性主图。
- 数据边界：只读取已有 DMS 与 fixed-B 持久化结果；不增加训练、模拟或实验。
- 实现状态：本文件是 Fig.2 的科学权威；现有 `fig2.yaml`、适配器和渲染器尚未按本契约迁移。
- 冻结日期：2026-07-30。

## 1. Fig.1 到 Fig.2 的唯一推进

Fig.1 已经建立一个静默、内容特异且能够影响后续处理的可继承 STSP 状态。Fig.2 不再重复证明“状态存在”，而是追问：

当一个新输入 B 到达时，继承状态是否会改变 B 的行为结果，并在 B 的处理中被写成下一层的新状态？

Fig.2 必须从行为现象走向状态分解，再走向事件落点和跨层因果写入。它的终点是“一次完整状态转移成立”，不是多次输入已经形成循环。

## 2. 整图唯一问题

在最小的一步 DMS 情形中，相同的当前输入 B 面对不同继承历史时，是否产生可见的双向行为改变，并形成由 B 主导、受历史条件化的跨层 successor update？

## 3. 整图允许形成的结论

一步历史并非单向提高准确率，而是会把原本错误的 B 挽救为正确，也会把原本正确的 B 改写为错误；这种双向改变取决于历史类别与 B 是否一致。对完全相同的 B，处理更新仍具有高度一致的共同方向，但同时保留稳定的历史残差 `Γ`。该残差富集于真实 spike-event 改变，而且只交换 L1 的继承 `u/x` 状态，就能把供体历史转移到 L2 update，并进一步转移到早期分类输出。

因此，B 不是被动读取一个固定记忆，也不是覆盖旧状态；B 在继承状态上被处理，并将“当前输入＋历史条件”共同写成下一层的 successor state。

## 4. 必要论证链

`a 定义一步 DMS 与 exact-B 方向 → b 历史确实双向改变 B 的行为 → c 相同 B 同时产生共同更新与历史残差 → d 历史残差落到真实事件改变 → e 继承状态因果写入下一层并到达早期输出`

五个面板分别封闭一个不可跳过的逻辑缺口：

- 没有 a，读者不知道“历史—当前输入—下一层状态”的方向，也无法理解 exact-B 反事实。
- 没有 b，后续状态几何仍缺少可见的行为后果。
- 没有 c，不能同时确立 B 的主导作用与历史的条件化作用。
- 没有 d，`Γ` 仍可能只是抽象状态空间中的数值残差。
- 没有 e，历史残差仍是相关性，不能证明继承状态被因果写入下游。

## 5. 面板契约

### a. 以 DMS 为母体的一步状态转移示意

**角色**：定义全图的输入方向和最小反事实。

**权威底图**：

`results/paper_figures/outputs/DMS-enhanced.svg`

原图的时间骨架保留为：

`Sample 200 ms → Delay 400 ms → Probe 200 ms`

在其下方将任务含义明确改写为：

`History A/C → inherited L1 u/x → same B → L1 processing → L2 successor update → early output`

**必要元素**：

- 两个不同的一步历史 A 与 C；
- 两个由历史形成的 L1 inherited `u/x` state；
- 一个共享的 exact B，或两个有明确 identical 标记的 B；
- 从 L1 processing 指向 L2 successor update 的跨层箭头；
- 从 L2 update 指向 early output 的短读出箭头。

**禁止元素**：

- `K1`、`K5`、多输入省略号、连续阶段重复箭头；
- 数值结果、显著性、`Γ` 大小或 donor-transfer 数值；
- 将两个 B 画成不同图像、颜色或编码；
- 横向拉伸原 SVG，或把可编辑矢量栅格化。

a 只建立方向，不提前承担任何结果证明。

### b. 一步历史对相同 B 的双向行为改写

**角色**：证明继承历史不只是改变内部状态，而是确实改变相同 B 的分类结局；同时避免把记忆效果误写成单向准确率提升。

**比较定义**：

- `S0`：B 在无前序历史状态下的分类结果；
- `history`：在一步 A 或 C 历史之后处理完全相同 B 的分类结果；
- `rescue`：`S0` 错误而 history 条件正确；
- `loss`：`S0` 正确而 history 条件错误。

**类别划分**：

- `aligned history`：一步历史的类别与 B 类别一致；
- `mismatched history`：一步历史的类别与 B 类别不一致。

这两个关系类别是 b 的必要分组，因为它们直接检验历史内容是否定向改变 B。数字 0–9 不进入主面板：已有 20 网络中，数字 1 和 6 没有 `S0-error` 挽救机会，其他多个数字也只有零至两个 eligible anchors/网络；逐数字主图会把机会缺失误画成类别效应。0–9 覆盖仅保留为 Source Data 审计。

**机会分母**：

- 挽救率使用 `S0-error anchors` 作为分母；
- 损失率使用 `S0-correct anchors` 作为分母；
- 两种率不得相连、相减或画在同一个配对轨迹中。

**首选编码**：

- b 内部使用上下两个独立的小坐标区；
- 上区显示 rescue：mismatched 与 aligned 的 20 个网络级点及配对变化；
- 下区显示 loss：mismatched 与 aligned 的 20 个网络级点及配对变化；
- 两区各自明确写出机会条件 `S0 wrong` 与 `S0 correct`；
- 共享 0–1 比例范围可以方便阅读，但不得暗示两个分母相同。

**已有结果方向的全 20 网络复核**：

- rescue：aligned 相对 mismatched 增加约 14.08 个百分点；
- loss：aligned 相对 mismatched 减少约 5.07 个百分点。

这两个方向共同支持“类别一致的近期历史定向改变相同 B 的分类结局”，不能写成“历史普遍提高准确率”或“历史定义了正确答案”。

**持久化来源**：

- `results/multi_seed_rollout/fig2/fixed_b_mechanism_confirmatory/seed_*/data/intermediates/fixed_b_rollout_bank/rollout_rows.csv`
- `results/multi_seed_rollout/fig2/fixed_b_mechanism_confirmatory/seed_*/data/intermediates/fixed_b_history_bank/history_specs.csv`
- `results/paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory/seed_*/data/raw/fixed_b_state_trajectory_rows.csv`
- `results/paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory/seed_*/data/trial_specs/fixed_b_history_specs.csv`

**固定过滤**：

`track == "stsp_isolated"`, `branch == "free"`, `prefix_k == 1`, networks `1000–1019`。

每个 `network_seed × b_anchor_id` 必须先核验十次重复保存的 `S0 prediction`、B label 和 exact-B tensor hash 完全一致，再把每个 anchor 在网络内汇总；history family、anchor 和 trial 都不是独立重复。

### c. 相同 B 的共同更新与历史残差

**角色**：在同一面板中同时建立 input-driven common component 与 history-conditioned residual。

**必要端点**：

- `same_B_common_update_cosine`；
- `processing_residual_gamma_energy_fraction`。

**解释顺序**：

1. common cosine 高，说明完全相同的 B 规定主要更新方向；
2. residual `Γ` 明确存在，说明继承历史仍改变 B 的处理结果。

**当前确认编码**：一张完整的 categorical x–y 两柱图，不拆分内部
子图。横轴为 `Common update` 与 `History residual`；纵轴为
`Value − threshold`。对每个 network 先用原始端点减去该端点的预设
阈值，再用两根柱显示 20-network 均值与 95% CI。柱从共同的 `y=0`
参照线起始，不显示 network 点、柱顶数字或图例。

**阈值**：

- common cosine 的条件阈值为 0.5；
- `Γ` 的预设最小效应阈值为 0.05；
- 纵轴零值只表示“原始端点恰好达到自身阈值”，不表示两个原始指标相等。

**禁止误读**：common cosine 与 `Γ` 是定义不同的互补端点，不能根据同轴高度直接比较二者“谁贡献更多”。

**持久化来源**：

`results/paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory/aggregate/fixed_b_confirmatory_network_scalars.csv`

**固定过滤**：`prefix_k == 1`，networks `1000–1019`。

### d. 历史残差富集于真实事件改变

**角色**：把 c 中的状态空间残差连接到 B 处理期间实际发生改变的 spike events。

**必要端点**：

`full_trace_event_gamma_enrichment`

**当前确认编码**：横轴直接比较 `Matched random` 与 `Changed events`，
纵轴为 `Residual magnitude`；两根柱显示 20-network 均值与 95% CI。
network seed 不作为横轴，也不叠加 cell、coordinate、trial 或 event
级伪重复点。

**限定解释**：d 只证明 history-sensitive residual 与真实事件改变相连，不在本图解释哪些局部 STSP–input overlap 产生这些事件，也不展开 advance、recruit 和 loss 等事件类型；这些属于下一张局部机制图。

**持久化来源**：

`results/paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory/aggregate/fixed_b_confirmatory_network_scalars.csv`

**固定过滤**：`prefix_k == 1`，networks `1000–1019`。

### e. L1-only 状态交换产生下游供体转移

**角色**：提供一次跨层 successor 写入的核心因果证据及其早期功能后果。

**必要端点**：

- `layer1_only_layer2_update_donor_transfer`；
- `layer1_only_early_class_score_donor_transfer`。

**解释顺序**：

1. 只交换 L1 inherited `u/x`，供体方向进入 L2 update；
2. 同一供体方向进一步进入 early class score。

**首选编码**：同一 donor-transfer index 轴上的两行网络级森林图，各显示 20 个网络点并保留零参照。

**明确排除**：

- `all_layers` transfer 只是工程 plumbing control，不进入主面板；
- `b_end_class_score`、电压、drive 和 cell-level swap 分析不进入主面板；
- e 不增加第三个端点重复 c 或 d。

**持久化来源**：

`results/paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory/aggregate/fixed_b_confirmatory_network_scalars.csv`

**固定过滤**：`prefix_k == 1`, `swap_scope == "layer1_only"`，networks `1000–1019`。

## 6. K5 的明确归宿

`prefix_k = 5` 不是 Fig.2 的稳健性装饰，而是累积历史下检验转移原则能否延续的递归性证据。

因此：

- Fig.2 的 b–e 必须显式过滤 `prefix_k == 1`；
- Fig.2 图面不出现 K 图例，也不写 “K1 versus K5”；
- K5 的行为结果、common component、`Γ`、event enrichment、L1→L2 transfer 和 early-output transfer 全部留给后续递归性主图；
- 不在 Fig.2 用 K5 的零结果削弱或装饰一步转移结论。

## 7. 冻结布局

画布与 Fig.1 完全一致：`165 mm × 152 mm`，外边距 2 mm，横向 gutter 2 mm，排间距 2 mm。

| 排 | 面板 | 冻结槽位，单位 mm | 逻辑任务 |
|---|---|---|---|
| 第一排 | a | `x=2, y=2, w=161, h=48` | DMS 母体、exact-B 与跨层方向 |
| 第二排左 | b | `x=2, y=52, w=79.5, h=48` | 双向行为改写 |
| 第二排右 | c | `x=83.5, y=52, w=79.5, h=48` | 共同更新与历史残差 |
| 第三排左 | d | `x=2, y=102, w=79.5, h=48` | 事件层落点 |
| 第三排右 | e | `x=83.5, y=102, w=79.5, h=48` | 跨层因果写入 |

**版式原则**：

- 阅读顺序固定为 a → b → c → d → e。
- a 满宽，承担任务与方向；不为数值面板让位。
- b 与 c 同排，形成“行为事实 → 状态分解”的直接推进。
- d 与 e 同排，形成“事件关联 → 因果写入”的直接推进。
- b 的上下两个机会分母是一个面板内的两个逻辑区，不新增面板字母。
- c–e 使用短直接标签，避免重复图例。

## 8. 统计与证据边界

- 所有主图显示与汇总统一使用 20 个独立训练网络：`seed_1000`–`seed_1019`。
- network 是独立重复和推断单位；history family、anchor、trial、cell 与 coordinate 均先在网络内汇总。
- 不继承任何只保留部分网络的筛选口径。
- 若正式图需要置信区间或检验，必须从现有 20 个网络值一次性计算并固化；不得把旧的子集推断值混入图中。
- b 是既有结果的 secondary reanalysis，应保持“类别一致近期历史的定向调制”这一限定，不升级为历史定义目标的工作记忆任务。
- c–e 只显示预先存在的核心端点，不根据图面效果更换指标或阈值。

## 9. 与现有资产的迁移关系

| 当前资产 | 新归宿 |
|---|---|
| `DMS-enhanced.svg` | Fig.2a 的权威底图 |
| fixed-B K1 rescue/loss 与 aligned/mismatched history | Fig.2b |
| fixed-B K1 common cosine 与 `Γ` | Fig.2c |
| fixed-B K1 event–`Γ` enrichment | Fig.2d |
| fixed-B K1 L1-only donor transfer | Fig.2e |
| fixed-B 全部 K5 endpoints | 后续递归性主图 |
| all-layer plumbing controls 与 cell-level diagnostics | 补充材料或验证记录 |
| 旧 Fig.2 pair retention、pair specificity、ping 和 partial cue | 最终结构化表征图 |

旧 `src/plotting/paper_fig/specs/fig2.yaml` 已随 plotting 重写移除。当前 Fig.2 绘图规格由 `src/plotting/paper_fig/final_six/specs.py` 与 bundle 内 `meta/final_plot_spec.json` 固定；plot-only 迁移只能读取父级结果，不得重新运行模拟。

## 10. 最终验收标准

- Fig.2 只回答“一步输入如何读取并改写继承状态”。
- a 清楚表达 History → inherited state → same B → L2 successor 的方向。
- b 将 rescue 与 loss 分开，并使用各自机会分母；类别分组是 aligned 与 mismatched history。
- c 同时保留共同输入成分和历史残差，不能只展示其中一半。
- d 只承担 event–`Γ` 连接，不抢先展开局部 overlap 机制。
- e 同时保留 L2 写入与 early output 两个必要读出。
- b–e 只显示 `prefix_k = 1`，整图不存在 K5。
- 所有定量面板都显示或汇总全部 20 个网络。
- 图面无标题、方法段落、样本量句子、双 y 轴、装饰网格、伪重复或工程控制堆叠。
- Fig.2 的终点严格停在“input-driven, history-conditioned single transition”，把多步递归推广留给后续主图。
