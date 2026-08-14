# Fig.3 面板与布局契约

## 状态

- 科学逻辑：已冻结。
- 面板集合：已冻结为 a–g；a–f 为定量结果，g 为全宽概念综合图。
- 布局拓扑：前三排两列（a–f），第四排为全宽 g。
- 网络口径：全部使用 `seed_1000`–`seed_1019`，共 20 个独立训练网络。
- 结果归并：本图合并原 overlap-reentry 结果与原 local-support/competition 结果；二者属于同一条局部机制链，不能继续拆成两张主图。
- 数据边界：只读取已有持久化结果，不增加训练、模拟或实验。
- 实现状态：当前 spec、builder、renderer 与合同一致；g 只读取登记 SVG，不新增实验数据。
- 冻结日期：2026-08-10。

## 1. 为什么原来的三面板版本不能成立

去掉示意图后，如果 Fig.3 只留下“overlap 相关性、一次 perturbation、一次 downstream readout”三块结果，它们只是三个并列观察，不能形成独立主图所需的闭合推理：

- overlap 重要，但不知道它承载了什么继承状态；
- perturbation 有效，但不知道它改变了 later-input processing 的哪一步；
- downstream readout 改变，但不知道中间如何经过 firing 与局部竞争；
- 三者之间缺少从 inherited state 到 downstream write-back 的连续桥梁。

因此，不能用示意图补足结果数量，也不能保留三块结果后仅靠文字宣称机制完整。正确处理是把原先被分到 local-support/competition 图中的同链证据收回 Fig.3，使它成为一张完整的局部机制图。

综合图只能在 a–f 已经闭合证据链之后出现。g 的职责是把正文中的四阶段状态演化语法压缩为读者模型，而不是替代、增加或伪装定量证据。

## 2. Fig.3 的唯一问题

继承的分布式 STSP 状态如何被当前输入选择性读取，转化为局部放电事件，并最终改变下一层的状态写回？

Fig.3 不再重复证明“相同输入仍受历史影响”这一现象；该事实已经由 Fig.2 建立。本图只解释这一影响通过什么空间选择规则和局部动力学步骤实现。

## 3. 整图唯一中心结论

历史 STSP 并非均匀作用于 later-input processing。当前输入优先进入 retained support 与输入通路重叠的位置；这些位置具有更强的输入前支持，更容易发生 advance/recruit，且在局部候选单元之间获得事件前优势。空间特异的 overlap intervention 与整体 STSP attenuation/reset 分别证明“作用位置”和“状态本身”均为必要条件，随后这一偏置表现为对 Layer 2 写回位置的历史依赖。

允许的终点表述是：

> Overlap-aligned inherited STSP defines a spatially selective entry route through which a later input changes early firing, local competition, and the distribution of downstream Layer 2 write-back.

不能升级为：

> 每个 Layer 1 winner 已被逐一追踪并直接写入一个对应的 Layer 2 site。

现有证据支持连续的群体级机制链，不支持单元到单元的一一突触谱系追踪。

## 4. 必要论证链

`a overlap 位置具有因果功能 → b overlap 单元携带更强 retained support → c 支持被转化为 advance/recruit/loss → d Layer-1 STSP attenuation/reset 消除早期转化 → e downstream L2 写回保留历史依赖 → f Layer-1-only state transfer 定向改变 Layer-2 successor → g 综合为可递归 inherited-state update`

六个定量面板分别封闭一个不可跳过的推理缺口；g 只负责在证据之后建立统一读者模型，不新增推断。

- 没有 a，只能说 overlap 与结果相关，不能证明历史作用具有空间选择性。
- 没有 b，不知道被选择的位置是否真的承载更强的输入前状态支持。
- 没有 c，retained support 仍只是静态状态量，尚未进入当前输入的 processing events。
- 没有 d，a–c 仍缺少 STSP 状态对早期 firing conversion 的直接必要性验证。
- 没有 e，机制链停在 Layer 1 firing，无法回到论文核心的跨层状态重写。
- 没有 f，不能确认 Layer-1 inherited state 可因果定向改变 Layer-2 successor。

a–f 依次回答“在哪里、带着什么、变成什么、是否依赖 STSP、如何写回、能否定向改变 successor”；g 再把这条机制链综合为 `previous input → inherited state → sparse firing → immediate gain → inter-input decay → later firing → next inherited state`。

## 5. 面板契约

### a. Overlap-aligned STSP 是空间特异的功能入口

**角色**：首先确立 later-input pathway 与 retained state 重叠的位置不是任意空间标签，而是决定行为效应的因果入口。

**必要比较**：

- full dynamic；
- overlap-aligned Layer 1 STSP reset；
- non-overlap reset；
- random matched reset。

**核心端点**：

- `dynamic_minus_overlap_reset`；
- `nonoverlap_reset_minus_overlap_reset`；
- `random_reset_minus_overlap_reset`。

non-overlap reset 用于排除“只要重置任意非目标区域就会产生相同效果”，random matched reset 用于排除“只要重置相同数量位置就会产生相同效果”。两类控制回答不同混淆，不能互相替代。

**首选编码**：三条件竖向箱线图，横轴依次为 Dynamic、Non-overlap、Random，纵轴为相对 overlap reset 的 network-level `Accuracy contrast (%)`；不叠加 network 或 trial 点，不显示额外离群点标记，并保留零参照。

**持久化来源**：

`results/paper_figure_multi_seed/fig4_overlap_reentry/seed_*/data/metrics/panel_d_l1_stsp_overlap_perturbation_contrast.csv`

**20 网络既有结果方向**：

- dynamic 相对 overlap reset 的 accuracy-drop contrast 约为 0.056；
- non-overlap reset 相对 overlap reset 的 contrast 约为 0.056；
- random matched reset 相对 overlap reset 的 contrast 约为 0.0134。

**限定**：a 证明 overlap-aligned state 的空间功能特异性，不单独宣称 overlap 已解释全部机制。

### b. Overlap-dominant 单元携带更强的输入前 retained support

**角色**：说明 a 中有功能意义的位置，在 later input 到来前确实承载更强的继承 STSP 支持。

**必要分组**：

- overlap-dominant；
- probe-only dominant；
- balanced；

probe-only dominant 区分“当前输入驱动强”与“历史支持和当前输入重叠”；balanced 区分 overlap dominance 与一般共同响应。random matched 继续保留在统计与 Source Data 审计中，但不进入本面板图面。

**核心端点**：

`preprobe_mean_support`

**首选编码**：Overlap、Probe-only、Balanced 三根实色柱，显示 20-network 均值与 95% CI；不叠加 network 点。原始 unit 和 trial 必须先在网络内汇总。

**持久化来源**：

`results/paper_figure_multi_seed/fig5_local_support_competition/seed_*/data/metrics/panel_a_preprobe_support_metrics.csv`

**20 网络既有结果方向**：overlap-dominant 相对 probe-only 和 balanced 的 mean-support contrast 均为正；random matched 方向保留在统计审计中。

**限定**：b 证明的是 pre-input support enrichment，不把 `g` 的绝对大小直接解释为独立可读出的记忆内容。

### c. Retained support 被转化为早期 transition composition

**角色**：把 b 的静态输入前支持连接到当前输入引起的真实放电事件改变。

**必要端点**：

- `P_advance`；
- `P_recruit`；
- `P_loss`。

**必要比较**：overlap-dominant、probe-only dominant、random matched；balanced 不进入图面。

**首选编码**：单一坐标中的 Advance、Recruit、Loss 堆叠柱图，三根柱分别对应 Overlap、Probe-only、Random matched；纵轴为 `Transition composition (%)`，范围 0–100%。不得把单元或 trial 当作独立重复。

**持久化来源**：

`results/paper_figure_multi_seed/fig5_local_support_competition/seed_*/data/metrics/panel_b_transition_summary_by_group.csv`

**20 网络既有结果方向**：overlap-dominant 的 advance/recruit 总体组成高于 probe-only 与 random matched。

**限定**：c 的核心是“状态进入 processing event”。`loss` 作为可见组成进入主图，`unchanged` 只保留在 Source Data 审计中；三类可见 transition 不重新归一化为 100%。

### d. Layer-1 STSP 对早期 firing conversion 是必要的

**角色**：对 a–c 提供状态层面的直接干预验证，区分 inherited STSP contribution 与一般输入驱动。

**必要条件**：

- dynamic intact；
- Layer-1 STSP attenuation；
- Layer-1 STSP reset。

**核心端点**：

`P(advance OR recruit)` in the first 50 ms

**首选编码**：Attenuate、Reset 两个类别上的 20 个 network contrast 点、均值菱形与 95% CI，共用零参照；纵轴为 `Change in P (%)`。

**持久化来源**：

`results/paper_figure_multi_seed/fig5_local_support_competition/seed_*/data/metrics/panel_d_l1_stsp_perturbation_unit_transitions.csv`

**20 网络既有结果方向**：attenuation 与 reset 均降低早期 advance/recruit，reset 的降低更强。

**限定**：a 回答空间位置特异性，d 回答 STSP 状态对 firing-event 转化的必要性；两者的干预对象与端点不同。

### e. 当前处理把历史依赖写回 Layer 2

**角色**：把 Layer 1 的局部事件机制重新接回全文的跨层状态转移命题。

**核心端点**：

- `P(L2 update | prior-updated)`；
- `P(L2 update | not-prior-updated)`；
- dynamic-minus-static difference-in-differences。

**必要条件**：

- dynamic processing；
- static-frozen update opportunity。

**首选编码**：Dynamic、Static opportunity 两组中的 Prior updated 与 Not prior 分组柱图，显示 20-network 均值与 95% CI；DID 只保留在图注、statistics 与 Source Data。

**持久化来源**：

`results/paper_figure_multi_seed/fig5_local_support_competition/seed_*/data/metrics/panel_postprobe_l2_reupdate_history_composition.csv`

**20 网络既有结果方向**：

- dynamic 下 prior-updated sites 的 update probability 更高；
- 该 prior-minus-nonprior 差异在 dynamic 下大于 static opportunity condition。

**限定**：static-frozen 数值是“若允许更新时会出现的 update opportunity”，不是实际 STSP mutation；图注和正文不得将其写成 static condition 中已发生的状态更新。

### f. Layer-1 inherited state 可定向改变 Layer-2 successor

**角色**：在 e 的历史依赖写回之后，以选择性 Layer-1-only `u/x` transfer 检验 Layer-2 successor 是否向 donor 方向移动。

**核心端点**：

`layer1_only_layer2_update_donor_transfer` at K1

**首选编码**：一个 `L2 successor` 类别上的 20 个 network donor-transfer 点与网络均值；纵轴为 `Donor-transfer index`，保留零参照，不增加第二 endpoint。

**持久化来源**：

`results/paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory/aggregate/fixed_b_confirmatory_network_scalars.csv`

**20 网络既有结果方向**：K1 Layer-1-only transfer 的 Layer-2 successor donor-transfer index 在全部网络中为正。

**限定**：f 支持群体级、方向性的 causal entry，不宣称 Layer-1 单元到 Layer-2 单元的一一谱系，也不替代 Fig.4 的 C5 multi-step transfer。

### g. 证据后的状态演化综合

**角色**：在 a–f 的局部机制链已经闭合后，把正文反复使用的状态演化语言压缩成一个连续读者模型。

**可见链条**：

`Previous input → inherited state → sparse firing → immediate gain → inter-input decay → later firing → immediate gain`

图中四个等宽阶段分别显示 inherited state、当前输入选择后的放大子集、输入间衰减后的 successor state，以及下一输入再次选择后的放大子集。圆点位置固定；大小与颜色只表达相对状态强弱和本次选择，不编码新的实测数值。

**限制**：g 不展示网络结构、层编号、donor/receiver、swap、效应量、坐标轴或统计量；它是 a–f 的综合，不是新的证据面板。

## 6. 冻结布局

画布：`165 mm × 202 mm`；外边距 2 mm，横向 gutter 2 mm，排间距 2 mm。

| 排 | 面板 | 冻结槽位，单位 mm | 逻辑任务 |
|---|---|---|---|
| 第一排左 | a | `x=2, y=2, w=79.5, h=48` | overlap 空间因果入口 |
| 第一排右 | b | `x=83.5, y=2, w=79.5, h=48` | 输入前 retained support |
| 第二排左 | c | `x=2, y=52, w=79.5, h=48` | advance/recruit 转化 |
| 第二排右 | d | `x=83.5, y=52, w=79.5, h=48` | Layer-1 STSP 因果必要性 |
| 第三排左 | e | `x=2, y=102, w=79.5, h=48` | Layer 2 历史依赖写回 |
| 第三排右 | f | `x=83.5, y=102, w=79.5, h=48` | Layer-1-only → Layer-2 successor transfer |
| 第四排全宽 | g | `x=2, y=152, w=161, h=48` | 状态演化综合 |

**逐排语义**：

- 第一排回答“历史从哪里进入，以及该位置携带什么”；
- 第二排回答“状态如何变成放电事件，以及该转化是否依赖 Layer-1 STSP”；
- 第三排回答“历史依赖如何写回 Layer 2，以及 Layer-1 state transfer 能否定向改变 successor”；
- 第四排只在证据之后综合为可递归状态更新模型。

g 必须保持整行并位于图末；不得移到 Fig.3 开头，也不得与 a–f 交换成为论证前提。

## 7. 明确移出 Fig.3 主图的已有结果

以下结果不是无效，而是不能承担本图的必要主链：

| 已有结果 | 归宿 | 原因 |
|---|---|---|
| pixel similarity 对 probe bias 的趋势 | 补充材料或 Source Data | 定义 operating regime，但不能替代 overlap-specific causal intervention |
| natural high-vs-low overlap accuracy-drop 分箱 | 补充材料 | 描述性、非因果，20 网络 contrast 未形成稳定主图结论 |
| overlap-preserved L3 trajectory | 补充材料 | 是强 downstream consequence，但 Fig.2 已经证明早期输出转移；放入 Fig.3 会重复功能读出 |
| accumulator/decision deflection | 补充材料 | 与 L3 trajectory 同属下游结果，不能填补局部 firing→competition→write-back 的中间缺口 |
| 任何额外的独立机制示意图 | 不制作 | g 已承担唯一的证据后综合；新增示意图不会提供新证据 |

## 8. 与 Fig.2 及 Fig.4 的边界

### Fig.3 不重复 Fig.2

- 不再展示 exact-B common component 或历史残差 `Γ`；
- 不再展示 post-B → identical-C donor transfer；
- 不再用 L3/output trajectory 重新证明历史影响存在；
- 不再展示 rescue/loss 行为分解。

### Fig.3 从原 local-support/competition 图收回的结果

原 local-support/competition 图中的 preprobe support、transition、STSP perturbation 与 L2 re-update 并入 Fig.3 定量主链；winner–loser trace 保留在 statistics／Source Data，不进入当前 artwork。

### 新 Fig.4 因此只承担递归推广

新 Fig.4 不再重复局部 support/competition。它只回答同一转移原则能否从一步历史推广到连续输入、累积历史和反复转移。

## 9. 统计与证据边界

- 所有主图显示与推断统一使用 20 个独立训练网络：`seed_1000`–`seed_1019`。
- network 是独立重复单位；trial、unit、site、coordinate 和 event 均须先按各端点的预定层级汇总到 network。
- 不继承任何只保留部分网络的旧 aggregate 或筛选口径。
- a、d 与 f 分别约束空间入口、STSP 必要性和跨层定向 transfer；任何单一面板都不能升级为完整机制链。
- c 的 15 ms transition profile 与 d 的 first-50-ms perturbation endpoint 必须分别直标时间窗，不能暗示二者使用同一时间窗。
- e 的 static condition 只能称为 update opportunity。
- 图面不得显示 trial/cell 伪重复点、双 y 轴、装饰性网格、工程 plumbing control 或未经预先定义的事后 subgroup。

## 10. 最终验收标准

- Fig.3 的 a–f 均提供不可替代的定量证据；g 只在证据之后提供全宽概念综合。
- 阅读顺序能够自然复述为：`where → support → firing conversion → causal necessity → write-back → successor transfer → synthesis`。
- a 与 d 的两个干预分别回答空间特异性和状态必要性，不被合并成含义不清的“perturbation panel”。
- d 只使用预定义的 first-50-ms advance/recruit endpoint。
- f 停在群体级 Layer-1-only → Layer-2 successor donor transfer，不宣称单元级一一谱系。
- g 不含额外数值、效应量、推断或未经 a–f 支持的机制符号。
- 原 overlap-reentry 与 local-support/competition 结果在科学上合并，但不重命名或重跑其父级 artifacts。
- 全图只汇总 `seed_1000`–`seed_1019` 的 20 个网络。
- Fig.3 的最后一句能够自然引出 Fig.4：既然单次 later-input processing 的局部实现已经闭合，这一转移原则能否在连续输入和累积历史中反复成立？
