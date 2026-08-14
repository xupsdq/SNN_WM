# Fig.5 面板与布局契约

## 状态

- 科学逻辑：已冻结。
- 面板集合：已冻结为 a–f，全部为定量结果面板，不设置示意图。
- 布局拓扑：两排等高、每排三列等宽；第一排 a–b–c 为 pair-organization
  证据带，第二排 d–e–f 为 multi-item organization 证据链。
- 网络口径：全部使用 `seed_1000`–`seed_1019`，共 20 个独立训练网络。
- 数据边界：只读取已有 pair-state、multi-item reanalysis 与 K × delay morphology 结果；不增加训练、模拟或实验。
- 整图角色：只证明 repeated transition 的结构结果，不使用 cue、行为 readout 或 firing perturbation 证明功能。
- 实现状态：本文件是新 Fig.5 的科学权威；既有 `fig2`、`fig3`、`fig6` 结果目录及旧 YAML 名称只是数据来源身份。
- 当前实现冻结日期：2026-07-31。

## 1. Fig.4 到 Fig.5 的唯一推进

Fig.4 已经证明新输入能够反复使状态偏离 matched-passive evolution，但“持续变化”并不等于“形成有组织的历史表征”。状态还可能只是：

- 被最新输入覆盖；
- 多个输入的无结构叠加；
- 与任意未经历组合同样相似；
- 随项目数增加而表面增大，却没有稳定的 constituent organization；
- 在负荷或延迟增加时无边界地保持。

因此 Fig.5 只回答：

> 反复状态转移之后，STSP 是否形成保留多个历史成分、对真实经历组合具有特异性、具有序列组织且受容量和保持时间约束的状态？

## 2. 整图中心结论

两项目后的 Layer 2 `u/x` 状态同时保留两个 constituent，并对真实经历的 pair 相对于 shuffled pair 具有特异性；这种特异性在扣除最佳线性 constituent mixture 后仍然存在。推广到多个项目时，Layer 2 state 的有效项目数随 K 扩展，但在高负荷下出现压缩，其 constituent weights 分布在多个序列位置且具有有序的早期项目优势，而不是坍缩为最后一个项目。相应的 inherited Layer 1 STSP morphology 在 K × delay 空间中呈现清楚的结构保持边界。

允许的终点表述是：

> Repeated transitions organize recent history into a distributed, history-specific and capacity-limited STSP state rather than an overwritten or unstructured sum of inputs.

不能将该状态命名为固定 slot、离散 chunk 或无损 episode，也不能声称 Layer 2 与 Layer 1 的数值是同一个 state block 的逐层追踪。

## 3. 必要论证链

`a 两个 constituent 均被保留 → b 状态对真实经历 pair 具有特异性 → c 特异性超出简单线性叠加 → d 组织推广到多项目并出现容量压缩 → e 多项目权重具有分布式序列结构 → f 结构保持具有 K × delay 边界`

六个面板分别封闭不同缺口：

- 没有 a，pair state 可能只是最近一个项目的残留。
- 没有 b，同时接近 A 和 B 仍可能是任何 A+B 组合都会产生的非特异平均。
- 没有 c，真实 pair specificity 仍可能完全由两个 singleton 的线性叠加解释。
- 没有 d，结论只能停在两项目，无法说明连续转移如何组织更长历史。
- 没有 e，`N_eff` 增加只说明系数数量变多，不说明这些权重如何沿序列组织，也不能排除 latest-only collapse。
- 没有 f，只能证明一个固定负荷／延迟切片中的组织，无法划定该组织在哪些历史长度与保持时间内成立。

## 4. 面板契约

### a. Layer 2 pair state 同时保留两个 constituent

**角色**：以最小的两项目状态建立“不是覆盖”的第一步。

**核心端点**：

`min_component_similarity`

该端点取 pair state 对 constituent A 与 B 相似度中的较小值；只有较小值仍然高，才能证明两个成分都被保留。只显示平均相似度会允许“一个很高、一个很低”的覆盖情形。

**首选编码**：恢复旧 Fig.2b 的双组成柱图语法，直接显示 Item A 与 Item B 的
20-network 均值和 95% CI；`min_component_similarity` 继续作为 Source Data 中的
保护端点，防止双组成均值掩盖某一组成丢失。pair 行必须先汇总到 network。

**持久化来源**：

`results/paper_figure_multi_seed/new_results_reanalysis/metrics/fig6_layer2_pair_network_metrics.csv`

**20 网络既有结果方向**：最小 constituent similarity 约为 0.99，两个 constituent 均被 pair state 保留。

**限定**：高相似度表示分布式状态中保留 constituent geometry，不表示可以从 STSP 单独、无 cue 地读出两个类别。

### b. Pair state 对真实经历组合具有特异性

**角色**：排除 a 只是“同时接近任意两个 singleton state”的一般几何结果。

**核心端点**：

`true_minus_shuffled`

**必要比较**：

- true experienced pair；
- constituent-matched shuffled pair。

**首选编码**：恢复旧 Fig.2d 的直接条件比较，显示 Experienced pair 与
Shuffled pair 的 20-network 箱线图；`true_minus_shuffled` 保留为正式配对推断，
不再把差值单独画成空旷的一维估计点。

**持久化来源**：

`results/paper_figure_multi_seed/new_results_reanalysis/metrics/fig6_layer2_pair_network_metrics.csv`

**20 网络既有结果方向**：true-minus-shuffled 为正。

**限定**：b 证明对经历组合的统计特异性，不将 shuffled control 描述成生物学上“未见过的新对象”。

### c. Pair-specific organization 超出最佳线性 constituent mixture

**角色**：排除 b 的 pair specificity 只是 singleton A 与 B 的线性加和。

**核心端点**：

`residual_pair_specificity`

**必要步骤**：

1. 用 constituent states 对 pair state 拟合最佳线性 mixture；
2. 在 residual state 上比较 true 与 shuffled pair correspondence；
3. 以 network 为独立重复汇总 residual specificity。

**首选编码**：直接显示 residual experienced-pair 与 residual shuffled-pair
两根实色柱及 95% CI；`residual_pair_specificity` 作为两者配对差保留在正式统计，
不复制旧 Fig.2c 的双 y 轴。

**持久化来源**：

`results/paper_figure_multi_seed/new_results_reanalysis/metrics/fig6_layer2_pair_network_metrics.csv`

**20 网络既有结果方向**：residual pair specificity 为正。

**为什么不把 `linear_mixture_gain` 并入主面板**：它描述线性模型相对基线的拟合收益，却不能替代 residual specificity 对“超出简单叠加”的直接检验；加入二者会把 c 变成模型诊断清单。完整拟合质量进入补充材料。

**限定**：c 支持非平凡的 pair-specific reorganization，不足以证明一个离散、不可分解的 chunk。

### d. Layer 2 多项目 state 随历史长度扩展并在高负荷下压缩

**角色**：把两项目组织推广到连续多项目历史。

**核心端点**：

- `N_eff`；
- `seq_len = 3, 5, 7, 10`。

**首选编码**：沿 K 显示 20-network 均值与 95% CI 的有序轨迹，删除 individual
network spaghetti，并以 `N_eff = K` 作为理论上界／无压缩参照；不得把该参照
写成性能阈值。

**持久化来源**：

`results/paper_figure_multi_seed/new_results_reanalysis/metrics/fig6_layer2_multi_network_metrics.csv`

**20 网络既有结果方向**：

- K3、K5 时 `N_eff` 接近 K；
- K7、K10 时仍随 K 增加，但相对 K 出现更明显压缩；
- K10 的平均 `N_eff` 约为 8，而不是 10 个独立、等权 traces。

**限定**：`N_eff` 是 NNLS constituent-weight distribution 的有效数量，不是可独立回忆项目数；功能访问必须由 Fig.6 另行检验。

### e. 多项目 constituent weights 分布在多个序列位置并具有有序结构

**角色**：说明 d 的有效项目数对应怎样的内部组织。

**必要内容**：

- 行：K3、K5、K7、K10；
- 列：相对 serial position；
- 单元格：在 sequence 内归一化后的 `item_weight`，再按 network 汇总。

**首选编码**：K × serial-position 热图；不存在的序列位置使用白色／透明留空，
不得填零或使用具有数值重量的灰色块。各 K 内权重和为 1，使读者比较分布形状
而非总量。

**持久化来源**：

`results/paper_figure_multi_seed/new_results_reanalysis/metrics/fig6_layer2_multi_item_weights.csv`

**既有结果方向**：权重分布跨越多个 constituent，并呈现较早项目权重较高的有序梯度；不是 latest-only collapse。

**禁止表述**：

- 不称为 recency advantage；现有 `recency_bias` 指标为负；
- 不称为固定 primacy slot；
- 不把空白位置解释为零权重项目。

### f. Inherited Layer 1 morphology 具有结构性的 K × delay operating boundary

**角色**：把“形成了什么组织”从单一时点推广到负荷与保持时间二维空间，并给 Fig.6 的功能边界提供结构基线。

**核心端点**：

- Layer 1 `g` 的 `N_eff_fraction`；
- `seq_len = 3, 5, 7, 10`；
- `delay_ms = 100, 200, 400, 800`。

**首选编码**：K × delay 热图，每个单元格先在 sequence 内汇总，再在每个 network 内汇总，最后显示全部 20 网络的中心估计。使用 `N_eff_fraction` 而不是 raw `N_eff`，避免 K 的机械上界主导色阶。

**持久化来源**：

`results/paper_figure_multi_seed/fig3_multiitem_peak_landscape/seed_*/data/metrics/panel_c_morphology_boundary_metrics.csv`

**20 网络既有结果方向**：

- 较短序列在较长 delay 下仍能保留较高 effective fraction；
- 长序列尤其 K10 随 delay 增加出现明显压缩；
- 结构保持不是无负荷、无时间限制的。

**跨层限定**：

- a–e 描述 repeated transition 之后的 Layer 2 `u/x` successor organization；
- f 描述后续输入到达前可继承的 Layer 1 `g` morphology；
- 二者共同支持网络中 homologous distributed STSP organization 的推广，但不是同一批 state coordinates 的纵向追踪；
- f 的绝对数值不得与 d、e 直接作层间大小比较。

## 5. 冻结布局

画布为 `165 mm × 102 mm`，外边距 2 mm，横向 gutter 2 mm，排间距 2 mm。
两排均为 48 mm 高，每排三列等宽；面板内容差异只通过 `plot_bbox`、图例和色条
留白处理，不改变 slot 等分关系。

| 排 | 面板 | 冻结槽位，单位 mm | 逻辑任务 |
|---|---|---|---|
| 第一排 | a | `x=2, y=2, w=52.333, h=48` | 两个 constituent 均保留 |
| 第一排 | b | `x=56.333, y=2, w=52.334, h=48` | 真实 pair specificity |
| 第一排 | c | `x=110.667, y=2, w=52.333, h=48` | 超出线性 mixture |
| 第二排 | d | `x=2, y=52, w=52.333, h=48` | 多项目扩展与压缩 |
| 第二排 | e | `x=56.333, y=52, w=52.334, h=48` | 序列位置组织 |
| 第二排 | f | `x=110.667, y=52, w=52.333, h=48` | K × delay 结构边界 |

**逐排语义**：

- 第一排连续排除 overwrite、任意组合和简单线性 mixture；三图直接显示构成
  这些差值的条件，而不是只显示摘要点；
- 第二排依次说明多项目数量、位置分布和结构边界；三图保持等宽，d–e–f 的数据区
  上下边界一致，e、f 的水平色条放在数据区上方并按顶部图例的方式占用装饰带；
  色条从上到下依次为文字标签、刻度数字和色带；
- 上下排共享三列 slot 边界和 48 mm 行高；数值尺度只在科学上可比较时共享。

全图不设置示意图。Fig.4 已经定义 repeated-transition protocol；Fig.5 的六个定量面板本身足以完成“constituents → organization → capacity boundary”的结构论证。

## 6. 明确移出 Fig.5 主图的既有结果

| 既有结果 | 归宿 | 原因 |
|---|---|---|
| pair `linear_mixture_gain`、cross-validated `R²` 与 residual norm | 补充材料 | 拟合诊断，不是结构主链的独立结论 |
| 每个 sequence／pair 的全部原始 similarity | Source Data | 防止 sequence 伪重复占满主图 |
| Layer 2 `u` 与 `x` 分开版本 | 补充材料 | 主图使用 joint `u/x` 回答组织；变量拆分用于稳健性 |
| raw `N_eff` 的完整 K × delay Layer 1 热图 | 补充材料 | 主图 f 使用 normalized structural fraction |
| partial-cue、item recovery、rescued fraction | Fig.6 | 属于功能访问，不是结构 |
| `support_gain_corr` | 不进入主图 | 现有条件间方向不稳定，不能为了连接结构与功能而强行使用 |

## 7. 与相邻主图的边界

### Fig.5 不重复 Fig.4

- Fig.4 的端点是 observed-minus-passive state displacement；
- Fig.5 不再显示 stage displacement，而分析累积状态内部的 constituent organization；
- “状态在变化”与“状态有组织”是两个不同论证层级。

### Fig.5 不提前回答 Fig.6

- 不使用 cue、target probability、rescued fraction、spike recruitment 或 perturbation；
- `N_eff` 不称为 accessible item count；
- Fig.5 的最后一句只提出：这种结构化、受限的 state 是否仍能被 later input 有内容选择性地利用？

## 8. 统计与证据边界

- 所有主图汇总统一使用 20 个独立训练网络。
- pair、sequence、item position、delay、coordinate 和 coefficient 均先按预定层级汇总到 network。
- a–c 使用 Layer 2 pair protocol；d–e 使用 Layer 2 multi-item reanalysis；f 使用 Layer 1 K × delay morphology protocol。不同协议之间只进行逻辑推广，不把数值放在同一尺度上比较。
- shuffled control、NNLS decomposition 和 normalized weights 的定义必须在 Methods／caption 中明确。
- 热图单元格、pair、sequence 和 item 都不是独立重复。
- 不根据 serial-position pattern 事后发明新的 primacy/recency cutoff。

## 9. 最终验收标准

- 阅读顺序能够自然复述为：`retain both → identify experienced pair → exceed linear sum → scale to many items → reveal serial organization → expose structural boundary`。
- a 使用最小 constituent similarity，不能只用平均值掩盖单一 constituent 丢失。
- b 与 c 分别回答经历组合特异性和超出线性叠加，不能合并成含义不清的“pair score”。
- d 明确区分 `N_eff` 与可访问项目数。
- e 不把不存在的 serial positions 填零，也不误称为 recency effect。
- f 明确标注 Layer 1 `g`，并禁止与 Layer 2 `u/x` 作绝对数值比较。
- 六个面板全部使用或汇总 `seed_1000`–`seed_1019`。
- 图内不含 cue/readout、行为结果或 firing intervention。
- Fig.5 的终点自然引出 Fig.6：结构已经形成，但它是否能在不完整且内容匹配的 later input 下被访问，仍需功能证据。
