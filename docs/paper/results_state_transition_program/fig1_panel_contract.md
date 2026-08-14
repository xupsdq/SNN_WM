# Fig.1 面板与布局契约

## 状态

- 科学逻辑：已冻结。
- 面板集合：已冻结为 a–e，不增加第六个主面板。
- 拓扑：已冻结为三排结构，第一排 a，第二排 b–c，第三排 d–e。
- 网络口径：全部使用 `seed_1000`–`seed_1019`，共 20 个独立训练网络。
- 数据边界：只使用现有持久化多网络结果，不新增模拟或实验。
- 实现状态：本文件先作为科学与布局权威；现有 `fig1.yaml`、适配器和渲染器尚未迁移到本契约。
- 冻结日期：2026-07-30。

## 1. 整图唯一问题

在完成工作记忆任务的固定 STSP-SNN 中，是否存在一个在延迟期无持续放电时仍保留输入内容、并能被后续计算读取的可继承状态？

## 2. 整图允许形成的结论

网络能够完成任务；延迟期持续放电消失；分布式 `u/x` 状态仍保留输入身份；置换该状态会把后续读出的归因从原样本方向推向供体样本方向。因此，延迟期 STSP 构成一个静默、内容特异且具有功能作用的可继承工作记忆状态。

Fig.1 只建立全文的状态前提。它不单独证明状态如何被新输入改写、如何形成下一层 successor state，也不单独证明状态转移会在序列中递归发生。

## 3. 明确禁止的越界结论

- 不把高任务准确率本身解释为 STSP 机制证据。
- 不把延迟解码高于机会水平解释为状态已经发生历史条件化改写。
- 不声称历史状态决定后续答案；这里只支持历史状态会被后续计算使用。
- 不声称 STSP 是工作记忆唯一可能的生物机制。
- 不把 L1、L2、L3 的结果外推为任意神经层或任意网络架构的普遍规律。

## 4. 必要论证链

`a 研究对象 → b 整体任务有效 → c 排除持续放电 → d 静默状态包含内容 → e 静默状态被后续计算读取`

五个面板分别封闭一个不可跳过的逻辑缺口：

- 没有 a，读者不知道状态位于什么回路及哪些连接中。
- 没有 b，后续状态分析可能来自一个没有实现目标功能的网络。
- 没有 c，内容维持仍可由持续放电解释。
- 没有 d，静默状态可能只是无结构的突触残留。
- 没有 e，可解码状态仍可能只是与任务相关但不参与后续计算的伴随量。

## 5. 面板契约

### a. 网络示意图

**角色**：定义全文研究对象和状态所在的固定回路。

**核心内容**：输入图像经 DoG temporal encoder 进入 Spiking Layer 1、Spiking Layer 2 和 decision/readout；明确标出 STSP 突触和层内抑制。

**权威底图**：

`results/paper_figures/outputs/structure-enhanced.svg`

**保留内容**：

- input image；
- DoG temporal encoder；
- Spiking Layer 1；
- Spiking Layer 2；
- decision/readout；
- STSP feedforward connections；
- layer-local inhibition；
- earliest-spike class readout。

**不得加入**：任务结果、状态转移结论、输入序列时间线或正文式解释。a 只承担网络结构，不同时承担机制结果。

**实现要求**：

- 以原 SVG 的可编辑矢量元素为基础，不栅格化重画。
- 适配 161 mm × 48 mm 的满宽槽位；原 SVG 的 `viewBox="331 0 960 296"` 与该横向比例兼容。
- 统一为项目字体和最小文字预算；面板内部不放标题。
- 简化并统一结构图图例，保留 STSP 与 inhibition 的必要语义。
- 清理导入预览中 decision/readout 区域后方的非预期深色矩形；只允许输入图像自身保留黑色底。
- 背景保持白色或透明，并在最终 SVG、PDF 和 PNG 中分别检查。

### b. 网络准确率轨迹

**角色**：建立整体功能前提，而不是承担机制证明。

**核心比较**：20 个独立 network seed 的整体分类准确率，以及其是否稳定落在 85%–95% 的视觉强调区间。

**持久化来源**：

`results/paper_figure_multi_seed/fig1_functional_stsp_substrate/fig1_functional_stsp_substrate/seed_*/data/metrics/panel_b_baseline_metrics_by_network.csv`

**独立重复单位**：network seed。

**主显示量**：`overall_recall × 100`。

**冻结编码**：横轴为具体 network seed（1000–1019），纵轴为 Accuracy (%)；每个 seed 一个点，并按 seed 顺序连接。85% 与 95% 各画一条横向虚线，两线之间使用无文字的淡蓝色视觉强调带。该带不是置信区间或统计阈值。图面不画、不标注 10% chance line。

**排除**：逐 trial 点、逐数字召回、混淆矩阵和训练过程。它们可以进入补充材料，但不是 Fig.1b 的必要内容。

### c. 延迟期放电消失

**角色**：排除持续放电维持工作记忆的解释。

**核心比较**：在真实时间轴上显示刺激期内的放电轨迹，以及刺激结束后 delay 期放电消失；分别保留 L1、L2、L3。

**持久化来源**：

主绘图来源：

`results/multi_seed_rollout/fig1_time_binned_firing/seed_*/data/metrics/supp_time_binned_firing_rates.csv`

一致性验证来源：

`results/paper_figure_multi_seed/fig1_functional_stsp_substrate/fig1_functional_stsp_substrate/seed_*/data/metrics/supp_phase_firing_rates.csv`

50 ms 来源是本轮经用户明确授权、使用原 checkpoint 与原 DMS trial 定义执行的 plot-support replay；它只生成新的派生数据，不训练网络、不增加条件，也不覆盖既有父结果。重汇总后的 200 ms phase rate 必须与持久化来源在 `network_seed × layer × phase` 层级保持一致。

**独立重复单位**：先在每个 network seed 内汇总 trial，再进行网络级显示和推断。

**冻结显示量**：每个 network seed、每一层、每个 50 ms 时间窗的 population spike rate，单位固定为 Hz；trial 先在 network 内求均值，再进行跨网络显示。

**冻结编码**：横轴为实际 Time (ms)，范围覆盖 0–600 ms；纵轴为 Spike rate (Hz)。在真实刺激开始 0 ms 与结束 200 ms 处各画一条竖向虚线，两线之间使用无文字的淡蓝色视觉强调带。L1、L2、L3 使用项目既定层级颜色和非颜色冗余；不再使用 Stimulus／Early delay／Late delay 作为分类横轴。

**排除**：probe 阶段不是“延迟期放电消失”结论所必需，不进入主面板；完整 phase firing rate 和 trial-level 明细保留在补充材料或 source data。

### d. 延迟期 `u/x` 内容解码

**角色**：证明无持续放电时，分布式突触状态仍保留输入身份。

**核心比较**：L1、L2、L3 的联合 `u/x` 解码准确率随 delay 的变化，并与 10% 机会水平比较。

**持久化来源**：

`results/paper_figure_multi_seed/fig1_functional_stsp_substrate/fig1_functional_stsp_substrate/seed_*/data/metrics/panel_c_delay_decode_metrics.csv`

**独立重复单位**：network seed；分类 trial 不能被当作独立网络重复。

**主显示范围**：已有 100、200、400、800 和 1200 ms delay；feature 固定为 `ux_concat`。

**首选编码**：delay × decoding accuracy 的网络级轨迹，L1、L2、L3 为三条语义一致的层级序列；保留 10% chance line。统计细节、训练样本量和分类器实现放入图注或方法，不写入图面。

**排除**：单 trial 预测点、训练损失、训练准确率和分类器运行信息。

### e. `u/x` 状态置换导致归因转移

**角色**：证明可解码的延迟状态会被后续计算读取，而不是无功能伴随量。

**核心比较**：在错误 trial 池内部，比较 intact dynamic state 与联合置换 `u/x` 后 Original、Donor、Other 三类预测归因组成。

**持久化来源**：

`results/paper_figure_multi_seed/fig1_functional_stsp_substrate/fig1_functional_stsp_substrate/seed_*/data/raw/panel_d_dms_condition_trial_readout.csv`

**独立重复单位**：network seed。

**冻结编码**：Dynamic STSP 与 `u/x` shuffle 各使用一根 100% 堆叠柱，组成顺序固定为 Original、Donor、Other。每个 network seed 先在自身错误池内计算组成比例，再显示 20-network 均值。Original 与 Donor 段内标注百分比数字；Other 保留在组成中但不标数字。

**排除**：正确 trial 不进入组成分母；static-frozen 的总体准确率比较也不属于 e。当前旧版 Fig.1d 的多条件 probe-accuracy 图转入补充材料。

## 6. 冻结布局

最终画布使用 165 mm × 152 mm，外边距 2 mm，行间与列间 gutter 均为 2 mm。

| 排 | 面板 | 槽位，单位 mm | 语义关系 |
|---|---|---|---|
| 第一排 | a | `x=2, y=2, w=161, h=48` | 满宽定义研究对象 |
| 第二排左 | b | `x=2, y=52, w=79.5, h=48` | 整体功能前提 |
| 第二排右 | c | `x=83.5, y=52, w=79.5, h=48` | 排除持续放电 |
| 第三排左 | d | `x=2, y=102, w=79.5, h=48` | 静默状态的内容证据 |
| 第三排右 | e | `x=83.5, y=102, w=79.5, h=48` | 静默状态的功能证据 |

布局不把 b–c 或 d–e 伪装成共享坐标的直接定量比较：

- b 与 c 相邻，是“任务有效→不是持续放电”的推理关系，不共享 y 轴。
- d 与 e 相邻，是“状态包含什么→状态是否被使用”的推理关系，不共享 y 轴。
- b/d 与 c/e 的左右槽位边界必须严格对齐，属于结构对齐。
- b–e 槽位内部边距固定为左 11 mm、右 3 mm、上 8 mm、下 10 mm，因此每个定量面板的数据区固定为 65.5 mm × 30 mm。
- 第二排 b/c 的 x 轴基线全局位置固定为 `y=90 mm`，第三排 d/e 固定为 `y=140 mm`；左列 b/d 的 y 轴固定为 `x=13 mm`，右列 c/e 固定为 `x=94.5 mm`。
- c 与 d 均保留 L1/L2/L3 语义映射，并分别在所属面板内使用紧凑图例。
- b 不需要图例；e 使用 Original、Donor、Other 的紧凑图例；a 只保留结构图自己的简化图例。
- 阅读顺序固定为 a → b → c → d → e。

## 7. 与当前 Fig.1 规格的迁移关系

| 当前结果或面板 | 新 Fig.1 归宿 |
|---|---|
| 当前 a 网络结构 | 保留为新 a，并按本契约清理 |
| 当前 b baseline performance | 保留为新 b，改为按 network seed 连接的准确率轨迹 |
| 新授权的 50 ms time-binned replay | 用作新 c 的绘图来源，并以 `supp_phase_firing_rates.csv` 验证 |
| 当前 c delay decoder | 移至新 d，并保留完整 delay 维度 |
| 当前 e attribution | 改为错误池内 Original／Donor／Other 的完整组成 |
| 当前 d condition comparison | 移入补充材料，不再占主面板 |

只有在本 Fig.1 契约确认后，才同步修改最终统计 builder、面板渲染器和布局。除上文明确记录的 Fig.1c 50 ms plot-support replay 外，迁移不得重新运行模拟。

## 8. 最终验收标准

- 五个面板恰好对应研究对象、整体效果、活动静默、内容保留和功能读取五个逻辑任务。
- 所有定量结果均追溯到现有多网络持久化文件。
- b–e 均使用全部 20 个 network seeds，且推断单位为 network seed。
- a 使用并清理指定 SVG，不另造一套网络结构。
- 图面无面板标题、样本量句子、统计检验段落或方法说明。
- 不出现双 y 轴、装饰性网格、灰色背景带、重复图例或 trial/cell 伪重复。
- Fig.1 的终点严格停在“存在功能性的可继承状态”，把“状态如何被改写”留给 Fig.2。
