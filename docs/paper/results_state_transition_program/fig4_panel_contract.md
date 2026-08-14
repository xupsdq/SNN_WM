# Fig.4 面板与布局契约

## 当前权威修订（2026-08-10）

本节覆盖下方历史合同；下方内容只保留为设计记录，不再约束当前主图。

- 面板集合：a–d，共四个定量面板；Fig.4 不再包含示意图。
- 唯一问题：相同的 input-driven、history-conditioned transition 能否在连续输入和更深历史中反复成立，并产生累积行为后果。
- 论证链：
  `a early Layer-2 donor transfer → b post-C Layer-3 successor transfer → c recurrence breadth → d accumulated behavioral cost`。
- a：K1/K5 下的 C5 early Layer-2 processing donor-transfer index。
- b：K1/K5 下的 C5 post-C Layer-3 successor donor-transfer index。
- c：stage 2–10 的 Observed 与 Persisted passive state displacement。
- d：K1/K5 relation-balanced Rescue/Loss。
- 原 successor-reuse/swap 示意图从 Fig.4 artwork 删除。C5 干预身份、B→C mapping、receiver/donor 约束与 selective transplant 仍由图注、Methods、Source Data 和 provenance 承担，不再占用主图面板。
- 状态演化的唯一概念综合位于 Fig.3g；Fig.4 只展示该原则的递归证据与累积后果。
- 画布：`165 mm × 102 mm`。
- slot：
  - a：`[2.000, 2.000, 79.500, 48.000] mm`；
  - b：`[83.500, 2.000, 79.500, 48.000] mm`；
  - c：`[2.000, 52.000, 79.500, 48.000] mm`；
  - d：`[83.500, 52.000, 79.500, 48.000] mm`。
- 科学边界：20 个独立 network（1000–1019）；plot-only 只读取既有 final bundle，不新增训练、模拟或 forward replay。
- 推断边界：donor transfer 不能升级为 necessity、complete mediation 或 uniqueness；stage recurrence 不表示每个 stage 都重复了完整 C5 transplant protocol。

当前状态：`four_panel_recurrence_chain_validated_plot_ready`。

## 历史状态（2026-08-01，已被上节覆盖）

### 状态

- 科学结果选择：已冻结为 F8 门控后的五面板链。
- 图型与布局：已授权生成基础版本，后续允许逐面板视觉调整。
- 网络口径：`seed_1000`–`seed_1019`，20 个独立训练网络。
- 数据边界：只读取 persisted progressive-update、fixed-B 与已验证的
  accumulated-history 统计 artifact；禁止训练、模拟或 forward replay。
- 画布：`165 mm × 102 mm`；第一排 a–b 等宽，第二排 c–e 等宽。
- 本文件覆盖 2026-07-30 的“协议示意＋early/late＋热图”旧合同。
- 更新日期：2026-08-01。

## 1. 唯一问题

当输入连续到来、历史从 K1 加深到 K5 时，每个新输入是否仍能改写继承的
STSP 状态；如果能，累积历史如何改变行为，而这种改变是否只是因为状态
转移机制已经消失？

Fig.2 建立一步 history relation 如何定向改变 exact-B。Fig.4 不重复该
一步关系，而推进到 successive-input 与 accumulated-history 层级。

## 2. 中心结论

> Successive inputs repeatedly rewrite the inherited STSP state beyond
> equal-time passive evolution. Accumulated K5 history reduces
> relation-balanced rescue and increases relation-balanced loss, even though
> the common update, history residual, event-linked residual and downstream
> donor transfer remain detectable across 20 networks.

中文口径：连续输入持续改写继承的 STSP 状态；相对 K1，累积 K5 历史降低
relation-balanced Rescue、提高 relation-balanced Loss，但共同更新、历史
残差、真实 spike-event 落点和向下游的供体转移仍然存在。因此行为转向更强
干扰，不能被解释成状态转移机制已经消失。

禁止升级为：

- K5 Rescue 的 Aligned−Mismatched 已统计确认反转；
- K5 Loss 的 relation effect 已证明等价于零；
- overall accuracy、完整序列组织或 long-term memory；
- progressive protocol 的每一阶段都完成行为或 donor-swap 复验；
- K5 exact-B 等于 progressive trajectory 的 stage 5。

## 3. 必要论证链

`a 递归改写存在 → b 累积历史改变行为 → c 状态组成仍在 → d 残差落到真实事件 → e 继承状态仍能因果进入下游`

- a 建立 successive input 相对 equal-time passive 的逐阶段状态改写；
- b 显示 accumulated history 的行为代价；
- c–e 用状态组成、事件落点与因果供体转移排除“机制已经消失”。

progressive a 与 fixed-B b–e 是互补协议，不共享数值横轴，也不通过箭头或
连续底纹伪装成同一批连续 trial。

## 4. 面板合同

### a. Successive-input state displacement

**结论**：stage 2–10 中，真实下一输入持续造成远大于同父状态、等时 passive
continuation 的 joint `u/x` state displacement。

**图型**：单一有序阶段轨迹。

- 横轴：`Stage`，整数 `2–10`；
- 纵轴：`State displacement`，线性范围 `0–0.65`；
- 蓝色实线圆点：Observed；
- 灰色虚线方点：Equal-time passive；
- 每阶段显示 20-network 均值及 persisted bootstrap 95% CI；
- 两条线末端直接标注，无顶部图例。

禁止对数轴、断轴、面积填充、20 条 network spaghetti、`u/x/joint` 三套
轨迹、数值或显著性批注。推断继续基于配对 observed-minus-passive，图面
保留两个原始条件以直接显示反事实。

### b. K1→K5 opportunity-conditioned behavior

**结论**：累积历史降低 relation-balanced Rescue，并提高 relation-balanced
Loss。

**图型**：单一四柱分组柱状图。

- 横轴：`K1`、`K5`；每组各有 Rescue、Loss 两柱；
- 纵轴：`Rate (%)`，范围 `0–70`；
- Rescue 蓝色，Loss 橙色；
- 每柱显示 20-network 均值及 persisted bootstrap 95% CI；
- 只有一个 panel-local `Rescue / Loss` 图例。

只允许比较同一 outcome 从 K1 到 K5 的变化。Rescue 与 Loss 使用不同
opportunity 分母，禁止互相相减或解释为净准确率。图面不显示 delta、柱顶
数字、显著性、network 点或 K5 Aligned/Mismatched contrasts。

### c. K5 state components

**结论**：K5 下 same-B common update 与 history residual 均仍存在。

**图型**：沿用 Fig.2c 的单坐标两柱阈值余量图。

- 横轴：`Common update`、`History residual`；
- 纵轴：`Value − threshold`，范围 `−0.05–0.45`；
- `y=0` 为预设阈值参照；阈值分别为 `0.5` 与 `0.05`；
- Common update 蓝色，History residual 洋红色；
- 20-network 均值及 supplied confirmatory bootstrap 95% CI。

阈值定义、检验和 multiplicity 只进入 caption／metrics；图内无图例、点云、
数值或机制解释句。

### d. K5 event-linked residual

**结论**：K5 history residual 仍集中在真实 changed spike events，而不是
count-matched random coordinates。

**图型**：沿用 Fig.2d 的直接两条件柱状图。

- 横轴：`Matched random`、`Changed events`；
- 纵轴：`Residual magnitude`，范围 `0–0.05`；
- random 为灰色，changed events 与 c 的 residual 同为洋红色；
- 20-network 均值及 persisted bootstrap 95% CI。

不显示 enrichment ratio、forest、seed 横轴、cell/event 点云、数字或文字
批注。

### e. K5 downstream donor transfer

**结论**：K5 下 inherited L1 state 仍能因果进入 L2 update 与 early output。

**图型**：单一 network point-range 坐标系。

- 横轴：`L2 update`、`Early score`；
- 纵轴：`Donor-transfer index`，范围 `0–1`；
- 每个 endpoint 显示 20 个独立 network 点；
- 前景显示 network 均值及 supplied confirmatory bootstrap 95% CI；
- 两 endpoint 不相连，统一使用 donor-transfer 橙色，以圆／方 marker 冗余
  区分，无图例。

network 点用于呈现 20/20 阳性与 Early score 的网络间变异，不允许加入
trial/cell 点或显著性。

## 5. 冻结布局

画布 `165 mm × 102 mm`；外边距 `2 mm`，横向 gutter `2 mm`，排间距
`2 mm`。

| 面板 | slot_bbox，mm | 初始 plot_bbox，mm |
|---|---|---|
| a | `x=2, y=2, w=79.5, h=48` | `x=15, y=9, w=63.5, h=31` |
| b | `x=83.5, y=2, w=79.5, h=48` | `x=96.5, y=9, w=63.5, h=31` |
| c | `x=2, y=52, w=52.333, h=48` | `x=15, y=59, w=36.333, h=30` |
| d | `x=56.333, y=52, w=52.334, h=48` | `x=69.333, y=59, w=36.334, h=30` |
| e | `x=110.667, y=52, w=52.333, h=48` | `x=123.667, y=59, w=36.333, h=30` |

a/b 的 slot 与 plot 均等宽，plot top、bottom 与 x-axis baseline 对齐；c/d/e
同样对齐。第一排表达
“递归改写前提 → 行为后果”，第二排表达“状态组成 → 事件落点 → 下游作用”。
两排不构成对应列，因此明确释放跨排左右 slot／plot 边界对齐；第一排等分只
表达两个连续论证步骤，不把 a/b 与 c/d/e 暗示为同一批连续协议。若真实 9 pt
字体测量需要不超过 1 mm 的修正，必须先回写 layout contract，再重绘，禁止
成图后手工拖动。

## 6. 视觉合同

- panel letter：小写 `a–e`，12 pt bold；其余文字 9 pt normal；
- 只保留左、下轴，轴线 `0.6 pt`；无网格、顶部／右侧 spine、灰底框、
  panel title 或解释句；
- 图内统一使用 `%`，不使用 `pp`；
- 图内不显示 `n`、p 值、Holm 或显著性星号；
- 颜色：Observed/Common/Rescue `#0072B2`；Loss/Donor `#D55E00`；
  Residual/Changed events `#CC79A7`；Passive/Random 为中性灰；
- PNG 300 dpi，PDF/SVG 保留可编辑矢量文字。

## 7. 统计与排除

- 独立重复单位始终为 network；所有 trial、anchor、event、coordinate 与
  stage 均先按预定层级汇总到 network；
- candidate-derived CI：20,000-draw percentile network bootstrap；
- candidate-derived tests：network-level exact sign-flip，并保留既定 Holm；
- fixed-B confirmatory endpoints 保留 supplied bootstrap CI、exact inference
  与原 multiplicity；
- `K5_rescue_aligned_minus_mismatched` (`p_Holm=0.06039`) 与
  `K5_loss_aligned_minus_mismatched` (`p_Holm=0.11949`) 只进入审计 metrics；
- depth × relation interactions 只进入 statistics／Source Data，不进入主图；
- 不按效果大小筛选 network、stage 或 endpoint。

## 8. 验收标准

- 阅读顺序能够复述为：`recurrence → behavioral cost → persistent state component → event linkage → causal downstream access`；
- 五个 panel 均为一个字母对应一个完整坐标系，无 inset 或微子图；
- a 显示 Observed 与 Equal-time passive，而不是只让读者推测差值；
- b 只需要同 outcome 的 K1→K5 比较；
- c/d 与 Fig.2c/d 保持相同坐标语法和尺度；
- e 的 raw marks 只代表 20 个独立网络；
- progressive a 与 fixed-B b–e 的协议边界在 caption 和 Source Data 中明确；
- PNG、PDF、SVG 在最终尺寸下无裁切、碰撞、编码损坏或颜色语义冲突。
