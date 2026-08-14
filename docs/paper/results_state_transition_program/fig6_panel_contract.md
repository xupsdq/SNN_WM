# Fig.6 面板与布局契约

## 状态

- 科学逻辑：已冻结。
- 面板集合：已冻结为 a–f，全部为定量结果面板，不设置示意图。
- 布局拓扑：三排混合网格；第一排把 Target A、Target B 收拢为一个紧凑 a 组，
  并为含 10 个 serial positions 的 b 分配更宽数据区；第二排 c–d、第三排 e–f
  各自两列等宽。三排等高。
- 网络口径：全部使用 `seed_1000`–`seed_1019`，共 20 个独立训练网络。
- 数据边界：只读取已有 partial-cue、multi-item access、K × delay boundary、targeted ablation 与 overlap-interaction 结果；不增加训练、模拟或实验。
- 整图角色：作为全文终点，只证明 Fig.5 的结构化 state 能否被 later input 使用，以及这种使用的内容、负荷、延迟和空间边界。
- 实现状态：本文件是新 Fig.6 的科学权威；既有 `fig2`、`fig3`、`fig6` 结果目录和旧 YAML 名称只是数据来源身份。
- 当前实现冻结日期：2026-07-31。

## 1. Fig.5 到 Fig.6 的唯一推进

Fig.5 已经证明 repeated transition 形成结构化、多成分且受容量限制的 STSP state。但结构相似度、pair specificity、`N_eff` 和 item weights 都不能证明该状态会参与后续 processing。

Fig.6 因而只回答：

> 当一个不完整的 later input 或 cue 到达时，结构化 state 是否能够选择性恢复其历史成分、提高多项目访问并改变早期 recruitment；这种访问又在哪些内容、负荷、延迟和 overlap 条件下成立？

## 2. 整图中心结论

两项目 state 在 partial cue 下提高对 A 与 B 的恢复，并优于无历史状态及相应 singleton state；多项目 sequence state 同样提高跨序列位置的 target access。该作用对 cue 内容具有选择性，并在 K × delay 空间中呈现清楚的功能 operating boundary。对 high-STSP-overlap sites 的定向移除比 matched removal 造成更大的 recruitment loss，而 STSP support 只有在 later input 与其支持通路重叠时才被转化为早期 firing change。

允许的终点表述是：

> The structured STSP state remains functionally accessible, but its expression requires an incoming cue or input that enters content- and overlap-matched pathways and is limited by sequence load and retention delay.

不能升级为“STSP 自身无需输入即可回放记忆”“STSP 单独预测最终类别”或“所有保留项目都能被完美读取”。

## 3. 必要论证链

`a 最小 pair state 可被 partial cue 读取 → b 访问推广到多项目与序列位置 → c 访问对 cue 内容具有特异性 → d 功能访问具有 K × delay 边界 → e 结构化 support 对 recruitment 具有因果贡献 → f overlap 决定该 support 是否被表达`

六个面板分别封闭不同缺口：

- 没有 a，Fig.5 的最小 pair organization 没有直接功能出口。
- 没有 b，功能结论停在两项目，不能与 Fig.5 的多项目结构对应。
- 没有 c，提高的 readout 可能只是一般 excitability，而不是对历史内容的选择性访问。
- 没有 d，只能证明两个预选工作点，无法说明负荷和延迟如何共同限制访问。
- 没有 e，a–d 仍主要是状态条件与 readout 的比较，缺少对 retained support 的定向因果检验。
- 没有 f，targeted removal 有效仍不能说明为何某些 later inputs 能访问该 state；overlap interaction 给出表达边界。

## 4. 面板契约

### a. Pair state 在 partial cue 下恢复两个 constituent

**角色**：把 Fig.5a–c 的最小结构结果直接连接到最小功能 readout。

**必要端点**：

- `SAB_vs_S0_auc_gain`；
- `SAB_vs_relevant_single_auc_gain`；
- target item A 与 B。

**首选编码**：沿既有 keep-probability sweep 显示四种 state condition 的恢复曲线，
并在同一面板内按 `Target A / Target B` 左右拆分，共用纵轴和图例。该内部拆分由
用户于 2026-07-31 明确授权，是为了恢复旧 Fig.2f 中最能说明剂量反应的视觉语法；
它是 Fig.6a 的面板特例，不构成全图或后续图的默认规则。20-network 均值与 95% CI
用于曲线，AUC contrasts 继续作为预定义推断端点写入 metrics／Source Data，避免
用单个事后挑选的 cue 强度作为主结论；不叠加 network 点云。

**持久化来源**：

`results/paper_figure_multi_seed/fig2_pair_fused_stsp_state/fig2_pair_fused_stsp_state/seed_*/data/metrics/panel_f_partial_cue_auc_metrics.csv`

**20 网络既有结果方向**：

- pair state 相对 S0 对 A、B 的 AUC gain 均为正；
- pair state 相对相应 singleton state 的增益也为正，表明功能不只是复制一个 constituent trace。

**限定**：a 证明 partial-cue recovery，不证明无 cue 自发回放，也不把 A、B 恢复率解释为同时输出两个类别。

### b. Multi-item sequence state 提高跨序列位置的 target access

**角色**：把 a 的两项目访问推广到更长历史。

**预先存在的聚焦条件**：

- `seq_len = 10`；
- `delay_ms = 400`；
- 全部 target positions。

该 K10/D400 条件来自既有 figure protocol，不是根据本次图面事后挑选；完整 K × delay 结果由 d 给出。

**必要比较**：

- `P_target_sequence_state`；
- `P_target_single_item_memory`；
- 主图直接差值固定为
  `sequence_minus_singleton_access_gain = P_target_sequence_state − P_target_single_item_memory`。

源文件中的既有 `G_i` 被验证为 `sequence − cue-only`，不能在只显示 sequence
与 singleton 的图中继续命名为两者的 Gain。

**首选编码**：沿 serial position 同时显示 `Cue only / Singleton / Sequence` 三条
绝对 readout 轨迹，以直接恢复旧 Fig.3b 的比较结构；显示 20-network 均值与 95%
CI，不叠加 network 轨迹。`sequence − singleton` 仍是预定义核心 contrast，保留在
metrics／caption，而不再作为第四条视觉系列。单个 sequence 或 cue trial 不作为
独立点。

**持久化来源**：

`results/paper_figure_multi_seed/fig3_multiitem_peak_landscape/seed_*/data/metrics/panel_d_item_functional_gain.csv`

**既有结果方向**：sequence state 在多个 serial positions 上提高 target access，尤其保留 singleton trace 已衰减的位置。

**限定**：b 证明 sequence-state access advantage，不称为逐项目无损 recall，也不把每个 serial position 当作独立重复。

### c. Access 对 cue 内容具有选择性

**角色**：排除 a、b 的增益只是全局兴奋性提高或任意 weak cue 都能触发相同输出。

**预先存在的聚焦条件**：

- `seq_len = 7`；
- `delay_ms = 400`；
- `state_condition = S_final`。

**必要比较**：

- matched cue；
- mismatched seen-item cue；
- unseen cue。

**核心 contrasts**：

- matched minus mismatched；
- matched minus unseen。

mismatched 控制“来自同一历史集合但不是目标内容”的输入，unseen 控制“未被当前历史支持”的输入；二者回答不同混淆，不能只保留一个。

**首选编码**：沿 serial position 同时显示 `Matched / Mismatched / Unseen` 三条
绝对 readout 轨迹，以直接恢复旧 Fig.3c 的比较结构；显示 20-network 均值与 95%
CI，不叠加 network 轨迹。`matched − mismatched` 与 `matched − unseen` 两个
预定义 network-level contrasts 继续进入 metrics／caption；面板仍只有一套坐标系。

**持久化来源**：

`results/paper_figure_multi_seed/fig3_multiitem_peak_landscape/seed_*/data/metrics/panel_c_cue_specificity_metrics.csv`

**20 网络既有结果方向**：matched cue 相对 mismatched 与 unseen cue 的 target access 均更高。

**限定**：mismatched cue 仍可能访问其他 seen items，因此 c 的结论是 target-content specificity，不是“mismatched 完全无记忆作用”。

### d. 功能访问具有 K × delay operating boundary

**角色**：将 a–c 的工作点推广到完整负荷和保持时间空间。

**核心端点**：

`rescued_fraction`

其定义为 sequence state 相对 slot-matched singleton state 恢复的项目比例，而不是总 accuracy 或结构 `N_eff`。

**必要维度**：

- `seq_len = 3, 5, 7, 10`；
- `delay_ms = 100, 200, 400, 800`。

**首选编码**：K × delay 热图，横轴为 K、纵轴为 delay，单元格标注两位小数，
色阶固定为 rescued fraction 的自然范围；水平色条置于数据区上方，作为与图例
同级的顶部装饰；色条从上到下依次为文字标签、刻度数字和色带。布局方向与
Fig.5f 对齐，使读者可以比较结构边界与功能边界的
位置，但两图使用不同指标、不同层级和不同色标。

**持久化来源**：

`results/paper_figure_multi_seed/fig3_multiitem_peak_landscape/seed_*/data/metrics/panel_f_boundary_summary.csv`

**既有结果方向**：

- sequence-state rescue 在多种 K × delay 条件下存在；
- load 与 delay 共同改变 rescued fraction；
- 既有 20 网络推断支持 sequence-length × delay interaction。

**限定**：d 不把 structural `N_eff` 和 functional rescued fraction 合并成单一“memory strength”，也不使用不稳定的 `support_gain_corr` 强行建立逐格相关。

### e. High-STSP-overlap support 对早期 recruitment 具有定向因果贡献

**角色**：对 retained support 的功能作用提供 targeted perturbation，而不只比较不同 state conditions。

**必要比较**：

- 移除 high-STSP-overlap input sites；
- 移除数量匹配的 control sites。

**核心端点**：

`high_stsp_overlap_minus_matched_loss`

**首选编码**：直接显示 `High-overlap removal` 与 `Matched removal` 两个
network-level recruitment-loss 条件及 95% CI；配对 difference 与推断保留在
metrics／caption。sequence 和 probe 先在网络内汇总。

**持久化来源**：

`results/paper_figure_multi_seed/fig6_peak_amplified_reentry/seed_*/data/metrics/supp_s11f_high_stsp_ablation_paired_difference.csv`

**20 网络既有结果方向**：high-STSP-overlap removal 造成的 recruitment loss 大于 matched removal。

**与 Fig.3a 的非重复性**：

- Fig.3a 在最小一步 DMS 中重置 inherited Layer 1 STSP，证明一次 transition 的空间入口；
- Fig.6e 在 multi-input state 后移除 later-input sites，检验结构化 support 对功能访问的贡献；
- 两者的历史深度、干预对象和 endpoint 均不同。

**限定**：e 证明被定向移除的 supported entry sites 对 recruitment 有贡献，不宣称这些 sites 单独编码最终类别。

### f. Later-input overlap 决定 retained STSP 是否被表达为 firing change

**角色**：给全文的功能结论设置最后也是最关键的局部边界：高 STSP support 并不自动产生 firing，必须有输入进入受支持通路。

**必要 2 × 2 结构**：

- high vs low retained STSP；
- overlap vs no-overlap later input。

**核心端点**：

`interaction_delta = stsp_effect_with_overlap - stsp_effect_without_overlap`

**冻结主协议**：

- `stsp_group_quantile = 0.50`；
- `overlap_threshold = 0.05`；
- primary early window = 10 ms；
- 5、15、20 ms 仅作同面板小型时间窗稳健性或补充材料，不增加独立推断。

**首选编码**：单坐标显示 10 ms 的 2 × 2 interaction，使“高低 STSP 差异只在
overlap 下出现”可直接读取；interaction contrast 与其他时间窗进入
metrics／Source Data，不设置 inset 或右侧第二坐标系。

**持久化来源**：

`results/paper_figure_multi_seed/fig6_peak_amplified_reentry/seed_*/data/metrics/panel_e_overlap_gated_stsp_interaction.csv`

**20 网络既有结果方向**：interaction 在预定义时间窗内为正；no-overlap 条件下 high-STSP effect 不足以产生同样 firing change。

**限定**：f 的结论是 overlap-gated expression，不是 STSP 单独决定 firing，也不是 overlap 本身等同于记忆 route。

## 5. 冻结布局

画布为 `165 mm × 152 mm`，外边距 2 mm，横向 gutter 2 mm，排间距 2 mm；
三排均为 48 mm 高。第一排采用用户明确确认的 `94:65 mm` 分组：Fig.6a 占
94 mm，内部两个 38 mm 数据区仅相隔 4 mm；Fig.6b 占 65 mm，并获得 51 mm
数据区。第二、三排各自划分为两个 79.5 mm 等宽面板。
除 Fig.6a 已授权的 Target A/B 内部拆分外，每个 panel 只使用一个数据坐标系。

| 排 | 面板 | 冻结槽位，单位 mm | 逻辑任务 |
|---|---|---|---|
| 第一排 | a | `x=2, y=2, w=94, h=48` | pair partial-cue access；内部 4 mm 紧凑间距 |
| 第一排 | b | `x=98, y=2, w=65, h=48` | multi-item serial access；扩大位置轨迹宽度 |
| 第二排 | c | `x=2, y=52, w=79.5, h=48` | cue-content specificity |
| 第二排 | d | `x=83.5, y=52, w=79.5, h=48` | K × delay functional boundary |
| 第三排 | e | `x=2, y=102, w=79.5, h=48` | targeted causal contribution |
| 第三排 | f | `x=83.5, y=102, w=79.5, h=48` | overlap-gated expression |

**逐排语义**：

- 第一排完成最小 pair access → multi-item access；Target A、Target B 均为
  `38 × 28 mm`，中间仅保留 4 mm 组内间距；b 的数据区为 `51 × 28 mm`，以容纳
  10 个 serial positions 和三系列图例；
- 第二排完成 content-specific control → global operating region；
- 第三排完成 targeted causal contribution → local overlap gate；
- 第一排的 94:65 分组与后两排的二等分没有跨排对应列，因此明确释放其 left/right
  alignment；每一排内部仍保持数据区的上下边界对齐。

全图不设置示意图。Fig.2a、Fig.4a 已经定义任务与转移方向；Fig.6 需要把有限版面全部用于“readout → specificity → boundary → causality”。

## 6. 明确移出 Fig.6 主图的既有结果

| 既有结果 | 归宿 | 原因 |
|---|---|---|
| regional ping 的 old/middle/recent composition | 补充材料 | 结果方向复杂，不能作为内容可访问性的最小主证据 |
| global-ping STSP-score trend | 补充材料 | 相关性证据，且与 Fig.3 的局部 support→firing 机制角色接近 |
| weak-cue STSP-score trend | 补充材料 | 与前项及 Fig.3 重复，不如 matched/mismatched/unseen control 直接 |
| 全量 quantile 与 early-window sweep | 补充材料 | 主图只保留冻结 q=.50、10 ms endpoint，其他用于稳健性 |
| `support_gain_corr` | 不进入主图 | 现有条件间方向不稳定，不能用作结构—功能耦合的核心证据 |
| cue-only 全部类别组成 | Source Data | 主图只保留回答 target access 与 specificity 的必要端点 |

## 7. 与 Fig.3 和 Fig.5 的边界

### Fig.6 不重复 Fig.3

- Fig.3 解释最小一步 transition 的局部实现：support、advance/recruit、competition、write-back；
- Fig.6 检验 repeated transitions 形成的 multi-input state 是否仍可被访问；
- Fig.6 不重复 pre-input support 分组、winner–loser 轨迹或 L2 re-update composition。

### Fig.6 不重复 Fig.5

- Fig.5 的端点是 pair/multi-item state organization；
- Fig.6 的每个主面板都包含实际 cue、target readout、recruitment 或 intervention；
- Fig.5f 与 Fig.6d 的 K × delay 网格刻意对齐，但分别是 structural `N_eff_fraction` 与 functional `rescued_fraction`，不得合并或同色标比较。

## 8. 统计与证据边界

- 所有主图显示与推断统一使用 20 个独立训练网络。
- pair、sequence、target position、cue trial、probe、site、window 和 K × delay cell 均先按预定层级汇总到 network。
- a 的 AUC 来自既有 cue-strength sweep；不得改用单一、效果最大的 keep probability。
- b、c 的焦点条件来自既有冻结协议，不进行事后 condition search；d 展示完整 K × delay 空间。
- e 与 f 是互补因果／interaction 证据，但 endpoint 与数值尺度不同，不能直接比较效应大小。
- 不把 target positions、热图单元格、sites 或 windows 当作独立重复。
- 不用结构—功能逐格相关替代 access comparison、content control 或 intervention。

## 9. 最终验收标准

- 阅读顺序能够自然复述为：`read a pair → generalize access → prove content specificity → map the global boundary → establish causal contribution → identify the local gate`。
- a 的曲线同时显示 S0、两个 singleton 与 pair state，且 metrics 同时保留相对 S0
  与相关 singleton 的 AUC gain，不能只证明“有状态优于无状态”。
- b 使用 slot-matched singleton 对照，不把 cue-only 作为唯一基线。
- c 同时保留 mismatched 与 unseen 两类内容控制。
- d 明确是 functional rescued fraction，不与 Fig.5 的结构指标混写。
- e 使用 high-STSP-overlap removal 与 matched removal 的配对差。
- f 以 interaction 为主结论，不能根据单个 bar 宣称 high STSP 自动驱动 firing。
- 六个面板全部使用或汇总 `seed_1000`–`seed_1019`。
- 全文最后停在“结构化 STSP 对后续输入的条件性利用”，不升级为无 cue 回放、最终标签预测或完美多项目回忆。
