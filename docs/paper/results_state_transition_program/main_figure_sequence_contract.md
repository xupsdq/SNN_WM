# 六张主图顺序合同

## 状态

- Fig.1：角色与面板已冻结。
- Fig.2：角色、面板与布局保持冻结，不增加 history-rewrite bridge 面板。
- Fig.3：a–f 定量机制链与末端全宽综合图 g 已冻结；详见 `fig3_panel_contract.md`。
- Fig.4：四个定量面板 a–d 与两排两列布局已冻结；详见 `fig4_panel_contract.md`。
- Fig.5：科学逻辑、a–f 面板与布局已冻结为 structural organization；详见 `fig5_panel_contract.md`。
- Fig.6：科学逻辑、a–f 面板与布局已冻结为 functional access；详见 `fig6_panel_contract.md`。
- 网络口径：所有正文定量图统一使用 `seed_1000`–`seed_1019`，共 20 个独立训练网络。
- 数据边界：只使用已持久化结果，不增加训练、模拟或实验。
- history-rewrite bridge：明确退出正文主图链，不再作为独立 Fig.3，也不并入 Fig.2 的定量面板。
- 当前计数：六张主图全部成立并冻结；Fig.5 与 Fig.6 分别拥有独立问题、独立证据链和独立终点。
- 冻结日期：2026-08-10。

## 1. 本轮顺序决定

原计划的新 Fig.3——post-B → identical-C history-rewrite bridge——取消正文主图身份。

取消原因不是单纯“只有三个结果面板”，而是它没有形成独立于 Fig.2 的新论证层级：

1. post-B/passive boundary separation 主要用于定义 donor 与 receiver，不是一个独立的论文结论；
2. C-induced Layer 2 donor transfer 与 early-output donor transfer 沿用了 Fig.2e 已经建立的同一类因果读出；
3. 去掉协议示意后，剩余证据共同回答的只是“Fig.2 的作用是否还能延后一拍”，而不是新的机制、推广或功能结果；
4. 把这三块结果塞进 Fig.2 只会让 Fig.2e 重复两遍，并破坏 Fig.2 已冻结的最小一步转移结构。

更关键的是，这组 bridge 的完整 protocol 只覆盖 `seed_1001`–`seed_1019`。已有 `seed_1000` 是 development/engineering run，仅包含 8 个 cells，而其他每个网络包含 1,000 个 cells。二者不能作为同协议的 20 个网络并列汇总。在不补实验的边界下，无法得到合格的第 20 个 comparable network。

因此，正确处理不是给薄弱 Fig.3 增加装饰面板，也不是把不等价的 `seed_1000` 强行补入，而是将整个 bridge 从正文定量证据链中移除。

## 2. 已冻结的六图主链

history-rewrite bridge 退出后，主图数量并不是靠把同一批结果换编号补回。六图之所以仍然成立，是因为终点证据经逐项审查后确实分成了两个不可互相替代的论证层级：

- Fig.5 不使用任何 cue、行为 readout 或 firing perturbation，独立回答 repeated transition 形成了什么内部组织；
- Fig.6 的每个面板都含实际 cue、readout 或 intervention，独立回答该组织何时能够被后续输入利用。

冻结顺序为：

```text
Fig.1  可继承的活动静默 STSP 状态
  ↓
Fig.2  相同当前输入上的一次 history-conditioned inter-layer transition
  ↓
Fig.3  overlap-gated firing、局部竞争与 downstream write-back
  ↓
Fig.4  同一转移原则在连续输入和累积历史中反复发生
  ↓
Fig.5  反复转移形成结构化的多项目 STSP 组织
  ↓
Fig.6  结构化状态可被后续输入访问，并具有明确 operating boundary
```

六图依次回答：

`状态从哪里来 → 一次转移是否存在 → 一次转移如何实现 → 转移能否反复发生 → 反复转移形成什么结构 → 该结构是否仍能被使用`

六图分别承担：

`inherit → transition → implement → recur → organize → access`

这里的“拆为结构与功能”不是按照结果数量平均分配，而是按照推理终点拆分：结构指标不能证明可用，功能 readout 也不能代替对内部组织的定义。两张图只有同时存在，全文才能从 repeated transition 完整走到 history-dependent processing。

## 3. Fig.2 的边界保持不变

### 唯一问题

不同继承历史面对完全相同的 B 时，是否产生由 B 主导、受历史条件化的 successor update？

### 冻结证据

- 一步 DMS 与 exact-B 反事实；
- rescue 与 loss 的双向行为改变；
- common B-driven update 与历史残差 `Γ`；
- event–`Γ` enrichment；
- pre-B Layer 1 inherited `u/x` 对 B-induced Layer 2 update 和 early output 的 donor transfer。

### 冻结终点

B 在继承状态上被处理，并写出 input-driven、history-conditioned downstream successor update。

Fig.2 不增加 post-B/passive boundary、same-C Layer 2 transfer 或 same-C early-output transfer。Fig.2 的最后一句只提出“这一基本转移能否反复发生”，不提前用一组三面板 bridge 作答。

## 4. 新 Fig.3：一次状态转移如何在局部实现

### 唯一问题

继承状态通过什么空间选择规则进入 later-input processing，并被转换为 downstream write-back？

### 冻结中心结论

历史 STSP 的作用不是均匀施加在所有位置，而是在 retained support 与 current-input pathway 重叠的位置进入早期 firing；由此改变 recruitment、局部竞争和下一层 STSP updating。

### 冻结七面板链

`a overlap-specific causal gate → b pre-input retained support → c advance/recruit conversion → d Layer-1 STSP necessity → e Layer-2 write-back → f Layer-1-only successor transfer → g state-evolution synthesis`

新 Fig.3 合并原 overlap-reentry 与原 local-support/competition 的同链结果。a–f 为三排两列的定量证据；g 作为第四排全宽综合图，只在证据之后建立统一状态演化读者模型。

## 5. 新 Fig.4：基本转移是否会反复发生

### 唯一问题

Fig.2 建立、Fig.3 解释的一次状态转移，是否会在连续输入到来时反复发生，使状态持续偏离 matched-passive evolution？

### 冻结四面板链

`a C5 early Layer-2 donor transfer → b post-C Layer-3 successor transfer → c observed-versus-passive recurrence → d K1/K5 Rescue/Loss`

### 权威既有来源

- `results/causal_closure_multi_seed_20260803/c5_l2_successor/`
- `results/paper_figure_multi_seed/fig4_accumulated_history_statistics/`

这些结果完整覆盖 `seed_1000`–`seed_1019` 的 20 个网络。Fig.4 不设置示意面板；C5 干预身份由图注、Methods、Source Data 与 provenance 记录。具体面板、限定和两排两列布局以 `fig4_panel_contract.md` 为权威。

### 允许的终点

连续输入后的 inherited state 显著偏离 matched-passive trajectory；相同 C5 干预在 early Layer-2 processing 与 post-C Layer-3 successor 上均保留 donor-directed transfer，且累积历史系统性改变 Rescue/Loss。

### 禁止越界

- 不能只凭状态数值随 stage 变化就称为 repeated transition；
- 必须保留 observed-next 与 matched-passive 的同阶段反事实；
- 不能把 `x` 的 early-minus-late 非正结果隐藏后宣称所有变量均以同样速率累积；
- K5 只作为累积历史对应证据，不能重复 Fig.2 的全部 K1 面板。

## 6. Fig.5 与 Fig.6 的冻结拆分

### Fig.5：反复转移形成什么结构

**唯一问题**：

反复转移后的 STSP state 是否保留多个历史 constituent，并形成对真实经历组合特异、超出简单线性叠加、具有序列组织且受容量／延迟限制的内部状态？

**冻结六面板链**：

`a 双 constituent 保留 → b true-pair specificity → c residual specificity beyond linear mixture → d multi-item N_eff → e serial-position weights → f structural K × delay boundary`

**冻结终点**：

Repeated transitions organize recent history into a distributed, history-specific and capacity-limited STSP state，而不是 overwrite、随机漂移或无结构加和。

Fig.5 全部为结构证据，不出现 partial cue、target probability、rescued fraction、recruitment 或 perturbation。具体面板与跨层限定以 `fig5_panel_contract.md` 为权威。

### Fig.6：结构化状态是否仍能被使用

**唯一问题**：

Fig.5 的结构化 state 能否在 incomplete later input／cue 下改变 target access 与 early recruitment，以及这种访问受哪些内容、负荷、延迟和 overlap 条件限制？

**冻结六面板链**：

`a pair partial-cue recovery → b multi-item serial access → c cue-content specificity → d functional K × delay boundary → e targeted high-STSP-overlap ablation → f STSP × overlap interaction`

**冻结终点**：

The structured state remains functionally accessible, but its expression requires an incoming content- and overlap-matched input and is constrained by sequence load and retention delay.

Fig.6 不重复 Fig.3 的一步局部机制，也不使用不稳定的 `support_gain_corr` 强行把结构和功能压成一个相关性。具体面板与限定以 `fig6_panel_contract.md` 为权威。

### 为什么两图都不可删除

- 删除 Fig.5，全文会从 repeated displacement 直接跳到 cue readout，却没有定义被访问的 state 具有什么内部组织；
- 删除 Fig.6，全文只能证明状态具有结构，却不能证明这种结构会参与后续 processing；
- 合并两图会迫使 pair specificity、multi-item morphology、cue specificity、functional boundary 和 causal interaction 争夺同一有限版面，最终只能变成结果清单；
- 当前拆分中，Fig.5 与 Fig.6 没有重复 endpoint，各自都能以独立一句中心结论结束，因此六图不是为了维持数量而拆分。

## 7. “之前的 Fig.3”编号辨析与逐项归宿

讨论过程中曾有两批不同结果先后被称为 Fig.3，必须分开处理。

### 7.1 最早的 overlap/local-mechanism Fig.3

这批结果没有降级为补充材料。它们已经被扩充并重组为现在的 Fig.3 主图：

| 原结果 | 当前归宿 |
|---|---|
| overlap-specific Layer 1 intervention | 新 Fig.3a |
| overlap-dominant pre-input support | 新 Fig.3b |
| advance/recruit transition | 新 Fig.3c |
| Layer-1 STSP attenuation/reset | 新 Fig.3d |
| Layer 2 history-dependent write-back | 新 Fig.3e |
| Layer-1-only → Layer-2 successor transfer | 新 Fig.3f |
| local winner–loser trace | statistics／Source Data，不进入当前 artwork |
| state-evolution synthesis | 新 Fig.3g，证据后综合 |

当前 Fig.3 由六个定量面板闭合机制证据，再由末端全宽 g 建立统一状态演化读者模型；g 不替代任何定量面板。

### 7.2 后来临时前移为 Fig.3 的 history-rewrite bridge

这批结果不是“全部降级为补充图”，而是退出 manuscript-facing figures：

| 原 bridge 面板/结果 | 当前归宿 | 原因 |
|---|---|---|
| History → B → post-B → same-C 协议示意 | 不进入论文图 | 所服务的 bridge 定量结论已退出正文；单独保留示意没有证据价值 |
| post-B vs matched-passive boundary displacement | 内部 protocol/construct validation | 主要定义 donor 与 receiver，不构成独立结论 |
| C-induced Layer 2 donor transfer | 内部实验结果 | 与 Fig.2e 属同类因果读出，且没有 20 个同协议网络 |
| C early Layer 3 donor transfer | 内部实验结果 | 与 Fig.2e early-output transfer 重复，且没有 20 个同协议网络 |
| K1/K5 joint-positive inference | 内部统计审计 | 不能把 development run 与正式 cohort 合并为 20 网络 |
| state-transition loop 示意 | 删除，不另行安置 | 是结论复述，不是独立证据 |

`results/paper_figure_multi_seed/archive/history_rewrite_bridge` 保留为实验审计与工程记录，不删除、不改写其父级结果，也不再作为正文或补充材料的 20 网络定量证据。

不允许把 development `seed_1000` 与正式 cohort 并列得到名义上的“20 网络”重汇总。该目录中的原 protocol 描述作为 provenance 保留，但不得进入 manuscript-facing figure、caption、Results claim 或统计报告。

## 8. 图间禁止重复

### Fig.2 与 Fig.3

- Fig.3 不重复 rescue/loss、common cosine、`Γ` 或 Fig.2 的双端点 B-induced donor transfer；
- Fig.2 不提前展开 overlap、advance/recruit、STSP necessity 或 write-back location。

### Fig.3 与 Fig.4

- Fig.3 解释一次 later-input processing 的空间入口、firing conversion、Layer-2 写回及 K1 successor transfer；
- Fig.4 证明 transfer 延伸至 C5 early processing/post-C successor，并在 successive stages 反复发生；
- Fig.4 不重新展示 Fig.3 的 overlap-specific reset、first-50-ms STSP necessity 或 Layer-2 update-composition panel。

### Fig.4 与 Fig.5

- Fig.4 的终点是 repeated displacement beyond passive；
- Fig.5 的起点是多次转移后的结构化状态；
- Fig.5 不用 stage displacement 换一种版式重复“状态在变化”。

### Fig.5 与 Fig.6

- Fig.5 只证明结构，任何面板都不使用 cue/readout；
- Fig.6 只证明访问、因果贡献及 operating boundary，每个面板都包含 cue、readout 或 intervention；
- Fig.5f 与 Fig.6d 使用对齐的 K × delay 网格，但分别绘制 structural `N_eff_fraction` 与 functional `rescued_fraction`；这是结构边界与功能边界的并列，不是重复 readout；
- 不设置不稳定的结构—功能相关性作为“连接面板”，也不在两图中重复绘制同一 endpoint。

## 9. 资产与编号迁移原则

| 既有结果身份 | 新编号/归宿 | 处理原则 |
|---|---|---|
| Fig.1 silent inherited state | Fig.1 | 保持 |
| fixed-B K1 single-transition evidence | Fig.2 | 保持五面板冻结结构 |
| history-rewrite bridge | 退出正文 | 仅保留实验 provenance，不进入 20 网络主图 |
| overlap-reentry + local-support/competition | Fig.3 | 合并为六个定量面板，并在末端加入唯一的状态演化综合图 g |
| progressive/repeated-transition evidence + C5 successor transfer | Fig.4 | 冻结为四个定量面板；不保留示意图 |
| pair/multi-item structural organization | Fig.5 | 冻结为六面板结构图，不使用功能 readout |
| partial-cue access、perturbation 与 operating boundary | Fig.6 | 冻结为六面板功能图，不重复结构 endpoint |

现有结果目录、runner 和旧 YAML 名称暂不重命名。它们是数据与实现身份，不是当前科学编号的权威。Fig.1–Fig.6 的科学编号与面板职责以本合同及各 `figX_panel_contract.md` 为准；后续迁移 figure specs、adapters、renderers 和输出路径时只能读取既有父级结果。

## 10. 20 网络统一规则

- 所有 manuscript-facing 定量面板必须显示或汇总 `seed_1000`–`seed_1019` 的 20 个可比网络。
- 不允许把不同 trial count、cell count、history family、anchor count 或开发角色的网络作为同协议重复并列。
- 不允许为了达到 20 而把 development/smoke run 补入 confirmatory cohort。
- 若现有结果不能提供 20 个可比网络，在“不补实验”前提下，该结果退出正文主图，而不是降回 19 网络口径。
- 任何从既有完整 20 网络结果进行的 plot-only/network-level 汇总都不得重新运行模拟或改变父级 artifacts。

## 11. 顺序验收标准

- Fig.1–Fig.6 的 `inherit → transition → implement → recur → organize → access` 已全部确定。
- 已取消的 bridge 不再以独立图、Fig.2 子面板或补充定量图的形式回流。
- Fig.2 保持已冻结的一步状态转移，不因删除 bridge 而膨胀。
- Fig.3 使用六个定量面板闭合局部机制，并仅在末端 g 进行证据后综合。
- Fig.4 使用四个定量面板保留 C5 early/successor transfer、observed-versus-passive recurrence 与累积 Rescue/Loss。
- Fig.5 使用六个结构面板回答形成何种组织，不含 cue、readout 或 firing perturbation。
- Fig.6 使用六个功能面板回答该组织是否可访问，并同时保留 content specificity、K × delay boundary、targeted ablation 与 overlap interaction。
- Fig.5 的最后一句自然提出结构是否可访问，Fig.6 用 later-input effect 与 operating boundary 作答。
- 不以 `support_gain_corr`、score trend 或装饰性示意图维持 Fig.5/Fig.6 的拆分；两图分别依靠自身的最小必要证据链成立。
- 所有主图只使用已持久化结果，不增加任何实验。
