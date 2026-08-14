# Results 统计数据增补实施记录

## 0. 目标与锁定范围

- **源文档**：`docs/paper/v6_results_fig1_fig2_restructured_20260811_results_formatted.docx`
- **源文档 SHA-256**：`daa836499262d068614e2e56012fad0f2b0aa0db67c78ea786c02146b2c8c1e0`
- **实施文档**：`docs/paper/v6_results_fig1_fig2_restructured_20260811_results_formatted_statistics_added.docx`
- **实施文档 SHA-256**：`40cd729b493422844b5264e124382cc4a1b243d0a806b629acf40e68c35f7ae1`
- **本文件状态**：R1–R8 已写入实施文档；作者追加批准的 rescue/loss 差值单位修正亦已完成。源文档未覆盖。
- **变更边界**：现有 Results 论证顺序、结论、段落和图号保持不变；只改动 6 个承载 R1–R8 的 Results 段落，其中 R2 另将两个差值单位由 `%` 修正为 `percentage points`。
- **证据边界**：只使用现有持久化数据；未重跑实验，未新增检验，未把 `descriptive_only` 结果升级为推断性结果。

## 1. 参考文献的数据列举方式及本稿采用规则

参考目标为：

`C:/Users/Administrator/Zotero/storage/G7EFKIEA/Masse 等 - 2019 - Circuit mechanisms for the maintenance and manipulation of information in working memory.pdf`

目标论文在 Results 中采用“**组别/效应估计 → 不确定性或显著性 → 样本量 → 图号**”的紧凑列举方式，例如在同一括号中给出 `R`、`P`、`n` 与 `Fig.`。本稿只借鉴这种信息组织方式，不照搬其统计口径：

1. 本稿继续优先报告跨网络均值或效应量与双侧 95% CI；不改用 `mean ± s.e.m.`。
2. 已有预注册/预声明对比继续报告原有 BH 或 Holm 校正后的 `P` 值。
3. `descriptive_only` 面板只加均值和 95% CI，绝不补造 `P` 值。
4. 每个小节只在首个适合的新增括号中写一次 `n = 20 networks`，避免逐句重复。
5. 只列能关闭当前论证问题的代表性数据：短序列列全关键条件；16 格的 load–delay 网格只列带条件与 CI 的跨条件均值边界。

## 2. 七图编号与数据源映射

新增机制图插入后，当前 v6 的图号与冻结数据包中的旧图号并不相同：

| 当前 v6 图号 | 当前角色 | 权威数据身份 |
|---|---|---|
| Fig. 1 | 新增 STSP 机制图 | `results/paper_figure_redesign_20260811/`；单条确定性机制轨迹，无网络级统计 |
| Fig. 2 | activity-silent inherited state | redesign 包中的 Fig. 2；等价于旧 Fig. 1b–e 重标为 a–d |
| Fig. 3 | identical-input history-conditioned update | 冻结包 `fig2/` |
| Fig. 4 | overlap gate and successor formation | 冻结包 `fig3/` |
| Fig. 5 | recurrent successor updating | 冻结包 `fig4/` |
| Fig. 6 | accumulated-state structural organization | 冻结包 `fig5/` |
| Fig. 7 | functional access | 冻结包 `fig6/` |

冻结包：

`results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/`

特别说明：当前 **Fig. 2a 不是示意图**，而是 recall 定量面板，且正文已经报告 90.95% 与 95% CI，因此按用户要求不再扩充。旧编号中的 Fig. 2a protocol schematic 已顺延为当前 **Fig. 3a**，仍排除统计量。

## 3. 全 Results 覆盖审计与取舍

| 当前面板 | 当前正文状态 | 决定 |
|---|---|---|
| Fig. 1a–d | 确定性机制示意，无独立重复单位 | 排除；不加统计量 |
| Fig. 2a | 已有均值、95% CI、`n = 20` | 保持不变 |
| Fig. 2b | 已明确报告 delay bin firing = 0 | 保持不变；零值不需要推断检验 |
| Fig. 2c | 只有“高于 chance”的范围性叙述 | **R1：补各层最长延迟的均值与 CI** |
| Fig. 2d | 已有两类预测比例及 CI | 保持不变 |
| Fig. 3a | protocol schematic | 排除；不加统计量 |
| Fig. 3b | 已有效应方向、差值和校正后 `P`，但没有两组绝对水平 | **R2：补 aligned/mismatched 组均值与 CI** |
| Fig. 3c | 已有阈值与校正后 `P`，但缺少实际 endpoint 均值与 CI | **R3–R4：补 common cosine 与 residual norm ratio** |
| Fig. 3d | 已有差值、CI、校正后 `P` | 保持不变 |
| Fig. 4a | 已有两组关键对比、CI、校正后 `P` | 保持不变 |
| Fig. 4b–c | 支撑局部化解释的描述性机制背景；逐组概率清单不会增加新的问题关闭 | 不单独堆数值 |
| Fig. 4d | 正文只有因果方向，没有干预效应量 | **R5：补 attenuation/reset 两个效应量、CI、校正后 `P`** |
| Fig. 4e–f | 已有效应量、CI、`P` | 保持不变 |
| Fig. 4g | conceptual synthesis schematic | 排除；不加统计量 |
| Fig. 5a–d | donor transfer、recurrence、rescue/loss 均已有估计、CI、校正后 `P` | 保持不变 |
| Fig. 6a | 只说保留两个 constituent，没有绝对 similarity | **R6：补 A/B constituent similarity 与 CI** |
| Fig. 6b | 已有 true-minus-shuffled 差值、CI、校正后 `P` | 保持不变 |
| Fig. 6c–d | 只有随 K 变化的方向 | **R7：补四个 K 的 `N_eff` 与 latest-item weight 及 CI** |
| Fig. 6e–f | 只有 load–delay 网格方向 | **R8：补两个网格 endpoint 的跨条件均值边界、条件与 CI** |
| Fig. 7a–f | AUC、serial access、cue specificity、boundary、perturbation、interaction 已有估计、CI、校正后 `P` | 保持不变 |

### 推荐执行优先级

1. **A（核心，建议全部执行）**：R2–R7。它们补上行为组绝对水平、阈值检验对应的实际 endpoint、干预效应量，以及 Fig. 6 结构结论的直接数据。
2. **B（重要）**：R1。它补足最长 delay 下三层解码水平，但该段已有 recall 与 firing 数据，因此优先级略低于 A。
3. **C（篇幅允许时执行）**：R8。它让 load–delay 网格不再只有方向性文字；若期刊字数约束要求压缩，应优先删去 R8，而不是删减 R2–R7。

推荐完整版本采用 R1–R8；上述优先级只用于出现严格字数压力时的删减顺序。

## 4. 逐句修改方案

### R1 — Fig. 2c：最长延迟下三层解码水平

**位置**：Results 小节 “Activity-silent STSP forms a functional inherited state”，Fig. 2c 结果句。

**旧句**

> During the same silent interval, item identity remained decodable from the joint u/x state throughout the circuit, exceeding the 10% chance level at every sampled delay from 100 to 1,200 ms (Fig. 2c).

**实施句**

> During the same silent interval, item identity remained decodable from the joint u/x state throughout the circuit, exceeding the 10% chance level at every sampled delay from 100 to 1,200 ms (mean decoding accuracy at 1,200 ms: Layer 1, 81.98% [95% CI, 81.36%–82.60%]; Layer 2, 93.36% [92.94%–93.78%]; Layer 3, 92.69% [92.22%–93.16%]; Fig. 2c).

**新增数据**：最长延迟 1,200 ms 时，Layer 1/2/3 的跨 20 网络均值与 95% CI。只报告最严格时间点，不把 15 个 layer × delay 条件全部塞入正文。

**证据类型**：`descriptive_only`；不添加 `P` 值。

**来源**：`results/paper_figure_redesign_20260811/metrics/fig2_panel_c_statistics.csv`，Layer 1/2/3、1,200-ms 三行。

---

### R2 — Fig. 3b：rescue/loss 的两组绝对水平

**位置**：Results 小节 “An identical input combines common input-driven and history-conditioned updating”，Fig. 3a,b 结果句。

**旧句**

> Relative to mismatched histories, aligned histories increased rescue by 14.1% and reduced loss by 5.1% (rescue, BH-adjusted P = 0.000307; loss, BH-adjusted P = 0.000178; Fig. 3a,b).

**实施句**

> Relative to mismatched histories, aligned histories increased rescue by 14.1 percentage points and reduced loss by 5.1 percentage points (n = 20 networks; rescue, aligned, 47.5% [95% CI, 41.1%–54.0%], versus mismatched, 33.5% [30.5%–36.4%]; BH-adjusted P = 0.000307; loss, aligned, 25.3% [95% CI, 23.0%–27.5%], versus mismatched, 30.3% [29.0%–31.6%]; BH-adjusted P = 0.000178; Fig. 3a,b).

**新增数据**：rescue 与 loss 在 aligned/mismatched 两组中的绝对均值及 95% CI；保留原有配对对比与 BH 校正后的 `P` 值。

**证据类型**：组值为 `descriptive_only`；原有 aligned-minus-mismatched 对比为 `predeclared_recomputed`。

**来源**：`results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig2/metrics/panel_b_statistics.csv`。

---

### R3 — Fig. 3c：common component 的实际均值

**位置**：同一小节，Fig. 3c 第一项 endpoint。

**旧句**

> The common component remained substantial across histories, exceeding the predefined cosine-similarity threshold of 0.5 (BH-adjusted P = 3.64 × 10−55; Fig. 3c).

**实施句**

> The common component remained substantial across histories, exceeding the predefined cosine-similarity threshold of 0.5 (mean cosine similarity, 0.9115; 95% CI, 0.9112–0.9118; BH-adjusted P = 3.64 × 10−55; Fig. 3c).

**新增数据**：实际 mean cosine similarity 与 95% CI；阈值、`P` 和结论全部不变。

**证据类型**：`predeclared_recomputed`。

**来源**：`results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig2/metrics/panel_c_statistics.csv`，`same_B_common_update_cosine`。

---

### R4 — Fig. 3c：history-conditioned residual 的实际均值

**位置**：同一段紧随 R3 的句子。

**旧句**

> At the same time, the history-conditioned residual exceeded its predefined norm-ratio threshold of 0.05 (BH-adjusted P = 1.53 × 10−45; Fig. 3c).

**实施句**

> At the same time, the history-conditioned residual exceeded its predefined norm-ratio threshold of 0.05 (mean norm ratio, 0.3892; 95% CI, 0.3884–0.3900; BH-adjusted P = 1.53 × 10−45; Fig. 3c).

**新增数据**：实际 mean norm ratio 与 95% CI；阈值、`P` 和结论全部不变。

**证据类型**：`predeclared_recomputed`。

**来源**：`results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig2/metrics/panel_c_statistics.csv`，`processing_residual_gamma_norm_ratio`。

---

### R5 — Fig. 4d：early-spike attenuation/reset 的因果效应量

**位置**：Results 小节 “Inherited STSP directs successor formation”，包含 Fig. 4b–e 的长结果句。

**旧句**

> Targeted attenuation or reset likewise reduced the history-associated advancement and recruitment of early spikes (Fig. 4b–d), while dynamic STSP produced stronger history-aligned downstream updating than the static-frozen control (difference-in-differences, 8.52%; 95% CI, 8.46%–8.58%; P = 1.24 × 10−36; Fig. 4e).

**实施句**

> Targeted attenuation or reset likewise reduced the history-associated advancement and recruitment of early spikes (n = 20 networks; mean first-50-ms dynamic-minus-attenuation change, 7.45 percentage points [95% CI, 7.36–7.53], BH-adjusted P = 1.99 × 10−32; dynamic-minus-reset change, 37.27 percentage points [37.01–37.53], BH-adjusted P = 3.85 × 10−36; Fig. 4b–d), while dynamic STSP produced stronger history-aligned downstream updating than the static-frozen control (difference-in-differences, 8.52%; 95% CI, 8.46%–8.58%; P = 1.24 × 10−36; Fig. 4e).

**新增数据**：Layer 1 STSP attenuation 与 reset 对 first-50-ms advance-or-recruit probability 的两个平均效应量、CI 与 BH 校正后的 `P`。

**证据类型**：`predeclared_recomputed`。

**来源**：`results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig3/metrics/panel_d_statistics.csv`。

---

### R6 — Fig. 6a：两个 constituent 的绝对 similarity

**位置**：Results 小节 “Accumulated states remain multi-component and history-specific”，Fig. 6a,b 结果句。

**旧句**

> The state remained similar to both constituent templates (Fig. 6a) and was more similar to the experienced pair than to a one-constituent-held shuffled pair (mean difference in centered-cosine similarity, 0.00666; 95% CI, 0.00663–0.00670; BH-adjusted P = 2.13 × 10−38; Fig. 6b).

**实施句**

> The state remained similar to both constituent templates (n = 20 networks; item A, mean similarity, 0.99118 [95% CI, 0.99114–0.99122]; item B, 0.99501 [0.99498–0.99504]; Fig. 6a) and was more similar to the experienced pair than to a one-constituent-held shuffled pair (mean difference in centered-cosine similarity, 0.00666; 95% CI, 0.00663–0.00670; BH-adjusted P = 2.13 × 10−38; Fig. 6b).

**新增数据**：pair terminal state 对 item A 与 item B template 的跨网络平均 similarity 与 95% CI。

**证据类型**：`descriptive_only`；不新增 `P` 值。

**来源**：`results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig5/metrics/panel_a_statistics.csv`。

---

### R7 — Fig. 6c,d：四个 load 下的 distributed-component 数据

**位置**：同一小节，Fig. 6c,d 结果句。

**旧句**

> Across sequence lengths K = 3, 5, 7, and 10, effective component number increased while the contribution of the latest item declined (Fig. 6c,d), indicating increasingly distributed component expression rather than latest-item domination.

**实施句**

> Across sequence lengths K = 3, 5, 7, and 10, effective component number increased while the contribution of the latest item declined (mean effective component number at K = 3, 5, 7 and 10, respectively: 2.994 [95% CI, 2.993–2.995], 4.957 [4.952–4.962], 6.743 [6.723–6.763] and 8.048 [7.913–8.182]; mean latest-item weight at the same loads: 0.331 [0.330–0.332], 0.195 [0.194–0.197], 0.133 [0.132–0.134] and 0.0726 [0.0714–0.0738]; Fig. 6c,d), indicating increasingly distributed component expression rather than latest-item domination.

**新增数据**：`K = 3, 5, 7, 10` 的全部 `N_eff` 与 latest-item weight 均值和 95% CI。这里条件只有四组，完整列出比只取端点更能支撑“distributed rather than latest-item dominated”。

**证据类型**：两个 endpoint 均为 `descriptive_only`；不新增 `P` 值，也不把 `N_eff` 改写为 capacity。

**来源**：

- `results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig5/metrics/panel_c_statistics.csv`
- `results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig5/metrics/panel_d_statistics.csv`

---

### R8 — Fig. 6e,f：load–delay 网格的代表性边界

**位置**：同一段紧随 R7 的句子。

**旧句**

> Across the same load–delay conditions, history-matched STSP morphology remained more similar to the terminal state than sequence-deranged composites (Fig. 6e,f).

**实施句**

> Across the same load–delay conditions, history-matched STSP morphology remained more similar to the terminal state than sequence-deranged composites (across-condition mean effective STSP-support area, 0.444 [K = 3, 800 ms; 95% CI, 0.437–0.451] to 0.566 [K = 10, 100 ms; 0.560–0.571]; mean matched-minus-deranged centered-cosine difference, 0.134 [K = 10, 100 ms; 0.130–0.138] to 0.387 [K = 3, 800 ms; 0.375–0.399]; Fig. 6e,f).

**新增数据**：两个 4 × 4 网格 endpoint 的跨条件均值最小/最大边界，并明确对应的 `K`、delay 与 CI。避免把全部 32 个均值逐项塞入正文。

**证据类型**：两个 endpoint 均为 `descriptive_only`；不新增 `P` 值。

**来源**：

- `results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig5/metrics/panel_e_statistics.csv`
- `results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig5/metrics/panel_f_statistics.csv`

## 5. 明确不做的修改

1. 不给 Fig. 1a–d、Fig. 3a、Fig. 4g 添加 `n`、CI 或 `P`；这些面板没有独立网络级推断单位。
2. 不对当前 Fig. 2a 再加数据；其 90.95% 与 95% CI 已完整。
3. 不把 Fig. 4b–c 的所有 support/advance/recruit/loss 组均值写成清单；R5 的预声明干预对比已经直接支撑该句的因果谓词。
4. 不重复已经完整的 Fig. 3d、Fig. 4a/e/f、Fig. 5a–d、Fig. 6b 与 Fig. 7a–f 数据。
5. 不引入旧编号、旧 package 结论、`support_gain_corr`、已退出正文的 history-rewrite bridge 或任何新计算。
6. 不调整标题、过渡句、结论句、段落边界、图注、Methods 或 References。

## 6. DOCX 写入采用的字符格式规则

- 统计符号 `P`、样本量符号 `n`、变量 `K` 使用斜体；数值、`CI`、`BH-adjusted`、`Holm-adjusted` 保持正体。
- 科学计数法使用现有 Word 格式：乘号 `×`，指数为真正上标，如 `10−32` 中的 `−32`。
- `Fig.`、panel 字母、`Layer`、`ms`、`percentage points` 保持正体。
- 小数精度按 endpoint 分辨率确定：百分比 1–2 位；一般相似度 3–4 位；窄 CI 的 constituent similarity 保留 5 位。
- 方括号仅包围估计值对应的 95% CI；第一个同类值写完整 `95% CI`，后续同组值可省略重复标签。
- 新增统计量继承当前 Results 字体、字号、行距与段落样式；OMML 公式和已有变量格式未改变。

## 7. 已批准并完成的单位修正

`panel_b_statistics.csv` 将 aligned-minus-mismatched rescue/loss 差值定义为 `percentage_points`。实施文档已将 R2 原句中的 “increased rescue by 14.1%” 和 “reduced loss by 5.1%” 分别修正为 “increased rescue by 14.1 percentage points” 和 “reduced loss by 5.1 percentage points”。新增的 aligned/mismatched 绝对 rescue/loss 水平仍按数据表正确使用百分比。

## 8. 执行与验收结果

1. 以源 DOCX 新建实施副本，没有覆盖源文件。
2. R1–R8 已全部写入；除作者批准的 R2 两处单位替换外，其余旧句均仅插入数据。
3. 每个新增均值、CI、统计状态、校正方法与当前权威 CSV 完全一致。
4. DOCX ZIP 共 27 个部件；除 `word/document.xml` 外全部逐字节相同。正文仅 6 个目标 Results 段落发生变化，OMML 数量保持 39。
5. LibreOffice 渲染仍为 30 页；已检查新增数据所在的第 4–8、12–13 页，无截断、重叠、坏字形、公式占位符或异常分页，Fig. 2–7 引用未改变。
6. 实施文档 SHA-256：`40cd729b493422844b5264e124382cc4a1b243d0a806b629acf40e68c35f7ae1`。
7. 独立只读复核结论为 PASS：数值、单位、图号映射、统计状态、字符格式、包级变更边界和既有 PDF 渲染均无可执行缺陷。
