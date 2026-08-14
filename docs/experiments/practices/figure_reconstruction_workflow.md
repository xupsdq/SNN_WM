---
type: workflow
subtype: figure-design
domain: paper_writing
project: paper-draft
created: 2026-05-12
tags:
  - workflow
  - figure-design
  - paper-draft
  - codex-prompt
---

# 三阶段论文图重构流程

> 本文件固定科研论文 Figure 重构的完整工作流。  
> 适用范围：任何需要从正文出发重新设计、重构或新建论文图的场景。  
> 核心原则：**先确定图要证明什么，再确定怎么画，再让代码实现。**

---

## 总体原则

```text
阶段 1：内容与图型对应 → 科学逻辑
阶段 2：版面与空间权重设计 → 视觉论证结构
阶段 3：Codex 实现提示词 → 工程实现约束
```

这三个阶段的顺序不能颠倒。

不能一开始就问：

```text
这张图怎么排？
这张图用什么颜色？
代码怎么写？
```

而应该先问：

```text
正文需要这张图证明什么？
每个 panel 的结论是什么？
什么图型最直接表达这个结论？
```

---

# 阶段 1：内容与图型对应

## 阶段目标

> **把正文结论转化为 figure-level claim、panel-level claim 和 plot type。**

这一阶段只处理科学内容和图型选择，不处理：具体 mm 尺寸、颜色、字体、线宽、最终图例样式、代码实现。

---

## 1.1 从正文中抽取图级结论

第一步不是看旧图，而是看正文。

要问：

```text
这个 Results 小节最终想让读者相信什么？
这张 figure 在全文论证链中承担哪一步？
它是 substrate validation、state outcome、mechanism entry、mechanism conversion，还是 causal closure？
```

然后写出一句：

```text
Fig.X should convince the reader that ...
```

这句话就是图级结论。如果这句话写不清楚，说明 figure 的任务没有收束。

---

## 1.2 把图级结论拆成 panel 级结论

每个 panel 都必须对应一个明确 claim。不要写「Panel A 画某个指标」，要写「Panel A 证明某个结论」。

推荐表格：

| 字段 | 内容 |
|------|------|
| Panel ID | A/B/C... |
| Panel claim | 这个 panel 要证明什么 |
| Manuscript claim | 对应正文哪一句/哪一段 |
| Metric | 用什么指标证明 |
| Conditions | 比较哪些条件 |
| Required contrast | 关键比较是什么 |
| Plot type | 最适合的图型 |
| Why this plot | 为什么这个图型最直接 |

阶段 1 最重要的标准是：**没有明确 claim 的 panel 不应该进入主文图。**

---

## 1.3 先看结论关系，再看旧图能不能继承

旧图不能直接决定新图结构。旧图只能作为：

```text
数据线索
可复用图型线索
需要修正的问题线索
```

但不能作为主图重构的起点。

核心经验：有些旧 panel 本身数据没错，但已经不符合正文重构后的 panel mapping。此时应跟随正文改图，而不是为了保留旧图去改正文。

---

## 1.4 根据 claim 类型选择图型

图型应该由结论类型决定，而不是由旧脚本决定。

| 结论类型 | 推荐图型 |
|----------|---------|
| 某指标大于 0 / 大于 chance | one-sample dot summary + reference line |
| 两个 paired condition 比较 | paired dot / slope plot |
| 多组 categorical 比较 | grouped dot/bar summary |
| 时间过程 | timecourse line plot |
| 连续变量关系 | scatter + regression |
| model comparison | paired R² / paired score plot |
| spatial structure | heatmap / support map |
| sequence / protocol | horizontal schematic |
| composition | stacked bar；donut 慎用 |
| causal perturbation | ordered condition dot/line plot |
| interaction | interaction plot / grouped line plot |

核心原则：**图型必须直接表达正文 claim。**如果读者需要从图里绕一圈才能推导出正文结论，图型就不够好。

---

## 1.5 判断哪些旧图型要降级

有些图型在实验探索阶段有用，但在论文主图中不一定最优。

通用经验：

```text
donut 适合 composition，但不适合表达两组差异。
histogram 适合大量 trial distribution，但不适合 n=20 network summary。
stacked bar 适合全体构成，但不适合突出单一比较。
heatmap 很直观，但如果正文 claim 是 COM shift，则 COM trajectory 应该是主图。
```

阶段 1 要判断：旧图型是否真正服务正文 claim？如果不服务，应该改成什么？

---

## 1.6 阶段 1 产物

```text
Figure Claim Map
Panel Claim Map
Metric/Condition Map
Plot Type Map
旧图继承与修改建议
```

---

# 阶段 2：版面结构与空间权重设计

## 阶段目标

> **把阶段 1 的结论链转化为可阅读的论文图结构。**

这一阶段仍然不处理最终颜色和字体，只处理：阅读顺序、论证层级、layout 类型、panel 空间权重、mm 尺寸、panel 位置。

---

## 2.1 先确定阅读路径

不要先问「几行几列」。先问：

```text
读者应该按什么顺序读这张图？
```

阅读路径必须等于正文论证路径。

---

## 2.2 划分论证层级

每张图通常不是 panel 的平铺，而是几个论证层级。

常见层级：

```text
model / protocol definition
state-structure evidence
functional-access evidence
mechanism-entry evidence
effect localization
functional restoration
mechanism conversion
causal closure
```

先划分层级，再决定 layout。

---

## 2.3 选择 layout 类型

不要默认 2×3 或 2×2。应根据结论链选择 layout 类型：

| Layout 类型 | 适用情况 |
|-------------|---------|
| top schematic strip | 有横向 model / protocol / timeline |
| three-band argument layout | 有三层清楚证据链 |
| two-band mechanism layout | 上层输入/支持，下层机制/闭合 |
| full-width anchor layout | 某个 panel 是关键桥接或总入口 |
| asymmetric evidence layout | 某个 panel 信息量明显更大 |
| balanced grid | panel 权重接近时才使用 |

核心经验：**好的顶刊图通常不是均分网格，而是证据权重网格。**

---

## 2.3.1 找到 anchor row / anchor column

不要平均直觉分配 panel。确定 layout 时，先找最能决定网格结构的 anchor row 或 anchor column。

常见 anchor 来源：

### 1. panel 数量最多的一行

如果某一行 panel 最多，例如 B/C/D 三个 panel，那么这一行通常应该先定义横向网格：

```text
available_width = canvas_width - left_margin - right_margin
unit_width = (available_width - total_column_gaps) / n_columns
```

然后其他行根据这个 unit width 进行对齐。

### 2. 信息最复杂的 panel

如果某个 panel 是 timecourse、scatter、trajectory、event-aligned traces 或 large heatmap，它可能需要跨列。跨列必须显式写出：

```text
E spans columns 1–2
F equals one standard column
```

不要只写：

```text
E is larger
F is smaller
```

### 3. 上下对应的 panel

如果上下两个 panel 在论证上对应，应显式写：

```text
B.same_size_as = E
B.axes.left = E.axes.left
B.axes.right = E.axes.right
```

核心经验：**先找 anchor row，再决定其他 row 如何继承这个网格。**

---

## 2.4 按 panel 功能分配空间

panel 尺寸应该由它的功能决定。

| Panel 类型 | 空间策略 |
|-----------|---------|
| 横向 timeline / architecture | 需要宽，不一定高 |
| 单指标 summary | 可以较小 |
| paired comparison | 中等 |
| scatter/regression | 需要较宽 |
| timecourse | 需要较宽 |
| heatmap / support map | 需要接近方形 |
| event-aligned traces | 需要较大空间 |
| causal closure panel | 不应过小 |
| final mechanism closure | 应给较高视觉权重 |

不要让所有 panel 一样大。

---

## 2.5 确定 A4 页面下的 mm 尺寸

因为目标是论文 A4 full-width 图，阶段 2 必须明确：

```text
canvas width / height
panel x/y/w/h
horizontal gutter / vertical gutter
```

原则：

```text
主图宽度直接按 165 mm 左右设计；
不要先按单栏小图生成，再放大；
不要在 Word/PDF 中拉伸图片。
```

每个 panel 的位置应明确如：

```text
A: x=0, y=0, w=165, h=28
B: x=0, y=33, w=80, h=42
C: x=84, y=33, w=81, h=42
```

这样后续代码就可以直接实现，而不是靠 `subplots` 自动均分。

---

## 2.5.1 使用 axes-box 对齐，而不是 outer panel box 对齐

在论文主图中，判断 panel 是否对齐时，优先使用实际 plotting axes region，也就是坐标轴框，而不是外部 panel bounding box。

不要把以下元素纳入 panel size matching：

```text
panel letter
legend
colorbar
长 y-axis label
tick label 外伸区域
directional arrow annotation
```

原因是这些元素会改变视觉外边界。如果按 outer box 对齐，坐标轴本身反而可能不对齐。

推荐在 layout specification 中写成显式等式：

```text
B.axes.left = D.axes.left
B.axes.right = D.axes.right
C.axes.left = E.axes.left
C.axes.bottom = E.axes.bottom
```

而不是只写：

```text
B and D are aligned
C and E have similar size
```

验收标准：

```text
同行 panel 的 plotting area top/bottom 对齐；
同列 panel 的 plotting area left/right 对齐；
panel letter 不参与尺寸计算；
legend / colorbar / annotation 不破坏 axes-box 对齐。
```

核心经验：**论文图的视觉对齐应以坐标轴框为准，而不是以所有文字和图例形成的外框为准。**

---

## 2.5.2 用 span 关系描述不等宽 panel

当某个 panel 需要更大空间时，不要直接给一个主观宽度，而应描述它跨越哪些 column。

推荐写法：

```text
Row 2 anchor:
B = C = D

Row 3:
E spans B + C
F.same_size_as = D
E.left = B.left
E.right = C.right
F.left = D.left
F.right = D.right
```

这种写法优于：

```text
E large, F small
```

因为 Codex 可以直接根据 span relation 更新 `position_mm`，而不会自由发挥。

验收标准：

```text
跨列 panel 的 left/right 边界必须与被跨越 column 的边界一致；
单列 panel 必须与 anchor row 的对应 column 同宽；
跨列 panel 不应破坏 row gap / column gap 的一致性。
```

---

## 2.5.3 把 legend、colorbar、annotation 作为 layout 对象

legend、colorbar、directional axis arrow、peak label、百分比标注都不是事后装饰。它们会占据空间，因此必须在 layout 阶段规划。

推荐优先级：

```text
1. 能不用 legend 时，优先直接标注。
2. 必须用 legend 时，优先放在 panel 上方一行或 panel 内部空白区。
3. 如果 legend 放 panel 外，必须把它纳入 layout 预留空间。
4. colorbar 只在它确实帮助理解数值时保留；否则不要为了完整性强行加。
5. directional arrow annotation 不能放进 data region 里省空间。
```

禁止做法：

```text
把 y-axis tick label 放进绘图区；
把 y-axis label 放进绘图区；
legend 压住 data；
colorbar 挤压主 heatmap；
directional annotation 与邻近 panel 重叠；
通过极端缩小字体来解决结构性拥挤。
```

正确修复顺序：

```text
先移动 legend / annotation；
再 wrap label；
再增加 gutter；
再调整 panel width / height；
最后才考虑小幅缩小字体。
```

核心经验：**如果 legend 或 annotation 改变了坐标轴对齐，它就不是 style 问题，而是 layout 问题。**

---

## 2.5.4 缺失 panel 的主图处理规则

当 manual asset 或 direct data source 缺失时：

```text
允许保留 panel slot；
允许保留 panel letter；
必须在 QC 中记录 warning；
不要在最终主图中显示大段 Missing source / placeholder text。
```

推荐处理：

| 缺失类型 | 主图处理 | QC 处理 |
|----------|---------|---------|
| manual schematic 缺失 | 留空 | warning |
| direct metric source 缺失 | 留空或极简 schematic | warning |
| adapter 找到 fallback summary | 可画，但必须标记 summary-only fallback | warning |
| single-seed fallback | 可画，但必须警告 n 不足 | warning |

核心经验：**debug 信息应该进入 QC report，不应该进入最终主图。**

---

## 2.6 判断主视觉锚点

每张图应该有一个或几个视觉锚点。要问：

```text
这张图的入口 panel 是谁？
主证据 panel 是谁？
最终 closure panel 是谁？
哪个 panel 不能被压小？
```

视觉权重应服务论证权重。

---

## 2.7 阶段 2 产物

阶段 2 不只输出 x/y/w/h，还应输出 layout reasoning 和 alignment constraints。

```text
Reading Order Map
Argument Band Map
Layout Type
Canvas mm size
Global Margin/Gutter Table
Anchor Row / Anchor Column
Column Unit Definition
Panel Size Table
Panel Position Table (panel_id → x/y/w/h)
Axes Alignment Map
Span Relation Map
Legend/Colorbar/Annotation Placement Map
Spatial Priority Map
Visual QC Checklist
```

推荐增加一个显式 Alignment Map：

```text
A.left = B.left = E.left
A.right = D.right = F.right
B = C = D
E spans B + C
F.same_size_as = D
D.axes.bottom = E.axes.bottom
```

这能减少 Codex 对 layout 的自由解释空间。

---

# 阶段 3：Codex 实现提示词

## 阶段目标

> **把阶段 1 和阶段 2 的决定转化为 Codex 可执行的实现提示词。**

这一阶段不是重新设计图，而是要求 Codex 在已有系统中实现。

---

## 3.1 先限定 Codex 任务边界

必须明确告诉 Codex：

```text
不要重新设计图
不要改变 panel 顺序
不要改变 layout
不要重跑实验
不要直接复用旧实验图作为最终 panel
不要处理最终颜色、字体、线宽
数据缺失时用 placeholder + QC warning
```

否则 Codex 可能会自作主张，重新组织 figure。

---

## 3.2 写入 figure scientific specification

来自阶段 1。应包含：

```text
figure-level claim
evidence chain
reading order
panel claims / types / metrics / conditions
x-axis / y-axis / reference lines / notes
```

每个 panel 都要写清楚：claim、panel_type、renderer、data_adapter、metric、conditions、axis labels、notes。

---

## 3.3 写入 layout specification

来自阶段 2。应包含：

```text
canvas width / height
每个 panel 的 x/y/w/h
gutters
layout rationale
```

明确要求：

```text
Do not use plt.subplots for final layout.
Use mm-based add_axes_mm.
```

---

## 3.4 定义 adapters

告诉 Codex 每个 panel 需要什么 adapter，并要求每个 adapter 输出：

```text
panel_data.csv
stats.json
source_manifest.json
```

规定 canonical fields：

```text
figure_id, panel_id, metric, condition, layer, network_id, seed_id, value, unit, source_file
```

以及 panel-specific fields。

---

## 3.5 定义 source discovery

Codex 需要根据已有 mapping / fig 文件夹 / results 文件夹寻找数据源。

要求：

```text
不要 hard-code absolute paths
使用 first_existing_path-style lookup
找不到数据时返回 missing_adapter_result
```

---

## 3.6 定义 renderers

每个 renderer 要写清楚：

```text
preferred rendering
fallback rendering
axis semantics
reference lines
what not to do
```

同一个数据可以画成很多图，但只有一种图型最符合正文 claim。

---

## 3.7 定义 QC requirements

通用 QC：

```text
spec parseable
each panel has claim
position_mm exists
canvas size correct
outputs exist
source manifest exists
no old source image used as final panel
```

术语 QC：

```text
no old stem labels
no internal panel labels
no raw code condition names
no undefined abbreviations
```

fallback QC：

```text
warn if single-seed data
warn if trial-level fallback
warn if summary-only fallback
warn if unit ambiguous
```

### 数据粒度 QC

仅报告 `n_networks = 20` 不够。必须区分：

```text
找到了 20 个 networks
vs.
保留了 20 个 networks 的 row-level data
```

对于每个 data-driven panel，QC 应报告：

```text
source files used
raw rows read
rows after filtering
rows written to panel_data
whether adapter performed network-level averaging
whether source appeared pre-aggregated
whether renderer performed additional aggregation
```

如果 adapter 做了 groupby mean，应在 stats/QC 中显式记录：

```text
aggregation = mean_by_network_condition
```

如果源文件本身已经是 `__by_network` summary，也要记录：

```text
source appears pre-aggregated; row-level source unavailable
```

核心经验：**n=20 PASS 只能证明网络数量正确，不能证明数据粒度正确。**

### 视觉 layout QC

```text
panel axes 是否对齐
legend 是否挤到相邻 panel
axis label 是否与邻近 panel 冲突
panel letter 是否漂移
总图是否出现异常留白
总图导出的 pdf/svg/png 是否完整
是否有 clipping 或 overlap
```

---

## 3.8 定义 acceptance criteria

```text
python -m src.plotting.paper_fig.final_six.figX_plot \
  --input-dir results/paper_figure_multi_seed/final_six_figures_v5_c5_revised_20260804_r2/figX \
  --check-only

python -m src.plotting.paper_fig.final_six.figX_plot \
  --input-dir results/paper_figure_multi_seed/final_six_figures_v5_c5_revised_20260804_r2/figX
```

`--check-only` 必须返回 `status=check_passed`。正式 plot-only render 的当前核心输出为：

```text
figures/figX.pdf / figures/figX.svg / figures/figX.png
figures/panels/ / figures/qa/
meta/final_plot_spec.json
meta/main_figure_panel_index.csv
meta/layout_measurements.csv
meta/visual_qa.json
artifact_manifest.json
```

缺少 required source、统计表、seed 或布局条件时必须失败；不得用 placeholder 掩盖缺失证据。

---

## 3.8.1 必须检查 integrated full figure

所有 panel 修改都必须反映到最终 full composite figure。不要只检查 standalone panel。

Codex prompt 中应明确写：

```text
Apply this change to the integrated full Fig.X, not only the standalone Panel Y.
Regenerate the final composite figure after the panel update.
```

视觉 QC 必须检查 full figure：

```text
panel axes 是否对齐；
legend 是否挤到相邻 panel；
axis label 是否与邻近 panel 冲突；
panel letter 是否漂移；
总图是否出现异常留白；
总图导出的 pdf/svg/png 是否完整。
```

核心经验：**最终判断对象永远是 full composite figure，不是单独 panel。**

---

## 3.9 阶段 3 产物

```text
Codex implementation prompt（完整可执行）
```

---

## 3.10 Layout-specific Codex prompt 模板

```text
Update the full composite Fig.X.

Do not:
- Do not modify manuscript text or caption.
- Do not rerun experiments.
- Do not redesign panel scientific meanings.
- Do not use plt.subplots for final layout.
- Do not update only standalone panels.

Layout principles:
- Use mm-based manual layout.
- Use axes-region alignment, not outer panel box alignment.
- Panel letters do not participate in size matching.
- Legends/colorbars/annotations must be treated as layout objects.

Canvas:
- width = ...
- height = ...

Global:
- left margin = ...
- right margin = ...
- top margin = ...
- bottom margin = ...
- row gap = ...
- column gap = ...

Anchor row:
- ...

Alignment map:
- ...
- ...

Span map:
- ...

Panel-specific:
- ...

QC:
- full composite figure regenerated
- axes boxes aligned
- no clipping
- no legend/data overlap
- no tick/axis label inside plotting region
- missing sources shown in QC, not as large figure text
```

---

# Figure layout 微经验库

下面是 figure 重构中容易反复出现的小问题，建议作为 Codex prompt 和人工 QC 的固定检查项。

## 1. tick label

```text
x tick label 默认正体，不要斜体；
rotation 只在文字确实放不下时使用；
如果是 ordered bin，不一定显示 bin_1/bin_2，可用方向箭头表达连续变化；
tick label 不得进入绘图区。
```

## 2. y-axis label

```text
每个 panel 的 y-axis 必须明确指标；
不能为了省空间把 y tick 或 y label 放进 axes 内部；
同列 panel 的 y-axis 位置应尽量对齐。
```

## 3. 数值标注

```text
bar summary 可标数值；
小 segment 不强制标，避免拥挤；
数值标注不能压 error bar；
数值标注不能超出 panel top。
```

## 4. legend

```text
legend 尽量单行；
composition panel 的 legend 优先放在上方；
timecourse panel 的 legend 不得压线；
如果 legend 与主数据抢空间，应移动而不是缩小到不可读。
```

## 5. axis-direction arrow

```text
方向箭头适合表达 ordered progression；
箭头应放在 axis 外部或预留区域；
不能与相邻 panel overlap；
箭头文字应简短。
```

## 6. colorbar

```text
colorbar 只在连续 heatmap 数值对 claim 有帮助时保留；
colorbar 必须预留空间；
colorbar 不应挤压主图，也不能破坏 axes-box 对齐。
```

## 7. panel letter

```text
panel letter 使用统一 offset；
不参与 panel size matching；
不要离 panel 太远；
不要与 tick label / legend / value label overlap。
```

## 8. blank panel

```text
缺失 schematic 或 source 时，主图留空；
debug 文本进入 QC，不进入最终 figure；
blank panel 仍保留 layout slot，避免破坏阅读路径。
```

## 9. full figure update

```text
任何 panel 修改都必须更新 full composite figure；
不要只生成 standalone panel；
最终验收看 full figure，不看单图。
```

## 10. check-only PASS

```text
check-only PASS 不等于视觉 PASS；
必须查看导出的 png/svg/pdf；
尤其检查 clipping、overlap、axis alignment、legend placement。
```

---

# 三阶段流程模板（快速参考）

## 阶段 1 模板

```text
1. 读取对应正文段落。
2. 提取 figure-level claim。
3. 拆解 panel-level claims。
4. 为每个 panel 确定 metric、condition、contrast。
5. 根据 claim 类型选择 plot type。
6. 判断旧图哪些保留、哪些替换、哪些移到 supplement。
7. 输出 Panel Claim Table 和 Plot Type Map。
```

## 阶段 2 模板

```text
1. 确定阅读路径。
2. 划分论证层级。
3. 选择 layout 类型。
4. 确定 canvas mm size。
5. 根据 panel 功能分配空间权重。
6. 写出每个 panel 的 x/y/w/h。
7. 输出 Layout Map 和 Panel Size Table。
```

## 阶段 3 模板

```text
1. 明确 Codex 不重新设计。
2. 写入 figure scientific specification。
3. 写入 panel specification。
4. 写入 layout specification。
5. 定义 adapters。
6. 定义 renderers。
7. 定义 source discovery keywords。
8. 定义 QC requirements。
9. 定义 acceptance criteria。
10. 输出完整 Codex prompt。
```

---

# 核心经验总结

| # | 经验 | 说明 |
|---|------|------|
| 1 | **先 claim，后 metric，后图型** | 正文 claim → metric → condition contrast → plot type。不从「数据能画什么」开始 |
| 2 | **旧图只能参考，不能主导** | 旧图价值：数据线索、分析线索、问题线索。不决定最终 panel 结构 |
| 3 | **layout 是论证结构，不是排版装饰** | 好的 layout 让读者按正文逻辑读图 |
| 4 | **mm 尺寸必须在阶段 2 确定** | 不提前确定 → 图太小/panel 压扁/比例拉伸/坐标轴不可读 |
| 5 | **先找 anchor row / anchor column** | panel 最多或信息最重的一行/列先定义网格，其他 panel 继承 |
| 6 | **用 axes-box 对齐，不用 outer box 对齐** | panel letter、legend、colorbar 不参与 panel size matching |
| 7 | **跨列 panel 必须显式写 span** | 写 `E spans B+C`，不要写「E 更大」 |
| 8 | **legend / colorbar / annotation 是 layout 对象** | 这些元素需要预留空间，不是最后随便塞进去 |
| 9 | **禁止把 tick / axis label 放进绘图区省空间** | 空间不足应调整 gutter、label wrap、panel size，而不是压入图内 |
| 10 | **placeholder 不污染主图** | missing source / missing asset 在 QC warning，主图留空或极简 |
| 11 | **n=20 不等于数据粒度正确** | QC 必须报告 raw rows、filter rows、panel_data rows、是否 network-level averaging |
| 12 | **Codex 提示词必须约束「不要做什么」** | 禁止重设计、改顺序、复用旧图、硬编码风格、静默用 single-seed、保留旧 label |
| 13 | **所有 panel 修改必须回到 full composite figure 验收** | standalone panel 好看不代表总图完成 |
| 14 | **QC 在提示词中提前定义** | data QC / label QC / unit QC / layout QC / fallback QC / granularity QC / visual QC |
| 15 | **颜色和字体最后统一做** | 先完成所有图的科学结构、数据、layout、QC，再统一做全篇 style system |

---

# 最终总结

> **阶段 1 决定图为什么存在；阶段 2 决定图如何被阅读；阶段 3 决定代码如何不偏离前两阶段的设计。**

```text
阶段 1：科学逻辑
阶段 2：视觉论证结构
阶段 3：工程实现约束
```

这套流程的价值在于，它能防止论文图从「实验结果截图」变成「堆 panel」，而是让每张图都成为正文论证链的可视化版本。

---

*Figure Reconstruction Workflow — 提炼自 paper-draft P1-11 交付过程，2026-05-12。Fig.1–6 轮重构 layout 经验追加 2026-05-14。*
