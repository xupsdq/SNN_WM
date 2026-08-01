# Paper Figure Practical Layout Contract

本契约规定 Fig.1–Fig.6 及补充图的多面板布局决策顺序。它建立在科研绘图工具作者的失败案例、研究者组图流程以及本项目 Fig.6 反例测量之上。期刊尺寸只定义最终画布边界，不替代布局设计。

## 1. 权威层级

布局必须按以下顺序决定，后一级不能推翻前一级：

1. 科学阅读关系与直接比较关系；
2. 语义单元和离散拓扑；
3. 图形类型的自然纵横比与装饰占用；
4. 面板槽位、数据绘图区和间距的毫米尺寸；
5. 柱宽、标记、图例和局部光学修正；
6. 最终出版尺寸下的渲染质检。

禁止从柱数、等宽网格或全局坐标轴对齐直接推导整图拓扑。

### 1.1 Reader contract 与任务图

在填写 `layout_contract` 之前，先为整张图冻结 `reader_contract`。它回答的不是“面板放在哪里”，而是目标读者需要完成哪些判断、这些判断存在什么依赖，以及整张图最终允许支持到什么程度。

`reader_contract` 至少声明：

- `figure_question`：读者打开图时要回答的唯一主问题；
- `terminal_inference`：读完全部证据后允许形成的限定性结论；
- `forbidden_inferences`：图中证据不能支持的越界结论；
- `semantic_units`：各面板承担的前提、核心证据、扰动验证或边界条件角色；
- `task_graph`：读者动作节点及其有向依赖边；
- `comparison_obligations`：必须直接比较的面板、比较动作和不得暗示的比较；
- `topology_invariants` 与 `topology_freedoms`：所有布局候选共同遵守的关系和可以探索的自由度。

任务图是读者推理的 DAG，不等同于面板字母顺序。一个布局候选只有在完整保留任务节点、依赖边和直接比较义务时，才算同一论证的不同拓扑；仅改变面板毫米尺寸、柱宽或留白不算拓扑变化。

## 2. 三种几何边界

每个数据面板必须分别记录：

- `slot_bbox`：布局分配给面板的完整槽位；
- `plot_bbox`：实际数据绘图区；
- `decorations_mm`：左、右、上、下的坐标标题、刻度文字、图例、色条和注释占用。

尺寸分配以最终画布毫米为单位。图例、色条和长标签是布局成员，不能在绘图完成后塞入剩余空间。

## 3. 语义单元和比较组

每个面板必须且只能属于一个 `semantic_unit`。一个语义单元可以包含一个面板，也可以包含必须共同阅读的多个面板。

直接比较组必须声明：

- 比较的科学量；
- 共享或对应的坐标方向；
- 是否要求相同尺度；
- 是否要求相同数据区纵横比；
- 读者需要执行的比较动作。

相邻不等于可比。同处一行不能自动产生坐标轴对齐约束。

## 4. 对齐契约

每条对齐声明必须包含：

- `panels`：参与面板；
- `target`：`slot`、`plot_area`、`legend` 或 `panel_label`；
- `edges`：`left`、`right`、`top`、`bottom` 中的明确子集；
- `rationale`：为什么这种对齐帮助科学阅读；
- `comparison_basis`：当 `target: plot_area` 且包含多个面板时必须提供。

对齐分为：

1. **比较对齐**：直接比较组的数据区、尺度或刻度对应；
2. **结构对齐**：面板槽位、行边界和面板标签形成稳定阅读秩序；
3. **光学对齐**：仅在不制造压缩、空洞或碰撞时使用。

允许按单侧释放对齐。禁止把装饰结构明显不同的面板强制成四边一致的数据区。

## 5. 拓扑规则

拓扑先于毫米尺寸。候选拓扑必须：

- 保持既定科学阅读顺序；
- 将直接比较面板放在同一语义单元内并保持邻接；
- 避免没有语义含义的孤立面板和空洞；
- 使用组内小间距、组间大间距表达关系；
- 允许嵌套网格和不等宽面板；
- 在带入数据前先通过空白线框检查。

默认的方形网格、等宽列或行优先填充只是软件启发式，不是设计规则。

## 6. 自然几何

每个面板必须声明：

- `chart_family`；
- `category_slots` 或连续位置数量；
- `natural_aspect` 范围；
- `decoration_sides`；
- `visual_weight`：`low`、`medium` 或 `high`。

自然纵横比是软目标，但直接比较面板的相同纵横比可以升级为硬约束。固定纵横比、等宽、等高和双向全局对齐发生冲突时，以科学比较和可读性优先。

## 7. 柱宽规则

柱宽优先级为：

1. 同一面板内柱宽和组距一致；
2. 直接比较的同构柱图使用一致类别带宽；
3. 无直接比较关系的面板不要求物理毫米柱宽相同。

不同柱数时只能选择以下策略之一：

- `preserve_slots`：保留统一类别槽位，缺失类别留空；
- `proportional_panel_width`：面板宽度随类别槽位数变化；
- `within_panel_only`：仅统一面板内部柱宽比例。

禁止通过缩小一个过大槽位中的数据区来追求跨面板固定毫米柱宽。

## 8. 图例、底纹和参照线

- 图例必须声明属于单面板、比较组或整图；
- 同一语义映射不得重复创建互相冲突的图例；
- 删除无信息价值的灰色底纹、次网格和装饰性框线；
- 可以保留零线、机会水平线、同一比较组必要的主刻度线等科学参照；
- 删除网格后，坐标轴或基准线必须仍能锚定数据。

## 9. 硬约束与软目标

硬约束包括：

- 阅读顺序和语义邻接；
- 直接比较所需的尺度、纵横比和对齐；
- 最终尺寸下无裁切、碰撞和文字脱离；
- 颜色语义和非颜色冗余保持一致；
- 画布尺寸、最小字体和输出格式；
- 无信息价值的灰底与网格不得出现。

软目标包括：

- 面板等宽；
- 全图坐标轴对齐；
- 跨面板物理柱宽相同；
- 行列对称；
- 留白和视觉重量相近。

软目标不得破坏任何硬约束。

## 10. 执行与验收

本节的 QA 专指视觉布局验收，不继承旧 `qc.py` 中的固定槽位、历史数据内容或旧版面板结构检查。旧 QC 不得作为新布局候选的通过门槛，也不得反向决定拓扑和尺寸。

每张图按以下顺序执行：

1. 填写面板关系和自然几何；
2. 写出 `layout_contract` 并运行校验；
3. 生成不含数据的线框；
4. 选择并冻结拓扑；
5. 在最终画布尺寸中分配毫米尺寸；
6. 带入真实数据；
7. 测量三种边界、装饰占用和声明的对齐边；
8. 在 PNG、PDF 和 SVG 的最终尺寸下检查；
9. 通过颜色视觉、灰度、裁切和碰撞质检后才可替换正式图。

自动布局只负责满足已声明拓扑中的间距和防碰撞，不负责选择拓扑。

## 11. 机器可校验字段

候选规格中的 `layout_contract` 至少包含：

```yaml
layout_contract:
  version: practical_layout_v1
  status: candidate
  semantic_units: []
  comparison_groups: []
  alignment_groups: []
  panel_geometry: {}
  bar_width_policy: {}
  topology: {}
  hard_constraints: []
  soft_targets: []
  qa: {}
```

`src.plotting.paper_fig.layout_contract.validate_layout_contract()` 对这些字段进行构图前校验。

## 12. 实操依据

- [Patchwork complex layout cases](https://patchwork.data-imaginist.com/articles/guides/layout.html)
- [Cowplot axis-specific alignment](https://wilkelab.org/cowplot/articles/aligning_plots.html)
- [Matplotlib constrained layout](https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html)
- [Caltech Designing Effective Figures](https://writing.caltech.edu/documents/32547/Designing_effective_figures.pdf)
- [PLOS multipanel planning workflow](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001161)
