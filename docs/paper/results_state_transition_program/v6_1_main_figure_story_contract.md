# V6.1 主图故事与跨图语义合同

## 状态

- 适用对象：`docs/paper/v6_1.docx` 中的稿件 Fig.1–Fig.7。
- 当前状态：工作合同；待作者逐项确认后冻结。
- 建立日期：2026-08-14。
- 目的：统一七张主图中的状态对象、过程术语、干预角色、视觉语法和跨图交接，使读者不依赖正文也能恢复主论证方向。
- 本合同不改变已冻结的实验、端点、统计、独立重复单位或证据边界，不授权新增训练、模拟、forward replay 或派生结论。

## 1. 权威边界与编号

适用权威顺序保持为：

1. 作者最新明确决定；
2. `docs/paper/PAPER_AUTHORITY.json`；
3. `docs/paper/CORE_SCIENTIFIC_LOGIC_CONTRACT.md`；
4. `docs/paper/RESULTS_EVIDENCE_BOUNDARIES.md`；
5. V6.1 当前内嵌 artwork 与其登记 bundle；
6. 当前 panel contracts、reader contracts 与 source manifests；
7. 历史方案和旧编号。

本合同使用**稿件图号**。内部 runtime/final-six 图号不得直接代入：

| 稿件图号 | 当前 artwork 身份 | 内部身份 |
|---|---|---|
| Fig.1 | redesign mechanism figure | 新增稿件图，无直接 internal figX 对应 |
| Fig.2 | redesign inherited-state figure | internal fig1 的功能证据拆分后重标 |
| Fig.3 | exact-input history conditioning | internal fig2 |
| Fig.4 | local successor formation | internal fig3 |
| Fig.5 | successor reuse and iterative updating | internal fig4 |
| Fig.6 | terminal structural organization | internal fig5 |
| Fig.7 | conditional effect on later processing | internal fig6 |

现有 `main_figure_sequence_contract.md` 和 `fig1_panel_contract.md`–`fig6_panel_contract.md` 继续保留内部编号与数据身份；本文件只新增稿件 Fig.1–Fig.7 的故事、术语和交接权威，不重命名旧合同或 result roots。

本文件 §6 的 panel chain 以当前 promoted artwork、`main_figure_panel_index.csv` 和 V6.1 authority overlay 为准。它不是对旧内部 panel contracts 的静默改写：若旧合同与当前 promoted artwork 冲突，故事导航先按上位 authority 和当前 artwork；任何涉及端点、统计或数据身份的真实冲突必须停止并单独解决，不能由本合同自行选择。本文所有 `Fig.n` 均指稿件图号。

## 2. 全文唯一故事图

七张图的论证结构不是简单的七节点直线，而是“一条核心机制链＋两个并列终点模块”：

```text
Fig.1  brief firing creates post-firing effective STSP support
  ↓
Fig.2  the distributed inherited state persists, carries content, and affects readout
  ↓
Fig.3  an identical current input is processed differently under different inherited histories
  ↓
Fig.4  history-conditioned processing forms a downstream successor state
  ↓
Fig.5  the successor is reused by the next input and the transition motif recurs
  ├──────────────→ Fig.6  terminal structural organization
  └──────────────→ Fig.7  conditional effect on later processing
```

对应的六个论证动作是：

```text
instantiate → inherit → condition → form → reuse/recur → characterize consequences
```

### 2.1 必须保持的核心方向

```text
layer l inherited STSP state
        +
current input
        ↓
history-conditioned processing in layer l
        ↓
downstream successor STSP state in layer l+1
```

不得画成或表述成同层自我写回：

```text
layer l inherited state → layer l processing → rewritten layer l inherited state
```

### 2.2 Fig.6 与 Fig.7 的关系

Fig.6 与 Fig.7 共享 Fig.5 作为父节点，但互相不是因果前后项：

- Fig.6 描述多次转移后终端状态的结构组织；
- Fig.7 独立检验保留状态在特定 cue/content/overlap 条件下对后续处理的影响；
- 不画 Fig.6 → Fig.7 的因果箭头；
- 不声称 Fig.7 “读取了 Fig.6 定义的 morphology”；
- 不把 Fig.6 的结构指标和 Fig.7 的功能指标压缩成共同的 `memory strength`。

稿件中 Fig.6 先于 Fig.7 只是呈现顺序，不是证据依赖顺序。

## 3. 跨图核心状态转移单元

后续所有示意、协议和跨图交接统一使用以下概念单元：

```text
S_k^(l) + I_(k+1)
        ↓
P_(k+1)^(l) | S_k^(l)
        ↓
S_(k+1)^(l+1)
```

其中：

- `S_k^(l)`：第 `k+1` 个输入到达前，Layer `l` 的 inherited STSP state；
- `I_(k+1)`：当前新输入；
- `P_(k+1)^(l) | S_k^(l)`：由当前输入驱动、受继承状态条件化的本层处理；
- `S_(k+1)^(l+1)`：当前处理在下游形成的 successor STSP state；
- 下一输入到达时，`S_(k+1)^(l+1)` 才作为 Layer `l+1` 的 inherited STSP state 进入下一次转移。

符号只作为跨图视觉缩写。凡表示 successor formation 或 successor reuse 的箭头，必须保留 layer superscript；不带 layer 的 `S_k → S_(k+1)` 不能充当跨图因果缩写。图注与正文第一次出现时必须先定义科学对象，不能只给符号。当前 Fig.4g 若继续复用，必须避免把其无 layer 的 `S_k/S_(k+1)` 读成同层自我写回。

## 4. 统一状态词典

### 4.1 状态对象

| 主名称 | 严格定义 | 图中允许短名 | 禁止混用 |
|---|---|---|---|
| **inherited STSP state** | 当前输入到达前、层特异的 joint `u/x` 配置 | inherited state；因果图中只用带 layer 的 `S_k^(l)` | inherited condition 作为独立状态名；support；memory strength |
| **joint `u/x` STSP state** | utilization `u` 与 available-resource `x` 组成的二变量状态 | joint `u/x` state | `ux` tensor；`u×x`；effective support |
| **effective STSP support** | `G_STSP = u ⊙ x` 的逐元素乘积，是 joint state 的派生支持量 | effective support；`G_STSP` | bare `ux`；conductance；joint state |
| **successor STSP state** | 当前输入经本层历史条件化处理后，在下游形成的边界状态 | successor state；因果图中只用带 layer 的 `S_(k+1)^(l+1)` | successor update；旧状态复制；普通前馈输出 |
| **successor update** | 产生 successor state 的输入相关变化 | successor update | successor state 本身 |
| **terminal sequence state** | 多次输入与终端延迟后保留的 joint `u/x` 状态 | terminal state；accumulated state | capacity；accessible items；Fig.7 readout |
| **retained STSP state** | 在已注明 layer、boundary 与 protocol 的 post-input/terminal 时刻仍保留、供 later-input probe 或 intervention 使用的 joint `u/x` 配置 | retained STSP state | bare retained state；retained support；自动等同 inherited/successor/terminal state |
| **retained effective STSP support** | 在指定输入边界仍存在的 `G_STSP` 派生支持量 | retained support（首次定义后） | inherited/terminal joint state |
| **effective-support morphology** | `G_STSP` 在空间上的面积、形状或 matched-versus-deranged specificity | support morphology | joint-state coefficients；functional access |
| **fast network state** | membrane potential、excitatory conductance、refractory state 与 lateral inhibition | fast state | STSP state；non-ux state |

### 4.2 过程对象

| 主名称 | 严格定义 | 禁止升级或替换 |
|---|---|---|
| **history-conditioned processing of the current input** | 当前输入主导、继承 STSP 调制的本层处理 | 历史决定当前答案；history-driven processing |
| **inter-layer state transition** | inherited state 条件化当前处理，并在下一层形成 successor state | 同层自我写回；任意状态变化 |
| **successor-state formation** | downstream successor state 的形成过程 | write-back 作为主名称；普通编码 |
| **successor-state reuse** | 已形成的 successor 在下一输入边界作为下一层的 inherited STSP state 参与处理 | 任意 state persistence；完整 mediation |
| **iterative updating across successive inputs** | 状态转移单元在连续输入中重复应用 | recurrent updating；recurrent network dynamics |
| **equal-time passive evolution** | 同一父状态在无新输入情况下经过相同时间的对照演化 | natural decay；static-frozen control |
| **static-frozen STSP control** | STSP 保持在基线且不发生实际 mutation 的控制条件 | passive evolution；static update |
| **firing-silent delay** | 延迟期实测 firing 已消失或接近零 | activity-silent 作为直接测量词 |
| **activity-silent STSP maintenance** | firing-silent 条件下，信息仍由突触状态维持的机制推断 | Fig.1 单突触 product 本身；持续放电 |

### 4.3 更新分解与空间入口

| 主名称 | 定义 | 与其他概念的边界 |
|---|---|---|
| **common input-driven update** | 在 identical current input 下跨历史共享的主要更新成分 | 不是 history residual 的对立实验条件 |
| **history-conditioned residual** | identical input 下由继承历史差异关联的额外处理成分 | 用 `Γ` 前必须定义；不是 memory amount |
| **overlap-aligned sites/pathways** | retained effective support 与 incoming input pathway 空间重叠的位置 | 不等于 aligned history；不等于类别相似度 |
| **input-engaged non-overlap sites** | 当前输入相关但不与 retained support 重叠的位置 | 不称 probe-only 而不定义 |
| **input-associated state displacement** | observed successor 与其前一边界状态之间的输入相关位移 | 不等于 state identity 或 functional access |
| **observed-minus-passive displacement** | observed next state 相对 equal-time passive counterfactual 的差异 | 不称任意 stage change 为 recurrence |
| **history-differential events** | identical input 下，与 history-conditioned residual 关联的事件位置 | Fig.4 的 advance/recruit/spike-loss taxonomy |
| **spike-transition classes** | dynamic 与 static-frozen response 之间定义的 advancement、recruitment 与 spike loss | Fig.3 的 residual-enriched event 集合 |

### 4.4 干预角色与结果

| 主名称 | 定义 | 禁止混用 |
|---|---|---|
| **donor history/state** | 在受控置换或移植中提供指定 STSP state 的实验角色 | 历史类别本身；current input |
| **receiver history/state** | 接受指定 state，同时保留明确 held-fixed 变量的实验角色 | original 作为未定义同义词 |
| **donor-transfer index** | 结果沿 donor-minus-receiver 方向的标准化投影 | DTI；百分比；必要性；完整 mediation |
| **behavioral rescue** | no-memory/reference 条件错误、历史条件后正确 | event recruitment；accuracy gain |
| **behavioral loss** | no-memory/reference 条件正确、历史条件后错误 | spike loss；rescue 的负值 |
| **opportunity set** | rescue 与 loss 各自独立的合格分母 | 共同分母；rescue-loss 净值 |
| **spike advancement** | 相对 static-frozen response 更早出现的 spike event | behavioral rescue |
| **spike recruitment** | static-frozen 中无事件、dynamic 中出现的 spike event | Functionally rescued fraction |
| **spike loss** | static-frozen 中存在、dynamic 中消失的 spike event | behavioral loss |

图面若同时出现两类 `loss`，必须明确写成 `Behavioral loss` 与 `Spike loss`，不得只靠颜色区分。跨 Fig.3/Fig.5 与 Fig.7 时，也必须分别写 `Behavioral rescue` 与 `Functionally rescued fraction`；后者不是前者的比例化版本。

### 4.5 历史长度、阶段与功能术语

| 主名称 | 定义 | 使用规则 |
|---|---|---|
| **history depth** | exact-input/accumulated-history protocol 中当前输入前的历史深度 | Fig.5 正文写作 `one, five or ten items before B`；`K` 仅在轴、Methods 或形式定义中使用 |
| **sequence length `K`** | multi-item protocol 中序列包含的项目数 | Fig.6–Fig.7 使用时必须写明 sequence length 或 Items (`K`) |
| **transition stage** | progressive protocol 中的状态转移阶段索引 | 不等于 history depth；不等于 sequence length |
| **no-memory reference** | 无先前 retained STSP state 的状态参照 | 不等于 cue-only input |
| **cue-only input** | 只保留部分目标编码事件的输入 | 不等于 no-memory state |
| **singleton state/reference** | slot-matched single-item STSP state | 不与 cue-only/no-memory 合并 |
| **keep probability** | cue 中保留的目标编码事件比例 | 不称 memory strength |
| **effective component number `N_eff`** | constituent-weight distribution 的结构表达量 | 不称容量、可访问项目数或 rescued fraction |
| **functionally rescued fraction** | sequence state 相对 slot-matched singleton 恢复的项目比例 | 不称 `N_eff` 或 retained-item count |

`one/five/ten items before B`、`transition stages 2–10` 和 `Items (K)` 不得在没有限定词的情况下并列出现。不得暗示 progressive Stage 5 等于 five items before B。

## 5. 跨图视觉语法

### 5.1 对象图形

| 对象 | 固定图形责任 |
|---|---|
| inherited STSP state | 绿色点阵／节点场；表示分布式 joint state，不表示实测 cell 数 |
| effective STSP support | 从 inherited state 派生的绿色强度、面积或路径强调；首次必须给出 `G_STSP = u ⊙ x` |
| current input | 中性黑白输入图块；输入身份通过短标签或图像本身定义，不用机制颜色代替 |
| history-conditioned processing | 蓝色 spike/event/短箭头；只表示当前输入处理过程 |
| successor state | 与 inherited state 使用同一状态图形语法，但必须位于下游并标成 `S_(k+1)^(l+1)`；本次改变的子集可用蓝色强调 |
| equal-time passive state | 灰色状态图形＋灰色虚线时间箭头 |
| donor intervention | 橙色 intervention 箭头；state glyph 保持其对象身份，并必须同时显示 donor、receiver、操作 layer 与 held-fixed 边界 |
| overlap gate | 绿色 retained support 与蓝色 input pathway 的明确交叠区域，不用孤立的“overlap”文字代替 |
| behavior/readout | 中性深色 target/readout 图标，不与状态节点混用 |

### 5.2 颜色层级与优先级

跨图机制角色采用现有项目色板：

| 角色 | 主色 | 非颜色冗余 |
|---|---|---|
| observed/current processing/receiver-side effect | `#0072B2` | 实线、实心圆 |
| Layer 1 / early layer identity | `#56B4E9` | 圆形 |
| retained support / overlap / high-support route | `#009E73` | 实心或连续路径 |
| donor / state shuffle / directed perturbation | `#D55E00` | 方形或虚线箭头 |
| history-conditioned residual / pair-specific contrast | `#CC79A7` | 独立 marker 形状 |
| passive/baseline/random/no-memory control | `#666666`、`#999999`、`#D9D9D9` | 虚线、空心 marker |

优先级：

1. 当面板比较 Layer 1–3 时，颜色首先表示 layer，且必须有显式图例；
2. 其他机制面板中，绿色 state/support、橙色 intervention arrow、灰色 counterfactual 与蓝色 current processing 才具有跨图角色；
3. Item A/B/Pair、Behavioral Rescue/Loss、Event classes 和 Original/Donor/Other 是彼此独立的 panel-local 色彩命名空间，必须有显式标签和 marker/line/fill 冗余；
4. 颜色本身不得建立跨图对象身份。跨图交接必须同时依赖 glyph、科学短名、layer 与箭头方向；
5. 同一面板不得让一种颜色同时承担 layer、item、outcome 和 intervention 多类意义。

局部类别保持：

- Item A / Item B / Pair：蓝 / 橙 / 绿，仅用于显式 item-comparison；
- Behavioral Rescue / Loss：蓝 / 橙；
- Event Advance / Recruit / Spike loss：蓝 / 橙 / 深灰；
- Original / Donor / Other：蓝 / 橙 / 浅灰。

### 5.3 操作符必须首次可见定义

任何新干预或任务操作首次进入主图时，必须用最小图形定义一次，之后复用完全相同的图标和短名：

| 操作 | 最小视觉定义 |
|---|---|
| joint-state shuffle / reassignment | 两个 trial 的 joint `u/x` state 之间使用双向交换箭头；标明 Original/Donor |
| selective inherited-Layer-1 STSP substitution | donor inherited Layer-1 state → receiver Layer 1；输出只标为被定向改变的 Layer-2 successor |
| selective post-B Layer-2 successor transplant | donor post-B Layer-2 successor → receiver Layer 2；标明 identical next input、held-fixed retained states 与 fast-state treatment |
| attenuation | state 点阵向基线淡化但不消失 |
| reset | state 点阵回到统一基线状态 |
| equal-time passive evolution | no input 图标＋灰色时间箭头 |
| partial cue | 输入图块被稀疏保留，并与 keep probability 轴连接 |
| overlap | input pathway 与 retained support 的交集区域 |
| area-/energy-matched removal | 与目标 removal 同面积／能量的中性控制图标 |

不得靠图注第一次解释一个已经出现在定量面板中的操作。若版面不允许独立 protocol panel，应使用所属面板顶部装饰带或共享行级 operator key；不得增加无字母 micro-panel。

## 6. 七张图的 reader contract

## Fig.1 — 机制基底：一次事件如何留下 post-firing effective support

**唯一问题**：一次短暂 presynaptic event 如何通过 facilitating STSP 动力学形成放电结束后仍存在的 effective STSP support？

**面板链**：

`a 在固定前馈网络中定位 STSP → b 定义短暂事件 → c 展示 u 与 x 动力学 → d 展示 G_STSP = u ⊙ x`

**允许终点**：模型动力学能够把短暂事件转换为在 firing 归零后仍暂时升高的 effective STSP support。

**禁止终点**：

- Fig.1 本身不证明 network-level content retention；
- 不证明 functional inheritance；
- 不证明 successor formation；
- 不把 deterministic probe 当作 20-network empirical result。

**交给 Fig.2 的唯一对象**：由 `u/x` 动力学产生、可在网络中形成分布式 inherited state 的 STSP substrate。

## Fig.2 — 继承前提：activity-silent state 是否携带并影响信息

**唯一问题**：任务网络中是否存在 firing-silent 但内容可解码、并能影响随后 readout attribution 的 distributed inherited STSP state？

**面板链**：

`a 网络具备任务功能 → b 延迟期 firing 消失 → c joint u/x state 仍可解码 → d joint-state reassignment 改变 Original/Donor attribution`

**允许终点**：在 20 个网络中，distributed joint `u/x` state 在 firing 消失后仍保留内容，并在受控 reassignment 后定向改变 readout attribution。

**禁止终点**：

- 准确率本身不是 STSP 机制证据；
- decodability 本身不是功能因果证据；
- joint-state reassignment 不证明 history 已经被新输入重写。

**交给 Fig.3 的唯一对象**：带明确 layer 的 `S_k^(l)`，作为下一输入到达前的 inherited STSP state。

## Fig.3 — 相同输入条件化：历史是否改变当前处理

**唯一问题**：不同 inherited histories 面对 identical encoded current input `B` 时，是否产生共同输入驱动但历史条件化的处理与结果？

**面板链**：

`a paired exact-input counterfactual → b bidirectional behavioral rescue/loss → c common input-driven update + history-conditioned residual → d residual enrichment at history-differential events`

**允许终点**：相同 `B` 的处理主要由当前输入驱动，但继承历史稳定地改变其处理残差、事件落点和行为结果。

**禁止终点**：

- history 不替代当前输入决定正确答案；
- Fig.3 不单独证明 overlap mechanism；
- Fig.3 不单独证明 downstream causal successor formation。

**交给 Fig.4 的唯一问题**：history-conditioned residual 通过什么空间入口和局部事件机制形成 downstream successor？

## Fig.4 — 层间实现：successor state 如何形成

**唯一问题**：inherited STSP 如何被 current input 选择性读取、转化为早期 firing events，并形成 downstream successor STSP state？

**面板链**：

`a overlap-specific causal entry → b pre-input retained support → c event advance/recruit/spike loss → d Layer-1 STSP contribution → e history-aligned Layer-2 updating → f Layer-1 state substitution redirects Layer-2 successor → g evidence-after-synthesis`

**允许终点**：overlap-aligned inherited STSP 提供空间选择性入口，使当前输入的早期 processing 与 downstream Layer-2 successor formation 具有历史条件性。

**禁止终点**：

- 不宣称逐个 Layer-1 unit 一一写入对应 Layer-2 site；
- donor transfer 不证明必要性、完整 mediation 或唯一性；
- Fig.4g 不增加新证据，也不得移到定量证据之前充当证明。

**交给 Fig.5 的唯一对象**：已形成的 post-input Layer-2 successor state。

## Fig.5 — reuse 与 recurrence：successor 是否成为下一次继承状态

**唯一问题**：post-input successor 是否能够在下一输入边界作为下一层的 inherited STSP state 参与处理，并且同一 input-driven、history-conditioned transition motif 是否在连续输入中反复出现？

**面板链**：

`a across-depth successor reuse after one, five or ten items before B → b overlap-selective entry after a ten-item history → c propagation through a following transition after a five-item history → d natural recurrence across transition stages 2–10 → e history-dependent Behavioral rescue/loss`

**允许终点**：模型内受控 successor transplantation 对 next-input processing 及随后形成的 successor 具有 bounded causal sufficiency；该作用在测试的历史深度中持续存在，通过 state–input overlap 选择性进入，并在没有第二次 transplant 时传播至下一次 transition。在独立 progressive protocol 中，input-associated displacement 相对 equal-time no-input evolution 反复出现，并伴随独立行为协议中的 Rescue/Loss 变化。Supplementary Figs. S3–S5 分别给出空间选择性、跨网络持续性和 recurrence 的 network/variable-resolved 补充结论。

**禁止终点**：

- 不证明 successor transplantation 的 necessity、complete mediation 或 uniqueness；
- Fig.5b 的 non-overlap 与 size-matched random controls 固定为零且只作描述；
- Fig.5c 的 next response 只作描述，following response 与 new successor 才是确认性端点；
- stage-wise displacement 不表示每个 stage 都重复了完整 transplant protocol；
- Fig.5d 的 passive branch 从对应的完整 observed preceding boundary 重新生成，不能传播为后续阶段的父状态；
- transition stage 不等于 history depth，Stage 5 不等于 five items before B；
- 不推断 cross-depth trend、Fig.5b-versus-c endpoint comparison 或 Rescue-minus-Loss contrast；
- accumulated Behavioral loss 不表示 transition motif 已消失。

**Fig.5 的两个并列输出**：

1. **给 Fig.6**：在独立 pair/multi-item structural protocols 中形成的 terminal sequence states，可询问其内部结构；
2. **给 Fig.7**：在独立 cue/intervention protocols 中形成的 retained STSP states，可询问其何时影响 later processing。

两分支复用的是状态**类型语法**，不是 Fig.5 trial 中同一个 state instance。

## Fig.6 — 并列终点 A：terminal state 的结构组织

**唯一问题**：多次转移后的 terminal STSP state 是否保留多个 constituent 及其经历的时间组织，而非 latest-item-only collapse 或无结构扩展？

**面板链**：

`a two-item state retains both constituents → b a separate four-item state identifies preceding order with set and latest D fixed → c effective component number grows across longer histories → d latest item does not dominate those longer histories → e effective area of retained Layer-1 support across sequence length K × delay → f history-specific Layer-1 support morphology across the same K × delay grid`

这里的 a、b、c–d 和 e–f 来自四个明确分开的结构协议；该顺序是读者论证顺序，不表示同一 state instance 从 K=2 连续扩展到 K=10，也不表示 c–d 将 b 的顺序识别外推到 K=10。

**允许终点**：Layer-2 joint `u/x` terminal states 保留多个 constituent；在 item set 和 latest item D 固定时，独立四项目协议仍能识别此前 A/B/C 的经历顺序；更长历史的 component distribution 不坍缩为 latest-only。独立的 Layer-1 `G_STSP` 分析显示 effective-support footprint 与 history-specific morphology 随 sequence length 和 delay 呈现受限形态。两层结果是结构推广，不是同一 state coordinates 的追踪。

**禁止终点**：

- structural order identification 不是 behavioral temporal-order recall；
- 同 set singleton item-by-slot references 下的识别不表示无需这些 references 的 unseen-image generalization；
- 该结果不证明 unique nonlinear binding code；
- `N_eff` 不是容量或可访问项目数；
- similarity 不是 functional recovery；
- Layer-2 joint-state coefficients 与 Layer-1 effective-support morphology 不是同一 coordinates 的纵向追踪；
- 不把 Fig.7 作为 Fig.6 morphology 的 readout 验证。

**终点类型**：结构描述分支闭合，不向 Fig.7 发出因果交接。

## Fig.7 — 并列终点 B：retained STSP state 对 later processing 的条件性影响

**唯一问题**：retained STSP state 在什么 cue content、sequence load、delay 和 pathway-overlap 条件下影响 later processing？

**面板链**：

`a partial-cue recovery of pair constituents → b access across multi-item positions → c cue-content specificity → d Functionally rescued fraction across Items (K) × delay → e targeted high-support overlap contribution → f overlap-gated firing expression`

**允许终点**：retained STSP 对 later processing 的影响需要 incoming cue/content 与 supported pathway 的匹配，并受 sequence load 与 retention delay 限制。

**禁止终点**：

- 不称为 cue-free replay；
- 不称为 perfect multi-item recall；
- 不声称 STSP 单独预测最终类别；
- targeted removal 只证明贡献，不证明 sole encoding；
- STSP × overlap interaction 不表示任一因素单独决定 firing；
- 不声称 Fig.7 直接访问 Fig.6 定义的 morphology。

**终点类型**：条件性功能分支闭合；它是稿件呈现顺序中的最终 operating-boundary 结论，不构成 Fig.6 → Fig.7 的证据边。

## 7. 跨图交接合同

### Fig.1 → Fig.2：从派生 support 到分布式 state

- Fig.1c 显示构成状态的 `u` 与 `x` 动力学，Fig.1d 只显示其派生 product `G_STSP = u ⊙ x`；Fig.2c/d 才分析完整的 network-level joint `u/x` state。
- `firing-silent` 是 Fig.2b 的实测结果；`activity-silent STSP maintenance` 是结合 Fig.2b–d 才允许形成的机制推断，不应用来命名 Fig.1d 的单突触 product。
- 当前 Fig.1d spec 中的 “combined u-x state” 属于待清理的非规范标签；正式重绘应只称 effective STSP support。
- 交接图形必须让读者看到“单事件动力学产生 derived support”与“网络级 distributed joint state 保留内容”属于不同对象与尺度。
- 禁止用裸 `ux` 同时指代 product 和 concatenated state。

### Fig.2 → Fig.3：从存在到被下一输入继承

- Fig.2d 中被 reassigned 的 state 图形，在 Fig.3a 中必须复用为 identical `B` 到达前的 inherited-state 图形。
- Fig.3 不重新证明 firing-silent 或 decodability；它只检验 inherited state 如何条件化 identical current input。
- `Original/Donor` 是 Fig.2d intervention roles；`History 1/2` 是 Fig.3a inherited-history conditions，不能靠同一颜色默认为同义。

### Fig.3 → Fig.4：从 residual/event association 到空间机制

- Fig.3c 定义 common input-driven update 与 history-conditioned residual；Fig.4 不重新画这两个端点。
- Fig.3d 的 `history-differential events` 与 Fig.4c 的 `spike-transition classes` 是两套不同 taxonomy：前者定位 residual enrichment，后者比较 dynamic 与 static-frozen 下的 advancement/recruitment/spike loss。二者通过“下一步解释事件如何产生”的问题连接，不复用为同一 glyph 身份。
- Fig.4新增的是 `overlap route → retained effective support → spike-transition classes → downstream successor`，不是重复“history matters”。
- Aligned/mismatched history 与 overlap/non-overlap pathway 必须使用不同图形语法。

### Fig.4 → Fig.5：从 successor formation 到 successor reuse

这是全篇最关键的跨图连接，但必须区分两个不同干预：

```text
Fig.4f protocol:
donor inherited Layer-1 STSP substitution
        ↓
receiver Layer-2 successor is redirected

handoff object:
post-B Layer-2 successor state
        ↓

Fig.5a–c controlled reuse protocols:
selective post-B Layer-2 successor transplant
        + held-fixed next-input conditions
        ↓
next-input response and new successor
        ↓
overlap-selective entry and following-transition propagation
```

- 跨图复用的是 **post-B Layer-2 successor 这一输出对象**，不是 Fig.4f 的 Layer-1 substitution operator，也不是 donor color；
- Fig.4f 与 Fig.5a 使用不同 intervention 和完整 layer 定义，避免把 inherited-L1 substitution 与 successor-L2 transplant 合并；
- Fig.5a、b、c 是三个相互约束但独立汇总的受控协议：a 检验 one/five/ten-item depth，b 检验 overlap-selective entry，c 检验没有第二次 transplant 时的 following-transition propagation；不得把它们伪装成同一批连续 trial；
- 当前 Fig.5 固定为五个定量面板，不新增 schematic、protocol key 或无字母 micro-panel；donor/receiver、transplanted Layer 2、held-fixed variables 与 fast-state treatment 由图注、Methods 和 Source Data 完整定义；
- Fig.5d 的 progressive recurrence 与 Fig.5e 的 behavior 均为独立协议，不得用连续箭头伪装成 Fig.5a–c 的下游 trial。

### Fig.5 → Fig.6 / Fig.7：显式分叉

- Fig.5末端不提出单一的“下一张图问题”，而是产生两个并列问题：
  1. repeated transitions 后 terminal sequence states 具有什么结构？
  2. independently retained STSP states 在什么 later-input 条件下产生功能影响？
- Fig.6 与 Fig.7 可复用“分布式 STSP state”这一**类型语法**，但必须用各自 protocol 标签明确它们来自独立 pair/multi-item 与 cue/intervention protocols；不得暗示同一个 Fig.5 state instance 流入两图；
- Fig.7 不使用 Fig.6 的 heatmap、`N_eff`、similarity 或 morphology 作为前提；
- Fig.6 与 Fig.7 可以对齐各自的 sequence length / Items (`K`) × delay 网格槽位以便并列观察 operating regimes，但色标、指标、layer、protocol 和解释必须独立。

## 8. 跨图禁止重复与禁止跳跃

### Fig.1 / Fig.2

- Fig.1 只建立动力学 substrate；Fig.2 才建立 network-level inheritance。
- Fig.2 不重新绘制 u、x 的单突触动力学。

### Fig.2 / Fig.3

- Fig.2 不提前引入 aligned/mismatched、common update 或 history residual；
- Fig.3 不重复 accuracy、firing-silent 和 delay decoding。

### Fig.3 / Fig.4

- Fig.3 建立 identical-input history conditioning；
- Fig.4 解释 overlap-gated local implementation 与 downstream successor formation；
- Fig.4 不把 rescue/loss 或 residual 重新画一遍。

### Fig.4 / Fig.5

- Fig.4 终点为单次 successor formation；
- Fig.5 依次建立 across-depth reuse、overlap-selective entry、following-transition propagation，再推广到 iterative updating 和行为后果；
- Fig.5b 的 overlap reset 检验 successor reuse 的选择性入口，不重复 Fig.4 的 local-transition formation 证据；Fig.5 不重新证明 pre-input support 或 early STSP necessity。

### Fig.5 / Fig.6 / Fig.7

- Fig.5 证明 reuse/recurrence；
- Fig.6 只描述 structural organization；
- Fig.7 只描述 conditional effect；
- 不建立 Fig.6 → Fig.7 的 active argument edge；
- 不使用 `support_gain_corr` 或共同“memory strength”连接两分支。

## 9. 面板编排硬规则

每张图进入重排或重绘前，必须先冻结：

1. `figure_question`；
2. `terminal_inference`；
3. `forbidden_inferences`；
4. `semantic_units`；
5. `task_graph`；
6. `comparison_obligations`；
7. 跨图输入对象与输出对象。

随后才允许调整 panel row、slot、plot area、legend 或局部编码。

编排必须满足：

- 每张图只回答一个问题；
- 每个 panel 只承担一个不可替代的科学责任；
- 直接比较保持邻接；
- master schematic 只综合已经展示的证据，不代替证据；
- 一个 panel letter 对应一个连续视觉/坐标单元；
- 新操作首次出现时必须有视觉定义；
- 连接关系通过状态对象、layer、方向和 operator 连续表达，不靠完整解释句填补；
- 图面不加入 panel title、正文句、样本量段落或统计方法说明；
- 图注继续承担 `n`、CI、test、held-fixed variables、排除项和完整 protocol。

## 10. 无文字故事验收

在隐藏正文与长图注、仅保留 artwork 必要短标签时，独立读者应能在 30–60 秒内回答：

1. 哪个图建立 STSP substrate？
2. 哪个对象是 inherited state，哪个是 derived effective support？
3. 当前输入如何进入状态转移？
4. successor 在哪一层形成？
5. successor 如何在下一输入边界作为下一层的 inherited STSP state 参与处理？
6. 哪个证据区分 observed update 与 passive evolution？
7. Fig.6 与 Fig.7 为什么是并列分支而不是因果串联？
8. 哪些指标是 structure，哪些是 conditional function？

必须通过以下审计：

- **state identity audit**：同一 state object 跨图使用同一 glyph、科学短名、layer 和方向；独立 protocol 只复用类型语法，不伪装成同一个 state instance；
- **operator audit**：shuffle、transplant、reset、passive、partial cue、overlap 均首次可见定义；
- **layer-direction audit**：每个 causal arrow 的 source layer、processing layer 和 successor layer明确；
- **term audit**：joint `u/x`、effective support、state、update、passive、static-frozen 不混用；
- **branch audit**：不存在 Fig.6 → Fig.7 因果箭头或共同 memory-strength 指标；
- **denominator audit**：behavioral rescue/loss 各自 opportunity set 不被视觉合并；
- **metric-class audit**：`N_eff`、similarity、donor-transfer、state displacement、Functionally rescued fraction、firing change 不互相改名；
- **color audit**：同一面板内颜色不同时编码 layer、item 和 intervention；
- **grayscale audit**：所有核心对比均有 marker/line/fill 冗余。

## 11. 本合同冻结前仍需作者确认的设计决策

以下属于故事和视觉编排决定，不改变证据，但需在正式改图前逐项确认：

1. 是否在实际 artwork 中采用带完整 layer index 的 `S_k^(l) + I_(k+1) → S_(k+1)^(l+1)`；任何缩略版都不得删除 layer direction；
2. Fig.2d 的 joint-state-shuffle operator 是否在所属 panel 顶部首次可见定义；
3. Fig.3a 是否把 post-B downstream state 直接标为 successor state，而不继续使用泛称 `Post-B STSP state`；
4. Fig.4g 保持证据后综合的前提下，哪些状态 glyph 可作为跨图类型词典复用，以及如何删除同层 rewrite 暗示；
5. Fig.5结束处如何在不增加解释句的前提下表达 Fig.6/Fig.7 的独立 protocol 并列分叉；
6. Fig.6 与 Fig.7 各自的 sequence length / Items (`K`) × delay heatmap 是否采用一致的网格方向和槽位语法，但保持独立色标、指标、layer 与 protocol；
7. 是否将所有 event-level `Loss` 图面标签明确改为 `Spike loss`，并将另一个功能指标写全为 `Functionally rescued fraction`。

已确认：Fig.5 保持五个定量面板，不加入 schematic、protocol key 或 micro-panel。

只有作者确认上述设计决定后，才进入逐图 panel 重排、spec 修改、plot-only 重绘和最终视觉 QA。
