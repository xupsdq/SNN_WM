# V6 正式版与“副本”版本逐项对比审查

**日期：** 2026-08-14  
**审查对象：**

- 正式主文：`docs/paper/v6.docx`
- 更新主文：`docs/paper/v6 - 副本.docx`
- 正式补充材料：`docs/paper/supplementary_information.docx`
- 更新补充材料：`docs/paper/supplementary_information - 副本.docx`

## 1. 结论先行

**新版不是“整体都更好”，但它是一个有价值的修订来源。**

- 主文的 Results/Discussion 多数局部修改确实改善了问题推进、机制方向和段落收束；其中 P027、P048、P051、P054 的提升最明确。
- Main Methods → Supplementary Methods 的重构方向正确，主文更精炼，17 个迁移公式也完整保留。
- 但新版尚不适合整体替换正式版：若干 Results 开头因追求简洁而丢失了关键实验关系；Methods 丢失了三类控制的操作定义并出现两处新的准确性问题；新版 Supplementary Table S2 虽然数值正确，却是明显的内部审计表，而不是可投稿的读者表，并且实际渲染发生横向裁切。

**推荐策略：选择性吸收新版修改，不要直接把两份“副本”提升为新的正式稿。**

## 2. 审查方法与范围

本次比较针对当前磁盘上的四份 DOCX，而不是 2026-08-12 映射文件所冻结的旧哈希。两份 DOCX 均无 Word 跟踪修订记录，因此差异由 OOXML 正文、表格、公式、媒体和样式直接提取并重新对齐。

### 2.1 文件与规模

| 对象 | 正式版 | 副本 | 变化 |
|---|---:|---:|---:|
| 主文非空段落 | 186 | 148 | −38 |
| 主文估算词数 | 9,780 | 7,718 | −2,062 |
| 主文 Methods 估算词数 | 3,934 | 1,873 | −2,061 |
| 补充材料估算词数（正文+表格） | 2,984 | 8,012 | +5,028 |
| LibreOffice PDF 页数（仅用于本次版式 QA） | 主文 34 / 补充 12 | 主文 27 / 补充 21 | Methods 大量转移至补充材料 |

### 2.2 已核验的不变项

- 主文和补充材料各自的 7 个嵌入媒体文件按内容哈希完全一致；本轮没有更换图件。
- 正式主文 Equations (6)–(22) 共 17 个公式，完整迁移为 Supplementary Equations (S1)–(S17)。公式顺序、OMML 数学结构和正文交叉引用一致。
- Main Equations (1)–(5) 保留在主文。
- 对文本完全相同且唯一匹配的段落，未检出新的粗粒度加粗、斜体、上下标或段落样式变化。
- Supplementary Methods 的 S1–S17 在 LibreOffice 26.2.2.2 渲染中未见公式裁切或明显破损。

### 2.3 判定标签

- **明确改善：** 建议保留。
- **改善但需微调：** 新方向优于旧文，但不能原样采用。
- **混合：** 同时有真实收益与真实损失，应重写而非简单保留/回退。
- **退步：** 旧文更准确或更自然；若无更好的第三版，应回退。
- **阻断：** 当前状态不能进入正式投稿稿件。

---

## 3. 主文 Results 与 Discussion：逐处判定

正文共有 10 个 Results/Discussion 替换块。

| ID | 位置 | 新版的主要变化 | 判定 | 对全文的影响 | 建议 |
|---|---|---|---|---|---|
| R1 | Results P018 | 将单句改为“跨 firing-silent interval 保留信息”与“影响后续输出”两个问题 | **改善但需微调** | 两级证据更清楚，问题—检验关系更自然；但 earlier input/preceding input、later computation/subsequent output 略重复 | 保留结构；压缩重复，并优先用 “without sustained firing between inputs” |
| R2 | Results P021 | 从证据复述提升为 “functional continuity … without sustained activity” | **改善但需微调** | 段末价值更强；但 “without sustained activity” 超过直接证据，且与 P024 重复 “functional continuity” | 改为 “without sustained firing between inputs”，并消除 P021/P024 复现 |
| R3 | Results P024 | 删除 identical-B 具体设计，仅保留继承状态能否调制新输入的问题 | **混合** | 开头更像科学问题，但删掉了隔离 history effect 的关键诊断条件；P025 的 “the same B” 变成局部悬空 | 保留新版问题句，同时恢复 “different inherited states condition the same newly arriving input” |
| R4 | Results P027 | 明确 current input 提供共同组织，history 选择性调制更新和行为 | **明确改善** | 与核心合同中的“新输入主导、历史提供条件”完全一致；句群收束有力 | 保留 |
| R5 | Results P030 | 从方法预告改为机制问题：“inherited state 如何 shape transition” | **改善但需微调** | 问题驱动性更强，但把 Layer 1 STSP 与 downstream successor 抽象掉了 | 保留问题式开头，同时明确 inherited STSP → current processing → downstream successor |
| R6 | Results P036 | 将 successor reuse 与 recurrence 两个问题压成宽泛的序列问题 | **混合** | 更简洁，但削弱了全文最关键的桥：successor reuse 和 matched-passive recurrence 被混成 general unfolding | 重写为两个并列的精确问题，不建议原样采用 |
| R7 | Results P042 | 一句扩成三句，增加 recurrence→representation 的概念桥 | **混合偏退步** | 宏观连接更明确；但 27 词扩为约 48 词，反复出现 process/transitions/history/represented/organized，并把“成分+配置”这一真实端点变成宽泛 representation | 不原样保留；压回一句，并同时保留“structured history”与“multiple constituent contributions and experienced configuration” |
| R8 | Results P048 | 删除 “access that organization”，改问 retained history 在何种条件下参与后续处理 | **明确改善** | 修复 Fig.6 morphology → Fig.7 function 的错误因果暗示，使两个模块保持并行 | 保留；可加 “separately” |
| R9 | Results P051 | 将泛化结论改为 cue-content match→readout、pathway overlap→early processing | **明确改善** | 两条证据各自承担不同结论，逻辑更完整，段末不再只是标题复述 | 保留 |
| R10 | Discussion P054 | 改为 inherited STSP 条件化当前处理、该处理形成 downstream successor | **明确改善，但需全稿联动** | 本轮最重要的科学修正；机制方向、当前输入与历史职责均更清楚 | 保留并加 tested-model/sequence 边界；必须同步修正仍写同层 rewriting 的其他段落 |

### 3.1 三处不能因“更简洁”而丢掉的关系

1. **P024：** identical input 是排除当前输入差异的诊断杠杆，不是可省略的程序细节。
2. **P036：** post-B Layer 2 successor reuse 与 observed-versus-passive recurrence 是两个不同证据环，必须分别命名。
3. **P042：** terminal morphology 只回答 constituent retention 与 configuration specificity；不能泛化成对“representation”整体的证明。

### 3.2 R7 的综合裁定

新版 P042 的宏观意图是对的，但语言实现不如旧版。建议的合并方向为：

> Having established recurrent history-conditioned updating, we next asked whether the resulting terminal STSP state preserved structured history—specifically, multiple constituent contributions and their experienced configuration.

该方向保留新版的 recurrence→outcome 桥，同时避免把 morphology 写成后续 conditional function 的因果前提。

---

## 4. 主文 Methods 与 Supplementary Methods：逐处判定

Main Methods 的 12 个差异块必须与新增 Supplementary Methods 合并审查，不能把“主文删除”单独视为信息丢失。

| ID | 位置/变化 | 判定 | 具体审查 | 建议 |
|---|---|---|---|---|
| M11 | Input encoding and network model → architecture；将编码与网络合并 | **改善但需微调** | 主文更紧凑，核心架构和 readout rule 均保留；P062 一段同时承载编码、数据集、架构和读出，密度偏高 | 保留重构；可拆成两段 |
| M12 | “STSP-scaled excitatory synaptic input to the layer” → “STSP-scaled excitatory input” | **轻微改善/等价** | Equation (1) 已提供语境，没有信息损失 | 保留 |
| M13 | 压缩 threshold、decision gate、refractory 和 lateral inhibition | **混合** | 大部分更流畅；但新版语法把 “7 × 7 Gaussian field” 延伸到 Layer 3，而旧文只把 7×7 Gaussian 限定于 Layers 1–2 | 恢复旧版的分层限定，确认 Layer 3 形态后再写 |
| M14 | effective presynaptic support → effective STSP support | **明确改善** | 与 Results、图注和 conductance/STSP 区分更一致 | 保留 |
| M15 | 将 elementwise product 与 conductance distinction 合为一句 | **明确改善** | 更紧凑且无含义损失 | 保留 |
| M16 | 压缩 post-spike 更新顺序 | **明确改善** | recovery-before-update、spiking/nonspiking site 和 [0,1] 边界均保留 | 保留 |
| M17a | 训练公式 (6)–(13) 转移为 S1–S8 | **明确改善** | 8 个公式完整且交叉引用正确；主文更符合“最低科学必要”原则 | 保留迁移 |
| M17b | post-training accuracy/readout 细节转入补充材料 | **改善** | 91.158% 与 readout rule 均仍存在 | 保留，但须与 90.95% assay endpoint 清楚区分 |
| M17c | decoding/restoration 压缩 | **混合** | 恢复流程基本保留；但主文 P081 将真实的 digit-class decoding 写成 “Item identity”，并把 φ 的 layer-specific 范围写得含糊；补充 Methods 还漏掉 100/200/400/800/1,200 ms 采样时点 | 改为 digit-class；写明 within the specified layer；补回采样时点 |
| M17d | 三类控制压为 P083 一句 | **退步，较严重** | static-frozen、equal-time passive、no-memory 的操作定义没有迁移到补充材料；复现性下降 | 在 Supplementary “State readout and restoration details” 中恢复三条最小操作定义 |
| M17e | 新增 “only u and x evolved” | **科学表述错误** | fast neuronal variables（V、conductance、refractory、inhibition）在 post-training simulation 中同样演化；真正固定的是 long-term weights/learned thresholds | 改为 “u and x were the only plastic synaptic variables” 或明确 fast state 仍动态演化 |
| M18 | exact-input 细节与 (14)–(17) → S9–S12 | **改善** | byte-identical B、branch definition、residual、阈值、transfer scope 均保留；公式完整 | 保留；统一 “history-conditioned residual” 等正式术语 |
| M19 | successor reuse/recurrence 压缩 | **改善但需术语统一** | pairing、receiver layers、fast-state equalization、identical C、passive correction 均保留 | 保留；将 own-state sham/event map 改为正式读者术语 |
| M20 | morphology 与 (18)–(20) → S13–S15 | **改善** | Layer 2 decomposition 与 Layer 1 coefficient-free morphology 仍保持不同尺度；fallback 与 grids 均保留 | 保留；统一渲染为 `N_eff` 的正式数学排版而非字面下划线 |
| M21 | cue/conditional function 与 (21)–(22) → S16–S17 | **改善** | cue strengths、20 masks、AUC、cue classes、4×4 grid、overlap definitions 均保留 | 保留；统一 “area- and input-energy–matched removal” |
| M22 | Statistics 五段压成一段并指向 Table S2 | **混合且当前阻断** | 主文更流畅，但 α=0.05 等信息移除；更重要的是 P102 声称 Table S2 提供完整端点/统计信息，而表内明确写 “not encoded/must be completed” | 完成 Table S2 后再接受；否则收缩 P102 的完整性声明 |

### 4.1 17 个公式迁移核对

| 正式主文 | 新补充材料 | 结论 |
|---|---|---|
| (6)–(13) | (S1)–(S8) | 训练公式完整迁移 |
| (14)–(17) | (S9)–(S12) | identical-input decomposition 完整迁移 |
| (18)–(20) | (S13)–(S15) | morphology decomposition 完整迁移 |
| (21)–(22) | (S16)–(S17) | conditional-function gains 完整迁移 |

17 对公式的 OMML 结构逐对一致，仅编号发生预期变化；可接受。

### 4.2 真正丢失而不是迁移的内容

新版组合稿中，以下三条操作定义没有在 Supplementary Methods 恢复：

- **static-frozen：** `u = U, x = 1` 在 assay 中保持固定；
- **equal-time passive：** 从 inherited boundary 出发，在 zero input 下按普通 recovery 演化相同步数；
- **no-memory：** 从 resting network state 出发并匹配总时长。

这不是文风偏好，而是复现性缺口。

---

## 5. 补充材料的其他逐项修改

### S1｜开场说明与新增 Supplementary Methods

**判定：整体方向改善，但需完成后才能接受。**

优点：

- 主文 Methods 的细节有明确承接位置；
- 8 个方法模块顺序合理：encoding → training → restoration → exact input → successor reuse → morphology → conditional function → statistics；
- 数学推导与协议细节大部分完整保留；
- 开场说明比旧版更简洁。

问题：

- 三类控制操作定义丢失；
- delay-decoding sampling times 未在 Methods 中写出；
- 在 Word outline 中，Supplementary Figures 和 Tables 仍落在最后一个 Heading 3 “Statistical analysis details” 之下，因为没有新的同级 Heading 2 结束 Methods 层级。

**建议：** 保留迁移架构；补控制定义与时间点；增加 “Supplementary Figures” 和 “Supplementary Tables” 同级标题或修正 outline level。

### S2｜Supplementary Fig. S2：early Layer 3 successor → class-score vector

**判定：明确改善，建议保留。**

- 实际端点是十类 class-score vector，不是 post-C Layer 3 `u/x` successor。
- 新术语避免与 Fig.5b 的真正 Layer 3 successor 混淆。
- 建议在图注或 Supplementary Methods 首次定义 class-score vector，并更新 `MAIN_SUPPLEMENT_SENTENCE_MAPPING_20260812.md` 中旧锚点。

### S3｜Supplementary Table S1：28 × 7 → 28 × 28

**判定：明确改善，必须保留。**

- MNIST 输入和 encoder 均为 28 × 28；旧版 28 × 7 是错误。

### S4｜Supplementary Table S2 整体替换

**总判定：数值审计能力改善，但作为读者/投稿表格明显退步；当前为阻断项。**

#### S4.1 科学内容的真实收益

- 旧表 8 列只记录 endpoint/test/family/null/adjusted P；新表把 main inferential contrasts 拆为 M01–M30，并加入 estimate、CI、test、alternative、statistic、df、raw P 和 adjusted P。
- M01–M30 的 estimate、CI、test、raw P 与 adjusted P 已逐行对照其命名 metrics 文件；**没有发现数值抄写错误。**
- 将复合 endpoint 拆成一行一个 contrast 是合理的。

#### S4.2 当前不能投稿的原因

1. M01–M30 每行都含 `Undefined n = not encoded`；绝大多数还含 `Family = not encoded in aggregate` 和 `Family size = not encoded`。
2. 表内出现 `prefix_k1`、`did`、`layer1_only_ux_swap`、`predeclared_recomputed` 等内部实现词。
3. 每行暴露仓库内部路径，并迫使正文解释“内部 fig 编号比稿件小 1”；这属于 provenance/QA，不是读者表述。
4. 主表有 26 列，continuation 有 11 列，横向结构不适合普通补充材料页面。
5. 新表删除了旧表中的 Fig.2 descriptive rows 和 Supplementary Fig. S2 inferential rows；若刻意改为 inferential-only，应明确范围并在其他位置保留 descriptive registry。
6. “Fig.5 recurrence audit” continuation 把 3 个主文 endpoint 误称为 secondary；只有 minimum-across-stages 行是 secondary/sensitivity。
7. reader-facing conditions 被 code tokens 取代：K/delay grid、exact-match eligibility、window、comparison group 等信息反而更难读。

#### S4.3 实际版式 QA

- 新补充材料页面可用宽度为 8,640 twips；26 列 Table S2 的网格总宽度为 23,400 twips。
- LibreOffice 26.2.2.2 导出的 PDF 中，Table S2 主表在第 17–20 页只显示到约第 12 列，右侧第 13–26 列整体越过页面边界并被裁切。
- 旧版 8 列 Table S2 在同样渲染流程中完整落在页面内。

因此，这不是“字体略小”的可选优化，而是确定的提交阻断。

#### S4.4 M01–M30 逐行/逐组审计

| 行 | 数值核对 | 需要处理的问题 |
|---|---|---|
| M01–M02 | rescue/loss 数值、CI、P 均正确 | 补全 family/undefined metadata；改为读者术语 |
| M03–M04 | residual ratio/common cosine 数值正确 | `Contrast = estimate_minus_threshold`，但 Estimate 列放的是 raw estimate；应改标签或真正报告 margin |
| M05 | 数值正确 | 恢复明确 window 和 reader-facing endpoint 名 |
| M06–M08 | 数值正确 | 表/metrics 为 percentage points，主文仍写 `%`；同步改主文；`overlap_reset` 需改为科学条件 |
| M09–M11 | 数值正确 | 新表丢掉 pre-input endpoint/window 和 pathway-group 定义 |
| M12–M13 | 数值正确 | 将 `first_50_ms` 改为 “first 50 ms” |
| M14 | 数值正确 | unit 是 percentage points，主文仍写 8.52%；`did` 应写完整 scientific contrast |
| M15 | 数值正确 | 明确 identical B、fast-state equalization、Layer 1 u/x substitution；与 S2 confirmatory inference 区分 |
| M16–M19 | 数值正确 | “paired donor–receiver networks” 容易把 pair 当独立单位；应写 within each of 20 independently trained networks；四行 family 已知，不应留空 |
| M20 | 数值正确 | 用 experienced pair vs one-constituent-held shuffled composite 等正式语言替代 `true_minus_shuffled` |
| M21–M24 | 数值正确 | 替换 `SAB_vs_S0` 等实现名，并补 cue-strength integration/condition |
| M25 | 数值正确 | 恢复 K=10、400 ms；difference 应核定为 percentage points，不应笼统写 percent |
| M26–M27 | 数值正确 | 恢复 K=7、400 ms 和 cue definitions；核定 percentage points |
| M28 | 数值正确 | 恢复 4×4 load-delay grid 与 within-network standardized coefficient |
| M29 | 数值正确 | 写明 exact area-and-energy matched eligibility；核定 percentage points |
| M30 | 数值正确 | 表为 percentage points，主文仍写 16.0%；恢复 primary 10-ms window、support split、overlap threshold |

#### S4.5 Fig.5 continuation 四行

| 行 | 判定 |
|---|---|
| mean observed-minus-passive | 数值正确；它是主文 Fig.5c endpoint，不是 secondary |
| minimum observed-minus-passive | 数值正确；可作为 secondary/sensitivity |
| loss K5−K1 | 数值正确；主文 endpoint，单位应为 percentage points |
| rescue K5−K1 | 数值正确；主文 endpoint，单位应为 percentage points |

#### S4.6 推荐的表格重构

1. **Endpoint definition table：** Figure、contrast、eligible set、aggregation、window、estimate、unit、95% CI。
2. **Inference table：** Figure/ID、null、test、alternative、statistic/df、raw P、family、adjusted P。
3. **Machine-readable Source Data：** 完整路径、status、全精度、内部 ID、provenance。

不要把完整内部 registry 直接粘贴进 Word。

---

## 6. 对全文自然感与逻辑一致性的综合影响

### 6.1 新版真正改善了什么

1. **Current input 与 history 的职责更清楚。** P027/P054 现在都让 current input 提供共同组织，让 inherited history 选择性调制。
2. **Inter-layer successor direction 在 P054 中得到正确表达。** 这是比单纯润色更重要的科学修正。
3. **Morphology 与 conditional function 分开。** P048/P051 不再暗示 Fig.7 访问 Fig.6 所定义的 morphology。
4. **Results 的多个段首/段末更像“问题→回答”，而不是实验清单。**
5. **Methods 的层级更符合主文与补充材料分工。**
6. **S2 endpoint 和 Table S1 input dimension 得到实质性纠正。**

### 6.2 新版为何仍不能整体替换

#### A. 正确的新 P054 与全文其他高可见位置冲突

新版 P054 已改为 downstream successor formation，但下列未改段落仍写 inherited state/self-state 被 rewritten：Abstract P005、Introduction P009/P010、Results overview P012、Results P033、Fig.4g caption P034、Results P039、Discussion P055。

这不是 P054 本身的错误，但局部改正确后，全文不一致变得更显眼。必须以 P054 的层间方向做全稿联动。

#### B. endpoint 术语仍不一致

- P019 先写 digit-class decoding，随后又写 item-specific information；Fig.2 caption 和新 Methods P081 又写 item identity。
- 90.95% 仍被称为 delayed-recall/recall accuracy；91.158% 是 STSP-disabled、rescaling 前的 10,000-image test accuracy。两者仍未在主文和补充材料中充分区分。
- S2 已改成 class-score vector，但 main P032 仍只写 early Layer 3 response；不算错误，但 traceability 不够。

#### C. 百分比与百分点仍系统混用

需全文同步处理：P031、P038、P049、P050，以及 Table S2 M06–M08、M14、M25–M27、M29、M30。差值是 percentage points 时不得写 `%`。

#### D. 论证边界仍有未改问题

- Discussion P056 仍容易把 structural morphology 与 conditional readout 提升为 chunking/capacity 结论；应明确为受支持解释或假说，而不是直接证据。
- P058 的 “without recurrent persistent activity” 仍不准确；直接排除的是 persistent firing between inputs。
- `N_eff`/`Neff`/数学下标写法和 effective STSP support/synaptic efficacy 仍未全稿统一。

#### E. 维护文件已失效

`MAIN_SUPPLEMENT_SENTENCE_MAPPING_20260812.md` 的冻结哈希既不匹配当前正式版，也不匹配副本；副本又改变了主文 anchors、插入 Supplementary Methods 并更名 S2 endpoint。按该文件自身维护规则，必须重新生成或重新核验映射。

#### F. 文档仍是带批注工作稿

新版主文仍保留 6 条未解决 Word 批注，其中部分锚定段落未修改。它不应作为 clean submission artifact。

---

## 7. 推荐采用策略

### 7.1 可直接保留的修改

- R4 P027；R8 P048；R9 P051；R10 P054 的机制方向。
- M12、M14、M15、M16。
- 公式迁移 (6)–(22) → (S1)–(S17)。
- Supplementary Fig. S2 的 class-score vector 修正。
- Table S1 的 28 × 28 修正。

### 7.2 应保留思路但重写的修改

- R1、R2、R3、R5、R6、R7。
- M11、M13、M17、M18–M22 中列出的术语、控制、范围和交叉引用问题。
- Supplementary Methods 的整体迁移架构。

### 7.3 当前不能接受的修改

- 新 Supplementary Table S2 的 26 列 Word 版本及其未完成 metadata。
- P079 “only u and x evolved”。
- P081 “Item identity was decoded”。
- 丢失三类控制操作定义的 Methods 状态。
- 将 Fig.5 三个主 endpoint 标成 secondary continuation。

## 8. 优先修改顺序

1. **先统一核心层间方向：** 用新版 P054 的关系修正 Abstract、Introduction、Results、Fig.4g 和 Discussion P055。
2. **修 Results 三个关键 hinge：** P024、P036、P042；随后微调 P018/P021/P030。
3. **修 Methods 准确性与复现性：** P079、P081、P069；补三类控制与 delay sampling times。
4. **重做 Table S2：** 完成 family/undefined metadata、修单位、改为两张紧凑读者表；内部 26 列 registry 进入 Source Data。
5. **统一 endpoint/术语：** digit class、两类 accuracy、percentage points、class-score vector、`N_eff`、effective STSP support。
6. **修 Supplement outline：** Supplementary Figures/Tables 与 Supplementary Methods 同级。
7. **更新 main–supplement mapping 和哈希。**
8. **生成 clean versioned copy：** 解决/删除已完成批注，再做 DOCX/PDF 渲染 QA。

## 9. 最终判断

若只问“副本是否让论文更好”，答案是：

> **主文的核心叙事多数变好，Methods 架构也变好；但补充统计表和若干过度压缩使组合稿尚未整体变好到可以直接替代正式稿。**

最稳妥的做法不是回退全部新版，也不是整体接受新版，而是：**以副本为修订来源，选择性合并其明确改善项，并在一次全稿一致性修订后再生成新的正式版本。**
