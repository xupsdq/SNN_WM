# V5 全文读者视角完整审稿报告（v5_discussion_revised_20260808.docx）

日期：2026-08-08
对象：`docs/archive/paper/intermediate-drafts/v5_discussion_revised_20260808.docx`（目标期刊 Communications Biology）
审稿定位：**读者与投稿完整性视角**。Intro 提出的问题与 Results→结论的科学链视为冻结权威，本报告不重新质疑新颖性、证据充分性、实验/端点/统计设计，也不要求新模拟或新数据。

## 0. 审稿方案（调研→综合→执行）

### 0.1 调研基础
- 本地技能：`manuscript-sentence-revision`（结构锁定句群润色）、`write-paper-results`（Results 论证审计）、`paper-figure`（图件 QA + Net_torch 版式/色彩契约）、`pdf`（渲染检查）及其全部相关 references。
- 仓库权威：`NATURE_PORTFOLIO_REVIEW_STANDARD_20260807.md`（D1–D8 rubric、P0–P3 分级）、`CORE_SCIENTIFIC_LOGIC_CONTRACT.md`、`RESULTS_EVIDENCE_BOUNDARIES.md`、`SOURCE_IDENTITY.json`、submission package 状态文件。

### 0.2 综合形成的审稿方案（六条审稿线）
1. **语言与句群阅读线**：语法/冠词/时态、连接词精确性、句子完成度、信息密度、词汇链、节奏与模板重复。
2. **章节功能与导航线**：段落职能、转承、定义时机、跨节冗余、读者定向。
3. **方法/统计/可复现完整度线**：参数自足、符号与公式定义、推断单元、α/单双侧/P 值、端点可追踪性、软件/数据/代码身份（只查完整度，不挑战冻结的科学选择）。
4. **图表与图注线**：成品尺寸可读性、阅读顺序、面板/图例/轴一致性、语义颜色连续、图注自足（n/CI/检验/P 值）、图文一致、无障碍文本。
5. **形式与投稿就绪线**：标题/前页、公式、引用与 DOI、图号交叉引用、样式、页眉页脚、破损字形、修订痕迹、元数据、Data/Code/Acknowledgements/Contributions/Competing interests。
6. **整稿读者模拟线**：首读理解、术语负荷、需要回读的位置、修订顺序。

### 0.3 证据纪律
- 每条意见 = 定位 + 原文/可观察症状 + 读者影响 + 最小修复；未直接观察的推断显式标注 `[INFERENCE]`。
- 不给出 accept/reject 结论；明示"不是问题"清单防 scope creep。
- 跨线去重：语言线（L-F#，46 条）、正式完整度线（F-F#，30 条）、图件视觉线（V-F#，1 条，已被哈希证据否决，见 §4）、主线（M#，本报告整合者的独立发现）。
- 每条意见均与 P0–P3 分级绑定：P0 阻断完整性；P1 投稿前必须修；P2 应修；P3 可选。

### 0.4 执行与核验
- 全文 168 段、22 个编号公式（OMML）、6 幅内嵌图、33 条参考文献逐段/逐图/逐式检查；
- 渲染核验：LibreOffice→PDF（28 页），公式字形（τ/Δ/⊙/‖/∈/∑ 等）全部正常渲染、无豆腐块；6 图分别落在第 4/6/8/10/12/14 页，图注同页；
- 哈希核验：6 幅内嵌图与冻结产物 `results/paper_figure_multi_seed/final_six_figures_v5_c5_revised_20260804_r2/fig{1..6}/figures/fig{1..6}.png` 逐字节一致（sha256 前 12 位逐一匹配），rel 映射（Fig1→image1，Fig2→image7，Fig3→image8，Fig4→image9，Fig5→image10，Fig6→image11）正确；
- 引用覆盖核验：正文上标引用 1–32 + Data availability 的 ref. 33，全部 33 条均有引用且均有文献条目，无孤儿引用、无缺失条目；
- 元数据/包体检：无修订痕迹、无批注；页脚含页码域；`docProps` 与媒体包存在若干卫生问题（见 §5）。

---

## 1. 总体读者判定

**结论：文稿处于"可读、论证完整、接近可投稿"状态，但"方法自足复现"与"统计披露完整度"两类问题必须在组包前关闭；语言与图件层面无阻断性问题。**

- 科学叙事：Results 六节全部遵循 Action→Data→Inference，六个小节标题均为主张式（无"Figure X shows"叙述、无方法名标题）；路线图段 [14] 与章节转承 [27→31→36→37→41] 干净。
- 读者体验：首读最大的障碍是**核心术语首次使用未加注释**（u/x、fast state、static-frozen、K、loss 等），读者需回退到 Methods 才能确认含义；这是 P2 集中区，修复成本低、收益大。
- 投稿就绪：存在 6 条 P1（全部集中在 Methods 参数/统计披露/代码快照身份），1 条 P1 性质的提交前完整性风险（Supplementary 图从未被正文引用，且 package 状态文件自报 submission_ready: false）。

---

## 2. P1 —— 投稿前必须修（6 条）

**R1（P1）Methods 全部模型参数仅有符号、无数值。**
定位：Methods "Spiking and short-term synaptic plasticity dynamics" / "Training and fixed-circuit simulations"。
证据：τe、τD、τF、τ+、U、Δt、Cm、gm、VL、VE、阈值、reset、不应期、η、xtar、wmax、wmin、Rmax、Pmax 及式(13)的平滑参数 a 全部仅以符号出现（式 1–13）；架构量同样无数值：层/特征图规模、核与步长（仅 5×5 感受野出现于重叠分析）、MNIST 划分与 trial 数、**决策窗时长**（"within the decision window"，窗口从未量化）。
影响：CommsBio referee guide 明确要求建模研究提供参数与推导（rubric D5；B3 参数自足表未落地）；读者无法复现甚至无法做量纲核查；决策窗是端点定义核心，缺时长等于缺协议常数。
最小修复：按 B3 表格补模型参数表（Δt、时间常数、阈值/reset/refractory、U/τ_fac/τ_rec、inhibition/top-k、增益补偿、窗口、MNIST 划分、trial 数、决策窗），或逐项内联给出。
来源：F-F1、F-F2。

**R2（P1）未声明显著性水平 α。**
定位：Methods "Statistics and reproducibility"。
证据：全文无任何 α 声明（唯一 0.05 是 margin 阈值 [100] 与 overlap 阈值 [118]）。
最小修复：统计节补 "α = 0.05（双侧）" 等声明，并同步到图注星标约定。
来源：F-F11。

**R3（P1）全文无任何精确 P 值，16 处全部为 "P < 0.001"。**
定位：Results 全部推断句。
证据：全文无 "P = 0.…" 出现；rubric D4/CommsBio 要求每个检验的实际 P 值（"不允许只写 significant 或 P < 0.05"）。
最小修复：至少在头条对比处给出精确值（如 P = 1.2×10⁻⁵），或确认每检验精确值完整落于 Source Data 统计表并在 Methods 明确指向；图注同步。
来源：F-F12。

**R4（P1）confirmatory-19 敏感性分析缺失。**
定位：Fig. 4 移植端点（C5）/Statistics。
证据：`RESULTS_EVIDENCE_BOUNDARIES.md` 明确要求 "retain the confirmatory-19 sensitivity that excludes development seed 1000"；正文与图注全文检索无 "confirmatory" 出现。
影响：证据边界权威记录要求报告该敏感性；缺失会预支审稿人提问。
最小修复：Statistics 或对应 Results 句补一句 19-network 敏感性结果。
来源：F-F14。

**R5（P1）代码快照身份不一致（提交号过期）。**
定位：Code availability [128]。
证据：正文引用 `ef5eabee7594a3b59f44e9c9b6b940144143fd4b`（2026-07-14，"Restore manuscript statistics reproducibility"）；仓库 HEAD 与打包快照 `SOURCE_IDENTITY.json` 为 `93d9b8295fdfbd603d3f181a3500773cb4689a75`（2026-08-01，"feat: finalize six main paper figures"），发行 zip 名即 `net_torch_final_six_source_93d9b829.zip`；ef5eabe 早于 final-six 成图约 2.5 周。`[INFERENCE]` 图件产物（2026-08-04 的 r2 目录）在 HEAD 未跟踪，仓库本身无法证明某提交可复现该批图，权威物是发行 zip。
影响：按正文提交号克隆的读者看不到产出本稿结果的代码（rubric A7 已知 P1，未关闭）。
最小修复：正文提交号更新为 93d9b829…，并核验 zip 内容与该提交一致。
来源：F-F18（含 git 实证）。

---

## 3. P2 —— 应修（约 22 条，按主题合并）

### 3.1 术语首次使用（读者首读障碍，语言线 13 条 P2 的主体）

**R6 术语一次性加注（逐条最小修复，均不改变主张）：**

| ID | 定位 | 现象 | 最小修复 |
|---|---|---|---|
| R6a | [16] "joint u/x state" | 中心构造首次出现无注释；定义在 Methods [61]/式(4) | 加注 "(the STSP utilization and resource variables)"（L-F1） |
| R6b | [27] "fast variables" | 定义在 [90]；且标签漂移：fast variables/fast state/fast-state initialization | 加注 "fast (non-synaptic) state variables"，全文统一 "fast state"（L-F2） |
| R6c | [26] "static-frozen baseline" | 自造词未解释（static AND frozen?）；定义在 [67] | 首次使用加 "(STSP held at baseline)"（L-F3） |
| R6d | [26] "retained support" / [42] "high-overlap contribution" | Results 用裸 "support"，正式名 "effective STSP support"（=u⊙x 逐元素积）在 Methods [63]/式(4) | [26] 引入 "effective STSP support（the u⊙x product）"（L-F4） |
| R6e | [21] "whereas loss captured the converse opportunities" | loss 与 opportunities 未定义；定义在 [94] | 扩展为 "whereas loss counted baseline-correct B trials made incorrect"（L-F5） |
| R6f | [31] K 未定义 | 首次出现 "at K = 1 and K = 5" 无解释；唯一注释在图 5 图注 [39]；Methods 又称 "history depths" | [31] 写 "at sequence lengths K = 1 and K = 5"（L-F6） |
| R6g | Fig.6 图注 [44] "keep probability 0.5" | 仅图注出现的参数，正文与 Methods 均未命名 | Methods [113] 或 Results 定义 "keep probability（cue 中保留的编码靶标 spike 比例）"（L-F7、F-F3） |
| R6h | [41] "no-memory" vs [113] "cue-only" | 同一参照条件两个名字，读者无法判断两阶段参照是否改变 | 统一为 "no-memory (cue-only) reference" 并定义一次（L-F8、F-F7） |
| R6i | [22] "common-update" / "shared component" / "shared updating" | 中心分解分量三种叫法（小节标题 [20] 为 "common input-driven"） | 统一 "common input-driven component/update"（L-F9） |
| R6j | [42] "area-and-energy-matched" vs [118] "area- and input-energy–matched" | 同一对照两种名字，且 [118] 复合形容词内用 en dash | 统一 "area- and energy-matched"（连字符）（L-F10） |
| R6k | [27] "donor-transfer index" | 0.8086 首次出现无解释；定义在 Methods [103] | 首次出现加一句指向定义或一句话定义（L-F13、F-F10 旁证） |

### 3.2 Methods 端点/参照定义缺口（正式线）

**R7（P2）"no-memory reference" 使用 8 次但从未操作化定义。** 最近似注释为 [21] "B errors observed without a preceding item"；no-memory 分支究竟由什么构成（空历史？零输入延迟？static-frozen？）未声明。它是 Fig.2 rescue/loss 与 Fig.6 对比的核心参照。补一句构造定义（F-F7）。

**R8（P2）"centered-cosine" 从未定义**（"centered" 指均值减除还是零中心化未声明）；Fig.5f 头条指标不可解析。首次使用处一句话定义（F-F8）。

**R9（P2）"Effective area"（Fig.5e）构造未说明**（阈上支持面积？支持值之和？单位？）。给出构造或指向 Source Data 图规格（F-F9）。

**R10（P2）"opportunity set(s)/denominators" 从未定义或量化**——rescue/loss 每网络哪些 trial 构成机会、数目多少缺失；行为端点的分母不可见。补构造定义并按网络报告 n（B3 端点表）（F-F10、L-F16）。

**R11（P2）"prespecified" 无指向**（9 处：85–95% 范围、0.5/0.05 阈值、校正集、机会分母、upper-20%、端点/窗/条件/cohort）。无协议/注册/分析计划引用。补指向（仓库/Source Data 中的分析计划）或弱化为 "set a priori in the simulation design" + 物证位置（F-F13）。

**R12（P2）训练基线分类准确率缺失。** Methods [86] 称 "measured before the working-memory assays to establish a task-capable substrate"，但全文无数值；"task-capable substrate" 无法核验。报告基线准确率及 CI（F-F15）。

**R13（P2）"the corresponding statistics tables" 悬空引用。** 文档 0 表（已核验 document.xml 无 w:tbl），投稿包 01_journal_upload 亦无统计表。补 B3 两张自足表（端点定义表 + 模型参数表），或将措辞改为 "statistics files in Source Data"（F-F16）。

**R14（P2）"BH" 在 Results 首次出现未展开**（[21/22] 用 "BH-adjusted"，全称在 Methods [122]）；Fig.6d 的 "P < 0.001" 未标 unadjusted 而 Methods 说 "endpoints labelled unadjusted were not included in a multiplicity family"，读者无法判断 6d 是否入族。Results 首处展开 "Benjamini–Hochberg"；6d 补标 unadjusted（F-F17）。

### 3.3 图注自足（D7）

**R15（P2）六条图注均无任何 P 值**（检验与校正名在，P 值全在正文）。按 D7 图注需含 P 值（或精确值区间）逐面板补入（F-F23）。

**R16（P2）Fig.6a 图注未声明推断对象是 normalized AUC 对比**（曲线仅描述；正文 [41] 报 AUC gain）。图注补 "inference on normalized AUC contrasts"（F-F26）。

### 3.4 形式与投稿包

**R17（P2）摘要主结论句为 "Here we used…"** 而非 CommsBio 风格 "Here we show…"（rubric D1；128 词 ≤150 合规）。末句改 "Here we show that iterative history-conditioned STSP updating links…"（F-F19、L-F22）。

**R18（P2）补充图 S1–S7 存在但正文零引用。** `RESULTS_EVIDENCE_BOUNDARIES.md` 指向 `supplementary_v5_c5_revised_20260804_r2`；PACKAGE_STATUS.json 亦报 supplementary_information: pending_reconciliation、submission_ready: false。二选一并落地：正文补 "Supplementary" 引用与 SI 节，或明示本投稿不含 SI（F-F22）。

**R19（P2）Reporting Summary（life-sciences）未提及。** CommsBio 送审要求；包状态 journal_policy_materials: pending。准备并随包提交（F-F21）。

**R20（P2）参考文献 "et al." 使用不一致。** 9 条截断为 "First author et al."（refs 5, 7, 11, 18, 19, 20, 21, 23, 26），其余多作者条目列全（10, 12, 25, 30, 33 等）；Nature Portfolio 风格列全作者（≤20）。统一规则（F-F27）。

**R21（P2）ref. 31 格式非期刊体例。** "eLife Reviewed Preprint 107005, version 2 (2026)" 既非 eLife 文章体例（eLife 15, RP107005 (2026)）也非 "Preprint at …"。`[INFERENCE]` 未对活 DOI 核验文号/版本；按期刊体例转换（F-F28）。

**R22（P2）Fig. 6 图注 keep-probability 档位缺失**（Fig.6a 只写 "across keep probability"，档位从未给出）。补档位（F-F3）。

---

## 4. 图件审计（视觉线 + 哈希核验）

### 4.1 图-文映射：正确（视觉线 P0 为误报）
- 视觉子代理报告 image8/image9 内容互换（P0）。**哈希核验否决**：6 幅内嵌图与冻结产物逐字节一致（sha256 前 12 位逐一匹配，见 §0.4），rel 映射正确。误报已剔除，不进入问题清单。
- 渲染核验：6 图分页位置正确（第 4/6/8/10/12/14 页）、图注同页、内嵌尺寸与像素纵横比一致（无拉伸变形）。

### 4.2 视觉契约合规（视觉线其余结论，建议作者成品尺寸人工过目确认）
- 面板字母、轴标签/单位、图例位置、语义颜色（≤3 色族/面板）、虚线参照（Fig.1b 85–95% 带、Fig.1d 10% 线、Fig.5d latest-item-only 参考）、热图色条结构（Fig.5e/f、Fig.6d）、灰度可辨、无裁切/碰撞/截断误差棒、无图内标题或样本量文字——视觉线逐项 PASS（42–62 项/图）。
- 注意：视觉线的 PASS 结论基于 @vision 模型读图，主审无法直接看渲染图；鉴于图件为项目自审通过产物（各图 qa/ 目录含 grayscale/wireframe 校验），残留风险低，但**提交前应人工在最终尺寸过目一次**。

### 4.3 图注-正文一致性（本报告主线核验）
- 数值交叉检查（正文数字 vs 图注条件）：Fig.1b 90.95% 与 "b–e descriptive" 一致；Fig.2c 阈值 0.5/0.05、Fig.4b–e 符号检验与单/双侧、Fig.5b 单条推断端点、Fig.6d 交互系数——全部与 Methods 统计节一致；
- 图注-正文同词检查：面板 a–f 描述与正文端点逐一对应，无改名/缺失面板；
- 唯一小瑕疵：正文 [31] 引用 "Fig. 4a-c"，但 a 是示意图、推断端点在 b,c——应改 "Fig. 4b,c"（P3，M1）。

---

## 5. P3 —— 可选（按主题合并）

**R23 排版一致性（语言线 F23 + 主线）：** 图范围连字符混用——正文 "Fig. 1b-d / 3b-d / 4a-c"（连字符）vs Methods "Fig. 1b–e / 4b–d"（en dash）；图 4 图注 "stages 2-10" vs Methods "stages 2–10"；CI 区间统一连字符但 [41] 出现一处 "−0.0739 to −0.0590"（"to"）。统一：面板/阶段/种子范围用 en dash，CI 区间全稿同一规则。

**R24 元数据与包卫生（正式线 F20/F29/F30 + 主线）：**
- 标题 "Working‑Memory" 含 U+2011（不换行连字符），标题段与 docProps 双重出现，会破坏检索/编辑管线 → 换普通连字符（F20）；
- 包内 image2–6.png（≈1.63 MB）为未被 document.xml 引用的孤儿媒体（旧版图），提交前清理（F29）；
- docProps dc:language = zh-CN（英文稿）→ 改 en/en-GB（F30）。

**R25 无障碍文本过期（主线 M2）：** 六图均有 alt text（好），但 Fig.3/5/6 的 alt 标题仍是旧版（"Overlap-aligned inherited STSP redirects…" / "Retained sequence states exhibit…" / "Retained STSP conditionally alters…"），与现图注标题不同步 → 更新 alt 标题。

**R26 措辞校准检查（主线 M3–M4）：**
- [42] "Removing the high-overlap contribution **caused** 2.52 percentage points more recruitment loss…" —— 匹配对照的定向移除支持 "produced/led to" 级表述；如 claim-proof matrix 允许 "caused" 则不动，否则收缩一级；
- [12] "online working-memory organization" —— "online" 全稿仅此一处，与 "continual organization" 术语不一致 → 改 "continual"。

**R27 语言微修簇（语言线 F24–F46 的 P3 项，逐条最小修复见附件/原文）：** [59] 冠词省略不一致、[60] 序列逗号不统一、[88] 列表边界歧义（"later inputs or cues and restored-state readout"）、[16] 时态混合（had fallen … and remained）、[111] "product u x" 应作 "u ⊙ x"、[115] 嵌套 when/while、[95] 花园路径句（见 R28 提升项）、[104]/[120] "within network" 缺冠词、[103] "STSP" 作可数名词（→ "STSP states"）、[114] "nonspecific" vs 连字符族、"complementary"×6 超载（[21][26][36][37][42]）、[10] "However" 同段两次、[32] "At the same time" 歧义、[7] "as" 子句歧义、[16] "Before…first" 冗余、[17] "distinct…differently" 冗余、[32] "Sufficiency at selected transitions" 省略主语、[49] "leaving whether…an empirical question" 宾语断裂、[48] "for temporal gradients and replay" 悬挂、[42] "six experiments"（→ "six analyses/assays"）、[114] "whose class was absent" 需补 "from the sequence"。

**R28（P2，语言线 F30 单列）Methods [95] 花园路径句**——exact-input 核心反事实定义句："Each history proceeded either unmanipulated or with the Layer 1 pooled event sequence produced by B from the no-memory reference replayed identically into both histories." 读者需两次解析。改为 "In each branch, the Layer 1 pooled event sequence produced by B from the no-memory reference was either left unmanipulated or replayed identically into both histories."（此条语义核心，建议按 P2 处理，已并入 §3 术语组旁注——列于此保持完整计数。）

---

## 6. 跨稿一致性审计（主线核验结果）

| 检查项 | 结果 |
|---|---|
| 引用覆盖 1–33 | ✓ 全部引用、全部有条目、无孤儿 |
| 公式编号 (1)–(22) 与正文引用 | ✓ 全部被引用、无孤式；OMML 渲染无豆腐块 |
| 图号交叉引用 Fig.1–6 | ✓ 按序引用、图注同页；小瑕疵 "Fig. 4a-c"（R23/M1） |
| 统计一致性（单/双侧、检验、校正、descriptive 标记） | ✓ Methods [120–124] 与六条图注逐项一致 |
| 术语稳定性（successor/inherited/STSP/passive evolution/regime） | ✓ 主词稳定；漂移项见 R6 表（fast state、common-update、singleton/single-item、no-memory/cue-only、opportunity 三变体、inherited condition/state） |
| 拼写体系 | ✓ Oxford 拼写一致（behaviour/analysed/-ize）；仅 "nonspecific" 例外 |
| 时态纪律 | ✓ 稳定主张用现在时、完成动作用过去时；仅 [16] 一处混合 |
| 修订痕迹/批注/超链接 | ✓ 无 ins/del/comment；URL 均为纯文本（期刊体例合规） |
| 页眉页脚 | ✓ 页脚为页码域；无页眉 |
| 摘要/标题长度 | ✓ 128 词 ≤150；标题 7 词 ≤15（仅 U+2011 与 "Here we used" 两项见 R17/R24） |

---

## 7. 明确判定为"不是问题"（防 scope creep）

- 六幅图为冻结验证产物、映射正确、比例无变形（哈希核验）；
- 结论与 Intro 问题的锁定关系未触及（本报告范围外）；
- n=20 网络推断单元、seeds 1000–1019、无功效计算/无正态性检验的披露、descriptive 面板枚举——均符合纪律；
- donor-transfer index 在 Methods [103] 有正式定义；α+ 与 αe、τ+ 与 τe 区分清晰；
- 图注均 ≤350 词（109–186 词）；全部含 n 与误差棒定义；
- 33 条参考文献全部带 DOI、页码齐全（除 preprint ref. 31）；
- "To test" 句式全文仅 3 处（每大节一次），无模板疲劳；
- 数据可用性声明完整；软件版本清单完整。

---

## 8. 优先修订路线（建议执行顺序）

1. **P1 批（1 天）**：R1 模型参数表 + R2 α + R3 精确 P 值（或 Source Data 指向）+ R4 confirmatory-19 + R5 代码提交号更新；同步把 R13 的两张 B3 自足表落地（参数表顺带满足 R1）。
2. **P2 术语与端点批（1 天）**：R6a–k 一次性加注与统一；R7–R11 Methods 端点定义（no-memory、centered-cosine、effective area、opportunity sets、prespecified 指向）；R12 基线准确率。
3. **P2 形式批（半天）**：R17 摘要 "Here we show"；R15 图注补 P 值；R18 SI 决策；R19 Reporting Summary；R20/R21 参考文献；R14 BH 展开；R22 keep-probability 档位。
4. **P3 扫尾（半天）**：R23 破折号；R24 元数据/孤儿媒体/U+2011；R25 alt 文本；R26/R27 措辞与微修簇；R28 花园路径句。
5. **复验**：改后重跑本方案 §0.4 核验（渲染、哈希、交叉引用），并按 rubric 做一次模拟审稿人复核。

---

## 附：来源与计数

- 语言线（L）：46 条（13×P2、33×P3），全文见 `language_reading_report.md`（会话 local 目录）；
- 正式线（F）：30 条（6×P1、15×P2、9×P3）；
- 图件线（V）：1 条（P0，哈希否决）；
- 主线（M）：独立发现并入 R23–R28（Fig.4a-c 引用、online 措辞、alt 过期、caused 校准、u/x 索引等）。
- 本报告最终计数：P1 = 6；P2 ≈ 22（含 R28）；P3 簇 ≈ 6 组（内含 30+ 单条微修）。
