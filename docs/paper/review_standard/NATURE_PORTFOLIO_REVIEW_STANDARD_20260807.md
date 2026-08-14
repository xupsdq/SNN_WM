# Nature Portfolio 审核标准与 AI 审稿能力综合判断

日期：2026-08-07
适用范围：Net_torch v5 稿件（目标期刊 Communications Biology）的送审前自审、修改优先级与"论文写得好不好"的判定方法。
上位证据：本文件由四路并行深度调研 + 本地能力盘点综合而成，每条标准可追溯到附录来源；未直接读到的资源一律不引用。

## 0. 调研结构与证据基础

| 路 | 内容 | 主要来源 |
|---|---|---|
| NaturePortfolioOfficial | Nature / Nature Neuroscience / Nature Communications / Communications Biology 官方标准（aims、editorial process、referee guide、formatting、统计与复现要求） | nature.com 官方页面（详见附录） |
| NatureEditorialLiterature | Nature 家族社论与审稿指南（摘要结构、统计显著性、可复现性、拒稿原因） | nature.com 社论/评论/官方文档 |
| DomainExemplars | 领域标杆论文（活动-静默工作记忆 / STSP / 网络计算）的写作与证据模式 | Stokes 2013 Neuron; Mongillo 2008 Science; Rose 2016 Science; Masse 2019 Nat Neurosci; Dunworth 2025 Nat Commun; Barri 2022 Nat Commun 等（全文或摘要级，均直接读取） |
| PublicAIReviewSystems | 2024–2026 公开 AI 审稿系统与基准（28 个一手来源，逐条核验） | arXiv / 官方政策页（详见附录） |
| 本地盘点 | OMP skill 注册表（21 个目录、16 个注册） + docs/paper 审稿轨迹（v5 四线审稿、证据冻结记录、权威契约） | `CORE_SCIENTIFIC_LOGIC_CONTRACT.md`、`RESULTS_EVIDENCE_BOUNDARIES.md`、`V5_COMPLETE_PRE_REVIEW_REPORT_20260802.md`、`V5_EVIDENCE_SUFFICIENCY_AND_WRITING_PLAN_20260805.md` |

重要勘误（防止把错误引用写进稿件）：Stokes et al. 2013 的正确出处是 Neuron 78:364–375《Dynamic Coding for Cognitive Control in Prefrontal Cortex》（不是 Nature Neuroscience）；Rose et al. 2016 的正确出处是 Science 354:1136–1139《Reactivation of Latent Working Memories with TMS》（不是 Neuron）。"Activity-silent working memory" 作为框架标题出自 Stokes 2015 Trends Cogn Sci 综述。

---

## 1. 现状盘点与综合判断（AI 审稿能力）

### 1.1 本地可复用能力（OMP Skills）

| Skill | 能力 | 在审稿流程中的角色 | 缺口 |
|---|---|---|---|
| review-agent | 只读、缺陷优先的**代码**审查，P0–P3 分级，明示"No findings" | 可借鉴其纪律（不虚构发现、逐条可定位、作者视角），但对象是代码 | 不能直接用于稿件 |
| write-paper-results | Results 论证审计：question→inference 单元、Action→Data→Inference、claim calibration、audit 模式 | Results 线审查的核心工具 | 只覆盖 Results |
| manuscript-sentence-revision | 结构锁定的句段润色，claim-proof matrix 权威、证据天花板 | 表述优化阶段（Abstract/Intro/Discussion/图注） | 明确不做结构调整 |
| paper-figure | 图件 QA：证据保真、audit 脚本、硬规则（伪重复、smoke 当终稿、不可比协议并列） | 图件线审查 | 未接入审稿编排（独立运行） |
| pdf | 版式渲染 QA（Poppler 渲染检查） | 提交前格式 | 非科学审查 |
| deep-requirement-research | 需求研究：证据路由、分支收敛、防候选者先验 | 可复用于"期刊到底要求什么"的调研 | 非审稿本身 |
| formal-proof-planner | 形式化证明规划（有界假设、引理分解） | 数学断言审计（本稿当前非必要） | — |

**结构缺口：注册表中没有"稿件审稿"skill。** 现有 21 个 skill 覆盖"写"（results/prose/figure）与"格式"（pdf），唯独没有"审"。上一次审稿靠一次性编排完成，未固化为可复用能力。

### 1.2 仓库内已有审稿轨迹（v5，2026-08-02 完成）

- 四路并行审稿 + 元审稿：FieldNoveltyReview / MechanismMethodsReview / StatisticsEvidenceReview / NarrativeFigureReproReview，产物为 `V5_COMPLETE_PRE_REVIEW_REPORT_20260802.md`，已解决/撤回条目归档于 `archive/v5_ai_review_20260802/`。
- 流程质量判断：**明显高于公开系统的平均水准**。它具备公开系统普遍缺失的要素——每条意见带"稿件定位→影响→反证检查→最小动作"、区分必须修正/可选增强、明示"不是问题"清单（防 scope creep）、拒绝"多做数据集/架构/湿实验"的默认答案、不给 accept/reject 结论。
- 证据权威链：`CORE_SCIENTIFIC_LOGIC_CONTRACT.md`（要证明什么）→ `RESULTS_EVIDENCE_BOUNDARIES.md`（证据边界）→ 面板合同/指标 → Source Data，审稿意见可沿此链核验。
- 缺口：未绑定期刊 rubric（当时以散文形式引用三家期刊标准，未成检查表）；无防幻觉的证据核验层；图件审计未并入审稿编排；无与专家审稿的校准记录。

### 1.3 公开 AI 审稿系统全景（2024–2026，28 个一手来源核验）

代表性新增条目（详见附录）：ReviewCritique、AAAR-1.0（PaperWeakness/REVIEWCRITIQUE）、SPOT（真实错误发现基准，最强模型 recall ≤21.1%）、Kahneman4Review、REVIEWBENCH + ReviewGrounder、RubricReviewer、DIAGPaper、DEFEND（作者在环反驳）、AgentReview、STRICTA（生物医学结构化评估）、Evidence-RAG（期刊编辑部工作台）、ReviewGuard（引用-影响力对齐）、RefChecker/Phantom References（约 1/20 NeurIPS/USENIX Security 2025 论文含 ≥2 条疑似幽灵引用）、SciScore（生命科学 rigor 筛查）、NeurIPS 2025 官方 LLM 政策（审稿人不得用 LLM 起草审稿意见）、跨厂商对齐研究（GPT-5.4/Gemini/Claude 与人类审稿对比）与 111 个会议/期刊的 AI 政策调查（含 Nature Communications 评估）。

**收敛出的审稿维度**（与 v5 四线几乎一一对应）：新颖性/原创性、重要性/影响、技术可靠性/严谨性、实验完整性、清晰度/呈现、证据与引用锚定。

**公开系统公认的失败模式**（本标准必须防御的六件事）：
1. 幻觉证据与引用（RefChecker；SPOT 显示 LLM 只能发现 ≤21.1% 的真实稿件错误）；
2. 过度宽容与校准漂移（"overly positive recommendations"、Gemini 系统性高分、严格 rubric 反而损害判断）；
3. 泛泛而谈、methods 中心的批评（过度标注缺 baseline，漏掉效率/重要性）；
4. 漏判新颖性与长期影响（人类与实现影响的相关系数也只有 ≈0.49）；
5. 多模态/图件理解弱（SciAssess 动机）；
6. 可操纵性（prompt injection、数据投毒、reward hacking）。

### 1.4 综合判断（结论先行）

- 现状定位：**高质量的一次性审稿编排 + 强有力的证据权威链**，而非"可重复的审稿系统"。
- 对"判断论文写得好不好"已具备：证据充分性判定（八角度）、统计推断纪律、表述纪律（claim-proof matrix）——这些不缺。
- 最缺三件事：(1) 期刊校准的评分 rubric（本文件第 2 节补上）；(2) 防幻觉的证据核验环节（引用解析、稿件段落锚定）；(3) 把审稿流程固化为注册 skill（第 4 节建议）。

---

## 2. Nature 审核标准（统一 Rubric）

### 2.1 期刊标尺（先定标尺再自审）

Communications Biology 官方标准（nature.com/commsbio/aims、journal-information、submit/editorial-process、referees/guide-to-referees）：

- 录用条件：**新颖（novel）、为结论提供强证据（strong evidence）、数据技术可靠（technically sound）、对特定生物学子领域重要**；
- 显著性标尺明确低于 Nature 品牌期刊（含 Nature Communications）："applying less stringent criteria for impact and significance than the Nature-branded journals"，但**技术有效性与伦理标准与整个 Nature Portfolio 相同**；
- 编辑评估点：novelty and potential impact、scope fit、conceptual or methodological advances、对读者的兴趣；拒稿理由：lack of novelty、insufficient conceptual advance、major technical and/or interpretational problems；
- 审稿人必答问题（CommsBio Guide to Referees）：主张是否新颖（"identify the major papers that compromise novelty"）；是否令人信服（"what further evidence is needed?"）；是否会"influence thinking in the field"；方法细节是否足以复现（建模研究明确要求 source code、protocols、mathematical derivations）；"Is the statistical analysis of the data sound?"。

推论：本稿按 CommsBio 标尺自审；Nature Communications 作为上探参照；不要用 Nature main 的 "outstanding scientific importance / interdisciplinary readership" 自我加压，也不要因"影响不够全球性"自我否证。

### 2.2 八维 Rubric（D1–D8）

每条检查点均带判定问题；严重度分级见 2.3。

**D1 问题与贡献（Problem & Advance）**
- 是否采用"两面对立"框架：命名领域默认假设（持续活动维持记忆 / 单一动力学状态）并与之对照，而非纯展示（领域标杆模式：Masse 2019 维护 vs 操纵；Rose 2016 持续活动 vs 突触痕迹；Dunworth 2025 平衡 vs 抑制-稳定）。
- 直接先例是否逐条对照（v5 B1：Ballintyn et al. 2019、Aitken & Mihalas 2023、多项目 STSP/序列位置工作），gap 是否落成一句可辩护的话："A controlled decomposition and causal tracing of an activity-silent STSP successor transition remain lacking."
- successor state 首次出现是否有动力系统定义（$S_{t+1}$），避免与强化学习 successor representation 混淆（v5 B1）。
- organization 是否有操作定义（v5 B4："the multi-component, history-specific structure of the inherited synaptic state and the input conditions under which that structure remains accessible"），且不与 executive manipulation / chunk formation / serial-order recall 混淆。
- 摘要：CommsBio ≤150 词、不带引用、以 "Here we show" 引出主结论（风格指南要求）；标题 ≤15 词（NComms 惯例，CommsBio 相近）。

**D2 新颖性与文献定位（Novelty & Prior Art）**
- 判"不新颖"必须能给引用（Nature Neuroscience 规则："support this opinion with a citation from the literature"）；反过来，稿件自己的新颖性声明也必须带引用支撑。
- 先例对照表是否完成（持续活动？前馈？固定后续输入？分解共同项/残差？跨层写回追踪？后续访问检验？）——v5 B1 的六列对比表。
- 稿件独有的增量是否明示：activity-silent spiking u/x state、identical-input 反事实、common update + history residual 分解、overlap localization、Layer 1→Layer 2 write-back 因果追踪、successor-state boundary、conditional re-entry/access。

**D3 主张—证据匹配（Claims–Evidence Match）**
- 每条主张能否锚定到具体图/表/统计；因果语言仅在设计能识别因果贡献时使用（当前 Fig.6f 只允许 stratified STSP-by-overlap interaction，不允许 factorial manipulation/causal gate——v5 A3 剩余项）。
- "activity-silent" 类主张本质上是从"未检出活动"做的推断，必须有阳性探针补足（领域模式：Rose 2016 TMS 脉冲复现潜在痕迹、Stokes 2013 固定中性刺激探针、Masse 2019 直接读出突触效力）；纯 decoder 相关性 ≠ 机制。
- 结论与证据比例一致："an advance in understanding likely to influence thinking in the field"（Portfolio 政策），且明确区分模型内机制充分性 vs 一般认知功能。

**D4 统计与不确定度（Statistics & Uncertainty）**
- CommsBio 强制：Methods 需有 "Statistics and Reproducibility" 小节，写明检验名、每项分析的 n、比较对象、检验选择理由（如正态性）、alpha、单/双侧、**每个检验的实际 P 值**（不允许只写 "significant" 或 "P < 0.05"）；描述统计给 n、中心值、变异性；明确 s.d. vs s.e.m.；图需显示完整数据分布（散点叠加或箱/点图）；处理多重比较、正态性、小样本（n<10）。
- 统计只用于独立数据（Vaux 2012：单个代表性实验 N=1 不适用统计；技术重复 ≠ 独立实验）；图注需写明独立数据点数。
- P 值不是唯一仲裁：结合效应量、CI、设计、先验证据；同一数据用多种分析看是否收敛（Nature 2019 显著性社论）。
- 交互项主张不得靠"一个显著一个不显著"（Nieuwenhuis）；无 double dipping（Kriegeskorte）；多重性控制与预设 family。
- 本稿现状核对：推断单元 n=20 网络、BH/Holm、exact-input 反事实、confirmatory-19、Fig.4 精确符号翻转——均已正确，但需在 Methods 的 Statistics 节完整呈现（当前表述阶段尚未确认该节存在）。

**D5 方法与复现（Methods & Reproducibility）**
- Methods 完整到"interpretation and replication"（Formatting Guide）；禁 "data not shown"（CommsBio 风格指南明确禁止）。
- 建模研究必须给出 source code、参数、数学推导（CommsBio referee guide 明示）。
- 两张自足表（v5 B3）：(1) Primary endpoint definitions——Figure/panel、reference condition、eligible set、numerator/denominator、window、exclusion/undefined rule、within-network aggregation、cross-network estimand、null/SESOI、multiplicity family（rescue ≈4–9 eligible anchors/网络、loss ≈41–46 等机会集差异必须写清）；(2) Model and protocol parameters——层/feature-map、kernel、stride、Δt、时间常数、threshold/reset/refractory、STSP（U, τ_fac, τ_rec）、inhibition/top-k、baseline gain compensation、窗口、MNIST 划分、trial 数与采样规则、每图对应 config/protocol ID。
- 代码快照身份（v5 A7）：Code availability 指向 `93d9b8295fdfbd603d3f181a3500773cb4689a75`（SOURCE_IDENTITY.json 记录的 final-six 快照），而非旧 commit `ef5eabee…`；提供六图 plot-only replay 单一入口。

**D6 数据与代码可得性（Data & Code Availability）**
- 强制 Data availability 与 Code availability 声明（每篇原始研究）；CommsBio 对核心结论依赖的自定义代码：**不可用可导致拒稿**（"reserve the right to decline the manuscript if important code is unavailable"）。
- Reporting Summary（life-sciences 版）送审即要求，接收后随文发布。
- 每图 Source Data 可追溯（本稿已有 `02_data_release/main_figure_source_data/FigX` 结构，需与正文端点一致）。

**D7 图件与叙事清晰度（Figures & Narrative Clarity）**
- 图注必须给：中心值（median/mean）、所有误差棒定义及计算方式、样本量 n、所用检验、P 值（Nat Neurosci/CommsBio 规则）；补充图同标准；图注 ≤350 词。
- 语义同步检查（v5 B5 四处）：Fig.1b 逐网络点+seed 连线（图注不得写 mean/CI 线）；Fig.2c/Fig.4c 是减阈值后的 margin（0 线=阈值，柱高不可跨端点解释）；Fig.3c transition composition 未含 unchanged（改为 events (%) 并注明补集省略）；Fig.6a 只画 curves（说明推断用文中报告的 AUC contrasts）。
- 图件多模态审计必须独立成线（SciAssess 教训；本地 paper-figure 的 audit 脚本可承担）。

**D8 伦理、范围与流程合规（Ethics, Scope, Compliance）**
- 本稿为 MNIST 模型研究，无人体/动物/敏感数据，不需要机械增设伦理批判线（沿 v5 纪律）。
- 利益冲突声明；AI 使用声明（Nature Portfolio 政策要求披露；自查场景用本地工具，不上传稿件到第三方服务）。
- 范围自洽：固定前馈架构、顺序 MNIST、模型内机制是主动声明的边界，不是缺陷；不主张 STSP 是唯一机制、完美回忆或脑内普遍机制。

### 2.3 判定方案（严重度与输出格式）

- 分级（沿 review-agent 的 P0–P3 语义）：**P0** 阻断——中心主张不成立或证据链断裂（需重跑/重设计）；**P1** 必须修——当前版本不能冻结（措辞超过证据、复现身份错误）；**P2** 应修——影响审稿人理解或可信度；**P3** 可选——增强项。
- 每条意见五件套：稿件定位（文件/节/图注）→ 影响 → 反证检查（什么证据已排除该问题）→ 最小动作 → 是否需要新模拟/数据（默认不需要，除非措辞无法收缩）。
- 必须附"明确判定为不是问题"清单（防 scope creep，沿用 v5 报告惯例）。
- **AI 只输出问题与证据，不输出 accept/reject**（沿 v5 报告纪律；NeurIPS 2025 政策也禁止 LLM 起草审稿结论，自查同理）。

---

## 3. 如何判定本稿"写得好不好"（工作流）

### 3.1 三层判定

1. **科学充分性层**（已冻结，2026-08-05）：八角度充分性成立（论证链完整性、因果识别、替代解释关闭、统计充分性、分析选择稳健性、结论边界校准、领域标准对比、审稿压力测试）；F1→F4 链每环有主图直接证据；S1–S7 逐图关闭替代解释。此层结论：**证据足够，无需新模拟/新网络/湿实验**。
2. **期刊符合层**（本文件第 2 节）：对 D1–D8 逐维核对，输出状态表（通过/需修/未验证），据此产生 P0–P3 清单。
3. **表述质量层**：claim-proof matrix（已批准措辞上限）→ manuscript-sentence-revision（结构锁定润色）→ write-paper-results audit（论证顺序/追踪性）→ paper-figure QA（图件）→ pdf 渲染检查（格式）。

### 3.2 v5 当前状态映射（截至 2026-08-07，基于审稿轨迹与权威契约，非全文重读）

- 已关闭：A1（Fig.5f）、A4（Fig.3c）、A5（Fig.2d）、A6（撤回，seeds 1000–1019 属完整 cohort）；A3 的端点与估计；证据冻结；主图 Fig.1–6 与 Fig.S1–S7 成图（39 文件）。
- 未关闭（均为 P1/P2，无 P0）：
  - **A2**（P1）：Fig.4 的 K=2–10 全阶段"同一基元"措辞超证据 → 收缩为 repeated displacement + K5 motif（Abstract/Fig.4 标题/Results 末段/Discussion 同步）；
  - **A3 剩余**（P1）：factorial/causal 措辞 → stratified STSP-by-overlap interaction（Results/Fig.6 图注/Discussion）；
  - **A7**（P1）：代码快照 → `93d9b829…` + plot-only replay 入口；
  - **B1**（P2）：直接先例对照表 + 独有增量清单；
  - **B2**（P2）：dynamic/static 改为 regime contrast，inherited-state 隔离改由 reset/attenuation/donor swap 承担；
  - **B3**（P2）：两张 Methods 自足表；
  - **B4**（P2）：organization 操作定义 + 带可测代理量与反驳条件的实验预测表；
  - **B5**（P2）：四处图注/轴名语义同步。
- 与 rubric 对应：A2/A3→D3；B1/B2/B4→D1/D2；A7/B3→D5；B5→D7；另需新增核对 D4（Statistics 节）、D6（Reporting Summary、Data/Code 声明）、D1（摘要 ≤150 词 "Here we show"、标题 ≤15 词）。

### 3.3 初步判定（口径：CommsBio 标尺）

证据层达到或超过领域惯例（exact-input 反事实、状态移植、干预身份审计、确认性 cohort、系数无关端点、事件级机会对照），与标杆论文模式一致（反事实扰动、替代机制显式关闭、统计在正确推断单元上、限制前瞻性声明）。剩余问题集中在**定位与表述**（D1/D2/D3）和**自足复现**（D5/D6/D7），无 P0。结论：**具备送审条件，但应先在 P1 关闭后再组包**；按 3.4 顺序执行。

### 3.4 执行顺序（推荐）

1. 关闭 P1：A2/A3 措辞同步（Abstract、Results、Fig.4 标题、Fig.6 图注、Discussion）；
2. 领域定位：B1（先例表）、B4（organization 定义与预测表）；
3. 自足复现：A7（代码 commit）、B3（两张表）、B5（图注）；
4. 格式合规：摘要 ≤150 词 + "Here we show"、标题、Methods Statistics & Reproducibility 节、Reporting Summary、Data/Code availability 声明、Source Data manifest；
5. 组包（上传物：期刊格式 PDF、Source Data 归档、图件、cover letter）；
6. 用本 rubric 做一次全维度"模拟审稿人"复核，保留人类 gate。

---

## 4. 后续建议（把能力固化）

1. 用 skill-creator 把"Nature Portfolio 稿件审稿"固化为注册 skill：四线编排（FieldNovelty / MechanismMethods / StatisticsEvidence / NarrativeFigureRepro）+ 元审稿去重 + 本 rubric + 五件套输出 + 明示非问题清单；这是当前注册表里最明显的空缺。
2. 审稿编排内接入 paper-figure 的 audit 脚本（图件线自动审计）与 pdf 渲染检查（格式线）。
3. 增加证据核验环节：引用解析（RefChecker 式）、每条意见的稿件段落锚定；SPOT 式校准探针（用已知错误的稿件测试审稿召回）。
4. 每次审稿登记 calibration 记录（与专家审稿/录用结果对比），逐步量化 AI 审稿的查全与查准。

---

## 附录：来源清单（全部直接读取核验）

官方标准（NaturePortfolioOfficial）：
- nature.com/commsbio/aims；/commsbio/journal-information；/commsbio/submit/editorial-process；/commsbio/referees/guide-to-referees；/commsbio/submit/submission-guidelines；/documents/commsj-life-style-formatting-guide-accept.pdf
- nature.com/nature/for-authors/editorial-criteria-and-processes；/nature/for-authors/formatting-guide；/nature/journal-information
- nature.com/neuro/aims；/neuro/submission-guidelines/editorial-process；/neuro/submission-guidelines/aip-and-formatting；/neuro/content
- nature.com/ncomms/aims；/ncomms/submit/guide-to-authors；/ncomms/submit/editorial-process；/documents/ncomms-manuscript-checklist.pdf
- nature.com/nature-portfolio/editorial-policies/peer-review；/nature-portfolio/editorial-policies/reporting-standards；/collections/qghhqm（Statistics for Biologists）

编辑社论与指南（NatureEditorialLiterature）：
- nature.com/documents/nature-summary-paragraph.pdf（摘要段落逐句注释样例）
- nature.com/articles/nn0109-1（Nat Neurosci, Striving for excellence in peer review）
- nature.com/articles/473253a（There's a time to be critical）
- nature.com/articles/d41586-019-00874-8（It's time to talk about ditching statistical significance）；d41586-019-00857-9（Scientists rise up）
- nature.com/articles/492180a（Vaux, Know when your numbers are significant）
- nature.com/articles/nmeth.4120（Points of Significance: P values and the search for significance）
- nature.com/articles/506150a（Statistical errors）；nrn3475（Power failure）；nn.2303（double dipping）；nn.2886（interactions）
- nature.com/articles/483509a（Must try harder：粗心错误清单）；496398a（Reducing our irreproducibility）；533437a（Reality check on reproducibility）
- nature.com/articles/d41586-018-02404-4（How to write a first-class paper，摘要级）

领域标杆（DomainExemplars）：
- Stokes et al. 2013, Neuron 78:364–375, doi:10.1016/j.neuron.2013.01.039（PMC3898895 全文）
- Mongillo et al. 2008, Science 319:1543–1546, doi:10.1126/science.1150769（摘要级，付费墙）
- Rose et al. 2016, Science 354:1136–1139, doi:10.1126/science.aah7011（PMC5221753 全文）
- Masse et al. 2019, Nat Neurosci 22:1159–1167, doi:10.1038/s41593-019-0414-3（PMC7321806 全文）
- Dunworth et al. 2025, Nat Commun 16:8657, doi:10.1038/s41467-025-63818-z（PMC12484685 全文）
- Barri et al. 2022, Nat Commun 13:7902, doi:10.1038/s41467-022-35395-y（摘要级）
- Pals et al. 2020, PLoS Comput Biol 16:e1007936（记录级）；Stokes 2015, TiCS 19:394–405（PMC4509720）

公开 AI 审稿系统（PublicAIReviewSystems，28 条，节选）：
- arxiv.org/abs/2406.16253（ReviewCritique）；2410.22394（AAAR-1.0）；2505.11855（SPOT）；2607.10511（Kahneman4Review）；2604.14261（REVIEWBENCH）；2608.06312（GB/T-Bench）；2608.00005（RubricReviewer）；2601.07611（DIAGPaper）；2603.27360（DEFEND）；2406.12708（AgentReview）；2409.05367（STRICTA）；2606.25837（Evidence-RAG）；2606.24892（ReviewGuard）；2607.00738（RefChecker/Phantom References）；2608.03659（跨厂商对齐研究）；2608.03581（111 会议/期刊 AI 政策调查，含 Nature Communications 评估）；2403.01976（SciAssess）；2504.09737（ICLR 2025 Review Feedback Agent RCT）
- neurips.cc/Conferences/2025/LLM（NeurIPS 2025 LLM 政策）；sciscore.com（SciScore）

本地：`docs/paper/CORE_SCIENTIFIC_LOGIC_CONTRACT.md`；`docs/paper/RESULTS_EVIDENCE_BOUNDARIES.md`；`docs/paper/revisions/v5_ai_review_20260802/V5_COMPLETE_PRE_REVIEW_REPORT_20260802.md`；`docs/paper/revisions/v5_ai_review_20260802/V5_EVIDENCE_SUFFICIENCY_AND_WRITING_PLAN_20260805.md`；`docs/archive/paper/submission-packages/communications_biology_20260802_v5_submission_candidate/README.md`。
