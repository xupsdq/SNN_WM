# V5 全文审读后修订执行方案

日期：2026-08-08  
目标稿：`docs/archive/paper/intermediate-drafts/v5_discussion_revised_20260808.docx`
目标期刊：Communications Biology  
上游审稿报告：`docs/paper/revisions/V5_READER_FULL_REVIEW_20260808.md`

## 0. 方案定位

本方案把上一轮“读者视角全文审稿”转化为可逐项执行、逐项验收的修订流程。它不重新讨论 Introduction 提出的问题是否正确，也不重新判断 Results 的科学结论、实验设计或证据充分性。

### 0.1 不可变项

1. 保持 Introduction 的研究问题、六段 Results 的结论顺序、Discussion 的三条预测及其证据边界不变。
2. 不增加实验、不重新运行模拟、不改动数值结果、不替换当前六张主图的科学内容。
3. 不把审稿意见扩展成新机制、新因果层级或新生物学外推。
4. 不从 `archive/` 提取当前参数、统计量或图稿；历史文档只能提供写作参考，不能覆盖当前 final-six 证据链。
5. 所有改写都遵守最小修改原则：修复自足性、术语、句群阅读、图文联动、统计报告、元数据与投稿包完整性。

### 0.2 目标产物

1. 新建修订稿，不覆盖原稿：建议命名 `docs/paper/v5_submission_ready_20260808.docx`。
2. 两张可编辑、自足的补充方法学表（不作为主文 Table 1/2）：
   - Supplementary Table S1：Model, encoding and training parameters；
   - Supplementary Table S2：Endpoint definitions and statistical analysis plan。
3. 一份由当前统计文件自动汇总的内部统计核对表：`manuscript_statistics_table.csv`。它用于防止正文、图注与 Source Data 抄录错误，可随 Source Data 提供，但不作为主文展示表。
4. 一份修改记录：每项修改对应“原位置—修改动作—权威来源—验收结果”。
5. 更新后的投稿包状态、Source Data、Supplementary Information、Code availability 与 Reporting Summary。

### 0.3 内容放置规则

修订项按“读者是否必须在阅读主结论时立即知道”分层，禁止把审计细节全部灌入正文。

| 放置层级 | 只放这一层的内容 | 主文处理 |
|---|---|---|
| Supplementary Information | 完整模型/编码/训练参数清单；全部 endpoint 的 eligible set、分子/分母、窗口、排除规则和 multiplicity family；untouched-19 的具体数值；S1–S7 的 robustness、sensitivity、coverage、estimability 与 secondary control 结果；50 shuffled pairs、nine deranged composites、winner cap、distance threshold 等构造细节 | 不进入主文，也不设置 Supplementary Table/Fig. 指针 |
| Source Data | 全部 network-level values；所有 raw/adjusted P；每个 anchor/trial/site 的计数；完整 correction-family 成员；bootstrap/resampling 输出；excluded/undefined rows | 主文只报告直接支撑主句的 effect、CI、n 和 primary P |
| Internal QA only | `manuscript_statistics_table.csv` 的 source path/hash/status 列；主图哈希核对、pixel/extent、grayscale/wireframe QA；DOCX orphan media、语言元数据和关系检查；`PACKAGE_STATUS.json` 与 open-item ledger | 不进入正文或 Supplementary prose |
| Separate submission/repository material | Reporting Summary；代码文件清单、runner inventory、environment lock、release manifest | 主文 Data/Code availability 只给稳定链接、release/DOI/SHA 与覆盖范围 |

主文只保留：核心三层 architecture；STSP 的科学参数与训练/测试启停方式；主要事件时间；核心 metric 和 control 的一句定义；独立重复单位、α、检验方向与校正原则；直接支撑六段 Results 主句的 effect/CI/P。上述内容不能全部下放，否则正文将不再自足。

## 1. 权威来源与冲突裁决

按以下顺序取证；低优先级来源不得覆盖高优先级来源。

| 优先级 | 权威来源 | 用途 |
|---|---|---|
| 1 | `docs/paper/CORE_SCIENTIFIC_LOGIC_CONTRACT.md` | 固定科学问题、核心机制边界与禁止扩写项 |
| 2 | `docs/paper/RESULTS_EVIDENCE_BOUNDARIES.md` | 固定 Results 可写与不可写的证据边界 |
| 3 | `results/paper_figure_multi_seed/final_six_figures_v5_c5_revised_20260804_r2/` | 六张主图当前统计量、图稿、manifest 与 plot spec |
| 4 | `docs/paper/submission_packages/communications_biology_20260801_final_six_results_candidate/04_internal_qa/story_and_contracts/FIGURE_EVIDENCE_INDEX.md` | panel→数据→统计→独立重复单位映射 |
| 5 | 当前模型、编码与训练实现：`src/core/network.py`、`src/data/encoding.py`、`src/training/train_sdnn.py`、`src/training/train_sdnn_ensemble.py` | 参数、结构、训练与读出规则 |
| 6 | `results/multi_snn/sdnn_ensemble_20/sdnn_ensemble_20/ensemble_run_config.json` 及各 seed `run_config.json` | 实际训练配置，而非代码默认值 |
| 7 | `docs/paper/submission_packages/communications_biology_20260801_final_six_results_candidate/03_code_release/SOURCE_IDENTITY.json` | 代码快照身份 |
| 8 | `results/paper_figure_multi_seed/supplementary_v5_c5_revised_20260804_r2/` 与 `docs/paper/SUPPLEMENTARY_FIGURE_ARGUMENT_AUDIT_V5.md` | Supplementary Figs. S1–S7 及其论证角色 |
| 9 | `docs/paper/review_standard/NATURE_PORTFOLIO_REVIEW_STANDARD_20260807.md` | 期刊格式、统计、自足性与投稿完整性标准 |
| 10 | `docs/paper/revisions/V5_METHODS_REVISION_PLAN_20260808.md`、`V5_METHODS_TERMINOLOGY_ALIGNMENT_20260808.md` | 已形成的方法学修订与术语约束 |

### 1.1 已核定且不得重复争论的事实

- 当前 DOCX 内六张主图与 `final_six_figures_v5_c5_revised_20260804_r2` 的 `fig1.png`–`fig6.png` 逐图哈希一致；不交换 Fig. 3/4，不重画主图。
- DOCX 中 22 个编号公式均存在且 PDF 渲染可读；修订只补定义和参数，不重写科学公式。
- 参考文献 1–33 均在正文被引用，均有 DOI；当前“作者不超过五人全部列出、六人及以上用 et al.”的模式不作为问题处理。
- 当前训练配置为 20 个网络、seed 1000–1019、batch size 512、Layer 1/2/3 分别训练 2/10/100 epochs，训练时 STSP disabled。20 个网络最终数字分类准确率均值为 91.158%，SD 为 0.343 个百分点，95% t CI 为 90.998–91.318%，范围为 90.76–91.90%。
- 当前代码释放身份文件记录的快照为 `93d9b8295fdfbd603d3f181a3500773cb4689a75`；稿件仍写旧 SHA `ef5eabee7594a3b59f44e9c9b6b940144143fd4b`。
- Supplementary Fig. S2d 已给出排除 development seed 1000 的 untouched-19 sensitivity：Layer 2 update donor-transfer index 0.80865（95% CI 0.80754–0.80979，Holm-adjusted P = 1.53 × 10^-5）；early Layer 3 endpoint 0.51903（95% CI 0.46201–0.58428，Holm-adjusted P = 1.53 × 10^-5）。

## 2. 修订门槛与优先级

### 2.1 P1：投稿前必须闭合

| 编号 | 必须修改项 | 当前风险 | 关闭条件 |
|---|---|---|---|
| P1-1 | 补全模型、编码、训练、读出和协议数值 | Methods 只有公式和概述，无法独立复现 | 主文 Methods 保留解释结论所需的核心参数；Supplementary Table S1 提供可复现的完整参数 |
| P1-2 | 建立 endpoint/statistics 自足记录 | eligible set、分母、窗口、null、family 分散在图注/代码/Source Data | 主文 Methods 定义核心 endpoint 与统计原则；Supplementary Table S2 覆盖全部 inferential endpoint |
| P1-3 | 明示显著性水平并报告实际 P 值 | 全文 16 处使用 `P < 0.001`，未声明 α | 作者确认 α；Methods 写明；正文与图注由统计主表替换为实际 P |
| P1-4 | 交付 confirmatory-19 sensitivity | 当前投稿包未把排除 development seed 的敏感性显式交付 | 只在 Supplementary Fig. S2d legend/Supplementary Table S2 写出 n = 19 两个 endpoint；主文不提及 |
| P1-5 | 修复 Code availability 快照身份 | 稿件 SHA 与 final-six 释放身份不一致 | 公共仓库可访问目标 SHA/标签/DOI 后，稿件、release manifest、Reporting Summary 三处一致 |
| P1-6 | 量化实际分类、输入与决策规则 | baseline accuracy、MNIST split、编码与 Layer 3 决策 gate 缺失 | 主文 Methods 报告 ensemble accuracy、split、核心输入时长与离散 decision gate；完整 DoG/latency 参数进入 Supplementary Table S1 |

### 2.2 P2：应在同一轮关闭

1. u/x、fast state、static-frozen、effective STSP support、loss、K、keep probability 等首次出现无定义。
2. common/shared/common-update、fast variables/fast state、inherited condition/state 等术语漂移。
3. 主图图注缺少 primary contrasts 的实际 P 值；“prespecified”没有可定位的历史 protocol 引用。
4. Supplementary Information 投稿包仍为 pending，S1–S7 尚未形成独立、可提交的完整文件。
5. Fig. 3/5/6 的 alt-text 标题仍是旧版本标题。
6. DOCX 语言元数据为 `zh-CN`，并包含 5 个未被 document relationships 引用的旧 PNG。
7. `PACKAGE_STATUS.json` 仍为 `submission_ready: false`，Reporting Summary 仍缺。

### 2.3 P3：完成 P1/P2 后统一清扫

包括 dash、serial comma、标题 U+2011、少量句法花园路径、重复连接词、Results 尾句重复、图号范围、caption 术语、Word 元数据。P3 不得先于 P1/P2 驱动改写。

## 3. 全文固定术语表

修订前先冻结以下稿件用语；之后只允许这些主名称。数学符号可在同一句括注，不能取代科学名称。

| 科学对象 | 主名称 | 首次定义要求 | 禁止/待消除变体 |
|---|---|---|---|
| 输入前携带的突触状态 | inherited STSP state | “the STSP state present before an input” | inherited condition（作为名词） |
| 输入处理后写入的状态 | successor state | “the downstream STSP state written by that input” | 未定义的 successor |
| STSP 状态变量 | joint u/x state | “utilization u and available-resource x” | 只写 u/x 而无首用解释 |
| 有效突触支持 | effective STSP support, u ⊙ x | 明示 elementwise product | bare support；`u x` |
| 输入共有部分 | common input-driven component | 首用后全文一致 | common-update direction、shared component、shared updating 混用 |
| 历史差异部分 | history-conditioned residual | 首用后全文一致 | history-sensitive component（未说明同义） |
| 非突触快速状态 | fast state | membrane potential, excitatory conductance, refractory state, lateral inhibition | fast variables、fast-state variables 混用 |
| 正常动态条件 | dynamic STSP condition | 首用可写 dynamic (intact) STSP | intact dynamics/intact STSP/dynamic state 任意切换 |
| 冻结对照 | static-frozen control | “STSP held at baseline; input scaled by fixed baseline support” | static-frozen baseline/update opportunity 无解释 |
| 行为改善 | rescue | baseline-incorrect trial corrected after history | 仅写 corrected errors |
| 行为损失 | loss | baseline-correct trial made incorrect after history | converse opportunities |
| 分母集合 | opportunity set | rescue 和 loss 各自独立定义 | opportunity/converse opportunity/denominator 漂移 |
| 历史深度 | K | Fig. 4 写 history/prefix depth；Figs. 5–6 写 sequence length；各处说明 K 计数对象 | shallow/deeper prefixes 而不定义 K |
| 无历史条件 | no-memory state/reference | 无先前记忆状态 | 与 cue-only 不加区分地互换 |
| 部分提示输入 | cue-only input | 只保留部分 target spikes 的提示 | 被误写成 no-memory state |
| 两者组合 | cue-only no-memory reference | cue-only input applied to no-memory state | 将 no-memory 与 cue-only 当纯同义词 |
| 单项目参照 | singleton state/reference | 全文统一 singleton | single-item 与 singleton 混用 |
| 提示强度 | keep probability | “fraction of encoded target spikes retained in the cue” | 只在图注写 0.5 |
| 状态移植指标 | donor-transfer index | donor-minus-receiver direction projection normalized by squared donor–receiver distance | 只有数值无定义 |
| 结构复杂度 | effective component number, Neff | 说明它是结构表达量，不是容量/可访问项目数 | 首用无 caveat |
| 历史打乱参照 | sequence-deranged (permuted-order) composite | 首用解释 permuted-order | deranged composite 无解释 |
| 距离/相似度 | centered-cosine distance/similarity | 首用给 centered 的计算对象 | centered 与普通 cosine 随意互换 |
| 控制删除 | area- and energy-matched removal | Results、Methods、caption 一致 | exact area-and-energy-matched / input-energy–matched |

## 4. 执行顺序与依赖

不得按“从 Abstract 顺着改到 Methods”的自然阅读顺序施工。正确顺序是先冻结证据和定义，再改 Methods，再改 Results/Abstract，最后做版式和投稿包。

### Batch A：建立修订底座

1. 复制原 DOCX 为工作稿；记录原稿 SHA-256；原文件只读保留。
2. 从当前 final-six bundle 生成 `manuscript_statistics_table.csv`。每行至少包含：figure、panel、endpoint、contrast、estimate、CI、independent unit、n、test、alternative、raw P、adjusted P、adjustment family、source path、source hash、statistics status。
3. 从实际 run config 与源代码生成 `model_protocol_parameters.csv`。严禁从类默认值代替实际运行值。
4. 建立术语检查表：记录首次出现、正式定义、允许缩写、禁止变体。
5. 对 `prespecified` 做逐项证据审计：若没有分析前形成且可引用的记录，则删除“prespecified”，在主文直接写数值阈值或 neutral defined threshold；不得用 Supplementary 指针替代定义，也不得事后制造预注册含义。

**Batch A 验收：**所有正文数值和术语修改都能回指一个当前权威文件；无法回指的项进入“author decision”，不得猜填。

### Batch B：先修 Methods 与 Statistics

#### B1. 准备 Supplementary Table S1，并压缩主文 Methods

下列已核定项目全部进入 Supplementary Table S1；主文 Methods 只保留理解模型、训练与主要分析所必需的数值，不复制参数清单，也不设置 S1 指针。

**Input and encoding**

- MNIST 官方 training/test split：60,000/10,000 images；28 × 28；只做 resize 和 tensor conversion。
- 两通道 ON/OFF DoG：kernel 7 × 7，σ1 = 1.0，σ2 = 2.0，threshold = 0.05，通道内最大值归一化。
- rank-latency encoding window = 20 steps；θ frequency = 5 Hz，γ frequency = 50 Hz；active gamma indices = 0, 3, 6；Δt = 1 ms。
- 说明每项 assay 的 encoded-input duration；不要用 encoder 的 `max_duration` 默认值代替实际 protocol 时长。

**Neuron and STSP dynamics**

- Vreset = −60 mV，VL = −70 mV，VE = 0 mV，Cm = 0.1 nF，gm = 10 nS，τe = 5 ms，refractory period = 20 ms，Δt = 1 ms。
- U = 0.2，τD = 100 ms，τF = 1,000 ms。
- 训练时 STSP disabled；post-training assays 启用 dynamic STSP；加载后所有层权重按 1/U = 5 缩放以保持 baseline synaptic gain。

**Architecture and competition**

- Layer 1：2→30 channels，kernel 5，stride 1，padding 2，2 × 2 max-pooling，top-k = 5，initial weight 0.6 nS，inhibition 20 mV，τinh = 10 ms。
- Layer 2：30→150 channels，kernel 3，stride 1，padding 2，2 × 2 max-pooling，top-k = 10，initial weight 0.6 nS，inhibition 20 mV，τinh = 10 ms。
- Layer 3：150 input channels，10 classes × 20 neurons/class，input spatial size 8，top-k = 1，initial weight 0.8 nS，inhibition 10 mV，τinh = 10 ms。
- 明确 Layer 3 不是一个含糊的“decision window”：候选读出发生在 `t mod 60 ms = 20 ms` 的离散 decision gate；分类采用本阶段首次 eligible Layer 3 spike。若 assay 使用 phase-reset clock，要在 protocol 行写明。

**Learning and ensemble**

- Layer 1/2 learning rate = 0.001；Layer 3 learning rate = 0.01；τ+ = 20 ms；τelig = 20 ms；wmin = 0，wmax = 1 nS；target trace level = 0.5；reward/punishment maxima = 1.0。
- 将 Eq. 7 的 target-level 符号由 `x_tar` 改为不与 STSP resource x 冲突的 `z_target`，公式含义不变。
- 20 个网络、seed 1000–1019、batch size 512、epochs 2/10/100；训练集 shuffle、测试集不 shuffle；无 validation split。
- 报告 final classification accuracy：91.158% mean，95% CI 90.998–91.318%，range 90.76–91.90%，n = 20 networks。

#### B1a. 主文 Methods 的落笔方式：原位整段替换，不新增主文表格

不采用“每个参数追加一句”的补丁式写法，也不在主文插入参数表。保留现有 Methods 标题、段落功能和公式顺序，在五个既有位置用紧凑句群替换/补足：

1. `Input encoding and network model` [101–103]：两句内补 MNIST split、三层核心 architecture、2 × 2 pooling 和离散 Layer 3 decision gate；DoG 细节不进入主文。
2. `Spiking and STSP dynamics` [107–131]：在 LIF 定义段加入一条集中参数句，在 STSP 定义段加入 `U, τD, τF` 一条参数句；不逐参数另起句。
3. `Training and fixed-circuit simulations` [135,167–169]：把 batch size、2/10/100 epochs、训练时 STSP disabled、测试时 1/U gain compensation、n = 20 和 baseline accuracy 合并为两个连续句群。
4. 核心 assay 定义 [185–233]：沿用现有 rescue/loss、passive、transfer、centered-cosine 等定义，只修术语；仅在 [223] 增加 keep probability 的同位语定义，在 [197] 直接给两个 thresholds。
5. `Statistics and reproducibility` [237–245]：整段压缩重写为三段——独立重复单位与 CI、n/cohort/α、test direction 与 BH/Holm/unadjusted families；不新增统计表，也不依赖 Supplementary 指针。

这种写法预计只增加少量净正文，同时让每个值紧邻其科学对象；不得在 Results 或 Discussion 补方法参数。

**主文 Methods 实际落笔示例（作为写法模板，不是最终定稿文字）：**

1. `Input encoding and network model` 段内直接增加一句：

   > Networks were trained and tested on the standard MNIST split (60,000 training and 10,000 test images; 28 × 28 grayscale). The trained model was a three-layer feedforward spiking network: Layer 1 (2 → 30 feature maps), Layer 2 (30 → 150 feature maps) and Layer 3 (ten classes × 20 readout neurons), with 2 × 2 max-pooling after Layers 1 and 2. During post-training simulations, the selected class was defined by the earliest eligible Layer 3 spike at a discrete decision gate (one 1-ms step every 60 ms, starting 20 ms into each decision window).

2. `Spiking and STSP dynamics` 段内在 LIF 与 STSP 定义处分别插入：

   > Each layer used a conductance-based leaky integrate-and-fire neuron with membrane capacitance 0.1 nF, leak conductance 10 nS, leak reversal −70 mV, excitatory reversal 0 mV, reset −60 mV, membrane time constant 5 ms and a 20-ms refractory period (simulation step, 1 ms). STSP used baseline utilization U = 0.2, depression recovery τD = 100 ms and facilitation recovery τF = 1,000 ms.

3. `Training and fixed-circuit simulations` 段内把训练与加载说明合并为：

   > All networks were trained with dynamic STSP disabled using stochastic mini-batches of 512 images; Layers 1, 2 and 3 were trained sequentially for 2, 10 and 100 epochs, respectively. After training, long-term weights were scaled by 1/U (= 5) to preserve training-time drive at the resting STSP state and then fixed. The main analyses used 20 independently trained networks (seeds 1000–1019); mean test accuracy before the working-memory assays was 91.158% (95% CI, 90.998–91.318%; n = 20 networks).

4. `Statistics and reproducibility` 整段压缩为三段：

   > **Inferential unit and intervals.** The independently trained network was the inferential unit. Trial-, pair-, sequence-, stage-, unit-, site- and event-level measurements were aggregated within each network and condition before any cross-network comparison; lower-level counts are descriptive sample sizes. Unless specified otherwise, estimates are cross-network arithmetic means with two-sided 95% confidence intervals.
   >
   > **Cohort and significance.** All current main-figure analyses used the 20-network cohort (seeds 1000–1019) fixed by the simulation design before endpoint analysis; no prospective power calculation was performed. No formal normality test or outlier-exclusion rule was applied. All inferential tests used a significance level of α = 0.05 unless a directional one-sided contrast was identified in the corresponding figure legend. Silent and no-response trials remained in unconditional readout denominators; unreached crossings and zero-denominator gains were treated as undefined.
   >
   > **Tests and multiplicity.** Figs. 2, 3, 5 and 6 used two-sided one-sample Student t tests of network-level contrasts against their stated nulls; Student-t CIs and Benjamini–Hochberg adjustment were applied only within the planned families listed in Source Data, and endpoints labelled unadjusted were not part of any family. Fig. 4 used one-sided exact sign-flip tests with 20,000-resample percentile-bootstrap CIs for directional transfer and recurrence endpoints, two-sided sign-flip tests for the rescue and loss contrasts, and Holm adjustment within the families listed in Source Data. Rescue and loss retained their separate opportunity denominators.

5. `Conditional effects` 段内在 [223] 的 partial-cue 句中加入同位语：

   > Partial cues retained a specified fraction of the encoded target (keep probability).

上述示例句群即为主文 Methods 的唯一增补载体；除 [197] 阈值句、[223] keep probability 和 [233] area- and energy-matched 外，不逐句追加参数。

#### B2. 准备 Supplementary Table S2，并保留主文核心定义

Supplementary Table S2 中每个 inferential endpoint 一行；同一 panel 内分母或 null 不同的 endpoint 必须分行。主文 Methods 只保留 rescue/loss 分母、独立重复单位、关键窗口、检验方向和多重校正原则。S2 的列固定为：

`Figure/panel | scientific endpoint | reference condition | eligible/opportunity set | numerator | denominator | time window | exclusion/undefined rule | within-network aggregation | cross-network estimand | independent unit | n | null/SESOI | test | sidedness | multiplicity family | reported P source`

必须显式覆盖：

- Fig. 1：全部为 descriptive endpoints；85–95% 只作为 descriptive reference range，不写成无出处的 prespecified range。
- Fig. 2b：rescue 只在 S0-error anchors 上计算；loss 只在 S0-correct anchors 上计算；两个 rate 不相减、不共享分母。
- Fig. 2c：common-update cosine threshold = 0.5；history-residual norm-ratio threshold = 0.05；说明 manuscript 报告的是 estimate-minus-threshold margin 还是原始 estimate。
- Fig. 2d：changed-event residual magnitude 与 size-matched random control 的配对方式。
- Fig. 3：30-ms descriptive early window、first-50-ms inferential window、overlap ≥ 0.05、zero-overlap = 0、random/attenuation/reset 的 matching 规则、static-frozen control 是否允许 STSP mutation。
- Fig. 4：state-transfer 时改变哪一层、保留哪些层、fast-state equalization、相同 C；stages 2–10；equal-time passive branch；K = 1/5 的 opportunity-set 平衡规则。
- Fig. 5：K = 3/5/7/10；50 shuffled pairs；nine sequence-deranged composites；Neff 非容量指标；effective area 的精确定义；20,000-draw bootstrap 的作用。
- Fig. 6：cue-only/no-memory/singleton/sequence-state 的正交定义；K = 10、400-ms、keep probability 0.5；target position；same-label novel 与 unseen-class；upper-20% support、overlap ≥ 0.05、primary 10-ms two-by-two interaction。

#### B3. Statistics 段落的固定改法

1. 作者确认并写明 significance level；默认不得由编辑擅自填入。若项目约定为 α = 0.05，写成：“All inferential tests used a two-sided significance level of α = 0.05 unless a directional one-sided contrast was identified in the corresponding figure legend.”
2. 明确 independent inferential unit 是 independently trained network；trials、anchors、history families、sites 和 network×condition cells 不是独立 n。
3. 明确 CI 角色：descriptive Student t CI 与 bootstrap CI 不是 hypothesis tests。
4. 分别列出 BH、Holm 和 unadjusted planned test 的 family；不得只写“corresponding statistics tables”。
5. 正文和图注由 `manuscript_statistics_table.csv` 自动填充实际 P；禁止人工复制科学计数法。
6. 将 confirmatory-19 sensitivity 写入 Supplementary Table S2/Fig. S2d，明确它排除 seed 1000 且不替换主 n = 20 cohort；主文不重复其具体数值。
7. 说明没有 formal power calculation，并保持现有“所有 20 个网络均作为独立重复”表述。
8. 把 exact-input assay 的花园路径句改为：`In each branch, the Layer 1 pooled event sequence produced by B from the no-memory reference was either left unmanipulated or replayed identically into both histories.`

**Batch B 验收：**Methods 单独交给未读过代码的读者后，读者能回答“模型有多大、参数是多少、训练如何完成、每个 endpoint 的分母是什么、何时采样、n 是什么、用了什么检验和校正”。

### Batch C：按段落修正文与阅读流

下表中的 `[N]` 沿用上一轮全文编号。修改时必须锁定段落功能和结论，不移动证据顺序。

| 位置 | 必做修改 | 推荐落法 |
|---|---|---|
| Abstract [7] | 用结果式主句替代纯方法式 `Here we used...`；消除 `as` 的因果/时间歧义 | 改为 `Here we show, using exact-input counterfactuals and selective state transfers, that...`；将 `as inherited STSP shaped...` 改为 `while inherited STSP shaped...` 或拆句；最终重算 ≤150 words |
| Introduction [10] | 同段两个 `However`；`therefore` 位置生硬 | 第二个改 `Yet`；末句改 `It therefore remains unclear how...` |
| Introduction [12] | inherited/passive/successor 首用缺口；`online` 偏离全文术语 | 写 `inherited (pre-input) state`；写 `state change expected under zero input`；将 `online working-memory organization` 改为 `continual organization of working-memory representations` |
| Results [16] | u/x 未定义；`Before...first`；时态；图号 dash | 首次写 `joint u/x state (STSP utilization u and available-resource x)`；删 `first`；`had fallen...and had remained`；`Fig. 1b–d` |
| Results [17] | `distinct...differently` 重复 | 保留一次差异标记，例如 `whether different inherited states transform the same later input` |
| Results [21] | loss 与 opportunity set 不清 | 直接写 `loss counted baseline-correct B trials made incorrect after history`；下一分句写 rescue/loss 使用 separate opportunity sets |
| Results [22] | common/shared/common-update 三套名称 | 全段统一为 `common input-driven component`；residual 保持 `history-conditioned residual`；在句内直接给出两个数值阈值，不用无出处 `prespecified`，也不增加 Supplementary 指针 |
| Results [26] | static-frozen、support、advance、条件名漂移 | 首次写 `static-frozen control (STSP held at baseline)`；首次写 `effective STSP support (u ⊙ x)`；`early spike advancement`；统一 `dynamic STSP` 和 `size-matched random control` |
| Results [27] | fast variables、donor-transfer index、inherited condition | 写 `fast state (membrane potential, excitatory conductance, refractory state and lateral inhibition)`；一行解释 donor-transfer index；改回 `inherited state` |
| Results [31] | K 未定义；shallow/deeper；Fig. 4 错引 a | 写 `at history depths K = 1 and K = 5` 或 `at shallow and deep prefix depths...`；引用由 `Fig. 4a–c` 改为 `Fig. 4b,c`，因为 a 为 schematic |
| Results [32] | `Sufficiency at...` 名词化；`motif` 无定义；`At the same time`；inherited condition | 改 `The sufficiency shown by selected transplants did not establish recurrence during unmanipulated sequences.`；用 `transition pattern`；`Over the same range`；改 `inherited states` |
| Results [36] | centered/plain cosine 可能混用 | 首次说明模板相似度采用 centered-cosine 还是 ordinary cosine；之后只保留真实 metric 名 |
| Results [37] | Neff 无首用 caveat；deranged 无解释；complementary 过密 | 首次写 `effective component number (a measure of structural expression, not storage capacity)`；写 `sequence-deranged (permuted-order) composites`；减少至少两处 `complementary` |
| Results [41] | no-memory/cue-only/singleton 关系不清；keep probability 只在 caption | 明确 state 与 input 两个轴：`cue-only no-memory reference`、`singleton state`、`sequence state`；首次定义 keep probability；CI dash 统一 |
| Results [42] | area/energy label；`caused`；six experiments；尾句重复 | 统一 `area- and energy-matched removal`；按 claim-proof matrix 校准 `caused`，若不允许则用 `produced`；`six analyses` 或 `six assays`；删除或改写与 Abstract/Discussion 重复的最后一句 |
| Discussion [46] | inter-layer transition 首次出现无解释 | 写 `inter-layer successor transition (from inherited Layer 1 state to downstream successor formation)`，不增加新层级结论 |
| Discussion [48] | `for temporal gradients and replay` 附着歧义 | 改 `accounts based on serial-position dynamics or on multi-timescale augmentation for temporal gradients and replay` |
| Discussion [49] | `leaving whether...an empirical question` 花园路径 | 改 `while leaving it an empirical question whether comparable transitions operate in biological circuits` |
| Methods [59] | 定义列举省略不一致 | 改 `V denotes the membrane voltage, Cm the membrane capacitance, gm the leak conductance, VL the leak reversal potential, and VE the excitatory reversal potential` |
| Methods [88] | 列表边界歧义 | 在 `later inputs or cues` 后加逗号 |
| Methods [103] | STSP 作不可数现象使用 | 写 `Receiver Layer 1 and Layer 3 STSP states were retained` |
| Methods [104],[120] | `within network` | 统一 `within each network` |
| Methods [111] | `u x` 与 Eq. 4 不一致；effective area 欠定义 | 写 `elementwise product u ⊙ x`；补 effective area 的计算公式/阈值/归一化 |
| Methods [113]–[115] | cue-only/no-memory、single-item/singleton、keep probability | 按第 3 节术语表重写条件组合；定义 keep probability；统一 singleton |
| Methods [114] | unseen-class 句尾省略 | 补 `whose class was absent from the sequence` |

#### C1. 统一性机械清扫

完成上表后一次性处理：

- panel、stage、seed、time ranges 全部用 en dash：`Fig. 1b–d`、`stages 2–10`、`seeds 1000–1019`、`100–1,200 ms`。
- CI 范围统一一种规则；优先使用 en dash，例如 `90.60–91.29%`。负数范围可用 `−0.0739 to −0.0590` 避免连续符号混淆。
- 统一 Oxford serial comma。
- `non-specific`、`area- and energy-matched` 等复合词按同一规则。
- 标题中的 U+2011 non-breaking hyphen 换成普通可搜索连字符，并同步修改 DOCX core properties，避免 PDF metadata 出现 `Working-?emory`。
- 不更改已确认的 British/Oxford spelling（`behaviour` + `-ize/-ization` 是允许组合）。

**Batch C 验收：**术语扫描不得再命中未定义/漂移的主名称；Results 段落仍保持 Action→Data→Inference，数字和 Figure callout 顺序不变。

### Batch D：统计数字、图注与正文联动

#### D1. 实际 P 值替换规则

1. 主文不再使用笼统的 `P < 0.001`；改为来源表中的 adjusted P 或 raw P，并写明校正。
2. P ≥ 0.0001 用小数；P < 0.0001 用科学计数法；保留 2–3 个有效数字。CSV 保留全精度。
3. 同一句含多个 endpoint 时，按出现顺序逐个列值，禁止写 `both/all P = ...`，除非数值确实相同。
4. 图注只报告 primary/confirmatory contrasts；descriptive panels 明确标为 descriptive，不人为添加检验。

已核出的主文替换源如下；最终仍由自动汇总表写入，避免手抄错误。

| Figure/panel | 当前主对比的 adjusted P |
|---|---|
| Fig. 2b | rescue 3.07 × 10^-4；loss 1.78 × 10^-4（BH） |
| Fig. 2c | common component 3.64 × 10^-55；residual 1.53 × 10^-45（BH） |
| Fig. 2d | 7.27 × 10^-50（BH） |
| Fig. 3a | 2.76 × 10^-11，2.76 × 10^-11，7.80 × 10^-4（BH） |
| Fig. 3b | 2.05 × 10^-91，7.08 × 10^-49，3.82 × 10^-15（BH） |
| Fig. 3d | 1.99 × 10^-32，3.85 × 10^-36（BH） |
| Fig. 3e | 1.24 × 10^-36（unadjusted planned contrast） |
| Fig. 3f | 2.44 × 10^-49（BH） |
| Fig. 4b,c | 四个 K-by-endpoint transfer contrasts 均为 3.81 × 10^-6（Holm） |
| Fig. 4d | across-stage mean observed-minus-passive 1.91 × 10^-5（Holm） |
| Fig. 4e | rescue decline、loss increase 均为 1.91 × 10^-5（Holm） |
| Fig. 5b | 2.13 × 10^-38（BH）；a,c–f 保持 descriptive |
| Fig. 6a | 3.03 × 10^-33、2.55 × 10^-20、3.03 × 10^-33、5.24 × 10^-8（BH；按 A/B 与两参照顺序） |
| Fig. 6b | 5.14 × 10^-19（BH） |
| Fig. 6c | 1.60 × 10^-6、1.03 × 10^-20（BH） |
| Fig. 6d | 1.13 × 10^-13（unadjusted planned interaction） |
| Fig. 6e | 5.01 × 10^-30（BH） |
| Fig. 6f | 3.32 × 10^-40（BH） |

#### D2. 主图图注具体修改

- Fig. 1：保留 b–e descriptive；把 “prespecified descriptive range” 改为 “descriptive reference range”。
- Fig. 2：加入 rescue/loss 的两个独立 opportunity-set 定义；c 明示两个 thresholds；b–d 各补 test、n、adjustment family 与实际 P。
- Fig. 3：将 `static-frozen update opportunity` 改为定义后的 `static-frozen control`；a,b,d,f 补实际 BH P；e 补实际 unadjusted P；c 明确仅 descriptive。
- Fig. 4：`stages 2-10` 改 `stages 2–10`；去掉无出处 `prespecified opportunity sets`；b–e 补实际 Holm P；d 保留 secondary-analysis 身份。
- Fig. 5：`Neff` 采用统一数学排版并加一句结构表达 caveat；定义 sequence-deranged；仅 b 报 inferential P，a,c–f 明确 descriptive。
- Fig. 6：定义 keep probability；把 no-memory/cue-only/state labels 按两个轴拆清；a–f 补实际 P 与 adjustment status；Fig. 6d 明确是 unadjusted planned interaction。

#### D3. 图本体与 alt text

- 不修改六张主图像素、panel 顺序、颜色或数据。
- 更新 Fig. 3/5/6 的 alt-text 标题，使其与当前 caption title 完全一致；同时检查 Fig. 1/2/4。
- 维持当前图像显示比例：Fig. 2 为 6.10 × 3.77 in；其余约 6.10 × 5.62 in；像素与 Word extent 比例已匹配，不拉伸。
- 最终人工在印刷尺寸检查 panel letters、axes、legends、CI、灰度与最小文字；视觉模型先前提出的 Fig. 3/4 swap 已被哈希证伪，不纳入修改。

**Batch D 验收：**正文数值、caption 数值、Supplementary Table S2 和 Source Data 可按 endpoint 一一连接；不存在 `P < 0.001`；所有 descriptive/inferential 标签一致。

### Batch E：Supplementary Information 独立成册

S1–S7 作为独立补充证据包交付，不向主文增加引用、指针或次级结论。

| Supplementary figure | 仅在补充材料中承担的内容 |
|---|---|
| S1 | delay decay、sample direction、donor flux 与 opportunity null |
| S2 | L2/early-L3 transfer、DTI robustness、untouched-19 sensitivity |
| S3 | 30-ms/50-ms windows、winner cap、distance threshold、original-winner fate |
| S4 | stages 2–10 prevalence、worst-stage endpoint、u/x trajectories |
| S5 | model/variable/continuous geometry/categorical pair robustness |
| S6 | amplitude、effective area、topology、history identity |
| S7 | estimability、coverage、site availability 与 undefined rules |

执行要求：

1. 形成正式 `Supplementary Information` 文件，含 S1–S7、完整 legends、panel labels 和自身可解析的条件定义。
2. Supplementary Tables S1–S2 与 S1–S7 在补充文件内部交叉一致，不依赖主文指针才能理解。
3. confirmatory-19 的具体数值只写入 S2 legend/Supplementary Table S2。
4. supplementary engineering gates、development checks、excluded underpowered contrasts 和 secondary controls 不进入主文。
5. Supplementary Source Data 通过投稿包 manifest 关联，不在主文 Data availability 中逐项枚举 S1–S7。

**Batch E 验收：**Supplementary 文件、Supplementary Tables、Source Data 和 figure audit 的 S1–S7 编号一致；主文无 S1–S7 或 Supplementary Table callout；`supplementary_information` 不再 pending。

### Batch F：Data/Code availability、参考文献与后置材料

#### F1. Data availability

保留 MNIST 与 Figs. 1–6 Source Data 声明，增加：

- Source Data 包的稳定文件名/仓库路径或 DOI；
- network-level statistics 与 manuscript statistics table 的位置；
- Supplementary Figs. S1–S7 的 Source Data 范围；
- 不写本地 `results/...` 路径作为读者唯一入口。

#### F2. Code availability

1. 在公共仓库确认目标 SHA 可访问后，把旧 SHA 替换为 `93d9b8295fdfbd603d3f181a3500773cb4689a75`，或优先替换为已发布 release tag/Zenodo DOI；若该 SHA 尚未推送，不能提前写入稿件。
2. release 必须包含模型、编码、训练、六图 experiment runners、plot-only entrypoints、configs/specs、环境版本与复现说明；否则缩窄 availability 声明，不得声称未发布的代码已提供。
3. 稿件、`SOURCE_IDENTITY.json`、release archive manifest、Reporting Summary 使用同一个身份。
4. 提供一条 plot-only replay 总入口或明确列出六个正式入口；不得要求重新模拟才能重画图。

#### F3. References

- 不重开文献科学适配性；只做格式机械核验。
- 保留现有 1–33 顺序与全部 DOI。
- Ref. 31 按 eLife Reviewed Preprint 的当前官方记录核验卷页/版本写法；确认后再改，不猜测。
- 用期刊 CSL/参考文献管理器统一 journal abbreviation、斜体、页码 dash；不手工改变作者截断规则。

#### F4. Reporting Summary 与声明

- 完成 Communications Biology Reporting Summary，至少同步：sample size、independent unit、replication、randomization、blinding/not applicable、data exclusion、software、data/code access、statistical tests。
- Acknowledgements、Author contributions、Competing interests 当前齐全，除作者事实变化外不改。

**Batch F 验收：**外部读者可从稿件给出的链接取得与当前六图一致的数据和代码身份；Reporting Summary 与 Methods 无矛盾。

### Batch G：DOCX 包与版式清洁

1. 将文档及 styles/default language 从 `zh-CN` 改为英语语言标签；正文保持 Oxford spelling。
2. 更新 core properties 的 title/subject/author；标题使用普通 hyphen，PDF metadata 不再乱码。
3. 清除未被 document relationships 引用的 `word/media/image2.png`–`image6.png`；保留当前实际引用的 image1、image7–image11。
4. 更新所有 figure alt descriptions，移除旧标题。
5. 确认无 tracked changes、comments、broken relationships、外链图片、缺失字体、重复 section break。
6. 保留并核验 page-number field；不手工写死页码。

**Batch G 验收：**DOCX 只包含六张实际引用主图及必要对象；重新打开、另存、转 PDF 均无修复提示。

## 5. 最终 QA 顺序

### QA-1 内容与术语

- 逐段比较原稿与修订稿：claim、evidence、citation、figure order、paragraph function 不变。
- 扫描禁止变体：`shared component`、`fast variables`、`inherited condition`、未定义 `static-frozen`、`online working-memory organization`、`six experiments`、`Fig. 4a-c`、无指针 `prespecified`。
- Abstract ≤150 words、无引用；标题词数与作者信息不变。

### QA-2 数值与统计

- 把正文/legend 中每个 estimate、CI、P、n 与 `manuscript_statistics_table.csv` 逐项 diff。
- 所有 P 都有 test、sidedness、adjustment status；所有 CI 都标明 Student t 或 bootstrap。
- n = 20 始终代表网络；untouched-19 明确为 sensitivity；不得把 trial/site/anchor 当 n。
- Supplementary Table S2 的 family 与 Source Data adjustment 列一致。

### QA-3 图与图注

- 六张嵌入图重新计算哈希并与 canonical `fig1.png`–`fig6.png` 对照。
- 正文首次引用、主图放置、caption 与 alt text 一致；Supplementary 文件单独核验，不检查主文 callout。
- PDF 页面按最终尺寸检查：无裁切、重叠、乱码、过小文字；公式符号 τ、Δ、⊙、≤、≥ 完整。

### QA-4 引用与可得性

- 引文 1–33 全覆盖、无 orphan citation、DOI/URL 可点击。
- Data/Code availability 的 release/DOI/SHA 可从未登录浏览器访问。
- Source Data、Supplementary Information、Reporting Summary 文件均在投稿包 manifest 中。

### QA-5 DOCX/package integrity

- 检查 comments=0、tracked changes=0、broken rels=0、orphan media=0。
- 语言元数据、core properties、页码、headings、caption styles 正确。
- 把工作稿转成 PDF，重新抽取全文与逐页渲染；核验 28 页左右的分页变化是由新增 tables/legends 合理造成，而不是内容丢失。
- 仅在上述检查全部通过后，把 `PACKAGE_STATUS.json` 的 article、supplementary、source data、journal policy materials、submission readiness 更新为真实状态。

## 6. 两项作者决策门

只有以下两项不能由编辑根据现有文件擅自决定；其余均可直接执行。

1. **α 的正式值。** 当前稿件与统计文件显示以 0.05 为常规阈值的迹象，但没有一份当前权威文件显式声明。作者确认后写入 Methods/Supplementary Table S2；未确认前不得猜填。
2. **公开代码身份。** 需要确认 `93d9b8295fdfbd603d3f181a3500773cb4689a75` 已推送且可公开访问，或提供最终 release tag/DOI。只有可访问身份才能进入 Code availability。

Supplementary S1–S7 的默认方案已由现有 audit 文件确定为保留；除非作者另行改变投稿组成，不再把它作为开放科学结论问题。

## 7. 最小优先路线

若按依赖关系执行，顺序必须是：

1. 生成参数表和统计主表；
2. 完成精简的 Methods/Statistics 与 Supplementary Tables S1–S2；
3. 替换 Results/Abstract/Discussion 的术语、定义、实际 P 和局部句法；
4. 同步六个主图 legends 与 alt text，并独立完成 Supplementary S1–S7；
5. 修复 Code/Data availability、Reporting Summary 和 release identity；
6. 清理 DOCX 元数据与 orphan media；
7. 做独立内容 diff、数值 diff、PDF 视觉检查和投稿包完整性检查。

不得跳过 1–2 直接润色 Abstract，也不得在统计主表完成前人工替换 `P < 0.001`。

## 8. 完成定义

本轮修订只有同时满足以下条件才算完成：

- P1 全部关闭，P2 无遗留或有明确书面保留理由；
- 原固定科学问题、六段 Results 结论与 Discussion 边界未改变；
- 主文 Methods 自足但不过载，Supplementary Tables S1–S2 完整且有当前来源；
- 正文、图注、Source Data 的全部主数值一致；
- 六张主图仍是当前 canonical artwork；
- Supplementary S1–S7 在补充文件与投稿包内部闭环，主文无额外指针；
- 可公开访问的数据/代码身份与稿件一致；
- DOCX/PDF 无可见版式缺陷、无 tracked changes、无 orphan media、无失效关系；
- `PACKAGE_STATUS.json` 只有在真实满足所有门槛后才标记 submission-ready。
