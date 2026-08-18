# V6.1 修改确认记录

- 基线主文：`docs/paper/v6.docx`
- 主文确认稿：`docs/paper/v6_1.docx`
- 补充材料迁移基线：`docs/paper/supplementary_information - 副本.docx`
- 补充材料确认稿：`docs/paper/supplementary_information_v6_1.docx`
- 原则：每一处经作者确认后才进入 V6.1；正式版与副本均保持不变。

## 已确认

### R1 — Results P018

**作者选择：C（融合重写）**

**写入文本：**

> We first asked whether STSP retained information about an earlier input after stimulus-evoked firing had ceased and whether the retained synaptic state influenced subsequent network output.

**理由：** 保留正式版的简洁性，同时将后续两组证据所回答的“信息保留”和“功能影响”两个问题明确并列；删除副本中的重复表达。

### R2 — Results P021

**作者选择：C（融合重写）**

**写入文本：**

> In the trained networks, persistent digit-class information and donor-directed readout after state exchange together showed that the retained STSP state linked earlier stimulus processing to subsequent network output without sustained firing between inputs.

**理由：** 保留两条决定性证据，同时把结论提升到 earlier processing 与 subsequent output 的功能联系；将过宽的 “without sustained activity” 收紧为证据直接支持的 “without sustained firing between inputs”。

### R3 — Results P024

**作者选择：C（融合重写）**

**写入文本：**

> Having established that retained STSP influenced subsequent network output, we next asked whether different inherited states would condition the behavioral outcome and synaptic update elicited by the same second input B.

**理由：** 保留问题驱动式衔接，同时恢复 identical-B 设计的诊断作用，并明确对应 behavioral outcome 与 synaptic update 两个证据层。

### R4 — Results P027

**作者选择：B（采用副本）**

**写入文本：**

> The same input therefore imposed a common organization on state updating across histories, while inherited history selectively conditioned the course of that update and its behavioral consequences.

**理由：** 先确立 current input 提供跨 history 的共同组织，再说明 inherited history 施加选择性调制；该信息顺序与本节证据及全文核心机制一致。

### R5 — Results P030

**作者选择：C（融合重写）**

**写入文本：**

> The coexistence of common input-driven organization and selective history conditioning raised a mechanistic question: how does inherited STSP condition current-input processing and thereby shape the downstream successor?

**理由：** 保留副本的问题驱动结构，同时恢复 inherited STSP、current-input processing 和 downstream successor 三个关键机制对象，并明确正确的层间方向。

### R6 — Results P036

**作者选择：D2（定制压缩稿）**

**写入文本：**

> The preceding experiments established a single history-conditioned transition from an inherited state to a downstream successor. Whether successor reuse and recurrent input-associated updating beyond passive STSP evolution extended this mechanism across a sequence remained unresolved.

**理由：** 删除重复出现的 “We next asked” 模板，以 sequence-level unresolved relation 自然推进；保留 successor reuse 与 recurrence 两条证据主线，同时把具体条件留给后续证据段。

### R7 — Results P042

**作者选择：C（融合压缩稿）**

**写入文本：**

> Repeated history-conditioned updating left open how accumulated history was organized in the terminal STSP state—specifically, whether it retained multiple constituent contributions and their experienced configuration.

**理由：** 用一句话完成 recurrence 到 terminal organization 的转折；删除模板化提问和重复解释，并把宽泛的 organization 立即限定到 constituent retention 与 configuration specificity 两个真实端点。

### R8 — Results P048

**作者选择：D4（平行概念句）**

**写入文本：**

> Structural organization concerns the arrangement of accumulated history within the terminal STSP state; functional organization concerns the expression of that history during subsequent processing.

**理由：** 以平行概念句将上一节的 structural organization 与本节的 functional organization 置于同一论述层级；不重复 Fig. 4 已回答的影响存在性，也不预告后续具体解释变量。句式多样性按相邻开头的整体语法架构与话语功能判断，而非禁用特定短语。

### R9 — Results P051

**作者选择：B（采用副本）**

**写入文本：**

> The incoming input therefore engaged retained STSP selectively: cue–content match constrained its contribution to readout, while pathway overlap concentrated its influence within early circuit processing.

**理由：** 作为小节终结句，分别抽象 cue evidence 与 pathway evidence 所承担的功能层级，完整回答 functional organization；主动机制主语与冒号后的平行证据综合也形成了合适的句式变化。

### M11 — Methods P061–P064

**作者选择：C（采用副本分工，保留两段结构）**

- Heading：`Input encoding and network architecture`
- Paragraph 1：保留主文最低必要 encoding 与 dataset 信息；详细 DoG/threshold/normalization/latency 参数依赖 Supplementary Methods。
- Paragraph 2：保留两 encoder channels、三层 architecture、无 recurrent excitation，并加入 earliest-spike/no-spike scoring rule。
- 下一 Heading：`Spiking and STSP dynamics`

**依赖：** 后续必须保留 Supplementary Methods 的完整 `Input encoding details`；若拒绝该迁移，需回滚主文压缩。

### M12 — Methods Equation (1) 后的定义句

**作者选择：C（精确化版本）**

**写入文本：**

> Here, \(I_{\mathrm{syn}}\) denotes the STSP-scaled excitatory synaptic input, and \(\alpha_e\) denotes the one-step decay factor for excitatory conductance.

**理由：** 删除含糊的 `to the layer`，保留 synaptic input 与外部编码输入的区分，并明确 \(\alpha_e\) 所衰减的是 excitatory conductance。原有 OMML 变量与下标保持不变。

### M13 — Methods P068–P069

**作者选择：C（采用副本 spike-rule 压缩，修正 lateral-inhibition 层级）**

- P068：采用副本对 threshold、decision timing、reset 和 refractory period 的压缩。
- P069：仅把 7 × 7 Gaussian field 限定于 Layers 1–2；Layer 3 只报告 10 mV inhibition strength，不把 `sigma_cross = 0` 的 Layer 3 写成同一 Gaussian spatial field。

**理由：** 保留副本的清晰度，同时避免压缩语法引入不受实现支持的 Layer 3 Gaussian-field 解释。原有 Equation (2)、\(V\)、\(V_{\mathrm{thr}}\) 与 \(\theta_j\) 的 OMML 数学结构保持不变。

### M14 — Methods P072

**作者选择：C（术语与定义精确化）**

**写入文本：**

> The subscript “r” denotes the post-recovery value within the current 1-ms step, before spike-triggered updating. The depression and facilitation time constants were \(\tau_D=100\) ms and \(\tau_F=1{,}000\) ms, respectively. Effective STSP support was defined as the elementwise product \(G_{\mathrm{STSP}}\), which scaled the incoming drive.

**理由：** `r` 明确表示 post-recovery value；采用统一术语 `effective STSP support`，并保留原有 OMML 变量和下标。

### M15 — Methods P074

**作者选择：B（采用副本）**

> Here, \(\odot\) denotes elementwise multiplication; \(G_{\mathrm{STSP}}\) is distinct from excitatory conductance \(g_e\) in Equations (1) and (2).

**理由：** 删除对 `effective-support variable` 的重复定义，用一个分号句紧凑保留 elementwise operation 与 support/conductance distinction。

### M16 — Methods P076

**作者选择：B（采用副本）**

**理由：** 完整保留 recovery-before-update、spiking/nonspiking site 行为和 [0,1] 约束，仅压缩重复表达；原有 \(u\)、\(x\) OMML 保持不变。

### M17a — Training rules 与 Equations (6)–(13) 迁移

**作者选择：C（融合概览并接受迁移）**

> Networks were trained sequentially by layer for 2, 10 and 100 epochs in Layers 1–3, respectively. During each stage, only the layer under training was plastic, while previously trained layers supplied fixed spike inputs; STSP was disabled throughout training. Layers 1 and 2 used local timing-dependent plasticity, whereas Layer 3 used reward-modulated plasticity over class-grouped readout neurons. Learning equations, rates, bounds and reward schedules are provided in Supplementary Methods and Supplementary Table S1.

**依赖：** 正式主文 Equations (6)–(13) 删除；后续必须保留 byte-identical Supplementary Equations (S1)–(S8) 及 Table S1。

### M17b — Training 后 fixed-circuit 状态

**作者选择：C（修正动态变量范围）**

> After training, all long-term weights and learned homeostatic thresholds were frozen. Post-training simulations were initialized at \(u=U=0.2\) and \(x=1\). To preserve the synaptic scale learned without STSP, each feedforward kernel was rescaled once by \(1/U=5\) when loaded. Throughout state formation, assays and controls, fast neuronal variables and protocol-appropriate STSP dynamics evolved without long-term plasticity.

**理由：** 删除错误的 `only u and x evolved`；明确 fast neuronal state 会演化，而 long-term weights/thresholds 固定，STSP dynamics 服从具体 protocol。

### M17c — State formation、decoding 与 restoration

**作者选择：C（范围、端点和采样时间修正）**

- Heading：`State formation, boundary capture and controlled restoration`
- \(\phi\)：明确为 specified layer 内的 joint STSP vector。
- Endpoint：`item identity` 改为 `digit class`。
- Sampling：恢复 100、200、400、800、1,200 ms。
- Restoration：保留 full-boundary、STSP-only 与 selective-transfer 的最小必要区别；详细 reset/classifier 参数指向 Supplementary Methods。

### M17d — 三类 control 的操作定义

**作者选择：C（压缩但保留操作区别）**

> Three controls isolated dynamic-STSP, passive-recovery and inherited-history contributions. In the static-frozen control, \(u=U\) and \(x=1\) were held constant throughout the assay. The equal-time passive control evolved the inherited boundary under zero input for the same number of 1-ms steps, whereas the duration-matched no-memory baseline began from the resting network state. Full control operations are provided in Supplementary Methods.

**理由：** 避免副本单句造成不透明或错误的 `respectively` 映射，同时保留三类 control 的最低操作定义。

### M18a — Exact-input assay 开场

**作者选择：B（采用副本）**

> The exact-input assay compared two outcome-blind, preselected history branches, \(A\) and \(C\), at history depth \(K=1\) or 5. Both subsequently received byte-identical encoded input \(B\). A branch was designated aligned when its final history-item label matched \(B\) and mismatched otherwise; branch selection did not use the measured response to \(B\).

**依赖：** `A`/`C` 仅为 branch identifiers、不是固定 alignment categories 或 digit-class labels 的完整说明保留于 Supplementary Methods。

### M18b — History-contrast decomposition 与 Equations (14)–(17) 迁移

**作者选择：B（采用副本）**

**理由：** 主文保留 passive correction、native/replay comparison、\(T=L+\Gamma\) 及两个主要 endpoint；Equations (14)–(17) 迁移为 byte-identical Supplementary Equations (S9)–(S12)，完整 thresholds 与 event-level definitions 保留于 Supplementary Methods。

### M18c — Behavioral rescue/loss

**作者选择：B（采用副本）**

> Behavioral rescue and loss were defined relative to the no-memory baseline using baseline-incorrect and baseline-correct opportunity sets, respectively. All paired conditions began from the same restored boundary and received the identical \(B\) input.

**依赖：** incorrect→\(B\)、\(B\)→non-\(B\) 的完整定义保留于 Supplementary Methods。

### M18d — Exact-input mechanistic analyses

**作者选择：B（采用副本）**

**理由：** 主文保留 overlap grouping、targeted intervention、event classes、selective Layer 1 transfer 和 donor-transfer index；thresholds、group definitions 与 intervention equations 完整保留于 Supplementary Methods。

### M19a — Successor reuse transfer assay

**作者选择：B（采用副本）**

**理由：** 保留 post-\(B\) Layer 2 successor transfer、receiver Layer 1/3 STSP、fast-variable equalization、identical \(C\)、两类 controls、passive correction 和两个 downstream endpoints；符合 inter-layer successor-state mechanism contract。

### M19b — Recurrence analysis

**作者选择：C（自然化压缩）**

> Successor transfer was evaluated at history depths \(K=1\) and 5. Across stages 2–10, iterative updating compared the centered-cosine distance from each preceding joint STSP boundary to the observed successor with the corresponding distance to an equal-duration passive boundary. Stage-wise measurements were aggregated within network before inference, and rescue and loss were compared between history depths. Full pairing, correction and distance definitions are provided in Supplementary Methods.

### M20a–M20c — Accumulated-state morphology

- **M20a：B**，采用 two-item morphology 压缩。
- **M20b：C**，迁移 Equations (18)–(20) 为 Supplementary Equations (S13)–(S15)，并将 \(N_{\mathrm{eff}}\) 保持为正式 OMML 数学排版。
- **M20c：B**，采用 coefficient-free Layer 1 morphology 压缩，同时保持其与 Layer 2 component decomposition 为不同 measurement scales。

### M21a–M21d — Conditional-function assays

- **M21a：B**，采用 degraded-cue/AUC 概览。
- **M21b：B**，采用 cue-specificity 压缩。
- **M21c：B**，迁移 Equations (21)–(22) 为 Supplementary Equations (S16)–(S17)。
- **M21d：C**，将 control 明确写为 `area- and input-energy–matched removal`，并保留 primary 10-ms support-by-overlap endpoint。

### M22 — Statistics and reproducibility

**作者选择：C（压缩但不夸大 Table S2 完备性）**

- 保留 network-level inferential unit、20-network cohort、\(\alpha=0.05\)、test direction、CI 和 multiplicity adjustment。
- Table S2 cross-reference 收缩为 endpoint-specific aggregation、nulls、sample sizes 和 test results，不声称未验证的内部 metadata 已完整报告。
- 整段设置为 keep-together，避免页面断裂破坏可读性。

### S1 — Supplementary Methods

**作者选择：C（采用迁移架构并补足复现信息）**

- 保留 Supplementary Equations (S1)–(S17)。
- 补入 delay-decoding 采样点：100、200、400、800、1,200 ms。
- 补入 static-frozen、equal-time passive 和 duration-matched no-memory 三类 control 的操作定义。
- 增加 `Supplementary Figures` / `Supplementary Tables` 同级结构，结束 `Statistical analysis details` 的错误 outline 继承。
- Statistical analysis details 删除 legacy/internal audit 语言，改为读者可核验的报告范围。

### S2 — Supplementary Fig. S2 endpoint

**作者选择：C（采用术语修正并首次定义）**

- `early Layer 3 successor` 改为 `early Layer 3 class-score vector`。
- 首次定义为 sampled early checkpoint 上，对每个 digit-class group 的 20 个 readout-neuron voltages 取最大值后形成的 ten-dimensional vector。

### S3 — Supplementary Table S1

**作者选择：B（采用副本纠错）**

- Input format：`28 × 7` → `28 × 28 grayscale MNIST`。
- 内部源码路径说明压缩为 Source Data 中提供 machine-readable settings 的读者表述。

### S4 — Supplementary Table S2

**作者选择：C（重构，不采用 26-column 内部 registry）**

- S2A：D01–D06 descriptive endpoints，含 Fig.5b 零值 controls 与 Fig.5c next-response gate。
- S2B：M01–M39、S01 与 SF01–SF04 的 endpoint、eligible set/window、\(n\)、estimate、unit 和 95% CI。
- S2C：对应 null、test/alternative/CI、statistic/df、raw \(P\)、adjustment 和 adjusted \(P\)。未由 frozen aggregate 持久化的 family member lists 不作推测；以 figure legends 与 Source Data 为权威。
- M16–M19 保留原四端点 transfer family；M34–M39 保留 ten-item reuse、overlap attenuation 和 following-transition propagation 三个两端点 families；S01 保留正式 recurrence sensitivity 记录。
- 删除 `not encoded`、内部路径、status、snake_case 和 figure-offset 说明；26 列裁切问题已消除；最终三张读者表分别为 4、7、7 列。

### S5 — Five-panel Fig.5 and independent Supplementary Fig. S3–S5 conclusions

**作者选择：确认同步。**

- Fig.5b 的 Holm-adjusted \(P\) 值明确归属于两个 overlap-specific attenuation endpoints；non-overlap/random controls 固定为零且只作描述。
- 主文以三个独立短句分别报告 S3 的 spatial selectivity、S4 的 cross-network/following-transition persistence 和 S5 的 network/variable-resolved recurrence，不与 Fig.5 共用同一括号结论。
- Statistics 段只增加 `overlap-attenuation`，其余锁定统计文字与 OMML 不变。
- S3–S5 legends 改为解释各自补充结论，不再写成 Fig.5 的 extension。
- S5b 使用 persisted S01 confirmatory record：mean 0.4890，95% CI 0.481787–0.496191，one-sided exact sign-flip，Holm-adjusted \(P=1.907\times10^{-5}\)。S5b artwork 只显示 network points，因此图像无需改变；重新导出的 PNG/PDF/SVG 哈希与原图完全一致。

## 当前构建与 QA

- 主文确认稿 SHA-256：`c3199d710812cbd6db60cdee560c6270b5dbec3e1a81ce9a2427c9b93a73239a`
- 补充材料确认稿 SHA-256：`44b10b6d221e8af25d1a99a48c4f2d70b7eba0595be00fb2bf2021bebedd0722`
- 主文 PDF：29 页，无空白页、无页面外文字；`word/document.xml` 中 65 个 OMML `oMath` elements 与修改前一致，display Equations 仅保留 (1)–(5)；6 条作者批注已按明确指示全部移除，当前 comments = 0，7 个 media 保持。
- 补充 PDF：24 页，无空白页、无页面外文字；`word/document.xml` 中 96 个 OMML `oMath` elements 与修改前一致，Supplementary Equations (S1)–(S17) 连续完整；7 个 media 保持。
- 补充表结构：Table S1 = 25 × 3；Table S2A = 7 × 4；Table S2B = 45 × 7；Table S2C = 45 × 7。
- 当前工作投稿包：`communications_biology_v6_1_reader_first_candidate_r2/`；Article/SI、Fig.1–Fig.7、Fig.5a–e Source Data、Supplementary Figs. S1–S7 Source Data 与 416 项 checksum 已核对通过，仍保持 `submission_ready=false`。
- 正式基线 `v6.docx`（`64b05e...`）、正式补充材料（`f87010...`）及两个副本均未改变。

## 部分完成、图件单独处理

### L1 — 层间 successor-state 机制全文联动

**作者决定：接受 Fig. 4g 之外的全部文字修改；Fig. 4g caption 与 artwork 暂不参与本轮，并作为同一图件包处理。**

已写入 `v6_1.docx`：

- Abstract：删除同层 self-rewrite，改为 processing 形成携带 history 的 downstream synaptic state。
- Introduction：问题段保持读者当前概念层级；研究概述自然表达 inherited STSP → processing → state formed downstream，不插入定义式 successor 段。
- Results roadmap：第二问改为 processing 如何改变被带入后续的 synaptic state。
- Fig. 4 Results 收束：采用相对层间关系，不在概念综合中重复具体 Layer 编号。
- Fig. 5 sequence 收束：`rewriting` 改为 `updating`，并写成 one transition → next-input processing → next state。
- Discussion：中心段保持 working-memory continuity 为主角；下一段恢复 maintenance → continuous processing 的领域意义。

继续单独处理：

- **L1.6 Fig. 4g caption + L1.7 artwork**：二者均保留现状，待图件方案确认后同批修改；当前不存在人为制造的 caption–artwork 不一致。

详细清单：`docs/paper/revisions/V6_1_DEFERRED_INTERLAYER_LINKAGE_20260814.md`

## 待确认

- L1.6 Fig. 4g caption + L1.7 artwork
- **本轮 Fig.6b order-identification integration（2026-08-16）**：主文 Results 句子、Fig.6 legend、Morphology Methods、Supplementary Table S2 的 M20 行和正式 Fig.1–Fig.7 artwork 已写入工作稿；待作者确认后再冻结哈希并提升投稿状态。
