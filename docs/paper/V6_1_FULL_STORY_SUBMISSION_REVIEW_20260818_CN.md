# V6.1 主文—补充材料联合科学审读与投稿裁决

**审读日期：** 2026-08-18  
**审读对象：** `v6_1.docx`、`supplementary_information_v6_1.docx` 及其中嵌入的 Fig.1–Fig.7、Supplementary Fig. S1–S7  
**审读方式：** 主文盲读 → 补充材料核查 → 当前持久化 Source Data/统计核对 → 外部文献新颖性核验 → 主文—补充材料联合闭环复读。  
**执行边界：** 未修改两个 DOCX；未重跑训练或大规模模拟。文中定量核验只使用当前持久化产物。当前 DOCX 内嵌七张主图的哈希已与 `results/paper_figures/outputs/fig1`–`fig7` 逐一核对一致；补充图也与当前登记来源一致。

---

## 1. 双层投稿裁决

### 1.1 Nature Communications

**裁决：核心故事尚未达到该刊门槛；当前版本不应投稿。**

这不是说模型内所有结果无效。相反，`继承状态 → 相同输入的历史条件化处理 → 下游 successor formation → successor reuse` 的模型内干预链有实质内容，Fig.4f、Fig.5a/c 是全稿最有价值的证据。但当前稿同时存在：

1. **投稿文件内部的硬性矛盾**：Fig.4g 图注与当前嵌图描述的是不同机制图；Fig.7d 正文宣称随 delay 单调减弱，而持久化数据明确非单调；Fig.2a 被称为 delayed-recall accuracy，但其来源是总体/当前 probe 分类表现而非延迟回忆旧项目。
2. **新颖性定位遗漏直接先例**：前馈 STP 短时记忆、隐状态与新输入相互作用、STP 多项目/顺序乃至 chunking 均已有直接或邻近先例；当前 Introduction 的 “remains unknown” 口径过宽。
3. **一般性不足**：全部主结论来自一个 MNIST、一个三层架构、一个 STSP 参数组和同一训练范式；没有任务、数据集、架构或参数层面的外部有效性包，也没有生物数据对齐。
4. **working-memory 解释仍有替代项**：大量“行为”端点实际是当前输入分类的 rescue/loss、样本残留导致的 probe 偏置或 partial-cue priming。现有结果证明短时历史状态影响后续处理，比证明一个有任务需求的 working-memory updating system 更直接。
5. **两个关键统计构造比文字结论更窄**：Fig.5d 的 passive centered-cosine displacement 由动力学与指标共同保证为零；Fig.7f 的 no-overlap cells 是端点构造的结构零。二者不能按普通经验对照或普通 2×2 interaction 解读。

**预期编辑风险：高概率 desk reject。** 即便先修正文字/图注，单一任务与缺少直接先例定位仍会使其更像一篇扎实的专业建模论文，而不是当前证据范围内具有足够广泛影响力的 Nature Communications 论文。

### 1.2 专业计算神经科学期刊

**裁决：完成关键补强后再投稿；核心模型内故事可成立。**

适合的定位是：**在一个固定长时权重、无 recurrent excitation 的分层脉冲网络中，利用可追踪的 presynaptic STSP 状态和选择性 state transfer，证明历史状态可以定向改变相同输入的处理并形成、复用下游 successor state。**

达到专业期刊可送审状态至少需要：

- 修正三项硬矛盾（Fig.4g、Fig.7d、Fig.2a/对应 protocol）；
- 重写 novelty framing 并补入直接先例；
- 把 Fig.5d、Fig.7f 的构造边界写到正文而不只藏在补充材料；
- 将“working memory”与“短时适应/priming/history-dependent computation”的区分变成可审查论证；
- 最好增加一个**最小但有诊断力**的跨任务或跨参数 robustness 包。仅再加更多 seeds 不解决一般性问题。

---

## 2. 一句话科学判断

**当前证据充分支持一个模型内、层特异、有限条件下的历史依赖 successor-state 机制；不足以支持其为 working-memory maintenance 与 organization 的一般机制，更不足以支持隐含的 chunking/capacity 框架。**

故事最强处是受控 state substitution；最弱处不是“数据量少”，而是**认知任务身份、指标构造和外部先例定位没有与强结论同步校准**。

---

## 3. 投稿前硬性阻塞项

### B0-1｜Fig.4g 当前 artwork 与图注不属于同一版本

- 当前 DOCX 内嵌 Fig.4 与正式 `fig4.png` 哈希一致。
- artwork 可见链为：`Inherited STSP → Selected firing → Downstream firing → Successor STSP`，当前 `Input` 箭头进入 selected-firing 阶段。
- 但主文 P039 图注仍写：先前输入留下 `S_k`、稀疏放电放大子集、**inter-input decay yields successor `S_{k+1}`**、later input 再选择该 successor，并说明“dot size”和“blue subset”。这些对象和编码已不在当前 artwork 中。
- 该旧图注还会把 successor 误读成同层状态经 decay 得到的新状态，破坏全文已冻结的层间方向。

**级别：所有期刊的提交阻塞项。** 应以当前 artwork 或当前 provenance caption 二选一同步，不能只做措辞润色。

### B0-2｜Fig.7d 正文给出与数据相反的单调解释

主文 P056 写：

> Across the load–delay grid, access weakened with longer retention delays ... functional expression attenuated at longer delays.

但 `panel_d_plot_data.csv` 明确非单调：

- `K=3`：0.018 → 0.060 → 0.260 → 0.337（100→800 ms，反而上升）；
- `K=10`：0.477 → 0.575 → 0.617 → 0.375（400 ms 达峰）；
- 800 ms 时各 K 收敛到约 0.34–0.375。

负的 `sequence length × delay` 交互表示**序列长度效应随 delay 改变/压缩**，不等于 access 随 delay 单调下降。当前 provenance 已明确写 “no monotonic claim”，但 DOCX 正文没有同步。

**级别：所有期刊的提交阻塞项。** 正文只能报告联合依赖和非单调格局；如要声称 delay 主效应，必须另有预先定义且可核验的主效应检验。

### B0-3｜Fig.2a 的科学对象被误写为 delayed recall

- Fig.2a 来源端点是 `overall_recall`/总体 assay accuracy；panel contract 将其角色定义为“建立网络任务可用”，不是机制证明。
- 其来源文件 `panel_b_baseline_metrics_by_network.csv` 只给每网络总体正确率，没有 delay 维度。
- 真正 sample–delay–probe 的 persisted run config 明确写：sample 与 probe 标签不同，主要端点是当前 probe accuracy，以及 static-minus-dynamic contrast。
- Supplementary Fig. S1 进一步显示 dynamic STSP 相对 static-frozen **降低当前 probe accuracy**，同时增加 sample-label bias；这证明旧状态干扰/偏置当前处理，不是旧项目被延迟回忆正确。

因此 P021 的 “delayed-recall accuracy remained high at 90.95%” 及 Fig.2 caption 的 “Recall accuracy” 均可能误导。

**级别：所有期刊的提交阻塞项。** 需要明确 Fig.2a 到底是 baseline classification、current-probe classification 还是另一个 assay；并在 Methods 给出 sample、delay、probe、目标标签、状态交换时刻和分母。

### B0-4｜核心 novelty gap 写得过宽且漏引直接先例

至少以下直接文献必须进入定位：

- Buonomano & Merzenich, *Science* (1995)：STP/慢状态与输入相互作用，把时间上下文转成空间响应；
- Buonomano & Maass, *Nat. Rev. Neurosci.* (2009)：明确提出 incoming stimuli 与包括 STP 在内的 hidden network state 相互作用产生 state-dependent computation；
- Hu et al., *PLoS Comput. Biol.* (2021)：训练**纯前馈、短时突触抑制**网络完成视觉 change-detection 短时记忆任务，并与小鼠神经/行为数据比较；
- Zhong, Katkov & Tsodyks, *Synaptic Theory of Chunking in Working Memory*（eLife Reviewed Preprint / arXiv v2）：直接提出 STP 驱动 WM chunking 与层级组织。

当前稿可保留的新颖点应是**特定组合与干预闭环**，不是 STSP memory、feedforward memory、state-dependent input processing、multi-item/order 或 chunking 任一单独概念。

### B0-5｜Fig.5d 的 passive 对照是指标上解析为零，不是普通经验对照

对每个 STSP 变量，零输入被动演化可写成：

\[
z_i(t+T)=b+[z_i(t)-b]c,\qquad c>0.
\]

中心化后：

\[
z_i(t+T)-\bar z(t+T)=c[z_i(t)-\bar z(t)].
\]

所以同一变量的 centered-cosine distance 必然为 0。Methods 又分别计算 `u` 和 `x` 的 centered-cosine distance 后取均值，因此 Fig.5d 的 no-input 线由数学构造而不是数据不确定性决定为零。

M31 实际证明的是：**每次输入都改变了状态的中心化方向/空间形态**；它不比较输入更新与真实被动衰减的幅度，也不证明每一阶段完整重现 Fig.4 的 history-conditioned causal motif。

该结果仍有价值，但 “exceeded passive evolution” 必须明确为 metric-specific morphology divergence。若要比较状态变化量，应使用对被动幅度变化敏感的补充指标；这可优先从既有边界状态派生，不必先重跑网络。

### B0-6｜Fig.7f 不是普通 2×2 interaction

Methods 与 caption 已承认 no-overlap cells 是 “structural zeros of the endpoint construction”。因此 15.997 pp interaction 基本等价于 overlap 条件中 high-minus-low STSP 差异；no-overlap 一侧不是可观测的同类端点。

- 可以说：在定义的 overlapping pathways 中，high-STSP 组的 early-firing change 大于 low-STSP 组；
- 不宜把该数值当成完整 factorial evidence，或单独据此把 overlap 称为唯一 circuit gate；
- Supplementary Fig. S7 的 spatial-score shuffle 说明 observed score arrangement 不是随机，但不能把 structural-zero cell 变成经验测量。

Fig.7e 的 targeted removal 是更直接的贡献证据，应承担主要因果结论；Fig.7f 应降为构造受限的空间一致性证据。

---

## 4. 故事完整性、充分性、新颖性和逻辑可靠性

### 4.1 完整性：**形式上完整，认知层级尚未闭合**

七图 DAG 清楚：

`Fig.1 instantiate → Fig.2 inherit → Fig.3 condition → Fig.4 form → Fig.5 reuse/recur → Fig.6/7 parallel outcomes`。

主文没有把 Fig.6 structure 与 Fig.7 function 错写成直接因果链，这是优点。真正未闭合的是：

- 模型为何不只是 stimulus-specific adaptation / sequential priming；
- 历史状态在一个**需要记住过去项目**的行为目标中提供什么，而不只是改变当前分类；
- 从单一 MNIST 参数点到“working-memory maintenance and organization”的外推边界。

### 4.2 证据充分性：**模型内单步机制强，递归与认知解释中等**

| 论证节点 | 证据判断 | 主要理由 |
|---|---|---|
| firing 后存在 STSP state | 充分 | 动力学与状态解码一致；Fig.1–2 清楚 |
| state 影响后续处理 | 基本充分 | state shuffle 有 donor-directed shift；S1 有 paired flux，但主图错误池分母不相同 |
| identical input 受 history 条件化 | 充分但范围窄 | exact-B 与 aligned/mismatched 证明 label/history dependence；也兼容 repetition/priming |
| 下游 successor formation | 强 | selective Layer-1 `u/x` transfer、fast-state equalization、Layer-2 donor shift 与 early Layer-3 extension |
| successor reuse | 强于 recurrence | post-B Layer-2 transfer 对相同下一输入和新 successor 有效；跨 K 复现 |
| natural recurrence | 中等 | 输入确实反复改变形态，但 passive centered-cosine 为解析零，未逐边界证明完整因果 motif |
| terminal multi-component/order | 中等至强 | order identification 很强；NNLS/相似性成分指标仍受模板共线性和构造影响 |
| conditional access | 中等至强 | partial cue、matched cue、removal 支持条件表达；Fig.7d/f 解释需收紧 |
| biological/general mechanism | 不充分 | 无生物拟合、无跨任务/跨架构/参数检验 |

### 4.3 新颖性：**组合新颖性可信，原子主张不新**

可保留的贡献：

> 在层特异、前馈脉冲网络中，对 pre-input inherited `u/x`、identical-current-input processing 和 downstream post-input `u/x` 做连续追踪，并通过 selective state substitution 证明 donor history 可以定向改变 downstream successor，且该 successor 可被下一输入复用并影响再下一 transition。

不能作为新颖性主张的内容：

- STSP 可 activity-silent maintenance；
- hidden state 可被 probe 读出；
- 相同输入响应依赖过去状态；
- 前馈 STP 网络可实现短时记忆；
- STP 可表达多项目、顺序、serial-position 或 chunking；
- 动态代码可在保持任务信息时 morph。

### 4.4 逻辑可靠性：**主 DAG 可靠，五处桥梁需降级或重建**

1. Fig.2a 任务可用性不能被当成 delayed recall。
2. Fig.2d 的两根柱使用各条件自己的 error pool，不能仅凭组成互换声称 paired reciprocal change；paired donor flux 在 S1c，主文应把证据归属写准。
3. Fig.5d 只证明反复 input-associated morphology change，不证明每个边界完整复现 history-conditioned successor formation。
4. Fig.7f 结构零限制 interaction 解释。
5. Discussion 的 chunking/capacity 段没有 chunking 操作、边界 cue、压缩指标、容量收益或层级单元证据。

---

## 5. 主文逐段读者路径记录

说明：标题、作者单位、纯 heading、图片占位符和参考文献条目不重复列入；每个有科学功能的正文/图注段均记录。`稳`=桥梁和边界基本合格；`修`=需修正；`阻`=投稿阻塞。

### 5.1 Abstract 与 Introduction

| 段落 | 读者预期 | 本段认知更新 | 逻辑桥梁 | 边界/问题 | 判定 |
|---|---|---|---|---|---|
| P006 Abstract | 问题、方法、关键发现、边界 | 给出 inherited STSP→downstream state→later input 的全链 | maintenance→evolution | 未写明单一 MNIST/模型内范围；“bridge”对 Nature 稍宽 | 修 |
| P008 | 从 WM 维持进入动态演化问题 | 说明新输入会重组表征 | retention→state evolution | 合理 | 稳 |
| P009 | 界定 persistent 与 silent 两类机制及缺口 | 把缺口落在 STSP 的 input-associated reorganization | 已知 maintenance→未知 updating | “remains unknown” 过宽；缺 Buonomano/Maass、Hu 直接先例 | 阻 |
| P010 | 提出可检验机制问题 | 定义 inherited history 影响新输入并被带入后继状态 | gap→hypothesis | 应限定“layered feedforward SNN 中的 causal successor formation” | 修 |
| P011 | 预览结果和贡献 | 概述 recurrence、organization、selective later influence | hypothesis→results | “working-memory representations” 需模型内限定 | 修 |

### 5.2 Results 总路线与 Fig.1–Fig.2

| 段落 | 读者预期 | 本段认知更新 | 逻辑桥梁 | 边界/问题 | 判定 |
|---|---|---|---|---|---|
| P013 | 获得全 Results 问题序列 | 四问与七图 DAG 对齐 | roadmap | 清楚 | 稳 |
| P015 | 知道网络和训练对象 | 三层、前馈、MNIST、固定长期权重 | model→mechanism isolation | 应提醒 post-training 才启用 STSP | 修 |
| P016 | 知道 STSP 状态与传输公式 | 定义 `u`、`x`、`G_STSP` 和固定 `W` | architecture→state variable | state 是 presynaptic-site shared gain，不是每条 connection 独立；局限未点明 | 修 |
| P018 Fig.1 legend | 判断示意是否有独立数据 | 参数、动力学、无推断单位均交代 | mechanism illustration→empirical assays | 良好 | 稳 |
| P020 | 进入 persistence + function 双问题 | 预告两类证据 | dynamics→inheritance | 良好 | 稳 |
| P021 | 先看任务可用、silence、decodability | 高 accuracy、延迟无 firing、`u/x` 可解码 | task→silence→content | 90.95% 被误称 delayed recall；真正任务对象不清 | 阻 |
| P022 | 从可解码进入 causal function | state exchange 导致 original↓、donor↑；S1 控制 | correlation→intervention | Fig.2d error pools不同；主文应以 S1 paired flux 为主要 paired 证据；protocol 不足 | 阻 |
| P023 | 收束 inherited-state 前提 | 旧刺激处理与后续输出由 STSP 相连 | Fig.2→Fig.3 | “without sustained firing” 边界合适 | 稳 |
| P025 Fig.2 legend | 独立理解四面板 | 给出层、delay、error composition、n | figure→claim | a 的 “Recall accuracy” 不准；d 未说明条件特异 error pool | 阻 |

### 5.3 Fig.3–Fig.4：identical input 与 successor formation

| 段落 | 读者预期 | 本段认知更新 | 逻辑桥梁 | 边界/问题 | 判定 |
|---|---|---|---|---|---|
| P027 | 从“状态有功能”进入“相同输入是否被 history 调制” | 明确 behavioral + synaptic update 双端点 | inherit→condition | 良好 | 稳 |
| P028 | 看 behavioral divergence | aligned 提高 rescue、降低 loss | identical-B→outcome | aligned 是 final-label match；结论只能到 match-specific history effect，也兼容 repetition priming | 修 |
| P029 | 区分 shared input organization 与 residual | common cosine 高且 residual 非零、changed events 富集 | behavior→state update decomposition | thresholds 0.5/0.05 的科学依据未解释；极窄 CI 表示 seed robustness而非普适性 | 修 |
| P030 | 收束 Fig.3 | 相同输入主导共同组织，history 选择性调制 | condition→mechanism | 边界合适 | 稳 |
| P032 Fig.3 legend | 恢复 exact-input protocol 和统计 | 图注定义主要端点 | artwork→text | panel c 的 criterion 较抽象；需在 caption 说明 threshold provenance | 修 |
| P034 | 提出 history 如何进入 successor | 将问题落在 overlap pathway 与 downstream state | condition→form | 良好 | 稳 |
| P035 | 看 localization 和 pathway effects | reset/attenuation、早期 spikes、Layer-2 update 都指向 overlap | observation→local intervention | “localized” 需保留 tested masks/conditions；Fig.4e 是 update opportunity，不是 frozen mutation | 修 |
| P036 | 看 selective transfer 是否定向 successor | 只换 Layer-1 `u/x`，Layer-2 successor 移向 donor | localization→causal sufficiency | 全稿最强证据；“causally redirect” 在模型内成立，不支持 necessity/uniqueness | 稳 |
| P037 | 形成层间机制单元 | inherited state + current input → downstream successor | form→reuse | 合理 | 稳 |
| P039 Fig.4 legend | 独立恢复七面板和概念链 | a–f 定量、g 综合 | figure→mechanism | g 图注与 artwork 完全不同版本；f 未说明 20,000 comparisons 的 descriptive 层级 | 阻 |

### 5.4 Fig.5：reuse 与 recurrence

| 段落 | 读者预期 | 本段认知更新 | 逻辑桥梁 | 边界/问题 | 判定 |
|---|---|---|---|---|---|
| P041 | 从一次 successor 进入多次输入 | 问 successor 是否成为下一次 inherited state | form→reuse | 清楚 | 稳 |
| P042 | 获得跨深度、entry route、下一 transition 三证据 | K=1/5/10 transfer；overlap reset；C→D 无二次 transfer | reuse→selective route→propagation | 一段承载三实验，认知负荷高；零 controls 由 construction 固定；“removed 96.9%”易被读成 complete mediation | 修 |
| P043 | 区分 transplant 与自然序列 | observed displacement>passive；K5 rescue↓ loss↑ | intervention→unmanipulated sequence | passive centered-cosine 解析为0；behavior change可为干扰累积，不直接证明 successor motif | 阻 |
| P044 | 收束为 iterative evolution | 声称 reuse 是单步到迭代的桥 | reuse+recurrence→core claim | 可保留，但应写成“supports”并限定 Fig.4 motif 未逐边界重演 | 修 |
| P046 Fig.5 legend | 恢复五面板 | 明示各 endpoint、bootstrap、sign-flip | figure→claim | b 的 thin zero caps 容易被当经验零；d 未说明 passive=metric-invariant zero | 修 |

### 5.5 Fig.6–Fig.7：并列的 structure 与 function

| 段落 | 读者预期 | 本段认知更新 | 逻辑桥梁 | 边界/问题 | 判定 |
|---|---|---|---|---|---|
| P048 | 从迭代进入 terminal organization | 提出 constituents + experienced configuration | recurrence→structure branch | 合理 | 稳 |
| P049 | 看二项目成分与固定集合的 order | 对 A/B 模板都高相似；六阶 order 99.51% | content→configuration | 仅“对两模板都相似”不能排除模板共线/共同形态；order identification 是更强证据 | 修 |
| P050 | 扩展到 K=10 和 delay | `N_eff`增加、latest share下降、matched>deranged | short→long history | `N_eff` 是分解表达量，不是容量/可访问项目数；措辞基本守界 | 稳 |
| P051 | structure branch 收束 | successive inputs 未抹去 internal structure | evidence→claim | “integrate”可接受但限于 tested decomposition/morphology | 稳 |
| P053 Fig.6 legend | 独立识别六面板 | 明示 Layer-2/Layer-1不同测量尺度 | figure→claim | b artwork无六个 order tick labels；d 的50%线不如 `1/K` uniform reference有解释力 | 修 |
| P055 | 明确 Fig.6/7 为并列概念 | structure=arrangement，function=expression | structure branch→function branch | 很好，避免伪因果 | 稳 |
| P056 | 看 partial cue、serial position、cue specificity、load×delay | 多项 access 端点均为正 | cue→load/delay | 最后两句错误声称 longer delay 单调削弱；全段过长 | 阻 |
| P057 | 看 pathway contribution | targeted removal + spatial endpoint | behavior/readout→circuit route | e 支持 targeted contribution；f 是 structural-zero 构造，不能承担普通 interaction/唯一 route | 阻 |
| P058 | function branch 收束 | content match 约束 readout，overlap 集中 early effect | function→Discussion | 在“tested conditions”内可保留 | 修 |
| P060 Fig.7 legend | 恢复六面板与统计 | 已明确 no-overlap structural zeros | figure→claim | legend 比 Results 更准确；d 应明确 non-monotonic joint dependence | 修 |

### 5.6 Discussion

| 段落 | 读者预期 | 本段认知更新 | 逻辑桥梁 | 边界/问题 | 判定 |
|---|---|---|---|---|---|
| P062 | 回答中心问题 | continuity 可位于 successive-state relations | results→concept | 作为模型内解释合理 | 稳 |
| P063 | 与 maintenance 文献对话 | 从静态保存扩到连续处理 | contribution→field | 缺 state-dependent computation 与 feedforward STP 直接先例，导致“extend”显得过宽 | 阻 |
| P064 | 讨论 organization/capacity 意义 | 把 multi-component + partial-cue 连接到 chunking | structure/function→capacity | 无 chunk cue、压缩、容量收益或 hierarchy；且存在直接 STP chunking reviewed preprint | 阻 |
| P065 | 处理 silent vs persistent debate | 改问各机制控制 state evolution 的哪部分 | specific mechanism→broader framework | 是合理理论展望，但需标为 hypothesis | 稳 |
| P066 | 给出模型边界 | 明确 sufficiency、非 necessity/uniqueness/prevalence | framework→limitations | 很好；还应加入单任务/单参数、sitewise STSP、post-training activation | 修 |
| P067 | 给出生物检验方案 | 提议 history-controlled probes 与 targeted perturbations | model→experiment | “conserved relations” 无跨系统证据；改为“structured relations”更稳 | 修 |

### 5.7 Methods、availability 与 reproducibility

| 段落 | 读者预期/更新 | 桥梁 | 边界/问题 | 判定 |
|---|---|---|---|---|
| P070 | 输入编码、训练/测试 split | dataset→network | 清楚；无 validation split，应说明模型选择/分析开发是否看过 test set | 修 |
| P071 | 三层结构、无 recurrent excitation、决策规则 | architecture→dynamics | “无 recurrent excitation”准确；仍有 lateral inhibition | 稳 |
| P073–P078 | LIF、conductance、threshold、top-k、inhibition | neuron→competition | 复现性较好；最终稿需视觉确认 OMML 分数 `/τ` 和 `/C_m` 没被纯文本提取丢失 | 稳 |
| P079–P085 | `u/x` recovery/update 和 `G_STSP` | spike→silent state | 定义完整；需强调所有 outgoing weights 是否共享同一 presynaptic-site gain | 修 |
| P087 | 分层训练且训练时禁用 STSP | training→assay | 这是重要设计决定；主文 Results 初次介绍时也应出现 | 修 |
| P088 | 固定长期权重、`1/U` rescale、protocol-specific dynamics | training→fixed circuit | 清楚；需要参数/替代 rescaling robustness | 修 |
| P089 | 后训练 test accuracy | model quality | 与 Fig.2a 的 90.945% 是不同端点，应解释差异 | 修 |
| P091 | state formation、采样、ridge decoder | state→measurement | decoder 训练/测试量在 SI；主文基本足够 | 稳 |
| P092 | full/STSP-only/selective restoration | measurement→intervention | 中心方法清楚；Fig.2 shuffle 与 Fig.4 selective transfer 的 fast-state treatment应分别写明 | 修 |
| P093 | static/passive/no-memory 三 controls | intervention→counterfactual | 定义好，但 passive metric invariance未披露 | 修 |
| P095 | exact-input A/C/B | inherited history→same input | outcome-blind branch selection是优点；需给试验/anchor数量 | 修 |
| P096 | `T=L+Γ` decomposition | same input→common/residual | 代数闭合；threshold来源不明 | 修 |
| P097 | rescue/loss opportunity sets | state→behavior | 分母区分清楚 | 稳 |
| P098 | overlap groups、event classes、Layer-1 transfer | localization→successor | 足够概览；详细规则在SI | 稳 |
| P100 | successor transplant、K、overlap reset、C→D propagation | formation→reuse | 方法充分；一段过密但可复现 | 修 |
| P101 | progressive recurrence、relation-balanced outcomes | reuse→natural sequence | 未指出 centered cosine使 passive分支为0；应修 | 阻 |
| P103 | two-item similarity与order identification | recurrence→structure | order protocol充分；候选模板和leave-one-set-out细节可再明确 | 修 |
| P104 | NNLS、`N_eff`、latest share | structure→summary | fallback uniform在SI；需要模板共线性诊断 | 修 |
| P105 | coefficient-free Layer-1 morphology | fitted→independent scale | 明确不同尺度，是优点 | 稳 |
| P107 | state restoration + same degraded cue + AUC | structure→function | 清楚；这是 cue-supported access，不是 free recall | 稳 |
| P108 | matched/same-label novel/unseen cues | access→specificity | silent repetitions进分母，合理 | 稳 |
| P109 | accessible/rescued定义和交互 | specificity→load×delay | 交互只能解释 joint dependence；不得导出单调 delay | 阻 |
| P110 | overlap endpoint和matched removal | access→pathway | no-overlap structural-zero性质应在本段明示，不只放SI | 阻 |
| P112 | inferential unit、tests、CI、adjustment | analysis→reporting | family member lists未完整重建；one-sided“prespecified”时间点不明 | 修 |
| P114 | Source Data范围 | reproducibility | 良好 | 稳 |
| P116 | code commit和环境 | reproducibility | 当前公开baseline与本地版本差异按既定边界属工程项；投稿前仍应保证可运行入口 | 修 |
| P118–P151 | 文献覆盖 | claims→prior work | 漏 Buonomano/Merzenich 1995、Buonomano/Maass 2009、Hu 2021、Zhong等；31是reviewed preprint，证据等级标注正确 | 阻 |
| P153–P157 | funding/contribution/competing interests | disclosure | 形式完整；政策表格另核 | 稳 |

---

## 6. 主图审稿式 QA

| 图 | artwork 独立可恢复的结论 | 主要问题 | 评级 |
|---|---|---|---|
| Fig.1 | 架构、一次 spike 后 `u/x` 与 `ux` 支持持续 | a 同时出现训练标签 STDP/R-STDP 与 post-training assay，读者可能误以为实验期仍学习；应在 caption 强调 fixed after training | minor |
| Fig.2 | 有任务表现、delay firing消失、`u/x`可解码、shuffle偏向donor | a 不是 delayed recall；d 没显示 protocol与条件特异 error pools；主图无法知道 donor state何时换入 | major |
| Fig.3 | 两 histories 后相同 B；behavior divergence；common+residual coexist | 图面最清楚；但 aligned 主要是 label match，未排除 ordinary repetition/priming解释 | moderate |
| Fig.4 | overlap localization、early events、Layer-2 update、donor transfer、inter-layer cartoon | g caption完全错版；f 是20网络等权后的20,000 comparison histogram，当前 legend没有阻止伪重复印象 | critical |
| Fig.5 | reuse跨K、overlap entry、下一transition传播、反复input change、behavior shift | b controls是构造零；d no-input是指标解析零；“Effect removed”近97%易被读成necessity/mediation | major |
| Fig.6 | constituents/order、`N_eff`、latest share、morphology across K×delay | b 无order tick labels；a 高相似缺 unrelated/template-similarity control；d 应优先画 `1/K` uniform reference而非50% | moderate |
| Fig.7 | partial-cue access、position access、cue specificity、load×delay、removal、overlap endpoint | d 与正文相反；f no-overlap structural zeros在 artwork 不可见；b/a是cue-supported access不是无cue recall | critical |

### Fig.4f 的独立单位说明

Panel f 的柱状分布不是 20 个独立网络点。每个网络先把自身 1,000 个有效 comparison rows 归一化为 100%，再对 20 个网络的直方图等权平均；竖线/菱形才对应 network-level mean/CI。当前 provenance caption 已写清，但 DOCX legend 没有。投稿图注必须恢复该句。

---

## 7. 补充材料核查

补充材料总体优点是：控制操作、state restoration、opportunity sets、sequence aggregation、网络级统计单位和主要数值均比主文完整。主要问题不是缺数据，而是主文没有总是准确继承这些边界。

| 补充模块 | 它真正保护的主张 | 核查结果与风险 |
|---|---|---|
| Supplementary Methods P007–P030 | encoding、training、decoder、restoration、controls | 参数详尽；但 Fig.2 sample–delay–probe 的认知任务目标仍不够直观，raw config 比稿件更清楚 |
| P032–P042 | exact-B decomposition、behavior、overlap、transfer | 操作定义好；0.5/0.05 thresholds仍缺独立依据；Layer-1 transfer因果身份较强 |
| P044–P046 | successor reuse、overlap intervention、natural recurrence | pairing/anchor清楚；P046没有披露 passive centered-cosine解析零 |
| P048–P054 | NNLS与coefficient-free morphology | 明确两测量尺度不可互换，是优点；相似性/NNLS受模板共线性风险仍需诊断 |
| P056–P063 | partial cue、specificity、rescued fraction、overlap/removal | 复现细节好；P062同时“无probe pixel排除”与“no-overlap structural zeros”使普通factorial读法不成立 |
| P065 | statistics | 明确无power、normality、outlier rule；诚实但需说明t检验稳健性与seed推断范围 |
| Fig. S1 | delay decline、sample bias、paired donor flux、donor calibration | 很关键；同时显示 dynamic history可损害当前probe accuracy，应在主文解释是memory influence/interference而非recall success |
| Fig. S2 | exact-B、L1-only、fast-state、L2/L3 validity、untouched-19 | 全稿最强支持图之一；20网络与1000 comparison audit在同一panel，需防独立单位混淆 |
| Fig. S3 | window/rank/distance robustness、winner fate、overlap attenuation | 支持local effect；e/f的零controls仍为construction，不是检测到“无效应”的随机样本 |
| Fig. S4 | K=1/5/10与following transition一致性 | 良好；不支持cross-depth trend，legend已守界 |
| Fig. S5 | network×transition、minimum、u/x变化 | 证明每个stage有input-driven morphology change；不修复passive metric解析零问题 |
| Fig. S6 | NNLS/similarity、Δg、Moran、matched–deranged | coefficient-free topology和matched–deranged是较强补充；similarity-based `N_eff=K` 近乎定义性，不应视为完全独立验证 |
| Fig. S7 | exact matching、window/definition/coverage、score shuffle | 有效排除部分匹配和score arrangement替代解释；不能把structural-zero interaction升级为经验2×2 |
| Table S1 | 参数 | 基本完整；应突出STSP训练时关闭、post-training开启和shared presynaptic-site gain |
| Table S2 | endpoint与统计 | 数字与主文大体一致；但正文自己承认未从aggregate重建family member lists，multiplicity透明度不完整 |

---

## 8. 统计与定量可靠性

### 8.1 做得好的部分

- 明确以 20 个独立训练网络为 inferential unit；低层 observation 先在网络内聚合。
- 大多数端点报告 estimate、CI、null、test 和 adjusted P。
- Rescue/Loss 使用不同 opportunity sets，并在方法和图注中反复声明。
- Fig.5 的 sign-flip tests、network bootstrap 和 source data 基本可追踪。
- 无响应/静默 trial 保留在 unconditional denominator，避免只分析成功 trial。

### 8.2 必须收紧的部分

1. **推断对象**：20 seeds 只支持对该 architecture、dataset、training pipeline 的初始化/训练随机性稳健；不构成20个独立生物系统或20个独立数据集。
2. **极大 t 值**：例如 M09 `t≈250,302`、M04 `t≈3,029` 不是算错，但说明网络均值几乎确定；它们不增加跨任务/跨模型一般性。主文应少依赖极端 P 值，多报告效应与边界。
3. **multiplicity family**：Table S2 写明 family lists 未由aggregate重建。对可复核统计，family名称、所有成员、预设时间和调整算法应机器可读且随稿提供。
4. **one-sided confirmatory 语言**：需要说明 direction/family 是在何时、相对于哪些数据预设；内部“predeclared”不自动等于独立预注册。
5. **constructed controls**：Fig.5b、Fig.5d、Fig.7f 的零值不是普通抽样估计，不能配合极小 P 值制造比实际更强的经验确定性。
6. **Fig.2d composition**：两个条件的 error-trial counts 不同，应避免把两柱的组成差写成同一paired pool上的reciprocal change。

### 8.3 数字一致性结论

除 Fig.7d 的文字方向和 Fig.2a 的端点命名外，抽查的主要数值、CI、P、网络数与当前 Source Data/Table S2 一致。Fig.4g 是语义/版本不一致，不是数值不一致。

---

## 9. 外部文献与新颖性核验

### 9.1 证据等级

- **A：同行评议、直接先例**——与本稿至少共享 architecture/mechanism/task 中两个核心维度；
- **B：同行评议、邻近先例**——共享核心概念但没有本稿完整干预链；
- **C：Reviewed Preprint/高相关预印本**——必须与同行评议证据分开。

### 9.2 风险导向文献表

| 文献 | 等级 | 已引？ | 对本稿 novelty 的含义 |
|---|---:|---:|---|
| Mongillo, Barak & Tsodyks, “Synaptic theory of working memory”, *Science* 2008, DOI: [10.1126/science.1150769](https://doi.org/10.1126/science.1150769) | B | 是 | STSP activity-silent WM 的奠基先例；maintenance本身不新 |
| Buonomano & Merzenich, “Temporal information transformed into a spatial code…”, *Science* 1995, DOI: [10.1126/science.7863330](https://doi.org/10.1126/science.7863330) | B | 否 | hidden short-term states与后续输入相互作用、产生不同空间响应的早期直接概念先例 |
| Buonomano & Maass, “State-dependent computations…”, *Nat. Rev. Neurosci.* 2009, DOI: [10.1038/nrn2558](https://doi.org/10.1038/nrn2558) | B | 否 | 明确把STP列为hidden network state，并把计算定义为incoming stimulus×internal state；Introduction核心gap不能忽略 |
| Pals et al., “A functional spiking-neuron model of activity-silent working memory…”, *PLoS Comput. Biol.* 2020, DOI: [10.1371/journal.pcbi.1007936](https://doi.org/10.1371/journal.pcbi.1007936) | A/B | 是 | spiking、STSP、activity-silent、functional behavior与多项目probe均已有；本稿需突出successor tracing而非功能STSP本身 |
| Hu et al., “Adaptation supports short-term memory in a visual change detection task”, *PLoS Comput. Biol.* 2021, DOI: [10.1371/journal.pcbi.1009246](https://doi.org/10.1371/journal.pcbi.1009246) | A | 否 | **纯前馈**STPNet、视觉序列、短时记忆、downstream decision 且与小鼠数据对齐；是当前最大漏引和最直接feedforward先例 |
| Masse et al., “Circuit mechanisms for the maintenance and manipulation…”, *Nat. Neurosci.* 2019, DOI: [10.1038/s41593-019-0414-3](https://doi.org/10.1038/s41593-019-0414-3) | B | 是 | STSP-RNN maintenance/manipulation；显示active与silent机制按任务需求协作 |
| Kozachkov et al., “Robust and brain-like working memory through STSP”, *PLoS Comput. Biol.* 2022, DOI: [10.1371/journal.pcbi.1010776](https://doi.org/10.1371/journal.pcbi.1010776) | B | 是 | STSP网络可与NHP数据对齐且具有robustness；提高了Nature级建模论文对empirical grounding的参照 |
| Mi, Katkov & Tsodyks, “Synaptic correlates of working memory capacity”, *Neuron* 2017, DOI: [10.1016/j.neuron.2016.12.004](https://doi.org/10.1016/j.neuron.2016.12.004) | B | 是 | 多项目/capacity不是新领域 |
| Zhou et al., “The synaptic correlates of serial position effects…”, *Front. Comput. Neurosci.* 2024, DOI: [10.3389/fncom.2024.1430244](https://doi.org/10.3389/fncom.2024.1430244) | A/B | 是 | STP与顺序、load、delay和serial-position已有直接同行评议模型 |
| Mongillo & Tsodyks, “Synaptic encoding of time in working memory”, eLife Reviewed Preprint 107005 v2 (2026), DOI: [10.7554/eLife.107005.2](https://doi.org/10.7554/eLife.107005.2) | C | 是且标注正确 | 时间/顺序的最新邻近证据；不是最终同行评议VOR |
| Zhong, Katkov & Tsodyks, “Synaptic Theory of Chunking in Working Memory”, eLife Reviewed Preprint 109538 / arXiv: [2408.07637v2](https://arxiv.org/html/2408.07637v2) | C | 否 | 直接STP chunking、hierarchy、capacity；P064必须引用并明确本稿没有真正chunking assay |
| Parthasarathy et al., code morphing, *Nat. Neurosci.* 2017 / *Nat. Commun.* 2019 | B | 是 | 动态代码/稳定任务信息已有；本稿的新意只能是STSP层间successor机制 |

### 9.3 新颖性最终判断

未发现与本稿**完全相同**的“layer-specific inherited `u/x` substitution → identical current input → downstream `u/x` successor → next-input reuse”完整证据链。因此组合新颖性仍可信。

但当前版本没有资格暗示以下“首次”：STSP支持silent WM、前馈STP支持短时记忆、hidden state调制相同输入、多项目/顺序组织、chunking。建议把 Introduction 的缺口改成一个可被现有证据真正填补的窄问句，并在 Discussion 逐项与 Hu 2021、Buonomano/Maass 2009 和最新chunking工作区分。

---

## 10. 主要替代解释是否被排除

| 替代解释 | 当前排除程度 | 还缺什么 |
|---|---|---|
| 持续放电维持 | 较好 | 50-ms bins显示delay期无spike；仍只适用于该模型 |
| fast state而非STSP | Fig.4/S2较好；Fig.2一般 | Fig.2 shuffle的fast reset/restore需更透明 |
| donor机会不均 | S1较好 | 主文应把paired calibration证据放到正确位置 |
| 任意扰动都改变output | overlap/non-overlap/random和own-state sham部分排除 | Fig.5b controls构造零，需给真正可影响同一路径的matched control |
| current input完全被history取代 | 已排除 | Fig.3 common component 0.91支持input主导 |
| history只产生无结构残差 | 部分排除 | changed-event enrichment和order/morphology支持结构 |
| 普通repetition priming/adaptation | **未充分排除** | 需要在非label-repeat、matched visual similarity、任务相关/无关history条件下比较 |
| generic multiplicative gain | **未充分排除** | 与无STSP的匹配gain/adaptation baseline、替代STP dynamics或参数扰动比较 |
| template共线导致multi-component | 部分排除 | unrelated-template similarity、condition number、recovery simulation、cross-validated component identification |
| passive decay解释recurrence | 当前指标不敏感 | 加入Euclidean/energy-sensitive passive comparison |
| overlap effect只是端点定义 | Fig.7e部分排除，Fig.7f未排除 | 让no-overlap有可观测同类端点，或停止称普通interaction |
| chunking/capacity | 未检验 | 需要chunk boundaries、compression、capacity benefit、hierarchical retrieval；否则删除/降为远期假说 |

---

## 11. 最小可行补强方案

### 11.1 不新增模拟即可完成的提交修复

1. 同步 Fig.4g caption 与当前 artwork；删除旧 `S_k`/decay/later-input/dot-size 叙述。
2. 改写 Fig.7d 为 non-monotonic joint dependence，删除“longer delay attenuated access”。
3. 纠正 Fig.2a 名称，并完整写出 Fig.2 sample–delay–probe protocol、目标和state exchange。
4. 在 Fig.2d 明确两柱为各条件error pool；把paired donor flux归于S1c。
5. 在 Fig.4f 图注写明 1,000 comparisons/network、网络内归一化、20网络等权和descriptive histogram。
6. 在 Fig.5d Methods/Results 写明 centered-cosine 对passive affine recovery不敏感；收窄 recurrence 结论。
7. 将 Fig.7f 改称 overlap-restricted high–low STSP contrast，或至少把 structural-zero限制写进Results。
8. 删除或大幅降级 P064 chunking/capacity 段，并补引直接chunking reviewed preprint。
9. 补 Hu 2021、Buonomano & Merzenich 1995、Buonomano & Maass 2009；重写“未知”与“extend”。
10. 把 multiplicity family完整成员列表放入可机读Source Data。

### 11.2 专业期刊前建议保留的一个最小新增证据包

优先级高于“再加更多seed”：

- **跨任务或跨数据的一次复制**：选择一个真正要求过去项目的 delayed comparison/sequence query，或至少一个非MNIST视觉任务；
- **一个参数/机制诊断网格**：`τ_F/τ_D/U` 的小型预设网格，加一个matched static/adaptation baseline；
- **一个认知替代解释控制**：history是否task-relevant × label repetition是否存在的正交设计。

只需一个预先声明、规模有限、使用同一DAG的补强包；不需要重建全部七图。

### 11.3 Nature Communications 额外门槛

除上述外，至少还需要其一：

- 与真实神经/行为数据的定量对齐和可证伪预测；或
- 跨两类任务/架构的一般性，并显示 successor mechanism 相对现有 STP/adaptation 模型带来不可替代的新解释；或
- 一个能够改变领域理解的理论结果，而非单一系统中的丰富表征分析。

---

## 12. 最终联合闭环

### 已闭合

- 主问题序列和七图DAG可恢复；
- layer-specific successor direction在Results主体中基本正确；
- Fig.6 structure 与 Fig.7 conditional function 被正确设为并列分支；
- 20-network inferential unit大体守住；
- 选择性transfer、own-state sham、fast-state equalization和untouched-19 sensitivity为因果链提供了可信核心；
- 主数值除已列问题外与持久化产物一致。

### 未闭合

- 当前投稿文件仍有 Fig.4g 版本错配、Fig.7d错误解释、Fig.2a任务误命名三项硬伤；
- working memory 与 adaptation/priming 的诊断边界不足；
- recurrence与interaction使用了结构性零，但结论语气仍像普通经验对照；
- novelty narrative未覆盖最直接文献；
- 单一MNIST/架构/参数组不足以支撑Nature Communications层面的广泛机制主张；
- chunking/capacity段跨越了现有证据。

## 结论

**模型内机制故事不是空的，也不是需要推倒重来；但当前稿件把一个有价值的“history-conditioned feedforward STSP successor”结果包装得比证据更普遍，并夹带三处可直接证伪的文件级不一致。**

- **Nature Communications：** 当前不投；先解决核心定位与一般性，否则即使清除文字错误仍很可能编辑拒稿。
- **专业计算神经科学期刊：** 完成上述关键补强后可形成有竞争力的投稿；最应保护的是 selective state-transfer 的层间 successor 证据，最应删除的是无直接实验支撑的 chunking/capacity 升级。
