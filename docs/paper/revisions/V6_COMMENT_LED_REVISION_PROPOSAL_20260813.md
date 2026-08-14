# V6 全文批注驱动修订建议（待作者审阅）

**对象：** `docs/paper/v6.docx`  
**日期：** 2026-08-13  
**状态：** 仅提出建议；**尚未写回 DOCX，尚未删除任何批注**。  
**审阅范围：** 全文 193 个正文段落、26 条 Word 批注，并联动核对标题、Abstract、Introduction、Results、图注、Discussion、Methods、Data/Code availability 与参考文献区。  
**科学边界：** 以 `CORE_SCIENTIFIC_LOGIC_CONTRACT.md`、`RESULTS_EVIDENCE_BOUNDARIES.md` 及当前 promoted metrics/source manifests 为准。

> 下文 P006、P021 等是本次提取文本中的内部段落索引，便于后续精确写回，并非论文显示的段落编号。

---

## 一、26 条批注逐条建议

### C0｜“这个不应该是最准确的 without 的对象”

**批注**  
`recurrent persistent activity` 混合了网络结构与时间活动状态。本文直接排除的是输入间持续放电，不是笼统排除一切 recurrent activity。

**原文（P006）**  
> To test this computationally, we tracked STSP across successive inputs in trained feedforward spiking networks, where working-memory history could evolve without recurrent persistent activity.

**修改建议**  
> To test this computationally, we tracked STSP across successive inputs in trained feedforward spiking networks, where working-memory history could evolve without persistent firing between inputs.

**全文同类扫描**  
P066 也出现 `without recurrent persistent activity or ongoing learning`，建议同步改为：
> ...without persistent firing between inputs or ongoing learning.

---

### C1｜“转折关系应该不准确”

**批注**  
“保持信息可用”与“表征持续变化”并非直接反转关系；这里要表达的是二者共存。

**原文（P008）**  
> Yet working-memory representations are not static: as new inputs arrive, they are transformed, selected and reorganized.5–8

**修改建议**  
> Such availability coexists with continual representational change: as new inputs arrive, working-memory representations are transformed, selected and reorganized.5–8

**全文同类扫描**  
P021 从“可解码”直接用 `therefore` 跳到“功能继承”的逻辑也不够严密，见 C10。

---

### C2｜“连接可以更自然”

**批注**  
`One solution` 的问题对象不明确，容易被理解为解决上一句的“记录困难”，而不是解释 state-to-state dynamics。

**原文（P009）**  
> One solution is offered by recurrent excitation–inhibition models, in which persistent firing keeps the current memory state active and thereby allows successive inputs to modify it progressively.13–16

**修改建议**  
> Recurrent excitation–inhibition models provide one mechanistic account of these state-to-state dynamics, in which persistent firing keeps the current memory state active and thereby allows successive inputs to modify it progressively.13–16

**全文同类扫描**  
Results 中 P036、P048 的段首承接也较笼统，分别在 C13、C16 中处理。

---

### C3｜“递进关系”

**批注**  
应明确从单项目保持推进到多项目和序列，而不是用指代宽泛的 `This principle has also...`。

**原文（P009）**  
> This principle has also been extended to multiple items and sequences

**修改建议**  
> Beyond single-item retention, STSP-based models have been extended to multiple items and sequences

**全文同类扫描**  
本项需与 C4 连续应用，才能完整建立“单项目保持 → 多项目分离式表示 → 跨输入交互仍未解释”的递进链。

---

### C4｜“具体的对比感觉还需增强”

**批注**  
原文列出了模型与记录结果，却没有明确比较维度：既有 STSP 模型侧重不同项目状态的保持，记录结果则显示新输入会重组已保持的表征。

**原文（P009）**  
> with successive inputs represented in distinct item-specific synaptic states for item maintenance, ordering or updating.28–31 By contrast, neural recordings show that an intervening input can reorganize an already retained working-memory code within the same prefrontal population, suggesting interaction between successive representations rather than their simple separation.7 What remains unclear is how activity-silent STSP supports such interaction and reorganization.

**修改建议（与 C3 合并后）**  
> Beyond single-item retention, STSP-based models have been extended to multiple items and sequences, with successive inputs represented by distinct item-specific synaptic states for maintenance, ordering or updating.28–31 This emphasis on distinct item-specific states contrasts with neural recordings showing that an intervening input can reorganize an already retained working-memory code within the same prefrontal population, implying interaction between successive representations.7 Whether activity-silent STSP can support this cross-input interaction and reorganization remains unclear.

**全文同类扫描**  
未发现另一处完全相同的文献对比问题；但 P055 将结构形态与条件功能写成因果递进，属于更严重的“比较/关系不准确”，见全文联动项 B。

---

### C5｜“指代不明确”

**批注**  
`These analyses` 不清楚具体指向何种证据；同句还把 inherited state 写成被直接改写，模糊了“当前层处理形成下游 successor”的方向。

**原文（P011）**  
> These analyses show that activity-silent STSP is more than a passive store: retained synaptic states shape new-input processing before being rewritten into successor states that influence what follows.

**修改建议**  
> Layer-resolved tracking across successive inputs shows that activity-silent STSP is more than a passive store: inherited STSP shapes new-input processing, which in turn forms downstream successor states that influence what follows.

**全文同类扫描**  
同层“自我写回”歧义还出现在 P006、P010、P013、P037、P039 图注、P041、P044、P058 和 P062；需按全文联动项 A 同步修正。

---

### C6｜“后面缺少一个总结价值句”

**批注**  
现有末句仍是结果清单，没有回到全文科学问题并给出机制层价值。

**原文（P011）**  
> Across the sequence, history-conditioned updating recurs, while the resulting states remain organized and selectively influence later processing.

**修改建议**  
> Across the sequence, this history-conditioned transition recurs, and the resulting states remain organized and selectively influence later processing. Together, these findings identify input-driven, history-conditioned inter-layer state transitions as the basic unit of working-memory evolution in this model.

**全文同类扫描**  
P030、P044、P051 和 P058 的节末也存在“复述结果/标题而没有关闭问题”的现象；P051、P058 分别在 C18、C20 中处理，P030、P044 见全文联动项 C。

---

### C7｜“指代不明确”

**批注**  
`these analyses` 空泛；第（2）项还把层间 successor formation 写成输入重写原状态。

**原文（P013）**  
> The goal of these analyses was to determine: (1) whether retained STSP shapes processing of a new input; (2) whether that input rewrites the retained state for what follows; (3) whether this process recurs across successive inputs; and (4) how the accumulated state is organized and when it influences later processing.

**修改建议**  
> The Results address four linked questions: (1) whether retained STSP shapes processing of an identical new input; (2) whether this history-conditioned processing forms a downstream successor state; (3) whether that successor conditions the next input and the inter-layer transition recurs; and (4) what structural organization remains after repeated transitions and, separately, under which conditions retained STSP influences later processing.

**全文同类扫描**  
除 C5 所列方向性问题外，P057 的 `The effect` 同时可能指 removal effect 与 interaction，建议改为 `These overlap-related effects` 并明确其鲁棒性对象。

---

### C8｜“用词很奇怪”

**批注**  
`Their product` 用代词承载关键定义；`synaptic efficacy` 又与图注和 Methods 的 `effective STSP support` 不统一。

**原文（P016）**  
> Their product defined the instantaneous synaptic efficacy, GSTSP(t) = u(t)x(t) (Fig. 1d).

**修改建议**  
> The product of the two state variables defined instantaneous effective STSP support, GSTSP(t) = u(t)x(t) (Fig. 1d).

**全文同类扫描**  
P018 图注和 P081–P083 Methods 已使用 `effective STSP support/effective presynaptic support`；建议统一到该术语。必须继续区分完整 STSP state（u 和 x）与乘积 GSTSP，不能把二者混为同一量。

---

### C9｜“是否准确？”

**批注**  
`item-identity decoding` 容易被理解为具体 MNIST exemplar 身份；实际端点是按 0–9 digit class 标注的十分类线性解码。

**原文（P021）**  
> Throughout this silent interval, item-identity decoding from the joint u/x state remained above the 10% chance level in every layer and at every sampled delay from 100 to 1,200 ms (...; Fig. 2c).

**修改建议**  
> Throughout this silent interval, linear decoding of digit class from the STSP state remained above the 10% chance level in every layer and at every sampled delay from 100 to 1,200 ms (...; Fig. 2c).

**全文同类扫描**  
同步修改：
- P021：`item-specific information` → `digit-class information`；
- P023：`persistent item information` → `persistent digit-class information`；
- Fig. 2c 图注：`decoding of item identity` → `decoding of digit class`；
- P107 Methods：`To quantify item information` → `To quantify digit-class information`。

---

### C10｜“与前一句逻辑并没有很严密”

**批注**  
前一句只建立了信息可解码；功能继承还要求该状态实际影响后续计算。需要显式区分两级证据。

**原文（P021）**  
> Establishing functional inheritance therefore required testing whether the retained state contributed to subsequent computation.

**修改建议**  
> Decodable digit-class information established persistence, but functional inheritance additionally required the retained STSP state to influence subsequent computation.

**全文同类扫描**  
P022 的 `Dynamic–static differences weakened with delay` 未明确是哪一个 endpoint；建议在写回前依据 Supplementary Fig. S1 将 `differences` 改为具体结果名，避免宽泛回指。

---

### C11｜“大部分可以直接用上面定义的 STSP 指代”

**批注**  
主文反复使用 `joint u/x state` 增加符号负担。P016 已定义 STSP state 由 u 和 x 组成，主叙事可用 STSP state；仅在精确描述变量组成或干预对象时保留 u/x。

**原文（P022）**  
> To test this functional contribution, we exchanged retained joint u/x states between trials at readout while holding the current input and fixed long-term weights constant.

**修改建议**  
> To test this functional contribution, we exchanged retained STSP states between trials at readout while preserving each recipient trial’s input and the fixed long-term weights.

**全文同类扫描**  
Results 和图注中的下列 `u/x state` 可按语境改为 `STSP state` 或首次写 `STSP state (u and x)`：P025、P036、P039、P042、P043、P046、P053。Methods 中的 u、x、φ、GSTSP 及变量级 restoration/transfer 定义必须保留，不做机械全局替换。

---

### C12｜“总起的逻辑感觉不是很准确”

**批注**  
原句没有准确承接前节的“共同输入更新 + 历史条件化残差”，也没有明确本节要定位贡献并检验下游 successor 的定向改变。

**原文（P034）**  
> With history-conditioned processing established, we next sought to identify its inherited source and determine whether that source also directed successor formation.

**修改建议**  
> Having shown that an identical input evoked a common input-driven update with a history-conditioned residual, we next asked where inherited STSP contributed to that residual and whether selectively changing it could redirect the downstream successor.

**全文同类扫描**  
P041 的跨节总起也应明确“Layer 1 inherited state → Layer 2 successor → 下一输入 → Layer 3 successor”，见全文联动项 A。

---

### C13｜“这段的目的定义应该不准确”

**批注**  
`changing inherited STSP alone was sufficient to redirect successor formation` 容易被理解为 STSP 脱离当前输入即可形成 successor。实验只证明：其他条件固定时，改变 inherited Layer 1 STSP 足以定向改变 downstream Layer 2 successor。

**原文（P036）**  
> Building on this localization, we next tested whether changing inherited STSP alone was sufficient to redirect successor formation.

**修改建议**  
> Having localized this contribution to input-overlapping Layer 1 STSP, we next asked whether selectively changing the inherited Layer 1 STSP state could redirect the downstream Layer 2 successor under otherwise identical conditions.

**全文同类扫描**  
P042 末句也把充分性写得过宽，需限定到 post-B Layer 2 successor 对 identical-C early Layer 2 processing 和 post-C Layer 3 successor 的模型内充分性，见全文联动项 A。

---

### C14｜“总结得很没有力度”

**批注**  
原句重复“held constant”，但没有把干预识别出的证据等级准确抽象出来。

**原文（P036）**  
> Because the current input and all other states were held constant, the donor-directed shift showed that inherited Layer 1 STSP was sufficient to redirect successor formation.

**修改建议**  
> By isolating inherited Layer 1 STSP as the only transferred state, the donor-directed shift established its model-internal causal sufficiency to redirect the downstream Layer 2 successor under the tested conditions.

**全文同类扫描**  
P042 的 successor transplant 应使用同样的“model-internal causal sufficiency + 精确端点 + tested conditions”格式；不得扩写为必要性、完全中介或唯一机制。

---

### C15｜“K 出现得莫名其妙，完全没有定义”

**批注**  
K 在 Results 首次出现时未定义；此处表示 B 之前的 history depth。

**原文（P042）**  
> At K = 1 and K = 5,

**修改建议**  
> At both tested pre-B history depths (K = 1 and 5, denoting one or five items preceding B),

**全文同类扫描**  
K 在不同协议中的参考边界需分别说明：
- P042/P043：pre-B history depth；
- P050、Fig. 6/7 图注：terminal sequence length，即 state-forming items 的总数。
不建议用一个含糊的全局 K 定义覆盖两种协议语境。

---

### C16｜“连接感觉不自然”

**批注**  
第一句只重复“产生 accumulated state”，第二句再重复“examined how organized”，没有从前节答案自然生成本节结构问题。

**原文（P048）**  
> Successive history-conditioned transitions gave rise to an accumulated STSP state. We therefore examined how repeated rewriting organized this state as history deepened, focusing on the expression of multiple constituent contributions and their experienced configuration.

**修改建议**  
> Having established repeated history-conditioned updating across the sequence, we next asked whether the terminal STSP state retained multiple constituent contributions and their experienced configuration as history accumulated.

**全文同类扫描**  
P055 也需要用“separately asked”开启并行的 conditional-function 模块，不能写成“access that organization”，见全文联动项 B。

---

### C17｜“这个统计量出现得很奇怪，其他都没有只有这里有”

**批注**  
数值本身正确；问题是 `n = 20 networks` 被孤立插入估计值括号。全文另有 P028 的同类内联 n，因此不是“只有这里有”，但展示方式确实不统一。

**原文（P049）**  
> The state remained similar to both constituent templates (n = 20 networks; item A, mean similarity, ...).

**修改建议**  
> Across 20 independently trained networks, the terminal state remained similar to both constituent templates (item A, mean similarity, ...).

**全文同类扫描**  
P028 的 `(n = 20 networks; ...)` 同样建议改成句首 `Across 20 independently trained networks, ...`。图注和 Methods 已统一声明全部主分析的独立单位为 20 个 networks，不需在每组估计值括号中反复插入 n。

---

### C18｜“只是结果复述，不像整体结论总结”

**批注**  
第一句基本复述标题和 P050；第二句又将 terminal morphology 写成 conditional function 的因果前提，违反两者为并行 outcome modules 的证据边界。

**原文（P051）**  
> Repeated rewriting therefore organized accumulated history as a distributed, multi-component STSP state in which constituent contributions and their history-specific spatial arrangement remained expressed. This organization raised the next question of when the accumulated state could influence subsequent processing.

**修改建议**  
> Together, constituent retention and configuration specificity showed that continual state evolution preserved structured historical information: accumulated content remained distributed across multiple components while retaining its experienced spatial configuration.

**全文同类扫描**  
P055、P058 仍会重新建立 morphology → function 的错误递进，需与本项同步修改，见全文联动项 B。

---

### C19｜“定义不完整”

**批注**  
`partial cues` 到 Methods 才操作化。Results 首次使用时应说明其为保留一定比例 active encoded-spike sites 的降质输入，并明确实际 readout 端点。

**原文（P056）**  
> We first tested whether partial cues could access constituent information retained in the accumulated STSP state.

**修改建议**  
> We first tested whether degraded partial cues—inputs retaining a specified fraction of active encoded-spike sites—could elicit constituent-specific target-class readout from the accumulated STSP state.

**全文同类扫描**  
- Fig. 7a 图注的 `partial-cue target recovery` 建议改为 `partial-cue target-class readout`；
- 首次出现 `keep probability` 时应说明其为 retained active-site fraction；
- P056 首次出现 AUC 时展开为 `area under the curve (AUC)`。

---

### C20｜“依然是结果复述”

**批注**  
`completed the ... cycle` 把并行的 conditional-function 模块并入核心机制链；第二句还把 cue assay 扩写为没有直接测量的 successor formation。

**原文（P058）**  
> Functional expression completed the history-conditioned updating cycle: successive inputs formed a multi-component, history-specific STSP state, and a later input selectively accessed that state through content match and pathway overlap. Once engaged, the inherited state shaped current processing and was rewritten into the successor available for the next transition.

**修改建议**  
> Together, the cue and pathway analyses showed that retained STSP influenced later computation conditionally: target-class readout depended on cue-content match, and early-processing effects were concentrated along input-overlapping pathways.

**全文同类扫描**  
P055 需同步改成并行问题总起；P057 的 `the circuit route` 暗示唯一性，建议改为：
> ...identified input-overlapping pathways as a circuit route contributing to the conditional expression of retained STSP.

---

### C21｜“有点啰嗦，铺垫太多导致结论不明确”

**批注**  
首尾重复“按 transformations 而非 instantaneous form 评价 identity”，核心判断出现过晚。

**原文（P063）**  
> This account of continuity suggests a different basis for evaluating mnemonic identity over time. Identity is commonly inferred from continuity of content or neural representation, yet dynamic coding makes instantaneous state similarity an incomplete criterion. A functional alternative is to compare the transformations supported by each state: distinct neural configurations may be mnemonically equivalent if they impose the same constraints on future computation, whereas similarly decodable states may differ if they support different transitions. Mnemonic identity may therefore be better characterized by the transformations a state supports than by its instantaneous neural form.

**修改建议**  
> Dynamic coding makes continuity of content or neural representation an incomplete criterion for mnemonic identity over time. A functional criterion instead compares the transformations supported by each state: distinct neural configurations may be mnemonically equivalent if they impose the same constraints on future computation, whereas similarly decodable states may differ if they support different transitions. Mnemonic identity may therefore be better characterized by a state’s functional consequences than by its instantaneous neural form.

**全文同类扫描**  
P066 后两句重复列举机制分工，可压缩为：
> The results establish the sufficiency of this mechanism within the model; its necessity, uniqueness and prevalence in biological working memory remain open, and other circuit processes may preserve, redirect or replace the constraints carried across successive transitions.

---

### C22｜“与 chunk 概念的连接不明确，甚至应与下一句合并”

**批注**  
应把两项并行证据与“integration can preserve internal differentiation”合成一个推进单元，同时避免声称 Fig. 7 直接访问 Fig. 6 所定义的 morphology。

**原文（P064）**  
> The accumulated STSP state retained multiple components and history-specific organization (Fig. 6a–f), and partial cues selectively accessed constituent information within that state (Fig. 7a–c). These complementary findings suggest that integration can preserve internal differentiation.

**修改建议（合并两句）**  
> In the present model, however, accumulated STSP retained multiple components in a history-specific organization (Fig. 6a–f), while partial cues elicited constituent-selective readout (Fig. 7a–c), suggesting that integration can preserve internal differentiation.

**全文同类扫描**  
与 C18/C20 相同，全文中凡把 morphology 与 conditional function 写成因果或访问链的地方均需改为并行证据关系。

---

### C23｜“可以直接融合到其他句子里，让结论更紧密”

**批注**  
独立的 `This formulation...` 打断了 store-based framing 与 transformation-based account 的对照；P064、P065 还连续以 `The same...` 起笔。

**原文（P065）**  
> The same emphasis on reconfiguration also reframes debates over the neural substrate of working memory, which have largely asked where remembered information resides—in persistent activity, activity-silent synaptic states, or some combination of the two. This formulation implicitly treats continuity as the persistence of a state instantiated between events.

**修改建议（合并两句）**  
> Beyond capacity, reconfiguration also reframes debates over the neural substrate of working memory: by asking where remembered information resides—in persistent activity, activity-silent synaptic states or some combination—these debates implicitly equate continuity with the persistence of a state between events.

**全文同类扫描**  
P063 的首尾重复已由 C21 压缩；P066 的重复机制清单见 C21 的联动建议。

---

### C24｜“过于具体、过于详细，可以压缩”

**批注**  
原文展开成三个方法清单，遮蔽了真正的概念判断。压缩时仍需保留：功能等价性检验、状态扰动、跨机制职责比较。

**原文（P067）**  
> History-controlled probes could determine whether distinct neural configurations preserve common input-to-state mappings; temporally targeted perturbations could test whether altering a latent state redirects both immediate processing and the state formed next; and comparisons across circuit mechanisms could identify which aspects of these transitions each substrate controls.

**修改建议**  
> Across candidate circuit mechanisms, history-controlled probes and temporally targeted perturbations could test whether distinct neural configurations support equivalent input-to-state transformations and whether altering a latent state redirects both current processing and successor formation, thereby distinguishing which parts of the transition each substrate controls.

**全文同类扫描**  
Discussion 中最明显的另一处过度展开是 P066 的候选机制清单，建议按 C21 联动项压缩；未发现需要调整段落边界的结构问题。

---

### C25｜“似乎与 Fig. 2 的结果有些不一致？”

**批注**  
两组数值**不冲突**，但当前端点标签容易让读者误以为它们应相等：
- 91.158%：训练结束、STSP disabled、工作记忆 assay 所用一次性 1/U checkpoint rescaling 之前，对完整 10,000-image MNIST test split 的分类准确率；
- 90.945%（稿件四舍五入为 90.95%）：加载 checkpoint、完成 1/U rescaling 并启用动态 STSP 后，独立 1,000-trial post-training single-item assay 的准确率。该 assay 没有显式 zero-input delay，因此 `delayed-recall accuracy` 命名不准确。

**原文（P104）**  
> Across the 20 independently trained networks used in the main analyses, mean post-training test accuracy was 91.158% (95% CI, 90.998–91.318%). A trial was assigned to the class group containing the earliest Layer 3 spike; trials without a Layer 3 spike were scored as incorrect.

**修改建议**  
> At the end of training, with STSP disabled and before the one-time checkpoint rescaling used for the working-memory assays, mean accuracy on the full 10,000-image MNIST test split was 91.158% (95% CI, 90.998–91.318%) across the 20 independently trained networks. In the subsequent assays, a trial was assigned to the class group containing the earliest Layer 3 spike; trials without a Layer 3 spike were scored as incorrect.

同时将 P021 改为：
> Across 20 independently trained networks, mean accuracy in the post-training single-item assay was 90.95% (95% CI, 90.60%–91.29%; Fig. 2a).

Fig. 2a 图注同步改为：
> a, Accuracy in the post-training single-item assay across 20 independently trained networks ...

**全文同类扫描**  
未发现另一组真正的数值冲突；但发现多处端点单位与标签问题，见全文联动项 D、E。

**证据核对**  
- 91.158%：`results/multi_snn/sdnn_ensemble_20/sdnn_ensemble_20/ensemble_summary.csv`；
- 注册值与 CI：`docs/paper/revisions/model_protocol_parameters.csv`；
- 90.945%：`results/paper_figure_multi_seed/final_six_figures_v5_c5_revised_20260804_r2/fig1/metrics/panel_b_statistics.csv`；
- 单项 assay 协议：相同 bundle 的 `panel_b_source_manifest.csv`、各 seed `run_config.json` 及 `src/experiments/paper_figures/fig1/subexperiments/baseline.py`；
- 项目内部映射已明确两者不同：`MAIN_SUPPLEMENT_SENTENCE_MAPPING_20260812.md`。

---

## 二、全文扫描后必须联动处理的问题

以下问题由批注触发后在全文发现。它们不是额外改写偏好，而是相同问题在其他段落中的真实复现。

### A. 层间 successor-state 方向需全文统一（高优先级）

核心方向必须保持：**当前层 inherited STSP → 条件化当前层处理 → 下游层形成 successor**。建议联动替换：

1. **P006 Abstract**  
原文：
> ...current-input processing was shaped by an inherited activity-silent STSP state and, in turn, rewrote that state into a successor...

建议：
> We found that inherited activity-silent STSP conditioned current-input processing, which in turn formed a downstream successor state that conditioned the next input.

2. **P010 Introduction**  
原文：
> ...retained synaptic history shapes each new input and is, in turn, rewritten for the next.

建议：
> ...retained synaptic history shapes processing of each new input, and that processing forms the downstream successor state inherited at the next transition.

3. **P037 Results**  
原文：
> ...this update rewrites the inherited state into a successor jointly organized by prior history and current input.

建议：
> These findings identify a history-conditioned inter-layer transition. The current input recruits inherited Layer 1 STSP along overlapping pathways, allowing retained history to condition the Layer 1 response; the resulting activity redirects the Layer 2 synaptic update, forming a downstream successor jointly organized by prior history and the current input (Fig. 4g).

4. **P039 Fig. 4g 图注**  
原文：
> ...inter-input decay yields successor Sk+1...

建议：
> An inherited state Sk at the current layer conditions processing of a new input; the resulting feedforward activity and downstream synaptic update form successor Sk+1 at the next layer, which can condition a later input.

5. **P041 sequence hinge**  
建议整段首句群改为：
> The preceding experiments established a single history-conditioned inter-layer transition from inherited Layer 1 STSP to a downstream Layer 2 successor. We next asked whether that successor could condition the identical next input and redirect the Layer 3 successor formed thereafter, and whether input-associated state updating recurred beyond passive STSP evolution across the tested sequence.

6. **P042 successor transplant 总结**  
建议：
> Under this controlled transplant, the donor-directed shifts established model-internal causal sufficiency of the post-B Layer 2 successor to redirect identical-C early Layer 2 processing and the downstream post-C Layer 3 successor under the tested conditions.

7. **P044 sequence 总结**  
建议：
> Together, successor transfer and matched-passive comparisons extend the mechanism from one transition to sequence-level evolution: a downstream successor can condition the next input and redirect the successor formed thereafter, while input-associated updating recurs beyond passive evolution across the tested boundaries. The accompanying rescue and loss trends indicate increasing behavioral costs as history deepens.

8. **P062 Discussion 首段**  
建议整段替换为：
> Our results show that, in the present model, working-memory continuity can be maintained through repeated transitions between activity-silent synaptic states. Inherited STSP conditioned each new input, and the resulting processing formed a downstream successor state that carried history into the next transition; across inputs, these states remained organized and their functional expression depended on subsequent input. Working-memory continuity may therefore reside in the history-conditioned relations linking successive states rather than in the persistence of any single representation.

### B. Terminal morphology 与 conditional function 必须保持并行

除 C18、C20 外，P055 建议改为：
> Having characterized the terminal morphology of accumulated STSP, we separately asked under which input conditions retained STSP influenced later processing. We tested whether this conditional function depended on cue content and on overlap between the incoming pathway and retained STSP support.

不得使用 `access that organization`、`completed the cycle` 或 Fig. 6 → Fig. 7 的因果表达。

### C. Results 节末综合不能只复述标题/数据

1. **P030**  
建议：
> Together, these results distinguish two contributions to the transition: the current input supplied the common update organization, while retained history selectively conditioned the behavioral outcome and synaptic update.

2. **P044**  
按联动项 A7 综合 successor reuse、matched-passive recurrence 与行为趋势，而不是再次逐项复述。

3. **P051、P058**  
分别按 C18、C20 关闭“结构是什么”和“何时/沿何处表达”两个独立问题。

### D. 百分比与百分点存在系统性混用（高优先级）

Promoted metrics 将以下差值明确标为 `percentage_points`，正文应同步：

- P035：`5.60%`、`1.34%` → `5.60 percentage points`、`1.34 percentage points`；`difference-in-differences, 8.52%` → `8.52 percentage points`；
- P043：rescue `23.1%`、loss `33.7%` → `23.1 percentage points`、`33.7 percentage points`；
- P056：mean gain `27.6%`、cue contrasts `5.05%` 和 `42.45%` → `percentage points`；
- P057：`2.52% more recruitment loss` → `2.52 percentage points more recruitment loss`；`16.0% interaction` → `16.0-percentage-point interaction`。

对应权威文件为各 figure bundle 的 `metrics/panel_*_statistics.csv`。

### E. 统计标签与端点命名需补齐

1. P035 的 `P = 1.24 × 10−36` 应标为 `unadjusted P`；
2. P056 的 interaction `P = 1.13 × 10−13` 应标为 `unadjusted P`；
3. P028/P043 使用 `behavioral rescue/loss rate`，Fig. 7d 使用 `sequence-only rescued-position fraction`，避免两个 rescue 定义混用；
4. P050 首次出现 Neff 时建议写：
   > the structural effective component number, Neff—an inverse-concentration measure of fitted constituent weights—...
5. P057 的 `The effect` 改为明确的复数或具体 endpoint。

### F. 首次定义与术语层级

- P028 首次定义 aligned/mismatched：final history item 是否与 B 的 digit class 匹配；
- P036 首次说明 `non-STSP fast state`，至少括注 membrane、conductance、refractory 与 inhibitory variables；
- P042、P050 分别定义各自协议中的 K；
- P056 定义 partial cue、target-class readout，并展开 AUC；
- 主文优先用 STSP state；Methods 保留 u、x、φ、GSTSP 的精确变量定义。

---

## 三、建议的作者审阅方式

作者可按以下任一方式回复：

1. **全部接受**；
2. 按 C0–C25 标注“接受 / 保留原文 / 按作者版本改”；
3. 对全文联动项 A–F 分别批准或否决。

获得确认后，再对 `v6.docx` 制作版本化副本，写入获批修改，并仅删除对应已解决批注；未获批批注保持原样。写回后还需核对 Word run 格式、上下标/斜体、图注、批注数量、公式与图片完整性。