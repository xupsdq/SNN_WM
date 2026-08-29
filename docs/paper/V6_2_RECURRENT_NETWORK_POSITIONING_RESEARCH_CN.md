# V6.2 工作记忆计算模型扩展中循环网络的定位研究

**状态：** 研究/定位记录（不构成实现授权）
**范围：** V6.2 主文与补充材料、当前分支结果、Git 历史中的 recurrent-STSP DMS 文档，以及一手外部论文。
**日期：** 2026-08-27

> **后续方向校正：** 压力测试仍是有效的机制验证框架，但不再被视为循环扩展的主要贡献。固定任务与两跳 successor 设计可以保留；当前主瓶颈与预期突破已转为“在 Masse 行为基线之上，建立可训练且更生物合理的循环-STSP 前向网络与学习桥”。具体架构分层和训练路线见 [V6.2 生物合理循环网络基线研究](V6_2_BIOLOGICALLY_PLAUSIBLE_RECURRENT_BASELINE_RESEARCH_CN.md)。

## 1. 执行摘要

V6.2 的核心不是证明“工作记忆里存在一个能存信息的状态”，而是证明固定脉冲回路中的工作记忆表征如何随连续输入逐步演化：当前层继承的 STSP 状态调制新输入的处理，该处理在下一层形成同时受历史与当前输入约束的 successor state，随后该状态又成为下一次转移的条件。[V6.2 主文](v6.2.docx)（尤其第 2、19–21 页）已经把持续放电、递归兴奋和自上而下控制明确留作可能影响因素，而没有把它们纳入当前模型的因果结论；补充材料还把“natural recurrence”限定为输入驱动的跨转移状态递归，而不是循环突触网络动力学。[V6.2 补充材料](supplementary_information_v6.2.docx)（第 6、12–13 页）

**一句话定位：**

> **循环网络是 V6.2 successor-state 假说的机制压力测试与边界扩展：它检验快速循环动力学何时保持、选择/重定向或替代 STSP 携带的历史约束，并检验这种作用如何改变 successor formation、操纵和访问。**

因此，循环网络不应被定位为“再做一次能完成 DMS 的 RNN”，也不应被写成与 STSP 争夺唯一存储位置的二元比较。最有信息量的问题是：在相同新输入下，慢的 STSP 历史约束与快的循环状态如何共同决定下一时间窗的处理和下一层 successor；规则输入又如何选择其中的读、写、稳定化或重定向路径。

当前结论可以压缩为四点：

1. V6.2 为“继承状态 → 历史条件化当前处理 → 下游 successor → 后续复用”建立了中心论证；循环网络应接在这条链上，而不是替换这条链。
2. 当前 formal 结果只证明一个含 800 ms 突触电流和 400 ms SFA 的 500 单元循环 LIF 模型可以完成任务；它没有分离 STSP、快速循环状态或其他慢状态的作用。
3. 当前 stripped 配对结果没有通过行为门，但 STSP 的 `x·u` 解码为 1.0；这只能说明存在可读的状态指纹，不能说明模型学会了行为记忆，更不能说明 STSP 或循环机制不可行。
4. 历史 10k/20M recurrent-STSP DMS 已经完成短延迟、无干扰条件下的行为—静默表征—单步 STSP 因果桥；再重复同一 DMS 端点不会回答 V6.2 的新问题。下一步应优先做 exact-input successor、连续两跳因果链和 fast/STSP 边界 2×2，再扩展到规则操纵、可变延迟、干扰和第二次 probe。

## 2. V6.2 已经主张什么、没有主张什么

### 2.1 已经主张的内容

根据 `CORE_SCIENTIFIC_LOGIC_CONTRACT.md`、`RESULTS_EVIDENCE_BOUNDARIES.md` 和 V6.2 主文/补充材料，当前稿件的论证骨架是：

```text
继承 STSP 状态
    ↓
相同新输入下的历史条件化处理
    ↓
下游层形成 successor state
    ↓
successor 被后续输入再次使用
    ↓
连续转移构成工作记忆表征的演化
```

其中有四个边界必须保持：

- 当前输入仍是当前处理和状态变化的主要驱动力；历史状态提供上下文和约束，不替代当前输入或直接给出正确答案。
- successor 不是旧状态的复制，也不是普通前馈重编码，而是历史与当前输入共同作用后的下游重新组织状态。
- 论证方向是“当前层继承状态 → 当前层处理 → 下一层 successor”，不是同层状态简单自我写回。
- V6.2 的因果措辞是测试体制内的有界机制充分性/贡献；它不等价于大脑中的普遍机制，也不主张 STSP 是唯一工作记忆机制。

### 2.2 有意没有主张的内容

V6.2 没有证明以下命题：

| 未主张的命题 | 为什么不能从 V6.2 推出 |
|---|---|
| 持续放电或循环兴奋不是工作记忆机制 | 当前模型有意不含循环兴奋持续活动；缺席只能定义模型边界，不能否定其他机制。 |
| STSP 是唯一、必要或完整中介 | 当前证据围绕 STSP 状态的继承、条件化和 successor 作用；没有穷尽快速神经状态、突触电流、规则输入等替代路径。 |
| 所有任务都由 activity-silent STSP 完成 | V6.2 研究的是固定输入协议和特定 operating regime；任务操纵、延迟、干扰和访问要求可能改变机制分工。 |
| 循环网络只负责“存储” | 循环动力学还可能负责输入整合、选择、稳定化、误差恢复和输出访问。 |
| “natural recurrence”就是 recurrent synaptic dynamics | 补充材料中的术语指 successive inputs 驱动的状态递归；它不能被直接改写成已验证的循环突触动力学。 |
| 任何可解码状态都是工作记忆成功 | 解码只证明某个特征含有可预测信息；它不证明行为依赖、因果必要性、充分性或唯一中介。 |

所以，加入循环网络的科学动机不是补一个“更像脑”的模块，而是对 V6.2 的历史约束在另一种动力学底盘上进行边界检验：哪些约束被保留，哪些被快速状态选择或重定向，哪些会被循环吸引子覆盖。

## 3. 当前分支结果的证据边界

当前任务合同见 [`masse_delayed_cue_lif_task_spec.md`](../experiments/protocols/masse_delayed_cue_lif_task_spec.md) 和 [`masse_delayed_cue_lif_stsp_match_spec.md`](../experiments/protocols/masse_delayed_cue_lif_stsp_match_spec.md)。下面只把持久化结果作为相应 profile 的局部证据，不把它们合并成超出设计的机制结论。

### 3.1 formal：行为可解，但机制混杂

`results/masse_delayed_cue_lif/formal/` 的配置为 500 个循环单元，其中 375 个普通 LIF、125 个 SFA-LIF；另有 800 ms 突触电流和 400 ms SFA 时间常数。测试时间点准确率为 **0.9419966**，试次准确率为 **1.0**，最佳检查点为 epoch 74。

这建立的是：

- 在该任务、训练预算和输入协议下，模型存在一条可行的行为解题路线；
- 该路线包含快速循环网络，但同时包含两个显式慢状态，因此不能把行为成功归因于 STSP、持续放电、循环吸引子或任何单一变量；
- 它是后续机制实验的行为/工程基线，而不是 V6.2 的 STSP 机制验证。

### 3.2 stripped 配对：行为阴性，内部状态可读

`stripped_stsp` 与 `stripped_no_stsp` 使用 500 个普通 LIF、去掉独立突触电流和 SFA，并复用同一试次表和训练合同；两者唯一预设结构差是循环边上是否存在脉冲触发的、按细胞共享的 STSP。持久化摘要给出：

| profile | 测试时间点准确率 | 测试试次准确率 | 延迟末放电解码 | 延迟末 STSP `x·u` 解码 | 行为门 |
|---|---:|---:|---:|---:|---|
| `stripped_stsp` | 0.4909 | 0.4922 | 0.125 | 1.000 | 未通过 |
| `stripped_no_stsp` | 0.4963 | 0.4922 | 0.125 | 不适用 | 未通过；合法阴性 |

已有规格还记录：`x·u` 的 1.0 解码在未训练初始化上也出现；STSP 分支训练结束时循环权重谱半径由约 0.94 变为 9.24。后者是需要单独诊断的动力学/训练稳定性信号，不是对 STSP 机制的科学结论。

这些结果的正确口径是：**当前这套去慢状态、单 seed、训练合同和参数下，STSP 分支没有建立行为可行性；no-STSP 分支同样没有建立行为可行性。** 负结果不能写成“STSP 不可行”或“循环网络不可行”，因为模型没有通过行为门，且只测试了一个训练底盘和一个 seed。

### 3.3 四类证据必须分开

| 证据层 | 当前例子 | 允许的表述 | 不允许的升级 |
|---|---|---|---|
| 行为可解 | formal 的时间点约 94.2%、试次 1.0 | 该模型/协议存在行为解题路线 | “STSP 解释了成功”或“V6.2 机制得到验证” |
| 可解码 | stripped STSP 的 `x·u` 为 1.0 | 该状态特征携带可预测的输入/历史指纹 | “模型学会了记忆”或“该状态是行为所需的存储” |
| 因果必要/贡献 | 需要 reset 后紧邻响应和行为变化 | 某状态在限定边界上有功能贡献/必要性 | 仅凭解码或终点相关性声称必要 |
| 因果充分/定向 | 需要 donor substitution 使相同输入下响应/后继沿供体方向移动 | 供体状态对响应或 successor 具有定向因果作用；范围是模型内、操作内的有界充分性 | 把 reset 当充分性，或把终点 donor 方向当完整中介 |
| 唯一中介 | 需要穷尽可替代状态和路径的组合干预 | 在预设因果图和替代集内排除其他等价路径 | 仅因 STSP 介入有效就称“唯一机制” |

特别是，`stripped_stsp` 的 STSP 解码结果不能抵消行为门失败；它最多是后续因果回放的输入线索。当前两条 stripped 结果也没有执行 V6.2 所要求的“同一输入、局部边界干预、下一窗口放电、下一层 successor、后续 reuse”链。

## 4. Git 历史已证明什么，以及为什么不能简单重复 DMS

### 4.1 `51c2849` 的 recurrent-STSP DMS 证据

提交 `51c2849`（`feat: add recurrent STSP causal DMS experiments`）曾加入一整套 10k/20M 固定循环图的行为—机制实验合同、实现和测试。其文档证据边界明确写成“pilot/非确认性”，但它已经比当前 stripped 配对更接近完整的单步因果桥：

- **短 pilot：** 一张 10,000 神经元、20,000,000 连接图，125 ms delay、无 distractor、12 个冻结 test trials。dynamic firing 行为 balanced accuracy 为 1.000，probe-only 为 0.500；延迟末放电解码 sample 为 0.250，STSP 解码为 1.000；STSP reset 后为 0.583；反向 donor swap 对供体标签准确率为 0.917，平均 donor projection 为 0.987。
- **三图 replication：** graph seeds `143202461`–`143202463`。行为准确率为 `0.972 ± 0.048`（范围 0.917–1.000），probe-only 为 0.500；延迟放电 sample 解码为 `0.306 ± 0.096`，查询前 STSP 为 1.000；reset 后行为为 `0.528 ± 0.048`；swap 对供体标签准确率均为 0.917，供体投影为 `1.079 ± 0.267`。
- **matched-query：** 在相同 query 下只替换 STSP，query 后 firing 有较弱的 donor-directed movement（平均约 0.165），而 post-query STSP successor 投影很强（平均约 0.920）；1k 缩放 tracer 的 firing 方向不成立，说明不能用小网络直接替代完整尺度动力学。

这已经支持如下有界结论：在冻结的 Tiddia 循环脉冲网络、短 delay、无干扰和小 test 集条件下，STSP 能携带 sample 信息，STSP reset 会破坏行为，donor STSP 能使最终判断朝供体方向移动；该单步因果角色在三张独立连接图上复现。

### 4.2 仍未闭合的部分

历史 DMS 文档同时明确没有证明：

1. 连续边界上的 `STSP(t) → firing(t+1) → STSP(t+1)`；
2. 第二次 probe 是否复用第一个 probe 后形成的 successor；
3. 可变 delay、distractor、规则输入和多项次序条件下的 operating regime；
4. STSP 与快速循环状态的相对贡献、交互作用或替代关系；
5. 网络级确认性结论（当时只有三张图、每图 12 个 test trials）。

### 4.3 `f42984d` 的含义

提交 `f42984d` 的主题是 `retain recurrent STSP core primitives`：删除 recurrent-STSP DMS 专用实验树，保留 Tiddia/NEST 等价内核、连接、检查点、逐边 STSP 和通用 DAG 工具。这是范围收缩和可复用底座整理，不是对旧 DMS 结果的反证。旧文档仍可通过 `git show 51c2849:<path>` 追溯，但不应把删除后的临时/历史结果重新当作当前稿件权威。

### 4.4 为什么不能把“再做一次 DMS”当作下一步

简单重复历史 DMS 会产生以下问题：

- 它只重复“循环 STSP 能做短延迟 DMS”的已知桥梁，不能回答 V6.2 的 successor-state 问题；
- 终点行为和 STSP 解码仍然无法区分“局部 successor 形成”与“其他慢状态/循环吸引子完成任务”；
- 即使重复得到阳性，也不会告诉我们 recurrence 是保存、选择、重定向还是替代历史约束；
- 若只换任务名或增加试次数，仍可能没有连续时间窗的局部因果链和第二次 reuse。

历史 DMS 最合适的角色是：**单步因果阳性参考、回放/干预平台和外部行为锚点**。新扩展的主问题应转向“V6.2 successor mechanism 在 fast recurrent dynamics 下的边界与分工”。

## 5. 一手外部文献综合

下表只使用原始论文或其正式期刊/全文页面；文献不是为了给项目结果背书，而是用来限定候选机制和实验判据。

| 一手来源 | 关键结果 | 对本项目定位的约束 |
|---|---|---|
| [Masse et al., 2019, Nature Neuroscience](https://www.nature.com/articles/s41593-019-0414-3) | 训练的循环 RNN 中，STSP 可支持短延迟维持；需要主动操纵信息时，持续活动会自然出现，且随操纵要求增加。 | STSP 与持续循环活动不是必须二选一；“维持”与“操纵/重组”可能调用不同动力学。项目应直接测量任务阶段和操纵阶段的分工。 |
| [Kozachkov et al., 2022, PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010776) | 有/无 STSP 的 RNN 都能在 distractor DMS 中保持记忆；STSP 网络更接近 NHP 活动并对突触消融更鲁棒。 | 行为阳性不证明 STSP 必要；STSP 的价值可能体现在 activity-silent 表征、鲁棒性或动力学形状，而不只是准确率。 |
| [Mongillo, Barak & Tsodyks, 2008, Science](https://doi.org/10.1126/science.1150769) | 循环脉冲网络中的突触短时易化可暂存输入痕迹，并在后续输入/脉冲下读出。 | STSP 是输入驱动的可读历史约束；应测边界状态和 reactivation，而不是只看延迟末平均放电。 |
| [Compte et al., 2000, Cerebral Cortex](https://academic.oup.com/cercor/article-abstract/10/9/910/289091) | NMDA 介导的循环兴奋与抑制性相互作用可形成空间工作记忆 attractor，并提高对干扰的稳定性。 | 快速循环状态可以承担稳定、恢复和吸引子校正；它不能被预先定义为“只是噪声”或“只是另一种存储”。 |
| [Mante et al., 2013, Nature](https://www.nature.com/articles/nature12742) | 训练循环网络复现 PFC 的 context-dependent computation；循环动力学参与输入选择和整合，而不是只维持一个静态内容。 | 规则/上下文输入应被当作选择动力学路径的控制变量；successor 的变化要同时分析内容和规则条件。 |
| [Chatham et al., 2014, Neuron（PMC 全文）](https://pmc.ncbi.nlm.nih.gov/articles/PMC3955887/) | 工作记忆中的 input/output gating 选择哪些内容被写入、更新或用于行为。 | 规则输入可能决定访问和操纵，而不是成为记忆内容本身；需要 rule-gated read/write 对照。 |
| [Wolff et al., 2017, Nature Neuroscience](https://www.nature.com/articles/nn.4546) | ping 可重新激活不表现为持续放电的 memory-specific hidden state，且 ping 强度与行为有关。 | “activity-silent”不能等同于“系统没有动力学”；访问事件本身可能是检验隐状态因果作用的必要扰动。 |
| [Orhan & Ma, 2019, Nature Neuroscience](https://www.nature.com/articles/s41593-018-0314-y) | 持续和序列型表示是受任务、时间尺度、输入结构等因素影响的连续谱，而非两类唯一解。 | 不应把一次训练得到的循环解当作唯一生物机制；需用 delay、distractor、操纵难度和规则条件扫描 operating regime。 |
| [Kim & Sejnowski, 2021, Nature Neuroscience](https://www.nature.com/articles/s41593-020-00753-w) | 脉冲循环网络中的兴奋性微回路和去抑制结构可产生长时间尺度、维持和灵活性。 | 快速状态的“持续性”可由回路结构和时间常数共同产生；必须记录膜、突触电流、refractory/延迟环等全部 fast state，避免把隐藏慢变量误归给 STSP。 |
| [Yang et al., 2019, Nature Neuroscience](https://www.nature.com/articles/s41593-018-0310-2) | 单一网络可学习多项认知任务，并形成任务/规则组合的内部表示。 | 规则输入和任务上下文可能重排共享动力学；扩展应先分离内容保持、规则控制和读出访问，而不是只看一项 DMS 的平均准确率。 |
| [Galgali et al., 2023, Nature Neuroscience](https://www.nature.com/articles/s41593-022-01230-2) | 从部分观测中区分局部循环、上游循环和外部输入并不容易；残差动力学和有针对性的扰动更有区分力。 | 本项目拥有完整模拟状态，应直接计算 recurrent drive、外部 drive、局部 Jacobian/向量场并做 clamp/lesion，而不是仅凭行为或相关轨迹推断 recurrence。 |

综合这些文献，可以形成三个外部约束：

1. **机制不是二元存储竞赛。** STSP 可以跨越低放电间隔保留约束，循环状态可以在需要时稳定、重定向、操纵或恢复；两者也可能冗余。
2. **任务需求决定动力学。** 简单保持、主动变换、规则切换、干扰恢复和行为访问不应共用一个“memory strength”指标。
3. **因果扰动高于解码。** 解码用于定位候选状态，reset/swap/clamp 用于检验必要性、充分性和方向；连续 successor 还必须在时间上逐段闭合。

## 6. 候选定位对比与推荐

| 候选定位 | 核心问题 | 新颖性/与 V6.2 的一致性 | 主要风险 | 结论 |
|---|---|---|---|---|
| A. STSP 与 persistent firing 的存储竞争 | 哪个状态在 delay 中更能解码内容？ | 低；把 V6.2 降成存储位置比较，忽略 successor。 | 解码不能给出必要性；可能把互补机制误判为竞争。 | 不推荐为主线。 |
| B. 循环网络能否完成 DMS | 加 recurrence 后行为是否过线？ | 低；历史 51c2849 已有短延迟正例。 | 重复端点，不闭合连续 successor。 | 只能作入场门/外部锚点。 |
| C. recurrence 作为 STSP 的替代者 | 去掉 STSP 后循环吸引子能否保留行为？ | 中；能检验 redundancy/replace，但不足以解释 V6.2 的层间重组。 | 需要排除隐含慢状态、输入捷径和训练差异。 | 作竞争假设，不作预设结论。 |
| D. recurrence 作为 successor 的快速选择/稳定/重定向机制 | 在相同输入和 STSP 边界下，fast state 如何改变下一窗口和 successor？ | 高；直接接到 V6.2 的“preserve, redirect, replace”。 | 需有精确边界状态与局部因果设计。 | **推荐。** |
| E. recurrence 作为 V6.2 机制压力测试与边界扩展 | V6.2 的 history-conditioned transition 在另一动力学底盘上何时成立、何时失效？ | 最高；允许互补、冗余、替代三类结果被证伪区分。 | 需要先定义机制分工和停止门，不宜一开始扩展所有生物约束。 | **推荐的论文/项目总定位。** |

推荐将 D 与 E 合并为一句工作定义：

> 循环网络不是 V6.2 的另一个存储器，而是决定历史约束如何在新输入中被表达、选择、稳定、重定向或替代的快速动力学层；successor state 是 STSP、fast state、规则输入和当前输入相互作用后的下游产物。

项目实现上保留两条角色不同的轨道：

- `recurrent_stsp` 的冻结 10k/20M Tiddia 网络：机制扰动和精确状态回放平台；历史三图 DMS 是单步因果阳性参考，不能直接当作 V6.2 新主结果。
- `masse_delayed_cue_lif`：训练任务和行为门平台；formal 是含慢状态的行为控制，stripped 是去占位慢状态后的当前失败对照。只有在行为门重新闭合后，才适合承载训练型 recurrence/STSP 分工实验。

## 7. 机制分工模型

用 (h_t) 表示 STSP 状态，(z_t) 表示快速神经状态，(r_t) 表示规则/上下文输入，(x_t) 表示当前外部输入。建议把每一步写成：

```text
(h_t, z_t, r_t, x_t)
          ↓
 fast recurrent processing + history-conditioned synaptic transmission
          ↓
     firing(t+1),  successor(t+1)
          ↓
          h(t+1), z(t+1)
```

| 变量/机制 | 主要职责假设 | 必须测量的证据 | 不应默认的结论 |
|---|---|---|---|
| STSP (h_t) | 跨低放电间隔保留历史依赖的慢约束；改变输入到循环网络的有效增益/释放。 | delay 末 `u/x/next-release` 解码；STSP-only reset/swap 对相同输入的即时 firing 和 successor 影响。 | 不是直接答案，也不是唯一存储；`x·u` 可读不等于行为必要。 |
| 快速循环状态 (z_t) | 输入整合、吸引子稳定、干扰恢复、重定向、再激活和行为读出。 | 膜电位、突触电流、refractory、delay ring、ongoing spikes、recurrent drive；fast-only reset/lesion/clamp。 | 不能把所有 persistent firing 都解释为记忆内容；也不能把 recurrence 只看成噪声。 |
| 规则/上下文输入 (r_t) | 选择 read/write/access 路径、调整增益和动力学 operating point。 | rule swap、规则 cue 时序、same-content/different-rule 对照；规则条件下的 Jacobian、读出和 successor。 | 不是自动的记忆存储；规则输入改变行为不等于改变内容状态。 |
| successor state (s_{t+1}) | 当前输入、既有历史和动力学处理后的下游关系状态；成为下一次转移的继承条件。 | 下游 successor 的 history×input 交互、donor projection、第二次 probe reuse、两跳链。 | 不是旧状态复制、普通前馈编码或只看 trial 终点的相关量。 |

该分工允许三种结果同时作为科学结果，而不是预先假设 STSP 必胜：

- **互补：** STSP 提供跨间隔约束，recurrence 提供即时选择/稳定/重定向；两者有交互项。
- **冗余：** 任一机制单独仍能完成任务，双重干预才损坏行为。
- **替代/覆盖：** fast recurrence 在某些操纵条件下恢复或重写 STSP 约束，或 STSP 在低活动条件下足以绕过 fast state。

## 8. 可证伪假设

以下假设应在预注册时视为相互竞争或可组合的候选，而非论文结论。

### H1：互补分工与交互

在相同新输入和相同长期权重下，STSP 状态决定历史约束，fast state 决定即时路由/稳定；二者对下一窗口 firing 或 successor 存在非加性交互。若 STSP-only 和 fast-only 干预效应可完全加和、且交互项在独立网络间接近零，则 H1 被削弱/证伪。

### H2：STSP 的跨静默边界必要性

在保持 fast recipient state、未来输入和随机流完全相同的情况下，只 reset 或替换 STSP，应改变共同 probe 后的紧邻 firing 与下游 successor，并在延迟/低放电条件下影响后续行为。若 STSP 边界干预不改变即时响应和 successor，或只在终点读出器上改变而没有局部动力学效应，则 H2 不成立。

### H3：recurrence 的操纵/访问作用

只在 probe、规则切换或操纵窗口抑制 recurrent drive，应显著改变即时选择、decision margin、recovery 和 successor reuse；只在静默 delay 抑制 recurrence 的效应可以较小。若 recurrence 在所有窗口的效应都相同，或只影响放电量而不影响路由/访问指标，则“快速循环负责操纵/访问”被削弱。

### H4：两种机制的冗余备份

单独 reset STSP 或单独 reset fast state 时，行为可能只小幅下降；同时 reset 两者时才出现大幅、超加性的损失。若任一单独干预已经完全消除行为，且第二个干预不增加损失，则 H4 被证伪，不能再称为冗余备份。

### H5：recurrence 对 STSP 的替代/吸引子校正

把历史 STSP 换成 donor 状态后，强 recurrent dynamics 可能将即时 firing 或后续 successor 拉回 recipient/规则吸引子；donor-directed effect 会随操纵强度、延迟或 recurrence gain 衰减。若 donor effect 在 fast state clamp 后反而保留或增强，则“recurrence 覆盖/校正 STSP”不成立。

### H6：successor 的跨输入复用

第一个输入/探针后形成的 successor state 被移植到第二个相同输入或规则化 probe 后，应使第二次响应和 decision margin 沿 donor history 方向移动；只改变第一次终点、对第二次 probe 无影响，则不能主张 V6.2 的 successive reuse。

### H7：规则输入控制访问而非改写内容

在内容和 STSP 历史相同的情况下，规则 cue 应改变 recurrent gain、读写路径或读出方向，而不应在无内容输入时凭空写入等价的 content-specific STSP。若规则 cue 单独即可重建完整内容状态，需重新划分规则和记忆状态边界。

## 9. 最小实验梯度与关键 2×2 边界设计

这里的“最小”指最小闭合科学链，不是最小代码改动。

### 阶段 0：冻结状态边界和行为入口

在任何大规模训练前冻结：连接/模型配置、trial manifest、输入随机流、边界时间窗、state schema、行为门和统计单位。明确列出 fast state 的全部组件：膜电位、突触电流、refractory、固定延迟环、ongoing spikes 以及任何不应隐藏在“神经状态”名下的慢变量。保留当前 formal 和 stripped bundle，不覆盖或热启动。

### 阶段 1：exact-input successor tracer

在既有可回放的循环 STSP 底座上，选两个 outcome-blind matched histories 和一个完全相同的新输入，保存同一边界的 (h_t,z_t,r_t)。先不追求行为任务，验证：

1. 同一输入下不同历史是否改变即时 firing；
2. 该 firing 差异是否形成下游 successor；
3. successor 是否同时具有历史和当前输入的可分离成分；
4. state transfer、sham 和 matched-random controls 是否方向一致。

这一阶段若只得到终点 STSP 距离而没有即时 firing/下游 successor，停止升级，不进入多 seed。

### 阶段 2：同一边界的 fast/STSP 2×2

在共同输入、长期权重、未来 RNG 和 recipient 非目标状态固定的条件下，构造四个分支：

| | STSP recipient（保留 recipient (h)） | STSP donor（换入 donor (h)） |
|---|---|---|
| **fast recipient**（保留 recipient (z)） | **FR–SR：完整 recipient/sham** | **FR–SD：只换 STSP** |
| **fast donor**（换入 donor (z)） | **FD–SR：只换 fast state** | **FD–SD：同时换 fast 与 STSP** |

另加完整 reset、STSP-only reset、fast-only reset 和 matched-random donor/sham。对每个分支预先计算：

- 紧邻下一时间窗 firing、recurrent drive 与 decision margin；
- 下游 successor 的状态距离、历史投影、当前输入投影；
- 再下一时间窗的 firing 和第二次 probe 响应。

核心交互量可写为：

```text
interaction = (FD–SD − FD–SR) − (FR–SD − FR–SR)
```

符号随 donor/recipient 编码预先固定。FR–SD 给 STSP 的局部效应，FD–SR 给 fast state 的局部效应，FD–SD 检验联合干预是否出现互补、冗余或覆盖。注意 fast donor 不能只复制一段 firing rate；必须包含使网络产生该段 firing 的完整状态边界。

### 阶段 3：训练型行为与规则访问

在阶段 1–2 的边界操作通过后，才在训练型循环 LIF 上加入 delayed-cue DMS/DMRS：至少覆盖 125/375/750 ms delay、无/有中途 distractor、平衡 probe 与 label，并加入 rule-gated access 或 ping/update 条件。`stripped` 当前的行为阴性必须先作为训练/动力学诊断处理，不能通过加回 800 ms 电流、SFA 或事后改变门槛来“修复科学结果”。

### 阶段 4：两跳 successor、网络复制和扩展

在第二次 probe 中重复同一边界干预，直接检验 `successor(t+1) → response(t+2) → successor(t+2)`。只有这一闭环稳定后，才扩展到更长序列、多项次序、跨规则复用和至少 20 个独立网络。旧三图结果可作为开发期阳性锚点，不能替代新的两跳确认。

## 10. 读出、因果证据门槛与停止/晋级标准

### 10.1 读出组合

每个阶段至少保留四类读出，且分别落在可追踪 artifact 中：

1. **行为：** held-out balanced accuracy，按 delay、distractor、rule 分层；probe-only、sample-only、random-seed controls；不把 probe 身份当作标签捷径。
2. **表示：** fast firing、STSP `u/x/next-release`、膜/电流/延迟环分别做 train-only decoder、冻结 test、cross-temporal generalization；低 firing rate 不是“firing 无信息”。
3. **动力学：** recurrent/external drive 分解、局部 Jacobian 或向量场、固定点/慢流形候选、扰动后 recovery time、donor projection、successor 历史×输入交互。
4. **因果：** sham、state reset、donor substitution、fast/STSP 2×2、double intervention、matched-random removal；所有分支使用相同输入和未来 RNG，并记录状态哈希。

### 10.2 证据门槛

| 门槛 | 最低要求 | 允许的结论 |
|---|---|---|
| 行为门 | 冻结 test 上通过既有任务合同的总体和分层门；行为读出独立于 probe-only 捷径。 | 该模型在该 operating regime 可解题。 |
| 表示门 | fast 与 STSP 的独立解码、跨时泛化、延迟/干扰条件完整报告。 | 哪些状态携带可读信息；不自动给出必要性。 |
| 局部必要性门 | 在相同输入下，state-specific reset/clamp 改变紧邻下一窗口 firing 或 successor，且 matched-random/sham 不能解释。 | 该状态在限定边界对当前处理有因果贡献。 |
| 定向充分性门 | donor substitution 使即时 response、successor 或 decision margin 沿供体方向移动，且 recipient 其它状态和输入保持不变。 | 供体状态对该输出具有模型内、操作内的有界定向充分性。 |
| 连续 reuse 门 | 至少两跳 `state(t) → firing(t+1) → successor(t+1) → firing(t+2)`，第二次 probe 仍有 donor-directed effect。 | 支持 V6.2 的 successor 形成与后续复用。 |
| 网络确认门 | 先开发期闭合，再以独立连接图为推断单位扩展至少 20 个网络；不把 trial 数当作 network 数。 | 可作跨网络确认性扩展。 |

**停止规则：**

- 行为门在预先冻结训练预算内失败：停止机制外推，保留为合法阴性；不把解码结果写成行为成功。
- 只有终点解码而局部 reset/swap 不改变下一窗口：停止“因果机制”晋级。
- reset 有效但 donor swap 无方向：只报告必要性/贡献，不报告携带特定内容的充分性。
- donor swap 有方向但第二次 probe 无复用：停止在单步因果角色，不写成 V6.2 连续 successor 机制。
- 2×2 结果只显示 fast 或 STSP 单独效应：按数据选择 STSP-dominant、recurrence-dominant 或冗余解释，不能强行写互补。

**晋级规则：** 只有行为门、局部必要性/定向充分性、两跳 reuse 和跨网络重复全部通过，才可把循环网络写入 V6.2 的扩展主论证；“唯一中介”不属于当前最低晋级目标，除非另行完成替代路径的全因果审计。

## 11. 与项目 DAG 的产物设计

遵循项目约束“persisted inputs → reusable artifacts → downstream outputs → plot-only leaves”。建议的新扩展不改变现有结果目录，而新增版本化 run root：

```text
persisted connectivity / weights / model config
              + frozen trial manifest / rule manifest
              + intervention boundary manifest
                           ↓
             baseline checkpoints and state snapshots
                           ↓
      replay branches: sham / reset / donor / fast-STSP 2×2
                           ↓
     per-network features: firing, STSP, fast state, drive, successor
                           ↓
    frozen decoder + behavior + causal endpoint summaries
                           ↓
          network-level aggregation and CI / effect table
                           ↓
                 plot-only figure/data consumers
```

建议的任务和产物职责：

| DAG 节点 | 持久化输入 | 结果产物 | 禁止事项 |
|---|---|---|---|
| `validate_config` | 模型、任务、规则、边界 schema | `run_config.json`、schema/hash report | 不生成模拟状态。 |
| `build_or_reuse_connectivity` | seed、连接配置 | `connectivity.pt`、identity manifest | 不在下游隐式重建图。 |
| `build_trials` | task/rule seed、平衡合同 | `trials.csv`、split manifest | 不让 test 参与 decoder/窗口选择。 |
| `capture_baseline` | 图、checkpoint、trial manifest | 边界 state snapshots、state schema/hash | 不丢弃 fast state 或随机流。 |
| `replay_interventions` | baseline snapshot、2×2 intervention manifest、共同输入 | 每分支 spikes、STSP、fast state、drive、successor | 不把 decoder 逻辑塞进 simulation。 |
| `within_network_endpoints` | replay artifacts | 行为、解码、projection、局部因果 summary | 不把 trial 当独立网络。 |
| `network_aggregator` | 多网络 summary | 网络级效应、CI、门槛状态、provenance | 不重跑 simulation，不读未声明的临时结果。 |
| `plot_only` | frozen source data/metrics | PNG/PDF/SVG、plot manifest | 不加载模型、不重跑仿真、不改变父哈希。 |

`results/masse_delayed_cue_lif/formal/`、`stripped_stsp/` 和 `stripped_no_stsp/` 应作为不可覆盖的历史/当前 profile 产物；历史 `tmp/recurrent_stsp_*` 只作开发痕迹，若要进入确认性阶段必须重新写入带配置、依赖、图 identity 和状态 hash 的版本化 `results/` bundle。分析和绘图只能消费已存在的父产物。

## 12. 现在不做什么，以及下一步建议

### 现在不做什么

- 不把循环网络扩展成“STSP versus persistent firing 谁赢”的存储位置竞赛。
- 不重复历史 125 ms、无 distractor 的短 DMS 作为新主结果。
- 不把 formal 的 800 ms 电流/400 ms SFA 解题路线写成 STSP 或 V6.2 机制证据。
- 不把 stripped 行为阴性写成 STSP 或 recurrence 不可行；先把它视为当前训练/动力学底盘的合法失败。
- 不把 `x·u` decode=1.0 写成成功机制；先检查初始化、分割、shuffle 和行为依赖。
- 不在没有 fast state schema 和边界 hash 的情况下做 donor swap；不把只复制 firing rate 的替换称为 fast-state intervention。
- 不在两跳机制闭合前直接上 10k/20M 全量、多规则、多项次序或全生物约束；生物逼真度应排在可识别性和因果闭合之后。
- 不恢复已删除的 DMS 专用实验树作为当前权威，不覆盖 formal 结果，不执行 Git commit/merge/push/tag 或删除分支。

### 下一步建议

1. 先把本报告第 9.1–9.2 节冻结成一页实验决策记录：exact-input successor + fast/STSP 2×2 + sham/reset/donor/matched-random 的边界 schema、输入 RNG 和判定方向。
2. 使用现有 `recurrent_stsp` 回放/检查点工具做开发期两跳 tracer，首先确认局部 `STSP(t) → firing(t+1) → successor(t+1)`，不要先追求行为大网。
3. 对 `stripped_stsp` 的训练稳定性单独做诊断；只有新的训练底盘通过行为门，才把它接入规则输入、distractor 和第二次 probe。
4. 机制和行为端点冻结后，再以独立网络为单位扩展至少 20 个 seed，并将统计、绘图和源数据接入项目 DAG。

## 13. 资料索引与复核边界

### 项目内资料

- [V6.2 主文](v6.2.docx)：第 2 页（问题与模型边界）、第 19–21 页（Discussion 与无循环兴奋说明）、第 22–25 页（STSP 和因果边界）。
- [V6.2 补充材料](supplementary_information_v6.2.docx)：第 4–6 页（状态捕获/恢复与有界充分性）、第 12–13 页（successor reuse 与输入驱动 recurrence）、第 19–21 页（表格与统计范围）。
- [核心科学逻辑合同](CORE_SCIENTIFIC_LOGIC_CONTRACT.md)：固定“继承 → 历史条件化处理 → 层间 successor → 连续复用”的方向。
- [结果证据边界](RESULTS_EVIDENCE_BOUNDARIES.md)：固定 bounded sufficiency、parallel outcome modules、network-level inference 和 plot-only 规则。
- [Masse-LIF 任务合同](../experiments/protocols/masse_delayed_cue_lif_task_spec.md) 与 [STSP 配对合同](../experiments/protocols/masse_delayed_cue_lif_stsp_match_spec.md)：固定当前 formal/stripped profile、行为门和合法阴性口径。
- 当前结果摘要：`results/masse_delayed_cue_lif/formal/summary.json`、`results/masse_delayed_cue_lif/stripped_stsp/summary.json`、`results/masse_delayed_cue_lif/stripped_no_stsp/summary.json`。
- 历史 Git 证据：`51c2849` 的 `RECURRENT_STSP_DMS_*` 文档和 `f42984d` 的删除/保留范围；历史文档不自动提升为当前稿件权威。

### 复核边界

本报告是定位和实验设计研究，不报告新的仿真，不把外部文献的行为结果迁移成项目结果，也不替代 V6.2 的正式 authority。任何后续数字必须回到当前结果 bundle、任务合同和独立网络统计单位重新核验。
