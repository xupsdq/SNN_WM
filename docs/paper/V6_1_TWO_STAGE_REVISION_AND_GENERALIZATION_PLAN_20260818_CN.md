# V6.1 两阶段修改与跨任务一般性计划

**创建日期：** 2026-08-18  
**状态：** OPEN PLAN  
**依据：** `V6_1_FULL_STORY_SUBMISSION_REVIEW_20260818_CN.md`  
**对象：** `v6_1.docx`、`supplementary_information_v6_1.docx` 及其当前图件、Table S2、Source Data 和投稿包。

## 0. 总目标与阶段边界

后续工作分成两个顺序阶段：

1. **第一阶段：补全当前结果和全部必须修改项。** 先让现有证据被准确、完整、内部一致地呈现，形成可投专业计算神经科学期刊的版本。
2. **第二阶段：跨任务一般性。** 在第一阶段冻结后，独立验证同一 STSP successor-state 机制能否支持一个行为目标不同、明确需要读取过去信息的任务。

第二阶段不得成为第一阶段修复事实错误的前置条件。第二阶段若结果为阴性，也不应反向改写第一阶段已经成立的模型内证据；此时只需进一步收窄一般性主张。

### 执行原则

- 第一阶段原则上不重跑训练或大规模模拟；优先使用当前持久化数据修正文稿、图注、统计解释和文献定位。
- 第一阶段如增加派生分析，只能读取既有输出，并生成新的版本化派生产物。
- 第二阶段必须作为独立实验 DAG：持久化输入与配置 → 每网络运行产物 → 网络内聚合 → 网络级统计 → 图件。
- 绘图是叶任务，只读取已经存在的统计/Source Data，不得隐式触发训练或模拟。
- 主推断单位继续是独立训练网络；低层 trial/site/comparison 先在网络内聚合。
- 每一项修改先更新证据/claim 合同，再同步正文、补充材料、图注和 Source Data；禁止在多个文件中各自修补出不同版本。

---

# 第一阶段：补全当前结果与必须修改项

## 1.1 阶段目标

形成一套满足以下条件的稿件：

- 图、图注、正文和 Methods 对同一操作给出同一含义；
- 所有主要数字可追溯到当前持久化证据；
- 结构零、解析零和描述性低层数据不被包装成普通经验推断；
- 新颖性不与直接先例冲突；
- working-memory、successor、organization 和 access 均在现有证据边界内使用；
- 主文、补充材料、Table S2、Source Data 和投稿包同步。

第一阶段完成后，可根据最终强化程度考虑 `Journal of Computational Neuroscience` 或 `Neural Computation`。

## 1.2 工作 DAG

```text
P1-00 证据与claim冻结
    ├── P1-01～P1-03 三项硬矛盾
    ├── P1-04～P1-09 因果与统计边界
    └── P1-10～P1-12 novelty、认知边界和Methods
                    ↓
          P1-13 主文/补充材料同步
                    ↓
          P1-14 Source Data与统计审计
                    ↓
          P1-15 联合盲读与投稿包冻结
```

除 P1-00 外，同一层级任务可以并行诊断；DOCX 只应由一个写入者统一同步，避免版本分叉。

## 1.3 P1-00｜冻结证据与 claim 清单

**依赖：** 无。  
**任务：**

- [ ] 以当前 `PAPER_AUTHORITY.json`、科学逻辑合同、证据边界和正式 artwork 为唯一输入；
- [ ] 建立“主张 → 正文段落 → 图/面板 → 统计端点 → Source Data”的逐项映射；
- [ ] 将需要删除、降级、保留或补强的主张分别标记；
- [ ] 明确本阶段不重跑模拟的边界和允许的派生分析。

**完成标准：** 后续每项修改都能指向一个已冻结证据端点，不再凭旧 caption、旧 package 或内部任务名选择数字。

---

## 1.4 P1-01～P1-03｜修复三项硬矛盾

### P1-01｜同步 Fig.4g artwork 与图注

**问题：** 当前 artwork 表达 `Inherited STSP → Selected firing → Downstream firing → Successor STSP`，但 DOCX 图注仍描述旧版 `S_k → decay → S_{k+1} → later input selection`，并包含当前图中不存在的 dot-size/blue-subset 编码。

**任务：**

- [ ] 以当前正式 artwork 为准重写 Fig.4g caption；
- [ ] 删除旧 `S_k`、inter-input decay、later-input 和 dot-size 描述；
- [ ] 检查 Results、Discussion、Methods 与 Fig.4g 是否都保持层间 successor 方向；
- [ ] 确认没有把 successor 写成同层状态自然衰减所得的新状态。

**完成标准：** 不看正文也能仅凭 Fig.4g 和 caption 恢复正确的层间因果链。

### P1-02｜修正 Fig.7d 的 delay 解释

**问题：** persisted data 对 delay 明显非单调；负的 sequence-length × delay interaction 不等于 access 随 delay 单调下降。

**任务：**

- [ ] 删除“longer delay attenuated access”等单调表述；
- [ ] 改写为 load–delay 联合依赖和效应随 delay 改变/压缩；
- [ ] 在 Results/caption 中明确曲线非单调；
- [ ] 检查 Abstract、Discussion 和 Conclusion 中是否存在同类外推。

**完成标准：** 文稿对 Fig.7d 的方向与 `panel_d_plot_data.csv`、interaction summary 和 provenance 中的 `no monotonic claim` 完全一致。

### P1-03｜确认并重命名 Fig.2a 端点

**问题：** 当前端点来源更接近 baseline/current-probe classification，而不是 delayed recall；Supplementary Fig. S1 还显示 dynamic history 可能降低当前 probe accuracy 并增加 sample-label bias。

**任务：**

- [ ] 确认 Fig.2a 的真实 protocol、预测目标、分母和来源文件；
- [ ] 统一 panel title、y-axis、Results 和 caption 的端点名称；
- [ ] 在 Methods 明确 sample、delay、probe、目标标签和状态交换时刻；
- [ ] 区分“历史状态有功能影响”与“成功回忆旧项目”。

**完成标准：** 任何读者都不会把 Fig.2a 误读为网络正确回忆 delay 前的旧项目。

---

## 1.5 P1-04～P1-09｜校准已有因果与统计结论

### P1-04｜Fig.2d error composition 与 paired 证据归属

- [ ] 明确两组柱来自各条件各自的 error pool，分母并不相同；
- [ ] 不凭 Fig.2d 两柱单独声称 paired reciprocal change；
- [ ] 将 paired donor-directed flux 的证据准确归于 Supplementary Fig. S1c；
- [ ] 在正文简述 donor calibration 和 opportunity balance。

**完成标准：** composition、paired transition 和 donor calibration 三种证据不再混用。

### P1-05｜Fig.4f 推断单位与伪重复防护

- [ ] caption 写明每网络约 1,000 个低层 comparisons；
- [ ] 写明先在网络内归一化为 100%，再对 20 网络等权平均；
- [ ] 明确 histogram 为 descriptive，network summaries 才承担跨网络推断；
- [ ] 确保图面或 caption 能区分 comparison count 与 network count。

**完成标准：** 读者不会把约 20,000 个 comparison rows 当成独立样本量。

### P1-06｜Fig.5b 精确零 controls

- [ ] 明确 non-overlap/random 是构造性描述对照；
- [ ] P 值只归属于实际检验的 overlap endpoints；
- [ ] 避免把精确零写成“经验上未检测到效应”；
- [ ] 将“effect removed”改写为 tested endpoint attenuation，避免暗示完整 mediation/唯一机制。

**完成标准：** 图注和 Results 能区分构造零、描述零和推断端点。

### P1-07｜Fig.5d passive centered-cosine zero

**最低必要路径：**

- [ ] 在 Methods/Results 明确 no-input passive centered-cosine displacement 对该仿射恢复解析为零；
- [ ] 将结论收窄为每次输入引起 state morphology/direction change；
- [ ] 不再把该对照写成输入更新幅度大于普通被动衰减幅度。

**可选增强路径：**

- [ ] 从既有边界状态派生一个对幅度敏感的距离指标；
- [ ] 保留网络内聚合和网络级统计；
- [ ] 将其作为补充分析，不覆盖原始 centered-cosine 指标。

**完成标准：** 即使不增加新指标，文字结论也不超出该指标实际能证明的范围。

### P1-08｜Fig.7f structural-zero interaction

- [ ] 在 Results 和 Methods 中同时写明 no-overlap cells 是 endpoint construction 的 structural zeros；
- [ ] 将主要结论改为 overlap pathways 内的 high–low STSP contrast；
- [ ] 不把 15.997 pp 当作普通经验 2×2 interaction；
- [ ] 让 Fig.7e targeted removal 承担主要因果贡献结论，Fig.7f 只作空间一致性证据。

**完成标准：** no-overlap structural zero 的限制不再只藏在补充材料或 caption 中。

### P1-09｜Fig.6 component/order 证据分工

- [ ] 明确 `N_eff` 是 NNLS decomposition 的有效表达量，不是 working-memory capacity；
- [ ] 让 six-order identification 承担 experienced configuration 的主要证据；
- [ ] 检查 constituent-template 共线性、unrelated-template similarity 或 condition number 是否可由现有数据诊断；
- [ ] 将 coefficient-free morphology 作为不同尺度的补充证据，不写成同一指标的独立重复。

**完成标准：** component、order、capacity 和 access 四个概念不再互换。

---

## 1.6 P1-10～P1-12｜重建 novelty、认知边界与 Methods

### P1-10｜外部先例与 novelty 重写

**必须纳入的定位：**

- [ ] Buonomano & Merzenich 1995；
- [ ] Buonomano & Maass 2009；
- [ ] Hu et al. 2021 feedforward STPNet；
- [ ] 最新 STP chunking reviewed preprint；
- [ ] 已引 activity-silent、sequence/order 和 code-morphing 工作。

**不得作为单独创新的内容：**

- STSP 支持 activity-silent memory；
- hidden state 改变后续输入响应；
- feedforward STP 支持短时记忆；
- STP 表达多项目、顺序或 chunking。

**保留的创新点：**

> 在分层前馈脉冲网络中，对 inherited `u/x`、identical-current-input processing 和 downstream post-input `u/x` successor 进行连续追踪，并通过选择性 state substitution 证明 successor 的定向形成与复用。

**完成标准：** Introduction 的 gap 是窄、可检验且确由当前结果填补的问题；Discussion 对直接先例逐项区分。

### P1-11｜working-memory 与 organization 边界

- [ ] 明确当前最直接证明的是 history-dependent computation；
- [ ] 承认 adaptation/repetition priming 尚未被完全排除；
- [ ] 将 partial-cue endpoint 称为 cue-supported access，不称 free recall；
- [ ] 删除或大幅降级 chunking、capacity、hierarchy 主张；
- [ ] 明确 Fig.6 structure 与 Fig.7 function 是并列分支，不构成 structure→access 的实证因果链；
- [ ] 保留 sufficiency、非 necessity/uniqueness/prevalence 的限制语句。

**完成标准：** 标题、Abstract、Results、Discussion 和 Conclusion 使用同一主张边界。

### P1-12｜Methods 与统计透明度

必须补齐：

- [ ] Fig.2 sample–delay–probe 完整 protocol、目标和分母；
- [ ] STSP 在训练时关闭、训练后开启；
- [ ] `static_frozen` 是 API approximation，不是精确冻结当前 `u/x` gain；
- [ ] presynaptic-site gain 是否由所有 outgoing connections 共享；
- [ ] exact-B 0.5/0.05 thresholds 的来源和稳健性边界；
- [ ] 每个 multiplicity family 的完整成员列表；
- [ ] one-sided tests 的方向和预设时点；
- [ ] 20 seeds 只支持对当前 architecture/dataset/pipeline 的训练随机性稳健；
- [ ] main–supplement 证据交叉引用完整。

**完成标准：** Table S2 和 Source Data 可以独立恢复 family、test、null、direction、estimate、CI、P 和 inferential unit。

---

## 1.7 P1-13～P1-15｜同步、联合复读与冻结

### P1-13｜统一修改主文和补充材料

**依赖：** P1-01～P1-12。  
**任务：**

- [ ] 先修改 claim/evidence 合同；
- [ ] 再统一修改 Results、Discussion、Methods 和 captions；
- [ ] 同步 Supplementary Methods、figure legends 和 Table S2；
- [ ] 检查所有主图/补图交叉引用；
- [ ] 不在多个 DOCX 中保留互相竞争的旧句子。

### P1-14｜Source Data 与统计审计

- [ ] 核对 Fig.1–Fig.7、S1–S7 内嵌图哈希；
- [ ] 抽查正文数字、CI、P、n 与 Source Data；
- [ ] 输出完整 multiplicity-family machine-readable manifest；
- [ ] 标记 descriptive、constructed、derived 和 inferential endpoints；
- [ ] 若增加 Fig.5d 派生指标，为其建立独立 provenance，不覆盖旧数据。

### P1-15｜联合盲读和投稿包冻结

执行顺序：

```text
主文盲读
  → 补充材料核查
  → 主文—补充材料闭环复读
  → artwork/数值/引用/哈希检查
  → 重建投稿包
```

**第一阶段退出门：**

- [ ] Fig.4g、Fig.7d、Fig.2a 三项硬矛盾清零；
- [ ] 每个主结论均有当前持久化证据；
- [ ] 结构零、解析零和低层描述数据正确标注；
- [ ] novelty 不与直接先例冲突；
- [ ] Discussion 不再越界到未检验的 chunking/capacity；
- [ ] 主文、补充材料、Table S2、Source Data 和 captions 一致；
- [ ] 主要数字无需重跑模拟即可复核；
- [ ] 投稿包明确标记本阶段的代码、环境和 artifact lineage 状态。

---

# 第二阶段：跨任务一般性补充

## 2.1 阶段目标

回答：

> 同一 STSP successor-state 机制是否不仅改变当前分类，还能被任务主动读取，用于回答关于过去信息的问题？

第二阶段不是“再换一批 MNIST 参数”，而是改变行为目标。增加 seeds、delay、sequence length 只属于参数稳健性；换 Fashion-MNIST 但仍分类当前图像只属于跨数据集。只有从当前分类转向 delayed recognition、match/non-match、顺序或位置查询，才构成真正跨任务。

第一阶段冻结是第二阶段的唯一前置依赖。

## 2.2 推荐的最小跨任务

优先采用一个延迟识别/序列查询任务：

1. 输入 3–5 个项目；
2. 经过预设延迟；
3. 给出候选项目或位置 cue；
4. 网络回答：
   - 候选项目是否出现在先前序列；或
   - 某个序列位置出现了哪个项目。

该任务必须让正确答案依赖过去信息，而不是仅依赖当前 cue 的类别。

### 最小实现边界

- 优先保留当前 backbone 和 STSP 动力学；
- 如需新增 readout，首先尝试简单线性 readout，不引入 recurrent memory；
- task readout 的训练、验证和 confirmatory test 要分开；
- 不重新制作完整七图，只做一个 tracer-bullet 证据链；
- 在任务合同冻结前不启动批量实验。

## 2.3 第二阶段实验 DAG

```text
P2-00 任务与验收合同
        ↓
P2-01 固定配置与持久化输入
        ↓
P2-02 每网络任务运行和行为端点
        ↓
P2-03 silent-state与state-substitution机制端点
        ↓
P2-04 successor reuse与pathway intervention
        ↓
P2-05 网络内聚合和网络级统计
        ↓
P2-06 Source Data与纯绘图叶任务
        ↓
P2-07 与第一阶段稿件整合或独立报告
```

## 2.4 P2-00｜任务与验收合同

在运行前冻结：

- [ ] 任务输入、sequence length、delay、cue 和目标标签；
- [ ] train/validation/confirmatory test split；
- [ ] 主要 endpoint、方向、null 和最小有意义效应；
- [ ] dynamic、static/no-memory、sham 和 targeted-removal controls；
- [ ] inferential unit 和 multiplicity families；
- [ ] 哪些结果将被视为机制复现，哪些只是行为可行性。

**完成标准：** 结果出来后不再根据图形选择 task、delay、endpoint 或方向。

## 2.5 P2-01～P2-04｜最小证据链

| 编号 | 科学问题 | 必需证据 |
|---|---|---|
| P2-01 | 行为是否真正依赖过去信息 | dynamic STSP 相对 no-memory/static control 改善预设 memory-required endpoint |
| P2-02 | 延迟期是否 activity-silent | delay firing 消失，但过去项目仍可从 `u/x` 解码或被任务 readout 使用 |
| P2-03 | state 是否具有定向因果作用 | donor state substitution 将答案或内部 successor 定向推向 donor history |
| P2-04 | 是否形成并复用 successor | 当前 cue 后的下游状态影响下一次查询/更新；targeted overlap removal 选择性削弱该作用 |

只增加行为 accuracy 而没有 state substitution 或 pathway intervention，不足以证明当前论文的机制跨任务复现。

## 2.6 必要替代解释控制

- [ ] positive/negative query 数量平衡；
- [ ] 标签重复、视觉相似性和序列位置平衡；
- [ ] 使用未见测试图像；
- [ ] 保留静默、失败和无响应 trial；
- [ ] 正交区分 task-relevant history 与仅重复刺激；
- [ ] 区分 cue-supported recognition 与当前 cue classification；
- [ ] 低层 trial 先在网络内聚合；
- [ ] 预先声明主要端点、方向和 multiplicity family；
- [ ] 不根据 delay 曲线事后声称单调性。

## 2.7 第二阶段退出门

- [ ] 一个行为目标不同的任务在独立测试集上可重复；
- [ ] memory-required endpoint 在 no-memory/static 条件显著或实质性下降；
- [ ] state substitution 产生 donor-directed change；
- [ ] targeted pathway removal 选择性削弱该效应；
- [ ] 主要结果不能由标签重复、视觉相似度或成功 trial 筛选解释；
- [ ] 结果可以浓缩成一张主图，最多再加一张补图；
- [ ] 所有图为读取既有输出的纯绘图叶任务。

### 第二阶段决策

- **阳性且闭环：** 第一阶段 + 第二阶段可首投 `PLOS Computational Biology`；若再有跨数据或生物对齐，可评估 `Communications Biology`。
- **行为阳性但机制未复现：** 不能声称 successor mechanism 跨任务；只作为行为扩展或不纳入主稿。
- **阴性：** 不强行调参到阳性；保留第一阶段稿件，进一步收窄一般性，考虑 `Journal of Computational Neuroscience` 或 `Neural Computation`。

---

# 3. 当前执行顺序

```text
第一阶段
P1-00 证据冻结
    ↓
P1-01～03 硬矛盾
    ↓
P1-04～09 统计与机制边界
    ↓
P1-10～12 novelty / 认知边界 / Methods
    ↓
P1-13～15 联合复读与投稿包冻结
    ↓
第二阶段
P2-00 任务合同
    ↓
P2-01～04 跨任务最小证据链
    ↓
P2-05～07 统计、图件和稿件整合
```

当前不要同时启动多个新任务、架构或数据集。先完整关闭第一阶段，再以一个预先声明的跨任务实验回答一个明确问题。
