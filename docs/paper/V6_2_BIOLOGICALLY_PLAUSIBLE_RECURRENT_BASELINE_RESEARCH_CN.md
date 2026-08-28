# V6.2 生物合理循环网络基线研究

## 结论先行

当前固定任务本身可以保留，Masse 等人在 2019 年使用的延迟匹配任务也足以作为行为与任务基线。真正需要解决的问题不是再设计一个压力测试，而是：在不引入不可解释的慢状态的前提下，让循环网络与短时突触可塑性（STSP）先稳定地学会同一个任务，再逐层增加生物约束。

**已接受的主线顺序：** 第一阶段先完整复现 Masse rate-STSP 网络，形成可独立交付的行为阳性基线、冻结 teacher checkpoint 和全时序训练目标；第二阶段在完全相同的任务时序上训练 STSP 从第一步即参与前向动力学的稀疏 Dale LIF student。单方向 recurrent-LIF 只作为 kernel/因果 tracer；full-task no-STSP 网络不再与主线并行，只有在 teacher 与 LIF-STSP student 路线经过预设诊断后仍无法推进时，才作为最后的失败隔离分支。该决策以 [ADR 0002](../adr/0002-masse-first-lif-stsp-student-mainline.md) 为准。

建议把项目拆成三层，而不是把“生物合理”当成一个二元标签：

1. **生物约束架构 + BPTT/替代梯度**：Dale E/I、固定稀疏拓扑、脉冲 LIF、边特异的 TM 状态；训练仍使用 BPTT。这一层是当前最小的可训练目标，也是最稳妥的第一阶段结果。
2. **近似局部学习**：在已经训练好的前向网络上，把 BPTT 换成 e-prop/eligibility trace、随机反馈或有界坐标更新，并明确反馈信号与近似假设。这一层可以检验局部信用分配的可行性，但尚不能自动升级为机制主张。
3. **真正的机制主张**：突触只使用局部前后突触变量和明确的第三因子，在在线、跨试次、无 BPTT/RLS 的条件下完成行为，并通过必要性、充分性和规则操纵证明 STSP 是机制而不是可解码的伴随状态。

因此，当前应先做一条**可训练的 LIF-Dale-sparse-TM 架构路线**，先用替代梯度 BPTT 稳定行为；只有行为、状态重置和因果操纵都通过后，才把 e-prop 或其他局部学习算法作为第二阶段。固定的 Tiddia 10k/20M 网络保留为大规模因果/基底验证，不作为第一步的端到端可训练对象。

## 1. 当前项目事实与诊断

项目中的事实应与文献结论分开记录：

| 对象 | 已知事实 | 应有的解释 |
|---|---|---|
| formal baseline | 500 个隐藏 LIF/SFA 单元（375 LIF、125 SFA），含 800 ms 电流时间常数；SuperSpike + BPTT + Adam 可训练，测试 timepoint accuracy 约 0.942，trial accuracy 为 1.0 | 这是“任务可训练”的阳性控制，但 800 ms 电流和 400 ms SFA 都是可携带工作记忆的慢状态，不能把成功归因于 STSP |
| stripped STSP / no-STSP | 去掉 800 ms 电流和 SFA 后，500 LIF 的 STSP 版与 no-STSP 版在当前训练设置下都约为 chance（timepoint 约 0.49、trial 约 0.492）；STSP 的 x·u 读出在未训练初始化时也可达 1.0 | 这是优化/动力学失败，而不是“循环或 STSP 原理上不能支持任务”的证据。初始化即可读出的 x·u 只说明状态可分，不能说明网络在行为上使用了它 |
| 当前 match 实现 | STSP 主要保留 x/u 的梯度，同时对 recurrent spike drive 做 detach；匹配规格中的 STSP 是按 presynaptic cell 共享，而不是每条边独立 | detach 是训练近似，不能被包装成局部生物学习；它可能切断直接 recurrent credit path，只留下 STSP 状态路径，应作为明确的 A/B 消融 |
| 历史 Tiddia 短 DMS | 固定 10k/20M substrate 上已有 reset、donor swap、donor projection 等正因果信号；reset BA 约 0.583，donor swap 约 0.917，donor projection 约 0.987，STSP 状态可解码而 firing 解码接近 chance；多图行为约 0.917–1.0 | 这是有价值的正因果证据，但尚未完成规则操纵或多步 successor closure，不能写成完整的规则机制已被证明 |

对应的任务与匹配边界见项目中的 [Masse delayed-cue LIF task spec](../experiments/protocols/masse_delayed_cue_lif_task_spec.md) 和 [STSP match spec](../experiments/protocols/masse_delayed_cue_lif_stsp_match_spec.md)。当前 Tiddia 模拟器的固定图、事件调度和边状态实现见 [recurrent_stsp README](../../src/experiments/recurrent_stsp/README.md) 与 [config.py](../../src/experiments/recurrent_stsp/config.py)。

这个诊断改变了研究问题的表述：

> 不是“再找一个更复杂的压力测试”，而是“哪些生物约束足以让带 STSP 的循环脉冲网络在固定延迟任务上可训练，并且哪些状态真正被行为使用”？

## 2. Masse et al. 2019：任务/行为基线，而不是生物学习基线

### 2.1 实际任务和网络

Masse 等人的论文和作者代码是当前最合适的行为基线来源：

- 任务是 delayed match-to-sample（DMS），固定期约 500 ms，sample 约 500 ms，delay 约 1000 ms，test 约 500 ms；网络需报告 test 方向是否与 sample 相同。
- 输入为 24 个方向调谐的运动输入，循环层为 80 个 excitatory 和 20 个 inhibitory 单元，输出为 3 个 decision units。
- 循环层是非负 firing-rate/ReLU 动力学，不是脉冲 LIF；循环权重具有 E/I 符号结构，代码中也可切换 EI 约束，但默认连接是连续 rate 网络的 dense 形式。
- 只有 recurrent edges 使用 STSP；输入和输出边不使用 STSP。facilitating 和 depressing 的比例按 presynaptic population 划分。
- 为了节省计算，x/u 是**同一个 presynaptic neuron 的所有输出边共享**的状态；这与真正每条突触具有不同释放概率的生物实现不同。

论文的方法部分给出连续动力学和 STSP efficacy，作者代码则给出任务生成、E/I 权重参数化和训练脚本。应直接使用论文与作者仓库：[Masse et al., Nature Neuroscience 2019](https://www.nature.com/articles/s41593-019-0414-3)、[论文 PDF](https://www.cns.nyu.edu/wanglab/publications/pdf/masse.nn2019.pdf)、[作者代码](https://github.com/nmasse/Short-term-plasticity-RNN)。

### 2.2 训练方法

论文训练的是监督 rate RNN，而不是局部突触学习：

- 通过 BPTT 优化主要权重、偏置和初始状态；
- 使用 Adam，batch size 1024，学习率 0.02，梯度裁剪 0.1，训练 2000 个 batch；
- 交叉熵损失带有循环活动的 L2 正则，抑制过高活动；
- 测试阶段使用 mask，使网络在 test grace period 后才承担决策损失；
- 多个网络都达到 90% 以上任务准确率。

论文还通过在 test 开始时打乱神经状态或突触 efficacy，测试行为是否依赖其中一类状态。这类实验能显示某个状态被网络利用，但单独不能证明其必要性、唯一性或完整中介关系；尤其 x/u 本身也是 leaky integrator。

### 2.3 生物合理性的边界

Masse 基线提供了三个很有用的参照：固定任务时序、facilitation/depression 的对照、以及“持续神经活动”和“突触状态”之间的因果比较。但它不能直接作为本项目的生物学习终点：

1. 单元使用非负连续 rate/ReLU，而不是脉冲；
2. 网络虽有 E/I 符号结构，却不是由局部突触更新训练出来的；
3. BPTT/Adam 使用了整个 trial 的未来误差信息；
4. x/u 按 presynaptic cell 共享，减少了计算量，但不等于 edge-specific 突触状态；
5. 输入、输出、权重、初始状态和优化器均可由全局监督共同调整。

所以，Masse 的准确表述应是：**STSP 与循环工作记忆的任务/行为阳性基线**，而不是“生物合理的脉冲局部学习模型”。本项目的创新不应是重复证明这个任务能做，而应是逐步从这个阳性基线走向可训练的 Dale 稀疏 LIF，并测量代价和因果边界。

## 3. 候选路线对比

| 路线 | 现在的可训练性 | 架构生物约束 | 学习是否局部 | 对 edge-specific STSP 的适配 | 机制价值 | 建议 |
|---|---|---|---|---|---|---|
| R1：Masse rate RNN + STSP | 高，已有成熟阳性控制 | 只有 E/I-like rate 约束；无脉冲 | 否，BPTT/Adam | 容易，但原论文为 presynaptic-shared | 任务与行为基准 | 保留为 baseline，不作为突破 |
| R2：Dale 稀疏 E/I LIF + surrogate BPTT + TM | 中；最接近当前需要 | 脉冲、Dale、固定稀疏图、E-only readout、快速突触电流 | 否，第一阶段明确使用 BPTT | 前向自然；反向需要管理 O(E) 的状态梯度 | 证明“生物架构仍可训练” | **第一阶段推荐** |
| R3：LSNN/ALIF + e-prop | 中；e-prop 可在线，但调参和反馈要求较高 | LIF/ALIF、稀疏 E/I 可实现 | 是近似局部 eligibility；反馈信号不完全局部 | 需要把 TM 状态导数并入 eligibility，当前项目尚未验证 | 可检验信用分配近似 | R2 稳定后再做 |
| R4：FORCE/RLS 脉冲 reservoir | 高到中；对目标轨迹常很快 | 可用 LIF/Izhikevich，但常依赖混沌 reservoir | 否；RLS 使用全局误差和协方差 | 可做前向 STSP，但 RLS 不学习局部 TM 机制 | 训练可行性/上限诊断 | 只作为诊断或 readout baseline |
| R5：Full-FORCE + BCD 平衡 E/I | 中；可处理符号约束和在线更新 | 脉冲、Dale、动态 E/I balance | 比 RLS 更接近逐神经元更新，但仍有 teacher/global target | 需额外处理 edge STSP 和 teacher currents | 平衡网络的后续路线 | 不作为最小首轮 |
| R6：固定 Tiddia 10k/20M + 外部 input/readout | 前向高，端到端训练低 | 大规模稀疏 E/I、短延迟、边状态 | 固定 recurrent substrate；外部训练不等于机制 | 与 edge-specific STSP 天然匹配 | 大规模因果与迁移验证 | 作为独立 substrate/reference |

R2 的关键点是先承认学习算法边界：**用 BPTT 训练并不损害“Dale 稀疏 LIF 架构”的结果价值，但必须禁止把它称为局部机制学习。** R3/R5 只有在与 R2 使用相同前向动力学、相同状态重置和相同 held-out trials 比较时，才有信用分配层面的科学意义。

## 4. 文献支持的生物约束组件

### 4.1 Dale、稀疏和 E/I 平衡

Song、Yang 和 Wang 的 E/I RNN 工作明确展示了一个重要分离：可以在网络架构中固定 Dale 符号、E:I 比例、稀疏连接和 E-only readout，同时仍使用 BPTT/SGD 训练；作者也明确指出，当时讨论的学习方法不能被视为生物学习。其 rate 模型和官方 pycog 仓库可作为符号掩码、稀疏掩码和 E/I 初始化的参考：[Song et al., PLoS Computational Biology 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004792)、[论文 PDF](https://www.cns.nyu.edu/wanglab/publications/pdf/song_ploscb2016.pdf)、[pycog](https://github.com/xjwanglab/pycog)。

对于脉冲网络，Ingrosso 和 Abbott 研究了动态平衡的 E/I LIF 网络，并指出普通 RLS 直接训练 recurrent weights 与符号约束不相容，因而使用 Full-FORCE teacher 和 bounded coordinate descent（BCD）处理受符号约束的更新：[Ingrosso & Abbott, PLoS ONE 2019](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220547)、[论文 PDF](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0220547&type=printable)。这说明“Dale + 可训练”是可行研究路线，但不意味着 BCD 已经是本项目所需的 STSP 局部机制。

建议使用固定稀疏掩码和符号重参数化：

- E neuron 的所有 outgoing recurrent weights 非负，I neuron 的所有 outgoing recurrent weights 非正；
- 只训练非负 magnitude，掩码与 E/I identity 固定；
- 初始 E:I 约 4:1，readout 只接 E 或单独的输出层；
- 初始连接概率可沿用 0.1–0.2；不要一开始同时学习拓扑、延迟和 STSP 参数；
- 网络应有低初始 recurrent gain，并监测 firing rate、膜电位和线性化谱半径代理。

### 4.2 LSNN、e-prop、eligibility trace 与三因子学习

Bellec 等人的 e-prop 将 BPTT 梯度分解为：

$$
\frac{\partial E}{\partial W_{ji}}
=\sum_t L_j^t e_{ji}^t ,
$$

其中 e 是由局部前后突触状态递推的 eligibility trace，L 是对 postsynaptic neuron 的学习信号。这样可以在线计算，不需要保存整个 trial 的 BPTT 计算图。其 LSNN 使用带适应阈值的 ALIF 单元，适应变量能在数百毫秒到秒级保留历史；论文在 TIMIT、导航/证据累积和 Pong 等任务上展示了 e-prop 的可行性：[Bellec et al., Nature Communications 2020](https://www.nature.com/articles/s41467-020-17236-y)、[PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/)、[作者代码](https://github.com/IGITUGraz/eligibility_propagation)。

但这里必须区分三件事：

- ideal e-prop 的学习信号是总导数，在线并不可直接获得；
- online e-prop 使用输出误差经 feedback weights 路由到神经元，symmetric、random 或 adaptive feedback 都有额外假设；
- eligibility 是局部的，不代表误差信号、反馈权重和目标本身都完全局部。

因此，e-prop 是“近似局部信用分配”的强候选，不是自动成立的“突触机制”。更重要的是，ALIF 的适应变量会给当前任务再增加一个慢状态。若同时加入 800 ms current、SFA、ALIF 和 STSP，行为上的长时记忆来源将不可区分。当前项目应先不加入 ALIF/SFA；等 edge-TM 前向模型已经通过因果测试后，再做“ALIF fraction × STSP”显式因子实验。

### 4.3 FORCE/RLS：可训练性工具，不是局部机制

Nicola 和 Clopath 的 FORCE 工作展示了脉冲网络通过 RLS 快速拟合时间轨迹、分类和序列的方式：高维 recurrent reservoir 先产生丰富动态，RLS 在线更新读出或低秩反馈。该路线对“网络能否拟合目标”很有用，但有三个限制：

1. RLS 维护全局误差与协方差矩阵，标准形式的代价和信息流都不是突触局部；
2. 训练通常依赖 teacher/target signal 与高维 chaotic reservoir；
3. 大多数 recurrent weight 更新不天然满足 Dale 符号；即使特定分类例子加入了符号处理，也不等于一般机制。

原论文和模型存档可查：[Nicola & Clopath, Nature Communications 2017](https://www.nature.com/articles/s41467-017-01827-3)、[ModelDB 190565](https://modeldb.science/190565)。在本项目中，FORCE/RLS 可以作为“固定 STSP substrate 的 readout 上限”或“优化器 sanity check”，但如果使用 RLS 训练循环连接，就不能声称获得了突触局部的 STSP 学习机制。

### 4.4 异质神经元、延迟和突触时间常数

异质性确实有生物学依据，但它也是最容易混淆机制的扩展。Song 等人的工作把不同单元时间常数、较快抑制单元、噪声相关性、饱和/超线性 f-I 曲线列为可增加的生物细节；Ingrosso–Abbott 使用了不同的膜/突触时间常数；FORCE 脉冲模型也使用了快速上升、较慢衰减的突触滤波。

对当前任务建议按以下顺序加入：

1. 首轮统一 LIF τm、统一快速 τsyn（例如约 20 ms 的单指数，或固定的 2/20 ms 双指数），不使用延迟分布；
2. 行为通过后，只加入 E/I 两种 τm 或一个固定短延迟；
3. 再做 tau/delay 的窄分布，并把每个分布作为单独因子；
4. 最后才考虑大范围、edge-specific 的延迟和突触参数。

Tiddia 的 0.1–1.0 ms 短延迟通常不是秒级工作记忆的主要储存器，但 delay buffer 仍应在 reset、donor swap 和 state-shuffle 中显式重置；不能因为它短就把它从状态审计中删除。

### 4.5 edge-specific Tsodyks–Markram STSP

Tsodyks 和 Markram 的早期成对记录与现象学模型表明，突触释放概率、facilitation/depression 速率和突触间差异具有重要作用；生物上更自然的对象是每条突触的释放/资源状态，而不是一个 presynaptic neuron 的所有 output edge 共享一个 x/u：[Tsodyks & Markram, PNAS 1997](https://pubmed.ncbi.nlm.nih.gov/9012851/)、[论文 PDF](https://www.cns.nyu.edu/csh/csh04/Articles/Tsodyks-Markram-97.pdf)。

Masse 的 presynaptic-shared x/u 是合理的计算简化；当前 stripped match 也沿用了这个简化。对于最终生物架构，建议在每条 recurrent edge e 上保存 x_e、u_e，并让 presynaptic spike 更新该 edge 的 TM 状态，再用 w_e x_e u_e 形成有效突触效能。固定图、固定 U/τ 类别、edge state 与事件调度相结合，正好与 Tiddia 的 20M edge substrate 一致。

但 edge-specific 不是首个优化台阶：

- 每条边增加状态、梯度和 reset 账本，内存从 O(N) 变为 O(E)；
- 20M edges 上做全 trial BPTT 的计算图和梯度存储都很重；
- edge τ/U 的异质性会引入额外慢历史，若行为变化，无法立即判定是“更多容量”还是 STSP 机制；
- 当前尚未有项目内验证的 e-prop + edge-TM 梯度分解，不能把这一步当成已有文献直接支持的组合。

因此先用 shared-TM 作为**工程 debug rung**，再在同一稀疏图上替换为 edge-TM，并把两者的 forward state、梯度范数、行为和因果操纵同时报告。

## 5. 明确推荐的最小可训练架构

这里区分“第一道训练台阶”和“最小生物架构目标”。

### 5.1 第一台阶：先让梯度和行为闭环

- 规模先用 128–256 个 recurrent LIF 做单方向识别与动力学 tracer，再复现到项目规定的 500 单元；
- 使用固定稀疏 E/I 图，E:I=4:1，无自连接，E outgoing 正、I outgoing 负；
- 使用快速 synaptic current（约 20 ms 量级），**不使用** 800 ms current、SFA、ALIF、长 readout filter 或可训练初始状态；
- 输入到 hidden 也使用固定符号/稀疏掩码，readout 只读 E 或固定输出层；
- 首轮 STSP 使用 presynaptic-shared x/u，仅作为与当前实现一致的 trainability bridge；U、τ 分为 facilitating/depressing 两类，先固定不学习；
- 脉冲发放使用 surrogate derivative，recurrent gain 从低值开始，监测所有状态无 NaN、firing rate 不爆炸。

单方向识别只验证感觉编码、LIF 放电、循环稳定性和 readout；它不训练 sample–probe 比较，也不能作为“已经学会 DMS 的脚手架”。进入完整任务时必须直接使用项目固定的 sample/delay/rule/test 时序和 held-out trials，并让 STSP 从第一步就参与前向动力学。

### 5.2 最小生物架构目标

当第一台阶通过后，目标模型为：

> 500 个 current-based LIF recurrent units；固定稀疏 Dale E/I 图；E:I 约 4:1；E-only readout；快速统一 synaptic filter；recurrent edges 上 edge-specific TM x_e/u_e；无 SFA/ALIF、无 800 ms slow current、无 delay 分布。

这个目标已经包含三个有意义的生物约束：脉冲单元、Dale/稀疏架构、突触级短时状态；它没有把适应阈值、异质延迟、复杂双指数、局部学习和 10k/20M 全部叠加，因此仍有机会定位失败来源。

### 5.3 训练梯度和当前 detach 的边界

第一阶段的默认训练应是 trial-level BPTT 或逐步扩大的 truncated BPTT + surrogate gradient：

1. 在短 delay pilot 中用 50–100 ms 反传窗口和 state carry 观察梯度；
2. 随 delay 增大，扩大窗口或切换到整 trial BPTT；最终行为不能依赖一个从未覆盖 delay 的反传窗口；
3. 对 recurrent spike path 与 TM state path 分别记录梯度范数，使用梯度裁剪和低 recurrent gain；
4. 活动正则应在行为先稳定后再逐步加入，避免首轮把可用的持续/脉冲活动压掉。

当前实现把 recurrent spike drive detach、却保留 x/u 梯度，这是一个有用的诊断分支，但不能作为默认生物解释：

- detach 切断了通过 recurrent spike 的直接时间信用分配；
- 仍保留的 x/u path 可能让模型只通过 STSP 状态传播梯度；
- 如果 detach 版能学而完整 surrogate recurrent path 不能学，这首先说明优化路径不同，不说明局部学习已经实现。

因此每个主要版本都应至少有两个 gradient condition：A 为完整 surrogate recurrent path，B 为当前的 recurrent-spike-detach、TM-state-gradient-retained path。B 可以暂时帮助稳定训练，但论文必须把它标成优化近似，并在 e-prop 阶段重新定义真正的 eligibility。

### 5.4 零延迟双输入脚手架不成立；可行突破在训练方法

把 sample 与 probe 压缩成零延迟或同窗输入，会让网络学习两个输入交融后的瞬态轨迹。插入真实 delay 后，输入联合分布、循环轨迹和决策边界都发生变化；因此这类 scaffold 不能被假定可迁移到完整 DMS。另一方面，若 no-STSP 网络先用真实非零 delay 学完整 DMS，它又会被奖励形成 persistent firing/attractor，随后加入 STSP 只可能成为旁路。这两条都不能作为主路线。

当前可行空间应明确分成三条：

1. **单方向 recurrent-LIF tracer：** 只训练一个输入方向的识别，随后接入 STSP，研究循环动力学如何把输入写入和读取突触状态。它适合验证 kernel、稳定性和因果干预，但科学上接近把 V6.2 前馈底盘替换成循环底盘，不能单独承担新的工作记忆行为结论。
2. **Masse rate-STSP baseline：** 从完整 DMS/DMRS 时序开始、让 STSP 全程存在，使用作者式 rate RNN + BPTT 获得可靠行为阳性。它是任务教师和上限，不是更生物合理的终点。
3. **full-task LIF-STSP student：** 保留真实 sample/delay/rule/test 时序，STSP 从训练第一步就参与前向；不通过修改任务消除记忆难度，而通过更强的训练信号解决信用分配。推荐使用已训练的 Masse-STSP teacher 提供 output logits、低维 task-state 或 target recurrent currents，以 surrogate BPTT/target-based learning 训练 Dale 稀疏 LIF-STSP student，随后移除 teacher loss 并只用行为目标微调。

第三条不是生物局部学习，但保留了目标前向网络的 LIF、Dale、稀疏 E/I 和 edge-STSP。其诚实结论是“生物约束前向架构在全局训练下可行”。若进一步要求学习本身也生物合理，必须在同一 full-task、full-STSP 前向模型上另行开发 e-prop/三因子 eligibility；这将是独立的高风险方法贡献，不能由 scaffold 或 teacher distillation 代替。

teacher-student 训练不应逐神经元复制 rate teacher 的活动，因为 rate 与 spike 表示没有一一对应。更稳妥的是匹配行为 logits 与经固定/训练期线性投影后的低维 recurrent-current targets，并保留原始 task loss。最终机制证据必须来自 student 自身的 STSP reset、shuffle、donor、fast-state intervention 和完整任务泛化；若 student 只复现 teacher 输出但 STSP 干预无效，就只能报告任务蒸馏成功。

小型 student 仍不能替代 Tiddia 的大规模证据。它负责回答“LIF + Dale + edge-STSP 的完整任务前向是否可训练”；Tiddia 负责冻结大规模 substrate 后复核状态边界和因果效应。

## 6. 训练课程与持久化 DAG

所有 trial table、随机种子、图拓扑、checkpoint、state-reset 记录和评估结果都应作为可复用 artifact 持久化；plot-only 任务只能读取已有结果，不得重新模拟。

| 阶段 | 只新增一个主要变量 | 通过条件 | 失败时动作 |
|---|---|---|---|
| G0 kernel | 单 LIF、单突触滤波、单 edge TM update | 与解析/参考实现一致；有限梯度；reset 后无残留状态 | 停止网络扩展，修 kernel/state ledger |
| G1 direction tracer | Dale 稀疏 LIF，只做单方向识别；可分别运行 no-STSP 与 full-STSP | 当前输入方向识别正确；放电、E/I 电流、TM 更新和梯度稳定；至少 3 seeds | 若单输入仍失败，先修输入、loss、kernel 或动力学；不得进入双输入任务 |
| G2 full-task STSP student | 从第一步使用完整 sample/delay/rule/test 与 full-STSP；Masse-STSP teacher 只提供训练期 target/logit 信号 | 完整行为开始高于 chance；teacher loss 可逐步移除；student 的 STSP 干预获得行为效应 | 若仅 teacher matching 成功而 task/因果门失败，只报告蒸馏阴性，不回退零延迟双输入 scaffold |
| G3 edge-TM | shared-TM 替换为 edge-specific TM，图固定 | forward 等价性、显存、梯度和行为均可审计；与 shared 版本做 matched comparison | 可暂留 shared-TM 作为工程基线，明确 edge-TM 未通过；不跳到 10k |
| G4 architecture | E/I τm、固定短 delay 或窄异质分布，一次一个因子 | 单因素的行为和动态平衡均通过 | 回退到 G3；禁止用组合细节掩盖失败 |
| G5 scale/substrate | 500 full model，再做 Tiddia fixed 10k/20M forward/causal audit | full task、held-out trials、reset/donor/state shuffle 稳定 | 保留小网络结果；不要把大 substrate 失败写成 STSP 失败 |
| G6 local learning | 固定前向模型，BPTT 替换为 e-prop/其他在线规则 | 多 seed、在线跨 trial、无 BPTT 的行为与因果指标接近 | 只能报告“架构 + BPTT”或“近似局部学习”层级，不作机制主张 |

短 delay curriculum 不能替代最终固定任务；每个阶段都要保存 full-delay 的独立评估。相同 trial table 可用于 matched comparison，但训练和 held-out 评估必须区分。

## 7. 失败门、停止门和机制判定

以下是应该在实验开始前写入 protocol 的停止门：

1. **控制失败门**：无 STSP 的 Dale LIF 在短 delay 或 full task 都不能达到行为阈值时，停止增加 STSP、ALIF、异质 delay 和 Tiddia scale；问题优先归类为优化/动力学。
2. **慢状态混淆门**：只有加入 800 ms current、SFA/ALIF 或长滤波器才成功时，不得声称 STSP 支持记忆；必须做 state reset、state shuffle 和 matched slow-state ablation。
3. **可解码不等于使用**：若随机初始化的 x·u 已高度可解码，或 shuffle x/u 不改变行为，不能把 STSP state decoding 当成机制证据。
4. **detach 依赖门**：若只有 recurrent-spike-detach 版本成功，应报告为训练路径依赖；不得把 detach 解释成局部生物信用分配。
5. **edge-state 资源门**：edge-TM 使显存、梯度或 event replay 超过预算时，停止在小网络增加细节，保留 shared-TM engineering baseline，并把 edge-TM 作为明确的未完成边界。
6. **符号/稀疏门**：无约束网络成功而 Dale/稀疏网络失败，只能说明该架构约束提高了训练难度；不能据此断言生物网络不能完成任务。
7. **规则机制门**：没有完成规则 cue 操纵、multi-step successor 或必要性/充分性测试时，不能把短 DMS reset/swap 结果扩写为规则机制。
8. **局部学习门**：使用 RLS、全局 teacher 或 BPTT 的结果最多属于前两层；若没有在线局部更新和跨试次因果验证，不进入第三层机制叙述。

## 8. 必须单独审计的慢状态

每次 reset、donor swap、state shuffle 和 transfer 都应列出以下状态，不能只重置膜电位：

- formal baseline 的 800 ms synaptic current；
- SFA/ALIF 的适应变量（当前 SFA 约 400 ms；ALIF 可到数百毫秒至秒级）；
- STSP 的 x/u（目标机制）；
- synaptic filter 的电流状态（20 ms 通常是短时状态，但仍需在审计中列出；50–100 ms filter 可能有明显影响）；
- membrane voltage、refractory timer 和 delay ring buffer；
- trainable initial state；
- output/readout 的低通状态；
- 外部输入、rule cue 或 test grace period 中持续存在的信号；
- 优化器的 Adam/RLS state（它不是网络内存，但会影响训练可重复性；不应被混为生物机制）。

判定 STSP 作用时，最小充分组合应包括：同一 trial 输入下的 STSP reset、donor state swap、x/u shuffle、firing-only decode、以及 no-STSP matched control。当前历史 Tiddia 的短 DMS 正因果结果值得保留，但还需要规则操纵才能支撑更强结论。

## 9. 与 Tiddia 固定 10k/20M substrate 的结合边界

当前 Tiddia 配置约为 8000 E + 2000 I、连接概率约 0.20、约 20M recurrent edges，并且已经有 edge-level STSP、0.1–1.0 ms delay 和事件调度。它最适合承担“固定大规模基底上的状态因果审计”，而不是承担第一轮 task-gradient search。

建议采用三步接入：

### A. 小网络先学会同一个前向问题

在 128–256 → 500 的小型 Dale LIF 上固定任务、输入编码、STSP 类别和评估。先得到可复现的 task behavior、state reset 和 causal shuffle 指标；这里可以用 shared-TM 解决优化闭环，再换 edge-TM。

### B. Tiddia 作为冻结 substrate

把 Tiddia recurrent graph、权重、delay 和 edge STSP 状态冻结，只训练或校准外部 input projection/readout；保存所有 edge state 的 reset 和 donor metadata。这个阶段回答的是“大型固定 substrate 是否保留同类因果信号”，不是“网络是否通过任务训练学会了规则”。

### C. 仅在需要时做在线局部更新

如果 G6 需要进入局部学习，可在固定 Tiddia 图上尝试 e-prop/局部 BCD 或受限的低维 input/readout 更新。任何 recurrent edge weight 的更新都应记录 Dale 符号、局部变量、teacher/error 信号和是否跨 trial；RLS/teacher 只能作为训练诊断，不能当成局部 STSP 机制。

10k/20M 的 edge-specific TM 前向本身并不与目标冲突，真正的边界是全 trial BPTT 的 O(E) 状态/梯度成本和因果归因难度。不要一次同时打开 10k scale、Dale、异质 delay、异质 τ/U、edge-TM、ALIF 和 e-prop；那会让任何失败都无法定位，也无法说明新结论来自哪一个机制。

## 10. 预期科学贡献的最小表述

如果 G1–G5 成功，最小而可信的结论是：

> 在固定 Masse delayed-cue 行为任务上，带 Dale 符号、固定稀疏 E/I 拓扑、脉冲 LIF 和 edge-specific TM 状态的循环网络可以在 BPTT/替代梯度下训练；持续活动、STSP 状态和慢状态的行为贡献可以通过 matched reset/shuffle/ablation 分开测量。

这已经是“生物约束架构的可训练性”结果，但不是局部学习机制结果。如果 G6 进一步成功，才可以增加：

> 在同一前向架构上，eligibility-based/三因子在线学习在明确的反馈假设下近似 BPTT，并保留任务与因果指标。

只有在没有 BPTT/RLS、局部变量定义完整、规则操纵与必要性/充分性均通过时，才可使用“STSP 是该规则工作记忆机制”一类强表述。这样的分层会让 Masse baseline、当前 formal positive control、stripped training failure 和历史 Tiddia causal evidence 各自处于正确位置，也避免因为一次性追求全部生物细节而牺牲可训练性和可归因性。

## 参考的一手论文与作者/官方代码

1. Masse, N. Y. et al. (2019). *Circuit mechanisms for the maintenance and manipulation of information in working memory*. Nature Neuroscience. [论文](https://www.nature.com/articles/s41593-019-0414-3)；[作者代码](https://github.com/nmasse/Short-term-plasticity-RNN)。
2. Song, H. F., Yang, G. R. & Wang, X.-J. (2016). *Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework*. PLoS Computational Biology. [论文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004792)；[pycog](https://github.com/xjwanglab/pycog)。
3. Bellec, G. et al. (2020). *A solution to the learning dilemma for recurrent networks of spiking neurons*. Nature Communications. [论文](https://www.nature.com/articles/s41467-020-17236-y)；[作者代码](https://github.com/IGITUGraz/eligibility_propagation)。
4. Nicola, W. & Clopath, C. (2017). *Supervised learning in spiking neural networks with FORCE training*. Nature Communications. [论文](https://www.nature.com/articles/s41467-017-01827-3)；[ModelDB 代码存档](https://modeldb.science/190565)。
5. Ingrosso, A. & Abbott, L. F. (2019). *Training dynamically balanced excitatory-inhibitory networks*. PLoS ONE. [论文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220547)。
6. Tsodyks, M. & Markram, H. (1997). *The neural code between neocortical pyramidal neurons depends on neurotransmitter release dynamics*. PNAS. [论文记录](https://pubmed.ncbi.nlm.nih.gov/9012851/)；[论文 PDF](https://www.cns.nyu.edu/csh/csh04/Articles/Tsodyks-Markram-97.pdf)。
7. Mongillo, G., Barak, O. & Tsodyks, M. (2008). *Synaptic theory of working memory*. Science. [论文记录](https://pubmed.ncbi.nlm.nih.gov/18339943/)。
