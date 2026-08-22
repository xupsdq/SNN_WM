# 循环 STSP delayed-match-to-sample 行为—机制实验合同

**状态：** OPEN / PILOT CONTRACT

**角色：** 替代“仅凭 matched-query 内部状态投影推广论文结论”的不足；现有 matched-query 仅作为检查点与 STSP 干预的技术底座。
**推断边界：** pilot 用于判断任务和端点是否可行，不用于选择性汇报或确认性网络级推断。

## 1. 要闭合的科学命题

独立循环脉冲网络必须先在一个真正依赖过去信息的经典行为范式中给出可判定答案，随后才能询问该答案是否由论文提出的连续 STSP 演化机制支持。完整证据链为：

```text
经典 delayed-match-to-sample 行为成立
        ↓
延迟期 sample 信息在放电中弱、在 STSP 中可读
        ↓
相同 probe 的响应依赖继承 STSP
        ↓
STSP reset 破坏行为，donor substitution 定向改变答案
        ↓
干预 STSP(t) 定向改变下一时间段 firing(t+1)
        ↓
firing(t+1) 进一步写入 STSP(t+1)，形成可继续传播的状态
        ↓
STSP(t+1) 可服务后续 probe
```

任何只证明其中一个内部状态距离、但没有独立测试集行为端点的结果，不进入跨模型一般性结论。

## 2. 外部范式依据

主范式采用 distracted object delayed-match-to-sample（DMS），任务和分析骨架参考：

- Kozachkov et al. (2022), *Robust and brain-like working memory through short-term synaptic plasticity*, PLOS Computational Biology：可变延迟、50% 中途干扰、任务输出，以及分别从放电和 STSP 解码 sample；论文和代码分别见 <https://doi.org/10.1371/journal.pcbi.1010776> 与 <https://github.com/kozleo/robust_wm_stsp>。
- Mongillo, Barak & Tsodyks (2008), *Synaptic Theory of Working Memory*：循环脉冲网络中以突触易化维持并由后续输入读取工作记忆；见 <https://doi.org/10.1126/science.1150769>。
- Wolff et al. (2017), *Dynamic hidden states underlying working-memory-guided behaviour*：相同 impulse/probe 的状态依赖响应可读取 activity-silent hidden state；见 <https://doi.org/10.1038/nn.4546>。

本实验借用的是经典任务逻辑、标签平衡和解码端点，不移植上述工作的网络架构或训练逻辑。循环连接、神经元、固定长期权重和 Tsodyks STSP 均使用本项目已经验证的 10k/20M PyTorch 内核。

## 3. 冻结任务定义

### 3.1 Trial

```text
sample A（population 0–3）
        ↓
variable delay
        ↓ 50% trials: task-irrelevant distractor（population 4）
probe B（population 0–3）
        ↓
独立线性读出：match(A=B) / non-match(A≠B)
```

- `sample` 与 `probe` 身份在每个 split、delay、distractor 条件和标签内平衡；
- 每个 probe 在 match/non-match 中出现次数相同，因此 probe 当前输入本身不能预测标签；
- match/non-match trial 使用成对的背景与未来 probe 随机流；
- train、validation、test 由随机流/replicate 隔离，行为读出只在 train 拟合、validation 选择正则化、test 冻结评估；
- 循环网络和长期连接权重完全冻结，只允许训练独立线性读出。

### 3.2 时间与条件

- sample cue：75 ms；
- 预设 delay：125、375、750 ms；
- distractor：位于 delay 中部，持续 75 ms，50% trial 存在；
- probe response：probe 起点后的三个 50 ms bins；
- silent readout：probe 前最后 50 ms；
- 行为特征：五个选择性群体的 probe-evoked binned firing；
- STSP 特征：probe 前和 probe 后按 plastic source/target population 聚合的连续 `u`、`x` 与 hypothetical next-spike release。

较短的时间尺度是当前 STSP 参数（尤其 `tau_fac=1500 ms`、`tau_rec=200 ms`）下对经典 DMS 结构的模型尺度映射，不声称等同于人或 NHP 的绝对秒数。

## 4. 预先声明的端点

### 4.1 行为门

主要端点为独立 test split 的 balanced accuracy。必须同时报告：

1. dynamic-STSP 行为 balanced accuracy；
2. 按 delay 和 distractor 分层的 accuracy；
3. 仅用 probe identity 的负对照；
4. query 前将 plastic state 置回基线的 STSP-reset accuracy；
5. 所有 trial，包括静默、失败和无响应 trial。

pilot 可行性门预先设为：dynamic test balanced accuracy ≥ 0.60，且高于 probe-only control。该门只决定是否进入多网络确认，不是论文效应阈值。

### 4.2 Activity-silent 门

在相同 test split 上比较 sample identity 解码：

- probe 前最后 50 ms 的 population firing；
- 同一边界的连续 STSP features。

必须报告两者的冻结 test accuracy，而不能只展示可视化分离。目标模式是 STSP sample decoding 高于 firing decoding；“低 firing rate”本身不等同于“firing 不含信息”。

### 4.3 因果门

- **sham：** 恢复同一 trial 的 STSP，答案应保持；
- **reset：** 仅将 `u/x/last_spike_time` 置于 query 边界基线，保持膜状态、突触电流、延迟环和未来输入 RNG；
- **donor substitution：** 在相同 probe、delay、distractor 与未来 RNG 下，在 match 与 non-match sample 间互换 STSP；冻结行为 decision margin 应沿 donor 标签方向移动；
- **targeted removal：** 仅重置由 task-selective source 形成的相关 plastic pathways，效应应强于 matched-random edge removal。

accuracy 下降只能证明功能必要性；donor-directed margin movement 才承担状态内容具有定向因果作用的证据。

### 4.4 Fig.4 连续机制门

必须在预先冻结的连续时间窗内直接检验以下有方向的链条：

1. 在时间窗 `t` 只替换或重置 STSP，保持神经元状态、延迟事件和未来输入不变；
2. 干预必须定向改变紧邻的下一时间窗 firing，而不能只改变 trial 终点的行为读出；
3. 该 firing 差异必须进一步形成可分离的下一层 STSP；
4. 对下一层 STSP 再做隔离干预时，必须定向改变再下一个时间窗的 firing；
5. 以上链条在第二个完全平衡的 probe 中仍能改变行为读出。

只证明 pre-query STSP 对最终判断有因果作用，或 post-query STSP 可以解码标签，均不足以证明 Fig.4 的逐段连续演化机制。

## 5. 对照和泄漏防护

- 每个 split 内正负标签严格平衡；
- sample、probe、delay、distractor 与标签机会平衡；
- 预处理统计量只由 train 估计；
- validation 只选择预先给定的 ridge 正则化；
- test 不参与特征、窗口、阈值或参数选择；
- probe-only、sample-only 和 trial seed 不得进入行为特征；
- 所有低层 trial 先在网络内聚合，确认性推断单位为独立连接图；
- 不根据 pilot delay 曲线事后声称单调性。

## 6. Artifact DAG

```text
persisted connectivity + frozen experiment_config
                    ↓
             balanced trial_manifest
                    ↓
          per-network trial features
                    ↓
       frozen decoder + control predictions
                    ↓
       within-network endpoint summary
                    ↓
          network-level statistics
                    ↓
             plot-only leaf task
```

模拟任务不得隐式拟合 decoder；分析任务不得重跑模拟；绘图任务只读取已经持久化的统计和 Source Data。

## 7. 结果解释规则

- **行为阴性：** 当前固定回路没有证明 DMS 功能，一切 STSP 内部投影仅保留为技术结果；
- **行为阳性、reset 无效：** 任务可能由残余活动或其他状态完成，不能归因于 STSP；
- **行为与 reset 阳性、substitution 无方向：** STSP 可能提供增益，但没有证明携带任务内容；
- **前三者阳性、Fig.4 连续机制门未做或阴性：** 只支持 STSP 对单步 activity-silent readout 的因果贡献，不支持论文的逐时间段连续演化机制；
- **全部闭合并跨独立图重复：** 才能主张同一连续 STSP 机制在独立循环网络和经典行为任务中得到推广。
