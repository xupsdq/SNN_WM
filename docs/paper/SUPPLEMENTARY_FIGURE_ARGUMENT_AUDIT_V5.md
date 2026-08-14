# v5 补充图论证与统计审计

> **2026-08-04 科学逻辑更新：** 本文件保存 current S1–S7 的来源、统计与旧 deletion test；其正文映射已由 `docs/paper/revisions/v5_ai_review_20260802/MAIN_SUPPLEMENTARY_ARGUMENT_ARCHITECTURE_20260804.md` 重新裁决。current S2 的 primary L2 transfer 需部分提升到 Fig.3，current S5 不再属于必要补图，新增 C5 supplement 进入 Fig.4 支撑位，current S6 的 coefficient-free endpoint 需部分提升到 Fig.5。

## 1. 审计结论

本轮不删除任何一张补充图，也不删除 S2 的任何科学端点，而是将原先的 7 图、41 panel 候选重建为 **7 图、共 28 panel**。四个是下限而不是预设目标；S1、S3–S7 的四个字母来自逐项 deletion test，S2 则在进一步检查直接比较关系后，把原 b/c 两个同 DTI、同干预、同 cohort 的端点合并为一个两行 panel。每张图只保护一个当前正文结论，每个 panel 必须承担一种不可由相邻 panel 替代的作用：定义因果前提、建立主要效应、关闭分析选择、关闭采样或 null 替代解释，或明确可估性边界。

本文件记录“为什么这些 panel 有资格进入图”；已实施的 panel 设计、数值和结论计划归档于 `docs/archive/paper/supplementary-figure-history/SUPPLEMENTARY_FIGURE_PLAN_V5.md`。正文阻断项与本轮面板数纠偏另见 `docs/paper/SUPPLEMENTARY_FIGURE_DECISION_LOG_V5.md`。

### 统一准入条件

一个结果只有同时满足以下条件才可成为 panel：

1. **来源有效**：数值必须来自真实持久化输出或可复现的只读重算，不能来自占位值、机械缩放、硬编码 flag 或合成 placeholder null。
2. **端点一致**：声称稳健性时，必须重算正文同一端点；不能用 spike count 代替 transition composition，也不能用 loser-only 指标代替 winner-minus-loser contrast。
3. **推断单位正确**：独立重复为 20 个训练网络。trial、unit、event、sequence、pair、site、window、K×delay cell 都必须先在网络内汇总。
4. **统计通过**：claim-bearing contrast 的网络级 CI 必须支持目标方向，并通过该图预定 family 的 multiplicity control。工程 hard gate、coverage 和 fit calibration 不伪造 P 值，但必须是解释后续因果或可估性的必要条件。
5. **逻辑必要**：显著结果只有在关闭正文一个真实替代解释时才获得 panel 身份。重复正文、模型清单、纯绘图 QA 和不改变 terminal inference 的结果进入 Source Data 或图注。
6. **边界明确**：每张图必须同时写明其不能支持的升级结论。

## 2. 七张图与正文正式结论的映射

| 补充图 | panel 数 | 直接保护的正文结论 | 自然逻辑链 | 最终作用 |
|---|---:|---|---|---|
| S1 | 4 | Fig.1e：继承状态对 later processing 有内容方向的功能影响 | delay 衰减 → sample 方向 → trial-paired donor flux → donor-opportunity null | 排除固定 probe 扰动、分母变化和 donor 标签基率 |
| S2 | 4 | Fig.2：相同当前输入仍产生 history-conditioned successor transition | causal identity → 同轴 L2/early-L3 transfer → DTI robustness → untouched cohort | 补回跨层因果桥，并排除指标极值与 development seed 污染 |
| S3 | 4 | Fig.3c,d,f：局部 transition/competition 与 L1 STSP 干预 | 真实 window/双 comparator → winner cap → distance threshold → original-winner fate | 排除时间、事件上限、空间阈值和总体计数重分配 |
| S4 | 4 | Fig.4a：successive inputs 反复产生 observed-over-passive displacement | raw prevalence → worst-stage network endpoint → `u` trajectory → `x` trajectory | 排除少数 network/stage 和单一 STSP variable 驱动 |
| S5 | 4 | Fig.5c：pair-specific residual 超出灵活线性 constituent mixture | CV baseline → variable specificity → continuous input geometry → ordered class pairs | 排除 gross underfit、joint 尺度占优和采样子集伪影 |
| S6 | 4 | Fig.5f：Layer 1 `g` 存在 coefficient-free morphology boundary | full-map mean → entropy effective area → spatial topology → matched item composition | 证明 `g` 场收缩但未退化为空间白噪声或无内容公共模板，同时避开病态 NNLS coefficient |
| S7 | 4 | Fig.6f：STSP×overlap interaction 表达条件性 access | early window → q/threshold → coverage/availability → score-map shuffle | 排除时间、cutoff、可估性和任意空间排名 |

这张映射也限定了补充图的权利边界。例如 S6 只解释 Fig.5f 的 Layer 1 `g`，不能拿来解释 Fig.5d,e 的 Layer 2 `u/x`；S7 直接保护 Fig.6f，不能声称重新证明 Fig.6e 的 targeted-removal 因果性。

## 3. 原候选中的禁用证据

### S1

- `supp_phase_firing_rates.csv` 的 probe=0 是初始化占位：producer 没有执行 probe step，却仍写出 probe window。该列禁止作为科学证据。
- early/late delay firing=0 已被正文 Fig.1c 的 50-ms 全层轨迹更直接覆盖，不应再占 panel。
- 五种 substrate 条件不是匹配的单变量干预。`pure_substrate_only=true` 会先 reset 全状态再只恢复目标 substrate；不能把 accuracy/silence 拼盘解释成 substrate specificity。
- 逐数字 recall 虽全部高于 10% chance，但只证明任务性能，不证明 `u/x` 的功能作用逐类别普适。逐类 donor gain 经 Holm10 后无一显著，禁止写成 class-general functional effect。

### S2

- B-end class-score DTI 虽为正，但不属于原确认性 core family，且当前 Fig.2 contract 明确排除；不能为了凑 panel 将其升级。
- all-layer DTI=1、coverage=1 和 plumbing equality 不作生物学显著性结果。
- 正式推断必须沿用原确认性 family-8 Holm，而不能用事后缩小的 family-3 获得更小 P 值。

### S3

- `supp_s9_event_chain_null_summary.csv` 的五类所谓 null 并未执行五类置换。producer 实际从 `Uniform(0, observed)` 生成 null，因此 observed-minus-null 被构造为正；所有相关 P 值无效。
- `supp_early_window_robustness.csv` 没有重算 window。相同 transition label 被复用，数值只乘 `min(1, window/15)`；15/20/30 ms 完全复制。
- 旧 event-selection audit 与当前 canonical event cohort 不一致；“仅排除 27 行”的说法禁止沿用。
- 旧 radius panel 使用 loser inhibitory rise/suppressed fraction，不能验证正文 −8 至 −1 ms winner-minus-loser ΔV；suppressed fraction 又受 loser 预筛选定义影响。
- target/restoration flag 在 producer 中是硬编码记录，不是状态相等性检验。
- 旧 same-winner 文件使用 overlap-high-support intervention，与正文 Fig.3f 的全 L1 intervention 不同，不能作为正文干预的 robustness。

### S4

- `inference_long.csv` 中 exact sign-flip `P=0` 不可能。n=20 时最小单侧 P 为 `2^-20=9.5367×10^-7`，最小双侧 P 为 `1.9073×10^-6`；必须重算 Holm。
- `x` early-minus-late = −0.00252，CI 跨 0，`P_Holm=0.662`。这只能写成“未检出改变”，不能写成 `x` 恒定、等价或不衰减。
- “`u` 显著而 `x` 不显著”不等于两变量变化显著不同；也不能用二者绝对数值比较机制重要性。
- terminal equality 是 extraction QA，进入 audit table，不占 panel。

### S5

- 原方案把 in-sample `linear_mixture_r2` 误写成 CV `R²`；正式图必须使用 `cv_r2`。
- joint `ux_concat` 未标准化，target norm 约为 `u=7.586`、`x=0.1568`、joint=7.587，joint 数值实际由 `u` 尺度主导。不能声称 `u/x` 等权。
- residual norm ratio 非负是构造性质，对 0 的检验没有科学意义，也不能把 0.293 解读成“29% memory”。
- pair-count heatmap 只是设计 QC，不能排除 sampling confound；必须直接检验 residual specificity 在几何分层和 ordered class pair 中是否仍为正。

### S6

- peak/valley mask 由同一 `delta_gain_map` 的 top/bottom quantile 定义，再用同一 map 的 peak-minus-valley 作结果；旧 random-mask null 没有重复选择流程，100% structured 是选择诱导，禁止使用。
- `shuffled_order_null` 不是经验置换分布。producer 使用 cyclic shifts 加 reverse，且恒等满足 `null_mean=-(2/K)×true_corr`；所谓 true-minus-shuffle 只是 true correlation 的确定性缩放。
- linear 与 saturating 均为二参数，但只拟合跨 delay 边际化后的四个 K 点。即使 LOO error 更小，也不能证明统一饱和律或固定 capacity；800 ms 下 linear 反而更好。拟合结果进入 Source Data，不作为 panel。
- Gini 是有效的全像素分布指标，但只证明不均匀，不能单独证明空间组织。最终改用不依赖 peak selection 的 Moran's I，并以 sequence-matched singleton composite 补足内容特异性。
- 高 reconstruction `R²` 不能证明 NNLS coefficient 稳定。K10/D800 design condition number 为 9940.71 [9676.92,10204.51]，主要由 singleton 列范数差异造成；原 `p_i=β_i/Σβ_i` 又不对列缩放保持不变。列 L2 归一化后 condition number 约降至 3.21，但 `N_eff` 的 K 规律改变，raw order sign reversal 也消失。因此 raw `p_i`、raw `N_eff` 与 order reorganization 均不得承担 S6 panel 推断。

重建后的 S6 采用四个相互递进、完全 coefficient-free 的端点：full-map mean-`Δg` load×delay DiD=0.029684 [0.028601,0.030767]；entropy effective-area DiD=0.071743 [0.067519,0.075968]；K10/D800 Moran excess=0.76339 [0.75808,0.76869]；每网络 16-cell minimum singleton-composite matched-minus-sequence-deranged centered cosine=0.133676 [0.129798,0.137555]。四项均为 20/20 networks 同向，双侧 exact raw `P=1.907×10^-6`，Holm4 后 `P=7.629×10^-6`。依赖 foreground threshold 的输入-mask版本在径向去偏后不显著（`P=0.0764`），因此禁用。

### S7

- global-ping Q5−Q1 虽显著，但 Q2→Q3 存在稳定反向变化，Q5 spike probability 达 ceiling=1；“连续单调 score law”不成立。
- marginal real-probe Q5−Q1 不是正文 Fig.6f 的 overlap interaction，不能冒充同端点 robustness。
- real-probe score shuffle 后仍有小而稳定残差 0.00617 [0.00506, 0.00727]；不能笼统写“所有 effect 精确归零”。最终只保留 interaction endpoint，其 shuffle null 的 CI 跨 0。

## 4. 统计口径修正

1. **post-hoc robustness 默认双侧**：若无既有方向性 contract，使用双侧 exact paired sign-flip，并在每张图的 panel-level primary/conjunction endpoints 内做 Holm。
2. **保留原确认性 family**：S2 的两个 transfer endpoints 保留原 family-8；S4 joint stage endpoints保留当前正文 family。不能事后缩小 family。
3. **sweep 不制造重复**：window、radius、q、threshold、class pair 等不是独立重复。优先为每网络计算跨 sweep 的最小效应或 conjunction endpoint，再以 n=20 推断。
4. **不报告 P=0**：精确检验必须报告有限枚举下界。
5. **零方差 ceiling 不做 t test**：例如 reset 后 original-winner disruption=100%，描述 ceiling，并检验有变异的 attenuation 或 reset-minus-attenuation。
6. **gate/coverage 不伪显著**：hash identity、isolation、coverage 和 fit fidelity 是解释前提；报告 pass count、coverage、range 或 calibration，不检验无意义的“是否大于 0”。

## 5. 自然 panel 组合为何严密

每图采用同一种阅读结构：

`问题定义或最弱条件 → 主要效应 →  surviving alternative → closure control`

- S1 从时间衰减推进到内容方向，再把聚合结果还原为 paired trial flux，最后以 donor-opportunity null 关闭标签基率。
- S2 先把 exact-B、L1-only isolation 与 coverage 锁为一个 causal-identity gate；随后在同一 DTI 坐标内依次读取 Layer 2 state 与 early Layer 3 readout；c 排除 DTI 极值/正交位移/少数 comparison，d 以 untouched n=19 cohort 关闭 development seed 污染。
- S3 先以真实重裁切和双 comparator 检查 transition，再检查 winner 数量上限与空间定义，最后回到同一全 L1 因果操作。
- S4 先给 raw network×stage prevalence，再给合法的 worst-stage network endpoint，之后分解 `u` 与 `x`。
- S5 先证明 baseline model 可泛化，再检验变量拆分，最后连续和离散两类 sampling closure。
- S6 先用全图 `mean(Δg)` 确立不依赖 coefficient 的更新收缩；再用对整体缩放不变的 entropy effective area 排除 uniform fading；Moran's I 证明最弱 K×delay cell 仍有局部空间拓扑；最后 matched-vs-deranged 等权 singleton composite 关闭公共中央偏置和平滑模板解释。
- S7 依次排除时间、cutoff、低 coverage 和任意 score-map 排名。

因此“四个字母”仍不是版式目标。S2 的 untouched n=19 confirmatory inference 必须保留，删除后会重新打开开发污染；但它现在作为 d 独立存在，而 L2 与 early-L3 primary endpoints 因直接比较义务共同组成 b。其余图的第五候选经 deletion test 后只会重复同一 endpoint、承担工程 QC、依赖非显著结果或使用无效来源，故不升级。

### 面板数裁决摘要

- **S1=4：**class gain 不显著，phase probe 未记录，工程 gate 可进入图注。
- **S2=4：**b 保留两个确认性 transfer rows；c/d 分别关闭指标和 cohort 两类独立问题。减少的是重复坐标字母，不是证据任务。
- **S3=4：**event flow/target gate 属于 Methods；fake null、机械 window、旧 cohort 与错协议文件禁用。
- **S4=4：**同一 `u` trajectory 已同时承载 terminal persistence 与 paired attenuation；拆出第五字母不新增推断。
- **S5=4：**model、variable、continuous geometry、categorical pair 已闭合；norm/count 只是校准。
- **S6=4：**amplitude、area、topology、history identity 恰为四个任务；额外 adjacency 属同一 sensitivity，foreground mask 未通过。
- **S7=4：**availability 与 coverage 同属 estimability，site count/nonzero fraction进入图注/Source Data；禁止用 inset 制造 micro-panel。

## 6. 冻结前硬阻断项

### 必须同步修正正文 Fig.3c

当前 `panel_b_transition_summary_by_group.csv` 的 `transition_type` 使用整段 trace 的 first spike；`early_window_ms=15` 只是标签，并未对 transition 做 15-ms 截尾。v5 当前列出的 overlap−random advance=13.56 pp、recruit=12.46 pp 不是严格 first-15-ms 值。

从持久化 first-spike 数据按 `<15 ms` 真实重裁切后：

- advance：7.839 pp [7.610, 8.068]；
- recruit：11.631 pp [11.466, 11.796]。

方向与显著性不变，但正文数值或时间定义必须在补图冻结前同步修正；不能让补图使用正确窗口而正文继续保留错误标签。

### 必须同步限定正文 Fig.5f

Fig.5f 当前 `N_eff_fraction` 基于未标准化 singleton columns 的 NNLS coefficient。raw design condition number 的 median 为 13.37、P95 为 8593.7；K10/D800 mean=9940.7，主要来自约 8010.9 的 singleton 列范数比。列 L2 归一化后 condition number 约降至 3.21；高负荷/长延迟的一个 joint contrast 仍显著，但 K3/5/7/10 loss 不再单调，K3/D800 从原 0.9947 变为 0.6215，raw K10/D800 serial correlation `−0.652` 甚至变为贡献权重下 `+0.687`。

因此当前 raw-NNLS Fig.5f 仍是冻结阻断项。两条合格路径只能二选一：①完成 column-normalized、正则化和 coefficient-bootstrap sensitivity 后，极度收窄 NNLS 解释；②把正文 panel 改成 S6 所采用的 coefficient-free morphology boundary。仅凭高 `R²` 或一次列归一化，不能继续声称稳定的 component capacity、短 K 近完整保持、统一负荷律或 order reorganization。

### 必须物化的 derived Source Data

下列 panel 是合法的只读重算，但正式出图前应将 network-first derived table 和分析脚本物化，供复现与审稿：

- S1c,d：trial-paired donor flux 与 batch/label opportunity null；
- S2c,d：DTI distribution/direction robustness 与 untouched confirmatory cohort；
- S3a–d：winner cap、真实 window、distance threshold 与 original-winner fate；
- S5c,d：geometry quartiles 与 90 ordered class pairs；
- S6a–d：full-map mean、positive-`Δg` entropy effective area、rook Moran's I，以及 equal-weight singleton matched-vs-sequence-deranged centered cosine；
- S7c：主协议四组 site count/nonzero fraction 与 q×threshold complete-case coverage；
- 所有新 post-hoc family 的 exact sign-flip/Holm 表。

这些都是现有数据的重分析，不需要新训练或新模拟。
