# 循环 STSP delayed-match-to-sample 短延迟初步结果

**状态：** PILOT / NON-CONFIRMATORY

**用途：** 检验固定 10k/20M 循环脉冲网络是否存在“经典行为成立—静默 STSP 可读—STSP 因果改变行为”的最小桥梁。
**边界：** 单连接图、125 ms delay、无 distractor、12 个冻结 test trials；不能代表完整可变延迟/干扰 DMS，也不是网络级推断。

## 1. 为什么替换此前的证据层级

此前 matched-query pilot 证明了 STSP 状态可被写入、替换，并使相同 query 后的 firing 与 trial 终点 post-query STSP 朝 donor 移动；但它没有要求模型完成一个关于过去信息的行为判断。因此该结果只能作为 checkpoint 和 STSP-only 干预的操作验证。

本实验先要求网络完成 delayed-match-to-sample：给出 sample A，经过 delay 后给出 probe B，独立线性读出只能从 probe-evoked firing 判断 `A=B`。循环权重和长期连接完全冻结，只有行为读出在 train split 拟合。

## 2. 冻结短 tracer

| 项目 | 设置 |
|---|---:|
| 网络 | 10,000 neurons / 20,000,000 edges |
| plastic edges | 12,800,000 |
| task populations | 0、1、2 |
| sample cue | 50–125 ms |
| delay | 125 ms |
| probe onset | 250 ms |
| response features | probe 后 3 × 25 ms population-rate bins |
| distractor | absent |
| train / validation / test | 12 / 12 / 12 trials |
| test labels | 6 match / 6 non-match |
| 每个 test probe 的标签机会 | 2 match / 2 non-match |
| 每个 test sample 的标签机会 | 2 match / 2 non-match |
| decoder | train-only standardization；validation-only ridge λ selection；frozen test |
| CUDA elapsed | 1,962.84 s |

match 与 non-match pair 使用相同 probe、timing、background seed 和未来 probe RNG。probe identity、sample identity、trial seed 与 STSP 数值均不进入行为 decoder。

## 3. 行为先成立

| 行为端点 | Frozen test balanced accuracy |
|---|---:|
| Dynamic-STSP probe-evoked firing decoder | 1.000 |
| Probe-identity-only control | 0.500 |

probe-only control 精确位于机会水平，而 dynamic firing readout 在未见随机流上正确区分 12/12 trials。选择的 ridge λ 为 0.1；validation 上 λ=0.001、0.01 和 0.1 均达到 1.0，预设 tie-break 选择其中正则化最强者。

这是行为可行性证据，而不是最终性能估计：test 只有 12 个 trial，置信区间很宽，且没有覆盖更长 delay 或 distractor。

## 4. 延迟期信息位于哪里

使用完全相同的 train/validation/test 边界解码 sample identity：

| Probe 前端点 | Frozen test balanced accuracy |
|---|---:|
| 最后 50 ms population firing | 0.250 |
| 连续 STSP `u/x/next-release` | 1.000 |

三个 sample 的机会水平为 0.333；firing decoder 没有在 test 泛化，而 STSP decoder 达到 1.0。所有选择性群体与 trial 的平均 delay firing rate 为 0.565 Hz。

因此该 tracer 支持“延迟期 sample 内容在所测群体放电中不可稳定读取，但在连续 STSP 状态中可读取”的 activity-silent 模式。它不证明所有神经元尺度和所有非线性 firing decoder 都不含信息。

## 5. STSP 是否只是相关状态

test trial 从同一个 query 前 checkpoint 分为：

- `dynamic_sham`：恢复完整 recipient state 并保持自身 STSP；
- `reset`：仅把全部 plastic edge 恢复到当前时间的 no-event baseline；
- `swap`：保持 recipient 膜状态、突触电流、延迟环和未来输入 RNG，只换入相反标签 paired trial 的全部 STSP。

| 因果端点 | 结果 |
|---|---:|
| Dynamic test balanced accuracy | 1.000 |
| STSP-reset test balanced accuracy | 0.583 |
| Reset accuracy change | −0.417 |
| Swap prediction accuracy against donor label | 0.917 |
| Mean reciprocal donor projection | 0.987 |
| Mean donor-directed decision-score change | +1.233 |

reset 表明 trial-evoked STSP 对当前行为读出具有功能贡献；reciprocal swap 进一步表明这种贡献不是无内容的全局增益，因为只替换 STSP 会把冻结 decision score 几乎完整地推向相反标签 donor 的 intact endpoint。

## 6. Probe 后 STSP 端点

从 dynamic branch 的 probe 后连续 STSP 解码 match/non-match，冻结 test balanced accuracy 为 1.000。这说明 probe 后 STSP 已包含 sample×probe 的关系信息，而不只是 probe identity。

该端点只证明 trial 终点的 STSP 含有关系信息。当前没有在中间边界隔离干预该 STSP 并测量紧邻下一时间窗的放电，也没有证明该放电继续写入下一层 STSP，因此它既不能确认 Fig.4 的 `STSP(t) → firing(t+1) → STSP(t+1)` 链条，也不能写成 successor 已被后续行为复用。

## 7. 本轮可以说与不可说

当前可以作出的单图描述性判断：

1. 固定循环脉冲网络能够在平衡 DMS tracer 中用 probe-evoked firing 回答关于过去 sample 的问题；
2. sample 在延迟末端可从 STSP 而非所测 population firing 稳定解码；
3. STSP reset 实质性降低行为；
4. reciprocal STSP substitution 将行为答案定向推向 donor；
5. probe 后 STSP 包含新形成的 sample×probe 关系信息。

当前不得声称：

- 已复现带可变延迟和 distractor 的完整经典 DMS 结果；
- 125 ms 的模型 delay 等同于人或 NHP 的秒级 delay；
- activity-silent 结论对所有 firing 特征和非线性 decoder 成立；
- STSP 是唯一机制；
- 已验证 Fig.4 的逐时间段连续演化机制；
- successor 已被下一输入复用；
- 结果在独立连接图间稳定；
- 该 pilot 可直接作为稿件确认性主结果。

## 8. 必须继续关闭的两层证据

### 8.1 经典范式范围

在不修改窗口和端点的前提下，联合训练一个 decoder 并覆盖：

- delays：125、375、750 ms；
- distractor：absent / present；
- 每个 condition 内独立平衡 sample、probe 和标签；
- dynamic、reset 与 donor-swap 的同一组 test endpoints。

目标不是事后寻找单调曲线，而是判断任务在预设 operating regime 内能否保持高于机会水平，以及 distractor/long delay 是否造成可重复的性能负担。

### 8.2 Fig.4 连续演化而非一次行为读取

先把第一个 probe 后过程切分为预先冻结的连续时间窗，再增加一个平衡的第二 probe：

1. 在每个边界保存 STSP、神经元状态、延迟事件和外部 RNG；
2. 仅干预 `STSP(t)`，检验紧邻的 `firing(t+1)` 是否沿 donor 定向移动；
3. 检验该 firing 差异是否继续写入可分离的 `STSP(t+1)`；
4. 再干预 `STSP(t+1)`，检验 `firing(t+2)`，闭合至少两跳连续链；
5. 保持第二 probe 和未来 RNG 相同，检验第二次行为 decision margin 是否沿 successor donor 定向移动；
6. 对 task-selective pathway 做 targeted removal，并与 matched-random edges 比较。

只有逐时间窗的 STSP→下一段放电→下一层 STSP 链、第二 probe reuse 与多连接图重复都成立，才完成“经典行为由 Fig.4 连续 STSP 机制实现”的推广闭环。

## 9. 可再生入口

```text
python -m src.experiments.recurrent_stsp.run dms-workflow \
  --connectivity tmp/recurrent_stsp_full_graph.pt \
  --output-directory tmp/recurrent_stsp_dms_pilot_10k_short \
  --device cuda \
  --task-populations 0,1,2 \
  --distractor-population 4 \
  --delays-ms 125 \
  --distractor-mode absent \
  --pairs-per-probe 2,2,2 \
  --response-bin-ms 25 \
  --response-bin-count 3 \
  --stsp-edges-per-source-population 512
```

临时原始产物位于 `tmp/recurrent_stsp_dms_pilot_10k_short/`。`tmp/` 不是正式科学权威；多网络阶段必须写入版本化 results bundle，并保留 trial manifest、graph identity、decoder 和网络内 summary。
