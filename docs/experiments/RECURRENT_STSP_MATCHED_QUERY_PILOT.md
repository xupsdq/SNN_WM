# 循环 STSP matched-query 因果替换初步实验

**状态：** PILOT / NON-CONFIRMATORY

**用途：** 验证独立循环网络中，继承 STSP 是否能够调制相同新输入，并改变 trial 终点的 post-input STSP 状态。
**推断边界：** 以下结果均为单连接图描述性结果，不构成网络级统计、行为任务成功或稿件最终证据。

## 1. 操作定义

实验比较两个历史群体 `H0` 与 `H1`，随后向完全相同的 query 群体 `Q2` 输入相同 Poisson cue。

```text
历史 H0 或 H1
      ↓
query 前边界 checkpoint
      ↓
仅替换 12.8M plastic edges 的 u/x/last_spike_time
      ↓
保持 recipient 膜状态、突触电流、延迟环和未来 Poisson RNG 不变
      ↓
共同 query Q2
      ↓
query-evoked firing 与 trial 终点 post-query STSP
```

四个分支为：

- `history0_from_stsp0`：H0 recipient + H0 STSP sham；
- `history1_from_stsp1`：H1 recipient + H1 STSP sham；
- `history0_from_stsp1`：H0 recipient + H1 donor STSP；
- `history1_from_stsp0`：H1 recipient + H0 donor STSP。

donor projection 定义为 swapped endpoint 沿 intact recipient→intact donor 轴的投影比例；0 表示没有 donor-directed movement，1 表示到达 intact donor endpoint。该指标分别用于五个选择性群体的 query-window firing-rate 向量和全 plastic graph 的 hypothetical next-spike release 向量。

## 2. 冻结 pilot 时序

| 事件 | 时间 |
|---|---:|
| 历史 cue 起点 | 50 ms |
| 历史 cue 终点 | 125 ms |
| delay firing readout | 200–250 ms |
| 共同 query 起点 | 250 ms |
| query cue 终点 | 325 ms |
| query response readout | 250–400 ms |
| trial 终点 | 400 ms |

背景、cue 振幅、Poisson 等价公式和连接分布保持上游 heterogeneous 配置；两个历史使用共同随机数设计。

## 3. 1k 缩放 tracer

缩放网络包含 1,000 个神经元、194,400 条连接和 122,400 条 plastic 连接。由于各 fixed-indegree block 在小尺度下独立取整，边数不等于简单的 200k 缩放值。

| 端点 | 结果 |
|---|---:|
| delay population rates | 8.0–14.5 Hz |
| intact response L2 distance | 0.6922 Hz |
| mean reciprocal spike donor projection | −0.2174 |
| mean reciprocal successor donor projection | 0.4982 |
| pilot direction supported | false |

该缩放网络的 post-query STSP 向 donor 明显移动，但 delay 仍存在较强放电，query firing 未向 donor 移动。因此 1k tracer 只证明实验代码和 post-query endpoint 可工作，不支持静默 STSP 对放电的完整机制方向。

## 4. 10k/20M 初步结果

完整网络包含 10,000 个神经元、20,000,000 条连接和 12,800,000 条 plastic 连接。运行使用 CUDA float32，共两个历史前缀和四个因果分支，耗时 161.95 s。

| 端点 | 结果 |
|---|---:|
| delay population rates | 0.275–0.825 Hz |
| intact response L2 distance | 0.1816 Hz |
| H0 recipient ← H1 STSP firing projection | 0.0653 |
| H1 recipient ← H0 STSP firing projection | 0.2653 |
| mean reciprocal firing projection | 0.1653 |
| H0 recipient ← H1 STSP successor projection | 0.9170 |
| H1 recipient ← H0 STSP successor projection | 0.9228 |
| mean reciprocal successor projection | 0.9199 |
| pilot direction supported | true |

共同 query 群体 Q2 在四个分支中的平均放电率为 9.367–9.550 Hz，其他选择性群体为 0.092–0.183 Hz。由于未来 Poisson RNG 与 recipient 的非 STSP 状态保持一致，swap–sham 差异可归属于被替换的 STSP 状态及其后续循环效应。

## 5. 当前可说与不可说

当前单网络结果支持以下描述性判断：

1. 完整尺度循环网络在 query 前保留了可分离的 STSP release state；
2. 仅替换 STSP 会使相同 query 后的 trial 终点 STSP 强烈、双向地朝 donor history 移动；
3. query-evoked firing 也出现双向 donor-directed movement，但幅度明显弱于 post-query STSP endpoint movement；
4. 该方向在 1k 缩放网络中不成立，说明完整规模动力学不能由小网络直接替代。

当前不得声称：

- 已证明 delayed-recognition 行为；
- 已证明 activity-silent memory 的确认性阈值；
- donor projection 在独立网络间稳定；
- STSP 是该效应的必要或唯一机制；
- 已验证 Fig.4 的 `STSP(t) → firing(t+1) → STSP(t+1)` 连续链；
- 该 pilot 可以直接进入稿件主结果。

## 6. 后续确认路径

下一阶段应在不改变本 pilot 指标的前提下：

1. 加入真正依赖过去信息的 balanced delayed-recognition trial manifest 和独立线性解码器；
2. 用多个 graph seeds 先在网络内聚合 trial，再以网络为推断单位；
3. 增加 no-memory/state-reset、same-history sham 和 matched-random edge controls；
4. 在连续边界仅替换 `STSP(t)`，依次检验紧邻 `firing(t+1)`、形成的 `STSP(t+1)` 及其对 `firing(t+2)` 的作用；
5. 将 post-query STSP substitution 延伸到下一次共同输入，直接检验后续复用；
6. 冻结确认性端点、方向、最小有意义效应和 multiplicity family 后再运行约 20 个独立网络。

## 7. 可再生入口与临时产物

实现入口为：

```text
python -m src.experiments.recurrent_stsp.run matched-query \
  --connectivity <persisted-graph.pt> \
  --output-directory <output-dir> \
  --device cuda
```

本次临时运行产物位于 `tmp/recurrent_stsp_mechanism_pilot_1k/run/` 和 `tmp/recurrent_stsp_mechanism_pilot_10k/`。`tmp/` 不是科学权威；正式多网络运行必须写入版本化 results bundle，并保留连接、配置、分支和聚合 manifest。
