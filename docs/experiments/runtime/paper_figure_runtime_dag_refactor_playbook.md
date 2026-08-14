# Paper Figure Runtime DAG Artifact Refactor Playbook

本文档固化当前 paper figure 实验重构的通用成功流程。它不再是某一个 figure 的专属方案，而是用于后续所有 paper figure、补充图和相似计算实验的 runtime DAG / artifact 化方法论。

已验证经验来源：

- Fig.1：证明了 DMS / delay / control 类任务可以从线性流水线拆成可恢复的计算节点。
- Fig.2：证明了 pair specs、state bank、mask specs、completion boundary bank 等多类 artifact 可以同时进入同一个 figure-local DAG，并且下游 task 可以在 `require` 模式独立恢复。

核心结论：

```text
不要把实验视为必须从头到尾执行的一条脚本流水线。
要把实验拆成一棵由计算节点、artifact 和依赖边组成的 DAG。
只要父节点 artifact 已存在且通过 cache key / manifest / hash 校验，
子节点就可以单独运行，不应重跑上游 simulation 或 state capture。
```

这套流程的第一目标不是“文件变多”或“代码显得更平台化”，而是让实验具备：

- 可局部重跑；
- 可复用中间结果；
- 可追踪输入输出；
- 可验证结果不变；
- 可限制 AI 只修改某个节点边界内的代码；
- 可逐步向平台化演进。

## 1. 适用范围与非目标

适用范围：

- `src/experiments/paper_figures/fig*/` 下的主图和补充图实验。
- 具有共享上游 state / feature / sequence / mask / trial specs 的实验。
- 单个下游 readout、control、shuffle、sweep 或 metric 失败后，希望从该节点恢复的实验。
- 需要 AI 辅助重构，但必须保护 scientific protocol 的实验。

暂不适用：

- 尚未确认 scientific protocol 的探索性 notebook。
- 一次性小统计脚本。
- 需要立即跨 figure 全局调度的新平台。
- 需要修改 STSP / spike encoding / simulation timestep 数值逻辑的性能重构。

明确非目标：

- 不在第一轮重构中追求全局平台化。
- 不为了抽象而改 scientific protocol。
- 不把 plot 层变成 compute 层。
- 不在没有 regression 的情况下改变 sampling、mask、delay、state capture、restore 或 readout endpoint。

## 2. 三阶段路线

```text
阶段 1：figure-local DAG artifact 化
  每个 figure 先拥有自己的 task、artifact、cache key、manifest 和验证路径。

阶段 2：跨 figure shared/common 提取
  只把已经在多个 figure 中稳定重复、语义完全一致的机制提取到 common。

阶段 3：平台化 runtime
  建立 global task registry、scheduler、artifact store、cache resolver 和 profiling layer。
```

当前推广重点是阶段 1。阶段 1 的成功标准是：

```text
实验可以从任意依赖已满足的下游节点恢复运行，
并且输出与原始 fresh / legacy smoke 在守门容差内一致。
```

只有多个 figure 都完成阶段 1 后，才适合做阶段 2 / 阶段 3。过早抽象平台会把尚未确认的 figure-specific protocol 错误固化。

## 3. 通用 DAG 节点模型

每个实验节点必须被归入下面一种类型。命名可以不同，但边界必须清楚。

| 节点类型 | 当前作用 | 典型输入 | 典型输出 | 是否可重跑 | 是否必须 artifact 化 |
|---|---|---|---|---|---|
| source-of-truth specs | 决定样本、trial、sequence、pairing、mask seed、condition grid | config, seed, dataset | CSV/JSON specs | 只允许显式 producer 重建 | 是 |
| heavy reusable bank | 昂贵 simulation、state capture、feature extraction、boundary snapshot | specs, model, dataset | NPZ/CSV/JSON shards + manifest | 不应被下游隐式重跑 | 是 |
| deterministic spec expansion | 从 source specs 派生 mask/sweep/job table | parent specs, config | mask specs / job specs | 可显式重建，但 require 不允许重建 | 是 |
| downstream compute | 读取 bank/spec 做 readout、control、shuffle、sweep | parent artifact | raw/metrics CSV | 可单独运行 | 输出必须记录 |
| aggregation/statistics | 读取 raw/metrics 生成 summary 或 claim table | downstream raw/metrics | metrics/summary CSV/JSON | 可单独运行 | 输出必须记录 |
| panel sink | plotting adapter / renderer 消费结果 | raw/metrics/summary | panel data / figure | 可单独 check-only | 否，不是 compute 节点 |

判断一个步骤是否应该成为 artifact producer，优先问：

1. 重跑它是否可能改变下游结果？
2. 它是否被多个下游 task 复用？
3. 它是否昂贵，或者是失败后最不想重复运行的步骤？
4. 它是否定义了 sampling / sequence / mask / boundary / state 的身份？

只要任一答案为“是”，就应优先 artifact 化，而不是继续藏在 monolith 里。

## 4. 成功结构的标准形态

一个合格的 figure-local DAG 应该能画成：

```text
source-of-truth specs
  |
  +-- heavy reusable bank
  |     |
  |     +-- downstream readout/control A
  |     |     |
  |     |     +-- panel / supplement sink
  |     |
  |     +-- downstream readout/control B
  |           |
  |           +-- panel / supplement sink
  |
  +-- deterministic spec expansion
        |
        +-- downstream sweep/readout C
              |
              +-- panel / supplement sink
```

允许一个 figure 有多个 heavy bank。允许一个下游 task 同时依赖多个 parent artifact。关键是每条依赖边都必须显式化：

```text
child task
  consumed artifacts:
    - parent task id
    - artifact path
    - cache key digest
    - manifest/hash
  produced outputs:
    - data/raw/...
    - data/metrics/...
```

不合格结构：

```text
child task 调用旧 monolith 私有函数
child task 发现 artifact 缺失后静默重跑 parent simulation
plot adapter import experiment code 并触发 compute
同一份 mask/trial/sequence 在 standalone rerun 时重新随机生成
```

## 5. 文件与目录约定

每个完成阶段 1 的 figure 最少应形成：

```text
src/experiments/paper_figures/figX/
  __init__.py
  registry.py        # figure id, task list, main/supp scope, legacy bridge metadata
  schemas.py         # task ids, reuse modes, artifact schema, manifest columns
  types.py           # FigXConfig, ExperimentContext, shared dataclasses
  cache_keys.py      # stable cache key, digest, file/table hash
  artifacts.py       # save/load/validate artifact
  run.py             # figure-level compatibility wrapper
  run_task.py        # task-level CLI and DAG dispatch
  subexperiments/
    helpers.py
    <producer_task>.py
    <downstream_task>.py
```

每个 result bundle 中的推荐结构：

```text
data/
  trial_specs/       # 面向 audit/plot compatibility 的显式 specs 副本
  raw/               # trial-level/readout-level outputs
  metrics/           # panel metrics / supplement metrics
  intermediates/     # reusable artifacts; compute-only dependency
figures/
logs/
meta/
summary.json
run_config.json
artifact_manifest.json
```

`data/intermediates/` 是 compute DAG 的状态目录，不应成为 plotting 层的直接数据源。Plotting 层应读取 `data/raw/`、`data/metrics/`、`summary.json` 或 source manifest。

## 6. Artifact Contract

每个 artifact 目录至少包含：

```text
data/intermediates/<task_id>/
  cache_key.json
  manifest.csv 或 manifest.json
  *.csv / *.json / *.npz / shards/
```

`cache_key.json` 至少应覆盖：

- schema name / schema version；
- task id；
- network seed；
- config 中影响结果的参数；
- dataset split/root 或 dataset id；
- model path/fingerprint，若依赖模型；
- parent specs hash；
- parent artifact digest，若依赖上游 artifact；
- smoke / non-smoke 关键差异；
- source-of-truth 语义版本，必要时。

manifest 至少应回答：

- 这个 artifact 从哪份 parent specs 产生；
- 每个文件/shard 的路径、shape、row count、hash；
- 每个 shard 对应哪些 condition / delay / trial / batch；
- 读取后如何验证与当前 specs 一致；
- 缺失、hash mismatch、row count mismatch、schema mismatch 时如何失败。

硬规则：

- `require` 模式只加载，不重建。
- artifact 损坏或 cache key mismatch 必须失败。
- `auto` 可以复用或生成，但不能掩盖 schema/hash 错误。
- `force` 只能用于显式重建当前 producer artifact。
- 下游 task 不允许覆盖 parent artifact，除非它就是该 artifact 的 producer。

## 7. Runtime Mode 语义

所有 figure-local runner 应尽量保持统一：

| 模式 | 语义 | 合法用途 | 禁止行为 |
|---|---|---|---|
| `off` | legacy/fresh 兼容路径，允许按原逻辑重算 | fresh baseline、legacy compare | 用作下游 require 验证 |
| `auto` | 有匹配 artifact 就复用，没有则生成 | 从头构建完整 DAG | 吞掉损坏 artifact 并继续混用 |
| `require` | 必须加载已有 artifact | 单独 rerun 下游 task、证明可恢复 | 加载失败后 fallback 重算 |
| `force` | 显式重建当前 producer artifact | 上游 specs/bank 主动更新 | 顺带重建不相关 parent/child |

`require` 是整套流程最重要的验证模式。只要 `require` 还会重新采样、重新 capture state、重新生成 mask 或重新跑 parent simulation，就说明 DAG 边界没有成功。

## 8. 标准重构流程

### Step 0：冻结 gold / fresh baseline

在改结构前，确认一个可对照的 baseline：

- legacy/fresh smoke root；
- command；
- `summary.json`、`run_config.json`、`artifact_manifest.json`；
- 关键 `data/trial_specs/`、`data/raw/`、`data/metrics/`、NPZ；
- plotting check-only 状态。

要求：

- baseline root 不再覆盖；
- 后续 compare 使用同一 checkpoint、seed、smoke 参数、dataset split；
- 如果当前没有可靠 baseline，先补 baseline，再改结构。

### Step 1：画真实 DAG

不要先写代码。先画：

- source-of-truth specs；
- reusable heavy bank；
- downstream readout/control/sweep；
- aggregation/statistics；
- panel sink；
- 每条边的输入输出；
- 哪些节点可独立重跑；
- 哪些节点改变会影响 scientific protocol。

产出一张 task 表：

| task id | node type | parent artifacts | outputs | consumers | protocol risk |
|---|---|---|---|---|---|

如果 task 表填不出来，不应开始重构。

### Step 2：先固化 source-of-truth specs

最先 artifact 化的永远是决定实验身份的表：

- trial specs；
- sequence specs；
- pair specs；
- mask specs；
- perturbation specs；
- sweep/job specs；
- region/target specs。

规则：

- specs producer 负责生成并保存正式 artifact；
- 下游 `require` 只能加载 specs；
- standalone rerun 不允许重新随机生成 specs；
- specs 的 hash 必须进入下游 cache key；
- specs 文件也应写回 `data/trial_specs/` 或对应兼容位置，便于 audit 和 plotting。

### Step 3：拆 heavy reusable bank

把昂贵且可复用的步骤单独变成 producer：

- network rollout；
- STSP / boundary state capture；
- feature extraction；
- encoded input bank；
- sequence state bank；
- perturbation baseline bank；
- completion/reference boundary bank；
- large trace bank。

规则：

- bank producer 只负责生成 bank；
- bank loader 必须校验 cache key、manifest、hash、shape、row count；
- downstream task 只能通过 loader 获取 bank；
- bank 的 parent specs hash 必须进入 cache key；
- 如果 bank 包含 state，必须记录 state capture phase/time/index/condition。

### Step 4：把下游 task 改成 artifact consumer

每个下游 task 都应能这样运行：

```powershell
python -m src.experiments.paper_figures.figX.run_task `
  --task <task_id> `
  --output-dir <new_output_root> `
  --artifact-root <existing_seed_dir>\data\intermediates `
  --reuse-artifacts require `
  --network-seed 1000 `
  --smoke `
  --device auto
```

验收：

- 只加载 parent artifacts；
- 不重建 parent specs/bank；
- 只写自己的 `data/raw/` / `data/metrics/` / summary metadata；
- standalone 输出与 full DAG 对应输出一致；
- run metadata 记录 artifact root、reuse mode、cache key digest。

### Step 5：补 panel producer mapping

每个数据 panel 必须在 plotting spec 中声明：

```yaml
producer_task: <task_id>
```

规则：

- panel 是 DAG sink，不是 compute node；
- panel check-only 只验证 source files/schema/claim，不运行 simulation；
- panel source manifest 应能追溯到 producing task；
- 一个 panel 可以依赖多个 producer task，但必须显式列出。

### Step 6：补 failure guards

每类关键 artifact 至少需要一个破坏性副本测试，验证 `require` 不会静默 fallback：

- 删除/改坏 specs 文件；
- 修改 manifest hash；
- 修改 state/boundary shard；
- 修改 mask specs；
- 删除 required delay/condition；
- 改 cache key digest。

这些测试应在临时 root 上做，不能破坏 golden root。

### Step 7：记录验证报告

每个 figure 完成后应写一个 verification report，内容包括：

- 代码范围；
- 测试环境；
- baseline root；
- DAG root；
- artifact root；
- generated artifacts；
- standalone require tasks；
- standalone 输出一致性 hash；
- fresh vs DAG regression compare；
- failure guards；
- layout validation；
- plot check-only；
- 当前限制，例如 smoke/single-seed 不是正式 manuscript rerun。

## 9. 验收矩阵

| 验收项 | 阶段 1 是否必须 | 通过标准 |
|---|---|---|
| compile | 是 | 新 runner/artifact/subexperiment 文件可编译 |
| baseline/fresh smoke | 是 | 原始或兼容路径可生成对照输出 |
| DAG full smoke | 是 | `--task all --reuse-artifacts auto` 成功 |
| artifact inventory | 是 | 每个 key artifact 有 `cache_key.json` 和 manifest |
| standalone require | 是 | 主要 downstream task 可单独运行 |
| source specs reuse | 是 | standalone 不重新采样 trial/sequence/mask |
| heavy bank reuse | 是 | downstream 不重跑 parent simulation/state capture |
| output equivalence | 是 | standalone vs full DAG hash 或数值一致 |
| fresh vs DAG compare | 是 | legacy/fresh vs DAG 关键输出 `atol=1e-6`, `rtol=1e-6` 或项目指定容差 |
| failure guard | 是 | 缺失/损坏 artifact 在 `require` 下失败 |
| layout validation | 是 | `src/plotting/paper_fig/LAYOUT_CONTRACT.md` 与当前 plot-only `--check-only` 均通过 |
| plot check-only | 是 | required source、seed、statistics 和 layout 全部通过；缺失项必须失败 |
| full multi-seed rerun | 否 | 结构验证后单独安排 |
| performance optimization | 否 | 需要 profiling 后进入后续阶段 |

## 10. AI 与人工分工

AI 可以做：

- 扫描现有代码，列出 task candidate；
- 画 DAG 草图；
- 提取输入输出表；
- 建 `schemas.py` / `cache_keys.py` / `artifacts.py` / `run_task.py` 骨架；
- 把纯 I/O、hash、manifest、loader helper 标准化；
- 给 plotting spec 增加 `producer_task`；
- 写 smoke、require、compare、failure guard 命令；
- 根据验证结果更新报告。

必须人工确认：

- specs 的科学语义是否正确；
- state capture 的时间点/phase/condition 是否正确；
- STSP restore、freeze、mutation 是否等价；
- mask、cue、ping、perturbation、sequence 定义是否可合并；
- 哪些 outputs 支撑 manuscript claim；
- 哪些 artifact 可以跨 figure 复用。

AI 不能擅自做：

- 修改 delay grid、sample length、sequence length；
- 修改 mask definition、sampling population、target scope；
- 修改 readout endpoint 或判定规则；
- 修改 STSP update / restore 顺序；
- 用 proxy metric 替代原始 manuscript metric；
- 在 `require` 失败时加 fallback 重跑；
- 把某个 figure 的 protocol helper 强行抽象成跨 figure common。

## 11. 代码修改边界

阶段 1 推荐的低风险顺序：

1. 新增 figure-local schema/cache/artifact/runner 层。
2. 保留 legacy/fresh 路径作为 regression baseline。
3. 只改必要下游 consumer，使其可以从 artifact 加载输入。
4. 不先移动大量 helper。
5. 不先改 simulation inner loop。
6. 不先做 batch/compile/AMP 等性能优化。

如果需要复用逻辑，优先放在 figure-local 模块。只有满足下面条件才进入 shared/common：

- 至少两个 figure 已实际使用；
- 输入输出语义完全一致；
- cache key 维度完全明确；
- 有 regression 测试保护；
- 人工确认不会改变 scientific protocol。

## 12. 通用迁移检查表

每个 figure 开始前先填：

```text
Figure / suite:
Legacy or fresh baseline root:
Baseline command:
DAG smoke root:
Artifact root:

Main panels:
Supplement panels:

Source-of-truth specs:
- task_id:
- files:
- columns/schema:
- cache key fields:
- consumers:
- human-confirmed protocol risk:

Heavy reusable artifacts:
- task_id:
- producer:
- parent specs:
- files/shards:
- capture phase/time/index:
- consumers:
- validation checks:

Deterministic spec expansions:
- task_id:
- parent specs:
- output specs:
- seed derivation:
- consumers:

Downstream standalone tasks:
- task_id:
- required artifacts:
- expected raw outputs:
- expected metric outputs:
- equivalence files:

Panel producer mapping:
- panel:
- producer_task:
- source files:
- check-only command:

Validation:
- compile command:
- DAG full smoke command:
- standalone require commands:
- fresh vs DAG compare command:
- failure guard commands:
- layout validation command:
- plot check-only command:

Known limits:
- smoke-only:
- single-seed-only:
- not manuscript-final:
- remaining platformization debt:
```

如果这张表填不出来，说明当前还不应该让 AI 直接改代码。

## 13. 推广顺序

推荐推广方式：

1. 先选一个 figure 做完整阶段 1，而不是同时动所有 figure。
2. 先固化 source specs，再拆 heavy bank。
3. 先验证一个 downstream task 的 standalone require。
4. 再扩展到同 figure 的其他 downstream tasks。
5. 再补 panel producer mapping。
6. 再补 failure guards。
7. 最后写 verification report。
8. 两个以上 figure 成功后，再提取 shared/common。

优先级判断：

| 优先级 | 任务 | 说明 |
|---|---|---|
| P0 | 画 DAG、冻结 baseline、补 producer_task | 不改变结果 |
| P1 | source specs artifact、cache key、manifest | 保护一致性 |
| P1 | heavy bank artifact、require loader | 减少重复 simulation |
| P2 | downstream standalone require | 建立局部重跑能力 |
| P2 | failure guards、verification report | 固化可验证性 |
| P3 | shared/common 提取 | 需要两个以上 figure 证明 |
| P3 | scheduler/platform/cache resolver | 等阶段 1 普遍完成 |
| P4 | simulation inner loop / AMP / torch.compile | 必须 profile + regression |

## 14. 硬规则

1. 重构第一目标是 runtime dependency 解耦，不是形式上的文件拆分。
2. 任何影响 sampling / pairing / sequence / mask / condition grid 的表都必须先 artifact 化。
3. 任何昂贵 simulation / state capture / feature bank 都必须能被下游复用。
4. `require` 模式只能加载，不能生成。
5. 下游 task 修改后，只应重跑该 task 和它的 child outputs。
6. Parent artifact 已通过校验时，不得被下游 task 隐式重跑。
7. Artifact cache key 必须包含 parent specs hash。
8. Manifest/hash/schema mismatch 必须失败。
9. Plotting 层只消费结果，不运行实验。
10. Panel 通过 `producer_task` 追溯 compute source。
11. Fresh vs DAG compare 是结果不变的最低证明。
12. Smoke/single-seed 只能证明结构，不等于正式 manuscript final。
13. AI 可以改工程边界，但不能未经确认修改 scientific protocol。
14. 不要把某个 figure 的特例过早提取为全局 common。
15. 平台化必须建立在多个 figure 的成功 DAG 化之后。

## 15. 平台化前完成标准

进入跨 figure 平台化前，至少应满足：

- 每个主 figure 有 task-level runner。
- 每个主 figure 有 source-of-truth specs artifact。
- 每个主 figure 至少有一个 heavy reusable artifact。
- 每个主 figure 的主要 downstream task 支持 `require`。
- 每个主 figure 能证明至少一个 downstream task 可在不重跑 parent artifact 的情况下单独重跑。
- 每个 manuscript data panel 有 `producer_task`。
- 每个主 figure 有 fresh vs DAG regression compare 记录。
- 每个主 figure 有 failure guard 记录。
- 每个主 figure 有 verification report。

达到这些条件后，再考虑：

- global task registry；
- shared artifact resolver；
- cross-figure cache；
- scheduler；
- provenance database；
- profiling-aware runtime optimizer。

否则，平台化会把当前 figure-specific 的不确定性放大成全局复杂度。
