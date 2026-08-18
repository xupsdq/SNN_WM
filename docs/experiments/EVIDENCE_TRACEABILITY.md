# Experiment Evidence Traceability

本上下文统一描述实验执行节点、持久化证据与稿件图之间的可追踪关系，避免把运行身份、结果目录和科学主张混为同一概念。

## Language

**Execution Node（执行节点）**:
能够被独立寻址并承担计算、调度、聚合、物化或展示职责的工作单元。
_Avoid_: 用“实验”泛指所有脚本、模块或图面

**DAG Task（DAG 任务）**:
实验 DAG 中显式声明、具有稳定身份和父子依赖的可执行节点。
_Avoid_: Subexperiment、panel、任意 Python 文件

**Runner Experiment（Runner 实验）**:
以一次独立实验运行形成结果 bundle 的执行节点；它可以包含内部阶段，但未显式声明的阶段不自动成为 DAG Task。
_Avoid_: Orchestrator、Materializer

**Orchestrator（编排器）**:
调度多个执行节点、网络或范围的控制节点；它只承担执行覆盖结论，不产生新的科学结论。
_Avoid_: Runner Experiment

**Materializer（物化节点）**:
只消费既有父证据并形成新的可读数据 bundle 的执行节点。
_Avoid_: Simulation、重新实验

**Aggregator（聚合节点）**:
把多个既有结果合并为统计或验证 bundle 的执行节点。
_Avoid_: 把聚合结论当成新的原始实验

**Plot Consumer（绘图消费者）**:
只读取既有结果并将其呈现到 figure 或 panel 的叶节点。
_Avoid_: Producer、Simulation

**Subexperiment（子实验选择项）**:
用于选择一次运行范围的逻辑标签；除非存在独立执行身份，否则不是 DAG Task。
_Avoid_: Task ID

**Result Bundle（结果包）**:
一次运行、物化或聚合所形成的、具有共同来源与状态的持久化结果集合。
_Avoid_: 单个 Artifact、任意目录

**Artifact（产物）**:
Result Bundle 中可独立寻址和追踪的文件记录。
_Avoid_: Result Bundle

**Input Artifact（输入产物）**:
被执行节点消费但不承载该节点结果结论的持久化输入，例如数据集、checkpoint 或父级数组。
_Avoid_: Result Artifact

**Result Artifact（结果产物）**:
由执行节点产生并可用于形成任务级结果陈述的持久化文件。
_Avoid_: Input Artifact

**Task Evidence Record（任务证据记录）**:
把一个执行节点的身份、依赖、结果包、产物、证据状态、结果陈述和任务结论绑定在一起的记录。
_Avoid_: 全局科学综述

**Result Statement（结果陈述）**:
由单个执行节点的可追踪证据直接支持的一条局部观察；一个任务可以有多条结果陈述。
_Avoid_: 跨任务综合结论

**Task Conclusion（任务结论）**:
针对一个执行节点且仅由该节点可追踪证据支持的唯一有界结论；无有效证据时必须明确写成不可验证。
_Avoid_: Manuscript-wide conclusion、空白结论

**Authority Tier（权威层级）**:
结果在当前稿件、活跃父证据、探索、smoke 或历史归档中的身份层级；它不等同于文件是否存在。
_Avoid_: 仅按修改时间判断权威性

**Provenance Link（来源关系）**:
任务与证据之间的可追踪强度，规范值为 direct、contractual、inferred、none 或 conflict。
_Avoid_: 根据相邻任务或文件名补全来源

**Conclusion Status（结论状态）**:
任务结论的证据充分性，规范值为 supported、bounded、operational-only、conflicted 或 unavailable。
_Avoid_: 用单一布尔值同时表示存在性、权威性和科学支持

**Derived Audit Snapshot（派生审计快照）**:
在明确时间点从源码、权威控制面和持久化 manifests 派生的可发现性文档；它不取代这些事实来源。
_Avoid_: Current task map、Canonical artifact map

**Runtime Figure ID（运行时图身份）**:
实验与统计运行使用的稳定内部图身份。
_Avoid_: 假定它与 Manuscript Figure ID 永久一一同号

**Manuscript Figure ID（稿件图身份）**:
当前稿件导航使用的图号，通过显式 authority mapping 关联到运行时图、结果 bundle 和 artwork。
_Avoid_: 通过目录名或 task 名推断稿件图号
