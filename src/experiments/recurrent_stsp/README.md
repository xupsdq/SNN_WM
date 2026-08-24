# recurrent_stsp

本目录提供独立于现有卷积 SNN 的 Tiddia 循环工作记忆模型 PyTorch 等价内核、可复用状态工具和基础 artifact DAG；当前不包含任何机制结论专用的科学实验树。

- `__init__.py`：导出神经元、突触、连接、输入、记录、检查点、STSP 状态和解码器的稳定公共接口。
- `artifact_plot_data.py`：只把持久化张量产物转换为绘图所需的 NumPy 数据，不运行模型。
- `checkpoint.py`：捕获、序列化和精确恢复神经元、延迟环与逐边 STSP 状态，并提供通用 STSP-only 状态操作。
- `config.py`：声明默认 10,000 神经元、20M 连接图及异质运行配置的科学参数。
- `connectivity.py`：按上游 fixed-indegree 规则生成、校验并原子持久化源排序 CSR 连接产物。
- `linear_decoder.py`：提供只由 train 拟合、由 validation 选正则化并在 test 上冻结评估的 ridge decoder。
- `nest_equivalent.py`：实现 NEST 3.1 `iaf_psc_exp` 步进、逐突触 Tsodyks3 更新和固定网格延迟缓冲。
- `plot_artifacts.py`：只读取已有 spike/STSP 产物，输出 PNG、PDF 和 SVG 诊断图。
- `protocol.py`：实现背景、载入、读出、噪声、周期输入和可精确回放的冻结稀疏事件输入。
- `recording.py`：稀疏记录放电、恢复连续 `u/x` 轨迹，并以显式阈值执行基础任务评价。
- `reference_protocol.py`：复现上游仓库随附的单突触 burst–pause–recovery 验证协议。
- `run.py`：提供连接生成、基础仿真、评价和 plot-only 的通用 DAG CLI 节点。
- `runner.py`：运行基础协议并写出规范化数据、元数据、依赖清单、指标和摘要。
- `scheduler.py`：在 CPU/CUDA 上按活跃源神经元调度稀疏事件、更新逐边 STSP、聚合延迟电流并推进循环网络。
- `temporal_state.py`：提供稀疏逐边 STSP 的索引快照、边界替换、被动演化、next-release 计算和 presynaptic-event 精确回放。

基础流程先生成或复用连接产物，再运行仿真；评价与绘图只消费仿真产物：

```powershell
python -m src.experiments.recurrent_stsp.run build-connectivity `
  --output artifacts/recurrent_stsp/tiddia_heterogeneous_graph.pt `
  --network-profile heterogeneous-upstream --reuse auto

python -m src.experiments.recurrent_stsp.run simulate `
  --connectivity artifacts/recurrent_stsp/tiddia_heterogeneous_graph.pt `
  --output-directory results/recurrent_stsp/upstream_seed143202461 `
  --device cuda

python -m src.experiments.recurrent_stsp.run evaluate `
  results/recurrent_stsp/upstream_seed143202461

python -m src.experiments.recurrent_stsp.run plot `
  results/recurrent_stsp/upstream_seed143202461
```

完整基础运行写出 `run_config.json`、`data/spikes.pt`、`data/stsp_probes.pt`、`metrics/task_metrics.json`、`meta/run_info.json`、`artifact_manifest.json` 和 `summary.json`；plot-only 节点在 `figures/` 中另写只读依赖清单，不会重新运行仿真。
