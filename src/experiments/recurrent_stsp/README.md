# recurrent_stsp

本目录提供独立于现有卷积 SNN 的 Tiddia 循环工作记忆模型 PyTorch 后端，并把连接、仿真、解码和绘图组织成可复用的实验 DAG。

- `__init__.py`：导出神经元、突触、输入协议、记录器、解码器和运行器公共接口。
- `artifact_plot_data.py`：只把持久化张量产物转换为绘图所需的 NumPy 数据，不调用模型。
- `checkpoint.py`：精确保存和恢复神经元、延迟环与逐边 STSP 状态，并支持 STSP-only donor substitution。
- `config.py`：声明默认 10,000 神经元/20M 连接图和上游异质运行配置的科学参数。
- `connectivity.py`：按上游 fixed-indegree 规则生成、校验并原子持久化源排序 CSR 连接产物。
- `dms_experiment.py`：运行 behavior-first delayed-match-to-sample trial、STSP reset/donor swap 分支，并从持久化特征执行独立 decoder 分析。
- `dms_multinetwork.py`：冻结独立连接图清单，并只读各网络分析产物生成等权网络级汇总。
- `dms_trials.py`：生成并验证 sample、probe、match 标签和随机流成对平衡的 train/validation/test trial manifest。
- `linear_decoder.py`：提供只由 train 拟合、validation 选择正则化、test 冻结评估的 ridge decoder。
- `nest_equivalent.py`：实现 NEST 3.1 `iaf_psc_exp` 精确步进、逐突触 tsodyks3 更新和固定网格延迟缓冲。
- `mechanism_experiment.py`：运行不同历史、相同 query 的完整四分支配对替换实验并计算 donor-direction 端点。
- `plot_artifacts.py`：只读取已有 spike/STSP 产物，输出 PNG、PDF 和 SVG 诊断图。
- `protocol.py`：实现背景、载入提示、非特异读出、随机噪声、周期读出和晚期背景抵消协议。
- `recording.py`：稀疏记录放电，解析恢复连续 `u/x` 轨迹，并以显式阈值判定任务成功。
- `reference_protocol.py`：复现上游仓库随附的单突触 burst–pause–recovery 验证协议。
- `run.py`：提供 `build-connectivity`、`simulate`、`evaluate`、`plot`、`matched-query`、`dms-simulate`、`dms-analyze`、`dms-aggregate` 和工作流 CLI 任务。
- `runner.py`：执行 120,000 步运行并写出规范化数据、元数据、依赖清单、指标和摘要。
- `scheduler.py`：在 CPU/CUDA 上按活跃源神经元展开稀疏边、更新逐边 STSP、聚合延迟电流并推进循环网络。

默认完整流程先显式生成或复用连接，再运行上游 6,000 ms 协议：

```powershell
python -m src.experiments.recurrent_stsp.run build-connectivity `
  --output artifacts/recurrent_stsp/tiddia_heterogeneous_graph.pt `
  --network-profile heterogeneous-upstream --reuse auto

python -m src.experiments.recurrent_stsp.run simulate `
  --connectivity artifacts/recurrent_stsp/tiddia_heterogeneous_graph.pt `
  --output-directory results/recurrent_stsp/upstream_seed143202461 `
  --device cuda
```

解码和绘图是下游任务，只消费上述持久化产物：

```powershell
python -m src.experiments.recurrent_stsp.run evaluate `
  results/recurrent_stsp/upstream_seed143202461

python -m src.experiments.recurrent_stsp.run plot `
  results/recurrent_stsp/upstream_seed143202461
```

完整默认运行写出 `run_config.json`、`data/spikes.pt`、`data/stsp_probes.pt`、`metrics/task_metrics.json`、`meta/run_info.json`、`artifact_manifest.json` 和 `summary.json`；绘图任务在 `figures/` 中另写只读依赖清单。

不同历史、相同 query 的 STSP-only 因果替换实验使用已有连接图，并写出四个分支、状态样本和 donor-direction 指标：

```powershell
python -m src.experiments.recurrent_stsp.run matched-query `
  --connectivity artifacts/recurrent_stsp/tiddia_heterogeneous_graph.pt `
  --output-directory results/recurrent_stsp/matched_query_seed143202461 `
  --device cuda
```

经典 DMS 行为与机制实验把模拟和 decoder 保持为两个独立 DAG 节点；`dms-workflow` 只是按依赖顺序调用二者：

```powershell
python -m src.experiments.recurrent_stsp.run dms-simulate `
  --connectivity artifacts/recurrent_stsp/tiddia_heterogeneous_graph.pt `
  --output-directory results/recurrent_stsp/dms_seed143202461 `
  --device cuda

python -m src.experiments.recurrent_stsp.run dms-analyze `
  results/recurrent_stsp/dms_seed143202461
```
