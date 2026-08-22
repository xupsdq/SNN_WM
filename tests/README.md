# tests

项目测试入口；`pytest.ini` 将 `tests/` 设为测试根目录。

测试覆盖当前实验协议、结果布局、图形合同、数据 lineage、数据集路径归一化和关键分析逻辑。`test_dataset_path_layout.py` 验证 canonical `data/MNIST` 布局，并用临时目录覆盖旧 nested torchvision 形态；`test_recurrent_stsp_equivalent_kernel.py` 验证独立 PyTorch `iaf_psc_exp`/tsodyks3 内核的更新顺序、延迟语义、上游黄金轨迹和 CPU/GPU 一致性；`test_recurrent_stsp_sparse_network.py` 验证 fixed-indegree CSR 生成、连接产物持久化、稀疏事件调度和可选的完整 10k/20M CUDA 分配；`test_recurrent_stsp_workflow.py` 验证全部外部输入族、连续 STSP 恢复、独立任务判定、规范化产物 DAG 和 plot-only 消费边界；`test_recurrent_stsp_mechanism_experiment.py` 验证完整状态逐位重放、STSP-only 隔离替换和 matched-query 四分支 DAG；`test_recurrent_stsp_dms_experiment.py` 验证 DMS trial 平衡、decoder split 防泄漏、STSP reset 隔离性、行为—机制 DAG 和独立图等权聚合。仓库本身只保留 canonical 数据。测试生成的 `__pycache__`、pytest 缓存和临时输出不属于源码，应保持在忽略目录中。
