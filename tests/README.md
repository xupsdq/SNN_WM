# tests

本目录是项目 pytest 测试入口；`pytest.ini` 将本目录设为测试根目录。

测试覆盖当前实验协议、结果布局、图形合同、数据 lineage、数据集路径归一化和关键分析逻辑。`test_dataset_path_layout.py` 验证 canonical `data/MNIST` 布局；`test_masse_delayed_cue_lif_stsp.py` 验证去慢电流/SFA 的 Masse-LIF 配对网：脉冲触发按细胞 STSP、无独立电流状态、权重 ÷3、循环脉冲 detach 与 \(x,u\) 回传、逐参数梯度裁剪、decode 节点和 plot-only 叶节点；`test_recurrent_stsp_equivalent_kernel.py` 验证独立 PyTorch `iaf_psc_exp`/Tsodyks3 内核的更新顺序、延迟语义、上游黄金轨迹和 CPU/GPU 一致性；`test_recurrent_stsp_sparse_network.py` 验证 fixed-indegree CSR 生成、连接产物持久化、稀疏事件调度、事件审计、强制放电包和可选的完整 10k/20M 分配；`test_recurrent_stsp_core_extensions.py` 验证冻结事件输入、外部 RNG/延迟状态恢复、网络检查点、索引 STSP 边界操作、精确事件回放和 split-safe ridge decoder；`test_recurrent_stsp_workflow.py` 验证基础外部输入、连续 STSP 记录、独立任务判定、规范化 artifact DAG 和 plot-only 消费边界。测试生成的缓存和临时输出不属于源码。
