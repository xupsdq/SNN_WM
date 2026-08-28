# subexperiments

本目录负责 Fig.1 各子实验的显式依赖实现，不从旧单体模块注入全局名称。

- `__init__.py` 标记 Fig.1 子实验 Python package。
- `baseline.py` 运行基线分类并写入 trial、网络和混淆矩阵指标。
- `delay_decode.py` 构建延迟期 STSP 特征并运行线性解码。
- `dms_delay_sweep.py` 比较不同延迟下 dynamic 与 static-frozen probe readout。
- `dms_shuffle.py` 运行 DMS 状态底物 shuffle、配对审计和归因指标。
- `firing_rate_control.py` 汇总刺激、延迟和 probe 阶段的放电率。
- `helpers.py` 提供共享的状态捕获、probe、配对校验、采样和指标计算原语。
- `time_binned_firing_rate.py` 生成 plot-only 可消费的时间分箱放电率 artifact。
