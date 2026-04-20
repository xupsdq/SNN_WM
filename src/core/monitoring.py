import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm  # <--- 添加这一行
import numpy as np
import os
import seaborn as sns

class ClassSensitivityMonitor:
    def __init__(self, num_classes, neurons_per_class, device='cuda'):
        self.num_classes = num_classes
        self.neurons_per_class = neurons_per_class
        self.device = device
        # 混淆矩阵维度：[True_Class, Pred_Class + 1] 最后一列留给 No-Spike
        self.confusion_matrix = torch.zeros((num_classes, num_classes + 1), device=device)
        # 神经元响应热图：[Class, Neurons_in_Class]
        self.neuron_fire_counts = torch.zeros((num_classes, num_classes * neurons_per_class), device=device)

    def update(self, labels, firing_times):
        """
        根据 Layer 3 的 firing_times 更新统计
        firing_times: [B, Out_Channels] (值为 inf 表示未发放)
        """
        B = labels.shape[0]
        # 找到每个样本最早发放的神经元索引
        min_times, min_indices = torch.min(firing_times, dim=1)

        # 掩码：哪些样本实际产生了脉冲
        has_fired = min_times < float('inf')

        # 计算预测类别
        # 如果没放电，我们将其分类到索引 num_classes (最后一列)
        pred_classes = torch.full_like(labels, self.num_classes)
        pred_classes[has_fired] = min_indices[has_fired] // self.neurons_per_class

        # 更新混淆矩阵 (使用 scatter_add_ 进行高性能更新)
        # 将 labels 和 pred_classes 组合成线性索引
        indices = labels * (self.num_classes + 1) + pred_classes
        self.confusion_matrix.put_(indices, torch.ones_like(indices, dtype=torch.float), accumulate=True)

        # 更新神经元响应分布 (仅针对有响应的样本)
        if has_fired.any():
            f_labels = labels[has_fired]
            f_indices = min_indices[has_fired]
            # 这里的坐标是 [真实标签, 具体的神经元全局索引]
            for i in range(len(f_labels)):
                self.neuron_fire_counts[f_labels[i], f_indices[i]] += 1

    def reset(self):
        self.confusion_matrix.zero_()
        self.neuron_fire_counts.zero_()

    def get_metrics(self):
        cm_np = self.confusion_matrix.cpu().numpy()
        # 计算每个类别的召回率 (不含 No-Spike 干扰)
        row_sums = cm_np.sum(axis=1, keepdims=True) + 1e-9
        normalized_cm = cm_np / row_sums
        return normalized_cm, self.neuron_fire_counts.cpu().numpy()