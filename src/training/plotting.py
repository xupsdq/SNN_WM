import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm
from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_mnist_skeleton_loader
from src.platform.legacy_adapters.network import SDNN_Network
from src.platform.legacy_adapters.units import *
import torch.nn.functional as F

def plot_learned_kernels(kernels, save_path, epoch_info):
    """
    更新版：支持双通道 (ON/OFF) 可视化。
    """
    out_c, in_c, k_h, k_w = kernels.shape
    grid_size = int(np.ceil(np.sqrt(out_c)))

    w_max_val = kernels.max() if kernels.max() > 0 else 1e-9

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10), facecolor='#121212')
    fig.suptitle(f'Learned Receptive Fields (Epoch_Batch: {epoch_info})\nRed=ON, Green=OFF',
                 color='white', fontsize=16)

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < out_c:
            k_weight = kernels[i]
            rgb_kernel = np.zeros((k_h, k_w, 3))

            on_ch = k_weight[0] / w_max_val
            off_ch = k_weight[1] / w_max_val

            rgb_kernel[..., 0] = np.clip(on_ch, 0, 1)
            rgb_kernel[..., 1] = np.clip(off_ch, 0, 1)
            rgb_kernel[..., 2] = 0.1

            ax.imshow(rgb_kernel, interpolation='nearest')
            ax.set_title(f"K{i}", color='gray', fontsize=8, pad=2)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close()

def plot_weight_distribution_evolution(net, save_dir, epoch, layer_idx=2):
    """
    可视化方案二：权重分布分析 (Weight Distribution & Polarization)
    目的：检查权重是否出现两极分化 (Bimodal Distribution)。
    理想状态：直方图呈 "U" 型，大部分权重接近 0 或 1 (w_max)。
    """
    if layer_idx == 1:
        kernels = net.layer1.kernels.detach().cpu()
    elif layer_idx == 2:
        kernels = net.layer2.kernels.detach().cpu()
    else:
        kernels = net.layer3.kernels.detach().cpu()

    # 展平所有权重 (全局分析)
    w_flat = kernels.flatten().numpy()

    # 归一化 (假设 w_min=0, w_max=1 或根据实际值)
    w_min, w_max = w_flat.min(), w_flat.max()

    # 绘图
    plt.figure(figsize=(10, 6))

    # 1. 直方图
    plt.hist(w_flat, bins=50, color='tab:blue', alpha=0.7, density=True, label='Weight Density')

    # 2. 统计指标
    mean_w = np.mean(w_flat)
    var_w = np.var(w_flat)

    # 极化指数 (Polarization Index): 简单的定义为 (Variance) / (Mean * (1-Mean)) 用于 0-1 范围
    # 或者简单看有多少权重在两端 (比如 <0.1 或 >0.9)
    # 这里我们用简单的文本标注
    norm_w = (w_flat - w_min) / (w_max - w_min + 1e-9)
    polarized_ratio = np.sum((norm_w < 0.1) | (norm_w > 0.9)) / len(norm_w)

    plt.axvline(mean_w, color='red', linestyle='--', label=f'Mean: {mean_w:.3f}')
    plt.title(f"Layer {layer_idx} Weight Distribution (Epoch {epoch})\nPolarization Ratio: {polarized_ratio:.2%}",
              fontsize=14)
    plt.xlabel("Weight Value (nS)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, f'weight_dist_L{layer_idx}_e{epoch}.png')
    plt.savefig(save_path)
    plt.close()


def plot_class_sensitivity(normalized_cm, neuron_counts, save_path, epoch_info):
    """
    绘制双面板诊断图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 1. 混淆矩阵热图
    num_classes = normalized_cm.shape[0]
    xtick_labels = [str(i) for i in range(num_classes)] + ['No-Spike']
    sns.heatmap(normalized_cm, annot=True, fmt=".2f", cmap="YlGnBu",
                ax=ax1, xticklabels=xtick_labels, yticklabels=range(num_classes))
    ax1.set_title(f"Normalized Confusion Matrix ({epoch_info})")
    ax1.set_xlabel("Predicted Class")
    ax1.set_ylabel("True Class")

    # 2. 神经元激活特异性
    # 理想情况下，每个类别的标签应该只激活属于该类别的 10 个神经元
    sns.heatmap(neuron_counts, cmap="Reds", ax=ax2)
    ax2.set_title("Neuron Response Distribution (Global Index)")
    ax2.set_xlabel("Neuron ID (0-99)")
    ax2.set_ylabel("True Class")

    # 画出类别界限线
    for i in range(num_classes + 1):
        ax2.axvline(i * (neuron_counts.shape[1] // num_classes), color='blue', lw=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
