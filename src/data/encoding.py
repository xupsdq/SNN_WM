from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ?
from src.config.units import ms


class DoGTransform:
    """
    ???(DoG) ??- ??(?Kheradpisheh ?

    ?
    ????split ????
    ????????
    1. ON-Center Kernel: Center(s1) - Surround(s2)
    2. OFF-Center Kernel: Center(s2) - Surround(s1)
    """

    def __init__(self, kernel_size=7, sigma1=1.0, sigma2=2.0, threshold=0.1):
        self.kernel_size = kernel_size
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.threshold = threshold
        # ???
        self.kernels = self._create_pair_kernels()

    def _get_gaussian_kernel(self, ksize, sigma):
        """Generate a standard 2D Gaussian kernel."""
        x = torch.linspace(-(ksize // 2), ksize // 2, ksize)
        y = torch.linspace(-(ksize // 2), ksize // 2, ksize)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        return kernel / kernel.sum()

    def _create_pair_kernels(self):
        """Create separate ON and OFF DoG kernels."""
        # ??
        g1 = self._get_gaussian_kernel(self.kernel_size, self.sigma1)
        g2 = self._get_gaussian_kernel(self.kernel_size, self.sigma2)

        # 1. ON-Center Kernel (???
        # ? G(s1) - G(s2)
        k_on = g1 - g2

        # 2. OFF-Center Kernel (???
        # ? G(s2) - G(s1)
        # ???-k_on?Kheradpisheh ?
        k_off = g2 - g1

        # ????[Out_Channels=2, In_Channels=1, H, W]
        # ?0: ON, ?1: OFF
        kernels = torch.stack([k_on, k_off], dim=0).unsqueeze(1)

        return kernels

    def __call__(self, img_tensor):
        """
        ??
        """
        is_batch = img_tensor.dim() == 4
        x = img_tensor if is_batch else img_tensor.unsqueeze(0)
        device = x.device
        B, C_in, H, W = x.shape

        # ???
        if self.kernels.device != device:
            self.kernels = self.kernels.to(device)

        # ?B)???groups ?repeat ??
        # ??MNIST (C_in=1)???2 ??(ON, OFF)
        # ???[B, 2, H, W]

        # 1. ??
        # ?groups=1, ??[2, 1, K, K]?[B, 1, H, W] -> ?[B, 2, H, W]
        # ???
        dog_out = F.conv2d(x, self.kernels, padding=self.kernel_size // 2)

        # 2. ??(Rectification)
        # Kheradpisheh ?Filter ?thresholds ?abs (??
        # ?SNN ????(ReLU ??
        # ON?-> ON?-> OFF?
        dog_out = torch.relu(dog_out)  # ???clamp(min=0)

        # 3. ?
        # ????
        final_map = torch.where(dog_out > self.threshold, dog_out, torch.zeros_like(dog_out))

        # 4. ?(?
        # --- ?(Independent Channel Normalization) ---
        # ?ON ?OFF ??1.0
        # ??, 2, H, W] -> [B, 2, H*W]
        flat = final_map.view(B, 2, -1)
        # ?2 () ??1 (?
        max_vals, _ = flat.max(dim=2, keepdim=True)
        # ?[B, 2, 1, 1] ?
        max_vals = max_vals.view(B, 2, 1, 1)

        return final_map / (max_vals + 1e-12)


class DoGSpikeEncoder:
    """
    DoG ?- ?
    (????????DoGTransform ???
    """
    def __init__(self, dt=1.0 * ms, theta_freq=5.0, gamma_freq=50.0, max_duration=1000 * ms, device='cpu'):
        self.dt = dt
        self.device = device
        self.max_duration = int(max_duration / dt)

        # ???
        self.theta_cycle_steps = max(1, int((1.0 / theta_freq) / dt)) if theta_freq > 0 else self.max_duration
        self.gamma_cycle_steps = max(1, int((1.0 / gamma_freq) / dt))

        # ?gamma ??
        self.active_gamma_indices = [0, 3, 6]
        self.encoding_window = 20

        # ?DoGTransform
        self.preprocessor = DoGTransform(kernel_size=7, sigma1=1.0, sigma2=2.0, threshold=0.05)

    def forward(self, images):
        images = images.to(self.device)

        # 1. ?[B, 2, H, W]
        dog_map = self.preprocessor(images)
        B, C, H, W = dog_map.shape

        # ?-1 (?
        latency_map = torch.full((B, C, H, W), -1, dtype=torch.long, device=self.device)

        # 2. ???
        for i in range(B):
            sample = dog_map[i]  # [2, H, W]

            # ?0 ???
            mask = sample > 0
            valid_values = sample[mask]
            num_valid = valid_values.numel()

            if num_valid > 0:
                # ???
                # ?argsort ?????
                sort_indices = torch.argsort(valid_values)
                valid_ranks = torch.argsort(sort_indices)

                # ???[0, encoding_window - 1]
                # ??-> ?-> ?
                # ??num_valid??
                valid_latency = ((num_valid - 1 - valid_ranks).float() * self.encoding_window / num_valid).long()
                valid_latency = torch.clamp(valid_latency, 0, self.encoding_window - 1)

                # ??
                latency_map[i][mask] = valid_latency

        # 3. ?(???
        spike_train = torch.zeros((B, self.max_duration, C, H, W), device=self.device)
        gamma_temp = torch.zeros((B, self.gamma_cycle_steps, C, H, W), device=self.device)

        valid_mask = (latency_map >= 0) & (latency_map < self.gamma_cycle_steps)
        b, c, h, w = torch.where(valid_mask)
        t = latency_map[valid_mask]
        gamma_temp[b, t, c, h, w] = 1.0

        for t_theta in range(0, self.max_duration, self.theta_cycle_steps):
            for g_idx in self.active_gamma_indices:
                start = t_theta + (g_idx * self.gamma_cycle_steps)
                if start >= self.max_duration: continue
                end = min(start + self.gamma_cycle_steps, self.max_duration)
                spike_train[:, start:end, ...] = gamma_temp[:, :end - start, ...]

        return spike_train


def visualize_encoding_pipeline():
    """
    ???ON ?OFF ???
    hionMNIST ?
    """
    print("?????(FashionMNIST, ON/OFF ? ??)...")
    device = 'cpu'
    encoder = DoGSpikeEncoder(device=device)
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])

    # --- ?1: ??FashionMNIST ---
    # root ???'./FashionMNIST' ??
    dataset = datasets.FashionMNIST(root='./FashionMNIST', train=False, download=True, transform=transform)

    # --- ? FashionMNIST ?---
    label_map = {
        0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
        5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'
    }

    # ?10 ?(0-9)? ?
    # ? ??| ON ??| OFF ??| ON ?| OFF ?
    fig, axes = plt.subplots(10, 5, figsize=(20, 25))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for lbl in range(10):
        img_found = None
        for img, label in dataset:
            if label == lbl:
                img_found = img.unsqueeze(0)
                break

        with torch.no_grad():
            dog_map = encoder.preprocessor(img_found)  # [1, 2, 28, 28]
            spike_train = encoder.forward(img_found)

        img_np = img_found[0, 0].numpy()
        on_intensity = dog_map[0, 0].numpy()
        off_intensity = dog_map[0, 1].numpy()

        # ? (??gamma ?
        first_gamma = spike_train[0, :encoder.gamma_cycle_steps]  # [T, 2, H, W]

        # ?ON ?OFF ???
        on_spikes = first_gamma[:, 0, :, :]
        off_spikes = first_gamma[:, 1, :, :]

        def get_latency_image(spike_tensor):
            lat_img = np.full((28, 28), np.nan)
            for t in range(encoder.gamma_cycle_steps):
                mask = (spike_tensor[t] > 0).numpy() & np.isnan(lat_img)
                lat_img[mask] = t
            return lat_img

        on_latency = get_latency_image(on_spikes)
        off_latency = get_latency_image(off_spikes)

        # 1. ??
        axes[lbl, 0].imshow(img_np, cmap='gray')
        # --- ?2: ??---
        axes[lbl, 0].set_title(f"{label_map[lbl]} (Orig)")
        axes[lbl, 0].axis('off')

        # 2. ON ??
        axes[lbl, 1].imshow(on_intensity, cmap='Reds')
        axes[lbl, 1].set_title("ON Intensity")
        axes[lbl, 1].axis('off')

        # 3. OFF ??
        axes[lbl, 2].imshow(off_intensity, cmap='Greens')
        axes[lbl, 2].set_title("OFF Intensity")
        axes[lbl, 2].axis('off')

        # ??colormap (????
        cmap = plt.cm.get_cmap('jet_r').copy()  # Reverse jet: Red=Early(Strong), Blue=Late(Weak)
        cmap.set_bad(color='black')

        # 4. ON ?
        im1 = axes[lbl, 3].imshow(on_latency, cmap=cmap, vmin=0, vmax=encoder.encoding_window)
        axes[lbl, 3].set_title("ON Latency")
        axes[lbl, 3].axis('off')

        # 5. OFF ?
        im2 = axes[lbl, 4].imshow(off_latency, cmap=cmap, vmin=0, vmax=encoder.encoding_window)
        axes[lbl, 4].set_title("OFF Latency")
        axes[lbl, 4].axis('off')

    plt.suptitle("FashionMNIST Dual-Kernel Encoding Analysis", fontsize=16)
    plt.show()


# ???
def _resolve_torchvision_mnist_root(root):
    root_path = Path(root)
    if (root_path / "MNIST" / "raw").is_dir():
        return str(root_path)
    if root_path.name.lower() == "mnist" and (root_path / "raw").is_dir():
        return str(root_path.parent)
    return str(root_path)


def build_mnist_skeleton_loader(root="./data/MNIST", batch_size=128, input_size=28, num_workers=0):
    transform_pipeline = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    torchvision_root = _resolve_torchvision_mnist_root(root)
    train_dataset = datasets.MNIST(root=torchvision_root, train=True, download=True, transform=transform_pipeline)
    test_dataset = datasets.MNIST(root=torchvision_root, train=False, download=True, transform=transform_pipeline)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, None, test_loader


def build_fashionmnist_skeleton_loader(root="./FashionMNIST", batch_size=128, input_size=28, num_workers=0):
    transform_pipeline = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    # --- ? ?FashionMNIST ---
    train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform_pipeline)
    test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, None, test_loader

if __name__ == "__main__":
    visualize_encoding_pipeline()
