import numpy as np
import torch
from pathlib import Path
import yaml


def compute_dataset_stats(data_dir, num_samples=1000, save_path=None):
    data_dir = Path(data_dir)
    files = sorted(list(data_dir.glob("*.npy")))
    if len(files) == 0:
        raise ValueError(f"No .npy files found in {data_dir}")
    print(f"Computing statistics from {len(files)} files...")
    all_data = []
    samples_collected = 0
    for file in files:
        data = np.load(file)
        T = data.shape[0]
        n_samples = min(num_samples - samples_collected, T)
        indices = np.random.choice(T, n_samples, replace=False)
        all_data.append(data[indices])
        samples_collected += n_samples
        if samples_collected >= num_samples:
            break
    all_data = np.concatenate(all_data, axis=0)
    mean = all_data.mean(axis=(0, 2, 3), keepdims=True)
    std = all_data.std(axis=(0, 2, 3), keepdims=True)
    mean = torch.from_numpy(mean[0]).float()
    std = torch.from_numpy(std[0]).float()
    print(f"Mean: {mean.squeeze().tolist()}")
    print(f"Std: {std.squeeze().tolist()}")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()}
        with open(save_path, "w") as f:
            yaml.dump(stats, f)
        print(f"Saved statistics to {save_path}")
    return (mean, std)


def load_dataset_stats(path):
    with open(path, "r") as f:
        stats = yaml.safe_load(f)
    mean = torch.tensor(stats["mean"]).float().view(-1, 1, 1)
    std = torch.tensor(stats["std"]).float().view(-1, 1, 1)
    return (mean, std)


def normalize(x, mean, std):
    return (x - mean) / (std + 1e-05)


def denormalize(x, mean, std):
    return x * std + mean
