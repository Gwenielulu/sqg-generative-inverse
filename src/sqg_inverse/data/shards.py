from collections import OrderedDict
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset


class SQGShardDataset(Dataset):

    def __init__(self, root, channels, resolution):
        self.root = Path(root)
        self.channels = int(channels)
        self.resolution = int(resolution)
        self.cache_size = 2
        self.paths = sorted(self.root.rglob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No .npy shards found under {self.root}")
        self.index = []
        for file_index, path in enumerate(self.paths):
            array = np.load(path, mmap_mode="r")
            expected_tail = (self.channels, self.resolution, self.resolution)
            if array.ndim != 4 or tuple(array.shape[1:]) != expected_tail:
                raise ValueError(
                    f"{path}: expected (N,{expected_tail[0]},{expected_tail[1]},{expected_tail[2]}), got {array.shape}"
                )
            self.index.extend(
                ((file_index, sample_index) for sample_index in range(array.shape[0]))
            )
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.index)

    def _array(self, file_index):
        if file_index in self._cache:
            array = self._cache.pop(file_index)
            self._cache[file_index] = array
            return array
        array = np.load(self.paths[file_index], mmap_mode="r")
        self._cache[file_index] = array
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return array

    def __getitem__(self, index):
        file_index, sample_index = self.index[index]
        sample = np.array(
            self._array(file_index)[sample_index], dtype=np.float32, copy=True
        )
        return (sample, {})


def infinite_loader(root, batch_size, channels, resolution, num_workers):
    dataset = SQGShardDataset(root, channels=channels, resolution=resolution)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    while True:
        yield from loader
