import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SQGNpyDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mode="iid",
        trajectory_length=4,
        trajectory_stride=1,
        transform=None,
        mmap_mode="r",
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.trajectory_length = trajectory_length
        self.trajectory_stride = trajectory_stride
        self.transform = transform
        self.mmap_mode = mmap_mode
        self.files = sorted(list(self.data_dir.glob("*.npy")))
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")
        self._file_cache = {}
        first_data = np.load(self.files[0], mmap_mode=self.mmap_mode)
        self.num_timesteps = first_data.shape[0]
        self.channels = first_data.shape[1]
        self.height = first_data.shape[2]
        self.width = first_data.shape[3]
        
        print(f"SQGNpyDataset initialized:")
        print(f"Files: {len(self.files)}")
        print(f"Mode: {mode}")
        print(f"Data shape: ({self.num_timesteps}, {self.channels}, {self.height}, {self.width})")
        
        if mode == "iid":
            self._length = len(self.files) * self.num_timesteps
        else:
            self.segments_per_file = (
                self.num_timesteps - self.trajectory_length
            ) // self.trajectory_stride + 1
            self._length = len(self.files) * self.segments_per_file
            print(f"Trajectory length: {trajectory_length}")
            print(f"Segments per file: {self.segments_per_file}")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx.item())
        if self.mode == "iid":
            return self._get_iid_sample(idx)
        else:
            return self._get_trajectory_sample(idx)

    def _get_file(self, file_idx):
        data = self._file_cache.get(file_idx, None)
        if data is None:
            data = np.load(self.files[file_idx], mmap_mode=self.mmap_mode)
            self._file_cache[file_idx] = data
            
        return data

    def _get_iid_sample(self, idx):
        file_idx = idx // self.num_timesteps
        time_idx = idx % self.num_timesteps
        data = self._get_file(file_idx)
        snapshot = np.array(data[time_idx], dtype=np.float32, copy=True)
        snapshot = torch.from_numpy(snapshot)
        if self.transform is not None:
            snapshot = self.transform(snapshot)
            
        return snapshot

    def _get_trajectory_sample(self, idx):
        file_idx = idx // self.segments_per_file
        segment_idx = idx % self.segments_per_file
        start_t = segment_idx * self.trajectory_stride
        end_t = start_t + self.trajectory_length
        data = self._get_file(file_idx)
        segment = np.array(data[start_t:end_t], dtype=np.float32, copy=True)
        segment = np.ascontiguousarray(np.transpose(segment, (1, 0, 2, 3)))
        segment = torch.from_numpy(segment)
        if self.transform is not None:
            segment = self.transform(segment)
            
        return segment


class LatentDataset(Dataset):
    def __init__(
        self,
        latent_dir,
        mode="iid",
        trajectory_length=4,
        trajectory_stride=1,
        mmap_mode="r",
    ):
        self.latent_dir = Path(latent_dir)
        self.mode = mode
        self.trajectory_length = trajectory_length
        self.trajectory_stride = trajectory_stride
        self.mmap_mode = mmap_mode
        self.files = sorted(list(self.latent_dir.glob("*_latent.npy")))
        
        if len(self.files) == 0:
            raise ValueError(f"No latent files found in {latent_dir}")
        self._file_cache = {}
        first_latent = np.load(self.files[0], mmap_mode=self.mmap_mode)
        self.num_timesteps = first_latent.shape[0]
        self.latent_channels = first_latent.shape[1]
        self.latent_h = first_latent.shape[2]
        self.latent_w = first_latent.shape[3]
        
        print(f"LatentDataset initialized:")
        print(f"Files: {len(self.files)}")
        print(f"Mode: {mode}")
        print(f"Latent shape: ({self.num_timesteps}, {self.latent_channels}, {self.latent_h}, {self.latent_w})")
        
        if mode == "iid":
            self._length = len(self.files) * self.num_timesteps
        else:
            self.segments_per_file = (
                self.num_timesteps - self.trajectory_length
            ) // self.trajectory_stride + 1
            self._length = len(self.files) * self.segments_per_file

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx.item())
        if self.mode == "iid":
            return self._get_iid_sample(idx)
        else:
            return self._get_trajectory_sample(idx)

    def _get_file(self, file_idx):
        latent = self._file_cache.get(file_idx, None)
        if latent is None:
            latent = np.load(self.files[file_idx], mmap_mode=self.mmap_mode)
            self._file_cache[file_idx] = latent
            
        return latent

    def _get_iid_sample(self, idx):
        file_idx = idx // self.num_timesteps
        time_idx = idx % self.num_timesteps
        latent_traj = self._get_file(file_idx)
        latent = np.array(latent_traj[time_idx], dtype=np.float32, copy=True)
        
        return torch.from_numpy(latent)

    def _get_trajectory_sample(self, idx):
        file_idx = idx // self.segments_per_file
        segment_idx = idx % self.segments_per_file
        start_t = segment_idx * self.trajectory_stride
        end_t = start_t + self.trajectory_length
        latent_traj = self._get_file(file_idx)
        segment = np.array(latent_traj[start_t:end_t], dtype=np.float32, copy=True)
        segment = np.ascontiguousarray(np.transpose(segment, (1, 0, 2, 3)))
        
        return torch.from_numpy(segment)
