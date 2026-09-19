import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class StreamDataset(Dataset):
    def __init__(self, h5_path, INPUT_DIM, mode='train'):
        self.h5_path = h5_path
        self.mode = mode
        self.file = None

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"file not found: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            total_samples = f['Log_Prog_Mass'].shape[0]
            
            train_split = int(total_samples * 0.6)
            val_split = int(total_samples * 0.8)

            if mode == 'train':
                self.active_indices = np.arange(0, train_split)
            elif mode == 'val':
                self.active_indices = np.arange(train_split, val_split)
            elif mode == 'test':
                self.active_indices = np.arange(val_split, total_samples)
            else:
                raise ValueError("Mode must be 'train', 'val', or 'test'")

            print(f"Mode {mode}: indices from {self.active_indices[0]} to {self.active_indices[-1]} (Total: {len(self.active_indices)})")

            if mode == 'train':
                print(f"Calculating stats from training set ({len(self.active_indices)} samples)...")
                subset_data = f['raw_data'][self.active_indices] 
                flat_data = subset_data.reshape(-1, INPUT_DIM)
                self.mean = torch.tensor(flat_data.mean(axis=0), dtype=torch.float32)
                self.std = torch.tensor(flat_data.std(axis=0), dtype=torch.float32) + 1e-6

    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        real_idx = self.active_indices[idx]
        label = torch.tensor(self.file['Log_Prog_Mass'][real_idx], dtype=torch.float32)

        data = torch.tensor(self.file['raw_data'][real_idx], dtype=torch.float32)
        if hasattr(self, 'mean'):
            data = (data - self.mean) / self.std

        return data, label, real_idx

    def __del__(self):
        if self.file is not None:
            self.file.close()
