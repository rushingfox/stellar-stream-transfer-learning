import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from exp_paths import compute_input_stats, find_label_key


def train_target_stats(h5_path):
    """Mean and std of the train split's target (the first 60%), which the model
    is trained to predict z-scored -- the same numbers prepare_data.py saves for
    OmniLearned in data/streams/target_stats.npz (same raw targets, same split)."""
    with h5py.File(h5_path, 'r') as f:
        y = f[find_label_key(f)]
        y = y[:int(y.shape[0] * 0.6)]
    return float(np.mean(y)), float(np.std(y))


def train_input_stats(h5_path):
    """Per-feature (mean, std) of the train split's inputs, from
    exp_paths.compute_input_stats -- the same numbers prepare_data.py saves for
    OmniLearned in data/streams/input_stats.npz (same raw coordinates, same split)."""
    with h5py.File(h5_path, 'r') as f:
        x = f['raw_data']
        x = x[:int(x.shape[0] * 0.6)]
    return compute_input_stats(x)


class StreamDataset(Dataset):
    """One split of deepsets_dataset.h5, with the inputs normalized and the
    target z-scored with the train split's statistics, as OmniLearned gets them."""
    def __init__(self, h5_path, mode='train'):
        self.h5_path = h5_path
        self.mode = mode
        self.file = None

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"file not found: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            self.label_key = find_label_key(f)
            total_samples = f[self.label_key].shape[0]
            
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

        self.target_mean, self.target_std = train_target_stats(h5_path)
        input_mean, input_std = train_input_stats(h5_path)
        self.input_mean = torch.tensor(input_mean, dtype=torch.float32)
        self.input_std = torch.tensor(input_std, dtype=torch.float32)

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        real_idx = self.active_indices[idx]
        label = (self.file[self.label_key][real_idx] - self.target_mean) / self.target_std
        label = torch.tensor(label, dtype=torch.float32)

        data = torch.tensor(self.file['raw_data'][real_idx], dtype=torch.float32)
        data = (data - self.input_mean) / self.input_std

        return data, label, real_idx

    def __del__(self):
        if self.file is not None:
            self.file.close()
