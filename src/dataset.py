import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from exp_paths import compute_input_stats, find_label_key


def train_target_stats(h5_path):
    """Computes the mean and std of the train split's target (the first 60%), which the model learns to predict z-scored.

    Returns (mean, std) as floats, where std is the population std (np.std); these are the same numbers that prepare_data.py saves for OmniLearned in data/streams/target_stats.npz, since both use the same raw targets and split.

    `h5_path`: a deepsets_dataset.h5.
    """
    with h5py.File(h5_path, 'r') as f:
        y = f[find_label_key(f)]
        y = y[:int(y.shape[0] * 0.6)]
    return float(np.mean(y)), float(np.std(y))


def train_input_stats(h5_path):
    """Computes the input statistics of the train split (the first 60%) with exp_paths.compute_input_stats.

    Returns (mean, std), one entry per axis: the mean is each axis's train mean and every std is the one shared length scale; the same numbers prepare_data.py saves for OmniLearned in data/streams/input_stats.npz (same raw coordinates, same split).

    `h5_path`: a deepsets_dataset.h5.
    """
    with h5py.File(h5_path, 'r') as f:
        x = f['raw_data']
        x = x[:int(x.shape[0] * 0.6)]
    return compute_input_stats(x)


class StreamDataset(Dataset):
    """One split of deepsets_dataset.h5, with the inputs normalized and the target z-scored with the train split's statistics, as OmniLearned gets them.

    `h5_path`: the deepsets_dataset.h5; the splits are its first 60% (train), next 20% (val) and last 20% (test).
    `mode`: 'train', 'val' or 'test'.
    """
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
        """Number of streams in the split."""
        return len(self.active_indices)

    def __getitem__(self, idx):
        """(normalized inputs, z-scored target, row index in the file) of the idx-th stream of the split; the file is opened on first use, in each DataLoader worker."""
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        real_idx = self.active_indices[idx]
        label = (self.file[self.label_key][real_idx] - self.target_mean) / self.target_std
        label = torch.tensor(label, dtype=torch.float32)

        data = torch.tensor(self.file['raw_data'][real_idx], dtype=torch.float32)
        data = (data - self.input_mean) / self.input_std

        return data, label, real_idx

    def __del__(self):
        """Closes the HDF5 file if it was opened."""
        if self.file is not None:
            self.file.close()
