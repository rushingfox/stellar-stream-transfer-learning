"""Shared constants and path helpers: task settings, experiment folder names and normalization-stat file names."""
from pathlib import Path

import numpy as np

VALID_DATASET_SIZES = (20, 40, 100, 300, 1000, 10000)

# Regression tasks.
# dir_prefix goes in front of the experiment folder name (<dir_prefix><target_columns>_<size>/), raw_prefix names raw_data/<raw_prefix>_<size>.h5, and label_key is the name of the target dataset in deepsets_dataset.h5.
# scripts/task_config.sh mirrors the first two.
TASKS = {
    "prog_mass": {"dir_prefix": "prog_mass_", "raw_prefix": "prog_mass_reg_dataset", "label_key": "Log_Prog_Mass"},
    "subhalo":   {"dir_prefix": "subhalo_",   "raw_prefix": "subhalo_reg_dataset",   "label_key": "Log_Pert_M10pc"},
}
DEFAULT_TASK = "prog_mass"
LABEL_KEYS = tuple(t["label_key"] for t in TASKS.values())

# Pretraining datasets (the synthetic proxies and the stream-eigenvalue proxy) belong to no downstream task, so their folders never get a task prefix.
PRETRAIN_COLUMN_PREFIXES = ("3d_proxy_", "3d_x_y_z_eigenvalues")

# Every downstream model is also scored on one shared test set, the test split of this dataset size.
# Each run keeps everything from that evaluation in run_<idx>/shared_test<N>/: the predictions (outputs__streams_0.npz for OmniLearned, predictions.npz otherwise), loss_history.csv and bootstrap.csv.
# OmniLearned reads that test split from its own size's data/shared_test<N>/, a copy whose inputs are normalized with that size's input stats.
EVAL_DATASET_SIZE = 10000
SHARED_TEST_CSV = "loss_history.csv"
BOOTSTRAP_CSV = "bootstrap.csv"

# Inputs and targets are normalized with each dataset's own train split, never with another dataset's.
# The inputs x, y, z are shifted by each axis's train mean and divided by one length scale shared by all three axes, so shapes stay undistorted.
# Every split, test included, is transformed; input_stats.npz in data/streams/ holds the per-axis mean and std (all that one scale).
# The downstream target is z-scored in train and val only; test keeps the raw target, and predictions are mapped back with target_stats.npz before scoring.
INPUT_STATS_NPZ = "input_stats.npz"
TARGET_STATS_NPZ = "target_stats.npz"


def validate_dataset_size(dataset_size):
    """Checks that a dataset size is one of VALID_DATASET_SIZES.

    Returns the size as an int; raises ValueError otherwise.

    `dataset_size`: the size, as an int or a string.
    """
    size = int(dataset_size)
    if size not in VALID_DATASET_SIZES:
        raise ValueError(
            f"Unsupported dataset_size={size}. Expected one of {VALID_DATASET_SIZES}."
        )
    return size


def find_label_key(h5_file):
    """Finds which task's target dataset an open deepsets_dataset.h5 holds.

    Returns the dataset name (one of the TASKS label_key values); raises KeyError if there is none.

    `h5_file`: the open h5py file.
    """
    for key in LABEL_KEYS:
        if key in h5_file:
            return key
    raise KeyError(f"No label dataset ({', '.join(LABEL_KEYS)}) in {h5_file.filename}")


def dataset_key(target_columns, dataset_size, task=DEFAULT_TASK):
    """Builds the experiment folder name of a dataset.

    Returns e.g. 'prog_mass_3d_x_y_z_1000', or '3d_proxy_cube_10000' for a pretraining dataset, which has no task prefix.

    `target_columns`: the input columns, e.g. '3d_x_y_z', or a pretraining dataset's name, e.g. '3d_proxy_cube'.
    `dataset_size`: the total number of streams (train+val+test).
    `task`: 'prog_mass' or 'subhalo', the folder prefix.
    """
    size = validate_dataset_size(dataset_size)
    prefix = "" if target_columns.startswith(PRETRAIN_COLUMN_PREFIXES) else TASKS[task]["dir_prefix"]
    return f"{prefix}{target_columns}_{size}"


def get_dataset_dir(repo_root, target_columns, dataset_size, task=DEFAULT_TASK):
    """Gives the dataset folder, <repo_root>/<dataset_key>/.

    Returns a Path.

    `repo_root`: the repo root.
    `target_columns`, `dataset_size`, `task`: as in dataset_key.
    """
    return Path(repo_root) / dataset_key(target_columns, dataset_size, task)


def get_model_dir(repo_root, target_columns, dataset_size, target_model, task=DEFAULT_TASK):
    """Gives a model's folder, <dataset dir>/<target_model>/.

    Returns a Path.

    `repo_root`, `target_columns`, `dataset_size`, `task`: as in get_dataset_dir.
    `target_model`: the model folder name, e.g. 'omnilearned_scratch'.
    """
    return get_dataset_dir(repo_root, target_columns, dataset_size, task) / target_model


def get_target_stats_path(repo_root, target_columns, dataset_size, task=DEFAULT_TASK):
    """Gives the path of a dataset's target_stats.npz, the train split's target mean and std.

    Returns <dataset dir>/data/streams/target_stats.npz as a Path.

    `repo_root`, `target_columns`, `dataset_size`, `task`: as in get_dataset_dir.
    """
    return get_dataset_dir(repo_root, target_columns, dataset_size, task) / "data" / "streams" / TARGET_STATS_NPZ


def load_target_stats(path):
    """Loads the train split's target mean and std, as prepare_data.py saved them.

    Returns (mean, std) as floats.

    `path`: a target_stats.npz.
    """
    stats = np.load(path)
    return float(stats["mean"]), float(stats["std"])


def compute_input_stats(train_inputs):
    """Computes the input normalization statistics of a train split.

    Returns (mean, std) arrays with one entry per axis, applied as (x - mean) / std: the mean is each axis's mean over every particle of every cloud, and every std is the same number, the std of all x, y and z values pooled after each axis is centred, i.e. the square root of the average of the three axis variances.

    `train_inputs`: the train split, of shape (n_clouds, n_particles, 3) holding x, y, z.
    """
    if train_inputs.shape[-1] != 3:
        raise ValueError(f"Expected x, y, z inputs, got {train_inputs.shape[-1]} features")
    mean = train_inputs.mean(axis=(0, 1), dtype=np.float64)
    scale = np.sqrt(train_inputs.var(axis=(0, 1), dtype=np.float64).mean())
    return mean, np.full(3, scale)


def load_input_stats(path):
    """Loads the train split's input mean and std, as prepare_data.py saved them.

    Returns (mean, std) arrays with one entry per input feature.

    `path`: an input_stats.npz.
    """
    stats = np.load(path)
    return stats["mean"], stats["std"]


def get_data_dir(repo_root, target_columns, dataset_size, target_model, task=DEFAULT_TASK):
    """Gives a model's own data folder, <model dir>/data/.

    Returns a Path.

    `repo_root`, `target_columns`, `dataset_size`, `target_model`, `task`: as in get_model_dir.
    """
    return get_model_dir(repo_root, target_columns, dataset_size, target_model, task) / "data"


def get_run_dir(repo_root, target_columns, dataset_size, target_model, run_idx, task=DEFAULT_TASK):
    """Gives one run's folder, <model dir>/run_<run_idx>/.

    Returns a Path.

    `repo_root`, `target_columns`, `dataset_size`, `target_model`, `task`: as in get_model_dir.
    `run_idx`: the run index (seed).
    """
    return get_model_dir(repo_root, target_columns, dataset_size, target_model, task) / f"run_{run_idx}"


def get_results_dir(repo_root, target_columns, dataset_size, target_model, run_idx, task=DEFAULT_TASK):
    """Gives a run's results folder, <run dir>/results/.

    Returns a Path.

    `repo_root`, `target_columns`, `dataset_size`, `target_model`, `run_idx`, `task`: as in get_run_dir.
    """
    return get_run_dir(repo_root, target_columns, dataset_size, target_model, run_idx, task) / "results"


def get_loss_history_csv_path(repo_root, target_columns, dataset_size, target_model, run_idx, filename="loss_history.csv", task=DEFAULT_TASK):
    """Gives the path of a run's CSV in its results folder.

    Returns <run dir>/results/<filename> as a Path.

    `repo_root`, `target_columns`, `dataset_size`, `target_model`, `run_idx`, `task`: as in get_run_dir.
    `filename`: the CSV name, loss_history.csv by default.
    """
    return get_results_dir(repo_root, target_columns, dataset_size, target_model, run_idx, task) / filename


def shared_test_dirname(eval_dataset_size=EVAL_DATASET_SIZE):
    """Gives the name of a run's shared-test folder.

    Returns e.g. 'shared_test10000'.

    `eval_dataset_size`: the size whose test split is the shared test set (10000 by default).
    """
    return f"shared_test{eval_dataset_size}"


def get_shared_test_dir(repo_root, target_columns, dataset_size, target_model, run_idx, eval_dataset_size=EVAL_DATASET_SIZE, task=DEFAULT_TASK):
    """Gives a run's shared-test folder, <run dir>/shared_test<N>/.

    Returns a Path.

    `repo_root`, `target_columns`, `dataset_size`, `target_model`, `run_idx`, `task`: as in get_run_dir.
    `eval_dataset_size`: as in shared_test_dirname.
    """
    return get_run_dir(repo_root, target_columns, dataset_size, target_model, run_idx, task) / shared_test_dirname(eval_dataset_size)
