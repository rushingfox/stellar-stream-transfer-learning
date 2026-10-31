from pathlib import Path

import numpy as np

VALID_DATASET_SIZES = (20, 40, 100, 300, 1000, 10000)

# Regression tasks. dir_prefix goes in front of the experiment folder name
# (<dir_prefix><target_columns>_<size>/); raw_prefix names
# raw_data/<raw_prefix>_<size>.h5; label_key is the label dataset in
# deepsets_dataset.h5. scripts/task_config.sh mirrors the first two.
TASKS = {
    "prog_mass": {"dir_prefix": "prog_mass_", "raw_prefix": "prog_mass_reg_dataset", "label_key": "Log_Prog_Mass"},
    "subhalo":   {"dir_prefix": "subhalo_",   "raw_prefix": "subhalo_reg_dataset",   "label_key": "Log_Pert_M10pc"},
}
DEFAULT_TASK = "prog_mass"
LABEL_KEYS = tuple(t["label_key"] for t in TASKS.values())

# Pretraining datasets (the synthetic proxies and the stream-eigenvalue proxy)
# belong to no downstream task, so their folders never get a task prefix.
PRETRAIN_COLUMN_PREFIXES = ("3d_proxy_", "3d_x_y_z_eigenvalues")

# Every downstream model is also scored on one shared test set: the test split
# of this dataset size. Each run keeps everything from that evaluation in
# run_<idx>/shared_test<N>/: the predictions (outputs__streams_0.npz for
# OmniLearned, predictions.npz otherwise), loss_history.csv and bootstrap.csv.
# OmniLearned reads that test split from its own size's data/shared_test<N>/,
# a copy whose inputs are normalized with that size's input stats.
EVAL_DATASET_SIZE = 10000
SHARED_TEST_CSV = "loss_history.csv"
BOOTSTRAP_CSV = "bootstrap.csv"

# Inputs and targets are normalized with each dataset's own train split, never
# with another dataset's. The inputs, x, y, z, are divided by one length scale,
# the std of all train x, y and z values pooled together, and nothing is
# subtracted: shapes stay undistorted and the origin stays where it is. Every
# split, test included, is transformed (input_stats.npz in data/streams/ holds
# a per-axis mean, all 0, and std, all that one scale). The downstream target
# is z-scored in train and val only; test keeps the raw target, and predictions
# are mapped back to it with target_stats.npz before scoring.
INPUT_STATS_NPZ = "input_stats.npz"
TARGET_STATS_NPZ = "target_stats.npz"


def validate_dataset_size(dataset_size):
    size = int(dataset_size)
    if size not in VALID_DATASET_SIZES:
        raise ValueError(
            f"Unsupported dataset_size={size}. Expected one of {VALID_DATASET_SIZES}."
        )
    return size


def find_label_key(h5_file):
    """The label dataset of an open deepsets_dataset.h5: whichever task's label key it holds."""
    for key in LABEL_KEYS:
        if key in h5_file:
            return key
    raise KeyError(f"No label dataset ({', '.join(LABEL_KEYS)}) in {h5_file.filename}")


def dataset_key(target_columns, dataset_size, task=DEFAULT_TASK):
    size = validate_dataset_size(dataset_size)
    prefix = "" if target_columns.startswith(PRETRAIN_COLUMN_PREFIXES) else TASKS[task]["dir_prefix"]
    return f"{prefix}{target_columns}_{size}"


def get_dataset_dir(repo_root, target_columns, dataset_size, task=DEFAULT_TASK):
    return Path(repo_root) / dataset_key(target_columns, dataset_size, task)


def get_model_dir(repo_root, target_columns, dataset_size, target_model, task=DEFAULT_TASK):
    return get_dataset_dir(repo_root, target_columns, dataset_size, task) / target_model


def get_target_stats_path(repo_root, target_columns, dataset_size, task=DEFAULT_TASK):
    return get_dataset_dir(repo_root, target_columns, dataset_size, task) / "data" / "streams" / TARGET_STATS_NPZ


def load_target_stats(path):
    """(mean, std) of the train split's target, as prepare_data.py saved them."""
    stats = np.load(path)
    return float(stats["mean"]), float(stats["std"])


def compute_input_stats(train_inputs):
    """(mean, std) arrays, one entry per axis, for a train split of shape
    (n_clouds, n_particles, 3) holding x, y, z; inputs are then normalized as
    (x - mean) / std. The mean is 0 and every std is the same number: the std
    of all x, y and z values pooled over every particle of every cloud."""
    if train_inputs.shape[-1] != 3:
        raise ValueError(f"Expected x, y, z inputs, got {train_inputs.shape[-1]} features")
    scale = train_inputs.std(dtype=np.float64)
    return np.zeros(3), np.full(3, scale)


def load_input_stats(path):
    """(mean, std) arrays, one entry per input feature, of the train split's
    inputs, as prepare_data.py saved them."""
    stats = np.load(path)
    return stats["mean"], stats["std"]


def get_data_dir(repo_root, target_columns, dataset_size, target_model, task=DEFAULT_TASK):
    return get_model_dir(repo_root, target_columns, dataset_size, target_model, task) / "data"


def get_data_file_path(repo_root, target_columns, dataset_size, target_model, filename, task=DEFAULT_TASK):
    return get_data_dir(repo_root, target_columns, dataset_size, target_model, task) / filename


def get_run_dir(repo_root, target_columns, dataset_size, target_model, run_idx, task=DEFAULT_TASK):
    return get_model_dir(repo_root, target_columns, dataset_size, target_model, task) / f"run_{run_idx}"


def get_results_dir(repo_root, target_columns, dataset_size, target_model, run_idx, task=DEFAULT_TASK):
    return get_run_dir(repo_root, target_columns, dataset_size, target_model, run_idx, task) / "results"


def get_loss_history_csv_path(repo_root, target_columns, dataset_size, target_model, run_idx, filename="loss_history.csv", task=DEFAULT_TASK):
    return get_results_dir(repo_root, target_columns, dataset_size, target_model, run_idx, task) / filename


def shared_test_dirname(eval_dataset_size=EVAL_DATASET_SIZE):
    return f"shared_test{eval_dataset_size}"


def get_shared_test_dir(repo_root, target_columns, dataset_size, target_model, run_idx, eval_dataset_size=EVAL_DATASET_SIZE, task=DEFAULT_TASK):
    return get_run_dir(repo_root, target_columns, dataset_size, target_model, run_idx, task) / shared_test_dirname(eval_dataset_size)
