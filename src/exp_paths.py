from pathlib import Path

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
EVAL_DATASET_SIZE = 10000
SHARED_TEST_CSV = "loss_history.csv"
BOOTSTRAP_CSV = "bootstrap.csv"


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
