from pathlib import Path

VALID_DATASET_SIZES = (20, 100, 300, 1000, 10000)

# Regression tasks. dir_prefix goes in front of the experiment folder name, so
# prog_mass (empty prefix) keeps the plain <target_columns>_<size>/ folders;
# raw_prefix names raw_data/<raw_prefix>_<size>.h5; label_key is the label
# dataset in baseline_dataset.h5. scripts/task_config.sh mirrors the first two.
TASKS = {
    "prog_mass": {"dir_prefix": "",         "raw_prefix": "prog_mass_reg_dataset", "label_key": "Log_Prog_Mass"},
    "subhalo":   {"dir_prefix": "subhalo_", "raw_prefix": "subhalo_reg_dataset",   "label_key": "Log_Pert_M10pc"},
}
DEFAULT_TASK = "prog_mass"
LABEL_KEYS = tuple(t["label_key"] for t in TASKS.values())


def validate_dataset_size(dataset_size):
    size = int(dataset_size)
    if size not in VALID_DATASET_SIZES:
        raise ValueError(
            f"Unsupported dataset_size={size}. Expected one of {VALID_DATASET_SIZES}."
        )
    return size


def find_label_key(h5_file):
    """The label dataset of an open baseline_dataset.h5: whichever task's label key it holds."""
    for key in LABEL_KEYS:
        if key in h5_file:
            return key
    raise KeyError(f"No label dataset ({', '.join(LABEL_KEYS)}) in {h5_file.filename}")


def dataset_key(target_columns, dataset_size, task=DEFAULT_TASK):
    size = validate_dataset_size(dataset_size)
    return f"{TASKS[task]['dir_prefix']}{target_columns}_{size}"


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
