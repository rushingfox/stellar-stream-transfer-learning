from pathlib import Path

VALID_DATASET_SIZES = (20, 100, 300, 1000, 10000)


def validate_dataset_size(dataset_size):
    size = int(dataset_size)
    if size not in VALID_DATASET_SIZES:
        raise ValueError(
            f"Unsupported dataset_size={size}. Expected one of {VALID_DATASET_SIZES}."
        )
    return size


def dataset_key(target_columns, dataset_size):
    size = validate_dataset_size(dataset_size)
    return f"{target_columns}_{size}"


def get_dataset_dir(repo_root, target_columns, dataset_size):
    return Path(repo_root) / dataset_key(target_columns, dataset_size)


def get_model_dir(repo_root, target_columns, dataset_size, target_model):
    return get_dataset_dir(repo_root, target_columns, dataset_size) / target_model


def get_data_dir(repo_root, target_columns, dataset_size, target_model):
    return get_model_dir(repo_root, target_columns, dataset_size, target_model) / "data"


def get_data_file_path(repo_root, target_columns, dataset_size, target_model, filename):
    return get_data_dir(repo_root, target_columns, dataset_size, target_model) / filename


def get_run_dir(repo_root, target_columns, dataset_size, target_model, run_idx):
    return get_model_dir(repo_root, target_columns, dataset_size, target_model) / f"run_{run_idx}"


def get_results_dir(repo_root, target_columns, dataset_size, target_model, run_idx):
    return get_run_dir(repo_root, target_columns, dataset_size, target_model, run_idx) / "results"


def get_loss_history_csv_path(repo_root, target_columns, dataset_size, target_model, run_idx, filename="loss_history.csv"):
    return get_results_dir(repo_root, target_columns, dataset_size, target_model, run_idx) / filename
