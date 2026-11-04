import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from exp_paths import DEFAULT_TASK, TASKS, VALID_DATASET_SIZES, find_label_key, get_dataset_dir, get_results_dir, get_shared_test_dir


def parse_args():
    """Reads the dataset size, run indices and optional shared-test size from the command line.

    Returns the argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate a simple global-average baseline result."
    )
    parser.add_argument(
        "--target_columns",
        type=str,
        default="3d_x_y_z",
        help="Feature set key, e.g. 3d_x_y_z",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=list(TASKS),
        default=DEFAULT_TASK,
        help="Regression task (sets the experiment-folder prefix).",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        choices=VALID_DATASET_SIZES,
        required=True,
        help="Dataset size used in first-level folder naming.",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[0],
        help="Run indices to emit, e.g. --indices 0 1 2",
    )
    parser.add_argument(
        "--eval_dataset_size",
        type=int,
        choices=VALID_DATASET_SIZES,
        default=None,
        help="If set, score mean_train against this dataset_size's test split "
             "instead of --dataset_size's own (train split unaffected). Writes "
             "run_<idx>/shared_test<N>/loss_history.csv (+ predictions.npz) instead "
             "of run_<idx>/results/loss_history.csv.",
    )
    return parser.parse_args()


def main():
    """Computes the mean-prediction baseline: the train split's mean target, predicted for every stream.

    Returns nothing; writes run_<idx>/results/loss_history.csv for each --indices, or with --eval_dataset_size run_<idx>/shared_test<N>/loss_history.csv and predictions.npz, all in raw target units.
    """
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent.parent

    # The shared DeepSets dataset of this size; only its raw targets are used.
    h5_path = get_dataset_dir(repo_dir, args.target_columns, args.dataset_size, args.task) / "data" / "deepsets_dataset.h5"
    
    if not h5_path.exists():
        print(f"Error: H5 file not found at {h5_path}. Please run scripts/prepare_data_deepsets.sh first.")
        return

    with h5py.File(h5_path, 'r') as f:
        y = f[find_label_key(f)][:]
        
    total_samples = y.shape[0]
    train_split = int(total_samples * 0.6)
    val_split = int(total_samples * 0.8)

    y_train = y[:train_split]
    y_val = y[train_split:val_split]
    y_test = y[val_split:]

    mean_train = np.mean(y_train)

    train_loss = np.mean((y_train - mean_train) ** 2)
    val_loss = np.mean((y_val - mean_train) ** 2)

    if args.eval_dataset_size is None:
        test_loss = np.mean((y_test - mean_train) ** 2)
    else:
        eval_h5_path = get_dataset_dir(repo_dir, args.target_columns, args.eval_dataset_size, args.task) / "data" / "deepsets_dataset.h5"
        with h5py.File(eval_h5_path, 'r') as f:
            y_eval = f[find_label_key(f)][:]
        eval_val_split = int(y_eval.shape[0] * 0.8)
        y_shared_test = y_eval[eval_val_split:]
        test_loss = np.mean((y_shared_test - mean_train) ** 2)
        # Per-stream predictions in raw units, under the same keys as OmniLearned's outputs__streams_*.npz (pid is the raw target), read downstream by compute_bootstrap_shared_test.py.
        # The prediction is the train mean, repeated for every stream.
        shared_pred = np.full_like(y_shared_test, mean_train, dtype=np.float64)

    # The results go in an "average" model folder, with the same numbers for every index, since the baseline has no randomness.
    for idx in args.indices:
        if args.eval_dataset_size is None:
            out_dir = get_results_dir(repo_dir, args.target_columns, args.dataset_size, "average", idx, task=args.task)
        else:
            out_dir = get_shared_test_dir(repo_dir, args.target_columns, args.dataset_size, "average", idx,
                                          eval_dataset_size=args.eval_dataset_size, task=args.task)
        out_path = out_dir / "loss_history.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            "Epoch": [1],
            "Train_Loss": [train_loss],
            "Val_Loss": [val_loss],
            "Test_Loss": [test_loss]
        })
        df.to_csv(out_path, index=False)

        if args.eval_dataset_size is not None:
            np.savez(out_path.parent / "predictions.npz", prediction=shared_pred, pid=y_shared_test)

    print(f"Successfully generated global Average Dummy results!")
    print(f"Dummy Test MSE: {test_loss:.6f}")

if __name__ == "__main__":
    main()
