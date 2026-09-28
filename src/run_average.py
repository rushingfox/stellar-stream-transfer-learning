import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from exp_paths import VALID_DATASET_SIZES, get_dataset_dir, get_results_dir


def parse_args():
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
             "to results/shared_test<N>/loss_history.csv instead of loss_history.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent.parent

    # The shared DeepSets dataset for this size (only its labels are used).
    h5_path = get_dataset_dir(repo_dir, args.target_columns, args.dataset_size) / "data" / "baseline_dataset.h5"
    
    if not h5_path.exists():
        print(f"Error: H5 file not found at {h5_path}. Please generate 3D baseline data first.")
        return

    with h5py.File(h5_path, 'r') as f:
        y = f['Log_Prog_Mass'][:]
        
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
        csv_name = "loss_history.csv"
    else:
        eval_h5_path = get_dataset_dir(repo_dir, args.target_columns, args.eval_dataset_size) / "data" / "baseline_dataset.h5"
        with h5py.File(eval_h5_path, 'r') as f:
            y_eval = f['Log_Prog_Mass'][:]
        eval_val_split = int(y_eval.shape[0] * 0.8)
        y_shared_test = y_eval[eval_val_split:]
        test_loss = np.mean((y_shared_test - mean_train) ** 2)
        csv_name = f"shared_test{args.eval_dataset_size}/loss_history.csv"
        # Per-sample predictions (same keys as OmniLearned's outputs__streams_*.npz)
        # for matched/paired per-stream comparisons downstream. Prediction is the
        # same mean_train scalar repeated for every stream.
        shared_pred = np.full_like(y_shared_test, mean_train, dtype=np.float64)

    # Use a dedicated "average" model directory under the dataset key.
    for idx in args.indices:
        res_dir = get_results_dir(
            repo_dir,
            args.target_columns,
            args.dataset_size,
            "average",
            idx,
        )
        out_path = res_dir / csv_name
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
