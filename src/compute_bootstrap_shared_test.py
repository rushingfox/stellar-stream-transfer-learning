"""Bootstrap error bar (OmniCosmos-style test-set resampling, see
bootstrap_uncertainty.py) for a single fixed run's shared-test10000
predictions, instead of the usual cross-seed std.

For each of --run_idxs (default just run_0, since bootstrap fixes one
trained model and resamples its test set -- it isn't a cross-seed
statistic): loads that run's shared-test predictions (works for
OmniLearned-family checkpoints and
baseline_256/average/eigenvalue_regression alike), computes the actual MSE
and resamples the (fixed) 2000 shared-test streams with replacement --n_boot
times to get bootstrap_std.

Usage:
    python compute_bootstrap_shared_test.py \
        --repo_root .. --target_columns 3d_x_y_z --dataset_size 100 \
        --target_model omnilearned_proxy_cube_finetune_1x4 --run_idxs 0
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from bootstrap_uncertainty import bootstrap_mse
from exp_paths import get_run_dir, get_results_dir


def load_predictions(run_dir, eval_dataset_size):
    """Returns (prediction, truth) 1-D arrays for the shared test split, or
    None if this run has no shared-test predictions saved yet."""
    omni_path = os.path.join(str(run_dir), f"shared_test{eval_dataset_size}", "outputs__streams_0.npz")
    generic_path = os.path.join(str(run_dir), "results", f"shared_test{eval_dataset_size}", "predictions.npz")
    if os.path.exists(omni_path):
        d = np.load(omni_path)
    elif os.path.exists(generic_path):
        d = np.load(generic_path)
    else:
        return None
    return d["prediction"].reshape(-1), d["pid"].reshape(-1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo_root', type=str, default="..")
    p.add_argument('--target_columns', type=str, default="3d_x_y_z")
    p.add_argument('--dataset_size', type=int, required=True)
    p.add_argument('--target_model', type=str, required=True)
    p.add_argument('--run_idxs', nargs='+', type=int, default=[0],
                    help="Which run(s) to bootstrap (default: just run_0 -- bootstrap "
                         "fixes a single trained model, it's not a cross-seed statistic).")
    p.add_argument('--eval_dataset_size', type=int, default=10000)
    p.add_argument('--n_boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output_csv_name', type=str, default=None,
                    help="Default: shared_test<eval_dataset_size>_bootstrap/loss_history.csv")
    return p.parse_args()


def main():
    args = parse_args()
    eval_ds = args.eval_dataset_size
    output_csv_name = args.output_csv_name or f"shared_test{eval_ds}_bootstrap/loss_history.csv"

    for idx in args.run_idxs:
        run_dir = get_run_dir(args.repo_root, args.target_columns, args.dataset_size, args.target_model, idx)
        loaded = load_predictions(run_dir, eval_ds)
        if loaded is None:
            print(f"[skip] no shared-test predictions for {run_dir}")
            continue
        pred, true = loaded
        actual_mse = float(mean_squared_error(true, pred))
        boot_mses = bootstrap_mse(pred, true, args.n_boot, seed=args.seed + idx)
        bootstrap_std = float(np.std(boot_mses))

        out_dir = get_results_dir(args.repo_root, args.target_columns, args.dataset_size, args.target_model, idx)
        out_path = out_dir / output_csv_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "Epoch": 1,
            "Val_Loss": actual_mse,
            "Test_Loss": actual_mse,
            "Test_Loss_Bootstrap_Std": bootstrap_std,
        }]).to_csv(out_path, index=False)
        print(f"[{run_dir}] MSE={actual_mse:.4f}  bootstrap_std={bootstrap_std:.4f} "
              f"(n_boot={args.n_boot}, n_test={pred.shape[0]}) -> {out_path}")


if __name__ == "__main__":
    main()
