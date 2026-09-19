"""bootstrap_uncertainty.py — decompose the reported Test MSE error bars into
two distinct sources of uncertainty:

  seed_std      — std of Test MSE across independently trained run_idxs
                  (training randomness: init, data order, etc.). This is what
                  the repo's existing error bars already measure.
  bootstrap_std — for a SINGLE trained run, resample the (fixed) test set with
                  replacement --n_boot times and recompute Test MSE each time;
                  std of that distribution. This is what OmniCosmos reports
                  (literature/2512.24422.pdf, Sec. IV): "bootstrapping the test
                  data 1000 times with replacement", no retraining involved.

No retraining or re-evaluation needed: per-sample (prediction, true label)
pairs are already saved in each run's outputs__streams_*.npz.

Usage:
    module load conda && conda activate stream_transfer_learning
    python src/bootstrap_uncertainty.py
    python src/bootstrap_uncertainty.py --target_model omnilearned_proxy_cube_finetune_1x4 --dataset_size 1000
"""
import argparse
import glob

import numpy as np
from sklearn.metrics import mean_squared_error

from exp_paths import get_run_dir, validate_dataset_size


def load_run_predictions(repo_root, target_columns, dataset_size, target_model, run_idx):
    """Concatenate this run's outputs__streams_*.npz (one per DDP rank) into a
    single (n_test, n_targets) predictions/targets pair. Returns (None, None)
    if the run has no saved outputs."""
    run_dir = get_run_dir(repo_root, target_columns, dataset_size, target_model, run_idx)
    npz_files = sorted(glob.glob(str(run_dir / "outputs__streams_*.npz")))
    if not npz_files:
        return None, None
    preds_list, targets_list = [], []
    for f in npz_files:
        d = np.load(f)
        preds_list.append(d["prediction"])
        targets_list.append(d["pid"])
    predictions = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    return predictions, targets


def bootstrap_mse(predictions, targets, n_boot, seed):
    """Resample rows (test samples) with replacement n_boot times; return the
    array of n_boot MSE values. Matches omnilearned_summary.py's Test MSE
    formula (mean_squared_error on flattened arrays) so results are comparable."""
    rng = np.random.default_rng(seed)
    n_test = predictions.shape[0]
    boot_mses = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_test, size=n_test)
        boot_mses[b] = mean_squared_error(targets[idx].flatten(), predictions[idx].flatten())
    return boot_mses


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target_columns", default="3d_x_y_z")
    p.add_argument("--dataset_size", type=int, default=100)
    p.add_argument("--target_model", default="omnilearned_proxy_gaussian_ellipsoid_sigmax_finetune_1x4",
                    help="Model folder name (default: the best N=100 performer).")
    p.add_argument("--run_indices", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repo_root", default=".")
    return p.parse_args()


def main():
    args = parse_args()
    validate_dataset_size(args.dataset_size)

    per_run = []
    for run_idx in args.run_indices:
        predictions, targets = load_run_predictions(
            args.repo_root, args.target_columns, args.dataset_size, args.target_model, run_idx
        )
        if predictions is None:
            print(f"[skip] run_{run_idx}: no outputs__streams_*.npz found")
            continue
        actual_mse = float(mean_squared_error(targets.flatten(), predictions.flatten()))
        boot_mses = bootstrap_mse(predictions, targets, args.n_boot, seed=args.seed + run_idx)
        per_run.append({
            "run_idx": run_idx,
            "n_test": predictions.shape[0],
            "actual_mse": actual_mse,
            "boot_mean": float(np.mean(boot_mses)),
            "boot_std": float(np.std(boot_mses)),
        })

    if not per_run:
        raise SystemExit("No runs with saved outputs found.")

    print(f"\n{args.target_model} @ {args.target_columns}_{args.dataset_size}")
    print(f"{'run':>4}{'n_test':>8}{'Test MSE':>14}{'boot mean':>14}{'boot std':>12}")
    for r in per_run:
        print(f"{r['run_idx']:>4}{r['n_test']:>8}{r['actual_mse']:>14.5f}"
              f"{r['boot_mean']:>14.5f}{r['boot_std']:>12.5f}")

    actual_mses = [r["actual_mse"] for r in per_run]
    seed_std = float(np.std(actual_mses))
    mean_boot_std = float(np.mean([r["boot_std"] for r in per_run]))

    print(f"\nseed_std (across {len(per_run)} runs, training randomness): {seed_std:.5f}")
    print(f"mean bootstrap_std (test-set resampling, OmniCosmos-style):   {mean_boot_std:.5f}")
    if mean_boot_std > 0:
        print(f"seed_std / bootstrap_std ratio: {seed_std / mean_boot_std:.2f}")


if __name__ == "__main__":
    main()
