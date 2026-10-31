"""Bootstrap error bar (OmniCosmos-style test-set resampling, see
bootstrap_uncertainty.py) for a single fixed run's shared-test10000
predictions, instead of the usual cross-seed std.

For each of --run_idxs (default just run_0, since bootstrap fixes one
trained model and resamples its test set -- it isn't a cross-seed
statistic): loads that run's shared-test predictions (works for
OmniLearned-family checkpoints and
DeepSets/average/eigenvalue_regression alike), computes the actual MSE
and resamples the (fixed) 2000 shared-test streams with replacement --n_boot
times to get bootstrap_std.

The output, run_<idx>/shared_test<N>/bootstrap.csv, is one row that also
carries the run's best-val Epoch and Val_Loss, copied from the row of
shared_test<N>/loss_history.csv that the shared-test evaluation scored, so
every column of it means what its name says.

Usage:
    python compute_bootstrap_shared_test.py \
        --repo_root .. --target_columns 3d_x_y_z --dataset_size 100 \
        --target_model omnilearned_proxy_cube_finetune --run_idxs 0
    python compute_bootstrap_shared_test.py --run_dir <run_<idx> dir>   # what the training runners call
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from bootstrap_uncertainty import bootstrap_mse
from exp_paths import (BOOTSTRAP_CSV, DEFAULT_TASK, SHARED_TEST_CSV, TARGET_STATS_NPZ, TASKS, get_run_dir,
                       get_target_stats_path, load_target_stats, shared_test_dirname)


def load_predictions(run_dir, eval_dataset_size, target_stats_path):
    """Returns (prediction, truth) 1-D arrays for the shared test split, both
    in raw target units, or None if this run has no shared-test predictions
    saved yet. OmniLearned predicts the target z-scored with its own size's
    train stats (target_stats_path) and is mapped back here; predictions.npz
    (DeepSets and the non-ML baselines) is raw already."""
    shared_dir = os.path.join(str(run_dir), shared_test_dirname(eval_dataset_size))
    omni_path = os.path.join(shared_dir, "outputs__streams_0.npz")
    generic_path = os.path.join(shared_dir, "predictions.npz")
    if os.path.exists(omni_path):
        d = np.load(omni_path)
        mean, std = load_target_stats(target_stats_path)
        return d["prediction"].reshape(-1) * std + mean, d["pid"].reshape(-1)
    if os.path.exists(generic_path):
        d = np.load(generic_path)
        return d["prediction"].reshape(-1), d["pid"].reshape(-1)
    return None


def best_val_row(shared_dir):
    """(Epoch, Val_Loss) of the best-val row that the shared-test evaluation
    scored: the only row of shared_test<N>/loss_history.csv with a Test_Loss."""
    csv_path = shared_dir / SHARED_TEST_CSV
    # utf-8-sig strips the BOM that OmniLearned CSVs start with; round_trip
    # parses the floats exactly, so the copied Val_Loss matches bit for bit.
    df = pd.read_csv(csv_path, encoding="utf-8-sig", float_precision="round_trip")
    scored = df[df["Test_Loss"].notna()]
    if len(scored) != 1:
        raise ValueError(f"Expected exactly one row with Test_Loss in {csv_path}, found {len(scored)}")
    return int(scored["Epoch"].iloc[0]), float(scored["Val_Loss"].iloc[0])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo_root', type=str, default="..")
    p.add_argument('--target_columns', type=str, default="3d_x_y_z")
    p.add_argument('--task', type=str, choices=list(TASKS), default=DEFAULT_TASK)
    p.add_argument('--dataset_size', type=int)
    p.add_argument('--target_model', type=str)
    p.add_argument('--run_dir', type=str, default=None,
                    help="One <task>_<target_columns>_<size>/<model>/run_<idx> dir instead of "
                         "--dataset_size/--target_model/--run_idxs; its target stats are read "
                         "from ../../data/streams/.")
    p.add_argument('--run_idxs', nargs='+', type=int, default=[0],
                    help="Which run(s) to bootstrap (default: just run_0 -- bootstrap "
                         "fixes a single trained model, it's not a cross-seed statistic).")
    p.add_argument('--eval_dataset_size', type=int, default=10000)
    p.add_argument('--n_boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    eval_ds = args.eval_dataset_size

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        idx = int(re.fullmatch(r"run_(\d+)", run_dir.name).group(1))
        runs = [(idx, run_dir, run_dir.parent.parent / "data" / "streams" / TARGET_STATS_NPZ)]
    else:
        if args.dataset_size is None or args.target_model is None:
            raise SystemExit("Give --dataset_size and --target_model, or --run_dir")
        stats_path = get_target_stats_path(args.repo_root, args.target_columns, args.dataset_size, args.task)
        runs = [(idx, get_run_dir(args.repo_root, args.target_columns, args.dataset_size, args.target_model, idx, args.task),
                 stats_path) for idx in args.run_idxs]

    for idx, run_dir, stats_path in runs:
        loaded = load_predictions(run_dir, eval_ds, stats_path)
        if loaded is None:
            print(f"[skip] no shared-test predictions for {run_dir}")
            continue
        pred, true = loaded
        actual_mse = float(mean_squared_error(true, pred))
        boot_mses = bootstrap_mse(pred, true, args.n_boot, seed=args.seed + idx)
        bootstrap_std = float(np.std(boot_mses))

        shared_dir = run_dir / shared_test_dirname(eval_ds)
        best_epoch, best_val = best_val_row(shared_dir)
        out_path = shared_dir / BOOTSTRAP_CSV
        pd.DataFrame([{
            "Epoch": best_epoch,
            "Val_Loss": best_val,
            "Test_Loss": actual_mse,
            "Test_Loss_Bootstrap_Std": bootstrap_std,
        }]).to_csv(out_path, index=False)
        print(f"[{run_dir}] MSE={actual_mse:.4f}  bootstrap_std={bootstrap_std:.4f} "
              f"(n_boot={args.n_boot}, n_test={pred.shape[0]}) -> {out_path}")


if __name__ == "__main__":
    main()
