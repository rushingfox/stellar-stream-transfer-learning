"""Scores a trained DeepSets checkpoint on the shared test set (the test split of --eval_dataset_size, 10000 by default) instead of its own test split.

The checkpoint is still the run's own best-val epoch, as main.py selected it.
The model was trained with inputs and target normalized by the train-split statistics of its own --dataset_size, which are not saved anywhere, so they are recomputed here from that size's data.
The shared test inputs are normalized with those input stats, and the predictions are mapped back to raw target units with pred * std + mean.
Writes run_<idx>/shared_test<N>/loss_history.csv and predictions.npz, both in raw units.
"""
import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd
import torch

from dataset import train_input_stats, train_target_stats
from exp_paths import DEFAULT_TASK, TASKS, find_label_key, get_dataset_dir, get_run_dir, get_shared_test_dir, validate_dataset_size
from model import DeepSetsRegressor


def parse_args():
    """Reads which run to score, and on which shared test set, from the command line.

    Returns the argparse namespace.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo_root', type=str, default="..")
    p.add_argument('--target_columns', type=str, default="3d_x_y_z")
    p.add_argument('--task', type=str, choices=list(TASKS), default=DEFAULT_TASK)
    p.add_argument('--dataset_size', type=int, required=True,
                    help="dataset_size the checkpoint was trained on (used to locate the run + its own train-stat file)")
    p.add_argument('--target_model', type=str, default="deepsets")
    p.add_argument('--run_idx', type=int, required=True)
    p.add_argument('--eval_dataset_size', type=int, default=10000,
                    help="dataset_size whose test split is used as the shared/fixed evaluation set")
    p.add_argument('--hidden_dim', type=int, default=None,
                    help="optional check: read from the checkpoint; an explicit value that disagrees is an error")
    p.add_argument('--input_dim', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=32)
    return p.parse_args()


def hidden_dim_from_checkpoint(state_dict, override):
    """Reads the DeepSets hidden width from the checkpoint weights rather than the folder name, so any width and any folder suffix works.

    Returns hidden_dim; raises ValueError if `override` disagrees with it.

    `state_dict`: the DeepSets state_dict; encoder.0 is Linear(input_dim, hidden_dim), so its weight is [hidden_dim, input_dim].
    `override`: --hidden_dim, or None to skip the check.
    """
    hidden_dim = int(state_dict["encoder.0.weight"].shape[0])
    if override is not None and override != hidden_dim:
        raise ValueError(f"--hidden_dim {override} disagrees with the checkpoint ({hidden_dim})")
    return hidden_dim


def shared_deepsets_h5(repo_root, target_columns, dataset_size, task):
    """Gives the path of one size's DeepSets dataset, a single copy shared by every DeepSets run at that size (like OmniLearned's data/streams/).

    Returns <task>_<target_columns>_<size>/data/deepsets_dataset.h5 under repo_root.

    `repo_root`: the repo root, where the experiment folders are.
    `target_columns`, `dataset_size`: pick the experiment folder, e.g. 3d_x_y_z and 1000.
    `task`: 'prog_mass' or 'subhalo', the experiment folder prefix.
    """
    return get_dataset_dir(repo_root, target_columns, dataset_size, task) / "data" / "deepsets_dataset.h5"


def load_test_split(h5_path):
    """Loads the test split (the last 20%) of a deepsets_dataset.h5, unnormalized.

    Returns (raw, y): the raw x, y, z coordinates of every particle and the raw targets.

    `h5_path`: the deepsets_dataset.h5 to read.
    """
    with h5py.File(h5_path, 'r') as f:
        label_key = find_label_key(f)
        total = f[label_key].shape[0]
        val_split = int(total * 0.8)
        raw = f['raw_data'][val_split:]
        y = f[label_key][val_split:]
    return raw, y


def find_checkpoint(run_dir):
    """Finds the best-val checkpoint main.py saved and renamed after training.

    Returns the path of the one checkpoints/best_model_epoch_*.pt; raises if there is none or more than one.

    `run_dir`: the run_<idx> folder.
    """
    matches = sorted(glob.glob(os.path.join(str(run_dir), 'checkpoints', 'best_model_epoch_*.pt')))
    if not matches:
        raise FileNotFoundError(f"No best_model_epoch_*.pt found under {run_dir}/checkpoints/")
    if len(matches) > 1:
        raise RuntimeError(f"Expected exactly one best_model_epoch_*.pt under {run_dir}/checkpoints/, found {matches}")
    return matches[0]


def read_own_best_val(run_dir):
    """Reads the best-val epoch and loss that main.py already selected; nothing is re-selected here.

    Returns (epoch, val_loss) of the lowest-Val_Loss row of results/loss_history.csv.

    `run_dir`: the run_<idx> folder.
    """
    csv_path = os.path.join(str(run_dir), 'results', 'loss_history.csv')
    df = pd.read_csv(csv_path)
    best_idx = df['Val_Loss'].idxmin()
    return int(df.loc[best_idx, 'Epoch']), float(df.loc[best_idx, 'Val_Loss'])


def main():
    """Scores one DeepSets run on the shared test set.

    Returns nothing; writes loss_history.csv (best-val epoch, its val loss and the shared test MSE) and predictions.npz to the run's shared_test<N>/, or skips with a warning if the run folder does not exist.
    """
    args = parse_args()
    validate_dataset_size(args.dataset_size)
    validate_dataset_size(args.eval_dataset_size)

    run_dir = get_run_dir(args.repo_root, args.target_columns, args.dataset_size, args.target_model, args.run_idx, args.task)
    if not os.path.isdir(run_dir):
        print(f"Warning: {run_dir} does not exist. Skipping.")
        return

    own_h5 = shared_deepsets_h5(args.repo_root, args.target_columns, args.dataset_size, args.task)
    eval_h5 = shared_deepsets_h5(args.repo_root, args.target_columns, args.eval_dataset_size, args.task)

    ckpt_path = find_checkpoint(run_dir)
    best_epoch, best_val = read_own_best_val(run_dir)

    target_mean, target_std = train_target_stats(own_h5)
    input_mean, input_std = train_input_stats(own_h5)
    raw, y = load_test_split(eval_h5)
    raw = (raw - input_mean) / input_std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(ckpt_path, map_location=device)
    hidden_dim = hidden_dim_from_checkpoint(state_dict, args.hidden_dim)
    model = DeepSetsRegressor(args.input_dim, hidden_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, raw.shape[0], args.batch_size):
            xb = torch.tensor(raw[i:i + args.batch_size], dtype=torch.float32)
            preds.append(model(xb.to(device)).view(-1).cpu())
    preds = torch.cat(preds, dim=0).numpy() * target_std + target_mean
    y = y.reshape(-1)
    test_mse = float(np.mean((preds - y) ** 2))

    out_dir = get_shared_test_dir(args.repo_root, args.target_columns, args.dataset_size, args.target_model, args.run_idx,
                                  eval_dataset_size=args.eval_dataset_size, task=args.task)
    out_path = out_dir / "loss_history.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Epoch": best_epoch, "Val_Loss": best_val, "Test_Loss": test_mse}]).to_csv(out_path, index=False)

    # Per-stream predictions under the same keys as OmniLearned's outputs__streams_*.npz (pid is the raw target), read downstream by compute_bootstrap_shared_test.py.
    # Unlike OmniLearned's z-scored predictions, these are already in raw units.
    npz_path = out_path.parent / "predictions.npz"
    np.savez(npz_path, prediction=preds, pid=y)

    print(f"[{run_dir}] shared test_{args.eval_dataset_size} MSE = {test_mse:.4f} "
          f"(own best_val={best_val:.4f} @ epoch {best_epoch}) -> {out_path}")


if __name__ == "__main__":
    main()
