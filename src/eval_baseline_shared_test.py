"""Re-scores a trained DeepSets baseline checkpoint against dataset_size=10000's
test split instead of its own. Checkpoint selection is unchanged (still each
run's own best-val epoch); features are normalized with the checkpoint's own
train-split mean/std (not persisted anywhere, so recomputed here from
--dataset_size's own data) rather than the eval set's stats.
"""
import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd
import torch

from exp_paths import get_dataset_dir, get_run_dir, get_results_dir, validate_dataset_size
from model import DeepSetsRegressor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo_root', type=str, default="..")
    p.add_argument('--target_columns', type=str, default="3d_x_y_z")
    p.add_argument('--dataset_size', type=int, required=True,
                    help="dataset_size the checkpoint was trained on (used to locate the run + its own train-stat file)")
    p.add_argument('--target_model', type=str, default="baseline_256")
    p.add_argument('--run_idx', type=int, required=True)
    p.add_argument('--eval_dataset_size', type=int, default=10000,
                    help="dataset_size whose test split is used as the shared/fixed evaluation set")
    p.add_argument('--hidden_dim', type=int, default=None,
                    help="optional check: read from the checkpoint; an explicit value that disagrees is an error")
    p.add_argument('--input_dim', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--output_csv_name', type=str, default="loss_history_shared_test10000.csv")
    return p.parse_args()


def hidden_dim_from_checkpoint(state_dict, override):
    """encoder.0 is Linear(input_dim, hidden_dim), so its weight is
    [hidden_dim, input_dim]. Read from the weights rather than guessed from the
    folder name, so any width (and any folder suffix) works."""
    hidden_dim = int(state_dict["encoder.0.weight"].shape[0])
    if override is not None and override != hidden_dim:
        raise ValueError(f"--hidden_dim {override} disagrees with the checkpoint ({hidden_dim})")
    return hidden_dim


def shared_baseline_h5(repo_root, target_columns, dataset_size):
    """<target_columns>_<size>/data/baseline_dataset.h5 -- one copy per size,
    shared by every DeepSets width (like OmniLearned's data/streams/)."""
    return get_dataset_dir(repo_root, target_columns, dataset_size) / "data" / "baseline_dataset.h5"


def compute_train_stats(h5_path, input_dim):
    """Reproduces StreamDataset's train-split mean/std (dataset.py) exactly."""
    with h5py.File(h5_path, 'r') as f:
        total = f['Log_Prog_Mass'].shape[0]
        train_split = int(total * 0.6)
        train_raw = f['raw_data'][:train_split].reshape(-1, input_dim)
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0) + 1e-6
    return mean, std


def load_test_split(h5_path):
    with h5py.File(h5_path, 'r') as f:
        total = f['Log_Prog_Mass'].shape[0]
        val_split = int(total * 0.8)
        raw = f['raw_data'][val_split:]
        y = f['Log_Prog_Mass'][val_split:]
    return raw, y


def find_checkpoint(run_dir):
    matches = sorted(glob.glob(os.path.join(str(run_dir), 'checkpoints', 'best_model_epoch_*.pt')))
    if not matches:
        raise FileNotFoundError(f"No best_model_epoch_*.pt found under {run_dir}/checkpoints/")
    if len(matches) > 1:
        raise RuntimeError(f"Expected exactly one best_model_epoch_*.pt under {run_dir}/checkpoints/, found {matches}")
    return matches[0]


def read_own_best_val(run_dir):
    """Best-val epoch/loss main.py already selected -- not re-selected here."""
    csv_path = os.path.join(str(run_dir), 'results', 'loss_history.csv')
    df = pd.read_csv(csv_path)
    best_idx = df['Val_Loss'].idxmin()
    return int(df.loc[best_idx, 'Epoch']), float(df.loc[best_idx, 'Val_Loss'])


def main():
    args = parse_args()
    validate_dataset_size(args.dataset_size)
    validate_dataset_size(args.eval_dataset_size)

    run_dir = get_run_dir(args.repo_root, args.target_columns, args.dataset_size, args.target_model, args.run_idx)
    if not os.path.isdir(run_dir):
        print(f"Warning: {run_dir} does not exist. Skipping.")
        return

    own_h5 = shared_baseline_h5(args.repo_root, args.target_columns, args.dataset_size)
    eval_h5 = shared_baseline_h5(args.repo_root, args.target_columns, args.eval_dataset_size)

    ckpt_path = find_checkpoint(run_dir)
    best_epoch, best_val = read_own_best_val(run_dir)

    mean, std = compute_train_stats(own_h5, args.input_dim)
    raw, y = load_test_split(eval_h5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(ckpt_path, map_location=device)
    hidden_dim = hidden_dim_from_checkpoint(state_dict, args.hidden_dim)
    model = DeepSetsRegressor(args.input_dim, hidden_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)

    preds = []
    with torch.no_grad():
        for i in range(0, raw.shape[0], args.batch_size):
            xb = torch.tensor(raw[i:i + args.batch_size], dtype=torch.float32)
            xb = (xb - mean_t) / std_t
            preds.append(model(xb.to(device)).view(-1).cpu())
    preds = torch.cat(preds, dim=0).numpy()
    y = y.reshape(-1)
    test_mse = float(np.mean((preds - y) ** 2))

    out_dir = get_results_dir(args.repo_root, args.target_columns, args.dataset_size, args.target_model, args.run_idx)
    out_path = out_dir / args.output_csv_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Epoch": best_epoch, "Val_Loss": best_val, "Test_Loss": test_mse}]).to_csv(out_path, index=False)

    # Per-sample predictions (same keys as OmniLearned's outputs__streams_*.npz)
    # for matched/paired per-stream comparisons downstream.
    npz_path = out_path.parent / "predictions.npz"
    np.savez(npz_path, prediction=preds, pid=y)

    print(f"[{run_dir}] shared test_{args.eval_dataset_size} MSE = {test_mse:.4f} "
          f"(own best_val={best_val:.4f} @ epoch {best_epoch}) -> {out_path}")


if __name__ == "__main__":
    main()
