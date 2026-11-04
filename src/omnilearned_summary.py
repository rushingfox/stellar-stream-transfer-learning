"""Turns an OmniLearned run's training_.json and test predictions into a loss_history.csv and diagnostic plots.
The OmniLearned runners call it after training; Test_Loss is filled in only at the best-val epoch.
"""
import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from exp_paths import load_target_stats

def parse_args():
    """Parses the command line.

    Returns the argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Export Train/Val/Test MSE to CSV and Plot Curves")
    parser.add_argument('--json_path', type=str, default="../training_.json")
    parser.add_argument('--npz_glob', type=str, default="../outputs__streams_*.npz",
                        help="Glob pattern matching one npz file per DDP rank. "
                             "Single-GPU runs have exactly one match (rank 0); "
                             "DDP runs (1x4, multinode) have one per rank, and all "
                             "matches are concatenated so Test MSE reflects the full "
                             "test set, not just rank 0's shard.")
    parser.add_argument('--output_csv', type=str, default="./loss_history.csv")
    parser.add_argument('--target_stats', type=str, default=None,
                        help="target_stats.npz of a downstream run's data (the train "
                             "split's target mean/std). The model then predicts the "
                             "z-scored target: predictions are mapped back to the raw "
                             "target (the npz's pid stays raw) and the train/val losses "
                             "multiplied by std^2, so every MSE in the csv is in raw "
                             "units. Omit for proxy pretrains, whose csv stays z-scored.")
    parser.add_argument('--ref', action='append', default=[],
                        help="Reference horizontal line on the R^2 plot. "
                             "Format 'LABEL:VAL1,VAL2,...' (one value per target). "
                             "Repeatable; caller (shell script) is responsible for what "
                             "the label means (e.g. paper scratch, paper finetune).")
    return parser.parse_args()


def _parse_refs(raw_refs):
    """Parses each --ref 'LABEL:V1,V2,...' into a (label, [floats]) tuple, skipping malformed ones with a warning."""
    parsed = []
    for raw in raw_refs:
        if ":" not in raw:
            print(f"Warning: --ref {raw!r} missing ':', skipping")
            continue
        label, val_str = raw.split(":", 1)
        try:
            values = [float(v) for v in val_str.split(",") if v.strip()]
        except ValueError:
            print(f"Warning: --ref {raw!r} has non-numeric values, skipping")
            continue
        if not values:
            print(f"Warning: --ref {raw!r} has no values, skipping")
            continue
        parsed.append((label.strip(), values))
    return parsed

def plot_loss_curve(train_losses, val_losses, best_epoch_num, overall_test_metric, out_path):
    """Plots train and val loss vs epoch, with the best-val epoch and its test MSE marked.

    Returns nothing; saves the figure to out_path.

    `train_losses`, `val_losses`: the per-epoch losses.
    `best_epoch_num`: the best-val epoch (1-based), or None.
    `overall_test_metric`: the test MSE at that epoch, or None/NaN to leave it out.
    `out_path`: the PNG path.
    """
    epochs = np.arange(1, len(train_losses) + 1)
    test_label = "Test MSE"

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)

    if best_epoch_num is not None:
        plt.axvline(x=best_epoch_num, color='gray', linestyle='--', alpha=0.6, label=f'Best Val Epoch ({best_epoch_num})')
        if overall_test_metric is not None and not np.isnan(overall_test_metric):
            plt.plot(best_epoch_num, overall_test_metric, marker='*', markersize=15,
                     color='#d62728', markeredgecolor='black', label=f'{test_label} ({overall_test_metric:.4f})')
            plt.annotate(f'{test_label}: {overall_test_metric:.4f}',
                         xy=(best_epoch_num, overall_test_metric),
                         xytext=(best_epoch_num + (len(epochs)*0.02), overall_test_metric),
                         fontsize=11, fontweight='bold', color='#d62728',
                         arrowprops=dict(arrowstyle='->', color='#d62728'))

    plt.title('Training/Validation Loss Curve & Best Epoch Test Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_true_vs_pred_combined(t_matrix, p_matrix, epoch_num, mse_val, out_path):
    """Scatter-plots predicted vs true values of every target, with the y = x line.

    Returns nothing; saves the figure to out_path.

    `t_matrix`, `p_matrix`: true and predicted values, of shape (n_samples, n_targets).
    `epoch_num`: the best-val epoch, shown in the title.
    `mse_val`: the test MSE, shown in the title.
    `out_path`: the PNG path.
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # up to 6D
    plt.figure(figsize=(10, 6))
    
    min_val = min(np.min(t_matrix), np.min(p_matrix))
    max_val = max(np.max(t_matrix), np.max(p_matrix))
    
    n_vars = t_matrix.shape[1]
    for i in range(n_vars):
        c = colors[i % len(colors)]
        plt.scatter(t_matrix[:, i], p_matrix[:, i], color=c, alpha=0.6, edgecolors='white', s=30, label=f'Var {i+1}')
        
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2.5, label='Perfect Prediction (y=x)')
    
    plt.title(f'Predicted vs. True Values | All Variables | Best Epoch {epoch_num} (Overall MSE: {mse_val:.4f})', fontsize=13, fontweight='bold')
    plt.xlabel('True Values (Targets)', fontsize=11)
    plt.ylabel('Predicted Values', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ratio_combined(t_matrix, p_matrix, epoch_num, mse_val, out_path):
    """Plots the predicted/true ratio vs the true value for every target, skipping samples whose true value is 0.

    Returns nothing; saves the figure to out_path.

    The y-range follows the 1st to 99th percentile of the ratios.

    `t_matrix`, `p_matrix`: true and predicted values, of shape (n_samples, n_targets).
    `epoch_num`: the best-val epoch, shown in the title.
    `mse_val`: the test MSE, shown in the title.
    `out_path`: the PNG path.
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    plt.figure(figsize=(10, 6))
    all_ratios = []

    n_vars = t_matrix.shape[1]
    for i in range(n_vars):
        t = t_matrix[:, i]
        p = p_matrix[:, i]
        mask = t != 0
        safe_t = t[mask]
        safe_p = p[mask]
        if len(safe_t) > 0:
            ratios = safe_p / safe_t
            all_ratios.extend(ratios)
            c = colors[i % len(colors)]
            plt.scatter(safe_t, ratios, color=c, alpha=0.6, edgecolors='white', s=30, label=f'Var {i+1}')

    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5, label='Perfect Prediction (Ratio=1)')
    plt.title(f'Ratio (Pred/True) vs. True | All Variables | Best Epoch {epoch_num} (Overall MSE: {mse_val:.4f})', fontsize=13, fontweight='bold')
    plt.xlabel('True Values (Targets)', fontsize=11)
    plt.ylabel('Ratio (Predicted / True)', fontsize=11)

    if all_ratios:
        y_min, y_max = np.percentile(all_ratios, [1, 99])
        plt.ylim(max(0, y_min - 0.2), y_max + 0.2)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_r2_curve(val_r2_ptt, best_epoch_num, out_path, refs=None):
    """Plots the validation R^2 of each target vs epoch.

    Returns nothing; saves the figure to out_path.

    The y-range is clipped to [-0.2, 1.05] so the 0-1 band is legible; early epochs far below 0 run off the bottom, and the title says so.

    `val_r2_ptt`: R^2 per epoch and target, of shape (n_epochs, n_targets).
    `best_epoch_num`: the best-val epoch, drawn as a vertical line, or None.
    `out_path`: the PNG path.
    `refs`: optional [(label, [v0, v1, ...])]; each label draws one horizontal line per target, in that target's color and a line style of its own, so a curve pairs with its reference value.
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    epochs = np.arange(1, val_r2_ptt.shape[0] + 1)
    n_targets = val_r2_ptt.shape[1]

    plt.figure(figsize=(10, 6))
    for i in range(n_targets):
        c = colors[i % len(colors)]
        plt.plot(epochs, val_r2_ptt[:, i], color=c, linewidth=2, label=f'Val R² target_{i}')

    if refs:
        ref_linestyles = ['--', ':', '-.']
        for r_idx, (label, values) in enumerate(refs):
            ls = ref_linestyles[r_idx % len(ref_linestyles)]
            for t, v in enumerate(values):
                if t >= n_targets:
                    break
                c = colors[t % len(colors)]
                plt.axhline(y=v, color=c, linestyle=ls, alpha=0.7, linewidth=1.5,
                            label=f'{label} target_{t} ({v:.3f})')

    if best_epoch_num is not None:
        plt.axvline(x=best_epoch_num, color='gray', linestyle='--', alpha=0.6,
                    label=f'Best Val Epoch ({best_epoch_num})')

    plt.axhline(y=0.0, color='black', linestyle=':', alpha=0.4, linewidth=1)
    plt.ylim(-0.2, 1.05)
    plt.title('Validation R² per Target vs. Epoch  (R² < -0.2 clipped)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mse_per_target(train_ptt, val_ptt, best_epoch_num, out_path):
    """Plots the train (solid) and val (dashed) MSE of each target vs epoch.

    Returns nothing; saves the figure to out_path.

    `train_ptt`, `val_ptt`: MSE per epoch and target, of shape (n_epochs, n_targets).
    `best_epoch_num`: the best-val epoch, drawn as a vertical line, or None.
    `out_path`: the PNG path.
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    epochs = np.arange(1, train_ptt.shape[0] + 1)
    n_targets = train_ptt.shape[1]

    plt.figure(figsize=(10, 6))
    for i in range(n_targets):
        c = colors[i % len(colors)]
        plt.plot(epochs, train_ptt[:, i], color=c, linewidth=2,
                 linestyle='-',  label=f'Train MSE target_{i}')
        plt.plot(epochs, val_ptt[:, i],   color=c, linewidth=2,
                 linestyle='--', label=f'Val   MSE target_{i}')

    if best_epoch_num is not None:
        plt.axvline(x=best_epoch_num, color='gray', linestyle='--', alpha=0.6,
                    label=f'Best Val Epoch ({best_epoch_num})')

    plt.title('Per-Target Train/Val MSE vs. Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Reads the training log and the predictions, then writes the CSV and the plots.

    Returns nothing.
    """
    args = parse_args()
    print(f"\nProcessing data for: {args.output_csv}")
    
    # The output dir, e.g. ./run_0/results/.
    out_dir = os.path.dirname(args.output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # --- 1. Read JSON ---
    train_losses, val_losses = [], []
    best_epoch = -1
    train_ptt = val_ptt = None
    train_r2_ptt = val_r2_ptt = None
    n_targets = 0
    if os.path.exists(args.json_path):
        try:
            with open(args.json_path, "r") as f:
                data = json.load(f)
            train_losses = data.get("train_loss", [])
            val_losses = data.get("val_loss", [])
            if len(val_losses) > 0:
                best_epoch = np.argmin(val_losses) + 1
            # Per-target fields; older training_.json files lack them, and then they are skipped.
            tptt = data.get("train_loss_per_target")
            vptt = data.get("val_loss_per_target")
            tvar = data.get("train_label_var")
            vvar = data.get("val_label_var")
            if tptt and vptt and tvar and vvar:
                train_ptt = np.asarray(tptt, dtype=float)     # [n_ep, C]
                val_ptt   = np.asarray(vptt, dtype=float)     # [n_ep, C]
                # training_.json files from before the 1-target fix may store per_target as a 1-D list when C == 1, so (n_ep,) is promoted to (n_ep, 1).
                if train_ptt.ndim == 1:
                    train_ptt = train_ptt.reshape(-1, 1)
                if val_ptt.ndim == 1:
                    val_ptt = val_ptt.reshape(-1, 1)
                train_var = np.asarray(tvar, dtype=float)     # [C]
                val_var   = np.asarray(vvar, dtype=float)     # [C]
                n_targets = train_ptt.shape[1]
                train_r2_ptt = 1.0 - train_ptt / train_var    # [n_ep, C]
                val_r2_ptt   = 1.0 - val_ptt   / val_var
        except Exception as e:
            print(f"JSON error: {e}")

    target_mean, target_std = 0.0, 1.0
    if args.target_stats:
        target_mean, target_std = load_target_stats(args.target_stats)
        print(f"Target stats: mean {target_mean:.6f}, std {target_std:.6f} (from {args.target_stats})")
        # The MSE of z-scored targets times std^2 is the MSE in raw units; the R^2 columns above are ratios of two such MSEs and need no change.
        train_losses = [v * target_std ** 2 for v in train_losses]
        val_losses = [v * target_std ** 2 for v in val_losses]
        if train_ptt is not None:
            train_ptt = train_ptt * target_std ** 2
            val_ptt = val_ptt * target_std ** 2

    # --- 2. Calculate Test metric from NPZ file(s) ---
    # Every DDP rank writes its own npz, so all matches are concatenated; rank 0 alone would hold only part of the test set (500 of 2,000 rows on 4 ranks).
    test_loss = np.nan
    predictions, targets = None, None
    npz_files = sorted(glob.glob(args.npz_glob))
    if npz_files:
        try:
            preds_list, targets_list = [], []
            for f in npz_files:
                d = np.load(f)
                preds_list.append(d["prediction"])
                targets_list.append(d["pid"])
            predictions = np.concatenate(preds_list, axis=0) * target_std + target_mean
            targets = np.concatenate(targets_list, axis=0)

            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            if targets.ndim == 1:
                targets = targets.reshape(-1, 1)
            test_loss = float(mean_squared_error(targets.flatten(), predictions.flatten()))
        except Exception as e:
            print(f"NPZ error: {e}")

    # --- 3. Generate csv data ---
    rows = []
    num_epochs = max(len(train_losses), len(val_losses))
    for i in range(num_epochs):
        ep = i + 1
        row = {
            "Epoch": ep,
            "Train_Loss": train_losses[i] if i < len(train_losses) else np.nan,
            "Val_Loss": val_losses[i] if i < len(val_losses) else np.nan,
            "Test_Loss": test_loss if ep == best_epoch else np.nan
        }
        if n_targets > 0 and i < len(train_ptt):
            for t in range(n_targets):
                row[f"Train_MSE_target_{t}"] = train_ptt[i, t]
                row[f"Val_MSE_target_{t}"]   = val_ptt[i, t]
                row[f"Train_R2_target_{t}"]  = train_r2_ptt[i, t]
                row[f"Val_R2_target_{t}"]    = val_r2_ptt[i, t]
        rows.append(row)

    test_col = "Test_Loss"
    columns = ["Epoch", "Train_Loss", "Val_Loss", test_col]
    if n_targets > 0:
        columns += [f"Train_MSE_target_{t}" for t in range(n_targets)]
        columns += [f"Val_MSE_target_{t}"   for t in range(n_targets)]
        columns += [f"Train_R2_target_{t}"  for t in range(n_targets)]
        columns += [f"Val_R2_target_{t}"    for t in range(n_targets)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {args.output_csv}")

    # --- 4. Generate Figures ---
    if len(train_losses) > 0 and len(val_losses) > 0:
        plot_loss_curve(train_losses, val_losses, best_epoch, test_loss, os.path.join(out_dir, "loss_curve.png"))
        print(f"Saved Plot: loss_curve.png")

    if n_targets > 0:
        refs = _parse_refs(args.ref)
        plot_r2_curve(val_r2_ptt, best_epoch, os.path.join(out_dir, "r2_curve.png"), refs=refs)
        print(f"Saved Plot: r2_curve.png")
        plot_mse_per_target(train_ptt, val_ptt, best_epoch, os.path.join(out_dir, "mse_per_target.png"))
        print(f"Saved Plot: mse_per_target.png")

    if predictions is not None and targets is not None:
        plot_true_vs_pred_combined(targets, predictions, best_epoch, test_loss, os.path.join(out_dir, "true_vs_pred.png"))
        plot_ratio_combined(targets, predictions, best_epoch, test_loss, os.path.join(out_dir, "ratio.png"))
        print(f"Saved Plots: true_vs_pred.png, ratio.png")

if __name__ == "__main__":
    main()