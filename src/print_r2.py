"""Print shared-test R^2 per run, plus an "R2 ± total (bootstrap)" summary per model
(the same numbers as the appendix table, for the same runs).

    python src/print_r2.py SIZE:MODEL[:RUNS] [SIZE:MODEL[:RUNS] ...]

    python src/print_r2.py \\
        300:omnilearned_scratch_1x4 \\
        300:omnilearned_proxy_gaussian_ball_finetune_1x4:0,1,2 \\
        300:eigenvalue_regression:0

MODEL is the on-disk folder name; RUNS defaults to 0,1,2. Each run needs its
shared-test evaluation and bootstrap; a missing file is an error, not a skipped run.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from build_results_table import (  # noqa: E402
    compute_test_label_variance_from_labels,
    load_one_run,
    summarize_model,
)
from exp_paths import get_loss_history_csv_path, get_model_dir, validate_dataset_size  # noqa: E402

EVAL_DATASET_SIZE = 10000
SHARED_CSV = f"shared_test{EVAL_DATASET_SIZE}/loss_history.csv"
BOOTSTRAP_CSV = f"shared_test{EVAL_DATASET_SIZE}_bootstrap/loss_history.csv"
LABELS_NPY = REPO_ROOT / "evaluation" / f"prog_mass_labels_{EVAL_DATASET_SIZE}_test.npy"
# The bootstrap CSV has one pre-selected row, so no epoch cutoff is needed.
NO_EPOCH_CUTOFF = 10**9


def parse_spec(spec):
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not all(parts[:2]):
        raise SystemExit(f"Bad spec '{spec}': expected SIZE:MODEL[:RUNS], e.g. 300:omnilearned_scratch_1x4:0,1,2")
    size = validate_dataset_size(parts[0])
    runs = [0, 1, 2]
    if len(parts) == 3:
        try:
            runs = [int(r) for r in parts[2].split(",")]
        except ValueError:
            raise SystemExit(f"Bad RUNS in '{spec}': expected comma-separated integers, e.g. 3,4,5")
    return size, parts[1], runs


def best_epoch(run_files):
    """(best-val epoch, total epochs); None if there is no training curve
    (eigenvalue_regression). Test_Loss is filled only on the best-val row."""
    # utf-8-sig strips the BOM that OmniLearned CSVs start with.
    n_epochs = len(pd.read_csv(run_files["train"], encoding="utf-8-sig"))
    if n_epochs <= 1:
        return None
    df = pd.read_csv(run_files["shared"], encoding="utf-8-sig")
    hit = df[df["Test_Loss"].notna()]
    if len(hit) != 1:
        raise SystemExit(f"Expected exactly one row with Test_Loss in {run_files['shared']}, found {len(hit)}")
    return int(hit["Epoch"].iloc[0]), n_epochs


def required_files(target_columns, size, model, idx):
    def path(name):
        return get_loss_history_csv_path(REPO_ROOT, target_columns, size, model, idx, filename=name)
    return {"train": path("loss_history.csv"), "shared": path(SHARED_CSV), "bootstrap": path(BOOTSTRAP_CSV)}


def report(target_columns, size, model, runs, var):
    model_dir = get_model_dir(REPO_ROOT, target_columns, size, model)
    if not model_dir.is_dir():
        raise SystemExit(f"Model folder not found: {model_dir}")

    files = {idx: required_files(target_columns, size, model, idx) for idx in runs}
    missing = [str(p) for f in files.values() for p in f.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing file(s) -- check the run indices, or run the shared-test "
                         "evaluation / bootstrap for these runs first:\n  " + "\n  ".join(missing))

    print(f"{target_columns}_{size} / {model}   (shared_test{EVAL_DATASET_SIZE})")
    r2_values = []
    for idx in runs:
        _, mse, boot = load_one_run(str(files[idx]["bootstrap"]), NO_EPOCH_CUTOFF,
                                    error_column="Test_Loss_Bootstrap_Std")
        r2 = 1.0 - mse / var
        r2_values.append(r2)
        ep = best_epoch(files[idx])
        ep_str = f"{ep[0]:>3}/{ep[1]:<3}" if ep else "   —   "
        print(f"  run_{idx}: best-val epoch {ep_str}  R2 {r2:.3f}   (bootstrap {boot / var:.3f})")

    # Same call as the appendix table: total = cross-run std (+) mean bootstrap std.
    _, row = summarize_model(REPO_ROOT, target_columns, size, model, runs, NO_EPOCH_CUTOFF,
                             test_csv_name=BOOTSTRAP_CSV, error_column="Test_Loss_Bootstrap_Std",
                             combine_errors=True)
    mean_r2 = 1.0 - row["Mean Test MSE"] / var
    total = row["Std Test MSE"] / var
    # Single run: no cross-run term, so total = bootstrap.
    boot_mean = row.get("Stat Std Test MSE", row["Std Test MSE"]) / var
    print(f"  R2 = {mean_r2:.3f} ± {total:.3f} ({boot_mean:.3f})   "
          f"min {min(r2_values):.3f}   max {max(r2_values):.3f}   (n={len(runs)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="+", metavar="SIZE:MODEL[:RUNS]")
    parser.add_argument("--target_columns", default="3d_x_y_z")
    args = parser.parse_args()

    specs = [parse_spec(s) for s in args.specs]
    var = compute_test_label_variance_from_labels(str(LABELS_NPY))
    for i, (size, model, runs) in enumerate(specs):
        if i:
            print()
        report(args.target_columns, size, model, runs, var)


if __name__ == "__main__":
    main()
