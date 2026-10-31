"""Print shared-test R^2 per run, plus an "R2 ± total (bootstrap)" summary per model
(the same numbers as the appendix table, for the same runs).

    python src/print_r2.py SIZE:MODEL[:RUNS] [SIZE:MODEL[:RUNS] ...]

    python src/print_r2.py \\
        300:omnilearned_scratch \\
        300:omnilearned_proxy_gaussian_ball_finetune:0,1,2 \\
        300:eigenvalue_regression:0

MODEL is the on-disk folder name; RUNS defaults to 0,1,2. Each run needs its
shared-test evaluation and bootstrap; a missing file is an error, not a skipped run.
--task picks the runs (prog_mass_ or subhalo_<target_columns>_<size>/) and that task's
evaluation/<task>_labels_10000_test.npy. --compact prints only the summary, one line
per model (N = training streams, 3/5 of SIZE; readable model names), with the same numbers;
--sort_r2 then orders each N's lines by mean R2, highest first, instead of the command order.

    python src/print_r2.py --table

prints the appendix table in plain text instead, and takes no SIZE:MODEL: every model of
the paper's figure and table (rows) at every size (columns), "R2 ± total", with * on the
best mean of each column and † on the scratch lr chosen by validation loss.
"""
import argparse
import itertools
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from build_results_table import (  # noqa: E402
    SCRATCH_LR_PREFIX,
    SCRATCH_MODEL,
    compute_test_label_variance_from_labels,
    load_one_run,
    select_scratch_lr,
    summarize_model,
)
from exp_paths import (  # noqa: E402
    BOOTSTRAP_CSV,
    DEFAULT_TASK,
    EVAL_DATASET_SIZE,
    SHARED_TEST_CSV,
    TASKS,
    dataset_key,
    get_loss_history_csv_path,
    get_model_dir,
    get_shared_test_dir,
    shared_test_dirname,
    validate_dataset_size,
)

# The bootstrap CSV has one pre-selected row, so no epoch cutoff is needed.
NO_EPOCH_CUTOFF = 10**9

# Model names for --compact: the figure's names for the paper's models (in plain text),
# with the folder's _lr<LR> / _ep<N> suffix spelled out. A folder not matched here
# prints as is.
_BASE_NAMES = {
    "average": "Mean Baseline",
    "eigenvalue_regression": "Eigenvalue Regression",
    "deepsets": "DeepSets",
    "deepsets_256": "DeepSets (256)",
    "deepsets_128": "DeepSets (128)",
    "omnilearned_scratch": "OmniLearned (Scratch)",
    "omnilearned_jet_finetune": "OmniLearned (HEP Jets)",
    "omnicosmos_camels_finetune": "OmniCosmos-C",
    "omnicosmos_quijote_finetune": "OmniCosmos-Q",
}
_PROXY_NAMES = {
    "cube": "Cube",
    "bounded_ball": "Bounded Ball",
    "gaussian_ball": "Gaussian Ball",
    "box_a": "Box",
    "bounded_ellipsoid_a": "Bounded Ellipsoid",
    "gaussian_ellipsoid_sigmax": "Gaussian Ellipsoid",
    "eigen_lambda1": "Real-Stream PCA (λ1)",
    "eigen_lambda2": "Real-Stream PCA (λ2)",
    "eigen_lambda3": "Real-Stream PCA (λ3)",
}
# What each task regresses, for the --compact header (both tasks take the same input).
_TASK_INPUT = "stream particle xyz (4000 per stream)"
_TASK_TARGETS = {
    "prog_mass": "log10 mass of the stream's progenitor",
    "subhalo": "log10 M(<10 pc) of the subhalo that perturbed the stream",
}
# The scratch runner's defaults, shown so its rows compare with the lr/epoch variants.
_SCRATCH_DEFAULT_LR, _SCRATCH_DEFAULT_EPOCHS = "5e-5", "100"

# --table: the rows and columns of the appendix table
# (evaluation/results_3d_shared_test10000.sh), as (folder, name) in table order.
_TABLE_SIZES = (40, 100, 300, 1000)
_TABLE_SCRATCH_LRS = ("1e-4", "2e-4", "5e-4", "1e-3")
_TABLE_MODELS = (
    [("eigenvalue_regression", "Eigenvalue Regression"),
     ("deepsets", "DeepSets"),
     # Not a folder: per size, the scratch lr below with the lowest mean best-val loss.
     (_VAL_SELECTED_ROW := "val_selected_scratch", "OmniLearned (Scratch, val-selected lr)"),
     ("omnilearned_scratch", f"OmniLearned (Scratch, lr {_SCRATCH_DEFAULT_LR})")]
    + [(f"omnilearned_scratch_lr{lr}", f"OmniLearned (Scratch, lr {lr})") for lr in _TABLE_SCRATCH_LRS]
    + [("omnilearned_jet_finetune", "OmniLearned (HEP Jets)"),
       ("omnicosmos_camels_finetune", "OmniCosmos-C"),
       ("omnicosmos_quijote_finetune", "OmniCosmos-Q"),
       ("omnilearned_proxy_cube_finetune", "Cube"),
       ("omnilearned_proxy_bounded_ball_finetune", "Bounded Ball"),
       ("omnilearned_proxy_gaussian_ball_finetune", "Gaussian Ball"),
       ("omnilearned_proxy_box_a_finetune", "Box"),
       ("omnilearned_proxy_bounded_ellipsoid_a_finetune", "Bounded Ellipsoid"),
       ("omnilearned_proxy_gaussian_ellipsoid_sigmax_finetune", "Gaussian Ellipsoid"),
       ("omnilearned_proxy_eigen_lambda1_finetune", "Real-Stream PCA")]
)


def display_name(folder):
    """Readable name for a model folder, e.g. omnilearned_proxy_cube_finetune
    -> 'Cube', omnilearned_scratch_lr1e-4_ep50 -> 'OmniLearned (Scratch),
    lr 1e-4, 50 ep'."""
    base, lr, epochs = re.fullmatch(r"(.+?)(?:_lr([0-9.e-]+))?(?:_ep(\d+))?", folder).groups()
    proxy = re.fullmatch(r"omnilearned_proxy_(.+?)_finetune", base)
    if proxy and proxy.group(1) in _PROXY_NAMES:
        name = _PROXY_NAMES[proxy.group(1)]
    elif base in _BASE_NAMES:
        name = _BASE_NAMES[base]
    else:
        return folder
    if base == "omnilearned_scratch":
        lr, epochs = lr or _SCRATCH_DEFAULT_LR, epochs or _SCRATCH_DEFAULT_EPOCHS
    details = ([f"lr {lr}"] if lr else []) + ([f"{epochs} ep"] if epochs else [])
    return ", ".join([name] + details)


def parse_spec(spec):
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not all(parts[:2]):
        raise SystemExit(f"Bad spec '{spec}': expected SIZE:MODEL[:RUNS], e.g. 300:omnilearned_scratch:0,1,2")
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


def required_files(target_columns, size, model, idx, task):
    shared_dir = get_shared_test_dir(REPO_ROOT, target_columns, size, model, idx, task=task)
    return {"train": get_loss_history_csv_path(REPO_ROOT, target_columns, size, model, idx, task=task),
            "shared": shared_dir / SHARED_TEST_CSV, "bootstrap": shared_dir / BOOTSTRAP_CSV}


def report(target_columns, size, model, runs, var, task, compact_width=None):
    """compact_width: instead of printing, return (mean R2, summary line) with the model
    name padded to this width."""
    model_dir = get_model_dir(REPO_ROOT, target_columns, size, model, task)
    if not model_dir.is_dir():
        raise SystemExit(f"Model folder not found: {model_dir}")

    compact = compact_width is not None
    files = {idx: required_files(target_columns, size, model, idx, task) for idx in runs}
    # The training curve only supplies the best-val epoch, which compact mode does not
    # print, so it is not required there (the subhalo non-ML baselines have none).
    missing = [str(p) for f in files.values() for name, p in f.items()
               if not p.exists() and not (compact and name == "train")]
    if missing:
        raise SystemExit("Missing file(s) -- check the run indices, or run the shared-test "
                         "evaluation / bootstrap for these runs first:\n  " + "\n  ".join(missing))

    if not compact:
        print(f"{dataset_key(target_columns, size, task)} / {model}   ({shared_test_dirname()})")
    r2_values = []
    for idx in runs:
        _, mse, boot = load_one_run(str(files[idx]["bootstrap"]), NO_EPOCH_CUTOFF,
                                    error_column="Test_Loss_Bootstrap_Std")
        r2 = 1.0 - mse / var
        r2_values.append(r2)
        if compact:
            continue
        ep = best_epoch(files[idx])
        ep_str = f"{ep[0]:>3}/{ep[1]:<3}" if ep else "   —   "
        print(f"  run_{idx}: best-val epoch {ep_str}  R2 {r2:.3f}   (bootstrap {boot / var:.3f})")

    # Same call as the appendix table: total = cross-run std (+) mean bootstrap std.
    _, row = summarize_model(REPO_ROOT, target_columns, size, model, runs, NO_EPOCH_CUTOFF,
                             test_csv_name=f"{shared_test_dirname()}/{BOOTSTRAP_CSV}",
                             error_column="Test_Loss_Bootstrap_Std", combine_errors=True, task=task)
    mean_r2 = 1.0 - row["Mean Test MSE"] / var
    total = row["Std Test MSE"] / var
    # Single run: no cross-run term, so total = bootstrap.
    boot_mean = row.get("Stat Std Test MSE", row["Std Test MSE"]) / var
    if compact:
        return mean_r2, (f"{int(size * 0.6):>5}  {display_name(model):<{compact_width}}  "
                         f"{mean_r2:6.3f} ± {total:.3f} ({boot_mean:.3f})  "
                         f"{min(r2_values):6.3f}  {max(r2_values):6.3f}  {len(runs)}")
    print(f"  R2 = {mean_r2:.3f} ± {total:.3f} ({boot_mean:.3f})   "
          f"min {min(r2_values):.3f}   max {max(r2_values):.3f}   (n={len(runs)})")


def print_table(target_columns, var, task):
    """The appendix table in plain text: the same numbers, bold best -> *, dagger -> †."""
    rows = []  # plot rows of summarize_model, named like the figure so select_scratch_lr applies
    for size in _TABLE_SIZES:
        for folder, _ in _TABLE_MODELS:
            if folder == _VAL_SELECTED_ROW:
                continue
            plot_name = (SCRATCH_MODEL if folder == "omnilearned_scratch"
                         else folder.replace("omnilearned_scratch_lr", SCRATCH_LR_PREFIX))
            _, row = summarize_model(REPO_ROOT, target_columns, size, folder, [0, 1, 2], NO_EPOCH_CUTOFF,
                                     display_name=plot_name,
                                     test_csv_name=f"{shared_test_dirname()}/{BOOTSTRAP_CSV}",
                                     error_column="Test_Loss_Bootstrap_Std", combine_errors=True, task=task)
            if row is not None:
                rows.append({**row, "folder": folder, "size": size,
                             "r2": 1.0 - row["Mean Test MSE"] / var, "total": row["Std Test MSE"] / var})
    selected, val_rows = select_scratch_lr(rows)
    best = {size: max(r["r2"] for r in rows if r["size"] == size) for size in _TABLE_SIZES}
    cell = {(r["folder"], r["size"]): f"{r['r2']:.3f} ± {r['total']:.3f}"
            + ("*" if r["r2"] == best[r["size"]] else "")
            + ("†" if selected.get(r["Dataset"]) == r["Model_Name"] else "")
            for r in rows}
    # The val-selected row repeats the chosen lr's numbers (copied rows keep their size/r2/total).
    cell.update({(_VAL_SELECTED_ROW, r["size"]): f"{r['r2']:.3f} ± {r['total']:.3f}"
                 + ("*" if r["r2"] == best[r["size"]] else "")
                 for r in val_rows})

    width = max(len(name) for _, name in _TABLE_MODELS)
    print(f"task: {task} | R2 ± total on {shared_test_dirname()} | "
          f"* best mean per column, † scratch lr with the lowest mean best-val loss")
    print(f"{'model':<{width}}" + "".join(f"  {'N = ' + str(int(size * 0.6)):<15}" for size in _TABLE_SIZES))
    for folder, name in _TABLE_MODELS:
        print(f"{name:<{width}}" + "".join(f"  {cell.get((folder, size), '--'):<15}" for size in _TABLE_SIZES))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="*", metavar="SIZE:MODEL[:RUNS]")
    parser.add_argument("--table", action="store_true",
                        help="Print the appendix table (all models x all sizes); takes no SIZE:MODEL")
    parser.add_argument("--target_columns", default="3d_x_y_z")
    parser.add_argument("--task", choices=list(TASKS), default=DEFAULT_TASK)
    parser.add_argument("--compact", action="store_true", help="One summary line per model")
    parser.add_argument("--sort_r2", action="store_true",
                        help="With --compact: order each N's lines by mean R2, highest first")
    args = parser.parse_args()
    if args.table and (args.specs or args.compact or args.sort_r2):
        parser.error("--table prints the whole table: it takes no SIZE:MODEL, --compact or --sort_r2")
    if not args.table and not args.specs:
        parser.error("give SIZE:MODEL[:RUNS] specs, or --table")
    if args.sort_r2 and not args.compact:
        parser.error("--sort_r2 needs --compact")

    labels_npy = REPO_ROOT / "evaluation" / f"{args.task}_labels_{EVAL_DATASET_SIZE}_test.npy"
    var = compute_test_label_variance_from_labels(str(labels_npy))
    if args.table:
        print_table(args.target_columns, var, args.task)
        return
    specs = [parse_spec(s) for s in args.specs]
    if args.compact:
        width = max(len(display_name(model)) for _, model, _ in specs)
        rows = [(size, *report(args.target_columns, size, model, runs, var, args.task, compact_width=width))
                for size, model, runs in specs]
        print(f"task: {args.task} | input: {_TASK_INPUT} | "
              f"target: {TASKS[args.task]['label_key']} = {_TASK_TARGETS[args.task]}")
        print(f"[{TASKS[args.task]['dir_prefix']}{args.target_columns}_<SIZE>, {shared_test_dirname()}]")
        print(f"{'N':>5}  {'model':<{width}}  {'R2':>6} ± total (bootstrap)  {'min':>6}  {'max':>6}  n")
        # One block per run of consecutive specs with the same N, separated by a blank line.
        for i, (_, block) in enumerate(itertools.groupby(rows, key=lambda row: row[0])):
            block = list(block)
            if args.sort_r2:
                block.sort(key=lambda row: row[1], reverse=True)
            if i:
                print()
            for _, _, line in block:
                print(line)
        return
    for i, (size, model, runs) in enumerate(specs):
        if i:
            print()
        report(args.target_columns, size, model, runs, var, args.task)


if __name__ == "__main__":
    main()
