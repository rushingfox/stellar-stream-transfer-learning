"""Prints the shared-test R^2 of each run, plus an "R2 ± total (bootstrap)" summary per model, the same numbers as the appendix table for the same runs.

    python src/print_r2.py SIZE:MODEL[:RUNS] [SIZE:MODEL[:RUNS] ...]

    python src/print_r2.py \\
        300:omnilearned_scratch \\
        300:omnilearned_proxy_gaussian_ball_finetune:0,1,2 \\
        300:eigenvalue_regression:0

MODEL is the on-disk folder name; RUNS defaults to the figure's runs at that SIZE (_FIGURE_RUNS: 0-9 at 40 and 100, 0-2 at 300 and 1000; 0 for eigenvalue_regression and average).
Each run needs its shared-test evaluation and bootstrap; a missing file is an error, not a skipped run.
--task picks the runs (prog_mass_ or subhalo_<target_columns>_<size>/) and that task's evaluation/<task>_labels_10000_test.npy.
--compact prints only the summary, one line per model (N = training streams, 3/5 of SIZE; readable model names), with the same numbers.
--sort_r2 then orders each N's lines by mean R2, highest first, instead of the command order.

    python src/print_r2.py --table

prints the appendix table in plain text instead, and takes no SIZE:MODEL: every model of the paper's figure and table (rows) at every size (columns), "R2 ± total", with * on the best mean of each column and † on the lr chosen by validation loss in each lr sweep (OmniLearned scratch and DeepSets).
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
    _SINGLE_RUN_MODELS,
    LR_SWEEPS,
    VAL_SELECTED_SUFFIX,
    compute_test_label_variance_from_labels,
    load_one_run,
    report_sweep_selection,
    requested_sweeps,
    select_sweep_lr,
    summarize_model,
    sweep_base,
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

# Model names for --compact: the figure's names for the paper's models, in plain text, with the folder's _lr<LR> / _ep<N> suffix spelled out.
# A folder not matched here prints as is.
_BASE_NAMES = {
    "average": "Mean Baseline",
    "eigenvalue_regression": "Eigenvalue Regression",
    "deepsets": "DeepSets",
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
# The scratch runner's default epochs, spelled out in its --compact name (as each lr sweep's default lr is, from build_results_table.LR_SWEEPS) so it compares with the _ep<N> variants.
_SCRATCH_DEFAULT_EPOCHS = "100"

# The runs (seeds) of the figure and table at each size; must match `indices` in evaluation/reproduce_main_figure.ipynb.
_FIGURE_RUNS = {40: range(10), 100: range(10), 300: range(3), 1000: range(3)}
# --table: the columns and rows of the appendix table (evaluation/reproduce_main_figure.ipynb), the rows as (folder, name) in table order.
_TABLE_SIZES = tuple(_FIGURE_RUNS)
def _sweep_rows(base):
    """Gives the --table rows of one lr sweep.

    Returns [(folder, name), ...]: first the val-selected row (no folder of its own; per size, the lr with the lowest mean best-val loss), then one row per lr; folders are the swept model's name in lowercase, which sweep_base matches too.

    `base`: an LR_SWEEPS key, e.g. 'DeepSets'.
    """
    sweep, folder = LR_SWEEPS[base], base.lower()
    return ([(folder + VAL_SELECTED_SUFFIX.lower(), sweep["table_name"])]
            + [(folder if lr == sweep["default_lr"] else f"{folder}_lr{lr}", sweep["label"].format(f"lr {lr}"))
               for lr in sweep["lrs"]])


_TABLE_MODELS = (
    [("eigenvalue_regression", "Eigenvalue Regression")]
    + _sweep_rows("DeepSets")
    + _sweep_rows("OmniLearned_Scratch")
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
_VAL_SELECTED_FOLDERS = {base.lower() + VAL_SELECTED_SUFFIX.lower() for base in LR_SWEEPS}


def display_name(folder):
    """Gives the readable --compact name of a model folder.

    Returns e.g. 'Cube' for omnilearned_proxy_cube_finetune and 'OmniLearned (Scratch), lr 1e-4, 50 ep' for omnilearned_scratch_lr1e-4_ep50; a folder it does not know, as is.

    `folder`: the on-disk model folder name.
    """
    base, lr, epochs = re.fullmatch(r"(.+?)(?:_lr([0-9.e-]+))?(?:_ep(\d+))?", folder).groups()
    proxy = re.fullmatch(r"omnilearned_proxy_(.+?)_finetune", base)
    if proxy and proxy.group(1) in _PROXY_NAMES:
        name = _PROXY_NAMES[proxy.group(1)]
    elif base in _BASE_NAMES:
        name = _BASE_NAMES[base]
    else:
        return folder
    if sweep_base(base) is not None:
        lr = lr or LR_SWEEPS[sweep_base(base)]["default_lr"]
    if base == "omnilearned_scratch":
        epochs = epochs or _SCRATCH_DEFAULT_EPOCHS
    details = ([f"lr {lr}"] if lr else []) + ([f"{epochs} ep"] if epochs else [])
    return ", ".join([name] + details)


def figure_runs(model, size):
    """Gives the runs the figure and table use for one model at one size.

    Returns a list of run indices, e.g. [0, ..., 9] for omnilearned_scratch at 40 and [0] for eigenvalue_regression; exits if the size has no entry in _FIGURE_RUNS.

    `model`: the on-disk model folder name.
    `size`: the dataset size.
    """
    if model in _SINGLE_RUN_MODELS:
        return [0]
    if size not in _FIGURE_RUNS:
        raise SystemExit(f"No default RUNS for size {size} (_FIGURE_RUNS has {', '.join(map(str, _FIGURE_RUNS))}); "
                         f"give them as SIZE:MODEL:RUNS")
    return list(_FIGURE_RUNS[size])


def parse_spec(spec):
    """Parses one SIZE:MODEL[:RUNS] argument.

    Returns (size, model folder, run indices), the figure's runs at that size (figure_runs) if not given; exits with a message on a malformed spec.

    `spec`: e.g. '300:omnilearned_scratch:0,1,2'.
    """
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not all(parts[:2]):
        raise SystemExit(f"Bad spec '{spec}': expected SIZE:MODEL[:RUNS], e.g. 300:omnilearned_scratch:0,1,2")
    size = validate_dataset_size(parts[0])
    if len(parts) == 2:
        runs = figure_runs(parts[1], size)
    else:
        try:
            runs = [int(r) for r in parts[2].split(",")]
        except ValueError:
            raise SystemExit(f"Bad RUNS in '{spec}': expected comma-separated integers, e.g. 3,4,5")
    return size, parts[1], runs


def best_epoch(run_files):
    """Finds a run's best-val epoch: the one row of its shared-test loss_history.csv with a Test_Loss, which is filled only on the best-val row.

    Returns (best-val epoch, total epochs); None if the run has no training curve (one epoch, as for eigenvalue_regression); exits if there is not exactly one such row.

    `run_files`: the run's files, from required_files.
    """
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
    """Lists the files print_r2 reads for one run.

    Returns {'train': results/loss_history.csv, 'shared': shared_test<N>/loss_history.csv, 'bootstrap': shared_test<N>/bootstrap.csv}, as paths in the run dir.

    `target_columns`, `size`, `model`, `idx`, `task`: pick the run dir.
    """
    shared_dir = get_shared_test_dir(REPO_ROOT, target_columns, size, model, idx, task=task)
    return {"train": get_loss_history_csv_path(REPO_ROOT, target_columns, size, model, idx, task=task),
            "shared": shared_dir / SHARED_TEST_CSV, "bootstrap": shared_dir / BOOTSTRAP_CSV}


def report(target_columns, size, model, runs, var, task, compact_width=None):
    """Prints the R^2 of each run of one model at one size, then its "R2 ± total (bootstrap)" summary from the same summarize_model call as the appendix table.

    Returns nothing; with compact_width it prints nothing and returns (mean R2, summary line) instead; exits if the model folder or a run's file is missing.

    `target_columns`, `size`, `model`, `task`: pick the model folder.
    `runs`: the run indices.
    `var`: Var(y_test), to turn MSE into R^2.
    `compact_width`: for --compact, the width the model name is padded to; None prints the full report.
    """
    model_dir = get_model_dir(REPO_ROOT, target_columns, size, model, task)
    if not model_dir.is_dir():
        raise SystemExit(f"Model folder not found: {model_dir}")

    compact = compact_width is not None
    files = {idx: required_files(target_columns, size, model, idx, task) for idx in runs}
    # The training curve only supplies the best-val epoch, which compact mode does not print, so compact mode does not require it (the subhalo non-ML baselines have none).
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

    # Same call as the appendix table: total is the standard error of the mean over runs and the mean bootstrap std added in quadrature.
    _, row = summarize_model(REPO_ROOT, target_columns, size, model, runs, NO_EPOCH_CUTOFF,
                             test_csv_name=f"{shared_test_dirname()}/{BOOTSTRAP_CSV}",
                             error_column="Test_Loss_Bootstrap_Std", combine_errors=True, task=task)
    mean_r2 = 1.0 - row["Mean Test MSE"] / var
    total = row["Std Test MSE"] / var
    # A single run has no cross-run term (and no Stat Std column), so its bootstrap std is the total.
    boot_mean = row.get("Stat Std Test MSE", row["Std Test MSE"]) / var
    if compact:
        return mean_r2, (f"{int(size * 0.6):>5}  {display_name(model):<{compact_width}}  "
                         f"{mean_r2:6.3f} ± {total:.3f} ({boot_mean:.3f})  "
                         f"{min(r2_values):6.3f}  {max(r2_values):6.3f}  {len(runs)}")
    print(f"  R2 = {mean_r2:.3f} ± {total:.3f} ({boot_mean:.3f})   "
          f"min {min(r2_values):.3f}   max {max(r2_values):.3f}   (n={len(runs)})")


def print_table(target_columns, var, task):
    """Prints the appendix table in plain text: the same numbers, with * for the bold best mean and † for the dagger.

    Returns nothing.

    `target_columns`: the experiment folders' input columns, e.g. 3d_x_y_z.
    `var`: Var(y_test), to turn MSE into R^2.
    `task`: 'prog_mass' or 'subhalo'.
    """
    model_list = [(target_columns, size, folder, folder) for size in _TABLE_SIZES
                  for folder, _ in _TABLE_MODELS if folder not in _VAL_SELECTED_FOLDERS]
    rows = []  # plot rows of summarize_model, for select_sweep_lr
    for _, size, folder, _ in model_list:
        _, row = summarize_model(REPO_ROOT, target_columns, size, folder, figure_runs(folder, size), NO_EPOCH_CUTOFF,
                                 display_name=folder,
                                 test_csv_name=f"{shared_test_dirname()}/{BOOTSTRAP_CSV}",
                                 error_column="Test_Loss_Bootstrap_Std", combine_errors=True, task=task)
        if row is not None:
            rows.append({**row, "folder": folder, "size": size,
                         "r2": 1.0 - row["Mean Test MSE"] / var, "total": row["Std Test MSE"] / var})
    n_required = {dataset_key(target_columns, size, task): len(_FIGURE_RUNS[size]) for size in _TABLE_SIZES}
    selected, val_rows, _, status = select_sweep_lr(rows, n_required, requested_sweeps(model_list, task))
    report_sweep_selection(status, problems_only=True)
    best = {size: max(r["r2"] for r in rows if r["size"] == size) for size in _TABLE_SIZES}
    cell = {(r["folder"], r["size"]): f"{r['r2']:.3f} ± {r['total']:.3f}"
            + ("*" if r["r2"] == best[r["size"]] else "")
            + ("†" if selected.get((r["Dataset"], sweep_base(r["Model_Name"]))) == r["Model_Name"] else "")
            for r in rows}
    # A val-selected row repeats the chosen lr's numbers (copied rows keep their size/r2/total).
    cell.update({(r["Model_Name"].lower(), r["size"]): f"{r['r2']:.3f} ± {r['total']:.3f}"
                 + ("*" if r["r2"] == best[r["size"]] else "")
                 for r in val_rows})

    width = max(len(name) for _, name in _TABLE_MODELS)
    print(f"task: {task} | R2 ± total on {shared_test_dirname()} | "
          f"* best mean per column, † the lr with the lowest mean best-val loss in each lr sweep")
    print("seeds: " + " | ".join(f"N = {int(size * 0.6)}: {min(runs)}-{max(runs)}" for size, runs in _FIGURE_RUNS.items())
          + " (eigenvalue_regression: run 0 only)")
    print(f"{'model':<{width}}" + "".join(f"  {'N = ' + str(int(size * 0.6)):<15}" for size in _TABLE_SIZES))
    for folder, name in _TABLE_MODELS:
        print(f"{name:<{width}}" + "".join(f"  {cell.get((folder, size), '--'):<15}" for size in _TABLE_SIZES))


def main():
    """Parses the command line and prints the per-run report, the --compact summary or the --table.

    Returns nothing.
    """
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
