import os
import re
import argparse
from collections import defaultdict

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from exp_paths import DEFAULT_TASK, TASKS, dataset_key, get_run_dir, validate_dataset_size

sns.set_theme(style="whitegrid")

# Paper style: LaTeX-rendered text, sans-serif Helvetica (matches
# BlueTilt-LensingPerturbers/src/btlp/style.py). Requires a working LaTeX on
# PATH (e.g. `module load texlive/2024` loaded AFTER activating the conda
# env, since conda activation otherwise takes PATH priority).
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
})


# Human-readable labels for plot legends/axes only — the underlying CSV
# "Model" column (and everything used for sorting/grouping) keeps the raw
# display_name strings, so the CSV stays parseable by exact name. Unmapped
# names fall back to "underscores -> spaces".
_PRETTY_MODEL_NAMES = {
    'OmniCosmos_CAMELS':     'OmniCosmos-C',
    'OmniCosmos_Quijote':    'OmniCosmos-Q',
    'DeepSets_256':          'DeepSets',
    'average':               'Mean Baseline',
    'eigenvalue_regression': 'Eigenvalue Regression',
    # OmniLearned_Scratch and its lr sweep are named in _pretty_model_name.
    'OmniLearned_Finetune':  'OmniLearned (HEP Jets)',
    'proxy_cube':            'Cube',
    'proxy_bounded_ball':    'Bounded Ball',
    'proxy_gaussian_ball':   'Gaussian Ball',
    'proxy_gaussian_ellipsoid_sigmax': 'Gaussian Ellipsoid',
    'proxy_bounded_ellipsoid_a':       'Bounded Ellipsoid',
    'proxy_box_a':           'Box',
    'proxy_stream_eigenvalues_lambda1': 'Real-Stream PCA',
    # Optional models, not in the paper's default list.
    'DeepSets_1024':         'DeepSets (1024)',
    'OmniLearned_Scratch_M': 'OmniLearned-m (Scratch)',
    'OmniLearned_Finetune_M': 'OmniLearned-m (HEP Jets)',
    'proxy_stream_eigenvalues_lambda2': r'Real-Stream PCA ($\lambda_2$)',
    'proxy_stream_eigenvalues_lambda3': r'Real-Stream PCA ($\lambda_3$)',
}


def _pretty_model_name(name):
    if name == SCRATCH_MODEL:
        return rf"OmniLearned (Scratch, lr $= {_latex_lr(DEFAULT_SCRATCH_LR)}$)"
    if name == VAL_SELECTED_SCRATCH_MODEL:
        return "OmniLearned (Scratch, val-selected lr)"
    if name.startswith(SCRATCH_LR_PREFIX):
        return rf"OmniLearned (Scratch, lr $= {_latex_lr(_scratch_lr(name))}$)"
    return _PRETTY_MODEL_NAMES.get(name, name.replace('_', ' '))


# Scratch learning-rate sweep. The default-lr run is OmniLearned_Scratch (the
# runner's LR default); each other lr is its own model, OmniLearned_Scratch_lr<lr>.
# The figure shows only one scratch point per size, for the lr with the lowest
# mean best-val loss (VAL_SELECTED_SCRATCH_MODEL); the table has that point as
# its own row, then every lr with the selected one marked.
SCRATCH_MODEL = 'OmniLearned_Scratch'
SCRATCH_LR_PREFIX = 'OmniLearned_Scratch_lr'
VAL_SELECTED_SCRATCH_MODEL = 'OmniLearned_Scratch_ValLR'
DEFAULT_SCRATCH_LR = '5e-5'  # LR default of scripts/omnilearned/run_omnilearned_scratch.sh


def _is_scratch_sweep_member(name):
    return name == SCRATCH_MODEL or name.startswith(SCRATCH_LR_PREFIX)


def _scratch_lr(name):
    """'OmniLearned_Scratch' -> '5e-5'; 'OmniLearned_Scratch_lr2e-4' -> '2e-4'."""
    return DEFAULT_SCRATCH_LR if name == SCRATCH_MODEL else name[len(SCRATCH_LR_PREFIX):]


def _latex_lr(lr):
    """'2e-4' -> '2\\times10^{-4}' (math mode, without the dollar signs)."""
    mantissa, exponent = lr.split('e')
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


# Pretrain train-split sizes, for generate_pretrain_size_scatter.
# Baselines have no pretraining and are intentionally absent.
_PRETRAIN_SIZE = {
    'OmniLearned_Finetune':  1_000_000_000,  # ~1.06B HEP jets (arXiv:2510.24066 Table I)
    'OmniCosmos_CAMELS':     600,             # CAMELS-SAM full train split
    # Checkpoint is named "..._s_10000", but OmniCosmos's own Fig. 4/5 only
    # scan 3 points (~10^2, ~10^3, and the full official split), with the
    # third sitting visibly right of the 10^4 gridline — so "s_10000" is
    # likely a rounded label for the full split, not a literal 10k subsample.
    # Using the full official train count (19651) until confirmed with Vini.
    'OmniCosmos_Quijote':    19_651,
    # Proxies: dataset_size=10000 is the bucket's TOTAL (train+val+test); the
    # actual pretrain train split is 60% of that (prepare_data.py's 0.6/0.8
    # split), i.e. 6000 — matching how OmniLearned/OmniCosmos report training-
    # set size above (train-only, not train+val+test).
    'proxy_cube':                       6_000,
    'proxy_bounded_ball':               6_000,
    'proxy_gaussian_ball':              6_000,
    'proxy_gaussian_ellipsoid_sigmax':  6_000,
    'proxy_bounded_ellipsoid_a':        6_000,
    'proxy_box_a':                      6_000,
    'proxy_stream_eigenvalues_lambda1': 6_000,
}


def _pretty_count_label(n):
    """600 -> '600'; 10000 -> '10,000'; 1_000_000_000 -> '~1B'."""
    if n >= 1_000_000_000:
        return f"~{n / 1_000_000_000:.3g}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3g}M"
    return f"{n:,}"


def _pretty_dataset_label(ds_key):
    """'prog_mass_3d_x_y_z_10000' -> '6,000'. Folder-name sizes are the total pool
    (train+val+test); display convention is the train-split count (60%)."""
    m = re.search(r'(\d+)$', ds_key)
    if not m:
        return ds_key
    train_n = int(m.group(1)) * 3 // 5
    return f"{train_n:,}"


def _pretty_combined_label(model_str):
    """'prog_mass_3d_x_y_z_10000+proxy_cube' -> 'N = 10,000 · Proxy: Cube'."""
    if '+' not in model_str:
        return model_str
    ds_part, name_part = model_str.split('+', 1)
    return f"{_pretty_dataset_label(ds_part)} · {_pretty_model_name(name_part)}"


# Four groups (marker/colors set below in _CATEGORY_STYLE):
#   baseline  — no pretraining
#   physics   — real-domain pretraining (HEP jets, cosmological sims)
#   synthetic — geometric proxy pretraining
#   eigen     — the PCA eigenvalue proxy: real stream data with a
#               swapped label. Neither baseline nor a normal proxy,
#               so it gets its own singleton group.
# Colors are hand-picked swatches, not a sequential colormap.
# Per advisor feedback, a gradient reads as "shades of one thing".
_CATEGORY_ORDER = ['baseline', 'physics', 'synthetic', 'eigen']
_CATEGORY_STYLE = {
    'baseline':  {'marker': 'o', 'colors': ['#999999', '#a6763d', '#5b7c8d', '#1a1a1a']},
    'physics':   {'marker': 's', 'colors': ['#e6550d', '#d62728', '#d4a017']},
    'synthetic': {'marker': '^', 'colors': ['#1f77b4', '#17becf', '#2ca02c',
                                            '#9467bd', '#08306b', '#66c2a5']},
    'eigen':     {'marker': 'D', 'colors': ['#c51b8a']},
}

# Optional models (not in the paper's default list) get a fixed color of their
# own, a darker or lighter shade of their base model's, instead of a slot in
# _CATEGORY_STYLE. The paper's models keep handing out the category swatches
# among themselves only, so adding an optional model never recolors them, and
# a figure without optional models is unchanged.
_OPTIONAL_MODEL_COLORS = {
    'deepsets_1024':                    '#7f3b08',  # DeepSets 256 '#a6763d'
    'omnilearned_scratch_m':            '#2c4a5a',  # OmniLearned Scratch '#5b7c8d'
    # Scratch lr sweep: the per-lr rows only appear in the table, so their
    # color is never drawn but keeps them out of the category swatches. The
    # val-selected point replaces OmniLearned Scratch in the figure and takes
    # its swatch.
    'omnilearned_scratch_lr':           '#5b7c8d',
    'omnilearned_finetune_m':           '#a63603',  # OmniLearned Finetune '#e6550d'
    'proxy_stream_eigenvalues_lambda2': '#f768a1',  # Real-Stream PCA '#c51b8a'
    'proxy_stream_eigenvalues_lambda3': '#7a0177',
}

# Baseline ordering (left-to-right / legend order within the category).
_BASELINE_RANK = {
    'average': 0,
    'eigenvalue_regression': 1,
    'deepsets': 2,                 # hidden 1024, the default DeepSets
    'deepsets_256': 2,
    'baseline_256': 2,
    'deepsets_1024': 2.5,          # optional, next to DeepSets 256
    'omnilearned_scratch': 3,
    'omnilearned_scratch_vallr': 2.9,   # first scratch row of the table; the figure's only scratch point
    'omnilearned_scratch_lr1e-4': 3.1,  # table only, in lr order
    'omnilearned_scratch_lr2e-4': 3.2,
    'omnilearned_scratch_lr5e-4': 3.3,
    'omnilearned_scratch_lr1e-3': 3.4,
    'omnilearned_scratch_m': 3.5,  # optional, next to OmniLearned Scratch
}

# Physics ordering: most-distant domain first.
_PHYSICS_RANK = {
    'omnilearned_finetune': 0,   # HEP jets — most distant domain
    'omnilearned_jet_finetune': 0,  # the same model by its folder name
    'omnilearned_finetune_m': 0.5,  # optional, next to OmniLearned Finetune
    'omnicosmos_camels':    1,   # cosmological, smaller-scale (CAMELS)
    'omnicosmos_quijote':   2,   # cosmological, larger-scale (Quijote)
}

# Synthetic ordering: by (rough) information content.
_SYNTHETIC_RANK = {
    'proxy_cube':               0,
    'proxy_bounded_ball':       1,
    'proxy_gaussian_ball':      2,
    'proxy_box_a':              3,
    'proxy_bounded_ellipsoid_a': 4,
    'proxy_gaussian_ellipsoid_sigmax': 5,
}

_RANK_TABLES = {
    'baseline': _BASELINE_RANK,
    'physics': _PHYSICS_RANK,
    'synthetic': _SYNTHETIC_RANK,
    'eigen': {'proxy_stream_eigenvalues_lambda1': 0,
              'proxy_stream_eigenvalues_lambda2': 1,
              'proxy_stream_eigenvalues_lambda3': 2},
}


def _rank(name, rank_table):
    n = name.lower()
    for key in sorted(rank_table, key=len, reverse=True):
        if key in n:
            return rank_table[key]
    return 99


def categorize_model(name):
    """Four categories — see _CATEGORY_ORDER comment above."""
    n = name.lower()
    if 'stream_eigenvalues' in n or 'eigen_lambda1' in n:
        return 'eigen'
    if 'omnilearned_finetune' in n or 'jet_finetune' in n or 'omnicosmos' in n:
        return 'physics'
    if ('scratch' in n or 'deepsets' in n or 'average' in n
            or n in ('baseline', 'baseline_256', 'eigenvalue_regression')):
        return 'baseline'
    return 'synthetic'


def build_model_styles(models):
    """Return [(model_name, marker, color), ...] grouped by category
    (marker encodes the category), with a distinct hand-picked color per member
    within the category (color encodes which specific model it is)."""
    by_category = defaultdict(list)
    for m in models:
        by_category[categorize_model(m)].append(m)

    out = []
    for cat in _CATEGORY_ORDER:
        members = by_category.get(cat, [])
        if not members:
            continue
        members = sorted(members, key=lambda m: _rank(m, _RANK_TABLES[cat]))
        style = _CATEGORY_STYLE[cat]
        colors = style['colors']
        # Category swatches go to the non-optional members only, in rank order.
        regular = [m for m in members if _optional_color(m) is None]
        if len(regular) > len(colors):
            raise ValueError(
                f"Category '{cat}' has {len(regular)} members but "
                f"only {len(colors)} colors defined in _CATEGORY_STYLE — add more."
            )
        color_of = dict(zip(regular, colors))
        for member in members:
            color = color_of.get(member) or _optional_color(member)
            out.append((member, style['marker'], color))
    return out


def _optional_color(name):
    n = name.lower()
    for key in sorted(_OPTIONAL_MODEL_COLORS, key=len, reverse=True):
        if key in n:
            return _OPTIONAL_MODEL_COLORS[key]
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Build results table and bar chart (best val MSE + test MSE at best val).")
    parser.add_argument('--indices', nargs='+', type=int, default=[0],
                        help="Index list, e.g. --indices 0 1 2")
    parser.add_argument('--output', type=str, default="./results.csv",
                        help="Output CSV path")
    parser.add_argument('--plot', type=str, default=None,
                        help="Output bar chart path (e.g., .png, .pdf). Omit to skip the "
                             "bar chart entirely (matches --scatter_plot/--pretrain_scatter_plot).")
    parser.add_argument('--scatter_plot', type=str, default=None,
                        help="Output scatter plot path grouped by dataset (e.g., ./results_scatter.png)")
    parser.add_argument('--pretrain_scatter_plot', type=str, default=None,
                        help="Output scatter plot path, x-axis = pretrain dataset size "
                             "instead of finetune dataset size (e.g., ./results_pretrain_scatter.png)")
    parser.add_argument('--latex_table', type=str, default=None,
                        help="Output path for a bare LaTeX `tabular` of R^2 per model "
                             "(rows) x fine-tuning dataset size (columns), for the paper "
                             "appendix. Requires --metric r2.")
    parser.add_argument('--sort_by', type=str, default="test", choices=["val", "test"],
                        help="Sort table by mean val MSE or mean test MSE (lower is better)")
    parser.add_argument('--yscale', type=str, default="log", choices=["linear", "log"],
                        help="Y-axis scale for all plots (default: log). Ignored (forced "
                             "linear) when --metric r2.")
    parser.add_argument('--metric', type=str, default="mse", choices=["mse", "r2"],
                        help="Scatter-plot y-axis metric (default: mse). r2 converts "
                             "Test MSE to R^2 = 1 - MSE / Var(y_test) using "
                             "--r2_reference_raw; only affects the scatter plot, not the "
                             "CSV table.")
    parser.add_argument('--r2_reference_raw', type=str, default=None,
                        help="Path to a raw prog_mass_reg_dataset_<N>.h5 file whose test "
                             "split's label variance is the R^2 denominator. One of this or "
                             "--r2_reference_labels is required with --metric r2.")
    parser.add_argument('--r2_reference_labels', type=str, default=None,
                        help="Path to a .npy sidecar of just the test-split labels (see "
                             "src/extract_r2_reference_labels.py), as a lightweight "
                             "alternative to --r2_reference_raw that doesn't need the full "
                             "raw dataset on disk -- used by the reproduce_main_figure "
                             "notebook. Takes priority over --r2_reference_raw if both are set.")
    parser.add_argument('--r2_ymin', type=float, default=-0.3,
                        help="Scatter plot y-axis lower bound with --metric r2 (default: -0.3). "
                             "Ignored with --relative_to_scratch (range is auto).")
    parser.add_argument('--relative_to_scratch', action='store_true',
                        help="Scatter plot shows (model - scratch) / scratch per dataset "
                             "size, for whichever --metric is selected, instead of the raw "
                             "value. Forces linear y-axis (values can be negative).")
    parser.add_argument('--relative_baseline_model', type=str, default="OmniLearned_Scratch",
                        help="Model_Name to use as the scratch reference for "
                             "--relative_to_scratch (default: OmniLearned_Scratch).")
    parser.add_argument('--relative_ymin', type=float, default=-1.0,
                        help="Scatter plot y-axis lower bound with --relative_to_scratch "
                             "(default: -1.0). Not a hard bound on the data -- override to "
                             "see values beyond it.")
    parser.add_argument('--relative_ymax', type=float, default=1.0,
                        help="Scatter plot y-axis upper bound with --relative_to_scratch "
                             "(default: 1.0). Not a hard bound on the data -- override to "
                             "see values beyond it.")
    parser.add_argument('--scatter_ylabel', type=str, default=None,
                        help="Override the scatter plot's y-axis label (default: "
                             "auto-derived from --metric/--relative_to_scratch). Use when "
                             "--test_csv_name points at a value that isn't plain Test MSE/R^2, "
                             "e.g. a matched/paired comparison already computed upstream.")
    parser.add_argument('--scatter_ymin', type=float, default=None,
                        help="Explicit scatter plot y-axis lower bound, overrides any "
                             "--metric/--relative_to_scratch default. Not a hard data bound.")
    parser.add_argument('--scatter_ymax', type=float, default=None,
                        help="Explicit scatter plot y-axis upper bound, overrides any "
                             "--metric/--relative_to_scratch default. Not a hard data bound.")
    parser.add_argument('--value_column', type=str, default="Test_Loss",
                        help="Which column of --test_csv_name to report as the test value "
                             "(default: Test_Loss). Model selection still uses Val_Loss.")
    parser.add_argument('--error_column', type=str, default=None,
                        help="If set, the error bar comes from this precomputed column "
                             "(averaged across --indices, typically just one run) instead of "
                             "the usual cross-seed std, e.g. Test_Loss_Bootstrap_Std from "
                             "compute_bootstrap_shared_test.py.")
    parser.add_argument('--connect_lines', action='store_true',
                        help="Draw a line connecting each model's own points across dataset "
                             "sizes in the scatter plot (default: off, markers only).")
    parser.add_argument('--combine_errors', action='store_true',
                        help="Requires --error_column and >=2 --indices. Combines the "
                             "cross-seed std (population std across the point estimates, "
                             "'systematic') with the mean of --error_column across the same "
                             "runs ('statistical', e.g. mean bootstrap std) in quadrature: "
                             "total = sqrt(seed_std**2 + mean_stat**2). The scatter plot then "
                             "draws two nested error bars per point -- inner: mean_stat only, "
                             "outer: total. Neither term is divided by sqrt(n_indices): both "
                             "describe single-realization noise, not the mean's standard error.")
    parser.add_argument('--test_csv_name', type=str, default="results/loss_history.csv",
                        help="Per-run CSV to read, relative to each run_<idx>/ dir. The "
                             "default is the run's own split; the shared test set's are "
                             "shared_test10000/loss_history.csv and shared_test10000/bootstrap.csv.")
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help=(
            "List of models in format target_columns:dataset_size:target_model"
            "[:display_name]. Optional 4th field overrides the name shown in the "
            "CSV / plots without changing the on-disk folder name."
        )
    )
    parser.add_argument(
        '--repo_root',
        type=str,
        default="..",
        help="Repository root used to resolve experiment folders."
    )
    parser.add_argument('--task', type=str, default=DEFAULT_TASK, choices=sorted(TASKS),
                        help="Downstream task of every --models entry: its folders are "
                             "<task>_<target_columns>_<size>/ (default: prog_mass)")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Only consider the first N epochs when selecting best loss "
                             "(default 100, the training default; runs trained longer "
                             "need at least their epoch count, or their rows past it "
                             "are dropped)")
    return parser.parse_args()


def load_one_run(csv_path, epochs, value_column="Test_Loss", error_column=None):
    """Reads one CSV, finds best Val_Loss epoch, returns (best_val, value_at_best, error_at_best).

    value_column: which column to report as the second return value (default
    Test_Loss). Model selection always uses Val_Loss, regardless -- only
    which column gets reported at that row changes.

    error_column: optional column holding a precomputed error bar (e.g.
    Test_Loss_Bootstrap_Std from compute_bootstrap_shared_test.py), returned
    as the third value; None when not requested."""
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

    # only consider first N epochs
    if "Epoch" in df.columns:
        df_sub = df[df["Epoch"] <= epochs].copy()
    else:
        df_sub = df.head(epochs).copy()

    if df_sub.empty:
        return None

    # basic column check
    if "Val_Loss" not in df_sub.columns or value_column not in df_sub.columns:
        raise ValueError(f"Missing Val_Loss/{value_column} in {csv_path}. Columns: {list(df_sub.columns)}")
    if error_column is not None and error_column not in df_sub.columns:
        raise ValueError(f"Missing {error_column} in {csv_path}. Columns: {list(df_sub.columns)}")

    best_idx = df_sub["Val_Loss"].idxmin()
    best_val = float(df_sub.loc[best_idx, "Val_Loss"])
    test_at_best = float(df_sub.loc[best_idx, value_column])
    error_at_best = float(df_sub.loc[best_idx, error_column]) if error_column is not None else None
    return best_val, test_at_best, error_at_best


def summarize_model(repo_root, target_columns, dataset_size, target_model, indices, epochs, sort_by="test", display_name=None, test_csv_name="results/loss_history.csv", value_column="Test_Loss", error_column=None, combine_errors=False, task=DEFAULT_TASK):
    """Summarize statistics and return them for table and figure respectively.

    `target_model` is the on-disk folder name and is used to resolve paths.
    `display_name` (defaults to target_model) is what appears in CSV / plots.
    """
    val_list = []
    test_list = []
    error_list = []
    ds_key = dataset_key(target_columns, dataset_size, task)
    if display_name is None:
        display_name = target_model

    for idx in indices:
        path = get_run_dir(repo_root, target_columns, dataset_size, target_model, idx, task) / test_csv_name

        res = load_one_run(path, epochs, value_column=value_column, error_column=error_column)
        if res is None:
            continue
        val, test, error = res
        val_list.append(val)
        test_list.append(test)
        if error is not None:
            error_list.append(error)

    if len(val_list) == 0:
        return None, None

    model_name = f"{ds_key}+{display_name}"
    sort_value = float(np.mean(test_list) if sort_by == "test" else np.mean(val_list))

    # For showing table and ranking
    display_row = {
        "Model": model_name,
        "_sort": sort_value,
        "n_indices": len(val_list)
    }
    # For plotting
    plot_row = {
        "Model": model_name,
        "Dataset": ds_key,
        "Model_Name": display_name,
        "n_indices": len(val_list)
    }

    # Error bar from a precomputed column (e.g. bootstrap std from resampling
    # a single fixed run's test set) instead of cross-run std -- averaged
    # across however many runs were given (typically just one, e.g. run_0).
    if error_column is not None and error_list:
        mean_val = float(np.mean(val_list))
        mean_test = float(np.mean(test_list))
        stat_error = float(np.mean(error_list))

        if combine_errors and len(test_list) > 1:
            # stat ⊕ syst, nested-error-bar style: cross-seed std ("systematic",
            # already the repo's usual error bar) combined in quadrature with
            # the averaged per-run --error_column ("statistical", e.g. mean
            # bootstrap std). Both terms are population std of a single
            # realization, not a standard error of the mean -- neither is
            # divided by sqrt(n_indices).
            seed_std_val = float(np.std(val_list))
            seed_std_test = float(np.std(test_list))
            total_val = float(np.sqrt(seed_std_val ** 2 + stat_error ** 2))
            total_test = float(np.sqrt(seed_std_test ** 2 + stat_error ** 2))

            display_row["Best Val MSE"] = f"{mean_val:.4f} ± {total_val:.4f} (stat {stat_error:.4f})"
            display_row["Test MSE (at best val)"] = f"{mean_test:.4f} ± {total_test:.4f} (stat {stat_error:.4f})"

            plot_row["Mean Val MSE"] = mean_val
            plot_row["Std Val MSE"] = total_val
            plot_row["Stat Std Val MSE"] = stat_error
            plot_row["Mean Test MSE"] = mean_test
            plot_row["Std Test MSE"] = total_test
            plot_row["Stat Std Test MSE"] = stat_error
        else:
            display_row["Best Val MSE"] = f"{mean_val:.4f} ± {stat_error:.4f}"
            display_row["Test MSE (at best val)"] = f"{mean_test:.4f} ± {stat_error:.4f}"

            plot_row["Mean Val MSE"] = mean_val
            plot_row["Std Val MSE"] = stat_error
            plot_row["Mean Test MSE"] = mean_test
            plot_row["Std Test MSE"] = stat_error

    # 1 index：just value
    elif len(val_list) == 1:
        display_row["Best Val MSE"] = val_list[0]
        display_row["Test MSE (at best val)"] = test_list[0]
        
        plot_row["Mean Val MSE"] = val_list[0]
        plot_row["Std Val MSE"] = 0.0
        plot_row["Mean Test MSE"] = test_list[0]
        plot_row["Std Test MSE"] = 0.0

    # 2 indices：use [min, max]
    elif len(val_list) == 2:
        val_min, val_max = min(val_list), max(val_list)
        test_min, test_max = min(test_list), max(test_list)
        
        display_row["Best Val MSE"] = f"[{val_min:.4f}, {val_max:.4f}]"
        display_row["Test MSE (at best val)"] = f"[{test_min:.4f}, {test_max:.4f}]"
        
        plot_row["Mean Val MSE"] = np.mean(val_list)
        plot_row["Std Val MSE"] = (val_max - val_min) / 2
        plot_row["Mean Test MSE"] = np.mean(test_list)
        plot_row["Std Test MSE"] = (test_max - test_min) / 2

    # >=3 indices：mean ± std
    else:
        mean_val, std_val = np.mean(val_list), np.std(val_list)
        mean_test, std_test = np.mean(test_list), np.std(test_list)
        
        display_row["Best Val MSE"] = f"{mean_val:.4f} ± {std_val:.4f}"
        display_row["Test MSE (at best val)"] = f"{mean_test:.4f} ± {std_test:.4f}"
        
        plot_row["Mean Val MSE"] = mean_val
        plot_row["Std Val MSE"] = std_val
        plot_row["Mean Test MSE"] = mean_test
        plot_row["Std Test MSE"] = std_test

    return display_row, plot_row


def select_scratch_lr(plot_rows):
    """For each dataset size with a scratch lr sweep, pick the lr whose mean
    best-val loss across seeds is lowest (ties go to the smaller lr).

    Returns (selected, val_rows): selected maps Dataset -> the chosen sweep
    member's Model_Name, and val_rows holds one VAL_SELECTED_SCRATCH_MODEL plot
    row per size, a copy of the chosen member's row carrying its lr in
    'Selected_LR'. Sizes without any non-default lr get no selection."""
    by_dataset = defaultdict(list)
    for row in plot_rows:
        if _is_scratch_sweep_member(row["Model_Name"]):
            by_dataset[row["Dataset"]].append(row)

    selected, val_rows = {}, []
    for ds, rows in by_dataset.items():
        if not any(r["Model_Name"].startswith(SCRATCH_LR_PREFIX) for r in rows):
            continue
        best = min(rows, key=lambda r: (r["Mean Val MSE"], float(_scratch_lr(r["Model_Name"]))))
        selected[ds] = best["Model_Name"]
        val_rows.append({**best,
                         "Model": f"{ds}+{VAL_SELECTED_SCRATCH_MODEL}",
                         "Model_Name": VAL_SELECTED_SCRATCH_MODEL,
                         "Selected_LR": _scratch_lr(best["Model_Name"])})
    return selected, val_rows


def generate_bar_chart(df, save_plot_path, sort_metric, yscale="log"):
    if df.empty:
        return

    if sort_metric == "val":
         x_col = "Mean Val MSE"
         error_col = "Std Val MSE"
         title_suffix = "Best Val MSE"
    else:
         x_col = "Mean Test MSE"
         error_col = "Std Test MSE"
         title_suffix = "Test MSE (at best val)"

    plt.figure(figsize=(12, 6))
    
    positions = np.arange(len(df['Model']))
    means = df[x_col]
    errors = df[error_col]
    
    palette = sns.color_palette("viridis", len(df))
    
    bars = plt.bar(positions, means, yerr=errors, capsize=7, 
                   color=palette, edgecolor='black', alpha=0.9, label=title_suffix)
    
    plt.xticks(positions, [_pretty_combined_label(m) for m in df['Model']], rotation=35, ha="right")
    
    plt.xlabel("Model Configuration (Task+Architecture)", fontsize=12, fontweight='bold')
    plt.ylabel("Test MSE (at best val)", fontsize=12, fontweight='bold')
    plt.yscale(yscale)
    unique_indices = df['n_indices'].unique()
    indices_str = f"n_indices={unique_indices[0]}" if len(unique_indices) == 1 else "Variable n_indices"
    plt.title(f"Comparison of Model Performance\n"
              f"(Metric: {title_suffix}, {indices_str})", fontsize=14, fontweight='bold')

    for bar, mean, err in zip(bars, means, errors):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + err + (max(means)*0.01), 
                f"{mean:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

    plt.tight_layout()
    
    plt.savefig(save_plot_path, dpi=300)
    print(f"Saved bar chart -> {save_plot_path}")
    plt.close()


def compute_test_label_variance(raw_h5_path):
    """Var(y_test) of a raw prog_mass_reg_dataset_<N>.h5's test split (last 20%),
    used as the R^2 denominator: R^2 = 1 - MSE / Var(y_test)."""
    with h5py.File(raw_h5_path, 'r') as f:
        y = f['pid'][:].ravel()
    val_end = int(y.shape[0] * 0.8)
    return float(np.var(y[val_end:]))


def compute_test_label_variance_from_labels(labels_npy_path):
    """Same Var(y_test) as compute_test_label_variance(), but from the small
    sidecar .npy of already-sliced test-split labels (see
    src/extract_r2_reference_labels.py) instead of the full raw h5 -- lets
    R^2 be reproduced without the (large) raw dataset on disk."""
    return float(np.var(np.load(labels_npy_path).ravel()))


def generate_scatter_plot(df, save_plot_path, sort_metric, yscale="log", metric="mse", r2_reference_var=None, r2_ymin=-0.3, relative_to_scratch=False, relative_baseline_model="OmniLearned_Scratch", relative_ymin=-1.0, relative_ymax=1.0, scatter_ylabel=None, scatter_ymin=None, scatter_ymax=None, connect_lines=False):

    if df.empty:
        return

    if sort_metric == "val":
         y_col, error_col, title_suffix = "Mean Val MSE", "Std Val MSE", "Best Val MSE"
         stat_col = "Stat Std Val MSE"
    else:
         y_col, error_col, title_suffix = "Mean Test MSE", "Std Test MSE", "Test MSE (at best val)"
         stat_col = "Stat Std Test MSE"

    has_nested_error = stat_col in df.columns

    df = df.copy()
    if has_nested_error:
        # A model with a single run (eigenvalue_regression — a deterministic
        # linear fit, so there is nothing for a second seed to vary) has no
        # cross-seed term: summarize_model puts the bootstrap std straight into
        # error_col and never writes a stat_col value at all. Without this
        # fallback its shaded band would be a NaN rectangle, i.e. invisible,
        # and the figure would read as "this model has no bootstrap
        # uncertainty" when in fact its whole error bar IS the bootstrap. Band
        # and bar coincide exactly for such a model, which is the honest
        # picture.
        df[stat_col] = df[stat_col].fillna(df[error_col])
    metric_label = "MSE"
    if metric == "r2":
        # R^2 = 1 - MSE / Var(y_test) is affine in MSE, so mean/std transform
        # directly -- no need to redo the per-run_idx aggregation.
        df[y_col] = 1.0 - df[y_col] / r2_reference_var
        df[error_col] = df[error_col] / r2_reference_var
        if has_nested_error:
            df[stat_col] = df[stat_col] / r2_reference_var
        metric_label = "R²"
        yscale = "linear"  # R^2 can be negative / near 1 -- log scale doesn't apply

    y_label = f"Test {metric_label}"

    if relative_to_scratch:
        # (model - scratch) / scratch, per dataset size -- scratch's own mean
        # at that size is treated as a fixed reference (only the model's own
        # std is rescaled by it; scratch's run-to-run uncertainty is not
        # propagated into the ratio -- an approximation, not full error propagation).
        rel_groups = []
        for ds in df['Dataset'].unique():
            ds_df = df[df['Dataset'] == ds]
            baseline_rows = ds_df[ds_df['Model_Name'] == relative_baseline_model]
            if baseline_rows.empty:
                print(f"Warning: no '{relative_baseline_model}' row for {ds} -- "
                      f"dropping this dataset from the relative-to-scratch plot.")
                continue
            baseline_val = baseline_rows.iloc[0][y_col]
            if baseline_val == 0:
                print(f"Warning: '{relative_baseline_model}' value is 0 for {ds} -- "
                      f"dropping this dataset (relative difference undefined).")
                continue
            sub = ds_df.copy()
            sub[y_col] = (sub[y_col] - baseline_val) / baseline_val
            sub[error_col] = sub[error_col] / abs(baseline_val)
            if has_nested_error:
                sub[stat_col] = sub[stat_col] / abs(baseline_val)
            rel_groups.append(sub)
        df = pd.concat(rel_groups, ignore_index=True) if rel_groups else df.iloc[0:0]
        y_label = f"Relative {metric_label} Difference vs. {_pretty_model_name(relative_baseline_model)}"
        yscale = "linear"  # can be negative

    if scatter_ylabel is not None:
        y_label = scatter_ylabel

    if df.empty:
        return

    plt.figure(figsize=(12, 5.5))

    datasets = df['Dataset'].unique()
    models = df['Model_Name'].unique()

    x_mapping = {ds: i for i, ds in enumerate(datasets)}

    # Group by category (scratch / external pretrain / proxy). Same marker
    # within a category, color shade disambiguates members. Iterating in
    # category order also makes the dodge offset cluster spatially per family.
    ordered_styles = build_model_styles(models)
    n_models = len(ordered_styles)

    if n_models > 1:
        # Each dataset size owns 1.0 of x-space; the cluster uses 0.7 of it,
        # leaving a 0.15 gutter on each side before the region boundary.
        dodge_width = 0.7
        offsets = np.linspace(-dodge_width / 2, dodge_width / 2, n_models)
    else:
        offsets = [0]
    # Nested-error shaded band width: fixed, and now just under the
    # model-to-model spacing (dodge_width / (n_models - 1)) so neighbouring
    # bands sit side by side instead of overlapping. It was deliberately wider
    # than the spacing back when the cluster was squeezed into 0.3 of x-space
    # and a narrow band would have been invisible.
    band_width = 0.05

    # (x, y, error) of every drawn point, for the annotations added below.
    points = {}

    for m_idx, (model, marker, color) in enumerate(ordered_styles):
        model_data = df[df['Model_Name'] == model]
        if model_data.empty:
            continue

        x_vals, y_vals, y_errs, y_stats = [], [], [], []

        for _, row in model_data.iterrows():
            x_vals.append(x_mapping[row['Dataset']] + offsets[m_idx])
            y_vals.append(row[y_col])
            y_errs.append(row[error_col])
            if has_nested_error:
                y_stats.append(row[stat_col])
            points[(model, row['Dataset'])] = (x_vals[-1], y_vals[-1], y_errs[-1])

        if has_nested_error:
            # Statistical component (e.g. mean bootstrap std across seeds) as
            # a translucent shaded band instead of a second error bar -- reads
            # better than two overlapping stick-and-cap error bars, and a
            # short band still shows up poking out sideways from under the
            # marker even when its height alone would be invisible. Drawn
            # first / at a lower zorder so the marker+outer bar sit on top.
            ax = plt.gca()
            for xv, yv, sv in zip(x_vals, y_vals, y_stats):
                ax.add_patch(Rectangle(
                    (xv - band_width / 2, yv - sv),
                    band_width, 2 * sv,
                    facecolor=color,
                    edgecolor='none',
                    alpha=0.35,
                    zorder=3,
                ))

        plt.errorbar(
            x_vals, y_vals, yerr=y_errs,
            fmt=marker,
            color=color,
            label=_pretty_model_name(model),
            capsize=5,
            elinewidth=1.5,
            markersize=8,
            markeredgecolor='black',
            markeredgewidth=0.6,
            linestyle='-' if connect_lines else 'none',
            linewidth=1.2,
            alpha=0.85,
            zorder=4,
        )

    plt.xticks(range(len(datasets)), [f"N = {_pretty_dataset_label(d)}" for d in datasets],
               fontsize=12, fontweight='bold')

    # A vertical gridline sits ON the tick, i.e. straight through the middle
    # of the cluster it is meant to label. Drop them and draw one boundary
    # between adjacent dataset sizes instead, so each size reads as its own
    # region. Horizontal gridlines stay -- those are what the y-values are
    # read against.
    # The three dataset sizes are CATEGORIES, not positions on a number line:
    # they are drawn evenly spaced even though 180 -> 600 is 3.3x and
    # 600 -> 6,000 is 10x, so anyone reading the x-axis as a (log) scale would
    # misjudge the spacing. Everything below exists to stop it looking like an
    # axis: no vertical gridlines (which sit ON the tick, i.e. straight through
    # the middle of the cluster they label), a boundary between regions
    # instead, alternating background shading so the regions read as panels,
    # and no tick marks. Horizontal gridlines stay -- those are what the
    # y-values are read against.
    ax = plt.gca()
    ax.xaxis.grid(False)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    for i in range(len(datasets)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='0.5', alpha=0.06,
                       linewidth=0, zorder=0)
        if i < len(datasets) - 1:
            ax.axvline(i + 0.5, color='0.75', linewidth=1.0, zorder=0)

    plt.xlabel("Fine-Tuning dataset size", fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    plt.yscale(yscale)
    if relative_to_scratch:
        plt.ylim(relative_ymin, relative_ymax)  # not a hard data bound, just a sane default
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    elif metric == "r2":
        # R^2 cannot exceed 1, but a little headroom above it keeps the best-
        # model star (drawn above the top error bar) inside the frame.
        plt.ylim(bottom=r2_ymin, top=1.0 + 0.03 * (1.0 - r2_ymin))
    if scatter_ymin is not None or scatter_ymax is not None:
        plt.ylim(bottom=scatter_ymin, top=scatter_ymax)  # explicit override, takes priority over the above

    style_of = {m: (marker, color) for m, marker, color in ordered_styles}
    # Dashed reference line across each size's region at the scratch value
    # (the val-selected lr when there is a sweep), so every point reads as
    # above or below training from scratch. Skipped in the relative view,
    # where scratch is the zero line.
    scratch_ref = next((m for m in (VAL_SELECTED_SCRATCH_MODEL, SCRATCH_MODEL) if m in style_of), None)
    if not relative_to_scratch and scratch_ref is not None:
        for ds, i in x_mapping.items():
            if (scratch_ref, ds) in points:
                ax.hlines(points[(scratch_ref, ds)][1], i - 0.5, i + 0.5,
                          colors=style_of[scratch_ref][1], linestyles='--',
                          linewidth=0.9, alpha=0.7, zorder=2)

    # A small star above the best mean of each size, matching the bold cell of
    # the LaTeX table (higher is better for R^2, lower for MSE).
    higher_is_better = metric == "r2"
    for ds in datasets:
        ds_points = {m: p for (m, d), p in points.items() if d == ds}
        if not ds_points:
            continue
        best = (max if higher_is_better else min)(ds_points, key=lambda m: ds_points[m][1])
        x, y, err = ds_points[best]
        ax.annotate(r"$\star$", xy=(x, y + err), xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, color='0.35', zorder=5,
                    annotation_clip=True)

    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot -> {save_plot_path}")
    pdf_plot_path = os.path.splitext(save_plot_path)[0] + '.pdf'
    plt.savefig(pdf_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot -> {pdf_plot_path}")
    plt.close()


def generate_pretrain_size_scatter(df, save_plot_path, sort_metric, finetune_dataset_size=10000, yscale="log", task=DEFAULT_TASK):
    """Like generate_scatter_plot, but x-axis is pretrain dataset size
    instead of finetune size — asks "does pretrain data volume alone explain the ranking?"

    Only the N=<finetune_dataset_size> result is shown per model.
    Baselines have no pretraining and are excluded.

    X positions are spaced by rank, not true magnitude — the counts aren't
    comparable units across domains, so only the tick labels carry the real numbers.
    """
    if df.empty:
        return

    if sort_metric == "val":
        y_col, error_col = "Mean Val MSE", "Std Val MSE"
    else:
        y_col, error_col = "Mean Test MSE", "Std Test MSE"

    ds_key = dataset_key("3d_x_y_z", finetune_dataset_size, task)
    sub = df[(df['Dataset'] == ds_key) & (df['Model_Name'].isin(_PRETRAIN_SIZE))]
    if sub.empty:
        print(f"[pretrain-size scatter] no models with a known pretrain size "
              f"at {ds_key}; skipping")
        return

    plt.figure(figsize=(10, 5.5))

    sizes = sorted({_PRETRAIN_SIZE[m] for m in sub['Model_Name'].unique()})
    x_mapping = {s: i for i, s in enumerate(sizes)}

    models = sub['Model_Name'].unique()
    ordered_styles = build_model_styles(models)

    # Dodge only within models that share the same x (pretrain size), and keep
    # the spread narrow — with a single global offset across ALL models (old
    # behavior), the ~7 models sharing one x got dodged across half the gap to
    # the next category, making them look like a gradient of different sizes
    # instead of all being exactly equal.
    models_by_size = defaultdict(list)
    for model, marker, color in ordered_styles:
        models_by_size[_PRETRAIN_SIZE[model]].append(model)

    dodge_width = 0.24
    per_model_offset = {}
    for size, members in models_by_size.items():
        n = len(members)
        local_offsets = np.linspace(-dodge_width / 2, dodge_width / 2, n) if n > 1 else [0]
        for model, off in zip(members, local_offsets):
            per_model_offset[model] = off

    for model, marker, color in ordered_styles:
        model_data = sub[sub['Model_Name'] == model]
        if model_data.empty:
            continue
        x_vals, y_vals, y_errs = [], [], []
        for _, row in model_data.iterrows():
            x_vals.append(x_mapping[_PRETRAIN_SIZE[model]] + per_model_offset[model])
            y_vals.append(row[y_col])
            y_errs.append(row[error_col])
        plt.errorbar(
            x_vals, y_vals, yerr=y_errs,
            fmt=marker,
            color=color,
            label=_pretty_model_name(model),
            capsize=5,
            elinewidth=1.5,
            markersize=8,
            markeredgecolor='black',
            markeredgewidth=0.6,
            linestyle='none',
        )

    plt.xticks(range(len(sizes)), [_pretty_count_label(s) for s in sizes],
              fontsize=12, fontweight='bold')
    plt.xlim(-0.5, len(sizes) - 0.5)
    plt.xlabel("Pretraining Dataset Size", fontsize=14, fontweight='bold')
    plt.ylabel("Test MSE (at best val)", fontsize=14, fontweight='bold')
    plt.title(f"Test MSE vs. Pretraining Dataset Size (N = {finetune_dataset_size:,} finetune)",
             fontsize=15, fontweight='bold')
    plt.yscale(yscale)
    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved pretrain-size scatter plot -> {save_plot_path}")
    pdf_plot_path = os.path.splitext(save_plot_path)[0] + '.pdf'
    plt.savefig(pdf_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved pretrain-size scatter plot -> {pdf_plot_path}")
    plt.close()


def generate_latex_table(df, save_path=None, sort_metric="test", r2_reference_var=None, selected_scratch=None):
    """R^2 table for the appendix: one row per model, one column per fine-tuning
    dataset size, cells "mean +- total (bootstrap)", best mean per column bolded.
    selected_scratch (Dataset -> Model_Name, from select_scratch_lr) puts a
    dagger on the scratch lr chosen by validation loss in each column.

    Emits a bare `tabular` (booktabs rules, no `table` float and no caption) so
    the paper supplies its own caption/label -- in particular the note that the
    parenthesised number is the bootstrap-only uncertainty.

    Returns (latex_str, display_df, best_mask). The latter two are the same
    content as plain DataFrames, for displaying the table inside a notebook;
    the CLI path ignores them.
    """
    if df.empty:
        return None, None, None
    if r2_reference_var is None:
        print("Skipping LaTeX table: needs --metric r2 (no Var(y_test) reference).")
        return None, None, None

    if sort_metric == "val":
        mean_col, err_col, stat_col = "Mean Val MSE", "Std Val MSE", "Stat Std Val MSE"
    else:
        mean_col, err_col, stat_col = "Mean Test MSE", "Std Test MSE", "Stat Std Test MSE"

    # R^2 = 1 - MSE / Var(y_test) is affine in MSE, so the mean and both error
    # terms transform directly -- same as generate_scatter_plot.
    r2_mean = 1.0 - df[mean_col] / r2_reference_var
    r2_err = df[err_col] / r2_reference_var
    # A model with a single run has no cross-seed term, so its total error IS
    # the bootstrap one and summarize_model never wrote a Stat Std column for
    # it -- fall back to the total so the parenthesised number is still right.
    if stat_col in df.columns:
        r2_stat = df[stat_col].fillna(df[err_col]) / r2_reference_var
    else:
        r2_stat = r2_err

    tidy = pd.DataFrame({
        "model": df["Model_Name"],
        "dataset": df["Dataset"],
        "r2": r2_mean,
        "err": r2_err,
        "stat": r2_stat,
    })

    # Rows follow the scatter plot's legend order (category grouping, then
    # within-category rank) so table and figure can be read side by side.
    # Sorted here rather than through build_model_styles, which also hands out
    # colors: with the lr sweep the table has more scratch rows than the
    # baseline category has colors.
    def _order_key(m):
        cat = categorize_model(m)
        return _CATEGORY_ORDER.index(cat), _rank(m, _RANK_TABLES[cat])
    model_order = sorted(tidy["model"].unique(), key=_order_key)
    ds_order = sorted(tidy["dataset"].unique(),
                      key=lambda k: int(re.search(r'(\d+)$', k).group(1)))
    best_r2 = {ds: tidy.loc[tidy["dataset"] == ds, "r2"].max() for ds in ds_order}

    col_labels = [_pretty_dataset_label(ds) for ds in ds_order]
    # At full size the 4-column body is ~450pt wide against NeurIPS's 397pt
    # single-column \textwidth, so it is wrapped in a group that shrinks the
    # font and the inter-column padding. Delete the two brace lines to get the
    # plain tabular back (and let the paper set its own size).
    lines = [r"{\setlength{\tabcolsep}{4pt}\small",
             r"\begin{tabular}{l" + "c" * len(ds_order) + "}", r"\toprule",
             " & ".join(["Model"] + col_labels) + r" \\"]

    display_rows, mask_rows, row_labels = [], [], []
    prev_category = None
    for model in model_order:
        category = categorize_model(model)
        if category != prev_category:
            lines.append(r"\midrule")
            prev_category = category

        cells, plain_cells, mask_cells = [], [], []
        for ds in ds_order:
            match = tidy[(tidy["model"] == model) & (tidy["dataset"] == ds)]
            if match.empty:
                cells.append("--")
                plain_cells.append("--")
                mask_cells.append(False)
                continue
            row = match.iloc[0]
            is_best = bool(np.isclose(row["r2"], best_r2[ds]))
            is_selected = bool(selected_scratch) and selected_scratch.get(ds) == model
            body = f"{row['r2']:.3f} \\pm {row['err']:.3f}\\,({row['stat']:.3f})"
            math = f"\\mathbf{{{body}}}" if is_best else body
            cells.append(f"${math}^\\dagger$" if is_selected else f"${math}$")
            plain_cells.append(f"{row['r2']:.3f} ± {row['err']:.3f} ({row['stat']:.3f})"
                               + (" †" if is_selected else ""))
            mask_cells.append(is_best)

        label = _pretty_model_name(model)
        lines.append(" & ".join([label] + cells) + r" \\")
        row_labels.append(label)
        display_rows.append(plain_cells)
        mask_rows.append(mask_cells)

    lines += [r"\bottomrule", r"\end{tabular}", "}"]
    if selected_scratch:
        # The table is a bare tabular, so the caption that explains the
        # dagger lives in the paper; this comment is the reminder.
        lines.append("% dagger: the scratch lr with the lowest mean best-val loss at this size.")
    latex_str = "\n".join(lines) + "\n"

    display_df = pd.DataFrame(display_rows, index=row_labels, columns=col_labels)
    best_mask = pd.DataFrame(mask_rows, index=row_labels, columns=col_labels)
    display_df.index.name = "Model"
    best_mask.index.name = "Model"

    if save_path:
        with open(save_path, "w") as fh:
            fh.write(latex_str)
        print(f"Saved LaTeX table -> {save_path}")

    return latex_str, display_df, best_mask


def build_table(model_list, repo_root, epochs, indices, save_csv=None, save_plot=None, save_scatter=None, save_pretrain_scatter=None, save_latex=None, return_table_frames=False, sort_by="test", test_csv_name="results/loss_history.csv", yscale="log", metric="mse", r2_reference_var=None, r2_ymin=-0.3, relative_to_scratch=False, relative_baseline_model="OmniLearned_Scratch", relative_ymin=-1.0, relative_ymax=1.0, scatter_ylabel=None, value_column="Test_Loss", scatter_ymin=None, scatter_ymax=None, error_column=None, connect_lines=False, combine_errors=False, task=DEFAULT_TASK):
    display_rows = []
    plot_rows = []

    for target_columns, dataset_size, target_model, display_name in model_list:
        result_display, result_plot = summarize_model(
            repo_root,
            target_columns,
            dataset_size,
            target_model,
            indices,
            epochs,
            sort_by=sort_by,
            display_name=display_name,
            test_csv_name=test_csv_name,
            value_column=value_column,
            error_column=error_column,
            combine_errors=combine_errors,
            task=task,
        )
        if result_display:
            display_rows.append(result_display)
            plot_rows.append(result_plot)

    if not display_rows:
        print("No valid results found (check paths / run_idx).")
        return (pd.DataFrame(), None, None) if return_table_frames else pd.DataFrame()

    df = pd.DataFrame(display_rows)
    df = df.sort_values("_sort", ascending=True).reset_index(drop=True)
    df_out = df.drop(columns=["_sort"])

    if save_csv:
        df_out.to_csv(save_csv, index=False)
        print(f"Saved CSV -> {save_csv}")

    # Scratch lr sweep: the table lists every lr, the figure only the
    # val-selected one. The relative view keeps the default-lr scratch, since
    # it is that view's reference model.
    selected_scratch, val_selected_rows = select_scratch_lr(plot_rows)
    for ds, model in sorted(selected_scratch.items(),
                            key=lambda kv: int(re.search(r'(\d+)$', kv[0]).group(1))):
        print(f"Val-selected scratch lr for {ds}: {_scratch_lr(model)}")
    drop_default_scratch = bool(val_selected_rows) and not relative_to_scratch
    scatter_rows = [r for r in plot_rows
                    if not r["Model_Name"].startswith(SCRATCH_LR_PREFIX)
                    and not (drop_default_scratch and r["Model_Name"] == SCRATCH_MODEL)]
    scatter_rows += val_selected_rows

    # Appendix table (R^2, models x dataset sizes)
    table_df = best_mask = None
    if (save_latex or return_table_frames) and plot_rows:
        _, table_df, best_mask = generate_latex_table(
            pd.DataFrame(plot_rows + val_selected_rows), save_latex, sort_by,
            r2_reference_var=r2_reference_var, selected_scratch=selected_scratch)

    # Figure 1
    if save_plot and plot_rows:
        plot_df = pd.DataFrame(plot_rows)
        plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=df['Model'], ordered=True)
        plot_df = plot_df.sort_values('Model').reset_index(drop=True)
        generate_bar_chart(plot_df, save_plot, sort_by, yscale=yscale)

    # Figure 2: scattering plot
    if save_scatter and plot_rows:
        plot_df_scatter = pd.DataFrame(scatter_rows)
        generate_scatter_plot(plot_df_scatter, save_scatter, sort_by, yscale=yscale,
                              metric=metric, r2_reference_var=r2_reference_var, r2_ymin=r2_ymin,
                              relative_to_scratch=relative_to_scratch,
                              relative_baseline_model=relative_baseline_model,
                              relative_ymin=relative_ymin, relative_ymax=relative_ymax,
                              scatter_ylabel=scatter_ylabel,
                              scatter_ymin=scatter_ymin, scatter_ymax=scatter_ymax,
                              connect_lines=connect_lines)

    # Figure 3: pretrain-dataset-size diagnostic scatter
    if save_pretrain_scatter and plot_rows:
        plot_df_pretrain = pd.DataFrame(plot_rows)
        generate_pretrain_size_scatter(plot_df_pretrain, save_pretrain_scatter, sort_by, yscale=yscale, task=task)

    # Opt-in third return value set, so the notebook can render the same
    # appendix table inline (bolding via a pandas Styler) instead of only
    # writing the .tex. Off by default -- the CLI path wants a bare DataFrame.
    if return_table_frames:
        return df_out, table_df, best_mask
    return df_out


if __name__ == "__main__":
    args = parse_args()

    model_list = []
    for item in args.models:
        parts = item.split(":")
        if len(parts) == 3:
            tc, ds, tm = parts
            display = tm
        elif len(parts) == 4:
            tc, ds, tm, display = parts
        else:
            raise ValueError(
                f"Invalid model format: {item}, expected "
                f"target_columns:dataset_size:target_model[:display_name]"
            )
        model_list.append((tc, validate_dataset_size(int(ds)), tm, display))

    r2_reference_var = None
    if args.metric == "r2":
        if args.r2_reference_labels:
            r2_reference_var = compute_test_label_variance_from_labels(args.r2_reference_labels)
            print(f"R^2 reference Var(y_test) = {r2_reference_var:.6f} (from {args.r2_reference_labels})")
        elif args.r2_reference_raw:
            r2_reference_var = compute_test_label_variance(args.r2_reference_raw)
            print(f"R^2 reference Var(y_test) = {r2_reference_var:.6f} (from {args.r2_reference_raw})")
        else:
            raise ValueError("--metric r2 requires --r2_reference_raw or --r2_reference_labels")

    df = build_table(
        model_list, 
        repo_root=args.repo_root,
        epochs=args.epochs, 
        indices=args.indices, 
        save_csv=args.output, 
        save_plot=args.plot,
        save_scatter=args.scatter_plot,
        save_pretrain_scatter=args.pretrain_scatter_plot,
        save_latex=args.latex_table,
        sort_by=args.sort_by,
        test_csv_name=args.test_csv_name,
        yscale=args.yscale,
        metric=args.metric,
        r2_reference_var=r2_reference_var,
        r2_ymin=args.r2_ymin,
        relative_to_scratch=args.relative_to_scratch,
        relative_baseline_model=args.relative_baseline_model,
        relative_ymin=args.relative_ymin,
        relative_ymax=args.relative_ymax,
        scatter_ylabel=args.scatter_ylabel,
        value_column=args.value_column,
        scatter_ymin=args.scatter_ymin,
        scatter_ymax=args.scatter_ymax,
        error_column=args.error_column,
        connect_lines=args.connect_lines,
        combine_errors=args.combine_errors,
        task=args.task,
    )
    
    print("\nFinal Aggregated Table:")
    print(df.to_string(index=False))
