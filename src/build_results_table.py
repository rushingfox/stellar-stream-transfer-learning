import os
import re
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import seaborn as sns
from compute_bootstrap_shared_test import load_predictions
from exp_paths import DEFAULT_TASK, EVAL_DATASET_SIZE, dataset_key, get_run_dir, get_target_stats_path

sns.set_theme(style="whitegrid")

# Paper style: LaTeX text in Computer Modern, the serif of the paper body.
# Needs LaTeX on PATH: run `module load texlive/2024` after activating the conda env, or conda takes PATH priority.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


# Names shown in the figure legend and the LaTeX table.
# The CSV, and the code that orders, colors and groups models, keep the raw display names.
# Unlisted names are shown with underscores replaced by spaces.
_PRETTY_MODEL_NAMES = {
    'OmniCosmos_CAMELS':     'OmniCosmos-C',
    'OmniCosmos_Quijote':    'OmniCosmos-Q',
    'eigenvalue_regression': 'Eigenvalue Regression',
    # The lr-swept models (LR_SWEEPS) are named in _pretty_model_name.
    'OmniLearned_Finetune':  'OmniLearned (HEP Jets)',
    'proxy_cube':            'Cube',
    'proxy_bounded_ball':    'Bounded Ball',
    'proxy_gaussian_ball':   'Gaussian Ball',
    'proxy_gaussian_ellipsoid_sigmax': 'Gaussian Ellipsoid',
    'proxy_bounded_ellipsoid_a':       'Bounded Ellipsoid',
    'proxy_box_a':           'Box',
    'proxy_stream_eigenvalues_lambda1': r'Real-Stream PCA ($\lambda_1$)',
    # Optional models, not in the paper's default list.
    'OmniLearned_Scratch_M': 'OmniLearned-m (Scratch)',
    'OmniLearned_Finetune_M': 'OmniLearned-m (HEP Jets)',
    'proxy_stream_eigenvalues_lambda2': r'Real-Stream PCA ($\lambda_2$)',
    'proxy_stream_eigenvalues_lambda3': r'Real-Stream PCA ($\lambda_3$)',
}


def _pretty_model_name(name, table=False):
    """Legend/table name: lr-swept models from their LR_SWEEPS name, table_name or label, the rest from _PRETTY_MODEL_NAMES; table=True names the val-selected row for the table."""
    val_base = name[:-len(VAL_SELECTED_SUFFIX)] if name.endswith(VAL_SELECTED_SUFFIX) else None
    if val_base in LR_SWEEPS:
        return LR_SWEEPS[val_base]['table_name' if table else 'name']
    base = sweep_base(name)
    if base is not None:
        return LR_SWEEPS[base]['label'].format(rf"lr $= {_latex_lr(sweep_lr(name))}$")
    return _PRETTY_MODEL_NAMES.get(name, name.replace('_', ' '))


# The two baselines trained at several learning rates.
# Folder names: the default lr has no suffix (deepsets), every other lr is <base>_lr<lr> (deepsets_lr5e-4).
# default_lr must match the LR default of the runner named next to it.
# Each size gets one point in the figure, named <base>_ValLR (see select_sweep_lr), and every lr gets a row in the table.
# name labels that point in the legend, without its lr, which the paper explains.
# table_name labels it in the table, where the per-lr rows (named by label) sit next to it and a plain name would read as the default lr.
LR_SWEEPS = {
    'OmniLearned_Scratch': {'default_lr': '5e-5',  # scripts/omnilearned/run_omnilearned_scratch.sh
                            'lrs': ('1e-5', '5e-5', '1e-4', '5e-4', '1e-3'),
                            'name': "OmniLearned (Scratch)",
                            'table_name': "OmniLearned (Scratch, val-selected lr)",
                            'label': "OmniLearned (Scratch, {})"},
    'DeepSets':            {'default_lr': '1e-3',  # scripts/run_deepsets_train_common.sh
                            'lrs': ('1e-4', '5e-4', '1e-3', '5e-3', '1e-2'),
                            'name': "DeepSets",
                            'table_name': "DeepSets (val-selected lr)",
                            'label': "DeepSets ({})"},
}
VAL_SELECTED_SUFFIX = '_ValLR'
SCRATCH_MODEL = 'OmniLearned_Scratch'
VAL_SELECTED_SCRATCH_MODEL = SCRATCH_MODEL + VAL_SELECTED_SUFFIX
# The main figure draws OmniLearned scratch as a grey band, not in its category color, so every figure draws it in the grey of the band's mean line.
_SCRATCH_LINE_GREY = '0.3'


def sweep_base(name):
    """Finds which swept model (an LR_SWEEPS key) a name belongs to.

    Returns the LR_SWEEPS key, or None if the name is not in a sweep:
    - 'DeepSets' -> 'DeepSets'; 'DeepSets_lr5e-4' -> 'DeepSets'
    - 'deepsets_lr5e-4' -> 'DeepSets' (case-insensitive, for print_r2's folder names)
    - 'omnilearned_scratch_lr1e-4_ep50' -> None (extra text after the lr)
    - 'OmniCosmos_CAMELS' -> None (not swept)

    `name`: a display name (Model_Name) or a folder name.
    """
    for base in LR_SWEEPS:
        if re.fullmatch(rf"{base}(_lr\d+(\.\d+)?e-?\d+)?", name, flags=re.IGNORECASE):
            return base
    return None


def _is_default_lr(name, base):
    """('deepsets', 'DeepSets') -> True; ('DeepSets_lr5e-4', 'DeepSets') -> False."""
    return name.lower() == base.lower()


def sweep_lr(name):
    """Gives the learning rate of a swept model's name.

    Returns the lr as a string: 'OmniLearned_Scratch' -> '5e-5' (its default_lr); 'DeepSets_lr5e-4' -> '5e-4'.

    `name`: a name that sweep_base recognizes.
    """
    base = sweep_base(name)
    return LR_SWEEPS[base]['default_lr'] if _is_default_lr(name, base) else name[len(base + '_lr'):]


def _latex_lr(lr):
    """'2e-4' -> '2\\times10^{-4}' (math mode, without the dollar signs)."""
    mantissa, exponent = lr.split('e')
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def _pretty_dataset_label(ds_key):
    """'prog_mass_3d_x_y_z_10000' -> '6,000': the folder size counts train+val+test, the label shows the train split (60%)."""
    m = re.search(r'(\d+)$', ds_key)
    if not m:
        return ds_key
    train_n = int(m.group(1)) * 3 // 5
    return f"{train_n:,}"


# Four model categories, each with its own marker and colors in _CATEGORY_STYLE.
# baseline: no pretraining.
# physics: pretrained on another physics domain (HEP jets, cosmological simulations).
# synthetic: pretrained on a geometric proxy.
# eigen: the PCA eigenvalue proxy, real streams with a new target, so neither a baseline nor a geometric proxy.
# Colors are hand-picked rather than a colormap, because the advisor found that a gradient reads as shades of one thing.
_CATEGORY_ORDER = ['baseline', 'physics', 'synthetic', 'eigen']
_CATEGORY_STYLE = {
    'baseline':  {'marker': 'o', 'colors': ['#999999', '#a6763d', '#5b7c8d', '#1a1a1a']},
    'physics':   {'marker': 's', 'colors': ['#e6550d', '#d62728', '#d4a017']},
    'synthetic': {'marker': '^', 'colors': ['#1f77b4', '#17becf', '#2ca02c',
                                            '#9467bd', '#08306b', '#66c2a5']},
    'eigen':     {'marker': 'D', 'colors': ['#c51b8a']},
}

# Optional models (not in the paper's list) get a fixed shade of their base model's color instead of a category swatch.
# So adding one never recolors the paper's models, and a figure without them is unchanged.
_OPTIONAL_MODEL_COLORS = {
    'omnilearned_scratch_m':            '#2c4a5a',  # OmniLearned Scratch '#5b7c8d'
    # lr sweeps: the per-lr rows appear only in the table, so these colors are never drawn; they only keep those rows out of the category swatches.
    # The val-selected point takes the default-lr model's place in the figure, and its swatch.
    'omnilearned_scratch_lr':           '#5b7c8d',
    'deepsets_lr':                      '#a6763d',
    'omnilearned_finetune_m':           '#a63603',  # OmniLearned Finetune '#e6550d'
    'proxy_stream_eigenvalues_lambda2': '#f768a1',  # Real-Stream PCA '#c51b8a'
    'proxy_stream_eigenvalues_lambda3': '#7a0177',
}

# Order of the baselines in the legend and in the table.
_BASELINE_RANK = {
    'eigenvalue_regression': 1,
    'deepsets_vallr': 1.9,         # first DeepSets row of the table; the figure's only DeepSets point
    'deepsets_lr1e-4': 1.93,       # table only, in lr order around the default 1e-3
    'deepsets_lr5e-4': 1.96,
    'deepsets': 2,                 # the default lr, 1e-3
    'deepsets_lr5e-3': 2.05,
    'deepsets_lr1e-2': 2.1,
    'omnilearned_scratch_vallr': 2.9,   # first scratch row of the table; the figure's only scratch point
    'omnilearned_scratch_lr1e-5': 2.95, # table only, in lr order around the default 5e-5
    'omnilearned_scratch': 3,
    'omnilearned_scratch_lr1e-4': 3.1,
    'omnilearned_scratch_lr5e-4': 3.2,
    'omnilearned_scratch_lr1e-3': 3.3,
    'omnilearned_scratch_m': 3.5,  # optional, next to OmniLearned Scratch
}

# Physics order: most distant domain first.
_PHYSICS_RANK = {
    'omnilearned_finetune': 0,   # HEP jets — most distant domain
    'omnilearned_jet_finetune': 0,  # the same model by its folder name
    'omnilearned_finetune_m': 0.5,  # optional, next to OmniLearned Finetune
    'omnicosmos_camels':    1,   # cosmological, smaller-scale (CAMELS)
    'omnicosmos_quijote':   2,   # cosmological, larger-scale (Quijote)
}

# Synthetic order: by rough information content.
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
    """Rank of the longest rank_table key contained in the name (case-insensitive); 99 if none."""
    n = name.lower()
    for key in sorted(rank_table, key=len, reverse=True):
        if key in n:
            return rank_table[key]
    return 99


def categorize_model(name):
    """Assigns a model to one of the four categories (see _CATEGORY_ORDER).

    Returns 'baseline', 'physics', 'synthetic' or 'eigen'.

    `name`: a display name or folder name, matched case-insensitively by substring.
    """
    n = name.lower()
    if 'stream_eigenvalues' in n or 'eigen_lambda1' in n:
        return 'eigen'
    if 'omnilearned_finetune' in n or 'jet_finetune' in n or 'omnicosmos' in n:
        return 'physics'
    if 'scratch' in n or 'deepsets' in n or n == 'eigenvalue_regression':
        return 'baseline'
    return 'synthetic'


def build_model_styles(models):
    """Gives every model its marker (one per category) and color (one per model) for the figure.

    Returns [(model, marker, color), ...], grouped by category in _CATEGORY_ORDER and ranked within each; raises ValueError if a category has more models than colors.

    `models`: the Model_Names to draw.
    """
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
    """Fixed color from _OPTIONAL_MODEL_COLORS (longest key contained in the name), or None for the paper's models."""
    n = name.lower()
    for key in sorted(_OPTIONAL_MODEL_COLORS, key=len, reverse=True):
        if key in n:
            return _OPTIONAL_MODEL_COLORS[key]
    return None


def load_one_run(csv_path, epochs, error_column=None):
    """Reads one run's CSV and picks its best-val epoch.

    Returns (best_val, test_at_best, error_at_best), with error_at_best None if error_column is None; None if the CSV is missing, unreadable or has no epoch within the cutoff.

    `csv_path`: the run's CSV (shared_test10000/bootstrap.csv in the notebook).
    `epochs`: cutoff; rows with Epoch above it are ignored.
    `error_column`: the uncertainty column to read too (Test_Loss_Bootstrap_Std in the notebook), or None.
    """
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

    # Keep epochs <= epochs (the first `epochs` rows if there is no Epoch column).
    if "Epoch" in df.columns:
        df_sub = df[df["Epoch"] <= epochs].copy()
    else:
        df_sub = df.head(epochs).copy()

    if df_sub.empty:
        return None

    # Required columns.
    if "Val_Loss" not in df_sub.columns or "Test_Loss" not in df_sub.columns:
        raise ValueError(f"Missing Val_Loss/Test_Loss in {csv_path}. Columns: {list(df_sub.columns)}")
    if error_column is not None and error_column not in df_sub.columns:
        raise ValueError(f"Missing {error_column} in {csv_path}. Columns: {list(df_sub.columns)}")

    best_idx = df_sub["Val_Loss"].idxmin()
    best_val = float(df_sub.loc[best_idx, "Val_Loss"])
    test_at_best = float(df_sub.loc[best_idx, "Test_Loss"])
    error_at_best = float(df_sub.loc[best_idx, error_column]) if error_column is not None else None
    return best_val, test_at_best, error_at_best


def summarize_model(repo_root, target_columns, dataset_size, target_model, indices, epochs, display_name=None, test_csv_name="results/loss_history.csv", error_column=None, combine_errors=False, task=DEFAULT_TASK):
    """Aggregates one model at one dataset size over the runs in `indices`.

    Returns (display_row, plot_row): formatted strings for the CSV, and numbers for the figure, the LaTeX table and select_sweep_lr; (None, None) if no run has a result.

    `repo_root`: the repo root, where the experiment folders are.
    `target_columns`, `dataset_size`: pick the experiment folder, e.g. 3d_x_y_z and 1000.
    `target_model`: the on-disk folder name, used to resolve paths.
    `indices`: the run indices (seeds) to aggregate.
    `epochs`: the epoch cutoff passed to load_one_run.
    `display_name`: the name in the CSV and plots; defaults to target_model.
    `test_csv_name`: the CSV inside each run dir (shared_test10000/bootstrap.csv in the notebook).
    `error_column`: the uncertainty column passed to load_one_run (Test_Loss_Bootstrap_Std in the notebook), or None.
    `combine_errors`: with error_column, add the standard error of the mean over seeds to it in quadrature.
    `task`: 'prog_mass' or 'subhalo', the experiment folder prefix.
    """
    val_list = []
    test_list = []
    error_list = []
    ds_key = dataset_key(target_columns, dataset_size, task)
    if display_name is None:
        display_name = target_model

    for idx in indices:
        path = get_run_dir(repo_root, target_columns, dataset_size, target_model, idx, task) / test_csv_name

        res = load_one_run(path, epochs, error_column=error_column)
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

    # CSV row; _sort orders the CSV by mean test MSE.
    display_row = {
        "Model": model_name,
        "_sort": float(np.mean(test_list)),
        "n_indices": len(val_list)
    }
    # Numeric row for the figure, the table and select_sweep_lr.
    plot_row = {
        "Model": model_name,
        "Dataset": ds_key,
        "Model_Name": display_name,
        "n_indices": len(val_list)
    }

    # stat = mean of the runs' error_column (bootstrap std).
    if error_column is not None and error_list:
        mean_val = float(np.mean(val_list))
        mean_test = float(np.mean(test_list))
        stat_error = float(np.mean(error_list))

        if combine_errors and len(test_list) > 1:
            # total = sqrt(seed s.e.m.^2 + stat^2): the seed term is the standard error of the mean over seeds (sample std / sqrt(n_runs)).
            # stat is not divided by sqrt(n_runs), since every seed is scored on the same test set and its sampling error does not average out.
            seed_std_val = float(np.std(val_list, ddof=1) / np.sqrt(len(val_list)))
            seed_std_test = float(np.std(test_list, ddof=1) / np.sqrt(len(test_list)))
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
            # One run (or combine_errors off): the error bar is stat alone.
            display_row["Best Val MSE"] = f"{mean_val:.4f} ± {stat_error:.4f}"
            display_row["Test MSE (at best val)"] = f"{mean_test:.4f} ± {stat_error:.4f}"

            plot_row["Mean Val MSE"] = mean_val
            plot_row["Std Val MSE"] = stat_error
            plot_row["Mean Test MSE"] = mean_test
            plot_row["Std Test MSE"] = stat_error

    # No error column, one run: the value alone.
    elif len(val_list) == 1:
        display_row["Best Val MSE"] = val_list[0]
        display_row["Test MSE (at best val)"] = test_list[0]
        
        plot_row["Mean Val MSE"] = val_list[0]
        plot_row["Std Val MSE"] = 0.0
        plot_row["Mean Test MSE"] = test_list[0]
        plot_row["Std Test MSE"] = 0.0

    # No error column, two runs: [min, max], plotted as mean ± half the range.
    elif len(val_list) == 2:
        val_min, val_max = min(val_list), max(val_list)
        test_min, test_max = min(test_list), max(test_list)
        
        display_row["Best Val MSE"] = f"[{val_min:.4f}, {val_max:.4f}]"
        display_row["Test MSE (at best val)"] = f"[{test_min:.4f}, {test_max:.4f}]"
        
        plot_row["Mean Val MSE"] = np.mean(val_list)
        plot_row["Std Val MSE"] = (val_max - val_min) / 2
        plot_row["Mean Test MSE"] = np.mean(test_list)
        plot_row["Std Test MSE"] = (test_max - test_min) / 2

    # No error column, three or more runs: mean ± population std.
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


def requested_sweeps(model_list, task):
    """Collects, for each dataset size and swept model, the lrs that model_list asks for.

    Returns {(dataset key, swept model): set of display names}, e.g. {('prog_mass_3d_x_y_z_1000', 'DeepSets'): {'DeepSets', 'DeepSets_lr1e-4', ...}}; the default lr's name has no _lr suffix.

    It reads the list rather than the results, so that select_sweep_lr can tell an lr that has not run yet from one that was never asked for.

    `model_list`: build_table's list of (target_columns, dataset_size, folder name, display name).
    `task`: 'prog_mass' or 'subhalo', the prefix of the dataset key.
    """
    requested = defaultdict(set)
    for target_columns, dataset_size, target_model, display_name in model_list:
        name = display_name or target_model
        base = sweep_base(name)
        if base is not None:
            requested[(dataset_key(target_columns, dataset_size, task), base)].add(name)
    return requested


def select_sweep_lr(plot_rows, n_required, requested):
    """Picks the point the figure shows for each lr-swept model (LR_SWEEPS) at each dataset size.

    The rule; a model asked for at one lr only is not a sweep and keeps its own row:
    - Every requested lr has all n_required seeds: the lr with the lowest mean best-val loss (ties go to the smaller lr).
    - The sweep is unfinished but the default lr has all its seeds: the default lr.
    - Neither: nothing for that model at that size.

    Returns (selected, val_rows, fallback_rows, status):
    - selected: {(Dataset, swept model): the chosen Model_Name}, for the table's dagger.
    - val_rows: one <base>_ValLR row per selection, a copy of the chosen row with its lr in 'Selected_LR'; goes to the table and the figure.
    - fallback_rows: the default-lr rows standing in for unfinished sweeps, also named <base>_ValLR so they keep its marker, color and legend entry; figure only.
    - status: {(Dataset, swept model): (kind, text)} for every requested model, where kind is 'val', 'default', 'single' or 'missing' and text says which lr the figure shows; for report_sweep_selection.

    `plot_rows`: summarize_model's plot rows of every model.
    `n_required`: the number of seeds a model needs to count as complete: an int, or a {Dataset: int} dict when sizes have different seed counts.
    `requested`: the lrs asked for, from requested_sweeps.
    """
    def _req(ds):
        return n_required[ds] if isinstance(n_required, dict) else n_required

    rows = {(r["Dataset"], r["Model_Name"]): r for r in plot_rows}
    complete = {(ds, m) for (ds, m), r in rows.items() if r["n_indices"] == _req(ds)}

    selected, val_rows, fallback_rows, status = {}, [], [], {}
    for (ds, base), names in requested.items():
        req = _req(ds)
        names = sorted(names, key=lambda n: float(sweep_lr(n)))
        seeds = {n: rows[(ds, n)]["n_indices"] if (ds, n) in rows else 0 for n in names}
        if len(names) < 2:
            only = names[0]
            if seeds[only] == 0:
                status[(ds, base)] = ("missing", f"not shown: lr {sweep_lr(only)}, the only lr requested, has no results")
            else:
                note = "" if seeds[only] == req else f", {seeds[only]}/{req} seeds"
                status[(ds, base)] = ("single", f"lr {sweep_lr(only)}, the only lr requested (no sweep){note}")
            continue
        val_name = base + VAL_SELECTED_SUFFIX
        if all((ds, n) in complete for n in names):
            best = min((rows[(ds, n)] for n in names),
                       key=lambda r: (r["Mean Val MSE"], float(sweep_lr(r["Model_Name"]))))
            selected[(ds, base)] = best["Model_Name"]
            val_rows.append({**best, "Model": f"{ds}+{val_name}", "Model_Name": val_name,
                             "Selected_LR": sweep_lr(best["Model_Name"])})
            status[(ds, base)] = ("val", f"val-selected lr {sweep_lr(best['Model_Name'])}, "
                                         f"chosen from {', '.join(sweep_lr(n) for n in names)}")
            continue
        unfinished = ", ".join(f"{sweep_lr(n)} has {seeds[n]}/{req} seeds"
                               for n in names if (ds, n) not in complete)
        default = next((n for n in names if _is_default_lr(n, base)), None)
        if default is not None and (ds, default) in complete:
            fallback_rows.append({**rows[(ds, default)], "Model": f"{ds}+{val_name}",
                                  "Model_Name": val_name, "Selected_LR": sweep_lr(default)})
            status[(ds, base)] = ("default", f"DEFAULT lr {sweep_lr(default)}, since the sweep is unfinished ({unfinished})")
        else:
            status[(ds, base)] = ("missing", f"not shown: the sweep is unfinished ({unfinished}) and the default lr is not complete")
    return selected, val_rows, fallback_rows, status


def report_sweep_selection(status, problems_only=False):
    """Prints which lr the figure shows for each lr-swept model at each dataset size, and warns if any size shows no val-selected lr.

    Returns nothing; the summary goes to stdout and the warning through warnings.warn, so that it stands out in a notebook.

    The legend names a swept model without its lr, so this is where a default-lr stand-in or a missing point becomes visible.

    `status`: from select_sweep_lr.
    `problems_only`: print only the sizes that show no val-selected lr.
    """
    for base, sweep in LR_SWEEPS.items():
        entries = sorted(((ds, kind, text) for (ds, b), (kind, text) in status.items() if b == base),
                         key=lambda e: int(re.search(r'(\d+)$', e[0]).group(1)))
        if problems_only:
            entries = [e for e in entries if e[1] in ("default", "missing")]
        if not entries:
            continue
        print(f"{sweep['name']} in the figure:")
        for ds, _, text in entries:
            print(f"  N = {_pretty_dataset_label(ds)}: {text}")
        problems = [_pretty_dataset_label(ds) for ds, kind, _ in entries if kind in ("default", "missing")]
        if problems:
            warnings.warn(f"{sweep['name']}: the figure shows no val-selected lr at N = {', '.join(problems)}; "
                          f"see the lines printed above.", stacklevel=2)


def compute_test_label_variance_from_labels(labels_npy_path):
    """Computes Var(y_test), the R^2 denominator (R^2 = 1 - MSE / Var(y_test)).

    Returns the variance as a float.

    `labels_npy_path`: the small .npy of test-split targets written by src/extract_r2_reference_labels.py, so R^2 needs no raw dataset.
    """
    return float(np.var(np.load(labels_npy_path).ravel()))


def _seed_counts(df):
    """Seeds per size for the x tick labels, e.g. {'prog_mass_3d_x_y_z_40': 10, ...}, the most any model has at that size."""
    return {ds: int(df.loc[df['Dataset'] == ds, 'n_indices'].max()) for ds in df['Dataset'].unique()}


def generate_scatter_plot(df, save_plot_path, r2_reference_var, r2_ymin=0.0):
    """Draws the main figure, the test R^2 of each model at each dataset size, as PNG and PDF.

    Returns nothing; does nothing if df is empty.

    `df`: plot rows (from summarize_model and select_sweep_lr), one per model and size, with MSEs.
    `save_plot_path`: the PNG path; the PDF goes next to it under the same name.
    `r2_reference_var`: Var(y_test), to turn MSE into R^2.
    `r2_ymin`: the bottom of the y-axis.
    """
    if df.empty:
        return

    y_col, error_col, stat_col = "Mean Test MSE", "Std Test MSE", "Stat Std Test MSE"
    has_nested_error = stat_col in df.columns

    df = df.copy()
    if has_nested_error:
        # A single-run model (eigenvalue_regression, a deterministic fit) has no stat_col value, because its whole error bar is the bootstrap std.
        # Its band therefore takes the error bar, instead of being an invisible NaN.
        df[stat_col] = df[stat_col].fillna(df[error_col])
    # R^2 = 1 - MSE / Var(y_test) is affine in MSE, so the mean and the stds convert directly, without redoing the per-run aggregation.
    df[y_col] = 1.0 - df[y_col] / r2_reference_var
    df[error_col] = df[error_col] / r2_reference_var
    if has_nested_error:
        df[stat_col] = df[stat_col] / r2_reference_var
    # A model below R^2 = 0, worse than predicting the test mean, is drawn at 0 without error bar or band, so the y-axis can start at 0.
    # The table keeps its real value, and the paper's caption says how these points are drawn.
    df['Below Zero'] = df[y_col] < 0
    df.loc[df['Below Zero'], [y_col, error_col] + ([stat_col] if has_nested_error else [])] = 0.0

    # The paper includes the figure at \textwidth (5.5 in), so every font size below prints at about 0.46x its value.
    # The height leaves room for the legend in the empty lower part of the axes.
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    datasets = df['Dataset'].unique()
    models = df['Model_Name'].unique()

    x_mapping = {ds: i for i, ds in enumerate(datasets)}

    # Same marker within a category and a color per model; drawing in category order sets the legend order.
    # Every model goes in, the scratch reference too, so that no color depends on how scratch is drawn.
    ordered_styles = build_model_styles(models)
    legend_rank = {m: i for i, (m, _, _) in enumerate(ordered_styles)}
    # OmniLearned scratch (the val-selected lr if swept) is the reference every pretraining is compared with, so it is a band across each size rather than a point.
    scratch_ref = next((m for m in (VAL_SELECTED_SCRATCH_MODEL, SCRATCH_MODEL) if m in models), None)

    # Each dataset size owns 1.0 of x-space; the points spread over 0.7 of it, leaving a 0.15 gap on each side.
    dodge_width = 0.7
    # Within each size the points go by R^2, highest on the left, so each cluster reads as a ranking; ties keep the legend order.
    offsets = {}
    for ds in datasets:
        ds_rows = df[(df['Dataset'] == ds) & (df['Model_Name'] != scratch_ref)]
        ranked = sorted(zip(ds_rows['Model_Name'], ds_rows[y_col]),
                        key=lambda mr: (-mr[1], legend_rank[mr[0]]))
        slots = np.linspace(-dodge_width / 2, dodge_width / 2, len(ranked)) if len(ranked) > 1 else [0]
        for (model, _), slot in zip(ranked, slots):
            offsets[(model, ds)] = slot
    # Fixed width of the shaded bootstrap band, about the spacing between neighbouring points (dodge_width / (points per size - 1)).
    band_width = 0.05
    # Greys of the scratch band, shared with the legend: outer band = total uncertainty, inner band = bootstrap std, line = mean.
    total_grey, stat_grey, line_grey = '0.85', '0.68', _SCRATCH_LINE_GREY

    # One errorbar per drawn model, in legend order, for the legend.
    model_handles = []

    for model, marker, color in ordered_styles:
        model_data = df[df['Model_Name'] == model]
        if model_data.empty or model == scratch_ref:
            continue

        x_vals, y_vals, y_errs, y_stats = [], [], [], []
        # x of the points below R^2 = 0, drawn at 0 without error bar or band.
        below_zero_x = []

        for _, row in model_data.iterrows():
            x = x_mapping[row['Dataset']] + offsets[(model, row['Dataset'])]
            if row['Below Zero']:
                below_zero_x.append(x)
                continue
            x_vals.append(x)
            y_vals.append(row[y_col])
            y_errs.append(row[error_col])
            if has_nested_error:
                y_stats.append(row[stat_col])

        if has_nested_error:
            # The bootstrap part as a translucent band rather than a second error bar, which reads better and stays visible when short.
            # Drawn at a lower zorder, so the marker and the outer bar sit on top.
            for xv, yv, sv in zip(x_vals, y_vals, y_stats):
                ax.add_patch(Rectangle(
                    (xv - band_width / 2, yv - sv),
                    band_width, 2 * sv,
                    facecolor=color,
                    edgecolor='none',
                    alpha=0.35,
                    zorder=3,
                ))

        marker_style = dict(marker=marker, color=color, markersize=8, markeredgecolor='black',
                            markeredgewidth=0.6, linestyle='none', alpha=0.85, zorder=4)
        model_handles.append(plt.errorbar(
            x_vals, y_vals, yerr=y_errs,
            label=_pretty_model_name(model),
            capsize=5,
            elinewidth=1.5,
            **marker_style,
        ))
        if below_zero_x:
            # clip_on off, so the whole marker shows on the bottom edge rather than only its upper half.
            plt.plot(below_zero_x, [0.0] * len(below_zero_x), clip_on=False, **marker_style)

    if scratch_ref is not None:
        # The scratch band spans its whole size region: the mean as a line, the total uncertainty as the outer band and the bootstrap std as the inner one.
        for _, row in df[df['Model_Name'] == scratch_ref].iterrows():
            i = x_mapping[row['Dataset']]
            span = [i - 0.5, i + 0.5]
            y = row[y_col]
            ax.fill_between(span, y - row[error_col], y + row[error_col], color=total_grey, linewidth=0, zorder=1)
            if has_nested_error:
                ax.fill_between(span, y - row[stat_col], y + row[stat_col], color=stat_grey, linewidth=0, zorder=1)
            ax.hlines(y, *span, colors=line_grey, linewidth=1.5, zorder=2)

    # Each size's tick also gives its number of seeds, the most any model has at that size.
    seeds = _seed_counts(df)
    plt.xticks(range(len(datasets)),
               [f"N = {_pretty_dataset_label(d)}\n({seeds[d]} seed{'s' if seeds[d] > 1 else ''})" for d in datasets],
               fontsize=19, fontweight='bold')
    ax.tick_params(axis='y', labelsize=18)

    # The dataset sizes are categories, drawn evenly spaced although their ratios differ.
    # So the x-axis must not look like a number line: no vertical gridlines (they would run through each cluster) and no tick marks, but a boundary line between sizes and alternating shading.
    # Horizontal gridlines stay, since the y-values are read against them.
    ax.xaxis.grid(False)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    for i in range(len(datasets)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='0.5', alpha=0.06,
                       linewidth=0, zorder=0)
        if i < len(datasets) - 1:
            ax.axvline(i + 0.5, color='0.75', linewidth=1.0, zorder=0)

    plt.xlabel("Fine-Tuning dataset size", fontsize=22, fontweight='bold')
    plt.ylabel("Test R²", fontsize=22, fontweight='bold')
    # R^2 cannot exceed 1, so the axis ends there.
    plt.ylim(bottom=r2_ymin, top=1.0)

    # One legend inside the axes, in the empty lower part of the three largest sizes.
    # Its bottom sits just above the markers drawn at R^2 = 0, and its top below the baselines at N = 180.
    # It lists the scratch band, every drawn model and the two uncertainties, which the paper's caption defines.
    handles, labels = [], []
    if scratch_ref is not None:
        handles.append((Rectangle((0, 0), 1, 1, facecolor=total_grey, edgecolor='none'),
                        Line2D([], [], color=line_grey, linewidth=1.5)))
        labels.append(_pretty_model_name(scratch_ref))
    handles += model_handles
    labels += [h.get_label() for h in model_handles]
    # Built by hand: an errorbar of empty data has no caps, and the legend copies only the caps it finds.
    grey = '0.45'
    cap = Line2D([], [], marker='_', markersize=6, markeredgewidth=1.2, color=grey)
    error_key = ErrorbarContainer(
        (Line2D([], [], marker='o', linestyle='none', markersize=4, color=grey,
                markeredgecolor='black', markeredgewidth=0.5),
         (cap, cap),
         (LineCollection([], colors=grey, linewidths=1.2),)),
        has_yerr=True)
    handles.append(error_key)
    labels.append("Total uncertainty")
    if has_nested_error:
        handles.append(Rectangle((0, 0), 1, 1, facecolor=stat_grey, edgecolor='none'))
        labels.append("Stat. uncertainty (bootstrap)")
    ax.legend(handles=handles, labels=labels, loc='lower right', bbox_to_anchor=(0.997, 0.04), borderaxespad=0,
              ncol=3, fontsize=16, framealpha=0.9, labelspacing=0.25, columnspacing=0.6, handlelength=1.3, handletextpad=0.5, borderpad=0.3,
              # A legend errorbar is drawn about a marker high by default, so the bar would hide behind the marker; a longer bar shows it.
              handler_map={error_key: HandlerErrorbar(yerr_size=0.7)})

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot -> {save_plot_path}")
    pdf_plot_path = os.path.splitext(save_plot_path)[0] + '.pdf'
    plt.savefig(pdf_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot -> {pdf_plot_path}")
    plt.close()


def load_pooled_predictions(repo_root, target_columns, dataset_size, target_model, run_idxs, task=DEFAULT_TASK):
    """Loads the shared-test predictions of several runs of one model and stacks them.

    Returns (prediction, truth), 1-D arrays in raw target units, len(run_idxs) * 2,000 long; raises FileNotFoundError naming every run without predictions, so a missing seed never shrinks the pool silently.

    `repo_root`, `target_columns`, `dataset_size`, `target_model`, `task`: locate the model folder, as in get_run_dir.
    `run_idxs`: the runs (seeds) to stack.
    """
    stats_path = get_target_stats_path(repo_root, target_columns, dataset_size, task)
    preds, truths, missing = [], [], []
    for idx in run_idxs:
        run_dir = get_run_dir(repo_root, target_columns, dataset_size, target_model, idx, task)
        loaded = load_predictions(run_dir, EVAL_DATASET_SIZE, stats_path)
        if loaded is None:
            missing.append(str(run_dir))
            continue
        preds.append(loaded[0])
        truths.append(loaded[1])
    if missing:
        raise FileNotFoundError("No shared-test predictions in: " + ", ".join(missing))
    return np.concatenate(preds), np.concatenate(truths)


def _binned_percentiles(x, values, edges, percentiles=(16, 50, 84)):
    """Percentiles of values in each bin of x, as an array of shape (len(percentiles), n_bins); the last bin includes its right edge."""
    bins = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2)
    return np.array([np.percentile(values[bins == b], percentiles) for b in range(len(edges) - 1)]).T


# The two residuals generate_residual_plot can draw, from the log residual d = y_pred - y_true: (values from d, y-axis label).
# The target y is log10 of the stellar mass, so the relative error of the mass itself is (M_pred - M_true) / M_true = 10^d - 1, in percent here.
# It is asymmetric, bounded below by -100% and unbounded above, so a symmetric error in d looks larger above 0 than below.
# The log residual d itself, in dex, is the quantity the MSE is computed on, and is symmetric.
_RESIDUAL_QUANTITIES = {
    'relative': (lambda d: 100 * (10 ** d - 1), r"$(M_{\rm pred} - M_{\rm true}) / M_{\rm true}$ [\%]"),
    'log':      (lambda d: d, r"$\log_{10} M_{\rm pred} - \log_{10} M_{\rm true}$ [dex]"),
}


def generate_residual_plot(panels, save_plot_path, quantity='relative', n_bins=9):
    """Draws a residual of a few models on the shared test set against the true mass, one panel per dataset size, as PNG and PDF.

    Returns nothing.

    Each panel pools every seed's prediction of every test stream and bins them by true mass, in n_bins equal-width bins over the test set's range.
    Per model it draws the median residual as a line and its 16th to 84th percentiles as a band.
    Both quantities are monotonic in y_pred - y_true, so the medians of the two figures are the same streams, transformed.

    `panels`: [(title, [(label, color, marker, prediction, truth), ...]), ...], top to bottom, models in legend order; marker None draws the line alone.
    `save_plot_path`: the PNG path; the PDF goes next to it under the same name.
    `quantity`: 'relative' for the relative mass error in percent, 'log' for the log residual in dex (see _RESIDUAL_QUANTITIES).
    `n_bins`: the number of mass bins.
    """
    transform, ylabel = _RESIDUAL_QUANTITIES[quantity]
    # Every model is scored on the same test set, so one set of edges serves every panel.
    all_truth = np.concatenate([truth for _, models in panels for *_, truth in models])
    edges = np.linspace(all_truth.min(), all_truth.max(), n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.6 * len(panels)), sharex=True, sharey=True, squeeze=False)
    for ax, (title, models) in zip(axes[:, 0], panels):
        ax.axhline(0.0, color='0.4', linestyle='--', linewidth=1.0, zorder=1)
        for label, color, marker, prediction, truth in models:
            p16, p50, p84 = _binned_percentiles(truth, transform(prediction - truth), edges)
            ax.fill_between(centers, p16, p84, color=color, alpha=0.2, linewidth=0, zorder=2)
            ax.plot(centers, p50, color=color, linewidth=2, zorder=3, label=label,
                    **({} if marker is None else dict(marker=marker, markersize=6,
                                                      markeredgecolor='black', markeredgewidth=0.5)))
        ax.set_title(title, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(labelsize=13)
    axes[-1, 0].set_xlabel(r"$\log_{10}(M_{\rm true} / M_\odot)$", fontsize=15)
    # One legend in a row above the panels, since the wide bands leave no empty corner inside them.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(labels), fontsize=13,
               frameon=False)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved residual plot -> {save_plot_path}")
    pdf_plot_path = os.path.splitext(save_plot_path)[0] + '.pdf'
    plt.savefig(pdf_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved residual plot -> {pdf_plot_path}")
    plt.close(fig)


def generate_latex_table(df, r2_reference_var, save_path=None, selected_lr=None):
    """Makes the appendix R^2 table: one row per model, one column per dataset size, the best mean per column in bold.

    Returns (latex_str, display_df, best_mask); the last two hold the same table as DataFrames, for showing it in the notebook; (None, None, None) if df is empty.

    Cells are "mean ± total", the figure's outer error bar; the bootstrap-only part is left out (it stays in the CSV and print_r2 --compact).
    The LaTeX is a bare tabular (booktabs rules, no table float or caption), so the paper adds its own caption and label.

    `df`: plot rows of every model, including the val-selected rows.
    `r2_reference_var`: Var(y_test), to turn MSE into R^2.
    `save_path`: the .tex path to write, or None.
    `selected_lr`: {(Dataset, swept model): Model_Name} from select_sweep_lr; that lr gets a dagger.
    """
    if df.empty:
        return None, None, None

    # R^2 is affine in MSE, so the mean and the error convert directly, as in generate_scatter_plot.
    tidy = pd.DataFrame({
        "model": df["Model_Name"],
        "dataset": df["Dataset"],
        "r2": 1.0 - df["Mean Test MSE"] / r2_reference_var,
        "err": df["Std Test MSE"] / r2_reference_var,
    })

    # Rows follow the figure's legend order (category, then rank), so table and figure read side by side.
    # Sorted here rather than with build_model_styles, which also assigns colors and has too few baseline colors for all the lr rows.
    def _order_key(m):
        cat = categorize_model(m)
        return _CATEGORY_ORDER.index(cat), _rank(m, _RANK_TABLES[cat])
    model_order = sorted(tidy["model"].unique(), key=_order_key)
    ds_order = sorted(tidy["dataset"].unique(),
                      key=lambda k: int(re.search(r'(\d+)$', k).group(1)))
    best_r2 = {ds: tidy.loc[tidy["dataset"] == ds, "r2"].max() for ds in ds_order}

    col_labels = [_pretty_dataset_label(ds) for ds in ds_order]
    # At full size the table is wider than NeurIPS's single-column \textwidth, so a group shrinks the font and the column padding.
    # Delete the two brace lines to get the plain tabular back.
    lines = [r"{\setlength{\tabcolsep}{4pt}\small",
             r"\begin{tabular}{l" + "c" * len(ds_order) + "}", r"\toprule",
             " & ".join(["Model"] + col_labels) + r" \\"]

    def _sweep_group(m):
        """The LR_SWEEPS key of a per-lr or val-selected row ('DeepSets_ValLR' -> 'DeepSets'), else None."""
        return sweep_base(m[:-len(VAL_SELECTED_SUFFIX)] if m.endswith(VAL_SELECTED_SUFFIX) else m)

    display_rows, mask_rows, row_labels = [], [], []
    prev_category = prev_group = None
    for model in model_order:
        category, group = categorize_model(model), _sweep_group(model)
        # A \midrule between categories, and a lighter \cmidrule around each lr sweep's rows inside one.
        if category != prev_category:
            lines.append(r"\midrule")
        elif group != prev_group and (group or prev_group):
            lines.append(rf"\cmidrule(lr){{1-{len(ds_order) + 1}}}")
        prev_category, prev_group = category, group

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
            is_selected = bool(selected_lr) and selected_lr.get((ds, sweep_base(model))) == model
            body = f"{row['r2']:.3f} \\pm {row['err']:.3f}"
            math = f"\\mathbf{{{body}}}" if is_best else body
            cells.append(f"${math}^\\dagger$" if is_selected else f"${math}$")
            plain_cells.append(f"{row['r2']:.3f} ± {row['err']:.3f}"
                               + (" †" if is_selected else ""))
            mask_cells.append(is_best)

        label = _pretty_model_name(model, table=True)
        lines.append(" & ".join([label] + cells) + r" \\")
        row_labels.append(label)
        display_rows.append(plain_cells)
        mask_rows.append(mask_cells)

    lines += [r"\bottomrule", r"\end{tabular}", "}"]
    if selected_lr:
        # The tabular has no caption, so this LaTeX comment reminds the paper's caption to explain the dagger.
        lines.append("% dagger: in each lr sweep, the lr with the lowest mean best-val loss at this size.")
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


_SINGLE_RUN_MODELS = {"eigenvalue_regression", "average"}


def _residual_panels(model_list, residual_models, residual_sizes, scatter_models, selected_lr, fallback_rows,
                     requested, repo_root, indices, task):
    """Collects the residual figure's data: for each size, each requested model's pooled predictions at the main figure's runs, with its main-figure label and color.

    Returns generate_residual_plot's panels; raises ValueError for a model or size the main figure does not show, and FileNotFoundError for a run without predictions.

    A swept model (LR_SWEEPS) is asked for by its base name and takes the lr that the main figure shows at that size.
    OmniLearned scratch is drawn in the grey of its band in the main figure, with the baseline marker, and every other model in its main-figure color and marker.

    `model_list`: build_table's list of (target_columns, dataset_size, folder name, display name).
    `residual_models`: the display names to draw, in legend order, e.g. ['OmniLearned_Scratch', 'OmniCosmos_Quijote'].
    `residual_sizes`: the dataset sizes, one panel each, top to bottom, e.g. [40, 100].
    `scatter_models`: the main figure's Model_Names, for its colors.
    `selected_lr`, `fallback_rows`: from select_sweep_lr, for the lr each swept model shows.
    `requested`: from requested_sweeps.
    `repo_root`, `indices`, `task`: as in build_table.
    """
    styles = {m: (marker, color) for m, marker, color in build_model_styles(scatter_models)}
    folder_of = {(size, display or folder): (target_columns, folder)
                 for target_columns, size, folder, display in model_list}
    fallback = {(r["Dataset"], sweep_base(r["Model_Name"][:-len(VAL_SELECTED_SUFFIX)])) for r in fallback_rows}

    panels = []
    for size in residual_sizes:
        columns = {target_columns for target_columns, s, _, _ in model_list if s == size}
        if len(columns) != 1:
            raise ValueError(f"Residual figure: N = {size} needs exactly one target_columns in model_list, found {sorted(columns)}")
        ds_key = dataset_key(columns.pop(), size, task)

        models, n_seeds = [], 0
        for name in residual_models:
            base = sweep_base(name)
            # A swept model shows one lr per size in the main figure: the val-selected one, or the default while the sweep is unfinished.
            if base is not None and len(requested.get((ds_key, base), ())) > 1:
                scatter_name = base + VAL_SELECTED_SUFFIX
                chosen = selected_lr.get((ds_key, base))
                if chosen is None and (ds_key, base) in fallback:
                    chosen = next(n for n in requested[(ds_key, base)] if _is_default_lr(n, base))
            else:
                scatter_name = chosen = name
            if chosen is None or scatter_name not in styles or (size, chosen) not in folder_of:
                raise ValueError(f"Residual figure: {name} has no point in the main figure at N = {_pretty_dataset_label(ds_key)}")

            target_columns, folder = folder_of[(size, chosen)]
            ds_indices = indices[size] if isinstance(indices, dict) else indices
            run_idxs = [0] if folder.lower() in _SINGLE_RUN_MODELS else list(ds_indices)
            prediction, truth = load_pooled_predictions(repo_root, target_columns, size, folder, run_idxs, task)
            n_seeds = max(n_seeds, len(run_idxs))

            if scatter_name in (VAL_SELECTED_SCRATCH_MODEL, SCRATCH_MODEL):
                marker, color = _CATEGORY_STYLE['baseline']['marker'], _SCRATCH_LINE_GREY
            else:
                marker, color = styles[scatter_name]
            models.append((_pretty_model_name(scatter_name), color, marker, prediction, truth))

        panels.append((f"N = {_pretty_dataset_label(ds_key)} ({n_seeds} seed{'s' if n_seeds > 1 else ''})", models))
    return panels


def build_table(model_list, repo_root, epochs, indices, r2_reference_var, save_csv=None, save_scatter=None, save_latex=None, return_table_frames=False, test_csv_name="results/loss_history.csv", r2_ymin=0.0, error_column=None, combine_errors=False, task=DEFAULT_TASK,
                save_residual=None, save_log_residual=None, residual_models=(), residual_sizes=(40, 100), residual_bins=9):
    """Builds the CSV, the main figure, the appendix table and optionally the residual figure from a list of models; the notebook's entry point.

    Returns df_out, the CSV table (empty if no run has a result); with return_table_frames, (df_out, table_df, best_mask).

    `model_list`: (target_columns, dataset_size, folder name, display name) tuples, one per model and size.
    `repo_root`: the repo root, where the experiment folders are.
    `epochs`: the epoch cutoff for picking the best-val row (see load_one_run).
    `indices`: the run indices (seeds) to aggregate, or a {dataset_size: indices} dict when sizes have different seed counts.
    `r2_reference_var`: Var(y_test), to turn MSE into R^2.
    `save_csv`, `save_scatter`, `save_latex`: output paths; None skips that output.
    `return_table_frames`: also return the table as DataFrames, for showing it in the notebook.
    `test_csv_name`, `error_column`, `combine_errors`, `task`: passed to summarize_model.
    `r2_ymin`: the bottom of the figure's y-axis.
    `save_residual`: the PNG path of the relative mass error figure, in percent (see generate_residual_plot); None skips it.
    `save_log_residual`: the PNG path of the log residual figure, in dex; None skips it.
    `residual_models`: the two residual figures' models, by display name, in legend order; a swept model by its base name, e.g. 'OmniLearned_Scratch', takes the lr the main figure shows.
    `residual_sizes`: their dataset sizes, one panel each, top to bottom.
    `residual_bins`: their number of equal-width true-mass bins.
    """
    if (save_residual or save_log_residual) and not residual_models:
        raise ValueError("save_residual and save_log_residual need residual_models")
    display_rows = []
    plot_rows = []
    n_required = {}
    incomplete = []

    for target_columns, dataset_size, target_model, display_name in model_list:
        ds_indices = indices[dataset_size] if isinstance(indices, dict) else indices
        ds_key = dataset_key(target_columns, dataset_size, task)
        n_required[ds_key] = len(ds_indices)
        result_display, result_plot = summarize_model(
            repo_root,
            target_columns,
            dataset_size,
            target_model,
            ds_indices,
            epochs,
            display_name=display_name,
            test_csv_name=test_csv_name,
            error_column=error_column,
            combine_errors=combine_errors,
            task=task,
        )
        if result_display:
            display_rows.append(result_display)
            plot_rows.append(result_plot)
        name = display_name or target_model
        if target_model.lower() not in _SINGLE_RUN_MODELS and sweep_base(name) is None:
            n_found = result_plot["n_indices"] if result_plot else 0
            if n_found < len(ds_indices):
                incomplete.append(f"{ds_key}+{name} ({n_found}/{len(ds_indices)} seeds)")

    if not display_rows:
        print("No valid results found (check paths / run_idx).")
        return (pd.DataFrame(), None, None) if return_table_frames else pd.DataFrame()

    if incomplete:
        warnings.warn("Incomplete seeds for non-swept model(s): " + ", ".join(incomplete), stacklevel=2)

    df = pd.DataFrame(display_rows)
    df = df.sort_values("_sort", ascending=True).reset_index(drop=True)
    df_out = df.drop(columns=["_sort"])

    if save_csv:
        df_out.to_csv(save_csv, index=False)
        print(f"Saved CSV -> {save_csv}")

    # lr sweeps: the table lists every lr, the figure one point per size (select_sweep_lr).
    requested = requested_sweeps(model_list, task)
    selected_lr, val_selected_rows, fallback_rows, sweep_status = select_sweep_lr(plot_rows, n_required, requested)
    report_sweep_selection(sweep_status)
    swept = {key for key, names in requested.items() if len(names) > 1}

    def _in_scatter(row):
        """False for a per-lr row of a sweep, which the figure replaces with the <base>_ValLR point."""
        base = sweep_base(row["Model_Name"])
        return base is None or (row["Dataset"], base) not in swept
    scatter_rows = [r for r in plot_rows if _in_scatter(r)] + val_selected_rows + fallback_rows

    # Appendix table (R^2, models x dataset sizes)
    table_df = best_mask = None
    if save_latex or return_table_frames:
        _, table_df, best_mask = generate_latex_table(
            pd.DataFrame(plot_rows + val_selected_rows), r2_reference_var,
            save_path=save_latex, selected_lr=selected_lr)

    # Main figure: R^2 per model and dataset size
    if save_scatter:
        generate_scatter_plot(pd.DataFrame(scatter_rows), save_scatter, r2_reference_var,
                              r2_ymin=r2_ymin)

    # Residual figures: relative mass error and log residual against true mass for a few models, pooled over the main figure's runs.
    # Both read the same predictions, which are loaded once.
    if save_residual or save_log_residual:
        panels = _residual_panels(model_list, residual_models, residual_sizes,
                                  {r["Model_Name"] for r in scatter_rows}, selected_lr, fallback_rows,
                                  requested, repo_root, indices, task)
        if save_residual:
            generate_residual_plot(panels, save_residual, quantity='relative', n_bins=residual_bins)
        if save_log_residual:
            generate_residual_plot(panels, save_log_residual, quantity='log', n_bins=residual_bins)

    # In case the notebook wants to display the table, return the DataFrame and the best-mask DataFrame as well.
    if return_table_frames:
        return df_out, table_df, best_mask
    return df_out
