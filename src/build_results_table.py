import os
import re
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from exp_paths import DEFAULT_TASK, dataset_key, get_run_dir

sns.set_theme(style="whitegrid")

# Paper style: LaTeX text in Helvetica.
# Needs LaTeX on PATH: run `module load texlive/2024` after activating the conda env, or conda takes PATH priority.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
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
    `combine_errors`: with error_column, add the seed std to it in quadrature.
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
            # total = sqrt(seed std^2 + stat^2); both are population stds, not divided by sqrt(n_runs).
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
    `n_required`: the number of seeds a model needs to count as complete (len(indices)).
    `requested`: the lrs asked for, from requested_sweeps.
    """
    rows = {(r["Dataset"], r["Model_Name"]): r for r in plot_rows}
    complete = {key for key, r in rows.items() if r["n_indices"] == n_required}

    selected, val_rows, fallback_rows, status = {}, [], [], {}
    for (ds, base), names in requested.items():
        names = sorted(names, key=lambda n: float(sweep_lr(n)))
        seeds = {n: rows[(ds, n)]["n_indices"] if (ds, n) in rows else 0 for n in names}
        if len(names) < 2:
            only = names[0]
            if seeds[only] == 0:
                status[(ds, base)] = ("missing", f"not shown: lr {sweep_lr(only)}, the only lr requested, has no results")
            else:
                note = "" if seeds[only] == n_required else f", {seeds[only]}/{n_required} seeds"
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
        unfinished = ", ".join(f"{sweep_lr(n)} has {seeds[n]}/{n_required} seeds"
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


def generate_scatter_plot(df, save_plot_path, r2_reference_var, r2_ymin=-0.3):
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

    plt.figure(figsize=(12, 5.5))

    datasets = df['Dataset'].unique()
    models = df['Model_Name'].unique()

    x_mapping = {ds: i for i, ds in enumerate(datasets)}

    # Same marker within a category and a color per model; drawing in category order sets the legend order.
    ordered_styles = build_model_styles(models)
    legend_rank = {m: i for i, (m, _, _) in enumerate(ordered_styles)}

    # Each dataset size owns 1.0 of x-space; the points spread over 0.7 of it, leaving a 0.15 gap on each side.
    dodge_width = 0.7
    # Within each size the points go by R^2, highest on the left, so each cluster reads as a ranking; ties keep the legend order.
    offsets = {}
    for ds in datasets:
        ds_rows = df[df['Dataset'] == ds]
        ranked = sorted(zip(ds_rows['Model_Name'], ds_rows[y_col]),
                        key=lambda mr: (-mr[1], legend_rank[mr[0]]))
        slots = np.linspace(-dodge_width / 2, dodge_width / 2, len(ranked)) if len(ranked) > 1 else [0]
        for (model, _), slot in zip(ranked, slots):
            offsets[(model, ds)] = slot
    # Fixed width of the shaded bootstrap band, about the spacing between neighbouring points (dodge_width / (points per size - 1)).
    band_width = 0.05

    # (x, y, error) of every drawn point, for the annotations added below.
    points = {}

    for model, marker, color in ordered_styles:
        model_data = df[df['Model_Name'] == model]
        if model_data.empty:
            continue

        x_vals, y_vals, y_errs, y_stats = [], [], [], []

        for _, row in model_data.iterrows():
            x_vals.append(x_mapping[row['Dataset']] + offsets[(model, row['Dataset'])])
            y_vals.append(row[y_col])
            y_errs.append(row[error_col])
            if has_nested_error:
                y_stats.append(row[stat_col])
            points[(model, row['Dataset'])] = (x_vals[-1], y_vals[-1], y_errs[-1])

        if has_nested_error:
            # The bootstrap part as a translucent band rather than a second error bar, which reads better and stays visible when short.
            # Drawn at a lower zorder, so the marker and the outer bar sit on top.
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
            linestyle='none',
            alpha=0.85,
            zorder=4,
        )

    plt.xticks(range(len(datasets)), [f"N = {_pretty_dataset_label(d)}" for d in datasets],
               fontsize=12, fontweight='bold')

    # The dataset sizes are categories, drawn evenly spaced although their ratios differ.
    # So the x-axis must not look like a number line: no vertical gridlines (they would run through each cluster) and no tick marks, but a boundary line between sizes and alternating shading.
    # Horizontal gridlines stay, since the y-values are read against them.
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
    plt.ylabel("Test R²", fontsize=14, fontweight='bold')
    # R^2 cannot exceed 1; a little headroom keeps the best-model star, drawn above the error bar, inside the frame.
    plt.ylim(bottom=r2_ymin, top=1.0 + 0.03 * (1.0 - r2_ymin))

    style_of = {m: (marker, color) for m, marker, color in ordered_styles}
    # Dashed line across each size's region at the scratch value (the val-selected lr if swept), so every point reads as above or below scratch.
    scratch_ref = next((m for m in (VAL_SELECTED_SCRATCH_MODEL, SCRATCH_MODEL) if m in style_of), None)
    if scratch_ref is not None:
        for ds, i in x_mapping.items():
            if (scratch_ref, ds) in points:
                ax.hlines(points[(scratch_ref, ds)][1], i - 0.5, i + 0.5,
                          colors=style_of[scratch_ref][1], linestyles='--',
                          linewidth=0.9, alpha=0.7, zorder=2)

    # A small star above the best mean at each size, matching the bold cell of the LaTeX table.
    for ds in datasets:
        ds_points = {m: p for (m, d), p in points.items() if d == ds}
        if not ds_points:
            continue
        best = max(ds_points, key=lambda m: ds_points[m][1])
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


def build_table(model_list, repo_root, epochs, indices, r2_reference_var, save_csv=None, save_scatter=None, save_latex=None, return_table_frames=False, test_csv_name="results/loss_history.csv", r2_ymin=-0.3, error_column=None, combine_errors=False, task=DEFAULT_TASK):
    """Builds the CSV, the main figure and the appendix table from a list of models; the notebook's entry point.

    Returns df_out, the CSV table (empty if no run has a result); with return_table_frames, (df_out, table_df, best_mask).

    `model_list`: (target_columns, dataset_size, folder name, display name) tuples, one per model and size.
    `repo_root`: the repo root, where the experiment folders are.
    `epochs`: the epoch cutoff for picking the best-val row (see load_one_run).
    `indices`: the run indices (seeds) to aggregate.
    `r2_reference_var`: Var(y_test), to turn MSE into R^2.
    `save_csv`, `save_scatter`, `save_latex`: output paths; None skips that output.
    `return_table_frames`: also return the table as DataFrames, for showing it in the notebook.
    `test_csv_name`, `error_column`, `combine_errors`, `task`: passed to summarize_model.
    `r2_ymin`: the bottom of the figure's y-axis.
    """
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
            display_name=display_name,
            test_csv_name=test_csv_name,
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

    # lr sweeps: the table lists every lr, the figure one point per size (select_sweep_lr).
    requested = requested_sweeps(model_list, task)
    selected_lr, val_selected_rows, fallback_rows, sweep_status = select_sweep_lr(plot_rows, len(indices), requested)
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

    # In case the notebook wants to display the table, return the DataFrame and the best-mask DataFrame as well.
    if return_table_frames:
        return df_out, table_df, best_mask
    return df_out
