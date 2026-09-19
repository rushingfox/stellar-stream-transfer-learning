# AGENTS.md

Instructions for AI coding agents working in this repository.
[README.md](README.md) is the source of truth for end-to-end reproduction commands; this
file documents architecture and conventions that the README does not spell out.

## What this repo is

Supervised regression of the **log progenitor mass** (`Log_Prog_Mass`) of synthetic
stellar streams, used to compare **transfer-learning strategies** — physics-domain
pretraining (jets, cosmology) and synthetic geometric-proxy pretraining — against
from-scratch baselines, at three dataset sizes. Each stream is a point cloud of 4,000
particles in 3D Cartesian coordinates.

Datasets are split train:val:test = 3:1:1, so the three sizes give **180 / 600 / 6,000**
training streams. Every model is finally scored on one **shared 2,000-stream test set**
taken from the 10,000-stream dataset, not on its own split.

## Infrastructure

- Runs on **NERSC Perlmutter**; all training is submitted via `sbatch`. The canonical
  `#SBATCH` parameters live in the heads of the `scripts/**/run_*.sh` files.
- Conda env: **`stream_transfer_learning`** (built from [pyproject.toml](pyproject.toml)
  via `pip install -e .`). The external `omnilearned` CLI
  (<https://github.com/ViniciusMikuni/OmniLearned>, installed `-e` from a separate local
  clone) is the workhorse for every non-baseline model and is **not** in the toml — see
  the README's Installation section.
- **LaTeX**: every figure script sets `text.usetex=True` with Helvetica, so `latex`,
  `dvipng` and `kpsewhich` must be on `PATH`. Run `module load texlive/2024` **after**
  activating the conda env (conda activation otherwise takes PATH priority). The runner
  scripts already do this in the right order. Jupyter kernels do **not** inherit
  `module load`; the notebook's first cell captures `$PATH` from a subprocess instead.
- **matplotlib is pinned `<3.11`.** 3.11 added the `axes3d.depthshade_minalpha` rcParam
  (default 0.3), which multiplies into the alpha already set on the 3D proxy point
  clouds and visibly washes them out. Do not relax this pin without regenerating and
  visually checking `proxy_gallery/`.
- **Reading PDFs**: for text extraction (e.g. `pypdf`) use NERSC's shared read-only
  `base` env (`module load conda && source activate base`). For rendering PDF pages to
  images (`pdftoppm`/poppler) use the user-owned `pdftools` env.
  `stream_transfer_learning` has neither.
- **`--qos=debug` has a hard 30-minute wall-time cap**, and NERSC rejects more than
  roughly 3 concurrent debug jobs per user. Any submission switched to `debug` must also
  pass `--time` ≤ `00:30:00`, because the runner scripts' own `#SBATCH --time` headers
  default to much longer (e.g. `04:00:00`) and the job otherwise fails immediately with
  `QOSMaxWallDurationPerJobLimit`.
- `--qos=shared` GPU jobs occasionally fail with `srun: error: Unable to create step for
  job <id>: Error generating job credential` → infinite retry → wall-time kill. This is a
  NERSC scheduler flake, not a code bug. Resubmit only the stuck array indices via
  `SBATCH_ARRAY="<indices>"`.

## Folder convention

Experiment results live at `<target_columns>_<dataset_size>/<model>/run_<seed_idx>/`.

- `dataset_size` ∈ `{300, 1000, 10000}` — the three sizes the paper reports, and the only
  ones with tracked results. [src/exp_paths.py](src/exp_paths.py)'s
  `VALID_DATASET_SIZES` also still accepts a couple of smaller exploratory sizes that are
  not used anywhere; treat those as legacy.
- `target_columns` for the real-stream task is `3d_x_y_z`. The proxy pretraining
  datasets get their own top-level dirs (see below).
- Path helpers in [src/exp_paths.py](src/exp_paths.py) (`dataset_key`,
  `get_loss_history_csv_path`, …) are the canonical way to construct paths — use them
  rather than f-strings.
- `.gitignore` ignores the `3d_*/ 4d_*/ 6d_*/` result dirs wholesale and then
  re-admits, by extension, only the small files the figures are rebuilt from
  (`*.csv`, `*.npz`, `hparams.json`, and the pretrain `loss_curve.png` / `r2_curve.png`).
  Read the comments in [.gitignore](.gitignore) before adding a rule — the directory
  negations must come before the file ones or git will not even traverse into them.
- Tracked top-level dirs: `src/`, `scripts/`, `evaluation/`, `visualization/`,
  `proxy_gallery/`, plus the ten result dirs above. Raw data (`raw_data*/`) and all
  model weights (`*.pt`, `*.pth`, `*.h5`) are **never** committed.

## The seven proxies

Six synthetic geometries plus one built from real streams. All are **1-target**: for the
anisotropic shapes only the longest axis is regressed, so isotropic and anisotropic
pretraining tasks stay comparable.

| proxy dir | target |
|---|---|
| `3d_proxy_cube_10000` | `a` (edge) |
| `3d_proxy_bounded_ball_10000` | `r` |
| `3d_proxy_gaussian_ball_10000` | `σ` |
| `3d_proxy_box_a_10000` | `a` (longest semi-edge) |
| `3d_proxy_bounded_ellipsoid_a_10000` | `a` (longest semi-axis) |
| `3d_proxy_gaussian_ellipsoid_sigmax_10000` | `σ_x` |
| `3d_x_y_z_eigenvalues_lambda1_10000` | `√λ₁` of the real-stream PCA |

[src/prepare_data.py](src/prepare_data.py) synthesises the **3-target** parents
(`3d_proxy_box`, `3d_proxy_hard_ellipsoid`, `3d_proxy_ellipsoid`,
`3d_x_y_z_eigenvalues`); the four `src/make_*_from_*.py` scripts then slice them down to
the 1-target datasets listed above. Those slicers infer the repo root from
`Path(__file__).resolve().parent.parent` and read from
`<parent>/omnilearned_proxy_pretrain_1x4/data/streams` — the `_1x4` suffix matters, it is
where `prepare_data_proxy_pretrain.sh` actually writes.

Each proxy z-scores its labels using **train-split statistics**, persisted to
`streams/<synthesis_tag>_stats.npz` so val/test reuse the same scaler. The filename uses
the **synthesis** tag, not the final 1-target name — e.g. `gaussian_ellipsoid_sigmax`
inherits `ellipsoid_stats.npz`. `prepare_data.py` raises `FileNotFoundError` if the train
split was not processed first.

Rotation augmentation (a random 3D rotation inside the synthesis branch) is applied to
`ellipsoid`, `cube`, `hard_ellipsoid` and `box`. It is **not** applied to `bounded_ball`
or `gaussian_ball` (isotropic, so it would be a no-op), nor to `3d_x_y_z_eigenvalues`
(those are real streams, already at arbitrary orientations).

Proxy point clouds read `num_particles` from the raw HDF5 (4,000) rather than hardcoding
it. Splits are `int(n*0.6)` / `int(n*0.8)`, i.e. 3:1:1.

## Two model families

1. **In-house DeepSets baseline** ([src/main.py](src/main.py) + [src/model.py](src/model.py)
   + [src/dataset.py](src/dataset.py)): MLP encoder → mean+max+std pooling. Hidden dim is
   configurable (`baseline` = 128, `baseline_256` = 256); the paper reports `baseline_256`
   as "DeepSets". Submitted through
   [scripts/submit_training_variant.sh](scripts/submit_training_variant.sh), which handles
   **only** the `baseline` / `baseline_256` variants.
2. **OmniLearned transformer** (external CLI): every `omnilearned_*` and `omnicosmos_*`
   model. All of these live in [scripts/stream_1x4/](scripts/stream_1x4/) and call
   `srun omnilearned train …` on 1 node × 4 GPUs. Pretrain checkpoints come either from
   the OmniLearned/OmniCosmos releases (placed in `omnicosmos_checkpoints/`, gitignored)
   or from local proxy pretraining.

Both families share the same per-run output convention: each `run_<idx>/` ends up with a
`loss_history.csv` — written by [src/omnilearned_summary.py](src/omnilearned_summary.py)
for the OmniLearned family, directly by `main.py` for baselines. **Evaluation reads only
that CSV.**

These models are regression-only. Classification support (`--mode`, accuracy, confusion
matrices) was removed; do not reintroduce it.

## Model sizes

| model | parameters |
|---|---|
| `baseline` (DeepSets, hidden 128) | 41,985 |
| `baseline_256` (DeepSets, hidden 256) | 116,609 |
| OmniLearned, every variant | 1,389,477 |

Every OmniLearned variant — from scratch, jet-pretrained, cosmology-pretrained,
proxy-pretrained — shares the same backbone, so differences between them come only from
initialisation. The DeepSets baselines are small sanity checks, **not** capacity-matched
competitors; read the figure with that in mind. (These counts are inherited from the
pre-cleanup archive; the `count_params.py` that produced them is no longer in the repo,
so re-derive from a checkpoint if you need to be certain.)

## Proxy pretrain → finetune wiring

The most coupled piece in the repo:

1. **Pretrain**: `TARGET_COLUMN=<proxy> bash scripts/stream_1x4/run_omnilearned_proxy_pretrain_1x4_submission.sh`
   → `<proxy>_10000/omnilearned_proxy_pretrain_1x4/run_0/best_model_.pt`.
   The array is `0-0` on purpose — only `run_0`'s checkpoint is consumed downstream.
2. **Finetune**: [scripts/stream_1x4/run_omnilearned_proxy_finetune_1x4_submission.sh](scripts/stream_1x4/run_omnilearned_proxy_finetune_1x4_submission.sh)
   takes `PROXY_DATASET` (short tag, e.g. `bounded_ball`) and `PRETRAIN_CKPT` (absolute
   path), copies the checkpoint into the finetune `run_dir` as
   `best_model_<pretrain_tag>.pt` (`PRETRAIN_TAG` defaults to `PROXY_DATASET`), then
   `sbatch`es the runner, which calls `omnilearned train --fine-tune --pretrain-tag <tag>`.
3. Every omnilearned-family model at a given `(target_columns, dataset_size)` trains on
   **identical** real-stream data — only the initial checkpoint differs — so the data dir
   is generated **once** into `<target_columns>_<size>/data/` by
   [scripts/prepare_data_omnilearned.sh](scripts/prepare_data_omnilearned.sh) and every
   `_1x4` runner reads it via `--path ../../data`.

**Adding a new proxy** touches: [src/prepare_data.py](src/prepare_data.py) (synthesis
branch + argparse choices), [src/proxy_gallery.py](src/proxy_gallery.py) (visualisation +
`_PROXY_DISPLAY_NAMES` + `_TARGET_DESCRIPTIONS`),
[scripts/prepare_data_proxy_pretrain.sh](scripts/prepare_data_proxy_pretrain.sh),
[scripts/prepare_data_omnilearned.sh](scripts/prepare_data_omnilearned.sh),
[scripts/omnilearned_summary.sh](scripts/omnilearned_summary.sh),
[scripts/compute_bootstrap_shared_test_all.sh](scripts/compute_bootstrap_shared_test_all.sh),
[scripts/eval_shared_test10000.sh](scripts/eval_shared_test10000.sh) and
[evaluation/results_3d_shared_test10000.sh](evaluation/results_3d_shared_test10000.sh).

**Forgetting any of the `target_models` lists is silent** — the model is simply skipped
with "no valid results found", or its bootstrap column never gets computed. Model folder
names are used for path resolution and must **never** be reused.

## Submitting jobs

Where a script's `sbatch` defaults live:

| script | where the defaults are |
|---|---|
| `scripts/run_baseline_submission.sh`, `scripts/run_baseline_256_submission.sh` | [scripts/submit_training_variant.sh](scripts/submit_training_variant.sh) |
| everything under `scripts/stream_1x4/` | the wrapper script itself |

The `stream_1x4/` wrappers interleave pre-submission logic (checkpoint copying,
shared-data existence checks, per-size time estimates) with the `sbatch` call, so they
cannot delegate to the shared helper. Override at call time with `SBATCH_QOS`,
`SBATCH_TIME`, `SBATCH_ARRAY`, or edit the runner's own `#SBATCH` header for a permanent
change. They resolve their sibling runner script via `BASH_SOURCE`, so they work from any
working directory.

For sizes 300 and 1000 these wrappers **auto-tighten `--time` to the 30-minute mark but
deliberately keep `--qos=regular`** — debug QOS caps concurrency at ~3 jobs per user,
which is worse than waiting. Size 10000 keeps the runner's own `#SBATCH --time`:
**3h** for scratch / jet-finetune / proxy-finetune / omnicosmos-finetune, **5h** for
proxy pretrain.

Useful patterns:

- **Quick smoke test**: `SBATCH_QOS=debug SBATCH_TIME=00:30:00 SBATCH_ARRAY=0-0 bash <runner>`.
- **Resubmit only failed array tasks**: `SBATCH_ARRAY="<idx[,idx...]>" bash <runner>` —
  the per-task dir is `run_$((BASE_IDX + SLURM_ARRAY_TASK_ID))`, so a resubmit overwrites
  in place.

## Evaluation flow

1. Each `_1x4` runner already calls [src/omnilearned_summary.py](src/omnilearned_summary.py)
   itself once training finishes, so `run_<idx>/results/loss_history.csv` exists without
   any manual step. [scripts/omnilearned_summary.sh](scripts/omnilearned_summary.sh) is a
   **backfill / regeneration** tool — use it for older runs, or to rebuild the per-run
   plots after changing plotting code. It merges `training_.json` +
   `outputs__streams_0.npz` into `results/loss_history.csv`.
2. [scripts/eval_shared_test10000.sh](scripts/eval_shared_test10000.sh) re-evaluates every
   checkpoint on the shared 2,000-stream test set (finetuning otherwise scores on each
   run's own, smaller and noisier, split). It writes **`shared_test10000/`** — both
   `loss_history.csv` and the per-stream predictions `outputs__streams_*.npz`.
   [scripts/eval_shared_test10000_nonML.sh](scripts/eval_shared_test10000_nonML.sh) does
   the same for the two training-free baselines (mean prediction, eigenvalue regression).
3. [scripts/compute_bootstrap_shared_test_all.sh](scripts/compute_bootstrap_shared_test_all.sh)
   reads those `.npz` predictions and writes a **separate**
   `shared_test10000_bootstrap/loss_history.csv` carrying the `Test_Loss_Bootstrap_Std`
   column (1,000 resamples of the test set), for `run_0/1/2`.
   Mind the two directories: step 2 produces `shared_test10000/`, step 3 produces
   `shared_test10000_bootstrap/`, and **step 4 reads only the latter** — so skipping
   step 3 makes the figure silently find nothing.
4. [evaluation/results_3d_shared_test10000.sh](evaluation/results_3d_shared_test10000.sh)
   feeds a list of `target_columns:dataset_size:target_model[:display_name]` strings to
   [src/build_results_table.py](src/build_results_table.py), which finds the best-val
   epoch, reads the test loss at that epoch, aggregates across seeds, and emits the CSV,
   the scatter plot and the LaTeX table.

The optional 4th colon-delimited field is a **display-name override**: paths still resolve
via the on-disk folder name, while the CSV `Model` column and the plot legend use the
override.

## Figure generation

Defaults reproduce the paper with zero flags. Env-var toggles on
`evaluation/results_3d_shared_test10000.sh`:

| var | default | effect |
|---|---|---|
| `METRIC` | `r2` | `mse` for the raw-loss view |
| `NESTED_ERROR` | `true` | two error bars: outer = stat ⊕ syst, inner band = mean bootstrap std |
| `BOOTSTRAP` | `true` | implied by `NESTED_ERROR` |
| `R2_YMIN` | `0` | y-axis floor; a non-zero value adds a `_ymin08`-style filename suffix, so the zoomed figure never overwrites the main one |
| `LATEX_TABLE` | `true` | set `false` on a second pass that only changes the zoom — the table does not depend on the y-axis crop |

Two conventions inside [src/build_results_table.py](src/build_results_table.py):

- `_PRETTY_MODEL_NAMES` affects **plot legends and the LaTeX table only**. The CSV
  `Model` column keeps the raw display names so it stays parseable by exact name.
- `generate_scatter_plot` treats the x-axis as **categorical**: the three dataset sizes
  are drawn evenly spaced even though 180→600 is 3.3× and 600→6,000 is 10×. Vertical
  gridlines are off, the regions get alternating shading and a boundary line, tick marks
  are removed and the labels read `N = 180`. Do not "fix" this into a log axis — the even
  spacing is deliberate and the styling exists to stop it being misread as a number line.

`evaluation/reproduce_main_figure.ipynb` produces the same main figure, zoom figure and
table **without any raw data** — it reads the tracked per-run CSVs and the sidecar
`evaluation/prog_mass_labels_10000_test.npy` for `Var(y_test)`. The shell script instead
needs `raw_data/prog_mass_reg_dataset_10000.h5`. Both paths must stay byte-identical in
their outputs; if you change one, re-run the other and diff.

[src/proxy_gallery.py](src/proxy_gallery.py) writes all 16 gallery files with
`--proxy_type all --input_raw raw_data/prog_mass_reg_dataset_10000.h5`. Vocabulary there
is **"target"**, never "label" — this is regression, and the repo (`target_columns`,
`--target_model`) and the paper both use "target".

After changing anything that feeds a figure, regenerate it and **look at the rendered
output**, not just the code. Bugs that were invisible in a diff and obvious in the image
include: washed-out 3D point clouds (a matplotlib minor-version default), panel titles
colliding after a string got longer, and vertical gridlines running through the middle of
the very clusters they labelled.

## README command-block conventions

`[...]` marks an optional flag — it does **not** by itself mean "this is the default".
State the actual default explicitly, either inline as `[--flag (X by default)]` or in a
nearby sentence/table. Bare boolean toggles (`[--skip_plot]`) need no default note.
Do not add historical or bug-fix narration ("this used to default to X") to the README —
that belongs in commit messages.
