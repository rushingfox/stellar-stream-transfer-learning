# AGENTS.md

Instructions for AI coding agents working in this repository.
[README.md](README.md) is the source of truth for end-to-end reproduction commands; this
file documents architecture and conventions that the README does not spell out.

## What this repo is

Supervised regression of the **log progenitor mass** (`Log_Prog_Mass`) of synthetic
stellar streams, used to compare **transfer-learning strategies** — physics-domain
pretraining (jets, cosmology) and synthetic geometric-proxy pretraining — against
from-scratch baselines, at four dataset sizes. Each stream is a point cloud of 4,000
particles in 3D Cartesian coordinates.

Datasets are split train:val:test = 3:1:1, so the four sizes (40, 100, 300 and 1,000
streams) give **24 / 60 / 180 / 600** training streams. Every model is finally scored on
one **shared 2,000-stream test set** taken from the 10,000-stream dataset, not on its own
split.

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
- **JupyterHub sessions** are Slurm jobs too. An exclusive GPU node (`gpu_jupyter`
  QOS, 4 A100s) lasts at most **6h** and only one can run per user at a time; a shared
  GPU node (`gpu_shared_jupyter`, 1 GPU) can run alongside it. A 4-GPU OmniLearned run fits
  in the exclusive one (see the interactive mode under Two model families), and the
  single-process `eval_shared_test10000.sh` needs only the shared one.
- `--qos=shared` GPU jobs occasionally fail with `srun: error: Unable to create step for
  job <id>: Error generating job credential` → infinite retry → wall-time kill. This is a
  NERSC scheduler flake, not a code bug. Resubmit only the stuck array indices via
  `SBATCH_ARRAY="<indices>"`.

## Folder convention

Experiment results live at `<task>_<target_columns>_<dataset_size>/<model>/run_<seed_idx>/`,
e.g. `prog_mass_3d_x_y_z_300/omnilearned_scratch/run_0/`.

- `task` is `prog_mass` (the paper's progenitor-mass target) or `subhalo`, set with
  `TASK` (default `prog_mass`, see [scripts/task_config.sh](scripts/task_config.sh)).
- `dataset_size` ∈ `{40, 100, 300, 1000}` — the four sizes the paper reports. Size
  10000 exists only for its data: its test split is the shared test set, and its train
  split feeds the eigenvalue proxy. These five are all
  [src/exp_paths.py](src/exp_paths.py)'s `VALID_DATASET_SIZES` accepts.
- `target_columns` for the real-stream task is `3d_x_y_z`, the only input. The proxy
  pretraining datasets get their own top-level dirs without a task prefix (see below).
- **Finetune data is shared per size**, in `<task>_<target_columns>_<size>/data/`: OmniLearned's
  `data/streams/{train,val,test}/` with `input_stats.npz` and `target_stats.npz`, plus
  `data/shared_test10000/` (both from
  [scripts/prepare_data_omnilearned.sh](scripts/prepare_data_omnilearned.sh); see
  Normalization) and
  DeepSets' `data/deepsets_dataset.h5` (from
  [scripts/prepare_data_deepsets.sh](scripts/prepare_data_deepsets.sh)). Every model of a
  family at that size reads the same copy; only the initial weights differ.
- Path helpers in [src/exp_paths.py](src/exp_paths.py) (`dataset_key`,
  `get_loss_history_csv_path`, …) are the canonical way to construct paths — use them
  rather than f-strings.
- `.gitignore` ignores every experiment folder (`3d_*/ prog_mass_*/ subhalo_*/`)
  wholesale: data, checkpoints and per-run results are not tracked. They are
  distributed as a separate archive with the same folder layout, to be unpacked at the
  repo root.
- Tracked top-level dirs: `src/`, `scripts/`, `evaluation/`, `visualization/`,
  `proxy_gallery/`. Raw data (`raw_data*/`) and all
  model weights (`*.pt`, `*.pth`, `*.h5`) are **never** committed. The one exception is
  `raw_data/FIRE_LatteELVIS_SMHM`, the 5-line stellar-mass–halo-mass table that
  [src/raw_data_generation.py](src/raw_data_generation.py) reads from its working
  directory (`raw_data/`); it is the FIRE Latte/ELVIS fit in Fig. 2(a) of Sales,
  Wetzel & Fattahi 2022 (arXiv:2206.05295). Since `raw_data*/` is ignored, it has to be
  added with `git add -f`; once tracked, git follows it normally.

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

The six geometries come from [src/prepare_data.py](src/prepare_data.py) via
[scripts/prepare_data_proxy_pretrain.sh](scripts/prepare_data_proxy_pretrain.sh). Each is
drawn from its own numpy Generator, seeded with `prepare_data.py --seed` (0 by default) and
the crc32 of its synthesis tag, so regenerating a proxy reproduces it exactly and no two
proxies share random numbers. Three
of them are synthesised with **3 targets** (`3d_proxy_box`, `3d_proxy_hard_ellipsoid`,
`3d_proxy_ellipsoid`), and three `src/make_*_from_*.py` slicers cut each down to its
longest axis. The slicers infer the repo root from
`Path(__file__).resolve().parent.parent` and read from
`<parent>/omnilearned_proxy_pretrain/data/streams`, where `prepare_data_proxy_pretrain.sh`
writes.

The eigen proxy is built separately, by
[src/make_eigen_lambda_from_streams.py](src/make_eigen_lambda_from_streams.py)
(`--index K`, K = 1, 2, 3 → `3d_x_y_z_eigenvalues_lambda{K}_10000`), directly from the
shared N=10000 finetune data, so the finetune data has to exist first. The
script copies the finetune inputs as they are (already normalized with the N=10000 train
split, which is the eigen proxy's own train split) and computes the √λ_K target, in kpc,
from the **raw** coordinates in `raw_data/prog_mass_reg_dataset_10000.h5`.
Note what that implies: the eigen proxy's inputs are
the **same** 6,000 / 2,000 / 2,000 streams as the N=10000 finetune split, only relabelled.
There is no test leakage (pretraining uses the train split only), but for the 6,000-stream
setting it is not independent data. λ₂ / λ₃ are whitelisted in the pretrain submitter but
are not among the paper's seven.

Each proxy z-scores its labels using **train-split statistics**, persisted to
`streams/<synthesis_tag>_stats.npz` so val/test reuse the same scaler (the slicers also
write the sliced stats under the final name, e.g. `gaussian_ellipsoid_sigmax_stats.npz`).
Its inputs are normalized with its own train split as well (`streams/input_stats.npz`, see
Normalization). `prepare_data.py` raises `FileNotFoundError` if the train split was not
processed first.

Rotation augmentation (a random 3D rotation inside the synthesis branch) is applied to
`ellipsoid`, `cube`, `hard_ellipsoid` and `box`. It is **not** applied to `bounded_ball`
or `gaussian_ball` (isotropic, so it would be a no-op), nor to the eigen proxy (real
streams, already at arbitrary orientations).

Proxy point clouds read `num_particles` from the raw HDF5 (4,000) rather than hardcoding
it. Splits are `int(n*0.6)` / `int(n*0.8)`, i.e. 3:1:1.

## Normalization

Every model, OmniLearned and DeepSets alike, sees normalized inputs and a z-scored target,
and every statistic comes from **the train split of the dataset being trained on** —
each proxy its own, each downstream size its own. Val and test reuse the train numbers.

- **Inputs**: every input is a point cloud of x, y, z coordinates (the only input
  columns, `3d_x_y_z` for the real streams). They are z-scored with **one** shared
  scale: each axis is shifted by its own train mean (over every particle of every train
  point cloud), and all three are divided by the std of the centred x, y and z values
  pooled together, i.e. the square root of the average of the three axis variances. One
  shared scale keeps shapes undistorted (a rotated box stays a box, which per-axis
  scales would shear). The shift moves the origin off the Galactic centre, so the two
  OmniLearned `astro` pair features that are angles seen from the origin (`get_dot`,
  `get_dot_diff`) are measured from the train mean position instead; this was accepted
  in exchange for a standard z-score (the pair distance `get_d3` is unaffected, and the
  angles do not depend on the scale). The statistics are dataset-wide,
  never per point cloud, so a cloud keeps its overall size and its position relative to
  the others (a proxy's target *is* its size). All splits are transformed, test included,
  because a model has to see test inputs on the scale it was trained on. One function
  computes the statistics, [src/exp_paths.py](src/exp_paths.py) `compute_input_stats`,
  which returns a per-axis mean and std (all that one scale) applied as
  `(x - mean) / std`. OmniLearned data stores the
  transformed inputs and the numbers in `data/streams/input_stats.npz`
  ([src/prepare_data.py](src/prepare_data.py), `standardize_input`); DeepSets keeps raw
  coordinates in `deepsets_dataset.h5` and applies the same transform on the fly
  ([src/dataset.py](src/dataset.py), `train_input_stats`).
- **Target**: the downstream target is z-scored in train and val
  (`data/streams/target_stats.npz`); the test split keeps the raw target, and predictions
  are mapped back with `pred * std + mean` before any loss or R² is computed, so every
  reported number is in raw units. Proxy targets are z-scored in all three splits (see
  The seven proxies).
- **Shared test set**: one set of 2,000 test streams scores every size, but each size's
  models must see its inputs transformed with that size's own input stats. So
  [scripts/prepare_data_omnilearned.sh](scripts/prepare_data_omnilearned.sh) writes, for
  every size, `data/shared_test10000/streams/test/`: the size-10000 test split with inputs
  normalized with that size's stats and the target raw.
  [scripts/eval_shared_test10000.sh](scripts/eval_shared_test10000.sh) scores each run on
  its own size's copy, and DeepSets transforms the size-10000 `deepsets_dataset.h5` test
  split on the fly.
- The released jet and OmniCosmos checkpoints are used as they are; only our own proxy
  pretraining follows the rules above.
- The training-free baselines (mean prediction, eigenvalue regression) read raw data and
  are unaffected.

## Two model families

1. **In-house DeepSets baseline** ([src/main.py](src/main.py) + [src/model.py](src/model.py)
   + [src/dataset.py](src/dataset.py)): MLP encoder → mean+max+std pooling, hidden dim
   1024, in the `deepsets` model folder. Submitted through
   `scripts/run_deepsets_submission.sh`, which calls
   [scripts/submit_training_variant.sh](scripts/submit_training_variant.sh); `MODEL_SUFFIX`
   is appended to the folder and job name, `LR` sets the learning rate (default 1e-3).
   Runs on 1 GPU (`--qos=shared`). Three behaviours to know:
   - **No silent resume.** If `run_N/checkpoints/checkpoint_latest.pth` already exists,
     [scripts/run_deepsets_train_common.sh](scripts/run_deepsets_train_common.sh) refuses
     to start unless `RESUME=1` (continue it, e.g. after a timeout) or `FORCE=1` (delete
     `run_N` and retrain). The epoch count comes from `EPOCHS` or `EPOCH_NUM` (default 100).
   - [src/eval_deepsets_shared_test.py](src/eval_deepsets_shared_test.py) reads the
     hidden size from the checkpoint (`encoder.0.weight` is `[hidden, input_dim]`), not
     from the folder name.
   - The DeepSets scripts take only `3d_x_y_z` (`TARGET_COLUMNS` anything else is an error).
2. **OmniLearned transformer** (external CLI): every `omnilearned_*` and `omnicosmos_*`
   model. All of these live in [scripts/omnilearned/](scripts/omnilearned/) and launch
   `torchrun --nproc_per_node=4 $(which omnilearned) train …` on 1 node × 4 GPUs.
   Pretrain checkpoints come either from the OmniLearned/OmniCosmos releases (placed in
   `omnicosmos_checkpoints/`, gitignored) or from local proxy pretraining. Both the
   submitters and the runners source
   [scripts/omnilearned/check_data.sh](scripts/omnilearned/check_data.sh) and refuse to
   start unless the data is complete and in the normalized form (see Normalization), so a
   missed regeneration step stops a job instead of training on stale data.

   The launcher is `torchrun`, not `srun`, so every OmniLearned runner also runs outside
   sbatch, e.g. in a JupyterHub exclusive 4-GPU session. The runners create and enter
   `run_<idx>/` only when `SLURM_ARRAY_TASK_ID` is set; otherwise they train in the
   current directory. For that interactive mode, skip the submission wrapper, `cd` into
   the `run_<idx>/` yourself and call the runner with the same positional arguments the
   wrapper would pass (by absolute path is fine; it finds the repo root via
   `BASH_SOURCE`). The `#SBATCH --output` lines do not apply there, so nothing goes to
   `logs/` unless you redirect it. Cancel any queued sbatch job for the same `run_<idx>`
   first, or it will retrain over the result when it starts.

Both families share the same per-run output convention: each `run_<idx>/` ends up with a
`loss_history.csv` — written by [src/omnilearned_summary.py](src/omnilearned_summary.py)
for the OmniLearned family, directly by `main.py` for baselines. **Evaluation reads only
that CSV.**

These models are regression-only. Classification support (`--mode`, accuracy, confusion
matrices) was removed; do not reintroduce it.

## Model sizes

| model | backbone | head | total |
|---|---|---|---|
| `deepsets` (DeepSets, hidden 1024) | 1,055,744 | 196,737 | 1,252,481 |
| OmniLearned-s, every variant | 1,389,477 | 1,315,334 | 2,704,811 |
| OmniLearned-m | 30,475,825 | 20,989,958 | 51,465,783 |

Backbone is DeepSets' `encoder` / OmniLearned's `body`; head is DeepSets' `regressor` /
OmniLearned's `classifier_head`. **Both parts are used for every prediction**: in
regression mode OmniLearned computes `classifier(body(x))`, and its own training log
prints the total (`Total params: 2.70M`). A count of `body` alone understates the model
by half. The checkpoint's `ema_body` is a running average of `body`, not extra weights,
and is never counted. Which column the paper reports as "model size" is not settled.
`deepsets` is 46% of OmniLearned-s by total and 76% by backbone.

[src/count_params.py](src/count_params.py) reproduces the table from the checkpoints
(terminal only): it counts every `run_0` checkpoint of each family at the four sizes
and warns if they disagree. It reports only the OmniLearned sizes listed in its
`OMNILEARNED_SIZES`, currently just small; the medium row above was counted from a model
built with the same settings, and matches the `Total params: 51.47M` that medium
training logs print.
`--inspect PATH [PATH]` lists every component of one or two checkpoint files.

Every OmniLearned variant — from scratch, jet-pretrained, cosmology-pretrained,
proxy-pretrained — trains the same architecture, so differences between them come only
from initialisation.

DeepSets parameters are h² + 199h + 129 for 3 input features, which reproduces the
DeepSets row. The OmniLearned counts are from our trained stream checkpoints. The
pretrain checkpoints differ, and finetuning loads every tensor whose name and shape
match except the final `out.*` layer (513 weights), which is always re-initialised:

- The seven proxies are trained on our streams' architecture, so they match exactly.
- CAMELS / Quijote match except `out` (2 targets instead of 1).
- The released jet `best_model_pretrain_s.pt` has a 1,494,320 backbone: it has 15 extra
  jet-only modules (time, pid and extra-info embeddings, global interaction,
  `MPFourier`), 4 input features instead of 3, and a 210-way `out`. Finetuning drops the
  extra modules and re-initialises the three input weights whose shapes differ
  (`embed.mlp.fc1.weight`, `embed.norm.weight`, `local_physics.mlp.fc1.weight`; their
  biases still load).

That leaves 99.98% of the stream model's parameters taken from the pretrain checkpoint
for the proxies and OmniCosmos, and 99.90% for jets. Each finetune log lists what was
skipped (`Skipping …`).

## Proxy pretrain → finetune wiring

The most coupled piece in the repo:

1. **Pretrain**: `TARGET_COLUMN=<proxy> bash scripts/omnilearned/run_omnilearned_proxy_pretrain_submission.sh`
   → `<proxy>_10000/omnilearned_proxy_pretrain/run_0/best_model_.pt`.
   The array is `0-0` on purpose — only `run_0`'s checkpoint is consumed downstream.
2. **Finetune**: [scripts/omnilearned/run_omnilearned_proxy_finetune_submission.sh](scripts/omnilearned/run_omnilearned_proxy_finetune_submission.sh)
   takes `PROXY_DATASET` (short tag, e.g. `bounded_ball`) and `PRETRAIN_CKPT` (absolute
   path), copies the checkpoint into the finetune model folder
   (`omnilearned_proxy_<tag>_finetune/`) as `best_model_<pretrain_tag>.pt`
   (`PRETRAIN_TAG` defaults to `PROXY_DATASET`), then `sbatch`es the runner. The runner
   copies that file into its `run_<idx>/` every time it starts, overwriting any older
   copy, and calls `omnilearned train --fine-tune --pretrain-tag <tag>`. The OmniCosmos
   finetunes use the same runner, with the checkpoint taken from `omnicosmos_checkpoints/`.
3. Every omnilearned-family model at a given `(target_columns, dataset_size)` trains on
   **identical** real-stream data — only the initial checkpoint differs — so the data dir
   is generated **once** into `<task>_<target_columns>_<size>/data/` by
   [scripts/prepare_data_omnilearned.sh](scripts/prepare_data_omnilearned.sh) and every
   OmniLearned runner reads it via `--path ../../data`.

**Adding a new proxy** touches: [src/prepare_data.py](src/prepare_data.py) (synthesis
branch + argparse choices), [src/proxy_gallery.py](src/proxy_gallery.py) (visualisation +
`_PROXY_DISPLAY_NAMES` + `_TARGET_DESCRIPTIONS`),
[scripts/prepare_data_proxy_pretrain.sh](scripts/prepare_data_proxy_pretrain.sh),
[scripts/prepare_data_omnilearned.sh](scripts/prepare_data_omnilearned.sh),
[scripts/omnilearned_summary.sh](scripts/omnilearned_summary.sh),
[scripts/compute_bootstrap_shared_test_all.sh](scripts/compute_bootstrap_shared_test_all.sh),
[scripts/eval_shared_test10000.sh](scripts/eval_shared_test10000.sh),
[evaluation/reproduce_main_figure.ipynb](evaluation/reproduce_main_figure.ipynb) (model list)
and [src/print_r2.py](src/print_r2.py) (`_TABLE_MODELS`).

**Forgetting any of the `target_models` lists is silent** — the model is simply skipped
with "no valid results found", or its bootstrap column never gets computed. Model folder
names are used for path resolution and must **never** be reused.

## Submitting jobs

Where a script's `sbatch` defaults live:

| script | where the defaults are |
|---|---|
| `scripts/run_deepsets_submission.sh` | [scripts/submit_training_variant.sh](scripts/submit_training_variant.sh) |
| everything under `scripts/omnilearned/` | the wrapper script itself |

The `scripts/omnilearned/` wrappers interleave pre-submission logic (checkpoint copying,
shared-data existence checks, per-size time estimates) with the `sbatch` call, so they
cannot delegate to the shared helper. Override at call time with `SBATCH_QOS`,
`SBATCH_TIME`, `SBATCH_ARRAY`, or edit the runner's own `#SBATCH` header for a permanent
change. They resolve their sibling runner script via `BASH_SOURCE`, so they work from any
working directory.

Every OmniLearned training defaults to **100 epochs** (`EPOCH_NUM`; `TOTAL_EPOCHS` for
the proxy pretrain). For the downstream sizes the wrappers **set `--time` per size but
deliberately keep `--qos=regular`** — debug QOS caps concurrency at ~3 jobs per user,
which is worse than waiting: 30 min for 40 and 100, 1h for 300, 1h30 for 1000. Any other
size keeps the runner's own `#SBATCH --time` of **5h**, as does the proxy pretrain. The
DeepSets wrapper sets 30 min for 40, 100 and 300, 1h for 1000 and 10h for 10000.

Those limits include the shared-test evaluation each job runs at its end and were sized
for 100 epochs; runtime grows linearly with `EPOCH_NUM`, so pass `SBATCH_TIME` explicitly
for longer runs. Measured for 100 epochs of `MODEL_SIZE=small` on 4 GPUs: about 30 min of
training at N=1000 and about 13 min at N=300, plus a few minutes of evaluation; a proxy
pretrain takes about 3h50m. The per-size limits apply only to `MODEL_SIZE=small`; larger
sizes keep the runner's own `--time`.

`MODEL_SIZE=medium` is about 2–2.4× slower per epoch than small (about 5.5h per 50
epochs at N=10000, so a run there does not fit a 6h Jupyter session) and **needs the
80 GB A100 nodes**: on the 40 GB ones it runs out of GPU memory in the first epoch. Pass
`SBATCH_CONSTRAINT="gpu&hbm80g"`, which sbatch reads in place of the runners'
`--constraint=gpu`, and in a Jupyter session check
`nvidia-smi --query-gpu=memory.total --format=csv` first.

Useful patterns:

- **Quick smoke test**: `SBATCH_QOS=debug SBATCH_TIME=00:30:00 SBATCH_ARRAY=0-0 bash <runner>`.
- **Resubmit only failed array tasks**: `SBATCH_ARRAY="<idx[,idx...]>" bash <runner>` —
  the per-task dir is `run_$((BASE_IDX + SLURM_ARRAY_TASK_ID))`, so a resubmit overwrites
  in place (DeepSets needs `RESUME=1` or `FORCE=1` for that, see above).
- **Run a variant without touching existing results**: every `scripts/omnilearned/`
  scratch and finetune submitter takes `MODEL_SUFFIX` (default empty) and `EPOCH_NUM`
  (default 100); a suffix puts the runs in a new sibling folder next to the existing one.
  The scratch learning-rate sweep uses this: `LR=1e-4 MODEL_SUFFIX=_lr1e-4` →
  `omnilearned_scratch_lr1e-4`, the folder name the figure notebook expects. DeepSets has
  the same kind of sweep, through `scripts/run_deepsets_submission.sh` with the same two
  variables (`deepsets_lr5e-4`, …). Both sweeps, with their lrs, are declared in
  `LR_SWEEPS` in [src/build_results_table.py](src/build_results_table.py). At each size
  the figure shows each swept model at the lr with the lowest mean best-val loss once
  every lr of the sweep has all its seeds; until then at the default lr if that has all
  its seeds, and not at all otherwise (each such size is printed). The table lists every
  lr, with the selected one marked †.
- **Model size**: the scratch and jet-finetune runners take `MODEL_SIZE` = `small`
  (default) | `medium` | `large`, and the jet finetune picks the matching checkpoint
  (`pretrain_s` / `_m` / `_l`). The submitters refuse a non-small size without an
  explicit `MODEL_SUFFIX`, so it can never land in a small model's folder, e.g.
  `MODEL_SUFFIX=_m` → `omnilearned_scratch_m`. The proxy and OmniCosmos runners stay small: their
  pretrain checkpoints only exist in that size.

Known flake: if `run_N` has no `best_model_pretrain_<s|m|l>.pt`, the jet-pretrained
finetune lets OmniLearned download it, and all four ranks download it at once.
Occasionally one rank reads the half-written file and the job dies within about two
minutes (`Cannot use weights_only=True with files saved in the legacy .tar format`, or
just a rank exiting with code 1). The runner now avoids this by copying
`omnicosmos_checkpoints/best_model_pretrain_<s|m|l>.pt` into `run_N` first when that
local copy exists, so keep it there. OmniLearned only downloads when the file is missing
from `run_N`, so putting it in by hand also protects jobs that were submitted before
this change (Slurm runs the copy of the script it stored at submission). Resubmitting a
failed index also works, since the complete file is then already there. Proxy and
OmniCosmos finetunes are unaffected, because their submitters copy the checkpoint in
before `sbatch`.

## Evaluation flow

1. Each training job (every OmniLearned downstream runner and the DeepSets runner) does
   all of its own evaluation when training finishes:
   [src/omnilearned_summary.py](src/omnilearned_summary.py) (OmniLearned) or `main.py`
   (DeepSets) writes `run_<idx>/results/loss_history.csv` on the run's own split; then
   [scripts/eval_shared_test10000.sh](scripts/eval_shared_test10000.sh) (called with
   `RUN_DIR=<run dir> FORCE=1`) scores the best-val checkpoint on the shared 2,000-stream
   test set, and [src/compute_bootstrap_shared_test.py](src/compute_bootstrap_shared_test.py)
   (`--run_dir`) adds the bootstrap error bar. Everything from the shared-test evaluation
   lands in `run_<idx>/shared_test10000/`: the predictions (`outputs__streams_0.npz` for
   OmniLearned, `predictions.npz` for DeepSets), `loss_history.csv` and `bootstrap.csv`.
   A failure there is reported as a warning and does not fail the job.
2. The manual scripts are for backfilling or redoing that step.
   [scripts/eval_shared_test10000.sh](scripts/eval_shared_test10000.sh) takes
   `TARGET_MODELS`, `RUN_IDXS` and `DATASET_SIZES` (default 40 100 300 1000), reads
   `--size` from each run's `hparams.json`, refuses runs whose data is not in the
   normalized form, and skips runs that already have a `shared_test10000/loss_history.csv`
   unless `FORCE=1`.
   [scripts/eval_shared_test10000_nonML.sh](scripts/eval_shared_test10000_nonML.sh) does
   the shared-test evaluation of the two training-free baselines (mean prediction,
   eigenvalue regression), which nothing does automatically.
   [scripts/compute_bootstrap_shared_test_all.sh](scripts/compute_bootstrap_shared_test_all.sh)
   computes `bootstrap.csv` for every model in its list, only `run_0` unless `RUN_IDXS` is
   set (`RUN_IDXS="0 1 2"` for the figure's nested error bars); for any other folder call
   [src/compute_bootstrap_shared_test.py](src/compute_bootstrap_shared_test.py) directly
   (`--target_model <folder> --run_idxs ...` or `--run_dir <run dir>`). The resampling
   seed is fixed, so reruns give identical numbers.
   **Only evaluate runs that have finished training.** `eval_shared_test10000.sh` treats
   any run with a `best_model_.pt` as trained, but that file appears at the first best
   epoch, so a run still in progress gets scored on a half-trained model, and every later
   call then skips it as already evaluated (`FORCE=1` redoes it).
3. `bootstrap.csv` is a single row: `Test_Loss` and `Test_Loss_Bootstrap_Std` (1,000
   resamples of the test set) are computed from the predictions, while `Epoch` and
   `Val_Loss` are copied bit for bit from the one row of `shared_test10000/loss_history.csv`
   that carries a `Test_Loss`, i.e. the best-val epoch at which `best_model_.pt` was saved.
   So the `Best Val MSE` column of the step-4 CSV is the real validation loss on each run's
   own split. **Step 4 reads only `bootstrap.csv`**, so a run without one is silently left
   out of the figure.
4. [evaluation/reproduce_main_figure.ipynb](evaluation/reproduce_main_figure.ipynb)
   feeds a list of `(target_columns, dataset_size, target_model, display_name)` tuples to
   `build_table` in [src/build_results_table.py](src/build_results_table.py), which finds
   the best-val epoch, reads the test loss at that epoch, aggregates across seeds, and
   emits the CSV, the scatter plot and the LaTeX table.

The 4th field is a **display name**: paths still resolve via the on-disk folder name,
while the CSV `Model` column and the plot legend use the display name.

To look at specific runs without touching any figure, use
[src/print_r2.py](src/print_r2.py): `python src/print_r2.py SIZE:MODEL[:RUNS] ...` prints
each run's R² and a summary in the appendix-table form, and writes nothing. Without
`:RUNS` it reads the figure's runs at that size, and `--table` uses the same runs; both
come from its `_FIGURE_RUNS`, which must match the notebook's `indices`.

## Figure generation

[evaluation/reproduce_main_figure.ipynb](evaluation/reproduce_main_figure.ipynb) is the one
place the paper's figures and table are made. Run top to bottom, it writes into
`evaluation/` the main figure, the appendix LaTeX table and the CSV. It needs **no raw data**: it reads each run's `shared_test10000/bootstrap.csv` in the
result folders (from the result archive) and the sidecar
`evaluation/prog_mass_labels_10000_test.npy` for `Var(y_test)`. Its settings are the
paper's: R², seeds 0–9 at N = 24 and 60 and 0–2 at N = 180 and 600 (`indices`, a
{dataset size: runs} dict; the training-free `eigenvalue_regression` has only `run_0`),
a y-axis from 0 to 1 (`r2_ymin`, 0 by default), and nested error bars (`combine_errors=True`
with `error_column="Test_Loss_Bootstrap_Std"`: outer = seed s.e.m. ⊕ mean bootstrap std,
where the seed term is the sample std over seeds / √n and the bootstrap term is not
divided by √n, because every seed is scored on the same test set and its sampling error
does not average out; inner band = mean bootstrap std). `build_results_table.py` makes only these R² outputs and has no command
line; to look at numbers in the terminal, use [src/print_r2.py](src/print_r2.py).

Conventions inside [src/build_results_table.py](src/build_results_table.py):

- **`epochs` (the notebook passes 100, the training default) is a cutoff on the `Epoch`
  column**: rows above it are dropped before the best-val row is picked, and a run left
  with no row disappears from the figure without a warning. The bootstrap CSVs carry the
  real best-val epoch (see Evaluation flow, step 3), as does the full per-epoch
  `shared_test10000/loss_history.csv`. So for runs trained longer than 100 epochs, pass
  at least `EPOCH_NUM`, or every run whose best epoch is past 100 is dropped.
- Each category (baseline / physics / synthetic / eigen) has a fixed list of colors in
  `_CATEGORY_STYLE`, handed out in rank order to the members present, and too many
  members in one category raises `ValueError`. Because of that a model's color depends
  on which models the figure has (one more baseline ranked first shifts every baseline
  color), so colors must not be hard-coded per model. The optional models (OmniLearned-m,
  PCA λ₂ / λ₃) are the exception: they take a fixed shade of their base
  model's color from `_OPTIONAL_MODEL_COLORS` and sit next to it in rank, while the
  category swatches still go to the paper's models only. Adding an optional model
  therefore never recolors anything, and every figure without one is byte-identical to
  before. A new model needs a `_PRETTY_MODEL_NAMES` entry and a rank, plus either an
  `_OPTIONAL_MODEL_COLORS` entry or another swatch.

- `_PRETTY_MODEL_NAMES` affects **plot legends and the LaTeX table only**. The CSV
  `Model` column keeps the raw display names so it stays parseable by exact name.
- `generate_scatter_plot` treats the x-axis as **categorical**: the dataset sizes
  are drawn evenly spaced even though their ratios differ. Vertical
  gridlines are off, the regions get alternating shading and a boundary line, tick marks
  are removed and the labels read `N = 180`. Do not "fix" this into a log axis — the even
  spacing is deliberate and the styling exists to stop it being misread as a number line.
- Within each size the points are placed by mean R², highest on the left, across all
  categories, so each cluster reads as a ranking and a model's position differs between
  sizes; the marker shape still shows the category. The rank tables set only the legend
  and the LaTeX table order, and break ties in R².
- A model with mean R² below 0 (worse than predicting the test mean) is drawn as a hollow
  marker at 0, without error bar or band, so the y-axis can start at 0; the threshold is
  always 0, whatever `r2_ymin` is. Only the figure does this: the table and the CSV keep
  the real value. The key then gets a hollow-marker entry saying so.
- Both legends sit under the x-axis label, so the axes use the full figure width: the
  Models legend in three rows (one column per category for the paper's 15 models), and
  under it a grey key for the error bar, the bootstrap band, the dashed scratch line and
  the star. The key's title gives the seeds per size, counted from the results (the most
  seeds any model has at that size), so it follows `indices` without being told.

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
