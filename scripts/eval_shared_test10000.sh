#!/bin/bash
# Re-scores already-trained checkpoints against dataset_size=10000's test split
# instead of each run's own (smaller, noisier) one. Checkpoint-based models
# only -- average/eigenvalue_regression have no checkpoint to re-score.
# No SBATCH header: run directly on an interactive GPU node. Idempotent
# (skips combos whose output CSV already exists; FORCE=1 to redo).
#
# Env overrides: TASK, DATASET_SIZES, RUN_IDXS, TARGET_MODELS, EVAL_DATASET_SIZE, FORCE.

set -euo pipefail

module load conda
mamba activate stream_transfer_learning

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# TASK picks the runs (prog_mass_ or subhalo_<target_columns>_<size>/) and that task's shared test set.
source "${script_dir}/task_config.sh"

target_columns="3d_x_y_z"
eval_dataset_size=${EVAL_DATASET_SIZE:-10000}
# Everything from this evaluation goes to run_<idx>/shared_test<N>/: the
# predictions, loss_history.csv (plus the plots omnilearned_summary.py writes
# next to it) and, later, compute_bootstrap_shared_test.py's bootstrap.csv.
shared_dirname="shared_test${eval_dataset_size}"
shared_data_dir="${repo_root}/${task_dir_prefix}${target_columns}_${eval_dataset_size}/data"

if [[ -n "${DATASET_SIZES:-}" ]]; then
    read -r -a dataset_sizes <<< "${DATASET_SIZES}"
else
    dataset_sizes=(300 1000 10000)
fi

if [[ -n "${RUN_IDXS:-}" ]]; then
    read -r -a run_idxs <<< "${RUN_IDXS}"
else
    run_idxs=(0 1 2)
fi

omnilearned_models_default=(
    "omnilearned_scratch"
    "omnilearned_jet_finetune"
    "omnicosmos_camels_finetune"
    "omnicosmos_quijote_finetune"
    "omnilearned_proxy_cube_finetune"
    "omnilearned_proxy_bounded_ball_finetune"
    "omnilearned_proxy_gaussian_ball_finetune"
    "omnilearned_proxy_gaussian_ellipsoid_sigmax_finetune"
    "omnilearned_proxy_bounded_ellipsoid_a_finetune"
    "omnilearned_proxy_box_a_finetune"
    "omnilearned_proxy_eigen_lambda1_finetune"
)
deepsets_models_default=(
    "deepsets"
)

if [[ -n "${TARGET_MODELS:-}" ]]; then
    read -r -a all_models <<< "${TARGET_MODELS}"
    omnilearned_models=()
    deepsets_models=()
    for m in "${all_models[@]}"; do
        if [[ "$m" == deepsets* ]]; then
            deepsets_models+=("$m")
        else
            omnilearned_models+=("$m")
        fi
    done
else
    omnilearned_models=("${omnilearned_models_default[@]}")
    deepsets_models=("${deepsets_models_default[@]}")
fi

# Single process (no torchrun/srun) -- a one-off forward pass isn't worth
# 4-GPU DDP overhead. OmniLearned still needs these env vars set regardless.
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=$(( 15000 + ($$ % 15000) ))

# The model size the run was trained with, from its hparams.json; small if
# that file or field is missing.
model_size_of() {
    python -c 'import json, sys
try:
    print(json.load(open(sys.argv[1]))["omnilearned_params"]["model_size"]["value"])
except (OSError, KeyError, ValueError):
    print("small")' "$1/hparams.json"
}

eval_omnilearned() {
    local run_dir="$1"
    local out_csv="${run_dir}/${shared_dirname}/loss_history.csv"

    if [[ ! -f "${run_dir}/best_model_.pt" ]]; then
        echo "[skip] no checkpoint: ${run_dir}/best_model_.pt"
        return
    fi
    if [[ -f "${out_csv}" && -z "${FORCE:-}" ]]; then
        echo "[done] ${out_csv} already exists"
        return
    fi

    local model_size
    model_size="$(model_size_of "${run_dir}")"
    echo "[eval] ${run_dir}  ->  ${task_dir_prefix}${target_columns}_${eval_dataset_size} test split (size ${model_size})"
    rm -rf "${run_dir:?}/${shared_dirname}"
    mkdir -p "${run_dir}/${shared_dirname}"
    (
        cd "${run_dir}"
        omnilearned evaluate \
            -i ./ \
            -o "./${shared_dirname}/" \
            --save-tag "" \
            --dataset streams \
            --path "${shared_data_dir}" \
            --size "${model_size}" \
            --num-classes 1 \
            --num-feat 3 \
            --num-coord 3 \
            --mode regression \
            --batch 4 \
            --local-interaction \
            --interaction-type astro
    )
    # omnilearned_summary.py creates os.path.dirname(--output_csv) itself.
    python "${repo_root}/src/omnilearned_summary.py" \
        --json_path "${run_dir}/training_.json" \
        --npz_glob "${run_dir}/${shared_dirname}/outputs__streams_*.npz" \
        --output_csv "${out_csv}"
}

eval_deepsets() {
    local dataset_size="$1"
    local model="$2"
    local idx="$3"
    local run_dir="${repo_root}/${task_dir_prefix}${target_columns}_${dataset_size}/${model}/run_${idx}"
    local out_csv="${run_dir}/${shared_dirname}/loss_history.csv"

    if [[ -f "${out_csv}" && -z "${FORCE:-}" ]]; then
        echo "[done] ${out_csv} already exists"
        return
    fi

    echo "[eval] ${run_dir}  ->  ${task_dir_prefix}${target_columns}_${eval_dataset_size} test split"
    python "${repo_root}/src/eval_deepsets_shared_test.py" \
        --repo_root "${repo_root}" \
        --target_columns "${target_columns}" \
        --task "${task}" \
        --dataset_size "${dataset_size}" \
        --target_model "${model}" \
        --run_idx "${idx}" \
        --eval_dataset_size "${eval_dataset_size}"
}

for dataset_size in "${dataset_sizes[@]}"; do
    for model in "${omnilearned_models[@]}"; do
        for idx in "${run_idxs[@]}"; do
            eval_omnilearned "${repo_root}/${task_dir_prefix}${target_columns}_${dataset_size}/${model}/run_${idx}"
        done
    done
    for model in "${deepsets_models[@]}"; do
        for idx in "${run_idxs[@]}"; do
            eval_deepsets "${dataset_size}" "${model}" "${idx}"
        done
    done
done

echo "All done."
