#!/bin/bash
# Re-scores already-trained checkpoints against dataset_size=10000's test split
# instead of each run's own (smaller, noisier) one. Checkpoint-based models
# only -- average/eigenvalue_regression have no checkpoint to re-score.
# No SBATCH header: run directly on an interactive GPU node. Idempotent
# (skips combos whose output CSV already exists; FORCE=1 to redo).
#
# Env overrides: DATASET_SIZES, RUN_IDXS, TARGET_MODELS, EVAL_DATASET_SIZE, FORCE.

set -euo pipefail

module load conda
mamba activate stream_transfer_learning

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

target_columns="3d_x_y_z"
eval_dataset_size=${EVAL_DATASET_SIZE:-10000}
# Nested subdir, not a flat sibling filename: omnilearned_summary.py writes
# plots alongside --output_csv, which would clobber the run's own plots.
output_csv_name="shared_test${eval_dataset_size}/loss_history.csv"
shared_data_dir="${repo_root}/${target_columns}_${eval_dataset_size}/data"

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
    "omnilearned_scratch_1x4"
    "omnilearned_finetune_1x4"
    "omnicosmos_camels_finetune_1x4"
    "omnicosmos_quijote_finetune_1x4"
    "omnilearned_proxy_cube_finetune_1x4"
    "omnilearned_proxy_bounded_ball_finetune_1x4"
    "omnilearned_proxy_gaussian_ball_finetune_1x4"
    "omnilearned_proxy_gaussian_ellipsoid_sigmax_finetune_1x4"
    "omnilearned_proxy_bounded_ellipsoid_a_finetune_1x4"
    "omnilearned_proxy_box_a_finetune_1x4"
    "omnilearned_proxy_eigen_lambda1_finetune_1x4"
)
baseline_models_default=(
    "baseline_256"
)

if [[ -n "${TARGET_MODELS:-}" ]]; then
    read -r -a all_models <<< "${TARGET_MODELS}"
    omnilearned_models=()
    baseline_models=()
    for m in "${all_models[@]}"; do
        if [[ "$m" == baseline* ]]; then
            baseline_models+=("$m")
        else
            omnilearned_models+=("$m")
        fi
    done
else
    omnilearned_models=("${omnilearned_models_default[@]}")
    baseline_models=("${baseline_models_default[@]}")
fi

# Single process (no torchrun/srun) -- a one-off forward pass isn't worth
# 4-GPU DDP overhead. OmniLearned still needs these env vars set regardless.
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=$(( 15000 + ($$ % 15000) ))

eval_omnilearned() {
    local run_dir="$1"
    local out_csv="${run_dir}/results/${output_csv_name}"

    if [[ ! -f "${run_dir}/best_model_.pt" ]]; then
        echo "[skip] no checkpoint: ${run_dir}/best_model_.pt"
        return
    fi
    if [[ -f "${out_csv}" && -z "${FORCE:-}" ]]; then
        echo "[done] ${out_csv} already exists"
        return
    fi

    echo "[eval] ${run_dir}  ->  ${target_columns}_${eval_dataset_size} test split"
    rm -rf "${run_dir}/shared_test${eval_dataset_size}"
    mkdir -p "${run_dir}/shared_test${eval_dataset_size}"
    (
        cd "${run_dir}"
        omnilearned evaluate \
            -i ./ \
            -o "./shared_test${eval_dataset_size}/" \
            --save-tag "" \
            --dataset streams \
            --path "${shared_data_dir}" \
            --size small \
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
        --npz_glob "${run_dir}/shared_test${eval_dataset_size}/outputs__streams_*.npz" \
        --output_csv "${out_csv}"
}

eval_baseline() {
    local dataset_size="$1"
    local model="$2"
    local idx="$3"
    local run_dir="${repo_root}/${target_columns}_${dataset_size}/${model}/run_${idx}"
    local out_csv="${run_dir}/results/${output_csv_name}"

    if [[ -f "${out_csv}" && -z "${FORCE:-}" ]]; then
        echo "[done] ${out_csv} already exists"
        return
    fi

    echo "[eval] ${run_dir}  ->  ${target_columns}_${eval_dataset_size} test split"
    python "${repo_root}/src/eval_baseline_shared_test.py" \
        --repo_root "${repo_root}" \
        --target_columns "${target_columns}" \
        --dataset_size "${dataset_size}" \
        --target_model "${model}" \
        --run_idx "${idx}" \
        --eval_dataset_size "${eval_dataset_size}" \
        --output_csv_name "${output_csv_name}"
}

for dataset_size in "${dataset_sizes[@]}"; do
    for model in "${omnilearned_models[@]}"; do
        for idx in "${run_idxs[@]}"; do
            eval_omnilearned "${repo_root}/${target_columns}_${dataset_size}/${model}/run_${idx}"
        done
    done
    for model in "${baseline_models[@]}"; do
        for idx in "${run_idxs[@]}"; do
            eval_baseline "${dataset_size}" "${model}" "${idx}"
        done
    done
done

echo "All done."
