#!/bin/bash
# Bootstrap error bar (test-set resampling of a single fixed run) for every
# checkpoint-based + non-ML model, on the shared_test10000 predictions. See
# src/compute_bootstrap_shared_test.py for the method. Requires each run's
# predictions.npz / outputs__streams_0.npz -- run eval_shared_test10000.sh /
# eval_shared_test10000_nonML.sh first if this skips everything.
#
# No SBATCH header: run directly, interactively (pure CPU/numpy resampling,
# no GPU needed).
#
# Usage: bash scripts/compute_bootstrap_shared_test_all.sh
# Env overrides: DATASET_SIZES (default "300 1000 10000"), RUN_IDXS (default
# "0" -- bootstrap fixes a single trained model, not a cross-seed statistic),
# EVAL_DATASET_SIZE (default 10000), N_BOOT (default 1000)

set -euo pipefail

module load conda
mamba activate stream_transfer_learning

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

target_columns="3d_x_y_z"
eval_dataset_size=${EVAL_DATASET_SIZE:-10000}
n_boot=${N_BOOT:-1000}

if [[ -n "${DATASET_SIZES:-}" ]]; then
    read -r -a dataset_sizes <<< "${DATASET_SIZES}"
else
    dataset_sizes=(300 1000 10000)
fi

if [[ -n "${RUN_IDXS:-}" ]]; then
    read -r -a run_idxs <<< "${RUN_IDXS}"
else
    run_idxs=(0)
fi

models=(
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
    "baseline_256"
    "average"
    "eigenvalue_regression"
)

for dataset_size in "${dataset_sizes[@]}"; do
    echo "=== dataset_size=${dataset_size} ==="
    for model in "${models[@]}"; do
        model_dir="${repo_root}/${target_columns}_${dataset_size}/${model}"
        [[ -d "${model_dir}" ]] || continue
        echo "[bootstrap] ${model}"
        python "${repo_root}/src/compute_bootstrap_shared_test.py" \
            --repo_root "${repo_root}" --target_columns "${target_columns}" \
            --dataset_size "${dataset_size}" --target_model "${model}" \
            --run_idxs "${run_idxs[@]}" \
            --eval_dataset_size "${eval_dataset_size}" --n_boot "${n_boot}"
    done
done

echo "All done."
