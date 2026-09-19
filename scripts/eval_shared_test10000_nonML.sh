#!/bin/bash
# Re-scores the two non-ML baselines (average, eigenvalue_regression) against
# dataset_size=10000's test split, mirroring eval_shared_test10000.sh for the
# checkpoint-based models. Both are deterministic (no random seed) -- one
# evaluation per model per dataset_size, always written to run_0. Pure CPU,
# no GPU needed; run directly (no SBATCH header).
#
# Usage: bash scripts/eval_shared_test10000_nonML.sh
#
# Env overrides: DATASET_SIZES (default "300 1000 10000"), EVAL_DATASET_SIZE (default 10000)

set -euo pipefail

module load conda
mamba activate stream_transfer_learning

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

target_columns="3d_x_y_z"
eval_dataset_size=${EVAL_DATASET_SIZE:-10000}

if [[ -n "${DATASET_SIZES:-}" ]]; then
    read -r -a dataset_sizes <<< "${DATASET_SIZES}"
else
    dataset_sizes=(300 1000 10000)
fi

for dataset_size in "${dataset_sizes[@]}"; do
    # Not skipped when dataset_size == eval_dataset_size: redundant (same
    # number as the existing own-test loss_history.csv) but keeps this size
    # present in the shared-test figure too, matching the checkpoint-based
    # eval_shared_test10000.sh, which doesn't skip that case either.
    echo "[eval] average @ dataset_size=${dataset_size} -> shared test${eval_dataset_size}"
    python "${repo_root}/src/run_average.py" \
        --target_columns "${target_columns}" \
        --dataset_size "${dataset_size}" \
        --eval_dataset_size "${eval_dataset_size}"

    echo "[eval] eigenvalue_regression @ dataset_size=${dataset_size} -> shared test${eval_dataset_size}"
    python "${repo_root}/src/eigenvalue_regression.py" \
        --input_raw "${repo_root}/raw_data/prog_mass_reg_dataset_${dataset_size}.h5" \
        --output_dir "${repo_root}/${target_columns}_${dataset_size}/eigenvalue_regression/run_0" \
        --eval_input_raw "${repo_root}/raw_data/prog_mass_reg_dataset_${eval_dataset_size}.h5" \
        --eval_dataset_size "${eval_dataset_size}"
done

echo "All done."
