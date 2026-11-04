#!/bin/bash
#SBATCH -A m4474
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=prepare_data_proxy_pretrain
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jianhao.wu@wisc.edu

# NOTE: cube, bounded_ball and gaussian_ball are already 1-target, so this
# output IS their final data. The other three are generated multi-target under
# prepare_data.py's own synthesis tags and need one slicing step to reach
# their final 1-target form (src/make_<new>_from_<old>.py, which copies the
# point clouds and keeps only the first/largest target column -- see README):
#   3d_proxy_ellipsoid      -> gaussian_ellipsoid_sigmax
#   3d_proxy_hard_ellipsoid -> bounded_ellipsoid_a
#   3d_proxy_box            -> box_a
# The eigen proxy is not made here: src/make_eigen_lambda_from_streams.py
# builds it from the N=10000 finetune data.

set -euo pipefail

echo "===== Script Content ====="
cat "$0"
echo "===== End of Script ====="

module load conda
mamba activate stream_transfer_learning

export HDF5_USE_FILE_LOCKING=FALSE

target_columns_default=(
    "3d_proxy_ellipsoid"
    "3d_proxy_cube"
    "3d_proxy_bounded_ball"
    "3d_proxy_gaussian_ball"
    "3d_proxy_hard_ellipsoid"
    "3d_proxy_box"
)
dataset_sizes_default=(10000)

if [[ -n "${TARGET_COLUMNS:-}" ]]; then
    read -r -a target_columns <<< "${TARGET_COLUMNS}"
else
    target_columns=("${target_columns_default[@]}")
fi

if [[ -n "${DATASET_SIZES:-}" ]]; then
    read -r -a dataset_sizes <<< "${DATASET_SIZES}"
else
    dataset_sizes=("${dataset_sizes_default[@]}")
fi

raw_data_prefix=${RAW_DATA_PREFIX:-"prog_mass_reg_dataset"}

# sbatch copies the script to a spool dir, breaking BASH_SOURCE; fall back to SLURM_SUBMIT_DIR.
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || _script_dir=""
if [[ -n "${_script_dir}" && -f "${_script_dir}/../src/prepare_data.py" ]]; then
    script_dir="${_script_dir}"
    repo_dir="$(cd "${script_dir}/.." && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/../src/prepare_data.py" ]]; then
    script_dir="${SLURM_SUBMIT_DIR}"
    repo_dir="$(cd "${script_dir}/.." && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/src/prepare_data.py" ]]; then
    repo_dir="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    script_dir="${repo_dir}/scripts"
else
    echo "ERROR: cannot locate repo_dir (src/prepare_data.py not found)" >&2
    exit 1
fi

for dataset_size in "${dataset_sizes[@]}"; do
    input_raw="${repo_dir}/raw_data/${raw_data_prefix}_${dataset_size}.h5"
    for target_column in "${target_columns[@]}"; do
        proxy_dir="${repo_dir}/${target_column}_${dataset_size}"
        # Never write over existing data: a regeneration interrupted halfway
        # would leave new and old splits side by side.
        if [[ -e "${proxy_dir}/omnilearned_proxy_pretrain/data/streams" ]]; then
            echo "Error: ${proxy_dir}/omnilearned_proxy_pretrain/data/streams already exists; delete it to regenerate" >&2
            exit 1
        fi
        echo "Preparing proxy pretrain data for ${target_column}, size=${dataset_size}"
        # Written to a temporary folder and published with a single rename only
        # once prepare_data.py has succeeded; a temporary folder left by an
        # interrupted run is redone.
        rm -rf "${proxy_dir}/_proxy_data_tmp"
        python "${repo_dir}/src/prepare_data.py" \
            --input_raw "${input_raw}" \
            --output_root "${repo_dir}" \
            --dataset_size "${dataset_size}" \
            --target_model "omnilearned" \
            --output_model "_proxy_data_tmp" \
            --target_columns "${target_column}"
        mkdir -p "${proxy_dir}/omnilearned_proxy_pretrain/data"
        mv "${proxy_dir}/_proxy_data_tmp/data/streams" "${proxy_dir}/omnilearned_proxy_pretrain/data/"
        rm -rf "${proxy_dir}/_proxy_data_tmp"
    done
done
