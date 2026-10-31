#!/bin/bash
#SBATCH -A m4474
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=prepare_data_deepsets
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jianhao.wu@wisc.edu

echo "===== Script Content ====="
cat $0
echo "===== End of Script ====="

module load conda
mamba activate stream_transfer_learning

# cd $SLURM_SUBMIT_DIR

export HDF5_USE_FILE_LOCKING=FALSE

# Locate the repo from the script itself, or under sbatch (which runs a spool
# copy) from the submission dir, scripts/ or the repo root; all paths below are
# absolute, so the working directory does not matter.
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || _script_dir=""
if [[ -n "${_script_dir}" && -f "${_script_dir}/../src/prepare_data.py" ]]; then
    script_dir="${_script_dir}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/../src/prepare_data.py" ]]; then
    script_dir="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/src/prepare_data.py" ]]; then
    script_dir="$(cd "${SLURM_SUBMIT_DIR}/scripts" && pwd)"
else
    echo "Error: cannot locate the repo (src/prepare_data.py not found)" >&2
    exit 1
fi
repo_dir="$(cd "${script_dir}/.." && pwd)"

target_columns_default=(
    "3d_x_y_z"
)
dataset_sizes_default=(40 100 300 1000 10000)

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

# TASK picks the folder prefix and raw data (prog_mass_ or subhalo_<target_columns>_<size>/).
source "${script_dir}/task_config.sh" || exit 1
raw_data_prefix=${RAW_DATA_PREFIX:-"${task_raw_prefix}"}

for tc in "${target_columns[@]}"; do
    if [[ "${tc}" != "3d_x_y_z" ]]; then
        echo "Error: TARGET_COLUMNS='${tc}': the real-stream input is 3d_x_y_z only" >&2
        exit 1
    fi
done

# Every DeepSets width (deepsets = 1024, deepsets_256, deepsets_128) trains on the
# IDENTICAL data, so it is written ONCE per (target_columns, dataset_size) into
# the shared "<task>_<target_columns>_<size>/data/" dir -- the same dir OmniLearned's
# data/streams/ lives in -- and the runner reads it via ../../data from run_N/.
for dataset_size in "${dataset_sizes[@]}"; do
    input_raw="${repo_dir}/raw_data/${raw_data_prefix}_${dataset_size}.h5"
    for tc in "${target_columns[@]}"; do
        dataset_dir="${repo_dir}/${task_dir_prefix}${tc}_${dataset_size}"
        shared_dir="${dataset_dir}/data"
        if [[ -f "${shared_dir}/deepsets_dataset.h5" ]]; then
            echo "Shared DeepSets data already exists at ${shared_dir}, skipping ${tc} size=${dataset_size}"
            continue
        fi
        echo "Processing shared DeepSets dataset for ${tc}, size=${dataset_size}"
        # Written to a temporary folder and moved into data/ only on success; a
        # temporary folder left by an interrupted run is redone.
        rm -rf "${dataset_dir}/_shared_deepsets_tmp"
        python "${repo_dir}/src/prepare_data.py" \
            --input_raw "$input_raw" \
            --output_root "${repo_dir}" \
            --dataset_size "$dataset_size" \
            --target_model "deepsets" \
            --output_model "_shared_deepsets_tmp" \
            --target_columns "$tc" \
            --task "${task}" \
            || { echo "Error: prepare_data.py failed for ${tc} size=${dataset_size}; nothing moved into ${shared_dir}" >&2; exit 1; }
        mkdir -p "${shared_dir}"
        mv "${dataset_dir}/_shared_deepsets_tmp/data/deepsets_dataset.h5" "${shared_dir}/" || exit 1
        rmdir "${dataset_dir}/_shared_deepsets_tmp/data" "${dataset_dir}/_shared_deepsets_tmp" 2>/dev/null || true
    done
done
