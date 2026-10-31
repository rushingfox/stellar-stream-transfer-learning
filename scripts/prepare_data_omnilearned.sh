#!/bin/bash
#SBATCH -A m4474
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=prepare_data_omnilearned
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jianhao.wu@wisc.edu

echo "===== Script Content ====="
cat $0
echo "===== End of Script ====="

module load conda
mamba activate stream_transfer_learning

#cd $SLURM_SUBMIT_DIR

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

# NOTE: every omnilearned-family model at a given (target_columns,
# dataset_size) — scratch, all 7 proxy finetunes, and the
# jet-pretrain finetune — trains/evaluates on the IDENTICAL real-stream data
# (only the initialization checkpoint differs). It is therefore generated
# ONCE per (target_columns, dataset_size) into a shared
# "<task>_<target_columns>_<size>/data/" dir, which every OmniLearned runner reads
# from via "--path ../../data" relative to its own run_N/. Inputs and target are
# normalized with that size's own train split (data/streams/input_stats.npz,
# target_stats.npz). data/shared_test10000/ holds the size-10000 test split with
# inputs normalized the same way, which scripts/eval_shared_test10000.sh scores
# this size's models on.
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
shared_test_raw="${repo_dir}/raw_data/${raw_data_prefix}_10000.h5"
source "${script_dir}/omnilearned/check_data.sh" || exit 1

for tc in "${target_columns[@]}"; do
    if [[ "${tc}" != "3d_x_y_z" ]]; then
        echo "Error: TARGET_COLUMNS='${tc}': the real-stream input is 3d_x_y_z only" >&2
        exit 1
    fi
done

for dataset_size in "${dataset_sizes[@]}"; do
    input_raw="${repo_dir}/raw_data/${raw_data_prefix}_${dataset_size}.h5"
    for tc in "${target_columns[@]}"; do
        dataset_dir="${repo_dir}/${task_dir_prefix}${tc}_${dataset_size}"
        # Skip only complete data in the normalized form this script writes
        # (the same check the OmniLearned runners make); refuse to build on
        # anything else rather than leave it in place.
        if require_downstream_data "${dataset_dir}/data" 2>/dev/null; then
            echo "Shared data already exists at ${dataset_dir}/data, skipping ${tc} size=${dataset_size}"
            continue
        fi
        if [[ -e "${dataset_dir}/data/streams" || -e "${dataset_dir}/data/shared_test10000" ]]; then
            echo "Error: incomplete or old-format data in ${dataset_dir}/data;" \
                 "delete ${dataset_dir}/data/streams and ${dataset_dir}/data/shared_test10000 and rerun" >&2
            exit 1
        fi
        echo "Processing shared data for ${tc}, size=${dataset_size}"
        # Everything is written to a temporary folder first and moved into
        # data/ only once prepare_data.py has succeeded. The two folders are
        # moved one after the other; an interruption between the two leaves a
        # state the check above (and the training scripts) refuse, to be
        # deleted by hand. A temporary folder left by an interrupted run is redone.
        rm -rf "${dataset_dir}/_shared_data_tmp"
        python "${repo_dir}/src/prepare_data.py" \
            --input_raw "$input_raw" \
            --output_root "${repo_dir}" \
            --dataset_size "$dataset_size" \
            --target_model "omnilearned" \
            --output_model "_shared_data_tmp" \
            --target_columns "$tc" \
            --task "${task}" \
            --shared_test_raw "${shared_test_raw}" \
            || { echo "Error: prepare_data.py failed for ${tc} size=${dataset_size}; nothing moved into ${dataset_dir}/data" >&2; exit 1; }
        # Move only streams/ and shared_test10000/, so a data/ that already holds
        # deepsets_dataset.h5 is kept rather than getting the new data/ nested inside it.
        mkdir -p "${dataset_dir}/data"
        mv "${dataset_dir}/_shared_data_tmp/data/streams" "${dataset_dir}/_shared_data_tmp/data/shared_test10000" "${dataset_dir}/data/" || exit 1
        rmdir "${dataset_dir}/_shared_data_tmp/data" "${dataset_dir}/_shared_data_tmp" 2>/dev/null || true
    done
done
