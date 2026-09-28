#!/bin/bash
#SBATCH -A m4474
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=prepare_data_baseline
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

target_columns_default=(
    "3d_x_y_z"
    "3d_ra_dec_parallax"
    "4d_x_y_z_vtotal"
    "4d_ra_dec_parallax_vtotal"
    "6d_x_y_z_vx_vy_vz"
    "6d_ra_dec_parallax_proper_motions"
)
dataset_sizes_default=(300 1000 10000)

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

# Every DeepSets width (baseline, baseline_256, baseline_1024, ...) trains on the
# IDENTICAL data, so it is written ONCE per (target_columns, dataset_size) into
# the shared "<target_columns>_<size>/data/" dir -- the same dir OmniLearned's
# data/streams/ lives in -- and the runner reads it via ../../data from run_N/.
for dataset_size in "${dataset_sizes[@]}"; do
    input_raw="../raw_data/${raw_data_prefix}_${dataset_size}.h5"
    for tc in "${target_columns[@]}"; do
        shared_dir="../${tc}_${dataset_size}/data"
        if [[ -f "${shared_dir}/baseline_dataset.h5" ]]; then
            echo "Shared baseline data already exists at ${shared_dir}, skipping ${tc} size=${dataset_size}"
            continue
        fi
        echo "Processing shared baseline dataset for ${tc}, size=${dataset_size}"
        python ../src/prepare_data.py \
            --input_raw "$input_raw" \
            --output_root .. \
            --dataset_size "$dataset_size" \
            --target_model "baseline" \
            --output_model "_shared_baseline_tmp" \
            --target_columns "$tc"
        mkdir -p "${shared_dir}"
        mv "../${tc}_${dataset_size}/_shared_baseline_tmp/data/baseline_dataset.h5" "${shared_dir}/"
        rmdir "../${tc}_${dataset_size}/_shared_baseline_tmp/data" "../${tc}_${dataset_size}/_shared_baseline_tmp" 2>/dev/null || true
    done
done
