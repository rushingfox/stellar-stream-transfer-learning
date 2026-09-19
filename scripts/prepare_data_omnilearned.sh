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

# NOTE: every omnilearned-family model at a given (target_columns,
# dataset_size) — scratch, all 7 proxy finetunes, and the
# jet-pretrain finetune — trains/evaluates on the IDENTICAL real-stream data
# (only the initialization checkpoint differs). It is therefore generated
# ONCE per (target_columns, dataset_size) into a shared
# "<target_columns>_<size>/data/" dir, which every _1x4 runner script reads
# from via "--path ../../data" relative to its own run_N/.
target_columns_default=(
    "3d_x_y_z"
    "4d_x_y_z_vtotal"
    "6d_x_y_z_vx_vy_vz"
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

for dataset_size in "${dataset_sizes[@]}"; do
    input_raw="../raw_data/${raw_data_prefix}_${dataset_size}.h5"
    for tc in "${target_columns[@]}"; do
        dataset_dir="../${tc}_${dataset_size}"
        if [[ -d "${dataset_dir}/data/streams/train" ]]; then
            echo "Shared data already exists at ${dataset_dir}/data, skipping ${tc} size=${dataset_size}"
            continue
        fi
        echo "Processing shared data for ${tc}, size=${dataset_size}"
        python ../src/prepare_data.py \
            --input_raw "$input_raw" \
            --output_root .. \
            --dataset_size "$dataset_size" \
            --target_model "omnilearned" \
            --output_model "_shared_data_tmp" \
            --target_columns "$tc"
        mv "${dataset_dir}/_shared_data_tmp/data" "${dataset_dir}/data"
        rmdir "${dataset_dir}/_shared_data_tmp" 2>/dev/null || true
    done
done
