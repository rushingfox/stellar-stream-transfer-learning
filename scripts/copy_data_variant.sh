#!/bin/bash

set -euo pipefail

src_model=${1:?Error: please provide source model, e.g. baseline}
dst_model=${2:?Error: please provide destination model, e.g. baseline_256}

case "$src_model" in
    baseline)
        target_columns_default=(
            "3d_x_y_z"
            "3d_ra_dec_parallax"
            "4d_x_y_z_vtotal"
            "4d_ra_dec_parallax_vtotal"
            "6d_x_y_z_vx_vy_vz"
            "6d_ra_dec_parallax_proper_motions"
        )
        ;;
    *)
        echo "Error: unsupported src_model='$src_model'. Expected baseline." >&2
        exit 1
        ;;
esac

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

for dataset_size in "${dataset_sizes[@]}"; do
    for target_column in "${target_columns[@]}"; do
        old_data_path="../${target_column}_${dataset_size}/${src_model}/data"
        new_data_dir="../${target_column}_${dataset_size}/${dst_model}/"
        echo "Copying ${src_model} -> ${dst_model} for ${target_column}, size=${dataset_size}"
        mkdir -p "$new_data_dir"
        cp -a "$old_data_path" "$new_data_dir"
    done
done
