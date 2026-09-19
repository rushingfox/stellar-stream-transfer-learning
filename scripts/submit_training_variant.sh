#!/bin/bash

set -euo pipefail

variant=${1:?Error: please provide variant baseline/baseline_256}

case "$variant" in
    baseline)
        model_dir="baseline"
        run_script="run_baseline.sh"
        names_default=("b_3d_xyz" "b_3d_radec" "b_4d_xyz" "b_4d_radec" "b_6d_xyz" "b_6d_radec")
        target_columns_default=(
            "3d_x_y_z"
            "3d_ra_dec_parallax"
            "4d_x_y_z_vtotal"
            "4d_ra_dec_parallax_vtotal"
            "6d_x_y_z_vx_vy_vz"
            "6d_ra_dec_parallax_proper_motions"
        )
        submit_idxs_default=(0)
        ;;
    baseline_256)
        model_dir="baseline_256"
        run_script="run_baseline_256.sh"
        names_default=("b256_3d_xyz" "b256_3d_radec" "b256_4d_xyz" "b256_4d_radec" "b256_6d_xyz" "b256_6d_radec")
        target_columns_default=(
            "3d_x_y_z"
            "3d_ra_dec_parallax"
            "4d_x_y_z_vtotal"
            "4d_ra_dec_parallax_vtotal"
            "6d_x_y_z_vx_vy_vz"
            "6d_ra_dec_parallax_proper_motions"
        )
        submit_idxs_default=(0)
        ;;
    *)
        echo "Error: unsupported variant='$variant'." >&2
        exit 1
        ;;
esac

dataset_size=${DATASET_SIZE:-1000}

if [[ -n "${TARGET_COLUMNS:-}" ]]; then
    read -r -a target_columns <<< "${TARGET_COLUMNS}"
else
    target_columns=("${target_columns_default[@]}")
fi

if [[ -n "${SUBMIT_IDXS:-}" ]]; then
    read -r -a submit_idxs <<< "${SUBMIT_IDXS}"
else
    if [[ -n "${TARGET_COLUMNS:-}" ]]; then
        submit_idxs=()
        for ((i = 0; i < ${#target_columns[@]}; i++)); do
            submit_idxs+=("$i")
        done
    else
        submit_idxs=("${submit_idxs_default[@]}")
    fi
fi

if [[ -n "${SUBMIT_JOB_NAMES:-}" ]]; then
    read -r -a names <<< "${SUBMIT_JOB_NAMES}"
else
    if [[ -n "${TARGET_COLUMNS:-}" ]]; then
        names=()
        for col in "${target_columns[@]}"; do
            safe_col="${col//[^[:alnum:]]/_}"
            names+=("${variant}_${safe_col}")
        done
    else
        names=("${names_default[@]}")
    fi
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

sbatch_args=()
if [[ -n "${SBATCH_QOS:-}" ]]; then
    sbatch_args+=(--qos="${SBATCH_QOS}")
fi
sbatch_time="${SBATCH_TIME:-}"
if [[ -z "$sbatch_time" ]]; then
    case "${variant}:${dataset_size}" in
        baseline:100|baseline:300|baseline:1000|baseline_256:100|baseline_256:300|baseline_256:1000)
            sbatch_time="00:30:00"
            ;;
        baseline:10000|baseline_256:10000)
            sbatch_time="05:00:00"
            ;;
    esac
fi
if [[ -n "$sbatch_time" ]]; then
    sbatch_args+=(--time="${sbatch_time}")
fi
if [[ -n "${SBATCH_ARRAY:-}" ]]; then
    sbatch_args+=(--array="${SBATCH_ARRAY}")
fi

for idx in "${submit_idxs[@]}"; do
    if (( idx < 0 || idx >= ${#target_columns[@]} )); then
        echo "Error: SUBMIT_IDXS contains out-of-range index '${idx}' for TARGET_COLUMNS size ${#target_columns[@]}." >&2
        exit 1
    fi

    if (( idx < ${#names[@]} )); then
        job_name="${names[$idx]}"
    else
        job_name="${variant}_${idx}"
    fi

    run_dir="${repo_dir}/${target_columns[$idx]}_${dataset_size}/${model_dir}"
    mkdir -p "${run_dir}/logs"
    (
        cd "${run_dir}"
        echo "[Submitting] job=${job_name}  qos=${SBATCH_QOS:-script-default}  time=${sbatch_time:-script-default}"
        sbatch "${sbatch_args[@]}" --job-name="${job_name}" "${script_dir}/${run_script}" "${idx}"
    )
done
