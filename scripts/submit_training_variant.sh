#!/bin/bash

set -euo pipefail

variant=${1:?Error: please provide variant deepsets/deepsets_256/deepsets_128}

# deepsets (hidden 1024, 1,252,481 params) is the default DeepSets; the
# narrower widths keep a _<width> suffix.
case "$variant" in
    deepsets)
        model_dir="deepsets"
        run_script="run_deepsets.sh"
        names_prefix="ds"
        ;;
    deepsets_256)
        model_dir="deepsets_256"
        run_script="run_deepsets_256.sh"
        names_prefix="ds256"
        ;;
    deepsets_128)
        model_dir="deepsets_128"
        run_script="run_deepsets_128.sh"
        names_prefix="ds128"
        ;;
    *)
        echo "Error: unsupported variant='$variant'." >&2
        exit 1
        ;;
esac
names_default=("${names_prefix}_3d_xyz")
target_columns_default=(
    "3d_x_y_z"
)
submit_idxs_default=(0)

dataset_size=${DATASET_SIZE:-1000}
# Appended to the model folder (and job name), e.g. _lr1e-4 -> deepsets_lr1e-4,
# so a run with different settings lands in its own folder instead of the
# existing one. Empty by default: the folder names stay exactly as before.
model_suffix=${MODEL_SUFFIX:-}
model_dir="${model_dir}${model_suffix}"

if [[ -n "${TARGET_COLUMNS:-}" ]]; then
    read -r -a target_columns <<< "${TARGET_COLUMNS}"
else
    target_columns=("${target_columns_default[@]}")
fi
for col in "${target_columns[@]}"; do
    if [[ "${col}" != "3d_x_y_z" ]]; then
        echo "Error: TARGET_COLUMNS='${col}': the real-stream input is 3d_x_y_z only" >&2
        exit 1
    fi
done

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
# TASK=subhalo trains under subhalo_<target_columns>_<size>/.
source "${script_dir}/task_config.sh"

sbatch_args=()
if [[ -n "${SBATCH_QOS:-}" ]]; then
    sbatch_args+=(--qos="${SBATCH_QOS}")
fi
sbatch_time="${SBATCH_TIME:-}"
if [[ -z "$sbatch_time" ]]; then
    # Sized for the default 100 epochs (the 10000 limit doubled from 50 epochs').
    case "${dataset_size}" in
        40|100|300)
            sbatch_time="00:30:00"
            ;;
        1000)
            sbatch_time="01:00:00"
            ;;
        10000)
            sbatch_time="10:00:00"
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
        job_name="${names[$idx]}${model_suffix}"
    else
        job_name="${variant}_${idx}${model_suffix}"
    fi

    run_dir="${repo_dir}/${task_dir_prefix}${target_columns[$idx]}_${dataset_size}/${model_dir}"
    mkdir -p "${run_dir}/logs"
    (
        cd "${run_dir}"
        echo "[Submitting] job=${job_name}  qos=${SBATCH_QOS:-script-default}  time=${sbatch_time:-script-default}"
        sbatch "${sbatch_args[@]}" --job-name="${job_name}" "${script_dir}/${run_script}" "${idx}"
    )
done
