#!/bin/bash
# Sbatch wrapper for the 1x4 real-data scratch (no-pretrain) runner.
#
# For JupyterHub interactive use, DO NOT invoke this wrapper. Instead:
#   1) mkdir run_0 && cd run_0
#   2) bash <path>/run_omnilearned_scratch.sh <num_feat> <epoch_num>
#
# Optional env:
#   DATASET_SIZE   default 10000
#   EPOCH_NUM      default 100
#   MODEL_SUFFIX   default "" (none)
#   MODEL_SIZE     small (default) | medium | large; non-small needs an explicit
#                  MODEL_SUFFIX, e.g. "_m"
#   SBATCH_QOS / SBATCH_TIME / SBATCH_ARRAY
#
# Data: uses the shared dataset-level data dir (<target_columns>_<size>/data/),
# referenced via --path ../../data — no per-model data/ or symlink needed.

set -euo pipefail

dataset_size=${DATASET_SIZE:-10000}
epoch_num=${EPOCH_NUM:-100}
model_suffix=${MODEL_SUFFIX:-}
model_size=${MODEL_SIZE:-small}

case "${model_size}" in
    small|medium|large) ;;
    *) echo "Error: MODEL_SIZE must be small, medium or large, got '${model_size}'" >&2; exit 1 ;;
esac
# Keep non-small runs out of the small models' folders.
if [[ "${model_size}" != "small" && -z "${MODEL_SUFFIX:-}" ]]; then
    echo "Error: MODEL_SIZE=${model_size} needs an explicit MODEL_SUFFIX (e.g. _m)" >&2
    exit 1
fi
export MODEL_SIZE="${model_size}"

# Auto-tighten the time estimate for the smaller sizes unless the user
# explicitly overrides. Each job trains for the default 100 epochs and then
# scores its run on the shared test set: measured ~15 min of training at 40
# (mostly start-up), ~16 min per 50 epochs at 1000, plus a few minutes of
# evaluation, so 40/100 get 30 min, 300 gets 1h and 1000 1h30. 10000 (~4h)
# stays on the runner script's own SBATCH --time default. QOS stays regular (not debug): debug caps concurrent
# jobs per user at ~3, which becomes a bottleneck once you're submitting many
# seeds/models in parallel; regular has no such cap and these jobs are short
# regardless. Explicit SBATCH_QOS/SBATCH_TIME always take priority.
# Only for the small model; larger ones keep the runner's own --time.
if [[ "${model_size}" == "small" ]]; then
    case "${dataset_size}" in
        40|100) SBATCH_QOS="${SBATCH_QOS:-regular}"; SBATCH_TIME="${SBATCH_TIME:-00:30:00}" ;;
        300)    SBATCH_QOS="${SBATCH_QOS:-regular}"; SBATCH_TIME="${SBATCH_TIME:-01:00:00}" ;;
        1000)   SBATCH_QOS="${SBATCH_QOS:-regular}"; SBATCH_TIME="${SBATCH_TIME:-01:30:00}" ;;
    esac
fi

target_column="3d_x_y_z"
model="omnilearned_scratch${model_suffix}"
job_name="o_sc${model_suffix}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"

# TASK=subhalo trains under subhalo_<target_column>_<size>/.
source "${script_dir}/../task_config.sh"
dataset_dir="${repo_dir}/${task_dir_prefix}${target_column}_${dataset_size}"
run_dir="${dataset_dir}/${model}"

# Refuse stale data before queueing (the runner checks again before training).
source "${script_dir}/check_data.sh"
require_downstream_data "${dataset_dir}/data" || exit 1

mkdir -p "${run_dir}/logs"

sbatch_args=()
[[ -n "${SBATCH_QOS:-}"   ]] && sbatch_args+=(--qos="${SBATCH_QOS}")
[[ -n "${SBATCH_TIME:-}"  ]] && sbatch_args+=(--time="${SBATCH_TIME}")
[[ -n "${SBATCH_ARRAY:-}" ]] && sbatch_args+=(--array="${SBATCH_ARRAY}")

cat <<EOF
[Submitting]
  job_name         : ${job_name}
  model_size       : ${model_size}
  run_dir          : ${run_dir}
  qos / time       : ${SBATCH_QOS:-script-default} / ${SBATCH_TIME:-script-default}
  array            : ${SBATCH_ARRAY:-script-default (0-2)}

  Runner hparams stay at OmniLearned CLI defaults (lr, wd).
  Only epoch_num is passed (${epoch_num}).
EOF

(
    cd "${run_dir}"
    sbatch "${sbatch_args[@]}" --job-name="${job_name}" \
        "${script_dir}/run_omnilearned_scratch.sh" \
        "3" "${epoch_num}"
)
