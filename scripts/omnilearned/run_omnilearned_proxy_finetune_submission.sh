#!/bin/bash
# Sbatch wrapper for the 1x4 proxy finetune runner.
# Copies the pretrain checkpoint into the finetune dir and submits the runner
# as an array job (default 3 seeds).
#
# For JupyterHub interactive use, DO NOT invoke this wrapper. Instead:
#   1) Create the finetune dir manually, cp checkpoint (data is shared at the
#      dataset level, <target_columns>_<size>/data/ — no per-model data/ needed)
#   2) mkdir run_0 && cd run_0
#   3) bash <path>/run_omnilearned_proxy_finetune.sh <num_feat> <load_tag> <epoch_num>
#
# Required env:
#   PROXY_DATASET   short tag (e.g. "ellipsoid", "ellipsoid_sigmax", "gaussian_ball")
#   PRETRAIN_CKPT   absolute path to pretrain best_model_.pt to use as finetune start
#
# Optional env:
#   PRETRAIN_TAG          default = ${PROXY_DATASET}
#   DATASET_SIZE          default 10000
#   EPOCH_NUM             default 100
#   MODEL_SUFFIX          default "" (appended to the finetune dir)
#   SBATCH_QOS / SBATCH_TIME / SBATCH_ARRAY

set -euo pipefail

proxy_dataset=${PROXY_DATASET:?Error: set PROXY_DATASET (e.g. ellipsoid_sigmax)}
pretrain_ckpt=${PRETRAIN_CKPT:?Error: set PRETRAIN_CKPT (path to best_model_.pt)}
dataset_size=${DATASET_SIZE:-10000}
epoch_num=${EPOCH_NUM:-100}
model_suffix=${MODEL_SUFFIX:-}
pretrain_tag=${PRETRAIN_TAG:-$proxy_dataset}

# Auto-tighten the time estimate for the smaller sizes unless the user
# explicitly overrides. Each job trains for the default 100 epochs and then
# scores its run on the shared test set: measured ~15 min of training at 40
# (mostly start-up), ~16 min per 50 epochs at 1000, plus a few minutes of
# evaluation, so 40/100 get 30 min, 300 gets 1h and 1000 1h30. 10000 (~4h)
# stays on the runner script's own SBATCH --time default. QOS stays regular (not debug): debug caps concurrent
# jobs per user at ~3, which becomes a bottleneck once you're submitting many
# seeds/models in parallel; regular has no such cap and these jobs are short
# regardless. Explicit SBATCH_QOS/SBATCH_TIME always take priority.
case "${dataset_size}" in
    40|100) SBATCH_QOS="${SBATCH_QOS:-regular}"; SBATCH_TIME="${SBATCH_TIME:-00:30:00}" ;;
    300)    SBATCH_QOS="${SBATCH_QOS:-regular}"; SBATCH_TIME="${SBATCH_TIME:-01:00:00}" ;;
    1000)   SBATCH_QOS="${SBATCH_QOS:-regular}"; SBATCH_TIME="${SBATCH_TIME:-01:30:00}" ;;
esac

target_column="3d_x_y_z"
model="omnilearned_proxy_${proxy_dataset}_finetune${model_suffix}"
job_name="olp_${proxy_dataset}_ft${model_suffix}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"

# TASK=subhalo trains under subhalo_<target_column>_<size>/.
source "${script_dir}/../task_config.sh"
dataset_dir="${repo_dir}/${task_dir_prefix}${target_column}_${dataset_size}"
run_dir="${dataset_dir}/${model}"

if [[ ! -f "${pretrain_ckpt}" ]]; then
    echo "Error: pretrain checkpoint not found at ${pretrain_ckpt}" >&2
    exit 1
fi

# Refuse stale data before queueing (the runner checks again before training).
source "${script_dir}/check_data.sh"
require_downstream_data "${dataset_dir}/data" || exit 1

mkdir -p "${run_dir}/logs"
cp -f "${pretrain_ckpt}" "${run_dir}/best_model_${pretrain_tag}.pt"

sbatch_args=()
[[ -n "${SBATCH_QOS:-}"   ]] && sbatch_args+=(--qos="${SBATCH_QOS}")
[[ -n "${SBATCH_TIME:-}"  ]] && sbatch_args+=(--time="${SBATCH_TIME}")
[[ -n "${SBATCH_ARRAY:-}" ]] && sbatch_args+=(--array="${SBATCH_ARRAY}")

cat <<EOF
[Submitting]
  job_name         : ${job_name}
  proxy_dataset    : ${proxy_dataset}
  pretrain_tag     : ${pretrain_tag}
  pretrain_ckpt    : ${pretrain_ckpt}
  run_dir          : ${run_dir}
  qos / time       : ${SBATCH_QOS:-script-default} / ${SBATCH_TIME:-script-default}
  array            : ${SBATCH_ARRAY:-script-default (0-2)}

  Runner hparams stay at OmniLearned CLI defaults (lr, wd, lr-factor).
  Only epoch_num is passed (${epoch_num}).
EOF

(
    cd "${run_dir}"
    sbatch "${sbatch_args[@]}" --job-name="${job_name}" \
        "${script_dir}/run_omnilearned_proxy_finetune.sh" \
        "3" "${pretrain_tag}" "${epoch_num}"
)
