#!/bin/bash
# Sbatch wrapper for the 1x4 finetune-from-OmniLearned's-own-pretrain_s runner
# (jet-physics baseline; NOT OmniCosmos — see run_omnilearned_finetune_1x4.sh
# header for the provenance distinction).
#
# For JupyterHub interactive use, DO NOT invoke this wrapper. Instead:
#   1) mkdir run_0 && cd run_0
#   2) bash <path>/run_omnilearned_finetune_1x4.sh <num_feat> <epoch_num>
#
# No PRETRAIN_CKPT/checkpoint-copy needed — pretrain_s is resolved by
# OmniLearned itself.
#
# Optional env:
#   DATASET_SIZE   default 10000
#   EPOCH_NUM      default 50
#   MODEL_SUFFIX   default "_1x4"
#   SBATCH_QOS / SBATCH_TIME / SBATCH_ARRAY
#
# Data: uses the shared dataset-level data dir (<target_columns>_<size>/data/),
# referenced via --path ../../data.

set -euo pipefail

dataset_size=${DATASET_SIZE:-10000}
epoch_num=${EPOCH_NUM:-50}
model_suffix=${MODEL_SUFFIX:-_1x4}

# Auto-tighten the time estimate for the two smaller sizes unless the user
# explicitly overrides — 100/300/1000 finish well within 30 min, but use the
# full 30-min cap rather than a tighter guess (a 1000-scale finetune seed can
# take just over 20 min). 10000 needs ~2h and stays on the runner script's own
# SBATCH --time default. QOS stays regular (not debug): debug caps concurrent
# jobs per user at ~3, which becomes a bottleneck once you're submitting many
# seeds/models in parallel; regular has no such cap and these jobs are short
# regardless. Explicit SBATCH_QOS/SBATCH_TIME always take priority.
if [[ "${dataset_size}" == "100" || "${dataset_size}" == "300" || "${dataset_size}" == "1000" ]]; then
    SBATCH_QOS="${SBATCH_QOS:-regular}"
    SBATCH_TIME="${SBATCH_TIME:-00:30:00}"
fi

target_column="3d_x_y_z"
model="omnilearned_finetune${model_suffix}"
job_name="o_ft${model_suffix}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"

dataset_dir="${repo_dir}/${target_column}_${dataset_size}"
run_dir="${dataset_dir}/${model}"

if [[ ! -d "${dataset_dir}/data/streams/train" ]]; then
    echo "Error: shared stream data not found at ${dataset_dir}/data/streams/train" >&2
    exit 1
fi

mkdir -p "${run_dir}/logs"

sbatch_args=()
[[ -n "${SBATCH_QOS:-}"   ]] && sbatch_args+=(--qos="${SBATCH_QOS}")
[[ -n "${SBATCH_TIME:-}"  ]] && sbatch_args+=(--time="${SBATCH_TIME}")
[[ -n "${SBATCH_ARRAY:-}" ]] && sbatch_args+=(--array="${SBATCH_ARRAY}")

cat <<EOF
[Submitting]
  job_name         : ${job_name}
  pretrain_tag     : pretrain_s (OmniLearned-provided, auto-fetched)
  run_dir          : ${run_dir}
  qos / time       : ${SBATCH_QOS:-script-default} / ${SBATCH_TIME:-script-default}
  array            : ${SBATCH_ARRAY:-script-default (0-2)}

  Runner hparams stay at OmniLearned CLI defaults (lr, wd, lr-factor).
  Only epoch_num is passed (${epoch_num}).
EOF

(
    cd "${run_dir}"
    sbatch "${sbatch_args[@]}" --job-name="${job_name}" \
        "${script_dir}/run_omnilearned_finetune_1x4.sh" \
        "3" "${epoch_num}"
)
