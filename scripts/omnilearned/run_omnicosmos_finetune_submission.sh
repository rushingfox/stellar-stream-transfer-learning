#!/bin/bash
# Sbatch wrapper for the 1x4 OmniCosmos-checkpoint stream finetune (CAMELS or
# Quijote pretrain, finetuned on the stream mass-regression task) — a borrowed
# pretrain-weights comparison row, same category as the jet-pretrain finetune.
#
# Reuses run_omnilearned_proxy_finetune.sh directly (no new runner script
# needed): its checkpoint-copy-by-tag + torchrun train/evaluate logic is
# identical for any pretrain source, proxy or OmniCosmos alike.
#
# For JupyterHub interactive use, DO NOT invoke this wrapper. Instead:
#   1) Create the finetune dir manually, cp the checkpoint in from
#      omnicosmos_checkpoints/ as best_model_<pretrain_tag>.pt (data is shared
#      at the dataset level, <target_columns>_<size>/data/ — no per-model
#      data/ needed)
#   2) mkdir run_0 && cd run_0
#   3) bash <path>/run_omnilearned_proxy_finetune.sh <num_feat> <pretrain_tag> <epoch_num>
#
# Required env:
#   PRETRAIN_SOURCE   "camels" or "quijote"
#
# Optional env:
#   CHECKPOINT_SIZE       default: 600 for camels, 10000 for quijote (the
#                         OmniCosmos pretrain's own simulation-count bucket —
#                         independent of our stream DATASET_SIZE below)
#   DATASET_SIZE          default 10000 (the stream finetune dataset size)
#   EPOCH_NUM             default 100
#   MODEL_SUFFIX          default "" (appended to the finetune dir)
#   SBATCH_QOS / SBATCH_TIME / SBATCH_ARRAY

set -euo pipefail

pretrain_source=${PRETRAIN_SOURCE:?Error: set PRETRAIN_SOURCE (camels or quijote)}
dataset_size=${DATASET_SIZE:-10000}
epoch_num=${EPOCH_NUM:-100}
model_suffix=${MODEL_SUFFIX:-}

case "${pretrain_source}" in
    camels)  default_checkpoint_size=600 ;;
    quijote) default_checkpoint_size=10000 ;;
    *) echo "Error: PRETRAIN_SOURCE must be 'camels' or 'quijote', got '${pretrain_source}'" >&2; exit 1 ;;
esac
checkpoint_size=${CHECKPOINT_SIZE:-${default_checkpoint_size}}

# Auto-tighten the time estimate for the smaller sizes unless the user
# explicitly overrides, as for every other finetune (see there): 40/100 get
# 30 min, 300 1h and 1000 1h30; 10000 stays on the runner script's own
# SBATCH --time default (5h). QOS stays regular (not debug): debug caps
# concurrent jobs per user at ~3, which becomes a bottleneck once you're
# submitting many seeds/models in parallel; regular has no such cap and
# these jobs are short regardless. Explicit SBATCH_QOS/SBATCH_TIME always
# take priority.
case "${dataset_size}" in
    40|100) SBATCH_QOS="${SBATCH_QOS:-regular}"; SBATCH_TIME="${SBATCH_TIME:-00:30:00}" ;;
    300)    SBATCH_QOS="${SBATCH_QOS:-regular}"; SBATCH_TIME="${SBATCH_TIME:-01:00:00}" ;;
    1000)   SBATCH_QOS="${SBATCH_QOS:-regular}"; SBATCH_TIME="${SBATCH_TIME:-01:30:00}" ;;
esac

target_column="3d_x_y_z"
pretrain_tag="fine_tune_${pretrain_source}_s_${checkpoint_size}"
model="omnicosmos_${pretrain_source}_finetune${model_suffix}"
job_name="oc_${pretrain_source}_ft${model_suffix}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"

pretrain_ckpt="${repo_dir}/omnicosmos_checkpoints/best_model_${pretrain_tag}.pt"
# TASK=subhalo trains under subhalo_<target_column>_<size>/.
source "${script_dir}/../task_config.sh"
dataset_dir="${repo_dir}/${task_dir_prefix}${target_column}_${dataset_size}"
run_dir="${dataset_dir}/${model}"

if [[ ! -f "${pretrain_ckpt}" ]]; then
    echo "Error: OmniCosmos checkpoint not found at ${pretrain_ckpt}" >&2
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
  pretrain_source  : ${pretrain_source}
  checkpoint_size  : ${checkpoint_size}
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
