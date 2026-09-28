#!/bin/bash
# Submit the 1x4 (1 node x 4 GPU DDP) proxy pretrain.
#
# Required env:
#   TARGET_COLUMN   proxy variant name, e.g. 3d_proxy_cube
#
# Optional env:
#   DATASET_SIZE    default 10000
#   TOTAL_EPOCHS    default 100
#   LR              default 5e-5 (OmniLearned CLI default)
#   WD              default 0.0  (OmniLearned CLI default)
#   BATCH           default 4 (per-rank; effective = 4 × 4 = 16)
#   MODEL_SUFFIX    default _1x4
#   SBATCH_QOS      default regular (script header)
#   SBATCH_TIME     default 05:00:00 (script header)
#   SBATCH_ARRAY    default 0-0 (single seed)
#
# Data: expected to already be in place at
#   <repo>/<target_column>_<size>/omnilearned_proxy_pretrain_1x4/data/streams/{train,val,test}/
# (placed there ahead of time by prepare_data_proxy_pretrain.sh plus the
# slicer step — this wrapper does not symlink or generate data itself).
#
# Usage:
#   TARGET_COLUMN=3d_proxy_cube \
#     bash scripts/stream_1x4/run_omnilearned_proxy_pretrain_1x4_submission.sh

set -euo pipefail

target_column=${TARGET_COLUMN:?Error: set TARGET_COLUMN (e.g. 3d_proxy_cube)}
dataset_size=${DATASET_SIZE:-10000}
epoch_num=${TOTAL_EPOCHS:-100}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"

model_suffix=${MODEL_SUFFIX:-_1x4}
model="omnilearned_proxy_pretrain${model_suffix}"
job_name="o_ppt${model_suffix}"

run_dir="${repo_dir}/${target_column}_${dataset_size}/${model}"

if [[ ! -d "${run_dir}/data/streams/train" ]]; then
    echo "Error: no data at ${run_dir}/data/streams/train" >&2
    echo "       Prepare/relocate this proxy's 1-target data first." >&2
    exit 1
fi

# Look up feat_num / num_classes per proxy — every unified 1-target
# proxy maps to num_classes=1. Keep this in sync if new proxies are added.
case "${target_column}" in
    3d_proxy_cube)                       num_feat=3; num_classes=1 ;;
    3d_proxy_bounded_ball)                num_feat=3; num_classes=1 ;;
    3d_proxy_gaussian_ball)               num_feat=3; num_classes=1 ;;
    3d_proxy_gaussian_ellipsoid_sigmax)   num_feat=3; num_classes=1 ;;
    3d_proxy_bounded_ellipsoid_a)          num_feat=3; num_classes=1 ;;
    3d_proxy_box_a)                        num_feat=3; num_classes=1 ;;
    3d_x_y_z_eigenvalues_lambda1)          num_feat=3; num_classes=1 ;;
    3d_x_y_z_eigenvalues_lambda2)          num_feat=3; num_classes=1 ;;
    3d_x_y_z_eigenvalues_lambda3)          num_feat=3; num_classes=1 ;;
    *) echo "Error: unknown TARGET_COLUMN='${target_column}'" >&2; exit 1 ;;
esac

mkdir -p "${run_dir}/logs"

sbatch_args=()
[[ -n "${SBATCH_QOS:-}"             ]] && sbatch_args+=(--qos="${SBATCH_QOS}")
[[ -n "${SBATCH_TIME:-}"            ]] && sbatch_args+=(--time="${SBATCH_TIME}")
[[ -n "${SBATCH_ARRAY:-}"           ]] && sbatch_args+=(--array="${SBATCH_ARRAY}")
[[ -n "${SBATCH_NODES:-}"           ]] && sbatch_args+=(--nodes="${SBATCH_NODES}")
[[ -n "${SBATCH_NTASKS_PER_NODE:-}" ]] && sbatch_args+=(--ntasks-per-node="${SBATCH_NTASKS_PER_NODE}")
[[ -n "${SBATCH_GPUS_PER_NODE:-}"   ]] && sbatch_args+=(--gpus-per-node="${SBATCH_GPUS_PER_NODE}")

cat <<EOF
[Submitting]
  job_name         : ${job_name}
  target_column    : ${target_column}
  dataset_size     : ${dataset_size}
  num_feat/classes : ${num_feat} / ${num_classes}
  run_dir          : ${run_dir}
  qos / time       : ${SBATCH_QOS:-script-default} / ${SBATCH_TIME:-script-default}
  array            : ${SBATCH_ARRAY:-script-default (0-0)}

  Hparams (env-passed through to sbatch, OmniLearned CLI defaults unless overridden):
    TOTAL_EPOCHS = ${epoch_num}
    LR           = ${LR:-default 5e-5}
    WD           = ${WD:-default 0.0}
    BATCH        = ${BATCH:-default 4}   (effective = 4 × 4 = 16)
EOF

(
    cd "${run_dir}"
    sbatch "${sbatch_args[@]}" --job-name="${job_name}" \
        "${script_dir}/run_omnilearned_proxy_pretrain_1x4.sh" \
        "${num_feat}" "${num_classes}" "${epoch_num}"
)
