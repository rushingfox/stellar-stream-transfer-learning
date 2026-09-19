#!/bin/bash
# Hybrid runner for 1 node x 4 GPU PROXY PRETRAIN training on synthetic
# geometric proxy data. Same dual-mode design as run_omnilearned_scratch_1x4.sh:
#   1) sbatch mode: submitted via a wrapper; SBATCH headers used.
#   2) bash mode:   run directly inside a JupyterHub Exclusive GPU session with
#                   `bash run_omnilearned_proxy_pretrain_1x4.sh <num_feat> <num_classes> <epoch_num>`
#                   from the pretrain dir's run_N subdirectory.
#
# Launcher: torchrun. All hparams stay at OmniLearned CLI defaults.
#
# --array defaults to a single seed (0-0): the finetune stage consumes run_0's
# checkpoint, so additional pretrain seeds would never be read. Override with
# SBATCH_ARRAY on the submission wrapper if you do want more.
#
# Positional args:
#   $1 : num_feat     (e.g. 3 for 3d_proxy_*)
#   $2 : num_classes  (label dimensionality; 1 for all seven unified 1-target proxies)
#   $3 : epoch_num    (e.g. 100)

#SBATCH -A m4474
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jianhao.wu@wisc.edu

set -euo pipefail

_INVOCATION_LINE="$(printf '%q ' "$0" "$@")"

# REPO_ROOT: prefer BASH_SOURCE derivation (works in both sbatch and bash
# modes), fall back to SLURM_SUBMIT_DIR only if BASH_SOURCE-based path
# doesn't contain src/write_hparams_json.py (which would indicate slurm
# copied the script to a temp dir instead of running it in-place).
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || _script_dir=""
if [[ -n "${_script_dir}" && -f "${_script_dir}/../../src/write_hparams_json.py" ]]; then
    REPO_ROOT="$(cd "${_script_dir}/../.." && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/../../src/write_hparams_json.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/../.." && pwd)"
else
    echo "ERROR: cannot locate REPO_ROOT (src/write_hparams_json.py not found)" >&2
    exit 1
fi

module load conda
mamba activate stream_transfer_learning

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p logs
    BASE_IDX=0
    CURRENT_IDX=$((BASE_IDX + SLURM_ARRAY_TASK_ID))
    mkdir -p "./run_${CURRENT_IDX}"
    cd "./run_${CURRENT_IDX}"
fi

NUM_FEAT=${1:?Error: need num_feat}
NUM_CLASSES=${2:?Error: need num_classes}
TOTAL_EPOCHS=${3:?Error: need epoch_num}

export MASTER_PORT=$(( 15000 + (${SLURM_JOB_ID:-$$} % 15000) ))
export MASTER_ADDR=localhost

# lr_factor is finetune-only (multiplies new-layer LR when --fine-tune is set);
# pretrain has no --fine-tune, so we omit it from both explicit hparams and
# invocation.txt to avoid noise (same reasoning as run_omnilearned_scratch_1x4.sh).
_RAW_LR=${LR-UNSET}
_RAW_WD=${WD-UNSET}
_RAW_BATCH=${BATCH-UNSET}

LR=${LR:-5e-5}
WD=${WD:-0.0}
BATCH=${BATCH:-4}
SIZE=small
MODE=regression

cat > invocation.txt <<EOF
# Recorded: $(date -Iseconds)
# Host:     $(hostname)
# User:     $(whoami)
# CWD:      $(pwd)
# SLURM_JOB_ID: ${SLURM_JOB_ID:-(none, bash mode)}
#
# --- User-set env vars at invocation (UNSET = user did not set; runner default applied) ---
#   LR    = ${_RAW_LR}
#   WD    = ${_RAW_WD}
#   BATCH = ${_RAW_BATCH}
#
# --- Command as invoked (verbatim; bash cannot capture env-var prefixes like LR=...) ---
${_INVOCATION_LINE}
#
# --- Fully reproducible command (defaults inlined; use this to re-run even if
#     the runner's built-in defaults change later; identical to what the training
#     actually saw) ---
LR=${LR} WD=${WD} BATCH=${BATCH} ${_INVOCATION_LINE}
EOF

python3 << PYEOF
import sys
sys.path.insert(0, "${REPO_ROOT}/src")
from write_hparams_json import write_hparams

explicit = {
    "outdir":            "./",
    "save_tag":          "",
    "dataset":           "streams",
    "path":              "../data",
    "model_size":        "${SIZE}",
    "num_classes":       int("${NUM_CLASSES}"),
    "num_feat":          int("${NUM_FEAT}"),
    "num_coord":         3,
    "mode":              "${MODE}",
    "epoch":             int("${TOTAL_EPOCHS}"),
    "lr":                float("${LR}"),
    "wd":                float("${WD}"),
    "batch":             int("${BATCH}"),
    "local_interaction": True,
    "interaction_type":  "astro",
    "fine_tune":         False,
}
slurm = {
    "run_idx":       "${CURRENT_IDX:-0}",
    "job_id":        "${SLURM_JOB_ID:-}",
    "array_job_id":  "${SLURM_ARRAY_JOB_ID:-}",
    "array_task_id": "${SLURM_ARRAY_TASK_ID:-}",
    "ntasks":        "${SLURM_NTASKS:-}",
    "nodelist":      "${SLURM_NODELIST:-}",
}
write_hparams("hparams.json", explicit=explicit, slurm=slurm)
PYEOF

echo "======================================="
echo "[1x4 proxy pretrain] launcher = torchrun"
echo "  num_feat / num_classes / epochs : ${NUM_FEAT} / ${NUM_CLASSES} / ${TOTAL_EPOCHS}"
echo "  lr / wd / batch                 : ${LR} / ${WD} / ${BATCH}"
echo "  MASTER_ADDR:MASTER_PORT         : ${MASTER_ADDR}:${MASTER_PORT}"
echo "======================================="

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "$(which omnilearned)" train \
    -o ./ \
    --save-tag "" \
    --dataset streams \
    --path ../data \
    --size "${SIZE}" \
    --num-classes "${NUM_CLASSES}" \
    --num-feat "${NUM_FEAT}" \
    --num-coord 3 \
    --mode "${MODE}" \
    --epoch "${TOTAL_EPOCHS}" \
    --lr "${LR}" \
    --wd "${WD}" \
    --batch "${BATCH}" \
    --local-interaction \
    --interaction-type astro

if [ $? -ne 0 ]; then
    echo "Error: Training failed." >&2
    exit 1
fi

echo "---------------------------------------"
echo "[1x4 proxy pretrain] Evaluation"
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "$(which omnilearned)" evaluate \
    -i ./ \
    -o ./ \
    --save-tag "" \
    --dataset streams \
    --path ../data \
    --size "${SIZE}" \
    --num-classes "${NUM_CLASSES}" \
    --num-feat "${NUM_FEAT}" \
    --num-coord 3 \
    --mode "${MODE}" \
    --batch "${BATCH}" \
    --local-interaction \
    --interaction-type astro

echo "---------------------------------------"
echo "[1x4 proxy pretrain] Generating results summary"
mkdir -p results
python3 "${REPO_ROOT}/src/omnilearned_summary.py" \
    --json_path "./training_.json" \
    --npz_glob "./outputs__streams_*.npz" \
    --output_csv "./results/loss_history.csv" \
    || echo "Warning: results summary generation failed (non-fatal, training/eval already succeeded)" >&2

echo "Run completed."
