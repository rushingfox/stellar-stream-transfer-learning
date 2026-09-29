#!/bin/bash
# Hybrid runner for 1 node x 4 GPU finetune from OmniLearned's own general
# "pretrain_s" jet-physics checkpoint (auto-fetched by the CLI itself when
# --pretrain-tag pretrain_s is given — NOT an OmniCosmos artifact; OmniCosmos
# separately further-finetunes this same base checkpoint on cosmological sims,
# which is a different, out-of-scope thing).
#
# Unlike run_omnilearned_proxy_finetune_1x4.sh, there is NO checkpoint-copy
# step here: pretrain_s is resolved/downloaded by OmniLearned itself, not
# copied in from a local proxy-pretrain run dir.
#
# Works in TWO modes:
#   1) sbatch mode: submitted via _1x4_submission.sh; SBATCH headers are used.
#   2) bash mode:   run directly inside a JupyterHub Exclusive GPU session with
#                   `bash run_omnilearned_finetune_1x4.sh <num_feat> <epoch_num>`
#                   from the finetune dir's run_N subdirectory.
#
# Launcher: torchrun. All hparams stay at OmniLearned CLI defaults.
#
# Positional args:
#   $1 : num_feat     (e.g. 3 for 3d_x_y_z)
#   $2 : epoch_num    (e.g. 50)
#
# Env: MODEL_SIZE = small | medium | large (small by default), which also picks
# the jet checkpoint (pretrain_s / _m / _l), plus LR / WD / LR_FACTOR / BATCH.

#SBATCH -A m4474
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --array=0-2
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
TOTAL_EPOCHS=${2:?Error: need epoch_num}

export MASTER_PORT=$(( 15000 + (${SLURM_JOB_ID:-$$} % 15000) ))
export MASTER_ADDR=localhost

_RAW_LR=${LR-UNSET}
_RAW_WD=${WD-UNSET}
_RAW_LR_FACTOR=${LR_FACTOR-UNSET}
_RAW_BATCH=${BATCH-UNSET}
_RAW_MODEL_SIZE=${MODEL_SIZE-UNSET}

LR=${LR:-5e-5}
WD=${WD:-0.0}
LR_FACTOR=${LR_FACTOR:-1.0}
BATCH=${BATCH:-4}
SIZE=${MODEL_SIZE:-small}
MODE=regression
NUM_CLASSES=1

case "${SIZE}" in
    small)  LOAD_TAG="pretrain_s" ;;
    medium) LOAD_TAG="pretrain_m" ;;
    large)  LOAD_TAG="pretrain_l" ;;
    *) echo "Error: MODEL_SIZE must be small, medium or large, got '${SIZE}'" >&2; exit 1 ;;
esac

# Use the local copy of the jet checkpoint when there is one. Otherwise all
# four ranks download it at once and one can read a half-written file.
# Without a local copy, OmniLearned downloads it as before.
LOCAL_CKPT="${REPO_ROOT}/omnicosmos_checkpoints/best_model_${LOAD_TAG}.pt"
if [[ ! -f "./best_model_${LOAD_TAG}.pt" && -f "${LOCAL_CKPT}" ]]; then
    cp "${LOCAL_CKPT}" ./
fi

cat > invocation.txt <<EOF
# Recorded: $(date -Iseconds)
# Host:     $(hostname)
# User:     $(whoami)
# CWD:      $(pwd)
# SLURM_JOB_ID: ${SLURM_JOB_ID:-(none, bash mode)}
#
# --- User-set env vars at invocation (UNSET = user did not set; runner default applied) ---
#   LR        = ${_RAW_LR}
#   WD        = ${_RAW_WD}
#   LR_FACTOR = ${_RAW_LR_FACTOR}
#   BATCH     = ${_RAW_BATCH}
#   MODEL_SIZE = ${_RAW_MODEL_SIZE}
#
# --- Command as invoked (verbatim; bash cannot capture env-var prefixes like LR=...) ---
${_INVOCATION_LINE}
#
# --- Fully reproducible command (defaults inlined; use this to re-run even if
#     the runner's built-in defaults change later; identical to what the training
#     actually saw) ---
LR=${LR} WD=${WD} LR_FACTOR=${LR_FACTOR} BATCH=${BATCH} MODEL_SIZE=${SIZE} ${_INVOCATION_LINE}
EOF

python3 << PYEOF
import sys
sys.path.insert(0, "${REPO_ROOT}/src")
from write_hparams_json import write_hparams

explicit = {
    "outdir":            "./",
    "save_tag":          "",
    "pretrain_tag":      "${LOAD_TAG}",
    "dataset":           "streams",
    "path":              "../../data",
    "model_size":        "${SIZE}",
    "num_classes":       ${NUM_CLASSES},
    "num_feat":          int("${NUM_FEAT}"),
    "num_coord":         3,
    "mode":              "${MODE}",
    "epoch":             int("${TOTAL_EPOCHS}"),
    "lr":                float("${LR}"),
    "wd":                float("${WD}"),
    "lr_factor":         float("${LR_FACTOR}"),
    "batch":             int("${BATCH}"),
    "local_interaction": True,
    "interaction_type":  "astro",
    "fine_tune":         True,
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
echo "[1x4 jet-pretrain finetune] launcher = torchrun"
echo "  pretrain_tag / num_feat / epochs : ${LOAD_TAG} / ${NUM_FEAT} / ${TOTAL_EPOCHS}"
echo "  lr / wd / lr_factor / batch      : ${LR} / ${WD} / ${LR_FACTOR} / ${BATCH}"
echo "  model size                       : ${SIZE}"
echo "  MASTER_ADDR:MASTER_PORT          : ${MASTER_ADDR}:${MASTER_PORT}"
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
    --path ../../data \
    --size "${SIZE}" \
    --num-classes "${NUM_CLASSES}" \
    --num-feat "${NUM_FEAT}" \
    --num-coord 3 \
    --mode "${MODE}" \
    --epoch "${TOTAL_EPOCHS}" \
    --lr "${LR}" \
    --wd "${WD}" \
    --lr-factor "${LR_FACTOR}" \
    --batch "${BATCH}" \
    --local-interaction \
    --interaction-type astro \
    --fine-tune \
    --pretrain-tag "${LOAD_TAG}"

if [ $? -ne 0 ]; then
    echo "Error: Training failed." >&2
    exit 1
fi

echo "---------------------------------------"
echo "[1x4 jet-pretrain finetune] Evaluation"
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
    --path ../../data \
    --size "${SIZE}" \
    --num-classes "${NUM_CLASSES}" \
    --num-feat "${NUM_FEAT}" \
    --num-coord 3 \
    --mode "${MODE}" \
    --batch "${BATCH}" \
    --local-interaction \
    --interaction-type astro

echo "---------------------------------------"
echo "[1x4 jet-pretrain finetune] Generating results summary"
mkdir -p results
python3 "${REPO_ROOT}/src/omnilearned_summary.py" \
    --json_path "./training_.json" \
    --npz_glob "./outputs__streams_*.npz" \
    --output_csv "./results/loss_history.csv" \
    || echo "Warning: results summary generation failed (non-fatal, training/eval already succeeded)" >&2

echo "Run completed."
