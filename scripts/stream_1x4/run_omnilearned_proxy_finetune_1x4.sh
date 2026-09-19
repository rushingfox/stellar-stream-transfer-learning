#!/bin/bash
# Hybrid runner for 1 node x 4 GPU proxy finetune on stream data.
# Works in TWO modes:
#   1) sbatch mode: submitted via _1x4_submission.sh; SBATCH headers are used.
#   2) bash mode:   run directly inside a JupyterHub Exclusive GPU session with
#                   `bash run_omnilearned_proxy_finetune_1x4.sh <num_feat> <load_tag> <epoch_num>`
#                   from the finetune dir's run_N subdirectory.
#
# Launcher: torchrun (portable across both modes; doesn't need srun).
# All hparams stay at OmniLearned CLI defaults (--lr, --wd, --lr-factor, --batch)
# to match the existing single-GPU proxy finetune convention.
#
# Data: uses the shared dataset-level data dir (<target_columns>_<size>/data/),
# referenced via --path ../../data (no per-model data/ dir/symlink).
#
# Positional args:
#   $1 : num_feat     (e.g. 3 for 3d_x_y_z)
#   $2 : load_tag     (e.g. "ellipsoid_sigmax"; determines best_model_${tag}.pt)
#   $3 : epoch_num    (e.g. 50)

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

# Capture full invocation for reproducibility (before positional args are consumed).
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

# module load / mamba activate: idempotent, safe to re-run in JupyterHub.
module load conda
mamba activate stream_transfer_learning

# Resolve run subdirectory:
#   sbatch mode: cwd = finetune dir (submission wrapper cd'd there); use array_task_id.
#   bash mode:   user is already inside run_N/, cwd is that subdir.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p logs
    BASE_IDX=0
    CURRENT_IDX=$((BASE_IDX + SLURM_ARRAY_TASK_ID))
    mkdir -p "./run_${CURRENT_IDX}"
    cd "./run_${CURRENT_IDX}"
fi

NUM_FEAT=${1:?Error: need num_feat}
LOAD_TAG=${2:?Error: need load_tag}
TOTAL_EPOCHS=${3:?Error: need epoch_num}

# Copy pretrain checkpoint from finetune dir (one level up) into run subdir.
CKPT_SRC="../best_model_${LOAD_TAG}.pt"
if [[ ! -f "./best_model_${LOAD_TAG}.pt" ]]; then
    if [[ ! -f "${CKPT_SRC}" ]]; then
        echo "Error: pretrain checkpoint not found at ${CKPT_SRC}" >&2
        exit 1
    fi
    cp "${CKPT_SRC}" ./
fi

# MASTER_PORT: derive from SLURM_JOB_ID (both sbatch and JupyterHub set this).
# Kept in 15000-30000 range to avoid Linux ephemeral port collisions (32768+).
export MASTER_PORT=$(( 15000 + (${SLURM_JOB_ID:-$$} % 15000) ))
export MASTER_ADDR=localhost

# Runtime hparams (env-overridable; each defaults to OmniLearned CLI default).
# Capture the "raw" pre-default state so invocation.txt can distinguish
# "user set X=Y" from "user didn't set X, default was used".
# `${VAR-UNSET}` returns "UNSET" only if VAR is truly unset; empty string
# ("VAR=") is treated as user-set.
_RAW_LR=${LR-UNSET}
_RAW_WD=${WD-UNSET}
_RAW_LR_FACTOR=${LR_FACTOR-UNSET}
_RAW_BATCH=${BATCH-UNSET}

LR=${LR:-5e-5}
WD=${WD:-0.0}
LR_FACTOR=${LR_FACTOR:-1.0}
BATCH=${BATCH:-4}
SIZE=small
MODE=regression
NUM_CLASSES=1

# --- Reproducibility artifacts (both bash and sbatch modes) ---

# 1) invocation.txt: the exact command line the user typed (or that sbatch fired),
#    plus env var state, so the run is fully reproducible. Written before training
#    so it's there even if training crashes early.
#
# NOTE: bash's `$0 $@` captures positional args only. Env-var prefixes
# (e.g. `LR=1e-5 bash runner.sh ...`) do NOT appear in `$@`. To avoid
# misleading readers, the file below records BOTH the raw pre-default env
# state (whether the user overrode each var) AND the effective values used.
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
#
# --- Command as invoked (verbatim; bash cannot capture env-var prefixes like LR=...) ---
${_INVOCATION_LINE}
#
# --- Fully reproducible command (defaults inlined; use this to re-run even if
#     the runner's built-in defaults change later; identical to what the training
#     actually saw) ---
LR=${LR} WD=${WD} LR_FACTOR=${LR_FACTOR} BATCH=${BATCH} ${_INVOCATION_LINE}
EOF

# 2) hparams.json: same live-introspection helper as the mn runner uses.
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
echo "[1x4 finetune] launcher = torchrun"
echo "  load_tag / num_feat / epochs : ${LOAD_TAG} / ${NUM_FEAT} / ${TOTAL_EPOCHS}"
echo "  lr / wd / lr_factor / batch  : ${LR} / ${WD} / ${LR_FACTOR} / ${BATCH}"
echo "  MASTER_ADDR:MASTER_PORT      : ${MASTER_ADDR}:${MASTER_PORT}"
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
echo "[1x4 finetune] Evaluation"
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
echo "[1x4 finetune] Generating results summary"
mkdir -p results
python3 "${REPO_ROOT}/src/omnilearned_summary.py" \
    --json_path "./training_.json" \
    --npz_glob "./outputs__streams_*.npz" \
    --output_csv "./results/loss_history.csv" \
    || echo "Warning: results summary generation failed (non-fatal, training/eval already succeeded)" >&2

echo "Run completed."
