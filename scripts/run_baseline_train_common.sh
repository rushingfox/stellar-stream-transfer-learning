#!/bin/bash

set -euo pipefail

DATA_IDX=${1:?Error: please provide index 0/1/2/3/4/5}
if [[ ! "$DATA_IDX" =~ ^[0-5]$ ]]; then
    echo "Error: DATA_IDX must be one of 0 1 2 3 4 5, got '$DATA_IDX'" >&2
    exit 1
fi

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
BASE_IDX=${BASE_IDX:-0}
CURRENT_IDX=$((BASE_IDX + SLURM_ARRAY_TASK_ID))

INPUT_DIMS=(3 3 4 4 6 6)
# Shared per-size dataset (<target_columns>_<size>/data/), relative to run_N/.
H5_FILE=${H5_FILE:-../../data/baseline_dataset.h5}
# EPOCH_NUM is the name the OmniLearned submission scripts use; accept either.
EPOCHS=${EPOCHS:-${EPOCH_NUM:-50}}
BATCH_SIZE=${BATCH_SIZE:-32}
HIDDEN_DIM=${HIDDEN_DIM:-128}

echo "Starting Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running experiment with run_idx: ${CURRENT_IDX}"
echo "Using hidden_dim: ${HIDDEN_DIM}"

# An existing checkpoint means this run_N already has results -- either a
# finished run, or one that was cut off. Continuing it silently (the old
# behaviour) would change a finished run in place, and retraining silently
# would overwrite it, so require an explicit choice:
#   RESUME=1  continue from checkpoints/checkpoint_latest.pth (e.g. after a timeout)
#   FORCE=1   delete this run_N and train from scratch
RESUME_FLAG=""
if [[ -f "./run_${CURRENT_IDX}/checkpoints/checkpoint_latest.pth" ]]; then
    if [[ "${RESUME:-}" == 1 && "${FORCE:-}" == 1 ]]; then
        echo "Error: set only one of RESUME=1 and FORCE=1." >&2
        exit 1
    elif [[ "${RESUME:-}" == 1 ]]; then
        echo "RESUME=1: continuing run_${CURRENT_IDX} from its latest checkpoint."
        RESUME_FLAG="--resume"
    elif [[ "${FORCE:-}" == 1 ]]; then
        echo "FORCE=1: deleting run_${CURRENT_IDX} and training from scratch."
        rm -rf "./run_${CURRENT_IDX}"
    else
        echo "Error: run_${CURRENT_IDX} already has a checkpoint ($(pwd)/run_${CURRENT_IDX})." >&2
        echo "Refusing to touch it. Set RESUME=1 to continue it, or FORCE=1 to retrain it," >&2
        echo "or use a different MODEL_SUFFIX / SBATCH_ARRAY to write somewhere else." >&2
        exit 1
    fi
fi

mkdir -p "./run_${CURRENT_IDX}"
cd "./run_${CURRENT_IDX}"

python ../../../src/main.py \
    --h5_file "$H5_FILE" \
    --run_idx "$CURRENT_IDX" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --input_dim "${INPUT_DIMS[$DATA_IDX]}" \
    --hidden_dim "$HIDDEN_DIM" \
    $RESUME_FLAG

echo "Job finished for run_${CURRENT_IDX} at $(date)"
