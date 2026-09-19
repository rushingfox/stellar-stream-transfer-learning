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
H5_FILE=${H5_FILE:-../data/baseline_dataset.h5}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
HIDDEN_DIM=${HIDDEN_DIM:-128}

echo "Starting Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running experiment with run_idx: ${CURRENT_IDX}"
echo "Using hidden_dim: ${HIDDEN_DIM}"

mkdir -p "./run_${CURRENT_IDX}"
cd "./run_${CURRENT_IDX}"

RESUME_FLAG=""
if [[ -f "checkpoints/checkpoint_latest.pth" ]]; then
    echo "Found existing checkpoint, will resume training."
    RESUME_FLAG="--resume"
fi

python ../../../src/main.py \
    --h5_file "$H5_FILE" \
    --run_idx "$CURRENT_IDX" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --input_dim "${INPUT_DIMS[$DATA_IDX]}" \
    --hidden_dim "$HIDDEN_DIM" \
    $RESUME_FLAG

echo "Job finished for run_${CURRENT_IDX} at $(date)"
