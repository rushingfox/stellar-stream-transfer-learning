#!/bin/bash
#SBATCH -A m4474
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --qos=shared
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jianhao.wu@wisc.edu

echo "===== Script Content ====="
cat $0
echo "===== End of Script ====="

module load conda
mamba activate stream_transfer_learning

submit_dir=${SLURM_SUBMIT_DIR:-$PWD}
cd "$submit_dir"

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DATA_IDX=${1:?Error: please provide index 0/1/2/3/4/5}
common_script="${submit_dir}/../../scripts/run_baseline_train_common.sh"
if [[ ! -f "$common_script" ]]; then
    echo "Error: cannot find helper script at ${common_script}" >&2
    exit 1
fi

HIDDEN_DIM=256 bash "$common_script" "$DATA_IDX"
