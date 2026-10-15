#!/bin/bash
#SBATCH -A m4474
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=raw_data_${NUM_STREAMS}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jianhao.wu@wisc.edu

echo "===== Script Content ====="
cat $0
echo "===== End of Script ====="
echo ""
echo "===== Job Info ====="
echo "Job ID          : $SLURM_JOB_ID"
echo "Node            : $SLURMD_NODENAME"
echo "CPUs per task   : $SLURM_CPUS_PER_TASK"
echo "NUM_STREAMS     : $NUM_STREAMS"
echo "TASK            : ${TASK:-prog_mass}"
echo "Start time      : $(date)"
echo ""

# TASK picks the dataset: prog_mass (progenitor mass, the default) or subhalo
# (subhalo-perturbed GD-1-like streams, label log10 M(<10 pc)).
case "${TASK:-prog_mass}" in
    prog_mass) gen_script=raw_data_generation.py;         output_prefix=prog_mass_reg_dataset ;;
    subhalo)   gen_script=raw_data_generation_subhalo.py; output_prefix=subhalo_reg_dataset ;;
    *) echo "Error: unknown TASK='${TASK}' (expected prog_mass or subhalo)" >&2; exit 1 ;;
esac

module load conda
mamba activate stream_transfer_learning

cd $SLURM_SUBMIT_DIR

# ---- Threading settings ------------------------------------------------
# Let Python multiprocessing use all allocated cores.
# Set OMP/MKL/OpenBLAS to 1 so individual workers don't over-subscribe.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# HDF5 locking off (required on Lustre / NERSC scratch)
export HDF5_USE_FILE_LOCKING=FALSE

python ../src/${gen_script} \
    --num_streams $NUM_STREAMS \
    --output ${output_prefix}_${NUM_STREAMS}.h5
# SLURM_CPUS_PER_TASK is read automatically by the script to set num_workers.
# You can override it explicitly with: --num_workers 64

echo ""
echo "===== End time: $(date) ====="
