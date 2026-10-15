#!/bin/bash
# submit_jobs.sh
# Usage: bash prepare_raw_data_submission.sh
# Submits one Slurm job per value of NUM_STREAMS.
# TASK=subhalo generates the subhalo-perturbed dataset instead of the
# progenitor-mass one (TASK=prog_mass by default); see prepare_raw_data.sh.

cd ../raw_data
mkdir -p logs

declare -A TIME_LIMITS
TIME_LIMITS[100]="00:30:00"
TIME_LIMITS[300]="00:30:00"
TIME_LIMITS[1000]="1:00:00"
TIME_LIMITS[10000]="3:00:00"

num_streams_default=(300 1000 10000)
if [[ -n "${NUM_STREAMS_LIST:-}" ]]; then
    read -r -a num_streams_list <<< "${NUM_STREAMS_LIST}"
else
    num_streams_list=("${num_streams_default[@]}")
fi

task=${TASK:-prog_mass}

for n in "${num_streams_list[@]}"; do
    echo "Submitting job for TASK=$task NUM_STREAMS=$n  (time limit: ${TIME_LIMITS[$n]}) ..."
    sbatch --time=${TIME_LIMITS[$n]} \
           --job-name=raw_data_${task}_${n} \
           --export=ALL,NUM_STREAMS=$n,TASK=$task \
           ../scripts/prepare_raw_data.sh
done
