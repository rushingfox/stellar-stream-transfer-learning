#!/bin/bash
# Packs the per-run results the figure notebook reads into results_archive.tar.gz at the repo root.
# Contents: every run's .csv, .npz, .json and invocation.txt, plus each size's target_stats.npz and input_stats.npz.
# Unpack with `tar -xzf results_archive.tar.gz` at the repo root.
# The output is deterministic, so unchanged results give a byte-identical archive.
# Rerun and commit the archive whenever results change.
#
# Usage: bash scripts/make_results_archive.sh
# Env overrides: TASK (default prog_mass), DATASET_SIZES (default "40 100 300 1000")

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source "${script_dir}/task_config.sh"

target_columns="3d_x_y_z"
archive="results_archive.tar.gz"

if [[ -n "${DATASET_SIZES:-}" ]]; then
    read -r -a dataset_sizes <<< "${DATASET_SIZES}"
else
    dataset_sizes=(40 100 300 1000)
fi

cd "${repo_root}"

dirs=()
for size in "${dataset_sizes[@]}"; do
    dir="${task_dir_prefix}${target_columns}_${size}"
    if [[ ! -d "${dir}" ]]; then
        echo "Error: ${dir}/ not found under ${repo_root}" >&2
        exit 1
    fi
    dirs+=("${dir}")
done

file_list="$(mktemp)"
trap 'rm -f "${file_list}"' EXIT

# From data/ only the two stats files.
find "${dirs[@]}" -type f ! -path '*/.ipynb_checkpoints/*' \( \
        \( -path '*/data/*' \( -name target_stats.npz -o -name input_stats.npz \) \) \
     -o \( ! -path '*/data/*' \( -name '*.csv' -o -name '*.npz' -o -name '*.json' -o -name invocation.txt \) \) \
    \) | LC_ALL=C sort > "${file_list}"

n_files=$(wc -l < "${file_list}")
if [[ "${n_files}" -eq 0 ]]; then
    echo "Error: no result files found in ${dirs[*]}" >&2
    exit 1
fi

tar --create --no-recursion --files-from="${file_list}" \
    --sort=name --mtime='2026-01-01 00:00:00 UTC' \
    --owner=0 --group=0 --numeric-owner --mode='u+rw,go+r' \
    | gzip -n -9 > "${archive}"

# stat, since du reports 0 on CFS for a fresh file.
echo "Wrote ${repo_root}/${archive}: ${n_files} files from ${dirs[*]}, $(numfmt --to=si --suffix=B "$(stat -c %s "${archive}")")"
