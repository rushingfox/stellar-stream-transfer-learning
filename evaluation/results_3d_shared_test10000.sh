#!/bin/bash
#SBATCH -A m4474
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=jobout.o%j
#SBATCH --error=joberr.o%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jianhao.wu@wisc.edu

# Aggregates each run's shared-test10000 CSV (produced by
# scripts/eval_shared_test10000.sh + eval_shared_test10000_nonML.sh), so every
# model is scored against one fixed test set rather than its own split.

echo "===== Script Content ====="
cat $0
echo "===== End of Script ====="

module load conda
mamba activate stream_transfer_learning
# Must load AFTER conda activation -- conda's own (incomplete) texlive-core,
# if ever reinstalled, would otherwise take PATH priority.
module load texlive/2024

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults reproduce the paper's main figure (R^2, bootstrap error bars,
# nested stat+syst decomposition) with zero flags. Pass e.g. METRIC=mse
# BOOTSTRAP=false NESTED_ERROR=false to get the plain cross-seed-std MSE view
# instead.
bootstrap=${BOOTSTRAP:-true}
# NESTED_ERROR=true draws two error bars per point (stat (+) syst, in
# quadrature): outer = total, inner = mean bootstrap std across seeds. It
# always reads from the same shared_test<N>_bootstrap/loss_history.csv as
# BOOTSTRAP=true (so it implies bootstrap=true), but -- unlike plain
# BOOTSTRAP=true, which fixes a single run_idx -- it needs every seed's own
# point estimate (for the cross-seed/"systematic" spread) AND every seed's
# own bootstrap std (for the "statistical" component), so it needs bootstrap
# data precomputed for run_0/1/2, not just run_0 (see
# scripts/compute_bootstrap_shared_test_all.sh RUN_IDXS="0 1 2").
nested_error=${NESTED_ERROR:-true}
[[ "${nested_error}" == "true" ]] && bootstrap=true
# The appendix R^2 table is identical no matter how the plot's y-axis is
# cropped, so a second pass that only changes the zoom (R2_YMIN=...) can skip
# rewriting it with LATEX_TABLE=false.
latex_table=${LATEX_TABLE:-true}

if [[ -n "${RUN_IDXS:-}" ]]; then
    read -r -a run_idxs <<< "${RUN_IDXS}"
elif [[ "${nested_error}" == "true" ]]; then
    run_idxs=(0 1 2)
elif [[ "${bootstrap}" == "true" ]]; then
    run_idxs=(0)  # bootstrap fixes a single trained model, not a cross-seed statistic
else
    run_idxs=(0 1 2)
fi

eval_dataset_size=${EVAL_DATASET_SIZE:-10000}
if [[ "${bootstrap}" == "true" ]]; then
    test_csv_name="shared_test${eval_dataset_size}_bootstrap/loss_history.csv"
else
    test_csv_name="shared_test${eval_dataset_size}/loss_history.csv"
fi

if [[ -n "${DATASET_SIZES:-}" ]]; then
    read -r -a agg_dataset_sizes <<< "${DATASET_SIZES}"
else
    # Paper only reports 300/1000/10000 -- 100 (and any smaller exploratory
    # size) scatters too much to support a conclusion on its own, so it's
    # opt-in via DATASET_SIZES rather than cluttering the default figure.
    agg_dataset_sizes=(300 1000 10000)
fi

metric=${METRIC:-r2}

if [[ -n "${MODELS:-}" ]]; then
    read -r -a models <<< "${MODELS}"
else
    models=()
    for dataset_size in "${agg_dataset_sizes[@]}"; do
        models+=(
            "3d_x_y_z:${dataset_size}:omnicosmos_camels_finetune_1x4:OmniCosmos_CAMELS"
            "3d_x_y_z:${dataset_size}:omnicosmos_quijote_finetune_1x4:OmniCosmos_Quijote"
            "3d_x_y_z:${dataset_size}:baseline_256:DeepSets_256"
            "3d_x_y_z:${dataset_size}:eigenvalue_regression"
            "3d_x_y_z:${dataset_size}:omnilearned_scratch_1x4:OmniLearned_Scratch"
            "3d_x_y_z:${dataset_size}:omnilearned_finetune_1x4:OmniLearned_Finetune"
            "3d_x_y_z:${dataset_size}:omnilearned_proxy_cube_finetune_1x4:proxy_cube"
            "3d_x_y_z:${dataset_size}:omnilearned_proxy_bounded_ball_finetune_1x4:proxy_bounded_ball"
            "3d_x_y_z:${dataset_size}:omnilearned_proxy_gaussian_ball_finetune_1x4:proxy_gaussian_ball"
            "3d_x_y_z:${dataset_size}:omnilearned_proxy_gaussian_ellipsoid_sigmax_finetune_1x4:proxy_gaussian_ellipsoid_sigmax"
            "3d_x_y_z:${dataset_size}:omnilearned_proxy_bounded_ellipsoid_a_finetune_1x4:proxy_bounded_ellipsoid_a"
            "3d_x_y_z:${dataset_size}:omnilearned_proxy_box_a_finetune_1x4:proxy_box_a"
            "3d_x_y_z:${dataset_size}:omnilearned_proxy_eigen_lambda1_finetune_1x4:proxy_stream_eigenvalues_lambda1"
        )
        # "average" (predict-the-training-mean dummy) is only informative
        # under MSE -- under R^2 it's ~0 by construction (R^2 = 1 - MSE/Var),
        # so it's excluded from the default r2 model list rather than plotting
        # a content-free point every time. Pass MODELS= explicitly to include it.
        if [[ "${metric}" != "r2" ]]; then
            models+=("3d_x_y_z:${dataset_size}:average")
        fi
    done
fi

yscale=${YSCALE:-log}
relative_to_scratch=${RELATIVE_TO_SCRATCH:-false}
r2_ymin=${R2_YMIN:-0}

# Only the plot filename gets suffixed -- CSV content doesn't depend on
# yscale/metric/relative_to_scratch/r2_ymin. r2 and relative_to_scratch both
# force linear internally, so their suffixes subsume the yscale one (no
# "_linear_r2" double-suffix).
if [[ "${metric}" == "r2" ]]; then
    scale_suffix="_r2"
    # Only a non-default r2_ymin gets its own suffix, so the plain "_r2" name
    # (ymin=0, the common case) is untouched -- e.g. R2_YMIN=0.7 -> "_ymin07".
    [[ "${r2_ymin}" != "0" ]] && scale_suffix="${scale_suffix}_ymin${r2_ymin//./}"
else
    scale_suffix=""
    [[ "${yscale}" == "linear" ]] && scale_suffix="_linear"
fi
[[ "${relative_to_scratch}" == "true" ]] && scale_suffix="${scale_suffix}_relscratch"
[[ "${bootstrap}" == "true" ]] && scale_suffix="${scale_suffix}_bootstrap"
[[ "${nested_error}" == "true" ]] && scale_suffix="${scale_suffix}_nested"

output_csv=${OUTPUT_CSV:-results_3d_shared_test${eval_dataset_size}.csv}
output_scatter=${OUTPUT_SCATTER:-results_3d_shared_test${eval_dataset_size}_scatter${scale_suffix}.png}
# Appendix table: R^2 per model x fine-tuning size, bare `tabular`, no caption.
# R^2-only (it IS an R^2 table), so it carries no scale_suffix.
output_tex=${OUTPUT_TEX:-results_3d_shared_test${eval_dataset_size}.tex}

metric_args=()
if [[ "${metric}" == "r2" ]]; then
    metric_args+=(
        --r2_reference_raw "${script_dir}/../raw_data/prog_mass_reg_dataset_${eval_dataset_size}.h5"
        --r2_ymin "${r2_ymin}"
    )
    if [[ "${latex_table}" == "true" ]]; then
        metric_args+=(--latex_table "${script_dir}/${output_tex}")
    fi
fi
if [[ "${relative_to_scratch}" == "true" ]]; then
    # r2+relscratch differences are naturally much smaller-scale than
    # mse+relscratch ones (R^2 is bounded to [0,1] to begin with), so it
    # defaults to a tighter +-10% zoom instead of +-100%.
    if [[ "${metric}" == "r2" ]]; then
        default_relative_ymin="-0.1"
        default_relative_ymax="0.1"
    else
        default_relative_ymin="-1"
        default_relative_ymax="1"
    fi
    metric_args+=(
        --relative_to_scratch
        --relative_baseline_model "${RELATIVE_BASELINE_MODEL:-OmniLearned_Scratch}"
        --relative_ymin "${RELATIVE_YMIN:-${default_relative_ymin}}"
        --relative_ymax "${RELATIVE_YMAX:-${default_relative_ymax}}"
    )
fi
if [[ "${bootstrap}" == "true" ]]; then
    # Error bar comes from compute_bootstrap_shared_test.py's precomputed
    # per-run column instead of the usual cross-seed std.
    metric_args+=(--error_column Test_Loss_Bootstrap_Std)
fi
[[ "${nested_error}" == "true" ]] && metric_args+=(--combine_errors)

line_args=()
[[ "${CONNECT_LINES:-false}" == "true" ]] && line_args+=(--connect_lines)

python "${script_dir}/../src/build_results_table.py" \
  --repo_root "${script_dir}/.." \
  --indices "${run_idxs[@]}" \
  --output "${script_dir}/${output_csv}" \
  --scatter_plot "${script_dir}/${output_scatter}" \
  --test_csv_name "${test_csv_name}" \
  --yscale "${yscale}" \
  --metric "${metric}" \
  "${metric_args[@]}" \
  "${line_args[@]}" \
  --models \
  "${models[@]}"
