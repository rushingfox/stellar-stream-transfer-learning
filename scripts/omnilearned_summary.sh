module load conda
mamba activate stream_transfer_learning

dataset_sizes_default=(300 1000 10000)
if [[ -n "${DATASET_SIZE:-}" ]]; then
    read -r -a dataset_sizes <<< "${DATASET_SIZE}"
else
    dataset_sizes=("${dataset_sizes_default[@]}")
fi
proxy_pretrain_size=${PROXY_PRETRAIN_SIZE:-10000}

# Finetune
target_columns_default=(
    "3d_x_y_z"
)
target_models_default=(
    "omnilearned_scratch_1x4"
    "omnilearned_finetune_1x4"
    "omnicosmos_camels_finetune_1x4"
    "omnicosmos_quijote_finetune_1x4"
    "omnilearned_proxy_cube_finetune_1x4"
    "omnilearned_proxy_bounded_ball_finetune_1x4"
    "omnilearned_proxy_gaussian_ball_finetune_1x4"
    "omnilearned_proxy_gaussian_ellipsoid_sigmax_finetune_1x4"
    "omnilearned_proxy_bounded_ellipsoid_a_finetune_1x4"
    "omnilearned_proxy_box_a_finetune_1x4"
    "omnilearned_proxy_eigen_lambda1_finetune_1x4"
)

# Proxy pretrain (regression) — the seven unified 1-target proxies (six
# geometric + stream eigenvalues), all on the 1x4 architecture.
proxy_pretrain_columns_default=(
    "3d_proxy_cube"
    "3d_proxy_bounded_ball"
    "3d_proxy_gaussian_ball"
    "3d_proxy_gaussian_ellipsoid_sigmax"
    "3d_proxy_bounded_ellipsoid_a"
    "3d_proxy_box_a"
    "3d_x_y_z_eigenvalues_lambda1"
)
proxy_pretrain_models_default=(
    "omnilearned_proxy_pretrain_1x4"
)

run_indices_default=(0 1 2)

if [[ -n "${TARGET_COLUMNS:-}" ]]; then
    read -r -a target_columns <<< "${TARGET_COLUMNS}"
else
    target_columns=("${target_columns_default[@]}")
fi

if [[ -n "${TARGET_MODELS:-}" ]]; then
    read -r -a target_models <<< "${TARGET_MODELS}"
else
    target_models=("${target_models_default[@]}")
fi

if [[ -n "${PROXY_PRETRAIN_COLUMNS:-}" ]]; then
    read -r -a proxy_pretrain_columns <<< "${PROXY_PRETRAIN_COLUMNS}"
else
    proxy_pretrain_columns=("${proxy_pretrain_columns_default[@]}")
fi

if [[ -n "${PROXY_PRETRAIN_MODELS:-}" ]]; then
    read -r -a proxy_pretrain_models <<< "${PROXY_PRETRAIN_MODELS}"
else
    proxy_pretrain_models=("${proxy_pretrain_models_default[@]}")
fi

if [[ -n "${RUN_IDXS:-}" ]]; then
    read -r -a run_indices <<< "${RUN_IDXS}"
else
    run_indices=("${run_indices_default[@]}")
fi

run_summary() {
    local run_dir="$1"
    if [ ! -d "$run_dir" ]; then
        echo "Warning: Directory ${run_dir} does not exist. Skipping."
        return
    fi
    local results_dir="${run_dir}/results"
    mkdir -p "$results_dir"
    python ../src/omnilearned_summary.py \
        --json_path "${run_dir}/training_.json" \
        --npz_glob "${run_dir}/outputs__streams_*.npz" \
        --output_csv "${results_dir}/loss_history.csv"
}

# Finetune loop
for dataset_size in "${dataset_sizes[@]}"; do
    for col in "${target_columns[@]}"; do
        for model in "${target_models[@]}"; do
            for idx in "${run_indices[@]}"; do
                run_summary "../${col}_${dataset_size}/${model}/run_${idx}"
            done
        done
    done
done

# Proxy pretrain loop (regression)
for col in "${proxy_pretrain_columns[@]}"; do
    for model in "${proxy_pretrain_models[@]}"; do
        for idx in "${run_indices[@]}"; do
            run_summary "../${col}_${proxy_pretrain_size}/${model}/run_${idx}"
        done
    done
done
