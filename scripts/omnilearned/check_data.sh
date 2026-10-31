#!/bin/bash
# Sourced by every OmniLearned submitter and runner: refuses to train on data
# that is not in the normalized form the data-preparation scripts write, so a
# missed regeneration step stops a job instead of training on stale data.

# $1: a downstream <task>_<target_columns>_<size>/data dir, written by
# scripts/prepare_data_omnilearned.sh.
require_downstream_data() {
    local data_dir="$1" need
    for need in streams/train/train_streams.hdf5 streams/val/val_streams.hdf5 \
            streams/test/test_streams.hdf5 streams/input_stats.npz streams/target_stats.npz \
            shared_test10000/streams/test/test_streams.hdf5; do
        if [[ ! -f "${data_dir}/${need}" ]]; then
            echo "Error: ${data_dir}/${need} not found; (re)generate the data with scripts/prepare_data_omnilearned.sh" >&2
            return 1
        fi
    done
}

# $1: a proxy's <proxy>_10000/omnilearned_proxy_pretrain/data dir, written by
# scripts/prepare_data_proxy_pretrain.sh (plus a slicer or the eigen script).
require_proxy_data() {
    local data_dir="$1" need
    for need in streams/train/train_streams.hdf5 streams/val/val_streams.hdf5 \
            streams/test/test_streams.hdf5 streams/input_stats.npz; do
        if [[ ! -f "${data_dir}/${need}" ]]; then
            echo "Error: ${data_dir}/${need} not found; prepare this proxy's normalized 1-target data first" >&2
            return 1
        fi
    done
}
