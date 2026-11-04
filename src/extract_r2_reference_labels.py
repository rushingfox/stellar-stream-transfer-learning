"""Extracts the test-split targets of a raw dataset, for the R^2 reference variance Var(y_test).

The output, a small sidecar file, stands in for the full raw dataset, so evaluation/reproduce_main_figure.ipynb can rebuild the paper's R^2 figure from the per-run bootstrap.csv files alone.
The test split matches prepare_data.py and dataset.py exactly: the last 20% of the raw dataset (val_end = int(N * 0.8), test = pid[val_end:]).
Anyone who downloads the raw data (e.g. from the paper's Zenodo record) can regenerate the file with the command below.

Usage:
    python src/extract_r2_reference_labels.py \
        --input_raw raw_data/prog_mass_reg_dataset_10000.h5 \
        --output evaluation/prog_mass_labels_10000_test.npy
"""
import argparse

import h5py
import numpy as np


def parse_args():
    """Reads the raw dataset and the output path from the command line.

    Returns the argparse namespace.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input_raw', type=str, required=True,
                    help="Path to the raw prog_mass_reg_dataset_<N>.h5 file")
    p.add_argument('--output', type=str, required=True,
                    help="Output .npy path for the extracted test-split labels")
    return p.parse_args()


def main():
    """Saves the raw test-split targets (pid) of --input_raw to --output as .npy and prints their variance.

    Returns nothing.
    """
    args = parse_args()
    with h5py.File(args.input_raw, 'r') as f:
        y = f['pid'][:].ravel()
    val_end = int(y.shape[0] * 0.8)
    test_labels = y[val_end:]
    np.save(args.output, test_labels)
    print(f"Extracted {test_labels.shape[0]} test-split labels from {args.input_raw} -> {args.output}")
    print(f"Var(y_test) = {np.var(test_labels):.6f}")


if __name__ == "__main__":
    main()
