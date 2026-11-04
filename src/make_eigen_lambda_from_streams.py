"""Builds the ``3d_x_y_z_eigenvalues_lambda{K}`` proxy (K = 1, 2 or 3) directly from the shared real-stream data.

The input point clouds are exactly the 10000-size finetune data (``prog_mass_3d_x_y_z_10000/data/streams``), the same streams in the same train/val/test split; only the target changes.
Those inputs are already normalized with that train split's input stats, which are this proxy's own train stats too, so they are copied as is, together with input_stats.npz.
The target is the K-th largest PCA spatial scale, sqrt(lambda_K), in kpc, computed from the raw coordinates in --input_raw rather than from the normalized inputs (np.cov, eigvalsh, sorted descending, sqrt).
It is z-scored with train-split statistics.
lambda1 ~ stream length, lambda2 / lambda3 ~ width / thickness.

    python src/make_eigen_lambda_from_streams.py --index 2
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from exp_paths import INPUT_STATS_NPZ

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "prog_mass_3d_x_y_z_10000/data/streams"
RAW_H5 = REPO_ROOT / "raw_data/prog_mass_reg_dataset_10000.h5"
SPLITS = ("train", "val", "test")


def sqrt_eigenvalues(points):
    """Computes each stream's PCA spatial scales, the square roots of its covariance eigenvalues.

    Returns an (n_streams, 3) array, sorted descending within each stream, in the units of `points`.

    `points`: an (n_streams, n_points, 3) array of coordinates.
    """
    out = np.empty((points.shape[0], 3))
    for i, pts in enumerate(points):
        centered = pts - pts.mean(axis=0)
        eigenvalues = np.linalg.eigvalsh(np.cov(centered.T))
        out[i] = np.sqrt(np.sort(eigenvalues)[::-1])
    return out


def main():
    """Writes the proxy dataset for --index K under --out_root: the three splits, the target stats and input_stats.npz; refuses to overwrite an existing one.

    Returns nothing.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=int, choices=(1, 2, 3), required=True, help="which lambda: 1 = largest")
    p.add_argument("--out_root", type=Path, default=REPO_ROOT,
                   help="repo-like root to write into (default: the repo itself)")
    p.add_argument("--input_raw", type=Path, default=RAW_H5,
                   help="raw HDF5 the 10000-size finetune data was made from (default: %(default)s)")
    args = p.parse_args()

    k = args.index
    dst_root = args.out_root / f"3d_x_y_z_eigenvalues_lambda{k}_10000/omnilearned_proxy_pretrain/data/streams"
    if dst_root.exists():
        raise SystemExit(f"{dst_root} already exists -- refusing to overwrite it.")

    scales = {}
    with h5py.File(args.input_raw, "r") as raw:
        for split in SPLITS:
            with h5py.File(SRC_ROOT / split / f"{split}_streams.hdf5", "r") as f:
                idx = f["original_idx"][:]
            # Each split is one contiguous block of the raw streams.
            assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)), f"{split}: original_idx is not contiguous"
            scales[split] = sqrt_eigenvalues(raw["Cartesian_data"][idx[0]:idx[-1] + 1, :, [0, 1, 2]])
            print(f"  {split}: {scales[split].shape[0]} streams")

    # z-score with train-split statistics only (population std).
    # They are taken over all three columns and then indexed, the same order of operations as prepare_data.py uses for its 3-target proxies.
    mean = scales["train"].mean(axis=0)[k - 1]
    std = scales["train"].std(axis=0)[k - 1]

    for split in SPLITS:
        src_dir, dst_dir = SRC_ROOT / split, dst_root / split
        dst_dir.mkdir(parents=True)
        with h5py.File(src_dir / f"{split}_streams.hdf5", "r") as f_in, \
                h5py.File(dst_dir / f"{split}_streams.hdf5", "w") as f_out:
            f_out.create_dataset("data", data=f_in["data"][:], compression="gzip")
            z = ((scales[split][:, k - 1] - mean) / std).astype(np.float32).reshape(-1, 1)
            f_out.create_dataset("pid", data=z, compression="gzip")
        (dst_dir / "file_index.npy").write_bytes((src_dir / "file_index.npy").read_bytes())

    np.savez(dst_root / f"eigenvalue_lambda{k}_stats.npz", mean=np.array([mean]), std=np.array([std]))
    (dst_root / INPUT_STATS_NPZ).write_bytes((SRC_ROOT / INPUT_STATS_NPZ).read_bytes())
    print(f"sqrt(lambda{k}) train mean={mean:.4f} std={std:.4f}\nDone: {dst_root}")


if __name__ == "__main__":
    main()
