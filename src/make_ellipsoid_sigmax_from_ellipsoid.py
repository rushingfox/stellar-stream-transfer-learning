"""Derives the ``3d_proxy_gaussian_ellipsoid_sigmax`` proxy from ``3d_proxy_ellipsoid``.

The point clouds are reused unchanged, and the target is sliced from ``(N, 3)`` to ``(N, 1)``, keeping only column 0, sigma_x, the longest axis.
This makes it a fair counterpart of ``3d_proxy_gaussian_ball`` (isotropic Gaussian, 1-target sigma): both are 1D regression tasks that differ only in the input distribution, isotropic vs anisotropic and rotated.
Copying the point clouds duplicates them on disk, but HDF5 external links, which would cost almost nothing, are flaky with the DDP dataloader.
"""

from pathlib import Path

import h5py
import numpy as np

from exp_paths import INPUT_STATS_NPZ

REPO_ROOT = Path(__file__).resolve().parent.parent

SRC_ROOT = REPO_ROOT / "3d_proxy_ellipsoid_10000/omnilearned_proxy_pretrain/data/streams"
DST_ROOT = REPO_ROOT / "3d_proxy_gaussian_ellipsoid_sigmax_10000/omnilearned_proxy_pretrain/data/streams"


def slice_split(split: str) -> None:
    """Writes one split of the gaussian_ellipsoid_sigmax dataset: the point clouds unchanged, the target ('pid') cut to column 0, and file_index.npy.

    Returns nothing.

    `split`: 'train', 'val' or 'test'.
    """
    src_dir = SRC_ROOT / split
    dst_dir = DST_ROOT / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_h5 = src_dir / f"{split}_streams.hdf5"
    dst_h5 = dst_dir / f"{split}_streams.hdf5"

    with h5py.File(src_h5, "r") as f_in, h5py.File(dst_h5, "w") as f_out:
        f_out.create_dataset(
            "data", data=f_in["data"][:], compression="gzip"
        )
        # Keep only sigma_x, column 0, the longest axis.
        # It is already z-scored with the train-split mean and std of sigma_x from the source ellipsoid dataset.
        f_out.create_dataset(
            "pid", data=f_in["pid"][:, 0:1], compression="gzip"
        )
        print(f"  {split}: data {f_out['data'].shape}, pid {f_out['pid'].shape}")

    # file_index.npy is copied as is, since the indices do not change.
    src_idx = src_dir / "file_index.npy"
    if src_idx.exists():
        (dst_dir / "file_index.npy").write_bytes(src_idx.read_bytes())


def slice_stats() -> None:
    """Writes gaussian_ellipsoid_sigmax_stats.npz, the sigma_x column of the source ellipsoid_stats.npz, and copies input_stats.npz.

    Returns nothing.
    """
    src_stats = SRC_ROOT / "ellipsoid_stats.npz"
    dst_stats = DST_ROOT / "gaussian_ellipsoid_sigmax_stats.npz"
    old = np.load(src_stats)
    # sigma_x is column 0 of the 3-column stats.
    np.savez(dst_stats, mean=old["mean"][:1], std=old["std"][:1])
    print(f"stats: mean={old['mean'][:1]}, std={old['std'][:1]} -> {dst_stats}")
    # The point clouds are copied unchanged, so their input stats carry over as-is.
    (DST_ROOT / INPUT_STATS_NPZ).write_bytes((SRC_ROOT / INPUT_STATS_NPZ).read_bytes())


def main() -> None:
    """Builds the gaussian_ellipsoid_sigmax dataset from the ellipsoid dataset, all three splits and the stats; refuses to run if the output already exists.

    Returns nothing.
    """
    # Never write over an existing output; input_stats.npz is copied last, so an interrupted run leaves data the training scripts refuse.
    if DST_ROOT.exists():
        raise SystemExit(f"{DST_ROOT} already exists -- delete it to regenerate.")
    DST_ROOT.mkdir(parents=True)
    for split in ["train", "val", "test"]:
        slice_split(split)
    slice_stats()
    print(f"\nDone. Output: {DST_ROOT}")


if __name__ == "__main__":
    main()
