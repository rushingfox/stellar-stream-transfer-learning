"""Derive ``3d_proxy_bounded_ellipsoid_a`` from ``3d_proxy_hard_ellipsoid``.

Reuses the point-cloud data and just slices the label from ``(N, 3)`` to
``(N, 1)`` keeping only the first column (a = largest semi-axis).

Motivation: fair comparison against ``3d_proxy_bounded_ball`` (isotropic
hard-boundary, 1-target radius). Both proxies become 1-D regression tasks;
the difference is purely input distribution (isotropic vs anisotropic +
rotated). Mirrors ``make_ellipsoid_sigmax_from_ellipsoid.py`` exactly.

Cost: ~430 MB disk duplication (point clouds copied). Alternative HDF5
external links would be near-zero disk but flaky with the DDP dataloader,
so we prefer the simple copy.
"""

from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

SRC_ROOT = REPO_ROOT / "3d_proxy_hard_ellipsoid_10000/omnilearned_proxy_pretrain_1x4/data/streams"
DST_ROOT = REPO_ROOT / "3d_proxy_bounded_ellipsoid_a_10000/omnilearned_proxy_pretrain_1x4/data/streams"


def slice_split(split: str) -> None:
    src_dir = SRC_ROOT / split
    dst_dir = DST_ROOT / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_h5 = src_dir / f"{split}_streams.hdf5"
    dst_h5 = dst_dir / f"{split}_streams.hdf5"

    with h5py.File(src_h5, "r") as f_in, h5py.File(dst_h5, "w") as f_out:
        f_out.create_dataset(
            "data", data=f_in["data"][:], compression="gzip"
        )
        # Keep only 'a' (largest semi-axis, column 0). Already z-scored using
        # the train-set 'a' mean/std of the source hard_ellipsoid dataset.
        f_out.create_dataset(
            "pid", data=f_in["pid"][:, 0:1], compression="gzip"
        )
        n = f_out["data"].shape[0]
        print(f"  {split}: data {f_out['data'].shape}, pid {f_out['pid'].shape}")

    # file_index.npy: preserved as-is (indexes are unchanged).
    src_idx = src_dir / "file_index.npy"
    if src_idx.exists():
        (dst_dir / "file_index.npy").write_bytes(src_idx.read_bytes())


def slice_stats() -> None:
    src_stats = SRC_ROOT / "hard_ellipsoid_stats.npz"
    dst_stats = DST_ROOT / "bounded_ellipsoid_a_stats.npz"
    old = np.load(src_stats)
    # 'a' is column 0 of the 3-D stats.
    np.savez(dst_stats, mean=old["mean"][:1], std=old["std"][:1])
    print(f"stats: mean={old['mean'][:1]}, std={old['std'][:1]} -> {dst_stats}")


def main() -> None:
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        slice_split(split)
    slice_stats()
    print(f"\nDone. Output: {DST_ROOT}")


if __name__ == "__main__":
    main()
