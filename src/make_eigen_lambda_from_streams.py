"""Build the ``3d_x_y_z_eigenvalues_lambda{K}`` proxy (K = 1, 2 or 3) directly from
the shared real-stream data.

Input point clouds are exactly the 10000-size finetune data
(``3d_x_y_z_10000/data/streams``), same streams and same train/val/test split;
only the label changes. The label is the K-th largest PCA spatial scale,
sqrt(lambda_K), computed exactly as ``prepare_data.py``'s
``3d_x_y_z_eigenvalues`` branch does (np.cov, eigvalsh, sorted descending,
sqrt) and z-scored with train-split statistics. lambda1 ~ stream length,
lambda2 / lambda3 ~ width / thickness.

No 3-target ``3d_x_y_z_eigenvalues`` parent dataset is needed; for K=1 the
output is byte-identical to the one the paper's lambda1 proxy was trained on.

    python src/make_eigen_lambda_from_streams.py --index 2
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "3d_x_y_z_10000/data/streams"
SPLITS = ("train", "val", "test")


def sqrt_eigenvalues(points):
    """(n_streams, n_points, 3) -> (n_streams, 3), sqrt(lambda) sorted descending."""
    out = np.empty((points.shape[0], 3))
    for i, pts in enumerate(points):
        centered = pts - pts.mean(axis=0)
        eigenvalues = np.linalg.eigvalsh(np.cov(centered.T))
        out[i] = np.sqrt(np.sort(eigenvalues)[::-1])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=int, choices=(1, 2, 3), required=True, help="which lambda: 1 = largest")
    p.add_argument("--out_root", type=Path, default=REPO_ROOT,
                   help="repo-like root to write into (default: the repo itself)")
    args = p.parse_args()

    k = args.index
    dst_root = args.out_root / f"3d_x_y_z_eigenvalues_lambda{k}_10000/omnilearned_proxy_pretrain_1x4/data/streams"
    if dst_root.exists():
        raise SystemExit(f"{dst_root} already exists -- refusing to overwrite it.")

    scales = {}
    for split in SPLITS:
        with h5py.File(SRC_ROOT / split / f"{split}_streams.hdf5", "r") as f:
            scales[split] = sqrt_eigenvalues(f["data"][:])
        print(f"  {split}: {scales[split].shape[0]} streams")

    # z-score with train-split statistics only (population std). Taken over all
    # three columns and then indexed -- the same operation order as
    # prepare_data.py, so even the stats file is bit-identical for K=1.
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
    print(f"sqrt(lambda{k}) train mean={mean:.4f} std={std:.4f}\nDone: {dst_root}")


if __name__ == "__main__":
    main()
