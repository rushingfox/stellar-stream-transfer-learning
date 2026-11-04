"""Prepares one dataset's training data: the real streams or a synthesized proxy for OmniLearned, or the real streams for DeepSets.

The OmniLearned data is split 3:1:1 into train/val/test in stream order, with inputs normalized and targets z-scored with train-split statistics (the real streams' test target stays raw).
It is run by scripts/prepare_data_omnilearned.sh, scripts/prepare_data_proxy_pretrain.sh and scripts/prepare_data_deepsets.sh.
"""
import argparse
import h5py
import numpy as np
from pathlib import Path
import time
import datetime
from tqdm import tqdm
import sys
import zlib
from exp_paths import (DEFAULT_TASK, EVAL_DATASET_SIZE, INPUT_STATS_NPZ, TARGET_STATS_NPZ, TASKS, VALID_DATASET_SIZES,
                       compute_input_stats, dataset_key, get_data_dir, load_input_stats, shared_test_dirname)

# The --target_columns of the real streams (the downstream tasks, not the proxies), whose inputs are the x, y, z columns of the raw Cartesian_data.
REAL_STREAM_COLUMNS = ('3d_x_y_z',)
XYZ_COLUMNS = [0, 1, 2]


def parse_args():
    """Parses the command line, and exits for a DeepSets request on a proxy, which has no DeepSets data.

    Returns the argparse Namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_raw', type=str, required=True)
    parser.add_argument('--output_root', type=str, default="..", help="Repository root for output datasets")
    parser.add_argument('--dataset_size', type=int, choices=VALID_DATASET_SIZES, required=True)
    parser.add_argument('--target_model', type=str, choices=['deepsets', 'omnilearned'], required=True)
    parser.add_argument('--output_model', type=str, default="", help="Optional model folder name override for saving outputs")
    parser.add_argument('--task', type=str, choices=list(TASKS), default=DEFAULT_TASK,
                        help="Regression task: sets the output folder prefix and the DeepSets label key")
    parser.add_argument('--target_columns', type=str, choices=['3d_x_y_z', '3d_proxy_ellipsoid', '3d_proxy_cube', '3d_proxy_bounded_ball', '3d_proxy_gaussian_ball', '3d_proxy_hard_ellipsoid', '3d_proxy_box'], required=True)
    parser.add_argument('--shared_test_raw', type=str, default=None,
                        help=f"Raw HDF5 of the size-{EVAL_DATASET_SIZE} dataset. With a real-stream "
                             f"--target_columns, also writes its test split, with inputs normalized with "
                             f"this dataset's input stats, to data/{shared_test_dirname()}/ for the "
                             f"shared-test evaluation")
    parser.add_argument('--seed', type=int, default=0,
                        help="base seed of the proxy synthesis, combined with the proxy's tag; the real streams ignore it")

    args = parser.parse_args()
    if args.target_model == 'deepsets' and args.target_columns not in REAL_STREAM_COLUMNS:
        print(f"[INFO] Skipping: DeepSets data exists only for the real streams, not columns='{args.target_columns}'.")
        sys.exit(0)

    return args


def validate_input_raw_size(input_raw, dataset_size):
    """Checks that the raw HDF5 holds dataset_size streams.

    Returns nothing; raises KeyError if the file has no 'pid' dataset and ValueError if its stream count differs.

    `input_raw`: the raw HDF5 path.
    `dataset_size`: the expected number of streams.
    """
    with h5py.File(input_raw, 'r') as f:
        if 'pid' not in f:
            raise KeyError(f"Dataset 'pid' not found in input file: {input_raw}")
        actual_size = int(f['pid'].shape[0])

    if actual_size != dataset_size:
        raise ValueError(
            f"Input raw size mismatch: expected dataset_size={dataset_size}, "
            f"but found {actual_size} samples in {input_raw}."
        )

def process_deepsets(input_raw, output_dir, label_key):
    """Copies every stream's raw x, y, z coordinates and target into DeepSets' data file.

    Returns nothing; writes <output_dir>/deepsets_dataset.h5.

    Nothing is split or normalized here: dataset.py splits the streams and normalizes them on the fly with train-split statistics.

    `input_raw`: the raw HDF5 path.
    `output_dir`: the data folder.
    `label_key`: the name of the target dataset in the output, from TASKS (e.g. Log_Prog_Mass).
    """
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    output_file = p / "deepsets_dataset.h5"
    with h5py.File(input_raw, 'r') as f:
        raw_data = f['Cartesian_data'][:, :, XYZ_COLUMNS]
        pid_reg_ds = f['pid'][:]

    with h5py.File(output_file, 'w') as out_f:
        out_f.create_dataset('raw_data', data=raw_data, compression="gzip")
        out_f.create_dataset(label_key, data=pid_reg_ds, compression="gzip")
    print(f"Created {output_file} with raw_data of {raw_data.shape} and pid_reg_ds of {pid_reg_ds.shape}")

def standardize_target(y, split, output_dir):
    """z-scores a real-stream target with the train split's mean and population std, which the train split saves to streams/target_stats.npz.

    Returns the z-scored target, or y unchanged for the test split.

    The test split keeps the raw target: training never reads it, and predictions are mapped back with these stats before scoring (as sklearn's TransformedTargetRegressor does), so every reported number is in raw units.

    `y`: the split's targets.
    `split`: 'train', 'val' or 'test'; val needs the train split processed first (FileNotFoundError otherwise).
    `output_dir`: the data folder, holding streams/.
    """
    stats_file = Path(output_dir) / "streams" / TARGET_STATS_NPZ
    if split == 'test':
        return y
    if split == 'train':
        mean, std = float(np.mean(y)), float(np.std(y))
        np.savez(stats_file, mean=mean, std=std)
        print(f"Train target - Mean: {mean:.6f}, Std: {std:.6f} (saved to {stats_file})")
    else:
        if not stats_file.exists():
            raise FileNotFoundError(f"Cannot find {stats_file}. Please process 'train' split first!")
        stats = np.load(stats_file)
        mean, std = float(stats['mean']), float(stats['std'])
    return (y - mean) / std


def standardize_input(x, split, output_dir):
    """Normalizes an x, y, z input array with the train split's statistics (exp_paths.compute_input_stats), which the train split saves to streams/input_stats.npz.

    Returns the normalized array, in x's dtype.

    Each axis is shifted by its own train mean, and all three are divided by one shared length scale.
    The statistics are dataset-wide, never per point cloud, so each cloud keeps its overall size, shape and position (a proxy's target is its size).
    Unlike the target, the test split is transformed too, because a model has to see test inputs on the scale it was trained on.

    `x`: the split's inputs, shape (n_clouds, n_particles, 3).
    `split`: 'train', 'val' or 'test'; val and test need the train split processed first (FileNotFoundError otherwise).
    `output_dir`: the data folder, holding streams/.
    """
    stats_file = Path(output_dir) / "streams" / INPUT_STATS_NPZ
    if split == 'train':
        mean, std = compute_input_stats(x)
        np.savez(stats_file, mean=mean, std=std)
        print(f"Train inputs - Mean: {np.round(mean, 4)}, Std: {np.round(std, 4)} (saved to {stats_file})")
    else:
        if not stats_file.exists():
            raise FileNotFoundError(f"Cannot find {stats_file}. Please process 'train' split first!")
        mean, std = load_input_stats(stats_file)
    return ((x - mean) / std).astype(x.dtype)


def write_shared_test(eval_raw, output_dir):
    """Writes the size-EVAL_DATASET_SIZE dataset's test split as the shared-test copy this dataset's models are scored on.

    Returns nothing; writes test_streams.hdf5 and file_index.npy to <output_dir>/shared_test<N>/streams/test/, and raises ValueError if eval_raw does not hold EVAL_DATASET_SIZE streams.

    The inputs are normalized with this dataset's own input stats, and the target stays raw (predictions are mapped back to it).
    The split boundary, the last 20% of the streams, is the same as in process_omnilearned.

    `eval_raw`: the raw HDF5 of the size-EVAL_DATASET_SIZE dataset.
    `output_dir`: this dataset's data folder, whose streams/input_stats.npz must already exist.
    """
    with h5py.File(eval_raw, 'r') as f:
        n_streams = int(f['pid'].shape[0])
        if n_streams != EVAL_DATASET_SIZE:
            raise ValueError(f"--shared_test_raw has {n_streams} streams, expected {EVAL_DATASET_SIZE}: {eval_raw}")
        start = int(n_streams * 0.8)
        data = f['Cartesian_data'][start:, :, XYZ_COLUMNS]
        pid = f['pid'][start:]

    mean, std = load_input_stats(Path(output_dir) / "streams" / INPUT_STATS_NPZ)
    data = ((data - mean) / std).astype(data.dtype)

    p = Path(output_dir) / shared_test_dirname() / "streams" / "test"
    p.mkdir(parents=True, exist_ok=True)
    with h5py.File(p / "test_streams.hdf5", 'w') as out_f:
        out_f.create_dataset('data', data=data, compression="gzip")
        out_f.create_dataset('pid', data=pid, compression="gzip")
        out_f.create_dataset('original_idx', data=np.arange(start, n_streams))
    np.save(p / "file_index", np.vstack((np.zeros(len(pid), dtype=int), np.arange(len(pid)))).T)
    print(f"Created {p / 'test_streams.hdf5'}: shared test split, indices {start} to {n_streams - 1}")


def process_omnilearned(input_raw, output_dir, target_columns, seed=0):
    """Writes OmniLearned's train/val/test splits (3:1:1, in stream order) of the real streams or of a synthesized proxy.

    Returns nothing; writes, for each split, <output_dir>/streams/<split>/<split>_streams.hdf5 (data, pid, original_idx) and file_index.npy, plus the stats files.

    The real streams take the x, y, z columns and the target of input_raw, and their target is z-scored with standardize_target.
    A proxy takes only the stream and particle counts from input_raw, synthesizes its point clouds and targets, and z-scores its targets in all three splits with train-split statistics saved to streams/<synthesis tag>_stats.npz.
    Every split's inputs are then normalized with standardize_input.

    `input_raw`: the raw HDF5 path.
    `output_dir`: the data folder.
    `target_columns`: '3d_x_y_z' for the real streams, or a 3d_proxy_* synthesis tag.
    `seed`: the base seed of a proxy's random generator; the real streams ignore it.
    """

    with h5py.File(input_raw, 'r') as f:
        cart_ds = f['Cartesian_data']
        pid_reg_ds = f['pid']

        num_streams = cart_ds.shape[0]
        num_particles = cart_ds.shape[1]  # particles per stream (4,000), so the proxies match the real streams
        
        train_split = int(num_streams * 0.6)
        val_split = int(num_streams * 0.8)
        
        ranges = [
            ('train', 0, train_split),
            ('val', train_split, val_split),
            ('test', val_split, num_streams)
        ]
        
        # Each proxy draws from its own generator, seeded with --seed and its synthesis tag, so a rerun reproduces it and no two proxies share random numbers.
        # The three splits are drawn one after another from the same generator, so they differ from each other.
        rng = np.random.default_rng([seed, zlib.crc32(target_columns.encode())])
        for name, start, end in ranges:
            p = Path(output_dir) / "streams" / name
            p.mkdir(parents=True, exist_ok=True)
            output_filename = p / f"{name}_streams.hdf5"
            
            if target_columns in REAL_STREAM_COLUMNS:
                sub_data = cart_ds[start:end, :, XYZ_COLUMNS]
                sub_pid_reg = pid_reg_ds[start:end]
            elif target_columns == '3d_proxy_ellipsoid':
                n_streams = end - start
                sub_data = np.zeros((n_streams, num_particles, 3), dtype=np.float32)
                
                sub_pid_reg = np.zeros((n_streams, 3), dtype=np.float32)

                from tqdm import tqdm
                for i in tqdm(range(n_streams), desc=f"Generating {name} streams"):
                    sigma_x = rng.uniform(10.0, 30.0)
                    sigma_y = rng.uniform(1.0, 5.0)
                    sigma_z = rng.uniform(0.1, 2.0)
                    
                    sub_pid_reg[i] = [sigma_x, sigma_y, sigma_z]
                    
                    x = rng.normal(0, sigma_x, num_particles)
                    y = rng.normal(0, sigma_y, num_particles)
                    z = rng.normal(0, sigma_z, num_particles)
                    points = np.vstack((x, y, z)).T
                    
                    
                    # random 3d rotation
                    theta = rng.uniform(0, 2 * np.pi, 3)
                    rx = np.array([[1, 0, 0], 
                                   [0, np.cos(theta[0]), -np.sin(theta[0])], 
                                   [0, np.sin(theta[0]), np.cos(theta[0])]])
                    ry = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])], 
                                   [0, 1, 0], 
                                   [-np.sin(theta[1]), 0, np.cos(theta[1])]])
                    rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0], 
                                   [np.sin(theta[2]), np.cos(theta[2]), 0], 
                                   [0, 0, 1]])
                    rot_mat = rz @ ry @ rx
                    
                    sub_data[i] = points @ rot_mat.T
                    
                    # sub_data[i] = points
                

                stats_file = Path(output_dir) / "streams" / "ellipsoid_stats.npz"
                
                if name == 'train':
                    mean_val = sub_pid_reg.mean(axis=0)
                    std_val = sub_pid_reg.std(axis=0)
                    
                    stats_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(stats_file, mean=mean_val, std=std_val)
                    print(f"Train set (Ellipsoid) - Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)} (saved)")
                else:
                    if not stats_file.exists():
                        raise FileNotFoundError(f"Cannot find {stats_file}. Please process 'train' split first!")
                    stats = np.load(stats_file)
                    mean_val = stats['mean']
                    std_val = stats['std']
                    print(f"{name} set (Ellipsoid) - Using train statistics: Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)}")
                
                sub_pid_reg[:] = (sub_pid_reg - mean_val) / std_val

            elif target_columns == '3d_proxy_cube':
                n_streams = end - start
                sub_data = np.zeros((n_streams, num_particles, 3), dtype=np.float32)

                # 1-D target: edge length a
                sub_pid_reg = np.zeros((n_streams, 1), dtype=np.float32)

                from tqdm import tqdm
                for i in tqdm(range(n_streams), desc=f"Generating {name} streams"):
                    a = rng.uniform(0.5, 50.0)

                    sub_pid_reg[i] = [a]

                    # uniform points in the axis-aligned cube [-a/2, a/2]^3
                    points = rng.uniform(-a / 2.0, a / 2.0, size=(num_particles, 3))

                    # random 3d rotation (same convention as ellipsoid branch)
                    theta = rng.uniform(0, 2 * np.pi, 3)
                    rx = np.array([[1, 0, 0],
                                   [0, np.cos(theta[0]), -np.sin(theta[0])],
                                   [0, np.sin(theta[0]), np.cos(theta[0])]])
                    ry = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                                   [0, 1, 0],
                                   [-np.sin(theta[1]), 0, np.cos(theta[1])]])
                    rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                                   [np.sin(theta[2]), np.cos(theta[2]), 0],
                                   [0, 0, 1]])
                    rot_mat = rz @ ry @ rx

                    sub_data[i] = points @ rot_mat.T

                stats_file = Path(output_dir) / "streams" / "cube_stats.npz"

                if name == 'train':
                    mean_val = sub_pid_reg.mean(axis=0)
                    std_val = sub_pid_reg.std(axis=0)

                    stats_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(stats_file, mean=mean_val, std=std_val)
                    print(f"Train set (Cube) - Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)} (saved)")
                else:
                    if not stats_file.exists():
                        raise FileNotFoundError(f"Cannot find {stats_file}. Please process 'train' split first!")
                    stats = np.load(stats_file)
                    mean_val = stats['mean']
                    std_val = stats['std']
                    print(f"{name} set (Cube) - Using train statistics: Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)}")

                sub_pid_reg[:] = (sub_pid_reg - mean_val) / std_val

            elif target_columns == '3d_proxy_bounded_ball':
                n_streams = end - start
                sub_data = np.zeros((n_streams, num_particles, 3), dtype=np.float32)

                # 1-D target: ball radius r
                sub_pid_reg = np.zeros((n_streams, 1), dtype=np.float32)

                from tqdm import tqdm
                for i in tqdm(range(n_streams), desc=f"Generating {name} streams"):
                    r = rng.uniform(0.5, 30.0)

                    sub_pid_reg[i] = [r]

                    # uniform-in-volume sampling inside a ball of radius r
                    # direction: uniform on the 2-sphere via normalized Gaussian
                    dirs = rng.normal(size=(num_particles, 3))
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    # magnitude: r * U^(1/3) gives uniform volume density
                    radii = r * rng.uniform(0.0, 1.0, size=num_particles) ** (1.0 / 3.0)
                    sub_data[i] = dirs * radii[:, None]

                stats_file = Path(output_dir) / "streams" / "bounded_ball_stats.npz"

                if name == 'train':
                    mean_val = sub_pid_reg.mean(axis=0)
                    std_val = sub_pid_reg.std(axis=0)

                    stats_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(stats_file, mean=mean_val, std=std_val)
                    print(f"Train set (Ball) - Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)} (saved)")
                else:
                    if not stats_file.exists():
                        raise FileNotFoundError(f"Cannot find {stats_file}. Please process 'train' split first!")
                    stats = np.load(stats_file)
                    mean_val = stats['mean']
                    std_val = stats['std']
                    print(f"{name} set (Ball) - Using train statistics: Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)}")

                sub_pid_reg[:] = (sub_pid_reg - mean_val) / std_val

            elif target_columns == '3d_proxy_gaussian_ball':
                n_streams = end - start
                sub_data = np.zeros((n_streams, num_particles, 3), dtype=np.float32)

                # 1-D target: isotropic sigma
                sub_pid_reg = np.zeros((n_streams, 1), dtype=np.float32)

                from tqdm import tqdm
                for i in tqdm(range(n_streams), desc=f"Generating {name} streams"):
                    sigma = rng.uniform(0.5, 30.0)

                    sub_pid_reg[i] = [sigma]

                    # isotropic Gaussian: equal sigma on all 3 axes
                    sub_data[i] = rng.normal(0.0, sigma, size=(num_particles, 3))

                stats_file = Path(output_dir) / "streams" / "gaussian_ball_stats.npz"

                if name == 'train':
                    mean_val = sub_pid_reg.mean(axis=0)
                    std_val = sub_pid_reg.std(axis=0)

                    stats_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(stats_file, mean=mean_val, std=std_val)
                    print(f"Train set (Gaussian Ball) - Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)} (saved)")
                else:
                    if not stats_file.exists():
                        raise FileNotFoundError(f"Cannot find {stats_file}. Please process 'train' split first!")
                    stats = np.load(stats_file)
                    mean_val = stats['mean']
                    std_val = stats['std']
                    print(f"{name} set (Gaussian Ball) - Using train statistics: Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)}")

                sub_pid_reg[:] = (sub_pid_reg - mean_val) / std_val

            elif target_columns == '3d_proxy_hard_ellipsoid':
                n_streams = end - start
                sub_data = np.zeros((n_streams, num_particles, 3), dtype=np.float32)

                # 3-D target: semi-axes (a, b, c), in the ellipsoid's sigma ranges, so hard_ellipsoid vs ellipsoid is a clean hard-vs-soft ablation.
                sub_pid_reg = np.zeros((n_streams, 3), dtype=np.float32)

                from tqdm import tqdm
                for i in tqdm(range(n_streams), desc=f"Generating {name} streams"):
                    a = rng.uniform(10.0, 30.0)
                    b = rng.uniform(1.0, 5.0)
                    c = rng.uniform(0.1, 2.0)

                    sub_pid_reg[i] = [a, b, c]

                    # Uniform in the unit ball, then scaled per axis to fill the ellipsoid, which stays uniform because a diagonal scaling has a constant Jacobian.
                    dirs = rng.normal(size=(num_particles, 3))
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    radii = rng.uniform(0.0, 1.0, size=num_particles) ** (1.0 / 3.0)
                    inside_unit_ball = dirs * radii[:, None]
                    points = inside_unit_ball * np.array([a, b, c])

                    # random 3d rotation (same convention as ellipsoid branch)
                    theta = rng.uniform(0, 2 * np.pi, 3)
                    rx = np.array([[1, 0, 0],
                                   [0, np.cos(theta[0]), -np.sin(theta[0])],
                                   [0, np.sin(theta[0]), np.cos(theta[0])]])
                    ry = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                                   [0, 1, 0],
                                   [-np.sin(theta[1]), 0, np.cos(theta[1])]])
                    rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                                   [np.sin(theta[2]), np.cos(theta[2]), 0],
                                   [0, 0, 1]])
                    rot_mat = rz @ ry @ rx

                    sub_data[i] = points @ rot_mat.T

                stats_file = Path(output_dir) / "streams" / "hard_ellipsoid_stats.npz"

                if name == 'train':
                    mean_val = sub_pid_reg.mean(axis=0)
                    std_val = sub_pid_reg.std(axis=0)

                    stats_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(stats_file, mean=mean_val, std=std_val)
                    print(f"Train set (Hard Ellipsoid) - Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)} (saved)")
                else:
                    if not stats_file.exists():
                        raise FileNotFoundError(f"Cannot find {stats_file}. Please process 'train' split first!")
                    stats = np.load(stats_file)
                    mean_val = stats['mean']
                    std_val = stats['std']
                    print(f"{name} set (Hard Ellipsoid) - Using train statistics: Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)}")

                sub_pid_reg[:] = (sub_pid_reg - mean_val) / std_val

            elif target_columns == '3d_proxy_box':
                n_streams = end - start
                sub_data = np.zeros((n_streams, num_particles, 3), dtype=np.float32)

                # 3-D target: semi-edge lengths (a, b, c), in hard_ellipsoid's ranges, so box vs hard_ellipsoid is a clean cornered-vs-smooth ablation with the same maximum extent per axis.
                sub_pid_reg = np.zeros((n_streams, 3), dtype=np.float32)

                from tqdm import tqdm
                for i in tqdm(range(n_streams), desc=f"Generating {name} streams"):
                    a = rng.uniform(10.0, 30.0)
                    b = rng.uniform(1.0, 5.0)
                    c = rng.uniform(0.1, 2.0)

                    sub_pid_reg[i] = [a, b, c]

                    # uniform inside the axis-aligned box [-a, a] x [-b, b] x [-c, c]
                    points = np.stack([
                        rng.uniform(-a, a, num_particles),
                        rng.uniform(-b, b, num_particles),
                        rng.uniform(-c, c, num_particles),
                    ], axis=1)

                    # random 3d rotation (same convention as ellipsoid / hard_ellipsoid)
                    theta = rng.uniform(0, 2 * np.pi, 3)
                    rx = np.array([[1, 0, 0],
                                   [0, np.cos(theta[0]), -np.sin(theta[0])],
                                   [0, np.sin(theta[0]), np.cos(theta[0])]])
                    ry = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                                   [0, 1, 0],
                                   [-np.sin(theta[1]), 0, np.cos(theta[1])]])
                    rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                                   [np.sin(theta[2]), np.cos(theta[2]), 0],
                                   [0, 0, 1]])
                    rot_mat = rz @ ry @ rx

                    sub_data[i] = points @ rot_mat.T

                stats_file = Path(output_dir) / "streams" / "box_stats.npz"

                if name == 'train':
                    mean_val = sub_pid_reg.mean(axis=0)
                    std_val = sub_pid_reg.std(axis=0)

                    stats_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(stats_file, mean=mean_val, std=std_val)
                    print(f"Train set (Box) - Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)} (saved)")
                else:
                    if not stats_file.exists():
                        raise FileNotFoundError(f"Cannot find {stats_file}. Please process 'train' split first!")
                    stats = np.load(stats_file)
                    mean_val = stats['mean']
                    std_val = stats['std']
                    print(f"{name} set (Box) - Using train statistics: Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)}")

                sub_pid_reg[:] = (sub_pid_reg - mean_val) / std_val

            if target_columns in REAL_STREAM_COLUMNS:
                sub_pid_reg = standardize_target(sub_pid_reg, name, output_dir)
            # Inputs are normalized only after the branches above, which synthesize each proxy's point clouds and target in raw units.
            sub_data = standardize_input(sub_data, name, output_dir)
            with h5py.File(output_filename, 'w') as out_f:
                out_f.create_dataset('data', data=sub_data, compression="gzip")
                out_f.create_dataset('pid', data=sub_pid_reg, compression="gzip")
                
                original_indices = np.arange(start, end)
                out_f.create_dataset('original_idx', data=original_indices)
                
            print(f"Created {output_filename}: pid shape {sub_pid_reg.shape}, indices {start} to {end-1}")
            
            n_rows = end-start
            col0 = np.zeros(n_rows, dtype=int)
            col1 = np.arange(n_rows, dtype=int)
            
            file_index_array = np.vstack((col0, col1)).T
            file_index_filename = p / "file_index"
            np.save(file_index_filename, file_index_array)
            print(f"Created {file_index_filename}: file_index_array shape {file_index_array.shape}, indices {start} to {end-1}")
            
if __name__ == "__main__":
    args = parse_args()

    start_time = time.time() 
    start_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[*] Processing started at: {start_timestamp}")
    validate_input_raw_size(args.input_raw, args.dataset_size)
    model_for_path = args.output_model if args.output_model else args.target_model

    output_dir = get_data_dir(
        args.output_root,
        args.target_columns,
        args.dataset_size,
        model_for_path,
        task=args.task,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Output directory: {output_dir}")

    if args.target_model == 'deepsets':
        process_deepsets(args.input_raw, output_dir, TASKS[args.task]["label_key"])
    elif "omni" in args.target_model:
        process_omnilearned(args.input_raw, output_dir, args.target_columns, args.seed)
        if args.shared_test_raw and args.target_columns in REAL_STREAM_COLUMNS:
            write_shared_test(args.shared_test_raw, output_dir)

    end_time = time.time()
    total_duration = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=int(total_duration)))
    
    print("-" * 50)
    print(
        f"Data preparation for {model_for_path} "
        f"with {dataset_key(args.target_columns, args.dataset_size, args.task)} finished!"
    )
    print(f"Total time elapsed: {formatted_time} (Total {total_duration:.2f} seconds)")
    print("-" * 50)
