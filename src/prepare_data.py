import argparse
import h5py
import numpy as np
from pathlib import Path
import os
import time
import datetime
from tqdm import tqdm
import sys
from exp_paths import VALID_DATASET_SIZES, dataset_key, get_data_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_raw', type=str, required=True)
    parser.add_argument('--output_root', type=str, default="..", help="Repository root for output datasets")
    parser.add_argument('--dataset_size', type=int, choices=VALID_DATASET_SIZES, required=True)
    parser.add_argument('--target_model', type=str, choices=['baseline', 'baseline_256', 'omnilearned'], required=True)
    parser.add_argument('--output_model', type=str, default="", help="Optional model folder name override for saving outputs")
    parser.add_argument('--target_columns', type=str, choices=['3d_x_y_z', '3d_ra_dec_parallax', '4d_x_y_z_vtotal', '4d_ra_dec_parallax_vtotal', '6d_x_y_z_vx_vy_vz', '6d_ra_dec_parallax_proper_motions', '3d_proxy_ellipsoid', '3d_proxy_cube', '3d_proxy_bounded_ball', '3d_proxy_gaussian_ball', '3d_proxy_hard_ellipsoid', '3d_proxy_box', '3d_x_y_z_eigenvalues'], required=True)

    args = parser.parse_args()
    omnilearned_valid_modes = [
        '3d_x_y_z',
        '4d_x_y_z_vtotal',
        '6d_x_y_z_vx_vy_vz',
        '3d_proxy_ellipsoid',
        '3d_proxy_cube',
        '3d_proxy_bounded_ball',
        '3d_proxy_gaussian_ball',
        '3d_proxy_hard_ellipsoid',
        '3d_proxy_box',
        '3d_x_y_z_eigenvalues'
    ]
    if 'omni' in args.target_model and args.target_columns not in omnilearned_valid_modes:
        print(f"[INFO] Skipping: model='{args.target_model}' cannot use columns='{args.target_columns}'. (Requires x/y/z)")
        sys.exit(0)

    return args


def validate_input_raw_size(input_raw, dataset_size):
    with h5py.File(input_raw, 'r') as f:
        if 'pid' not in f:
            raise KeyError(f"Dataset 'pid' not found in input file: {input_raw}")
        actual_size = int(f['pid'].shape[0])

    if actual_size != dataset_size:
        raise ValueError(
            f"Input raw size mismatch: expected dataset_size={dataset_size}, "
            f"but found {actual_size} samples in {input_raw}."
        )

def process_baseline(input_raw, output_dir, target_columns):
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    output_file = p / "baseline_dataset.h5"
    with h5py.File(input_raw, 'r') as f:
        cart_ds = f['Cartesian_data']
        icrs_ds = f['ICRS_data']
        pid_reg_ds = f['pid'][:]
        if target_columns == '3d_x_y_z':
            raw_data = cart_ds[:,:,[0, 1, 2]]
        elif target_columns == '3d_ra_dec_parallax':
            raw_data = icrs_ds[:,:,[0, 1, -1]]
        elif target_columns == '4d_x_y_z_vtotal':
            raw_data = cart_ds[:,:,[0, 1, 2, -1]]
        elif target_columns == '4d_ra_dec_parallax_vtotal':
            raw_data = icrs_ds[:, :, [0, 1, -1]]   # (N, T, 3)
            cart_last = cart_ds[:, :, -1]          # (N, T)
            # extend to (N, T, 1)
            cart_last = cart_last[..., np.newaxis]
            # concatenate to (N, T, 4)
            raw_data = np.concatenate([raw_data, cart_last], axis=-1)
        elif target_columns == '6d_x_y_z_vx_vy_vz':
            raw_data = cart_ds[:,:,[0, 1, 2, 3, 4, 5]]
        elif target_columns == '6d_ra_dec_parallax_proper_motions':
            last_col = icrs_ds.shape[2] - 1
            tmp = icrs_ds[:, :, [0, 1, 3, 4, 5, last_col]]
            raw_data = tmp[:, :, [0, 1, 5, 2, 3, 4]]

    with h5py.File(output_file, 'w') as out_f:
        out_f.create_dataset('raw_data', data=raw_data, compression="gzip")
        out_f.create_dataset('Log_Prog_Mass', data=pid_reg_ds, compression="gzip")
    print(f"Created {output_file} with raw_data of {raw_data.shape} and pid_reg_ds of {pid_reg_ds.shape}")

def process_omnilearned(input_raw, output_dir, target_columns):

    with h5py.File(input_raw, 'r') as f:
        cart_ds = f['Cartesian_data']
        pid_reg_ds = f['pid']
        
        num_streams = cart_ds.shape[0]
        num_particles = cart_ds.shape[1]  # Get the number of particles，usually 4000
        
        train_split = int(num_streams * 0.6)
        val_split = int(num_streams * 0.8)
        
        ranges = [
            ('train', 0, train_split),
            ('val', train_split, val_split),
            ('test', val_split, num_streams)
        ]
        
        for name, start, end in ranges:
            p = Path(output_dir) / "streams" / name
            p.mkdir(parents=True, exist_ok=True)
            output_filename = p / f"{name}_streams.hdf5"
            
            if target_columns == "3d_x_y_z":
                sub_data = cart_ds[start:end,:,[0, 1, 2]]
                sub_pid_reg = pid_reg_ds[start:end]
            elif target_columns == '4d_x_y_z_vtotal':
                sub_data = cart_ds[start:end,:,[0, 1, 2, -1]]
                sub_pid_reg = pid_reg_ds[start:end]
            elif target_columns == '6d_x_y_z_vx_vy_vz':
                sub_data = cart_ds[start:end,:,[0, 1, 2, 3, 4, 5]]
                sub_pid_reg = pid_reg_ds[start:end]
            elif target_columns == '3d_proxy_ellipsoid':
                n_streams = end - start
                sub_data = np.zeros((n_streams, num_particles, 3), dtype=np.float32)
                
                sub_pid_reg = np.zeros((n_streams, 3), dtype=np.float32)

                from tqdm import tqdm
                for i in tqdm(range(n_streams), desc=f"Generating {name} streams"):
                    sigma_x = np.random.uniform(10.0, 30.0)
                    sigma_y = np.random.uniform(1.0, 5.0)
                    sigma_z = np.random.uniform(0.1, 2.0)
                    
                    sub_pid_reg[i] = [sigma_x, sigma_y, sigma_z]
                    
                    x = np.random.normal(0, sigma_x, num_particles)
                    y = np.random.normal(0, sigma_y, num_particles)
                    z = np.random.normal(0, sigma_z, num_particles)
                    points = np.vstack((x, y, z)).T
                    
                    
                    # random 3d rotation
                    theta = np.random.uniform(0, 2 * np.pi, 3)
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

                # 1-D label: edge length a
                sub_pid_reg = np.zeros((n_streams, 1), dtype=np.float32)

                from tqdm import tqdm
                for i in tqdm(range(n_streams), desc=f"Generating {name} streams"):
                    a = np.random.uniform(0.5, 50.0)

                    sub_pid_reg[i] = [a]

                    # uniform points in the axis-aligned cube [-a/2, a/2]^3
                    points = np.random.uniform(-a / 2.0, a / 2.0, size=(num_particles, 3))

                    # random 3d rotation (same convention as ellipsoid branch)
                    theta = np.random.uniform(0, 2 * np.pi, 3)
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

                # 1-D label: ball radius r
                sub_pid_reg = np.zeros((n_streams, 1), dtype=np.float32)

                from tqdm import tqdm
                for i in tqdm(range(n_streams), desc=f"Generating {name} streams"):
                    r = np.random.uniform(0.5, 30.0)

                    sub_pid_reg[i] = [r]

                    # uniform-in-volume sampling inside a ball of radius r
                    # direction: uniform on the 2-sphere via normalized Gaussian
                    dirs = np.random.normal(size=(num_particles, 3))
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    # magnitude: r * U^(1/3) gives uniform volume density
                    radii = r * np.random.uniform(0.0, 1.0, size=num_particles) ** (1.0 / 3.0)
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

                # 1-D label: isotropic sigma
                sub_pid_reg = np.zeros((n_streams, 1), dtype=np.float32)

                from tqdm import tqdm
                for i in tqdm(range(n_streams), desc=f"Generating {name} streams"):
                    sigma = np.random.uniform(0.5, 30.0)

                    sub_pid_reg[i] = [sigma]

                    # isotropic Gaussian: equal sigma on all 3 axes
                    sub_data[i] = np.random.normal(0.0, sigma, size=(num_particles, 3))

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

                # 3-D label: (a, b, c) semi-axes; reuse ellipsoid's sigma ranges
                # so hard_ellipsoid <-> ellipsoid is a clean hard-vs-soft ablation.
                sub_pid_reg = np.zeros((n_streams, 3), dtype=np.float32)

                from tqdm import tqdm
                for i in tqdm(range(n_streams), desc=f"Generating {name} streams"):
                    a = np.random.uniform(10.0, 30.0)
                    b = np.random.uniform(1.0, 5.0)
                    c = np.random.uniform(0.1, 2.0)

                    sub_pid_reg[i] = [a, b, c]

                    # uniform-in-volume sampling inside the unit ball, then
                    # scale per-axis to fill an ellipsoid (Jacobian is constant
                    # under diagonal scaling, so the result is uniform inside
                    # the ellipsoid).
                    dirs = np.random.normal(size=(num_particles, 3))
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    radii = np.random.uniform(0.0, 1.0, size=num_particles) ** (1.0 / 3.0)
                    inside_unit_ball = dirs * radii[:, None]
                    points = inside_unit_ball * np.array([a, b, c])

                    # random 3d rotation (same convention as ellipsoid branch)
                    theta = np.random.uniform(0, 2 * np.pi, 3)
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

                # 3-D label: (a, b, c) semi-edge-lengths. Reuse hard_ellipsoid's
                # ranges so box <-> hard_ellipsoid is a clean cornered-vs-smooth
                # ablation under the same per-axis maximum extent.
                sub_pid_reg = np.zeros((n_streams, 3), dtype=np.float32)

                from tqdm import tqdm
                for i in tqdm(range(n_streams), desc=f"Generating {name} streams"):
                    a = np.random.uniform(10.0, 30.0)
                    b = np.random.uniform(1.0, 5.0)
                    c = np.random.uniform(0.1, 2.0)

                    sub_pid_reg[i] = [a, b, c]

                    # uniform inside the axis-aligned box [-a, a] x [-b, b] x [-c, c]
                    points = np.stack([
                        np.random.uniform(-a, a, num_particles),
                        np.random.uniform(-b, b, num_particles),
                        np.random.uniform(-c, c, num_particles),
                    ], axis=1)

                    # random 3d rotation (same convention as ellipsoid / hard_ellipsoid)
                    theta = np.random.uniform(0, 2 * np.pi, 3)
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

            elif target_columns == '3d_x_y_z_eigenvalues':
                n_streams = end - start
                
                # Input：Still use the real stream data
                sub_data = cart_ds[start:end, :, [0, 1, 2]]  # (n_streams, 4000, 3)
                
                # Target：the 3D eigenvalues for each stream
                sub_pid_reg = np.zeros((n_streams, 3), dtype=np.float32)
                
                from tqdm import tqdm
                print(f"Computing eigenvalues for {name} streams...")
                
                # First pass: compute all eigenvalues
                eigenvalues_list = []
                for i in tqdm(range(n_streams), desc=f"Processing {name}"):
                    points = sub_data[i]  # (4000, 3)
                    
                    points_centered = points - points.mean(axis=0)
                    
                    cov_matrix = np.cov(points_centered.T)  # (3, 3)
                    
                    eigenvalues = np.linalg.eigvalsh(cov_matrix)  # (3,)
            
                    # Make sure the eigenvalues are corresponding to Length>Width>Thickness
                    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
                    
                    # eigenvalues = variance = sigma^2
                    # sqrt(eigenvalues) = sigma
                    eigenvalues_list.append(np.sqrt(eigenvalues_sorted))
                
                # Convert to array
                eigenvalues_array = np.array(eigenvalues_list)
                
                # Compute or load z-score statistics
                stats_file = Path(output_dir) / "streams" / "eigenvalue_stats.npz"
                
                if name == 'train':
                    # Compute statistics from train set
                    mean_val = eigenvalues_array.mean(axis=0)
                    std_val = eigenvalues_array.std(axis=0)
                    
                    # Save for val and test
                    stats_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(stats_file, mean=mean_val, std=std_val)
                    print(f"Train set - Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)} (saved)")
                else:
                    # Load statistics from train set
                    if not stats_file.exists():
                        raise FileNotFoundError(f"Cannot find {stats_file}. Please process 'train' split first!")
                    stats = np.load(stats_file)
                    mean_val = stats['mean']
                    std_val = stats['std']
                    print(f"{name} set - Using train statistics: Mean: {np.round(mean_val, 4)}, Std: {np.round(std_val, 4)}")
                
                # Apply z-score normalization using train statistics
                z_scores = (eigenvalues_array - mean_val) / std_val
                
                # Store the z-scores
                sub_pid_reg[:] = z_scores
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
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Output directory: {output_dir}")
    
    if args.target_model == 'baseline' or args.target_model == 'baseline_256':
        process_baseline(args.input_raw, output_dir, args.target_columns)
    elif "omni" in args.target_model:
        process_omnilearned(args.input_raw, output_dir, args.target_columns)
        
    end_time = time.time()
    total_duration = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=int(total_duration)))
    
    print("-" * 50)
    print(
        f"Data preparation for {model_for_path} "
        f"with {dataset_key(args.target_columns, args.dataset_size)} finished!"
    )
    print(f"Total time elapsed: {formatted_time} (Total {total_duration:.2f} seconds)")
    print("-" * 50)
