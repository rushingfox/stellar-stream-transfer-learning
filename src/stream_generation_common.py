"""
stream_generation_common.py
---------------------------
Machinery shared by the stream-dataset generators (raw_data_generation.py and
raw_data_generation_subhalo.py): the seeded task stream, the parallel
accept/skip loop, the per-stream particle table, and the HDF5 writer whose
layout (Cartesian_data / ICRS_data / pid) prepare_data.py reads.

The physics of each dataset lives in its own generator script, which supplies a
worker function taking one task tuple and returning either None (skip) or
(stream_df, props_df, info_str).
"""

import functools
print = functools.partial(print, flush=True)

import os
from multiprocessing import Pool

import astropy.coordinates as coord
import h5py
import numpy as np
import pandas as pd


def resolve_num_workers(num_workers=None):
    """Explicit value if given, else $SLURM_CPUS_PER_TASK (set by Slurm on NERSC), else os.cpu_count()."""
    if num_workers is not None:
        return num_workers
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(slurm_cpus) if slurm_cpus else os.cpu_count()


def seed_iter(base_seed, *task_args):
    """Yields (seed, *task_args) tasks with monotonically increasing seeds."""
    attempt = 0
    while True:
        # Combine base seed and attempt index into a unique seed.
        # Multiply base_seed by a large prime to spread the space.
        seed = (base_seed * 1_000_003 + attempt) % (2**32)
        yield (seed, *task_args)
        attempt += 1


def stream_to_df(stream, stream_id, galactocentric_frame=None):
    """Per-particle table of a gala mock stream: Galactocentric Cartesian
    phase space plus the ICRS observables. galactocentric_frame=None uses
    astropy's current Galactocentric defaults."""
    if galactocentric_frame is None:
        stream_icrs = stream.to_coord_frame(coord.ICRS())
    else:
        stream_icrs = stream.to_coord_frame(coord.ICRS(), galactocentric_frame=galactocentric_frame)

    return pd.DataFrame({
        "stream_id": stream_id,
        "x":  stream.x,
        "y":  stream.y,
        "z":  stream.z,
        "v_x": stream.v_x,
        "v_y": stream.v_y,
        "v_z": stream.v_z,
        "ra":  stream_icrs.ra,
        "dec": stream_icrs.dec,
        "distance": stream_icrs.distance,
        "pm_ra_cosdec": stream_icrs.pm_ra_cosdec,
        "pm_dec": stream_icrs.pm_dec,
        "radial_velocity": stream_icrs.radial_velocity,
    })


def collect_streams(worker, tasks, num_streams, num_workers):
    """Runs worker over tasks in parallel until num_streams are accepted.

    Results are consumed in task order (imap), so the accepted set depends only
    on the seeds, not on the worker count. Returns the per-stream particle
    tables and the concatenated per-stream properties, both renumbered with
    stream_id 0..num_streams-1.
    """
    all_streams = []
    all_props   = []

    generated_count = 0
    attempt_count   = 0

    # chunksize: each worker gets a small batch to reduce scheduling overhead.
    # Tune if needed; 2–4 works well when each task takes ~seconds.
    chunksize = max(1, num_workers // 2)

    with Pool(processes=num_workers) as pool:
        for result in pool.imap(worker, tasks, chunksize=chunksize):
            attempt_count += 1

            if result is None:
                continue

            stream_df, props_df, info = result

            # Reassign canonical stream_id
            stream_df["stream_id"] = generated_count
            props_df["stream_id"]  = generated_count

            all_streams.append(stream_df)
            all_props.append(props_df)
            generated_count += 1

            print(info)
            print(
                f"[{generated_count}/{num_streams}] "
                f"Stream accepted  (attempts so far: {attempt_count})"
            )

            if generated_count >= num_streams:
                pool.terminate()
                pool.join()
                break

    print(
        f"\nDone. Generated {generated_count} streams "
        f"from {attempt_count} attempts "
        f"(skip rate: {1 - generated_count/attempt_count:.1%})"
    )

    return all_streams, pd.concat(all_props, ignore_index=True)


def _as_float(x):
    return x.value if hasattr(x, "value") else float(x)


def save_dataset(output, all_streams, props_df, prop_cols, prop_units, props_key,
                 label_col, log_label):
    """Writes the HDF5 layout that prepare_data.py reads:

    Cartesian_data  (N, T, 7)  x, y, z, v_x, v_y, v_z, v_total
    ICRS_data       (N, T, 7)  ra, dec, distance, pm_ra_cosdec, pm_dec, radial_velocity, parallax
    <props_key>     (N, len(prop_cols)), with 'columns' and 'units' attrs
    pid             (N, 1)     props_df[label_col], log10'd if log_label
    """
    # ---- build feature arrays ----------------------------------------------
    ICRS_raw_data_list      = []
    Cartesian_raw_data_list = []

    for df in all_streams:
        v_total  = np.sqrt(df["v_x"] ** 2 + df["v_y"] ** 2 + df["v_z"] ** 2)
        parallax = 1.0 / df["distance"]

        Cartesian_raw_data_list.append(
            df.assign(v_total=v_total)[
                ["x", "y", "z", "v_x", "v_y", "v_z", "v_total"]
            ].to_numpy()
        )
        ICRS_raw_data_list.append(
            df.assign(parallax=parallax)[
                ["ra", "dec", "distance", "pm_ra_cosdec", "pm_dec",
                 "radial_velocity", "parallax"]
            ].to_numpy()
        )

    ICRS_raw_data      = np.stack(ICRS_raw_data_list)
    Cartesian_raw_data = np.stack(Cartesian_raw_data_list)

    # ---- build properties array --------------------------------------------
    props_array = np.column_stack([
        props_df[col].apply(_as_float).to_numpy()
        for col in prop_cols
    ])

    # ---- build label (pid) -------------------------------------------------
    label_vals = props_df[label_col].apply(_as_float).to_numpy()
    if log_label:
        label_vals = np.log10(label_vals)
    labels = label_vals.reshape(-1, 1)

    # ---- save to HDF5 ------------------------------------------------------
    with h5py.File(output, "w") as f:
        f.create_dataset("ICRS_data",      data=ICRS_raw_data)
        f.create_dataset("Cartesian_data", data=Cartesian_raw_data)

        ds_props = f.create_dataset(props_key, data=props_array)
        ds_props.attrs["columns"] = prop_cols
        ds_props.attrs["units"]   = [prop_units[c] for c in prop_cols]

        ds_pid = f.create_dataset("pid", data=labels)
        ds_pid.attrs["label_col"]      = label_col
        ds_pid.attrs["log_transformed"] = log_label

    print("\nDataset saved to:", output)
    print(f"  ICRS data shape:        {ICRS_raw_data.shape}  (7 features)")
    print(f"  Cartesian data shape:   {Cartesian_raw_data.shape}  (7 features)")
    print(f"  {props_key} shape:  {props_array.shape}  ({len(prop_cols)} columns)")
    print(f"  Labels shape:           {labels.shape}  (label_col={label_col}, log={log_label})")
