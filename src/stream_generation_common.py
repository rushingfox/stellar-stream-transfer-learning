"""Machinery shared by the stream-dataset generators, raw_data_generation.py and raw_data_generation_subhalo.py.

It holds the seeded task stream, the parallel accept/skip loop, the per-stream particle table, and the HDF5 writer whose layout (Cartesian_data / ICRS_data / pid) prepare_data.py reads.
The physics of each dataset lives in its own generator script, which supplies a worker that takes one task tuple and returns None (skip) or (stream_df, props_df, info_str).
"""

import functools
print = functools.partial(print, flush=True)

import os
from collections import deque
from multiprocessing import Pool

import astropy.coordinates as coord
import h5py
import numpy as np
import pandas as pd


def resolve_num_workers(num_workers=None):
    """Picks the number of worker processes.

    Returns num_workers if given, else $SLURM_CPUS_PER_TASK (set by Slurm on NERSC), else os.cpu_count().

    `num_workers`: an explicit count, or None.
    """
    if num_workers is not None:
        return num_workers
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(slurm_cpus) if slurm_cpus else os.cpu_count()


def seed_iter(base_seed, *task_args):
    """Yields the worker tasks, one per attempt, without end.

    Returns a generator of (seed, *task_args) tuples, with seed = (base_seed * 1_000_003 + attempt) % 2**32 for attempt = 0, 1, 2, ...

    `base_seed`: the dataset's base seed.
    `task_args`: extra arguments appended to every task.
    """
    attempt = 0
    while True:
        # One unique seed per attempt, with base_seed scaled by a large prime to spread the seeds of different base seeds apart.
        seed = (base_seed * 1_000_003 + attempt) % (2**32)
        yield (seed, *task_args)
        attempt += 1


def stream_to_df(stream, stream_id, galactocentric_frame=None):
    """Builds the per-particle table of a gala mock stream.

    Returns a DataFrame with one row per particle: stream_id, the Galactocentric Cartesian phase space (x, y, z, v_x, v_y, v_z) and the ICRS observables (ra, dec, distance, pm_ra_cosdec, pm_dec, radial_velocity).

    `stream`: the gala mock stream.
    `stream_id`: the id written in every row.
    `galactocentric_frame`: the Galactocentric frame of the ICRS transform; None uses astropy's current Galactocentric defaults.
    """
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

    Returns (all_streams, props_df): the list of per-stream particle tables and the concatenated per-stream properties, both renumbered with stream_id 0..num_streams-1.

    Results are consumed in task order, so the accepted set (the first num_streams successes in seed order) depends only on the seeds, not on the worker count.
    At most 4 tasks per worker are in flight, topped up as results come in, so every worker stays busy.
    Once enough streams are accepted no more tasks are submitted, and the pool finishes the ones in flight and shuts down with close()/join().
    Stopping early with terminate() instead would kill workers mid-task, and one killed while holding the result queue's lock leaves the main process waiting on it forever.

    `worker`: a function of one task that returns None (skip) or (stream_df, props_df, info_str).
    `tasks`: an iterator of tasks, e.g. from seed_iter.
    `num_streams`: the number of streams to accept.
    `num_workers`: the number of worker processes.
    """
    all_streams = []
    all_props   = []

    generated_count = 0
    attempt_count   = 0

    max_in_flight = 4 * num_workers
    in_flight = deque()   # AsyncResults, oldest (lowest seed) first

    with Pool(processes=num_workers) as pool:
        while generated_count < num_streams:
            while len(in_flight) < max_in_flight:
                in_flight.append(pool.apply_async(worker, (next(tasks),)))

            result = in_flight.popleft().get()
            attempt_count += 1

            if result is None:
                continue

            stream_df, props_df, info = result

            # The canonical stream_id is the acceptance order.
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

        # The surplus tasks still in flight run to completion; their results are discarded.
        pool.close()
        pool.join()

    print(
        f"\nDone. Generated {generated_count} streams "
        f"from {attempt_count} attempts "
        f"(skip rate: {1 - generated_count/attempt_count:.1%})"
    )

    return all_streams, pd.concat(all_props, ignore_index=True)


def _as_float(x):
    """3.0 * u.kpc -> 3.0 (the value in its own unit); 2 -> 2.0."""
    return x.value if hasattr(x, "value") else float(x)


def save_dataset(output, all_streams, props_df, prop_cols, prop_units, props_key,
                 label_col, log_label):
    """Writes the stream dataset to HDF5, in the layout prepare_data.py reads.

    Returns nothing; writes `output` with these datasets, for N streams of T particles:
    - Cartesian_data (N, T, 7): x, y, z, v_x, v_y, v_z, v_total.
    - ICRS_data (N, T, 7): ra, dec, distance, pm_ra_cosdec, pm_dec, radial_velocity, parallax.
    - <props_key> (N, len(prop_cols)), with 'columns' and 'units' attrs.
    - pid (N, 1): the target, props_df[label_col], log10'd if log_label.

    `output`: the HDF5 path.
    `all_streams`, `props_df`: the per-stream particle tables and properties, from collect_streams.
    `prop_cols`: the property columns stored under props_key, in order.
    `prop_units`: {column: unit string}, for the 'units' attr.
    `props_key`: the name of the properties dataset, e.g. prog_properties.
    `label_col`: the property column that becomes the target.
    `log_label`: store log10 of that column instead of its value.
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

    # ---- build target (pid) ------------------------------------------------
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
