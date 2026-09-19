"""
raw_data_generation.py
--------------
Command-line script for generating synthetic stellar stream datasets.
Converted from dataset_gen_v6.ipynb.
Parallelised with multiprocessing.Pool for NERSC CPU nodes.

Usage example:
    python raw_data_generation.py --num_streams 1000 --output prog_mass_reg_dataset_1000.h5
"""

import functools
print = functools.partial(print, flush=True)

import argparse
import math
import os
import random
import sys

import astropy.coordinates as coord
import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
import h5py
import numpy as np
import pandas as pd
from colossus.cosmology import cosmology
from colossus.halo import concentration, mass_defs
from gala.dynamics import mockstream as ms
from gala.units import galactic
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist

# multiprocessing — use 'spawn' to avoid issues with C extensions on Linux
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from multiprocessing import Pool


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic stellar stream datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_streams", type=int, default=1000,
        help=(
            "Number of streams to generate. "
            "num_streams itself is used as the base seed."
        ),
    )
    parser.add_argument(
        "--output", type=str, default="prog_mass_reg_dataset.h5",
        help="Output HDF5 file path for the processed feature dataset.",
    )
    parser.add_argument(
        "--smhm_model", type=str, default="FIRE_LatteELVIS",
        help=(
            "Name of the SMHM data file to use (without the '_SMHM' suffix). "
            "The file '<smhm_model>_SMHM' must exist in the current directory."
        ),
    )
    parser.add_argument(
        "--star_pot_form", type=str, default="hernquist",
        choices=["hernquist", "plummer"],
        help="Stellar potential profile for the progenitor.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help=(
            "Number of parallel worker processes. "
            "Defaults to $SLURM_CPUS_PER_TASK if set, else os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--label_col", type=str, default="progenitor_mstar",
        help=(
            "Column from prog_properties to use as the label (pid). "
            "Must be one of the numeric columns stored in prog_properties."
        ),
    )
    parser.add_argument(
        "--log_label", action=argparse.BooleanOptionalAction, default=True,
        help="Apply log10 to the label value. Use --no_log_label to disable.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Global setup (cosmology, MW potential, galactocentric frame)
# ---------------------------------------------------------------------------

def setup_globals():
    """Initialise the galactocentric frame, cosmology and MW potential."""
    _ = coord.galactocentric_frame_defaults.set("v4.0")
    cosmo = cosmology.setCosmology("planck18", persistence="")
    mw = gp.BovyMWPotential2014(units=galactic)
    return cosmo, mw


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ProgenitorParams:
    def __init__(
        self,
        ra, dec, distance,
        pm_ra_cosdec, pm_dec, radial_velocity,
        mstar, t1, t2, dt,
    ):
        self.ra = ra
        self.dec = dec
        self.distance = distance
        self.pm_ra_cosdec = pm_ra_cosdec
        self.pm_dec = pm_dec
        self.radial_velocity = radial_velocity
        self.mstar = mstar
        self.t1 = t1
        self.t2 = t2
        self.dt = dt

    def __iter__(self):
        return iter(self.__dict__.items())

    def to_dict(self):
        return {key: value for key, value in self}


# ---------------------------------------------------------------------------
# Physical relations
# ---------------------------------------------------------------------------

def SMHM(Mstar_in_Msun, smhm_model: str = "FIRE_LatteELVIS"):
    """Stellar-to-Halo Mass Relation."""
    data = np.genfromtxt(f"./{smhm_model}_SMHM")
    log_Mstar = data[:, 0]
    log_Mhalo = data[:, 1]
    interp = interp1d(log_Mstar, log_Mhalo, kind="linear", fill_value="extrapolate")
    M200c_in_Msun = 10 ** interp(np.log10(Mstar_in_Msun))
    return M200c_in_Msun


def SSHS(Mhalo_200c_in_Msun, z, cosmo):
    """Stellar-Size Halo-Size relation."""
    C_200c = concentration.modelLudlow16(
        Mhalo_200c_in_Msun * (cosmo.H0 / 100), z=z
    )[0]
    M_vir, R_vir, C_vir = mass_defs.changeMassDefinition(
        Mhalo_200c_in_Msun * (cosmo.H0 / 100), C_200c, z, "200c", "vir"
    )
    R_vir_in_kpc = R_vir / (cosmo.H0 / 100)
    Re_in_kpc = 0.02 * (1 + z) ** (-0.2) * (C_vir / 10) ** (-0.7) * R_vir_in_kpc
    return Re_in_kpc


def Re_to_Rscale(Re_in_kpc, profile: str = "hernquist"):
    """Convert half-light radius to scale radius for a given profile."""
    if profile == "plummer":
        return Re_in_kpc * math.sqrt(2 ** (2 / 3) - 1)
    elif profile == "hernquist":
        return Re_in_kpc / (1.0 + math.sqrt(2.0))
    else:
        raise ValueError(
            "Unsupported profile type. Choose from 'plummer' or 'hernquist'."
        )


def max_distance(x, y, z):
    coords = np.column_stack([x, y, z])
    return np.max(pdist(coords))


# ---------------------------------------------------------------------------
# Orbit integration
# ---------------------------------------------------------------------------

def compute_orbit(progenitor_params: ProgenitorParams, mw, whole_time_space: bool = False):
    icrs = coord.ICRS(
        ra=progenitor_params.ra,
        dec=progenitor_params.dec,
        distance=progenitor_params.distance,
        pm_ra_cosdec=progenitor_params.pm_ra_cosdec,
        pm_dec=progenitor_params.pm_dec,
        radial_velocity=progenitor_params.radial_velocity,
    )
    cart = icrs.transform_to(coord.Galactocentric()).cartesian
    ics = gd.PhaseSpacePosition(cart)

    if whole_time_space:
        orbit = gp.Hamiltonian(mw).integrate_orbit(
            ics,
            dt=progenitor_params.dt,
            t1=progenitor_params.t1,
            t2=progenitor_params.t2,
        )
    else:
        orbit = gp.Hamiltonian(mw).integrate_orbit(
            ics, dt=progenitor_params.dt, t1=-1.5 * u.Gyr, t2=0 * u.Gyr
        )
    return orbit


# ---------------------------------------------------------------------------
# Stream generation
# ---------------------------------------------------------------------------

def stream_gen(
    progenitor_params: ProgenitorParams,
    stream_id: int,
    orbit,
    mw,
    cosmo,
    smhm_model: str = "FIRE_LatteELVIS",
    star_pot_form: str = "hernquist",
    attempt_seed: int = 0,
):
    icrs = coord.ICRS(
        ra=progenitor_params.ra,
        dec=progenitor_params.dec,
        distance=progenitor_params.distance,
        pm_ra_cosdec=progenitor_params.pm_ra_cosdec,
        pm_dec=progenitor_params.pm_dec,
        radial_velocity=progenitor_params.radial_velocity,
    )
    cart = icrs.transform_to(coord.Galactocentric()).cartesian
    w0 = gd.PhaseSpacePosition(cart)

    mstar = progenitor_params.mstar
    mhalo_200c = SMHM(mstar.value, smhm_model)
    redshift = cosmo.lookbackTime(-1 * progenitor_params.t1.value, inverse=True)
    R_half = SSHS(mhalo_200c, z=redshift, cosmo=cosmo)

    if star_pot_form == "plummer":
        rscale = Re_to_Rscale(R_half, profile="plummer")
        pot_star = gp.PlummerPotential(m=mstar, b=rscale * u.kpc, units=galactic)
    elif star_pot_form == "hernquist":
        rscale = Re_to_Rscale(R_half, profile="hernquist")
        pot_star = gp.HernquistPotential(m=mstar, c=rscale * u.kpc, units=galactic)
    else:
        raise ValueError(f"Unknown star_pot_form: {star_pot_form}")

    C_200c = concentration.modelLudlow16(
        mhalo_200c * (cosmo.H0 / 100), z=redshift
    )[0]
    pot_DM = gp.NFWPotential.from_M200_c(mhalo_200c * u.Msun, C_200c, units=galactic)
    pot_total = pot_DM + pot_star
    
    rng = np.random.RandomState(attempt_seed % (2**32))
    df_model = ms.FardalStreamDF(gala_modified=False, random_state=rng)
    #df_model = ms.FardalStreamDF(gala_modified=False)
    gen_stream = ms.MockStreamGenerator(df_model, mw, progenitor_potential=pot_total)
    stream, _ = gen_stream.run(
        w0, mstar,
        t1=progenitor_params.t1,
        t2=progenitor_params.t2,
        dt=progenitor_params.dt,
    )

    stream_icrs = stream.to_coord_frame(coord.ICRS())

    stream_df = pd.DataFrame({
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

    criterion_2 = 1 if max_distance(stream.x, stream.y, stream.z) > 120 else 0

    prog_df = pd.DataFrame({
        "progenitor_type":  ["DG"],
        "stream_id":        [stream_id],
        "analytical_peri":  [orbit.pericenter()],
        "analytical_apo":   [orbit.apocenter()],
        "analytical_ecc":   [orbit.eccentricity()],
        "Criterion_2":      [criterion_2],
    })
    for key, value in progenitor_params:
        prog_df[f"progenitor_{key}"] = [value]

    return stream_df, prog_df


# ---------------------------------------------------------------------------
# Worker function (runs inside each subprocess)
# ---------------------------------------------------------------------------

def _try_generate_one(task):
    """
    Attempt to generate one stream.

    Parameters
    ----------
    task : tuple
        (attempt_seed, smhm_model, star_pot_form)

    Returns
    -------
    (stream_df, prog_df, info_str)  on success
    None                            on skip
    """
    attempt_seed, smhm_model, star_pot_form = task

    # Each worker gets its own independent RNG state
    random.seed(attempt_seed)
    np.random.seed(attempt_seed % (2**32))   # numpy seed must be < 2^32

    # Re-initialise gala/astropy objects inside the subprocess
    cosmo, mw = setup_globals()

    try:
        progenitor_params = ProgenitorParams(
            ra=round(random.uniform(0, 360), 0) * u.degree,
            dec=round(random.uniform(-90, 90), 0) * u.degree,
            distance=round(random.uniform(5, 100), 1) * u.kpc,
            pm_ra_cosdec=round(random.uniform(-5, 5), 1) * u.mas / u.yr,
            pm_dec=round(random.uniform(-5, 5), 1) * u.mas / u.yr,
            radial_velocity=round(random.uniform(-200, 200), 0) * u.km / u.s,
            mstar=round(10 ** random.uniform(6.5, 8.5)) * u.Msun,
            t1=-2.0 * u.Gyr,
            t2= 0.0 * u.Gyr,
            dt= 1.0 * u.Myr,
        )

        orbit = compute_orbit(progenitor_params, mw, whole_time_space=True)

        if any(np.isnan(v) for v in [
            orbit.pericenter(), orbit.apocenter(), orbit.eccentricity()
        ]):
            return None  # NaN orbit — skip

        if orbit.pericenter() > 100 * u.kpc:
            return None  # pericenter too large — skip

        # stream_id is a placeholder; the main process reassigns it
        stream_df, prog_df = stream_gen(
            progenitor_params,
            stream_id=attempt_seed,   # temporary unique id
            orbit=orbit,
            mw=mw,
            cosmo=cosmo,
            smhm_model=smhm_model,
            star_pot_form=star_pot_form,
            attempt_seed=attempt_seed,
        )

        if prog_df["Criterion_2"].values[0] == 0:
            return None  # Criterion 2 failed — skip

        info = (
            f"  Orbit → peri={orbit.pericenter():.2f}  "
            f"apo={orbit.apocenter():.2f}  "
            f"ecc={orbit.eccentricity():.3f}"
        )
        return stream_df, prog_df, info

    except RuntimeError as e:
        if "Larger nmax is needed" in str(e):
            return None   # known benign error — skip
        raise


# ---------------------------------------------------------------------------
# Infinite task iterator
# ---------------------------------------------------------------------------

def _task_iter(base_seed, smhm_model, star_pot_form):
    """Yields tasks with monotonically increasing seeds."""
    attempt = 0
    while True:
        # Combine base seed and attempt index into a unique seed.
        # Multiply base_seed by a large prime to spread the space.
        seed = (base_seed * 1_000_003 + attempt) % (2**32)
        yield (seed, smhm_model, star_pot_form)
        attempt += 1


# ---------------------------------------------------------------------------
# Progenitor property columns saved into prog_properties dataset
# ---------------------------------------------------------------------------

_PROG_COLS = [
    "analytical_peri",
    "analytical_apo",
    "analytical_ecc",
    "progenitor_ra",
    "progenitor_dec",
    "progenitor_distance",
    "progenitor_pm_ra_cosdec",
    "progenitor_pm_dec",
    "progenitor_radial_velocity",
    "progenitor_mstar",
    "progenitor_t1",
    "progenitor_t2",
    "progenitor_dt",
]

_PROG_UNITS = {
    "analytical_peri":           "kpc",
    "analytical_apo":            "kpc",
    "analytical_ecc":            "dimensionless",
    "progenitor_ra":             "deg",
    "progenitor_dec":            "deg",
    "progenitor_distance":       "kpc",
    "progenitor_pm_ra_cosdec":   "mas/yr",
    "progenitor_pm_dec":         "mas/yr",
    "progenitor_radial_velocity": "km/s",
    "progenitor_mstar":          "Msun",
    "progenitor_t1":             "Gyr",
    "progenitor_t2":             "Gyr",
    "progenitor_dt":             "Myr",
}


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset(args):
    base_seed = args.num_streams
    random.seed(base_seed)
    np.random.seed(base_seed)

    # ---- worker count ------------------------------------------------------
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        # On NERSC, SLURM_CPUS_PER_TASK is set automatically by Slurm
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        num_workers = int(slurm_cpus) if slurm_cpus else os.cpu_count()

    print(f"Random base seed : {base_seed}")
    print(f"Target streams   : {args.num_streams}")
    print(f"Worker processes : {num_workers}")
    print(f"SMHM model       : {args.smhm_model}")
    print(f"Star pot form    : {args.star_pot_form}")
    print()

    all_streams = []
    all_progs   = []

    generated_count        = 0
    attempt_count          = 0

    # chunksize: each worker gets a small batch to reduce scheduling overhead.
    # Tune if needed; 2–4 works well when each task takes ~seconds.
    chunksize = max(1, num_workers // 2)

    tasks = _task_iter(base_seed, args.smhm_model, args.star_pot_form)

    with Pool(processes=num_workers) as pool:
        for result in pool.imap(_try_generate_one, tasks, chunksize=chunksize):
            attempt_count += 1

            if result is None:
                continue

            stream_df, prog_df, info = result

            # Reassign canonical stream_id
            stream_df["stream_id"] = generated_count
            prog_df["stream_id"]   = generated_count

            all_streams.append(stream_df)
            all_progs.append(prog_df)
            generated_count += 1

            print(info)
            print(
                f"[{generated_count}/{args.num_streams}] "
                f"Stream accepted  (attempts so far: {attempt_count})"
            )

            if generated_count >= args.num_streams:
                pool.terminate()
                pool.join()
                break

    print(
        f"\nDone. Generated {generated_count} streams "
        f"from {attempt_count} attempts "
        f"(skip rate: {1 - generated_count/attempt_count:.1%})"
    )

    # ---- assemble dataframes -----------------------------------------------
    all_progs_df = pd.concat(all_progs, ignore_index=True)

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

    # ---- build prog_properties array ---------------------------------------
    prog_array = np.column_stack([
        all_progs_df[col]
        .apply(lambda x: x.value if hasattr(x, "value") else float(x))
        .to_numpy()
        for col in _PROG_COLS
    ])

    # ---- build label (pid) -------------------------------------------------
    label_vals = (
        all_progs_df[args.label_col]
        .apply(lambda x: x.value if hasattr(x, "value") else float(x))
        .to_numpy()
    )
    if args.log_label:
        label_vals = np.log10(label_vals)
    labels = label_vals.reshape(-1, 1)

    # ---- save to HDF5 ------------------------------------------------------
    with h5py.File(args.output, "w") as f:
        f.create_dataset("ICRS_data",      data=ICRS_raw_data)
        f.create_dataset("Cartesian_data", data=Cartesian_raw_data)

        ds_progs = f.create_dataset("prog_properties", data=prog_array)
        ds_progs.attrs["columns"] = _PROG_COLS
        ds_progs.attrs["units"]   = [_PROG_UNITS[c] for c in _PROG_COLS]

        ds_pid = f.create_dataset("pid", data=labels)
        ds_pid.attrs["label_col"]      = args.label_col
        ds_pid.attrs["log_transformed"] = args.log_label

    print("\nDataset saved to:", args.output)
    print(f"  ICRS data shape:        {ICRS_raw_data.shape}  (7 features)")
    print(f"  Cartesian data shape:   {Cartesian_raw_data.shape}  (7 features)")
    print(f"  prog_properties shape:  {prog_array.shape}  ({len(_PROG_COLS)} columns)")
    print(f"  Labels shape:           {labels.shape}  (label_col={args.label_col}, log={args.log_label})")


# ---------------------------------------------------------------------------
# Entry point — must be guarded for 'spawn' multiprocessing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)
