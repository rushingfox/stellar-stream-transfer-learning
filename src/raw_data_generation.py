"""Generates the synthetic stellar-stream dataset of the progenitor-mass task, whose target is log10 of the progenitor's stellar mass.

It runs in parallel with multiprocessing.Pool on NERSC CPU nodes; the parallel loop and the HDF5 writer live in stream_generation_common.py.

Usage example:
    python raw_data_generation.py --num_streams 1000 --output prog_mass_reg_dataset_1000.h5
"""

import functools
print = functools.partial(print, flush=True)

import argparse
import math
import random

import astropy.coordinates as coord
import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
import numpy as np
import pandas as pd
from colossus.cosmology import cosmology
from colossus.halo import concentration, mass_defs
from gala.dynamics import mockstream as ms
from gala.units import galactic
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist

from stream_generation_common import (
    collect_streams,
    resolve_num_workers,
    save_dataset,
    seed_iter,
    stream_to_df,
)

# Workers are spawned rather than forked, to avoid issues with C extensions on Linux.
import multiprocessing as mp
mp.set_start_method("spawn", force=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parses the command line.

    Returns the argparse Namespace.

    `argv`: the argument list, or None for sys.argv.
    """
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
    """Sets astropy's Galactocentric frame defaults to v4.0 and builds the cosmology and the Milky Way potential.

    Returns (cosmo, mw): colossus's planck18 cosmology and gala's BovyMWPotential2014.
    """
    _ = coord.galactocentric_frame_defaults.set("v4.0")
    cosmo = cosmology.setCosmology("planck18", persistence="")
    mw = gp.BovyMWPotential2014(units=galactic)
    return cosmo, mw


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ProgenitorParams:
    """Initial conditions, stellar mass and integration times of one stream's progenitor, as astropy quantities.

    `ra`, `dec`, `distance`: the progenitor's ICRS position.
    `pm_ra_cosdec`, `pm_dec`, `radial_velocity`: its ICRS proper motions and line-of-sight velocity.
    `mstar`: its stellar mass.
    `t1`, `t2`, `dt`: the start, end and step of the integration.
    """
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
        """Iterates over (attribute name, value) pairs, e.g. ('mstar', mstar)."""
        return iter(self.__dict__.items())

    def to_dict(self):
        """The attributes as {name: value}."""
        return {key: value for key, value in self}


# ---------------------------------------------------------------------------
# Physical relations
# ---------------------------------------------------------------------------

def SMHM(Mstar_in_Msun, smhm_model: str = "FIRE_LatteELVIS"):
    """Stellar-to-halo mass relation: the halo mass M200c of a given stellar mass.

    Returns M200c in Msun, interpolated linearly in log-log (and extrapolated) from the table ./<smhm_model>_SMHM.

    `Mstar_in_Msun`: the stellar mass in Msun.
    `smhm_model`: the table's name without the '_SMHM' suffix; the default is the FIRE Latte/ELVIS fit in Fig. 2(a) of https://arxiv.org/pdf/2206.05295.
    """
    data = np.genfromtxt(f"./{smhm_model}_SMHM")
    log_Mstar = data[:, 0]
    log_Mhalo = data[:, 1]
    interp = interp1d(log_Mstar, log_Mhalo, kind="linear", fill_value="extrapolate")
    M200c_in_Msun = 10 ** interp(np.log10(Mstar_in_Msun))
    return M200c_in_Msun


def SSHS(Mhalo_200c_in_Msun, z, cosmo):
    """Stellar-size halo-size relation: the stellar half-light radius of a halo.

    Returns R_e = 0.02 (1 + z)^-0.2 (C_vir / 10)^-0.7 R_vir in kpc, with the Ludlow et al. 2016 concentration converted from the 200c to the virial definition.

    `Mhalo_200c_in_Msun`: the halo mass M200c in Msun.
    `z`: the redshift.
    `cosmo`: the colossus cosmology, for h = H0 / 100, since colossus takes masses in Msun/h and returns radii in kpc/h.
    """
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
    """Converts a half-light radius to the scale radius of a Plummer or Hernquist profile.

    Returns the scale radius in kpc, R_e sqrt(2^(2/3) - 1) for Plummer and R_e / (1 + sqrt(2)) for Hernquist; raises ValueError for any other profile.

    `Re_in_kpc`: the half-light radius in kpc.
    `profile`: 'plummer' or 'hernquist'.
    """
    if profile == "plummer":
        return Re_in_kpc * math.sqrt(2 ** (2 / 3) - 1)
    elif profile == "hernquist":
        return Re_in_kpc / (1.0 + math.sqrt(2.0))
    else:
        raise ValueError(
            "Unsupported profile type. Choose from 'plummer' or 'hernquist'."
        )


def max_distance(x, y, z):
    """Computes the largest distance between any two of the points.

    Returns that distance, in the unit of the coordinates.

    `x`, `y`, `z`: the points' coordinates.
    """
    coords = np.column_stack([x, y, z])
    return np.max(pdist(coords))


# ---------------------------------------------------------------------------
# Orbit integration
# ---------------------------------------------------------------------------

def compute_orbit(progenitor_params: ProgenitorParams, mw, whole_time_space: bool = False):
    """Integrates the progenitor's orbit in the Milky Way potential.

    Returns the gala Orbit, from t1 to t2 if whole_time_space, else from -1.5 Gyr to 0, with step dt.

    `progenitor_params`: the progenitor's ICRS initial conditions and times.
    `mw`: the Milky Way potential.
    `whole_time_space`: integrate over the progenitor's own t1 to t2 instead of the fixed -1.5 Gyr to 0.
    """
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
    """Generates the mock stream of one progenitor in the Milky Way potential.

    Returns (stream_df, prog_df): the per-particle table, and a one-row table of the progenitor (progenitor_type 'DG', stream_id, the orbit's pericenter, apocenter and eccentricity, Criterion_2, and every ProgenitorParams field as progenitor_<name>).

    The progenitor potential is an NFW halo plus a stellar Plummer or Hernquist profile: the halo's M200c comes from SMHM and its concentration from Ludlow et al. 2016 at the redshift of lookback time -t1, and the stellar scale radius from the SSHS half-light radius.
    Criterion_2 is 1 if the stream's largest pairwise particle distance exceeds 120 kpc, else 0.

    `progenitor_params`: the progenitor's initial conditions, stellar mass and times.
    `stream_id`: the id written in both tables.
    `orbit`: the progenitor's orbit from compute_orbit, for its pericenter, apocenter and eccentricity.
    `mw`: the Milky Way potential.
    `cosmo`: the colossus cosmology.
    `smhm_model`: the SMHM table's name (see SMHM).
    `star_pot_form`: the stellar profile, 'hernquist' or 'plummer'.
    `attempt_seed`: seeds the stream's particle release.
    """
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

    stream_df = stream_to_df(stream, stream_id)

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
    """Draws one progenitor and generates its stream, in a worker process.

    Returns (stream_df, prog_df, info_str) for an accepted stream; None to skip it (NaN orbit, pericenter above 100 kpc, Criterion_2 failed, or gala's "Larger nmax is needed" error).

    `task`: (attempt_seed, smhm_model, star_pot_form), from seed_iter.
    """
    attempt_seed, smhm_model, star_pot_form = task

    # Every attempt reseeds the RNGs with its own seed, so its draws depend only on that seed.
    random.seed(attempt_seed)
    np.random.seed(attempt_seed % (2**32))   # numpy seed must be < 2^32

    # The gala/astropy objects are rebuilt inside the subprocess.
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
    """Generates args.num_streams accepted streams, with num_streams as the base seed.

    Returns nothing; writes args.output through save_dataset.

    `args`: the parsed command line, from parse_args.
    """
    base_seed = args.num_streams
    random.seed(base_seed)
    np.random.seed(base_seed)

    num_workers = resolve_num_workers(args.num_workers)

    print(f"Random base seed : {base_seed}")
    print(f"Target streams   : {args.num_streams}")
    print(f"Worker processes : {num_workers}")
    print(f"SMHM model       : {args.smhm_model}")
    print(f"Star pot form    : {args.star_pot_form}")
    print()

    tasks = seed_iter(base_seed, args.smhm_model, args.star_pot_form)
    all_streams, all_progs_df = collect_streams(
        _try_generate_one, tasks, args.num_streams, num_workers
    )

    save_dataset(
        args.output, all_streams, all_progs_df,
        prop_cols=_PROG_COLS, prop_units=_PROG_UNITS, props_key="prog_properties",
        label_col=args.label_col, log_label=args.log_label,
    )


# ---------------------------------------------------------------------------
# Entry point — must be guarded for 'spawn' multiprocessing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)
