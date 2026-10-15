"""
raw_data_generation_subhalo.py
------------------------------
Command-line script for generating GD-1-like stellar streams perturbed by a
dark-matter subhalo, for regressing the subhalo's mass enclosed within 10 pc.

Every stream shares one fixed encounter geometry: the same progenitor orbit and
the same subhalo orbit, with the impact 495 Myr ago (the impact time of the
fiducial GD-1 spur-and-gap model of Bonaca et al. 2019, arXiv:1811.03631), so
every stream is perturbed. Only the subhalo's Hernquist mass M and scale radius
r_s vary, log-uniformly in [10^6, 10^7.5] Msun and [1, 10^1.5] pc. The label is
log10 M(<10 pc), with M(<r) = M r^2 / (r + r_s)^2 for a Hernquist profile; M and
r_s are stored alongside it in pert_properties.

The parallel loop and the HDF5 layout (shared with raw_data_generation.py)
live in stream_generation_common.py.

Usage example:
    python raw_data_generation_subhalo.py --num_streams 1000 --output subhalo_reg_dataset_1000.h5
"""

import functools
print = functools.partial(print, flush=True)

import argparse
import random

import astropy.coordinates as coord
import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
import numpy as np
import pandas as pd
from gala.dynamics import mockstream as ms
from gala.units import galactic

from stream_generation_common import (
    collect_streams,
    resolve_num_workers,
    save_dataset,
    seed_iter,
    stream_to_df,
)

# multiprocessing — use 'spawn' to avoid issues with C extensions on Linux
import multiprocessing as mp
mp.set_start_method("spawn", force=True)


# Added to num_streams to form the base seed, so this dataset's seeds never
# coincide with those of the progenitor-mass dataset of the same size.
SEED_OFFSET = 1_000_000

T_START  = -3.0 * u.Gyr
T_IMPACT = -495 * u.Myr
# 2000 release steps x 2 tails = 4000 particles per stream, the same count as
# the progenitor-mass dataset. gala integrates mock streams with the adaptive
# DOP853 integrator, so dt sets the release cadence, not the integration accuracy.
DT = 1.5 * u.Myr

PROG_MASS = 2.5e4 * u.Msun
PROG_RS   = 10 * u.pc

LOG_PERT_MASS_RANGE = (6.0, 7.5)   # log10 Msun
LOG_PERT_RS_RANGE   = (0.0, 1.5)   # log10 pc
ENCLOSED_RADIUS_PC  = 10.0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate subhalo-perturbed GD-1-like stream datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_streams", type=int, default=1000,
        help=(
            "Number of streams to generate. "
            f"num_streams + {SEED_OFFSET} is used as the base seed."
        ),
    )
    parser.add_argument(
        "--output", type=str, default="subhalo_reg_dataset.h5",
        help="Output HDF5 file path for the processed feature dataset.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help=(
            "Number of parallel worker processes. "
            "Defaults to $SLURM_CPUS_PER_TASK if set, else os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--label_col", type=str, default="pert_m10pc",
        choices=["pert_mass", "pert_r_s", "pert_m10pc"],
        help="Column from pert_properties to use as the label (pid).",
    )
    parser.add_argument(
        "--log_label", action=argparse.BooleanOptionalAction, default=True,
        help="Apply log10 to the label value. Use --no_log_label to disable.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Global setup (MW potential, galactocentric frame)
# ---------------------------------------------------------------------------

def setup_globals():
    """Spherical logarithmic MW potential and the Galactocentric frame used for
    the ICRS observables."""
    ham = gp.Hamiltonian(gp.LogarithmicPotential(
        v_c=225 * u.km / u.s, r_h=0 * u.kpc, q1=1, q2=1, q3=1, units=galactic
    ))
    gc_frame = coord.Galactocentric(galcen_distance=8 * u.kpc, z_sun=0 * u.pc)
    return ham, gc_frame


# ---------------------------------------------------------------------------
# Physical relations
# ---------------------------------------------------------------------------

def hernquist_enclosed_mass(mass, r_s, r):
    """Mass of a Hernquist sphere (total mass, scale radius r_s) within radius r."""
    return mass * r**2 / (r + r_s) ** 2


# ---------------------------------------------------------------------------
# Stream generation
# ---------------------------------------------------------------------------

def stream_gen(ham, pert_mass, pert_rs, attempt_seed):
    """Mock stream of the fixed GD-1-like progenitor, perturbed by a Hernquist
    subhalo of mass pert_mass [Msun] and scale radius pert_rs [pc]."""
    prog_w0 = gd.PhaseSpacePosition(
        pos=[-19.36473268, -7.23486128, 5.02312534] * u.kpc,
        vel=[0.07986908, -0.12925221, -0.12822027] * u.kpc / u.Myr,
        frame=ham.frame,
    )
    prog_pot = gp.HernquistPotential(PROG_MASS, PROG_RS, units=galactic)

    # The subhalo's phase-space position at impact, integrated back to the start.
    pert_at_impact = gd.PhaseSpacePosition(
        pos=[-18.198455, 2.7829913, 11.173008] * u.kpc,
        vel=[-51.051972, -216.7645, -113.60831] * u.km / u.s,
        frame=ham.frame,
    )
    pert_orbit = ham.integrate_orbit(pert_at_impact, dt=-0.1 * u.Myr, t1=T_IMPACT, t2=T_START)
    pert_pot = gp.HernquistPotential(pert_mass * u.Msun, pert_rs * u.pc, units=galactic)
    nbody = gd.DirectNBody(
        pert_orbit[-1],
        particle_potentials=[pert_pot],
        external_potential=ham.potential,
        frame=ham.frame,
    )

    rng = np.random.RandomState(attempt_seed % (2**32))
    df_model = ms.FardalStreamDF(gala_modified=False, random_state=rng)
    gen_stream = ms.MockStreamGenerator(df_model, ham, progenitor_potential=prog_pot)
    stream, _ = gen_stream.run(
        prog_w0, prog_pot.parameters["m"],
        nbody=nbody, dt=DT, t1=T_START, t2=0 * u.Gyr,
    )
    return stream


# ---------------------------------------------------------------------------
# Worker function (runs inside each subprocess)
# ---------------------------------------------------------------------------

def _try_generate_one(task):
    """
    Attempt to generate one stream.

    Parameters
    ----------
    task : tuple
        (attempt_seed,)

    Returns
    -------
    (stream_df, pert_df, info_str)  on success
    None                            on skip
    """
    (attempt_seed,) = task

    # Each worker gets its own independent RNG state
    random.seed(attempt_seed)
    np.random.seed(attempt_seed % (2**32))   # numpy seed must be < 2^32

    # Re-initialise gala/astropy objects inside the subprocess
    ham, gc_frame = setup_globals()

    pert_mass = 10 ** random.uniform(*LOG_PERT_MASS_RANGE)   # Msun
    pert_rs   = 10 ** random.uniform(*LOG_PERT_RS_RANGE)     # pc
    pert_m10pc = hernquist_enclosed_mass(pert_mass, pert_rs, ENCLOSED_RADIUS_PC)

    try:
        stream = stream_gen(ham, pert_mass, pert_rs, attempt_seed)
    except RuntimeError as e:
        if "Larger nmax is needed" in str(e):
            return None   # known benign error — skip
        raise

    # stream_id is a placeholder; the main process reassigns it
    stream_df = stream_to_df(stream, attempt_seed, galactocentric_frame=gc_frame)
    pert_df = pd.DataFrame({
        "stream_id":  [attempt_seed],
        "pert_mass":  [pert_mass],
        "pert_r_s":   [pert_rs],
        "pert_m10pc": [pert_m10pc],
    })

    info = (
        f"  Subhalo → M={pert_mass:.3e} Msun  r_s={pert_rs:.2f} pc  "
        f"M(<10 pc)={pert_m10pc:.3e} Msun"
    )
    return stream_df, pert_df, info


# ---------------------------------------------------------------------------
# Subhalo property columns saved into pert_properties dataset
# ---------------------------------------------------------------------------

_PERT_COLS = ["pert_mass", "pert_r_s", "pert_m10pc"]

_PERT_UNITS = {
    "pert_mass":  "Msun",
    "pert_r_s":   "pc",
    "pert_m10pc": "Msun",
}


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset(args):
    base_seed = args.num_streams + SEED_OFFSET
    num_workers = resolve_num_workers(args.num_workers)

    print(f"Random base seed : {base_seed}")
    print(f"Target streams   : {args.num_streams}")
    print(f"Worker processes : {num_workers}")
    print(f"Release step dt  : {DT}")
    print()

    all_streams, all_perts_df = collect_streams(
        _try_generate_one, seed_iter(base_seed), args.num_streams, num_workers
    )

    save_dataset(
        args.output, all_streams, all_perts_df,
        prop_cols=_PERT_COLS, prop_units=_PERT_UNITS, props_key="pert_properties",
        label_col=args.label_col, log_label=args.log_label,
    )


# ---------------------------------------------------------------------------
# Entry point — must be guarded for 'spawn' multiprocessing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)
