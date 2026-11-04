import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Paper style: LaTeX text in Helvetica, as in build_results_table.py.
# Needs LaTeX on PATH: run `module load texlive/2024` after activating the conda env, or conda takes PATH priority.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
})

def parse_args():
    """Parses the command line.

    Returns the argparse Namespace.
    """
    parser = argparse.ArgumentParser(description="Stream visualizations: mass-eigenvalue scaling relations and a presentation-quality single-stream figure.")
    parser.add_argument('--input_raw', type=str, required=True, help="Path to the raw HDF5 dataset")

    # Scaling plot (--plot_scaling).
    parser.add_argument('--plot_scaling', action='store_true', help="If set, loop over all streams and plot Mass vs. Eigenvalues")
    parser.add_argument('--output_scaling', type=str, default="mass_vs_lambda.png", help="Output path for the scaling plot")
    parser.add_argument('--mass_key', type=str, default="pid", help="HDF5 key for progenitor mass in raw data (default: pid)")
    # Single-stream figure (--plot_stream_beauty).
    parser.add_argument('--plot_stream_beauty', action='store_true',
                        help="If set, plot a presentation-quality single stream (dark bg, PCA coloring)")
    parser.add_argument('--output_stream_beauty', type=str, default=None,
                        help="Output path for the beauty stream plot (default: stream_beauty_idx<N>.png)")
    parser.add_argument('--beauty_stream_idx', type=int, default=0,
                        help="Index of the stream to visualize in beauty mode (default: 0)")
    parser.add_argument('--beauty_cmap', type=str, default="plasma",
                        help="Colormap for beauty stream particles (default: plasma)")
    parser.add_argument('--beauty_mw_potential', action='store_true',
                        help="If set, overlay MW potential as background colormap (uses real galactocentric coordinates)")
    parser.add_argument('--beauty_planes', type=str, nargs='+', default=['xy', 'xz', 'yz'],
                        choices=['xy', 'xz', 'yz', 'all'],
                        help="Which projection planes to show (default: xy xz yz). Use 'all' for all three.")

    return parser.parse_args()

def _extract_sigmas(cart_ds):
    """Cartesian_data (n_streams, n_points, >=3) -> (n_streams, 3) sqrt of each stream's covariance eigenvalues, sorted descending, in kpc."""
    num_streams = cart_ds.shape[0]
    all_sigmas = np.zeros((num_streams, 3))
    for i in range(num_streams):
        if i % 1000 == 0:
            print(f"  Processing stream {i}/{num_streams}...")
        pts = cart_ds[i, :, 0:3]
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[np.argsort(eigvals)[::-1]]
        all_sigmas[i] = np.sqrt(np.maximum(eigvals, 0))
    return all_sigmas


def plot_scaling_relations(args, f):
    """Scatters each stream's PCA scales sqrt(lambda_j) against its log progenitor mass, one panel per j, each with a least-squares line.

    Returns nothing; writes args.output_scaling and a PDF of the same name next to it, and raises KeyError if args.mass_key is not in the file.

    `args`: the parsed command line (uses mass_key and output_scaling).
    `f`: the open raw HDF5 file.
    """
    if args.mass_key not in f:
        raise KeyError(f"Key '{args.mass_key}' not found. Available: {list(f.keys())}")

    masses = f[args.mass_key][:]

    print("Extracting eigenvalues for all streams...")
    all_sigmas = _extract_sigmas(f['Cartesian_data'])

    print(f"Plotting scaling relations to {args.output_scaling}...")
    # The paper includes this at \columnwidth (397.5pt = 5.5in), so the 18in-wide figure is scaled to ~0.31x and a default 10pt label would land at about 3pt.
    # The font sizes below are therefore chosen in source units to come out at roughly 6-9pt once scaled.
    # Changing figsize means rescaling all of them.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 30, 26, 22, 20

    titles = [r'$\sqrt{\lambda_1}$', r'$\sqrt{\lambda_2}$', r'$\sqrt{\lambda_3}$']
    ylabels = [r'$\sqrt{\lambda_1}\ \mathrm{[kpc]}$',
               r'$\sqrt{\lambda_2}\ \mathrm{[kpc]}$',
               r'$\sqrt{\lambda_3}\ \mathrm{[kpc]}$']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for j in range(3):
        ax = axes[j]
        ax.scatter(masses, all_sigmas[:, j], s=5, alpha=0.3, c=colors[j], edgecolors='none')

        # Least-squares line, with its slope and R^2 in the legend.
        m_flat = masses.flatten()
        s_flat = all_sigmas[:, j].flatten()
        slope, intercept, r_value, _, _ = stats.linregress(m_flat, s_flat)
        x_fit = np.linspace(m_flat.min(), m_flat.max(), 200)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='k', linewidth=1.5, linestyle='--',
                label=f'slope={slope:.2f}, $R^2$={r_value**2:.2f}')
        ax.legend(fontsize=LEGEND_FS, loc='upper left')

        ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$', fontsize=LABEL_FS)
        ax.set_ylabel(ylabels[j], fontsize=LABEL_FS)
        ax.set_title(titles[j], fontsize=TITLE_FS)
        ax.tick_params(axis='both', labelsize=TICK_FS)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(args.output_scaling, dpi=300)
    pdf_output_scaling = os.path.splitext(args.output_scaling)[0] + '.pdf'
    plt.savefig(pdf_output_scaling, dpi=300)
    print(f"Saved scaling plot -> {args.output_scaling}, {pdf_output_scaling}")
    print("Done!")

def _mw_potential_slice(potential, axis_i, axis_j, pts, n=300):
    """MW potential on an n x n grid over axes (axis_i, axis_j), reaching 30% of the points' range past them on each side, at the third axis's median; returns (gi, gj, values)."""
    import astropy.units as u
    all_axes = [0, 1, 2]
    axis_k = [a for a in all_axes if a not in (axis_i, axis_j)][0]

    i_min, i_max = pts[:, axis_i].min(), pts[:, axis_i].max()
    j_min, j_max = pts[:, axis_j].min(), pts[:, axis_j].max()
    pad_i = (i_max - i_min) * 0.3
    pad_j = (j_max - j_min) * 0.3
    gi = np.linspace(i_min - pad_i, i_max + pad_i, n)
    gj = np.linspace(j_min - pad_j, j_max + pad_j, n)
    GI, GJ = np.meshgrid(gi, gj)
    GK = np.full_like(GI, np.median(pts[:, axis_k]))

    coords = np.zeros((3, n * n))
    coords[axis_i] = GI.ravel()
    coords[axis_j] = GJ.ravel()
    coords[axis_k] = GK.ravel()

    q = coords * u.kpc
    pot_vals = potential.energy(q).value.reshape(n, n)
    return gi, gj, pot_vals


def plot_stream_beauty(args, f):
    """Draws one stream for presentations: dark background, particles colored along the first principal axis, no axes.

    Returns nothing; writes args.output_stream_beauty (default stream_beauty_idx<N>.png).

    `args`: the parsed command line (uses the beauty_* options).
    `f`: the open raw HDF5 file.
    """
    if args.output_stream_beauty is None:
        args.output_stream_beauty = f"stream_beauty_idx{args.beauty_stream_idx}.png"
    pts = f['Cartesian_data'][args.beauty_stream_idx, :, 0:3].astype(float)

    # Centre the stream, unless the potential is drawn, which needs the real galactocentric coordinates.
    if not args.beauty_mw_potential:
        pts = pts - pts.mean(axis=0)

    # Color by the projection onto the first principal component, the stream's long axis.
    cov = np.cov(pts.T)
    _, vecs = np.linalg.eigh(cov)
    pc1 = vecs[:, -1]
    proj = pts @ pc1
    c = (proj - proj.min()) / (proj.max() - proj.min())

    # The --beauty_planes choices as axis-index pairs.
    plane_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    requested = args.beauty_planes
    if 'all' in requested:
        requested = ['xy', 'xz', 'yz']
    pairs = [plane_map[p] for p in requested]
    labels = {0: 'X (kpc)', 1: 'Y (kpc)', 2: 'Z (kpc)'}

    n_panels = len(pairs)
    fig_w = 5 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5), facecolor='black')
    if n_panels == 1:
        axes = [axes]

    potential = None
    if args.beauty_mw_potential:
        import gala.potential as gp
        potential = gp.MilkyWayPotential()

    for ax, (i, j) in zip(axes, pairs):
        ax.set_facecolor('black')

        if potential is not None:
            gi, gj, pot_vals = _mw_potential_slice(potential, i, j, pts)
            ax.imshow(pot_vals, origin='lower',
                      extent=[gi[0], gi[-1], gj[0], gj[-1]],
                      cmap='magma', alpha=0.4, aspect='auto',
                      vmin=np.percentile(pot_vals, 5),
                      vmax=np.percentile(pot_vals, 95))

        ax.scatter(pts[:, i], pts[:, j], c=c, cmap=args.beauty_cmap,
                   s=0.8, alpha=0.5, linewidths=0, rasterized=True)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(args.output_stream_beauty, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"Saved {args.output_stream_beauty}")


def main():
    """Makes the plots requested on the command line from --input_raw.

    Returns nothing; exits if neither --plot_scaling nor --plot_stream_beauty is passed.
    """
    args = parse_args()

    if not (args.plot_scaling or args.plot_stream_beauty):
        raise SystemExit("Nothing to do: pass --plot_scaling and/or --plot_stream_beauty.")

    with h5py.File(args.input_raw, 'r') as f:
        if args.plot_scaling:
            plot_scaling_relations(args, f)
        if args.plot_stream_beauty:
            plot_stream_beauty(args, f)

if __name__ == "__main__":
    main()