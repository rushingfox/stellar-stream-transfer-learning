r"""Draws the gallery of the proxy pretraining tasks, one figure per proxy, each with its regression target marked.

Each proxy gets its own figure, so that no correspondence between proxies is implied (the cube, for example, has no analogue in real streams).
The synthetic shapes are sampled afresh here, the same way src/prepare_data.py samples them; the eigen panels are real streams from --input_raw.
In gaussian_ellipsoid_sigmax, bounded_ellipsoid_a, box_a and eigen the point cloud is genuinely 3D and anisotropic, but only the first (longest) axis is the target: it is drawn in the accent color, and the other two axes are muted, for shape context only.
Every figure is written as PNG and as PDF.

Output files (under --output_dir):
    gallery_cube.png                    proxy_cube                       - target = $a$
    gallery_gaussian_ellipsoid_sigmax.png proxy_gaussian_ellipsoid_sigmax - target = $\sigma_x$
    gallery_bounded_ball.png            proxy_bounded_ball               - target = $r$
    gallery_gaussian_ball.png           proxy_gaussian_ball              - target = $\sigma$
    gallery_bounded_ellipsoid_a.png     proxy_bounded_ellipsoid_a        - target = $a$
    gallery_box_a.png                   proxy_box_a                      - target = $a$
    gallery_eigen.png                   proxy_stream_eigenvalues_lambda1 - target = $\sqrt{\lambda_1}$
    gallery_combined.png                the six synthetic proxies, 2 rows (isotropic, anisotropic) x 3 columns, plus one real stream
                                        in a 4th column, drawn with its PCA target and with its progenitor-mass target (needs --input_raw)
"""
import argparse
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# Paper style: LaTeX text in Computer Modern, the serif of the paper body, as in build_results_table.py.
# Needs LaTeX on PATH: run `module load texlive/2024` after activating the conda env, or conda takes PATH priority.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Proxy names in the gallery suptitles (make_gallery draws none for eigen).
_PROXY_DISPLAY_NAMES = {
    'cube':                     'proxy_cube',
    'bounded_ball':             'proxy_bounded_ball',
    'gaussian_ball':            'proxy_gaussian_ball',
    'gaussian_ellipsoid_sigmax': 'proxy_gaussian_ellipsoid_sigmax',
    'bounded_ellipsoid_a':      'proxy_bounded_ellipsoid_a',
    'box_a':                    'proxy_box_a',
    'eigen':                    'proxy_stream_eigenvalues_lambda1',
}

# Target in each gallery suptitle, in LaTeX.
# For the anisotropic proxies it is the one scalar actually trained on, and the other shape parameters are named only for context.
_TARGET_DESCRIPTIONS = {
    'cube':                     r'$a$',
    'bounded_ball':             r'$r$',
    'gaussian_ball':            r'$\sigma$',
    'gaussian_ellipsoid_sigmax': r'$\sigma_x$ (shape also has $\sigma_y,\sigma_z$)',
    'bounded_ellipsoid_a':      r'$a$ (shape also has $b,c$)',
    'box_a':                    r'$a$ (shape also has $b,c$)',
    'eigen':                    r'$\sqrt{\lambda_1}$ (shape also has $\sqrt{\lambda_2},\sqrt{\lambda_3}$)',
}

ALL_PROXY_TYPES = ['cube', 'gaussian_ellipsoid_sigmax', 'bounded_ball',
                   'gaussian_ball', 'bounded_ellipsoid_a', 'box_a', 'eigen', 'combined']

VIEW_INIT = dict(elev=20, azim=45)
# matplotlib is pinned <3.11 (pyproject.toml), because 3.11's axes3d.depthshade_minalpha multiplies into this alpha and washes the point clouds out.
POINT_KW = dict(s=1, c='steelblue', alpha=0.2, edgecolors='none', zorder=1)
ACCENT = 'crimson'
WIREFRAME_COLOR = '#555555'  # neutral dark gray, won't clash with the colored axis arrows
# Default per-axis arrow colors of draw_principal_axes; every gallery panel passes its own (ACCENT and MUTED_AXIS_COLOR).
AXIS_COLORS = ('#e03030', '#d68020', '#9467bd')  # red, amber, purple (colorblind-friendly)
# In the anisotropic and eigen panels the target axis (dim 0) is drawn in ACCENT and the other two in this gray, as shape context that is not part of the target.
MUTED_AXIS_COLOR = '#999999'
LABEL_BBOX = dict(facecolor='white', alpha=0.0, edgecolor='none', pad=1.2)
# Draw ellipsoids at 2-sigma so they stay visible even when one axis is tiny.
ELLIPSOID_K = 2.0


# ---------- random helpers ----------

def random_rotation_matrix(rng):
    """Draws a random 3D rotation, rz @ ry @ rx with three angles uniform in [0, 2pi), the convention of src/prepare_data.py.

    Returns the 3x3 rotation matrix.

    `rng`: a numpy Generator.
    """
    theta = rng.uniform(0, 2 * np.pi, 3)
    rx = np.array([[1, 0, 0],
                   [0, np.cos(theta[0]), -np.sin(theta[0])],
                   [0, np.sin(theta[0]),  np.cos(theta[0])]])
    ry = np.array([[ np.cos(theta[1]), 0, np.sin(theta[1])],
                   [0, 1, 0],
                   [-np.sin(theta[1]), 0, np.cos(theta[1])]])
    rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                   [np.sin(theta[2]),  np.cos(theta[2]), 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


# ---------- proxy synthesis (mirrors src/prepare_data.py) ----------

def synth_cube(a, n_particles, rng, rotated=False):
    """Samples a uniformly filled cube of edge a, centred at the origin.

    Returns (points, R): the (n_particles, 3) points and the rotation applied (the identity if not rotated).

    `a`: the edge length, the cube proxy's target.
    `n_particles`: the number of points.
    `rng`: a numpy Generator.
    `rotated`: apply a random rotation, as training does.
    """
    points = rng.uniform(-a / 2, a / 2, size=(n_particles, 3))
    R = random_rotation_matrix(rng) if rotated else np.eye(3)
    return points @ R.T, R


def synth_ellipsoid(sigmas, n_particles, rng, rotated=False):
    """Samples a 3D Gaussian with one sigma per axis.

    Returns (points, R): the (n_particles, 3) points and the rotation applied (the identity if not rotated).

    `sigmas`: (sigma_x, sigma_y, sigma_z); sigma_x is the gaussian_ellipsoid_sigmax target.
    `n_particles`, `rng`, `rotated`: as in synth_cube.
    """
    sx, sy, sz = sigmas
    points = np.stack([
        rng.normal(0, sx, n_particles),
        rng.normal(0, sy, n_particles),
        rng.normal(0, sz, n_particles),
    ], axis=1)
    R = random_rotation_matrix(rng) if rotated else np.eye(3)
    return points @ R.T, R


def _sample_isotropic_directions(n_particles, rng):
    """n_particles unit vectors, uniform on the sphere (normalized Gaussian draws)."""
    dirs = rng.normal(size=(n_particles, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def synth_ball(r, n_particles, rng):
    """Samples a uniformly filled ball of radius r; a radius of r * U^(1/3) gives the uniform volume density.

    Returns the (n_particles, 3) points, unrotated, since a ball is isotropic.

    `r`: the radius, the bounded_ball target.
    `n_particles`, `rng`: as in synth_cube.
    """
    dirs = _sample_isotropic_directions(n_particles, rng)
    radii = r * rng.uniform(0.0, 1.0, size=n_particles) ** (1.0 / 3.0)
    return dirs * radii[:, None]


def synth_gaussian_ball(sigma, n_particles, rng):
    """Samples an isotropic 3D Gaussian.

    Returns the (n_particles, 3) points.

    `sigma`: the std on every axis, the gaussian_ball target.
    `n_particles`, `rng`: as in synth_cube.
    """
    return rng.normal(0.0, sigma, size=(n_particles, 3))


def synth_hard_ellipsoid(semi_axes, n_particles, rng, rotated=False):
    """Samples a uniformly filled ellipsoid with a hard boundary: a uniform unit ball scaled per axis, which keeps the density uniform.

    Returns (points, R): the (n_particles, 3) points and the rotation applied (the identity if not rotated).

    `semi_axes`: (a, b, c); a is the bounded_ellipsoid_a target.
    `n_particles`, `rng`, `rotated`: as in synth_cube.
    """
    a, b, c = semi_axes
    dirs = _sample_isotropic_directions(n_particles, rng)
    radii = rng.uniform(0.0, 1.0, size=n_particles) ** (1.0 / 3.0)
    inside_unit_ball = dirs * radii[:, None]
    points = inside_unit_ball * np.array([a, b, c])
    R = random_rotation_matrix(rng) if rotated else np.eye(3)
    return points @ R.T, R


def synth_box(semi_edges, n_particles, rng, rotated=False):
    """Samples the uniformly filled box [-a, a] x [-b, b] x [-c, c].

    Returns (points, R): the (n_particles, 3) points and the rotation applied (the identity if not rotated).

    `semi_edges`: (a, b, c), the half edge lengths; a is the box_a target.
    `n_particles`, `rng`, `rotated`: as in synth_cube.
    """
    a, b, c = semi_edges
    points = np.stack([
        rng.uniform(-a, a, n_particles),
        rng.uniform(-b, b, n_particles),
        rng.uniform(-c, c, n_particles),
    ], axis=1)
    R = random_rotation_matrix(rng) if rotated else np.eye(3)
    return points @ R.T, R


def compute_pca_axes(points):
    """Runs a PCA of one point cloud.

    Returns (sigmas, eigvecs, center): the square roots of the covariance eigenvalues sorted descending, the matching eigenvectors as columns (used as the rotation R), and the centroid.

    `points`: an (n, 3) array.
    """
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]
    sigmas = np.sqrt(np.maximum(eigvals, 0))
    return sigmas, eigvecs, center


# ---------- geometry helpers ----------

def cube_edges(a, R):
    """Gives the 12 edges of the cube of edge a, centred at the origin and rotated by R.

    Returns a list of (start, end) vertex pairs.

    `a`: the edge length.
    `R`: the 3x3 rotation.
    """
    h = a / 2
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    verts = (signs * h) @ R.T
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if int(np.sum(signs[i] != signs[j])) == 1:
                edges.append((verts[i], verts[j]))
    return edges


def box_edges(semi_edges, R):
    """Gives the 12 edges of a box, as cube_edges does but with one half-length per axis.

    Returns a list of (start, end) vertex pairs.

    `semi_edges`: (a, b, c), the half edge lengths.
    `R`: the 3x3 rotation.
    """
    a, b, c = semi_edges
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    verts = (signs * np.array([a, b, c])) @ R.T
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if int(np.sum(signs[i] != signs[j])) == 1:
                edges.append((verts[i], verts[j]))
    return edges


def ellipsoid_surface(sigmas, R, scale=1.0, n_u=24, n_v=16):
    """Gives a grid of points on an ellipsoid surface, for plot_wireframe.

    Returns (X, Y, Z), each an (n_u, n_v) array.

    `sigmas`: the three semi-axes, before `scale`.
    `R`: the 3x3 rotation.
    `scale`: multiplier on the sigmas (ELLIPSOID_K draws the 2-sigma surface).
    `n_u`, `n_v`: the number of grid points in azimuth and in polar angle.
    """
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    sx, sy, sz = (s * scale for s in sigmas)
    x = sx * np.outer(np.cos(u), np.sin(v))
    y = sy * np.outer(np.sin(u), np.sin(v))
    z = sz * np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([x, y, z], axis=-1)  # (n_u, n_v, 3)
    pts_rot = pts @ R.T
    return pts_rot[..., 0], pts_rot[..., 1], pts_rot[..., 2]


def set_equal_aspect(ax, points, padding=1.15, half_extent=None):
    """Gives a 3D axes the same range on all three axes, so shapes are not distorted.

    Returns nothing.

    `ax`: the 3D axes.
    `points`: the points; the range is centred on their bounding box.
    `padding`: multiplier on half the points' largest extent, used when half_extent is None.
    `half_extent`: a fixed half-width, e.g. shared by all panels of a figure so their sizes compare; None fits the points.
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2
    if half_extent is None:
        half_extent = (maxs - mins).max() / 2 * padding
    ax.set_xlim(centers[0] - half_extent, centers[0] + half_extent)
    ax.set_ylim(centers[1] - half_extent, centers[1] + half_extent)
    ax.set_zlim(centers[2] - half_extent, centers[2] + half_extent)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


# ---------- annotation helpers ----------

def draw_cube_length_arrow(ax, a, R, color=ACCENT):
    """Draws a double-headed arrow labelled a along one cube edge, just outside the cube.

    Returns nothing.

    `ax`: the 3D axes.
    `a`, `R`: the cube's edge length and rotation.
    `color`: the color of the arrow and its label.
    """
    h = a / 2
    # Along the unrotated +x edge, shifted in unrotated -z so it sits below the cube and is not mistaken for a wireframe edge.
    z_off = a * 0.06
    p_start = R @ np.array([-h, -h, -h - z_off])
    p_end   = R @ np.array([ h, -h, -h - z_off])
    mid = (p_start + p_end) / 2
    half = (p_end - p_start) / 2

    # Two single-headed quivers from the midpoint, in opposite directions.
    ax.quiver(*mid, *half,
              color=color, arrow_length_ratio=0.18, linewidth=2)
    ax.quiver(*mid, *(-half),
              color=color, arrow_length_ratio=0.18, linewidth=2)

    # Straight down in world coordinates, not in the cube's own frame, so the label lands below the arrow on screen whatever the rotation.
    # At VIEW_INIT's 20 degree elevation, world down is close to screen down, while a shift in the cube's frame can point along the line of sight and leave the label on the arrow.
    # va='top' hangs the label below that point; the default baseline alignment would grow it upward, back onto the arrow.
    text_off = np.array([0, 0, -a * 0.15])
    ax.text(*(mid + text_off), '$a$',
            color=color, fontsize=11, ha='center', va='top', fontweight='bold',
            bbox=LABEL_BBOX)


def draw_principal_axes(ax, sigmas, R, names, scale=1.0, with_values=False,
                        colors=None, label_pad=1.5):
    """Draws the three half-axes as arrows from the origin, each labelled near its tip.

    Returns nothing.

    `ax`: the 3D axes.
    `sigmas`: the three half-axis lengths, before `scale`; an axis with sigma <= 0 is skipped.
    `R`: the rotation of the shape.
    `names`: three LaTeX labels; an empty string draws the arrow without a label.
    `scale`: multiplier on the sigmas, matching the ellipsoid surface's scale.
    `with_values`: append "=value" to each label, with the unscaled sigma.
    `colors`: three colors; defaults to AXIS_COLORS.
    `label_pad`: puts each label at this multiple of its arrow's shaft vector; one float for all axes, or a tuple of three.
    """
    if colors is None:
        colors = AXIS_COLORS
    pads = (label_pad, label_pad, label_pad) if np.isscalar(label_pad) else label_pad
    for i, (sigma, name, color, pad) in enumerate(zip(sigmas, names, colors, pads)):
        if sigma <= 0:
            continue
        axis_local = np.zeros(3)
        axis_local[i] = sigma * scale
        v = R @ axis_local          # shaft endpoint = parameter value
        shaft_len = np.linalg.norm(v)
        # Fixed absolute arrowhead length, so all three axes look the same.
        head_abs = max(sigmas) * scale * 0.12   # 12% of longest axis
        head_ratio = head_abs / (shaft_len + head_abs) if shaft_len > 0 else 0.18
        vq = v + v / shaft_len * head_abs if shaft_len > 0 else v
        ax.quiver(0, 0, 0, vq[0], vq[1], vq[2],
                  color=color, arrow_length_ratio=head_ratio, linewidth=1.2, zorder=10)
        tip = v * pad
        label = f'{name}={sigma:.2f}' if with_values else name
        ax.text(*tip, label, color=color, fontsize=10, ha='center',
                fontweight='bold', bbox=LABEL_BBOX, zorder=10)


# ---------- gallery builders ----------

def render_cube_sample(ax, a, n_particles, rng, rotated=False, half_extent=None):
    """Draws one cube sample with its wireframe edges and edge-length arrow, titled with a.

    Returns nothing.

    `ax`: the 3D axes.
    `a`: the edge length.
    `n_particles`, `rng`, `rotated`: passed to synth_cube.
    `half_extent`: the panel's fixed half-width (see set_equal_aspect); None fits the points.
    """
    points, R = synth_cube(a, n_particles, rng, rotated=rotated)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **POINT_KW)
    for p1, p2 in cube_edges(a, R):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='dimgray', lw=0.8, alpha=0.7)
    draw_cube_length_arrow(ax, a, R)
    set_equal_aspect(ax, points, half_extent=half_extent)
    ax.view_init(**VIEW_INIT)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f'$a = {a:.2f}$', fontsize=10)


def render_ellipsoid_sample(ax, sigmas, n_particles, rng, rotated=False, half_extent=None):
    """Draws one Gaussian-ellipsoid sample with its 2-sigma wireframe and half-axes, titled with the target sigma_x and the two shape sigmas.

    The cloud is genuinely 3D and anisotropic, but only sigma_x (dim 0) is the gaussian_ellipsoid_sigmax target, so the sigma_y and sigma_z axes are muted.

    Returns nothing.

    `ax`: the 3D axes.
    `sigmas`: (sigma_x, sigma_y, sigma_z).
    `n_particles`, `rng`, `rotated`: passed to synth_ellipsoid.
    `half_extent`: the panel's fixed half-width; None uses 1.2 x the 2-sigma longest axis.
    """
    points, R = synth_ellipsoid(sigmas, n_particles, rng, rotated=rotated)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **POINT_KW)
    X, Y, Z = ellipsoid_surface(sigmas, R, scale=ELLIPSOID_K)
    ax.plot_wireframe(X, Y, Z, color=WIREFRAME_COLOR, alpha=0.28, linewidth=0.6)
    # The values are in the title, so the in-plot labels carry names only and do not pile up at the origin.
    # Only sigma_x, the target, is accented; sigma_y and sigma_z are muted.
    draw_principal_axes(ax, sigmas, R,
                        names=(r'$2\sigma_x$', r'$2\sigma_y$', r'$2\sigma_z$'),
                        scale=ELLIPSOID_K, with_values=False,
                        colors=(ACCENT, MUTED_AXIS_COLOR, MUTED_AXIS_COLOR),
                        label_pad=(1.4, 3.5, 2.0))
    # The box is sized from the 2-sigma longest axis, not from the points, whose long Gaussian tails would blow it up and shrink the wireframe to a dot.
    if half_extent is None:
        half_extent = max(sigmas) * ELLIPSOID_K * 1.2
    set_equal_aspect(ax, points, half_extent=half_extent)
    ax.view_init(**VIEW_INIT)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(
        rf'target: $\sigma_x = {sigmas[0]:.2f}$'
        rf'  (shape also has $\sigma_y,\sigma_z = {sigmas[1]:.2f},\ {sigmas[2]:.2f}$)',
        fontsize=9,
    )


def _draw_radius_arrow(ax, length, label, color=ACCENT, axis_dir=np.array([1.0, 0.0, 0.0]),
                       text_pad=1.25):
    """Labelled arrow from the origin along axis_dir whose shaft is `length` long, with the head beyond it."""
    v = axis_dir * length           # shaft endpoint = parameter value
    head_ratio = 0.12
    vq = v / (1 - head_ratio)      # total quiver so shaft = length
    ax.quiver(0, 0, 0, vq[0], vq[1], vq[2],
              color=color, arrow_length_ratio=head_ratio, linewidth=2, zorder=10)
    ax.text(*(v * text_pad), label, color=color, fontsize=10, ha='center',
            fontweight='bold', bbox=LABEL_BBOX, zorder=10)


def render_ball_sample(ax, r, n_particles, rng, half_extent=None):
    """Draws one bounded-ball sample with its boundary sphere and radius arrow, titled with r.

    Returns nothing.

    `ax`: the 3D axes.
    `r`: the radius.
    `n_particles`, `rng`: passed to synth_ball.
    `half_extent`: the panel's fixed half-width; None uses 1.2 r.
    """
    points = synth_ball(r, n_particles, rng)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **POINT_KW)
    # Wireframe sphere at radius r (ellipsoid_surface with equal sigmas).
    X, Y, Z = ellipsoid_surface((r, r, r), np.eye(3), scale=1.0)
    ax.plot_wireframe(X, Y, Z, color=WIREFRAME_COLOR, alpha=0.28, linewidth=0.6)
    _draw_radius_arrow(ax, r, '$r$', text_pad=2.0)
    if half_extent is None:
        half_extent = r * 1.2
    set_equal_aspect(ax, points, half_extent=half_extent)
    ax.view_init(**VIEW_INIT)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f'$r = {r:.2f}$', fontsize=10)


def render_gaussian_ball_sample(ax, sigma, n_particles, rng, half_extent=None):
    """Draws one Gaussian-ball sample with its 2-sigma sphere and radius arrow, titled with sigma.

    Returns nothing.

    `ax`: the 3D axes.
    `sigma`: the std on every axis.
    `n_particles`, `rng`: passed to synth_gaussian_ball.
    `half_extent`: the panel's fixed half-width; None uses 1.4 x the 2-sigma radius.
    """
    points = synth_gaussian_ball(sigma, n_particles, rng)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **POINT_KW)
    # Wireframe sphere at 2-sigma, comparable to the ellipsoid panels.
    X, Y, Z = ellipsoid_surface((sigma, sigma, sigma), np.eye(3), scale=ELLIPSOID_K)
    ax.plot_wireframe(X, Y, Z, color=WIREFRAME_COLOR, alpha=0.28, linewidth=0.6)
    _draw_radius_arrow(ax, sigma * ELLIPSOID_K, r'$2\sigma$', text_pad=2.0)
    if half_extent is None:
        half_extent = sigma * ELLIPSOID_K * 1.4
    set_equal_aspect(ax, points, half_extent=half_extent)
    ax.view_init(**VIEW_INIT)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(rf'$\sigma = {sigma:.2f}$', fontsize=10)


def render_hard_ellipsoid_sample(ax, semi_axes, n_particles, rng, rotated=False, half_extent=None):
    """Draws one bounded-ellipsoid sample with its boundary wireframe and semi-axes, titled with the target a and the shape values b and c.

    The cloud is genuinely 3D and anisotropic, but only a (dim 0, the longest semi-axis) is the bounded_ellipsoid_a target, so the b and c axes are muted.

    Returns nothing.

    `ax`: the 3D axes.
    `semi_axes`: (a, b, c).
    `n_particles`, `rng`, `rotated`: passed to synth_hard_ellipsoid.
    `half_extent`: the panel's fixed half-width; None uses 1.2 x the longest semi-axis.
    """
    points, R = synth_hard_ellipsoid(semi_axes, n_particles, rng, rotated=rotated)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **POINT_KW)
    # Wireframe at the boundary itself (scale=1.0), since the points fill the ellipsoid uniformly up to it.
    X, Y, Z = ellipsoid_surface(semi_axes, R, scale=1.0)
    ax.plot_wireframe(X, Y, Z, color=WIREFRAME_COLOR, alpha=0.28, linewidth=0.6)
    draw_principal_axes(ax, semi_axes, R,
                        names=(r'$a$', r'$b$', r'$c$'),
                        scale=1.0, with_values=False,
                        colors=(ACCENT, MUTED_AXIS_COLOR, MUTED_AXIS_COLOR),
                        label_pad=(1.5, 2.0, 2.5))
    if half_extent is None:
        half_extent = max(semi_axes) * 1.2
    set_equal_aspect(ax, points, half_extent=half_extent)
    ax.view_init(**VIEW_INIT)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(
        rf'target: $a = {semi_axes[0]:.2f}$'
        rf'  (shape also has $b,c = {semi_axes[1]:.2f},\ {semi_axes[2]:.2f}$)',
        fontsize=9,
    )


def render_box_sample(ax, semi_edges, n_particles, rng, rotated=False, half_extent=None):
    """Draws one box sample with its wireframe edges and semi-edges, titled with the target a and the shape values b and c.

    The cloud is genuinely 3D and anisotropic, but only a (dim 0, the longest semi-edge) is the box_a target, so the b and c axes are muted.

    Returns nothing.

    `ax`: the 3D axes.
    `semi_edges`: (a, b, c), the half edge lengths.
    `n_particles`, `rng`, `rotated`: passed to synth_box.
    `half_extent`: the panel's fixed half-width; None uses 1.2 x the longest semi-edge.
    """
    points, R = synth_box(semi_edges, n_particles, rng, rotated=rotated)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **POINT_KW)
    for p1, p2 in box_edges(semi_edges, R):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='dimgray', lw=0.8, alpha=0.7)
    draw_principal_axes(ax, semi_edges, R,
                        names=(r'$a$', r'$b$', r'$c$'),
                        scale=1.0, with_values=False,
                        colors=(ACCENT, MUTED_AXIS_COLOR, MUTED_AXIS_COLOR),
                        label_pad=(1.5, 2.0, 2.5))
    if half_extent is None:
        half_extent = max(semi_edges) * 1.2
    set_equal_aspect(ax, points, half_extent=half_extent)
    ax.view_init(**VIEW_INIT)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(
        rf'target: $a = {semi_edges[0]:.2f}$'
        rf'  (shape also has $b,c = {semi_edges[1]:.2f},\ {semi_edges[2]:.2f}$)',
        fontsize=9,
    )


def render_eigen_sample(ax, points, stream_idx):
    """Draws one real stream, centred, with its PCA 2-sigma ellipsoid and principal axes, titled with the target sqrt(lambda1) and the two shape values in kpc.

    PCA gives three eigenvalues, but only lambda1 (the largest, dim 0, since compute_pca_axes sorts descending) is the eigen proxy's target, so the lambda2 and lambda3 axes are muted.

    Returns nothing.

    `ax`: the 3D axes.
    `points`: the stream's (n, 3) raw coordinates, in kpc.
    `stream_idx`: the stream's index in the raw file, for the title.
    """
    sigmas, R, center = compute_pca_axes(points)
    pts_c = points - center
    ax.scatter(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], **POINT_KW)
    X, Y, Z = ellipsoid_surface(sigmas, R, scale=ELLIPSOID_K)
    ax.plot_wireframe(X, Y, Z, color=WIREFRAME_COLOR, alpha=0.28, linewidth=0.6)
    # Only the target axis is labelled: the lambda2 and lambda3 arrows stay, to show the shape, but without text.
    # label_pad is a multiple of the arrow vector, so on those two short arrows both labels landed near the origin and overlapped.
    # Their values are in the panel title.
    draw_principal_axes(
        ax, sigmas, R,
        names=(r'$2\sqrt{\lambda_1}$', '', ''),
        scale=ELLIPSOID_K, with_values=False,
        colors=(ACCENT, MUTED_AXIS_COLOR, MUTED_AXIS_COLOR),
        label_pad=(1.5, 4.0, 3.0),
    )
    # Real streams are curved and non-Gaussian and can reach beyond the 2-sigma ellipsoid, so the box is fitted to the points to keep the whole stream visible.
    set_equal_aspect(ax, pts_c, padding=1.05)
    ax.view_init(**VIEW_INIT)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # Unlike the synthetic proxies, real streams carry a unit, kpc, as in src/visualization.py.
    # Two lines, because one line with the stream id, the target and both shape values is wider than the 4-inch panel and collides with the neighbouring title.
    ax.set_title(
        rf'Stream {stream_idx}' '\n'
        rf'target: $\sqrt{{\lambda_1}} = {sigmas[0]:.2f}$ kpc'
        rf'  (shape also has $\sqrt{{\lambda_2}},\sqrt{{\lambda_3}} = {sigmas[1]:.2f},\ {sigmas[2]:.2f}$ kpc)',
        fontsize=9,
    )


def render_target_stream(ax, points):
    """Draws one real stream, centred, as the downstream task sees it: the point cloud alone, in the same box as render_eigen_sample.

    Its target, the progenitor mass, is not a length, so there is no arrow to draw; the caller labels it in text.

    Returns nothing.

    `ax`: the 3D axes.
    `points`: the stream's (n, 3) raw coordinates, in kpc.
    """
    pts_c = points - points.mean(axis=0)
    ax.scatter(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], **POINT_KW)
    set_equal_aspect(ax, pts_c, padding=1.05)
    ax.view_init(**VIEW_INIT)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def make_gallery(proxy_type, num_samples, output_path,
                 input_raw=None, num_particles=4000, seed=0, rotated=False):
    """Draws one proxy's gallery figure, num_samples panels side by side.

    The synthetic panels share one bounding box, so their sizes compare; each eigen panel is fitted to its own stream.

    Returns nothing; writes output_path and a PDF of the same name next to it.

    `proxy_type`: one of ALL_PROXY_TYPES except 'combined'.
    `num_samples`: the number of panels; the anisotropic proxies have four curated shapes, so they show at most four.
    `output_path`: the PNG path.
    `input_raw`: the raw HDF5 with Cartesian_data, required for 'eigen', which shows num_samples streams evenly spaced through it.
    `num_particles`: the points per synthetic sample.
    `seed`: the seed of the sampling rng.
    `rotated`: rotate the cube and the anisotropic shapes randomly, as training does.
    """
    rng = np.random.default_rng(seed)

    if proxy_type == 'cube':
        a_values = np.linspace(15.0, 45.0, num_samples)
        shared_half = float(a_values[-1]) / 2 * 1.15
        fig = plt.figure(figsize=(4.0 * num_samples, 4.5))
        for i, a in enumerate(a_values):
            ax = fig.add_subplot(1, num_samples, i + 1, projection='3d')
            render_cube_sample(ax, float(a), num_particles, rng,
                               rotated=rotated, half_extent=shared_half)

    elif proxy_type == 'gaussian_ellipsoid_sigmax':
        curated_all = [
            (25.0, 6.0, 1.5),
            (22.0, 12.0, 1.0),
            (10.0, 5.0, 2.0),
            (20.0, 1.0, 0.5),
        ]
        curated = curated_all[:num_samples]
        fig = plt.figure(figsize=(4.0 * len(curated), 4.5))
        shared_half = max(max(s) for s in curated) * ELLIPSOID_K * 1.2
        for i, sig in enumerate(curated):
            ax = fig.add_subplot(1, len(curated), i + 1, projection='3d')
            render_ellipsoid_sample(ax, sig, num_particles, rng,
                                    rotated=rotated, half_extent=shared_half)

    elif proxy_type == 'bounded_ball':
        r_values = np.linspace(8.0, 25.0, num_samples)
        shared_half = float(r_values[-1]) * 1.2
        fig = plt.figure(figsize=(4.0 * num_samples, 4.5))
        for i, r in enumerate(r_values):
            ax = fig.add_subplot(1, num_samples, i + 1, projection='3d')
            render_ball_sample(ax, float(r), num_particles, rng,
                               half_extent=shared_half)

    elif proxy_type == 'gaussian_ball':
        sigma_values = np.linspace(8.0, 25.0, num_samples)
        shared_half = float(sigma_values[-1]) * ELLIPSOID_K * 1.4
        fig = plt.figure(figsize=(4.0 * num_samples, 4.5))
        for i, s in enumerate(sigma_values):
            ax = fig.add_subplot(1, num_samples, i + 1, projection='3d')
            render_gaussian_ball_sample(ax, float(s), num_particles, rng,
                                        half_extent=shared_half)

    elif proxy_type == 'bounded_ellipsoid_a':
        curated_all = [
            (25.0, 6.0, 1.5),
            (22.0, 12.0, 1.0),
            (10.0, 5.0, 2.0),
            (20.0, 1.0, 0.5),
        ]
        curated = curated_all[:num_samples]
        fig = plt.figure(figsize=(4.0 * len(curated), 4.5))
        shared_half = max(max(s) for s in curated) * 1.2
        for i, sig in enumerate(curated):
            ax = fig.add_subplot(1, len(curated), i + 1, projection='3d')
            render_hard_ellipsoid_sample(ax, sig, num_particles, rng,
                                         rotated=rotated, half_extent=shared_half)

    elif proxy_type == 'box_a':
        curated_all = [
            (25.0, 6.0, 1.5),
            (22.0, 12.0, 1.0),
            (10.0, 5.0, 2.0),
            (20.0, 1.0, 0.5),
        ]
        curated = curated_all[:num_samples]
        fig = plt.figure(figsize=(4.0 * len(curated), 4.5))
        shared_half = max(max(s) for s in curated) * 1.2
        for i, sig in enumerate(curated):
            ax = fig.add_subplot(1, len(curated), i + 1, projection='3d')
            render_box_sample(ax, sig, num_particles, rng,
                              rotated=rotated, half_extent=shared_half)

    elif proxy_type == 'eigen':
        if input_raw is None:
            raise ValueError("eigen mode requires --input_raw")
        with h5py.File(input_raw, 'r') as f:
            cart = f['Cartesian_data']
            n_streams = int(cart.shape[0])
            indices = np.linspace(0, n_streams - 1, num_samples).astype(int)
            fig = plt.figure(figsize=(4.0 * num_samples, 4.5))
            for i, idx in enumerate(indices):
                ax = fig.add_subplot(1, num_samples, i + 1, projection='3d')
                points = np.asarray(cart[int(idx), :, 0:3])
                render_eigen_sample(ax, points, int(idx))
    else:
        raise ValueError(f"unknown proxy_type: {proxy_type}")

    # eigen gets no suptitle: its panel titles already give every symbol with its value, and in the paper the figure has a caption.
    # Without the suptitle the panels can also start higher (top=0.94 instead of 0.86).
    if proxy_type != 'eigen':
        fig.suptitle(f"{_PROXY_DISPLAY_NAMES[proxy_type]}    "
                     f"target: {_TARGET_DESCRIPTIONS[proxy_type]}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.86 if proxy_type != 'eigen' else 0.94)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_output_path = os.path.splitext(output_path)[0] + '.pdf'
    plt.savefig(pdf_output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}, {pdf_output_path}')


# ---------- Combined 2×4 gallery ----------

def make_combined_gallery(output_path, input_raw, stream_idx=0, num_particles=4000, seed=0, rotated=True):
    """Draws the combined publication figure: the six synthetic proxies in 2 rows (isotropic, anisotropic) x 3 columns, and one real stream in a 4th column.

    Only the isotropic/anisotropic split is labelled, because it is the one with a clear performance difference.
    Each column pairs shape-analogous proxies, so the boundary type (sharp, smooth, soft) is still visible from column to column, just not annotated.
    The 4th column shows the same stream twice: with its PCA target, as the Real-Stream PCA pretraining sees it, and with its progenitor-mass target, as the downstream task sees it.
    Only the target differs between the two, which is exactly how the PCA pretraining data are made.

    Layout:
        row 0 (Isotropic / 1D target): Cube, Bounded Ball, Gaussian Ball, Real-Stream PCA
        row 1 (Anisotropic / 1D target): Box, Bounded Ellipsoid, Gaussian Ellipsoid, Target Stream

    The paper includes the figure at \textwidth (5.5 in), so with the 12 in canvas every font prints at about 0.46x its size, as in the main result figure.
    The titles and row labels match that figure's tick labels, and the in-panel labels its legend.

    Returns nothing; writes output_path and a PDF of the same name next to it.

    `output_path`: the PNG path.
    `input_raw`: the raw HDF5 with Cartesian_data and pid (the log10 progenitor mass), for the stream column.
    `stream_idx`: the stream's index in input_raw.
    `num_particles`: the points per synthetic sample.
    `seed`: the seed of the sampling rng.
    `rotated`: rotate the cube and the anisotropic shapes randomly, as training does.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(input_raw, 'r') as f:
        stream_points = np.asarray(f['Cartesian_data'][stream_idx, :, 0:3])
        stream_log_mass = float(f['pid'][stream_idx, 0])

    # One curated sample per proxy (chosen to fill ~50-85% of the unified panel).
    CUBE_A        = 10.0
    BOX_EDGES     = (10.0, 5.0, 2.0)
    BALL_R        = 5.0
    HELL_AXES     = (10.0, 5.0, 2.0)
    GBALL_SIGMA   = 5.0    # 2σ wireframe ≈ 10
    GELL_SIGMAS   = (10.0, 5.0, 2.0)  # 2σ: 20/10/4 — within training range
    HALF_EXTENT   = 22.0   # shared bounding-box half-width of every synthetic panel, in the proxies' unitless coordinates

    TITLE_FS = 19   # panel titles and row labels, as the main result figure's tick labels
    LABEL_FS = 16   # labels inside the panels, as the main result figure's legend
    # Enlarges each 3D box inside its axes, so the panels sit closer together; the data range of each box is unchanged.
    BOX_ZOOM = 1.18
    PANEL_NAMES = {
        'cube':                     'Cube',
        'box_a':                    'Box',
        'bounded_ball':             'Bounded Ball',
        'bounded_ellipsoid_a':      'Bounded Ellipsoid',
        'gaussian_ball':            'Gaussian Ball',
        'gaussian_ellipsoid_sigmax': 'Gaussian Ellipsoid',
        'stream_pca':               'Real-Stream PCA',
        'stream_target':            'Target Stream',
    }
    # All six proxies are 1-target: in the isotropic row the target sets the whole shape, in the anisotropic row it is one of three axes and the other two are muted.
    # The row labels describe the synthetic columns only; the stream column is set apart by a line.
    ROW_LABELS  = ['1D target\nIsotropic', '1D target\nAnisotropic']
    GRID = [('cube',  'bounded_ball',          'gaussian_ball',             'stream_pca'),
            ('box_a', 'bounded_ellipsoid_a',   'gaussian_ellipsoid_sigmax', 'stream_target')]

    fig = plt.figure(figsize=(12, 6.4))
    # A narrow empty column before the stream column holds the separating line, and room for labels that reach past the last synthetic panel.
    gs  = fig.add_gridspec(2, 5, width_ratios=(1, 1, 1, 0.12, 1), hspace=0.12, wspace=0.0,
                           left=0.12, right=0.995, top=0.93, bottom=0.01)

    axes = [[fig.add_subplot(gs[r, c], projection='3d')
             for c in (0, 1, 2, 4)] for r in range(2)]

    render_fns = {
        'cube':          lambda ax: render_cube_sample(
                             ax, CUBE_A, num_particles, rng,
                             rotated=rotated, half_extent=HALF_EXTENT),
        'box_a':         lambda ax: render_box_sample(
                             ax, BOX_EDGES, num_particles, rng,
                             rotated=rotated, half_extent=HALF_EXTENT),
        'bounded_ball':  lambda ax: render_ball_sample(
                             ax, BALL_R, num_particles, rng,
                             half_extent=HALF_EXTENT),
        'bounded_ellipsoid_a': lambda ax: render_hard_ellipsoid_sample(
                             ax, HELL_AXES, num_particles, rng,
                             rotated=rotated, half_extent=HALF_EXTENT),
        'gaussian_ball': lambda ax: render_gaussian_ball_sample(
                             ax, GBALL_SIGMA, num_particles, rng,
                             half_extent=HALF_EXTENT),
        'gaussian_ellipsoid_sigmax': lambda ax: render_ellipsoid_sample(
                             ax, GELL_SIGMAS, num_particles, rng,
                             rotated=rotated, half_extent=HALF_EXTENT),
        'stream_pca':    lambda ax: render_eigen_sample(ax, stream_points, stream_idx),
        'stream_target': lambda ax: render_target_stream(ax, stream_points),
    }
    for row, row_types in enumerate(GRID):
        for col, ptype in enumerate(row_types):
            ax = axes[row][col]
            render_fns[ptype](ax)
            # The render functions title each panel with its values, as the per-proxy galleries do; here the name replaces it.
            ax.set_title(PANEL_NAMES[ptype], fontsize=TITLE_FS, fontweight='bold', pad=0)
            # In every panel the target is the accent-colored label and the muted labels name the other axes.
            # At this font size the muted labels pile up on each other, so only the target is labelled, as render_eigen_sample already does; the muted arrows stay, for the shape.
            # The remaining labels are drawn at the per-proxy galleries' size, so they are raised here.
            for text in list(ax.texts):
                if text.get_color() == MUTED_AXIS_COLOR:
                    text.remove()
                else:
                    text.set_fontsize(LABEL_FS)
            ax.set_box_aspect((1, 1, 1), zoom=BOX_ZOOM)
            # A transparent background, so a label reaching past its panel is not hidden under the next one.
            ax.patch.set_visible(False)

    # The progenitor mass is not a length, so the target stream has no arrow; its target is written in the accent color instead, with its value, since the stream is real data with units.
    axes[1][3].text2D(0.5, 0.97, rf'$\log_{{10}}(M_\star / M_\odot) = {stream_log_mass:.2f}$',
                      transform=axes[1][3].transAxes, ha='center', va='top', fontsize=LABEL_FS, color=ACCENT)

    # --- Row labels (left margin) ---
    for row, label in enumerate(ROW_LABELS):
        ax_pos = axes[row][0].get_position()
        fig.text(ax_pos.x0 - 0.005, ax_pos.y0 + ax_pos.height / 2,
                 label, ha='right', va='center',
                 fontsize=TITLE_FS, fontweight='bold', multialignment='center')

    # --- Line setting the real-stream column apart from the synthetic ones ---
    x_sep = (axes[0][2].get_position().x1 + axes[0][3].get_position().x0) / 2
    fig.add_artist(plt.Line2D([x_sep, x_sep], [0.03, 0.97], color='0.6', linewidth=1.2))

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_output_path = os.path.splitext(output_path)[0] + '.pdf'
    plt.savefig(pdf_output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}, {pdf_output_path}')


# ---------- CLI ----------

def parse_args():
    """Parses the command line.

    Returns the argparse Namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--proxy_type', nargs='+', required=True, metavar='TYPE',
        help=(f'One or more proxy types, or "all". '
              f'Valid: {", ".join(ALL_PROXY_TYPES)}'),
    )
    ap.add_argument('--num_samples', type=int, default=2)
    ap.add_argument('--num_particles', type=int, default=4000)
    ap.add_argument('--output_dir', default='proxy_gallery')
    ap.add_argument('--input_raw', default=None,
                    help='Required for eigen. HDF5 with Cartesian_data.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rotated', action=argparse.BooleanOptionalAction, default=True,
                    help='Apply random 3D rotation to cube/gaussian_ellipsoid_sigmax/'
                         'bounded_ellipsoid_a/box_a (matches training data; default: rotated. '
                         'Pass --no-rotated for axis-aligned).')
    return ap.parse_args()


def main():
    """Draws the figure of every requested --proxy_type into --output_dir.

    Returns nothing; skips eigen when --input_raw is missing and exits on an unknown type.
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if 'all' in args.proxy_type:
        types = ALL_PROXY_TYPES
    else:
        invalid = [t for t in args.proxy_type if t not in ALL_PROXY_TYPES]
        if invalid:
            sys.exit(f"error: unknown proxy type(s): {invalid}. Valid: {ALL_PROXY_TYPES}")
        types = args.proxy_type

    for t in types:
        out = os.path.join(args.output_dir, f'gallery_{t}.png')
        if t in ('eigen', 'combined') and args.input_raw is None:
            print(f"[skip] {t}: --input_raw not provided")
            continue
        if t == 'combined':
            make_combined_gallery(
                output_path=out,
                input_raw=args.input_raw,
                num_particles=args.num_particles,
                seed=args.seed,
                rotated=args.rotated,
            )
            continue
        make_gallery(
            t,
            num_samples=args.num_samples,
            output_path=out,
            input_raw=args.input_raw,
            num_particles=args.num_particles,
            seed=args.seed,
            rotated=args.rotated,
        )


if __name__ == '__main__':
    main()
