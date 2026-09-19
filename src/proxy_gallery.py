"""Proxy gallery: visualise the TARGET of each proxy task on a fresh sample.

Each proxy gets its own PNG so that no false correspondence is implied
between the proxies (e.g. cube has no physical analogue in real streams).

gaussian_ellipsoid_sigmax, bounded_ellipsoid_a and box_a are 1-target proxies
(only the first/largest axis is the training target). The point clouds shown
are still genuinely 3-D
anisotropic shapes (that's the input geometry); only the TITLE/target and axis
emphasis reflect the fact that just one dimension is the actual regression
target now (the other two axes are drawn muted, for shape context only).

Output files (under --output_dir):
    gallery_cube.png                    proxy_cube                       - target = $a$
    gallery_gaussian_ellipsoid_sigmax.png proxy_gaussian_ellipsoid_sigmax - target = $\sigma_x$
    gallery_bounded_ball.png            proxy_bounded_ball               - target = $r$
    gallery_gaussian_ball.png           proxy_gaussian_ball              - target = $\sigma$
    gallery_bounded_ellipsoid_a.png     proxy_bounded_ellipsoid_a        - target = $a$
    gallery_box_a.png                   proxy_box_a                      - target = $a$
    gallery_eigen.png                   proxy_stream_eigenvalues_lambda1 - target = $\sqrt{\lambda_1}$
"""
import argparse
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# Paper style: LaTeX-rendered text, sans-serif Helvetica (matches
# build_results_table.py). Requires a working LaTeX on PATH (e.g.
# `module load texlive/2024` loaded AFTER activating the conda env, since
# conda activation otherwise takes PATH priority).
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
})

# Display names shown in figure titles and evaluation plots.
_PROXY_DISPLAY_NAMES = {
    'cube':                     'proxy_cube',
    'bounded_ball':             'proxy_bounded_ball',
    'gaussian_ball':            'proxy_gaussian_ball',
    'gaussian_ellipsoid_sigmax': 'proxy_gaussian_ellipsoid_sigmax',
    'bounded_ellipsoid_a':      'proxy_bounded_ellipsoid_a',
    'box_a':                    'proxy_box_a',
    'eigen':                    'proxy_stream_eigenvalues_lambda1',
}

# LaTeX target description shown in the figure suptitle. Note: for the three
# 1-target proxies this is the ACTUAL training target (one scalar), not the
# full 3-D shape parameters shown in the point-cloud geometry.
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
POINT_KW = dict(s=1, c='steelblue', alpha=0.2, edgecolors='none', zorder=1)
ACCENT = 'crimson'
WIREFRAME_COLOR = '#555555'  # neutral dark gray, won't clash with RGB axis arrows
# Per-axis colors for principal-axis arrows (matches the conventional RGB xyz)
AXIS_COLORS = ('#e03030', '#d68020', '#9467bd')  # red, amber, purple (colorblind-friendly)
# For the three 1-target proxies (gaussian_ellipsoid_sigmax, bounded_ellipsoid_a,
# box_a): the target axis (dim 0) uses ACCENT; the other two are muted gray to
# show they're shape geometry only, not part of the training target.
MUTED_AXIS_COLOR = '#999999'
LABEL_BBOX = dict(facecolor='white', alpha=0.0, edgecolor='none', pad=1.2)
# Draw ellipsoids at 2-sigma so they stay visible even when one axis is tiny.
ELLIPSOID_K = 2.0


# ---------- random helpers ----------

def random_rotation_matrix(rng):
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
    points = rng.uniform(-a / 2, a / 2, size=(n_particles, 3))
    R = random_rotation_matrix(rng) if rotated else np.eye(3)
    return points @ R.T, R


def synth_ellipsoid(sigmas, n_particles, rng, rotated=False):
    sx, sy, sz = sigmas
    points = np.stack([
        rng.normal(0, sx, n_particles),
        rng.normal(0, sy, n_particles),
        rng.normal(0, sz, n_particles),
    ], axis=1)
    R = random_rotation_matrix(rng) if rotated else np.eye(3)
    return points @ R.T, R


def _sample_isotropic_directions(n_particles, rng):
    dirs = rng.normal(size=(n_particles, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def synth_ball(r, n_particles, rng):
    dirs = _sample_isotropic_directions(n_particles, rng)
    radii = r * rng.uniform(0.0, 1.0, size=n_particles) ** (1.0 / 3.0)
    return dirs * radii[:, None]


def synth_gaussian_ball(sigma, n_particles, rng):
    return rng.normal(0.0, sigma, size=(n_particles, 3))


def synth_hard_ellipsoid(semi_axes, n_particles, rng, rotated=False):
    a, b, c = semi_axes
    dirs = _sample_isotropic_directions(n_particles, rng)
    radii = rng.uniform(0.0, 1.0, size=n_particles) ** (1.0 / 3.0)
    inside_unit_ball = dirs * radii[:, None]
    points = inside_unit_ball * np.array([a, b, c])
    R = random_rotation_matrix(rng) if rotated else np.eye(3)
    return points @ R.T, R


def synth_box(semi_edges, n_particles, rng, rotated=False):
    a, b, c = semi_edges
    points = np.stack([
        rng.uniform(-a, a, n_particles),
        rng.uniform(-b, b, n_particles),
        rng.uniform(-c, c, n_particles),
    ], axis=1)
    R = random_rotation_matrix(rng) if rotated else np.eye(3)
    return points @ R.T, R


def compute_pca_axes(points):
    """Return (sigmas sorted desc, rotation matrix, centroid)."""
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
    """Same as cube_edges but each axis has its own half-length."""
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
    """Double-headed arrow parallel to one cube edge, offset OUTSIDE the cube."""
    h = a / 2
    # Place along original +x direction, offset in original -z so it sits
    # under the cube and is not confused with the wireframe edges themselves.
    z_off = a * 0.06
    p_start = R @ np.array([-h, -h, -h - z_off])
    p_end   = R @ np.array([ h, -h, -h - z_off])
    mid = (p_start + p_end) / 2
    half = (p_end - p_start) / 2

    # Two single-headed quivers from the midpoint, opposite directions.
    ax.quiver(*mid, *half,
              color=color, arrow_length_ratio=0.18, linewidth=2)
    ax.quiver(*mid, *(-half),
              color=color, arrow_length_ratio=0.18, linewidth=2)

    text_off = R @ np.array([0, 0, -a * 0.16])
    ax.text(*(mid + text_off), 'a',
            color=color, fontsize=11, ha='center', fontweight='bold',
            bbox=LABEL_BBOX)


def draw_principal_axes(ax, sigmas, R, names, scale=1.0, with_values=False,
                        colors=None, label_pad=1.5):
    """Draw three half-axes from origin and annotate each tip.

    `names`     : tuple of 3 LaTeX strings used as the label.
    `scale`     : multiplier on the sigmas (matches the ellipsoid surface scale).
    `with_values`: if True, append "=value" to each label (using *unscaled* sigma).
    `colors`    : sequence of 3 colours; defaults to AXIS_COLORS (RGB).
    `label_pad` : push the text outward by this factor of the arrow length.
                  Can be a single float (applied to all axes) or a tuple of 3
                  floats for per-axis control.
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
        # Fixed absolute arrowhead length so all three axes look the same
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
    """Point cloud is a genuine 3-D anisotropic Gaussian ellipsoid, but only
    sigma_x (dim 0) is the gaussian_ellipsoid_sigmax proxy's 1-target —
    sigma_y/z are shape geometry only, so their axes are drawn muted."""
    points, R = synth_ellipsoid(sigmas, n_particles, rng, rotated=rotated)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **POINT_KW)
    X, Y, Z = ellipsoid_surface(sigmas, R, scale=ELLIPSOID_K)
    ax.plot_wireframe(X, Y, Z, color=WIREFRAME_COLOR, alpha=0.28, linewidth=0.6)
    # Ellipsoid axes are already explicit in the title; keep the in-plot labels
    # purely directional so they don't pile up at the origin. Only sigma_x is
    # accented (it's the actual training target); y/z are muted.
    draw_principal_axes(ax, sigmas, R,
                        names=(r'$2\sigma_x$', r'$2\sigma_y$', r'$2\sigma_z$'),
                        scale=ELLIPSOID_K, with_values=False,
                        colors=(ACCENT, MUTED_AXIS_COLOR, MUTED_AXIS_COLOR),
                        label_pad=(1.4, 3.5, 2.0))
    # Long Gaussian tails would otherwise blow up the bounding box and shrink
    # the 2-sigma wireframe to a dot. Tie the box to max(sigma)*ELLIPSOID_K so
    # the wireframe fills most of the panel regardless of axis ratio.
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
    """Quiver from origin along +axis_dir of given length, with annotated label.
    The shaft length equals `length`; the arrowhead extends beyond it."""
    v = axis_dir * length           # shaft endpoint = parameter value
    head_ratio = 0.12
    vq = v / (1 - head_ratio)      # total quiver so shaft = length
    ax.quiver(0, 0, 0, vq[0], vq[1], vq[2],
              color=color, arrow_length_ratio=head_ratio, linewidth=2, zorder=10)
    ax.text(*(v * text_pad), label, color=color, fontsize=10, ha='center',
            fontweight='bold', bbox=LABEL_BBOX, zorder=10)


def render_ball_sample(ax, r, n_particles, rng, half_extent=None):
    points = synth_ball(r, n_particles, rng)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **POINT_KW)
    # Wireframe sphere at radius r (use ellipsoid_surface with equal sigmas).
    X, Y, Z = ellipsoid_surface((r, r, r), np.eye(3), scale=1.0)
    ax.plot_wireframe(X, Y, Z, color=WIREFRAME_COLOR, alpha=0.28, linewidth=0.6)
    _draw_radius_arrow(ax, r, 'r', text_pad=2.0)
    if half_extent is None:
        half_extent = r * 1.2
    set_equal_aspect(ax, points, half_extent=half_extent)
    ax.view_init(**VIEW_INIT)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f'$r = {r:.2f}$', fontsize=10)


def render_gaussian_ball_sample(ax, sigma, n_particles, rng, half_extent=None):
    points = synth_gaussian_ball(sigma, n_particles, rng)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **POINT_KW)
    # Wireframe sphere at 2-sigma so it stays comparable to the ellipsoid panels.
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
    """Point cloud is a genuine 3-D anisotropic hard-boundary ellipsoid, but
    only 'a' (dim 0, the largest semi-axis) is the bounded_ellipsoid_a proxy's
    1-target — b/c are shape geometry only, so their axes are muted."""
    points, R = synth_hard_ellipsoid(semi_axes, n_particles, rng, rotated=rotated)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **POINT_KW)
    # Wireframe at exactly the boundary (scale=1.0): the points are uniformly
    # distributed all the way out to that surface, so this is the *real* edge.
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
    """Point cloud is a genuine 3-D anisotropic box, but only 'a' (dim 0, the
    largest semi-edge-length) is the box_a proxy's 1-target — b/c are
    shape geometry only, so their axes are muted."""
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
    """Point cloud is a real stream; PCA gives 3 eigenvalues, but only
    lambda1 (largest, dim 0 — compute_pca_axes sorts descending) is the
    stream_eigenvalues_lambda1 proxy's 1-target. lambda2/3 are shape
    context only, so their axes are drawn muted."""
    sigmas, R, center = compute_pca_axes(points)
    pts_c = points - center
    ax.scatter(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], **POINT_KW)
    X, Y, Z = ellipsoid_surface(sigmas, R, scale=ELLIPSOID_K)
    ax.plot_wireframe(X, Y, Z, color=WIREFRAME_COLOR, alpha=0.28, linewidth=0.6)
    # Only the target axis is annotated. The lambda2/3 arrows stay (they show
    # the shape), but their text is dropped: label_pad is a MULTIPLE of the
    # arrow vector, and those two arrows are short, so both texts landed near
    # the origin -- which one you could actually read came down to how close
    # lambda2 and lambda3 happened to be for that particular stream. Their
    # values are already in the panel title.
    draw_principal_axes(
        ax, sigmas, R,
        names=(r'$2\sqrt{\lambda_1}$', '', ''),
        scale=ELLIPSOID_K, with_values=False,
        colors=(ACCENT, MUTED_AXIS_COLOR, MUTED_AXIS_COLOR),
        label_pad=(1.5, 4.0, 3.0),
    )
    # Real streams are curved/non-Gaussian; their spatial extent can exceed
    # the PCA 2-sigma ellipsoid. Fit the box to the actual point cloud so
    # the full stream stays visible.
    set_equal_aspect(ax, pts_c, padding=1.05)
    ax.view_init(**VIEW_INIT)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # Unlike the six synthetic proxies these are real streams, so the numbers
    # carry a unit (kpc) -- same convention as src/visualization.py.
    # Two lines: a single line with the stream id, the target and both shape
    # values is wider than the 4-inch panel, so neighbouring titles collide.
    ax.set_title(
        rf'Stream {stream_idx}' '\n'
        rf'target: $\sqrt{{\lambda_1}} = {sigmas[0]:.2f}$ kpc'
        rf'  (shape also has $\sqrt{{\lambda_2}},\sqrt{{\lambda_3}} = {sigmas[1]:.2f},\ {sigmas[2]:.2f}$ kpc)',
        fontsize=9,
    )


def make_gallery(proxy_type, num_samples, output_path,
                 input_raw=None, num_particles=4000, seed=0, rotated=False):
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

    # eigen gets no suptitle: every symbol in target_desc already appears in
    # both panel titles (with values), and in the paper this figure carries a
    # real \caption on top of that. Skipping it also frees the vertical space
    # that used to sit between the panel titles and the 3-D boxes.
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


# ---------- Combined 3×2 gallery ----------

def make_combined_gallery(output_path, num_particles=4000, seed=0, rotated=True):
    """3×2 publication figure: rows = isotropic vs anisotropic target.

    Only the isotropic/anisotropic split is emphasized (that's the axis with
    a clear performance difference); boundary type (sharp/smooth/soft) is no
    longer called out — it's still visually present column-to-column (each
    column pairs shape-analogous proxies), just not annotated or arrowed.

    Layout:
        row 0 (Isotropic / 1D target): Cube, Bounded Ball, Gaussian Ball
        row 1 (Anisotropic / 1D target): Box, Bounded Ellipsoid, Gaussian Ellipsoid
    """
    rng = np.random.default_rng(seed)

    # One curated sample per proxy (chosen to fill ~50-85% of the unified panel).
    CUBE_A        = 10.0
    BOX_EDGES     = (10.0, 5.0, 2.0)
    BALL_R        = 5.0
    HELL_AXES     = (10.0, 5.0, 2.0)
    GBALL_SIGMA   = 5.0    # 2σ wireframe ≈ 10
    GELL_SIGMAS   = (10.0, 5.0, 2.0)  # 2σ: 20/10/4 — within training range
    HALF_EXTENT   = 22.0   # unified bounding box half-width [kpc]

    ARROW_COLOR = '#555555'
    PANEL_NAMES = {
        'cube':                     'Cube',
        'box_a':                    'Box',
        'bounded_ball':             'Bounded Ball',
        'bounded_ellipsoid_a':      'Bounded Ellipsoid',
        'gaussian_ball':            'Gaussian Ball',
        'gaussian_ellipsoid_sigmax': 'Gaussian Ellipsoid',
    }
    # All six proxies shown here are 1-target; the row distinction is isotropic
    # (target IS the whole shape) vs anisotropic (target is one of 3 axes,
    # the other two are shape geometry only, shown muted in each panel).
    ROW_LABELS  = ['1D target\nIsotropic', '1D target\nAnisotropic']
    GRID = [('cube',  'bounded_ball',          'gaussian_ball'),
            ('box_a', 'bounded_ellipsoid_a',   'gaussian_ellipsoid_sigmax')]

    fig = plt.figure(figsize=(14, 8))
    gs  = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.05,
                           left=0.14, right=0.97, top=0.92, bottom=0.05)

    axes = [[fig.add_subplot(gs[r, c], projection='3d')
             for c in range(3)] for r in range(2)]

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
    }

    for row, row_types in enumerate(GRID):
        for col, ptype in enumerate(row_types):
            ax = axes[row][col]
            render_fns[ptype](ax)
            ax.set_title(PANEL_NAMES[ptype], fontsize=13, fontweight='bold', pad=4)

    # --- Row labels (left margin) ---
    for row, label in enumerate(ROW_LABELS):
        ax_pos = axes[row][0].get_position()
        fig.text(0.13, ax_pos.y0 + ax_pos.height / 2,
                 label, ha='right', va='center',
                 fontsize=12, fontweight='bold', multialignment='center')

    # --- Vertical arrow: Increasing Anisotropy (left) ---
    ax_varrow = fig.add_axes([0.03, 0.05, 0.015, 0.82])
    ax_varrow.set_xlim(0, 1); ax_varrow.set_ylim(0, 1); ax_varrow.axis('off')
    ax_varrow.annotate('', xy=(0.5, 0.0), xytext=(0.5, 1.0),
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                       lw=1.8, mutation_scale=14))
    fig.text(0.015, 0.46, 'Increasing Anisotropy',
             ha='center', va='center', fontsize=11, rotation=90,
             color=ARROW_COLOR, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_output_path = os.path.splitext(output_path)[0] + '.pdf'
    plt.savefig(pdf_output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}, {pdf_output_path}')


# ---------- CLI ----------

def parse_args():
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
        if t == 'combined':
            make_combined_gallery(
                output_path=out,
                num_particles=args.num_particles,
                seed=args.seed,
                rotated=args.rotated,
            )
            continue
        if t == 'eigen' and args.input_raw is None:
            print(f"[skip] {t}: --input_raw not provided")
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
