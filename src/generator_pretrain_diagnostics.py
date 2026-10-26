"""Shows whether a generator-mode (diffusion) pretraining run is well trained, the
role pred-vs-true plays for the regression pretrains: generate clouds from pure noise
and compare them with the run's real val clouds.

Generation follows omnilearned.diffusion.generate: the probability-flow ODE
dz/dt = v_hat(z, t) from t = 1 (noise ~ N(0, s^2), s the run's gen_noise_std) to
t = 0, 128 time points (127 midpoint steps), here unconditional and without its
particle-count mask.
Since the generator starts from noise alone, matching the real size distribution is
something it has to have learned. It writes, to <run_dir>/results/:

  generator_samples.png   10 generated clouds above 10 real val clouds drawn at random,
                          the two sets independent of each other and each sorted by size
                          (x-y projections in kpc, each panel scaled to its own cloud and
                          titled with its RMS radius; the shape statistic is in the csv only)
  generator_samples.csv   generated vs real: RMS radius, the shape statistic
                          r_99.5 / RMS radius (99.5th-percentile distance from the centre
                          over the RMS radius) and the centroid's distance from the origin,
                          each as median and 16/84th percentiles. Size and shape are taken
                          about each cloud's own centroid, so position is measured only by
                          the centroid distance. The shape statistic depends on neither size
                          nor orientation; for 4000 points: uniform cube 1.56, uniform ball
                          1.30, Gaussian 2.07

Supports unconditional generator runs on 3-feature data with one file per split (all our
proxies); refuses anything else. Uses the body + generator_head weights that finetuning
starts from (not the EMA copies).

Usage:
    python src/generator_pretrain_diagnostics.py \\
        --run_dir 3d_proxy_cube_10000/omnilearned_proxy_pretrain_generator/run_0
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint

from omnilearned.network import PET2
from omnilearned.utils import get_model_parameters

GENERATION_STEPS = 128   # as in omnilearned.diffusion.generate
N_SHOWN = 10             # generated, and real, clouds drawn in generator_samples.png
N_COL = 5                # panels per row


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=str, required=True,
                   help="Generator pretrain run dir holding best_model_.pt and hparams.json")
    p.add_argument("--n_generate", type=int, default=64,
                   help=f"Clouds to generate from noise (at least {N_SHOWN}, the number drawn)")
    p.add_argument("--n_samples", type=int, default=None,
                   help=f"Real val clouds to compare with (default: all; at least {N_SHOWN}, the number drawn)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_hparams(run_dir):
    with open(run_dir / "hparams.json") as f:
        return {k: v["value"] for k, v in json.load(f)["omnilearned_params"].items()}


def build_model(hp, ckpt_path, device):
    """PET2 exactly as train.run() builds it, with the checkpoint's body and generator head."""
    if hp["mode"] != "generator":
        raise ValueError(f"{ckpt_path.parent} is a mode={hp['mode']} run, not a generator pretrain")
    if hp["conditional"] or hp["use_pid"] or hp["use_add"]:
        raise ValueError("Only unconditional generation is supported: this run uses "
                         "conditioning, particle IDs or additional features")
    if hp["num_feat"] != 3:
        raise ValueError(f"num_feat={hp['num_feat']}: OmniLearned's noise for 4 features uses "
                         "jet-specific means and stds, which this script does not reproduce")
    model = PET2(
        input_dim=hp["num_feat"],
        use_int=hp["interaction"],
        local_int=hp["local_interaction"],
        int_type=hp["interaction_type"],
        conditional=hp["conditional"],
        cond_dim=hp["num_cond"],
        pid=hp["use_pid"],
        pid_dim=hp["pid_dim"],
        add_info=hp["use_add"],
        add_dim=hp["num_add"],
        mode=hp["mode"],
        num_classes=hp["num_classes"],
        num_gen_classes=hp["num_gen_classes"],
        mlp_drop=hp["mlp_drop"],
        attn_drop=hp["attn_drop"],
        feature_drop=hp["feature_drop"],
        num_coord=hp["num_coord"],
        K=hp["K"],
        **get_model_parameters(hp["model_size"]),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.body.load_state_dict(ckpt["body"])
    model.generator.load_state_dict(ckpt["generator_head"])
    return model.to(device).eval(), int(ckpt["epoch"])


def load_val_clouds(run_dir, hp, n_samples):
    """The run's own val split (its --path is relative to the run dir)."""
    val_dir = run_dir / hp["path"] / "streams" / "val"
    files = sorted(val_dir.glob("*.hdf5"))
    if len(files) != 1:
        raise ValueError(f"Expected one val file under {val_dir}, found {len(files)}")
    with h5py.File(files[0], "r") as f:
        x = f["data"][:n_samples] if n_samples else f["data"][:]
    return x.astype(np.float32)


@torch.no_grad()
def generate(model, n_clouds, n_points, n_feat, noise_std, batch, device, seed):
    """Clouds generated from pure noise: the probability-flow ODE dz/dt = v_hat(z, t)
    integrated from t = 1 to 0, as omnilearned.diffusion.generate does, without its
    conditioning and particle-count mask."""
    rng = torch.Generator().manual_seed(seed)
    t_grid = torch.linspace(1.0, 0.0, GENERATION_STEPS, device=device)
    out = []
    for i in range(0, n_clouds, batch):
        nb = min(batch, n_clouds - i)
        z1 = torch.randn((nb, n_points, n_feat), generator=rng).to(device) * noise_std

        def velocity(t, z):
            return model.generator(model.body(z, None, None, None, t * torch.ones(nb, device=device)), None)

        out.append(odeint(velocity, z1, t_grid, method="midpoint")[-1].cpu().numpy())
        print(f"  generated {i + nb}/{n_clouds} clouds")
    return np.concatenate(out)


def rms_radius(x):
    """(B, N, 3) -> (B,): root-mean-square distance of the points from their centroid."""
    return np.sqrt(((x - x.mean(axis=1, keepdims=True)) ** 2).sum(-1).mean(-1))


def shape_ratio(x):
    """(B, N, 3) -> (B,): 99.5th-percentile distance from the centre over the RMS radius."""
    r = np.linalg.norm(x - x.mean(axis=1, keepdims=True), axis=-1)
    return np.percentile(r, 99.5, axis=1) / np.sqrt((r ** 2).mean(-1))


def centroid_distance(x):
    """(B, N, 3) -> (B,): distance of each cloud's centroid from the origin."""
    return np.linalg.norm(x.mean(axis=1), axis=-1)


def describe(prefix, v):
    """Median and 16/84th percentiles of v, as csv columns named <prefix>_*."""
    return {f"{prefix}_median": np.median(v),
            f"{prefix}_p16": np.percentile(v, 16), f"{prefix}_p84": np.percentile(v, 84)}


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    hp = load_hparams(run_dir)
    noise_std = float(hp.get("gen_noise_std") or 1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, best_epoch = build_model(hp, run_dir / "best_model_.pt", device)
    x = load_val_clouds(run_dir, hp, args.n_samples)
    if len(x) < N_SHOWN:
        raise ValueError(f"{len(x)} real val clouds, but the figure draws {N_SHOWN}: raise --n_samples")
    n_gen = max(args.n_generate, N_SHOWN)
    print(f"{run_dir}: best epoch {best_epoch}, noise std {noise_std:.3f}, device {device}; "
          f"generating {n_gen} clouds ({GENERATION_STEPS}-step midpoint ODE), {len(x)} real val clouds")

    g = generate(model, n_gen, x.shape[1], x.shape[2], noise_std, args.batch, device, args.seed)
    rms_g, rms_x = rms_radius(g), rms_radius(x)
    shape_g, shape_x = shape_ratio(g), shape_ratio(x)
    cen_g, cen_x = centroid_distance(g), centroid_distance(x)

    out_dir = run_dir / "results"
    out_dir.mkdir(exist_ok=True)
    summary = pd.DataFrame([
        {"clouds": name, "n": len(r),
         **describe("rms_radius", r), **describe("shape", s), **describe("centroid_dist", c)}
        for name, r, s, c in (("generated", rms_g, shape_g, cen_g), ("real_val", rms_x, shape_x, cen_x))
    ])
    summary.to_csv(out_dir / "generator_samples.csv", index=False)
    for _, s in summary.iterrows():
        print(f"  {s.clouds:9s}: RMS radius median {s.rms_radius_median:.2f} kpc "
              f"(16-84%: {s.rms_radius_p16:.2f}-{s.rms_radius_p84:.2f}) | shape r_99.5/RMS median {s.shape_median:.3f} "
              f"(16-84%: {s.shape_p16:.3f}-{s.shape_p84:.3f}) | centroid distance median {s.centroid_dist_median:.2f} kpc "
              f"(16-84%: {s.centroid_dist_p16:.2f}-{s.centroid_dist_p84:.2f})")
    print("  shape reference: uniform cube 1.56, ball 1.30, Gaussian 2.07")

    # Two independent sets, not pairs: the first generated clouds (already random draws)
    # and real val clouds drawn at random, each sorted by size.
    real_idx = np.random.default_rng(args.seed).choice(len(x), N_SHOWN, replace=False)
    groups = ((f"generated samples (the first {N_SHOWN} of {n_gen})", g[:N_SHOWN], rms_g[:N_SHOWN], "C3"),
              (f"real val clouds picked at random ({N_SHOWN} of {len(x)}; not matched to the generated ones above)",
               x[real_idx], rms_x[real_idx], "k"))
    nrow = int(np.ceil(N_SHOWN / N_COL))
    fig = plt.figure(figsize=(3.0 * N_COL, 2 * 3.1 * nrow + 1.0), layout="constrained")
    for subfig, (title, clouds, rms, color) in zip(fig.subfigures(2, 1), groups):
        axes = subfig.subplots(nrow, N_COL, squeeze=False)
        for ax, k in zip(axes.flat, np.argsort(rms)):
            p = clouds[k] - clouds[k].mean(0)
            lim = 1.1 * np.abs(p[:, :2]).max()
            ax.scatter(p[:, 0], p[:, 1], s=0.5, alpha=0.4, color=color)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
            ax.tick_params(labelsize=7)
            ax.set_title(f"RMS radius {rms[k]:.1f} kpc", fontsize=8)
        for ax in axes.flat[N_SHOWN:]:
            ax.set_axis_off()
        subfig.suptitle(f"{title} [kpc]", fontsize=10, color=color)
    fig.suptitle(f"{run_dir.parent.parent.name} generator pretrain (best epoch {best_epoch})\n"
                 "size = RMS distance of the points from the cloud's centre")
    fig.savefig(out_dir / "generator_samples.png", dpi=110)
    plt.close(fig)
    print(f"Saved generator_samples.png, generator_samples.csv -> {out_dir}")


if __name__ == "__main__":
    main()
