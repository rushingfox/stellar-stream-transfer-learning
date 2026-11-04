"""Prints the parameter count of every model family from the trained stream checkpoints, as backbone / head / total.

Terminal only; writes nothing.

    python src/count_params.py
    python src/count_params.py --inspect PATH [PATH]

Every family is counted on all of its run_0 checkpoints across every size in DATASET_SIZES, with a warning if they do not all agree.
Backbone is DeepSets' ``encoder`` / OmniLearned's ``body``; head is DeepSets' ``regressor`` / OmniLearned's ``classifier_head``.
OmniLearned's ``ema_body`` (a running average of ``body``, not extra model weights) and optimizer state are left out.

--inspect prints every component of one or two checkpoint files instead.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from exp_paths import get_dataset_dir  # noqa: E402

TARGET_COLUMNS = "3d_x_y_z"
DATASET_SIZES = (40, 100, 300, 1000)
# OmniLearned model sizes to report.
# Only small has final runs so far.
OMNILEARNED_SIZES = ("small",)


def load(path):
    """Loads a checkpoint file onto the CPU.

    Returns what torch.load gives: a state_dict for DeepSets, a dict of components (body, classifier_head, ...) for OmniLearned.

    `path`: the checkpoint file.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def n_params(state):
    """Counts the parameters of a state_dict.

    Returns the total number of elements of its tensors; non-tensor entries are skipped.

    `state`: a {name: tensor} dict.
    """
    return sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))


def deepsets_counts(path):
    """Counts the parameters of a DeepSets checkpoint.

    Returns (family label, (backbone, head)): the label names the hidden width (read from encoder.0.weight), the backbone is ``encoder`` and the head is everything else (``regressor``).

    `path`: a DeepSets best_model_epoch_*.pt state_dict.
    """
    sd = load(path)
    hidden = sd["encoder.0.weight"].shape[0]
    backbone = n_params({k: v for k, v in sd.items() if k.startswith("encoder.")})
    return f"DeepSets (hidden {hidden})", (backbone, n_params(sd) - backbone)


def omnilearned_counts(path):
    """Counts the parameters of an OmniLearned checkpoint.

    Returns (family label, (backbone, head)), backbone ``body`` and head ``classifier_head``; None if its model size, read from the run's hparams.json, is not in OMNILEARNED_SIZES.

    `path`: a best_model_.pt, with its run's hparams.json next to it.
    """
    hparams = json.loads((path.parent / "hparams.json").read_text())
    size = hparams["omnilearned_params"]["model_size"]["value"]
    if size not in OMNILEARNED_SIZES:
        return None
    ck = load(path)
    return f"OmniLearned-{size[0]}", (n_params(ck["body"]), n_params(ck["classifier_head"]))


def discover():
    """Finds the run_0 checkpoint of every trained stream model (deepsets* and omni* folders) at each of DATASET_SIZES.

    Yields (path, counter), with counter the function that counts that checkpoint (deepsets_counts or omnilearned_counts).
    """
    for size in DATASET_SIZES:
        dataset_dir = get_dataset_dir(REPO_ROOT, TARGET_COLUMNS, size)
        for path in sorted(dataset_dir.glob("deepsets*/run_0/checkpoints/best_model_epoch_*.pt")):
            yield path, deepsets_counts
        for path in sorted(dataset_dir.glob("omni*/run_0/best_model_.pt")):
            yield path, omnilearned_counts


def summarize():
    """Prints the backbone / head / total table, one row per family (DeepSets first, then by total) and per distinct count.

    Returns nothing; also prints a warning for every family whose checkpoints disagree, and the checkpoints it could not read.
    """
    families = defaultdict(lambda: defaultdict(list))  # label -> (backbone, head) -> paths
    unreadable = []
    for path, counter in discover():
        try:
            result = counter(path)
        except Exception as e:  # e.g. a checkpoint a running job is rewriting
            unreadable.append((path, e))
            continue
        if result is not None:
            label, counts = result
            families[label][counts].append(path)

    order = sorted(families, key=lambda label: (not label.startswith("DeepSets"),
                                                min(sum(c) for c in families[label])))
    print(f"{'Model':<24} {'Backbone':>12} {'Head':>12} {'Total':>12}")
    print("-" * 63)
    for label in order:
        for backbone, head in sorted(families[label], key=lambda c: -len(families[label][c])):
            print(f"{label:<24} {backbone:>12,} {head:>12,} {backbone + head:>12,}")

    # Every checkpoint of a family should give the same counts; say so only if not.
    for label in order:
        by_counts = families[label]
        if len(by_counts) > 1:
            n_ckpts = sum(len(p) for p in by_counts.values())
            print(f"\nWARNING: {label} checkpoints disagree ({len(by_counts)} different counts "
                  f"across {n_ckpts} checkpoints). Minority ones:")
            for paths in sorted(by_counts.values(), key=len)[:-1]:
                for p in paths:
                    print(f"  {p.relative_to(REPO_ROOT)}")

    for path, e in unreadable:
        print(f"\nSkipped unreadable {path.relative_to(REPO_ROOT)}: {type(e).__name__}: {e}")


def inspect(path):
    """Prints every top-level entry of one checkpoint: the parameter count of a tensor or of a dict of tensors, the length of any other dict, and the value of anything else.

    Returns nothing.

    `path`: the checkpoint file.
    """
    obj = load(path)
    print(f"=== {path} ===")
    if not isinstance(obj, dict):
        print(f"  (not a dict: {type(obj).__name__})")
        return
    for key, val in obj.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key + ':':30s} {val.numel():>15,} params")
        elif isinstance(val, dict) and val and all(isinstance(v, torch.Tensor) for v in val.values()):
            print(f"  {key + ':':30s} {n_params(val):>15,} params")
        elif isinstance(val, dict):
            print(f"  {key + ':':30s} (dict with {len(val)} entries)")
        else:
            print(f"  {key + ':':30s} {val}")
    print()


def main():
    """Prints the parameter table, or with --inspect the components of one or two checkpoints.

    Returns nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inspect", nargs="+", metavar="PATH", type=Path,
                        help="print every component of 1 or 2 checkpoint files")
    args = parser.parse_args()

    if args.inspect:
        if len(args.inspect) > 2:
            parser.error("--inspect takes at most 2 paths")
        for path in args.inspect:
            inspect(path)
        return
    summarize()


if __name__ == "__main__":
    main()
