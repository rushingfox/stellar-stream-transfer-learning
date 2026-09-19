"""Write ``hparams.json`` using the *actual* OmniLearned CLI signature.

Defaults are read by live introspection of ``omnilearned.cli.train`` (the
Typer command) rather than mirrored by hand, so they cannot drift out of
sync with the installed CLI. Note it must be ``omnilearned.cli.train``, NOT
``omnilearned.train.run``.
The two have diverged: cli.py has lr=5e-5, wd=0.0, warmup_epoch=0, while
train.run has lr=5e-4, wd=0.3, warmup_epoch=1. The CLI is what actually
runs when the shell scripts call ``omnilearned train ...``, so cli.py is
the ground truth for defaults.

Runner scripts pass:
* ``explicit`` : dict of ``{param_name: value}`` for params the srun command
  actually passes on the CLI. Everything else is treated as default.
* ``slurm`` : dict of slurm metadata to record.

For each param in OmniLearned's real signature the helper records:
* ``value``       — what will actually be in effect during training
* ``cli_default`` — True iff that value matches the CLI default
"""

import inspect
import json
from typing import Any, Mapping


def _true_cli_defaults() -> dict[str, Any]:
    """Snapshot OmniLearned CLI ``train`` command defaults at call time.

    Reads from ``omnilearned.cli.train`` (Typer command), not train.run.
    Typer wraps each default in an OptionInfo object; we unwrap via
    ``.default`` to get the actual scalar.

    Import is done inside the function so the module is importable in
    contexts where the ``omnilearned`` package isn't yet on ``sys.path``.
    """
    from omnilearned.cli import train as _cli_train

    result = {}
    for name, p in inspect.signature(_cli_train).parameters.items():
        if p.default is inspect.Parameter.empty:
            continue
        raw = p.default
        # Typer stores the actual default inside OptionInfo.default
        result[name] = raw.default if hasattr(raw, "default") else raw
    return result


def build_hparams(
    explicit: Mapping[str, Any],
    slurm: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the hparams dict that will be written to disk."""
    defaults = _true_cli_defaults()

    hp: dict[str, dict[str, Any]] = {}
    for key, default_val in defaults.items():
        val = explicit[key] if key in explicit else default_val
        hp[key] = {"value": val, "cli_default": val == default_val}

    # Any explicit key not in OmniLearned's signature (typo or forward-
    # compat placeholder) is recorded with cli_default False so it's
    # visible in the JSON rather than silently dropped.
    for key, val in explicit.items():
        if key not in hp:
            hp[key] = {"value": val, "cli_default": False}

    # Explicit params first, defaults after — nicer for human review.
    hp = dict(sorted(hp.items(), key=lambda kv: kv[1]["cli_default"]))

    out: dict[str, Any] = {"omnilearned_params": hp}
    if slurm is not None:
        out["slurm"] = dict(slurm)
    return out


def write_hparams(
    output_path: str,
    explicit: Mapping[str, Any],
    slurm: Mapping[str, Any] | None = None,
    echo: bool = True,
) -> dict[str, Any]:
    """Build the dict, write it to ``output_path``, optionally echo to stdout."""
    d = build_hparams(explicit, slurm=slurm)
    with open(output_path, "w") as f:
        json.dump(d, f, indent=2)
    if echo:
        print("===== hparams =====")
        print(json.dumps(d, indent=2))
        print("===================")
    return d
