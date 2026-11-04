"""Writes ``hparams.json`` from the *actual* OmniLearned CLI signature.

Defaults are read by live introspection of ``omnilearned.cli.train`` (the Typer command) rather than mirrored by hand, so they cannot drift out of sync with the installed CLI.
It must be ``omnilearned.cli.train``, NOT ``omnilearned.train.run``, because the two have diverged: cli.py has lr=5e-5, wd=0.0, warmup_epoch=0, while train.run has lr=5e-4, wd=0.3, warmup_epoch=1.
The CLI is what runs when the shell scripts call ``omnilearned train ...``, so cli.py is the ground truth for defaults.

Runner scripts pass:
* ``explicit``: ``{param_name: value}`` for the params the runner's ``omnilearned train`` command passes; everything else is treated as default.
* ``slurm``: the slurm metadata to record.

For each param in OmniLearned's real signature the helper records:
* ``value``: what is in effect during training.
* ``cli_default``: True iff that value matches the CLI default.
"""

import inspect
import json
from typing import Any, Mapping


def _true_cli_defaults() -> dict[str, Any]:
    """Snapshots the defaults of OmniLearned's CLI ``train`` command at call time.

    Returns {param_name: default} for every param that has a default, unwrapped from Typer's OptionInfo.

    The import is inside the function, so this module imports even where ``omnilearned`` is not yet on ``sys.path``.
    """
    from omnilearned.cli import train as _cli_train

    result = {}
    for name, p in inspect.signature(_cli_train).parameters.items():
        if p.default is inspect.Parameter.empty:
            continue
        raw = p.default
        # Typer stores the actual default in OptionInfo.default.
        result[name] = raw.default if hasattr(raw, "default") else raw
    return result


def build_hparams(
    explicit: Mapping[str, Any],
    slurm: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds the hparams dict that write_hparams writes to disk.

    Returns {"omnilearned_params": {name: {"value": ..., "cli_default": ...}}, "slurm": {...}}, the params that differ from the CLI default first; "slurm" only if given.

    `explicit`: {param_name: value} for the params the runner passes; the rest take the CLI default.
    `slurm`: the slurm metadata to record, or None.
    """
    defaults = _true_cli_defaults()

    hp: dict[str, dict[str, Any]] = {}
    for key, default_val in defaults.items():
        val = explicit[key] if key in explicit else default_val
        hp[key] = {"value": val, "cli_default": val == default_val}

    # An explicit key not in OmniLearned's signature (a typo or a placeholder for a future param) is recorded with cli_default False, so it shows in the JSON instead of being silently dropped.
    for key, val in explicit.items():
        if key not in hp:
            hp[key] = {"value": val, "cli_default": False}

    # Params that differ from the CLI default first, then the defaults, for human review; the sort is stable, so each group keeps its order.
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
    """Builds the hparams dict and writes it to ``output_path`` as JSON.

    Returns the dict; with echo, also prints it.

    `output_path`: the JSON file to write, hparams.json in the runners.
    `explicit`, `slurm`: passed to build_hparams.
    `echo`: also print the JSON to stdout.
    """
    d = build_hparams(explicit, slurm=slurm)
    with open(output_path, "w") as f:
        json.dump(d, f, indent=2)
    if echo:
        print("===== hparams =====")
        print(json.dumps(d, indent=2))
        print("===================")
    return d
