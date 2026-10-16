"""pyperf plumbing shared by ``bench_runtime.py`` and ``bench_compilation.py``.

Both scripts use pyperf's normal :class:`pyperf.Runner`, so they accept all
of pyperf's options (``-o``, ``--fast``, ``--rigorous``, ``-p``,
``--affinity``, ``--debug-single-value``, ...) plus ``--filter``.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pyperf

from speedysnek import _native
from speedysnek._decorator import _WARMUP_CALLS

REPOSITORY = Path(__file__).resolve().parent.parent


def _forward_filter(cmd: list[str], args) -> None:
    # pyperf re-runs this script in worker processes; they must select the
    # same benchmarks, in the same order, as the parent.
    if args.filter:
        cmd.extend(("--filter", args.filter))


def make_runner() -> tuple[pyperf.Runner, Callable[[str], bool]]:
    """Return a configured runner and a predicate selecting benchmark names."""
    runner = pyperf.Runner(add_cmdline_args=_forward_filter)
    runner.argparser.add_argument(
        "--filter",
        metavar="REGEX",
        help="only run benchmarks whose name matches this regular expression",
    )
    args = runner.parse_args()
    pattern = re.compile(args.filter) if args.filter else None
    runner.metadata.update(environment_metadata())
    if _native.DEBUG_BUILD and not args.worker:
        print(
            "WARNING: speedysnek._native is a debug build; its timings are not "
            "representative. Rebuild with `maturin develop --release`.",
            file=sys.stderr,
        )

    def selected(name: str) -> bool:
        return pattern is None or pattern.search(name) is not None

    return runner, selected


def environment_metadata() -> dict[str, str | int]:
    """Metadata that pyperf does not collect but the report needs."""
    metadata: dict[str, str | int] = {
        "speedysnek_build": "debug" if _native.DEBUG_BUILD else "release",
        "speedysnek_warmup_calls": _WARMUP_CALLS,
        "machine": platform.machine() or "unknown",
        "cpython_jit": _cpython_jit(),
    }
    for key, command in (
        ("rustc_version", ["rustc", "--version"]),
        (
            "git_commit",
            ["git", "-C", str(REPOSITORY), "describe", "--always", "--dirty"],
        ),
    ):
        output = _output(command)
        if output:
            metadata[key] = output
    return metadata


def _cpython_jit() -> str:
    jit = getattr(sys, "_jit", None)
    if jit is None:
        return "unavailable"
    return "enabled" if jit.is_enabled() else "disabled"


def _output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            env={**os.environ, "LC_ALL": "C"},
        )
    except OSError, subprocess.SubprocessError:
        return None
    return result.stdout.strip() or None
