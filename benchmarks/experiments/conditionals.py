"""Quick measurement: warmed ``@snekit`` calls with and without branches.

A short, interleaved measurement, not part of the pyperf suite. Results
are in ``conditionals.md`` next to this file. Run it with a release build
(``maturin develop --release``), from the repository root::

    python benchmarks/experiments/conditionals.py
    python benchmarks/experiments/conditionals.py --cases straight_20 locals

Each case times the plain function and the warmed ``@snekit`` wrapper on
the same arguments. Before anything is timed, every case is verified:

- the wrapper returns what the plain function returns, with the same type;
- the call takes the native path: one native fast call and no frame of the
  original function (``workloads.call_path``).

Compilation happens during that setup, outside every timed loop. Rounds
alternate between the two variants so that drift affects both alike. With
the defaults the whole run takes well under a minute.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import workloads  # noqa: E402
from speedysnek import _native, snekit  # noqa: E402


def calculate_score(wins, penalties):
    score = wins * 3 - penalties
    if score < 0:
        score = 0
    else:
        score += 5
    return score


def adjust(a, b):
    result = a
    if a < b:
        result += b
    return result * 2


def nested(a, b, c):
    r = 0
    if a < b:
        if b < c:
            r = c - a
        else:
            r = b * 2
    else:
        r = a - c
    if r == 0:
        r += 7
    return r


def locals_score(wins, draws, penalties):
    points = wins * 3 + draws
    deduction = penalties * 2
    result = points - deduction
    return result


# name -> (function, arguments). The ``straight_*`` and ``locals`` cases
# have no branches and compile on the main branch too.
CASES: dict[str, tuple[Callable[..., Any], tuple[int, ...]]] = {
    "if_else_then": (calculate_score, (1, 5)),
    "if_else_else": (calculate_score, (10, 3)),
    "if_join": (adjust, (1, 2)),
    "nested": (nested, (1, 2, 3)),
    "straight_20": (workloads.make_workload(20), workloads.ARGS),
    "locals": (locals_score, (10, 4, 3)),
}


def prepare(name: str) -> tuple[Callable[..., Any], Callable[..., Any], tuple]:
    """Return ``(plain, wrapped, args)`` with ``wrapped`` compiled and
    verified to run natively."""
    func, args = CASES[name]
    wrapped = snekit(func)
    workloads.warm_up(wrapped, *args)
    expected = func(*args)
    result = wrapped(*args)
    if type(result) is not type(expected) or result != expected:
        raise SystemExit(f"{name}: {result!r} != {expected!r}")
    path = workloads.call_path(wrapped, *args)
    if path != "native":
        raise SystemExit(f"{name}: expected the native path, got {path}")
    return func, wrapped, args


def sample(func: Callable[..., Any], args: tuple, calls: int) -> float:
    """Nanoseconds per call over ``calls`` calls, loop included."""
    loop = range(calls)
    start = time.perf_counter_ns()
    for _ in loop:
        func(*args)
    return (time.perf_counter_ns() - start) / calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--calls", type=int, default=500_000)
    options = parser.parse_args()

    if _native.DEBUG_BUILD:
        print("warning: debug build; the numbers are not representative")
    print(f"CPython {platform.python_version()}, {platform.machine()}")
    print(f"{options.rounds} rounds x {options.calls} calls per variant\n")
    print(f"{'case':<14} {'python ns':>10} {'snekit ns':>10} {'ratio':>7}")

    for name in options.cases:
        plain, wrapped, args = prepare(name)
        times: dict[str, list[float]] = {"python": [], "snekit": []}
        sample(plain, args, options.calls // 10)  # Warm the loop itself.
        sample(wrapped, args, options.calls // 10)
        for _ in range(options.rounds):
            times["python"].append(sample(plain, args, options.calls))
            times["snekit"].append(sample(wrapped, args, options.calls))
        python = statistics.median(times["python"])
        native = statistics.median(times["snekit"])
        # Every timed call must have been native.
        if workloads.call_path(wrapped, *args) != "native":
            raise SystemExit(f"{name}: left the native path while timing")
        print(f"{name:<14} {python:>10.1f} {native:>10.1f} {native / python:>7.2f}")


if __name__ == "__main__":
    main()
