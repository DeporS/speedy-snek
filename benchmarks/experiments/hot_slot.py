"""Quick experiment: does a Rust-owned hot slot cut warmed dispatch cost?

A short, interleaved measurement, not part of the pyperf suite. Results
are in ``hot_slot.md`` next to this file. Run it with a release build
(``maturin develop --release``), from the repository root::

    python benchmarks/experiments/hot_slot.py
    python benchmarks/experiments/hot_slot.py --baseline ../main/python

On a noisy machine, count instructions instead of timing: ``--calls CASE
VARIANT N`` sets one case up and then makes ``N`` calls of one variant,
so that the difference between two runs under ``valgrind
--tool=callgrind`` with different ``N`` is the cost of the calls alone
(see ``hot_slot.md``).

Variants
--------

``python``
    The undecorated function, with the same call statement.
``generic``
    A ``NativeWrapper`` made without a hot slot, so that every call takes
    the generic path: it reads the state's specialization, ``spec.code``,
    ``func.__code__`` and ``spec.function`` through Python attribute
    lookups. That is the dispatch ``snekit`` used before the hot slot; it
    stays as the fallback for unusual states and test doubles.
``snekit``
    ``speedysnek.snekit`` as built: with the hot slot, a warmed call
    reads no Python attribute.

By default every process times the three variants interleaved, round by
round. ``--baseline DIR`` also runs processes that import ``speedysnek``
from ``DIR`` (a checkout of the base commit with its own release build
copied into ``DIR/speedysnek``), alternating with processes of the
current build. There ``generic`` and ``snekit`` are the same path, so the
comparison of ``snekit`` across builds checks the in-process one.

Before anything is timed, every case is verified in every process: each
wrapper returns what the plain function returns, with the same type, and
takes the expected path (``workloads.call_path``: native, guard miss or
Python). ``snekit`` must have a live hot slot exactly when its function
compiled, if the build has one, and ``generic`` must have none. Every
wrapper's state is checked again after timing.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import workloads  # noqa: E402
from speedysnek import _native, snekit  # noqa: E402
from speedysnek._decorator import _JitState, _Status  # noqa: E402

VARIANTS = ("python", "generic", "snekit")


def generic(func: Callable[..., Any]) -> Callable[..., Any]:
    """``snekit`` without a hot slot: every call takes the generic path."""
    state = _JitState(func)
    wrapper = _native.NativeWrapper(func, state, _Status.COLD)
    functools.update_wrapper(wrapper, func)
    wrapper._jit_state = state
    return wrapper


def sum_squares(n):
    total = 0
    for i in range(n):
        total += i * i
    return total


class Case:
    """One timed call, its warmup arguments and the path it must take."""

    def __init__(
        self,
        make: Callable[[], Callable[..., Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        warm: tuple[Any, ...] | None = None,
        path: str = "native",
    ) -> None:
        self.make = make
        self.args = args
        self.kwargs = kwargs or {}
        self.warm = args if warm is None else warm
        self.path = path

    @property
    def status(self) -> _Status:
        return _Status.DISABLED if self.path == "disabled" else _Status.COMPILED


def _pair(params: str = "a, b") -> Callable[[], Any]:
    return lambda: workloads._pair(params)


# Break-even: the committed baseline's estimated crossover is about 29
# operations, so ``ops_030`` is the smallest straight-line function where a
# warmed call could pay off.
CASES: dict[str, Case] = {
    "add": Case(lambda: workloads._define("add", "a, b", "a + b"), (3, 4)),
    "ops_030": Case(lambda: workloads.make_workload(30), workloads.ARGS),
    "ops_100": Case(lambda: workloads.make_workload(100), workloads.ARGS),
    "range_32": Case(lambda: sum_squares, (32,)),
    "float_miss": Case(_pair(), (3.5, 4.25), warm=(3, 4), path="guard_miss"),
    "overflow_miss": Case(_pair(), (2**62, 4), warm=(3, 4), path="guard_miss"),
    "keyword": Case(_pair(), kwargs={"a": 3, "b": 4}, warm=(3, 4), path="python"),
    "default": Case(_pair("a, b=4"), (3,), warm=(3, 4), path="python"),
    "disabled": Case(workloads._rejected_pair, (3, 4), path="disabled"),
}


class SetupError(Exception):
    """A case would not measure what it claims to."""


def build(name: str) -> dict[str, Callable[..., Any]]:
    """The callable of every variant of one case, verified."""
    case = CASES[name]
    func = case.make()
    callables = {"python": func}
    for variant, decorate in (("generic", generic), ("snekit", snekit)):
        wrapper = decorate(func)
        workloads.warm_up(wrapper, *case.warm)
        verify(f"{variant}/{name}", variant, wrapper, func, case)
        callables[variant] = wrapper
    return callables


def verify(
    where: str, variant: str, wrapper: Any, func: Callable[..., Any], case: Case
) -> None:
    state = wrapper._jit_state
    if state.status is not case.status:
        raise SetupError(f"{where}: status {state.status}")
    expected = func(*case.args, **case.kwargs)
    actual = wrapper(*case.args, **case.kwargs)
    if actual != expected or type(actual) is not type(expected):
        raise SetupError(f"{where}: result {actual!r}, expected {expected!r}")
    path = workloads.call_path(wrapper, *case.args, **case.kwargs)
    if path != ("python" if case.path == "disabled" else case.path):
        raise SetupError(f"{where}: path {path!r}, expected {case.path!r}")
    # ``getattr``: a build without the hot slot has no ``_hot``.
    hot = getattr(state, "_hot", None)
    if variant == "generic" and hot is not None:
        raise SetupError(f"{where}: the generic variant has a hot slot")
    if hot is not None and hot.live != (case.status is _Status.COMPILED):
        raise SetupError(f"{where}: hot slot live is {hot.live}")


def time_call(statement: str, namespace: dict[str, Any], loops: int) -> float:
    """Nanoseconds per call of ``statement``, unrolled 10 times."""
    body = "\n".join(f"    {statement}" for _ in range(10))
    source = f"def _run(_n):\n  for _ in range(_n):\n{body}\n"
    scope = dict(namespace)
    exec(source, scope)
    run = scope["_run"]
    start = time.perf_counter_ns()
    run(loops)
    return (time.perf_counter_ns() - start) / (loops * 10)


def statement(f: Callable[..., Any], case: Case) -> tuple[str, dict[str, Any]]:
    names = [f"x{i}" for i in range(len(case.args))]
    names += [f"{key}=k_{key}" for key in case.kwargs]
    namespace: dict[str, Any] = {"f": f}
    namespace.update((f"x{i}", value) for i, value in enumerate(case.args))
    namespace.update((f"k_{key}", value) for key, value in case.kwargs.items())
    return f"f({', '.join(names)})", namespace


def measure(names: list[str], rounds: int, loops: int) -> dict[str, float]:
    """Median ns per call of every ``case/variant`` cell, in this process."""
    grid = []
    for name in names:
        for variant, f in build(name).items():
            grid.append((f"{name}/{variant}", f, CASES[name]))
    samples: dict[str, list[float]] = {cell: [] for cell, _, _ in grid}
    for _, f, case in grid:
        time_call(*statement(f, case), max(1, loops // 4))  # warm the code
    for _ in range(rounds):
        for cell, f, case in grid:
            samples[cell].append(time_call(*statement(f, case), loops))
    for cell, f, case in grid:
        if not cell.endswith("/python") and f._jit_state.status is not case.status:
            raise SetupError(f"{cell}: the state changed while it was timed")
    return {cell: statistics.median(values) for cell, values in samples.items()}


def calls(name: str, variant: str, n: int) -> None:
    """Set one case up, then make ``n`` calls of one variant."""
    f = build(name)[variant]
    call, namespace = statement(f, CASES[name])
    scope = dict(namespace)
    exec(f"def _run(_n):\n  for _ in range(_n):\n    {call}\n", scope)
    scope["_run"](n)


def worker(options: argparse.Namespace) -> None:
    import speedysnek

    if _native.DEBUG_BUILD:
        sys.exit("debug build of speedysnek._native: rebuild with --release")
    medians = measure(options.cases, options.rounds, options.loops)
    hot = hasattr(_native, "HotSlot")
    print(json.dumps({"path": speedysnek.__file__, "hot": hot, "medians": medians}))


def run_worker(options: argparse.Namespace, path: str | None) -> dict[str, Any]:
    command = [sys.executable, __file__, "--worker"]
    command += ["--rounds", str(options.rounds), "--loops", str(options.loops)]
    command += ["--cases", *options.cases]
    if options.cpu is not None:
        command = ["taskset", "-c", str(options.cpu), *command]
    env = dict(os.environ)
    if path is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            [path, *filter(None, [env.get("PYTHONPATH")])]
        )
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(result.stderr or result.stdout)
    return json.loads(result.stdout)


def summary(values: list[float]) -> str:
    return f"{statistics.median(values):.0f} ({min(values):.0f}–{max(values):.0f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--loops", type=int, default=20000)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES)
    parser.add_argument("--cpu", type=int, help="pin every process to this CPU")
    parser.add_argument(
        "--baseline", help="also time the speedysnek importable from this directory"
    )
    parser.add_argument(
        "--calls",
        nargs=3,
        metavar=("CASE", "VARIANT", "N"),
        help="set CASE up, make N calls of VARIANT, time nothing",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.calls is not None:
        name, variant, n = options.calls
        calls(name, variant, int(n))
        return
    if options.worker:
        worker(options)
        return

    builds: dict[str, list[dict[str, float]]] = {"current": []}
    if options.baseline is not None:
        builds["baseline"] = []
    for _ in range(options.processes):
        builds["current"].append(run_worker(options, None)["medians"])
        if options.baseline is not None:
            report = run_worker(options, os.path.abspath(options.baseline))
            if report["hot"] or not report["path"].startswith(
                os.path.abspath(options.baseline)
            ):
                sys.exit(f"the baseline imported the wrong build: {report['path']}")
            builds["baseline"].append(report["medians"])

    print(
        f"# {platform.python_implementation()} {platform.python_version()}, "
        f"{platform.machine()}, {options.processes} process(es) per build x "
        f"{options.rounds} interleaved rounds x {options.loops * 10} calls"
    )
    print()
    header = ["case", "python", "generic", "snekit", "snekit / generic"]
    if "baseline" in builds:
        header += ["baseline snekit", "snekit / baseline"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name in options.cases:
        cells = {
            variant: [run[f"{name}/{variant}"] for run in builds["current"]]
            for variant in VARIANTS
        }
        ratios = [s / g for s, g in zip(cells["snekit"], cells["generic"], strict=True)]
        row = [name, *(summary(cells[variant]) for variant in VARIANTS)]
        row.append(f"{min(ratios):.2f}–{max(ratios):.2f}")
        if "baseline" in builds:
            base = [run[f"{name}/snekit"] for run in builds["baseline"]]
            row.append(summary(base))
            across = statistics.median(cells["snekit"]) / statistics.median(base)
            row.append(f"{across:.2f}")
        print("| " + " | ".join(row) + " |")
    print(
        "\nMedian of process medians, ns per call (min–max over processes). "
        "snekit / generic is the per-process range."
    )


if __name__ == "__main__":
    main()
