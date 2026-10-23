"""Quick experiment: does parking stop calls that never fit from paying for
native attempts, without slowing down calls that do fit?

A short, interleaved measurement, not part of the pyperf suite. Results
are in ``miss_policy.md`` next to this file. Run it with a release build
(``maturin develop --release``), from the repository root::

    python benchmarks/experiments/miss_policy.py
    python benchmarks/experiments/miss_policy.py --baseline ../main/python

``--baseline DIR`` also runs processes that import ``speedysnek`` from
``DIR`` (a checkout of the base commit with its own release build copied
into ``DIR/speedysnek``), alternating with processes of the current build.

On a noisy machine, count instructions instead of timing: ``--calls CASE
N`` sets one case up and then makes ``N`` calls, so that the difference
between two runs under ``valgrind --tool=callgrind`` with different ``N``
is the cost of the calls alone (see ``miss_policy.md``).

Every case is a steady state: the wrapper is warmed up with the case's own
warmup calls, then called ``SETTLE`` more times with the timed calls, so
that a build with the miss policy has parked whatever it parks. Before
anything is timed, every process verifies each case: every result equals
the plain function's, with the same type, and over ``PROBE`` timed calls
the native attempts and the frames of the original function are counted
and checked against what that build must do (see ``EXPECTED``). The status
is checked again after timing.
"""

from __future__ import annotations

import argparse
import collections
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
from speedysnek._decorator import _WARMUP_CALLS  # noqa: E402

# Calls made after warmup, before verifying and timing: well past the
# miss limit and several re-probe intervals.
SETTLE = 1000
# Timed calls whose path is counted during verification.
PROBE = 650

HAS_POLICY = hasattr(_native, "MISS_LIMIT")


def sum_squares(n):
    total = 0
    for i in range(n):
        total += i * i
    return total


def _pair(params: str = "a, b") -> Callable[..., Any]:
    return workloads._pair(params)


def _method(decorate: Callable[[Any], Any]) -> Any:
    """An instance whose ``double`` method is ``decorate``d."""

    class Scaled:
        def double(self, x):
            return x * 2

    Scaled.double = decorate(Scaled.__dict__["double"])
    return Scaled()


class Case:
    """Timed calls, their warmup and the path each build must take.

    ``calls`` are ``(args, kwargs)`` pairs, made in turn; ``warm`` is the
    warmup call, made ``_WARMUP_CALLS + 1`` times. ``expected`` maps
    ``"main"`` and ``"policy"`` to ``(status, native attempts per call,
    original frames per call)``; a ``None`` count is not checked exactly
    (see ``check``).
    """

    def __init__(
        self,
        make: Callable[[], Any],
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]],
        warm: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
        main: tuple[str, float, float] = ("compiled", 1, 0),
        policy: tuple[str, float, float] | None = None,
        method: bool = False,
    ) -> None:
        self.make = make
        self.calls = calls
        self.warm = calls[0] if warm is None else warm
        self.expected = {"main": main, "policy": main if policy is None else policy}
        self.method = method


def _args(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return args, kwargs


INTS = _args(3, 4)
# A parked call re-probes once per ``REPROBE_INTERVAL + 1`` calls.
REPROBE = 1 / 65

# Break-even: the committed baseline's estimated crossover is about 29
# operations, so ``ops_030`` is the smallest straight-line function where a
# warmed call could pay off; ``ops_100`` is a medium one.
CASES: dict[str, Case] = {
    "add": Case(lambda: workloads._define("add", "a, b", "a + b"), [INTS]),
    "ops_030": Case(lambda: workloads.make_workload(30), [_args(*workloads.ARGS)]),
    "ops_100": Case(lambda: workloads.make_workload(100), [_args(*workloads.ARGS)]),
    "range_32": Case(lambda: sum_squares, [_args(32)]),
    "float_only": Case(
        _pair, [_args(3.5, 4.25)], main=("compiled", 1, 1), policy=("parked", 0, 1)
    ),
    "method": Case(
        lambda: None,  # See ``_method``.
        [_args(21)],
        main=("compiled", 1, 1),
        policy=("parked", 0, 1),
        method=True,
    ),
    "keyword": Case(
        _pair, [_args(a=3, b=4)], main=("compiled", 0, 1), policy=("parked", 0, 1)
    ),
    "default": Case(
        lambda: _pair("a, b=4"),
        [_args(3)],
        main=("compiled", 0, 1),
        policy=("parked", 0, 1),
    ),
    "overflow": Case(
        _pair,
        [_args(2**62, 4)],
        warm=INTS,
        main=("compiled", 1, 1),
        policy=("parked", REPROBE, 1),
    ),
    "alternating": Case(_pair, [INTS, _args(3.5, 4)], main=("compiled", 1, 0.5)),
    "disabled": Case(
        workloads._rejected_pair, [INTS], main=("disabled", 0, 1), policy=None
    ),
}


class SetupError(Exception):
    """A case would not measure what it claims to."""


def _callee(target: Any, case: Case) -> Any:
    """What a timed call calls: the target, or its bound ``double``."""
    return target.double if case.method else target


def _wrapper(target: Any, case: Case) -> Any:
    """The ``@snekit`` wrapper behind a decorated target."""
    return type(target).__dict__["double"] if case.method else target


def _call(target: Any, case: Case, i: int) -> Any:
    args, kwargs = case.calls[i % len(case.calls)]
    return _callee(target, case)(*args, **kwargs)


def _attempts(wrapper: Any) -> int:
    """The native attempts ``wrapper`` has made so far."""
    compiled = wrapper._jit_state.compiled
    return 0 if compiled is None else compiled.function.try_calls


def build(name: str) -> dict[str, Any]:
    """The plain and the decorated target of one case, settled and verified.

    For a method case, each target is an instance whose ``double`` method
    is called, so both sides pay for binding it.
    """
    case = CASES[name]
    if case.method:
        targets = {"python": _method(lambda f: f), "snekit": _method(snekit)}
    else:
        func = case.make()
        targets = {"python": func, "snekit": snekit(func)}
    args, kwargs = case.warm
    for _ in range(_WARMUP_CALLS + 1):
        _callee(targets["snekit"], case)(*args, **kwargs)
    for i in range(SETTLE):
        _call(targets["snekit"], case, i)
    verify(name, targets)
    return targets


def _status(name: str, target: Any) -> str:
    return _wrapper(target, CASES[name])._jit_state.status.value


def _expected(name: str) -> tuple[str, float, float]:
    return CASES[name].expected["policy" if HAS_POLICY else "main"]


def check(where: str, actual: float, expected: float) -> None:
    # A re-probing case makes about one attempt per interval; allow for
    # where the probe window starts.
    tolerance = 0.01 if expected not in (0, 0.5, 1) else 0
    if abs(actual - expected) > tolerance:
        raise SetupError(f"{where}: {actual:.3f} per call, expected {expected}")


def verify(name: str, targets: dict[str, Any]) -> None:
    case = CASES[name]
    snek = targets["snekit"]
    wrapper = _wrapper(snek, case)
    original = wrapper.__wrapped__.__code__
    status, attempts, python_frames = _expected(name)
    if _status(name, snek) != status:
        raise SetupError(f"{name}: status {_status(name, snek)}")
    for i in range(len(case.calls)):
        expected, actual = _call(targets["python"], case, i), _call(snek, case, i)
        if actual != expected or type(actual) is not type(expected):
            raise SetupError(f"{name}: result {actual!r}, expected {expected!r}")
    frames: collections.Counter = collections.Counter()

    def profiler(frame, event, arg):
        if event == "call":
            frames[frame.f_code] += 1

    before = _attempts(wrapper)
    sys.setprofile(profiler)
    try:
        for i in range(PROBE):
            _call(snek, case, i)
    finally:
        sys.setprofile(None)
    check(f"{name}: native attempts", (_attempts(wrapper) - before) / PROBE, attempts)
    check(f"{name}: original frames", frames[original] / PROBE, python_frames)
    if _status(name, snek) != status:
        raise SetupError(f"{name}: the status changed while verifying")


def time_call(statement: str, namespace: dict[str, Any], loops: int) -> float:
    """Nanoseconds per run of ``statement``, unrolled 10 times."""
    body = "\n".join(f"    {statement}" for _ in range(10))
    source = f"def _run(_n):\n  for _ in range(_n):\n{body}\n"
    scope = dict(namespace)
    exec(source, scope)
    run = scope["_run"]
    start = time.perf_counter_ns()
    run(loops)
    return (time.perf_counter_ns() - start) / (loops * 10)


def statement(target: Any, case: Case) -> tuple[str, dict[str, Any], int]:
    """The timed statement (every call of the case once), its namespace,
    and how many calls it makes."""
    namespace: dict[str, Any] = {"f": target}
    calls = []
    for n, (args, kwargs) in enumerate(case.calls):
        names = [f"x{n}_{i}" for i in range(len(args))]
        names += [f"{key}=k{n}_{key}" for key in kwargs]
        namespace.update((f"x{n}_{i}", value) for i, value in enumerate(args))
        namespace.update((f"k{n}_{key}", value) for key, value in kwargs.items())
        callee = "f.double" if case.method else "f"
        calls.append(f"{callee}({', '.join(names)})")
    return "; ".join(calls), namespace, len(calls)


def measure(names: list[str], rounds: int, loops: int) -> dict[str, float]:
    """Median ns per call of every ``case/variant`` cell, in this process."""
    grid = []
    for name in names:
        for variant, f in build(name).items():
            grid.append((f"{name}/{variant}", f, CASES[name]))
    samples: dict[str, list[float]] = {cell: [] for cell, _, _ in grid}
    statements = {cell: statement(f, case) for cell, f, case in grid}
    for cell, _, _ in grid:
        text, namespace, _ = statements[cell]
        time_call(text, namespace, max(1, loops // 4))  # warm the code
    for _ in range(rounds):
        for cell, _, _ in grid:
            text, namespace, per = statements[cell]
            samples[cell].append(time_call(text, namespace, loops) / per)
    for cell, target, _ in grid:
        name, variant = cell.split("/")
        if variant == "snekit" and _status(name, target) != _expected(name)[0]:
            raise SetupError(f"{name}: the state changed while it was timed")
    return {cell: statistics.median(values) for cell, values in samples.items()}


def calls(name: str, n: int) -> None:
    """Set one case up, then make ``n`` timed calls of its wrapper."""
    case = CASES[name]
    text, namespace, per = statement(build(name)["snekit"], case)
    scope = dict(namespace)
    exec(f"def _run(_n):\n  for _ in range(_n):\n    {text}\n", scope)
    scope["_run"](n // per)


def worker(options: argparse.Namespace) -> None:
    import speedysnek

    if _native.DEBUG_BUILD:
        sys.exit("debug build of speedysnek._native: rebuild with --release")
    medians = measure(options.cases, options.rounds, options.loops)
    print(
        json.dumps(
            {"path": speedysnek.__file__, "policy": HAS_POLICY, "medians": medians}
        )
    )


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
        nargs=2,
        metavar=("CASE", "N"),
        help="set CASE up, make N calls of its wrapper, time nothing",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.calls is not None:
        name, n = options.calls
        calls(name, int(n))
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
            if report["policy"] or not report["path"].startswith(
                os.path.abspath(options.baseline)
            ):
                sys.exit(f"the baseline imported the wrong build: {report['path']}")
            builds["baseline"].append(report["medians"])

    print(
        f"# {platform.python_implementation()} {platform.python_version()}, "
        f"{platform.machine()}, {options.processes} process(es) per build x "
        f"{options.rounds} interleaved rounds x {options.loops * 10} statements"
    )
    print()
    header = ["case", "python", "snekit"]
    if "baseline" in builds:
        header += ["baseline python", "baseline snekit", "snekit / baseline"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name in options.cases:
        current = {
            variant: [run[f"{name}/{variant}"] for run in builds["current"]]
            for variant in ("python", "snekit")
        }
        row = [name, summary(current["python"]), summary(current["snekit"])]
        if "baseline" in builds:
            base = {
                variant: [run[f"{name}/{variant}"] for run in builds["baseline"]]
                for variant in ("python", "snekit")
            }
            row += [summary(base["python"]), summary(base["snekit"])]
            ratio = statistics.median(current["snekit"]) / statistics.median(
                base["snekit"]
            )
            row.append(f"{ratio:.2f}")
        print("| " + " | ".join(row) + " |")
    print("\nMedian of process medians, ns per call (min–max over processes).")


if __name__ == "__main__":
    main()
