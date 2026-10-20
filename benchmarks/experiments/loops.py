"""Quick measurement: warmed ``@snekit`` calls of functions with a loop.

A short, interleaved measurement, not part of the pyperf suite. Results
are in ``loops.md`` next to this file. Run it with a release build
(``maturin develop --release``), from the repository root::

    python benchmarks/experiments/loops.py
    python benchmarks/experiments/loops.py --cases straight_20 --no-budget

Each case times the plain function and the warmed ``@snekit`` wrapper on
the same arguments. Before anything is timed, every case is verified:

- the wrapper returns what the plain function returns, with the same type;
- the call takes the native path: one native fast call and no frame of the
  original function (``workloads.call_path``), so no timed call fell back
  to Python, whether for overflow or for the backedge budget;
- every loop case stays below the backedge budget.

The native path is checked again after timing. Compilation is measured
separately (frontend analysis and Cranelift compilation, median of
several), outside every timed loop, and so is one call that needs more
backedges than the budget allows: it runs natively until the budget is
used up, then again in Python. Rounds alternate between the two variants
so that drift affects both alike. With the defaults the whole run takes
well under a minute.
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
from speedysnek import _compiler, _native, snekit  # noqa: E402
from speedysnek._frontend import analyze  # noqa: E402


def sum_squares(n):
    total = 0
    i = 0
    while i < n:
        total += i * i
        i += 1
    return total


def conditional_sum(n, cutoff):
    total = 0
    i = 0
    while i < n:
        if i < cutoff:
            total += i
        else:
            total += i * 2
        i += 1
    return total


# name -> (function, arguments). ``straight_20`` has no loop and compiles
# on ``main`` too: it is the regression check.
CASES: dict[str, tuple[Callable[..., Any], tuple[int, ...]]] = {
    **{f"sum_squares_{n}": (sum_squares, (n,)) for n in (10, 100, 1000, 10_000)},
    **{
        f"conditional_sum_{n}": (conditional_sum, (n, n // 2))
        for n in (10, 100, 1000, 10_000)
    },
    "straight_20": (workloads.make_workload(20), workloads.ARGS),
}

# Iterations of each loop case; ``straight_20`` has none.
ITERATIONS = {
    name: args[0]
    for name, (func, args) in CASES.items()
    if func
    in (
        sum_squares,
        conditional_sum,
    )
}


def prepare(name: str) -> tuple[Callable[..., Any], Callable[..., Any], tuple]:
    """Return ``(plain, wrapped, args)`` with ``wrapped`` compiled and
    verified to run natively."""
    func, args = CASES[name]
    budget = getattr(_native, "DEFAULT_BACKEDGE_BUDGET", None)
    if name in ITERATIONS and not ITERATIONS[name] < budget:
        raise SystemExit(f"{name}: {ITERATIONS[name]} iterations exceed the budget")
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


def calls_for(func: Callable[..., Any], args: tuple, target_ms: float) -> int:
    """Calls of ``func`` that take about ``target_ms``, at least one."""
    calls = 1
    while True:
        elapsed = sample(func, args, calls) * calls
        if elapsed >= 2e6 or calls >= 1 << 30:
            return max(1, int(calls * target_ms * 1e6 / elapsed))
        calls *= 2


def compile_time_us(func: Callable[..., Any], repeats: int) -> tuple[float, float]:
    """Median microseconds for frontend analysis and for Cranelift
    compilation of ``func``."""
    analyses, compilations = [], []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        instructions = analyze(func).instructions
        middle = time.perf_counter_ns()
        _compiler.compile_i64(instructions, func.__code__.co_argcount)
        end = time.perf_counter_ns()
        analyses.append((middle - start) / 1e3)
        compilations.append((end - middle) / 1e3)
    return statistics.median(analyses), statistics.median(compilations)


def budget_exhaustion(repeats: int) -> None:
    """Time one call that needs one backedge more than the budget."""
    budget = _native.DEFAULT_BACKEDGE_BUDGET
    n = budget + 1
    _, wrapped, _ = prepare("sum_squares_10")
    if workloads.call_path(wrapped, n) != "guard_miss":
        raise SystemExit("an exhausted budget must fall back to Python")
    if wrapped(n) != sum_squares(n):
        raise SystemExit("the fallback result differs")
    native_only = _compiler.compile_i64(analyze(sum_squares).instructions, 1).fast_call
    times: dict[str, list[float]] = {"python": [], "snekit": [], "native": []}
    for _ in range(repeats):
        times["python"].append(sample(sum_squares, (n,), 1))
        times["snekit"].append(sample(wrapped, (n,), 1))
        times["native"].append(sample(native_only, ((n,),), 1))
    python, snekit_ns, native = (statistics.median(times[k]) for k in times)
    print(
        f"\nbudget exhaustion, sum_squares({n:,}) (median of {repeats}):\n"
        f"  python {python / 1e6:.1f} ms, @snekit {snekit_ns / 1e6:.1f} ms "
        f"(ratio {snekit_ns / python:.3f}), of which the native attempt "
        f"{native / 1e6:.2f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument(
        "--sample-ms", type=float, default=40.0, help="target duration of a sample"
    )
    parser.add_argument("--compile-repeats", type=int, default=15)
    parser.add_argument("--budget-repeats", type=int, default=9)
    parser.add_argument(
        "--no-budget", action="store_true", help="skip the exhaustion measurement"
    )
    options = parser.parse_args()

    started = time.perf_counter()
    if _native.DEBUG_BUILD:
        print("warning: debug build; the numbers are not representative")
    print(f"CPython {platform.python_version()}, {platform.machine()}")
    print(f"{options.rounds} rounds of ~{options.sample_ms:g} ms per variant\n")

    print("compilation (median us):")
    print(f"  {'function':<16} {'analyze':>9} {'compile':>9}")
    seen = set()
    for name in options.cases:
        func = CASES[name][0]
        if func in seen:
            continue
        seen.add(func)
        analysis, compilation = compile_time_us(func, options.compile_repeats)
        print(f"  {func.__name__:<16} {analysis:>9.1f} {compilation:>9.1f}")

    print(f"\n{'case':<22} {'python ns':>11} {'snekit ns':>11} {'ratio':>7}")
    for name in options.cases:
        plain, wrapped, args = prepare(name)
        calls = calls_for(plain, args, options.sample_ms)
        times: dict[str, list[float]] = {"python": [], "snekit": []}
        sample(plain, args, max(1, calls // 10))  # Warm the loop itself.
        sample(wrapped, args, max(1, calls // 10))
        for _ in range(options.rounds):
            times["python"].append(sample(plain, args, calls))
            times["snekit"].append(sample(wrapped, args, calls))
        python = statistics.median(times["python"])
        native = statistics.median(times["snekit"])
        # Every timed call must have been native.
        if workloads.call_path(wrapped, *args) != "native":
            raise SystemExit(f"{name}: left the native path while timing")
        print(f"{name:<22} {python:>11.1f} {native:>11.1f} {native / python:>7.3f}")

    if not options.no_budget:
        budget_exhaustion(options.budget_repeats)
    print(f"\ntotal {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
