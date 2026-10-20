"""Quick measurement: ``for`` loops over ``range(n)``.

A short, interleaved measurement, not part of the pyperf suite. Results
are in ``for_range.md`` next to this file. Run it with a release build
(``maturin develop --release``), from the repository root::

    python benchmarks/experiments/for_range.py
    python benchmarks/experiments/for_range.py --cases straight_20

Each loop size is timed three ways, round-robin: the plain ``for``
function, its warmed ``@snekit`` wrapper, and the warmed ``@snekit``
wrapper of the equivalent ``while`` function. ``straight_20`` (no loop,
no ``range``) times plain Python against ``@snekit``; it compiles on
``main`` too, so running ``--cases straight_20`` on both builds shows
whether the range guard adds work to functions that do not use ``range``.

Before anything is timed, every case is verified: each wrapper returns
what the plain function returns, takes the native path (one native fast
call, no frame of the original function: ``workloads.call_path``), and
stays below the backedge budget; the ``for`` wrappers have a valid range
guard and the others have none. The native path is checked again after
timing. Compilation (frontend analysis, the range guard and Cranelift) is
measured separately, outside every timed loop. With the defaults the whole
run takes well under a minute.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loops import calls_for, sample  # noqa: E402

import workloads  # noqa: E402
from speedysnek import _compiler, _native, snekit  # noqa: E402
from speedysnek._frontend import analyze  # noqa: E402


def sum_squares(n):
    total = 0
    for i in range(n):
        total += i * i
    return total


def conditional_sum(n, cutoff):
    total = 0
    for i in range(n):
        if i < cutoff:
            total += i
        else:
            total += i * 2
    return total


def sum_squares_while(n):
    total = 0
    i = 0
    while i < n:
        total += i * i
        i += 1
    return total


def conditional_sum_while(n, cutoff):
    total = 0
    i = 0
    while i < n:
        if i < cutoff:
            total += i
        else:
            total += i * 2
        i += 1
    return total


SIZES = (10, 100, 1000, 10_000)

# name -> (for function, equivalent while function or None, arguments).
CASES: dict[str, tuple[Callable[..., Any], Callable[..., Any] | None, tuple]] = {
    **{f"sum_squares_{n}": (sum_squares, sum_squares_while, (n,)) for n in SIZES},
    **{
        f"conditional_sum_{n}": (conditional_sum, conditional_sum_while, (n, n // 2))
        for n in SIZES
    },
    "straight_20": (workloads.make_workload(20), None, workloads.ARGS),
}

# Iterations of each loop case; ``straight_20`` has none.
ITERATIONS = {name: args[0] for name, (_, other, args) in CASES.items() if other}


def compiled(func: Callable[..., Any], args: tuple) -> Callable[..., Any]:
    """The warmed ``@snekit`` wrapper of ``func``, verified to run natively."""
    wrapped = snekit(func)
    workloads.warm_up(wrapped, *args)
    expected = func(*args)
    result = wrapped(*args)
    if type(result) is not type(expected) or result != expected:
        raise SystemExit(f"{func.__name__}: {result!r} != {expected!r}")
    path = workloads.call_path(wrapped, *args)
    if path != "native":
        raise SystemExit(f"{func.__name__}: expected the native path, got {path}")
    # ``getattr``: ``main`` has neither, and runs ``straight_20`` too.
    guard = getattr(wrapped._jit_state.specialization, "guard", None)
    uses_range = "range" in getattr(analyze(func), "assumed_builtins", ())
    if uses_range != (guard is not None) or (guard is not None and not guard.valid):
        raise SystemExit(f"{func.__name__}: unexpected range guard {guard!r}")
    return wrapped


def prepare(name: str) -> dict[str, tuple[Callable[..., Any], tuple]]:
    """The variants of one case, by label, each ``(callable, args)``."""
    func, other, args = CASES[name]
    if name in ITERATIONS and not ITERATIONS[name] < _native.DEFAULT_BACKEDGE_BUDGET:
        raise SystemExit(f"{name}: {ITERATIONS[name]} iterations exceed the budget")
    variants = {"python": (func, args), "snekit": (compiled(func, args), args)}
    if other is not None:
        if other(*args) != func(*args):
            raise SystemExit(f"{name}: the while loop computes something else")
        variants["while"] = (compiled(other, args), args)
    return variants


def compile_time_us(
    func: Callable[..., Any], repeats: int
) -> tuple[float, float | None, float]:
    """Median microseconds for frontend analysis, the range guard (``None``
    for a function without one) and Cranelift compilation of ``func``."""
    analyses, guards, compilations = [], [], []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        analysis = analyze(func)
        analyzed = time.perf_counter_ns()
        if getattr(analysis, "assumed_builtins", ()):
            _native.RangeGuard(func.__globals__, func.__builtins__)
            guards.append((time.perf_counter_ns() - analyzed) / 1e3)
        start_compile = time.perf_counter_ns()
        _compiler.compile_i64(analysis.instructions, func.__code__.co_argcount)
        end = time.perf_counter_ns()
        analyses.append((analyzed - start) / 1e3)
        compilations.append((end - start_compile) / 1e3)
    guard = statistics.median(guards) if guards else None
    return statistics.median(analyses), guard, statistics.median(compilations)


def lifecycle_ns(
    func: Callable[..., Any], args: tuple, repeats: int
) -> tuple[float, float, float]:
    """Median nanoseconds of a plain call, a warmup call of a fresh wrapper
    and the wrapper's compiling call (analysis, guard, compilation and the
    first native call). Calls are timed one at a time, timer included; each
    repeat times as many plain calls as there are warmup calls, right
    before them, so both see the same caches."""
    plain, warmup, compiling = [], [], []
    for _ in range(repeats):
        for _ in range(workloads.LIFECYCLE_CALLS - 1):
            start = time.perf_counter_ns()
            func(*args)
            plain.append(time.perf_counter_ns() - start)
        wrapped = snekit(func)
        for _ in range(workloads.LIFECYCLE_CALLS - 1):
            start = time.perf_counter_ns()
            wrapped(*args)
            warmup.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        wrapped(*args)
        compiling.append(time.perf_counter_ns() - start)
        if workloads.status_of(wrapped).value != "compiled":
            raise SystemExit(f"{func.__name__}: the compiling call did not compile")
    return tuple(statistics.median(times) for times in (plain, warmup, compiling))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument(
        "--sample-ms", type=float, default=40.0, help="target duration of a sample"
    )
    parser.add_argument("--compile-repeats", type=int, default=15)
    options = parser.parse_args()

    started = time.perf_counter()
    if _native.DEBUG_BUILD:
        print("warning: debug build; the numbers are not representative")
    print(f"CPython {platform.python_version()}, {platform.machine()}")
    print(f"{options.rounds} rounds of ~{options.sample_ms:g} ms per variant\n")

    print("compilation (median us):")
    print(f"  {'function':<22} {'analyze':>9} {'guard':>7} {'compile':>9}")
    seen = set()
    for name in options.cases:
        func, other, _ = CASES[name]
        for f in (func, other):
            if f is None or f in seen:
                continue
            seen.add(f)
            analysis, guard, compilation = compile_time_us(f, options.compile_repeats)
            label = "straight_20" if f is func and other is None else f.__name__
            shown = "-" if guard is None else f"{guard:.1f}"
            print(f"  {label:<22} {analysis:>9.1f} {shown:>7} {compilation:>9.1f}")

    print("\nlifecycle of a fresh wrapper (median):")
    print(f"  {'function':<22} {'plain ns':>9} {'warmup ns':>10} {'compiling us':>13}")
    seen = set()
    for name in options.cases:
        func, _, args = CASES[name]
        if func in seen:
            continue
        seen.add(func)
        plain, warmup, compiling = lifecycle_ns(func, args, options.compile_repeats)
        label = "straight_20" if name == "straight_20" else func.__name__
        print(f"  {label:<22} {plain:>9.0f} {warmup:>10.0f} {compiling / 1e3:>13.1f}")

    print(
        f"\n{'case':<22} {'python ns':>11} {'snekit ns':>11} {'ratio':>7}"
        f" {'while ns':>11} {'ratio':>7}"
    )
    for name in options.cases:
        variants = prepare(name)
        plain, args = variants["python"]
        calls = calls_for(plain, args, options.sample_ms)
        times: dict[str, list[float]] = {label: [] for label in variants}
        for f, f_args in variants.values():
            sample(f, f_args, max(1, calls // 10))  # Warm the loop itself.
        for _ in range(options.rounds):
            for label, (f, f_args) in variants.items():
                times[label].append(sample(f, f_args, calls))
        for label in ("snekit", "while"):
            if label in variants:
                f, f_args = variants[label]
                if workloads.call_path(f, *f_args) != "native":
                    raise SystemExit(f"{name}: {label} left the native path")
        medians = {label: statistics.median(t) for label, t in times.items()}
        python = medians["python"]
        line = (
            f"{name:<22} {python:>11.1f} {medians['snekit']:>11.1f}"
            f" {medians['snekit'] / python:>7.3f}"
        )
        if "while" in medians:
            line += f" {medians['while']:>11.1f} {medians['while'] / python:>7.3f}"
        print(line)

    print(f"\ntotal {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
