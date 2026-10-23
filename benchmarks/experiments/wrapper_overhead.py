"""Quick experiment: does a signature-specific wrapper make ``@snekit`` cheaper?

This is a short, interleaved A/B measurement, not part of the pyperf
suite. Its conclusion is in ``wrapper_overhead.md`` next to this file.
Run it with a release build, from the repository root::

    python benchmarks/experiments/wrapper_overhead.py
    taskset -c 3 python benchmarks/experiments/wrapper_overhead.py --rounds 15

Variants
--------

``production``
    The Python ``wrapper(*args, **kwargs)`` closure that ``snekit``
    returned when this experiment ran. ``snekit`` now returns a native
    wrapper (see ``native_wrapper.md``), so this is the verbatim copy kept
    in ``native_wrapper.py``.

``positional``
    The prototype. For a plain function with ``1 <= co_argcount <=
    MAX_ARGS`` it generates ``wrapper(_p0=M, ..., _pN=M, /, *args,
    **kwargs)`` with a private sentinel ``M``. A call with exactly ``N``
    positional arguments and no keywords takes a fast path that builds the
    ``fast_call`` tuple from the locals; everything else rebuilds
    ``args`` and runs the production logic. The generated source contains
    only fixed identifiers and integers, never a user name or ``repr()``.
    Every call still binds (so ``call_count`` counts invalid calls), and
    keywords never bind to user parameters (so positional-only parameters,
    ``__defaults__`` changes and a replaced ``__code__`` with another
    signature all reach the original function unchanged). One known gap:
    the sentinel is reachable as ``wrapper.__defaults__[i]``, and passing
    it explicitly as a trailing argument looks like an omitted one.

The ablations below are **not** contract-preserving. They only attribute
the wrapper's cost to individual lines of its body, and are timed on the
warmed path only:

``exact_signature``
    A generated ``wrapper(_p0, ..., _pN, /)`` with no ``*args``,
    ``**kwargs`` or defaults: the upper bound for any signature-specific
    wrapper. It breaks the contract for keywords, defaults, invalid calls
    and ``__code__`` replacement, so it is never a candidate.
``no_arity_check``
    ``production`` without ``len(args) == spec.arg_count``
    (``try_call`` checks the arity in Rust anyway).
``dict_counter``
    ``production`` counting with ``wrapper.__dict__["call_count"] += 1``
    instead of an attribute update.
``floor``
    A wrapper that only does ``spec.fast_call(args)``: no counter, no
    ``__code__`` check, no fallback. The lowest cost of any Python-level
    wrapper around the native fast call.

Every wrapper is built and verified with the benchmark suite's own
helpers (``workloads.prepare_layers`` and ``workloads.fallback_cases``),
so a warmed call is proven, with ``sys.setprofile``, to make exactly one
native fast call and to enter neither the original function nor
``CompiledI64.__call__`` before anything is timed.
"""

from __future__ import annotations

import argparse
import functools
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import FunctionType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from native_wrapper import python_wrapper  # noqa: E402

import workloads  # noqa: E402
from speedysnek import _native, snekit  # noqa: E402
from speedysnek._decorator import _JitState, _Status  # noqa: E402

# --- The prototype --------------------------------------------------------

_MISSING = object()


def _positional_source(count: int) -> str:
    """The factory source for a wrapper with ``count`` positional slots.

    Only fixed identifiers and ``count``-derived integers appear in it.
    """
    slots = [f"_p{i}" for i in range(count)]
    names = ", ".join(slots)
    params = ", ".join(f"{slot}=_missing" for slot in slots)
    # The positional arguments actually passed are a prefix of the slots;
    # find its end by identity only, so no user ``__eq__`` runs.
    prefix = [f"        if {slots[0]} is _missing:\n            given = ()\n"]
    for i in range(1, count):
        prefix.append(
            f"        elif {slots[i]} is _missing:\n"
            f"            given = ({', '.join(slots[:i])},)\n"
        )
    prefix.append(f"        else:\n            given = ({names},)\n")
    return (
        "def _factory(func, state, cold, _missing):\n"
        f"    def wrapper({params}, /, *args, **kwargs):\n"
        "        wrapper.call_count += 1\n"
        f"        if {slots[-1]} is not _missing and not args and not kwargs:\n"
        "            spec = state._specialization\n"
        "            if spec is None:\n"
        "                if state.status is not cold:\n"
        f"                    return func({names})\n"
        "                spec = state.cold_call()\n"
        "                if spec is None:\n"
        f"                    return func({names})\n"
        "            if func.__code__ is not spec.code:\n"
        "                state.invalidate()\n"
        "            else:\n"
        "                try:\n"
        # ``try_call`` returns None if the specialization has another
        # arity (``__code__`` replaced before compilation).
        f"                    result = spec.fast_call(({names},))\n"
        "                except Exception as error:\n"
        "                    state.native_failed(error)\n"
        "                else:\n"
        "                    if result is not None:\n"
        "                        return result\n"
        f"            return func({names})\n"
        + "".join(prefix)
        + "        args = given + args\n"
        "        spec = state._specialization\n"
        "        if spec is None:\n"
        "            if state.status is not cold:\n"
        "                return func(*args, **kwargs)\n"
        "            spec = state.cold_call()\n"
        "            if spec is None:\n"
        "                return func(*args, **kwargs)\n"
        "        if func.__code__ is not spec.code:\n"
        "            state.invalidate()\n"
        "        elif not kwargs and len(args) == spec.arg_count:\n"
        "            try:\n"
        "                result = spec.fast_call(args)\n"
        "            except Exception as error:\n"
        "                state.native_failed(error)\n"
        "            else:\n"
        "                if result is not None:\n"
        "                    return result\n"
        "        return func(*args, **kwargs)\n"
        "    return wrapper\n"
    )


def positional(func: Callable[..., Any]) -> Callable[..., Any]:
    """The signature-specific prototype; ``production`` outside its subset."""
    if not isinstance(func, FunctionType):
        return python_wrapper(func)
    count = func.__code__.co_argcount
    if not 1 <= count <= _native.MAX_ARGS:
        return python_wrapper(func)
    namespace: dict[str, Any] = {}
    exec(_positional_source(count), namespace)
    state = _JitState(func)
    wrapper = namespace["_factory"](func, state, _Status.COLD, _MISSING)
    functools.update_wrapper(wrapper, func)
    wrapper.call_count = 0
    wrapper._jit_state = state
    return wrapper


# --- Ablations (attribution only, not contract-preserving) ---------------
#
# Each is ``production`` with one thing changed, written out in full so
# that no ablation pays for a configuration branch the others do not.


def no_arity_check(func: Callable[..., Any]) -> Callable[..., Any]:
    state = _JitState(func)
    cold = _Status.COLD

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wrapper.call_count += 1
        spec = state._specialization
        if spec is None:
            if state.status is not cold:
                return func(*args, **kwargs)
            spec = state.cold_call()
            if spec is None:
                return func(*args, **kwargs)
        if func.__code__ is not spec.code:
            state.invalidate()
        elif not kwargs:
            try:
                result = spec.fast_call(args)
            except Exception as error:
                state.native_failed(error)
            else:
                if result is not None:
                    return result
        return func(*args, **kwargs)

    wrapper.call_count = 0
    wrapper._jit_state = state
    return wrapper


def dict_counter(func: Callable[..., Any]) -> Callable[..., Any]:
    state = _JitState(func)
    cold = _Status.COLD

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        counters["call_count"] += 1
        spec = state._specialization
        if spec is None:
            if state.status is not cold:
                return func(*args, **kwargs)
            spec = state.cold_call()
            if spec is None:
                return func(*args, **kwargs)
        if func.__code__ is not spec.code:
            state.invalidate()
        elif not kwargs and len(args) == spec.arg_count:
            try:
                result = spec.fast_call(args)
            except Exception as error:
                state.native_failed(error)
            else:
                if result is not None:
                    return result
        return func(*args, **kwargs)

    wrapper.call_count = 0
    counters = wrapper.__dict__
    wrapper._jit_state = state
    return wrapper


def floor(func: Callable[..., Any]) -> Callable[..., Any]:
    state = _JitState(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        spec = state._specialization
        if spec is None:
            state.cold_call()
            return func(*args, **kwargs)
        return spec.fast_call(args)

    wrapper._jit_state = state
    return wrapper


def _exact_source(count: int) -> str:
    names = ", ".join(f"_p{i}" for i in range(count))
    return (
        "def _factory(func, state):\n"
        f"    def wrapper({names}, /):\n"
        "        wrapper.call_count += 1\n"
        "        spec = state._specialization\n"
        "        if spec is None:\n"
        "            state.cold_call()\n"
        f"            return func({names})\n"
        "        if func.__code__ is spec.code:\n"
        f"            result = spec.fast_call(({names},))\n"
        "            if result is not None:\n"
        "                return result\n"
        f"        return func({names})\n"
        "    return wrapper\n"
    )


def exact_signature(func: Callable[..., Any]) -> Callable[..., Any]:
    """The cheapest generated wrapper: exactly ``N`` positional parameters.

    Unsafe, so never a candidate: binding fails before the body runs for
    keywords, omitted defaults and invalid calls (no ``call_count``, a
    different ``TypeError``), and a replaced ``__code__`` with another
    signature cannot be called through it. It bounds what any
    signature-specific wrapper could gain on the warmed path.
    """
    namespace: dict[str, Any] = {}
    exec(_exact_source(func.__code__.co_argcount), namespace)
    state = _JitState(func)
    wrapper = namespace["_factory"](func, state)
    functools.update_wrapper(wrapper, func)
    wrapper.call_count = 0
    wrapper._jit_state = state
    return wrapper


VARIANTS: dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]] = {
    "production": python_wrapper,
    "positional": positional,
    "exact_signature": exact_signature,
    "no_arity_check": no_arity_check,
    "dict_counter": dict_counter,
    "floor": floor,
}

# Variants measured on the fallback cases too. The ablations only explain
# the warmed path: some have no fallback or reject keywords, and
# ``no_arity_check`` makes a native call where the suite expects none.
FULL_VARIANTS = ("production", "positional")

# --- Measurement ---------------------------------------------------------


def build(variant: str) -> dict[str, tuple[str, dict[str, Any]]]:
    """Every case for one variant, built and verified by the suite."""
    workloads.snekit = VARIANTS[variant]
    try:
        cases: dict[str, tuple[str, dict[str, Any]]] = {}
        for ops in (1, 20, 50):
            layers = workloads.prepare_layers(ops)
            cases[f"ops_{ops:03d}"] = layers.statement("snekit")
            if variant == "production":
                cases[f"ops_{ops:03d}.python"] = layers.statement("python")
        if variant not in FULL_VARIANTS:
            return cases
        for case in workloads.fallback_cases():
            if case.name in ("disabled", "float_args", "keyword_args", "default_args"):
                cases[case.name] = case.statement("snekit")
                if variant == "production":
                    cases[f"{case.name}.python"] = case.statement("python")
        return cases
    finally:
        workloads.snekit = snekit


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--loops", type=int, default=4000)
    options = parser.parse_args()
    if _native.DEBUG_BUILD:
        sys.exit("debug build of speedysnek._native: rebuild with --release")

    grid = [
        (variant, name, stmt)
        for variant in VARIANTS
        for name, stmt in build(variant).items()
    ]
    samples: dict[tuple[str, str], list[float]] = {(v, n): [] for v, n, _ in grid}
    for _, _, (statement, namespace) in grid:
        time_call(statement, namespace, options.loops // 4)  # warm the code
    for _ in range(options.rounds):
        for variant, name, (statement, namespace) in grid:
            samples[variant, name].append(
                time_call(statement, namespace, options.loops)
            )

    print(
        f"# {platform.python_implementation()} {platform.python_version()}, "
        f"{platform.machine()}, pid {os.getpid()}, "
        f"{options.rounds} interleaved rounds x {options.loops * 10} calls"
    )
    names = sorted({n for _, n, _ in grid if not n.endswith(".python")})
    header = ["case", "python"] + list(VARIANTS)
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name in names:
        row = [name]
        python = samples.get(("production", f"{name}.python"))
        row.append(f"{statistics.median(python):.0f}" if python else "")
        for variant in VARIANTS:
            values = samples.get((variant, name))
            row.append(f"{statistics.median(values):.0f}" if values else "-")
        print("| " + " | ".join(row) + " |")
    print("\nMedian ns per call. Spread (max/min of medians across processes) is")
    print("reported in wrapper_overhead.md; compare variants within one run.")


if __name__ == "__main__":
    main()
