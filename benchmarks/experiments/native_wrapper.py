"""Quick experiment: does a native (Rust) ``@snekit`` wrapper cut call overhead?

This is a short, interleaved A/B measurement, not part of the pyperf
suite. Its conclusion is in ``native_wrapper.md`` next to this file. Run
it with a release build, from the repository root::

    python benchmarks/experiments/native_wrapper.py
    taskset -c 3 python benchmarks/experiments/native_wrapper.py --rounds 15

Variants
--------

``python_wrapper``
    The Python ``wrapper(*args, **kwargs)`` closure that ``snekit``
    returned before this experiment, copied verbatim below.
``native_wrapper``
    ``speedysnek.snekit`` as shipped since this experiment: a
    ``speedysnek._native.NativeWrapper``, the same per-call logic in a
    PyO3 ``__call__``, sharing the Python ``_JitState`` and calling the
    shared Rust ``try_call`` helper directly.

Both variants use the same ``_JitState``, the same compiled code and the
same Python baselines.

Before anything is timed, every case is verified:

- the wrapper returns what the undecorated function returns, with the
  same type;
- the call takes the expected path, proven by counting frames of the
  original function with ``sys.setprofile`` and native fast calls with
  ``I64Function.try_calls`` (a call from Rust emits no profile event):
  a warmed call makes one native call and no Python call, a guard miss
  one of each, and every other fallback one Python call and no native
  call;
- for ``native_wrapper``, a second wrapper of the same function is
  compiled to a canary ``I64Function`` that returns a constant, and it
  must return that constant: the direct Rust path really runs the
  compiled machine code.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import workloads  # noqa: E402
from speedysnek import _compiler, _native, snekit  # noqa: E402
from speedysnek._compiler import CompiledI64  # noqa: E402
from speedysnek._decorator import _JitState, _Status  # noqa: E402

# --- The variants ---------------------------------------------------------


def python_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    """The Python wrapper ``snekit`` returned before this experiment.

    Verbatim, except that it reads ``state._specialization``: the slot that
    was then the plain ``specialization`` attribute, which is now a
    property in front of it.
    """
    if not callable(func):
        raise TypeError(f"snekit expected a callable, got {type(func).__name__}")

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
    wrapper._jit_state = state
    return wrapper


VARIANTS: dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]] = {
    "python_wrapper": python_wrapper,
    "native_wrapper": snekit,
}

# --- Cases ----------------------------------------------------------------

# Extra case beyond the requested ones: an instance-method call, which
# binds through ``__get__`` and then misses the guards (``self`` is not an
# int).
CASES = (
    "ops_001",
    "ops_020",
    "ops_050",
    "float_args",
    "keyword_args",
    "default_args",
    "disabled",
    "method",
)


class Case:
    """One timed call: a statement and its namespace, plus the expectation."""

    def __init__(
        self,
        func: Callable[..., Any],
        wrapper: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        path: str = "native",
        status: _Status = _Status.COMPILED,
        receiver: Any = None,
    ) -> None:
        self.func = func
        self.wrapper = wrapper
        self.args = args
        self.kwargs = kwargs or {}
        self.path = path
        self.status = status
        self.receiver = receiver

    def statement(self) -> tuple[str, dict[str, Any]]:
        names = [f"x{i}" for i in range(len(self.args))]
        names += [f"{key}=k_{key}" for key in self.kwargs]
        namespace: dict[str, Any] = {"f": self.wrapper, "obj": self.receiver}
        namespace.update((f"x{i}", value) for i, value in enumerate(self.args))
        namespace.update((f"k_{key}", value) for key, value in self.kwargs.items())
        callee = "obj.m" if self.receiver is not None else "f"
        return f"{callee}({', '.join(names)})", namespace

    def call(self) -> Any:
        if self.receiver is not None:
            return self.wrapper.__get__(self.receiver)(*self.args, **self.kwargs)
        return self.wrapper(*self.args, **self.kwargs)


def _warm(wrapper: Callable[..., Any], *args: Any) -> Callable[..., Any]:
    workloads.warm_up(wrapper, *args)
    return wrapper


def build(decorate: Callable[..., Any], python: bool = False) -> dict[str, Case]:
    """Every case for one variant, or undecorated (``python=True``)."""

    def wrap(func: Callable[..., Any], *warm: Any) -> Callable[..., Any]:
        return func if python else _warm(decorate(func), *warm)

    cases = {}
    for ops in (1, 20, 50):
        func = workloads.make_workload(ops)
        cases[f"ops_{ops:03d}"] = Case(
            func, wrap(func, *workloads.ARGS), workloads.ARGS
        )
    func = workloads._pair()
    cases["float_args"] = Case(func, wrap(func, 3, 4), (3.5, 4.25), path="guard_miss")
    func = workloads._pair()
    cases["keyword_args"] = Case(
        func, wrap(func, 3, 4), kwargs={"a": 3, "b": 4}, path="python"
    )
    func = workloads._pair("a, b=4")
    cases["default_args"] = Case(func, wrap(func, 3, 4), (3,), path="python")
    func = workloads._rejected_pair()
    cases["disabled"] = Case(
        func, wrap(func, 3, 4), (3, 4), path="python", status=_Status.DISABLED
    )
    func = workloads._define("m", "self, a, b", "a * b + a - b")
    method = wrap(func, 0, 3, 4)
    cls = type("Receiver", (), {"m": method})
    cases["method"] = Case(func, method, (3, 4), path="guard_miss", receiver=cls())
    return cases


# --- Verification ---------------------------------------------------------


class SetupError(Exception):
    """A case would not measure what it claims to."""


@contextlib.contextmanager
def _frames(code: Any) -> Iterator[list[int]]:
    """Count frames of ``code`` entered in this thread."""
    count = [0]

    def profiler(frame, event, arg):
        if event == "call" and frame.f_code is code:
            count[0] += 1

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        yield count
    finally:
        sys.setprofile(previous)


def _function(case: Case) -> Any:
    spec = case.wrapper._jit_state.specialization
    return None if spec is None else spec.native._function


def path_of(case: Case) -> str:
    function = _function(case)
    before = function.try_calls if function is not None else 0
    with _frames(case.func.__code__) as frames:
        case.call()
    native = (function.try_calls if function is not None else 0) - before
    paths = {(0, 1): "native", (1, 1): "guard_miss", (1, 0): "python"}
    observed = (frames[0], native)
    if observed not in paths:
        raise SetupError(f"unexpected path: {observed[0]} frame(s), {native} native")
    return paths[observed]


def verify(name: str, variant: str, case: Case, python: Case) -> None:
    where = f"{variant}/{name}"
    if case.wrapper._jit_state.status is not case.status:
        raise SetupError(f"{where}: status {case.wrapper._jit_state.status}")
    expected = python.call()
    actual = case.call()
    if actual != expected or type(actual) is not type(expected):
        raise SetupError(f"{where}: result {actual!r}, expected {expected!r}")
    if path_of(case) != case.path:
        raise SetupError(f"{where}: path {path_of(case)!r}, expected {case.path!r}")
    if variant == "native_wrapper" and case.path == "native":
        _verify_canary(where, case)
    if case.wrapper._jit_state.status is not case.status:
        raise SetupError(f"{where}: state changed while verifying")


def _verify_canary(where: str, case: Case) -> None:
    # The published specialization is immutable, so the canary is what a
    # second wrapper of the same function compiles to.
    real = _compiler.compile_i64

    def compile_canary(instructions: Any, arg_count: int, **kwargs: Any) -> Any:
        canary = _native.compile_i64(
            ((_native.OP_LOAD_CONST, 424242), (_native.OP_RETURN,)), arg_count
        )
        return CompiledI64(canary, tuple(instructions))

    _compiler.compile_i64 = compile_canary
    try:
        wrapper = _warm(snekit(case.func), *case.args)
    finally:
        _compiler.compile_i64 = real
    canary = wrapper._jit_state.specialization.function
    before = canary.try_calls
    with _frames(case.func.__code__) as frames:
        result = wrapper(*case.args, **case.kwargs)
    if (result, frames[0], canary.try_calls - before) != (424242, 0, 1):
        raise SetupError(f"{where}: the canary machine code did not run")


# --- Measurement ----------------------------------------------------------


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
    parser.add_argument("--loops", type=int, default=20000)
    options = parser.parse_args()
    if _native.DEBUG_BUILD:
        sys.exit("debug build of speedysnek._native: rebuild with --release")

    python = build(python_wrapper, python=True)
    grid = [("python", name, case) for name, case in python.items()]
    for variant, decorate in VARIANTS.items():
        for name, case in build(decorate).items():
            verify(name, variant, case, python[name])
            grid.append((variant, name, case))

    samples: dict[tuple[str, str], list[float]] = {(v, n): [] for v, n, _ in grid}
    for _, _, case in grid:
        time_call(*case.statement(), options.loops // 4)  # warm the code
    for _ in range(options.rounds):
        for variant, name, case in grid:
            samples[variant, name].append(time_call(*case.statement(), options.loops))
    for _, _, case in grid:
        if case.wrapper is not case.func and case.wrapper._jit_state.status is not (
            case.status
        ):
            raise SetupError("a wrapper changed state while it was timed")

    print(
        f"# {platform.python_implementation()} {platform.python_version()}, "
        f"{platform.machine()}, pid {os.getpid()}, "
        f"{options.rounds} interleaved rounds x {options.loops * 10} calls"
    )
    header = ["case", "python", *VARIANTS, "ratio"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name in CASES:
        medians = [statistics.median(samples[v, name]) for v in ("python", *VARIANTS)]
        ratio = medians[2] / medians[1]
        print(
            f"| {name} | "
            + " | ".join(f"{m:.0f}" for m in medians)
            + f" | {ratio:.2f} |"
        )
    print("\nMedian ns per call; ratio = native_wrapper / python_wrapper.")


if __name__ == "__main__":
    main()
