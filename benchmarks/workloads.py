"""Deterministic workloads shared by the benchmarks and their tests.

Nothing here measures time and nothing imports pyperf, so the tests can
check every workload without the benchmark dependencies.

Runtime workloads
-----------------

:func:`make_workload` builds ``ops_NNN(a, b, c, d)``: one ``return`` of
straight-line arithmetic with exactly ``NNN`` normalized ``ADD``,
``SUBTRACT`` and ``MULTIPLY`` instructions. The expression is a flat sum of
products of arguments (``a + b * c + c * d - d * 4 ...``), so:

- every operation reads a runtime argument and nothing can be constant
  folded by CPython;
- every product is bounded by ``max(|arg|) ** 2`` and the running sum by
  ``ops * max(|arg|) ** 2``, far inside the signed 64-bit range for
  :data:`ARGS`;
- the stack never gets deeper than three values.

:func:`prepare_layers` produces the same workload at every execution layer
the runtime benchmark measures, and checks that all of them return the same
value and that the decorator really runs native code.

Fallback cases
--------------

:func:`fallback_cases` builds a decorated function in a specific lifecycle
state together with a *separate*, undecorated copy of the same function, so
the fallback benchmarks compare the wrapper against exactly the work it
falls back to.
"""

from __future__ import annotations

import collections
import contextlib
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from types import FunctionType
from typing import Any

from speedysnek import _native, snekit
from speedysnek._compiler import I64_MAX, I64_MIN, CompiledI64
from speedysnek._decorator import _WARMUP_CALLS, _Status
from speedysnek._frontend import Instruction, Op, analyze

# Approximate workload sizes, in normalized arithmetic operations.
OP_COUNTS = (1, 5, 20, 50, 100, 250)

ARG_NAMES = ("a", "b", "c", "d")

# Runtime inputs. They are compact CPython ints (|x| < 2**30), as are all
# workload results, so the Python baseline uses CPython's fast int paths.
ARGS = (1234, -567, 89, -1011)

# Execution layers measured by ``bench_runtime.py``, outermost first.
LAYERS = ("python", "snekit", "compiled_i64", "native_call")

# Fallback and wrapper-overhead cases, in report order.
FALLBACK_CASES = (
    "cold",
    "disabled",
    "float_args",
    "big_int_args",
    "overflow",
    "keyword_args",
    "default_args",
)

# The number of calls that take a fresh wrapper through warmup and its
# first (compiling) call.
LIFECYCLE_CALLS = _WARMUP_CALLS + 1

# ``warmup_calls`` value that keeps a wrapper on its cold path for any
# realistic number of benchmark iterations. It stays a compact int, so the
# counter increment costs the same as during a real warmup.
FOREVER_COLD = -1_000_000_000

_ARITHMETIC = frozenset({Op.ADD, Op.SUBTRACT, Op.MULTIPLY})


class BenchmarkSetupError(RuntimeError):
    """A benchmark would not measure what it claims to measure."""


# --- Workload construction ----------------------------------------------


def expression(ops: int) -> str:
    """Return the source of a flat expression with exactly ``ops`` operations.

    >>> expression(1)
    'a + b'
    >>> expression(5)
    'a + b * c + c * d - d'
    """
    if type(ops) is not int or ops < 1:
        raise ValueError(f"ops must be a positive int, got {ops!r}")
    parts = [ARG_NAMES[0]]
    remaining = ops
    term = 0
    while remaining:
        sign = "-" if term % 3 == 2 else "+"
        left = ARG_NAMES[(term + 1) % 4]
        if remaining >= 2:
            # Every fifth product uses a constant; it still reads an
            # argument, so nothing folds.
            if term % 5 == 4:
                right = str(term % 9 + 2)
            else:
                right = ARG_NAMES[(term + 2 + term // 4) % 4]
            parts.append(f"{sign} {left} * {right}")
            remaining -= 2
        else:
            parts.append(f"{sign} {left}")
            remaining -= 1
        term += 1
    return " ".join(parts)


def make_workload(ops: int) -> FunctionType:
    """Return a new, undecorated ``ops_NNN(a, b, c, d)`` function."""
    return _define(f"ops_{ops:03d}", ", ".join(ARG_NAMES), expression(ops))


def _define(name: str, params: str, body: str) -> FunctionType:
    # A fresh namespace per function: no globals, no closures, no sharing.
    namespace: dict[str, Any] = {"__name__": __name__}
    exec(f"def {name}({params}):\n    return {body}\n", namespace)
    return namespace[name]


def arithmetic_count(instructions: tuple[Instruction, ...]) -> int:
    """Count the normalized ``ADD``, ``SUBTRACT`` and ``MULTIPLY``."""
    return sum(1 for instruction in instructions if instruction.op in _ARITHMETIC)


def intermediate_values(
    instructions: tuple[Instruction, ...], args: tuple[int, ...]
) -> list[int]:
    """Every value the program pushes, computed with exact Python ints."""
    stack: list[int] = []
    pushed: list[int] = []
    for instruction in instructions:
        op = instruction.op
        if op is Op.LOAD_ARG:
            value = args[instruction.arg]
        elif op is Op.LOAD_CONST:
            value = instruction.arg
        elif op is Op.RETURN:
            break
        else:
            right = stack.pop()
            left = stack.pop()
            if op is Op.ADD:
                value = left + right
            elif op is Op.SUBTRACT:
                value = left - right
            else:
                value = left * right
        stack.append(value)
        pushed.append(value)
    return pushed


def fits_i64(value: int) -> bool:
    return I64_MIN <= value <= I64_MAX


# --- Proving which path a call takes ------------------------------------


@contextlib.contextmanager
def _recording() -> Iterator[collections.Counter]:
    """Count the Python frames entered, by code object, in this thread."""
    calls: collections.Counter = collections.Counter()

    def profiler(frame, event, _arg):
        if event == "call":
            calls[frame.f_code] += 1

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def call_path(wrapper: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """Call ``wrapper`` once and report ``"native"`` or ``"python"``.

    A native call enters ``CompiledI64.__call__`` and no frame of the
    original function; a fallback enters the original function once.
    """
    original = wrapper.__wrapped__.__code__
    with _recording() as calls:
        wrapper(*args, **kwargs)
    python_calls = calls[original]
    native_calls = calls[CompiledI64.__call__.__code__]
    if (python_calls, native_calls) == (0, 1):
        return "native"
    if python_calls == 1:
        return "python"
    raise BenchmarkSetupError(
        f"unexpected call path: {python_calls} Python, {native_calls} native frame(s)"
    )


def status_of(wrapper: Callable[..., Any]) -> _Status:
    return wrapper._jit_state.status


# --- Runtime layers -----------------------------------------------------


@dataclass(frozen=True)
class Layers:
    """One workload at every execution layer, verified and ready to time.

    Each layer has a ``(statement, globals)`` pair for ``pyperf.timeit``.
    The first three layers use the identical statement ``f(a, b, c, d)``,
    so they pay the same Python call-site cost.
    """

    ops: int
    python: FunctionType
    snekit: Callable[..., int]
    compiled: CompiledI64
    native_call: Callable[[tuple[int, ...]], tuple[int, int]]
    args: tuple[int, ...]
    expected: int

    def statement(self, layer: str) -> tuple[str, dict[str, Any]]:
        values = dict(zip(ARG_NAMES, self.args, strict=True))
        if layer == "python":
            return "f(a, b, c, d)", {"f": self.python, **values}
        if layer == "snekit":
            return "f(a, b, c, d)", {"f": self.snekit, **values}
        if layer == "compiled_i64":
            return "f(a, b, c, d)", {"f": self.compiled, **values}
        if layer == "native_call":
            # The argument tuple is built once, outside the timed statement.
            return "call(args)", {"call": self.native_call, "args": self.args}
        raise ValueError(f"unknown layer {layer!r}")

    def run(self, layer: str) -> int:
        """Run one layer once, the way its benchmark does, and return the int."""
        statement, namespace = self.statement(layer)
        result = eval(statement, dict(namespace))
        if layer == "native_call":
            status, value = result
            if status != _native.STATUS_OK:
                raise BenchmarkSetupError(f"native status {status}")
            return value
        return result


def warm_up(wrapper: Callable[..., Any], *args: Any) -> None:
    """Make the warmup calls and the compiling call (the lifecycle)."""
    for _ in range(LIFECYCLE_CALLS):
        wrapper(*args)


def prepare_layers(ops: int, args: tuple[int, ...] = ARGS) -> Layers:
    """Build, compile and verify one workload at every layer.

    Raises :class:`BenchmarkSetupError` if any layer disagrees, if the
    inputs could overflow, or if the warmed decorator would not run native
    code: the benchmark must never silently measure a fallback.
    """
    python = make_workload(ops)
    analysis = analyze(python)
    if not analysis.supported:
        raise BenchmarkSetupError(f"ops_{ops:03d} was rejected: {analysis.rejection}")
    instructions = analysis.instructions
    if arithmetic_count(instructions) != ops:
        raise BenchmarkSetupError(
            f"ops_{ops:03d} has {arithmetic_count(instructions)} operations"
        )
    if not all(fits_i64(v) for v in (*args, *intermediate_values(instructions, args))):
        raise BenchmarkSetupError(f"ops_{ops:03d} overflows i64 for {args}")

    wrapper = snekit(python)
    warm_up(wrapper, *args)
    if status_of(wrapper) is not _Status.COMPILED:
        raise BenchmarkSetupError(
            f"ops_{ops:03d} did not compile: {status_of(wrapper).value}"
        )
    if call_path(wrapper, *args) != "native":
        raise BenchmarkSetupError(f"ops_{ops:03d} fell back to Python")

    # The exact object the wrapper calls, not a second compilation.
    compiled = wrapper._jit_state.specialization.native
    layers = Layers(
        ops=ops,
        python=python,
        snekit=wrapper,
        compiled=compiled,
        native_call=compiled._function.call,
        args=args,
        expected=python(*args),
    )
    for layer in LAYERS:
        if layers.run(layer) != layers.expected:
            raise BenchmarkSetupError(f"ops_{ops:03d}: layer {layer} disagrees")
    return layers


def reference_noop() -> FunctionType:
    """A four-argument function that does no arithmetic at all."""
    return _define("noop", ", ".join(ARG_NAMES), "a")


# --- Fallback and wrapper overhead --------------------------------------


@dataclass(frozen=True)
class FallbackCase:
    """A wrapper in one lifecycle state and its matched Python baseline.

    ``python`` is an undecorated copy of the wrapped function, so the
    Python side never shares state with the wrapper. Both sides are timed
    with the same statement, built by :meth:`statement` from ``args`` and
    ``kwargs``, for example ``f(x0, x1)`` or ``f(a=k_a, b=k_b)``.
    """

    name: str
    description: str
    python: FunctionType
    wrapper: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    status: _Status = _Status.COMPILED

    def statement(self, side: str) -> tuple[str, dict[str, Any]]:
        if side == "python":
            f = self.python
        elif side == "snekit":
            f = self.wrapper
        else:
            raise ValueError(f"unknown side {side!r}")
        names = [f"x{i}" for i in range(len(self.args))]
        names += [f"{key}=k_{key}" for key in self.kwargs]
        namespace = {"f": f}
        namespace.update((f"x{i}", value) for i, value in enumerate(self.args))
        namespace.update((f"k_{key}", value) for key, value in self.kwargs.items())
        return f"f({', '.join(names)})", namespace

    def run(self, side: str) -> Any:
        statement, namespace = self.statement(side)
        return eval(statement, namespace)


_PAIR = "a * b + a - b"


def _pair(params: str = "a, b") -> FunctionType:
    return _define("pair", params, _PAIR)


def _rejected_pair() -> FunctionType:
    return _define("pair", "a, b", "(a * b + a) // b")


def _compiled(make: Callable[[], FunctionType]) -> Callable[..., Any]:
    wrapper = snekit(make())
    warm_up(wrapper, 3, 4)
    return wrapper


def fallback_cases() -> list[FallbackCase]:
    """Every fallback and wrapper-overhead case, prepared and verified."""
    cold = snekit(_pair())
    # Benchmark-only: keep the wrapper on the exact code path that the
    # first ``_WARMUP_CALLS`` calls take, for every timed iteration.
    cold._jit_state.warmup_calls = FOREVER_COLD

    cases = [
        FallbackCase(
            "cold",
            "warmup call: counted, then runs Python",
            _pair(),
            cold,
            (3, 4),
            status=_Status.COLD,
        ),
        FallbackCase(
            "disabled",
            "rejected function (`//`), any call after it was disabled",
            _rejected_pair(),
            _compiled(_rejected_pair),
            (3, 4),
            status=_Status.DISABLED,
        ),
        FallbackCase(
            "float_args",
            "compiled, `float` arguments fail the exact-`int` guard",
            _pair(),
            _compiled(_pair),
            (3.5, 4.25),
        ),
        FallbackCase(
            "big_int_args",
            "compiled, an `int` outside i64 fails the range guard",
            _pair(),
            _compiled(_pair),
            (2**64, 4),
        ),
        FallbackCase(
            "overflow",
            "compiled, native `*` overflows, then Python recomputes",
            _pair(),
            _compiled(_pair),
            (2**62, 2**62),
        ),
        FallbackCase(
            "keyword_args",
            "compiled, keyword arguments skip native code",
            _pair(),
            _compiled(_pair),
            kwargs={"a": 3, "b": 4},
        ),
        FallbackCase(
            "default_args",
            "compiled, an omitted default argument skips native code",
            _pair("a, b=4"),
            _compiled(lambda: _pair("a, b=4")),
            (3,),
        ),
    ]
    assert tuple(case.name for case in cases) == FALLBACK_CASES
    for case in cases:
        _verify_fallback(case)
    return cases


def _verify_fallback(case: FallbackCase) -> None:
    if status_of(case.wrapper) is not case.status:
        raise BenchmarkSetupError(
            f"{case.name}: wrapper is {status_of(case.wrapper).value}, "
            f"expected {case.status.value}"
        )
    expected = case.run("python")
    actual = case.run("snekit")
    if actual != expected or type(actual) is not type(expected):
        raise BenchmarkSetupError(f"{case.name}: wrapper result differs")
    if call_path(case.wrapper, *case.args, **case.kwargs) != "python":
        raise BenchmarkSetupError(f"{case.name}: did not fall back to Python")
    if status_of(case.wrapper) is not case.status:
        raise BenchmarkSetupError(f"{case.name}: state changed while verifying")


# --- Compilation --------------------------------------------------------


def lifecycle(func: FunctionType, args: tuple[Any, ...]) -> Callable[..., Any]:
    """Decorate ``func`` afresh and drive it through warmup and compilation.

    Every call creates a new wrapper, and with it a new ``_JitState``, so
    nothing compiled by an earlier iteration is ever reused.
    """
    wrapper = snekit(func)
    for _ in range(LIFECYCLE_CALLS):
        wrapper(*args)
    return wrapper


def unsupported_workload() -> FunctionType:
    """``ops_005`` with one extra ``//``: the frontend rejects it."""
    return _define(
        "rejected", ", ".join(ARG_NAMES), f"({expression(5)}) // {ARG_NAMES[1]}"
    )
