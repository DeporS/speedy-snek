"""Tests for ``@snekit``: warmup, compilation, guards and fallback.

Native execution is proven with ``sys.setprofile``, never inferred from an
equal result. A warmed native call makes exactly one C call to the
specialization's bound ``I64Function.try_call`` and enters no frame of the
original function and no frame of ``CompiledI64.__call__`` (the diagnostic
path, which the decorator never uses). A Python call runs exactly one
frame of the original function.
"""

import asyncio
import collections
import contextlib
import doctest
import functools
import json
import subprocess
import sys
import threading
import time
from types import BuiltinMethodType

import pytest

import speedysnek
from speedysnek import _compiler, _decorator, _frontend, _native, snekit
from speedysnek._compiler import I64_MAX, I64_MIN, CompiledI64, NativeGuardFailure
from speedysnek._decorator import (
    _WARMUP_CALLS,
    _DisabledReason,
    _Specialization,
    _Status,
)
from speedysnek._frontend import RejectionReason, analyze
from speedysnek._interpreter import execute

# Keys of the counter that ``recording`` fills.
FAST_CALL = "I64Function.try_call"
RAW_CALL = "I64Function.call"
LEGACY_CALL = CompiledI64.__call__.__code__


# --- Helpers -------------------------------------------------------------


def native_method(callable_):
    """The name of a bound ``I64Function`` method, else ``None``."""
    if type(callable_) is BuiltinMethodType and (
        type(callable_.__self__) is _native.I64Function
    ):
        return f"I64Function.{callable_.__name__}"
    return None


@contextlib.contextmanager
def recording():
    """Count, in this thread, the Python frames entered (by code object)
    and the calls to native ``I64Function`` methods (by ``FAST_CALL`` and
    ``RAW_CALL``)."""
    calls = collections.Counter()

    def profiler(frame, event, arg):
        if event == "call":
            calls[frame.f_code] += 1
        elif event == "c_call":
            name = native_method(arg)
            if name is not None:
                calls[name] += 1

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def traced(f, *args, **kwargs):
    """Call ``f`` and return ``(result, python_calls, fast_calls)``.

    Also checks that the decorator never takes the diagnostic path.
    """
    original = f.__wrapped__.__code__
    with recording() as calls:
        result = f(*args, **kwargs)
    assert calls[LEGACY_CALL] == 0
    assert calls[RAW_CALL] == 0
    return result, calls[original], calls[FAST_CALL]


def assert_native(f, *args):
    result, python_calls, fast_calls = traced(f, *args)
    assert (python_calls, fast_calls) == (0, 1)
    return result


def assert_python(f, *args, **kwargs):
    result, python_calls, fast_calls = traced(f, *args, **kwargs)
    assert python_calls == 1
    return result, fast_calls


def state_of(f):
    return f._jit_state


def warm(f, *args):
    """Make the warmup calls; the next call to ``f`` attempts compilation."""
    for _ in range(_WARMUP_CALLS):
        f(*args)
    assert state_of(f).status is _Status.COLD


def compiled(f, *args):
    """Warm ``f`` up and compile it with one compatible call."""
    warm(f, *args)
    f(*args)
    assert state_of(f).status is _Status.COMPILED
    return f


@pytest.fixture
def compile_spy(monkeypatch):
    """Count calls to the backend's ``compile_i64``."""
    calls = []
    real = _compiler.compile_i64

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(_compiler, "compile_i64", spy)
    return calls


def make_add():
    @snekit
    def add(a, b):
        return a + b

    return add


def make_expression():
    @snekit
    def expression(a, b):
        return (a + b) * 2

    return expression


# --- Warmup --------------------------------------------------------------


def test_warmup_calls_run_python(compile_spy):
    add = make_add()
    for i in range(_WARMUP_CALLS):
        result, native_calls = assert_python(add, i, 1)
        assert (result, native_calls) == (i + 1, 0)
    state = state_of(add)
    assert state.status is _Status.COLD
    assert state.compile_attempts == 0
    assert compile_spy == []


def test_compiles_once_on_the_call_after_warmup(compile_spy):
    add = make_add()
    warm(add, 1, 2)
    # This call compiles and then runs natively.
    assert assert_native(add, 3, 4) == 7
    state = state_of(add)
    assert state.status is _Status.COMPILED
    assert state.compile_attempts == 1
    assert len(compile_spy) == 1
    assert isinstance(state.compiled, CompiledI64)
    assert state.compiled_code is add.__wrapped__.__code__
    assert state.specialization.native is state.compiled
    assert state.disabled_reason is None


def test_compiled_code_is_cached_and_reused(compile_spy):
    add = compiled(make_add(), 1, 2)
    first = state_of(add).compiled
    for i in range(50):
        assert assert_native(add, i, -i) == 0
    assert state_of(add).compiled is first
    assert state_of(add).compile_attempts == 1
    assert len(compile_spy) == 1


def test_compilation_uses_the_full_positional_arity(compile_spy):
    @snekit
    def second(a, b, c):
        return b

    compiled(second, 1, 2, 3)
    instructions, arg_count = compile_spy[0]
    assert instructions == analyze(second.__wrapped__).instructions
    assert arg_count == 3
    assert state_of(second).specialization.arg_count == 3
    # Unread arguments must still pass the guards.
    assert assert_python(second, 1.5, 2, 3) == (2, 1)
    assert assert_native(second, 1, 2, 3) == 2


def test_warmup_counts_every_call_including_ineligible_ones():
    add = make_add()
    for _ in range(_WARMUP_CALLS):
        add(a=1, b=2)
    # The threshold is about calls, not argument types: the ninth call
    # compiles, then falls back because of its keyword arguments.
    assert assert_python(add, 1, b=2) == (3, 0)
    assert state_of(add).status is _Status.COMPILED
    assert assert_native(add, 1, 2) == 3


def test_states_are_independent():
    add = make_add()
    expression = make_expression()
    compiled(add, 1, 2)
    expression(1, 2)
    assert state_of(add).status is _Status.COMPILED
    assert state_of(expression).status is _Status.COLD
    assert state_of(expression).warmup_calls == 1
    assert (add.call_count, expression.call_count) == (_WARMUP_CALLS + 1, 1)
    assert state_of(add) is not state_of(expression)


def test_call_count_counts_every_path():
    add = compiled(make_add(), 1, 2)
    add(1, 2)  # native
    add(1.5, 2)  # guard failure
    add(1, b=2)  # keyword arguments
    with pytest.raises(TypeError):
        add(1, "x")  # raises
    with pytest.raises(TypeError):
        add(1)  # wrong arity
    assert add.call_count == _WARMUP_CALLS + 6


def test_calls_that_raise_are_counted_while_cold():
    add = make_add()
    for _ in range(3):
        with pytest.raises(TypeError):
            add(1, "x")
    assert add.call_count == 3
    assert state_of(add).warmup_calls == 3


def test_warmup_does_not_depend_on_call_count():
    add = make_add()
    for _ in range(_WARMUP_CALLS):
        add(1, 2)
        add.call_count = 0  # Users may reset the public counter.
    add(1, 2)
    assert state_of(add).status is _Status.COMPILED


# --- Native execution ----------------------------------------------------


def test_public_usage_example():
    expression = make_expression()
    for _ in range(100):
        result = expression(10, 20)
    assert result == 60
    assert state_of(expression).status is _Status.COMPILED
    assert assert_native(expression, 10, 20) == 60


@pytest.mark.parametrize(
    "args",
    [
        (0, 0),
        (10, 20),
        (-7, 3),
        (-(2**40), 2**40 - 1),
        (I64_MAX // 2, 0),
        (I64_MIN // 2, 0),
    ],
)
def test_native_results_match_python_and_the_interpreter(args):
    expression = compiled(make_expression(), 1, 2)
    original = expression.__wrapped__
    expected = original(*args)
    assert assert_native(expression, *args) == expected
    assert execute(analyze(original).instructions, args) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((I64_MAX, 0), I64_MAX),
        ((I64_MIN, 0), I64_MIN),
        ((I64_MAX, I64_MIN), -1),
        ((I64_MAX - 1, 1), I64_MAX),
        ((I64_MIN + 1, -1), I64_MIN),
    ],
)
def test_i64_boundary_values(args, expected):
    add = compiled(make_add(), 1, 2)
    assert assert_native(add, *args) == expected


def test_constants_and_negative_values():
    @snekit
    def f(a, b, c):
        return (a - 1000) * b - c * -3

    compiled(f, 1, 2, 3)
    for args in [(1, 2, 3), (-5, -6, -7), (0, 0, 0), (2**31, -(2**31), 99)]:
        expected = f.__wrapped__(*args)
        assert assert_native(f, *args) == expected
        assert execute(analyze(f.__wrapped__).instructions, args) == expected


def test_zero_argument_function():
    @snekit
    def answer():
        return 6 * 7

    compiled(answer)
    assert assert_native(answer) == 42


def test_native_results_are_plain_ints():
    add = compiled(make_add(), 1, 2)
    result = assert_native(add, 2**62, 5)
    assert type(result) is int


# --- Fallback ------------------------------------------------------------


class MyInt(int):
    def __add__(self, other):
        return "custom add"


class Opaque:
    def __radd__(self, other):
        return ("radd", other)


class Convertible:
    """Defines ``__int__`` and ``__index__``; the guards must not use them."""

    def __int__(self):
        raise AssertionError("__int__ was called")

    def __index__(self):
        raise AssertionError("__index__ was called")

    def __add__(self, other):
        return ("add", other)

    def __radd__(self, other):
        return ("radd", other)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((True, True), 2),
        ((True, 1), 2),
        ((MyInt(1), 2), "custom add"),
        ((1.5, 2), 3.5),
        ((1, 2.5), 3.5),
        ((1j, 2), 2 + 1j),
        (("a", "b"), "ab"),
        (([1], [2]), [1, 2]),
        (((1,), (2,)), (1, 2)),
        ((1, Opaque()), ("radd", 1)),
        ((1, Convertible()), ("radd", 1)),
        ((Convertible(), 1), ("add", 1)),
        ((2**63, 0), 2**63),
        ((I64_MIN - 1, 0), I64_MIN - 1),
        ((2**200, -(2**200)), 0),
    ],
)
def test_guard_failures_fall_back_to_python(args, expected):
    add = compiled(make_add(), 1, 2)
    result, fast_calls = assert_python(add, *args)
    assert result == expected
    assert type(result) is type(expected)
    # The native tier was tried and its guards said no.
    assert fast_calls == 1
    assert state_of(add).status is _Status.COMPILED


@pytest.mark.parametrize("args", [(None, 1), (1, None), (1, "x"), (object(), 2)])
def test_guard_failures_that_raise_in_python(args):
    add = compiled(make_add(), 1, 2)
    with pytest.raises(TypeError) as expected:
        add.__wrapped__(*args)
    with recording() as calls:
        with pytest.raises(TypeError) as actual:
            add(*args)
    assert str(actual.value) == str(expected.value)
    assert actual.value.__context__ is None
    assert (calls[add.__wrapped__.__code__], calls[FAST_CALL]) == (1, 1)
    assert calls[LEGACY_CALL] == 0
    assert state_of(add).status is _Status.COMPILED


class RecordingMeta(type):
    """Records any attribute lookup on its classes, then refuses it."""

    lookups = []

    def __getattribute__(cls, name):
        RecordingMeta.lookups.append(name)
        raise RuntimeError(f"metaclass hook ran for {name!r}")


class Hostile(metaclass=RecordingMeta):
    pass


class HostileInt(int, metaclass=RecordingMeta):
    pass


class Recorder:
    """Records any attempt to inspect, convert or compare it."""

    calls = []

    def __getattribute__(self, name):
        Recorder.calls.append(name)
        return object.__getattribute__(self, name)

    def _record(self, *args):
        Recorder.calls.append("dunder")
        return NotImplemented

    __index__ = __int__ = __float__ = __eq__ = __lt__ = __le__ = _record
    __gt__ = __ge__ = __hash__ = __bool__ = __len__ = __iter__ = _record

    def __repr__(self):
        Recorder.calls.append("__repr__")
        return "Recorder()"

    def __radd__(self, other):
        return "user code ran in the fallback only"


def test_guard_misses_run_no_user_code():
    @snekit
    def second(a, b):
        return b

    compiled(second, 1, 2)
    hostile, hostile_int, recorder = Hostile(), HostileInt(5), Recorder()
    RecordingMeta.lookups.clear()
    Recorder.calls.clear()
    # ``second`` returns ``b`` without touching it, so any recorded access
    # could only come from the guards.
    for args in [(hostile, 1), (1, hostile), (hostile_int, 1), (recorder, 2)]:
        result, python_calls, fast_calls = traced(second, *args)
        assert result is args[1]
        assert (python_calls, fast_calls) == (1, 1)
    assert RecordingMeta.lookups == []
    assert Recorder.calls == []
    assert state_of(second).status is _Status.COMPILED


def test_every_guard_miss_kind_keeps_the_specialization():
    @snekit
    def f(a, b):
        return a * b - a * b + 1

    compiled(f, 1, 2)
    specialization = state_of(f).specialization
    misses = [
        (True, 2),
        (MyInt(1), 2),
        (1.5, 2),
        (I64_MAX + 1, 1),
        (I64_MIN - 1, 1),
        (2**32, 2**32),  # Intermediate overflow; the final result fits.
    ]
    for args in misses:
        result, fast_calls = assert_python(f, *args)
        assert result == f.__wrapped__(*args)
        assert fast_calls == 1
        assert state_of(f).specialization is specialization
    assert assert_native(f, 3, 4) == 1
    assert f.call_count == _WARMUP_CALLS + 1 + len(misses) + 1
    assert state_of(f).native_error is None
    assert state_of(f).compile_attempts == 1


@pytest.mark.parametrize(
    ("source", "args", "expected"),
    [
        ("a + b", (I64_MAX, 1), 2**63),
        ("a - b", (I64_MIN, 1), I64_MIN - 1),
        ("a * b", (2**32, 2**32), 2**64),
        ("a * b", (I64_MIN, -1), 2**63),
        # Intermediate overflow even though the final result fits in i64.
        ("a * b - a * b + 1", (2**32, 2**32), 1),
        ("(a + b) - b", (I64_MAX, 1), I64_MAX),
    ],
)
def test_overflow_falls_back_to_exact_python_result(source, args, expected):
    namespace = {}
    exec(f"def f(a, b):\n    return {source}\n", namespace)
    f = compiled(snekit(namespace["f"]), 1, 2)
    result, native_calls = assert_python(f, *args)
    assert (result, native_calls) == (expected, 1)
    assert state_of(f).status is _Status.COMPILED


def test_guard_failure_does_not_disable_the_specialization():
    add = compiled(make_add(), 1, 2)
    specialization = state_of(add).specialization
    assert assert_python(add, True, 1) == (2, 1)
    assert assert_python(add, I64_MAX, 1) == (2**63, 1)
    assert state_of(add).specialization is specialization
    assert assert_native(add, 3, 4) == 7


def test_keyword_arguments_use_python():
    add = compiled(make_add(), 1, 2)
    assert assert_python(add, 1, b=2) == (3, 0)
    assert assert_python(add, a=1, b=2) == (3, 0)
    assert state_of(add).status is _Status.COMPILED


def test_default_arguments_use_python():
    @snekit
    def f(a, b=5):
        return a + b

    compiled(f, 1, 2)
    assert assert_python(f, 1) == (6, 0)
    f.__wrapped__.__defaults__ = (100,)
    assert assert_python(f, 1) == (101, 0)
    assert assert_native(f, 1, 2) == 3


@pytest.mark.parametrize("args", [(), (1,), (1, 2, 3)])
def test_wrong_arity_preserves_cpython_type_error(args):
    add = compiled(make_add(), 1, 2)
    with pytest.raises(TypeError) as expected:
        add.__wrapped__(*args)
    with recording() as calls:
        with pytest.raises(TypeError) as actual:
            add(*args)
    assert str(actual.value) == str(expected.value)
    assert calls[FAST_CALL] == 0
    assert state_of(add).status is _Status.COMPILED


def test_unexpected_keyword_preserves_cpython_type_error():
    add = compiled(make_add(), 1, 2)
    with pytest.raises(TypeError) as expected:
        add.__wrapped__(1, 2, c=3)
    with pytest.raises(TypeError) as actual:
        add(1, 2, c=3)
    assert str(actual.value) == str(expected.value)


def test_exceptions_after_a_guard_failure_propagate_unchanged():
    error = ValueError("from user code")

    class Raises:
        def __radd__(self, other):
            raise error

    add = compiled(make_add(), 1, 2)
    with recording() as calls:
        with pytest.raises(ValueError) as excinfo:
            add(1, Raises())
    assert excinfo.value is error
    # Not chained to anything from the native attempt.
    assert excinfo.value.__context__ is None
    assert excinfo.value.__cause__ is None
    assert calls[add.__wrapped__.__code__] == 1


def test_python_type_errors_are_not_chained():
    add = compiled(make_add(), 1, 2)
    with pytest.raises(TypeError) as excinfo:
        add(1, "x")
    assert excinfo.value.__context__ is None


def test_user_exceptions_keep_their_normal_context():
    error = ValueError("from user code")

    class Raises:
        def __radd__(self, other):
            raise error

    add = compiled(make_add(), 1, 2)
    # Raised while another exception is being handled, the user's error
    # must get exactly the context it gets without the decorator.
    for f in (add.__wrapped__, add):
        outer = KeyError("outer")
        try:
            raise outer
        except KeyError:
            with pytest.raises(ValueError) as excinfo:
                f(1, Raises())
        assert excinfo.value is error
        assert excinfo.value.__context__ is outer
        error.__context__ = None


def test_guard_failures_never_leak():
    add = compiled(make_add(), 1, 2)
    for args in [(True, 2), (I64_MAX, I64_MAX), (1.0, 2), (2**64, 1)]:
        try:
            add(*args)
        except NativeGuardFailure:  # pragma: no cover - the failure mode
            pytest.fail("NativeGuardFailure leaked through @snekit")


# --- Unsupported functions -----------------------------------------------

GLOBAL_OFFSET = 10


def _closure():
    offset = 3

    def closure(a):
        return a + offset

    return closure


def _branch(a):
    if a > 0:
        return a
    return -a


def _call(a):
    return abs(a) + 1


def _global(a):
    return a + GLOBAL_OFFSET


def _local(a):
    b = a * 2
    return b + 1


def _divide(a, b):
    return a // b


def _varargs(*args):
    return len(args)


def _generator(n):
    yield from range(n)


async def _coroutine(a, b):
    return a + b


class _CallableObject:
    def __call__(self, a, b):
        return a + b


UNSUPPORTED = [
    (_closure(), (4,), RejectionReason.CLOSURE),
    (_branch, (-4,), RejectionReason.CONTROL_FLOW),
    (_call, (-4,), RejectionReason.UNSUPPORTED_OPCODE),
    (_global, (4,), RejectionReason.UNSUPPORTED_OPCODE),
    (_local, (4,), RejectionReason.UNSUPPORTED_OPCODE),
    (_divide, (9, 2), RejectionReason.UNSUPPORTED_OPERATOR),
    (_varargs, (1, 2), RejectionReason.UNSUPPORTED_SIGNATURE),
    (abs, (-3,), RejectionReason.NOT_A_FUNCTION),
    (_CallableObject(), (1, 2), RejectionReason.NOT_A_FUNCTION),
    (functools.partial(_divide, 9), (2,), RejectionReason.NOT_A_FUNCTION),
]


@pytest.mark.parametrize(("func", "args", "reason"), UNSUPPORTED)
def test_unsupported_functions_keep_working(func, args, reason, compile_spy):
    f = snekit(func)
    expected = func(*args)
    for _ in range(_WARMUP_CALLS + 20):
        assert f(*args) == expected
    state = state_of(f)
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.REJECTED
    assert state.rejection.reason is reason
    assert state.compile_attempts == 1
    assert state.specialization is None
    assert compile_spy == []
    assert f.call_count == _WARMUP_CALLS + 20


def test_rejection_is_not_retried(monkeypatch):
    analyses = []
    real = _frontend.analyze

    def spy(func):
        analyses.append(func)
        return real(func)

    monkeypatch.setattr(_frontend, "analyze", spy)
    f = snekit(_branch)
    for _ in range(_WARMUP_CALLS * 5):
        f(3)
    assert analyses == [_branch]


def test_generators_and_coroutines_keep_working():
    gen = snekit(_generator)
    coro = snekit(_coroutine)
    for _ in range(_WARMUP_CALLS + 2):
        assert list(gen(3)) == [0, 1, 2]
        assert asyncio.run(coro(1, 2)) == 3
    assert state_of(gen).rejection.reason is RejectionReason.GENERATOR
    assert state_of(coro).rejection.reason is RejectionReason.COROUTINE


def test_backend_unsupported_program_disables():
    @snekit
    def huge(a):
        return a + 10**30  # Folded into one constant outside i64.

    for _ in range(_WARMUP_CALLS + 3):
        assert huge(1) == 10**30 + 1
    state = state_of(huge)
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.UNSUPPORTED
    assert isinstance(state.compile_error, _compiler.UnsupportedProgramError)
    assert state.rejection is None
    assert state.compile_attempts == 1


@pytest.mark.parametrize("target", ["analyze", "compile_i64"])
def test_unexpected_compiler_errors_disable_and_fall_back(monkeypatch, target):
    error = RuntimeError("compiler bug")
    attempts = []

    def broken(*args, **kwargs):
        attempts.append(args)
        raise error

    module = _frontend if target == "analyze" else _compiler
    monkeypatch.setattr(module, target, broken)
    add = make_add()
    warm(add, 1, 2)
    assert assert_python(add, 1, 2) == (3, 0)
    for _ in range(10):
        assert add(2, 3) == 5
    state = state_of(add)
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.COMPILER_ERROR
    assert state.compile_error is error
    assert len(attempts) == 1


def test_base_exceptions_during_compilation_propagate(monkeypatch):
    def interrupted(func):
        raise KeyboardInterrupt

    monkeypatch.setattr(_frontend, "analyze", interrupted)
    add = make_add()
    warm(add, 1, 2)
    with recording() as calls:
        with pytest.raises(KeyboardInterrupt):
            add(1, 2)
    assert calls[add.__wrapped__.__code__] == 0
    state = state_of(add)
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.INTERRUPTED
    assert state.compile_error is None
    # Never retried; later calls simply run Python.
    monkeypatch.undo()
    assert assert_python(add, 1, 2) == (3, 0)
    assert state.compile_attempts == 1


class BrokenNative:
    """Stands in for a ``CompiledI64`` whose fast call raises."""

    arg_count = 2

    def __init__(self, error):
        self.error = error
        self.calls = 0

    def fast_call(self, args):
        self.calls += 1
        raise self.error


def test_unexpected_native_errors_disable_and_fall_back():
    add = compiled(make_add(), 1, 2)
    state = state_of(add)
    error = RuntimeError("native bug")
    broken = BrokenNative(error)
    state.specialization = _Specialization(state.compiled_code, broken)
    assert assert_python(add, 1, 2) == (3, 0)
    assert broken.calls == 1
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.NATIVE_ERROR
    assert state.native_error is error
    assert state.specialization is None
    assert assert_python(add, 1, 2) == (3, 0)
    assert broken.calls == 1
    assert add.call_count == _WARMUP_CALLS + 1 + 2


def test_real_native_errors_disable_and_fall_back():
    # The real ``try_call`` raises ``TypeError`` for anything but an exact
    # tuple: a caller bug, never a guard miss.
    add = compiled(make_add(), 1, 2)
    state = state_of(add)
    real = state.specialization
    real.fast_call = lambda args: state.compiled._function.try_call(list(args))
    assert add(1, 2) == 3
    assert state.disabled_reason is _DisabledReason.NATIVE_ERROR
    assert isinstance(state.native_error, TypeError)


def test_native_errors_do_not_chain_user_exceptions():
    error = ValueError("from user code")

    class Raises:
        def __radd__(self, other):
            raise error

    add = compiled(make_add(), 1, 2)
    state = state_of(add)
    state.specialization = _Specialization(
        state.compiled_code, BrokenNative(RuntimeError("native bug"))
    )
    with pytest.raises(ValueError) as excinfo:
        add(1, Raises())
    assert excinfo.value is error
    assert excinfo.value.__context__ is None
    assert state.disabled_reason is _DisabledReason.NATIVE_ERROR


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_base_exceptions_from_native_code_propagate(error):
    add = compiled(make_add(), 1, 2)
    state = state_of(add)
    broken = BrokenNative(error())
    specialization = _Specialization(state.compiled_code, broken)
    state.specialization = specialization
    with recording() as calls:
        with pytest.raises(error):
            add(1, 2)
    # Not converted into a fallback, and not a reason to disable.
    assert calls[add.__wrapped__.__code__] == 0
    assert state.status is _Status.COMPILED
    assert state.specialization is specialization
    assert state.native_error is None


def test_warmed_calls_never_enter_compiled_i64(monkeypatch):
    def forbidden(self, *args):
        raise AssertionError("CompiledI64.__call__ was called")

    add = compiled(make_add(), 1, 2)
    monkeypatch.setattr(CompiledI64, "__call__", forbidden)
    assert assert_native(add, 3, 4) == 7
    assert assert_python(add, 1.5, 2) == (3.5, 1)
    assert assert_python(add, I64_MAX, 1) == (2**63, 1)
    assert state_of(add).status is _Status.COMPILED


def test_the_specialization_caches_the_bound_fast_call():
    add = compiled(make_add(), 1, 2)
    state = state_of(add)
    spec = state.specialization
    fast_call = spec.fast_call
    assert type(fast_call) is BuiltinMethodType
    assert fast_call.__self__ is state.compiled._function
    assert fast_call.__name__ == "try_call"
    assert spec.arg_count == state.compiled.arg_count == 2
    # Every call uses this very object; nothing is looked up per call.
    seen = []

    def profiler(frame, event, arg):
        if event == "c_call" and native_method(arg) == FAST_CALL:
            seen.append(arg)

    sys.setprofile(profiler)
    try:
        for i in range(5):
            add(i, i)
        add(0.5, 1)
    finally:
        sys.setprofile(None)
    assert len(seen) == 6
    assert all(arg is fast_call for arg in seen)
    assert state.specialization is spec


@pytest.mark.parametrize("phase", ["cold", "compiled", "disabled"])
def test_original_exceptions_propagate_unchanged(phase):
    error = ArithmeticError("boom")

    class Raises:
        def __radd__(self, other):
            raise error

    if phase == "disabled":

        @snekit
        def f(a, b):
            if a:
                pass
            return a + b

        for _ in range(_WARMUP_CALLS + 1):
            f(1, 2)
    else:
        f = make_add()
        if phase == "compiled":
            compiled(f, 1, 2)
    assert state_of(f).status.value == phase
    with pytest.raises(ArithmeticError) as excinfo:
        f(1, Raises())
    assert excinfo.value is error
    assert excinfo.value.__context__ is None


# --- Invalidation --------------------------------------------------------


def test_replacing_code_prevents_stale_native_execution():
    add = compiled(make_add(), 1, 2)
    assert assert_native(add, 6, 7) == 13

    def multiply(a, b):
        return a * b

    add.__wrapped__.__code__ = multiply.__code__
    result, python_calls, native_calls = traced(add, 6, 7)
    assert (result, python_calls, native_calls) == (42, 1, 0)

    state = state_of(add)
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.CODE_CHANGED
    assert state.specialization is None
    for _ in range(20):
        assert traced(add, 6, 7) == (42, 1, 0)
    assert state.compile_attempts == 1


def test_replacing_code_before_compilation_compiles_the_new_code():
    add = make_add()
    warm(add, 1, 2)

    def subtract(a, b):
        return a - b

    add.__wrapped__.__code__ = subtract.__code__
    assert assert_native(add, 10, 3) == 7
    assert state_of(add).compiled_code is subtract.__code__


def test_code_replaced_during_compilation_is_not_used(monkeypatch):
    def multiply(a, b):
        return a * b

    add = make_add()
    real = _compiler.compile_i64

    def racing(*args, **kwargs):
        result = real(*args, **kwargs)
        add.__wrapped__.__code__ = multiply.__code__
        return result

    monkeypatch.setattr(_compiler, "compile_i64", racing)
    warm(add, 1, 2)
    with recording() as calls:
        assert add(6, 7) == 42
    assert calls[multiply.__code__] == 1
    assert calls[FAST_CALL] == 0
    assert state_of(add).disabled_reason is _DisabledReason.CODE_CHANGED


# --- Metadata and methods ------------------------------------------------


def test_metadata_is_preserved_after_compilation():
    def original(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    wrapped = compiled(snekit(original), 1, 2)
    assert wrapped.__name__ == "original"
    assert wrapped.__qualname__ == original.__qualname__
    assert wrapped.__module__ == original.__module__
    assert wrapped.__doc__ == "Add two numbers."
    assert wrapped.__annotations__ == {"a": int, "b": int, "return": int}
    assert wrapped.__wrapped__ is original


class Scaled:
    def __init__(self, factor):
        self.factor = factor

    @snekit
    def double(self, x):
        return x * 2

    @snekit
    def scale(self, x):
        return self.factor * x


def test_methods_bind_and_fall_back_on_self():
    instances = [Scaled(i) for i in range(3)]
    for _ in range(_WARMUP_CALLS + 5):
        for i, instance in enumerate(instances):
            assert instance.double(i) == 2 * i
    # The method itself compiles (``self`` is never read) ...
    state = state_of(Scaled.double)
    assert state.status is _Status.COMPILED
    assert state.specialization.arg_count == 2
    # ... but a bound call passes ``self``, which is not an int.
    assert assert_python(instances[0].double, 21) == (42, 1)
    # Called through the class with an int in ``self``'s place, it can run
    # natively, exactly like the original function would compute.
    assert assert_native(Scaled.double, 0, 21) == 42


def test_methods_reading_attributes_stay_in_python():
    instances = [Scaled(i) for i in range(3)]
    for _ in range(_WARMUP_CALLS + 5):
        for i, instance in enumerate(instances):
            assert instance.scale(10) == 10 * i
    assert state_of(Scaled.scale).status is _Status.DISABLED
    assert Scaled.scale.call_count == 3 * (_WARMUP_CALLS + 5)


def test_rejects_non_callable():
    with pytest.raises(TypeError):
        snekit(None)


# --- Threading -----------------------------------------------------------


def test_concurrent_warmup_compiles_at_most_once(monkeypatch):
    attempts = []
    real = _compiler.compile_i64

    def slow(*args, **kwargs):
        attempts.append(threading.get_ident())
        time.sleep(0.05)  # Widen the race window.
        return real(*args, **kwargs)

    monkeypatch.setattr(_compiler, "compile_i64", slow)
    expression = make_expression()
    threads = 8
    barrier = threading.Barrier(threads)
    errors = []

    def worker(seed):
        barrier.wait()
        for i in range(200):
            a, b = seed * 1000 + i, -i
            args = (a, b) if i % 7 else (float(a), b)
            if expression(*args) != (a + b) * 2:
                errors.append(args)
            spec = expression._jit_state.specialization
            if spec is not None and spec.native.arg_count != 2:
                errors.append("partial specialization")

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    assert errors == []
    assert len(attempts) == 1
    state = state_of(expression)
    assert state.compile_attempts == 1
    assert state.status is _Status.COMPILED
    assert assert_native(expression, 1, 2) == 6


# --- Imports and privacy -------------------------------------------------

IMPORT_SCRIPT = """
import json, sys
import speedysnek

PRIVATE = ["speedysnek._frontend", "speedysnek._compiler",
           "speedysnek._interpreter", "speedysnek._native"]

def loaded():
    return sorted(name for name in PRIVATE if name in sys.modules)

report = {"import": loaded()}

@speedysnek.snekit
def add(a, b):
    return a + b

@speedysnek.snekit
def divide(a, b):
    return a / b

for _ in range(WARMUP):
    add(1, 2)
    divide(1, 2)
report["warm"] = loaded()
divide(1, 2)
report["rejected"] = loaded()
add(1, 2)
report["compiled"] = loaded()
report["status"] = [add._jit_state.status.value, divide._jit_state.status.value]
print(json.dumps(report))
"""


def test_compilation_modules_are_imported_lazily():
    script = IMPORT_SCRIPT.replace("WARMUP", str(_WARMUP_CALLS))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    assert report["import"] == []
    assert report["warm"] == []
    # A rejected function never loads the native backend.
    assert report["rejected"] == ["speedysnek._frontend"]
    assert "speedysnek._compiler" in report["compiled"]
    assert "speedysnek._native" in report["compiled"]
    assert report["status"] == ["compiled", "disabled"]


def test_public_api_is_unchanged_after_compilation():
    compiled(make_add(), 1, 2)
    assert speedysnek.__all__ == ["snekit", "jit"]
    public = {name for name in dir(speedysnek) if not name.startswith("_")}
    assert public == {"snekit", "jit"}
    for name in (
        "CompiledI64",
        "NativeGuardFailure",
        "UnsupportedProgramError",
        "analyze",
        "compile_i64",
        "_Status",
        "_JitState",
    ):
        assert not hasattr(speedysnek, name)


def test_module_doctests():
    failures, tests = doctest.testmod(_decorator)
    assert tests > 0
    assert failures == 0
