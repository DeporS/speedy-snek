"""Tests for ``@snekit``: warmup, compilation, guards and fallback.

Execution paths are proven, never inferred from an equal result. The
native wrapper runs the compiled function's fast path from Rust, which
``sys.setprofile`` cannot see, so native calls are counted with the
compiled function's ``I64Function.try_calls`` counter, and Python frames
with ``sys.setprofile``. A warmed native call makes exactly one native
fast call and enters no frame of the original function and no frame of
``CompiledI64.__call__`` (the diagnostic path, which the decorator never
uses). A Python call runs exactly one frame of the original function.
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
FAST_CALL = "native fast call"
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


def native_calls(f):
    """How many native fast calls ``f`` has made so far.

    The wrapper enters its compiled function's fast path from Rust, which
    emits no ``sys.setprofile`` event; the function counts the entries.
    """
    compiled = state_of(f).compiled
    return 0 if compiled is None else compiled.function.try_calls


@contextlib.contextmanager
def recording(f=None):
    """Count, in this thread, the Python frames entered (by code object),
    the calls to native ``I64Function`` methods made from Python (by name,
    such as ``RAW_CALL``) and, if ``f`` is given, its native fast calls
    (by ``FAST_CALL``)."""
    calls = collections.Counter()
    before = 0 if f is None else native_calls(f)

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
        if f is not None:
            calls[FAST_CALL] += native_calls(f) - before


def traced(f, *args, **kwargs):
    """Call ``f`` and return ``(result, python_calls, fast_calls)``.

    Also checks that the decorator never takes the diagnostic path.
    """
    original = f.__wrapped__.__code__
    with recording(f) as calls:
        result = f(*args, **kwargs)
    assert calls[LEGACY_CALL] == 0
    assert calls[RAW_CALL] == 0
    # The wrapper never calls a Python-level fast call either.
    assert calls["I64Function.try_call"] == 0
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


# --- Local variables -----------------------------------------------------


def calculate_score(wins, draws, penalties):
    points = wins * 3 + draws
    deduction = penalties * 2
    result = points - deduction
    return result


def reuse(a, b):
    total = a + b
    return total * total


def reassign(a, b):
    value = a + b
    value = value * 3
    return value - a


def overwrite_argument(a, b):
    a = a + 1
    return a * b


def copy_value(a, b):
    x = a
    y = x
    x = b
    return y - x


def intermediate_overflow(a):
    value = a + 1
    value = 0
    return value


def unused_product(a, b):
    product = a * b  # noqa: F841 -- computed, never read.
    return a - b


def helper_names(stack, args):
    # Names that resemble the compiler's own internals.
    locals = stack * 2
    LOAD_LOCAL = locals + args
    result = LOAD_LOCAL - stack
    stack = result * result
    return stack - LOAD_LOCAL


LOCAL_EXAMPLES = [
    (calculate_score, (5, 2, 3)),
    (calculate_score, (-4, 10**6, -(2**40))),
    (reuse, (3, 4)),
    (reuse, (-(2**31), 2**31 - 1)),
    (reassign, (3, 4)),
    (reassign, (-7, I64_MAX // 3 - 10)),
    (overwrite_argument, (3, 4)),
    (overwrite_argument, (I64_MAX - 1, 1)),
    (overwrite_argument, (-1, I64_MIN)),
    (copy_value, (10, 3)),
    (copy_value, (I64_MIN, 0)),
    (intermediate_overflow, (I64_MAX - 1,)),
    (unused_product, (2**31, -(2**31))),
    (helper_names, (6, 7)),
]


@pytest.mark.parametrize(
    ("func", "args"), LOCAL_EXAMPLES, ids=lambda v: getattr(v, "__name__", "")
)
def test_locals_run_natively(func, args):
    f = compiled(snekit(func), *args)
    expected = func(*args)
    assert assert_native(f, *args) == expected
    assert execute(analyze(func).instructions, args) == expected
    assert state_of(f).compiled.instructions == analyze(func).instructions


def test_copy_value_returns_the_original_a_minus_b():
    f = compiled(snekit(copy_value), 1, 2)
    for a, b in [(10, 3), (-5, 8), (0, 0), (I64_MAX, 0)]:
        assert assert_native(f, a, b) == a - b


@pytest.mark.parametrize(
    ("func", "args"),
    [
        # The overwritten ``a + 1`` overflows; Python returns 0.
        (intermediate_overflow, (I64_MAX,)),
        # The never-read ``a * b`` overflows; Python returns a - b.
        (unused_product, (2**32, 2**32)),
        (unused_product, (I64_MIN, -1)),
        # ``points`` overflows although the final result fits.
        (calculate_score, (I64_MAX // 3 + 1, 0, I64_MAX // 2)),
        # The reassigned argument overflows.
        (overwrite_argument, (I64_MAX, 0)),
        # The final multiply overflows.
        (reuse, (2**32, 0)),
    ],
    ids=lambda v: getattr(v, "__name__", ""),
)
def test_overflow_in_locals_falls_back_to_python(func, args):
    f = compiled(snekit(func), *(1,) * len(args))
    specialization = state_of(f).specialization
    result, fast_calls = assert_python(f, *args)
    assert result == func(*args)
    # The native code ran, and its overflow guard sent the call to Python.
    assert fast_calls == 1
    assert state_of(f).specialization is specialization
    assert state_of(f).status is _Status.COMPILED
    assert assert_native(f, *(2,) * len(args)) == func(*(2,) * len(args))


def test_unread_reassigned_argument_is_still_guarded():
    @snekit
    def f(a, b):
        a = 1
        return a + b

    compiled(f, 1, 2)
    assert assert_native(f, 100, 2) == 3
    # CPython never reads the old ``a``, so a non-int works; natively every
    # argument must be an exact i64, so this falls back.
    assert assert_python(f, "x", 2) == (3, 1)
    assert assert_python(f, 2**64, 2) == (3, 1)


def test_read_before_assignment_falls_back_to_python():
    @snekit
    def f(a):
        y = x + a  # noqa: F821 -- ``x`` is the local assigned below.
        x = 1  # noqa: F841
        return y

    for _ in range(_WARMUP_CALLS + 5):
        with pytest.raises(UnboundLocalError):
            f(1)
    state = state_of(f)
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.REJECTED
    assert state.rejection.reason is RejectionReason.UNSUPPORTED_VARIABLE
    assert state.compiled is None
    assert f.call_count == _WARMUP_CALLS + 5
    with recording() as calls:
        with pytest.raises(UnboundLocalError):
            f(1)
    assert calls[f.__wrapped__.__code__] == 1


def test_readme_locals_example():
    @snekit
    def score(wins, draws, penalties):
        points = wins * 3 + draws
        deduction = penalties * 2
        result = points - deduction
        return result

    for _ in range(100):
        assert score(10, 4, 3) == 28
    assert state_of(score).status is _Status.COMPILED
    assert assert_native(score, 10, 4, 3) == 28
    assert score.call_count == 101


def test_replacing_code_of_a_function_with_locals():
    @snekit
    def f(a, b):
        value = a + b
        value = value * 3
        return value - a

    compiled(f, 1, 2)
    assert assert_native(f, 3, 4) == 18
    f.__wrapped__.__code__ = overwrite_argument.__code__
    assert assert_python(f, 3, 4) == (16, 0)
    assert state_of(f).disabled_reason is _DisabledReason.CODE_CHANGED


# --- Augmented assignment ------------------------------------------------


def aug_calculate_score(wins, draws, penalties):
    points = wins * 3
    points += draws
    points -= penalties * 2
    return points


def update_argument(a, b):
    a += b
    a *= 3
    a -= b
    return a


def copy_before_update(a, b):
    x = a
    y = x
    x += b
    return y - x


def update_from_itself(a, b):
    x = a
    x -= x * b
    return x


def overwritten_update(a):
    a += 1
    a = 0
    return a


AUGMENTED = [
    aug_calculate_score,
    update_argument,
    copy_before_update,
    update_from_itself,
    overwritten_update,
]


@pytest.mark.parametrize("func", AUGMENTED, ids=lambda f: f.__name__)
def test_augmented_assignment_runs_natively(func):
    arity = func.__code__.co_argcount
    f = compiled(snekit(func), *(1,) * arity)
    values = [0, 1, -1, 7, -12, 2**20, -(2**31)]
    for i in range(len(values)):
        args = tuple(values[(i + k) % len(values)] for k in range(arity))
        expected = func(*args)
        assert assert_native(f, *args) == expected
        assert execute(analyze(func).instructions, args) == expected


def test_augmented_examples():
    score = compiled(snekit(aug_calculate_score), 1, 1, 1)
    assert assert_native(score, 10, 4, 3) == 28
    update = compiled(snekit(update_argument), 1, 1)
    assert assert_native(update, 5, 2) == (5 + 2) * 3 - 2
    # ``x = x - x * b``: the right-hand side reads the old ``x``.
    itself = compiled(snekit(update_from_itself), 1, 1)
    assert assert_native(itself, 5, 3) == 5 - 15


def test_augmented_assignment_does_not_change_a_copy():
    f = compiled(snekit(copy_before_update), 1, 2)
    # ``y`` keeps the original ``a``, so the result is always ``-b``.
    for a, b in [(10, 3), (-5, 8), (0, 0), (I64_MAX - 5, 5), (I64_MIN, 1)]:
        assert assert_native(f, a, b) == -b


@pytest.mark.parametrize(
    ("func", "args", "expected"),
    [
        (update_argument, (I64_MAX // 3 - 7, 7), I64_MAX // 3 * 3 - 7),
        (update_argument, (-(I64_MAX // 3) + 1, -1), -(I64_MAX // 3) * 3 + 1),
        (aug_calculate_score, (0, I64_MAX, 0), I64_MAX),
        (aug_calculate_score, (0, I64_MIN, 0), I64_MIN),
        (copy_before_update, (I64_MIN, I64_MAX), -I64_MAX),
        (update_from_itself, (I64_MAX, 0), I64_MAX),
        (overwritten_update, (I64_MAX - 1,), 0),
        (overwritten_update, (I64_MIN,), 0),
    ],
    ids=lambda v: getattr(v, "__name__", ""),
)
def test_augmented_i64_boundaries(func, args, expected):
    f = compiled(snekit(func), *(1,) * len(args))
    assert func(*args) == expected
    assert assert_native(f, *args) == expected


@pytest.mark.parametrize(
    ("func", "args"),
    [
        # The overwritten ``a += 1`` overflows; Python returns 0.
        (overwritten_update, (I64_MAX,)),
        # ``+=``, ``*=`` and ``-=`` each overflow in turn.
        (update_argument, (I64_MAX, 1)),
        (update_argument, (I64_MAX // 3, 1)),
        (update_argument, (-(I64_MAX // 3) - 3, 3)),
        (aug_calculate_score, (0, I64_MAX, -1)),
        (copy_before_update, (1, I64_MAX)),
        (copy_before_update, (I64_MIN + 1, I64_MIN)),
        (update_from_itself, (2**32, 2**32)),
    ],
    ids=lambda v: getattr(v, "__name__", ""),
)
def test_augmented_overflow_falls_back_to_python(func, args):
    f = compiled(snekit(func), *(1,) * len(args))
    specialization = state_of(f).specialization
    result, fast_calls = assert_python(f, *args)
    assert result == func(*args)
    assert fast_calls == 1
    assert state_of(f).specialization is specialization
    assert state_of(f).status is _Status.COMPILED
    assert assert_native(f, *(2,) * len(args)) == func(*(2,) * len(args))


def test_overwritten_augmented_overflow_returns_zero():
    f = compiled(snekit(overwritten_update), 1)
    assert assert_python(f, I64_MAX) == (0, 1)
    assert assert_native(f, 5) == 0
    assert state_of(f).status is _Status.COMPILED


def test_augmented_read_before_assignment_falls_back_to_python():
    @snekit
    def unbound(a):
        x += a  # noqa: F821
        return x

    for _ in range(_WARMUP_CALLS + 5):
        with pytest.raises(UnboundLocalError):
            unbound(1)
    state = state_of(unbound)
    assert state.status is _Status.DISABLED
    assert state.rejection.reason is RejectionReason.UNSUPPORTED_VARIABLE
    with recording() as calls:
        with pytest.raises(UnboundLocalError):
            unbound(1)
    assert calls[unbound.__wrapped__.__code__] == 1


class InPlace:
    """Updates itself in place and records each in-place operator."""

    def __init__(self):
        self.log = []

    def _update(self, name, other):
        self.log.append((name, other))
        return self

    def __iadd__(self, other):
        return self._update("iadd", other)

    def __isub__(self, other):
        return self._update("isub", other)

    def __imul__(self, other):
        return self._update("imul", other)

    def __add__(self, other):
        raise AssertionError("augmented assignment must use __iadd__")

    __sub__ = __mul__ = __radd__ = __rsub__ = __rmul__ = __add__


class InPlaceInt(int):
    """An ``int`` subclass, so not an exact ``int``, with ``__iadd__``."""

    calls = 0

    def __iadd__(self, other):
        InPlaceInt.calls += 1
        return "iadd"


def test_fallback_keeps_in_place_operators():
    f = compiled(snekit(update_argument), 1, 2)
    target = InPlace()
    result, fast_calls = assert_python(f, target, 4)
    assert result is target
    # Each operator ran exactly once, in order, with the right operand.
    assert target.log == [("iadd", 4), ("imul", 3), ("isub", 4)]
    assert fast_calls == 1
    assert assert_native(f, 5, 2) == 19
    assert state_of(f).status is _Status.COMPILED


def test_fallback_extends_a_list_in_place_once():
    @snekit
    def extend(a, b):
        a += b
        return a

    compiled(extend, 1, 2)
    items = [1]
    result, fast_calls = assert_python(extend, items, [2, 3])
    # ``list.__iadd__`` mutates the list it returns; ``a + b`` would not.
    assert result is items
    assert items == [1, 2, 3]
    assert fast_calls == 1
    assert assert_native(extend, 4, 5) == 9


def test_fallback_uses_an_int_subclass_iadd():
    @snekit
    def add_to(a, b):
        a += b
        return a

    compiled(add_to, 1, 2)
    InPlaceInt.calls = 0
    assert assert_python(add_to, InPlaceInt(1), 2) == ("iadd", 1)
    assert InPlaceInt.calls == 1


def test_readme_augmented_example():
    @snekit
    def calculate_score(wins, draws, penalties):
        points = wins * 3
        points += draws
        points -= penalties * 2
        return points

    for _ in range(100):
        assert calculate_score(10, 4, 3) == 28
    assert state_of(calculate_score).status is _Status.COMPILED
    assert assert_native(calculate_score, 10, 4, 3) == 28


# --- Fallback ------------------------------------------------------------


# --- Conditionals --------------------------------------------------------


def branch_calculate_score(wins, penalties):
    score = wins * 3 - penalties
    if score < 0:
        score = 0
    else:
        score += 5
    return score


def choose(a, b):
    if a >= b:
        result = a - b
    else:
        result = b - a
    return result


def adjust(a, b):
    result = a
    if a < b:
        result += b
    return result * 2


def selected_branch(a, b):
    result = a
    if a < b:
        result += 1
    else:
        result *= b
    return result


def possibly_unbound(a, b):
    if a < b:
        result = a
    return result


def flag(a, b):
    r = 0
    if a < b:
        r = 1
    return r


BRANCH_EXAMPLES = [
    (branch_calculate_score, [(10, 3), (1, 5), (0, 0), (-4, -20)]),
    (choose, [(5, 3), (3, 5), (4, 4), (I64_MIN, -1)]),
    (adjust, [(1, 2), (2, 1), (3, 3), (-7, 7)]),
    (selected_branch, [(1, 2), (5, 3), (2, I64_MAX), (I64_MAX, 1)]),
]


@pytest.mark.parametrize(
    ("func", "cases"), BRANCH_EXAMPLES, ids=lambda v: getattr(v, "__name__", "")
)
def test_conditionals_run_natively(func, cases):
    f = compiled(snekit(func), *cases[0])
    for args in cases:
        expected = func(*args)
        result = assert_native(f, *args)
        assert type(result) is int and result == expected
        assert execute(analyze(func).instructions, args) == expected
    assert state_of(f).compiled.instructions == analyze(func).instructions


def test_overflow_only_on_the_branch_not_taken_stays_native():
    f = compiled(snekit(selected_branch), 1, 2)
    # ``result *= b`` would overflow, but a < b selects ``result += 1``.
    assert assert_native(f, 2, I64_MAX) == 3
    # ``result += 1`` would overflow, but a >= b selects ``result *= b``.
    assert assert_native(f, I64_MAX, 1) == I64_MAX


def test_overflow_on_the_selected_branch_falls_back_once():
    f = compiled(snekit(selected_branch), 1, 2)
    specialization = state_of(f).specialization
    result, fast_calls = assert_python(f, I64_MAX, 2)
    assert result == I64_MAX * 2
    assert fast_calls == 1
    # The specialization is kept and used again.
    assert state_of(f).specialization is specialization
    assert state_of(f).status is _Status.COMPILED
    assert assert_native(f, 3, 2) == 6
    assert assert_native(f, 2, 3) == 3


def test_comparison_results_are_never_bools():
    f = compiled(snekit(flag), 1, 2)
    assert type(assert_native(f, 1, 2)) is int
    assert type(assert_native(f, 2, 1)) is int

    @snekit
    def compare(a, b):
        return a < b

    for _ in range(_WARMUP_CALLS + 3):
        assert compare(1, 2) is True
    assert state_of(compare).status is _Status.DISABLED
    assert state_of(compare).rejection.reason is RejectionReason.UNSUPPORTED_CONDITION


def test_bool_arguments_fall_back_for_conditions():
    f = compiled(snekit(flag), 1, 2)
    for args in [(True, 2), (False, True), (0, True)]:
        result, fast_calls = assert_python(f, *args)
        assert result == flag(*args)
        assert fast_calls == 1
    assert state_of(f).status is _Status.COMPILED


def test_possibly_unbound_local_is_rejected_and_keeps_python_behavior():
    f = snekit(possibly_unbound)
    for _ in range(_WARMUP_CALLS + 3):
        assert f(1, 2) == 1
    state = state_of(f)
    assert state.status is _Status.DISABLED
    assert state.rejection.reason is RejectionReason.UNSUPPORTED_VARIABLE
    with pytest.raises(UnboundLocalError):
        f(2, 1)
    assert assert_python(f, 3, 4) == (3, 0)


class Ordered:
    """Records every comparison and arithmetic operation made on it."""

    def __init__(self, value, log):
        self.value = value
        self.log = log

    def __lt__(self, other):
        self.log.append(("lt", other))
        return self.value < other

    def __ge__(self, other):
        self.log.append(("ge", other))
        return self.value >= other

    def __rsub__(self, other):
        self.log.append(("rsub", other))
        return other - self.value

    def __sub__(self, other):
        self.log.append(("sub", other))
        return self.value - other

    def __iadd__(self, other):
        self.log.append(("iadd", other))
        return self

    def __mul__(self, other):
        self.log.append(("mul", other))
        return self.value * other


def test_custom_comparisons_run_once_in_python():
    f = compiled(snekit(choose), 5, 3)
    log = []
    result, fast_calls = assert_python(f, Ordered(10, log), 4)
    assert result == 6
    assert log == [("ge", 4), ("sub", 4)]
    assert fast_calls == 1

    g = compiled(snekit(adjust), 1, 2)
    log = []
    value = Ordered(1, log)
    result, fast_calls = assert_python(g, value, 4)
    assert result == 2
    # Exactly one comparison, one in-place add and one multiply.
    assert log == [("lt", 4), ("iadd", 4), ("mul", 2)]
    assert assert_native(g, 1, 4) == 10


def test_comparison_with_user_visible_side_effect_runs_once():
    calls = []

    class Loud(int):
        def __lt__(self, other):
            calls.append(other)
            return True

    f = compiled(snekit(flag), 1, 2)
    result, fast_calls = assert_python(f, Loud(5), 1)
    assert result == 1
    assert calls == [1]
    assert fast_calls == 1


def test_readme_conditional_example():
    @snekit
    def calculate_score(wins, penalties):
        score = wins * 3 - penalties
        if score < 0:
            score = 0
        else:
            score += 5
        return score

    results = [calculate_score(10, 3) for _ in range(_WARMUP_CALLS + 2)]
    assert results[-1] == 32
    assert state_of(calculate_score).status is _Status.COMPILED
    assert assert_native(calculate_score, 1, 5) == 0
    assert calculate_score(2**62, 0) == 3 * 2**62 + 5


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
    with recording(add) as calls:
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
    with recording(add) as calls:
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
    # ``or`` compiles to POP_JUMP_IF_TRUE, outside the supported subset.
    if a > 0 or a < -10:
        return a
    return 0 - a


def _truthy_branch(a):
    if a:
        return a
    return 0 - a


def _call(a):
    return abs(a) + 1


def _global(a):
    return a + GLOBAL_OFFSET


def _augmented(a):
    a %= 3
    return a


def _tuple_assignment(a, b):
    a, b = b, a
    return a - b


def _loop_with_locals(a):
    total = 0
    # Only range with one argument is supported.
    for i in range(1, a):
        total = total + i
    return total


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
    (_truthy_branch, (-4,), RejectionReason.UNSUPPORTED_CONDITION),
    (_call, (-4,), RejectionReason.UNSUPPORTED_OPCODE),
    (_global, (4,), RejectionReason.UNSUPPORTED_OPCODE),
    (_augmented, (4,), RejectionReason.UNSUPPORTED_OPERATOR),
    (_tuple_assignment, (4, 5), RejectionReason.UNSUPPORTED_OPCODE),
    (_loop_with_locals, (4,), RejectionReason.UNSUPPORTED_ITERATION),
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
    # Not an ``I64Function``, so the wrapper calls ``fast_call`` instead.
    function = None

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
    real.function = None  # Make the wrapper use ``fast_call``.
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


def test_the_wrapper_runs_the_specializations_function_directly():
    add = compiled(make_add(), 1, 2)
    state = state_of(add)
    spec = state.specialization
    assert spec.function is state.compiled._function
    assert spec.arg_count == state.compiled.arg_count == 2
    # ``fast_call`` stays the bound ``try_call`` of the same function, for
    # benchmarks; the wrapper does not call it.
    assert type(spec.fast_call) is BuiltinMethodType
    assert spec.fast_call.__self__ is spec.function
    assert spec.fast_call.__name__ == "try_call"
    # A canary function that returns a constant proves that the wrapper
    # runs exactly ``spec.function``'s machine code, with no Python frame.
    canary = _native.compile_i64(
        ((_native.OP_LOAD_CONST, 424242), (_native.OP_RETURN,)), 2
    )
    spec.function = canary
    with recording() as calls:
        results = [add(i, i) for i in range(5)] + [add(0.5, 1)]
    assert results == [424242] * 5 + [1.5]
    assert canary.try_calls == 6
    assert calls[add.__wrapped__.__code__] == 1
    assert calls["I64Function.try_call"] == 0
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
    with recording(add) as calls:
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

report["decorated"] = loaded()
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
    # The wrapper is a ``NativeWrapper``, so decorating loads the extension
    # module, but nothing that compiles.
    assert report["decorated"] == ["speedysnek._native"]
    assert report["warm"] == ["speedysnek._native"]
    # A rejected function never loads the compiler.
    assert report["rejected"] == ["speedysnek._frontend", "speedysnek._native"]
    assert "speedysnek._compiler" in report["compiled"]
    assert "speedysnek._native" in report["compiled"]
    assert report["status"] == ["compiled", "disabled"]


def test_public_api_is_unchanged_after_compilation():
    compiled(make_add(), 1, 2)
    assert speedysnek.__all__ == [
        "snekit",
        "jit",
        "hunt",
        "HuntReport",
        "HuntFinding",
        "bite",
        "BiteReport",
        "BiteFinding",
        "peek",
        "PeekReport",
        "PeekFinding",
        "chill",
        "ChillReport",
        "ChillFinding",
        "nosnek",
    ]
    public = {name for name in dir(speedysnek) if not name.startswith("_")}
    assert public == {
        "snekit",
        "jit",
        "hunt",
        "HuntReport",
        "HuntFinding",
        "bite",
        "BiteReport",
        "BiteFinding",
        "peek",
        "PeekReport",
        "PeekFinding",
        "chill",
        "ChillReport",
        "ChillFinding",
        "nosnek",
    }
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
