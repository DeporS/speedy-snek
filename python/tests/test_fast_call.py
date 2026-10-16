"""Tests for the native fast-call path, ``I64Function.try_call``.

``try_call(args)`` is the private entry point ``@snekit`` uses on its hot
path. Its contract: the result ``int`` when the call fits the signed i64
tier, ``None`` for every guard miss (wrong arity, non-exact ``int``, value
outside i64, overflow), no exception and no user code for a guard miss,
and an exception only for a caller bug.

The diagnostic path, ``CompiledI64.__call__``, is the oracle for *why* a
call misses: whenever it raises ``NativeGuardFailure``, ``try_call`` must
return ``None``, and whenever it returns a value, ``try_call`` must return
the same value.
"""

import gc
import random
import sys
import threading

import pytest

from speedysnek import _native
from speedysnek._compiler import (
    I64_MAX,
    I64_MIN,
    CompiledI64,
    NativeGuardFailure,
    compile_i64,
)
from speedysnek._frontend import Instruction, Op, analyze
from speedysnek._interpreter import execute

ADD = Instruction(Op.ADD)
SUBTRACT = Instruction(Op.SUBTRACT)
MULTIPLY = Instruction(Op.MULTIPLY)
RETURN = Instruction(Op.RETURN)


def arg(index):
    return Instruction(Op.LOAD_ARG, index)


def const(value):
    return Instruction(Op.LOAD_CONST, value)


def fast(program, arg_count=None):
    """Compile ``program`` and return its bound ``try_call``."""
    return compile_i64(program, arg_count).fast_call


def binary(op):
    return fast((arg(0), arg(1), op, RETURN))


def check_miss(compiled, args):
    """``try_call`` returns ``None`` exactly where the diagnostic path raises."""
    with pytest.raises(NativeGuardFailure):
        compiled(*args)
    assert compiled.fast_call(args) is None


# --- The bound method ----------------------------------------------------


def test_fast_call_is_the_bound_native_method():
    compiled = compile_i64((arg(0), RETURN))
    fast_call = compiled.fast_call
    assert fast_call.__self__ is compiled._function
    assert fast_call.__name__ == "try_call"
    assert type(compiled._function) is _native.I64Function
    assert fast_call((5,)) == 5


def test_fast_call_keeps_the_code_alive():
    fast_call = fast((arg(0), const(3), MULTIPLY, RETURN))
    gc.collect()
    # Neither the CompiledI64 nor the I64Function is referenced elsewhere.
    assert fast_call((14,)) == 42


# --- Success -------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "left", "right", "expected"),
    [
        (ADD, 2, 3, 5),
        (ADD, -5, -6, -11),
        (SUBTRACT, 10, 3, 7),
        (SUBTRACT, -5, -6, 1),
        (MULTIPLY, 6, -7, -42),
        (MULTIPLY, -6, -7, 42),
        (ADD, 0, 0, 0),
        (MULTIPLY, 0, I64_MAX, 0),
    ],
)
def test_each_operation(op, left, right, expected):
    assert binary(op)((left, right)) == expected


def test_results_are_exact_ints():
    for value in (0, 1, -1, 255, 257, -6, 2**40, I64_MAX, I64_MIN):
        result = fast((arg(0), RETURN))((value,))
        assert result == value
        assert type(result) is int


@pytest.mark.parametrize(
    ("op", "left", "right", "expected"),
    [
        (ADD, I64_MAX, 0, I64_MAX),
        (ADD, I64_MIN, 0, I64_MIN),
        (ADD, I64_MIN, I64_MAX, -1),
        (ADD, I64_MAX - 1, 1, I64_MAX),
        (ADD, I64_MIN + 1, -1, I64_MIN),
        (SUBTRACT, -1, I64_MAX, I64_MIN),
        (SUBTRACT, 0, I64_MAX, -I64_MAX),
        (MULTIPLY, I64_MIN, 1, I64_MIN),
        (MULTIPLY, I64_MAX, -1, -I64_MAX),
        (MULTIPLY, -(2**31), 2**32, I64_MIN),
        (MULTIPLY, 3037000499, 3037000499, 3037000499**2),
    ],
)
def test_i64_boundaries(op, left, right, expected):
    assert binary(op)((left, right)) == expected


def test_constants_and_zero_arguments():
    assert fast((const(42), RETURN))(()) == 42
    assert fast((const(I64_MAX), RETURN))(()) == I64_MAX
    assert fast((const(I64_MIN), RETURN))(()) == I64_MIN
    assert fast((const(-7), const(3), MULTIPLY, const(100), ADD, RETURN))(()) == 79


def test_nested_expressions():
    # a - (b - (c - d)) needs four stack slots.
    program = (arg(0), arg(1), arg(2), arg(3), SUBTRACT, SUBTRACT, SUBTRACT, RETURN)
    assert fast(program)((10, 4, 3, 1)) == 10 - (4 - (3 - 1))
    # (a + 2) * b - 1000
    program = (arg(0), const(2), ADD, arg(1), MULTIPLY, const(1000), SUBTRACT, RETURN)
    assert fast(program)((3, 4)) == -980


def test_reused_arguments():
    # a * a - b + c * a
    program = (arg(0), arg(0), MULTIPLY, arg(1), SUBTRACT, arg(2), arg(0))
    program += (MULTIPLY, ADD, RETURN)
    assert fast(program)((5, 3, 2)) == 25 - 3 + 10


def test_unused_arguments_are_still_validated():
    fast_call = fast((arg(1), RETURN), arg_count=3)
    assert fast_call((7, 8, 9)) == 8
    for bad in ("x", 1.0, True, 2**64, None):
        assert fast_call((bad, 8, 9)) is None
        assert fast_call((7, 8, bad)) is None


@pytest.mark.parametrize("arg_count", [7, 8, 9, 16, 300])
def test_many_arguments(arg_count):
    # Crosses the size of the inline argument buffer in Rust.
    program = (arg(arg_count - 1), arg(0), SUBTRACT, RETURN)
    fast_call = fast(program, arg_count)
    args = tuple(range(10, 10 + arg_count))
    assert fast_call(args) == args[-1] - args[0]
    assert fast_call(args[:-1]) is None
    assert fast_call((*args[:-1], 1.5)) is None
    assert fast_call((*args[:-1], 2**63)) is None


def test_maximum_argument_count():
    fast_call = fast((arg(_native.MAX_ARGS - 1), RETURN), _native.MAX_ARGS)
    args = tuple(range(_native.MAX_ARGS))
    assert fast_call(args) == _native.MAX_ARGS - 1
    assert fast_call(args[:-1]) is None


def test_repeated_calls_are_deterministic():
    def reuse(a, b, c):
        return c - a * a + b

    fast_call = compile_i64(analyze(reuse).instructions).fast_call
    args = (12345, -678, 2**40)
    first = fast_call(args)
    assert first == reuse(*args)
    assert all(fast_call(args) == first for _ in range(1000))


def test_multiple_compiled_functions_are_independent():
    functions = [fast((arg(0), const(scale), MULTIPLY, RETURN)) for scale in range(50)]
    for scale, fast_call in enumerate(functions):
        assert fast_call((3,)) == 3 * scale
    # Interleaved calls, and overflow in one, do not affect the others.
    assert functions[2]((I64_MAX,)) is None
    for scale, fast_call in reversed(list(enumerate(functions))):
        assert fast_call((-2,)) == -2 * scale


# --- Guard misses return None ---------------------------------------------


class MyInt(int):
    pass


class WithInt:
    def __int__(self):
        raise AssertionError("__int__ was called")


class WithIndex:
    def __index__(self):
        raise AssertionError("__index__ was called")


NOT_EXACT_INT = [
    True,
    False,
    MyInt(3),
    1.0,
    2.5,
    "1",
    b"1",
    None,
    1j,
    [1],
    (1,),
    {1: 1},
    WithInt(),
    WithIndex(),
    object(),
    int,
]


@pytest.mark.parametrize("value", NOT_EXACT_INT, ids=lambda v: type(v).__name__)
@pytest.mark.parametrize("position", [0, 1])
def test_non_exact_ints_miss(value, position):
    compiled = compile_i64((arg(0), arg(1), ADD, RETURN))
    args = (value, 1) if position == 0 else (1, value)
    check_miss(compiled, args)


@pytest.mark.parametrize(
    "value", [I64_MAX + 1, I64_MIN - 1, 2**64, -(2**64), 2**100, -(2**1000)]
)
def test_ints_outside_i64_miss(value):
    compiled = compile_i64((arg(0), arg(1), ADD, RETURN))
    check_miss(compiled, (value, 0))
    check_miss(compiled, (0, value))


@pytest.mark.parametrize(
    "args", [(), (1,), (1, 2, 3), (1, 2, 3, 4)], ids=lambda a: f"{len(a)}-args"
)
def test_wrong_arity_misses(args):
    check_miss(compile_i64((arg(0), arg(1), ADD, RETURN)), args)


def test_wrong_arity_is_checked_before_the_arguments():
    fast_call = fast((arg(0), arg(1), ADD, RETURN))
    assert fast_call((WithIndex(),)) is None
    assert fast_call((1, 2, WithInt())) is None
    assert fast((const(1), RETURN))((1,)) is None


@pytest.mark.parametrize(
    ("op", "left", "right"),
    [
        (ADD, I64_MAX, 1),
        (ADD, I64_MIN, -1),
        (ADD, I64_MAX, I64_MAX),
        (ADD, I64_MIN, I64_MIN),
        (SUBTRACT, I64_MAX, -1),
        (SUBTRACT, I64_MIN, 1),
        (SUBTRACT, 0, I64_MIN),
        (SUBTRACT, I64_MAX, I64_MIN),
        (MULTIPLY, I64_MAX, 2),
        (MULTIPLY, I64_MIN, -1),
        (MULTIPLY, -1, I64_MIN),
        (MULTIPLY, 2**32, 2**31),
        (MULTIPLY, 3037000500, 3037000500),
    ],
)
def test_overflow_misses(op, left, right):
    compiled = compile_i64((arg(0), arg(1), op, RETURN))
    check_miss(compiled, (left, right))
    exact = execute(compiled.instructions, (left, right))
    assert not I64_MIN <= exact <= I64_MAX


def test_intermediate_overflow_misses_even_if_the_result_fits():
    # (a + 1) - 1 is ``a`` in Python, but a + 1 overflows for I64_MAX.
    program = (arg(0), const(1), ADD, const(1), SUBTRACT, RETURN)
    compiled = compile_i64(program)
    assert execute(program, (I64_MAX,)) == I64_MAX
    check_miss(compiled, (I64_MAX,))
    assert compiled.fast_call((I64_MAX - 1,)) == I64_MAX - 1

    # a * a - a * a is 0 in Python for any a.
    program = (arg(0), arg(0), MULTIPLY, arg(0), arg(0), MULTIPLY, SUBTRACT, RETURN)
    compiled = compile_i64(program)
    assert execute(program, (2**32,)) == 0
    check_miss(compiled, (2**32,))
    assert compiled.fast_call((2**31,)) == 0


def test_a_miss_does_not_affect_the_next_call():
    fast_call = fast((arg(0), arg(1), MULTIPLY, RETURN))
    for _ in range(100):
        assert fast_call((I64_MAX, 2)) is None
        assert fast_call((1.5, 2)) is None
        assert fast_call((2**64, 2)) is None
        assert fast_call((6, 7)) == 42


# --- Guard misses run no user code and raise nothing -----------------------


class Recorder:
    """Records any attempt to inspect it, convert it or do arithmetic."""

    calls = []

    def __getattribute__(self, name):
        Recorder.calls.append(name)
        return object.__getattribute__(self, name)

    def _record(self, *args):
        Recorder.calls.append("dunder")
        return 0

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _record
    __index__ = __int__ = __float__ = __eq__ = __lt__ = __le__ = _record
    __gt__ = __ge__ = __hash__ = __iter__ = __len__ = __bool__ = _record

    def __repr__(self):
        Recorder.calls.append("__repr__")
        return "Recorder()"


class RecordingMeta(type):
    """Records, then refuses, any attribute lookup on its classes."""

    lookups = []

    def __getattribute__(cls, name):
        RecordingMeta.lookups.append(name)
        raise RuntimeError(f"metaclass hook ran for {name!r}")


class Hostile(metaclass=RecordingMeta):
    pass


class HostileInt(int, metaclass=RecordingMeta):
    pass


def test_guard_misses_run_no_user_code():
    fast_call = fast((arg(0), arg(1), arg(2), ADD, ADD, RETURN))
    Recorder.calls.clear()
    RecordingMeta.lookups.clear()
    values = [Recorder(), Hostile(), HostileInt(7)]
    for value in values:
        for args in [(value, 1, 2), (1, value, 2), (1, 2, value)]:
            assert fast_call(args) is None
    # Wrong arity with hostile values: still nothing runs.
    assert fast_call(tuple(values)[:2]) is None
    assert Recorder.calls == []
    assert RecordingMeta.lookups == []


def test_guard_misses_leave_the_exception_state_alone():
    fast_call = fast((arg(0), arg(1), ADD, RETURN))
    handled = KeyError("handled")
    try:
        raise handled
    except KeyError:
        for args in [(1.5, 2), (2**64, 2), (I64_MAX, 1), (1,)]:
            assert fast_call(args) is None
            assert sys.exception() is handled
    assert sys.exception() is None


# --- Caller bugs raise ----------------------------------------------------


class TupleSubclass(tuple):
    pass


@pytest.mark.parametrize(
    "args",
    [[1, 2], TupleSubclass((1, 2)), None, 1, "12", iter((1, 2))],
    ids=lambda a: type(a).__name__,
)
def test_non_tuple_arguments_raise_type_error(args):
    # Not a guard miss: the decorator always passes its exact ``*args``.
    with pytest.raises(TypeError, match="args must be a tuple"):
        fast((arg(0), arg(1), ADD, RETURN))(args)


def test_calling_convention_is_strict():
    fast_call = fast((arg(0), RETURN))
    with pytest.raises(TypeError):
        fast_call()
    with pytest.raises(TypeError):
        fast_call((1,), (2,))
    with pytest.raises(TypeError):
        fast_call(args=(1,))


# --- Agreement with the diagnostic path and the oracles -------------------


def run_i64(program, args):
    """An independent i64 oracle; returns ``None`` on any overflow."""
    stack = []
    for instruction in program:
        if instruction.op is Op.LOAD_ARG:
            stack.append(args[instruction.arg])
        elif instruction.op is Op.LOAD_CONST:
            stack.append(instruction.arg)
        elif instruction.op is Op.RETURN:
            return stack.pop()
        else:
            right = stack.pop()
            left = stack.pop()
            if instruction.op is Op.ADD:
                value = left + right
            elif instruction.op is Op.SUBTRACT:
                value = left - right
            else:
                value = left * right
            if not I64_MIN <= value <= I64_MAX:
                return None
            stack.append(value)
    raise AssertionError("program did not return")


def random_value(rng):
    kind = rng.randrange(7)
    if kind == 0:
        return rng.randint(-100, 100)
    if kind == 1:
        return rng.randint(-(2**34), 2**34)
    if kind == 2:
        return rng.choice([I64_MIN, I64_MIN + 1, -1, 0, 1, I64_MAX - 1, I64_MAX])
    if kind == 3:
        return rng.randint(I64_MIN, I64_MAX)
    if kind == 4:
        return rng.choice([I64_MAX + 1, I64_MIN - 1, 2**100])
    if kind == 5:
        return rng.choice([True, 1.0, "1", None, MyInt(1)])
    return rng.randint(-(2**20), 2**20)


def random_program(rng, arg_count, size):
    program = []

    def emit(budget):
        if budget <= 1 or rng.random() < 0.25:
            if arg_count and rng.random() < 0.7:
                program.append(arg(rng.randrange(arg_count)))
            else:
                value = random_value(rng)
                if type(value) is not int or not I64_MIN <= value <= I64_MAX:
                    value = rng.randint(-1000, 1000)
                program.append(const(value))
            return
        left = rng.randint(1, budget - 1)
        emit(left)
        emit(budget - left)
        program.append(Instruction(rng.choice([Op.ADD, Op.SUBTRACT, Op.MULTIPLY])))

    emit(size)
    program.append(RETURN)
    return tuple(program)


def test_differential_fast_call_matches_every_oracle():
    rng = random.Random("speedysnek-fast-call-differential")
    outcomes = {"ok": 0, "overflow": 0, "argument": 0, "arity": 0}
    for _ in range(400):
        arg_count = rng.randrange(6)
        program = random_program(rng, arg_count, rng.randint(1, 12))
        compiled = compile_i64(program, arg_count)
        for _ in range(8):
            count = arg_count if rng.random() < 0.9 else rng.randrange(7)
            args = tuple(random_value(rng) for _ in range(count))
            try:
                diagnostic = compiled(*args)
            except NativeGuardFailure:
                diagnostic = None
            result = compiled.fast_call(args)
            assert result == diagnostic, (program, args)
            if count != arg_count:
                outcomes["arity"] += 1
                assert result is None
            elif not all(type(a) is int and I64_MIN <= a <= I64_MAX for a in args):
                outcomes["argument"] += 1
                assert result is None
            else:
                expected = run_i64(program, args)
                assert result == expected, (program, args)
                if expected is None:
                    outcomes["overflow"] += 1
                else:
                    outcomes["ok"] += 1
                    assert type(result) is int
                    assert result == execute(program, args)
    # Every path must be well exercised for the test to mean anything.
    assert outcomes["ok"] > 400
    assert outcomes["overflow"] > 200
    assert outcomes["argument"] > 500
    assert outcomes["arity"] > 100


def test_fast_call_and_diagnostic_path_share_the_code():
    def affine(a, b):
        return (a + 2) * b - 1000

    compiled = compile_i64(analyze(affine).instructions)
    assert isinstance(compiled, CompiledI64)
    for args in [(3, 4), (-(2**20), 2**20), (0, 0)]:
        assert compiled.fast_call(args) == compiled(*args) == affine(*args)
    status, value = compiled._function.call((3, 4))
    assert (status, value) == (_native.STATUS_OK, affine(3, 4))


# --- Threads ----------------------------------------------------------------


def test_calls_from_several_threads():
    def affine(a, b):
        return (a + 2) * b - 1000

    fast_call = compile_i64(analyze(affine).instructions).fast_call
    errors = []

    def worker(seed):
        rng = random.Random(seed)
        for _ in range(2000):
            a, b = rng.randint(-(2**20), 2**20), rng.randint(-(2**20), 2**20)
            if fast_call((a, b)) != affine(a, b):
                errors.append((a, b))
            if fast_call((I64_MAX, b or 2)) is not None:
                errors.append(("overflow", b))
            if fast_call((float(a), b)) is not None:
                errors.append(("float", a))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
