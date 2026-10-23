"""Tests for the private Cranelift i64 backend.

The reference interpreter is the oracle: wherever every intermediate value
fits in a signed 64-bit integer, native code must produce the same result,
and wherever one does not, native code must report a guard failure rather
than a wrapped value.
"""

import doctest
import gc
import random
import subprocess
import sys
import threading

import pytest

import speedysnek
from speedysnek import _compiler, _native
from speedysnek._compiler import (
    I64_MAX,
    I64_MIN,
    CompiledI64,
    GuardReason,
    NativeGuardFailure,
    UnsupportedProgramError,
    compile_i64,
    serialize,
)
from speedysnek._frontend import Instruction, Op, analyze
from speedysnek._interpreter import GuardFailure, InvalidProgramError, execute

ADD = Instruction(Op.ADD)
SUBTRACT = Instruction(Op.SUBTRACT)
MULTIPLY = Instruction(Op.MULTIPLY)
RETURN = Instruction(Op.RETURN)


def arg(index):
    return Instruction(Op.LOAD_ARG, index)


def const(value):
    return Instruction(Op.LOAD_CONST, value)


def load(slot):
    return Instruction(Op.LOAD_LOCAL, slot)


def store(slot):
    return Instruction(Op.STORE_LOCAL, slot)


def binary(op):
    return (arg(0), arg(1), op, RETURN)


def assert_guard(compiled, args, reason, *, arg_index=None, instruction_index=None):
    with pytest.raises(NativeGuardFailure) as excinfo:
        compiled(*args)
    failure = excinfo.value
    assert failure.reason is reason
    assert failure.arg_index == arg_index
    assert failure.instruction_index == instruction_index
    return failure


# Signed comparisons for ``run_i64``, written out independently of the
# interpreter.
COMPARE = {
    Op.LESS_THAN: lambda left, right: left < right,
    Op.LESS_EQUAL: lambda left, right: left <= right,
    Op.GREATER_THAN: lambda left, right: left > right,
    Op.GREATER_EQUAL: lambda left, right: left >= right,
    Op.EQUAL: lambda left, right: left == right,
    Op.NOT_EQUAL: lambda left, right: left != right,
}


def run_i64(program, args, budget=_native.DEFAULT_BACKEDGE_BUDGET):
    """An independent i64 oracle: Python ints, checked after every operation
    that runs. It follows jumps, so operations on a path not taken are
    never checked. A backward ``JUMP`` spends one unit of ``budget``, as in
    native code.

    Returns ``("ok", value)``, ``("overflow", instruction_index)`` or
    ``("budget", backward_jump_index)``.
    """
    stack = []
    local_slots = {}
    index = 0
    backedges = 0
    # Between two backward jumps a valid program only jumps forward, so it
    # runs at most ``len(program)`` instructions; the bound catches a broken
    # one.
    for _ in range((budget + 1) * len(program)):
        instruction = program[index]
        if instruction.op is Op.LOAD_ARG:
            stack.append(args[instruction.arg])
        elif instruction.op is Op.LOAD_CONST:
            stack.append(instruction.arg)
        elif instruction.op is Op.LOAD_LOCAL:
            stack.append(local_slots[instruction.arg])
        elif instruction.op is Op.STORE_LOCAL:
            local_slots[instruction.arg] = stack.pop()
        elif instruction.op is Op.RETURN:
            return ("ok", stack.pop())
        elif instruction.op is Op.JUMP:
            if instruction.arg <= index:
                backedges += 1
                if backedges > budget:
                    return ("budget", index)
            index = instruction.arg
            continue
        elif instruction.op is Op.JUMP_IF_FALSE:
            if not stack.pop():
                index = instruction.arg
                continue
        elif instruction.op in COMPARE:
            right = stack.pop()
            left = stack.pop()
            stack.append(COMPARE[instruction.op](left, right))
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
                return ("overflow", index)
            stack.append(value)
        index += 1
    raise AssertionError("program did not return")


# --- Each opcode ---------------------------------------------------------


def test_doc_example():
    def expression(a, b):
        return (a + b) * 2

    compiled = compile_i64(analyze(expression).instructions)
    assert compiled(10, 20) == 60
    assert compiled.arg_count == 2


def test_load_arg_and_return():
    assert compile_i64((arg(0), RETURN))(-7) == -7


def test_load_const_and_return():
    compiled = compile_i64((const(42), RETURN))
    assert compiled.arg_count == 0
    assert compiled() == 42


def test_add():
    assert compile_i64(binary(ADD))(2, 3) == 5


def test_subtract_operand_order():
    assert compile_i64(binary(SUBTRACT))(10, 3) == 7
    assert compile_i64((arg(1), arg(0), SUBTRACT, RETURN))(10, 3) == -7
    assert compile_i64((const(1), arg(0), SUBTRACT, RETURN))(10) == -9


def test_multiply():
    assert compile_i64(binary(MULTIPLY))(6, -7) == -42


def test_negative_numbers():
    compiled = compile_i64(binary(MULTIPLY))
    assert compiled(-6, -7) == 42
    assert compiled(-6, 7) == -42
    assert compile_i64(binary(ADD))(-5, -6) == -11
    assert compile_i64(binary(SUBTRACT))(-5, -6) == 1


def test_constants():
    program = (const(-7), const(3), MULTIPLY, const(100), ADD, RETURN)
    assert compile_i64(program)() == 79
    assert compile_i64((const(I64_MAX), RETURN))() == I64_MAX
    assert compile_i64((const(I64_MIN), RETURN))() == I64_MIN


def test_nested_expressions():
    # a - (b - (c - d)) needs four stack slots.
    program = (arg(0), arg(1), arg(2), arg(3), SUBTRACT, SUBTRACT, SUBTRACT, RETURN)
    assert compile_i64(program)(10, 4, 3, 1) == 10 - (4 - (3 - 1))

    # (a + 2) * b - 1000
    program = (arg(0), const(2), ADD, arg(1), MULTIPLY, const(1000), SUBTRACT, RETURN)
    assert compile_i64(program)(3, 4) == -980


def test_multiple_and_reused_arguments():
    # a * a - b + c * a
    program = (arg(0), arg(0), MULTIPLY, arg(1), SUBTRACT, arg(2), arg(0))
    program += (MULTIPLY, ADD, RETURN)
    assert compile_i64(program)(5, 3, 2) == 25 - 3 + 10


def test_many_arguments():
    program = (arg(299), arg(64), ADD, arg(0), SUBTRACT, RETURN)
    compiled = compile_i64(program)
    assert compiled.arg_count == 300
    args = list(range(300))
    assert compiled(*args) == 299 + 64 - 0


# --- Frontend plus backend -----------------------------------------------


def add(a, b):
    return a + b


def affine(a, b):
    return (a + 2) * b - 1000


def reuse(a, b, c):
    return c - a * a + b


def nested(a, b, c, d):
    return a - (b - (c - d))


def ignores_first(a, b):
    return b * 3


def constant():
    return -12345


SHAPES = [add, affine, reuse, nested, ignores_first, constant]


@pytest.mark.parametrize("func", SHAPES, ids=lambda f: f.__name__)
def test_end_to_end(func):
    arity = func.__code__.co_argcount
    compiled = compile_i64(analyze(func).instructions, arity)
    rng = random.Random(f"e2e-{func.__name__}")
    for _ in range(200):
        args = tuple(rng.randint(-(2**20), 2**20) for _ in range(arity))
        assert compiled(*args) == func(*args) == execute(compiled.instructions, args)


def test_frontend_constant_outside_i64_is_unsupported():
    def f(a):
        return a + 10**30

    with pytest.raises(UnsupportedProgramError) as excinfo:
        compile_i64(analyze(f).instructions)
    assert excinfo.value.index == 1


# --- i64 boundaries ------------------------------------------------------


def test_boundary_values_that_fit():
    assert compile_i64((arg(0), RETURN))(I64_MAX) == I64_MAX
    assert compile_i64((arg(0), RETURN))(I64_MIN) == I64_MIN
    assert compile_i64(binary(ADD))(I64_MAX, 0) == I64_MAX
    assert compile_i64(binary(ADD))(I64_MIN, I64_MAX) == -1
    assert compile_i64(binary(ADD))(I64_MAX - 1, 1) == I64_MAX
    assert compile_i64(binary(ADD))(I64_MIN + 1, -1) == I64_MIN
    assert compile_i64(binary(SUBTRACT))(I64_MIN, 0) == I64_MIN
    assert compile_i64(binary(SUBTRACT))(-1, I64_MAX) == I64_MIN
    assert compile_i64(binary(SUBTRACT))(0, I64_MAX) == -I64_MAX
    assert compile_i64(binary(MULTIPLY))(I64_MIN, 1) == I64_MIN
    assert compile_i64(binary(MULTIPLY))(I64_MAX, -1) == -I64_MAX
    assert compile_i64(binary(MULTIPLY))(-(2**31), 2**32) == I64_MIN
    assert compile_i64(binary(MULTIPLY))(3037000499, 3037000499) == 3037000499**2


@pytest.mark.parametrize(
    ("op", "left", "right"),
    [
        # Addition overflow and underflow.
        (ADD, I64_MAX, 1),
        (ADD, 1, I64_MAX),
        (ADD, I64_MAX, I64_MAX),
        (ADD, I64_MIN, -1),
        (ADD, I64_MIN, I64_MIN),
        # Subtraction overflow and underflow.
        (SUBTRACT, I64_MAX, -1),
        (SUBTRACT, 0, I64_MIN),
        (SUBTRACT, I64_MIN, 1),
        (SUBTRACT, -2, I64_MAX),
        (SUBTRACT, I64_MAX, I64_MIN),
        # Multiplication overflow and underflow.
        (MULTIPLY, I64_MAX, 2),
        (MULTIPLY, 2**32, 2**31),
        (MULTIPLY, I64_MIN, -1),
        (MULTIPLY, -1, I64_MIN),
        (MULTIPLY, I64_MIN, 2),
        (MULTIPLY, 3037000500, 3037000500),
        (MULTIPLY, -(2**32), 2**31 + 1),
    ],
)
def test_overflow_is_a_guard_failure(op, left, right):
    compiled = compile_i64(binary(op))
    failure = assert_guard(
        compiled, (left, right), GuardReason.OVERFLOW, instruction_index=2
    )
    assert op.op.name in str(failure)
    # The interpreter computes the exact result, which does not fit.
    exact = execute(compiled.instructions, (left, right))
    assert not I64_MIN <= exact <= I64_MAX


def test_overflow_reports_the_first_overflowing_instruction():
    # (a + b) * c: with a + b fine, the multiply at index 4 overflows.
    program = (arg(0), arg(1), ADD, arg(2), MULTIPLY, RETURN)
    compiled = compile_i64(program)
    assert_guard(compiled, (1, 2, I64_MAX), GuardReason.OVERFLOW, instruction_index=4)
    assert_guard(compiled, (I64_MAX, 1, 0), GuardReason.OVERFLOW, instruction_index=2)


def test_intermediate_overflow_with_representable_final_result():
    # (a + 1) - 1 is just ``a`` in Python, but a + 1 overflows i64 when
    # a == I64_MAX. Wrapping arithmetic would return the right final
    # value by accident; the native tier must refuse instead.
    program = (arg(0), const(1), ADD, const(1), SUBTRACT, RETURN)
    compiled = compile_i64(program)
    assert execute(program, (I64_MAX,)) == I64_MAX
    assert_guard(compiled, (I64_MAX,), GuardReason.OVERFLOW, instruction_index=2)
    assert compiled(I64_MAX - 1) == I64_MAX - 1

    # a * a - a * a is 0 in Python for any a.
    program = (arg(0), arg(0), MULTIPLY, arg(0), arg(0), MULTIPLY, SUBTRACT, RETURN)
    compiled = compile_i64(program)
    assert execute(program, (2**32,)) == 0
    assert_guard(compiled, (2**32,), GuardReason.OVERFLOW, instruction_index=2)
    assert compiled(2**31) == 0


# --- Compile-time rejection ----------------------------------------------


@pytest.mark.parametrize("value", [I64_MAX + 1, I64_MIN - 1, 2**100, -(2**100)])
def test_constants_outside_i64_are_unsupported(value):
    program = (arg(0), const(value), ADD, RETURN)
    with pytest.raises(UnsupportedProgramError, match="index 1") as excinfo:
        compile_i64(program)
    assert excinfo.value.index == 1
    # The same program is fine for the reference interpreter.
    assert execute(program, (1,)) == value + 1


@pytest.mark.parametrize(
    "program",
    [
        (),
        (arg(0),),
        (ADD, RETURN),
        (arg(0), RETURN, arg(0), RETURN),
        (arg(0), arg(1), RETURN),
        (arg(0), "RETURN"),
    ],
)
def test_malformed_instructions_are_invalid(program):
    with pytest.raises(InvalidProgramError):
        compile_i64(program)


def test_non_iterable_program_is_invalid():
    with pytest.raises(InvalidProgramError, match="must be iterable"):
        compile_i64(None)


def test_explicit_arg_count():
    compiled = compile_i64((arg(1), RETURN), arg_count=3)
    assert compiled.arg_count == 3
    assert compiled(7, 8, 9) == 8

    with pytest.raises(InvalidProgramError, match="arg_count is 1"):
        compile_i64((arg(1), RETURN), arg_count=1)
    with pytest.raises(TypeError):
        compile_i64((arg(0), RETURN), arg_count=True)
    with pytest.raises(UnsupportedProgramError):
        compile_i64((arg(0), RETURN), arg_count=_native.MAX_ARGS + 1)
    assert compile_i64((const(1), RETURN), arg_count=_native.MAX_ARGS).arg_count == (
        _native.MAX_ARGS
    )


def test_serialize():
    program = (arg(0), const(-5), SUBTRACT, RETURN)
    assert serialize(program) == ((1, 0), (2, -5), (4,), (6,))
    program = (arg(0), store(3), load(3), RETURN)
    assert serialize(program) == ((1, 0), (8, 3), (7, 3), (6,))
    assert serialize(iter(program)) == serialize(list(program))
    with pytest.raises(UnsupportedProgramError):
        serialize((const(2**63), RETURN))


def test_opcode_numbers_are_stable():
    # Expression-only programs keep their wire format; locals and then
    # conditionals were added after them.
    assert (
        _native.OP_LOAD_ARG,
        _native.OP_LOAD_CONST,
        _native.OP_ADD,
        _native.OP_SUBTRACT,
        _native.OP_MULTIPLY,
        _native.OP_RETURN,
        _native.OP_LOAD_LOCAL,
        _native.OP_STORE_LOCAL,
        _native.OP_JUMP,
        _native.OP_JUMP_IF_FALSE,
        _native.OP_LESS_THAN,
        _native.OP_LESS_EQUAL,
        _native.OP_GREATER_THAN,
        _native.OP_GREATER_EQUAL,
        _native.OP_EQUAL,
        _native.OP_NOT_EQUAL,
    ) == tuple(range(1, 17))


def test_opcode_table_covers_every_op():
    codes = _compiler._OPCODES
    assert set(codes) == set(Op)
    assert len(set(codes.values())) == len(Op)
    assert all(type(code) is int for code in codes.values())


# --- Runtime guards ------------------------------------------------------


class Number:
    """Records any attempt to inspect it, convert it or do arithmetic."""

    calls = []

    def __getattribute__(self, name):
        Number.calls.append(name)
        return object.__getattribute__(self, name)

    def _record(self, *args):
        Number.calls.append("dunder")
        return 0

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _record
    __index__ = __int__ = __float__ = __eq__ = __lt__ = __le__ = _record
    __gt__ = __ge__ = __hash__ = __iter__ = __len__ = __bool__ = _record

    def __repr__(self):
        Number.calls.append("__repr__")
        return "Number()"


class MyInt(int):
    pass


@pytest.mark.parametrize(
    "value",
    [True, False, MyInt(3), 1.0, 2.5, "1", None, Number(), 1j, [1], (1,)],
    ids=lambda v: type(v).__name__,
)
def test_non_exact_int_arguments_are_rejected(value):
    compiled = compile_i64(binary(ADD))
    failure = assert_guard(compiled, (1, value), GuardReason.NOT_EXACT_INT, arg_index=1)
    assert str(failure) == "argument 1 must be an exact int"


class RecordingMeta(type):
    """Records, then refuses, any attribute lookup on its classes."""

    lookups = []

    def __getattribute__(cls, name):
        RecordingMeta.lookups.append(name)
        raise RuntimeError(f"metaclass hook ran for {name!r}")


class Opaque(metaclass=RecordingMeta):
    pass


def test_guards_do_not_touch_the_rejected_type():
    # Reading ``type(value).__name__`` (or any class attribute) would run
    # the metaclass hook, which raises instead of a NativeGuardFailure.
    value = Opaque()
    RecordingMeta.lookups.clear()
    compiled = compile_i64((arg(0), arg(1), arg(2), ADD, ADD, RETURN))
    failure = assert_guard(
        compiled, (1, 2, value), GuardReason.NOT_EXACT_INT, arg_index=2
    )
    assert str(failure) == "argument 2 must be an exact int"
    assert RecordingMeta.lookups == []


def test_guards_run_no_user_code():
    Number.calls.clear()
    compiled = compile_i64(binary(ADD))
    with pytest.raises(NativeGuardFailure):
        compiled(1, Number())
    with pytest.raises(NativeGuardFailure):
        compiled(Number(), 1)
    with pytest.raises(TypeError):
        compiled._function.call((1, Number()))
    assert Number.calls == []


@pytest.mark.parametrize("value", [I64_MAX + 1, I64_MIN - 1, 2**100, -(2**100)])
def test_arguments_outside_i64_are_rejected(value):
    compiled = compile_i64(binary(ADD))
    assert_guard(compiled, (value, 0), GuardReason.OUT_OF_I64_RANGE, arg_index=0)
    assert_guard(compiled, (0, value), GuardReason.OUT_OF_I64_RANGE, arg_index=1)


def test_missing_and_extra_arguments():
    compiled = compile_i64(binary(ADD))
    assert_guard(compiled, (), GuardReason.WRONG_ARITY)
    assert_guard(compiled, (1,), GuardReason.WRONG_ARITY)
    # Extra arguments are rejected too: the arity is exact.
    assert_guard(compiled, (1, 2, 3), GuardReason.WRONG_ARITY)
    assert_guard(compile_i64((const(1), RETURN)), (1,), GuardReason.WRONG_ARITY)


def test_keyword_arguments_are_not_accepted():
    compiled = compile_i64(binary(ADD))
    with pytest.raises(TypeError):
        compiled(1, b=2)


def test_unread_arguments_are_still_guarded():
    # Every argument is passed to native code, so all must be exact i64.
    compiled = compile_i64((arg(1), RETURN), arg_count=2)
    assert compiled(0, 5) == 5
    assert_guard(compiled, ("x", 5), GuardReason.NOT_EXACT_INT, arg_index=0)
    assert_guard(compiled, (2**64, 5), GuardReason.OUT_OF_I64_RANGE, arg_index=0)


def test_arity_is_checked_before_types():
    compiled = compile_i64(binary(ADD))
    assert_guard(compiled, ("x",), GuardReason.WRONG_ARITY)


def test_bool_rejected_end_to_end():
    compiled = compile_i64(analyze(add).instructions)
    assert add(True, 1) == 2
    assert_guard(compiled, (True, 1), GuardReason.NOT_EXACT_INT, arg_index=0)


# --- Differential testing against the reference interpreter --------------


def random_value(rng):
    kind = rng.randrange(5)
    if kind == 0:
        return rng.randint(-100, 100)
    if kind == 1:
        return rng.randint(-(2**20), 2**20)
    if kind == 2:
        return rng.randint(-(2**34), 2**34)
    if kind == 3:
        return rng.choice([I64_MIN, I64_MIN + 1, -1, 0, 1, I64_MAX - 1, I64_MAX])
    return rng.randint(I64_MIN, I64_MAX)


def random_program(rng, arg_count, size):
    """A random expression tree over ``arg_count`` arguments, as IR."""
    program = []

    def emit(budget):
        if budget <= 1 or rng.random() < 0.25:
            if arg_count and rng.random() < 0.7:
                program.append(arg(rng.randrange(arg_count)))
            else:
                program.append(const(random_value(rng)))
            return
        left = rng.randint(1, budget - 1)
        emit(left)
        emit(budget - left)
        program.append(Instruction(rng.choice([Op.ADD, Op.SUBTRACT, Op.MULTIPLY])))

    emit(size)
    program.append(RETURN)
    return tuple(program)


def test_differential_random_programs():
    rng = random.Random("speedysnek-native-differential")
    outcomes = {"ok": 0, "overflow": 0}
    for _ in range(400):
        arg_count = rng.randrange(5)
        program = random_program(rng, arg_count, rng.randint(1, 12))
        compiled = compile_i64(program, arg_count)
        for _ in range(8):
            args = tuple(random_value(rng) for _ in range(arg_count))
            expected, detail = run_i64(program, args)
            outcomes[expected] += 1
            reference = execute(program, args)
            if expected == "ok":
                assert compiled(*args) == reference == detail, (program, args)
            else:
                assert_guard(
                    compiled, args, GuardReason.OVERFLOW, instruction_index=detail
                )
    # Both paths must be well exercised for the test to mean anything.
    assert outcomes["ok"] > 1000
    assert outcomes["overflow"] > 300


def test_differential_small_values_never_overflow():
    # Small inputs in shallow programs stay far inside i64: the native
    # result must always equal the reference interpreter's.
    rng = random.Random("speedysnek-native-small")
    for _ in range(200):
        arg_count = rng.randrange(1, 4)
        program = random_program(rng, arg_count, rng.randint(1, 6))
        program = tuple(
            const(i.arg % 1000) if i.op is Op.LOAD_CONST else i for i in program
        )
        compiled = compile_i64(program, arg_count)
        for _ in range(10):
            args = tuple(rng.randint(-1000, 1000) for _ in range(arg_count))
            assert compiled(*args) == execute(program, args)


def test_reference_succeeds_where_native_overflows():
    rng = random.Random("speedysnek-native-bigint")
    seen = 0
    for _ in range(300):
        program = random_program(rng, 2, rng.randint(3, 10))
        compiled = compile_i64(program, 2)
        args = (random_value(rng), random_value(rng))
        reference = execute(program, args)  # Big ints: always succeeds.
        status, index = run_i64(program, args)
        if status == "overflow":
            seen += 1
            assert type(reference) is int
            assert_guard(compiled, args, GuardReason.OVERFLOW, instruction_index=index)
    assert seen > 50


# --- Local variables -----------------------------------------------------


def calculate_score(wins, draws, penalties):
    points = wins * 3 + draws
    deduction = penalties * 2
    result = points - deduction
    return result


def reuse_local(a, b):
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


def unused(a, b):
    t = a * b  # noqa: F841 -- computed, never read.
    return a - b


LOCAL_SHAPES = [
    calculate_score,
    reuse_local,
    reassign,
    overwrite_argument,
    copy_value,
    intermediate_overflow,
    unused,
]


def differential(func, args):
    """Check native code against CPython, the interpreter and ``run_i64``.

    Returns ``"ok"`` or ``"overflow"``.
    """
    instructions = analyze(func).instructions
    compiled = compile_i64(instructions, func.__code__.co_argcount)
    expected = func(*args)
    assert execute(instructions, args) == expected
    status, detail = run_i64(instructions, args)
    if status == "ok":
        assert compiled(*args) == detail == expected, (func, args)
        assert compiled.fast_call(args) == expected
    else:
        assert_guard(compiled, args, GuardReason.OVERFLOW, instruction_index=detail)
        assert compiled.fast_call(args) is None
    return status


@pytest.mark.parametrize("func", LOCAL_SHAPES, ids=lambda f: f.__name__)
def test_locals_end_to_end(func):
    arity = func.__code__.co_argcount
    rng = random.Random(f"locals-{func.__name__}")
    for _ in range(200):
        args = tuple(rng.randint(-(2**20), 2**20) for _ in range(arity))
        assert differential(func, args) == "ok"


def test_local_examples():
    assert compile_i64(analyze(calculate_score).instructions)(5, 2, 3) == 11
    assert compile_i64(analyze(reuse_local).instructions)(3, 4) == 49
    assert compile_i64(analyze(reassign).instructions)(3, 4) == 18
    assert compile_i64(analyze(overwrite_argument).instructions)(3, 4) == 16


def test_copy_is_not_an_alias():
    # ``y = x`` copies; reassigning ``x`` afterwards leaves ``y`` alone.
    compiled = compile_i64(analyze(copy_value).instructions)
    assert compiled(10, 3) == 10 - 3
    assert compiled(-5, 7) == -12


@pytest.mark.parametrize("func", LOCAL_SHAPES, ids=lambda f: f.__name__)
def test_locals_at_i64_boundaries(func):
    arity = func.__code__.co_argcount
    values = [I64_MIN, I64_MIN + 1, -(2**32), -1, 0, 1, 2**32, I64_MAX - 1, I64_MAX]
    seen = set()
    for position in range(arity):
        for value in values:
            for other in (0, 1, -1):
                args = tuple(value if i == position else other for i in range(arity))
                seen.add(differential(func, args))
    assert "ok" in seen


def test_overflow_in_an_overwritten_computation():
    # value = a + 1 (index 2) is overwritten by 0 before anything reads it.
    compiled = compile_i64(analyze(intermediate_overflow).instructions)
    assert intermediate_overflow(I64_MAX) == 0
    assert_guard(compiled, (I64_MAX,), GuardReason.OVERFLOW, instruction_index=2)
    assert compiled.fast_call((I64_MAX,)) is None
    assert compiled(I64_MAX - 1) == 0
    assert compiled(I64_MIN) == 0


def test_overflow_in_an_unused_computation():
    # t = a * b (index 2) is never read; a - b fits.
    compiled = compile_i64(analyze(unused).instructions)
    assert unused(2**32, 2**32) == 0
    assert_guard(compiled, (2**32, 2**32), GuardReason.OVERFLOW, instruction_index=2)
    assert compiled(2**31, 2**31) == 0


def test_overflow_in_an_intermediate_local():
    # points = wins * 3 + draws overflows, although the final result fits.
    compiled = compile_i64(analyze(calculate_score).instructions)
    args = (I64_MAX // 3 + 1, 0, I64_MAX // 2)
    assert I64_MIN <= calculate_score(*args) <= I64_MAX
    assert_guard(compiled, args, GuardReason.OVERFLOW, instruction_index=2)


def test_overflow_in_a_reassigned_argument():
    compiled = compile_i64(analyze(overwrite_argument).instructions)
    assert_guard(compiled, (I64_MAX, 0), GuardReason.OVERFLOW, instruction_index=2)
    assert compiled(I64_MAX - 1, 1) == I64_MAX


def test_unread_reassigned_argument_is_still_guarded():
    # The argument rule: every argument must be an exact i64.
    program = (const(1), store(0), load(0), RETURN)
    compiled = compile_i64(program, arg_count=1)
    assert compiled(5) == 1
    assert_guard(compiled, ("x",), GuardReason.NOT_EXACT_INT, arg_index=0)


def test_local_index_limit():
    program = (const(5), store(_native.MAX_LOCALS - 1))
    program += (load(_native.MAX_LOCALS - 1), RETURN)
    assert compile_i64(program)() == 5

    program = (const(5), store(_native.MAX_LOCALS), load(_native.MAX_LOCALS), RETURN)
    assert execute(program, ()) == 5  # Fine for the interpreter.
    with pytest.raises(UnsupportedProgramError, match="native limit") as excinfo:
        compile_i64(program)
    assert excinfo.value.index == 1


@pytest.mark.parametrize(
    "program",
    [
        (load(0), RETURN),
        (const(1), store(0), load(1), RETURN),
        (load(0), const(1), store(0), RETURN),
        (store(0), const(1), RETURN),
        (const(1), store(0)),
        (const(1), store(0), RETURN),
        (const(1), store(0), load(0), load(0), RETURN),
    ],
)
def test_malformed_local_programs_are_invalid(program):
    with pytest.raises(InvalidProgramError):
        compile_i64(program)
    with pytest.raises(InvalidProgramError):
        serialize(program)


def test_rust_rejects_uninitialized_reads_on_its_own():
    # Bypass Python's validation entirely.
    with pytest.raises(ValueError, match="reads local 0 before it is assigned"):
        _native.compile_i64(((7, 0), (6,)), 0)
    with pytest.raises(ValueError, match="reads local 1 before it is assigned"):
        _native.compile_i64(((2, 4), (8, 0), (7, 1), (6,)), 0)
    function = _native.compile_i64(((2, 4), (8, 0), (7, 0), (6,)), 0)
    assert function.call(()) == (_native.STATUS_OK, 4)


def random_local_program(rng, arg_count, statements, slots):
    """Random straight-line IR: assignments to ``slots`` local slots, then a
    returned expression. Leaves read arguments, constants and assigned
    slots."""
    program = []
    assigned = set()

    def emit(budget):
        if budget <= 1 or rng.random() < 0.25:
            choice = rng.random()
            if assigned and choice < 0.45:
                program.append(load(rng.choice(sorted(assigned))))
            elif arg_count and choice < 0.85:
                program.append(arg(rng.randrange(arg_count)))
            else:
                program.append(const(random_value(rng)))
            return
        left = rng.randint(1, budget - 1)
        emit(left)
        emit(budget - left)
        program.append(Instruction(rng.choice([Op.ADD, Op.SUBTRACT, Op.MULTIPLY])))

    for _ in range(statements):
        emit(rng.randint(1, 5))
        slot = rng.randrange(slots)
        program.append(store(slot))
        assigned.add(slot)
    emit(rng.randint(1, 5))
    program.append(RETURN)
    return tuple(program)


def test_differential_random_local_programs():
    rng = random.Random("speedysnek-native-locals")
    outcomes = {"ok": 0, "overflow": 0}
    for _ in range(300):
        arg_count = rng.randrange(4)
        program = random_local_program(
            rng, arg_count, rng.randint(1, 6), rng.randint(1, 4)
        )
        compiled = compile_i64(program, arg_count)
        for _ in range(8):
            args = tuple(random_value(rng) for _ in range(arg_count))
            expected, detail = run_i64(program, args)
            outcomes[expected] += 1
            if expected == "ok":
                assert compiled(*args) == execute(program, args) == detail, (
                    program,
                    args,
                )
            else:
                assert_guard(
                    compiled, args, GuardReason.OVERFLOW, instruction_index=detail
                )
    assert outcomes["ok"] > 500
    assert outcomes["overflow"] > 300


def random_local_source(rng, arg_count):
    """Random Python source for a straight-line function with assignments,
    reassignments (arguments included) and copies."""
    params = [f"a{i}" for i in range(arg_count)]
    names = params + ["x", "y", "total"]
    bound = list(params)

    def expr(depth=0):
        if depth >= 2 or rng.random() < 0.35:
            if bound and rng.random() < 0.8:
                return rng.choice(bound)
            return str(rng.choice([0, 1, 2, 3, 7, 255, 256, 1000, -1, -5]))
        op = rng.choice("+-*")
        return f"({expr(depth + 1)} {op} {expr(depth + 1)})"

    lines = []
    for _ in range(rng.randint(1, 5)):
        target = rng.choice(names)
        lines.append(f"    {target} = {expr()}")
        if target not in bound:
            bound.append(target)
    lines.append(f"    return {expr()}")
    return f"def f({', '.join(params)}):\n" + "\n".join(lines) + "\n"


def test_differential_random_python_functions():
    # Python source -> frontend -> native, against CPython itself.
    rng = random.Random("speedysnek-native-local-source")
    outcomes = {"ok": 0, "overflow": 0}
    for _ in range(150):
        namespace = {}
        exec(random_local_source(rng, rng.randrange(4)), namespace)
        func = namespace["f"]
        assert analyze(func).supported, analyze(func).rejection
        for _ in range(6):
            args = tuple(
                rng.choice([rng.randint(-50, 50), random_value(rng)])
                for _ in range(func.__code__.co_argcount)
            )
            outcomes[differential(func, args)] += 1
    assert outcomes["ok"] > 400
    assert outcomes["overflow"] > 50


def test_expression_only_programs_are_unchanged():
    # Functions without assignments never use local slots.
    for func in SHAPES:
        ops = {i.op for i in analyze(func).instructions}
        assert not ops & {Op.LOAD_LOCAL, Op.STORE_LOCAL}


# --- Native ABI and serialized IR ----------------------------------------


def test_native_status_codes():
    function = _native.compile_i64(((1, 0), (1, 1), (3,), (6,)), 2)
    assert function.call((2, 3)) == (_native.STATUS_OK, 5)
    assert function.call((I64_MAX, 1)) == (_native.STATUS_OVERFLOW, 2)
    # The generated code checks the length itself and reads nothing.
    assert function.call((1,)) == (_native.STATUS_BAD_ARITY, 0)
    assert function.call((1, 2, 3)) == (_native.STATUS_BAD_ARITY, 0)
    assert function.call(()) == (_native.STATUS_BAD_ARITY, 0)
    statuses = {_native.STATUS_OK, _native.STATUS_OVERFLOW, _native.STATUS_BAD_ARITY}
    assert len(statuses) == 3


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ([1, 2], TypeError),
        ((1, True), TypeError),
        ((1, MyInt(2)), TypeError),
        ((1, 2.0), TypeError),
        ((1, 2**63), OverflowError),
        ((I64_MIN - 1, 2), OverflowError),
    ],
)
def test_native_call_validates_arguments(args, error):
    function = _native.compile_i64(((1, 0), (1, 1), (3,), (6,)), 2)
    with pytest.raises(error):
        function.call(args)


LOAD_ARG_0 = (1, 0)
MAX_LOCALS = _native.MAX_LOCALS


@pytest.mark.parametrize(
    ("program", "arg_count", "error"),
    [
        ([LOAD_ARG_0, (6,)], 1, TypeError),  # Not a tuple.
        ((), 0, ValueError),  # Empty.
        (([1, 0], (6,)), 1, TypeError),  # Entry is not a tuple.
        (((), (6,)), 1, ValueError),  # No opcode.
        (((1, 0, 0), (6,)), 1, ValueError),  # Too many fields.
        (((99,),), 0, ValueError),  # Unknown opcode.
        (((0,),), 0, ValueError),
        (((-1,),), 0, ValueError),
        (((True, 0), (6,)), 1, TypeError),  # bool opcode.
        (((1.0, 0), (6,)), 1, TypeError),
        (((1, "0"), (6,)), 1, TypeError),
        (((1, MyInt(0)), (6,)), 1, TypeError),
        (((1,), (6,)), 1, ValueError),  # LOAD_ARG without operand.
        (((2,), (6,)), 0, ValueError),  # LOAD_CONST without operand.
        (((2, 1), (3, 1), (6,)), 0, ValueError),  # ADD with operand.
        (((2, 1), (6, 0)), 0, ValueError),  # RETURN with operand.
        (((1, -1), (6,)), 1, ValueError),  # Negative argument index.
        (((1, 1), (6,)), 1, ValueError),  # Argument index >= arg_count.
        (((1, 2**40), (6,)), 1, ValueError),
        (((2, 2**63), (6,)), 0, ValueError),  # Constant outside i64.
        (((2, 1), (3,), (6,)), 0, ValueError),  # Stack underflow.
        (((6,),), 0, ValueError),  # RETURN on an empty stack.
        (((2, 1),), 0, ValueError),  # No RETURN.
        (((2, 1), (6,), (2, 1), (6,)), 0, ValueError),  # Two RETURNs.
        (((2, 1), (6,), (2, 1)), 0, ValueError),  # Code after RETURN.
        (((2, 1), (2, 2), (6,)), 0, ValueError),  # Extra stack value.
        ((LOAD_ARG_0, (6,)), -1, ValueError),
        ((LOAD_ARG_0, (6,)), _native.MAX_ARGS + 1, ValueError),
        ((LOAD_ARG_0, (6,)), 2**70, ValueError),
        ((LOAD_ARG_0, (6,)), True, TypeError),
        ((LOAD_ARG_0, (6,)), 1.0, TypeError),
        ((LOAD_ARG_0, (6,)), None, TypeError),
        # Local variables, checked by Rust alone.
        (((7,), (6,)), 0, ValueError),  # LOAD_LOCAL without operand.
        (((2, 1), (8,), (2, 1), (6,)), 0, ValueError),  # STORE_LOCAL without.
        (((7, -1), (6,)), 0, ValueError),  # Negative local index.
        (((2, 1), (8, -1), (2, 1), (6,)), 0, ValueError),
        (((2, 1), (8, 2**40), (7, 2**40), (6,)), 0, ValueError),  # Not a u32.
        (((2, 1), (8, MAX_LOCALS), (7, MAX_LOCALS), (6,)), 0, ValueError),
        (((7, 0), (6,)), 0, ValueError),  # Read before any store.
        (((2, 1), (8, 0), (7, 1), (6,)), 0, ValueError),  # Another slot.
        (((7, 0), (2, 1), (8, 0), (6,)), 0, ValueError),  # Stored only later.
        (((8, 0), (2, 1), (6,)), 0, ValueError),  # STORE_LOCAL underflow.
        (((2, 1), (8, 0)), 0, ValueError),  # Ends without RETURN.
        (((2, 1), (8, 0), (6,)), 0, ValueError),  # RETURN on an empty stack.
        (((2, 1), (8, 0), (7, 0), (7, 0), (6,)), 0, ValueError),  # Extra value.
        (((2, 1), (6,), (8, 0)), 0, ValueError),  # Store after RETURN.
        (((2, 1), (8, True), (7, 0), (6,)), 0, TypeError),  # bool index.
    ],
)
def test_malformed_serialized_programs(program, arg_count, error):
    with pytest.raises(error):
        _native.compile_i64(program, arg_count)


def test_malformed_input_does_not_crash_the_interpreter():
    # Run a batch of garbage in a child process: a panic would surface as
    # PanicException, an abort or segfault as a non-zero exit code.
    code = """
import random
from speedysnek import _native
rng = random.Random(0)
values = [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, 2**63, -(2**63) - 1, True, None, 1.5, "x", ()]
for _ in range(10000):
    program = tuple(
        tuple(rng.choice(values) for _ in range(rng.randrange(4)))
        for _ in range(rng.randrange(6))
    )
    try:
        _native.compile_i64(program, rng.choice([0, 1, 2, -1, True]))
    except (TypeError, ValueError):
        pass
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "ok"


# --- Conditionals --------------------------------------------------------

LESS_THAN = Instruction(Op.LESS_THAN)
CMP_OPS = [
    Op.LESS_THAN,
    Op.LESS_EQUAL,
    Op.GREATER_THAN,
    Op.GREATER_EQUAL,
    Op.EQUAL,
    Op.NOT_EQUAL,
]
CMP_SYMBOLS = dict(zip(CMP_OPS, ["<", "<=", ">", ">=", "==", "!="], strict=True))


def jump(target):
    return Instruction(Op.JUMP, target)


def jump_if_false(target):
    return Instruction(Op.JUMP_IF_FALSE, target)


def define(source):
    namespace = {}
    exec(source, namespace)
    return namespace["f"]


def cond_calculate_score(wins, penalties):
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


def nested_branches(a, b, c):
    r = 0
    if a < b:
        if b < c:
            r = c - a
        else:
            r = b * 2
    else:
        r = a - c
    if r == 0:
        r += 7
    return r


def sequential_branches(a, b):
    if a < 0:
        a = 0 - a
    if b < 0:
        b = 0 - b
    return a * b


def keep_or_replace(a, b):
    x = a * 3
    if a != b:
        x = b
    return x + a


def swap_if_less(a, b):
    c = a
    if a < b:
        a = b
        b = c
    return a - b * 2


BRANCH_SHAPES = [
    cond_calculate_score,
    choose,
    adjust,
    selected_branch,
    nested_branches,
    sequential_branches,
    keep_or_replace,
    swap_if_less,
]


@pytest.mark.parametrize("func", BRANCH_SHAPES, ids=lambda f: f.__name__)
def test_conditionals_end_to_end(func):
    # Like ``differential``, but compiled once for every input.
    instructions = analyze(func).instructions
    compiled = compile_i64(instructions, func.__code__.co_argcount)
    arity = func.__code__.co_argcount
    rng = random.Random(f"branches-{func.__name__}")
    edges = [I64_MIN, I64_MIN + 1, -1, 0, 1, I64_MAX - 1, I64_MAX]
    inputs = [tuple(rng.randint(-50, 50) for _ in range(arity)) for _ in range(150)]
    inputs += [tuple(random_value(rng) for _ in range(arity)) for _ in range(150)]
    inputs += [tuple(rng.choice(edges) for _ in range(arity)) for _ in range(50)]
    outcomes = {"ok": 0, "overflow": 0}
    for args in inputs:
        expected = func(*args)
        assert execute(instructions, args) == expected
        status, detail = run_i64(instructions, args)
        outcomes[status] += 1
        if status == "ok":
            assert compiled(*args) == detail == expected, (func, args)
            assert compiled.fast_call(args) == expected
        else:
            assert_guard(compiled, args, GuardReason.OVERFLOW, instruction_index=detail)
            assert compiled.fast_call(args) is None
    assert outcomes["ok"] > 150


def test_conditional_examples():
    score = compile_i64(analyze(cond_calculate_score).instructions)
    assert (score(1, 5), score(4, 2)) == (0, 15)
    pick = compile_i64(analyze(choose).instructions)
    assert (pick(5, 3), pick(3, 5), pick(4, 4)) == (2, 2, 0)
    tweak = compile_i64(analyze(adjust).instructions)
    assert (tweak(1, 2), tweak(2, 1)) == (6, 4)


@pytest.mark.parametrize("op", CMP_OPS, ids=lambda op: op.name)
def test_comparisons_at_signed_boundaries(op):
    f = define(
        f"def f(a, b):\n    r = 0\n    if a {CMP_SYMBOLS[op]} b:\n        r = 1\n"
        f"    return r\n"
    )
    compiled = compile_i64(analyze(f).instructions)
    values = [I64_MIN, I64_MIN + 1, -2, -1, 0, 1, 2, I64_MAX - 1, I64_MAX]
    for a in values:
        for b in values:
            result = compiled(a, b)
            assert type(result) is int and result == f(a, b), (op, a, b)
            assert compiled.fast_call((a, b)) == result


def test_comparisons_are_signed_and_never_subtract():
    lt = compile_i64(
        (
            arg(0),
            arg(1),
            LESS_THAN,
            jump_if_false(6),
            const(1),
            RETURN,
            const(0),
            RETURN,
        )
    )
    # Unsigned comparison would get these wrong; ``a - b < 0`` would
    # overflow on the first two.
    assert lt(I64_MIN, I64_MAX) == 1
    assert lt(I64_MAX, I64_MIN) == 0
    assert lt(I64_MIN, 1) == 1
    assert lt(-1, 0) == 1
    assert lt(0, -1) == 0
    assert lt(I64_MIN, I64_MIN) == 0


def test_overflow_on_the_branch_not_taken_is_not_a_guard_failure():
    compiled = compile_i64(analyze(selected_branch).instructions)
    # a < b: only ``result += 1`` runs; ``result *= b`` would overflow.
    assert compiled(2, I64_MAX) == 3
    assert compiled.fast_call((2, I64_MAX)) == 3
    # a >= b: only ``result *= b`` runs; ``result += 1`` would overflow.
    assert compiled(I64_MAX, 1) == I64_MAX
    assert compiled.fast_call((I64_MAX, 1)) == I64_MAX


def test_overflow_on_the_selected_branch_is_a_guard_failure():
    instructions = analyze(selected_branch).instructions
    compiled = compile_i64(instructions)
    status, index = run_i64(instructions, (I64_MAX, 2))
    assert status == "overflow"
    assert_guard(compiled, (I64_MAX, 2), GuardReason.OVERFLOW, instruction_index=index)
    assert instructions[index].op is Op.MULTIPLY
    assert compiled.fast_call((I64_MAX, 2)) is None
    # The code is unchanged and keeps working.
    assert compiled(1, 2) == 2
    assert compiled(3, 2) == 6


def test_overwritten_overflow_on_the_selected_branch_still_fails():
    def f(a, b):
        r = 0
        if a < b:
            r = b * b
            r = 1
        return r

    instructions = analyze(f).instructions
    compiled = compile_i64(instructions)
    assert compiled(1, 2) == 1
    assert compiled(3, 2) == 0
    # Not taken: no overflow check runs.
    assert compiled(I64_MAX, 2) == 0
    status, index = run_i64(instructions, (1, I64_MAX))
    assert status == "overflow" and instructions[index].op is Op.MULTIPLY
    assert_guard(compiled, (1, I64_MAX), GuardReason.OVERFLOW, instruction_index=index)


def test_values_are_preserved_across_joins():
    f = compile_i64(analyze(keep_or_replace).instructions)
    assert f(2, 2) == 8  # Not assigned on this path: keeps a * 3.
    assert f(2, 5) == 7
    swap = compile_i64(analyze(swap_if_less).instructions)
    assert swap(1, 5) == 5 - 1 * 2  # Swapped through the copy.
    assert swap(5, 1) == 5 - 1 * 2  # Unchanged.


def test_hand_written_paths_with_separate_returns():
    program = (
        arg(0),
        arg(1),
        Instruction(Op.EQUAL),
        jump_if_false(4),  # Both outcomes continue at index 4.
        arg(0),
        const(0),
        Instruction(Op.NOT_EQUAL),
        jump_if_false(10),
        arg(0),
        RETURN,
        const(-1),  # 10
        RETURN,
    )
    compiled = compile_i64(program)
    assert (compiled(3, 3), compiled(0, 0), compiled(3, 4)) == (3, -1, 3)


def random_branch_program(rng, arg_count, slots):
    """Random acyclic IR built from assignments, if/else statements, early
    returns and a final return. Every read slot is assigned on every path."""
    program = []

    def emit(budget, assigned):
        if budget <= 1 or rng.random() < 0.3:
            choice = rng.random()
            if assigned and choice < 0.45:
                program.append(load(rng.choice(sorted(assigned))))
            elif arg_count and choice < 0.85:
                program.append(arg(rng.randrange(arg_count)))
            else:
                program.append(const(random_value(rng)))
            return
        left = rng.randint(1, budget - 1)
        emit(left, assigned)
        emit(budget - left, assigned)
        program.append(Instruction(rng.choice([Op.ADD, Op.SUBTRACT, Op.MULTIPLY])))

    def block(statements, assigned, depth):
        """Emit statements; return the slots assigned on every path out, or
        None if every path returned."""
        assigned = set(assigned)
        for _ in range(statements):
            choice = rng.random()
            if depth and choice < 0.4:
                emit(rng.randint(1, 3), assigned)
                emit(rng.randint(1, 3), assigned)
                program.append(Instruction(rng.choice(CMP_OPS)))
                branch = len(program)
                program.append(None)
                then = block(rng.randint(0, 2), assigned, depth - 1)
                skip = None
                if rng.random() < 0.6:
                    if then is not None:
                        skip = len(program)
                        program.append(None)
                    program[branch] = jump_if_false(len(program))
                    other = block(rng.randint(0, 2), assigned, depth - 1)
                else:
                    program[branch] = jump_if_false(len(program))
                    other = assigned
                if skip is not None:
                    program[skip] = jump(len(program))
                paths = [path for path in (then, other) if path is not None]
                if not paths:
                    return None
                assigned = set.intersection(*paths)
            elif choice < 0.48:
                emit(rng.randint(1, 4), assigned)
                program.append(RETURN)
                return None
            else:
                emit(rng.randint(1, 4), assigned)
                slot = rng.randrange(slots)
                program.append(store(slot))
                assigned.add(slot)
        return assigned

    assigned = block(rng.randint(1, 5), set(), 2)
    if assigned is not None:
        emit(rng.randint(1, 4), assigned)
        program.append(RETURN)
    return tuple(program)


def test_differential_random_branch_programs():
    rng = random.Random("speedysnek-native-branches")
    outcomes = {"ok": 0, "overflow": 0}
    jumps = 0
    for _ in range(250):
        arg_count = rng.randrange(1, 4)
        program = random_branch_program(rng, arg_count, rng.randint(1, 3))
        jumps += sum(i.op is Op.JUMP or i.op is Op.JUMP_IF_FALSE for i in program)
        compiled = compile_i64(program, arg_count)
        for _ in range(6):
            if rng.random() < 0.5:
                args = tuple(rng.randint(-3, 3) for _ in range(arg_count))
            else:
                args = tuple(random_value(rng) for _ in range(arg_count))
            expected, detail = run_i64(program, args)
            outcomes[expected] += 1
            reference = execute(program, args)
            if expected == "ok":
                assert compiled(*args) == reference == detail, (program, args)
                assert compiled.fast_call(args) == detail
            else:
                assert_guard(
                    compiled, args, GuardReason.OVERFLOW, instruction_index=detail
                )
                assert compiled.fast_call(args) is None
    assert outcomes["ok"] > 500
    assert outcomes["overflow"] > 100
    assert jumps > 300


def test_serialize_conditionals():
    program = (arg(0), arg(1), LESS_THAN, jump_if_false(5), jump(5), const(1), RETURN)
    assert serialize(program) == ((1, 0), (1, 1), (11,), (10, 5), (9, 5), (2, 1), (6,))
    wire = {
        op: serialize(
            (arg(0), arg(0), Instruction(op), jump_if_false(4), const(1), RETURN)
        )[2][0]
        for op in CMP_OPS
    }
    assert list(wire.values()) == [11, 12, 13, 14, 15, 16]


def test_jump_limit():
    limit = _native.MAX_JUMPS
    chain = tuple(jump(index + 1) for index in range(limit))
    compiled = compile_i64(chain + (const(5), RETURN))
    assert compiled() == 5
    too_many = tuple(jump(index + 1) for index in range(limit + 1)) + (const(5), RETURN)
    with pytest.raises(UnsupportedProgramError, match="native limit") as excinfo:
        compile_i64(too_many)
    assert excinfo.value.index == limit
    with pytest.raises(ValueError, match="maximum of 4096 jumps"):
        _native.compile_i64(_compiler._encode(too_many), 0)


# Wire opcodes for the hand-written programs below.
J, JIF, LT, LE, GT, GE, EQ, NE = range(9, 17)


@pytest.mark.parametrize(
    ("program", "message"),
    [
        # Targets.
        (((9, 0), (2, 1), (6,)), "must start with its condition"),  # Jumps to itself.
        (((2, 1), (8, 0), (9, 1), (6,)), "loop at index 1 must start"),  # No test.
        (((9, 1), (9, 0), (2, 1), (6,)), "first jump or RETURN from there, at index 0"),
        (
            ((2, 0), (2, 0), (LT,), (JIF, 2), (2, 1), (6,)),
            "JUMP_IF_FALSE at index 3 goes back",
        ),
        (((9, 3), (2, 1), (6,)), "index 3, past the end"),
        (((9, 2**31), (2, 1), (6,)), "past the end"),
        (((9, -1), (2, 1), (6,)), "invalid target -1"),
        (((9, 2**40), (2, 1), (6,)), "invalid target"),
        (((9,), (2, 1), (6,)), "requires an operand"),
        (((10,), (2, 1), (6,)), "requires an operand"),
        (((2, 0), (2, 0), (11, 0), (10, 5), (2, 1), (6,)), "takes no operand"),
        # Reachability and termination.
        (((9, 2), (2, 1), (2, 2), (6,)), "index 1 is unreachable"),
        (((2, 1), (6,), (2, 2), (6,)), "index 2 is unreachable"),
        (((2, 0), (2, 0), (LT,), (JIF, 6), (2, 1), (6,), (2, 2)), "falls off the end"),
        # Conditions.
        (((2, 0), (2, 0), (LT,), (6,)), "comparison at index 2 must be immediately"),
        (((2, 0), (2, 0), (EQ,), (8, 0), (2, 1), (6,)), "must be immediately"),
        (
            ((2, 0), (JIF, 3), (2, 1), (2, 2), (6,)),
            "needs a condition but found an integer",
        ),
        (((JIF, 2), (2, 1), (6,)), "stack underflow at index 0"),
        (((2, 0), (GE,), (JIF, 4), (2, 1), (6,)), "stack underflow at index 1"),
        # The stack must be empty at jumps and joins.
        (((2, 1), (9, 2), (6,)), "index 1 leaves 1 value"),
        (
            ((2, 9), (2, 0), (2, 0), (LT,), (JIF, 6), (2, 1), (6,)),
            "index 4 leaves 1 value",
        ),
        (
            ((2, 0), (2, 0), (LT,), (JIF, 5), (2, 1), (2, 2), (3,), (6,)),
            "jump target at index 5 is reached from index 4 with 1 value",
        ),
        # Locals must be assigned on every path into a read.
        (
            ((2, 0), (2, 0), (LT,), (JIF, 6), (2, 1), (8, 0), (7, 0), (6,)),
            "index 6 reads local 0 before it is assigned on every path",
        ),
        (
            (
                (2, 0),
                (2, 0),
                (NE,),
                (JIF, 7),
                (2, 1),
                (8, 0),
                (9, 9),
                (2, 2),
                (8, 1),
                (7, 0),
                (6,),
            ),
            "index 9 reads local 0",
        ),
        (
            ((2, 0), (2, 0), (LT,), (JIF, 5), (7, 3), (2, 1), (6,)),
            "index 4 reads local 3",
        ),
    ],
)
def test_rust_validates_control_flow_on_its_own(program, message):
    # Bypass Python's validation entirely.
    with pytest.raises(ValueError, match=message):
        _native.compile_i64(program, 0)


def test_rust_accepts_well_formed_branches_on_its_own():
    # r = 1; if 0 < a: r = a; return r   (wire format, written by hand)
    program = (
        (2, 1),
        (8, 0),
        (2, 0),
        (1, 0),
        (LT,),
        (JIF, 8),
        (1, 0),
        (8, 0),
        (7, 0),
        (6,),
    )
    function = _native.compile_i64(program, 1)
    assert function.call((5,)) == (_native.STATUS_OK, 5)
    assert function.call((-5,)) == (_native.STATUS_OK, 1)
    assert function.try_call((I64_MIN,)) == 1


def test_malformed_control_flow_does_not_crash_the_interpreter():
    # Random programs over every opcode, run in a child process: a panic
    # would surface as PanicException, an abort or segfault as a non-zero
    # exit code, and a native loop that ignored its budget as a timeout.
    # Small targets make jumps, cycles and joins likely.
    code = """
import random
from speedysnek import _native
rng = random.Random(1)
opcodes = list(range(0, 18))
operands = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 2**32, 2**63 - 1]
compiled = 0
for _ in range(20000):
    program = []
    for _ in range(rng.randrange(1, 12)):
        opcode = rng.choice(opcodes)
        if rng.random() < 0.5:
            program.append((opcode, rng.choice(operands)))
        else:
            program.append((opcode,))
    try:
        function = _native.compile_i64(tuple(program), rng.choice([0, 1, 2]))
    except (TypeError, ValueError):
        continue
    compiled += 1
    for args in [(), (0,), (1, -1), (2**63 - 1, -(2**63))]:
        function.try_call(args)
print("ok", compiled)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        # Some random programs are loops; the backedge budget must end them.
        timeout=300,
    )
    assert result.stdout.startswith("ok")


# --- Ownership, lifetime and determinism ---------------------------------


def make_compiled(scale):
    # Every compiler helper, builder and intermediate tuple is local here
    # and is gone once this returns.
    program = (arg(0), const(scale), MULTIPLY, arg(1), ADD, RETURN)
    wire = serialize(program)
    function = _native.compile_i64(wire, 2)
    del wire
    return CompiledI64(function, program)


def test_code_outlives_compiler_helpers():
    compiled = make_compiled(7)
    gc.collect()
    assert compiled(3, 4) == 25


def test_native_function_owns_its_code():
    compiled = compile_i64(binary(SUBTRACT))
    function = compiled._function
    del compiled
    gc.collect()
    assert function.call((10, 3)) == (_native.STATUS_OK, 7)


def test_independent_programs_with_churn():
    kept = {}
    for scale in range(300):
        compiled = make_compiled(scale)
        if scale % 3 == 0:
            kept[scale] = compiled
        # The rest are freed now, possibly letting later code reuse memory.
    gc.collect()
    for scale in range(300, 400):
        assert make_compiled(scale)(1, 1) == scale + 1
    for scale, compiled in kept.items():
        assert compiled(2, 5) == 2 * scale + 5


def test_repeated_execution_is_deterministic():
    compiled = compile_i64(analyze(reuse).instructions)
    first = compiled(12345, -678, 2**40)
    assert all(compiled(12345, -678, 2**40) == first for _ in range(1000))
    assert first == reuse(12345, -678, 2**40)


def test_repeated_compilation_is_deterministic():
    program = analyze(affine).instructions
    results = {compile_i64(program)(9, -4) for _ in range(20)}
    assert results == {affine(9, -4)}


def test_calls_from_several_threads():
    compiled = compile_i64(analyze(affine).instructions)
    errors = []

    def worker(seed):
        rng = random.Random(seed)
        for _ in range(500):
            a, b = rng.randint(-(2**20), 2**20), rng.randint(-(2**20), 2**20)
            if compiled(a, b) != affine(a, b):
                errors.append((a, b))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_inputs_are_not_mutated():
    program = [arg(0), arg(1), MULTIPLY, RETURN]
    before = list(program)
    compiled = compile_i64(program)
    program.append(RETURN)  # Changing the list later has no effect.
    assert program[:-1] == before
    assert compiled(6, 7) == 42
    assert compiled.instructions == tuple(before)


def test_does_not_call_the_original_function():
    def f(a, b):
        return a * b + 1

    compiled = compile_i64(analyze(f).instructions)

    def boom(a, b):
        raise AssertionError("the original function was called")

    f.__code__ = boom.__code__
    called = []

    def profiler(frame, event, _arg):
        if event == "call":
            called.append(frame.f_code)

    sys.setprofile(profiler)
    try:
        result = compiled(6, 7)
    finally:
        sys.setprofile(None)

    assert result == 43
    assert boom.__code__ not in called


# --- Failure categories and privacy --------------------------------------


def test_failure_categories_are_distinct():
    classes = [
        NativeGuardFailure,
        UnsupportedProgramError,
        InvalidProgramError,
        GuardFailure,
    ]
    for cls in classes:
        assert cls.__bases__ == (Exception,)
    assert len(set(classes)) == len(classes)
    assert issubclass(_native.CodegenError, RuntimeError)


def test_guard_reasons_are_stable_strings():
    assert GuardReason.OVERFLOW == "overflow"
    assert GuardReason.WRONG_ARITY == "wrong_arity"
    assert GuardReason.NOT_EXACT_INT == "not_exact_int"
    assert GuardReason.OUT_OF_I64_RANGE == "out_of_i64_range"


def test_backend_is_private():
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
    for name in ("compile_i64", "CompiledI64", "NativeGuardFailure", "I64Function"):
        assert not hasattr(speedysnek, name)


def test_importing_speedysnek_does_not_load_the_backend():
    code = "import sys, speedysnek; print('speedysnek._compiler' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_module_doctests():
    failures, tests = doctest.testmod(_compiler)
    assert tests > 0
    assert failures == 0
