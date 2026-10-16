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


def run_i64(program, args):
    """An independent i64 oracle: Python ints, checked after every operation.

    Returns ``("ok", value)`` or ``("overflow", instruction_index)``.
    """
    stack = []
    for index, instruction in enumerate(program):
        if instruction.op is Op.LOAD_ARG:
            stack.append(args[instruction.arg])
        elif instruction.op is Op.LOAD_CONST:
            stack.append(instruction.arg)
        elif instruction.op is Op.RETURN:
            return ("ok", stack.pop())
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
    assert serialize(iter(program)) == serialize(list(program))
    with pytest.raises(UnsupportedProgramError):
        serialize((const(2**63), RETURN))


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
values = [0, 1, 2, 3, 4, 5, 6, 7, -1, 2**63, -(2**63) - 1, True, None, 1.5, "x", ()]
for _ in range(5000):
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
    assert speedysnek.__all__ == ["jit"]
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
