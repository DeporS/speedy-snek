"""Tests for the private reference interpreter.

The interpreter is the correctness oracle for future native code, so most
tests compare it against CPython running the original function.
"""

import doctest
import random
import subprocess
import sys

import pytest

import speedysnek
from speedysnek import _interpreter
from speedysnek._frontend import Instruction, Op, analyze
from speedysnek._interpreter import (
    GuardFailure,
    InvalidProgramError,
    execute,
    validate,
)

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


def instructions_of(func):
    analysis = analyze(func)
    assert analysis.supported, analysis.rejection
    return analysis.instructions


def run(func, *args):
    """Analyze ``func``, execute its instructions and check against CPython."""
    result = execute(instructions_of(func), args)
    assert type(result) is int
    assert result == func(*args)
    return result


# --- Direct execution ----------------------------------------------------


def test_doc_example():
    program = (arg(0), const(2), MULTIPLY, RETURN)
    assert execute(program, args=(5,)) == 10


def test_add():
    assert execute((arg(0), arg(1), ADD, RETURN), (2, 3)) == 5


def test_subtract_operand_order():
    # Pops the right operand first: LOAD_ARG 0; LOAD_ARG 1; SUBTRACT is a - b.
    assert execute((arg(0), arg(1), SUBTRACT, RETURN), (10, 3)) == 7
    assert execute((arg(1), arg(0), SUBTRACT, RETURN), (10, 3)) == -7
    assert execute((const(1), arg(0), SUBTRACT, RETURN), (10,)) == -9


def test_multiply():
    assert execute((arg(0), arg(1), MULTIPLY, RETURN), (6, -7)) == -42


def test_combined_expression():
    # (a + 2) * b - 1000
    program = (arg(0), const(2), ADD, arg(1), MULTIPLY, const(1000), SUBTRACT, RETURN)
    assert execute(program, (3, 4)) == -980


def test_nested_right_operand():
    # a - (b - c) needs a stack depth of three.
    program = (arg(0), arg(1), arg(2), SUBTRACT, SUBTRACT, RETURN)
    assert execute(program, (10, 4, 1)) == 7


def test_constants():
    assert execute((const(42), RETURN), ()) == 42
    assert execute((const(-7), const(3), MULTIPLY, RETURN), ()) == -21
    assert execute((const(0), RETURN), (1, 2)) == 0


def test_reused_arguments():
    # a * a - a
    program = (arg(0), arg(0), MULTIPLY, arg(0), SUBTRACT, RETURN)
    assert execute(program, (5,)) == 20


def test_identity():
    assert execute((arg(0), RETURN), (-3,)) == -3


def test_unread_arguments_are_ignored():
    # def f(a, b): return b -- CPython never looks at ``a``.
    assert execute((arg(1), RETURN), ("not an int", 5)) == 5
    assert execute((arg(0), RETURN), (5, 1.5, None)) == 5


def test_store_and_load_local():
    # x = a + b; return x * x
    program = (arg(0), arg(1), ADD, store(0), load(0), load(0), MULTIPLY, RETURN)
    assert execute(program, (3, 4)) == 49


def test_store_copies_the_value():
    # x = a; y = x; x = b; return y - x -- ``y`` keeps the old value of ``x``.
    program = (arg(0), store(0), load(0), store(1), arg(1), store(0))
    program += (load(1), load(0), SUBTRACT, RETURN)
    assert execute(program, (10, 3)) == 7


def test_reassigned_argument_slot_and_original_argument():
    # a2 = a + 1; return a2 * a -- LOAD_ARG still reads the original ``a``.
    program = (arg(0), const(1), ADD, store(0), load(0), arg(0), MULTIPLY, RETURN)
    assert execute(program, (5,)) == 30


def test_overwritten_local():
    program = (arg(0), const(1), ADD, store(0), const(0), store(0), load(0), RETURN)
    assert execute(program, (2**63 - 1,)) == 0


def test_sparse_local_slots():
    # Hand-written IR may use any non-negative slot numbers.
    program = (const(7), store(10**9), load(10**9), RETURN)
    assert execute(program, ()) == 7


def test_unread_reassigned_argument_is_not_guarded():
    # def f(a): a = 1; return a -- CPython never looks at the old ``a``.
    program = (const(1), store(0), load(0), RETURN)
    assert validate(program) == ()
    assert execute(program, ("not an int",)) == 1


def test_accepts_lists_and_other_iterables():
    assert execute([arg(0), arg(1), ADD, RETURN], [2, 3]) == 5
    assert execute(iter([arg(0), RETURN]), iter([9])) == 9


# --- Frontend plus interpreter -------------------------------------------


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def rsub(a, b):
    return b - a


def mul(a, b):
    return a * b


def identity(a):
    return a


def constant():
    return 42


def affine(a, b):
    return (a + 2) * b - 1000


def reuse(a, b, c):
    return c - a * a + b


def cubic(a):
    return a * a * a - 3 * a + 7


def mixed(a, b, c):
    return a * b - b * c + c * a - (a - b) * (b - c)


def nested(a, b, c, d):
    return a - (b - (c - d))


def big_constants(a, b):
    # 10**30 and 2**70 as literals: CPython only folds ``**`` when its size
    # estimate is small, and an unfolded ``**`` is rejected by the frontend.
    return (
        a * 1000000000000000000000000000000
        - b * -1180591620717411303424
        + 12345678901234567890
    )


def ignores_first(a, b):
    return b * 2


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


def overwritten(a):
    value = a + 1
    value = 0
    return value


def unused(a, b):
    t = a * b  # noqa: F841 -- computed, never read.
    return a - b


def swap_through_temporary(a, b, c):
    t = a
    a = b
    b = t
    c = c * a - b
    return a * 1000 + b * 10 + c


SHAPES = [
    add,
    sub,
    rsub,
    mul,
    identity,
    constant,
    affine,
    reuse,
    cubic,
    mixed,
    nested,
    big_constants,
    ignores_first,
    calculate_score,
    reuse_local,
    reassign,
    overwrite_argument,
    copy_value,
    overwritten,
    unused,
    swap_through_temporary,
]


def arity(func):
    return func.__code__.co_argcount


@pytest.mark.parametrize("func", SHAPES, ids=lambda f: f.__name__)
def test_end_to_end(func):
    args = (3, -5, 7, 11)[: arity(func)]
    run(func, *args)


def test_end_to_end_with_frontend_example():
    def f(a, b):
        return (a + 2) * b - 3 + 1000 * -7

    assert run(f, 4, 5) == 30 - 3 - 7000


# --- Differential testing ------------------------------------------------

# Values where fixed-width integer code typically goes wrong.
EDGE_VALUES = [
    0,
    1,
    -1,
    2,
    255,
    256,
    2**31 - 1,
    2**31,
    -(2**31),
    2**32,
    2**63 - 1,
    2**63,
    -(2**63),
    -(2**63) - 1,
    2**64 - 1,
    2**64,
    2**100,
    -(2**100),
]


def random_int(rng):
    """Mostly small integers, sometimes huge ones, sometimes edge values."""
    kind = rng.randrange(4)
    if kind == 0:
        return rng.randint(-1000, 1000)
    if kind == 1:
        return rng.randint(-(2**64), 2**64)
    if kind == 2:
        return rng.randint(-(2**200), 2**200)
    return rng.choice(EDGE_VALUES)


@pytest.mark.parametrize("func", SHAPES, ids=lambda f: f.__name__)
def test_differential_random_inputs(func):
    # Seeded per shape, so every run uses the same inputs.
    rng = random.Random(f"speedysnek-{func.__name__}")
    instructions = instructions_of(func)
    for _ in range(300):
        args = tuple(random_int(rng) for _ in range(arity(func)))
        assert execute(instructions, args) == func(*args), args


@pytest.mark.parametrize("func", SHAPES, ids=lambda f: f.__name__)
def test_differential_edge_values(func):
    # Every edge value in every argument position, with the others fixed.
    instructions = instructions_of(func)
    n = arity(func)
    for position in range(n):
        for value in EDGE_VALUES:
            args = tuple(value if i == position else 2**63 - 1 for i in range(n))
            assert execute(instructions, args) == func(*args), args


# --- Large integers ------------------------------------------------------


def wrap_i64(value):
    """What two's-complement 64-bit arithmetic would produce."""
    return (value + 2**63) % 2**64 - 2**63


def test_no_i64_wraparound_on_add():
    result = execute((arg(0), const(1), ADD, RETURN), (2**63 - 1,))
    assert result == 2**63
    assert wrap_i64(result) == -(2**63)  # What i64 arithmetic would give.


def test_no_i64_wraparound_on_subtract():
    result = execute((arg(0), const(1), SUBTRACT, RETURN), (-(2**63),))
    assert result == -(2**63) - 1
    assert wrap_i64(result) == 2**63 - 1


def test_no_i64_wraparound_on_multiply():
    program = (arg(0), arg(1), MULTIPLY, RETURN)
    assert execute(program, (2**63, 2**63)) == 2**126
    assert execute(program, (-(2**63), 2**63)) == -(2**126)
    assert execute(program, (2**100, 2**100)) == 2**200
    assert execute(program, (-(2**100), 3)) == -3 * 2**100


def test_large_constants_and_results():
    assert run(big_constants, 2**100, -(2**63)) == big_constants(2**100, -(2**63))
    assert run(cubic, 2**100) == 2**300 - 3 * 2**100 + 7
    assert run(mul, 10**50, -(10**50)) == -(10**100)


def test_large_constant_from_ir():
    assert execute((const(2**100), const(2**100), MULTIPLY, RETURN), ()) == 2**200


# --- Guard failures ------------------------------------------------------


class Number:
    """Records any attempt to treat it as a number."""

    calls = []

    def __getattribute__(self, name):
        Number.calls.append(name)
        return object.__getattribute__(self, name)

    def __add__(self, other):
        Number.calls.append("__add__")
        return 0

    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __add__

    def __index__(self):
        Number.calls.append("__index__")
        return 0

    def __int__(self):
        Number.calls.append("__int__")
        return 0

    def __repr__(self):
        Number.calls.append("__repr__")
        return "Number()"


class MyInt(int):
    pass


@pytest.mark.parametrize(
    "value",
    [True, False, 1.0, 2.5, "1", None, Number(), MyInt(3), 1j],
    ids=lambda v: type(v).__name__,
)
def test_guard_rejects_non_int(value):
    program = (arg(0), const(1), ADD, RETURN)
    with pytest.raises(GuardFailure, match=f"argument 0 is a {type(value).__name__}"):
        execute(program, (value,))


def test_guard_reports_the_failing_argument():
    program = (arg(0), arg(2), ADD, RETURN)
    with pytest.raises(GuardFailure) as excinfo:
        execute(program, (1, 2, 3.0))
    assert excinfo.value.arg_index == 2


def test_guard_does_not_touch_user_objects():
    Number.calls.clear()
    with pytest.raises(GuardFailure):
        execute((arg(0), arg(1), ADD, RETURN), (1, Number()))
    with pytest.raises(GuardFailure):
        execute((arg(0), RETURN), (Number(),))
    assert Number.calls == []


def test_guard_rejects_missing_arguments():
    with pytest.raises(GuardFailure, match="reads argument 0") as excinfo:
        execute((arg(0), RETURN), ())
    assert excinfo.value.arg_index == 0

    with pytest.raises(GuardFailure, match="reads argument 2") as excinfo:
        execute((arg(0), arg(2), ADD, RETURN), (1, 2))
    assert excinfo.value.arg_index == 2


def test_guards_run_before_execution_in_argument_order():
    # Both arguments fail; the lowest index is reported, whatever the
    # order in which the program reads them.
    with pytest.raises(GuardFailure) as excinfo:
        execute((arg(1), arg(0), ADD, RETURN), ("x", 1.0))
    assert excinfo.value.arg_index == 0


def test_bool_rejected_end_to_end():
    # CPython happily computes True + 1; the exact-int specialization
    # does not apply, so a JIT would fall back.
    assert add(True, 1) == 2
    with pytest.raises(GuardFailure):
        execute(analyze(add).instructions, (True, 1))


# --- Invalid programs ----------------------------------------------------


@pytest.mark.parametrize(
    ("program", "message", "index"),
    [
        ((), "program is empty", None),
        ([], "program is empty", None),
        ((arg(0), "RETURN"), "entry at index 1 is a str", 1),
        ((Op.LOAD_ARG, RETURN), "entry at index 0 is a Op", 0),
        ((None,), "entry at index 0 is a NoneType", 0),
        ((ADD, RETURN), "stack underflow at index 0: ADD needs 2", 0),
        ((arg(0), SUBTRACT, RETURN), "stack underflow at index 1", 1),
        ((const(1), MULTIPLY, RETURN), "stack underflow at index 1", 1),
        ((RETURN,), "stack underflow at index 0: RETURN needs 1", 0),
        ((arg(0),), "program has no RETURN", None),
        ((arg(0), arg(1), ADD), "program has no RETURN", None),
        # Code after a RETURN that nothing jumps to can never run.
        ((arg(0), RETURN, const(1)), "instruction at index 2 is unreachable", 2),
        ((arg(0), RETURN, arg(1), arg(2), ADD), "index 2 is unreachable", 2),
        ((arg(0), RETURN, arg(0), RETURN), "index 2 is unreachable", 2),
        ((RETURN, RETURN, RETURN), "stack underflow at index 0: RETURN needs 1", 0),
        ((arg(0), arg(1), RETURN), "leaves 1 extra value", 2),
        ((const(1), const(2), const(3), ADD, RETURN), "leaves 1 extra value", 4),
        ((arg(0), arg(1), arg(2), RETURN), "leaves 2 extra value", 3),
        # Local variables.
        (
            (load(0), RETURN),
            "LOAD_LOCAL at index 0 reads local 0 before it is assigned",
            0,
        ),
        ((arg(0), store(0), load(1), RETURN), "reads local 1 before it is assigned", 2),
        ((load(0), arg(0), store(0), RETURN), "reads local 0 before", 0),
        # A store later in the program does not help an earlier read.
        ((arg(0), load(3), ADD, const(1), store(3), RETURN), "reads local 3", 1),
        (
            (store(0), const(1), RETURN),
            "stack underflow at index 0: STORE_LOCAL needs 1",
            0,
        ),
        (
            (arg(0), store(0), store(1), load(0), RETURN),
            "stack underflow at index 2",
            2,
        ),
        ((arg(0), store(0), RETURN), "stack underflow at index 2: RETURN needs 1", 2),
        ((arg(0), store(0)), "program has no RETURN", None),
        ((arg(0), store(0), load(0), load(0), RETURN), "leaves 1 extra value", 4),
        ((arg(0), RETURN, store(0)), "index 2 is unreachable", 2),
    ],
)
def test_invalid_programs(program, message, index):
    with pytest.raises(InvalidProgramError, match=message) as excinfo:
        execute(program, (1, 2, 3))
    assert excinfo.value.index == index
    with pytest.raises(InvalidProgramError, match=message):
        validate(program)


def test_non_iterable_program():
    with pytest.raises(InvalidProgramError, match="must be iterable"):
        execute(None, ())
    with pytest.raises(InvalidProgramError, match="must be iterable"):
        execute(arg(0), ())


def corrupt(instruction, **fields):
    """Bypass ``Instruction``'s own validation, as buggy code could."""
    for name, value in fields.items():
        object.__setattr__(instruction, name, value)
    return instruction


@pytest.mark.parametrize(
    ("instruction", "message"),
    [
        (corrupt(arg(0), arg=-1), "negative argument index -1"),
        (corrupt(arg(0), arg=None), "LOAD_ARG at index 0 needs an int"),
        (corrupt(arg(0), arg=True), "needs an int argument, got bool"),
        (corrupt(const(1), arg=1.5), "LOAD_CONST at index 0 needs an int"),
        (corrupt(const(1), op="LOAD_CONST"), "unknown operation 'LOAD_CONST'"),
        (corrupt(const(1), op=Op.ADD), "ADD at index 0 takes no argument"),
        (corrupt(load(0), arg=-1), "LOAD_LOCAL at index 0 has negative local index"),
        (corrupt(load(0), arg=None), "LOAD_LOCAL at index 0 needs an int"),
        (corrupt(load(0), arg=False), "needs an int argument, got bool"),
        (corrupt(load(0), arg=1.0), "needs an int argument, got float"),
        (corrupt(store(0), arg=-2), "STORE_LOCAL at index 0 has negative local"),
        (corrupt(store(0), arg="0"), "STORE_LOCAL at index 0 needs an int"),
        (corrupt(store(0), arg=True), "needs an int argument, got bool"),
    ],
)
def test_corrupted_instructions(instruction, message):
    with pytest.raises(InvalidProgramError, match=message):
        execute((instruction, RETURN), (1,))


def test_instruction_subclasses_are_rejected():
    class Sneaky(Instruction):
        __slots__ = ()

    with pytest.raises(InvalidProgramError, match="is a Sneaky"):
        execute((Sneaky(Op.LOAD_CONST, 1), RETURN), ())


def test_validation_runs_before_guards():
    # The arguments would fail their guards, but the program is checked first.
    with pytest.raises(InvalidProgramError):
        execute((arg(0), ADD, RETURN), ("x",))
    with pytest.raises(InvalidProgramError):
        execute((arg(5), arg(0), RETURN), ())


def test_validate_reports_read_arguments():
    assert validate((const(1), RETURN)) == ()
    assert validate((arg(3), arg(0), ADD, arg(3), MULTIPLY, RETURN)) == (0, 3)
    assert validate(analyze(reuse).instructions) == (0, 1, 2)
    assert validate(analyze(ignores_first).instructions) == (1,)
    # Local slots are not arguments; a reassigned argument read only
    # afterwards is not read as an argument at all.
    assert validate(analyze(copy_value).instructions) == (0, 1)
    assert validate((arg(1), store(0), load(0), RETURN)) == (1,)


def test_error_categories_are_distinct():
    assert not issubclass(GuardFailure, InvalidProgramError)
    assert not issubclass(InvalidProgramError, GuardFailure)
    # Neither is a built-in error that user code might catch by accident.
    for error in (GuardFailure, InvalidProgramError):
        assert error.__bases__ == (Exception,)


# --- Conditionals --------------------------------------------------------

LT = Instruction(Op.LESS_THAN)

COMPARISONS = {
    Op.LESS_THAN: "<",
    Op.LESS_EQUAL: "<=",
    Op.GREATER_THAN: ">",
    Op.GREATER_EQUAL: ">=",
    Op.EQUAL: "==",
    Op.NOT_EQUAL: "!=",
}


def jump(target):
    return Instruction(Op.JUMP, target)


def jump_if_false(target):
    return Instruction(Op.JUMP_IF_FALSE, target)


def pick(op):
    """``return 1 if a <op> b else 0`` as IR, one RETURN per branch."""
    return (
        arg(0),
        arg(1),
        Instruction(op),
        jump_if_false(6),
        const(1),
        RETURN,
        const(0),
        RETURN,
    )


@pytest.mark.parametrize("op", list(COMPARISONS), ids=lambda op: op.name)
def test_comparisons_are_exact_for_any_size(op):
    program = pick(op)
    values = [-(2**100), -(2**63) - 1, -(2**63), -1, 0, 1, 2**63 - 1, 2**63, 2**100]
    for a in values:
        for b in values:
            expected = int(eval(f"a {COMPARISONS[op]} b"))
            result = execute(program, (a, b))
            assert type(result) is int and result == expected, (op, a, b)


def test_only_the_selected_branch_runs():
    # ``r = a; if a < b: r += 1 else: r *= b; return r``.
    def selected_branch(a, b):
        result = a
        if a < b:
            result += 1
        else:
            result *= b
        return result

    for args in [(1, 2), (2, 1), (2, 2), (-(2**70), 3), (2**70, 3)]:
        run(selected_branch, *args)


def calculate_score(wins, penalties):
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


def nested(a, b, c):
    r = a
    if a < b:
        if b <= c:
            r = c - a
        else:
            r += 1
    elif a == c:
        b = 0
    if b != 0:
        r *= b
    return r + b


@pytest.mark.parametrize(
    "func", [calculate_score, choose, adjust, nested], ids=lambda f: f.__name__
)
def test_conditionals_match_python(func):
    arity = func.__code__.co_argcount
    values = [-(2**64), -3, -1, 0, 1, 2, 3, 2**64]
    rng = random.Random(f"interpreter-{func.__name__}")
    for _ in range(300):
        run(func, *(rng.choice(values) for _ in range(arity)))


def test_reads_on_any_branch_are_guarded():
    program = (arg(0), const(0), LT, jump_if_false(6), arg(1), RETURN, arg(2), RETURN)
    assert validate(program) == (0, 1, 2)
    # Arguments read only on a branch not taken are still guarded.
    with pytest.raises(GuardFailure) as excinfo:
        execute(program, (1, 2, "x"))
    assert excinfo.value.arg_index == 2


def test_values_join_through_locals():
    # r = 10; if a < 0: r = a; return r   (r keeps 10 on the other path)
    program = (
        const(10),
        store(0),
        arg(0),
        const(0),
        LT,
        jump_if_false(8),
        arg(0),
        store(0),
        load(0),
        RETURN,
    )
    assert execute(program, (-5,)) == -5
    assert execute(program, (5,)) == 10


@pytest.mark.parametrize(
    ("program", "message", "index"),
    [
        # Targets.
        (
            (jump(0), const(1), RETURN),
            "loop at index 0 must start with its condition",
            0,
        ),
        ((const(1), jump(0), RETURN), "must start with its condition", 1),
        ((const(1), store(0), jump(1), RETURN), "loop at index 1 must start", 2),
        ((jump(3), const(1), RETURN), "jumps to index 3, past the end", 0),
        ((arg(0), arg(0), LT, jump_if_false(9), const(1), RETURN), "past the end", 3),
        # Reachability and termination.
        ((jump(2), const(1), const(2), RETURN), "index 1 is unreachable", 1),
        ((const(1), RETURN, const(2), RETURN), "index 2 is unreachable", 2),
        (
            (arg(0), arg(0), LT, jump_if_false(6), const(1), RETURN, const(2)),
            "index 6 falls off the end",
            6,
        ),
        # Conditions.
        ((arg(0), arg(0), LT, RETURN), "LESS_THAN at index 2 must be immediately", 2),
        # A condition cannot be stored: it must go straight to the jump.
        (
            (arg(0), arg(0), LT, store(0), const(1), RETURN),
            "LESS_THAN at index 2 must be immediately followed by JUMP_IF_FALSE",
            2,
        ),
        (
            (arg(0), arg(0), LT, const(1), ADD, jump_if_false(7), const(1), RETURN),
            "must be immediately followed",
            2,
        ),
        (
            (arg(0), jump_if_false(3), const(1), const(2), RETURN),
            "JUMP_IF_FALSE at index 1 needs a condition but found an integer",
            1,
        ),
        (
            (jump_if_false(2), const(1), RETURN),
            "stack underflow at index 0: JUMP_IF_FALSE",
            0,
        ),
        (
            (arg(0), LT, jump_if_false(4), const(1), RETURN),
            "stack underflow at index 1",
            1,
        ),
        # The stack must be empty at jumps and joins.
        ((const(1), jump(2), RETURN), "JUMP at index 1 leaves 1 extra value", 1),
        (
            (const(9), arg(0), arg(0), LT, jump_if_false(6), const(1), RETURN),
            "JUMP_IF_FALSE at index 4 leaves 1 extra",
            4,
        ),
        (
            (arg(0), arg(0), LT, jump_if_false(5), const(1), const(2), ADD, RETURN),
            "jump target at index 5 is reached from index 4 with 1 value",
            5,
        ),
        # Locals must be assigned on every path.
        (
            (arg(0), arg(0), LT, jump_if_false(6), const(1), store(0), load(0), RETURN),
            "LOAD_LOCAL at index 6 reads local 0 before it is assigned on every path",
            6,
        ),
        (
            (
                arg(0),
                arg(0),
                LT,
                jump_if_false(7),
                const(1),
                store(0),
                jump(9),
                const(2),
                store(1),
                load(0),
                RETURN,
            ),
            "index 9 reads local 0",
            9,
        ),
        # Corrupted targets.
        (
            (corrupt(jump(1), arg=-1), RETURN),
            "JUMP at index 0 has negative target index",
            0,
        ),
        ((corrupt(jump(1), arg=None), RETURN), "JUMP at index 0 needs an int", 0),
        (
            (corrupt(jump_if_false(1), arg=True), RETURN),
            "needs an int argument, got bool",
            0,
        ),
        (
            (corrupt(Instruction(Op.LESS_THAN), arg=1), RETURN),
            "LESS_THAN at index 0 takes no argument",
            0,
        ),
    ],
)
def test_invalid_control_flow(program, message, index):
    with pytest.raises(InvalidProgramError, match=message) as excinfo:
        validate(program)
    assert excinfo.value.index == index
    with pytest.raises(InvalidProgramError, match=message):
        execute(program, (1,))


def test_validation_covers_every_path_before_running():
    # The malformed branch is never taken for these arguments, but the
    # whole program is still rejected before anything runs.
    program = (arg(0), const(0), LT, jump_if_false(6), ADD, RETURN, arg(0), RETURN)
    with pytest.raises(InvalidProgramError, match="stack underflow at index 4"):
        execute(program, (5,))


# --- Determinism and side effects ----------------------------------------


def test_repeated_execution_is_deterministic():
    instructions = instructions_of(mixed)
    args = (2**70, -3, 12345)
    first = execute(instructions, args)
    assert all(execute(instructions, args) == first for _ in range(100))
    assert first == mixed(*args)


def test_inputs_are_not_mutated():
    program = [arg(0), arg(1), MULTIPLY, arg(0), ADD, RETURN]
    program_before = list(program)
    args = [6, 7]
    assert execute(program, args) == 48
    assert program == program_before
    assert args == [6, 7]


def test_execute_does_not_call_the_original_function():
    def f(a, b):
        return a * b + 1

    instructions = analyze(f).instructions

    # Make any call to ``f`` fail loudly, then watch every Python-level call
    # made while executing.
    def boom(a, b):
        raise AssertionError("the original function was called")

    f.__code__ = boom.__code__
    called_code = []

    def profiler(frame, event, _arg):
        if event == "call":
            called_code.append(frame.f_code)

    sys.setprofile(profiler)
    try:
        result = execute(instructions, (6, 7))
    finally:
        sys.setprofile(None)

    assert result == 43
    assert f.__code__ not in called_code
    assert boom.__code__ not in called_code
    with pytest.raises(AssertionError, match="original function was called"):
        f(6, 7)


# --- Packaging -----------------------------------------------------------


def test_interpreter_is_private():
    for name in ("execute", "validate", "GuardFailure", "InvalidProgramError"):
        assert name not in speedysnek.__all__
        assert not hasattr(speedysnek, name)


def test_importing_speedysnek_does_not_load_the_interpreter():
    code = "import sys, speedysnek; print('speedysnek._interpreter' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_module_doctests():
    failures, tests = doctest.testmod(_interpreter)
    assert tests > 0
    assert failures == 0
