"""Tests for ``while`` loops: frontend shapes, validation, the reference
interpreter, native code, the backedge budget and ``@snekit``'s fallback.

See ``docs/loops.md`` for the design. Execution paths are proven as in
``test_jit_compile.py``: native calls with the compiled function's
``try_calls`` counter, Python calls with ``sys.setprofile``. Anything that
could run forever runs in a child process with a timeout.
"""

import collections
import contextlib
import dis
import random
import subprocess
import sys
import textwrap

import pytest
from test_compiler import run_i64

from speedysnek import _compiler, _native, snekit
from speedysnek._compiler import (
    I64_MAX,
    GuardReason,
    NativeGuardFailure,
    compile_i64,
)
from speedysnek._decorator import _WARMUP_CALLS, _DisabledReason, _Status
from speedysnek._frontend import Instruction, Op, RejectionReason, analyze
from speedysnek._interpreter import (
    DEFAULT_MAX_BACKEDGES,
    ExecutionLimitExceeded,
    InvalidProgramError,
    execute,
    validate,
)

RETURN = Instruction(Op.RETURN)
ADD = Instruction(Op.ADD)
SUBTRACT = Instruction(Op.SUBTRACT)
MULTIPLY = Instruction(Op.MULTIPLY)
LESS_THAN = Instruction(Op.LESS_THAN)
CMP_OPS = [
    Op.LESS_THAN,
    Op.LESS_EQUAL,
    Op.GREATER_THAN,
    Op.GREATER_EQUAL,
    Op.EQUAL,
    Op.NOT_EQUAL,
]


def arg(index):
    return Instruction(Op.LOAD_ARG, index)


def const(value):
    return Instruction(Op.LOAD_CONST, value)


def load(slot):
    return Instruction(Op.LOAD_LOCAL, slot)


def store(slot):
    return Instruction(Op.STORE_LOCAL, slot)


def jump(target):
    return Instruction(Op.JUMP, target)


def jump_if_false(target):
    return Instruction(Op.JUMP_IF_FALSE, target)


def define(source, name="f"):
    """Compile one function from source and return it, without calling it."""
    namespace = {}
    exec(textwrap.dedent(source), namespace)
    return namespace[name]


def run_child(code, timeout=60):
    """Run ``code`` in a fresh interpreter; return its stdout."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


# --- The functions this feature is for --------------------------------------


def sum_squares(n):
    total = 0
    i = 0
    while i < n:
        total += i * i
        i += 1
    return total


def countdown(n):
    total = 0
    while n > 0:
        total += n
        n -= 1
    return total


def conditional_sum(n, cutoff):
    total = 0
    i = 0
    while i < n:
        if i < cutoff:
            total += i
        else:
            total += i * 2
        i += 1
    return total


def possibly_unbound(n):
    i = 0
    while i < n:
        value = i
        i += 1
    return value


def last_copy(n):
    # ``last`` copies ``i``, which then changes.
    last = -1
    i = 0
    while i < n:
        last = i
        i += 1
    return last * 100 + i


def fibonacci(n):
    a = 0
    b = 1
    while n > 0:
        t = a + b
        a = b
        b = t
        n -= 1
    return a


def after_the_loop(n, k):
    total = 0
    i = 0
    while i < n:
        total += k
        i += 1
    if total > 10:
        return total * 2 + i
    return total - i


def trailing_if(n, cutoff):
    # An ``if`` without ``else`` ends the body: POP_JUMP_IF_TRUE.
    total = 0
    i = 0
    while i < n:
        i += 1
        if i < cutoff:
            total += i
    return total


def trailing_if_else(n, cutoff):
    # Two backedges, one per branch.
    total = 0
    i = 0
    while i < n:
        i += 1
        if i < cutoff:
            total += i
        else:
            total -= 1
    return total


def prefix_branch(n, start):
    if start < 0:
        start = 0
    while start < n:
        start += 2
    return start


def loop_in_if(a, n):
    if a < n:
        while a < n:
            a += 3
    return a


def mul_condition(n):
    i = 0
    while i * i < n:
        i += 1
    return i


# Functions and bounded arguments for the differential tests. Every case
# terminates in Python; some overflow i64.
SMALL = [-3, -1, 0, 1, 2, 3, 7, 20]
CORPUS = [
    (sum_squares, [(n,) for n in SMALL]),
    (countdown, [(n,) for n in SMALL]),
    (conditional_sum, [(n, c) for n in SMALL for c in (-1, 0, 3, 50)]),
    (last_copy, [(n,) for n in SMALL]),
    (fibonacci, [(n,) for n in SMALL + [91, 92, 93, 100]]),
    (after_the_loop, [(n, k) for n in SMALL for k in (0, 1, 5, 2**62)]),
    (trailing_if, [(n, c) for n in SMALL for c in (0, 2, 100)]),
    (trailing_if_else, [(n, c) for n in SMALL for c in (0, 2, 100)]),
    (prefix_branch, [(n, s) for n in SMALL for s in (-5, 0, 1, 4)]),
    (loop_in_if, [(a, n) for a in (-5, 0, 7) for n in SMALL]),
    (mul_condition, [(n,) for n in SMALL + [10**6, 10**8]]),
]


# --- Frontend: recognized shapes --------------------------------------------


def test_sum_squares_ir():
    assert analyze(sum_squares).instructions == (
        const(0),
        store(0),  # total
        const(0),
        store(1),  # i
        load(1),  # 4: the header
        arg(0),
        LESS_THAN,
        jump_if_false(19),  # The only exit.
        load(0),
        load(1),
        load(1),
        MULTIPLY,
        ADD,
        store(0),
        load(1),
        const(1),
        ADD,
        store(1),
        jump(4),  # The backedge.
        load(0),  # 19
        RETURN,
    )


def test_an_argument_updated_in_the_loop_is_copied_into_its_slot():
    # ``n`` may be the argument or a stored value at the header.
    assert analyze(countdown).instructions[:4] == (
        arg(0),
        store(1),
        const(0),
        store(0),
    )
    assert arg(0) not in analyze(countdown).instructions[2:]


def test_conditional_sum_ir():
    instructions = analyze(conditional_sum).instructions
    jumps = [
        (index, instruction)
        for index, instruction in enumerate(instructions)
        if instruction.op in (Op.JUMP, Op.JUMP_IF_FALSE)
    ]
    # The loop test, the if test, the jump over ``else`` to the join, and
    # one backedge after the join.
    assert jumps == [
        (7, jump_if_false(28)),
        (11, jump_if_false(17)),
        (16, jump(23)),
        (27, jump(4)),
    ]


def test_trailing_if_negates_its_comparison():
    instructions = analyze(trailing_if).instructions
    # ``if i < cutoff`` compiles to POP_JUMP_IF_TRUE over the backedge; the
    # IR tests ``i >= cutoff`` and falls into the backedge.
    position = instructions.index(Instruction(Op.GREATER_EQUAL))
    assert instructions[position + 1].op is Op.JUMP_IF_FALSE
    assert instructions[position + 2] == jump(4)
    assert Op.LESS_THAN in {i.op for i in instructions}  # The loop test.


@pytest.mark.parametrize("symbol", ["<", "<=", ">", ">=", "==", "!="])
def test_every_negated_comparison_is_exact(symbol):
    f = define(
        f"""
        def f(n, c):
            total = 0
            i = 0
            while i < n:
                i += 1
                if i {symbol} c:
                    total += i
            return total
        """
    )
    assert "POP_JUMP_IF_TRUE" in {i.opname for i in dis.get_instructions(f)}
    compiled = compile_i64(analyze(f).instructions, 2)
    for n in range(-1, 8):
        for c in range(-1, 9):
            assert compiled(n, c) == f(n, c), (n, c)


def test_trailing_if_else_has_two_backedges():
    instructions = analyze(trailing_if_else).instructions
    backedges = [
        i for i, x in enumerate(instructions) if x.op is Op.JUMP and x.arg <= i
    ]
    assert len(backedges) == 2
    assert {instructions[i].arg for i in backedges} == {4}


@pytest.mark.parametrize(
    "func",
    [
        sum_squares,
        countdown,
        conditional_sum,
        last_copy,
        fibonacci,
        after_the_loop,
        trailing_if,
        trailing_if_else,
        prefix_branch,
        loop_in_if,
        mul_condition,
    ],
)
def test_supported_loops_are_valid_ir(func):
    analysis = analyze(func)
    assert analysis.supported, analysis.rejection
    validate(analysis.instructions)


def _bytecode(func):
    return [
        (i.opname, i.arg)
        for i in dis.get_instructions(func)
        if i.opname not in ("RESUME", "NOP")
    ]


def test_continue_is_bytecode_identical_to_a_supported_if_else():
    # Documented in docs/loops.md: a ``continue`` is one more backedge.
    with_continue = define(
        """
        def f(n, c):
            t = 0
            i = 0
            while i < n:
                i += 1
                if i > c:
                    continue
                t += i
            return t
        """
    )
    with_else = define(
        """
        def f(n, c):
            t = 0
            i = 0
            while i < n:
                i += 1
                if i > c:
                    pass
                else:
                    t += i
            return t
        """
    )
    assert _bytecode(with_continue) == _bytecode(with_else)
    assert analyze(with_continue).instructions == analyze(with_else).instructions
    compiled = compile_i64(analyze(with_continue).instructions, 2)
    for n, c in [(0, 0), (5, 2), (10, 20), (10, -1)]:
        assert compiled(n, c) == with_continue(n, c) == with_else(n, c)


def test_continue_with_not_is_bytecode_identical_to_a_trailing_if():
    with_continue = define(
        """
        def f(n, c):
            t = 0
            i = 0
            while i < n:
                i += 1
                if not i < c:
                    continue
                t += 1
            return t
        """
    )
    with_if = define(
        """
        def f(n, c):
            t = 0
            i = 0
            while i < n:
                i += 1
                if i < c:
                    t += 1
            return t
        """
    )
    assert _bytecode(with_continue) == _bytecode(with_if)
    assert analyze(with_continue).instructions == analyze(with_if).instructions


def test_loop_else_without_break_is_code_after_the_loop():
    with_else = define(
        """
        def f(n):
            i = 0
            while i < n:
                i += 1
            else:
                i += 100
            return i
        """
    )
    after = define(
        """
        def f(n):
            i = 0
            while i < n:
                i += 1
            i += 100
            return i
        """
    )
    assert _bytecode(with_else) == _bytecode(after)
    assert analyze(with_else).supported


# --- Frontend: rejected shapes ----------------------------------------------


def assert_rejected(func, reason, opname, message):
    rejection = analyze(func).rejection
    assert rejection is not None
    assert (rejection.reason, rejection.opname) == (reason, opname), rejection
    assert message in rejection.message
    return rejection


@pytest.mark.parametrize(
    ("source", "reason", "opname", "message"),
    [
        (
            """
            def f(a):
                t = 0
                for i in a:
                    t += i
                return t
            """,
            RejectionReason.UNSUPPORTED_ITERATION,
            "FOR_ITER",
            "only for loops over range(n) are supported",
        ),
        (
            """
            def f(n):
                t = 0
                i = 0
                while i < n:
                    j = 0
                    while j < n:
                        t += j
                        j += 1
                    i += 1
                return t
            """,
            RejectionReason.CONTROL_FLOW,
            "JUMP_BACKWARD",
            "only one loop per function",
        ),
        (
            """
            def f(n):
                i = 0
                while i < n:
                    i += 1
                while i > 0:
                    i -= 2
                return i
            """,
            RejectionReason.CONTROL_FLOW,
            "JUMP_BACKWARD",
            "only one loop per function",
        ),
        (
            # A short tail after ``break`` is copied into the body.
            """
            def f(n):
                i = 0
                while i < n:
                    if i > 5:
                        break
                    i += 1
                return i
            """,
            RejectionReason.CONTROL_FLOW,
            "RETURN_VALUE",
            "return or break inside a while loop",
        ),
        (
            # A longer tail is jumped to.
            """
            def f(n):
                i = 0
                while i < n:
                    if i > 5:
                        break
                    i += 1
                i = i * 2 + 1
                i = i * 3 + 1
                i = i * 5 + 1
                return i
            """,
            RejectionReason.CONTROL_FLOW,
            "JUMP_FORWARD",
            "a jump out of a while loop",
        ),
        (
            """
            def f(n):
                i = 0
                while i < n:
                    if i > 5:
                        return i
                    i += 1
                return -1
            """,
            RejectionReason.CONTROL_FLOW,
            "RETURN_VALUE",
            "return or break inside a while loop",
        ),
        (
            """
            def f(n):
                while True:
                    n += 1
                return n
            """,
            RejectionReason.CONTROL_FLOW,
            "JUMP_BACKWARD",
            "must start with one comparison",
        ),
        (
            """
            def f(n):
                while n:
                    n -= 1
                return n
            """,
            RejectionReason.UNSUPPORTED_CONDITION,
            "POP_JUMP_IF_FALSE",
            "only a comparison can be an if or while condition",
        ),
        (
            """
            def f(a, n):
                while a < n and a < 100:
                    a += 1
                return a
            """,
            RejectionReason.CONTROL_FLOW,
            "POP_JUMP_IF_FALSE",
            "a jump out of a while loop",
        ),
        (
            """
            def f(a, n):
                while not a >= n:
                    a += 1
                return a
            """,
            RejectionReason.CONTROL_FLOW,
            "POP_JUMP_IF_TRUE",
            "must start with one comparison",
        ),
        (
            # ``or`` inside the body, not in the backedge shape.
            """
            def f(a, n):
                i = 0
                while i < n:
                    if i < a or i > 5:
                        a += 1
                    i += 1
                return a
            """,
            RejectionReason.CONTROL_FLOW,
            "POP_JUMP_IF_TRUE",
            "unsupported jump POP_JUMP_IF_TRUE",
        ),
        (
            # A loop ``else`` with ``break``: the break copies the tail.
            """
            def f(n):
                i = 0
                while i < n:
                    i += 1
                    if i > 3:
                        break
                else:
                    i += 100
                return i
            """,
            RejectionReason.CONTROL_FLOW,
            "RETURN_VALUE",
            "return or break inside a while loop",
        ),
        (
            """
            def f(n):
                i = 0
                while i < n:
                    i = i / 2 + 1
                return i
            """,
            RejectionReason.UNSUPPORTED_OPERATOR,
            "BINARY_OP",
            "unsupported binary operator '/'",
        ),
    ],
)
def test_rejected_loops(source, reason, opname, message):
    assert_rejected(define(source), reason, opname, message)


def test_possibly_unbound_after_zero_iterations_is_rejected():
    assert_rejected(
        possibly_unbound,
        RejectionReason.UNSUPPORTED_VARIABLE,
        "LOAD_FAST_CHECK",
        "'value' may be read before it is assigned",
    )


def test_a_store_later_in_the_body_does_not_initialize_the_first_iteration():
    # ``x`` is stored at the end of every iteration, but the first one
    # reads it before any store.
    f = define(
        """
        def f(n):
            t = 0
            i = 0
            while i < n:
                t += x
                x = i
                i += 1
            return t
        """
    )
    assert_rejected(
        f,
        RejectionReason.UNSUPPORTED_VARIABLE,
        "LOAD_FAST_CHECK",
        "'x' may be read",
    )
    # In the header, too.
    g = define(
        """
        def f(n):
            i = 0
            while i < y:
                y = n
                i += 1
            return i
        """
    )
    assert_rejected(g, RejectionReason.UNSUPPORTED_VARIABLE, "LOAD_FAST_CHECK", "'y'")


def test_a_value_stored_before_the_loop_is_assigned_after_it():
    f = define(
        """
        def f(n):
            value = -1
            i = 0
            while i < n:
                value = i
                i += 1
            return value
        """
    )
    compiled = compile_i64(analyze(f).instructions, 1)
    assert [compiled(n) for n in (-2, 0, 1, 5)] == [-1, -1, 0, 4]


@pytest.mark.parametrize("n", [-2, -1, 0, 1, 2, 5])
def test_possibly_unbound_keeps_python_behaviour(n):
    f = snekit(possibly_unbound)
    for _ in range(_WARMUP_CALLS + 3):
        try:
            expected = ("ok", possibly_unbound(n))
        except UnboundLocalError as error:
            expected = ("raised", type(error))
        try:
            outcome = ("ok", f(n))
        except UnboundLocalError as error:
            outcome = ("raised", type(error))
        assert outcome == expected
    assert (outcome[0] == "raised") == (n <= 0)
    assert f._jit_state.disabled_reason is _DisabledReason.REJECTED


# --- Validation of hand-written IR ------------------------------------------


# A valid loop, ``i = 0; while i < a: i += 1; return i``, to mutate:
#   0 LOAD_CONST 0      5 JUMP_IF_FALSE 11
#   1 STORE_LOCAL 0     6 LOAD_LOCAL 0
#   2 LOAD_LOCAL 0      7 LOAD_CONST 1
#   3 LOAD_ARG 0        8 ADD
#   4 LESS_THAN         9 STORE_LOCAL 0
#                      10 JUMP 2
#                      11 LOAD_LOCAL 0
#                      12 RETURN
LOOP = (
    const(0),
    store(0),
    load(0),
    arg(0),
    LESS_THAN,
    jump_if_false(11),
    load(0),
    const(1),
    ADD,
    store(0),
    jump(2),
    load(0),
    RETURN,
)


def replaced(program, **changes):
    program = list(program)
    for index, instruction in changes.items():
        program[int(index[1:])] = instruction
    return tuple(program)


def test_the_base_loop_is_valid():
    assert validate(LOOP) == (0,)
    assert execute(LOOP, (5,)) == 5
    assert compile_i64(LOOP)(5) == 5


# (program, Python message, Rust message)
MALFORMED_LOOPS = [
    # A backward conditional jump.
    (
        replaced(LOOP, i10=jump_if_false(2)),
        "JUMP_IF_FALSE at index 10 jumps back to index 2",
        "JUMP_IF_FALSE at index 10 goes back to index 2",
    ),
    # Two loops: two backedge targets.
    (
        (const(0), store(0), jump(1)) + LOOP[3:],
        "go to different targets (1 and 2)",
        "go to different targets (1 and 2)",
    ),
    (
        LOOP[:11] + (jump(3),) + LOOP[11:],
        "go to different targets (2 and 3)",
        "go to different targets (2 and 3)",
    ),
    # No test: the header's first jump is the backedge.
    (
        (const(0), store(0), load(0), const(1), ADD, store(0), jump(2), RETURN),
        "the loop at index 2 must start with its condition",
        "the loop at index 2 must start with its condition",
    ),
    # A self-loop.
    ((const(1), jump(1), RETURN), "loop at index 1 must start", "loop at index 1"),
    # A RETURN in the header, before any test.
    (
        (arg(0), RETURN, jump(0)),
        "the first jump or RETURN from there, at index 1",
        "the first jump or RETURN from there, at index 1",
    ),
    # The test exits into the body.
    (
        replaced(LOOP, i5=jump_if_false(9)),
        "is not after the loop's last backward jump",
        "is not after the loop's last backward jump",
    ),
    # A RETURN inside the body: ``if i < 5: return i``.
    (
        LOOP[:5]
        + (jump_if_false(17), load(0), const(5), LESS_THAN, jump_if_false(12))
        + (load(0), RETURN)
        + (load(0), const(1), ADD, store(0), jump(2), load(0), RETURN),
        "RETURN at index 11 is inside the loop",
        "RETURN at index 11 is inside the loop",
    ),
    # A jump from the body to the exit: ``if i < 5: ... else: break``.
    (
        LOOP[:5]
        + (jump_if_false(15), load(0), const(5), LESS_THAN, jump_if_false(15))
        + (load(0), const(1), ADD, store(0), jump(2), load(0), RETURN),
        "the jump at index 9 leaves the loop body",
        "the jump at index 9 leaves the loop body",
    ),
    # A jump from before the loop into its body.
    (
        (arg(0), arg(0), LESS_THAN, jump_if_false(9))
        + tuple(
            Instruction(i.op, i.arg + 4) if i.op in (Op.JUMP, Op.JUMP_IF_FALSE) else i
            for i in LOOP
        ),
        "the jump at index 3 enters the loop at index 9",
        "the jump at index 3 enters the loop at index 9",
    ),
    # The body falls through into the exit.
    (
        (
            const(0),
            store(0),
            load(0),
            arg(0),
            LESS_THAN,
            jump_if_false(9),
            jump(2),
            load(0),
            store(0),
            load(0),
            RETURN,
        ),
        "falls through from index 8 into the loop's exit at index 9",
        "falls through from index 8 into the loop's exit at index 9",
    ),
    # Entering the header with a value on the stack.
    (
        (const(7),)
        + tuple(
            Instruction(i.op, i.arg + 1) if i.op in (Op.JUMP, Op.JUMP_IF_FALSE) else i
            for i in LOOP
        )[:-1]
        + (ADD, RETURN),
        "the loop header at index 3 is reached from index 2 with 1 value",
        "the loop header at index 3 is reached from index 2 with 1 value",
    ),
    # A backedge with a value on the stack.
    (
        replaced(LOOP, i9=Instruction(Op.LOAD_LOCAL, 0)),
        "JUMP at index 10 leaves 2 extra value(s)",
        "instruction at index 10 leaves 2 value(s)",
    ),
    # The header is only reachable through its backedge.
    (
        (arg(0), RETURN) + LOOP[2:],
        "instruction at index 2 is unreachable",
        "instruction at index 2 is unreachable",
    ),
    # A slot stored only in the body, read on the first iteration.
    (
        (
            const(0),
            store(0),
            load(1),
            arg(0),
            LESS_THAN,
            jump_if_false(11),
            load(0),
            const(1),
            ADD,
            store(1),
            jump(2),
            load(0),
            RETURN,
        ),
        "LOAD_LOCAL at index 2 reads local 1 before it is assigned",
        "LOAD_LOCAL at index 2 reads local 1 before it is assigned",
    ),
    # A slot stored only in the body, read after zero iterations.
    (
        replaced(LOOP, i9=store(1), i11=load(1)),
        "LOAD_LOCAL at index 11 reads local 1 before it is assigned",
        "LOAD_LOCAL at index 11 reads local 1 before it is assigned",
    ),
]


@pytest.mark.parametrize(("program", "python", "rust"), MALFORMED_LOOPS)
def test_malformed_loops_are_rejected_by_both_validators(program, python, rust):
    with pytest.raises(InvalidProgramError) as excinfo:
        validate(program)
    assert python in str(excinfo.value)
    with pytest.raises(InvalidProgramError):
        execute(program, (1,))
    with pytest.raises(InvalidProgramError):
        compile_i64(program, 1)
    # Rust on its own, bypassing Python's validation.
    with pytest.raises(ValueError) as excinfo:
        _native.compile_i64(_compiler._encode(program), 1)
    assert rust in str(excinfo.value)


def test_a_jump_from_before_the_loop_to_its_exit_is_valid():
    # i = -1; if a < 0: skip the loop, else: i = 0; while i < a: i += 1
    program = (
        const(-1),
        store(0),
        arg(0),
        const(0),
        LESS_THAN,
        jump_if_false(7),
        jump(18),  # Past the whole loop, to its exit.
        const(0),
        store(0),
        load(0),  # 9: the header.
        arg(0),
        LESS_THAN,
        jump_if_false(18),
        load(0),
        const(1),
        ADD,
        store(0),
        jump(9),
        load(0),  # 18: the exit.
        RETURN,
    )
    validate(program)
    compiled = compile_i64(program, 1)
    for a in (-3, 0, 4):
        assert compiled(a) == execute(program, (a,)) == run_i64(program, (a,))[1]
    assert [compiled(a) for a in (-3, 0, 4)] == [-1, 0, 4]


# --- Reference interpreter: the step limit ----------------------------------


def test_the_default_limits_agree():
    assert DEFAULT_MAX_BACKEDGES == _native.DEFAULT_BACKEDGE_BUDGET
    assert _native.MAX_BACKEDGE_BUDGET == _native.DEFAULT_BACKEDGE_BUDGET
    assert _compiler.DEFAULT_BACKEDGE_BUDGET == 1_000_000
    assert _native.STATUS_BUDGET_EXHAUSTED not in (
        _native.STATUS_OK,
        _native.STATUS_OVERFLOW,
        _native.STATUS_BAD_ARITY,
    )


@pytest.mark.parametrize("limit", [0, 1, 5])
def test_the_interpreter_stops_after_its_limit(limit):
    assert execute(LOOP, (limit,), max_backedges=limit) == limit
    with pytest.raises(ExecutionLimitExceeded) as excinfo:
        execute(LOOP, (limit + 1,), max_backedges=limit)
    assert (excinfo.value.limit, excinfo.value.index) == (limit, 10)


def test_the_interpreter_limit_is_not_a_guard_failure_or_a_result():
    assert not issubclass(ExecutionLimitExceeded, InvalidProgramError)
    from speedysnek._interpreter import GuardFailure

    assert not issubclass(ExecutionLimitExceeded, GuardFailure)


@pytest.mark.parametrize(
    ("value", "error"), [(-1, ValueError), (1.0, TypeError), (True, TypeError)]
)
def test_the_interpreter_limit_is_checked(value, error):
    with pytest.raises(error):
        execute(LOOP, (1,), max_backedges=value)


def test_a_nonterminating_program_stops_in_the_interpreter():
    # while 0 < 1: pass
    forever = (
        const(0),
        const(1),
        LESS_THAN,
        jump_if_false(5),
        jump(0),
        const(0),
        RETURN,
    )
    validate(forever)
    with pytest.raises(ExecutionLimitExceeded):
        execute(forever, (), max_backedges=1000)
    assert run_i64(forever, (), budget=1000) == ("budget", 4)


def test_acyclic_programs_ignore_the_limit():
    program = (arg(0), const(1), ADD, RETURN)
    assert execute(program, (1,), max_backedges=0) == 2


# --- Native code: results ---------------------------------------------------


def differential(func, args, budget=None):
    """Check native code (both entry points) against CPython, the reference
    interpreter and ``run_i64`` for one call. Returns ``run_i64``'s status."""
    instructions = analyze(func).instructions
    compiled = compile_i64(
        instructions, func.__code__.co_argcount, backedge_budget=budget
    )
    limit = compiled.backedge_budget
    assert limit is not None
    status, detail = run_i64(instructions, args, limit)
    expected = func(*args)
    try:
        reference = execute(instructions, args, max_backedges=limit)
    except ExecutionLimitExceeded as error:
        assert (status, detail) == ("budget", error.index)
    else:
        # Arbitrary precision, like CPython, whatever native code does.
        assert reference == expected, (func, args)
    if status == "ok":
        assert compiled(*args) == detail == expected, (func, args)
        assert compiled.fast_call(args) == detail
    else:
        with pytest.raises(NativeGuardFailure) as excinfo:
            compiled(*args)
        assert excinfo.value.reason is REASONS[status]
        assert excinfo.value.instruction_index == detail
        assert compiled.fast_call(args) is None
    return status


REASONS = {"overflow": GuardReason.OVERFLOW, "budget": GuardReason.BUDGET_EXHAUSTED}


def test_differential_corpus():
    statuses = collections.Counter()
    for func, argsets in CORPUS:
        for args in argsets:
            statuses[differential(func, args)] += 1
    assert statuses["ok"] > 200
    assert statuses["overflow"] >= 5


@pytest.mark.parametrize("n", [-(2**63), -5, -1, 0])
def test_zero_iterations(n):
    compiled = compile_i64(analyze(sum_squares).instructions)
    assert compiled(n) == 0
    assert compiled.fast_call((n,)) == 0


@pytest.mark.parametrize(("n", "expected"), [(1, 0), (2, 1), (3, 5), (10, 285)])
def test_one_and_several_iterations(n, expected):
    compiled = compile_i64(analyze(sum_squares).instructions)
    assert compiled(n) == expected == sum_squares(n)


@pytest.mark.parametrize(
    ("condition", "start", "step"),
    [
        ("i < n", 0, 1),
        ("i <= n", 0, 1),
        ("i > n", 0, -1),
        ("i >= n", 0, -1),
        ("i != n", 0, 1),
        ("i == n", 0, 1),
        ("n > i", 0, 1),
        ("n < i", 0, -1),
    ],
)
def test_every_comparison_as_a_loop_condition(condition, start, step):
    f = define(
        f"""
        def f(n):
            t = 0
            i = {start}
            while {condition}:
                t += i * 3 + 1
                i += {step}
            return t * 1000 + i
        """
    )
    compiled = compile_i64(analyze(f).instructions)
    # ``i != n`` with ``n < 0`` never ends; leave it to the budget test.
    values = range(0, 12) if "!=" in condition else range(-12, 12)
    for n in values:
        assert compiled(n) == f(n) == execute(analyze(f).instructions, (n,)), n


def test_counting_down_an_argument():
    compiled = compile_i64(analyze(countdown).instructions)
    assert [compiled(n) for n in (-1, 0, 1, 4)] == [0, 0, 1, 10]


def test_two_locals_carried_across_iterations():
    compiled = compile_i64(analyze(fibonacci).instructions)
    assert [compiled(n) for n in range(10)] == [fibonacci(n) for n in range(10)]
    assert compiled(91) == fibonacci(91)
    # fibonacci(92) fits, but its last iteration also computes F(93) into
    # ``b``, which never reaches the result: that addition still overflows.
    assert fibonacci(92) <= I64_MAX < fibonacci(93)
    with pytest.raises(NativeGuardFailure) as excinfo:
        compiled(92)
    assert excinfo.value.reason is GuardReason.OVERFLOW


def test_a_copy_keeps_its_value_when_its_source_changes():
    compiled = compile_i64(analyze(last_copy).instructions)
    assert [compiled(n) for n in (-1, 0, 1, 2, 7)] == [-100, -100, 1, 102, 607]
    f = define(
        """
        def f(n):
            total = 1
            previous = 0
            i = 0
            while i < n:
                previous = total
                total += total + i
                i += 1
            return previous * 1000000 + total
        """
    )
    compiled = compile_i64(analyze(f).instructions)
    assert [compiled(n) for n in range(6)] == [f(n) for n in range(6)]


def test_code_after_the_loop_sees_the_final_values():
    compiled = compile_i64(analyze(after_the_loop).instructions)
    assert compiled(3, 5) == after_the_loop(3, 5) == 15 * 2 + 3
    assert compiled(2, 1) == after_the_loop(2, 1) == 0


# --- Native code: overflow --------------------------------------------------


def overflow_index(func, args):
    instructions = analyze(func).instructions
    compiled = compile_i64(instructions)
    with pytest.raises(NativeGuardFailure) as excinfo:
        compiled(*args)
    assert excinfo.value.reason is GuardReason.OVERFLOW
    assert compiled.fast_call(args) is None
    assert run_i64(instructions, args) == ("overflow", excinfo.value.instruction_index)
    assert execute(instructions, args) == func(*args)
    return instructions[excinfo.value.instruction_index].op


def test_overflow_after_several_iterations():
    f = define(
        """
        def f(n):
            x = 1
            i = 0
            while i < n:
                x *= 3
                i += 1
            return x
        """
    )
    compiled = compile_i64(analyze(f).instructions)
    assert compiled(39) == 3**39  # 39 iterations fit.
    assert overflow_index(f, (40,)) is Op.MULTIPLY


def test_overflow_in_the_condition():
    f = define(
        """
        def f(n):
            x = 1
            while x * 1000 < n:
                x = x * 2 + 1
            return x
        """
    )
    assert compile_i64(analyze(f).instructions)(10**9) == f(10**9)
    assert overflow_index(f, (I64_MAX,)) is Op.MULTIPLY


def test_overflow_in_the_counter_update():
    f = define(
        """
        def f(n, step):
            i = 0
            while i < n:
                i += step
            return i
        """
    )
    assert compile_i64(analyze(f).instructions)(2**62, 2**61) == 2**62
    assert overflow_index(f, (I64_MAX, 2**62)) is Op.ADD


def test_overflow_after_the_loop():
    f = define(
        """
        def f(n, k):
            i = 0
            while i < n:
                i += 1
            return i * k
        """
    )
    assert compile_i64(analyze(f).instructions)(3, 2**61) == 3 * 2**61
    assert overflow_index(f, (4, 2**61)) is Op.MULTIPLY


def test_an_overwritten_result_still_overflows():
    f = define(
        """
        def f(n, big):
            i = 0
            junk = 0
            while i < n:
                junk = i * big
                junk = 0
                i += 1
            return i + junk
        """
    )
    # ``i * big`` fits for i = 0 and 1, not for 2.
    assert compile_i64(analyze(f).instructions)(2, 2**62) == 2
    assert overflow_index(f, (3, 2**62)) is Op.MULTIPLY


def test_an_untaken_branch_does_not_overflow():
    f = define(
        """
        def f(n, cutoff, big):
            total = 0
            i = 0
            while i < n:
                if i < cutoff:
                    total += i
                else:
                    total += i * big
                i += 1
            return total
        """
    )
    compiled = compile_i64(analyze(f).instructions)
    # The ``else`` branch never runs, so ``i * big`` never overflows.
    assert compiled(50, 100, I64_MAX) == sum(range(50))
    assert compiled.fast_call((50, 100, I64_MAX)) == sum(range(50))
    # Once it runs (i = 3), it does.
    assert overflow_index(f, (50, 3, I64_MAX)) is Op.MULTIPLY


# --- Native code: the backedge budget ---------------------------------------


@pytest.mark.parametrize("budget", [0, 1, 7, 1000])
def test_the_budget_is_exact(budget):
    instructions = analyze(sum_squares).instructions
    compiled = compile_i64(instructions, backedge_budget=budget)
    assert compiled.backedge_budget == budget
    # n iterations take n backedges.
    assert compiled(budget) == sum_squares(budget)
    assert compiled.fast_call((budget,)) == sum_squares(budget)
    with pytest.raises(NativeGuardFailure) as excinfo:
        compiled(budget + 1)
    failure = excinfo.value
    assert failure.reason is GuardReason.BUDGET_EXHAUSTED
    assert instructions[failure.instruction_index] == jump(4)
    assert "backward jumps" in str(failure) and "overflow" not in str(failure)
    assert compiled.fast_call((budget + 1,)) is None
    # The raw ABI: the status, and the backedge's index, never a result.
    assert compiled.function.call((budget + 1,)) == (
        _native.STATUS_BUDGET_EXHAUSTED,
        failure.instruction_index,
    )
    with pytest.raises(ExecutionLimitExceeded):
        execute(instructions, (budget + 1,), max_backedges=budget)
    assert run_i64(instructions, (budget + 1,), budget)[0] == "budget"


def test_the_default_budget():
    compiled = compile_i64(analyze(sum_squares).instructions)
    budget = _native.DEFAULT_BACKEDGE_BUDGET
    assert compiled.backedge_budget == budget
    assert compiled(budget) == sum_squares(budget)
    assert compiled.fast_call((budget + 1,)) is None


def test_every_backedge_spends_the_budget():
    # ``trailing_if_else`` has a backedge in each branch.
    compiled = compile_i64(analyze(trailing_if_else).instructions, backedge_budget=6)
    for cutoff in (0, 3, 100):
        assert compiled(6, cutoff) == trailing_if_else(6, cutoff)
        assert compiled.fast_call((7, cutoff)) is None


def test_budget_exhaustion_is_reported_before_a_later_overflow():
    f = define(
        """
        def f(n):
            x = 1
            i = 0
            while i < n:
                x *= 3
                i += 1
            return x
        """
    )
    compiled = compile_i64(analyze(f).instructions, backedge_budget=10)
    with pytest.raises(NativeGuardFailure) as excinfo:
        compiled(100)
    assert excinfo.value.reason is GuardReason.BUDGET_EXHAUSTED
    compiled = compile_i64(analyze(f).instructions, backedge_budget=100)
    with pytest.raises(NativeGuardFailure) as excinfo:
        compiled(100)
    assert excinfo.value.reason is GuardReason.OVERFLOW


def test_programs_without_a_loop_have_no_budget():
    for source in (
        "def f(a, b):\n    return (a + b) * 2\n",
        "def f(a, b):\n    if a < b:\n        return a\n    return b\n",
    ):
        compiled = compile_i64(analyze(define(source)).instructions)
        assert compiled.backedge_budget is None
        assert compiled.function.backedge_budget is None


@pytest.mark.parametrize(
    ("budget", "error"),
    [
        (-1, ValueError),
        (_native.MAX_BACKEDGE_BUDGET + 1, ValueError),
        (2**64, ValueError),
        (1.5, TypeError),
        (True, TypeError),
    ],
)
def test_the_budget_can_only_be_lowered(budget, error):
    instructions = analyze(sum_squares).instructions
    with pytest.raises(error):
        compile_i64(instructions, backedge_budget=budget)
    with pytest.raises(error):
        _native.compile_i64(_compiler._encode(instructions), 1, budget)


def test_a_nonterminating_native_loop_returns():
    # ``while 0 < 1: pass``, sent straight to Rust. It is valid; only the
    # budget ends it. In a child process, in case it does not.
    stdout = run_child(
        """
        import time
        from speedysnek import _native
        forever = ((2, 0), (2, 1), (11,), (10, 5), (9, 0), (2, 0), (6,))
        function = _native.compile_i64(forever, 0)
        start = time.perf_counter()
        print(function.call(()), function.try_call(()))
        print(time.perf_counter() - start < 5)
        """
    )
    assert stdout.split("\n")[:2] == [
        f"({_native.STATUS_BUDGET_EXHAUSTED}, 4) None",
        "True",
    ]


def test_mutated_loops_never_hang_or_crash():
    # Valid loops with one to three random instructions replaced, sent
    # straight to Rust in a child process: most are malformed, some are
    # valid loops that may never end. Rust must reject the first kind with
    # an exception and stop the second with its budget.
    templates = [
        _compiler._encode(program)
        for program in (
            LOOP,
            analyze(conditional_sum).instructions,
            analyze(trailing_if_else).instructions,
            analyze(countdown).instructions,
        )
    ]
    stdout = run_child(
        f"""
        import random
        from speedysnek import _native
        rng = random.Random(7)
        templates = {templates!r}
        pool = [(1, 0), (1, 1), (2, 0), (2, 1), (7, 0), (7, 1), (8, 0), (8, 1),
                (3,), (4,), (5,), (6,), (11,), (13,), (15,), (16,)]
        compiled = loops = 0
        for _ in range(5000):
            program = list(rng.choice(templates))
            for _ in range(rng.randrange(1, 4)):
                at = rng.randrange(len(program))
                if rng.random() < 0.4:
                    kind = rng.choice([9, 10])
                    program[at] = (kind, rng.randrange(len(program) + 1))
                else:
                    program[at] = rng.choice(pool)
            try:
                function = _native.compile_i64(tuple(program), 2, rng.randrange(1000))
            except (TypeError, ValueError):
                continue
            compiled += 1
            loops += function.backedge_budget is not None
            for args in [(0, 0), (3, 1), (-(2**63), 5), (2**63 - 1, -1)]:
                function.try_call(args)
        print(compiled, loops)
        """,
        timeout=300,
    )
    compiled, loops = map(int, stdout.split())
    assert compiled >= loops > 100


# --- Bounded random differential tests --------------------------------------


def random_loop_program(rng, arg_count, slots):
    """Random valid IR with one loop: slots assigned up front, a header that
    compares an expression with another, a body of assignments and
    if/else statements (some branches end with their own backedge), and
    code after the loop."""
    program = []
    # Slots that may be read: all of them, once the prologue stored them.
    readable = [0]

    def expression(budget):
        if budget <= 1 or rng.random() < 0.35:
            choice = rng.random()
            if choice < 0.5 and readable[0]:
                program.append(load(rng.randrange(readable[0])))
            elif choice < 0.8:
                program.append(arg(rng.randrange(arg_count)))
            else:
                program.append(const(rng.choice([-2, -1, 0, 1, 2, 3, 2**40, I64_MAX])))
            return
        left = rng.randint(1, budget - 1)
        expression(left)
        if rng.random() < 0.3:
            # Multiplying only by constants keeps the interpreter's exact
            # values small even after an i64 overflow; squaring in a loop
            # would double their size every iteration.
            program.append(const(rng.choice([-3, 2, 5, 2**31])))
            program.append(MULTIPLY)
            return
        expression(budget - left)
        program.append(Instruction(rng.choice([Op.ADD, Op.SUBTRACT])))

    def statements(count, depth, header):
        """Emit statements; return True if every path already jumped back."""
        for position in range(count):
            choice = rng.random()
            if depth and choice < 0.35:
                expression(rng.randint(1, 3))
                expression(rng.randint(1, 3))
                program.append(Instruction(rng.choice(CMP_OPS)))
                branch = len(program)
                program.append(None)
                last = position == count - 1
                then_back = statements(rng.randint(0, 2), depth - 1, header)
                if last and header is not None and not then_back and rng.random() < 0.5:
                    program.append(jump(header))  # A duplicated backedge.
                    then_back = True
                skip = None
                if not then_back:
                    skip = len(program)
                    program.append(None)
                program[branch] = jump_if_false(len(program))
                other_back = statements(rng.randint(0, 2), depth - 1, header)
                if skip is not None:
                    program[skip] = jump(len(program))
                if then_back and other_back:
                    return True  # Nothing after this if is reachable.
            elif choice < 0.42 and header is not None:
                program.append(jump(header))  # ``continue``.
                return True
            else:
                expression(rng.randint(1, 4))
                program.append(store(rng.randrange(slots)))
        return False

    for slot in range(slots):
        expression(2)
        program.append(store(slot))
        readable[0] = slot + 1
    statements(rng.randint(0, 2), 1, None)
    header = len(program)
    expression(rng.randint(1, 3))
    expression(rng.randint(1, 2))
    program.append(Instruction(rng.choice(CMP_OPS)))
    test = len(program)
    program.append(None)
    if not statements(rng.randint(1, 4), 2, header):
        program.append(jump(header))
    program[test] = jump_if_false(len(program))
    statements(rng.randint(0, 2), 1, None)
    expression(rng.randint(1, 3))
    program.append(RETURN)
    return tuple(program)


def test_differential_random_loop_programs():
    rng = random.Random("speedysnek-native-loops")
    outcomes = collections.Counter()
    for _ in range(200):
        arg_count = rng.randrange(1, 4)
        program = random_loop_program(rng, arg_count, rng.randint(1, 3))
        budget = rng.choice([0, 1, 5, 40])
        validate(program)
        compiled = compile_i64(program, arg_count, backedge_budget=budget)
        for _ in range(5):
            args = tuple(rng.randint(-4, 6) for _ in range(arg_count))
            status, detail = run_i64(program, args, budget)
            outcomes[status] += 1
            try:
                reference = execute(program, args, max_backedges=budget)
            except ExecutionLimitExceeded as error:
                reference = ("budget", error.index)
            if status == "ok":
                assert compiled(*args) == reference == detail, (program, args)
                assert compiled.fast_call(args) == detail
                continue
            if status == "budget":
                assert reference == ("budget", detail)
            with pytest.raises(NativeGuardFailure) as excinfo:
                compiled(*args)
            assert excinfo.value.reason is REASONS[status], (program, args)
            assert excinfo.value.instruction_index == detail
            assert compiled.fast_call(args) is None
    assert min(outcomes["ok"], outcomes["budget"], outcomes["overflow"]) > 30, outcomes


def random_python_loop(rng):
    """Source of a random terminating function with one while loop.

    ``i`` only changes in the first statement of the body, on every path,
    so the loop runs at most ``n`` times; everything else is random."""
    names = ["a", "t", "u"]

    def expression(depth=2):
        if depth == 0 or rng.random() < 0.4:
            return rng.choice(names + ["i", "n", "1", "2", "7", str(2**40)])
        if rng.random() < 0.3:
            # Only by constants: values grow linearly, not exponentially.
            return f"({expression(depth - 1)} * {rng.choice([-3, 5, 2**40])})"
        op = rng.choice(["+", "-"])
        return f"({expression(depth - 1)} {op} {expression(depth - 1)})"

    def condition():
        op = rng.choice(["<", "<=", ">", ">=", "==", "!="])
        return f"{expression(1)} {op} {expression(1)}"

    def block(indent, depth, count):
        lines = []
        for _ in range(count):
            choice = rng.random()
            if depth and choice < 0.3:
                lines.append(f"{indent}if {condition()}:")
                lines += block(indent + "    ", depth - 1, rng.randint(1, 2))
                if rng.random() < 0.6:
                    lines.append(f"{indent}else:")
                    lines += block(indent + "    ", depth - 1, rng.randint(1, 2))
            else:
                target = rng.choice(names)
                if rng.random() < 0.2:
                    lines.append(f"{indent}{target} *= {rng.choice([-2, 3, 2**33])}")
                else:
                    op = rng.choice(["=", "+=", "-="])
                    lines.append(f"{indent}{target} {op} {expression()}")
        return lines

    step = rng.choice(["i += 1", "i += 2"])
    test = rng.choice(["i < n", "i <= n", "n > i", "i * 2 < n + 3"])
    lines = [
        "def f(n, a):",
        "    t = 1",
        "    u = a - 1",
        "    i = 0",
        *block("    ", 1, rng.randint(0, 2)),
        f"    while {test}:",
        f"        {step}",
        *block("        ", 2, rng.randint(1, 3)),
        *block("    ", 1, rng.randint(0, 2)),
        f"    return {expression()}",
    ]
    return "\n".join(lines) + "\n"


def test_differential_random_python_loops():
    # CPython, @snekit, the native code, the interpreter and ``run_i64``
    # on random terminating loops with small arguments.
    rng = random.Random("speedysnek-python-loops")
    paths = collections.Counter()
    for _ in range(120):
        source = random_python_loop(rng)
        func = define(source)
        analysis = analyze(func)
        assert analysis.supported, (source, analysis.rejection)
        wrapped = snekit(func)
        for _ in range(_WARMUP_CALLS + 1):
            wrapped(0, 0)
        if wrapped._jit_state.disabled_reason is _DisabledReason.UNSUPPORTED:
            # CPython folded a product of constants beyond i64.
            assert "LOAD_CONST" in str(wrapped._jit_state.compile_error)
            paths["constant"] += 1
            continue
        assert wrapped._jit_state.status is _Status.COMPILED, source
        for _ in range(4):
            args = (rng.randint(-2, 12), rng.randint(-5, 5))
            expected = func(*args)
            status = differential(func, args)
            with counting_frames() as frames:
                before = wrapped._jit_state.compiled.function.try_calls
                assert wrapped(*args) == expected
                fast = wrapped._jit_state.compiled.function.try_calls - before
            assert fast == 1
            assert frames[func.__code__] == (0 if status == "ok" else 1), source
            paths[status] += 1
    assert paths["ok"] > 200 and paths["overflow"] > 10, paths
    assert paths["constant"] < 20, paths


@contextlib.contextmanager
def counting_frames():
    frames = collections.Counter()

    def profiler(frame, event, arg):
        if event == "call":
            frames[frame.f_code] += 1

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        yield frames
    finally:
        sys.setprofile(previous)


# --- @snekit -----------------------------------------------------------------


def warmed(func, *args):
    wrapped = snekit(func)
    for _ in range(_WARMUP_CALLS + 1):
        wrapped(*args)
    assert wrapped._jit_state.status is _Status.COMPILED
    return wrapped


def path_of(wrapped, *args):
    """Call ``wrapped``; return ``(result, path)``, where the path is
    ``"native"``, ``"guard_miss"`` or ``"python"``, as in
    ``benchmarks/workloads.call_path``."""
    function = wrapped._jit_state.compiled.function
    before = function.try_calls
    with counting_frames() as frames:
        result = wrapped(*args)
    observed = (frames[wrapped.__wrapped__.__code__], function.try_calls - before)
    paths = {(0, 1): "native", (1, 1): "guard_miss", (1, 0): "python"}
    return result, paths[observed]


@pytest.mark.parametrize(
    ("func", "args"),
    [
        (sum_squares, (100,)),
        (sum_squares, (0,)),
        (sum_squares, (-7,)),
        (countdown, (100,)),
        (conditional_sum, (100, 40)),
        (conditional_sum, (0, 0)),
    ],
)
def test_the_goal_functions_run_natively(func, args):
    wrapped = warmed(func, 3, 1) if func is conditional_sum else warmed(func, 3)
    assert path_of(wrapped, *args) == (func(*args), "native")


def test_overflow_falls_back_and_keeps_the_specialization():
    f = warmed(fibonacci, 10)
    assert path_of(f, 91) == (fibonacci(91), "native")
    assert path_of(f, 92) == (fibonacci(92), "guard_miss")
    assert path_of(f, 100) == (fibonacci(100), "guard_miss")
    assert f(100) > I64_MAX
    assert path_of(f, 50) == (fibonacci(50), "native")
    assert f._jit_state.status is _Status.COMPILED


@pytest.fixture
def small_budget(monkeypatch):
    """Make ``@snekit`` compile loops with a backedge budget of 10."""
    monkeypatch.setattr(_compiler, "_BACKEDGE_BUDGET", 10)
    return 10


def test_budget_exhaustion_falls_back_to_python(small_budget):
    f = warmed(sum_squares, 3)
    assert f._jit_state.compiled.backedge_budget == small_budget
    assert path_of(f, small_budget) == (sum_squares(small_budget), "native")
    # One more iteration than the budget: Python computes the result.
    assert path_of(f, small_budget + 1) == (sum_squares(small_budget + 1), "guard_miss")
    assert path_of(f, 10_000) == (sum_squares(10_000), "guard_miss")
    # The specialization stays and is used again.
    assert f._jit_state.status is _Status.COMPILED
    assert path_of(f, 4) == (sum_squares(4), "native")


def test_the_fallback_gets_the_original_arguments(small_budget):
    # Native code has already decremented ``n`` ten times when it stops.
    f = warmed(countdown, 3)
    seen = []

    def profiler(frame, event, arg):
        if event == "call" and frame.f_code is countdown.__code__:
            seen.append(frame.f_locals["n"])

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        result = f(25)
    finally:
        sys.setprofile(previous)
    assert seen == [25]
    assert result == countdown(25) == 325


def test_budget_exhaustion_is_not_an_exception(small_budget):
    f = warmed(conditional_sum, 3, 1)
    assert f(1000, 500) == conditional_sum(1000, 500)
    assert f._jit_state.native_error is None
    assert f._jit_state.disabled_reason is None


class Counted:
    """An object whose comparison and arithmetic count their calls."""

    calls = 0

    def __init__(self, value):
        self.value = value

    def _count(self):
        type(self).calls += 1

    def __lt__(self, other):
        self._count()
        return self.value < getattr(other, "value", other)

    def __gt__(self, other):
        self._count()
        return self.value > getattr(other, "value", other)

    def __rmul__(self, other):
        self._count()
        return other * self.value

    def __mul__(self, other):
        self._count()
        return self.value * getattr(other, "value", other)

    def __radd__(self, other):
        self._count()
        return other + self.value

    def __add__(self, other):
        self._count()
        return Counted(self.value + getattr(other, "value", other))

    def __sub__(self, other):
        self._count()
        return Counted(self.value - getattr(other, "value", other))


class IntSubclass(int):
    calls = 0

    def __lt__(self, other):
        IntSubclass.calls += 1
        return int(self) < other

    def __gt__(self, other):
        IntSubclass.calls += 1
        return int(self) > other


@pytest.mark.parametrize("make", [Counted, IntSubclass])
def test_custom_objects_only_run_in_python(make):
    f = warmed(countdown, 3)
    make.calls = 0
    plain = countdown(make(5))
    expected_calls = make.calls
    assert expected_calls > 0
    make.calls = 0
    result, path = path_of(f, make(5))
    assert path == "guard_miss"
    assert result == plain
    # Every method ran once per Python operation, none natively.
    assert make.calls == expected_calls


def test_bools_run_in_python():
    f = warmed(sum_squares, 3)
    assert path_of(f, True) == (0, "guard_miss")


def test_an_infinite_python_loop_eventually_runs_in_python():
    # ``i != n`` never ends for a negative ``n``. Native code stops after its
    # budget; then the original function runs forever, in the normal
    # interpreter, where other threads get the GIL. Never in this process.
    stdout = run_child(
        """
        import os, sys, threading, time
        from speedysnek import snekit

        def f(n):
            i = 0
            while i != n:
                i += 1
            return i

        g = snekit(f)
        for _ in range(20):
            g(5)
        assert g._jit_state.status.value == "compiled"
        entered = threading.Event()

        def profiler(frame, event, arg):
            if event == "call" and frame.f_code is f.__code__:
                entered.set()

        threading.setprofile(profiler)
        worker = threading.Thread(target=g, args=(-1,), daemon=True)
        before = g._jit_state.compiled.function.try_calls
        worker.start()
        # The main thread gets the GIL back, and the original function runs.
        print(entered.wait(30), worker.is_alive())
        time.sleep(0.2)
        print(worker.is_alive(), g._jit_state.compiled.function.try_calls - before)
        sys.stdout.flush()
        os._exit(0)
        """,
        timeout=60,
    )
    # One native attempt, then Python, still running when the child exits.
    assert stdout.split() == ["True", "True", "True", "1"]


def test_straight_line_and_conditional_functions_still_run_natively():
    def straight(a, b):
        return (a + b) * 2

    def branch(a, b):
        if a < b:
            return a
        return b

    for func, overflowing in ((straight, "guard_miss"), (branch, "native")):
        f = warmed(func, 1, 2)
        assert f._jit_state.compiled.backedge_budget is None
        assert path_of(f, 3, 4) == (func(3, 4), "native")
        assert path_of(f, I64_MAX, 1) == (func(I64_MAX, 1), overflowing)
