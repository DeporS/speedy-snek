"""Tests for ``for`` loops over ``range(n)``: bytecode recognition, the
lowering into the ``while`` loop IR, ``range`` semantics, overflow, the
backedge budget, the range-binding guard and ``@snekit``'s fallback.

See ``docs/for-range.md`` for the design. Execution paths are proven as in
``test_loops.py``: native calls with the compiled function's ``try_calls``
counter, Python calls with ``sys.setprofile``. Tests that change the real
``builtins`` module run in a child process.
"""

import builtins
import collections
import dis
import gc
import random
import sys
import types

import pytest
from test_compiler import run_i64
from test_loops import (
    define,
    differential,
    path_of,
    run_child,
    warmed,
)

from speedysnek import _compiler, _native, snekit
from speedysnek._compiler import (
    I64_MAX,
    GuardReason,
    NativeGuardFailure,
    compile_i64,
)
from speedysnek._decorator import _WARMUP_CALLS, _DisabledReason, _Status
from speedysnek._frontend import Instruction, Op, RejectionReason, analyze
from speedysnek._interpreter import ExecutionLimitExceeded, execute, validate

RANGE = range


def sum_squares(n):
    total = 0
    for i in range(n):
        total += i * i
    return total


def conditional_sum(n, cutoff):
    total = 0
    for i in range(n):
        if i < cutoff:
            total += i
        else:
            total += i * 2
    return total


def sum_squares_while(n):
    total = 0
    i = 0
    while i < n:
        total += i * i
        i += 1
    return total


def bound_once(n):
    total = 0
    for i in range(n):
        total += i
        n = 0
    return total


def reassigned_target(n):
    total = 0
    i = -1
    for i in range(n):
        total += i
        i = 100
    return total + i


def last(n):
    i = 99
    for i in range(n):  # noqa: B007 -- the value after the loop is the point.
        pass
    return i


def argument_target(n, i):
    for i in range(n):  # noqa: B007
        pass
    return i


def bound_is_target(n):
    for n in range(n):  # noqa: B007, B020
        pass
    return n


def possibly_unbound(n):
    for i in range(n):  # noqa: B007
        pass
    return i


def bound_expression(n, m):
    total = 0
    for i in range(n * 2 - m + 1):
        total += i
    return total


def trailing_if(n, cutoff):
    total = 0
    for i in range(n):
        if i < cutoff:
            total += 1
    return total


def unused_target(n):
    total = 7
    for _ in range(n):
        total += 3
    return total


def code_around(a, n):
    total = 0
    if a < n:
        total = 100
    for i in range(n - a):
        total += i * a
    if total > 50:
        total -= 50
    else:
        total += 1
    return total


def loop_in_if(a, n):
    total = 0
    if a < n:
        for i in range(n):
            total += i
    return total


GOAL = [
    (sum_squares, (100,)),
    (sum_squares, (0,)),
    (sum_squares, (-7,)),
    (conditional_sum, (100, 40)),
    (conditional_sum, (0, 0)),
]

SUPPORTED = [
    sum_squares,
    conditional_sum,
    bound_once,
    reassigned_target,
    last,
    argument_target,
    bound_is_target,
    bound_expression,
    trailing_if,
    unused_target,
    code_around,
    loop_in_if,
]


def args_for(func, n):
    """Small arguments for any function above: ``n`` first."""
    extra = (3, -1, 5)[: func.__code__.co_argcount - 1]
    return (n, *extra)


# --- Frontend: the lowering --------------------------------------------------


def ins(op, arg=None):
    return Instruction(op, arg)


def test_sum_squares_ir():
    # Slots: total 0, i 1; the bound 2 and the counter 3 come after them.
    assert analyze(sum_squares).instructions == (
        ins(Op.LOAD_CONST, 0),
        ins(Op.STORE_LOCAL, 0),
        ins(Op.LOAD_ARG, 0),  # the bound: range(n)
        ins(Op.STORE_LOCAL, 2),
        ins(Op.LOAD_CONST, 0),  # GET_ITER: the counter starts at 0
        ins(Op.STORE_LOCAL, 3),
        ins(Op.LOAD_LOCAL, 3),  # 6, FOR_ITER: the loop header
        ins(Op.LOAD_LOCAL, 2),
        ins(Op.LESS_THAN),
        ins(Op.JUMP_IF_FALSE, 23),
        ins(Op.LOAD_LOCAL, 3),  # STORE_FAST i: i = counter
        ins(Op.STORE_LOCAL, 1),
        ins(Op.LOAD_LOCAL, 3),  # counter += 1
        ins(Op.LOAD_CONST, 1),
        ins(Op.ADD),
        ins(Op.STORE_LOCAL, 3),
        ins(Op.LOAD_LOCAL, 0),  # total += i * i
        ins(Op.LOAD_LOCAL, 1),
        ins(Op.LOAD_LOCAL, 1),
        ins(Op.MULTIPLY),
        ins(Op.ADD),
        ins(Op.STORE_LOCAL, 0),
        ins(Op.JUMP, 6),
        ins(Op.LOAD_LOCAL, 0),  # 23: after END_FOR and POP_ITER
        ins(Op.RETURN),
    )


def test_the_analysis_assumes_range():
    assert analyze(sum_squares).assumed_builtins == ("range",)
    assert analyze(sum_squares_while).assumed_builtins == ()


@pytest.mark.parametrize("func", SUPPORTED, ids=lambda f: f.__name__)
def test_supported_loops_are_valid_ir(func):
    analysis = analyze(func)
    assert analysis.supported, analysis.rejection
    validate(analysis.instructions)
    # The Rust validator accepts it too.
    compile_i64(analysis.instructions, func.__code__.co_argcount)


def test_hidden_slots_never_collide_with_variables():
    names = [f"v{k}" for k in range(20)]
    source = (
        "def f(n):\n"
        + "".join(f"    {name} = {k}\n" for k, name in enumerate(names))
        + "    for i in range(n):\n"
        + "".join(f"        {name} += i\n" for name in names)
        + f"    return {' + '.join(names)}\n"
    )
    f = define(source)
    instructions = analyze(f).instructions
    # Variables: v0..v19 are slots 0..19, i is 20; the hidden slots follow.
    stored = {instr.arg for instr in instructions if instr.op is Op.STORE_LOCAL}
    assert stored == set(range(23))
    compiled = compile_i64(instructions)
    for n in (-1, 0, 1, 5):
        assert compiled(n) == f(n) == execute(instructions, (n,))


def _bytecode(func):
    return [
        (instr.opname, instr.arg)
        for instr in dis.get_instructions(func)
        if instr.opname not in ("RESUME", "NOP")
    ]


def test_continue_is_bytecode_identical_to_a_supported_if_else():
    # Documented in docs/for-range.md, as for while loops.
    with_continue = define(
        """
        def f(n, c):
            t = 0
            for i in range(n):
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
            for i in range(n):
                if i > c:
                    pass
                else:
                    t += i
            return t
        """
    )
    assert _bytecode(with_continue) == _bytecode(with_else)
    assert analyze(with_continue).instructions == analyze(with_else).instructions
    compiled = compile_i64(analyze(with_continue).instructions)
    for n, c in [(0, 0), (5, 2), (10, 20), (10, -1)]:
        assert compiled(n, c) == with_continue(n, c) == with_else(n, c)


def test_loop_else_without_break_is_code_after_the_loop():
    with_else = define(
        """
        def f(n):
            t = 0
            for i in range(n):
                t += i
            else:
                t += 100
            return t
        """
    )
    after = define(
        """
        def f(n):
            t = 0
            for i in range(n):
                t += i
            t += 100
            return t
        """
    )
    assert _bytecode(with_else) == _bytecode(after)
    assert analyze(with_else).instructions == analyze(after).instructions


# --- Frontend: rejected forms ------------------------------------------------


def assert_rejected(func, reason, opname, message):
    analysis = analyze(func)
    rejection = analysis.rejection
    assert rejection is not None, analysis.instructions
    assert (rejection.reason, rejection.opname) == (reason, opname), rejection
    assert message in rejection.message, rejection.message
    assert analysis.assumed_builtins == ()
    return rejection


ITERATION = RejectionReason.UNSUPPORTED_ITERATION
CONTROL = RejectionReason.CONTROL_FLOW


@pytest.mark.parametrize(
    ("loop", "reason", "opname", "message"),
    [
        ("for i in range(1, n):", ITERATION, "CALL", "called with 2 arguments"),
        ("for i in range(0, n, 2):", ITERATION, "CALL", "called with 3 arguments"),
        ("for i in range():", ITERATION, "CALL", "called with 0 arguments"),
        ("for i in range(stop=n):", ITERATION, "CALL_KW", "keyword"),
        ("for i in range(*a):", ITERATION, "CALL_FUNCTION_EX", "unpacked"),
        ("for i in a:", ITERATION, "FOR_ITER", "does not iterate over a call"),
        ("for i in (n, a):", ITERATION, "FOR_ITER", "does not iterate over a call"),
        ("for i in iter(range(n)):", ITERATION, "CALL", "unsupported CALL"),
        ("for i in reversed(range(n)):", ITERATION, "CALL", "unsupported CALL"),
        ("for i in xrange(n):", ITERATION, "LOAD_GLOBAL", "xrange(...)"),
        ("for i in a(n):", ITERATION, "PUSH_NULL", "PUSH_NULL"),
        ("for i in a.range(n):", ITERATION, "LOAD_ATTR", "LOAD_ATTR"),
        ("for i in range(len(a)):", ITERATION, "CALL", "unsupported CALL"),
        (
            "for i in range(n // 2):",
            RejectionReason.UNSUPPORTED_OPERATOR,
            "BINARY_OP",
            "'//'",
        ),
        ("for i in range(-n):", ITERATION, "UNARY_NEGATIVE", "UNARY_NEGATIVE"),
        ("for i in range(n if a < n else a):", ITERATION, "JUMP_FORWARD", "JUMP"),
        ("for i, j in range(n):", ITERATION, "UNPACK_SEQUENCE", "one local"),
        ("for a.x in range(n):", ITERATION, "LOAD_FAST_BORROW", "one local"),
        ("for a[0] in range(n):", ITERATION, "LOAD_FAST_BORROW", "one local"),
    ],
)
def test_unsupported_iterations(loop, reason, opname, message):
    f = define(
        f"""
        def f(n, a):
            t = 0
            {loop}
                t += 1
            return t
        """
    )
    assert_rejected(f, reason, opname, message)


@pytest.mark.parametrize(
    ("source", "reason", "opname", "message"),
    [
        (
            """
            def f(n):
                global i
                t = 0
                for i in range(n):
                    t += 1
                return t
            """,
            ITERATION,
            "STORE_GLOBAL",
            "one local",
        ),
        (
            """
            def f(n, range):
                t = 0
                for i in range(n):
                    t += i
                return t
            """,
            ITERATION,
            "PUSH_NULL",
            "unsupported PUSH_NULL before the range bound",
        ),
        (
            """
            def f(n):
                r = range
                t = 0
                for i in r(n):
                    t += i
                return t
            """,
            ITERATION,
            "PUSH_NULL",
            "PUSH_NULL",
        ),
        (
            """
            def f(n):
                t = 0
                x = range(n)
                for i in x:
                    t += i
                return t
            """,
            ITERATION,
            "FOR_ITER",
            "does not iterate over a call",
        ),
        (
            # A second, unrelated use of the global.
            """
            def f(n):
                t = 0
                for i in range(n):
                    t += i
                r = range
                return t
            """,
            RejectionReason.UNSUPPORTED_OPCODE,
            "LOAD_GLOBAL",
            "unsupported opcode LOAD_GLOBAL",
        ),
        (
            """
            def f(n):
                return len(range(n))
            """,
            RejectionReason.UNSUPPORTED_OPCODE,
            "LOAD_GLOBAL",
            "unsupported opcode LOAD_GLOBAL",
        ),
        (
            """
            def f(n):
                t = 0
                for i in range(n):
                    for j in range(n):
                        t += j
                return t
            """,
            CONTROL,
            "FOR_ITER",
            "only one loop per function",
        ),
        (
            """
            def f(n):
                t = 0
                for i in range(n):
                    t += i
                for j in range(n):
                    t += j
                return t
            """,
            CONTROL,
            "FOR_ITER",
            "only one loop per function",
        ),
        (
            """
            def f(n):
                t = 0
                for i in range(n):
                    t += i
                while t > 10:
                    t -= 3
                return t
            """,
            CONTROL,
            "JUMP_BACKWARD",
            "only one loop per function",
        ),
        (
            """
            def f(n):
                t = 0
                while t < n:
                    for i in range(n):
                        t += 1
                return t
            """,
            CONTROL,
            "JUMP_BACKWARD",
            "only one loop per function",
        ),
        (
            """
            def f(n):
                t = 0
                for i in range(n):
                    if i > 3:
                        break
                    t += i
                return t
            """,
            CONTROL,
            "RETURN_VALUE",
            "return or break inside a for loop",
        ),
        (
            """
            def f(n):
                t = 0
                for i in range(n):
                    if i > 3:
                        break
                    t += i
                t = t * 2 + 1
                t = t * 3 + 1
                t = t * 5 + 1
                return t
            """,
            CONTROL,
            "JUMP_FORWARD",
            "a jump out of a for loop",
        ),
        (
            """
            def f(n):
                t = 0
                for i in range(n):
                    if i > 3:
                        break
                else:
                    t += 1
                return t
            """,
            CONTROL,
            "RETURN_VALUE",
            "return or break inside a for loop",
        ),
        (
            """
            def f(n):
                for i in range(n):
                    if i > 3:
                        return i
                return -1
            """,
            CONTROL,
            "RETURN_VALUE",
            "return or break inside a for loop",
        ),
        (
            """
            def f(n):
                for i in range(n):
                    return i
                return -1
            """,
            CONTROL,
            "FOR_ITER",
            "return or break inside a for loop",
        ),
        (
            """
            def f(n):
                return [i for i in range(n)]
            """,
            CONTROL,
            None,
            "exception handling",
        ),
    ],
)
def test_rejected_for_loops(source, reason, opname, message):
    assert_rejected(define(source), reason, opname, message)


def test_possibly_unbound_after_zero_iterations_is_rejected():
    rejection = assert_rejected(
        possibly_unbound,
        RejectionReason.UNSUPPORTED_VARIABLE,
        "LOAD_FAST_CHECK",
        "'i' may be read before it is assigned",
    )
    assert rejection.lineno == possibly_unbound.__code__.co_firstlineno + 3


def test_a_variable_first_stored_in_the_body_is_possibly_unbound():
    f = define(
        """
        def f(n):
            for i in range(n):
                value = i * 2
            return value
        """
    )
    assert_rejected(
        f, RejectionReason.UNSUPPORTED_VARIABLE, "LOAD_FAST_CHECK", "'value'"
    )


@pytest.mark.parametrize("n", [-3, 0, 1, 4])
def test_possibly_unbound_keeps_python_behaviour(n):
    f = snekit(possibly_unbound)
    for _ in range(_WARMUP_CALLS + 2):
        if n > 0:
            assert f(n) == n - 1
        else:
            with pytest.raises(UnboundLocalError):
                f(n)
    assert f._jit_state.disabled_reason is _DisabledReason.REJECTED


# --- Semantics: native code, the interpreter and CPython agree ---------------


@pytest.mark.parametrize("func", SUPPORTED, ids=lambda f: f.__name__)
def test_supported_functions_agree_everywhere(func):
    for n in (-5, -1, 0, 1, 2, 3, 10, 57):
        args = args_for(func, n)
        assert differential(func, args) == "ok", (func, args)


@pytest.mark.parametrize("n", [-(2**63), -100, -1, 0])
def test_zero_and_negative_bounds_run_no_iteration(n):
    assert compile_i64(analyze(sum_squares).instructions)(n) == 0
    assert compile_i64(analyze(last).instructions)(n) == 99


@pytest.mark.parametrize(("n", "expected"), [(1, 0), (2, 1), (3, 5), (10, 285)])
def test_one_and_several_iterations(n, expected):
    compiled = compile_i64(analyze(sum_squares).instructions)
    assert compiled(n) == sum_squares(n) == expected


def test_the_bound_is_evaluated_once():
    compiled = compile_i64(analyze(bound_once).instructions)
    for n in (0, 1, 5, 100):
        assert compiled(n) == bound_once(n) == sum(range(n))


def test_assigning_the_target_does_not_change_the_next_item():
    compiled = compile_i64(analyze(reassigned_target).instructions)
    # Every item is still yielded; the body's last assignment survives.
    assert compiled(5) == reassigned_target(5) == 10 + 100
    assert compiled(1) == reassigned_target(1) == 100
    assert compiled(0) == reassigned_target(0) == -1


def test_assigning_an_uninitialized_target_is_possibly_unbound():
    # Without ``i = -1``, ``i`` is unbound after an empty range.
    f = define(
        """
        def f(n):
            total = 0
            for i in range(n):
                total += i
                i = 100
            return total + i
        """
    )
    assert_rejected(f, RejectionReason.UNSUPPORTED_VARIABLE, "LOAD_FAST_CHECK", "'i'")
    wrapped = snekit(f)
    for _ in range(_WARMUP_CALLS + 2):
        assert wrapped(5) == 110
    with pytest.raises(UnboundLocalError):
        wrapped(0)


def test_the_target_keeps_the_last_item():
    compiled = compile_i64(analyze(last).instructions)
    for n in (-2, 0):
        assert compiled(n) == last(n) == 99
    for n in (1, 2, 50):
        assert compiled(n) == last(n) == n - 1


def test_empty_ranges_keep_argument_bindings():
    compiled = compile_i64(analyze(argument_target).instructions)
    for n, i in [(0, 42), (-1, -7), (3, 42)]:
        assert compiled(n, i) == argument_target(n, i)
    assert compiled(0, 42) == 42 and compiled(3, 42) == 2
    compiled = compile_i64(analyze(bound_is_target).instructions)
    for n in (-4, 0, 1, 9):
        assert compiled(n) == bound_is_target(n) == (n if n <= 0 else n - 1)


def test_arithmetic_bounds():
    compiled = compile_i64(analyze(bound_expression).instructions)
    for n, m in [(0, 0), (3, 2), (5, 20), (10, -4)]:
        assert compiled(n, m) == bound_expression(n, m)


def test_if_else_in_the_body():
    compiled = compile_i64(analyze(conditional_sum).instructions)
    for n in range(-1, 12):
        for cutoff in range(-1, 13):
            assert compiled(n, cutoff) == conditional_sum(n, cutoff)
    compiled = compile_i64(analyze(trailing_if).instructions)
    assert [compiled(9, c) for c in range(-1, 11)] == [
        trailing_if(9, c) for c in range(-1, 11)
    ]


# --- Overflow -----------------------------------------------------------------


def overflow_at(func, args):
    """The op that overflows natively; Python and the oracle still agree."""
    instructions = analyze(func).instructions
    compiled = compile_i64(instructions)
    with pytest.raises(NativeGuardFailure) as excinfo:
        compiled(*args)
    assert excinfo.value.reason is GuardReason.OVERFLOW
    assert compiled.fast_call(args) is None
    assert run_i64(instructions, args) == ("overflow", excinfo.value.instruction_index)
    assert execute(instructions, args) == func(*args)
    return excinfo.value.instruction_index, instructions


def test_body_overflow_after_several_iterations():
    f = define(
        """
        def f(n):
            t = 2**60
            for i in range(n):
                t *= 2
            return t
        """
    )
    compiled = compile_i64(analyze(f).instructions)
    assert compiled(2) == f(2) == 2**62
    index, instructions = overflow_at(f, (10,))
    assert instructions[index].op is Op.MULTIPLY
    wrapped = warmed(f, 1)
    assert path_of(wrapped, 2) == (2**62, "native")
    assert path_of(wrapped, 10) == (2**70, "guard_miss")
    assert wrapped._jit_state.status is _Status.COMPILED


def test_overflow_in_the_bound():
    f = define(
        """
        def f(a, b, c):
            t = 0
            for i in range(a + b - c):
                t += i
            return t
        """
    )
    args = (I64_MAX, 10, I64_MAX)
    index, instructions = overflow_at(f, args)
    # The ADD of the bound, before the hidden slots are even stored.
    assert instructions[index].op is Op.ADD
    assert instructions[index + 3] == ins(Op.STORE_LOCAL, 2)
    wrapped = warmed(f, 1, 2, 3)
    assert path_of(wrapped, *args) == (45, "guard_miss")


def test_an_overwritten_result_still_overflows():
    f = define(
        """
        def f(n, big):
            t = 0
            for i in range(n):
                x = i * big
                t += i
            return t
        """
    )
    # i = 2 makes ``x`` overflow; nothing reads ``x``.
    index, instructions = overflow_at(f, (5, 2**62))
    assert instructions[index].op is Op.MULTIPLY
    assert compile_i64(analyze(f).instructions)(2, 2**62) == 1
    wrapped = warmed(f, 1, 1)
    assert path_of(wrapped, 5, 2**62) == (10, "guard_miss")


def test_the_hidden_counter_never_overflows_near_i64_max():
    # The lowering of ``last`` with the counter starting just below the
    # largest bound: the increment runs right up to I64_MAX, never past it.
    program = list(analyze(last).instructions)
    start = program.index(ins(Op.LOAD_CONST, 0))
    assert program[start + 1] == ins(Op.STORE_LOCAL, 2)  # the counter
    program[start] = ins(Op.LOAD_CONST, I64_MAX - 3)
    compiled = compile_i64(program)
    assert compiled(I64_MAX) == I64_MAX - 1 == execute(program, (I64_MAX,))
    assert run_i64(program, (I64_MAX,)) == ("ok", I64_MAX - 1)
    assert compiled(I64_MAX - 2) == I64_MAX - 3
    assert compiled(5) == 99


# --- The backedge budget -------------------------------------------------------


@pytest.mark.parametrize("budget", [0, 1, 7, 1000])
def test_n_iterations_take_n_backedges(budget):
    instructions = analyze(conditional_sum).instructions
    compiled = compile_i64(instructions, backedge_budget=budget)
    assert compiled(budget, 3) == conditional_sum(budget, 3)
    with pytest.raises(NativeGuardFailure) as excinfo:
        compiled(budget + 1, 3)
    assert excinfo.value.reason is GuardReason.BUDGET_EXHAUSTED
    assert instructions[excinfo.value.instruction_index].op is Op.JUMP
    with pytest.raises(ExecutionLimitExceeded):
        execute(instructions, (budget + 1, 3), max_backedges=budget)


@pytest.mark.parametrize("n", [10**6 + 1, 2**40, I64_MAX])
def test_huge_bounds_stop_at_the_budget(n):
    instructions = analyze(sum_squares).instructions
    compiled = compile_i64(instructions, backedge_budget=1000)
    with pytest.raises(NativeGuardFailure) as excinfo:
        compiled(n)
    assert excinfo.value.reason is GuardReason.BUDGET_EXHAUSTED
    assert compiled.fast_call((n,)) is None
    assert compiled.function.call((n,))[0] == _native.STATUS_BUDGET_EXHAUSTED
    with pytest.raises(ExecutionLimitExceeded):
        execute(instructions, (n,), max_backedges=1000)
    # Also with the default budget, through @snekit's own fast path: the
    # native attempt returns, without a result.
    wrapped = warmed(sum_squares, 3)
    assert wrapped._jit_state.specialization.fast_call((n,)) is None


@pytest.fixture
def small_budget(monkeypatch):
    """Make ``@snekit`` compile loops with a backedge budget of 10."""
    monkeypatch.setattr(_compiler, "_BACKEDGE_BUDGET", 10)
    return 10


def test_budget_exhaustion_falls_back_with_the_original_arguments(small_budget):
    f = warmed(bound_once, 3)
    assert f._jit_state.compiled.backedge_budget == small_budget
    assert path_of(f, small_budget) == (bound_once(small_budget), "native")
    seen = []

    def profiler(frame, event, arg):
        if event == "call" and frame.f_code is bound_once.__code__:
            seen.append(frame.f_locals["n"])

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        # Native code has set ``n = 0`` ten times when it stops.
        result = f(25)
    finally:
        sys.setprofile(previous)
    assert seen == [25]
    assert result == bound_once(25) == sum(range(25))
    assert f._jit_state.status is _Status.COMPILED
    assert f._jit_state.native_error is None
    assert path_of(f, 4) == (6, "native")


# --- @snekit -------------------------------------------------------------------


@pytest.mark.parametrize(("func", "args"), GOAL)
def test_the_goal_functions_run_natively(func, args):
    wrapped = warmed(func, *args_for(func, 3))
    assert path_of(wrapped, *args) == (func(*args), "native")
    spec = wrapped._jit_state.specialization
    assert isinstance(spec.function, _native.GuardedI64Function)
    assert spec.guard.valid


def test_functions_without_range_get_no_guard():
    for func, args in ((sum_squares_while, (3,)), (lambda a, b: a * b + 1, (2, 3))):
        wrapped = warmed(func, *args)
        spec = wrapped._jit_state.specialization
        assert spec.guard is None
        assert type(spec.function) is _native.I64Function


class CountedIndex:
    calls = 0

    def __init__(self, value):
        self.value = value

    def __index__(self):
        CountedIndex.calls += 1
        return self.value


class IntSubclass(int):
    pass


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (CountedIndex, sum_squares(5)),
        (IntSubclass, sum_squares(5)),
        (lambda n: n == 5, 0),  # True: one iteration of i = 0
    ],
)
def test_non_int_bounds_run_only_in_python(make, expected):
    f = warmed(sum_squares, 3)
    CountedIndex.calls = 0
    assert path_of(f, make(5)) == (expected, "guard_miss")
    # ``range`` called ``__index__`` once; the guards never did.
    assert CountedIndex.calls == (1 if make is CountedIndex else 0)
    assert f._jit_state.status is _Status.COMPILED


# --- The range guard: @snekit ---------------------------------------------------

SOURCE = """
def f(n):
    total = 0
    for i in range(n):
        total += i
    return total
"""


def counting_range(calls):
    def fake(n):
        calls.append(n)
        return RANGE(2)

    return fake


def compiled_in(namespace, source=SOURCE):
    """Define ``f`` in ``namespace``, warm it up and check it is compiled."""
    exec(source, namespace)
    wrapped = snekit(namespace["f"])
    for _ in range(_WARMUP_CALLS + 1):
        assert wrapped(4) == 6
    assert wrapped._jit_state.status is _Status.COMPILED
    assert path_of(wrapped, 5) == (10, "native")
    return wrapped


def test_a_global_shadow_before_compilation_is_never_compiled():
    calls = []
    namespace = {"range": counting_range(calls)}
    exec(SOURCE, namespace)
    wrapped = snekit(namespace["f"])
    for k in range(_WARMUP_CALLS + 3):
        assert wrapped(10) == 1
        assert len(calls) == k + 1
    state = wrapped._jit_state
    assert state.disabled_reason is _DisabledReason.RANGE_UNGUARDED
    assert "not the builtin range" in str(state.compile_error)
    assert state.compiled is None


def test_a_global_bound_to_the_builtin_range_is_compiled():
    namespace = {"range": RANGE}
    wrapped = compiled_in(namespace)
    # Deleting it changes nothing in Python, but is a change to ``range``.
    del namespace["range"]
    assert path_of(wrapped, 5) == (10, "python")
    assert wrapped._jit_state.disabled_reason is _DisabledReason.RANGE_CHANGED


@pytest.mark.parametrize("how", ["add", "replace", "setdefault", "update"])
def test_a_global_shadow_after_compilation_runs_once_in_python(how):
    calls = []
    namespace = {"range": RANGE} if how == "replace" else {}
    wrapped = compiled_in(namespace)
    fake = counting_range(calls)
    if how == "setdefault":
        namespace.setdefault("range", fake)
    elif how == "update":
        namespace.update(range=fake)
    else:
        namespace["range"] = fake
    assert path_of(wrapped, 10) == (1, "python")
    assert calls == [10]
    state = wrapped._jit_state
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.RANGE_CHANGED
    assert state.specialization is None
    # Disabled for good, even once the shadow is gone again.
    del namespace["range"]
    assert wrapped(10) == 45
    assert len(calls) == 1
    assert state.status is _Status.DISABLED


def test_deleting_a_shadow_after_compilation_is_noticed():
    # ``range`` cannot be shadowed at compilation and still compile, but a
    # global that is deleted and re-added is a change all the same.
    namespace = {}
    wrapped = compiled_in(namespace)
    namespace["range"] = RANGE
    del namespace["range"]
    assert path_of(wrapped, 5) == (10, "python")
    assert wrapped._jit_state.disabled_reason is _DisabledReason.RANGE_CHANGED


@pytest.mark.parametrize("change", ["clear", "pop", "non_str_key"])
def test_other_changes_that_could_affect_range(change):
    namespace = {}
    wrapped = compiled_in(namespace)
    if change == "clear":
        namespace.clear()
        namespace["__builtins__"] = builtins.__dict__
    elif change == "pop":
        namespace["range"] = RANGE
        namespace.pop("range")
    else:
        namespace[1] = "a key whose __eq__ CPython could call"
    assert path_of(wrapped, 5) == (10, "python")
    assert wrapped._jit_state.disabled_reason is _DisabledReason.RANGE_CHANGED


def test_unrelated_global_changes_keep_the_specialization():
    namespace = {}
    wrapped = compiled_in(namespace)
    for k in range(50):
        namespace[f"name{k}"] = k
    namespace["rang"] = 1
    namespace["ranges"] = 2
    del namespace["name3"]
    assert path_of(wrapped, 6) == (15, "native")
    assert wrapped._jit_state.status is _Status.COMPILED


class Builtins(dict):
    """A dict subclass: CPython looks names up with ``__getitem__``."""

    lookups = 0

    def __getitem__(self, key):
        Builtins.lookups += 1
        return super().__getitem__(key)


class Key:
    """A non-str key; counts every call of ``__hash__`` and ``__eq__``."""

    calls = 0

    def __hash__(self):
        Key.calls += 1
        return 12345

    def __eq__(self, other):
        Key.calls += 1
        return False


@pytest.mark.parametrize(
    ("make_namespace", "message"),
    [
        (lambda: {"__builtins__": Builtins(range=RANGE)}, "builtins are not"),
        (lambda: {Key(): 1}, "globals have a key"),
        (lambda: {"__builtins__": {Key(): 1, "range": RANGE}}, "builtins have a key"),
        (lambda: {"__builtins__": {}}, "neither a global nor a builtin"),
        (lambda: {"__builtins__": {"range": lambda n: [n]}}, "not the builtin range"),
    ],
)
def test_unusual_namespaces_are_never_compiled(make_namespace, message):
    namespace = make_namespace()
    exec(SOURCE, namespace)
    func = namespace["f"]
    expected = _try(func, 4)
    wrapped = snekit(func)
    calls = _WARMUP_CALLS + 3
    Builtins.lookups = Key.calls = 0
    assert [_try(wrapped, 4) for _ in range(calls)] == [expected] * calls
    state = wrapped._jit_state
    assert state.disabled_reason is _DisabledReason.RANGE_UNGUARDED
    assert message in str(state.compile_error)
    # The guard ran no Python code: every lookup was CPython's own, one
    # per call.
    if isinstance(func.__builtins__, Builtins):
        assert Builtins.lookups == calls
    assert Key.calls == 0


def _try(func, *args):
    try:
        return func(*args)
    except NameError as error:
        return type(error)


def test_a_dict_subclass_as_globals_is_never_compiled():
    class Globals(dict):
        pass

    code = define(SOURCE).__code__
    func = types.FunctionType(code, Globals(__builtins__=builtins.__dict__))
    wrapped = snekit(func)
    for _ in range(_WARMUP_CALLS + 2):
        assert wrapped(4) == 6
    assert wrapped._jit_state.disabled_reason is _DisabledReason.RANGE_UNGUARDED
    assert "globals are not an exact dict" in str(wrapped._jit_state.compile_error)


def test_custom_builtins():
    custom = {"range": RANGE}
    namespace = {"__builtins__": custom}
    wrapped = compiled_in(namespace)
    assert wrapped.__wrapped__.__builtins__ is custom
    # Other builtins and other names change nothing.
    custom["len"] = len
    assert path_of(wrapped, 3) == (3, "native")
    calls = []
    custom["range"] = counting_range(calls)
    assert path_of(wrapped, 7) == (1, "python")
    assert calls == [7]
    assert wrapped._jit_state.disabled_reason is _DisabledReason.RANGE_CHANGED


def test_custom_builtins_from_a_module():
    module = types.ModuleType("fake_builtins")
    module.range = RANGE
    namespace = {"__builtins__": module}
    wrapped = compiled_in(namespace)
    assert wrapped.__wrapped__.__builtins__ is module.__dict__
    del module.range
    with pytest.raises(NameError):
        wrapped(3)
    assert wrapped._jit_state.disabled_reason is _DisabledReason.RANGE_CHANGED


def test_a_global_shadowing_custom_builtins_after_compilation():
    custom = {"range": RANGE}
    namespace = {"__builtins__": custom}
    wrapped = compiled_in(namespace)
    namespace["range"] = lambda n: RANGE(n, n + 2)
    assert path_of(wrapped, 3) == (3 + 4, "python")


def test_the_hot_slot_holds_the_range_guard():
    namespace = {}
    wrapped = compiled_in(namespace)
    state = wrapped._jit_state
    assert state._hot.live
    function = state.compiled.function
    calls = function.try_calls
    # The guard is checked before every native call, without reading the
    # JIT state: a failed guard sends the call to the generic path, which
    # disables the specialization exactly as before.
    namespace["range"] = lambda n: RANGE(1)
    assert path_of(wrapped, 5) == (0, "python")
    assert function.try_calls == calls
    assert state.disabled_reason is _DisabledReason.RANGE_CHANGED
    assert state.specialization is None
    assert not state._hot.live
    # Restoring the builtin does not revive the slot.
    del namespace["range"]
    assert wrapped(5) == 10
    assert function.try_calls == calls
    assert not state._hot.live


def test_a_range_change_between_two_hot_calls_is_noticed():
    namespace = {}
    wrapped = compiled_in(namespace)
    for k in range(20):
        assert path_of(wrapped, k) == (k * (k - 1) // 2, "native")
    namespace["range"] = RANGE
    assert path_of(wrapped, 5) == (10, "python")
    assert not wrapped._jit_state._hot.live


def test_code_and_range_invalidation_work_together():
    other = """
def g(n):
    total = 0
    for i in range(n):
        total += 2 * i
    return total
"""
    # __code__ first, then range: CODE_CHANGED, and the shadow is used.
    namespace = {}
    wrapped = compiled_in(namespace)
    exec(other, namespace)
    func = wrapped.__wrapped__
    func.__code__ = namespace["g"].__code__
    assert path_of(wrapped, 5) == (20, "python")
    assert wrapped._jit_state.disabled_reason is _DisabledReason.CODE_CHANGED
    namespace["range"] = lambda n: RANGE(1)
    assert wrapped(5) == 0
    assert wrapped._jit_state.disabled_reason is _DisabledReason.CODE_CHANGED

    # range first, then __code__: RANGE_CHANGED, and the new code is used.
    namespace = {}
    wrapped = compiled_in(namespace)
    namespace["range"] = lambda n: RANGE(1)
    assert path_of(wrapped, 5) == (0, "python")
    assert wrapped._jit_state.disabled_reason is _DisabledReason.RANGE_CHANGED
    exec(other, namespace)
    wrapped.__wrapped__.__code__ = namespace["g"].__code__
    del namespace["range"]
    assert wrapped(5) == 20
    assert wrapped._jit_state.disabled_reason is _DisabledReason.RANGE_CHANGED

    # Both before the next call: the code check comes first.
    namespace = {}
    wrapped = compiled_in(namespace)
    exec(other, namespace)
    namespace["range"] = RANGE
    wrapped.__wrapped__.__code__ = namespace["g"].__code__
    assert path_of(wrapped, 5) == (20, "python")
    assert wrapped._jit_state.disabled_reason is _DisabledReason.CODE_CHANGED


def test_a_range_function_whose_code_is_replaced_by_a_loop_free_one():
    namespace = {}
    wrapped = compiled_in(namespace)
    wrapped.__wrapped__.__code__ = (lambda n: n + 1).__code__
    assert wrapped(5) == 6
    assert wrapped._jit_state.disabled_reason is _DisabledReason.CODE_CHANGED


def test_shadowing_range_does_not_affect_functions_without_it():
    namespace = {}
    exec(
        "def f(n):\n    i = 0\n    t = 0\n    while i < n:\n"
        "        t += i\n        i += 1\n    return t\n",
        namespace,
    )
    wrapped = warmed(namespace["f"], 3)
    namespace["range"] = None
    assert path_of(wrapped, 5) == (10, "native")
    assert wrapped._jit_state.status is _Status.COMPILED


# --- The range guard: the native object -----------------------------------------


def test_the_guard_watches_only_what_matters():
    globals_, builtins_ = {}, {"range": RANGE, "len": len}
    guard = _native.RangeGuard(globals_, builtins_)
    other = _native.RangeGuard(globals_, builtins_)
    assert guard.valid and other.valid
    globals_["x"] = 1
    builtins_["len"] = abs
    del builtins_["len"]
    assert guard.valid
    # Storing the same object again is no change: CPython does not notify.
    builtins_["range"] = RANGE
    assert guard.valid
    builtins_["range"] = list
    assert not guard.valid and not other.valid
    # It never becomes valid again.
    builtins_["range"] = RANGE
    assert not guard.valid


@pytest.mark.parametrize(
    "change",
    [
        lambda g, b: g.update({"y": 1, "range": 2}),
        lambda g, b: b.clear(),
        lambda g, b: g.__setitem__((1, 2), 3),
        lambda g, b: b.pop("range"),
        lambda g, b: b.popitem(),
        lambda g, b: g.update(b),  # g is empty: CLONED
    ],
)
def test_guard_invalidating_changes(change):
    globals_, builtins_ = {}, {"range": RANGE}
    guard = _native.RangeGuard(globals_, builtins_)
    change(globals_, builtins_)
    assert not guard.valid


def test_a_guard_on_a_dict_that_dies_is_invalid():
    globals_, builtins_ = {}, {"range": RANGE}
    guard = _native.RangeGuard(globals_, builtins_)
    assert guard.valid
    del builtins_
    gc.collect()
    assert not guard.valid
    # New dicts, possibly at the same address, are watched afresh.
    for _ in range(100):
        fresh = {"range": RANGE}
        assert _native.RangeGuard(globals_, fresh).valid


@pytest.mark.parametrize(
    ("globals_", "builtins_", "message"),
    [
        ({"range": list}, {"range": RANGE}, "not the builtin range"),
        ({}, {"range": lambda n: n}, "not the builtin range"),
        ({}, {}, "neither a global nor a builtin"),
        ({}, types.MappingProxyType({"range": RANGE}), "builtins are not"),
        (collections.OrderedDict(), {"range": RANGE}, "globals are not"),
        ({1: 2}, {"range": RANGE}, "globals have a key"),
        ({}, {"range": RANGE, 2.5: 1}, "builtins have a key"),
    ],
)
def test_unguardable_bindings(globals_, builtins_, message):
    with pytest.raises(_native.UnguardableBinding, match=message):
        _native.RangeGuard(globals_, builtins_)


def test_a_global_str_subclass_key_is_unguardable():
    class Name(str):
        def __eq__(self, other):
            raise AssertionError("the guard must not compare keys")

        __hash__ = str.__hash__

    with pytest.raises(_native.UnguardableBinding, match="globals have a key"):
        _native.RangeGuard({Name("range"): list}, {"range": RANGE})


def test_the_guarded_function_checks_its_guard():
    namespace = {}
    wrapped = compiled_in(namespace)
    guarded = wrapped._jit_state.specialization.function
    assert guarded.arg_count == 1
    assert guarded.try_call((4,)) == 6
    namespace["range"] = RANGE
    # ``try_call`` reports a failed guard as a guard miss, and runs nothing.
    before = guarded.function.try_calls
    assert guarded.try_call((4,)) is None
    assert guarded.function.try_calls == before
    with pytest.raises(TypeError):
        guarded.try_call([4])


# --- The real builtins module: in a child process -------------------------------


def test_changing_builtins_range():
    stdout = run_child(
        """
        import builtins
        from speedysnek import snekit

        def f(n):
            t = 0
            for i in range(n):
                t += i
            return t

        def g(n):
            i = 0
            t = 0
            while i < n:
                t += i
                i += 1
            return t

        real = builtins.range
        wf, wg = snekit(f), snekit(g)
        for _ in range(10):
            wf(5), wg(5)
        print(wf._jit_state.status.value, wg._jit_state.status.value)
        calls = []
        builtins.range = lambda n: (calls.append(n), real(1))[1]
        print(wf(5), calls, wf._jit_state.disabled_reason.value)
        print(wg(5), wg._jit_state.status.value)
        del builtins.range
        try:
            wf(5)
        except NameError as error:
            print("NameError")
        builtins.range = real
        print(wf(5), wf._jit_state.status.value)

        # Shadowed before compilation by an equivalent that is not the
        # builtin (SpeedySnek's own code calls range too): never compiled.
        class Shadow:
            def __new__(cls, *args):
                calls.append(args)
                return real(*args)

        builtins.range = Shadow
        h = snekit(f)
        del calls[:]
        print([h(9) for _ in real(12)][-1], h._jit_state.disabled_reason.value)
        print(calls.count((9,)))
        builtins.range = real
        print(h(9))
        """
    )
    assert stdout.split("\n")[:-1] == [
        "compiled compiled",
        "0 [5] range_changed",
        "10 compiled",
        "NameError",
        "10 disabled",
        "36 range_unguarded",
        "12",
        "36",
    ]


# --- Differential: random for loops ---------------------------------------------


def random_for_loop(rng):
    """Source of a random function with one for loop over range."""
    names = ["a", "t", "u", "n"]

    def expression(depth=2, extra=()):
        if depth == 0 or rng.random() < 0.4:
            return rng.choice([*names, *extra, "1", "2", "7", str(2**40)])
        if rng.random() < 0.3:
            return f"({expression(depth - 1, extra)} * {rng.choice([-3, 5, 2**40])})"
        op = rng.choice(["+", "-"])
        return f"({expression(depth - 1, extra)} {op} {expression(depth - 1, extra)})"

    def block(indent, depth, count, extra=(), targets=("t", "u")):
        lines = []
        for _ in range(count):
            if depth and rng.random() < 0.3:
                op = rng.choice(["<", "<=", ">", ">=", "==", "!="])
                lines.append(
                    f"{indent}if {expression(1, extra)} {op} {expression(1, extra)}:"
                )
                inner = (indent + "    ", depth - 1, rng.randint(1, 2), extra, targets)
                lines += block(*inner)
                if rng.random() < 0.5:
                    lines.append(f"{indent}else:")
                    lines += block(*inner)
            else:
                target = rng.choice(targets)
                if rng.random() < 0.2:
                    lines.append(f"{indent}{target} *= {rng.choice([-2, 3, 2**33])}")
                else:
                    op = rng.choice(["=", "+=", "-="])
                    lines.append(f"{indent}{target} {op} {expression(2, extra)}")
        return lines

    bound = rng.choice(["n", "n + a", "n * 2 - a", "a - n", "7", "n - 3"])
    initialized = rng.random() < 0.6
    lines = [
        "def f(n, a):",
        "    t = 1",
        "    u = a - 1",
        *(["    i = a * 3"] if initialized else []),
        *block("    ", 1, rng.randint(0, 2)),
        f"    for i in range({bound}):",
        # The body may assign the loop variable and the bound's inputs; the
        # code before the loop does not, so the range stays small.
        *block("        ", 2, rng.randint(1, 3), ("i",), ("t", "u", "n", "a", "i")),
        *block("    ", 1, rng.randint(0, 2)),
        f"    return {expression(2, ('i',) if initialized else ())}",
    ]
    return "\n".join(lines) + "\n"


def test_differential_random_for_loops():
    # CPython, @snekit, the native code, the interpreter and ``run_i64``.
    rng = random.Random("speedysnek-for-range")
    paths = collections.Counter()
    for _ in range(150):
        source = random_for_loop(rng)
        func = define(source)
        analysis = analyze(func)
        assert analysis.supported, (source, analysis.rejection)
        assert analysis.assumed_builtins == ("range",)
        wrapped = snekit(func)
        for _ in range(_WARMUP_CALLS + 1):
            wrapped(0, 0)
        if wrapped._jit_state.disabled_reason is _DisabledReason.UNSUPPORTED:
            # CPython folded a product of constants beyond i64.
            paths["constant"] += 1
            continue
        assert wrapped._jit_state.status is _Status.COMPILED, source
        for _ in range(4):
            args = (rng.randint(-3, 12), rng.randint(-5, 5))
            expected = func(*args)
            status = differential(func, args)
            result, path = path_of(wrapped, *args)
            assert result == expected, source
            assert path == ("native" if status == "ok" else "guard_miss"), source
            paths[status] += 1
    assert paths["ok"] > 250 and paths["overflow"] > 10, paths
    assert paths["constant"] < 40, paths
