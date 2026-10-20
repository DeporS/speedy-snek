"""Tests for the private CPython 3.14 bytecode frontend.

Expected opcodes and offsets are exact: they are what CPython 3.14 emits,
which is the only version the frontend supports.
"""

import dis
from types import SimpleNamespace

import pytest

import speedysnek
from speedysnek import _frontend, snekit
from speedysnek._frontend import (
    Analysis,
    Instruction,
    Op,
    Rejection,
    RejectionReason,
    analyze,
)
from speedysnek._interpreter import execute, validate

LOAD_ARG_0 = Instruction(Op.LOAD_ARG, 0)
LOAD_ARG_1 = Instruction(Op.LOAD_ARG, 1)
ADD = Instruction(Op.ADD)
SUBTRACT = Instruction(Op.SUBTRACT)
MULTIPLY = Instruction(Op.MULTIPLY)
RETURN = Instruction(Op.RETURN)


def const(value):
    return Instruction(Op.LOAD_CONST, value)


def assert_supported(func, expected):
    analysis = analyze(func)
    assert analysis.supported, analysis.rejection
    assert analysis.rejection is None
    assert analysis.instructions == tuple(expected)


def assert_rejected(func, reason, opname=None, offset=None):
    analysis = analyze(func)
    assert not analysis.supported
    assert analysis.instructions == ()
    rejection = analysis.rejection
    assert rejection.reason is reason
    assert rejection.opname == opname
    assert rejection.offset == offset
    return rejection


def define(source):
    """Compile a single ``def f`` from source and return it, without calling it."""
    namespace = {}
    exec(source, namespace)
    return namespace["f"]


# --- Supported functions -------------------------------------------------


def test_add():
    def add(a, b):
        return a + b

    assert_supported(add, [LOAD_ARG_0, LOAD_ARG_1, ADD, RETURN])


def test_subtract():
    def sub(a, b):
        return a - b

    assert_supported(sub, [LOAD_ARG_0, LOAD_ARG_1, SUBTRACT, RETURN])


def test_multiply():
    def mul(a, b):
        return a * b

    assert_supported(mul, [LOAD_ARG_0, LOAD_ARG_1, MULTIPLY, RETURN])


def test_arguments_and_constants():
    # 2 and 3 use LOAD_SMALL_INT; 1000 and -7 use LOAD_CONST.
    def f(a, b):
        return (a + 2) * b - 3 + 1000 * -7

    assert_supported(
        f,
        [
            LOAD_ARG_0,
            const(2),
            ADD,
            LOAD_ARG_1,
            MULTIPLY,
            const(3),
            SUBTRACT,
            const(-7000),  # 1000 * -7 is constant-folded by CPython.
            ADD,
            RETURN,
        ],
    )


def test_constant_folding_happens_before_analysis():
    def f():
        return 2 * 3 + 1

    assert_supported(f, [const(7), RETURN])


def test_small_int_boundary_and_large_ints():
    def f(a):
        return a + 0 + 255 + 256 + 10**30

    assert_supported(
        f,
        [
            LOAD_ARG_0,
            const(0),
            ADD,
            const(255),
            ADD,
            const(256),
            ADD,
            const(10**30),
            ADD,
            RETURN,
        ],
    )


def test_return_argument_or_constant():
    def identity(a):
        return a

    def five():
        return 5

    assert_supported(identity, [LOAD_ARG_0, RETURN])
    assert_supported(five, [const(5), RETURN])


def test_argument_order_and_reuse():
    def f(a, b, c):
        return c - a * a + b

    assert_supported(
        f,
        [
            Instruction(Op.LOAD_ARG, 2),
            LOAD_ARG_0,
            LOAD_ARG_0,
            MULTIPLY,
            SUBTRACT,
            LOAD_ARG_1,
            ADD,
            RETURN,
        ],
    )


def test_argument_names_do_not_affect_output():
    def f(a, b):
        return a * 3 + b

    def g(width, height):
        return width * 3 + height

    assert analyze(f) == analyze(g)


def test_lambda_docstring_annotations_defaults_and_positional_only():
    def documented(a: int, b: int = 1, /) -> int:
        """CPython 3.14 keeps the docstring out of the bytecode."""
        return a + b

    assert_supported(documented, [LOAD_ARG_0, LOAD_ARG_1, ADD, RETURN])
    assert_supported(lambda a, b: a + b, [LOAD_ARG_0, LOAD_ARG_1, ADD, RETURN])


def test_fused_loads_split_at_index_16():
    # LOAD_FAST_BORROW_LOAD_FAST_BORROW packs both indices into 4 bits each,
    # so a14 and a15 are fused but a16 is loaded on its own.
    params = ", ".join(f"a{i}" for i in range(17))
    f = define(f"def f({params}):\n    return a14 + a15 + a16\n")

    opnames = [i.opname for i in dis.get_instructions(f)]
    assert opnames[1:3] == ["LOAD_FAST_BORROW_LOAD_FAST_BORROW", "BINARY_OP"]
    assert "LOAD_FAST_BORROW" in opnames

    assert_supported(
        f,
        [
            Instruction(Op.LOAD_ARG, 14),
            Instruction(Op.LOAD_ARG, 15),
            ADD,
            Instruction(Op.LOAD_ARG, 16),
            ADD,
            RETURN,
        ],
    )


def test_many_arguments_use_load_fast_check_and_extended_arg():
    # Arguments at index >= 64 are loaded with LOAD_FAST_CHECK, and indices
    # >= 256 need an EXTENDED_ARG prefix.
    params = ", ".join(f"a{i}" for i in range(300))
    f = define(f"def f({params}):\n    return a299 + a64 + a0\n")

    opnames = [i.opname for i in dis.get_instructions(f)]
    assert opnames[1:3] == ["EXTENDED_ARG", "LOAD_FAST_CHECK"]

    assert_supported(
        f,
        [
            Instruction(Op.LOAD_ARG, 299),
            Instruction(Op.LOAD_ARG, 64),
            ADD,
            LOAD_ARG_0,
            ADD,
            RETURN,
        ],
    )


# --- Determinism and side effects ----------------------------------------


def test_repeated_analysis_is_deterministic():
    def f(a, b):
        return a * b - 1

    first = analyze(f)
    assert all(analyze(f) == first for _ in range(5))


def test_specialization_after_running_does_not_change_the_result():
    def f(a, b):
        return a + b

    before = analyze(f)
    for i in range(10_000):
        f(i, 1)  # Lets CPython specialize BINARY_OP into BINARY_OP_ADD_INT.
    assert analyze(f) == before


def test_analysis_does_not_call_the_function():
    calls = []

    def f():
        calls.append(1)

    analyze(f)
    assert calls == []


def test_analyzing_a_jit_wrapper_does_not_call_it():
    @snekit
    def add(a, b):
        return a + b

    # The wrapper is a native object, not a function, so it is rejected as
    # a whole.
    assert_rejected(add, RejectionReason.NOT_A_FUNCTION)
    assert_supported(add.__wrapped__, [LOAD_ARG_0, LOAD_ARG_1, ADD, RETURN])
    assert add.call_count == 0


# --- Rejections ----------------------------------------------------------


def test_rejects_division():
    def true_div(a, b):
        return a / b

    def floor_div(a, b):
        return a // b

    rejection = assert_rejected(
        true_div, RejectionReason.UNSUPPORTED_OPERATOR, "BINARY_OP", 4
    )
    assert "'/'" in rejection.message
    assert "offset 4" in rejection.message
    assert rejection.lineno == true_div.__code__.co_firstlineno + 1

    assert_rejected(floor_div, RejectionReason.UNSUPPORTED_OPERATOR, "BINARY_OP", 4)


def test_rejects_other_operators():
    def f(a, b):
        return a % b

    def neg(a):
        return -a

    assert_rejected(f, RejectionReason.UNSUPPORTED_OPERATOR, "BINARY_OP", 4)
    assert_rejected(neg, RejectionReason.UNSUPPORTED_OPCODE, "UNARY_NEGATIVE", 4)


def test_rejects_calls():
    def call_builtin(a):
        return abs(a)

    def call_argument(g, a):
        return g(a)

    # Calling a builtin first needs a global load.
    assert_rejected(call_builtin, RejectionReason.UNSUPPORTED_OPCODE, "LOAD_GLOBAL", 2)
    assert_rejected(call_argument, RejectionReason.UNSUPPORTED_OPCODE, "PUSH_NULL", 4)


GLOBAL = 1


def test_rejects_globals():
    def f(a):
        return a + GLOBAL

    rejection = assert_rejected(f, RejectionReason.UNSUPPORTED_OPCODE, "LOAD_GLOBAL", 4)
    assert rejection.message == "unsupported opcode LOAD_GLOBAL at offset 4"


def test_rejects_attribute_access():
    def f(a):
        return a.real

    assert_rejected(f, RejectionReason.UNSUPPORTED_OPCODE, "LOAD_ATTR", 4)


def test_rejects_conditions_that_are_not_comparisons():
    def f(a):
        if a:
            return 1
        return 2

    def ternary(a, b):
        return a if b else 1

    def identity(a, b):
        if a is b:
            return 1
        return 2

    def membership(a, b):
        if a in b:
            return 1
        return 2

    # Each condition is reported at its jump, not at TO_BOOL, IS_OP or
    # CONTAINS_OP.
    for func, offset in ((f, 12), (ternary, 12), (identity, 6), (membership, 8)):
        rejection = assert_rejected(
            func, RejectionReason.UNSUPPORTED_CONDITION, "POP_JUMP_IF_FALSE", offset
        )
        assert "only a comparison can be an if or while condition" in rejection.message


def test_rejects_loops_outside_the_subset():
    # Supported while loops are tested in test_loops.py.
    def while_truthy(a):
        while a:
            a = a - 1
        return a

    def for_loop(a):
        for _ in a:
            pass
        return 0

    # A while condition must be a comparison, like an if condition.
    assert_rejected(
        while_truthy, RejectionReason.UNSUPPORTED_CONDITION, "POP_JUMP_IF_FALSE", 12
    )
    assert_rejected(for_loop, RejectionReason.UNSUPPORTED_ITERATION, "FOR_ITER", 6)


def test_rejects_reading_a_non_argument_local():
    # Nothing ever stores ``y``, so CPython emits a lone LOAD_FAST_CHECK.
    f = define("def f():\n    return y\n    y = 1\n")
    rejection = assert_rejected(
        f, RejectionReason.UNSUPPORTED_VARIABLE, "LOAD_FAST_CHECK", 2
    )
    assert "'y' may be read before it is assigned" in rejection.message


def test_rejects_read_before_assignment():
    def f(a):
        y = x + a  # noqa: F821 -- ``x`` is the local assigned below.
        x = 1  # noqa: F841
        return y

    assert_rejected(f, RejectionReason.UNSUPPORTED_VARIABLE, "LOAD_FAST_CHECK", 2)
    with pytest.raises(UnboundLocalError):
        f(1)


def test_rejects_self_referencing_first_assignment():
    def f(a):
        x = x + a  # noqa: F821
        return x

    assert_rejected(f, RejectionReason.UNSUPPORTED_VARIABLE, "LOAD_FAST_CHECK", 2)


def test_rejects_assignment_forms_outside_the_subset():
    # Each is rejected at its first unsupported instruction.
    cases = [
        # Tuple assignment and swaps use STORE_FAST_STORE_FAST directly.
        ("x, y = a, b\n    return x - y", "STORE_FAST_STORE_FAST", 4),
        ("a, b = b, a\n    return a - b", "STORE_FAST_STORE_FAST", 4),
        # Chained assignment and the walrus operator duplicate a value first.
        ("x = y = a + 1\n    return x * y", "COPY", 18),
        ("return (x := a + 1) * x", "COPY", 18),
        ("del a\n    return b", "DELETE_FAST", 2),
    ]
    for body, opname, offset in cases:
        f = define(f"def f(a, b):\n    {body}\n")
        assert_rejected(f, RejectionReason.UNSUPPORTED_OPCODE, opname, offset)

    # Augmented assignment is BINARY_OP with an in-place operator; only
    # ``+=``, ``-=`` and ``*=`` are supported.
    for operator in ("/=", "//=", "%=", "**=", "<<=", ">>=", "&=", "|=", "^=", "@="):
        f = define(f"def f(a):\n    a {operator} 1\n    return a\n")
        rejection = assert_rejected(
            f, RejectionReason.UNSUPPORTED_OPERATOR, "BINARY_OP", 6
        )
        assert repr(operator) in rejection.message


def test_rejects_non_local_stores():
    sources = {
        "global G\n    G = a\n    return a": "STORE_GLOBAL",
        "a.x = 1\n    return 1": "STORE_ATTR",
        "a[0] = 1\n    return 1": "STORE_SUBSCR",
    }
    for body, opname in sources.items():
        rejection = analyze(define(f"def f(a):\n    {body}\n")).rejection
        assert rejection.reason is RejectionReason.UNSUPPORTED_OPCODE
        assert rejection.opname == opname


def test_rejects_augmented_assignment_to_non_locals():
    # Each is rejected before its in-place BINARY_OP is reached.
    cases = [
        ("a.x += 1", "COPY", 4),
        ("a[0] += 1", "COPY", 6),
        ("global G\n    G += a", "LOAD_GLOBAL", 2),
        # A variable captured by an inner function lives in a cell.
        ("a += 1\n    g = lambda: a", "MAKE_CELL", 0),
    ]
    for body, opname, offset in cases:
        f = define(f"def f(a):\n    {body}\n    return a\n")
        assert_rejected(f, RejectionReason.UNSUPPORTED_OPCODE, opname, offset)

    def outer():
        x = 0

        def inner(a):
            nonlocal x
            x += a
            return x

        return inner

    assert_rejected(outer(), RejectionReason.CLOSURE)


# --- Local variables -----------------------------------------------------


def load(slot):
    return Instruction(Op.LOAD_LOCAL, slot)


def store(slot):
    return Instruction(Op.STORE_LOCAL, slot)


def test_local_variables():
    def calculate_score(wins, draws, penalties):
        points = wins * 3 + draws
        deduction = penalties * 2
        result = points - deduction
        return result

    assert_supported(
        calculate_score,
        [
            LOAD_ARG_0,
            const(3),
            MULTIPLY,
            LOAD_ARG_1,
            ADD,
            store(0),  # points
            Instruction(Op.LOAD_ARG, 2),
            const(2),
            MULTIPLY,
            store(1),  # deduction
            load(0),
            load(1),
            SUBTRACT,
            store(2),  # result
            load(2),
            RETURN,
        ],
    )


def test_reused_local_is_computed_once():
    # ``total`` is read twice through the fused LOAD_FAST_BORROW pair, but
    # ``a + b`` appears once.
    def reuse(a, b):
        total = a + b
        return total * total

    assert_supported(
        reuse,
        [LOAD_ARG_0, LOAD_ARG_1, ADD, store(0), load(0), load(0), MULTIPLY, RETURN],
    )


def test_reassigned_local_keeps_its_slot():
    def reassign(a, b):
        value = a + b
        value = value * 3
        return value - a

    assert_supported(
        reassign,
        [
            LOAD_ARG_0,
            LOAD_ARG_1,
            ADD,
            store(0),
            load(0),
            const(3),
            MULTIPLY,
            store(0),
            load(0),
            LOAD_ARG_0,
            SUBTRACT,
            RETURN,
        ],
    )


def test_reassigned_argument_gets_a_local_slot():
    def overwrite_argument(a, b):
        a = a + 1
        return a * b

    assert_supported(
        overwrite_argument,
        [LOAD_ARG_0, const(1), ADD, store(0), load(0), LOAD_ARG_1, MULTIPLY, RETURN],
    )


def test_argument_reads_before_and_after_reassignment():
    # The original ``a`` is LOAD_ARG until the store, LOAD_LOCAL after it;
    # ``b`` is never assigned and stays LOAD_ARG.
    def f(a, b):
        c = a * 2
        a = b
        return a + c - b

    assert_supported(
        f,
        [
            LOAD_ARG_0,
            const(2),
            MULTIPLY,
            store(0),  # c
            LOAD_ARG_1,
            store(1),  # a
            load(1),
            load(0),
            ADD,
            LOAD_ARG_1,
            SUBTRACT,
            RETURN,
        ],
    )


def test_copying_between_locals():
    # CPython uses the owned LOAD_FAST for a copy.
    def copy_value(a, b):
        x = a
        y = x
        x = b
        return y - x

    opnames = [i.opname for i in dis.get_instructions(copy_value)]
    assert opnames[1:4] == ["LOAD_FAST", "STORE_FAST", "LOAD_FAST"]
    assert_supported(
        copy_value,
        [
            LOAD_ARG_0,
            store(0),  # x
            load(0),
            store(1),  # y
            LOAD_ARG_1,
            store(0),  # x again
            load(1),
            load(0),
            SUBTRACT,
            RETURN,
        ],
    )


def test_unused_and_overwritten_computations_are_kept():
    def overwritten(a):
        value = a + 1
        value = 0
        return value

    def unused(a):
        t = a * a  # noqa: F841
        return a

    assert_supported(
        overwritten,
        [LOAD_ARG_0, const(1), ADD, store(0), const(0), store(0), load(0), RETURN],
    )
    assert_supported(
        unused, [LOAD_ARG_0, LOAD_ARG_0, MULTIPLY, store(0), LOAD_ARG_0, RETURN]
    )


def test_self_assignment_and_constant_locals():
    def self_assign(a):
        a = a
        return a

    def constants():
        x = 1
        y = 1000
        return x + y

    assert_supported(self_assign, [LOAD_ARG_0, store(0), load(0), RETURN])
    assert_supported(
        constants,
        [const(1), store(0), const(1000), store(1), load(0), load(1), ADD, RETURN],
    )


def test_annotated_assignment_is_a_plain_assignment():
    def f(a):
        x: int = a + 1
        return x

    assert_supported(f, [LOAD_ARG_0, const(1), ADD, store(0), load(0), RETURN])


@pytest.mark.parametrize(
    "names",
    [
        ("stack", "slot", "locals"),
        ("LOAD_LOCAL", "STORE_LOCAL", "LOAD_ARG"),
        ("_", "__", "__return__"),
        ("result", "value", "index"),
        ("arg0", "local0", "tmp"),
    ],
)
def test_local_names_do_not_affect_output(names):
    x, y, z = names
    f = define(
        f"def f(a, b):\n    {x} = a + b\n    {y} = {x} * a\n"
        f"    {z} = {y} - {x}\n    {x} = {z}\n    return {x} * {y}\n"
    )

    def reference(a, b):
        p = a + b
        q = p * a
        r = q - p
        p = r
        return p * q

    assert analyze(f) == analyze(reference)


def test_slots_follow_first_assignment():
    # Slots are numbered by first store, not by ``co_varnames`` position:
    # the argument ``a`` is assigned second, so it gets slot 1.
    def f(a, b):
        x = b
        a = x
        y = a
        return y

    assert f.__code__.co_varnames == ("a", "b", "x", "y")
    assert_supported(
        f, [LOAD_ARG_1, store(0), load(0), store(1), load(1), store(2), load(2), RETURN]
    )


def test_many_locals_use_extended_arg():
    # Local indexes >= 256 need an EXTENDED_ARG prefix.
    body = "".join(f"    v{i} = a + {i}\n" for i in range(300))
    f = define(f"def f(a):\n{body}    return v299 - v70 + v0\n")
    opnames = [i.opname for i in dis.get_instructions(f)]
    assert "EXTENDED_ARG" in opnames

    expected = []
    for i in range(300):
        expected += [LOAD_ARG_0, const(i), ADD, store(i)]
    expected += [load(299), load(70), SUBTRACT, load(0), ADD, RETURN]
    assert_supported(f, expected)


def test_reassigning_an_argument_read_with_load_fast_check():
    # Arguments at index >= 64 are read with LOAD_FAST_CHECK; after the
    # store, CPython reads ``a65`` with LOAD_FAST_BORROW. Either way the
    # frontend's own tracking decides between LOAD_ARG and LOAD_LOCAL.
    params = ", ".join(f"a{i}" for i in range(66))
    f = define(f"def f({params}):\n    a65 = a65 + 1\n    return a65 * a64\n")
    opnames = [i.opname for i in dis.get_instructions(f)]
    assert opnames[1] == "LOAD_FAST_CHECK"
    assert opnames[5:7] == ["LOAD_FAST_BORROW", "LOAD_FAST_CHECK"]
    assert_supported(
        f,
        [
            Instruction(Op.LOAD_ARG, 65),
            const(1),
            ADD,
            store(0),
            load(0),
            Instruction(Op.LOAD_ARG, 64),
            MULTIPLY,
            RETURN,
        ],
    )


# --- Augmented assignment ------------------------------------------------


def test_augmented_assignment_on_a_local():
    def calculate_score(wins, draws, penalties):
        points = wins * 3
        points += draws
        points -= penalties * 2
        return points

    operators = [i.argrepr for i in dis.get_instructions(calculate_score)]
    assert "+=" in operators and "-=" in operators
    assert_supported(
        calculate_score,
        [
            LOAD_ARG_0,
            const(3),
            MULTIPLY,
            store(0),
            load(0),
            LOAD_ARG_1,
            ADD,
            store(0),
            load(0),
            Instruction(Op.LOAD_ARG, 2),
            const(2),
            MULTIPLY,
            SUBTRACT,
            store(0),
            load(0),
            RETURN,
        ],
    )


def test_augmented_assignment_on_an_argument():
    # The first ``a`` is the argument; after ``a += b`` it is slot 0.
    def update_argument(a, b):
        a += b
        a *= 3
        a -= b
        return a

    assert_supported(
        update_argument,
        [
            LOAD_ARG_0,
            LOAD_ARG_1,
            ADD,
            store(0),
            load(0),
            const(3),
            MULTIPLY,
            store(0),
            load(0),
            LOAD_ARG_1,
            SUBTRACT,
            store(0),
            load(0),
            RETURN,
        ],
    )


def test_augmented_assignment_matches_the_plain_form():
    # ``x op= e`` is ``x = x op e`` with an in-place operator, so both
    # normalize to the same instructions.
    for op in "+-*":
        augmented = define(f"def f(a, b):\n    a {op}= b * a\n    return a - b\n")
        plain = define(f"def f(a, b):\n    a = a {op} (b * a)\n    return a - b\n")
        assert analyze(augmented) == analyze(plain)
        assert analyze(augmented).supported


def test_augmented_assignment_reads_the_target_first():
    # ``x`` is loaded before the right-hand side, which also reads it.
    def example(a, b):
        x = a
        x -= x * b
        return x

    assert_supported(
        example,
        [
            LOAD_ARG_0,
            store(0),
            load(0),
            load(0),
            LOAD_ARG_1,
            MULTIPLY,
            SUBTRACT,
            store(0),
            load(0),
            RETURN,
        ],
    )


def test_augmented_assignment_after_a_copy():
    # ``y`` keeps slot 1, which the update of ``x`` never stores to.
    def copy_before_update(a, b):
        x = a
        y = x
        x += b
        return y - x

    assert_supported(
        copy_before_update,
        [
            LOAD_ARG_0,
            store(0),
            load(0),
            store(1),
            load(0),
            LOAD_ARG_1,
            ADD,
            store(0),
            load(1),
            load(0),
            SUBTRACT,
            RETURN,
        ],
    )


def test_overwritten_augmented_result_is_kept():
    def overwritten(a):
        a += 1
        a = 0
        return a

    assert_supported(
        overwritten,
        [LOAD_ARG_0, const(1), ADD, store(0), const(0), store(0), load(0), RETURN],
    )


def test_rejects_augmented_assignment_before_assignment():
    def unbound(a):
        x += a  # noqa: F821
        return x

    assert_rejected(unbound, RejectionReason.UNSUPPORTED_VARIABLE, "LOAD_FAST_CHECK", 2)
    with pytest.raises(UnboundLocalError):
        unbound(1)


def test_ir_size_is_linear_in_the_source():
    # Each line squares the previous value. Inlining assignments would
    # double the IR per line; slots keep it at a constant per line.
    lines = "".join("    x = x * x\n" for _ in range(200))
    f = define(f"def f(a):\n    x = a\n{lines}    return x\n")
    instructions = analyze(f).instructions
    assert len(instructions) == 2 + 200 * 4 + 2
    assert instructions.count(Instruction(Op.MULTIPLY)) == 200


@pytest.mark.parametrize(
    "source",
    [
        "def f(a):\n    return a + 1.5\n",
        "def f(a):\n    return a + 'x'\n",
        "def f(a):\n    return a + True\n",  # bool is not accepted as int.
        "def f(a):\n    return a + 1j\n",
    ],
)
def test_rejects_non_int_constants(source):
    assert_rejected(
        define(source), RejectionReason.UNSUPPORTED_CONSTANT, "LOAD_CONST", 4
    )


def test_rejects_discarded_expression():
    def f(a):
        a + 1

    assert_rejected(f, RejectionReason.UNSUPPORTED_OPCODE, "POP_TOP", 18)


def test_rejects_implicit_return_none():
    def f(a):
        pass

    rejection = assert_rejected(
        f, RejectionReason.UNSUPPORTED_CONSTANT, "LOAD_CONST", 2
    )
    assert "NoneType None" in rejection.message


def test_rejects_generators():
    def gen(a):
        yield a

    rejection = assert_rejected(gen, RejectionReason.GENERATOR)
    assert rejection.message == "generators are not supported"


def test_rejects_async_functions():
    async def coro(a, b):
        return a + b

    async def agen(a):
        yield a

    assert_rejected(coro, RejectionReason.COROUTINE)
    assert_rejected(agen, RejectionReason.ASYNC_GENERATOR)


def test_rejects_closures():
    x = 1

    def inner(a):
        return a + x

    def outer(a):
        def g():
            return a

        return a

    assert_rejected(inner, RejectionReason.CLOSURE)
    # The enclosing function turns ``a`` into a cell.
    assert_rejected(outer, RejectionReason.UNSUPPORTED_OPCODE, "MAKE_CELL", 0)


def test_rejects_try_except():
    def f(a, b):
        try:
            return a + b
        except:  # noqa: E722
            return 0

    def with_branch(a, b):
        try:
            if a < b:
                a = b
        except Exception:
            pass
        return a

    # The exception table adds edges the analysis does not model, so it is
    # rejected before any instruction is looked at.
    for func in (f, with_branch):
        rejection = assert_rejected(func, RejectionReason.CONTROL_FLOW)
        assert "exception handling" in rejection.message


@pytest.mark.parametrize(
    "source",
    [
        "def f(a, *, b):\n    return a + b\n",
        "def f(*args):\n    return 1\n",
        "def f(**kwargs):\n    return 1\n",
    ],
)
def test_rejects_non_positional_signatures(source):
    assert_rejected(define(source), RejectionReason.UNSUPPORTED_SIGNATURE)


@pytest.mark.parametrize("obj", [len, print, 42, None, object(), str.upper])
def test_rejects_non_functions(obj):
    assert_rejected(obj, RejectionReason.NOT_A_FUNCTION)


def test_rejects_other_python_versions(monkeypatch):
    fake_sys = SimpleNamespace(
        implementation=SimpleNamespace(name="cpython"), version_info=(3, 15, 0)
    )
    monkeypatch.setattr(_frontend, "sys", fake_sys)

    def add(a, b):
        return a + b

    assert_rejected(add, RejectionReason.UNSUPPORTED_PYTHON_VERSION)


# --- Conditionals --------------------------------------------------------

LOAD_ARG_2 = Instruction(Op.LOAD_ARG, 2)


def jump(target):
    return Instruction(Op.JUMP, target)


def jump_if_false(target):
    return Instruction(Op.JUMP_IF_FALSE, target)


def assert_valid(func):
    """``func`` is supported and its IR passes the reference validator."""
    analysis = analyze(func)
    assert analysis.supported, analysis.rejection
    validate(analysis.instructions)
    return analysis.instructions


def test_if_else_with_duplicated_return():
    def calculate_score(wins, penalties):
        score = wins * 3 - penalties
        if score < 0:
            score = 0
        else:
            score += 5
        return score

    # CPython 3.14 copies the short ``return score`` into both branches,
    # so each branch ends with its own RETURN and there is no join.
    assert_supported(
        calculate_score,
        [
            LOAD_ARG_0,
            const(3),
            MULTIPLY,
            LOAD_ARG_1,
            SUBTRACT,
            store(0),
            load(0),
            const(0),
            Instruction(Op.LESS_THAN),
            jump_if_false(14),
            const(0),
            store(0),
            load(0),
            RETURN,
            load(0),
            const(5),
            ADD,
            store(0),
            load(0),
            RETURN,
        ],
    )


def test_if_else_assigning_one_local():
    def choose(a, b):
        if a >= b:
            result = a - b
        else:
            result = b - a
        return result

    assert_supported(
        choose,
        [
            LOAD_ARG_0,
            LOAD_ARG_1,
            Instruction(Op.GREATER_EQUAL),
            jump_if_false(10),
            LOAD_ARG_0,
            LOAD_ARG_1,
            SUBTRACT,
            store(0),
            load(0),
            RETURN,
            LOAD_ARG_1,
            LOAD_ARG_0,
            SUBTRACT,
            store(0),
            load(0),
            RETURN,
        ],
    )


def test_if_without_else_joins():
    def adjust(a, b):
        result = a
        if a < b:
            result += b
        return result * 2

    # The tail is too long to copy, so the jump goes to a shared join.
    assert_supported(
        adjust,
        [
            LOAD_ARG_0,
            store(0),
            LOAD_ARG_0,
            LOAD_ARG_1,
            Instruction(Op.LESS_THAN),
            jump_if_false(10),
            load(0),
            LOAD_ARG_1,
            ADD,
            store(0),
            load(0),
            const(2),
            MULTIPLY,
            RETURN,
        ],
    )


@pytest.mark.parametrize(
    ("operator", "op"),
    [
        ("<", Op.LESS_THAN),
        ("<=", Op.LESS_EQUAL),
        (">", Op.GREATER_THAN),
        (">=", Op.GREATER_EQUAL),
        ("==", Op.EQUAL),
        ("!=", Op.NOT_EQUAL),
    ],
)
def test_every_comparison(operator, op):
    f = define(
        f"def f(a, b):\n    r = 0\n    if a {operator} b:\n        r = 1\n"
        f"    return r\n"
    )
    assert_supported(
        f,
        [
            const(0),
            store(0),
            LOAD_ARG_0,
            LOAD_ARG_1,
            Instruction(op),
            jump_if_false(8),
            const(1),
            store(0),
            load(0),
            RETURN,
        ],
    )


def test_comparison_operands_are_expressions():
    def f(a, b):
        r = 0
        if 0 < a * 2 - b:
            r = 1
        return r

    assert assert_valid(f)[2:9] == (
        const(0),
        LOAD_ARG_0,
        const(2),
        MULTIPLY,
        LOAD_ARG_1,
        SUBTRACT,
        Instruction(Op.LESS_THAN),
    )


def test_nested_and_sequential_branches():
    def f(a, b, c):
        r = 0
        if a < b:
            if b < c:
                r = 1
            else:
                r = 2
        else:
            r = 3
        if a == c:
            r += 10
        return r

    assert_supported(
        f,
        [
            const(0),
            store(0),
            LOAD_ARG_0,
            LOAD_ARG_1,
            Instruction(Op.LESS_THAN),
            jump_if_false(16),  # else: r = 3
            LOAD_ARG_1,
            LOAD_ARG_2,
            Instruction(Op.LESS_THAN),
            jump_if_false(13),  # inner else: r = 2
            const(1),
            store(0),
            jump(18),
            const(2),
            store(0),
            jump(18),
            const(3),
            store(0),
            LOAD_ARG_0,  # 18: the second if.
            LOAD_ARG_2,
            Instruction(Op.EQUAL),
            jump_if_false(26),
            load(0),
            const(10),
            ADD,
            store(0),
            load(0),  # 26
            RETURN,
        ],
    )


def test_elif_chain():
    def f(a, b):
        if a < b:
            r = 1
        elif a > b:
            r = 2
        else:
            r = 3
        return r

    instructions = assert_valid(f)
    assert [i.op for i in instructions].count(Op.RETURN) == 3
    assert [i.op for i in instructions].count(Op.JUMP_IF_FALSE) == 2


def test_empty_branch_jumps_to_the_next_instruction():
    def f(a, b):
        if a < b:
            pass
        return a

    # ``pass`` is a NOP; both outcomes continue at index 4.
    assert_supported(
        f,
        [
            LOAD_ARG_0,
            LOAD_ARG_1,
            Instruction(Op.LESS_THAN),
            jump_if_false(4),
            LOAD_ARG_0,
            RETURN,
        ],
    )


def test_argument_reassigned_on_one_path_is_copied_on_entry():
    def f(a, b):
        if a < b:
            a = b
        return a

    # After the join ``a`` is the argument on one path and a stored value
    # on the other, so its slot starts out holding the argument.
    assert_supported(
        f,
        [
            LOAD_ARG_0,
            store(0),
            LOAD_ARG_0,
            LOAD_ARG_1,
            Instruction(Op.LESS_THAN),
            jump_if_false(8),
            LOAD_ARG_1,
            store(0),
            load(0),
            RETURN,
        ],
    )


def test_argument_reassigned_on_every_path_needs_no_copy():
    def f(a, b):
        if a < b:
            a = b
        else:
            a = 0
        return a + b

    instructions = assert_valid(f)
    assert instructions[:3] == (LOAD_ARG_0, LOAD_ARG_1, Instruction(Op.LESS_THAN))


def test_swap_through_a_copy_across_a_join():
    def f(a, b):
        c = a
        if a < b:
            a = b
            b = c
        return a - b

    instructions = assert_valid(f)
    # ``a`` (slot 1) and ``b`` (slot 2) are copied in on entry; ``c`` is
    # slot 0, the first variable stored.
    assert instructions[:4] == (LOAD_ARG_0, store(1), LOAD_ARG_1, store(2))
    for a, b in [(1, 2), (2, 1), (3, 3)]:
        assert execute(instructions, (a, b)) == f(a, b)


def test_straight_line_functions_have_no_jumps_or_copies():
    def f(a, b):
        a = a + 1
        a += b
        return a * b

    instructions = assert_valid(f)
    assert instructions[0] == LOAD_ARG_0
    assert not any(i.op is Op.JUMP or i.op is Op.JUMP_IF_FALSE for i in instructions)


def test_rejects_possibly_unbound_read_after_a_join():
    def possibly_unbound(a, b):
        if a < b:
            result = a
        return result

    rejection = assert_rejected(
        possibly_unbound,
        RejectionReason.UNSUPPORTED_VARIABLE,
        "LOAD_FAST_CHECK",
        18,
    )
    assert "'result' may be read before it is assigned" in rejection.message
    # The original function still behaves as CPython defines it.
    assert possibly_unbound(1, 2) == 1
    with pytest.raises(UnboundLocalError):
        possibly_unbound(2, 1)


def test_rejects_read_assigned_only_in_the_other_branch():
    def f(a, b):
        if a < b:
            x = 1
        else:
            y = 2  # noqa: F841
            return x  # noqa: F821 -- unbound on this path.
        return x

    assert_rejected(f, RejectionReason.UNSUPPORTED_VARIABLE, "LOAD_FAST_CHECK", 26)


@pytest.mark.parametrize(
    "body",
    [
        "x = a < b\n    return x",
        "return a < b",
        "return (a < b) + 1",
    ],
)
def test_rejects_comparison_results_as_values(body):
    rejection = assert_rejected(
        define(f"def f(a, b):\n    {body}\n"),
        RejectionReason.UNSUPPORTED_CONDITION,
        "COMPARE_OP",
        4,
    )
    assert "only supported as an if or while condition" in rejection.message


def test_rejects_or_and_not():
    # Both compile to POP_JUMP_IF_TRUE.
    for condition in ("a < b or b < 0", "not a < b"):
        f = define(
            f"def f(a, b):\n    r = 0\n    if {condition}:\n        r = 1\n"
            f"    return r\n"
        )
        assert_rejected(f, RejectionReason.CONTROL_FLOW, "POP_JUMP_IF_TRUE", 12)


def test_and_of_comparisons_is_the_same_bytecode_as_nested_ifs():
    nested = define(
        "def f(a, b, c):\n    x = 0\n    if a < b:\n        if b < c:\n"
        "            x = 1\n    return x\n"
    )
    conjunction = define(
        "def f(a, b, c):\n    x = 0\n    if a < b and b < c:\n        x = 1\n"
        "    return x\n"
    )
    assert analyze(conjunction).instructions == assert_valid(nested)


def test_conditional_expressions_compiled_like_statements():
    # CPython emits the same bytecode shape for these as for if statements
    # with a return or an assignment in each branch, so they are accepted.
    ternary = define("def f(a, b):\n    return a if a < b else b\n")
    statement = define("def f(a, b):\n    if a < b:\n        return a\n    return b\n")
    assert assert_valid(ternary) == assert_valid(statement)
    assigned = define("def f(a, b):\n    x = a if a < b else b\n    return x\n")
    assert_valid(assigned)


def test_rejects_conditional_expressions_inside_expressions():
    # A value on the stack at the jump or at the join.
    after = define("def f(a, b):\n    x = (a if a < b else b) + 1\n    return x\n")
    before = define("def f(a, b):\n    x = 1 + (a if a < b else b)\n    return x\n")
    assert_rejected(after, RejectionReason.CONTROL_FLOW, "JUMP_FORWARD", 16)
    assert_rejected(before, RejectionReason.CONTROL_FLOW, "POP_JUMP_IF_FALSE", 10)


def test_rejects_chained_comparisons():
    f = define(
        "def f(a, b, c):\n    x = 0\n    if a < b < c:\n        x = 1\n    return x\n"
    )
    assert_rejected(f, RejectionReason.UNSUPPORTED_OPCODE, "SWAP", 8)


def test_rejects_unsupported_comparison_operands_and_paths():
    cases = [
        # A float or bool constant.
        (
            "if a < 1.5:\n        a = 1\n    return a",
            RejectionReason.UNSUPPORTED_CONSTANT,
        ),
        (
            "if a == True:\n        a = 1\n    return a",
            RejectionReason.UNSUPPORTED_CONSTANT,
        ),
        # A path that returns None.
        (
            "if a < b:\n        return\n    return a",
            RejectionReason.UNSUPPORTED_CONSTANT,
        ),
        ("if a < b:\n        a = 1", RejectionReason.UNSUPPORTED_CONSTANT),
        # A call in a branch.
        (
            "if a < b:\n        a = abs(b)\n    return a",
            RejectionReason.UNSUPPORTED_OPCODE,
        ),
    ]
    for body, reason in cases:
        rejection = analyze(define(f"def f(a, b):\n    {body}\n")).rejection
        assert rejection is not None and rejection.reason is reason, body


def test_branch_ir_matches_python_on_small_inputs():
    sources = [
        "if a < b:\n        a = b\n    return a",
        "r = a\n    if a < b:\n        r += b\n    else:\n        r -= b\n"
        "    return r * 2",
        "if a < 0:\n        a = 0 - a\n    if b < 0:\n        b = 0 - b\n"
        "    return a + b",
        "r = 0\n    if a != b:\n        if a > b:\n            r = a\n"
        "        r += 1\n    return r",
    ]
    for body in sources:
        f = define(f"def f(a, b):\n    {body}\n")
        instructions = assert_valid(f)
        for a in range(-2, 3):
            for b in range(-2, 3):
                assert execute(instructions, (a, b)) == f(a, b), (body, a, b)


# --- The instruction model -----------------------------------------------


def test_instruction_str():
    def add(a, b):
        return a + 2

    listing = [str(i) for i in analyze(add).instructions]
    assert listing == ["LOAD_ARG 0", "LOAD_CONST 2", "ADD", "RETURN"]


def test_results_are_immutable():
    def add(a, b):
        return a + b

    analysis = analyze(add)
    with pytest.raises(AttributeError):
        analysis.instructions = ()
    with pytest.raises(AttributeError):
        analysis.instructions[0].arg = 5
    assert isinstance(analysis.instructions, tuple)

    rejection = analyze(len).rejection
    with pytest.raises(AttributeError):
        rejection.reason = RejectionReason.CLOSURE


def test_instruction_validates_its_argument():
    with pytest.raises(TypeError):
        Instruction(Op.LOAD_ARG)
    with pytest.raises(TypeError):
        Instruction(Op.LOAD_CONST, True)
    with pytest.raises(ValueError):
        Instruction(Op.LOAD_ARG, -1)
    with pytest.raises(TypeError):
        Instruction(Op.ADD, 1)
    for op in (Op.LOAD_LOCAL, Op.STORE_LOCAL):
        with pytest.raises(TypeError):
            Instruction(op)
        with pytest.raises(TypeError):
            Instruction(op, False)
        with pytest.raises(ValueError, match=f"{op.name} index must be non-negative"):
            Instruction(op, -1)
        assert str(Instruction(op, 3)) == f"{op.name} 3"


def test_analysis_has_instructions_xor_rejection():
    rejection = Rejection(RejectionReason.CLOSURE, "closures are not supported")
    with pytest.raises(ValueError):
        Analysis()
    with pytest.raises(ValueError):
        Analysis(instructions=(RETURN,), rejection=rejection)
    assert Analysis(instructions=[RETURN]).instructions == (RETURN,)


def test_rejection_reasons_are_stable_strings():
    assert RejectionReason.UNSUPPORTED_OPCODE == "unsupported_opcode"
    assert RejectionReason.CONTROL_FLOW == "control_flow"
    assert RejectionReason.UNSUPPORTED_CONDITION == "unsupported_condition"


def test_frontend_is_private():
    assert "_frontend" not in speedysnek.__all__
    assert "analyze" not in speedysnek.__all__
    assert not hasattr(speedysnek, "analyze")
