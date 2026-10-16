"""Tests for the private CPython 3.14 bytecode frontend.

Expected opcodes and offsets are exact: they are what CPython 3.14 emits,
which is the only version the frontend supports.
"""

import dis
from types import SimpleNamespace

import pytest

import speedysnek
from speedysnek import _frontend, jit
from speedysnek._frontend import (
    Analysis,
    Instruction,
    Op,
    Rejection,
    RejectionReason,
    analyze,
)

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
    @jit
    def add(a, b):
        return a + b

    # The wrapper is a closure over ``func``, so it is rejected as a whole.
    assert_rejected(add, RejectionReason.CLOSURE)
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


def test_rejects_if():
    def f(a):
        if a:
            return 1
        return 2

    def ternary(a, b):
        return a if b else 1

    assert_rejected(f, RejectionReason.CONTROL_FLOW, "POP_JUMP_IF_FALSE", 12)
    assert_rejected(ternary, RejectionReason.CONTROL_FLOW, "POP_JUMP_IF_FALSE", 12)


def test_rejects_loops():
    def while_loop(a):
        while a:
            a = a - 1
        return a

    def for_loop(a):
        for _ in a:
            pass
        return 0

    assert_rejected(while_loop, RejectionReason.CONTROL_FLOW, "POP_JUMP_IF_FALSE", 12)
    assert_rejected(for_loop, RejectionReason.CONTROL_FLOW, "FOR_ITER", 6)


def test_rejects_local_variables():
    def store(a):
        b = a + 1
        return b

    assert_rejected(store, RejectionReason.UNSUPPORTED_OPCODE, "STORE_FAST", 18)


def test_rejects_reading_a_non_argument_local():
    # Nothing ever stores ``y``, so CPython emits a lone LOAD_FAST_CHECK.
    f = define("def f():\n    return y\n    y = 1\n")
    assert_rejected(f, RejectionReason.UNSUPPORTED_VARIABLE, "LOAD_FAST_CHECK", 2)


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

    # The protected body is straight-line; the handler after it is not.
    assert_rejected(f, RejectionReason.UNSUPPORTED_OPCODE, "PUSH_EXC_INFO", 20)


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


def test_frontend_is_private():
    assert "_frontend" not in speedysnek.__all__
    assert "analyze" not in speedysnek.__all__
    assert not hasattr(speedysnek, "analyze")
