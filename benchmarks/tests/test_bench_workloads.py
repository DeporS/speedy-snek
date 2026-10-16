"""Correctness of the benchmark workloads. No timing is asserted anywhere."""

import re

import pytest

import workloads
from speedysnek import _native, snekit
from speedysnek._compiler import CompiledI64, compile_i64
from speedysnek._decorator import _Status
from speedysnek._frontend import Op, analyze
from speedysnek._interpreter import execute
from workloads import (
    ARGS,
    LAYERS,
    OP_COUNTS,
    BenchmarkSetupError,
    arithmetic_count,
    call_path,
    expression,
    fallback_cases,
    fits_i64,
    intermediate_values,
    lifecycle,
    make_workload,
    prepare_layers,
    status_of,
)


@pytest.fixture(scope="module", params=OP_COUNTS, ids=lambda n: f"ops_{n:03d}")
def layers(request):
    return prepare_layers(request.param)


# --- Workload shape ------------------------------------------------------


def test_workload_sizes():
    assert OP_COUNTS == (1, 5, 20, 50, 100, 250)


@pytest.mark.parametrize("ops", OP_COUNTS)
def test_workload_has_exactly_the_requested_operations(ops):
    analysis = analyze(make_workload(ops))
    assert analysis.supported, analysis.rejection
    assert arithmetic_count(analysis.instructions) == ops


@pytest.mark.parametrize("ops", OP_COUNTS)
def test_nothing_is_constant_folded(ops):
    source = expression(ops)
    # Every operator written in the source survives into the IR...
    assert len(re.findall(r" [-+*] ", source)) == ops
    # ...and no operation works on constants alone.
    instructions = analyze(make_workload(ops)).instructions
    stack = []
    for instruction in instructions:
        if instruction.op in (Op.LOAD_ARG, Op.LOAD_CONST):
            stack.append(instruction.op is Op.LOAD_ARG)
        elif instruction.op is not Op.RETURN:
            right, left = stack.pop(), stack.pop()
            assert left or right
            stack.append(True)


@pytest.mark.parametrize("ops", OP_COUNTS)
def test_result_depends_on_runtime_input(ops):
    f = make_workload(ops)
    assert f(*ARGS) != f(*(value + 1 for value in ARGS))


@pytest.mark.parametrize("ops", OP_COUNTS)
def test_benchmark_inputs_never_overflow_i64(ops):
    instructions = analyze(make_workload(ops)).instructions
    values = intermediate_values(instructions, ARGS)
    assert len(values) == len(instructions) - 1
    assert all(fits_i64(value) for value in (*ARGS, *values))
    # The native code checks every operation for overflow; it must agree.
    assert compile_i64(instructions, len(ARGS))(*ARGS) == make_workload(ops)(*ARGS)


def test_intermediate_values_follow_the_stack():
    instructions = analyze(make_workload(5)).instructions
    a, b, c, d = ARGS
    # a + b * c + c * d - d
    assert intermediate_values(instructions, ARGS) == [
        a,
        b,
        c,
        b * c,
        a + b * c,
        c,
        d,
        c * d,
        a + b * c + c * d,
        d,
        a + b * c + c * d - d,
    ]


def test_expression_rejects_bad_sizes():
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            expression(bad)


def test_workloads_are_fresh_functions():
    assert make_workload(5) is not make_workload(5)
    assert make_workload(5).__code__ is not make_workload(5).__code__


# --- Runtime layers ------------------------------------------------------


def test_every_layer_returns_the_same_result(layers):
    expected = layers.python(*ARGS)
    assert layers.expected == expected
    assert execute(analyze(layers.python).instructions, ARGS) == expected
    for layer in LAYERS:
        assert layers.run(layer) == expected
    assert layers.native_call(ARGS) == (_native.STATUS_OK, expected)


def test_warmed_snekit_really_runs_native_code(layers):
    state = layers.snekit._jit_state
    assert status_of(layers.snekit) is _Status.COMPILED
    assert state.compile_attempts == 1
    assert call_path(layers.snekit, *ARGS) == "native"
    # The direct layers wrap the very code the decorator runs.
    assert layers.compiled is state.specialization.native
    assert isinstance(layers.compiled, CompiledI64)


def test_layer_statements(layers):
    statements = {layer: layers.statement(layer)[0] for layer in LAYERS}
    # The Python, decorator and CompiledI64 layers share one call site...
    assert statements["python"] == statements["snekit"] == statements["compiled_i64"]
    # ...and the raw call reuses a prebuilt tuple.
    assert statements["native_call"] == "call(args)"
    assert layers.statement("native_call")[1]["args"] is layers.args
    with pytest.raises(ValueError):
        layers.statement("nope")


def test_timing_does_not_change_state(layers):
    before = layers.snekit._jit_state.specialization
    for _ in range(100):
        layers.run("snekit")
    assert layers.snekit._jit_state.specialization is before


def test_prepare_layers_refuses_overflowing_inputs():
    with pytest.raises(BenchmarkSetupError, match="overflows"):
        prepare_layers(5, args=(2**40,) * 4)


def test_call_path_tells_native_from_python():
    @snekit
    def add(a, b):
        return a + b

    assert call_path(add, 1, 2) == "python"
    workloads.warm_up(add, 1, 2)
    assert call_path(add, 1, 2) == "native"
    assert call_path(add, 1.0, 2) == "python"


# --- Fallback cases ------------------------------------------------------


@pytest.fixture(scope="module")
def cases():
    return {case.name: case for case in fallback_cases()}


def test_fallback_cases_are_all_present(cases):
    assert tuple(cases) == workloads.FALLBACK_CASES


@pytest.mark.parametrize("name", workloads.FALLBACK_CASES)
def test_fallback_returns_the_same_result(cases, name):
    case = cases[name]
    expected = case.run("python")
    actual = case.run("snekit")
    assert actual == expected
    assert type(actual) is type(expected)


@pytest.mark.parametrize("name", workloads.FALLBACK_CASES)
def test_fallback_really_runs_python(cases, name):
    case = cases[name]
    assert call_path(case.wrapper, *case.args, **case.kwargs) == "python"
    assert status_of(case.wrapper) is case.status


@pytest.mark.parametrize("name", workloads.FALLBACK_CASES)
def test_fallback_sides_use_the_same_statement(cases, name):
    case = cases[name]
    python, python_globals = case.statement("python")
    wrapped, wrapped_globals = case.statement("snekit")
    assert python == wrapped
    assert python_globals["f"] is case.python
    assert wrapped_globals["f"] is case.wrapper
    # The baseline is an undecorated copy, never the wrapped function itself.
    assert case.python is not case.wrapper.__wrapped__
    assert case.python.__code__.co_code == case.wrapper.__wrapped__.__code__.co_code


def test_fallback_case_statements(cases):
    assert cases["keyword_args"].statement("python")[0] == "f(a=k_a, b=k_b)"
    assert cases["default_args"].statement("python")[0] == "f(x0)"
    assert cases["float_args"].statement("python")[0] == "f(x0, x1)"


def test_cold_case_stays_cold(cases):
    case = cases["cold"]
    for _ in range(1000):
        case.run("snekit")
    assert status_of(case.wrapper) is _Status.COLD
    assert case.wrapper._jit_state.compile_attempts == 0


def test_disabled_case_was_rejected(cases):
    state = cases["disabled"].wrapper._jit_state
    assert state.disabled_reason.value == "rejected"


@pytest.mark.parametrize(
    "args",
    [(1, None), ("a", 2), (None, None)],
    ids=["int-None", "str-int", "None-None"],
)
def test_fallback_raises_the_same_exception(cases, args):
    case = cases["float_args"]
    with pytest.raises(Exception) as python_error:
        case.python(*args)
    with pytest.raises(Exception) as wrapper_error:
        case.wrapper(*args)
    assert type(wrapper_error.value) is type(python_error.value)
    assert str(wrapper_error.value) == str(python_error.value)
    assert wrapper_error.value.__context__ is None


# --- Compilation ---------------------------------------------------------


def test_lifecycle_compiles_afresh_every_time():
    func = make_workload(20)
    first = lifecycle(func, ARGS)
    second = lifecycle(func, ARGS)
    assert first is not second
    for wrapper in (first, second):
        assert status_of(wrapper) is _Status.COMPILED
        assert wrapper._jit_state.compile_attempts == 1
        assert wrapper.call_count == workloads.LIFECYCLE_CALLS
    assert first._jit_state.compiled is not second._jit_state.compiled


def test_unsupported_workload_is_rejected_and_disabled():
    func = workloads.unsupported_workload()
    assert analyze(func).rejection is not None
    wrapper = lifecycle(func, ARGS)
    assert status_of(wrapper) is _Status.DISABLED
    assert wrapper(*ARGS) == func(*ARGS)
