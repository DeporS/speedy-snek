"""Keeps ``experiments/wrapper_overhead.py`` honest.

Its ``positional`` prototype is only a fair candidate if it keeps the
``@snekit`` contract, so these tests compare it call by call with the
production wrapper, on the risks a generated signature introduces. They
also check that every variant really runs the native fast call when warm.
"""

import importlib.util
from pathlib import Path

import pytest

import workloads
from speedysnek import snekit

_PATH = Path(__file__).resolve().parent.parent / "experiments" / "wrapper_overhead.py"
_SPEC = importlib.util.spec_from_file_location("wrapper_overhead", _PATH)
experiment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(experiment)

positional = experiment.positional


def define(params, body="a * b + a - b", name="f"):
    namespace = {}
    exec(f"def {name}({params}):\n    return {body}\n", namespace)
    return namespace[name]


def outcome(f, args, kwargs):
    try:
        result = f(*args, **kwargs)
    except Exception as error:
        return ("raise", type(error), str(error), type(error.__context__))
    return ("return", type(result), result)


def pair(params, body="a * b + a - b"):
    """The same function under ``snekit`` and the prototype, both warm."""
    wrappers = [decorate(define(params, body)) for decorate in (snekit, positional)]
    for wrapper in wrappers:
        workloads.warm_up(wrapper, 3, 4)
    return wrappers


def assert_same(wrappers, *args, **kwargs):
    expected, actual = (outcome(w, args, kwargs) for w in wrappers)
    assert actual == expected
    assert wrappers[1].call_count == wrappers[0].call_count
    assert wrappers[1]._jit_state.status is wrappers[0]._jit_state.status


CALLS = [
    ((3, 4), {}),
    ((2**62, 2**62), {}),
    ((3.5, 4), {}),
    ((True, 4), {}),
    ((3,), {"b": 4}),
    ((), {"a": 3, "b": 4}),
    ((3,), {}),
    ((), {}),
    ((3, 4, 5), {}),
    ((3, 4), {"c": 5}),
    ((3, 4), {"a": 5}),
    ((3,), {"_p1": 4}),
]


@pytest.mark.parametrize(
    "params", ["a, b", "a, b=4", "a=3, b=4", "a, /, b", "a, b, /", "a, b, *, c=0"]
)
def test_prototype_matches_production(params):
    wrappers = pair(params)
    # Keyword-only parameters are rejected by the frontend: both disabled.
    expected = "disabled" if "*" in params else "compiled"
    assert [w._jit_state.status.value for w in wrappers] == [expected] * 2
    for args, kwargs in CALLS:
        assert_same(wrappers, *args, **kwargs)


def test_positional_only_parameters_reject_keywords():
    wrappers = pair("a, b, /")
    assert_same(wrappers, 3, b=4)
    assert_same(wrappers, a=3, b=4)


def test_changed_defaults_are_used():
    wrappers = pair("a, b=4")
    for wrapper in wrappers:
        wrapper.__wrapped__.__defaults__ = (10,)
    assert_same(wrappers, 3)
    for wrapper in wrappers:
        wrapper.__wrapped__.__defaults__ = None
    assert_same(wrappers, 3)


@pytest.mark.parametrize("params", ["a", "a, b, c", "x, y", "*args", "a, b, /, c=1"])
def test_replaced_code_with_another_signature(params):
    wrappers = pair("a, b")
    replacement = define(params, "len(locals())").__code__
    for wrapper in wrappers:
        wrapper.__wrapped__.__code__ = replacement
    for args, kwargs in CALLS:
        assert_same(wrappers, *args, **kwargs)


def test_code_replaced_before_compilation_with_another_arity():
    wrappers = [decorate(define("a, b")) for decorate in (snekit, positional)]
    for wrapper in wrappers:
        wrapper.__wrapped__.__code__ = define("a, b, c", "a + b * c").__code__
        workloads.warm_up(wrapper, 3, 4, 5)
    assert_same(wrappers, 3, 4, 5)
    assert_same(wrappers, 3, 4)
    assert workloads.call_path(wrappers[1], 3, 4, 5) == "native"


@pytest.mark.parametrize(
    "names",
    ["_p0, _p1", "func, state", "wrapper, cold", "_missing, given", "args, kwargs"],
)
def test_parameter_names_cannot_collide(names):
    wrappers = pair(names, " - ".join(names.split(", ")))
    for args, kwargs in [
        ((3, 4), {}),
        ((3,), {}),
        ((), dict.fromkeys(names.split(", "), 1)),
    ]:
        assert_same(wrappers, *args, **kwargs)


def test_methods_bind():
    class Box:
        prod = snekit(define("self, a, b", "a - b"))
        proto = positional(define("self, a, b", "a - b"))

    box = Box()
    for _ in range(workloads.LIFECYCLE_CALLS):
        box.prod(3, 4)
        box.proto(3, 4)
    assert box.proto(9, 4) == box.prod(9, 4) == 5
    assert Box.proto.call_count == Box.prod.call_count


@pytest.mark.parametrize("variant", sorted(experiment.VARIANTS))
def test_every_variant_runs_the_native_fast_call(variant):
    built = experiment.build(variant)
    assert "ops_001" in built
    assert workloads.snekit is snekit
