"""Every way CPython can call a ``@snekit`` wrapper behaves the same.

CPython calls an object through one of two C protocols: ``tp_call``, with
an argument tuple and a keyword dict, or vectorcall, with an array of
arguments and a tuple of keyword names. ``NativeWrapper`` implements only
``tp_call``, so for a call site such as ``add(3, 4)`` CPython builds the
tuple. ``benchmarks/experiments/vectorcall.md`` measured what a vectorcall
entry would save and why it is not adopted; prototypes of one passed the
whole suite.

These tests pin that the observable behaviour of a call does not depend
on how it reaches the wrapper, call by call: the result or exception, the
original function's frames, native attempts, ``call_count`` and every
state transition (warmup, compilation, guard misses, parking and
re-probing, ``__code__`` replacement, disabling). With today's single
entry point they hold by construction; any second entry point, such as a
vectorcall one, must keep them. ``type(wrapper).__call__`` always reaches
``tp_call``, so it is the reference for the others:

- a call site (``CALL``/``CALL_KW``), which uses vectorcall when the callee
  has it, and otherwise builds the tuple;
- ``f(*args, **kwargs)``, which passes a tuple and a dict;
- ``functools.partial`` and bound methods, which forward a vectorcall
  with room for one more argument (``PY_VECTORCALL_ARGUMENTS_OFFSET``).
"""

import functools
import sys
import types

import pytest

from speedysnek import _native, snekit
from speedysnek._decorator import _WARMUP_CALLS, _Status

# ``Py_TPFLAGS_HAVE_VECTORCALL``.
HAVE_VECTORCALL = 1 << 11

_SITES = {}


def call_site(nargs, names):
    """A function whose body is one call site: ``f(a0, ..., name=k0, ...)``."""
    key = (nargs, names)
    if key not in _SITES:
        params = [f"a{i}" for i in range(nargs)]
        keywords = [f"k{i}" for i in range(len(names))]
        arguments = params + [f"{n}=k{i}" for i, n in enumerate(names)]
        source = (
            f"def site(f, {', '.join(['_', *params, *keywords])}):\n"
            f"    return f({', '.join(arguments)})\n"
        )
        namespace = {}
        exec(source, namespace)
        _SITES[key] = namespace["site"]
    return _SITES[key]


def bind(wrapper, prefix):
    """The wrapper, or a method that binds ``prefix[0]`` to it."""
    return types.MethodType(wrapper, prefix[0]) if prefix else wrapper


def via_call_site(wrapper, prefix, args, kwargs):
    site = call_site(len(args), tuple(kwargs))
    return site(bind(wrapper, prefix), None, *args, *kwargs.values())


def via_star_call(wrapper, prefix, args, kwargs):
    return bind(wrapper, prefix)(*args, **kwargs)


def via_partial(wrapper, prefix, args, kwargs):
    return functools.partial(wrapper, *prefix)(*args, **kwargs)


def via_tp_call(wrapper, prefix, args, kwargs):
    return type(wrapper).__call__(wrapper, *prefix, *args, **kwargs)


PROTOCOLS = {
    "call site": via_call_site,
    "star call": via_star_call,
    "partial": via_partial,
}


def frames_of(code):
    """A profiler that counts the frames of ``code``, and its count."""
    count = [0]

    def profiler(frame, event, arg):
        if event == "call" and frame.f_code is code:
            count[0] += 1

    return profiler, count


def native_counts(wrapper):
    compiled = wrapper._jit_state.compiled
    return (0, 0) if compiled is None else compiled.function.try_counts


def outcome(protocol, wrapper, prefix, args, kwargs):
    """Make one call through ``protocol`` and record what it did."""
    before = native_counts(wrapper)
    profiler, frames = frames_of(wrapper.__wrapped__.__code__)
    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        result = protocol(wrapper, prefix, args, kwargs)
    except Exception as error:
        observed = ("raise", type(error), str(error))
    else:
        observed = ("ok", type(result), result)
    finally:
        sys.setprofile(previous)
    after = native_counts(wrapper)
    state = wrapper._jit_state
    hot = state._hot
    return (
        observed,
        frames[0],
        after[0] - before[0],
        after[1] - before[1],
        wrapper.call_count,
        state.status,
        state.disabled_reason,
        state.parked_reason,
        state.compile_attempts,
        state.warmup_calls,
        hot.live,
        hot.parked,
        hot.misses,
        hot.parked_calls,
    )


def copy_function(func):
    return types.FunctionType(
        func.__code__, func.__globals__, func.__name__, func.__defaults__
    )


def twins(func):
    """Two wrappers of separate copies of ``func``."""
    return snekit(copy_function(func)), snekit(copy_function(func))


def same_calls(protocol, reference, other, calls, prefix=()):
    """Drive both wrappers through the same calls, and compare every one.

    ``reference`` is called through ``tp_call``, ``other`` through
    ``protocol``. Returns the reference's outcomes.
    """
    outcomes = []
    for index, (args, kwargs) in enumerate(calls):
        expected = outcome(via_tp_call, reference, prefix, args, kwargs)
        actual = outcome(protocol, other, prefix, args, kwargs)
        assert actual == expected, f"call {index}: {args!r}, {kwargs!r}"
        outcomes.append(expected)
    return outcomes


WARM = [((1, 2), {})] * (_WARMUP_CALLS + 1)

# Every kind of call, in an order that keeps the wrapper compiled.
MIXED = [
    ((5, 7), {}),
    ((5,), {}),
    ((5,), {"b": 7}),
    ((), {"a": 5, "b": 7}),
    ((1.5, 2), {}),
    ((True, 2), {}),
    ((2**40, 2**40), {}),
    ((2**63, 1), {}),
    ((-(2**63), 2), {}),
    (("x", 2), {}),
    ((-4, -9), {}),
    # Invalid calls: the original's own ``TypeError``.
    ((), {}),
    ((1, 2, 3), {}),
    ((1,), {"c": 2}),
    ((1,), {"a": 2}),
    # Every positional argument and a keyword: never a native call.
    ((1, 2), {"c": 3}),
    ((1, 2), {"b": 3}),
    ((-4, -9), {}),
]


# ``MIXED`` for a method whose ``self`` fills the first parameter.
BOUND = [
    ((7,), {}),
    ((), {}),
    ((1.5,), {}),
    ((2**63,), {}),
    ((), {"b": 7}),
    ((7, 8), {}),
    ((), {"a": 1}),
    ((2,), {"c": 3}),
    ((2,), {"b": 3}),
    ((-9,), {}),
]


def pair(a, b=3):
    return a * b - (a + b)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_compiled_calls_match_call_by_call(protocol):
    reference, other = twins(pair)
    outcomes = same_calls(PROTOCOLS[protocol], reference, other, WARM + MIXED)
    assert reference._jit_state.status is _Status.COMPILED
    assert reference._jit_state._hot.live
    # A warmed native hit enters no frame of the original function.
    native = outcomes[len(WARM)]
    assert native[0] == ("ok", int, 5 * 7 - 12)
    assert native[1:4] == (0, 1, 0)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_code_replacement_matches_call_by_call(protocol):
    reference, other = twins(pair)
    same_calls(PROTOCOLS[protocol], reference, other, WARM)
    for wrapper in (reference, other):
        wrapper.__wrapped__.__code__ = (lambda a, b=3: a - b).__code__
    outcomes = same_calls(PROTOCOLS[protocol], reference, other, MIXED)
    # The first call after the replacement already runs the new code.
    assert outcomes[0][0] == ("ok", int, -2)
    assert not reference._jit_state._hot.live
    # Another signature: the new code's own ``TypeError``.
    for wrapper in (reference, other):
        wrapper.__wrapped__.__code__ = (lambda a, b, c: a + b + c).__code__
    same_calls(PROTOCOLS[protocol], reference, other, MIXED + [((1, 2, 3), {})])


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_parking_and_reprobing_match_call_by_call(protocol):
    reference, other = twins(pair)
    floats = [((1.5, 2.5), {})] * (_native.MISS_LIMIT + 2)
    ints = [((3, 4), {})] * (_native.REPROBE_INTERVAL + 2)
    calls = WARM + floats + MIXED + ints + [((5, 7), {})]
    outcomes = same_calls(PROTOCOLS[protocol], reference, other, calls)
    statuses = [entry[5] for entry in outcomes]
    assert _Status.PARKED in statuses
    # Re-probed and unparked: the last call is native again.
    assert statuses[-1] is _Status.COMPILED
    assert outcomes[-1][1:4] == (0, 1, 0)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_keywords_and_defaults_park_the_same(protocol):
    reference, other = twins(pair)
    keywords = [((), {"a": 3, "b": 4})] * (_native.MISS_LIMIT + 2)
    defaults = [((3,), {})] * (_native.REPROBE_INTERVAL + 2)
    calls = WARM + keywords + defaults + MIXED
    outcomes = same_calls(PROTOCOLS[protocol], reference, other, calls)
    assert outcomes[len(WARM) + _native.MISS_LIMIT][5] is _Status.PARKED
    # Neither ever reaches native code.
    assert all(entry[2] == 0 for entry in outcomes[len(WARM) : -len(MIXED)])


def rejected(a, b):
    return (a * b + a) // b


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_disabled_wrappers_match_call_by_call(protocol):
    reference, other = twins(rejected)
    calls = WARM + MIXED + [((1, 0), {}), ((7, 2), {})]
    outcomes = same_calls(PROTOCOLS[protocol], reference, other, calls)
    assert reference._jit_state.status is _Status.DISABLED
    assert outcomes[-2][0] == ("raise", ZeroDivisionError, "division by zero")
    assert outcomes[-1][0] == ("ok", int, 10)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_methods_bound_to_an_int_run_natively_the_same(protocol):
    # ``types.MethodType(wrapper, 5)(3)`` is ``wrapper(5, 3)``: a bound
    # method forwards its arguments with ``self`` in front.
    reference, other = twins(pair)
    calls = [((2,), {})] * (_WARMUP_CALLS + 1) + BOUND
    outcomes = same_calls(PROTOCOLS[protocol], reference, other, calls, prefix=(5,))
    assert reference._jit_state._hot.live
    assert outcomes[-1][0] == ("ok", int, 5 * -9 - (5 - 9))
    assert outcomes[-1][1:4] == (0, 1, 0)


class Box:
    pass


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_methods_bound_to_objects_match_call_by_call(protocol):
    def method(self, a, b=3):
        return (type(self).__name__, a, b)

    reference, other = twins(method)
    calls = [((1, 2), {})] * (_WARMUP_CALLS + 1) + BOUND
    same_calls(PROTOCOLS[protocol], reference, other, calls, prefix=(Box(),))
    assert reference._jit_state.status is _Status.PARKED


@pytest.mark.parametrize("protocol", [*PROTOCOLS, "tp_call"])
def test_methods_through_the_descriptor(protocol):
    class Point:
        @snekit
        def shifted(self, dx):
            return dx + 1

    point = Point()
    call = {**PROTOCOLS, "tp_call": via_tp_call}[protocol]
    assert call(point.shifted, (), (2,), {}) == 3
    assert call(Point.shifted, (point,), (2,), {}) == 3
    with pytest.raises(TypeError, match="missing 1 required positional"):
        call(point.shifted, (), (), {})
    assert Point.__dict__["shifted"].call_count == 3


@pytest.mark.parametrize("protocol", [*PROTOCOLS, "tp_call"])
def test_deep_recursion_raises_recursion_error(protocol):
    call = {**PROTOCOLS, "tp_call": via_tp_call}[protocol]

    @snekit
    def down(n):
        return 0 if n == 0 else call(down, (), (n - 1,), {})

    limit = sys.getrecursionlimit()
    depth, frame = 0, sys._getframe()
    while frame is not None:
        depth, frame = depth + 1, frame.f_back
    sys.setrecursionlimit(depth + 200)
    try:
        with pytest.raises(RecursionError):
            down(10_000)
    finally:
        sys.setrecursionlimit(limit)
    assert down(10) == 0


def test_the_wrapper_has_no_vectorcall_entry():
    # Pinned so that adding one is a deliberate decision: see
    # ``benchmarks/experiments/vectorcall.md``.
    assert not _native.NativeWrapper.__flags__ & HAVE_VECTORCALL
