"""``@snekit`` returns a native callable object, not a Python function.

These tests cover what such an object could get wrong that a function gets
for free: binding as a method, metadata and introspection, lazy
annotations, pickling, ``call_count``, tracebacks and garbage collection.
The last section pins the documented differences from a function (see
``benchmarks/experiments/native_wrapper.md``), so that changing one is a
deliberate decision.
"""

import abc
import contextlib
import copy
import functools
import gc
import inspect
import pickle
import random
import sys
import threading
import time
import traceback
import types
import typing
import weakref

import pytest

from speedysnek import _compiler, _native, snekit
from speedysnek._compiler import CompiledI64, compile_i64
from speedysnek._decorator import (
    _WARMUP_CALLS,
    _DisabledReason,
    _JitState,
    _Specialization,
    _Status,
)
from speedysnek._frontend import analyze


@snekit
def module_level_add(a, b):
    return a + b


def warmed(f, *args):
    for _ in range(_WARMUP_CALLS + 1):
        f(*args)
    assert f._jit_state.status is _Status.COMPILED
    return f


def native_calls(f):
    compiled = f._jit_state.compiled
    return 0 if compiled is None else compiled.function.try_calls


def make_add():
    @snekit
    def add(a, b):
        return a + b

    return add


# --- Methods and descriptors ----------------------------------------------


class Point:
    def __init__(self, x):
        self.x = x

    @snekit
    def shifted(self, dx):
        return dx + 1

    @staticmethod
    @snekit
    def static(a, b):
        return a * b

    @classmethod
    @snekit
    def named(cls, a):
        return cls.__name__, a

    @property
    @snekit
    def prop(self):
        return "prop"

    @functools.cached_property
    @snekit
    def cached(self):
        return [self.x]


def test_instance_access_binds_like_a_function():
    point = Point(1)
    wrapper = Point.__dict__["shifted"]
    bound = point.shifted
    assert type(bound) is types.MethodType
    assert bound.__self__ is point
    assert bound.__func__ is wrapper
    assert bound(2) == 3
    assert str(inspect.signature(bound)) == "(dx)"


def test_class_access_returns_the_wrapper_itself():
    wrapper = Point.__dict__["shifted"]
    assert Point.shifted is wrapper
    assert wrapper.__get__(None, Point) is wrapper
    # Like a function's ``__get__``.
    with pytest.raises(TypeError, match=r"__get__\(None, None\) is invalid"):
        wrapper.__get__(None)
    assert Point.shifted(Point(1), 2) == 3


def test_methods_count_calls_and_compile():
    shifted = Point.__dict__["shifted"]
    before = shifted.call_count
    points = [Point(i) for i in range(3)]
    for _ in range(_WARMUP_CALLS + 2):
        for point in points:
            assert point.shifted(5) == 6
    assert shifted.call_count == before + 3 * (_WARMUP_CALLS + 2)
    assert shifted._jit_state.status is _Status.COMPILED
    # Bound calls pass ``self``, which is not an int: a guard miss.
    calls = native_calls(shifted)
    assert points[0].shifted(5) == 6
    assert native_calls(shifted) == calls + 1
    # Through the class, with an int in ``self``'s place, it runs natively.
    assert Point.shifted(0, 41) == 42
    assert native_calls(shifted) == calls + 2


def test_decorator_combinations():
    point = Point(7)
    assert point.static(3, 4) == Point.static(3, 4) == 12
    assert point.named(1) == Point.named(1) == ("Point", 1)
    assert point.prop == "prop"
    assert point.cached is point.cached
    assert point.cached == [7]


def test_snekit_over_staticmethod_binds_like_the_function_wrapper_did():
    # The wrapper binds as a method, so ``self`` is passed to the
    # staticmethod, exactly as with a function wrapper.
    class C:
        @snekit
        @staticmethod
        def s(x):
            return x

    assert C.s(1) == 1
    with pytest.raises(TypeError, match="positional argument"):
        C().s(1)


def test_abstractmethod_over_snekit():
    class Base(abc.ABC):
        @abc.abstractmethod
        @snekit
        def run(self): ...

    with pytest.raises(TypeError, match="abstract"):
        Base()


# --- Metadata and introspection ------------------------------------------


def test_metadata_and_introspection():
    def original[T](a: int, b: T = 2) -> int:
        """Doc."""
        return a + b

    wrapper = snekit(original)
    assert wrapper.__name__ == "original"
    assert wrapper.__qualname__ == original.__qualname__
    assert wrapper.__module__ == __name__
    assert wrapper.__doc__ == "Doc."
    assert wrapper.__type_params__ == original.__type_params__
    assert wrapper.__wrapped__ is original
    assert inspect.signature(wrapper) == inspect.signature(original)
    assert inspect.getsource(wrapper) == inspect.getsource(original)
    assert inspect.getdoc(wrapper) == "Doc."
    assert inspect.unwrap(wrapper) is original
    assert typing.get_type_hints(wrapper) == typing.get_type_hints(original)


def test_functools_wraps_over_snekit():
    wrapper = make_add()

    @functools.wraps(wrapper)
    def outer(*args):
        return wrapper(*args)

    assert outer.__name__ == "add"
    assert outer.__wrapped__ is wrapper
    assert str(inspect.signature(outer)) == "(a, b)"
    assert functools.lru_cache(wrapper)(2, 3) == 5
    assert functools.partial(wrapper, 2)(3) == 5


def test_annotations_are_lazy_like_a_functions():
    def f(a: UndefinedName) -> int:  # noqa: F821
        return a

    wrapper = snekit(f)  # Does not evaluate the annotations.
    with pytest.raises(NameError):
        _ = f.__annotations__
    with pytest.raises(NameError):
        _ = wrapper.__annotations__
    assert callable(wrapper.__annotate__)


def _annotation_steps(target):
    """The same sequence of annotation operations, recording each outcome."""
    steps = [
        lambda: target.__annotations__,
        lambda: target.__annotations__ is target.__annotations__,
        lambda: target.__annotations__.setdefault("added", 1),
        lambda: target.__annotations__,
        lambda: setattr(target, "__annotations__", {"z": 1}),
        lambda: (target.__annotations__, target.__annotate__),
        lambda: setattr(target, "__annotate__", lambda format: {"q": 2}),
        lambda: target.__annotations__,
        lambda: setattr(target, "__annotate__", 5),
        lambda: delattr(target, "__annotate__"),
        lambda: setattr(target, "__annotate__", None),
        lambda: target.__annotations__,
        lambda: setattr(target, "__annotations__", 5),
        lambda: delattr(target, "__annotations__"),
        lambda: (target.__annotations__, target.__annotate__),
        lambda: setattr(target, "__annotate__", lambda format: [1]),
        lambda: target.__annotations__,
        lambda: setattr(target, "__annotations__", None),
        lambda: (target.__annotations__, target.__annotate__),
    ]
    outcomes = []
    for step in steps:
        try:
            outcomes.append(("ok", step()))
        except Exception as error:
            outcomes.append(("raise", type(error), str(error)))
    return outcomes


def test_annotations_behave_like_a_functions():
    def make():
        def f(a: int, b: list[int]) -> int:
            return a

        return f

    assert _annotation_steps(snekit(make())) == _annotation_steps(make())


def test_a_function_without_annotations():
    def f(a):
        return a

    wrapper = snekit(f)
    assert wrapper.__annotate__ is None
    assert wrapper.__annotations__ == {} == f.__annotations__


def test_pickle_and_copy_by_reference_like_a_function():
    assert pickle.loads(pickle.dumps(module_level_add)) is module_level_add
    assert copy.copy(module_level_add) is module_level_add
    assert copy.deepcopy(module_level_add) is module_level_add
    with pytest.raises(pickle.PicklingError, match="local object"):
        pickle.dumps(make_add())


def test_repr_names_the_original_function():
    add = make_add()
    assert repr(add) == f"<snekit wrapper of {add.__wrapped__!r}>"


def test_arbitrary_attributes_and_weak_references():
    add = make_add()
    add.custom = 1
    assert vars(add)["custom"] == 1
    assert weakref.ref(add)() is add


# --- Calls, call_count and errors ----------------------------------------


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [((), {}), ((1,), {}), ((1, 2, 3), {}), ((1, 2), {"c": 3}), ((1,), {"a": 2})],
)
def test_invalid_calls_raise_the_originals_type_error_and_count(args, kwargs):
    add = warmed(make_add(), 1, 2)
    with pytest.raises(TypeError) as expected:
        add.__wrapped__(*args, **kwargs)
    count, calls = add.call_count, native_calls(add)
    with pytest.raises(TypeError) as actual:
        add(*args, **kwargs)
    assert str(actual.value) == str(expected.value)
    assert actual.value.__context__ is None
    assert add.call_count == count + 1
    assert native_calls(add) == calls
    assert add._jit_state.status is _Status.COMPILED


def test_an_empty_keyword_dict_is_a_positional_call():
    add = warmed(make_add(), 1, 2)
    calls = native_calls(add)
    assert add(1, 2, **{}) == 3
    assert native_calls(add) == calls + 1


def test_call_count_can_be_set_to_an_int():
    add = make_add()
    add.call_count = 41
    add(1, 2)
    assert add.call_count == 42
    assert type(add.call_count) is int


def _count_in_threads(f, start, threads=4, calls=5000):
    f.call_count = start
    workers = [
        threading.Thread(target=lambda: [f(i, 1) for i in range(calls)])
        for _ in range(threads)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return f.call_count - start


def test_call_count_is_exact_under_threads():
    add = warmed(make_add(), 1, 2)
    assert _count_in_threads(add, 0) == 20000


# The counter is an ``i64`` that every call increments, plus an exact
# Python ``int`` that takes over beyond the i64 range.
I64_MAX = 2**63 - 1


def test_call_count_crosses_the_i64_limit_exactly():
    add = warmed(make_add(), 1, 2)
    add.call_count = I64_MAX
    assert add(1, 2) == 3
    assert add.call_count == I64_MAX + 1
    assert type(add.call_count) is int


def test_call_count_keeps_counting_after_crossing():
    add = make_add()
    add.call_count = I64_MAX - 2
    for i in range(1, 11):
        add(1, 2)
        assert add.call_count == I64_MAX - 2 + i
    assert add.call_count == I64_MAX + 8


def test_call_count_crosses_on_a_call_that_raises():
    @snekit
    def fails(a):
        raise ValueError(a)

    fails.call_count = I64_MAX
    with pytest.raises(ValueError) as excinfo:
        fails("boom")
    # The original function ran and raised normally.
    assert excinfo.value.args == ("boom",)
    assert excinfo.value.__context__ is None
    assert fails.call_count == I64_MAX + 1
    with pytest.raises(ValueError):
        fails("again")
    assert fails.call_count == I64_MAX + 2


def test_call_count_crosses_on_a_native_call_and_a_guard_miss():
    add = warmed(make_add(), 1, 2)
    calls = native_calls(add)
    add.call_count = I64_MAX
    assert add(20, 22) == 42  # native
    assert add(0.5, 1) == 1.5  # guard miss
    assert native_calls(add) == calls + 2
    assert add.call_count == I64_MAX + 2


def test_call_count_can_be_reset_after_overflow():
    add = make_add()
    add.call_count = I64_MAX
    add(1, 2)
    add.call_count = 0
    add(1, 2)
    assert add.call_count == 1
    add.call_count = I64_MAX
    add(1, 2)
    add(1, 2)
    assert add.call_count == I64_MAX + 2


@pytest.mark.parametrize("start", [2**64, 2**200, -(2**63) - 1, -(2**100)])
def test_call_count_accepts_ints_beyond_i64(start):
    add = make_add()
    add.call_count = start
    assert add.call_count == start
    add(1, 2)
    add(1, 2)
    assert add.call_count == start + 2
    assert type(add.call_count) is int


def test_call_count_crosses_the_limit_under_threads():
    add = warmed(make_add(), 1, 2)
    assert _count_in_threads(add, I64_MAX - 7000) == 20000


def test_defaults_and_code_changes_reach_the_original():
    @snekit
    def f(a, b=4):
        return a * b + a - b

    warmed(f, 3, 4)
    assert f(3) == 11
    f.__wrapped__.__defaults__ = (10,)
    assert f(3) == 23
    # Another signature; the function keeps its ``__defaults__``.
    f.__wrapped__.__code__ = (lambda a, b, c=1: a + b + c).__code__
    assert f(1, 2) == 13
    assert f(1, 2, c=3) == 6
    assert f._jit_state.status is _Status.DISABLED


def test_tracebacks_end_in_the_original_function():
    add = warmed(make_add(), 1, 2)
    with pytest.raises(TypeError) as excinfo:
        add(1, "x")
    frames = traceback.extract_tb(excinfo.value.__traceback__)
    assert frames[-1].name == "add"
    # No wrapper frame: the native wrapper has none.
    assert [frame.name for frame in frames] == [
        "test_tracebacks_end_in_the_original_function",
        "add",
    ]


def test_exceptions_keep_the_callers_context():
    add = warmed(make_add(), 1, 2)
    outer = KeyError("outer")
    try:
        raise outer
    except KeyError:
        with pytest.raises(TypeError) as excinfo:
            add(1, "x")
    assert excinfo.value.__context__ is outer


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_base_exceptions_from_the_original_propagate(error):
    @snekit
    def f(a):
        raise error

    with pytest.raises(error):
        f(1)
    assert f.call_count == 1


# --- Ownership and garbage collection ------------------------------------


def _live_compiled():
    gc.collect()
    return sum(type(obj) is CompiledI64 for obj in gc.get_objects())


def _cycle(kind):
    """A compiled wrapper with a reference cycle of the given ``kind``."""

    def original(a, b):
        return a + b

    wrapper = snekit(original)
    warmed(wrapper, 1, 2)
    if kind == "wrapper dict":
        wrapper.me = wrapper
    elif kind == "original dict":
        original.wrapper = wrapper
    elif kind == "annotations":
        wrapper.__annotate__ = lambda format: {"self": wrapper}
        assert wrapper.__annotations__ == {"self": wrapper}
    elif kind == "state":
        wrapper._jit_state.native_error = Exception(wrapper)
    return weakref.ref(wrapper), weakref.ref(original)


@pytest.mark.parametrize(
    "kind", ["none", "wrapper dict", "original dict", "annotations", "state"]
)
def test_wrappers_and_their_machine_code_are_collected(kind):
    baseline = _live_compiled()
    wrapper, original = _cycle(kind)
    if kind == "none":
        # No cycle at all: freed by reference counting alone.
        assert wrapper() is None
    gc.collect()
    assert wrapper() is None
    assert original() is None
    assert _live_compiled() == baseline


def test_a_recursive_function_is_collected():
    baseline = _live_compiled()

    def make():
        # ``f`` reads its own wrapper from a closure cell: a cycle through
        # the original function. (The frontend rejects closures.)
        @snekit
        def f(a, b):
            return a + b if a is not None else f

        for _ in range(_WARMUP_CALLS + 1):
            f(1, 2)
        assert f(None, 1) is f
        return weakref.ref(f)

    wrapper = make()
    gc.collect()
    assert wrapper() is None
    assert _live_compiled() == baseline


def test_the_wrapper_owns_its_original_and_state():
    add = make_add()
    original, state = add.__wrapped__, add._jit_state
    del add.__wrapped__, add._jit_state
    for _ in range(_WARMUP_CALLS + 2):
        assert add(1, 2) == 3
    assert state.status is _Status.COMPILED
    referents = gc.get_referents(add)
    assert any(r is original for r in referents)
    assert any(r is state for r in referents)


# --- Documented differences from a function ------------------------------


def test_the_wrapper_is_not_a_function():
    add = make_add()
    assert type(add) is _native.NativeWrapper
    assert not inspect.isfunction(add)
    assert inspect.isroutine(inspect.unwrap(add))
    assert not hasattr(add, "__code__")
    assert not hasattr(add, "__defaults__")
    assert not hasattr(add, "__globals__")


def test_call_count_accepts_only_integers():
    # A function wrapper stored any value, and failed on the next call for
    # a non-number. The native wrapper accepts what ``operator.index``
    # accepts, of any size, and rejects the rest when it is set.
    add = make_add()
    add.call_count = True
    assert add.call_count == 1
    assert type(add.call_count) is int
    for value in (1.5, "1", None):
        with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
            add.call_count = value
    with pytest.raises(AttributeError):
        del add.call_count
    assert add.call_count == 1


def test_profilers_see_no_wrapper_frame():
    add = warmed(make_add(), 1, 2)
    events = []

    def profiler(frame, event, arg):
        events.append((event, frame.f_code.co_name))

    sys.setprofile(profiler)
    try:
        add(1, 2.5)
    finally:
        sys.setprofile(None)
    names = {name for event, name in events if event == "call"}
    assert names == {"add"}


# --- The hot slot ----------------------------------------------------------
#
# Once compiled, a wrapper keeps its specialization in a Rust ``HotSlot``,
# and a call that can use it reads no attribute of the JIT state. These
# tests pin that warmed calls use it, that it changes nothing observable,
# and that it is retired whenever the specialization stops being usable.


class Poison:
    """Stands in for ``state.specialization``: reading it fails the test."""

    def __getattribute__(self, name):
        raise AssertionError(f"the generic path read specialization.{name}")


@contextlib.contextmanager
def generic_path_forbidden(f):
    """Make every call that reads the state's specialization fail.

    Stores ``_specialization`` directly: assigning ``specialization`` would
    retire the slot.
    """
    state = f._jit_state
    real = state._specialization
    state._specialization = Poison()
    try:
        yield
    finally:
        state._specialization = real


@contextlib.contextmanager
def python_frames(f):
    """Count the frames of ``f``'s original function entered meanwhile."""
    code = f.__wrapped__.__code__
    count = [0]

    def profiler(frame, event, arg):
        if event == "call" and frame.f_code is code:
            count[0] += 1

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        yield count
    finally:
        sys.setprofile(previous)


def generic_wrapper(func):
    """``snekit(func)`` without a hot slot: every call takes the generic
    path, the dispatch ``snekit`` used before the slot existed."""
    state = _JitState(func)
    wrapper = _native.NativeWrapper(func, state, _Status.COLD)
    functools.update_wrapper(wrapper, func)
    wrapper._jit_state = state
    return wrapper


def copy_function(func):
    return types.FunctionType(
        func.__code__, func.__globals__, func.__name__, func.__defaults__
    )


def test_compilation_publishes_the_hot_slot():
    add = make_add()
    state = add._jit_state
    assert type(state._hot) is _native.HotSlot
    for _ in range(_WARMUP_CALLS):
        add(1, 2)
        assert not state._hot.live
    add(1, 2)  # Compiles.
    assert state.status is _Status.COMPILED
    assert state._hot.live


def test_a_cold_call_runs_no_state_code_but_cold_call():
    # Retiring the slot hooks only the assignment to ``specialization``:
    # counting a warmup call runs no Python code besides ``cold_call``.
    add = make_add()
    add(1, 2)
    decorator = _JitState.cold_call.__code__.co_filename
    names = []

    def profiler(frame, event, arg):
        if event == "call" and frame.f_code.co_filename == decorator:
            names.append(frame.f_code.co_name)

    sys.setprofile(profiler)
    try:
        add(1, 2)
    finally:
        sys.setprofile(None)
    assert names == ["cold_call"]


def test_warmed_calls_use_the_hot_slot():
    add = warmed(make_add(), 1, 2)
    calls, count = native_calls(add), add.call_count
    with generic_path_forbidden(add), python_frames(add) as frames:
        results = [add(i, -3) for i in range(100)]
    assert results == [i - 3 for i in range(100)]
    assert all(type(result) is int for result in results)
    assert frames[0] == 0
    assert native_calls(add) == calls + 100
    assert add.call_count == count + 100
    assert add._jit_state.status is _Status.COMPILED
    assert add._jit_state._hot.live


class MyInt(int):
    pass


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((1.5, 2), 3.5),
        ((True, 2), 3),
        ((MyInt(1), 2), 3),
        ((2**63, 1), 2**63 + 1),
        ((2**62, 2**62), 2**63),  # Overflow.
        ((-(2**63), -1), -(2**63) - 1),  # Overflow.
        (("a", "b"), "ab"),
    ],
)
def test_guard_misses_in_the_hot_slot_fall_back_once(args, expected):
    add = warmed(make_add(), 1, 2)
    calls = native_calls(add)
    with generic_path_forbidden(add), python_frames(add) as frames:
        result = add(*args)
    assert result == expected
    assert type(result) is type(expected)
    assert frames[0] == 1
    assert native_calls(add) == calls + 1
    assert add._jit_state.status is _Status.COMPILED
    assert add._jit_state._hot.live
    # The next call runs natively again.
    with generic_path_forbidden(add):
        assert add(20, 22) == 42
    assert native_calls(add) == calls + 2


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [((), {}), ((1,), {}), ((1, 2, 3), {}), ((1, 2), {"c": 3}), ((1,), {"a": 2})],
)
def test_invalid_calls_in_the_hot_slot_raise_the_originals_type_error(args, kwargs):
    add = warmed(make_add(), 1, 2)
    with pytest.raises(TypeError) as expected:
        add.__wrapped__(*args, **kwargs)
    count, calls = add.call_count, native_calls(add)
    with generic_path_forbidden(add), pytest.raises(TypeError) as actual:
        add(*args, **kwargs)
    assert str(actual.value) == str(expected.value)
    assert actual.value.__context__ is None
    assert add.call_count == count + 1
    assert native_calls(add) == calls
    assert add._jit_state._hot.live


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [((1,), {"b": 2}), ((), {"a": 1, "b": 2}), ((), {"b": 2, "a": 1})],
)
def test_keyword_calls_never_reach_native_code(args, kwargs):
    add = warmed(make_add(), 1, 2)
    calls = native_calls(add)
    with generic_path_forbidden(add), python_frames(add) as frames:
        assert add(*args, **kwargs) == 3
    assert frames[0] == 1
    assert native_calls(add) == calls
    # An empty keyword dict is a positional call.
    with generic_path_forbidden(add):
        assert add(1, 2, **{}) == 3
    assert native_calls(add) == calls + 1


def test_defaults_in_the_hot_slot():
    @snekit
    def f(a, b=4):
        return a * b + a - b

    warmed(f, 3, 4)
    calls = native_calls(f)
    with generic_path_forbidden(f), python_frames(f) as frames:
        # An omitted default: another arity, so no native call.
        assert f(3) == 11
        assert f(3, 5) == 13
        assert f(3, b=5) == 13
        f.__wrapped__.__defaults__ = (10,)
        assert f(3) == 23
    assert frames[0] == 3
    assert native_calls(f) == calls + 1


def test_replacing_code_after_compilation_retires_the_slot():
    add = warmed(make_add(), 1, 2)
    state = add._jit_state
    original = add.__wrapped__.__code__
    calls = native_calls(add)
    add.__wrapped__.__code__ = (lambda a, b: a * b).__code__
    # The first call after the change already runs the new code.
    with python_frames(add) as frames:
        assert add(6, 7) == 42
    assert frames[0] == 1
    assert native_calls(add) == calls
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.CODE_CHANGED
    assert not state._hot.live
    # Restoring the compiled code object does not revive the slot.
    add.__wrapped__.__code__ = original
    for _ in range(5):
        assert add(6, 7) == 13
    assert native_calls(add) == calls
    assert not state._hot.live


def test_replacing_code_with_another_signature_retires_the_slot():
    add = warmed(make_add(), 1, 2)
    add.__wrapped__.__code__ = (lambda a, b, c: a + b + c).__code__
    with pytest.raises(TypeError, match="missing 1 required positional"):
        add(1, 2)
    assert add(1, 2, 3) == 6
    assert add._jit_state.disabled_reason is _DisabledReason.CODE_CHANGED
    assert not add._jit_state._hot.live


def test_replacing_code_before_compilation_publishes_the_new_code():
    add = make_add()
    for _ in range(_WARMUP_CALLS):
        add(1, 2)
    multiply = (lambda a, b: a * b).__code__
    add.__wrapped__.__code__ = multiply
    assert add(6, 7) == 42  # Compiles the new code.
    state = add._jit_state
    assert state.compiled_code is multiply
    assert state._hot.live
    calls = native_calls(add)
    with generic_path_forbidden(add):
        assert add(5, 7) == 35
    assert native_calls(add) == calls + 1


RANGE_SOURCE = "def f(a, b):\n    for i in range(a):\n        b += i\n    return b"


def _disabled(reason, monkeypatch):
    """A wrapper disabled for ``reason``, through the real transition."""
    reasons = _DisabledReason
    namespace = {}
    if reason in (reasons.RANGE_UNGUARDED, reasons.RANGE_CHANGED):
        if reason is reasons.RANGE_UNGUARDED:
            namespace["range"] = lambda n: range(n)
        exec(RANGE_SOURCE, namespace)
    else:
        bodies = {reasons.REJECTED: "a / b", reasons.UNSUPPORTED: "a + 10**30"}
        exec(f"def f(a, b):\n    return {bodies.get(reason, 'a + b')}", namespace)
    f = snekit(namespace["f"])
    if reason in (reasons.COMPILER_ERROR, reasons.INTERRUPTED):
        error = RuntimeError if reason is reasons.COMPILER_ERROR else KeyboardInterrupt

        def broken(*args, **kwargs):
            raise error

        monkeypatch.setattr(_compiler, "compile_i64", broken)
    elif reason in (reasons.CODE_CHANGED, reasons.RANGE_CHANGED, reasons.NATIVE_ERROR):
        warmed(f, 1, 2)
        assert f._jit_state._hot.live
        if reason is reasons.CODE_CHANGED:
            # An equal code object, but not the one that was compiled.
            f.__wrapped__.__code__ = f.__wrapped__.__code__.replace()
        elif reason is reasons.RANGE_CHANGED:
            namespace["range"] = range
        else:
            # What the wrapper does when a native call raises; a genuine
            # ``I64Function`` cannot be made to raise from Python.
            f._jit_state.native_failed(RuntimeError("native bug"))
    for _ in range(_WARMUP_CALLS + 1):
        try:
            f(1, 2)
        except KeyboardInterrupt:
            assert reason is reasons.INTERRUPTED
    monkeypatch.undo()
    return f


@pytest.mark.parametrize("reason", list(_DisabledReason), ids=lambda r: r.value)
def test_every_disabled_reason_retires_the_slot(reason, monkeypatch):
    f = _disabled(reason, monkeypatch)
    state = f._jit_state
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is reason
    assert not state._hot.live
    calls = native_calls(f)
    with python_frames(f) as frames:
        f(1, 2)
    assert frames[0] == 1
    assert native_calls(f) == calls


class BrokenNative:
    """A test double for a ``CompiledI64`` whose fast call raises."""

    arg_count = 2
    function = None

    def __init__(self):
        self.calls = 0

    def fast_call(self, args):
        self.calls += 1
        raise RuntimeError("native bug")


def test_an_injected_specialization_retires_the_slot():
    add = warmed(make_add(), 1, 2)
    state = add._jit_state
    broken = BrokenNative()
    state.specialization = _Specialization(state.compiled_code, broken)
    assert not state._hot.live
    assert add(1, 2) == 3
    assert broken.calls == 1
    assert state.disabled_reason is _DisabledReason.NATIVE_ERROR


def test_a_wrapper_without_a_slot_takes_the_generic_path():
    def add(a, b):
        return a + b

    wrapper = warmed(generic_wrapper(add), 1, 2)
    assert wrapper._jit_state._hot is None
    calls = native_calls(wrapper)
    assert wrapper(20, 22) == 42
    assert native_calls(wrapper) == calls + 1
    with generic_path_forbidden(wrapper), pytest.raises(AssertionError, match="read"):
        wrapper(1, 2)


def test_a_slot_is_never_used_for_a_callable_that_is_not_a_function():
    def add(a, b):
        return a + b

    function = compile_i64(analyze(add).instructions).function
    hot = _native.HotSlot()
    hot.publish(add.__code__, function)
    assert hot.live

    class Forged:
        __code__ = add.__code__

        def __call__(self, a, b):
            return "python"

    forged = Forged()
    wrapper = _native.NativeWrapper(forged, _JitState(forged), _Status.COLD, hot)
    assert wrapper(1, 2) == "python"
    assert function.try_calls == 0


def test_publishing_rules():
    def add(a, b):
        return a + b

    function = compile_i64(analyze(add).instructions).function
    # Only a genuine native function is published.
    for double in (None, BrokenNative(), function.try_call):
        hot = _native.HotSlot()
        hot.publish(add.__code__, double)
        assert not hot.live
        hot.publish(add.__code__, function)
        assert hot.live
    with pytest.raises(TypeError, match="code object"):
        _native.HotSlot().publish(add, function)
    # Write-once.
    with pytest.raises(RuntimeError, match="already published"):
        hot.publish(add.__code__, function)
    # One-way and idempotent: a retired slot is never live again.
    hot.retire()
    hot.retire()
    assert not hot.live
    retired = _native.HotSlot()
    retired.retire()
    retired.publish(add.__code__, function)
    assert not retired.live
    assert {name for name in dir(hot) if not name.startswith("_")} == {
        "live",
        "publish",
        "retire",
    }


def _outcome(f, *args, **kwargs):
    calls = native_calls(f)
    with python_frames(f) as frames:
        try:
            result = ("ok", type(f(*args, **kwargs)), f.__wrapped__(*args, **kwargs))
        except Exception as error:
            result = ("raise", type(error), str(error))
    state = f._jit_state
    return (
        result,
        frames[0],
        native_calls(f) - calls,
        f.call_count,
        state.status,
        state.disabled_reason,
        state.compile_attempts,
        state.warmup_calls,
    )


def test_the_hot_slot_matches_the_generic_path_call_by_call():
    def f(a, b=3):
        return a * b - (a + b)

    hot, generic = snekit(copy_function(f)), generic_wrapper(copy_function(f))
    calls = [((1, 2), {})] * (_WARMUP_CALLS + 1)
    calls += [
        ((5, 7), {}),
        ((5,), {}),
        ((5,), {"b": 7}),
        ((1.5, 2), {}),
        ((True, 2), {}),
        ((2**40, 2**40), {}),
        ((2**63, 1), {}),
        ((-(2**63), 2), {}),
        (("x", 2), {}),
        ((), {}),
        ((1, 2, 3), {}),
        ((1,), {"c": 2}),
        ((-4, -9), {}),
    ]
    for args, kwargs in calls:
        assert _outcome(hot, *args, **kwargs) == _outcome(generic, *args, **kwargs)
    assert hot._jit_state._hot.live
    hot.__wrapped__.__code__ = generic.__wrapped__.__code__ = (
        lambda a, b=3: a - b
    ).__code__
    for args, kwargs in calls[-6:]:
        assert _outcome(hot, *args, **kwargs) == _outcome(generic, *args, **kwargs)
    assert not hot._jit_state._hot.live


def _wait_for(condition):
    deadline = time.monotonic() + 30
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.001)


def test_hot_slot_calls_from_several_threads():
    @snekit
    def f(a, b):
        return (a + 2) * b - 1000

    warmed(f, 1, 2)
    f.call_count = 0
    replaced, stop = threading.Event(), threading.Event()
    counts, errors = [0] * 4, []

    def worker(index):
        rng = random.Random(index)
        while not stop.is_set():
            a, b = rng.randint(-(2**20), 2**20), rng.randint(-(2**20), 2**20)
            args = rng.choice([(a, b), (float(a), b), (2**62, b | 1)])
            after = replaced.is_set()
            result = f(*args)
            counts[index] += 1
            new = args[0] - args[1]
            old = (args[0] + 2) * args[1] - 1000
            # Once the new code is visible, no call may run the old one.
            if result != new and (after or result != old):
                errors.append((after, args, result))

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in workers:
        thread.start()
    _wait_for(lambda: min(counts) > 500)
    native = native_calls(f)
    assert f._jit_state._hot.live
    f.__wrapped__.__code__ = (lambda a, b: a - b).__code__
    replaced.set()
    _wait_for(lambda: min(counts) > 1000)
    stop.set()
    for thread in workers:
        thread.join()
    assert errors == []
    assert native > 0
    assert f.call_count == sum(counts)
    assert f._jit_state.disabled_reason is _DisabledReason.CODE_CHANGED
    assert not f._jit_state._hot.live


def test_the_hot_slot_releases_the_machine_code_with_the_wrapper():
    # The reference count of an ``I64Function`` that only a local holds.
    lone = _native.compile_i64(((_native.OP_LOAD_CONST, 1), (_native.OP_RETURN,)), 0)
    alone = sys.getrefcount(lone)
    add = warmed(make_add(), 1, 2)
    hot = add._jit_state._hot
    function = add._jit_state.compiled.function
    # Nothing in the slot can lead back to it, so it needs no GC support.
    assert not gc.is_tracked(hot)
    del add
    gc.collect()
    # Still held by the slot, which is still alive.
    assert hot.live
    assert sys.getrefcount(function) == alone + 1
    del hot
    assert sys.getrefcount(function) == alone


def test_a_module_level_range_function_is_collected():
    # The wrapper is reachable from its own function's globals: a cycle.
    namespace = {}
    exec(
        "def f(n):\n    t = 0\n    for i in range(n):\n        t += i\n    return t",
        namespace,
    )
    wrapper = warmed(snekit(namespace["f"]), 4)
    namespace["f"] = wrapper
    assert wrapper._jit_state._hot.live
    ref = weakref.ref(wrapper)
    del wrapper, namespace
    gc.collect()
    assert ref() is None
