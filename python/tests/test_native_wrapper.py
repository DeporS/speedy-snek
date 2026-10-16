"""``@snekit`` returns a native callable object, not a Python function.

These tests cover what such an object could get wrong that a function gets
for free: binding as a method, metadata and introspection, lazy
annotations, pickling, ``call_count``, tracebacks and garbage collection.
The last section pins the documented differences from a function (see
``benchmarks/experiments/native_wrapper.md``), so that changing one is a
deliberate decision.
"""

import abc
import copy
import functools
import gc
import inspect
import pickle
import sys
import threading
import traceback
import types
import typing
import weakref

import pytest

from speedysnek import _native, snekit
from speedysnek._compiler import CompiledI64
from speedysnek._decorator import _WARMUP_CALLS, _Status


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


def test_call_count_is_exact_under_threads():
    add = warmed(make_add(), 1, 2)
    add.call_count = 0
    threads = [
        threading.Thread(target=lambda: [add(i, 1) for i in range(5000)])
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert add.call_count == 20000


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


def test_call_count_accepts_only_ints_in_i64_range():
    add = make_add()
    with pytest.raises(TypeError):
        add.call_count = 1.5
    with pytest.raises(OverflowError):
        add.call_count = 2**63
    with pytest.raises(AttributeError):
        del add.call_count
    assert add.call_count == 0


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
