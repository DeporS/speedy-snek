"""Tests for the miss policy: parking and re-probing a function whose calls
do not fit its i64 specialization.

Paths are proven, never inferred from an equal result: native attempts are
counted by the compiled function itself (``I64Function.try_counts``), and
frames of the original function with ``sys.setprofile``. Every result and
exception is compared with the undecorated function's.
"""

import collections
import contextlib
import json
import random
import subprocess
import sys
import textwrap
import threading
import time
import types

import pytest

import speedysnek as snek
from speedysnek import _native, peek, snekit
from speedysnek._decorator import (
    _MIN_FITTING_WARMUP_CALLS,
    _WARMUP_CALLS,
    _DisabledReason,
    _JitState,
    _ParkedReason,
    _Specialization,
    _Status,
)

MISS_LIMIT = _native.MISS_LIMIT
REPROBE_INTERVAL = _native.REPROBE_INTERVAL
I64_MAX = 2**63 - 1


# --- Helpers -------------------------------------------------------------


def make_add():
    @snekit
    def add(a, b):
        return a + b

    return add


def make_pair(params="a, b"):
    namespace = {}
    exec(f"def pair({params}):\n    return a * b + a - b\n", namespace)
    return snekit(namespace["pair"])


def state_of(f):
    return f._jit_state


def counts(f):
    """``(attempts, misses)`` of ``f``'s native function; zeros if none."""
    compiled = state_of(f).compiled
    return (0, 0) if compiled is None else compiled.function.try_counts


def attempts(f):
    return counts(f)[0]


@contextlib.contextmanager
def frames():
    """Count, in this thread, the Python frames entered, by code object."""
    calls = collections.Counter()

    def profiler(frame, event, arg):
        if event == "call":
            calls[frame.f_code] += 1

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def outcome(f, *args, **kwargs):
    try:
        result = f(*args, **kwargs)
    except Exception as error:
        return ("raise", type(error), str(error))
    return ("ok", type(result), result)


def path(f, *args, **kwargs):
    """Call ``f`` once, check its outcome, and say which path it took:
    ``"native"``, ``"miss"`` (a native attempt, then the original) or
    ``"python"`` (the original, no native attempt)."""
    expected = outcome(f.__wrapped__, *args, **kwargs)
    before = attempts(f)
    with frames() as calls:
        actual = outcome(f, *args, **kwargs)
    assert actual == expected
    native = attempts(f) - before
    python = calls[f.__wrapped__.__code__]
    paths = {(1, 0): "native", (1, 1): "miss", (0, 1): "python"}
    return paths[native, python]


def warm(f, *args, **kwargs):
    """The warmup calls and the deciding call, all with these arguments."""
    for _ in range(_WARMUP_CALLS + 1):
        f(*args, **kwargs)
    return state_of(f)


def until_reprobe(f, *args, **kwargs):
    """Make parked calls until the next call that fits re-probes."""
    hot = state_of(f)._hot
    assert hot.parked
    while hot.parked_calls < REPROBE_INTERVAL:
        assert path(f, *args, **kwargs) == "python"


def park_compiled(f, *args):
    """Compile ``f`` with ints, then park it with ``MISS_LIMIT`` misses."""
    state = warm(f, 3, 4)
    assert state.status is _Status.COMPILED
    for _ in range(MISS_LIMIT):
        assert state.status is _Status.COMPILED
        assert path(f, *args) == "miss"
    assert state.status is _Status.PARKED
    assert state.parked_reason is _ParkedReason.MISSES
    return state


@pytest.fixture
def compiles(monkeypatch):
    """Record every compilation attempt."""
    calls = []
    real = _JitState._compile

    def spy(self):
        calls.append(self._func.__name__)
        return real(self)

    monkeypatch.setattr(_JitState, "_compile", spy)
    return calls


class Scaled:
    def __init__(self, factor):
        self.factor = factor

    @snekit
    def double(self, x):
        return x * 2


class MyInt(int):
    pass


# --- Warmup decisions ------------------------------------------------------


def test_an_exact_int_function_still_compiles(compiles):
    add = make_add()
    state = warm(add, 1, 2)
    assert state.status is _Status.COMPILED
    assert state.warmup_fits == _WARMUP_CALLS
    assert compiles == ["add"]
    assert path(add, 3, 4) == "native"
    assert state._hot.live and not state._hot.parked


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((1.5, 2.5), {}),  # floats
        ((1, 2.5), {}),  # one float
        ((True, 2), {}),  # bool
        ((MyInt(1), 2), {}),  # an int subclass
        ((2**63, 1), {}),  # outside i64
        ((-(2**63) - 1, 1), {}),  # outside i64
        (("a", "b"), {}),  # not numbers at all
        ((1,), {"b": 2}),  # a keyword
        ((), {"a": 1, "b": 2}),  # only keywords
    ],
)
def test_calls_that_cannot_fit_park_without_compiling(args, kwargs, compiles):
    add = make_add()
    state = warm(add, *args, **kwargs)
    assert state.status is _Status.PARKED
    assert state.parked_reason is _ParkedReason.WARMUP
    assert (state.warmup_fits, state.compile_attempts) == (0, 0)
    assert compiles == []
    assert state.compiled is None and state.specialization is None
    assert state._hot.parked and not state._hot.live
    assert path(add, *args, **kwargs) == "python"
    assert add.call_count == _WARMUP_CALLS + 2


def test_an_instance_method_parks_without_compiling(compiles):
    instances = [Scaled(i) for i in range(3)]
    for _ in range(_WARMUP_CALLS + 1):
        for i, instance in enumerate(instances):
            assert instance.double(i) == 2 * i
    state = state_of(Scaled.double)
    assert state.status is _Status.PARKED
    assert state.parked_reason is _ParkedReason.WARMUP
    assert state.compile_attempts == 0
    assert compiles == []
    # A bound call passes the instance as ``self``.
    assert instances[0].double(21) == 42
    assert path(Scaled.double, instances[0], 21) == "python"


def test_omitted_defaults_park(compiles):
    @snekit
    def f(a, b=5):
        return a + b

    state = warm(f, 1)
    assert state.status is _Status.PARKED
    assert state.warmup_fits == 0
    assert compiles == []
    assert path(f, 1) == "python"
    # A re-probe with the omitted default checks the arity and compiles
    # nothing; the interval starts again.
    until_reprobe(f, 1)
    assert path(f, 1) == "python"
    assert state._hot.parked_calls == 0
    assert (state.status, state.compile_attempts) == (_Status.PARKED, 0)
    # A call that passes both arguments is the next re-probe's chance.
    until_reprobe(f, 1)
    assert path(f, 1, 2) == "native"
    assert (state.status, state.compile_attempts) == (_Status.COMPILED, 1)


def test_repeated_keyword_calls_park(compiles):
    add = make_add()
    state = warm(add, a=1, b=2)
    assert state.status is _Status.PARKED
    assert compiles == []
    # Keyword calls never re-probe: they could not run natively anyway.
    for _ in range(3 * REPROBE_INTERVAL):
        assert path(add, a=1, b=2) == "python"
    assert state._hot.parked_calls == REPROBE_INTERVAL
    assert (state.status, state.compile_attempts) == (_Status.PARKED, 0)


@pytest.mark.parametrize(
    ("fitting", "status"),
    [
        (_MIN_FITTING_WARMUP_CALLS, _Status.COMPILED),
        (_MIN_FITTING_WARMUP_CALLS - 1, _Status.PARKED),
    ],
)
def test_the_warmup_threshold(fitting, status):
    add = make_add()
    calls = [(1, 2)] * fitting + [(1.5, 2)] * (_WARMUP_CALLS - fitting)
    random.Random(fitting).shuffle(calls)
    for args in calls:
        assert add(*args) == args[0] + 2
    # The deciding call itself is not part of the profile.
    add(1.5, 2)
    state = state_of(add)
    assert (state.warmup_fits, state.status) == (fitting, status)


def test_alternating_warmup_compiles():
    add = make_add()
    for i in range(_WARMUP_CALLS + 1):
        add(i, 1) if i % 2 else add(i + 0.5, 1)
    assert state_of(add).status is _Status.COMPILED


def test_a_function_parked_by_floats_compiles_at_a_later_reprobe(compiles):
    add = make_add()
    state = warm(add, 1.5, 2.5)
    assert state.status is _Status.PARKED
    until_reprobe(add, 1.5, 2.5)
    # Calls that do not fit wait for one that does, at no native cost.
    for _ in range(10):
        assert path(add, 1.5, 2.5) == "python"
    assert state._hot.parked_calls == REPROBE_INTERVAL
    assert compiles == []
    # The first call that fits compiles the function and runs natively.
    assert path(add, 3, 4) == "native"
    assert (state.status, state.compile_attempts) == (_Status.COMPILED, 1)
    assert state.parked_reason is None
    assert compiles == ["add"]
    assert state._hot.live
    assert path(add, 5, 6) == "native"
    assert counts(add) == (2, 0)


def test_a_reprobe_compiles_at_most_once(compiles):
    add = make_add()
    warm(add, 1.5, 2.5)
    until_reprobe(add, 1.5, 2.5)
    assert path(add, 3, 4) == "native"
    for _ in range(MISS_LIMIT):
        add(1.5, 2.5)
    state = state_of(add)
    assert state.status is _Status.PARKED
    until_reprobe(add, 1.5, 2.5)
    assert path(add, 3, 4) == "native"
    assert compiles == ["add"]
    assert state.compile_attempts == 1


def test_a_rejected_function_parked_during_warmup_is_rejected_at_its_reprobe():
    @snekit
    def divide(a, b):
        return a // b

    state = warm(divide, 7.5, 2)
    assert state.status is _Status.PARKED
    until_reprobe(divide, 7.5, 2)
    assert path(divide, 7, 2) == "python"
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.REJECTED
    assert state.parked_reason is None
    assert path(divide, 7, 2) == "python"


class Recorder:
    """Records any attempt to inspect, convert or compare it."""

    calls = []

    def __getattribute__(self, name):
        Recorder.calls.append(name)
        return object.__getattribute__(self, name)

    def _record(self, *args):
        Recorder.calls.append("dunder")
        return NotImplemented

    __index__ = __int__ = __float__ = __eq__ = __lt__ = __le__ = _record
    __gt__ = __ge__ = __hash__ = __bool__ = __len__ = __iter__ = _record

    def __repr__(self):
        Recorder.calls.append("__repr__")
        return "Recorder()"


class RecordingMeta(type):
    lookups = []

    def __getattribute__(cls, name):
        RecordingMeta.lookups.append(name)
        raise RuntimeError(f"metaclass hook ran for {name!r}")


class Hostile(metaclass=RecordingMeta):
    pass


class HostileInt(int, metaclass=RecordingMeta):
    pass


def test_profiling_and_reprobing_run_no_user_code():
    @snekit
    def second(a, b):
        return b

    hostile, hostile_int, recorder = Hostile(), HostileInt(5), Recorder()
    RecordingMeta.lookups.clear()
    Recorder.calls.clear()
    arguments = [(hostile, 1), (1, hostile_int), (recorder, 2), (1, recorder)]
    # Warmup, the parking call, parked calls and due re-probes.
    for i in range(_WARMUP_CALLS + 3 * REPROBE_INTERVAL):
        args = arguments[i % len(arguments)]
        assert second(*args) is args[1]
    assert RecordingMeta.lookups == []
    assert Recorder.calls == []
    state = state_of(second)
    assert (state.status, state.compile_attempts) == (_Status.PARKED, 0)


def test_float_only_calls_load_no_compiler():
    script = textwrap.dedent(
        f"""\
        import json, sys
        import speedysnek

        @speedysnek.snekit
        def mean(a, b):
            return (a + b) * 0.5

        @speedysnek.snekit
        def scale(a, b=2):
            return a * b

        for i in range({_WARMUP_CALLS + 5 * REPROBE_INTERVAL}):
            mean(i + 0.5, 1.5)
            scale(i)
        print(json.dumps({{
            "loaded": sorted(name for name in sys.modules
                             if name.startswith("speedysnek._")),
            "status": [mean._jit_state.status.value,
                       scale._jit_state.status.value],
        }}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    assert report["status"] == ["parked", "parked"]
    assert "speedysnek._frontend" not in report["loaded"]
    assert "speedysnek._compiler" not in report["loaded"]


def test_only_functions_with_a_hot_slot_park():
    # A callable object is still rejected by the frontend, as before.
    class Callable:
        def __call__(self, a):
            return a

    wrapper = snekit(Callable())
    state = warm(wrapper, 1.5)
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.REJECTED

    # A state without a slot never profiles its way out of compiling.
    def add(a, b):
        return a + b

    state = _JitState(add)
    for _ in range(_WARMUP_CALLS):
        state.cold_call(-1)
    assert state.cold_call(-1) is not None
    assert state.status is _Status.COMPILED


# --- Compiled miss behaviour ----------------------------------------------


def test_an_occasional_float_miss_keeps_the_specialization():
    add = make_add()
    state = warm(add, 1, 2)
    specialization = state.specialization
    for i in range(10 * MISS_LIMIT):
        if i % (MISS_LIMIT // 2) == 0:
            assert path(add, 1.5, i) == "miss"
        else:
            assert path(add, i, 1) == "native"
    assert state.status is _Status.COMPILED
    assert state.specialization is specialization
    assert path(add, 3, 4) == "native"


@pytest.mark.parametrize(
    "args",
    [
        (1.5, 2),  # a float
        (2**62, 2**62),  # overflow
        (True, 2),  # bool
        (2**64, 2),  # outside i64
    ],
)
def test_repeated_misses_park_after_the_limit(args):
    f = make_pair()
    state = warm(f, 3, 4)
    for _ in range(MISS_LIMIT - 1):
        assert path(f, *args) == "miss"
    assert state.status is _Status.COMPILED
    assert state._hot.misses == MISS_LIMIT - 1
    # The limit-th miss in a row parks it, and still falls back once.
    assert path(f, *args) == "miss"
    assert state.status is _Status.PARKED
    assert state.parked_reason is _ParkedReason.MISSES
    assert state._hot.parked and not state._hot.live
    # Parked: the specialization is kept, but no native attempt is made.
    assert state.specialization is not None
    assert path(f, *args) == "python"
    assert f.call_count == _WARMUP_CALLS + 1 + MISS_LIMIT + 1


def test_keyword_and_default_calls_count_towards_parking():
    f = make_pair("a, b=4")
    state = warm(f, 3, 4)
    for i in range(MISS_LIMIT):
        assert state.status is _Status.COMPILED
        assert (path(f, a=1, b=i) if i % 2 else path(f, i)) == "python"
    assert state.status is _Status.PARKED
    assert attempts(f) == 1


def test_one_native_result_restarts_the_count():
    f = make_pair()
    state = warm(f, 3, 4)
    for _ in range(20):
        for _ in range(MISS_LIMIT - 1):
            assert path(f, 1.5, 2) == "miss"
        assert path(f, 2, 3) == "native"
        assert state._hot.misses == 0
    assert state.status is _Status.COMPILED


def test_alternating_inputs_never_park():
    f = make_pair()
    state = warm(f, 3, 4)
    for i in range(20 * MISS_LIMIT):
        args = (i, 3) if i % 2 else (i + 0.5, 3)
        assert path(f, *args) == ("native" if i % 2 else "miss")
    assert state.status is _Status.COMPILED
    assert counts(f) == (1 + 20 * MISS_LIMIT, 10 * MISS_LIMIT)


def test_parked_calls_make_no_native_attempts():
    f = make_pair()
    park_compiled(f, 1.5, 2)
    before = counts(f)
    # Floats never fit, so they never re-probe, however many there are.
    for _ in range(5 * REPROBE_INTERVAL):
        assert path(f, 1.5, 2) == "python"
    assert counts(f) == before


def test_parked_overflowing_calls_reprobe_once_per_interval():
    f = make_pair()
    state = park_compiled(f, 2**62, 2**62)
    before = attempts(f)
    rounds = 5
    for _ in range(rounds * (REPROBE_INTERVAL + 1)):
        assert f(2**62, 2**62) == f.__wrapped__(2**62, 2**62)
    # Every call fits the types, so every interval ends in a re-probe,
    # which overflows and stays parked.
    assert attempts(f) == before + rounds
    assert counts(f)[1] == MISS_LIMIT + rounds
    assert state.status is _Status.PARKED


def test_a_successful_reprobe_unparks():
    f = make_pair()
    state = park_compiled(f, 1.5, 2)
    specialization = state.specialization
    until_reprobe(f, 1.5, 2)
    assert path(f, 2, 3) == "native"
    assert state.status is _Status.COMPILED
    assert state.parked_reason is None
    assert state.specialization is specialization
    assert state._hot.live and not state._hot.parked
    assert state._hot.misses == 0
    # The fast path is back, and so is the miss count.
    for i in range(10):
        assert path(f, i, 3) == "native"
    for _ in range(MISS_LIMIT):
        assert path(f, 1.5, 2) == "miss"
    assert state.status is _Status.PARKED


def test_a_failed_reprobe_stays_parked():
    f = make_pair()
    state = park_compiled(f, 1.5, 2)
    until_reprobe(f, 1.5, 2)
    # Fits the types, but overflows: one native attempt, then Python.
    assert path(f, 2**62, 2**62) == "miss"
    assert state.status is _Status.PARKED
    assert state._hot.parked_calls == 0
    # The interval starts again: no attempt for the next calls.
    for _ in range(REPROBE_INTERVAL):
        assert path(f, 2**62, 2**62) == "python"
    assert path(f, 2**62, 2**62) == "miss"


def test_a_reprobe_needs_the_compiled_arity():
    f = make_pair("a, b=4")
    state = park_compiled(f, 1.5, 2)
    until_reprobe(f, 1.5)
    assert path(f, 2) == "python"
    assert path(f, a=1, b=2) == "python"
    assert state._hot.parked_calls == REPROBE_INTERVAL
    assert path(f, 2, 4) == "native"
    assert state.status is _Status.COMPILED


def test_results_and_exceptions_match_python_under_every_policy_state():
    f = make_pair("a, b=4")
    original = f.__wrapped__
    rng = random.Random(1234)
    choices = [
        lambda: ((rng.randint(-100, 100), rng.randint(-100, 100)), {}),
        lambda: ((rng.random(), 2), {}),
        lambda: ((2**62, 2**62), {}),
        lambda: ((2**64, 3), {}),
        lambda: ((True, 3), {}),
        lambda: ((MyInt(3), 3), {}),
        lambda: ((5,), {}),
        lambda: ((), {"a": 1, "b": 2}),
        lambda: (("x", 2), {}),  # raises TypeError
        lambda: ((1, 2, 3), {}),  # raises TypeError
        lambda: ((), {"c": 1}),  # raises TypeError
    ]
    seen = set()
    for _ in range(20):
        # Long runs of one kind of call drive the state through every
        # transition; mixed runs keep it compiled.
        weights = [rng.random() ** 4 for _ in choices]
        for _ in range(rng.randint(1, 300)):
            args, kwargs = rng.choices(choices, weights)[0]()
            assert outcome(f, *args, **kwargs) == outcome(original, *args, **kwargs)
            seen.add(state_of(f).status)
    assert seen >= {_Status.COLD, _Status.COMPILED, _Status.PARKED}
    total = counts(f)
    assert 0 < total[1] < total[0]


# --- Existing guarantees ---------------------------------------------------


def test_replacing_code_while_parked_disables():
    f = make_pair()
    state = park_compiled(f, 1.5, 2)
    f.__wrapped__.__code__ = (lambda a, b: a - b).__code__
    # The next call, re-probe or not, sees the new code and disables.
    assert path(f, 1.5, 2) == "python"
    assert f(7, 2) == 5
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.CODE_CHANGED
    assert state.parked_reason is None
    assert not state._hot.parked and not state._hot.live
    assert peek(f, quiet=True).findings[0].status == "disabled"


def test_replacing_code_while_parked_during_warmup_compiles_the_new_code():
    add = make_add()
    state = warm(add, 1.5, 2)
    add.__wrapped__.__code__ = (lambda a, b: a - b).__code__
    until_reprobe(add, 1.5, 2)
    assert path(add, 7, 2) == "native"
    assert add(7, 2) == 5
    assert state.compiled_code is add.__wrapped__.__code__


# --- __code__ replaced during a transition --------------------------------
#
# The miss policy's transitions call into the state, which runs Python code
# and takes its lock: another thread can run while a call waits for the
# GIL there, or for the lock. These tests replace ``__code__`` at each such
# point and make a call that starts after the replacement, either on this
# thread (where no lock is held) or on another thread (which then waits for
# the lock). That call must not run the old machine code, and the state
# must end up disabled with ``CODE_CHANGED``, whatever the slot was doing.


def _new_code(f):
    f.__wrapped__.__code__ = (lambda a, b: a - b).__code__


def _disabled_by_the_new_code(f):
    state = state_of(f)
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.CODE_CHANGED
    assert state.parked_reason is None
    assert not state._hot.live and not state._hot.parked
    assert path(f, 7, 2) == "python"
    assert f(7, 2) == 5


class _SlotProxy:
    """Stands in for a state's hot slot, to run ``hook`` while the state
    flips it, with the state's lock held. The wrapper keeps the real slot."""

    def __init__(self, slot, hook, before):
        self._slot, self._hook, self._before = slot, hook, before

    def __getattr__(self, name):
        return getattr(self._slot, name)

    def _flip(self, flip):
        if self._before:
            self._hook()
        flipped = flip()
        if not self._before:
            self._hook()
        return flipped

    def park(self):
        return self._flip(self._slot.park)

    def unpark(self):
        return self._flip(self._slot.unpark)


def _call_from_another_thread(f, args, outcomes):
    """Replace the code, then call ``f`` on another thread, and give it time
    to run until it waits for the state's lock."""
    _new_code(f)

    def call():
        before = attempts(f)
        result = f(*args)
        outcomes.append((result, attempts(f) - before))

    thread = threading.Thread(target=call)
    thread.start()
    thread.join(0.05)
    return thread


def test_code_replaced_before_the_miss_limit_parks(monkeypatch):
    f = make_pair()
    warm(f, 3, 4)
    real = _JitState.repeated_misses
    seen = []

    def interleaved(self):
        _new_code(f)
        # A call that starts here, before the state takes its lock.
        seen.append(path(f, 2, 3))
        real(self)

    monkeypatch.setattr(_JitState, "repeated_misses", interleaved)
    for _ in range(MISS_LIMIT):
        f(1.5, 2)
    assert seen == ["python"]
    _disabled_by_the_new_code(f)


@pytest.mark.parametrize("before", [True, False], ids=["before_flip", "after_flip"])
def test_code_replaced_while_the_slot_parks(before):
    f = make_pair()
    state = warm(f, 3, 4)
    outcomes, threads = [], []
    hook = lambda: threads.append(_call_from_another_thread(f, (2, 3), outcomes))  # noqa: E731
    state._hot = _SlotProxy(state._hot, hook, before)
    for _ in range(MISS_LIMIT):
        f(1.5, 2)
    threads[0].join()
    # The other thread's call ran the new code, without a native attempt.
    assert outcomes == [(-1, 0)]
    state._hot = state._hot._slot
    _disabled_by_the_new_code(f)


def test_code_replaced_before_a_successful_reprobe_unparks(monkeypatch):
    f = make_pair()
    park_compiled(f, 1.5, 2)
    until_reprobe(f, 1.5, 2)
    real = _JitState.reprobe_succeeded
    seen = []

    def interleaved(self):
        _new_code(f)
        seen.append(path(f, 2, 3))
        # The state is disabled by now: unparking must not revive it.
        real(self)

    monkeypatch.setattr(_JitState, "reprobe_succeeded", interleaved)
    # This call started before the replacement, so it may still return the
    # old code's result.
    assert f(2, 3) == 2 * 3 + 2 - 3
    assert seen == ["python"]
    _disabled_by_the_new_code(f)


@pytest.mark.parametrize("before", [True, False], ids=["before_flip", "after_flip"])
def test_code_replaced_while_a_reprobe_unparks(before):
    f = make_pair()
    state = park_compiled(f, 1.5, 2)
    until_reprobe(f, 1.5, 2)
    outcomes, threads = [], []
    hook = lambda: threads.append(_call_from_another_thread(f, (2, 3), outcomes))  # noqa: E731
    state._hot = _SlotProxy(state._hot, hook, before)
    assert f(2, 3) == 2 * 3 + 2 - 3
    threads[0].join()
    assert outcomes == [(-1, 0)]
    state._hot = state._hot._slot
    _disabled_by_the_new_code(f)


def test_code_replaced_while_a_reprobe_compiles(monkeypatch):
    from speedysnek import _compiler

    add = make_add()
    warm(add, 1.5, 2)
    until_reprobe(add, 1.5, 2)
    real = _compiler.compile_i64
    outcomes, threads = [], []

    def compile_i64(*args, **kwargs):
        threads.append(_call_from_another_thread(add, (7, 2), outcomes))
        return real(*args, **kwargs)

    monkeypatch.setattr(_compiler, "compile_i64", compile_i64)
    # The re-probe compiles the old code, notices the replacement, and runs
    # the new code instead.
    assert add(7, 2) == 5
    threads[0].join()
    assert outcomes == [(5, 0)]
    state = state_of(add)
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.CODE_CHANGED
    assert state.specialization is None and not state._hot.live
    assert state.compiled is None


@pytest.mark.parametrize("seed", range(10))
def test_code_replaced_while_threads_park_and_unpark(seed):
    f = make_pair()
    warm(f, 3, 4)
    original = f.__wrapped__.__code__
    stop = threading.Event()
    late, errors = [0] * 4, []

    def worker(index):
        rng = random.Random(seed * 10 + index)
        i = 0
        while not stop.is_set():
            # Runs of misses park the slot; runs of ints re-probe and
            # unpark it.
            phase = (i // (MISS_LIMIT + 8)) % 3
            i += 1
            n = rng.randint(-1000, 1000)
            args = [(n, 3), (n + 0.5, 3), (2**62, 2**62)][phase]
            after = f.__wrapped__.__code__ is not original
            result = f(*args)
            late[index] += after
            new = args[0] - args[1]
            if after and (result != new or type(result) is not type(new)):
                errors.append((args, result))

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in workers:
        thread.start()
    time.sleep(random.Random(seed).uniform(0.005, 0.05))
    _new_code(f)
    deadline = time.monotonic() + 30
    while min(late) < 200:
        assert time.monotonic() < deadline
        time.sleep(0.001)
    stop.set()
    for thread in workers:
        thread.join()
    assert errors == []
    _disabled_by_the_new_code(f)


RANGE_SOURCE = """
def f(n):
    total = 0
    for i in range(n):
        total += i
    return total
"""


def test_a_changed_range_binding_while_parked_disables():
    namespace = {}
    exec(RANGE_SOURCE, namespace)
    f = snekit(namespace["f"])
    state = warm(f, 4)
    # Outside i64: a guard miss every time.
    for _ in range(MISS_LIMIT):
        assert path(f, -(2**63) - 1) == "miss"
    assert state.status is _Status.PARKED
    namespace["range"] = lambda n: [7]
    assert f(-(2**63) - 1) == 7
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.RANGE_CHANGED
    assert f(4) == 7


def test_a_native_error_while_parked_disables():
    class Broken:
        arg_count = 2
        function = None

        def fast_call(self, args):
            raise RuntimeError("native bug")

    f = make_pair()
    state = park_compiled(f, 1.5, 2)
    # Injecting a specialization retires the slot, as for a compiled one,
    # and the generic path then reports the error.
    state.specialization = _Specialization(state.compiled_code, Broken())
    assert f(2, 3) == 5
    assert state.status is _Status.DISABLED
    assert state.disabled_reason is _DisabledReason.NATIVE_ERROR
    assert str(state.native_error) == "native bug"
    assert state.parked_reason is None
    finding = peek(f, quiet=True).findings[0]
    assert (finding.status, finding.reason) == ("disabled", "native_error")
    assert finding.summary == "a native call failed unexpectedly (RuntimeError)"


def test_metadata_and_call_count_survive_parking():
    f = make_pair()
    park_compiled(f, 1.5, 2)
    until_reprobe(f, 1.5, 2)
    f(1, 2)
    assert f.__name__ == "pair"
    assert f.__wrapped__.__name__ == "pair"
    assert f.call_count == _WARMUP_CALLS + 1 + MISS_LIMIT + REPROBE_INTERVAL + 1
    f.call_count = 5
    f(1.5, 2)
    assert f.call_count == 6


MODULE = """
from speedysnek import nosnek

def mean(a, b):
    return a + b

def total(n):
    s = 0
    for i in range(n):
        s += i
    return s

@nosnek
def kept(a):
    return a
"""


def test_bite_chill_and_peek_with_a_parked_function():
    module = types.ModuleType("parked_sample")
    exec(MODULE, module.__dict__)
    original = module.mean
    applied = snek.bite(module, quiet=True)
    assert {f.name for f in applied.bitten} == {"mean", "total"}
    alias = module.mean
    for i in range(_WARMUP_CALLS + 5):
        module.mean(i + 0.5, 1.5)
        module.total(10)
    report = peek(module, quiet=True)
    assert [f.status for f in report] == ["parked", "compiled"]
    assert report.render().endswith("1 compiled, 1 parked, 0 cold, 0 disabled.")
    # ``hunt`` inspects only plain functions, and changes nothing.
    hunted = snek.hunt(module, quiet=True)
    assert [(f.name, f.ready) for f in hunted] == [("kept", False)]
    assert state_of(module.mean).status is _Status.PARKED
    chilled = snek.chill(module, quiet=True)
    assert {f.name for f in chilled.chilled} == {"mean", "total"}
    assert module.mean is original
    # A reference taken before chilling keeps its wrapper and its state.
    assert alias(1.5, 1.5) == 3.0
    assert state_of(alias).status is _Status.PARKED
    assert module.kept(1.5) == 1.5


def test_threads_under_the_gil():
    f = make_pair()
    warm(f, 3, 4)
    f.call_count = 0
    errors = []
    calls_per_thread = 4000

    def worker(seed):
        rng = random.Random(seed)
        for i in range(calls_per_thread):
            # Long runs of floats park it, and ints re-probe it.
            if (i // 300) % 2:
                args = (rng.random(), 3)
            else:
                args = (rng.randint(-1000, 1000), 3)
            if f(*args) != f.__wrapped__(*args):
                errors.append(args)
            state = state_of(f)
            if state.status not in (_Status.COMPILED, _Status.PARKED):
                errors.append(state.status)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert f.call_count == 4 * calls_per_thread
    state = state_of(f)
    assert state._hot.parked == (state.status is _Status.PARKED)
    assert state.compile_attempts == 1
    attempts_, misses = counts(f)
    assert 0 <= misses <= attempts_


# --- Diagnostics -----------------------------------------------------------


def test_rendered_parked_after_warmup():
    add = make_add()
    warm(add, 1.5, 2)
    for _ in range(10):
        add(1.5, 2)
    assert peek(add, quiet=True).render() == textwrap.dedent(
        f"""\
        🐍 Snek peeked at add.

        status:           parked
        calls observed:   19
        warmup calls:     8
        compile attempts: 0
        reason:           calls have not matched the i64 specialization
        detail:           0 of 8 warmup calls fit (one positional, exact 64-bit int per parameter)
        re-probe:         every {REPROBE_INTERVAL} parked calls; next after {REPROBE_INTERVAL - 10} more calls"""  # noqa: E501
    )
    finding = peek(add, quiet=True).findings[0]
    assert finding.reason == "warmup_mismatch"
    assert finding.native_attempts is finding.native_hits is None
    assert finding.native_misses is None
    assert finding.compiler is None
    assert finding.reprobe_remaining == REPROBE_INTERVAL - 10


def test_rendered_parked_after_misses():
    f = make_pair()
    for _ in range(_WARMUP_CALLS + 1):
        f(3, 4)
    for i in range(5):
        f(i, 2)
    for _ in range(MISS_LIMIT):
        f(1.5, 2)
    assert peek(f, quiet=True).render() == textwrap.dedent(
        f"""\
        🐍 Snek peeked at pair.

        status:           parked
        calls observed:   {_WARMUP_CALLS + 1 + 5 + MISS_LIMIT}
        warmup calls:     8
        compile attempts: 1
        native attempts:  {6 + MISS_LIMIT} (6 ran natively, {MISS_LIMIT} fell back to Python)
        compiler:         Cranelift i64 specialization (not used while parked)
        reason:           calls have not matched the i64 specialization
        detail:           {MISS_LIMIT} calls in a row could not run natively
        re-probe:         every {REPROBE_INTERVAL} parked calls; next after {REPROBE_INTERVAL} more calls"""  # noqa: E501
    )
    until_reprobe(f, 1.5, 2)
    finding = peek(f, quiet=True).findings[0]
    assert finding.reprobe_remaining == 0
    assert finding.render().endswith(
        f"re-probe:         every {REPROBE_INTERVAL} parked calls; "
        "the next call that fits re-probes"
    )


def test_rendered_cold_compiled_and_disabled_are_unchanged_but_counted():
    add = make_add()
    assert peek(add, quiet=True).findings[0].render() == textwrap.dedent(
        """\
        status:           cold
        calls observed:   0
        warmup calls:     0 of 8, 8 to go
        compile attempts: 0"""
    )
    warm(add, 1, 2)
    add(1.5, 2)
    assert peek(add, quiet=True).findings[0].render() == textwrap.dedent(
        """\
        status:           compiled
        calls observed:   10
        warmup calls:     8
        compile attempts: 1
        native attempts:  2 (1 ran natively, 1 fell back to Python)
        compiler:         Cranelift i64 specialization"""
    )
    add.__wrapped__.__code__ = (lambda a, b: a - b).__code__
    add(1, 2)
    assert peek(add, quiet=True).findings[0].render() == textwrap.dedent(
        """\
        status:           disabled
        calls observed:   11
        warmup calls:     8
        compile attempts: 1
        native attempts:  2 (1 ran natively, 1 fell back to Python)
        compiler:         Cranelift i64 specialization (no longer used)
        reason:           its __code__ was replaced after compilation"""
    )


def test_native_hits_and_misses_are_exact():
    f = make_pair()
    warm(f, 3, 4)  # The deciding call is the first hit.
    hits, misses = 1, 0
    rng = random.Random(7)
    for _ in range(500):
        if rng.random() < 0.7:
            f(rng.randint(-50, 50), 3)
            hits += 1
        else:
            f(rng.choice([1.5, 2**62, True]), 2**62 // 3)
            misses += 1
    finding = peek(f, quiet=True).findings[0]
    assert finding.status == "compiled"
    assert (finding.native_hits, finding.native_misses) == (hits, misses)
    assert finding.native_attempts == hits + misses
    # Calls that never reach the fast path are not attempts.
    f(1, b=2)
    finding = peek(f, quiet=True).findings[0]
    assert finding.native_attempts == hits + misses


def test_parked_findings_do_not_overclaim():
    f = make_pair()
    park_compiled(f, 1.5, 2)
    finding = peek(f, quiet=True).findings[0]
    assert finding.native_hits == 1  # Only the deciding call ran natively.
    assert finding.native_misses == MISS_LIMIT
    text = finding.render().lower()
    for word in ("success", "faster", "speedup", "error", "disabled"):
        assert word not in text


def test_parked_is_distinguishable_from_disabled():
    parked, disabled = make_add(), make_add()
    warm(parked, 1.5, 2)
    warm(disabled, 1, 2)
    disabled.__wrapped__.__code__ = (lambda a, b: a - b).__code__
    disabled(1, 2)
    report = snek.PeekReport(
        [peek(parked, quiet=True).findings[0], peek(disabled, quiet=True).findings[0]]
    )
    assert [f.status for f in report.parked] == ["parked"]
    assert [f.status for f in report.disabled] == ["disabled"]
    assert report.parked[0].reason == "warmup_mismatch"
    assert report.disabled[0].reason == "code_changed"
    assert repr(report) == "<PeekReport: 1 parked, 1 disabled>"
