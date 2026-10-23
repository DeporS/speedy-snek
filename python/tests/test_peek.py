"""Tests for ``speedysnek.peek``, the read-only JIT diagnostics report."""

import functools
import io
import subprocess
import sys
import textwrap
import types

import pytest

import speedysnek as snek
from speedysnek import PeekFinding, PeekReport, bite, peek, snekit
from speedysnek._decorator import _WARMUP_CALLS, _DisabledReason, _JitState, _Status
from speedysnek._native import NativeWrapper

# Ready and rejected functions, a private function, a class, a callable
# object and an alias, in a deliberately non-alphabetical order. ``CALLS``
# records every call a function body makes.
SAMPLE = textwrap.dedent(
    """\
    CALLS = []

    def sum_squares(n):
        total = 0
        for i in range(n):
            total += i * i
        return total

    def format_report(total):
        return f"total: {total}"

    def clamp(x, lo):
        if x < lo:
            return lo
        return x

    def _private(x):
        return x + 1

    class Calculator:
        def add(self, a, b):
            return a + b

    class Multiplier:
        def __call__(self, x):
            CALLS.append("Multiplier")
            return x * 2

    double = Multiplier()
    total_of_squares = sum_squares
    """
)


def make_module(source=SAMPLE, name="peek_sample"):
    module = types.ModuleType(name)
    exec(compile(source, f"{name}.py", "exec"), module.__dict__)
    return module


def bitten_module(source=SAMPLE, name="peek_sample"):
    module = make_module(source, name)
    bite(module, quiet=True)
    return module


def by_name(report):
    return {finding.name: finding for finding in report}


def only(report):
    (finding,) = report
    return finding


def peek_one(target):
    return only(peek(target, quiet=True))


def jit_snapshot(wrapper):
    """Everything about a wrapper that ``peek`` must leave alone."""
    state = wrapper._jit_state
    return (
        wrapper.call_count,
        state.status,
        state.specialization,
        state.warmup_calls,
        state.compile_attempts,
        state.compiled,
        state.disabled_reason,
        None if state.compiled is None else state.compiled.function.try_calls,
    )


def sum_squares(n):
    total = 0
    for i in range(n):
        total += i * i
    return total


def add(a, b):
    return a + b


def greet(name):
    return f"{name}"


# --- Cold ----------------------------------------------------------------------


def test_new_wrapper_is_cold():
    wrapper = snekit(sum_squares)
    finding = peek_one(wrapper)
    assert finding.name == "sum_squares"
    assert finding.names == ("sum_squares",)
    assert finding.wrapper is wrapper
    assert finding.function is sum_squares
    assert finding.status == "cold"
    assert finding.call_count == 0
    assert finding.warmup_calls == 0
    assert finding.warmup_remaining == _WARMUP_CALLS == 8
    assert finding.compile_attempts == 0
    assert finding.native_attempts is None
    assert finding.compiler is None
    assert finding.reason is finding.detail is finding.lineno is None
    assert finding.rejection is finding.compile_error is finding.native_error is None


def test_cold_after_a_few_calls():
    wrapper = snekit(sum_squares)
    for _ in range(3):
        wrapper(10)
    finding = peek_one(wrapper)
    assert finding.status == "cold"
    assert (finding.call_count, finding.warmup_calls) == (3, 3)
    assert finding.warmup_remaining == 5
    assert finding.compile_attempts == 0
    assert finding.summary == "warming up: 5 more calls before compiling"


def test_fully_warmed_but_not_yet_compiled():
    wrapper = snekit(sum_squares)
    for _ in range(_WARMUP_CALLS):
        wrapper(10)
    finding = peek_one(wrapper)
    assert finding.status == "cold"
    assert finding.warmup_remaining == 0
    assert finding.summary == "warmed up: the next call tries to compile it"
    # Peeking never takes that step.
    assert wrapper._jit_state.compile_attempts == 0
    assert "warmup calls:     8 of 8, the next call compiles" in str(finding)


def test_one_warmup_call_left():
    wrapper = snekit(add)
    for _ in range(_WARMUP_CALLS - 1):
        wrapper(1, 2)
    assert peek_one(wrapper).summary == "warming up: 1 more call before compiling"


# --- Compiled ------------------------------------------------------------------


def test_compiled_after_normal_warmup():
    wrapper = snekit(sum_squares)
    for _ in range(100):
        assert wrapper(10_000) == sum_squares(10_000)
    finding = peek_one(wrapper)
    assert finding.status == "compiled"
    assert finding.call_count == 100
    assert finding.warmup_calls == 8
    assert finding.warmup_remaining is None
    assert finding.compile_attempts == 1
    # The call that compiled it was its first native attempt.
    assert finding.native_attempts == 92
    assert finding.native_attempts == wrapper._jit_state.compiled.function.try_calls
    assert finding.compiler == "Cranelift i64 specialization"
    assert finding.reason is None


@pytest.mark.parametrize(
    "args",
    [
        (2**62, 2**62),  # overflows i64
        (2**64, 1),  # outside i64
        (True, 1),  # not an exact int
        (1.5, 2),  # not an int at all
    ],
)
def test_native_fallbacks_are_attempts_not_results(args):
    wrapper = snekit(add)
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(1, 2)
    before = peek_one(wrapper)
    assert wrapper(*args) == add(*args)
    after = peek_one(wrapper)
    # The attempt is counted, but nothing claims it ran natively.
    assert after.native_attempts == before.native_attempts + 1
    assert after.status == "compiled"
    assert after.call_count == before.call_count + 1
    for name in ("native_calls", "native_results", "native_successes", "speedup"):
        assert not hasattr(after, name)
    text = str(after).lower()
    for word in ("success", "faster", "speedup", "x faster"):
        assert word not in text


def test_a_call_with_another_arity_is_no_native_attempt():
    def with_default(a, b=1):
        return a + b

    wrapper = snekit(with_default)
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(1, 2)
    before = peek_one(wrapper).native_attempts
    assert wrapper(5) == 6
    assert peek_one(wrapper).native_attempts == before


def test_the_hot_slot_changes_nothing_peek_reports():
    def without_hot_slot(func):
        # ``snekit`` before the hot slot: every call takes the generic path.
        state = _JitState(func)
        wrapper = NativeWrapper(func, state, _Status.COLD)
        functools.update_wrapper(wrapper, func)
        wrapper._jit_state = state
        return wrapper

    def with_default(a, b=1):
        return a * b - a

    fields = [name for name in PeekFinding.__slots__ if name != "wrapper"]
    hot, generic = snekit(with_default), without_hot_slot(with_default)
    calls = [(1, 2)] * _WARMUP_CALLS + [(1, 2), (3, 4), (5,), (1.5, 2), (2**62, 4)]
    for args in calls + [(7, 8)] * 5:
        assert hot(*args) == generic(*args)
        found = peek_one(hot), peek_one(generic)
        assert [getattr(found[0], name) for name in fields] == [
            getattr(found[1], name) for name in fields
        ]
    assert hot._jit_state._hot.live
    assert peek_one(hot).native_attempts == 9


# --- Disabled ------------------------------------------------------------------


def test_frontend_rejection():
    # Called with an int, so that warmup does not park it.
    wrapper = snekit(greet)
    for _ in range(5):
        wrapper(1)
    finding = peek_one(wrapper)
    # Still warming up: nothing has been tried yet.
    assert finding.status == "cold"
    for _ in range(_WARMUP_CALLS):
        wrapper(1)
    finding = peek_one(wrapper)
    state = wrapper._jit_state
    assert finding.status == "disabled"
    assert finding.reason == "rejected"
    assert finding.summary == "f-strings are not supported"
    assert finding.rejection is state.rejection
    assert finding.detail == state.rejection.message
    assert finding.lineno == state.rejection.lineno == greet.__code__.co_firstlineno + 1
    assert finding.compile_attempts == 1
    assert finding.native_attempts is None
    assert finding.compiler is None
    assert finding.call_count == 13
    assert finding.warmup_remaining is None


def test_backend_limit():
    def huge(a):
        return a + 10**30  # Folded into one constant outside i64.

    wrapper = snekit(huge)
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(1)
    finding = peek_one(wrapper)
    assert finding.status == "disabled"
    assert finding.reason == "unsupported"
    assert finding.summary == "beyond the native compiler's limits"
    assert finding.compile_error is wrapper._jit_state.compile_error
    assert "does not fit in a signed 64-bit integer" in finding.detail
    assert finding.rejection is None


def test_unexpected_compiler_error(monkeypatch):
    from speedysnek import _compiler

    def broken(*args, **kwargs):
        raise RuntimeError("compiler bug")

    monkeypatch.setattr(_compiler, "compile_i64", broken)
    wrapper = snekit(add)
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(1, 2)
    finding = peek_one(wrapper)
    assert finding.reason == "compiler_error"
    assert finding.summary == "the compiler failed unexpectedly (RuntimeError)"
    assert finding.detail == "compiler bug"
    assert type(finding.compile_error) is RuntimeError


def test_unexpected_native_error():
    wrapper = snekit(add)
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(1, 2)
    error = RuntimeError("native bug")
    wrapper._jit_state.native_failed(error)
    finding = peek_one(wrapper)
    assert finding.status == "disabled"
    assert finding.reason == "native_error"
    assert finding.summary == "a native call failed unexpectedly (RuntimeError)"
    assert finding.native_error is error
    assert finding.detail == "native bug"
    assert finding.compiler == "Cranelift i64 specialization"


def test_interrupted_compilation(monkeypatch):
    from speedysnek import _frontend

    def interrupted(func):
        raise KeyboardInterrupt

    monkeypatch.setattr(_frontend, "analyze", interrupted)
    wrapper = snekit(add)
    for _ in range(_WARMUP_CALLS):
        wrapper(1, 2)
    with pytest.raises(KeyboardInterrupt):
        wrapper(1, 2)
    finding = peek_one(wrapper)
    assert finding.status == "disabled"
    assert finding.reason == "interrupted"
    assert finding.summary == "compilation was interrupted (for example by Ctrl+C)"
    assert finding.detail is finding.compile_error is None


def test_cold_wrapper_never_reports_compiled_code():
    # Mid-compilation in another thread, ``compiled`` is set while the
    # status is still ``COLD``.
    donor = snekit(add)
    for _ in range(_WARMUP_CALLS + 1):
        donor(1, 2)
    wrapper = snekit(add)
    wrapper._jit_state.compiled = donor._jit_state.compiled
    finding = peek_one(wrapper)
    assert finding.status == "cold"
    assert finding.compiler is finding.native_attempts is None


def test_error_messages_are_read_without_str():
    class Loud(Exception):
        def __str__(self):
            raise AssertionError("peek must not call __str__")

    wrapper = snekit(add)
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(1, 2)
    wrapper._jit_state.native_failed(Loud("quiet", "please"))
    finding = peek_one(wrapper)
    assert finding.detail == "Loud"
    assert finding.summary == "a native call failed unexpectedly (Loud)"
    str(PeekReport([finding]))


def test_replaced_code_invalidates():
    def triple(n):
        return n * 3

    def quadruple(n):
        return n * 4

    wrapper = snekit(triple)
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(2)
    attempts = peek_one(wrapper).native_attempts
    triple.__code__ = quadruple.__code__
    assert wrapper(2) == 8
    finding = peek_one(wrapper)
    assert finding.status == "disabled"
    assert finding.reason == "code_changed"
    assert finding.summary == "its __code__ was replaced after compilation"
    assert finding.detail is None
    # The stale code was never entered again.
    assert finding.native_attempts == attempts
    assert finding.compiler == "Cranelift i64 specialization"
    text = str(finding)
    assert "compiler:         Cranelift i64 specialization (no longer used)" in text


RANGE_SOURCE = """
def f(n):
    total = 0
    for i in range(n):
        total += i
    return total
"""


def test_changed_range_binding_invalidates():
    namespace = {}
    exec(RANGE_SOURCE, namespace)
    wrapper = snekit(namespace["f"])
    for _ in range(_WARMUP_CALLS + 1):
        assert wrapper(4) == 6
    assert peek_one(wrapper).status == "compiled"
    namespace["range"] = lambda n: range(2)
    assert wrapper(10) == 1
    finding = peek_one(wrapper)
    assert finding.status == "disabled"
    assert finding.reason == "range_changed"
    assert finding.summary == "its range binding changed after compilation"


def test_unguardable_range_binding():
    namespace = {"range": lambda n: range(2)}
    exec(RANGE_SOURCE, namespace)
    wrapper = snekit(namespace["f"])
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(10)
    finding = peek_one(wrapper)
    assert finding.status == "disabled"
    assert finding.reason == "range_unguarded"
    assert finding.summary == "range could not be proven to be the builtin"
    assert "not the builtin range" in finding.detail
    assert finding.compile_error is wrapper._jit_state.compile_error
    assert finding.compiler is None


# --- Wrappers from bite() and plain functions ------------------------------------


def test_wrappers_installed_by_bite():
    module = bitten_module()
    finding = peek_one(module.sum_squares)
    assert finding.status == "cold"
    assert finding.wrapper is module.sum_squares
    assert finding.function is module.sum_squares._jit_state._func
    assert finding.call_count == 0
    for _ in range(20):
        module.sum_squares(100)
    finding = peek_one(module.total_of_squares)
    assert finding.status == "compiled"
    assert finding.call_count == 20
    assert finding.native_attempts == 12


def test_plain_function_is_not_bitten():
    finding = peek_one(add)
    assert finding.status == "not_bitten"
    assert finding.name == "add"
    assert finding.function is add
    assert finding.wrapper is None
    assert finding.summary == (
        "not wrapped with @snekit; use snek.bite(module) or @snekit first"
    )
    for field in (
        "call_count",
        "warmup_calls",
        "warmup_remaining",
        "compile_attempts",
        "native_attempts",
        "compiler",
        "reason",
        "detail",
    ):
        assert getattr(finding, field) is None


def test_a_reference_taken_before_bite_points_to_the_wrapper(monkeypatch):
    module = make_module()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original = module.sum_squares
    bite(module, quiet=True)
    finding = peek_one(original)
    assert finding.status == "not_bitten"
    assert finding.summary == (
        "not wrapped itself, but peek_sample.sum_squares is its @snekit wrapper"
    )


def test_bound_methods():
    class Box:
        @snekit
        def add(self, a):
            return a + 1

        def plain(self, a):
            return a

    box = Box()
    finding = peek_one(box.add)
    assert finding.status == "cold"
    assert finding.wrapper is Box.__dict__["add"]
    assert peek_one(box.plain).status == "not_bitten"


def test_wrapper_of_a_non_function():
    wrapper = snekit(abs)
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(-3)
    finding = peek_one(wrapper)
    assert finding.name == "abs"
    assert finding.function is abs
    assert finding.status == "disabled"
    assert finding.reason == "rejected"


# --- Modules ---------------------------------------------------------------------


def test_module_discovery_order_and_aliases():
    source = SAMPLE + textwrap.dedent(
        """\
        from speedysnek import snekit

        @snekit
        def adder(a, b):
            return a + b

        also_clamp = clamp
        """
    )
    module = make_module(source)
    bite(module, quiet=True)
    report = peek(module, quiet=True)
    assert report.module_name == "peek_sample"
    assert [finding.name for finding in report] == ["sum_squares", "clamp", "adder"]
    findings = by_name(report)
    assert findings["sum_squares"].names == ("sum_squares", "total_of_squares")
    assert findings["clamp"].names == ("clamp", "also_clamp")
    assert findings["sum_squares"].wrapper is module.total_of_squares
    assert findings["adder"].names == ("adder",)
    assert all(finding.status == "cold" for finding in report)


def test_two_wrappers_of_one_function_are_two_findings():
    module = make_module("from speedysnek import snekit\n\ndef f(a):\n    return a\n")
    module.g = snekit(module.f)
    module.h = snekit(module.f)
    report = peek(module, quiet=True)
    assert [finding.names for finding in report] == [("g",), ("h",)]


def test_module_without_wrappers():
    module = make_module()
    report = peek(module, quiet=True)
    assert len(report) == 0
    assert report.findings == ()
    assert report.render() == "🐍 Snek found no bitten functions in peek_sample."


def test_private_imported_and_other_objects_are_left_out():
    source = bitten_module()
    module = make_module(
        "from speedysnek import snekit\n\n"
        "def own(x):\n    return x + 1\n\n"
        "_hidden = snekit(own)\n",
        name="peek_importer",
    )
    module.sum_squares = source.sum_squares  # imported wrapper
    module.wrapped_builtin = snekit(abs)  # not one of its own functions
    module.not_a_wrapper = types.SimpleNamespace(_jit_state=None)
    report = peek(module, quiet=True)
    assert len(report) == 0
    # The same wrapper is reported in its own module.
    assert [finding.name for finding in peek(source, quiet=True)] == [
        "sum_squares",
        "clamp",
    ]


def test_private_alias_is_not_listed():
    module = bitten_module()
    module._secret = module.clamp
    finding = by_name(peek(module, quiet=True))["clamp"]
    assert finding.names == ("clamp",)


def test_module_hooks_are_not_used():
    calls = []

    class Hooked(types.ModuleType):
        def __getattr__(self, name):
            calls.append(name)
            raise AttributeError(name)

        def __getattribute__(self, name):
            calls.append(name)
            return super().__getattribute__(name)

    module = Hooked("peek_hooked")
    exec(
        "def f(a):\n    return a\n",
        types.ModuleType.__dict__["__dict__"].__get__(module),
    )
    bite(module, quiet=True)
    calls.clear()
    report = peek(module, quiet=True)
    assert [finding.name for finding in report] == ["f"]
    assert calls == []


# --- Read-only -------------------------------------------------------------------


def test_peek_changes_nothing():
    module = bitten_module()
    wrappers = [module.sum_squares, module.clamp]
    for _ in range(_WARMUP_CALLS):
        module.sum_squares(5)
    for _ in range(_WARMUP_CALLS + 4):
        module.clamp(1, 2)
    namespace = dict(module.__dict__)
    before = [jit_snapshot(wrapper) for wrapper in wrappers]
    for _ in range(3):
        peek(module, quiet=True)
        for wrapper in wrappers:
            peek(wrapper, quiet=True)
            str(peek(wrapper, quiet=True))
    assert [jit_snapshot(wrapper) for wrapper in wrappers] == before
    assert dict(module.__dict__) == namespace
    assert all(namespace[key] is value for key, value in module.__dict__.items())
    assert module.CALLS == []
    # The fully warmed wrapper is still waiting for its next call.
    assert module.sum_squares._jit_state.status is _Status.COLD
    assert module.sum_squares._jit_state.compile_attempts == 0


def test_peek_never_calls_the_function():
    calls = []

    def recorded(a):
        calls.append(a)
        return a

    wrapper = snekit(recorded)
    peek(wrapper, quiet=True)
    peek(recorded, quiet=True)
    assert calls == []
    assert wrapper.call_count == 0


def test_call_count_is_read_as_is():
    wrapper = snekit(add)
    wrapper.call_count = 10**30
    finding = peek_one(wrapper)
    # The documented, user-assignable count; the JIT state is separate.
    assert finding.call_count == 10**30
    assert finding.warmup_calls == 0


def run_isolated(tmp_path, source, code):
    (tmp_path / "peek_sample.py").write_text(source)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    return result.stdout.strip()


LOADED = """\
    loaded = [
        name
        for name in (
            "speedysnek._native", "speedysnek._compiler", "speedysnek._frontend"
        )
        if name in sys.modules
    ]
"""


def test_peeking_at_plain_code_loads_nothing(tmp_path):
    code = (
        "import sys\nimport speedysnek as snek\nimport peek_sample\n"
        "a = snek.peek(peek_sample, quiet=True)\n"
        "b = snek.peek(peek_sample.sum_squares, quiet=True)\n"
        + textwrap.dedent(LOADED)
        + "print(len(a), b.findings[0].status, loaded)\n"
    )
    assert run_isolated(tmp_path, SAMPLE, code) == "0 not_bitten []"


def test_peeking_at_a_cold_wrapper_loads_no_compiler(tmp_path):
    code = (
        "import sys\nimport speedysnek as snek\nimport peek_sample\n"
        "snek.bite(peek_sample, quiet=True)\n"
        "del sys.modules['speedysnek._frontend']\n"
        "report = snek.peek(peek_sample, quiet=True)\n"
        + textwrap.dedent(LOADED)
        + "print(len(report), loaded)\n"
    )
    assert run_isolated(tmp_path, SAMPLE, code) == "2 ['speedysnek._native']"


# --- Forged objects ----------------------------------------------------------------


def test_lookalike_objects_are_rejected():
    real = snekit(add)

    class Lookalike:
        def __call__(self, *args):
            return add(*args)

    fake = Lookalike()
    fake._jit_state = real._jit_state
    fake.__wrapped__ = add
    fake.call_count = 5
    with pytest.raises(TypeError, match="got Lookalike"):
        peek(fake, quiet=True)


def test_native_wrapper_without_a_jit_state_is_rejected():
    forged = NativeWrapper(add, object(), _Status.COLD)
    forged._jit_state = snekit(add)._jit_state
    with pytest.raises(TypeError, match="unrecognized"):
        peek(forged, quiet=True)
    module = make_module("X = 1\n")
    module.forged = forged
    assert len(peek(module, quiet=True)) == 0


def test_rewritten_wrapper_attributes_are_not_trusted():
    def other(a, b):
        return a - b

    wrapper = snekit(add)
    decoy = snekit(other)
    for _ in range(_WARMUP_CALLS + 1):
        decoy(1, 2)
    wrapper.__wrapped__ = other
    wrapper._jit_state = decoy._jit_state
    finding = peek_one(wrapper)
    # Reported from the state the wrapper really uses.
    assert finding.function is add
    assert finding.status == "cold"
    assert finding.compile_attempts == 0


def test_function_with_a_forged_wrapped_attribute_is_not_bitten():
    def plain(a):
        return a

    plain.__wrapped__ = snekit(add)
    assert peek_one(plain).status == "not_bitten"


# --- Output ------------------------------------------------------------------------


def test_rendered_compiled_finding():
    wrapper = snekit(sum_squares)
    for _ in range(100):
        wrapper(10_000)
    assert peek(wrapper, quiet=True).render() == textwrap.dedent(
        """\
        🐍 Snek peeked at sum_squares.

        status:           compiled
        calls observed:   100
        warmup calls:     8
        compile attempts: 1
        native attempts:  92 (92 ran natively, 0 fell back to Python)
        compiler:         Cranelift i64 specialization"""
    )


def test_rendered_disabled_and_not_bitten_findings():
    wrapper = snekit(greet)
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(1)
    line = greet.__code__.co_firstlineno + 1
    assert peek(wrapper, quiet=True).render() == textwrap.dedent(
        f"""\
        🐍 Snek peeked at greet.

        status:           disabled
        calls observed:   9
        warmup calls:     8
        compile attempts: 1
        reason:           f-strings are not supported (line {line})"""
    )
    assert peek(add, quiet=True).render() == (
        "🐍 Snek peeked at add.\n\n"
        "status:           not bitten\n"
        "note:             not wrapped with @snekit; use snek.bite(module) or "
        "@snekit first"
    )


def test_rendered_module_report():
    source = SAMPLE + textwrap.dedent(
        """\
        from speedysnek import snekit

        @snekit
        def shout(name):
            return f"{name}"
        """
    )
    module = make_module(source)
    bite(module, quiet=True)
    for _ in range(10):
        module.sum_squares(10)
        module.shout("hi")
    module.clamp(1, 2)
    fits = "0 of 8 warmup calls fit (one positional, exact 64-bit int per parameter)"
    expected = textwrap.dedent(
        f"""\
        🐍 Snek peeked at peek_sample.

        sum_squares
          status:           compiled
          aliases:          total_of_squares
          calls observed:   10
          warmup calls:     8
          compile attempts: 1
          native attempts:  2 (2 ran natively, 0 fell back to Python)
          compiler:         Cranelift i64 specialization

        clamp
          status:           cold
          calls observed:   1
          warmup calls:     1 of 8, 7 to go
          compile attempts: 0

        shout
          status:           parked
          calls observed:   10
          warmup calls:     8
          compile attempts: 0
          reason:           calls have not matched the i64 specialization
          detail:           {fits}
          re-probe:         every 64 parked calls; next after 63 more calls

        1 compiled, 1 parked, 1 cold, 0 disabled."""
    )
    report = peek(module, quiet=True)
    assert report.render() == expected
    assert str(report) == expected
    counts = [len(report.compiled), len(report.parked), len(report.cold)]
    assert counts == [1, 1, 1]
    assert report.disabled == report.not_bitten == ()
    assert repr(report) == "<PeekReport: 1 compiled, 1 parked, 1 cold>"


def test_finding_repr_and_str():
    finding = peek_one(snekit(add))
    assert repr(finding) == (
        "PeekFinding(name='add', status='cold', "
        "summary='warming up: 8 more calls before compiling')"
    )
    assert str(finding) == finding.render()
    assert str(finding).startswith("status:           cold\n")
    assert repr(PeekReport([])) == "<PeekReport: empty>"
    assert PeekReport([]).render() == "🐍 Snek found no bitten functions."


def test_prints_the_report_by_default(capsys):
    report = peek(add)
    assert capsys.readouterr().out == report.render() + "\n"


def test_quiet_prints_nothing(capsys):
    peek(add, quiet=True)
    peek(make_module(), quiet=True)
    assert capsys.readouterr().out == ""


def test_stream(capsys):
    stream = io.StringIO()
    report = peek(snekit(add), stream=stream)
    assert stream.getvalue() == report.render() + "\n"
    assert capsys.readouterr().out == ""


def test_stream_that_cannot_encode_the_snek():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", newline="\n")
    peek(add, stream=stream)
    stream.flush()
    assert raw.getvalue().decode("ascii").startswith("Snek peeked at add.\n")


# --- Invalid input -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("arg", "message"),
    [
        (None, "got NoneType"),
        (42, "got int"),
        ([add], "got list"),
        (abs, "got builtin_function_or_method"),
        (int, "got type"),
        (types.SimpleNamespace(), "got SimpleNamespace"),
        ("my_module", "got str; to peek at a module, import it"),
    ],
)
def test_invalid_input_raises_type_error(arg, message):
    with pytest.raises(TypeError, match=f"^peek\\(\\) expected .*{message}"):
        peek(arg, quiet=True)


def test_public_api():
    assert snek.peek is peek
    for name in ("peek", "PeekReport", "PeekFinding"):
        assert name in snek.__all__
    assert snek.PeekReport is PeekReport
    assert snek.PeekFinding is PeekFinding
    assert _DisabledReason.REJECTED.value == "rejected"
