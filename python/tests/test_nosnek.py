"""Tests for ``speedysnek.nosnek``, the opt-out marker."""

import functools
import gc
import inspect
import subprocess
import sys
import textwrap
import types
import weakref

import pytest

import speedysnek as snek
from speedysnek import _bite, bite, hunt, nosnek, peek, snekit
from speedysnek._decorator import _WARMUP_CALLS, _Status
from speedysnek._native import NativeWrapper

# Marked and unmarked ready functions, a marked rejected function, a marked
# private function and an alias, in a deliberately non-alphabetical order.
# ``CALLS`` records every call a function body makes.
SAMPLE = textwrap.dedent(
    """\
    from speedysnek import nosnek

    CALLS = []

    def sum_squares(n):
        total = 0
        for i in range(n):
            total += i * i
        return total

    @nosnek
    def update_database(n):
        CALLS.append("update_database")
        return n + 1

    @nosnek
    def format_report(total):
        CALLS.append("format_report")
        return f"total: {total}"

    def clamp(x, lo):
        if x < lo:
            return lo
        return x

    @nosnek
    def _private_step(x):
        return x + 1

    database_update = update_database
    """
)


def make_module(source=SAMPLE, name="nosnek_sample"):
    module = types.ModuleType(name)
    exec(compile(source, f"{name}.py", "exec"), module.__dict__)
    return module


def by_name(report):
    return {finding.name: finding for finding in report}


def add_one(value):
    """Add one."""
    return value + 1


# --- The marker itself -------------------------------------------------------


def test_returns_the_same_function():
    def f(x):
        return x

    assert nosnek(f) is f
    assert snek.nosnek is nosnek


def test_marked_function_is_still_a_plain_python_function():
    @nosnek
    def sensitive_logic(value):
        return value + 1

    assert inspect.isfunction(sensitive_logic)
    assert type(sensitive_logic) is types.FunctionType
    assert sensitive_logic(41) == 42
    with pytest.raises(TypeError):
        sensitive_logic("x")


def test_metadata_signature_and_code_are_unchanged():
    def f(a, b=2, *, c=3):
        """Docs."""
        return a + b + c

    f.custom = "kept"
    fields = ("__code__", "__defaults__", "__kwdefaults__", "__globals__")
    before = {name: getattr(f, name) for name in fields}
    signature = inspect.signature(f)
    nosnek(f)
    for name, value in before.items():
        assert getattr(f, name) is value
    assert inspect.signature(f) == signature
    assert (f.__name__, f.__qualname__, f.__doc__, f.__module__) == (
        "f",
        "test_metadata_signature_and_code_are_unchanged.<locals>.f",
        "Docs.",
        __name__,
    )
    # Nothing is added to the function's own attributes.
    assert f.__dict__ == {"custom": "kept"}
    assert f(1) == 6


def test_marking_twice_is_harmless():
    def f(x):
        return x

    assert nosnek(nosnek(f)) is f
    assert f.__dict__ == {}
    finding = hunt(f, quiet=True).findings[0]
    assert finding.reason == "opted_out"


def test_methods_and_lambdas_can_be_marked():
    class Box:
        @nosnek
        def plain(self, x):
            return x

        @staticmethod
        @nosnek
        def static(x):
            return x * 2

    square = nosnek(lambda x: x * x)
    assert Box().plain(3) == 3
    assert Box.static(3) == 6
    assert square(4) == 16


class Callable:
    def __call__(self, x):
        return x


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (int, "got the class int"),
        (Callable, "got the class Callable"),
        (len, "got the built-in len"),
        (Callable(), "got Callable"),
        (Callable().__call__, "got a bound method"),
        (staticmethod(add_one), "got staticmethod"),
        (functools.partial(add_one), "got partial"),
        (None, "got NoneType"),
        ("add_one", "got str"),
    ],
)
def test_non_functions_raise_type_error(value, message):
    with pytest.raises(
        TypeError, match="^nosnek\\(\\) expected a Python function"
    ) as e:
        nosnek(value)
    assert message in str(e.value)


def test_a_snekit_wrapper_is_rejected_with_a_hint():
    def f(x):
        return x

    with pytest.raises(TypeError, match="cannot undo @snekit"):
        nosnek(snekit(f))
    # The function underneath was not marked either.
    assert hunt(f, quiet=True).findings[0].status == "ready"


def test_the_mark_is_not_copied_by_functools_wraps():
    @nosnek
    def inner(x):
        return x

    @functools.wraps(inner)
    def outer(x):
        return inner(x)

    assert hunt(inner, quiet=True).findings[0].reason == "opted_out"
    assert hunt(outer, quiet=True).findings[0].reason != "opted_out"


def test_marking_does_not_keep_a_function_alive():
    def f(x):
        return x

    ref = weakref.ref(nosnek(f))
    del f
    gc.collect()
    assert ref() is None


def test_explicit_snekit_still_wraps_a_marked_function():
    @nosnek
    def f(x):
        return x * 2

    wrapper = snekit(f)
    assert type(wrapper) is NativeWrapper
    for _ in range(_WARMUP_CALLS + 1):
        assert wrapper(3) == 6
    assert wrapper._jit_state.status is _Status.COMPILED


def test_marking_after_wrapping_leaves_the_wrapper_alone():
    def f(x):
        return x * 2

    wrapper = snekit(f)
    nosnek(f)
    for _ in range(_WARMUP_CALLS + 1):
        assert wrapper(3) == 6
    assert wrapper._jit_state.status is _Status.COMPILED


# --- hunt --------------------------------------------------------------------


def test_hunt_reports_marked_functions_in_definition_order():
    report = hunt(make_module(), quiet=True)
    assert [(f.name, f.status, f.reason) for f in report] == [
        ("sum_squares", "ready", None),
        ("update_database", "skipped", "opted_out"),
        ("format_report", "skipped", "opted_out"),
        ("clamp", "ready", None),
    ]
    finding = by_name(report)["update_database"]
    assert finding.summary == "marked with @nosnek"
    assert finding.lineno is None
    assert finding.detail == (
        "marked with @nosnek, so SpeedySnek leaves it as plain Python"
    )
    assert report.ready_count == 2
    assert report.skipped_count == 2


def test_hunt_does_not_analyze_marked_functions(monkeypatch):
    from speedysnek import _frontend

    analyzed = []
    original = _frontend.analyze

    def spy(function):
        analyzed.append(function.__name__)
        return original(function)

    monkeypatch.setattr(_frontend, "analyze", spy)
    module = make_module()
    hunt(module, quiet=True)
    assert analyzed == ["sum_squares", "clamp"]
    analyzed.clear()
    finding = hunt(module.update_database, quiet=True).findings[0]
    assert analyzed == []
    assert finding.status == "skipped"
    assert finding.reason == "opted_out"
    assert finding.function is module.update_database


def test_hunt_of_a_marked_function_among_others():
    module = make_module()
    report = hunt(module.clamp, module.format_report, quiet=True)
    assert [(f.name, f.status, f.reason) for f in report] == [
        ("clamp", "ready", None),
        ("format_report", "skipped", "opted_out"),
    ]


def test_aliases_and_private_names_follow_the_discovery_rules():
    module = make_module()
    report = hunt(module, quiet=True)
    names = [finding.name for finding in report]
    # An alias is reported once, under the first name; private names are
    # left out of module reports as usual.
    assert names.count("update_database") == 1
    assert "database_update" not in names
    assert "_private_step" not in names
    # Explicitly, both behave like the function they are.
    assert hunt(module.database_update, quiet=True).findings[0].reason == "opted_out"
    assert hunt(module._private_step, quiet=True).findings[0].reason == "opted_out"


def test_rendered_hunt_report():
    module = make_module()
    assert hunt(module, quiet=True).render() == (
        "🐍 Snek went hunting.\n"
        "\n"
        "sum_squares      ready    integer arithmetic and one range loop\n"
        "update_database  skipped  marked with @nosnek\n"
        "format_report    skipped  marked with @nosnek\n"
        "clamp            ready    integer arithmetic and if/else branches\n"
        "\n"
        "2 ready, 2 skipped."
    )


# --- bite --------------------------------------------------------------------


def test_bite_leaves_marked_bindings_untouched_and_bites_the_rest():
    module = make_module()
    before = dict(module.__dict__)
    report = bite(module, quiet=True)

    for name in ("update_database", "database_update", "format_report"):
        assert module.__dict__[name] is before[name]
    assert module._private_step is before["_private_step"]
    assert type(module.sum_squares) is NativeWrapper
    assert type(module.clamp) is NativeWrapper
    assert module.sum_squares.__wrapped__ is before["sum_squares"]

    assert module.update_database(1) == 2
    assert module.database_update(1) == 2
    assert module.CALLS == ["update_database", "update_database"]
    assert report.bitten_count == 2
    assert report.skipped_count == 2


def test_bite_reports_the_opt_out():
    module = make_module()
    original = module.update_database
    finding = by_name(bite(module, quiet=True))["update_database"]
    assert finding.status == "skipped"
    assert not finding.bitten
    assert finding.wrapper is None
    assert finding.function is original
    assert finding.names == ("update_database", "database_update")
    assert finding.summary == "marked with @nosnek"
    assert finding.hunt.status == "skipped"
    assert finding.hunt.reason == "opted_out"
    assert finding.hunt.function is original


def test_rendered_bite_report():
    module = make_module()
    assert bite(module, quiet=True).render() == (
        "🐍 Snek took a bite.\n"
        "\n"
        "sum_squares      bitten   wrapped with @snekit\n"
        "update_database  skipped  marked with @nosnek\n"
        "format_report    skipped  marked with @nosnek\n"
        "clamp            bitten   wrapped with @snekit\n"
        "\n"
        "2 functions bitten, 2 skipped, 0 already bitten."
    )


def test_bite_never_wraps_or_analyzes_marked_functions(monkeypatch):
    from speedysnek import _frontend

    wrapped, analyzed = [], []
    original = _frontend.analyze

    def spy(function):
        analyzed.append(function.__name__)
        return original(function)

    def wrap(function):
        wrapped.append(function.__name__)
        return snekit(function)

    monkeypatch.setattr(_frontend, "analyze", spy)
    monkeypatch.setattr(_bite, "_wrap", wrap)
    bite(make_module(), quiet=True)
    assert wrapped == ["sum_squares", "clamp"]
    assert analyzed == ["sum_squares", "clamp"]


def test_second_bite_still_skips_marked_functions():
    module = make_module()
    bite(module, quiet=True)
    before = dict(module.__dict__)
    report = bite(module, quiet=True)
    assert module.__dict__ == before
    assert [(f.name, f.status) for f in report] == [
        ("sum_squares", "already_bitten"),
        ("update_database", "skipped"),
        ("format_report", "skipped"),
        ("clamp", "already_bitten"),
    ]


def test_a_changed_marked_binding_does_not_block_the_others(monkeypatch):
    module = make_module()

    def wrap_and_rebind(function):
        # Marked functions are never checked or replaced, so a change to
        # one of their names cannot stop the module update.
        module.database_update = add_one
        return snekit(function)

    monkeypatch.setattr(_bite, "_wrap", wrap_and_rebind)
    report = bite(module, quiet=True)
    assert report.bitten_count == 2
    assert type(module.sum_squares) is NativeWrapper
    assert module.database_update is add_one


def test_marking_after_bite_keeps_the_existing_wrapper():
    module = make_module()
    original = module.sum_squares
    bite(module, quiet=True)
    wrapper = module.sum_squares
    nosnek(original)
    for _ in range(_WARMUP_CALLS + 1):
        assert wrapper(4) == 14
    finding = by_name(bite(module, quiet=True))["sum_squares"]
    assert module.sum_squares is wrapper
    assert wrapper._jit_state.status is _Status.COMPILED
    assert finding.status == "skipped"
    assert finding.hunt.reason == "opted_out"
    assert finding.summary == (
        "marked with @nosnek; its existing @snekit wrapper sum_squares is left as it is"
    )


def test_a_marked_function_with_a_wrapper_keeps_its_plain_aliases():
    source = textwrap.dedent(
        """\
        from speedysnek import nosnek, snekit

        @nosnek
        def triple(n):
            return n * 3

        fast_triple = snekit(triple)
        """
    )
    module = make_module(source)
    plain, wrapper = module.triple, module.fast_triple
    finding = bite(module, quiet=True).findings[0]
    # Not pointed at the wrapper, as an unmarked function would be.
    assert module.triple is plain
    assert module.fast_triple is wrapper
    assert finding.status == "skipped"
    assert finding.names == ("triple", "fast_triple")
    assert finding.summary == (
        "marked with @nosnek; its existing @snekit wrapper fast_triple is left as it is"
    )


def test_unmarked_functions_behave_as_before():
    source = SAMPLE.replace("@nosnek\n", "")
    module = make_module(source)
    report = bite(module, quiet=True)
    assert [(f.name, f.status) for f in report] == [
        ("sum_squares", "bitten"),
        ("update_database", "skipped"),
        ("format_report", "skipped"),
        ("clamp", "bitten"),
    ]
    assert by_name(report)["update_database"].hunt.reason == "unsupported_opcode"


# --- peek --------------------------------------------------------------------


def test_peek_at_a_marked_function():
    module = make_module()
    finding = peek(module.update_database, quiet=True).findings[0]
    assert finding.status == "not_bitten"
    assert finding.reason == "opted_out"
    assert finding.function is module.update_database
    assert finding.wrapper is None
    assert finding.summary == (
        "marked with @nosnek: SpeedySnek will not wrap it automatically"
    )
    assert "disabled" not in finding.summary
    assert peek(module.update_database, quiet=True).render() == (
        "🐍 Snek peeked at update_database.\n\n"
        "status:           not bitten\n"
        "note:             marked with @nosnek: SpeedySnek will not wrap it "
        "automatically"
    )


def test_peek_at_an_unmarked_plain_function_is_unchanged():
    finding = peek(add_one, quiet=True).findings[0]
    assert finding.status == "not_bitten"
    assert finding.reason is None
    assert finding.summary == (
        "not wrapped with @snekit; use snek.bite(module) or @snekit first"
    )


def test_peek_at_a_function_marked_after_bite(monkeypatch):
    module = make_module()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original = module.sum_squares
    bite(module, quiet=True)
    nosnek(original)
    finding = peek(original, quiet=True).findings[0]
    assert finding.status == "not_bitten"
    assert finding.reason == "opted_out"
    assert finding.summary == (
        "marked with @nosnek, but nosnek_sample.sum_squares is its @snekit wrapper"
    )


def test_peek_module_lists_only_wrappers():
    module = make_module()
    assert len(peek(module, quiet=True)) == 0
    bite(module, quiet=True)
    report = peek(module, quiet=True)
    assert [finding.name for finding in report] == ["sum_squares", "clamp"]
    assert report.not_bitten == ()


# --- Lazy loading ------------------------------------------------------------


def run_isolated(tmp_path, source, code):
    (tmp_path / "nosnek_sample.py").write_text(source)
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


def test_importing_and_applying_nosnek_loads_nothing(tmp_path):
    code = (
        "import sys\nfrom speedysnek import nosnek\n"
        "@nosnek\ndef f(x):\n    return x + 1\n" + LOADED + "print(f(1), loaded)\n"
    )
    assert run_isolated(tmp_path, "", code) == "2 []"


ONLY_MARKED = textwrap.dedent(
    """\
    import speedysnek as snek

    @snek.nosnek
    def sum_squares(n):
        total = 0
        for i in range(n):
            total += i * i
        return total

    @snek.nosnek
    def update_database():
        pass
    """
)


def test_hunting_only_marked_functions_loads_nothing(tmp_path):
    code = (
        "import sys\nimport speedysnek as snek\nimport nosnek_sample\n"
        "a = snek.hunt(nosnek_sample, quiet=True)\n"
        "b = snek.hunt(nosnek_sample.sum_squares, quiet=True)\n"
        + LOADED
        + "print(a.skipped_count, b.findings[0].reason, loaded)\n"
    )
    assert run_isolated(tmp_path, ONLY_MARKED, code) == "2 opted_out []"


def test_biting_only_marked_functions_loads_nothing(tmp_path):
    code = (
        "import sys\nimport speedysnek as snek\nimport nosnek_sample\n"
        "report = snek.bite(nosnek_sample, quiet=True)\n"
        "peeked = snek.peek(nosnek_sample.sum_squares, quiet=True)\n"
        + LOADED
        + "print(report.skipped_count, peeked.findings[0].reason, loaded)\n"
    )
    assert run_isolated(tmp_path, ONLY_MARKED, code) == "2 opted_out []"


def test_public_api():
    assert "nosnek" in snek.__all__
    assert snek.nosnek is nosnek
