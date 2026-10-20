"""Tests for ``speedysnek.bite``, which applies ``@snekit`` to a module."""

import io
import subprocess
import sys
import textwrap
import types

import pytest

import speedysnek as snek
from speedysnek import BiteFinding, BiteReport, HuntFinding, _bite, bite, hunt
from speedysnek._decorator import _WARMUP_CALLS, _Status
from speedysnek._native import NativeWrapper

# Ready and rejected functions, a private and an imported function, a class,
# a callable object and an alias, in a deliberately non-alphabetical order.
# ``CALLS`` records every call a function body makes.
SAMPLE = textwrap.dedent(
    """\
    from os.path import join

    CALLS = []

    def sum_squares(n):
        \"\"\"Add up the squares below n.\"\"\"
        total = 0
        for i in range(n):
            total += i * i
        return total

    def format_report(total):
        CALLS.append("format_report")
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


def make_module(source=SAMPLE, name="bite_sample"):
    module = types.ModuleType(name)
    exec(compile(source, f"{name}.py", "exec"), module.__dict__)
    return module


def by_name(report):
    return {finding.name: finding for finding in report}


def snapshot(module):
    return dict(module.__dict__)


# --- Wrapping ----------------------------------------------------------------


def test_ready_functions_become_snekit_wrappers():
    module = make_module()
    originals = snapshot(module)
    report = bite(module, quiet=True)

    assert type(module.sum_squares) is NativeWrapper
    assert type(module.clamp) is NativeWrapper
    findings = by_name(report)
    assert findings["sum_squares"].function is originals["sum_squares"]
    assert findings["sum_squares"].wrapper is module.sum_squares
    assert findings["clamp"].wrapper is module.clamp
    assert module.sum_squares.__wrapped__ is originals["sum_squares"]


def test_rejected_functions_stay_the_same_object():
    module = make_module()
    original = module.format_report
    finding = by_name(bite(module, quiet=True))["format_report"]
    assert module.format_report is original
    assert finding.status == "skipped"
    assert finding.wrapper is None
    assert finding.function is original
    assert finding.summary == "function calls and global names are not supported"
    assert module.format_report(3) == "total: 3"
    assert module.CALLS == ["format_report"]


def test_skipped_findings_match_hunt():
    module = make_module()
    hunted = by_name(hunt(module, quiet=True))
    for finding in bite(module, quiet=True):
        assert isinstance(finding.hunt, HuntFinding)
        assert finding.hunt.function is finding.function
        assert finding.hunt.summary == hunted[finding.name].summary
        if finding.status == "skipped":
            assert finding.summary == hunted[finding.name].summary


def test_wrapper_preserves_results_and_metadata():
    module = make_module()
    original = module.sum_squares
    bite(module, quiet=True)
    wrapper = module.sum_squares
    assert wrapper.__name__ == "sum_squares"
    assert wrapper.__qualname__ == "sum_squares"
    assert wrapper.__doc__ == "Add up the squares below n."
    assert wrapper.__module__ == "bite_sample"
    assert wrapper.__wrapped__ is original
    assert wrapper(10) == original(10) == 285
    assert module.clamp(1, 5) == 5
    assert module.clamp(9, 5) == 9


def test_wrapper_warms_up_then_compiles():
    module = make_module()
    bite(module, quiet=True)
    wrapper = module.sum_squares
    state = wrapper._jit_state
    # Biting neither calls nor compiles the function.
    assert wrapper.call_count == 0
    assert (state.status, state.warmup_calls, state.compile_attempts) == (
        _Status.COLD,
        0,
        0,
    )
    for _ in range(_WARMUP_CALLS):
        assert wrapper(100) == 328350
    assert state.status is _Status.COLD
    assert state.compile_attempts == 0
    assert wrapper(100) == 328350
    assert state.compile_attempts == 1
    assert state.status is _Status.COMPILED
    # Arguments the native path cannot take still run in Python.
    assert wrapper(True) == 0
    assert module.clamp(2**80, 1) == 2**80


def test_wrapper_binds_as_a_method():
    module = make_module(
        "def add_one(self, x):\n    return x + 1\n\n\nclass Box:\n    pass\n"
    )
    bite(module, quiet=True)
    module.Box.add_one = module.add_one
    assert module.Box().add_one(41) == 42


def test_bite_never_calls_candidates():
    calls = []
    module = make_module()
    module.CALLS = calls
    bite(module, quiet=True)
    assert calls == []
    for wrapper in (module.sum_squares, module.clamp):
        assert wrapper.call_count == 0
        assert wrapper._jit_state.compile_attempts == 0


def test_functions_are_not_modified():
    module = make_module()
    functions = {
        name: (value, value.__code__, dict(value.__dict__), value.__defaults__)
        for name, value in snapshot(module).items()
        if isinstance(value, types.FunctionType)
    }
    bite(module, quiet=True)
    for function, code, attributes, defaults in functions.values():
        assert function.__code__ is code
        assert function.__dict__ == attributes
        assert function.__defaults__ is defaults


# --- What is left alone ------------------------------------------------------


def test_private_imported_classes_and_callables_are_untouched():
    module = make_module()
    before = snapshot(module)
    report = bite(module, quiet=True)
    names = {name for finding in report for name in finding.names}
    for name in ("join", "_private", "Calculator", "double", "CALLS"):
        assert module.__dict__[name] is before[name]
        assert name not in names
    assert module.join is before["join"]
    assert module.Calculator.add is before["Calculator"].add
    assert module._private(1) == 2


def test_only_the_planned_names_change():
    module = make_module()
    before = snapshot(module)
    bite(module, quiet=True)
    after = snapshot(module)
    assert after.keys() == before.keys()
    changed = {name for name in before if after[name] is not before[name]}
    assert changed == {"sum_squares", "clamp", "total_of_squares"}


def test_imported_function_is_not_mutated_in_its_own_module():
    source = make_module()
    importer = make_module("def own(x):\n    return x + 1\n", name="bite_importer")
    importer.sum_squares = source.sum_squares
    report = bite(importer, quiet=True)
    assert [finding.name for finding in report] == ["own"]
    assert type(importer.sum_squares) is types.FunctionType
    assert type(source.sum_squares) is types.FunctionType


def test_references_captured_before_bite_keep_the_original():
    module = make_module()
    captured = module.sum_squares
    holder = {"f": module.sum_squares}
    bite(module, quiet=True)
    assert type(captured) is types.FunctionType
    assert holder["f"] is captured
    assert module.sum_squares is not captured
    assert captured(4) == 14


# --- Aliases -------------------------------------------------------------------


def test_aliases_become_the_same_wrapper():
    module = make_module()
    original = module.sum_squares
    bite(module, quiet=True)
    assert module.total_of_squares is module.sum_squares
    assert type(module.sum_squares) is NativeWrapper
    assert module.sum_squares.__wrapped__ is original


def test_one_finding_per_aliased_function():
    module = make_module()
    report = bite(module, quiet=True)
    assert [finding.name for finding in report] == [
        "sum_squares",
        "format_report",
        "clamp",
    ]
    finding = by_name(report)["sum_squares"]
    assert finding.names == ("sum_squares", "total_of_squares")
    assert finding.summary == "wrapped with @snekit, also as total_of_squares"
    assert by_name(report)["clamp"].names == ("clamp",)


def test_aliases_of_skipped_functions_stay_unchanged():
    module = make_module()
    module.report_alias = module.format_report
    bite(module, quiet=True)
    assert module.report_alias is module.format_report
    assert type(module.report_alias) is types.FunctionType


# --- Idempotency -------------------------------------------------------------


def test_second_bite_is_idempotent():
    module = make_module()
    bite(module, quiet=True)
    after_first = snapshot(module)
    wrapper = module.sum_squares
    state = wrapper._jit_state
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(10)
    counts = (wrapper.call_count, state.compile_attempts, state.status)

    report = bite(module, quiet=True)
    after_second = snapshot(module)
    assert after_second.keys() == after_first.keys()
    for name, value in after_first.items():
        assert after_second[name] is value
    assert (wrapper.call_count, state.compile_attempts, state.status) == counts
    assert wrapper._jit_state is state

    findings = by_name(report)
    assert findings["sum_squares"].status == "already_bitten"
    assert findings["sum_squares"].wrapper is wrapper
    assert findings["sum_squares"].hunt is None
    assert findings["sum_squares"].names == ("sum_squares", "total_of_squares")
    assert findings["clamp"].status == "already_bitten"
    assert findings["format_report"].status == "skipped"
    assert report.bitten_count == 0


def test_existing_snekit_wrappers_are_unchanged():
    module = make_module(
        textwrap.dedent(
            """\
            from speedysnek import snekit

            @snekit
            def old_calculation(n):
                return n * 3

            def new_calculation(n):
                return n * 4
            """
        )
    )
    existing = module.old_calculation
    report = bite(module, quiet=True)
    assert module.old_calculation is existing
    findings = by_name(report)
    assert findings["old_calculation"].status == "already_bitten"
    assert findings["old_calculation"].wrapper is existing
    assert findings["old_calculation"].function is existing.__wrapped__
    assert findings["old_calculation"].summary == "already wrapped with @snekit"
    assert findings["new_calculation"].status == "bitten"
    assert existing.call_count == 0


TRIPLE = textwrap.dedent(
    """\
    from speedysnek import snekit

    def triple(n):
        return n * 3

    {wrapper_name} = snekit(triple)
    times_three = triple
    """
)


def counting_wrap(monkeypatch):
    built = []

    def wrap(function):
        built.append(function)
        return snek.snekit(function)

    monkeypatch.setattr(_bite, "_wrap", wrap)
    return built


def jit_snapshot(wrapper):
    state = wrapper._jit_state
    return (
        state,
        wrapper.call_count,
        state.warmup_calls,
        state.compile_attempts,
        state.status,
        state.specialization,
    )


def warm(wrapper):
    """Compile ``wrapper``, then snapshot its JIT state."""
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(7)
    assert wrapper._jit_state.status is _Status.COMPILED
    return jit_snapshot(wrapper)


def test_plain_aliases_get_the_existing_public_wrapper(monkeypatch):
    built = counting_wrap(monkeypatch)
    module = make_module(TRIPLE.format(wrapper_name="fast_triple"))
    plain, wrapper = module.triple, module.fast_triple
    warmed = warm(wrapper)

    report = bite(module, quiet=True)
    assert built == []
    assert module.fast_triple is wrapper
    assert module.triple is wrapper
    assert module.times_three is wrapper
    assert wrapper.__wrapped__ is plain
    assert jit_snapshot(wrapper) == warmed
    (finding,) = report
    assert finding.name == "triple"
    assert finding.names == ("triple", "fast_triple", "times_three")
    assert finding.status == "already_bitten"
    assert finding.function is plain
    assert finding.wrapper is wrapper
    assert finding.summary == (
        "already wrapped with @snekit as fast_triple; "
        "triple, times_three now point to it too"
    )
    assert module.triple(5) == 15
    assert wrapper.call_count == warmed[1] + 1


def test_plain_names_get_an_existing_private_wrapper(monkeypatch):
    built = counting_wrap(monkeypatch)
    module = make_module(TRIPLE.format(wrapper_name="_fast"))
    plain, wrapper = module.triple, module._fast
    warmed = warm(wrapper)

    (finding,) = bite(module, quiet=True)
    assert built == []
    assert module._fast is wrapper
    assert module.triple is wrapper
    assert module.times_three is wrapper
    assert jit_snapshot(wrapper) == warmed
    assert finding.status == "already_bitten"
    assert finding.function is plain
    assert finding.wrapper is wrapper
    # The private name is used, but not reported or changed.
    assert finding.names == ("triple", "times_three")
    assert finding.summary == (
        "already wrapped with @snekit as _fast; triple, times_three now point to it too"
    )


@pytest.mark.parametrize("wrapper_name", ["fast_triple", "_fast"])
def test_unified_aliases_stay_put_on_a_second_bite(monkeypatch, wrapper_name):
    built = counting_wrap(monkeypatch)
    module = make_module(TRIPLE.format(wrapper_name=wrapper_name))
    bite(module, quiet=True)
    wrapper = module.triple
    after_first = snapshot(module)
    warmed = warm(wrapper)

    (finding,) = bite(module, quiet=True)
    assert built == []
    after_second = snapshot(module)
    assert after_second.keys() == after_first.keys()
    for name, value in after_first.items():
        assert after_second[name] is value
    assert jit_snapshot(wrapper) == warmed
    assert finding.status == "already_bitten"
    assert finding.wrapper is wrapper
    assert finding.summary == "already wrapped with @snekit"


def test_imported_wrapper_is_not_reported():
    source = make_module()
    bite(source, quiet=True)
    importer = make_module("X = 1\n", name="bite_importer")
    importer.sum_squares = source.sum_squares
    report = bite(importer, quiet=True)
    assert len(report) == 0
    assert importer.sum_squares is source.sum_squares


def test_rewritten_wrapped_attribute_is_not_trusted():
    module = make_module("def triple(n):\n    return n * 3\n")
    other = make_module("def triple(n):\n    return n * 3\n", name="elsewhere")
    wrapper = snek.snekit(other.triple)
    # Claims to wrap this module's function, but its JIT state says no.
    wrapper.__wrapped__ = module.triple
    module.fast = wrapper
    (finding,) = bite(module, quiet=True)
    assert finding.name == "triple"
    assert finding.status == "bitten"
    assert module.fast is wrapper


# --- Safety and atomicity ------------------------------------------------------


def test_failed_wrapper_preparation_changes_nothing(monkeypatch):
    module = make_module()
    before = snapshot(module)
    built = []

    def failing_wrap(function):
        if built:
            raise RuntimeError("wrapper preparation failed")
        built.append(function)
        return snek.snekit(function)

    monkeypatch.setattr(_bite, "_wrap", failing_wrap)
    with pytest.raises(RuntimeError, match="wrapper preparation failed"):
        bite(module, quiet=True)
    # The first wrapper was built, but no name was replaced.
    assert built == [before["sum_squares"]]
    after = snapshot(module)
    assert after.keys() == before.keys()
    for name, value in before.items():
        assert after[name] is value


def test_binding_changed_before_commit_changes_nothing(monkeypatch):
    module = make_module()
    before = snapshot(module)
    replacement = object()
    built = []

    def wrap_and_rebind(function):
        # ``clamp`` is wrapped last, after ``sum_squares`` is ready.
        if function.__name__ == "clamp":
            module.__dict__["clamp"] = replacement
        wrapper = snek.snekit(function)
        built.append(wrapper)
        return wrapper

    monkeypatch.setattr(_bite, "_wrap", wrap_and_rebind)
    report = bite(module, quiet=True)
    # The newer binding is kept, and no other name was replaced either.
    assert module.clamp is replacement
    after = snapshot(module)
    assert after.keys() == before.keys()
    for name, value in before.items():
        if name != "clamp":
            assert after[name] is value
    assert len(built) == 2
    for wrapper in built:
        assert wrapper.call_count == 0

    findings = by_name(report)
    assert findings["clamp"].status == "skipped"
    assert findings["clamp"].wrapper is None
    assert findings["clamp"].summary == (
        "its module binding changed before it could be updated"
    )
    assert findings["sum_squares"].status == "skipped"
    assert findings["sum_squares"].wrapper is None
    assert findings["sum_squares"].summary == (
        "module changed before it could be updated"
    )
    assert findings["format_report"].summary == (
        "function calls and global names are not supported"
    )
    assert (report.bitten_count, report.skipped_count) == (0, 3)


def test_binding_change_also_blocks_alias_unification(monkeypatch):
    module = make_module(
        TRIPLE.format(wrapper_name="fast_triple")
        + "\n\ndef double(n):\n    return n * 2\n"
    )
    plain, wrapper, double = module.triple, module.fast_triple, module.double
    replacement = object()

    def wrap_and_rebind(function):
        module.__dict__["double"] = replacement
        return snek.snekit(function)

    monkeypatch.setattr(_bite, "_wrap", wrap_and_rebind)
    report = bite(module, quiet=True)
    assert module.double is replacement
    assert module.triple is plain
    assert module.times_three is plain
    assert module.fast_triple is wrapper
    findings = by_name(report)
    assert findings["double"].summary == (
        "its module binding changed before it could be updated"
    )
    assert findings["triple"].status == "skipped"
    assert findings["triple"].wrapper is None
    assert findings["triple"].summary == "module changed before it could be updated"

    # Once the module is stable again, a new bite applies everything.
    monkeypatch.setattr(_bite, "_wrap", snek.snekit)
    module.double = double
    report = bite(module, quiet=True)
    assert module.triple is module.times_three is wrapper
    assert module.double.__wrapped__ is double
    assert report.bitten_count == report.already_bitten_count == 1


def test_already_bitten_without_changes_is_still_reported(monkeypatch):
    module = make_module(
        "from speedysnek import snekit\n\n"
        "@snekit\ndef old(n):\n    return n + 1\n\n"
        "def new(n):\n    return n + 2\n"
    )
    existing = module.old

    def wrap_and_rebind(function):
        module.__dict__["new"] = None
        return snek.snekit(function)

    monkeypatch.setattr(_bite, "_wrap", wrap_and_rebind)
    findings = by_name(bite(module, quiet=True))
    # Nothing was planned for ``old``, so its status is still true.
    assert findings["old"].status == "already_bitten"
    assert findings["old"].wrapper is existing
    assert findings["new"].status == "skipped"
    assert module.old is existing


def test_changed_alias_leaves_every_name_alone(monkeypatch):
    module = make_module()
    original = module.sum_squares
    replacement = object()

    def wrap_and_rebind(function):
        if function is original:
            module.__dict__["total_of_squares"] = replacement
        return snek.snekit(function)

    monkeypatch.setattr(_bite, "_wrap", wrap_and_rebind)
    finding = by_name(bite(module, quiet=True))["sum_squares"]
    assert finding.status == "skipped"
    # Neither name was rebound: a half-updated alias would bypass the JIT.
    assert module.sum_squares is original
    assert module.total_of_squares is replacement


def test_module_attribute_hooks_are_not_used():
    class Sneaky(types.ModuleType):
        touched = []

        def __getattr__(self, name):
            self.touched.append(name)
            raise AttributeError(name)

        def __getattribute__(self, name):
            type(self).touched.append(name)
            return super().__getattribute__(name)

        def __setattr__(self, name, value):
            type(self).touched.append(("set", name))
            super().__setattr__(name, value)

        @property
        def __dict__(self):
            type(self).touched.append("__dict__")
            return {}

    module = make_module()
    namespace = vars(module)
    Sneaky.touched.clear()
    module.__class__ = Sneaky
    report = bite(module, quiet=True)
    assert Sneaky.touched == []
    assert report.bitten_count == 2
    assert type(namespace["sum_squares"]) is NativeWrapper


# --- Report and rendering ----------------------------------------------------


def test_rendered_report_is_stable():
    module = make_module(
        textwrap.dedent(
            """\
            from speedysnek import snekit

            def sum_squares(n):
                total = 0
                for i in range(n):
                    total += i * i
                return total

            def format_report(total):
                return f"{total}"

            @snekit
            def old_calculation(n):
                return n * 3
            """
        )
    )
    report = bite(module, quiet=True)
    assert report.render() == (
        "🐍 Snek took a bite.\n"
        "\n"
        "sum_squares      bitten          wrapped with @snekit\n"
        "format_report    skipped         f-strings are not supported (line 10)\n"
        "old_calculation  already bitten  already wrapped with @snekit\n"
        "\n"
        "1 function bitten, 1 skipped, 1 already bitten."
    )
    assert str(report) == report.render()


def test_rendered_report_after_a_second_bite():
    module = make_module()
    bite(module, quiet=True)
    assert bite(module, quiet=True).render() == (
        "🐍 Snek found nothing new to bite.\n"
        "\n"
        "sum_squares    already bitten  already wrapped with @snekit\n"
        "format_report  skipped         function calls and global names are not"
        " supported (line 13)\n"
        "clamp          already bitten  already wrapped with @snekit\n"
        "\n"
        "0 functions bitten, 1 skipped, 2 already bitten."
    )


def test_module_with_no_functions():
    report = bite(make_module("X = 1\n", "empty"), quiet=True)
    assert len(report) == 0
    assert report.render() == "🐍 Snek found no functions to bite."


def test_counts_and_groups():
    report = bite(make_module(), quiet=True)
    assert isinstance(report, BiteReport)
    assert all(isinstance(f, BiteFinding) for f in report.findings)
    assert list(report) == list(report.findings)
    assert len(report) == 3
    assert (report.bitten_count, report.skipped_count) == (2, 1)
    assert report.already_bitten_count == 0
    assert [f.name for f in report.bitten] == ["sum_squares", "clamp"]
    assert [f.name for f in report.skipped] == ["format_report"]
    assert report.already_bitten == ()
    assert [f.bitten for f in report] == [True, False, True]
    assert repr(report) == "<BiteReport: 2 bitten, 1 skipped, 0 already bitten>"


def test_finding_repr():
    finding = by_name(bite(make_module(), quiet=True))["clamp"]
    assert repr(finding) == (
        "BiteFinding(name='clamp', status='bitten', summary='wrapped with @snekit')"
    )


def test_prints_the_report_by_default(capsys):
    report = bite(make_module())
    assert capsys.readouterr().out == report.render() + "\n"


def test_quiet_prints_nothing(capsys):
    bite(make_module(), quiet=True)
    bite(make_module(), quiet=True, stream=io.StringIO())
    assert capsys.readouterr() == ("", "")


def test_stream(capsys):
    stream = io.StringIO()
    report = bite(make_module(), stream=stream)
    assert stream.getvalue() == report.render() + "\n"
    assert capsys.readouterr() == ("", "")


def test_stream_that_cannot_encode_the_snek():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", newline="\n")
    report = bite(make_module(), stream=stream)
    stream.flush()
    expected = report.render().removeprefix("🐍 ") + "\n"
    assert raw.getvalue().decode("ascii") == expected


# --- Invalid input -----------------------------------------------------------


@pytest.mark.parametrize(
    ("arg", "message"),
    [
        (42, "bite() expected a module, got int"),
        (None, "bite() expected a module, got NoneType"),
        (len, "bite() expected a module, got builtin_function_or_method"),
        ([sys], "bite() expected a module, got list"),
        ("my_module", "got str; import the module and pass the module itself"),
    ],
)
def test_invalid_input_raises_type_error(arg, message):
    with pytest.raises(TypeError) as error:
        bite(arg, quiet=True)
    assert message in str(error.value)


def test_takes_exactly_one_module():
    with pytest.raises(TypeError):
        bite()
    with pytest.raises(TypeError):
        bite(make_module(), make_module())
    with pytest.raises(TypeError):
        bite(make_module(), True)


def test_a_function_is_rejected_not_wrapped():
    module = make_module()
    original = module.clamp
    with pytest.raises(TypeError, match="got function"):
        bite(module.clamp, quiet=True)
    assert module.clamp is original


# --- Lazy loading ----------------------------------------------------------------


def run_isolated(tmp_path, source, code):
    (tmp_path / "bite_sample.py").write_text(source)
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
        name for name in ("speedysnek._native", "speedysnek._compiler")
        if name in sys.modules
    ]
"""


def test_import_and_hunt_load_neither_the_extension_nor_the_compiler(tmp_path):
    code = (
        "import sys\nimport speedysnek as snek\nimport bite_sample\n"
        "snek.hunt(bite_sample, quiet=True)\n"
        + textwrap.dedent(LOADED)
        + "print(loaded)\n"
    )
    assert run_isolated(tmp_path, SAMPLE, code) == "[]"


def test_bite_without_ready_functions_does_not_load_the_extension(tmp_path):
    source = 'def greet(name):\n    return f"hi {name}"\n'
    code = (
        "import sys\nimport speedysnek as snek\nimport bite_sample\n"
        "report = snek.bite(bite_sample, quiet=True)\n"
        + textwrap.dedent(LOADED)
        + "print(report.skipped_count, loaded)\n"
    )
    assert run_isolated(tmp_path, source, code) == "1 []"


def test_bite_loads_only_the_extension(tmp_path):
    code = (
        "import sys\nimport speedysnek as snek\nimport bite_sample\n"
        "report = snek.bite(bite_sample, quiet=True)\n"
        + textwrap.dedent(LOADED)
        + "print(report.bitten_count, loaded, bite_sample.CALLS)\n"
    )
    assert run_isolated(tmp_path, SAMPLE, code) == ("2 ['speedysnek._native'] []")


def test_public_api():
    assert snek.bite is bite
    for name in ("bite", "BiteReport", "BiteFinding"):
        assert name in snek.__all__
    assert snek.BiteReport is BiteReport
    assert snek.BiteFinding is BiteFinding
