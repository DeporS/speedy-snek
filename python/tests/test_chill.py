"""Tests for ``speedysnek.chill``, which undoes the wrappers ``bite`` installed."""

import io
import subprocess
import sys
import textwrap
import types

import pytest

import speedysnek as snek
from speedysnek import ChillFinding, ChillReport, _chill, bite, chill, nosnek
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


def make_module(source=SAMPLE, name="chill_sample"):
    module = types.ModuleType(name)
    exec(compile(source, f"{name}.py", "exec"), module.__dict__)
    return module


def bitten_module(source=SAMPLE, name="chill_sample"):
    module = make_module(source, name)
    bite(module, quiet=True)
    return module


def by_name(report):
    return {finding.name: finding for finding in report}


def snapshot(module):
    return dict(module.__dict__)


def jit_snapshot(wrapper):
    state = wrapper._jit_state
    return (
        wrapper.call_count,
        state.status,
        state.warmup_calls,
        state.compile_attempts,
        state.compiled,
        state.specialization,
        state.disabled_reason,
    )


def warm(wrapper, *args):
    for _ in range(_WARMUP_CALLS + 1):
        wrapper(*args)


# --- Restoring -----------------------------------------------------------------


def test_bite_then_chill_restores_the_exact_original():
    module = make_module()
    originals = snapshot(module)
    bite(module, quiet=True)
    wrapper = module.clamp
    assert type(wrapper) is NativeWrapper

    report = chill(module, quiet=True)
    assert module.clamp is originals["clamp"]
    assert module.sum_squares is originals["sum_squares"]
    finding = by_name(report)["clamp"]
    assert finding.status == "chilled"
    assert finding.chilled
    assert finding.function is originals["clamp"]
    assert finding.wrapper is wrapper
    assert finding.names == ("clamp",)
    assert finding.summary == "restored plain Python function"


def test_the_whole_module_is_back_to_how_it_was():
    module = make_module()
    before = snapshot(module)
    bite(module, quiet=True)
    chill(module, quiet=True)
    after = snapshot(module)
    assert after.keys() == before.keys()
    for name, value in before.items():
        assert after[name] is value


def test_results_and_python_behaviour_are_preserved():
    module = bitten_module()
    expected = (module.sum_squares(10), module.clamp(1, 5), module.clamp(9, 5))
    chill(module, quiet=True)
    assert type(module.sum_squares) is types.FunctionType
    assert (module.sum_squares(10), module.clamp(1, 5), module.clamp(9, 5)) == (
        expected
    )
    assert module.sum_squares(10) == 285
    # Plain Python again: big integers, bools and keywords as Python has them.
    assert module.clamp(2**80, 1) == 2**80
    assert module.clamp(x=True, lo=0) is True
    assert module.sum_squares.__doc__ == "Add up the squares below n."
    assert module.format_report(3) == "total: 3"


def test_a_compiled_wrapper_is_chilled_without_touching_its_state():
    module = bitten_module()
    wrapper = module.sum_squares
    warm(wrapper, 100)
    assert wrapper._jit_state.status is _Status.COMPILED
    before = jit_snapshot(wrapper)

    finding = by_name(chill(module, quiet=True))["sum_squares"]
    assert finding.status == "chilled"
    assert finding.wrapper is wrapper
    assert jit_snapshot(wrapper) == before
    assert wrapper._jit_state.status is _Status.COMPILED


def test_a_cold_wrapper_is_chilled_without_compiling():
    module = bitten_module()
    wrapper = module.clamp
    wrapper(1, 2)
    before = jit_snapshot(wrapper)
    chill(module, quiet=True)
    assert jit_snapshot(wrapper) == before
    assert wrapper._jit_state.status is _Status.COLD
    assert wrapper._jit_state.compile_attempts == 0


def test_wrappers_held_elsewhere_keep_working():
    module = bitten_module()
    wrapper = module.sum_squares
    warm(wrapper, 100)
    chill(module, quiet=True)
    calls = wrapper.call_count
    assert wrapper(100) == 328350
    assert wrapper.call_count == calls + 1
    assert wrapper._jit_state.status is _Status.COMPILED


def test_chill_never_calls_candidates():
    calls = []
    module = bitten_module()
    module.CALLS = calls
    wrappers = (module.sum_squares, module.clamp)
    chill(module, quiet=True)
    assert calls == []
    for wrapper in wrappers:
        assert wrapper.call_count == 0
        assert wrapper._jit_state.warmup_calls == 0
        assert wrapper._jit_state.compile_attempts == 0


def test_functions_are_not_modified():
    module = make_module()
    functions = [
        (value, value.__code__, dict(value.__dict__), value.__defaults__)
        for value in snapshot(module).values()
        if isinstance(value, types.FunctionType)
    ]
    bite(module, quiet=True)
    chill(module, quiet=True)
    for function, code, attributes, defaults in functions:
        assert function.__code__ is code
        assert function.__dict__ == attributes
        assert function.__defaults__ is defaults


# --- Aliases -------------------------------------------------------------------


def test_aliases_are_all_restored_to_the_same_function():
    module = make_module()
    original = module.sum_squares
    bite(module, quiet=True)
    assert module.total_of_squares is module.sum_squares
    report = chill(module, quiet=True)
    assert module.sum_squares is original
    assert module.total_of_squares is original
    finding = by_name(report)["sum_squares"]
    assert finding.names == ("sum_squares", "total_of_squares")
    assert finding.summary == "restored plain Python function, also as total_of_squares"
    assert [finding.name for finding in report] == [
        "sum_squares",
        "format_report",
        "clamp",
    ]


def test_private_aliases_stay_untouched():
    module = make_module("def triple(n):\n    return n * 3\n\n_impl = triple\n")
    original = module.triple
    bite(module, quiet=True)
    wrapper = module.triple
    module._fast_triple = wrapper
    report = chill(module, quiet=True)
    assert module.triple is original
    assert module._impl is original
    assert module._fast_triple is wrapper
    assert by_name(report)["triple"].names == ("triple",)


def test_references_captured_before_bite_stay_the_same_throughout():
    module = make_module()
    captured = module.sum_squares
    holder = {"f": module.sum_squares}
    bite(module, quiet=True)
    assert holder["f"] is captured
    chill(module, quiet=True)
    assert holder["f"] is captured
    assert module.sum_squares is captured
    assert type(captured) is types.FunctionType


def test_references_captured_after_bite_keep_the_wrapper():
    module = bitten_module()
    captured = module.sum_squares
    chill(module, quiet=True)
    assert type(captured) is NativeWrapper
    assert module.sum_squares is captured.__wrapped__


# --- What is left alone ----------------------------------------------------------


def test_ordinary_functions_are_reported_unchanged():
    module = make_module()
    before = snapshot(module)
    report = chill(module, quiet=True)
    assert [(finding.name, finding.status) for finding in report] == [
        ("sum_squares", "unchanged"),
        ("format_report", "unchanged"),
        ("clamp", "unchanged"),
    ]
    finding = by_name(report)["clamp"]
    assert finding.wrapper is None
    assert finding.function is before["clamp"]
    assert finding.summary == "not wrapped with @snekit"
    assert by_name(report)["sum_squares"].names == ("sum_squares", "total_of_squares")
    after = snapshot(module)
    for name, value in before.items():
        assert after[name] is value


def test_rejected_functions_stay_untouched():
    module = make_module()
    original = module.format_report
    bite(module, quiet=True)
    finding = by_name(chill(module, quiet=True))["format_report"]
    assert module.format_report is original
    assert finding.status == "unchanged"
    assert finding.summary == "not wrapped with @snekit"


def test_private_imported_classes_and_callables_are_untouched():
    module = bitten_module()
    before = snapshot(module)
    report = chill(module, quiet=True)
    names = {name for finding in report for name in finding.names}
    for name in ("join", "_private", "Calculator", "double", "CALLS"):
        assert module.__dict__[name] is before[name]
        assert name not in names


def test_only_the_bitten_names_change():
    module = bitten_module()
    before = snapshot(module)
    chill(module, quiet=True)
    after = snapshot(module)
    assert after.keys() == before.keys()
    changed = {name for name in before if after[name] is not before[name]}
    assert changed == {"sum_squares", "clamp", "total_of_squares"}


def test_imported_wrappers_are_untouched_and_excluded():
    source = bitten_module()
    wrapper = source.sum_squares
    importer = make_module("def own(x):\n    return x + 1\n", name="chill_importer")
    importer.sum_squares = wrapper
    report = chill(importer, quiet=True)
    assert [finding.name for finding in report] == ["own"]
    assert importer.sum_squares is wrapper
    # Chilling the defining module leaves the importer's name alone too.
    chill(source, quiet=True)
    assert importer.sum_squares is wrapper
    assert type(source.sum_squares) is types.FunctionType


def test_wrappers_from_snekit_in_the_source_are_left_in_place():
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
    bite(module, quiet=True)
    report = chill(module, quiet=True)
    assert module.old_calculation is existing
    assert type(module.new_calculation) is types.FunctionType
    finding = by_name(report)["old_calculation"]
    assert finding.status == "skipped"
    assert finding.wrapper is existing
    assert finding.summary == "wrapped with @snekit, but not by snek.bite()"
    assert by_name(report)["new_calculation"].status == "chilled"
    assert existing.call_count == 0


def test_a_wrapper_made_by_hand_next_to_a_bitten_one_is_kept():
    module = bitten_module("def triple(n):\n    return n * 3\n")
    original = module.triple.__wrapped__
    by_hand = snek.snekit(original)
    module.fast_triple = by_hand
    finding = by_name(chill(module, quiet=True))["triple"]
    assert module.triple is original
    assert module.fast_triple is by_hand
    assert finding.status == "chilled"
    assert finding.names == ("triple", "fast_triple")
    assert finding.summary == (
        "restored plain Python function; fast_triple kept a @snekit wrapper "
        "not made by snek.bite()"
    )


def test_a_wrapper_installed_by_hand_is_not_bites():
    module = make_module("def triple(n):\n    return n * 3\n")
    original = module.triple
    # A wrapper of the same function, installed by hand, is not bite's.
    module.triple = snek.snekit(original)
    (finding,) = chill(module, quiet=True)
    assert finding.status == "skipped"
    assert type(module.triple) is NativeWrapper


# --- @nosnek -------------------------------------------------------------------


def test_nosnek_functions_stay_unchanged():
    module = make_module(
        "import speedysnek as snek\n\n"
        "@snek.nosnek\ndef keep(n):\n    return n + 1\n\n"
        "def fast(n):\n    return n + 2\n"
    )
    keep = module.keep
    bite(module, quiet=True)
    finding = by_name(chill(module, quiet=True))["keep"]
    assert module.keep is keep
    assert finding.status == "unchanged"
    assert finding.summary == "marked with @nosnek, not wrapped"
    assert type(module.fast) is types.FunctionType


def test_marked_after_bite_is_restored_and_stays_protected():
    module = make_module("def triple(n):\n    return n * 3\n")
    original = module.triple
    bite(module, quiet=True)
    nosnek(original)

    finding = by_name(chill(module, quiet=True))["triple"]
    assert finding.status == "chilled"
    assert module.triple is original

    report = bite(module, quiet=True)
    assert module.triple is original
    assert by_name(report)["triple"].status == "skipped"
    assert by_name(report)["triple"].hunt.reason == "opted_out"
    assert by_name(chill(module, quiet=True))["triple"].summary == (
        "marked with @nosnek, not wrapped"
    )


# --- Forged wrappers -------------------------------------------------------------


def test_a_rewritten_wrapped_attribute_is_not_trusted():
    module = make_module("def triple(n):\n    return n * 3\n")
    other = bitten_module("def triple(n):\n    return n * 3\n", name="elsewhere")
    foreign = other.triple
    # Claims to wrap this module's function, but its JIT state says no.
    foreign.__wrapped__ = module.triple
    module.fast = foreign
    (finding,) = chill(module, quiet=True)
    assert finding.name == "triple"
    assert finding.status == "unchanged"
    assert module.fast is foreign


def test_a_bitten_wrapper_is_restored_to_its_real_original():
    module = make_module(
        "def triple(n):\n    return n * 3\n\ndef decoy(n):\n    return n\n"
    )
    original, decoy = module.triple, module.decoy
    bite(module, quiet=True)
    wrapper = module.triple
    wrapper.__wrapped__ = decoy
    wrapper._jit_state = module.decoy._jit_state
    chill(module, quiet=True)
    assert module.triple is original
    assert module.decoy is decoy


def test_lookalikes_are_ignored():
    module = make_module("def triple(n):\n    return n * 3\n")
    original = module.triple
    real = snek.snekit(original)

    class Lookalike:
        __wrapped__ = original
        _jit_state = real._jit_state

        def __call__(self, *args):
            raise AssertionError("called")

        def __getattribute__(self, name):
            raise AssertionError(f"read {name}")

    module.fake = Lookalike()
    report = chill(module, quiet=True)
    assert [(finding.name, finding.status) for finding in report] == [
        ("triple", "unchanged")
    ]
    assert type(module.fake) is Lookalike


# --- Idempotency -----------------------------------------------------------------


def test_second_chill_is_idempotent():
    module = bitten_module()
    chill(module, quiet=True)
    after_first = snapshot(module)
    report = chill(module, quiet=True)
    after_second = snapshot(module)
    assert after_second.keys() == after_first.keys()
    for name, value in after_first.items():
        assert after_second[name] is value
    assert [finding.status for finding in report] == ["unchanged"] * 3
    assert report.chilled_count == 0


def test_bite_chill_bite_wraps_again():
    module = make_module()
    original = module.sum_squares
    bite(module, quiet=True)
    first = module.sum_squares
    chill(module, quiet=True)
    bite(module, quiet=True)
    assert module.sum_squares is not first
    assert module.sum_squares.__wrapped__ is original
    assert by_name(chill(module, quiet=True))["sum_squares"].status == "chilled"
    assert module.sum_squares is original


# --- Atomicity -------------------------------------------------------------------


def rebind_while_inspecting(monkeypatch, module, when, key, value):
    """Rebind ``key`` in ``module`` when ``chill`` inspects ``when``."""
    original_of = _chill._original_of

    def inspect_and_rebind(wrapper):
        if wrapper is when:
            module.__dict__[key] = value
        return original_of(wrapper)

    monkeypatch.setattr(_chill, "_original_of", inspect_and_rebind)


def test_binding_changed_before_commit_changes_nothing(monkeypatch):
    module = bitten_module()
    before = snapshot(module)
    replacement = object()
    # ``clamp`` is inspected last, after ``sum_squares`` is planned.
    rebind_while_inspecting(monkeypatch, module, module.clamp, "clamp", replacement)
    report = chill(module, quiet=True)

    assert module.clamp is replacement
    after = snapshot(module)
    assert after.keys() == before.keys()
    for name, value in before.items():
        if name != "clamp":
            assert after[name] is value
    assert type(module.sum_squares) is NativeWrapper

    findings = by_name(report)
    assert findings["clamp"].status == "skipped"
    assert findings["clamp"].wrapper is before["clamp"]
    assert findings["clamp"].summary == (
        "its module binding changed before it could be restored"
    )
    assert findings["sum_squares"].status == "skipped"
    assert findings["sum_squares"].summary == (
        "module changed before it could be restored"
    )
    assert findings["format_report"].status == "unchanged"
    assert (report.chilled_count, report.skipped_count, report.unchanged_count) == (
        0,
        2,
        1,
    )


def test_a_changed_alias_blocks_every_restoration(monkeypatch):
    module = bitten_module()
    wrapper = module.sum_squares
    replacement = object()
    rebind_while_inspecting(
        monkeypatch, module, module.clamp, "total_of_squares", replacement
    )
    findings = by_name(chill(module, quiet=True))
    # Neither name was restored: a half-restored alias would be confusing.
    assert module.sum_squares is wrapper
    assert module.total_of_squares is replacement
    assert type(module.clamp) is NativeWrapper
    assert findings["sum_squares"].summary == (
        "its module binding changed before it could be restored"
    )
    assert findings["clamp"].summary == "module changed before it could be restored"


def test_a_later_chill_applies_everything(monkeypatch):
    module = bitten_module()
    wrapper = module.clamp
    rebind_while_inspecting(monkeypatch, module, wrapper, "clamp", None)
    assert chill(module, quiet=True).chilled_count == 0
    monkeypatch.undo()
    module.clamp = wrapper
    report = chill(module, quiet=True)
    assert report.chilled_count == 2
    assert type(module.clamp) is types.FunctionType
    assert type(module.sum_squares) is types.FunctionType


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

    module = bitten_module()
    namespace = vars(module)
    Sneaky.touched.clear()
    module.__class__ = Sneaky
    report = chill(module, quiet=True)
    assert Sneaky.touched == []
    assert report.chilled_count == 2
    assert type(namespace["sum_squares"]) is types.FunctionType


# --- Report and rendering --------------------------------------------------------


def test_rendered_report_is_stable():
    module = make_module(
        textwrap.dedent(
            """\
            def sum_squares(n):
                total = 0
                for i in range(n):
                    total += i * i
                return total

            def format_report(total):
                return f"{total}"
            """
        )
    )
    bite(module, quiet=True)
    report = chill(module, quiet=True)
    assert report.render() == (
        "🐍 Snek cooled down.\n"
        "\n"
        "sum_squares    chilled    restored plain Python function\n"
        "format_report  unchanged  not wrapped with @snekit\n"
        "\n"
        "1 function chilled, 1 unchanged."
    )
    assert str(report) == report.render()


def test_rendered_report_with_skipped_functions():
    module = make_module(
        "from speedysnek import snekit\n\n"
        "@snekit\ndef old(n):\n    return n + 1\n\n"
        "def new(n):\n    return n + 2\n\n"
        "def other(n):\n    return n + 3\n"
    )
    bite(module, quiet=True)
    assert chill(module, quiet=True).render() == (
        "🐍 Snek cooled down.\n"
        "\n"
        "old    skipped  wrapped with @snekit, but not by snek.bite()\n"
        "new    chilled  restored plain Python function\n"
        "other  chilled  restored plain Python function\n"
        "\n"
        "2 functions chilled, 1 skipped, 0 unchanged."
    )


def test_rendered_report_with_nothing_to_chill():
    assert chill(make_module(), quiet=True).render() == (
        "🐍 Snek found nothing to chill.\n"
        "\n"
        "sum_squares    unchanged  not wrapped with @snekit\n"
        "format_report  unchanged  not wrapped with @snekit\n"
        "clamp          unchanged  not wrapped with @snekit\n"
        "\n"
        "0 functions chilled, 3 unchanged."
    )


def test_module_with_no_functions():
    report = chill(make_module("X = 1\n", "empty"), quiet=True)
    assert len(report) == 0
    assert report.render() == "🐍 Snek found no functions to chill."


def test_counts_and_groups():
    report = chill(bitten_module(), quiet=True)
    assert isinstance(report, ChillReport)
    assert all(isinstance(f, ChillFinding) for f in report.findings)
    assert list(report) == list(report.findings)
    assert len(report) == 3
    assert (report.chilled_count, report.skipped_count) == (2, 0)
    assert report.unchanged_count == 1
    assert [f.name for f in report.chilled] == ["sum_squares", "clamp"]
    assert [f.name for f in report.unchanged] == ["format_report"]
    assert report.skipped == ()
    assert [f.chilled for f in report] == [True, False, True]
    assert repr(report) == "<ChillReport: 2 chilled, 0 skipped, 1 unchanged>"


def test_finding_repr():
    finding = by_name(chill(bitten_module(), quiet=True))["clamp"]
    assert repr(finding) == (
        "ChillFinding(name='clamp', status='chilled', "
        "summary='restored plain Python function')"
    )


def test_prints_the_report_by_default(capsys):
    report = chill(bitten_module())
    assert capsys.readouterr().out == report.render() + "\n"


def test_quiet_prints_nothing(capsys):
    chill(bitten_module(), quiet=True)
    chill(bitten_module(), quiet=True, stream=io.StringIO())
    assert capsys.readouterr() == ("", "")


def test_stream(capsys):
    stream = io.StringIO()
    report = chill(bitten_module(), stream=stream)
    assert stream.getvalue() == report.render() + "\n"
    assert capsys.readouterr() == ("", "")


def test_stream_that_cannot_encode_the_snek():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", newline="\n")
    report = chill(bitten_module(), stream=stream)
    stream.flush()
    expected = report.render().removeprefix("🐍 ") + "\n"
    assert raw.getvalue().decode("ascii") == expected


# --- Invalid input -----------------------------------------------------------


@pytest.mark.parametrize(
    ("arg", "message"),
    [
        (42, "chill() expected a module, got int"),
        (None, "chill() expected a module, got NoneType"),
        (len, "chill() expected a module, got builtin_function_or_method"),
        ([sys], "chill() expected a module, got list"),
        ("my_module", "got str; import the module and pass the module itself"),
    ],
)
def test_invalid_input_raises_type_error(arg, message):
    with pytest.raises(TypeError) as error:
        chill(arg, quiet=True)
    assert message in str(error.value)


def test_takes_exactly_one_module():
    with pytest.raises(TypeError):
        chill()
    with pytest.raises(TypeError):
        chill(make_module(), make_module())
    with pytest.raises(TypeError):
        chill(make_module(), True)


def test_a_wrapper_is_rejected_not_unwrapped():
    module = bitten_module()
    wrapper = module.clamp
    with pytest.raises(TypeError, match="got NativeWrapper"):
        chill(module.clamp, quiet=True)
    assert module.clamp is wrapper


# --- Lazy loading ----------------------------------------------------------------


def run_isolated(tmp_path, source, code):
    (tmp_path / "chill_sample.py").write_text(source)
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
            "speedysnek._native",
            "speedysnek._frontend",
            "speedysnek._compiler",
        )
        if name in sys.modules
    ]
"""


def test_chill_without_wrappers_loads_nothing(tmp_path):
    code = (
        "import sys\nimport speedysnek as snek\nimport chill_sample\n"
        "report = snek.chill(chill_sample, quiet=True)\n"
        + textwrap.dedent(LOADED)
        + "print(report.unchanged_count, loaded, chill_sample.CALLS)\n"
    )
    assert run_isolated(tmp_path, SAMPLE, code) == "3 [] []"


def test_chill_imports_nothing_after_bite(tmp_path):
    code = (
        "import sys\nimport speedysnek as snek\nimport chill_sample\n"
        "snek.bite(chill_sample, quiet=True)\n"
        "before = set(sys.modules)\n"
        "report = snek.chill(chill_sample, quiet=True)\n"
        + textwrap.dedent(LOADED)
        + "print(report.chilled_count, sorted(set(sys.modules) - before), loaded,"
        " chill_sample.CALLS)\n"
    )
    assert run_isolated(tmp_path, SAMPLE, code) == (
        "2 [] ['speedysnek._native', 'speedysnek._frontend'] []"
    )


def test_public_api():
    assert snek.chill is chill
    for name in ("chill", "ChillReport", "ChillFinding"):
        assert name in snek.__all__
    assert snek.ChillReport is ChillReport
    assert snek.ChillFinding is ChillFinding
