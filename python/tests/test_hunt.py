"""Tests for ``speedysnek.hunt``, the discovery API."""

import io
import subprocess
import sys
import textwrap
import types

import pytest

import speedysnek as snek
from speedysnek import HuntFinding, HuntReport, hunt

# A module with ready, rejected, private and imported functions, a class
# and a callable object, in a deliberately non-alphabetical order.
SAMPLE = textwrap.dedent(
    """\
    from os.path import join
    from textwrap import dedent

    CALLS = []

    def sum_squares(n):
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

    def load_data(path):
        CALLS.append("load_data")
        return open(path).read()

    def countdown(n):
        steps = 0
        while n > 0:
            n -= 1
            steps += 1
        return steps

    def _private(x):
        return x

    class Calculator:
        def add(self, a, b):
            return a + b

    class Multiplier:
        def __call__(self, x):
            CALLS.append("Multiplier")
            return x * 2

    double = Multiplier()
    alias = sum_squares
    square = lambda x: x * x
    """
)


def make_module(source=SAMPLE, name="hunt_sample"):
    module = types.ModuleType(name)
    exec(compile(source, f"{name}.py", "exec"), module.__dict__)
    return module


def by_name(report):
    return {finding.name: finding for finding in report}


def line_of(source, text):
    return source.splitlines().index(text) + 1


# --- Module discovery ------------------------------------------------------


def test_module_finds_its_own_public_functions_in_definition_order():
    report = hunt(make_module(), quiet=True)
    assert [finding.name for finding in report] == [
        "sum_squares",
        "format_report",
        "clamp",
        "load_data",
        "countdown",
        "square",
    ]


def test_module_skips_private_imported_classes_callables_and_aliases():
    module = make_module()
    names = {finding.name for finding in hunt(module, quiet=True)}
    # Imported Python functions, a private function, a class, its method,
    # a callable object and a second name for a function are all left out.
    for name in ("join", "dedent", "_private", "Calculator", "add", "double"):
        assert name not in names
    assert "alias" not in names
    assert sum(f.function is module.sum_squares for f in hunt(module, quiet=True)) == 1


def test_module_statuses_and_summaries():
    report = hunt(make_module(), quiet=True)
    findings = by_name(report)
    assert findings["sum_squares"].status == "ready"
    assert findings["sum_squares"].summary == "integer arithmetic and one range loop"
    assert findings["clamp"].summary == "integer arithmetic and if/else branches"
    assert findings["countdown"].summary == "integer arithmetic and one while loop"
    assert findings["square"].summary == "integer arithmetic"
    assert findings["load_data"].status == "skipped"
    assert findings["load_data"].summary == (
        "function calls and global names are not supported"
    )


def test_findings_keep_the_original_function_object():
    module = make_module()
    findings = by_name(hunt(module, quiet=True))
    assert findings["sum_squares"].function is module.sum_squares
    assert findings["load_data"].function is module.load_data


def test_module_with_no_functions():
    report = hunt(make_module("X = 1\n", "empty"), quiet=True)
    assert len(report) == 0
    assert report.render() == "🐍 Snek went hunting, but found no functions."


def test_module_reads_its_namespace_without_running_module_code():
    class Sneaky(types.ModuleType):
        touched = []

        def __getattr__(self, name):
            self.touched.append(name)
            raise AttributeError(name)

        def __getattribute__(self, name):
            type(self).touched.append(name)
            return super().__getattribute__(name)

        @property
        def __dict__(self):
            type(self).touched.append("__dict__")
            return {}

    module = make_module()
    Sneaky.touched.clear()
    module.__class__ = Sneaky
    report = hunt(module, quiet=True)
    assert Sneaky.touched == []
    assert "sum_squares" in by_name(report)
    # The traps are real: ordinary attribute access runs them.
    assert module.__dict__ == {}
    assert Sneaky.touched


def test_function_with_a_foreign_module_name_is_left_out():
    module = make_module()
    module.sum_squares.__module__ = "elsewhere"
    assert "sum_squares" not in by_name(hunt(module, quiet=True))


# --- Explicit functions ----------------------------------------------------


def test_explicit_functions_keep_the_supplied_order():
    module = make_module()
    report = hunt(module.load_data, module.clamp, module.sum_squares, quiet=True)
    assert [f.name for f in report] == ["load_data", "clamp", "sum_squares"]
    assert [f.function for f in report] == [
        module.load_data,
        module.clamp,
        module.sum_squares,
    ]


def test_one_iterable_of_functions():
    module = make_module()
    expected = ["countdown", "_private", "join"]
    for iterable in (
        [module.countdown, module._private, module.join],
        (module.countdown, module._private, module.join),
        (f for f in (module.countdown, module._private, module.join)),
    ):
        report = hunt(iterable, quiet=True)
        assert [f.name for f in report] == expected
    # Explicitly passed private and imported functions are inspected.
    assert by_name(report)["_private"].status == "ready"
    assert by_name(report)["join"].status == "skipped"


def test_a_single_function():
    module = make_module()
    report = hunt(module.sum_squares, quiet=True)
    assert len(report) == 1
    (finding,) = report
    assert finding.name == "sum_squares"
    assert finding.ready


def test_empty_iterable():
    report = hunt([], quiet=True)
    assert report.findings == ()
    assert report.ready_count == report.skipped_count == 0


def test_non_python_callables_are_skipped_not_rejected():
    module = make_module()
    report = hunt(
        len,
        [].append,
        module.Calculator,
        module.Calculator().add,
        module.double,
        quiet=True,
    )
    assert [f.name for f in report] == [
        "len",
        "append",
        "Calculator",
        "add",
        "<Multiplier object>",
    ]
    assert [f.summary for f in report] == [
        "built-in functions cannot be compiled",
        "built-in functions cannot be compiled",
        "classes cannot be compiled, only functions",
        "bound methods cannot be compiled; pass the plain function",
        "not a Python function",
    ]
    for finding in report:
        assert finding.status == "skipped"
        assert finding.reason == "not_a_function"
        assert finding.detail.startswith("expected a Python function, got ")
    assert module.CALLS == []


def test_snekit_wrapper_is_skipped_with_a_hint():
    module = make_module()
    wrapped = snek.snekit(module.clamp)
    (finding,) = hunt(wrapped, quiet=True)
    assert finding.name == "clamp"
    assert finding.function is wrapped
    assert finding.summary == "already wrapped by @snekit; pass the original function"
    assert wrapped.call_count == 0
    assert wrapped._jit_state.compile_attempts == 0


def test_callable_objects_are_not_probed():
    touched = []

    class Meta(type):
        @property
        def __name__(cls):
            touched.append("Meta.__name__")
            return "Fake"

    class Probe(metaclass=Meta):
        def __call__(self):
            touched.append("__call__")

        def __getattr__(self, name):
            touched.append(name)
            raise AttributeError(name)

        @property
        def __class__(self):
            touched.append("__class__")
            return types.FunctionType

    report = hunt(Probe(), Probe, quiet=True)
    assert touched == []
    assert [f.name for f in report] == ["<Probe object>", "Probe"]
    # The traps are real: ordinary attribute access runs them.
    assert Probe.__name__ == "Fake"
    assert isinstance(Probe(), types.FunctionType)
    assert touched == ["Meta.__name__", "__class__"]


# --- Invalid input -----------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((), "hunt() needs a module, a function or an iterable of functions"),
        (
            (42,),
            "hunt() expected a module, a function or an iterable of functions, got int",
        ),
        ((None,), "got NoneType"),
        (("my_module",), "import it and pass the module itself"),
        ((b"x",), "got bytes"),
        (([1, 2],), "hunt() expected a function, got int"),
        ((len, 3), "hunt() expected a function, got int"),
        ((len, [len]), "hunt() expected a function, got list"),
        ((sys, len), "hunt() takes a module on its own: hunt(module)"),
        (([sys],), "hunt() takes a module on its own: hunt(module)"),
        ((sys, sys), "hunt() takes a module on its own: hunt(module)"),
    ],
)
def test_invalid_input_raises_type_error(args, message):
    with pytest.raises(TypeError) as error:
        hunt(*args, quiet=True)
    assert message in str(error.value)


# --- Rejection reasons -------------------------------------------------------


def test_rejection_keeps_the_frontend_reason_and_message():
    module = make_module()
    finding = by_name(hunt(module, quiet=True))["load_data"]
    assert finding.reason == "unsupported_opcode"
    assert finding.detail.startswith("unsupported opcode LOAD_GLOBAL at offset ")
    assert finding.lineno == line_of(SAMPLE, '    CALLS.append("load_data")')

    finding = by_name(hunt(module, quiet=True))["format_report"]
    assert finding.reason == "unsupported_opcode"
    assert finding.summary == "function calls and global names are not supported"


def test_ready_findings_have_no_rejection():
    finding = by_name(hunt(make_module(), quiet=True))["sum_squares"]
    assert (finding.reason, finding.detail, finding.lineno) == (None, None, None)


REJECTIONS = [
    ("def f(x): return f'{x}'", "f-strings are not supported"),
    ("def f(x): return x.real", "attributes and methods are not supported"),
    ("def f(x): return [x]", "lists, tuples, sets and dicts are not supported"),
    ("def f(x): return -x", "unary operators like -x are not supported"),
    ("def f(x): return not x", "truth tests and not are not supported"),
    ("def f(x):\n    a, b = x\n    return a", "unpacking is not supported"),
    ("def f(x): return lambda: x", "nested functions and lambdas are not supported"),
    ("def f(x): return x / 2", "unsupported binary operator '/'"),
    ("def f(x): return x // 2", "unsupported binary operator '//'"),
    ("def f(x): return 1.5", "only integer constants are supported, found float"),
    ("def f(x): return x is None", "only integer constants are supported, found None"),
    ("def f(x):\n    y = x", "only integer constants are supported, found None"),
    (
        "def f(x):\n    x + 1\n    return x",
        "unused expression values are not supported",
    ),
    ("def f(x): return x < 3", "comparisons only work as if or while conditions"),
    (
        "def f(x):\n    if x:\n        return 1\n    return 2",
        "conditions must be one integer comparison",
    ),
    (
        "def f(x):\n    for k in x:\n        x = k\n    return x",
        "only for loops over range(n) are supported",
    ),
    ("def f(*args): return 1", "only positional arguments are supported"),
    ("def f(*, x): return x", "only positional arguments are supported"),
    ("def f(x):\n    yield x", "generators are not supported"),
    ("async def f(x): return x", "async functions are not supported"),
    (
        "def f(x):\n    try:\n        return x\n    except ValueError:\n"
        "        return 0",
        "exception handling (try, with) is not supported",
    ),
    (
        "def f(x):\n    for i in range(x):\n        for j in range(x):\n"
        "            x += j\n    return x",
        "only one loop per function is supported, not nested loops or several loops",
    ),
    (
        "def f(x):\n    if x < 1:\n        y = 1\n    return y",
        "local variable 'y' may be read before it is assigned",
    ),
]


@pytest.mark.parametrize(("source", "summary"), REJECTIONS)
def test_rejection_summaries_are_useful(source, summary):
    namespace = {}
    exec(source, namespace)
    (finding,) = hunt(namespace["f"], quiet=True)
    assert finding.status == "skipped"
    assert finding.summary == summary
    assert finding.reason is not None
    assert finding.detail
    assert "offset" not in finding.summary


def test_closure_summary():
    def outer():
        y = 1

        def inner(x):
            return x + y

        return inner

    (finding,) = hunt(outer(), quiet=True)
    assert finding.reason == "closure"
    assert finding.summary == "closures are not supported"
    assert "free variables" in finding.detail


def test_ready_summary_with_branches_and_a_loop():
    def mixed(n):
        total = 0
        for i in range(n):
            if i < 3:
                total += i
        return total

    def early_continue(n):
        i = 0
        while i < n:
            i += 1
            if i == 5:
                continue
            i += 2
        return i

    report = hunt(mixed, early_continue, quiet=True)
    assert [f.summary for f in report] == [
        "integer arithmetic, if/else branches and one range loop",
        "integer arithmetic, if/else branches and one while loop",
    ]


# --- Report and rendering ----------------------------------------------------


def test_rendered_report_is_stable():
    report = hunt(make_module(), quiet=True)
    format_line = line_of(SAMPLE, '    CALLS.append("format_report")')
    load_line = line_of(SAMPLE, '    CALLS.append("load_data")')
    assert report.render() == (
        "🐍 Snek went hunting.\n"
        "\n"
        "sum_squares    ready    integer arithmetic and one range loop\n"
        "format_report  skipped  function calls and global names are not supported"
        f" (line {format_line})\n"
        "clamp          ready    integer arithmetic and if/else branches\n"
        "load_data      skipped  function calls and global names are not supported"
        f" (line {load_line})\n"
        "countdown      ready    integer arithmetic and one while loop\n"
        "square         ready    integer arithmetic\n"
        "\n"
        "4 ready, 2 skipped."
    )
    assert str(report) == report.render()


def test_summary_counts():
    report = hunt(make_module(), quiet=True)
    assert isinstance(report, HuntReport)
    assert (report.ready_count, report.skipped_count) == (4, 2)
    assert [f.name for f in report.ready] == [
        "sum_squares",
        "clamp",
        "countdown",
        "square",
    ]
    assert [f.name for f in report.skipped] == ["format_report", "load_data"]
    assert len(report) == 6
    assert all(isinstance(f, HuntFinding) for f in report.findings)
    assert repr(report) == "<HuntReport: 4 ready, 2 skipped>"


def test_finding_repr():
    (finding,) = hunt(make_module().clamp, quiet=True)
    assert repr(finding) == (
        "HuntFinding(name='clamp', status='ready', "
        "summary='integer arithmetic and if/else branches')"
    )


def test_prints_the_report_by_default(capsys):
    report = hunt(make_module())
    assert capsys.readouterr().out == report.render() + "\n"


def test_quiet_prints_nothing(capsys):
    hunt(make_module(), quiet=True)
    hunt(make_module(), quiet=True, stream=io.StringIO())
    assert capsys.readouterr() == ("", "")


def test_stream(capsys):
    stream = io.StringIO()
    report = hunt(make_module(), stream=stream)
    assert stream.getvalue() == report.render() + "\n"
    assert capsys.readouterr() == ("", "")


def test_stream_that_cannot_encode_the_snek():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", newline="\n")
    report = hunt(make_module(), stream=stream)
    stream.flush()
    expected = report.render().removeprefix("🐍 ") + "\n"
    assert raw.getvalue().decode("ascii") == expected


# --- No execution, no mutation, no compilation -------------------------------


def test_hunt_never_calls_candidates():
    module = make_module()
    hunt(module, quiet=True)
    hunt(module.format_report, module.load_data, module.double, quiet=True)
    assert module.CALLS == []


def test_side_effecting_function_is_only_analyzed():
    calls = []

    def ready_but_tracked(n):
        return n + 1

    def side_effect(n):
        calls.append(n)
        return n

    report = hunt(ready_but_tracked, side_effect, quiet=True)
    assert calls == []
    assert [f.status for f in report] == ["ready", "skipped"]
    # The function still works and was not replaced or wrapped.
    assert side_effect(3) == 3
    assert calls == [3]


def test_hunt_does_not_mutate_the_module_or_functions():
    module = make_module()
    namespace = dict(module.__dict__)
    functions = {
        name: (value.__code__, dict(value.__dict__), value.__defaults__)
        for name, value in namespace.items()
        if isinstance(value, types.FunctionType)
    }
    hunt(module, quiet=True)
    hunt([module.sum_squares, module.clamp], quiet=True)
    assert module.__dict__.keys() == namespace.keys()
    for name, value in namespace.items():
        assert module.__dict__[name] is value
    for name, (code, attributes, defaults) in functions.items():
        function = module.__dict__[name]
        assert function.__code__ is code
        assert function.__dict__ == attributes
        assert function.__defaults__ is defaults
        assert type(function) is types.FunctionType


def test_hunt_loads_neither_the_extension_nor_the_compiler(tmp_path):
    (tmp_path / "hunt_sample.py").write_text(SAMPLE)
    code = textwrap.dedent(
        """\
        import sys
        import speedysnek as snek
        import hunt_sample

        report = snek.hunt(hunt_sample, quiet=True)
        loaded = [
            name for name in ("speedysnek._native", "speedysnek._compiler")
            if name in sys.modules
        ]
        print(report.ready_count, report.skipped_count, loaded, hunt_sample.CALLS)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    assert result.stdout.strip() == "4 2 [] []"


def test_importing_speedysnek_does_not_load_the_frontend_for_hunt():
    code = (
        "import sys, speedysnek; "
        "print(speedysnek.hunt.__name__, 'speedysnek._frontend' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "hunt False"


def test_public_api():
    assert snek.hunt is hunt
    assert "hunt" in snek.__all__
    assert snek.HuntReport is HuntReport
    assert snek.HuntFinding is HuntFinding
