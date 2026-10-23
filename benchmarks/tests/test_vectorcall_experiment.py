"""Keeps ``experiments/vectorcall.py`` honest.

Its numbers only mean something if every case takes the path and has the
status it claims, and if the callgrind parser counts what it says, so
these tests run its own verification, without timing or valgrind.
"""

import importlib.util
from pathlib import Path

import pytest

from speedysnek._decorator import _Status

_PATH = Path(__file__).resolve().parent.parent / "experiments" / "vectorcall.py"
_SPEC = importlib.util.spec_from_file_location("vectorcall", _PATH)
experiment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(experiment)


@pytest.mark.parametrize("name", sorted(experiment.CASES))
def test_every_case_is_verified(name):
    callables = experiment.build(name)
    assert tuple(callables) == experiment.VARIANTS
    case = experiment.CASES[name]
    wrapper = getattr(callables["snekit"], "__func__", callables["snekit"])
    assert wrapper._jit_state.status is case.status


def test_the_required_cases_are_measured():
    assert {
        "add",
        "ops_030",
        "ops_100",
        "float_miss",
        "keyword",
        "default",
        "disabled",
    } <= set(experiment.CASES)


def test_verification_rejects_a_wrong_path():
    case = experiment.CASES["add"]
    wrong = experiment.Case(case.make, case.args, path="guard_miss")
    experiment.CASES["wrong"] = wrong
    try:
        with pytest.raises(experiment.SetupError, match="path"):
            experiment.build("wrong")
    finally:
        del experiment.CASES["wrong"]


def test_verification_rejects_a_wrong_status():
    case = experiment.CASES["add"]
    wrong = experiment.Case(case.make, case.args, status=_Status.PARKED)
    experiment.CASES["wrong"] = wrong
    try:
        with pytest.raises(experiment.SetupError, match="status"):
            experiment.build("wrong")
    finally:
        del experiment.CASES["wrong"]


def test_statements_call_like_the_case_says():
    f = object()
    assert experiment.statement(f, experiment.CASES["add"])[0] == "f(x0, x1)"
    assert experiment.statement(f, experiment.CASES["keyword"])[0] == (
        "f(a=k_a, b=k_b)"
    )
    assert experiment.statement(f, experiment.CASES["star_args"])[0] == "f(*xs)"
    assert experiment.statement(f, experiment.CASES["float_miss"])[0] == (
        "f(x0, x1); f(r0, r1)"
    )


def test_measure_runs_every_cell():
    names = ["add", "keyword", "method"]
    medians = experiment.measure(names, rounds=1, loops=1)
    assert set(medians) == {
        f"{name}/{variant}" for name in names for variant in experiment.VARIANTS
    }
    assert all(value > 0 for value in medians.values())


def test_calls_mode_makes_the_calls():
    experiment.calls("method", "snekit", 5)
    with pytest.raises(KeyError):
        experiment.calls("add", "nonexistent", 1)


def test_the_callgrind_parser_counts_calls_through_compressed_names(tmp_path):
    out = tmp_path / "callgrind.out"
    out.write_text(
        "events: Ir\n"
        "fn=(1) caller\n"
        "cfn=(2) tuple_dealloc\n"
        "calls=3 0\n"
        "5 10\n"
        "cfn=(3) dict_dealloc\n"
        "calls=1 0\n"
        "fn=(2)\n"
        "cfn=(2)\n"
        "calls=4 0\n"
        "summary: 1234\n"
        "totals: 1234\n"
    )
    total, calls = experiment.callgrind_calls(str(out))
    assert total == 1234
    assert calls["tuple_dealloc"] == 7
    assert calls["dict_dealloc"] == 1


def test_builds_must_be_named():
    options = experiment.argparse.Namespace(build=["main"])
    with pytest.raises(SystemExit):
        experiment.parse_builds(options)
    options = experiment.argparse.Namespace(build=None)
    assert experiment.parse_builds(options) == {"current": None}


def test_this_build_reports_no_vectorcall():
    assert experiment.vectorcall() is False
