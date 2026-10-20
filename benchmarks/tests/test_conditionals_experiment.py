"""Keeps ``experiments/conditionals.py`` honest.

Its numbers only mean something if every case runs natively, so these
tests run its own verification, without timing.
"""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "experiments" / "conditionals.py"
_SPEC = importlib.util.spec_from_file_location("conditionals", _PATH)
experiment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(experiment)


@pytest.mark.parametrize("name", sorted(experiment.CASES))
def test_every_case_runs_natively(name):
    plain, wrapped, args = experiment.prepare(name)
    assert wrapped(*args) == plain(*args)


def test_branch_cases_cover_both_arms():
    func_then, args_then = experiment.CASES["if_else_then"]
    func_else, args_else = experiment.CASES["if_else_else"]
    assert func_then is func_else
    assert func_then(*args_then) == 0  # The ``score < 0`` arm.
    assert func_else(*args_else) > 0
