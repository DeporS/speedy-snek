"""Keeps ``experiments/loops.py`` honest.

Its numbers only mean something if every case runs natively, so these
tests run its own verification, without timing.
"""

import importlib.util
from pathlib import Path

import pytest

from speedysnek import _native

_PATH = Path(__file__).resolve().parent.parent / "experiments" / "loops.py"
_SPEC = importlib.util.spec_from_file_location("loops", _PATH)
experiment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(experiment)


@pytest.mark.parametrize("name", sorted(experiment.CASES))
def test_every_case_runs_natively(name):
    plain, wrapped, args = experiment.prepare(name)
    assert wrapped(*args) == plain(*args)


def test_every_loop_case_stays_below_the_budget():
    assert experiment.ITERATIONS
    assert max(experiment.ITERATIONS.values()) < _native.DEFAULT_BACKEDGE_BUDGET


def test_conditional_cases_take_both_branches():
    cases = [
        args
        for func, args in experiment.CASES.values()
        if func is experiment.conditional_sum
    ]
    assert cases
    for n, cutoff in cases:
        assert 0 < cutoff < n
