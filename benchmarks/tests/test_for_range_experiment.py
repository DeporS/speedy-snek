"""Keeps ``experiments/for_range.py`` honest.

Its numbers only mean something if every case runs natively, with a range
guard exactly where ``range`` is used, so these tests run its own
verification, without timing.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from speedysnek import _native

_DIR = Path(__file__).resolve().parent.parent / "experiments"
sys.path.insert(0, str(_DIR))
_SPEC = importlib.util.spec_from_file_location("for_range", _DIR / "for_range.py")
experiment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(experiment)


@pytest.mark.parametrize("name", sorted(experiment.CASES))
def test_every_case_runs_natively(name):
    variants = experiment.prepare(name)
    expected = variants["python"][0](*variants["python"][1])
    for f, args in variants.values():
        assert f(*args) == expected
    guarded = variants["snekit"][0]._jit_state.specialization.guard
    assert (guarded is not None) == (name != "straight_20")
    if "while" in variants:
        assert variants["while"][0]._jit_state.specialization.guard is None


def test_every_loop_case_stays_below_the_budget():
    assert experiment.ITERATIONS
    assert max(experiment.ITERATIONS.values()) < _native.DEFAULT_BACKEDGE_BUDGET


def test_conditional_cases_take_both_branches():
    cases = [
        args
        for func, _, args in experiment.CASES.values()
        if func is experiment.conditional_sum
    ]
    assert cases
    for n, cutoff in cases:
        assert 0 < cutoff < n
