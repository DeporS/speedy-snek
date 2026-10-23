"""Keeps ``experiments/hot_slot.py`` honest.

Its numbers only mean something if every case takes the path it claims,
with a live hot slot exactly where the function compiled and none in the
``generic`` variant, so these tests run its own verification, without
timing.
"""

import importlib.util
from pathlib import Path

import pytest

import workloads

_PATH = Path(__file__).resolve().parent.parent / "experiments" / "hot_slot.py"
_SPEC = importlib.util.spec_from_file_location("hot_slot", _PATH)
experiment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(experiment)


@pytest.mark.parametrize("name", sorted(experiment.CASES))
def test_every_case_is_verified(name):
    callables = experiment.build(name)
    assert tuple(callables) == experiment.VARIANTS
    case = experiment.CASES[name]
    hot = callables["snekit"]._jit_state._hot
    assert hot.live == (case.path != "disabled")
    assert callables["generic"]._jit_state._hot is None


def test_the_generic_variant_never_uses_a_slot():
    wrapper = experiment.generic(workloads.make_workload(1))
    workloads.warm_up(wrapper, *workloads.ARGS)
    assert workloads.call_path(wrapper, *workloads.ARGS) == "native"
    assert wrapper.call_count == workloads.LIFECYCLE_CALLS + 1


def test_verification_rejects_a_wrong_path():
    case = experiment.CASES["add"]
    func = case.make()
    wrapper = experiment.snekit(func)
    workloads.warm_up(wrapper, *case.warm)
    wrong = experiment.Case(case.make, case.args, path="guard_miss")
    with pytest.raises(experiment.SetupError, match="path"):
        experiment.verify("snekit/add", "snekit", wrapper, func, wrong)


def test_measure_runs_every_cell():
    medians = experiment.measure(["add", "disabled"], rounds=1, loops=1)
    assert set(medians) == {
        f"{name}/{variant}"
        for name in ("add", "disabled")
        for variant in experiment.VARIANTS
    }
    assert all(value > 0 for value in medians.values())


def test_calls_mode_makes_the_calls():
    experiment.calls("add", "snekit", 5)
    with pytest.raises(KeyError):
        experiment.calls("add", "nonexistent", 1)
