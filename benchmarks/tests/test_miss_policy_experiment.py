"""Keeps ``experiments/miss_policy.py`` honest.

Its numbers only mean something if every case is in the state and takes
the path it claims, so these tests run its own verification, without
timing.
"""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "experiments" / "miss_policy.py"
_SPEC = importlib.util.spec_from_file_location("miss_policy", _PATH)
experiment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(experiment)


def test_this_build_has_the_policy():
    assert experiment.HAS_POLICY


@pytest.mark.parametrize("name", sorted(experiment.CASES))
def test_every_case_is_verified(name):
    targets = experiment.build(name)
    assert tuple(targets) == ("python", "snekit")
    assert experiment._status(name, targets["snekit"]) == experiment._expected(name)[0]


def test_the_cases_cover_the_policy():
    parked = {
        name
        for name, case in experiment.CASES.items()
        if case.expected["policy"][0] == "parked"
    }
    assert parked == {"float_only", "method", "keyword", "default", "overflow"}
    # On ``main`` the same calls kept paying for native attempts.
    assert all(
        experiment.CASES[name].expected["main"][0] == "compiled" for name in parked
    )


def test_verification_rejects_a_wrong_path(monkeypatch):
    case = experiment.CASES["float_only"]
    monkeypatch.setattr(case, "expected", {**case.expected, "policy": ("parked", 1, 1)})
    with pytest.raises(experiment.SetupError, match="native attempts"):
        experiment.build("float_only")


def test_verification_rejects_a_wrong_status(monkeypatch):
    case = experiment.CASES["overflow"]
    monkeypatch.setattr(
        case, "expected", {**case.expected, "policy": ("compiled", 1, 1)}
    )
    with pytest.raises(experiment.SetupError, match="status"):
        experiment.build("overflow")


def test_measure_runs_every_cell():
    medians = experiment.measure(["add", "alternating"], rounds=1, loops=1)
    assert set(medians) == {
        f"{name}/{variant}"
        for name in ("add", "alternating")
        for variant in ("python", "snekit")
    }
    assert all(value > 0 for value in medians.values())


def test_calls_mode_makes_the_calls():
    experiment.calls("method", 5)
