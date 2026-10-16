"""Keeps ``experiments/native_wrapper.py`` honest.

Its numbers are only meaningful if every case is what it claims to be, so
these tests run its own verification, without timing: matching results,
the expected path for each case, and the canary proof that the native
wrapper runs the specialization's machine code.
"""

import importlib.util
from pathlib import Path

import pytest

import workloads

_PATH = Path(__file__).resolve().parent.parent / "experiments" / "native_wrapper.py"
_SPEC = importlib.util.spec_from_file_location("native_wrapper", _PATH)
experiment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(experiment)


@pytest.mark.parametrize("variant", sorted(experiment.VARIANTS))
def test_every_case_is_verified(variant):
    python = experiment.build(experiment.python_wrapper, python=True)
    cases = experiment.build(experiment.VARIANTS[variant])
    assert tuple(cases) == experiment.CASES
    for name, case in cases.items():
        experiment.verify(name, variant, case, python[name])


def test_the_python_wrapper_is_the_previous_production_logic():
    wrapper = experiment.python_wrapper(workloads.make_workload(1))
    assert workloads.call_path(wrapper, *workloads.ARGS) == "python"
    workloads.warm_up(wrapper, *workloads.ARGS)
    assert workloads.call_path(wrapper, *workloads.ARGS) == "native"
    assert wrapper.call_count == workloads.LIFECYCLE_CALLS + 2


def test_verification_rejects_a_wrong_path():
    cases = experiment.build(experiment.VARIANTS["native_wrapper"])
    python = experiment.build(experiment.python_wrapper, python=True)
    case = cases["ops_001"]
    case.path = "python"
    with pytest.raises(experiment.SetupError, match="path"):
        experiment.verify("ops_001", "native_wrapper", case, python["ops_001"])
