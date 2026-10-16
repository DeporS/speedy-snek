import pytest
import speedysnek


def test_sum_as_string():
    assert speedysnek.sum_as_string(1, 1) == "2"
