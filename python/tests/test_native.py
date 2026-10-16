import pytest

from speedysnek import _native


def test_sum_as_string():
    assert _native.sum_as_string(1, 1) == "2"
    assert _native.sum_as_string(0, 0) == "0"


def test_sum_as_string_rejects_negative():
    # The Rust signature takes ``usize``.
    with pytest.raises(OverflowError):
        _native.sum_as_string(-1, 1)
