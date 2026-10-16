import asyncio
import subprocess
import sys

import pytest

import speedysnek
from speedysnek import jit


def test_public_api():
    assert speedysnek.__all__ == ["jit"]
    assert speedysnek.jit is jit


def test_result_is_unchanged():
    @jit
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert add("a", "b") == "ab"


def test_metadata_is_preserved():
    def original(x: int) -> int:
        """Docstring."""
        return x

    original.custom = "attr"
    wrapped = jit(original)

    assert wrapped.__name__ == "original"
    assert wrapped.__qualname__ == original.__qualname__
    assert wrapped.__module__ == original.__module__
    assert wrapped.__doc__ == "Docstring."
    assert wrapped.__annotations__ == {"x": int, "return": int}
    assert wrapped.__wrapped__ is original
    assert wrapped.custom == "attr"


def test_call_count():
    @jit
    def f():
        return None

    assert f.call_count == 0
    for _ in range(3):
        f()
    assert f.call_count == 3


def test_call_counts_are_independent():
    @jit
    def f():
        pass

    @jit
    def g():
        pass

    f()
    f()
    g()
    assert (f.call_count, g.call_count) == (2, 1)


def test_arguments_are_forwarded():
    @jit
    def f(a, /, b, *args, c, d=4, **kwargs):
        return a, b, args, c, d, kwargs

    assert f(1, 2, 3, c=5, e=6) == (1, 2, (3,), 5, 4, {"e": 6})
    assert f(1, b=2, c=3, d=0) == (1, 2, (), 3, 0, {})


def test_exceptions_propagate_and_are_counted():
    class Boom(Exception):
        pass

    error = Boom("boom")

    @jit
    def f():
        raise error

    with pytest.raises(Boom) as excinfo:
        f()
    assert excinfo.value is error
    assert f.call_count == 1


def test_bad_call_signature_raises_type_error():
    @jit
    def f(a):
        return a

    with pytest.raises(TypeError):
        f()


def test_works_as_method():
    class C:
        @jit
        def m(self, x):
            return self, x

    c = C()
    assert c.m(1) == (c, 1)
    assert C.m.call_count == 1


def test_rejects_non_callable():
    with pytest.raises(TypeError):
        jit(42)


def test_wrapper_gains_no_new_attributes():
    def add(a, b):
        return a + b

    wrapped = jit(add)
    assert vars(wrapped).keys() == {"__wrapped__", "call_count"}
    assert wrapped.__wrapped__ is add


def test_functions_the_frontend_rejects_still_work():
    @jit
    def divide(a, b):
        return a / b

    @jit
    def gen(n):
        yield from range(n)

    @jit
    async def coro(a, b):
        return a + b

    assert divide(1, 4) == 0.25
    assert list(gen(3)) == [0, 1, 2]
    assert asyncio.run(coro(1, 2)) == 3
    assert (divide.call_count, gen.call_count, coro.call_count) == (1, 1, 1)


def test_importing_speedysnek_does_not_load_the_frontend():
    code = "import sys, speedysnek; print('speedysnek._frontend' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
