import asyncio
import subprocess
import sys

import pytest

import speedysnek
from speedysnek import snekit


def test_public_api():
    assert speedysnek.__all__ == [
        "snekit",
        "jit",
        "hunt",
        "HuntReport",
        "HuntFinding",
        "bite",
        "BiteReport",
        "BiteFinding",
        "peek",
        "PeekReport",
        "PeekFinding",
        "chill",
        "ChillReport",
        "ChillFinding",
        "nosnek",
    ]
    assert speedysnek.snekit is snekit


def test_jit_is_the_same_object_as_snekit():
    assert speedysnek.snekit is speedysnek.jit


def test_snekit_is_the_canonical_name():
    # The implementation is named ``snekit``; ``jit`` is only an alias.
    assert snekit.__name__ == "snekit"
    assert snekit.__qualname__ == "snekit"
    assert speedysnek.__all__[0] == "snekit"
    assert "@snekit" in snekit.__doc__
    with pytest.raises(TypeError, match="^snekit expected a callable"):
        snekit(42)


def test_jit_alias_still_works():
    """Backwards compatibility: existing ``@jit`` code keeps working."""
    from speedysnek import jit

    @jit
    def expression(a, b):
        return (a + b) * 2

    for _ in range(20):
        assert expression(10, 20) == 60
    assert expression(2**62, 2**62) == 2**64
    assert expression.call_count == 21
    assert expression._jit_state.status.value == "compiled"
    assert expression.__wrapped__.__name__ == "expression"


def test_result_is_unchanged():
    @snekit
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert add("a", "b") == "ab"


def test_metadata_is_preserved():
    def original(x: int) -> int:
        """Docstring."""
        return x

    original.custom = "attr"
    wrapped = snekit(original)

    assert wrapped.__name__ == "original"
    assert wrapped.__qualname__ == original.__qualname__
    assert wrapped.__module__ == original.__module__
    assert wrapped.__doc__ == "Docstring."
    assert wrapped.__annotations__ == {"x": int, "return": int}
    assert wrapped.__wrapped__ is original
    assert wrapped.custom == "attr"


def test_call_count():
    @snekit
    def f():
        return None

    assert f.call_count == 0
    for _ in range(3):
        f()
    assert f.call_count == 3


def test_call_counts_are_independent():
    @snekit
    def f():
        pass

    @snekit
    def g():
        pass

    f()
    f()
    g()
    assert (f.call_count, g.call_count) == (2, 1)


def test_arguments_are_forwarded():
    @snekit
    def f(a, /, b, *args, c, d=4, **kwargs):
        return a, b, args, c, d, kwargs

    assert f(1, 2, 3, c=5, e=6) == (1, 2, (3,), 5, 4, {"e": 6})
    assert f(1, b=2, c=3, d=0) == (1, 2, (), 3, 0, {})


def test_exceptions_propagate_and_are_counted():
    class Boom(Exception):
        pass

    error = Boom("boom")

    @snekit
    def f():
        raise error

    with pytest.raises(Boom) as excinfo:
        f()
    assert excinfo.value is error
    assert f.call_count == 1


def test_bad_call_signature_raises_type_error():
    @snekit
    def f(a):
        return a

    with pytest.raises(TypeError):
        f()


def test_works_as_method():
    class C:
        @snekit
        def m(self, x):
            return self, x

    c = C()
    assert c.m(1) == (c, 1)
    assert C.m.call_count == 1


def test_rejects_non_callable():
    with pytest.raises(TypeError):
        snekit(42)


def test_wrapper_gains_no_new_public_attributes():
    def add(a, b):
        return a + b

    wrapped = snekit(add)
    assert {name for name in dir(wrapped) if not name.startswith("_")} == {"call_count"}
    # The wrapper is a native object, not a function, so the metadata that
    # ``functools.wraps`` copies lives in its ``__dict__``. ``call_count`` is
    # a property. ``_jit_state`` is private diagnostic state (see
    # test_jit_compile.py).
    copied = {"__module__", "__name__", "__qualname__", "__doc__", "__type_params__"}
    assert vars(wrapped).keys() == copied | {"__wrapped__", "_jit_state"}
    assert wrapped.__wrapped__ is add


def test_functions_the_frontend_rejects_still_work():
    @snekit
    def divide(a, b):
        return a / b

    @snekit
    def gen(n):
        yield from range(n)

    @snekit
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
