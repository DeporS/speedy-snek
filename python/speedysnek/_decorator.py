"""The ``jit`` decorator.

Stage 1: this is a plain Python wrapper. It records how many times the
function was called and then runs the original function unchanged. Nothing
is compiled or optimized yet.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any


def jit[F: Callable[..., Any]](func: F) -> F:
    """Mark ``func`` for future JIT compilation.

    For now the returned wrapper simply calls ``func`` with the same
    arguments, returning its result and propagating any exception unchanged.

    The wrapper exposes ``call_count``: the number of times it has been
    called, including calls that raised. The original function is available
    as ``__wrapped__`` (set by :func:`functools.wraps`).

    >>> @jit
    ... def add(a, b):
    ...     return a + b
    >>> add(1, 2)
    3
    >>> add.call_count
    1
    """
    if not callable(func):
        raise TypeError(f"jit expected a callable, got {type(func).__name__}")

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wrapper.call_count += 1
        return func(*args, **kwargs)

    # Set after functools.wraps, which copies func.__dict__ onto the wrapper.
    wrapper.call_count = 0
    return wrapper  # type: ignore[return-value]
