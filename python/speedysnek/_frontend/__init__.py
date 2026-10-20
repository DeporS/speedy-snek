"""Private bytecode frontend: Python function -> normalized instructions.

This is the first stage of the compiler pipeline::

    Python function
        -> CPython 3.14 bytecode
        -> SpeedySnek normalized instructions  (this package)
        -> reference interpreter  (``speedysnek._interpreter``)
           or Cranelift i64 backend  (``speedysnek._compiler``)

Only straight-line functions of positional arguments, ``int`` constants,
``+``, ``-``, ``*``, assignments to local variables (including arguments)
and ``return`` are supported. Everything else is rejected with a
machine-readable :class:`RejectionReason`.

:func:`speedysnek.snekit` calls :func:`analyze` once per decorated function,
after warmup; a rejection makes it run the original function from then on.
The frontend is not part of the public API.

>>> def add(a, b):
...     return a + b
>>> for instruction in analyze(add).instructions:
...     print(instruction)
LOAD_ARG 0
LOAD_ARG 1
ADD
RETURN
"""

from __future__ import annotations

import inspect
import sys
from types import FunctionType

from . import cpython314
from .ir import Analysis, Instruction, Op, Rejection, RejectionReason

__all__ = [
    "Analysis",
    "Instruction",
    "Op",
    "Rejection",
    "RejectionReason",
    "analyze",
]

# Code object flags (from ``inspect``) for functions that do not simply run
# and return, checked in this order.
_FLAG_REJECTIONS = (
    (inspect.CO_ASYNC_GENERATOR, RejectionReason.ASYNC_GENERATOR, "async generator"),
    (inspect.CO_COROUTINE, RejectionReason.COROUTINE, "async function"),
    (inspect.CO_GENERATOR, RejectionReason.GENERATOR, "generator"),
)


def analyze(func: object) -> Analysis:
    """Analyze ``func`` without calling it.

    Returns an :class:`Analysis` with normalized instructions if ``func``
    is supported, or with a :class:`Rejection` explaining why not. Never
    raises for unsupported input.
    """
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 14):
        return _reject(
            RejectionReason.UNSUPPORTED_PYTHON_VERSION,
            "the bytecode frontend requires CPython 3.14",
        )
    if not isinstance(func, FunctionType):
        return _reject(
            RejectionReason.NOT_A_FUNCTION,
            f"expected a Python function, got {type(func).__name__}",
        )

    code = func.__code__
    for flag, reason, kind in _FLAG_REJECTIONS:
        if code.co_flags & flag:
            return _reject(reason, f"{kind}s are not supported")
    if code.co_freevars:
        return _reject(
            RejectionReason.CLOSURE,
            f"closures are not supported (free variables: {code.co_freevars})",
        )
    if (
        code.co_kwonlyargcount
        or code.co_flags & inspect.CO_VARARGS
        or code.co_flags & inspect.CO_VARKEYWORDS
    ):
        return _reject(
            RejectionReason.UNSUPPORTED_SIGNATURE,
            "only positional arguments are supported, "
            "not keyword-only arguments, *args or **kwargs",
        )

    return cpython314.translate(code)


def _reject(reason: RejectionReason, message: str) -> Analysis:
    return Analysis(rejection=Rejection(reason=reason, message=message))
