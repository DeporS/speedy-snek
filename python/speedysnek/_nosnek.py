"""``nosnek``: mark a function SpeedySnek must leave alone.

``@nosnek`` is a marker, not a wrapper::

    import speedysnek as snek

    @snek.nosnek
    def update_database():
        ...

It returns the same function object, unchanged: same ``__code__``, same
attributes, nothing added to its ``__dict__`` and nothing on its call path.
The mark lives in a private weak set of the marked functions, which
SpeedySnek's automatic tools check: :func:`speedysnek.hunt` and
:func:`speedysnek.bite` report a marked function as skipped with the
reason ``"opted_out"``, without analyzing it, and ``bite`` never wraps it
or rebinds its names. It is a convenience, not a security boundary.

Marking a function later does not change a ``@snekit`` wrapper made from it
earlier, and an explicit ``@snekit`` still wraps a marked function: the
mark only keeps it out of automatic wrapping.

This module imports neither the native extension, the compiler nor the
frontend.
"""

from __future__ import annotations

import sys
import weakref
from collections.abc import Callable
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import Any

# Read directly, so no metaclass can intercept it.
_TYPE_NAME = type.__dict__["__name__"]

# The marked functions. Weak, so marking never keeps a function alive, and
# private, so nothing like ``functools.wraps`` copies the mark along.
_OPTED_OUT: weakref.WeakSet[FunctionType] = weakref.WeakSet()


def nosnek[F: Callable[..., Any]](func: F) -> F:
    """Mark ``func`` as Python-only and return it unchanged.

    SpeedySnek's automatic tools never wrap or compile a marked function:
    :func:`speedysnek.hunt` and :func:`speedysnek.bite` report it as
    skipped, with the reason ``"opted_out"``, and ``bite`` leaves every
    module name bound to it as it is. The function itself is not modified
    and calls to it are not affected. Marking it twice is harmless.

    Only plain Python functions can be marked. Apply ``@nosnek`` to the
    object the module binds, outside any other decorator that returns a
    new function.

    >>> @nosnek
    ... def update_database():
    ...     return "saved"
    >>> update_database()
    'saved'

    Raises :class:`TypeError` for anything else, such as a class, a builtin,
    a bound method, a callable object or a ``@snekit`` wrapper.
    """
    if type(func) is not FunctionType:
        raise TypeError(f"nosnek() expected a Python function, {_describe(func)}")
    _OPTED_OUT.add(func)
    return func


def _is_opted_out(func: FunctionType) -> bool:
    """Whether ``func``, an exact ``FunctionType``, is marked with @nosnek.

    Functions hash by identity, so the lookup runs no Python code.
    """
    return func in _OPTED_OUT


def _describe(obj: Any) -> str:
    kind = type(obj)
    name = _TYPE_NAME.__get__(kind)
    if issubclass(kind, type):
        return f"got the class {_TYPE_NAME.__get__(obj)}"
    if kind is BuiltinFunctionType:
        return f"got the built-in {obj.__name__}"
    if kind is MethodType:
        return "got a bound method; mark the plain function where it is defined"
    native = sys.modules.get("speedysnek._native")
    if native is not None and kind is native.NativeWrapper:
        return (
            "got a @snekit wrapper; @nosnek cannot undo @snekit, so remove "
            "@snekit instead"
        )
    return f"got {name}"
