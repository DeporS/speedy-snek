"""SpeedySnek: a tiny experimental JIT compiler for Python.

Need more speed? Just ``@snekit``.

The public API is :func:`snekit`, also available under its descriptive
alias :func:`jit` (the same object), and :func:`hunt`, which reports which
functions the compiler currently understands, as a :class:`HuntReport` of
:class:`HuntFinding` objects. Everything else, including the native (Rust)
``speedysnek._native`` module, is private. The compilation modules are
imported lazily, the first time a decorated function becomes hot, and the
bytecode frontend the first time it is needed.
"""

from ._decorator import jit, snekit
from ._hunt import HuntFinding, HuntReport, hunt

__all__ = ["snekit", "jit", "hunt", "HuntReport", "HuntFinding"]
