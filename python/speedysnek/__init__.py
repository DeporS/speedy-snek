"""SpeedySnek: a tiny experimental JIT compiler for Python.

Need more speed? Just ``@snekit``.

The only public API is :func:`snekit`, also available under its
descriptive alias :func:`jit` (the same object). Everything else, including
the native (Rust) ``speedysnek._native`` module, is private. The
compilation modules are imported lazily, the first time a decorated
function becomes hot.
"""

from ._decorator import jit, snekit

__all__ = ["snekit", "jit"]
