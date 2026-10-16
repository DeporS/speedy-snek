"""SpeedySnek: a tiny experimental JIT compiler for Python.

The only public API is :func:`jit`. Everything else, including the native
(Rust) ``speedysnek._native`` module, is private. The compilation modules
are imported lazily, the first time a decorated function becomes hot.
"""

from ._decorator import jit

__all__ = ["jit"]
