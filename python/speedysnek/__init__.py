"""SpeedySnek: a tiny experimental JIT compiler for Python.

Native (Rust) functionality lives in the private ``speedysnek._native``
module and is not part of the public API.
"""

from ._decorator import jit

__all__ = ["jit"]
