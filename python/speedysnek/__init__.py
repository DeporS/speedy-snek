"""SpeedySnek: a tiny experimental JIT compiler for Python.

Need more speed? Just ``@snekit``.

The public API is :func:`snekit`, also available under its descriptive
alias :func:`jit` (the same object); :func:`hunt`, which reports which
functions the compiler currently understands, as a :class:`HuntReport` of
:class:`HuntFinding` objects; :func:`bite`, which applies ``@snekit``
to the ready functions of one module and reports what it did, as a
:class:`BiteReport` of :class:`BiteFinding` objects; :func:`peek`, which
reports what the JIT has done with wrapped functions so far, as a
:class:`PeekReport` of :class:`PeekFinding` objects; and :func:`nosnek`,
which marks a function that ``hunt`` and ``bite`` must leave as plain
Python. Everything else, including the native (Rust)
``speedysnek._native`` module, is private.
The compilation modules are imported lazily, the first time a decorated
function becomes hot, and the bytecode frontend the first time it is
needed.
"""

from ._bite import BiteFinding, BiteReport, bite
from ._decorator import jit, snekit
from ._hunt import HuntFinding, HuntReport, hunt
from ._nosnek import nosnek
from ._peek import PeekFinding, PeekReport, peek

__all__ = [
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
    "nosnek",
]
