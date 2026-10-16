"""The ``snekit`` decorator, also available as ``jit``.

``snekit`` is the canonical name; ``jit`` is the same object, kept as a
descriptive, backwards-compatible alias.

The decorator connects the compiler pipeline::

    decorated function
        -> warmup                            (_WARMUP_CALLS plain Python calls)
        -> frontend analysis                 (speedysnek._frontend)
        -> normalized IR
        -> cached Cranelift i64 specialization  (speedysnek._compiler)
        -> native execution when guards pass
        -> original CPython function otherwise

Every decorated function gets its own :class:`_JitState`, a small state
machine with three states (:class:`_Status`):

- ``COLD``: the first ``_WARMUP_CALLS`` calls run the original function.
  The next call tries to compile it, exactly once, and then picks a path.
- ``COMPILED``: a native specialization is cached. A call uses it when the
  function's ``__code__`` is still the compiled code object, no keyword
  arguments are passed and the number of positional arguments is exactly
  the compiled arity. If the native guards reject the arguments (``bool``,
  ``int`` subclasses, any non-``int``, values outside the signed 64-bit
  range, or overflow), that one call runs the original function instead.
  The specialization stays cached for later calls.
- ``DISABLED``: the frontend rejected the function, the backend could not
  compile it, the compilation machinery failed, a native call failed
  unexpectedly, or ``__code__`` was replaced after compilation. Every later
  call runs the original function.

A warmed, eligible call makes one native call, with no Python frame in
between::

    wrapper(*args)
        -> spec.fast_call(args)         (I64Function.try_call, in Rust)
            -> validate every argument once: exact int, in i64
            -> run the machine code, which checks every operation for overflow
            -> return the result int, or None for a guard miss
        -> original function, only if the result was None

Guard misses are expected, so they are reported with the ``None``
sentinel instead of an exception. Only an unexpected native error (an
``Exception`` from ``fast_call``) disables the specialization.
``CompiledI64.__call__``, which explains each guard miss with a
``NativeGuardFailure``, is not on this path.

The only correctness rule is that ``@snekit`` never changes what a call
returns or raises. Whenever native execution does not apply, the original
function is called exactly once with the original arguments, and whatever
it returns or raises is passed through unchanged.

The compilation modules are imported lazily, on the first compilation
attempt, so ``import speedysnek`` stays cheap.
"""

from __future__ import annotations

import enum
import functools
import threading
from collections.abc import Callable
from types import CodeType, FunctionType
from typing import Any

# How many calls run the original function before the next call tries to
# compile it.
_WARMUP_CALLS = 8


class _Status(enum.Enum):
    """Where a decorated function is in its lifecycle."""

    COLD = "cold"
    COMPILED = "compiled"
    DISABLED = "disabled"


class _DisabledReason(enum.Enum):
    """Why a decorated function was switched to ``DISABLED``."""

    # The frontend rejected the function; see ``_JitState.rejection``.
    REJECTED = "rejected"
    # The backend raised ``UnsupportedProgramError``; see ``compile_error``.
    UNSUPPORTED = "unsupported"
    # The compilation machinery raised anything else; see ``compile_error``.
    COMPILER_ERROR = "compiler_error"
    # A ``BaseException`` such as ``KeyboardInterrupt`` interrupted
    # compilation. It was propagated, not stored.
    INTERRUPTED = "interrupted"
    # ``__code__`` was replaced during or after compilation.
    CODE_CHANGED = "code_changed"
    # A native call raised an ``Exception``; guard misses never raise. See
    # ``native_error``.
    NATIVE_ERROR = "native_error"


class _Specialization:
    """Everything a native call needs, published as one object.

    The wrapper reads ``_JitState.specialization`` once per call, so it
    either sees a complete specialization or none at all. Everything the
    hot path needs is resolved here, once.
    """

    __slots__ = ("arg_count", "code", "fast_call", "native")

    def __init__(self, code: CodeType, native: Any) -> None:
        self.code = code
        # The ``CompiledI64``, kept for diagnostics; the wrapper never calls it.
        self.native = native
        self.arg_count = native.arg_count
        # The bound ``I64Function.try_call``: ``fast_call(args)`` returns
        # the result ``int``, or ``None`` for a guard miss. It also keeps
        # the machine code alive while a call is using it.
        self.fast_call = native.fast_call


class _JitState:
    """Per-function JIT state, exposed as the wrapper's ``_jit_state``.

    It is private and exists for tests and debugging.

    Invariant: ``specialization`` is not ``None`` exactly when ``status``
    is ``COMPILED``. ``compiled`` and ``compiled_code`` keep the last
    successful compilation for inspection even after it is invalidated.
    """

    __slots__ = (
        "_func",
        "_lock",
        "compile_attempts",
        "compile_error",
        "compiled",
        "compiled_code",
        "disabled_reason",
        "native_error",
        "rejection",
        "specialization",
        "status",
        "warmup_calls",
    )

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func
        self._lock = threading.Lock()
        self.status = _Status.COLD
        self.specialization: _Specialization | None = None
        self.warmup_calls = 0
        self.compile_attempts = 0
        self.rejection = None
        self.compile_error: Exception | None = None
        self.native_error: Exception | None = None
        self.compiled = None
        self.compiled_code: CodeType | None = None
        self.disabled_reason: _DisabledReason | None = None

    def cold_call(self) -> _Specialization | None:
        """Count one call while ``COLD``; compile once warmup is over.

        Returns the specialization to try for this call, if there is one.
        """
        # Not atomic across threads: a lost update only delays compilation.
        self.warmup_calls += 1
        if self.warmup_calls <= _WARMUP_CALLS:
            return None
        with self._lock:
            # Another thread may have finished compiling while we waited.
            if self.status is _Status.COLD:
                self._compile()
        return self.specialization

    def invalidate(self) -> None:
        """``__code__`` no longer matches the compiled code object."""
        with self._lock:
            if self.status is _Status.COMPILED:
                self._disable(_DisabledReason.CODE_CHANGED)

    def native_failed(self, error: Exception) -> None:
        """A native call raised; guard misses return ``None`` instead."""
        with self._lock:
            if self.status is _Status.COMPILED:
                self.native_error = error
                self._disable(_DisabledReason.NATIVE_ERROR)

    def _compile(self) -> None:
        # Called with the lock held, at most once per wrapper.
        self.compile_attempts += 1
        try:
            self._try_compile()
        except Exception as error:
            # A compiler bug must never break valid user code.
            self.compile_error = error
            self._disable(_DisabledReason.COMPILER_ERROR)
        finally:
            # Only a BaseException leaves the state COLD here. It
            # propagates; this function is never compiled again.
            if self.status is _Status.COLD:
                self._disable(_DisabledReason.INTERRUPTED)

    def _try_compile(self) -> None:
        from ._frontend import analyze

        func = self._func
        # ``analyze`` rejects everything that is not a plain function.
        code = func.__code__ if isinstance(func, FunctionType) else None
        analysis = analyze(func)
        if analysis.rejection is not None:
            self.rejection = analysis.rejection
            self._disable(_DisabledReason.REJECTED)
            return

        from . import _compiler

        try:
            # Pass the full positional arity: arguments the bytecode never
            # reads are still part of every call.
            compiled = _compiler.compile_i64(analysis.instructions, code.co_argcount)
        except _compiler.UnsupportedProgramError as error:
            self.compile_error = error
            self._disable(_DisabledReason.UNSUPPORTED)
            return

        if func.__code__ is not code:
            # Replaced by another thread while we were compiling.
            self._disable(_DisabledReason.CODE_CHANGED)
            return

        self.compiled = compiled
        self.compiled_code = code
        # Publish the complete specialization before changing the status.
        self.specialization = _Specialization(code, compiled)
        self.status = _Status.COMPILED

    def _disable(self, reason: _DisabledReason) -> None:
        self.specialization = None
        self.status = _Status.DISABLED
        self.disabled_reason = reason


def snekit[F: Callable[..., Any]](func: F) -> F:
    """Compile ``func`` to native code once it is hot, where possible.

    The returned wrapper behaves exactly like ``func``: same results, same
    exceptions, same metadata (via :func:`functools.wraps`, so the original
    function is ``__wrapped__``). After a short warmup it tries, once, to
    compile ``func`` to signed 64-bit machine code. Calls that the compiled
    code cannot handle, and functions that cannot be compiled at all, simply
    run ``func``.

    The wrapper exposes ``call_count``: the number of times it has been
    called, including calls that raised.

    >>> @snekit
    ... def expression(a, b):
    ...     return (a + b) * 2
    >>> [expression(10, 20) for _ in range(10)][-1]
    60
    >>> expression(2**62, 2**62)  # overflows i64: Python computes it
    18446744073709551616
    >>> expression.call_count
    11
    """
    if not callable(func):
        raise TypeError(f"snekit expected a callable, got {type(func).__name__}")

    state = _JitState(func)
    cold = _Status.COLD

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wrapper.call_count += 1
        spec = state.specialization
        if spec is None:
            if state.status is not cold:
                return func(*args, **kwargs)
            spec = state.cold_call()
            if spec is None:
                return func(*args, **kwargs)
        if func.__code__ is not spec.code:
            # Never run machine code compiled from a stale code object.
            state.invalidate()
        elif not kwargs and len(args) == spec.arg_count:
            try:
                result = spec.fast_call(args)
            except Exception as error:
                # Unexpected: guard misses return None. A BaseException
                # such as KeyboardInterrupt propagates.
                state.native_failed(error)
            else:
                if result is not None:
                    return result
        # Called outside the ``except`` block, so an exception raised by
        # ``func`` gets no ``__context__`` from the native attempt.
        return func(*args, **kwargs)

    # Set after functools.wraps, which copies func.__dict__ onto the wrapper.
    wrapper.call_count = 0
    # Private diagnostics only; the wrapper uses its closure, not this.
    wrapper._jit_state = state
    return wrapper  # type: ignore[return-value]


# The descriptive, backwards-compatible name. It is the same object, so
# ``@jit`` and ``@snekit`` behave identically.
jit = snekit
