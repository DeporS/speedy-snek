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
machine with four states (:class:`_Status`):

- ``COLD``: the first ``_WARMUP_CALLS`` calls run the original function,
  and the wrapper notes, for each, whether it could have run natively (see
  *the miss policy* below). The next call decides: it tries to compile the
  function, exactly once, or parks it.
- ``COMPILED``: a native specialization is cached. A call uses it when the
  function's ``__code__`` is still the compiled code object, no keyword
  arguments are passed and the number of positional arguments is exactly
  the compiled arity. If the native guards reject the arguments (``bool``,
  ``int`` subclasses, any non-``int``, values outside the signed 64-bit
  range, or overflow), or a loop uses up the call's backedge budget, that
  one call runs the original function instead, from the start and with
  the original arguments. The specialization stays cached for later calls.
- ``PARKED``: the observed calls do not fit the i64 specialization, so
  calls run the original function without a native attempt. Either warmup
  showed it (nothing is compiled then) or a compiled function's calls kept
  missing. It is not an error and not final: now and then a call that fits
  re-probes, which compiles the function or makes it ``COMPILED`` again.
- ``DISABLED``: the frontend rejected the function, the backend could not
  compile it, the compilation machinery failed, a native call failed
  unexpectedly, ``__code__`` was replaced after compilation, or the
  function's ``range`` could not be guarded or stopped being the builtin.
  Every later call runs the original function.

**The range guard.** A ``for`` loop over ``range(n)`` compiles to integer
arithmetic that assumes ``range`` is the builtin (the frontend reports it
in ``Analysis.assumed_builtins``). Compilation then creates a
``_native.RangeGuard`` from the function's ``__globals__`` and
``__builtins__``: it proves, without running Python code, that ``range``
resolves to the genuine builtin ``range`` type, and a dict watcher
invalidates it on any later change to either dict that could affect that
lookup. If the proof fails, the function is ``DISABLED``
(``RANGE_UNGUARDED``) and always runs as Python. Otherwise the
specialization's ``function`` is a ``_native.GuardedI64Function``, whose
guard the wrapper checks before every native call. When it has failed,
that call runs the original function once, with the original arguments,
and the specialization is disabled for good (``RANGE_CHANGED``), exactly
as when ``__code__`` is replaced: restoring the old binding does not
re-enable it. Functions that do not use ``range`` get no guard and no
check.

The wrapper is a :class:`speedysnek._native.NativeWrapper`: the per-call
logic runs in Rust, so a call through it enters no Python frame. It
holds the original function and its ``_JitState``. Compilation and every
state transition stay in Python: the wrapper only reads
``state._specialization`` (the slot behind ``state.specialization``) and
``state.status`` and calls the state's
``cold_call``, ``invalidate`` and ``native_failed`` methods, in the same
places the earlier Python closure did, plus the miss policy's
``repeated_misses``, ``reprobe_succeeded`` and ``reprobe`` (see below). A
warmed, eligible call runs no Python code at all.

**The hot slot.** Reading those attributes on every call is most of a
warmed call's cost, so ``snekit`` also gives the wrapper and its state one
``_native.HotSlot``. Compilation publishes the specialization's code
object, machine code and range guard into it, once, after the status is
``COMPILED``; every later assignment to ``state.specialization``, on any
path to ``DISABLED``, retires it for good first. While it is live, a call
checks it without reading any attribute: the function's ``__code__`` is
read through the C API, and the range guard is one atomic load. When the
code object or the guard no longer match, the call takes the generic path,
which reads the state and makes the transition exactly as before. A
wrapper made without a slot, or a state whose specialization a test
replaced, always takes the generic path::

    wrapper(*args)                      (NativeWrapper.__call__, in Rust)
        -> live hot slot: __code__ and the range guard still match?
           otherwise the generic path reads the state's specialization
           and checks __code__ and the range guard there
        -> check keywords and arity
        -> spec.function's fast path    (the same Rust code as try_call)
            -> validate every argument once: exact int, in i64
            -> run the machine code, which checks every operation for overflow
               and every loop iteration against the backedge budget
            -> the result int, or a guard miss
        -> original function, only for a guard miss

    parked hot slot                     (the miss policy, below)
        -> __code__ and the range guard still match? (compiled only)
        -> original function, without a native attempt,
           except for a re-probe

**The miss policy.** Native execution only pays off for calls that fit
the specialization; a call that does not fit pays for the attempt and then
runs the original function anyway. So the wrapper stops attempting when the
calls it sees clearly do not fit, using only facts it already has: whether
there are keywords, the number of positional arguments, and whether every
argument is an exact ``int`` in the signed 64-bit range (read without
running Python code). It never measures time.

- *Warmup.* A warmup call *fits* if it has no keywords, exactly
  ``co_argcount`` positional arguments, and only exact ints in i64. So a
  method called on an instance (``self``), a call relying on a default, a
  keyword call, ``bool``, ``float`` and ``int`` subclasses do not fit.
  With fewer than ``_MIN_FITTING_WARMUP_CALLS`` fitting warmup calls, the
  function is parked (``_ParkedReason.WARMUP``) instead of compiled, and
  neither the frontend nor the compiler is loaded. Only a Python function
  with a hot slot is parked; anything else compiles (or is rejected) as
  before.
- *Compiled.* The hot slot counts the calls in a row that could not run
  natively: keywords, another arity, a guard miss. A native result resets
  the count, so occasional misses and an alternating mix never park. The
  ``MISS_LIMIT``th miss in a row parks the specialization
  (``_ParkedReason.MISSES``).
- *Parked.* Calls run the original function directly. After
  ``REPROBE_INTERVAL`` parked calls, the next call that fits is a
  re-probe: a parked specialization runs it natively and is ``COMPILED``
  again if that gives a result; a function parked during warmup is
  compiled then, at most once in its life. A failed re-probe stays parked
  and starts the interval again. ``__code__`` replacement and a changed
  range binding are still checked on every parked call of a compiled
  function and disable it for good, as before.

``MISS_LIMIT`` and ``REPROBE_INTERVAL`` are constants of the extension
module (``_native``), with their rationale next to them in ``src/lib.rs``.

Guard misses are expected, so they are reported without an exception.
Only an unexpected native error (an ``Exception`` from the fast path)
disables the specialization. ``CompiledI64.__call__``, which explains each
guard miss with a ``NativeGuardFailure``, is not on this path.

The only correctness rule is that ``@snekit`` never changes what a call
returns or raises. Whenever native execution does not apply, the original
function is called exactly once with the original arguments, and whatever
it returns or raises is passed through unchanged. Running it after a native
attempt that stopped part-way (overflow, or an exhausted loop budget) is
sound because native code only computes with exact integers in registers:
it calls nothing and changes no Python object, so the attempt has no
effect that the restart could repeat. It can only repeat the computation.

The extension module is loaded on the first decoration, which needs its
``NativeWrapper`` type, and the compilation modules on the first
compilation attempt, so ``import speedysnek`` stays cheap.
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

# How many of the warmup calls must fit the i64 specialization for the next
# call to compile it; with fewer, it is parked instead. Half: a function
# whose calls mostly fit is compiled at once, and one whose calls mostly do
# not, such as a function only ever called with floats, does not load the
# compiler for code that would only miss. A function parked with a few
# fitting calls can still compile at its first re-probe.
_MIN_FITTING_WARMUP_CALLS = _WARMUP_CALLS // 2


class _Status(enum.Enum):
    """Where a decorated function is in its lifecycle."""

    COLD = "cold"
    COMPILED = "compiled"
    PARKED = "parked"
    DISABLED = "disabled"


class _ParkedReason(enum.Enum):
    """Why a decorated function is ``PARKED``."""

    # Fewer than ``_MIN_FITTING_WARMUP_CALLS`` warmup calls fit; nothing
    # was compiled. See ``_JitState.warmup_fits``.
    WARMUP = "warmup_mismatch"
    # ``_native.MISS_LIMIT`` calls in a row could not use the compiled
    # specialization.
    MISSES = "repeated_misses"


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
    # The function uses ``range``, which could not be proven to be the
    # builtin without running Python code, or is not; see ``compile_error``.
    RANGE_UNGUARDED = "range_unguarded"
    # The function's ``range`` binding may have changed after compilation.
    RANGE_CHANGED = "range_changed"
    # A native call raised an ``Exception``; guard misses never raise. See
    # ``native_error``.
    NATIVE_ERROR = "native_error"


class _Specialization:
    """Everything a native call needs, published as one object.

    The wrapper reads ``_JitState.specialization`` once per call, so it
    either sees a complete specialization or none at all. Everything the
    hot path needs is resolved here, once.

    It is immutable: the wrapper's hot slot holds a copy of the published
    one. A test that needs another specialization assigns a new one to
    ``_JitState.specialization``, which retires the slot.
    """

    __slots__ = ("arg_count", "code", "fast_call", "function", "guard", "native")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"_Specialization.{name} cannot be changed")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"_Specialization.{name} cannot be deleted")

    def __init__(self, code: CodeType, native: Any, guard: Any = None) -> None:
        # The class itself refuses assignments.
        assign = object.__setattr__
        assign(self, "code", code)
        # The ``CompiledI64``, kept for diagnostics; the wrapper never calls it.
        assign(self, "native", native)
        assign(self, "arg_count", native.arg_count)
        # The ``RangeGuard`` the native code depends on, or ``None``.
        assign(self, "guard", guard)
        if guard is None:
            # The ``I64Function`` itself: the wrapper runs its fast path
            # directly, in Rust. Any other value (only test doubles use
            # one) makes the wrapper call ``fast_call`` instead.
            assign(self, "function", native.function)
            # The bound ``I64Function.try_call``: ``fast_call(args)``
            # returns the result ``int``, or ``None`` for a guard miss. The
            # wrapper only calls it for a test double; benchmarks time it
            # as the fast path seen from Python.
            assign(self, "fast_call", native.fast_call)
        else:
            from ._native import GuardedI64Function

            # The wrapper checks the guard, then runs the same fast path;
            # ``try_call`` also treats a failed guard as a guard miss.
            assign(self, "function", GuardedI64Function(native.function, guard))
            assign(self, "fast_call", self.function.try_call)


class _JitState:
    """Per-function JIT state, exposed as the wrapper's ``_jit_state``.

    It is private and exists for tests and debugging.

    Invariant: ``specialization`` is not ``None`` exactly when ``status``
    is ``COMPILED``, or ``PARKED`` after compiling (``parked_reason`` is
    then ``MISSES``). ``compiled`` and ``compiled_code`` keep the last
    successful compilation for inspection even after it is invalidated.

    ``_hot`` is the wrapper's ``_native.HotSlot``, or ``None`` for a state
    used without a ``NativeWrapper``. The compilation that stores the
    specialization publishes the same one into it, after setting
    ``status`` to ``COMPILED``. Every later assignment to
    ``specialization`` retires it first, for good: whether ``_disable``
    makes it, on any path to ``DISABLED``, or a test does. So the slot is
    live only while the wrapper would run exactly this specialization.
    The slot is parked exactly while ``status`` is ``PARKED``: ``_park``
    changes both, and every ``_unpark`` is followed by the new status,
    with the lock held and no Python code in between.

    ``warmup_fits`` counts the warmup calls that could have run natively;
    ``parked_reason`` says why the state is ``PARKED``.
    """

    __slots__ = (
        "_func",
        "_hot",
        "_lock",
        "_specialization",
        "compile_attempts",
        "compile_error",
        "compiled",
        "compiled_code",
        "disabled_reason",
        "native_error",
        "parked_reason",
        "rejection",
        "status",
        "warmup_calls",
        "warmup_fits",
    )

    def __init__(self, func: Callable[..., Any], hot: Any = None) -> None:
        self._func = func
        self._hot = hot
        self._lock = threading.Lock()
        self.status = _Status.COLD
        # The wrapper's generic path reads this slot directly; see
        # ``specialization``.
        self._specialization: _Specialization | None = None
        self.warmup_calls = 0
        self.warmup_fits = 0
        self.parked_reason: _ParkedReason | None = None
        self.compile_attempts = 0
        self.rejection = None
        self.compile_error: Exception | None = None
        self.native_error: Exception | None = None
        self.compiled = None
        self.compiled_code: CodeType | None = None
        self.disabled_reason: _DisabledReason | None = None

    @property
    def specialization(self) -> _Specialization | None:
        """The specialization a native call uses, or ``None``.

        Assigning it retires the hot slot first, so that the wrapper reads
        the new value. Only the compilation that publishes the slot stores
        ``_specialization`` without retiring it.
        """
        return self._specialization

    @specialization.setter
    def specialization(self, value: _Specialization | None) -> None:
        if self._hot is not None:
            self._hot.retire()
        self._specialization = value

    def cold_call(self, fits: int | None = None) -> _Specialization | None:
        """Count one call while ``COLD``; decide once warmup is over.

        ``fits`` is the wrapper's profile of the call: its number of
        positional arguments if it has no keywords and every argument is an
        exact ``int`` in the signed 64-bit range, else -1. ``None``, from a
        caller that does not profile, counts as a call that fits.

        Returns the specialization to try for this call, if there is one.
        """
        # Not atomic across threads: a lost update only delays compilation
        # or shifts the profile by one call.
        self.warmup_calls += 1
        if self.warmup_calls <= _WARMUP_CALLS:
            # ``FunctionType`` cannot be subclassed, so reading
            # ``__code__.co_argcount`` runs no Python code.
            func = self._func
            if fits is None or (
                fits >= 0
                and type(func) is FunctionType
                and fits == func.__code__.co_argcount
            ):
                self.warmup_fits += 1
            return None
        with self._lock:
            # Another thread may have finished compiling while we waited.
            if self.status is _Status.COLD:
                if self._parkable() and self.warmup_fits < _MIN_FITTING_WARMUP_CALLS:
                    self._park(_ParkedReason.WARMUP)
                else:
                    self._compile()
        return self._specialization

    def repeated_misses(self) -> None:
        """``MISS_LIMIT`` calls in a row could not run natively."""
        with self._lock:
            if self.status is _Status.COMPILED and self._specialization is not None:
                self._park(_ParkedReason.MISSES)

    def reprobe_succeeded(self) -> None:
        """A re-probe of the parked specialization ran natively."""
        with self._lock:
            if self.status is _Status.PARKED and self._specialization is not None:
                self._unpark()
                self.status = _Status.COMPILED

    def reprobe(self, args: int) -> _Specialization | None:
        """A call to a function parked during warmup may fit: compile it.

        The wrapper has checked that the call has no keywords and only
        exact ints in i64; ``args`` is their number. Returns the
        specialization to try for this call, if there is one.
        """
        # Parked only when ``FunctionType``, so this runs no Python code.
        if args != self._func.__code__.co_argcount:
            return None
        with self._lock:
            # Compiled at most once: a parked state without a
            # specialization has never compiled.
            if self.status is _Status.PARKED and self.compile_attempts == 0:
                self._unpark()
                # ``_compile`` treats a state left ``COLD`` as interrupted.
                self.status = _Status.COLD
                self._compile()
        return self._specialization

    def invalidate(self) -> None:
        """``__code__`` no longer matches the compiled code object."""
        with self._lock:
            if self._has_native():
                self._disable(_DisabledReason.CODE_CHANGED)

    def binding_changed(self) -> None:
        """The range guard failed: ``range`` may no longer be the builtin."""
        with self._lock:
            if self._has_native():
                self._disable(_DisabledReason.RANGE_CHANGED)

    def native_failed(self, error: Exception) -> None:
        """A native call raised; guard misses return ``None`` instead."""
        with self._lock:
            if self._has_native():
                self.native_error = error
                self._disable(_DisabledReason.NATIVE_ERROR)

    def _has_native(self) -> bool:
        # ``COMPILED``, or ``PARKED`` with compiled code.
        return self.status is _Status.COMPILED or (
            self.status is _Status.PARKED and self._specialization is not None
        )

    def _parkable(self) -> bool:
        # Only the hot slot parks and re-probes, and only for a function.
        return self._hot is not None and type(self._func) is FunctionType

    def _park(self, reason: _ParkedReason) -> None:
        # Called with the lock held.
        self.status = _Status.PARKED
        self.parked_reason = reason
        self._hot.park()

    def _unpark(self) -> None:
        # Called with the lock held; the caller sets the new status.
        self.parked_reason = None
        self._hot.unpark()

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

        guard = None
        if "range" in analysis.assumed_builtins:
            from ._native import RangeGuard, UnguardableBinding

            try:
                # ``FunctionType`` cannot be subclassed, so reading these
                # attributes runs no Python code. Created before compiling,
                # so the guard sees every change from here on.
                guard = RangeGuard(func.__globals__, func.__builtins__)
            except UnguardableBinding as error:
                self.compile_error = error
                self._disable(_DisabledReason.RANGE_UNGUARDED)
                return

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
        specialization = _Specialization(code, compiled, guard)
        # Publish the complete specialization before changing the status.
        # This is the one assignment that does not retire the hot slot.
        self._specialization = specialization
        self.status = _Status.COMPILED
        if self._hot is not None:
            # Last: the slot is only ever live while ``COMPILED``.
            self._hot.publish(code, specialization.function)

    def _disable(self, reason: _DisabledReason) -> None:
        # Retires the hot slot first (see ``specialization``).
        self.specialization = None
        self.status = _Status.DISABLED
        self.disabled_reason = reason
        self.parked_reason = None


def snekit[F: Callable[..., Any]](func: F) -> F:
    """Compile ``func`` to native code once it is hot, where possible.

    The returned wrapper behaves exactly like ``func``: same results, same
    exceptions, same metadata (via :func:`functools.wraps`, so the original
    function is ``__wrapped__``), and it binds as a method like a function.
    After a short warmup it tries, once, to compile ``func`` to signed
    64-bit machine code. Calls that the compiled code cannot handle, and
    functions that cannot be compiled at all, simply run ``func``.

    The wrapper is a native callable object, not a Python function: it has
    no ``__code__``, ``__defaults__`` or ``__globals__`` of its own, and
    :func:`inspect.isfunction` is false for it.

    The wrapper exposes ``call_count``: the number of times it has been
    called, including calls that raised. It is an exact ``int`` with no
    upper bound, and it can be set to any integer (anything
    :func:`operator.index` accepts).

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
    # Loads the extension module, but not the compiler.
    from ._native import HotSlot, NativeWrapper

    hot = HotSlot()
    state = _JitState(func, hot)
    wrapper = NativeWrapper(func, state, _Status.COLD, hot)
    functools.update_wrapper(wrapper, func)
    # Private diagnostics only; the wrapper holds ``state`` itself.
    wrapper._jit_state = state
    return wrapper  # type: ignore[return-value]


# The descriptive, backwards-compatible name. It is the same object, so
# ``@jit`` and ``@snekit`` behave identically.
jit = snekit
