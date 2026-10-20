"""``peek``: report what SpeedySnek did with a decorated function.

``peek`` is diagnostics only. It reads the state a ``@snekit`` wrapper
already keeps (its ``call_count``, its ``_JitState`` and, once compiled,
the native function's own attempt counter) and turns it into a small,
stable report::

    import calculations
    import speedysnek as snek

    snek.bite(calculations)
    ...
    report = snek.peek(calculations.sum_squares)

It never calls, decorates, compiles, warms up or replaces anything, and it
changes no counter or state. It adds nothing to the call path: every number
it reports already exists. Nothing in it is a measurement of speed.

Like ``hunt`` and ``bite``, ``peek`` runs no user code: modules and types
are read through their own descriptors, and a wrapper is trusted only if
it is an exact ``NativeWrapper``. Its JIT state is the one the wrapper
itself holds, found through the wrapper's garbage-collector references
(which its Rust ``__traverse__`` reports without running Python code),
not its ``_jit_state`` or ``__wrapped__`` attributes, which anyone can
rewrite. The extension and compiler modules are only consulted if they
are already loaded, never imported: no wrapper can exist before the
extension is loaded, and no compiled code before the compiler is.
"""

from __future__ import annotations

import gc
import sys
from collections.abc import Iterable, Iterator
from types import FunctionType, MethodType, ModuleType
from typing import Any, TextIO

from ._decorator import _WARMUP_CALLS, _DisabledReason, _JitState, _Status
from ._hunt import _MODULE_DICT, _TYPE_NAME, _explain, _is_snekit_wrapper, _write

COLD = "cold"
COMPILED = "compiled"
DISABLED = "disabled"
NOT_BITTEN = "not_bitten"

# The only compiled code the decorator produces today.
_CRANELIFT_I64 = "Cranelift i64 specialization"

_EMPTY = "🐍 Snek found no bitten functions"

# Read directly, so an exception subclass cannot intercept it.
_EXCEPTION_ARGS = BaseException.__dict__["args"]

_NOT_BITTEN_SUMMARY = "not wrapped with @snekit; use snek.bite(module) or @snekit first"

# Plain-English summaries of why a wrapper was disabled. Rejections and
# errors get their own text, built from the stored diagnostic.
_DISABLED_SUMMARIES = {
    _DisabledReason.INTERRUPTED: "compilation was interrupted (for example by Ctrl+C)",
    _DisabledReason.CODE_CHANGED: "its __code__ was replaced after compilation",
    _DisabledReason.RANGE_UNGUARDED: "range could not be proven to be the builtin",
    _DisabledReason.RANGE_CHANGED: "its range binding changed after compilation",
}

# The labels ``render`` uses, all padded to one width.
_WIDTH = len("compile attempts: ")


class PeekFinding:
    """What :func:`peek` found out about one function.

    Every count is read from state the wrapper already keeps; a field is
    ``None`` when that state does not exist or cannot be read safely.

    - ``name``: the function's name (its first public name, for a module).
    - ``names``: every public module name bound to the wrapper, ``name``
      first; just ``(name,)`` outside a module.
    - ``wrapper``: the inspected ``@snekit`` wrapper; ``None`` for a
      function that is not bitten.
    - ``function``: the Python function the wrapper runs, read from its JIT
      state, not ``__wrapped__``; for a function that is not bitten, the
      function itself.
    - ``status``: ``"cold"``, ``"compiled"``, ``"disabled"`` or
      ``"not_bitten"``.
    - ``summary``: a short, human-readable explanation.
    - ``call_count``: the wrapper's ``call_count``: calls it has observed,
      raising ones included. Users can assign it, so it is not proof.
    - ``warmup_calls``: calls that ran as plain Python during warmup, at
      most the warmup length (8).
    - ``warmup_remaining``: while cold, how many more warmup calls run
      before the next one tries to compile; ``None`` otherwise.
    - ``compile_attempts``: compilations attempted, 0 or 1.
    - ``native_attempts``: calls that entered the compiled code's fast path,
      counted by the native function itself. Each one validated its
      arguments and either returned a native result or fell back to Python
      (a non-``int`` argument, overflow, an exhausted loop budget), so this
      is not a count of native results. ``None`` before compilation.
    - ``compiler``: what compilation produced, such as ``"Cranelift i64
      specialization"``; ``None`` if nothing was compiled.
    - ``reason``: when disabled, the stable reason code, such as
      ``"rejected"`` or ``"code_changed"``; ``None`` otherwise.
    - ``detail``: when disabled, the underlying diagnostic message, where
      there is one.
    - ``lineno``: the source line of a rejected instruction, where known.
    - ``rejection``: the frontend's ``Rejection`` when it rejected the
      function; ``compile_error`` and ``native_error``: the stored
      exception when compilation or a native call failed.
    """

    __slots__ = (
        "call_count",
        "compile_attempts",
        "compile_error",
        "compiler",
        "detail",
        "function",
        "lineno",
        "name",
        "names",
        "native_attempts",
        "native_error",
        "reason",
        "rejection",
        "status",
        "summary",
        "warmup_calls",
        "warmup_remaining",
        "wrapper",
    )

    def __init__(
        self,
        name: str,
        function: Any,
        status: str,
        summary: str,
        *,
        wrapper: Any = None,
        names: Iterable[str] = (),
        call_count: int | None = None,
        warmup_calls: int | None = None,
        warmup_remaining: int | None = None,
        compile_attempts: int | None = None,
        native_attempts: int | None = None,
        compiler: str | None = None,
        reason: str | None = None,
        detail: str | None = None,
        lineno: int | None = None,
        rejection: Any = None,
        compile_error: BaseException | None = None,
        native_error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.names = tuple(names) or (name,)
        self.function = function
        self.wrapper = wrapper
        self.status = status
        self.summary = summary
        self.call_count = call_count
        self.warmup_calls = warmup_calls
        self.warmup_remaining = warmup_remaining
        self.compile_attempts = compile_attempts
        self.native_attempts = native_attempts
        self.compiler = compiler
        self.reason = reason
        self.detail = detail
        self.lineno = lineno
        self.rejection = rejection
        self.compile_error = compile_error
        self.native_error = native_error

    def render(self) -> str:
        """The finding's fields, one per line, without its name."""
        rows = [("status", self.status.replace("_", " "))]
        if len(self.names) > 1:
            rows.append(("aliases", ", ".join(self.names[1:])))
        if self.call_count is not None:
            rows.append(("calls observed", str(self.call_count)))
        if self.warmup_calls is not None:
            warmup = str(self.warmup_calls)
            if self.warmup_remaining is not None:
                warmup += f" of {_WARMUP_CALLS}"
                if self.warmup_remaining:
                    warmup += f", {self.warmup_remaining} to go"
                else:
                    warmup += ", the next call compiles"
            rows.append(("warmup calls", warmup))
        if self.compile_attempts is not None:
            rows.append(("compile attempts", str(self.compile_attempts)))
        if self.native_attempts is not None:
            rows.append(("native attempts", str(self.native_attempts)))
        if self.compiler is not None:
            compiler = self.compiler
            if self.status != COMPILED:
                compiler += " (no longer used)"
            rows.append(("compiler", compiler))
        if self.status == DISABLED:
            reason = self.summary
            if self.lineno is not None:
                reason += f" (line {self.lineno})"
            rows.append(("reason", reason))
            # A rejection's message repeats its summary.
            if self.detail is not None and self.rejection is None:
                rows.append(("detail", self.detail))
        elif self.status == NOT_BITTEN:
            rows.append(("note", self.summary))
        return "\n".join(f"{f'{label}:':<{_WIDTH}}{value}" for label, value in rows)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return (
            f"PeekFinding(name={self.name!r}, status={self.status!r}, "
            f"summary={self.summary!r})"
        )


class PeekReport:
    """The result of :func:`peek`: one :class:`PeekFinding` per wrapper.

    ``findings`` keeps definition order for a module. ``module_name`` is the
    peeked module's name, or ``None`` when a function was peeked at.
    Iterating the report yields the findings; ``str(report)`` is the same
    text that :func:`peek` prints.
    """

    __slots__ = ("findings", "module_name")

    def __init__(
        self, findings: Iterable[PeekFinding], module_name: str | None = None
    ) -> None:
        self.findings = tuple(findings)
        self.module_name = module_name

    def _with(self, status: str) -> tuple[PeekFinding, ...]:
        return tuple(finding for finding in self.findings if finding.status == status)

    @property
    def cold(self) -> tuple[PeekFinding, ...]:
        """The findings still warming up."""
        return self._with(COLD)

    @property
    def compiled(self) -> tuple[PeekFinding, ...]:
        """The findings with compiled code in use."""
        return self._with(COMPILED)

    @property
    def disabled(self) -> tuple[PeekFinding, ...]:
        """The findings that always run as Python from now on."""
        return self._with(DISABLED)

    @property
    def not_bitten(self) -> tuple[PeekFinding, ...]:
        """The findings for functions without a ``@snekit`` wrapper."""
        return self._with(NOT_BITTEN)

    def __iter__(self) -> Iterator[PeekFinding]:
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    def render(self) -> str:
        """The human-readable report, without a trailing newline."""
        if self.module_name is None:
            if not self.findings:
                return f"{_EMPTY}."
            if len(self.findings) == 1:
                (finding,) = self.findings
                return f"🐍 Snek peeked at {finding.name}.\n\n{finding.render()}"
        elif not self.findings:
            return f"{_EMPTY} in {self.module_name}."
        lines = [f"🐍 Snek peeked at {self.module_name or 'some functions'}."]
        for finding in self.findings:
            lines += ["", finding.name]
            lines += [f"  {line}" for line in finding.render().splitlines()]
        counts = (
            f"{len(self.compiled)} compiled, {len(self.cold)} cold, "
            f"{len(self.disabled)} disabled."
        )
        lines += ["", counts]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        counts = ", ".join(
            f"{len(self._with(status))} {status}"
            for status in (COMPILED, COLD, DISABLED, NOT_BITTEN)
            if self._with(status)
        )
        return f"<PeekReport: {counts or 'empty'}>"


def peek(
    target: Any, *, quiet: bool = False, stream: TextIO | None = None
) -> PeekReport:
    """Report what SpeedySnek has done with a function so far.

    ::

        snek.bite(calculations)
        ...
        report = snek.peek(calculations.sum_squares)
        report = snek.peek(calculations)

    Pass a ``@snekit`` wrapper (including one installed by :func:`bite`, or
    a method bound from one) to get one finding: whether it is ``"cold"``
    (warming up), ``"compiled"`` or ``"disabled"`` (and why), with the
    counts its state holds. A plain Python function gets one
    ``"not_bitten"`` finding.

    Pass a module to get one finding per ``@snekit`` wrapper of the module's
    own functions under a public name, in definition order. Aliases of one
    wrapper share a finding. Plain functions, imported wrappers, private
    names, classes and callable objects are left out; a module without
    wrappers gives an empty report.

    The report is printed to ``stream`` (``sys.stdout`` by default) unless
    ``quiet`` is true, and returned as a :class:`PeekReport`. Nothing is
    called, compiled or changed, and nothing is timed: the report describes
    state, not speed.

    Raises :class:`TypeError` for anything else, including objects that
    merely look like a wrapper.
    """
    kind = type(target)
    if issubclass(kind, ModuleType):
        report = _peek_module(target)
    else:
        report = PeekReport([_peek_one(target)])
    if not quiet:
        _write(sys.stdout if stream is None else stream, report.render())
    return report


def _peek_one(target: Any) -> PeekFinding:
    kind = type(target)
    if kind is MethodType:
        # A member of an exact type: reading it runs no Python code.
        inner = target.__func__
        if type(inner) is FunctionType or _is_snekit_wrapper(type(inner)):
            target, kind = inner, type(inner)
    if kind is FunctionType:
        return _not_bitten(target)
    if _is_snekit_wrapper(kind):
        state = _wrapper_state(target)
        if state is not None:
            finding = _finding(_wrapper_name(target, state), target, state)
            if finding is not None:
                return finding
        raise TypeError("peek() expected a @snekit wrapper, got an unrecognized one")
    hint = ""
    if issubclass(kind, (str, bytes)):
        hint = "; to peek at a module, import it and pass the module itself"
    raise TypeError(
        f"peek() expected a @snekit wrapper, a Python function or a module, "
        f"got {_TYPE_NAME.__get__(kind)}{hint}"
    )


def _peek_module(module: ModuleType) -> PeekReport:
    namespace = _MODULE_DICT.__get__(module)
    module_name = namespace.get("__name__")
    if type(module_name) is not str:
        return PeekReport([], "<module>")
    # Grouped by wrapper: (wrapper, state, names), in definition order.
    found: dict[int, tuple[Any, _JitState, list[str]]] = {}
    for key, value in _wrappers(namespace):
        entry = found.get(id(value))
        if entry is not None:
            entry[2].append(key)
            continue
        state = _wrapper_state(value)
        if state is None:
            continue
        function = state._func
        # ``FunctionType`` cannot be subclassed, and its ``__module__`` is a
        # plain member: reading it runs no Python code.
        if type(function) is not FunctionType:
            continue
        defined_in = function.__module__
        if type(defined_in) is str and defined_in == module_name:
            found[id(value)] = (value, state, [key])
    findings = []
    for wrapper, state, names in found.values():
        finding = _finding(names[0], wrapper, state, names)
        if finding is not None:
            findings.append(finding)
    return PeekReport(findings, module_name)


def _wrappers(namespace: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    """The ``@snekit`` wrappers under public names, in definition order."""
    if "speedysnek._native" not in sys.modules:
        # No wrapper can exist yet; checked first so it stays unloaded.
        return
    for key, value in tuple(namespace.items()):
        if type(key) is str and not key.startswith("_"):
            if _is_snekit_wrapper(type(value)):
                yield key, value


def _wrapper_state(wrapper: Any) -> _JitState | None:
    """The JIT state ``wrapper`` itself holds, or ``None``.

    The wrapper's ``_jit_state`` attribute is only a copy, which anyone
    can replace; the native wrapper reports the state it really uses to the
    garbage collector, alongside its function and a few other objects, none
    of them a ``_JitState``.
    """
    states = [ref for ref in gc.get_referents(wrapper) if type(ref) is _JitState]
    if len(states) != 1:
        return None
    return states[0]


def _wrapper_name(wrapper: Any, state: _JitState) -> str:
    """The wrapper's ``__name__``, read without running Python code."""
    attributes = wrapper.__dict__
    if type(attributes) is dict:
        name = dict.get(attributes, "__name__")
        if type(name) is str:
            return name
    function = state._func
    if type(function) is FunctionType:
        return function.__name__
    return "<snekit wrapper>"


def _finding(
    name: str, wrapper: Any, state: _JitState, names: Iterable[str] = ()
) -> PeekFinding | None:
    """The finding for a genuine wrapper, or ``None`` if its state is odd."""
    status = state.status
    if status is _Status.COLD:
        status = COLD
    elif status is _Status.COMPILED:
        status = COMPILED
    elif status is _Status.DISABLED:
        status = DISABLED
    else:
        return None

    warmup = _count(state.warmup_calls)
    warmup_remaining = None
    if warmup is not None:
        warmup = min(warmup, _WARMUP_CALLS)
        if status == COLD:
            warmup_remaining = _WARMUP_CALLS - warmup
    native_attempts = compiler = None
    # While another thread compiles, ``compiled`` is set just before the
    # status changes: only trust it once the wrapper has left ``COLD``.
    native_function = None if status == COLD else _native_function(state.compiled)
    if native_function is not None:
        native_attempts = native_function.try_calls
        compiler = _CRANELIFT_I64

    reason = detail = lineno = rejection = compile_error = native_error = None
    if status == COLD:
        if warmup_remaining:
            calls = "call" if warmup_remaining == 1 else "calls"
            summary = f"warming up: {warmup_remaining} more {calls} before compiling"
        else:
            summary = "warmed up: the next call tries to compile it"
    elif status == COMPILED:
        summary = "compiled: calls run natively when their arguments fit"
    else:
        disabled = state.disabled_reason
        summary = "disabled"
        if type(disabled) is _DisabledReason:
            reason = disabled.value
            summary = _DISABLED_SUMMARIES.get(disabled, summary)
        if disabled is _DisabledReason.REJECTED:
            rejection = _rejection(state.rejection)
            if rejection is not None:
                summary = _explain(rejection)
                detail = rejection.message
                lineno = rejection.lineno
            else:
                summary = "the compiler frontend rejected it"
        elif (
            disabled is _DisabledReason.UNSUPPORTED
            or disabled is _DisabledReason.COMPILER_ERROR
            or disabled is _DisabledReason.RANGE_UNGUARDED
        ):
            compile_error = _exception(state.compile_error)
            detail = _message(compile_error)
            if disabled is _DisabledReason.UNSUPPORTED:
                summary = "beyond the native compiler's limits"
            elif disabled is _DisabledReason.COMPILER_ERROR:
                summary = "the compiler failed unexpectedly"
                if compile_error is not None:
                    summary += f" ({_TYPE_NAME.__get__(type(compile_error))})"
        elif disabled is _DisabledReason.NATIVE_ERROR:
            native_error = _exception(state.native_error)
            detail = _message(native_error)
            summary = "a native call failed unexpectedly"
            if native_error is not None:
                summary += f" ({_TYPE_NAME.__get__(type(native_error))})"

    function = state._func
    return PeekFinding(
        name,
        function,
        status,
        summary,
        wrapper=wrapper,
        names=names,
        call_count=_count(wrapper.call_count),
        warmup_calls=warmup,
        warmup_remaining=warmup_remaining,
        compile_attempts=_count(state.compile_attempts),
        native_attempts=_count(native_attempts),
        compiler=compiler,
        reason=reason,
        detail=detail,
        lineno=lineno if type(lineno) is int else None,
        rejection=rejection,
        compile_error=compile_error,
        native_error=native_error,
    )


def _not_bitten(function: FunctionType) -> PeekFinding:
    summary = _NOT_BITTEN_SUMMARY
    holder = _module_wrapper_of(function)
    if holder is not None:
        # A reference taken before ``bite``: the module has moved on.
        summary = f"not wrapped itself, but {holder} is its @snekit wrapper"
    return PeekFinding(function.__name__, function, NOT_BITTEN, summary)


def _module_wrapper_of(function: FunctionType) -> str | None:
    """``module.name`` of a public wrapper of ``function`` in its module."""
    module_name = function.__module__
    modules = sys.modules
    if type(module_name) is not str or type(modules) is not dict:
        return None
    module = dict.get(modules, module_name)
    if not issubclass(type(module), ModuleType):
        return None
    for key, value in _wrappers(_MODULE_DICT.__get__(module)):
        state = _wrapper_state(value)
        if state is not None and state._func is function:
            return f"{module_name}.{key}"
    return None


def _native_function(compiled: Any) -> Any:
    """The native ``I64Function`` of a ``CompiledI64``, or ``None``."""
    compiler = sys.modules.get("speedysnek._compiler")
    native = sys.modules.get("speedysnek._native")
    if compiler is None or native is None:
        return None
    if type(compiled) is not compiler.CompiledI64:
        return None
    function = compiled._function
    return function if type(function) is native.I64Function else None


def _rejection(rejection: Any) -> Any:
    """The frontend's ``Rejection``, if ``rejection`` is one."""
    ir = sys.modules.get("speedysnek._frontend.ir")
    if ir is None or type(rejection) is not ir.Rejection:
        return None
    return rejection


def _exception(error: Any) -> BaseException | None:
    return error if issubclass(type(error), BaseException) else None


def _message(error: BaseException | None) -> str | None:
    """An exception's message, read without calling its ``__str__``."""
    if error is None:
        return None
    args = _EXCEPTION_ARGS.__get__(error)
    if type(args) is tuple and len(args) == 1 and type(args[0]) is str:
        return args[0]
    return _TYPE_NAME.__get__(type(error))


def _count(value: Any) -> int | None:
    return value if type(value) is int else None
