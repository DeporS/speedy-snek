"""``bite``: apply ``@snekit`` to the ready functions a module defines.

``bite`` is the explicit, opt-in step after :func:`speedysnek.hunt`::

    import my_calculations
    import speedysnek as snek

    snek.hunt(my_calculations)  # inspect first
    report = snek.bite(my_calculations)

It finds candidates with the same rules as ``hunt``, inspects them with
the same frontend analysis, and replaces the module's public names for
every ready function with one ``@snekit`` wrapper. Nothing is called or
compiled: the wrapper warms up and compiles exactly as if the function had
been decorated in its source. Functions marked with ``@nosnek`` are
skipped without being analyzed or wrapped, and every name bound to them
is left alone.

``bite`` changes the module's namespace only. Every public name in the
module bound to a ready function, aliases included, is rebound to the same
wrapper; references held anywhere else, captured before ``bite``, still
point to the original function. Function objects are never modified.

The module is changed all at once or not at all: every candidate is
inspected and every wrapper built first, then every name to replace is
checked to still hold its original function, and only then are all of
them replaced in one update. If any has changed, nothing is replaced. Like
``hunt``, ``bite`` reads the module through ``ModuleType``'s own
``__dict__`` descriptor and runs no user code. The native extension is
loaded only when there is at least one function to wrap.

``bite`` remembers the wrappers it installed, so that
:func:`speedysnek.chill` can undo exactly those.
"""

from __future__ import annotations

import sys
import weakref
from collections.abc import Iterable, Iterator
from types import FunctionType, ModuleType
from typing import Any, TextIO

from ._decorator import _JitState, snekit
from ._hunt import (
    _MODULE_DICT,
    _TYPE_NAME,
    HuntFinding,
    _inspect,
    _is_snekit_wrapper,
    _opted_out,
    _write,
)
from ._nosnek import _is_opted_out

BITTEN = "bitten"
SKIPPED = "skipped"
ALREADY_BITTEN = "already_bitten"

_HEADER = "🐍 Snek took a bite."
_NOTHING_NEW = "🐍 Snek found nothing new to bite."
_EMPTY = "🐍 Snek found no functions to bite."

# Builds each wrapper. A seam for tests; always ``snekit`` otherwise.
_wrap = snekit

# Every wrapper ``bite`` installed in a module, mapped to the function it
# wraps, so that ``chill`` undoes only ``bite``'s own work. Weak, so it
# never keeps a wrapper alive; the wrapper already holds its function.
_INSTALLED: weakref.WeakKeyDictionary[Any, FunctionType] = weakref.WeakKeyDictionary()


class BiteFinding:
    """What :func:`bite` did with one function.

    - ``name``: the function's first public name in the module.
    - ``names``: every public name in the module bound to the function or
      to its ``@snekit`` wrapper, in definition order (``name`` first).
    - ``function``: the original Python function.
    - ``wrapper``: the ``@snekit`` wrapper the module's names now hold: the
      new one for ``"bitten"``, the existing one for ``"already_bitten"``;
      ``None`` when skipped.
    - ``status``: ``"bitten"``, ``"skipped"`` or ``"already_bitten"``.
    - ``summary``: a short, human-readable explanation.
    - ``hunt``: the :class:`HuntFinding` for the function, or ``None`` for
      an already bitten one, which is not inspected again. Its ``reason``
      says why a function was skipped: ``"opted_out"`` for one marked with
      ``@nosnek``, otherwise the frontend's rejection code.
    """

    __slots__ = ("function", "hunt", "name", "names", "status", "summary", "wrapper")

    def __init__(
        self,
        name: str,
        function: FunctionType,
        status: str,
        summary: str,
        wrapper: Any = None,
        hunt: HuntFinding | None = None,
        names: Iterable[str] = (),
    ) -> None:
        self.name = name
        self.function = function
        self.status = status
        self.summary = summary
        self.wrapper = wrapper
        self.hunt = hunt
        self.names = tuple(names) or (name,)

    @property
    def bitten(self) -> bool:
        return self.status == BITTEN

    def __repr__(self) -> str:
        return (
            f"BiteFinding(name={self.name!r}, status={self.status!r}, "
            f"summary={self.summary!r})"
        )


class BiteReport:
    """The result of :func:`bite`: one :class:`BiteFinding` per function.

    ``findings`` keeps the module's definition order. Iterating the report
    yields the findings; ``str(report)`` is the same text that :func:`bite`
    prints.
    """

    __slots__ = ("findings",)

    def __init__(self, findings: Iterable[BiteFinding]) -> None:
        self.findings = tuple(findings)

    def _with(self, status: str) -> tuple[BiteFinding, ...]:
        return tuple(finding for finding in self.findings if finding.status == status)

    @property
    def bitten(self) -> tuple[BiteFinding, ...]:
        """The findings whose module names now hold a new wrapper."""
        return self._with(BITTEN)

    @property
    def skipped(self) -> tuple[BiteFinding, ...]:
        """The findings left unchanged: not compilable, or marked @nosnek."""
        return self._with(SKIPPED)

    @property
    def already_bitten(self) -> tuple[BiteFinding, ...]:
        """The findings that already had a ``@snekit`` wrapper."""
        return self._with(ALREADY_BITTEN)

    @property
    def bitten_count(self) -> int:
        return len(self.bitten)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def already_bitten_count(self) -> int:
        return len(self.already_bitten)

    def __iter__(self) -> Iterator[BiteFinding]:
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    def render(self) -> str:
        """The human-readable report, without a trailing newline."""
        if not self.findings:
            return _EMPTY
        rows = []
        for finding in self.findings:
            summary = finding.summary
            if finding.status == SKIPPED and finding.hunt is not None:
                lineno = finding.hunt.lineno
                if lineno is not None:
                    summary += f" (line {lineno})"
            rows.append((finding.name, finding.status.replace("_", " "), summary))
        name_width = max(len(name) for name, _, _ in rows)
        status_width = max(len(status) for _, status, _ in rows)
        lines = [_HEADER if self.bitten_count else _NOTHING_NEW, ""]
        for name, status, summary in rows:
            lines.append(f"{name:<{name_width}}  {status:<{status_width}}  {summary}")
        bitten = self.bitten_count
        counts = f"{bitten} function{'' if bitten == 1 else 's'} bitten"
        counts += f", {self.skipped_count} skipped"
        counts += f", {self.already_bitten_count} already bitten."
        lines += ["", counts]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return (
            f"<BiteReport: {self.bitten_count} bitten, {self.skipped_count} "
            f"skipped, {self.already_bitten_count} already bitten>"
        )


class _Candidate:
    """One function and every public module name bound to it or its wrapper.

    ``status``, ``wrapper`` and ``hunt`` record what ``bite`` plans for it.
    """

    __slots__ = (
        "function",
        "hunt",
        "names",
        "plain_names",
        "status",
        "wrapper",
        "wrappers",
    )

    def __init__(self, function: FunctionType) -> None:
        self.function = function
        # Every public name, in definition order.
        self.names: list[str] = []
        # The names still bound to the plain function.
        self.plain_names: list[str] = []
        # The distinct ``@snekit`` wrappers of the function, with a name.
        self.wrappers: list[tuple[str, Any]] = []
        self.status = SKIPPED
        # The wrapper every plain name will hold: a new or an existing one.
        self.wrapper: Any = None
        self.hunt: HuntFinding | None = None


def bite(
    module: ModuleType, *, quiet: bool = False, stream: TextIO | None = None
) -> BiteReport:
    """Wrap the ready functions ``module`` defines with ``@snekit``.

    ::

        snek.hunt(my_module)  # inspect first
        report = snek.bite(my_module)

    The candidates are the ones :func:`hunt` reports for the module: its
    own public Python functions, in definition order. Each function the
    compiler frontend accepts gets one new ``@snekit`` wrapper, and every
    public name in the module bound to it, aliases included, is rebound to
    that wrapper. Functions ``hunt`` would skip are left as they are. A
    function that already has a ``@snekit`` wrapper in the module is
    reported as ``"already_bitten"`` and never wrapped again; any public
    name still holding the plain function is pointed at that existing
    wrapper. Calling ``bite`` twice changes nothing the second time.

    A function marked with :func:`speedysnek.nosnek` is reported as
    skipped, with the reason ``"opted_out"``: it is not analyzed or
    wrapped, no name bound to it changes, and a ``@snekit`` wrapper of it
    that already exists is left as it is. Skipping it never stops the
    other functions from being bitten.

    The module is updated all at once or not at all: if any name to be
    replaced no longer holds its original function by the time the
    wrappers are ready, no name is replaced and the affected functions are
    reported as skipped.

    Only the module's names change. References taken before ``bite``, such
    as ``from my_module import f`` elsewhere, still hold the original
    function. Nothing is called or compiled: each wrapper runs the original
    function during warmup and compiles it afterwards, as ``@snekit`` does,
    and falls back to Python whenever native execution does not apply.

    The report is printed to ``stream`` (``sys.stdout`` by default) unless
    ``quiet`` is true, and returned as a :class:`BiteReport`.

    Raises :class:`TypeError` if ``module`` is not a module.
    """
    if not issubclass(type(module), ModuleType):
        kind = _TYPE_NAME.__get__(type(module))
        hint = ""
        if issubclass(type(module), (str, bytes)):
            hint = "; import the module and pass the module itself"
        raise TypeError(f"bite() expected a module, got {kind}{hint}")

    namespace = _MODULE_DICT.__get__(module)
    candidates = _module_candidates(namespace)

    # Decide everything before changing anything.
    analyze = None
    for candidate in candidates:
        if _is_opted_out(candidate.function):
            # Left exactly as it is, wrappers and plain names alike.
            candidate.hunt = _opted_out(candidate.names[0], candidate.function)
            continue
        if candidate.wrappers:
            # Never wrap twice: plain aliases get the existing wrapper.
            candidate.status = ALREADY_BITTEN
            candidate.wrapper = candidate.wrappers[0][1]
            continue
        if analyze is None:
            from ._frontend import analyze
        candidate.hunt = _inspect(candidate.names[0], candidate.function, analyze)
        candidate.status = BITTEN if candidate.hunt.ready else SKIPPED

    # Build every wrapper before replacing any name: if one fails, the
    # module is left exactly as it was.
    to_wrap = [candidate for candidate in candidates if candidate.status == BITTEN]
    wrappers = [_wrap(candidate.function) for candidate in to_wrap]
    for candidate, wrapper in zip(to_wrap, wrappers, strict=True):
        candidate.wrapper = wrapper

    # Every name to replace must still hold the original function. If any
    # does not, the module changed under us: replace nothing at all.
    updates: dict[str, Any] = {}
    changed: list[_Candidate] = []
    for candidate in candidates:
        if candidate.status == SKIPPED or not candidate.plain_names:
            continue
        for key in candidate.plain_names:
            if namespace.get(key) is not candidate.function:
                changed.append(candidate)
                break
            updates[key] = candidate.wrapper
    if not changed:
        # One update of the real namespace: no module hooks, and every
        # alias changes together.
        dict.update(namespace, updates)
        for candidate in to_wrap:
            _INSTALLED[candidate.wrapper] = candidate.function

    findings = [_finding(candidate, changed) for candidate in candidates]
    report = BiteReport(findings)
    if not quiet:
        _write(sys.stdout if stream is None else stream, report.render())
    return report


def _module_candidates(namespace: dict[str, Any]) -> list[_Candidate]:
    """The module's own public functions, by first public name.

    Like ``hunt``'s module discovery, but a function also counts when one
    of its names holds its ``@snekit`` wrapper, and every name is kept.
    """
    module_name = namespace.get("__name__")
    if type(module_name) is not str:
        return []
    candidates: dict[int, _Candidate] = {}
    # Wrappers under private names: not reported, but they still count as
    # the function's wrapper, so it is not wrapped a second time.
    private_wrappers: dict[int, tuple[str, Any]] = {}
    for key, value in tuple(namespace.items()):
        if type(key) is not str:
            continue
        kind = type(value)
        if kind is FunctionType:
            if key.startswith("_"):
                continue
            function = value
        elif _is_snekit_wrapper(kind):
            function = _wrapped_function(value)
            if function is None:
                continue
            if key.startswith("_"):
                private_wrappers.setdefault(id(function), (key, value))
                continue
        else:
            continue
        # ``FunctionType`` cannot be subclassed, and its ``__module__`` is a
        # plain member: reading it runs no Python code.
        defined_in = function.__module__
        if type(defined_in) is not str or defined_in != module_name:
            continue
        candidate = candidates.get(id(function))
        if candidate is None:
            candidate = candidates[id(function)] = _Candidate(function)
        candidate.names.append(key)
        if value is function:
            candidate.plain_names.append(key)
        elif all(value is not wrapper for _, wrapper in candidate.wrappers):
            candidate.wrappers.append((key, value))
    for candidate in candidates.values():
        if not candidate.wrappers:
            private = private_wrappers.get(id(candidate.function))
            if private is not None:
                candidate.wrappers.append(private)
    return list(candidates.values())


def _wrapped_function(wrapper: Any) -> FunctionType | None:
    """The Python function a ``@snekit`` wrapper calls, if it wraps one.

    Read from the wrapper's JIT state rather than ``__wrapped__``, which is
    an ordinary, writable attribute.
    """
    attributes = wrapper.__dict__
    if type(attributes) is not dict:
        return None
    state = dict.get(attributes, "_jit_state")
    if type(state) is not _JitState:
        return None
    function = state._func
    return function if type(function) is FunctionType else None


def _finding(candidate: _Candidate, changed: list[_Candidate]) -> BiteFinding:
    """The finding for ``candidate``, once ``bite`` has committed or not."""
    status = candidate.status
    wrapper = candidate.wrapper
    if status == BITTEN:
        summary = _bitten_summary(candidate.names)
    elif status == ALREADY_BITTEN:
        summary = "already wrapped with @snekit"
        if candidate.plain_names:
            name = candidate.wrappers[0][0]
            aliases = ", ".join(candidate.plain_names)
            summary += f" as {name}; {aliases} now point to it too"
    else:
        summary = candidate.hunt.summary
        if candidate.wrappers and _is_opted_out(candidate.function):
            name = candidate.wrappers[0][0]
            summary += f"; its existing @snekit wrapper {name} is left as it is"
    if changed and status != SKIPPED and candidate.plain_names:
        # Planned, but not applied: nothing in the module was replaced.
        status, wrapper = SKIPPED, None
        if any(candidate is other for other in changed):
            summary = "its module binding changed before it could be updated"
        else:
            summary = "module changed before it could be updated"
    return BiteFinding(
        candidate.names[0],
        candidate.function,
        status,
        summary,
        wrapper=wrapper,
        hunt=candidate.hunt,
        names=candidate.names,
    )


def _bitten_summary(names: list[str]) -> str:
    if len(names) == 1:
        return "wrapped with @snekit"
    return f"wrapped with @snekit, also as {', '.join(names[1:])}"
