"""``chill``: undo the ``@snekit`` wrappers ``bite`` installed in a module.

``chill`` is the explicit counterpart of :func:`speedysnek.bite`::

    import calculations
    import speedysnek as snek

    snek.bite(calculations)
    ...
    report = snek.chill(calculations)

Every public name in the module that holds a wrapper ``bite`` installed is
pointed back at the original Python function, aliases included. Nothing
else changes: references to a wrapper held anywhere else keep working, the
wrapper's call count and JIT state are left exactly as they are, and
nothing is called, compiled or warmed up. ``chill`` is not a way to stop
code that is already running.

Only wrappers ``bite`` made are undone. A wrapper is trusted only if it is
an exact ``NativeWrapper`` whose own JIT state (found as ``peek`` finds
it, not through ``_jit_state`` or ``__wrapped__``) runs a function the
module defines itself, and which ``bite`` recorded as its own for that
function. Wrappers from ``@snekit`` in the source, wrappers imported from
other modules, private names, classes and callable objects are left alone.

Like ``bite``, the module is changed all at once or not at all: every name
to restore is checked to still hold the inspected wrapper, then all of
them are restored in one update. ``chill`` reads the module through
``ModuleType``'s own ``__dict__`` descriptor, runs no user code, and loads
neither the frontend, the compiler nor the native extension.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from types import FunctionType, ModuleType
from typing import Any, TextIO

from ._bite import _INSTALLED
from ._hunt import _MODULE_DICT, _TYPE_NAME, _is_snekit_wrapper, _write
from ._nosnek import _is_opted_out
from ._peek import _wrapper_state

CHILLED = "chilled"
SKIPPED = "skipped"
UNCHANGED = "unchanged"

_HEADER = "🐍 Snek cooled down."
_NOTHING_TO_CHILL = "🐍 Snek found nothing to chill."
_EMPTY = "🐍 Snek found no functions to chill."


class ChillFinding:
    """What :func:`chill` did with one function.

    - ``name``: the function's first public name in the module.
    - ``names``: every public name in the module bound to the function or
      to a ``@snekit`` wrapper of it, in definition order (``name`` first).
    - ``function``: the original Python function.
    - ``wrapper``: the ``@snekit`` wrapper that was found: the one
      ``chill`` removed from the module for ``"chilled"``; ``None`` when
      the function had none.
    - ``status``: ``"chilled"``, ``"skipped"`` or ``"unchanged"``.
    - ``summary``: a short, human-readable explanation.
    """

    __slots__ = ("function", "name", "names", "status", "summary", "wrapper")

    def __init__(
        self,
        name: str,
        function: FunctionType,
        status: str,
        summary: str,
        wrapper: Any = None,
        names: Iterable[str] = (),
    ) -> None:
        self.name = name
        self.function = function
        self.status = status
        self.summary = summary
        self.wrapper = wrapper
        self.names = tuple(names) or (name,)

    @property
    def chilled(self) -> bool:
        return self.status == CHILLED

    def __repr__(self) -> str:
        return (
            f"ChillFinding(name={self.name!r}, status={self.status!r}, "
            f"summary={self.summary!r})"
        )


class ChillReport:
    """The result of :func:`chill`: one :class:`ChillFinding` per function.

    ``findings`` keeps the module's definition order. Iterating the report
    yields the findings; ``str(report)`` is the same text that
    :func:`chill` prints.
    """

    __slots__ = ("findings",)

    def __init__(self, findings: Iterable[ChillFinding]) -> None:
        self.findings = tuple(findings)

    def _with(self, status: str) -> tuple[ChillFinding, ...]:
        return tuple(finding for finding in self.findings if finding.status == status)

    @property
    def chilled(self) -> tuple[ChillFinding, ...]:
        """The findings whose module names now hold the plain function."""
        return self._with(CHILLED)

    @property
    def skipped(self) -> tuple[ChillFinding, ...]:
        """The findings with a wrapper that was left in place."""
        return self._with(SKIPPED)

    @property
    def unchanged(self) -> tuple[ChillFinding, ...]:
        """The findings for functions that had no wrapper to undo."""
        return self._with(UNCHANGED)

    @property
    def chilled_count(self) -> int:
        return len(self.chilled)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def unchanged_count(self) -> int:
        return len(self.unchanged)

    def __iter__(self) -> Iterator[ChillFinding]:
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    def render(self) -> str:
        """The human-readable report, without a trailing newline."""
        if not self.findings:
            return _EMPTY
        name_width = max(len(finding.name) for finding in self.findings)
        status_width = max(len(finding.status) for finding in self.findings)
        lines = [_HEADER if self.chilled_count else _NOTHING_TO_CHILL, ""]
        for finding in self.findings:
            lines.append(
                f"{finding.name:<{name_width}}  "
                f"{finding.status:<{status_width}}  {finding.summary}"
            )
        chilled = self.chilled_count
        counts = f"{chilled} function{'' if chilled == 1 else 's'} chilled"
        if self.skipped_count:
            counts += f", {self.skipped_count} skipped"
        counts += f", {self.unchanged_count} unchanged."
        lines += ["", counts]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return (
            f"<ChillReport: {self.chilled_count} chilled, {self.skipped_count} "
            f"skipped, {self.unchanged_count} unchanged>"
        )


class _Candidate:
    """One function and every public module name bound to it or a wrapper."""

    __slots__ = ("function", "kept", "names", "restore")

    def __init__(self, function: FunctionType) -> None:
        self.function = function
        # Every public name, in definition order.
        self.names: list[str] = []
        # The names holding a wrapper ``bite`` installed, with that wrapper.
        self.restore: list[tuple[str, Any]] = []
        # The names holding any other wrapper of the function.
        self.kept: list[tuple[str, Any]] = []


def chill(
    module: ModuleType, *, quiet: bool = False, stream: TextIO | None = None
) -> ChillReport:
    """Restore the plain functions behind the wrappers ``bite`` installed.

    ::

        snek.bite(my_module)
        ...
        report = snek.chill(my_module)

    Every public name in ``module`` that holds a ``@snekit`` wrapper
    :func:`bite` installed for one of the module's own functions is
    pointed back at that original function object, aliases included, and
    the function is reported as ``"chilled"``. The module's other public
    functions, including ones marked with :func:`speedysnek.nosnek`, are
    reported as ``"unchanged"``. A wrapper ``bite`` did not make, such as a
    ``@snekit`` in the source, is left in place and reported as
    ``"skipped"``. Imported wrappers, private names, classes and callable
    objects are not reported. Calling ``chill`` twice changes nothing the
    second time.

    The module is updated all at once or not at all: if any name to be
    restored no longer holds the inspected wrapper, no name is restored
    and the affected functions are reported as skipped.

    Only the module's names change. References to a wrapper held anywhere
    else still work, and the wrapper keeps its call count and JIT state.
    Nothing is called or compiled, and code that is already running through
    a wrapper is not interrupted.

    The report is printed to ``stream`` (``sys.stdout`` by default) unless
    ``quiet`` is true, and returned as a :class:`ChillReport`.

    Raises :class:`TypeError` if ``module`` is not a module.
    """
    if not issubclass(type(module), ModuleType):
        kind = _TYPE_NAME.__get__(type(module))
        hint = ""
        if issubclass(type(module), (str, bytes)):
            hint = "; import the module and pass the module itself"
        raise TypeError(f"chill() expected a module, got {kind}{hint}")

    namespace = _MODULE_DICT.__get__(module)
    candidates = _module_candidates(namespace)

    # Every name to restore must still hold the inspected wrapper. If any
    # does not, the module changed under us: restore nothing at all.
    updates: dict[str, FunctionType] = {}
    changed: list[_Candidate] = []
    for candidate in candidates:
        for key, wrapper in candidate.restore:
            if namespace.get(key) is not wrapper:
                changed.append(candidate)
                break
            updates[key] = candidate.function
    if not changed:
        # One update of the real namespace: no module hooks, and every
        # alias changes together.
        dict.update(namespace, updates)

    report = ChillReport(_finding(candidate, changed) for candidate in candidates)
    if not quiet:
        _write(sys.stdout if stream is None else stream, report.render())
    return report


def _module_candidates(namespace: dict[str, Any]) -> list[_Candidate]:
    """The module's own public functions, plain or wrapped, by first name."""
    module_name = namespace.get("__name__")
    if type(module_name) is not str:
        return []
    candidates: dict[int, _Candidate] = {}
    for key, value in tuple(namespace.items()):
        if type(key) is not str or key.startswith("_"):
            continue
        kind = type(value)
        if kind is FunctionType:
            function = value
        elif _is_snekit_wrapper(kind):
            function = _original_of(value)
            if function is None:
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
            continue
        # A genuine wrapper hashes by identity: the lookup runs no Python code.
        if _INSTALLED.get(value) is function:
            candidate.restore.append((key, value))
        else:
            candidate.kept.append((key, value))
    return list(candidates.values())


def _original_of(wrapper: Any) -> FunctionType | None:
    """The Python function a genuine ``@snekit`` wrapper runs, or ``None``.

    Read from the JIT state the wrapper itself holds, as ``peek`` reads it,
    never from ``__wrapped__`` or ``_jit_state``, which anyone can rewrite.
    """
    state = _wrapper_state(wrapper)
    if state is None:
        return None
    function = state._func
    return function if type(function) is FunctionType else None


def _finding(candidate: _Candidate, changed: list[_Candidate]) -> ChillFinding:
    """The finding for ``candidate``, once ``chill`` has committed or not."""
    wrapper = None
    if candidate.restore:
        status = CHILLED
        wrapper = candidate.restore[0][1]
        summary = "restored plain Python function"
        if len(candidate.restore) > 1:
            aliases = ", ".join(key for key, _ in candidate.restore[1:])
            summary += f", also as {aliases}"
        if candidate.kept:
            kept = ", ".join(key for key, _ in candidate.kept)
            summary += f"; {kept} kept a @snekit wrapper not made by snek.bite()"
        if changed:
            # Planned, but not applied: nothing in the module was restored.
            status = SKIPPED
            if any(candidate is other for other in changed):
                summary = "its module binding changed before it could be restored"
            else:
                summary = "module changed before it could be restored"
    elif candidate.kept:
        status = SKIPPED
        wrapper = candidate.kept[0][1]
        summary = "wrapped with @snekit, but not by snek.bite()"
    else:
        status = UNCHANGED
        summary = "not wrapped with @snekit"
        if _is_opted_out(candidate.function):
            summary = "marked with @nosnek, not wrapped"
    return ChillFinding(
        candidate.names[0],
        candidate.function,
        status,
        summary,
        wrapper=wrapper,
        names=candidate.names,
    )
