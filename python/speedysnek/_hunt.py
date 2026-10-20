"""``hunt``: find the functions SpeedySnek's compiler understands.

``hunt`` is discovery only. It reads each candidate's bytecode with the
frontend (``speedysnek._frontend``) and reports which functions it accepts
and why the others were skipped. It never calls, decorates, compiles or
replaces anything, and it does not load the native extension or the
compiler::

    import my_calculations
    import speedysnek as snek

    report = snek.hunt(my_calculations)

"Ready" means only that the frontend accepts the function's bytecode. It
says nothing about whether the native version would be faster, and the
backend can still refuse a ready function (for example, one with too many
local variables).

``hunt`` runs no user code. Everything it reads is a C-level attribute of
an exact built-in type (``FunctionType``, ``ModuleType`` and ``type`` are
read through their own descriptors, so overridden ``__class__``,
``__dict__``, ``__name__`` or ``__getattr__`` are never consulted), apart
from iterating an iterable the caller passed in and writing the report to
the output stream.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from types import (
    BuiltinFunctionType,
    ClassMethodDescriptorType,
    FunctionType,
    MethodDescriptorType,
    MethodType,
    MethodWrapperType,
    ModuleType,
    WrapperDescriptorType,
)
from typing import Any, TextIO

READY = "ready"
SKIPPED = "skipped"

# Descriptors read directly, so no subclass or metaclass can intercept them.
_MODULE_DICT = ModuleType.__dict__["__dict__"]
_TYPE_NAME = type.__dict__["__name__"]

_BUILTIN_CALLABLES = (
    BuiltinFunctionType,
    ClassMethodDescriptorType,
    MethodDescriptorType,
    MethodWrapperType,
    WrapperDescriptorType,
)

_HEADER = "🐍 Snek went hunting."
_EMPTY = "🐍 Snek went hunting, but found no functions."

# Plain-English summaries of rejected CPython opcodes. Any other opcode is
# reported by name.
_OPCODE_SUMMARIES = {
    **dict.fromkeys(
        ("LOAD_GLOBAL", "LOAD_NAME"),
        "function calls and global names are not supported",
    ),
    **dict.fromkeys(
        ("CALL", "CALL_KW", "CALL_FUNCTION_EX"), "function calls are not supported"
    ),
    **dict.fromkeys(
        ("LOAD_ATTR", "STORE_ATTR", "DELETE_ATTR", "LOAD_SUPER_ATTR"),
        "attributes and methods are not supported",
    ),
    **dict.fromkeys(
        ("BUILD_LIST", "BUILD_TUPLE", "BUILD_SET", "BUILD_MAP", "BUILD_SLICE"),
        "lists, tuples, sets and dicts are not supported",
    ),
    **dict.fromkeys(
        ("STORE_SUBSCR", "DELETE_SUBSCR"), "item assignment is not supported"
    ),
    **dict.fromkeys(("UNPACK_SEQUENCE", "UNPACK_EX"), "unpacking is not supported"),
    **dict.fromkeys(
        (
            "FORMAT_SIMPLE",
            "FORMAT_WITH_SPEC",
            "CONVERT_VALUE",
            "BUILD_STRING",
            "BUILD_TEMPLATE",
            "BUILD_INTERPOLATION",
        ),
        "f-strings are not supported",
    ),
    **dict.fromkeys(
        ("UNARY_NEGATIVE", "UNARY_INVERT"),
        "unary operators like -x are not supported",
    ),
    **dict.fromkeys(("UNARY_NOT", "TO_BOOL"), "truth tests and not are not supported"),
    "CONTAINS_OP": "in and not in are not supported",
    "IS_OP": "is and is not are not supported",
    **dict.fromkeys(
        ("MAKE_FUNCTION", "MAKE_CELL", "LOAD_DEREF", "STORE_DEREF"),
        "nested functions and lambdas are not supported",
    ),
    **dict.fromkeys(("IMPORT_NAME", "IMPORT_FROM"), "imports are not supported"),
    **dict.fromkeys(
        ("STORE_GLOBAL", "DELETE_GLOBAL"), "global variables are not supported"
    ),
    "DELETE_FAST": "del is not supported",
    "POP_TOP": "unused expression values are not supported",
    "RAISE_VARARGS": "raise is not supported",
    "LOAD_ASSERTION_ERROR": "assert is not supported",
}

# Summaries of rejections that do not depend on the instruction. The
# frontend's own message (without its offset) is used for every other one.
_REASON_SUMMARIES = {
    "unsupported_python_version": "the compiler needs CPython 3.14",
    "generator": "generators are not supported",
    "coroutine": "async functions are not supported",
    "async_generator": "async generators are not supported",
    "closure": "closures are not supported",
    "unsupported_signature": "only positional arguments are supported",
    "unsupported_iteration": "only for loops over range(n) are supported",
}


class HuntFinding:
    """What :func:`hunt` found out about one function.

    - ``name``: the function's name (its name in the module, for a module).
    - ``function``: the inspected object itself.
    - ``status``: ``"ready"`` if the current compiler frontend accepts the
      function's bytecode, ``"skipped"`` otherwise.
    - ``summary``: a short, human-readable explanation.
    - ``reason``: for a skipped function, the frontend's stable rejection
      code, such as ``"unsupported_opcode"``; ``None`` when ready.
    - ``detail``: for a skipped function, the frontend's exact rejection
      message; ``None`` when ready.
    - ``lineno``: the source line of the rejected instruction, where known.
    """

    __slots__ = ("detail", "function", "lineno", "name", "reason", "status", "summary")

    def __init__(
        self,
        name: str,
        function: Any,
        status: str,
        summary: str,
        reason: str | None = None,
        detail: str | None = None,
        lineno: int | None = None,
    ) -> None:
        self.name = name
        self.function = function
        self.status = status
        self.summary = summary
        self.reason = reason
        self.detail = detail
        self.lineno = lineno

    @property
    def ready(self) -> bool:
        return self.status == READY

    def __repr__(self) -> str:
        return (
            f"HuntFinding(name={self.name!r}, status={self.status!r}, "
            f"summary={self.summary!r})"
        )


class HuntReport:
    """The result of :func:`hunt`: one :class:`HuntFinding` per function.

    ``findings`` keeps the order in which the functions were found.
    Iterating the report yields the findings; ``str(report)`` is the same
    text that :func:`hunt` prints.
    """

    __slots__ = ("findings",)

    def __init__(self, findings: Iterable[HuntFinding]) -> None:
        self.findings = tuple(findings)

    @property
    def ready(self) -> tuple[HuntFinding, ...]:
        """The findings whose function the frontend accepts."""
        return tuple(finding for finding in self.findings if finding.ready)

    @property
    def skipped(self) -> tuple[HuntFinding, ...]:
        """The findings whose function the frontend does not accept."""
        return tuple(finding for finding in self.findings if not finding.ready)

    @property
    def ready_count(self) -> int:
        return len(self.ready)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    def __iter__(self) -> Iterator[HuntFinding]:
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    def render(self) -> str:
        """The human-readable report, without a trailing newline."""
        if not self.findings:
            return _EMPTY
        width = max(len(finding.name) for finding in self.findings)
        lines = [_HEADER, ""]
        for finding in self.findings:
            summary = finding.summary
            if finding.lineno is not None:
                summary += f" (line {finding.lineno})"
            lines.append(f"{finding.name:<{width}}  {finding.status:<7}  {summary}")
        lines += ["", f"{self.ready_count} ready, {self.skipped_count} skipped."]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return f"<HuntReport: {self.ready_count} ready, {self.skipped_count} skipped>"


def hunt(
    *targets: Any, quiet: bool = False, stream: TextIO | None = None
) -> HuntReport:
    """Report which functions SpeedySnek can currently compile.

    Pass one module, one or more functions, or one iterable of functions::

        report = snek.hunt(my_module)
        report = snek.hunt(function_a, function_b)
        report = snek.hunt([function_a, function_b])

    For a module, the report covers the public Python functions the module
    defines itself, in definition order: imported functions, classes,
    callable objects, builtins and names starting with ``_`` are left out.
    Explicit functions are reported in the order given; a callable that is
    not a Python function, such as a builtin or a class, is reported as
    skipped.

    The report is printed to ``stream`` (``sys.stdout`` by default) unless
    ``quiet`` is true, and returned as a :class:`HuntReport`. Nothing is
    called, decorated or compiled: "ready" means only that the compiler
    frontend accepts the function's bytecode, not that it will be faster.

    Raises :class:`TypeError` for anything else, such as a module mixed with
    other arguments or a value that is not callable.
    """
    candidates = _candidates(targets)

    from ._frontend import analyze

    report = HuntReport(_inspect(name, obj, analyze) for name, obj in candidates)
    if not quiet:
        _write(sys.stdout if stream is None else stream, report.render())
    return report


def _candidates(targets: tuple[Any, ...]) -> list[tuple[str, Any]]:
    """The ``(name, object)`` pairs to inspect, in report order."""
    if not targets:
        raise TypeError("hunt() needs a module, a function or an iterable of functions")
    if len(targets) == 1:
        target = targets[0]
        kind = type(target)
        if issubclass(kind, ModuleType):
            return _module_functions(target)
        if issubclass(kind, (str, bytes)):
            raise TypeError(
                f"hunt() expected a module, a function or an iterable of "
                f"functions, got {_TYPE_NAME.__get__(kind)}; to hunt a module, "
                f"import it and pass the module itself"
            )
        if not callable(target):
            try:
                items = iter(target)
            except TypeError:
                raise TypeError(
                    f"hunt() expected a module, a function or an iterable of "
                    f"functions, got {_TYPE_NAME.__get__(kind)}"
                ) from None
            targets = tuple(items)
    for target in targets:
        kind = type(target)
        if issubclass(kind, ModuleType):
            raise TypeError("hunt() takes a module on its own: hunt(module)")
        if not callable(target):
            raise TypeError(
                f"hunt() expected a function, got {_TYPE_NAME.__get__(kind)}"
            )
    return [(_name(target), target) for target in targets]


def _module_functions(module: ModuleType) -> list[tuple[str, FunctionType]]:
    """The public functions ``module`` defines, in definition order."""
    namespace = _MODULE_DICT.__get__(module)
    module_name = namespace.get("__name__")
    if type(module_name) is not str:
        return []
    found = []
    seen = set()
    for key, value in tuple(namespace.items()):
        if type(key) is not str or key.startswith("_"):
            continue
        # ``FunctionType`` cannot be subclassed, and its ``__module__`` is a
        # plain member: reading it runs no Python code.
        if type(value) is not FunctionType or id(value) in seen:
            continue
        defined_in = value.__module__
        if type(defined_in) is str and defined_in == module_name:
            seen.add(id(value))
            found.append((key, value))
    return found


def _name(obj: Any) -> str:
    """A display name for ``obj``, read without running Python code."""
    kind = type(obj)
    if kind is FunctionType or kind in _BUILTIN_CALLABLES:
        return obj.__name__
    if issubclass(kind, type):
        return _TYPE_NAME.__get__(obj)
    if kind is MethodType and type(obj.__func__) is FunctionType:
        return obj.__func__.__name__
    if _is_snekit_wrapper(kind):
        # ``functools.update_wrapper`` copied the name into its ``__dict__``.
        name = obj.__dict__.get("__name__")
        if type(name) is str:
            return name
    return f"<{_TYPE_NAME.__get__(kind)} object>"


def _is_snekit_wrapper(kind: type) -> bool:
    # A wrapper exists only once ``@snekit`` has loaded the extension.
    native = sys.modules.get("speedysnek._native")
    return native is not None and kind is native.NativeWrapper


def _inspect(name: str, obj: Any, analyze: Any) -> HuntFinding:
    if type(obj) is not FunctionType:
        return HuntFinding(
            name,
            obj,
            SKIPPED,
            _not_a_function(obj),
            reason="not_a_function",
            detail=f"expected a Python function, got {_TYPE_NAME.__get__(type(obj))}",
        )
    analysis = analyze(obj)
    rejection = analysis.rejection
    if rejection is None:
        return HuntFinding(name, obj, READY, _describe(analysis))
    return HuntFinding(
        name,
        obj,
        SKIPPED,
        _explain(rejection),
        reason=rejection.reason.value,
        detail=rejection.message,
        lineno=rejection.lineno,
    )


def _not_a_function(obj: Any) -> str:
    kind = type(obj)
    if kind in _BUILTIN_CALLABLES:
        return "built-in functions cannot be compiled"
    if issubclass(kind, type):
        return "classes cannot be compiled, only functions"
    if kind is MethodType:
        return "bound methods cannot be compiled; pass the plain function"
    if _is_snekit_wrapper(kind):
        return "already wrapped by @snekit; pass the original function"
    return "not a Python function"


def _describe(analysis: Any) -> str:
    """Summarize what a supported function uses, from its instructions."""
    from ._frontend import Op

    instructions = analysis.instructions
    # The only backward jumps are a loop's backedges, to its header, which
    # starts with the comparison whose JUMP_IF_FALSE leaves the loop.
    header = next(
        (
            instr.arg
            for index, instr in enumerate(instructions)
            if instr.op is Op.JUMP and instr.arg <= index
        ),
        None,
    )
    loop_exit = None
    if header is not None:
        loop_exit = next(
            index
            for index in range(header, len(instructions))
            if instructions[index].op is Op.JUMP_IF_FALSE
        )
    branches = any(
        instr.op is Op.JUMP_IF_FALSE and index != loop_exit
        for index, instr in enumerate(instructions)
    )

    parts = ["integer arithmetic"]
    if branches:
        parts.append("if/else branches")
    if header is not None:
        if "range" in analysis.assumed_builtins:
            parts.append("one range loop")
        else:
            parts.append("one while loop")
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _explain(rejection: Any) -> str:
    """A short, friendly summary of a frontend rejection."""
    reason = rejection.reason.value
    if reason == "unsupported_opcode":
        return _OPCODE_SUMMARIES.get(
            rejection.opname, f"unsupported opcode: {rejection.opname}"
        )
    if reason == "unsupported_condition":
        if rejection.opname == "COMPARE_OP":
            return "comparisons only work as if or while conditions"
        return "conditions must be one integer comparison"
    if reason == "unsupported_constant":
        # The frontend's message ends with "found <type> <value>".
        found = rejection.message.partition(", found ")[2].split(" ", 1)[0]
        summary = "only integer constants are supported"
        if found:
            return f"{summary}, found {'None' if found == 'NoneType' else found}"
        return summary
    summary = _REASON_SUMMARIES.get(reason)
    if summary is not None:
        return summary
    # The frontend's message is written for people; drop the offset it
    # appends for an instruction.
    message = rejection.message
    if rejection.offset is not None:
        message = message.removesuffix(f" at offset {rejection.offset}")
    return message


def _write(stream: TextIO, text: str) -> None:
    text += "\n"
    try:
        stream.write(text)
    except UnicodeEncodeError:
        # A console or pipe that cannot encode the snek, or a name.
        encoding = getattr(stream, "encoding", None) or "ascii"
        text = text.replace("🐍 ", "", 1)
        stream.write(text.encode(encoding, "replace").decode(encoding))
