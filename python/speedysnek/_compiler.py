"""Private, experimental Cranelift backend for signed 64-bit integers.

This is the native branch of the compiler pipeline::

    Python function
        -> CPython bytecode frontend         (speedysnek._frontend)
        -> normalized IR
        -> reference interpreter             (speedysnek._interpreter)
           or Cranelift i64 backend          (this module)

:func:`compile_i64` turns normalized instructions into host machine code
that works on signed 64-bit integers:

>>> from speedysnek._frontend import analyze
>>> def expression(a, b):
...     return (a + b) * 2
>>> compiled = compile_i64(analyze(expression).instructions)
>>> compiled(10, 20)
60

The native tier is narrower than Python, and never silently differs from
it. Anything outside the tier raises instead of returning a wrong value:

- :class:`UnsupportedProgramError` at compile time: valid IR that cannot be
  represented, such as a constant outside the signed 64-bit range.
- :class:`NativeGuardFailure` at call time: the call does not fit the tier.
  Its :attr:`~NativeGuardFailure.reason` says why: the wrong number of
  arguments, an argument that is not an exact ``int`` (``bool`` and ``int``
  subclasses included), an argument outside the signed 64-bit range, or an
  ``ADD``, ``SUBTRACT`` or ``MULTIPLY`` whose exact result does not fit.

Malformed IR raises :class:`~speedysnek._interpreter.InvalidProgramError`,
exactly as in the reference interpreter.

**Argument rule:** a compiled program takes exactly ``arg_count``
positional arguments, and every one of them, read or not, must be an exact
``int`` in the signed 64-bit range, because all of them are passed to
native code. This is stricter than CPython, which ignores arguments a
function never reads; the stricter rule only ever causes a guard failure.

Nothing here calls the original Python function, and no guard runs Python
code on an argument: only ``type(value) is int`` and integer comparisons.

The backend is not connected to :func:`speedysnek.jit` and is not part of
the public API. A later stage will fall back to normal Python execution
when either failure is raised.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable

from . import _native
from ._frontend.ir import Instruction, Op
from ._interpreter import InvalidProgramError, validate

__all__ = [
    "I64_MAX",
    "I64_MIN",
    "CompiledI64",
    "GuardReason",
    "NativeGuardFailure",
    "UnsupportedProgramError",
    "compile_i64",
    "serialize",
]

I64_MIN = -(2**63)
I64_MAX = 2**63 - 1

# The wire format's opcode numbers are defined once, in Rust
# (``src/program.rs``), and read from the extension here.
_OPCODES = {
    Op.LOAD_ARG: _native.OP_LOAD_ARG,
    Op.LOAD_CONST: _native.OP_LOAD_CONST,
    Op.ADD: _native.OP_ADD,
    Op.SUBTRACT: _native.OP_SUBTRACT,
    Op.MULTIPLY: _native.OP_MULTIPLY,
    Op.RETURN: _native.OP_RETURN,
}


class UnsupportedProgramError(Exception):
    """Valid IR that the native i64 tier cannot represent.

    ``index`` is the position of the offending instruction, or ``None`` when
    the problem is not about one instruction.
    """

    def __init__(self, message: str, index: int | None = None) -> None:
        super().__init__(message)
        self.index = index


class GuardReason(enum.StrEnum):
    """Why a call does not fit the native tier. Values are stable strings."""

    WRONG_ARITY = enum.auto()
    NOT_EXACT_INT = enum.auto()
    OUT_OF_I64_RANGE = enum.auto()
    OVERFLOW = enum.auto()


class NativeGuardFailure(Exception):
    """A compiled program cannot run on these arguments natively.

    ``reason`` is a :class:`GuardReason`. ``arg_index`` names the offending
    argument for ``NOT_EXACT_INT`` and ``OUT_OF_I64_RANGE``, and
    ``instruction_index`` names the overflowing instruction for
    ``OVERFLOW``; otherwise they are ``None``.
    """

    def __init__(
        self,
        message: str,
        reason: GuardReason,
        *,
        arg_index: int | None = None,
        instruction_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.arg_index = arg_index
        self.instruction_index = instruction_index


def serialize(instructions: Iterable[Instruction]) -> tuple[tuple[int, ...], ...]:
    """Validate ``instructions`` and encode them for the Rust extension.

    Each instruction becomes ``(opcode,)`` or ``(opcode, operand)``, using
    only plain ints and tuples.

    >>> serialize([Instruction(Op.LOAD_ARG, 0), Instruction(Op.RETURN)])
    ((1, 0), (6,))
    """
    program = _as_tuple(instructions)
    _check(program)
    return _encode(program)


def _as_tuple(instructions: Iterable[Instruction]) -> tuple[Instruction, ...]:
    try:
        return tuple(instructions)
    except TypeError:
        raise InvalidProgramError(
            f"instructions must be iterable, got {type(instructions).__name__}"
        ) from None


def _check(program: tuple[Instruction, ...]) -> tuple[int, ...]:
    """Apply the interpreter's validation plus the i64 constant rule.

    Returns the indexes of the arguments the program reads.
    """
    read_args = validate(program)
    for index, instruction in enumerate(program):
        if instruction.op is Op.LOAD_CONST and not (
            I64_MIN <= instruction.arg <= I64_MAX
        ):
            raise UnsupportedProgramError(
                f"LOAD_CONST at index {index} does not fit in a signed 64-bit integer",
                index=index,
            )
    return read_args


def _encode(program: tuple[Instruction, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        (_OPCODES[i.op],) if i.arg is None else (_OPCODES[i.op], i.arg) for i in program
    )


def compile_i64(
    instructions: Iterable[Instruction], arg_count: int | None = None
) -> CompiledI64:
    """Compile normalized instructions to native code over signed i64.

    ``arg_count`` is the exact number of arguments the compiled program
    takes. It defaults to the smallest count that covers every ``LOAD_ARG``;
    pass the function's real arity when it has arguments the program never
    reads.

    Raises :class:`~speedysnek._interpreter.InvalidProgramError` for
    malformed IR (or an ``arg_count`` smaller than the program needs) and
    :class:`UnsupportedProgramError` for valid IR outside the native tier.
    """
    program = _as_tuple(instructions)
    read_args = _check(program)
    needed = read_args[-1] + 1 if read_args else 0
    if arg_count is None:
        arg_count = needed
    elif type(arg_count) is not int:
        raise TypeError(f"arg_count must be an int, got {type(arg_count).__name__}")
    elif arg_count < needed:
        raise InvalidProgramError(
            f"the program reads argument {needed - 1}, but arg_count is {arg_count}"
        )
    if arg_count > _native.MAX_ARGS:
        raise UnsupportedProgramError(
            f"{arg_count} arguments exceed the native limit of {_native.MAX_ARGS}"
        )

    try:
        function = _native.compile_i64(_encode(program), arg_count)
    except _native.CodegenError as error:
        raise UnsupportedProgramError(str(error)) from error
    return CompiledI64(function, program)


class CompiledI64:
    """A natively compiled program. Call it like the original function.

    The object owns the generated machine code, which stays valid for as
    long as the object exists. Calls are deterministic and have no side
    effects.
    """

    __slots__ = ("_function", "_instructions")

    def __init__(
        self, function: _native.I64Function, instructions: tuple[Instruction, ...]
    ) -> None:
        self._function = function
        self._instructions = instructions

    @property
    def arg_count(self) -> int:
        """The exact number of positional arguments a call must pass."""
        return self._function.arg_count

    @property
    def instructions(self) -> tuple[Instruction, ...]:
        """The normalized instructions this code was compiled from."""
        return self._instructions

    def __call__(self, /, *args: object) -> int:
        """Run the native code, or raise :class:`NativeGuardFailure`."""
        if len(args) != self._function.arg_count:
            raise NativeGuardFailure(
                f"expected {self._function.arg_count} argument(s), got {len(args)}",
                GuardReason.WRONG_ARITY,
            )
        for index, value in enumerate(args):
            # The exact type check comes first and calls nothing on the
            # value; after it, the comparisons are between builtin ints.
            # The message deliberately does not name the rejected type:
            # reading ``type(value).__name__`` can run a metaclass's
            # ``__getattribute__``.
            if type(value) is not int:
                raise NativeGuardFailure(
                    f"argument {index} must be an exact int",
                    GuardReason.NOT_EXACT_INT,
                    arg_index=index,
                )
            if not I64_MIN <= value <= I64_MAX:
                raise NativeGuardFailure(
                    f"argument {index} does not fit in a signed 64-bit integer",
                    GuardReason.OUT_OF_I64_RANGE,
                    arg_index=index,
                )

        status, value = self._function.call(args)
        if status == _native.STATUS_OK:
            return value
        if status == _native.STATUS_OVERFLOW:
            op = self._instructions[value].op.name
            raise NativeGuardFailure(
                f"{op} at index {value} overflows a signed 64-bit integer",
                GuardReason.OVERFLOW,
                instruction_index=value,
            )
        # The arity was checked above, so the native check cannot fail.
        raise RuntimeError(f"unexpected native status {status}")
