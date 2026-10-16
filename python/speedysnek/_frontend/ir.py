"""SpeedySnek's normalized instruction representation.

This module is independent of any CPython version. It describes a tiny
stack machine:

- ``LOAD_ARG n`` pushes the ``n``-th positional argument.
- ``LOAD_CONST v`` pushes the integer constant ``v``.
- ``ADD``, ``SUBTRACT`` and ``MULTIPLY`` pop the right operand, then the
  left operand, and push the result.
- ``RETURN`` pops the top of the stack and returns it.

So ``def add(a, b): return a + b`` becomes::

    LOAD_ARG 0
    LOAD_ARG 1
    ADD
    RETURN

All types here are immutable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Op(enum.Enum):
    """A normalized SpeedySnek operation."""

    LOAD_ARG = "LOAD_ARG"
    LOAD_CONST = "LOAD_CONST"
    ADD = "ADD"
    SUBTRACT = "SUBTRACT"
    MULTIPLY = "MULTIPLY"
    RETURN = "RETURN"


# Operations that carry an integer operand.
_OPS_WITH_ARG = frozenset({Op.LOAD_ARG, Op.LOAD_CONST})


@dataclass(frozen=True, slots=True)
class Instruction:
    """One normalized instruction.

    ``arg`` is the argument index for ``LOAD_ARG``, the integer value for
    ``LOAD_CONST`` and ``None`` for every other operation.
    """

    op: Op
    arg: int | None = None

    def __post_init__(self) -> None:
        if self.op in _OPS_WITH_ARG:
            if type(self.arg) is not int:
                raise TypeError(f"{self.op.name} requires an int argument")
            if self.op is Op.LOAD_ARG and self.arg < 0:
                raise ValueError("LOAD_ARG index must be non-negative")
        elif self.arg is not None:
            raise TypeError(f"{self.op.name} takes no argument")

    def __str__(self) -> str:
        if self.arg is None:
            return self.op.name
        return f"{self.op.name} {self.arg}"


class RejectionReason(enum.StrEnum):
    """Why a function cannot be normalized. Values are stable strings."""

    UNSUPPORTED_PYTHON_VERSION = enum.auto()
    NOT_A_FUNCTION = enum.auto()
    GENERATOR = enum.auto()
    COROUTINE = enum.auto()
    ASYNC_GENERATOR = enum.auto()
    CLOSURE = enum.auto()
    UNSUPPORTED_SIGNATURE = enum.auto()
    CONTROL_FLOW = enum.auto()
    UNSUPPORTED_OPCODE = enum.auto()
    UNSUPPORTED_OPERATOR = enum.auto()
    UNSUPPORTED_CONSTANT = enum.auto()
    UNSUPPORTED_VARIABLE = enum.auto()


@dataclass(frozen=True, slots=True)
class Rejection:
    """A machine-readable rejection plus a human-readable message.

    ``opname``, ``offset`` and ``lineno`` describe the offending CPython
    instruction. They are ``None`` when the rejection is about the function
    as a whole (for example, a generator).
    """

    reason: RejectionReason
    message: str
    opname: str | None = None
    offset: int | None = None
    lineno: int | None = None


@dataclass(frozen=True, slots=True)
class Analysis:
    """The result of analyzing one function.

    A supported function has normalized ``instructions`` and no
    ``rejection``. A rejected function has a ``rejection`` and no
    instructions: partial translations are never exposed.
    """

    instructions: tuple[Instruction, ...] = ()
    rejection: Rejection | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", tuple(self.instructions))
        if (self.rejection is None) == (not self.instructions):
            raise ValueError(
                "an Analysis needs instructions or a rejection, but not both"
            )

    @property
    def supported(self) -> bool:
        return self.rejection is None
