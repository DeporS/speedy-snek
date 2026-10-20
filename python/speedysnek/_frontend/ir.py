"""SpeedySnek's normalized instruction representation.

This module is independent of any CPython version. It describes a tiny
stack machine:

- ``LOAD_ARG n`` pushes the ``n``-th positional argument, exactly as the
  caller passed it.
- ``LOAD_CONST v`` pushes the integer constant ``v``.
- ``STORE_LOCAL n`` pops the top of the stack into local slot ``n``.
- ``LOAD_LOCAL n`` pushes the value last stored in local slot ``n``. A slot
  must be stored before it is read.
- ``ADD``, ``SUBTRACT`` and ``MULTIPLY`` pop the right operand, then the
  left operand, and push the result.
- ``RETURN`` pops the top of the stack and returns it.

So ``def add(a, b): return a + b`` becomes::

    LOAD_ARG 0
    LOAD_ARG 1
    ADD
    RETURN

Local slots are separate from arguments, and every slot starts out
unassigned. An argument that the function reassigns gets a local slot of
its own: reads before the assignment use ``LOAD_ARG``, reads after it use
``LOAD_LOCAL``. So ``def f(a, b): a = a + 1; return a * b`` becomes::

    LOAD_ARG 0
    LOAD_CONST 1
    ADD
    STORE_LOCAL 0
    LOAD_LOCAL 0
    LOAD_ARG 1
    MULTIPLY
    RETURN

A value stored in a slot is a copy: storing into one slot never changes
another.

All types here are immutable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Op(enum.Enum):
    """A normalized SpeedySnek operation."""

    LOAD_ARG = "LOAD_ARG"
    LOAD_CONST = "LOAD_CONST"
    LOAD_LOCAL = "LOAD_LOCAL"
    STORE_LOCAL = "STORE_LOCAL"
    ADD = "ADD"
    SUBTRACT = "SUBTRACT"
    MULTIPLY = "MULTIPLY"
    RETURN = "RETURN"


# Operations that carry an integer operand.
_OPS_WITH_ARG = frozenset({Op.LOAD_ARG, Op.LOAD_CONST, Op.LOAD_LOCAL, Op.STORE_LOCAL})

# Operations whose integer operand is an index, which cannot be negative.
_OPS_WITH_INDEX = frozenset({Op.LOAD_ARG, Op.LOAD_LOCAL, Op.STORE_LOCAL})


@dataclass(frozen=True, slots=True)
class Instruction:
    """One normalized instruction.

    ``arg`` is the argument index for ``LOAD_ARG``, the integer value for
    ``LOAD_CONST``, the local slot for ``LOAD_LOCAL`` and ``STORE_LOCAL``,
    and ``None`` for every other operation.
    """

    op: Op
    arg: int | None = None

    def __post_init__(self) -> None:
        if self.op in _OPS_WITH_ARG:
            if type(self.arg) is not int:
                raise TypeError(f"{self.op.name} requires an int argument")
            if self.op in _OPS_WITH_INDEX and self.arg < 0:
                raise ValueError(f"{self.op.name} index must be non-negative")
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
