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

Conditions and forward jumps express ``if`` statements:

- ``LESS_THAN``, ``LESS_EQUAL``, ``GREATER_THAN``, ``GREATER_EQUAL``,
  ``EQUAL`` and ``NOT_EQUAL`` pop the right operand, then the left operand,
  compare them as signed integers and push a *condition*. A condition is
  not an integer: it has exactly one permitted use, by the ``JUMP_IF_FALSE``
  that must immediately follow the comparison. It is never stored,
  returned or used in arithmetic, so it never becomes a Python value.
- ``JUMP_IF_FALSE t`` pops a condition and continues at instruction index
  ``t`` if it is false, or with the next instruction if it is true.
- ``JUMP t`` continues at instruction index ``t``.
- ``RETURN`` ends the path it is on. A program may contain several: CPython
  copies a short final ``return`` into each branch.

Jump targets are instruction indexes in the same program. They lie ahead
of the jump, except for the backedges of at most one ``while`` loop: a
``JUMP`` back to the loop's header, the comparison that decides whether
the loop runs again (see ``docs/loops.md``). The operand stack is empty at
every jump and at every jump target: values flow between branches, and
from one iteration to the next, only through local slots. So
``def f(a, b): r = a; if a < b: r += b; return r`` becomes::

    0  LOAD_ARG 0
    1  STORE_LOCAL 0
    2  LOAD_ARG 0
    3  LOAD_ARG 1
    4  LESS_THAN
    5  JUMP_IF_FALSE 10
    6  LOAD_LOCAL 0
    7  LOAD_ARG 1
    8  ADD
    9  STORE_LOCAL 0
    10 LOAD_LOCAL 0
    11 RETURN

A slot can be read only where every path that leads there has stored it.
A ``while`` loop is a test, a body and a backward ``JUMP``. So
``def f(n): i = 0; while i < n: i += 1; return i`` becomes::

    0  LOAD_CONST 0
    1  STORE_LOCAL 0
    2  LOAD_LOCAL 0     <- loop header
    3  LOAD_ARG 0
    4  LESS_THAN
    5  JUMP_IF_FALSE 11  <- the only exit
    6  LOAD_LOCAL 0
    7  LOAD_CONST 1
    8  ADD
    9  STORE_LOCAL 0
    10 JUMP 2           <- backedge
    11 LOAD_LOCAL 0
    12 RETURN

The full rules are in ``speedysnek._interpreter.validate``.

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
    LESS_THAN = "LESS_THAN"
    LESS_EQUAL = "LESS_EQUAL"
    GREATER_THAN = "GREATER_THAN"
    GREATER_EQUAL = "GREATER_EQUAL"
    EQUAL = "EQUAL"
    NOT_EQUAL = "NOT_EQUAL"
    JUMP = "JUMP"
    JUMP_IF_FALSE = "JUMP_IF_FALSE"


# Operations that pop two integers and push a condition.
COMPARISONS = frozenset(
    {
        Op.LESS_THAN,
        Op.LESS_EQUAL,
        Op.GREATER_THAN,
        Op.GREATER_EQUAL,
        Op.EQUAL,
        Op.NOT_EQUAL,
    }
)

# Operations whose operand is the index of the instruction they jump to.
JUMPS = frozenset({Op.JUMP, Op.JUMP_IF_FALSE})

# Operations whose integer operand is an index, which cannot be negative.
_OPS_WITH_INDEX = frozenset({Op.LOAD_ARG, Op.LOAD_LOCAL, Op.STORE_LOCAL}) | JUMPS

# Operations that carry an integer operand.
_OPS_WITH_ARG = _OPS_WITH_INDEX | {Op.LOAD_CONST}


@dataclass(frozen=True, slots=True)
class Instruction:
    """One normalized instruction.

    ``arg`` is the argument index for ``LOAD_ARG``, the integer value for
    ``LOAD_CONST``, the local slot for ``LOAD_LOCAL`` and ``STORE_LOCAL``,
    the target instruction index for ``JUMP`` and ``JUMP_IF_FALSE``, and
    ``None`` for every other operation.
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
    UNSUPPORTED_CONDITION = enum.auto()
    UNSUPPORTED_ITERATION = enum.auto()


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

    ``assumed_builtins`` names the builtins whose binding the instructions
    take for granted: ``("range",)`` for a ``for`` loop over ``range``,
    which the instructions replace with integer arithmetic. They are only
    equivalent to the function while each name resolves, through the
    function's globals and then its builtins, to the genuine builtin.
    Whoever runs the instructions in place of the function must check that
    (``speedysnek.snekit`` does).
    """

    instructions: tuple[Instruction, ...] = ()
    rejection: Rejection | None = None
    assumed_builtins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "assumed_builtins", tuple(self.assumed_builtins))
        if (self.rejection is None) == (not self.instructions):
            raise ValueError(
                "an Analysis needs instructions or a rejection, but not both"
            )
        if self.rejection is not None and self.assumed_builtins:
            raise ValueError("a rejected Analysis assumes no builtins")

    @property
    def supported(self) -> bool:
        return self.rejection is None
