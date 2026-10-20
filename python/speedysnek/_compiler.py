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
  represented, such as a constant outside the signed 64-bit range, a
  local slot index of ``MAX_LOCALS`` or more, or more than ``MAX_JUMPS``
  jumps.
- :class:`NativeGuardFailure` at call time: the call does not fit the tier.
  Its :attr:`~NativeGuardFailure.reason` says why: the wrong number of
  arguments, an argument that is not an exact ``int`` (``bool`` and ``int``
  subclasses included), an argument outside the signed 64-bit range, an
  ``ADD``, ``SUBTRACT`` or ``MULTIPLY`` whose exact result does not fit, or
  a loop that used up the call's backedge budget.

Malformed IR, including a ``LOAD_LOCAL`` of a slot that is not stored on
every path to it, a backward jump that is not a supported loop's backedge
or a stack value left at a join, raises
:class:`~speedysnek._interpreter.InvalidProgramError`, exactly as in the
reference interpreter. The Rust extension validates the serialized program
again on its own.

**Loops.** A program may contain one loop: a ``while`` loop, or a ``for``
loop over ``range``, which the frontend lowers to the same shape (the
program itself never refers to ``range``; ``@snekit`` guards that
binding). Native code holds the GIL and cannot be interrupted, so a call
may take at most a fixed number of backward jumps, the *backedge budget*
(``_native.DEFAULT_BACKEDGE_BUDGET``, one million). The next backward
jump stops the call: the fast path returns
``None`` and the diagnostic path raises :class:`NativeGuardFailure` with
reason ``BUDGET_EXHAUSTED``. No partial result is returned. The budget is
fixed when a program is compiled; :func:`compile_i64`'s
``backedge_budget`` argument can only lower it, for tests. Programs
without a loop get no budget check at all. See ``docs/loops.md``.

Local slots are Cranelift variables: within a block a slot names a value
that was already computed, so assignments copy nothing at run time, and
where branches join, or a loop's backedges meet its entry at the loop
header, SSA block parameters carry each path's value.
Comparisons are signed and only steer the following branch; they never
produce a Python value. Only the selected branch runs. Every ``ADD``,
``SUBTRACT`` and ``MULTIPLY`` checks for overflow when it runs, whether or
not its result is used later; an operation on a branch not taken never
runs and so cannot cause a guard failure.

**Argument rule:** a compiled program takes exactly ``arg_count``
positional arguments, and every one of them, read or not, must be an exact
``int`` in the signed 64-bit range, because all of them are passed to
native code. This is stricter than CPython, which ignores arguments a
function never reads; the stricter rule only ever causes a guard failure.

Nothing here calls the original Python function, and no guard runs Python
code on an argument: only ``type(value) is int`` and integer comparisons.

A :class:`CompiledI64` has two ways to run its code:

- Calling it (``compiled(*args)``) is the *diagnostic* path. Its Python
  guards raise :class:`NativeGuardFailure` with the exact reason, and Rust
  then checks the arguments again. Tests and direct users of this module
  rely on those reasons.
- :attr:`CompiledI64.fast_call` (``fast_call(args)``) is the *fast* path:
  the native ``I64Function.try_call`` method. Rust checks the arguments
  once, with no Python code in between, and returns the result ``int``, or
  ``None`` for any call the native tier cannot run. A guard miss raises
  nothing.

:func:`speedysnek.snekit` compiles a hot function here once and runs its
fast path (the same Rust code as :attr:`~CompiledI64.fast_call`, called
directly from the native wrapper). It runs the original Python function
instead when :class:`UnsupportedProgramError` is raised at compile time
(from then on) or ``fast_call`` returns ``None`` (for that call only). The
backend is not part of the public API.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable

from . import _native
from ._frontend.ir import JUMPS, Instruction, Op
from ._interpreter import InvalidProgramError, validate

__all__ = [
    "DEFAULT_BACKEDGE_BUDGET",
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

DEFAULT_BACKEDGE_BUDGET = _native.DEFAULT_BACKEDGE_BUDGET

# The budget ``compile_i64`` uses when it is not given one. Private: tests
# lower it to make ``@snekit`` compile loops with a small budget.
_BACKEDGE_BUDGET = DEFAULT_BACKEDGE_BUDGET

# The wire format's opcode numbers are defined once, in Rust
# (``src/program.rs``), and read from the extension here.
_OPCODES = {
    Op.LOAD_ARG: _native.OP_LOAD_ARG,
    Op.LOAD_CONST: _native.OP_LOAD_CONST,
    Op.LOAD_LOCAL: _native.OP_LOAD_LOCAL,
    Op.STORE_LOCAL: _native.OP_STORE_LOCAL,
    Op.ADD: _native.OP_ADD,
    Op.SUBTRACT: _native.OP_SUBTRACT,
    Op.MULTIPLY: _native.OP_MULTIPLY,
    Op.RETURN: _native.OP_RETURN,
    Op.JUMP: _native.OP_JUMP,
    Op.JUMP_IF_FALSE: _native.OP_JUMP_IF_FALSE,
    Op.LESS_THAN: _native.OP_LESS_THAN,
    Op.LESS_EQUAL: _native.OP_LESS_EQUAL,
    Op.GREATER_THAN: _native.OP_GREATER_THAN,
    Op.GREATER_EQUAL: _native.OP_GREATER_EQUAL,
    Op.EQUAL: _native.OP_EQUAL,
    Op.NOT_EQUAL: _native.OP_NOT_EQUAL,
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
    BUDGET_EXHAUSTED = enum.auto()


class NativeGuardFailure(Exception):
    """A compiled program cannot run on these arguments natively.

    ``reason`` is a :class:`GuardReason`. ``arg_index`` names the offending
    argument for ``NOT_EXACT_INT`` and ``OUT_OF_I64_RANGE``, and
    ``instruction_index`` names the overflowing instruction for
    ``OVERFLOW`` and the backward jump that found the budget used up for
    ``BUDGET_EXHAUSTED``; otherwise they are ``None``.
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
    """Apply the interpreter's validation plus the native tier's limits:
    i64 constants, local indexes below ``MAX_LOCALS`` and at most
    ``MAX_JUMPS`` jumps.

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
        if (
            instruction.op is Op.LOAD_LOCAL or instruction.op is Op.STORE_LOCAL
        ) and instruction.arg >= _native.MAX_LOCALS:
            raise UnsupportedProgramError(
                f"{instruction.op.name} at index {index} uses local "
                f"{instruction.arg}, beyond the native limit of "
                f"{_native.MAX_LOCALS} locals",
                index=index,
            )
    jumps = [index for index, i in enumerate(program) if i.op in JUMPS]
    if len(jumps) > _native.MAX_JUMPS:
        raise UnsupportedProgramError(
            f"the program has {len(jumps)} jumps, beyond the native limit of "
            f"{_native.MAX_JUMPS}",
            index=jumps[_native.MAX_JUMPS],
        )
    return read_args


def _encode(program: tuple[Instruction, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        (_OPCODES[i.op],) if i.arg is None else (_OPCODES[i.op], i.arg) for i in program
    )


def compile_i64(
    instructions: Iterable[Instruction],
    arg_count: int | None = None,
    *,
    backedge_budget: int | None = None,
) -> CompiledI64:
    """Compile normalized instructions to native code over signed i64.

    ``arg_count`` is the exact number of arguments the compiled program
    takes. It defaults to the smallest count that covers every ``LOAD_ARG``;
    pass the function's real arity when it has arguments the program never
    reads.

    ``backedge_budget`` is the number of backward jumps one call may take.
    It defaults to ``_BACKEDGE_BUDGET`` (``DEFAULT_BACKEDGE_BUDGET``) and
    must lie between 0 and ``DEFAULT_BACKEDGE_BUDGET``; it only matters for
    a program with a loop.

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
    if backedge_budget is None:
        backedge_budget = _BACKEDGE_BUDGET
    if type(backedge_budget) is not int:
        raise TypeError(
            f"backedge_budget must be an int, got {type(backedge_budget).__name__}"
        )
    if not 0 <= backedge_budget <= _native.MAX_BACKEDGE_BUDGET:
        raise ValueError(
            f"backedge_budget must be between 0 and {_native.MAX_BACKEDGE_BUDGET}"
        )

    try:
        function = _native.compile_i64(_encode(program), arg_count, backedge_budget)
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
    def backedge_budget(self) -> int | None:
        """The backward jumps one call may take, or ``None`` without a loop."""
        return self._function.backedge_budget

    @property
    def instructions(self) -> tuple[Instruction, ...]:
        """The normalized instructions this code was compiled from."""
        return self._instructions

    @property
    def function(self) -> _native.I64Function:
        """The native ``I64Function`` that owns the machine code.

        ``@snekit``'s native wrapper calls its fast path directly, without
        going through :attr:`fast_call`.
        """
        return self._function

    @property
    def fast_call(self) -> Callable[[tuple[object, ...]], int | None]:
        """The bound native ``I64Function.try_call`` for this code.

        It takes the positional arguments as one tuple and returns the
        result ``int``, or ``None`` if the call does not fit the native tier
        (the cases :class:`NativeGuardFailure` describes, budget exhaustion
        included). It raises nothing
        for such a call and does not say why. The bound method keeps the
        machine code alive.

        >>> from speedysnek._frontend import analyze
        >>> def add(a, b):
        ...     return a + b
        >>> fast_call = compile_i64(analyze(add).instructions).fast_call
        >>> fast_call((2, 3))
        5
        >>> print(fast_call((2**63 - 1, 1)))  # overflow
        None
        """
        return self._function.try_call

    def __call__(self, /, *args: object) -> int:
        """Run the native code, or raise :class:`NativeGuardFailure`.

        This is the diagnostic path: unlike :attr:`fast_call`, it reports
        exactly why a call does not fit the native tier.
        """
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
        if status == _native.STATUS_BUDGET_EXHAUSTED:
            raise NativeGuardFailure(
                f"the loop took more than {self.backedge_budget} backward jumps; "
                f"the backward JUMP at index {value} stopped it",
                GuardReason.BUDGET_EXHAUSTED,
                instruction_index=value,
            )
        # The arity was checked above, so the native check cannot fail.
        raise RuntimeError(f"unexpected native status {status}")
