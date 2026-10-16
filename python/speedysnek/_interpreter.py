"""Private reference interpreter for SpeedySnek's normalized instructions.

This is the third stage of the compiler pipeline::

    Python function
        -> CPython 3.14 bytecode
        -> SpeedySnek normalized instructions
        -> Python reference interpreter  (this module)
           or Cranelift i64 backend  (``speedysnek._compiler``)

The interpreter runs a normalized program directly, without the original
Python function. It is a *correctness oracle*, not an optimization: the
native backend must produce exactly the results computed here whenever it
produces a result at all.

>>> from speedysnek._frontend import Instruction, Op
>>> program = (
...     Instruction(Op.LOAD_ARG, 0),
...     Instruction(Op.LOAD_CONST, 2),
...     Instruction(Op.MULTIPLY),
...     Instruction(Op.RETURN),
... )
>>> execute(program, args=(5,))
10

Running a program has three steps, each with its own failure mode:

1. :func:`validate` checks the whole program before anything runs. A
   malformed program raises :class:`InvalidProgramError`. The frontend
   never produces one, so this always means a compiler bug (or hand-written
   IR that is wrong).
2. Guards check the runtime arguments against the current specialization:
   every argument the program reads must be an exact ``int``. Otherwise
   :class:`GuardFailure` is raised. The program is fine; it just does not
   apply to these arguments, and a future JIT will fall back to CPython.
3. The stack machine executes the program. Once validation and guards pass,
   this step cannot fail (short of running out of memory).

Arithmetic uses ordinary Python ``int`` operations, so results have
arbitrary precision exactly like CPython. There is deliberately no 64-bit
wrap-around or overflow check.

The interpreter is not connected to :func:`speedysnek.jit` and is not part
of the public API.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._frontend.ir import Instruction, Op

__all__ = ["GuardFailure", "InvalidProgramError", "execute", "validate"]


class InvalidProgramError(Exception):
    """A normalized program is malformed or internally inconsistent.

    ``index`` is the position of the offending instruction, or ``None`` when
    the problem is about the program as a whole (for example, it is empty).
    """

    def __init__(self, message: str, index: int | None = None) -> None:
        super().__init__(message)
        self.index = index


class GuardFailure(Exception):
    """Runtime arguments do not satisfy the exact-``int`` specialization.

    The program itself is valid. ``arg_index`` is the position of the
    argument that failed its guard.
    """

    def __init__(self, message: str, arg_index: int) -> None:
        super().__init__(message)
        self.arg_index = arg_index


# How many values each operation pops from and pushes onto the stack.
_STACK_EFFECT = {
    Op.LOAD_ARG: (0, 1),
    Op.LOAD_CONST: (0, 1),
    Op.ADD: (2, 1),
    Op.SUBTRACT: (2, 1),
    Op.MULTIPLY: (2, 1),
    Op.RETURN: (1, 0),
}


def validate(instructions: Iterable[Instruction]) -> tuple[int, ...]:
    """Check that ``instructions`` form a well-formed program.

    A well-formed program is a non-empty sequence of :class:`Instruction`
    objects that never pops from an empty stack and ends with its only
    ``RETURN``, which finds exactly one value on the stack.

    Returns the sorted indexes of the arguments the program reads, which
    are the arguments the runtime guards must check. Raises
    :class:`InvalidProgramError` otherwise.
    """
    return _validate(_as_tuple(instructions))


def execute(instructions: Iterable[Instruction], args: Iterable[object]) -> int:
    """Run a normalized program with positional ``args`` and return its result.

    Raises :class:`InvalidProgramError` if the program is malformed (checked
    first, regardless of ``args``) and :class:`GuardFailure` if an argument
    the program reads is missing or is not an exact ``int``. Neither
    ``instructions`` nor ``args`` is modified, and no method of any argument
    is called.
    """
    program = _as_tuple(instructions)
    read_args = _validate(program)
    args = tuple(args)
    _check_guards(args, read_args)

    stack: list[int] = []
    for instruction in program:
        match instruction.op:
            case Op.LOAD_ARG:
                stack.append(args[instruction.arg])
            case Op.LOAD_CONST:
                stack.append(instruction.arg)
            case Op.ADD:
                right = stack.pop()
                left = stack.pop()
                stack.append(left + right)
            case Op.SUBTRACT:
                right = stack.pop()
                left = stack.pop()
                stack.append(left - right)
            case Op.MULTIPLY:
                right = stack.pop()
                left = stack.pop()
                stack.append(left * right)
            case Op.RETURN:
                return stack.pop()
    # _validate guarantees the program ends with RETURN.
    raise AssertionError("unreachable: validated program did not return")


def _as_tuple(instructions: Iterable[Instruction]) -> tuple[object, ...]:
    # Taking a private snapshot means a caller's list cannot change between
    # validation and execution, and is never modified by us.
    try:
        return tuple(instructions)
    except TypeError:
        raise InvalidProgramError(
            f"instructions must be iterable, got {type(instructions).__name__}"
        ) from None


def _validate(program: tuple[object, ...]) -> tuple[int, ...]:
    if not program:
        raise InvalidProgramError("program is empty")

    for index, instruction in enumerate(program):
        _check_instruction(index, instruction)

    # Every entry is now a well-formed Instruction.
    returns = [i for i, instr in enumerate(program) if instr.op is Op.RETURN]
    last = len(program) - 1
    if not returns:
        raise InvalidProgramError("program has no RETURN")
    if len(returns) > 1:
        raise InvalidProgramError(
            f"program has {len(returns)} RETURN instructions "
            f"(at indexes {', '.join(map(str, returns))}); exactly one is allowed",
            index=returns[1],
        )
    if returns[0] != last:
        follow = last - returns[0]
        raise InvalidProgramError(
            f"RETURN at index {returns[0]} is not the final instruction; "
            f"{follow} instruction(s) follow it",
            index=returns[0],
        )

    # Simulate the stack depth. Straight-line code has exactly one possible
    # depth at every instruction, so one pass covers every execution.
    depth = 0
    read_args: set[int] = set()
    for index, instruction in enumerate(program):
        pops, pushes = _STACK_EFFECT[instruction.op]
        if depth < pops:
            raise InvalidProgramError(
                f"stack underflow at index {index}: {instruction.op.name} "
                f"needs {pops} value(s) but the stack has {depth}",
                index=index,
            )
        if instruction.op is Op.RETURN and depth != pops:
            raise InvalidProgramError(
                f"RETURN at index {index} leaves {depth - pops} extra value(s) "
                f"on the stack; it must find exactly one",
                index=index,
            )
        if instruction.op is Op.LOAD_ARG:
            read_args.add(instruction.arg)
        depth += pushes - pops

    return tuple(sorted(read_args))


def _check_instruction(index: int, instruction: object) -> None:
    # ``Instruction`` validates itself when created, but it can still be
    # subclassed or corrupted with ``object.__setattr__``. The oracle
    # re-checks rather than trusting that nobody did.
    if type(instruction) is not Instruction:
        raise InvalidProgramError(
            f"entry at index {index} is a {type(instruction).__name__}, "
            f"not an Instruction",
            index=index,
        )
    op, arg = instruction.op, instruction.arg
    if type(op) is not Op:
        raise InvalidProgramError(
            f"instruction at index {index} has unknown operation {op!r}",
            index=index,
        )
    if op is Op.LOAD_ARG or op is Op.LOAD_CONST:
        if type(arg) is not int:
            raise InvalidProgramError(
                f"{op.name} at index {index} needs an int argument, "
                f"got {type(arg).__name__}",
                index=index,
            )
        if op is Op.LOAD_ARG and arg < 0:
            raise InvalidProgramError(
                f"LOAD_ARG at index {index} has negative argument index {arg}",
                index=index,
            )
    elif arg is not None:
        raise InvalidProgramError(
            f"{op.name} at index {index} takes no argument, got {arg!r}",
            index=index,
        )


def _check_guards(args: tuple[object, ...], read_args: tuple[int, ...]) -> None:
    # All guards run before execution starts, where a native backend would
    # check its entry conditions. Arguments the program never reads are not
    # guarded: CPython would not look at them either.
    for arg_index in read_args:
        if arg_index >= len(args):
            raise GuardFailure(
                f"the program reads argument {arg_index}, "
                f"but only {len(args)} argument(s) were given",
                arg_index=arg_index,
            )
        value = args[arg_index]
        # ``type(...) is int`` rejects bool and every other int subclass,
        # and calls nothing on the value. Only the type name is reported:
        # ``repr`` could run user code.
        if type(value) is not int:
            raise GuardFailure(
                f"argument {arg_index} is a {type(value).__name__}, not an int",
                arg_index=arg_index,
            )
