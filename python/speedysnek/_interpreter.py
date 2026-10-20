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
   apply to these arguments.
3. The stack machine executes the program. Once validation and guards pass,
   this step cannot fail (short of running out of memory).

Arithmetic uses ordinary Python ``int`` operations, so results have
arbitrary precision exactly like CPython. There is deliberately no 64-bit
wrap-around or overflow check.

Local slots hold plain ``int`` values: ``STORE_LOCAL`` copies a value into
a slot, so storing into one slot never changes another. Validation proves
that every ``LOAD_LOCAL`` reads a slot that a ``STORE_LOCAL`` wrote on
every path leading to it; an unassigned slot is never read, and never has
a default value.

Comparisons compare Python ``int`` values, so they are exact for any size.
Their result, a condition, is held as a Python ``bool`` only until the
``JUMP_IF_FALSE`` right after the comparison consumes it; it is never
returned or stored. Only the selected branch runs.

:func:`speedysnek.snekit` never uses the interpreter: it runs either native code
or the original function. The interpreter remains the tests' oracle, and is
not part of the public API.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._frontend.ir import COMPARISONS, JUMPS, Instruction, Op

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


# The kinds of values on the operand stack. Comparisons produce conditions;
# everything else produces and consumes integers.
_INT = "an integer"
_CONDITION = "a condition"

# The kinds each operation pops (top of the stack last) and pushes. Jumps
# are handled separately.
_STACK_EFFECT = {
    Op.LOAD_ARG: ((), (_INT,)),
    Op.LOAD_CONST: ((), (_INT,)),
    Op.LOAD_LOCAL: ((), (_INT,)),
    Op.STORE_LOCAL: ((_INT,), ()),
    Op.ADD: ((_INT, _INT), (_INT,)),
    Op.SUBTRACT: ((_INT, _INT), (_INT,)),
    Op.MULTIPLY: ((_INT, _INT), (_INT,)),
    Op.RETURN: ((_INT,), ()),
    Op.JUMP: ((), ()),
    Op.JUMP_IF_FALSE: ((_CONDITION,), ()),
    **{op: ((_INT, _INT), (_CONDITION,)) for op in COMPARISONS},
}


def validate(instructions: Iterable[Instruction]) -> tuple[int, ...]:
    """Check that ``instructions`` form a well-formed program.

    A well-formed program is a non-empty sequence of :class:`Instruction`
    objects in which:

    - every ``JUMP`` and ``JUMP_IF_FALSE`` targets an instruction index
      ahead of itself, inside the program, so control flow has no cycles;
    - every instruction is reachable from the first one;
    - no instruction pops from an empty stack, and every operation finds
      the kind of value it needs: arithmetic, ``STORE_LOCAL``, comparisons
      and ``RETURN`` pop integers, and only ``JUMP_IF_FALSE`` pops a
      condition;
    - every comparison is immediately followed by ``JUMP_IF_FALSE``;
    - the stack is empty after every jump and on every path into a jump
      target, so paths join only through local slots;
    - every ``LOAD_LOCAL`` reads a slot that is stored on every path from
      the start of the program to it;
    - every path ends with a ``RETURN`` that finds exactly one value on the
      stack; none runs past the last instruction.

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

    stack: list[int | bool] = []
    # A dict, not a list: slot numbers in hand-written IR may be sparse.
    local_slots: dict[int, int] = {}
    # Validation guarantees that every jump goes forward, so this loop ends.
    pc = 0
    while True:
        instruction = program[pc]
        pc += 1
        match instruction.op:
            case Op.LOAD_ARG:
                stack.append(args[instruction.arg])
            case Op.LOAD_CONST:
                stack.append(instruction.arg)
            case Op.LOAD_LOCAL:
                stack.append(local_slots[instruction.arg])
            case Op.STORE_LOCAL:
                local_slots[instruction.arg] = stack.pop()
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
            case Op.LESS_THAN:
                right = stack.pop()
                stack.append(stack.pop() < right)
            case Op.LESS_EQUAL:
                right = stack.pop()
                stack.append(stack.pop() <= right)
            case Op.GREATER_THAN:
                right = stack.pop()
                stack.append(stack.pop() > right)
            case Op.GREATER_EQUAL:
                right = stack.pop()
                stack.append(stack.pop() >= right)
            case Op.EQUAL:
                right = stack.pop()
                stack.append(stack.pop() == right)
            case Op.NOT_EQUAL:
                right = stack.pop()
                stack.append(stack.pop() != right)
            case Op.JUMP:
                pc = instruction.arg
            case Op.JUMP_IF_FALSE:
                if not stack.pop():
                    pc = instruction.arg
            case Op.RETURN:
                return stack.pop()


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
    if not any(instr.op is Op.RETURN for instr in program):
        raise InvalidProgramError("program has no RETURN")
    count = len(program)
    for index, instruction in enumerate(program):
        if instruction.op in JUMPS:
            target = instruction.arg
            if target <= index:
                raise InvalidProgramError(
                    f"{instruction.op.name} at index {index} jumps back to index "
                    f"{target}; only forward jumps are allowed",
                    index=index,
                )
            if target >= count:
                raise InvalidProgramError(
                    f"{instruction.op.name} at index {index} jumps to index "
                    f"{target}, past the end of the program",
                    index=index,
                )

    # Every edge goes forward, so visiting instructions in index order sees
    # every predecessor of an instruction before the instruction itself:
    # one pass covers every path. The state is the operand stack's kinds
    # and the local slots stored on every path so far.
    #
    # ``assigned`` is None where the previous instruction does not fall
    # through (a JUMP or RETURN). ``joins`` holds, for every jump target
    # not yet reached, the slots stored on every jump into it; the stack is
    # always empty there.
    stack: list[str] = []
    assigned: frozenset[int] | None = frozenset()
    joins: dict[int, frozenset[int]] = {}
    read_args: set[int] = set()
    for index, instruction in enumerate(program):
        op = instruction.op
        jumped = joins.pop(index, None)
        if jumped is not None:
            if assigned is None:
                assigned = jumped
            else:
                if stack:
                    raise InvalidProgramError(
                        f"jump target at index {index} is reached from index "
                        f"{index - 1} with {len(stack)} value(s) on the stack; "
                        f"the stack must be empty where paths join",
                        index=index,
                    )
                assigned &= jumped
        elif assigned is None:
            raise InvalidProgramError(
                f"instruction at index {index} is unreachable", index=index
            )

        pops, pushes = _STACK_EFFECT[op]
        if len(stack) < len(pops):
            raise InvalidProgramError(
                f"stack underflow at index {index}: {op.name} "
                f"needs {len(pops)} value(s) but the stack has {len(stack)}",
                index=index,
            )
        operands = stack[len(stack) - len(pops) :]
        for needed, found in zip(pops, operands, strict=True):
            if needed is not found:
                raise InvalidProgramError(
                    f"{op.name} at index {index} needs {needed} but found {found}",
                    index=index,
                )
        del stack[len(stack) - len(pops) :]
        stack.extend(pushes)

        if op is Op.LOAD_LOCAL and instruction.arg not in assigned:
            raise InvalidProgramError(
                f"LOAD_LOCAL at index {index} reads local {instruction.arg} "
                f"before it is assigned on every path",
                index=index,
            )
        if op is Op.LOAD_ARG:
            read_args.add(instruction.arg)
        elif op is Op.STORE_LOCAL:
            assigned |= {instruction.arg}
        elif op in COMPARISONS:
            following = program[index + 1].op if index + 1 < count else None
            if following is not Op.JUMP_IF_FALSE:
                raise InvalidProgramError(
                    f"{op.name} at index {index} must be immediately followed "
                    f"by JUMP_IF_FALSE",
                    index=index,
                )
        elif op is Op.RETURN or op in JUMPS:
            if stack:
                raise InvalidProgramError(
                    f"{op.name} at index {index} leaves {len(stack)} extra "
                    f"value(s) on the stack; it must leave none",
                    index=index,
                )
            if op in JUMPS:
                target = instruction.arg
                previous = joins.get(target)
                joins[target] = assigned if previous is None else previous & assigned
            if op is not Op.JUMP_IF_FALSE:
                assigned = None

    if assigned is not None:
        raise InvalidProgramError(
            f"the instruction at index {count - 1} falls off the end of the "
            f"program; every path must end with RETURN",
            index=count - 1,
        )
    return tuple(sorted(read_args))


# Operations whose argument is an int: a constant, or an argument, local or
# instruction index (which cannot be negative).
_OPS_WITH_INT = frozenset(
    {Op.LOAD_ARG, Op.LOAD_CONST, Op.LOAD_LOCAL, Op.STORE_LOCAL} | JUMPS
)


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
    if op in _OPS_WITH_INT:
        if type(arg) is not int:
            raise InvalidProgramError(
                f"{op.name} at index {index} needs an int argument, "
                f"got {type(arg).__name__}",
                index=index,
            )
        if op is not Op.LOAD_CONST and arg < 0:
            if op is Op.LOAD_ARG:
                kind = "argument"
            elif op in JUMPS:
                kind = "target"
            else:
                kind = "local"
            raise InvalidProgramError(
                f"{op.name} at index {index} has negative {kind} index {arg}",
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
