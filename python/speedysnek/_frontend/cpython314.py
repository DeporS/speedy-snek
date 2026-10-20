"""Translate CPython 3.14 bytecode into SpeedySnek instructions.

Everything that depends on CPython 3.14's bytecode lives in this module. It
only uses the public ``dis`` API (``Instruction`` fields such as ``opname``,
``opcode``, ``argval``, ``argrepr``, ``offset`` and ``positions``, plus
``dis.hasjump``) and public code object attributes.

Bytecode that CPython 3.14 emits for the supported subset, as shown by
``dis.dis`` for ``def f(a, b): return (a + 2) * b - 1000``::

    RESUME                   0
    LOAD_FAST_BORROW         0 (a)
    LOAD_SMALL_INT           2
    BINARY_OP                0 (+)
    LOAD_FAST_BORROW         1 (b)
    BINARY_OP                5 (*)
    LOAD_CONST               1 (1000)
    BINARY_OP               10 (-)
    RETURN_VALUE

Details of CPython 3.14 that matter here:

- ``RESUME`` starts every function. It is interpreter bookkeeping (tracing
  and eval-breaker checks) and is skipped, as are ``NOP`` and
  ``NOT_TAKEN`` (which only follows conditional jumps, rejected anyway).
- ``EXTENDED_ARG`` prefixes instructions whose argument is >= 256. ``dis``
  already folds it into the next instruction's ``arg``/``argval``, so it is
  skipped too.
- Variables are read with ``LOAD_FAST_BORROW`` (new in 3.14: it pushes a
  borrowed reference). Two consecutive reads of variables with index < 16
  are fused into the superinstruction ``LOAD_FAST_BORROW_LOAD_FAST_BORROW``,
  whose ``argval`` is a tuple of both names.
- Variables with index >= 64 are read with ``LOAD_FAST_CHECK``, because
  the compiler only proves 64 locals definitely bound. A read of a local
  that may be unbound uses ``LOAD_FAST_CHECK`` too. The frontend does not
  rely on which form CPython picked: it tracks assignments itself (see
  below).
- The owned-reference forms ``LOAD_FAST`` and ``LOAD_FAST_LOAD_FAST`` read
  variables the same way. ``LOAD_FAST`` is emitted, for example, when a
  variable is copied into another (``x = a``).
- An assignment statement ``x = <expression>`` is the expression followed
  by ``STORE_FAST`` (``argval`` is the name). The fused forms
  ``STORE_FAST_LOAD_FAST`` and ``STORE_FAST_STORE_FAST`` were not observed
  for plain assignments; CPython 3.14 emits them for tuple assignment
  (``x, y = a, b``), chained assignment (``x = y = a``, after a ``COPY``)
  and the walrus operator, all outside the subset, so they are rejected
  like any other unsupported opcode. Augmented assignment (``a += 1``) is
  a ``BINARY_OP`` with operator ``+=``, rejected as an unsupported
  operator, and ``del x`` is a rejected ``DELETE_FAST``.
- Integers 0..255 are loaded with ``LOAD_SMALL_INT`` (new in 3.14); other
  constants, including negative literals such as ``-1``, use
  ``LOAD_CONST``. Constant expressions such as ``2 * 3`` are folded by the
  compiler before we see them.
- All arithmetic is ``BINARY_OP``; ``argrepr`` names the operator.
- ``RETURN_CONST`` (3.12, 3.13) no longer exists: ``return 1`` is
  ``LOAD_SMALL_INT 1`` followed by ``RETURN_VALUE``.

Local variables. Straight-line code runs every instruction once, in
bytecode order, so a variable is bound at a read exactly when a store to it
(or, for an argument, the call itself) comes earlier in that order. The
frontend maps variables like this:

- an argument that has not been assigned yet is ``LOAD_ARG`` with its
  position;
- the first ``STORE_FAST`` to a name gives it the next free local slot,
  counting from 0, and stores go to ``STORE_LOCAL`` with that slot. An
  argument that is assigned gets a slot like any other variable;
- a read of a name with a slot is ``LOAD_LOCAL``;
- a read of any other name is rejected (``UNSUPPORTED_VARIABLE``): it may
  be unbound, and running the original function raises CPython's usual
  ``UnboundLocalError``.

Each load becomes one instruction: an assigned expression is computed once,
by the instructions before its ``STORE_LOCAL``, and never copied into the
places that read it.

``dis.get_instructions`` is called with ``adaptive=False`` (the default),
so specialized instructions such as ``BINARY_OP_ADD_INT``, which appear
after a function has run, are never seen.
"""

from __future__ import annotations

import dis
from types import CodeType

from .ir import Analysis, Instruction, Op, Rejection, RejectionReason

# Interpreter bookkeeping that has no meaning in the normalized form.
_BOOKKEEPING = frozenset({"RESUME", "NOP", "NOT_TAKEN", "EXTENDED_ARG"})

# Instructions that push one variable; ``argval`` is its name.
_LOAD_LOCAL = frozenset({"LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK"})

# Superinstructions that push two variables; ``argval`` is a tuple of both
# names.
_LOAD_LOCAL_PAIR = frozenset(
    {"LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"}
)

# Instructions that push a constant; ``argval`` is its value.
_LOAD_CONSTANT = frozenset({"LOAD_SMALL_INT", "LOAD_CONST"})

# Supported ``BINARY_OP`` operators, keyed by ``argrepr``.
_BINARY_OPS = {
    "+": Op.ADD,
    "-": Op.SUBTRACT,
    "*": Op.MULTIPLY,
}


def translate(code: CodeType) -> Analysis:
    """Translate the bytecode of ``code`` into normalized instructions.

    The caller has already checked the code object's flags and signature.
    The first unsupported instruction, in bytecode order, is reported.
    """
    bytecode = list(dis.get_instructions(code))

    # Branches and loops are rejected before anything else so that, for
    # example, ``if a:`` reports the jump rather than its ``TO_BOOL``.
    for instr in bytecode:
        if instr.opcode in dis.hasjump:
            return _reject(
                RejectionReason.CONTROL_FLOW,
                instr,
                f"only straight-line code is supported, found jump {instr.opname}",
            )

    instructions: list[Instruction] = []
    # Local slots by variable name, in order of first assignment.
    slots: dict[str, int] = {}
    for instr in bytecode:
        name = instr.opname
        if name in _BOOKKEEPING:
            continue
        if name in _LOAD_LOCAL or name in _LOAD_LOCAL_PAIR:
            names = instr.argval if name in _LOAD_LOCAL_PAIR else (instr.argval,)
            for var in names:
                if var in slots:
                    instructions.append(Instruction(Op.LOAD_LOCAL, slots[var]))
                    continue
                index = code.co_varnames.index(var)
                if index >= code.co_argcount:
                    return _reject(
                        RejectionReason.UNSUPPORTED_VARIABLE,
                        instr,
                        f"local variable {var!r} may be read before it is assigned",
                    )
                instructions.append(Instruction(Op.LOAD_ARG, index))
        elif name == "STORE_FAST":
            slot = slots.setdefault(instr.argval, len(slots))
            instructions.append(Instruction(Op.STORE_LOCAL, slot))
        elif name in _LOAD_CONSTANT:
            value = instr.argval
            # ``bool`` is a subclass of ``int``; reject it explicitly.
            if type(value) is not int:
                return _reject(
                    RejectionReason.UNSUPPORTED_CONSTANT,
                    instr,
                    f"only int constants are supported, "
                    f"found {type(value).__name__} {value!r}",
                )
            instructions.append(Instruction(Op.LOAD_CONST, value))
        elif name == "BINARY_OP":
            op = _BINARY_OPS.get(instr.argrepr)
            if op is None:
                return _reject(
                    RejectionReason.UNSUPPORTED_OPERATOR,
                    instr,
                    f"unsupported binary operator {instr.argrepr!r}",
                )
            instructions.append(Instruction(op))
        elif name == "RETURN_VALUE":
            instructions.append(Instruction(Op.RETURN))
        else:
            return _reject(
                RejectionReason.UNSUPPORTED_OPCODE,
                instr,
                f"unsupported opcode {name}",
            )

    return Analysis(instructions=tuple(instructions))


def _reject(reason: RejectionReason, instr: dis.Instruction, message: str) -> Analysis:
    lineno = instr.positions.lineno if instr.positions else None
    return Analysis(
        rejection=Rejection(
            reason=reason,
            message=f"{message} at offset {instr.offset}",
            opname=instr.opname,
            offset=instr.offset,
            lineno=lineno,
        )
    )
