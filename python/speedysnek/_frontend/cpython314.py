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
  and eval-breaker checks) and is skipped, as are ``NOP`` (``pass``, or
  ``if True:``) and ``NOT_TAKEN``, which follows every conditional jump and
  only records branch statistics.
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
  like any other unsupported opcode. ``del x`` is a rejected
  ``DELETE_FAST``.
- Augmented assignment to a local, ``x += <expression>``, loads ``x``,
  evaluates the expression, then ``BINARY_OP`` with the in-place operator
  (``argrepr`` ``+=``) and ``STORE_FAST x``. For ``def f(a, b): a += b``::

      LOAD_FAST_BORROW_LOAD_FAST_BORROW 1 (a, b)
      BINARY_OP               13 (+=)
      STORE_FAST               0 (a)

  That is the same sequence as ``x = x + <expression>`` apart from the
  operator, so ``+=``, ``-=`` and ``*=`` become ``ADD``, ``SUBTRACT`` and
  ``MULTIPLY``. They differ only for objects with ``__iadd__``,
  ``__isub__`` or ``__imul__``, and an exact ``int`` has none of them; the
  native guards admit nothing else. Other targets are rejected before
  their ``BINARY_OP``: ``a.x += 1`` and ``a[0] += 1`` start with a
  ``COPY``, a global with ``LOAD_GLOBAL``, a captured variable with
  ``MAKE_CELL`` or a closure. ``x += a`` on a local not yet assigned reads
  ``x`` first and is rejected like any other such read.
- Integers 0..255 are loaded with ``LOAD_SMALL_INT`` (new in 3.14); other
  constants, including negative literals such as ``-1``, use
  ``LOAD_CONST``. Constant expressions such as ``2 * 3`` are folded by the
  compiler before we see them.
- All arithmetic is ``BINARY_OP``; ``argrepr`` names the operator.
- ``RETURN_CONST`` (3.12, 3.13) no longer exists: ``return 1`` is
  ``LOAD_SMALL_INT 1`` followed by ``RETURN_VALUE``.

Conditionals. ``if a < b:`` compiles to the two operands, ``COMPARE_OP``
and ``POP_JUMP_IF_FALSE``, which jumps forward past the body when the
condition is false. For ``def f(a, b): r = a; if a < b: r += b; return
r * 2``::

    LOAD_FAST                0 (a)
    STORE_FAST               2 (r)
    LOAD_FAST_BORROW_LOAD_FAST_BORROW 1 (a, b)
    COMPARE_OP              18 (bool(<))
    POP_JUMP_IF_FALSE        9 (to L1)
    NOT_TAKEN
    LOAD_FAST_BORROW_LOAD_FAST_BORROW 33 (r, b)
    BINARY_OP               13 (+=)
    STORE_FAST               2 (r)
    L1: LOAD_FAST_BORROW     2 (r)
    LOAD_SMALL_INT           2
    BINARY_OP                5 (*)
    RETURN_VALUE

- ``COMPARE_OP``'s ``argval`` is the operator (``<``, ``<=``, ``>``,
  ``>=``, ``==``, ``!=``); its ``argrepr`` is ``bool(<)`` when the result
  is converted to ``bool`` for a jump. The conversion changes nothing for
  exact ``int`` operands, whose comparisons already return a ``bool``.
- An ``else`` branch ends the ``if`` body with ``JUMP_FORWARD`` to the
  join. ``elif`` is a nested ``if`` in the ``else`` branch.
- CPython 3.14 copies a short exit block, such as ``return r``, into each
  branch instead of jumping to it. The ``if``/``else`` in
  ``calculate_score`` therefore has two ``RETURN_VALUE`` and no join, and
  ``return a if a < b else b`` is the same bytecode as ``if a < b: return
  a`` followed by ``return b``. The frontend works on bytecode, so it
  accepts every such form, with one ``RETURN`` per path.
- ``if a < b and b < c:`` is the same bytecode as two nested ``if``
  statements, and is accepted as such.

Loops. ``while i < n:`` compiles to a header that tests the condition
and jumps forward out of the loop when it is false, the body, and a
``JUMP_BACKWARD`` to the header. For ``def f(n): i = 0; while i < n: i +=
1; return i``::

    LOAD_SMALL_INT           0
    STORE_FAST               1 (i)
    L1: LOAD_FAST_BORROW_LOAD_FAST_BORROW 16 (i, n)
    COMPARE_OP              18 (bool(<))
    POP_JUMP_IF_FALSE       12 (to L2)
    NOT_TAKEN
    LOAD_FAST_BORROW         1 (i)
    LOAD_SMALL_INT           1
    BINARY_OP               13 (+=)
    STORE_FAST               1 (i)
    JUMP_BACKWARD           17 (to L1)
    L2: LOAD_FAST_BORROW     1 (i)
    RETURN_VALUE

- CPython 3.14 does not copy the test to the bottom of the loop, and
  ``POP_JUMP_IF_FALSE`` and ``POP_JUMP_IF_TRUE`` never jump backward: a
  ``JUMP_BACKWARD`` is the only backward jump.
- CPython copies the ``JUMP_BACKWARD`` into each branch of an ``if``/``else``
  that ends the body, so a loop can have several, all to the same header.
- An ``if`` without ``else`` that ends the body is inverted:
  ``COMPARE_OP``, ``POP_JUMP_IF_TRUE`` to the ``if`` body, ``NOT_TAKEN``,
  ``JUMP_BACKWARD``. That is the only ``POP_JUMP_IF_TRUE`` accepted, and it
  becomes ``JUMP_IF_FALSE`` after the negated comparison (``<`` becomes
  ``>=``, ``==`` becomes ``!=``, ...), which is exact for integers.
- ``continue`` is a ``JUMP_BACKWARD`` to the header too, so ``if c:
  continue`` followed by more code is the same bytecode as an ``if c:
  pass`` / ``else:`` that ends the body, and is accepted. A loop ``else``
  without ``break`` is the same bytecode as code after the loop.

One loop of this shape is accepted: every ``JUMP_BACKWARD`` goes to the
same header; the first jump or return from the header on is a
``POP_JUMP_IF_FALSE`` after a ``COMPARE_OP``, whose target, the exit, lies
after every ``JUMP_BACKWARD``; nothing jumps into the loop past its header;
and nothing in the loop leaves it other than that test (``break``, a
``return`` in the loop, and an ``and`` or ``or`` in the ``while``
condition all would). ``docs/loops.md`` explains the shape in detail.

Only these jumps are recognized: ``POP_JUMP_IF_FALSE`` directly after a
``COMPARE_OP`` (bookkeeping aside), ``POP_JUMP_IF_TRUE`` in the shape
above, ``JUMP_FORWARD`` and ``JUMP_BACKWARD``. Everything else is
rejected:

- any other jump (``CONTROL_FLOW``): ``for`` loops (``FOR_ITER``), ``or``
  and ``not`` (``POP_JUMP_IF_TRUE`` elsewhere), ``is None`` tests
  (``POP_JUMP_IF_NONE``);
- a second loop, nested or not, and a loop of any other shape
  (``CONTROL_FLOW``), including ``while True:``;
- a condition that is not a comparison (``UNSUPPORTED_CONDITION``):
  truthiness (``TO_BOOL``), ``is`` (``IS_OP``), ``in`` (``CONTAINS_OP``);
- a ``COMPARE_OP`` whose result is not consumed by the jump right after it
  (``UNSUPPORTED_CONDITION``), such as ``x = a < b`` or ``return a < b``;
  chained comparisons (``a < b < c``) already fail at their ``SWAP``;
- a jump, or a join, with a value on the operand stack (``CONTROL_FLOW``),
  such as a conditional expression inside a larger expression;
- unreachable bytecode, and any code object with an exception table
  (``try``, ``with``), whose handlers add edges this analysis does not
  model (``CONTROL_FLOW``).

Local variables. All jumps but a loop's backedges go forward, so bytecode
order visits every other predecessor of an instruction before the
instruction itself, and a pass in that order computes, for every variable
at every instruction, what it may hold on the paths that reach it: the
argument as passed, a value the function stored, or nothing. At a join the
possibilities of the incoming paths are combined. With a loop, the pass is
repeated, each time also combining what the backedges brought back on the
previous pass into the header, until that no longer changes; on the first
pass the header only sees the state on entry, so a value stored later in
the body never counts as assigned on the first iteration. The frontend
maps variables like this:

- the first ``STORE_FAST`` to a name, in bytecode order, gives it the next
  free local slot, counting from 0, and stores go to ``STORE_LOCAL`` with
  that slot. An argument that is assigned gets a slot like any other
  variable;
- a read of a name that holds its argument on every path is ``LOAD_ARG``
  with its position;
- a read of a name that holds a stored value on every path is
  ``LOAD_LOCAL``;
- a read of an argument that is stored on some paths but not others is
  ``LOAD_LOCAL`` too, and the program starts with ``LOAD_ARG``,
  ``STORE_LOCAL`` to put the argument in that slot. Straight-line code
  never needs this; a loop that updates an argument, as in ``while n > 0:
  n -= 1``, always does;
- a read of a name that may be unbound is rejected
  (``UNSUPPORTED_VARIABLE``): running the original function then raises
  CPython's usual ``UnboundLocalError`` on the paths where it is unbound.

Each load becomes one instruction: an assigned expression is computed once,
by the instructions before its ``STORE_LOCAL``, and never copied into the
places that read it. Jump operands are translated from bytecode offsets to
the index of the first normalized instruction at or after the target.

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

# The jumps that ``if`` statements and ``while`` loops compile to. Only
# ``JUMP_BACKWARD`` goes backward in CPython 3.14; this is checked anyway.
_FORWARD_JUMPS = frozenset({"POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE", "JUMP_FORWARD"})
_JUMPS = _FORWARD_JUMPS | {"JUMP_BACKWARD"}
_CONDITIONAL_JUMPS = frozenset({"POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"})

# ``COMPARE_OP`` operators, keyed by ``argval``.
_COMPARISONS = {
    "<": Op.LESS_THAN,
    "<=": Op.LESS_EQUAL,
    ">": Op.GREATER_THAN,
    ">=": Op.GREATER_EQUAL,
    "==": Op.EQUAL,
    "!=": Op.NOT_EQUAL,
}

# The comparison that is true exactly when the key is false. Exact for
# integers, which are totally ordered; nothing else reaches the IR.
_NEGATED = {
    Op.LESS_THAN: Op.GREATER_EQUAL,
    Op.LESS_EQUAL: Op.GREATER_THAN,
    Op.GREATER_THAN: Op.LESS_EQUAL,
    Op.GREATER_EQUAL: Op.LESS_THAN,
    Op.EQUAL: Op.NOT_EQUAL,
    Op.NOT_EQUAL: Op.EQUAL,
}

# Supported ``BINARY_OP`` operators, keyed by ``argrepr``. The in-place
# forms come from augmented assignment and act like the plain ones on an
# exact ``int``, the only type that runs natively.
_BINARY_OPS = {
    "+": Op.ADD,
    "-": Op.SUBTRACT,
    "*": Op.MULTIPLY,
    "+=": Op.ADD,
    "-=": Op.SUBTRACT,
    "*=": Op.MULTIPLY,
}

# Every opcode the translator handles, apart from bookkeeping.
_SUPPORTED = (
    _LOAD_LOCAL
    | _LOAD_LOCAL_PAIR
    | _LOAD_CONSTANT
    | _JUMPS
    | {"STORE_FAST", "BINARY_OP", "COMPARE_OP", "RETURN_VALUE"}
)


def translate(code: CodeType) -> Analysis:
    """Translate the bytecode of ``code`` into normalized instructions.

    The caller has already checked the code object's flags and signature.
    Unsupported jumps are reported first, then a loop of an unsupported
    shape, then conditions that are not a comparison; after that, the first
    unsupported instruction in bytecode order is reported.
    """
    bytecode = list(dis.get_instructions(code))

    # Exception handlers add edges from every protected instruction that
    # the analysis below does not model, so they are rejected first.
    if code.co_exceptiontable:
        return Analysis(
            rejection=Rejection(
                reason=RejectionReason.CONTROL_FLOW,
                message="exception handling (try, with) is not supported",
            )
        )

    # Unsupported jumps are rejected before anything else so that, for
    # example, a ``for`` loop reports its FOR_ITER rather than a call.
    header = None
    for instr in bytecode:
        if instr.opcode not in dis.hasjump:
            continue
        if instr.opname not in _JUMPS:
            return _reject(
                RejectionReason.CONTROL_FLOW,
                instr,
                f"unsupported jump {instr.opname}: only if statements and while "
                f"loops on integer comparisons are supported",
            )
        if (instr.opname == "JUMP_BACKWARD") != (instr.argval <= instr.offset):
            return _reject(
                RejectionReason.CONTROL_FLOW,
                instr,
                f"{instr.opname} jumps the wrong way",
            )
        if instr.opname == "JUMP_BACKWARD":
            if header is not None and instr.argval != header:
                return _reject(
                    RejectionReason.CONTROL_FLOW,
                    instr,
                    "only one while loop per function is supported, "
                    "not nested loops or several loops",
                )
            header = instr.argval
    significant = [instr for instr in bytecode if instr.opname not in _BOOKKEEPING]
    rejection = _check_loop(significant, header)
    if rejection is not None:
        return rejection
    # Every condition must be a comparison, consumed directly by the jump.
    # ``if a:`` reports the jump rather than its TO_BOOL.
    for position, instr in enumerate(significant):
        if instr.opname in _CONDITIONAL_JUMPS and (
            position == 0 or significant[position - 1].opname != "COMPARE_OP"
        ):
            found = significant[position - 1].opname if position else "nothing"
            return _reject(
                RejectionReason.UNSUPPORTED_CONDITION,
                instr,
                f"only a comparison can be an if or while condition, found {found}",
            )

    # Local slots, in order of first assignment in the bytecode. A loop can
    # read a variable before its first assignment in bytecode order (on the
    # next iteration), so they are numbered before translating.
    slots: dict[str, int] = {}
    for instr in bytecode:
        if instr.opname == "STORE_FAST":
            slots.setdefault(instr.argval, len(slots))

    # Without a loop, one pass sees every path. With one, the pass is
    # repeated, feeding what the backedges bring back into the header,
    # until that stops changing. Each repetition adds at least one of three
    # possibilities to some variable, so this ends after at most
    # ``2 * len(co_varnames) + 2`` passes.
    backedge: _State | None = None
    for _ in range(2 * len(code.co_varnames) + 2):
        translator = _Translator(code, slots, header, backedge)
        rejection = translator.run(bytecode, significant)
        if rejection is not None:
            # Later passes only add possibilities, so they would reject too.
            return rejection
        if header is None or translator.backedge == backedge:
            return Analysis(instructions=translator.instructions())
        backedge = translator.backedge
    return Analysis(
        rejection=Rejection(
            reason=RejectionReason.CONTROL_FLOW,
            message="the analysis of the loop did not converge",
        )
    )


def _check_loop(
    significant: list[dis.Instruction], header: int | None
) -> Analysis | None:
    """Check that the loop starting at offset ``header``, if any, has the
    one supported shape (see the module documentation).

    ``POP_JUMP_IF_TRUE`` is only accepted around a backedge, so it is
    checked here too.
    """

    def following(position: int) -> dis.Instruction | None:
        return significant[position + 1] if position + 1 < len(significant) else None

    if header is None:
        for instr in significant:
            if instr.opname == "POP_JUMP_IF_TRUE":
                return _reject_jump_if_true(instr)
        return None

    # The test: the first jump or return from the header on.
    test = next(
        instr
        for instr in significant
        if instr.offset >= header
        and (instr.opcode in dis.hasjump or instr.opname == "RETURN_VALUE")
    )
    if test.opname != "POP_JUMP_IF_FALSE":
        return _reject(
            RejectionReason.CONTROL_FLOW,
            test,
            f"a while loop must start with one comparison that leaves the loop "
            f"when it is false, found {test.opname} (while True is not "
            f"supported, nor not or or in a while condition)",
        )
    exit_ = test.argval
    last_backedge = max(
        instr.offset for instr in significant if instr.opname == "JUMP_BACKWARD"
    )
    if exit_ <= last_backedge:
        return _reject(
            RejectionReason.CONTROL_FLOW,
            test,
            "the while condition does not jump past the end of the loop",
        )

    for position, instr in enumerate(significant):
        inside = header <= instr.offset < exit_
        if instr.opname == "RETURN_VALUE" and inside:
            # CPython compiles a ``break`` followed by a short tail as a
            # copy of that tail, RETURN_VALUE included.
            return _reject(
                RejectionReason.CONTROL_FLOW,
                instr,
                "return or break inside a while loop is not supported",
            )
        if instr.opcode not in dis.hasjump or instr is test:
            continue
        if instr.opname == "POP_JUMP_IF_TRUE":
            # Only the inverted ``if`` at the end of the body: if the
            # comparison is false, the next instruction starts the next
            # iteration.
            after = following(position)
            if not (inside and after is not None and after.opname == "JUMP_BACKWARD"):
                return _reject_jump_if_true(instr)
        target = instr.argval
        if instr.opname == "JUMP_BACKWARD":
            continue  # Inside the loop, to the header: checked above.
        if inside and target >= exit_:
            return _reject(
                RejectionReason.CONTROL_FLOW,
                instr,
                "a jump out of a while loop (break, or and in the while "
                "condition) is not supported; only the condition may end it",
            )
        if not inside and header < target < exit_:
            return _reject(
                RejectionReason.CONTROL_FLOW,
                instr,
                "a jump into the middle of a while loop is not supported",
            )
    # The body must not fall through into the exit. CPython ends it with a
    # JUMP_BACKWARD, so this only guards against surprises.
    before_exit = [instr for instr in significant if instr.offset < exit_][-1]
    if before_exit.opname != "JUMP_BACKWARD":
        return _reject(
            RejectionReason.CONTROL_FLOW,
            test,
            "the body of the while loop does not end with its backward jump",
        )
    return None


def _reject_jump_if_true(instr: dis.Instruction) -> Analysis:
    return _reject(
        RejectionReason.CONTROL_FLOW,
        instr,
        "unsupported jump POP_JUMP_IF_TRUE: only if statements and while loops "
        "on integer comparisons are supported, not or and not",
    )


# What a variable can hold on the paths that reach an instruction: the
# argument as passed, a value stored by the function, or nothing.
_ARG = "argument"
_STORED = "stored"
_UNBOUND = "unbound"


class _State:
    """The facts that hold at one point of the bytecode, on every path to it.

    ``depth`` is the operand stack depth and ``bindings`` maps each variable
    to the set of things it may hold there.
    """

    __slots__ = ("bindings", "depth")

    def __init__(self, depth: int, bindings: dict[str, frozenset[str]]) -> None:
        self.depth = depth
        self.bindings = bindings

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _State)
            and self.depth == other.depth
            and self.bindings == other.bindings
        )

    __hash__ = None  # type: ignore[assignment]

    def copy(self) -> _State:
        return _State(self.depth, dict(self.bindings))

    def join(self, other: _State) -> _State:
        # Both sides are at a jump or a jump target, where the depth has
        # already been checked to be 0.
        return _State(
            0,
            {
                name: possible | other.bindings[name]
                for name, possible in self.bindings.items()
            },
        )


class _Translator:
    """Translate one code object's bytecode, in offset order.

    Every jump goes forward except the loop's backedges to ``header``, so
    offset order visits all forward predecessors of an instruction before
    the instruction itself. ``backedge`` is what the backedges brought back
    to the header on the previous pass, or ``None`` on the first; after
    :meth:`run`, ``self.backedge`` is what they bring back on this one.
    """

    def __init__(
        self,
        code: CodeType,
        slots: dict[str, int],
        header: int | None,
        backedge: _State | None,
    ) -> None:
        self.code = code
        self.header = header
        self.entering_backedge = backedge
        self.backedge: _State | None = None
        # Emitted instructions. A jump's operand is still a bytecode
        # offset here; ``instructions`` turns it into an index.
        self.emitted: list[Instruction] = []
        # For every reachable bytecode offset, the index of the first
        # instruction emitted at or after it.
        self.index_at_offset: dict[int, int] = {}
        # Local slots by variable name, in order of first assignment.
        self.slots = slots
        # Arguments read where they may hold either the argument or a stored
        # value; their slot is initialized from the argument on entry.
        self.copied_args: set[str] = set()

    def run(
        self, bytecode: list[dis.Instruction], significant: list[dis.Instruction]
    ) -> Analysis | None:
        code = self.code
        state: _State | None = _State(
            0,
            {
                name: frozenset({_ARG if index < code.co_argcount else _UNBOUND})
                for index, name in enumerate(code.co_varnames)
            },
        )
        # States arriving by forward jumps at targets not reached yet.
        incoming: dict[int, _State] = {}
        next_significant = {
            id(instr): significant[position + 1]
            if position + 1 < len(significant)
            else None
            for position, instr in enumerate(significant)
        }

        for instr in bytecode:
            name = instr.opname
            arrived = incoming.pop(instr.offset, None)
            if arrived is not None:
                if state is None:
                    state = arrived
                elif state.depth:
                    return _reject(
                        RejectionReason.CONTROL_FLOW,
                        instr,
                        "branches that join with a value on the stack "
                        "(a conditional expression inside an expression) "
                        "are not supported",
                    )
                else:
                    state = state.join(arrived)
            if instr.offset == self.header and state is not None:
                if state.depth:
                    return _reject(
                        RejectionReason.CONTROL_FLOW,
                        instr,
                        "a while loop entered with a value on the stack "
                        "is not supported",
                    )
                if self.entering_backedge is not None:
                    state = state.join(self.entering_backedge)
            if name in _BOOKKEEPING:
                if state is not None:
                    self.index_at_offset[instr.offset] = len(self.emitted)
                continue
            if name not in _SUPPORTED:
                return _reject(
                    RejectionReason.UNSUPPORTED_OPCODE,
                    instr,
                    f"unsupported opcode {name}",
                )
            if state is None:
                return _reject(
                    RejectionReason.CONTROL_FLOW,
                    instr,
                    "unreachable bytecode is not supported",
                )
            self.index_at_offset[instr.offset] = len(self.emitted)

            if name in _LOAD_LOCAL or name in _LOAD_LOCAL_PAIR:
                names = instr.argval if name in _LOAD_LOCAL_PAIR else (instr.argval,)
                for var in names:
                    rejection = self.load(instr, state, var)
                    if rejection is not None:
                        return rejection
            elif name == "STORE_FAST":
                self.emit(
                    Instruction(Op.STORE_LOCAL, self.slots[instr.argval]), state, -1
                )
                state.bindings[instr.argval] = frozenset({_STORED})
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
                self.emit(Instruction(Op.LOAD_CONST, value), state, 1)
            elif name == "BINARY_OP":
                op = _BINARY_OPS.get(instr.argrepr)
                if op is None:
                    return _reject(
                        RejectionReason.UNSUPPORTED_OPERATOR,
                        instr,
                        f"unsupported binary operator {instr.argrepr!r}",
                    )
                self.emit(Instruction(op), state, -1)
            elif name == "COMPARE_OP":
                op = _COMPARISONS.get(instr.argval)
                if op is None:
                    return _reject(
                        RejectionReason.UNSUPPORTED_OPERATOR,
                        instr,
                        f"unsupported comparison {instr.argval!r}",
                    )
                following = next_significant[id(instr)]
                if following is None or following.opname not in _CONDITIONAL_JUMPS:
                    return _reject(
                        RejectionReason.UNSUPPORTED_CONDITION,
                        instr,
                        "a comparison is only supported as an if or while "
                        "condition, not as a value",
                    )
                if following.opname == "POP_JUMP_IF_TRUE":
                    # It becomes JUMP_IF_FALSE on the opposite comparison.
                    op = _NEGATED[op]
                self.emit(Instruction(op), state, -1)
            elif name in _JUMPS:
                op = Op.JUMP_IF_FALSE if name in _CONDITIONAL_JUMPS else Op.JUMP
                self.emit(
                    Instruction(op, instr.argval),
                    state,
                    -1 if op is Op.JUMP_IF_FALSE else 0,
                )
                if state.depth:
                    return _reject(
                        RejectionReason.CONTROL_FLOW,
                        instr,
                        "jumps with a value on the stack (a conditional "
                        "expression inside an expression) are not supported",
                    )
                if name == "JUMP_BACKWARD":
                    # To the header, already translated; what arrives there
                    # is joined in on the next pass.
                    self.backedge = (
                        state.copy()
                        if self.backedge is None
                        else self.backedge.join(state)
                    )
                else:
                    target = instr.argval
                    previous = incoming.get(target)
                    incoming[target] = (
                        state.copy() if previous is None else previous.join(state)
                    )
                if op is Op.JUMP:
                    state = None
            else:  # RETURN_VALUE
                self.emit(Instruction(Op.RETURN), state, -1)
                state = None

        if state is not None or incoming:
            # CPython always ends a function with a return or a jump, and
            # never jumps past its last instruction.
            return Analysis(
                rejection=Rejection(
                    reason=RejectionReason.CONTROL_FLOW,
                    message="the bytecode does not end every path with a return",
                )
            )
        return None

    def emit(self, instruction: Instruction, state: _State, effect: int) -> None:
        self.emitted.append(instruction)
        state.depth += effect

    def load(self, instr: dis.Instruction, state: _State, var: str) -> Analysis | None:
        possible = state.bindings[var]
        if _UNBOUND in possible:
            return _reject(
                RejectionReason.UNSUPPORTED_VARIABLE,
                instr,
                f"local variable {var!r} may be read before it is assigned",
            )
        if possible == {_ARG}:
            self.emit(
                Instruction(Op.LOAD_ARG, self.code.co_varnames.index(var)), state, 1
            )
            return None
        if _ARG in possible:
            # The argument on some paths, a stored value on others: the
            # slot starts out holding the argument.
            self.copied_args.add(var)
        self.emit(Instruction(Op.LOAD_LOCAL, self.slots[var]), state, 1)
        return None

    def instructions(self) -> tuple[Instruction, ...]:
        # Arguments that need it are copied into their slots on entry, in
        # slot order. Straight-line code never needs a copy.
        prologue: list[Instruction] = []
        for var in sorted(self.copied_args, key=self.slots.__getitem__):
            prologue.append(Instruction(Op.LOAD_ARG, self.code.co_varnames.index(var)))
            prologue.append(Instruction(Op.STORE_LOCAL, self.slots[var]))
        shift = len(prologue)
        body = [
            Instruction(i.op, self.index_at_offset[i.arg] + shift)
            if i.op is Op.JUMP or i.op is Op.JUMP_IF_FALSE
            else i
            for i in self.emitted
        ]
        return tuple(prologue + body)


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
