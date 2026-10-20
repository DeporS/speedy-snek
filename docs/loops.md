# Design note: `while` loops over exact integers

This note describes how SpeedySnek compiles one `while` loop per function.
It builds on [`conditionals.md`](conditionals.md), whose rules for
comparisons, branches, joins and locals still hold. It is developer
documentation; the README only lists what is supported.

The goal is to run functions like these natively, repeated arithmetic and
all, inside one native call:

```python
@snekit
def sum_squares(n):
    total = 0
    i = 0
    while i < n:
        total += i * i
        i += 1
    return total

@snekit
def conditional_sum(n, cutoff):
    total = 0
    i = 0
    while i < n:
        if i < cutoff:
            total += i
        else:
            total += i * 2
        i += 1
    return total
```

Supported: one loop that is not nested in another loop, whose condition
is one integer comparison; arithmetic, assignments, augmented assignments
and `if`/`else` in its body; ordinary code (including `if`/`else` and
`return`) before and after it. Everything listed under
[Out of scope](#out-of-scope) is rejected, and the original function runs.

## 1. Recognizing the loop

### What CPython 3.14 emits

From `dis.get_instructions` for `sum_squares`:

```
     LOAD_SMALL_INT 0 / STORE_FAST total / LOAD_SMALL_INT 0 / STORE_FAST i
L1:  LOAD_FAST_BORROW_LOAD_FAST_BORROW (i, n)       <- loop header
     COMPARE_OP bool(<)
     POP_JUMP_IF_FALSE L2                            <- the only exit
     NOT_TAKEN
     ... total += i * i; i += 1 ...                  <- body
     JUMP_BACKWARD L1                                <- backedge
L2:  LOAD_FAST_BORROW total                          <- exit
     RETURN_VALUE
```

What the frontend relies on, each checked on real 3.14 output:

- **The test is not duplicated.** CPython 3.11 inverted loops by copying
  the test to the bottom; 3.12+ does not. A sweep of 972 generated loop
  functions (different conditions, bodies, code before and after) found
  the `COMPARE_OP` of the `while` line exactly once every time, and
  `test_differential_random_python_loops` requires every one of its
  random loops to be accepted. The frontend does not depend on this: a
  copied test would be a backedge to a block that is not a loop test,
  which is rejected.
- **Conditional jumps are never backward.** `POP_JUMP_IF_FALSE` and
  `POP_JUMP_IF_TRUE` always jump forward; the only backward jump is the
  unconditional `JUMP_BACKWARD`. A conditional backedge is spelled as a
  forward conditional jump around a `JUMP_BACKWARD`.
- **Backedges are tail-duplicated.** When a body ends in `if`/`else`,
  CPython copies the `JUMP_BACKWARD` into each branch instead of joining
  them first, so one loop can have several backedges, all to the header.
- **An `if` without `else` at the end of the body is inverted:**

  ```
  COMPARE_OP bool(<)
  POP_JUMP_IF_TRUE L2        <- if true, run the body of the if
  NOT_TAKEN
  JUMP_BACKWARD L1           <- else: next iteration
  L2: ... JUMP_BACKWARD L1
  ```

  This is the only `POP_JUMP_IF_TRUE` the frontend accepts: right after a
  `COMPARE_OP`, inside the loop body, and immediately followed (apart from
  `NOT_TAKEN`) by a `JUMP_BACKWARD`. It is normalized to the negated
  comparison followed by `JUMP_IF_FALSE`: on exact integers `not (a < b)`
  is exactly `a >= b` (there is no NaN), and nothing but exact integers
  ever reaches the IR.
- **Joins.** The exit `L2` is the target of the header's
  `POP_JUMP_IF_FALSE`, and nothing in the body jumps there. The header is
  reached by falling in from the code before the loop (or by forward jumps
  from it, as for an `if` right before the loop) and by the backedges.
  `if a < n: while a < n: ...` jumps from before the loop straight to the
  exit, past the whole loop; that is accepted.

### The accepted shape

In bytecode and, independently, in the IR (where `JUMP` may now go
backward), a function may contain at most one loop, of this shape:

```
prefix      [0, H)       acyclic code; may return, may jump to H or past X
header      [H, k]       straight-line test: loads, arithmetic, one comparison,
                         and at k the JUMP_IF_FALSE X that is the loop's only exit
body        (k, X)       acyclic code, forward jumps inside the body only,
                         every backward jump goes to H, no RETURN, and the
                         instruction at X - 1 is a backward jump
suffix      [X, end)     acyclic code
```

1. Every backward jump is an unconditional jump, and all of them target
   the same instruction `H`, the header. A second target means a second
   (nested or consecutive) loop.
2. The first jump or `RETURN` at or after `H` is the `JUMP_IF_FALSE` at
   `k`, after one comparison. Its target `X` lies beyond every backward
   jump.
3. Only `H` is entered from outside `[H, X)`: no jump from outside lands
   in `(H, X)`.
4. Only `k` leaves `[H, X)`: no jump from inside it targets `X` or beyond,
   it contains no `RETURN`, and its last instruction does not fall through
   into `X`.

The stack is empty at every jump and join, as before, now including the
header.

### Source forms with the same bytecode

Bytecode carries no `continue` or loop `else`. The frontend does not look
at source or line numbers, so it accepts these exactly because they are
indistinguishable from supported forms:

| Accepted form | Same bytecode as |
| --- | --- |
| `if i > c: continue` then `t += i` | `if i > c: pass` / `else: t += i` at the end of the body |
| `if not i < c: continue` then `t += 1` | `if i < c: t += 1` at the end of the body |
| `while c: ...` / `else: x += 1` (no `break`) | `while c: ...` followed by `x += 1` |
| `if a < n:` / `while a < n: ...` | (a loop inside an `if` is not nested in a loop) |

In general a `continue` is one more `JUMP_BACKWARD` to the header, the
same instruction CPython itself writes at the end of every branch that
ends the body. Its semantics are those of the bytecode, so every accepted
form computes exactly what Python computes.

`break` and `return` inside the loop are always visible: `break` either
jumps from the body to the exit or beyond (rule 4) or, for a short tail,
CPython copies the tail, including its `RETURN_VALUE`, into the body (also
rule 4). Both are rejected.

## 2. Control-flow validation with cycles

`speedysnek._interpreter.validate` and Rust's `program::validate`
implement the shape rules above independently, then make the same single
pass in index order as for acyclic programs. One pass is still enough:

- Every forward edge goes from a lower to a higher index, so every
  forward predecessor of an instruction is visited before it.
- The only other edges are the backedges into `H`. When the pass reaches
  `H` it has seen every *forward* predecessor (the prefix lies below `H`),
  so it takes the header's state from them alone; that is the state on the
  first iteration.
- At each backedge the pass checks that the state flowing back is
  compatible with the state assumed at `H`: an empty stack, and every local
  assigned at `H` still assigned. If so, joining the backedges into `H`
  would change nothing, so the one-pass result is a fixed point of the
  dataflow equations. The check never fails for programs built from
  stores (nothing unassigns a slot), but it is checked rather than
  assumed.

The shape rules make the loop *natural*: `H` dominates the body (rule 3),
and the only cycle goes through `H` (rule 1). Unsupported cycles,
self-loops, backward conditional jumps, jumps into the middle of a loop,
exits from its body and second loops are rejected with distinct messages.
Everything is linear in program length plus one bit set per pending jump
target; the existing limits (`MAX_JUMPS`, `MAX_LOCALS`, `MAX_ARGS`) and
opcode numbers are unchanged.

## 3. Definite initialization

Two analyses, one per layer, both conservative:

- **IR (must-analysis).** The set of slots assigned on every path. At a
  join it is the intersection of the incoming sets; at the header it is
  the intersection over the forward predecessors, which (section 2) is the
  fixed point once the backedges are accounted for. A store reached only
  through a backedge therefore never initializes a read in the header or
  the body: on the first iteration the store has not happened yet.
- **Frontend (may-analysis on bytecode).** For each variable at each
  instruction, the set of things it may hold: the argument, a stored
  value, or nothing. Joins are unions. With a loop, the translator runs its
  forward pass repeatedly, feeding the union of the backedge states into
  the header, until the header's state stops changing. Each pass can only
  add possibilities, each variable has three, so this converges after at
  most `2 * len(co_varnames) + 2` passes; the frontend stops with a
  rejection if it ever did not (it cannot).

So for

```python
def possibly_unbound(n):
    i = 0
    while i < n:
        value = i
        i += 1
    return value
```

`value` may be unbound at the header (it is on entry), therefore at the
exit, so the read after the loop is rejected and the function always runs
as Python: it returns for `n >= 1` and raises `UnboundLocalError` for
`n <= 0`, exactly as without `@snekit`.

An argument reassigned in the loop (`countdown`'s `n`) may hold the
argument or a stored value at the header. As for branches, it is read with
`LOAD_LOCAL` and the program starts with `LOAD_ARG n; STORE_LOCAL slot`.

## 4. Loop-carried values in Cranelift SSA

Locals stay Cranelift `Variable`s (`def_var` / `use_var`). Every block is
sealed once all its predecessors have been emitted. For a forward-only
block that is when code generation reaches it, as before. The header also
has backedge predecessors at higher indexes, so it is sealed only after
the whole program, including every backedge, has been emitted. Until then
`use_var` in the header (or in a block it dominates) creates a provisional
block parameter; sealing resolves each one from all predecessors: the
value before the loop on the entry edge, and the value at the end of the
iteration on each backedge. That is how the header sees updated `i`,
`total` and reassigned arguments on every iteration.

A copy (`last = i`) defines `last` as the SSA value `i` has at that
point. A later `i += 1` defines a new value for `i` and leaves `last`
alone, so copies keep normal integer assignment semantics.

Every `ADD`, `SUBTRACT` and `MULTIPLY` keeps its own overflow branch in the
block it belongs to. It runs every time its block runs, in the header, the
body or after the loop, even if its result is overwritten later, and never
on a branch not taken. Comparisons are signed `icmp`s, never
subtractions.

## 5. Bounded native execution and fallback

Native code runs with the GIL held and cannot be interrupted, so every
compiled loop carries a **backedge budget**:

- The budget is a compile-time constant, `DEFAULT_BACKEDGE_BUDGET` =
  1,000,000 backedges per call. On the benchmark machine the bodies of
  `sum_squares` and `conditional_sum` take 1.5–1.8 ns per iteration, so
  an exhausted attempt holds the GIL for about 1.5 ms, below CPython's
  default 5 ms thread switch interval
  ([measurements](../benchmarks/experiments/loops.md)). Larger bodies take
  proportionally longer per backedge.
- A program with a backward jump gets one extra `i64` variable, set to the
  budget on entry. Every backward jump checks it: if it is zero, the code
  returns `STATUS_BUDGET_EXHAUSTED` (and the index of that jump); otherwise
  it decrements the counter and continues at the header. So a call may
  take up to `budget` backedges natively; a loop of `n` iterations takes
  `n`.
- The check is emitted on **every** backward jump the code generator sees,
  whatever validation decided. Forward edges cannot form a cycle, so any
  cycle in the generated code passes a check: no serialized program, valid
  or not, can make native code loop more than `budget` times. The budget
  argument of the private `_native.compile_i64` must lie in
  `0..=MAX_BACKEDGE_BUDGET`, and `MAX_BACKEDGE_BUDGET` equals the default,
  so it can only make the budget smaller.
- Programs without a backward jump get no counter and no checks: their
  machine code is unchanged.

Exhaustion is a guard miss like overflow. The fast path returns `None`;
`@snekit` then runs the original function once, with the **original
arguments** (the native locals are discarded), and keeps the
specialization for later calls. A partial native result is never returned
and no new exception is raised. `CompiledI64.__call__`, the diagnostic
path, raises `NativeGuardFailure` with `GuardReason.BUDGET_EXHAUSTED`
(and the backedge's `instruction_index`), never `OVERFLOW`.

Restarting from the beginning is sound because the accepted native subset
is pure: exact-`int` arithmetic and comparisons on values in registers,
with no calls, no globals and no mutation of any Python object. The type
guards reject every argument that is not an exact `int` before any native
code runs, so no user-defined `__lt__`, `__add__` or `__iadd__` ever runs
natively and then again in Python.

The cost: a call that needs more than the budget does the first million
iterations twice, once natively and once in Python; measured, that made
such a call about 3–4% slower than plain Python. The check itself costs a
compare, a branch and a decrement per iteration: about 0.3 ns per
iteration on `sum_squares`, and nothing measurable on `conditional_sum`.
There is no "always faster" guarantee. The budget caps the native attempt only; a loop that
does not terminate in Python still does not terminate, but it runs in the
normal interpreter, where signals and thread switches work.

The private `compile_i64(..., backedge_budget=...)` argument and
`speedysnek._compiler._BACKEDGE_BUDGET` (read at compile time) let tests
use a small budget. There is no public tuning option.

### The reference interpreter's limit

`execute(program, args, max_backedges=...)` counts taken backward jumps in
the same unit, with the same default, and raises `ExecutionLimitExceeded`
when a call would take one more than `max_backedges`. That is a distinct
internal condition, neither a result nor a `GuardFailure`. With equal
limits, the interpreter raises exactly when native code returns
`STATUS_BUDGET_EXHAUSTED`. Every stretch between two backedges is acyclic,
so a run executes at most `(max_backedges + 1) * len(program)`
instructions: malformed or non-terminating test programs cannot hang the
test suite.

## Serialized format

Opcode numbers 1–16 are unchanged; no opcode is new. What changed is that
`JUMP` may now target an index at or before itself, under the shape rules
above. `JUMP_IF_FALSE` still only jumps forward. One status is new:

| Status | Number | `*result` |
| --- | --- | --- |
| `STATUS_BUDGET_EXHAUSTED` | 3 | index of the backward `JUMP` |

## Out of scope

`for` loops other than over `range(n)` (that one is compiled; see
[`for-range.md`](for-range.md)) and iterators in general (`GET_ITER`,
`FOR_ITER`); nested or consecutive loops;
`break`, `return` inside a loop, and loop `else` combined with `break`;
`while True:` and other conditions that are not one comparison (`while a:`,
`while a < b and b < c`, `while not a >= b`); calls; globals; closures;
object mutation; new types; releasing the GIL; a general deoptimization
framework; adaptive optimization. There is no new public API.
