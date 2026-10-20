# Design note: comparisons and acyclic `if`/`else`

This note describes how SpeedySnek compiles `if` statements whose condition
is one integer comparison. It is developer documentation; the README only
lists what is supported.

The goal is to compile functions like this one and run warmed calls
natively, without changing what any call returns or raises:

```python
@snekit
def calculate_score(wins, penalties):
    score = wins * 3 - penalties
    if score < 0:
        score = 0
    else:
        score += 5
    return score
```

Loops, early exits other than `return`, and everything listed under
[Out of scope](#out-of-scope) are rejected, and the original function runs.

## What CPython 3.14 emits

The frontend reads bytecode, not source. The relevant shapes, from
`dis.get_instructions` on CPython 3.14:

```
# def adjust(a, b):
#     result = a
#     if a < b:
#         result += b
#     return result * 2
LOAD_FAST                         0 (a)
STORE_FAST                        2 (result)
LOAD_FAST_BORROW_LOAD_FAST_BORROW 1 (a, b)
COMPARE_OP                       18 (bool(<))
POP_JUMP_IF_FALSE                 9 (to L1)
NOT_TAKEN
LOAD_FAST_BORROW_LOAD_FAST_BORROW 33 (result, b)
BINARY_OP                        13 (+=)
STORE_FAST                        2 (result)
L1: LOAD_FAST_BORROW              2 (result)
LOAD_SMALL_INT                    2
BINARY_OP                         5 (*)
RETURN_VALUE
```

- A condition is a `COMPARE_OP` immediately followed by
  `POP_JUMP_IF_FALSE`. The `COMPARE_OP` argument has the "convert to bool"
  flag set (`argrepr` is `bool(<)`); for exact `int` operands the
  comparison already returns a `bool`, so the flag changes nothing.
- `POP_JUMP_IF_FALSE` always jumps forward. `NOT_TAKEN` follows it and only
  records branch statistics.
- An `else` ends the `if` body with `JUMP_FORWARD` to the join point.
  `elif` is a nested `if` in the `else` branch.
- **CPython duplicates short exit blocks.** In `calculate_score` and
  `choose` the final `return score` is copied into both branches, so the
  bytecode has two `RETURN_VALUE` and no join at all. Only a longer tail,
  as in `adjust`, gets a real join.

Because of that duplication, several source forms compile to the same
bytecode, and the frontend does not try to tell them apart (it would need
the source or line-number heuristics):

| Accepted form | Same bytecode as |
| --- | --- |
| `if c: x = 1` `else: x = 2` / `return x` | `if c: x = 1; return x` / `x = 2; return x` |
| `return a if a < b else b` | `if a < b: return a` / `return b` |
| `x = a if a < b else b` / `return x` | an `if`/`else` assigning `x` in each branch |
| `if a < b and b < c: ...` | `if a < b: if b < c: ...` |

So a `return` inside a branch, a conditional expression whose two arms end
up in separate paths, and an `and` of comparisons in an `if` are accepted
exactly when CPython emits one of these shapes. None of them needs new
machinery: every path still ends with its own return of an `int`.

## Normalized IR

The IR stays a flat tuple of instructions for a stack machine. Eight
operations are new:

| Operation | Operand | Effect |
| --- | --- | --- |
| `LESS_THAN`, `LESS_EQUAL`, `GREATER_THAN`, `GREATER_EQUAL`, `EQUAL`, `NOT_EQUAL` | none | pop right, pop left (integers), push a *condition* |
| `JUMP_IF_FALSE t` | target index | pop a condition; if false continue at `t`, else at the next instruction |
| `JUMP t` | target index | continue at `t` |

`RETURN` is unchanged, except that a program may now contain several: each
one ends its path.

### Branch targets and basic blocks

A jump's operand is the index of an instruction in the same program, and it
must be **ahead of the jump** (`index < target < len(program)`). Every
index is an instruction boundary, so there are no "mid-instruction" targets
to validate, and forward-only targets make cycles impossible.

Basic blocks are implicit: a block starts at index 0, at every jump target,
and after every `JUMP_IF_FALSE`; it ends at a jump, a `RETURN`, or just
before the next block start. Nothing else is stored. Because every edge
points forward, index order is a topological order of the control-flow
graph: when an instruction is reached, all of its predecessors have already
been visited. Both validators and the code generator rely on this to do a
single pass. Supporting loops later means allowing backward targets and
replacing that single pass with a worklist; the representation itself does
not need to change.

### Conditions

A condition is a stack value of a separate kind, not an integer. Its only
permitted use is by the `JUMP_IF_FALSE` that **immediately follows** the
comparison. It is never stored in a local, returned, used in arithmetic or
compared again. So a comparison result can never surface as a Python value:
no `0`/`1` and no `bool` is ever returned, and the decorator's exact-`int`
guards are unchanged (`True` and `False` arguments still fall back).

The reference interpreter holds a condition as a Python `bool` for that one
step; the native backend holds it as the `i8` result of a Cranelift `icmp`.

### Stack state at joins

The operand stack must be **empty after every jump and on every edge into a
jump target**. Values therefore cross from one block to another only through
local slots, and "stack heights and kinds agree at joins" reduces to "both
are empty". This is what `if` statements compile to. It deliberately rules
out bytecode such as `1 + (a if c else b)`, where CPython jumps with a value
on the stack; that is rejected by the frontend and would be rejected by
both validators.

### Locals across branches

Every slot starts out unassigned, as before. A `LOAD_LOCAL` is valid only if
its slot is stored on **every** path from the start of the program to it.
The validators compute this with the forward pass: the set of definitely
assigned slots at a join is the intersection of the sets on the incoming
edges.

The frontend makes the same kind of analysis on bytecode, one level up:
for every variable it tracks what it may hold at each instruction (the
argument as passed, a value the function stored, or nothing) and joins
these by union.

- A read where the variable may be unbound is rejected
  (`UNSUPPORTED_VARIABLE`); the original function then raises its usual
  `UnboundLocalError` on the paths where it is unbound, and returns
  normally on the others.
- A read of an argument that is still the argument on every path is
  `LOAD_ARG`, as in straight-line code.
- A read of an argument that is reassigned on some paths but not others
  (`if a < b: a = b` / `return a`) is `LOAD_LOCAL` of the argument's slot,
  and the program starts with `LOAD_ARG n; STORE_LOCAL slot`. That keeps
  the IR rule "slots start unassigned" intact without a special case.
  Straight-line code never needs this, so its IR is unchanged.

### Merging different assignments to the same local

The native backend no longer maps slots to SSA values by hand. Each local
slot is a Cranelift `Variable`: `STORE_LOCAL` is `def_var`, `LOAD_LOCAL` is
`use_var`. `cranelift_frontend`'s SSA construction then adds a block
parameter at a join for every slot whose value differs between incoming
paths, and every jump passes its own path's value. A path that did not
assign the slot passes the value the slot had before the branch. Inside a
block, `use_var` simply returns the last defined SSA value, so straight-line
code still copies nothing at run time.

Each IR block start gets its own Cranelift block. Since every predecessor
of a block has a lower index, a block is sealed as soon as code generation
reaches it. Validation guarantees definite assignment before codegen runs,
which matters: Cranelift would silently zero-initialize a variable that is
used before it is defined.

### Comparisons in native code

A comparison is one signed `icmp` (`slt`, `sle`, `sgt`, `sge`, `eq`, `ne`)
feeding the `brif` of the following `JUMP_IF_FALSE`. It is never computed
by subtracting the operands, which could overflow at the `i64` limits.

### Overflow and the selected branch

Every `ADD`, `SUBTRACT` and `MULTIPLY` keeps its own overflow check, placed
in the block the operation belongs to, so it runs exactly when the
operation runs: always on the selected path, even if the result is later
overwritten or never read, and never on a path not taken. An overflow on
the selected path returns `STATUS_OVERFLOW` as before; the decorator then
runs the original function once and keeps the specialization.

## Validation

`speedysnek._interpreter.validate` (Python) and `program::validate` (Rust)
implement the same rules independently. A program is valid only if:

1. it is non-empty and contains a `RETURN`;
2. every jump targets an index ahead of itself and inside the program;
3. every instruction is reachable from index 0;
4. no instruction pops from an empty stack, and each finds the kind it
   needs: only `JUMP_IF_FALSE` pops a condition, everything else pops
   integers;
5. every comparison is immediately followed by `JUMP_IF_FALSE`;
6. the stack is empty after every jump and on every edge into a jump target;
7. every `LOAD_LOCAL` reads a slot stored on every path to it;
8. every path ends with a `RETURN` that finds exactly one value; nothing
   falls off the end.

Existing limits stay: `LOAD_ARG` below `arg_count`, local indexes below
`MAX_LOCALS`, constants in `i64`, at most `MAX_ARGS` arguments. One limit is
new: at most `MAX_JUMPS` (4096) jumps. Rust keeps one bit set of assigned
locals per pending jump target, and the limit bounds that memory to 32 MiB
even for adversarial input. `speedysnek._compiler` checks it too and raises
`UnsupportedProgramError`.

Rule 3 means unreachable instructions cannot hide malformed code: there is
no "dead" instruction that validation skips. The frontend rejects
unreachable bytecode and exception tables before building IR.

Malformed serialized programs passed directly to
`speedysnek._native.compile_i64` raise `TypeError` or `ValueError`; the
code generator also re-checks every invariant it relies on and returns an
error rather than panicking.

## Serialized format

Opcode numbers 1–8 are unchanged. The new ones are appended:

| Opcode | Number | Operand |
| --- | --- | --- |
| `JUMP` | 9 | target index |
| `JUMP_IF_FALSE` | 10 | target index |
| `LESS_THAN` | 11 | none |
| `LESS_EQUAL` | 12 | none |
| `GREATER_THAN` | 13 | none |
| `GREATER_EQUAL` | 14 | none |
| `EQUAL` | 15 | none |
| `NOT_EQUAL` | 16 | none |

The one rule that changed for existing programs is about `RETURN`: before,
a program had exactly one, at the end; now every path ends with one. A
straight-line program with code after a `RETURN` is still rejected, now
because that code is unreachable.

## Rejected bytecode

| Bytecode | Source | Reason |
| --- | --- | --- |
| `JUMP_BACKWARD`, `FOR_ITER` | loops | `CONTROL_FLOW` |
| `POP_JUMP_IF_TRUE` | `or`, `not` | `CONTROL_FLOW` |
| `POP_JUMP_IF_NONE`, `POP_JUMP_IF_NOT_NONE` | `is None` tests | `CONTROL_FLOW` |
| a jump or join with a value on the stack | conditional expression inside an expression | `CONTROL_FLOW` |
| exception table, unreachable code | `try`, `with` | `CONTROL_FLOW` |
| `TO_BOOL`, `IS_OP`, `CONTAINS_OP` before the jump | `if a:`, `is`, `in` | `UNSUPPORTED_CONDITION` |
| `COMPARE_OP` not followed by `POP_JUMP_IF_FALSE` | `x = a < b`, `return a < b` | `UNSUPPORTED_CONDITION` |
| `SWAP`, `COPY` | chained comparisons `a < b < c` | `UNSUPPORTED_OPCODE` |
| `LOAD_CONST None` | a path that returns `None` | `UNSUPPORTED_CONSTANT` |

## Out of scope

Loops and backward jumps; `and`/`or`/`not` beyond the bytecode-identical
forms above; general truthiness; chained comparisons; comparison results as
values; calls; exceptions; globals; closures; object mutation; adaptive
optimization. There is no new public API.
