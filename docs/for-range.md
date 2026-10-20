# Design note: `for` loops over `range(n)`

This note describes how SpeedySnek compiles one `for` loop over the
builtin `range` with one argument. It builds on
[`loops.md`](loops.md), whose loop shape, validation, definite
initialization, SSA construction and backedge budget are reused unchanged,
and on [`conditionals.md`](conditionals.md). It is developer documentation;
the README only lists what is supported.

The goal is to run functions like these natively:

```python
@snekit
def sum_squares(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

@snekit
def conditional_sum(n, cutoff):
    total = 0
    for i in range(n):
        if i < cutoff:
            total += i
        else:
            total += i * 2
    return total
```

Supported: `for name in range(bound):` where `range` is the builtin,
`bound` is an integer expression of the subset (locals, arguments,
constants, `+`, `-`, `*`) and `name` is one local variable; one such loop
per function, not nested and not combined with a `while` loop; anything
the subset allows in its body, before it and after it. Only `range(n)`:
`range(start, stop)`, `range(start, stop, step)`, keyword arguments, other
iterables, iterator arguments and aliases of `range` are rejected, and the
original function runs.

## 1. What CPython 3.14 emits

From `dis.get_instructions` for `sum_squares`:

```
     LOAD_SMALL_INT 0 / STORE_FAST total
     LOAD_GLOBAL 1 (range + NULL)       <- the callable, then NULL
     LOAD_FAST_BORROW n                 <- the bound expression
     CALL 1                             <- range(n): pops 3, pushes the range
     GET_ITER                           <- replaces it with its iterator
L1:  FOR_ITER (to L2)                   <- header and only test
     STORE_FAST i                       <- the loop variable
     ... total += i * i ...             <- body
     JUMP_BACKWARD (to L1)              <- backedge
L2:  END_FOR                            <- FOR_ITER's argval
     POP_ITER                           <- removes the iterator
     LOAD_FAST_BORROW total / RETURN_VALUE
```

What the frontend relies on, each checked on real 3.14 output (the
examples in `test_for_range.py` show each form):

- `LOAD_GLOBAL` with the low bit of its `arg` set pushes the global and a
  `NULL`; `argval` is the name. `range(a, b)` is `CALL 2`, `range(stop=n)`
  is `CALL_KW`, `range(*a)` is `CALL_FUNCTION_EX`.
- The iterator stays on the stack under everything the body does.
  `FOR_ITER`'s `argval` is the offset of `END_FOR`; when the iterator is
  exhausted, control continues there (or just after it) and `POP_ITER`
  drops the iterator.
- Every backward jump of the loop is a `JUMP_BACKWARD` to `FOR_ITER`. As
  for `while` loops, CPython copies it into each branch of an `if`/`else`
  that ends the body, and an `if` without `else` at the end of the body is
  inverted around one (`POP_JUMP_IF_TRUE`).
- `break` first pops the iterator (`POP_TOP`) and then either jumps past
  `POP_ITER` or, for a short tail, copies the tail with its
  `RETURN_VALUE` into the body. `return` inside the loop is
  `SWAP`/`POP_TOP`/`RETURN_VALUE`. A body that always returns has no
  `JUMP_BACKWARD` at all.
- A comprehension (`[i for i in range(n)]`) is inlined with
  `LOAD_FAST_AND_CLEAR`, `SWAP` and an exception table; a target other
  than one local is `UNPACK_SEQUENCE`, `STORE_ATTR`, `STORE_SUBSCR` or
  `STORE_GLOBAL` after `FOR_ITER`.
- No `for` statement starts with a value on the stack: it is a statement,
  and the comprehension forms are rejected anyway.

## 2. Recognizing the pattern

`speedysnek._frontend.cpython314` does not enable `LOAD_GLOBAL`, `CALL`,
`GET_ITER`, `FOR_ITER`, `END_FOR` or `POP_ITER` in general. It looks for
exactly one `FOR_ITER` (a second one is a second loop) and matches the
instructions around it, ignoring bookkeeping (`NOT_TAKEN`, `NOP`, ...):

1. The instruction before `FOR_ITER` is `GET_ITER`, and the one before
   that is `CALL` with `arg == 1`.
2. Walking back from the `CALL`, the bound is a run of single and paired
   local loads, constants and `BINARY_OP`s that leaves exactly one value.
   Anything else there (a call, an attribute, a jump of a conditional
   expression, a unary operator) rejects the loop. The operators
   themselves are checked later, like everywhere else (`//` is rejected
   as an unsupported operator).
3. The instruction before the bound is `LOAD_GLOBAL` of `range` with the
   `NULL` bit set. A local or argument named `range`, an alias
   (`r = range; for i in r(n)`), `iter(range(n))` and `a.range(n)` all
   fail here.
4. The instruction after `FOR_ITER` is `STORE_FAST`.
5. `FOR_ITER`'s target is `END_FOR`, immediately followed by `POP_ITER`.

Every other `LOAD_GLOBAL`, `CALL`, `GET_ITER` and so on in the function
stays an unsupported opcode. The loop shape rules of `loops.md` then apply
with `FOR_ITER` as the test and `END_FOR` as the exit: every backward jump
goes to `FOR_ITER`, nothing in the body returns or jumps out (so `break`
and `return` are rejected), and nothing jumps into the loop. In addition,
no jump may land inside the `for` statement before the body (between
`LOAD_GLOBAL` and `FOR_ITER`, where the callable, the bound or the
iterator is on the stack) or on `END_FOR`/`POP_ITER` from anywhere but
`FOR_ITER`. A jump from before the loop to `LOAD_GLOBAL` itself (an `if`
right before the loop) or past `POP_ITER` (a loop inside an `if`) is
fine.

All of this reads instruction names and arguments only. Nothing is looked
up, imported or called during analysis.

Rejections use the new `RejectionReason.UNSUPPORTED_ITERATION` for an
iterable or target outside the pattern, and the existing `CONTROL_FLOW`
and `UNSUPPORTED_VARIABLE` for shape and initialization problems.

## 3. Lowering: native loop state

The range object and its iterator never reach the IR. The frontend
translates the pattern into the `while` loop shape of `loops.md`, with two
compiler-owned local slots, numbered after every variable's slot (slots
are numbered by first `STORE_FAST`, so `len(slots)` and `len(slots) + 1`
can never collide with a user variable):

```
<bound>                                         bound expression
STORE_LOCAL bound                               CALL 1
LOAD_CONST 0; STORE_LOCAL counter               GET_ITER
header:
  LOAD_LOCAL counter; LOAD_LOCAL bound
  LESS_THAN; JUMP_IF_FALSE exit                 FOR_ITER
  LOAD_LOCAL counter; STORE_LOCAL i             STORE_FAST i
  LOAD_LOCAL counter; LOAD_CONST 1; ADD
  STORE_LOCAL counter
  <body>
  JUMP header                                   each JUMP_BACKWARD
exit:                                           END_FOR, POP_ITER: nothing
```

The operand stack the translator tracks is the IR's. `LOAD_GLOBAL` emits
nothing and requires an empty IR stack; the bound pushes one value, which
`CALL` stores. From then on the IR stack is empty exactly where CPython's
holds only the iterator, so the existing rule "the stack is empty at every
jump and join" still holds, at the header and in the body. No new opcode,
operand or status exists: the program is an ordinary one-loop program, and
both validators (`speedysnek._interpreter.validate` and Rust's
`program::validate`) check it with the rules they already had. Neither
changed. Opcode numbers are unchanged.

What this gives, matching `range(n)` on exact integers:

- **The bound is evaluated once.** It lives in its own slot; reassigning
  `n` in the body cannot change it.
- **Negative and zero bounds** fail `counter < bound` on entry: no
  iteration runs.
- **The counter and the visible variable are separate.** `i` receives a
  copy of the counter at the top of each iteration, then the counter is
  incremented. Assigning `i` in the body does not change the next item,
  and after the loop `i` holds whatever was stored in it last: the last
  item, or the body's last assignment. Nothing writes the counter back to
  `i` on exit.
- **Zero iterations preserve prior bindings.** The exit is reached from
  the header without touching `i`'s slot, so `i` keeps the value it had
  before the loop: a local assigned earlier, or the argument (an argument
  reassigned by the loop is read with `LOAD_LOCAL` after the usual
  prologue `LOAD_ARG; STORE_LOCAL`, as for `while` loops).
- **Possibly unbound variables are rejected.** The frontend's
  may-analysis sees that `i` (or a variable first assigned in the body) is
  unbound at the header on entry, so it is possibly unbound at the exit,
  and a read after the loop is rejected. The function then always runs as
  Python, which raises `UnboundLocalError` for an empty range, exactly as
  without `@snekit`.
- **The counter never wraps or traps.** The increment runs only after
  `counter < bound`, and `bound` fits in an i64, so `counter + 1 <=
  I64_MAX`. It is an ordinary checked `ADD` whose overflow branch can
  never be taken. A test starts the counter just below `I64_MAX` in a
  hand-edited program to check the boundary natively.
- **Checked arithmetic is unchanged.** The bound's operations and the
  body's are ordinary `ADD`, `SUBTRACT` and `MULTIPLY`, each checked where
  it runs, even when its result is overwritten later. Overflow anywhere,
  in the bound or in iteration 5, is a guard miss.

Only exact `int`s reach the native code, and `range` on an exact `int`
never calls `__index__`. A `bool`, an `int` subclass or an object with
`__index__` fails the existing argument guards, and the original function
runs, where `range` calls its `__index__` once, as usual.

## 4. The range binding guard

The lowering is only the function's meaning while the function's `range`
is the builtin. The frontend cannot know that, so a supported `for` loop
sets `Analysis.assumed_builtins = ("range",)`, and `@snekit` guards it.

### What is checked

In CPython 3.14 a function's `LOAD_GLOBAL` looks in `func.__globals__`
and then `func.__builtins__`. Both are fixed when the function is
created: `__builtins__` is `__globals__["__builtins__"]` at that time (a
module's dict, or the object itself), otherwise the interpreter's
builtins. When both are exact dicts, the lookup is a plain dict lookup;
otherwise CPython calls `__getitem__` on the mapping.

When compiling such a function, `_native.RangeGuard(func.__globals__,
func.__builtins__)`:

1. requires both to be exact dicts (a dict subclass or other mapping is
   looked up with user code);
2. watches both with a CPython dict watcher (`PyDict_AddWatcher`, public C
   API since 3.12) and registers the guard's flag, *before* looking;
3. walks the globals and then the builtins with `PyDict_Next`, requiring
   every key to be an exact `str` (a key of another type could have an
   `__eq__` that makes CPython's own lookup find it), and compares keys
   with `"range"` by their characters;
4. compares the value found by identity with the address of CPython's
   `PyRange_Type`, the genuine `range` type. It does not look at the
   `builtins` module this process imported, which may not be the
   function's builtins and may itself be patched.

If any step fails, the function is disabled (`RANGE_UNGUARDED`) and
always runs as Python: a global `range`, a custom builtins dict whose
`range` is something else, a missing `range` (Python raises `NameError` as
usual), a dict subclass or a non-`str` key. A global `range` that *is* the
builtin is accepted. No step runs Python code: no `__eq__`, `__hash__`,
`__getitem__` or call.

### When it is invalidated

CPython calls the watcher *before* every change to a watched dict. The
callback clears every registered flag for that dict on:

- adding, replacing or deleting the key `"range"` (storing the identical
  object again is no change, and CPython does not report it);
- adding, replacing or deleting any key that is not an exact `str`;
- clearing the dict, merging a whole dict into it while it is empty
  (`CLONED`), or deallocating it.

Changes to other `str` keys are ignored, so ordinary global assignments
keep the specialization. The callback runs no Python code: it compares an
exact `str` key by its characters, then takes a Rust lock over a map from
dict address to the flags of live guards (weak references; the map holds
no Python objects). A dict's entry is removed when it is deallocated, so a
new dict at the same address, which is not watched, cannot reach an old
guard. Watched dicts stay watched; for unrelated keys the cost is one call
and one string comparison per change.

The flag never becomes true again. This is deliberately conservative: a
`range` deleted from the globals (so the builtin is found again), a
global `range` set to the builtin, or any non-`str` key invalidates the
guard even though Python's result would not change.

### Where it is checked

The guard protects the only place where native code stands in for the
Python function, the `@snekit` wrapper. The compiled program itself never
mentions `range`: `compile_i64` and `CompiledI64` run IR, whose meaning
does not depend on any binding, so they need no guard.

For a function with a guard, the specialization's `function` is a
`_native.GuardedI64Function` (the `I64Function` plus the guard). Before
every native call the wrapper checks:

1. `__code__` is still the compiled code object (as before);
2. for a `GuardedI64Function`, the guard is still valid: one relaxed
   atomic load;
3. keywords and arity, then the native fast path (as before).

If the guard has failed, nothing native runs. The wrapper calls the
state's `binding_changed()`, which disables the specialization for good
(`RANGE_CHANGED`), then runs the original function exactly once with the
original arguments, so it uses whatever `range` now is. Like a replaced
`__code__`, it is never re-enabled, even if the original binding is
restored. If `__code__` was replaced too, the code check comes first and
the reason is `CODE_CHANGED`; either way the function runs as Python from
then on.

A function that does not use `range` gets no guard: its `function` is the
plain `I64Function`, which the wrapper recognizes first, exactly as
before. `GuardedI64Function.try_call` (the specialization's `fast_call`)
treats a failed guard as one more guard miss and returns `None`.

The guard is created before Cranelift compiles anything, so every change
after the proof is seen, and there is no window between checking and
watching: the dicts are watched first.

## 5. Execution budget

Nothing changes: each `JUMP_BACKWARD` is one IR backward `JUMP`, so a
`for` loop of `n` iterations takes exactly `n` backedges, like the
equivalent `while` loop, and `n` up to the budget (1,000,000) completes
natively. A larger `n`, including bounds up to `I64_MAX`, stops at the
budget: the fast path returns `None`, the diagnostic path raises
`NativeGuardFailure(BUDGET_EXHAUSTED)`, and `@snekit` runs the original
function once with the original arguments (so the work up to the budget
is done twice; see `loops.md`). Native locals, the hidden counter
included, are discarded. The reference interpreter's `max_backedges`
counts the same jumps. Programs sent directly to `_native.compile_i64` are
budgeted exactly as before.

## Source forms with the same bytecode

The frontend never looks at source or line numbers. These forms compile
because their bytecode is identical to a supported one (tests check the
identity):

| Accepted form | Same bytecode as |
| --- | --- |
| `if i > c: continue` then `t += i` | `if i > c: pass` / `else: t += i` at the end of the body |
| `for i in range(n): ...` / `else: t += 100` (no `break`) | the loop followed by `t += 100` |
| `if a < n:` / `for i in range(n): ...` | (a loop inside an `if` is not nested in a loop) |

A `continue` is one more `JUMP_BACKWARD` to `FOR_ITER`, the instruction
CPython writes at the end of every branch that ends the body anyway.

## Limitations

- Only `range` with one positional argument, iterated directly by one
  `for` statement, assigning one local variable.
- One loop per function: no nested loops, no second `for`, no `for`
  combined with a `while`.
- No `break`, no `return` inside the loop, no reading the loop variable
  after the loop if an empty range would leave it unassigned.
- The guard is all or nothing and permanent: any relevant change disables
  the specialization; there is no re-validation or recompilation.
- The guard rejects namespaces it cannot check without running Python
  code: dict subclasses or other mappings as globals or builtins, and any
  key that is not an exact `str`.
- Watched dicts stay watched for the life of the process (or the dict).
  An interpreter has at most 8 dict watchers in CPython 3.14, shared with
  CPython itself and other extensions; SpeedySnek registers one, on first
  use. If none is free, compilation fails (`COMPILER_ERROR`) and the
  function runs as Python.
- No GIL release, no adaptive optimization, no new numeric types, no new
  public API.
