# `snek.peek()`: what did the snek do with this function?

`peek()` is SpeedySnek's diagnostics API. After [`hunt()`](hunt.md) has
inspected a module and [`bite()`](bite.md) (or `@snekit`) has wrapped its
functions, `peek()` reports where each wrapper is now: still warming up,
compiled, or disabled, and why.

```python
import calculations
import speedysnek as snek

snek.bite(calculations)

for _ in range(100):
    calculations.sum_squares(10_000)

report = snek.peek(calculations.sum_squares)
```

```text
🐍 Snek peeked at sum_squares.

status:           compiled
calls observed:   100
warmup calls:     8
compile attempts: 1
native attempts:  92
compiler:         Cranelift i64 specialization
```

`peek()` is read-only. It never calls, decorates, compiles, warms up,
benchmarks or replaces anything, and it changes no count or state. Every
number it shows is state the wrapper already keeps; `peek()` adds nothing
to the call path. It reports what happened, not how fast it was: there is
no timing in it and no promise of a speedup.

## What you can peek at

```python
snek.peek(calculations.sum_squares)  # one wrapper: one finding
snek.peek(calculations)              # every wrapper in a module
```

- **A `@snekit` wrapper**, whether from `@snekit` in the source or from
  `bite()`, or a method bound from one: one finding.
- **A plain Python function**: one finding with status `not_bitten`. If
  it is the original of a function that `bite()` has since wrapped (a
  reference taken before biting), the finding says which module name
  holds the wrapper. If it is marked with
  [`@nosnek`](bite.md#opting-out-with-nosnek), the summary says so, and
  that SpeedySnek will not wrap it automatically, and `reason` is
  `"opted_out"`. It is not reported as disabled: it was never wrapped.
- **A module**: one finding per `@snekit` wrapper of the module's own
  functions under a public name, in definition order. Plain functions
  (`@nosnek` ones included), wrappers imported from other modules, private names, classes and
  callable objects are left out. A module with no wrappers gives an empty
  report.

Anything else, such as a class, a builtin, a callable object or a module
name as a string, raises `TypeError`.

## Statuses

| status | meaning |
| --- | --- |
| `cold` | warming up: the first 8 calls run the original function; the next one tries to compile it |
| `compiled` | compiled code is cached; eligible calls try it and fall back to Python otherwise |
| `disabled` | compilation was refused or the compiled code was invalidated; every call runs as Python from now on |
| `not_bitten` | a plain function, not wrapped with `@snekit`: use `snek.bite(module)` or `@snekit` first |

A disabled finding explains why in a few words, and keeps the underlying
diagnostic:

| `reason` | summary |
| --- | --- |
| `rejected` | the frontend's explanation, as `hunt()` gives it, such as "f-strings are not supported" |
| `unsupported` | beyond the native compiler's limits |
| `compiler_error` | the compiler failed unexpectedly |
| `interrupted` | compilation was interrupted (for example by Ctrl+C) |
| `code_changed` | its `__code__` was replaced after compilation |
| `range_unguarded` | `range` could not be proven to be the builtin |
| `range_changed` | its `range` binding changed after compilation |
| `native_error` | a native call failed unexpectedly |

```text
🐍 Snek peeked at format_report.

status:           disabled
calls observed:   9
warmup calls:     8
compile attempts: 1
reason:           f-strings are not supported (line 12)
```

## Calls, warmup and native attempts

These counts come from different places and mean different things:

- **calls observed** (`call_count`) is the wrapper's own `call_count`:
  every call it has received, including calls that raised. It is the
  same public attribute you can read on the wrapper, and you can also
  assign to it, so it is an observation, not proof.
- **warmup calls** (`warmup_calls`) are the calls the JIT state counted
  while cold, at most 8. While cold, `warmup_remaining` says how many
  are left before the next call tries to compile.
- **compile attempts** (`compile_attempts`) is 0 or 1: SpeedySnek tries
  to compile a function once.
- **native attempts** (`native_attempts`) is a lower-level count kept by
  the compiled code itself: the calls that entered its fast path. An
  attempt is not a native result. The fast path checks the arguments and
  runs the machine code, and a call that has a non-`int` argument, a value
  outside 64 bits, overflows or uses up its loop budget falls back to
  Python. SpeedySnek does not count which attempts succeeded, so `peek()`
  does not report it. Calls that never reach the fast path, such as
  calls with keyword arguments or a missing default, are not attempts.

In the example above, 100 calls were observed: 8 ran during warmup, the
ninth compiled the function and was its first native attempt, and the
other 91 attempts followed.

## Aliases

In a module, one wrapper under several public names is reported once,
under its first name; `finding.names` lists all of them, and the printed
report shows the rest as `aliases`. Two different wrappers of one
function are two findings.

## The report

`peek()` prints the report to `sys.stdout` and returns it. Pass
`quiet=True` to print nothing, or `stream=` to print to another text
file. `str(report)` and `report.render()` give the same text. A module
report names each function and ends with the counts:

```text
🐍 Snek peeked at calculations.

sum_squares
  status:           compiled
  aliases:          total_of_squares
  calls observed:   10
  warmup calls:     8
  compile attempts: 1
  native attempts:  2
  compiler:         Cranelift i64 specialization

clamp
  status:           cold
  calls observed:   1
  warmup calls:     1 of 8, 7 to go
  compile attempts: 0

1 compiled, 1 cold, 0 disabled.
```

`PeekReport`:

- `findings`: a tuple of `PeekFinding` (iterating the report yields them
  too);
- `module_name`: the peeked module's name, or `None` for a function;
- `cold`, `compiled`, `disabled`, `not_bitten`: the findings with that
  status.

`PeekFinding`:

- `name` and `names`: the function's name; in a module, its first public
  name and every alias;
- `wrapper`: the inspected `@snekit` wrapper (`None` when not bitten);
- `function`: the function the wrapper runs, read from its JIT state,
  not from `__wrapped__`;
- `status`: `"cold"`, `"compiled"`, `"disabled"` or `"not_bitten"`;
- `summary`: a short, human-readable explanation;
- `call_count`, `warmup_calls`, `warmup_remaining`, `compile_attempts`,
  `native_attempts`: the counts described above;
- `compiler`: what compilation produced, today always
  `"Cranelift i64 specialization"`; it stays set after the code is
  invalidated, and the printed report then marks it "no longer used";
- `reason`, `detail`: for a disabled function, the stable reason code
  and the underlying diagnostic message (`reason` is also `"opted_out"`
  for a plain function marked with `@nosnek`);
- `lineno`: the source line of a rejected instruction, where known;
- `rejection`, `compile_error`, `native_error`: the frontend rejection or
  the stored exception, where there is one.

A field is `None` when the information does not exist for that state,
for example `native_attempts` before compilation, or every count for a
function that is not bitten. `peek()` never guesses one.

## Limitations

- `peek()` describes state. It does not time anything, and "compiled"
  does not mean "faster": see the benchmark notes in the README.
- Counts are read without stopping other threads, so a function being
  called concurrently may move on while it is being peeked at.
- Only wrappers made by `@snekit` (directly or through `bite()`) are
  recognized. An object that merely carries `_jit_state`, `__wrapped__`
  or `call_count` attributes is not one, and a wrapper's report always
  comes from the JIT state it really uses, even if those attributes are
  rewritten.
- `peek()` runs no user code: modules and objects are read through their
  own descriptors, as in `hunt()` and `bite()`. It never loads the native
  extension or the compiler; a wrapper cannot exist before the extension
  is loaded.
