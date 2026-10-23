# `snek.chill()`: undoing a bite

`chill()` is the explicit counterpart of [`bite()`](bite.md). `bite()`
replaces the compatible public functions of one module with `@snekit`
wrappers; `chill()` points those names back at the original Python
functions:

```python
import calculations
import speedysnek as snek

snek.bite(calculations)
calculations.sum_squares(100)

report = snek.chill(calculations)
```

Afterwards, `calculations.sum_squares` is the same function object it was
before `bite()`: plain Python, with no JIT in its calls.

`chill()` takes exactly one module. Anything else, including a wrapper or
a function, raises `TypeError`.

## What changes, and what does not

`chill()` rebinds names in the module's namespace, and nothing else:

- **Other references keep their wrapper.** Code that ran
  `from calculations import sum_squares` after `bite()` still holds the
  wrapper, and that wrapper keeps working.
- **Wrappers are not reset.** A wrapper's call count, warmup progress and
  compiled code are left exactly as they were. `chill()` only removes the
  module's names for it.
- **Nothing runs.** `chill()` calls, compiles and warms up nothing, and
  never calls the functions it restores.
- **Running code is not interrupted.** A call already running through a
  wrapper finishes normally. `chill()` is not a cancellation mechanism.

## What gets chilled

`chill()` looks at the module's public names, in definition order, with
the same rules as `hunt()`, `bite()` and `peek()`. Each of the module's
own public functions gets one finding:

- **chilled**: a wrapper `bite()` installed for it was found, and every
  public name holding that wrapper now holds the original function again;
- **unchanged**: it is an ordinary function with no wrapper to undo, so
  it is left as it is. This includes functions `bite()` skipped and
  functions marked with `@nosnek`;
- **skipped**: it has a wrapper that `chill()` left in place, because
  `bite()` did not make it, or because the module changed while
  `chill()` was running (see [Safety](#safety)).

Only wrappers that `bite()` installed are undone. A wrapper from `@snekit`
in the source, or one you made by hand, is yours: it is reported as
skipped, "wrapped with @snekit, but not by snek.bite()", and left alone.

Private names, classes, callable objects and functions imported from
other modules are not reported and never change.

## Imported wrappers

A wrapper imported from another module is never unwrapped, even under a
public name:

```python
# report.py
from calculations import sum_squares   # a wrapper, after bite(calculations)
```

`chill(report)` leaves `report.sum_squares` alone and does not list it: it
wraps a function `report` does not define. `chill(calculations)` restores
`calculations.sum_squares`, but `report.sum_squares` still holds the
wrapper, like any other reference.

## Aliases

When several public names in the module hold the same wrapper, as
`bite()` leaves aliases, every one of them is restored to the same
original function:

```python
def sum_squares(n): ...
total_of_squares = sum_squares
```

After `bite()` and `chill()`, `sum_squares` and `total_of_squares` are
both the original function again. The report has one finding, under the
first name; `finding.names` lists all of them.

Private aliases, such as `_fast = calculations.sum_squares`, are left as
they are.

## Chilling twice

`chill()` is idempotent. Once a module's wrappers are gone, another
`chill()` changes nothing and reports its functions as unchanged. Biting
the module again wraps them anew, with fresh wrappers.

## `@nosnek`

[`@nosnek`](bite.md#opting-out-with-nosnek) affects future automatic
wrapping, not wrappers that already exist:

- A function `bite()` wrapped before it was marked is restored normally.
  Afterwards it stays marked, so a later `bite()` leaves it alone.
- A function that was marked before `bite()` was never wrapped: `chill()`
  reports it as unchanged, "marked with @nosnek, not wrapped".

## Safety

`chill()` updates a module all at once or not at all. It inspects every
name first and decides every restoration. Then it checks that every name
it is about to restore still holds the exact wrapper it inspected. If even
one has changed in the meantime, no name in the module is restored: that
function is reported as skipped because its binding changed, and every
other function that would have been restored is reported as skipped
because the module changed. Otherwise all names are restored together, in
one update of the module's namespace.

A wrapper is trusted only if it is a genuine SpeedySnek wrapper whose own
JIT state runs a function the module defines, read the same way `peek()`
reads it. A rewritten `__wrapped__` or `_jit_state` attribute is never
believed, and objects that only look like a wrapper are ignored.

Like `bite()`, `chill()` reads and writes the module's real namespace
directly, so custom module `__getattr__`, `__setattr__` or `__dict__`
hooks are never consulted, and it runs no user code. It loads neither the
frontend nor the compiler, and it does not load the native extension for
a module that has no wrappers.

## The report

`chill()` prints the report to `sys.stdout` and returns it. Pass
`quiet=True` to print nothing, or `stream=` to print to another text
file. `str(report)` and `report.render()` give the same text:

```text
🐍 Snek cooled down.

sum_squares    chilled    restored plain Python function
format_report  unchanged  not wrapped with @snekit

1 function chilled, 1 unchanged.
```

`ChillReport`:

- `findings`: a tuple of `ChillFinding`, in definition order (iterating
  the report yields them too);
- `chilled`, `skipped`, `unchanged`: the findings with that status;
- `chilled_count`, `skipped_count`, `unchanged_count`: how many of each.

`ChillFinding`:

- `name`: the function's first public name in the module;
- `names`: every public name in the module bound to the function or a
  wrapper of it, `name` first;
- `function`: the original Python function;
- `wrapper`: the wrapper that was found (for a chilled function, the one
  the module no longer holds); `None` when there was none;
- `status`: `"chilled"`, `"skipped"` or `"unchanged"` (`finding.chilled`
  is true for the first);
- `summary`: a short, human-readable explanation.
