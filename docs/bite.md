# `snek.bite()`: applying `@snekit` to a module

`bite()` is the explicit, opt-in step after [`hunt()`](hunt.md). `hunt()`
inspects; `bite()` replaces the compatible public functions of one module
with `@snekit` wrappers:

```python
import my_calculations
import speedysnek as snek

snek.hunt(my_calculations)            # inspect first
report = snek.bite(my_calculations)   # then apply @snekit
```

Afterwards, `my_calculations.sum_squares` is a `@snekit` wrapper, exactly
as if the function had been decorated in its source. It runs the original
function during warmup, tries to compile it once warmup is over, and falls
back to Python whenever native execution does not apply. `bite()` itself
calls, compiles and times nothing: biting means "try the existing JIT
after warmup", not "guaranteed faster".

`bite()` takes exactly one module. Anything else raises `TypeError`.

## What gets bitten

`bite()` picks candidates with the same rules as `hunt(module)`: the
public, ordinary Python functions the module defines itself, in
definition order. Imported functions, classes, methods, callable objects,
builtins and names starting with `_` are left out. Each candidate is
inspected exactly as `hunt()` inspects it:

- **bitten**: the frontend accepts it, so the module's names for it now
  hold one new `@snekit` wrapper;
- **skipped**: the frontend does not accept it, so it is left exactly as
  it was, with the same explanation `hunt()` gives;
- **already bitten**: the module already holds a `@snekit` wrapper of it,
  so no new wrapper is made.

## Only the module's names change

`bite()` rebinds names in the module's namespace. It does not modify
function objects, and it does not rewrite references held anywhere else:
code that ran `from my_calculations import sum_squares` before `bite()`
keeps the original function. Bite a module before other code imports
names from it, or access its functions through the module.

Functions imported into the module from elsewhere are never wrapped, in
either module.

## Aliases

When one function is bound to several public names in the module, every
one of those names is rebound to the same wrapper, so no alias silently
bypasses the JIT:

```python
def sum_squares(n): ...
total_of_squares = sum_squares   # also becomes the wrapper
```

The report has one finding for the function, under its first name;
`finding.names` lists all of them.

## Biting twice

`bite()` is idempotent. A function that already has a `@snekit` wrapper in
the module, whether from an earlier `bite()` or a `@snekit` in the source,
never gets a second one: the wrapper object, its call count and its JIT
state are left untouched, and the function is reported as
`already_bitten`.

If a public name still holds the plain function next to such a wrapper,
that name is pointed at the existing wrapper too, so it does not bypass
the JIT:

```python
def triple(n):
    return n * 3

fast_triple = snekit(triple)   # unchanged
# after bite(): module.triple is module.fast_triple
```

This also applies when the wrapper is under a private name, such as
`_fast_triple = snekit(triple)`: the private name is left as it is, and
public names of `triple` get that wrapper.

## Safety

`bite()` updates a module all at once or not at all. It inspects every
candidate and builds every wrapper first; if building one fails, the
error propagates and the module is left exactly as it was. Then it checks
that every name it is about to replace still holds the original function.
If even one has changed in the meantime, no name in the module is
replaced: that function is reported as skipped because its binding
changed, and every other function that would have been updated is
reported as skipped because the module changed. Otherwise all names are
replaced together, in one update of the module's namespace.

Like `hunt()`, `bite()` reads and writes the module's real namespace
directly, so custom module `__getattr__`, `__setattr__` or `__dict__`
hooks are never consulted, and it runs no user code. The native extension
is loaded only when there is at least one function to wrap.

## The report

`bite()` prints the report to `sys.stdout` and returns it. Pass
`quiet=True` to print nothing, or `stream=` to print to another text
file. `str(report)` and `report.render()` give the same text:

```text
🐍 Snek took a bite.

sum_squares      bitten          wrapped with @snekit
format_report    skipped         f-strings are not supported (line 9)
old_calculation  already bitten  already wrapped with @snekit

1 function bitten, 1 skipped, 1 already bitten.
```

`BiteReport`:

- `findings`: a tuple of `BiteFinding`, in definition order (iterating
  the report yields them too);
- `bitten`, `skipped`, `already_bitten`: the findings with that status;
- `bitten_count`, `skipped_count`, `already_bitten_count`: how many of
  each.

`BiteFinding`:

- `name`: the function's first public name in the module;
- `names`: every public name in the module bound to the function or its
  wrapper, `name` first;
- `function`: the original Python function;
- `wrapper`: the `@snekit` wrapper the module's names now hold (the new
  one, or the existing one when already bitten); `None` when skipped;
- `status`: `"bitten"`, `"skipped"` or `"already_bitten"` (`finding.bitten`
  is true for the first);
- `summary`: a short, human-readable explanation;
- `hunt`: the `HuntFinding` for the function, with the frontend's
  `reason`, `detail` and `lineno`; `None` when already bitten.
