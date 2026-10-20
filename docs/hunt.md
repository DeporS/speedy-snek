# `snek.hunt()`: finding functions the compiler understands

`hunt()` is SpeedySnek's discovery API. It answers one question: which of
these functions can the current compiler frontend translate? It is
discovery only. It never calls, decorates, compiles, benchmarks or
replaces a function, and it does not load the native extension or the
compiler.

```python
import speedysnek as snek

report = snek.hunt(my_module)                 # a module's own functions
report = snek.hunt(function_a, function_b)    # explicit functions
report = snek.hunt([function_a, function_b])  # one iterable of functions
```

**Ready** means only that the frontend accepts the function's bytecode.
It does not promise a speedup: small functions can run slower through
`@snekit`, and the backend can still refuse a ready function (for
example, one with more local variables than it supports).

## What gets inspected

For a module, `hunt()` reads the module's own namespace, in definition
order, and keeps only ordinary Python functions defined by that module
(`types.FunctionType` whose `__module__` is the module's name). It leaves
out imported functions, classes, methods, callable objects, builtins,
functions already wrapped by `@snekit` and names starting with `_`. A
function bound to several names is reported once, under the first.

Explicit functions are reported in the order given, private and imported
ones included. A callable that is not a Python function, such as `len`, a
class, a bound method or a `@snekit` wrapper, is reported as skipped with
an explanation. Anything else raises `TypeError`: a value that is not
callable, a string (pass the module object, not its name), or a module
mixed with other arguments.

`hunt()` runs no user code. It reads modules, functions and classes
through the built-in types' own descriptors, so custom `__getattr__`,
properties and `__class__` overrides are never consulted. The only
exceptions are iterating an iterable you pass in and writing to `stream`.

## The report

`hunt()` prints the report to `sys.stdout` and returns it. Pass
`quiet=True` to print nothing, or `stream=` to print to another text
file. `str(report)` and `report.render()` give the same text.

`HuntReport`:

- `findings`: a tuple of `HuntFinding`, in report order (iterating the
  report yields them too);
- `ready`, `skipped`: the findings with that status;
- `ready_count`, `skipped_count`: how many of each.

`HuntFinding`:

- `name`: the function's name (its name in the module, for a module);
- `function`: the inspected object itself;
- `status`: `"ready"` or `"skipped"` (`finding.ready` is the same as a
  boolean);
- `summary`: a short, human-readable explanation, such as
  `"integer arithmetic and one range loop"` or
  `"f-strings are not supported"`;
- `reason`: for a skipped function, the frontend's stable rejection code,
  such as `"unsupported_opcode"`, or `"not_a_function"`; `None` when ready;
- `detail`: for a skipped function, the frontend's exact message, such as
  `"unsupported opcode FORMAT_SIMPLE at offset 4"`; `None` when ready;
- `lineno`: the source line of the rejected instruction, where known.

The frontend reports the first unsupported construct it meets, so fixing
one reason can reveal the next.
