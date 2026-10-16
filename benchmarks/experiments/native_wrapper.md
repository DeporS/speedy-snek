# Quick experiment: a native (Rust) `@snekit` wrapper

**Conclusion: adopted.** Moving the wrapper's per-call logic into a PyO3
callable makes a warmed `@snekit` call about **90–100 ns cheaper** (0.58–0.66×
the Python wrapper's time) and a guard miss about **110 ns cheaper**. The
results repeated in all three processes. Omitted defaults, disabled wrappers
and method calls also got faster. Keyword calls stayed the same, and no
case got slower. `snekit` now returns a `speedysnek._native.NativeWrapper`
instead of a Python closure. The observable differences this causes are
listed [below](#observable-differences).

This is a short, interleaved A/B run (about 75 s of timing in total), not a
pyperf baseline. It does not replace or update [`../RESULTS.md`](../RESULTS.md).

## Question

The [previous experiment](wrapper_overhead.md) found that a warmed call
spends about 140 ns in the Python wrapper's own body, on top of the
fastest Python-level wrapper possible. Generating a wrapper for each
function's signature did not reduce that cost. Does running the same
per-call logic in Rust help, while keeping `@snekit`'s behaviour?

## What was built

**Calling convention (PyO3 0.29.3, checked in its source).** A Rust
`__call__` becomes a `tp_call` slot with the tuple-and-dict (`Varargs`)
convention, not vectorcall. A call from Python therefore goes through
CPython's `_PyObject_MakeTpCall`, which builds an argument tuple, and a
kwargs dict when keywords are passed. With `signature = (*args, **kwargs)`,
PyO3 passes the tuple through unchanged (`PyTuple_GetSlice(0, len)` on an
exact tuple returns the same object). For a keyword call it copies the
kwargs dict into a new one. No custom vectorcall or other new `unsafe` code
was written. The only `unsafe` in the change is the existing
`PyLong_AsLongLongAndOverflow` helper, reused unchanged.

**`NativeWrapper`** (`src/lib.rs`, about 250 lines) holds the original
function, its existing Python `_JitState`, and an atomic `call_count`. Its
`__call__` is the old Python wrapper's body, line for line:

1. Count the call.
2. Read `state.specialization`. While it is `None`, use `state.status` and
   `state.cold_call()`, as before.
3. Check `func.__code__ is spec.code`, and call `state.invalidate()` if
   not.
4. Skip native code for keyword calls and for calls of another arity.
5. Run the fast path of `spec.function`, the specialization's
   `I64Function`.
6. On `None`, call the original function once. On an unexpected
   `Exception`, call `state.native_failed()` and then fall back. Let a
   `BaseException` propagate.

Compilation, warmup, locking and every state transition stay in the
existing Python helpers. The state machine is not duplicated.

**Shared helper.** `I64Function.try_call`'s body moved into
`try_call_tuple`, which both `try_call` and the wrapper call. Argument
validation and the native calling convention exist once.
`_Specialization` gains `function` (the `I64Function`, via the new
`CompiledI64.function`). Any other value, used only by test doubles,
makes the wrapper call `spec.fast_call` instead.

**Function-like behaviour.**

- `__get__` binds like a function: `MethodType(wrapper, instance)` for
  instance access, the wrapper itself for class access, and the same
  `TypeError` for `__get__(None, None)`.
- `functools.update_wrapper` copies the metadata into the instance
  `__dict__`.
- `__annotate__` and `__annotations__` are a coupled pair with a
  function's exact semantics: lazy (PEP 649 annotations are not evaluated
  at decoration), cached, the same reset rules and the same errors.
- `__reduce__` returns the qualified name, so pickle and
  `copy`/`deepcopy` treat the wrapper by reference, like a function.
- `__repr__` names the original function.
- GC: `__traverse__` visits the original function, the state and the
  annotations. PyO3 visits and clears the instance `__dict__`.

**Executable memory.** The wrapper holds a strong `Bound` reference to
`spec.function` for the whole native call. Nothing in that call runs
Python code, so the machine code cannot be freed under it. The GIL is held
throughout, as before.

**Proving the path without a Python-visible call.** A call from Rust emits
no `sys.setprofile` event. `I64Function.try_calls` counts the fast path's
entries: a relaxed load and store, not a locked read-modify-write, because
callers hold the GIL. The test helpers and `workloads.call_path` now count
native calls with it and Python frames with `sys.setprofile`. A canary
`I64Function` that returns a constant, swapped into the specialization,
proves that the wrapper runs exactly `spec.function`'s machine code.

## Results

Each cell is the median of three process medians (min–max in brackets),
in ns per public call. Δ is `native_wrapper − python_wrapper`, measured in
the same process. `python` is the undecorated function with the same
statement.

| case | python | python_wrapper | native_wrapper | Δ | ratio |
|---|---|---|---|---|---|
| warmed `ops_001` | 33 | 238 (232–243) | 139 (137–141) | **−99** | 0.58–0.59 |
| warmed `ops_020` | 159 | 241 (237–250) | 153 (150–157) | **−88** | 0.61–0.66 |
| warmed `ops_050` | 366 | 255 (255–266) | 168 (166–170) | **−87** | 0.64–0.66 |
| guard miss (`float_args`) | 35 | 251 (248–262) | 140 (139–140) | **−111** | 0.53–0.57 |
| keyword call | 41 | 259 (252–269) | 255 (253–261) | −4 | 0.95–1.01 |
| omitted default | 33 | 195 (169–197) | 125 (123–126) | **−70** | 0.63–0.74 |
| disabled wrapper | 36 | 164 (156–167) | 100 (99–102) | **−64** | 0.59–0.64 |
| instance method (guard miss) | 35 | 262 (259–268) | 212 (206–223) | **−50** | 0.79–0.85 |

The ratio range is the per-process range. A fourth process on the final
integrated build gave the same picture (0.61, 0.61, 0.63, 0.52, 0.91,
0.63, 0.62, 0.82).

What the rows show:

- **Warmed calls: the wrapper's cost roughly halves.** Over the plain
  function, `ops_001` costs 106 ns instead of 205 ns. A warmed call now
  beats plain Python from about 20 operations (153 vs 159 ns at
  `ops_020`). The saving comes from removing the Python frame, the
  `call_count` int allocation, the `**kwargs` dict and the bytecode for
  the checks. What remains is the `tp_call` argument tuple, PyO3's
  trampoline, a few attribute reads (`specialization`, `code`,
  `__code__`, `function`) and the fast path itself.
- **Fallbacks.** A guard miss saves the most (−111 ns), because it is the
  warmed path plus one Python call. Omitted defaults and disabled
  wrappers save 64–70 ns.
- **Keyword calls: no gain, no regression.** CPython builds a kwargs dict
  for `tp_call`, and PyO3 copies it again. The two dicts cost about what
  the Python wrapper saved.
- **Instance methods: smaller gain.** A function gets CPython's
  method-call specialization. A non-function descriptor makes a
  `MethodType` on every attribute access, which then calls `tp_call`
  through another argument tuple. A bound call is still 50 ns cheaper
  than before.
- **The new counter costs nothing measurable.** Production `snekit` on
  `main` was compared, process by process, with the same Python wrapper
  on this branch, which also increments `try_calls`. `fast_call` measured
  50.0/52.2/49.9 ns on `main` and 50.5/51.5/56.4 ns here. The ranges
  overlap and the wrapper rows interleave, so the cost is below this
  run's resolution.

Every case was verified before timing (see `native_wrapper.py`):

- each result equals the undecorated function's, with the same type;
- each call takes the expected path: warmed calls make one native call
  and no Python call, guard misses one of each, and other fallbacks one
  Python call and no native call;
- for the native wrapper, the canary check passed.

Setup, warmup and compilation are outside the timed loops. The state of
every wrapper is checked again after timing.

## Correctness

- The existing suites pass. The test helpers count native calls with
  `try_calls` instead of profile events. Four tests describe something
  that changed on purpose and were updated in place:
  - lazy imports: decorating now loads `_native`;
  - the wrapper's `__dict__` contents;
  - the frontend's rejection reason for a wrapper;
  - the bound `fast_call` is no longer what the wrapper calls (a canary
    test replaces this one).
- The fault-injection doubles (`BrokenNative`, and two tests that replace
  `spec.fast_call`) now also set `function = None`, because the wrapper
  only calls `spec.fast_call` when there is no `I64Function`.
- `test_wrapper_experiment.py` now compares the previous experiment's
  Python prototype with the native wrapper, call by call. That includes
  replaced `__code__` with another signature, positional-only parameters,
  changed defaults and colliding parameter names.
- [`../../python/tests/test_native_wrapper.py`](../../python/tests/test_native_wrapper.py)
  adds 37 test cases:
  - methods, class access, `staticmethod`/`classmethod`/`property`/
    `cached_property`/`abstractmethod` combinations, and `snekit` over a
    `staticmethod`;
  - metadata, `inspect.signature`/`getsource`/`unwrap`, `get_type_hints`,
    `functools.wraps`/`lru_cache`/`partial`;
  - annotation semantics checked against a real function, step by step;
  - pickle and copy;
  - invalid calls: the original's `TypeError`, counted, no native call;
  - `call_count` exact under four threads;
  - defaults and `__code__` changes, tracebacks, exception context and
    `BaseException`s;
  - collection of the wrapper, the original and the `CompiledI64` that
    owns the machine code, with no cycle (refcounting alone) and with
    cycles through the wrapper's `__dict__`, the original's `__dict__`,
    the annotations cache, the JIT state and a closure cell.
- Six bugs were injected into the Rust wrapper and each was caught:
  - an uncounted call;
  - keyword calls reaching native code;
  - no `__code__` check;
  - no method binding;
  - `__traverse__` skipping the original function;
  - no arity check.

## Observable differences

The call contract is unchanged. Results, exceptions and their context,
argument forwarding, defaults, invalid-call `TypeError`s, `call_count`,
metadata, `__wrapped__`, `inspect.signature`, method binding, state,
one-time compilation, invalidation and fallback all behave as before.
Replacing a function with a callable object does change the following,
which are now documented and pinned by tests:

- **It is not a function.**
  - `inspect.isfunction(wrapper)` and `isinstance(wrapper, FunctionType)`
    are false.
  - It has no `__code__`, `__defaults__`, `__kwdefaults__`, `__globals__`,
    `__closure__` or `__builtins__`. Before, these described the closure,
    not the original.
  - `inspect.isroutine(inspect.unwrap(wrapper))`, doctest discovery,
    pydoc and `typing.get_type_hints` still work.
- **`repr`** is `<snekit wrapper of <function add at 0x…>>` instead of
  `<function add at 0x…>`.
- **`vars(wrapper)`** also contains `__module__`, `__name__`,
  `__qualname__`, `__doc__` and `__type_params__`, which a function keeps
  in slots. `call_count` is a property, not a `__dict__` entry, and is
  the only public attribute.
- **`call_count`**:
  - Increments are atomic, so concurrent calls are no longer
    undercounted.
  - It accepts only an `int` in the signed 64-bit range (`TypeError` or
    `OverflowError` otherwise). Before, any value was stored and the next
    call failed for a non-number.
  - It cannot be deleted. Before, deleting it made every later call raise
    `AttributeError`.
- **No wrapper frame.** Tracebacks through `@snekit` no longer contain a
  `wrapper` frame, and `sys.setprofile`/`cProfile` no longer see one.
- **Import timing.** Decorating loads the extension module `_native` once
  per process, about 0.5–0.75 ms here; `import speedysnek` stays at
  4.7–6.3 ms. `import speedysnek` still loads nothing private, and a
  rejected function still never loads the compiler.
- **Frontend.** Analyzing a wrapper, as in `snekit(snekit(f))`, still
  rejects it, now as `NOT_A_FUNCTION` instead of `CLOSURE`.

## Limitations

- **PyO3's supported `__call__` is not vectorcall.** The argument tuple, and
  for keyword calls two dicts, are the remaining avoidable allocations.
  Removing them would need the type to implement vectorcall itself, which
  is custom `unsafe` machinery and was out of scope.
- **Methods still pay for a `MethodType` per bound call.** PyO3 does not
  expose `Py_TPFLAGS_METHOD_DESCRIPTOR`.
- **Error paths.** The native path's own unexpected-error handling shares
  its code with the `fast_call` path that the fault-injection tests
  exercise. The direct helper's errors (an unexpected native status, or a
  pending CPython error) cannot be triggered from Python.
- **Measurement scope.**
  - One machine, one CPython version, a short untuned run, and only the
    cases above. Differences under about 10 ns are not resolved.
  - Linux was verified locally. Windows and macOS are left to CI.
  - Free-threaded CPython is not supported or claimed. `try_calls` is
    exact only because callers hold the GIL.

## Environment and commands

- Intel Xeon @ 2.10 GHz, 4 vCPU cloud VM, Linux 6.18; runs pinned with
  `taskset -c 3`. No `pyperf system tune`: absolute numbers are noisy, so
  compare columns within a row.
- CPython 3.14.6, Rust 1.97.0, PyO3 0.29.3, release build
  (`maturin develop --release`, `DEBUG_BUILD == False`), base commit
  `9cefa1f`.

```sh
maturin develop --release
for i in 1 2 3; do
    taskset -c 3 python benchmarks/experiments/native_wrapper.py --rounds 15 --loops 20000
done
```

Each process interleaves every (variant, case) cell for 15 rounds of 200,000
calls (10 unrolled calls per loop iteration) and prints medians.
`python_wrapper` in the script is a verbatim copy of the closure `snekit`
returned before. [`wrapper_overhead.py`](wrapper_overhead.py) now uses it as
its `production` variant, so that experiment still measures what its
report describes.
