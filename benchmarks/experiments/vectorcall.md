# Quick experiment: vectorcall for `NativeWrapper`

**Conclusion: not adopted. Nothing in the runtime changes.**

A vectorcall entry point would pay off. In a prototype, a warmed `a + b`
call creates no argument tuple and executes **40% fewer instructions**
(1084 → 655). It took 0.46× the time.

PyO3 0.29 cannot give a `#[pyclass]` vectorcall, though. The only working
route is a CPython ABI hack, which this experiment was told not to ship:

- patch the type object after PyO3 has created it;
- compute an offset into PyO3's private object layout;
- replace PyO3's call trampoline with hand-written `unsafe extern "C"`.

A version limited to the common path (positional calls the hot slot can
take, with `tp_call` for the rest) also fails the regression limit on its
own. Every call it declines costs 9–12% more instructions than on `main`.

The branch keeps what is useful without the hack:

- tests that pin call-protocol independence;
- the measurement script;
- the two prototypes, as patches, for whoever revisits this.

This is a short, interleaved A/B measurement plus deterministic
instruction counts, not a pyperf baseline. It does not replace or update
[`../RESULTS.md`](../RESULTS.md).

## Question

`NativeWrapper` implements only `tp_call`. PyO3 generates it from
`__call__(*args, **kwargs)`, so a call site such as `add(3, 4)` makes
CPython build an argument tuple, and with keywords a dict, for every
call.

[`hot_slot.md`](hot_slot.md) ended on that point: what remains of a
warmed call is the tuple, PyO3's trampoline and argument handling, the
counter, validation and the result `int`.

Would vectorcall avoid the tuple and lower the cost of a warmed native
hit, while keeping every guarantee:

- `call_count`;
- methods (`__get__`);
- keywords, defaults and their `TypeError`s;
- fallback to the original function;
- `__code__` invalidation;
- the hot slot, parking and re-probing, and the range guard;
- no Python frame on a native hit?

## What PyO3 and CPython offer

**CPython.** A class opts into vectorcall with two things:

- the `Py_TPFLAGS_HAVE_VECTORCALL` flag;
- `tp_vectorcall_offset`, the offset of a `vectorcallfunc` pointer
  inside each instance.

For a heap type, the documented way to set both is at creation: a
`__vectorcalloffset__` entry in `Py_tp_members` and the flag in the
`PyType_Spec`. CPython 3.12+ keeps the two consistent, and clears the
flag if `__call__` is reassigned. Nothing documents setting either on a
type that is already ready.

**PyO3 0.29.3** (the version this project uses):

- `__call__` always becomes `tp_call`.
- `create_type_object` builds the spec itself. It only ever adds
  `__dictoffset__` and `__weaklistoffset__` to `Py_tp_members`, and it
  computes the type flags itself.
- No attribute, trait or hook lets a `#[pyclass]` add a member or a flag.
- The pieces a hand-written entry point would need are crate-private:
  - the attach guard every PyO3 trampoline takes (`AttachGuard`,
    `pub(crate)`);
  - the conversion of a panic payload into `PanicException`.

**Upstream.**
[PyO3/pyo3#4252](https://github.com/PyO3/pyo3/issues/4252), "Support
`__vectorcalloffset__` for `__call__`", has been open since June 2024
with no linked pull request. The maintainers' sketch:

- 3.12+ only;
- vectorcall by default, except for a `__call__` that accepts
  `**kwargs`, which is exactly this wrapper's signature.

Even where it applied, PyO3 collects `*args` with
`TupleVarargs::handle_varargs_fastcall`, which is `PyTuple::new`. The
tuple would move from CPython into PyO3, not disappear. As planned
upstream, the feature would not help a variadic wrapper. That needs a
raw entry point that sees the argument array.

So there is no supported way to do this with PyO3 today, or on its
roadmap.

## What was measured

Three throwaway prototypes, all against `main` at `d27833d`. None is
part of this branch's build. All three pass the whole suite, including
this branch's new tests, except the two that pin that the wrapper has
no vectorcall entry.

All three turn vectorcall on the same way: `enable_vectorcall()` in
module init.

- It makes one probe instance and takes the address of a new
  `vectorcall` field in it, minus the object's address, as the offset.
- It writes `tp_vectorcall_offset` and `Py_TPFLAGS_HAVE_VECTORCALL` into
  PyO3's `PyTypeObject`, then calls `PyType_Modified`.

- **hot-only** ([`vectorcall_hot_only.patch`](vectorcall_hot_only.patch)):
  what the brief asked for.
  - The entry point handles a positional call that the live slot can
    take (current `__code__` and guard, compiled arity). It counts the
    call, runs native code over the argument array without a tuple, and
    on a guard miss calls the original function with `PyObject_Vectorcall`.
  - Any other call rebuilds `tp_call`'s tuple and dict, as CPython's
    private `_PyObject_MakeTpCall` would, and runs the unchanged PyO3
    `__call__`.
- **full**: [`vectorcall_full.patch`](vectorcall_full.patch) without its
  `Python::attach` hunk.
  - `__call__`'s body becomes `dispatch`, generic over a small
    `CallArgs` trait with two implementations: a `tp_call` tuple and dict,
    or a vectorcall array and `kwnames`. One copy of the logic serves both
    entry points.
  - Fallbacks forward the caller's arguments, keyword names included, to
    the original function with `PyObject_Vectorcall`. No path builds a
    tuple or a dict, and `__call__` stays only for
    `type(w).__call__(w, ...)`.
  - The test doubles' `fast_call` still gets a tuple, built on demand.
- **full + attach** ([`vectorcall_full.patch`](vectorcall_full.patch)):
  as full, but the entry point runs inside the public `Python::attach`
  instead of `Python::assume_attached` (see [Why not](#why-not)).

Also measured: `main` built with PyO3's documented
`--cfg pyo3_disable_reference_pool`, a cheaper change that needs no
vectorcall (see [Alternatives](#alternatives)).

### Cases

[`vectorcall.py`](vectorcall.py) verifies each case before measuring it:
result and type, status, and path (native, guard miss or Python).

| case | call | path | status |
|---|---|---|---|
| `add` | `f(3, 4)`, `a + b` | native | compiled |
| `ops_030`, `ops_100` | `f(a, b, c, d)`, 30 / 100 operations | native | compiled |
| `float_miss` | `f(3.5, 4.25); f(3, 4)`: a guard miss, then a native call that restarts the miss count, so the wrapper never parks | guard miss + native | compiled |
| `keyword` | `f(a=3, b=4)` | Python | parked (by its own misses) |
| `default` | `f(3)` for `def f(a, b=4)` | Python | parked |
| `disabled` | `f(3, 4)`, a rejected body | Python | disabled |
| `parked` | `f(3.5, 4.25)` after the floats parked it | Python | parked |
| `method` | `o.f(3, 4)`; `self` is never an `int` | Python | parked |
| `star_args` | `f(*xs)` with a prebuilt tuple | native | compiled |

`keyword` and `default` park after `MISS_LIMIT` calls, so their steady
state is the parked path. `float_miss` times two calls per iteration.

## Results

### Instructions, tuples and dicts

Retired instructions per call, from `valgrind --tool=callgrind` with
`PYTHONHASHSEED=0`: the difference between runs of 200,000 and 100,000
calls, divided by 100,000. Repeated runs give identical counts.

| case | `main` | hot-only | full | full + attach | no ref pool |
|---|---|---|---|---|---|
| `add` | 1084 | 644 (0.594) | 655 (0.604) | 877 (0.809) | 1041 (0.960) |
| `ops_030` | 1591 | 1087 (0.683) | 1096 (0.689) | 1318 (0.828) | 1548 (0.973) |
| `ops_100` | 1940 | 1436 (0.740) | 1445 (0.745) | 1667 (0.859) | 1897 (0.978) |
| `float_miss` (2 calls) | 2553 | 1655 (0.648) | 1691 (0.662) | 2135 (0.836) | 2467 (0.966) |
| `keyword` | 3309 | 3350 (**1.012**) | 1223 (0.370) | 1446 (0.437) | 3266 (0.987) |
| `default` | 1436 | 1603 (**1.116**) | 1051 (0.732) | 1273 (0.886) | 1393 (0.970) |
| `disabled` | 1883 | 2047 (**1.087**) | 1476 (0.784) | 1698 (0.902) | 1840 (0.977) |
| `parked` | 1553 | 1718 (**1.106**) | 1139 (0.733) | 1360 (0.876) | 1510 (0.972) |
| `method` | 1458 | 1593 (**1.093**) | 1006 (0.690) | 1227 (0.842) | 1415 (0.971) |
| `star_args` | 847 | 554 (0.654) | 565 (0.667) | 787 (0.929) | 804 (0.949) |

The script also counts, per call, the calls to `tuple_dealloc` and
`dict_dealloc`: every argument tuple and keyword dict that was built
and freed.

| case | `main` | hot-only | full, full + attach |
|---|---|---|---|
| `add`, `ops_030`, `ops_100` | 1 tuple | **0** | **0** |
| `float_miss` (2 calls) | 2 tuples | **0** | **0** |
| `keyword` | 1 tuple, 2 dicts | 1 tuple, 2 dicts | **0** |
| `default`, `disabled`, `parked`, `method` | 1 tuple | 1 tuple | **0** |
| `star_args` | 0 (the caller's tuple is reused) | 0 | 0 |

So yes: with vectorcall the tuple is really gone. On `main`, `keyword`
frees CPython's kwargs dict and PyO3's own `**kwargs` dict. The full
prototype passes the caller's keyword names straight through.

### Timing

`python benchmarks/experiments/vectorcall.py --processes 5 --cpu 3
--build …`: 5 processes per build, alternating builds, each interleaving
every cell for 15 rounds of 200,000 calls. Median of process medians in
ns per call, min–max over processes.

| case | python | `main` | hot-only | full | full + attach |
|---|---|---|---|---|---|
| `add` | 21 | 64 (55–65) | 28 (0.43) | 30 (0.46) | 50 (0.78) |
| `ops_030` | 227 | 108 (96–113) | 56 (0.52) | 57 (0.53) | 89 (0.82) |
| `ops_100` | 680 | 145 (116–149) | 83 (0.57) | 84 (0.58) | 107 (0.74) |
| `float_miss` (2 calls) | 78 | 171 (148–173) | 85 (0.50) | 89 (0.52) | 133 (0.78) |
| `keyword` | 40 | 215 (177–219) | 212 (0.99) | 65 (0.30) | 87 (0.41) |
| `default` | 33 | 77 (63–79) | 87 (**1.13**) | 49 (0.64) | 70 (0.91) |
| `disabled` | 35 | 108 (89–113) | 118 (**1.09**) | 74 (0.68) | 93 (0.86) |
| `parked` | 35 | 85 (72–87) | 98 (**1.15**) | 56 (0.66) | 77 (0.91) |
| `method` | 26 | 77 (67–83) | 94 (**1.22**) | 45 (0.59) | 69 (0.90) |
| `star_args` | 26 | 49 (44–52) | 26 (0.53) | 29 (0.58) | 50 (1.02) |

A second run timed `main` against a separate build of this branch,
which has the same Rust code. The branch came out 0.92–1.19× `main`,
row by row. That is this shared VM's noise floor, so the instruction
counts are the evidence for differences under about 20%. In the same
run, full took 0.31–0.68× `main` and the no-reference-pool build
0.81–1.05×.

### What the numbers say

- **Where the cost was.** On `main`, a warmed `a + b` call spends about
  430 of its 1084 instructions on `tp_call`:
  - CPython builds and frees the tuple: about 150 of the 232 in
    `CALL_NON_PY_GENERAL`, plus 62 in `tuple_dealloc`;
  - PyO3 re-slices it (`PyTuple_GetSlice`, `extract_arguments_tuple_dict`:
    about 110);
  - PyO3's trampoline (about 130, with the reference pool's mutex and TLS
    lookups).

  The prototype removes all of it. What remains in the full prototype is
  the entry point itself (149, with `try_call` inlined), the argument
  conversion (`PyLong_AsLongLongAndOverflow`, 50), the result `int`,
  and CPython's generic `CALL_NON_PY_GENERAL`, which does not specialize
  calls to a custom vectorcall class.
- **Short functions gain the most:** the saving is a fixed 430–500
  instructions, 35–60 ns. That is 40% of `a + b`, 31% of `ops_030` and
  26% of `ops_100`. It moves the break-even point further down.
- **Hot-only regresses everything it declines.** Once the type has
  vectorcall, CPython never builds the tuple for it any more. A call the
  fast path does not take has to rebuild the tuple and dict with public
  API (`PyTuple_New`, `PyDict_SetItem`, `Py_EnterRecursiveCall`, a
  non-inlined `_Py_Dealloc`). That costs about 165 instructions more than
  CPython's inlined private `_PyObject_MakeTpCall`.

  `default`, `disabled`, `parked` and `method` then execute 8.7–11.6%
  more instructions and took 1.09–1.22× the time. So "vectorcall for the
  common path only" fails the >5% regression limit by itself. Every
  path has to go through vectorcall.
- **The full version improves every path:**
  - 0.60–0.78× the instructions;
  - `keyword` 0.37×, because no dict is built at all;
  - `star_args` too, because `PyObject_Call` on a vectorcall type unpacks
    the tuple into the array.
- **Paying for safety with public API costs half the gain** (full +
  attach). `Python::attach`, called with PyO3's attach count at zero, as
  it is whenever CPython calls in directly, goes through
  `PyGILState_Ensure`/`Release`, a `pthread_getspecific` and the
  reference pool's mutex: about 220 instructions per call. PyO3's own
  trampolines avoid this with the crate-private `AttachGuard::assume`.

## Why not

The full prototype is fast and passes every test. It is still not
mergeable:

1. **It mutates a ready type object.** It writes `tp_flags` and
   `tp_vectorcall_offset` into the `PyTypeObject` that PyO3 made with
   `PyType_FromSpec`. CPython documents no way to do that. It also needs
   the full, non-limited `PyTypeObject` layout. That is what the brief
   called a CPython ABI hack.
2. **It depends on PyO3's private layout.** It computes the offset of a
   Rust field inside `PyClassObject<NativeWrapper>` from a probe
   instance. That is correct only as long as every instance has the same
   layout, which PyO3 neither documents nor promises. A PyO3 upgrade
   could break it silently.
3. **It replaces PyO3's trampoline.** The entry point is an
   `unsafe extern "C"` function:
   - With `Python::assume_attached`, PyO3's thread-local attach count
     stays at zero during the call. Every `Py<T>` dropped inside the call
     (for example the old `count_base` in `carry`) is then treated as
     dropped without the GIL and queued in PyO3's reference pool instead
     of released. With PyO3's `pyo3_disable_reference_pool` it would
     abort the process instead.
   - The public fix, `Python::attach`, costs half the gain.
   - PyO3's conversion of a panic payload is crate-private, so a panic
     loses its message.
4. **It adds a second entry point.** `tp_call` stays reachable, through
   `type(w).__call__` and through PyO3's machinery. Every guarantee then
   has two implementations to keep in step. The full prototype shares
   the logic through `dispatch`, but the risk is why this branch adds
   [`test_call_protocols.py`](../../python/tests/test_call_protocols.py).

## Alternatives

In order of preference:

1. **Ask PyO3 for a raw vectorcall hook** and adopt the full design when
   it lands. Concretely: an opt-in way for a `#[pyclass]` to receive the
   argument array and `kwnames` (say, a `__vectorcall__` taking
   `&[Borrowed<PyAny>]` and the keyword names). PyO3 would set
   `__vectorcalloffset__` at type creation, keep `tp_call` as
   `PyVectorcall_Call`, and wrap the call in its own `AttachGuard::assume`
   and panic handling.

   The numbers above, about 0.6× the instructions for every path, are
   the case for it on #4252. The tests added here are ready for it.
2. **Leave `#[pyclass]` for this one type**, and create it with
   `PyType_FromSpec`, declaring `__vectorcalloffset__` and the flag
   through the documented API. That avoids points 1 and 2 above, not 3:
   - it would hand-write `NativeWrapper`'s getset, `__dict__`, weakref,
     GC, dealloc and `__get__` glue in `unsafe` code;
   - it would still need its own trampoline and the attach count fix.

   Not worth it for 35–60 ns per call at this stage.
3. **A small, safe gain available now, independent of vectorcall:**
   `--cfg pyo3_disable_reference_pool`.
   - PyO3's reference pool is created while the module is imported (by
     `intern!` in `add_function`). From then on, every PyO3 trampoline
     locks an uncontended mutex to check it. The flag removes the pool:
     43 fewer instructions on every path (0.95–0.99×), and the suite
     passes.
   - The cost is that a `Py<T>` dropped without the GIL then aborts the
     process instead of being deferred. It deserves its own audit and
     pull request, so it is not changed here.

## What this branch adds

- [`vectorcall.py`](vectorcall.py), the measurement:
  - interleaved timing of any number of builds (`--build NAME=DIR`);
  - `--count`, which runs every case under callgrind and reports
    instructions, tuples and dicts per call;
  - `--calls` for a single case.

  [`../tests/test_vectorcall_experiment.py`](../tests/test_vectorcall_experiment.py)
  runs its verification and its callgrind parser without timing or
  valgrind.
- [`test_call_protocols.py`](../../python/tests/test_call_protocols.py).
  Twin wrappers make the same calls, one through `type(w).__call__`
  (always `tp_call`) and the other through:
  - a call site (`CALL`/`CALL_KW`, vectorcall when the type has it);
  - `f(*args, **kwargs)`;
  - `functools.partial`;
  - a bound method.

  Every call must agree on:
  - the result or exception;
  - the original function's frames;
  - native attempts and misses;
  - `call_count`;
  - status and reasons, compile attempts and warmup;
  - the slot's live and parked flags and both counters.

  The sequences cover compiled calls, guard misses, invalid arities,
  keywords and defaults, `__code__` replacement (same and other
  signature), parking and re-probing, disabled wrappers, and methods
  bound to an `int` (which run natively) and to an object. Deep
  recursion must raise `RecursionError` through every protocol. A native
  hit must enter no Python frame.

  One test pins that `NativeWrapper` has no vectorcall entry, so that
  adding one is a deliberate decision.
- Both prototypes as patches against `d27833d`.

### Checking the tests catch a divergence

The new tests pass on this branch and on all three prototypes. Seven
bugs were injected into the prototypes, one at a time:

| injected bug | prototype | caught |
|---|---|---|
| no `__code__`/guard check on the vectorcall path | hot-only | 3 tests |
| vectorcall path does not count `call_count` | hot-only | 12 tests |
| a guard miss is not counted towards parking | hot-only | 12 tests |
| no arity check before the native attempt | hot-only | 9 tests |
| keyword names ignored when deciding on a native attempt | full | 9 tests |
| keyword names dropped when falling back | full | 21 tests |
| `PY_VECTORCALL_ARGUMENTS_OFFSET` not forwarded | full | 0: harmless |

The last one is an equivalent mutant: forwarding the flag is allowed,
not required. The "keyword names ignored" bug first went undetected.
`MIXED` now includes calls with every positional argument and a keyword,
such as `f(1, 2, c=3)`.

## Limitations

- One machine, one CPython version (3.14.6, the tail-calling
  interpreter), and a noisy shared VM. Instruction counts are not cycles:
  they include no cache or branch effects.
- The prototypes are throwaway code, written to measure. Panic
  handling, the reference-pool issue in point 3 of [Why not](#why-not),
  and free-threaded CPython were not made production-ready.
- CPython does not specialize `CALL` for a custom vectorcall type; it
  stays on `CALL_NON_PY_GENERAL`. A future CPython that did would change
  these numbers in vectorcall's favour.

## Environment and commands

- Intel Xeon @ 2.10 GHz, 4 vCPU cloud VM, Linux 6.18.
- CPython 3.14.6, Rust 1.97.0, PyO3 0.29.3.
- Release builds (`DEBUG_BUILD == False`).

```sh
# One worktree and release build per variant, each with its own target
# directory, imported through PYTHONPATH.
git worktree add ../main d27833d
git worktree add ../full d27833d
git -C ../full apply "$PWD/benchmarks/experiments/vectorcall_full.patch"
for tree in main full; do
    (cd ../$tree && maturin build --release -o dist &&
     unzip -o -j dist/speedysnek-*.whl 'speedysnek/_native*.so' \
         -d python/speedysnek/)
done

python benchmarks/experiments/vectorcall.py --processes 5 --cpu 3 \
    --build main=../main/python --build full=../full/python
python benchmarks/experiments/vectorcall.py --count \
    --build main=../main/python --build full=../full/python

# The reference-pool build of main.
(cd ../main && RUSTFLAGS="--cfg pyo3_disable_reference_pool" \
    maturin build --release -o dist-norefpool)
```

Give each build its own `CARGO_TARGET_DIR`. Two checkouts of the same
crate sharing one target directory can leave one build's library in the
other's wheel.
