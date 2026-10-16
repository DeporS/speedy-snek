# Quick experiment: a signature-specific `@snekit` wrapper

**Conclusion: not adopted.** On the warmed path, a wrapper generated for the
function's positional signature costs the same as the production
`wrapper(*args, **kwargs)` or slightly more. The only signature-specific
form that is faster breaks the `@snekit` contract, and even that form saves
just ~45 ns. The production wrapper is unchanged.

This is a short, interleaved A/B run (about 75 s of timing in total), not a
pyperf baseline. It does not replace or update [`../RESULTS.md`](../RESULTS.md).

## Question

A warmed `@snekit` call is the Python wrapper around one native
`I64Function.try_call(args)`. Earlier measurements put roughly 340–430 ns per
call on that wrapper. Could a wrapper generated for the function's positional
signature reduce that cost without changing behaviour?

## What was tried

The prototype is `positional` in [`wrapper_overhead.py`](wrapper_overhead.py).
It applies to plain functions with `1 <= co_argcount <= MAX_ARGS` and uses
production `snekit` for everything else. For `N` positional parameters it
generates:

```python
def wrapper(_p0=_missing, ..., _pN=_missing, /, *args, **kwargs):
    wrapper.call_count += 1
    if _pN is not _missing and not args and not kwargs:
        ...  # production logic, with fast_call((_p0, ..., _pN))
    ...      # otherwise rebuild args and run the production logic unchanged
```

The safe form is constrained by the contract:

- **Every call must bind, so the wrapper keeps `*args, **kwargs`.** Without
  them, an invalid call or a keyword call would fail before `call_count` is
  incremented. It would also raise a different `TypeError`, and a
  replaced `__code__` with another signature could not be called through
  the wrapper. CPython creates the `**kwargs` dict on every call anyway, so
  the generated form avoids no allocation. It swaps the `*args` tuple for a
  `BUILD_TUPLE` of the locals.
- **The parameters are positional-only placeholders with private names
  (`_p0`…).** Keywords therefore always land in `kwargs` and reach the
  original function. Positional-only parameters, omitted defaults and later
  `__defaults__` changes all behave as before, and no user name can collide
  with a helper.
- **The generated source contains only fixed identifiers and integers.** It
  never includes a user name or a `repr()`. The end of the positional
  arguments is found by identity checks only, so no user `__eq__` runs.
- **Known gap:** the sentinel is reachable as `wrapper.__defaults__[i]`.
  Passing it explicitly as a trailing argument is indistinguishable from
  omitting that argument. Accepting this hole would itself be a reason not to
  ship the prototype.

[`../tests/test_wrapper_experiment.py`](../tests/test_wrapper_experiment.py)
compares the prototype call by call with production: results, exceptions and
their context, `call_count`, state, positional-only and keyword-only
parameters, defaults, `__defaults__` changes, `__code__` replaced (before and
after compilation, including with another signature), colliding parameter
names, and method binding. The existing `@snekit` suites (`test_jit.py`,
`test_jit_compile.py`) also pass with the prototype swapped in. The only two
failures check the decorator's own `__name__` and the real module's
doctests. Four injected bugs were each caught by the new tests: keywords
reaching native code, an uncounted slow path, a wrong positional prefix,
and a skipped `__code__` check.

## Results

Each cell is the median of three process medians (min–max in brackets), in ns
per public call. Δ is the change against `production` in the same process;
negative is faster. `positional` is the candidate. The last four columns are
**ablations**: they break the contract and only attribute the cost (see the
script's docstring).

| case | python | production | positional | exact_signature | no_arity_check | dict_counter | floor |
|---|---|---|---|---|---|---|---|
| warmed `ops_001` | 37 | 274 (268–286) | 280 (276–283) **Δ+8** | 228 Δ−46 | 249 Δ−19 | 252 Δ−22 | 132 Δ−142 |
| warmed `ops_020` | 184 | 288 (285–295) | 297 (291–298) **Δ+10** | 240 Δ−48 | 268 Δ−17 | 271 Δ−23 | 142 Δ−144 |
| warmed `ops_050` | 428 | 314 (311–315) | 328 (321–335) **Δ+14** | 268 Δ−47 | 290 Δ−24 | 295 Δ−19 | 164 Δ−150 |
| guard miss (`float_args`) | 41 | 304 (304–305) | 280 (280–282) **Δ−24** | | | | |
| keyword call | 48 | 309 (308–319) | 388 (385–404) **Δ+80** | | | | |
| omitted default | 38 | 232 (229–232) | 289 (288–326) **Δ+57** | | | | |
| disabled wrapper | 41 | 195 (194–199) | 195 (193–196) **Δ−2** | | | | |

What the columns show:

- **`positional`: no warmed-path gain.** It is 8–14 ns slower, which is within
  or near the run-to-run spread. It is 57–80 ns slower for keyword and
  omitted-default calls, which must first rebuild `args`. It is 24 ns faster
  on a guard miss. That gain does not come from the signature: the
  prototype calls `func(_p0, _p1)` instead of `func(*args, **kwargs)`.
- **`exact_signature`: the ceiling for signature-specific wrappers.** A
  wrapper with exactly `N` positional parameters, no `*args`/`**kwargs`, no
  defaults and no keyword or arity checks saves about 46 ns, or 15–17% of the
  wrapper. Every safe design gives up most of that.
- **Where the cost really is.** Relative to `floor`, a wrapper that only runs
  `spec.fast_call(args)`, production pays about 140 ns for:
  - `wrapper.call_count += 1`: ~55 ns. A new `int` object on every call.
    Counting through `wrapper.__dict__` saves only ~20 ns of that.
  - the `__code__` identity check: ~25–30 ns.
  - `not kwargs and len(args) == spec.arg_count`: ~20 ns.
  - the `try`/`None` check and the cold-state branch: the rest.

  The first two are required by the contract, and none of them depend on the
  signature.

Every wrapper was built and verified by the suite's own helpers
(`workloads.prepare_layers`, `workloads.fallback_cases`) before timing.
`sys.setprofile` proves that a warmed call makes exactly one native
`try_call` and enters neither the original function nor
`CompiledI64.__call__`. Each guard miss makes one native call and then runs
Python once. Setup, warmup and compilation are outside the timed loops.

## Possible follow-ups (not done here)

These are measured above, but they change behaviour in edge cases or trade
one path for another, so they need a decision of their own:

- Drop the Python arity check and let `try_call`'s Rust check reject wrong
  arities: about −20 ns warmed. An omitted-default call would then make one
  native call before falling back.
- Count through a cached `wrapper.__dict__`: about −20 ns warmed. It
  diverges if a user replaces `wrapper.__dict__` or deletes `call_count`.
- Call `func(*args)` instead of `func(*args, **kwargs)` when there are no
  keywords: about −10 to −25 ns on fallback paths only.

Reducing the remaining ~130 ns floor needs native work (for example a Rust
callable or vectorcall wrapper), which is outside this experiment.

## Environment and commands

- Intel Xeon @ 2.10 GHz, 4 vCPU cloud VM, Linux 6.18; runs pinned with
  `taskset -c 3`. No `pyperf system tune`: absolute numbers are noisy, so
  compare columns within a row.
- CPython 3.14.6, Rust 1.97.0, release build (`maturin develop --release`,
  `DEBUG_BUILD == False`), base commit `11e4d1e` (PR #7 merged).

```sh
maturin develop --release
for i in 1 2 3; do
    taskset -c 3 python benchmarks/experiments/wrapper_overhead.py --rounds 15 --loops 20000
done
```

Each process interleaves every (variant, case) cell for 15 rounds of 200,000
calls (10 unrolled calls per loop iteration) and prints medians.

**Limitations:** one machine and one CPython version. The run is short and
the machine was not tuned. Only `ops_001/020/050` with four `int` arguments
and the two-argument fallback cases were measured. Differences under about
10 ns are not resolved.
