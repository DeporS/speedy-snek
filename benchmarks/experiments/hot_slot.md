# Quick experiment: a Rust-owned hot specialization slot

**Conclusion: adopted.** Keeping a compiled wrapper's specialization in a
write-once Rust `HotSlot` removes every Python attribute lookup from a
warmed call. Against `main`, a warmed `a + b` call executes **42% fewer
instructions** and took about half the time (0.52× in the final run).
The other compiled cases, guard misses and omitted defaults execute
25–36% fewer instructions and took 0.59–0.81× the time. Keyword calls
execute 15% fewer. A disabled wrapper, which cannot use the slot,
executes 8 more instructions per call (+0.4%) and took 0.98–0.99× the
time in two long runs. No behaviour changed; see
[Correctness](#correctness).

This is a short, interleaved A/B measurement plus deterministic
instruction counts, not a pyperf baseline. It does not replace or update
[`../RESULTS.md`](../RESULTS.md).

## Question

The audit found that a warmed call through `NativeWrapper` spends much of
its time on generic Python attribute lookups: `state.specialization`,
`spec.code`, `func.__code__` and `spec.function`. Only `status` changes
after compilation, and it only matters once the specialization is gone.
A scratch prototype that cached them in Rust cut a simple warmed call by
about 40%. Does a clean version keep that gain without changing
behaviour, and without slowing down the paths that cannot use it?

## What was built

**`HotSlot`** (`src/lib.rs`) is a small private pyclass. It holds a
`OnceLock` with the compiled code object, the `I64Function` and the
`RangeGuard` (if any), plus a one-way `retired` flag.

- `publish(code, function)` stores them once. It accepts only a genuine
  `I64Function` or `GuardedI64Function` (whose function and guard are
  unpacked) and leaves the slot empty for anything else, so test doubles
  never reach it. A second publish raises `RuntimeError`.
- `retire()` sets the flag for good.
- `live` (published and not retired) is a read-only diagnostic for tests.
- Reading it is two atomic loads and no lock. Compilation is rare and
  already serialized by the state's lock, so a `OnceLock` plus an atomic
  flag is all the synchronization it needs.
- It holds strong references to everything a native call uses, and the
  wrapper holds it, so nothing can be freed during a call. Its contents
  are never replaced or dropped before the slot itself.
- It is not tracked by the garbage collector. Code objects,
  `I64Function`s and `RangeGuard`s are not GC objects either, so no cycle
  the collector could break runs through it. The wrapper still reports
  the slot in `__traverse__`.

**Ownership.** `snekit` creates one slot and gives it to both the
`NativeWrapper` (a new optional fourth constructor argument) and the
`_JitState` (a new optional second argument). That keeps it out of any
reference cycle: the state never holds the wrapper. A wrapper or state
made without a slot, as the older experiments and a few tests do, behaves
exactly as before.

**Publishing and retiring** stay in the Python state machine:

- `_try_compile` publishes after it has stored the complete
  `_Specialization` and set `status = COMPILED`, so the slot is only ever
  live while the state is `COMPILED`.
- `state.specialization` is now a property in front of a plain
  `_specialization` slot. Its setter retires the hot slot before storing
  the new value. `_disable` assigns it on every path to `DISABLED`, so
  all eight `_DisabledReason`s retire the slot, and a test that injects a
  test double into `state.specialization` retires it too. Only
  `_try_compile` stores `_specialization` directly, to publish. The
  wrapper's generic path reads `_specialization`, a plain member, so its
  cost is unchanged; only writers and Python readers (tests, benchmarks)
  go through the property.
- `_Specialization` is now immutable. The slot is a copy of the published
  specialization, so mutating that object in place would silently stop
  affecting the wrapper. The tests and benchmark checks that did this
  now assign a replacement to `state.specialization` instead.

**The call.** `NativeWrapper.__call__` counts the call, then tries the
slot. It may use it when:

- the slot is live;
- the wrapped object is a `FunctionType` (checked once, at construction:
  `FunctionType` cannot be subclassed and the wrapper never replaces it);
- `PyFunction_GetCode(func)` is the compiled code object;
- the range guard, if any, is still valid.

Then a call with keywords runs the original function, a call with another
arity runs it without a native attempt, and anything else runs the same
`try_call_tuple` as before: the result, or the original function once
for a guard miss, or `native_failed` for an unexpected `Exception`.
Those are exactly the generic path's outcomes for the same call.

When the code object or the guard does not match, the call takes the
unchanged generic path. That path reads the state, sees the same
mismatch and calls `invalidate()` or `binding_changed()` exactly as
before, which disables the specialization and retires the slot. So no
stale machine code runs, the first call after the change already runs the
new code or binding, and `peek()` reports the same state as before.

**Deviations from the suggested outline.**

- The slot is a separate object shared by the wrapper and the state, not
  a field of the wrapper. The state has to publish into it, and giving
  the state the wrapper would create a wrapper ↔ state cycle, so a
  wrapper that is not in any other cycle would no longer be freed by
  reference counting alone.
- Retirement hooks the assignment to `state.specialization` instead of
  adding a call at each disable site. Every transition already goes
  through that assignment, and so does every existing test double.
  A first version hooked it with `_JitState.__setattr__`. That ran a
  Python function on every attribute store, including `cold_call`'s
  `warmup_calls += 1`, and made a cold call execute 49% more
  instructions (3,005 → 4,494). The property only costs on assignments
  to `specialization`: a cold call now executes 2,988.
  `test_a_cold_call_runs_no_state_code_but_cold_call` pins this.
- A failed code or guard check falls through to the generic path instead
  of making the state transition itself, so the transition logic exists
  once.

## Results

### Timing

`python benchmarks/experiments/hot_slot.py --processes 5 --cpu 3
--baseline …`: 5 processes per build, alternating `main` (`e7616fe`) and
this branch, each interleaving every cell for 15 rounds of 200,000 calls.
Median of process medians in ns per call, min–max over processes.
`generic` is this branch's wrapper made without a slot, the pre-change
dispatch, timed in the same processes.

| case | python | generic | snekit | snekit / generic | `main` snekit | snekit / `main` |
|---|---|---|---|---|---|---|
| `add` (`a + b`) | 22 (22–25) | 113 (109–114) | 59 (59–64) | 0.52–0.56 | 114 (113–120) | 0.52 |
| `ops_030` | 231 (225–263) | 163 (155–168) | 105 (101–112) | 0.62–0.68 | 160 (155–169) | 0.66 |
| `ops_100` | 753 (691–816) | 208 (193–240) | 157 (136–160) | 0.66–0.79 | 194 (189–201) | 0.81 |
| `range_32` | 735 (649–754) | 177 (165–182) | 116 (111–130) | 0.64–0.72 | 173 (163–187) | 0.67 |
| float guard miss | 37 (35–39) | 152 (147–168) | 88 (85–103) | 0.54–0.61 | 148 (138–152) | 0.59 |
| overflow guard miss | 80 (73–89) | 220 (197–268) | 158 (142–189) | 0.61–0.78 | 221 (206–235) | 0.71 |
| keyword call | 43 (40–48) | 264 (255–280) | 217 (200–239) | 0.77–0.94 | 266 (250–311) | 0.82 |
| omitted default | 33 (33–37) | 128 (127–139) | 77 (76–81) | 0.56–0.60 | 131 (125–149) | 0.59 |
| disabled wrapper | 35 (30–41) | 102 (102–122) | 111 (103–116) | 0.95–1.13 | 100 (99–104) | 1.11 |

Earlier runs of the same command, with an intermediate version that
differed only in how the slot is retired, gave the same picture for the
compiled cases. The disabled row is the noisiest: across runs it read
1.01, 1.09, 1.24 and 1.11 against `main`, while plain `python` itself
varied by up to 2× between processes of one build (for example 31–61 ns).
Two longer runs of only that case, `--cases disabled --processes 15
--rounds 30`, on CPUs 3 and 1, gave:

| run | python | generic | snekit | snekit / generic | `main` snekit | snekit / `main` |
|---|---|---|---|---|---|---|
| CPU 3 | 35 (31–38) | 102 (93–121) | 102 (90–126) | 0.95–1.08 | 103 (98–164) | 0.99 |
| CPU 1 | 35 (28–39) | 103 (84–110) | 104 (84–114) | 0.98–1.05 | 106 (79–130) | 0.98 |

So this shared VM cannot resolve a few-percent difference by timing; the
instruction counts below can.

### Instructions

Retired instructions per loop iteration (one call plus the loop), from
`valgrind --tool=callgrind` with `PYTHONHASHSEED=0`: the difference
between a run with 200,000 and one with 100,000 calls, divided by
100,000, so setup, warmup and compilation cancel out. Repeated runs give
identical counts.

| case | python | `main` snekit | generic | snekit | snekit / `main` |
|---|---|---|---|---|---|
| `add` (`a + b`) | 541 | 1883 | 1913 | 1082 | 0.575 |
| `ops_030` | 4003 | 2390 | 2420 | 1589 | 0.665 |
| `ops_100` | 12041 | 2739 | 2769 | 1938 | 0.708 |
| `range_32` | 11524 | 2648 | 2678 | 1795 | 0.678 |
| float guard miss | 740 | 2392 | 2421 | 1592 | 0.666 |
| overflow guard miss | 1452 | 3281 | 3281 | 2471 | 0.753 |
| keyword call | 873 | 3891 | 3896 | 3298 | 0.848 |
| omitted default | 728 | 2221 | 2250 | 1423 | 0.641 |
| disabled wrapper | 747 | 1871 | 1876 | 1879 | 1.004 |
| cold wrapper | | 3005 | | 2988 | 0.994 |

The cold row is a wrapper kept cold for good, as in the pyperf suite's
`fallback.cold`, counted with a separate script with 20,000 and 40,000
calls.

What the rows show:

- **Warmed calls.** Every compiled case saves about 800 instructions and
  35–60 ns per call, whatever the function does: that is the cost of the
  four attribute lookups. For `a + b`, it is 42% of the call's
  instructions. Relative to plain Python, `ops_030` is now 0.45× as long
  instead of 0.69×, so the break-even point moves to a smaller function.
  The committed baseline's crossover estimate is not recomputed here.
- **Guard misses** (`float`, overflow) and **omitted defaults** save the
  same amount: they no longer read the state before falling back.
- **Keyword calls** never run native code, but they now skip the lookups
  too. CPython's and PyO3's two kwargs dicts still dominate.
- **Disabled wrappers** pay one extra check of a retired slot: 8 more
  instructions out of 1,871 (0.4%), well below the 5% regression limit.
  Their timing in the long runs is 0.98–0.99× `main`.
- **`generic`**, the fallback that only wrappers without a slot and
  injected test doubles still take, executes up to 30 instructions more
  than `main` (one more branch and different code layout); no production
  call takes it once a function is compiled.
- **Range loops.** The guard check stays one atomic load; `range_32`
  saves the same fixed cost as straight-line code.

Every case was verified in every process before timing (see
`hot_slot.py`): the result equals the undecorated function's, with the
same type; the call takes the expected path (native, guard miss or
Python, counted with `sys.setprofile` and `I64Function.try_calls`);
`snekit` has a live slot exactly when the function compiled; `generic`
has none. Each wrapper's state is checked again after timing.

## Correctness

- The existing suites pass unchanged except where a test mutated the
  published `_Specialization` in place (two `test_jit_compile.py` tests,
  one benchmark workload test, and the canary check in
  `native_wrapper.py`). They now inject a replacement specialization or,
  for the canary, compile a second wrapper to the canary code.
  `test_the_hot_slot_runs_the_compiled_machine_code` adds the canary
  proof for the slot itself.
- New tests in `test_native_wrapper.py`, `test_for_range.py`,
  `test_peek.py` and `test_chill.py` cover:
  - warmed calls use the slot: the specialization is swapped, with
    `object.__setattr__`, for an object that fails the test when read,
    and calls still return native results with the same counters;
  - guard misses (`float`, `bool`, `int` subclasses, out-of-range ints,
    overflow, non-numbers), invalid arities and keywords, keyword calls
    and defaults, each with the original's result or `TypeError`, the
    same `call_count` and native-attempt counts, and the slot still live;
  - a call-by-call comparison with a wrapper without a slot, including a
    `__code__` replacement: result or exception, Python frames, native
    attempts, `call_count`, status, reason, compile attempts and warmup;
  - `__code__` replaced before compilation (the new code is published),
    after it (retired; restoring the old code does not revive it), and
    with another signature;
  - range binding changes before and between hot calls;
  - every `_DisabledReason` leaves the slot retired or never published,
    and later calls run Python;
  - injected test doubles, a wrapper without a slot and a wrapper of a
    callable object (with a published slot) all take the generic path;
  - `publish`'s rules: only genuine native functions, write-once,
    one-way;
  - four threads calling while `__code__` is replaced: no stale result
    once the replacement is visible, and `call_count` is exact;
  - the slot releases the machine code with the wrapper; a
    module-level range function whose globals hold its wrapper is
    collected;
  - `peek()` reports field for field the same findings for a wrapper with
    and without the slot over the same calls;
  - a wrapper installed by `bite()` uses its slot before and after
    `chill()`.
- Nine bugs were injected one at a time: a missing code check, a missing
  guard check, keyword calls reaching native code, a missing arity
  pre-check, a no-op `retire`, a guard miss returning `None` instead of
  falling back, a `specialization` setter that does not retire, the
  publishing compilation retiring its own slot, and no publish. Each was
  caught. So was putting back a `_JitState.__setattr__`.
- The suite also passes with the `ci` build profile (debug assertions and
  overflow checks on).

## Limitations

- One machine, one CPython version, and a noisy shared VM. Timing
  differences under about 10 ns are not resolved; the instruction counts
  are the evidence for the disabled case.
- Instruction counts under Valgrind are not cycles: they say how much
  work was removed, not exactly how much time. Cache and branch effects
  are not in them; the long timing runs of the disabled case are the
  check on those.
- Free-threaded CPython is not supported or claimed. The slot relies on
  the GIL, like `try_calls` and the range guard.
- What remains of a warmed call is the `tp_call` argument tuple, PyO3's
  trampoline and argument handling, the call counter, argument
  validation and the result `int`. Removing more would need vectorcall,
  which is out of scope here.

## Environment and commands

- Intel Xeon @ 2.10 GHz, 4 vCPU cloud VM, Linux 6.18. No
  `pyperf system tune`.
- CPython 3.14.6, Rust 1.97.0, PyO3 0.29.3, release builds
  (`DEBUG_BUILD == False`) of `main` at `e7616fe` and of this branch.

```sh
# A release build of the base commit, importable from ../main/python
# without touching this checkout's editable install.
git worktree add ../main e7616fe
(cd ../main && maturin build --release -o ../main-dist)
unzip -o -j ../main-dist/speedysnek-*.whl 'speedysnek/_native*.so' \
    -d ../main/python/speedysnek/
maturin develop --release              # this checkout

python benchmarks/experiments/hot_slot.py --processes 5 --cpu 3 \
    --baseline ../main/python
python benchmarks/experiments/hot_slot.py --processes 15 --rounds 30 \
    --cpu 3 --cases disabled --baseline ../main/python

# Instructions per call: the difference between 2N and N calls.
for n in 100000 200000; do
    PYTHONHASHSEED=0 PYTHONPATH=../main/python valgrind --tool=callgrind \
        --callgrind-out-file=/dev/null \
        python benchmarks/experiments/hot_slot.py --calls add snekit $n
done
```

Drop `PYTHONPATH` for this branch's build, and use `python` or `generic`
as the variant for the other columns.
