# Quick experiment: a conservative miss policy (parking)

**Conclusion: adopted.** A wrapper whose calls clearly cannot use its
i64 specialization now *parks*: it runs the original function without a
native attempt, and re-probes now and then. Warmed successful native
calls execute **1–3 more instructions** than on `main` (at most 0.2%,
well under a nanosecond), and their timings match `main` within this
VM's noise. A function whose warmup calls do not fit is no longer
compiled at all: its first nine float calls took **26 µs instead of
15 ms**, because neither the frontend nor the compiler is loaded. In the
steady state, parked calls execute 3–10% fewer instructions than
`main`'s guard-miss path. The gain is largest for repeated overflow,
where the native code ran before falling back. Keyword and default calls
cost the same as before. Behaviour is unchanged; see
[Correctness](#correctness).

This is a short, interleaved A/B measurement plus deterministic
instruction counts, not a pyperf baseline. It does not replace or update
[`../RESULTS.md`](../RESULTS.md).

## Question

After the [hot slot](hot_slot.md), a compiled call that misses the guard
still pays for the native attempt on every call, forever. That covers a
function only ever called with floats, a method called on an instance
(`self` is not an int), and repeated overflow. Keyword calls and calls
that omit a default make no attempt, but warmup still imported the
frontend and the compiler for them and compiled code that could never
run. Can the wrapper stop paying for those, using only facts it already
has at the call boundary, without slowing down the calls that do fit?

## What was built

See `speedysnek/_decorator.py` (*the miss policy*) and `HotSlot` in
`src/lib.rs`.

- **Warmup profile.** During the 8 warmup calls, the Rust wrapper passes
  `cold_call` the call's positional count when the call has no keywords
  and only exact `int`s in i64, and -1 otherwise. The state compares that
  count with `co_argcount`. These checks run no user code: the wrapper
  checks exact types and calls `PyLong_AsLongLongAndOverflow`, and
  `FunctionType` cannot be subclassed. If fewer than
  `_MIN_FITTING_WARMUP_CALLS` (4) warmup calls fit, the state becomes
  `PARKED` (`warmup_mismatch`) and nothing is imported or compiled.
  Otherwise compilation goes ahead as before.
- **Miss count.** While the slot is active, it counts the calls in a row
  that could not run natively (keywords, another arity, a guard miss). A
  native result resets the count with one relaxed store. The
  `MISS_LIMIT`th (32nd) miss in a row calls `state.repeated_misses()`,
  which parks the specialization (`repeated_misses`) and keeps it.
- **Parked.** The slot's mode is parked. A call reads no state attribute:
  it counts towards the interval and runs the original function. For a
  compiled function, it first checks `__code__` and the range guard,
  exactly as an active slot does, so a replaced `__code__` or a changed
  binding still disables it for good on the next call.
- **Re-probe.** After `REPROBE_INTERVAL` (64) parked calls, the next call
  that fits (no keywords, the compiled arity, only exact ints in i64) is
  the re-probe. It tries the compiled code, and `state.reprobe_succeeded()`
  makes the state `COMPILED` again if the call runs natively. A function
  parked during warmup is compiled instead, at most once in its life
  (`state.reprobe()`). A failed re-probe starts the interval again. Calls
  that do not fit never re-probe, so a float-only function makes no
  native attempt at all.
- **Counts.** `I64Function` now also counts `try_misses`, on the miss
  path only, so `peek()` reports exact native hits and misses
  (`try_counts` reads both together). A native result pays only for the
  existing `try_calls` counter.

The constants are fixed and documented next to their definitions in
`src/lib.rs`. They are not tuned from timings. 32 misses cost at most a
few microseconds before parking, far less than compiling did, and one
native result restarts the count, so an occasional miss never parks.
With an interval of 64, a parked function pays for about one native
attempt per 64 calls, and a workload that changes back gets native code
within 64 calls.

Terminal disables keep their meanings: compiler failures, `__code__`
replacement, range-binding changes and native errors all still end in
`DISABLED`, from either `COMPILED` or `PARKED`.

### Two codegen findings

- The first version made the new `hot_call` helper `#[inline]`. rustc
  outlined it anyway, and a warmed `ops_030` call executed **+48**
  instructions. Making it, `slot` and `is_current` `#[inline(always)]`,
  and counting `try_misses` at the miss returns instead of wrapping the
  result, brought the success path back to +1–3 instructions.
- With the miss bookkeeping in a `#[cold]` out-of-line function, each
  miss cost about 30 instructions more than `main`. Inlining the counter
  and keeping only the rare Python call out of line cut that to about 10.

## Results

### Timing

`python benchmarks/experiments/miss_policy.py --processes 5 --cpu 3
--baseline …`: 5 processes per build, alternating `main` (`c064528`) and
this branch, each interleaving every cell for 15 rounds of 200,000
statements. Every case is a steady state: warmup, then 1,000 timed-style
calls, then verification (see below) before anything is timed. Median of
process medians, in ns per call, with the min–max over processes.

| case | python | `main` snekit | snekit | snekit / `main` |
|---|---|---|---|---|
| `add` (`a + b`) | 26 (25–28) | 72 (71–73) | 74 (72–82) | 1.02 |
| `ops_030` (break-even) | 271 (264–289) | 123 (120–123) | 123 (121–132) | 1.01 |
| `ops_100` (medium) | 852 (809–910) | 175 (174–177) | 175 (172–188) | 1.00 |
| `range_32` | 800 (745–898) | 117 (114–120) | 121 (119–123) | 1.03 |
| float-only calls | 43 (41–45) | 104 (102–107) | 106 (100–110) | 1.02 |
| instance method | 26 (26–34) | 183 (170–187) | 168 (165–193) | 0.92 |
| keyword calls | 48 (47–53) | 260 (246–269) | 258 (248–269) | 1.00 |
| omitted default | 40 (39–41) | 95 (92–99) | 93 (93–99) | 0.98 |
| repeated overflow | 82 (80–91) | 176 (168–177) | 153 (145–174) | 0.87 |
| alternating int/float | 55 (53–58) | 104 (100–106) | 102 (101–120) | 0.98 |
| disabled wrapper | 42 (42–44) | 124 (122–132) | 128 (121–136) | 1.03 |

An earlier run of the same command (with the miss bookkeeping still out
of line) gave 0.99, 1.02, 1.04 and 1.03 for the four compiled cases, and
0.94 for float-only calls. So differences of a few ns are noise on this
shared VM. The instruction counts below resolve them.

### Instructions

Retired instructions per call, from `valgrind --tool=callgrind` with
`PYTHONHASHSEED=0`: the difference between 260,000 and 130,000 calls,
divided by 130,000, so setup, warmup, compilation and settling cancel
out. Repeated runs give the same counts to within one instruction.

| case | `main` | this branch | branch / `main` |
|---|---|---|---|
| `add` (`a + b`) | 1081 | 1083 | 1.002 |
| `ops_030` | 1589 | 1590 | 1.001 |
| `ops_100` | 1937 | 1940 | 1.002 |
| `range_32` | 1794 | 1796 | 1.001 |
| float-only calls | 1591 | 1529 | 0.961 |
| instance method | 2306 | 2243 | 0.973 |
| keyword calls | 3298 | 3288 | 0.997 |
| omitted default | 1423 | 1427 | 1.003 |
| repeated overflow | 2462 | 2219 | 0.901 |
| alternating int/float | 1391 | 1397 | 1.004 |
| disabled wrapper | 1879 | 1883 | 1.002 |

### Warmup that does not fit

A fresh process decorates `(a + b) * 2` and makes its first nine calls,
the warmup and the deciding call. The median of 15 processes on CPU 3,
including every import those calls trigger:

| calls | `main` | this branch |
|---|---|---|
| floats | 15.1 ms (compiled) | 26 µs (parked) |
| ints | 17.2 ms (compiled) | 16.0 ms (compiled) |

What the rows show:

- **Successful native calls** pay for one relaxed store that resets the
  miss count: 1–3 instructions, ≤0.2%, under the 3 ns target. The timing
  differences (0.99–1.04 across two runs) are within noise.
- **Float-only calls and methods** are parked at warmup. They never load
  the compiler and make no native attempt. In the steady state they
  execute 3–4% fewer instructions than `main`'s guard miss. That saving
  is small because the hot slot already made a float guard miss cheap:
  the first argument fails the exact-`int` check at once.
- **Repeated overflow** parks after 32 misses and saves the most (10%),
  because `main` ran the machine code on every call before falling back.
  It re-probes once per 65 calls, and each re-probe overflows again.
- **Keyword calls and omitted defaults** never made a native attempt.
  Parked, they cost the same as before: the default case is +4
  instructions, the Python `reprobe()` call its re-probes make (one per
  65 calls) to check the arity. Their gain is the compilation they no
  longer trigger.
- **Alternating int/float** never parks: each native result resets the
  count. It pays about 10 instructions per miss for the bookkeeping,
  +0.4% per call.
- **Disabled wrappers** execute 4 more instructions (+0.2%): one more
  comparison of the slot's mode.
- **Parked calls vs. the fallbacks.** A parked call (1529 for floats)
  executes fewer instructions than a disabled wrapper (1883), which still
  reads two state attributes, and fewer than the old guard miss (1591).
- **Recovery.** A later compatible workload gets native code back at its
  first re-probe: the tests drive each case through it, and the timing
  harness verifies the steady state of each.

Every case was verified in every process before timing. Each result
equals the undecorated function's, with the same type. Over 650 calls,
native attempts were counted with `I64Function.try_calls` and frames of
the original function with `sys.setprofile`, and both were checked
against the path that build must take. On `main`, float-only calls,
methods and overflow made one attempt and one Python frame per call. On
this branch they made none (overflow: one per re-probe interval) and one
frame. The status (compiled, parked or disabled) was checked again after
timing. `benchmarks/tests/test_miss_policy_experiment.py` runs the same
verification in CI.

## Correctness

- New tests in `python/tests/test_miss_policy.py` cover:
  - warmup: exact ints compile; floats, one float, `bool`, an `int`
    subclass, values outside i64, strings, keyword calls, instance
    methods and omitted defaults park without compiling; the threshold
    at exactly 4 and 3 fitting calls;
  - no frontend or compiler import in a fresh process that only makes
    calls that do not fit;
  - re-probes: a function parked by floats compiles at its first fitting
    call after the interval, only once in its life; an arity mismatch
    restarts the interval; a rejected function is rejected then;
  - no user code runs during profiling or re-probing: metaclass hooks,
    `__index__`, `__bool__`, `__eq__` and attribute lookups;
  - compiled misses: occasional misses keep the specialization; floats,
    overflow, `bool` and values outside i64 park on exactly the 32nd
    miss in a row; keyword and default calls count; one native result
    restarts the count; alternating calls never park; parked float calls
    make no native attempts; overflow re-probes once per interval;
    successful and failed re-probes; the compiled arity;
  - results, exceptions and their types matching the undecorated function
    over thousands of calls that drive the state through every
    transition;
  - `__code__` replacement, range-binding changes and native errors while
    parked still disable it for good; `call_count` and metadata stay
    right; `bite()`, `chill()`, `hunt()` and `@nosnek` with a parked
    function; four threads under the GIL;
  - `peek()`: exact rendering of parked findings of both kinds and of
    cold, compiled and disabled ones; exact native hits and misses; no
    overclaiming; parked distinguishable from disabled.
- Changed existing tests: methods called on instances, warmup with only
  keyword calls, and rejected functions warmed up with strings now park
  (the tests now warm the rejected functions up with ints). `peek()`
  output gained the hit/miss split and the `parked` count.
- Seven bugs were injected one at a time, and each was caught: a hit
  that does not reset the count, re-probes that never come due, parked
  calls that skip the `__code__` check, a re-probe that does not restart
  the interval, overflow not counted as a miss, re-probes without the
  exact-int check, and warmup that never parks.
- The suite passes with the `release` and `ci` build profiles.

## Limitations

- One machine, one CPython version, and a noisy shared VM. Timing
  differences under about 10 ns are not resolved; the instruction counts
  are the evidence for those.
- The policy only uses call shape and native outcomes. It cannot tell
  whether native code is faster for a function: a tiny function like
  `a + b` stays compiled although plain Python is faster. That would
  need the timing-based adaptation this change deliberately leaves out.
- A function whose calls fit only now and then (say, 1 in 50) can cycle:
  it parks after 32 misses, unparks at a fitting re-probe, and parks
  again. The cost is bounded: about 32 miss overheads per cycle of
  roughly 100 calls, and no recompilation.
- A function parked during warmup is compiled by a single fitting
  re-probe call, which is weaker evidence than a warmup. Compilation
  still happens at most once per wrapper.
- Free-threaded CPython is not supported or claimed. The counters rely
  on the GIL, like `try_calls` and the hot slot.

## Environment and commands

- Intel Xeon @ 2.10 GHz, 4 vCPU cloud VM, Linux 6.18. No
  `pyperf system tune`.
- CPython 3.14.6, PyO3 0.29.3, release builds (`DEBUG_BUILD == False`)
  of `main` at `c064528` and of this branch.

```sh
git worktree add ../main c064528
(cd ../main && maturin build --release -o ../main-dist)
unzip -o -j ../main-dist/speedysnek-*.whl 'speedysnek/_native*.so' \
    -d ../main/python/speedysnek/
maturin develop --release              # this checkout

python benchmarks/experiments/miss_policy.py --processes 5 --cpu 3 \
    --baseline ../main/python

# Instructions per call: the difference between 2N and N calls.
for n in 130000 260000; do
    PYTHONHASHSEED=0 PYTHONPATH=../main/python valgrind --tool=callgrind \
        --callgrind-out-file=/dev/null \
        python benchmarks/experiments/miss_policy.py --calls overflow $n
done
```

Drop `PYTHONPATH` for this branch's build.
