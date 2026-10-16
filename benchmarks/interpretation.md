*Written by hand from the numbers above. Rewrite it whenever the baseline
is regenerated.*

**What changed.** A warmed `@snekit` call used to run the wrapper, then
`CompiledI64.__call__` (a Python method that checked arity, exact `int`
type and the i64 range of every argument), then `I64Function.call`
(which checked the same things again in Rust and returned a
`(status, value)` tuple), then a Python status check. It now runs the
wrapper and one native call, `I64Function.try_call(args)`. That call
validates every argument once, runs the machine code and returns the
result `int`, or `None` for a guard miss. `CompiledI64.__call__` still
exists as the diagnostic path; the decorator no longer calls it.

**How to read the comparisons.** This machine ran plain Python
noticeably slower than it did during the Stage 6 baseline run: the
four-argument no-op call takes 52 ns instead of 41 ns, and Python's
fitted cost per operation is 17.8 ns instead of 9.5 ns. Comparing
against the historical numbers alone would therefore mix the
optimization with the machine. The unchanged previous code (`3df7644`)
was measured again in the same session on the same machine
([`baseline/control-3df7644/`](baseline/control-3df7644)); that is the
controlled comparison. Its plain-Python rows differ from this run by
7–13%, against 28–87% for the historical run.

**The warmed call is about 2.5–3× cheaper.** Against the same-session
control, warmed `@snekit` takes 0.31–0.42× the time at every workload
size, for example 1,446 → 544 ns at 1 operation and 1,741 → 725 ns at
250. The fitted fixed cost per call fell from 1,449 ns to 498 ns. That
saves about 950 ns per call, roughly what the diagnostic layer costs
(`CompiledI64` total − `I64Function.call` is 885–1,119 ns in this run).
Against the historical Stage 6 numbers, measured on the faster machine
state, warmed `@snekit` is still 0.41–0.51× (1,154 → 544 ns at 1
operation, 1,410 → 725 ns at 250; fixed cost 1,158 → 498 ns).

**The native fast call is cheaper than the old raw call.**
`try_call` costs 117–318 ns, against 188–406 ns for `I64Function.call`
in the same run. It skips the `(status, value)` tuple and its status
`int`, keeps up to 8 arguments in a stack buffer instead of a `Vec`, and
takes its argument positionally.

**Crossover.** Warmed `@snekit` now beats plain Python for `ops_050`
(0.61×), `ops_100` (0.36×) and `ops_250` (0.16×). It is still slower
for `ops_001` (7.7×), `ops_005` (3.6×) and `ops_020` (1.4×). The fitted
lines cross at about 29 operations, against about 94 for the control run
and 134 for the Stage 6 run. Part of the move from 134 is the slower
machine: with the Stage 6 run's Python line, today's `@snekit` line
would cross at about 55 operations. Either way, the crossover moved by
roughly a factor of 2.5–3. A separate CPU-pinned repeat that interleaved the new and control code
([`baseline/repeat-pinned/`](baseline/repeat-pinned)) confirmed the
bracket: `@snekit` took 504 ns against 345 ns for Python at `ops_020`
(1.46×, slower) and 495 ns against 811 ns at `ops_050` (0.61×, faster),
while the control code took 1,486 ns at `ops_050` (2.0× slower than
Python).

**Break-even.** Three workloads now break even: `ops_050` after about
7,200 calls, `ops_100` after about 4,500 and `ops_250` after about
3,200. Previously only `ops_250` did, after about 4,200 calls in the
control run (8,300 in the Stage 6 run). Lifecycle cost and saving per
call both carry 15–25% noise, so treat these as orders of magnitude.
Compilation itself did not change: the control run's lifecycle costs
match this run's within noise.

**Fallback became cheaper where native code is tried.** A guard miss
used to construct, raise and catch a `NativeGuardFailure`. It now
returns `None`. Against the control, the `@snekit` call with `float`
arguments fell from 2,014 to 547 ns, with an `int` outside i64 from 2,246
to 668 ns, and with an overflowing `*` from 3,016 to 650 ns (0.22–0.30×).
The native miss itself takes 74–106 ns; the rest of the 430–515 ns
overhead is the decorator wrapper and the Python fallback call. Paths
that never reach native code (`keyword_args`, `default_args`) are
unchanged within noise (0.99×). In the main runs the
`cold` and `disabled` paths looked 20–23% slower than the control,
although their code is byte-for-byte unchanged. The pinned, interleaved
repeat did not reproduce that: `cold` 462 against 444 ns, `disabled`
360 against 336 ns and `keyword_args` 540 against 550 ns, all within
noise and in both directions. The main-run difference was machine drift
between the runs.

**The largest remaining cost is the decorator wrapper.** It is now
64–78% of a warmed call for up to 100 operations (343–427 ns), and 56%
at 250 operations. It is the same cost that makes the cold, disabled,
keyword and default-argument paths 315–525 ns slower than plain Python.
The wrapper is a Python closure taking `*args` and `**kwargs`. Every call
builds an argument tuple and an empty keyword dict, increments
`call_count` in the function's `__dict__`, and reads the specialization,
`__code__`, the arity and the bound fast call. The native fast call
itself (112 ns fixed, 0.84 ns per operation) is no longer the
bottleneck. It still costs about twice a plain Python four-argument call
(52 ns), so 1–5 operation functions cannot win even with a free wrapper.

**Recommended next experiment: make the wrapper cheaper.** Two options
are worth measuring before choosing:

- a wrapper generated per function with the function's own positional
  signature, so the common call needs neither `*args` packing nor a
  keyword dict; or
- a native callable (a Rust type with vectorcall) that does the
  `call_count`, `__code__` and arity checks and the fast call without a
  Python frame, and calls the original function for everything else.

Both must keep every guarantee in the decorator's correctness contract:
`call_count`, metadata, method binding, `__code__` invalidation and
exception context. Measure the cold, disabled and keyword paths too: they
pay the same wrapper cost.

**Reliability caveats.** As before, this is a shared 4-vCPU KVM guest
without `pyperf system tune`. Nearly every benchmark trips pyperf's
stability checks (standard deviation 15–25% of the mean). Nothing else
ran on the machine during the main runs, the new and control runs ran
back to back, and each run records a single clean commit. The
conclusions rest on 2.5–4.5× differences, much larger than that noise.
The individual nanosecond values, and the exact crossover and
break-even counts, should not be compared across machines or treated as
precise.
