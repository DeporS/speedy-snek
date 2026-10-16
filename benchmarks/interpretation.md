*Written by hand from the numbers above. Rewrite it whenever the baseline
is regenerated.*

**Does warmed `@snekit` beat plain Python?** Only for the largest
workload. At 250 operations a warmed call takes 1,410 ns against
2,408 ns for Python (0.59×). It is slower everywhere else: 25.2× at
1 operation, 13.4× at 5, 5.9× at 20, 2.4× at 50 and 1.33× at 100. A
separate, CPU-pinned repeat of the 100- and 250-operation workloads
([`baseline/repeat-ops100-250.json.gz`](baseline/repeat-ops100-250.json.gz))
reproduced both results: 0.58× at 250 operations and 1.23× at 100. The two
fitted lines cross near 134 operations
((1,157.8 − 23.3) / (9.478 − 0.987)). Below roughly that size, the
current hot path cannot win. Typical small numeric functions are far
below it.

**Break-even.** Only `ops_250` breaks even, after about 8,300 calls: an
8.3 ms lifecycle divided by about 1.0 µs saved per call. Both terms have
15–20% noise, so treat this as an order of magnitude (about 10⁴ calls),
not a precise count. No other workload breaks even with the current hot
path.

**Where the per-call overhead comes from.** A warmed `@snekit` call has
a nearly constant cost: about 1.16 µs fixed, plus about 1 ns per
operation. Most of that is spent in Python *before* the native boundary.

- The `CompiledI64.__call__` guard layer costs 680–780 ns per call,
  55–66% of the total. It is a Python method that re-checks arity, exact
  `int` type and i64 range for every argument. The Rust side checks the
  same things again in `I64Function.call`.
- The decorator wrapper adds another 240–410 ns (21–33%).
- The raw native call is only 142–320 ns (12–23%).

The Python↔Rust boundary is real but not dominant. The raw native call's
fixed cost (about 140 ns) is about 3.5× a plain four-argument Python call
(40.6 ns). Even so, it is only an eighth to a quarter of a warmed
`@snekit` call.

**Is native arithmetic faster than Python arithmetic?** Yes, by a wide
margin. The fitted marginal cost is 0.72 ns per operation for the raw
native call against 9.48 ns for Python, about 13× cheaper. The raw native
call beats the *entire* Python function from 20 operations up (0.73× at
20, 0.13× at 250). It loses at 1 and 5 operations, where its ~140 ns fixed
cost exceeds Python's whole call (46–85 ns). The generated code is not the
problem; everything wrapped around it is.

**Fallback cost.** When the wrapper skips native code without trying it
(cold, disabled, keyword arguments, omitted defaults), it adds
220–375 ns to the Python call. When a guard rejects the arguments, it
adds 1.65–2.24 µs. In those cases `NativeGuardFailure` is constructed,
raised and caught on every call, and the overflow case also runs the
native code first. Functions that are usually called with floats or
large ints pay that penalty on every call.

**One-time costs.** The full lifecycle (warmup plus first compilation)
grows roughly linearly, from 256 µs at 1 operation to 8.3 ms at 250.
Cranelift code generation is the largest part (140 µs to 4.2 ms); the
frontend costs 29 µs to 2.6 ms. `import speedysnek` adds about 12 ms over
bare interpreter startup, and loading the backend adds about 34 ms. Both
are paid only once per process.

**Recommended next experiment: take the Python guard layer out of the
hot path.** When a specialization is published, the wrapper should call
one native entry point directly with its `*args`. Rust already validates
exact `int` and i64 range and the generated code checks arity, so the
entry point would return the result `int` directly and signal a guard
failure with a cheap sentinel instead of a Python exception. This removes
the largest measured component, 55–66% of every warmed call. In the best
case a warmed call would cost roughly what the decorator wrapper and the
raw native call cost today, about 450–630 ns instead of 1.15–1.41 µs.
With a fixed cost of about 440 ns (today's wrapper plus raw native
call), the fitted crossover would move from about 134 operations to about
48 ((440 − 23.3) / (9.478 − 0.716)). Measure it with the existing `runtime.*` and `fallback.*`
benchmarks. Batching or compiling larger units of work only matters
after that: the fixed boundary cost (~140 ns) would still keep 1–5
operation functions slower than Python.

**Reliability caveats.** This run comes from a shared 4-vCPU KVM guest
with no `pyperf system tune`. Almost every benchmark trips pyperf's
stability checks (std dev 10–25% of the mean; see *Stability*), and light
interactive work, such as git commands, ran on the machine during the
runtime suite. The commit metadata differs between runs only because
documentation was committed mid-run; the measured code was identical.
The conclusions above rest on differences much larger than this noise:
25× ratios, layer shares stable across all six workloads, and a 13×
per-operation gap. The `ops_100` and `ops_250` results are the closest to
parity, and the repeat run confirmed both. Individual nanosecond values
should not be compared across machines or treated as precise.
