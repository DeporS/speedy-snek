# speedy-snek benchmark results

> **This is one machine-specific baseline, not a universal performance claim.** Absolute numbers depend on the CPU, operating system, Python build and system load. Regenerate it on your own machine with the commands below; see `benchmarks/README.md`.

## Environment

|  |  |
|---|---|
| **Date** | 2026-10-16 11:46:08 to 2026-10-16 12:08:04 |
| **Report generated** | 2026-10-16 12:12 UTC |
| **Operating system** | Linux-6.18.44-fc-v139-x86_64-with-glibc2.39 |
| **CPU** | Intel(R) Xeon(R) Processor @ 2.80GHz |
| **Architecture** | x86_64 |
| **Python** | cpython 3.14.6 (64-bit) |
| **CPython's own JIT** | disabled |
| **Rust** | rustc 1.97.0 (2d8144b78 2026-07-07) |
| **speedysnek native build** | release |
| **speedysnek commit** | 201edf8, 201edf8-dirty, fac74be (differs between runs) |
| **`@snekit` warmup calls** | 8 |

### Benchmark command

```sh
maturin develop --release
python benchmarks/bench_runtime.py -o results/runtime.json
python benchmarks/bench_compilation.py -o results/compilation.json
# repeat check, not part of the tables above:
python benchmarks/bench_runtime.py --affinity=3 --filter '^runtime\.ops_(100|250)\.' -o results/repeat.json
```

### pyperf metadata relevant to interpretation

- **Samples per benchmark:** 20 worker processes × 3 values (+ 1 warmup)
- **pyperf version:** 2.10.0
- **Timer:** clock_gettime(CLOCK_MONOTONIC), resolution: 1.00 ns
- **Logical CPUs:** 4
- **CPU configuration:** idle:none
- **CPU frequency:** 0-3=2800 MHz
- **ASLR:** Full randomization
- **Python compiler:** Clang 22.1.3

## Key measurements

- Warmed `@snekit` is faster than plain Python for `ops_250` (1 of 6 workloads).
- `@snekit` / Python ranges from 0.59× to 25.21×.
- The raw native call alone is faster than the Python function for `ops_020`, `ops_050`, `ops_100`, `ops_250`.
- For `ops_001`, the largest part of a `@snekit` call is the `CompiledI64` Python guards: 710.0 ns of 1,153.9 ns (62%).
- For `ops_250`, the largest part of a `@snekit` call is the `CompiledI64` Python guards: 778.7 ns of 1,410.2 ns (55%).
- Marginal cost per operation: Python 9.48 ns, raw native 0.716 ns. Fixed cost per call: Python 23.3 ns, raw native 140.2 ns, `@snekit` 1,157.8 ns.
- Break-even: `ops_250` ≈ 8,299 calls.
- Fallback overhead ranges from 223.5 ns (`disabled`) to 2,240.4 ns (`overflow`) per call.

## Steady-state runtime

Nanoseconds per call (mean ± std dev). `@snekit` was warmed up and verified to run native code before timing. Ratios are layer / Python: **below 1.0 means the JIT path is faster, above 1.0 means it is slower.**

| Workload | Python | `@snekit` | `CompiledI64` | raw native | `@snekit` / Python | raw native / Python |
|---|--:|--:|--:|--:|--:|--:|
| `ops_001` | 45.8 ± 10.4 | 1,153.9 ± 155.6 | 851.6 ± 152.8 | 141.6 ± 24.2 | 25.21× (slower) | 3.09× (slower) |
| `ops_005` | 84.9 ± 21.6 | 1,141.6 ± 157.3 | 903.9 ± 170.9 | 147.1 ± 29.0 | 13.44× (slower) | 1.73× (slower) |
| `ops_020` | 210.3 ± 36.1 | 1,240.7 ± 224.5 | 834.2 ± 79.9 | 152.8 ± 19.5 | 5.90× (slower) | 0.73× (faster) |
| `ops_050` | 493.2 ± 75.3 | 1,173.0 ± 171.1 | 868.1 ± 89.1 | 173.3 ± 25.9 | 2.38× (slower) | 0.35× (faster) |
| `ops_100` | 935.2 ± 131.9 | 1,248.0 ± 164.9 | 972.8 ± 182.1 | 211.9 ± 49.8 | 1.33× (slower) | 0.23× (faster) |
| `ops_250` | 2,407.9 ± 396.7 | 1,410.2 ± 174.7 | 1,098.7 ± 158.1 | 320.0 ± 58.4 | 0.59× (faster) | 0.13× (faster) |

References:

- empty statement (timing loop residue): 1.7 ± 0.2 ns
- Python call of `noop(a, b, c, d)`, no arithmetic: 40.6 ± 5.3 ns

## Where a warmed `@snekit` call spends its time

Nanoseconds per call, as differences between adjacent layers:

- **decorator wrapper** = `@snekit` − `CompiledI64`: the closure, `call_count`, the specialization and `__code__` checks, the `*args` and `**kwargs` packing and the `try`.
- **`CompiledI64` Python guards** = `CompiledI64` − raw native: the Python-level `__call__`, its arity, exact-`int` and i64 range checks, and the status check on the result.
- **raw native call** = `I64Function.call(args)`: method call through PyO3, tuple and int conversion, a `Vec` allocation, the machine code itself, and building the `(status, value)` result tuple.

| Workload | `@snekit` total | decorator wrapper | `CompiledI64` Python guards | raw native call |
|---|--:|--:|--:|--:|
| `ops_001` | 1,153.9 | 302.3 (26%) | 710.0 (62%) | 141.6 (12%) |
| `ops_005` | 1,141.6 | 237.7 (21%) | 756.8 (66%) | 147.1 (13%) |
| `ops_020` | 1,240.7 | 406.5 (33%) | 681.4 (55%) | 152.8 (12%) |
| `ops_050` | 1,173.0 | 304.8 (26%) | 694.9 (59%) | 173.3 (15%) |
| `ops_100` | 1,248.0 | 275.2 (22%) | 760.9 (61%) | 211.9 (17%) |
| `ops_250` | 1,410.2 | 311.5 (22%) | 778.7 (55%) | 320.0 (23%) |

## Fixed cost versus cost per operation

A least-squares line `time = fixed + per_op × operations` through all runtime workloads of each layer. The fixed part is what every call pays regardless of size; the slope is the marginal cost of one more `+`, `-` or `*`.

| Layer | fixed cost per call (ns) | cost per operation (ns) |
|---|--:|--:|
| Python | 23.3 | 9.478 |
| `@snekit` | 1,157.8 | 0.987 |
| `CompiledI64` | 851.1 | 0.992 |
| raw native | 140.2 | 0.716 |

## Fallback and wrapper overhead

Nanoseconds per call. Each `@snekit` row runs the original function; its Python baseline is an undecorated copy of that same function called with the same arguments, so the overhead column is the cost of the wrapper alone.

| Case | What happens | Python | `@snekit` | overhead | `@snekit` / Python |
|---|--:|--:|--:|--:|--:|
| `cold` | warmup call: counted, then Python | 48.3 ± 10.3 | 344.9 ± 54.4 | 296.5 | 7.14× (slower) |
| `disabled` | rejected function (`//`), after it was disabled | 53.6 ± 8.7 | 277.1 ± 47.3 | 223.5 | 5.17× (slower) |
| `float_args` | compiled; `float` arguments fail the exact-`int` guard | 54.0 ± 11.3 | 1,705.8 ± 283.7 | 1,651.8 | 31.57× (slower) |
| `big_int_args` | compiled; an `int` outside i64 fails the range guard | 106.3 ± 12.7 | 1,812.5 ± 242.9 | 1,706.2 | 17.05× (slower) |
| `overflow` | compiled; native `*` overflows, Python recomputes | 122.4 ± 14.4 | 2,362.8 ± 193.7 | 2,240.4 | 19.31× (slower) |
| `keyword_args` | compiled; keyword arguments skip native code | 59.2 ± 9.5 | 434.9 ± 70.2 | 375.7 | 7.35× (slower) |
| `default_args` | compiled; an omitted default skips native code | 48.6 ± 5.3 | 311.8 ± 33.8 | 263.2 | 6.41× (slower) |

## Compilation and warmup

Microseconds per compilation (mean ± std dev). Nothing is reused between iterations: every backend iteration builds a new Cranelift module and every lifecycle iteration a new wrapper.

- **frontend**: `analyze(func)`, bytecode to normalized IR.
- **native compilation (from IR)**: `_compiler.compile_i64`: validation, serialization, Cranelift and `CompiledI64`.
- **codegen only**: `_native.compile_i64` on an already serialized program.
- **full lifecycle**: a new `@snekit` wrapper through 8 warmup calls and the compiling call.

| Workload | frontend | native compilation (from IR) | codegen only | full lifecycle |
|---|--:|--:|--:|--:|
| `ops_001` | 28.7 ± 2.4 | 148.2 ± 14.5 | 140.2 ± 18.5 | 256.4 ± 29.6 |
| `ops_005` | 72.4 ± 8.2 | 239.5 ± 21.7 | 218.5 ± 26.8 | 400.3 ± 21.9 |
| `ops_020` | 225.9 ± 40.2 | 552.4 ± 81.0 | 510.2 ± 83.3 | 967.1 ± 179.3 |
| `ops_050` | 504.7 ± 86.5 | 1,159.2 ± 197.5 | 965.3 ± 109.1 | 1,663.8 ± 160.0 |
| `ops_100` | 1,047.5 ± 161.3 | 1,941.1 ± 179.4 | 1,841.9 ± 292.5 | 3,288.1 ± 392.9 |
| `ops_250` | 2,556.7 ± 379.8 | 4,905.1 ± 1,056.9 | 4,189.8 ± 675.0 | 8,279.2 ± 1,487.7 |

- Frontend rejection of an unsupported function: 90.5 ± 9.2 µs
- Full lifecycle of an unsupported function (ends disabled): 108.5 ± 16.5 µs

### Cold import (fresh process)

| Command | wall time (ms) | above bare startup (ms) |
|---|--:|--:|
| `python -c pass` | 17.8 ± 1.5 | — |
| `python -c 'import speedysnek'` | 29.9 ± 3.0 | 12.1 |
| `python -c 'import speedysnek._compiler'` | 51.9 ± 5.0 | 34.2 |

## Break-even

`full lifecycle cost / (Python time per call − @snekit time per call)`, computed in seconds. The lifecycle cost includes the warmup calls, so this slightly overestimates the break-even point. It is only computed when the warmed `@snekit` call is faster than Python.

| Workload | lifecycle cost | Python − `@snekit` per call | break-even |
|---|--:|--:|--:|
| `ops_001` | 256.4 µs | -1.1 µs | No break-even with the current hot path. |
| `ops_005` | 400.3 µs | -1.1 µs | No break-even with the current hot path. |
| `ops_020` | 967.1 µs | -1.0 µs | No break-even with the current hot path. |
| `ops_050` | 1.7 ms | -679.8 ns | No break-even with the current hot path. |
| `ops_100` | 3.3 ms | -312.8 ns | No break-even with the current hot path. |
| `ops_250` | 8.3 ms | 997.7 ns | ≈ 8,299 calls |

## Stability

pyperf's stability checks flagged these benchmarks; treat their numbers with extra caution:

- reference.loop: std dev is 12% of the mean
- reference.python_noop: std dev is 13% of the mean
- runtime.ops_001.python: std dev is 23% of the mean
- runtime.ops_001.python: maximum is +115% from the mean
- runtime.ops_001.snekit: std dev is 13% of the mean
- runtime.ops_001.compiled_i64: std dev is 18% of the mean
- runtime.ops_001.compiled_i64: maximum is +97% from the mean
- runtime.ops_001.native_call: std dev is 17% of the mean
- runtime.ops_001.native_call: maximum is +85% from the mean
- runtime.ops_005.python: std dev is 25% of the mean
- runtime.ops_005.python: maximum is +90% from the mean
- runtime.ops_005.snekit: std dev is 14% of the mean
- runtime.ops_005.snekit: maximum is +62% from the mean
- runtime.ops_005.compiled_i64: std dev is 19% of the mean
- runtime.ops_005.compiled_i64: maximum is +67% from the mean
- runtime.ops_005.native_call: std dev is 20% of the mean
- runtime.ops_005.native_call: maximum is +85% from the mean
- runtime.ops_020.python: std dev is 17% of the mean
- runtime.ops_020.python: maximum is +58% from the mean
- runtime.ops_020.snekit: std dev is 18% of the mean
- runtime.ops_020.snekit: maximum is +57% from the mean
- runtime.ops_020.compiled_i64: not enough processes for a stable result (20 run, 725 suggested)
- runtime.ops_020.native_call: std dev is 13% of the mean
- runtime.ops_050.python: std dev is 15% of the mean
- runtime.ops_050.python: maximum is +57% from the mean
- runtime.ops_050.snekit: std dev is 15% of the mean
- runtime.ops_050.snekit: maximum is +69% from the mean
- runtime.ops_050.compiled_i64: std dev is 10% of the mean
- runtime.ops_050.native_call: std dev is 15% of the mean
- runtime.ops_100.python: std dev is 14% of the mean
- runtime.ops_100.snekit: std dev is 13% of the mean
- runtime.ops_100.compiled_i64: std dev is 19% of the mean
- runtime.ops_100.compiled_i64: maximum is +60% from the mean
- runtime.ops_100.native_call: std dev is 23% of the mean
- runtime.ops_100.native_call: maximum is +70% from the mean
- runtime.ops_250.python: std dev is 16% of the mean
- runtime.ops_250.python: maximum is +69% from the mean
- runtime.ops_250.snekit: std dev is 12% of the mean
- runtime.ops_250.compiled_i64: std dev is 14% of the mean
- runtime.ops_250.native_call: std dev is 18% of the mean
- runtime.ops_250.native_call: maximum is +59% from the mean
- fallback.cold.python: std dev is 21% of the mean
- fallback.cold.python: maximum is +67% from the mean
- fallback.cold.snekit: std dev is 16% of the mean
- fallback.cold.snekit: maximum is +57% from the mean
- fallback.disabled.python: std dev is 16% of the mean
- fallback.disabled.python: maximum is +51% from the mean
- fallback.disabled.snekit: std dev is 17% of the mean
- fallback.disabled.snekit: maximum is +60% from the mean
- fallback.float_args.python: std dev is 21% of the mean
- fallback.float_args.python: maximum is +61% from the mean
- fallback.float_args.snekit: std dev is 17% of the mean
- fallback.float_args.snekit: maximum is +61% from the mean
- fallback.big_int_args.python: std dev is 12% of the mean
- fallback.big_int_args.python: maximum is +51% from the mean
- fallback.big_int_args.snekit: std dev is 13% of the mean
- fallback.overflow.python: std dev is 12% of the mean
- fallback.overflow.snekit: not enough processes for a stable result (20 run, 314 suggested)
- fallback.keyword_args.python: std dev is 16% of the mean
- fallback.keyword_args.python: maximum is +66% from the mean
- fallback.keyword_args.snekit: std dev is 16% of the mean
- fallback.default_args.python: std dev is 11% of the mean
- fallback.default_args.snekit: std dev is 11% of the mean
- compile.frontend.ops_001: not enough processes for a stable result (20 run, 484 suggested)
- compile.backend.ops_001: not enough processes for a stable result (20 run, 842 suggested)
- compile.codegen.ops_001: std dev is 13% of the mean
- compile.lifecycle.ops_001: std dev is 12% of the mean
- compile.frontend.ops_005: std dev is 11% of the mean
- compile.backend.ops_005: not enough processes for a stable result (20 run, 435 suggested)
- compile.codegen.ops_005: std dev is 12% of the mean
- compile.codegen.ops_005: maximum is +58% from the mean
- compile.lifecycle.ops_005: not enough processes for a stable result (20 run, 213 suggested)
- compile.frontend.ops_020: std dev is 18% of the mean
- compile.frontend.ops_020: maximum is +70% from the mean
- compile.backend.ops_020: std dev is 15% of the mean
- compile.backend.ops_020: maximum is +52% from the mean
- compile.codegen.ops_020: std dev is 16% of the mean
- compile.lifecycle.ops_020: std dev is 19% of the mean
- compile.frontend.ops_050: std dev is 17% of the mean
- compile.frontend.ops_050: maximum is +73% from the mean
- compile.backend.ops_050: std dev is 17% of the mean
- compile.backend.ops_050: maximum is +61% from the mean
- compile.codegen.ops_050: std dev is 11% of the mean
- compile.lifecycle.ops_050: not enough processes for a stable result (20 run, 656 suggested)
- compile.frontend.ops_100: std dev is 15% of the mean
- compile.frontend.ops_100: maximum is +53% from the mean
- compile.backend.ops_100: not enough processes for a stable result (20 run, 788 suggested)
- compile.codegen.ops_100: std dev is 16% of the mean
- compile.codegen.ops_100: maximum is +56% from the mean
- compile.lifecycle.ops_100: std dev is 12% of the mean
- compile.frontend.ops_250: std dev is 15% of the mean
- compile.frontend.ops_250: maximum is +65% from the mean
- compile.backend.ops_250: std dev is 22% of the mean
- compile.backend.ops_250: maximum is +78% from the mean
- compile.codegen.ops_250: std dev is 16% of the mean
- compile.codegen.ops_250: maximum is +59% from the mean
- compile.lifecycle.ops_250: std dev is 18% of the mean
- compile.lifecycle.ops_250: maximum is +54% from the mean
- compile.reject.frontend: std dev is 10% of the mean
- compile.reject.lifecycle: std dev is 15% of the mean
- import.python: not enough processes for a stable result (20 run, 496 suggested)
- import.speedysnek: not enough processes for a stable result (20 run, 620 suggested)
- import.speedysnek_backend: not enough processes for a stable result (20 run, 635 suggested)

## Interpretation

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
