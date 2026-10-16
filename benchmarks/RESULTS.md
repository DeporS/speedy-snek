# speedy-snek benchmark results

> **This is one machine-specific baseline, not a universal performance claim.** Absolute numbers depend on the CPU, operating system, Python build and system load. Regenerate it on your own machine with the commands below; see `benchmarks/README.md`.

## Environment

|  |  |
|---|---|
| **Date** | 2026-10-16 12:51:09 to 2026-10-16 13:34:36 |
| **Report generated** | 2026-10-16 13:56 UTC |
| **Operating system** | Linux-6.18.44-fc-v139-x86_64-with-glibc2.39 |
| **CPU** | Intel(R) Xeon(R) Processor @ 2.80GHz |
| **Architecture** | x86_64 |
| **Python** | cpython 3.14.6 (64-bit) |
| **CPython's own JIT** | disabled |
| **Rust** | rustc 1.97.0 (2d8144b78 2026-07-07) |
| **speedysnek native build** | release |
| **speedysnek commit** | 9810f2f |
| **`@snekit` warmup calls** | 8 |

### Benchmark command

```sh
maturin develop --release
python benchmarks/bench_runtime.py -o results/runtime.json
python benchmarks/bench_compilation.py -o results/compilation.json
# controlled comparison: the previous code (3df7644), same commands, run right after in the same session
# pinned repeat, two interleaved rounds per version, not part of the tables above:
python benchmarks/bench_runtime.py --affinity=3 -p 10 --filter '^(runtime\.ops_(020|050)\.(python|snekit)|fallback\.(cold|disabled|keyword_args)\.(python|snekit))$' --append results/repeat.json
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

- Warmed `@snekit` is faster than plain Python for `ops_050`, `ops_100`, `ops_250` (3 of 6 workloads).
- `@snekit` / Python ranges from 0.16× to 7.66×.
- The native fast call alone is faster than the Python function for `ops_005`, `ops_020`, `ops_050`, `ops_100`, `ops_250`.
- For `ops_001`, the largest part of a `@snekit` call is the decorator overhead: 426.6 ns of 544.0 ns (78%).
- For `ops_250`, the largest part of a `@snekit` call is the decorator overhead: 406.8 ns of 724.7 ns (56%).
- Marginal cost per operation: Python 17.76 ns, native fast call 0.835 ns. Fixed cost per call: Python 2.1 ns, native fast call 112.5 ns, `@snekit` 497.7 ns.
- Estimated `@snekit` crossover: ≈ 29 operations.
- Break-even: `ops_050` ≈ 7,236 calls, `ops_100` ≈ 4,510 calls, `ops_250` ≈ 3,157 calls.
- Fallback overhead ranges from 314.7 ns (`disabled`) to 525.1 ns (`keyword_args`) per call.

## Comparison: same-session control run of the previous code (3df7644)

Previous results: commit 3df7644, measured 2026-10-16 13:09:21 to 2026-10-16 13:45:54. Change is current / previous: **below 1.0× is faster or fewer calls.** Both runs are means with 10–25% noise; compare large changes only. The plain Python rows do not depend on speedy-snek and show how much the machine itself differs between the runs.

| Measurement | previous | current | change |
|---|--:|--:|--:|
| plain Python call, `noop(a, b, c, d)` | 47.9 ns | 52.1 ns | 1.09× (+9%) |
| plain Python, `ops_001` | 63.1 ns | 71.0 ns | 1.13× (+13%) |
| plain Python, `ops_250` | 4.2 µs | 4.5 µs | 1.07× (+7%) |
| warmed `@snekit`, `ops_001` | 1.4 µs | 544.0 ns | 0.38× (−62%) |
| warmed `@snekit`, `ops_005` | 1.5 µs | 462.7 ns | 0.31× (−69%) |
| warmed `@snekit`, `ops_020` | 1.5 µs | 516.8 ns | 0.35× (−65%) |
| warmed `@snekit`, `ops_050` | 1.5 µs | 529.8 ns | 0.36× (−64%) |
| warmed `@snekit`, `ops_100` | 1.6 µs | 592.7 ns | 0.36× (−64%) |
| warmed `@snekit`, `ops_250` | 1.7 µs | 724.7 ns | 0.42× (−58%) |
| warmed `@snekit` fixed cost (fit) | 1.4 µs | 497.7 ns | 0.34× (−66%) |
| warmed `@snekit` cost per operation (fit) | 1.2 ns | 0.9 ns | 0.74× (−26%) |
| fallback `cold`, `@snekit` call | 419.7 ns | 505.2 ns | 1.20× (+20%) |
| fallback `disabled`, `@snekit` call | 317.4 ns | 390.3 ns | 1.23× (+23%) |
| fallback `float_args`, `@snekit` call | 2.0 µs | 547.1 ns | 0.27× (−73%) |
| fallback `big_int_args`, `@snekit` call | 2.2 µs | 668.1 ns | 0.30× (−70%) |
| fallback `overflow`, `@snekit` call | 3.0 µs | 650.0 ns | 0.22× (−78%) |
| fallback `keyword_args`, `@snekit` call | 612.9 ns | 608.8 ns | 0.99× (−1%) |
| fallback `default_args`, `@snekit` call | 423.3 ns | 417.3 ns | 0.99× (−1%) |
| estimated `@snekit` crossover | ≈ 94 operations | ≈ 29 operations | 0.31× (−69%) |
| break-even, `ops_001` | No break-even with the current hot path. | No break-even with the current hot path. | — |
| break-even, `ops_005` | No break-even with the current hot path. | No break-even with the current hot path. | — |
| break-even, `ops_020` | No break-even with the current hot path. | No break-even with the current hot path. | — |
| break-even, `ops_050` | No break-even with the current hot path. | ≈ 7,236 calls | — |
| break-even, `ops_100` | No break-even with the current hot path. | ≈ 4,510 calls | — |
| break-even, `ops_250` | ≈ 4,226 calls | ≈ 3,157 calls | 0.75× (−25%) |

## Comparison: Stage 6 baseline (historical)

Previous results: commit 201edf8, 201edf8-dirty, fac74be (differs between runs), measured 2026-10-16 11:46:08 to 2026-10-16 12:08:04. Change is current / previous: **below 1.0× is faster or fewer calls.** Both runs are means with 10–25% noise; compare large changes only. The plain Python rows do not depend on speedy-snek and show how much the machine itself differs between the runs.

| Measurement | previous | current | change |
|---|--:|--:|--:|
| plain Python call, `noop(a, b, c, d)` | 40.6 ns | 52.1 ns | 1.28× (+28%) |
| plain Python, `ops_001` | 45.8 ns | 71.0 ns | 1.55× (+55%) |
| plain Python, `ops_250` | 2.4 µs | 4.5 µs | 1.87× (+87%) |
| warmed `@snekit`, `ops_001` | 1.2 µs | 544.0 ns | 0.47× (−53%) |
| warmed `@snekit`, `ops_005` | 1.1 µs | 462.7 ns | 0.41× (−59%) |
| warmed `@snekit`, `ops_020` | 1.2 µs | 516.8 ns | 0.42× (−58%) |
| warmed `@snekit`, `ops_050` | 1.2 µs | 529.8 ns | 0.45× (−55%) |
| warmed `@snekit`, `ops_100` | 1.2 µs | 592.7 ns | 0.47× (−53%) |
| warmed `@snekit`, `ops_250` | 1.4 µs | 724.7 ns | 0.51× (−49%) |
| warmed `@snekit` fixed cost (fit) | 1.2 µs | 497.7 ns | 0.43× (−57%) |
| warmed `@snekit` cost per operation (fit) | 1.0 ns | 0.9 ns | 0.92× (−8%) |
| fallback `cold`, `@snekit` call | 344.9 ns | 505.2 ns | 1.46× (+46%) |
| fallback `disabled`, `@snekit` call | 277.1 ns | 390.3 ns | 1.41× (+41%) |
| fallback `float_args`, `@snekit` call | 1.7 µs | 547.1 ns | 0.32× (−68%) |
| fallback `big_int_args`, `@snekit` call | 1.8 µs | 668.1 ns | 0.37× (−63%) |
| fallback `overflow`, `@snekit` call | 2.4 µs | 650.0 ns | 0.28× (−72%) |
| fallback `keyword_args`, `@snekit` call | 434.9 ns | 608.8 ns | 1.40× (+40%) |
| fallback `default_args`, `@snekit` call | 311.8 ns | 417.3 ns | 1.34× (+34%) |
| estimated `@snekit` crossover | ≈ 134 operations | ≈ 29 operations | 0.22× (−78%) |
| break-even, `ops_001` | No break-even with the current hot path. | No break-even with the current hot path. | — |
| break-even, `ops_005` | No break-even with the current hot path. | No break-even with the current hot path. | — |
| break-even, `ops_020` | No break-even with the current hot path. | No break-even with the current hot path. | — |
| break-even, `ops_050` | No break-even with the current hot path. | ≈ 7,236 calls | — |
| break-even, `ops_100` | No break-even with the current hot path. | ≈ 4,510 calls | — |
| break-even, `ops_250` | ≈ 8,299 calls | ≈ 3,157 calls | 0.38× (−62%) |

## Steady-state runtime

Nanoseconds per call (mean ± std dev). `@snekit` was warmed up and verified to make exactly one native fast call, and no `CompiledI64.__call__`, before timing. Ratios are layer / Python: **below 1.0 means the JIT path is faster, above 1.0 means it is slower.**

- **native fast call**: `I64Function.try_call(args)` on a prebuilt tuple, the one native call a warmed `@snekit` call makes.
- **diagnostic** layers: `CompiledI64.__call__` and the `I64Function.call` it wraps. They report *why* a call misses and are **not part of the warmed `@snekit` path**.

| Workload | Python | `@snekit` | native fast call | `CompiledI64` (diagnostic) | `I64Function.call` (diagnostic) | `@snekit` / Python | fast call / Python |
|---|--:|--:|--:|--:|--:|--:|--:|
| `ops_001` | 71.0 ± 13.7 | 544.0 ± 124.6 | 117.4 ± 28.1 | 1,073.3 ± 162.4 | 187.9 ± 43.2 | 7.66× (slower) | 1.65× (slower) |
| `ops_005` | 128.1 ± 27.4 | 462.7 ± 105.1 | 118.9 ± 32.3 | 1,128.1 ± 191.2 | 211.3 ± 59.7 | 3.61× (slower) | 0.93× (faster) |
| `ops_020` | 364.9 ± 74.2 | 516.8 ± 123.7 | 123.0 ± 28.3 | 1,348.6 ± 295.0 | 229.9 ± 55.0 | 1.42× (slower) | 0.34× (faster) |
| `ops_050` | 864.7 ± 206.2 | 529.8 ± 121.1 | 142.0 ± 29.2 | 1,296.7 ± 257.7 | 274.6 ± 77.3 | 0.61× (faster) | 0.16× (faster) |
| `ops_100` | 1,655.2 ± 316.2 | 592.7 ± 163.0 | 211.2 ± 55.4 | 1,272.3 ± 275.2 | 286.7 ± 59.1 | 0.36× (faster) | 0.13× (faster) |
| `ops_250` | 4,495.5 ± 920.3 | 724.7 ± 173.8 | 317.9 ± 63.8 | 1,349.4 ± 300.9 | 406.0 ± 99.5 | 0.16× (faster) | 0.07× (faster) |

References:

- empty statement (timing loop residue): 2.4 ± 0.5 ns
- Python call of `noop(a, b, c, d)`, no arithmetic: 52.1 ± 11.4 ns

## Where a warmed `@snekit` call spends its time

Nanoseconds per call. A warmed call is the decorator wrapper around exactly one native fast call, so:

- **decorator overhead** = `@snekit` − native fast call: the closure call with `*args`/`**kwargs` packing, `call_count`, the specialization, `__code__`, keyword and arity checks, and the `None` check on the result.
- **native fast call** = `I64Function.try_call(args)`: the method call through PyO3, one exact-`int` check and i64 conversion per argument, the machine code with its overflow branches, and the result `int`.

| Workload | `@snekit` total | decorator overhead | native fast call |
|---|--:|--:|--:|
| `ops_001` | 544.0 | 426.6 (78%) | 117.4 (22%) |
| `ops_005` | 462.7 | 343.8 (74%) | 118.9 (26%) |
| `ops_020` | 516.8 | 393.9 (76%) | 123.0 (24%) |
| `ops_050` | 529.8 | 387.8 (73%) | 142.0 (27%) |
| `ops_100` | 592.7 | 381.5 (64%) | 211.2 (36%) |
| `ops_250` | 724.7 | 406.8 (56%) | 317.9 (44%) |

### Diagnostic path (not part of warmed `@snekit`)

`CompiledI64.__call__` still exists for direct compiler users and tests: it raises `NativeGuardFailure` with the exact reason for a miss. It checks the arguments in Python, then `I64Function.call` checks them again in Rust and returns `(status, value)`. These numbers are **not a component of the `@snekit` total above**; they show what the decorator no longer pays.

| Workload | `CompiledI64` total | Python guards in `CompiledI64.__call__` | `I64Function.call` |
|---|--:|--:|--:|
| `ops_001` | 1,073.3 | 885.3 (82%) | 187.9 (18%) |
| `ops_005` | 1,128.1 | 916.7 (81%) | 211.3 (19%) |
| `ops_020` | 1,348.6 | 1,118.6 (83%) | 229.9 (17%) |
| `ops_050` | 1,296.7 | 1,022.1 (79%) | 274.6 (21%) |
| `ops_100` | 1,272.3 | 985.6 (77%) | 286.7 (23%) |
| `ops_250` | 1,349.4 | 943.4 (70%) | 406.0 (30%) |

## Fixed cost versus cost per operation

A least-squares line `time = fixed + per_op × operations` through all runtime workloads of each layer. The fixed part is what every call pays regardless of size; the slope is the marginal cost of one more `+`, `-` or `*`.

| Layer | fixed cost per call (ns) | cost per operation (ns) |
|---|--:|--:|
| Python | 2.1 | 17.762 |
| `@snekit` | 497.7 | 0.903 |
| native fast call | 112.5 | 0.835 |
| `CompiledI64` (diagnostic) | 1,193.1 | 0.727 |
| `I64Function.call` (diagnostic) | 208.9 | 0.805 |

Estimated crossover, where the fitted line meets Python's (`(fixed − Python fixed) / (Python per op − per op)`):

- `@snekit`: ≈ 29 operations
- native fast call: ≈ 7 operations

## Fallback and wrapper overhead

Nanoseconds per call. Each `@snekit` row runs the original function; its Python baseline is an undecorated copy of that same function called with the same arguments, so the overhead column is the cost of the wrapper alone. A guard miss is reported by the native fast call returning `None`, not by an exception. **native miss** times that native call alone (part of the overhead); it is not attempted when the wrapper falls back before any native code.

| Case | What happens | Python | `@snekit` | overhead | native miss | `@snekit` / Python |
|---|--:|--:|--:|--:|--:|--:|
| `cold` | warmup call: counted, then Python | 60.1 ± 13.0 | 505.2 ± 115.5 | 445.1 | not attempted | 8.41× (slower) |
| `disabled` | rejected function (`//`), after it was disabled | 75.6 ± 16.1 | 390.3 ± 91.3 | 314.7 | not attempted | 5.16× (slower) |
| `float_args` | compiled; `float` arguments miss the exact-`int` guard | 75.1 ± 19.2 | 547.1 ± 113.5 | 472.1 | 74.3 ± 14.2 | 7.29× (slower) |
| `big_int_args` | compiled; an `int` outside i64 misses the range guard | 153.9 ± 29.8 | 668.1 ± 154.0 | 514.2 | 90.6 ± 18.5 | 4.34× (slower) |
| `overflow` | compiled; native `*` overflows, Python recomputes | 217.9 ± 53.4 | 650.0 ± 120.9 | 432.0 | 105.7 ± 26.1 | 2.98× (slower) |
| `keyword_args` | compiled; keyword arguments skip native code | 83.7 ± 13.5 | 608.8 ± 126.7 | 525.1 | not attempted | 7.27× (slower) |
| `default_args` | compiled; an omitted default skips native code | 65.3 ± 11.9 | 417.3 ± 87.3 | 352.1 | not attempted | 6.39× (slower) |

## Compilation and warmup

Microseconds per compilation (mean ± std dev). Nothing is reused between iterations: every backend iteration builds a new Cranelift module and every lifecycle iteration a new wrapper.

- **frontend**: `analyze(func)`, bytecode to normalized IR.
- **native compilation (from IR)**: `_compiler.compile_i64`: validation, serialization, Cranelift and `CompiledI64`.
- **codegen only**: `_native.compile_i64` on an already serialized program.
- **full lifecycle**: a new `@snekit` wrapper through 8 warmup calls and the compiling call.

| Workload | frontend | native compilation (from IR) | codegen only | full lifecycle |
|---|--:|--:|--:|--:|
| `ops_001` | 49.1 ± 9.2 | 225.7 ± 31.6 | 181.2 ± 32.5 | 368.9 ± 50.5 |
| `ops_005` | 104.4 ± 21.1 | 370.8 ± 45.7 | 332.1 ± 50.2 | 603.1 ± 81.7 |
| `ops_020` | 292.9 ± 52.3 | 809.8 ± 127.4 | 672.0 ± 91.5 | 1,399.4 ± 230.4 |
| `ops_050` | 795.6 ± 186.2 | 1,417.7 ± 240.0 | 1,455.3 ± 321.1 | 2,423.2 ± 535.5 |
| `ops_100` | 1,454.2 ± 321.9 | 2,732.1 ± 638.5 | 2,762.1 ± 728.0 | 4,791.3 ± 1,004.4 |
| `ops_250` | 3,539.1 ± 685.2 | 6,470.9 ± 1,345.3 | 6,126.2 ± 1,473.9 | 11,901.7 ± 2,565.7 |

- Frontend rejection of an unsupported function: 119.3 ± 19.7 µs
- Full lifecycle of an unsupported function (ends disabled): 149.6 ± 34.5 µs

### Cold import (fresh process)

| Command | wall time (ms) | above bare startup (ms) |
|---|--:|--:|
| `python -c pass` | 29.6 ± 2.5 | — |
| `python -c 'import speedysnek'` | 42.6 ± 4.8 | 13.0 |
| `python -c 'import speedysnek._compiler'` | 72.3 ± 9.7 | 42.8 |

## Break-even

`full lifecycle cost / (Python time per call − @snekit time per call)`, computed in seconds. The lifecycle cost includes the warmup calls, so this slightly overestimates the break-even point. It is only computed when the warmed `@snekit` call is faster than Python.

| Workload | lifecycle cost | Python − `@snekit` per call | break-even |
|---|--:|--:|--:|
| `ops_001` | 368.9 µs | -473.0 ns | No break-even with the current hot path. |
| `ops_005` | 603.1 µs | -334.6 ns | No break-even with the current hot path. |
| `ops_020` | 1.4 ms | -152.0 ns | No break-even with the current hot path. |
| `ops_050` | 2.4 ms | 334.9 ns | ≈ 7,236 calls |
| `ops_100` | 4.8 ms | 1.1 µs | ≈ 4,510 calls |
| `ops_250` | 11.9 ms | 3.8 µs | ≈ 3,157 calls |

## Stability

pyperf's stability checks flagged these benchmarks; treat their numbers with extra caution:

- reference.loop: std dev is 20% of the mean
- reference.python_noop: std dev is 22% of the mean
- reference.python_noop: maximum is +51% from the mean
- runtime.ops_001.python: std dev is 19% of the mean
- runtime.ops_001.snekit: std dev is 23% of the mean
- runtime.ops_001.snekit: maximum is +51% from the mean
- runtime.ops_001.fast_call: std dev is 24% of the mean
- runtime.ops_001.fast_call: maximum is +68% from the mean
- runtime.ops_001.compiled_i64: std dev is 15% of the mean
- runtime.ops_001.native_call: std dev is 23% of the mean
- runtime.ops_001.native_call: maximum is +67% from the mean
- runtime.ops_005.python: std dev is 21% of the mean
- runtime.ops_005.python: maximum is +53% from the mean
- runtime.ops_005.snekit: std dev is 23% of the mean
- runtime.ops_005.snekit: maximum is +67% from the mean
- runtime.ops_005.fast_call: std dev is 27% of the mean
- runtime.ops_005.fast_call: maximum is +78% from the mean
- runtime.ops_005.compiled_i64: std dev is 17% of the mean
- runtime.ops_005.compiled_i64: maximum is +54% from the mean
- runtime.ops_005.native_call: std dev is 28% of the mean
- runtime.ops_005.native_call: maximum is +94% from the mean
- runtime.ops_020.python: std dev is 20% of the mean
- runtime.ops_020.snekit: std dev is 24% of the mean
- runtime.ops_020.snekit: maximum is +68% from the mean
- runtime.ops_020.fast_call: std dev is 23% of the mean
- runtime.ops_020.fast_call: maximum is +61% from the mean
- runtime.ops_020.compiled_i64: std dev is 22% of the mean
- runtime.ops_020.compiled_i64: maximum is +75% from the mean
- runtime.ops_020.native_call: std dev is 24% of the mean
- runtime.ops_020.native_call: maximum is +73% from the mean
- runtime.ops_050.python: std dev is 24% of the mean
- runtime.ops_050.python: maximum is +56% from the mean
- runtime.ops_050.snekit: std dev is 23% of the mean
- runtime.ops_050.snekit: maximum is +51% from the mean
- runtime.ops_050.fast_call: std dev is 21% of the mean
- runtime.ops_050.fast_call: maximum is +68% from the mean
- runtime.ops_050.compiled_i64: std dev is 20% of the mean
- runtime.ops_050.native_call: std dev is 28% of the mean
- runtime.ops_050.native_call: maximum is +80% from the mean
- runtime.ops_100.python: std dev is 19% of the mean
- runtime.ops_100.snekit: std dev is 27% of the mean
- runtime.ops_100.snekit: maximum is +92% from the mean
- runtime.ops_100.fast_call: std dev is 26% of the mean
- runtime.ops_100.fast_call: maximum is +51% from the mean
- runtime.ops_100.compiled_i64: std dev is 22% of the mean
- runtime.ops_100.compiled_i64: maximum is +56% from the mean
- runtime.ops_100.native_call: std dev is 21% of the mean
- runtime.ops_100.native_call: maximum is +70% from the mean
- runtime.ops_250.python: std dev is 20% of the mean
- runtime.ops_250.snekit: std dev is 24% of the mean
- runtime.ops_250.snekit: maximum is +61% from the mean
- runtime.ops_250.fast_call: std dev is 20% of the mean
- runtime.ops_250.compiled_i64: std dev is 22% of the mean
- runtime.ops_250.native_call: std dev is 25% of the mean
- runtime.ops_250.native_call: maximum is +59% from the mean
- fallback.cold.python: std dev is 22% of the mean
- fallback.cold.python: maximum is +56% from the mean
- fallback.cold.snekit: std dev is 23% of the mean
- fallback.cold.snekit: maximum is +74% from the mean
- fallback.disabled.python: std dev is 21% of the mean
- fallback.disabled.snekit: std dev is 23% of the mean
- fallback.disabled.snekit: maximum is +72% from the mean
- fallback.float_args.python: std dev is 26% of the mean
- fallback.float_args.python: maximum is +64% from the mean
- fallback.float_args.snekit: std dev is 21% of the mean
- fallback.float_args.snekit: maximum is +66% from the mean
- fallback.float_args.fast_call: std dev is 19% of the mean
- fallback.big_int_args.python: std dev is 19% of the mean
- fallback.big_int_args.python: maximum is +56% from the mean
- fallback.big_int_args.snekit: std dev is 23% of the mean
- fallback.big_int_args.snekit: maximum is +69% from the mean
- fallback.big_int_args.fast_call: std dev is 20% of the mean
- fallback.overflow.python: std dev is 25% of the mean
- fallback.overflow.snekit: std dev is 19% of the mean
- fallback.overflow.snekit: maximum is +55% from the mean
- fallback.overflow.fast_call: std dev is 25% of the mean
- fallback.overflow.fast_call: maximum is +58% from the mean
- fallback.keyword_args.python: std dev is 16% of the mean
- fallback.keyword_args.snekit: std dev is 21% of the mean
- fallback.keyword_args.snekit: maximum is +51% from the mean
- fallback.default_args.python: std dev is 18% of the mean
- fallback.default_args.python: maximum is +61% from the mean
- fallback.default_args.snekit: std dev is 21% of the mean
- fallback.default_args.snekit: maximum is +57% from the mean
- compile.frontend.ops_001: std dev is 19% of the mean
- compile.backend.ops_001: std dev is 14% of the mean
- compile.codegen.ops_001: std dev is 18% of the mean
- compile.lifecycle.ops_001: std dev is 14% of the mean
- compile.frontend.ops_005: std dev is 20% of the mean
- compile.frontend.ops_005: maximum is +55% from the mean
- compile.backend.ops_005: std dev is 12% of the mean
- compile.codegen.ops_005: std dev is 15% of the mean
- compile.lifecycle.ops_005: std dev is 14% of the mean
- compile.frontend.ops_020: std dev is 18% of the mean
- compile.frontend.ops_020: maximum is +52% from the mean
- compile.backend.ops_020: std dev is 16% of the mean
- compile.codegen.ops_020: std dev is 14% of the mean
- compile.lifecycle.ops_020: std dev is 16% of the mean
- compile.frontend.ops_050: std dev is 23% of the mean
- compile.frontend.ops_050: maximum is +53% from the mean
- compile.backend.ops_050: std dev is 17% of the mean
- compile.codegen.ops_050: std dev is 22% of the mean
- compile.lifecycle.ops_050: std dev is 22% of the mean
- compile.lifecycle.ops_050: maximum is +63% from the mean
- compile.frontend.ops_100: std dev is 22% of the mean
- compile.frontend.ops_100: maximum is +75% from the mean
- compile.backend.ops_100: std dev is 23% of the mean
- compile.backend.ops_100: maximum is +70% from the mean
- compile.codegen.ops_100: std dev is 26% of the mean
- compile.codegen.ops_100: maximum is +59% from the mean
- compile.lifecycle.ops_100: std dev is 21% of the mean
- compile.frontend.ops_250: std dev is 19% of the mean
- compile.frontend.ops_250: maximum is +56% from the mean
- compile.backend.ops_250: std dev is 21% of the mean
- compile.backend.ops_250: maximum is +55% from the mean
- compile.codegen.ops_250: std dev is 24% of the mean
- compile.codegen.ops_250: maximum is +61% from the mean
- compile.lifecycle.ops_250: std dev is 22% of the mean
- compile.reject.frontend: std dev is 16% of the mean
- compile.reject.lifecycle: std dev is 23% of the mean
- compile.reject.lifecycle: maximum is +52% from the mean
- import.python: not enough processes for a stable result (20 run, 485 suggested)
- import.speedysnek: std dev is 11% of the mean
- import.speedysnek_backend: std dev is 13% of the mean

## Interpretation

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
