# speedy-snek benchmarks

A reproducible [pyperf](https://pyperf.readthedocs.io/) suite that measures
where the time goes in a `@snekit` call, what compilation costs, and what
the wrapper costs when it falls back to Python. It exists to guide the
next optimization, not to advertise speedups. The committed baseline is in
[`RESULTS.md`](RESULTS.md).

## Install

Tested on Linux with CPython 3.14.6 and Rust 1.97, starting from a clean
checkout:

```sh
python3.14 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install maturin
python -m pip install -e ".[tests,benchmarks]"
maturin develop --release
```

`pip install -e` already builds the extension in release mode. Run
`maturin develop --release` again after every Rust change. pyperf is an
optional dependency (`benchmarks` extra), so ordinary users of the
package never install it.

> **Debug builds are not representative.** A plain `maturin develop`
> builds the extension with debug assertions and no optimization. On the
> baseline machine that makes the raw native call more than 10× slower.
> The scripts print a warning and record the build mode
> (`speedysnek_build` in the metadata); the report shows it too.
>
> An editable install keeps the compiled extension in `python/speedysnek/`.
> Every virtual environment that uses this checkout therefore shares it,
> and a debug build from any of them replaces the release build for all.
> To check what you have:
> `python -c "import speedysnek._native as n; print(n.DEBUG_BUILD)"`.

## Run

From the repository root (`results/` is git-ignored):

```sh
mkdir -p results
python benchmarks/bench_runtime.py     -o results/runtime.json
python benchmarks/bench_compilation.py -o results/compilation.json
```

With pyperf's defaults (20 worker processes × 3 values, 1 warmup value
each, loops calibrated to about 100 ms per value) the two scripts took
about 15 minutes on the 4-vCPU baseline machine. pyperf refuses to
overwrite an existing `-o` file; delete it first or pass `--append`.

Both scripts accept every pyperf option:

| Goal | Option |
|---|---|
| Quick look (less reliable) | `--fast` |
| More reliable baseline | `--rigorous`, or e.g. `-p 40 -n 5` |
| Pin workers to one CPU | `--affinity=3` |
| Run a subset | `--filter 'ops_(001\|250)'` (a regex on benchmark names) |
| One value, to check it starts | `--debug-single-value` |

For a more reliable run, also:

- run `python -m pyperf system tune` (needs root). It disables turbo
  boost, fixes the CPU frequency, and so on. Undo it with
  `python -m pyperf system reset`;
- close other programs and keep the machine on AC power;
- compare against your own earlier runs on the same machine, never against
  another machine's numbers. Use `python -m pyperf compare_to old.json
  new.json` for that.

pyperf prints a warning for every unstable benchmark. The scripts do not
suppress these warnings, and the report repeats them in its *Stability*
section. Use `python -m pyperf stats results/runtime.json` and
`python -m pyperf hist ...` to dig into one benchmark.

## Report

```sh
python benchmarks/report.py results/runtime.json results/compilation.json \
    --command "python benchmarks/bench_runtime.py -o results/runtime.json" \
    --command "python benchmarks/bench_compilation.py -o results/compilation.json" \
    -o results/RESULTS.md
```

The report gives the environment, a ratio table, a per-layer breakdown,
a fixed versus per-operation fit, fallback overhead, compilation costs,
break-even estimates and stability warnings. A *Key measurements* section
is generated from the data. `--interpretation FILE` appends a
hand-written analysis.

**Ratios are `layer time / Python time`. Below 1.0 the measured JIT path
is faster; above 1.0 it is slower.**

Break-even is `full lifecycle cost / (Python time per call − @snekit time
per call)`, with every time in seconds. It is computed only when the
warmed `@snekit` call is faster than Python. Otherwise the report says
*No break-even with the current hot path.*

### Updating the committed baseline

```sh
gzip -k results/runtime.json results/compilation.json
mv results/*.json.gz benchmarks/baseline/
python benchmarks/report.py benchmarks/baseline/*.json.gz \
    --command "..." --interpretation benchmarks/interpretation.md \
    -o benchmarks/RESULTS.md
```

[`interpretation.md`](interpretation.md) was written by hand from the
committed numbers. Rewrite it whenever the baseline changes. The raw
pyperf files in [`baseline/`](baseline) let anyone regenerate or
re-analyze the report, for example with
`python -m pyperf stats benchmarks/baseline/runtime.json.gz`.

## What is measured

### Runtime (`bench_runtime.py`)

`workloads.make_workload(n)` builds `ops_NNN(a, b, c, d)` for
n = 1, 5, 20, 50, 100 and 250. Each is one `return` of a flat sum of
products, for example `a + b * c + c * d - d`, containing exactly n
normalized `+`, `-` and `*`. Every operation reads a runtime argument,
so CPython cannot constant-fold anything. With the fixed inputs
`(1234, -567, 89, -1011)` every intermediate value stays below 10⁷, far
inside signed i64. The tests check the operation counts, the absence of
folding, the absence of overflow, and that all layers agree.

Each workload is timed at four layers, outermost first:

| Layer | Timed statement | What it includes |
|---|---|---|
| `python` | `f(a, b, c, d)` | The plain function: CPython call plus bytecode arithmetic. |
| `snekit` | `f(a, b, c, d)` | The public wrapper: `*args`/`**kwargs` packing, `call_count`, state and `__code__` checks, the `try`, then everything below. |
| `compiled_i64` | `f(a, b, c, d)` | `CompiledI64.__call__`, a Python method: argument tuple, arity check, exact-`int` and i64 range check per argument, the native call, the status check. |
| `native_call` | `call(args)` | `I64Function.call` as a bound method on a **prebuilt** tuple. |

What remains in `native_call`: one vectorcall into PyO3, the
exact-tuple check, a `Vec<i64>` allocation, one exact-`int` check and
`i64` extraction per argument, the generated machine code (including its
overflow branches), and building the returned `(status, value)` tuple,
including a new `int` object for any result outside CPython's
small-int cache. The arguments are not rebuilt per call.

Before anything is timed, `workloads.prepare_layers` runs in every pyperf
worker:

1. It makes the 8 warmup calls and the compiling call, outside timing.
2. It asserts that the wrapper is `COMPILED`.
3. It proves, with `sys.setprofile`, that the next call enters
   `CompiledI64.__call__` and no frame of the original function. This
   check is the same technique the main test suite uses, and it means
   the benchmark cannot silently measure a fallback.
4. It checks that all four layers return the same value.

`compiled_i64` and `native_call` use the exact `CompiledI64` object the
wrapper runs; no second compilation is made.

Each statement is repeated 10 times per loop iteration (pyperf's
`duplicate`), so the timing loop adds well under 1 ns per call.
`reference.loop` (an empty statement) shows what remains, and
`reference.python_noop` (a four-argument function returning `a`) shows the
cost of a Python call with no arithmetic.

### Fallback and wrapper overhead (`bench_runtime.py`)

Each `fallback.<case>.snekit` benchmark is a wrapper in a known state.
Its `fallback.<case>.python` baseline is a separate, undecorated copy of
the same function, called with the same statement and arguments. The
difference between the two is therefore the wrapper's cost alone:

| Case | State | Why it falls back |
|---|---|---|
| `cold` | `COLD` | A warmup call. The wrapper's private `warmup_calls` counter is set to −10⁹ so that every timed call takes the warmup path; the code path is otherwise unchanged. |
| `disabled` | `DISABLED` | The function uses `//`; the frontend rejected it. |
| `float_args` | `COMPILED` | `float` arguments: `NativeGuardFailure` is raised and caught. |
| `big_int_args` | `COMPILED` | `2**64` is outside i64: the same exception path. |
| `overflow` | `COMPILED` | `2**62 * 2**62`: the native code runs, reports overflow, and Python recomputes. |
| `keyword_args` | `COMPILED` | `f(a=..., b=...)`: native code is skipped before any guard. |
| `default_args` | `COMPILED` | `f(x)` with `b=4` omitted: arity differs, native code is skipped. |

Setup verifies each case's state, result and Python fallback path.

### Compilation (`bench_compilation.py`)

| Benchmark | Measures |
|---|---|
| `compile.frontend.ops_NNN` | `analyze(func)`: bytecode to IR. |
| `compile.backend.ops_NNN` | `_compiler.compile_i64(ir, 4)`: validation, serialization, Cranelift, `CompiledI64`. |
| `compile.codegen.ops_NNN` | `_native.compile_i64(serialized, 4)`: the Rust side only. |
| `compile.lifecycle.ops_NNN` | A new `@snekit` wrapper, 8 warmup calls, and the compiling call. |
| `compile.reject.frontend` / `.lifecycle` | The same for a rejected function (it ends `DISABLED`). |
| `import.python` / `.speedysnek` / `.speedysnek_backend` | A fresh process running `pass`, `import speedysnek`, or `import speedysnek._compiler`. |

**Fresh compilation state.** No iteration reuses earlier work.

- The frontend stores nothing, so repeating `analyze` repeats the work.
- Each backend iteration creates a new Cranelift `JITModule`. Its
  executable memory is freed when the discarded result is dropped.
- Each lifecycle iteration calls `snekit(func)` again. That creates a new
  wrapper and a new `_JitState`, so the 9th call compiles again.

Two things are process-wide and paid once: importing the compiler
modules, and building the host ISA description in Rust (a `OnceLock`).
pyperf's warmup value absorbs both. The `import.*` benchmarks measure the
import side separately, including the native extension load.

## Tests and CI

`benchmarks/tests/` checks workload correctness (operation counts, no
folding, no overflow, equal results on every layer, the compiled state,
fallback results and exceptions) and the report calculations (ratios,
break-even, no-break-even, the linear fit, stability checks). It also
smoke-runs each script with `--debug-single-value`. CI runs these tests
and lints the benchmark code. It never runs the full suite and never
compares timings: shared CI runners are too noisy for a performance gate.
