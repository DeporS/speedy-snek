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
about 18 and 11 minutes on the 4-vCPU baseline machine. pyperf refuses to
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

The report gives the environment, a ratio table, a breakdown of the
warmed `@snekit` call, a separate table for the diagnostic path, a fixed
versus per-operation fit with the estimated crossover, fallback
overhead, compilation costs, break-even estimates and stability
warnings. A *Key measurements* section is generated from the data.
`--interpretation FILE` appends a hand-written analysis.
`--compare LABEL FILE...` (repeatable) adds a before/after table against
earlier results, including plain-Python rows that show how much the
machine itself differs between the runs.

**Ratios are `layer time / Python time`. Below 1.0 the measured JIT path
is faster; above 1.0 it is slower.**

Break-even is `full lifecycle cost / (Python time per call − @snekit time
per call)`, with every time in seconds. It is computed only when the
warmed `@snekit` call is faster than Python. Otherwise the report says
*No break-even with the current hot path.*

The crossover is where the fitted `@snekit` line meets Python's:
`(@snekit fixed − Python fixed) / (Python per op − @snekit per op)`.
It depends on how fast the machine runs plain Python as much as on
speedy-snek.

### Updating the committed baseline

```sh
gzip -k results/runtime.json results/compilation.json
mv results/*.json.gz benchmarks/baseline/
python benchmarks/report.py \
    benchmarks/baseline/runtime.json.gz benchmarks/baseline/compilation.json.gz \
    --compare "same-session control run of the previous code (3df7644)" \
        benchmarks/baseline/control-3df7644/runtime.json.gz \
        benchmarks/baseline/control-3df7644/compilation.json.gz \
    --compare "Stage 6 baseline (historical)" \
        benchmarks/baseline/stage6/runtime.json.gz \
        benchmarks/baseline/stage6/compilation.json.gz \
    --command "..." --interpretation benchmarks/interpretation.md \
    -o benchmarks/RESULTS.md
```

[`interpretation.md`](interpretation.md) was written by hand from the
committed numbers. Rewrite it whenever the baseline changes. The raw
pyperf files in [`baseline/`](baseline) let anyone regenerate or
re-analyze the report, for example with
`python -m pyperf stats benchmarks/baseline/runtime.json.gz`:

| Path | What it is |
|---|---|
| `baseline/runtime.json.gz`, `baseline/compilation.json.gz` | The current baseline (commit `9810f2f`). |
| `baseline/control-3df7644/` | The previous code (commit `3df7644`, before the native fast call), run with the same commands in the same session, right after the current baseline. The controlled before/after comparison. |
| `baseline/stage6/` | The previous committed baseline, measured earlier on the same kind of machine, while it ran plain Python noticeably faster. Historical only. |
| `baseline/stage6/repeat-ops100-250.json.gz` | That baseline's CPU-pinned repeat of `ops_100` and `ops_250`. |
| `baseline/repeat-pinned/` | A CPU-pinned (`--affinity=3`) repeat of `ops_020`, `ops_050` and the `cold`, `disabled` and `keyword_args` fallbacks, run in two interleaved rounds for the current and the control code (`-p 10 --append` each round). |

To compare your own runs of two commits, measure both in the same
session, one after the other, rather than comparing against a committed
file from another day.

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

Each workload is timed at five layers. The first three are the public
path, outermost first; a warmed `@snekit` call is exactly the wrapper
around one `fast_call`:

| Layer | Timed statement | What it includes |
|---|---|---|
| `python` | `f(a, b, c, d)` | The plain function: CPython call plus bytecode arithmetic. |
| `snekit` | `f(a, b, c, d)` | The public wrapper: `*args`/`**kwargs` packing, `call_count`, state, `__code__`, keyword and arity checks, the `try`, the `None` check, then `fast_call`. |
| `fast_call` | `fast_call(args)` | The wrapper's own cached bound `I64Function.try_call` on a **prebuilt** tuple: what `@snekit` runs below the decorator. |

The other two are the diagnostic path. `@snekit` no longer runs it; it is
measured to show what the decorator stopped paying, and for direct
compiler users:

| Layer | Timed statement | What it includes |
|---|---|---|
| `compiled_i64` | `f(a, b, c, d)` | `CompiledI64.__call__`, a Python method: argument tuple, arity check, exact-`int` and i64 range check per argument, `I64Function.call`, the status check. |
| `native_call` | `call(args)` | `I64Function.call` as a bound method on a **prebuilt** tuple, returning `(status, value)`. |

What remains in `fast_call`: one vectorcall into PyO3, the exact-tuple
and arity checks, one exact-`int` check and non-raising `i64` conversion
per argument into a stack buffer (a `Vec` above 8 arguments), the
generated machine code (including its overflow branches), and the
result `int`, a new object for any result outside CPython's small-int
cache. `native_call` does the same work, but its conversion raises for
rejected arguments, it always allocates a `Vec`, and it builds a
`(status, value)` tuple. Neither layer rebuilds the arguments per call.

Do not add the layers up. Only `snekit − fast_call` is a nested
difference; `compiled_i64` and `native_call` are a separate path, and
the report shows them in their own table.

Before anything is timed, `workloads.prepare_layers` runs in every pyperf
worker:

1. It makes the 8 warmup calls and the compiling call, outside timing.
2. It asserts that the wrapper is `COMPILED`.
3. It proves, with `sys.setprofile`, that the next call makes exactly one
   C call to a bound `I64Function.try_call` and enters neither
   `CompiledI64.__call__` nor any frame of the original function. This
   check is the same technique the main test suite uses, and it means
   the benchmark cannot silently measure a fallback or the diagnostic
   path.
4. It checks that all five layers return the same value.

`fast_call` is the very bound method cached in the wrapper's
specialization, and `compiled_i64` and `native_call` use the exact
`CompiledI64` object behind it; no second compilation is made.

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
| `float_args` | `COMPILED` | `float` arguments: the native fast call returns `None` without running the code. |
| `big_int_args` | `COMPILED` | `2**64` is outside i64: the same `None` path. |
| `overflow` | `COMPILED` | `2**62 * 2**62`: the native code runs, detects overflow, returns `None`, and Python recomputes. |
| `keyword_args` | `COMPILED` | `f(a=..., b=...)`: native code is skipped before any guard. |
| `default_args` | `COMPILED` | `f(x)` with `b=4` omitted: arity differs, native code is skipped. |

Setup verifies each case's state, result and path: the three guard-miss
cases must make exactly one native fast call and then run Python once,
and the diagnostic `CompiledI64.__call__` must agree that the call is a
guard miss. The others must not reach native code at all.

For the three guard-miss cases, `fallback.<case>.fast_call` also times
the native miss alone: the wrapper's own `fast_call` on a prebuilt tuple,
returning `None`. No exception is raised anywhere on these paths.

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
that warmed `@snekit` uses the native fast call and never
`CompiledI64.__call__`, fallback paths, results and exceptions) and the
report calculations (ratios, break-even, no-break-even, the linear fit,
the crossover, the breakdowns, comparisons, stability checks). It also
smoke-runs each script with `--debug-single-value`, including the
`fast_call` layers. CI runs these tests
and lints the benchmark code. It never runs the full suite and never
compares timings: shared CI runners are too noisy for a performance gate.

## Experiments

[`experiments/`](experiments) holds short, self-contained A/B experiments
that are not part of the pyperf suite and do not update the baseline:

| Experiment | Conclusion |
|---|---|
| [`wrapper_overhead`](experiments/wrapper_overhead.md) | A wrapper generated for the function's positional signature does not make warmed `@snekit` calls cheaper; not adopted. Also attributes the wrapper's remaining cost. |

`benchmarks/tests/test_wrapper_experiment.py` keeps the experiment's
prototype honest against the production wrapper.
