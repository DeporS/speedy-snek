# speedy-snek

Because Python doesn’t have to be a slow snek 🐍

Need more speed? Just `@snekit`.

SpeedySnek is a small educational JIT compiler experiment for CPython 3.14.
**`@snekit` performs experimental native compilation**: a decorated
function in the supported straight-line integer subset is compiled once,
after a short warmup, to signed 64-bit machine code with Cranelift.
Everything else, including every unsupported function, argument or call,
safely falls back to normal Python.

The project has a reproducible [benchmark suite](benchmarks/README.md)
measuring where the time goes. **The optimizer is experimental, and for
small functions it still makes calls slower, not faster.** On the
committed baseline machine, a warmed `@snekit` call has a fixed cost of
about 500 ns. That makes it 7.7× slower than plain Python for a
one-operation function and 1.4× slower at 20 operations. It is faster from
about 30 operations up: 0.61× at 50, 0.16× at 250. See
[Benchmarks](#benchmarks) and [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

## Usage

```python
from speedysnek import snekit

@snekit
def expression(a, b):
    return (a + b) * 2

for _ in range(100):
    result = expression(10, 20)   # 60

expression(2**62, 2**62)          # 18446744073709551616, computed by Python
expression.call_count             # 101
```

`@jit` is the descriptive, backwards-compatible name for the same
decorator: `speedysnek.jit is speedysnek.snekit`, so existing
`from speedysnek import jit` code keeps working unchanged. New code should
use `@snekit`.

### Guarantees

`@snekit` must never change what a call returns or raises. It wraps the
function with `functools.wraps` (so `__name__`, `__qualname__`, `__doc__`,
`__module__`, `__annotations__` and `__wrapped__` are preserved, and methods
bind normally) and exposes `call_count`: the number of times the wrapper has
been called, including calls that raised. It is an exact `int` with no
upper bound, its increments are atomic, and it can be set to any integer
(anything `operator.index` accepts).

The wrapper is a native callable object, not a Python function: it has no
`__code__`, `__defaults__` or `__globals__` of its own,
`inspect.isfunction` is false for it, and tracebacks and profilers show no
wrapper frame. The full list of differences is in
[`benchmarks/experiments/native_wrapper.md`](benchmarks/experiments/native_wrapper.md#observable-differences).

### Runtime pipeline

```
decorated function
    → warmup
    → frontend analysis
    → normalized IR
    → cached Cranelift i64 specialization
    → native execution when guards pass
    → original CPython function otherwise
```

Once compiled, a call that fits runs no Python code at all: the wrapper
itself is written in Rust.

```
@snekit wrapper (Rust)
    → native fast path (Rust)
        → validate every argument once: exact int, inside signed 64-bit
        → run the machine code, which checks every operation for overflow
        → return the result int, or None for a guard miss
    → original function, only if the result was None
```

Each decorated function has its own small state machine:

- **Cold.** The first 8 calls run the original function. Nothing is
  analyzed, and the function is never run just to collect types.
- On the **next call**, the function is analyzed and compiled, **once**.
  Compilation modules are imported only at this point, so a plain
  `import speedysnek` stays cheap.
- **Compiled.** The native code is cached and reused. A call runs natively
  only if it passes **positional arguments only**, exactly as many as the
  function has parameters, and the function's `__code__` is still the code
  object that was compiled. Calls with keyword arguments, omitted default
  arguments, or the wrong number of arguments run the original function, so
  CPython produces its usual results and error messages.
- If the native guards reject a call (`bool`, `int` subclasses, `float`,
  `str` or any non-`int` argument, an `int` outside the signed 64-bit
  range, or a `+`, `-` or `*` that would overflow, even in an intermediate
  value), **that call** runs the original function, which computes Python's
  exact arbitrary-precision result. The native code stays cached for later
  calls. Such a guard miss is expected, so it is reported with a cheap
  `None` sentinel, not an exception, and the guards never run any of the
  argument's Python code (no `__index__`, `__int__`, `__repr__` or
  metaclass hooks). Only an unexpected error from the native code
  disables the specialization.
- **Disabled.** If the function is outside the supported subset (branches,
  loops, calls, globals, closures, generators, async functions, keyword
  arguments, …), the backend cannot compile it, or the compiler fails
  unexpectedly, it is never compiled again and every call runs the original
  function. Replacing the function's `__code__` after compilation also
  disables the cached code; stale machine code is never run, and there is
  no recompilation yet.

Whenever native execution does not apply, the original function is called
exactly once, and its result or exception is passed through unchanged.

Native Rust code lives in the private `speedysnek._native` extension module.

## Compiler pipeline

```
Python function
    → CPython 3.14 bytecode frontend
    → normalized IR
    → reference interpreter (test oracle) or Cranelift i64 backend (@snekit)
```

The private `speedysnek._frontend` package implements the first two
arrows. It reads a function's bytecode with the public `dis` module, without
calling the function, and normalizes it into a tiny stack-machine
instruction set:

```python
from speedysnek._frontend import analyze

def add(a, b):
    return a + b

for instruction in analyze(add).instructions:
    print(instruction)
# LOAD_ARG 0
# LOAD_ARG 1
# ADD
# RETURN
```

Current status:

- Only **straight-line integer arithmetic** is supported: positional
  arguments, `int` constants, `+`, `-`, `*` and `return`.
- Anything else (division, calls, globals, attributes, branches, loops,
  closures, generators, async functions, non-`int` constants, …) is
  rejected with a machine-readable reason and, where applicable, the
  offending CPython opcode and bytecode offset.
- `@snekit` runs the frontend once per hot function. It is not public API.
- All CPython 3.14-specific handling lives in
  `python/speedysnek/_frontend/cpython314.py`.

### Reference interpreter

The private `speedysnek._interpreter` module executes normalized
instructions directly, without the original function:

```python
from speedysnek._frontend import Instruction, Op
from speedysnek._interpreter import execute

program = (
    Instruction(Op.LOAD_ARG, 0),
    Instruction(Op.LOAD_CONST, 2),
    Instruction(Op.MULTIPLY),
    Instruction(Op.RETURN),
)
execute(program, args=(5,))  # 10
```

- It is a **correctness oracle**, not a performance optimization: the native
  backend must produce exactly the same results.
- It executes only **exact-`int` arithmetic**. Every argument the program
  reads must satisfy `type(value) is int` (so `bool`, `float`, `str` and
  `int` subclasses are rejected); otherwise it raises `GuardFailure`.
- The whole program is validated before it runs. Malformed instructions
  (empty program, stack underflow, missing or misplaced `RETURN`, leftover
  stack values, …) raise `InvalidProgramError`, which means a compiler bug.
- Arithmetic uses Python `int`, so **arbitrary precision is intentional**:
  `2**63 + 1` is not wrapped to a 64-bit value.
- `@snekit` never uses it; it is used by the tests and is not public API.

### Cranelift i64 backend

The private `speedysnek._compiler` module compiles normalized instructions
into host machine code with [Cranelift](https://cranelift.dev/), working on
signed 64-bit integers:

```python
from speedysnek._compiler import compile_i64
from speedysnek._frontend import analyze

def expression(a, b):
    return (a + b) * 2

compiled = compile_i64(analyze(expression).instructions)
compiled(10, 20)  # 60, computed by generated machine code
```

- It is **private and experimental**. `@snekit` uses it to compile hot
  functions.
- It supports only the current arithmetic subset (`+`, `-`, `*` on
  arguments and constants).
- It never returns a wrapped 64-bit value. Constants outside the signed
  64-bit range are rejected at compile time (`UnsupportedProgramError`).
  At call time, arguments that are not exact `int`s in that range, a wrong
  number of arguments, or any `+`, `-` or `*` that overflows are a guard
  miss.
- A compiled program has two entry points:
  - `compiled(*args)`, the **diagnostic** path, raises
    `NativeGuardFailure` with the exact reason for a miss (wrong arity, not
    an exact `int`, outside i64, or overflow at a given instruction). It
    checks the arguments in Python, then again in Rust. Tests and direct
    users of the compiler rely on it.
  - `compiled.fast_call(args)`, the **fast** path, checks the arguments
    once, in Rust, and returns the result `int`, or `None` for any miss,
    without raising and without saying why. `@snekit`'s native wrapper
    runs the same Rust code directly.
- Every generated function has the same internal ABI,
  `status = f(args_ptr, args_len, result_ptr)`, with overflow handled by
  explicit branches rather than traps.
- Native code beats Python by a wide margin per operation, but each call
  still has a fixed cost; see [Benchmarks](#benchmarks).

## Benchmarks

[`benchmarks/`](benchmarks) is a [pyperf](https://pyperf.readthedocs.io/)
suite. It times the plain function, the warmed `@snekit` wrapper and the
native fast path called from Python, and separately the diagnostic `CompiledI64`
path. It also measures fallback overhead, compilation and warmup cost,
and cold import. The benchmarks are optional, and pyperf is not installed
for ordinary users.

```sh
python -m pip install maturin
python -m pip install -e ".[tests,benchmarks]"   # builds a release extension
maturin develop --release                         # after every Rust change

mkdir -p results
python benchmarks/bench_runtime.py     -o results/runtime.json
python benchmarks/bench_compilation.py -o results/compilation.json
python benchmarks/report.py results/runtime.json results/compilation.json \
    -o results/RESULTS.md
```

> **Debug builds are not representative.** A plain `maturin develop`
> builds an unoptimized extension that is many times slower; the scripts
> warn about it and the report records the build mode.

[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) is one committed baseline
from one machine, not a universal claim. It was measured on a shared
4-vCPU KVM guest with CPython 3.14.6 and a release build; most benchmarks
show 15–25% noise. That machine ran plain Python much slower than during
the previous baseline, so the previous code was measured again in the
same session as a controlled comparison. On that machine:

- Removing the Python guard layer made a warmed `@snekit` call **about
  2.5–3× cheaper** than the previous code: 1,446 → 544 ns at 1 operation
  and 1,741 → 725 ns at 250. Its fixed cost fell from about 1.45 µs to
  about 500 ns.
- Warmed `@snekit` is now **faster than plain Python from about 30
  operations** (0.61× at 50, 0.36× at 100, 0.16× at 250). It is still
  slower for smaller functions: 7.7× at 1 operation, 3.6× at 5 and 1.4× at
  20. The previous code crossed over at about 94 operations in the same
  session (134 in the previous baseline).
- Three workloads now break even, after about 7,200 (`ops_050`), 4,500
  (`ops_100`) and 3,200 (`ops_250`) calls; previously only `ops_250` did.
- A guard miss (`float`, an `int` outside i64, overflow) now costs 430–515
  ns over plain Python instead of 1.9–2.8 µs, because it no longer raises
  and catches an exception. Paths that skip native code are unchanged
  (315–525 ns extra).
- The **decorator wrapper itself** is now the largest cost: 340–430 ns,
  56–78% of a warmed call. The native fast call costs about 112 ns plus
  0.84 ns per operation (Python: about 18 ns per operation on this
  machine).

The [interpretation](benchmarks/RESULTS.md#interpretation) recommends
making the wrapper cheaper next. Since that baseline, the wrapper has moved
into Rust: a short A/B run measured warmed calls 0.58–0.66× as long as with
the Python wrapper (about 90–100 ns less per call) and guard misses about
0.55×, with no slower case
([experiment](benchmarks/experiments/native_wrapper.md)). The committed
baseline predates this change.

[`benchmarks/README.md`](benchmarks/README.md) explains the methodology,
how to get more stable numbers, and how to update the baseline.

## Development

Requires CPython 3.14 and a Rust toolchain.

```sh
python -m venv .venv && source .venv/bin/activate
python -m pip install maturin
python -m pip install -e ".[tests,benchmarks]"
maturin develop --release
pytest
```

`pytest` runs the compiler tests in `python/tests/` and the benchmark
correctness tests in `benchmarks/tests/`. No test asserts a timing.
