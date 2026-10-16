# speedy-snek

Because Python doesn’t have to be a slow snek 🐍

Need more speed? Just `@snekit`.

SpeedySnek is a small educational JIT compiler experiment for CPython 3.14.
**`@snekit` performs experimental native compilation**: a decorated
function in the supported straight-line integer subset is compiled once,
after a short warmup, to signed 64-bit machine code with Cranelift.
Everything else, including every unsupported function, argument or call,
safely falls back to normal Python.

The project now has a reproducible [benchmark suite](benchmarks/README.md)
measuring where the time goes. **The optimizer is experimental, and for
most functions it currently makes calls slower, not faster.** On the
committed baseline machine, a warmed `@snekit` call has a fixed cost of
about 1.2 µs. That makes it 25× slower than plain Python for a one-operation
function, still 1.3× slower at 100 operations, and faster (0.59×) only at
250 operations. See [Benchmarks](#benchmarks) and
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

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
been called, including calls that raised. `call_count` is a plain attribute
and its increments are not atomic, so concurrent calls from several threads
can occasionally be undercounted.

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
  calls.
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
  Arguments that are not exact `int`s in that range, a wrong number of
  arguments, or any `+`, `-` or `*` that overflows raise
  `NativeGuardFailure` instead. `@snekit` handles both by running the original
  Python function.
- Every generated function has the same internal ABI,
  `status = f(args_ptr, args_len, result_ptr)`, with overflow handled by
  explicit branches rather than traps.
- It is not yet faster than Python; see [Benchmarks](#benchmarks).

## Benchmarks

[`benchmarks/`](benchmarks) is a [pyperf](https://pyperf.readthedocs.io/)
suite. It times the plain function, the warmed `@snekit` wrapper, the
internal `CompiledI64` object and the raw native call separately, along
with fallback overhead, compilation and warmup cost, and cold import. The
benchmarks are optional, and pyperf is not installed for ordinary users.

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
show 10–25% noise. On that machine:

- A warmed `@snekit` call is **slower than plain Python for 1 to 100
  operations**: 25.2× at 1, 5.9× at 20 and 1.33× at 100. It beats Python
  only for the 250-operation workload (0.59×). A separate repeat run
  confirmed both of the results closest to parity.
- Only that workload breaks even, after about 8,300 calls (an 8.3 ms
  warmup-and-compile lifecycle, saving about 1 µs per call).
- The generated machine code is not the bottleneck. The raw native call
  costs about 0.7 ns per extra operation, against about 9.5 ns in Python.
  Most of each warmed call (55–66%) goes to the Python-level argument
  guards in `CompiledI64.__call__`, and another 21–33% to the decorator
  wrapper.
- Fallback calls cost 220–375 ns extra when native code is skipped, and
  1.65–2.24 µs extra when a guard rejects the arguments (`float`, large
  `int`, overflow).

The [interpretation](benchmarks/RESULTS.md#interpretation) recommends
next moving argument validation out of the Python guard layer, so the
wrapper calls native code directly.

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
