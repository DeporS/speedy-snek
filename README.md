# speedy-snek

Because Python doesn’t have to be a slow snek 🐍

SpeedySnek is a small educational JIT compiler experiment for CPython 3.14.
It is at Stage 5: **`@jit` now performs experimental native compilation**.
A decorated function in the supported straight-line integer subset is
compiled once, after a short warmup, to signed 64-bit machine code with
Cranelift. Everything else keeps running as normal Python.

**No performance claims are made yet.** There are no benchmarks, and the
wrapper's per-call overhead has not been optimized; that is the next stage.

## Usage

```python
from speedysnek import jit

@jit
def expression(a, b):
    return (a + b) * 2

for _ in range(100):
    result = expression(10, 20)   # 60

expression(2**62, 2**62)          # 18446744073709551616, computed by Python
expression.call_count             # 101
```

`@jit` must never change what a call returns or raises. It wraps the
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
    → reference interpreter (test oracle) or Cranelift i64 backend (@jit)
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
- `@jit` runs the frontend once per hot function. It is not public API.
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
- `@jit` never uses it; it is used by the tests and is not public API.

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

- It is **private and experimental**. `@jit` uses it to compile hot
  functions.
- It supports only the current arithmetic subset (`+`, `-`, `*` on
  arguments and constants).
- It never returns a wrapped 64-bit value. Constants outside the signed
  64-bit range are rejected at compile time (`UnsupportedProgramError`).
  Arguments that are not exact `int`s in that range, a wrong number of
  arguments, or any `+`, `-` or `*` that overflows raise
  `NativeGuardFailure` instead. `@jit` handles both by running the original
  Python function.
- Every generated function has the same internal ABI,
  `status = f(args_ptr, args_len, result_ptr)`, with overflow handled by
  explicit branches rather than traps.
- No performance claims are made: there are no benchmarks yet.

## Development

Requires CPython 3.14 and a Rust toolchain.

```sh
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop
pytest
```
