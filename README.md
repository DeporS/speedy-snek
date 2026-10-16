# speedy-snek

Because Python doesn’t have to be a slow snek 🐍

SpeedySnek is a small educational JIT compiler experiment for CPython 3.14.
It is at Stage 3: a bytecode frontend plus a reference interpreter. **No
decorated Python function is compiled or optimized yet.**

## Usage

```python
from speedysnek import jit

@jit
def add(a, b):
    return a + b

add(1, 2)        # 3, computed by the original Python function
add.call_count   # 1
```

`jit` wraps the function with `functools.wraps`, forwards all arguments,
results and exceptions unchanged, and exposes `call_count`: the number of
times the wrapper has been called (including calls that raised).

Native Rust code lives in the private `speedysnek._native` extension module.

## Compiler pipeline

```
Python function
    → CPython 3.14 bytecode
    → SpeedySnek normalized instructions
    → Python reference interpreter
    → native code (future)
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
- The frontend is **not connected to `@jit`** and is not public API.
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

- It is a **correctness oracle**, not a performance optimization: the future
  native backend must produce exactly the same results.
- It executes only **exact-`int` arithmetic**. Every argument the program
  reads must satisfy `type(value) is int` (so `bool`, `float`, `str` and
  `int` subclasses are rejected); otherwise it raises `GuardFailure`, which a
  future JIT will handle by falling back to CPython.
- The whole program is validated before it runs. Malformed instructions
  (empty program, stack underflow, missing or misplaced `RETURN`, leftover
  stack values, …) raise `InvalidProgramError`, which means a compiler bug.
- Arithmetic uses Python `int`, so **arbitrary precision is intentional**:
  `2**63 + 1` is not wrapped to a 64-bit value.
- It is **not connected to `@jit`** and is not public API.

## Development

Requires CPython 3.14 and a Rust toolchain.

```sh
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop
pytest
```
