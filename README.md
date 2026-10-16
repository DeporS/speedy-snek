# speedy-snek

Because Python doesn’t have to be a slow snek 🐍

SpeedySnek is a small educational JIT compiler experiment for CPython 3.14.
It is at Stage 2: the first bytecode frontend. **No decorated Python
function is compiled or optimized yet.**

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

## Development

Requires CPython 3.14 and a Rust toolchain.

```sh
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop
pytest
```
