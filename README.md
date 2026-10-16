# speedy-snek

Because Python doesn’t have to be a slow snek 🐍

SpeedySnek is a small educational JIT compiler experiment for CPython 3.14.
It is at Stage 1: the project foundation. **No decorated Python
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

## Development

Requires CPython 3.14 and a Rust toolchain.

```sh
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop
pytest
```
