# speedy-snek

Because Python doesn’t have to be a slow snek 🐍

![SpeedySnek Demo](assets/speedysnek-demo-v10.gif)

Need more speed? Just `@snekit`.

SpeedySnek compiles supported Python functions to native machine code.
Add a decorator, keep writing Python, and let the snek handle the compilation.

No new language. No manual type annotations. No rewriting your functions in C or Rust.

When a function or its inputs cannot use the native path, SpeedySnek runs
the original Python function.

## Meet the snek

```python
from speedysnek import snekit

@snekit
def sum_squares(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

for _ in range(10):
    result = sum_squares(10_000)

print(result)  # 333283335000
```

The first few calls run in Python. After warmup, SpeedySnek compiles the
function and reuses its native version on eligible calls.

**Currently in active development:** support focuses on integer calculations
and loops. Performance depends on the workload; small functions can still
run slower.

## Go hunting

Not sure which functions the snek can handle? Send it hunting:

```python
import my_calculations
import speedysnek as snek

report = snek.hunt(my_calculations)
```

```text
🐍 Snek went hunting.

sum_squares    ready    integer arithmetic and one range loop
format_report  skipped  f-strings are not supported (line 9)
load_data      skipped  function calls and global names are not supported (line 13)

1 ready, 2 skipped.
```

`hunt()` reads the bytecode of a module's functions, or of functions you
pass it, and reports which ones the current compiler understands. It does
not run, decorate or optimize them yet. **Ready** means only that the
compiler understands a function's bytecode; it does not guarantee a
speedup.

`hunt()` also returns the report as data. Pass `quiet=True` to skip the
printout; see [the hunt guide](docs/hunt.md) for details.

## How it works

1. **Warm up.** The first eight calls run the original function.
2. **Compile.** On the next call, SpeedySnek checks whether the function is
   supported and compiles it using Cranelift.
3. **Reuse.** Subsequent eligible calls execute the cached native code
   through a Rust wrapper.
4. **Fall back.** Unsupported inputs, integer overflow or an exhausted
   loop budget send that call back to Python with its original arguments.

You do not need to choose integer types. The native path uses checked
64-bit integers; values it cannot handle are computed by Python instead.

Compilation currently happens once per decorated function. SpeedySnek does
not yet measure whether the compiled version is faster before using it.

## What works today

- Integer arguments and constants.
- Arithmetic with `+`, `-` and `*`.
- Local variables and updates with `+=`, `-=` and `*=`.
- `if` / `else` branches using integer comparisons.
- One loop per function: `while` or `for i in range(n)`.

Native execution currently requires positional arguments that are exact
built-in `int` values. Floats, collections, nested loops and general
function calls are outside the compiled subset and run in Python.

**Requires standard, GIL-enabled CPython 3.14.** Other Python versions and
free-threaded builds are not supported yet.

See the detailed support notes for
[conditionals](docs/conditionals.md),
[while loops](docs/loops.md) and
[for loops](docs/for-range.md).

## Is it faster?

For supported calculations with enough work per call, it can be.

In the repository's short `for`-loop benchmarks, warmed SpeedySnek functions
ran **21–24× faster than plain Python at 10,000 iterations** on the measured
machine. Those results cover two integer workloads and exclude compilation.

Small functions can lose to the overhead of calling through the decorator.
Compilation also has an upfront cost, so repeated use matters.

See the [loop measurements](benchmarks/experiments/for_range.md) and
[benchmark guide](benchmarks/README.md) for workloads, methodology and limitations.

## Try it

SpeedySnek is **not on PyPI yet**.

Prebuilt wheels are available from completed **Build wheels** runs in
[GitHub Actions](https://github.com/DeporS/speedy-snek/actions).
Download the artifact matching your platform, extract it, then install
the `.whl` file with your CPython 3.14 environment:

```sh
python -m pip install /path/to/downloaded-file.whl
```

Replace the path with the actual wheel filename. Prebuilt wheels need no
Rust toolchain or C/C++ compiler.

Supported wheel platforms:

- Windows x86_64.
- Linux x86_64 with glibc 2.28 or newer.
- macOS 11 or newer on Apple silicon.

GitHub requires you to sign in to download artifacts. If no matching
artifact is available, a repository maintainer can trigger a new build.

## Where we're heading

**The goal: make faster Python require less work.**

The next direction is adaptive optimization: helping SpeedySnek decide
when an optimization is worth using and making that decision understandable.

Planned areas of development:

- **Adaptive execution:** choose between Python and native execution based
  on measured performance.
- **Performance budgets:** account for profiling, compilation and call
  overhead when deciding whether an optimization pays off.
- **Explainable decisions:** show what was compiled, what was skipped and why.
- **Broader support:** handle more useful Python patterns while preserving
  results and fallback behavior.

These are development goals, not features available today.

## Under the scales

SpeedySnek is built with **Python, Rust, PyO3 and Cranelift**.

For implementation details and development:

- [Python frontend](python/speedysnek/_frontend) — analyzes supported Python bytecode.
- [Hunting](docs/hunt.md) — the `snek.hunt()` discovery report.
- [Rust implementation](src) — native wrapper, guards and code generation.
- [Loop design](docs/loops.md) — execution limits and fallback behavior.
- [Range support](docs/for-range.md) — lowering and builtin binding checks.
- [Benchmarks](benchmarks/README.md) — measuring performance and reproducing results.

To build from source, install CPython 3.14 and a Rust toolchain.
Windows also requires Visual Studio C++ Build Tools.

Inside an activated virtual environment:

```sh
python -m pip install maturin
python -m pip install -e ".[tests]"
maturin develop --release
pytest
```

Use a release build when measuring performance.
