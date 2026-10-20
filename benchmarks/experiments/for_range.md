# Quick measurement: `for` loops over `range(n)`

**Conclusion: on this machine, warmed `@snekit` ran the `for`/`range`
versions of `sum_squares` and `conditional_sum` 1.6–2.0× faster than plain
Python at 10 iterations and 21–24× faster at 10,000, about as fast as the
equivalent warmed `while` loops; for `conditional_sum` the `for` version
was 15–25% faster than the `while` version from 100 iterations up. The
range guard took 6–8 µs once, at compilation. The loop-free regression
check showed no difference from `main` beyond noise.** No timing threshold
is asserted anywhere; these numbers are short runs on one shared machine.

Script: [`for_range.py`](for_range.py). Before and after timing, every
case is verified to return the plain function's result and to take the
native path (one native fast call, no frame of the original function), so
no timed call fell back to Python, for overflow, the budget or the range
guard. The `for` wrappers are checked to hold a valid range guard and the
`while` and loop-free wrappers to hold none. Every loop case stays below
the backedge budget. Compilation is measured separately, outside every
timed loop. Each round times the three variants of a case one after the
other.

## Setup

- Shared 4-vCPU KVM guest (Intel Xeon @ 2.10 GHz), CPython 3.14.6,
  x86_64, release build (`maturin develop --release`), no CPU pinning.
  Expect 10–30% noise.
- Defaults: 25 rounds of about 40 ms per variant; median ns per call,
  call loop overhead included in every variant. One run of the script
  takes about 13 seconds.
- Regression check: `straight_20` (20 operations, no loop, no `range`) on
  this branch against a release build of `main` (286cfd0), five runs each,
  alternating, with the same script (`--cases straight_20`).

## Results

Two runs of the committed script (ns per call, median; ratio to plain
Python):

| Case | Python | `@snekit` `for` | Ratio | `@snekit` `while` | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sum_squares_10` | 269 / 255 | 164 / 154 | 0.61 / 0.60 | 153 / 150 | 0.57 / 0.59 |
| `sum_squares_100` | 2,300 / 2,410 | 295 / 308 | 0.13 / 0.13 | 288 / 303 | 0.13 / 0.13 |
| `sum_squares_1000` | 25,241 / 27,399 | 1,609 / 1,748 | 0.064 / 0.064 | 1,591 / 1,673 | 0.063 / 0.061 |
| `sum_squares_10000` | 339,969 / 384,667 | 14,688 / 16,281 | 0.043 / 0.042 | 14,634 / 15,942 | 0.043 / 0.041 |
| `conditional_sum_10` | 287 / 306 | 151 / 153 | 0.53 / 0.50 | 150 / 163 | 0.53 / 0.53 |
| `conditional_sum_100` | 2,612 / 2,819 | 277 / 289 | 0.11 / 0.10 | 330 / 347 | 0.13 / 0.12 |
| `conditional_sum_1000` | 29,125 / 34,080 | 1,552 / 1,722 | 0.053 / 0.051 | 1,983 / 2,265 | 0.068 / 0.066 |
| `conditional_sum_10000` | 322,280 / 310,136 | 14,907 / 14,884 | 0.046 / 0.048 | 19,561 / 18,159 | 0.061 / 0.059 |
| `straight_20` (no loop) | 199 / 206 | 157 / 162 | 0.79 / 0.79 | | |

The "Python" column is the plain `for` function. Earlier development runs
agree within noise.

### Compilation and warmup

Measured apart from the timed calls, median of 15, µs (two runs):

| Function | Frontend analysis | Range guard | Cranelift compilation |
| --- | ---: | ---: | ---: |
| `sum_squares` (`for`) | 293–295 | 6.1–7.7 | 429–519 |
| `sum_squares_while` | 246–270 | – | 394–465 |
| `conditional_sum` (`for`) | 397–425 | 6.5–7.1 | 541–612 |
| `conditional_sum_while` | 363–402 | – | 523–597 |
| `straight_20` (no loop) | 318–322 | – | 648–721 |

Across all development runs on this machine analysis took 0.25–0.6 ms
and Cranelift 0.4–1.0 ms per function, so analysis plus compilation is
about 0.7–1.4 ms. The range guard walks the function's globals and
builtins dicts once (a few hundred entries here) and registers two dict
watchers; that is the only cost a `for`/`range` function adds.

The life of a fresh wrapper, timed one call at a time (median of 15
wrappers, timer included):

| Function | Plain call | Warmup call | Compiling (9th) call |
| --- | ---: | ---: | ---: |
| `sum_squares` | 362–440 ns | 520–584 ns | 0.70–0.88 ms |
| `conditional_sum` | 374–378 ns | 552–559 ns | 0.98–1.01 ms |
| `straight_20` | 305–312 ns | 450–470 ns | 0.92–1.02 ms |

So the eight warmup calls each cost 0.1–0.2 µs more than a plain call (the
cold wrapper counts them), under 2 µs for all eight, while the ninth call
pays roughly a millisecond once for analysis, the guard and compilation.
These are two different costs: the first is per warmup call, the second
per function.

### Loop-free regression check

`straight_20`, ns per warmed `@snekit` call (ratio to plain Python), runs
alternating between the two builds:

| Build | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| `main` (286cfd0) | 158.7 (0.768) | 145.9 (0.769) | 153.2 (0.769) | 154.1 (0.788) | 156.0 (0.774) |
| this branch | 157.7 (0.767) | 158.5 (0.754) | 172.4 (0.836) | 158.6 (0.816) | 163.9 (0.781) |

No consistent difference: the spread within each build is as large as the
gap between them. A function that does not use `range` gets no guard: its
specialization's `function` is the plain `I64Function`, and the wrapper's
first type check takes the same branch as on `main`. Only a
`GuardedI64Function` pays the extra atomic load.

## Interpretation

- The `for` lowering is the `while` loop's shape with a hidden counter,
  so per iteration it does the same work as the equivalent `while` loop:
  one comparison, one counter increment with its (never taken) overflow
  branch, and the backedge budget check. `sum_squares` confirms this.
- `conditional_sum` as a `for` loop is faster than as a `while` loop. The
  likely reason is code layout: the lowering increments the counter at
  the top of the body, so both branches of the `if`/`else` jump straight
  back to the header, while the `while` version joins them before
  `i += 1`. This was not investigated further.
- The range guard costs nothing measurable per call: a warmed call of a
  `for`/`range` function is as fast as a `while` function's at 10
  iterations, where the fixed cost dominates.
- As for `while` loops, a call needing more than the 1,000,000-iteration
  budget runs its first million iterations natively and then everything
  again in Python ([`loops.md`](loops.md#using-up-the-budget)); that was
  not measured again here.
