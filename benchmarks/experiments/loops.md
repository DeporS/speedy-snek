# Quick measurement: functions with a `while` loop

**Conclusion: on this machine, warmed `@snekit` ran `sum_squares` and
`conditional_sum` 2.2–2.8× faster than plain Python at 10 iterations and
about 20–30× faster from 1,000 iterations up. A native iteration took
about 1.5–1.8 ns, a Python one 42–48 ns. The straight-line regression
check showed no difference from `main` beyond noise. The backedge budget
costs something per iteration in one of the two loops, and a call that
uses it up costs about 3–4% more than plain Python.** No timing threshold
is asserted anywhere; these numbers are short runs on one shared machine.

Script: [`loops.py`](loops.py). Before and after timing, every case is
verified to return the plain function's result and to take the native
path (one native fast call, no frame of the original function), so no
timed call fell back to Python, for overflow or for the budget. Every loop
case stays below the budget. Compilation is measured separately, outside
every timed loop. Rounds alternate between plain Python and the warmed
wrapper.

## Setup

- Shared 4-vCPU KVM guest (Intel Xeon @ 2.10 GHz), CPython 3.14.6,
  x86_64, release build (`maturin develop --release`), no CPU pinning.
  Expect 10–30% noise.
- Defaults: 25 rounds of about 40 ms per variant; median ns per call,
  call loop overhead included in both variants. One run of the script
  takes about 13 seconds.
- Two controlled comparisons, interleaved process by process, twice each:
  - `straight_20` on this branch against a release build of `main`
    (3d3e307);
  - the loop cases against a throwaway local build of this branch with the
    budget check removed from the code generator (not committed), to see
    what the check costs.

## Results

Final run of the committed script (ns per call, median):

| Case | Python | `@snekit` | Ratio |
| --- | ---: | ---: | ---: |
| `sum_squares_10` | 353.6 | 147.3 | 0.417 |
| `sum_squares_100` | 3,513.7 | 300.0 | 0.085 |
| `sum_squares_1000` | 37,381.6 | 1,636.4 | 0.044 |
| `sum_squares_10000` | 476,004.7 | 15,544.7 | 0.033 |
| `conditional_sum_10` | 437.8 | 153.6 | 0.351 |
| `conditional_sum_100` | 3,723.9 | 321.7 | 0.086 |
| `conditional_sum_1000` | 40,727.3 | 1,968.2 | 0.048 |
| `conditional_sum_10000` | 423,740.8 | 17,569.7 | 0.041 |
| `straight_20` (no loop) | 189.4 | 150.7 | 0.796 |

The two earlier runs of this branch agree within noise: ratios 0.453–0.461
at `sum_squares_10`, 0.033–0.035 at `sum_squares_10000`, 0.382–0.385 at
`conditional_sum_10` and 0.044 at `conditional_sum_10000`.

Compilation and startup, measured apart from the calls (median of 15, µs;
three runs):

| Function | Frontend analysis | Cranelift compilation |
| --- | ---: | ---: |
| `sum_squares` | 248–395 | 384–605 |
| `conditional_sum` | 330–532 | 467–695 |
| `ops_020` (no loop) | 262–477 | 598–942 |

This happens once per function, during the ninth call. Analysis and
compilation together took about 0.6–1.0 ms for `sum_squares`, 0.8–1.2 ms
for `conditional_sum` and 0.9–1.4 ms for `ops_020`, so a function needs
roughly a millisecond of extra work before its first native call, not
"well under a millisecond" as an earlier version of this note said. That
one-time cost is separate from the eight warmup calls before it: they run
the original function at Python speed through the cold wrapper, which a
later measurement ([`for_range.md`](for_range.md#compilation-and-warmup))
put at 0.1–0.2 µs extra per call, under 2 µs for all eight.

### Straight-line regression check

`straight_20`, ns per `@snekit` call (ratio to Python):

| Build | Run 1 | Run 2 |
| --- | --- | --- |
| `main` (3d3e307) | 153.4 (0.786) | 155.6 (0.793) |
| this branch | 159.1 (0.706) | 150.9 (0.782) |

No difference beyond noise. Programs without a loop get no budget counter,
no checks and no extra blocks, so their machine code is unchanged.

### What the budget check costs

The same loop cases, `@snekit` ns per call, with the budget check (this
branch) and without it (local build, check removed), two runs each:

| Case | With check | Without check |
| --- | --- | --- |
| `sum_squares_1000` | 1,698 / 1,607 | 1,237 / 1,307 |
| `sum_squares_10000` | 14,682 / 15,289 | 11,902 / 10,924 |
| `conditional_sum_1000` | 1,976 / 2,051 | 2,325 / 2,354 |
| `conditional_sum_10000` | 17,928 / 17,877 | 21,317 / 21,884 |

For `sum_squares` the check (a compare, a branch and a decrement per
iteration) adds about 0.3–0.4 ns per iteration, 25–35% of a very small
body. For `conditional_sum` the build *with* the check was 15–20% faster,
which can only be a code layout effect of the extra block. So the check is
cheap but not free, and its cost depends on the loop; it is never paid by
functions without a loop.

### Using up the budget

`sum_squares(1,000,001)` needs one backedge more than the default budget
of 1,000,000. `@snekit` runs it natively until the budget is used up,
then again, from the start, in Python. Median of 9:

| Run | Python | `@snekit` | Ratio | Native attempt alone |
| --- | ---: | ---: | ---: | ---: |
| 1 | 51.9 ms | 53.8 ms | 1.038 | 1.45 ms |
| 2 | 51.2 ms | 53.1 ms | 1.039 | 1.43 ms |
| final | 51.8 ms | 53.3 ms | 1.030 | 1.44 ms |

So the restart repeats about 1.4 ms of native work, about 3% of the Python
time, and holds the GIL for that long. This is the case where `@snekit`
is slower than plain Python; it grows with the size of the loop body.

## Interpretation

- At 10 iterations the fixed cost of a warmed call (about 145 ns) already
  costs less than the Python loop. From about 1,000 iterations up the
  fixed cost no longer matters and the ratio approaches the per-iteration
  ratio, about 1:24–31.
- `conditional_sum` costs about 0.3 ns more per iteration than
  `sum_squares`, for its extra comparison and branch.
- Long loops are where native code gains most, but a call that needs more
  than the budget allows is always a little slower than plain Python,
  because the first million iterations run twice. There is no "always
  faster" guarantee.
