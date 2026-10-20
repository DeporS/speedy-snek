# Quick measurement: functions with `if`/`else`

**Conclusion: branch functions run natively and behave like other small
functions: the fixed cost of a `@snekit` call dominates, so they are about
2× slower than plain Python. Straight-line functions did not get slower.**
No timing threshold is asserted anywhere; these numbers are one short run
on one machine.

Script: [`conditionals.py`](conditionals.py). Every case is verified to
return the plain function's result and to take the native path (one native
fast call, no frame of the original function) before and after timing.
Compilation happens before timing. Rounds alternate between plain Python
and the warmed wrapper.

## Setup

- Shared 4-vCPU KVM guest, CPython 3.14.6, x86_64, release build
  (`maturin develop --release`), no CPU pinning. Expect 10–30% noise.
- Defaults: 15 rounds × 500,000 calls per variant; median ns per call,
  loop overhead included in both variants.
- The straight-line cases were also run against a release build of
  `main` (c317805), interleaved with this branch, twice each.

## Results

Two runs of this branch (ns per call, median):

| Case | Python | `@snekit` | Ratio | Python (2) | `@snekit` (2) | Ratio (2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `if_else_then` (`calculate_score`, `score < 0`) | 74.8 | 147.8 | 1.98 | 48.4 | 112.6 | 2.33 |
| `if_else_else` (`calculate_score`, else branch) | 51.1 | 108.3 | 2.12 | 49.0 | 103.2 | 2.11 |
| `if_join` (`adjust`, `if` without `else`) | 46.1 | 108.6 | 2.35 | 46.5 | 110.1 | 2.37 |
| `nested` (nested and sequential `if`) | 60.1 | 114.3 | 1.90 | 71.3 | 138.2 | 1.94 |
| `straight_20` (20 operations, no locals) | 181.3 | 129.3 | 0.71 | 170.7 | 129.5 | 0.76 |
| `locals` (README `calculate_score`, no branches) | 50.2 | 111.6 | 2.22 | 45.0 | 88.6 | 1.97 |

Straight-line regression check against `main`:

| Case | `main` ratio (run 1, 2) | This branch ratio (run 1, 2) |
| --- | --- | --- |
| `straight_20` | 0.75, 0.77 | 0.71, 0.76 |
| `locals` | 2.08, 2.13 | 2.22, 1.97 |

## Interpretation

- A warmed call of a branch function costs about 105–150 ns, the same as a
  straight-line function of similar size. The plain functions take 45–75
  ns, so the wrapper's fixed cost (see [`../RESULTS.md`](../RESULTS.md))
  makes these small functions slower under `@snekit`, as it does for
  straight-line functions below about 30 operations.
- The two arms of `calculate_score` cost the same within noise.
- Straight-line functions show no difference beyond noise. Within one
  block, Cranelift variables resolve to the same SSA values the previous
  hand-written slot map produced.

This does not claim a speedup for branch functions; larger branch bodies
would amortize the fixed cost the same way larger expressions do.
