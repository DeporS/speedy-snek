"""Steady-state call cost of every execution layer, plus fallback overhead.

Run with a release build (see ``benchmarks/README.md``)::

    python benchmarks/bench_runtime.py -o results/runtime.json

Benchmark names:

- ``runtime.ops_NNN.<layer>``: one workload with NNN arithmetic operations.
  Layers, outermost first:

  - ``python``: the plain, undecorated function.
  - ``snekit``: the public wrapper, warmed up and verified to be running
    native code before timing starts.
  - ``compiled_i64``: the ``CompiledI64`` object that the wrapper calls,
    called directly (Python-level arity, type and range guards, then the
    native call).
  - ``native_call``: ``I64Function.call(args)`` on a prebuilt tuple, the
    lowest-level native entry point exposed through PyO3.

- ``reference.loop``: an empty statement, the timing loop itself.
- ``reference.python_noop``: a four-argument Python function that returns
  its first argument; the cost of a Python call without arithmetic.
- ``fallback.<case>.python`` and ``fallback.<case>.snekit``: a wrapper in a
  specific state that runs the original function, and the same function
  undecorated.
"""

from __future__ import annotations

import harness
import workloads

# Each timed statement is repeated this many times per loop iteration, so
# pyperf's own ``for`` loop adds well under a nanosecond per call.
DUPLICATE = 10


def main() -> None:
    runner, selected = harness.make_runner()

    def timeit(name: str, statement: str, namespace: dict) -> None:
        if selected(name):
            runner.timeit(name, stmt=statement, globals=namespace, duplicate=DUPLICATE)

    timeit("reference.loop", "pass", {})
    noop = workloads.reference_noop()
    timeit(
        "reference.python_noop",
        "f(a, b, c, d)",
        {"f": noop, **dict(zip(workloads.ARG_NAMES, workloads.ARGS, strict=True))},
    )

    for ops in workloads.OP_COUNTS:
        names = {layer: f"runtime.ops_{ops:03d}.{layer}" for layer in workloads.LAYERS}
        if not any(selected(name) for name in names.values()):
            continue
        # Built, compiled and verified in every process, before any timing.
        layers = workloads.prepare_layers(ops)
        for layer, name in names.items():
            timeit(name, *layers.statement(layer))

    cases = None
    for name in workloads.FALLBACK_CASES:
        for side in ("python", "snekit"):
            full_name = f"fallback.{name}.{side}"
            if not selected(full_name):
                continue
            if cases is None:
                cases = {case.name: case for case in workloads.fallback_cases()}
            timeit(full_name, *cases[name].statement(side))


if __name__ == "__main__":
    main()
