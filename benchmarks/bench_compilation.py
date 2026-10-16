"""One-time costs: analysis, native compilation, warmup and import.

Run with a release build (see ``benchmarks/README.md``)::

    python benchmarks/bench_compilation.py -o results/compilation.json

Benchmark names:

- ``compile.frontend.ops_NNN``: ``analyze(func)``, CPython bytecode to
  normalized IR. It reads the code object and stores nothing, so every
  iteration repeats the full analysis.
- ``compile.backend.ops_NNN``: ``_compiler.compile_i64(instructions, 4)``
  from already produced IR: validation, serialization, Cranelift code
  generation and ``CompiledI64`` construction.
- ``compile.codegen.ops_NNN``: ``_native.compile_i64(program, 4)`` from an
  already serialized program: only the Rust side and Cranelift.
- ``compile.lifecycle.ops_NNN``: decorate the function with a *new*
  ``@snekit`` wrapper and make the warmup calls plus the compiling call,
  exactly what a user's first ``_WARMUP_CALLS + 1`` calls cost.
- ``compile.reject.frontend`` and ``compile.reject.lifecycle``: the same
  for a function the frontend rejects (it ends ``DISABLED``).
- ``import.python``, ``import.speedysnek``, ``import.speedysnek_backend``:
  fresh interpreter processes running ``pass``, ``import speedysnek`` and
  ``import speedysnek._compiler`` (which loads the native extension).

No compilation is ever reused. Each backend iteration creates a new
Cranelift module whose machine code is freed when the result is dropped,
and each lifecycle iteration creates a new wrapper, and with it a new
``_JitState``. Process-wide one-time work (importing the compiler modules,
building the host ISA description in Rust) happens during pyperf's warmup
values, not in the measured ones; ``import.*`` measures it separately.
"""

from __future__ import annotations

import sys

import harness
import workloads
from speedysnek import _compiler, _native
from speedysnek._decorator import _Status
from speedysnek._frontend import analyze


def main() -> None:
    runner, selected = harness.make_runner()

    for ops in workloads.OP_COUNTS:
        suffix = f"ops_{ops:03d}"
        if not any(
            selected(f"compile.{kind}.{suffix}")
            for kind in ("frontend", "backend", "codegen", "lifecycle")
        ):
            continue
        func = workloads.make_workload(ops)
        instructions = analyze(func).instructions
        program = _compiler.serialize(instructions)
        arg_count = func.__code__.co_argcount
        # Verified once per process, outside timing.
        if workloads.status_of(workloads.lifecycle(func, workloads.ARGS)) is not (
            _Status.COMPILED
        ):
            raise workloads.BenchmarkSetupError(f"{suffix} did not compile")

        for kind, statement, namespace in (
            ("frontend", "analyze(f)", {"analyze": analyze, "f": func}),
            (
                "backend",
                "compile_i64(instructions, n)",
                {
                    "compile_i64": _compiler.compile_i64,
                    "instructions": instructions,
                    "n": arg_count,
                },
            ),
            (
                "codegen",
                "compile_i64(program, n)",
                {
                    "compile_i64": _native.compile_i64,
                    "program": program,
                    "n": arg_count,
                },
            ),
            (
                "lifecycle",
                "lifecycle(f, args)",
                {"lifecycle": workloads.lifecycle, "f": func, "args": workloads.ARGS},
            ),
        ):
            name = f"compile.{kind}.{suffix}"
            if selected(name):
                runner.timeit(name, stmt=statement, globals=namespace)

    rejected = workloads.unsupported_workload()
    if workloads.status_of(workloads.lifecycle(rejected, workloads.ARGS)) is not (
        _Status.DISABLED
    ):
        raise workloads.BenchmarkSetupError("the unsupported function compiled")
    if selected("compile.reject.frontend"):
        runner.timeit(
            "compile.reject.frontend",
            stmt="analyze(f)",
            globals={"analyze": analyze, "f": rejected},
        )
    if selected("compile.reject.lifecycle"):
        runner.timeit(
            "compile.reject.lifecycle",
            stmt="lifecycle(f, args)",
            globals={
                "lifecycle": workloads.lifecycle,
                "f": rejected,
                "args": workloads.ARGS,
            },
        )

    for name, code in (
        ("import.python", "pass"),
        ("import.speedysnek", "import speedysnek"),
        ("import.speedysnek_backend", "import speedysnek._compiler"),
    ):
        if selected(name):
            runner.bench_command(name, [sys.executable, "-c", code])


if __name__ == "__main__":
    main()
