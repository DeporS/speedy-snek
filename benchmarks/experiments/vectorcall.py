"""Quick experiment: would vectorcall make ``NativeWrapper`` calls cheaper?

A short, interleaved measurement plus deterministic instruction counts,
not part of the pyperf suite. Results are in ``vectorcall.md`` next to this
file. Run it with release builds, from the repository root::

    python benchmarks/experiments/vectorcall.py \\
        --build main=../main/python --build proto=../proto/python
    python benchmarks/experiments/vectorcall.py --count \\
        --build main=../main/python --build proto=../proto/python

Each ``--build NAME=DIR`` imports ``speedysnek`` from ``DIR`` (a checkout
with its own release build of ``_native`` copied into ``DIR/speedysnek``);
without any, the importable build is measured as ``current``. The first
build is the baseline the others are compared with.

The default mode times every case, alternating processes of each build,
each process interleaving every cell round by round. ``--count`` runs
every case under ``valgrind --tool=callgrind`` instead, twice with
different call counts, so that the difference is the cost of the calls
alone: instructions per call, and how many argument tuples and keyword
dicts each call freed (calls to ``tuple_dealloc`` and ``dict_dealloc``),
which is how the experiment checks whether a call still builds a tuple.

``--calls CASE VARIANT N`` sets one case up and makes ``N`` calls of one
variant, timing nothing; ``--count`` runs it.

Before anything is timed or counted, every case is verified in every
process: the wrapper returns what the plain function returns, with the
same type, its state has the expected status, and a call takes the
expected path (``workloads.call_path``: native, guard miss or Python).
Every wrapper's status is checked again after timing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import workloads  # noqa: E402
from speedysnek import _native, snekit  # noqa: E402
from speedysnek._decorator import _Status  # noqa: E402

VARIANTS = ("python", "snekit")

# ``Py_TPFLAGS_HAVE_VECTORCALL``: whether CPython calls ``NativeWrapper``
# through vectorcall instead of ``tp_call``.
HAVE_VECTORCALL = 1 << 11


class Case:
    """One timed call, its warmup and the path and status it must have.

    ``call`` is the call statement, with ``f`` the callable and ``x0``,
    ``x1``, ... the positional arguments; ``kwargs`` become ``k=k_k``.
    A ``method`` case stores the callable on a class and calls it through
    an instance. A case with ``reset`` follows each call with a call with
    those arguments, so that a run of guard misses never parks it. A case
    whose status is ``PARKED`` is called ``MISS_LIMIT`` times after the
    warmup, which parks a compiled wrapper, as the steady state of calls
    like it.
    """

    def __init__(
        self,
        make: Callable[[], Callable[..., Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        warm: tuple[Any, ...] | None = None,
        path: str = "native",
        status: _Status = _Status.COMPILED,
        method: bool = False,
        star: bool = False,
        reset: tuple[Any, ...] | None = None,
    ) -> None:
        self.make = make
        self.args = args
        self.kwargs = kwargs or {}
        self.warm = args if warm is None else warm
        self.path = path
        self.status = status
        self.method = method
        self.star = star
        self.reset = reset


def _pair(params: str = "a, b") -> Callable[[], Any]:
    return lambda: workloads._pair(params)


def _add() -> Callable[..., Any]:
    return workloads._define("add", "a, b", "a + b")


def _method() -> Callable[..., Any]:
    return workloads._define("add", "self, a, b", "a + b")


CASES: dict[str, Case] = {
    "add": Case(_add, (3, 4)),
    "ops_030": Case(lambda: workloads.make_workload(30), workloads.ARGS),
    "ops_100": Case(lambda: workloads.make_workload(100), workloads.ARGS),
    # A guard miss and then a native call, which starts the miss count
    # again: ``MISS_LIMIT`` misses in a row would park the wrapper.
    "float_miss": Case(
        _pair(), (3.5, 4.25), warm=(3, 4), path="guard_miss", reset=(3, 4)
    ),
    # Keyword calls and omitted defaults never run natively, so a steady
    # stream of them parks the wrapper.
    "keyword": Case(
        _pair(),
        kwargs={"a": 3, "b": 4},
        warm=(3, 4),
        path="python",
        status=_Status.PARKED,
    ),
    "default": Case(
        _pair("a, b=4"), (3,), warm=(3, 4), path="python", status=_Status.PARKED
    ),
    "disabled": Case(
        workloads._rejected_pair, (3, 4), path="python", status=_Status.DISABLED
    ),
    # Compiled, then parked by float calls: the original function runs.
    "parked": Case(
        _pair(), (3.5, 4.25), warm=(3, 4), path="python", status=_Status.PARKED
    ),
    # ``o.f(x0, x1)``: ``self`` is never an int, so the method is parked.
    "method": Case(_method, (3, 4), path="python", status=_Status.PARKED, method=True),
    # ``f(*args)``: CPython calls with a tuple whatever the callee supports.
    "star_args": Case(_add, (3, 4), star=True),
}


class SetupError(Exception):
    """A case would not measure what it claims to."""


class _Box:
    """An instance to call a ``method`` case through."""


def build(name: str) -> dict[str, Callable[..., Any]]:
    """The callable of every variant of one case, verified.

    For a ``method`` case each callable is the bound method.
    """
    case = CASES[name]
    func = case.make()
    callables: dict[str, Callable[..., Any]] = {}
    for variant in VARIANTS:
        f = func if variant == "python" else snekit(func)
        if case.method:
            box = type(f"Box_{variant}", (_Box,), {"f": f})()
            bound = box.f
        else:
            box, bound = None, f
        if variant == "snekit":
            for _ in range(workloads.LIFECYCLE_CALLS):
                bound(*case.warm)
            if case.status is _Status.PARKED:
                for _ in range(_native.MISS_LIMIT):
                    bound(*case.args, **case.kwargs)
            verify(f"{variant}/{name}", f, func, case, box)
        callables[variant] = bound
    return callables


def verify(
    where: str, wrapper: Any, func: Callable[..., Any], case: Case, box: Any
) -> None:
    state = wrapper._jit_state
    if state.status is not case.status:
        raise SetupError(f"{where}: status {state.status}")
    prefix = (box,) if case.method else ()
    expected = func(*prefix, *case.args, **case.kwargs)
    actual = wrapper(*prefix, *case.args, **case.kwargs)
    if actual != expected or type(actual) is not type(expected):
        raise SetupError(f"{where}: result {actual!r}, expected {expected!r}")
    path = workloads.call_path(wrapper, *prefix, *case.args, **case.kwargs)
    if path != case.path:
        raise SetupError(f"{where}: path {path!r}, expected {case.path!r}")


def vectorcall() -> bool:
    """Whether this build's ``NativeWrapper`` supports vectorcall."""
    return bool(_native.NativeWrapper.__flags__ & HAVE_VECTORCALL)


def statement(f: Callable[..., Any], case: Case) -> tuple[str, dict[str, Any]]:
    names = [f"x{i}" for i in range(len(case.args))]
    namespace: dict[str, Any] = {"f": f}
    namespace.update((f"x{i}", value) for i, value in enumerate(case.args))
    namespace.update((f"k_{key}", value) for key, value in case.kwargs.items())
    if case.star:
        namespace["xs"] = case.args
        return "f(*xs)", namespace
    names += [f"{key}=k_{key}" for key in case.kwargs]
    call = f"f({', '.join(names)})"
    if case.reset is not None:
        namespace.update((f"r{i}", value) for i, value in enumerate(case.reset))
        resets = ", ".join(f"r{i}" for i in range(len(case.reset)))
        call += f"; f({resets})"
    return call, namespace


def compile_loop(statement: str, namespace: dict[str, Any], unroll: int) -> Any:
    body = "\n".join(f"    {statement}" for _ in range(unroll))
    scope = dict(namespace)
    exec(f"def _run(_n):\n  for _ in range(_n):\n{body}\n", scope)
    return scope["_run"]


def time_call(run: Any, loops: int) -> float:
    """Nanoseconds per call of a loop unrolled 10 times."""
    start = time.perf_counter_ns()
    run(loops)
    return (time.perf_counter_ns() - start) / (loops * 10)


def measure(names: list[str], rounds: int, loops: int) -> dict[str, float]:
    """Median ns per call of every ``case/variant`` cell, in this process."""
    grid = []
    for name in names:
        for variant, f in build(name).items():
            run = compile_loop(*statement(f, CASES[name]), 10)
            grid.append((f"{name}/{variant}", f, run, CASES[name]))
    samples: dict[str, list[float]] = {cell: [] for cell, *_ in grid}
    for _, _, run, _ in grid:
        time_call(run, max(1, loops // 4))  # warm the code
    for _ in range(rounds):
        for cell, _, run, _ in grid:
            samples[cell].append(time_call(run, loops))
    for cell, f, _, case in grid:
        wrapper = getattr(f, "__func__", f)
        if cell.endswith("/snekit") and wrapper._jit_state.status is not case.status:
            raise SetupError(f"{cell}: the state changed while it was timed")
    return {cell: statistics.median(values) for cell, values in samples.items()}


def calls(name: str, variant: str, n: int) -> None:
    """Set one case up, then make ``n`` calls of one variant."""
    f = build(name)[variant]
    compile_loop(*statement(f, CASES[name]), 1)(n)


def worker(options: argparse.Namespace) -> None:
    import speedysnek

    if _native.DEBUG_BUILD:
        sys.exit("debug build of speedysnek._native: rebuild with --release")
    medians = measure(options.cases, options.rounds, options.loops)
    report = {"path": speedysnek.__file__, "vectorcall": vectorcall()}
    print(json.dumps({**report, "medians": medians}))


def _env(path: str | None) -> dict[str, str]:
    env = dict(os.environ)
    if path is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            [path, *filter(None, [env.get("PYTHONPATH")])]
        )
    return env


def _check_import(report: dict[str, Any], path: str | None) -> None:
    if path is not None and not report["path"].startswith(os.path.abspath(path)):
        sys.exit(f"{path}: imported the wrong build: {report['path']}")


def run_worker(options: argparse.Namespace, path: str | None) -> dict[str, Any]:
    command = [sys.executable, __file__, "--worker"]
    command += ["--rounds", str(options.rounds), "--loops", str(options.loops)]
    command += ["--cases", *options.cases]
    if options.cpu is not None:
        command = ["taskset", "-c", str(options.cpu), *command]
    result = subprocess.run(command, env=_env(path), capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(result.stderr or result.stdout)
    report = json.loads(result.stdout)
    _check_import(report, path)
    return report


def callgrind_calls(path: str) -> tuple[int, Counter[str]]:
    """Total instructions and the calls to each function, from a
    ``callgrind.out`` file (with its default name compression)."""
    names: dict[str, str] = {}
    counts: Counter[str] = Counter()
    total = 0
    callee = None
    with open(path) as lines:
        for line in lines:
            if line.startswith(("fn=", "cfn=")):
                key, _, rest = line.rstrip("\n").partition("=")
                match = re.fullmatch(r"\((\d+)\)(?: (.*))?", rest)
                if match is None:
                    name = rest
                else:
                    if match.group(2) is not None:
                        names[match.group(1)] = match.group(2)
                    name = names[match.group(1)]
                callee = name if key == "cfn" else None
            elif line.startswith("calls=") and callee is not None:
                counts[callee] += int(line[len("calls=") :].split()[0])
                callee = None
            elif line.startswith(("summary:", "totals:")):
                total = max(total, int(line.split()[1]))
    return total, counts


def count(name: str, path: str | None, n: int) -> dict[str, float]:
    """Per-call instructions, tuples and dicts freed, for ``snekit``."""
    runs = []
    for calls_made in (n, 2 * n):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "callgrind.out")
            command = [
                "valgrind",
                "--tool=callgrind",
                f"--callgrind-out-file={out}",
                sys.executable,
                __file__,
                "--calls",
                name,
                "snekit",
                str(calls_made),
            ]
            env = {**_env(path), "PYTHONHASHSEED": "0"}
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                sys.exit(result.stderr)
            runs.append(callgrind_calls(out))
    (total_n, calls_n), (total_2n, calls_2n) = runs
    return {
        "instructions": (total_2n - total_n) / n,
        "tuples": (calls_2n["tuple_dealloc"] - calls_n["tuple_dealloc"]) / n,
        "dicts": (calls_2n["dict_dealloc"] - calls_n["dict_dealloc"]) / n,
    }


def summary(values: list[float]) -> str:
    return f"{statistics.median(values):.0f} ({min(values):.0f}–{max(values):.0f})"


def parse_builds(options: argparse.Namespace) -> dict[str, str | None]:
    builds: dict[str, str | None] = {}
    for spec in options.build or []:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            sys.exit(f"--build {spec!r}: expected NAME=DIR")
        builds[name] = os.path.abspath(path)
    return builds or {"current": None}


def probe(path: str | None) -> dict[str, Any]:
    """Which ``speedysnek`` a build imports, and whether it vectorcalls."""
    code = (
        "import json, speedysnek, speedysnek._native as n;"
        "print(json.dumps({'path': speedysnek.__file__,"
        f"'vectorcall': bool(n.NativeWrapper.__flags__ & {HAVE_VECTORCALL})}}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=_env(path), capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.exit(result.stderr)
    report = json.loads(result.stdout)
    _check_import(report, path)
    return report


def print_counts(options: argparse.Namespace, builds: dict[str, str | None]) -> None:
    flags = {name: probe(path)["vectorcall"] for name, path in builds.items()}
    print(f"# instructions, tuples and dicts freed per call; vectorcall: {flags}")
    print()
    header = ["case"]
    for name in builds:
        header += [f"{name} instr", f"{name} tuples", f"{name} dicts"]
    base = next(iter(builds))
    header += [f"{name} / {base}" for name in list(builds)[1:]]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for case in options.cases:
        counts = {name: count(case, path, options.n) for name, path in builds.items()}
        row = [case]
        for name in builds:
            c = counts[name]
            row += [f"{c['instructions']:.0f}", f"{c['tuples']:g}", f"{c['dicts']:g}"]
        for name in list(builds)[1:]:
            ratio = counts[name]["instructions"] / counts[base]["instructions"]
            row.append(f"{ratio:.3f}")
        print("| " + " | ".join(row) + " |", flush=True)


def print_timings(options: argparse.Namespace, builds: dict[str, str | None]) -> None:
    runs: dict[str, list[dict[str, float]]] = {name: [] for name in builds}
    flags: dict[str, bool] = {}
    for _ in range(options.processes):
        for name, path in builds.items():
            report = run_worker(options, path)
            flags[name] = report["vectorcall"]
            runs[name].append(report["medians"])
    print(
        f"# {platform.python_implementation()} {platform.python_version()}, "
        f"{platform.machine()}, {options.processes} process(es) per build x "
        f"{options.rounds} interleaved rounds x {options.loops * 10} calls; "
        f"vectorcall: {flags}"
    )
    print()
    base = next(iter(builds))
    header = ["case", f"python ({base})"]
    header += [f"{name} snekit" for name in builds]
    header += [f"{name} / {base}" for name in list(builds)[1:]]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for case in options.cases:
        cells = {name: [run[f"{case}/snekit"] for run in runs[name]] for name in builds}
        row = [case, summary([run[f"{case}/python"] for run in runs[base]])]
        row += [summary(cells[name]) for name in builds]
        for name in list(builds)[1:]:
            ratio = statistics.median(cells[name]) / statistics.median(cells[base])
            row.append(f"{ratio:.2f}")
        print("| " + " | ".join(row) + " |")
    print(
        "\nMedian of process medians, ns per call (min–max over processes); "
        "ratios are of the medians."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--loops", type=int, default=20000)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES)
    parser.add_argument("--cpu", type=int, help="pin every process to this CPU")
    parser.add_argument(
        "--build",
        action="append",
        metavar="NAME=DIR",
        help="time the speedysnek importable from DIR; repeat for each build",
    )
    parser.add_argument(
        "--count", action="store_true", help="count instructions with callgrind"
    )
    parser.add_argument(
        "-n", type=int, default=100_000, help="calls per --count run (and twice that)"
    )
    parser.add_argument(
        "--calls",
        nargs=3,
        metavar=("CASE", "VARIANT", "N"),
        help="set CASE up, make N calls of VARIANT, time nothing",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.calls is not None:
        name, variant, n = options.calls
        calls(name, variant, int(n))
        return
    if options.worker:
        worker(options)
        return
    builds = parse_builds(options)
    if options.count:
        print_counts(options, builds)
    else:
        print_timings(options, builds)


if __name__ == "__main__":
    main()
