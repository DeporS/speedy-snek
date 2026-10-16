"""Turn pyperf JSON results into a Markdown report.

Usage::

    python benchmarks/report.py results/runtime.json results/compilation.json \\
        --command "<the benchmark commands you ran>" \\
        --interpretation benchmarks/interpretation.md \\
        -o benchmarks/RESULTS.md

``--command`` and ``--interpretation`` are optional; ``benchmarks/README.md``
has the exact commands used for the committed baseline.

Every time is pyperf's mean (with its standard deviation) of one call or
one compilation. Ratios are ``layer / Python``: below 1.0 means the measured
JIT path is faster than plain Python, above 1.0 means it is slower.

The calculation helpers (:func:`ratio`, :func:`break_even_calls`,
:func:`linear_fit`, :func:`stability_warnings`, ...) are pure functions with
unit tests; only :func:`load` needs pyperf.
"""

from __future__ import annotations

import argparse
import datetime
import math
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import workloads

NO_BREAK_EVEN = "No break-even with the current hot path."

LAYER_TITLES = {
    "python": "Python",
    "snekit": "`@snekit`",
    "compiled_i64": "`CompiledI64`",
    "native_call": "raw native",
}

# pyperf metadata worth showing, in display order.
PYPERF_METADATA = (
    ("perf_version", "pyperf version"),
    ("timer", "Timer"),
    ("cpu_count", "Logical CPUs"),
    ("cpu_affinity", "CPU affinity"),
    ("cpu_config", "CPU configuration"),
    ("cpu_freq", "CPU frequency"),
    ("aslr", "ASLR"),
    ("load_avg_1min", "Load average (1 min)"),
    ("runnable_threads", "Runnable threads"),
    ("python_compiler", "Python compiler"),
    ("python_hash_seed", "Python hash seed"),
)


# --- Data ---------------------------------------------------------------


@dataclass(frozen=True)
class Stats:
    """Summary of one pyperf benchmark. All times are in seconds."""

    values: tuple[float, ...]
    shortest_raw: float | None = None
    processes: int = 1
    required_processes: int | None = None

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) >= 2 else 0.0


@dataclass
class Results:
    benchmarks: dict[str, Stats] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    dates: tuple[str, str] | None = None
    shape: str | None = None
    # Every ``git_commit`` recorded by any run, in first-seen order.
    commits: list[str] = field(default_factory=list)

    def get(self, name: str) -> Stats | None:
        return self.benchmarks.get(name)

    def mean(self, name: str) -> float | None:
        stats = self.benchmarks.get(name)
        return None if stats is None else stats.mean


def load(paths: Iterable[str | Path]) -> Results:
    """Read pyperf JSON (or ``.json.gz``) files into :class:`Results`."""
    import pyperf

    results = Results()
    starts, ends, shapes = [], [], set()
    for path in paths:
        suite = pyperf.BenchmarkSuite.load(str(path))
        for key, value in suite.get_metadata().items():
            results.metadata.setdefault(key, value)
        for bench in suite.get_benchmarks():
            runs = [run for run in bench.get_runs() if run.values]
            for run in runs:
                commit = run.get_metadata().get("git_commit")
                if commit and commit not in results.commits:
                    results.commits.append(commit)
            raw = [
                value * run.get_total_loops() for run in runs for value in run.values
            ]
            required = getattr(bench, "required_nprocesses", lambda: None)()
            results.benchmarks[bench.get_name()] = Stats(
                values=tuple(bench.get_values()),
                shortest_raw=min(raw) if raw else None,
                processes=len(runs),
                required_processes=required,
            )
            dates = bench.get_dates()
            if dates:
                starts.append(dates[0])
                ends.append(dates[1])
            shapes.add(
                (
                    len(runs),
                    len(runs[0].values),
                    runs[0].warmups and len(runs[0].warmups),
                )
            )
    if starts:
        results.dates = (
            min(starts).isoformat(sep=" ", timespec="seconds"),
            max(ends).isoformat(sep=" ", timespec="seconds"),
        )
    if shapes:
        results.shape = "; ".join(
            f"{processes} worker processes × {values} values (+ {warmups or 0} warmup)"
            for processes, values, warmups in sorted(shapes)
        )
    return results


# --- Calculations -------------------------------------------------------


def ratio(value: float, baseline: float) -> float:
    """``value / baseline``; below 1.0 means ``value`` is faster."""
    if not baseline > 0:
        raise ValueError(f"baseline must be positive, got {baseline!r}")
    if value < 0:
        raise ValueError(f"value must not be negative, got {value!r}")
    return value / baseline


def break_even_calls(
    compilation_cost: float, python_time: float, jit_time: float
) -> float | None:
    """Calls after which a one-time cost is paid back, or ``None``.

    ``compilation_cost / (python_time - jit_time)``, all in the same unit.
    There is no break-even (``None``) unless the JIT path is strictly
    faster, so the result is never negative or infinite.
    """
    if compilation_cost < 0 or python_time < 0 or jit_time < 0:
        raise ValueError("times must not be negative")
    saving = python_time - jit_time
    if saving <= 0:
        return None
    return compilation_cost / saving


def format_break_even(calls: float | None) -> str:
    if calls is None:
        return NO_BREAK_EVEN
    return f"≈ {math.ceil(calls):,} calls"


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Least-squares ``(intercept, slope)`` of ``y = intercept + slope * x``."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need at least two (x, y) pairs")
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    spread = sum((x - mean_x) ** 2 for x in xs)
    if spread == 0:
        raise ValueError("x values must not all be equal")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    slope /= spread
    return mean_y - slope * mean_x, slope


def stability_warnings(name: str, stats: Stats) -> list[str]:
    """pyperf's own stability checks, applied to one benchmark."""
    warnings = []
    mean = stats.mean
    if len(stats.values) >= 2 and mean > 0:
        percent = stats.stdev * 100 / mean
        if percent >= 10:
            warnings.append(f"{name}: std dev is {percent:.0f}% of the mean")
        elif (
            stats.required_processes is not None
            and stats.required_processes > stats.processes
        ):
            warnings.append(
                f"{name}: not enough processes for a stable result "
                f"({stats.processes} run, {stats.required_processes} suggested)"
            )
    if mean > 0:
        for label, value in (
            ("minimum", min(stats.values)),
            ("maximum", max(stats.values)),
        ):
            percent = (value - mean) * 100 / mean
            if abs(percent) >= 50:
                warnings.append(f"{name}: {label} is {percent:+.0f}% from the mean")
    if stats.shortest_raw is not None and stats.shortest_raw < 1e-3:
        warnings.append(
            f"{name}: shortest raw value is only {format_time(stats.shortest_raw)}"
        )
    return warnings


# --- Formatting ---------------------------------------------------------


def format_time(seconds: float | None) -> str:
    """Format a duration with a unit suited to its size."""
    if seconds is None:
        return "—"
    magnitude = abs(seconds)
    for limit, scale, unit in (
        (1e-6, 1e9, "ns"),
        (1e-3, 1e6, "µs"),
        (1.0, 1e3, "ms"),
    ):
        if magnitude < limit:
            return f"{seconds * scale:,.1f} {unit}"
    return f"{seconds:,.2f} s"


def format_ns(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds * 1e9:,.1f}"


def format_us(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds * 1e6:,.1f}"


def format_stat(stats: Stats | None, formatter=format_ns) -> str:
    if stats is None:
        return "—"
    return f"{formatter(stats.mean)} ± {formatter(stats.stdev)}"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "—"
    verdict = "faster" if value < 1 else "slower" if value > 1 else "equal"
    return f"{value:.2f}× ({verdict})"


def _ratio_or_none(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0:
        return None
    return ratio(value, baseline)


def _table(
    header: Sequence[str], rows: Iterable[Sequence[str]], numeric: bool = True
) -> list[str]:
    """A Markdown table, right-aligning all but the first column if ``numeric``."""
    lines = ["| " + " | ".join(header) + " |"]
    align = ["---"] + ["--:" if numeric else "---"] * (len(header) - 1)
    lines.append("|" + "|".join(align) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _workload(ops: int) -> str:
    return f"`ops_{ops:03d}`"


# --- Report sections ----------------------------------------------------


def _commits(results: Results) -> str:
    if len(results.commits) > 1:
        # pyperf drops metadata that differs between runs; show all of it.
        return ", ".join(results.commits) + " (differs between runs)"
    if results.commits:
        return results.commits[0]
    return str(results.metadata.get("git_commit", "unknown"))


def _python_description(meta: dict[str, object]) -> str:
    implementation = meta.get("python_implementation", "")
    version = meta.get("python_version", "")
    return f"{implementation} {version}".strip() or "unknown"


def _environment(
    results: Results, commands: Sequence[str], generated: str
) -> list[str]:
    meta = results.metadata
    build = meta.get("speedysnek_build", "unknown")
    if build == "debug":
        build = "**debug — not representative**"
    rows = [
        ("Date", " to ".join(results.dates) if results.dates else "unknown"),
        ("Report generated", generated),
        ("Operating system", meta.get("platform", "unknown")),
        ("CPU", meta.get("cpu_model_name", "unknown")),
        ("Architecture", meta.get("machine", "unknown")),
        ("Python", _python_description(meta)),
        ("CPython's own JIT", meta.get("cpython_jit", "unknown")),
        ("Rust", meta.get("rustc_version", "not available")),
        ("speedysnek native build", build),
        ("speedysnek commit", _commits(results)),
        ("`@snekit` warmup calls", meta.get("speedysnek_warmup_calls", "unknown")),
    ]
    lines = ["## Environment", ""]
    lines += _table(
        ("", ""), ((f"**{key}**", str(value)) for key, value in rows), numeric=False
    )
    lines += ["", "### Benchmark command", "", "```sh"]
    lines += list(commands) or ["# not recorded"]
    lines += ["```", "", "### pyperf metadata relevant to interpretation", ""]
    if results.shape:
        lines.append(f"- **Samples per benchmark:** {results.shape}")
    for key, title in PYPERF_METADATA:
        if key in meta:
            lines.append(f"- **{title}:** {meta[key]}")
    return lines


def _runtime(results: Results) -> list[str]:
    rows = []
    for ops in workloads.OP_COUNTS:
        prefix = f"runtime.ops_{ops:03d}."
        stats = {layer: results.get(prefix + layer) for layer in workloads.LAYERS}
        if not any(stats.values()):
            continue
        python = results.mean(prefix + "python")
        rows.append(
            (
                _workload(ops),
                *(format_stat(stats[layer]) for layer in workloads.LAYERS),
                format_ratio(_ratio_or_none(results.mean(prefix + "snekit"), python)),
                format_ratio(
                    _ratio_or_none(results.mean(prefix + "native_call"), python)
                ),
            )
        )
    lines = [
        "## Steady-state runtime",
        "",
        "Nanoseconds per call (mean ± std dev). `@snekit` was warmed up and "
        "verified to run native code before timing. Ratios are layer / "
        "Python: **below 1.0 means the JIT path is faster, above 1.0 means "
        "it is slower.**",
        "",
    ]
    lines += _table(
        (
            "Workload",
            "Python",
            "`@snekit`",
            "`CompiledI64`",
            "raw native",
            "`@snekit` / Python",
            "raw native / Python",
        ),
        rows,
    )
    references = [
        (name, label)
        for name, label in (
            ("reference.loop", "empty statement (timing loop residue)"),
            (
                "reference.python_noop",
                "Python call of `noop(a, b, c, d)`, no arithmetic",
            ),
        )
        if results.get(name)
    ]
    if references:
        lines += ["", "References:", ""]
        lines += [
            f"- {label}: {format_stat(results.get(name))} ns"
            for name, label in references
        ]
    return lines


def breakdown(results: Results, ops: int) -> dict[str, float] | None:
    """Split one warmed ``@snekit`` call into its layers, in seconds."""
    prefix = f"runtime.ops_{ops:03d}."
    snekit = results.mean(prefix + "snekit")
    compiled = results.mean(prefix + "compiled_i64")
    native = results.mean(prefix + "native_call")
    if None in (snekit, compiled, native):
        return None
    return {
        "wrapper": snekit - compiled,
        "guards": compiled - native,
        "native": native,
        "total": snekit,
    }


COMPONENTS = {
    "wrapper": "decorator wrapper",
    "guards": "`CompiledI64` Python guards",
    "native": "raw native call",
}


def _breakdown(results: Results) -> list[str]:
    rows = []
    for ops in workloads.OP_COUNTS:
        parts = breakdown(results, ops)
        if parts is None:
            continue
        total = parts["total"]
        rows.append(
            (
                _workload(ops),
                format_ns(total),
                *(
                    f"{format_ns(parts[key])} ({parts[key] * 100 / total:.0f}%)"
                    for key in COMPONENTS
                ),
            )
        )
    if not rows:
        return []
    lines = [
        "## Where a warmed `@snekit` call spends its time",
        "",
        "Nanoseconds per call, as differences between adjacent layers:",
        "",
        "- **decorator wrapper** = `@snekit` − `CompiledI64`: the closure, "
        "`call_count`, the specialization and `__code__` checks, the `*args` "
        "and `**kwargs` packing and the `try`.",
        "- **`CompiledI64` Python guards** = `CompiledI64` − raw native: the "
        "Python-level `__call__`, its arity, exact-`int` and i64 range "
        "checks, and the status check on the result.",
        "- **raw native call** = `I64Function.call(args)`: method call through "
        "PyO3, tuple and int conversion, a `Vec` allocation, the machine "
        "code itself, and building the `(status, value)` result tuple.",
        "",
    ]
    lines += _table(("Workload", "`@snekit` total", *COMPONENTS.values()), rows)
    return lines


def fits(results: Results) -> dict[str, tuple[float, float]]:
    """``layer -> (fixed cost, cost per operation)`` in seconds."""
    fitted = {}
    for layer in workloads.LAYERS:
        points = [
            (ops, results.mean(f"runtime.ops_{ops:03d}.{layer}"))
            for ops in workloads.OP_COUNTS
        ]
        points = [(x, y) for x, y in points if y is not None]
        if len({x for x, _ in points}) >= 2:
            fitted[layer] = linear_fit([x for x, _ in points], [y for _, y in points])
    return fitted


def _scaling(results: Results) -> list[str]:
    fitted = fits(results)
    if not fitted:
        return []
    lines = [
        "## Fixed cost versus cost per operation",
        "",
        "A least-squares line `time = fixed + per_op × operations` through "
        "all runtime workloads of each layer. The fixed part is what every "
        "call pays regardless of size; the slope is the marginal cost of one "
        "more `+`, `-` or `*`.",
        "",
    ]
    lines += _table(
        ("Layer", "fixed cost per call (ns)", "cost per operation (ns)"),
        (
            (LAYER_TITLES[layer], format_ns(fixed), f"{slope * 1e9:,.3f}")
            for layer, (fixed, slope) in fitted.items()
        ),
    )
    return lines


def _fallback(results: Results) -> list[str]:
    rows = []
    descriptions = _fallback_descriptions()
    for case in workloads.FALLBACK_CASES:
        python = results.get(f"fallback.{case}.python")
        snekit = results.get(f"fallback.{case}.snekit")
        if python is None and snekit is None:
            continue
        overhead = (
            None if python is None or snekit is None else snekit.mean - python.mean
        )
        rows.append(
            (
                f"`{case}`",
                descriptions.get(case, ""),
                format_stat(python),
                format_stat(snekit),
                format_ns(overhead),
                format_ratio(
                    _ratio_or_none(snekit and snekit.mean, python and python.mean)
                ),
            )
        )
    if not rows:
        return []
    lines = [
        "## Fallback and wrapper overhead",
        "",
        "Nanoseconds per call. Each `@snekit` row runs the original function; "
        "its Python baseline is an undecorated copy of that same function "
        "called with the same arguments, so the overhead column is the cost "
        "of the wrapper alone.",
        "",
    ]
    lines += _table(
        (
            "Case",
            "What happens",
            "Python",
            "`@snekit`",
            "overhead",
            "`@snekit` / Python",
        ),
        rows,
    )
    return lines


def _fallback_descriptions() -> dict[str, str]:
    return {
        "cold": "warmup call: counted, then Python",
        "disabled": "rejected function (`//`), after it was disabled",
        "float_args": "compiled; `float` arguments fail the exact-`int` guard",
        "big_int_args": "compiled; an `int` outside i64 fails the range guard",
        "overflow": "compiled; native `*` overflows, Python recomputes",
        "keyword_args": "compiled; keyword arguments skip native code",
        "default_args": "compiled; an omitted default skips native code",
    }


def _compilation(results: Results) -> list[str]:
    rows = []
    for ops in workloads.OP_COUNTS:
        names = [
            f"compile.{kind}.ops_{ops:03d}"
            for kind in ("frontend", "backend", "codegen", "lifecycle")
        ]
        if not any(results.get(name) for name in names):
            continue
        rows.append(
            (
                _workload(ops),
                *(format_stat(results.get(name), format_us) for name in names),
            )
        )
    lines = []
    if rows:
        lines = [
            "## Compilation and warmup",
            "",
            "Microseconds per compilation (mean ± std dev). Nothing is "
            "reused between iterations: every backend iteration builds a new "
            "Cranelift module and every lifecycle iteration a new wrapper.",
            "",
            "- **frontend**: `analyze(func)`, bytecode to normalized IR.",
            "- **native compilation (from IR)**: `_compiler.compile_i64`: "
            "validation, serialization, Cranelift and `CompiledI64`.",
            "- **codegen only**: `_native.compile_i64` on an already "
            "serialized program.",
            "- **full lifecycle**: a new `@snekit` wrapper through "
            f"{workloads.LIFECYCLE_CALLS - 1} warmup calls and the compiling "
            "call.",
            "",
        ]
        lines += _table(
            (
                "Workload",
                "frontend",
                "native compilation (from IR)",
                "codegen only",
                "full lifecycle",
            ),
            rows,
        )
    extra = [
        (label, results.get(name))
        for name, label in (
            (
                "compile.reject.frontend",
                "Frontend rejection of an unsupported function",
            ),
            (
                "compile.reject.lifecycle",
                "Full lifecycle of an unsupported function (ends disabled)",
            ),
        )
        if results.get(name)
    ]
    if extra:
        lines += [""] + [
            f"- {label}: {format_stat(s, format_us)} µs" for label, s in extra
        ]

    startup = results.mean("import.python")
    imports = [
        (label, results.get(name))
        for name, label in (
            ("import.python", "`python -c pass`"),
            ("import.speedysnek", "`python -c 'import speedysnek'`"),
            ("import.speedysnek_backend", "`python -c 'import speedysnek._compiler'`"),
        )
        if results.get(name)
    ]
    if imports:
        lines += [
            "",
            "### Cold import (fresh process)",
            "",
        ]
        lines += _table(
            ("Command", "wall time (ms)", "above bare startup (ms)"),
            (
                (
                    label,
                    format_stat(stats, lambda s: f"{s * 1e3:,.1f}"),
                    "—"
                    if startup is None or label == "`python -c pass`"
                    else f"{(stats.mean - startup) * 1e3:,.1f}",
                )
                for label, stats in imports
            ),
        )
    return lines


def break_even_rows(results: Results) -> list[tuple[int, float, float, float | None]]:
    """``(ops, lifecycle cost, saving per call, break-even calls)``."""
    rows = []
    for ops in workloads.OP_COUNTS:
        cost = results.mean(f"compile.lifecycle.ops_{ops:03d}")
        python = results.mean(f"runtime.ops_{ops:03d}.python")
        snekit = results.mean(f"runtime.ops_{ops:03d}.snekit")
        if None in (cost, python, snekit):
            continue
        rows.append(
            (ops, cost, python - snekit, break_even_calls(cost, python, snekit))
        )
    return rows


def _break_even(results: Results) -> list[str]:
    rows = break_even_rows(results)
    if not rows:
        return []
    lines = [
        "## Break-even",
        "",
        "`full lifecycle cost / (Python time per call − @snekit time per "
        "call)`, computed in seconds. The lifecycle cost includes the warmup "
        "calls, so this slightly overestimates the break-even point. It is "
        "only computed when the warmed `@snekit` call is faster than Python.",
        "",
    ]
    lines += _table(
        ("Workload", "lifecycle cost", "Python − `@snekit` per call", "break-even"),
        (
            (
                _workload(ops),
                format_time(cost),
                format_time(saving),
                format_break_even(calls),
            )
            for ops, cost, saving, calls in rows
        ),
    )
    return lines


def key_measurements(results: Results) -> list[str]:
    """Factual, generated statements about the results."""
    bullets = []
    runtime = []
    for ops in workloads.OP_COUNTS:
        prefix = f"runtime.ops_{ops:03d}."
        python = results.mean(prefix + "python")
        snekit = results.mean(prefix + "snekit")
        native = results.mean(prefix + "native_call")
        if None not in (python, snekit, native):
            runtime.append((ops, python, snekit, native))
    if runtime:
        faster = [ops for ops, python, snekit, _ in runtime if snekit < python]
        if faster:
            bullets.append(
                "Warmed `@snekit` is faster than plain Python for "
                + ", ".join(_workload(ops) for ops in faster)
                + f" ({len(faster)} of {len(runtime)} workloads)."
            )
        else:
            bullets.append(
                "Warmed `@snekit` is **slower than plain Python in every "
                f"measured workload** ({len(runtime)} of {len(runtime)})."
            )
        ratios = [snekit / python for _, python, snekit, _ in runtime]
        bullets.append(
            f"`@snekit` / Python ranges from {min(ratios):.2f}× to {max(ratios):.2f}×."
        )
        native_faster = [ops for ops, python, _, native in runtime if native < python]
        if native_faster:
            bullets.append(
                "The raw native call alone is faster than the Python function for "
                + ", ".join(_workload(ops) for ops in native_faster)
                + "."
            )
        else:
            bullets.append("Even the raw native call is slower than Python everywhere.")

    for ops in (workloads.OP_COUNTS[0], workloads.OP_COUNTS[-1]):
        parts = breakdown(results, ops)
        if parts:
            key = max(COMPONENTS, key=parts.__getitem__)
            bullets.append(
                f"For {_workload(ops)}, the largest part of a `@snekit` call is "
                f"the {COMPONENTS[key]}: {format_ns(parts[key])} ns of "
                f"{format_ns(parts['total'])} ns "
                f"({parts[key] * 100 / parts['total']:.0f}%)."
            )

    fitted = fits(results)
    if "python" in fitted and "native_call" in fitted:
        (py_fixed, py_slope), (nat_fixed, nat_slope) = (
            fitted["python"],
            fitted["native_call"],
        )
        bullets.append(
            f"Marginal cost per operation: Python {py_slope * 1e9:.2f} ns, raw "
            f"native {nat_slope * 1e9:.3f} ns. Fixed cost per call: Python "
            f"{format_ns(py_fixed)} ns, raw native {format_ns(nat_fixed)} ns"
            + (
                f", `@snekit` {format_ns(fitted['snekit'][0])} ns."
                if "snekit" in fitted
                else "."
            )
        )

    rows = break_even_rows(results)
    if rows:
        even = [(ops, calls) for ops, _, _, calls in rows if calls is not None]
        if even:
            bullets.append(
                "Break-even: "
                + ", ".join(
                    f"{_workload(ops)} {format_break_even(c)}" for ops, c in even
                )
                + "."
            )
        else:
            bullets.append(f"Break-even: none. {NO_BREAK_EVEN}")

    overheads = []
    for case in workloads.FALLBACK_CASES:
        python = results.mean(f"fallback.{case}.python")
        snekit = results.mean(f"fallback.{case}.snekit")
        if python is not None and snekit is not None:
            overheads.append((snekit - python, case))
    if overheads:
        low, high = min(overheads), max(overheads)
        bullets.append(
            f"Fallback overhead ranges from {format_ns(low[0])} ns (`{low[1]}`) "
            f"to {format_ns(high[0])} ns (`{high[1]}`) per call."
        )
    return bullets


def _stability(results: Results) -> list[str]:
    warnings = [
        warning
        for name, stats in results.benchmarks.items()
        for warning in stability_warnings(name, stats)
    ]
    lines = ["## Stability", ""]
    if warnings:
        lines.append(
            "pyperf's stability checks flagged these benchmarks; treat their "
            "numbers with extra caution:"
        )
        lines.append("")
        lines += [f"- {warning}" for warning in warnings]
    else:
        lines.append("No benchmark triggered pyperf's stability checks.")
    return lines


def render(
    results: Results,
    commands: Sequence[str] = (),
    interpretation: str | None = None,
    generated: str | None = None,
) -> str:
    generated = generated or datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    lines = [
        "# speedy-snek benchmark results",
        "",
        "> **This is one machine-specific baseline, not a universal "
        "performance claim.** Absolute numbers depend on the CPU, operating "
        "system, Python build and system load. Regenerate it on your own "
        "machine with the commands below; see `benchmarks/README.md`.",
        "",
    ]
    if results.metadata.get("speedysnek_build") == "debug":
        lines += [
            "> **Warning: these results come from a debug build of the native "
            "extension and are not representative.**",
            "",
        ]
    sections = [
        _environment(results, commands, generated),
        ["## Key measurements", ""] + [f"- {b}" for b in key_measurements(results)],
        _runtime(results),
        _breakdown(results),
        _scaling(results),
        _fallback(results),
        _compilation(results),
        _break_even(results),
        _stability(results),
    ]
    if interpretation:
        sections.append(["## Interpretation", "", interpretation.strip()])
    for section in sections:
        if section:
            lines += section + [""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", nargs="+", help="pyperf JSON result files")
    parser.add_argument("-o", "--output", help="write Markdown here (default: stdout)")
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="a command used to produce the results (repeatable)",
    )
    parser.add_argument(
        "--interpretation",
        help="Markdown file appended as the report's Interpretation section",
    )
    args = parser.parse_args(argv)
    results = load(args.results)
    interpretation = (
        Path(args.interpretation).read_text(encoding="utf-8")
        if args.interpretation
        else None
    )
    text = render(results, args.command, interpretation)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
