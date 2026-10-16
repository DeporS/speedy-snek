"""Report calculations and rendering, with synthetic results.

Nothing here measures or compares real timings.
"""

import math
import subprocess
import sys
from pathlib import Path

import pytest

import report
import workloads
from report import (
    NO_BREAK_EVEN,
    Results,
    Stats,
    break_even_calls,
    change,
    crossover,
    format_break_even,
    format_ratio,
    format_time,
    linear_fit,
    ratio,
    render,
    stability_warnings,
)

BENCHMARKS = Path(__file__).resolve().parent.parent


def steady(seconds):
    """A perfectly stable benchmark with mean ``seconds``."""
    return Stats(values=(seconds,) * 6, shortest_raw=0.1, processes=20)


def synthetic(python_ns, snekit_ns, *, lifecycle_us=500.0, native_ns=None):
    """Results where every workload has the same per-call times."""
    native_ns = native_ns if native_ns is not None else snekit_ns / 4
    benchmarks = {}
    for ops in workloads.OP_COUNTS:
        prefix = f"runtime.ops_{ops:03d}."
        benchmarks[prefix + "python"] = steady((python_ns + ops) * 1e-9)
        benchmarks[prefix + "snekit"] = steady(snekit_ns * 1e-9)
        benchmarks[prefix + "fast_call"] = steady(native_ns * 1e-9)
        benchmarks[prefix + "compiled_i64"] = steady(snekit_ns * 2e-9)
        benchmarks[prefix + "native_call"] = steady(native_ns * 1.5e-9)
        for kind in ("frontend", "backend", "codegen", "lifecycle"):
            benchmarks[f"compile.{kind}.ops_{ops:03d}"] = steady(lifecycle_us * 1e-6)
    for case in workloads.FALLBACK_CASES:
        benchmarks[f"fallback.{case}.python"] = steady(200e-9)
        benchmarks[f"fallback.{case}.snekit"] = steady(800e-9)
    for case in workloads.GUARD_MISS_CASES:
        benchmarks[f"fallback.{case}.fast_call"] = steady(60e-9)
    return Results(
        benchmarks=benchmarks,
        metadata={"speedysnek_build": "release", "cpu_model_name": "Test CPU"},
    )


# --- Ratios ----------------------------------------------------------------


def test_ratio_direction():
    assert ratio(50, 100) == 0.5  # faster
    assert ratio(200, 100) == 2.0  # slower
    assert ratio(100, 100) == 1.0


@pytest.mark.parametrize("baseline", [0, -1.0])
def test_ratio_rejects_non_positive_baseline(baseline):
    with pytest.raises(ValueError):
        ratio(1.0, baseline)


def test_ratio_rejects_negative_value():
    with pytest.raises(ValueError):
        ratio(-1.0, 1.0)


def test_format_ratio_states_the_direction():
    assert format_ratio(0.5) == "0.50× (faster)"
    assert format_ratio(2.0) == "2.00× (slower)"
    assert format_ratio(1.0) == "1.00× (equal)"
    assert format_ratio(None) == "—"


# --- Break-even -----------------------------------------------------------


def test_break_even_calculation():
    # 1 ms of compilation, 2 µs saved per call: 500 calls.
    assert break_even_calls(1e-3, 3e-6, 1e-6) == pytest.approx(500)
    assert break_even_calls(0.0, 3e-6, 1e-6) == 0


def test_break_even_uses_consistent_units():
    in_seconds = break_even_calls(2e-3, 5e-7, 1e-7)
    in_nanoseconds = break_even_calls(2e6, 500, 100)
    assert in_seconds == pytest.approx(in_nanoseconds) == pytest.approx(5000)


@pytest.mark.parametrize(
    ("python", "jit"),
    [(1e-6, 2e-6), (1e-6, 1e-6), (0.0, 0.0)],
    ids=["slower", "equal", "zero"],
)
def test_no_break_even_when_not_faster(python, jit):
    assert break_even_calls(1e-3, python, jit) is None
    assert format_break_even(break_even_calls(1e-3, python, jit)) == NO_BREAK_EVEN


def test_no_break_even_message():
    assert NO_BREAK_EVEN == "No break-even with the current hot path."


@pytest.mark.parametrize("args", [(-1, 2, 1), (1, -2, 1), (1, 2, -1)])
def test_break_even_rejects_negative_times(args):
    with pytest.raises(ValueError):
        break_even_calls(*args)


def test_format_break_even_rounds_up():
    assert format_break_even(999.2) == "≈ 1,000 calls"
    assert format_break_even(12.0) == "≈ 12 calls"


def test_break_even_rows_and_render_when_faster():
    results = synthetic(python_ns=2000, snekit_ns=1000, lifecycle_us=500)
    rows = report.break_even_rows(results)
    assert [ops for ops, *_ in rows] == list(workloads.OP_COUNTS)
    for ops, cost, saving, calls in rows:
        assert cost == pytest.approx(500e-6)
        assert saving == pytest.approx((1000 + ops) * 1e-9)
        assert calls == pytest.approx(500e-6 / ((1000 + ops) * 1e-9))
        assert calls > 0
    text = render(results)
    assert NO_BREAK_EVEN not in text
    assert "≈ 500 calls" in text  # ops_001: 500 µs / 1.001 µs, rounded up


def test_render_reports_no_break_even_when_slower():
    results = synthetic(python_ns=200, snekit_ns=1000)
    assert all(calls is None for *_, calls in report.break_even_rows(results))
    text = render(results)
    # One table row per workload, plus the key-measurements bullet.
    assert text.count(NO_BREAK_EVEN) == len(workloads.OP_COUNTS) + 1
    assert "slower than plain Python in every measured workload" in text
    assert "≈" not in text.split("## Break-even")[1].split("##")[0]


# --- Other calculations --------------------------------------------------


def test_linear_fit_recovers_a_line():
    xs = [1, 5, 20, 50, 100, 250]
    intercept, slope = linear_fit(xs, [100 + 3 * x for x in xs])
    assert intercept == pytest.approx(100)
    assert slope == pytest.approx(3)


def test_linear_fit_rejects_degenerate_input():
    with pytest.raises(ValueError):
        linear_fit([1], [1])
    with pytest.raises(ValueError):
        linear_fit([2, 2], [1, 3])
    with pytest.raises(ValueError):
        linear_fit([1, 2], [1])


def test_breakdown_subtracts_only_nested_layers():
    results = synthetic(python_ns=200, snekit_ns=1000, native_ns=100)
    parts = report.breakdown(results, 5)
    assert set(parts) == {"decorator", "fast_call", "total"}
    assert parts["total"] == pytest.approx(1000e-9)
    assert parts["fast_call"] == pytest.approx(100e-9)
    assert parts["decorator"] == pytest.approx(900e-9)
    assert parts["decorator"] + parts["fast_call"] == pytest.approx(parts["total"])


def test_diagnostic_breakdown_is_separate():
    results = synthetic(python_ns=200, snekit_ns=1000, native_ns=100)
    parts = report.diagnostic_breakdown(results, 5)
    # CompiledI64 is 2,000 ns and I64Function.call 150 ns in the synthetic
    # data: slower than the whole @snekit call, which it is not part of.
    assert parts["total"] == pytest.approx(2000e-9)
    assert parts["native"] == pytest.approx(150e-9)
    assert parts["guards"] == pytest.approx(1850e-9)
    text = render(results)
    public = text.split("## Where a warmed")[1].split("###")[0]
    assert "CompiledI64" not in public.split("|---")[1]
    assert "### Diagnostic path (not part of warmed `@snekit`)" in text


def test_breakdown_needs_the_fast_call_layer():
    results = synthetic(python_ns=200, snekit_ns=1000)
    del results.benchmarks["runtime.ops_005.fast_call"]
    assert report.breakdown(results, 5) is None
    assert report.diagnostic_breakdown(results, 5) is not None


def test_crossover():
    # Python: 20 + 10x; JIT: 420 + 2x; they meet at x = 400 / 8 = 50.
    assert crossover((20, 10), (420, 2)) == pytest.approx(50)
    # Faster everywhere, slower everywhere, or the same slope: no crossover.
    assert crossover((20, 10), (10, 2)) is None
    assert crossover((20, 10), (420, 12)) is None
    assert crossover((20, 10), (420, 10)) is None


def test_change():
    assert change(1000, 400) == "0.40× (−60%)"
    assert change(100, 150) == "1.50× (+50%)"
    assert change(100, 100) == "1.00× (+0%)"
    with pytest.raises(ValueError):
        change(0, 1)


def test_comparison_with_a_previous_baseline():
    previous = synthetic(python_ns=200, snekit_ns=1000, lifecycle_us=500)
    current = synthetic(python_ns=200, snekit_ns=400, lifecycle_us=500)
    rows = report.comparison_rows(previous, current)
    labels = [label for label, *_ in rows]
    for ops in (1, 20, 50, 100, 250):
        assert f"warmed `@snekit`, `ops_{ops:03d}`" in labels
    assert "warmed `@snekit` fixed cost (fit)" in labels
    assert "estimated `@snekit` crossover" in labels
    assert "fallback `float_args`, `@snekit` call" in labels
    assert "break-even, `ops_250`" in labels
    row = dict((label, rest) for label, *rest in rows)["warmed `@snekit`, `ops_001`"]
    assert row == ["1.0 µs", "400.0 ns", "0.40× (−60%)"]
    text = render(current, previous=previous)
    assert "## Comparison with the previous baseline" in text
    assert "## Comparison" not in render(current)


def test_render_reports_the_crossover():
    results = synthetic(python_ns=20, snekit_ns=1000)
    # Python grows by 1 ns per operation; @snekit is flat at 1,000 ns.
    assert "Estimated `@snekit` crossover: ≈ 980 operations." in render(results)


def test_stability_warnings():
    assert stability_warnings("ok", steady(1e-6)) == []
    noisy = Stats(values=(1.0, 1.5, 0.5, 1.0), shortest_raw=0.1)
    assert any("std dev" in w for w in stability_warnings("noisy", noisy))
    outlier = Stats(values=(1.0,) * 20 + (2.5,), shortest_raw=0.1)
    assert any("maximum" in w for w in stability_warnings("outlier", outlier))
    short = Stats(values=(1e-6,) * 3, shortest_raw=1e-5)
    assert any("shortest raw" in w for w in stability_warnings("short", short))


def test_format_time_units():
    assert format_time(12.34e-9) == "12.3 ns"
    assert format_time(12.34e-6) == "12.3 µs"
    assert format_time(12.34e-3) == "12.3 ms"
    assert format_time(2.5) == "2.50 s"
    assert format_time(-391.6e-9) == "-391.6 ns"
    assert format_time(None) == "—"


# --- Rendering -----------------------------------------------------------


def test_render_contains_required_sections():
    text = render(
        synthetic(python_ns=200, snekit_ns=1000),
        commands=["python benchmarks/bench_runtime.py -o r.json"],
        interpretation="The wrapper dominates.",
        generated="2026-01-01 00:00 UTC",
    )
    for heading in (
        "## Environment",
        "### Benchmark command",
        "### pyperf metadata relevant to interpretation",
        "## Key measurements",
        "## Steady-state runtime",
        "## Where a warmed `@snekit` call spends its time",
        "## Fixed cost versus cost per operation",
        "## Fallback and wrapper overhead",
        "## Compilation and warmup",
        "## Break-even",
        "## Stability",
        "## Interpretation",
    ):
        assert heading in text
    assert "not a universal performance claim" in text
    assert "below 1.0 means the JIT path is faster" in text
    assert "python benchmarks/bench_runtime.py -o r.json" in text
    assert "The wrapper dominates." in text
    assert "Test CPU" in text
    assert "debug" not in text


def test_render_runtime_row():
    text = render(synthetic(python_ns=200, snekit_ns=1000, native_ns=100))
    row = next(line for line in text.splitlines() if line.startswith("| `ops_005`"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    # Python is 205 ns for ops_005 in the synthetic data.
    assert cells[1] == "205.0 ± 0.0"
    assert cells[2] == "1,000.0 ± 0.0"
    assert cells[3] == "100.0 ± 0.0"  # native fast call
    assert cells[4] == "2,000.0 ± 0.0"  # CompiledI64, diagnostic
    assert cells[5] == "150.0 ± 0.0"  # I64Function.call, diagnostic
    assert cells[6] == format_ratio(1000 / 205)
    assert cells[7] == format_ratio(100 / 205)


def test_render_fallback_native_miss_column():
    text = render(synthetic(python_ns=200, snekit_ns=1000))
    rows = {
        line.split("|")[1].strip(): [c.strip() for c in line.strip("|").split("|")]
        for line in text.split("## Fallback")[1].split("##")[0].splitlines()
        if line.startswith("| `")
    }
    assert rows["`float_args`"][5] == "60.0 ± 0.0"
    assert rows["`overflow`"][5] == "60.0 ± 0.0"
    assert rows["`keyword_args`"][5] == "not attempted"


def test_render_warns_about_debug_builds():
    results = synthetic(python_ns=200, snekit_ns=1000)
    results.metadata["speedysnek_build"] = "debug"
    assert "not representative" in render(results)


def test_render_tolerates_partial_results():
    results = Results(benchmarks={"runtime.ops_001.python": steady(1e-7)})
    text = render(results)
    assert "`ops_001`" in text
    assert "## Break-even" not in text


# --- pyperf integration --------------------------------------------------


def _write_suite(path, benchmarks):
    pyperf = pytest.importorskip("pyperf")
    suite = None
    for name, values in benchmarks.items():
        runs = [
            pyperf.Run(
                values,
                warmups=[(1, values[0])],
                metadata={
                    "name": name,
                    "loops": 1000,
                    "unit": "second",
                    "git_commit": commit,
                },
                collect_metadata=False,
            )
            for commit in ("abc1234", "abc1234-dirty")
        ]
        bench = pyperf.Benchmark(runs)
        if suite is None:
            suite = pyperf.BenchmarkSuite([bench])
        else:
            suite.add_benchmark(bench)
    suite.dump(str(path))


def test_load_reads_pyperf_json(tmp_path):
    path = tmp_path / "results.json"
    _write_suite(
        path,
        {
            "runtime.ops_001.python": [1e-7, 1.1e-7, 0.9e-7],
            "runtime.ops_001.snekit": [5e-7, 5e-7, 5e-7],
        },
    )
    results = report.load([path])
    python = results.get("runtime.ops_001.python")
    assert python.mean == pytest.approx(1e-7)
    assert len(python.values) == 6
    assert python.processes == 2
    assert python.shortest_raw == pytest.approx(0.9e-7 * 1000)
    assert results.shape == "2 worker processes × 3 values (+ 1 warmup)"
    assert math.isclose(results.mean("runtime.ops_001.snekit"), 5e-7)
    # Metadata that differs between runs is kept, not silently dropped.
    assert results.commits == ["abc1234", "abc1234-dirty"]
    assert "abc1234, abc1234-dirty (differs between runs)" in render(results)


def test_main_writes_markdown(tmp_path):
    path = tmp_path / "results.json"
    _write_suite(path, {"runtime.ops_001.python": [1e-7, 1e-7, 1e-7]})
    output = tmp_path / "RESULTS.md"
    notes = tmp_path / "notes.md"
    notes.write_text("Hand-written notes.\n", encoding="utf-8")
    assert (
        report.main(
            [
                str(path),
                "-o",
                str(output),
                "--command",
                "cmd",
                "--interpretation",
                str(notes),
            ]
        )
        == 0
    )
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# speedy-snek benchmark results")
    assert "Hand-written notes." in text


def test_benchmark_entry_points_start(tmp_path):
    """Smoke test: each script starts and writes pyperf JSON. No timing."""
    pytest.importorskip("pyperf")
    outputs = []
    for script, name in (
        (
            "bench_runtime.py",
            r"^(runtime\.ops_001\.(snekit|fast_call)|fallback\.float_args\.fast_call)$",
        ),
        ("bench_compilation.py", r"^compile\.lifecycle\.ops_001$"),
    ):
        output = tmp_path / f"{script}.json"
        subprocess.run(
            [
                sys.executable,
                str(BENCHMARKS / script),
                "--debug-single-value",
                "--quiet",
                "--filter",
                name,
                "-o",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(output)
    results = report.load(outputs)
    assert set(results.benchmarks) == {
        "runtime.ops_001.snekit",
        "runtime.ops_001.fast_call",
        "fallback.float_args.fast_call",
        "compile.lifecycle.ops_001",
    }
    assert results.metadata["speedysnek_build"] in {"debug", "release"}
