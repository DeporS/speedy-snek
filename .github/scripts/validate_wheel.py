"""Install one built wheel into a fresh virtual environment and check it.

Used by ``.github/workflows/build-wheels.yml``. Run it with the CPython
3.14 interpreter the wheel was built for::

    python -I .github/scripts/validate_wheel.py --wheel-dir dist \\
        --expect-tag cp314-cp314-win_amd64

It has two stages, both in this one file:

1. **Driver** (the default). Finds the single ``.whl`` in ``--wheel-dir``,
   checks its tags (and, for Linux, with ``auditwheel`` that the binary
   really meets its manylinux tag), creates a virtual environment in a new temporary
   directory, installs exactly that file with ``pip --no-index --no-deps``
   (so nothing can be built or downloaded), copies this script next to the
   environment and runs stage 2 there.
2. **Checks** (``--installed``). Runs with the environment's interpreter in
   isolated mode (``-I``), from the temporary directory, so neither the
   checkout nor the current directory is on ``sys.path``: a packaging
   error cannot be masked by the source tree. It checks where the package
   was imported from, and that ``@snekit`` compiles and runs native code.

Execution paths are proven with the same private diagnostics as
``python/tests/test_jit_compile.py``: the compiled function's
``I64Function.try_calls`` counter for native fast calls, which
``sys.setprofile`` cannot see, and ``sys.setprofile`` for frames of the
original Python function. Nothing here runs Cargo or Maturin, imports the
test suite, or times anything.

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname


def fail(message: str) -> None:
    raise SystemExit(f"wheel validation failed: {message}")


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


# --- Stage 1: the driver ---------------------------------------------------


def wheel_tags(wheel: Path) -> set[str]:
    """The ``python-abi-platform`` tags in the filename, expanded."""
    parts = wheel.name.removesuffix(".whl").split("-")
    check(len(parts) == 5, f"unexpected wheel filename {wheel.name!r}")
    _, _, pythons, abis, platforms = parts
    return {
        f"{python}-{abi}-{platform}"
        for python in pythons.split(".")
        for abi in abis.split(".")
        for platform in platforms.split(".")
    }


def check_wheel_file(wheel: Path, expected_tag: str) -> None:
    tags = wheel_tags(wheel)
    print(f"wheel: {wheel.name}")
    print(f"tags:  {', '.join(sorted(tags))}")
    check(expected_tag in tags, f"{expected_tag!r} is not among the wheel's tags")
    # Standard GIL-enabled CPython 3.14 only: no abi3, no free-threaded build.
    check(
        all(tag.startswith("cp314-cp314-") for tag in tags),
        "every tag must be for cp314-cp314",
    )

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        info = next(n for n in names if n.endswith(".dist-info/WHEEL"))
        metadata = archive.read(info).decode()
    declared = {
        line.split(":", 1)[1].strip()
        for line in metadata.splitlines()
        if line.startswith("Tag:")
    }
    check(declared == tags, f"WHEEL declares {declared}, filename says {tags}")
    check("Root-Is-Purelib: false" in metadata, "wheel must be platlib")
    extensions = [
        n
        for n in names
        if n.startswith("speedysnek/_native.") and n.endswith((".so", ".pyd"))
    ]
    check(len(extensions) == 1, f"expected one native extension, found {extensions}")
    check("speedysnek/__init__.py" in names, "speedysnek/__init__.py is missing")
    check(
        not any(n.startswith(("src/", "python/")) or n.endswith(".rs") for n in names),
        "the wheel contains source-tree files",
    )


def check_manylinux(wheel: Path, expected_tag: str, auditwheel: str) -> None:
    """Check with ``auditwheel`` that the binary really meets the tag.

    A ``linux_*`` wheel renamed to ``manylinux_*`` would fail: auditwheel
    inspects the extension's external libraries and versioned symbols.
    """
    platform = expected_tag.split("-")[2]
    claimed = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", platform)
    check(claimed is not None, f"{platform!r} is not a manylinux tag")
    output = subprocess.run(
        [auditwheel, "show", str(wheel)], capture_output=True, text=True, check=True
    ).stdout
    print(output)
    # auditwheel wraps long lines; normalize whitespace before matching.
    found = re.search(
        r'consistent with the following platform tag: "([^"]+)"',
        " ".join(output.split()),
    )
    check(found is not None, "could not read auditwheel's platform tag")
    actual = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", found[1])
    check(actual is not None, f"auditwheel says the wheel is only {found[1]!r}")
    check(actual[3] == claimed[3], f"architecture {actual[3]!r} != {claimed[3]!r}")
    check(
        (int(actual[1]), int(actual[2])) <= (int(claimed[1]), int(claimed[2])),
        f"the binary needs {found[1]}, newer than the {platform} tag",
    )


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def run(*command: str | Path, cwd: Path) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def drive(wheel_dir: Path, expected_tag: str, auditwheel: str | None) -> None:
    wheels = sorted(wheel_dir.glob("*.whl"))
    check(
        len(wheels) == 1, f"expected exactly one wheel in {wheel_dir}, found {wheels}"
    )
    wheel = wheels[0].resolve()
    check_wheel_file(wheel, expected_tag)
    if "manylinux" in expected_tag:
        check(auditwheel is not None, "--auditwheel is required for manylinux")
        check_manylinux(wheel, expected_tag, auditwheel)

    workdir = Path(tempfile.mkdtemp(prefix="speedysnek-wheel-check-"))
    venv = workdir / "venv"
    run(sys.executable, "-m", "venv", venv, cwd=workdir)
    python = venv_python(venv)
    # A wheel file is installed as is. --no-index and --no-deps make sure
    # pip neither downloads nor builds anything.
    run(
        python,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--only-binary=:all:",
        "--no-cache-dir",
        "--disable-pip-version-check",
        wheel,
        cwd=workdir,
    )
    script = workdir / Path(__file__).name
    shutil.copyfile(__file__, script)
    run(python, "-I", script, "--installed", wheel, cwd=workdir)


# --- Stage 2: checks inside the fresh environment ---------------------------


def check_environment(wheel: Path) -> None:
    import importlib.metadata

    check(sys.implementation.name == "cpython", "not CPython")
    check(sys.version_info[:2] == (3, 14), f"not CPython 3.14: {sys.version}")
    check(
        not sysconfig.get_config_var("Py_GIL_DISABLED"),
        "free-threaded CPython is not supported",
    )
    check(sys.prefix != sys.base_prefix, "not running in a virtual environment")
    cwd = Path.cwd().resolve()
    check(
        not any(Path(entry or ".").resolve() == cwd for entry in sys.path),
        "the current directory is on sys.path",
    )

    # pip records where the distribution came from (PEP 610): it must be
    # this wheel file, not a directory or an editable checkout.
    dist = importlib.metadata.distribution("speedysnek")
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    print(f"installed from: {direct_url.get('url')}")
    check("archive_info" in direct_url, f"not installed from an archive: {direct_url}")
    source = Path(url2pathname(urlparse(direct_url["url"]).path))
    check(os.path.samefile(source, wheel), "installed from another file")


def check_import() -> None:
    import speedysnek
    from speedysnek import _native

    package = Path(speedysnek.__file__).resolve().parent
    site_packages = Path(sysconfig.get_path("platlib")).resolve()
    print(f"speedysnek: {package}")
    print(f"_native:    {_native.__file__}")
    check(package.parent == site_packages, "speedysnek was not imported from the venv")
    native = Path(_native.__file__).resolve()
    check(native.parent == package, "_native was not loaded from the package")
    check(native.suffix in {".so", ".pyd"}, "_native is not an extension module")
    check(_native.DEBUG_BUILD is False, "the extension is a debug build")
    check(speedysnek.jit is speedysnek.snekit, "speedysnek.jit is not snekit")


def check_snekit() -> None:
    from speedysnek import snekit
    from speedysnek._decorator import _WARMUP_CALLS, _Status

    @snekit
    def expression(a, b=3):
        return (a + b) * 2

    state = expression._jit_state
    original = expression.__wrapped__.__code__

    def native_calls() -> int:
        compiled = state.compiled
        return 0 if compiled is None else compiled.function.try_calls

    @contextlib.contextmanager
    def recording():
        calls = collections.Counter()
        before = native_calls()

        def profiler(frame, event, arg):
            if event == "call" and frame.f_code is original:
                calls["python"] += 1

        sys.setprofile(profiler)
        try:
            yield calls
        finally:
            sys.setprofile(None)
            calls["native"] = native_calls() - before

    def call(*args, **kwargs):
        """Call ``expression``; return ``(result, python_calls, native_calls)``."""
        with recording() as calls:
            result = expression(*args, **kwargs)
        return result, calls["python"], calls["native"]

    # Warmup: plain Python, nothing compiled yet.
    for i in range(_WARMUP_CALLS):
        check(call(i, 1) == ((i + 1) * 2, 1, 0), f"warmup call {i}")
    check(state.status is _Status.COLD, f"still warming up: {state.status}")

    # The next call compiles, then runs native code.
    check(call(10, 20) == (60, 0, 1), "the compiling call did not run natively")
    check(state.status is _Status.COMPILED, f"not compiled: {state.status}")
    check(state.compile_attempts == 1, "compiled more than once")

    # A later valid call runs native code and no Python frame.
    check(call(-7, 4) == (-6, 0, 1), "a warmed call did not run natively")

    # Outside i64: a guard miss, then Python's exact result.
    big = 2**63
    check(call(big, 1) == ((big + 1) * 2, 1, 1), "the i64 fallback is wrong")

    # Keyword and default arguments skip native code and run Python.
    check(call(1, b=2) == (6, 1, 0), "the keyword call is wrong")
    check(call(5) == (16, 1, 0), "the default-argument call is wrong")

    # A Python exception propagates unchanged.
    try:
        expression.__wrapped__(5, "x")
    except TypeError as error:
        expected = str(error)
    with recording() as calls:
        try:
            expression(5, "x")
        except TypeError as error:
            check(str(error) == expected, f"wrong TypeError: {error}")
        else:
            fail("expression(5, 'x') did not raise TypeError")
    check(calls["python"] == 1, "the exception did not come from the original")

    # Fallbacks keep the native code; it still runs.
    check(state.status is _Status.COMPILED, f"specialization lost: {state.status}")
    check(call(2, 3) == (10, 0, 1), "native code no longer runs")
    check(expression.call_count == _WARMUP_CALLS + 7, "wrong call_count")


def check_installed(wheel: Path) -> None:
    check_environment(wheel)
    check_import()
    check_snekit()
    print("ok: the installed wheel imports, compiles and runs @snekit natively")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel-dir", type=Path, default=Path("dist"))
    parser.add_argument("--expect-tag", help="a tag the wheel must have")
    parser.add_argument("--auditwheel", help="auditwheel executable (manylinux only)")
    parser.add_argument("--installed", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.installed is not None:
        check_installed(args.installed)
    else:
        check(args.expect_tag is not None, "--expect-tag is required")
        drive(args.wheel_dir, args.expect_tag, args.auditwheel)


if __name__ == "__main__":
    main()
