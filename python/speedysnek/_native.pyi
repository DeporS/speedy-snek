"""Type stubs for the private Rust extension module."""

from typing import final

# Wire-format opcodes for ``compile_i64`` programs (see ``src/program.rs``).
OP_LOAD_ARG: int
OP_LOAD_CONST: int
OP_ADD: int
OP_SUBTRACT: int
OP_MULTIPLY: int
OP_RETURN: int

# Status codes returned by ``I64Function.call``.
STATUS_OK: int
STATUS_OVERFLOW: int
STATUS_BAD_ARITY: int

MAX_ARGS: int

# ``True`` when the extension was built with debug assertions, as by a plain
# ``maturin develop``. Benchmarks record it; debug builds are not
# representative.
DEBUG_BUILD: bool

class CodegenError(RuntimeError):
    """Cranelift could not compile a valid program for this host."""

@final
class I64Function:
    """Natively compiled code for one program over signed 64-bit integers."""

    @property
    def arg_count(self) -> int: ...
    def call(self, args: tuple[int, ...], /) -> tuple[int, int]:
        """Run the code; return ``(status, value)`` from the native ABI.

        Raises ``TypeError`` or ``OverflowError`` for an argument that is
        not an exact ``int`` in the signed 64-bit range.
        """
    def try_call(self, args: tuple[object, ...], /) -> int | None:
        """Run the code if the call fits; the decorator's hot path.

        Returns the result, or ``None`` for a wrong argument count, an
        argument that is not an exact ``int`` in the signed 64-bit range,
        or overflow. A guard miss raises nothing and runs no Python code on
        the arguments. ``args`` must be an exact tuple (``TypeError``
        otherwise); any exception means a bug, not a guard miss.
        """

def compile_i64(program: tuple[tuple[int, ...], ...], arg_count: int, /) -> I64Function:
    """Compile a serialized program for exactly ``arg_count`` arguments."""

def sum_as_string(a: int, b: int) -> str:
    """Temporary smoke-test function: return ``str(a + b)``."""
