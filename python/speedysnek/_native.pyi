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

class CodegenError(RuntimeError):
    """Cranelift could not compile a valid program for this host."""

@final
class I64Function:
    """Natively compiled code for one program over signed 64-bit integers."""

    @property
    def arg_count(self) -> int: ...
    def call(self, args: tuple[int, ...], /) -> tuple[int, int]:
        """Run the code; return ``(status, value)`` from the native ABI."""

def compile_i64(program: tuple[tuple[int, ...], ...], arg_count: int, /) -> I64Function:
    """Compile a serialized program for exactly ``arg_count`` arguments."""

def sum_as_string(a: int, b: int) -> str:
    """Temporary smoke-test function: return ``str(a + b)``."""
