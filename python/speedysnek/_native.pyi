"""Type stubs for the private Rust extension module."""

from collections.abc import Callable
from typing import Any, final

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
    @property
    def try_calls(self) -> int:
        """How many fast calls this function has handled, guard misses
        included: through ``try_call`` or directly from a ``NativeWrapper``.
        A diagnostic for tests and benchmarks."""
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

@final
class NativeWrapper:
    """The callable ``@snekit`` returns; its per-call logic is in Rust.

    Metadata copied by ``functools.update_wrapper`` lives in the instance
    ``__dict__``; ``__annotate__`` and ``__annotations__`` behave like a
    function's. Binds as a method like a function (``__get__``).
    """

    call_count: int
    __annotate__: Callable[[int], dict[str, Any]] | None
    __annotations__: dict[str, Any]
    def __init__(
        self, func: Callable[..., Any], state: object, cold: object, /
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __get__(self, instance: object, owner: type | None = None, /) -> Any: ...

def compile_i64(program: tuple[tuple[int, ...], ...], arg_count: int, /) -> I64Function:
    """Compile a serialized program for exactly ``arg_count`` arguments."""

def sum_as_string(a: int, b: int) -> str:
    """Temporary smoke-test function: return ``str(a + b)``."""
