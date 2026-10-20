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
OP_LOAD_LOCAL: int
OP_STORE_LOCAL: int
OP_JUMP: int
OP_JUMP_IF_FALSE: int
OP_LESS_THAN: int
OP_LESS_EQUAL: int
OP_GREATER_THAN: int
OP_GREATER_EQUAL: int
OP_EQUAL: int
OP_NOT_EQUAL: int

# Status codes returned by ``I64Function.call``.
STATUS_OK: int
STATUS_OVERFLOW: int
STATUS_BAD_ARITY: int
STATUS_BUDGET_EXHAUSTED: int

# Backward jumps one call of a compiled loop may take. The maximum equals
# the default: ``compile_i64``'s budget can only lower it.
DEFAULT_BACKEDGE_BUDGET: int
MAX_BACKEDGE_BUDGET: int

MAX_ARGS: int
MAX_LOCALS: int
MAX_JUMPS: int

# ``True`` when the extension was built with debug assertions, as by a plain
# ``maturin develop``. Benchmarks record it; debug builds are not
# representative.
DEBUG_BUILD: bool

class CodegenError(RuntimeError):
    """Cranelift could not compile a valid program for this host."""

class UnguardableBinding(Exception):
    """A function's ``range`` is not the builtin ``range``, or cannot be
    proven to be without running Python code."""

@final
class I64Function:
    """Natively compiled code for one program over signed 64-bit integers."""

    @property
    def arg_count(self) -> int: ...
    @property
    def backedge_budget(self) -> int | None:
        """Backward jumps one call may take; ``None`` without a loop."""
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
        overflow, or a loop that used up its backedge budget. A guard miss
        raises nothing and runs no Python code on the arguments. ``args``
        must be an exact tuple (``TypeError`` otherwise); any exception
        means a bug, not a guard miss.
        """

@final
class RangeGuard:
    """Proof that ``range``, looked up in ``globals`` and then ``builtins``
    as a function's ``LOAD_GLOBAL`` does, is the builtin ``range`` type.

    Both must be exact dicts whose keys are all exact ``str``; otherwise,
    or if ``range`` resolves to anything else, the constructor raises
    ``UnguardableBinding``. A dict watcher clears ``valid`` before any
    change to either dict that could affect the lookup. It runs no Python
    code.
    """

    def __init__(
        self, globals: dict[str, Any], builtins: dict[str, Any], /
    ) -> None: ...
    @property
    def valid(self) -> bool:
        """Whether the binding still holds. Once false, it stays false."""

@final
class GuardedI64Function:
    """An ``I64Function`` that may only run while ``guard`` holds.

    ``NativeWrapper`` checks the guard before each native call;
    ``try_call`` returns ``None`` once it fails.
    """

    def __init__(self, function: I64Function, guard: RangeGuard, /) -> None: ...
    @property
    def function(self) -> I64Function: ...
    @property
    def guard(self) -> RangeGuard: ...
    @property
    def arg_count(self) -> int: ...
    def try_call(self, args: tuple[object, ...], /) -> int | None: ...

@final
class NativeWrapper:
    """The callable ``@snekit`` returns; its per-call logic is in Rust.

    Metadata copied by ``functools.update_wrapper`` lives in the instance
    ``__dict__``; ``__annotate__`` and ``__annotations__`` behave like a
    function's. Binds as a method like a function (``__get__``).
    """

    call_count: int
    """Calls so far, raising ones included: exact, with no upper bound.
    Setting it accepts anything ``operator.index`` accepts."""
    __annotate__: Callable[[int], dict[str, Any]] | None
    __annotations__: dict[str, Any]
    def __init__(
        self, func: Callable[..., Any], state: object, cold: object, /
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __get__(self, instance: object, owner: type | None = None, /) -> Any: ...

def compile_i64(
    program: tuple[tuple[int, ...], ...],
    arg_count: int,
    /,
    backedge_budget: int | None = None,
) -> I64Function:
    """Compile a serialized program for exactly ``arg_count`` arguments.

    A loop may take at most ``backedge_budget`` backward jumps per call
    (default ``DEFAULT_BACKEDGE_BUDGET``, at most ``MAX_BACKEDGE_BUDGET``).
    """

def sum_as_string(a: int, b: int) -> str:
    """Temporary smoke-test function: return ``str(a + b)``."""
