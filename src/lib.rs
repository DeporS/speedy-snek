use pyo3::prelude::*;

mod codegen;
mod program;

/// A Python module implemented in Rust.
///
/// Everything here is private to SpeedySnek. Programs are compiled through
/// `speedysnek._compiler`, which validates them first and wraps the result
/// in a `CompiledI64`. A compiled `I64Function` has two entry points:
///
/// - `call(args)` returns the native ABI's `(status, value)` and raises for
///   arguments that are not exact i64 ints. `CompiledI64.__call__` builds
///   its detailed `NativeGuardFailure` diagnostics on top of it.
/// - `try_call(args)` is the decorator's hot path. It validates the
///   arguments once and returns the result `int`, or `None` for any call
///   the i64 specialization cannot handle, without raising.
#[pymodule]
mod _native {
    use std::ffi::c_int;

    use pyo3::create_exception;
    use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
    use pyo3::ffi;
    use pyo3::prelude::*;
    use pyo3::types::{PyInt, PyTuple};

    use crate::codegen::{NativeCode, STATUS_BAD_ARITY, STATUS_OK, STATUS_OVERFLOW};
    use crate::program::{
        MAX_ARGS, OP_ADD, OP_LOAD_ARG, OP_LOAD_CONST, OP_MULTIPLY, OP_RETURN, OP_SUBTRACT, Op,
    };

    create_exception!(
        _native,
        CodegenError,
        PyRuntimeError,
        "Cranelift could not compile a valid program for this host."
    );

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("OP_LOAD_ARG", OP_LOAD_ARG)?;
        m.add("OP_LOAD_CONST", OP_LOAD_CONST)?;
        m.add("OP_ADD", OP_ADD)?;
        m.add("OP_SUBTRACT", OP_SUBTRACT)?;
        m.add("OP_MULTIPLY", OP_MULTIPLY)?;
        m.add("OP_RETURN", OP_RETURN)?;
        m.add("STATUS_OK", STATUS_OK)?;
        m.add("STATUS_OVERFLOW", STATUS_OVERFLOW)?;
        m.add("STATUS_BAD_ARITY", STATUS_BAD_ARITY)?;
        m.add("MAX_ARGS", MAX_ARGS)?;
        // Lets benchmarks record, and warn about, unrepresentative builds.
        m.add("DEBUG_BUILD", cfg!(debug_assertions))?;
        m.add("CodegenError", m.py().get_type::<CodegenError>())?;
        Ok(())
    }

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    /// Natively compiled code for one program over signed 64-bit integers.
    ///
    /// The object owns the executable memory; it is freed when the object
    /// is garbage collected.
    #[pyclass(frozen, module = "speedysnek._native")]
    struct I64Function {
        code: NativeCode,
    }

    #[pymethods]
    impl I64Function {
        /// The exact number of arguments the code expects.
        #[getter]
        fn arg_count(&self) -> usize {
            self.code.arg_count()
        }

        /// Run the code with a tuple of exact ints that fit in i64.
        ///
        /// Returns ``(status, value)`` straight from the native ABI. The
        /// argument count is deliberately not checked here: the generated
        /// code checks it and returns ``STATUS_BAD_ARITY``.
        fn call(&self, args: &Bound<'_, PyAny>) -> PyResult<(i32, i64)> {
            let args = args
                .cast_exact::<PyTuple>()
                .map_err(|_| PyTypeError::new_err("args must be a tuple"))?;
            let mut values = Vec::with_capacity(args.len());
            for (index, item) in args.iter().enumerate() {
                values.push(exact_i64(&item).map_err(|kind| match kind {
                    IntError::NotExactInt => {
                        PyTypeError::new_err(format!("argument {index} must be an exact int"))
                    }
                    IntError::OutOfRange => PyOverflowError::new_err(format!(
                        "argument {index} does not fit in a signed 64-bit integer"
                    )),
                })?);
            }
            Ok(self.code.call(&values))
        }

        /// Run the code if the call fits the i64 specialization.
        ///
        /// Private contract, relied on by `speedysnek._decorator`:
        ///
        /// - `args` must be an exact tuple; anything else raises
        ///   ``TypeError``. That is a caller bug, not a guard miss.
        /// - Returns the result as a Python ``int`` when the tuple has
        ///   exactly ``arg_count`` items, each an exact ``int`` in the
        ///   signed 64-bit range, and no operation overflows.
        /// - Returns ``None`` for every other call: wrong arity, ``bool``,
        ///   ``int`` subclasses, any non-``int``, values outside i64, and
        ///   overflow in any intermediate or final operation. ``None`` is
        ///   unambiguous because every compiled program returns an ``int``.
        /// - A guard miss raises nothing and runs no Python code on the
        ///   arguments: no ``__int__``, ``__index__``, ``__repr__`` or
        ///   (metaclass) attribute lookup.
        /// - Any exception it does raise is unexpected.
        ///
        /// The GIL stays held throughout.
        #[pyo3(signature = (args, /))]
        fn try_call(&self, args: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
            let args = args
                .cast_exact::<PyTuple>()
                .map_err(|_| PyTypeError::new_err("args must be a tuple"))?;
            let arg_count = self.code.arg_count();
            if args.len() != arg_count {
                return Ok(None);
            }
            // Typical functions have a few arguments; avoid a heap
            // allocation for them.
            let mut inline = [0_i64; INLINE_ARGS];
            let mut heap = Vec::new();
            let values = if arg_count <= INLINE_ARGS {
                &mut inline[..arg_count]
            } else {
                heap.resize(arg_count, 0);
                &mut heap[..]
            };
            for (slot, item) in values.iter_mut().zip(args.iter_borrowed()) {
                match exact_i64_or_none(&item)? {
                    Some(value) => *slot = value,
                    None => return Ok(None),
                }
            }
            match self.code.call(values) {
                (STATUS_OK, value) => Ok(Some(value)),
                (STATUS_OVERFLOW, _) => Ok(None),
                // The arity was checked above, so this is a backend bug.
                (status, _) => Err(PyRuntimeError::new_err(format!(
                    "unexpected native status {status}"
                ))),
            }
        }
    }

    /// Arguments that `try_call` passes without a heap allocation.
    const INLINE_ARGS: usize = 8;

    /// Compile a serialized program (see `src/program.rs`) for exactly
    /// `arg_count` arguments.
    ///
    /// Raises ``TypeError`` or ``ValueError`` for a malformed program and
    /// ``CodegenError`` if Cranelift fails.
    #[pyfunction]
    fn compile_i64(
        program: &Bound<'_, PyAny>,
        arg_count: &Bound<'_, PyAny>,
    ) -> PyResult<I64Function> {
        let out_of_range = || PyValueError::new_err(format!("arg_count must be in 0..={MAX_ARGS}"));
        let arg_count = match exact_i64(arg_count) {
            Ok(n) => usize::try_from(n)
                .ok()
                .filter(|n| *n <= MAX_ARGS)
                .ok_or_else(out_of_range)?,
            Err(IntError::OutOfRange) => return Err(out_of_range()),
            Err(IntError::NotExactInt) => {
                return Err(PyTypeError::new_err("arg_count must be an exact int"));
            }
        };
        let ops = decode(program)?;
        // `NativeCode::compile` validates again; this gives a clear error.
        crate::program::validate(&ops, arg_count).map_err(PyValueError::new_err)?;
        let code = NativeCode::compile(&ops, arg_count).map_err(CodegenError::new_err)?;
        Ok(I64Function { code })
    }

    enum IntError {
        NotExactInt,
        OutOfRange,
    }

    /// Read an exact Python `int` that fits in i64.
    ///
    /// The exact-type check comes first, so `bool`, `int` subclasses and
    /// objects with `__index__` are rejected without running any of their
    /// Python code.
    fn exact_i64(value: &Bound<'_, PyAny>) -> Result<i64, IntError> {
        if !value.is_exact_instance_of::<PyInt>() {
            return Err(IntError::NotExactInt);
        }
        value.extract::<i64>().map_err(|_| IntError::OutOfRange)
    }

    /// Like [`exact_i64`], but a rejected value is `Ok(None)` and creates
    /// no Python exception.
    ///
    /// `extract::<i64>()` reports an out-of-range `int` by raising
    /// ``OverflowError``; `PyLong_AsLongLongAndOverflow` reports it through
    /// its `overflow` flag instead. An error is returned only if CPython
    /// unexpectedly sets one.
    fn exact_i64_or_none(value: &Borrowed<'_, '_, PyAny>) -> PyResult<Option<i64>> {
        if !value.is_exact_instance_of::<PyInt>() {
            return Ok(None);
        }
        let mut overflow: c_int = 0;
        // SAFETY:
        // - `value` is a live, borrowed object and holding a `Borrowed`
        //   proves the GIL is held for the duration of the call.
        // - It is an exact `int` (checked above), so CPython converts it
        //   directly and never calls `__index__` or any other Python code.
        // - `overflow` is a valid, writable local `c_int`.
        // The result is a `c_longlong`, which Rust defines as `i64`.
        let result = unsafe { ffi::PyLong_AsLongLongAndOverflow(value.as_ptr(), &mut overflow) };
        if overflow != 0 {
            return Ok(None);
        }
        // -1 is also the error return; only then can an error be pending.
        if result == -1
            && let Some(error) = PyErr::take(value.py())
        {
            return Err(error);
        }
        Ok(Some(result))
    }

    /// Turn the Python-side tuple of tuples into decoded instructions.
    fn decode(program: &Bound<'_, PyAny>) -> PyResult<Vec<Op>> {
        let program = program
            .cast_exact::<PyTuple>()
            .map_err(|_| PyTypeError::new_err("program must be a tuple"))?;
        let mut ops = Vec::with_capacity(program.len());
        for (index, entry) in program.iter().enumerate() {
            let entry = entry.cast_exact::<PyTuple>().map_err(|_| {
                PyTypeError::new_err(format!("instruction {index} must be a tuple"))
            })?;
            let field = |position: usize| -> PyResult<i64> {
                let item = entry.get_item(position)?;
                exact_i64(&item).map_err(|kind| match kind {
                    IntError::NotExactInt => {
                        PyTypeError::new_err(format!("instruction {index} must contain exact ints"))
                    }
                    IntError::OutOfRange => PyValueError::new_err(format!(
                        "instruction {index} has a value outside the signed 64-bit range"
                    )),
                })
            };
            let (opcode, operand) = match entry.len() {
                1 => (field(0)?, None),
                2 => (field(0)?, Some(field(1)?)),
                n => {
                    return Err(PyValueError::new_err(format!(
                        "instruction {index} has {n} fields, expected 1 or 2"
                    )));
                }
            };
            ops.push(Op::decode(index, opcode, operand).map_err(PyValueError::new_err)?);
        }
        Ok(ops)
    }
}
