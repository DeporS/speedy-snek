use pyo3::prelude::*;

mod codegen;
mod program;

/// A Python module implemented in Rust.
///
/// Everything here is private to SpeedySnek. The Cranelift backend is used
/// through `speedysnek._compiler`, which validates programs and runtime
/// arguments before they reach this module.
#[pymodule]
mod _native {
    use pyo3::create_exception;
    use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
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
    }

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
