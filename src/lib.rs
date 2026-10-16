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
/// - `try_call(args)` is the fast path. It validates the arguments once
///   and returns the result `int`, or `None` for any call the i64
///   specialization cannot handle, without raising.
///
/// `NativeWrapper` is the object `@snekit` returns. It runs the fast path's
/// Rust code directly, without calling `try_call` through Python.
#[pymodule]
mod _native {
    use std::ffi::c_int;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

    use pyo3::create_exception;
    use pyo3::exceptions::{
        PyException, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError,
    };
    use pyo3::ffi;
    use pyo3::intern;
    use pyo3::prelude::*;
    use pyo3::sync::PyOnceLock;
    use pyo3::types::{PyDict, PyInt, PyTuple, PyType};
    use pyo3::{PyTraverseError, PyVisit};

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
        /// Diagnostics: how many times `try_call_tuple` ran.
        try_calls: AtomicU64,
    }

    #[pymethods]
    impl I64Function {
        /// The exact number of arguments the code expects.
        #[getter]
        fn arg_count(&self) -> usize {
            self.code.arg_count()
        }

        /// How many fast calls this function has handled, through
        /// `try_call` or directly from a `NativeWrapper`: guard misses
        /// included. Tests and benchmarks use it to prove which path a call
        /// took, because a call from Rust is invisible to `sys.setprofile`.
        #[getter]
        fn try_calls(&self) -> u64 {
            self.try_calls.load(Ordering::Relaxed)
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
            self.try_call_tuple(args)
        }
    }

    impl I64Function {
        /// The body of `try_call`, shared with `NativeWrapper`, which calls
        /// it directly instead of through the Python method.
        fn try_call_tuple(&self, args: &Bound<'_, PyTuple>) -> PyResult<Option<i64>> {
            // Not a locked read-modify-write: callers hold the GIL, which
            // serializes them, and this counter is only for diagnostics.
            let try_calls = self.try_calls.load(Ordering::Relaxed);
            self.try_calls
                .store(try_calls.wrapping_add(1), Ordering::Relaxed);
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

    /// `types.MethodType`, used to bind a `NativeWrapper` like a function.
    static METHOD_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

    /// `operator.index`, used to accept a new `call_count`.
    static OPERATOR_INDEX: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

    /// The callable that `@snekit` returns (see `speedysnek._decorator`).
    ///
    /// It runs the decorator's per-call logic without a Python frame.
    /// Compilation and every state transition stay in the Python
    /// `_JitState`; this object only reads `state.specialization` and
    /// `state.status` and calls the state's `cold_call`, `invalidate` and
    /// `native_failed` methods. It binds as a method like a function.
    ///
    /// Metadata copied by `functools.update_wrapper` lives in the instance
    /// `__dict__`, except `__annotate__` and `__annotations__`, which behave
    /// like a function's: computed lazily, cached, and kept consistent.
    ///
    /// `call_count` is exact and unbounded: it is `count_base + calls`, where
    /// `calls` is the atomic counter every call increments and `count_base`
    /// an optional Python `int` (`None` means 0). When an increment wraps
    /// `calls` from `i64::MAX` to `i64::MIN`, it lost exactly 2**64, which
    /// [`NativeWrapper::carry`] adds to `count_base`. `calls` is never reset,
    /// so increments made meanwhile are never lost. The base is only changed
    /// with the GIL held and by exact-`int` arithmetic, which runs no Python
    /// code and lets no other thread run, so no thread can observe a
    /// wrapped `calls` without its carry.
    #[pyclass(frozen, dict, weakref, module = "speedysnek._native")]
    struct NativeWrapper {
        func: Py<PyAny>,
        state: Py<PyAny>,
        /// `_Status.COLD`.
        cold: Py<PyAny>,
        calls: AtomicI64,
        count_base: Mutex<Option<Py<PyAny>>>,
        annotations: Mutex<Annotations>,
    }

    /// A function's `__annotate__` and its cached `__annotations__`.
    #[derive(Default)]
    struct Annotations {
        annotate: Option<Py<PyAny>>,
        cached: Option<Py<PyAny>>,
    }

    #[pymethods]
    impl NativeWrapper {
        #[new]
        fn new(func: Py<PyAny>, state: Py<PyAny>, cold: Py<PyAny>) -> Self {
            NativeWrapper {
                func,
                state,
                cold,
                calls: AtomicI64::new(0),
                count_base: Mutex::new(None),
                annotations: Mutex::default(),
            }
        }

        /// The number of calls, including calls that raised: an exact
        /// `int`, with no upper bound.
        #[getter]
        fn get_call_count<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let base = self
                .count_base
                .lock()
                .unwrap()
                .as_ref()
                .map(|b| b.clone_ref(py));
            let calls = self.calls.load(Ordering::Relaxed).into_pyobject(py)?;
            match base {
                None => Ok(calls.into_any()),
                Some(base) => base.bind(py).add(calls),
            }
        }

        /// Accepts what `operator.index` accepts (an `int`, or an object
        /// with `__index__`), of any size.
        #[setter]
        fn set_call_count(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
            let py = value.py();
            // Runs `__index__` once, before any state changes; the result is
            // an exact `int`, so nothing below runs Python code.
            let value = OPERATOR_INDEX
                .import(py, "operator", "index")?
                .call1((value,))?;
            let (calls, base) = match value.extract::<i64>() {
                Ok(calls) => (calls, None),
                // Outside i64: keep the exact value as the base.
                Err(_) => (0, Some(value.unbind())),
            };
            let old = {
                let mut count_base = self.count_base.lock().unwrap();
                self.calls.store(calls, Ordering::Relaxed);
                std::mem::replace(&mut *count_base, base)
            };
            // Dropped without the lock held.
            drop(old);
            Ok(())
        }

        /// The same logic, line for line, as the Python closure `@snekit`
        /// returned before (kept in `benchmarks/experiments/native_wrapper.py`).
        #[pyo3(signature = (*args, **kwargs))]
        fn __call__<'py>(
            &self,
            py: Python<'py>,
            args: &Bound<'py, PyTuple>,
            kwargs: Option<&Bound<'py, PyDict>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            // Counted first, so calls that raise are counted too.
            if self.calls.fetch_add(1, Ordering::Relaxed) == i64::MAX {
                self.carry(py)?;
            }
            let func = self.func.bind(py);
            let state = self.state.bind(py);
            let mut spec = state.getattr(intern!(py, "specialization"))?;
            if spec.is_none() {
                if !state.getattr(intern!(py, "status"))?.is(&self.cold) {
                    return func.call(args, kwargs);
                }
                spec = state.call_method0(intern!(py, "cold_call"))?;
                if spec.is_none() {
                    return func.call(args, kwargs);
                }
            }
            let code = spec.getattr(intern!(py, "code"))?;
            if !func.getattr(intern!(py, "__code__"))?.is(&code) {
                // Never run machine code compiled from a stale code object.
                state.call_method0(intern!(py, "invalidate"))?;
            } else if kwargs.is_none_or(|kwargs| kwargs.is_empty()) {
                match native_call(&spec, args) {
                    Ok(Some(result)) => return Ok(result),
                    // A guard miss.
                    Ok(None) => {}
                    // Unexpected: disable, then fall back.
                    Err(error) if error.is_instance_of::<PyException>(py) => {
                        state
                            .call_method1(intern!(py, "native_failed"), (error.into_value(py),))?;
                    }
                    // A BaseException such as KeyboardInterrupt propagates.
                    Err(error) => return Err(error),
                }
            }
            // No exception is being handled here, so an exception raised by
            // `func` gets no `__context__` from the native attempt.
            func.call(args, kwargs)
        }

        /// Bind like a function: `MethodType(self, instance)`, or `self`
        /// when accessed through the class.
        fn __get__<'py>(
            slf: Bound<'py, Self>,
            instance: Option<Bound<'py, PyAny>>,
            _owner: Option<Bound<'py, PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            match instance {
                Some(instance) if !instance.is_none() => {
                    let py = slf.py();
                    METHOD_TYPE
                        .import(py, "types", "MethodType")?
                        .call1((slf, instance))
                }
                _ => Ok(slf.into_any()),
            }
        }

        /// Like a function's `__annotate__`: a callable or `None`.
        #[getter(__annotate__)]
        fn get_annotate(&self, py: Python<'_>) -> Py<PyAny> {
            let annotations = self.annotations.lock().unwrap();
            annotations
                .annotate
                .as_ref()
                .map_or_else(|| py.None(), |annotate| annotate.clone_ref(py))
        }

        /// Setting a callable discards the cached `__annotations__`.
        #[setter(__annotate__)]
        fn set_annotate(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
            let old = if value.is_none() {
                (self.annotations.lock().unwrap().annotate.take(), None)
            } else if value.is_callable() {
                let mut annotations = self.annotations.lock().unwrap();
                let cached = annotations.cached.take();
                (annotations.annotate.replace(value.unbind()), cached)
            } else {
                return Err(PyTypeError::new_err(
                    "__annotate__ must be callable or None",
                ));
            };
            // Dropped without the lock held: a decref can run Python code.
            drop(old);
            Ok(())
        }

        #[deleter(__annotate__)]
        fn delete_annotate(&self) -> PyResult<()> {
            Err(PyTypeError::new_err("__annotate__ cannot be deleted"))
        }

        /// Like a function's: computed from `__annotate__` on first use,
        /// then cached.
        #[getter(__annotations__)]
        fn get_annotations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let annotate = {
                let annotations = self.annotations.lock().unwrap();
                if let Some(cached) = &annotations.cached {
                    return Ok(cached.bind(py).clone());
                }
                annotations.annotate.as_ref().map(|a| a.clone_ref(py))
            };
            let value = match annotate {
                // `annotationlib.Format.VALUE`; run without the lock held.
                Some(annotate) => {
                    let value = annotate.bind(py).call1((1,))?;
                    if !value.is_instance_of::<PyDict>() {
                        return Err(PyTypeError::new_err(format!(
                            "__annotate__ returned non-dict of type '{}'",
                            value.get_type().name()?
                        )));
                    }
                    value
                }
                None => PyDict::new(py).into_any(),
            };
            let mut annotations = self.annotations.lock().unwrap();
            // Another thread may have computed them while `__annotate__` ran.
            Ok(annotations
                .cached
                .get_or_insert_with(|| value.unbind())
                .bind(py)
                .clone())
        }

        /// A dict or `None`; either clears `__annotate__`.
        #[setter(__annotations__)]
        fn set_annotations(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
            if value.is_none() {
                return self.replace_annotations(None);
            }
            if !value.is_instance_of::<PyDict>() {
                return Err(PyTypeError::new_err(
                    "__annotations__ must be set to a dict object",
                ));
            }
            self.replace_annotations(Some(value.unbind()))
        }

        #[deleter(__annotations__)]
        fn delete_annotations(&self) -> PyResult<()> {
            self.replace_annotations(None)
        }

        fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
            Ok(format!(
                "<snekit wrapper of {}>",
                self.func.bind(py).repr()?
            ))
        }

        /// Pickle and copy by reference, like a function.
        fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
            slf.getattr(intern!(slf.py(), "__qualname__"))
        }

        fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
            visit.call(&self.func)?;
            visit.call(&self.state)?;
            visit.call(&self.cold)?;
            // Never contended: these locks are only taken with the GIL held,
            // and never across a call into Python.
            if let Ok(count_base) = self.count_base.try_lock() {
                visit.call(&*count_base)?;
            }
            if let Ok(annotations) = self.annotations.try_lock() {
                visit.call(&annotations.annotate)?;
                visit.call(&annotations.cached)?;
            }
            Ok(())
        }
    }

    impl NativeWrapper {
        /// Record the 2**64 that an increment from `i64::MAX` just lost.
        ///
        /// Only exact-`int` arithmetic happens between the wrap and the new
        /// base being stored, so with the GIL held no other thread can run
        /// in between (see the struct documentation). The only possible
        /// error is a `MemoryError`; the increment is then undone and the
        /// error propagates before the call runs, exactly as when the old
        /// Python closure's `call_count += 1` failed.
        #[cold]
        fn carry(&self, py: Python<'_>) -> PyResult<()> {
            let new_base = (|| {
                let base = self
                    .count_base
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|b| b.clone_ref(py));
                let carry = (1_u128 << 64).into_pyobject(py)?.into_any();
                match base {
                    None => Ok(carry),
                    Some(base) => base.bind(py).add(carry),
                }
            })();
            let new_base = match new_base {
                Ok(new_base) => new_base,
                Err(error) => {
                    self.calls.fetch_sub(1, Ordering::Relaxed);
                    return Err(error);
                }
            };
            let old = self.count_base.lock().unwrap().replace(new_base.unbind());
            // Dropped without the lock held.
            drop(old);
            Ok(())
        }

        fn replace_annotations(&self, value: Option<Py<PyAny>>) -> PyResult<()> {
            let old = {
                let mut annotations = self.annotations.lock().unwrap();
                let cached = std::mem::replace(&mut annotations.cached, value);
                (cached, annotations.annotate.take())
            };
            // Dropped without the lock held: a decref can run Python code.
            drop(old);
            Ok(())
        }
    }

    /// The native attempt: `Some(result)`, or `None` for a guard miss or a
    /// call with another arity.
    ///
    /// A real specialization's `function` is an `I64Function`, run through
    /// the same helper as `try_call`. `function` is a strong reference, so
    /// the machine code stays alive for the whole call even if the
    /// specialization is replaced meanwhile. Any other `function` (only test
    /// doubles use one) means: call `spec.fast_call`, which has `try_call`'s
    /// contract.
    fn native_call<'py>(
        spec: &Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = spec.py();
        let function = spec.getattr(intern!(py, "function"))?;
        if let Ok(function) = function.cast::<I64Function>() {
            let function = function.get();
            // As in the Python wrapper, a call with another arity (an
            // omitted default) makes no native call at all.
            if args.len() != function.code.arg_count() {
                return Ok(None);
            }
            return match function.try_call_tuple(args)? {
                Some(value) => Ok(Some(value.into_pyobject(py)?.into_any())),
                None => Ok(None),
            };
        }
        if args.len() != spec.getattr(intern!(py, "arg_count"))?.extract::<usize>()? {
            return Ok(None);
        }
        let result = spec.getattr(intern!(py, "fast_call"))?.call1((args,))?;
        Ok((!result.is_none()).then_some(result))
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
        Ok(I64Function {
            code,
            try_calls: AtomicU64::new(0),
        })
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
