use pyo3::prelude::*;

mod codegen;
mod guard;
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
/// Rust code directly, without calling `try_call` through Python. Once its
/// function is compiled, a `HotSlot` holds the specialization in Rust, so
/// a warmed call reads no Python attribute at all.
///
/// The slot also carries the wrapper's miss policy: a compiled function
/// whose calls keep missing the i64 specialization is *parked*, so that
/// later calls run the original function without a native attempt, and
/// it re-probes the specialization now and then (see `HotSlot`).
///
/// A function with a `for` loop over `range` is compiled to a program that
/// no longer mentions `range`. `RangeGuard` proves that the function's
/// `range` is the builtin and watches its globals and builtins for
/// changes; a `GuardedI64Function` pairs it with the machine code, and the
/// wrapper checks it before every native call (see `guard.rs`).
#[pymodule]
mod _native {
    use std::ffi::c_int;
    use std::sync::atomic::{AtomicI64, AtomicU8, AtomicU32, AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

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

    use crate::guard::{Guard, Unguardable};

    use crate::codegen::{
        DEFAULT_BACKEDGE_BUDGET, MAX_BACKEDGE_BUDGET, NativeCode, STATUS_BAD_ARITY,
        STATUS_BUDGET_EXHAUSTED, STATUS_OK, STATUS_OVERFLOW,
    };
    use crate::program::{
        MAX_ARGS, MAX_JUMPS, MAX_LOCALS, OP_ADD, OP_EQUAL, OP_GREATER_EQUAL, OP_GREATER_THAN,
        OP_JUMP, OP_JUMP_IF_FALSE, OP_LESS_EQUAL, OP_LESS_THAN, OP_LOAD_ARG, OP_LOAD_CONST,
        OP_LOAD_LOCAL, OP_MULTIPLY, OP_NOT_EQUAL, OP_RETURN, OP_STORE_LOCAL, OP_SUBTRACT, Op,
    };

    create_exception!(
        _native,
        CodegenError,
        PyRuntimeError,
        "Cranelift could not compile a valid program for this host."
    );

    create_exception!(
        _native,
        UnguardableBinding,
        PyException,
        "A function's range cannot be proven to be the builtin range without \
         running Python code, or is not the builtin range."
    );

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("OP_LOAD_ARG", OP_LOAD_ARG)?;
        m.add("OP_LOAD_CONST", OP_LOAD_CONST)?;
        m.add("OP_ADD", OP_ADD)?;
        m.add("OP_SUBTRACT", OP_SUBTRACT)?;
        m.add("OP_MULTIPLY", OP_MULTIPLY)?;
        m.add("OP_RETURN", OP_RETURN)?;
        m.add("OP_LOAD_LOCAL", OP_LOAD_LOCAL)?;
        m.add("OP_STORE_LOCAL", OP_STORE_LOCAL)?;
        m.add("OP_JUMP", OP_JUMP)?;
        m.add("OP_JUMP_IF_FALSE", OP_JUMP_IF_FALSE)?;
        m.add("OP_LESS_THAN", OP_LESS_THAN)?;
        m.add("OP_LESS_EQUAL", OP_LESS_EQUAL)?;
        m.add("OP_GREATER_THAN", OP_GREATER_THAN)?;
        m.add("OP_GREATER_EQUAL", OP_GREATER_EQUAL)?;
        m.add("OP_EQUAL", OP_EQUAL)?;
        m.add("OP_NOT_EQUAL", OP_NOT_EQUAL)?;
        m.add("STATUS_OK", STATUS_OK)?;
        m.add("STATUS_OVERFLOW", STATUS_OVERFLOW)?;
        m.add("STATUS_BAD_ARITY", STATUS_BAD_ARITY)?;
        m.add("STATUS_BUDGET_EXHAUSTED", STATUS_BUDGET_EXHAUSTED)?;
        m.add("DEFAULT_BACKEDGE_BUDGET", DEFAULT_BACKEDGE_BUDGET)?;
        m.add("MAX_BACKEDGE_BUDGET", MAX_BACKEDGE_BUDGET)?;
        m.add("MAX_ARGS", MAX_ARGS)?;
        m.add("MAX_LOCALS", MAX_LOCALS)?;
        m.add("MAX_JUMPS", MAX_JUMPS)?;
        m.add("MISS_LIMIT", MISS_LIMIT)?;
        m.add("REPROBE_INTERVAL", REPROBE_INTERVAL)?;
        // Lets benchmarks record, and warn about, unrepresentative builds.
        m.add("DEBUG_BUILD", cfg!(debug_assertions))?;
        m.add("CodegenError", m.py().get_type::<CodegenError>())?;
        m.add(
            "UnguardableBinding",
            m.py().get_type::<UnguardableBinding>(),
        )?;
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
        /// Diagnostics: how many of those returned no native result (a
        /// guard miss or an error). Counted on the miss path only, so a
        /// native result costs nothing extra.
        try_misses: AtomicU64,
    }

    #[pymethods]
    impl I64Function {
        /// The exact number of arguments the code expects.
        #[getter]
        fn arg_count(&self) -> usize {
            self.code.arg_count()
        }

        /// The backward jumps one call may take, or `None` for a program
        /// without a loop.
        #[getter]
        fn backedge_budget(&self) -> Option<u64> {
            self.code.backedge_budget()
        }

        /// How many fast calls this function has handled, through
        /// `try_call` or directly from a `NativeWrapper`: guard misses
        /// included. Tests and benchmarks use it to prove which path a call
        /// took, because a call from Rust is invisible to `sys.setprofile`.
        #[getter]
        fn try_calls(&self) -> u64 {
            self.try_calls.load(Ordering::Relaxed)
        }

        /// How many of the `try_calls` returned no native result: guard
        /// misses, and any call that raised. `try_calls - try_misses` is
        /// exactly the number of native results.
        #[getter]
        fn try_misses(&self) -> u64 {
            self.try_misses.load(Ordering::Relaxed)
        }

        /// `(try_calls, try_misses)`, read together. Holding the GIL, no
        /// call can run in between, so the pair is consistent.
        #[getter]
        fn try_counts(&self) -> (u64, u64) {
            (
                self.try_calls.load(Ordering::Relaxed),
                self.try_misses.load(Ordering::Relaxed),
            )
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
        ///   ``int`` subclasses, any non-``int``, values outside i64,
        ///   overflow in any intermediate or final operation, and a loop
        ///   that used up its backedge budget. ``None`` is unambiguous
        ///   because every compiled program returns an ``int``; no partial
        ///   result is ever returned.
        /// - A guard miss raises nothing and runs no Python code on the
        ///   arguments: no ``__int__``, ``__index__``, ``__repr__`` or
        ///   (metaclass) attribute lookup.
        /// - Any exception it does raise is unexpected.
        ///
        /// The GIL stays held throughout; the backedge budget bounds how
        /// long a loop can hold it.
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
            // Not locked read-modify-writes: callers hold the GIL, which
            // serializes them, and these counters are only for diagnostics.
            // Misses are counted where they return, so that a native
            // result pays for nothing but `try_calls`.
            bump(&self.try_calls);
            let arg_count = self.code.arg_count();
            if args.len() != arg_count {
                return self.miss();
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
                match exact_i64_or_none(&item) {
                    Ok(Some(value)) => *slot = value,
                    Ok(None) => return self.miss(),
                    Err(error) => return self.failed(error),
                }
            }
            match self.code.call(values) {
                (STATUS_OK, value) => Ok(Some(value)),
                (STATUS_OVERFLOW | STATUS_BUDGET_EXHAUSTED, _) => self.miss(),
                // The arity was checked above, so this is a backend bug.
                (status, _) => self.failed(PyRuntimeError::new_err(format!(
                    "unexpected native status {status}"
                ))),
            }
        }

        /// Count a guard miss and report it.
        #[inline(always)]
        fn miss(&self) -> PyResult<Option<i64>> {
            bump(&self.try_misses);
            Ok(None)
        }

        /// Count a call that raised, as one more call with no native
        /// result, and raise.
        #[cold]
        fn failed(&self, error: PyErr) -> PyResult<Option<i64>> {
            bump(&self.try_misses);
            Err(error)
        }
    }

    /// A proof that `range`, as one function looks it up, is the builtin
    /// `range`, which stays valid until the function's globals or builtins
    /// change in a way that could affect it.
    ///
    /// `RangeGuard(globals, builtins)` takes the function's `__globals__` and
    /// `__builtins__`. It raises `UnguardableBinding` if they are not exact
    /// dicts, if either has a key that is not an exact `str`, or if `range`
    /// does not resolve to the builtin `range` type. It runs no Python code.
    #[pyclass(frozen, module = "speedysnek._native")]
    struct RangeGuard {
        guard: Guard,
    }

    #[pymethods]
    impl RangeGuard {
        #[new]
        fn new(globals: &Bound<'_, PyAny>, builtins: &Bound<'_, PyAny>) -> PyResult<Self> {
            match Guard::new(globals.py(), globals, builtins)? {
                Ok(guard) => Ok(RangeGuard { guard }),
                Err(Unguardable(reason)) => Err(UnguardableBinding::new_err(reason)),
            }
        }

        /// Whether the binding still holds. Once false, it stays false.
        #[getter]
        fn valid(&self) -> bool {
            self.guard.valid()
        }
    }

    /// An `I64Function` that may only stand in for its Python function
    /// while a `RangeGuard` holds.
    ///
    /// The `NativeWrapper` checks the guard before every native call and
    /// reports a failed guard to the JIT state instead of running the code.
    /// `try_call` has `I64Function.try_call`'s contract, with a failed
    /// guard as one more guard miss.
    #[pyclass(frozen, module = "speedysnek._native")]
    struct GuardedI64Function {
        function: Py<I64Function>,
        guard: Py<RangeGuard>,
    }

    #[pymethods]
    impl GuardedI64Function {
        #[new]
        fn new(function: Py<I64Function>, guard: Py<RangeGuard>) -> Self {
            GuardedI64Function { function, guard }
        }

        #[getter]
        fn function(&self, py: Python<'_>) -> Py<I64Function> {
            self.function.clone_ref(py)
        }

        #[getter]
        fn guard(&self, py: Python<'_>) -> Py<RangeGuard> {
            self.guard.clone_ref(py)
        }

        #[getter]
        fn arg_count(&self) -> usize {
            self.function.get().code.arg_count()
        }

        #[pyo3(signature = (args, /))]
        fn try_call(&self, args: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
            let args = args
                .cast_exact::<PyTuple>()
                .map_err(|_| PyTypeError::new_err("args must be a tuple"))?;
            if !self.guard.get().guard.valid() {
                return Ok(None);
            }
            self.function.get().try_call_tuple(args)
        }
    }

    /// After this many calls in a row that could not run natively, a
    /// compiled wrapper parks its specialization.
    ///
    /// An occasional miss (one float, one overflow) costs one fallback and
    /// must not throw a useful specialization away: any native result
    /// starts the count again. 32 misses with no native result in between
    /// say the calls have changed shape, while bounding what they cost
    /// before parking to a few microseconds, far less than compiling did.
    /// A fixed count, never tuned from timings.
    const MISS_LIMIT: u32 = 32;

    /// A parked wrapper re-probes after this many parked calls: the next
    /// call that could run natively then tries the specialization again
    /// (or, if it was parked during warmup, compiles it).
    ///
    /// Long enough that a parked function pays about one native attempt
    /// per 64 calls, at most a nanosecond or two per call on average;
    /// short enough that a workload that changes back to exact ints gets
    /// native code again within 64 calls. A fixed count, never tuned from
    /// timings.
    const REPROBE_INTERVAL: u32 = 64;

    /// `HotSlot` modes.
    const ACTIVE: u8 = 0;
    const PARKED: u8 = 1;
    const RETIRED: u8 = 2;

    /// One wrapper's compiled specialization, held in Rust so that a warmed
    /// call reads no Python attribute (see `speedysnek._decorator`), and the
    /// wrapper's miss policy.
    ///
    /// `snekit` gives one empty slot to both the `NativeWrapper` and its
    /// `_JitState`. The state publishes its specialization into it once,
    /// after the specialization is complete and the status is `COMPILED`,
    /// and retires it whenever `state.specialization` is assigned again:
    /// on every transition to `DISABLED`, and when a test replaces the
    /// specialization. Both happen with the GIL held.
    ///
    /// - Write-once: `publish` succeeds at most once, and what it stored is
    ///   never changed or dropped before the slot itself.
    /// - One-way retirement: once retired, the slot is never used again.
    /// - Only a genuine `I64Function` or `GuardedI64Function` is published.
    ///   Anything else (only test doubles use another `function`) leaves
    ///   the slot empty, and the wrapper keeps reading the state.
    ///
    /// Its `mode` is active, parked or retired. Only the state changes it,
    /// with `park`, `unpark` and `retire`, in step with its `status`; the
    /// wrapper asks the state to when a call shows it is time:
    ///
    /// - While active and published, calls use the slot. `misses` counts
    ///   the calls in a row that could not run natively (keywords, another
    ///   arity, a guard miss); a native result resets it. The call that
    ///   makes it `MISS_LIMIT` asks the state to park.
    /// - While parked, published or not (a function parked during warmup
    ///   has nothing to publish yet), calls run the original function
    ///   without a native attempt and count `parked_calls`. Once it has
    ///   reached `REPROBE_INTERVAL`, the next call that could run natively
    ///   is a re-probe, and starts the count again.
    ///
    /// Reading it costs two atomic loads and takes no lock. It holds strong
    /// references to everything a native call uses, and the wrapper holds
    /// the slot, so nothing is freed while a call runs. The counters are
    /// plain loads and stores: every caller holds the GIL.
    ///
    /// It is not tracked by the garbage collector: a code object, an
    /// `I64Function` and a `RangeGuard` are not either, so no reference
    /// cycle the collector could break runs through the slot.
    #[pyclass(frozen, module = "speedysnek._native")]
    struct HotSlot {
        hot: OnceLock<Hot>,
        mode: AtomicU8,
        /// While active: calls in a row that could not run natively.
        misses: AtomicU32,
        /// While parked: calls since parking or the last re-probe, at most
        /// `REPROBE_INTERVAL`.
        parked_calls: AtomicU32,
    }

    /// What a published `HotSlot` holds.
    struct Hot {
        /// The code object that was compiled. A call may only use the slot
        /// while it is still the function's `__code__`.
        code: Py<PyAny>,
        function: Py<I64Function>,
        /// The `RangeGuard` the machine code depends on, if any.
        guard: Option<Py<RangeGuard>>,
    }

    #[pymethods]
    impl HotSlot {
        #[new]
        fn new() -> Self {
            HotSlot {
                hot: OnceLock::new(),
                mode: AtomicU8::new(ACTIVE),
                misses: AtomicU32::new(0),
                parked_calls: AtomicU32::new(0),
            }
        }

        /// Publish `function`, compiled from `code`: the specialization's
        /// `function`, an `I64Function` or a `GuardedI64Function`. Any other
        /// value leaves the slot empty. Raises `RuntimeError` if the slot
        /// was already published.
        #[pyo3(signature = (code, function, /))]
        fn publish(&self, code: &Bound<'_, PyAny>, function: &Bound<'_, PyAny>) -> PyResult<()> {
            // SAFETY: `code` is a live object and the GIL is held (`Bound`).
            if unsafe { ffi::PyCode_Check(code.as_ptr()) } == 0 {
                return Err(PyTypeError::new_err("code must be a code object"));
            }
            let (function, guard) = if let Ok(function) = function.cast_exact::<I64Function>() {
                (function.clone().unbind(), None)
            } else if let Ok(guarded) = function.cast_exact::<GuardedI64Function>() {
                let guarded = guarded.get();
                let py = function.py();
                (
                    guarded.function.clone_ref(py),
                    Some(guarded.guard.clone_ref(py)),
                )
            } else {
                return Ok(());
            };
            let hot = Hot {
                code: code.clone().unbind(),
                function,
                guard,
            };
            self.hot
                .set(hot)
                .map_err(|_| PyRuntimeError::new_err("the hot slot is already published"))
        }

        /// Stop every later call from using the slot. Idempotent.
        fn retire(&self) {
            self.mode.store(RETIRED, Ordering::Release);
        }

        /// Park an active slot: later calls run the original function and
        /// count towards the next re-probe. Returns whether it parked; a
        /// parked or retired slot stays as it is.
        fn park(&self) -> bool {
            let parked = self
                .mode
                .compare_exchange(ACTIVE, PARKED, Ordering::AcqRel, Ordering::Acquire)
                .is_ok();
            if parked {
                self.parked_calls.store(0, Ordering::Relaxed);
            }
            parked
        }

        /// Make a parked slot active again, with no misses counted. Returns
        /// whether it changed; an active or retired slot stays as it is.
        fn unpark(&self) -> bool {
            self.misses.store(0, Ordering::Relaxed);
            self.mode
                .compare_exchange(PARKED, ACTIVE, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        }

        /// Whether calls can use the slot: published, active and not
        /// retired. A diagnostic for tests.
        #[getter]
        fn live(&self) -> bool {
            self.get().is_some()
        }

        /// Whether the slot is parked. A diagnostic.
        #[getter]
        fn parked(&self) -> bool {
            self.mode.load(Ordering::Acquire) == PARKED
        }

        /// While active, the calls in a row that could not run natively. A
        /// diagnostic.
        #[getter]
        fn misses(&self) -> u32 {
            self.misses.load(Ordering::Relaxed)
        }

        /// While parked, the calls since parking or the last re-probe, at
        /// most `REPROBE_INTERVAL`. A diagnostic.
        #[getter]
        fn parked_calls(&self) -> u32 {
            self.parked_calls.load(Ordering::Relaxed)
        }
    }

    impl HotSlot {
        /// The published specialization, if the slot is active.
        #[inline]
        fn get(&self) -> Option<&Hot> {
            if self.mode.load(Ordering::Acquire) != ACTIVE {
                return None;
            }
            self.hot.get()
        }

        /// A call that used the slot ran natively.
        #[inline]
        fn hit(&self) {
            self.misses.store(0, Ordering::Relaxed);
        }

        /// A call that used the slot could not run natively. Returns `true`
        /// exactly once per run of misses, on the one that reaches
        /// `MISS_LIMIT`.
        #[inline]
        fn miss(&self) -> bool {
            let misses = self.misses.load(Ordering::Relaxed).saturating_add(1);
            self.misses.store(misses, Ordering::Relaxed);
            misses == MISS_LIMIT
        }

        /// Count a parked call. Returns whether the interval has passed, so
        /// that this call may re-probe if it could run natively.
        #[inline]
        fn reprobe_due(&self) -> bool {
            let calls = self.parked_calls.load(Ordering::Relaxed);
            if calls >= REPROBE_INTERVAL {
                return true;
            }
            self.parked_calls.store(calls + 1, Ordering::Relaxed);
            false
        }

        /// A re-probe is being made: start the interval again.
        #[inline]
        fn reprobing(&self) {
            self.parked_calls.store(0, Ordering::Relaxed);
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
    /// `_JitState`; this object only reads `state._specialization` (the
    /// slot behind its `specialization` property) and `state.status` and
    /// calls the state's `cold_call`, `invalidate`, `binding_changed`,
    /// `native_failed`, `repeated_misses`, `reprobe_succeeded` and `reprobe`
    /// methods. It binds as a method like a function.
    ///
    /// While its `HotSlot` is live, a call that can use the slot reads none
    /// of the state's attributes: the slot holds the same specialization,
    /// and every call that would make the state change goes to the generic
    /// path, which makes the change exactly as before. While the slot is
    /// parked, a call reads none of them either: it runs the original
    /// function, and only a re-probe calls into the state.
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
        /// The slot the state publishes into; `None` for a wrapper made
        /// without one, which always takes the generic path.
        hot: Option<Py<HotSlot>>,
        /// Whether `func` is a Python function (`FunctionType`, which
        /// cannot be subclassed). Only then can a call use the slot.
        plain_function: bool,
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
        #[pyo3(signature = (func, state, cold, hot = None, /))]
        fn new(
            func: Py<PyAny>,
            state: Py<PyAny>,
            cold: Py<PyAny>,
            hot: Option<Py<HotSlot>>,
        ) -> Self {
            // SAFETY: `func` is a live object; constructing a pyclass holds
            // the GIL. `PyFunction_Check` only compares the type pointer.
            let plain_function = unsafe { ffi::PyFunction_Check(func.as_ptr()) } != 0;
            NativeWrapper {
                func,
                state,
                cold,
                hot,
                plain_function,
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

        /// The hot slot first, when the call may use it. Otherwise the
        /// generic path: the same logic, line for line, as the Python
        /// closure `@snekit` returned before (kept in
        /// `benchmarks/experiments/native_wrapper.py`).
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
            if let Some(slot) = self.slot() {
                match slot.mode.load(Ordering::Acquire) {
                    ACTIVE => {
                        if let Some(hot) = slot.hot.get()
                            && self.is_current(py, hot)
                        {
                            return self.hot_call(py, slot, hot, args, kwargs);
                        }
                    }
                    PARKED => {
                        if let Some(result) = self.parked_call(py, slot, args, kwargs)? {
                            return Ok(result);
                        }
                    }
                    _ => {}
                }
            }
            let state = self.state.bind(py);
            let mut spec = state.getattr(intern!(py, "_specialization"))?;
            if spec.is_none() {
                if !state.getattr(intern!(py, "status"))?.is(&self.cold) {
                    return func.call(args, kwargs);
                }
                // Warmup profile: could this call have run natively? Its
                // positional count when it could (the state checks it
                // against the function's arity), -1 when it could not.
                let fits = if kwargs.is_none_or(|kwargs| kwargs.is_empty()) && exact_i64s(args) {
                    args.len() as isize
                } else {
                    -1
                };
                spec = state.call_method1(intern!(py, "cold_call"), (fits,))?;
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
                    Ok(Native::Result(result)) => return Ok(result),
                    Ok(Native::Miss) => {}
                    // `range` may no longer be the builtin: never again.
                    Ok(Native::BindingChanged) => {
                        state.call_method0(intern!(py, "binding_changed"))?;
                    }
                    Err(error) => self.native_failed(py, error)?,
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
            visit.call(&self.hot)?;
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
        /// The hot slot, if this wrapper may use one: it has a slot and
        /// wraps a Python function.
        #[inline(always)]
        fn slot(&self) -> Option<&HotSlot> {
            if !self.plain_function {
                return None;
            }
            Some(self.hot.as_ref()?.get())
        }

        /// Whether the function's current `__code__` and `range` binding
        /// are still the ones `hot` was compiled for. Reads no Python
        /// attribute. When they are not, the call takes the generic path,
        /// which makes the state transition a stale `__code__` or a failed
        /// range guard calls for.
        #[inline(always)]
        fn is_current(&self, _py: Python<'_>, hot: &Hot) -> bool {
            // SAFETY: `func` is a `FunctionType` (checked when the wrapper
            // was made; it is never replaced), it is alive because `self`
            // owns it, and `_py` proves the GIL is held. `PyFunction_GetCode`
            // returns a borrowed reference, which is only compared, never
            // dereferenced. It cannot fail for a function.
            let code = unsafe { ffi::PyFunction_GetCode(self.func.as_ptr()) };
            if code != hot.code.as_ptr() {
                return false;
            }
            hot.guard
                .as_ref()
                .is_none_or(|guard| guard.get().guard.valid())
        }

        /// A call through the live slot. The slot holds exactly
        /// `state.specialization`, so each outcome is the generic path's
        /// for the same call; it also counts native results and misses.
        #[inline(always)]
        fn hot_call<'py>(
            &self,
            py: Python<'py>,
            slot: &HotSlot,
            hot: &Hot,
            args: &Bound<'py, PyTuple>,
            kwargs: Option<&Bound<'py, PyDict>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let func = self.func.bind(py);
            let function = hot.function.get();
            // As in the generic path, a call with keywords or another
            // arity (an omitted default) makes no native call at all.
            if kwargs.is_some_and(|kwargs| !kwargs.is_empty())
                || args.len() != function.code.arg_count()
            {
                self.missed(py, slot)?;
                return func.call(args, kwargs);
            }
            match function.try_call_tuple(args) {
                Ok(Some(value)) => {
                    slot.hit();
                    return Ok(value.into_pyobject(py)?.into_any());
                }
                Ok(None) => self.missed(py, slot)?,
                Err(error) => self.native_failed(py, error)?,
            }
            func.call(args, kwargs)
        }

        /// Count a call that could not run natively; on the `MISS_LIMIT`th
        /// in a row, ask the state to park the specialization.
        #[inline(always)]
        fn missed(&self, py: Python<'_>, slot: &HotSlot) -> PyResult<()> {
            if slot.miss() {
                return self.repeated_misses(py);
            }
            Ok(())
        }

        #[cold]
        #[inline(never)]
        fn repeated_misses(&self, py: Python<'_>) -> PyResult<()> {
            self.state
                .bind(py)
                .call_method0(intern!(py, "repeated_misses"))?;
            Ok(())
        }

        /// A call while the slot is parked: the original function, with no
        /// native attempt, except for a re-probe.
        ///
        /// Once `REPROBE_INTERVAL` parked calls have passed, the next call
        /// that could run natively (no keywords, only exact ints in i64,
        /// and, when compiled, the compiled arity) is the re-probe: it
        /// tries the specialization, and unparks it if that gives a native
        /// result, or, for a function parked during warmup, asks the state
        /// to compile it. Other calls never pay for a native attempt.
        ///
        /// Returns `None` to send the call to the generic path: when
        /// `__code__` or the range binding changed, which disables the
        /// specialization exactly as for a live slot, or when a re-probe
        /// just compiled the function, which then runs the call natively.
        fn parked_call<'py>(
            &self,
            py: Python<'py>,
            slot: &HotSlot,
            args: &Bound<'py, PyTuple>,
            kwargs: Option<&Bound<'py, PyDict>>,
        ) -> PyResult<Option<Bound<'py, PyAny>>> {
            let func = self.func.bind(py);
            let hot = slot.hot.get();
            if let Some(hot) = hot
                && !self.is_current(py, hot)
            {
                return Ok(None);
            }
            if !slot.reprobe_due()
                || kwargs.is_some_and(|kwargs| !kwargs.is_empty())
                || hot.is_some_and(|hot| args.len() != hot.function.get().code.arg_count())
                || !exact_i64s(args)
            {
                return func.call(args, kwargs).map(Some);
            }
            slot.reprobing();
            let state = self.state.bind(py);
            let Some(hot) = hot else {
                // Parked during warmup: nothing is compiled yet. The state
                // checks the arity and compiles at most once.
                let spec = state.call_method1(intern!(py, "reprobe"), (args.len(),))?;
                if spec.is_none() {
                    return func.call(args, kwargs).map(Some);
                }
                return Ok(None);
            };
            match hot.function.get().try_call_tuple(args) {
                Ok(Some(value)) => {
                    state.call_method0(intern!(py, "reprobe_succeeded"))?;
                    return Ok(Some(value.into_pyobject(py)?.into_any()));
                }
                Ok(None) => {}
                Err(error) => self.native_failed(py, error)?,
            }
            func.call(args, kwargs).map(Some)
        }

        /// Report an error from the native attempt to the state, which
        /// disables the specialization; the call then falls back. A
        /// `BaseException` such as `KeyboardInterrupt` propagates instead.
        fn native_failed(&self, py: Python<'_>, error: PyErr) -> PyResult<()> {
            if !error.is_instance_of::<PyException>(py) {
                return Err(error);
            }
            self.state
                .bind(py)
                .call_method1(intern!(py, "native_failed"), (error.into_value(py),))?;
            Ok(())
        }

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

    /// What a native attempt did.
    enum Native<'py> {
        /// The native code ran and returned this.
        Result(Bound<'py, PyAny>),
        /// A guard miss, or a call with another arity: nothing ran.
        Miss,
        /// The specialization's `RangeGuard` failed: nothing ran, and the
        /// specialization must not be used again.
        BindingChanged,
    }

    /// The native attempt.
    ///
    /// A real specialization's `function` is an `I64Function`, run through
    /// the same helper as `try_call`, or, for a function that uses `range`,
    /// a `GuardedI64Function`, whose guard is checked first. Only the second
    /// kind pays for the check. `function` is a strong reference, so the
    /// machine code stays alive for the whole call even if the
    /// specialization is replaced meanwhile. Any other `function` (only test
    /// doubles use one) means: call `spec.fast_call`, which has `try_call`'s
    /// contract.
    fn native_call<'py>(
        spec: &Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
    ) -> PyResult<Native<'py>> {
        let py = spec.py();
        let function = spec.getattr(intern!(py, "function"))?;
        let function = if let Ok(function) = function.cast::<I64Function>() {
            function.get()
        } else if let Ok(guarded) = function.cast::<GuardedI64Function>() {
            let guarded = guarded.get();
            if !guarded.guard.get().guard.valid() {
                return Ok(Native::BindingChanged);
            }
            // Borrowed from `guarded`, which `function` keeps alive.
            guarded.function.get()
        } else {
            if args.len() != spec.getattr(intern!(py, "arg_count"))?.extract::<usize>()? {
                return Ok(Native::Miss);
            }
            let result = spec.getattr(intern!(py, "fast_call"))?.call1((args,))?;
            return Ok(if result.is_none() {
                Native::Miss
            } else {
                Native::Result(result)
            });
        };
        // As in the Python wrapper, a call with another arity (an omitted
        // default) makes no native call at all.
        if args.len() != function.code.arg_count() {
            return Ok(Native::Miss);
        }
        match function.try_call_tuple(args)? {
            Some(value) => Ok(Native::Result(value.into_pyobject(py)?.into_any())),
            None => Ok(Native::Miss),
        }
    }

    /// Whether every argument is an exact `int` in the signed 64-bit range,
    /// as the fast path requires. Runs no Python code.
    fn exact_i64s(args: &Bound<'_, PyTuple>) -> bool {
        args.iter_borrowed()
            .all(|item| matches!(exact_i64_or_none(&item), Ok(Some(_))))
    }

    /// Arguments that `try_call` passes without a heap allocation.
    const INLINE_ARGS: usize = 8;

    /// Add one to a diagnostic counter. A load and a store, not a locked
    /// read-modify-write: every caller holds the GIL.
    #[inline]
    fn bump(counter: &AtomicU64) {
        let value = counter.load(Ordering::Relaxed);
        counter.store(value.wrapping_add(1), Ordering::Relaxed);
    }

    /// Compile a serialized program (see `src/program.rs`) for exactly
    /// `arg_count` arguments.
    ///
    /// A loop may take at most `backedge_budget` backward jumps per call
    /// (`DEFAULT_BACKEDGE_BUDGET` if omitted); it must lie in
    /// `0..=MAX_BACKEDGE_BUDGET`, so it can only be lowered.
    ///
    /// Raises ``TypeError`` or ``ValueError`` for a malformed program or
    /// budget and ``CodegenError`` if Cranelift fails.
    #[pyfunction]
    #[pyo3(signature = (program, arg_count, /, backedge_budget = None))]
    fn compile_i64(
        program: &Bound<'_, PyAny>,
        arg_count: &Bound<'_, PyAny>,
        backedge_budget: Option<&Bound<'_, PyAny>>,
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
        let budget_out_of_range = || {
            PyValueError::new_err(format!(
                "backedge_budget must be in 0..={MAX_BACKEDGE_BUDGET}"
            ))
        };
        let backedge_budget = match backedge_budget.map(exact_i64) {
            None => DEFAULT_BACKEDGE_BUDGET,
            Some(Ok(n)) => u64::try_from(n)
                .ok()
                .filter(|n| *n <= MAX_BACKEDGE_BUDGET)
                .ok_or_else(budget_out_of_range)?,
            Some(Err(IntError::OutOfRange)) => return Err(budget_out_of_range()),
            Some(Err(IntError::NotExactInt)) => {
                return Err(PyTypeError::new_err("backedge_budget must be an exact int"));
            }
        };
        let ops = decode(program)?;
        // `NativeCode::compile` validates again; this gives a clear error.
        crate::program::validate(&ops, arg_count).map_err(PyValueError::new_err)?;
        let code =
            NativeCode::compile(&ops, arg_count, backedge_budget).map_err(CodegenError::new_err)?;
        Ok(I64Function {
            code,
            try_calls: AtomicU64::new(0),
            try_misses: AtomicU64::new(0),
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
