//! Guards on the binding of the builtin `range`, kept current by CPython's
//! dict watchers.
//!
//! A `for` loop over `range(n)` compiles to integer arithmetic. That is only
//! the function's meaning while the name `range`, looked up the way the
//! function's `LOAD_GLOBAL` looks it up, is the genuine builtin `range`
//! type. CPython 3.14 resolves `LOAD_GLOBAL` in the function's globals dict
//! and then in its builtins dict (`func.__builtins__`, fixed when the
//! function is created), and only takes that plain-dict path when both are
//! exact dicts. A [`RangeGuard`] proves the binding once, when it is
//! created, and from then on a dict watcher on both dicts clears its
//! `valid` flag before any change that could affect it takes effect:
//!
//! - adding, replacing or deleting the key `"range"`;
//! - adding, replacing or deleting a key that is not an exact `str` (its
//!   `__eq__` could make it match `"range"`);
//! - clearing either dict, merging a whole dict into an empty one
//!   (`CLONED`), or deallocating it.
//!
//! The flag never becomes true again. Checking it costs one atomic load; a
//! function that does not use `range` has no guard and pays nothing.
//!
//! Nothing here runs Python code: the proof walks the dicts with
//! `PyDict_Next` and compares keys only when both are exact `str`s, and the
//! watcher callback does the same. A namespace that cannot be checked that
//! way (a dict subclass or another mapping, or a key that is not an exact
//! `str`) is rejected.

use std::collections::HashMap;
use std::ffi::{c_char, c_int};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, Weak};

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// `PyDict_WatchEvent` values, from CPython's `cpython/dictobject.h`.
const EVENT_ADDED: c_int = 0;
const EVENT_MODIFIED: c_int = 1;
const EVENT_DELETED: c_int = 2;
const EVENT_DEALLOCATED: c_int = 5;

/// The guarded name, NUL-terminated for `PyUnicode_CompareWithASCIIString`.
const RANGE: &std::ffi::CStr = c"range";

/// `PyDict_WatchCallback`. The C enum argument is passed as an `int`.
type WatchCallback = unsafe extern "C" fn(
    event: c_int,
    dict: *mut ffi::PyObject,
    key: *mut ffi::PyObject,
    new_value: *mut ffi::PyObject,
) -> c_int;

// pyo3-ffi declares `PyDict_Watch`, but not `PyDict_AddWatcher`, whose
// callback type it skips. It is public, non-limited C API since 3.12.
#[cfg_attr(windows, link(name = "python314", kind = "raw-dylib"))]
unsafe extern "C" {
    fn PyDict_AddWatcher(callback: WatchCallback) -> c_int;
}

/// The flags of the live guards on each watched dict, by the dict's
/// address. A dict's entry is removed when it is deallocated, so a new dict
/// at the same address (which is not watched) never inherits it. It holds
/// no Python objects, so no Python code can run while it is locked.
static GUARDS: LazyLock<Mutex<HashMap<usize, Vec<Weak<AtomicBool>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// This module's watcher id, registered on first use.
static WATCHER_ID: Mutex<Option<c_int>> = Mutex::new(None);

fn guards() -> MutexGuard<'static, HashMap<usize, Vec<Weak<AtomicBool>>>> {
    // Nothing panics while holding the lock; recover anyway.
    GUARDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Whether `key` is an exact `str` equal to `"range"`, or `None` if it is
/// not an exact `str`. Runs no Python code.
///
/// # Safety
///
/// `key` must be a valid object pointer and the GIL must be held.
unsafe fn is_range(key: *mut ffi::PyObject) -> Option<bool> {
    // SAFETY: the caller guarantees `key` is valid and the GIL is held. An
    // exact `str` compares by its characters, never through `__eq__`, and
    // `PyUnicode_CompareWithASCIIString` cannot fail.
    unsafe {
        if ffi::PyUnicode_CheckExact(key) == 0 {
            return None;
        }
        Some(ffi::PyUnicode_CompareWithASCIIString(key, RANGE.as_ptr() as *const c_char) == 0)
    }
}

/// Called by CPython before a watched dict changes, with the GIL held.
unsafe extern "C" fn on_dict_event(
    event: c_int,
    dict: *mut ffi::PyObject,
    key: *mut ffi::PyObject,
    _new_value: *mut ffi::PyObject,
) -> c_int {
    let relevant = match event {
        EVENT_ADDED | EVENT_MODIFIED | EVENT_DELETED if !key.is_null() => {
            // SAFETY: CPython passes a valid key for these events and holds
            // the GIL.
            unsafe { is_range(key) }.unwrap_or(true)
        }
        // CLONED, CLEARED, DEALLOCATED and any future event.
        _ => true,
    };
    if relevant {
        let mut guards = guards();
        let address = dict as usize;
        let flags = if event == EVENT_DEALLOCATED {
            guards.remove(&address)
        } else {
            guards.get(&address).cloned()
        };
        drop(guards);
        for flag in flags.into_iter().flatten() {
            if let Some(flag) = flag.upgrade() {
                flag.store(false, Ordering::Relaxed);
            }
        }
    }
    0
}

fn watcher_id(py: Python<'_>) -> PyResult<c_int> {
    let mut id = WATCHER_ID
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(id) = *id {
        return Ok(id);
    }
    // SAFETY: the GIL is held (`py`), and `on_dict_event` has the
    // `PyDict_WatchCallback` signature. The watcher is never cleared: the
    // module is never unloaded.
    let new = unsafe { PyDict_AddWatcher(on_dict_event) };
    if new < 0 {
        return Err(PyErr::fetch(py));
    }
    *id = Some(new);
    Ok(new)
}

/// Why a function's `range` cannot be guarded: it cannot be checked
/// without running Python code, or it is not the builtin `range`.
pub struct Unguardable(pub &'static str);

/// The value bound to `"range"` in `dict`, if any.
///
/// Fails if a key is not an exact `str`: CPython's own lookup could call
/// its `__eq__`, so the result could differ from what the function sees.
fn lookup_range(dict: &Bound<'_, PyDict>) -> Result<Option<*mut ffi::PyObject>, ()> {
    let mut position: ffi::Py_ssize_t = 0;
    let mut key: *mut ffi::PyObject = std::ptr::null_mut();
    let mut value: *mut ffi::PyObject = std::ptr::null_mut();
    let mut found = None;
    // SAFETY: `dict` is a live dict and the GIL is held (it is `Bound`).
    // `PyDict_Next` returns borrowed references that stay valid because
    // nothing in this loop runs Python code or changes the dict.
    unsafe {
        while ffi::PyDict_Next(dict.as_ptr(), &mut position, &mut key, &mut value) != 0 {
            match is_range(key) {
                None => return Err(()),
                Some(true) => found = Some(value),
                Some(false) => {}
            }
        }
    }
    Ok(found)
}

/// A proof that `range`, looked up in one function's globals and then its
/// builtins, is the builtin `range`, valid until either dict changes in a
/// way that could affect it.
pub struct Guard {
    valid: Arc<AtomicBool>,
}

impl Guard {
    pub fn new(
        py: Python<'_>,
        globals: &Bound<'_, PyAny>,
        builtins: &Bound<'_, PyAny>,
    ) -> PyResult<Result<Guard, Unguardable>> {
        // Other mappings, dict subclasses included, are looked up with
        // `__getitem__`, which can run Python code.
        let Ok(globals) = globals.cast_exact::<PyDict>() else {
            return Ok(Err(Unguardable(
                "the function's globals are not an exact dict",
            )));
        };
        let Ok(builtins) = builtins.cast_exact::<PyDict>() else {
            return Ok(Err(Unguardable(
                "the function's builtins are not an exact dict",
            )));
        };

        // Watch first, so that no change between the proof and the first
        // call can be missed.
        let id = watcher_id(py)?;
        let valid = Arc::new(AtomicBool::new(true));
        for dict in [globals, builtins] {
            // SAFETY: `dict` is a live dict and the GIL is held.
            if unsafe { ffi::PyDict_Watch(id, dict.as_ptr()) } != 0 {
                return Err(PyErr::fetch(py));
            }
            let mut guards = guards();
            let flags = guards.entry(dict.as_ptr() as usize).or_default();
            flags.retain(|flag| flag.strong_count() > 0);
            flags.push(Arc::downgrade(&valid));
        }

        // Only the address is taken; nothing is read or written.
        let builtin_range = (&raw mut ffi::PyRange_Type).cast::<ffi::PyObject>();
        let found = match lookup_range(globals) {
            Err(()) => {
                return Ok(Err(Unguardable(
                    "the function's globals have a key that is not an exact str",
                )));
            }
            Ok(Some(value)) => value,
            Ok(None) => match lookup_range(builtins) {
                Err(()) => {
                    return Ok(Err(Unguardable(
                        "the function's builtins have a key that is not an exact str",
                    )));
                }
                Ok(Some(value)) => value,
                Ok(None) => {
                    return Ok(Err(Unguardable(
                        "range is neither a global nor a builtin of the function",
                    )));
                }
            },
        };
        if found != builtin_range {
            return Ok(Err(Unguardable(
                "range is not the builtin range for this function",
            )));
        }
        Ok(Ok(Guard { valid }))
    }

    /// Whether the binding still holds. One relaxed atomic load: guards
    /// are only read and invalidated with the GIL held.
    #[inline]
    pub fn valid(&self) -> bool {
        self.valid.load(Ordering::Relaxed)
    }
}
