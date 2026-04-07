//! Per-step "current session ref registry" plumbing for Python.
//!
//! Python event constructors (`StopEvent(result=obj)`, `Event("Foo", k=v)`)
//! have no `Context` argument — they're called with positional/keyword
//! arguments only. To let those constructors auto-route non-JSON Python
//! values into the per-`Context`
//! [`SessionRefRegistry`](blazen_core::session_ref::SessionRefRegistry),
//! we install the current run's registry as a Python
//! [`contextvars.ContextVar`](https://docs.python.org/3/library/contextvars.html)
//! that lives for the duration of one step's execution and is
//! automatically copied into any `asyncio.Task` the user may spawn.
//!
//! ## Why a Python `ContextVar` instead of `tokio::task_local!`
//!
//! `pyo3-async-runtimes` runs Python coroutines on Python's asyncio
//! event loop (a separate OS thread from the tokio worker that awaits
//! the resulting Rust future, via `call_soon_threadsafe`). A Tokio
//! `task_local!` set on the Tokio worker thread is therefore *not*
//! visible from inside the user's `async def` body. Python `ContextVar`s
//! flow through asyncio Tasks by design, so we use one of those.
//!
//! A Tokio `task_local!` (`CURRENT_SESSION_REGISTRY`) is *also* kept as
//! a fallback for the few synchronous paths that run entirely on the
//! Tokio worker thread — e.g., a Python step that turns out not to be
//! a coroutine, the `WorkflowHandler::result` future, or
//! `PyEventStream::__anext__`.
//!
//! [`current_session_registry`] consults the Python `ContextVar` first
//! (via `Python::attach`) and falls back to the Tokio `task_local!` if
//! Python doesn't have a registry installed.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

use blazen_core::session_ref::SessionRefRegistry;

tokio::task_local! {
    /// Tokio-side fallback for the active session-ref registry. Used by
    /// the handler-result and stream-event paths, which run on the Tokio
    /// worker thread that awaits the result oneshot.
    pub static CURRENT_SESSION_REGISTRY: Arc<SessionRefRegistry>;
}

/// Opaque PyO3-visible holder for an `Arc<SessionRefRegistry>`. Stored
/// inside the Python `ContextVar` so the user-side asyncio Task can
/// carry the registry across `await` points without exposing the inner
/// type to user code.
#[pyclass(name = "_SessionRegistryHandle", frozen)]
pub(crate) struct PySessionRegistryHandle {
    pub(crate) inner: Arc<SessionRefRegistry>,
}

impl PySessionRegistryHandle {
    pub(crate) fn new(inner: Arc<SessionRefRegistry>) -> Self {
        Self { inner }
    }
}

/// Cached `contextvars.ContextVar` instance, lazily created on first
/// use. Initialised via `ContextVar('_blazen_session_registry', default=None)`.
static SESSION_REGISTRY_VAR: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Get (or lazily create) the Python `ContextVar` that stores the
/// current step's [`PySessionRegistryHandle`].
pub(crate) fn session_registry_var(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    let cell = SESSION_REGISTRY_VAR.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        let cv_cls = py.import("contextvars")?.getattr("ContextVar")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("default", py.None())?;
        let var = cv_cls.call(("_blazen_session_registry",), Some(&kwargs))?;
        Ok(var.unbind())
    })?;
    Ok(cell.bind(py))
}

/// Cached Python coroutine helper that sets the session-registry
/// contextvar inside the asyncio Task before awaiting the user
/// coroutine. This is the only reliable way to make the contextvar
/// visible from inside `pyo3-async-runtimes`-driven coroutines, since
/// the Tokio worker that schedules the future runs on a different OS
/// thread from Python's asyncio loop.
static STEP_RUNNER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Get (or lazily compile) the Python helper:
///
/// ```python
/// async def _blazen_run_step(handle, coro):
///     _blazen_session_registry.set(handle)
///     return await coro
/// ```
///
/// The contextvar `set` happens *inside* the asyncio Task, so the
/// inner `coro` (the user's `@step` body) sees the registry on every
/// `current_session_registry()` lookup.
pub(crate) fn step_runner(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    let cell = STEP_RUNNER.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        let var = session_registry_var(py)?;
        let globals = pyo3::types::PyDict::new(py);
        globals.set_item("_blazen_session_registry", var)?;
        let locals = pyo3::types::PyDict::new(py);
        let code = "async def _blazen_run_step(handle, coro):\n    \
                    _blazen_session_registry.set(handle)\n    \
                    return await coro\n";
        py.run(
            &std::ffi::CString::new(code).unwrap(),
            Some(&globals),
            Some(&locals),
        )?;
        let runner = locals.get_item("_blazen_run_step")?.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("failed to compile _blazen_run_step helper")
        })?;
        Ok(runner.unbind())
    })?;
    Ok(cell.bind(py))
}

/// Install `registry` as the current session registry for the duration
/// of `f`. Sets the Python `ContextVar` (so user code inside `f`'s
/// asyncio Task can see it via [`current_session_registry`]) and resets
/// it on the way out using the token returned by `ContextVar.set`.
pub(crate) fn with_python_session_registry<R>(
    py: Python<'_>,
    registry: Arc<SessionRefRegistry>,
    f: impl FnOnce(Python<'_>) -> PyResult<R>,
) -> PyResult<R> {
    let handle = Py::new(py, PySessionRegistryHandle::new(registry))?;
    let var = session_registry_var(py)?;
    let token = var.call_method1("set", (handle,))?;
    let result = f(py);
    let _ = var.call_method1("reset", (token,));
    result
}

/// Look up the registry currently installed in either the Python
/// `ContextVar` (preferred) or the Tokio `task_local!` (fallback).
///
/// Returns `None` when called from a context that has neither installed —
/// typically because the user constructed an event outside of a workflow
/// step.
#[must_use]
pub(crate) fn current_session_registry() -> Option<Arc<SessionRefRegistry>> {
    // Prefer the Python ContextVar so the lookup works from inside an
    // asyncio Task driven by `pyo3-async-runtimes`.
    if let Some(reg) = Python::attach(current_session_registry_py) {
        return Some(reg);
    }
    // Fallback: the Tokio task_local!, used by the handler-result and
    // stream-event paths that run on the Tokio worker thread.
    CURRENT_SESSION_REGISTRY.try_with(Arc::clone).ok()
}

/// Python-side lookup helper. Returns `None` if the contextvar isn't set
/// or holds the default `None` sentinel.
fn current_session_registry_py(py: Python<'_>) -> Option<Arc<SessionRefRegistry>> {
    let var = session_registry_var(py).ok()?;
    let val = var.call_method0("get").ok()?;
    if val.is_none() {
        return None;
    }
    let handle: PyRef<'_, PySessionRegistryHandle> = val.extract().ok()?;
    Some(Arc::clone(&handle.inner))
}

/// Run an async block with the given registry installed as the current
/// Tokio `task_local!`. Used by paths that run on the Tokio worker
/// thread (handler result, stream events) where Python contextvars are
/// not yet on the stack.
pub(crate) async fn with_session_registry<F, T>(registry: Arc<SessionRefRegistry>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CURRENT_SESSION_REGISTRY.scope(registry, fut).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn current_returns_none_outside_scope() {
        assert!(current_session_registry().is_none());
    }

    #[tokio::test]
    async fn current_returns_registry_inside_tokio_scope() {
        let reg = Arc::new(SessionRefRegistry::new());
        let reg_clone = Arc::clone(&reg);
        with_session_registry(reg, async move {
            let got = current_session_registry().expect("registry should be set");
            assert!(Arc::ptr_eq(&got, &reg_clone));
        })
        .await;
    }

    #[tokio::test]
    async fn scope_isolates_concurrent_tasks() {
        // Two concurrent tasks each get their own registry.
        let reg_a = Arc::new(SessionRefRegistry::new());
        let reg_b = Arc::new(SessionRefRegistry::new());
        let _ = reg_a.insert(1_i32).await.unwrap();
        let _ = reg_b.insert(2_i32).await.unwrap();

        let a_clone = Arc::clone(&reg_a);
        let b_clone = Arc::clone(&reg_b);

        let task_a = tokio::spawn(async move {
            with_session_registry(a_clone, async move {
                let got = current_session_registry().unwrap();
                got.len().await
            })
            .await
        });
        let task_b = tokio::spawn(async move {
            with_session_registry(b_clone, async move {
                let got = current_session_registry().unwrap();
                got.len().await
            })
            .await
        });

        assert_eq!(task_a.await.unwrap(), 1);
        assert_eq!(task_b.await.unwrap(), 1);
    }
}
