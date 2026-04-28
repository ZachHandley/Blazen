//! Abstract HTTP-client base class for user-supplied HTTP backends.
//!
//! Subclass [`PyHttpClient`] from Python to plug a custom HTTP transport into
//! Blazen (e.g., wrapping ``httpx``, ``aiohttp``, or a corporate proxy). The
//! subclass overrides the async ``send`` and ``send_streaming`` methods;
//! Blazen drives them through the [`blazen_llm::http::HttpClient`] trait when
//! the user passes the subclass instance into a provider that accepts a
//! custom HTTP client.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

// ---------------------------------------------------------------------------
// PyHttpClient
// ---------------------------------------------------------------------------

/// Abstract base class for an HTTP transport.
///
/// Subclass and override ``send`` and ``send_streaming`` to provide a custom
/// HTTP backend. ``send`` returns a fully-buffered response as a dict
/// ``{"status": int, "headers": list[tuple[str, str]], "body": bytes}``;
/// ``send_streaming`` returns an async iterator of ``bytes`` chunks.
///
/// Example:
///     >>> class HttpxClient(HttpClient):
///     ...     async def send(self, request):
///     ...         resp = await self._client.request(
///     ...             request["method"], request["url"],
///     ...             headers=request["headers"], content=request.get("body"),
///     ...         )
///     ...         return {
///     ...             "status": resp.status_code,
///     ...             "headers": list(resp.headers.items()),
///     ...             "body": resp.content,
///     ...         }
#[gen_stub_pyclass]
#[pyclass(name = "HttpClient", subclass)]
pub struct PyHttpClient;

#[gen_stub_pymethods]
#[pymethods]
impl PyHttpClient {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Send a request and return a fully-buffered response.
    ///
    /// Subclasses must override. Default implementation raises
    /// ``NotImplementedError``.
    #[pyo3(signature = (request))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, dict[str, typing.Any]]", imports = ("typing",)))]
    fn send<'py>(&self, py: Python<'py>, request: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let _ = request;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Err::<Py<PyAny>, _>(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override send()",
            ))
        })
    }

    /// Send a request and return a streaming response.
    ///
    /// Subclasses must override. Default implementation raises
    /// ``NotImplementedError``.
    #[pyo3(signature = (request))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, typing.AsyncIterator[bytes]]", imports = ("typing",)))]
    fn send_streaming<'py>(
        &self,
        py: Python<'py>,
        request: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = request;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Err::<Py<PyAny>, _>(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override send_streaming()",
            ))
        })
    }
}
