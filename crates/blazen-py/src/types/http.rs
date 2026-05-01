//! Abstract HTTP-client base class for user-supplied HTTP backends.
//!
//! Subclass [`PyHttpClient`] from Python to plug a custom HTTP transport into
//! Blazen (e.g., wrapping ``httpx``, ``aiohttp``, or a corporate proxy). The
//! subclass overrides the async ``send`` and ``send_streaming`` methods;
//! Blazen drives them through the [`blazen_llm::http::HttpClient`] trait when
//! the user passes the subclass instance into a provider that accepts a
//! custom HTTP client.

use std::time::Duration;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::http::HttpClientConfig;

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

// ---------------------------------------------------------------------------
// PyHttpClientConfig
// ---------------------------------------------------------------------------

/// Configuration knobs applied when constructing an HTTP client.
///
/// Mirrors [`blazen_llm::http::HttpClientConfig`]: caps the wall-clock
/// duration of a single HTTP request and the TCP/TLS connection-establishment
/// phase. `None` means *no timeout* — the underlying client will wait
/// indefinitely. ``user_agent`` is sent on every request when set.
///
/// Construct via the keyword constructor or the [`HttpClientConfig.unlimited`]
/// factory (no request / connect timeout).
///
/// Example:
///     >>> cfg = HttpClientConfig(request_timeout=30.0, connect_timeout=5.0)
///     >>> unlimited = HttpClientConfig.unlimited()
#[gen_stub_pyclass]
#[pyclass(name = "HttpClientConfig", from_py_object)]
#[derive(Clone)]
pub struct PyHttpClientConfig {
    pub(crate) inner: HttpClientConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHttpClientConfig {
    /// Build a config from optional wall-clock timeouts (in seconds) and an
    /// optional user-agent string.
    #[new]
    #[pyo3(signature = (*, request_timeout=Some(60.0), connect_timeout=Some(10.0), user_agent=None))]
    fn new(
        request_timeout: Option<f64>,
        connect_timeout: Option<f64>,
        user_agent: Option<String>,
    ) -> Self {
        Self {
            inner: HttpClientConfig {
                request_timeout: request_timeout.map(Duration::from_secs_f64),
                connect_timeout: connect_timeout.map(Duration::from_secs_f64),
                user_agent,
            },
        }
    }

    /// Construct a config with neither request nor connect timeout. Useful
    /// for long-polling style endpoints that hold the connection open
    /// indefinitely.
    #[staticmethod]
    fn unlimited() -> Self {
        Self {
            inner: HttpClientConfig::unlimited(),
        }
    }

    /// Maximum wall-clock duration of a single request (seconds).
    /// `None` = unlimited.
    #[getter]
    fn request_timeout(&self) -> Option<f64> {
        self.inner.request_timeout.map(|d| d.as_secs_f64())
    }

    /// Maximum duration of the connection-establishment phase (seconds).
    /// `None` = unlimited.
    #[getter]
    fn connect_timeout(&self) -> Option<f64> {
        self.inner.connect_timeout.map(|d| d.as_secs_f64())
    }

    /// User-Agent header, when set. `None` falls back to the underlying
    /// client's default User-Agent.
    #[getter]
    fn user_agent(&self) -> Option<String> {
        self.inner.user_agent.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "HttpClientConfig(request_timeout={:?}, connect_timeout={:?}, user_agent={:?})",
            self.request_timeout(),
            self.connect_timeout(),
            self.inner.user_agent,
        )
    }
}
