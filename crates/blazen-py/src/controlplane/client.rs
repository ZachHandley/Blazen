//! Python wrapper for the orchestrator-side control-plane
//! [`blazen_controlplane::Client`].

use std::path::PathBuf;
use std::sync::Arc;

use futures_util::StreamExt;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::{Mutex, mpsc};
use uuid::Uuid;

use blazen_controlplane::Client;
use blazen_core::distributed::{OrchestratorClient, RunEvent, SubmitWorkflowRequest};
use blazen_core::error::WorkflowError;

use crate::convert::py_to_json;

use super::types::{
    PyControlPlaneResourceHint, run_event_to_pydict, snapshot_to_pydict, worker_info_to_pydict,
};
use super::worker::{ControlPlaneException, cp_err};

// ===========================================================================
// PyControlPlaneClient
// ===========================================================================

/// Orchestrator-side handle for the Blazen control plane. Construct via
/// [`PyControlPlaneClient::connect`].
#[gen_stub_pyclass]
#[pyclass(name = "ControlPlaneClient")]
pub struct PyControlPlaneClient {
    inner: Client,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyControlPlaneClient {
    /// Open a connection to the control plane at ``endpoint``.
    ///
    /// Args:
    ///     endpoint: gRPC URI, e.g. ``"http://cp.example.com:7445"`` or
    ///         ``"https://cp.example.com"`` for TLS.
    ///     mtls: Optional ``(cert_path, key_path, ca_path)`` triple for
    ///         mutual TLS. If supplied the three files are loaded as
    ///         PEM and used to build the client TLS config.
    ///
    /// Raises:
    ///     ControlPlaneError: If the endpoint URI is invalid, the TLS
    ///         materials fail to load, or the TCP/HTTP-2 handshake
    ///         fails.
    #[classmethod]
    #[pyo3(signature = (endpoint, *, mtls=None))]
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, ControlPlaneClient]",
        imports = ("typing",)
    ))]
    fn connect<'py>(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'py>,
        endpoint: String,
        mtls: Option<(String, String, String)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let client = match mtls {
                Some((cert, key, ca)) => Client::with_mtls(
                    endpoint,
                    &PathBuf::from(cert),
                    &PathBuf::from(key),
                    &PathBuf::from(ca),
                )
                .await
                .map_err(cp_err)?,
                None => Client::connect(endpoint, None).await.map_err(cp_err)?,
            };
            Ok(Self { inner: client })
        })
    }

    /// Submit a new workflow run.
    ///
    /// Returns:
    ///     A dict mirroring `RunStateSnapshot` (keys: ``run_id``,
    ///     ``status``, ``started_at_ms``, ``completed_at_ms``,
    ///     ``assigned_to``, ``last_event_at_ms``, ``output``,
    ///     ``error``).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        workflow_name,
        input,
        *,
        workflow_version=None,
        required_tags=None,
        idempotency_key=None,
        deadline_ms=None,
        wait_for_worker=true,
        resource_hint=None,
    ))]
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn submit_workflow<'py>(
        &self,
        py: Python<'py>,
        workflow_name: String,
        input: Option<&Bound<'_, PyAny>>,
        workflow_version: Option<u32>,
        required_tags: Option<Vec<String>>,
        idempotency_key: Option<String>,
        deadline_ms: Option<u64>,
        wait_for_worker: bool,
        resource_hint: Option<PyControlPlaneResourceHint>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input_json = match input {
            Some(value) if !value.is_none() => py_to_json(py, value)?,
            _ => serde_json::Value::Null,
        };
        let request = SubmitWorkflowRequest {
            workflow_name,
            workflow_version,
            input: input_json,
            required_tags: required_tags.unwrap_or_default(),
            idempotency_key,
            deadline_ms,
            wait_for_worker,
            resource_hint: resource_hint.map(|h| h.inner),
        };
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let snap = client
                .submit_workflow(request)
                .await
                .map_err(workflow_err_to_py)?;
            Python::attach(|py| Ok(snapshot_to_pydict(py, &snap)?.into_any().unbind()))
        })
    }

    /// Cancel an in-flight run.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict[str, typing.Any]]",
        imports = ("typing",)
    ))]
    fn cancel_workflow<'py>(&self, py: Python<'py>, run_id: String) -> PyResult<Bound<'py, PyAny>> {
        let uuid = parse_run_id(&run_id)?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let snap = client
                .cancel_workflow(uuid)
                .await
                .map_err(workflow_err_to_py)?;
            Python::attach(|py| Ok(snapshot_to_pydict(py, &snap)?.into_any().unbind()))
        })
    }

    /// Look up the current state of a run.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict[str, typing.Any]]",
        imports = ("typing",)
    ))]
    fn describe_workflow<'py>(
        &self,
        py: Python<'py>,
        run_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let uuid = parse_run_id(&run_id)?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let snap = client
                .describe_workflow(uuid)
                .await
                .map_err(workflow_err_to_py)?;
            Python::attach(|py| Ok(snapshot_to_pydict(py, &snap)?.into_any().unbind()))
        })
    }

    /// List currently-connected workers.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, list[dict]]",
        imports = ("typing",)
    ))]
    fn list_workers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let workers = client.list_workers().await.map_err(workflow_err_to_py)?;
            Python::attach(|py| -> PyResult<Py<PyAny>> {
                let list = PyList::empty(py);
                for w in &workers {
                    list.append(worker_info_to_pydict(py, w)?)?;
                }
                Ok(list.into_any().unbind())
            })
        })
    }

    /// Send a drain instruction to the named worker.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, None]",
        imports = ("typing",)
    ))]
    fn drain_worker<'py>(
        &self,
        py: Python<'py>,
        node_id: String,
        immediate: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client
                .drain_worker(node_id, immediate)
                .await
                .map_err(cp_err)
        })
    }

    /// Subscribe to events for a single run. Returns an async iterator
    /// that yields a dict per event until the run terminates.
    fn subscribe_run_events(&self, run_id: String) -> PyResult<PyRunEventStream> {
        let uuid = parse_run_id(&run_id)?;
        let client = self.inner.clone();
        Ok(PyRunEventStream::new_per_run(client, uuid))
    }

    /// Subscribe to events across all runs, optionally filtered by an
    /// AND-list of tag predicates. Returns an async iterator that
    /// yields a dict per event.
    #[pyo3(signature = (required_tags=None))]
    fn subscribe_all(&self, required_tags: Option<Vec<String>>) -> PyRunEventStream {
        let tags = required_tags.unwrap_or_default();
        let client = self.inner.clone();
        PyRunEventStream::new_subscribe_all(client, tags)
    }

    fn __repr__(&self) -> String {
        "ControlPlaneClient(...)".to_owned()
    }
}

// ===========================================================================
// PyRunEventStream — async iterator of run events
// ===========================================================================

/// Channel item carried from the forwarder task into Python land.
type EventItem = Result<RunEvent, WorkflowError>;
/// Lazy slot holding the per-stream `mpsc::Receiver` (after init) or
/// nothing (before init). Boxed via `Arc<Mutex<...>>` so `__anext__`
/// can mutate it with only `&self`.
type LazyReceiver = Arc<Mutex<Option<mpsc::Receiver<EventItem>>>>;

/// Async iterator over run events.
///
/// On first `__anext__` call, spawns a tokio task that owns a clone of
/// the underlying `Client`, opens the subscribe RPC, and forwards each
/// decoded event through an mpsc channel. This sidesteps the borrowed
/// lifetime on `Client::subscribe_run_events` / `subscribe_all` (which
/// would otherwise require an unsafe lifetime extension) — the worker
/// task holds its own clone for as long as the stream is alive.
#[gen_stub_pyclass]
#[pyclass(name = "RunEventStream")]
pub struct PyRunEventStream {
    /// Channel receiver, plumbed in lazily on the first `__anext__`
    /// call.
    rx: LazyReceiver,
    /// Lazy initializer that opens the upstream RPC and starts the
    /// forwarder task. Consumed on first call.
    init: Arc<Mutex<Option<StreamInit>>>,
}

struct StreamInit {
    client: Client,
    kind: SubscribeKind,
}

enum SubscribeKind {
    PerRun(Uuid),
    All(Vec<String>),
}

impl PyRunEventStream {
    fn new_per_run(client: Client, run_id: Uuid) -> Self {
        Self {
            rx: Arc::new(Mutex::new(None)),
            init: Arc::new(Mutex::new(Some(StreamInit {
                client,
                kind: SubscribeKind::PerRun(run_id),
            }))),
        }
    }

    fn new_subscribe_all(client: Client, tags: Vec<String>) -> Self {
        Self {
            rx: Arc::new(Mutex::new(None)),
            init: Arc::new(Mutex::new(Some(StreamInit {
                client,
                kind: SubscribeKind::All(tags),
            }))),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRunEventStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rx_slot = Arc::clone(&self.rx);
        let init_slot = Arc::clone(&self.init);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Lazy init: open the subscribe RPC + spawn forwarder on
            // first __anext__ invocation.
            {
                let mut rx_guard = rx_slot.lock().await;
                if rx_guard.is_none() {
                    let init = init_slot.lock().await.take();
                    if let Some(init) = init {
                        *rx_guard = Some(open_subscribe(init));
                    } else {
                        return Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                            "stream exhausted",
                        ));
                    }
                }
            }

            // Pull one item from the receiver.
            let item = {
                let mut rx_guard = rx_slot.lock().await;
                let Some(rx) = rx_guard.as_mut() else {
                    return Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                        "stream exhausted",
                    ));
                };
                rx.recv().await
            };

            match item {
                Some(Ok(event)) => {
                    Python::attach(|py| Ok(run_event_to_pydict(py, &event)?.into_any().unbind()))
                }
                Some(Err(e)) => Err(workflow_err_to_py(e)),
                None => Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                    "stream exhausted",
                )),
            }
        })
    }
}

/// Open the upstream subscribe RPC and spawn a forwarder task that
/// pumps the borrowed stream's items into an `mpsc::Receiver`. The
/// `Client` clone is owned by the spawned task, which keeps the
/// gRPC channel alive for the stream's lifetime.
fn open_subscribe(init: StreamInit) -> mpsc::Receiver<EventItem> {
    let (tx, rx) = mpsc::channel::<EventItem>(32);
    let StreamInit { client, kind } = init;
    tokio::spawn(async move {
        let stream_result = match kind {
            SubscribeKind::PerRun(uuid) => client.subscribe_run_events(uuid).await,
            SubscribeKind::All(tags) => client
                .subscribe_all(tags)
                .await
                .map_err(|e| WorkflowError::Other(anyhow::anyhow!("subscribe_all failed: {e}"))),
        };
        let mut stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(Err(e)).await;
                return;
            }
        };
        while let Some(item) = stream.next().await {
            if tx.send(item).await.is_err() {
                // Receiver dropped — the Python side cancelled.
                break;
            }
        }
    });
    rx
}

// ===========================================================================
// Helpers
// ===========================================================================

fn parse_run_id(s: &str) -> PyResult<Uuid> {
    Uuid::parse_str(s)
        .map_err(|e| ControlPlaneException::new_err(format!("invalid run_id `{s}`: {e}")))
}

fn workflow_err_to_py(err: WorkflowError) -> PyErr {
    ControlPlaneException::new_err(err.to_string())
}
