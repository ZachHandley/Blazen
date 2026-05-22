//! Python wrapper for the gRPC
//! [`blazen_controlplane::client::ModelClient`].
//!
//! Mirrors [`super::client::PyControlPlaneClient`]: each method is an
//! `async fn` returning a Python coroutine via
//! [`pyo3_async_runtimes::tokio::future_into_py`]. Errors lift through
//! the shared [`super::worker::cp_err`] helper to
//! [`super::worker::ControlPlaneException`].
//!
//! Current surface (this file): `connect`, `connect_with_tls`, `status`,
//! `is_loaded`, `load`, `unload`, `load_from_hf`, plus the adapters
//! (`load_adapter`, `unload_adapter`, `list_adapters`), inference
//! (`complete`, `embed`), and multimodal (`generate_image`,
//! `text_to_speech`, `generate_music`, `transcribe`) verbs. Streaming +
//! blob transfer land in later waves.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::StreamExt;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::{Mutex, mpsc};

use blazen_controlplane::ControlPlaneError;
use blazen_controlplane::client::ModelClient;
use blazen_controlplane::model_protocol::{
    AdapterMountStrategyWire, AdapterStatusWire, BackendHintWire, ChatMessageWire, CompleteRequest,
    CompleteResponse, EmbedRequest, EmbedResponse, FetchBlobChunk, FetchBlobRequest,
    GenerateImageRequest, GenerateImageResponse, GenerateMusicRequest, GenerateMusicResponse,
    ImageBlobWire, IsLoadedRequest, ListAdaptersRequest, ListAdaptersResponse, LoadAdapterRequest,
    LoadAdapterResponse, LoadFromHfRequest, LoadFromHfResponse, LoadRequest, LoadResponse,
    MODEL_ENVELOPE_VERSION, ModelStatusWire, PoolWire, StatusRequest, StatusResponse,
    StreamCompleteChunk, TextToSpeechRequest, TextToSpeechResponse, TranscribeRequest,
    TranscribeResponse, UnloadAdapterRequest, UnloadAdapterResponse, UnloadRequest, UnloadResponse,
    UploadBlobChunk, UploadBlobResponse,
};

use super::worker::cp_err;

// ===========================================================================
// PyModelClient
// ===========================================================================

/// gRPC client for a Blazen `BlazenModelServer`. Construct via
/// [`PyModelClient::connect`] or [`PyModelClient::connect_with_tls`].
///
/// Cheaply cloneable on the Rust side; the Python wrapper guards a
/// single shared connection.
#[gen_stub_pyclass]
#[pyclass(name = "ModelClient")]
pub struct PyModelClient {
    inner: ModelClient,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelClient {
    /// Open a plaintext gRPC connection to a Blazen model server.
    ///
    /// Args:
    ///     endpoint: gRPC URI, e.g. ``"http://model.example.com:7446"``.
    ///
    /// Raises:
    ///     ControlPlaneError: If the URI is invalid or the TCP / HTTP-2
    ///         handshake fails.
    #[classmethod]
    #[pyo3(signature = (endpoint))]
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, ModelClient]",
        imports = ("typing",)
    ))]
    fn connect<'py>(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'py>,
        endpoint: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let client = ModelClient::connect(endpoint).await.map_err(cp_err)?;
            Ok(Self { inner: client })
        })
    }

    /// Open a TLS gRPC connection to a Blazen model server.
    ///
    /// Args:
    ///     endpoint: gRPC URI, typically ``"https://..."``.
    ///     ca_cert: Path to a PEM-encoded CA certificate trusted to sign
    ///         the server's leaf certificate.
    ///     client_cert: Optional path to a PEM client certificate for
    ///         mTLS. Must be supplied together with ``client_key``.
    ///     client_key: Optional path to the matching PEM client private
    ///         key. Must be supplied together with ``client_cert``.
    ///
    /// Raises:
    ///     ControlPlaneError: If a PEM file cannot be read, the resulting
    ///         TLS config is rejected, the URI is invalid, or the
    ///         handshake fails.
    #[classmethod]
    #[pyo3(signature = (endpoint, *, ca_cert, client_cert=None, client_key=None))]
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, ModelClient]",
        imports = ("typing",)
    ))]
    fn connect_with_tls<'py>(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'py>,
        endpoint: String,
        ca_cert: String,
        client_cert: Option<String>,
        client_key: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Enforce the cert/key pairing at the Python boundary — passing
        // exactly one is a programming error, not a runtime fault.
        match (&client_cert, &client_key) {
            (Some(_), None) | (None, Some(_)) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "client_cert and client_key must be supplied together",
                ));
            }
            _ => {}
        }
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let tls = build_client_tls(&ca_cert, client_cert.as_deref(), client_key.as_deref())?;
            let client = ModelClient::connect_with_tls(endpoint, Some(tls))
                .await
                .map_err(cp_err)?;
            Ok(Self { inner: client })
        })
    }

    /// Fetch a snapshot of every model registered on the server.
    ///
    /// Args:
    ///     model_id: Reserved for future use (the underlying RPC takes no
    ///         filter today). Accepted but ignored so callers can already
    ///         pass a hint forward.
    ///
    /// Returns:
    ///     A dict ``{"envelope_version": int, "models": list[dict]}``
    ///     where each model dict carries ``id``, ``loaded``,
    ///     ``memory_estimate_bytes``, ``pool`` (one of ``"cpu"``,
    ///     ``"gpu:<index>"``, ``"remote"``), and ``adapters`` (a list of
    ///     ``{adapter_id, scale, source_dir, memory_bytes}`` dicts).
    #[pyo3(signature = (model_id=None))]
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn status<'py>(
        &self,
        py: Python<'py>,
        model_id: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // `model_id` is accepted for forward compatibility; the wire
        // request currently has no filter field, so we discard it.
        let _ = model_id;
        let req = StatusRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
        };
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.status(req).await.map_err(cp_err)?;
            Python::attach(|py| Ok(status_response_to_pydict(py, &resp)?.into_any().unbind()))
        })
    }

    /// Check whether a single model is currently loaded.
    ///
    /// Args:
    ///     model_id: Identifier under which the model was registered.
    ///
    /// Returns:
    ///     ``True`` if the model is loaded, ``False`` otherwise.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, bool]",
        imports = ("typing",)
    ))]
    fn is_loaded<'py>(&self, py: Python<'py>, model_id: String) -> PyResult<Bound<'py, PyAny>> {
        let req = IsLoadedRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id,
        };
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.is_loaded(req).await.map_err(cp_err)?;
            Ok(resp.loaded)
        })
    }

    /// Issue a `Load` RPC.
    ///
    /// Args:
    ///     request: ``{"model_id": str}``. The envelope version is filled
    ///         in automatically.
    ///
    /// Returns:
    ///     ``{"envelope_version": int}``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn load<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model_id = dict_required_str(&request, "model_id")?;
        let req = LoadRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id,
        };
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.load(req).await.map_err(cp_err)?;
            Python::attach(|py| Ok(load_response_to_pydict(py, &resp)?.into_any().unbind()))
        })
    }

    /// Issue an `Unload` RPC.
    ///
    /// Args:
    ///     request: ``{"model_id": str}``.
    ///
    /// Returns:
    ///     ``{"envelope_version": int}``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn unload<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model_id = dict_required_str(&request, "model_id")?;
        let req = UnloadRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id,
        };
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.unload(req).await.map_err(cp_err)?;
            Python::attach(|py| Ok(unload_response_to_pydict(py, &resp)?.into_any().unbind()))
        })
    }

    /// Issue a `LoadFromHf` RPC.
    ///
    /// Args:
    ///     request: dict with required ``model_id`` and ``repo``, and
    ///         optional ``memory_estimate_bytes`` (int), ``backend_hint``
    ///         (one of ``"auto"``, ``"mistral_rs"``, ``"candle"``,
    ///         ``"llama_cpp"``), ``gguf_file`` (str), ``revision`` (str),
    ///         ``hf_token`` (str), and ``extra_options_json`` (str — raw
    ///         JSON, empty string for none).
    ///
    /// Returns:
    ///     ``{"envelope_version": int, "chosen_backend": str}`` where
    ///     ``chosen_backend`` is one of ``"auto"``, ``"mistral_rs"``,
    ///     ``"candle"``, ``"llama_cpp"``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn load_from_hf<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req = load_from_hf_request_from_pydict(&request)?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.load_from_hf(req).await.map_err(cp_err)?;
            Python::attach(|py| {
                Ok(load_from_hf_response_to_pydict(py, &resp)?
                    .into_any()
                    .unbind())
            })
        })
    }

    /// Issue a `LoadAdapter` RPC.
    ///
    /// Args:
    ///     request: dict with required ``model_id`` (str),
    ///         ``adapter_dir`` (str), ``adapter_id`` (str), and ``scale``
    ///         (float).
    ///
    /// Returns:
    ///     ``{"envelope_version": int, "adapter_id": str,
    ///        "memory_bytes": int, "mount_strategy": str}`` where
    ///     ``mount_strategy`` is one of ``"attached"``, ``"rebuilt"``,
    ///     ``"merged"``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn load_adapter<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req = LoadAdapterRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: dict_required_str(&request, "model_id")?,
            adapter_dir: dict_required_str(&request, "adapter_dir")?,
            adapter_id: dict_required_str(&request, "adapter_id")?,
            scale: dict_required_f32(&request, "scale")?,
        };
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.load_adapter(req).await.map_err(cp_err)?;
            Python::attach(|py| {
                Ok(load_adapter_response_to_pydict(py, &resp)?
                    .into_any()
                    .unbind())
            })
        })
    }

    /// Issue an `UnloadAdapter` RPC.
    ///
    /// Args:
    ///     request: ``{"model_id": str, "adapter_id": str}``.
    ///
    /// Returns:
    ///     ``{"envelope_version": int}``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn unload_adapter<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req = UnloadAdapterRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: dict_required_str(&request, "model_id")?,
            adapter_id: dict_required_str(&request, "adapter_id")?,
        };
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.unload_adapter(req).await.map_err(cp_err)?;
            Python::attach(|py| {
                Ok(unload_adapter_response_to_pydict(py, &resp)?
                    .into_any()
                    .unbind())
            })
        })
    }

    /// Issue a `ListAdapters` RPC.
    ///
    /// Args:
    ///     request: ``{"model_id": str}``.
    ///
    /// Returns:
    ///     ``{"envelope_version": int, "adapters": list[dict]}`` — each
    ///     adapter dict carries ``adapter_id``, ``scale``, ``source_dir``,
    ///     ``memory_bytes``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn list_adapters<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req = ListAdaptersRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: dict_required_str(&request, "model_id")?,
        };
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.list_adapters(req).await.map_err(cp_err)?;
            Python::attach(|py| {
                Ok(list_adapters_response_to_pydict(py, &resp)?
                    .into_any()
                    .unbind())
            })
        })
    }

    /// Issue a `Complete` RPC.
    ///
    /// Args:
    ///     request: dict with required ``model_id`` (str) and ``messages``
    ///         (list of ``{"role": str, "text": str, "content_json": str}``
    ///         where ``content_json`` is optional raw JSON). Optional
    ///         keys: ``max_tokens`` (int), ``temperature`` (float),
    ///         ``top_p`` (float), ``stop`` (list[str]),
    ///         ``response_format_json`` (str — raw JSON, empty for none),
    ///         ``extra_json`` (str — raw JSON, empty for none), and
    ///         ``tags`` (dict[str, str]).
    ///
    /// Returns:
    ///     ``{"envelope_version": int, "text": str,
    ///        "prompt_tokens": int|None, "completion_tokens": int|None,
    ///        "finish_reason": str|None, "tool_calls_json": str}``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req = complete_request_from_pydict(&request)?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.complete(req).await.map_err(cp_err)?;
            Python::attach(|py| Ok(complete_response_to_pydict(py, &resp)?.into_any().unbind()))
        })
    }

    /// Issue a `StreamComplete` server-streaming RPC.
    ///
    /// Returns an async iterator that yields one dict per chunk until
    /// the server closes the stream. Each dict carries a ``kind`` field:
    ///
    /// * ``{"kind": "delta", "envelope_version": int, "text": str}`` —
    ///   incremental token group emitted by the backend.
    /// * ``{"kind": "done", "envelope_version": int,
    ///      "prompt_tokens": int|None, "completion_tokens": int|None,
    ///      "finish_reason": str|None}`` — terminal frame.
    ///
    /// Args:
    ///     request: identical schema to :meth:`complete`.
    ///
    /// Raises:
    ///     ControlPlaneError: If the initial RPC fails or a frame
    ///         arrives that cannot be decoded.
    #[gen_stub(override_return_type(
        type_repr = "typing.AsyncIterator[dict]",
        imports = ("typing",)
    ))]
    fn stream_complete(&self, request: Bound<'_, PyDict>) -> PyResult<PyStreamCompleteStream> {
        let req = complete_request_from_pydict(&request)?;
        Ok(PyStreamCompleteStream::new(self.inner.clone(), req))
    }

    /// Issue an `Embed` RPC.
    ///
    /// Args:
    ///     request: dict with required ``model_id`` (str) and ``inputs``
    ///         (list[str]). Optional ``dimensions`` (int) and
    ///         ``extra_json`` (str — raw JSON, empty for none).
    ///
    /// Returns:
    ///     ``{"envelope_version": int, "vectors": list[list[float]],
    ///        "prompt_tokens": int|None}``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn embed<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req = embed_request_from_pydict(&request)?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.embed(req).await.map_err(cp_err)?;
            Python::attach(|py| Ok(embed_response_to_pydict(py, &resp)?.into_any().unbind()))
        })
    }

    /// Issue a `GenerateImage` RPC.
    ///
    /// Args:
    ///     request: dict with required ``model_id`` (str) and ``prompt``
    ///         (str). Optional ``negative_prompt`` (str), ``width`` (int),
    ///         ``height`` (int), ``num_images`` (int), ``seed`` (int), and
    ///         ``image_config_json`` (str — raw JSON, empty for none).
    ///
    /// Returns:
    ///     ``{"envelope_version": int, "images": list[dict]}`` where each
    ///     image dict is ``{"mime": str, "data": bytes}``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn generate_image<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req = generate_image_request_from_pydict(&request)?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.generate_image(req).await.map_err(cp_err)?;
            Python::attach(|py| {
                Ok(generate_image_response_to_pydict(py, &resp)?
                    .into_any()
                    .unbind())
            })
        })
    }

    /// Issue a `TextToSpeech` RPC.
    ///
    /// Args:
    ///     request: dict with required ``model_id`` (str) and ``text``
    ///         (str). Optional ``voice`` (str), ``language`` (str),
    ///         ``sample_rate_hz`` (int), and ``audio_config_json`` (str —
    ///         raw JSON, empty for none).
    ///
    /// Returns:
    ///     ``{"envelope_version": int, "mime": str, "data": bytes,
    ///        "sample_rate_hz": int|None}``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn text_to_speech<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req = text_to_speech_request_from_pydict(&request)?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.text_to_speech(req).await.map_err(cp_err)?;
            Python::attach(|py| {
                Ok(text_to_speech_response_to_pydict(py, &resp)?
                    .into_any()
                    .unbind())
            })
        })
    }

    /// Issue a `GenerateMusic` RPC.
    ///
    /// Args:
    ///     request: dict with required ``model_id`` (str) and ``prompt``
    ///         (str). Optional ``duration_secs`` (float), ``seed`` (int),
    ///         and ``extra_json`` (str — raw JSON, empty for none).
    ///
    /// Returns:
    ///     ``{"envelope_version": int, "mime": str, "data": bytes,
    ///        "sample_rate_hz": int|None}``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn generate_music<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req = generate_music_request_from_pydict(&request)?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.generate_music(req).await.map_err(cp_err)?;
            Python::attach(|py| {
                Ok(generate_music_response_to_pydict(py, &resp)?
                    .into_any()
                    .unbind())
            })
        })
    }

    /// Issue a `Transcribe` RPC.
    ///
    /// Args:
    ///     request: dict with required ``model_id`` (str), ``audio``
    ///         (bytes — raw audio matching ``mime``), and ``mime`` (str).
    ///         Optional ``language`` (str) and ``extra_json`` (str — raw
    ///         JSON, empty for none).
    ///
    /// Returns:
    ///     ``{"envelope_version": int, "text": str,
    ///        "language": str|None, "segments_json": str}``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn transcribe<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req = transcribe_request_from_pydict(&request)?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = client.transcribe(req).await.map_err(cp_err)?;
            Python::attach(|py| {
                Ok(transcribe_response_to_pydict(py, &resp)?
                    .into_any()
                    .unbind())
            })
        })
    }

    /// Upload a blob via the `UploadBlob` client-streaming RPC.
    ///
    /// The supplied async iterator must yield ``bytes`` (or any
    /// buffer-protocol object) one chunk at a time. The Rust side
    /// prepends an [`UploadBlobChunk::Start`] frame with the given
    /// ``blob_id`` / ``mime`` and appends a terminal
    /// [`UploadBlobChunk::End`] after the iterator is exhausted.
    ///
    /// Args:
    ///     chunks: async iterator (anything implementing
    ///         ``__aiter__``/``__anext__``) yielding ``bytes``-like
    ///         objects.
    ///     blob_id: caller-chosen blob id. Empty string is allowed but
    ///         server policy may reject it.
    ///     mime: optional MIME / content-type hint. Empty string omits
    ///         the field on the wire (``content_type: None``).
    ///
    /// Returns:
    ///     ``{"envelope_version": int, "blob_id": str,
    ///        "bytes_received": int}`` once the server reads ``End``.
    ///
    /// Raises:
    ///     ControlPlaneError: If the upstream RPC fails.
    ///     TypeError: If the iterator yields a non-bytes-like value.
    #[pyo3(signature = (chunks, *, blob_id = String::new(), mime = String::new()))]
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, dict]",
        imports = ("typing",)
    ))]
    fn upload_blob<'py>(
        &self,
        py: Python<'py>,
        chunks: Bound<'py, PyAny>,
        blob_id: String,
        mime: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Capture asyncio task locals up front — the forwarder task is
        // spawned on tokio and the calling event loop will be gone by
        // the time `__anext__` fires.
        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
        // Promote to a `Py<PyAny>` so we can move it into the spawned
        // task. The Python object may not already be an async iterator —
        // call `__aiter__` if present to be tolerant of both
        // `AsyncIterable` and `AsyncIterator` arguments.
        let iter_obj: Py<PyAny> = if chunks.hasattr("__aiter__")? {
            chunks.call_method0("__aiter__")?.unbind()
        } else {
            chunks.unbind()
        };
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = mpsc::channel::<UploadBlobChunk>(16);
            // Send the `Start` frame synchronously so the server sees
            // metadata before any data lands.
            let content_type = if mime.is_empty() { None } else { Some(mime) };
            if tx
                .send(UploadBlobChunk::Start {
                    envelope_version: MODEL_ENVELOPE_VERSION,
                    blob_id,
                    total_bytes: None,
                    content_type,
                })
                .await
                .is_err()
            {
                return Err(cp_err(ControlPlaneError::Transport(
                    "upload_blob: receiver dropped before Start frame".to_owned(),
                )));
            }

            // Forwarder: walks the Python async iterator, pushing one
            // `Data` chunk per yielded payload, then a single `End`.
            let forwarder_tx = tx.clone();
            tokio::spawn(forward_python_chunks(iter_obj, locals, forwarder_tx));
            // Drop our own clone so the channel closes when the
            // forwarder is done. Otherwise `upload_blob` would block
            // forever on `recv()`.
            drop(tx);

            let resp = client.upload_blob(rx).await.map_err(cp_err)?;
            Python::attach(|py| {
                Ok(upload_blob_response_to_pydict(py, &resp)?
                    .into_any()
                    .unbind())
            })
        })
    }

    /// Fetch a blob via the `FetchBlob` server-streaming RPC.
    ///
    /// Returns an async iterator that yields one ``bytes`` per
    /// [`FetchBlobChunk::Data`] frame received from the server. The
    /// initial ``Start`` and terminal ``End`` envelope frames are
    /// consumed internally and not surfaced to Python — callers see
    /// only the body bytes, ordered as they arrive.
    ///
    /// Args:
    ///     request: dict with required ``blob_id`` (str) and optional
    ///         ``offset`` (int) and ``chunk_size`` (int — per-frame
    ///         hint).
    ///
    /// Raises:
    ///     ControlPlaneError: If the initial RPC fails or a frame
    ///         arrives that cannot be decoded.
    #[gen_stub(override_return_type(
        type_repr = "typing.AsyncIterator[bytes]",
        imports = ("typing",)
    ))]
    fn fetch_blob(&self, request: Bound<'_, PyDict>) -> PyResult<PyFetchBlobStream> {
        let req = FetchBlobRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            blob_id: dict_required_str(&request, "blob_id")?,
            offset: dict_optional_u64(&request, "offset")?,
            chunk_size: dict_optional_u32(&request, "chunk_size")?,
        };
        Ok(PyFetchBlobStream::new(self.inner.clone(), req))
    }

    fn __repr__(&self) -> String {
        "ModelClient(...)".to_owned()
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a `ClientTlsConfig` from PEM file paths. When the optional
/// client identity is supplied, both `client_cert` and `client_key` are
/// required (enforced by the caller).
fn build_client_tls(
    ca_cert: &str,
    client_cert: Option<&str>,
    client_key: Option<&str>,
) -> PyResult<tonic::transport::ClientTlsConfig> {
    use tonic::transport::{Certificate, ClientTlsConfig, Identity};

    let ca_bytes = std::fs::read(ca_cert).map_err(|e| {
        cp_err(blazen_controlplane::ControlPlaneError::Tls(format!(
            "failed to read CA cert {ca_cert}: {e}",
        )))
    })?;
    let mut tls = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(ca_bytes));

    if let (Some(cert_path), Some(key_path)) = (client_cert, client_key) {
        let cert_bytes = std::fs::read(cert_path).map_err(|e| {
            cp_err(blazen_controlplane::ControlPlaneError::Tls(format!(
                "failed to read client cert {cert_path}: {e}",
            )))
        })?;
        let key_bytes = std::fs::read(key_path).map_err(|e| {
            cp_err(blazen_controlplane::ControlPlaneError::Tls(format!(
                "failed to read client key {key_path}: {e}",
            )))
        })?;
        tls = tls.identity(Identity::from_pem(cert_bytes, key_bytes));
    }

    Ok(tls)
}

/// Build a Python dict representation of a [`StatusResponse`].
fn status_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &StatusResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    let models = PyList::empty(py);
    for m in &resp.models {
        models.append(model_status_to_pydict(py, m)?)?;
    }
    dict.set_item("models", models)?;
    Ok(dict)
}

fn model_status_to_pydict<'py>(
    py: Python<'py>,
    m: &ModelStatusWire,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", m.id.clone())?;
    dict.set_item("loaded", m.loaded)?;
    dict.set_item("memory_estimate_bytes", m.memory_estimate_bytes)?;
    dict.set_item("pool", pool_to_str(m.pool))?;
    let adapters = PyList::empty(py);
    for a in &m.adapters {
        let ad = PyDict::new(py);
        ad.set_item("adapter_id", a.adapter_id.clone())?;
        ad.set_item("scale", a.scale)?;
        ad.set_item("source_dir", a.source_dir.clone())?;
        ad.set_item("memory_bytes", a.memory_bytes)?;
        adapters.append(ad)?;
    }
    dict.set_item("adapters", adapters)?;
    Ok(dict)
}

fn pool_to_str(pool: PoolWire) -> String {
    match pool {
        PoolWire::Cpu => "cpu".to_owned(),
        PoolWire::Gpu(idx) => format!("gpu:{idx}"),
        PoolWire::Remote => "remote".to_owned(),
    }
}

/// Extract a required string field from a Python dict, raising
/// `KeyError` when missing and `TypeError` when present-but-wrong-type.
fn dict_required_str(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    match dict.get_item(key)? {
        Some(v) => v.extract::<String>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be a string"))
        }),
        None => Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "missing required key '{key}'"
        ))),
    }
}

/// Extract an optional string field; `None` when the key is absent or
/// its value is Python `None`.
fn dict_optional_str(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match dict.get_item(key)? {
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(v.extract::<String>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be a string"))
        })?)),
        None => Ok(None),
    }
}

/// Extract an optional `u64` field; same `None` semantics as
/// [`dict_optional_str`].
fn dict_optional_u64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<u64>> {
    match dict.get_item(key)? {
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(v.extract::<u64>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be an int"))
        })?)),
        None => Ok(None),
    }
}

fn backend_hint_from_str(s: &str) -> PyResult<BackendHintWire> {
    match s {
        "auto" => Ok(BackendHintWire::Auto),
        "mistral_rs" => Ok(BackendHintWire::MistralRs),
        "candle" => Ok(BackendHintWire::Candle),
        "llama_cpp" => Ok(BackendHintWire::LlamaCpp),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid backend_hint '{other}'; expected one of \
             'auto', 'mistral_rs', 'candle', 'llama_cpp'",
        ))),
    }
}

fn backend_hint_to_str(b: BackendHintWire) -> &'static str {
    match b {
        BackendHintWire::Auto => "auto",
        BackendHintWire::MistralRs => "mistral_rs",
        BackendHintWire::Candle => "candle",
        BackendHintWire::LlamaCpp => "llama_cpp",
    }
}

fn load_from_hf_request_from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<LoadFromHfRequest> {
    let backend_hint = match dict.get_item("backend_hint")? {
        Some(v) if v.is_none() => None,
        Some(v) => {
            let s = v.extract::<String>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("'backend_hint' must be a string")
            })?;
            Some(backend_hint_from_str(&s)?)
        }
        None => None,
    };
    let extra_options_json = match dict.get_item("extra_options_json")? {
        Some(v) if v.is_none() => Vec::new(),
        Some(v) => v
            .extract::<String>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("'extra_options_json' must be a string")
            })?
            .into_bytes(),
        None => Vec::new(),
    };
    Ok(LoadFromHfRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: dict_required_str(dict, "model_id")?,
        repo: dict_required_str(dict, "repo")?,
        memory_estimate_bytes: dict_optional_u64(dict, "memory_estimate_bytes")?,
        backend_hint,
        gguf_file: dict_optional_str(dict, "gguf_file")?,
        revision: dict_optional_str(dict, "revision")?,
        hf_token: dict_optional_str(dict, "hf_token")?,
        extra_options_json,
    })
}

fn load_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &LoadResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    Ok(dict)
}

fn unload_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &UnloadResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    Ok(dict)
}

fn load_from_hf_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &LoadFromHfResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    dict.set_item("chosen_backend", backend_hint_to_str(resp.chosen_backend))?;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// Adapters / inference / multimodal helpers
// ---------------------------------------------------------------------------

/// Extract a required `f32` field from a Python dict.
fn dict_required_f32(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f32> {
    match dict.get_item(key)? {
        Some(v) => v.extract::<f32>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be a float"))
        }),
        None => Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "missing required key '{key}'"
        ))),
    }
}

/// Extract an optional `u32` field; `None` when absent or Python `None`.
fn dict_optional_u32(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<u32>> {
    match dict.get_item(key)? {
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(v.extract::<u32>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be an int"))
        })?)),
        None => Ok(None),
    }
}

/// Extract an optional `f32` field; `None` when absent or Python `None`.
fn dict_optional_f32(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f32>> {
    match dict.get_item(key)? {
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(v.extract::<f32>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be a float"))
        })?)),
        None => Ok(None),
    }
}

/// Extract an optional JSON-bearing field as raw UTF-8 bytes. Treats
/// missing / Python-`None` / empty-string as "no JSON supplied", matching
/// the wire protocol's empty-`Vec<u8>` sentinel.
fn dict_optional_json_bytes(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<u8>> {
    match dict.get_item(key)? {
        Some(v) if v.is_none() => Ok(Vec::new()),
        Some(v) => Ok(v
            .extract::<String>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be a string"))
            })?
            .into_bytes()),
        None => Ok(Vec::new()),
    }
}

/// Extract an optional `list[str]` field; absent / `None` yields an
/// empty `Vec`.
fn dict_optional_str_list(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<String>> {
    match dict.get_item(key)? {
        Some(v) if v.is_none() => Ok(Vec::new()),
        Some(v) => v.extract::<Vec<String>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be a list of strings"))
        }),
        None => Ok(Vec::new()),
    }
}

/// Extract an optional `dict[str, str]` field as a `BTreeMap` (the wire
/// type) — absent / `None` yields an empty map.
fn dict_optional_str_map(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<BTreeMap<String, String>> {
    match dict.get_item(key)? {
        Some(v) if v.is_none() => Ok(BTreeMap::new()),
        Some(v) => {
            let map: BTreeMap<String, String> = v.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be a dict[str, str]"))
            })?;
            Ok(map)
        }
        None => Ok(BTreeMap::new()),
    }
}

/// Required `bytes` field — used by `transcribe.audio`.
fn dict_required_bytes(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<u8>> {
    match dict.get_item(key)? {
        Some(v) => v
            .extract::<Vec<u8>>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be bytes"))),
        None => Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "missing required key '{key}'"
        ))),
    }
}

fn adapter_mount_strategy_to_str(strategy: AdapterMountStrategyWire) -> &'static str {
    match strategy {
        AdapterMountStrategyWire::Attached => "attached",
        AdapterMountStrategyWire::Rebuilt => "rebuilt",
        AdapterMountStrategyWire::Merged => "merged",
    }
}

fn adapter_status_to_pydict<'py>(
    py: Python<'py>,
    a: &AdapterStatusWire,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("adapter_id", a.adapter_id.clone())?;
    dict.set_item("scale", a.scale)?;
    dict.set_item("source_dir", a.source_dir.clone())?;
    dict.set_item("memory_bytes", a.memory_bytes)?;
    Ok(dict)
}

fn load_adapter_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &LoadAdapterResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    dict.set_item("adapter_id", resp.adapter_id.clone())?;
    dict.set_item("memory_bytes", resp.memory_bytes)?;
    dict.set_item(
        "mount_strategy",
        adapter_mount_strategy_to_str(resp.mount_strategy),
    )?;
    Ok(dict)
}

fn unload_adapter_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &UnloadAdapterResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    Ok(dict)
}

fn list_adapters_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &ListAdaptersResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    let adapters = PyList::empty(py);
    for a in &resp.adapters {
        adapters.append(adapter_status_to_pydict(py, a)?)?;
    }
    dict.set_item("adapters", adapters)?;
    Ok(dict)
}

/// Parse a single chat message dict into a [`ChatMessageWire`].
fn chat_message_from_pyany(item: &Bound<'_, PyAny>) -> PyResult<ChatMessageWire> {
    let dict = item.cast::<PyDict>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "each entry in 'messages' must be a dict with 'role' and 'text'",
        )
    })?;
    Ok(ChatMessageWire {
        role: dict_required_str(dict, "role")?,
        text: dict_required_str(dict, "text")?,
        content_json: dict_optional_json_bytes(dict, "content_json")?,
    })
}

fn complete_request_from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<CompleteRequest> {
    let messages: Vec<ChatMessageWire> = match dict.get_item("messages")? {
        Some(v) if v.is_none() => Vec::new(),
        Some(v) => {
            let list = v.cast::<PyList>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("'messages' must be a list of dicts")
            })?;
            let mut out = Vec::with_capacity(list.len());
            for item in list.iter() {
                out.push(chat_message_from_pyany(&item)?);
            }
            out
        }
        None => {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                "missing required key 'messages'",
            ));
        }
    };
    Ok(CompleteRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: dict_required_str(dict, "model_id")?,
        messages,
        max_tokens: dict_optional_u32(dict, "max_tokens")?,
        temperature: dict_optional_f32(dict, "temperature")?,
        top_p: dict_optional_f32(dict, "top_p")?,
        stop: dict_optional_str_list(dict, "stop")?,
        response_format_json: dict_optional_json_bytes(dict, "response_format_json")?,
        extra_json: dict_optional_json_bytes(dict, "extra_json")?,
        tags: dict_optional_str_map(dict, "tags")?,
    })
}

fn complete_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &CompleteResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    dict.set_item("text", resp.text.clone())?;
    dict.set_item("prompt_tokens", resp.prompt_tokens)?;
    dict.set_item("completion_tokens", resp.completion_tokens)?;
    dict.set_item("finish_reason", resp.finish_reason.clone())?;
    // Pre-serialised JSON travels as a Python `str` — empty when absent.
    let tool_calls = String::from_utf8(resp.tool_calls_json.clone()).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "server returned non-UTF-8 tool_calls_json: {e}",
        ))
    })?;
    dict.set_item("tool_calls_json", tool_calls)?;
    Ok(dict)
}

fn embed_request_from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<EmbedRequest> {
    let inputs: Vec<String> = match dict.get_item("inputs")? {
        Some(v) => v.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("'inputs' must be a list of strings")
        })?,
        None => {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                "missing required key 'inputs'",
            ));
        }
    };
    Ok(EmbedRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: dict_required_str(dict, "model_id")?,
        inputs,
        dimensions: dict_optional_u32(dict, "dimensions")?,
        extra_json: dict_optional_json_bytes(dict, "extra_json")?,
    })
}

fn embed_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &EmbedResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    let vectors = PyList::empty(py);
    for v in &resp.vectors {
        let row = PyList::empty(py);
        for x in v {
            row.append(*x)?;
        }
        vectors.append(row)?;
    }
    dict.set_item("vectors", vectors)?;
    dict.set_item("prompt_tokens", resp.prompt_tokens)?;
    Ok(dict)
}

fn generate_image_request_from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<GenerateImageRequest> {
    Ok(GenerateImageRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: dict_required_str(dict, "model_id")?,
        prompt: dict_required_str(dict, "prompt")?,
        negative_prompt: dict_optional_str(dict, "negative_prompt")?,
        width: dict_optional_u32(dict, "width")?,
        height: dict_optional_u32(dict, "height")?,
        num_images: dict_optional_u32(dict, "num_images")?,
        seed: dict_optional_u64(dict, "seed")?,
        image_config_json: dict_optional_json_bytes(dict, "image_config_json")?,
    })
}

fn image_blob_to_pydict<'py>(py: Python<'py>, img: &ImageBlobWire) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("mime", img.mime.clone())?;
    dict.set_item("data", PyBytes::new(py, &img.data))?;
    Ok(dict)
}

fn generate_image_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &GenerateImageResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    let images = PyList::empty(py);
    for img in &resp.images {
        images.append(image_blob_to_pydict(py, img)?)?;
    }
    dict.set_item("images", images)?;
    Ok(dict)
}

fn text_to_speech_request_from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<TextToSpeechRequest> {
    Ok(TextToSpeechRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: dict_required_str(dict, "model_id")?,
        text: dict_required_str(dict, "text")?,
        voice: dict_optional_str(dict, "voice")?,
        language: dict_optional_str(dict, "language")?,
        sample_rate_hz: dict_optional_u32(dict, "sample_rate_hz")?,
        audio_config_json: dict_optional_json_bytes(dict, "audio_config_json")?,
    })
}

fn text_to_speech_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &TextToSpeechResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    dict.set_item("mime", resp.mime.clone())?;
    dict.set_item("data", PyBytes::new(py, &resp.data))?;
    dict.set_item("sample_rate_hz", resp.sample_rate_hz)?;
    Ok(dict)
}

fn generate_music_request_from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<GenerateMusicRequest> {
    Ok(GenerateMusicRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: dict_required_str(dict, "model_id")?,
        prompt: dict_required_str(dict, "prompt")?,
        duration_secs: dict_optional_f32(dict, "duration_secs")?,
        seed: dict_optional_u64(dict, "seed")?,
        extra_json: dict_optional_json_bytes(dict, "extra_json")?,
    })
}

fn generate_music_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &GenerateMusicResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    dict.set_item("mime", resp.mime.clone())?;
    dict.set_item("data", PyBytes::new(py, &resp.data))?;
    dict.set_item("sample_rate_hz", resp.sample_rate_hz)?;
    Ok(dict)
}

fn transcribe_request_from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<TranscribeRequest> {
    Ok(TranscribeRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: dict_required_str(dict, "model_id")?,
        audio: dict_required_bytes(dict, "audio")?,
        mime: dict_required_str(dict, "mime")?,
        language: dict_optional_str(dict, "language")?,
        extra_json: dict_optional_json_bytes(dict, "extra_json")?,
    })
}

fn transcribe_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &TranscribeResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    dict.set_item("text", resp.text.clone())?;
    dict.set_item("language", resp.language.clone())?;
    let segments = String::from_utf8(resp.segments_json.clone()).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "server returned non-UTF-8 segments_json: {e}",
        ))
    })?;
    dict.set_item("segments_json", segments)?;
    Ok(dict)
}

// ===========================================================================
// PyStreamCompleteStream — async iterator of StreamCompleteChunk dicts
// ===========================================================================

/// Channel item carried from the forwarder task into Python land.
type ChunkItem = Result<StreamCompleteChunk, ControlPlaneError>;

/// Lazy slot holding the per-stream `mpsc::Receiver` (after init) or
/// nothing (before init). Boxed via `Arc<Mutex<...>>` so `__anext__` can
/// mutate it with only `&self`.
type LazyChunkReceiver = Arc<Mutex<Option<mpsc::Receiver<ChunkItem>>>>;

/// Async iterator over [`StreamCompleteChunk`]s.
///
/// Mirrors the lazy-init pattern used by
/// [`super::client::PyRunEventStream`]: the upstream RPC is opened on
/// the first `__anext__` call, and a tokio task owns a clone of the
/// `ModelClient` so the borrowed stream stays valid for the receiver's
/// lifetime.
#[gen_stub_pyclass]
#[pyclass(name = "StreamCompleteStream")]
pub struct PyStreamCompleteStream {
    rx: LazyChunkReceiver,
    init: Arc<Mutex<Option<StreamCompleteInit>>>,
}

struct StreamCompleteInit {
    client: ModelClient,
    request: CompleteRequest,
}

impl PyStreamCompleteStream {
    fn new(client: ModelClient, request: CompleteRequest) -> Self {
        Self {
            rx: Arc::new(Mutex::new(None)),
            init: Arc::new(Mutex::new(Some(StreamCompleteInit { client, request }))),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStreamCompleteStream {
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
            // Lazy init: open the StreamComplete RPC + spawn forwarder
            // on first __anext__ invocation.
            {
                let mut rx_guard = rx_slot.lock().await;
                if rx_guard.is_none() {
                    let init = init_slot.lock().await.take();
                    if let Some(init) = init {
                        *rx_guard = Some(open_stream_complete(init));
                    } else {
                        return Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                            "stream exhausted",
                        ));
                    }
                }
            }

            // Pull one chunk from the receiver.
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
                Some(Ok(chunk)) => Python::attach(|py| {
                    Ok(stream_complete_chunk_to_pydict(py, &chunk)?
                        .into_any()
                        .unbind())
                }),
                Some(Err(e)) => Err(cp_err(e)),
                None => Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                    "stream exhausted",
                )),
            }
        })
    }
}

/// Open the upstream `StreamComplete` RPC and spawn a forwarder task
/// that pumps decoded chunks into an `mpsc::Receiver`. The `ModelClient`
/// clone is owned by the spawned task, keeping the gRPC channel alive
/// for the stream's lifetime.
fn open_stream_complete(init: StreamCompleteInit) -> mpsc::Receiver<ChunkItem> {
    let (tx, rx) = mpsc::channel::<ChunkItem>(32);
    let StreamCompleteInit { client, request } = init;
    tokio::spawn(async move {
        let mut stream = match client.stream_complete(request).await {
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

/// Build a Python dict representation of an [`UploadBlobResponse`].
fn upload_blob_response_to_pydict<'py>(
    py: Python<'py>,
    resp: &UploadBlobResponse,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("envelope_version", resp.envelope_version)?;
    dict.set_item("blob_id", resp.blob_id.clone())?;
    dict.set_item("bytes_received", resp.bytes_received)?;
    Ok(dict)
}

/// Forwarder body for [`PyModelClient::upload_blob`]: walks a Python
/// async iterator and pushes one [`UploadBlobChunk::Data`] per yielded
/// payload into `tx`, then a single [`UploadBlobChunk::End`] before
/// dropping the sender.
///
/// Mirrors [`crate::content::store::pystream_into_byte_stream`] in
/// structure: it uses `block_in_place` + `Python::attach` to dispatch
/// `__anext__` from the tokio worker, then drives the resulting
/// coroutine under the captured `TaskLocals` via
/// `pyo3_async_runtimes::tokio::scope`. Errors (Python exceptions,
/// non-bytes payloads, etc.) are dropped to the channel via the
/// sender being closed — the parent task surfaces the resulting
/// `Transport` error from the upstream RPC.
async fn forward_python_chunks(
    iter_obj: Py<PyAny>,
    locals: pyo3_async_runtimes::TaskLocals,
    tx: mpsc::Sender<UploadBlobChunk>,
) {
    loop {
        let next_fut = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let bound = iter_obj.bind(py);
                let coro = bound.call_method0("__anext__")?;
                pyo3_async_runtimes::into_future_with_locals(&locals, coro)
            })
        });

        // `__anext__` itself failing closes the channel and lets the
        // server surface the resulting EOF as a wire error.
        let Ok(next_fut) = next_fut else { return };

        let py_result = pyo3_async_runtimes::tokio::scope(locals.clone(), next_fut).await;
        let py_obj = match py_result {
            Ok(obj) => obj,
            Err(e) => {
                let stop = Python::attach(|py| {
                    e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py)
                });
                if stop {
                    // Clean EOF — send the terminal End frame and
                    // let the channel close.
                    let _ = tx
                        .send(UploadBlobChunk::End {
                            envelope_version: MODEL_ENVELOPE_VERSION,
                        })
                        .await;
                }
                return;
            }
        };

        let bytes = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<Vec<u8>> {
                let bound = py_obj.bind(py);
                if let Ok(b) = bound.cast::<PyBytes>() {
                    return Ok(b.as_bytes().to_vec());
                }
                bound.extract::<Vec<u8>>().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "upload_blob iterator must yield bytes / bytearray / buffer-protocol \
                         objects",
                    )
                })
            })
        });

        let Ok(bytes) = bytes else { return };

        if tx
            .send(UploadBlobChunk::Data {
                envelope_version: MODEL_ENVELOPE_VERSION,
                bytes,
            })
            .await
            .is_err()
        {
            // Upstream stopped consuming — abandon the iterator.
            return;
        }
    }
}

/// Build a Python dict representation of a [`StreamCompleteChunk`].
fn stream_complete_chunk_to_pydict<'py>(
    py: Python<'py>,
    chunk: &StreamCompleteChunk,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match chunk {
        StreamCompleteChunk::Delta {
            envelope_version,
            text,
        } => {
            dict.set_item("kind", "delta")?;
            dict.set_item("envelope_version", *envelope_version)?;
            dict.set_item("text", text.clone())?;
        }
        StreamCompleteChunk::Done {
            envelope_version,
            prompt_tokens,
            completion_tokens,
            finish_reason,
        } => {
            dict.set_item("kind", "done")?;
            dict.set_item("envelope_version", *envelope_version)?;
            dict.set_item("prompt_tokens", *prompt_tokens)?;
            dict.set_item("completion_tokens", *completion_tokens)?;
            dict.set_item("finish_reason", finish_reason.clone())?;
        }
    }
    Ok(dict)
}

// ===========================================================================
// PyFetchBlobStream — async iterator of `bytes`, one per Data frame
// ===========================================================================

/// Channel item carried from the forwarder task into Python land for
/// [`PyFetchBlobStream`].
type BlobChunkItem = Result<FetchBlobChunk, ControlPlaneError>;

/// Lazy slot holding the per-stream `mpsc::Receiver` (after init) or
/// nothing (before init). Same shape as `LazyChunkReceiver` — duplicated
/// because the item type differs.
type LazyBlobReceiver = Arc<Mutex<Option<mpsc::Receiver<BlobChunkItem>>>>;

/// Async iterator over the [`FetchBlobChunk::Data`] payloads from a
/// `FetchBlob` server-stream.
///
/// Mirrors [`PyStreamCompleteStream`]: the upstream RPC opens on the
/// first `__anext__` call and a spawned tokio task owns the
/// [`ModelClient`] clone for the stream's lifetime. `Start` / `End`
/// frames are consumed internally and never surface to Python — the
/// iterator yields raw chunk bytes only.
#[gen_stub_pyclass]
#[pyclass(name = "FetchBlobStream")]
pub struct PyFetchBlobStream {
    rx: LazyBlobReceiver,
    init: Arc<Mutex<Option<FetchBlobInit>>>,
}

struct FetchBlobInit {
    client: ModelClient,
    request: FetchBlobRequest,
}

impl PyFetchBlobStream {
    fn new(client: ModelClient, request: FetchBlobRequest) -> Self {
        Self {
            rx: Arc::new(Mutex::new(None)),
            init: Arc::new(Mutex::new(Some(FetchBlobInit { client, request }))),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFetchBlobStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, bytes]",
        imports = ("typing",)
    ))]
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rx_slot = Arc::clone(&self.rx);
        let init_slot = Arc::clone(&self.init);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Lazy init: open the FetchBlob RPC + spawn forwarder on
            // the first __anext__ invocation.
            {
                let mut rx_guard = rx_slot.lock().await;
                if rx_guard.is_none() {
                    let init = init_slot.lock().await.take();
                    if let Some(init) = init {
                        *rx_guard = Some(open_fetch_blob(init));
                    } else {
                        return Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                            "stream exhausted",
                        ));
                    }
                }
            }

            // Pull frames until we get a `Data` payload (or EOF /
            // error). `Start` / `End` envelopes are control-plane
            // metadata and not surfaced to Python.
            loop {
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
                    Some(Ok(FetchBlobChunk::Data { bytes, .. })) => {
                        return Python::attach(|py| {
                            Ok(PyBytes::new(py, &bytes).into_any().unbind())
                        });
                    }
                    // Skip `Start` and `End` frames — keep looping.
                    Some(Ok(_)) => {}
                    Some(Err(e)) => return Err(cp_err(e)),
                    None => {
                        return Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                            "stream exhausted",
                        ));
                    }
                }
            }
        })
    }
}

/// Open the upstream `FetchBlob` RPC and spawn a forwarder task that
/// pumps decoded frames into an `mpsc::Receiver`. The `ModelClient`
/// clone is owned by the spawned task, keeping the gRPC channel alive
/// for the stream's lifetime. Mirrors [`open_stream_complete`].
fn open_fetch_blob(init: FetchBlobInit) -> mpsc::Receiver<BlobChunkItem> {
    let (tx, rx) = mpsc::channel::<BlobChunkItem>(32);
    let FetchBlobInit { client, request } = init;
    tokio::spawn(async move {
        let mut stream = match client.fetch_blob(request).await {
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
