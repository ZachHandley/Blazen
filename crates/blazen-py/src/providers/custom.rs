//! Python wrapper for the universal [`blazen_llm::CustomProvider`] trait.
//!
//! Lets a Python user write a normal class with async capability methods
//! and have it surface as a first-class Blazen provider that implements
//! [`CompletionModel`], [`AudioGeneration`], [`VoiceCloning`],
//! [`ImageGeneration`], [`VideoGeneration`], [`Transcription`],
//! [`ThreeDGeneration`], and [`BackgroundRemoval`]. Every method missing
//! on the Python side falls through to ``BlazenError::Unsupported`` —
//! the same default the Rust trait offers.
//!
//! ## Two construction paths
//!
//! 1. **Subclass**: `class MyProv(CustomProvider): ...` overriding any
//!    subset of the 16 typed methods. Instantiation auto-installs a
//!    [`PyCustomProviderAdapter`] which wraps the Python instance and
//!    routes each typed Rust method to the matching Python `async def`.
//! 2. **Built-in protocol**: `CustomProvider(provider_id,
//!    protocol=ApiProtocol.openai(cfg))` instantiates the plain wrapper
//!    backed by Blazen's `openai_compat` factory — no host dispatch is
//!    installed. Use this when the framework should speak the OpenAI
//!    Chat Completions wire format itself.
//!
//! For ergonomic local-server setup, three classmethod factories
//! (`ollama`, `lm_studio`, `openai_compat`) wrap the free functions in
//! [`blazen_llm`].
//!
//! ## Bridging Python async -> Rust async
//!
//! Each method on the trait impl:
//!
//! 1. Acquires the GIL via [`Python::attach`].
//! 2. Checks `hasattr(instance, method_name)`; if absent, returns
//!    [`BlazenError::Unsupported`] (the trait's documented contract).
//! 3. Serializes the typed request via [`pythonize::pythonize`] (request
//!    types derive `Serialize`).
//! 4. Calls the named method on the held `Py<PyAny>` to get a coroutine.
//! 5. Captures the active asyncio task locals.
//! 6. Converts the coroutine into a Rust future via
//!    [`pyo3_async_runtimes::into_future_with_locals`].
//! 7. Drives the future under
//!    [`pyo3_async_runtimes::tokio::scope`] (GIL released).
//! 8. Re-acquires the GIL and deserializes the result back via
//!    [`pythonize::depythonize`] into the typed Rust response.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::compute::requests::{
    BackgroundRemovalRequest, ImageRequest, MusicRequest, SpeechRequest, ThreeDRequest,
    TranscriptionRequest, UpscaleRequest, VideoRequest, VoiceCloneRequest,
};
use blazen_llm::compute::results::{
    AudioResult, ImageResult, ThreeDResult, TranscriptionResult, VideoResult, VoiceHandle,
};
use blazen_llm::compute::traits::ComputeProvider;
use blazen_llm::error::BlazenError;
use blazen_llm::traits::CompletionModel;
use blazen_llm::types::{CompletionRequest, CompletionResponse, EmbeddingResponse, StreamChunk};
use blazen_llm::{ApiProtocol, CustomProvider, CustomProviderHandle};

use crate::compute::request_types::{
    PyBackgroundRemovalRequest, PyImageRequest, PyMusicRequest, PySpeechRequest, PyThreeDRequest,
    PyTranscriptionRequest, PyUpscaleRequest, PyVideoRequest, PyVoiceCloneRequest,
};
use crate::compute::result_types::{
    PyAudioResult, PyImageResult, PyThreeDResult, PyTranscriptionResult, PyVideoResult,
    PyVoiceHandle,
};
use crate::error::blazen_error_to_pyerr;
use crate::providers::api_protocol::{ApiProtocolKind, PyApiProtocol};
use crate::providers::base::PyBaseProvider;
use crate::providers::completion_model::PyCompletionModel;
use crate::providers::config::PyRetryConfig;
use crate::providers::defaults::PyCompletionProviderDefaults;
use crate::providers::openai_compat::PyOpenAiCompatConfig;
use crate::types::PyHttpClientHandle;

// ---------------------------------------------------------------------------
// PyCustomProviderAdapter
// ---------------------------------------------------------------------------

/// `CustomProvider` impl that delegates each typed trait method to the
/// matching Python `async def` on the held instance.
///
/// `instance` is `Arc`-wrapped so the adapter is `Send + Sync + 'static`
/// without requiring `Py<PyAny>` itself to be `Sync` — the inner
/// `Py<PyAny>` is touched only inside [`Python::attach`].
pub(crate) struct PyCustomProviderAdapter {
    instance: Arc<Py<PyAny>>,
    provider_id: String,
    model_id: String,
}

impl PyCustomProviderAdapter {
    /// Build an adapter wrapping the Python instance.
    ///
    /// `provider_id` is read from the instance under the GIL (via
    /// `provider_id` getter / attribute / method) if present, falling back to
    /// the supplied `fallback_provider_id`. `model_id` is resolved similarly,
    /// defaulting to the resolved `provider_id`.
    pub(crate) fn new(instance: Py<PyAny>, fallback_provider_id: String) -> Self {
        let (provider_id, model_id) = Python::attach(|py| {
            let bound = instance.bind(py);
            let provider_id =
                read_string_attr(bound, "provider_id").unwrap_or(fallback_provider_id);
            let model_id =
                read_string_attr(bound, "model_id").unwrap_or_else(|| provider_id.clone());
            (provider_id, model_id)
        });
        Self {
            instance: Arc::new(instance),
            provider_id,
            model_id,
        }
    }
}

/// Read a `String` attribute or zero-arg method from a Python object.
///
/// Tries `getattr(name)`. If the attribute is callable, calls it with no
/// args. The result is then extracted as `String`. Returns `None` on any
/// failure (missing, not stringy, raised, etc.).
fn read_string_attr(obj: &Bound<'_, PyAny>, name: &str) -> Option<String> {
    let attr = obj.getattr(name).ok()?;
    let value = if attr.is_callable() {
        attr.call0().ok()?
    } else {
        attr
    };
    value.extract::<String>().ok()
}

/// Shared "call this Python coroutine and deserialize the result" routine.
///
/// `serialize_request` builds the Python argument(s) under the GIL.
/// `deserialize_result` turns the awaited Python return value into the
/// caller's typed `T` (also under the GIL).
async fn dispatch_call<T, S, D>(
    instance: &Py<PyAny>,
    method: &str,
    serialize_request: S,
    deserialize_result: D,
) -> Result<T, BlazenError>
where
    S: for<'py> FnOnce(Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> + Send,
    D: for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<T> + Send,
    T: Send,
{
    // Phase 1: under the GIL, build the args, call the method to get a
    // coroutine, capture asyncio task locals, and convert it into a Rust
    // future.
    let setup = tokio::task::block_in_place(|| {
        Python::attach(|py| -> PyResult<_> {
            let bound = instance.bind(py);
            if !bound.hasattr(method)? {
                return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                    "missing method `{method}`"
                )));
            }
            let args = serialize_request(py)?;
            // Convert Vec<Bound<PyAny>> into a Python tuple for call.
            let py_args = pyo3::types::PyTuple::new(py, args.iter())?;
            let coro = bound.call_method1(method, py_args).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "host method `{method}` raised before yielding a coroutine: {e}"
                ))
            })?;
            let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
            let fut = pyo3_async_runtimes::into_future_with_locals(&locals, coro).map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "host method `{method}` did not return an awaitable coroutine (custom \
                     providers require `async def` methods): {e}"
                ))
            })?;
            Ok((fut, locals))
        })
    });

    // Distinguish "method missing -> Unsupported" from "everything else ->
    // provider error" so the rest of Blazen can fall through to the
    // trait's documented default semantics.
    let (fut, locals) = match setup {
        Ok(v) => v,
        Err(e) => {
            let is_missing =
                Python::attach(|py| e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py));
            if is_missing {
                return Err(BlazenError::unsupported(format!(
                    "CustomProvider::{method} not implemented on Python class"
                )));
            }
            return Err(BlazenError::provider(
                "custom",
                format!("dispatch setup failed: {e}"),
            ));
        }
    };

    // Phase 2: drive the Python coroutine to completion outside the GIL.
    let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
        .await
        .map_err(|e: PyErr| {
            BlazenError::provider("custom", format!("host method `{method}` raised: {e}"))
        })?;

    // Phase 3: re-acquire the GIL and deserialize the result.
    tokio::task::block_in_place(|| {
        Python::attach(|py| -> PyResult<T> {
            let bound = py_result.bind(py);
            deserialize_result(py, bound)
        })
    })
    .map_err(|e: PyErr| {
        BlazenError::provider(
            "custom",
            format!("failed to decode result of `{method}`: {e}"),
        )
    })
}

/// Pythonize a `Serialize` value into a `Bound<'py, PyAny>`.
fn pythonize_value<'py, T: serde::Serialize>(
    py: Python<'py>,
    value: &T,
) -> PyResult<Bound<'py, PyAny>> {
    pythonize::pythonize(py, value).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("failed to pythonize request: {e}"))
    })
}

/// Depythonize a `Bound<'_, PyAny>` (typically a dict) into a
/// `Deserialize` value. `None` deserializes to `Value::Null` first so unit
/// returns are tolerated.
fn depythonize_value<T>(bound: &Bound<'_, PyAny>) -> PyResult<T>
where
    T: for<'de> serde::Deserialize<'de>,
{
    let value: serde_json::Value = if bound.is_none() {
        serde_json::Value::Null
    } else {
        pythonize::depythonize(bound).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("failed to depythonize result: {e}"))
        })?
    };
    serde_json::from_value(value).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "result did not match expected schema: {e}"
        ))
    })
}

#[async_trait]
impl CustomProvider for PyCustomProviderAdapter {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        dispatch_call(
            &self.instance,
            "complete",
            move |py| Ok(vec![pythonize_value(py, &request)?]),
            |_py, bound| depythonize_value::<CompletionResponse>(bound),
        )
        .await
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        // Mirrors `PySubclassCompletionModel::stream`: streaming through
        // the Python adapter is not yet wired. Subclasses that need
        // streaming should call `stream()` directly on the Python side.
        Err(BlazenError::unsupported(
            "CustomProvider::stream() over a Python instance is not yet supported; call \
             stream() directly on the Python object",
        ))
    }

    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, BlazenError> {
        dispatch_call(
            &self.instance,
            "embed",
            move |py| Ok(vec![pythonize_value(py, &texts)?]),
            |_py, bound| depythonize_value::<EmbeddingResponse>(bound),
        )
        .await
    }

    async fn text_to_speech(&self, req: SpeechRequest) -> Result<AudioResult, BlazenError> {
        dispatch_call(
            &self.instance,
            "text_to_speech",
            move |py| Ok(vec![pythonize_value(py, &req)?]),
            |_py, bound| depythonize_value::<AudioResult>(bound),
        )
        .await
    }

    async fn generate_music(&self, req: MusicRequest) -> Result<AudioResult, BlazenError> {
        dispatch_call(
            &self.instance,
            "generate_music",
            move |py| Ok(vec![pythonize_value(py, &req)?]),
            |_py, bound| depythonize_value::<AudioResult>(bound),
        )
        .await
    }

    async fn generate_sfx(&self, req: MusicRequest) -> Result<AudioResult, BlazenError> {
        dispatch_call(
            &self.instance,
            "generate_sfx",
            move |py| Ok(vec![pythonize_value(py, &req)?]),
            |_py, bound| depythonize_value::<AudioResult>(bound),
        )
        .await
    }

    async fn clone_voice(&self, req: VoiceCloneRequest) -> Result<VoiceHandle, BlazenError> {
        dispatch_call(
            &self.instance,
            "clone_voice",
            move |py| Ok(vec![pythonize_value(py, &req)?]),
            |_py, bound| depythonize_value::<VoiceHandle>(bound),
        )
        .await
    }

    async fn list_voices(&self) -> Result<Vec<VoiceHandle>, BlazenError> {
        dispatch_call(
            &self.instance,
            "list_voices",
            |_py| Ok(Vec::new()),
            |_py, bound| depythonize_value::<Vec<VoiceHandle>>(bound),
        )
        .await
    }

    async fn delete_voice(&self, voice: VoiceHandle) -> Result<(), BlazenError> {
        dispatch_call(
            &self.instance,
            "delete_voice",
            move |py| Ok(vec![pythonize_value(py, &voice)?]),
            |_py, _bound| Ok(()),
        )
        .await
    }

    async fn generate_image(&self, req: ImageRequest) -> Result<ImageResult, BlazenError> {
        dispatch_call(
            &self.instance,
            "generate_image",
            move |py| Ok(vec![pythonize_value(py, &req)?]),
            |_py, bound| depythonize_value::<ImageResult>(bound),
        )
        .await
    }

    async fn upscale_image(&self, req: UpscaleRequest) -> Result<ImageResult, BlazenError> {
        dispatch_call(
            &self.instance,
            "upscale_image",
            move |py| Ok(vec![pythonize_value(py, &req)?]),
            |_py, bound| depythonize_value::<ImageResult>(bound),
        )
        .await
    }

    async fn text_to_video(&self, req: VideoRequest) -> Result<VideoResult, BlazenError> {
        dispatch_call(
            &self.instance,
            "text_to_video",
            move |py| Ok(vec![pythonize_value(py, &req)?]),
            |_py, bound| depythonize_value::<VideoResult>(bound),
        )
        .await
    }

    async fn image_to_video(&self, req: VideoRequest) -> Result<VideoResult, BlazenError> {
        dispatch_call(
            &self.instance,
            "image_to_video",
            move |py| Ok(vec![pythonize_value(py, &req)?]),
            |_py, bound| depythonize_value::<VideoResult>(bound),
        )
        .await
    }

    async fn transcribe(
        &self,
        req: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        dispatch_call(
            &self.instance,
            "transcribe",
            move |py| Ok(vec![pythonize_value(py, &req)?]),
            |_py, bound| depythonize_value::<TranscriptionResult>(bound),
        )
        .await
    }

    async fn generate_3d(&self, req: ThreeDRequest) -> Result<ThreeDResult, BlazenError> {
        dispatch_call(
            &self.instance,
            "generate_3d",
            move |py| Ok(vec![pythonize_value(py, &req)?]),
            |_py, bound| depythonize_value::<ThreeDResult>(bound),
        )
        .await
    }

    async fn remove_background(
        &self,
        req: BackgroundRemovalRequest,
    ) -> Result<ImageResult, BlazenError> {
        dispatch_call(
            &self.instance,
            "remove_background",
            move |py| Ok(vec![pythonize_value(py, &req)?]),
            |_py, bound| depythonize_value::<ImageResult>(bound),
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Helpers shared by `__new__` / `__init__` / classmethod factories
// ---------------------------------------------------------------------------

/// Build a `(PyCustomProvider, PyBaseProvider)` tuple ready to hand back
/// to Python from `__new__`. Centralizes the `PyCompletionModel`-wrapping
/// and parent-defaults wiring.
fn build_parent_pair(
    handle: CustomProviderHandle,
    protocol: PyApiProtocol,
    defaults: Option<PyCompletionProviderDefaults>,
    provider_id: String,
) -> PyResult<(PyCustomProvider, PyBaseProvider)> {
    let arc = Arc::new(handle);
    // `CustomProviderHandle` itself implements `CompletionModel`, so we
    // can hand it straight to `PyCompletionModel` as the inner trait
    // object — that lets `PyBaseProvider`/`PyCompletionModel.complete`
    // reach the same defaults-applying path as the Rust handle.
    let completion: Arc<dyn CompletionModel> = arc.clone();
    let py_completion = PyCompletionModel {
        inner: Some(completion),
        local_model: None,
        config: None,
    };
    let py_inner: Py<PyCompletionModel> = Python::attach(|py| Py::new(py, py_completion))?;
    let base = PyBaseProvider {
        inner: py_inner,
        defaults: defaults.unwrap_or_default(),
    };
    let me = PyCustomProvider {
        handle: arc,
        provider_id_str: provider_id,
        protocol,
    };
    Ok((me, base))
}

/// Stub `CustomProvider` used to satisfy `__new__` before a real adapter
/// has been installed. Every method falls through to the trait's
/// `Unsupported` default, so any accidental use surfaces a clear error
/// rather than a panic.
struct UninitializedAdapter {
    provider_id: String,
}

#[async_trait]
impl CustomProvider for UninitializedAdapter {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }
}

/// Did `cls` evaluate to a real subclass of `PyCustomProvider` (rather
/// than `PyCustomProvider` itself)?
fn is_real_subclass(cls: &Bound<'_, PyType>) -> bool {
    let py = cls.py();
    let base = py.get_type::<PyCustomProvider>();
    !cls.is(&base)
}

// ---------------------------------------------------------------------------
// PyCustomProvider
// ---------------------------------------------------------------------------

/// A user-defined Blazen provider.
///
/// Two construction modes:
///
/// 1. **Subclass**: define `class MyProv(CustomProvider): ...` and
///    override any subset of the 16 typed `async def` methods
///    (``complete``, ``stream``, ``embed``, ``text_to_speech``,
///    ``generate_music``, ``generate_sfx``, ``clone_voice``,
///    ``list_voices``, ``delete_voice``, ``generate_image``,
///    ``upscale_image``, ``text_to_video``, ``image_to_video``,
///    ``transcribe``, ``generate_3d``, ``remove_background``). Missing
///    methods raise ``UnsupportedError`` when called.
/// 2. **Built-in OpenAI-protocol**: ``CustomProvider("my-server",
///    protocol=ApiProtocol.openai(cfg))`` builds an OpenAI-compat
///    backed provider — no host dispatch installed; the framework
///    speaks the wire format itself.
///
/// Convenience classmethod factories (``ollama``, ``lm_studio``,
/// ``openai_compat``) wrap the free functions in
/// :rust:func:`blazen_llm::ollama`.
///
/// Example (subclass):
///
/// ```text
/// class ElevenLabsProvider(CustomProvider):
///     def __init__(self, api_key):
///         super().__init__(provider_id="elevenlabs")
///         self._client = AsyncElevenLabs(api_key=api_key)
///
///     async def text_to_speech(self, request):
///         audio = b"".join([
///             chunk async for chunk in self._client.text_to_speech.convert(
///                 voice_id=request["voice"],
///                 text=request["text"],
///                 model_id="eleven_multilingual_v2",
///             )
///         ])
///         return {
///             "audio": [{"media": {"base64": base64.b64encode(audio).decode(),
///                                  "media_type": "mpeg"}}],
///             "timing": {"total_ms": 0, "queue_ms": None, "execution_ms": None},
///             "metadata": {},
///         }
///
/// provider = ElevenLabsProvider(api_key="...")
/// result = await provider.text_to_speech(SpeechRequest(text="hi", voice="rachel"))
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "CustomProvider", extends = PyBaseProvider, subclass, from_py_object)]
pub struct PyCustomProvider {
    /// The wrapped handle. Holds an `Arc<dyn CustomProvider>` plus
    /// per-instance defaults and applies them before dispatching.
    pub(crate) handle: Arc<CustomProviderHandle>,
    /// Cached provider id for cheap `provider_id` access.
    pub(crate) provider_id_str: String,
    /// Cached protocol selector for `protocol` introspection and for
    /// rebuilds in [`PyCustomProvider::with_retry_config`].
    pub(crate) protocol: PyApiProtocol,
}

impl Clone for PyCustomProvider {
    fn clone(&self) -> Self {
        Self {
            handle: Arc::clone(&self.handle),
            provider_id_str: self.provider_id_str.clone(),
            protocol: self.protocol.clone(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCustomProvider {
    /// Construct a [`CustomProvider`].
    ///
    /// When called as ``CustomProvider(provider_id, protocol=...)`` (no
    /// subclassing), ``protocol`` is required and must be
    /// ``ApiProtocol.openai(cfg)`` — the framework speaks the OpenAI
    /// Chat Completions wire format directly.
    ///
    /// When called as a subclass constructor (``MyProv(...)``),
    /// ``protocol`` is unused and the subclass instance acts as the
    /// host. The 16 typed methods on the subclass are dispatched to
    /// directly.
    ///
    /// Args:
    ///     provider_id: Short identifier used for logging.
    ///     protocol: Optional [`ApiProtocol`] — only used when *not*
    ///         subclassing. Required for the non-subclass path.
    #[new]
    #[pyo3(signature = (provider_id, protocol=None))]
    fn new(
        provider_id: String,
        protocol: Option<PyApiProtocol>,
    ) -> PyResult<(PyCustomProvider, PyBaseProvider)> {
        // The real wiring happens in `__init__` (below) once we have
        // access to `slf` / `cls`. `__new__` only needs to seed enough
        // state to keep PyO3 happy — the parent slot must hold a valid
        // `PyCompletionModel`, so we build one wrapping an
        // "uninitialized" `CustomProviderHandle` whose every method
        // falls back to `Unsupported`.
        let stub: Arc<dyn CustomProvider> = Arc::new(UninitializedAdapter {
            provider_id: provider_id.clone(),
        });
        let handle = CustomProviderHandle::new(stub);
        let proto = protocol.unwrap_or(PyApiProtocol {
            kind: ApiProtocolKind::Custom,
        });
        build_parent_pair(handle, proto, None, provider_id)
    }

    /// Initialize the provider.
    ///
    /// Detects subclassing by comparing ``type(self)`` against
    /// ``CustomProvider``. When the caller is a subclass, installs a
    /// [`PyCustomProviderAdapter`] that wraps ``self`` and routes every
    /// trait method through the Python instance. Otherwise validates
    /// the supplied ``protocol`` and builds a backend via
    /// :rust:func:`blazen_llm::openai_compat`.
    #[pyo3(signature = (provider_id, protocol=None))]
    fn __init__(
        slf: Bound<'_, Self>,
        provider_id: String,
        protocol: Option<PyApiProtocol>,
    ) -> PyResult<()> {
        let py = slf.py();
        let cls = slf.get_type();
        let (handle, resolved_protocol) = if is_real_subclass(&cls) {
            // Subclass case: wrap `slf` in `PyCustomProviderAdapter`.
            let instance: Py<PyAny> = slf.clone().into_any().unbind();
            let adapter = PyCustomProviderAdapter::new(instance, provider_id.clone());
            let arc: Arc<dyn CustomProvider> = Arc::new(adapter);
            let handle = CustomProviderHandle::new(arc);
            let proto = protocol.unwrap_or(PyApiProtocol {
                kind: ApiProtocolKind::Custom,
            });
            (handle, proto)
        } else {
            // Direct-instantiation case: protocol is required.
            let proto = protocol.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "CustomProvider(...) without subclassing requires a `protocol` argument \
                     (e.g. ApiProtocol.openai(cfg)); subclass CustomProvider to install a \
                     typed Python implementation instead",
                )
            })?;
            let handle = match &proto.kind {
                ApiProtocolKind::OpenAi(cfg) => {
                    blazen_llm::openai_compat(provider_id.clone(), cfg.inner.clone())
                }
                ApiProtocolKind::Custom => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "ApiProtocol.custom() requires subclassing CustomProvider — \
                         passing it to the constructor directly is no longer supported",
                    ));
                }
            };
            (handle, proto)
        };

        // Replace the placeholder state installed by `__new__`.
        let arc = Arc::new(handle);
        let completion: Arc<dyn CompletionModel> = arc.clone();
        let py_completion = PyCompletionModel {
            inner: Some(completion),
            local_model: None,
            config: None,
        };
        let py_inner: Py<PyCompletionModel> = Py::new(py, py_completion)?;
        {
            let mut me = slf.borrow_mut();
            me.handle = arc;
            me.provider_id_str = provider_id;
            me.protocol = resolved_protocol;
        }
        // Update the parent `PyBaseProvider` slot so completion calls go
        // through the freshly-built handle rather than the stub.
        let parent = slf.into_super();
        parent.borrow_mut().inner = py_inner;
        Ok(())
    }

    /// Build a [`CustomProvider`] for a local Ollama server.
    ///
    /// Equivalent to constructing an OpenAI-compatible provider with
    /// ``base_url = f"http://{host}:{port}/v1"`` and no API key.
    ///
    /// Args:
    ///     model: Model identifier loaded on the server (e.g. ``"llama3.1"``).
    ///     host: Hostname or IP. Defaults to ``"localhost"``.
    ///     port: TCP port. Defaults to Ollama's standard ``11434``.
    #[classmethod]
    #[pyo3(signature = (model, host=None, port=None))]
    #[gen_stub(override_return_type(type_repr = "CustomProvider"))]
    fn ollama(
        cls: &Bound<'_, PyType>,
        model: String,
        host: Option<String>,
        port: Option<u16>,
    ) -> PyResult<Py<PyCustomProvider>> {
        let py = cls.py();
        let host = host.unwrap_or_else(|| "localhost".to_owned());
        let port = port.unwrap_or(11434);
        let handle = blazen_llm::ollama(&host, port, model);
        let cfg = match handle.protocol().clone() {
            ApiProtocol::OpenAi(c) => c,
            ApiProtocol::Custom => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "internal: blazen_llm::ollama did not yield OpenAi protocol",
                ));
            }
        };
        let proto = PyApiProtocol {
            kind: ApiProtocolKind::OpenAi(PyOpenAiCompatConfig { inner: cfg }),
        };
        let pair = build_parent_pair(handle, proto, None, "ollama".to_owned())?;
        Py::new(py, pair)
    }

    /// Build a [`CustomProvider`] for an LM Studio server.
    ///
    /// Equivalent to constructing an OpenAI-compatible provider with
    /// ``base_url = f"http://{host}:{port}/v1"`` and no API key. LM
    /// Studio's default port is ``1234``.
    ///
    /// Args:
    ///     model: Model identifier loaded on the server.
    ///     host: Hostname or IP. Defaults to ``"localhost"``.
    ///     port: TCP port. Defaults to LM Studio's standard ``1234``.
    #[classmethod]
    #[pyo3(signature = (model, host=None, port=None))]
    #[gen_stub(override_return_type(type_repr = "CustomProvider"))]
    fn lm_studio(
        cls: &Bound<'_, PyType>,
        model: String,
        host: Option<String>,
        port: Option<u16>,
    ) -> PyResult<Py<PyCustomProvider>> {
        let py = cls.py();
        let host = host.unwrap_or_else(|| "localhost".to_owned());
        let port = port.unwrap_or(1234);
        let handle = blazen_llm::lm_studio(&host, port, model);
        let cfg = match handle.protocol().clone() {
            ApiProtocol::OpenAi(c) => c,
            ApiProtocol::Custom => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "internal: blazen_llm::lm_studio did not yield OpenAi protocol",
                ));
            }
        };
        let proto = PyApiProtocol {
            kind: ApiProtocolKind::OpenAi(PyOpenAiCompatConfig { inner: cfg }),
        };
        let pair = build_parent_pair(handle, proto, None, "lm_studio".to_owned())?;
        Py::new(py, pair)
    }

    /// Build a [`CustomProvider`] from an arbitrary OpenAI-compatible config.
    ///
    /// Use for OpenAI-compatible servers not pre-configured by the
    /// ``ollama`` / ``lm_studio`` helpers (vLLM, llama.cpp's server, TGI,
    /// or hosted OpenAI-compat services).
    ///
    /// Args:
    ///     provider_id: Short identifier used for logging.
    ///     config: A fully-specified [`OpenAiCompatConfig`].
    #[classmethod]
    #[pyo3(signature = (provider_id, config))]
    #[gen_stub(override_return_type(type_repr = "CustomProvider"))]
    fn openai_compat(
        cls: &Bound<'_, PyType>,
        provider_id: String,
        config: PyOpenAiCompatConfig,
    ) -> PyResult<Py<PyCustomProvider>> {
        let py = cls.py();
        let handle = blazen_llm::openai_compat(provider_id.clone(), config.inner.clone());
        let proto = PyApiProtocol {
            kind: ApiProtocolKind::OpenAi(config),
        };
        let pair = build_parent_pair(handle, proto, None, provider_id)?;
        Py::new(py, pair)
    }

    // -----------------------------------------------------------------
    // Audio generation
    // -----------------------------------------------------------------

    /// Synthesize speech from text.
    ///
    /// Dispatches to ``self.text_to_speech(request)`` on subclasses, or
    /// to the underlying handle's typed method otherwise. Raises
    /// ``UnsupportedError`` if the underlying provider doesn't expose
    /// the capability.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, AudioResult]", imports = ("typing",)))]
    fn text_to_speech<'py>(
        &self,
        py: Python<'py>,
        request: PySpeechRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = CustomProvider::text_to_speech(handle.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyAudioResult { inner: result })
        })
    }

    /// Generate music.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, AudioResult]", imports = ("typing",)))]
    fn generate_music<'py>(
        &self,
        py: Python<'py>,
        request: PyMusicRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = CustomProvider::generate_music(handle.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyAudioResult { inner: result })
        })
    }

    /// Generate sound effects.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, AudioResult]", imports = ("typing",)))]
    fn generate_sfx<'py>(
        &self,
        py: Python<'py>,
        request: PyMusicRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = CustomProvider::generate_sfx(handle.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyAudioResult { inner: result })
        })
    }

    // -----------------------------------------------------------------
    // Voice cloning
    // -----------------------------------------------------------------

    /// Clone a voice from reference audio clips. Returns a persistent
    /// [`VoiceHandle`] that can be passed as ``SpeechRequest.voice`` on
    /// subsequent TTS calls.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, VoiceHandle]", imports = ("typing",)))]
    fn clone_voice<'py>(
        &self,
        py: Python<'py>,
        request: PyVoiceCloneRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = CustomProvider::clone_voice(handle.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyVoiceHandle { inner: result })
        })
    }

    /// List all voices known to the provider.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, list[VoiceHandle]]", imports = ("typing",)))]
    fn list_voices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let voices = CustomProvider::list_voices(handle.as_ref())
                .await
                .map_err(blazen_error_to_pyerr)?;
            let wrapped: Vec<PyVoiceHandle> = voices
                .into_iter()
                .map(|v| PyVoiceHandle { inner: v })
                .collect();
            Ok(wrapped)
        })
    }

    /// Delete a previously-cloned voice.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn delete_voice<'py>(
        &self,
        py: Python<'py>,
        voice: PyVoiceHandle,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_voice = voice.inner;
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            CustomProvider::delete_voice(handle.as_ref(), rust_voice)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(())
        })
    }

    // -----------------------------------------------------------------
    // Image generation
    // -----------------------------------------------------------------

    /// Generate an image.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ImageResult]", imports = ("typing",)))]
    fn generate_image<'py>(
        &self,
        py: Python<'py>,
        request: PyImageRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = CustomProvider::generate_image(handle.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyImageResult { inner: result })
        })
    }

    /// Upscale an image.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ImageResult]", imports = ("typing",)))]
    fn upscale_image<'py>(
        &self,
        py: Python<'py>,
        request: PyUpscaleRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = CustomProvider::upscale_image(handle.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyImageResult { inner: result })
        })
    }

    // -----------------------------------------------------------------
    // Video generation
    // -----------------------------------------------------------------

    /// Generate a video from text.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, VideoResult]", imports = ("typing",)))]
    fn text_to_video<'py>(
        &self,
        py: Python<'py>,
        request: PyVideoRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = CustomProvider::text_to_video(handle.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyVideoResult { inner: result })
        })
    }

    /// Generate a video from a source image.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, VideoResult]", imports = ("typing",)))]
    fn image_to_video<'py>(
        &self,
        py: Python<'py>,
        request: PyVideoRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = CustomProvider::image_to_video(handle.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyVideoResult { inner: result })
        })
    }

    // -----------------------------------------------------------------
    // Transcription
    // -----------------------------------------------------------------

    /// Transcribe audio.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, TranscriptionResult]", imports = ("typing",)))]
    fn transcribe<'py>(
        &self,
        py: Python<'py>,
        request: PyTranscriptionRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = CustomProvider::transcribe(handle.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyTranscriptionResult { inner: result })
        })
    }

    // -----------------------------------------------------------------
    // 3D generation
    // -----------------------------------------------------------------

    /// Generate a 3D model.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ThreeDResult]", imports = ("typing",)))]
    fn generate_3d<'py>(
        &self,
        py: Python<'py>,
        request: PyThreeDRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = CustomProvider::generate_3d(handle.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyThreeDResult { inner: result })
        })
    }

    // -----------------------------------------------------------------
    // Background removal
    // -----------------------------------------------------------------

    /// Remove the background from an image.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ImageResult]", imports = ("typing",)))]
    fn remove_background<'py>(
        &self,
        py: Python<'py>,
        request: PyBackgroundRemovalRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let handle = Arc::clone(&self.handle);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = CustomProvider::remove_background(handle.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyImageResult { inner: result })
        })
    }

    // -----------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------

    /// The provider id used for logging (e.g. ``"elevenlabs"``).
    #[getter]
    fn provider_id(&self) -> &str {
        ComputeProvider::provider_id(self.handle.as_ref())
    }

    /// Set the provider-level default retry config.
    ///
    /// Returns a new provider sharing the same underlying handle, with
    /// the given retry config applied. Pipeline / workflow / step /
    /// call scopes can override this; if all are unset, this is the
    /// fallback.
    #[gen_stub(override_return_type(type_repr = "CustomProvider"))]
    pub fn with_retry_config(
        &self,
        py: Python<'_>,
        config: PyRetryConfig,
    ) -> PyResult<Py<PyCustomProvider>> {
        let new_handle = (*self.handle).clone().with_retry_config(config.inner);
        let pair = build_parent_pair(
            new_handle,
            self.protocol.clone(),
            None,
            self.provider_id_str.clone(),
        )?;
        Py::new(py, pair)
    }

    /// Return an opaque handle to the underlying HTTP client, if any.
    pub fn http_client(&self) -> Option<PyHttpClientHandle> {
        self.handle
            .http_client()
            .map(|inner| PyHttpClientHandle { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "CustomProvider(provider_id={:?})",
            ComputeProvider::provider_id(self.handle.as_ref())
        )
    }
}
