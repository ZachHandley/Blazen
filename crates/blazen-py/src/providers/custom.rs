//! Python wrapper for user-defined custom providers.
//!
//! Lets Python users write a normal class with async capability methods
//! and wrap it as a first-class Blazen provider via
//! [`CustomProvider`](blazen_llm::CustomProvider). The workflow engine
//! then sees the wrapped object as a provider that implements whichever
//! combination of [`AudioGeneration`], [`VoiceCloning`],
//! [`ImageGeneration`], [`VideoGeneration`], [`Transcription`],
//! [`ThreeDGeneration`], and [`BackgroundRemoval`] traits the host
//! class's methods cover.
//!
//! ## Bridging Python async -> Rust async
//!
//! Each capability method on [`PyCustomProvider`] looks like a standard
//! async method from Python's perspective: it returns an awaitable. Under
//! the hood, the Rust side:
//!
//! 1. Serializes the typed request (`PySpeechRequest`, etc.) into
//!    `serde_json::Value` via the `CustomProvider::call_typed` helper.
//! 2. Hands the JSON to [`PyHostDispatch::call`], which:
//!     - Acquires the GIL
//!     - Converts `serde_json::Value` into a Python dict via `pythonize`
//!     - Calls the named method on the held `Py<PyAny>` to get a coroutine
//!     - Converts the coroutine to a Rust future via
//!       [`pyo3_async_runtimes::into_future_with_locals`], using the task
//!       locals captured from the active `future_into_py` scope
//!     - Awaits the future (GIL released)
//!     - Reacquires the GIL and depythonizes the result back into
//!       `serde_json::Value`
//! 3. Deserializes the JSON into the capability method's return type.
//!
//! The `has_method` fast-path uses a cached `hasattr` check to avoid
//! paying for a GIL acquisition on every call after the first one.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::compute::{
    AudioGeneration, BackgroundRemoval, ImageGeneration, ThreeDGeneration, Transcription,
    VideoGeneration, VoiceCloning,
};
use blazen_llm::error::BlazenError;
use blazen_llm::{CustomProvider, HostDispatch};

use crate::compute::request_types::{
    PyBackgroundRemovalRequest, PyImageRequest, PyMusicRequest, PySpeechRequest, PyThreeDRequest,
    PyTranscriptionRequest, PyUpscaleRequest, PyVideoRequest, PyVoiceCloneRequest,
};
use crate::compute::result_types::{
    PyAudioResult, PyImageResult, PyThreeDResult, PyTranscriptionResult, PyVideoResult,
    PyVoiceHandle,
};
use crate::error::blazen_error_to_pyerr;

// ---------------------------------------------------------------------------
// PyHostDispatch
// ---------------------------------------------------------------------------

/// [`HostDispatch`] implementation over a Python object held as `Py<PyAny>`.
///
/// Translates each JSON `call` into:
///
/// 1. `pythonize` the JSON request into a Python dict
/// 2. `host.call_method1(method, (request,))` to obtain a coroutine
/// 3. Convert to a Rust future via `pyo3_async_runtimes::into_future_with_locals`
/// 4. Await the future outside the GIL
/// 5. `depythonize` the result into JSON
///
/// `has_method` uses a `Mutex<HashMap>` cache so repeated calls only pay
/// the GIL cost once per method name.
pub struct PyHostDispatch {
    /// The Python host object (typically a class instance with async methods).
    host: Py<PyAny>,
    /// Cache of `hasattr` results to avoid re-acquiring the GIL on every
    /// `has_method` call. Keyed by method name.
    has_method_cache: Mutex<HashMap<String, bool>>,
}

impl PyHostDispatch {
    /// Construct a new dispatch wrapping the given Python host object.
    pub fn new(host: Py<PyAny>) -> Self {
        Self {
            host,
            has_method_cache: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl HostDispatch for PyHostDispatch {
    async fn call(
        &self,
        method: &str,
        request: serde_json::Value,
    ) -> Result<serde_json::Value, BlazenError> {
        // Phase 1: acquire GIL, convert JSON -> Python, call method to get
        // a coroutine, and convert the coroutine into a Rust future. All
        // of this happens synchronously under the GIL, hence `block_in_place`
        // to avoid blocking the tokio worker thread.
        let future_result = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                // Serialize the JSON request into a Python object.
                let py_request = pythonize::pythonize(py, &request).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "failed to pythonize request for method `{method}`: {e}"
                    ))
                })?;

                // Call the named method on the host object. This produces a
                // coroutine (assuming the host method is `async def`).
                let host_bound = self.host.bind(py);
                let call_result = host_bound
                    .call_method1(method, (py_request,))
                    .map_err(|e| {
                        // If the host raises before producing a coroutine
                        // (e.g. the method is missing, or it's a sync method
                        // that raised directly), surface that as a PyErr.
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "host method `{method}` raised before yielding a coroutine: {e}"
                        ))
                    })?;

                // Capture the active asyncio task locals so the coroutine
                // runs on the correct event loop when we await it below.
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;

                // Convert the coroutine into a Rust future. If the host
                // method was synchronous (returned a plain value instead of
                // a coroutine), this will raise a clear TypeError.
                let fut = pyo3_async_runtimes::into_future_with_locals(&locals, call_result)
                    .map_err(|e| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "host method `{method}` did not return an awaitable coroutine (custom providers require `async def` methods): {e}"
                        ))
                    })?;

                Ok((fut, locals))
            })
        });

        let (fut, locals) = future_result.map_err(|e: PyErr| {
            BlazenError::provider("custom", format!("dispatch setup failed: {e}"))
        })?;

        // Phase 2: drive the Python coroutine to completion. Must run
        // inside `scope` so `into_future`-style calls made from nested
        // host code can find the same task locals.
        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| {
                BlazenError::provider("custom", format!("host method `{method}` raised: {e}"))
            })?;

        // Phase 3: reacquire GIL and convert the Python result back into
        // serde JSON for the `CustomProvider::call_typed` deserializer.
        let json_value = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<serde_json::Value> {
                let bound = py_result.bind(py);
                // Special-case `None` so host methods like `delete_voice`
                // (which return nothing meaningful) deserialize cleanly
                // into `()`/`Value::Null` downstream.
                if bound.is_none() {
                    return Ok(serde_json::Value::Null);
                }
                let value: serde_json::Value = pythonize::depythonize(bound).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "failed to depythonize result of `{method}`: {e}"
                    ))
                })?;
                Ok(value)
            })
        })
        .map_err(|e: PyErr| {
            BlazenError::provider(
                "custom",
                format!("failed to convert result of `{method}` to JSON: {e}"),
            )
        })?;

        Ok(json_value)
    }

    fn has_method(&self, method: &str) -> bool {
        // Fast path: check the cache without touching the GIL.
        if let Ok(cache) = self.has_method_cache.lock()
            && let Some(&cached) = cache.get(method)
        {
            return cached;
        }

        // Slow path: acquire the GIL and call `hasattr`. Any GIL error
        // conservatively reports "not supported" so the caller falls
        // through to `BlazenError::Unsupported`.
        let has = Python::attach(|py| -> PyResult<bool> {
            let host_bound = self.host.bind(py);
            let attr = host_bound.hasattr(method)?;
            if !attr {
                return Ok(false);
            }
            // Also confirm it's callable, not a plain data attribute.
            let attr_obj = host_bound.getattr(method)?;
            Ok(attr_obj.is_callable())
        })
        .unwrap_or(false);

        if let Ok(mut cache) = self.has_method_cache.lock() {
            cache.insert(method.to_owned(), has);
        }
        has
    }
}

// ---------------------------------------------------------------------------
// PyCustomProvider
// ---------------------------------------------------------------------------

/// A user-defined Blazen provider backed by a Python class instance.
///
/// Wraps an arbitrary Python object whose async methods match Blazen's
/// capability trait names (``text_to_speech``, ``clone_voice``,
/// ``generate_image``, etc.) and exposes them as a first-class provider.
/// The workflow engine treats the result as implementing every capability
/// trait whose methods the wrapped object provides; missing methods
/// return ``UnsupportedError`` when called.
///
/// Request/response shapes use Blazen's typed request/result classes on
/// the Python side and get serialized through ``pythonize`` to the
/// wrapped object's methods, which receive/return plain dicts.
///
/// Example:
///     >>> import base64
///     >>> from elevenlabs.client import AsyncElevenLabs
///     >>> class ElevenLabsProvider:
///     ...     def __init__(self, api_key):
///     ...         self._client = AsyncElevenLabs(api_key=api_key)
///     ...     async def text_to_speech(self, request):
///     ...         audio = b"".join([
///     ...             chunk async for chunk in self._client.text_to_speech.convert(
///     ...                 voice_id=request["voice"],
///     ...                 text=request["text"],
///     ...                 model_id="eleven_multilingual_v2",
///     ...             )
///     ...         ])
///     ...         return {
///     ...             "audio": [{
///     ...                 "media": {
///     ...                     "base64": base64.b64encode(audio).decode(),
///     ...                     "media_type": "mpeg",
///     ...                 },
///     ...             }],
///     ...             "timing": {"total_ms": 0, "queue_ms": None, "execution_ms": None},
///     ...             "metadata": {},
///     ...         }
///     >>> provider = CustomProvider(
///     ...     ElevenLabsProvider(api_key="..."),
///     ...     provider_id="elevenlabs",
///     ... )
///     >>> result = await provider.text_to_speech(SpeechRequest(text="hi", voice="rachel"))
#[gen_stub_pyclass]
#[pyclass(name = "CustomProvider", from_py_object)]
#[derive(Clone)]
pub struct PyCustomProvider {
    inner: Arc<CustomProvider<PyHostDispatch>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCustomProvider {
    /// Wrap a Python host object as a Blazen [`CustomProvider`].
    ///
    /// Args:
    ///     host_object: A Python class instance whose async methods match
    ///         Blazen capability trait methods
    ///         (``text_to_speech``, ``generate_image``, ``clone_voice``, ...).
    ///         Host methods must be ``async def`` and accept a single dict
    ///         argument shaped like the corresponding Blazen request type.
    ///     provider_id: Optional short identifier used for logging and
    ///         returned by ``provider_id``. Defaults to ``"custom"``.
    #[new]
    #[pyo3(signature = (host_object, *, provider_id=None))]
    fn new(host_object: Py<PyAny>, provider_id: Option<String>) -> Self {
        let id = provider_id.unwrap_or_else(|| "custom".to_owned());
        let dispatch = PyHostDispatch::new(host_object);
        Self {
            inner: Arc::new(CustomProvider::new(id, dispatch)),
        }
    }

    // -----------------------------------------------------------------
    // Audio generation
    // -----------------------------------------------------------------

    /// Synthesize speech from text by calling the host's
    /// ``text_to_speech`` async method.
    async fn text_to_speech(&self, request: PySpeechRequest) -> PyResult<PyAudioResult> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        let result = AudioGeneration::text_to_speech(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_pyerr)?;
        Ok(PyAudioResult { inner: result })
    }

    /// Generate music by calling the host's ``generate_music`` async method.
    async fn generate_music(&self, request: PyMusicRequest) -> PyResult<PyAudioResult> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        let result = AudioGeneration::generate_music(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_pyerr)?;
        Ok(PyAudioResult { inner: result })
    }

    /// Generate sound effects by calling the host's ``generate_sfx`` async method.
    async fn generate_sfx(&self, request: PyMusicRequest) -> PyResult<PyAudioResult> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        let result = AudioGeneration::generate_sfx(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_pyerr)?;
        Ok(PyAudioResult { inner: result })
    }

    // -----------------------------------------------------------------
    // Voice cloning
    // -----------------------------------------------------------------

    /// Clone a voice from reference audio clips by calling the host's
    /// ``clone_voice`` async method. Returns a persistent
    /// [`VoiceHandle`] that can be passed as ``SpeechRequest.voice`` on
    /// subsequent TTS calls.
    async fn clone_voice(&self, request: PyVoiceCloneRequest) -> PyResult<PyVoiceHandle> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        let result = VoiceCloning::clone_voice(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_pyerr)?;
        Ok(PyVoiceHandle { inner: result })
    }

    /// List all voices known to the host by calling its ``list_voices``
    /// async method (which must return a list of dicts shaped like
    /// [`VoiceHandle`]).
    async fn list_voices(&self) -> PyResult<Vec<PyVoiceHandle>> {
        let inner = self.inner.clone();
        let voices = VoiceCloning::list_voices(inner.as_ref())
            .await
            .map_err(blazen_error_to_pyerr)?;
        let wrapped: Vec<PyVoiceHandle> = voices
            .into_iter()
            .map(|v| PyVoiceHandle { inner: v })
            .collect();
        Ok(wrapped)
    }

    /// Delete a previously cloned voice by calling the host's
    /// ``delete_voice`` async method.
    async fn delete_voice(&self, voice: PyVoiceHandle) -> PyResult<()> {
        let rust_voice = voice.inner;
        let inner = self.inner.clone();
        VoiceCloning::delete_voice(inner.as_ref(), &rust_voice)
            .await
            .map_err(blazen_error_to_pyerr)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Image generation
    // -----------------------------------------------------------------

    /// Generate an image by calling the host's ``generate_image`` async method.
    async fn generate_image(&self, request: PyImageRequest) -> PyResult<PyImageResult> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        let result = ImageGeneration::generate_image(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_pyerr)?;
        Ok(PyImageResult { inner: result })
    }

    /// Upscale an image by calling the host's ``upscale_image`` async method.
    async fn upscale_image(&self, request: PyUpscaleRequest) -> PyResult<PyImageResult> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        let result = ImageGeneration::upscale_image(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_pyerr)?;
        Ok(PyImageResult { inner: result })
    }

    // -----------------------------------------------------------------
    // Video generation
    // -----------------------------------------------------------------

    /// Generate a video from text by calling the host's ``text_to_video``
    /// async method.
    async fn text_to_video(&self, request: PyVideoRequest) -> PyResult<PyVideoResult> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        let result = VideoGeneration::text_to_video(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_pyerr)?;
        Ok(PyVideoResult { inner: result })
    }

    /// Generate a video from an image by calling the host's
    /// ``image_to_video`` async method.
    async fn image_to_video(&self, request: PyVideoRequest) -> PyResult<PyVideoResult> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        let result = VideoGeneration::image_to_video(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_pyerr)?;
        Ok(PyVideoResult { inner: result })
    }

    // -----------------------------------------------------------------
    // Transcription
    // -----------------------------------------------------------------

    /// Transcribe audio by calling the host's ``transcribe`` async method.
    async fn transcribe(&self, request: PyTranscriptionRequest) -> PyResult<PyTranscriptionResult> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        let result = Transcription::transcribe(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_pyerr)?;
        Ok(PyTranscriptionResult { inner: result })
    }

    // -----------------------------------------------------------------
    // 3D generation
    // -----------------------------------------------------------------

    /// Generate a 3D model by calling the host's ``generate_3d`` async method.
    async fn generate_3d(&self, request: PyThreeDRequest) -> PyResult<PyThreeDResult> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        let result = ThreeDGeneration::generate_3d(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_pyerr)?;
        Ok(PyThreeDResult { inner: result })
    }

    // -----------------------------------------------------------------
    // Background removal
    // -----------------------------------------------------------------

    /// Remove the background from an image by calling the host's
    /// ``remove_background`` async method.
    async fn remove_background(
        &self,
        request: PyBackgroundRemovalRequest,
    ) -> PyResult<PyImageResult> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        let result = BackgroundRemoval::remove_background(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_pyerr)?;
        Ok(PyImageResult { inner: result })
    }

    // -----------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------

    /// The provider id used for logging (e.g. ``"elevenlabs"``).
    #[getter]
    fn provider_id(&self) -> &str {
        use blazen_llm::compute::ComputeProvider;
        ComputeProvider::provider_id(self.inner.as_ref())
    }

    fn __repr__(&self) -> String {
        use blazen_llm::compute::ComputeProvider;
        format!(
            "CustomProvider(provider_id={:?})",
            ComputeProvider::provider_id(self.inner.as_ref())
        )
    }
}
