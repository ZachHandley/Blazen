//! `VcStream` — lazy async iterator over streamed voice-conversion chunks.
//!
//! Mirrors the pattern used by
//! [`PyMusicStream`](crate::music::PyMusicStream) and
//! [`PyLazyCompletionStream`](crate::providers::model::PyLazyCompletionStream):
//! the underlying `Stream<Item = Result<Vec<f32>, VcError>>` is requested
//! from the backend on the first `__anext__` call, not at construction
//! time, so Python callers can do the natural one-liner:
//!
//! ```python
//! async for chunk in model.stream_convert_pcm(samples, "speaker-01"):
//!     ...
//! ```

use std::pin::Pin;
use std::sync::Arc;

use blazen_audio_vc::{VcError, VoiceConversionBackend};
use futures_core::Stream;
use futures_util::{StreamExt, stream};
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use crate::error::vc_error_to_pyerr;
use crate::vc::chunk::PyVcChunk;

/// Pending stream initialization data for [`LazyVcStreamState::NotStarted`].
///
/// Boxed to keep the enum variants size-balanced (avoids
/// `clippy::large_enum_variant`).
pub(crate) struct PendingVcStream {
    pub(crate) backend: Arc<dyn VoiceConversionBackend>,
    /// Source PCM samples, consumed exactly once on the first
    /// `__anext__` call when the underlying backend stream is requested.
    pub(crate) input_pcm: Option<Vec<f32>>,
    pub(crate) target_voice_id: String,
}

type PinnedVcStream = Pin<Box<dyn Stream<Item = Result<Vec<f32>, VcError>> + Send>>;

/// Internal state for [`PyVcStream`].
pub(crate) enum LazyVcStreamState {
    /// Underlying stream has not yet been requested.
    NotStarted(Box<PendingVcStream>),
    /// Stream is active and yielding chunks.
    Active(PinnedVcStream),
    /// Stream has been fully consumed or errored out.
    Exhausted,
}

/// Async iterator over streamed [`VcChunk`](super::PyVcChunk) emissions.
///
/// Each `__anext__` resolves to a fresh
/// [`VcChunk`](super::PyVcChunk) carrying a chunk of converted PCM
/// samples at the target voice's native sample rate. When the underlying
/// backend stream ends, the next call raises `StopAsyncIteration`. The
/// iterator is single-pass: once exhausted it stays exhausted.
#[gen_stub_pyclass]
#[pyclass(name = "VcStream")]
pub struct PyVcStream {
    pub(crate) state: Arc<Mutex<LazyVcStreamState>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVcStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, VcChunk]", imports = ("typing",)))]
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state = self.state.clone();
        future_into_py(py, async move {
            let mut guard = state.lock().await;

            // Lazily initialise the stream on the first call. We wrap
            // the caller-supplied PCM in a single-item source stream
            // (the backend trait takes `Stream<Item = Vec<f32>>`).
            if let LazyVcStreamState::NotStarted(pending) = &mut *guard {
                let input_pcm = pending.input_pcm.take().ok_or_else(|| {
                    PyRuntimeError::new_err("vc stream input PCM already consumed")
                })?;
                let target_voice_id = pending.target_voice_id.clone();
                let backend = pending.backend.clone();
                let source = Box::pin(stream::once(async move { input_pcm }));
                let result = backend.stream_convert(source, &target_voice_id).await;
                match result {
                    Ok(stream) => *guard = LazyVcStreamState::Active(stream),
                    Err(e) => {
                        *guard = LazyVcStreamState::Exhausted;
                        return Err(vc_error_to_pyerr(e));
                    }
                }
            }

            match &mut *guard {
                LazyVcStreamState::Active(stream) => match stream.next().await {
                    Some(Ok(samples)) => Ok(PyVcChunk::from_streamed_samples(samples)),
                    Some(Err(e)) => {
                        *guard = LazyVcStreamState::Exhausted;
                        Err(vc_error_to_pyerr(e))
                    }
                    None => {
                        *guard = LazyVcStreamState::Exhausted;
                        Err(PyStopAsyncIteration::new_err("vc stream exhausted"))
                    }
                },
                LazyVcStreamState::Exhausted => {
                    Err(PyStopAsyncIteration::new_err("vc stream exhausted"))
                }
                // Unreachable: we just initialised above.
                LazyVcStreamState::NotStarted(_) => {
                    Err(PyRuntimeError::new_err("vc stream in inconsistent state"))
                }
            }
        })
    }

    fn __repr__(&self) -> String {
        "VcStream(...)".to_owned()
    }
}
