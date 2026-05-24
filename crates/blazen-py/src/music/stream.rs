//! `MusicStream` — lazy async iterator over streamed music chunks.
//!
//! Mirrors the pattern used by
//! [`PyLazyCompletionStream`](crate::providers::model::PyLazyCompletionStream):
//! the underlying `Stream<Item = MusicChunk>` is requested from the
//! backend on the first `__anext__` call, not at construction time, so
//! Python callers can do the natural one-liner:
//!
//! ```python
//! async for chunk in model.stream_generate_music(prompt, 5.0):
//!     ...
//! ```

use std::pin::Pin;
use std::sync::Arc;

use blazen_audio_music::{MusicBackend, MusicChunk, MusicError};
use futures_core::Stream;
use futures_util::StreamExt;
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use crate::error::music_error_to_pyerr;
use crate::music::chunk::PyMusicChunk;

/// Which streaming entry point on the backend the iterator will call.
pub(crate) enum MusicStreamKind {
    Music,
    Sfx,
}

/// Pending stream initialization data for [`LazyMusicStreamState::NotStarted`].
///
/// Boxed to keep the enum variants size-balanced (avoids
/// `clippy::large_enum_variant`).
pub(crate) struct PendingMusicStream {
    pub(crate) backend: Arc<dyn MusicBackend>,
    pub(crate) prompt: Option<String>,
    pub(crate) duration_seconds: f32,
    pub(crate) kind: MusicStreamKind,
}

type PinnedMusicStream = Pin<Box<dyn Stream<Item = Result<MusicChunk, MusicError>> + Send>>;

/// Internal state for [`PyMusicStream`].
pub(crate) enum LazyMusicStreamState {
    /// Underlying stream has not yet been requested.
    NotStarted(Box<PendingMusicStream>),
    /// Stream is active and yielding chunks.
    Active(PinnedMusicStream),
    /// Stream has been fully consumed or errored out.
    Exhausted,
}

/// Async iterator over streamed [`MusicChunk`] emissions.
///
/// Each `__anext__` resolves to a fresh
/// [`MusicChunk`](crate::music::chunk::PyMusicChunk). When the underlying
/// backend stream ends, the next call raises `StopAsyncIteration`. The
/// iterator is single-pass: once exhausted it stays exhausted.
#[gen_stub_pyclass]
#[pyclass(name = "MusicStream")]
pub struct PyMusicStream {
    pub(crate) state: Arc<Mutex<LazyMusicStreamState>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMusicStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, MusicChunk]", imports = ("typing",)))]
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state = self.state.clone();
        future_into_py(py, async move {
            let mut guard = state.lock().await;

            // Lazily initialise the stream on the first call.
            if let LazyMusicStreamState::NotStarted(pending) = &mut *guard {
                let prompt = pending.prompt.take().ok_or_else(|| {
                    PyRuntimeError::new_err("music stream prompt already consumed")
                })?;
                let duration_seconds = pending.duration_seconds;
                let backend = pending.backend.clone();
                let result = match pending.kind {
                    MusicStreamKind::Music => {
                        backend
                            .stream_generate_music(&prompt, duration_seconds)
                            .await
                    }
                    MusicStreamKind::Sfx => {
                        backend.stream_generate_sfx(&prompt, duration_seconds).await
                    }
                };
                match result {
                    Ok(stream) => *guard = LazyMusicStreamState::Active(stream),
                    Err(e) => {
                        *guard = LazyMusicStreamState::Exhausted;
                        return Err(music_error_to_pyerr(e));
                    }
                }
            }

            match &mut *guard {
                LazyMusicStreamState::Active(stream) => match stream.next().await {
                    Some(Ok(chunk)) => Ok(PyMusicChunk::from(chunk)),
                    Some(Err(e)) => {
                        *guard = LazyMusicStreamState::Exhausted;
                        Err(music_error_to_pyerr(e))
                    }
                    None => {
                        *guard = LazyMusicStreamState::Exhausted;
                        Err(PyStopAsyncIteration::new_err("music stream exhausted"))
                    }
                },
                LazyMusicStreamState::Exhausted => {
                    Err(PyStopAsyncIteration::new_err("music stream exhausted"))
                }
                // Unreachable: we just initialised above.
                LazyMusicStreamState::NotStarted(_) => Err(PyRuntimeError::new_err(
                    "music stream in inconsistent state",
                )),
            }
        })
    }

    fn __repr__(&self) -> String {
        "MusicStream(...)".to_owned()
    }
}
