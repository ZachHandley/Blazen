//! Python wrappers for streaming completion types.
//!
//! Exposes `StreamChunk` (per-chunk delta payload) plus the workflow-level
//! events `StreamChunkEvent` and `StreamCompleteEvent`.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::events::{StreamChunkEvent, StreamCompleteEvent};
use blazen_llm::types::StreamChunk as RustStreamChunk;

use super::{PyArtifact, PyCitation, PyTokenUsage, PyToolCall};

// Keep the legacy public alias so existing imports continue to compile.
pub use blazen_llm::types::StreamChunk;

// ---------------------------------------------------------------------------
// PyStreamChunk
// ---------------------------------------------------------------------------

/// A single chunk from a streaming completion response.
///
/// Yielded by ``CompletionModel.stream(...)`` either via async iteration or
/// the ``on_chunk`` callback. Carries an incremental text ``delta`` plus any
/// tool calls, reasoning deltas, citations, or artifacts that landed in this
/// chunk. The ``finish_reason`` is set on the terminal chunk.
#[gen_stub_pyclass]
#[pyclass(name = "StreamChunk", frozen, from_py_object)]
#[derive(Clone, Default)]
pub struct PyStreamChunk {
    pub(crate) inner: RustStreamChunk,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStreamChunk {
    /// Construct a stream chunk explicitly.
    #[new]
    #[pyo3(signature = (
        *,
        delta=None,
        tool_calls=None,
        finish_reason=None,
        reasoning_delta=None,
    ))]
    fn new(
        delta: Option<String>,
        tool_calls: Option<Vec<PyToolCall>>,
        finish_reason: Option<String>,
        reasoning_delta: Option<String>,
    ) -> Self {
        Self {
            inner: RustStreamChunk {
                delta,
                tool_calls: tool_calls
                    .map(|v| v.into_iter().map(|tc| tc.inner).collect())
                    .unwrap_or_default(),
                finish_reason,
                reasoning_delta,
                citations: Vec::new(),
                artifacts: Vec::new(),
            },
        }
    }

    /// Incremental text content from this chunk, if any.
    #[getter]
    fn delta(&self) -> Option<&str> {
        self.inner.delta.as_deref()
    }

    /// Tool invocations completed in this chunk.
    #[getter]
    fn tool_calls(&self) -> Vec<PyToolCall> {
        self.inner.tool_calls.iter().map(PyToolCall::from).collect()
    }

    /// Finish reason on the terminal chunk.
    #[getter]
    fn finish_reason(&self) -> Option<&str> {
        self.inner.finish_reason.as_deref()
    }

    /// Reasoning text delta (Anthropic thinking, R1 reasoning_content, o-series).
    #[getter]
    fn reasoning_delta(&self) -> Option<&str> {
        self.inner.reasoning_delta.as_deref()
    }

    /// Citations completed in this chunk.
    #[getter]
    fn citations(&self) -> Vec<PyCitation> {
        self.inner.citations.iter().map(PyCitation::from).collect()
    }

    /// Artifacts completed in this chunk.
    #[getter]
    fn artifacts(&self) -> Vec<PyArtifact> {
        self.inner.artifacts.iter().map(PyArtifact::from).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamChunk(delta={:?}, finish_reason={:?})",
            self.inner.delta, self.inner.finish_reason
        )
    }
}

impl From<RustStreamChunk> for PyStreamChunk {
    fn from(inner: RustStreamChunk) -> Self {
        Self { inner }
    }
}

impl From<&RustStreamChunk> for PyStreamChunk {
    fn from(inner: &RustStreamChunk) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyStreamChunkEvent
// ---------------------------------------------------------------------------

/// Workflow event emitted for each incremental chunk during a streaming
/// completion.
///
/// Mirrors [`blazen_llm::events::StreamChunkEvent`]. Steps that bridge LLM
/// streaming into the workflow event bus publish one of these per chunk so
/// downstream consumers can observe progress in real time.
#[gen_stub_pyclass]
#[pyclass(name = "StreamChunkEvent", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyStreamChunkEvent {
    pub(crate) inner: StreamChunkEvent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStreamChunkEvent {
    /// Construct a stream-chunk event.
    #[new]
    #[pyo3(signature = (*, delta, model, finish_reason=None))]
    fn new(delta: String, model: String, finish_reason: Option<String>) -> Self {
        Self {
            inner: StreamChunkEvent {
                delta,
                finish_reason,
                model,
            },
        }
    }

    /// The incremental text content from this chunk.
    #[getter]
    fn delta(&self) -> &str {
        &self.inner.delta
    }

    /// Finish reason set on the final chunk, if applicable.
    #[getter]
    fn finish_reason(&self) -> Option<&str> {
        self.inner.finish_reason.as_deref()
    }

    /// The model that produced this chunk.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamChunkEvent(model={:?}, delta={:?})",
            self.inner.model, self.inner.delta
        )
    }
}

impl From<StreamChunkEvent> for PyStreamChunkEvent {
    fn from(inner: StreamChunkEvent) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyStreamCompleteEvent
// ---------------------------------------------------------------------------

/// Workflow event emitted once a streaming completion has fully finished.
///
/// Mirrors [`blazen_llm::events::StreamCompleteEvent`]. Carries the
/// concatenated full text, optional usage stats, and the model id.
#[gen_stub_pyclass]
#[pyclass(name = "StreamCompleteEvent", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyStreamCompleteEvent {
    pub(crate) inner: StreamCompleteEvent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStreamCompleteEvent {
    /// Construct a stream-complete event.
    #[new]
    #[pyo3(signature = (*, full_text, model, usage=None))]
    fn new(full_text: String, model: String, usage: Option<PyTokenUsage>) -> Self {
        Self {
            inner: StreamCompleteEvent {
                full_text,
                usage: usage.map(|u| u.inner),
                model,
            },
        }
    }

    /// The full concatenated text of the streamed response.
    #[getter]
    fn full_text(&self) -> &str {
        &self.inner.full_text
    }

    /// Token usage statistics, if reported by the provider.
    #[getter]
    fn usage(&self) -> Option<PyTokenUsage> {
        self.inner.usage.as_ref().map(PyTokenUsage::from)
    }

    /// The model that produced the response.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamCompleteEvent(model={:?}, full_text_len={})",
            self.inner.model,
            self.inner.full_text.len()
        )
    }
}

impl From<StreamCompleteEvent> for PyStreamCompleteEvent {
    fn from(inner: StreamCompleteEvent) -> Self {
        Self { inner }
    }
}
