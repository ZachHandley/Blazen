//! Python wrappers for `blazen_events` foundation types.
//!
//! Surfaces [`UsageEvent`](blazen_events::UsageEvent),
//! [`Modality`](blazen_events::Modality),
//! [`ProgressEvent`](blazen_events::ProgressEvent),
//! [`ProgressKind`](blazen_events::ProgressKind), and the
//! [`ProgressSnapshot`](blazen_pipeline::ProgressSnapshot) read-only handle to
//! Python so callers can both listen for the events on a workflow / pipeline
//! event stream and synthesise their own (e.g. from a custom provider that
//! wants to participate in the cost-rollup machinery).
//!
//! `Modality` is a discriminator-style enum — the `Custom(String)` variant
//! is exposed via the `custom_label: Option<str>` getter alongside an
//! `is_custom()` predicate, since `pyo3-stub-gen`'s enum-with-payload story
//! is awkward enough that a single `kind` getter + label pair is friendlier
//! than fighting it.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use uuid::Uuid;

use blazen_events::{Modality, ProgressEvent, ProgressKind, UsageEvent};
use blazen_pipeline::ProgressSnapshot;

// ---------------------------------------------------------------------------
// PyModality
// ---------------------------------------------------------------------------

/// The kind of provider call that produced a [`UsageEvent`].
///
/// Mirrors [`blazen_events::Modality`]. Use the static constructors for the
/// fixed variants and ``Modality.custom("name")`` for the user-defined slot.
/// Inspect via the `kind` getter (returns the variant name) and the
/// `custom_label` getter (the inner string when `kind == "Custom"`).
#[gen_stub_pyclass]
#[pyclass(name = "Modality", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyModality {
    pub(crate) inner: Modality,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModality {
    /// Default construct as `Modality.Llm`.
    #[new]
    fn new() -> Self {
        Self {
            inner: Modality::default(),
        }
    }

    /// LLM completion / chat / structured-output call.
    #[staticmethod]
    fn llm() -> Self {
        Self {
            inner: Modality::Llm,
        }
    }

    /// Embedding call.
    #[staticmethod]
    fn embedding() -> Self {
        Self {
            inner: Modality::Embedding,
        }
    }

    /// Image-generation call.
    #[staticmethod]
    fn image_gen() -> Self {
        Self {
            inner: Modality::ImageGen,
        }
    }

    /// Text-to-speech call.
    #[staticmethod]
    fn audio_tts() -> Self {
        Self {
            inner: Modality::AudioTts,
        }
    }

    /// Speech-to-text call.
    #[staticmethod]
    fn audio_stt() -> Self {
        Self {
            inner: Modality::AudioStt,
        }
    }

    /// Video-generation call.
    #[staticmethod]
    fn video() -> Self {
        Self {
            inner: Modality::Video,
        }
    }

    /// 3D-asset generation call.
    #[staticmethod]
    fn three_d() -> Self {
        Self {
            inner: Modality::ThreeD,
        }
    }

    /// Background-removal call.
    #[staticmethod]
    fn background_removal() -> Self {
        Self {
            inner: Modality::BackgroundRemoval,
        }
    }

    /// User-defined modality with a free-form label.
    #[staticmethod]
    fn custom(label: &str) -> Self {
        Self {
            inner: Modality::Custom(label.to_owned()),
        }
    }

    /// Variant tag — one of `Llm`, `Embedding`, `ImageGen`, `AudioTts`,
    /// `AudioStt`, `Video`, `ThreeD`, `BackgroundRemoval`, `Custom`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            Modality::Llm => "Llm",
            Modality::Embedding => "Embedding",
            Modality::ImageGen => "ImageGen",
            Modality::AudioTts => "AudioTts",
            Modality::AudioStt => "AudioStt",
            Modality::Video => "Video",
            Modality::ThreeD => "ThreeD",
            Modality::BackgroundRemoval => "BackgroundRemoval",
            Modality::Custom(_) => "Custom",
        }
    }

    /// The custom label when `kind == "Custom"`; `None` otherwise.
    #[getter]
    fn custom_label(&self) -> Option<String> {
        match &self.inner {
            Modality::Custom(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Whether this is a user-defined `Custom(name)` variant.
    fn is_custom(&self) -> bool {
        matches!(self.inner, Modality::Custom(_))
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Modality::Custom(s) => format!("Modality.Custom({s:?})"),
            other => format!("Modality.{other:?}"),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl From<Modality> for PyModality {
    fn from(inner: Modality) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyUsageEvent
// ---------------------------------------------------------------------------

/// Emitted by every provider call to surface tokens, modality-specific
/// quantities, and cost from the provider's actual API response. Mirrors
/// [`blazen_events::UsageEvent`].
///
/// `Pipeline` and `WorkflowHandler` aggregate these into the rollups
/// surfaced via `usage_total` / `cost_total_usd`. Custom providers can
/// build and emit their own via `Context.emit_usage(event)` so they
/// participate in the same accounting.
#[gen_stub_pyclass]
#[pyclass(name = "UsageEvent", from_py_object)]
#[derive(Clone)]
pub struct PyUsageEvent {
    pub(crate) inner: UsageEvent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyUsageEvent {
    /// Build a `UsageEvent`.
    ///
    /// Args:
    ///     provider: The provider that served the call (e.g. `"openai"`).
    ///     model: The model identifier (e.g. `"gpt-4o-mini"`).
    ///     modality: The kind of call. Defaults to `Modality.Llm`.
    ///     prompt_tokens: Prompt / input tokens billed.
    ///     completion_tokens: Completion / output tokens billed.
    ///     total_tokens: Total tokens billed.
    ///     reasoning_tokens: Hidden reasoning tokens (o-series, R1, ...).
    ///     cached_input_tokens: Tokens served from a prompt cache.
    ///     audio_input_tokens: Audio input tokens (multimodal).
    ///     audio_output_tokens: Audio output tokens (multimodal).
    ///     image_count: Images generated or processed.
    ///     audio_seconds: Audio duration (TTS / STT).
    ///     video_seconds: Video duration (video generation).
    ///     cost_usd: USD cost for this call (provider-reported or computed).
    ///     latency_ms: Wall-clock latency.
    ///     run_id: Pipeline / workflow run identifier (UUID string). When
    ///         `None`, a fresh `uuid4` is generated so events are unique.
    #[new]
    #[pyo3(signature = (
        *,
        provider,
        model,
        modality=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        reasoning_tokens=0,
        cached_input_tokens=0,
        audio_input_tokens=0,
        audio_output_tokens=0,
        image_count=0,
        audio_seconds=0.0,
        video_seconds=0.0,
        cost_usd=None,
        latency_ms=0,
        run_id=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        provider: String,
        model: String,
        modality: Option<PyModality>,
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        reasoning_tokens: u32,
        cached_input_tokens: u32,
        audio_input_tokens: u32,
        audio_output_tokens: u32,
        image_count: u32,
        audio_seconds: f64,
        video_seconds: f64,
        cost_usd: Option<f64>,
        latency_ms: u64,
        run_id: Option<String>,
    ) -> PyResult<Self> {
        let parsed_run_id = match run_id {
            Some(s) => Uuid::parse_str(&s).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid run_id: {e}"))
            })?,
            None => Uuid::new_v4(),
        };
        Ok(Self {
            inner: UsageEvent {
                provider,
                model,
                modality: modality.map(|m| m.inner).unwrap_or_default(),
                prompt_tokens,
                completion_tokens,
                total_tokens,
                reasoning_tokens,
                cached_input_tokens,
                audio_input_tokens,
                audio_output_tokens,
                image_count,
                audio_seconds,
                video_seconds,
                cost_usd,
                latency_ms,
                run_id: parsed_run_id,
            },
        })
    }

    #[getter]
    fn provider(&self) -> &str {
        &self.inner.provider
    }
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }
    #[getter]
    fn modality(&self) -> PyModality {
        PyModality {
            inner: self.inner.modality.clone(),
        }
    }
    #[getter]
    fn prompt_tokens(&self) -> u32 {
        self.inner.prompt_tokens
    }
    #[getter]
    fn completion_tokens(&self) -> u32 {
        self.inner.completion_tokens
    }
    #[getter]
    fn total_tokens(&self) -> u32 {
        self.inner.total_tokens
    }
    #[getter]
    fn reasoning_tokens(&self) -> u32 {
        self.inner.reasoning_tokens
    }
    #[getter]
    fn cached_input_tokens(&self) -> u32 {
        self.inner.cached_input_tokens
    }
    #[getter]
    fn audio_input_tokens(&self) -> u32 {
        self.inner.audio_input_tokens
    }
    #[getter]
    fn audio_output_tokens(&self) -> u32 {
        self.inner.audio_output_tokens
    }
    #[getter]
    fn image_count(&self) -> u32 {
        self.inner.image_count
    }
    #[getter]
    fn audio_seconds(&self) -> f64 {
        self.inner.audio_seconds
    }
    #[getter]
    fn video_seconds(&self) -> f64 {
        self.inner.video_seconds
    }
    #[getter]
    fn cost_usd(&self) -> Option<f64> {
        self.inner.cost_usd
    }
    #[getter]
    fn latency_ms(&self) -> u64 {
        self.inner.latency_ms
    }
    /// The run identifier as an RFC-4122 UUID string.
    #[getter]
    fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "UsageEvent(provider={:?}, model={:?}, modality={:?}, total_tokens={}, cost_usd={:?})",
            self.inner.provider,
            self.inner.model,
            self.inner.modality,
            self.inner.total_tokens,
            self.inner.cost_usd,
        )
    }
}

impl From<UsageEvent> for PyUsageEvent {
    fn from(inner: UsageEvent) -> Self {
        Self { inner }
    }
}

impl From<&UsageEvent> for PyUsageEvent {
    fn from(inner: &UsageEvent) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyProgressKind
// ---------------------------------------------------------------------------

/// The kind of progress this event describes.
#[gen_stub_pyclass_enum]
#[pyclass(name = "ProgressKind", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyProgressKind {
    Pipeline,
    Workflow,
    SubWorkflow,
    Stage,
}

impl From<ProgressKind> for PyProgressKind {
    fn from(k: ProgressKind) -> Self {
        match k {
            ProgressKind::Pipeline => Self::Pipeline,
            ProgressKind::Workflow => Self::Workflow,
            ProgressKind::SubWorkflow => Self::SubWorkflow,
            ProgressKind::Stage => Self::Stage,
        }
    }
}

impl From<PyProgressKind> for ProgressKind {
    fn from(k: PyProgressKind) -> Self {
        match k {
            PyProgressKind::Pipeline => Self::Pipeline,
            PyProgressKind::Workflow => Self::Workflow,
            PyProgressKind::SubWorkflow => Self::SubWorkflow,
            PyProgressKind::Stage => Self::Stage,
        }
    }
}

// ---------------------------------------------------------------------------
// PyProgressEvent
// ---------------------------------------------------------------------------

/// Emitted by Pipeline and Workflow runners to surface progress to callers.
#[gen_stub_pyclass]
#[pyclass(name = "ProgressEvent", from_py_object)]
#[derive(Clone)]
pub struct PyProgressEvent {
    pub(crate) inner: ProgressEvent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProgressEvent {
    /// Build a progress event.
    #[new]
    #[pyo3(signature = (
        *,
        kind,
        current,
        total=None,
        percent=None,
        label,
        run_id=None,
    ))]
    fn new(
        kind: PyProgressKind,
        current: u32,
        total: Option<u32>,
        percent: Option<f32>,
        label: String,
        run_id: Option<String>,
    ) -> PyResult<Self> {
        let parsed_run_id = match run_id {
            Some(s) => Uuid::parse_str(&s).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid run_id: {e}"))
            })?,
            None => Uuid::new_v4(),
        };
        Ok(Self {
            inner: ProgressEvent {
                kind: kind.into(),
                current,
                total,
                percent,
                label,
                run_id: parsed_run_id,
            },
        })
    }

    #[getter]
    fn kind(&self) -> PyProgressKind {
        self.inner.kind.clone().into()
    }
    #[getter]
    fn current(&self) -> u32 {
        self.inner.current
    }
    #[getter]
    fn total(&self) -> Option<u32> {
        self.inner.total
    }
    #[getter]
    fn percent(&self) -> Option<f32> {
        self.inner.percent
    }
    #[getter]
    fn label(&self) -> &str {
        &self.inner.label
    }
    #[getter]
    fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "ProgressEvent(kind={:?}, current={}, total={:?}, label={:?})",
            self.inner.kind, self.inner.current, self.inner.total, self.inner.label,
        )
    }
}

impl From<ProgressEvent> for PyProgressEvent {
    fn from(inner: ProgressEvent) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyProgressSnapshot
// ---------------------------------------------------------------------------

/// A best-effort, latch-style snapshot of pipeline progress.
///
/// Returned by [`PipelineHandler.progress`] without taking any locks; values
/// may briefly be one stage stale.
#[gen_stub_pyclass]
#[pyclass(name = "ProgressSnapshot", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyProgressSnapshot {
    pub(crate) inner: ProgressSnapshot,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProgressSnapshot {
    /// 1-based index of the stage currently executing (or just completed).
    /// `0` before the first stage starts.
    #[getter]
    fn current_stage_index(&self) -> u32 {
        self.inner.current_stage_index
    }

    /// Total number of stages declared on the pipeline.
    #[getter]
    fn total_stages(&self) -> u32 {
        self.inner.total_stages
    }

    /// Progress as a percentage in `0.0..=100.0`.
    #[getter]
    fn percent(&self) -> f32 {
        self.inner.percent
    }

    /// Name of the current stage when available; `None` from the simple
    /// atomic-index implementation today.
    #[getter]
    fn current_stage_name(&self) -> Option<String> {
        self.inner.current_stage_name.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ProgressSnapshot(current_stage_index={}, total_stages={}, percent={:.2})",
            self.inner.current_stage_index, self.inner.total_stages, self.inner.percent,
        )
    }
}

impl From<ProgressSnapshot> for PyProgressSnapshot {
    fn from(inner: ProgressSnapshot) -> Self {
        Self { inner }
    }
}
