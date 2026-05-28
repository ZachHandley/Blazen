//! Voice-conversion (RVC and friends) DTOs and streaming sink for the
//! UniFFI bindings.
//!
//! The central capability-erased `VcModel` opaque handle and its
//! per-backend factory (`new_rvc_model`) have moved to the per-engine
//! concrete provider classes under [`crate::concrete`] (e.g.
//! `RvcProvider`). This module retains only the wire-format records
//! ([`TargetVoice`], [`VcChunk`], [`VcResult`]), the
//! foreign-implementable [`VcStreamSink`] trait, and the
//! [`drive_vc_stream`] helper used by the concrete providers to pump a
//! chunk stream into a sink.
//!
//! ## Wire-format shape
//!
//! [`VcResult::bytes`] is a self-describing WAV (RIFF/`fmt `/`data`)
//! container holding 16-bit signed little-endian PCM samples at the
//! target voice's native sample rate, so callers can write the buffer
//! straight to disk or hand it to a decoder. Streaming uses raw 32-bit
//! float PCM ([`VcChunk::samples`]) at the target voice's native sample
//! rate — UniFFI marshals `Vec<f32>` natively across Go / Swift /
//! Kotlin / Ruby, so the foreign side can build the playback buffer
//! directly.

use std::pin::Pin;
use std::sync::Arc;

use futures_util::{Stream, StreamExt};

use crate::errors::{BlazenError, BlazenResult};
use crate::streaming::clone_error;

// ---------------------------------------------------------------------------
// Wire-format records
// ---------------------------------------------------------------------------

/// A registered target speaker that a voice-conversion provider can
/// render source audio into.
///
/// Mirrors [`blazen_llm::TargetVoice`] (when the `audio-vc` feature is on)
/// 1:1 across the FFI boundary so foreign code sees a stable record shape
/// regardless of whether the underlying engine is the native RVC backend
/// or a cloud-side provider added later.
#[derive(Debug, Clone, uniffi::Record, serde::Serialize, serde::Deserialize)]
pub struct TargetVoice {
    /// Backend-scoped identifier passed to the concrete provider's
    /// convert / register entry points.
    pub id: String,
    /// Optional human-readable display name. `None` when the backend did
    /// not record one.
    pub label: Option<String>,
    /// Native sample rate (Hz) the backend renders this voice at.
    pub sample_rate_hz: u32,
}

#[cfg(feature = "audio-vc")]
impl From<blazen_llm::TargetVoice> for TargetVoice {
    fn from(voice: blazen_llm::TargetVoice) -> Self {
        Self {
            id: voice.id,
            label: voice.label,
            sample_rate_hz: voice.sample_rate_hz,
        }
    }
}

/// One emission from a streaming voice-conversion call.
///
/// `samples` is 32-bit float PCM in `[-1.0, 1.0]` at the target voice's
/// native sample rate (see [`TargetVoice::sample_rate_hz`]).
///
/// `is_final` is purely an advisory hint — the sink's `on_done` callback
/// is the canonical end-of-stream signal, matching the contract used by
/// [`crate::compute_music::MusicChunk`].
///
/// `latency_seconds`, when present, is the measured latency from the
/// stream's call-start to the moment this chunk was produced — handy for
/// surfacing first-token-latency metrics through the binding.
#[derive(Debug, Clone, uniffi::Record, serde::Serialize, serde::Deserialize)]
pub struct VcChunk {
    /// 32-bit float PCM samples in `[-1, 1]` at the voice's native sample
    /// rate.
    pub samples: Vec<f32>,
    /// `true` on the final emitted chunk; otherwise `false`. Always
    /// `false` for the RVC backend today (end-of-stream is signalled by
    /// the sink's `on_done` callback).
    pub is_final: bool,
    /// Optional per-chunk latency from call-start in seconds.
    pub latency_seconds: Option<f32>,
}

/// A fully-rendered voice-conversion result.
///
/// `bytes` carries a complete WAV (RIFF/`fmt `/`data`) container holding
/// 16-bit signed little-endian PCM samples at the target voice's native
/// sample rate. `sample_rate` echoes that rate for convenience so callers
/// don't have to re-parse the WAV header.
#[derive(Debug, Clone, uniffi::Record)]
pub struct VcResult {
    /// Encoded audio bytes (WAV container, 16-bit signed PCM).
    pub bytes: Vec<u8>,
    /// IANA MIME type of `bytes` (always `"audio/wav"` for the native
    /// backends shipped today).
    pub mime_type: String,
    /// Sample rate in Hz, taken from the target voice's
    /// [`TargetVoice::sample_rate_hz`].
    pub sample_rate: u32,
    /// Duration of the clip in seconds. Zero when the backend did not
    /// report one (no extra WAV header parsing happens here).
    pub duration_seconds: f32,
}

// ---------------------------------------------------------------------------
// Streaming sink + pump
// ---------------------------------------------------------------------------

/// Sink for streaming voice-conversion output, implemented in foreign
/// code.
///
/// Symmetric to [`crate::compute_music::MusicStreamSink`] and
/// [`crate::streaming::CompletionStreamSink`]: the streaming engine calls
/// [`on_chunk`](Self::on_chunk) for each emitted chunk, then exactly one
/// of [`on_done`](Self::on_done) or [`on_error`](Self::on_error).
/// Implementations should treat the terminal callbacks as cleanup hooks
/// (close channels, complete async iterators, signal flow completion).
#[uniffi::export(with_foreign)]
#[async_trait::async_trait]
pub trait VcStreamSink: Send + Sync {
    /// Receive a single chunk from the streaming response.
    ///
    /// Returning an `Err` aborts the stream — the engine delivers the
    /// error via [`on_error`](Self::on_error) and stops dispatching
    /// further chunks.
    async fn on_chunk(&self, chunk: VcChunk) -> BlazenResult<()>;

    /// Receive the terminal completion signal. Called exactly once at the
    /// end of a successful stream.
    async fn on_done(&self) -> BlazenResult<()>;

    /// Receive a fatal error from the stream. Called exactly once when
    /// the stream fails midway (or fails to start at all).
    async fn on_error(&self, cause: BlazenError) -> BlazenResult<()>;
}

pub(crate) async fn drive_vc_stream(
    mut stream: Pin<Box<dyn Stream<Item = Result<VcChunk, BlazenError>> + Send>>,
    sink: Arc<dyn VcStreamSink>,
) -> BlazenResult<()> {
    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => {
                if let Err(sink_err) = sink.on_chunk(chunk).await {
                    let _ = sink.on_error(clone_error(&sink_err)).await;
                    return Ok(());
                }
            }
            Err(err) => {
                let _ = sink.on_error(err).await;
                return Ok(());
            }
        }
    }
    if let Err(sink_err) = sink.on_done().await {
        let _ = sink.on_error(clone_error(&sink_err)).await;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vc_chunk_roundtrips_through_serde_json() {
        let chunk = VcChunk {
            samples: vec![0.0, 0.5, -0.5, 1.0],
            is_final: false,
            latency_seconds: Some(0.125),
        };
        let json = serde_json::to_string(&chunk).expect("serialize");
        let back: VcChunk = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.samples, chunk.samples);
        assert_eq!(back.is_final, chunk.is_final);
        assert_eq!(back.latency_seconds, chunk.latency_seconds);
    }

    #[test]
    fn target_voice_roundtrips_through_serde_json() {
        let voice = TargetVoice {
            id: "speaker-01".into(),
            label: Some("Alice".into()),
            sample_rate_hz: 40_000,
        };
        let json = serde_json::to_string(&voice).expect("serialize");
        let back: TargetVoice = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.id, "speaker-01");
        assert_eq!(back.label.as_deref(), Some("Alice"));
        assert_eq!(back.sample_rate_hz, 40_000);
    }
}
