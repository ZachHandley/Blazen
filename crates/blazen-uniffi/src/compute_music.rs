//! Music + sound-effect generation wire-format records and streaming sink
//! for the UniFFI bindings.
//!
//! The central capability-erased `MusicModel` opaque handle and its
//! per-backend factories (`new_musicgen_model`, `new_stable_audio_model`,
//! `new_audiogen_model`, `new_fal_music_model`) have moved to the
//! per-engine concrete provider classes under [`crate::concrete`] (e.g.
//! `MusicGenProvider`, `StableAudioProvider`, `AudioGenProvider`,
//! `FalMusicProvider`). This module retains only the wire-format records
//! ([`MusicChunk`], [`MusicResult`]), the foreign-implementable
//! [`MusicStreamSink`] trait, and the [`drive_music_stream`] helper used
//! by the concrete providers to pump a chunk stream into a sink.
//!
//! ## Wire-format shape
//!
//! Streaming chunks carry raw 32-bit float PCM samples at the backend's
//! sample rate. UniFFI marshals `Vec<f32>` natively across Go / Swift /
//! Kotlin / Ruby, so we avoid per-chunk base64 encode / decode and let the
//! foreign side build the playback buffer directly.

use std::pin::Pin;
use std::sync::Arc;

use futures_util::{Stream, StreamExt};

use crate::errors::{BlazenError, BlazenResult};
use crate::streaming::clone_error;

// ---------------------------------------------------------------------------
// Wire-format records
// ---------------------------------------------------------------------------

/// One emission from a streaming music backend.
///
/// `samples` is 32-bit float PCM in `[-1.0, 1.0]` at the backend's expected
/// output sample rate (the same `sample_rate` field on the
/// [`MusicResult`] returned by the non-streaming generate calls).
///
/// `is_final` is `true` for the final chunk of a generation call;
/// implementations should treat it as a UI hint rather than the
/// authoritative completion signal — the sink's `on_done` callback is the
/// canonical end-of-stream marker.
///
/// `latency_seconds`, when present, is the measured latency from the
/// stream's call-start to the moment this chunk was produced — handy for
/// surfacing first-token-latency metrics through the binding.
#[derive(Debug, Clone, uniffi::Record, serde::Serialize, serde::Deserialize)]
pub struct MusicChunk {
    /// 32-bit float PCM samples in `[-1, 1]` at the backend's sample rate.
    pub samples: Vec<f32>,
    /// `true` on the final emitted chunk; otherwise `false`.
    pub is_final: bool,
    /// Optional per-chunk latency from call-start in seconds.
    pub latency_seconds: Option<f32>,
}

#[cfg(any(
    feature = "audio-music-musicgen",
    feature = "audio-music-stable-audio",
    feature = "audio-music-audiogen"
))]
impl From<blazen_llm::MusicChunk> for MusicChunk {
    fn from(chunk: blazen_llm::MusicChunk) -> Self {
        Self {
            samples: chunk.samples,
            is_final: chunk.is_final,
            latency_seconds: chunk.latency_seconds,
        }
    }
}

/// A fully-rendered music / SFX result.
///
/// `bytes` carries the encoded audio (typically a WAV container for the
/// native backends; whatever the cloud provider returned for fal.ai). The
/// non-empty `url` field signals a URL-only response (e.g. fal.ai returning
/// a CDN link without inlining bytes); `bytes` will be empty in that case.
/// Callers should pick whichever payload is present.
#[derive(Debug, Clone, uniffi::Record)]
pub struct MusicResult {
    /// Encoded audio bytes. Empty when the upstream provider only returned
    /// a URL.
    pub bytes: Vec<u8>,
    /// IANA MIME type of `bytes` (e.g. `"audio/wav"`, `"audio/mpeg"`).
    pub mime_type: String,
    /// Sample rate in Hz. Zero when the upstream provider didn't report
    /// one.
    pub sample_rate: u32,
    /// Channel count (1 = mono, 2 = stereo). Zero when the upstream
    /// provider didn't report it.
    pub channels: u32,
    /// Duration of the clip in seconds. Zero when the upstream provider
    /// didn't report a duration.
    pub duration_seconds: f32,
    /// URL of the audio asset when the upstream provider only returned a
    /// link. Empty string for inline-bytes results.
    pub url: String,
}

// ---------------------------------------------------------------------------
// Streaming sink + pump
// ---------------------------------------------------------------------------

/// Sink for streaming music / SFX output, implemented in foreign code.
///
/// Symmetric to [`crate::streaming::CompletionStreamSink`]: the streaming
/// engine calls [`on_chunk`](Self::on_chunk) for each emitted chunk, then
/// exactly one of [`on_done`](Self::on_done) or [`on_error`](Self::on_error).
/// Implementations should treat the terminal callbacks as cleanup hooks
/// (close channels, complete async iterators, signal flow completion, ...).
#[uniffi::export(with_foreign)]
#[async_trait::async_trait]
pub trait MusicStreamSink: Send + Sync {
    /// Receive a single chunk from the streaming response.
    ///
    /// Returning an `Err` aborts the stream — the engine delivers the error
    /// via [`on_error`](Self::on_error) and stops dispatching further
    /// chunks.
    async fn on_chunk(&self, chunk: MusicChunk) -> BlazenResult<()>;

    /// Receive the terminal completion signal. Called exactly once at the
    /// end of a successful stream.
    async fn on_done(&self) -> BlazenResult<()>;

    /// Receive a fatal error from the stream. Called exactly once when the
    /// stream fails midway (or fails to start at all).
    async fn on_error(&self, cause: BlazenError) -> BlazenResult<()>;
}

pub(crate) async fn drive_music_stream(
    mut stream: Pin<Box<dyn Stream<Item = Result<MusicChunk, BlazenError>> + Send>>,
    sink: Arc<dyn MusicStreamSink>,
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
    fn music_chunk_roundtrips_through_serde_json() {
        let chunk = MusicChunk {
            samples: vec![0.0, 0.5, -0.5, 1.0],
            is_final: true,
            latency_seconds: Some(0.125),
        };
        let json = serde_json::to_string(&chunk).expect("serialize");
        let back: MusicChunk = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.samples, chunk.samples);
        assert_eq!(back.is_final, chunk.is_final);
        assert_eq!(back.latency_seconds, chunk.latency_seconds);
    }
}
