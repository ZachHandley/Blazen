//! Streaming orchestration for the faster-whisper backend (Wave F.2.4).
//!
//! # Why window-based and not "true" streaming
//!
//! `ct2rs` (the Rust binding over `CTranslate2`) exposes
//! [`ct2rs::Whisper::generate`] as a single-shot blocking call: hand it
//! the full sample buffer, get back the concatenated chunk list. There
//! is **no callback hook** for partial results, and the lower-level
//! `ct2rs::sys::Whisper` surface (`encode` / `generate` / `align`) is
//! also single-shot per call — it just exposes the encoded mel features
//! to the caller. Upstream `faster-whisper` (the Python project) handles
//! streaming the same way we do here: window the input on the caller
//! side, dispatch one [`generate`][ct2rs::Whisper::generate] per window,
//! glue the per-window transcripts together. We mirror that pattern.
//!
//! # Window geometry
//!
//! - Window size: **30 s @ 16 kHz mono = 480 000 f32 samples**
//!   ([`STREAM_WINDOW_SAMPLES`]).
//! - **No overlap.** Each window is independent; the boundary between
//!   adjacent windows may clip a word in the middle. Overlapping windows
//!   (with seam-deduplication via `LocalAgreement` or similar) is
//!   reserved for a future wave; for the first cut we keep the logic
//!   trivial.
//! - **Final partial flush**: when the input stream ends with a partial
//!   window in the buffer (`0 < buf.len() < STREAM_WINDOW_SAMPLES`), we
//!   issue one last [`generate`][ct2rs::Whisper::generate] on the partial
//!   buffer and emit its transcript. Empty trailing buffers are dropped.
//!
//! # Per-emission shape
//!
//! Every emission is marked [`StreamingTranscript::is_final == true`].
//! ct2rs returns a *complete* transcript for the window we hand it; we
//! don't synthesise interim guesses on top of that. The `latency_seconds`
//! field carries the cumulative start-offset of the window (in seconds
//! from the start of the input stream) so downstream consumers can stitch
//! the transcript timeline back together.
//!
//! # Threading
//!
//! [`ct2rs::Whisper::generate`] is a blocking C++ FFI call; we route each
//! per-window dispatch through [`tokio::task::spawn_blocking`] so the
//! tokio worker pool doesn't stall. The output stream is fed through an
//! [`mpsc`] channel (same convention as `whisper_streaming`'s pipeline)
//! so back-pressure naturally propagates from the consumer to the
//! per-window dispatcher.

#![cfg(feature = "faster-whisper")]

use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use futures_util::StreamExt;
use tokio::sync::mpsc;

use super::decoder::{FasterWhisperDecoder, TranscribeOptions};
use crate::SttError;
use crate::traits::StreamingTranscript;

/// 30 s at 16 kHz mono — Whisper's native window length. Per-window
/// dispatches to [`ct2rs::Whisper::generate`] use exactly this many
/// samples; final-tail partial windows may be shorter.
pub(super) const STREAM_WINDOW_SAMPLES: usize = 30 * 16_000;

/// Whisper's fixed sample rate. Used to convert window offsets in samples
/// into seconds for the [`StreamingTranscript::latency_seconds`] field.
const SAMPLE_RATE_HZ: usize = 16_000;

/// Bounded channel capacity. 4 windows in flight (= 120 s of audio worth
/// of back-pressure headroom) is plenty for real-time use — the consumer
/// is expected to drain the stream as transcripts arrive, not buffer
/// entire conversations.
const OUTPUT_CHANNEL_CAPACITY: usize = 4;

/// Drive one window-based streaming run. See the module docs for the
/// geometry and per-emission contract.
///
/// `decoder` is shared across windows: each [`tokio::task::spawn_blocking`]
/// dispatch clones the [`Arc`] and runs [`FasterWhisperDecoder::transcribe`]
/// on the next window's samples.
pub(super) fn streaming_transcribe(
    decoder: Arc<FasterWhisperDecoder>,
    audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>,
    language: Option<String>,
) -> Pin<Box<dyn Stream<Item = Result<StreamingTranscript, SttError>> + Send>> {
    let (tx, rx) = mpsc::channel::<Result<StreamingTranscript, SttError>>(OUTPUT_CHANNEL_CAPACITY);

    tokio::spawn(async move {
        if let Err(err) = run_pipeline(decoder, audio, language.as_deref(), &tx).await {
            // Best-effort propagation; if the consumer dropped rx we just
            // drop the final error too.
            let _ = tx.send(Err(err)).await;
        }
    });

    receiver_into_stream(rx)
}

/// Pull `Vec<f32>` chunks from `audio`, buffer until the buffer holds at
/// least one full window, dispatch the window through `spawn_blocking`,
/// and forward each window's [`StreamingTranscript`] to `tx`. On EOS,
/// flush any non-empty tail as a final partial-window emission.
async fn run_pipeline(
    decoder: Arc<FasterWhisperDecoder>,
    mut audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>,
    language: Option<&str>,
    tx: &mpsc::Sender<Result<StreamingTranscript, SttError>>,
) -> Result<(), SttError> {
    let mut buffer: Vec<f32> = Vec::with_capacity(STREAM_WINDOW_SAMPLES * 2);
    // Cumulative offset (in samples) from the start of the input stream
    // to the start of the *next* window we'll dispatch. Used to populate
    // [`StreamingTranscript::latency_seconds`].
    let mut window_start_samples: usize = 0;

    while let Some(pcm) = audio.next().await {
        if pcm.is_empty() {
            continue;
        }
        buffer.extend_from_slice(&pcm);

        while let Some(window) = window_drain(&mut buffer) {
            let start_seconds = samples_to_seconds(window_start_samples);
            window_start_samples = window_start_samples.saturating_add(window.len());

            let transcript = dispatch_window(&decoder, window, language, start_seconds).await?;
            if tx.send(Ok(transcript)).await.is_err() {
                // Consumer dropped the receiver — stop pulling input.
                return Ok(());
            }
        }
    }

    // EOS: flush the final partial window (if non-empty). We hand any
    // remaining samples to `generate` even if they're shorter than 30 s —
    // ct2rs/CTranslate2 internally pads short inputs up to the encoder's
    // expected length.
    if !buffer.is_empty() {
        let start_seconds = samples_to_seconds(window_start_samples);
        let tail = std::mem::take(&mut buffer);
        let transcript = dispatch_window(&decoder, tail, language, start_seconds).await?;
        let _ = tx.send(Ok(transcript)).await;
    }

    Ok(())
}

/// Drain exactly one full window off the front of `buf` if available.
/// Returns `None` when fewer than [`STREAM_WINDOW_SAMPLES`] samples are
/// buffered (the caller keeps accumulating).
///
/// Exposed at `pub(super)` so the buffering policy is unit-testable
/// without requiring a downloaded `CTranslate2` model.
pub(super) fn window_drain(buf: &mut Vec<f32>) -> Option<Vec<f32>> {
    if buf.len() < STREAM_WINDOW_SAMPLES {
        return None;
    }
    // `Vec::drain` returns an iterator; collect into the owned window.
    let window: Vec<f32> = buf.drain(..STREAM_WINDOW_SAMPLES).collect();
    Some(window)
}

/// Dispatch one window through [`tokio::task::spawn_blocking`] and wrap
/// the resulting transcript into a [`StreamingTranscript`].
async fn dispatch_window(
    decoder: &Arc<FasterWhisperDecoder>,
    window: Vec<f32>,
    language: Option<&str>,
    start_seconds: f32,
) -> Result<StreamingTranscript, SttError> {
    let handle = Arc::clone(decoder);
    let lang_owned = language.map(str::to_owned);
    let transcript = tokio::task::spawn_blocking(move || {
        handle.transcribe(
            &window,
            lang_owned.as_deref(),
            &TranscribeOptions {
                // Segment timestamps are useful but not free; for the
                // streaming first-cut we ask for them so the per-window
                // text mirrors what `transcribe()` returns.
                want_segment_timestamps: true,
                ..TranscribeOptions::default()
            },
        )
    })
    .await
    .map_err(|e| SttError::Transcription(format!("join error: {e}")))?
    .map_err(SttError::from)?;

    Ok(StreamingTranscript {
        text: transcript.text,
        // Each per-window dispatch produces a complete transcript for the
        // audio it saw; there's no interim revision later. Mark final.
        is_final: true,
        confidence: None,
        // Carry the cumulative offset (seconds from stream start) of this
        // window's *start*. Lets downstream consumers reconstruct a
        // sample-aligned transcript timeline across windows.
        latency_seconds: Some(start_seconds),
    })
}

/// Convert a sample count at 16 kHz into a wall-clock offset in seconds.
///
/// Precision-loss is intentional — [`StreamingTranscript::latency_seconds`]
/// is an `f32` and a 30 s window's worth of samples (480 000) fits losslessly
/// in `f32`'s 24-bit mantissa.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn samples_to_seconds(samples: usize) -> f32 {
    (samples as f64 / SAMPLE_RATE_HZ as f64) as f32
}

/// Convert an [`mpsc::Receiver`] into a `Pin<Box<dyn Stream>>` without
/// pulling in `tokio-stream`. Same helper shape as the
/// `whisper_streaming` pipeline.
fn receiver_into_stream<T: Send + 'static>(
    rx: mpsc::Receiver<T>,
) -> Pin<Box<dyn Stream<Item = T> + Send>> {
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|v| (v, rx))
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_window_size_is_30_seconds_at_16khz() {
        // Wave F.2.4 commits to a 30 s @ 16 kHz mono window — that's
        // 480 000 f32 samples per ct2rs dispatch. Whisper's encoder is
        // architecturally fixed at this length; changing it would force
        // ct2rs to repad on every call.
        assert_eq!(STREAM_WINDOW_SAMPLES, 480_000);
        assert_eq!(STREAM_WINDOW_SAMPLES, 30 * SAMPLE_RATE_HZ);
    }

    #[test]
    fn window_drain_returns_none_when_buffer_under_window() {
        let mut buf: Vec<f32> = vec![0.0; STREAM_WINDOW_SAMPLES - 1];
        assert!(window_drain(&mut buf).is_none());
        // Buffer must be left intact when no window is drained.
        assert_eq!(buf.len(), STREAM_WINDOW_SAMPLES - 1);
    }

    #[test]
    fn window_drain_returns_exactly_one_window_when_buffer_at_threshold() {
        let mut buf: Vec<f32> = vec![0.0; STREAM_WINDOW_SAMPLES];
        let window = window_drain(&mut buf).expect("drained one window");
        assert_eq!(window.len(), STREAM_WINDOW_SAMPLES);
        assert!(buf.is_empty(), "buffer must be empty after exact drain");
    }

    #[test]
    fn window_drain_leaves_remainder_in_buffer() {
        // Two full windows + a 7-sample residual.
        let mut buf: Vec<f32> = vec![0.5; STREAM_WINDOW_SAMPLES * 2 + 7];
        let first = window_drain(&mut buf).expect("first window");
        assert_eq!(first.len(), STREAM_WINDOW_SAMPLES);
        assert_eq!(buf.len(), STREAM_WINDOW_SAMPLES + 7);
        let second = window_drain(&mut buf).expect("second window");
        assert_eq!(second.len(), STREAM_WINDOW_SAMPLES);
        assert_eq!(buf.len(), 7);
        // No third window — only 7 samples remain.
        assert!(window_drain(&mut buf).is_none());
        assert_eq!(buf.len(), 7);
    }

    #[test]
    fn samples_to_seconds_uses_16khz_rate() {
        assert!((samples_to_seconds(0) - 0.0).abs() < f32::EPSILON);
        assert!((samples_to_seconds(16_000) - 1.0).abs() < f32::EPSILON);
        assert!((samples_to_seconds(STREAM_WINDOW_SAMPLES) - 30.0).abs() < f32::EPSILON);
        // 1.5 s = 24 000 samples.
        assert!((samples_to_seconds(24_000) - 1.5).abs() < f32::EPSILON);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn receiver_into_stream_terminates_on_sender_drop() {
        let (tx, rx) = mpsc::channel::<u32>(4);
        drop(tx);
        let mut s = receiver_into_stream(rx);
        assert!(s.next().await.is_none(), "stream must end when tx drops");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn receiver_into_stream_forwards_items_in_order() {
        let (tx, rx) = mpsc::channel::<u32>(4);
        tx.send(1).await.unwrap();
        tx.send(2).await.unwrap();
        tx.send(3).await.unwrap();
        drop(tx);
        let collected: Vec<u32> = receiver_into_stream(rx).collect().await;
        assert_eq!(collected, vec![1, 2, 3]);
    }
}
