//! Streaming pipeline that bridges an incoming PCM chunk stream to
//! [`StreamingTranscript`] emissions via Silero VAD chunking and the
//! chunked candle Whisper decoder.
//!
//! # Sample-rate contract
//!
//! All input chunks **must** be 16 kHz mono f32 PCM in `[-1.0, 1.0]`.
//! Silero VAD v4 and Whisper both fix this sample rate; resampling has
//! to happen upstream (typically in `blazen-audio`'s capture layer).
//!
//! # Pipeline shape
//!
//! 1. Incoming `Vec<f32>` chunks land in a leftover buffer and are
//!    re-cut into exactly 512-sample VAD frames
//!    ([`SileroVad::frame_size`]).
//! 2. Each frame goes through [`SileroVad::feed`], yielding a
//!    [`VadFrame`] with utterance-boundary flags.
//! 3. A per-utterance audio buffer accumulates the raw PCM (NOT the
//!    pre-framed slices — we keep the full continuous signal so the
//!    decoder sees the natural window).
//! 4. When the utterance buffer reaches `chunk_seconds * 16_000`
//!    samples *or* `utterance_end` fires, we hand the buffered window
//!    to [`ChunkedWhisperDecoder::decode_chunk`].
//! 5. Each resulting [`DecodedChunk`] is split into a partial
//!    ([`StreamingTranscript::is_final == false`]) emission and a final
//!    emission, both sent on the output channel if non-empty.
//! 6. On input-stream EOS we call [`ChunkedWhisperDecoder::finalize`]
//!    to flush the pending tail as a terminal final emission.
//!
//! # Language hint
//!
//! [`SttBackend::stream`] accepts a per-call `language` hint. The
//! underlying [`ChunkedWhisperDecoder`] only exposes a config-level
//! language (set at construction). When the per-call hint differs from
//! the configured one this implementation **ignores** the per-call
//! value — the decoder is shared across stream invocations and we do
//! not mutate its config mid-stream. Callers that need per-stream
//! language control should construct a fresh
//! [`WhisperStreamingBackend`] per language.

#![cfg(feature = "whisper-streaming")]

use std::pin::Pin;

use futures_core::Stream;
use futures_util::StreamExt;
use tokio::sync::OwnedMutexGuard;
use tokio::sync::mpsc;

use super::decoder::{ChunkedWhisperDecoder, DecodedChunk};
use super::vad::{SileroVad, VadFrame};
use crate::error::SttError;
use crate::traits::StreamingTranscript;

/// Whisper's (and Silero VAD v4's) fixed sample rate.
const SAMPLE_RATE_HZ: usize = 16_000;

/// Bounded channel capacity for the output stream. 32 emissions of
/// back-pressure is plenty for a transcript at human-speech rates
/// (roughly 1-3 emissions/second from Whisper).
const OUTPUT_CHANNEL_CAPACITY: usize = 32;

/// Drive one whisper-streaming pipeline run.
///
/// Returns a boxed `Stream` of [`StreamingTranscript`] results. The
/// pipeline holds `vad_guard` and `decoder_guard` for its entire
/// lifetime — concurrent calls to [`super::WhisperStreamingBackend::stream`]
/// will serialise on the underlying [`tokio::sync::Mutex`]es, which is
/// the intended behaviour: VAD and decoder both carry per-stream state
/// (LSTM hidden, `pending_tail`) that must not interleave.
pub(super) fn spawn_pipeline(
    audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>,
    vad_guard: OwnedMutexGuard<Option<SileroVad>>,
    decoder_guard: OwnedMutexGuard<Option<ChunkedWhisperDecoder>>,
    chunk_seconds: f32,
) -> Pin<Box<dyn Stream<Item = Result<StreamingTranscript, SttError>> + Send>> {
    let (tx, rx) = mpsc::channel::<Result<StreamingTranscript, SttError>>(OUTPUT_CHANNEL_CAPACITY);

    tokio::spawn(async move {
        // Move the guards into the task and reborrow inner mutably.
        // We hold the guards for the entire stream lifetime.
        let mut vad_guard = vad_guard;
        let mut decoder_guard = decoder_guard;
        let Some(vad) = vad_guard.as_mut() else {
            let _ = tx
                .send(Err(SttError::Transcription(
                    "whisper-streaming pipeline: vad slot empty after lazy init".into(),
                )))
                .await;
            return;
        };
        let Some(decoder) = decoder_guard.as_mut() else {
            let _ = tx
                .send(Err(SttError::Transcription(
                    "whisper-streaming pipeline: decoder slot empty after lazy init".into(),
                )))
                .await;
            return;
        };

        if let Err(err) = run_pipeline(audio, vad, decoder, chunk_seconds, &tx).await {
            // Best-effort propagation; if rx is closed we just drop.
            let _ = tx.send(Err(err)).await;
        }
    });

    receiver_into_stream(rx)
}

/// Convert an `mpsc::Receiver` into a `Pin<Box<dyn Stream>>` without
/// pulling in `tokio-stream`. Uses `futures_util::stream::unfold`.
fn receiver_into_stream<T: Send + 'static>(
    rx: mpsc::Receiver<T>,
) -> Pin<Box<dyn Stream<Item = T> + Send>> {
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|v| (v, rx))
    }))
}

/// Pull frames from `audio`, feed VAD, accumulate utterance audio,
/// and decode chunks. Returns `Err` on the first fatal error
/// (forwarded to the channel by the caller).
async fn run_pipeline(
    mut audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>,
    vad: &mut SileroVad,
    decoder: &mut ChunkedWhisperDecoder,
    chunk_seconds: f32,
    tx: &mpsc::Sender<Result<StreamingTranscript, SttError>>,
) -> Result<(), SttError> {
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    let chunk_target_samples = (chunk_seconds * SAMPLE_RATE_HZ as f32) as usize;
    let frame_size = SileroVad::frame_size();

    // Leftover PCM that didn't fill a full VAD frame on the previous
    // input item. Cleared as soon as it grows past `frame_size`.
    let mut leftover: Vec<f32> = Vec::with_capacity(frame_size * 2);
    // Buffer of PCM samples that belong to the *current* utterance.
    // Reset to empty after each `decode_chunk` call (utterance done) or
    // after a `chunk_target_samples` flush.
    let mut utterance_buf: Vec<f32> = Vec::with_capacity(chunk_target_samples);
    // Are we currently inside an utterance? Flipped by VAD
    // `utterance_start`/`utterance_end` events. Until the first start
    // we drop incoming samples.
    let mut in_utterance = false;
    // 0-based index of the next chunk we'll hand to the decoder.
    let mut chunk_index: usize = 0;

    while let Some(pcm) = audio.next().await {
        if pcm.is_empty() {
            continue;
        }

        leftover.extend_from_slice(&pcm);

        // Peel off full VAD frames.
        let mut offset = 0;
        while leftover.len() - offset >= frame_size {
            let frame = &leftover[offset..offset + frame_size];
            let vad_frame: VadFrame = vad.feed(frame)?;

            if vad_frame.utterance_start {
                in_utterance = true;
            }

            if in_utterance {
                utterance_buf.extend_from_slice(frame);
            }

            // Did this frame's utterance_end flip us out?
            if vad_frame.utterance_end && in_utterance {
                in_utterance = false;
                if !utterance_buf.is_empty() {
                    decode_and_emit(
                        &mut utterance_buf,
                        decoder,
                        chunk_index,
                        chunk_target_samples,
                        tx,
                    )
                    .await?;
                    chunk_index += 1;
                }
            } else if utterance_buf.len() >= chunk_target_samples {
                // Long utterance: flush a full window without ending
                // it. The decoder's `LocalAgreement` logic stitches the
                // overlapping chunks back together.
                decode_and_emit(
                    &mut utterance_buf,
                    decoder,
                    chunk_index,
                    chunk_target_samples,
                    tx,
                )
                .await?;
                chunk_index += 1;
            }

            offset += frame_size;

            // Caller closed the rx? Short-circuit.
            if tx.is_closed() {
                return Ok(());
            }
        }

        // Drop consumed prefix.
        leftover.drain(..offset);
    }

    // End of input. Flush any remaining utterance audio.
    if !utterance_buf.is_empty() {
        decode_and_emit(
            &mut utterance_buf,
            decoder,
            chunk_index,
            chunk_target_samples,
            tx,
        )
        .await?;
    }

    // Promote any pending partial to final. Always emit this terminal
    // record (even when `text` is empty) so consumers get a definitive
    // `is_final` stream-end marker — e.g. a non-speech input the VAD never
    // segmented still yields exactly one final emission at EOS.
    let final_tail = decoder.finalize();
    let _ = tx.send(Ok(final_tail)).await;

    Ok(())
}

/// Pad `utterance_buf` up to (or down to, by truncation) the decoder's
/// expected chunk length, call `decode_chunk`, send partial + final
/// emissions on `tx`, then clear the buffer for the next utterance/
/// flush. The buffer is consumed in-place to avoid an extra copy.
async fn decode_and_emit(
    utterance_buf: &mut Vec<f32>,
    decoder: &mut ChunkedWhisperDecoder,
    chunk_index: usize,
    chunk_target_samples: usize,
    tx: &mpsc::Sender<Result<StreamingTranscript, SttError>>,
) -> Result<(), SttError> {
    // The decoder validates audio length against `chunk_seconds *
    // SAMPLE_RATE` with a 10% tolerance. Pad short utterances with
    // silence and truncate long ones at the boundary — long utterances
    // are already chunked by the caller, but the residual may
    // overshoot slightly because we only check after each VAD frame.
    if utterance_buf.len() < chunk_target_samples {
        utterance_buf.resize(chunk_target_samples, 0.0);
    } else if utterance_buf.len() > chunk_target_samples {
        utterance_buf.truncate(chunk_target_samples);
    }

    let chunk_out: DecodedChunk = decoder.decode_chunk(utterance_buf, chunk_index).await?;

    utterance_buf.clear();

    if !chunk_out.partial_text.is_empty() {
        let partial = StreamingTranscript {
            text: chunk_out.partial_text,
            is_final: false,
            confidence: None,
            latency_seconds: Some(chunk_out.latency_seconds),
        };
        if tx.send(Ok(partial)).await.is_err() {
            return Ok(());
        }
    }
    if !chunk_out.final_text.is_empty() {
        let finalised = StreamingTranscript {
            text: chunk_out.final_text,
            is_final: true,
            confidence: None,
            latency_seconds: Some(chunk_out.latency_seconds),
        };
        if tx.send(Ok(finalised)).await.is_err() {
            return Ok(());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
