package dev.zorpx.blazen

import kotlinx.serialization.Serializable

/**
 * Result of a text-to-speech synthesis call.
 *
 * `audioBase64` holds the rendered audio bytes; `mimeType` identifies the
 * container (`"audio/wav"`, `"audio/mp3"`, ...). `durationSeconds` is best-
 * effort and may be zero when the upstream model doesn't report it.
 */
@Serializable
public data class TtsResult(
    val audioBase64: String,
    val mimeType: String,
    val durationSeconds: Double = 0.0,
)

/**
 * Result of a speech-to-text transcription call.
 *
 * `text` is the recognised transcript; `language` is an optional BCP-47 tag
 * for the detected language (empty when the model didn't report one).
 */
@Serializable
public data class SttResult(
    val text: String,
    val language: String = "",
)

/**
 * Result of a text-to-image generation call.
 *
 * `images` is one base64-encoded blob per generated image; `mimeType`
 * applies to every entry (typically `"image/png"`).
 */
@Serializable
public data class ImageGenResult(
    val images: List<String>,
    val mimeType: String,
)
