//! Shared media types for format detection and generated media output.
//!
//! These types are used across compute providers and are fundamental to
//! media generation workflows. [`MediaType`] provides exhaustive format
//! detection via magic bytes, MIME types, and file extensions.
//! [`MediaOutput`] represents a single piece of generated media content.
//! The `Generated*` structs add modality-specific metadata on top.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// MediaType
// ---------------------------------------------------------------------------

/// Exhaustive enumeration of media formats with detection support.
///
/// Covers images, video, audio, 3D models, documents, and a catch-all
/// `Other` variant for MIME types not explicitly listed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaType {
    // -- Images -------------------------------------------------------------
    /// Portable Network Graphics.
    Png,
    /// JPEG / JFIF.
    Jpeg,
    /// WebP (Google).
    WebP,
    /// Graphics Interchange Format.
    Gif,
    /// Scalable Vector Graphics (XML-based).
    Svg,
    /// Windows Bitmap.
    Bmp,
    /// Tagged Image File Format.
    Tiff,
    /// AV1 Image File Format.
    Avif,
    /// Windows Icon / Favicon.
    Ico,

    // -- Video --------------------------------------------------------------
    /// MPEG-4 Part 14.
    Mp4,
    /// `WebM` (Matroska + VP8/VP9/AV1).
    WebM,
    /// `QuickTime` Movie.
    Mov,
    /// Audio Video Interleave (Microsoft).
    Avi,
    /// Matroska Video.
    Mkv,

    // -- Audio --------------------------------------------------------------
    /// MPEG Audio Layer III.
    Mp3,
    /// Waveform Audio File Format.
    Wav,
    /// Ogg Vorbis.
    Ogg,
    /// Free Lossless Audio Codec.
    Flac,
    /// Advanced Audio Coding.
    Aac,
    /// MPEG-4 Audio (Apple).
    M4a,
    /// `WebM` audio-only container.
    WebmAudio,

    // -- 3D Models ----------------------------------------------------------
    /// GL Transmission Format Binary.
    Glb,
    /// GL Transmission Format (JSON).
    Gltf,
    /// Wavefront OBJ.
    Obj,
    /// Autodesk FBX.
    Fbx,
    /// Universal Scene Description (Apple AR).
    Usdz,
    /// Stereolithography / 3D printing.
    Stl,
    /// Polygon File Format / Stanford Triangle.
    Ply,

    // -- Documents ----------------------------------------------------------
    /// Portable Document Format.
    Pdf,

    // -- Catch-all ----------------------------------------------------------
    /// Any format not explicitly listed, identified by MIME string.
    Other {
        /// The MIME type string (e.g. `"application/octet-stream"`).
        mime: String,
    },
}

impl MediaType {
    /// Return the MIME type string for this media type.
    #[must_use]
    pub fn mime(&self) -> &str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::WebP => "image/webp",
            Self::Gif => "image/gif",
            Self::Svg => "image/svg+xml",
            Self::Bmp => "image/bmp",
            Self::Tiff => "image/tiff",
            Self::Avif => "image/avif",
            Self::Ico => "image/x-icon",
            Self::Mp4 => "video/mp4",
            Self::WebM => "video/webm",
            Self::Mov => "video/quicktime",
            Self::Avi => "video/x-msvideo",
            Self::Mkv => "video/x-matroska",
            Self::Mp3 => "audio/mpeg",
            Self::Wav => "audio/wav",
            Self::Ogg => "audio/ogg",
            Self::Flac => "audio/flac",
            Self::Aac => "audio/aac",
            Self::M4a => "audio/mp4",
            Self::WebmAudio => "audio/webm",
            Self::Glb => "model/gltf-binary",
            Self::Gltf => "model/gltf+json",
            Self::Obj => "model/obj",
            Self::Fbx => "application/octet-stream",
            Self::Usdz => "model/vnd.usdz+zip",
            Self::Stl => "model/stl",
            Self::Ply => "application/x-ply",
            Self::Pdf => "application/pdf",
            Self::Other { mime } => mime.as_str(),
        }
    }

    /// Return the canonical file extension (without leading dot).
    #[must_use]
    pub fn extension(&self) -> &str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::WebP => "webp",
            Self::Gif => "gif",
            Self::Svg => "svg",
            Self::Bmp => "bmp",
            Self::Tiff => "tiff",
            Self::Avif => "avif",
            Self::Ico => "ico",
            Self::Mp4 => "mp4",
            Self::WebM | Self::WebmAudio => "webm",
            Self::Mov => "mov",
            Self::Avi => "avi",
            Self::Mkv => "mkv",
            Self::Mp3 => "mp3",
            Self::Wav => "wav",
            Self::Ogg => "ogg",
            Self::Flac => "flac",
            Self::Aac => "aac",
            Self::M4a => "m4a",
            Self::Glb => "glb",
            Self::Gltf => "gltf",
            Self::Obj => "obj",
            Self::Fbx => "fbx",
            Self::Usdz => "usdz",
            Self::Stl => "stl",
            Self::Ply => "ply",
            Self::Pdf => "pdf",
            Self::Other { mime } => {
                // Best-effort: extract subtype from MIME.
                mime.rsplit('/').next().unwrap_or("bin")
            }
        }
    }

    /// Return the magic byte signature for formats that have one.
    ///
    /// Not all formats have magic bytes (e.g. SVG, GLTF JSON, OBJ are
    /// text-based). Returns `None` for those.
    #[must_use]
    pub fn magic_bytes(&self) -> Option<&'static [u8]> {
        match self {
            Self::Png => Some(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
            Self::Jpeg => Some(&[0xFF, 0xD8, 0xFF]),
            Self::Gif => Some(b"GIF8"),
            Self::Bmp => Some(b"BM"),
            Self::Tiff => Some(&[0x49, 0x49, 0x2A, 0x00]), // little-endian
            Self::Flac => Some(b"fLaC"),
            Self::Ogg => Some(b"OggS"),
            Self::Pdf => Some(b"%PDF"),
            Self::Glb => Some(&[0x67, 0x6C, 0x54, 0x46]),
            _ => None,
        }
    }

    /// Attempt to detect the media type from the first bytes of a file.
    ///
    /// Checks magic byte signatures in order from most specific to least.
    /// Returns `None` if no known signature matches.
    #[must_use]
    pub fn detect(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }

        // PNG: 8-byte signature
        if bytes.len() >= 8 && bytes[..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
            return Some(Self::Png);
        }

        // JPEG: FF D8 FF
        if bytes.len() >= 3 && bytes[..3] == [0xFF, 0xD8, 0xFF] {
            return Some(Self::Jpeg);
        }

        // GIF: GIF8
        if bytes[..4] == *b"GIF8" {
            return Some(Self::Gif);
        }

        // RIFF container: WebP, AVI, WAV (need at least 12 bytes)
        if bytes.len() >= 12 && bytes[..4] == *b"RIFF" {
            let fourcc = &bytes[8..12];
            if fourcc == b"WEBP" {
                return Some(Self::WebP);
            }
            if fourcc == b"AVI " {
                return Some(Self::Avi);
            }
            if fourcc == b"WAVE" {
                return Some(Self::Wav);
            }
        }

        // BMP: BM
        if bytes[..2] == *b"BM" {
            return Some(Self::Bmp);
        }

        // TIFF: little-endian or big-endian
        if bytes[..4] == [0x49, 0x49, 0x2A, 0x00] || bytes[..4] == [0x4D, 0x4D, 0x00, 0x2A] {
            return Some(Self::Tiff);
        }

        // EBML container: WebM / MKV
        if bytes[..4] == [0x1A, 0x45, 0xDF, 0xA3] {
            // Both WebM and MKV use EBML; default to WebM since it is more
            // common in media-generation contexts.
            return Some(Self::WebM);
        }

        // MP3: frame sync or ID3 header
        if bytes[..2] == [0xFF, 0xFB] || (bytes.len() >= 3 && bytes[..3] == [0x49, 0x44, 0x33]) {
            return Some(Self::Mp3);
        }

        // OGG: OggS
        if bytes[..4] == *b"OggS" {
            return Some(Self::Ogg);
        }

        // FLAC: fLaC
        if bytes[..4] == *b"fLaC" {
            return Some(Self::Flac);
        }

        // PDF: %PDF
        if bytes[..4] == *b"%PDF" {
            return Some(Self::Pdf);
        }

        // GLB: glTF magic
        if bytes[..4] == [0x67, 0x6C, 0x54, 0x46] {
            return Some(Self::Glb);
        }

        // ftyp-based containers (MP4, MOV, M4A, AVIF) -- the ftyp box starts
        // at offset 4 in the first 12 bytes.
        if bytes.len() >= 12 && bytes[4..8] == *b"ftyp" {
            let brand = &bytes[8..12];
            if brand == b"avif" || brand == b"avis" {
                return Some(Self::Avif);
            }
            if brand == b"M4A " || brand == b"M4B " {
                return Some(Self::M4a);
            }
            if brand == b"qt  " {
                return Some(Self::Mov);
            }
            // Default ftyp to MP4
            return Some(Self::Mp4);
        }

        None
    }

    /// Parse a MIME type string into a [`MediaType`].
    ///
    /// Always returns a value -- unknown MIME strings produce
    /// [`MediaType::Other`].
    #[must_use]
    pub fn from_mime(mime: &str) -> Self {
        match mime.to_ascii_lowercase().as_str() {
            "image/png" => Self::Png,
            "image/jpeg" | "image/jpg" => Self::Jpeg,
            "image/webp" => Self::WebP,
            "image/gif" => Self::Gif,
            "image/svg+xml" => Self::Svg,
            "image/bmp" | "image/x-bmp" => Self::Bmp,
            "image/tiff" => Self::Tiff,
            "image/avif" => Self::Avif,
            "image/x-icon" | "image/vnd.microsoft.icon" => Self::Ico,
            "video/mp4" => Self::Mp4,
            "video/webm" => Self::WebM,
            "video/quicktime" => Self::Mov,
            "video/x-msvideo" => Self::Avi,
            "video/x-matroska" => Self::Mkv,
            "audio/mpeg" | "audio/mp3" => Self::Mp3,
            "audio/wav" | "audio/x-wav" | "audio/wave" => Self::Wav,
            "audio/ogg" => Self::Ogg,
            "audio/flac" => Self::Flac,
            "audio/aac" => Self::Aac,
            "audio/mp4" | "audio/x-m4a" | "audio/m4a" => Self::M4a,
            "audio/webm" => Self::WebmAudio,
            "model/gltf-binary" => Self::Glb,
            "model/gltf+json" => Self::Gltf,
            "model/obj" => Self::Obj,
            "model/vnd.usdz+zip" => Self::Usdz,
            "model/stl" => Self::Stl,
            "application/x-ply" => Self::Ply,
            "application/pdf" => Self::Pdf,
            other => Self::Other {
                mime: other.to_owned(),
            },
        }
    }

    /// Parse a file extension into a [`MediaType`].
    ///
    /// The extension should not include a leading dot. Always returns a
    /// value -- unknown extensions produce [`MediaType::Other`].
    #[must_use]
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            "png" => Self::Png,
            "jpg" | "jpeg" => Self::Jpeg,
            "webp" => Self::WebP,
            "gif" => Self::Gif,
            "svg" => Self::Svg,
            "bmp" => Self::Bmp,
            "tiff" | "tif" => Self::Tiff,
            "avif" => Self::Avif,
            "ico" => Self::Ico,
            "mp4" | "m4v" => Self::Mp4,
            "webm" => Self::WebM,
            "mov" => Self::Mov,
            "avi" => Self::Avi,
            "mkv" => Self::Mkv,
            "mp3" => Self::Mp3,
            "wav" => Self::Wav,
            "ogg" | "oga" => Self::Ogg,
            "flac" => Self::Flac,
            "aac" => Self::Aac,
            "m4a" => Self::M4a,
            "glb" => Self::Glb,
            "gltf" => Self::Gltf,
            "obj" => Self::Obj,
            "fbx" => Self::Fbx,
            "usdz" => Self::Usdz,
            "stl" => Self::Stl,
            "ply" => Self::Ply,
            "pdf" => Self::Pdf,
            other => Self::Other {
                mime: format!("application/x-{other}"),
            },
        }
    }

    /// Whether this type is a raster or vector image format.
    #[must_use]
    pub fn is_image(&self) -> bool {
        matches!(
            self,
            Self::Png
                | Self::Jpeg
                | Self::WebP
                | Self::Gif
                | Self::Svg
                | Self::Bmp
                | Self::Tiff
                | Self::Avif
                | Self::Ico
        )
    }

    /// Whether this type is a video format.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            Self::Mp4 | Self::WebM | Self::Mov | Self::Avi | Self::Mkv
        )
    }

    /// Whether this type is an audio format.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            Self::Mp3
                | Self::Wav
                | Self::Ogg
                | Self::Flac
                | Self::Aac
                | Self::M4a
                | Self::WebmAudio
        )
    }

    /// Whether this type is a 3D model format.
    #[must_use]
    pub fn is_3d(&self) -> bool {
        matches!(
            self,
            Self::Glb | Self::Gltf | Self::Obj | Self::Fbx | Self::Usdz | Self::Stl | Self::Ply
        )
    }

    /// Whether this type is a vector/text-based format (SVG, GLTF, OBJ).
    #[must_use]
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Svg | Self::Gltf | Self::Obj)
    }
}

impl std::fmt::Display for MediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.mime())
    }
}

// ---------------------------------------------------------------------------
// MediaOutput
// ---------------------------------------------------------------------------

/// A single piece of generated media content.
///
/// At least one of `url`, `base64`, or `raw_content` will be populated.
/// `raw_content` is used for text-based formats like SVG, OBJ, and GLTF.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct MediaOutput {
    /// URL where the media can be downloaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Base64-encoded media data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base64: Option<String>,
    /// Raw text content for text-based formats (SVG, OBJ, GLTF JSON).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_content: Option<String>,
    /// The format of the media.
    pub media_type: MediaType,
    /// File size in bytes, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_size: Option<u64>,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

impl MediaOutput {
    /// Create a [`MediaOutput`] from a URL.
    #[must_use]
    pub fn from_url(url: impl Into<String>, media_type: MediaType) -> Self {
        Self {
            url: Some(url.into()),
            base64: None,
            raw_content: None,
            media_type,
            file_size: None,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Create a [`MediaOutput`] from base64-encoded data.
    #[must_use]
    pub fn from_base64(data: impl Into<String>, media_type: MediaType) -> Self {
        Self {
            url: None,
            base64: Some(data.into()),
            raw_content: None,
            media_type,
            file_size: None,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
        }
    }
}

// ---------------------------------------------------------------------------
// Generated media types
// ---------------------------------------------------------------------------

/// A single generated image with optional dimension metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct GeneratedImage {
    /// The image media output.
    pub media: MediaOutput,
    /// Image width in pixels, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Image height in pixels, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

/// A single generated video with optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct GeneratedVideo {
    /// The video media output.
    pub media: MediaOutput,
    /// Video width in pixels, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Video height in pixels, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    /// Duration in seconds, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f32>,
    /// Frames per second, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<f32>,
}

/// A single generated audio clip with optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct GeneratedAudio {
    /// The audio media output.
    pub media: MediaOutput,
    /// Duration in seconds, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f32>,
    /// Sample rate in Hz, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    /// Number of audio channels, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channels: Option<u8>,
}

/// A single generated 3D model with optional mesh metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct Generated3DModel {
    /// The 3D model media output.
    pub media: MediaOutput,
    /// Total vertex count, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vertex_count: Option<u64>,
    /// Total face/triangle count, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub face_count: Option<u64>,
    /// Whether the model includes texture data.
    pub has_textures: bool,
    /// Whether the model includes animation data.
    pub has_animations: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // MediaType::detect
    // -----------------------------------------------------------------------

    #[test]
    fn detect_png() {
        let bytes = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00];
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::Png));
    }

    #[test]
    fn detect_jpeg() {
        let bytes = [0xFF, 0xD8, 0xFF, 0xE0, 0x00];
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::Jpeg));
    }

    #[test]
    fn detect_gif() {
        let bytes = b"GIF89a\x00\x00";
        assert_eq!(MediaType::detect(bytes), Some(MediaType::Gif));
    }

    #[test]
    fn detect_webp() {
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(b"RIFF");
        bytes[8..12].copy_from_slice(b"WEBP");
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::WebP));
    }

    #[test]
    fn detect_bmp() {
        let bytes = b"BM\x00\x00\x00\x00";
        assert_eq!(MediaType::detect(bytes), Some(MediaType::Bmp));
    }

    #[test]
    fn detect_tiff_le() {
        let bytes = [0x49, 0x49, 0x2A, 0x00, 0x08];
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::Tiff));
    }

    #[test]
    fn detect_tiff_be() {
        let bytes = [0x4D, 0x4D, 0x00, 0x2A, 0x00];
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::Tiff));
    }

    #[test]
    fn detect_webm() {
        let bytes = [0x1A, 0x45, 0xDF, 0xA3, 0x01];
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::WebM));
    }

    #[test]
    fn detect_avi() {
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(b"RIFF");
        bytes[8..12].copy_from_slice(b"AVI ");
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::Avi));
    }

    #[test]
    fn detect_wav() {
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(b"RIFF");
        bytes[8..12].copy_from_slice(b"WAVE");
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::Wav));
    }

    #[test]
    fn detect_mp3_sync() {
        let bytes = [0xFF, 0xFB, 0x90, 0x00];
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::Mp3));
    }

    #[test]
    fn detect_mp3_id3() {
        let bytes = [0x49, 0x44, 0x33, 0x04, 0x00];
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::Mp3));
    }

    #[test]
    fn detect_ogg() {
        let bytes = b"OggS\x00\x02";
        assert_eq!(MediaType::detect(bytes), Some(MediaType::Ogg));
    }

    #[test]
    fn detect_flac() {
        let bytes = b"fLaC\x00\x00";
        assert_eq!(MediaType::detect(bytes), Some(MediaType::Flac));
    }

    #[test]
    fn detect_pdf() {
        let bytes = b"%PDF-1.7\x00";
        assert_eq!(MediaType::detect(bytes), Some(MediaType::Pdf));
    }

    #[test]
    fn detect_glb() {
        let bytes = [0x67, 0x6C, 0x54, 0x46, 0x02, 0x00];
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::Glb));
    }

    #[test]
    fn detect_mp4_ftyp() {
        let mut bytes = [0u8; 16];
        // ftyp box at offset 4
        bytes[4..8].copy_from_slice(b"ftyp");
        bytes[8..12].copy_from_slice(b"isom");
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::Mp4));
    }

    #[test]
    fn detect_avif_ftyp() {
        let mut bytes = [0u8; 16];
        bytes[4..8].copy_from_slice(b"ftyp");
        bytes[8..12].copy_from_slice(b"avif");
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::Avif));
    }

    #[test]
    fn detect_mov_ftyp() {
        let mut bytes = [0u8; 16];
        bytes[4..8].copy_from_slice(b"ftyp");
        bytes[8..12].copy_from_slice(b"qt  ");
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::Mov));
    }

    #[test]
    fn detect_m4a_ftyp() {
        let mut bytes = [0u8; 16];
        bytes[4..8].copy_from_slice(b"ftyp");
        bytes[8..12].copy_from_slice(b"M4A ");
        assert_eq!(MediaType::detect(&bytes), Some(MediaType::M4a));
    }

    #[test]
    fn detect_unknown() {
        let bytes = [0x00, 0x01, 0x02, 0x03, 0x04];
        assert_eq!(MediaType::detect(&bytes), None);
    }

    #[test]
    fn detect_too_short() {
        let bytes = [0x89, 0x50];
        assert_eq!(MediaType::detect(&bytes), None);
    }

    // -----------------------------------------------------------------------
    // MediaType::from_mime / from_extension roundtrips
    // -----------------------------------------------------------------------

    #[test]
    fn mime_roundtrip_known_types() {
        let types = vec![
            MediaType::Png,
            MediaType::Jpeg,
            MediaType::WebP,
            MediaType::Gif,
            MediaType::Svg,
            MediaType::Bmp,
            MediaType::Tiff,
            MediaType::Avif,
            MediaType::Ico,
            MediaType::Mp4,
            MediaType::WebM,
            MediaType::Mov,
            MediaType::Avi,
            MediaType::Mkv,
            MediaType::Mp3,
            MediaType::Wav,
            MediaType::Ogg,
            MediaType::Flac,
            MediaType::Aac,
            MediaType::M4a,
            MediaType::WebmAudio,
            MediaType::Glb,
            MediaType::Gltf,
            MediaType::Obj,
            MediaType::Usdz,
            MediaType::Stl,
            MediaType::Ply,
            MediaType::Pdf,
        ];
        for t in types {
            let mime = t.mime();
            let roundtripped = MediaType::from_mime(mime);
            assert_eq!(roundtripped, t, "from_mime({mime:?}) should produce {t:?}");
        }
    }

    #[test]
    fn extension_roundtrip_known_types() {
        // Some types share the same extension (WebM / WebmAudio both use "webm"),
        // so we test a subset that has unique extensions.
        let types = vec![
            MediaType::Png,
            MediaType::Jpeg,
            MediaType::Gif,
            MediaType::Svg,
            MediaType::Bmp,
            MediaType::Tiff,
            MediaType::Avif,
            MediaType::Ico,
            MediaType::Mp4,
            MediaType::Mov,
            MediaType::Avi,
            MediaType::Mkv,
            MediaType::Mp3,
            MediaType::Wav,
            MediaType::Ogg,
            MediaType::Flac,
            MediaType::Aac,
            MediaType::M4a,
            MediaType::Glb,
            MediaType::Gltf,
            MediaType::Obj,
            MediaType::Fbx,
            MediaType::Usdz,
            MediaType::Stl,
            MediaType::Ply,
            MediaType::Pdf,
        ];
        for t in types {
            let ext = t.extension();
            let roundtripped = MediaType::from_extension(ext);
            assert_eq!(
                roundtripped, t,
                "from_extension({ext:?}) should produce {t:?}"
            );
        }
    }

    #[test]
    fn from_mime_unknown_returns_other() {
        let t = MediaType::from_mime("application/x-custom");
        assert_eq!(
            t,
            MediaType::Other {
                mime: "application/x-custom".into()
            }
        );
    }

    #[test]
    fn from_extension_unknown_returns_other() {
        let t = MediaType::from_extension("xyz");
        assert!(matches!(t, MediaType::Other { .. }));
    }

    #[test]
    fn from_mime_case_insensitive() {
        assert_eq!(MediaType::from_mime("IMAGE/PNG"), MediaType::Png);
        assert_eq!(MediaType::from_mime("Audio/MPEG"), MediaType::Mp3);
    }

    #[test]
    fn from_extension_case_insensitive() {
        assert_eq!(MediaType::from_extension("PNG"), MediaType::Png);
        assert_eq!(MediaType::from_extension("JPEG"), MediaType::Jpeg);
    }

    // -----------------------------------------------------------------------
    // MediaType::is_* category methods
    // -----------------------------------------------------------------------

    #[test]
    fn is_image_categories() {
        assert!(MediaType::Png.is_image());
        assert!(MediaType::Jpeg.is_image());
        assert!(MediaType::Svg.is_image());
        assert!(!MediaType::Mp4.is_image());
        assert!(!MediaType::Mp3.is_image());
        assert!(!MediaType::Glb.is_image());
    }

    #[test]
    fn is_video_categories() {
        assert!(MediaType::Mp4.is_video());
        assert!(MediaType::WebM.is_video());
        assert!(MediaType::Mkv.is_video());
        assert!(!MediaType::Png.is_video());
        assert!(!MediaType::Mp3.is_video());
    }

    #[test]
    fn is_audio_categories() {
        assert!(MediaType::Mp3.is_audio());
        assert!(MediaType::Wav.is_audio());
        assert!(MediaType::Flac.is_audio());
        assert!(MediaType::WebmAudio.is_audio());
        assert!(!MediaType::WebM.is_audio());
        assert!(!MediaType::Png.is_audio());
    }

    #[test]
    fn is_3d_categories() {
        assert!(MediaType::Glb.is_3d());
        assert!(MediaType::Gltf.is_3d());
        assert!(MediaType::Obj.is_3d());
        assert!(MediaType::Usdz.is_3d());
        assert!(!MediaType::Png.is_3d());
        assert!(!MediaType::Mp4.is_3d());
    }

    #[test]
    fn is_vector_categories() {
        assert!(MediaType::Svg.is_vector());
        assert!(MediaType::Gltf.is_vector());
        assert!(MediaType::Obj.is_vector());
        assert!(!MediaType::Png.is_vector());
        assert!(!MediaType::Glb.is_vector());
    }

    // -----------------------------------------------------------------------
    // MediaOutput constructors
    // -----------------------------------------------------------------------

    #[test]
    fn media_output_from_url() {
        let output = MediaOutput::from_url("https://example.com/image.png", MediaType::Png);
        assert_eq!(output.url.as_deref(), Some("https://example.com/image.png"));
        assert!(output.base64.is_none());
        assert!(output.raw_content.is_none());
        assert_eq!(output.media_type, MediaType::Png);
        assert!(output.file_size.is_none());
    }

    #[test]
    fn media_output_from_base64() {
        let output = MediaOutput::from_base64("iVBORw0KGgo=", MediaType::Png);
        assert!(output.url.is_none());
        assert_eq!(output.base64.as_deref(), Some("iVBORw0KGgo="));
        assert_eq!(output.media_type, MediaType::Png);
    }
}
