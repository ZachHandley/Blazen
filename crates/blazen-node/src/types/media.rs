//! Media output types and media type constants.

use napi_derive::napi;

// ---------------------------------------------------------------------------
// Media output types
// ---------------------------------------------------------------------------

/// A single piece of generated media content.
#[napi(object)]
pub struct JsMediaOutput {
    /// URL where the media can be downloaded.
    pub url: Option<String>,
    /// Base64-encoded media data.
    pub base64: Option<String>,
    /// Raw text content for text-based formats (SVG, OBJ, GLTF JSON).
    #[napi(js_name = "rawContent")]
    pub raw_content: Option<String>,
    /// The MIME type of the media (e.g. "image/png", "video/mp4").
    #[napi(js_name = "mediaType")]
    pub media_type: String,
    /// File size in bytes, if known.
    #[napi(js_name = "fileSize")]
    pub file_size: Option<f64>,
    /// Arbitrary provider-specific metadata.
    pub metadata: serde_json::Value,
}

/// A single generated image with optional dimension metadata.
#[napi(object)]
pub struct JsGeneratedImage {
    /// The image media output.
    pub media: JsMediaOutput,
    /// Image width in pixels, if known.
    pub width: Option<u32>,
    /// Image height in pixels, if known.
    pub height: Option<u32>,
}

/// A single generated video with optional metadata.
#[napi(object)]
pub struct JsGeneratedVideo {
    /// The video media output.
    pub media: JsMediaOutput,
    /// Video width in pixels, if known.
    pub width: Option<u32>,
    /// Video height in pixels, if known.
    pub height: Option<u32>,
    /// Duration in seconds, if known.
    #[napi(js_name = "durationSeconds")]
    pub duration_seconds: Option<f64>,
    /// Frames per second, if known.
    pub fps: Option<f64>,
}

/// A single generated audio clip with optional metadata.
#[napi(object)]
pub struct JsGeneratedAudio {
    /// The audio media output.
    pub media: JsMediaOutput,
    /// Duration in seconds, if known.
    #[napi(js_name = "durationSeconds")]
    pub duration_seconds: Option<f64>,
    /// Sample rate in Hz, if known.
    #[napi(js_name = "sampleRate")]
    pub sample_rate: Option<u32>,
    /// Number of audio channels, if known.
    pub channels: Option<u32>,
}

/// A single generated 3D model with optional mesh metadata.
#[napi(object)]
pub struct JsGenerated3DModel {
    /// The 3D model media output.
    pub media: JsMediaOutput,
    /// Total vertex count, if known.
    #[napi(js_name = "vertexCount")]
    pub vertex_count: Option<f64>,
    /// Total face/triangle count, if known.
    #[napi(js_name = "faceCount")]
    pub face_count: Option<f64>,
    /// Whether the model includes texture data.
    #[napi(js_name = "hasTextures")]
    pub has_textures: bool,
    /// Whether the model includes animation data.
    #[napi(js_name = "hasAnimations")]
    pub has_animations: bool,
}

// ---------------------------------------------------------------------------
// Media type constants
// ---------------------------------------------------------------------------

/// Returns an object mapping friendly names to MIME type strings.
///
/// ```typescript
/// import { mediaTypes } from 'blazen';
///
/// const types = mediaTypes();
/// console.log(types.png);  // "image/png"
/// console.log(types.mp4);  // "video/mp4"
/// ```
#[napi(js_name = "mediaTypes")]
#[must_use]
pub fn media_types() -> JsMediaTypeMap {
    JsMediaTypeMap {
        // Images
        png: "image/png".to_owned(),
        jpeg: "image/jpeg".to_owned(),
        webp: "image/webp".to_owned(),
        gif: "image/gif".to_owned(),
        svg: "image/svg+xml".to_owned(),
        bmp: "image/bmp".to_owned(),
        tiff: "image/tiff".to_owned(),
        avif: "image/avif".to_owned(),
        ico: "image/x-icon".to_owned(),
        // Video
        mp4: "video/mp4".to_owned(),
        webm: "video/webm".to_owned(),
        mov: "video/quicktime".to_owned(),
        avi: "video/x-msvideo".to_owned(),
        mkv: "video/x-matroska".to_owned(),
        // Audio
        mp3: "audio/mpeg".to_owned(),
        wav: "audio/wav".to_owned(),
        ogg: "audio/ogg".to_owned(),
        flac: "audio/flac".to_owned(),
        aac: "audio/aac".to_owned(),
        m4a: "audio/mp4".to_owned(),
        // 3D Models
        glb: "model/gltf-binary".to_owned(),
        gltf: "model/gltf+json".to_owned(),
        obj: "model/obj".to_owned(),
        fbx: "application/octet-stream".to_owned(),
        usdz: "model/vnd.usdz+zip".to_owned(),
        stl: "model/stl".to_owned(),
        ply: "application/x-ply".to_owned(),
        // Documents
        pdf: "application/pdf".to_owned(),
    }
}

/// Map of friendly format names to their MIME type strings.
#[napi(object)]
pub struct JsMediaTypeMap {
    // Images
    pub png: String,
    pub jpeg: String,
    pub webp: String,
    pub gif: String,
    pub svg: String,
    pub bmp: String,
    pub tiff: String,
    pub avif: String,
    pub ico: String,
    // Video
    pub mp4: String,
    pub webm: String,
    pub mov: String,
    pub avi: String,
    pub mkv: String,
    // Audio
    pub mp3: String,
    pub wav: String,
    pub ogg: String,
    pub flac: String,
    pub aac: String,
    pub m4a: String,
    // 3D Models
    pub glb: String,
    pub gltf: String,
    pub obj: String,
    pub fbx: String,
    pub usdz: String,
    pub stl: String,
    pub ply: String,
    // Documents
    pub pdf: String,
}
