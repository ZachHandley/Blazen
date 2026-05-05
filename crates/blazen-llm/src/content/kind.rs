//! Content-kind taxonomy used by the content store and tool-input system.
//!
//! [`ContentKind`] classifies any binary or textual blob that flows through a
//! Blazen conversation into one of a small set of buckets. Tool input
//! declarations use these kinds to advertise what they accept (e.g. `image`,
//! `audio`, `cad`), and the content store uses them as a routing hint when
//! deciding how to upload / persist / serialize the bytes.
//!
//! The categories cover what the major LLM APIs distinguish natively
//! (Image / Audio / Video / Document) plus specialized buckets for content
//! that the framework should recognize and route appropriately even though
//! providers will largely treat them as opaque files (3D models, CAD files,
//! archives, fonts, code, structured data).

use serde::{Deserialize, Serialize};

/// Taxonomy of multimodal content kinds.
///
/// `#[non_exhaustive]` so future kinds (e.g. `Pointcloud`, `Volumetric`) can
/// be added without breaking semver downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentKind {
    /// Raster or vector image content. Includes static (PNG/JPEG/WebP/AVIF/
    /// HEIC/TIFF/BMP/SVG) and animated images (GIF). All providers that
    /// accept images natively map here.
    Image,
    /// Audio content (MP3/WAV/FLAC/OGG/Opus/AAC/M4A).
    Audio,
    /// Video content (MP4/MOV/WebM/MKV/AVI).
    Video,
    /// Documents intended for reading (PDF/DOCX/PPTX/XLSX/ODT/RTF/TXT/MD/HTML).
    Document,
    /// 3D model formats (glTF / GLB / OBJ / FBX / USDZ / STL / PLY / DAE).
    /// On most providers these degrade to `Document` / generic file.
    ThreeDModel,
    /// CAD formats (STEP / IGES / DWG / DXF / SAT / `X_T` / `X_B`).
    Cad,
    /// Compressed archive (ZIP / TAR / GZ / 7Z / RAR).
    Archive,
    /// Font file (TTF / OTF / WOFF / WOFF2).
    Font,
    /// Source code or build configuration. Identification is best-effort
    /// via extension; the framework does not perform language detection.
    Code,
    /// Structured data files (JSON / YAML / TOML / CSV / Parquet / Arrow).
    Data,
    /// Unknown or unclassified — fallback.
    Other,
}

impl ContentKind {
    /// Map a MIME type string to a [`ContentKind`].
    ///
    /// Unknown MIME types resolve to [`ContentKind::Other`].
    #[must_use]
    pub fn from_mime(mime: &str) -> Self {
        let lower = mime.to_ascii_lowercase();
        let (top, sub) = lower.split_once('/').unwrap_or((lower.as_str(), ""));
        match top {
            "image" => Self::Image,
            "audio" => Self::Audio,
            "video" => Self::Video,
            "font" => Self::Font,
            "model" => Self::ThreeDModel,
            "text" => match sub {
                "html" | "markdown" | "rtf" | "richtext" => Self::Document,
                "csv" | "tab-separated-values" => Self::Data,
                _ => Self::Code,
            },
            "application" => match sub {
                "pdf"
                | "msword"
                | "vnd.openxmlformats-officedocument.wordprocessingml.document"
                | "vnd.openxmlformats-officedocument.presentationml.presentation"
                | "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                | "vnd.oasis.opendocument.text"
                | "vnd.oasis.opendocument.spreadsheet"
                | "vnd.oasis.opendocument.presentation"
                | "rtf"
                | "epub+zip" => Self::Document,
                "json"
                | "ld+json"
                | "x-yaml"
                | "yaml"
                | "toml"
                | "x-toml"
                | "x-parquet"
                | "vnd.apache.arrow.file"
                | "vnd.apache.arrow.stream" => Self::Data,
                "zip" | "x-tar" | "gzip" | "x-7z-compressed" | "vnd.rar" | "x-bzip" | "x-bzip2"
                | "x-zip-compressed" => Self::Archive,
                "x-step" | "iges" | "x-iges" | "step" | "x-acad" | "acad" | "vnd.dwg"
                | "vnd.dxf" | "x-dxf" | "x-dwg" | "vnd.ms-pki.stl" => Self::Cad,
                "octet-stream" => Self::Other,
                "vnd.ms-fontobject" => Self::Font,
                _ if sub.contains("gltf")
                    || sub.contains("ply")
                    || sub.contains("usdz")
                    || sub.contains("fbx")
                    || sub.contains("obj")
                    || sub.contains("stl") =>
                {
                    Self::ThreeDModel
                }
                _ => Self::Other,
            },
            _ => Self::Other,
        }
    }

    /// Map a filename extension (without leading dot) to a [`ContentKind`].
    ///
    /// Unknown extensions resolve to [`ContentKind::Other`]. Matching is
    /// case-insensitive.
    #[must_use]
    pub fn from_extension(ext: &str) -> Self {
        let lower = ext.to_ascii_lowercase();
        match lower.as_str() {
            // Images
            "png" | "jpg" | "jpeg" | "webp" | "gif" | "avif" | "heic" | "heif" | "tif" | "tiff"
            | "bmp" | "svg" | "ico" => Self::Image,
            // Audio
            "mp3" | "wav" | "flac" | "ogg" | "opus" | "aac" | "m4a" | "wma" => Self::Audio,
            // Video
            "mp4" | "mov" | "webm" | "mkv" | "avi" | "wmv" | "flv" | "m4v" => Self::Video,
            // Documents
            "pdf" | "docx" | "doc" | "pptx" | "ppt" | "xlsx" | "xls" | "odt" | "ods" | "odp"
            | "rtf" | "txt" | "md" | "markdown" | "html" | "htm" | "epub" => Self::Document,
            // 3D models
            "gltf" | "glb" | "obj" | "fbx" | "usdz" | "usd" | "stl" | "ply" | "dae" | "3ds"
            | "blend" | "abc" => Self::ThreeDModel,
            // CAD
            "step" | "stp" | "iges" | "igs" | "dwg" | "dxf" | "sat" | "x_t" | "x_b" | "ipt"
            | "iam" | "prt" | "asm" | "sldprt" | "sldasm" => Self::Cad,
            // Archives
            "zip" | "tar" | "gz" | "tgz" | "bz2" | "xz" | "7z" | "rar" | "zst" => Self::Archive,
            // Fonts
            "ttf" | "otf" | "woff" | "woff2" | "eot" => Self::Font,
            // Code
            "rs" | "py" | "js" | "ts" | "tsx" | "jsx" | "go" | "java" | "c" | "cc" | "cpp"
            | "cxx" | "h" | "hh" | "hpp" | "cs" | "rb" | "php" | "swift" | "kt" | "scala"
            | "lua" | "sh" | "bash" | "zsh" | "fish" | "ps1" | "sql" | "vue" | "svelte"
            | "astro" => Self::Code,
            // Data
            "json" | "jsonl" | "ndjson" | "yaml" | "yml" | "toml" | "csv" | "tsv" | "parquet"
            | "arrow" | "ipc" | "feather" | "xml" => Self::Data,
            _ => Self::Other,
        }
    }

    /// Return the canonical short name (the JSON / `serde` tag).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Document => "document",
            Self::ThreeDModel => "three_d_model",
            Self::Cad => "cad",
            Self::Archive => "archive",
            Self::Font => "font",
            Self::Code => "code",
            Self::Data => "data",
            Self::Other => "other",
        }
    }
}

impl std::fmt::Display for ContentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::ContentKind;

    #[test]
    fn mime_image_family() {
        assert_eq!(ContentKind::from_mime("image/png"), ContentKind::Image);
        assert_eq!(ContentKind::from_mime("image/jpeg"), ContentKind::Image);
        assert_eq!(ContentKind::from_mime("image/svg+xml"), ContentKind::Image);
    }

    #[test]
    fn mime_audio_video() {
        assert_eq!(ContentKind::from_mime("audio/wav"), ContentKind::Audio);
        assert_eq!(ContentKind::from_mime("video/mp4"), ContentKind::Video);
    }

    #[test]
    fn mime_document() {
        assert_eq!(
            ContentKind::from_mime("application/pdf"),
            ContentKind::Document
        );
        assert_eq!(
            ContentKind::from_mime(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            ContentKind::Document
        );
    }

    #[test]
    fn mime_3d_model_top_level() {
        assert_eq!(
            ContentKind::from_mime("model/gltf-binary"),
            ContentKind::ThreeDModel
        );
    }

    #[test]
    fn mime_data_and_archive() {
        assert_eq!(
            ContentKind::from_mime("application/json"),
            ContentKind::Data
        );
        assert_eq!(
            ContentKind::from_mime("application/zip"),
            ContentKind::Archive
        );
    }

    #[test]
    fn extension_image_family() {
        assert_eq!(ContentKind::from_extension("png"), ContentKind::Image);
        assert_eq!(ContentKind::from_extension("HEIC"), ContentKind::Image);
        assert_eq!(ContentKind::from_extension("svg"), ContentKind::Image);
    }

    #[test]
    fn extension_3d_and_cad() {
        assert_eq!(ContentKind::from_extension("glb"), ContentKind::ThreeDModel);
        assert_eq!(ContentKind::from_extension("stl"), ContentKind::ThreeDModel);
        assert_eq!(ContentKind::from_extension("step"), ContentKind::Cad);
        assert_eq!(ContentKind::from_extension("dwg"), ContentKind::Cad);
    }

    #[test]
    fn unknown_extension_is_other() {
        assert_eq!(ContentKind::from_extension("xyzzy"), ContentKind::Other);
    }

    #[test]
    fn as_str_matches_serde_tag() {
        let serialized = serde_json::to_string(&ContentKind::ThreeDModel).unwrap();
        assert_eq!(serialized, "\"three_d_model\"");
        assert_eq!(ContentKind::ThreeDModel.as_str(), "three_d_model");
    }

    #[test]
    fn display_uses_short_name() {
        assert_eq!(format!("{}", ContentKind::Image), "image");
    }
}
