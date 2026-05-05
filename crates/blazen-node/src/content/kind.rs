//! [`JsContentKind`] — string-enum mirror of [`blazen_llm::content::ContentKind`].
//!
//! The Rust enum is `#[non_exhaustive]` and `#[serde(rename_all = "snake_case")]`.
//! On the napi side we expose it as a string enum whose variant names match the
//! `TypeScript` convention (`PascalCase` identifiers, snake-case wire values) so that
//! JS callers can write `ContentKind.Image` / `ContentKind.ThreeDModel` and pass
//! the values directly to factory functions.

use blazen_llm::content::ContentKind;
use napi_derive::napi;

/// Taxonomy of multimodal content kinds. Mirrors
/// [`blazen_llm::content::ContentKind`].
///
/// String values match the JSON / `serde` tag (`"image"`, `"three_d_model"`,
/// etc.) so they can be round-tripped against any Blazen API that accepts a
/// kind string.
#[napi(string_enum)]
pub enum JsContentKind {
    #[napi(value = "image")]
    Image,
    #[napi(value = "audio")]
    Audio,
    #[napi(value = "video")]
    Video,
    #[napi(value = "document")]
    Document,
    #[napi(value = "three_d_model")]
    ThreeDModel,
    #[napi(value = "cad")]
    Cad,
    #[napi(value = "archive")]
    Archive,
    #[napi(value = "font")]
    Font,
    #[napi(value = "code")]
    Code,
    #[napi(value = "data")]
    Data,
    #[napi(value = "other")]
    Other,
}

impl From<ContentKind> for JsContentKind {
    fn from(k: ContentKind) -> Self {
        match k {
            ContentKind::Image => Self::Image,
            ContentKind::Audio => Self::Audio,
            ContentKind::Video => Self::Video,
            ContentKind::Document => Self::Document,
            ContentKind::ThreeDModel => Self::ThreeDModel,
            ContentKind::Cad => Self::Cad,
            ContentKind::Archive => Self::Archive,
            ContentKind::Font => Self::Font,
            ContentKind::Code => Self::Code,
            ContentKind::Data => Self::Data,
            // `ContentKind` is `#[non_exhaustive]`. Future variants degrade to
            // `Other` rather than panicking; the JS side can detect this and
            // fall back appropriately.
            _ => Self::Other,
        }
    }
}

impl From<JsContentKind> for ContentKind {
    fn from(k: JsContentKind) -> Self {
        Self::from(&k)
    }
}

impl From<&JsContentKind> for ContentKind {
    fn from(k: &JsContentKind) -> Self {
        match k {
            JsContentKind::Image => Self::Image,
            JsContentKind::Audio => Self::Audio,
            JsContentKind::Video => Self::Video,
            JsContentKind::Document => Self::Document,
            JsContentKind::ThreeDModel => Self::ThreeDModel,
            JsContentKind::Cad => Self::Cad,
            JsContentKind::Archive => Self::Archive,
            JsContentKind::Font => Self::Font,
            JsContentKind::Code => Self::Code,
            JsContentKind::Data => Self::Data,
            JsContentKind::Other => Self::Other,
        }
    }
}
