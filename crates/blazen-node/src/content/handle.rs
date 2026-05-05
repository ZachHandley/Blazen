//! [`JsContentHandle`] — plain-data wrapper around
//! [`blazen_llm::content::ContentHandle`].
//!
//! Exposed as a `#[napi(object)]` so JS callers can construct one with a
//! plain object literal (`{ id: "blazen_abc", kind: "image", ... }`) and so
//! the `ContentStore` methods can accept / return one without round-tripping
//! through a class instance.

use blazen_llm::content::ContentHandle;
use napi_derive::napi;

use super::kind::JsContentKind;

/// Reference to content stored in a [`super::store::JsContentStore`].
///
/// Mirrors [`blazen_llm::content::ContentHandle`]. Pass instances to a
/// store's `resolve` / `fetchBytes` / `metadata` / `delete` methods, or
/// embed the `id` in tool arguments where a content reference is expected.
#[napi(object)]
pub struct JsContentHandle {
    /// Opaque, store-defined identifier. Treat as a black box.
    pub id: String,
    /// What kind of content this handle refers to.
    pub kind: JsContentKind,
    /// MIME type if known.
    #[napi(js_name = "mimeType")]
    pub mime_type: Option<String>,
    /// Byte size if known. `i64` because napi has no `u64`.
    #[napi(js_name = "byteSize")]
    pub byte_size: Option<i64>,
    /// Human-readable display name (e.g. original filename).
    #[napi(js_name = "displayName")]
    pub display_name: Option<String>,
}

impl JsContentHandle {
    /// Convert this JS-side handle into the Rust [`ContentHandle`].
    #[must_use]
    pub fn to_rust(&self) -> ContentHandle {
        let mut handle = ContentHandle::new(self.id.clone(), (&self.kind).into());
        handle.mime_type.clone_from(&self.mime_type);
        #[allow(clippy::cast_sign_loss)]
        {
            handle.byte_size = self.byte_size.map(|n| n as u64);
        }
        handle.display_name.clone_from(&self.display_name);
        handle
    }

    /// Build a JS-side handle from a Rust [`ContentHandle`].
    #[must_use]
    pub fn from_rust(h: &ContentHandle) -> Self {
        #[allow(clippy::cast_possible_wrap)]
        Self {
            id: h.id.clone(),
            kind: h.kind.into(),
            mime_type: h.mime_type.clone(),
            byte_size: h.byte_size.map(|n| n as i64),
            display_name: h.display_name.clone(),
        }
    }
}
