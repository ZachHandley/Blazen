//! Stable references to content stored in a [`ContentStore`].
//!
//! A [`ContentHandle`] is the conversation-scoped, opaque identifier the
//! framework hands to the model when multimodal content is registered. The
//! model passes the handle's `id` back as a tool argument; the framework
//! resolves the id to typed content via the active store before invoking
//! the tool's `execute`.
//!
//! Handles are also embedded in [`ImageSource::Handle`] so that messages
//! constructed before serialization can defer "how do I render this?" to
//! the active store at request-build time.
//!
//! [`ContentStore`]: super::store::ContentStore
//! [`ImageSource::Handle`]: crate::types::ImageSource::Handle

use serde::{Deserialize, Serialize};

use super::kind::ContentKind;

/// Reference to a piece of content registered with a [`ContentStore`].
///
/// `id` is the only required field; everything else is metadata that the
/// framework attaches when the handle is created and that consumers (the
/// model, tool runner, provider serializers) can use to make routing
/// decisions without dereferencing the bytes.
///
/// Handles are stable for the lifetime of the conversation. They are *not*
/// guaranteed to be unique across conversations — different conversations
/// may have their own content stores and their own ID spaces.
///
/// [`ContentStore`]: super::store::ContentStore
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ContentHandle {
    /// Opaque, stable identifier. Format is store-defined; treat as a
    /// black box.
    pub id: String,
    /// What kind of content this handle refers to. Used for type-checking
    /// at the tool-input boundary (e.g. an `image_input` rejects a handle
    /// whose `kind` is `Audio`).
    pub kind: ContentKind,
    /// MIME type if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Byte size if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_size: Option<u64>,
    /// Human-readable display name (e.g. original filename) if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
}

impl ContentHandle {
    /// Build a minimal handle with just an ID and a kind.
    #[must_use]
    pub fn new(id: impl Into<String>, kind: ContentKind) -> Self {
        Self {
            id: id.into(),
            kind,
            mime_type: None,
            byte_size: None,
            display_name: None,
        }
    }

    /// Builder: attach a MIME type.
    #[must_use]
    pub fn with_mime_type(mut self, mime: impl Into<String>) -> Self {
        self.mime_type = Some(mime.into());
        self
    }

    /// Builder: attach a byte size.
    #[must_use]
    pub fn with_byte_size(mut self, size: u64) -> Self {
        self.byte_size = Some(size);
        self
    }

    /// Builder: attach a display name.
    #[must_use]
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_round_trip_serde() {
        let handle = ContentHandle::new("h_abc", ContentKind::Image)
            .with_mime_type("image/png")
            .with_byte_size(1234)
            .with_display_name("photo.png");
        let json = serde_json::to_string(&handle).unwrap();
        let back: ContentHandle = serde_json::from_str(&json).unwrap();
        assert_eq!(handle, back);
    }

    #[test]
    fn handle_minimal_serializes_without_optional_fields() {
        let handle = ContentHandle::new("h_x", ContentKind::Audio);
        let json = serde_json::to_value(&handle).unwrap();
        assert_eq!(json["id"], "h_x");
        assert_eq!(json["kind"], "audio");
        assert!(json.get("mime_type").is_none());
        assert!(json.get("byte_size").is_none());
        assert!(json.get("display_name").is_none());
    }
}
