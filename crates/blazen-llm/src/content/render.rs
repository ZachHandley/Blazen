//! Shared rendering helpers for [`MediaSource`] / [`ImageSource`] across
//! every provider's request builder.
//!
//! The new [`ImageSource::ProviderFile`] and [`ImageSource::Handle`] variants
//! cannot be rendered by every provider. Each provider's match arms collapse
//! into one of two cases at the boundary:
//!
//! 1. A `ProviderFile` whose `provider` matches the current provider can be
//!    rendered natively (each provider knows its own file-API ID shape).
//! 2. Anything else — non-matching `ProviderFile` IDs and unresolved
//!    `Handle`s — needs a [`ContentStore`] to either rehost the bytes or
//!    materialize them. Until [`ContentStore`] is wired into the request
//!    pipeline, those variants are dropped with a structured warning.
//!
//! These helpers centralize the warning emission so the message format is
//! uniform across providers and so the future "resolve via store" code path
//! has a single integration point.
//!
//! [`MediaSource`]: crate::types::MediaSource
//! [`ImageSource`]: crate::types::ImageSource
//! [`ImageSource::ProviderFile`]: crate::types::ImageSource::ProviderFile
//! [`ImageSource::Handle`]: crate::types::ImageSource::Handle
//! [`ContentStore`]: super::store::ContentStore

use crate::types::ProviderId;

use super::handle::ContentHandle;

/// Categorical label for the kind of media being dropped, used in the
/// warning message and structured tracing fields. Free-form string rather
/// than [`super::ContentKind`] because the caller often knows the precise
/// wire concept (e.g. "image", "audio", "document") and not always the
/// `ContentKind` of the underlying bytes.
#[derive(Debug, Clone, Copy)]
pub enum MediaKindLabel {
    Image,
    Audio,
    Video,
    File,
    Document,
}

impl MediaKindLabel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::File => "file",
            Self::Document => "document",
        }
    }
}

/// Emit a structured `tracing::warn` for a [`ImageSource::ProviderFile`]
/// whose `provider` does not match the rendering provider.
///
/// Until a [`ContentStore`] is plumbed through the request pipeline,
/// non-matching `ProviderFile` references cannot be rehosted and are
/// dropped from the outgoing request.
///
/// [`ImageSource::ProviderFile`]: crate::types::ImageSource::ProviderFile
/// [`ContentStore`]: super::store::ContentStore
pub fn warn_provider_file_mismatch(
    target: ProviderId,
    source_provider: ProviderId,
    id: &str,
    kind: MediaKindLabel,
) {
    tracing::warn!(
        target: "blazen::content",
        target_provider = ?target,
        source_provider = ?source_provider,
        file_id = %id,
        kind = kind.as_str(),
        "ProviderFile from a different provider requires a wired ContentStore to rehost; \
         {kind} content dropped.",
        kind = kind.as_str(),
    );
}

/// Emit a structured `tracing::warn` for an unresolved
/// [`ImageSource::Handle`].
///
/// Until [`ContentStore`] is wired into the request pipeline, handle-
/// referenced content cannot be materialized and is dropped from the
/// outgoing request.
///
/// [`ImageSource::Handle`]: crate::types::ImageSource::Handle
/// [`ContentStore`]: super::store::ContentStore
pub fn warn_handle_unresolved(target: ProviderId, handle: &ContentHandle, kind: MediaKindLabel) {
    tracing::warn!(
        target: "blazen::content",
        target_provider = ?target,
        handle_id = %handle.id,
        handle_kind = %handle.kind,
        kind = kind.as_str(),
        "ContentHandle requires a wired ContentStore to resolve; {kind} content dropped.",
        kind = kind.as_str(),
    );
}
