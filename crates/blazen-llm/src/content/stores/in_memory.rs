//! [`InMemoryContentStore`] — default content store backed by a `HashMap`.
//!
//! Bytes (or references to bytes) are kept in a process-local hash map keyed
//! by the issued [`ContentHandle::id`]. Resolution always returns the
//! cheapest representation: a recorded URL or `ProviderFile` reference is
//! returned verbatim; raw bytes resolve to base64 (with the recorded
//! MIME type).
//!
//! This is the default store wired into an agent when the user doesn't
//! supply one. Suitable for small / ephemeral content; switch to
//! [`super::LocalFileContentStore`] or a provider-file store when content
//! grows or persistence is required.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;
use base64::Engine;
use uuid::Uuid;

use crate::content::detect::detect;
use crate::content::handle::ContentHandle;
use crate::content::kind::ContentKind;
use crate::content::store::{ContentBody, ContentHint, ContentMetadata, ContentStore};
use crate::error::BlazenError;
use crate::types::{MediaSource, ProviderId};

/// Internal record kept per handle.
#[derive(Debug, Clone)]
#[allow(dead_code)] // `mime` informational; reserved for resolution upgrades (e.g. data URIs).
enum Stored {
    Bytes {
        bytes: Vec<u8>,
        mime: Option<String>,
    },
    Url(String),
    LocalPath(PathBuf),
    ProviderFile {
        provider: ProviderId,
        id: String,
    },
}

#[derive(Debug, Clone)]
struct Record {
    stored: Stored,
    kind: ContentKind,
    mime_type: Option<String>,
    byte_size: Option<u64>,
    display_name: Option<String>,
}

/// Default content store: an in-memory `HashMap<id, Record>`.
#[derive(Debug, Default)]
pub struct InMemoryContentStore {
    inner: Mutex<HashMap<String, Record>>,
}

impl InMemoryContentStore {
    /// Create a new empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    fn next_id() -> String {
        // Short, URL-safe id; collisions astronomically unlikely.
        let raw = Uuid::new_v4().simple().to_string();
        format!("blazen_{}", &raw[..16])
    }
}

#[async_trait]
impl ContentStore for InMemoryContentStore {
    #[allow(clippy::too_many_lines)]
    async fn put(
        &self,
        body: ContentBody,
        hint: ContentHint,
    ) -> Result<ContentHandle, BlazenError> {
        let id = Self::next_id();

        let (record, kind, mime, byte_size): (Stored, ContentKind, Option<String>, Option<u64>) =
            match body {
                ContentBody::Bytes { data: bytes } => {
                    let (auto_kind, auto_mime) = if hint.kind_hint.is_some() {
                        (ContentKind::Other, None)
                    } else {
                        detect(
                            Some(&bytes),
                            hint.mime_type.as_deref(),
                            hint.display_name.as_deref(),
                        )
                    };
                    let kind = hint.kind_hint.unwrap_or(auto_kind);
                    let mime = hint.mime_type.clone().or(auto_mime);
                    #[allow(clippy::cast_possible_truncation)]
                    let size = Some(bytes.len() as u64);
                    let record = Stored::Bytes {
                        bytes,
                        mime: mime.clone(),
                    };
                    (record, kind, mime, size)
                }
                ContentBody::Url { url } => {
                    let kind = hint
                        .kind_hint
                        .unwrap_or_else(|| detect(None, hint.mime_type.as_deref(), Some(&url)).0);
                    (
                        Stored::Url(url),
                        kind,
                        hint.mime_type.clone(),
                        hint.byte_size,
                    )
                }
                ContentBody::LocalPath { path } => {
                    let detected = detect(
                        None,
                        hint.mime_type.as_deref(),
                        path.file_name().and_then(|n| n.to_str()),
                    );
                    let kind = hint.kind_hint.unwrap_or(detected.0);
                    let mime = hint.mime_type.clone().or(detected.1);
                    (Stored::LocalPath(path), kind, mime, hint.byte_size)
                }
                ContentBody::ProviderFile {
                    provider,
                    id: file_id,
                } => {
                    let kind = hint.kind_hint.unwrap_or(ContentKind::Other);
                    (
                        Stored::ProviderFile {
                            provider,
                            id: file_id,
                        },
                        kind,
                        hint.mime_type.clone(),
                        hint.byte_size,
                    )
                }
                ContentBody::Stream { stream, size_hint } => {
                    use futures_util::StreamExt;
                    let mut buf: Vec<u8> =
                        Vec::with_capacity(usize::try_from(size_hint.unwrap_or(0)).unwrap_or(0));
                    let mut s = stream;
                    while let Some(chunk) = s.next().await {
                        buf.extend_from_slice(&chunk?);
                    }
                    // Now treat it like the existing Bytes path.
                    let (auto_kind, auto_mime) = if hint.kind_hint.is_some() {
                        (ContentKind::Other, None)
                    } else {
                        detect(
                            Some(&buf),
                            hint.mime_type.as_deref(),
                            hint.display_name.as_deref(),
                        )
                    };
                    let kind = hint.kind_hint.unwrap_or(auto_kind);
                    let mime = hint.mime_type.clone().or(auto_mime);
                    #[allow(clippy::cast_possible_truncation)]
                    let size = Some(buf.len() as u64);
                    let record = Stored::Bytes {
                        bytes: buf,
                        mime: mime.clone(),
                    };
                    (record, kind, mime, size)
                }
            };

        let mut handle = ContentHandle::new(id.clone(), kind);
        handle.mime_type.clone_from(&mime);
        handle.byte_size = byte_size;
        handle.display_name.clone_from(&hint.display_name);

        let full_record = Record {
            stored: record,
            kind,
            mime_type: mime,
            byte_size,
            display_name: hint.display_name,
        };

        self.inner
            .lock()
            .map_err(|_| BlazenError::internal("InMemoryContentStore: mutex poisoned during put"))?
            .insert(id, full_record);

        Ok(handle)
    }

    async fn resolve(&self, handle: &ContentHandle) -> Result<MediaSource, BlazenError> {
        let record = self
            .inner
            .lock()
            .map_err(|_| {
                BlazenError::internal("InMemoryContentStore: mutex poisoned during resolve")
            })?
            .get(&handle.id)
            .cloned()
            .ok_or_else(|| {
                BlazenError::internal(format!(
                    "InMemoryContentStore: handle '{}' not found",
                    handle.id
                ))
            })?;
        Ok(match record.stored {
            Stored::Url(url) => MediaSource::Url { url },
            Stored::Bytes { bytes, .. } => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                MediaSource::Base64 { data: b64 }
            }
            Stored::LocalPath(path) => MediaSource::File { path },
            Stored::ProviderFile { provider, id } => MediaSource::ProviderFile { provider, id },
        })
    }

    async fn fetch_bytes(&self, handle: &ContentHandle) -> Result<Vec<u8>, BlazenError> {
        let record = self
            .inner
            .lock()
            .map_err(|_| {
                BlazenError::internal("InMemoryContentStore: mutex poisoned during fetch_bytes")
            })?
            .get(&handle.id)
            .cloned()
            .ok_or_else(|| {
                BlazenError::internal(format!(
                    "InMemoryContentStore: handle '{}' not found",
                    handle.id
                ))
            })?;
        match record.stored {
            Stored::Bytes { bytes, .. } => Ok(bytes),
            Stored::Url(_) | Stored::LocalPath(_) | Stored::ProviderFile { .. } => {
                Err(BlazenError::unsupported(
                    "InMemoryContentStore::fetch_bytes only supports inline-bytes records; \
                     use a store that knows how to materialize URL / file / provider-file refs",
                ))
            }
        }
    }

    async fn metadata(&self, handle: &ContentHandle) -> Result<ContentMetadata, BlazenError> {
        let record = self
            .inner
            .lock()
            .map_err(|_| {
                BlazenError::internal("InMemoryContentStore: mutex poisoned during metadata")
            })?
            .get(&handle.id)
            .cloned()
            .ok_or_else(|| {
                BlazenError::internal(format!(
                    "InMemoryContentStore: handle '{}' not found",
                    handle.id
                ))
            })?;
        Ok(ContentMetadata {
            kind: record.kind,
            mime_type: record.mime_type,
            byte_size: record.byte_size,
            display_name: record.display_name,
        })
    }

    async fn delete(&self, handle: &ContentHandle) -> Result<(), BlazenError> {
        self.inner
            .lock()
            .map_err(|_| {
                BlazenError::internal("InMemoryContentStore: mutex poisoned during delete")
            })?
            .remove(&handle.id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn put_and_resolve_bytes_yields_base64() {
        let store = InMemoryContentStore::new();
        let handle = store
            .put(
                ContentBody::Bytes {
                    data: vec![1, 2, 3],
                },
                ContentHint::default().with_kind(ContentKind::Image),
            )
            .await
            .unwrap();
        assert_eq!(handle.kind, ContentKind::Image);
        assert_eq!(handle.byte_size, Some(3));
        let resolved = store.resolve(&handle).await.unwrap();
        match resolved {
            MediaSource::Base64 { data } => assert_eq!(data, "AQID"),
            other => panic!("expected base64, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_url_resolves_to_url_directly() {
        let store = InMemoryContentStore::new();
        let handle = store
            .put(
                ContentBody::Url {
                    url: "https://example.com/cat.png".into(),
                },
                ContentHint::default().with_mime_type("image/png"),
            )
            .await
            .unwrap();
        let resolved = store.resolve(&handle).await.unwrap();
        match resolved {
            MediaSource::Url { url } => assert_eq!(url, "https://example.com/cat.png"),
            other => panic!("expected url, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_provider_file_round_trips() {
        let store = InMemoryContentStore::new();
        let handle = store
            .put(
                ContentBody::ProviderFile {
                    provider: ProviderId::OpenAi,
                    id: "file_abc".into(),
                },
                ContentHint::default().with_kind(ContentKind::Document),
            )
            .await
            .unwrap();
        let resolved = store.resolve(&handle).await.unwrap();
        match resolved {
            MediaSource::ProviderFile { provider, id } => {
                assert_eq!(provider, ProviderId::OpenAi);
                assert_eq!(id, "file_abc");
            }
            other => panic!("expected provider file, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn fetch_bytes_only_for_inline_bytes() {
        let store = InMemoryContentStore::new();
        let h = store
            .put(
                ContentBody::Url {
                    url: "https://example.com/x.png".into(),
                },
                ContentHint::default(),
            )
            .await
            .unwrap();
        assert!(store.fetch_bytes(&h).await.is_err());
    }

    #[tokio::test]
    async fn delete_removes_handle() {
        let store = InMemoryContentStore::new();
        let h = store
            .put(
                ContentBody::Bytes { data: vec![0u8] },
                ContentHint::default(),
            )
            .await
            .unwrap();
        store.delete(&h).await.unwrap();
        assert!(store.resolve(&h).await.is_err());
    }

    #[tokio::test]
    async fn stream_body_round_trips_through_in_memory_store() {
        use bytes::Bytes;
        use futures_util::stream;
        let store = InMemoryContentStore::new();
        let chunks = vec![
            Ok(Bytes::from_static(b"hello ")),
            Ok(Bytes::from_static(b"wor")),
            Ok(Bytes::from_static(b"ld")),
        ];
        let body = ContentBody::Stream {
            stream: Box::pin(stream::iter(chunks)),
            size_hint: Some(11),
        };
        let handle = store
            .put(body, ContentHint::default().with_kind(ContentKind::Other))
            .await
            .unwrap();
        let bytes = store.fetch_bytes(&handle).await.unwrap();
        assert_eq!(bytes, b"hello world");
    }
}
