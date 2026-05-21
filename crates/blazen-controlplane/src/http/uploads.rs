//! In-memory chunked upload store.
//!
//! REST endpoints accept either inline payloads (small adapter blobs,
//! audio under ~100 MB, image generations) or multi-part uploads
//! addressed by a content-store handle. The handle is opaque to the
//! caller — internally it's a UUID v4 that maps into a [`DashMap`] of
//! [`StoredBlob`]s.
//!
//! ## Lifecycle
//!
//! 1. Caller posts to `POST /v1/blazen/content` with `multipart/form-data`
//!    containing a `file` field. The server reads the entire body into a
//!    `StoredBlob` and returns the handle.
//! 2. Caller references the handle in a subsequent admin or RPC body
//!    (e.g. the `adapter` payload of
//!    `POST /v1/blazen/adapters/{model_id}/load`).
//! 3. The route resolves the handle, hands the bytes off to the
//!    [`crate::server::model_manager::ManagerHandle`], and (optionally)
//!    deletes the entry once the underlying RPC has acknowledged.
//!
//! ## Limits
//!
//! `ContentStore` is a pure in-memory backing for the MVP. Production
//! deployments that need spillover-to-disk or per-tenant quotas should
//! wrap the store behind their own trait — the REST routes only depend
//! on the inherent `put` / `get` / `delete` methods.

use std::sync::Arc;

use dashmap::DashMap;
use uuid::Uuid;

/// A single uploaded blob.
#[derive(Clone)]
pub struct StoredBlob {
    /// Original filename if the multipart part declared one.
    pub filename: Option<String>,
    /// MIME type the client declared, or `application/octet-stream`.
    pub content_type: String,
    /// Raw bytes.
    pub data: Arc<Vec<u8>>,
}

impl std::fmt::Debug for StoredBlob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredBlob")
            .field("filename", &self.filename)
            .field("content_type", &self.content_type)
            .field("len", &self.data.len())
            .finish()
    }
}

impl StoredBlob {
    /// Number of bytes in the stored payload.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Returns `true` for a zero-length payload.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Concurrent in-memory map of UUID handles -> `StoredBlob`.
#[derive(Default)]
pub struct ContentStore {
    inner: DashMap<Uuid, StoredBlob>,
}

impl ContentStore {
    /// Insert a blob and return its handle.
    #[must_use]
    pub fn put(&self, blob: StoredBlob) -> Uuid {
        let id = Uuid::new_v4();
        self.inner.insert(id, blob);
        id
    }

    /// Look up a blob without removing it.
    #[must_use]
    pub fn get(&self, id: Uuid) -> Option<StoredBlob> {
        self.inner.get(&id).map(|r| r.clone())
    }

    /// Atomically remove and return a blob.
    #[must_use]
    pub fn take(&self, id: Uuid) -> Option<StoredBlob> {
        self.inner.remove(&id).map(|(_, blob)| blob)
    }

    /// Drop a blob, ignoring whether it existed.
    pub fn delete(&self, id: Uuid) {
        self.inner.remove(&id);
    }

    /// Total number of stored blobs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` when no blobs are currently stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Drain a `multipart/form-data` field whose declared name matches
/// `field_name`. Returns the first matching part. Other parts are
/// silently discarded.
///
/// # Errors
///
/// Returns [`super::error::HttpError::BadRequest`] if the multipart
/// body is malformed, the requested field is absent, or a part's body
/// cannot be read.
pub async fn read_multipart_file(
    mut multipart: axum::extract::Multipart,
    field_name: &str,
) -> Result<StoredBlob, super::error::HttpError> {
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| super::error::HttpError::bad_request(format!("multipart: {e}")))?
    {
        let name = field.name().map(str::to_owned);
        if name.as_deref() != Some(field_name) {
            continue;
        }
        let filename = field.file_name().map(str::to_owned);
        let content_type = field
            .content_type()
            .map_or_else(|| "application/octet-stream".to_owned(), str::to_owned);
        let data = field
            .bytes()
            .await
            .map_err(|e| super::error::HttpError::bad_request(format!("multipart bytes: {e}")))?;
        return Ok(StoredBlob {
            filename,
            content_type,
            data: Arc::new(data.to_vec()),
        });
    }
    Err(super::error::HttpError::bad_request(format!(
        "missing multipart field '{field_name}'"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_take_roundtrip() {
        let store = ContentStore::default();
        let id = store.put(StoredBlob {
            filename: Some("a.bin".into()),
            content_type: "application/octet-stream".into(),
            data: Arc::new(vec![1, 2, 3]),
        });
        let got = store.get(id).expect("get");
        assert_eq!(got.len(), 3);
        let taken = store.take(id).expect("take");
        assert_eq!(taken.data.as_slice(), &[1, 2, 3]);
        assert!(store.get(id).is_none(), "take should remove");
        assert!(store.is_empty());
    }

    #[test]
    fn delete_is_idempotent() {
        let store = ContentStore::default();
        let id = Uuid::new_v4();
        store.delete(id);
        store.delete(id);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn stored_blob_helpers() {
        let blob = StoredBlob {
            filename: None,
            content_type: "x".into(),
            data: Arc::new(vec![]),
        };
        assert_eq!(blob.len(), 0);
        assert!(blob.is_empty());
    }
}
