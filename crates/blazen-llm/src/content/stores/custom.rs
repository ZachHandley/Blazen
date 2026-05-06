//! [`CustomContentStore`] — wraps user-supplied async callbacks.
//!
//! Use this when you have your own storage backend (S3, GCS, Postgres LO,
//! a custom CDN, etc.) and want to plug it into Blazen without writing a
//! full [`ContentStore`] implementation. Provide three callbacks — `put`,
//! `resolve`, `fetch_bytes` — and the wrapper handles the trait wiring.
//!
//! All callbacks must return [`futures_util::future::BoxFuture`] so the
//! store can be `dyn`-friendly.

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::future::BoxFuture;

use crate::content::handle::ContentHandle;
use crate::content::store::{ContentBody, ContentHint, ContentStore};
use crate::error::BlazenError;
use crate::types::MediaSource;

type PutFn = Arc<
    dyn Fn(ContentBody, ContentHint) -> BoxFuture<'static, Result<ContentHandle, BlazenError>>
        + Send
        + Sync,
>;
type ResolveFn = Arc<
    dyn Fn(ContentHandle) -> BoxFuture<'static, Result<MediaSource, BlazenError>> + Send + Sync,
>;
type FetchFn =
    Arc<dyn Fn(ContentHandle) -> BoxFuture<'static, Result<Vec<u8>, BlazenError>> + Send + Sync>;
type DeleteFn =
    Arc<dyn Fn(ContentHandle) -> BoxFuture<'static, Result<(), BlazenError>> + Send + Sync>;
pub type FetchStreamFn = Arc<
    dyn Fn(
            ContentHandle,
        ) -> BoxFuture<'static, Result<crate::content::store::ByteStream, BlazenError>>
        + Send
        + Sync,
>;

/// User-defined content store backed by callbacks.
///
/// Construct via [`CustomContentStore::builder`]. `put`, `resolve`, and
/// `fetch_bytes` are required; `delete` defaults to a no-op.
pub struct CustomContentStore {
    name: String,
    put: PutFn,
    resolve: ResolveFn,
    fetch_bytes: FetchFn,
    fetch_stream: Option<FetchStreamFn>,
    delete: Option<DeleteFn>,
}

impl std::fmt::Debug for CustomContentStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomContentStore")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl CustomContentStore {
    /// Start building a custom store. The `name` is used in error /
    /// tracing messages; choose something identifying like
    /// `"s3://my-bucket/blazen-content"`.
    pub fn builder(name: impl Into<String>) -> CustomContentStoreBuilder {
        CustomContentStoreBuilder {
            name: name.into(),
            put: None,
            resolve: None,
            fetch_bytes: None,
            fetch_stream: None,
            delete: None,
        }
    }
}

#[async_trait]
impl ContentStore for CustomContentStore {
    async fn put(
        &self,
        body: ContentBody,
        hint: ContentHint,
    ) -> Result<ContentHandle, BlazenError> {
        (self.put)(body, hint).await
    }

    async fn resolve(&self, handle: &ContentHandle) -> Result<MediaSource, BlazenError> {
        (self.resolve)(handle.clone()).await
    }

    async fn fetch_bytes(&self, handle: &ContentHandle) -> Result<Vec<u8>, BlazenError> {
        (self.fetch_bytes)(handle.clone()).await
    }

    async fn fetch_stream(
        &self,
        handle: &ContentHandle,
    ) -> Result<crate::content::store::ByteStream, BlazenError> {
        if let Some(f) = &self.fetch_stream {
            f(handle.clone()).await
        } else {
            // Fall back to default-impl behavior: buffer fetch_bytes.
            let bytes = self.fetch_bytes(handle).await?;
            Ok(Box::pin(futures_util::stream::once(async move {
                Ok(bytes::Bytes::from(bytes))
            })))
        }
    }

    async fn delete(&self, handle: &ContentHandle) -> Result<(), BlazenError> {
        if let Some(delete) = &self.delete {
            (delete)(handle.clone()).await
        } else {
            Ok(())
        }
    }
}

/// Builder for [`CustomContentStore`].
#[must_use]
pub struct CustomContentStoreBuilder {
    name: String,
    put: Option<PutFn>,
    resolve: Option<ResolveFn>,
    fetch_bytes: Option<FetchFn>,
    fetch_stream: Option<FetchStreamFn>,
    delete: Option<DeleteFn>,
}

impl CustomContentStoreBuilder {
    /// Set the `put` callback.
    pub fn put<F>(mut self, f: F) -> Self
    where
        F: Fn(ContentBody, ContentHint) -> BoxFuture<'static, Result<ContentHandle, BlazenError>>
            + Send
            + Sync
            + 'static,
    {
        self.put = Some(Arc::new(f));
        self
    }

    /// Set the `resolve` callback.
    pub fn resolve<F>(mut self, f: F) -> Self
    where
        F: Fn(ContentHandle) -> BoxFuture<'static, Result<MediaSource, BlazenError>>
            + Send
            + Sync
            + 'static,
    {
        self.resolve = Some(Arc::new(f));
        self
    }

    /// Set the `fetch_bytes` callback.
    pub fn fetch_bytes<F>(mut self, f: F) -> Self
    where
        F: Fn(ContentHandle) -> BoxFuture<'static, Result<Vec<u8>, BlazenError>>
            + Send
            + Sync
            + 'static,
    {
        self.fetch_bytes = Some(Arc::new(f));
        self
    }

    /// Set the `delete` callback. Optional — default behavior is a no-op.
    pub fn delete<F>(mut self, f: F) -> Self
    where
        F: Fn(ContentHandle) -> BoxFuture<'static, Result<(), BlazenError>> + Send + Sync + 'static,
    {
        self.delete = Some(Arc::new(f));
        self
    }

    /// Set the `fetch_stream` callback. Optional — when absent, the trait's
    /// default impl buffers `fetch_bytes` into a single chunk.
    pub fn fetch_stream<F>(mut self, f: F) -> Self
    where
        F: Fn(
                ContentHandle,
            )
                -> BoxFuture<'static, Result<crate::content::store::ByteStream, BlazenError>>
            + Send
            + Sync
            + 'static,
    {
        self.fetch_stream = Some(Arc::new(f));
        self
    }

    /// Finalize the builder.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Request`] if any of `put` / `resolve` /
    /// `fetch_bytes` were not provided.
    pub fn build(self) -> Result<CustomContentStore, BlazenError> {
        Ok(CustomContentStore {
            name: self.name,
            put: self.put.ok_or_else(|| {
                BlazenError::request("CustomContentStore::builder: missing `put` callback")
            })?,
            resolve: self.resolve.ok_or_else(|| {
                BlazenError::request("CustomContentStore::builder: missing `resolve` callback")
            })?,
            fetch_bytes: self.fetch_bytes.ok_or_else(|| {
                BlazenError::request("CustomContentStore::builder: missing `fetch_bytes` callback")
            })?,
            fetch_stream: self.fetch_stream,
            delete: self.delete,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::kind::ContentKind;
    use futures_util::FutureExt;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn custom_store_round_trips_via_callbacks() {
        let put_count = Arc::new(AtomicUsize::new(0));
        let resolve_count = Arc::new(AtomicUsize::new(0));

        let put_count_c = Arc::clone(&put_count);
        let resolve_count_c = Arc::clone(&resolve_count);

        let store = CustomContentStore::builder("test")
            .put(move |_body, _hint| {
                put_count_c.fetch_add(1, Ordering::SeqCst);
                async move { Ok(ContentHandle::new("custom_xyz", ContentKind::Image)) }.boxed()
            })
            .resolve(move |h| {
                resolve_count_c.fetch_add(1, Ordering::SeqCst);
                async move {
                    assert_eq!(h.id, "custom_xyz");
                    Ok(MediaSource::Url {
                        url: "https://example.com/x.png".into(),
                    })
                }
                .boxed()
            })
            .fetch_bytes(|_h| async move { Ok(vec![1, 2, 3]) }.boxed())
            .build()
            .unwrap();

        let handle = store
            .put(
                ContentBody::Bytes { data: vec![0u8] },
                ContentHint::default(),
            )
            .await
            .unwrap();
        assert_eq!(handle.id, "custom_xyz");
        assert_eq!(put_count.load(Ordering::SeqCst), 1);

        let resolved = store.resolve(&handle).await.unwrap();
        assert!(matches!(resolved, MediaSource::Url { .. }));
        assert_eq!(resolve_count.load(Ordering::SeqCst), 1);

        let bytes = store.fetch_bytes(&handle).await.unwrap();
        assert_eq!(bytes, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn missing_required_callback_errors() {
        let res = CustomContentStore::builder("test")
            .resolve(|_h| async { Ok(MediaSource::Url { url: "x".into() }) }.boxed())
            .fetch_bytes(|_h| async { Ok(vec![]) }.boxed())
            .build();
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn custom_store_streaming_fetch_callback() {
        use bytes::Bytes;
        use futures_util::TryStreamExt;
        use futures_util::stream;
        let store = CustomContentStore::builder("test")
            .put(|_body, _hint| {
                Box::pin(async {
                    Ok(crate::content::handle::ContentHandle::new(
                        "h_test",
                        crate::content::kind::ContentKind::Other,
                    ))
                })
            })
            .resolve(|_h| {
                Box::pin(async { Ok(crate::types::MediaSource::Url { url: "x".into() }) })
            })
            .fetch_bytes(|_h| Box::pin(async { Ok(b"buffered".to_vec()) }))
            .fetch_stream(|_h| {
                Box::pin(async {
                    let chunks = vec![
                        Ok(Bytes::from_static(b"chunk1 ")),
                        Ok(Bytes::from_static(b"chunk2")),
                    ];
                    Ok(Box::pin(stream::iter(chunks)) as crate::content::store::ByteStream)
                })
            })
            .build()
            .unwrap();
        let handle = crate::content::handle::ContentHandle::new(
            "h_test",
            crate::content::kind::ContentKind::Other,
        );
        let mut s = store.fetch_stream(&handle).await.unwrap();
        let mut got = Vec::new();
        while let Some(chunk) = s.try_next().await.unwrap() {
            got.extend_from_slice(&chunk);
        }
        assert_eq!(got, b"chunk1 chunk2");
    }

    #[tokio::test]
    async fn custom_store_streaming_fetch_falls_back_to_bytes() {
        use futures_util::TryStreamExt;
        let store = CustomContentStore::builder("test")
            .put(|_body, _hint| {
                Box::pin(async {
                    Ok(crate::content::handle::ContentHandle::new(
                        "h_test",
                        crate::content::kind::ContentKind::Other,
                    ))
                })
            })
            .resolve(|_h| {
                Box::pin(async { Ok(crate::types::MediaSource::Url { url: "x".into() }) })
            })
            .fetch_bytes(|_h| Box::pin(async { Ok(b"all bytes".to_vec()) }))
            // No .fetch_stream — should fall back to fetch_bytes.
            .build()
            .unwrap();
        let handle = crate::content::handle::ContentHandle::new(
            "h_test",
            crate::content::kind::ContentKind::Other,
        );
        let mut s = store.fetch_stream(&handle).await.unwrap();
        let mut got = Vec::new();
        while let Some(chunk) = s.try_next().await.unwrap() {
            got.extend_from_slice(&chunk);
        }
        assert_eq!(got, b"all bytes");
    }
}
