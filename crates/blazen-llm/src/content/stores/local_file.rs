//! [`LocalFileContentStore`] — persists content to a local directory.
//!
//! Each `put` writes the bytes (or copies the source file) to a configured
//! root directory and records the resulting path. `resolve` returns
//! [`MediaSource::File`] referencing the on-disk path; tools running
//! locally can read it directly. For provider request bodies, this is
//! generally **not** the right store — most cloud APIs reject `File`
//! sources. Compose with a provider-file store via
//! [`super::layered::LayeredContentStore`] if you need both.
//!
//! Not available on `wasm32` targets — the browser has no filesystem.

#![cfg(not(target_arch = "wasm32"))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use uuid::Uuid;

use crate::content::detect::{detect, detect_from_path};
use crate::content::handle::ContentHandle;
use crate::content::kind::ContentKind;
use crate::content::store::{ByteStream, ContentBody, ContentHint, ContentStore};
use crate::error::BlazenError;
use crate::types::MediaSource;

/// Filesystem-backed content store.
#[derive(Debug)]
pub struct LocalFileContentStore {
    root: PathBuf,
    /// Maps handle id -> on-disk path. We don't trust the file system to
    /// be the source of truth (paths can move) so the index is in memory.
    index: Mutex<HashMap<String, IndexEntry>>,
}

#[derive(Debug, Clone)]
struct IndexEntry {
    path: PathBuf,
}

impl LocalFileContentStore {
    /// Create a new store rooted at `root`. The directory is created
    /// (recursively) if it doesn't yet exist.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Request`] if the directory cannot be created.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, BlazenError> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(|e| {
            BlazenError::request(format!(
                "LocalFileContentStore: failed to create root '{}': {e}",
                root.display()
            ))
        })?;
        Ok(Self {
            root,
            index: Mutex::new(HashMap::new()),
        })
    }

    fn next_id_and_path(&self, suggested_extension: Option<&str>) -> (String, PathBuf) {
        let raw = Uuid::new_v4().simple().to_string();
        let id = format!("blazen_{}", &raw[..16]);
        let mut path = self.root.join(&id);
        if let Some(ext) = suggested_extension.filter(|s| !s.is_empty()) {
            path.set_extension(ext);
        }
        (id, path)
    }

    fn record(&self, id: String, path: PathBuf) -> Result<(), BlazenError> {
        self.index
            .lock()
            .map_err(|_| BlazenError::internal("LocalFileContentStore: index mutex poisoned"))?
            .insert(id, IndexEntry { path });
        Ok(())
    }

    fn lookup(&self, id: &str) -> Result<IndexEntry, BlazenError> {
        self.index
            .lock()
            .map_err(|_| BlazenError::internal("LocalFileContentStore: index mutex poisoned"))?
            .get(id)
            .cloned()
            .ok_or_else(|| {
                BlazenError::internal(format!("LocalFileContentStore: handle '{id}' not found"))
            })
    }
}

/// Best-effort: derive a file extension hint from a MIME type.
fn mime_to_ext(mime: &str) -> Option<&'static str> {
    match mime.to_ascii_lowercase().as_str() {
        "image/png" => Some("png"),
        "image/jpeg" => Some("jpg"),
        "image/webp" => Some("webp"),
        "image/gif" => Some("gif"),
        "image/svg+xml" => Some("svg"),
        "audio/mpeg" | "audio/mp3" => Some("mp3"),
        "audio/wav" | "audio/x-wav" => Some("wav"),
        "audio/flac" => Some("flac"),
        "audio/ogg" => Some("ogg"),
        "video/mp4" => Some("mp4"),
        "video/webm" => Some("webm"),
        "application/pdf" => Some("pdf"),
        "application/json" => Some("json"),
        "application/zip" => Some("zip"),
        "model/gltf-binary" => Some("glb"),
        "model/gltf+json" => Some("gltf"),
        _ => None,
    }
}

#[async_trait]
impl ContentStore for LocalFileContentStore {
    async fn put(
        &self,
        body: ContentBody,
        hint: ContentHint,
    ) -> Result<ContentHandle, BlazenError> {
        match body {
            ContentBody::Bytes { data: bytes } => {
                let (auto_kind, auto_mime) = detect(
                    Some(&bytes),
                    hint.mime_type.as_deref(),
                    hint.display_name.as_deref(),
                );
                let kind = hint.kind_hint.unwrap_or(auto_kind);
                let mime = hint.mime_type.clone().or(auto_mime);
                let ext = mime.as_deref().and_then(mime_to_ext).or_else(|| {
                    hint.display_name
                        .as_deref()
                        .and_then(|n| Path::new(n).extension().and_then(|e| e.to_str()))
                });
                let (id, path) = self.next_id_and_path(ext);
                std::fs::write(&path, &bytes).map_err(|e| {
                    BlazenError::request(format!(
                        "LocalFileContentStore: write failed for '{}': {e}",
                        path.display()
                    ))
                })?;
                #[allow(clippy::cast_possible_truncation)]
                let size = Some(bytes.len() as u64);
                self.record(id.clone(), path)?;
                let mut handle = ContentHandle::new(id, kind);
                handle.mime_type = mime;
                handle.byte_size = size;
                handle.display_name = hint.display_name;
                Ok(handle)
            }
            ContentBody::LocalPath { path: src } => {
                // Reference, don't copy. The store's index points at the
                // user-supplied path directly.
                let (auto_kind, auto_mime) = detect_from_path(&src);
                let kind = hint.kind_hint.unwrap_or(auto_kind);
                let mime = hint.mime_type.clone().or(auto_mime);
                let id = format!("blazen_{}", &Uuid::new_v4().simple().to_string()[..16]);
                let byte_size = std::fs::metadata(&src)
                    .ok()
                    .map(|m| m.len())
                    .or(hint.byte_size);
                self.record(id.clone(), src.clone())?;
                let mut handle = ContentHandle::new(id, kind);
                handle.mime_type = mime;
                handle.byte_size = byte_size;
                handle.display_name = hint
                    .display_name
                    .or_else(|| src.file_name().map(|n| n.to_string_lossy().into_owned()));
                Ok(handle)
            }
            ContentBody::Url { .. } | ContentBody::ProviderFile { .. } => {
                Err(BlazenError::unsupported(
                    "LocalFileContentStore: only Bytes and LocalPath bodies are supported. \
                     Use InMemoryContentStore or a provider-file store for URL / ProviderFile \
                     references, or fetch the bytes first.",
                ))
            }
            ContentBody::Stream {
                stream,
                size_hint: _,
            } => {
                use futures_util::StreamExt;
                use tokio::io::AsyncWriteExt;

                // We can't sniff bytes ahead of time without buffering, so the
                // path-extension hint comes only from the caller-supplied MIME
                // (or display-name) — fall back to an extension-less filename
                // when neither is provided.
                let ext = hint.mime_type.as_deref().and_then(mime_to_ext).or_else(|| {
                    hint.display_name
                        .as_deref()
                        .and_then(|n| Path::new(n).extension().and_then(|e| e.to_str()))
                });
                let (id, path) = self.next_id_and_path(ext);
                let mut file = tokio::fs::File::create(&path).await.map_err(|e| {
                    BlazenError::request(format!(
                        "LocalFileContentStore: failed to create '{}': {e}",
                        path.display()
                    ))
                })?;
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk?;
                    file.write_all(&chunk).await.map_err(|e| {
                        BlazenError::request(format!(
                            "LocalFileContentStore: write failed for '{}': {e}",
                            path.display()
                        ))
                    })?;
                }
                file.flush().await.map_err(|e| {
                    BlazenError::request(format!(
                        "LocalFileContentStore: flush failed for '{}': {e}",
                        path.display()
                    ))
                })?;
                drop(file);
                let kind = hint.kind_hint.unwrap_or(ContentKind::Other);
                let byte_size = std::fs::metadata(&path).ok().map(|m| m.len());
                self.record(id.clone(), path)?;
                let mut handle = ContentHandle::new(id, kind);
                handle.mime_type = hint.mime_type;
                handle.byte_size = byte_size;
                handle.display_name = hint.display_name;
                Ok(handle)
            }
        }
    }

    async fn resolve(&self, handle: &ContentHandle) -> Result<MediaSource, BlazenError> {
        let entry = self.lookup(&handle.id)?;
        Ok(MediaSource::File { path: entry.path })
    }

    async fn fetch_bytes(&self, handle: &ContentHandle) -> Result<Vec<u8>, BlazenError> {
        let entry = self.lookup(&handle.id)?;
        std::fs::read(&entry.path).map_err(|e| {
            BlazenError::request(format!(
                "LocalFileContentStore: read failed for '{}': {e}",
                entry.path.display()
            ))
        })
    }

    async fn fetch_stream(&self, handle: &ContentHandle) -> Result<ByteStream, BlazenError> {
        use futures_util::StreamExt;

        let entry = self.lookup(&handle.id)?;
        let file = tokio::fs::File::open(&entry.path).await.map_err(|e| {
            BlazenError::request(format!(
                "LocalFileContentStore: failed to open '{}': {e}",
                entry.path.display()
            ))
        })?;
        let path_for_err = entry.path.clone();
        let stream = tokio_util::io::ReaderStream::new(file).map(move |res| {
            res.map_err(|e| {
                BlazenError::request(format!(
                    "LocalFileContentStore: read error for '{}': {e}",
                    path_for_err.display()
                ))
            })
        });
        Ok(Box::pin(stream))
    }

    async fn delete(&self, handle: &ContentHandle) -> Result<(), BlazenError> {
        if let Some(entry) = self
            .index
            .lock()
            .map_err(|_| BlazenError::internal("LocalFileContentStore: index mutex poisoned"))?
            .remove(&handle.id)
        {
            // Best-effort: don't error if file was already removed.
            let _ = std::fs::remove_file(&entry.path);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir() -> PathBuf {
        let raw = Uuid::new_v4().simple().to_string();
        let dir = std::env::temp_dir().join(format!("blazen-content-test-{}", &raw[..12]));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn put_bytes_writes_to_disk_and_resolve_returns_file() {
        let dir = tmp_dir();
        let store = LocalFileContentStore::new(dir.clone()).unwrap();
        // Minimal PNG header so `infer` recognizes the kind.
        let png = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
        ];
        let handle = store
            .put(
                ContentBody::Bytes { data: png.clone() },
                ContentHint::default(),
            )
            .await
            .unwrap();
        assert_eq!(handle.kind, ContentKind::Image);
        assert_eq!(handle.byte_size, Some(png.len() as u64));
        let resolved = store.resolve(&handle).await.unwrap();
        match resolved {
            MediaSource::File { path } => {
                assert!(path.starts_with(&dir));
                let read_back = std::fs::read(&path).unwrap();
                assert_eq!(read_back, png);
            }
            other => panic!("expected file source, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn put_local_path_records_reference_without_copy() {
        let dir = tmp_dir();
        let store = LocalFileContentStore::new(dir.clone()).unwrap();
        let src = dir.join("source.txt");
        std::fs::write(&src, b"hello").unwrap();
        let handle = store
            .put(
                ContentBody::LocalPath { path: src.clone() },
                ContentHint::default(),
            )
            .await
            .unwrap();
        let resolved = store.resolve(&handle).await.unwrap();
        match resolved {
            MediaSource::File { path } => assert_eq!(path, src),
            other => panic!("expected file, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn url_body_is_unsupported() {
        let dir = tmp_dir();
        let store = LocalFileContentStore::new(dir.clone()).unwrap();
        let res = store
            .put(
                ContentBody::Url {
                    url: "https://example.com/x.png".into(),
                },
                ContentHint::default(),
            )
            .await;
        assert!(res.is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn stream_body_round_trips_through_local_file_store() {
        use bytes::Bytes;
        use futures_util::{TryStreamExt, stream};

        let dir = tempfile::tempdir().unwrap();
        let store = LocalFileContentStore::new(dir.path()).unwrap();
        let chunks: Vec<Result<Bytes, BlazenError>> = vec![
            Ok(Bytes::from_static(b"hello ")),
            Ok(Bytes::from_static(b"world")),
        ];
        let body = ContentBody::Stream {
            stream: Box::pin(stream::iter(chunks)),
            size_hint: Some(11),
        };
        let handle = store
            .put(body, ContentHint::default().with_kind(ContentKind::Other))
            .await
            .unwrap();
        assert_eq!(handle.byte_size, Some(11));
        let bytes = store.fetch_bytes(&handle).await.unwrap();
        assert_eq!(bytes, b"hello world");

        let mut s = store.fetch_stream(&handle).await.unwrap();
        let mut got = Vec::new();
        while let Some(chunk) = s.try_next().await.unwrap() {
            got.extend_from_slice(&chunk);
        }
        assert_eq!(got, b"hello world");
    }

    #[tokio::test]
    async fn delete_removes_file() {
        let dir = tmp_dir();
        let store = LocalFileContentStore::new(dir.clone()).unwrap();
        let handle = store
            .put(
                ContentBody::Bytes {
                    data: vec![0u8, 1u8],
                },
                ContentHint::default(),
            )
            .await
            .unwrap();
        let path = match store.resolve(&handle).await.unwrap() {
            MediaSource::File { path } => path,
            other => panic!("expected file, got {other:?}"),
        };
        assert!(path.exists());
        store.delete(&handle).await.unwrap();
        assert!(!path.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
