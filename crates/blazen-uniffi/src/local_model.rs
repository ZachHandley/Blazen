//! Foreign-implementable `LocalModel` callback interface.
//!
//! Exposes [`ForeignLocalModel`] as a UniFFI-with-foreign trait so Go, Swift,
//! Kotlin, and Ruby callers can write their own `LocalModel` implementations
//! and register them with the [`crate::manager::UniffiModelManager`]. The
//! Python binding's `PyLocalModelWrapper` (`crates/blazen-py/src/manager.rs`)
//! is the analogue surface.
//!
//! A private [`ForeignLocalModelAdapter`] bridges the FFI-friendly
//! [`ForeignLocalModel`] back to the upstream [`blazen_llm::LocalModel`]
//! trait that [`blazen_manager::ModelManager`] consumes. Wire-format records
//! ([`AdapterOptionsRecord`], [`AdapterStatusRecord`], [`AdapterHandleRecord`])
//! flatten the upstream `PathBuf` / enum types into UniFFI-compatible shapes
//! (`String` paths, `String` discriminators).

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use crate::errors::{BlazenError, BlazenResult};

/// Adapter mount options handed to [`ForeignLocalModel::load_adapter`].
///
/// Mirrors [`blazen_llm::AdapterOptions`] but lives as a UniFFI Record so
/// foreign code can construct it natively.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AdapterOptionsRecord {
    pub adapter_id: String,
    pub scale: f32,
}

impl From<AdapterOptionsRecord> for blazen_llm::AdapterOptions {
    fn from(r: AdapterOptionsRecord) -> Self {
        Self {
            adapter_id: r.adapter_id,
            scale: r.scale,
        }
    }
}

impl From<blazen_llm::AdapterOptions> for AdapterOptionsRecord {
    fn from(o: blazen_llm::AdapterOptions) -> Self {
        Self {
            adapter_id: o.adapter_id,
            scale: o.scale,
        }
    }
}

/// Result returned by [`ForeignLocalModel::load_adapter`], mirroring
/// [`blazen_llm::AdapterHandle`].
///
/// `mount_strategy` is one of `"attached"`, `"rebuilt"`, `"merged"` — kept
/// as a string discriminator so adding a new strategy to the upstream enum
/// does not break the FFI contract.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AdapterHandleRecord {
    pub adapter_id: String,
    pub memory_bytes: u64,
    pub mount_strategy: String,
}

impl From<blazen_llm::AdapterHandle> for AdapterHandleRecord {
    fn from(h: blazen_llm::AdapterHandle) -> Self {
        Self {
            adapter_id: h.adapter_id,
            memory_bytes: h.memory_bytes,
            mount_strategy: mount_strategy_to_str(h.mount_strategy).to_owned(),
        }
    }
}

impl From<AdapterHandleRecord> for blazen_llm::AdapterHandle {
    fn from(r: AdapterHandleRecord) -> Self {
        Self {
            adapter_id: r.adapter_id,
            memory_bytes: r.memory_bytes,
            mount_strategy: mount_strategy_from_str(&r.mount_strategy),
        }
    }
}

/// Snapshot of a single mounted adapter — wire form of
/// [`blazen_llm::AdapterStatus`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct AdapterStatusRecord {
    pub adapter_id: String,
    pub scale: f32,
    pub source_dir: String,
    pub memory_bytes: u64,
}

impl From<blazen_llm::AdapterStatus> for AdapterStatusRecord {
    fn from(s: blazen_llm::AdapterStatus) -> Self {
        Self {
            adapter_id: s.adapter_id,
            scale: s.scale,
            source_dir: s.source_dir.display().to_string(),
            memory_bytes: s.memory_bytes,
        }
    }
}

impl From<AdapterStatusRecord> for blazen_llm::AdapterStatus {
    fn from(r: AdapterStatusRecord) -> Self {
        Self {
            adapter_id: r.adapter_id,
            scale: r.scale,
            source_dir: PathBuf::from(r.source_dir),
            memory_bytes: r.memory_bytes,
        }
    }
}

fn mount_strategy_to_str(s: blazen_llm::AdapterMountStrategy) -> &'static str {
    match s {
        blazen_llm::AdapterMountStrategy::Attached => "attached",
        blazen_llm::AdapterMountStrategy::Rebuilt => "rebuilt",
        blazen_llm::AdapterMountStrategy::Merged => "merged",
    }
}

fn mount_strategy_from_str(s: &str) -> blazen_llm::AdapterMountStrategy {
    match s.to_ascii_lowercase().as_str() {
        "rebuilt" => blazen_llm::AdapterMountStrategy::Rebuilt,
        "merged" => blazen_llm::AdapterMountStrategy::Merged,
        _ => blazen_llm::AdapterMountStrategy::Attached,
    }
}

/// Foreign-language implementation of a local (on-device) model.
///
/// Implementors mirror the upstream [`blazen_llm::LocalModel`] trait but in
/// FFI-friendly form: paths are `String`, the `device()` accessor returns a
/// `String` ("cpu", "cuda:0", "metal", ...) that gets parsed back into
/// [`blazen_llm::Device`] when forwarded to the manager.
///
/// `is_loaded`, `memory_bytes`, `device`, `load_adapter`, `unload_adapter`,
/// and `list_adapters` are NOT optional on this trait — UniFFI does not have a
/// concept of "default trait method" that the foreign side can opt out of.
/// Foreign implementors that don't care about a verb should return a sensible
/// neutral value (`false` for `is_loaded`, `0` / `None` for `memory_bytes`,
/// `"cpu"` for `device`, an empty `list_adapters`, or raise
/// [`BlazenError::Unsupported`] from the adapter verbs).
#[uniffi::export(with_foreign)]
#[async_trait]
pub trait ForeignLocalModel: Send + Sync {
    async fn load(&self) -> BlazenResult<()>;

    async fn unload(&self) -> BlazenResult<()>;

    async fn is_loaded(&self) -> bool;

    fn device(&self) -> String;

    async fn memory_bytes(&self) -> Option<u64>;

    async fn load_adapter(
        &self,
        adapter_dir: String,
        options: AdapterOptionsRecord,
    ) -> BlazenResult<AdapterHandleRecord>;

    async fn unload_adapter(&self, handle: AdapterHandleRecord) -> BlazenResult<()>;

    async fn list_adapters(&self) -> Vec<AdapterStatusRecord>;
}

/// Bridges an `Arc<dyn ForeignLocalModel>` (FFI side) into the upstream
/// [`blazen_llm::LocalModel`] trait expected by
/// [`blazen_manager::ModelManager`].
pub(crate) struct ForeignLocalModelAdapter {
    inner: Arc<dyn ForeignLocalModel>,
    // Why: device() is sync on LocalModel but we want to evaluate it lazily
    // and cache it — the foreign call can be expensive (Kotlin JNI hop) and
    // the manager queries device() once at register() time anyway.
    cached_device: parking_lot::Mutex<Option<blazen_llm::Device>>,
}

impl ForeignLocalModelAdapter {
    pub(crate) fn new(inner: Arc<dyn ForeignLocalModel>) -> Self {
        Self {
            inner,
            cached_device: parking_lot::Mutex::new(None),
        }
    }
}

#[async_trait]
impl blazen_llm::LocalModel for ForeignLocalModelAdapter {
    async fn load(&self) -> Result<(), blazen_llm::BlazenError> {
        self.inner.load().await.map_err(blazen_error_to_core)
    }

    async fn unload(&self) -> Result<(), blazen_llm::BlazenError> {
        self.inner.unload().await.map_err(blazen_error_to_core)
    }

    async fn is_loaded(&self) -> bool {
        self.inner.is_loaded().await
    }

    fn device(&self) -> blazen_llm::Device {
        if let Some(d) = self.cached_device.lock().clone() {
            return d;
        }
        let raw = self.inner.device();
        let parsed = blazen_llm::Device::parse(&raw).unwrap_or(blazen_llm::Device::Cpu);
        *self.cached_device.lock() = Some(parsed.clone());
        parsed
    }

    async fn memory_bytes(&self) -> Option<u64> {
        self.inner.memory_bytes().await
    }

    async fn load_adapter(
        &self,
        adapter_dir: &std::path::Path,
        options: blazen_llm::AdapterOptions,
    ) -> Result<blazen_llm::AdapterHandle, blazen_llm::BlazenError> {
        let path = adapter_dir.display().to_string();
        let record = self
            .inner
            .load_adapter(path, AdapterOptionsRecord::from(options))
            .await
            .map_err(blazen_error_to_core)?;
        Ok(record.into())
    }

    async fn unload_adapter(
        &self,
        handle: &blazen_llm::AdapterHandle,
    ) -> Result<(), blazen_llm::BlazenError> {
        let record = AdapterHandleRecord {
            adapter_id: handle.adapter_id.clone(),
            memory_bytes: handle.memory_bytes,
            mount_strategy: mount_strategy_to_str(handle.mount_strategy).to_owned(),
        };
        self.inner
            .unload_adapter(record)
            .await
            .map_err(blazen_error_to_core)
    }

    async fn list_adapters(&self) -> Vec<blazen_llm::AdapterStatus> {
        self.inner
            .list_adapters()
            .await
            .into_iter()
            .map(Into::into)
            .collect()
    }
}

/// Convert a UniFFI-surface [`BlazenError`] back into the upstream
/// [`blazen_llm::BlazenError`] when forwarding from a foreign-implemented
/// trait into the Rust callsite (`blazen_manager`'s LocalModel hooks).
///
/// We only need a lossy mapping: the manager surfaces these errors back
/// out via [`crate::errors::BlazenError::from`], which already handles the
/// reverse direction faithfully. For variants without an exact upstream
/// counterpart, fall back to `provider("foreign_local_model", ...)`.
fn blazen_error_to_core(err: BlazenError) -> blazen_llm::BlazenError {
    match err {
        BlazenError::Auth { message } => blazen_llm::BlazenError::Auth { message },
        BlazenError::RateLimit { retry_after_ms, .. } => {
            blazen_llm::BlazenError::RateLimit { retry_after_ms }
        }
        BlazenError::Timeout { elapsed_ms, .. } => blazen_llm::BlazenError::Timeout { elapsed_ms },
        BlazenError::Validation { message } => blazen_llm::BlazenError::validation(message),
        BlazenError::ContentPolicy { message } => {
            blazen_llm::BlazenError::ContentPolicy { message }
        }
        BlazenError::Unsupported { message } => blazen_llm::BlazenError::unsupported(message),
        BlazenError::Tool { message } => blazen_llm::BlazenError::Tool {
            name: None,
            message,
        },
        other => blazen_llm::BlazenError::provider("foreign_local_model", other.to_string()),
    }
}
