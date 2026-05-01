//! Node.js binding for the VRAM-aware model manager.

use napi::bindgen_prelude::BigInt;
use napi_derive::napi;
use std::sync::Arc;

use blazen_manager::ModelManager;

use crate::providers::completion_model::JsCompletionModel;

/// Configuration for creating a [`JsModelManager`].
///
/// Exactly one of `budgetGb` or `budgetBytes` must be provided.
#[napi(object)]
pub struct ModelManagerConfig {
    /// VRAM budget in gigabytes (e.g. `8.0` for 8 GiB).
    pub budget_gb: Option<f64>,
    /// VRAM budget in bytes (pass as JS `BigInt` to support values >4 GiB).
    pub budget_bytes: Option<BigInt>,
}

/// Status snapshot for a single registered model.
#[napi(object)]
pub struct JsModelStatus {
    /// Model identifier.
    pub id: String,
    /// Whether the model is currently loaded into VRAM.
    pub loaded: bool,
    /// Estimated VRAM footprint in bytes.
    pub vram_estimate: BigInt,
}

/// VRAM budget-aware model manager with LRU eviction.
///
/// Tracks registered local models and their estimated VRAM footprint.
/// When loading a model that would exceed the budget, the least-recently-used
/// loaded model is unloaded first.
///
/// ```javascript
/// const manager = new ModelManager({ budgetGb: 8.0 });
/// await manager.register("llama-7b", model, 4_000_000_000n);  // BigInt
/// await manager.load("llama-7b");
/// ```
#[napi(js_name = "ModelManager")]
pub struct JsModelManager {
    inner: Arc<ModelManager>,
}

#[napi]
#[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
impl JsModelManager {
    /// Create a new model manager with the given VRAM budget.
    ///
    /// Provide either `budgetGb` (gigabytes as a float) or `budgetBytes`
    /// (exact byte count). If both are given, `budgetGb` takes precedence.
    #[napi(constructor)]
    pub fn new(config: ModelManagerConfig) -> napi::Result<Self> {
        let bytes = match (config.budget_gb, config.budget_bytes) {
            (Some(gb), _) => {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                {
                    (gb * 1_073_741_824.0) as u64
                }
            }
            (_, Some(b)) => b.get_u64().1,
            (None, None) => {
                return Err(napi::Error::from_reason(
                    "must provide either budgetGb or budgetBytes",
                ));
            }
        };
        Ok(Self {
            inner: Arc::new(ModelManager::new(bytes)),
        })
    }

    /// Register a model with the manager.
    ///
    /// The model starts in the unloaded state.  An optional
    /// `vramEstimateBytes` overrides the model's self-reported estimate.
    ///
    /// Only local in-process providers (mistral.rs, llama.cpp, candle) can be
    /// registered -- remote HTTP providers will throw.
    #[napi]
    pub async fn register(
        &self,
        id: String,
        model: &JsCompletionModel,
        vram_estimate_bytes: Option<BigInt>,
    ) -> napi::Result<()> {
        let local_model = model
            .local_model
            .clone()
            .ok_or_else(|| napi::Error::from_reason("model does not support local loading"))?;
        let vram = vram_estimate_bytes.map_or(0, |b| b.get_u64().1);
        self.inner.register(&id, local_model, vram).await;
        Ok(())
    }

    /// Load a model, evicting LRU models if the budget would be exceeded.
    ///
    /// Throws if the model is not registered or its VRAM estimate exceeds the
    /// total budget.
    #[napi]
    pub async fn load(&self, id: String) -> napi::Result<()> {
        self.inner
            .load(&id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Unload a model, freeing its VRAM budget.
    ///
    /// Idempotent: unloading an already-unloaded model is a no-op.
    #[napi]
    pub async fn unload(&self, id: String) -> napi::Result<()> {
        self.inner
            .unload(&id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Check whether a model is currently loaded.
    #[napi(js_name = "isLoaded")]
    pub async fn is_loaded(&self, id: String) -> bool {
        self.inner.is_loaded(&id).await
    }

    /// Ensure a model is loaded.
    ///
    /// If already loaded, updates the LRU timestamp. If not loaded, loads it
    /// (potentially evicting other models).
    #[napi(js_name = "ensureLoaded")]
    pub async fn ensure_loaded(&self, id: String) -> napi::Result<()> {
        self.inner
            .ensure_loaded(&id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Total VRAM currently used by loaded models (in bytes).
    #[napi(js_name = "usedBytes")]
    pub async fn used_bytes(&self) -> BigInt {
        BigInt::from(self.inner.used_bytes().await)
    }

    /// Available VRAM within the budget (in bytes).
    #[napi(js_name = "availableBytes")]
    pub async fn available_bytes(&self) -> BigInt {
        BigInt::from(self.inner.available_bytes().await)
    }

    /// Status of all registered models.
    #[napi]
    pub async fn status(&self) -> Vec<JsModelStatus> {
        self.inner
            .status()
            .await
            .into_iter()
            .map(|s| JsModelStatus {
                id: s.id,
                loaded: s.loaded,
                vram_estimate: BigInt::from(s.vram_estimate),
            })
            .collect()
    }
}
