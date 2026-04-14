//! VRAM budget-aware model manager with LRU eviction.
//!
//! Tracks registered [`LocalModel`] instances and their estimated VRAM
//! footprint. When loading a model that would exceed the configured budget
//! the least-recently-used loaded model is unloaded first.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use blazen_llm::{BlazenError, LocalModel};
use tokio::sync::Mutex;

/// Status of a registered model.
#[derive(Debug, Clone)]
pub struct ModelStatus {
    /// Identifier of the model.
    pub id: String,
    /// Whether the model is currently loaded.
    pub loaded: bool,
    /// Estimated VRAM footprint in bytes.
    pub vram_estimate: u64,
}

struct RegisteredModel {
    model: Arc<dyn LocalModel>,
    vram_estimate: u64,
    loaded: bool,
    last_used: Option<Instant>,
}

/// VRAM budget-aware model manager with LRU eviction.
///
/// Tracks registered local models and their estimated VRAM footprint.
/// When loading a model that would exceed the budget, the least-recently-used
/// loaded model is unloaded first.
pub struct ModelManager {
    budget_bytes: u64,
    state: Mutex<HashMap<String, RegisteredModel>>,
}

impl ModelManager {
    /// Create a new manager with the given VRAM budget in bytes.
    #[must_use]
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            state: Mutex::new(HashMap::new()),
        }
    }

    /// Create a new manager with the given VRAM budget in gigabytes.
    #[must_use]
    pub fn with_budget_gb(gb: f64) -> Self {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bytes = (gb * 1_073_741_824.0) as u64;
        Self::new(bytes)
    }

    /// Register a model with its estimated VRAM footprint.
    /// The model starts in the unloaded state.
    pub async fn register(&self, id: &str, model: Arc<dyn LocalModel>, vram_estimate: u64) {
        let mut state = self.state.lock().await;
        state.insert(
            id.to_owned(),
            RegisteredModel {
                model,
                vram_estimate,
                loaded: false,
                last_used: None,
            },
        );
    }

    /// Load a model, evicting LRU models if the budget would be exceeded.
    ///
    /// Returns an error if:
    /// - The model ID is not registered
    /// - The model's VRAM estimate exceeds the total budget (can never fit)
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Validation`] when the model is not registered or
    /// cannot fit within the budget.  Propagates any error from the underlying
    /// [`LocalModel::load`] or [`LocalModel::unload`] calls.
    ///
    /// # Panics
    ///
    /// Panics if internal state becomes inconsistent (a model ID that was just
    /// verified to exist is no longer present). This cannot happen under normal
    /// operation.
    pub async fn load(&self, id: &str) -> Result<(), BlazenError> {
        let mut state = self.state.lock().await;

        // Check model exists
        let entry = state
            .get(id)
            .ok_or_else(|| BlazenError::validation(format!("model '{id}' is not registered")))?;

        // Already loaded -- just update LRU
        if entry.loaded {
            let entry = state.get_mut(id).expect("checked above");
            entry.last_used = Some(Instant::now());
            return Ok(());
        }

        let needed = entry.vram_estimate;

        // Check if model can ever fit
        if needed > self.budget_bytes {
            return Err(BlazenError::validation(format!(
                "model '{id}' requires {needed} bytes but total budget is only {} bytes",
                self.budget_bytes
            )));
        }

        // Evict LRU models until we have space
        let mut used = Self::used_bytes_inner(&state);
        while used + needed > self.budget_bytes {
            // Find the loaded model with the oldest last_used
            let lru_id = state
                .iter()
                .filter(|(k, v)| v.loaded && k.as_str() != id)
                .min_by_key(|(_, v)| v.last_used)
                .map(|(k, _)| k.clone());

            let Some(lru_id) = lru_id else {
                return Err(BlazenError::validation(format!(
                    "cannot free enough VRAM to load model '{id}' \
                     (need {needed}, used {used}, budget {})",
                    self.budget_bytes
                )));
            };

            // Unload the LRU model
            let lru_model = state
                .get(&lru_id)
                .expect("lru_id came from iteration")
                .model
                .clone();
            // Release lock during unload to avoid holding it across await
            drop(state);
            lru_model.unload().await?;
            state = self.state.lock().await;
            if let Some(e) = state.get_mut(&lru_id) {
                e.loaded = false;
                e.last_used = None;
            }
            used = Self::used_bytes_inner(&state);
        }

        // Load the requested model
        let model = state
            .get(id)
            .expect("checked at top of function")
            .model
            .clone();
        drop(state);
        model.load().await?;
        let mut state = self.state.lock().await;
        if let Some(e) = state.get_mut(id) {
            e.loaded = true;
            e.last_used = Some(Instant::now());
        }

        Ok(())
    }

    /// Unload a model, freeing its VRAM budget.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Validation`] when the model is not registered.
    /// Propagates any error from the underlying [`LocalModel::unload`] call.
    pub async fn unload(&self, id: &str) -> Result<(), BlazenError> {
        let state = self.state.lock().await;
        let entry = state
            .get(id)
            .ok_or_else(|| BlazenError::validation(format!("model '{id}' is not registered")))?;

        if !entry.loaded {
            return Ok(()); // idempotent
        }

        let model = entry.model.clone();
        drop(state);
        model.unload().await?;
        let mut state = self.state.lock().await;
        if let Some(e) = state.get_mut(id) {
            e.loaded = false;
            e.last_used = None;
        }
        Ok(())
    }

    /// Check if a model is currently loaded.
    pub async fn is_loaded(&self, id: &str) -> bool {
        let state = self.state.lock().await;
        state.get(id).is_some_and(|e| e.loaded)
    }

    /// Ensure a model is loaded. If already loaded, updates the LRU timestamp.
    /// If not loaded, loads it (potentially evicting other models).
    ///
    /// # Errors
    ///
    /// See [`Self::load`].
    pub async fn ensure_loaded(&self, id: &str) -> Result<(), BlazenError> {
        self.load(id).await
    }

    /// Total VRAM currently used by loaded models.
    pub async fn used_bytes(&self) -> u64 {
        let state = self.state.lock().await;
        Self::used_bytes_inner(&state)
    }

    /// Available VRAM within the budget.
    pub async fn available_bytes(&self) -> u64 {
        let used = self.used_bytes().await;
        self.budget_bytes.saturating_sub(used)
    }

    /// Status of all registered models.
    pub async fn status(&self) -> Vec<ModelStatus> {
        let state = self.state.lock().await;
        state
            .iter()
            .map(|(id, entry)| ModelStatus {
                id: id.clone(),
                loaded: entry.loaded,
                vram_estimate: entry.vram_estimate,
            })
            .collect()
    }

    fn used_bytes_inner(state: &HashMap<String, RegisteredModel>) -> u64 {
        state
            .values()
            .filter(|e| e.loaded)
            .map(|e| e.vram_estimate)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    const GB: u64 = 1_073_741_824;

    struct MockLocalModel {
        loaded: StdMutex<bool>,
    }

    impl MockLocalModel {
        fn new() -> Self {
            Self {
                loaded: StdMutex::new(false),
            }
        }
    }

    #[async_trait::async_trait]
    impl LocalModel for MockLocalModel {
        async fn load(&self) -> Result<(), BlazenError> {
            *self.loaded.lock().unwrap() = true;
            Ok(())
        }
        async fn unload(&self) -> Result<(), BlazenError> {
            *self.loaded.lock().unwrap() = false;
            Ok(())
        }
        async fn is_loaded(&self) -> bool {
            *self.loaded.lock().unwrap()
        }
    }

    #[tokio::test]
    async fn test_register_and_load() {
        let mgr = ModelManager::new(24 * GB);
        let model = Arc::new(MockLocalModel::new());

        mgr.register("m1", model.clone(), 4 * GB).await;
        mgr.load("m1").await.unwrap();

        assert!(mgr.is_loaded("m1").await);
        assert_eq!(mgr.used_bytes().await, 4 * GB);
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let mgr = ModelManager::new(20 * GB);
        let m1 = Arc::new(MockLocalModel::new());
        let m2 = Arc::new(MockLocalModel::new());
        let m3 = Arc::new(MockLocalModel::new());

        mgr.register("m1", m1.clone(), 4 * GB).await;
        mgr.register("m2", m2.clone(), 8 * GB).await;
        mgr.register("m3", m3.clone(), 12 * GB).await;

        // Load m1 (4GB) then m2 (8GB) -> 12GB used
        mgr.load("m1").await.unwrap();
        mgr.load("m2").await.unwrap();
        assert_eq!(mgr.used_bytes().await, 12 * GB);

        // Load m3 (12GB) -> needs 24GB total, exceeds 20GB budget.
        // m1 is the oldest LRU -> evict m1 (frees 4GB -> 8GB used),
        // still not enough -> but 8+12=20 fits exactly.
        mgr.load("m3").await.unwrap();

        assert!(
            !mgr.is_loaded("m1").await,
            "m1 should have been evicted (oldest LRU)"
        );
        assert!(mgr.is_loaded("m2").await, "m2 should still be loaded");
        assert!(mgr.is_loaded("m3").await, "m3 should now be loaded");
    }

    #[tokio::test]
    async fn test_model_exceeds_budget() {
        let mgr = ModelManager::new(24 * GB);
        let model = Arc::new(MockLocalModel::new());

        mgr.register("big", model.clone(), 32 * GB).await;

        let result = mgr.load("big").await;
        assert!(
            result.is_err(),
            "loading a model larger than the budget must fail"
        );
    }

    #[tokio::test]
    async fn test_ensure_loaded_updates_lru() {
        let mgr = ModelManager::new(20 * GB);
        let m1 = Arc::new(MockLocalModel::new());
        let m2 = Arc::new(MockLocalModel::new());
        let m3 = Arc::new(MockLocalModel::new());

        mgr.register("m1", m1.clone(), 8 * GB).await;
        mgr.register("m2", m2.clone(), 8 * GB).await;
        mgr.register("m3", m3.clone(), 8 * GB).await;

        // Load m1 then m2 -> 16GB used.  m1 is the older LRU entry.
        mgr.load("m1").await.unwrap();
        mgr.load("m2").await.unwrap();

        // Touch m1 via ensure_loaded -> updates its LRU timestamp so m2
        // becomes the oldest.
        mgr.ensure_loaded("m1").await.unwrap();

        // Load m3 (8GB) -> needs to evict 4GB.  m2 is now the oldest LRU.
        mgr.load("m3").await.unwrap();

        assert!(
            mgr.is_loaded("m1").await,
            "m1 was touched via ensure_loaded and must NOT be evicted"
        );
        assert!(
            !mgr.is_loaded("m2").await,
            "m2 should have been evicted (oldest LRU after m1 was touched)"
        );
        assert!(mgr.is_loaded("m3").await, "m3 should now be loaded");
    }

    #[tokio::test]
    async fn test_unload_frees_budget() {
        let mgr = ModelManager::new(24 * GB);
        let model = Arc::new(MockLocalModel::new());

        mgr.register("m1", model.clone(), 8 * GB).await;
        mgr.load("m1").await.unwrap();
        assert_eq!(mgr.used_bytes().await, 8 * GB);

        mgr.unload("m1").await.unwrap();
        assert_eq!(mgr.used_bytes().await, 0);
    }

    #[tokio::test]
    async fn test_status() {
        let mgr = ModelManager::new(24 * GB);
        let m1 = Arc::new(MockLocalModel::new());
        let m2 = Arc::new(MockLocalModel::new());

        mgr.register("m1", m1.clone(), 4 * GB).await;
        mgr.register("m2", m2.clone(), 8 * GB).await;

        mgr.load("m1").await.unwrap();

        let statuses = mgr.status().await;
        assert_eq!(statuses.len(), 2);

        let s1 = statuses.iter().find(|s| s.id == "m1").expect("m1 missing");
        let s2 = statuses.iter().find(|s| s.id == "m2").expect("m2 missing");

        assert!(s1.loaded, "m1 should be loaded");
        assert_eq!(s1.vram_estimate, 4 * GB);

        assert!(!s2.loaded, "m2 should not be loaded");
        assert_eq!(s2.vram_estimate, 8 * GB);
    }
}
