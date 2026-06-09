//! Provider-selection factory with an optional local-model fallback seam.
//!
//! [`build_model`] is the single entry point that decides — given a provider
//! name, [`ProviderOptions`], and a [`FallbackPolicy`] — whether to construct a
//! *remote* (HTTP API) provider or route to a *locally-servable* model.
//!
//! ## Layering
//!
//! Local execution is **injected** via the [`LocalModelProbe`] and
//! [`LocalModelFactory`] traits. `blazen-llm` deliberately does NOT depend on
//! `blazen-controlplane`: the dependency edge runs controlplane → llm, never the
//! reverse. The controlplane-side adapter (which wraps its `ManagerHandle` to
//! actually serve a local model) implements these traits and is supplied by the
//! caller. Standalone `blazen-llm` ships [`NoLocalModels`] (probe that always
//! says "not servable"), so the default behaviour is byte-identical to a plain
//! remote build.
//!
//! ## Selection logic
//!
//! 1. Resolve the API key via [`resolve_api_key`](crate::keys::resolve_api_key)
//!    (explicit → resolver chain → env). On success the **remote** provider is
//!    built, regardless of policy — a present key always wins.
//! 2. On an [`Auth`](crate::BlazenError::Auth) miss, if the policy permits a
//!    fallback (anything other than [`FallbackPolicy::Never`]) AND the probe
//!    reports the model is locally servable, the [`LocalModelFactory`] builds
//!    the local model.
//! 3. Otherwise the original key-resolution error is propagated.
//!
//! Any non-`Auth` error from key resolution (there are none today, but the
//! contract is future-proof) is propagated unchanged without consulting the
//! local seam.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::BlazenError;
use crate::keys::resolve_api_key;
use crate::traits::Model;
use crate::types::provider_options::ProviderOptions;

/// When [`build_model`] is allowed to fall back from a remote provider to a
/// locally-servable model.
///
/// A present API key always builds the remote provider, regardless of this
/// policy — the policy only governs what happens on an [`Auth`](BlazenError::Auth)
/// miss. The default is [`FallbackPolicy::Never`], which keeps standalone
/// behaviour byte-identical to a plain remote build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[serde(rename_all = "snake_case")]
pub enum FallbackPolicy {
    /// Never fall back. A missing key is a hard [`Auth`](BlazenError::Auth)
    /// error. This is the standalone default.
    #[default]
    Never,
    /// Fall back to a locally-servable model only when no API key resolves.
    /// A present key still builds the remote provider.
    WhenNoKey,
    /// Same selection logic as [`WhenNoKey`](FallbackPolicy::WhenNoKey) at this
    /// layer: a present key builds remote, a missing key routes local if the
    /// model is servable. The "prefer local even when a key exists" intent is
    /// honoured by callers that consult the policy *before* calling
    /// [`build_model`] (e.g. a router that skips key resolution entirely); the
    /// factory itself never discards a usable remote key.
    PreferLocal,
}

impl FallbackPolicy {
    /// Whether this policy permits any local fallback at all.
    #[must_use]
    pub fn allows_local(self) -> bool {
        !matches!(self, FallbackPolicy::Never)
    }
}

/// Reports whether a given `(provider, model)` pair can be served from a model
/// running locally in-process (mistral.rs, llama.cpp, candle, …).
///
/// Implemented on the controlplane side over its model manager. Standalone
/// `blazen-llm` uses [`NoLocalModels`], which always returns `false`.
#[async_trait]
pub trait LocalModelProbe: Send + Sync {
    /// Return `true` when a local model can serve `provider` (optionally a
    /// specific `model`). `model = None` asks "can this provider be served
    /// locally at all?".
    async fn is_locally_servable(&self, provider: &str, model: Option<&str>) -> bool;
}

/// Builds a [`Model`] backed by a locally-served model.
///
/// Separated from [`LocalModelProbe`] so a caller can cheaply probe
/// servability (hot path) without paying construction cost until the factory
/// has actually decided to route local.
///
/// The request shape is the strongly-typed
/// [`blazen_local_llm::LocalModelRequest`] — it carries the provider name,
/// model identifier, and a full [`blazen_local_llm::LocalLlmOptions`] payload
/// plus per-backend extras (encoded as `serde_json::Value`) so downstream
/// implementations route to candle / mistralrs / llama.cpp without losing
/// any of the caller's configuration.
#[async_trait]
pub trait LocalModelFactory: Send + Sync {
    /// Construct a local [`Model`] for the given [`LocalModelRequest`].
    ///
    /// # Errors
    ///
    /// Returns a [`BlazenError`] if the local model cannot be constructed
    /// (e.g. weights unavailable, device out of memory).
    async fn build_local(
        &self,
        request: blazen_local_llm::LocalModelRequest,
    ) -> Result<Box<dyn Model>, BlazenError>;
}

/// The standalone default probe: no model is ever locally servable.
///
/// With this probe installed, [`build_model`] never routes local and the
/// [`LocalModelFactory`] is never consulted, so behaviour is identical to a
/// plain remote build.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoLocalModels;

#[async_trait]
impl LocalModelProbe for NoLocalModels {
    async fn is_locally_servable(&self, _provider: &str, _model: Option<&str>) -> bool {
        false
    }
}

/// Default model identifier handed to [`LocalModelFactory::build_local`] when
/// the caller did not pin one in [`ProviderOptions::model`]. The local factory
/// is free to interpret this however it likes (e.g. "the provider's default
/// local checkpoint").
const DEFAULT_LOCAL_MODEL: &str = "default";

/// Build a [`Model`] for `provider`, selecting remote-vs-local per `policy`.
///
/// `build_remote` is the caller-supplied constructor for the concrete remote
/// provider (typically `SomeProvider::from_options(...).map(|p| Box::new(p) as
/// _)`). Keeping it a closure means the factory does not need a giant
/// provider-name `match` — `blazen-llm` has no centralized name→provider
/// dispatch, so the concrete constructor stays the caller's choice.
///
/// ## Logic
///
/// - Key resolves → `build_remote(key, opts)` (remote wins regardless of
///   `policy`).
/// - Key misses with [`Auth`](BlazenError::Auth) AND `policy` permits local AND
///   the probe reports the model servable → `local_factory.build_local(...)`,
///   using `opts.model` or [`DEFAULT_LOCAL_MODEL`].
/// - Otherwise the key-resolution error is propagated.
///
/// # Errors
///
/// Returns the key-resolution [`Auth`](BlazenError::Auth) error when no key
/// resolves and no local fallback applies, any non-`Auth` key error unchanged,
/// or whatever error `build_remote` / `build_local` produces.
pub async fn build_model<F>(
    provider: &str,
    opts: ProviderOptions,
    policy: FallbackPolicy,
    local_probe: &dyn LocalModelProbe,
    local_factory: &dyn LocalModelFactory,
    build_remote: F,
) -> Result<Box<dyn Model>, BlazenError>
where
    F: FnOnce(String, ProviderOptions) -> Result<Box<dyn Model>, BlazenError>,
{
    match resolve_api_key(provider, opts.api_key.clone()) {
        // A key resolved: a remote provider always wins.
        Ok(key) => build_remote(key, opts),

        // No key. Consider the local seam only on an Auth miss with a
        // permitting policy; propagate everything else unchanged.
        Err(err @ BlazenError::Auth { .. }) => {
            if policy.allows_local()
                && local_probe
                    .is_locally_servable(provider, opts.model.as_deref())
                    .await
            {
                let model = opts.model.as_deref().unwrap_or(DEFAULT_LOCAL_MODEL);
                // Build a typed LocalModelRequest from ProviderOptions: model
                // gets pre-stamped onto the LocalLlmOptions.model_id field so
                // the request is self-consistent at the seam.
                let local_opts = blazen_local_llm::LocalLlmOptions {
                    model_id: Some(model.to_string()),
                    ..blazen_local_llm::LocalLlmOptions::default()
                };
                let request = blazen_local_llm::LocalModelRequest::new(provider, model, local_opts);
                local_factory.build_local(request).await
            } else {
                Err(err)
            }
        }

        // Non-Auth key-resolution error: never consult the local seam.
        Err(other) => Err(other),
    }
}

#[cfg(test)]
// SERIAL is a std `Mutex<()>` held across `.await` to serialize tests that
// touch process-global resolver state. Switching to a tokio Mutex would
// require every test fn to also be `async fn` for `.lock().await` and adds
// no real safety here (the guard carries no data and the lock is only ever
// used for cross-test exclusion, not for protecting an inner value across
// the await).
#[allow(clippy::await_holding_lock)]
mod tests {
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};

    use futures_util::Stream;

    use super::{FallbackPolicy, LocalModelFactory, LocalModelProbe, NoLocalModels, build_model};
    use crate::error::BlazenError;
    use crate::keys::{clear_key_resolvers, set_current_place};
    use crate::traits::Model;
    use crate::types::provider_options::ProviderOptions;
    use crate::types::{ModelRequest, ModelResponse, StreamChunk};

    /// Tests touch the process-global resolver chain / current place, so they
    /// must serialize and tear that state down.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn reset() {
        clear_key_resolvers();
        set_current_place(None);
    }

    /// A trivial [`Model`] tagged with an id so tests can assert which builder
    /// produced it.
    struct MockModel {
        id: String,
    }

    #[async_trait::async_trait]
    impl Model for MockModel {
        fn model_id(&self) -> &str {
            &self.id
        }

        async fn complete(&self, _request: ModelRequest) -> Result<ModelResponse, BlazenError> {
            Err(BlazenError::unsupported("mock"))
        }

        async fn stream(
            &self,
            _request: ModelRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
        {
            Err(BlazenError::unsupported("mock"))
        }
    }

    /// Probe that reports servability for a fixed provider only.
    struct SelectiveProbe {
        servable_provider: &'static str,
    }

    #[async_trait::async_trait]
    impl LocalModelProbe for SelectiveProbe {
        async fn is_locally_servable(&self, provider: &str, _model: Option<&str>) -> bool {
            provider == self.servable_provider
        }
    }

    /// Factory that records what model id it was asked for and returns a
    /// `MockModel` tagged `"local:<provider>:<model>"`.
    struct RecordingLocalFactory {
        called: AtomicBool,
    }

    #[async_trait::async_trait]
    impl LocalModelFactory for RecordingLocalFactory {
        async fn build_local(
            &self,
            request: blazen_local_llm::LocalModelRequest,
        ) -> Result<Box<dyn Model>, BlazenError> {
            self.called.store(true, Ordering::SeqCst);
            Ok(Box::new(MockModel {
                id: format!("local:{}:{}", request.provider, request.model),
            }))
        }
    }

    /// A local factory that must never be called; panics if it is.
    struct NeverLocalFactory;

    #[async_trait::async_trait]
    impl LocalModelFactory for NeverLocalFactory {
        async fn build_local(
            &self,
            _request: blazen_local_llm::LocalModelRequest,
        ) -> Result<Box<dyn Model>, BlazenError> {
            panic!("local factory must not be called");
        }
    }

    #[tokio::test]
    async fn never_policy_no_key_errors_without_calling_remote() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();

        let remote_called = AtomicBool::new(false);
        // `nonexistent` has no env var, so the key never resolves.
        let result = build_model(
            "nonexistent",
            ProviderOptions::default(),
            FallbackPolicy::Never,
            &NoLocalModels,
            &NeverLocalFactory,
            |_key, _opts| {
                remote_called.store(true, Ordering::SeqCst);
                Ok(Box::new(MockModel {
                    id: "remote".into(),
                }) as Box<dyn Model>)
            },
        )
        .await;

        assert!(matches!(result, Err(BlazenError::Auth { .. })));
        assert!(
            !remote_called.load(Ordering::SeqCst),
            "remote must not be built"
        );
        reset();
    }

    #[tokio::test]
    async fn when_no_key_servable_builds_local() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();

        let factory = RecordingLocalFactory {
            called: AtomicBool::new(false),
        };
        let result = build_model(
            "nonexistent",
            ProviderOptions::default(),
            FallbackPolicy::WhenNoKey,
            &SelectiveProbe {
                servable_provider: "nonexistent",
            },
            &factory,
            |_key, _opts| {
                panic!("remote builder must not be called when no key resolves");
            },
        )
        .await;

        let model = result.expect("local model should be built");
        assert_eq!(model.model_id(), "local:nonexistent:default");
        assert!(factory.called.load(Ordering::SeqCst));
        reset();
    }

    #[tokio::test]
    async fn when_no_key_servable_uses_pinned_model() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();

        let opts = ProviderOptions {
            model: Some("llama-3-8b".into()),
            ..Default::default()
        };
        let result = build_model(
            "nonexistent",
            opts,
            FallbackPolicy::WhenNoKey,
            &SelectiveProbe {
                servable_provider: "nonexistent",
            },
            &RecordingLocalFactory {
                called: AtomicBool::new(false),
            },
            |_key, _opts| panic!("remote builder must not be called"),
        )
        .await;

        let model = result.expect("local model should be built");
        assert_eq!(model.model_id(), "local:nonexistent:llama-3-8b");
        reset();
    }

    #[tokio::test]
    async fn when_no_key_not_servable_errors() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();

        let result = build_model(
            "nonexistent",
            ProviderOptions::default(),
            FallbackPolicy::WhenNoKey,
            // Servable for a DIFFERENT provider → this one is not servable.
            &SelectiveProbe {
                servable_provider: "something-else",
            },
            &NeverLocalFactory,
            |_key, _opts| panic!("remote builder must not be called"),
        )
        .await;

        assert!(matches!(result, Err(BlazenError::Auth { .. })));
        reset();
    }

    #[tokio::test]
    async fn key_present_builds_remote_regardless_of_policy() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();

        for policy in [
            FallbackPolicy::Never,
            FallbackPolicy::WhenNoKey,
            FallbackPolicy::PreferLocal,
        ] {
            let opts = ProviderOptions {
                api_key: Some("sk-explicit".into()),
                ..Default::default()
            };
            let result = build_model(
                "nonexistent",
                opts,
                policy,
                // Even though the model is servable, a present key wins.
                &SelectiveProbe {
                    servable_provider: "nonexistent",
                },
                &NeverLocalFactory,
                |key, _opts| {
                    assert_eq!(key, "sk-explicit");
                    Ok(Box::new(MockModel {
                        id: "remote".into(),
                    }) as Box<dyn Model>)
                },
            )
            .await;

            let model = result.expect("remote model should be built");
            assert_eq!(model.model_id(), "remote", "policy {policy:?}");
        }
        reset();
    }
}
