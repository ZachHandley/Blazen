//! JavaScript bindings for the composable middleware system.
//!
//! Mirrors [`blazen_llm::middleware`]: a [`Middleware`] is a unit that
//! wraps an `Arc<dyn CompletionModel>` with additional behaviour, and a
//! [`MiddlewareStack`] composes several layers in a predictable order.
//!
//! The first layer added to a stack becomes the **outermost** wrapper --
//! it executes first on the request path and last on the response path.
//!
//! ```javascript
//! const stack = new MiddlewareStack();
//! stack.withRetry({ maxRetries: 3 });
//! stack.withCache({ ttlSeconds: 300 });
//!
//! const wrapped = stack.apply(CompletionModel.openai());
//! const response = await wrapped.complete([ChatMessage.user("hi")]);
//!
//! // Or, using the concrete middleware classes:
//! const stack2 = new MiddlewareStack();
//! stack2.layer(new RetryMiddleware({ maxRetries: 3 }).toMiddleware());
//! stack2.layer(new CacheMiddleware({ ttlSeconds: 300 }).toMiddleware());
//! ```
//!
//! Subclassing [`Middleware`] from JavaScript is supported but currently
//! a no-op at the wrap stage: only built-in middleware variants
//! (`RetryMiddleware`, `CacheMiddleware`) carry an active Rust
//! implementation. Subclassing is provided so that user code can keep a
//! uniform shape for organisational layers (logging, tracing, metrics)
//! that only run on the JavaScript side.

use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::CompletionModel;
use blazen_llm::cache::CacheConfig;
use blazen_llm::middleware::{
    CacheMiddleware as CoreCacheMiddleware, Middleware as CoreMiddleware, MiddlewareStack,
    RetryMiddleware as CoreRetryMiddleware,
};
use blazen_llm::retry::RetryConfig;

use crate::generated::{JsCacheConfig, JsRetryConfig};
use crate::providers::JsCompletionModel;

// ---------------------------------------------------------------------------
// Layer enum (Rust-internal)
// ---------------------------------------------------------------------------

/// Internal representation of a middleware layer.
///
/// Each variant carries the configuration needed to construct the
/// underlying `blazen_llm::middleware` middleware on demand. Storing the
/// config (rather than a `Box<dyn Middleware>`) keeps the layer `Clone`,
/// which is required because [`MiddlewareStack::apply`] consumes its
/// layers but our JavaScript-facing stack is a value type that may be
/// applied multiple times.
#[derive(Clone)]
enum Layer {
    /// Built-in retry layer.
    Retry(RetryConfig),
    /// Built-in cache layer.
    Cache(CacheConfig),
    /// User-defined layer with no Rust-side effect.
    ///
    /// JavaScript subclasses of [`JsMiddleware`] hit this branch -- the
    /// stack records the layer for ordering purposes but does not modify
    /// the wrapped model. The label is preserved for diagnostics.
    Custom { label: String },
}

// ---------------------------------------------------------------------------
// JsMiddleware (subclassable base + tag)
// ---------------------------------------------------------------------------

/// Base class for middleware layers.
///
/// The built-in [`JsRetryMiddleware`] and [`JsCacheMiddleware`] subclasses
/// carry a Rust-side wrap implementation. Direct instances of this class
/// (or unrecognised subclasses) act as pass-through layers from the
/// Rust side -- useful as no-op organisational anchors for JavaScript
/// logging or metrics.
///
/// ```javascript
/// class LoggingMiddleware extends Middleware {
///     constructor() { super({ label: "logging" }); }
/// }
/// ```
#[napi(js_name = "Middleware")]
pub struct JsMiddleware {
    /// The layer this instance contributes when added to a stack.
    /// Concrete subclasses set this in their constructor; the bare
    /// `Middleware` base defaults to [`Layer::Custom`].
    layer: Mutex<Layer>,
}

/// Optional configuration for the [`JsMiddleware`] base constructor.
#[napi(object)]
pub struct JsMiddlewareConfig {
    /// Diagnostic label for this layer. Defaults to `"middleware"` when
    /// omitted.
    pub label: Option<String>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]
impl JsMiddleware {
    /// Construct a new base middleware. Used as a `super(config)` target
    /// by JavaScript subclasses or directly when a no-op layer is
    /// desired.
    #[napi(constructor)]
    pub fn new(config: Option<JsMiddlewareConfig>) -> Self {
        let label = config
            .and_then(|c| c.label)
            .unwrap_or_else(|| "middleware".to_owned());
        Self {
            layer: Mutex::new(Layer::Custom { label }),
        }
    }

    /// The diagnostic label for this layer.
    #[napi(getter)]
    pub fn label(&self) -> String {
        match &*self.layer.lock().expect("middleware layer mutex poisoned") {
            Layer::Retry(_) => "retry".to_owned(),
            Layer::Cache(_) => "cache".to_owned(),
            Layer::Custom { label } => label.clone(),
        }
    }
}

impl JsMiddleware {
    /// Snapshot the current layer configuration. Used by
    /// [`JsMiddlewareStack::layer`] when copying middleware into the
    /// stack.
    fn snapshot(&self) -> Layer {
        self.layer
            .lock()
            .expect("middleware layer mutex poisoned")
            .clone()
    }

    /// Construct a [`JsMiddleware`] base instance pre-configured with a
    /// built-in [`Layer`] variant. Used by
    /// [`JsRetryMiddleware::to_middleware`] and
    /// [`JsCacheMiddleware::to_middleware`] to surface their layer as
    /// the canonical type accepted by [`JsMiddlewareStack::layer`].
    fn from_layer(layer: Layer) -> Self {
        Self {
            layer: Mutex::new(layer),
        }
    }
}

// ---------------------------------------------------------------------------
// JsRetryMiddleware
// ---------------------------------------------------------------------------

/// Built-in middleware that wraps the inner model with retry-on-transient-
/// error behaviour. Equivalent to constructing a
/// [`super::wrappers::JsRetryCompletionModel`] but composable inside a
/// [`JsMiddlewareStack`].
///
/// ```javascript
/// const layer = new RetryMiddleware({ maxRetries: 5, initialDelayMs: 500 });
/// ```
#[napi(js_name = "RetryMiddleware")]
pub struct JsRetryMiddleware {
    /// The retry configuration captured at construction.
    config: RetryConfig,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsRetryMiddleware {
    /// Build a retry middleware layer with the given configuration.
    /// Defaults to [`RetryConfig::default()`] when omitted.
    #[napi(constructor)]
    pub fn new(config: Option<JsRetryConfig>) -> Self {
        let config: RetryConfig = config.map(Into::into).unwrap_or_default();
        Self { config }
    }

    /// The maximum number of retry attempts captured by this layer.
    #[napi(js_name = "maxRetries", getter)]
    pub fn max_retries(&self) -> u32 {
        self.config.max_retries
    }

    /// Convert this concrete middleware into a [`JsMiddleware`] base
    /// instance so it can be passed to [`JsMiddlewareStack::layer`].
    #[napi(js_name = "toMiddleware")]
    pub fn to_middleware(&self) -> JsMiddleware {
        JsMiddleware::from_layer(Layer::Retry(self.config.clone()))
    }
}

// ---------------------------------------------------------------------------
// JsCacheMiddleware
// ---------------------------------------------------------------------------

/// Built-in middleware that wraps the inner model with an in-memory
/// response cache. Equivalent to constructing a
/// [`super::wrappers::JsCachedCompletionModel`] but composable inside a
/// [`JsMiddlewareStack`].
///
/// ```javascript
/// const layer = new CacheMiddleware({ ttlSeconds: 300, maxEntries: 1000 });
/// ```
#[napi(js_name = "CacheMiddleware")]
pub struct JsCacheMiddleware {
    /// The cache configuration captured at construction.
    config: CacheConfig,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsCacheMiddleware {
    /// Build a cache middleware layer with the given configuration.
    /// Defaults to [`CacheConfig::default()`] when omitted.
    #[napi(constructor)]
    pub fn new(config: Option<JsCacheConfig>) -> Self {
        let config: CacheConfig = config.map(Into::into).unwrap_or_default();
        Self { config }
    }

    /// The TTL captured by this layer, in seconds.
    #[napi(js_name = "ttlSeconds", getter)]
    pub fn ttl_seconds(&self) -> u32 {
        u32::try_from(self.config.ttl_seconds).unwrap_or(u32::MAX)
    }

    /// Convert this concrete middleware into a [`JsMiddleware`] base
    /// instance so it can be passed to [`JsMiddlewareStack::layer`].
    #[napi(js_name = "toMiddleware")]
    pub fn to_middleware(&self) -> JsMiddleware {
        JsMiddleware::from_layer(Layer::Cache(self.config.clone()))
    }
}

// ---------------------------------------------------------------------------
// JsMiddlewareStack
// ---------------------------------------------------------------------------

/// A composable stack of middleware layers.
///
/// Layers are added with [`JsMiddlewareStack::layer`] (or the
/// `withRetry` / `withCache` convenience methods) and applied to a
/// model with [`JsMiddlewareStack::apply`]. The first layer added is
/// the outermost wrapper (executes first on the request path).
///
/// ```javascript
/// const stack = new MiddlewareStack();
/// stack.withRetry({ maxRetries: 3 });
/// stack.withCache({ ttlSeconds: 300 });
/// const wrapped = stack.apply(CompletionModel.openai());
/// ```
#[napi(js_name = "MiddlewareStack")]
pub struct JsMiddlewareStack {
    layers: Mutex<Vec<Layer>>,
}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]
impl JsMiddlewareStack {
    /// Create an empty middleware stack.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            layers: Mutex::new(Vec::new()),
        }
    }

    /// Add a layer to this stack.
    ///
    /// Built-in layers (`RetryMiddleware`, `CacheMiddleware`) contribute
    /// their underlying Rust middleware. Bare [`JsMiddleware`] instances
    /// or unrecognised subclasses are recorded as pass-through layers.
    ///
    /// JavaScript callers wanting fluent chaining can bind the stack
    /// once and invoke methods sequentially:
    ///
    /// ```javascript
    /// const stack = new MiddlewareStack();
    /// stack.layer(new RetryMiddleware());
    /// stack.layer(new CacheMiddleware());
    /// const wrapped = stack.apply(model);
    /// ```
    #[napi]
    pub fn layer(&self, middleware: &JsMiddleware) {
        self.push_layer(middleware.snapshot());
    }

    /// Convenience: append a [`JsRetryMiddleware`] layer.
    #[napi(js_name = "withRetry")]
    pub fn with_retry(&self, config: Option<JsRetryConfig>) {
        let cfg: RetryConfig = config.map(Into::into).unwrap_or_default();
        self.push_layer(Layer::Retry(cfg));
    }

    /// Convenience: append a [`JsCacheMiddleware`] layer.
    #[napi(js_name = "withCache")]
    pub fn with_cache(&self, config: Option<JsCacheConfig>) {
        let cfg: CacheConfig = config.map(Into::into).unwrap_or_default();
        self.push_layer(Layer::Cache(cfg));
    }

    /// Number of layers currently registered in the stack.
    #[napi(getter)]
    pub fn length(&self) -> u32 {
        u32::try_from(
            self.layers
                .lock()
                .expect("middleware stack mutex poisoned")
                .len(),
        )
        .unwrap_or(u32::MAX)
    }

    /// Apply every registered layer to `model` and return the wrapped
    /// model as a fresh [`JsCompletionModel`].
    ///
    /// The stack itself is left intact and can be re-applied to other
    /// models.
    #[napi]
    pub fn apply(&self, model: &JsCompletionModel) -> Result<JsCompletionModel> {
        let inner = model.inner.clone().ok_or_else(|| {
            napi::Error::from_reason(
                "MiddlewareStack.apply() cannot wrap a subclassed CompletionModel that has no concrete provider",
            )
        })?;

        let layers = self
            .layers
            .lock()
            .expect("middleware stack mutex poisoned")
            .clone();

        let mut stack = MiddlewareStack::new();
        for layer in layers {
            match layer {
                Layer::Retry(cfg) => {
                    stack = stack.layer(CoreRetryMiddleware { config: cfg });
                }
                Layer::Cache(cfg) => {
                    stack = stack.layer(CoreCacheMiddleware { config: cfg });
                }
                Layer::Custom { .. } => {
                    stack = stack.layer(NoopMiddleware);
                }
            }
        }
        let wrapped: Arc<dyn CompletionModel> = stack.apply(inner);

        Ok(JsCompletionModel {
            inner: Some(wrapped),
            local_model: None,
            config: None,
        })
    }
}

impl JsMiddlewareStack {
    /// Append a layer to the internal vector.
    fn push_layer(&self, layer: Layer) {
        self.layers
            .lock()
            .expect("middleware stack mutex poisoned")
            .push(layer);
    }
}

// ---------------------------------------------------------------------------
// NoopMiddleware
// ---------------------------------------------------------------------------

/// Pass-through middleware used to back JavaScript-side custom layers
/// that do not (currently) carry a Rust implementation. Returns the
/// inner model unchanged.
struct NoopMiddleware;

impl CoreMiddleware for NoopMiddleware {
    fn wrap(&self, inner: Arc<dyn CompletionModel>) -> Arc<dyn CompletionModel> {
        inner
    }
}
