//! Standalone `wasm-bindgen` wrappers for the [`blazen_llm::middleware`]
//! composable middleware system.
//!
//! Exposes:
//! - [`WasmMiddlewareStack`] (`MiddlewareStack`) -- the builder used to
//!   compose multiple middleware layers around a [`WasmCompletionModel`].
//! - [`WasmRetryMiddleware`] (`RetryMiddleware`) -- a built-in middleware
//!   that wraps a model with [`blazen_llm::retry::RetryCompletionModel`].
//! - [`WasmCacheMiddleware`] (`CacheMiddleware`) -- a built-in middleware
//!   that wraps a model with [`blazen_llm::cache::CachedCompletionModel`].
//! - [`WasmCustomMiddleware`] (`Middleware`) -- a JS-callback-based
//!   middleware that delegates the wrap step to a JavaScript function.
//!
//! # Custom middleware (the ABC pattern)
//!
//! `Middleware` is a JS-callback class: callers supply a function
//! `(inner: CompletionModel) => CompletionModel` that returns a new
//! `CompletionModel` decorating `inner`. The most common way to produce
//! such a model is via `inner.withRetry({ ... })`, `new RetryCompletionModel(...)`,
//! or `CompletionModel.fromJsHandler(...)`.
//!
//! ```js
//! const stack = new MiddlewareStack()
//!   .layer(new RetryMiddleware({ maxRetries: 3 }))
//!   .layer(new Middleware((inner) => inner.withCache({ ttlSeconds: 60 })));
//! const model = stack.apply(CompletionModel.openai());
//! ```

use std::sync::Arc;

use wasm_bindgen::convert::TryFromJsValue;
use wasm_bindgen::prelude::*;

use blazen_llm::cache::{CacheConfig, CachedCompletionModel};
use blazen_llm::middleware::{CacheMiddleware, Middleware, RetryMiddleware};
use blazen_llm::retry::{RetryCompletionModel, RetryConfig};
use blazen_llm::traits::CompletionModel;

use crate::completion_model::WasmCompletionModel;
use crate::decorators::{build_cache_config, build_retry_config};

// ---------------------------------------------------------------------------
// Internal: a Middleware impl that delegates to a JS function.
// ---------------------------------------------------------------------------

/// A [`Middleware`] whose `wrap` step is delegated to a JavaScript function.
///
/// The JS function is called with a `CompletionModel` instance and must
/// return another `CompletionModel` that decorates it.
struct JsMiddleware {
    apply: js_sys::Function,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for JsMiddleware {}
unsafe impl Sync for JsMiddleware {}

impl Middleware for JsMiddleware {
    fn wrap(&self, inner: Arc<dyn CompletionModel>) -> Arc<dyn CompletionModel> {
        // Wrap the inner Arc in a WasmCompletionModel and pass it to the JS
        // function. The JS side returns a WasmCompletionModel (or an object
        // shaped like one) whose inner Arc we extract back.
        let inner_js = WasmCompletionModel::from_arc(Arc::clone(&inner));
        let inner_value: JsValue = inner_js.into();

        match self.apply.call1(&JsValue::NULL, &inner_value) {
            Ok(returned) => {
                // Try to convert the returned value back into a
                // WasmCompletionModel.
                match WasmCompletionModel::try_from_js_value(returned) {
                    Ok(wm) => wm.inner_arc(),
                    Err(_) => {
                        // The JS side did not return a CompletionModel: fall
                        // back to the inner model so the chain still works.
                        inner
                    }
                }
            }
            Err(_) => inner,
        }
    }
}

// ---------------------------------------------------------------------------
// RetryMiddleware
// ---------------------------------------------------------------------------

/// Built-in middleware that wraps a model with retry-with-exponential-backoff.
///
/// ```js
/// const middleware = new RetryMiddleware({ maxRetries: 5, jitter: true });
/// const stack = new MiddlewareStack().layer(middleware);
/// ```
#[wasm_bindgen(js_name = "RetryMiddleware")]
pub struct WasmRetryMiddleware {
    config: RetryConfig,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmRetryMiddleware {}
unsafe impl Sync for WasmRetryMiddleware {}

#[wasm_bindgen(js_class = "RetryMiddleware")]
impl WasmRetryMiddleware {
    /// Create a new retry middleware with the given options.
    ///
    /// `options` accepts the same fields as
    /// [`crate::decorators::WasmRetryCompletionModel::new`].
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue) -> WasmRetryMiddleware {
        Self {
            config: build_retry_config(&options),
        }
    }

    /// Apply this middleware directly to a `CompletionModel`, returning a
    /// new wrapped model. Equivalent to `model.withRetry(...)` or
    /// `new RetryCompletionModel(model, ...)` but lets the same
    /// configuration object be reused across multiple stacks.
    #[wasm_bindgen]
    pub fn wrap(&self, model: &WasmCompletionModel) -> WasmCompletionModel {
        let wrapped: Arc<dyn CompletionModel> = Arc::new(RetryCompletionModel::from_arc(
            model.inner_arc(),
            self.config.clone(),
        ));
        WasmCompletionModel::from_arc(wrapped)
    }
}

impl WasmRetryMiddleware {
    pub(crate) fn into_config(self) -> RetryConfig {
        self.config
    }
}

// ---------------------------------------------------------------------------
// CacheMiddleware
// ---------------------------------------------------------------------------

/// Built-in middleware that wraps a model with content-hash response caching.
///
/// ```js
/// const middleware = new CacheMiddleware({ ttlSeconds: 600 });
/// const stack = new MiddlewareStack().layer(middleware);
/// ```
#[wasm_bindgen(js_name = "CacheMiddleware")]
pub struct WasmCacheMiddleware {
    config: CacheConfig,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmCacheMiddleware {}
unsafe impl Sync for WasmCacheMiddleware {}

#[wasm_bindgen(js_class = "CacheMiddleware")]
impl WasmCacheMiddleware {
    /// Create a new cache middleware with the given options.
    ///
    /// `options` accepts the same fields as
    /// [`crate::decorators::WasmCachedCompletionModel::new`].
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue) -> WasmCacheMiddleware {
        Self {
            config: build_cache_config(&options),
        }
    }

    /// Apply this middleware directly to a `CompletionModel`, returning a
    /// new wrapped model.
    #[wasm_bindgen]
    pub fn wrap(&self, model: &WasmCompletionModel) -> WasmCompletionModel {
        let wrapped: Arc<dyn CompletionModel> = Arc::new(CachedCompletionModel::from_arc(
            model.inner_arc(),
            self.config.clone(),
        ));
        WasmCompletionModel::from_arc(wrapped)
    }
}

impl WasmCacheMiddleware {
    pub(crate) fn into_config(self) -> CacheConfig {
        self.config
    }
}

// ---------------------------------------------------------------------------
// Custom (JS-callback) Middleware
// ---------------------------------------------------------------------------

/// A user-defined middleware whose wrap step is a JavaScript function.
///
/// The supplied function receives the inner `CompletionModel` and must
/// synchronously return a new `CompletionModel` that decorates it (any
/// async work must be performed inside the resulting model's
/// `complete`/`stream` methods, not inside the wrap step itself).
///
/// ```js
/// const logger = new Middleware((inner) => {
///   // For example, wrap with a cache:
///   return inner.withCache({ ttlSeconds: 60 });
/// });
/// const stack = new MiddlewareStack().layer(logger);
/// ```
#[wasm_bindgen(js_name = "Middleware")]
pub struct WasmCustomMiddleware {
    apply: js_sys::Function,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmCustomMiddleware {}
unsafe impl Sync for WasmCustomMiddleware {}

#[wasm_bindgen(js_class = "Middleware")]
impl WasmCustomMiddleware {
    /// Create a new middleware from a JS apply callback.
    ///
    /// The callback signature is `(inner: CompletionModel) => CompletionModel`.
    #[wasm_bindgen(constructor)]
    pub fn new(apply: js_sys::Function) -> WasmCustomMiddleware {
        Self { apply }
    }

    /// Apply this middleware directly to a `CompletionModel`, returning a
    /// new wrapped model. Calls the JS apply callback exactly once.
    #[wasm_bindgen]
    pub fn wrap(&self, model: &WasmCompletionModel) -> Result<WasmCompletionModel, JsValue> {
        let inner_js = WasmCompletionModel::from_arc(model.inner_arc());
        let inner_value: JsValue = inner_js.into();
        let returned = self.apply.call1(&JsValue::NULL, &inner_value)?;
        WasmCompletionModel::try_from_js_value(returned).map_err(|_| {
            JsValue::from_str(
                "Middleware apply callback must return a CompletionModel instance",
            )
        })
    }
}

// ---------------------------------------------------------------------------
// MiddlewareStack
// ---------------------------------------------------------------------------

/// A builder for composing multiple middleware layers around a
/// `CompletionModel`.
///
/// Layers are applied so that the **first layer added becomes the outermost
/// wrapper** -- it executes first on the request path and last on the
/// response path.
///
/// ```js
/// const stack = new MiddlewareStack()
///   .layerRetry({ maxRetries: 3 })
///   .layerCache({ ttlSeconds: 60 });
/// const model = stack.apply(CompletionModel.openai());
/// ```
#[wasm_bindgen(js_name = "MiddlewareStack")]
pub struct WasmMiddlewareStack {
    layers: Vec<Box<dyn Middleware>>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmMiddlewareStack {}
unsafe impl Sync for WasmMiddlewareStack {}

impl Default for WasmMiddlewareStack {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen(js_class = "MiddlewareStack")]
impl WasmMiddlewareStack {
    /// Create an empty middleware stack.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> WasmMiddlewareStack {
        Self { layers: Vec::new() }
    }

    /// Add a [`RetryMiddleware`] layer using the given options.
    ///
    /// Returns the same stack for chaining (the caller's `MiddlewareStack`
    /// reference is consumed and re-borrowed in JS via the return value).
    #[wasm_bindgen(js_name = "layerRetry")]
    pub fn layer_retry(mut self, options: JsValue) -> WasmMiddlewareStack {
        let config = build_retry_config(&options);
        self.layers.push(Box::new(RetryMiddleware { config }));
        self
    }

    /// Add a [`CacheMiddleware`] layer using the given options.
    #[wasm_bindgen(js_name = "layerCache")]
    pub fn layer_cache(mut self, options: JsValue) -> WasmMiddlewareStack {
        let config = build_cache_config(&options);
        self.layers.push(Box::new(CacheMiddleware { config }));
        self
    }

    /// Add a pre-built [`RetryMiddleware`] layer (consumes the value).
    #[wasm_bindgen(js_name = "layerRetryMiddleware")]
    pub fn layer_retry_middleware(
        mut self,
        middleware: WasmRetryMiddleware,
    ) -> WasmMiddlewareStack {
        self.layers.push(Box::new(RetryMiddleware {
            config: middleware.into_config(),
        }));
        self
    }

    /// Add a pre-built [`CacheMiddleware`] layer (consumes the value).
    #[wasm_bindgen(js_name = "layerCacheMiddleware")]
    pub fn layer_cache_middleware(
        mut self,
        middleware: WasmCacheMiddleware,
    ) -> WasmMiddlewareStack {
        self.layers.push(Box::new(CacheMiddleware {
            config: middleware.into_config(),
        }));
        self
    }

    /// Add a custom JS-callback middleware layer (consumes the value).
    ///
    /// The callback is invoked exactly once when [`Self::apply`] runs,
    /// receiving the inner `CompletionModel` and returning the wrapped one.
    #[wasm_bindgen(js_name = "layerCustom")]
    pub fn layer_custom(mut self, middleware: WasmCustomMiddleware) -> WasmMiddlewareStack {
        self.layers.push(Box::new(JsMiddleware {
            apply: middleware.apply,
        }));
        self
    }

    /// Apply all middleware layers to `model`, returning the fully wrapped
    /// model as a generic `CompletionModel`.
    ///
    /// Layers are applied in reverse insertion order so that the first
    /// layer added becomes the outermost wrapper.
    #[wasm_bindgen]
    pub fn apply(self, model: &WasmCompletionModel) -> WasmCompletionModel {
        let mut wrapped = model.inner_arc();
        for layer in self.layers.into_iter().rev() {
            wrapped = layer.wrap(wrapped);
        }
        WasmCompletionModel::from_arc(wrapped)
    }

    /// The number of layers currently in the stack.
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> u32 {
        u32::try_from(self.layers.len()).unwrap_or(u32::MAX)
    }
}
