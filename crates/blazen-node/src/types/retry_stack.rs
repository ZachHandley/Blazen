//! Plain-object mirror of [`blazen_llm::retry::RetryStack`].
//!
//! [`JsRetryStack`] surfaces the four scope tiers (provider / pipeline /
//! workflow / step) as optional [`JsRetryConfig`](crate::generated::JsRetryConfig)
//! fields. A [`JsRetryStack::resolve`] free function mirrors
//! [`RetryStack::resolve`] for callers that want to compute the effective
//! retry policy at the JS boundary.

use napi_derive::napi;

use blazen_llm::retry::RetryStack;

use crate::generated::JsRetryConfig;

/// Snapshot of every scope's retry configuration. Mirrors
/// [`RetryStack`].
///
/// All fields are optional; any combination of `null` / `undefined`
/// scopes is valid and falls through to the next-outer non-`None` scope
/// when [`resolveRetryStack`] is called.
#[napi(object, js_name = "RetryStack")]
#[derive(Default)]
pub struct JsRetryStack {
    /// Provider-level default (lowest priority).
    pub provider: Option<JsRetryConfig>,
    /// Pipeline-level default.
    pub pipeline: Option<JsRetryConfig>,
    /// Workflow-level override.
    pub workflow: Option<JsRetryConfig>,
    /// Step-level override (highest priority before the per-call override).
    pub step: Option<JsRetryConfig>,
}

impl From<&RetryStack> for JsRetryStack {
    fn from(s: &RetryStack) -> Self {
        Self {
            provider: s.provider.as_ref().map(|c| c.as_ref().clone().into()),
            pipeline: s.pipeline.as_ref().map(|c| c.as_ref().clone().into()),
            workflow: s.workflow.as_ref().map(|c| c.as_ref().clone().into()),
            step: s.step.as_ref().map(|c| c.as_ref().clone().into()),
        }
    }
}

impl From<JsRetryStack> for RetryStack {
    fn from(s: JsRetryStack) -> Self {
        Self {
            provider: s.provider.map(|c| std::sync::Arc::new(c.into())),
            pipeline: s.pipeline.map(|c| std::sync::Arc::new(c.into())),
            workflow: s.workflow.map(|c| std::sync::Arc::new(c.into())),
            step: s.step.map(|c| std::sync::Arc::new(c.into())),
        }
    }
}

/// Build an empty [`JsRetryStack`] with every scope set to `null`.
#[napi(js_name = "newRetryStack")]
#[must_use]
pub fn new_retry_stack() -> JsRetryStack {
    JsRetryStack::default()
}

/// Resolve the effective [`JsRetryConfig`] for the given stack and an
/// optional per-call override. Mirrors [`RetryStack::resolve`].
///
/// ```typescript
/// const effective = resolveRetryStack(
///   { workflow: { maxRetries: 5 } },
///   { maxRetries: 9 }, // per-call override wins
/// );
/// // effective.maxRetries === 9
/// ```
#[napi(js_name = "resolveRetryStack")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn resolve_retry_stack(
    stack: JsRetryStack,
    call_override: Option<JsRetryConfig>,
) -> JsRetryConfig {
    let stack: RetryStack = stack.into();
    let call_override = call_override.map(|c| std::sync::Arc::new(c.into()));
    let resolved = stack.resolve(call_override.as_ref());
    resolved.as_ref().clone().into()
}
