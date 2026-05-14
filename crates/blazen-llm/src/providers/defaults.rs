//! Provider-defaults hierarchy. A `BaseProviderDefaults` carries universal
//! fields applicable to ANY provider role (completion, embedding, compute,
//! media). Role-specific subtypes (`CompletionProviderDefaults`, etc.)
//! compose `BaseProviderDefaults` and add role-specific fields.
//!
//! These are applied by `BaseProvider` (for completion) and by the trait
//! impls on `CustomProvider` (for compute roles), via each type's `apply()`
//! method which fires the universal `before_request` hook first, then the
//! role-specific hook.

use std::sync::Arc;

use futures_util::future::BoxFuture;

use crate::compute::requests::{
    BackgroundRemovalRequest, ImageRequest, MusicRequest, SpeechRequest, ThreeDRequest,
    TranscriptionRequest, UpscaleRequest, VideoRequest, VoiceCloneRequest,
};
use crate::error::BlazenError;
use crate::types::{CompletionRequest, ToolDefinition};

/// Universal before-request hook. Fires for ANY provider request — receives
/// the Rust method name (e.g. `"complete"`, `"text_to_speech"`) and a
/// mutable JSON view of the request. Runs BEFORE any role-specific hook so
/// downstream typed processing sees the mutated payload.
pub type BeforeRequestHook = Arc<
    dyn Fn(&str, &mut serde_json::Value) -> BoxFuture<'static, Result<(), BlazenError>>
        + Send
        + Sync,
>;

/// Typed completion-side before-request hook. Fires AFTER the universal
/// `before_request` hook on the base defaults, with a typed view of the
/// completion request the caller can mutate.
pub type BeforeCompletionRequestHook = Arc<
    dyn Fn(&mut CompletionRequest) -> BoxFuture<'static, Result<(), BlazenError>> + Send + Sync,
>;

/// Universal provider defaults applicable across every provider role.
///
/// Carries cross-cutting fields like a JSON-level `before_request` hook.
/// Role-specific defaults (`CompletionProviderDefaults` etc.) embed this
/// struct as a `base` field.
#[derive(Default, Clone)]
pub struct BaseProviderDefaults {
    pub before_request: Option<BeforeRequestHook>,
    // Future: default_timeout, default_user_agent, default_metadata, etc.
}

impl std::fmt::Debug for BaseProviderDefaults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseProviderDefaults")
            .field("has_before_request", &self.before_request.is_some())
            .finish_non_exhaustive()
    }
}

impl BaseProviderDefaults {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_before_request(mut self, hook: BeforeRequestHook) -> Self {
        self.before_request = Some(hook);
        self
    }

    /// Internal: run the universal hook against any JSON request body.
    ///
    /// # Errors
    ///
    /// Returns whatever error the user-supplied `before_request` hook
    /// returns; if no hook is configured, returns `Ok(())`.
    pub async fn run_before_request(
        &self,
        method: &str,
        request_json: &mut serde_json::Value,
    ) -> Result<(), BlazenError> {
        if let Some(hook) = &self.before_request {
            hook(method, request_json).await?;
        }
        Ok(())
    }
}

/// Completion-role defaults. System prompt, default tools, default
/// `response_format`, plus a typed before-completion hook.
#[derive(Default, Clone)]
pub struct CompletionProviderDefaults {
    pub base: BaseProviderDefaults,
    pub system_prompt: Option<String>,
    pub tools: Vec<ToolDefinition>,
    pub response_format: Option<serde_json::Value>,
    pub before_completion: Option<BeforeCompletionRequestHook>,
}

impl std::fmt::Debug for CompletionProviderDefaults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionProviderDefaults")
            .field("base", &self.base)
            .field("system_prompt", &self.system_prompt.is_some())
            .field("tools_len", &self.tools.len())
            .field("has_response_format", &self.response_format.is_some())
            .field("has_before_completion", &self.before_completion.is_some())
            .finish()
    }
}

impl CompletionProviderDefaults {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_system_prompt(mut self, s: impl Into<String>) -> Self {
        self.system_prompt = Some(s.into());
        self
    }

    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    #[must_use]
    pub fn with_response_format(mut self, fmt: serde_json::Value) -> Self {
        self.response_format = Some(fmt);
        self
    }

    #[must_use]
    pub fn with_before_completion(mut self, hook: BeforeCompletionRequestHook) -> Self {
        self.before_completion = Some(hook);
        self
    }

    #[must_use]
    pub fn with_base(mut self, base: BaseProviderDefaults) -> Self {
        self.base = base;
        self
    }

    /// Apply defaults to a `CompletionRequest`:
    ///
    /// 1. Run universal `before_request` hook (JSON view).
    /// 2. Prepend a system message if no system message is present and
    ///    `system_prompt` is set.
    /// 3. Append `tools` to the request's tool list (request additions win
    ///    on name collisions).
    /// 4. Set `response_format` if request has none.
    /// 5. Run the typed `before_completion` hook.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Serialization`] if the request cannot be
    /// round-tripped through `serde_json` for the universal hook (this
    /// should never happen for a well-formed `CompletionRequest`), or
    /// whatever error the user-supplied hooks return.
    pub async fn apply(&self, req: &mut CompletionRequest) -> Result<(), BlazenError> {
        // 1. JSON-level universal hook.
        if self.base.before_request.is_some() {
            let mut json = serde_json::to_value(&*req)
                .map_err(|e| BlazenError::Serialization(e.to_string()))?;
            self.base.run_before_request("complete", &mut json).await?;
            *req = serde_json::from_value(json)
                .map_err(|e| BlazenError::Serialization(e.to_string()))?;
        }
        // 2. System prompt — prepend only if no system message exists.
        if let Some(sp) = &self.system_prompt {
            use crate::types::{ChatMessage, Role};
            let has_system = req.messages.iter().any(|m| matches!(m.role, Role::System));
            if !has_system {
                req.messages.insert(0, ChatMessage::system(sp));
            }
        }
        // 3. Append tools. Request tools win on name collision.
        if !self.tools.is_empty() {
            for t in &self.tools {
                if !req.tools.iter().any(|rt| rt.name == t.name) {
                    req.tools.push(t.clone());
                }
            }
        }
        // 4. Response format — only set if request lacks one.
        if let Some(rf) = &self.response_format
            && req.response_format.is_none()
        {
            req.response_format = Some(rf.clone());
        }
        // 5. Typed completion hook.
        if let Some(hook) = &self.before_completion {
            hook(req).await?;
        }
        Ok(())
    }
}

/// Macro to generate a role-defaults struct that wraps `BaseProviderDefaults`
/// plus a typed before-`<role>`-request hook. Each role's request type plugs in.
macro_rules! role_defaults {
    ($name:ident, $request:ty, $hook_alias:ident, $method:expr) => {
        pub type $hook_alias =
            Arc<dyn Fn(&mut $request) -> BoxFuture<'static, Result<(), BlazenError>> + Send + Sync>;

        #[derive(Default, Clone)]
        pub struct $name {
            pub base: BaseProviderDefaults,
            pub before: Option<$hook_alias>,
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("base", &self.base)
                    .field("has_before", &self.before.is_some())
                    .finish()
            }
        }

        impl $name {
            #[must_use]
            pub fn new() -> Self {
                Self::default()
            }

            #[must_use]
            pub fn with_base(mut self, base: BaseProviderDefaults) -> Self {
                self.base = base;
                self
            }

            #[must_use]
            pub fn with_before(mut self, hook: $hook_alias) -> Self {
                self.before = Some(hook);
                self
            }

            /// Apply the universal `before_request` hook (JSON view) followed
            /// by the typed `before` hook to a request of the role's type.
            ///
            /// # Errors
            ///
            /// Returns [`BlazenError::Serialization`] if the request cannot be
            /// round-tripped through `serde_json` for the universal hook, or
            /// whatever error the user-supplied hooks return.
            pub async fn apply(&self, req: &mut $request) -> Result<(), BlazenError> {
                if self.base.before_request.is_some() {
                    let mut json = serde_json::to_value(&*req)
                        .map_err(|e| BlazenError::Serialization(e.to_string()))?;
                    self.base.run_before_request($method, &mut json).await?;
                    *req = serde_json::from_value(json)
                        .map_err(|e| BlazenError::Serialization(e.to_string()))?;
                }
                if let Some(hook) = &self.before {
                    hook(req).await?;
                }
                Ok(())
            }
        }
    };
}

role_defaults!(
    AudioSpeechProviderDefaults,
    SpeechRequest,
    BeforeSpeechRequestHook,
    "text_to_speech"
);
role_defaults!(
    AudioMusicProviderDefaults,
    MusicRequest,
    BeforeMusicRequestHook,
    "generate_music"
);
role_defaults!(
    VoiceCloningProviderDefaults,
    VoiceCloneRequest,
    BeforeVoiceCloneRequestHook,
    "clone_voice"
);
role_defaults!(
    ImageGenerationProviderDefaults,
    ImageRequest,
    BeforeImageRequestHook,
    "generate_image"
);
role_defaults!(
    ImageUpscaleProviderDefaults,
    UpscaleRequest,
    BeforeUpscaleRequestHook,
    "upscale_image"
);
role_defaults!(
    VideoProviderDefaults,
    VideoRequest,
    BeforeVideoRequestHook,
    "text_to_video"
);
role_defaults!(
    TranscriptionProviderDefaults,
    TranscriptionRequest,
    BeforeTranscriptionRequestHook,
    "transcribe"
);
role_defaults!(
    ThreeDProviderDefaults,
    ThreeDRequest,
    BeforeThreeDRequestHook,
    "generate_3d"
);
role_defaults!(
    BackgroundRemovalProviderDefaults,
    BackgroundRemovalRequest,
    BeforeBackgroundRemovalRequestHook,
    "remove_background"
);

/// Embedding-role defaults. V1 carries only `base` plus a typed hook
/// (request type is `EmbeddingRequest` if it exists, otherwise omit the hook).
#[derive(Default, Clone, Debug)]
pub struct EmbeddingProviderDefaults {
    pub base: BaseProviderDefaults,
}

impl EmbeddingProviderDefaults {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_base(mut self, base: BaseProviderDefaults) -> Self {
        self.base = base;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    #[tokio::test]
    async fn completion_defaults_prepend_system_when_absent() {
        let d = CompletionProviderDefaults::new().with_system_prompt("be terse");
        let mut req = CompletionRequest::new(vec![ChatMessage::user("hi")]);
        d.apply(&mut req).await.unwrap();
        assert_eq!(req.messages.len(), 2);
        assert!(matches!(req.messages[0].role, crate::types::Role::System));
    }

    #[tokio::test]
    async fn completion_defaults_skip_system_when_present() {
        let d = CompletionProviderDefaults::new().with_system_prompt("ignored");
        let mut req = CompletionRequest::new(vec![
            ChatMessage::system("existing"),
            ChatMessage::user("hi"),
        ]);
        d.apply(&mut req).await.unwrap();
        assert_eq!(req.messages.len(), 2);
        // Existing system prompt unchanged.
    }

    #[tokio::test]
    async fn completion_defaults_universal_hook_runs() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        let hook: BeforeRequestHook = Arc::new(move |_method, _json| {
            let c = Arc::clone(&counter_clone);
            Box::pin(async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        });
        let d = CompletionProviderDefaults::new()
            .with_base(BaseProviderDefaults::new().with_before_request(hook));
        let mut req = CompletionRequest::new(vec![ChatMessage::user("hi")]);
        d.apply(&mut req).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
