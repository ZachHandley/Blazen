//! Provider implementations for the Node.js bindings.

pub mod capability_providers;
pub mod completion_model;
pub mod custom;
pub mod fal;
pub mod openai;
pub mod transcription;

// Re-export the main types.
pub use capability_providers::{
    CapabilityProviderConfig, JsBackgroundRemovalProvider, JsImageProvider, JsMusicProvider,
    JsTTSProvider, JsThreeDProvider, JsVideoProvider, JsVoiceProvider,
};
pub use completion_model::JsCompletionModel;
pub use custom::{CustomProviderOptions, JsCustomProvider};
pub use fal::{JsFalEmbeddingModel, JsFalProvider};
pub use openai::JsOpenAiProvider;
pub use transcription::JsTranscription;
