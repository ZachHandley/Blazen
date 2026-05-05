//! JS bindings for the tool-input schema builders in
//! [`blazen_llm::content::tool_input`].
//!
//! Each function returns a JSON Schema fragment as a plain `serde_json::Value`
//! that napi serializes into a JS object. Pass the result directly to
//! `ToolDefinition.parameters` (or any place a JSON Schema is expected) to
//! declare a tool input that takes a content handle id.

use blazen_llm::content::tool_input;
use napi_derive::napi;

/// Build a JSON Schema declaring a single required image-handle input.
///
/// The model fills the property with a content-handle id string; Blazen
/// resolves it against the active [`super::store::JsContentStore`] before
/// the tool's handler runs.
#[napi(js_name = "imageInput")]
#[must_use]
pub fn image_input(name: String, description: String) -> serde_json::Value {
    tool_input::image_input(name, description)
}

/// Build a JSON Schema declaring a single required audio-handle input.
#[napi(js_name = "audioInput")]
#[must_use]
pub fn audio_input(name: String, description: String) -> serde_json::Value {
    tool_input::audio_input(name, description)
}

/// Build a JSON Schema declaring a single required video-handle input.
#[napi(js_name = "videoInput")]
#[must_use]
pub fn video_input(name: String, description: String) -> serde_json::Value {
    tool_input::video_input(name, description)
}

/// Build a JSON Schema declaring a single required document/file-handle input.
#[napi(js_name = "fileInput")]
#[must_use]
pub fn file_input(name: String, description: String) -> serde_json::Value {
    tool_input::file_input(name, description)
}

/// Build a JSON Schema declaring a single required 3D-model-handle input.
#[napi(js_name = "threeDInput")]
#[must_use]
pub fn three_d_input(name: String, description: String) -> serde_json::Value {
    tool_input::three_d_input(name, description)
}

/// Build a JSON Schema declaring a single required CAD-file-handle input.
#[napi(js_name = "cadInput")]
#[must_use]
pub fn cad_input(name: String, description: String) -> serde_json::Value {
    tool_input::cad_input(name, description)
}
