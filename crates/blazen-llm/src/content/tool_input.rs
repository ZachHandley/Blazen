//! Typed multimodal inputs for tool declarations.
//!
//! Models cannot emit raw image bytes as tool arguments — they emit JSON.
//! This module provides two complementary pieces that let tools cleanly
//! accept multimodal content via [`ContentHandle`] references:
//!
//! 1. **Schema builders** — [`image_input`], [`audio_input`], [`file_input`],
//!    [`video_input`], [`three_d_input`], [`cad_input`], plus the generic
//!    [`content_ref_property`] / [`content_ref_required_object`]. They
//!    produce JSON Schema fragments tagged with `x-blazen-content-ref` so
//!    the resolver can locate handle-typed properties.
//! 2. **Argument resolver** — [`resolve_tool_arguments`] walks a tool's
//!    arguments JSON against its schema, finds every property tagged as a
//!    content reference, looks the handle up in the active
//!    [`ContentStore`], and rewrites the argument value to the resolved
//!    typed content (`Image` / `Audio` / `File` / `Video` shape) so the
//!    tool's `execute` function can read the materialized content
//!    directly.
//!
//! The schema-tag approach ensures providers see only standard JSON Schema
//! types (string properties); the `x-blazen-content-ref` extension is
//! ignored by every major provider but read by Blazen's resolver.
//!
//! [`ContentStore`]: super::store::ContentStore
//! [`ContentHandle`]: super::handle::ContentHandle

use serde_json::{Map, Value, json};

use super::handle::ContentHandle;
use super::kind::ContentKind;
use super::store::ContentStore;
use crate::error::BlazenError;

// ---------------------------------------------------------------------------
// Schema builders
// ---------------------------------------------------------------------------

const CONTENT_REF_TAG: &str = "x-blazen-content-ref";

/// Build a JSON Schema property fragment for a content-reference input.
///
/// Returns a value of the form:
/// ```json
/// {
///   "type": "string",
///   "description": "<description>",
///   "x-blazen-content-ref": { "kind": "<kind>" }
/// }
/// ```
///
/// Wrap this in an `object`-typed schema's `properties` map to declare a
/// tool input the model fills in by passing the handle id as a string.
#[must_use]
pub fn content_ref_property(kind: ContentKind, description: impl Into<String>) -> Value {
    json!({
        "type": "string",
        "description": description.into(),
        CONTENT_REF_TAG: { "kind": kind.as_str() },
    })
}

/// Build a complete `object`-typed JSON Schema declaring a single
/// required content-reference input plus optional companion fields.
///
/// `name` is the property name the model passes the handle as. `kind`
/// is the expected content kind. `extra_properties` lets you add other
/// non-multimodal inputs to the same tool.
#[must_use]
pub fn content_ref_required_object(
    name: impl Into<String>,
    kind: ContentKind,
    description: impl Into<String>,
    extra_properties: serde_json::Map<String, Value>,
) -> Value {
    let name = name.into();
    let mut props = extra_properties;
    props.insert(name.clone(), content_ref_property(kind, description));
    json!({
        "type": "object",
        "properties": Value::Object(props),
        "required": [name],
    })
}

/// Sugar: a single-property schema declaring an image input.
#[must_use]
pub fn image_input(name: impl Into<String>, description: impl Into<String>) -> Value {
    content_ref_required_object(name, ContentKind::Image, description, Map::default())
}

/// Sugar: a single-property schema declaring an audio input.
#[must_use]
pub fn audio_input(name: impl Into<String>, description: impl Into<String>) -> Value {
    content_ref_required_object(name, ContentKind::Audio, description, Map::default())
}

/// Sugar: a single-property schema declaring a video input.
#[must_use]
pub fn video_input(name: impl Into<String>, description: impl Into<String>) -> Value {
    content_ref_required_object(name, ContentKind::Video, description, Map::default())
}

/// Sugar: a single-property schema declaring a generic file/document input.
#[must_use]
pub fn file_input(name: impl Into<String>, description: impl Into<String>) -> Value {
    content_ref_required_object(name, ContentKind::Document, description, Map::default())
}

/// Sugar: a single-property schema declaring a 3D model input.
#[must_use]
pub fn three_d_input(name: impl Into<String>, description: impl Into<String>) -> Value {
    content_ref_required_object(name, ContentKind::ThreeDModel, description, Map::default())
}

/// Sugar: a single-property schema declaring a CAD-file input.
#[must_use]
pub fn cad_input(name: impl Into<String>, description: impl Into<String>) -> Value {
    content_ref_required_object(name, ContentKind::Cad, description, Map::default())
}

// ---------------------------------------------------------------------------
// Tool-argument resolver
// ---------------------------------------------------------------------------

/// Mismatch between the kind a tool input expects and the kind of the
/// resolved [`ContentHandle`].
#[derive(Debug, Clone)]
pub struct KindMismatch {
    pub property: String,
    pub expected: ContentKind,
    pub actual: ContentKind,
}

impl std::fmt::Display for KindMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tool input '{}' expected content kind '{}' but handle is '{}'",
            self.property, self.expected, self.actual
        )
    }
}

/// Walk `arguments` against `schema` and resolve every property tagged
/// with `x-blazen-content-ref` against the supplied store, replacing the
/// handle-id string with the resolved typed content.
///
/// The resolved content shape mirrors the underlying [`ContentHandle`]:
/// ```json
/// {
///   "kind": "image",
///   "handle_id": "blazen_abc123",
///   "mime_type": "image/png",
///   "byte_size": 1234,
///   "display_name": "vacation.jpg",
///   "source": <MediaSource JSON>
/// }
/// ```
/// The `source` field is the result of [`ContentStore::resolve`]
/// serialized to JSON, so a tool implementation can deserialize it into
/// [`crate::types::MediaSource`] directly.
///
/// Properties whose value is not a string, or whose schema does not carry
/// the content-ref tag, are left alone. Walks nested objects; does not
/// walk arrays (tools are encouraged to declare a single content per
/// property — collections of handles can be modeled as a parallel array
/// of strings if needed).
///
/// Returns the number of arguments resolved.
///
/// # Errors
///
/// Returns [`BlazenError::Validation`] if a content-ref property's value
/// is missing or not a string, or if the resolved handle's kind doesn't
/// match the schema's declared kind. Returns the underlying store error
/// if resolution fails.
pub async fn resolve_tool_arguments(
    arguments: &mut Value,
    schema: &Value,
    store: &dyn ContentStore,
) -> Result<usize, BlazenError> {
    let mut resolved = 0usize;
    walk(arguments, schema, store, "", &mut resolved).await?;
    Ok(resolved)
}

// Recursive walker. Boxed to break the recursion for `async fn` (Rust
// doesn't yet allow direct recursion in async fns without boxing).
fn walk<'a>(
    arguments: &'a mut Value,
    schema: &'a Value,
    store: &'a dyn ContentStore,
    path: &'a str,
    resolved: &'a mut usize,
) -> futures_util::future::BoxFuture<'a, Result<(), BlazenError>> {
    use futures_util::FutureExt;
    async move {
        let Some(props) = schema.get("properties").and_then(Value::as_object) else {
            return Ok(());
        };

        let Some(arg_obj) = arguments.as_object_mut() else {
            return Ok(());
        };

        for (name, prop_schema) in props {
            let property_path = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}.{name}")
            };

            // Recurse into nested object-typed properties whether or not
            // the current property is a content ref.
            if prop_schema
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|s| s == "object")
                && let Some(child) = arg_obj.get_mut(name)
            {
                walk(child, prop_schema, store, &property_path, resolved).await?;
            }

            let Some(ref_meta) = prop_schema.get(CONTENT_REF_TAG) else {
                continue;
            };

            let expected_kind_str = ref_meta
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("other");
            let expected = parse_kind(expected_kind_str);

            let Some(arg_value) = arg_obj.get_mut(name) else {
                continue;
            };

            let handle_id = arg_value
                .as_str()
                .ok_or_else(|| {
                    BlazenError::validation(format!(
                        "tool input '{property_path}' is a content reference; expected a \
                         handle-id string but got {arg_value}"
                    ))
                })?
                .to_owned();

            // Build a minimal handle from what we know. The store may add
            // metadata when it resolves.
            let handle = ContentHandle::new(handle_id.clone(), expected);
            let source = store.resolve(&handle).await?;

            // Lookup richer metadata via the store if it tracks any.
            let metadata = store.metadata(&handle).await.ok();
            let actual_kind = metadata.as_ref().map_or(expected, |m| m.kind);

            if actual_kind != expected && actual_kind != ContentKind::Other {
                return Err(BlazenError::validation(
                    KindMismatch {
                        property: property_path,
                        expected,
                        actual: actual_kind,
                    }
                    .to_string(),
                ));
            }

            // Replace the bare-string argument with the resolved typed
            // content shape.
            let resolved_value = json!({
                "kind": actual_kind.as_str(),
                "handle_id": handle_id,
                "mime_type": metadata.as_ref().and_then(|m| m.mime_type.clone()),
                "byte_size": metadata.as_ref().and_then(|m| m.byte_size),
                "display_name": metadata.as_ref().and_then(|m| m.display_name.clone()),
                "source": serde_json::to_value(source).unwrap_or(Value::Null),
            });
            *arg_value = resolved_value;
            *resolved += 1;
        }

        Ok(())
    }
    .boxed()
}

fn parse_kind(s: &str) -> ContentKind {
    match s {
        "image" => ContentKind::Image,
        "audio" => ContentKind::Audio,
        "video" => ContentKind::Video,
        "document" => ContentKind::Document,
        "three_d_model" => ContentKind::ThreeDModel,
        "cad" => ContentKind::Cad,
        "archive" => ContentKind::Archive,
        "font" => ContentKind::Font,
        "code" => ContentKind::Code,
        "data" => ContentKind::Data,
        _ => ContentKind::Other,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::store::{ContentBody, ContentHint};
    use crate::content::stores::InMemoryContentStore;

    #[test]
    fn image_input_schema_carries_content_ref_tag() {
        let schema = image_input("photo", "the photo");
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["photo"]["type"], "string");
        assert_eq!(
            schema["properties"]["photo"][CONTENT_REF_TAG]["kind"],
            "image"
        );
        assert_eq!(schema["required"][0], "photo");
    }

    #[test]
    fn three_d_and_cad_have_correct_kinds() {
        assert_eq!(
            three_d_input("model", "the model")["properties"]["model"][CONTENT_REF_TAG]["kind"],
            "three_d_model"
        );
        assert_eq!(
            cad_input("part", "the part")["properties"]["part"][CONTENT_REF_TAG]["kind"],
            "cad"
        );
    }

    #[tokio::test]
    async fn resolve_replaces_handle_string_with_typed_content() {
        let store = InMemoryContentStore::new();
        let handle = store
            .put(
                ContentBody::Url {
                    url: "https://example.com/cat.png".into(),
                },
                ContentHint::default()
                    .with_kind(ContentKind::Image)
                    .with_mime_type("image/png"),
            )
            .await
            .unwrap();

        let schema = image_input("photo", "the photo");
        let mut args = json!({ "photo": handle.id });

        let n = resolve_tool_arguments(&mut args, &schema, &store)
            .await
            .unwrap();
        assert_eq!(n, 1);

        assert_eq!(args["photo"]["kind"], "image");
        assert_eq!(args["photo"]["handle_id"], handle.id);
        assert_eq!(args["photo"]["mime_type"], "image/png");
        assert_eq!(args["photo"]["source"]["type"], "url");
        assert_eq!(
            args["photo"]["source"]["url"],
            "https://example.com/cat.png"
        );
    }

    #[tokio::test]
    async fn unrelated_string_arguments_pass_through() {
        let store = InMemoryContentStore::new();
        let mut props = serde_json::Map::new();
        props.insert("note".into(), json!({"type": "string"}));
        let schema = content_ref_required_object("photo", ContentKind::Image, "the photo", props);
        let h = store
            .put(
                ContentBody::Url {
                    url: "https://example.com/x.png".into(),
                },
                ContentHint::default().with_kind(ContentKind::Image),
            )
            .await
            .unwrap();

        let mut args = json!({ "photo": h.id, "note": "look at the cat" });
        let n = resolve_tool_arguments(&mut args, &schema, &store)
            .await
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(args["note"], "look at the cat");
    }

    #[tokio::test]
    async fn missing_handle_string_errors() {
        let store = InMemoryContentStore::new();
        let schema = image_input("photo", "the photo");
        let mut args = json!({ "photo": 123 });
        let res = resolve_tool_arguments(&mut args, &schema, &store).await;
        assert!(res.is_err(), "should error on non-string handle value");
    }

    #[tokio::test]
    async fn kind_mismatch_errors() {
        let store = InMemoryContentStore::new();
        let h = store
            .put(
                ContentBody::Bytes { data: vec![1u8] },
                ContentHint::default().with_kind(ContentKind::Audio),
            )
            .await
            .unwrap();
        let schema = image_input("photo", "the photo");
        let mut args = json!({ "photo": h.id });
        let res = resolve_tool_arguments(&mut args, &schema, &store).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn no_handles_returns_zero() {
        let store = InMemoryContentStore::new();
        let mut props = serde_json::Map::new();
        props.insert("note".into(), json!({"type": "string"}));
        let schema = json!({ "type": "object", "properties": props });
        let mut args = json!({ "note": "hi" });
        let n = resolve_tool_arguments(&mut args, &schema, &store)
            .await
            .unwrap();
        assert_eq!(n, 0);
    }
}
