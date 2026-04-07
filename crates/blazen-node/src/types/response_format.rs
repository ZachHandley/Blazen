//! Node wrapper for `ResponseFormat`.

use napi_derive::napi;

use blazen_llm::ResponseFormat;

/// Typed response-format hint for structured output.
///
/// Use one of the variants by setting `kind` to `"text"`, `"json_object"`, or
/// `"json_schema"`. For `"json_schema"`, populate `schemaName`, `schema`, and
/// optionally `strict`.
#[napi(object)]
pub struct JsResponseFormat {
    /// Discriminator. One of `"text"`, `"json_object"`, `"json_schema"`.
    pub kind: String,
    /// Schema name for `json_schema` variants.
    #[napi(js_name = "schemaName")]
    pub schema_name: Option<String>,
    /// Schema body for `json_schema` variants.
    pub schema: Option<serde_json::Value>,
    /// Whether the schema is strict (`json_schema` variants only).
    pub strict: Option<bool>,
}

impl From<&ResponseFormat> for JsResponseFormat {
    fn from(rf: &ResponseFormat) -> Self {
        match rf {
            ResponseFormat::Text => Self {
                kind: "text".to_owned(),
                schema_name: None,
                schema: None,
                strict: None,
            },
            ResponseFormat::JsonObject => Self {
                kind: "json_object".to_owned(),
                schema_name: None,
                schema: None,
                strict: None,
            },
            ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            } => Self {
                kind: "json_schema".to_owned(),
                schema_name: Some(name.clone()),
                schema: Some(schema.clone()),
                strict: Some(*strict),
            },
        }
    }
}

impl From<ResponseFormat> for JsResponseFormat {
    fn from(rf: ResponseFormat) -> Self {
        (&rf).into()
    }
}
