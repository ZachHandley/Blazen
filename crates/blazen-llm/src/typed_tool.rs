//! Ergonomic typed-handler wrapper for tool authors.
//!
//! [`TypedTool`] adapts a Rust async closure with strongly-typed `Args` and
//! `Output` into the dynamic `Tool` trait, doing exactly one
//! `serde_json::from_value` for arguments and one `serde_json::to_value` for
//! the result -- no string round-trip. The handler itself returns
//! `ToolOutput<Output>` so it can carry an optional `llm_override`.
//!
//! For the common case where you don't need an override, use
//! [`typed_tool_simple`] which wraps a `Fn(Args) -> Result<Output, _>`.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::{JsonSchema, schema_for};
use serde::{Serialize, de::DeserializeOwned};

use crate::error::BlazenError;
use crate::traits::Tool;
use crate::types::{ToolDefinition, ToolOutput};

type BoxFut<O> = Pin<Box<dyn Future<Output = Result<ToolOutput<O>, BlazenError>> + Send>>;

/// A `Tool` whose handler accepts a typed `Args` and returns a typed
/// `ToolOutput<Output>`. The trait-object boundary erases `Args`/`Output`
/// to `serde_json::Value` exactly once.
pub struct TypedTool<Args, Output, F>
where
    Args: DeserializeOwned + JsonSchema + Send + 'static,
    Output: Serialize + Send + 'static,
    F: Fn(Args) -> BoxFut<Output> + Send + Sync + 'static,
{
    name: String,
    description: String,
    handler: Arc<F>,
    _phantom: PhantomData<fn(Args) -> Output>,
}

impl<Args, Output, F> TypedTool<Args, Output, F>
where
    Args: DeserializeOwned + JsonSchema + Send + 'static,
    Output: Serialize + Send + 'static,
    F: Fn(Args) -> BoxFut<Output> + Send + Sync + 'static,
{
    pub fn new(name: impl Into<String>, description: impl Into<String>, handler: F) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            handler: Arc::new(handler),
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<Args, Output, F> Tool for TypedTool<Args, Output, F>
where
    Args: DeserializeOwned + JsonSchema + Send + 'static,
    Output: Serialize + Send + 'static,
    F: Fn(Args) -> BoxFut<Output> + Send + Sync + 'static,
{
    fn definition(&self) -> ToolDefinition {
        let schema = schema_for!(Args);
        ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: serde_json::to_value(&schema).unwrap_or_else(|_| serde_json::json!({})),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<ToolOutput<serde_json::Value>, BlazenError> {
        let typed_args: Args = serde_json::from_value(arguments).map_err(|e| {
            BlazenError::tool_error(format!("argument deserialization failed: {e}"))
        })?;
        let typed_output: ToolOutput<Output> = (self.handler)(typed_args).await?;
        let data = serde_json::to_value(&typed_output.data)
            .map_err(|e| BlazenError::tool_error(format!("output serialization failed: {e}")))?;
        Ok(ToolOutput {
            data,
            llm_override: typed_output.llm_override,
        })
    }
}

/// Convenience constructor for the common case: a typed handler that
/// returns `Result<Output, _>` (no LLM override). Wraps the result in
/// `ToolOutput::new(output)` so callers don't have to construct a
/// `ToolOutput` themselves.
pub fn typed_tool_simple<Args, Output, Fut, F>(
    name: impl Into<String>,
    description: impl Into<String>,
    handler: F,
) -> impl Tool
where
    Args: DeserializeOwned + JsonSchema + Send + 'static,
    Output: Serialize + Send + 'static,
    Fut: Future<Output = Result<Output, BlazenError>> + Send + 'static,
    F: Fn(Args) -> Fut + Send + Sync + 'static,
{
    TypedTool::new(name, description, move |args| {
        let fut = handler(args);
        Box::pin(async move {
            let output = fut.await?;
            Ok(ToolOutput::new(output))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Deserialize, JsonSchema)]
    struct AddArgs {
        a: i64,
        b: i64,
    }

    #[derive(Serialize)]
    struct AddOut {
        sum: i64,
    }

    #[tokio::test]
    async fn typed_tool_round_trip() {
        let tool = TypedTool::new("add", "Add two numbers", |args: AddArgs| {
            Box::pin(async move {
                Ok(ToolOutput::new(AddOut {
                    sum: args.a + args.b,
                }))
            })
        });
        let result = tool
            .execute(serde_json::json!({"a": 2, "b": 3}))
            .await
            .unwrap();
        assert_eq!(result.data, serde_json::json!({"sum": 5}));
        assert!(result.llm_override.is_none());
    }

    #[tokio::test]
    async fn typed_tool_simple_wraps_output() {
        let tool = typed_tool_simple("add", "Add two numbers", |args: AddArgs| async move {
            Ok::<_, BlazenError>(AddOut {
                sum: args.a + args.b,
            })
        });
        let result = tool
            .execute(serde_json::json!({"a": 10, "b": 20}))
            .await
            .unwrap();
        assert_eq!(result.data, serde_json::json!({"sum": 30}));
        assert!(result.llm_override.is_none());
    }

    #[tokio::test]
    async fn typed_tool_definition_contains_schema() {
        let tool = TypedTool::new("add", "Add two numbers", |args: AddArgs| {
            Box::pin(async move {
                Ok(ToolOutput::new(AddOut {
                    sum: args.a + args.b,
                }))
            })
        });
        let def = tool.definition();
        assert_eq!(def.name, "add");
        assert_eq!(def.description, "Add two numbers");
        // schemars 1.x emits the schema with `properties` listing the fields.
        let params = def.parameters;
        assert!(
            params.get("properties").is_some(),
            "schema missing `properties`: {params}"
        );
    }
}
