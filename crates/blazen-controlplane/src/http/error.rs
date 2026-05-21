//! Central error type for the REST surface.
//!
//! Every fallible route in [`super::openai_compat`] and
//! [`super::blazen_admin`] returns `Result<_, HttpError>`. The
//! [`IntoResponse`] impl on [`HttpError`] renders the failure as
//! OpenAI-style JSON:
//!
//! ```json
//! {"error": {"message": "...", "type": "invalid_request_error", "code": "bad_request"}}
//! ```
//!
//! `code` is the snake-case variant name and `type` is one of the
//! OpenAI-documented error-type strings. The HTTP status code is chosen
//! per-variant so well-behaved `OpenAI` clients (and `curl` + `jq` pipelines)
//! continue to work without modification.

use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde_json::json;

use crate::model_protocol::{
    RPC_ERR_INCOMPATIBLE, RPC_ERR_INTERNAL, RPC_ERR_INVALID, RPC_ERR_NOT_FOUND, RPC_ERR_QUOTA,
    RPC_ERR_TIMEOUT, RPC_ERR_UNSUPPORTED, RpcError,
};

/// All errors surfaceable by an OpenAI-compat or Blazen-admin route.
///
/// Each variant maps deterministically onto an HTTP status code and an
/// OpenAI-style `{type, code}` pair. Constructor helpers
/// ([`HttpError::bad_request`] etc.) keep call sites short.
#[derive(Debug)]
pub enum HttpError {
    /// Malformed JSON, missing required field, bad enum, ... — 400.
    BadRequest(String),
    /// JSON parsed but failed semantic validation (e.g. unknown model,
    /// dimensions out of range) — 422.
    Unprocessable(String),
    /// Bearer token missing / wrong — 401.
    Unauthorized(String),
    /// Authenticated but not permitted — 403.
    Forbidden(String),
    /// Resource not registered — 404.
    NotFound(String),
    /// Quota exceeded — 429.
    Quota(String),
    /// Backend reports it can't honor this verb — 501.
    Unsupported(String),
    /// Upstream operation timed out — 504.
    Timeout(String),
    /// Caller's protocol version is newer than the server understands —
    /// 412 (`Precondition Failed`), mirroring the postcard envelope
    /// behavior on the gRPC tier.
    Incompatible(String),
    /// Anything else — 500.
    Internal(String),
}

impl HttpError {
    /// Construct a [`HttpError::BadRequest`].
    #[must_use]
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::BadRequest(msg.into())
    }
    /// Construct a [`HttpError::Unprocessable`].
    #[must_use]
    pub fn unprocessable(msg: impl Into<String>) -> Self {
        Self::Unprocessable(msg.into())
    }
    /// Construct a [`HttpError::Unauthorized`].
    #[must_use]
    pub fn unauthorized(msg: impl Into<String>) -> Self {
        Self::Unauthorized(msg.into())
    }
    /// Construct a [`HttpError::Forbidden`].
    #[must_use]
    pub fn forbidden(msg: impl Into<String>) -> Self {
        Self::Forbidden(msg.into())
    }
    /// Construct a [`HttpError::NotFound`].
    #[must_use]
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }
    /// Construct a [`HttpError::Internal`].
    #[must_use]
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    fn parts(&self) -> (StatusCode, &'static str, &'static str, &str) {
        match self {
            Self::BadRequest(m) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "bad_request",
                m,
            ),
            Self::Unprocessable(m) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request_error",
                "unprocessable_entity",
                m,
            ),
            Self::Unauthorized(m) => (
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "unauthorized",
                m,
            ),
            Self::Forbidden(m) => (StatusCode::FORBIDDEN, "permission_error", "forbidden", m),
            Self::NotFound(m) => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "not_found",
                m,
            ),
            Self::Quota(m) => (
                StatusCode::TOO_MANY_REQUESTS,
                "quota_exceeded",
                "quota_exceeded",
                m,
            ),
            Self::Unsupported(m) => (
                StatusCode::NOT_IMPLEMENTED,
                "invalid_request_error",
                "unsupported",
                m,
            ),
            Self::Timeout(m) => (StatusCode::GATEWAY_TIMEOUT, "server_error", "timeout", m),
            Self::Incompatible(m) => (
                StatusCode::PRECONDITION_FAILED,
                "invalid_request_error",
                "incompatible_envelope_version",
                m,
            ),
            Self::Internal(m) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "internal_error",
                m,
            ),
        }
    }
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (_, _, code, m) = self.parts();
        write!(f, "{code}: {m}")
    }
}

impl std::error::Error for HttpError {}

impl IntoResponse for HttpError {
    fn into_response(self) -> axum::response::Response {
        let (status, ty, code, msg) = self.parts();
        let body = json!({
            "error": {
                "message": msg,
                "type": ty,
                "code": code,
            }
        });
        (status, Json(body)).into_response()
    }
}

impl From<RpcError> for HttpError {
    fn from(err: RpcError) -> Self {
        let RpcError { code, message, .. } = err;
        match code {
            c if c == RPC_ERR_INVALID => Self::BadRequest(message),
            c if c == RPC_ERR_NOT_FOUND => Self::NotFound(message),
            c if c == RPC_ERR_UNSUPPORTED => Self::Unsupported(message),
            c if c == RPC_ERR_TIMEOUT => Self::Timeout(message),
            c if c == RPC_ERR_QUOTA => Self::Quota(message),
            c if c == RPC_ERR_INCOMPATIBLE => Self::Incompatible(message),
            c if c == RPC_ERR_INTERNAL => Self::Internal(message),
            _ => Self::Internal(message),
        }
    }
}

impl From<serde_json::Error> for HttpError {
    fn from(err: serde_json::Error) -> Self {
        Self::BadRequest(format!("json: {err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    #[tokio::test]
    async fn renders_openai_shape() {
        let err = HttpError::BadRequest("missing model".into());
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "bad_request");
        assert_eq!(body["error"]["message"], "missing model");
    }

    #[tokio::test]
    async fn rpc_error_maps_to_status() {
        let rpc = RpcError::not_found("no such model 'qwen'");
        let http: HttpError = rpc.into();
        let resp = http.into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], "not_found");
    }

    #[tokio::test]
    async fn incompatible_envelope_maps_to_412() {
        let rpc = RpcError::incompatible("v9 > v1");
        let http: HttpError = rpc.into();
        assert!(matches!(http, HttpError::Incompatible(_)));
        let resp = http.into_response();
        assert_eq!(resp.status(), StatusCode::PRECONDITION_FAILED);
    }

    #[tokio::test]
    async fn unsupported_maps_to_501() {
        let rpc = RpcError::unsupported("no embeddings");
        let http: HttpError = rpc.into();
        let resp = http.into_response();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    }

    #[tokio::test]
    async fn quota_maps_to_429() {
        let rpc = RpcError::quota("budget");
        let http: HttpError = rpc.into();
        let resp = http.into_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn internal_default_for_unknown_code() {
        let rpc = RpcError {
            code: 9999,
            message: "weird".into(),
            retryable: false,
        };
        let http: HttpError = rpc.into();
        assert!(matches!(http, HttpError::Internal(_)));
    }
}
