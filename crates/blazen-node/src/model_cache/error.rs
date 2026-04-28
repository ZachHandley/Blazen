//! Error conversion for [`blazen_model_cache::CacheError`].

use napi::Status;

/// Convert a [`blazen_model_cache::CacheError`] into a [`napi::Error`].
///
/// Prefixes the readable message with the variant name so JS logs are clear.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn cache_error_to_napi(err: blazen_model_cache::CacheError) -> napi::Error {
    use blazen_model_cache::CacheError;

    let prefix = match &err {
        CacheError::Download(_) => "DownloadError",
        CacheError::CacheDir(_) => "CacheDirError",
        CacheError::Io(_) => "IoError",
        CacheError::Unsupported(_) => "UnsupportedError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}
